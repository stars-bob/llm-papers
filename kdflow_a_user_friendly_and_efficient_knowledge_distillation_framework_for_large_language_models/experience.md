# KDFlow 实验设计详情

## 实验环境

### 硬件配置
- **服务器**: 单台服务器
- **GPU**: 8 × NVIDIA H20 GPUs

### 软件环境
- **训练后端**: PyTorch FSDP2
- **推理后端**: SGLang
- **分布式框架**: Ray
- **通信机制**: 共享内存 + Ray Shared Object（零拷贝传输）

---

## 模型配置

### 教师模型
| 模型 | 架构 | 参数规模 | 激活参数 |
|-----|------|---------|---------|
| Qwen3-14B | Dense | 14B | 14B |
| Qwen3-32B | Dense | 32B | 32B |
| Qwen3-30B-A3B | MoE | 30B | 3B |

### 学生模型
| 模型 | 架构 | 参数规模 |
|-----|------|---------|
| Qwen3-4B | Dense | 4B |
| Qwen3-1.7B | Dense | 1.7B |

### 模型家族特点
选用Qwen3系列的原因：
- 覆盖多种模型尺寸
- 包含Dense和MoE两种架构
- 便于进行系统性的教师-学生组合实验

---

## 数据集

### 训练数据
- **来源**: LMSys-Chat-1M
- **采样规模**: 100k prompts（随机采样）
- **响应生成**: 使用 Qwen3-14B 生成

### 评测数据
- **基准**: AlpacaEval 2.0
- **指标**: 
  - LC-Win Rate (Length-Controlled Win Rate)
  - Win Rate

---

## 训练配置

### 通用超参数
| 参数 | 数值 |
|-----|------|
| Global Batch Size | 128 |
| Gradient Accumulation | 8 |
| Max Sequence Length | 4096 |
| 优化器 | AdamW |

### 蒸馏算法
- **散度度量**: Forward KL (FKL) Divergence（主要实验）
- 支持的其他度量：Reverse KL (RKL)、Jensen-Shannon Divergence (JSD)、Total Variation Distance (TVD)

### 通信配置
- **教师输出**: 隐状态 (hidden states)
- **传输方式**: 零拷贝共享内存
- **学生侧处理**: 使用教师的语言模型头 (LM head) 重新计算logits

### 精度配置
- **KDFlow (BF16 Teacher)**: 教师使用BF16精度推理
- **KDFlow (FP8 Teacher)**: 教师使用FP8精度推理（更高效）

---

## 基线框架

| 框架 | 训练后端 | 架构特点 |
|-----|---------|---------|
| TRL | DeepSpeed ZeRO-3 | 统一后端 |
| MS-SWIFT | DeepSpeed ZeRO-3 | 统一后端 |
| ROLL | FSDP2 | 统一后端 |
| KDFlow | FSDP2 + SGLang | 解耦后端 |

---

## 关键实验结果

### 1. 训练效率对比（秒/迭代）

#### 学生: Qwen3-4B

| 框架 | 后端 | Qwen3-14B | Qwen3-32B | Qwen3-30B-A3B |
|-----|------|-----------|-----------|---------------|
| TRL | ZeRO-3 | 21.3 | 31.5 | - |
| MS-SWIFT | ZeRO-3 | 16.6 | 24.8 | 43.2 |
| ROLL | FSDP2 | 38.4 | 56.9 | 67.9 |
| KDFlow (BF16) | FSDP2 | 12.3 | 15.7 | 11.3 |
| KDFlow (FP8) | FSDP2 | 11.5 | 13.5 | 11.1 |

#### 学生: Qwen3-1.7B

| 框架 | 后端 | Qwen3-14B | Qwen3-32B | Qwen3-30B-A3B |
|-----|------|-----------|-----------|---------------|
| TRL | ZeRO-3 | 13.3 | 23.4 | - |
| MS-SWIFT | ZeRO-3 | 11.5 | 20.1 | 36.9 |
| ROLL | FSDP2 | 26.8 | 45.6 | 53.8 |
| KDFlow (BF16) | FSDP2 | 7.6 | 10.9 | 5.9 |
| KDFlow (FP8) | FSDP2 | 6.7 | 8.7 | 5.8 |

### 2. 关键发现

#### MoE模型优化效果显著
- 对于Qwen3-30B-A3B（MoE）教师：
  - 基线框架训练时间：36.9s/it ~ 67.9s/it
  - KDFlow训练时间：5.8s/it ~ 11.3s/it
  - **加速比高达6.36×**

- 原因分析：
  - 标准训练引擎（ZeRO-3/FSDP2）对MoE的稀疏路由和专家管理支持不佳
  - SGLang针对MoE架构有高度优化的kernel和灵活的并行策略

#### 通信开销控制
- 直接传输完整logits数据量：~160GB（128 seq × 4096 len × 151936 vocab × 2 bytes）
- 传输隐状态数据量：显著降低（hidden dim ≈ 4096 vs vocab size ≈ 151936）
- 通过零拷贝传输，避免了内存带宽成为瓶颈

### 3. 性能验证

#### 损失曲线等价性
- KDFlow的损失曲线与纯FSDP基线几乎完全重合
- FP8和BF16教师推理的损失曲线高度一致
- 证明SGLang推理不会引入显著的数值不稳定性

#### 下游任务性能（AlpacaEval 2.0）
蒸馏设置：Qwen3-30B-A3B → Qwen3-1.7B，使用FKL

| 框架 | LC-Win Rate | Win Rate |
|-----|-------------|----------|
| 无蒸馏基线 | 26.09% | 21.99% |
| MS-SWIFT | 28.40% | 27.86% |
| FSDP基线 | 28.18% | 28.20% |
| KDFlow | 28.23% | 28.32% |

**结论**：KDFlow在显著提升速度的同时，保持了与基线相当的蒸馏质量。

---

## 架构设计要点

### 1. 解耦架构优势
- 教师使用SGLang：获得极致推理吞吐量
- 学生使用FSDP2：获得灵活的梯度更新和优化器状态管理
- 通过Ray实现高效的分布式进程管理

### 2. 隐状态传输设计
```
Teacher (SGLang) → hidden states → Student (FSDP2) → recompute logits → compute loss
```
- 避免传输庞大的logit张量
- 在学生本地重新计算保持数学等价性
- 使用共享内存实现零拷贝跨进程通信

### 3. 支持的蒸馏范式
- **离线蒸馏**：Trainer → TeacherActorGroup → StudentActorGroup
- **在线蒸馏**：Trainer → RolloutActorGroup → TeacherActorGroup → StudentActorGroup → RolloutActorGroup（权重同步）

---

## 经验教训

1. **架构适配性**：教师和学生模型的不同角色需要不同的优化后端，强行统一会导致效率损失

2. **通信优化关键性**：在分布式训练中，通信开销往往是瓶颈，通过传输紧凑表示（隐状态）而非完整输出可以有效缓解

3. **MoE架构的特殊性**：MoE模型对推理引擎有特殊要求，使用专门的推理引擎（如SGLang）比通用训练引擎效果更好

4. **数值稳定性**：即使使用FP8低精度推理，只要设计得当，仍能保持训练稳定性和最终性能

5. **用户友好性**：通过抽象复杂的分布式通信逻辑，可以让研究者专注于算法本身而非工程实现
