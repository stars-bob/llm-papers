# Attention Residuals 实验设计详情

## 1. 模型架构

### 基础架构
实验基于 **Kimi Linear** 架构，这是一个MoE（混合专家）Transformer，遵循 Moonlight / DeepSeek-V3 设计：

| 组件 | 配置 |
|------|------|
| 注意力机制 | Kimi Delta Attention (KDA) 和 Multi-Head Latent Attention (MLA) 3:1交错 |
| 前馈网络 | MoE层（每个注意力层后接一个MoE FFN） |
| 归一化 | PreNorm (RMSNorm) |

### 最大模型配置（48B参数模型）

| 参数 | 数值 |
|------|------|
| Transformer块数 | 27 (54层) |
| 路由专家数 | 256 |
| 激活专家数 | 8 + 1共享专家 |
| 总参数量 | 48B |
| 激活参数量 | 3B |
| Block AttnRes配置 | 6层/块，共9个块 + token嵌入 = 10个深度源 |

### AttnRes特定参数

每层引入：
- 1个RMSNorm
- 1个伪查询向量 `wl ∈ R^d`

**初始化关键**：所有伪查询向量必须初始化为零，确保训练开始时注意力权重均匀分布（等价于等权重平均），防止训练不稳定。

---

## 2. 缩放定律实验

### 模型规模

| 激活参数量 | Lb (块数) | H (头数) | dmodel | dff | 学习率 | Batch Size | 训练Tokens |
|-----------|----------|---------|--------|-----|--------|-----------|-----------|
| 194M | 12 | 12 | 896 | 400 | 2.99e-3 | 192 | 38.7B |
| 241M | 13 | 13 | 960 | 432 | 2.80e-3 | 256 | 45.4B |
| 296M | 14 | 14 | 1024 | 464 | 2.50e-3 | 320 | 62.1B |
| 436M | 16 | 16 | 1168 | 528 | 2.20e-3 | 384 | 87.9B |
| 528M | 17 | 17 | 1264 | 560 | 2.02e-3 | 432 | 119.0B |

*注：Lb = L/2 表示Transformer块数（每个块包含Attention+MLP）*

### 训练设置

| 配置项 | 设置 |
|--------|------|
| 上下文长度 | 8192 tokens |
| 学习率调度 | Cosine |
| 对比变体 | Baseline, Full AttnRes, Block AttnRes (N≈8) |
| 缩放定律拟合 | L = A × C^(-α)，其中C为PFLOP/s-days |

**公平性保证**：每个规模组内所有变体共享相同的超参数（在Baseline下选择），故意偏向Baseline使比较更保守。

---

## 3. 主实验：48B参数模型预训练

### 训练阶段

#### Stage 1: WSD预训练
- **数据**: 1T tokens
- **上下文长度**: 4096 tokens
- **优化器**: Muon
- **学习率调度**: WSD (Warmup–Stable–Decay)
- **全局Batch Size**: 8M tokens

#### Stage 2: 中程训练（Mid-training）
- **数据**: ~400B高质量tokens
- **方法**: 退火（annealing）recipe，遵循Moonlight

#### Stage 3: 长上下文扩展
- **序列长度**: 逐步扩展到32K tokens
- **位置编码**: 无需YaRN或温度重缩放（MLA使用NoPE - No Positional Encoding）

---

## 4. 消融实验设置

### 测试模型
16层模型（来自缩放定律中的436M激活参数配置）

### 消融变体

| 变体 | 描述 |
|------|------|
| DenseFormer | 固定、输入无关的标量系数聚合所有先前输出 |
| mHC | m=4并行流，学习混合矩阵 |
| Full AttnRes | 完整注意力残差 |
| w/ input-dependent query | 从隐藏状态投影查询（而非独立学习） |
| w/ input-independent mixing | 移除Q/K，使用可学习标量权重 |
| w/ sigmoid | 用sigmoid替代softmax |
| w/o RMSNorm | 移除keys上的RMSNorm |
| SWA | 滑动窗口聚合（W=8层+嵌入） |
| Block (S=4) | 块大小为4的Block AttnRes |
| w/ multihead (H=16) | 每头独立深度聚合 |

### 块大小扫描
从S=1（Full AttnRes）到S=32的系统扫描，评估验证loss。

---

## 5. 架构偏好分析实验

### 实验设计
- **固定计算量**: ~6.5 × 10^19 FLOPs
- **固定激活参数**: ~2.3 × 10^8
- **MLP扩展比**: dff/dmodel ≈ 0.45（基于内部经验观察）
- **扫描网格**: 5×5配置
  - dmodel/Lb ∈ {15, 30, 45, 60, 75}
  - H/Lb ∈ {0.3, 0.4, 0.5, 0.6, 0.7}

*目标：评估AttnRes是否改变最优的深度-宽度-注意力权衡*

---

## 6. 下游评估基准

### 英文理解推理
| 基准 | 描述 |
|------|------|
| MMLU | 多学科多任务语言理解 |
| MMLU-Pro Hard | 更鲁棒的多任务基准 |
| GPQA-Diamond | 研究生级别问答 |
| BBH | 大基准困难任务 |
| ARC-Challenge | 科学问答挑战 |
| HellaSwag | 常识推理 |
| TriviaQA | 阅读理解 |

### 数学与代码
| 基准 | 描述 |
|------|------|
| GSM8K | 数学文字问题 |
| MGSM | 多语言数学推理 |
| Math (Minerva) | 定量推理 |
| CMath | 中文数学 |
| HumanEval | 代码生成 |
| MBPP | 编程问题 |

### 中文理解
| 基准 | 描述 |
|------|------|
| CMMLU | 中文多学科理解 |
| C-Eval | 中文综合评估套件 |

---

## 7. 系统优化详情

### 7.1 训练优化

#### 流水线并行通信
- **配置**: P物理阶段，V虚拟阶段/物理阶段
- **Baseline**: 传输固定大小隐藏状态
- **Block AttnRes**: 需要传输积累的块表示

#### 跨阶段缓存效果
| 指标 | Naïve | Cached | 改进 |
|------|-------|--------|------|
| 每次传输峰值开销 | O(C) | O(P) | V× |
| 端到端训练开销 | - | - | <4% |

*C = PV = 总chunks数*

### 7.2 推理优化

#### 两阶段计算成本（每层）

| 操作 | Read | Write | 总计 |
|------|------|-------|------|
| Standard Residuals | 2d | d | 3d |
| mHC (m=4) | - | - | 34d |
| **Block AttnRes Phase 1** (均摊) | (N/S)d | d | (N/S+1)d |
| **Block AttnRes Phase 2** | 3d | d | 4d |
| **Block AttnRes 总计** | - | - | **(N/S+5)d ≈ 5.5d** |

*典型值：L=128, N=8, S=16*

#### 长上下文预填充内存
| 配置 | 内存占用 |
|------|---------|
| 无分片，128K上下文，8块 | 15 GB |
| 8路TP分片 | ~1.9 GB/设备 |
| 16K分块预填充 | <0.3 GB/设备 |

#### 推理延迟开销
- **典型工作负载**: <2%

---

## 8. 关键实现细节

### Block AttnRes伪代码要点

```python
def block_attn_res(blocks, partial_block, proj, norm):
    """
    块间注意力：对块表示+部分和进行注意力计算
    blocks: N个 [B, T, D] 张量 - 前面已完成块的表示
    partial_block: [B, T, D] - 块内部分和
    """
    V = torch.stack(blocks + [partial_block])  # [N+1, B, T, D]
    K = norm(V)
    logits = torch.einsum('d, n b t d -> n b t', proj.weight.squeeze(), K)
    h = torch.einsum('n b t, n b t d -> b t d', logits.softmax(0), V)
    return h
```

### 在线Softmax合并
使用经典在线softmax算法[31]合并Phase 1（块间）和Phase 2（块内）结果，保持精确等价。

---

## 9. 实验关键发现总结

| 发现 | 证据 |
|------|------|
| AttnRes解决PreNorm稀释 | 输出幅度有界、梯度分布均匀 |
| 1.25×计算等效性 | Block AttnRes 1.692 ≈ Baseline@1.25×计算 |
| 块大小N≈8是甜点 | 恢复大部分Full AttnRes收益 |
| 深度利用效率提升 | 最优架构向更深更窄偏移 |
| 推理开销极小 | <2%延迟开销 |
| 多步推理收益最大 | GPQA+7.5, Math+3.6, HumanEval+3.1 |
