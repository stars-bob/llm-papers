# SkillsBench: Benchmarking How Well Agent Skills Work Across Diverse Tasks

## 论文元数据

| 项目 | 内容 |
|------|------|
| **标题** | SkillsBench: Benchmarking How Well Agent Skills Work Across Diverse Tasks |
| **来源** | arXiv:2602.12670 [cs.AI] |
| **发布时间** | 2026年2月13日 (v1), 2026年3月7日 (v2) |
| **作者** | Xiangyi Li, Wenbo Chen, Yimin Liu 等 40+ 位作者 |
| **机构** | BenchFlow, Ohio State University, Amazon, Dartmouth College, Stanford, UC Berkeley 等 |
| **页数** | 34 页 |
| **数据集** | https://skillsbench.ai |

---

## 研究背景

大型语言模型（LLM）已从文本生成器发展为能在真实环境中执行复杂多步任务的自主智能体。Claude Code、Gemini CLI、Codex CLI 等工具使开发者能够将前沿模型作为终端环境中的智能助手。然而，基础模型虽然具备广泛能力，却缺乏特定领域工作流所需的程序性知识，而微调成本高昂且会牺牲泛化能力。

**Agent Skills** 是一种新兴的解决方案：Skill 是一个结构化包，包含指令、代码模板、资源和验证逻辑，可在推理时增强智能体行为而无需修改模型。尽管 Skills 生态系统快速发展（社区仓库已有数千个用户贡献的 Skills），但此前没有基准测试系统地评估 Skills 如何以及何时能提升智能体性能。

---

## 核心贡献

### 1. Skills-centric 评估框架
- 精心策划 **84 个任务**，跨越 **11 个领域**
- 每个任务在 **三种条件** 下执行：无 Skills、人工精选 Skills、自生成 Skills
- 使用确定性验证器和完整轨迹日志

### 2. 大规模实证评估
- 评估 **7 种 agent-model 配置**
- 生成 **7,308 条有效轨迹**
- 首次系统性地提供 Skills 效果、方差和失效模式的证据

---

## 关键实验结果

### 主要发现 1：Skills 提供显著但可变的收益
- **精选 Skills** 平均提升通过率 **+16.2 个百分点 (pp)**
- 不同配置间差异显著（范围：+13.6pp 到 +23.3pp）
- **自生成 Skills** 平均效果为 **-1.8pp**（几乎无收益甚至负面）

| 配置 | 无 Skills | 有 Skills | 提升 |
|------|----------|----------|------|
| Gemini 3 Flash (Gemini CLI) | 31.3% | 48.7% | +17.4pp |
| Claude Opus 4.5 (Claude Code) | 22.0% | 45.3% | +23.3pp |
| GPT-5.2 (Codex CLI) | 30.6% | 44.7% | +20.3pp |
| Claude Opus 4.6 (Claude Code) | 30.6% | 44.5% | +13.9pp |
| Gemini 3 Pro (Gemini CLI) | 27.6% | 41.2% | +13.6pp |
| Claude Sonnet 4.5 (Claude Code) | 17.3% | 31.8% | +14.5pp |
| Claude Haiku 4.5 (Claude Code) | 11.0% | 27.7% | +16.7pp |

### 主要发现 2：领域差异巨大
Skills 效果在不同领域间差异显著：

| 领域 | 无 Skills | 有 Skills | 提升 |
|------|----------|----------|------|
| Healthcare | 34.2% | 86.1% | **+51.9pp** |
| Manufacturing | 1.0% | 42.9% | **+41.9pp** |
| Cybersecurity | 20.8% | 44.0% | +23.2pp |
| Natural Science | 23.1% | 44.9% | +21.9pp |
| Energy | 29.5% | 47.5% | +17.9pp |
| Office & White Collar | 24.7% | 42.5% | +17.8pp |
| Finance | 12.5% | 27.6% | +15.1pp |
| Media & Content Production | 23.8% | 37.6% | +13.9pp |
| Robotics | 20.0% | 27.0% | +7.0pp |
| Mathematics | 41.3% | 47.3% | +6.0pp |
| Software Engineering | 34.4% | 38.9% | **+4.5pp** |

**洞察**：模型预训练中代表性不足的领域（如临床数据协调、制造工作流）收益最大；而预训练覆盖强的领域（如软件工程、数学）收益较小。

### 主要发现 3：Skills 设计因素

**Skills 数量**：
- 1 个 Skill: +17.8pp
- **2-3 个 Skills: +18.6pp** (最优)
- 4+ 个 Skills: +5.9pp (收益递减)

**Skills 复杂度**：
- Detailed: +18.8pp
- Compact: +17.1pp
- Standard: +10.1pp
- Comprehensive: -2.9pp (负面效果)

**结论**：适中的 Skills 优于详尽文档；过度内容会造成认知负担。

### 主要发现 4：Skills 可以弥补模型规模差距
- Claude Haiku 4.5 + Skills (27.7%) > Claude Opus 4.5 无 Skills (22.0%)
- **小模型 + Skills 可以超越大模型无 Skills 的表现**

---

## 重要洞察

### 1. 模型无法自生成有效 Skills
自生成 Skills 平均效果为 -1.8pp，表明：
- 模型**无法可靠地编写**它们能从消费中受益的程序性知识
- 有效 Skills 需要**人工策划的领域专业知识**

### 2. Skills 失效的任务
84 个任务中有 16 个显示负面 Skills 效果：
- taxonomy-tree-merge (–39.3pp)
- energy-ac-optimal-power-flow (–14.3pp)
- trend-anomaly-causal-inference (–12.9pp)

失效模式：Skills 可能引入冲突指导或不必要的复杂性。

### 3. Harness 实现影响 Skills 效果
- **Claude Code**: Skills 利用率最高 (+13.9pp 到 +23.3pp)
- **Gemini CLI**: 原始性能最高 (Gemini 3 Flash 达 48.7%)
- **Codex CLI**: 经常忽视提供的 Skills

---

## 结论

SkillsBench 是首个将 Skills 视为一等评估工件的基准测试，证明：

1. **精选 Skills 显著提升性能** (+16.2pp 平均)
2. **自生成 Skills 无效** (-1.8pp 平均)
3. **2-3 个聚焦的 Skills 最优**，过多会适得其反
4. **领域差异巨大** (Healthcare +51.9pp vs SE +4.5pp)
5. **Skills 可以部分补偿模型规模限制**

对开发者的启示：有效的 Skills 需要简洁、分步的指导，包含至少一个工作示例，而非详尽文档。

---

## 引用

```bibtex
@article{li2026skillsbench,
  title={SkillsBench: Benchmarking How Well Agent Skills Work Across Diverse Tasks},
  author={Li, Xiangyi and Chen, Wenbo and Liu, Yimin and Zheng, Shenghan and others},
  journal={arXiv preprint arXiv:2602.12670},
  year={2026}
}
```
