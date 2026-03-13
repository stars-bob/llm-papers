# SkillsBench 实验设计详情

## 1. 基准测试构建

### 1.1 数据来源
- **Skills 来源**: 47,150 个唯一 Skills
  - 开源仓库: 12,847 个
  - Claude Code 生态系统: 28,412 个
  - 企业合作伙伴: 5,891 个
- **任务提交**: 105 位贡献者提交 322 个候选任务
- **最终筛选**: 84 个任务，跨越 11 个领域

### 1.2 任务难度分层

| 难度级别 | 任务数 | 占比 | 预估人工完成时间 |
|----------|--------|------|------------------|
| Core (核心) | 17 | 19.8% | < 60 分钟 |
| Extended (扩展) | 43 | 50.0% | 1-4 小时 |
| Extreme (极端) | 26 | 30.2% | > 4 小时 |

### 1.3 领域分布

| 领域 | 任务数 |
|------|--------|
| Software Engineering | 16 |
| Office & White Collar | 14 |
| Natural Science | 12 |
| Media & Content Production | 11 |
| Finance | 8 |
| Energy | 8 |
| Cybersecurity | 5 |
| Robotics | 5 |
| Manufacturing | 3 |
| Mathematics | 2 |
| Healthcare | 2 |

---

## 2. 实验设置

### 2.1 Agent Harnesses（智能体框架）

| Harness | 厂商 | 版本 |
|---------|------|------|
| Claude Code | Anthropic | 2025 |
| Gemini CLI | Google | 2025 |
| Codex CLI | OpenAI | 2025 |

### 2.2 模型配置

| 模型 | 厂商 | 温度设置 |
|------|------|----------|
| GPT-5.2 | OpenAI | 0 |
| Claude Opus 4.5 | Anthropic | 0 |
| Claude Opus 4.6 | Anthropic | 0 |
| Claude Sonnet 4.5 | Anthropic | 0 |
| Claude Haiku 4.5 | Anthropic | 0 |
| Gemini 3 Pro | Google | 0 |
| Gemini 3 Flash | Google | 0 |

**配置矩阵**: 共 7 种 model-harness 组合

### 2.3 Skills 条件

每个任务在三种条件下评估：

1. **No Skills（无 Skills）**: 仅接收 instruction.md，环境无 Skills
2. **With Skills（有 Skills）**: 完整的 environment/skills/ 目录，包含所有示例、代码片段和资源
3. **Self-Generated Skills（自生成 Skills）**: 不提供 Skills，但提示模型在解决任务前生成相关程序性知识

**注意**: Gemini CLI 不支持自生成 Skills 条件。

---

## 3. 任务规格

### 3.1 任务结构
每个任务包含四个组件：

1. **Instruction**: 人工编写的任务描述，指定目标、输入格式和期望输出
2. **Environment**: Docker 容器，包含任务特定数据文件和 skills/ 子目录
3. **Solution**: 参考实现，验证任务可解决性
4. **Verifier**: 确定性测试脚本，使用程序化断言（包括数值容差）

### 3.2 Skills 定义

**Skills 满足四个标准**：
- **程序性内容**: 包含操作指南（流程、工作流、SOP），非事实检索
- **任务类别适用性**: 适用于一类问题，非单一实例
- **结构化组件**: 包含 SKILL.md 文件 + 可选资源（脚本、模板、示例）
- **可移植性**: 仅基于文件系统，易于编辑、版本控制、共享

**Skills 包含**：
- `SKILL.md`: 自然语言指令，指定如何接近一类任务
- **Resources**: 可执行脚本、代码模板、参考文档、工作示例

---

## 4. 评估协议

### 4.1 实验规模
- **总轨迹数**: 7,308 条有效轨迹
- **任务数**: 84 个
- **重复次数**: 每个任务 5 次试验
- **评分方法**: 对每任务 5 次试验取平均，再对 84 个任务取平均

### 4.2 评估指标

**主要指标**：
- **Pass Rate (通过率)**: 二元奖励的平均值

**归一化增益** (Hake's formulation):
```
g = (pass_skill - pass_vanilla) / (1 - pass_vanilla)
```

### 4.3 质量保障

**自动化验证**：
- 结构验证：必需文件存在性、目录布局、TOML/YAML 语法
- Oracle 执行：参考解决方案必须达到 100% 测试通过率
- 指令质量：人工编写（GPTZero 辅助检测，100% 人工标签）

**人工审核标准** (5 项)：
1. **数据有效性**: 输入数据反映真实世界复杂性
2. **任务真实性**: 场景反映真实专业工作流
3. **Oracle 质量**: 参考解决方案匹配领域专家解法
4. **Skill 质量**: Skills 无错误、内部一致、对类似任务真正有用
5. **反作弊**: 任务防止捷径解法

**防泄漏措施**：
- Skills 不得包含任务特定文件名、路径或标识符
- 不得包含解决基准任务的确切命令序列
- 不得包含任务规范中的常量、魔法数字
- 不得引用特定测试用例或期望输出

---

## 5. 实验结果详情

### 5.1 主要结果表格

| Harness | Model | 无 Skills | 有 Skills | g (%) | 自生成 Skills | g (%) |
|---------|-------|----------|----------|-------|--------------|-------|
| Gemini CLI | Gemini 3 Flash | 31.3% | 48.7% | 25.3% | — | — |
| Claude Code | Opus 4.5 | 22.0% | 45.3% | 29.9% | 21.6% | -0.5% |
| Codex | GPT-5.2 | 30.6% | 44.7% | 20.3% | 25.0% | -8.1% |
| Claude Code | Opus 4.6 | 30.6% | 44.5% | 20.0% | 32.0% | +2.0% |
| Gemini CLI | Gemini 3 Pro | 27.6% | 41.2% | 18.8% | — | — |
| Claude Code | Sonnet 4.5 | 17.3% | 31.8% | 17.5% | 15.2% | -2.5% |
| Claude Code | Haiku 4.5 | 11.0% | 27.7% | 18.8% | 11.0% | 0.0% |
| **Mean** | | **24.3%** | **40.6%** | **21.5%** | **21.0%** | **-1.8%** |

### 5.2 成本分析

**Token 消耗对比** (Gemini 3 Flash vs Pro)：
- Flash 每任务输入 token: 1.08M
- Pro 每任务输入 token: 0.47M
- Flash 是 Pro 的 2.3×

**每任务成本** (按官方 API 定价)：
- Gemini 3 Flash: $0.55
- Gemini 3 Pro: $0.98
- **Flash 便宜 44%**

补偿策略：小模型用迭代探索替代推理深度。

### 5.3 Skills 数量实验

| Skills 数量 | 有 Skills | 无 Skills | ∆abs |
|-------------|----------|----------|------|
| 1 skill | 42.2% | 24.4% | +17.8pp |
| 2-3 skills | 42.0% | 23.4% | **+18.6pp** |
| 4+ skills | 32.7% | 26.9% | +5.9pp |

### 5.4 Skills 复杂度实验

| 复杂度 | 通过率 | ∆abs | N |
|--------|--------|------|---|
| Detailed | 42.7% | +18.8pp | 1165 |
| Compact | 37.6% | +17.1pp | 845 |
| Standard | 37.1% | +10.1pp | 773 |
| Comprehensive | 39.9% | -2.9pp | 140 |

---

## 6. 关键发现总结

### 6.1 Harness 特性对比

| Harness | Skills 利用率 | 提升范围 | 特点 |
|---------|--------------|----------|------|
| Claude Code | 最高 | +13.9pp ~ +23.3pp | 原生 Skills 集成优化 |
| Gemini CLI | 高 | +13.6pp ~ +17.4pp | 原始性能最高 |
| Codex CLI | 中等 | +20.3pp | 经常忽视提供的 Skills |

### 6.2 失效模式分析

**自生成 Skills 失效模式**：
1. 模型识别需要领域特定知识，但生成不精确或不完整的程序
2. 高领域知识任务（制造、金融）中，模型常无法识别需要专业 Skills

**Skills 负面效果任务** (16/84)：
- Skills 可能引入冲突指导
- 对模型已能很好处理的任务增加不必要的复杂性

---

## 7. 实验环境

- **框架**: Harbor framework
- **容器**: Docker（确保可复现性、隔离依赖、干净文件系统状态）
- **验证**: pytest 确定性断言
- **日志**: 完整轨迹记录

---

## 8. 数据集发布

- **网站**: https://skillsbench.ai
- **GitHub**: 基准测试和数据集公开发布
- **许可证**: 开源（具体许可证见项目页面）
