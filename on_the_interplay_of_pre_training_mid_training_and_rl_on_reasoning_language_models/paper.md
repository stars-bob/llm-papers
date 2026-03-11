# On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models
Source: https://arxiv.org/abs/2512.07783

--- Page 1 ---
On the Interplay of Pre-Training, Mid-Training, and RL
on Reasoning Language Models
Charlie Zhang*
Graham Neubig
Xiang Yue†
Carnegie Mellon University, Language Technologies Institute
Interplay-LM-Reasoning
Interplay-LM-Reasoning
{chariezhang0106,xiangyue.work}@gmail.com
gneubig@cs.cmu.edu
Abstract
Recent reinforcement learning (RL) techniques have yielded impressive reasoning improvements in lan-
guage models, yet it remains unclear whether post-training truly extends a model’s reasoning ability beyond
what it acquires during pre-training. A central challenge is the lack of control in modern training pipelines:
large-scale pre-training corpora are opaque, mid-training is often underexamined, and RL objectives interact
with unknown prior knowledge in complex ways. To resolve this ambiguity, we develop a fully controlled
experimental framework that isolates the causal contributions of pre-training, mid-training, and RL-based
post-training. Our approach employs synthetic reasoning tasks with explicit atomic operations, parseable
step-by-step reasoning traces, and systematic manipulation of training distributions. We evaluate models
along two axes: extrapolative generalization to more complex compositions and contextual generalization
across surface contexts. Using this framework, we reconcile competing views on RL’s effectiveness. We
show that: 1) RL produces true capability gains (pass@128) only when pre-training leaves sufficient
headroom and when RL data target the model’s edge of competence, tasks at the boundary that are difficult
but not yet out of reach. 2) Contextual generalization requires minimal yet sufficient pre-training exposure,
after which RL can reliably transfer. 3) Mid-training significantly enhances performance under fixed
compute compared with RL only, demonstrating its central but underexplored role in training pipelines.
4) Process-level rewards reduce reward hacking and improve reasoning fidelity. Together, these results
clarify the interplay between pre-training, mid-training, and RL, offering a foundation for understanding
and improving reasoning LM training strategies.
different long-tail exposure 
ratio during pre-training
different RL data recipe
under the same 
training budget
Figure 1: Interplay of pre-, mid-, and post-training in LM reasoning. Left: RL yields genuine extrapolative gains
only when task difficulty slightly exceeds the pre-training range; gains vanish when tasks are already covered or too
out-of-distribution (up to +42% pass@128 when well-calibrated). Mid: Contextual generalization requires minimal
yet sufficient pre-training exposure to long-tail contexts. RL fails with near-zero exposure but generalizes robustly
with sparse exposure (≥1%), yielding up to +60% pass@128. Right: A mid-training stage bridging pre-training
and RL substantially improves OOD reasoning under fixed compute, with mid-training + RL outperforming RL
alone by +10.8% on OOD-hard tasks.
1
Introduction
Recent advances in reinforcement learning (RL) have led to significant improvements in the reasoning capabilities
of language models (LMs) [DeepSeek-AI et al., 2025, OpenAI et al., 2024]. Yet despite this progress, a fundamental
*Work done when interning at CMU.
†Corresponding Author.
1
arXiv:2512.07783v1  [cs.CL]  8 Dec 2025

--- Page 2 ---
conceptual question remains unresolved: does post-training truly extend a model’s reasoning ability beyond what is
acquired during pre-training? The literature offers conflicting views: some work characterizes RL as a capability
refiner [Yue et al., 2025, Wu et al., 2025, Shao et al., 2025, Yeo et al., 2025], while others present evidence of
substantial reasoning gains beyond pre-training [Wen et al., 2025, Yuan et al., 2025, Sun et al., 2025a].
A major source of this discrepancy is that prior analyses rely on uncontrolled training environments. Modern
LMs are pre-trained on massive, opaque internet corpora whose composition is fundamentally unknown. As a result,
we cannot ascertain which reasoning primitives the base model has already internalized. Consequently, this lack of
control makes it challenging to isolate the causal effect of post-training and to understand how pre-training and
post-training jointly shape reasoning behavior.
Meanwhile, an additional stage, mid-training,1 has recently emerged as a key component of modern LM
pipelines [Wang et al., 2025, Liu et al., 2025a]. Mid-training acts as an intermediate distributional bridge between
broad pre-training corpora and specialized post-training objectives, expanding the model’s primitive coverage and
aligning its internal representations with the tasks emphasized during RL. As a result, mid-training has become
increasingly central to the debate: it may explain why RL sometimes produces striking generalization improvements,
yet fails in other settings [Wang et al., 2025]. This motivates the core question of our work: What is the interplay
between pre-training, mid-training, and RL in shaping the reasoning capabilities of LMs?
The goal of this work is to convincingly answer this question in a controlled manner, following previous work in
this vein [Allen-Zhu, 2024, Ye et al., 2024, Zhou et al., 2025a]. Specifically, we perform controlled experiments to
disentangle how pre-training, mid-training, and RL-based post-training individually and jointly influence reasoning
generalization.
To this end, we build a fully controlled framework that isolates the contributions of each training stage. Our
design is based on three principles: (i) fully controllable synthetic reasoning tasks with explicit atomic operations
and DAG-defined dependency structure; (ii) observable, parseable reasoning processes enabling process-level
evaluation and reducing reward or evaluation hacking; and (iii) systematic manipulation of pre-/mid-/post-training
distributions to attribute causal effects to each stage.
We evaluate reasoning along two key dimensions: 1) Extrapolative (Depth) generalization assesses whether
models can solve problems more complex than those encountered during pre-training by composing learned
primitives in deeper structures. 2) Contextual (Breadth) generalization evaluates whether models can transfer
reasoning skills across novel surface contexts that share equivalent underlying logic. Together, these axes capture a
broad spectrum of compositional and transfer reasoning abilities relevant to real-world LMs. Using our controlled
framework, we uncover several insights into how the three training stages interact.
Firstly, the two competing views on whether RL genuinely improves a base model’s reasoning ability do not
truly conflict. RL produces true capability gains only when two conditions hold: (i) the task was not heavily covered
during pre-training, leaving sufficient headroom for RL to explore. (ii) the RL data are calibrated to the model’s
edge of competence, neither too easy (in-domain) nor too hard (out-of-domain). When either condition is violated,
RL tends to sharpen existing abilities rather than genuinely improve.
Secondly, RL incentivizes contextual generalization only when the relevant primitives or base skills are present
in the base model. Without minimal pre-training exposure to a new context, RL does not induce transfer. But even
very sparse coverage (e.g., ≥1%) provides a sufficient seed that RL can then robustly reinforce, yielding strong
cross-context generalization.
Thirdly, introducing a mid-training phase that bridges pre- and post-training distributions substantially strength-
ens both in-domain and out-of-domain performance under a fixed compute budget, highlighting mid-training as an
underexplored but powerful lever in training design.
Fourthly, process rewards mitigate reward hacking and enhance reasoning fidelity. Incorporating process verifi-
cation into the reward function aligns reinforcement signals with valid reasoning behavior, leading to measurable
improvements in both accuracy and generalization under complex, compositional settings.
2
Preliminaries
In this section, we introduce a) the synthetic data generation framework grounded in dependency graphs and contex-
tual rendering that specify the reasoning process, (b) the task setup for extrapolative and contextual generalization,
and (c) the process-verified evaluation framework, which assesses the accuracy of both the reasoning process and
the final answer. Together, these components allow us to isolate the distinct effects of pre-training, mid-training, and
post-training on reasoning generalization.
2.1
Controllable Synthetic Reasoning Dataset
We build on the GSM-Infinite [Zhou et al., 2025a] data generation framework to create a testbed with precise control
over reasoning structure, complexity, and context. Specifically, the data generation pipeline (Figure 2 (a)) involves
1In some literature, this stage is called continued pre-training (CPT).

--- Page 3 ---
…
o
u
f
K
…
dependency graph 𝓖
contextual templates 𝝉
controls reasoning complexity 
& explicit dependencies
controls task contexts & 
implicit dependencies
𝜏𝑧oo
Location: [South zoo, Cedar 
Valley, …]
Animal: [lion, elephant, …]
𝜏teacher
rendering Φ 𝒢, 𝜏
f: “elephants in Cedar Valley”
o: “lions in South zoo”
...
(a) Data Generation Framework
(b) Task Setup
Pre-training
Task Difficulty
Post-Training
Extrapolation (OOD)
Extrapolative Generalization
Contextual Generalization
(c) Process-Verified Evaluation
Generated Solution
Pre-training: 
𝜏𝑧oo: full primitives
Task Difficulty
[Problem] 
The number of lions in South zoo equals the number of elephants in 
Cedar Valley plus… .What is the total number of Animals in South zoo.
[Solution] 
Step 1. Define elephants in Cedar Valley as f; so f = 2. 
…
Step N. Define total number of Animals in South zoo as K; u = o + f = 6 
+ 2 = 8; so K = u - o = 8 - 6 = 2.
[Answer]
4
Post-Training: α 𝜏𝑧oo + (1 − α) 𝜏teacher
RQ1: Which Pre-/Post-Training Interplay enables reasoning 
extrapolation to unseen OOD?
RQ2: Which Pre-/Post−Training Interplay enables reasoning 
primitives transfer from 𝜏𝑧oo to 𝜏teacher?
[Solution] 
Step 1. Define elephants in Cedar Valley as f; so f = 2. 
…
Step N. Define total number of Animals in South zoo 
as K; u = o + f = 6 + 2 = 8; so K = u - o = 8 - 6 = 2.
[Answer]
4
Solution parser
…
+
𝑛𝑜𝑑𝑒𝑗
𝑛𝑜𝑑𝑒𝑗+1
𝑛𝑜𝑑𝑒𝑗+2
𝑛𝑜𝑑𝑒𝑁
predicted graph ෡𝓖
…
—
𝑛𝑜𝑑𝑒𝑗
𝑛𝑜𝑑𝑒𝑗+1
𝑛𝑜𝑑𝑒𝑗+2
𝑛𝑜𝑑𝑒𝑁
gold graph 𝓖
Incorrect operation
𝑪𝒐𝒓𝒓𝒆𝒄𝒕𝒏𝒆𝒔𝒔= 𝑷𝒓𝒐𝒄𝒆𝒔𝒔𝑨𝒄𝒄෡𝓖; 𝓖== 1 and ෝ𝒂== 𝒂∗
Incorrect dependency
…
ොa
𝑎∗
𝜏teacher: basic primitives
Contextual Transfer
Figure 2: Overview of the data generation framework, task setup, and process-verified evaluation. The
figure depicts the dependency graph G and contextual templates τ, the task setup for extrapolative and contextual
generalization, and the process-verified evaluation framework that checks for correctness of reasoning steps.
three key components:
Dependency Graphs. Each reasoning problem is represented by a direct G = (V, E), where nodes v ∈V
correspond to variables, and directed edges e ∈E denote dependencies between them. The graph culminates in a
designated answer node v∗, which yields the final answer a∗.
Reasoning Complexity Control. We quantify the complexity of a graph by the number of arithmetic operations:
op(G) = |E|,
which controls task difficulty from basic arithmetic to complex multi-step reasoning.
Contextual Rendering. Given a pre-defined contextual template τ (e.g., animals–zoo, teachers–school) with
natural language descriptions, we render the dependency graph G to produce a complete math problem. Finally, we
generate diverse math problems by sampling different graphs G and templates τ, and rendering them into text.
Our motivation for using this framework lies in three main advantages: 1) Contamination-free control over
training phases. We specify separate data distributions for pre-, mid-, and post-training to avoid overlap. 2)
Factorized control over structure and context. Each problem is generated from a DAG, encoding the reasoning
structure and dependencies, with numeric values and context instantiated on top. 3) Process-level verification. The
ground-truth DAG serves as a reference for verifying intermediate steps and preventing incorrect reasoning. We
provide a detailed formulation and explanation in Appendix A.1.
2.2
Task Setup
In real-world deployments, language models usually need to generalize reasoning along two complementary axes:
extrapolative (depth-wise) and contextual (breadth-wise) generalization [Setlur et al., 2025, Zhou et al., 2025b,
Huan et al., 2025]. Our controlled experiments expose these two dimensions (Figure 2(b)), enabling a precise
examination of how pre-training, mid-training, and post-training influence each type of generalization.
Extrapolative (Depth) Generalization. This dimension evaluates a model’s ability to maintain correctness as
reasoning depth op(G) increases [Zhang et al., 2025]. A model exhibits strong extrapolative generalization if it can
solve problems whose operation chains exceed those encountered during training.
Contextual (Breadth) Generalization. This dimension measures whether a model can transfer its reasoning
primitives to novel domains that differ in surface forms but share similar underlying reasoning structure. A model
generalizes contextually when its performance remains stable under changes in templates or surface forms while the
underlying computation graph remains the same.
Formal notation, dataset construction, and full definitions of the generalization axes are provided in Appendix A.2.
2.3
Evaluation Protocol.
We report all results under a process-verified evaluation scheme (Figure 2 (c)). For each instance with ground-truth
dependency graph (G, a∗), the model produces a free-form solution, which we parse into a predicted dependency
graph ˆG and final answer ˆa. The process is evaluated at the step level for each gold node v ∈V by comparing the
predicted and ground-truth nodes, their dependencies, and their numeric values. The process accuracy is computed
as the average step-level accuracy across all gold nodes. A prediction is considered fully correct only when both the
reasoning steps and the final answer match. All pass@k metrics (e.g., pass@1, pass@128) are reported with
respect to this strict criterion. Detailed implementation and parsing methods are provided in Appendix A.4.

--- Page 4 ---
2.4
Training Setup.
We train decoder-only Qwen2.5-style [Qwen et al., 2025] models with 100M parameters on a large-scale synthetic
reasoning dataset generated using the GSM-Infinite framework. The full corpus contains 30B tokens spanning
multiple operation ranges and contextual templates, and is partitioned into disjoint splits for pre-training, mid-
training, and post-training to avoid distribution contamination.
Pre-training. Pre-training exposes the model to a diverse corpus to acquire general knowledge. In our controlled
reasoning tasks, it focuses on equipping the model with foundational reasoning skills and rules for arithmetic
operations in our synthetic dataset. The emphasis is on mastering basic reasoning primitives rather than broad
knowledge. Following Chinchilla scaling [Hoffmann et al., 2022] and trends in data-rich regimes [Li et al., 2025],
we pre-train our 100M model on 10B tokens (100× parameters). The dataset consists of op=2-10 operations across
templates, allowing the model to master reasoning while retaining headroom for complex tasks. The model achieves
near-saturated pass@128 accuracy, ensuring that improvements in deeper tasks reflect true generalization.
Mid-training. Mid-training is an intermediate phase between pre-training and post-training, gaining attention for
its role in improving downstream fine-tuning and RL performance [Liu et al., 2025a, Wang et al., 2025, Akter
et al., 2025]. It typically involves using higher-quality or instruction-formatted data with next-token prediction or
SFT objectives. Mid-training stabilizes optimization and facilitates RL scaling by providing structured reasoning
supervision, bridging the gap between broad pre-training corpora and reward-oriented RL data. In our setup, we
implement a streamlined version of mid-training, maintaining the same pre-training objective but narrowing the data
distribution similar to RL, where the model exhibits emerging but incomplete competence. By focusing supervision
on this boundary, we aim to strengthen higher-level reasoning priors that RL can amplify.2
Post-training. Post-training refines the model’s performance on specific tasks after pre-training with task-specific
data or objectives. It generally involves two strategies: 1) Supervised Fine-tuning (SFT): Training on labeled datasets
or task-specific instructions; 2) Reinforcement Learning (RL): The model optimizes by receiving rewards for its
actions. As our pre-training data is already structured and task-specific, we mainly focus on RL for post-training.
Using GRPO [Shao et al., 2024], we train on curated subsets designed to probe generalization in deeper operation
ranges and novel templates.
3
When Does Post-Training Incentivize Reasoning Beyond the Base Model?
To disentangle the contributions of pre-training and post-training to reasoning capabilities, we isolate the specific
impact of RL. We ask: whether and when RL extends a base model’s reasoning capabilities beyond those inherited
from pre-training. By fixing the pre-training stage and varying the difficulty and coverage of post-training data, we
identify the specific regimes where RL drives genuine compositional generalization rather than merely amplifying
existing skills.
Task Setting. We focus on extrapolative generalization (we examine contextual transfer for post-training in
Appendix A.6), defining three problem categories based on operation counts: In-Distribution (ID) problems within
the pre-training range (op=2-10); OOD-edge problems just beyond this range (op=11-14), where the base model
retains non-zero pass@128 accuracy; and OOD-hard problems substantially beyond the pre-training distribution
(op=15-20), where the base model exhibits near-zero accuracy3. Solving OOD-hard problems requires composing
atomic operations learned from ID data in novel ways to accommodate increased reasoning depth. The experimental
setup proceeds as follows:
• Pre-training: The base model is pre-trained on 10B tokens consisting of ID problems.
• Post-training: We apply GRPO with a total of 200K samples from four distinct difficulty ranges: op=7-10
(ID), op=9-12 (mixed), op=11-14 (edge), and op=17-20 (hard).
For additional information on the training dynamics and the data recipe, see A.5 and A.9.
Observation 1
As shown in Figure 3, the efficacy of post-training is highly sensitive to the pre-training and post-training data
regime: (i) For ID tasks (op=2-10), there are obvious performance gains on pass@1 but no improvement
on pass@128 regardless of the RL data regime, which indicates that RL only sharpens existing capabilities
without extending them. (ii) However, for OOD tasks (op=11-14 and op=15-20), RL always improves
pass@128 performance when applied on the edge of competence data (op=11-14), demonstrating genuine
capability gains beyond pre-training.
2Mid-training is only applied in Section 5.
3We illustrate this performance ladder in Appendix A.3.4.

--- Page 5 ---
1
2
4
8
16 32 64 128
85
90
95
100
performance (%)
ID(op=2-10)
Base
RL op=7-10
RL op=9-12
RL op=11-14
RL op=17-20
1
2
4
8
16 32 64 128
20
40
60
80
OOD-mid(op=11-14)
1
2
4
8
16 32 64 128
0
10
20
OOD-hard(op=15-20)
pass@k
Figure 3: pass@k performance on three tasks: ID (op=2-10), OOD-edge (op=11-14), OOD-hard (op=(15-
20)). RL is applied to four different data regimes (colors). RL on ID tasks never improves beyond the base model at
pass@128. RL consistently improves pass@128 on harder tasks when applied beyond the base model’s capacity.
Takeaway 1
RL produces true capability gains (pass@128) beyond base models only when two conditions hold: (i) The
task is not heavily covered during pre-training, leaving sufficient headroom for exploration; and (ii) the RL data is
calibrated to the model’s edge of competence, neither too easy (in-distribution) nor too hard (out-of-distribution).
Discussion 1
Connection with recent work. Recent studies report seemingly conflicting conclusions about whether RL can
enhance a base model’s reasoning ability. On the one hand, Zhao et al. [2025], Yue et al. [2025] argue that RL
does not improve pass@128 accuracy when evaluated on standard tasks such as math and coding—domains
that are already well covered during pre-training. On the other hand, work on synthetic tasks with little pre-
training coverage [Liu et al., 2025b, Yuan et al., 2025, Sun et al., 2025a] reports substantial post-training
gains. Our controlled setting reconciles these findings by showing that they arise from different regions of the
post-training difficulty spectrum. RL yields no advantage on in-domain tasks that the base model already solves,
as performance saturates with increasing pass@k. In contrast, when RL targets genuinely OOD tasks where the
base model fails, we observe clear extrapolative improvements, provided the RL data lie near the model’s “edge
of competence.”
Practical Guidance 1
Design RL data around the model’s edge of competence. We recommend filtering the RL dataset to target
tasks where the model fails at pass@1 but succeeds at pass@k. This strategy avoids redundancy on high-
pass@1 tasks while preventing reward sparsity on zero-pass@k tasks. This process could also be iterative:
we can periodically re-evaluate the pool of “edge of competence” tasks; as the model gets stronger, previously
out-of-distribution tasks will drift into the solvability gap, creating a natural, self-paced curriculum.
4
How Does Pre-training Exposure Shape Post-Training Generalization?
Having established the conditions under which post-training incentivizes generalization, we turn to a foundational
question: How does pre-training exposure shape post-training generalization? We hypothesize that pre-training
exposure to fundamental reasoning primitives is crucial for effective post-training generalization. To explore this
question, with a fixed RL data recipe and setup, we vary the distribution of pre-training data and examine its effect
on post-training generalization.
Task Setting. In this study, we focus on contextual generalization to long-tailed context B contexts with atomic
reasoning primitives (op=2 examples) during pre-training (experiments on simple contextual generalization and
extrapolation are provided in the Appendix A.6.1 and A.7 respectively). By manipulating the ratio of long-tailed
context B atomic op=2 examples during pre-training, we aim to assess how exposure to these basic primitives shapes
the model’s ability to transfer learned skills and extrapolate effectively during post-training. Our experimental setup
is structured as follows:
• Pre-training: The base model is pre-trained on 10B tokens consisting of op=2-20 context A and long-tailed
op=2 context B examples, where we vary the ratio of atomic op=2 examples to long-tailed context B exposure.
• Post-training: RL is applied on 200K samples, consisting of 50% context A and 50% context B, spanning
op=2-20. Further details on the training dynamics and data recipe can be found in Appendix A.8 and A.9.

--- Page 6 ---
Fail to generalize
Generalize well
Generalize well
Figure 4: pass@128 performance on context B after post-trained with a 50% context A + 50% context B mixture.
Different lines represent levels of pre-training exposure to long-tailed context B atomic op=2 examples. RL
incentivizes contextual generalization when the model has minimal exposure (≥1%) to context B in pre-training.
Observation 2
As shown in Figure 4, the impact of pre-training exposure to long-tailed contexts on post-training generalization
is substantial: (i) When pre-training excludes context B or provides no (0%) or very little exposure (0.1%), RL
fails to transfer to context B. (ii) Introducing even 1% of context B data during pre-training significantly enhances
post-training generalization even to the hardest tasks of op=20. This observation underscores that while RL
plays a crucial role in generalization, its effectiveness is heavily dependent on the coverage of the pre-training
data, particularly the inclusion of long-tailed contexts.
Takeaway 2
RL incentivizes contextual generalization only when the base model already contains the necessary
primitives. Without minimal pre-training exposure to a new context, RL cannot induce transfer. However, even
sparse exposure (e.g., ≥1%) provides a sufficient seed that RL can reinforce during post-training, yielding robust
cross-context generalization.
Discussion 2
Replication or Creation? We examine the distribution of topological similarity between the generated correct
context B graphs and the ground-truth topology from context A in Figure 5. High similarity indicates that the
model primarily replicates existing context A reasoning patterns, while low similarity suggests the emergence of
novel reasoning structures distinct from context A.
0.2
0.4
0.6
0.8
1.0
1
5
10
20
40
probability (%)
0.1% B
op=2-10
op=11-14
op=15-20
0.2
0.4
0.6
0.8
1.0
1
5
10
20
40
1% B
0.2
0.4
0.6
0.8
1.0
1
5
10
20
40
10% B
similarity
Figure 5: Distribution of topological similarity between generated correct context B and gold context A graphs.
We observe effects between task difficulty and pre-training exposure: 1) For simpler compositions (op=2-10),
models tend to replicate existing patterns from context A. 2) As task complexity increases (op=11-20), models
generate more novel structures, especially when pre-trained with sufficient exposure to context B.
Practical Guidance 2
Seed long-tail primitives in pre-training to unlock RL potential. RL cannot synthesize capabilities from a
void; it requires latent “seeds” to amplify. However, these seeds need not be complex. Our results show that RL
can successfully extrapolate to hard tasks as long as the atomic reasoning primitives are present in pre-training.
Practitioners should prioritize broad coverage of basic domain knowledge, rules, and skills (at ≈1% density)
rather than striving for complex data samples. Once these fundamental primitives are established, RL effectively
acts as a compositor, combining them to solve complex out-of-distribution problems.

--- Page 7 ---
5
How Does Mid-Training Interact with Post-Training?
While RL effectively enhances extrapolative generalization, its success is often contingent on the representational
priors established during pre-training. Recent work [Wang et al., 2025, Liu et al., 2025a] proposes mid-training as
an intermediate phase between pre-training and post-training, designed to bridge data distributions and strengthen
reasoning priors before downstream adaptation.
This raises a key question: how do mid-training and RL interact under a fixed compute budget, and what balance
between them yields the greatest generalization gains? In this section, we examine the synergy between mid-training
and post-training, seeking to define how their interaction drives reasoning generalization.
Compute Budget Formulation. For fair comparison, we normalize both phases to equivalent training tokens
based on flops. For mid-training, the consumption Tmid is the number of supervised tokens processed. For RL, the
token-equivalent cost is approximated as:
TRL ≈5
3N · r · Ltotal,
(1)
where N is the number of RL samples, r = 6 the rollout multiplicity, and Ltotal= = 2048 the total token length4.
We systematically vary the RL allocation ratio β ∈[0, 1] to distribute the total budget T between the two phases:
Tmid = (1 −β) · T,
TRL = β · T.
(2)
Task Setting. In this section, we explore the performance of five training configurations using the same base model
pre-trained on 10B op=2-10 data: Full mid-training on 1B supervised tokens from the op=11-14 range, Full
RL with 100 steps of batch size 1024 from the same op=11-14 range, and three mixing strategies—Light-RL
(β = 0.2), Medium-RL (β = 0.5), and Heavy-RL (β = 0.8), which balance mid-training and RL under an equivalent
compute budget. The compute budget formulation in Section 5 allows for a direct comparison of data mixture
strategies. Detailed training setup can be found in Appendix A.10.
11
12
13
14
50
60
performance (%)
OOD-edge (op=11-14)
15
16
17
18
19
20
0
10
20
OOD-hard (op=15-20)
11
12
13
14
80
90
OOD-edge (op=11-14)
15
16
17
18
19
20
0
25
50
75
OOD-hard (op=15-20)
Task Difficulty
pass@1
pass@128
Full Mid
Full RL
Light RL
Heavy RL
Figure 6: pass@1 and pass@128 performances on extrapolative tasks under varying mid- and post-training
mixture ratios. The data used in mid- and post-training is applied within the OOD-edge ranges. Different lines
indicate the compute allocation strategies. Heavy-RL always improves the unseen OOD-hard tasks, while Light-RL
improves best pass@1 on OOD-edge tasks.
Observation 3
As shown in Figure 6, compute allocation induces qualitatively different behaviors across the generalization
spectrum. (1) On OOD-edge tasks, configurations with full mid-training and light RL outperform those with
heavy or full RL, with light RL achieving the best pass@1 performance. (2) For OOD-hard tasks, reallocating
more budget toward heavy RL substantially improves performance on the hardest instances in both pass@1 and
pass@128. These trends suggest that RL-driven exploration is indispensable for generalizing to harder tasks,
but a substantial mid-training allocation remains critical for instilling the priors that RL can effectively exploit.
We further analyze the impact of varying compute budgets in Appendix A.10.
Takeaway 3
Introducing a mid-training phase that bridges pre- and post-training distributions substantially strength-
ens generalization under a fixed compute budget. This highlights mid-training as an underexplored but
powerful lever in training design. Compute should be allocated in a task-aware manner: (i) when prioritizing
in-distribution performance, allocate more budget to mid-training with only light RL; (ii) for out-of-distribution
generalization, reserve a modest portion of compute for mid-training to establish essential priors, and dedicate
the remaining budget to heavier RL exploration.
4Detailed budget derivation are provided in Appendix A.10.1

--- Page 8 ---
Discussion 3
The Role of Mid-Training. Recent work [Shao et al., 2025, Gandhi et al., 2025] has noted that models like
Qwen [Qwen et al., 2025] respond far more effectively to RL than architectures such as LLaMA [Touvron et al.,
2023]. A converging explanation is the presence of a mid-training stage that aligns supervision more closely
with the post-training distribution. Reasoning-oriented mid-training has been shown to substantially increase a
model’s RL readiness. Wang et al. [2025] find that LLaMA models mid-trained on structured reasoning data
achieve RL performance comparable to stronger Qwen bases, indicating that mid-training largely determines
downstream RL responsiveness. Complementarily, Liu et al. [2025a] shows that mid-training serves as a
distributional bridge, reducing forgetting and easing adaptation by narrowing the gap between pre-training and
RL tasks. This perspective is further consistent with the frontloading principle of Akter et al. [2025]: injecting
structured reasoning supervision earlier provides the scaffolding that later training stages, including RL, can
efficiently amplify. Together, these works point to a unified conclusion: mid-training is a strategically important
component that conditions models for stable and sample-efficient RL, enabling improvements that go beyond
merely sharpening existing abilities.
Practical Guidance 3
Balance mid-training and post-training around complementary strengths. Design the training pipeline
by treating mid-training as the phase for installing priors and RL as the phase for scaling exploration. For
mid-training, curate datasets that lie at the model’s “edge of competence”, which stabilizes the primitives required
for RL. Practitioners should adjust the compute budget based on the deployment goal: (1) For reliability on
similar tasks (OOD-edge), allocate the majority of compute to mid-training and use light RL. (2) For exploration
on complex tasks (OOD-hard), allocate mid-training to a modest budget (sufficient only to establish priors) and
spend heavy compute on RL exploration.
6
Mitigating Reward Hacking via Process Supervision in Outcome Rewards
Post-training with outcome-based rewards has proven highly effective in improving reasoning performance, yet it
remains vulnerable to reward hacking—a failure mode where models achieve high final accuracy by exploiting
spurious shortcuts or producing correct answers through invalid reasoning chains. Earlier, we introduced process
verification as an evaluation criterion that rewards models only when both intermediate steps and the final outcome
are correct. Here, we extend this principle into the reward design itself, asking: Can process-aware supervision
mitigate reward hacking while preserving generalization performance?
Task Setting. To encourage models to generate not only correct final answers but also valid intermediate reasoning
steps, we augment the outcome reward with process-level verification. We define a composite reward function:
R = αRout + (1 −α)Rpv.
(3)
Rout denotes the traditional outcome-based reward (1 for a correct final answer, 0 otherwise), which may be
sparse and susceptible to outcome reward hacking. Rpv represents the process verification reward defined by the
process-level accuracy criteria in Section A.2, which is a dense reward reflecting the correctness of each reasoning
step. α ∈[0, 1] controls the balance between outcome accuracy and process fidelity. We also consider a stricter
formulation:
R =
(
Rout,
if Rpv = 1,
0,
otherwise.
which grants outcome rewards only when the entire reasoning process is verified as correct. This setup pro-
vides process-level supervision to reduce reward hacking. Under this reward setup, we conduct post-training
on op=11-14 using different reward compositions to assess how varying degrees of process supervision affect
reasoning generalization.
Observation 4
As shown in Figure 7, integrating process verification notably improves pass@1 by 4–5% across extrapolative
(op=15-20) settings. Moderate reward mixes (0.2, Rout + 0.8, Rpv) achieve the best balance between outcome
accuracy and reasoning consistency, while the strict reward (Rout only if Rpv=1) further enhances substantial
improvements. These results confirm that process-level supervision effectively mitigates reward hacking and
encourages faithful reasoning behavior.

--- Page 9 ---
70
72
74
76
pass@k (%)
70.7
+4.7
+4.6
+5.2
op=2-14
6
8
10
5.6
+4.3
+3.4
+4.1
op=15-20
94.0
94.5
95.0
95.5
96.0
pass@k (%)
94.7
-0.0
-0.1
+0.3
op=2-14
24
26
28
23.2
+4.5
+3.0
+2.6
op=15-20
pass@1                                                                  pass@128
1.0 Outcome
0.2 Outcome + 0.8 Process
0.5 Outcome + 0.5 Process
1.0 Outcome if Process is Correct
Figure 7: pass@k performance under different reward compositions. Each bar corresponds to a distinct reward-
mixing strategy. Incorporating process-level information into the outcome reward consistently yields measurable
performance gains across evaluation settings.
Takeaway 4
Process-aware rewards mitigate reward hacking and enhance reasoning fidelity. Incorporating process
verification into the reward function aligns reinforcement signals with valid reasoning behavior, leading to
measurable improvements in both accuracy and generalization under complex, compositional settings.
Discussion 4
How does process verification reshape RL generalization? We investigate whether incorporating process
verification can better guide RL toward faithful reasoning. We analyze how different reward formulations affect
both correctness and structural error patterns during RL fine-tuning.
20
30
40
50
op=11-14
Percentage (%)
correct
40
50
60
70
80
dependency mismatch
0
5
10
15
missing nodes
0
5
10
15
op=15-20
Percentage (%)
80
85
90
95
100
5
10
15
Main Error Types for Different Reward Mixtures
Base model
1.0 Outcome
0.2 Outcome + 0.8 Process
1.0 Outcome if Process is Correct
Figure 8: Effects of reward mixtures on reasoning correctness and structural error types.
As evidenced in Figure 8, integrating process verification consistently shifts the model away from shortcut
exploitation toward structurally faithful reasoning. By reducing structural errors and reinforcing correct inter-
mediate steps, process-aware rewards enable more reliable improvements under extrapolative (op=15{20)
settings. These results highlight that aligning rewards with valid reasoning traces is crucial for scaling RL-based
generalization.
Practical Guidance 4
Combine sparse outcome signals with dense process-level feedback. In practice, blending the sparse final-
outcome signal with richer, dense process-level information is beneficial [Gunjal et al., 2025, Khalifa et al.,
2025]. Provided that the process supervision is of high quality [Cui et al., 2025], we recommend incorporating
process-level information into the outcome reward. This helps mitigate reward-hacking and consistently improves
performance.
7
Related Work
RL Generalization of Reasoning LMs. The role of RL in driving generalization in LMs has been the subject
of extensive discussion. Recent work presents differing views on whether RL can extend reasoning beyond the
capabilities of the base model, with contrasting arguments emerging in the literature.
On the one hand, several studies caution against overestimating RL’s ability to push the boundaries of a base
model. Yue et al. [2025] argue that while RL-trained models may outperform base models at small values of
pass@k (e.g., k = 1), the performance advantage diminishes as k increases (e.g., k = 128). Their coverage and

--- Page 10 ---
perplexity analyses suggest that the reasoning capabilities of RL-trained models remain ultimately constrained by
the base model’s representational capacity. Additionally, Wu et al. [2025] provides a theoretical framework asserting
that RL cannot surpass the base model’s inherent limitations, thus challenging the notion that RL can enable new,
generalizable reasoning skills.
On the other hand, there are strong arguments in favor of RL’s ability to enable generalization, particularly in tasks
where the base model performs poorly. Liu et al. [2025b] highlights the success of ProRL in improving performance
on synthesized reasoning tasks, where base models demonstrate significant limitations. Further supporting this
view, Sun et al. [2025a,b] provides clear evidence of RL’s potential to induce novel strategies for complex problem
families. Yuan et al. [2025] propose a synthetic function composition task, demonstrating that RL-trained models
can generalize to unseen function compositions that the base model cannot handle.
In our work, we contribute to this ongoing debate by providing empirical evidence that the two perspectives are
not mutually exclusive. Instead, we show that the conditions under which RL can drive generalization are nuanced
and depend on the base model’s reasoning primitives as well as the nature of the post-training data used during RL
fine-tuning.
Understanding LMs via Controlled Experiments. Several prior work Yuan et al. [2025], Liu et al. [2025b], Sun
et al. [2025a] has emphasized the importance of controlled experiments in understanding the capabilities of LMs.
However, this line of work mainly focuses on synthetic tasks designed for post-training RL, which may not fully
capture the complexities of the full spectrum of reasoning tasks from pre-training to post-training. Especially in the
context of reasoning tasks, controlled settings allow researchers to isolate specific factors, e.g., data contamination,
random-guess answers, as well as controlling the reasoning primitives for different training phases. We build
upon this line of work by designing controlled experiments motivated by Ye et al. [2024] to synthesize GSM-style
reasoning tasks [Cobbe et al., 2021, Liu et al., 2023, Mirzadeh et al., 2025, Zhou et al., 2025a]
8
Conclusion
In this work, we presented a controlled investigation into how pre-training and post-training jointly determine the
reasoning capabilities of language models. By disentangling the contributions of each stage, our study clarifies the
causal mechanisms through which RL enhances or fails to enhance reasoning generalization. Using fully controllable
synthetic reasoning tasks and process-level evaluations, we demonstrated that genuine reasoning improvements
through post-training arise only when key reasoning primitives are established during pre-training. Together, these
results refine our understanding of reasoning development in language models and provide actionable guidance for
constructing data curricula, designing reward functions, and allocating compute across training stages.
Acknowledgment
The authors would like to thank Kai Zhang, Yuetai Li, Ge Zhang, Boshi Wang, Seungone Kim, Yuanzhi Li, Xinyu
Yang, Yao Fu, Ziqiao Ma, Jinjie Ni, and Junyang Lin for their constructive feedback and comments on the early
draft of the paper. Xiang Yue was supported in part by a Carnegie Bosch Institute Fellowship.
References
DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu,
Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao,
Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang
Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin,
Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng
Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang,
Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong,
Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang,
Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Meng Li, Miaojun Wang,
Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge,
Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou,
Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou,
Shaoqing Wu, Shengfeng Ye, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wanjia Zhao, Wen Liu,
Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang,
Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng
Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan
Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao
Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao,
Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng
Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanhong Xu,

--- Page 11 ---
Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren,
Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang
Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang,
Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. Deepseek-r1: Incentivizing reasoning capability in llms via
reinforcement learning, 2025. URL https://arxiv.org/abs/2501.12948.
OpenAI, :, Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar,
Aleksander Madry, Alex Beutel, Alex Carney, Alex Iftimie, Alex Karpenko, Alex Tachard Passos, Alexander
Neitz, Alexander Prokofiev, Alexander Wei, Allison Tam, Ally Bennett, Ananya Kumar, Andre Saraiva, Andrea
Vallone, Andrew Duberstein, Andrew Kondrich, Andrey Mishchenko, Andy Applebaum, Angela Jiang, Ashvin
Nair, Barret Zoph, Behrooz Ghorbani, Ben Rossen, Benjamin Sokolowsky, Boaz Barak, Bob McGrew, Borys
Minaiev, Botao Hao, Bowen Baker, Brandon Houghton, Brandon McKinzie, Brydon Eastman, Camillo Lugaresi,
Cary Bassin, Cary Hudson, Chak Ming Li, Charles de Bourcy, Chelsea Voss, Chen Shen, Chong Zhang, Chris
Koch, Chris Orsinger, Christopher Hesse, Claudia Fischer, Clive Chan, Dan Roberts, Daniel Kappler, Daniel
Levy, Daniel Selsam, David Dohan, David Farhi, David Mely, David Robinson, Dimitris Tsipras, Doug Li,
Dragos Oprica, Eben Freeman, Eddie Zhang, Edmund Wong, Elizabeth Proehl, Enoch Cheung, Eric Mitchell,
Eric Wallace, Erik Ritter, Evan Mays, Fan Wang, Felipe Petroski Such, Filippo Raso, Florencia Leoni, Foivos
Tsimpourlas, Francis Song, Fred von Lohmann, Freddie Sulit, Geoff Salmon, Giambattista Parascandolo, Gildas
Chabot, Grace Zhao, Greg Brockman, Guillaume Leclerc, Hadi Salman, Haiming Bao, Hao Sheng, Hart Andrin,
Hessam Bagherinezhad, Hongyu Ren, Hunter Lightman, Hyung Won Chung, Ian Kivlichan, Ian O’Connell, Ian
Osband, Ignasi Clavera Gilaberte, Ilge Akkaya, Ilya Kostrikov, Ilya Sutskever, Irina Kofman, Jakub Pachocki,
James Lennon, Jason Wei, Jean Harb, Jerry Twore, Jiacheng Feng, Jiahui Yu, Jiayi Weng, Jie Tang, Jieqi Yu,
Joaquin Qui˜nonero Candela, Joe Palermo, Joel Parish, Johannes Heidecke, John Hallman, John Rizzo, Jonathan
Gordon, Jonathan Uesato, Jonathan Ward, Joost Huizinga, Julie Wang, Kai Chen, Kai Xiao, Karan Singhal,
Karina Nguyen, Karl Cobbe, Katy Shi, Kayla Wood, Kendra Rimbach, Keren Gu-Lemberg, Kevin Liu, Kevin
Lu, Kevin Stone, Kevin Yu, Lama Ahmad, Lauren Yang, Leo Liu, Leon Maksin, Leyton Ho, Liam Fedus, Lilian
Weng, Linden Li, Lindsay McCallum, Lindsey Held, Lorenz Kuhn, Lukas Kondraciuk, Lukasz Kaiser, Luke Metz,
Madelaine Boyd, Maja Trebacz, Manas Joglekar, Mark Chen, Marko Tintor, Mason Meyer, Matt Jones, Matt
Kaufer, Max Schwarzer, Meghan Shah, Mehmet Yatbaz, Melody Y. Guan, Mengyuan Xu, Mengyuan Yan, Mia
Glaese, Mianna Chen, Michael Lampe, Michael Malek, Michele Wang, Michelle Fradin, Mike McClay, Mikhail
Pavlov, Miles Wang, Mingxuan Wang, Mira Murati, Mo Bavarian, Mostafa Rohaninejad, Nat McAleese, Neil
Chowdhury, Neil Chowdhury, Nick Ryder, Nikolas Tezak, Noam Brown, Ofir Nachum, Oleg Boiko, Oleg Murk,
Olivia Watkins, Patrick Chao, Paul Ashbourne, Pavel Izmailov, Peter Zhokhov, Rachel Dias, Rahul Arora, Randall
Lin, Rapha Gontijo Lopes, Raz Gaon, Reah Miyara, Reimar Leike, Renny Hwang, Rhythm Garg, Robin Brown,
Roshan James, Rui Shu, Ryan Cheu, Ryan Greene, Saachi Jain, Sam Altman, Sam Toizer, Sam Toyer, Samuel
Miserendino, Sandhini Agarwal, Santiago Hernandez, Sasha Baker, Scott McKinney, Scottie Yan, Shengjia Zhao,
Shengli Hu, Shibani Santurkar, Shraman Ray Chaudhuri, Shuyuan Zhang, Siyuan Fu, Spencer Papay, Steph
Lin, Suchir Balaji, Suvansh Sanjeev, Szymon Sidor, Tal Broda, Aidan Clark, Tao Wang, Taylor Gordon, Ted
Sanders, Tejal Patwardhan, Thibault Sottiaux, Thomas Degry, Thomas Dimson, Tianhao Zheng, Timur Garipov,
Tom Stasi, Trapit Bansal, Trevor Creech, Troy Peterson, Tyna Eloundou, Valerie Qi, Vineet Kosaraju, Vinnie
Monaco, Vitchyr Pong, Vlad Fomenko, Weiyi Zheng, Wenda Zhou, Wes McCabe, Wojciech Zaremba, Yann
Dubois, Yinghai Lu, Yining Chen, Young Cha, Yu Bai, Yuchen He, Yuchen Zhang, Yunyun Wang, Zheng Shao,
and Zhuohan Li. Openai o1 system card, 2024. URL https://arxiv.org/abs/2412.16720.
Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, and Gao Huang. Does
reinforcement learning really incentivize reasoning capacity in llms beyond the base model?, 2025. URL
https://arxiv.org/abs/2504.13837.
Fang Wu, Weihao Xuan, Ximing Lu, Mingjie Liu, Yi Dong, Zaid Harchaoui, and Yejin Choi. The invisible leash:
Why rlvr may or may not escape its origin, 2025. URL https://arxiv.org/abs/2507.14843.
Rulin Shao, Shuyue Stella Li, Rui Xin, Scott Geng, Yiping Wang, Sewoong Oh, Simon Shaolei Du, Nathan Lambert,
Sewon Min, Ranjay Krishna, Yulia Tsvetkov, Hannaneh Hajishirzi, Pang Wei Koh, and Luke Zettlemoyer.
Spurious rewards: Rethinking training signals in rlvr, 2025.
URL https://arxiv.org/abs/2506.
10947.
Edward Yeo, Yuxuan Tong, Morry Niu, Graham Neubig, and Xiang Yue. Demystifying long chain-of-thought
reasoning in llms, 2025. URL https://arxiv.org/abs/2502.03373.
Xumeng Wen, Zihan Liu, Shun Zheng, Shengyu Ye, Zhirong Wu, Yang Wang, Zhijian Xu, Xiao Liang, Junjie Li,
Ziming Miao, Jiang Bian, and Mao Yang. Reinforcement learning with verifiable rewards implicitly incentivizes
correct reasoning in base llms, 2025. URL https://arxiv.org/abs/2506.14245.
Lifan Yuan, Weize Chen, Yuchen Zhang, Ganqu Cui, Hanbin Wang, Ziming You, Ning Ding, Zhiyuan Liu, Maosong
Sun, and Hao Peng. From f(x) and g(x) to f(g(x)): Llms learn new skills in rl by composing old ones, 2025.
URL https://arxiv.org/abs/2509.25123.

--- Page 12 ---
Yiyou Sun, Yuhan Cao, Pohao Huang, Haoyue Bai, Hannaneh Hajishirzi, Nouha Dziri, and Dawn Song. Rl grokking
recipe: How does rl unlock and transfer new algorithms in llms?, 2025a. URL https://arxiv.org/abs/
2509.21016.
Zengzhi Wang, Fan Zhou, Xuefeng Li, and Pengfei Liu. Octothinker: Mid-training incentivizes reinforcement
learning scaling, 2025. URL https://arxiv.org/abs/2506.20512.
Emmy Liu, Graham Neubig, and Chenyan Xiong. Midtraining bridges pretraining and posttraining distributions,
2025a. URL https://arxiv.org/abs/2510.14865.
Zeyuan Allen-Zhu. ICML 2024 Tutorial: Physics of Language Models, July 2024. Project page: https:
//physics.allen-zhu.com/.
Tian Ye, Zicheng Xu, Yuanzhi Li, and Zeyuan Allen-Zhu. Physics of language models: Part 2.1, grade-school math
and the hidden reasoning process, 2024. URL https://arxiv.org/abs/2407.20311.
Yang Zhou, Hongyi Liu, Zhuoming Chen, Yuandong Tian, and Beidi Chen. Gsm-infinite: How do your llms behave
over infinitely increasing context length and reasoning complexity?, 2025a. URL https://arxiv.org/
abs/2502.05252.
Amrith Setlur, Matthew Y. R. Yang, Charlie Snell, Jeremy Greer, Ian Wu, Virginia Smith, Max Simchowitz,
and Aviral Kumar. e3: Learning to explore enables extrapolation of test-time compute for llms, 2025. URL
https://arxiv.org/abs/2506.09026.
Ruochen Zhou, Minrui Xu, Shiqi Chen, Junteng Liu, Yunqi Li, Xinxin Lin, Zhengyu Chen, and Junxian He. Does
learning mathematical problem-solving generalize to broader reasoning?, 2025b. URL https://arxiv.
org/abs/2507.04391.
Maggie Huan, Yuetai Li, Tuney Zheng, Xiaoyu Xu, Seungone Kim, Minxin Du, Radha Poovendran, Graham
Neubig, and Xiang Yue. Does math reasoning improve general llm capabilities? understanding transferability of
llm reasoning. arXiv preprint arXiv:2507.00432, 2025. URL https://arxiv.org/abs/2507.00432.
Kai Zhang, Xiangchao Chen, Bo Liu, Tianci Xue, Zeyi Liao, Zhihan Liu, Xiyao Wang, Yuting Ning, Zhaorun
Chen, Xiaohan Fu, Jian Xie, Yuxuan Sun, Boyu Gou, Qi Qi, Zihang Meng, Jianwei Yang, Ning Zhang, Xian
Li, Ashish Shah, Dat Huynh, Hengduo Li, Zi Yang, Sara Cao, Lawrence Jang, Shuyan Zhou, Jiacheng Zhu,
Huan Sun, Jason Weston, Yu Su, and Yifan Wu. Agent learning via early experience, 2025. URL https:
//arxiv.org/abs/2510.08558.
Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng
Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren
Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang,
Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan,
Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical
report, 2025. URL https://arxiv.org/abs/2412.15115.
Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego
de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican,
George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen,
Jack W. Rae, Oriol Vinyals, and Laurent Sifre. Training compute-optimal large language models, 2022. URL
https://arxiv.org/abs/2203.15556.
Houyi Li, Wenzhen Zheng, Qiufeng Wang, Zhenyu Ding, Haoying Wang, Zili Wang, Shijie Xuyang, Ning Ding,
Shuigeng Zhou, Xiangyu Zhang, and Daxin Jiang. Predictable scale: Part ii, farseer: A refined scaling law in
large language models, 2025. URL https://arxiv.org/abs/2506.10972.
Syeda Nahida Akter, Shrimai Prabhumoye, Eric Nyberg, Mostofa Patwary, Mohammad Shoeybi, Yejin Choi, and
Bryan Catanzaro. Front-loading reasoning: The synergy between pretraining and post-training data, 2025. URL
https://arxiv.org/abs/2510.03264.
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang,
Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language
models, 2024. URL https://arxiv.org/abs/2402.03300.
Rosie Zhao, Alexandru Meterez, Sham Kakade, Cengiz Pehlevan, Samy Jelassi, and Eran Malach. Echo chamber:
Rl post-training amplifies behaviors learned in pretraining, 2025. URL https://arxiv.org/abs/2504.
07912.

--- Page 13 ---
Mingjie Liu, Shizhe Diao, Ximing Lu, Jian Hu, Xin Dong, Yejin Choi, Jan Kautz, and Yi Dong. ProRL: Prolonged
Reinforcement Learning Expands Reasoning Boundaries in Large Language Models, May 2025b. URL https:
//arxiv.org/abs/2505.24864.
Kanishk Gandhi, Ayush Chakravarthy, Anikait Singh, Nathan Lile, and Noah D. Goodman. Cognitive behaviors
that enable self-improving reasoners, or, four habits of highly effective stars, 2025. URL https://arxiv.
org/abs/2503.01307.
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste
Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and
Guillaume Lample. Llama: Open and efficient foundation language models, 2023. URL https://arxiv.
org/abs/2302.13971.
Anisha Gunjal, Anthony Wang, Elaine Lau, Vaskar Nath, Yunzhong He, Bing Liu, and Sean Hendryx. Rubrics as
rewards: Reinforcement learning beyond verifiable domains, 2025. URL https://arxiv.org/abs/2507.
17746.
Muhammad Khalifa, Rishabh Agarwal, Lajanugen Logeswaran, Jaekyeom Kim, Hao Peng, Moontae Lee, Honglak
Lee, and Lu Wang. Process reward models that think, 2025. URL https://arxiv.org/abs/2504.
16828.
Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Yuchen Zhang, Jiacheng Chen, Wendi Li, Bingxiang He,
Yuchen Fan, Tianyu Yu, Qixin Xu, Weize Chen, Jiarui Yuan, Huayu Chen, Kaiyan Zhang, Xingtai Lv, Shuo
Wang, Yuan Yao, Xu Han, Hao Peng, Yu Cheng, Zhiyuan Liu, Maosong Sun, Bowen Zhou, and Ning Ding.
Process reinforcement through implicit rewards, 2025. URL https://arxiv.org/abs/2502.01456.
Yiyou Sun, Shawn Hu, Georgia Zhou, Ken Zheng, Hannaneh Hajishirzi, Nouha Dziri, and Dawn Song. Omega: Can
llms reason outside the box in math? evaluating exploratory, compositional, and transformative generalization,
2025b. URL https://arxiv.org/abs/2506.18880.
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert,
Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve
math word problems. arXiv preprint arXiv:2110.14168, 2021.
Bingbin Liu, Sebastien Bubeck, Ronen Eldan, Janardhan Kulkarni, Yuanzhi Li, Anh Nguyen, Rachel Ward, and
Yi Zhang. Tinygsm: achieving ¿80URL https://arxiv.org/abs/2312.09241.
Iman Mirzadeh, Keivan Alizadeh, Hooman Shahrokhi, Oncel Tuzel, Samy Bengio, and Mehrdad Farajtabar. Gsm-
symbolic: Understanding the limitations of mathematical reasoning in large language models, 2025. URL
https://arxiv.org/abs/2410.05229.

--- Page 14 ---
A
Appendix
Contents
A.1
Data Generation Framework . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
15
A.1.1
Graph-Level Formalism
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
15
A.1.2
Abstract and Instance Parameters
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
15
A.1.3
Contextual Rendering
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
16
A.1.4
Generation Pipeline and Structural Knobs . . . . . . . . . . . . . . . . . . . . . . . . . .
18
A.1.5
Deduplication and Canonicalization . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
18
A.2 Task Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
19
A.3 Training Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
20
A.3.1
Model Architecture . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
20
A.3.2
Tokenizer and Input Representation . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
20
A.3.3
Hyperparameters . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
20
A.3.4
Performance Ladder
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
21
A.4
Process-Verified Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
22
A.5 Training Dynamics for § 3
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
23
A.6
Detailed Analysis of Post-Training Effects on Contextual Generalization . . . . . . . . . . . . . .
24
A.6.1
When Reasoning Primitives are Shared During Pre-Training . . . . . . . . . . . . . . . .
24
A.6.2
When Only Atomic Primitives are Exposed During Pre-Training . . . . . . . . . . . . . .
25
A.6.3
Training Dynamics for § A.6.2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
25
A.7
Detailed Analysis of Pre-Training Effects on Extrapolative Generalization . . . . . . . . . . . . .
25
A.7.1
Training Dynamics for § A.7 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
27
A.8 Training Dynamics for § 4
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
27
A.9
Post-Training and Pre-Training Data Recipe . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
28
A.10 Mid-/Post-Training Mixing with Different Computation Budget
. . . . . . . . . . . . . . . . . .
29
A.10.1 Compute Budget of Mid-Training and RL Equivalence . . . . . . . . . . . . . . . . . . .
29

--- Page 15 ---
A.1
Data Generation Framework
This section provides the formal details of the controllable data generation framework used throughout the paper.
We describe (i) the graph-level formalism underlying each reasoning instance, (ii) the abstraction mechanism that
separates structure from numeric and linguistic instantiations, (iii) the contextual rendering function that maps
graphs to natural-language problems, and (iv) the concrete generation pipeline and deduplication procedure.
A.1.1
Graph-Level Formalism
Each reasoning instance is grounded in a directed acyclic graph (DAG)
G = (V, E),
where each node vi ∈V represents a latent quantity (e.g., “number of adult lions”) and each directed edge
(vj →vi) ∈E encodes a functional dependency. We restrict dependencies to elementary arithmetic operations:
vi = fi
 (vj)j∈pa(i)

,
fi ∈{+, −, ×, ÷},
where pa(i) is the parent set of node i.
Given numeric assignments to all leaf nodes, we define an evaluation map
val : V →R
recursively by
val(vi) = fi
 {val(vj)}j∈pa(i)

,
with base cases given by the leaf values. For a designated query node v∗, the ground-truth answer is
a∗:= val(v∗).
In the GSM-Infinite implementation that we build upon [Zhou et al., 2025a], the query node v∗corresponds to:
• the last numeric node in the topological order of the forward generator, or
• the distinguished unknown parameter in the equation-style reverse generator.
Throughout, the DAG G is treated as the symbolic reasoning graph whose structure is shared across different
numerical instantiations and linguistic realizations.
Reasoning Complexity. We quantify the structural complexity of an instance by the number of arithmetic operations:
op(G) = |E|.
This quantity lower-bounds the minimal length of the compositional reasoning chain needed to compute a∗, and is
the primary knob we vary when studying extrapolative (depth-wise) generalization.
A.1.2
Abstract and Instance Parameters
Following the abstraction mechanism of GSM-Infinite, we explicitly separate structure, numeric instantiation, and
linguistic context.
Abstract Parameters. Each graph G is associated with a set of abstract parameters that:
• specify which variables exist and how they decompose (e.g., that “total animals” decomposes into “lions” and
“elephants”), and
• determine the edge set E and the operation fi attached to each node.
These parameters define a purely symbolic graph, independent of particular numbers or entities.
Instance Parameters. Given an abstract graph, instance parameters instantiate it with concrete values and entities:
• numeric assignments to leaf nodes (e.g., “there are 12 adult lions and 7 elephant calves”), and
• bindings of variables to context-specific surface forms (e.g., “adult lions in the city zoo”).
Instantiating different numeric values on the same abstract graph leads to a family of structurally identical problems
that differ only in their concrete numbers.
Implicit Reasoning. Not all abstract dependencies need to be explicitly verbalized in the natural-language problem.
For a given linguistic rendering, the edge set can be partitioned as
E = Eexplicit ∪Eimplicit,
Eexplicit ∩Eimplicit = ∅,
where (vj →vi) ∈Eexplicit denotes a relation that is directly stated in the text (e.g., “there are 5 more elephants
than lions”), while (vj →vi) ∈Eimplicit denotes a relation that is part of the ground-truth reasoning graph but never
directly verbalized (e.g., “total animals equals lions plus elephants”). This separation allows explicit and implicit
reasoning steps to coexist within the same underlying graph and enables us to probe models’ ability to recover
unspoken dependencies.

--- Page 16 ---
A.1.3
Contextual Rendering
To map a symbolic graph to a natural-language problem, we introduce a contextual rendering function
Φ : (G, τ) 7→x,
where τ ∈T is a contextual template and x is the resulting text instance.
Templates. A template τ (e.g., animals–zoo, teachers–school, movie-festival) specifies:
• how abstract variables are lexicalized into domain-specific surface forms (e.g., “adult lions”, “children in class
A”, “tickets sold on day 1”), and
• which subset of edges is realized explicitly in the wording, thereby determining the split between Eexplicit and
Eimplicit.
For any two templates τa, τb ∈T that differ only in surface context, the induced problems remain structurally
identical:
Struct(Φ(G, τa)) = Struct(Φ(G, τb)),
∀τa, τb ∈T ,
even though their surface realizations, entities, and explicit/implicit splits may differ. Thus, a single abstract
graph can be rendered into semantically distinct yet structurally equivalent problems, which we leverage to study
contextual (breadth-wise) generalization.
Solution Format. The rendering function produces a triple
x =
 [question], [solution], [answer]

,
where:
• [question] is the natural-language representation of the problem posed by the symbolic graph G, typically
including a query regarding some aspect of the graph (e.g., ”How many tickets were sold on day 1?”). It abstracts
away the underlying structure and provides the context for the solution.
• [solution] is a step-by-step derivation that follows the topological order of the symbolic graph G. It includes
intermediate reasoning steps and logical connections between the graph’s elements, ultimately leading to the final
answer. The solution explicitly shows how each part of the problem is derived or calculated.
• [answer] is the final response to the query posed in the [question], derived through the [solution] process. It is
typically a numerical value or a specific entity that answers the question posed.
This structure ensures that the rendered output is both human-readable and logically consistent with the underlying
symbolic graph, maintaining the integrity of the original problem while making it accessible in natural language.

--- Page 17 ---
Example Rendered Instance for teachers–school Context
[Question]
[question]
The number of elementary school in Westhaven City equals the number of public highschool in Westhaven
City. The number of elementary school in Evervale City equals the sum of the number of public highschool
in Evervale City and the number of regional medical school in Westhaven City. The total number of
schools in Evervale City equals 22. The number of elementary school in Brightford equals 3. The number
of public highschool in Brightford equals 2. The number of regional medical school in Brightford equals
the total number of schools in Westhaven City. The number of regional medical school in Westhaven City
equals 2. The number of regional medical school in Evervale City equals 2 times the number of regional
medical school in Brightford. The number of public highschool in Westhaven City equals 3. The number
of public highschool in Evervale City exists, and its number is greater than 0.
How many public highschool does Evervale City have?
[/question]
Solution
[solution]
The question is difficult, so we use equations to solve it.
Define public highschool in Westhaven City as U; so U = 3. Define elementary school in Westhaven City
as B; so B = U = 3. Define regional medical school in Westhaven City as h; so h = 2.
Define total number of schools in Westhaven City as y; d = U +B = 3+3 = 6, so y = d+h = 6+2 = 8.
Define regional medical school in Brightford as Q; so Q = y = 8. Define regional medical school in
Evervale City as S; z = Q = 8, so S = 2z = 16.
Define public highschool in Evervale City as x (unknown). Define elementary school in Evervale City as
m; so m = x + h = x + 2. Define total number of schools in Evervale City as k.
n = x + (x + 2) = 2x + 2,
k = n + S = 2x + 18.
Since k = 22:
2x + 18 = 22,
2x = 4,
x = 2.
[/solution]
Answer
[answer] 2 [/answer]

--- Page 18 ---
A.1.4
Generation Pipeline and Structural Knobs
Our data generator follows a stage-wise procedure reminiscent of GSM-Infinite forward and reverse generators:
1. Structural sampling. We first sample structural knobs that define the dependency graph:
• a target operation count range for op(G);
• graph shape parameters (e.g., allowable in-degree, layering pattern) that control fan-in and depth; and
• operation types fi ∈{+, −, ×, ÷} attached to nodes.
These choices determine a layered DAG G with a unique query node v∗.
2. Abstract and instance parameterization. Given G, we sample abstract parameters (variable roles and decom-
positions) and instance parameters (numeric values on leaves) and evaluate all node values in topological order
using the evaluation map val defined above.
3. Contextual rendering. We choose a template τ ∈T and apply the rendering function Φ(G, τ) to obtain a
natural-language triple (problem, question, solution), deciding which dependencies are verbalized (explicit) and
which remain implicit.
4. Forward vs. reverse modes. Following [Zhou et al., 2025a], we support two modes of generation: In forward
mode, we generate a standard arithmetic word problem where the final node in the topological order is queried.
In reverse mode, we treat one node as an unknown and phrase an equation-style problem where the model must
solve for that quantity, while the rest of the graph remains fully specified.
By jointly varying (i) the operation count op(G) and (ii) the template τ, we obtain a clean two-dimensional testbed
for studying depth scaling and context transfer. The same framework is used to define distinct data distributions for
pre-training, mid-training, and post-training by sampling from different regions of (op(G), τ)-space.
A.1.5
Deduplication and Canonicalization
To guarantee cleanliness and avoid contamination across training and evaluation splits, we perform exact hash-based
deduplication at the level of rendered triples. Each instance is canonicalized by:
• serializing the triple (problem, question, solution) into a normalized string representation (e.g., stripping extrane-
ous whitespace and normalizing numeric formatting), and
• hashing this canonical form to obtain a global identifier.
We discard any duplicate hashes within and across splits, ensuring that no identical problem–solution triple appears
in both training and evaluation.

--- Page 19 ---
A.2
Task Setup
In real-world deployments, language models are expected to generalize reasoning along two complementary
dimensions [Setlur et al., 2025, Zhou et al., 2025b, Huan et al., 2025]. Our controllable dataset makes these
dimensions explicit and allows us to probe how pre-training, mid-training, and post-training shape each type of
generalization.
Notation. Let f pre
θ , f mid
θ
, and f post
θ
denote the language models after pre-training, after additional mid-training, and
after post-training (RL), respectively. We write Correct(f, G, τ) for correctness on instances generated from graph
G under template τ, using the strict metric defined in the evaluation protocol below.
Extrapolative (Depth) Generalization. We parameterize each training phase ϕ ∈{pre, mid, post} by the range of
operation counts it sees. Let Oϕ be the set of op(G) values present in the training distribution of phase ϕ, and let
Otrain = Opre ∪Omid ∪Opost.
An in-distribution evaluation condition uses graphs with op(G) ∈Otrain, while an extrapolative (out-of-distribution,
OOD) condition evaluates on graphs with
op(G) > max Otrain.
A model exhibits extrapolative generalization if it maintains high process-verified accuracy on these longer, unseen
operations while remaining stable on in-distribution ones. By the varied difficulty ranges populate Opre, Omid, and
Opost, we can isolate how each phase contributes to depth-wise generalization.
Contextual (Breadth) Generalization. A fixed reasoning graph G can be rendered into structurally equivalent
instances under different templates,
Struct(Φ(G, τa)) = Struct(Φ(G, τb))
in principle,
Our dataset is randomly sampled during training and does not deliberately align graphs across templates. As a
result, most graphs are observed only under a subset of contexts during training. Let T train
ϕ
denote the templates
exposed during training phase ϕ, and T eval the broader evaluation pool, including long-tailed templates. A model at
phase ϕ demonstrates contextual generalization if it preserves reasoning performance when the narrative surface
form shifts, even when the new context was never encountered during training:
Acc(f ϕ
θ , G, τa) ≈Acc(f ϕ
θ , G, τb),
τb /∈T train
ϕ
.
Under this setup, contextual generalization measures whether the model has learned transferable reasoning primitives
rather than memorized task styles, allowing it to apply the same structural reasoning across known, unseen, and
long-tailed narrative environments.

--- Page 20 ---
A.3
Training Setup
A.3.1
Model Architecture
We conduct experiments using decoder-only Qwen2.5 Architecture [Qwen et al., 2025] models with 100M parame-
ters. The detailed architecture configurations are as in Table 1
Component
Configuration
Model Type
Qwen2.5
Number of Layers
12
Hidden Size
768
Intermediate Size
3,072
Number of Attention Heads
12
Number of Key-Value Heads
2
Activation Function
SiLU
RMS Norm Epsilon
1e-06
Table 1: Model architecture details for the 100M-parameter Qwen2.5 model used in experiments.
A.3.2
Tokenizer and Input Representation
We follow the Physics of Language Models series [Ye et al., 2024] and train a byte-pair encoding (BPE) tokenizer
directly on our synthetic reasoning corpus. The resulting vocabulary has 2,200 tokens (including special tokens).
All problems, questions, and solutions are tokenized with a maximum sequence length of 2,048 tokens.
A.3.3
Hyperparameters
Pre-training. All experiments start from a 100M-parameter Qwen2.5 model trained from scratch on our controllable
reasoning corpus, using a 100× token-to-parameter ratio, pre-training on 10B tokens. We use a context length of
2048 tokens, batch-size 512K tokens, learning rate 2 × 10−4 with weight decay 0.1, cosine decay with minimum
learning rate 3 × 10−5, warmup ratio 5%, and a single epoch over the corpus. All models are trained in bf16
precision.
Mid-training (Continue Pre-training). Starting from the pre-trained checkpoint, we perform an additional and
optional curriculum in § 5. We train with maximum sequence length 2,048. We use a global batch size of 512K
tokens, learning rate 1 × 10−4, weight decay 0.1, cosine decay with minimum learning rate 3 × 10−5, and a higher
warmup ratio of 15%.
Post-training. Finally, we apply RL fine-tuning usin GRPO [Shao et al., 2024]. We use a global batch size of 1,024
examples, maximum prompt and response lengths of 1024 tokens, and two training epochs. The actor uses learning
rate 1 × 10−6, PPO mini-batch size 256, micro-batch size 16 per GPU, KL regularization with coefficient 10−3
(low-variance KL penalty), and zero entropy bonus. During RL rollouts we sample with temperature TRL = 1.0,
top-p = 1.0, and no top-k truncation (full nucleus sampling). For offline evaluation and reporting we generate with
temperature Teval = 0.7, top-p = 1.0, and top-k = −1 (no truncation), using a maximum of 1,024 new tokens per
problem.

--- Page 21 ---
A.3.4
Performance Ladder
5000 10000 15000 20000
20
40
60
80
100
performance (%)
op=2-10
1
128
pass@k
5000 10000 15000 20000
0
20
40
60
op=11-14
1
128
pass@k
5000 10000 15000 20000
0.0
2.5
5.0
7.5
10.0
op=15-20
1
128
pass@k
steps
5000 10000 15000 20000
0
25
50
75
100
performance (%)
op=11
pass@1
pass@128
pass@k
5000 10000 15000 20000
0
25
50
75
100
op=12
pass@1
pass@128
pass@k
5000 10000 15000 20000
0
25
50
75
100
op=13
pass@1
pass@128
pass@k
5000 10000 15000 20000
0
5
10
15
op=14
pass@1
pass@128
pass@k
steps
Figure 9: Pre-training dynamics across varying operation ranges: In-distribution tasks (op=2-10), edge-of-
competence OOD tasks (op=11-14), and OOD-hard tasks (op=15-20). The plots show the performance
measured by pass@k over training steps.
The performance ladder defines three key levels based on task difficulty: 1)In-distribution tasks (op=2-10):
Aim for near-100% pass@128 accuracy; 2)OOD-edge tasks (op=11-14): Ensure non-zero pass@128 perfor-
mance; 3) OOD-hards tasks (op=15-20): Aim for zero pass@128, signaling the model’s competence limits.
Post-training is performed on the edge of competence, ensuring the model generalizes to harder tasks. A breakdown
of training dynamics across these performance levels is shown in Figure 9.

--- Page 22 ---
A.4
Process-Verified Evaluation
Given an input instance with ground-truth graph (G, a∗), the model produces a free-form solution s. We determinis-
tically parse s into a predicted dependency graph
ˆG = (ˆV, ˆE, c
val),
ˆa,
where nodes in ˆV correspond to named intermediate quantities in the solution, ˆE encodes which previously defined
quantities each step depends on, c
val stores the inferred numeric value for each node, and ˆa is the extracted final
answer. The parser segments the solution into “Define ...as ...” steps, infers each step’s dependencies from the
variables it uses, and evaluates the last computable arithmetic expression in the step (falling back to the last numeric
literal if needed) to obtain a numeric value. This yields a graph-level representation of the model’s reasoning trace
aligned with the gold dependency graph.
Let the gold graph be
G = (V, E, val),
a∗,
with node set V, edge set E, and value map val. We evaluate the reasoning process at the step level. For each gold
node v ∈V, define a per-step correctness indicator
s(v; ˆG, G) =







1,
if v ∈ˆV, pa ˆG(v) = paG(v), and
val(v), c
val(v) are both defined and c
val(v) = val(v),
0,
otherwise,
where paG(v) and pa ˆG(v) denote the parent sets (dependencies) of v in the gold and predicted graphs, respectively.
Missing nodes, incorrect dependency sets, or mismatched values all yield s(v; ˆG, G) = 0.
We then define the process accuracy of a predicted reasoning trace as the average step-level accuracy over all
gold nodes:
ProcessAcc( ˆG; G) = 1
|V|
X
v∈V
s(v; ˆG, G).
Extra predicted nodes v ∈ˆV \ V are allowed and do not affect ProcessAcc; they correspond to redundant but
compatible intermediate steps.
A prediction is regarded as fully correct only when both the reasoning graph and the final answer match. We
formalize this via a verified correctness:
VerifiedCorrect(ˆa, ˆG; a∗, G) =
(
1,
if ProcessAcc( ˆG; G) = 1 and ˆa = a∗,
0,
otherwise.
Accordingly, all pass@k metrics (e.g., pass@1, pass@128) reported in this work treat a sample as correct
only when the model (i) predicts every gold step correctly (step-level process accuracy = 1) and (ii) produces the
correct final answer. This strict criterion ensures that reported gains reflect genuine, faithful reasoning rather than
coincidental correctness.

--- Page 23 ---
A.5
Training Dynamics for § 3
In this section, we provide a detailed analysis on the training dynamics for different post-training recipes in
extrapolative generalization. NLL Reduction Across Evaluation Ranges. We analyze the post-training across
different post-training data recipes used in § 3 and their impact on NLL reduction across various evaluation operation
ranges.
Figure 10: NLL reduction compared with the base model. White boxes denote RL-trained operation ranges. NLL
gains decay smoothly as the evaluation range diverges from the RL-trained operations. Notably, RL on op=11-14
achieves the largest NLL reduction on op=15-20.
We can observe from Figure 10 that post-training consistently reduces NLL across all evaluation ranges, with the
most significant gains occurring in op=11-14 range. This indicates that the model effectively learns to compose
atomic skills to tackle more complex problems. Post-training Dynamics. We further investigate the reward
dynamics during post-training across different data recipes.
0.72
0.74
0.76
0.78
0.80
rewards
RL op=7-10
0.50
0.55
0.60
0.65
RL op=9-12
0.2
0.3
0.4
0.5
RL op=11-14
0.04
0.02
0.00
0.02
0.04
RL op=17-20
0
100
200
Step
245
250
255
260
response length
0
100
200
Step
280.0
282.5
285.0
287.5
290.0
292.5
0
100
200
Step
280
290
300
310
320
330
0
100
200
Step
277.5
280.0
282.5
285.0
287.5
Figure 11: Reward dynamics across different post-training data recipes. RL on op=9-12 and op=11-14
tasks, which are calibrated to the model’s edge of competence, leads to genuine improvements in reasoning. However,
when the task difficulty is either too easy or too hard, the reward stagnates, indicating limited learning progress.
From Figure 11, we observe that post-training on tasks aligned with the model’s edge of competence (op=9-12
and op=11-14) leads to significant reward improvements, indicating effective learning. In contrast, when the tasks
are too easy (op=7-10) or too hard (op=17-20), the reward plateaus, suggesting limited learning progress in
these regimes.

--- Page 24 ---
A.6
Detailed Analysis of Post-Training Effects on Contextual Generalization
In this section, we provide a detailed analysis of how different post-training data recipes affect contextual general-
ization to long-tailed contexts given atomic reasoning primitives during pre-training.
A.6.1
When Reasoning Primitives are Shared During Pre-Training
Beyond mastering fundamental reasoning skills, an essential dimension of model generalization lies in contextual
generalization—the capacity to transfer learned reasoning behaviors across diverse problem contexts, such as
varying surface narratives or domains. In this section, we investigate whether post-training can incentivize models
to generalize reasoning competence to long-tailed or underrepresented contexts that were scarcely observed during
pre-training.
Task Settting. We study two distinct problem contexts: a frequent, canonical context A and a long-tailed context B,
both sharing the same underlying reasoning priors (logical-arithmetic reasoning in our case, detailed context settings
can be found in Appendix A.9). The pre-training corpus consists of 99.9% context A and only 0.1% context B, both
spanning op=2-20. During post-training, we vary the exposure to context B across 200K samples with different
ratios: 0%, 2%, 10%, 50%, and 100%.
1
2
4
8
16 32 64 128
75
80
85
90
95
context A
1
2
4
8
16 32 64 128
40
60
80
context B
Number of Samples k
pass@128 (%)
context B RL data ratio
Base
0%
2%
10%
50%
100%
Figure 12: pass@k performance on contextual generalization tasks after post-training with varying exposure to
context B. With shared reasoning primitives during pre-training, models exhibit strong transfer to context B even
with limited or no exposure during post-training.
Observation 5
With shared reasoning primitives during pre-training, there is a positive relation between exposure to context B
during post-training and performance on context B. Notably, even without any context B exposure during post-
training, the model still achieves significant transfer, underscoring the role of shared primitives in enabling
contextual generalization.
Takeaway 5
When atomic primitives are shared, post-training can incentivize generalization to long-tailed contexts.
Remarkably, even with a 0% exposure to context B during post-training, the model achieves substantial transfer,
highlighting the critical role of shared reasoning structures during pre-training.

--- Page 25 ---
A.6.2
When Only Atomic Primitives are Exposed During Pre-Training
We next examine contextual generalization when the base model has only been exposed to basic atomic primitives
in the long-tailed context during pre-training.
Task Setting. With the same contextual data distribution as above, we restrict context B data during pre-training to
only atomic operations, while context A spans the full range. The pre-training corpus consists of 99% context A
(op=2-20) and only 1% context B, with context B restricted to atomic operations (op=2). Thus, the model learns
reasoning structures primarily through context A, while having minimal exposure to the surface forms of
context B. During post-training, we perform RL fine-tuning with 200K samples where the ratio of context B data
varies across five regimes: 0%, 1%, 10%, 50%, and 100%. Detailed data recipes can be found in Appendix A.9.
1
2
4
8
16 32 64 128
0
25
50
75
100
performance (%)
pass@k on context A
Base
RL 100% A
RL 99% A / 1% B
RL 90% A / 10% B
RL 50% A / 50% B
RL 100% B
1
2
4
8
16 32 64 128
0
25
50
75
100
pass@k on context B
pass@k
Figure 13: pass@k performance for different contexts with base model limited to basic atoms for context B.
Post-training on context A maintains stable performance, while exposure of 10% context B during RL enables
contextual transfer.
Observation 6
As shown in Figure 13, post-training exclusively on context A or with only extremely sparse exposure to context B
(0–1%) maintains strong performance within context A but yields minimal transfer to the long-tailed context B.
However, once a small amount of context B data is introduced—around 10% of total samples—context B
performance improves dramatically, with pass@128 accuracy increasing by over +76 points. Further increasing
the proportion of context B data (50%, 100%) brings diminishing gains, indicating that RL rapidly establishes
robust cross-context reasoning once minimal supervision is available. Notably, even when post-training uses
100% context B data—entirely distinct from the dominant pre-training context—context A performance remains
stable. This shows that RL enables model to learn transferable reasoning policies that extend across surface
forms while preserving competence in previously mastered contexts.
Takeaway 6
RL enables stable cross-context generalization under extreme imbalance. Even when the base model has only
minimal exposure to long-tailed contexts during pre-training, RL fine-tuning can transfer reasoning competence
across domains by leveraging shared reasoning structures.
A.6.3
Training Dynamics for § A.6.2
We plot the post-training reward dynamics across different data recipes used in § A.6.2 to further understand how
varying exposure to long-tailed contexts during RL affects learning progress.
From Figure 14, we can observe that when the exposure to context B during post-training is extremely limited
(0-1%), the reward plateaus, indicating minimal learning progress. However, with moderate exposure (10-100%),
the reward improves significantly, reflecting effective learning and transfer to the long-tailed context.
A.7
Detailed Analysis of Pre-Training Effects on Extrapolative Generalization
Pre-training defines the atomic reasoning primitives that post-training can later compose and extend. If the base
model already encounters moderately complex problems during pre-training, post-training may push those primitives
toward deeper, compositional reasoning. Otherwise, post-training may lack the scaffolding to explore beyond
its inherited competence. We thus study how varying pre-training difficulty influences subsequent extrapolative
generalization.
Task Setting. We fix the post-training recipe to 200K samples from the op=11-14 range, previously identified
as a edge of competence (see Figure 3). We then vary the proportion of “hard” data (op=7-10) included during

--- Page 26 ---
0.68
0.70
0.72
0.74
0.76
rewards
RL 100% A
0.70
0.72
0.74
0.76
RL 99% A / 1% B
0.62
0.64
0.66
0.68
0.70
0.72
RL 90% A / 10% B
0.4
0.5
0.6
RL 50% A / 50% B
0.2
0.4
0.6
RL 100% B
0
50
100
150
200
Step
260
265
270
275
280
response length
0
50
100
150
200
Step
260
265
270
275
280
0
50
100
150
200
Step
255
260
265
270
275
0
50
100
150
200
Step
230
240
250
260
270
280
0
50
100
150
200
Step
200
220
240
260
280
Figure 14: Reward dynamics across different post-training data recipes. When RL exposure to context B is
extremely limited (0-1%), the reward stagnates. However, with moderate exposure (10-100%), the reward improves
significantly, reflecting effective learning and transfer.
pre-training to assess how exposure to complex primitives affects the base model’s ability to generalize after RL.
(See Appendix A.9 for detailed data recipes.)
0.1%
5%
20%
33.3%
50%
92.5
95.0
97.5
100.0
102.5
105.0
pass@128 (%)
-1.7
-0.5
-1.1
-0.7
-0.4
pass@128 on op=2-10
RL
Base
0.1%
5%
20%
33.3%
50%
20
40
60
80
+29.0
+29.2
+16.6
+45.3
+42.0
pass@128 on op=11-14
0.1%
5%
20%
33.3%
50%
0
10
20
30
+5.9
+10.4
+22.1
+18.0
+25.8
pass@128 on op=15-20
Pre-training hard ratio (%)
Figure 15: pass@128 performance on extrapolative tasks after post-training on op=11-14, under varying levels
of hard-data exposure during pre-training.
Observation 7
As shown in Figure 15, greater exposure to hard problems during pre-training consistently improves both base
and post-trained performance. However, the marginal gain from RL diminishes as pre-training becomes more
comprehensive. When pre-training already covers a substantial fraction of mid-depth tasks, RL adds only modest
improvement. By contrast, when pre-training includes limited but nontrivial exposure to difficult primitives (e.g.,
20% of op=7-10), RL produces the largest relative boost—enhancing pass@128 accuracy on op=15-20
by more than +22 points. This suggests that RL is most effective when the model’s prior competence is
partial—strong enough to support exploration, but incomplete enough to leave room for discovery.
Takeaway 7
Pre-training establishes the foundation, RL extends it. Rich exposure to compositional primitives during
pre-training enables RL to push reasoning depth beyond the pre-training range. Yet the benefits of RL taper off
once those primitives are fully mastered, highlighting the complementary roles of the two stages.

--- Page 27 ---
A.7.1
Training Dynamics for § A.7
We analyze the training dynamics during post-training across different pre-training data recipes.
0.05
0.10
0.15
0.20
rewards
0.1% hard pre-training
0.10
0.15
0.20
0.25
5% hard pre-training
0.25
0.30
0.35
0.40
0.45
20% hard pre-training
0.2
0.3
0.4
33.3% hard pre-training
0.2
0.3
0.4
0.5
50% hard pre-training
0
50
100
150
200
Step
280
300
320
response length
0
50
100
150
200
Step
280
290
300
310
320
330
0
50
100
150
200
Step
300
310
320
330
0
50
100
150
200
Step
280
290
300
310
320
330
0
50
100
150
200
Step
280
290
300
310
320
330
Figure 16: Reward dynamics across different pre-training data recipes. Models with moderate hard-data exposure
(20-50%) during pre-training exhibit significant reward improvements during post-training, indicating effective
learning and extrapolation. In contrast, models with either too little (0%) or too much (100%) hard-data exposure
show limited reward gains, suggesting constrained learning progress.
A.8
Training Dynamics for § 4
In this section, we provide an analysis of the training dynamics for different pre-training data recipes in contextual
generalization in § 3. From Figure 17, we observe that moderate exposure ratio to long-tailed contexts, even with
0.32
0.34
0.36
0.38
0.40
0.42
rewards
0.1%B(op=2) Pre-Training
0.4
0.5
0.6
1%B(op=2) Pre-Training
0.4
0.5
0.6
0.7
10%B(op=2) Pre-Training
0
50
100
150
200
Step
210
215
220
225
230
response length
0
50
100
150
200
Step
230
240
250
260
270
280
0
50
100
150
200
Step
240
250
260
270
280
Figure 17: Reward dynamics across different pre-training data recipes. Models with minimal exposure to long-tailed
contexts exhibit no reward improvement during post-training. While models with moderate to full exposure show
significant reward improvements, indicating effective learning and contextual generalization.
basic primitives during pre-training, is necessary for the model to make significant reward improvements during
post-training.

--- Page 28 ---
A.9
Post-Training and Pre-Training Data Recipe
In this section, we detail the data recipes employed in § 3 § 4, § A.6.1, § A.6.2, and § A.7. Table 2 summarizes the
specific operation count ranges, contextual templates, and training budgets utilized across different experimental
sections.
Pre-training
Post-training (RL)
Section
op(G)
Contexts
Training Budget
op(G)
Contexts
Training Budget
§ 3
20%op=2-4 + 30%op=5-7 + 50%op=8-10
33%A+33%B+33%C
10B tokens
op=8-10
33%A+33%B+33%C
204.8k samples
op=9-12
op=11-14
op=17-20
§ 4
100%op=2-20 A + 0%op=2 B
10B tokens
op=2-20
50% A + 50% B
204.8k samples
99.9%op=2-20 A + 0.1%op=2 B
99%op=2-20 A + 1%op=2 B
90%op=2-20 A + 10%op=2 B
§ A.6.1
op=2-20
99.9%A+0.1%B
10B tokens
op=2-20
100% A
204.8k samples
98%A + 2%B
90%A + 10%B
50%A + 50%B
100%B
§ A.6.2
99%op=2-20 A + 1%op=2 B
10B tokens
op=2-20
100% A
204.8k samples
99%A + 1%B
90%A + 10%B
50%A + 50%B
100%B
§ A.7
99.9% op=2-6 + 0.1% op=8-20
33%A+33%B+33%C
10B tokens
op=11-14
33%A+33%B+33%C
204.8k samples
49.95% op=2-4 + 49.95% op=5-7 + 0.1% op=8-10
47.5% op=2-4 + 47.5% op=5-7 + 5% op=8-10
50% op=2-4 + 30% op=5-7 + 20% op=8-10
20% op=2-4 + 30% op=5-7 + 50% op=8-10
Table 2: Data recipes for pre-/post-training experiments in § 3, § 4, § A.6.1, § A.6.2, and § A.7. op(G) ranges
indicate the operation counts during each training phase. Contexts A, B, C correspond to distinct templates: A =
animals–zoo, B = teachers–school, C = movie-festival. The data recipes for different operation ranges and contexts
are uniformly sampled within the specified proportions. Shaded cells indicate the ablated settings.

--- Page 29 ---
A.10
Mid-/Post-Training Mixing with Different Computation Budget
In this section, we first detail the compute budget formulation for mid-training and RL equivalence, then provide the
exact data recipes for combining mid-training and post-training under different total compute budgets.
A.10.1
Compute Budget of Mid-Training and RL Equivalence
Training Computation. Following the Chinchilla scaling law [Hoffmann et al., 2022], a decoder-only Transformer
with P non-embedding parameters trained on T tokens consumes approximately
Ctrain ≈6P T
flops.
(4)
Thus, a mid-training phase with budget Tmid incurs Cmid = 6P Tmid
flops.
Fine-Grained RL Computation. For on-policy GRPO, computation can be decomposed as:
• Rollout: actor model forward (2P),
• Reference (optional): reference model forward (2P),
• Policy Update: forward (2P) and backward (4P) passes.
Summing these terms yields:
CRL = (8 + 2γ)P N r Ltotal,
(5)
where γ ∈{0, 1} toggles the reference-model pass, N is the number of RL samples, r is the rollout size, and Ltotal
is the total sequence length (including both prompt and completion).
Mid-training Token Equivalence. Normalizing by Equation 4 gives the equivalent mid-training token cost:
TRL = CRL
6P
=

4
3 + γ
3

NrLtotal.
(6)
When γ = 1, we obtain the equivalence used in the main text:
TRL = 5
3NrLtotal.
Budget Allocation and Step Calculation. Given total budget T and RL ratio β,
Tmid = (1 −β) · T,
TRL,eq = β · T.
(7)
The corresponding number of RL samples N(p) and update steps are:
N(β) = 3
5 ·
βT
rLtotal
,
stepsRL(p) = N(β)
B
,
(8)
where r = 6 is the rollout size, Ltotal = 2048 is the total sequence length, B = 1024 is the RL batch size, and T is
the total token budget. The mid-training steps are:
stepsmid(β) =
Tmid
Bmid · Lmid
,
(9)
where Bmid = 512 × 1024 is the mid-training batch size and Lmid = 2048 is the mid-training sequence length.
Task Setting. We use 10B tokens with 20% op=2-4, 30% op=5-7, and 50% op=8-10 for pre-training. To avoid
catastrophic forgetting during mid-training, we use 20% budget for op=2-10 and 80% for op=11-14 during
mid-training. For fair comparison, RL is performed with the same data distribution as mid-training. Table 3 details
the exact step counts for mid-training and RL across varying total token budgets T and mid-training ratios p. We
perform mid-/post-training with Full mid-training, Full RL, Light-RL (β = 0.2), Medium-RL (β = 0.5), and
Heavy-RL (β = 0.8) under different total compute budgets.
Observation 8
As shown in Figure 18, across all compute budgets Light-RL achieves the best OOD-edge pass@1. While
Heavy-RL consistently attains the highest OOD-hard pass@1 performance. For pass@128, when the compute
budget is limited (4.2B tokens), Heavy-RL achieves the best performance in the OOD-hard setting. When the
budget increases (8.4B tokens and above), Full RL attains the highest OOD-hard pass@128 performance.

--- Page 30 ---
50
52
54
 ~1.0B
op=11-14
2
4
6
op=15-20
86
88
90
92
op=11-14
12
15
18
21
24
op=15-20
51
52
53
54
55
 ~2.1B
5
6
7
8
86
88
90
92
94
18
21
24
27
50
52
54
56
 ~4.2B
5
6
7
87
90
93
14
16
18
20
22
54
55
56
57
58
 ~8.4B
6
7
8
87
90
93
96
12
15
18
21
54
56
58
60
 ~16.8B
6
8
10
84
87
90
93
96
14
16
18
20
22
0% 20%
50%
80%100%
RL Ratio 
56
58
60
 ~20.0B
0% 20%
50%
80%100%
RL Ratio 
6
8
10
0% 20%
50%
80%100%
RL Ratio 
84
88
92
96
0% 20%
50%
80%100%
RL Ratio 
16
20
24
28
pass@k performance with different RL ratio by budget
pass@1
pass@128
Figure 18: pass@k performance for different mid-training and RL mixing ratios under varying total compute
budgets.

--- Page 31 ---
Total
Mid-Only
RL-Only
80% Mid / 20% RL
50% Mid / 50% RL
20% Mid / 80% RL
(B)
stepsmid
stepsRL
Samp.(k)
stepsmid
stepsRL
stepsmid
stepsRL
stepsmid
stepsRL
1.05
2,000
50
51.2
1,600
10
1,000
25
400
40
2.10
4,000
100
102.4
3,200
20
2,000
50
800
80
4.20
8,000
200
204.8
6,400
40
4,000
100
1,600
160
8.40
16,000
400
409.6
12,800
80
8,000
200
3,200
320
12.58
24,000
600
614.4
19,200
120
12,000
300
4,800
480
16.78
32,000
800
819.2
25,600
160
16,000
400
6,400
640
20.00
38,147
954
976.6
30,517
191
19,073
477
7,629
763
Table 3: Experimental configurations across varying compute budget scales. We fix the mid-training batch size at
512K tokens. The table maps the total token budget T to the specific step counts required for pure mid-training
(p = 1.0), pure RL (p = 0.0), and hybrid splits.
Takeaway 8
Mid-training and post-training complement each other across varying compute budgets. A combination
of mid-training and RL post-training consistently outperforms either approach individually for pass@1 tasks.
For pass@128, the optimal post-training allocation depends on the available compute budget: with limited
resources, allocating around 80% to RL strikes a balance between stability and exploration, while with more
compute, full RL maximizes extrapolative gains.

