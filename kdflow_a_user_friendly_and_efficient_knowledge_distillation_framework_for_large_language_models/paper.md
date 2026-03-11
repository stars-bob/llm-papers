# KDFlow: A User-Friendly and Efficient Knowledge Distillation Framework for Large Language Models

Source: https://arxiv.org/pdf/2603.01875

Pages: 8

---

--- Page 1 ---
KDFlow: A User-Friendly and Efficient Knowledge Distillation
Framework for Large Language Models
Songming Zhang1,2,3*, Xue Zhang1,2, Tong Zhang3, Bojie Hu3,
Yufeng Chen1,2†, and Jinan Xu1,2
1Key Laboratory of Big Data & Artificial Intelligence in Transportation,
(Beijing Jiaotong University), Ministry of Education
2School of Computer Science and Technology, Beijing Jiaotong University, Beijing, China
3 Tencent Inc, China
{smzhang22,zhang_xue,chenyf,jaxu}@bjtu.edu.cn
Abstract
Knowledge distillation (KD) is an essential
technique to compress large language models
(LLMs) into smaller ones. However, despite
the distinct roles of the student model and the
teacher model in KD, most existing frameworks
still use a homogeneous training backend (e.g.,
FSDP and DeepSpeed) for both models, lead-
ing to suboptimal training efficiency. In this pa-
per, we present a novel framework for LLM dis-
tillation, termed KDFlow, which features a de-
coupled architecture and employs SGLang for
teacher inference. By bridging the training effi-
ciency of FSDP2 and the inference efficiency
of SGLang, KDFlow achieves full utilization of
both advantages in a unified system. Moreover,
instead of transferring full logits across differ-
ent processes, our framework only transmits
the teacher’s hidden states using zero-copy data
transfer and recomputes the logits on the stu-
dent side, effectively balancing the communi-
cation cost and KD performance. Furthermore,
our framework supports both off-policy and on-
policy distillation and incorporates KD algo-
rithms for cross-tokenizer KD through highly
extensible and user-friendly APIs.
Experi-
ments show that KDFlow can achieve 1.44× to
6.36× speedup compared to current KD frame-
works, enabling researchers to rapidly proto-
type and scale LLM distillation with minimal
engineering overhead. Code is available at:
https://github.com/songmzhang/KDFlow
1
Introduction
Large Language Models (LLMs) have demon-
strated remarkable capabilities across a wide range
of fields and tasks. However, the massive number
of parameters in these models presents significant
challenges for deployment, particularly in resource-
constrained environments. Knowledge Distillation
(KD) has emerged as a pivotal technique to miti-
gate this issue, effectively transferring information
* Work was done when Songming was interning at Tencent.
† Yufeng Chen is the corresponding author.
Figure 1: Training time per step and teacher forward
time occupation under different distillation configura-
tions. The teacher’s MoE architecture poses challenges
for FSDP, while being well-supported by SGLang.
from a cumbersome teacher model into a compact
student model (Hinton et al., 2015).
Despite the wide research and application of KD,
the underlying infrastructure for KD of LLMs re-
mains suboptimal. In a typical KD process, the
teacher model and the student model play distinct
roles: the teacher performs only forward passes (in-
ference), while the student requires both forward
and backward passes (training). However, most ex-
isting frameworks, such as TRL (von Werra et al.,
2020) or MS-SWIFT (Zhao et al., 2025), run both
models with a unified training backend. This archi-
tecture creates a structural mismatch: a heavy train-
ing engine designed for gradient computation and
optimizer state management is not suitable for the
teacher model that requires high inference through-
put. Therefore, the overall training throughput of
LLM distillation is often bounded by the inefficient
execution of the teacher model, especially when the
teacher model has an Mixture-of-Experts (MoE)
architecture (see Figure 1).
To address this issue, we present KDFlow, a
novel and high-performance framework designed
specifically for LLM distillation.
The core de-
arXiv:2603.01875v1  [cs.CL]  2 Mar 2026


--- Page 2 ---
sign of KDFlow is the decoupling of backends
for the teacher and the student: we assign the
student model to a dedicated training engine Py-
Torch FSDP2 and the teacher model to the high-
performance inference engine SGLang (Zheng
et al., 2024). By bridging these two disparate sys-
tems, KDFlow combines the best of both worlds:
the flexible gradient updates of training engines
and the extreme inference throughput of inference
engines. In this architecture, a core problem is
how to efficiently transfer full teacher logits from
SGLang processes to FSDP processes. Directly
transferring the original logits is infeasible due to
the huge size1, while only transferring the top-k
logits breaks the mathematical equivalence of the
loss function. KDFlow solves this by collecting the
teacher’s compact hidden states from SGLang and
recomputing the full logit distribution on the stu-
dent side. This design significantly reduces commu-
nication overhead while maintaining mathematical
equivalence to standard KD. Compared to preva-
lent frameworks like TRL and MS-SWIFT, KD-
Flow delivers a 1.44× to 6.36× training speedup,
with the performance gap widening significantly
for Mixture-of-Experts (MoE) teachers.
Despite the decoupled design, KDFlow abstracts
away the distributed communication logic. Users
can easily initiate a KD process with just a few
lines of configuration, seamlessly integrating with
standard Hugging Face model formats. Therefore,
KDFlow incurs low costs for both user deployment
and development.
Overall, the contributions of KDFlow include:
• Efficient Architecture: By assigning the
heavy inference workload to SGLang and opti-
mizing the inter-process communication with
hidden states, KDFlow achieves a 1.44× to
6.36× speedup over existing unified-backend
frameworks (e.g., TRL, MS-SWIFT).
• Comprehensiveness: KDFlow supports off-
policy, on-policy, and cross-tokenizer distilla-
tion. Moreover, it has multiple built-in diver-
gence metrics and algorithms, which make it
an out-of-the-box toolkit for LLM distillation.
• User-Friendly
Design:
KDFlow
is
a
lightweight framework based on FSDP2, and
1For 128 sequences with length 4096, full BF16 logits from
Qwen3 models occupy 128 × 4096 × 151936 × 2bytes ≈
160GB of memory.
decouples the algorithms from the whole dis-
tillation pipeline.
2
Related Work
2.1
Knowledge Distillation for LLMs
Knowledge Distillation (KD) technique was first
proposed by Hinton et al. (2015) to compress large
models into smaller models. In the era of LLMs,
KD is broadly categorized into black-box (Kim
and Rush, 2016) and white-box distillation (Agar-
wal et al., 2024; Gu et al., 2023). Among these,
white-box KD usually aligns the output distribu-
tions of the student model with those of the teacher
model through divergence metrics. This approach
provides richer supervisory information and thus
frequently yields superior performance. Further-
more, the paradigm has expanded from traditional
off-policy distillation, where the models learn from
static datasets, to on-policy distillation, where the
student learns from data generated by the student
itself (Gu et al., 2023; Agarwal et al., 2024; Xiao
et al., 2026). Recent advancements also explore
cross-tokenizer KD, addressing the failure of white-
box KD when the teacher and student have different
vocabularies (Wan et al., 2024; Boizard et al., 2024;
Zhang et al., 2024, 2025; Cui et al., 2025; Chen
et al., 2025; Minixhofer et al., 2025). Despite the
algorithmic-level breakthroughs, a flexible and ef-
ficient framework for KD research is still lacking,
which is exactly the aim of this work.
2.2
Existing Frameworks
To facilitate LLM training and distillation, several
open-source frameworks have been developed. Ex-
isting libraries like Hugging Face TRL (von Werra
et al., 2020), EasyDistill (Wang et al., 2025a),
OpenRLHF (Hu et al., 2024), and MS-SWIFT
(Zhao et al., 2025) provide accessible KD function-
alities but inherently rely on homogeneous training
engines for both the teacher and student models,
leading to severe underutilization of hardware dur-
ing the teacher’s forward pass. Additionally, sev-
eral reinforcement learning frameworks, such as
Slime (Zhu et al., 2025) and verl (Sheng et al.,
2025), have pioneered the use of decoupled archi-
tectures by integrating high-throughput inference
engines like vLLM/SGLang for teacher forward
computation. However, these frameworks are not
specialized for KD and only support on-policy dis-
tillation with incomplete logit information, over-
looking wider KD scenarios. By contrast, KDFlow


--- Page 3 ---
Features
TRL
MS-SWIFT
EasyDistill
OpenRLHF
ROLL
Slime
verl
KDFlow (Ours)
Designed for KD
!
!
Decoupled Backends
!
!
!
Off-policy KD
!
!
!
!
!
!
On-policy KD
!
!
!
!
!
Self Distillation
!
Cross-Tokenizer KD
!
!
Logits/Logprobs
full
full
full
full
top-k
single
top-k
full
Table 1: Comparisons between KDFlow and existing frameworks.
is specifically designed as a KD framework with
high efficiency and full KD scenarios. The detailed
comparison between KDFlow and existing frame-
works is presented in Table 1.
3
The Design of KDFlow
The overall framework of KDFlow is presented in
Figure 2. To address the inefficiency of using a
single homogeneous engine for both inference and
training, we adopt a top-down decoupled design. In
this section, we introduce the system architecture,
the core communication mechanism, the distilla-
tion workflows, and the algorithm abstractions.
3.1
System Architecture
The KD process typically involves multiple stages:
teacher inference, student training, and student roll-
out, which requires to operate different backends
during training. Therefore, KDFlow is built upon
Ray to manage distributed processes efficiently. As
shown in the middle part of Figure 2, the architec-
ture consists of a single controller (Trainer) and
three functionally independent actor groups:
• Trainer (Single Controller): Trainer is the
central coordinator that manages the dataset,
controls the training loop, and organizes the
data flow among different actor groups. KD-
Flow supports both OffPolicyKDTrainer and
OnPolicyKDTrainer.
• RolloutActorGroup: RolloutActorGroup is
used for the rollout process of the student
model during on-policy distillation. Follow-
ing frontier RL frameworks like Slime (Zhu
et al., 2025), it uses SGLang Router to con-
nect multiple SGLang http servers for load
balance inference. Moreover, KDFlow is de-
signed as the colocate mode and update model
weights in SGLang via CUDA Interprocess
Communication (IPC).
• TeacherActorGroup: TeacherActorGroup
manages multiple TeacherRayActor that exe-
cutes the forward pass of the teacher. Specif-
ically, we initialize an SGLang engine2 in
each TeacherRayActor to obain the teacher’s
hidden states.
Therefore, we can utilize
SGLang’s high-throughput and flexible paral-
lel strategies in inference for teacher models.
• StudentActorGroup: StudentActorGroup is
deployed with PyTorch FSDP2. It handles
the standard training processes for the stu-
dent model, including forward pass, back-
ward passes, and optimizer state management.
Then it returns training status (e.g., the loss
values) for logging.
This decoupled architecture ensures that the
inference-heavy teacher model and the training-
heavy student model run on their respectively opti-
mized backends, significantly improving hardware
utilization.
3.2
Efficient Communication via Hidden
States
In the decoupled architecture, the teacher model
and the student model typically run in separate pro-
cesses. A critical bottleneck in this setup is how to
efficiently transfer the teacher’s knowledge to the
student. While directly transmitting the full logit
distributions can be feasible when both models re-
side in the same process, this approach becomes
prohibitive in the decoupled architecture due to the
excessively large data volume. Conversely, only
transferring the top-k logits saves bandwidth but in-
evitably undermines the mathematical equivalence
of distillation and degrades the final performance.
To address this issue, KDFlow adopts a mechanism
2The SGLang engine is more compatible with Numpy
ndarray objects than the SGLang http server.


--- Page 4 ---
Figure 2: The overview of KDFlow. The whole framework is built based on Ray (Moritz et al., 2018) and decouples
the distillation pipeline by allocating the teacher model to SGLang and the student model to FSDP2. Solid and
dashed arrows illustrate the data flow for off-policy and on-policy distillation, respectively. Notably, KDFlow
transfers compact hidden states from the teacher rather than full logits to reduce communication overhead.
based on hidden-state transfer and logit recompu-
tation, as shown in Figure 3. Instead of sending
the full logits, the TeacherActorGroup only out-
puts the final hidden states of the teacher model.
Since the dimension of hidden states (e.g., 4096)
is far smaller than that of the logit distributions
(e.g., 151936), the corresponding communication
overhead becomes affordable and practical. Fur-
thermore, we leverage shared memory and the Ray
Shared Objective mechanism to enable zero-copy
data transfer across processes. When receiving the
teacher’s hidden states, each StudentRayActor lo-
cally recomputes the full logit distributions using
the teacher’s language model head to locally. This
design drastically reduces communication volume
while preserving the mathematical equivalence to
standard logit-based KD.
3.3
Distillation Workflows
Governed by the Single Controller, KDFlow seam-
lessly supports two main distillation workflows,
illustrating the flexible data routing among the de-
coupled actors:
Off-Policy Distillation (Solid lines in Figure
2): The student learns from a static dataset. The
Trainer sends the prompts and the corresponding
responses directly to the TeacherActorGroup to
obtain the teacher’s hidden states. These hidden
states, along with the inputs, are then passed to the
StudentActorGroup to compute the distillation loss
and update the student’s weights.
On-Policy Distillation (Dashed lines in Figure
2): The student learns from data generated by it-
self. First, the Trainer sends prompts to the Roll-
outActorGroup to generate responses. Next, these
prompt-response pairs are sent to the TeacherActor-
Group to obtain the teacher’s hidden states. Then,
the data and hidden states flow into the StudentAc-
torGroup for gradient updates. Finally, the updated
weights of the student model are synchronized back
to the RolloutActorGroup to ensure the next gener-
ation step uses the latest policy.
3.4
Comprehensive Abstractions and
Algorithms
To provide a user-friendly and out-of-the-box
toolkit, KDFlow strictly separates the underlying
system pipeline from the distillation algorithms. As


--- Page 5 ---
Figure 3: Comparisons between different decoupled
distillation approaches.
Framework
AlpacaEval 2.0
LC-Win Rate (%)
Win Rate (%)
Qwen3-1.7B (w/o KD)
26.09
21.99
MS-SWIFT
28.40
27.86
FSDP Student + Teacher
28.18
28.20
KDFlow
28.23
28.32
Table 2: Student model performance on AlpacaEval 2.0
after distillation with different frameworks.
shown at the bottom of Figure 2, KDFlow provides
built-in support for various KD algorithms and di-
vergence metrics, including Forward KL (FKL),
Reverse KL (RKL), Jensen-Shannon Divergence
(JSD), and Total Variation Distance (TVD). Users
can also easily implement their custom distillation
losses or algorithms with minimal code, eliminat-
ing the need of understand the complex distributed
communication logic. Furthermore, KDFlow na-
tively supports cross-tokenizer distillation. When
the teacher and student models have different vo-
cabularies, directly aligning their full logit distri-
butions is impossible. Therefore, KDFlow imple-
ments the DSKDv2 (Zhang et al., 2025) algorithm
for cross-tokenzier knowledge distillation.
With this decoupled design and efficient com-
munication mechanism, KDFlow significantly im-
proves the overall training throughput, which we
will demonstrate in the following experiments.
Figure 4: Loss curves of KDFlow and the pure FSDP
implementation when distilling Qwen3-30B-A3B to
Qwen3-4B.
4
Experiments
In this section, we comprehensively evaluate the
KDFlow framework from the following three per-
spectives: (1) Loss Correctness: We validate the
loss correctness of KDFlow by recording and ana-
lyzing its loss curves against the one from the stan-
dard FSDP baseline. (2) KD Performance: We
compare the knowledge distillation performance
of KDFlow with that of baseline frameworks on
a representative downstream task. (3) Training
Efficiency: We test the training speed of KDFlow
and existing frameworks across multiple teacher-
student setups to show the efficiency of KDFlow.
4.1
Experimental Setup
Models and Datasets.
We evaluate the correct-
ness and performance of KDFlow on instruction-
following tasks. Specifically, we randomly sample
100k prompt data from LMSys-Chat-1M (Zheng
et al., 2023) and generate responses with Qwen3-
14B. Then, we choose the Qwen3 model family
(Yang et al., 2025) for distillation since it covers
multiple model sizes and architectures. We use
Qwen3-14B, Qwen3-32B and Qwen3-30B-A3B
as the teacher models and Qwen3-4B and Qwen3-
1.7B as the student models. We respectively report
the training loss curves, model performance on Al-
pacaEval 2.0 (Dubois et al., 2024), and training
speed in the following parts.
Baselines and Hardware.
We compare KDFlow
against TRL (von Werra et al., 2020), ROLL (Wang
et al., 2025b), and MS-SWIFT (Zhao et al., 2025),
three representative frameworks for LLM distilla-
tion. All experiments are conducted on a single
server equipped with 8 NVIDIA H20 GPUs.


--- Page 6 ---
Frameworks
Training
Backend
Student: Qwen3-4B
Student: Qwen3-1.7B
Qwen3-14B
Qwen3-32B
Qwen3-30B-A3B
Qwen3-14B
Qwen3-32B
Qwen3-30B-A3B
TRL
ZeRO-3
21.3s/it
31.5s/it
-
13.3s/it
23.4s/it
-
MS-SWIFT
ZeRO-3
16.6s/it
24.8s/it
43.2s/it
11.5s/it
20.1s/it
36.9s/it
ROLL
FSDP2
38.4s/it
56.9s/it
67.9s/it
26.8s/it
45.6s/it
53.8s/it
KDFlow (BF16 Teacher)
FSDP2
12.3s/it
15.7s/it
11.3s/it
7.6s/it
10.9s/it
5.9s/it
KDFlow (FP8 Teacher)
FSDP2
11.5s/it
13.5s/it
11.1s/it
6.7s/it
8.7s/it
5.8s/it
Speedup
-
1.44×
1.84×
3.89×
1.72×
2.31×
6.36×
Table 3: Training efficiency comparison (seconds per iteration) across different distillation frameworks. Speedup
is calculated using KDFlow (FP8) against the best-performing baseline (i.e., MS-SWIFT). All frameworks use
identical training settings: global batch size=128, gradient accumulation=8, and max length=4096.
4.2
Loss Curve Validation
The core technical innovation of KDFlow is to
leverage SGLang as the teacher backend to col-
lect the teacher’s hidden states and recompute full
logits on the student side. Recent literature on
reinforcement learning (RL) has pointed out that
current inference engines like SGLang may intro-
duce numerical instability or precision loss during
inference, especially for MoE models, which de-
grades the stability and performance of RL train-
ing (Yao et al., 2025; Qi et al., 2025; Ma et al.,
2025; Zheng et al., 2025). We also verify whether
this issue has significant influence on distillation.
Specifically, we compare the training loss curves
of KDFlow against a standard KD implementa-
tion (where both teacher and student are placed
within the same FSDP process, computing logits
locally). As shown in Figure 4, the KL loss curves
of KDFlow are well aligns with the baseline frame-
work throughout the entire training process. More-
over, the loss curves are almost overlapped for FP8
and BF16 teacher inference, which suggests FP8
teacher inference a more efficient solution for LLM
distillation.
4.3
Downstream Task Performance
Beyond training loss, we evaluate the effectiveness
of the student models trained via KDFlow on down-
stream tasks to ensure our decoupled architecture
does not compromise distillation performance. We
perform off-policy distillation from Qwen3-30B-
A3B to Qwen3-1.7B using the Forward KL (FKL)
divergence across all frameworks. Table 2 presents
the performance evaluation on the AlpacaEval 2.0
benchmark. The student model distilled via KD-
Flow achieves an LC-Win Rate of 28.23% and a
Win Rate of 28.32%, demonstrating significant im-
provement over the un-distilled Qwen3-1.7B base-
line. Moreover, when compared to the students
distilled using the MS-SWIFT and pure FSDP base-
lines, the performance differences remain within a
negligible range. These results prove that KDFlow
safely optimizes the system’s execution pipeline
and reduces communication overhead while strictly
preserving the distillation quality. KDFlow acts as
a transparent, high-performance infrastructure that
faithfully executes KD algorithms without sacrific-
ing downstream performance.
4.4
Training Speed and Efficiency
In this subsection, we systematically compare the
training speed of KDFlow with existing popular
frameworks for LLM distillation.
Specifically,
we measure the average training time per step
across different teacher-student setups. As illus-
trated in Table 3, KDFlow significantly accelerates
the KD process and emerges as the fastest frame-
work among all FSDP2 and DeepSpeed ZeRO3-
based candidates, achieving a speedup ranging
from 1.44× to 6.36×. Notably, for the MoE-based
teacher model (Qwen3-30B-A3B), the baseline
frameworks suffer from severe execution bottle-
necks, with training times soaring to over 36s/it.
This is primarily due to the inefficient handling
of sparse routing and expert management in stan-
dard training engines like ZeRO-3 or FSDP2. In
contrast, KDFlow serves the MoE-based teacher
model with SGLang, fully exploiting its highly-
optimized kernels and flexible parallel strategies on
model inference. This allows KDFlow to maintain
high throughput regardless of teacher complexity,
reducing the distillation time for the Qwen3-30B-
A3B teacher to 5.8s/it. Furthermore, our hidden-
state communication strategy successfully prevents
memory bandwidth from being a bottleneck in Ray-
based training, ensuring that the student’s training
engine is continuously fed without idling, even
when processing massive logit distributions from
large-scale teachers.


--- Page 7 ---
5
Conclusion
In this paper, we introduce KDFlow, an effi-
cient and user-friendly knowledge distillation (KD)
framework for LLMs.
To overcome the ineffi-
ciency of existing frameworks, KDFlow decouples
the architecture by deploying the teacher model
on a high-throughput inference engine (SGLang)
and the student on a dedicated training backend
(FSDP2). To eliminate the massive network com-
munication bottleneck caused by this decoupling,
we propose a hidden-state transfer and logit re-
computation mechanism, ensuring strict mathemat-
ical equivalence to standard full-logit KD. Experi-
ments demonstrate that KDFlow achieves a 1.44×
to 6.36× training speedup over state-of-the-art base-
lines without compromising downstream task per-
formance. By natively supporting off-policy, on-
policy, and cross-tokenizer distillation, KDFlow
serves as a comprehensive infrastructure to accel-
erate future LLM compression and post-training
research.
Limitations
While KDFlow significantly improves the effi-
ciency and flexibility of LLM distillation, it has cer-
tain limitations. First, the current student training
backend of KDFlow is built entirely upon PyTorch
FSDP2, which still struggles to match the training
efficiency of Megatron-LM (Shoeybi et al., 2019)
that supports complex 3D parallelism. Second, as
a research-oriented framework, KDFlow currently
lacks some industrial-grade optimizations, such as
asynchronous training, which are crucial for train-
ing on clusters with thousands of GPUs. However,
KDFlow explicitly prioritizes the needs of the re-
search community: user-friendliness, high flexi-
bility, and rapid prototyping. By abstracting away
complex distributed communication logic, KDFlow
allows researchers to easily implement and test
novel off-policy, on-policy, or cross-tokenizer dis-
tillation algorithms with minimal engineering over-
head. Integrating Megatron-LM and further opti-
mizations remain important directions for our fu-
ture work.
References
Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Pi-
otr Stanczyk, Sabela Ramos Garea, Matthieu Geist,
and Olivier Bachem. 2024. On-policy distillation
of language models: Learning from self-generated
mistakes. In The twelfth international conference on
learning representations.
Nicolas Boizard, Kevin El Haddad, Céline Hudelot,
and Pierre Colombo. 2024. Towards cross-tokenizer
distillation: the universal logit distillation loss for
llms. arXiv preprint arXiv:2402.12030.
Yijie Chen, Yijin Liu, Fandong Meng, Yufeng Chen,
Jinan Xu, and Jie Zhou. 2025.
Enhancing cross-
tokenizer knowledge distillation with contextual dy-
namical mapping. In Findings of the Association for
Computational Linguistics: ACL 2025, pages 8005–
8018.
Xiao Cui, Mo Zhu, Yulei Qin, Liang Xie, Wengang
Zhou, and Houqiang Li. 2025. Multi-level optimal
transport for universal cross-tokenizer knowledge
distillation on language models. In Proceedings of
the AAAI Conference on Artificial Intelligence, vol-
ume 39, pages 23724–23732.
Yann Dubois, Balázs Galambosi, Percy Liang, and Tat-
sunori B Hashimoto. 2024. Length-controlled al-
pacaeval: A simple way to debias automatic evalua-
tors. arXiv preprint arXiv:2404.04475.
Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. 2023.
Minillm: Knowledge distillation of large language
models. arXiv preprint arXiv:2306.08543.
Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. 2015.
Distilling the knowledge in a neural network. arXiv
preprint arXiv:1503.02531.
Jian Hu, Xibin Wu, Zilin Zhu, Weixun Wang, De-
hao Zhang, Yu Cao, and 1 others. 2024. Openrlhf:
An easy-to-use, scalable and high-performance rlhf
framework. arXiv preprint arXiv:2405.11143, 6.
Yoon Kim and Alexander M Rush. 2016. Sequence-
level knowledge distillation. In Proceedings of the
2016 conference on empirical methods in natural
language processing, pages 1317–1327.
Wenhan Ma, Hailin Zhang, Liang Zhao, Yifan Song,
Yudong Wang, Zhifang Sui, and Fuli Luo. 2025.
Stabilizing moe reinforcement learning by align-
ing training and inference routers. arXiv preprint
arXiv:2510.11370.
Benjamin Minixhofer, Ivan Vuli´c, and Edoardo M Ponti.
2025. Cross-tokenizer distillation via approximate
likelihood matching.
Philipp Moritz, Robert Nishihara, Stephanie Wang,
Alexey Tumanov, Richard Liaw, Eric Liang, Melih
Elibol, Zongheng Yang, William Paul, Michael I
Jordan, and 1 others. 2018.
Ray: A distributed
framework for emerging {AI} applications. In 13th
USENIX symposium on operating systems design and
implementation (OSDI 18), pages 561–577.
Penghui Qi, Zichen Liu, Xiangxin Zhou, Tianyu Pang,
Chao Du, Wee Sun Lee, and Min Lin. 2025. Defeat-
ing the training-inference mismatch via fp16. arXiv
preprint arXiv:2510.26788.


--- Page 8 ---
Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin
Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin
Lin, and Chuan Wu. 2025. Hybridflow: A flexible
and efficient rlhf framework. In Proceedings of the
Twentieth European Conference on Computer Sys-
tems, pages 1279–1297.
Mohammad Shoeybi, Mostofa Patwary, Raul Puri,
Patrick LeGresley, Jared Casper, and Bryan Catan-
zaro. 2019.
Megatron-lm: Training multi-billion
parameter language models using model parallelism.
arXiv preprint arXiv:1909.08053.
Leandro von Werra, Younes Belkada, Lewis Tunstall,
Edward Beeching, Tristan Thrush, Nathan Lambert,
Shengyi Huang, Kashif Rasul, and Quentin Gal-
louédec. 2020. TRL: Transformers Reinforcement
Learning.
Fanqi Wan, Xinting Huang, Deng Cai, Xiaojun Quan,
Wei Bi, and Shuming Shi. 2024.
Knowledge fu-
sion of large language models.
arXiv preprint
arXiv:2401.10491.
Chengyu Wang, Junbing Yan, Wenrui Cai, Yuanhao Yue,
and Jun Huang. 2025a. Easydistill: A comprehensive
toolkit for effective knowledge distillation of large
language models. In Proceedings of the 2025 Con-
ference on Empirical Methods in Natural Language
Processing: System Demonstrations, pages 787–795.
Weixun Wang, Shaopan Xiong, Gengru Chen, Wei Gao,
Sheng Guo, Yancheng He, Ju Huang, Jiaheng Liu,
Zhendong Li, Xiaoyang Li, and 1 others. 2025b.
Reinforcement learning optimization for large-scale
learning: An efficient and user-friendly scaling li-
brary. arXiv preprint arXiv:2506.06122.
Bangjun Xiao, Bingquan Xia, Bo Yang, Bofei Gao,
Bowen Shen, Chen Zhang, Chenhong He, Chiheng
Lou, Fuli Luo, Gang Wang, and 1 others. 2026.
Mimo-v2-flash technical report.
arXiv preprint
arXiv:2601.02780.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui,
Bo Zheng,
Bowen Yu,
Chang
Gao, Chengen Huang, Chenxu Lv, and 1 others.
2025.
Qwen3 technical report.
arXiv preprint
arXiv:2505.09388.
Feng Yao, Liyuan Liu, Dinghuai Zhang, Chengyu Dong,
Jingbo Shang, and Jianfeng Gao. 2025. Your efficient
rl framework secretly brings you off-policy rl train-
ing.
Songming Zhang, Xue Zhang, Zengkui Sun, Yufeng
Chen, and Jinan Xu. 2024. Dual-space knowledge
distillation for large language models. In Proceed-
ings of the 2024 Conference on Empirical Methods in
Natural Language Processing, pages 18164–18181.
Xue Zhang, Songming Zhang, Yunlong Liang, Fandong
Meng, Yufeng Chen, Jinan Xu, and Jie Zhou. 2025.
A dual-space framework for general knowledge dis-
tillation of large language models. arXiv preprint
arXiv:2504.11426.
Yuze Zhao, Jintao Huang, Jinghan Hu, Xingjun Wang,
Yunlin Mao, Daoze Zhang, Zeyinzi Jiang, Zhikai Wu,
Baole Ai, Ang Wang, and 1 others. 2025. Swift:
a scalable lightweight infrastructure for fine-tuning.
In Proceedings of the AAAI Conference on Artificial
Intelligence, volume 39, pages 29733–29735.
Chujie Zheng, Kai Dang, Bowen Yu, Mingze Li,
Huiqiang Jiang, Junrong Lin, Yuqiong Liu, Hao Lin,
Chencan Wu, Feng Hu, and 1 others. 2025. Stabiliz-
ing reinforcement learning with llms: Formulation
and practices. arXiv preprint arXiv:2512.01374.
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Tianle
Li, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang,
Zhuohan Li, Zi Lin, Eric. P Xing, Joseph E. Gonza-
lez, Ion Stoica, and Hao Zhang. 2023. Lmsys-chat-
1m: A large-scale real-world llm conversation dataset.
Preprint, arXiv:2309.11998.
Lianmin Zheng,
Liangsheng Yin,
Zhiqiang Xie,
Chuyue Livia Sun, Jeff Huang, Cody Hao Yu, Shiyi
Cao, Christos Kozyrakis, Ion Stoica, Joseph E Gonza-
lez, and 1 others. 2024. Sglang: Efficient execution
of structured language model programs. Advances
in neural information processing systems, 37:62557–
62583.
Zilin Zhu, Chengxing Xie, Xin Lv, and slime Contrib-
utors. 2025. slime: An llm post-training framework
for rl scaling. https://github.com/THUDM/slime.
GitHub repository. Corresponding author: Xin Lv.

