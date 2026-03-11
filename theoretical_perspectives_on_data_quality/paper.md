--- Page 1 ---
Theoretical Perspectives on Data Quality and Synergistic Effects in
Pre- and Post-Training Reasoning Models
Adel Javanmard1,3, Baharan Mirzasoleiman2,3, and Vahab Mirrokni3
1University of Southern California
2University of California Los Angeles
3Google Research
Abstract
Large Language Models (LLMs) are pretrained on massive datasets and later instruction-tuned via
supervised fine-tuning (SFT) or reinforcement learning (RL). Best practices emphasize large, diverse
pretraining data, whereas post-training operates differently: SFT relies on smaller, high-quality datasets,
while RL benefits more from scale, with larger amounts of feedback often outweighing label quality. Yet
it remains unclear why pretraining and RL require large datasets, why SFT excels on smaller ones, and
what defines high-quality SFT data. In this work, we theoretically analyze transformers trained on an
in-context weight prediction task for linear regression. Our analysis reveals several key findings: (i)
balanced pretraining data can induce latent capabilities later activated during post-training, and (ii)
SFT learns best from a small set of examples challenging for the pretrained model, while excessively large
SFT datasets may dilute informative pretraining signals. In contrast, RL is most effective on large-scale
data that is not overly difficult for the pretrained model. We validate these theoretical insights with
experiments on large nonlinear transformer architectures.
1
Introduction
Pretraining on massive language datasets, followed by post-training, is essential for unlocking and shaping
the capabilities of large language models (LLMs). While pretraining endows models with broad linguistic
knowledge and general world understanding, post-training transforms these latent capabilities into usable skills
that can be reliably elicited through instructions. This transformation is typically achieved through either
supervised fine-tuning (SFT), which trains models to imitate high-quality demonstrations, or reinforcement
learning (RL), which optimizes model behavior using scalar feedback to refine global properties such as
reasoning quality and preference alignment. Despite their central role, the interaction between pretraining
data and post-training data—and how this interaction determines the resulting model capabilities—remains
poorly understood.
In practice, pretraining commonly relies on massive and diverse data mixtures, whereas post-training
follows a variety of recipes.
For example, OpenAI o1 (OpenAI, 2024) and DeepSeek R1 (Guo et al.,
2025) achieve state-of-the-art reasoning performance through RL applied to large-scale datasets, while s1
(Muennighoff et al., 2025) demonstrates comparable math reasoning performance using SFT on a small,
manually curated set of hard and diverse examples. More recently, Llama 4 (Meta, 2025) adopts iterative
rounds of SFT and RL on progressively harder data. Yet, which characteristics of pretraining data unlock
superior post-training performance, and what requirements on the quality and scale of post-training data are
needed to bring a pretrained model to optimal performance, have remained unclear.
In this work, we answer the above questions by studying an in-context weight prediction task for linear
regression, where the goal is to predict the linear weight vector from the sequence of input prompts. This
framework has been used previously for analyzing the mechanism underlying training CoT (Huang et al.,
1
arXiv:2603.01293v1  [cs.LG]  1 Mar 2026

--- Page 2 ---
2025b; Javanmard et al., 2025). In this work, we propose a novel pipeline where during pretraining, the model
performs direct in-context-learning and outputs its prediction of the weight vector. During post-training, the
transformer performs CoT with SFT or RL and generates multiple intermediate steps before arriving at its
final prediction of the weight vector. We test the model on a combination of pretraining and post-training
tasks.
While our theoretical setup captures the key distinction between outcome supervision (RL, rewarding
final answers) and process supervision (SFT, supervising intermediate steps), it significantly abstracts from
standard RL algorithms that involve sampling, advantage estimation, and policy gradients. Here, we model
RL as outcome-supervised regression on the transformer’s in-context prediction task. This simplification
enables clean theoretical analysis but limits direct applicability to full RLHF implementations in LLMs.
Our analysis shed light on several questions:
(i) What characteristics of pretraining data enable models to develop latent capabilities that can be
effectively unlocked during post-training?
(ii) Given a pretrained model, what properties define effective SFT data that promote adaptation to new
skills, while minimizing interference with capabilities acquired during pretraining?
(iii) Given a pretrained model, what properties of RL data are most critical? How does the RL optimization
landscape differ from that of SFT, and when can RL achieve outcomes comparable to SFT?
Our analysis helps to rigorously understand several empirically observed phenomena reported in the
literature. Specifically, for our in-context setting, it shows that (i) effective pretraining data contains a
balanced mixture of data from all categories. Such data can induce latent capabilities that are activated
during post-training. (ii) Post-training with SFT benefits the most from a small set of challenging examples
for the pretrained model, and larger SFT data can harm the performance. (iii) RL requires large-scale data
that is informative but not overly difficult for the pretrained model.
We confirm our findings with experiments on an in-context weight prediction task for linear regression on
transformer with a single linear self-attention (LSA), as well as large, nonlinear transformer architectures,
namely GPT2 (Radford et al., 2019).
2
Related Work
Recent work has highlighted several phenomena relevant to our study.
Pretraining. For pretraining LLMs, common practice is to use a large mixture of language data. Recent
studies mostly focused on data filtering (Li et al., 2024), data selection (Nguyen et al., 2024; Yang et al., 2024),
and mixture reweighting (Xie et al., 2023). Empirically, high-quality pretraining data should be large and
diverse. Such high-quality pretraining data can induce latent capabilities that are not necessarily observed
after pretraining but are activated during post-training (Akter et al., 2025).
Post-training. For post-training, recent studies mostly focused on comparing post-training with SFT
and RL (Aminian et al., 2025; Xiong et al.; Zhao et al., 2025). Theoretically, SFT is mode covering: by
minimizing forward KL to demonstration data, it encourages the model to assign probability mass to all
plausible responses. In contrast, reinforcement learning (RL) is mode seeking: by optimizing reward (typically
under a KL constraint), it concentrates probability on high-reward responses and suppresses lower-ranked
alternatives. As a result, SFT defines the space of acceptable behaviors, while RL selects and amplifies the
most preferred ones within that space. Empirically, SFT data should be small and high-quality, i.e. hard and
diverse (Guha et al., 2025; Huang et al., 2025b; Muennighoff et al., 2025), and larger SFT data washes away
benefits of high-quality pretraining data (Akter et al., 2025). In contrast, RL benefits from larger data that
is still challenging but not overly difficult for the pretrained model (Meta, 2025; Yue et al., 2025; Zeng et al.,
2025).
Nevertheless, the reasons why certain characteristics of pretraining data unlock superior post-training
performance, why SFT benefits from a small set of hard and diverse examples while larger datasets can
degrade its effectiveness, and why data scale matters more than apparent quality in RL have remained unclear.
2

--- Page 3 ---
Our theoretical framework demystifies these observations, bridging the gap between empirical results and a
principled understanding of data dynamics.
3
Problem Setup
We focus on in-context learning (ICL) setting, where a model is presented with a context dataset D =
{(xi,yi)}n
i=1 and each (xi,yi) pair is sampled independently from some underlying distribution P. Here,
the input vectors {xi}n
i=1 belong to Rd, and the corresponding labels {yi}n
i=1 may be real numbers (for
regression tasks) or binary values such as {0,1} (for classification tasks).
The model is then given a
new test input xn+1 ∼Px and is tasked to predict its associated label or corresponding in-context weight
predictor. In other words, in-context learning operates on sequences, called prompts, of input-output pairs
(x1,y1,...,xn,yn,xn+1) and each prompt may have its own distribution.
Linear Self Attention (lSA) Let Z be an embedding formed from the prompt (We will discuss the specific
construction later). The softmax self-attention module takes as input an embedding matrix and outputs a
matrix of the same size,
fAttn(Z;WK,WQ,WV ,WP )
= Z + WP WV Z ⋅softmax((WKZ)⊺WQZ
λ
)
where softmax is applied column-wise. In Linear-Self-Attention (LSA) the softmax nonlinearity is removed.
By defining W ∶= W ⊺
KWQ, V = WP WV and θ = (W,V ) we arrive at:
fLSA(Z;θ) = Z + V Z ⋅Z⊺WZ
λ
(3.1)
We will focus on in-context linear predictors. Each prompt is of the form Pτ = (xτ,1,yτ,1,...,xτ,n,yτ,n,xτ,n+1),
with yτ,i = ⟨wτ,xτ,i⟩, where wτ ∼N(0,Id).
Supervised Fine-Tuning and Outcome Supervision. We begin by describing outcome supervision
(OS) training with k steps of chain-of-thought reasoning. As noted in the introduction, this formulation
simplifies standard RL—which involves sampling, advantage estimation, and policy gradients—by modeling
it as outcome-supervised regression that rewards final answers, while still capturing the core distinction from
process-supervised SFT.
Suppose we are given a prompt Pτ = (xτ,1,yτ,1,...,xτ,n,yτ,n). We construct the embedding
ˆZτ,0 =
⎡⎢⎢⎢⎢⎢⎢⎢⎣
xτ,1
...
xτ,n
0
yτ,1
...
yτ,n
0
0
...
0
wτ,0
0
...
0
1
⎤⎥⎥⎥⎥⎥⎥⎥⎦
,
(3.2)
and iteratively define ˆZτ,i+1 = [ ˆZτ,i,fLSA( ˆ
Zτ,i)[∶,−1]]. We initialize wτ,0 = 0d×1 and set ˆwτ,i+1 ∶= fLSA( ˆ
Zτ,i)[d+2∶2d+1,−1].
This yields
ˆZτ,i =
⎡⎢⎢⎢⎢⎢⎢⎢⎣
xτ,1
...
xτ,n
0
∗
...
∗
yτ,1
...
yτ,n
0
∗
...
∗
0
...
0
wτ,0
ˆwτ,1
...
ˆwτ,i
0
...
0
1
1
...
1
⎤⎥⎥⎥⎥⎥⎥⎥⎦
,
(3.3)
Let w∗
τ be the ground-truth weight for prompt Pτ, for τ ∈[B]. The outcome supervision (OS) loss is
LOS(V,W) = 1
2B
B
∑
τ=1
∥ˆwτ,k −w∗
τ∥2
ℓ2 ,
(3.4)
3

--- Page 4 ---
i.e., OS penalizes only the final step of the k-step reasoning process.
For Supervised fine-tuning (SFT), we use ground-truth chain-of-thought (CoT) sequences
Zi,τ =
⎡⎢⎢⎢⎢⎢⎢⎢⎣
x1
...
xn
0
∗
...
∗
y1
...
yn
0
∗
...
∗
0
...
0
w0,τ
w1,τ
...
wi,τ
0
...
0
1
1
...
1
⎤⎥⎥⎥⎥⎥⎥⎥⎦
,
(3.5)
where wi,τ = (1 −(1 −η)i)w∗
τ with w0,τ = 0 provides exponentially converging intermediate targets, with an
arbitrary but fixed rate η. The model is trained to predict the next token Zi+1,τ[∶,−1] ∶= (0d,0,wi+1,τ,1)
given Zi,τ. Over B training prompts, the SFT loss is
LSFT(V,W) ∶=
1
2B
B
∑
τ=1
k
∑
i=0
∥fLSA(Zi,τ)[∶,−1] −(0,0,wi+1,τ,1)∥
2
ℓ2 .
Pipeline: Pre-training, Post-training, Post-testing. Our pipeline has three stages distinguished by
data covariances: pre-training on Σ0, post-testing on Σ = Σ0 + ∆(low-rank ∆), and post-training on a chosen
intermediate distribution (discussed later for optimal post-test performance). Inputs x ∈Rd are Gaussian
throughout.
Assuming infinite pre-training prompts, population analysis of (Huang et al., 2025a) shows that with
proper initialization, the pretrained parameters are given by:
ˆV0 =
⎡⎢⎢⎢⎢⎢⎢⎢⎣
0
0
0
0
0
0
0
0
−Γ−1
0
0
0
0
0
0
0
0
⎤⎥⎥⎥⎥⎥⎥⎥⎦
,
ˆW0 =
⎡⎢⎢⎢⎢⎢⎢⎢⎣
0
0
I
0
0
0
0
−1
0
0
0
0
0
0
0
0
⎤⎥⎥⎥⎥⎥⎥⎥⎦
,
(3.6)
where
Γ0 ∶= (1 + 1
n)Σ0 + 1
ntr(Σ0)Id ∈Rd×d,
(3.7)
with n the prompt length. Post-training initializes from ( ˆV0, ˆW0), and updates the transformer weights by
minimizing either the SFT loss or the OS loss.
Sparsity structure motivated by the population regime. (Huang et al., 2025a) shows that training
with chain-of-thought (paralleling our SFT loss) in the population regime (B →∞before d,n) preserves
sparsity in the weights from initialization (3.6). Specifically, Lemma C.2 in (Huang et al., 2025a) proves that
the gradient flow trajectory preserves the following sparsity structure:
V (t) =
⎡⎢⎢⎢⎢⎢⎢⎢⎣
0
0 0 0
0
0 0 0
V31(t) 0 0 0
0
0 0 0
⎤⎥⎥⎥⎥⎥⎥⎥⎦
,
W(t) =
⎡⎢⎢⎢⎢⎢⎢⎢⎣
0 0 W13(t) 0
0 0
0
−1
0 0
0
0
0 0
0
0
⎤⎥⎥⎥⎥⎥⎥⎥⎦
,
(3.8)
where V31(t),W13(t) ∈Rd×d are the parameters at time t. While their analysis assumes identity-covariance
Gaussians and intermediate weights wi,τ derived from standard gradient descent the proof of Lemma C.2
in (Huang et al., 2025a) relies only on the symmetry properties of w∗
τ ∼N(0,I) and the fact that wi,τ is an
odd function of w∗
τ. Consequently, this structural result extends to our setting of general covariances and
supervised sequences. Although our analysis moves beyond the population regime, these insights motivate us
to constrain our transformer model to follow similar sparsity pattern. Throughout, we use the shorthands ̃V
and ̃
W to indicate the nonzero blocks of V and W.
4

--- Page 5 ---
4
Analysis of the SFT loss
Let Sτ ∶= 1
n ∑n
i=1 xi,τxT
i,τ be the empirical features covariance for τ ∈[B]. We also define the following matrices:
Ω∶= [w∗
1,...,w∗
B] ∈Rd×B
Φ ∶= [S1w∗
1,...,SBw∗
B] ∈Rd×B ,
M ∶= ΦΦT ∈Rd×d
The next theorem characterizes the minimizer of the SFT loss that is closest to the initialization (−Γ−1
0 ,I).
Theorem 4.1 Define
(̃Vλ, ̃
Wλ) =
arg min
(̃V ,̃
W )
LSFT(̃V , ̃
W) + λ∥̃V + Γ−1
0 ∥
2
F + λ∥̃
W −I∥
2
F .
We then have limλ→0+(̃Vλ, ̃
Wλ) = (̃V∗, ̃
W∗), where
̃
W∗= I,
̃V∗= −ηΩΦ† −Γ−1
0 (I −ΦΦ†)
(4.1)
Our next theorem shows that the solution (̃V∗, ̃
W∗) can be attained by gradient descent initialized at (−Γ−1
0 ,I),
and establishes conditions on the step size for convergence along with its convergence rate.
Theorem 4.2 Fix ̃
W = I. Consider the sequence of weights {̃Vt}t≥0 generated by the gradient descent update
̃Vt+1 = ̃Vt −γ∇̃V LSFT(̃Vt,I) with initialization ̃V0 = −Γ−1
0
and a constant step size 0 < γ. Define ρ ∶= 1 −η and
ck ∶= ∑k
i=0 ρ2i <
1
1−ρ2 =
1
2η−η2 . If γ <
2B
ckλmax(M), then the GD updates converges to ̃V∗at the following rate:
∥̃Vt −̃V∗∥F ≤αt∥Γ−1
0 + ̃V∗∥F ,
α ∶= max(∣1 −γck
B λmax(M)∣,∣1 −γck
B λ+
min(M)∣)
where λmax(M) and λ+
min(M) respectively denote the maximum and the minimum (nonzero) eigenvalues of
M. In particular, setting γ =
B
ckλmax(M), we obtain
∥̃Vt −̃V∗∥F ≤(1 −λ+
min(M)
λmax(M))
t
∥Γ−1
0 + ̃V∗∥F
Remark 4.1 Note that the loss minimizer (̃V∗, ̃
W∗) given by (4.1) depends on n (prompt length) and B
(number of prompts), the step size η in the supervised weight path, but not on k (length of reasoning paths).
However, if we fix the gradient step size γ < 2B(2η−η2)
λmax(M) , by Theorem 4.2 larger k implies larger ck and so faster
convergence rate.
It is worth deriving the limit of ̃V∗in the population regime, where B →∞, while n,d are kept fixed.
Proposition 4.3 Suppose that the features are generated as xi,τ ∼N(0,A) for a positive semidefinite matrix
A ∈Rd×d. Suppose n,d are fixed but the number of prompts B →∞. Then ̃V∗will converge to a limit ̃V∞
given by
̃V∞= −η (n + 1
n
A + tr(A)
n
AA†)
†
−Γ−1
0 (I −AA†)
(4.2)
5

--- Page 6 ---
5
Data Selection for Post-training via SFT
Proposition 5.1 Consider an LSA model with parameters (̃V , ̃
W). We fix ̃
W = I and assume a test prompt
of the form P = (x1,⟨w,x1⟩,...,xm,⟨w,xm⟩). Initializing the in-context learning with w0 = 0, the predicted
weight is given by ˆw = −1
n ̃V XX⊺w∗with X = [x1∣...∣xn] ∈Rd×n. In addition, if xi ∼N(0,Σ), we have
EX,w∗[∥ˆw −w∗∥2] = EX[∥I + ̃V ̂Σ∥
2
F ] =
= ∥I + ̃V Σ∥
2
F + 1
n (tr(̃V Σ2̃V T) + tr(̃V Σ̃V T)tr(Σ))
(5.1)
where the expectation is with respect to randomness in X and w∗∼N(0,Id).
In the test error (5.1), we focus on the dominant term ∥I + ̃V Σ∥F for large prompt length n. Assuming
post-training features are i.i.d. from N(0,A) for some A ⪰0, the post-training weights ̃V∗(A) depend on the
covariance A via Φ in (4.1). Thus, optimal data selection reduces to choosing covariance A that minimizes
the post-test error.
5.1
Optimal Data Allocation
To analyze the interaction between pre-training and post-training, we consider the test-time covariance
Σ = Σ0 + ∆, where Σ0 represents the distribution seen during pre-training and ∆denotes the adaptation task
shift. We now characterize how the choice of the post-training covariance A affects the post-test error across
different subspaces.
Let U = range(A). From (4.1), the term Φ shares the range U, while on the orthogonal complement
U ⊥, the weight matrix ˜V∗acts simply as the pre-trained inverse −Γ−1
0 . Furthermore, outside the range of
the adaptation shift ∆, the test-time covariance Σ coincides with the pre-training covariance Σ0. Since
Γ−1
0 Σ0 ≈I by the definition of Γ0 in (3.7), the residual error I + ˜V Σ on U ⊥becomes negligible if we align U
with range(∆). This alignment ensures that the post-training resources are concentrated exclusively on the
subspace where the pre-trained model exhibits a deficit.
Restricted to the adaptation subspace U = range(∆), the population-limit error operator is expressed as:
PU(I + V∞Σ)PU
= I −η (n + 1
n
A + tr(A)
n
I)
−1
(PUΣ0PU + ∆)
In the high-dimensional regime (large n), the trace term and the 1/n scaling factors become secondary,
implying that the optimal choice for the post-training covariance is approximately A ≈η(PUΣ0PU + ∆).
Connection to example hardness. In practice, post-training is often employed to address “gaps” in the
model—specifically, skills or topics that were missing or underrepresented during pre-training. To capture
such scenarios, we assume that the range of the pre-training covariance Σ0 and the range of the adaptation
shift ∆have a small inner product (i.e., they are nearly orthogonal). Consequently, PUΣ0PU constitutes
only a small component of Σ0. We argue that in these scenarios, the most effective strategy is to select
post-training examples that the pre-trained model finds “hard”. Specifically, Proposition 5.1 establishes that
the error of a pre-trained model on a task with prompts xi,τ ∼N(0,A) is approximately Lpre ≈∥I −Γ−1
0 A∥2
F .
Because the support of Σ0 is small on range(∆), the operator Γ−1
0 —which essentially acts as the inverse of
the pre-training density—takes its largest values on this space. Therefore, examples whose covariance is
spanned by range(∆) represent directions where the pre-trained model has the least confidence and highest
residual error. This leads to our first key insight:
Insight 1: Selecting examples that are “hard” for the pre-trained model (i.e., those aligned with
the adaptation shift ∆) is the most effective strategy for post-training.
6

--- Page 7 ---
0
200
400
600
800
1000
1200
1400
1600
1800
2000
100
101
102
n = 400
n = 800
n = 1200
(a) Optimal data selection for SFT
(r = 0)
0
200
400
600
800
1000
1200
1400
1600
1800
2000
100
101
102
103
104
n = 400
n = 800
n = 1200
(b) Data selection for SFT under in-
terference (r = 0.01).
0
200
400
600
800
1000
1200
1400
1600
1800
2000
100
101
102
n = 400
n = 800
n = 1200
(c) Data selection for SFT under in-
terference (r = 0.1).
Figure 1: Post-test error as the number or prompts B varies. Here, d = 400,m = 200 with different prompt
lengths (n). Pre-trained covariance is Σ0 = diag(ρ1m,1d−m), ∆= diag(1m,0n−m). Left panel represents
the optimal SFT data allocation with covariance A = diag(η(ρ + 1)1m,0n−m), with ρ = 0.1. The right
panel represents the case that SFT data distribution interferes with the pretraining distribution. Here,
A = diag(η(ρ + 1)1m,r1n−m), with r = 0.01.
5.2
Data Scaling in SFT
We study how SFT data size affects post-training performance by analyzing the expected error (Proposition 5.1)
on post-test prompts N(0,Σ).We examine how this error varies with the number of prompts B and the
prompt length n during SFT.
We first present experiments, followed by theory supporting the resulting insights. The pretraining
distribution is N(0,Σ0) with Σ0 = diag(ρ1m,0n−m), d = 400 and m = 200.
The post-test distribution
uses Σ = Σ0 + ∆, where ∆= diag(1m,0d−m).
During post-training, data is drawn from N(0,A) with
A = diag(η(ρ + 1)1m,r1n−m), matching ηΣ on the first m coordinates and using r on the rest. We set ρ
and r small so the first m directions are underrepresented in pretraining and can be strengthened during
post-training. When r = 0, the post-train distribution matches the optimal allocation of Section 5.1. However,
nonzero r introduces interference between post-training and pretraining data, which is often the case in
practice. By (4.1), the transformer parameters depend on the pseudo-inverse of the empirical covariance, so
smaller nonzero r yields stronger interference.
In the first experiment, we vary the number of prompts B from 50 to 2000, for prompt lengths n ∈
{400,800,1200}, fix ρ = 0.1, and consider interference levels r ∈{0,0.01,0.1}. Fig. 1 shows that the error
exhibits double descent, with an overshoot at B = m when r = 0 and at B = d when r ≠0. The error
first decreases with B, then increases again, and the crossover point grows with the prompt length n.
When interference is strong, the error remains above its value at optimal B even in large B limit (Fig. 1b).
In the second experiment, we vary the prompt length n from 20 to 1000 and evaluate post-test error at
B ∈{50,150,300,500}. As shown in Figure 2, the error trends differ across choice ofB. Under interference
and for small to moderate values of B, it first decreases with n and then becomes monotonically increasing,
yielding a U-shaped curve and indicating an optimal prompt length that minimizes test error.
These results show that increasing SFT data volume—either the number of prompts B or the prompt
length n—can paradoxically degrade performance in the presence of interference. The key trade-off is that
more SFT data helps the model learn underrepresented dimensions from pretraining, but also amplifies
interference that erodes pretrained capabilities. Our findings therefore suggest an optimal data size that
balances these competing effects. This further supports the empirical preference for small, high-quality
datasets, whose high information density enables effective adaptation without the catastrophic costs of
over-parameterization and interference. We formalize this observation as follows:
Insight 2: To mitigate the effects of interference between pretraining and post-training, SFT
datasets should be curated to be relatively small in volume and high in quality.
7

--- Page 8 ---
100
200
300
400
500
600
700
800
900
1000
1
2
3
4
5
6
7
B = 50
B = 150
B = 300
B = 500
(a) Optimal data selection for SFT
(r = 0)
0
200
400
600
800
1000
101
102
B = 50
B = 150
B = 300
B = 500
(b) Data selection for SFT under in-
terference (r = 0.01).
0
200
400
600
800
1000
2
3
4
5
6
7
8
9
B = 50
B = 150
B = 300
B = 500
(c) Data selection for SFT under in-
terference (r = 0.1).
Figure 2: Behavior of the post-test error as we varying the prompt length n, under the same setup as in
Figure 1.
In Appendix B we analyze the post-test error. The analysis, consistent with our experiments, predicts
that the test error diverges as B →d when interference is present (r ≠0) and as B →m when r = 0. We
further characterize the asymptotic limit of the post-test error in the scaling regime where d,m, and B →∞
while their relative ratios remain constant. This analysis demystifies the quantitative effect of different factors
on the test error behavior.
6
Analysis of the OS loss
We begin by deriving a more direct characterization of the outcome supervision (OS) loss.
Proposition 6.1 For the LSA model with k-step of thinking during the post-training the OS loss can be
written as
LOS(̃V , ̃
W)= 1
2B
B
∑
τ=1
∥(I +
k−1
∑
i=0
(̃V Sτ̃
W + I)ĩV Sτ)w∗
τ∥
2
ℓ2
The parameters (̃V , ̃
W) are initialized at (−Γ−1
0 ,I) from the pretraining stage. We next study the landscape
of the OS loss which demystifies several intriguing characteristics of post-training via OS and how it compares
with SFT post training. To simplify our discussions and derivations, we fix ̃
W = I and only update ̃V
via gradient descent. However, we expect our discussion to extend to the general case of updating both
parameters, albeit with a more complicated derivations. In our experiments, we update all of the transformer
weights and showing our insights from analysis are empirically observed as well.
By fixing ̃
W = I, the OS loss simplifies to:
LOS(̃V ,I) = 1
2B
B
∑
τ=1
∥(I + ̃V Sτ)kw∗
τ∥
2
ℓ2 .
Let Mτ = I + ̃V Sτ. As derived in Appendix E, the gradient of the OS loss with respect to the operator V
is given by:
∇V LOS = 1
B
B
∑
τ=1
k−1
∑
j=0
(M T
τ )jM k
τ w∗
τ(w∗
τ)T (M T
τ )k−1−jST
τ .
Vanishing and growing gradients in OS Loss. The gradient contains the term M k
τ , which acts as a
powerful scaling factor. In the stable region (ρ(Mτ) < 1), the term M k
τ shrinks the gradient toward zero
exponentially fast as the chain length k increases. In this regime, the model is already stable on the task, but
the vanishing gradient makes it increasingly difficult to “nudge” the matrix ̃V into the optimal subspace for
further refinement. Conversely, if ρ(Mτ) > 1, the gradient has an exponential growth in k. This creates a
8

--- Page 9 ---
sharp “cliff” in the loss landscape near the edge of stability (ρ ≈1), and training requires infinitesimally small
step sizes to prevent numerical divergence.
Sharpness and curvature of the landscape. Because the OS loss is effectively a degree-2k polynomial,
the Hessian ∇2L is highly sensitive to the operator’s spectral properties. As shown in Appendix E, near a
global minimum where M k
τ w∗
τ ≈0, the Hessian spectral norm λmax scales as:
λmax(H) ∝1
B
B
∑
τ=1
k2 ⋅ρ(Mτ)2k−2
(6.1)
This indicates that the curvature grows quadratically with the number of iterations k near the boundary
of stability. If gradient descent is not run for a sufficient duration, the model remains near this high-
curvature “cliff.” In this state, small variations—arising from finite n, B, or sample noise during post-test
evaluations—can push the model back into the unstable region, leading to “overthinking”, even if it pulled
into the stable region during training.
Insight 3: High sensitivity to sample variation. The sharp curvature near ρ ≈1 suggests
that Outcome Supervision (OS) is prone to instability unless trained with large amounts of data
(n,B) and many gradient steps. Insufficient training leaves the model at a “sharp” minimum
where minor distribution shifts cause large errors.
Pretraining and Generalization. The pretrained model, which serves as the initialization for the OS loss,
plays a critical role in OS stability. Consider a new task drawn from the test-time covariance Σ = Σ0 +∆, with
Σ0 the pretraining covariance and ∆the adaptation shift. Near initialization, and assuming a sufficiently
large prompt length n such that Sτ →Σ, the learned operator V is dominated by the prior V0 ≈−Γ−1
0 .
Consequently, we have V Sτ ≈−Γ−1
0 (Σ0 + ∆) ≈−I −Γ−1
0 ∆. Thus, the transition matrix becomes:
Mτ = I + V Sτ ≈−Γ−1
0 ∆Ô⇒ρ(Mτ) ≈ρ(Γ−1
0 ∆).
This relationship reveals two distinct optimization regimes based on the spectral alignment between the
pretraining distribution and the adaptation shift:
• Case 1: Incremental adaptation (spectral alignment). When Γ0 is large in the directions where ∆
is prominent—implying the pretraining distribution effectively covers the shift—the spectral radius ρ(Mτ)
remains small. In this regime, the model initializes within the stable region (ρ < 1), permitting a safe, albeit
gradual, refinement of the model parameters.
• Case 2: New task adaptation (spectral misalignment). If the task involves novel subspaces where
Γ0 is small but ∆is large, the spectral radius becomes large, i.e., ρ(Γ−1
0 ∆) ≫1. The model starts deep in
the unstable region, requiring a drastically reduced step size η to maintain stability:
η <
2
λmax(H) ∝
C
k2ρ(Mτ)2k−2 ,
by (6.1). These observations are summarized below:
Insight 4: Synergy of pretraining and Outcome Supervision. OS is most effective at
improving performance on tasks already partially learned during pretraining. For novel tasks, the
high initial spectral radius necessitates a slow and potentially unstable training procedure.
Practical Implications for Training. The requirement for stability dictates several constraints on Outcome
Supervision and RL. To ensure the eigenvalues remain within the stable regime, the learning rate must be
carefully tuned to the sharpest direction of the Hessian. This creates a stark disparity in the optimization
landscape: the step size η, forced to be infinitesimally small by the unstable directions, can be too small
to make meaningful progress in the data-aligned directions. In addition, while RL does not require the
high-quality, human-curated labels necessary for SFT, it compensates by requiring massive data diversity and
volume. A large number of gradient steps is needed to overcome the slow progress in “flat” directions, while a
high volume of data ensures the model is pushed deep into the stable region across a broad spectrum of tasks,
reducing the risk of “overthinking” during inference.
9

--- Page 10 ---
7
Data Diversity and Distributional Balance in Pretraining
In our analysis, the influence of the pretrained model on post-trained model is mathematically encapsulated
in the initialization V0 = −Γ−1
0 , where by definition (3.7), Γ0 ≈Σ0 the pretraining covariance. The post-test
error, characterized by Proposition 5.1, is governed by the product V Σ = V (Σ0 + ∆); at initialization, this
yields V Σ ≈−I −Γ−1
0 ∆. Consequently, an imbalanced pretraining distribution—characterized by a singular
or ill-conditioned Γ0—imposes a severe penalty on adaptation in new directions where Γ0 is small but ∆is
large. While SFT can partially mitigate a misaligned prior through the stabilizing influence of supervised
signals, the OS and RL optimization is strictly bottlenecked by the spectral alignment between Γ0 and ∆.
If Γ0 lacks sufficient diversity, even minor shifts in novel subspaces trigger an exponential escalation of the
Hessian’s spectral norm, scaling as k2ρ2k−2. This spectral divergence necessitates infinitesimally small step
sizes and renders the model sensitive to variations in sample prompts in training. Such instability often
manifests as “overthinking” during inference. Therefore, pretraining must prioritize distributional balance
and data diversity as essential mechanisms for optimization stability. A broad spectral prior ensures the
model initializes within the stable regime (ρ < 1), effectively smoothing the high-curvature “cliffs” of the RL
landscape into manageable, flat regions for downstream adaptation.
8
Experiments
In this section, we conduct experiments to validate our theoretical results.
Setting. We conduct experiments in two settings. First, we consider a transformer with a single linear
self-attention (LSA) to confirm the results of our theorems. Then, we consider large, nonlinear transformer
architecture namely GPT2 to validate the generality of our conclusions.
In both sets of experiments, the data distribution follows our in-context weight prediction task in Sec.
3, where in the pre-training, data has a covariance of Σ0, and in the post-testing with SFT or OS we have
Σ = Σ0 + ∆. During post-training, we let the model to output multiple steps before returning the final
predicted weight vector, i.e., at each step i we concatenate the embedding with [0d, ˆwi,1] as in Eq. (3.3)
and input the concatenated embedding matrix to the model. The estimated ˆwk will be returned after k steps
of Chain of Thought (CoT). We report the average results and error bars over 10 runs.
Pretrain, post-train, and test data. We generate pretraining data using Σ0 where Σi,i = 0.1 for
i ∈{1,...,d/5} and Σi,i = 1 for i ∈{d/5,...,d}. Then, we post-train the transformer on the synthetic data
generated with ∆, where ∆is a low rank PSD matrix with ∆i,i = 1. For testing the model, we use Σ = Σ0 + ∆.
Large, nonlinear transformer architectures.
We use a decoder-only Transformer architecture
(Vaswani et al., 2017) from the GPT-2 family (Radford et al., 2019), consisting of 12 layers, 8 attention
heads and a 256-dimensional embedding space. In total model contains 9.5M parameters. This architecture
takes as input a sequence of vectors in its embedding space and predicts the weight vector within the same
space. We apply this architecture to prompts of form (xτ,1,yτ,1,⋯,xτ,m,yτ,m,w0,1) in the following manner.
In line with (Garg et al., 2022), we map each yτ,i to the same dimension as xτ,i by appending zeros, and
map xτ,i,yτ,i into the latent embedding space of the Transformer through a (learnable) linear transformation.
We get the predicted wτ as the model output. Similarly, we map the model output, i.e., wτ from the
latent embedding space of the Transformer to a d-dimensional vector through another (learnable) linear
transformation. Training is performed with a batch size of 64 over 100 steps for SFT and 12k steps for OS.
The model is first pretrained with a CoT length k = 8. During both training and test, we apply CoT with
length k = 3. We used curriculum learning (Garg et al., 2022) to speed up training.
Fig. 3 (a)-(c) show the results when post-training is done with the SFT loss. Fig. 3a,3b show that
increasing the sample size (B) or context length (n) initially yields a lower test loss but further increasing the
sample size or context length increases the test loss. Fig. 3c shows that the test loss is relatively robust and
not sensitive to the length of post-training CoT (k). Fig 3 (d)-(f) show the results when post-training is done
with the OS loss. In contrast to SFT, we see that OS benefits from larger sample size (B) and context length
10

--- Page 11 ---
(a) Supervised fine-tuning
(b) Supervised fine-tuning
(c) Supervised fine-tuning
(d) Outcome Supervision
(e) Outcome Supervision
(f) Outcome Supervision
Figure 3: GPT-2 experiments: Test loss for (a)-(c) post-training with SFT, and (d)-(f) post-training with
Outcome Supervision (OS). For SFT, there is a turning point where larger sample size (B) and context-length
(n) hurt the performance. In contrast, for OS larger B,n improves the performance.
(n). In addition, longer CoT (k) during post-training increases the test loss and degrades the performance,
confirming insight 4 in Section 6.
Linear self-attention (LSA) experiments. We next present our results on transformers with a single
linear self-attention (LSA) layer. We choose the token dimensions d = 100, and post-train the model for 130
epochs using Adam with learning rate η = 0.001. During inference, we return the final predicted weight vector
without CoT, i.e. at test time we use k = 1.
Fig. 4 (a)-(c) show the results when post-training is done with the SFT loss. Fig. 4a, 4b show that
increasing the sample size (B) or context length (n) initially yields a lower test loss but further increasing
the sample size or context length increases the test loss. Fig. 4c shows that the test loss is relatively robust
and not sensitive to the length of post-training CoT (k). Fig 4 (d)-(f) show the results when post-training is
done with the OS loss. In contrast to SFT, Fig. 4d, 4e show that OS benefits from larger sample size (B)
and context length (n), and Fig. 4f shows that longer CoT (k) during post-training increases the test loss
and degrades the performance.
9
Conclusion
Our work provides a theoretical and empirical framework for jointly designing pretraining and post-training
for LLMs. Balanced pretraining creates latent capabilities best activated by SFT on small numbers of carefully
selected, hard examples aligned with the target shift. Scaling up SFT data introduces interference that erodes
pretrained structure, favoring small, high-quality datasets. Outcome Supervision and RL have a sharply
curved, unstable landscape that make them data-hungry, yet effective for refining partially learned pretrained
capabilities. These insights guide optimal combined use: targeted SFT for efficient adaptation on challenging
examples, complemented by large-scale RL (Outcome Supervision) for robust skill refinement.
11

--- Page 12 ---
(a) Supervised fine-tuning
(b) Supervised fine-tuning
(c) Supervised fine-tuning
(d) Outcome Supervision
(e) Outcome Supervision
(f) Outcome Supervision
Figure 4: LSA experiments: Test loss for (a)-(c) post-training with SFT, and (d)-(f) post-training with
Outcome Supervision (OS). For SFT, there is a turning point where larger sample size (B) and context-length
(n) hurt the performance. In contrast, for OS larger B,n improves the performance.
Acknowledgments
AJ was supported in part by the NSF Award DMS-2311024, an Amazon Faculty Research Award, an Adobe
Faculty Research Award, and an iORB grant form USC Marshall School of Business. BM was supported in
part by the NSF CAREER Award 2146492, NSF-Simons AI Institute for Cosmic Origins (CosmicAI) and
NSF AI Institute for Foundations of Machine Learning (IFML).
12

--- Page 13 ---
A
Proof of theorems and technical lemmas
A.1
Proof of Theorem 4.1
As λ →0+, the minimizer (̃Vλ, ̃
Wλ) must converge to a point (̃V∗, ̃
W∗) in the zero-loss manifold of L(̃V , ̃
W)
that is closest to the initialization (−Γ−1
0 ,I) in the Frobenius norm.
We first simplify the dynamic of LSA into a recurrent update on the estimated weight ˆwi. We have We
have
fLSA(Zi,θ∗)[∶,−1] =
⎡⎢⎢⎢⎢⎢⎢⎢⎣
0d×1
0
ˆwi
1
⎤⎥⎥⎥⎥⎥⎥⎥⎦
+ V Zi ⋅Z⊺
i WZi[∶,−1]
n
=
⎡⎢⎢⎢⎢⎢⎢⎢⎣
0d×1
0
ˆwi
1
⎤⎥⎥⎥⎥⎥⎥⎥⎦
+ 1
nV ZiZ⊺
i
⎡⎢⎢⎢⎢⎢⎢⎢⎣
̃
W ˆwi
−1
0
0
⎤⎥⎥⎥⎥⎥⎥⎥⎦
=
⎡⎢⎢⎢⎢⎢⎢⎢⎣
0d×1
0
ˆwi
1
⎤⎥⎥⎥⎥⎥⎥⎥⎦
+ 1
n
⎡⎢⎢⎢⎢⎢⎢⎢⎣
0d×n
0d×1
0d×1
0d×1
01×n
0
0
0
̃V X
0d×1
0d×1
0d×1
01×n
0
0
0
⎤⎥⎥⎥⎥⎥⎥⎥⎦
⎡⎢⎢⎢⎢⎢⎢⎢⎣
X
0
0
...
0
y
0
0
...
0
0d×n
w0
ˆw1
...
ˆwi
01×n
1
1
...
1
⎤⎥⎥⎥⎥⎥⎥⎥⎦
T ⎡⎢⎢⎢⎢⎢⎢⎢⎣
̃
Ww0
−1
0
0
⎤⎥⎥⎥⎥⎥⎥⎥⎦
=
⎡⎢⎢⎢⎢⎢⎢⎢⎣
0d×1
0
ˆwi
1
⎤⎥⎥⎥⎥⎥⎥⎥⎦
+ 1
n
⎡⎢⎢⎢⎢⎢⎢⎢⎣
0d×n
0d×1
0d×1
0d×1
01×n
0
0
0
̃V X
0d×1
0d×1
0d×1
01×n
0
0
0
⎤⎥⎥⎥⎥⎥⎥⎥⎦
[XT̃
W ˆwi −yT
0
]
=
⎡⎢⎢⎢⎢⎢⎢⎢⎣
0d×1
0
ˆwi
1
⎤⎥⎥⎥⎥⎥⎥⎥⎦
+ 1
n
⎡⎢⎢⎢⎢⎢⎢⎢⎣
0d×1
0
̃V XXT(̃
W ˆwi −w∗)
0
⎤⎥⎥⎥⎥⎥⎥⎥⎦
.
Hence, we obtain the following recursions for each of the prompt weight vectors:
ˆwi+1,τ = ˆwi,τ + ̃V Sτ(̃
W ˆwi,τ −w∗
τ).
(A.1)
Now note that in the SFT loss, at each step we give the model the CoT ground-truth sequence (w1,τ,...,wi,τ)
and compute the error ∥wi+1,τ −ˆwi,τ∥2
ℓ2. Let ρ = 1 −η. Given wi,τ = (1 −ρi)w∗
τ, we define the residual Ri,τ for
i = 0,...,k and τ = 1,...,B as follows:
Ri,τ = wi,τ + ̃V Sτ(̃
Wwi,τ −w∗
τ) −wi+1,τ
= (1 −ρi)w∗
τ + ̃V Sτ(̃
W(1 −ρi)w∗
τ −w∗
τ) −(1 −ρi+1)w∗
τ
= ̃V Sτ(̃
W −I)w∗
τ −ρi(̃V Sτ̃
W + ηI)w∗
τ
We characterize this manifold by analyzing the residual Ri,τ for each block τ and iteration i ∈{0,...,k}. The
loss function can be written as
LSFT(̃V , ̃
W) = 1
2B
B
∑
τ=1
k
∑
i=0
∥Ri,τ∥2
ℓ2 .
To characterize the zero-loss manifold, note that for L(̃V , ̃
W) = 0, we require Ri,τ = 0 for all i. Since 1
and ρi are linearly independent for i ≠0, the coefficients of the polynomial in ρi must vanish independently:
1. ̃V Sτ(̃
W −I)w∗
τ = 0
13

--- Page 14 ---
2. (̃V Sτ̃
W + ηI)w∗
τ = 0 Ô⇒̃V Sτ̃
Ww∗
τ = −ηw∗
τ
Substituting the second condition into the first, we obtain:
̃V Sτ̃
Ww∗
τ −̃V Sτw∗
τ = 0 Ô⇒−ηw∗
τ −̃V Sτw∗
τ = 0 Ô⇒̃V Sτw∗
τ = −ηw∗
τ ,
for all τ = 1,...,B. Let Ω= [w∗
1,...,w∗
B] and Φ = [S1w∗
1,...,SBw∗
B]. The system is expressed as ̃V Φ = −ηΩ.
The limit ̃V∗minimizes ∥̃V + Γ−1
0 ∥
2
F subject to ̃V Φ = −ηΩ, which is solved via the Moore-Penrose pseudoinverse:
̃V∗= −ηΩΦ† −Γ−1
0 (I −ΦΦ†).
The term (I −ΦΦ†) is the orthogonal projection onto the null space of Φ⊺, ensuring ̃V follows the initialization
−Γ−1
0
in directions not spanned by the data.
Now that ̃V∗is characterized, we proceed with proving that ̃
W∗= I. Note that this choice of ̃V∗, ̃
W∗
satisfies both of the gradient condition (1) and (2) above. In addition, due to the penalty λ∥̃
W −I∥
2
F , we get
̃
W∗= I as the unique minimizer.
A.2
Proof of Theorem 4.2
Let ρ = 1 −η and ck = ∑k
i=0 ρ2i. Given ̃
W = I and wi,τ = (1 −ρi)w∗
τ, the residual is Ri,τ = −ρi(̃V Sτ + ηI)w∗
τ
and the loss can be written as
LSFT(̃V ,I) = ck
2B ∥̃V Φ + ηΩ∥
2
F ,
where we recall Φ = [S1w∗
1,...,SBw∗
B] and Ω= [w∗
1,...,w∗
B]. The gradient of the loss is given by
∇̃V LSF = ck
B (̃V Φ + ηΩ)Φ⊺
Defining ∆t = ̃Vt −̃V∗and noting ̃V∗Φ = −ηΩ, the GD update ̃Vt+1 = ̃Vt −γ∇̃V LSF (̃Vt,I) yields:
∆t+1 = ∆t (I −γck
B M),
M = ΦΦ⊺
The error norm evolves as ∥∆t+1∥F ≤∥∆t∥F ⋅∥I −γck
B M∥op, with ∥⋅∥op indicating the operator norm.
Note that the condition γ <
2B
ckλmax(M) ensures that ∥I −γck
B M∥op < 1 and so the GD updates converges to
̃V∗. Specifically, the contraction factor is determined by the most extreme eigenvalues that the error ∆t sees
in the subspace spanned by the data Φ. On the range of Φ, the contraction factor is given by
α ∶= max(∣1 −γck
B λmax(M)∣,∣1 −γck
B λ+
min(M)∣)
By choosing γ =
B
ckλmax(M), the rate simplifies to
α = 1 −λ+
min(M)
λmax(M)
Substituting ∆0 = ̃V0 −̃V∗= −Γ−1
0 −̃V∗, we obtain the desired bound:
∥̃Vt −̃V∗∥F ≤(1 −λ+
min(M)
λmax(M))
t
∥Γ−1
0 + ̃V∗∥F ,
which completes the proof.
14

--- Page 15 ---
A.3
Proof of Proposition 4.3
Recalling from (4.1), V∗satisfies the system ̃V Φ = −ηΩ. To find the explicit limit as B →∞, we analyze the
normal equations:
̃V ( 1
B ΦΦ⊺) = −η
B ΩΦ⊺
Recall w∗
τ ∼N(0,I) and Sτ being the empirical covariance of n samples from N(0,A). In addition, w∗
τ and
Sτ are independent.
We have
E[ 1
B ΩΦ⊺] = E[ 1
B
B
∑
τ=1
w∗
τ(Sτw∗
τ)⊺] = E[w∗w∗⊺S⊺
τ ]
By independence and the fact that E[w∗w∗⊺] = I and E[Sτ] = A, we get
E[ 1
B ΩΦ⊺] = A
In addition,
E[ 1
B ΦΦ⊺] = E[ 1
B
B
∑
τ=1
(Sτw∗
τ)(Sτw∗
τ)⊺] = E[Sτw∗w∗⊺S⊺
τ ] = E[S2
τ]
Using the properties of the Wishart distribution for Sτ = 1
n ∑n
i=1 xix⊺
i with xi ∼N(0,A), (see Lemma A.2
in (Javanmard et al., 2025)) we have
E[S2
τ] = n + 1
n
A2 + 1
ntr(A)A
First consider the case A is invertible. By Slutsky’s Theorem and the consistency of the sample covariance,
as B →∞, the learned operator ̃V converges in probability to:
̃V∞= −ηA(E[S2
τ])
−1
Substituting the explicit form of E[S2
τ]:
̃V∞= −ηA(n + 1
n
A2 + tr(A)
n
A)
−1
= −η (n + 1
n
A + tr(A)
n
I)
−1
When A is singular, the same derivation holds in the range of A. In the null space of A, ̃V∞stays at its
initialization −Γ−1
0 . Both cases can be unified as follows:
̃V∞= −η (n + 1
n
A + tr(A)
n
AA†)
†
−Γ−1
0 (I −AA†),
which completes the proof.
A.4
Proof of Proposition 5.1
Specializing the recursion (A.1) to i = 0 and ̃
W = I, we have ˆw = w0 + 1
n ̃V XXT(w0 −w∗). By choosing the
initialization w0 = 0 we arrive at ˆw = −1
n ̃V XXTw∗.
Letting ̂Σ = 1
nXXT, we have
E[∥ˆw −w∗∥2
ℓ2] = E[∥I + ̃V ̂Σ∥
2
F ] = ∥I + ̃V Σ∥
2
F + 1
n (tr(̃V Σ2̃V T) + tr(̃V Σ̃V T)tr(Σ))
where the last step follows from Lemma A.1 below.
15

--- Page 16 ---
Lemma A.1 Let X = [x1∣...∣xn]T with xi ∼N(0,Σ) with Σ ∈Rd×d. Define ̂Σ ∶= 1
nXTX. Then, for any
matrix A ∈Rd×d, we have
E[∥I + ÂΣ∥
2
F ] = ∥I + AΣ∥2
F + 1
n (tr(AΣ2AT) + tr(AΣAT)tr(Σ))
(A.2)
Proof (Proof of Lemma A.1) We write
E[∥I + ÂΣ∥
2
F ] = d + E[∥ÂΣ∥
2
F ] −2E[tr(AΣ)]
(A.3)
From (Javanmard et al., 2025)(Lemma A.2) we have
E[̂Σ(ATA)̂Σ)] = n −1
n
Σ(ATA)Σ + 1
n (2Σ(ATA)Σ + tr(ΣATA)Σ) .
Hence, by taking the trace of both sides and changing the orde of expectation and trace (since it is a linear
operator), we get
E[∥ÂΣ∥
2
F ] = n + 1
n
tr(AΣ2AT) + 1
ntr(AΣAT)tr(Σ).
Here we also used the identity tr(AB) = tr(BA) for square matrices of the same size.
Substituting back in (A.3) we obtain
E[∥I + ÂΣ∥
2
F ] = d + tr(ATΣ2A) −2E[tr(AΣ)] + 1
n (tr(AΣ2AT) + tr(AΣAT)tr(Σ))
= ∥I + AΣ∥2
F + 1
n (tr(AΣ2AT) + 2tr(AΣAT)tr(Σ))
which completes the proof of lemma.
A.5
Proof of Proposition 6.1
We begin by recalling the recursion (A.1):
ˆwi+1,τ = ˆwi,τ + ̃V Sτ(̃
W ˆwi,τ −w∗
τ)
= (I + ̃V Sτ̃
W) ˆwi,τ −̃V Sτw∗
τ
Solving this recursion, we obtain
ˆwk,τ = (I + ̃V Sτ̃
W)k ˆw0 −
k−1
∑
i=0
(I + ̃V Sτ̃
W)ĩV Sτw∗
τ .
(A.4)
Next, using that ˆw0 = w0 = 0, we get
LOS(V,W) = 1
2B
B
∑
τ=1
∥ˆwτ,k −w∗
τ∥2
ℓ2
= 1
2B
B
∑
τ=1
∥(I +
k−1
∑
i=0
(̃V Sτ̃
W + I)ĩV Sτ)w∗
τ∥
2
ℓ2
,
which completes the proof.
16

--- Page 17 ---
B
Asymptotic Analysis of SFT post-training
We recall our notations from Section 4. Let Sτ ∶= 1
n ∑n
i=1 xi,τxT
i,τ be the empirical features covariance for
τ = 1,...,B. We also define the following matrices:
Ω∶= [w∗
1,...,w∗
B] ∈Rd×B ,
Φ ∶= [S1w∗
1,...,SBw∗
B] ∈Rd×B ,
(B.1)
Also recall that the SFT data are generated as xi ∼N(0,A) where A = η(PUΣ0PU + ∆) + rPU ⊥, with
U = range(∆). When r = 0 this corresponds to the optimal data allocation discussed in Section 5.1 and r ≠0
models the interference between SFT data and the pretrained model.
We consider the following specific structure for the pretrained covariance Σ0 and distribution shift
covariance ∆similar to our experiments in Section 5.2, namely
Σ0 = diag(ρ1m,1d−m),
∆= diag(1m,0d−m).
During post-training, SFT data is generated from N(0,A) with
A = diag(η(ρ + 1)1m,r1d−m),
(B.2)
and the post-test distribution is given by the covariance Σ = Σ0 + ∆. Notably, Our asymptotic framework
generalizes to arbitrary covariance structures Γ0,∆, and A, provided the empirical spectral distributions
of these matrices converge weakly to probability measures on R≥0 with finite second moments. Under this
Mean-Field regime, the macroscopic behavior of the learned operator ̃V∗is determined by the spectral
densities of the data and shift matrices, rather than their specific coordinate-level realizations.
Decomposition of ̃V∗: Starting from ̃V∗= −ηΩΦ† −Γ−1
0 (I −ΠΦ), with projection ΠΦ = ΦΦ†.
Let Φ = M + E, where M = AΩand E is the perturbation of Φ from its expectation AΩwith respect
to randomness in the empirical features covariances Sτ, for τ ∈[B]. Using the first-order expansion of the
pseudoinverse:
ΠΦ ≈(M + E)(M † −M †EM † + ...)
Multiplying this out and keeping only terms up to the first power of E:
ΠΦ ≈MM †
´¹¹¹¸¹¹¶
ΠΩ
+EM † −MM †EM †
´¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¸¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¶
First-order correction
We can simplify the correction term by factoring MM †:
ΠΦ ≈MM † + (I −MM †)EM † ,
(I −ΠΦ) ≈(I −MM †) −(I −MM †)EM † .
By substituting this expanded projection back into the definition of ̃V∗, we get the following first-order
approximation:
̃V∗≈−ηΩ(M † −M †EM †) −Γ−1
0 [(I −MM †) −(I −MM †)EM †]
(B.3)
Now, group the terms into deterministic (VS) and stochastic (VN) components. The Zero order component is
given by:
VS = −ηΩM † −Γ−1
0 (I −MM †)
The first order component is given by:
VN = −VSEM †
Equation (B.3) can be written as
̃V∗≈̃V ∶= VS + VN.
(B.4)
17

--- Page 18 ---
We next characterize the limit of test error using (5.1). For convenience we rewrite the characterization
for the expected test error below:
Err(̃V ) = 1
d E[∥I + ̃V Σ∥
2
F ]
´¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¸¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¶
Term I
+ 1
nd E[tr(̃V Σ2̃V T) + tr(̃V Σ̃V T)tr(Σ)]
´¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¸¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¶
Term II
(B.5)
where expectation is with respect to both the training and the test data. We also normalized the test error
by the dimension d.
Proportional regime. We consider the proportional asymptotic regime, where d,m,n,B →∞, with n the
prompt length and B the number of prompts. In addition, B/d →β, m/d →µ1, d/n →γ for some arbitrary
but fixed constants β,µ1,γ. We also let µ2 = 1 −µ1.
Notations. The deterministic feature covariance A is diagonal with block entries a1,a2. The test covariance
Σ is also diagonal with block diagonals Σ1,Σ2, namely
a1 ∶= η(ρ + 1),
a2 ∶= r
Σ1 ∶= ρ + 1,
Σ2 ∶= 1
(B.6)
Let D1 = I −Γ−1
0 Σ and Dpre = Γ−1
0 −ηA−1. Both are block-diagonal deterministic matrices and in the
proportional asymptotic regime, we let αk and ˜δk be their respective scalar values on block k ∈{1,2}, and let
δk = ˜δkΣk. A simple calculation shows that with κ ∶= γ(µρ + 1 −µ), we have
α1 = κ −1
ρ + κ,
α2 =
κ
κ + 1
δ1 = 1 −κ
ρ + κ,
˜δ1 = δ1
Σ1
=
1 −κ
(ρ + κ)(ρ + 1)
δ2 =
1
1 + κ −η
r ,
˜δ2 = δ2
Σ2
= δ2
(B.7)
In addition, −Γ−1
0
is also a block-diagonal deterministic matrix and in the asymptotic regime, we let gk be its
respective scalar values on block k ∈{1,2}. It is easy to see that
g1 = −
1
ρ + κ,
g2 = −
1
1 + κ
(B.8)
The matrix ΣA ∶= 1
d (tr(A)A + A2) is also block-diagonal and in the proportional regime, its respective scalar
values on block k ∈{1,2} converge to
s1 = (µ1a1 + µ2a2)a1 = (µ1η(ρ + 1) + µ2r)η(ρ + 1)
s2 = (µ1a1 + µ2a2)a2 = (µ1η(ρ + 1) + µ2r)r
(B.9)
Theorem B.1 Consider ̃V = VS +VN the first order approximation of ̃V∗as in (B.4). Under the proportional
asymptotic regime, the following holds true:
lim
d→∞Err(̃V ) = Bias + γTinvTvar + γ ¯ΣTvar,Σ + γ2 ¯ΣTinv,ΣTvar
(B.10)
where the terms are defined as follows, in terms of the notations defined by (B.6), (B.7), (B.8) and (B.9):
• Bias: For β < 1, let q be the non-negative solution to:
β =
2
∑
k=1
µk
a2
kq
1 + a2
kq
(B.11)
18

--- Page 19 ---
and define wk =
a2
kq
1+a2
kq, vk = wk(1 −wk), for k ∈{1,2} and
T12 =
µ1µ2v1v2
µ1v1 + µ2v2
(B.12)
For β ≥1, set wk = 1, vk = 0, and T12 = 0. We then have
Bias ∶=
2
∑
k=1
µk [α2
k(1 −wk) + (αk + δk)2wk] + T12(˜δ2
2 −˜δ2
1)(Σ2
1 −Σ2
2)
(B.13)
• The terms Tinv and Tinv,Σ are given by
Tinv =
⎧⎪⎪⎪⎨⎪⎪⎪⎩
q
∑2
k=1 µk
Σ2
k
a2
k w2
k
∑2
k=1 µk 1
a2
k w2
k
⎫⎪⎪⎪⎬⎪⎪⎪⎭
1(β < 1) + {
1
β −1
2
∑
k=1
µk
Σ2
k
a2
k
}1(β > 1)
(B.14)
Tinv,Σ =
⎧⎪⎪⎪⎨⎪⎪⎪⎩
q
∑2
k=1 µk
Σk
a2
k w2
k
∑2
k=1 µk 1
a2
k w2
k
⎫⎪⎪⎪⎬⎪⎪⎪⎭
1(β < 1) + {
1
β −1
2
∑
k=1
µk
Σk
a2
k
}1(β > 1)
(B.15)
• The terms Tvar,E and Tvar,Σ are given by
Tvar =
2
∑
k=1
µksk [g2
k(1 −wk) + (gk + ˜δk)2wk] −T12(˜δ1 −˜δ2)(˜δ1s1 −˜δ2s2)
(B.16)
Tvar,Σ =
2
∑
k=1
µkΣk [g2
k(1 −wk) + (gk + ˜δk)2wk] −T12(˜δ1 −˜δ2)(˜δ1Σ1 −˜δ2Σ2)
(B.17)
We next compare the predicted asymptotic limit of Err with numerical experiment. Recall ̃V∗as the SFT loss
minimizer given by (4.1), ̃V its first order approximation, given by (B.4). In Figure 5 we plot Err(̃V∗), Err(̃V )
and our theoretical curve (B.10). As we see there is a great match between our theoretical prediction and
simulation result for (Err(̃V )). In addition, it approximates Err(̃V∗) reasonably well and the approximation
becomes tighter as the prompt length (n) grows (Figure 5b shows a better approximation at n = 5000
compared to Figure 5a for n = 1000).
Using Theorem B.10 we prove several properties of the asymptotic error and show that under interference,
its minimum is achieved in the regime of β < 1. This confirms our Insight 2 in the main text, namely that
SFT datasets should be curated to be relatively small in volume.
We denote the predicted theoretical error (right hand side of (B.10)) by F(β), as function of β, as we
would like to understand its behavior as β varies.
Proposition B.2 The followings hold true:
(i) limβ→1 F(β) = ∞. For β > 1, F(β) is strictly decreasing. As β →∞, it converges to a finite asymptotic
floor:
F(↑∞) ∶= lim
β→∞F(β) =
2
∑
k=1
µk(αk + δk)2 + γ ¯Σ
2
∑
k=1
µkΣk(gk + ˜δk)2
(ii) We have
F(0) =
2
∑
k=1
µkα2
k + γ ¯Σ
2
∑
k=1
µkΣkg2
k
Also, F(↑∞) −F(0) scales as O(1/r2). Consequently, for sufficiently small r > 0, F(↑∞) > F(0). This
guarantees that the global minimum of F(β) is strictly achieved in the overparameterized regime (β < 1).
(iii) Suppose that µ1 ≥
ρ2
1+ρ2 .
For sufficiently small r and γ, the initial derivative is strictly negative
(F ′(0) < 0). Hence, introducing a small number of prompts immediately and strictly decreases the test
error.
19

--- Page 20 ---
0
100
200
300
400
500
600
700
800
900
1000
101
102
103
104
105
106
(a) prompt length (n = 1000)
0
100
200
300
400
500
600
700
800
900
1000
101
102
103
104
105
106
(b) prompt length (n = 5000)
Figure 5: Comparison between theoretical prediction of the asymptotic error Err(̃V ), the simulation results
for Err(̃V ) and Err(̃V∗). We see a great match between theoretical prediction and simulation results. Here,
d = 600, m = 300, n = 600 (prompt size), ρ = 0.1, η = 0.2, r = 0.1 (interference parameter). The simulations are
averaged over 10 realizations.
C
Proof of Theorem B.1
C.1
Analysis of Term I
We start by analyzing Term I. We have
E[∥I + ̃V Σ∥
2
F ] = E[∥I + VSΣ + VNΣ∥2
F ] = E[∥I + VSΣ∥2
F ] + E[∥VNΣ∥2
F ]
because conditioned on Ω, E = [(S1 −A)w∗
1,...,(SB −A)w∗
B] is zero mean and independent of VS. Hence,
lim
d→∞
1
d E[∥I + ̃V Σ∥
2
F ] = lim
d→∞
1
d E[∥I + VSΣ∥2
F ] + lim
d→∞
1
d E[∥VNΣ∥2
F ].
(C.1)
Analysis of the Bias term. The deterministic component of the test error (Bias) is governed by the matrix
MS = I + VSΣ. We first express VS in terms of the orthogonal projection matrix ΠM = MM †. Using the
identity A−1ΠM = ΩM †, we have:
VS = −ηA−1ΠM −Γ−1
0 (I −ΠM)
(C.2)
MS = (I −Γ−1
0 Σ) + (Γ−1
0 −ηA−1)ΠMΣ
(C.3)
Let D1 = I −Γ−1
0 Σ and Dpre = Γ−1
0 −ηA−1. Thus, MS = D1 + DpreΠMΣ. Note that D1 and Dpre are
block-diagonal deterministic matrices, and in the asymptotic regime, their respective scalar values on block
k ∈{1,2} converge to αk and ˜δk given by (B.7). Let δk = ˜δkΣk.
Expanding the normalized squared Frobenius norm, we obtain:
1
d∥MS∥2
F = 1
dtr(D2
1) + 2
dtr(D1ΣΠMDpre) + 1
dtr(D2
preΠMΣ2ΠM)
(C.4)
We next note that
1
dtr(D2
1) =
2
∑
k=1
dk
d α2
k =
2
∑
k=1
µkα2
k
(C.5)
20

--- Page 21 ---
In addition, we have
2
dtr(D1ΣΠMDpre) = 2
dtr(DpreD1ΣΠM) = 2
d
2
∑
k=1
αkδktr(Πkk)
(C.6)
To evaluate the quadratic trace Quad ∶= 1
dtr(D2
preΠMΣ2ΠM), we partition the projection matrix into
blocks Πij with i,j ∈{1,2} with Π11 of size m and Π2,2 of size d −m. Let Tij = 1
dtr(ΠijΠji). Expanding the
trace block-by-block yields:
Quad = δ2
1T11 + δ2
2T22 + (˜δ2
2Σ2
1 + ˜δ2
1Σ2
2)T12 .
(C.7)
Because ΠM is a true orthogonal projection, Π2
M = ΠM. Examining the diagonal blocks of this identity
gives
Π2
kk + ΠkjΠjk = Πkk,
(C.8)
with k ≠j ∈{1,2}.
Our next lemma characterizes the limit of normalized trace of Πkk, using Stieltjes transform and Silverstein
equation from the Random Matrix Theory.
Lemma C.1 Let wk ∶= limdk→∞1
dk tr(Πkk). Then, the following holds: For β < 1,
wk =
a2
kq
1 + a2
kq ,
(C.9)
with q being the non-negative solution to:
β =
2
∑
k=1
µk
a2
kq
1 + a2
kq
(C.10)
For β ≥1, we have wk = 1.
Taking the normalized trace from (C.8) gives:
Tkk = µkwk −T12,
k ∈{1,2}
(C.11)
Substituting this into the quadratic term and simplifying:
Quad = µ1δ2
1w1 + µ2δ2
2w2 + T12(˜δ2
2 −˜δ2
1)(Σ2
1 −Σ2
2)
(C.12)
Also by recalling (C.6) we have
lim
d→∞
2
dtr(D1ΣΠMDpre) = 2
dtr(D1ΣΠMDpre) = lim
d→∞2
2
∑
k=1
(dk
d )αkδk ( 1
dk
tr(Πkk)) = 2
2
∑
k=1
µkαkδkwk
(C.13)
Combining the linear and quadratic traces given by (C.5), (C.13) and (C.12), the complete rigorous bias
evaluates to:
lim
d→∞
1
d∥MS∥2
F =
2
∑
k=1
µk [α2
k(1 −wk) + (αk + δk)2wk] + T12(˜δ2
2 −˜δ2
1)(Σ2
1 −Σ2
2).
(C.14)
In the next lemma, we characterize T12 which completes our analysis of the Bias term.
Lemma C.2 Let ΠM be the orthogonal projection matrix onto the column space of M = AΩ, where Ω∈Rd×B
has i.i.d. entries of variance 1/d, and A is a deterministic block-diagonal matrix with block dimensions
dk = µkd and corresponding squared eigenvalues a2
k for k ∈{1,2}. Let Πij denote the sub-blocks of ΠM. As
21

--- Page 22 ---
d,B →∞with B/d →β, the normalized cross-subspace leakage trace T12 = limd→∞1
dtr(Π12Π21) is almost
surely given by:
T12 =
µ1µ2v1v2
µ1v1 + µ2v2
(C.15)
where vk = wk(1 −wk) is the variance factor of the projection on block k, and wk =
a2
kq
1+a2
kq are the Stieltjes
weights defined by the fixed-point root q.
Analysis of the noise term. We recall the dimension ratios as µ1 = m/d and γ = d/n. The noise operator
acting on the test covariance is defined exactly as VNΣ = −VSEM †Σ. We seek the limit of the normalized
expected squared Frobenius norm:
1
dEE [∥VNΣ∥2
F ] = 1
dEE [tr(VSEM †Σ2(M †)T ET V T
S )]
(C.16)
Let Q = M †Σ2(M †)T . Because M = E[Φ] is deterministic, Q is constant with respect to the noise realization
E.
Let ετ = (Sτ −A)w∗
τ be the τ-th column of E. We first compute the expectation over the feature samples
xi,τ conditioned on the weight matrix Ω. Since the feature samples are independent across different weights
τ, the columns of E are mutually independent with zero mean:
E[ετε⊺
γ ∣Ω] = 0
for τ ≠γ
For the diagonal terms, we use the standard identity for the covariance of a Wishart quadratic form. For any
fixed vector u and S ∼Wd(n, 1
nA):
E[(S −A)uu⊺(S −A)] = 1
n ((u⊺Au)A + Auu⊺A)
Summing over the entries of Q:
EE∣Ω[EQE⊺] = ∑
τ≠γ
QτγE[ετε⊺
γ ∣Ω] = 1
n
B
∑
τ=1
Qττ ((w∗⊺
τ Aw∗
τ)A + Aw∗
τw∗⊺
τ A)
We now take the expectation over Ω. By the rotational invariance of the Gaussian distribution, the
expectation of any function of Ωthat is equivariant under orthogonal transformations must be isotropic. In
particular, the term Z = E[∑τ Qττw∗
τw∗⊺
τ ] must satisfy Z = cId. Taking the trace:
cd = E[
B
∑
τ=1
Qττ∥w∗
τ∥2] = dE[tr(Q)] Ô⇒c = E[tr(Q)]
In the high-dimensional limit, the correlation between the weight-norm quadratic form (w∗⊺
τ Aw∗
τ) and the
kernel diagonal Qττ vanishes, by which we obtain
E[
B
∑
τ=1
Qττ(w∗⊺
τ Aw∗
τ)] = E[tr(Q)]tr(A)
Combining these, we obtain:
E[EQE⊺] = E[tr(Q)]
n
(tr(A)A + A2)
(C.17)
We set ΣA ∶= 1
d (tr(A)A + A2). Substituting the above identity into (C.16) we obtain:
1
dEE [∥VNΣ∥2
F ] = 1
dtr(VS [γtr(Q)ΣA]V T
S )
(C.18)
22

--- Page 23 ---
Because tr(Q) is a scalar, it factors entirely out of the matrix product. Using the property (M †)T M † =
(MM T )†, the expectation rigorously splits into the product of two independent, normalized trace functionals:
1
dE[∥VNΣ∥2
F ] = γ [tr(Σ2(MM T )†)]
´¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¸¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¶
Tinv
⋅[1
dtr(VSΣAV T
S )]
´¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¸¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¶
Tvar
(C.19)
Derivation of the pseudo-inverse trace (Tinv). We must evaluate the target trace Tinv = tr(Σ2(MM T )†),
which governs the variance functional. Because the feature matrix M = AΩis constructed with Ω∼N(0,1)
i.i.d. entries, the unscaled matrix G = MM T has eigenvalues scaling as O(d). To rigorously apply the Stieltjes
transform, we define the normalized matrix ˆG = 1
dG, which has O(1) eigenvalues. The target trace scales as:
Tinv = tr(Σ2(d ˆG)†) = 1
dtr(Σ2 ˆG†)
(C.20)
●Over-parameterized regime (β < 1). Because ˆG is strictly singular in the over-parameterized regime
(B < d), direct inversion is invalid. To evaluate the trace rigorously for any aspect ratio β, we introduce a
strictly positive regularization parameter z > 0 and define the perturbed resolvent:
R(z,t) = ( ˆG + tAΣ2A + zId)−1
(C.21)
Let m(z,t) = 1
dtr(R(z,t)) be its normalized trace. Because z > 0, R(z,t) is unconditionally invertible and
bounded for all B. Taking the derivative of m(z,t) with respect to the continuous perturbation t at t = 0
yields:
∂
∂tm(z,t)∣
t=0 = −1
dtr(R(z,0)(AΣ2A)R(z,0))
(C.22)
Because R(z,0), A, and Σ are all well-defined, finite d × d matrices, we can validly apply cyclic permutation
to the trace. We move the rightmost R(z,0) to the left, and use the fact that the diagonal matrices A and
Σ2 commute (AΣ2A = Σ2A2):
∂
∂tm(z,t)∣
t=0 = −1
dtr((AΣ2A)R(z,0)2) = −1
dtr(Σ2A2R(z,0)2)
(C.23)
We now define our target Stieltjes derivative q′(0) as the limit of this regularized derivative as z →0+. By
defining the operator limit limz→0+ R(z,0)2 ≡( ˆG†)2 strictly on the non-null subspace, this identically maps
to our target variance trace Tinv across all parameterization regimes:
−q′(0) = lim
z→0+
1
dtr(Σ2A2R(z,0)2) ≡1
dtr(Σ2A2( ˆG†)2) = Tinv
(C.24)
To find q′(0) analytically, we differentiate the fixed-point equation of the perturbed resolvent. The eigenvalues
of the perturbed deterministic envelope are a2
k(1 + tΣ2
k). Using the Silverstein equation, we have the following
fixed-point equation:
β =
2
∑
k=1
µk
a2
k(1 + tΣ2
k)q(t)
1 + a2
k(1 + tΣ2
k)q(t)
(C.25)
Differentiating both sides with respect to t at t = 0 (where q(0) = q) gives:
0 =
2
∑
k=1
µk
a2
kΣ2
kq + a2
kq′(0)
(1 + a2
kq)2
(C.26)
Separating the terms and recognizing that the effective block weights are wk =
a2
kq
1+a2
kq, we observe the algebraic
identity
a2
k
(1+a2
kq)2 =
w2
k
a2
kq2 . Substituting this into the differential equation gives:
−q′(0)
2
∑
k=1
µk
w2
k
a2
kq2 = q
2
∑
k=1
µkΣ2
k
w2
k
a2
kq2
(C.27)
23

--- Page 24 ---
Multiplying by q2 and isolating −q′(0), we obtain the exact closed-form limit:
Tinv = q
∑2
k=1 µk
Σ2
k
a2
k w2
k
∑2
k=1 µk 1
a2
k w2
k
(C.28)
●Under-parameterized regime (β > 1). The differential Stieltjes approach relies on the fixed-point root
q being finite, which holds strictly for the over-parameterized regime (β < 1). When β > 1, the number
of samples exceeds the ambient dimension (B > d), causing the rank fraction to saturate at ¯β = 1, which
mathematically drives q →∞.
However, in this over-parameterized regime, the unscaled feature covariance matrix G = MM T becomes
strictly full rank almost surely. Consequently, the normalized matrix ˆG = 1
dG is strictly invertible, and its
pseudoinverse reduces to the standard inverse ˆG−1. We skip the perturbation derivative and evaluate the
trace directly using the deterministic equivalent for the inverse of a generalized sample covariance matrix.
Note that ˆG−1 = A−1W −1A−1 with W = 1
dΩΩT a standard Wishart matrix of size d × B and so by the inverse
moments of the Marchenko-Pastur law, its deterministic equivalent is given by W ≍
1
β−1Id, which implies that
ˆG−1 ≍
1
β −1(A2)−1
(C.29)
Substituting this deterministic equivalent directly into the target trace functional yields the exact closed-form
limit for β > 1:
Tinv = 1
dtr(Σ2 [
1
β −1A−2]) =
1
β −1
2
∑
k=1
µk
Σ2
k
a2
k
(C.30)
Equations (C.28) and (C.30) both diverge at the interpolation threshold (β = 1).
We combine both equation into one unifying relation:
Tinv =
⎧⎪⎪⎪⎨⎪⎪⎪⎩
q
∑2
k=1 µk
Σ2
k
a2
k w2
k
∑2
k=1 µk 1
a2
k w2
k
⎫⎪⎪⎪⎬⎪⎪⎪⎭
1(β < 1) + {
1
β −1
2
∑
k=1
µk
Σ2
k
a2
k
}1(β > 1)
(C.31)
Derivation of the trace term (Tvar). We evaluate the trace term Tvar = limd→∞1
dtr(VSΣAV T
S ). Recall
the deterministic test operator VS = −Γ−1
0 + DpreΠM, where Dpre = Γ−1
0 −ηA−1. Note that Dpre and −Γ−1
0
are both block-diagonal deterministic matrices. Also their respective scalar values on block k ∈{1,2} in the
proportional asymptotic regime converges to ˜δk and gk given by (B.7) and (B.8). Expanding the trace yields:
Tvar = 1
dtr(Γ−2
0 ΣA) + 2
dtr(−Γ−1
0 ΣADpreΠM) + 1
dtr(DpreΠMΣADpreΠM)
(C.32)
Let Tij = 1
dtr(ΠijΠji). The linear traces evaluate strictly on the diagonal blocks. Similar to derivations (C.5)
and (C.13) we have
lim
d→∞tr(Γ−2
0 ΣA) =
2
∑
k=1
µkg2
ksk
where s1 and s2 are the limit of the scalar on the blocks of ΣA given by (B.9). In addition,
lim
d→∞
2
dtr(−Γ−1
0 ΣADpreΠM) = 2
2
∑
k=1
µkgk˜δkskwk
The quadratic trace Quad ∶= 1
dtr(DpreΠMΣADpreΠM) expands over the 2 × 2 block partition as:
Quad = ˜δ2
1s1T11 + ˜δ2
2s2T22 + ˜δ1˜δ2(s1 + s2)T12
(C.33)
24

--- Page 25 ---
Invoking (C.11), we have Tkk = µkwk −T12. Substituting these constraints into the quadratic expansion yields:
Quad =
2
∑
k=1
µk˜δ2
kskwk −T12 [˜δ2
1s1 + ˜δ2
2s2 −˜δ1˜δ2s1 −˜δ1˜δ2s2]
(C.34)
The bracketed multiplier for T12 factors analytically into (˜δ1 −˜δ2)(˜δ1s1 −˜δ2s2). Recombining the linear and
quadratic components completes the square for the diagonal elements, yielding:
Tvar =
2
∑
k=1
µksk [g2
k(1 −wk) + (gk + ˜δk)2wk] −T12(˜δ1 −˜δ2)(˜δ1s1 −˜δ2s2)
(C.35)
By recalling (C.19), the noise limit 1
dEE [∥VNΣ∥2
F ] is given by the product of equations (C.28) and (C.35).
C.2
Analysis of Term II
Since tr(Σ) scales as O(d), the tr(̃V Σ̃V T)tr(Σ) term dominates the tr(̃V Σ2̃V T) term in the high-dimensional
limit. Letting ¯Σ = limd→∞1
dtr(Σ), the dominant component of Term II evaluates to:
Term II = γ ¯Σ ⋅1
dE[tr( ˜V Σ ˜V T )]
(C.36)
Recall that ˜V = VS +VN with VN zero mean. Since the cross-terms are zero, we get the following decomposition:
Term II = γ ¯Σ1
dtr(VSΣV T
S )
´¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¸¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¶
Term II Signal
+γ ¯Σ1
dEE [tr(VNΣV T
N )]
´¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¸¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¶
Term II Noise
(C.37)
Derivation of Term II Signal.
We evaluate Tvar,Σ =
1
dtr(VSΣV T
S ).
The deterministic operator is
VS = −Γ−1
0 + DpreΠM, where Dpre = Γ−1
0 −ηA−1 are block-diagonal. Expanding the trace yields:
Tvar,Σ = 1
dtr(Γ−2
0 Σ) + 2
dtr(−Γ−1
0 ΣDpreΠM) + 1
dtr(DpreΠMΣDpreΠM)
(C.38)
As we observe the expression for Tvar,Σ is same as Tvar with ΣA replaced by Σ. Hence, by a similar derivation
of (C.35) we get
Tvar,Σ =
2
∑
k=1
µkΣk [g2
k(1 −wk) + (gk + ˜δk)2wk] −T12(˜δ1 −˜δ2)(˜δ1Σ1 −˜δ2Σ2)
(C.39)
Derivation of Term II Noise. We must evaluate 1
dE[tr(VNΣV T
N )]. Following similar derivation of (C.19),
replacing Σ by Σ1/2, we arrive at
1
dE[∥VNΣ1/2∥2
F ] = γ [tr(Σ(MM T )†)]
´¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¸¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¶
Tinv,Σ
⋅[1
dtr(VS ˆΣEV T
S )]
´¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¸¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¶
Tvar
(C.40)
Notice that we already characterized Tvar in the analysis of Term I.
We next evaluate Tinv,Σ = limd→∞1
dtr(ΣG†), where G = MM † = AΩΩT A. Note that the expression for
Tinv,Σ is same as Tinv where Σ2 is replaced by Σ. Following the same derivation for (C.31), we arrive at
Tinv,Σ =
⎧⎪⎪⎪⎨⎪⎪⎪⎩
q
∑2
k=1 µk
Σk
a2
k w2
k
∑2
k=1 µk 1
a2
k w2
k
⎫⎪⎪⎪⎬⎪⎪⎪⎭
1(β < 1) + {
1
β −1
2
∑
k=1
µk
Σk
a2
k
}1(β > 1)
(C.41)
25

--- Page 26 ---
Combining the above characterizations, the limit for the components of Term II are given by:
Term II Signal = γ ¯Σ ⋅Tvar,Σ
(C.42)
Term II Noise = γ2 ¯Σ ⋅Tinv,Σ ⋅Tvar
(C.43)
where Tvar,Σ is given by Eq. (C.39), Tinv,Σ by Eq. (C.41), and Tvar is given by (C.35) from the Term I
derivation. Putting the characterizations derived for Term I and Term II in (B.5) completes the proof.
C.2.1
Proof of Lemma C.1
To evaluate the asymptotic trace of ΠM, we express the orthogonal projection operator onto the column
space of the empirical feature matrix M as the limit of a Ridge-regularized inverse as the regularization
parameter z →0+:
ΠM = lim
z→0+ M(M T M + zIB)−1M T = Id −lim
z→0+ z(G + zId)−1
(C.44)
where G = MM T = AΩΩT A is the generalized sample covariance matrix, and R(z) = (G + zId)−1 is its
resolvent.
By the Bai-Silverstein theorem, as d,B →∞with B/d →β, the random resolvent R(z) is asymptotically
equivalent to a deterministic diagonal matrix T(z). For any bounded deterministic matrix D, the normalized
trace converges almost surely:
lim
d→∞
1
dtr(DR(z)) −1
dtr(DT(z))
a.s.
ÐÐ→0
(C.45)
where T(z) is given by
T(z) = (zId + v(z)A2)
−1 ,
and v(z) is the Stieltjes transform of the companion matrix ˜G = ΩT A2Ω.
We define the effective rank fraction preserved in the k-th block as the normalized trace of the projection
matrix restricted to that subspace:
wk = lim
d→∞
1
dk
tr(Πkk)
(C.46)
Substituting the resolvent limit and its deterministic equivalent T(z):
wk = 1 −lim
z→0+
1
dk
∑
i∈Block k
zTii(z)
= 1 −lim
z→0+
z
z + a2
kv(z)
= 1 −lim
z→0+
1
1 + a2
k
v(z)
z
(C.47)
We define the strict Stieltjes fixed-point root q as the limit of this ratio near the origin:
q = lim
z→0+
v(z)
z
(C.48)
Substituting q into the limit yields the following relation for the block weights:
wk = 1 −
1
1 + a2
kq =
a2
kq
1 + a2
kq
(C.49)
To determine the fixed-point root q, we utilize the trace identity between the resolvents of the d × d
generalized sample covariance matrix G and its B × B companion matrix ˜G = ΩT A2Ω. Because their non-
zero eigenvalues are strictly identical, the normalized trace of the feature resolvent, m(z) = 1
dtr(R(z)) =
1
dtr[(G + zId)−1], is given by:
zm(z) = 1 −β + βzv(z).
(C.50)
26

--- Page 27 ---
By the Bai-Silverstein theorem, m(z) is asymptotically equivalent to the trace of the deterministic matrix
T(z). Substituting this deterministic equivalent yields:
m(z) =
K
∑
k=1
µk
1
z + a2
kv(z)
(C.51)
Multiplying by z and equating this to the trace identity (C.50) establishes the exact relation:
1 −β + βzv(z) =
K
∑
k=1
µk
z
z + a2
kv(z) =
K
∑
k=1
µk
1
1 + a2
k
v(z)
z
(C.52)
We evaluate the strict limit of this equation as z →0+. On the right side, we substitute our definition of the
root q = limz→0+ v(z)
z . On the left side, the limit of zv(z) is governed by the dimension of the null space of
the companion matrix ˜G. The maximum rank of ˜G is bounded by d. If B > d (i.e., β > 1), the companion
matrix has exactly B −d strict zero eigenvalues and the resolvent trace scales proportionally to B−d
B
1
z. We
therefore evaluate the limit exactly as:
lim
z→0+ zv(z) = max(1 −1
β ,0)
(C.53)
Substituting these limits into both sides of the trace identity yields:
1 −β + β max(1 −1
β ,0) =
K
∑
k=1
µk
1
1 + a2
kq
(C.54)
The left side mathematically simplifies exactly to max(1−β,0). On the right side, we substitute the definition
of the block weights wk =
a2
kq
1+a2
kq, utilizing the identity
1
1+a2
kq = 1 −wk:
max(1 −β,0) =
K
∑
k=1
µk(1 −wk) = 1 −
K
∑
k=1
µkwk
(C.55)
Rearranging the terms immediately yields:
K
∑
k=1
µkwk = 1 −max(1 −β,0) = min(β,1) = ¯β
(C.56)
This derivation holds universally across all parameterization regimes. In the under-parameterized regime
(β > 1), the effective rank fraction saturates at ¯β = 1, which mathematically forces wk →1 and q →∞.
Equations (C.49) and (C.56) completely and deterministically parameterize the finite-dimensional traces of
the random projection ΠM.
C.2.2
Proof of Lemma C.2
We express the orthogonal projection matrix ΠM as the limit of the regularized resolvent R(z) = (AΩΩT A +
zId)−1 as z →0+:
ΠM = Id −lim
z→0+ zR(z)
(C.57)
Let D1 and D2 be the orthogonal block indicator matrices for subspaces 1 and 2, such that D1D2 = 0.
Specifically,
D1 = [
Im
0m×(d−m)
0(d−m)×m
0d−m
],
D2 = [
0m
0m×(d−m)
0(d−m)×m
Id−m
]
(C.58)
27

--- Page 28 ---
The cross-trace can be written as tr(Π12Π21) = tr(D1ΠMD2ΠM). Substituting the resolvent limit into the
trace definition yields:
T12 = lim
z→0+ lim
d→∞
1
dtr(D1(Id −zR(z))D2(Id −zR(z)))
(C.59)
Because D1D2 = 0, expanding the product causes all terms of order lower than R(z)2 to vanish exactly:
T12 = lim
z→0+ z2 [ lim
d→∞
1
dtr(D1R(z)D2R(z))]
(C.60)
In the next lemma, we characterize the inner limit.
Lemma C.3 Let R(z) = (AΩΩT A + zId)−1 be the resolvent of the generalized sample covariance matrix, and
let T(z) = (zId + v(z)A2)−1 be its deterministic equivalent. Let D1 and D2 be d × d diagonal orthogonal block
indicator matrices such that D1D2 = 0. In the high-dimensional limit d,B →∞with B/d →β, the normalized
trace of the product of the two resolvents converges almost surely to:
lim
d→∞
1
dtr(D1R(z)D2R(z)) = v(z)2
β
Ψ1(z)Ψ2(z)
∆(z)
(C.61)
where Ψk and ∆k are defined as:
Ψk(z) = lim
d→∞
1
dtr(DkA2T(z)2) ,
∆(z) = 1 −v(z)2
β
lim
d→∞
1
dtr(A4T(z)2) .
Using Lemma C.3, we have
Ψk(z) = lim
d→∞
1
dtr(DkT(z)A2T(z)) =
µka2
k
(z + a2
kv(z))2
(C.62)
∆(z) = 1 −v(z)2
β
lim
d→∞
1
dtr(A4T(z)2) = 1 −v(z)2
β
2
∑
k=1
µka4
k
(z + a2
kv(z))2
(C.63)
We now evaluate the limit as z →0+. Using the Stieltjes fixed-point definition q = limz→0+ v(z)
z , we have
v(z) = qz + o(z).
First, we evaluate the limit of the scaled block traces z2Ψk(z):
lim
z→0+ z2Ψk(z) = lim
z→0+
z2µka2
k
z2(1 + a2
kq)2 =
µka2
k
(1 + a2
kq)2
(C.64)
Recall that wk =
a2
kq
1+a2
kq, which implies the variance factor is vk = wk(1 −wk) =
a2
kq
(1+a2
kq)2 . Dividing by q, we
map the block trace exactly to the variance factor:
lim
z→0+ z2Ψk(z) = µkvk
q
(C.65)
Second, we use (C.61) to evaluate T12 given by (C.60). Distributing the z2 multiplier from the projection
limit alongside the v(z)2/z2 →q2 convergence yields:
lim
z→0+
1
β (v(z)2
z2
)[z2Ψ1(z)][z2Ψ2(z)] = 1
β (q2)(µ1v1
q
)(µ2v2
q
) = µ1µ2v1v2
β
(C.66)
Third, we evaluate the denominator ∆(0) as z →0+:
∆(0) = lim
z→0+ [1 −1
β (v(z)2
z2
)
2
∑
k=1
µka4
k
(1 + a2
kq)2 ] = 1 −1
β
2
∑
k=1
µk
a4
kq2
(1 + a2
kq)2
(C.67)
28

--- Page 29 ---
Recognizing the squared weight w2
k = ( a2
kq
1+a2
kq)
2
, we get ∆(0) = 1 −1
β ∑2
k=1 µkw2
k. We apply the identity
β = ∑2
k=1 µkwk, given by (C.56), to replace the leading 1:
∆(0) = ∑2
k=1 µkwk −∑2
k=1 µkw2
k
β
= 1
β
2
∑
k=1
µkwk(1 −wk) = µ1v1 + µ2v2
β
(C.68)
Finally, taking the ratio of the evaluated numerator and denominator, we get
T12 =
µ1µ2v1v2
β
µ1v1+µ2v2
β
=
µ1µ2v1v2
µ1v1 + µ2v2
(C.69)
which concludes the proof.
C.2.3
Proof of Lemma C.3
We evaluate the cross-trace by introducing a continuous, deterministic perturbation t to the resolvent. We
define the perturbed resolvent matrix as R(z,t) = (AΩΩT A + tD2 + zId)−1. Let m1(z,t) = 1
dtr(D1R(z,t)) be
its normalized trace on the first subspace.
Taking the derivative of the random trace m1(z,t) with respect to the perturbation t at t = 0 directly
yields the target cross-trace. Using the matrix derivative identity
∂
∂tM −1 = −M −1 ∂M
∂t M −1:
∂
∂tm1(z,t)∣
t=0 = −1
dtr(D1R(z,0)D2R(z,0)) = −1
dtr(D1R(z)D2R(z))
(C.70)
By the Bai-Silverstein theorem, R(z,t) is asymptotically equivalent to the perturbed deterministic matrix
T(z,t). Because the perturbation tD2 simply shifts the diagonal, the perturbed Stieltjes root v(z,t) enforces
the following exact structural form for the deterministic equivalent:
T(z,t) = (zId + v(z,t)A2 + tD2)−1
(C.71)
Taking the derivative of the deterministic trace ¯m1(z,t) = 1
dtr(D1T(z,t)) at t = 0 gives:
¯m′
1(0) = −1
dtr(D1T(z)[v′(0)A2 + D2]T(z))
(C.72)
Because D1 and D2 are strictly orthogonal (D1D2 = 0) and T(z) is diagonal, the terms commute and the D2
cross-term becomes zero (D1T(z)D2T(z) = 0). Therefore,
¯m′
1(0) = −v′(0)[1
dtr(D1A2T(z)2)] = −v′(0)Ψ1(z)
(C.73)
To evaluate the scalar derivative v′(0), we must construct the fixed-point equation for the perturbed root
v(z,t). Also by the Silverstein equation (Silverstein, 1995), we have
1
v(z,t) = z + 1
βdtr(A2T(z,t))
(C.74)
We differentiate both sides of this fixed-point equation with respect to t at t = 0:
−v′(0)
v(z)2 = 1
βdtr(A2 ∂
∂tT(z,t)∣
t=0) = −1
βdtr(A2T(z)[v′(0)A2 + D2]T(z))
(C.75)
Distributing the trace operator linearly across the sum yields:
−v′(0)
v(z)2 = −v′(0)
βd tr(A4T(z)2) −1
βdtr(D2A2T(z)2)
(C.76)
29

--- Page 30 ---
Multiplying both sides by −v(z)2 and substituting the definition Ψ2(z) = 1
dtr(D2A2T(z)2) gives:
v′(0) = v′(0)v(z)2
βd tr(A4T(z)2) + v(z)2
β
Ψ2(z)
(C.77)
Grouping the v′(0) terms on the left side exposes the exact macroscopic fluctuation denominator ∆(z):
v′(0)[1 −v(z)2
βd tr(A4T(z)2)]
´¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¸¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¶
∆(z)
= v(z)2
β
Ψ2(z) Ô⇒v′(0) = v(z)2
β
Ψ2(z)
∆(z)
(C.78)
Because the asymptotic limit of the random trace derivative (C.70) equals the deterministic trace derivative
(C.73), we substitute the analytical solution for v′(0) into the equivalence −lim 1
dtr(D1RD2R) = −v′(0)Ψ1(z).
The negative signs cancel, yielding the exact closed-form limit:
lim
d→∞
1
dtr(D1R(z)D2R(z)) = v(z)2
β
Ψ1(z)Ψ2(z)
∆(z)
(C.79)
which concludes the proof.
D
Proof of Proposition B.2
(i) As β →1+, the terms Tinv and Tinv,Σ diverge clearly due to the term 1/(β −1) in (C.28), (C.41). Also
as β →1−, then q →∞and so Tinv and Tinv,Σ diverge. This shows that limβ→1 F(β) = ∞.
We next note that for β ≥1, the model definitions dictate that wk = 1, vk = 0, and T12 = 0. We substitute
these constants into the components of F(β):
Bias(β) =
2
∑
k=1
µk(αk + δk)2
Tvar(β) =
2
∑
k=1
µksk(gk + ˜δk)2
Tvar,Σ(β) =
2
∑
k=1
µkΣk(gk + ˜δk)2
Tinv(β) =
1
β −1
2
∑
k=1
µk
Σ2
k
a2
k
Tinv,Σ(β) =
1
β −1
2
∑
k=1
µk
Σk
a2
k
By substituting these components into the objective function F(β), we can write it in the form:
F(β) = C1 + C2
β −1
where C1 and C2 are finite, strictly positive constants independent of β. The derivative is F ′(β) = −
C2
(β−1)2 .
Since C2 > 0, we have F ′(β) < 0, proving F(β) is strictly decreasing for β > 1. As β →∞, the term
C2
β−1 →0. The function converges to the constant C1 given by: F(↑∞):
F(↑∞) =
2
∑
k=1
µk(αk + δk)2 + γ ¯Σ
2
∑
k=1
µkΣk(gk + ˜δk)2
30

--- Page 31 ---
(ii) At β = 0, the implicit variable q = 0, which implies wk = 0, vk = 0, and T12 = 0. Furthermore, the leading
q multiplier in Tinv and Tinv,Σ sets both inverse trace terms exactly to zero. Substituting these into
F(β) eliminates all cross-terms, yielding:
F(0) =
2
∑
k=1
µkα2
k + γ ¯Σ
2
∑
k=1
µkΣkg2
k
Next, we evaluate the gap ∆F = F(↑∞) −F(0):
∆F =
2
∑
k=1
µk [(αk + δk)2 −α2
k] + γ ¯Σ
2
∑
k=1
µkΣk [(gk + ˜δk)2 −g2
k]
Based on the parameter definitions, a2 = r and δ2 = ˜δ2 =
1
1+κ −η
r . Expanding the squared perturbations
for k = 2 yields:
(α2 + δ2)2 −α2
2 = (1 −η
r )
2
−α2
2 = η2
r2 −2η
r + 1 −α2
2
(g2 + ˜δ2)2 −g2
2 = (−η
r )
2
−g2
2 = η2
r2 −g2
2
Substituting these into ∆F, the leading-order behavior as r →0+ is dominated by the 1/r2 terms:
∆F = η2
r2 µ2 (1 + γ ¯Σ) + O (1
r )
Because µ2(1+γ ¯Σ)η2 > 0, the gap diverges to positive infinity as r →0+. Thus, there exists a sufficiently
small r > 0 such that ∆F > 0, or F(↑∞) > F(0).
(iii) We next calculate F ′(0) = dF
dβ ∣
β=0. The asymptotic test error in the β < 1 regime is given by:
F(β) = Bias + γTinvTvar + γ ¯ΣTvar,Σ + γ2 ¯ΣTinv,ΣTvar
By applying the product rule with respect to β, the full derivative is:
F ′(β) = Bias′ + γ (T ′
invTvar + TinvT ′
var) + γ ¯ΣT ′
var,Σ + γ2 ¯Σ(T ′
inv,ΣTvar + Tinv,ΣT ′
var)
To evaluate this at β = 0, we must look at the inverse trace terms. Both Tinv and Tinv,Σ are defined with
a leading factor of q. When β →0, the implicit root q →0. Because the fraction following q converges
to a finite constant as q →0, we have exactly:
Tinv(0) = 0
and
Tinv,Σ(0) = 0
Substituting these zeros into the product rule eliminates the T ′
var(0) terms entirely. The derivative
simplifies to:
F ′(0) = Bias′(0) + γT ′
inv(0)Tvar(0) + γ ¯ΣT ′
var,Σ(0) + γ2 ¯ΣT ′
inv,Σ(0)Tvar(0)
We define the rightmost terms collectively as the variance penalty V (γ):
V (γ) ∶= γ(T ′
inv(0)Tvar(0) + ¯ΣT ′
var,Σ(0)) + γ2(¯ΣT ′
inv,Σ(0)Tvar(0))
Hence, F ′(0) = Bias′(0) + V (γ). Because all the terms in V (γ) are finite for any strictly positive r > 0,
the derivatives evaluated at β = 0 are all finite constants. Since every term in V (γ) is scaled by either γ
or γ2, we have:
lim
γ→0V (γ) = 0
31

--- Page 32 ---
We next derive Bias′(0). Recall the definition of Bias given by:
Bias(β) =
2
∑
k=1
µk [α2
k(1 −wk) + (αk + δk)2wk] + T12(˜δ2
2 −˜δ2
1)(Σ2
1 −Σ2
2)
By expanding the inner bracket and grouping the wk terms, we obtain:
α2
k −α2
kwk + (α2
k + 2αkδk + δ2
k)wk = α2
k + wk(2αkδk + δ2
k)
This gives the reformulated Bias equation:
Bias(β) =
2
∑
k=1
µkα2
k +
2
∑
k=1
µkwk(2αkδk + δ2
k) + T12(˜δ2
2 −˜δ2
1)(Σ2
1 −Σ2
2)
To differentiate this with respect to β, we apply the chain rule via the implicit variable q. First, define
the constant c = µ1a2
1 + µ2a2
2. From the defining equation β(q) = ∑2
k=1 µk
a2
kq
1+a2
kq, we take the derivative
with respect to q:
dβ
dq =
2
∑
k=1
µk
a2
k
(1 + a2
kq)2
Evaluating at q = 0 gives
dβ
dq ∣
0 = µ1a2
1 + µ2a2
2 = c. By the inverse function theorem, q′(0) =
dq
dβ ∣
0 = 1
c.
Now we sequentially compute the initial derivatives of the sub-components wk and T12:
Since wk =
a2
kq
1+a2
kq, the chain rule yields w′
k(0) = a2
kq′(0) = a2
k
c . In addition, for small q, the variables
vk = wk(1 −wk) expand to first order as vk = a2
kq + O(q2). Substituting this into the definition of T12
gives:
T12(q) =
µ1µ2(a2
1q)(a2
2q)
µ1(a2
1q) + µ2(a2
2q) + O(q2) = q µ1µ2a2
1a2
2
c
+ O(q2)
Taking the derivative with respect to β evaluates to T ′
12(0) = q′(0) µ1µ2a2
1a2
2
c
= µ1µ2a2
1a2
2
c2
. Finally, we
substitute w′
k(0) and T ′
12(0) directly into the differentiated Bias equation:
Bias′(0) =
2
∑
k=1
µkw′
k(0)(2αkδk + δ2
k) + T ′
12(0)(˜δ2
2 −˜δ2
1)(Σ2
1 −Σ2
2)
= 1
c
2
∑
k=1
µka2
k(2αkδk + δ2
k) + µ1µ2a2
1a2
2
c2
(˜δ2
2 −˜δ2
1)(Σ2
1 −Σ2
2)
For the first class (k = 1), the definitions give δ1 = −α1, resulting in 2α1δ1 + δ2
1 = −α2
1. For the second
class (k = 2), as r →0+, the term a2
2(2α2δ2 + δ2
2) →r2(η2/r2) = η2. The cross-term converges to
µ2
µ1a2
1 η2((ρ + 1)2 −1). Summing these asymptotic components gives:
lim
r→0+ Bias′(0) =
1
µ1a2
1
[µ1a2
1(−α2
1) + µ2η2] + µ2η2
µ1a2
1
((ρ + 1)2 −1)
= −α2
1 + µ2η2
µ1a2
1
(ρ + 1)2
Since a2
1 = η2(ρ + 1)2, this simplifies to:
lim
r→0+ Bias′(0) = −α2
1 + µ2
µ1
32

--- Page 33 ---
Recall that α1 = (κ −1)/(ρ + κ) and κ = γ(µρ + 1 −µ). Hence
d
dγ α2
1(γ)∣
0 < 0. Also,
α2
1(0) = 1
ρ2 ≥1 −µ1
µ1
= µ2
µ1
,
by our assumption. By continuity, for small enough γ, we have α2
1 ≥µ2/µ1 and so we have limr→0+ Bias′(0) <
0. Because Bias′(0) is strictly negative for small r, and the variance penalty V (γ) can be made arbi-
trarily small for small γ, there must exist constants r0 > 0 and γ0 > 0 such that for all r < r0 and γ < γ0,
we have F ′(0) < 0. This completes the proof of the proposition.
E
Gradient and Hessian Calculations for Outcome Supervision (OS)
Loss
For clarity, let M = I + V S (dropping the index τ for a single batch) and define the loss as f(V ) = 1
2∥M kw∗∥2.
We use the differential approach. Let dV be a small perturbation in V . Then dM = (dV )S. The
differential of the loss is:
df = ⟨M kw∗,d(M kw∗)⟩= (w∗)T (M k)T d(M k)w∗
Using the power rule for differentials, d(M k) = ∑k−1
j=0 M j(dM)M k−1−j. Substituting dM = (dV )S:
df =
k−1
∑
j=0
(w∗)T (M k)T M j(dV )SM k−1−jw∗
Using the property tr(AT BC) = tr(CAT B), we isolate dV :
df = tr⎛
⎝dV
k−1
∑
j=0
SM k−1−jw∗(w∗)T (M k)T M j⎞
⎠
The gradient ∇V L is the transpose of the matrix multiplying dV :
∇V L =
k−1
∑
j=0
(M T )jM kw∗(w∗)T (M T )k−1−jST
We next proceed to calculate the Hessian of the loss. The gradient can be viewed as a product of terms:
G(V ) = ∑k−1
j=0 Aj(V )M k(V )Bj(V ), with Aj(V ) = (M T)j and Bj(V ) = w∗(w∗)T (M T )k−1−jST . Applying the
product rule for the differential dG:
dG =
k−1
∑
j=0
((dAj)M kBj + Aj(dM k)Bj + AjM k(dBj))
Near the global minimum, the term M kw∗≈0. In this regime, terms containing M k (the outer factors)
vanish, leaving only the term where the differential acts directly on M k. Thus,
dG ≈
k−1
∑
j=0
(M T )j(dM k)w∗(w∗)T (M T )k−1−jST
Substituting for d(M k) = ∑k−1
j=0 M j(dM)M k−1−j, we get
H[dM] ≈
k−1
∑
j=0
k−1
∑
l=0
(M T )j (M l(dM)M k−1−l)w∗(w∗)T (M T )k−1−jST
33

--- Page 34 ---
where for a direction E, we have H[E] = d
dt∇V L(V + tE)∣t=0.
We next upper bound the spectral norm of the Hessian as
∥H∥op ≤
k−1
∑
j=0
k−1
∑
l=0
∥M j∥op ∥M l∥op ∥M k−1−l∥op ∥M k−1−j∥op ∥w∗∥2
ℓ2 ∥S∥op
≤
k−1
∑
j=0
k−1
∑
l=0
∥M∥j
op ∥M∥l
op ∥M∥k−1−l
op
∥M∥k−1−j
op
∥w∗∥2 ∥S∥op
= k2ρ(M)2k−2 ∥w∗∥2
ℓ2 ∥S∥op ,
where the second step follows from sub-multiplicativity of the operator norm.
References
S. N. Akter, S. Prabhumoye, E. Nyberg, M. Patwary, M. Shoeybi, Y. Choi, and B. Catanzaro. Front-loading
reasoning: The synergy between pretraining and post-training data. arXiv preprint arXiv:2510.03264,
2025.
G. Aminian, A. R. Asadi, I. Shenfeld, and Y. Mroueh. Kl-regularized rlhf with multiple reference models:
Exact solutions and sample complexity. In The Thirty-ninth Annual Conference on Neural Information
Processing Systems, 2025.
S. Garg, D. Tsipras, P. S. Liang, and G. Valiant. What can transformers learn in-context? a case study of
simple function classes. Advances in neural information processing systems, 35:30583–30598, 2022.
E. Guha, R. Marten, S. Keh, N. Raoof, G. Smyrnis, H. Bansal, M. Nezhurina, J. Mercat, T. Vu, Z. Sprague,
et al. Openthoughts: Data recipes for reasoning models. arXiv preprint arXiv:2506.04178, 2025.
D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma, P. Wang, X. Bi, et al. Deepseek-r1:
Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948,
2025.
J. Huang, Z. Wang, and J. D. Lee. Transformers learn to implement multi-step gradient descent with chain
of thought. arXiv preprint arXiv:2502.21212, 2025a.
X. Huang, J. Wu, H. Liu, X. Tang, and Y. Zhou. m1: Unleash the potential of test-time scaling for medical
reasoning with large language models. arXiv preprint arXiv:2504.00869, 2025b.
A. Javanmard, B. Mirzasoleiman, and V. Mirrokni. Understanding the role of training data in test-time
scaling. arXiv preprint arXiv:2510.03605, 2025.
J. Li, A. Fang, G. Smyrnis, M. Ivgi, M. Jordan, S. Y. Gadre, H. Bansal, E. Guha, S. S. Keh, K. Arora, et al.
Datacomp-lm: In search of the next generation of training sets for language models. Advances in Neural
Information Processing Systems, 37:14200–14282, 2024.
Meta.
The llama 4 herd:
The beginning of a new era of natively multimodal ai innovation.
https://ai.meta.com/blog/llama-4-multimodal-intelligence/, April 2025.
N. Muennighoff, Z. Yang, W. Shi, X. L. Li, L. Fei-Fei, H. Hajishirzi, L. Zettlemoyer, P. Liang, E. Candès,
and T. B. Hashimoto. s1: Simple test-time scaling. In Proceedings of the 2025 Conference on Empirical
Methods in Natural Language Processing, pages 20286–20332, 2025.
D. Nguyen, W. Yang, R. Anand, Y. Yang, and B. Mirzasoleiman. Mini-batch coresets for memory-efficient
language model training on data mixtures. arXiv preprint arXiv:2407.19580, 2024.
34

--- Page 35 ---
OpenAI. Learning to reason with llms. https://openai.com/index/learning-to-reason-with-llms/, 2024.
A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever, et al. Language models are unsupervised
multitask learners. OpenAI blog, 1(8):9, 2019.
J. W. Silverstein. Strong convergence of the empirical distribution of eigenvalues of large dimensional random
matrices. Journal of Multivariate Analysis, 55(2):331–339, 1995.
A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin.
Attention is all you need. Advances in neural information processing systems, 30, 2017.
S. M. Xie, H. Pham, X. Dong, N. Du, H. Liu, Y. Lu, P. S. Liang, Q. V. Le, T. Ma, and A. W. Yu. Doremi:
Optimizing data mixtures speeds up language model pretraining. Advances in Neural Information Processing
Systems, 36:69798–69818, 2023.
W. Xiong, H. Dong, C. Ye, Z. Wang, H. Zhong, H. Ji, N. Jiang, and T. Zhang. Iterative preference learning
from human feedback: Bridging theory and practice for rlhf under kl-constraint. In Forty-first International
Conference on Machine Learning.
Y. Yang, S. Mishra, J. Chiang, and B. Mirzasoleiman. Smalltolarge (s2l): Scalable data selection for fine-
tuning large language models by summarizing training trajectories of small models. Advances in Neural
Information Processing Systems, 37:83465–83496, 2024.
Y. Yue, Z. Chen, R. Lu, A. Zhao, Z. Wang, S. Song, and G. Huang. Does reinforcement learning really
incentivize reasoning capacity in llms beyond the base model? arXiv preprint arXiv:2504.13837, 2025.
W. Zeng, Y. Huang, Q. Liu, W. Liu, K. He, Z. Ma, and J. He. Simplerl-zoo: Investigating and taming zero
reinforcement learning for open base models in the wild. arXiv preprint arXiv:2503.18892, 2025.
H. Zhao, C. Ye, W. Xiong, Q. Gu, and T. Zhang. Logarithmic regret for online kl-regularized reinforcement
learning. CoRR, 2025.
35
