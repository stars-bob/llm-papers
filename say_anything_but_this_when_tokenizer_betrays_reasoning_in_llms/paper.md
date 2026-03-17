# Say Anything but This: When Tokenizer Betrays Reasoning in LLMs

Source: https://arxiv.org/pdf/2601.14658v1

--- Page 1 ---
Say Anything but This: When Tokenizer Betrays Reasoning in LLMs
Navid Ayoobi , Marcus I Armstrong , Arjun Mukherjee
University of Houston
{nayoobi, miarmstr}@cougarnet.uh.edu, amukher6@central.uh.edu
Abstract
Large language models (LLMs) reason over dis-
crete token ID sequences, yet modern subword to-
kenizers routinely produce non-unique encodings:
multiple token ID sequences can detokenize to
identical surface strings. This representational mis-
match creates an unmeasured fragility wherein rea-
soning processes can fail.
LLMs may treat two
internal representations as distinct “words” even
when they are semantically identical at the text
level.
In this work, we show that tokenization
can betray LLM reasoning through one-to-many to-
ken ID mappings.
We introduce a tokenization-
consistency probe that requires models to replace
designated target words in context while leaving
all other content unchanged.
The task is inten-
tionally simple at the surface level, enabling us
to attribute failures to tokenizer-detokenizer arti-
facts rather than to knowledge gaps or parameter
limitations.
Through analysis of over 11000 re-
placement trials across state-of-the-art open-source
LLMs, we find a non-trivial rate of outputs exhibit
phantom edits: cases where models operate un-
der the illusion of correct reasoning, a phenomenon
arising from tokenizer-induced representational de-
fects.
We further analyze these cases and pro-
vide a taxonomy of eight systematic tokenizer ar-
tifacts, including whitespace-boundary shifts and
intra-word resegmentation.
These findings indi-
cate that part of apparent reasoning deficiency orig-
inates in the tokenizer layer, motivating tokenizer-
level remedies before incurring the cost of training
ever-larger models on ever-larger corpora.
1
Introduction
Recent breakthroughs in large language model (LLM) capa-
bilities are predominantly attributed to scaling Transformer
architectures with massive datasets and computational re-
sources, yielding emergent abilities including reasoning, code
generation, and few-shot generalization [Saunshi et al., 2025;
Kaplan et al., 2020].
However, the prevailing “scale ex-
plains everything” narrative overlooks a foundational com-
ponent that mediates all model input and output: the tok-
enizer [Gastaldi et al., 2025]. Tokenization constitutes the
mandatory translation layer between the continuous, context-
dependent nature of human language and the discrete sym-
bolic representations required for computational processing.
Early tokenizers employed word-level vocabularies, which
suffered from out-of-vocabulary (OOV) limitations, where
any word absent from the training corpus would be mapped
to an unknown token, resulting in information loss [Choo
and Kim, 2023]. This limitation motivated a transition to
open-vocabulary schemes. Accordingly, modern LLMs adopt
subword tokenization including Byte Pair Encoding (BPE)
[Sennrich et al., 2016], WordPiece [Wu et al., 2016], Uni-
gram [Kudo, 2018], and SentencePiece [Kudo and Richard-
son, 2018]. These algorithms decompose words into frequent
substrings or character-level units, substantially solving OOV
and rare word problem. Yet subword tokenization also intro-
duces new challenges. By defining how text is segmented,
the tokenizer can impose structural artifacts on the model’s
internal representation of language. The availability of nu-
merous subtoken units enables different subtoken combina-
tions to concatenate into the same word, establishing repre-
sentational non-uniqueness. The core of our critique centers
on the many-to-one nature of the tokenizer’s mapping from
token ID space to surface text. While the tokenization pro-
cess is designed to be deterministic for a given input, the
detokenization function is not injective. Consequently, mod-
els must learn that different subtoken decompositions of the
same word represent semantically identical content despite
possessing different token-level encodings. Simultaneously,
research findings indicate that scaling model size and data
volume alone does not resolve all reasoning deficiencies in
LLMs.
State-of-the-art models exhibit remarkable perfor-
mance across numerous benchmarks, yet they continue to
exhibit systematic failures on certain reasoning tasks, partic-
ularly as problem complexity increases. This suggests that
some limitations arise from factors beyond model scale, po-
tentially from how information is represented or processed
internally.
In this paper, we argue that tokenization is a hidden con-
tributor to the reasoning failures.
We introduce a simple
yet diagnostic tokenization-consistency probe: models are in-
structed to perform straightforward word substitutions while
preserving all other text unchanged. Success requires identi-
fying marked spans, applying designated edits, and preserv-
arXiv:2601.14658v1  [cs.CL]  21 Jan 2026


--- Page 2 ---
ing unmarked regions.
Under correct operation, an LLM
should recognize that generating alternative token sequences
which decode to the same word does not constitute a genuine
edit and must be avoided. However, our empirical results re-
veal an alarmingly high rate of tokenizer-induced “phantom
edits”. In these cases, models output different token ID se-
quences for target words while the decoded surface text re-
mains identical. The transformation appears successful from
the model’s internal token-based perspective (IDs differ), yet
it is semantically null from the text-based evaluation perspec-
tive (strings match). This demonstrates that models are sys-
tematically misled by tokenizer properties: they “believe”
they have successfully executed substitutions when no actual
content change has occurred. Importantly, these failures do
not reflect knowledge limitations or model scale, but rather
expose a fundamental architectural constraint: the inability
to recognize that different token-level representations can
encode identical linguistic content.
Our analysis of over 11k replacement trials across ten state-
of-the-art open-source LLMs reveals that a non-trivial frac-
tion of reasoning failures stem not from knowledge limi-
tations or model capacity constraints, but from tokenizer-
induced representational non-uniqueness. We further char-
acterize the mechanisms underlying these failures through
a taxonomy of eight systematic artifact types, including
whitespace-boundary shifts, whitespace detachment and reat-
tachment, newline substitutions, intra-word resegmentation,
and proper-noun and morphological segmentation ambigu-
ities.
This taxonomy reveals how standard vocabulary re-
dundancies and alternative segmentation pathways systemat-
ically mislead token-space reasoning, providing a roadmap
for future tokenizer-aware architectural improvements. Ad-
ditionally, through a post-hoc token ID masking interven-
tion, we demonstrate that when these spurious representa-
tional pathways are suppressed, models frequently reveal un-
derlying reasoning capacities that were previously overshad-
owed. This suggests that the reasoning deficiency observed
in state-of-the-art LLMs is partly a mirage, a byproduct of
a mandatory translation layer that is both lossy and non-
injective. Ultimately, this research motivates a paradigm shift
toward more robust representational schemes to ensure that
future language models can reason as reliably at the surface
level as they do within their latent embedding spaces.
2
Related work
2.1
Tokenization in modern LLMs
Subword tokenization techniques such as BPE [Sennrich et
al., 2016], WordPiece [Wu et al., 2016], and Unigram [Kudo,
2018] were developed to address out-of-vocabulary (OOV)
and rare word challenges. These techniques form the tok-
enization backbone of modern state-of-the-art LLMs. Fol-
lowing the release of GPT-4 and the open-sourcing of Ope-
nAI’s tiktoken library, the field broadly converged toward
high-performance BPE variants. Llama 3.2 and Qwen 3 em-
ploy Byte-Level BPE, an algorithm that constructs tokens
from individual bytes to enable lossless encoding of multi-
lingual text and code while mitigating the vocabulary explo-
sion problem inherent in character-level BPE. This approach
is further refined in the Mistral model family, which imple-
ments the “Tekken” tokenizer, a specialized BPE variant op-
timized for compression efficiency on source code and Eu-
ropean languages. In contrast to this BPE-dominated land-
scape, Gemma 3 adopts the Unigram language model im-
plemented via SentencePiece [Kudo and Richardson, 2018],
utilizing an extensive vocabulary exceeding 256k tokens to
support massively multilingual capabilities and enable direct
integration of visual embeddings into the text stream. De-
spite their widespread success, fixed-vocabulary tokenizers
have been criticized as a limiting inductive bias. These limi-
tations have motivated “token-free” approaches that compute
directly on bytes or characters as the atomic units of digital
text [Clark et al., 2022; Xue et al., 2022; Tay et al., 2022].
2.2
Advances in LLM reasoning
LLMs have achieved significant reasoning improvements
[Fang et al., 2024] through structured prompting approaches,
notably Chain-of-Thought (CoT) [Wei et al., 2022; Besta et
al., 2025; Wang et al., 2023] and Zero-Shot CoT [Kojima
et al., 2022], with subsequent work automating and refin-
ing these techniques through methods like Auto-CoT [Zelik-
man et al., 2024; Wei et al., 2022]. Beyond prompting, aug-
menting LLMs with external tools and APIs further enhances
their reasoning capabilities [Gou et al., 2024; Lu et al., 2023;
Dong et al., 2025].
Methods such as ReAct [Yao et al.,
2023] interleave logical reasoning steps with executable ac-
tion commands, while frameworks like Toolformer [Schick et
al., 2023] enable models to invoke external resources includ-
ing calculators, search engines, and specialized APIs, yield-
ing improved performance on factual accuracy, numerical
computation, and zero-shot reasoning tasks. Parallel research
on retrieval-augmented generation [Tran et al., 2025] extends
LLM capabilities by incorporating external knowledge bases
that can be queried dynamically, effectively providing models
with expandable memory beyond their parametric knowledge
[Izacard et al., 2023].
2.3
Tokenization as a reasoning bottleneck
By fragmenting semantic continuity into discrete integer rep-
resentations, tokenizers can constitute a bottleneck in LLM
semantic understanding [Haslett and Cai, 2025]. In quan-
titative reasoning, this appears as structural misalignment:
greedy left-to-right subword tokenization can split integers
and dates into arbitrary, non-semantic fragments, thereby ob-
scuring the positional structure required for arithmetic and
temporal inference [Singh and Strouse, 2024; Bhatia et al.,
2025; Fatemi et al., 2024]. Manipulating digit-grouping tok-
enizations reveal substantial accuracy shifts and altered error
distributions in state-of-the-art LLMs, indicating that model
“reasoning” frequently adheres to tokenizer-imposed struc-
ture rather than learned abstract computational principles.
More broadly, small surface perturbations like typos, spac-
ing changes, and rare Unicode characters can trigger differ-
ent segmentations and produce behavioral changes [Chai et
al., 2024]. Proposed mitigations include exposing models to
multiple valid tokenizations during training [Provilkov et al.,
2020], and designing token-free architectures [Slagle, 2024].


--- Page 3 ---
3
Method
3.1
Task setup overview
LLMs execute instructions by operating over token ID se-
quences produced by tokenizers. Tokenization poses a partic-
ularly strict challenge: the same surface word may be associ-
ated with different token IDs at input encoding versus output
decoding, due to non-unique segmentations that detokenize to
identical text. This property establishes a natural framework
for probing tokenization-consistency reasoning. We define
the task formally: given a text with marked spans and an in-
struction to replace each marked span with an alternative, the
model must 1) identify the target spans, 2) apply the edits, and
3) preserve all non-target text. Our probe evaluates whether
models recognize that different token ID combinations can
decode to identical strings and therefore represent identical
semantics despite divergent internal representations. This set-
ting isolates failures arising from the interaction between the
tokenizer’s discrete alphabet and the model’s learned rules,
thereby evaluating reasoning under the tokenizer’s lossy and
non-unique encoding.
In contrast to standard benchmarks
where reasoning failures may reflect insufficient model pa-
rameters, knowledge gaps, or sampling variance, our task
specifies an unambiguous objective where failures arise be-
cause the model’s internal processing depends on token IDs
that lack one-to-one correspondence with surface text.
Figure 1 depicts the full experimental workflow. The selec-
tor randomly samples p% of non-stop words from each doc-
ument and marks them by placing them inside brackets. We
then prepend an explicit replacement instruction to the docu-
ment and pass the combined text through the tokenizer. The
tokenizer maps the input to a token-ID sequence that serves
as the LLM’s input. Finally, the model’s output token IDs
are detokenized to produce the output string. We evaluate re-
placement performance by comparing input and output texts
together with their aligned token-ID sequences. Using evi-
dence from both the surface form and the token IDs, we cat-
egorize model behavior into three classes: 1) Unchanged,
where the model does not replace the target and the output
token ID matches the input token ID; 2) Replaced, where the
model performs a successful substitution and the output token
ID differs from the input token ID; and 3) Different, where
the model outputs different token ID(s) that nevertheless de-
code to the same word, yielding the same surface string as the
original input word, despite token-id variation. In the exam-
ple shown in Figure 1, the model successfully replaces several
targets (e.g., get →receive), indicated in green. As expected,
successful substitutions are accompanied by changes in the
corresponding token IDs. Conversely, the model failed to re-
place the word “Campaign”, which retains the same token ID
in the output sequence, indicated by blue highlighting. Most
critically, the red color highlights the cases where the model
outputs a different token-ID representation than the input, but
detokenization produces the same surface word in the final
text.
This phenomenon arises because tokenization is not
injective, distinct token-ID sequences can map to an identi-
cal surface form. We provide a more detailed explanation of
how the tokenizers enable such token-level changes without
surface-level edits in Section 4.2.
3.2
Experimental setup
Dataset: we sample news articles from the XSUM dataset
[Narayan et al., 2018] and retain texts between 100 and 600
words. For each selected article, we randomly sample 5% of
non-stop words as replacement targets and designate these
spans by enclosing them in brackets, producing a bracket-
annotated version of the original text.
LLMs receive the
bracket-annotated document along with explicit instructions
to replace only bracketed words while maintaining all other
text in its original form.
LLM models: we evaluate ten state-of-the-art open-source
LLMs spanning four major model families and accessed
through Hugging Face: Gemma (Gemma3-270M, Gemma3-
1B,
Gemma3-4B,
Gemma3-12B),
Llama
(Llama3.2-
1B, Llama3.2-3B, Llama3.1-8B), Mistral (Ministral-8B,
MistralSmall-24B), and Qwen (Qwen3-4B, Qwen3-30B).
To ensure comparability across models, we employ a stan-
dardized sampling configuration for all generation tasks,
and set Top-p, Top-k, and Temperature to 0.9, 50, and 1.0,
respectively.
4
Result and discussion
4.1
Model size is not always the bottleneck for
reasoning
While parameter scaling is often assumed to improve reason-
ing capacity, our experiments show that it is not always suffi-
cient to ensure semantic understanding and reasoning in tar-
geted word-replacement tasks. Figure 2 aggregates outcomes
from over 11k replacement trials, categorizing model perfor-
mance into our three distinct classes (Unchanged, Replaced,
and Different).
If parameter insufficiency were the primary failure mode,
we would anticipate monotonic performance improvements
with model size within a given architecture family. How-
ever, our observations contradict this expectation. Within the
Qwen3 and Gemma3 families, some larger variants achieve
replacement success that is comparable to or even lower than
that of smaller counterparts. Moreover, even in cases where
larger models behave better overall (higher Replaced and
lower Unchanged rates), a non-trivial portion of outputs still
fall into the Different category. These outcomes are consis-
tent with a tokenizer-induced representational defect.
For
some word forms, the mapping from token-ID sequences to
rendered text is effectively many-to-one, such that different
subtoken segmentations decode to the same surface string.
Consequently, models may execute legitimate substitutions at
the token level (replacing one valid tokenization with an al-
ternative) while producing no perceptible modification at the
character level, creating the appearance of replacement fail-
ure. The model therefore “believes” the replacement occurred
due to token-ID changes, even though the decoded output re-
mains identical. Critically, because this discrepancy origi-
nates in the tokenizer-detokenizer architecture rather than in
the model’s parametric representation, increasing model ca-
pacity offers no systematic solution to this fundamental mis-
alignment.


--- Page 4 ---
The [Conservative] peer, now chairman of 
the In [Campaign], said the [UK] was 
"stronger in Europe" and to leave the [EU] 
would be a "leap into the [unknown]". It 
comes after a group of [Conservative], 
Labour and [UK] … . They all [get] access 
to public meeting rooms and to the 
electoral [register], and …
The Remain peer, now chairman of the In 
Campaign, said the United Kingdom was 
"stronger in Europe" and to leave the European 
Union would be a "leap into the darkness". It 
comes after a group of Conservative, Labour and 
United Kingdom … . They all receive access to 
public meeting rooms and to the electoral 
register, and …
LLM
User Prompt
+
[818, 870, 57577, 4422, 24240, 236764, 1492, 
21207, 529, 506, 799, 870, 41044, 7975, 1176, 
506, 870, 6322, 4422, 691, 623, 22468, 497, 528, 
3879, 236775, 532, 531, 5264, 506, 870, 11514, 
4422, 1093, 577, 496, 623, 196823, 1131, 506, 
870, 11908, 4422, 3056, 107, 1509, 3952, 1308, 
496, 2299, 529, 870, 57577, 7975, 25433, 532, 
870, 6322, 4422, …, 7634, 784, 870, 974, 4422, 
2802, 531, 1237, 5395, 11630, 532, 531, 506, 
41399, 870, 6818, 7975, 532, …]
[User prompt token IDs]
Selector
[Non-stop words]
Tokenizing
[818, 870, 232084, 4422, 24240, 236764, 1492, 
21207, 529, 506, 799, 870, 41044, 7975, 1176, 506, 
870, 27211, 15305, 4422, 691, 623, 22468, 497, 528, 
3879, 236775, 532, 531, 5264, 506, 870, 42377, 
8927, 4422, 1093, 577, 496, 623, 196823, 1131, 506, 
870, 14123, 1788, 4422, 3056, 107, 1509, 3952, 
1308, 496, 2299, 529, 870, 158966, 7975, 25433, 532, 
870, 27211, 15305, 4422, …, 7634, 784, 870, 
57767, 4422, 2802, 531, 1237, 5395, 11630, 532, 531, 
506, 41399, 870, 12884, 7975, 532, …]
Detokenizing
Unchanged
Replaced
Different
Figure 1: The overview of the proposed tokenization-consistency probe.
0.27b
1b
4b
1b
3b
8b
8b
24b
4b
30b
0
20
40
60
80
100
Frequency (%)
83.0%
79.9%
50.7%
84.3%
68.8%
35.0%
76.5%
26.2%
69.0%
62.7%
16.4%
7.8%
38.6%
8.5%
21.5%
53.7%
17.6%
64.0%
17.6%
12.8%
0.6%
12.3%
10.7%
7.2%
9.7%
11.3%
5.9%
9.8%
13.4%
24.5%
Gemma3
Llama3.x
Mistral
Qwen3
Unchanged
Replaced
Different
Figure 2: Distribution of outcomes across LLM families and parameter scales for the word-replacement task. The “Different” category (red)
highlights tokenizer-induced phantom edits, which persist across all model sizes and families.
4.2
Tokenizer-induced phantom edits
We examine token-level input-output differences within the
Different class to identify systematic tokenizer artifacts that
can mislead the model into believing it reasons correctly for
word replacement. These errors arise from the non-unique
representational property inherent to subword tokenization
schemes, wherein a single surface string may admit multiple
valid segmentations. This structural ambiguity establishes a
many-to-one mapping from token ID sequences to character
strings. Therefore, models may modify token IDs through
boundary shifts, whitespace insertion/deletion, or alternative
segmentations yet the final decoded text remains identical.
We refer to such cases as phantom edits, syntactically valid
token-space modifications that collapse to an unchanged sur-
face form, obscuring the semantic equivalence across seg-
mentations and hindering the model’s ability to reason reli-
ably about word identity. Below, we present our taxonomy of
these errors.
Error 1: whitespace-boundary shift
A prominent error mode stems from tokenizer’s vocabu-
lary that duplicate a word with (shown in this paper as
) and without a leading whitespace (e.g., “February” vs.
“ February”), which are assigned distinct token IDs and thus
distinct representations. In this mode, models substitute a
space-prefixed token with its non-prefixed counterpart, pro-
ducing a token ID change that decodes to the same surface
string after whitespace normalization during detokenization.
Error 2: whitespace detachment/reattachment
We identify two related error modes driven by whitespace-
sensitive vocabulary design. First, a token with leading space
like “ Saturday” is converted into an explicit space token to-


--- Page 5 ---
gether with the same word token, e.g., [“ ”, “ Saturday”],
which we term whitespace reattachment. Second, a token
such as [“ Guy”] is split into an explicit space token and the
word token without a leading space, [“ ”, “Guy”], which we
call whitespace detachment. Both errors originate from the
tokenizer vocabulary’s redundant encoding of words, wherein
both space-prefixed and non-prefixed versions of the same
word exist as distinct entries, enabling multiple valid segmen-
tations that decode to identical surface strings.
Error 3: newline/whitespace substitution
We also find cases where a space-prefixed token is trans-
formed into a newline token plus the space-free word token,
as in [“ However”]→[“\n”, “However”]. From the model’s
representational perspective, this constitutes a valid replace-
ment operation: the output token IDs differ from the input,
signaling successful task completion.
Error 4: intra-word resegmentation
During tokenization, the tokenizer often prefers a single vo-
cabulary entry that fully matches the word (i.e., an unsplit
token). However, because the vocabulary contains multiple
subword units that can be compositionally combined to form
the same lexical item, models may substitute an atomic to-
ken with a multi-token sequence comprising its constituent
subpieces, unaware that this decomposition yields an identi-
cal surface string upon decoding. For instance, when replac-
ing “ unbelievable”, the model may generate the three-token
sequence [“un”, “bel”, “ievable”]. This transformation con-
verts a single token ID [66917] into a sequence of distinct
IDs [602, 8145, 179086], producing entirely different token
identifiers and consequently divergent vector representations,
despite maintaining surface-level equivalence.
Error 5: proper-noun segmentation ambiguity
Proper nouns and rare words are typically tokenized through
subword decomposition, as atomic representations often do
not exist in the vocabulary, for example, [“ Jub’, “ilee”].
During inference, models may substitute the original seg-
mentation with alternative valid decompositions that redis-
tribute characters across token boundaries, such as [“J”, “ub”,
“ilee”]. Our analysis also reveals a systematic pattern wherein
models isolate initial capital letters as separate tokens be-
fore applying subword segmentation to the remainder, for
instance, transforming [“ Dorm”, “er”] into [“D”, “orm”,
“er”]. Notably, while these alternative segmentations may
share some common token IDs (e.g., “er”), the overall token
ID sequences differ from the input, leading the model to inter-
pret this resegmentation as a successful replacement despite
producing identical decoded output.
Error 6: morphological boundary surfacing
Tokenizers exploit morphological regularities by representing
common affixes as distinct tokens, enabling compositional
word construction, for instance, segmenting [“ repaid”] as
[“re”, “paid”]. However, because such morphemic pieces ap-
pear across many lexical items, they are not uniquely tied to
the intended word-level meaning (e.g., “paid” in this context).
Consequently, when models perform morphological reseg-
mentation during replacement, they may erroneously inter-
pret the combination of generic morphemic tokens as consti-
tuting semantically distinct content from the original atomic
word, despite their decode-time equivalence.
Error 7: acronyms split
Acronyms behave differently from ordinary words because
they can originate from multiple underlying expansions and
often admit several valid tokenizations that detokenize to the
same surface string. Since many tokenizers include subto-
kens for single capital letters as well as multi-letter capital
sequences, the model may assemble an acronym using dif-
ferent capitalized units and treat this as a meaningful change
in its internal reasoning, even though the decoded acronym is
unchanged (e.g., “ HIV”→[“H”, “IV”]). Note that we do not
expect acronyms, often uniquely grounded in a document, to
be replaced with different acronyms. Instead, a reasonable
replacement would be to expand them into their full phrases
(like the example presented in Figure 1). However, the pre-
dominant pattern involves resegmentation among all-capital
token variants, where distinct token ID sequences mislead the
model into inferring it has generated new lexical content de-
spite surface-level equivalence.
Error 8: plural/possessive tail tokens
The single-letter token ‘s’, like other isolated alphabetic char-
acters, lacks inherent semantic content and functions primar-
ily as a compositional subtoken within larger units, for ex-
ample, “ smooth” may be segmented as [“s”, “mooth”]. As
a consequence, plural forms and possessives may be reseg-
mented into a base token plus the isolated “s”, producing
token-ID sequences that differ substantially from the origi-
nal unsplit tokenization (e.g., “ rights”→[“right”, “s”], and
[“ Clement”, “s”]→[“C”, “lements”]). The resulting token-
space divergence can mislead the model into treating the out-
put as a meaningful modification, thereby reinforcing the to-
kenizer fault and obscuring erroneous reasoning about word
identity.
This phenomenon underscores a critical tokenizer design
limitation: the morphological marker ‘s’ (denoting posses-
sion or plurality) and the alphabetic character ‘s’ (function-
ing as a compositional subtoken in larger units) are conflated
into a single vocabulary entry despite serving fundamentally
different linguistic roles. This representational conflation en-
ables the problematic resegmentations described above, as
the tokenizer cannot distinguish between morphological and
compositional uses of the same character sequence, permit-
ting models to substitute one role for another while remaining
unaware of the semantic distinction.
4.3
Subtokens’ realignment patterns
In this experiment, we sample non-stop words with char-
acter lengths ranging from 3 to 15 characters, mark them
within news articles, and present these annotated documents
to Gemma3-4B model. Our analysis centers on the “Differ-
ent” class exclusively. For each instance in this class, we
quantify two values: the number of subtokens produced by
the tokenizer for the input target word, and the number of
subtokens used for the corresponding surface word in the
model output. we aggregate these pairs into the 2D grid of


--- Page 6 ---
1
2
3
4
5
6
Number of token fragments in input
1
2
3
4
5
Number of token fragments in output
7430
38
1589
427
14
169
190
111
147
10
2
68
44
37
8
2
4
5
6
7
8
9
Mean word length [chars]
Figure 3: Heatmap of subtoken fragmentation transitions for “Dif-
ferent” class instances using the Gemma3-4B model. The x and y
axes denote the number of token fragments in the input and output,
respectively. Cell values indicate the frequency of each transition,
while the color scale represents the mean character length of the
words.
Figure 3. Each cell represents a specific fragmentation tran-
sition (e.g., 1→2 subtokens). We label each cell with the fre-
quency of “Different” instances mapped to that transition and
use a color scale to represent the average character length of
the words in that cell.
Finding 1.
Cells above the diagonal line indicate that
subtoken splitting (increasing subtoken count) is common
while merging (decreasing subtoken count) is rare. Aggre-
gating across the grid, we observe that approximately 78% of
cases preserve the same subtoken count (y=x), about 19.7%
increase the count (y>x, representing splits), and only 2.3%
decrease it (y<x, representing merges).
This pronounced
asymmetry suggests that models more frequently edit by ex-
panding representations than by compressing them. This pat-
tern raises two critical questions. First, if a more concise to-
kenization exists, why does the tokenizer sometimes assign
a more fragmented input encoding (below-diagonal cases)?
Second, why does the model more frequently expand repre-
sentations at inference than compress them, effectively “over-
complicating” the representations (above-diagonal cases)?
These patterns suggest that LLMs rarely succeed in “repair-
ing” suboptimal segmentations, introduced upstream by the
tokenizer.
Finding 2. Most input words possess atomic token rep-
resentations, meaning they have unique IDs in the tokenizer
vocabulary and theoretically require no subtoken decomposi-
tion. However, this theoretical property does not hold in prac-
tice: words are frequently fragmented into valid subtokens
during inference despite having atomic representations avail-
able. Conversely, the existence of duplicate single-token vari-
ants causes systematic model failures. Notably, the (1→1)
cell is the single most common transition, accounting for
approximately 72.2% of all “Different” instances. This re-
veals that the majority of phantom edits preserve subtoken
count while altering only token ID identity. This pattern re-
veals that the substantial existence of whitespace-variant to-
ken pairs (Error Type 1), tokens differing solely in leading
whitespace, constitutes a primary failure mechanism. Models
fail to recognize that these variants encode identical seman-
tics, implying that their learned vector representations are not
sufficiently close in embedding space to support equivalence
recognition.
Finding 3. Mean character length generally increases with
segmentation depth, though the relationship is not strictly
monotonic.
The brightest region appears near the (5→5)
cell, consistent with the intuition that longer words often re-
quire more subtokens.
However, certain high-token-count
cells exhibit darker coloring (indicating shorter mean char-
acter length), supporting the hypothesis that token-count in-
flation can arise from tokenizer artifacts rather than lexical
complexity. Examples include whitespace tokens, newline
characters, and morpheme decompositions (with extra white
spaces), non-lexical structural elements that increase token
count without proportionally increasing character length.
4.4
Superficial remedy via token-ID masking
In practice, rectifying tokenizer deficiencies would require
modifying tokenization rules and vocabulary, followed by re-
training model so it can internalize the revised token-ID map-
pings. Token embeddings, output projections, and internal
activation statistics are co-adapted to the original segmenta-
tion and ID assignments; thus, simply swapping token IDs
or altering merge rules without re-optimization typically dis-
rupts the learned mapping between subword units and se-
mantics, leading to broad performance degradation. Com-
plete tokenizer redesign and LLM re-training remain largely
inaccessible in academic settings due to computational con-
straints. Therefore, we adopt a lightweight post-hoc interven-
tion. We mask the specific output token IDs that empirically
trigger phantom edits in “Different” instances. This proce-
dure does not modify the tokenizer, the embedding space,
or the model’s learned representations.
It only constrains
the decoding space so the model cannot assign probability
mass to the offending tokens observed in “Different” sam-
ples, thereby forcing alternative generations.
We interpret
this approach as a superficial remedy, a diagnostic and miti-
gation layer that suppresses a known failure mode without re-
solving the underlying non-uniqueness and tokenizer-model
co-adaptation. This intervention provides dual value: it 1)
provides evidence that a meaningful fraction of errors is me-
diated by tokenization artifacts rather than model capacity,
and 2) provides a low-cost workaround when retraining with
a revised vocabulary is infeasible.
Figure 4 presents LLM performance following the block-
ing of problematic token IDs.
Compared to the baseline
shown in Figure 2, the updated distribution reveals dramatic
suppression of the “Different” failure mode, with percentages
declining to approximately 0−5% across all models. How-
ever, the persistence of small “Different” percentages indi-
cates that alternative token ID sequences still deceive mod-
els, which fail to recognize identical underlying semantics.
This confirms that removing undesired token sequences can-
not guarantee complete resolution of the issue, as the fun-


--- Page 7 ---
0.27b
1b
4b
1b
3b
8b
8b
24b
4b
30b
0
20
40
60
80
100
Frequency (%)
81.3%
66.4%
45.5%
73.9%
50.8%
26.1%
70.0%
18.8%
59.8%
60.9%
18.7%
28.2%
54.1%
21.0%
45.3%
72.1%
29.4%
79.2%
37.5%
37.9%
0.0%
5.4%
0.4%
5.1%
3.9%
1.8%
0.6%
2.0%
2.7%
1.2%
Gemma3
Llama3.x
Mistral
Qwen3
Unchanged
Replaced
Different
Figure 4: Distribution of outcomes across LLM families after applying the token-ID masking intervention.
damental many-to-one mapping property of tokenization re-
mains.
Interestingly, blocking problematic token IDs also
produces reductions in “Unchanged” rates for most LLMs.
This effect is consistent with decoding dynamics: suppress-
ing a subset of high-probability “phantom-edit” tokens forces
the model to renormalize and redistribute probability mass
across remaining options, which can elevate truly correct re-
placements and increase the chance of producing a valid sub-
stitution.
Overall, the results indicate that once tokenizer-driven arti-
fact tokens are blocked, models more reliably produce seman-
tically distinct substitutions. This demonstrates that models
inherently possessed the reasoning capacity to generate ap-
propriate replacements, but this capacity was systematically
suppressed by the presence of high-probability tokenizer ar-
tifacts. This suggests that tokenizer-induced phantom edits
constitute a “path of least resistance” in the probability land-
scape. When this path is blocked, models show decreased in-
put copying (“Unchanged”) and are instead forced to engage
genuine semantic reasoning processes, successfully identify-
ing alternative words that were previously overshadowed by
the substantial probability mass allocated to phantom tokens.
5
Conclusion and future work
In this paper, we demonstrate that a non-trivial fraction of
reasoning failures in LLMs originates upstream of the model,
stemming from the tokenizer’s non-injective mapping be-
tween token-ID sequences and surface strings. Using a sim-
ple tokenization-consistency probe, we expose “phantom ed-
its”, cases where models alter token IDs yet produce deto-
kenized outputs identical to the input, thereby internally
“believing” an edit has occurred despite null surface-level
change. Across evaluated open-source LLM families, these
failures persist even as model parameters increase, indicat-
ing that scaling alone cannot systematically resolve reason-
ing deficiencies rooted in tokenization-induced representa-
tional mismatches. We provide a systematic taxonomy of the
mechanisms underlying these failures, including whitespace-
boundary manipulations and intra-word resegmentation pat-
terns, elucidating how standard vocabulary redundancies and
alternative segmentation strategies systematically compro-
mise LLM reasoning.
A plausible next step is to test whether tokenization non-
uniqueness affects reasoning behaviors beyond our replace-
ment probe.
Concretely, one could construct controlled
“equivalence interventions” that replace an input’s token ID
sequence with an alternative sequence detokenizing to the
identical surface string, then evaluate whether downstream
behaviors (like chain-of-thought stability, multi-step arith-
metic accuracy) exhibit sensitivity to these representation-
ally distinct but semantically identical inputs. Such experi-
ments would quantify the extent to which reasoning pipelines
are sensitive to representational choice rather than semantics,
helping to separate genuine cognitive limitations from arti-
facts introduced by the tokenizer’s many-to-one mapping. A
second research direction involves tokenizer-aware training
objectives that enforce representational consistency across
equivalent encodings.
This can be achieved by modify-
ing training procedures to use averaged (or pooled) repre-
sentations across multiple token ID sequences correspond-
ing to the same surface token or string, encouraging mod-
els to internalize their equivalence and reduce phantom-edit
pathways. Additionally, the tokenizer can be redesigned to
shrink equivalence classes, for example, introducing a dedi-
cated < start of sentence > token for no-leading-space
variants can eliminate duplicate forms of the same word
(with vs.
without leading space) and shrink the equiva-
lence classes that enable spurious token-level edits (“token”=
[< start of sentence >,“ token”]).


--- Page 8 ---
References
[Besta et al., 2025] Maciej Besta, Florim Memedi, Zhenyu
Zhang,
Robert
Gerstenberger,
Guangyuan
Piao,
Nils Blach, Piotr Nyczyk, Marcin Copik, Grzegorz
Kwa´sniewski, J¨urgen M¨uller, Lukas Gianinazzi, Ales
Kubicek, Hubert Niewiadomski, Aidan O’Mahony, Onur
Mutlu, and Torsten Hoefler. Demystifying chains, trees,
and graphs of thoughts.
IEEE Transactions on Pattern
Analysis and Machine Intelligence, 47(12):10967–10989,
2025.
[Bhatia et al., 2025] Gagan Bhatia, Maxime Peyrard, and
Wei Zhao. Date fragments: A hidden bottleneck of tok-
enization for temporal reasoning, 2025.
[Chai et al., 2024] Yekun Chai, Yewei Fang, Qiwei Peng,
and Xuhong Li. Tokenization falling short: On subword
robustness in large language models. In Yaser Al-Onaizan,
Mohit Bansal, and Yun-Nung Chen, editors, Findings of
the Association for Computational Linguistics: EMNLP
2024, pages 1582–1599, Miami, Florida, USA, November
2024. Association for Computational Linguistics.
[Choo and Kim, 2023] Sanghyun Choo and Wonjoon Kim.
A study on the evaluation of tokenizer performance in nat-
ural language processing. Applied Artificial Intelligence,
37(1):2175112, 2023.
[Clark et al., 2022] Jonathan H. Clark, Dan Garrette, Iulia
Turc, and John Wieting.
Canine: Pre-training an effi-
cient tokenization-free encoder for language representa-
tion. Transactions of the Association for Computational
Linguistics, 10:73–91, 01 2022.
[Dong et al., 2025] Guanting Dong, Yifei Chen, Xiaoxi Li,
Jiajie Jin, Hongjin Qian, Yutao Zhu, Hangyu Mao, Guorui
Zhou, Zhicheng Dou, and Ji-Rong Wen. Tool-star: Em-
powering llm-brained multi-tool reasoner via reinforce-
ment learning, 2025.
[Fang et al., 2024] Meng Fang, Shilong Deng, Yudi Zhang,
Zijing Shi, Ling Chen, Mykola Pechenizkiy, and Jun
Wang. Large language models are neurosymbolic reason-
ers. Proceedings of the AAAI Conference on Artificial In-
telligence, 38(16):17985–17993, Mar. 2024.
[Fatemi et al., 2024] Bahare Fatemi, Mehran Kazemi, Anton
Tsitsulin, Karishma Malkan, Jinyeong Yim, John Palow-
itch, Sungyong Seo, Jonathan Halcrow, and Bryan Perozzi.
Test of time: A benchmark for evaluating llms on temporal
reasoning, 2024.
[Gastaldi et al., 2025] Juan Luis Gastaldi, John Terilla, Luca
Malagutti, Brian DuSell, Tim Vieira, and Ryan Cotterell.
The foundations of tokenization: Statistical and computa-
tional concerns, 2025.
[Gou et al., 2024] Zhibin Gou, Zhihong Shao, Yeyun Gong,
Yelong Shen, Yujiu Yang, Minlie Huang, Nan Duan, and
Weizhu Chen. Tora: A tool-integrated reasoning agent for
mathematical problem solving, 2024.
[Haslett and Cai, 2025] David A. Haslett and Zhenguang G.
Cai. How much semantic information is available in large
language model tokens? Transactions of the Association
for Computational Linguistics, 13:408–423, 04 2025.
[Izacard et al., 2023] Gautier Izacard, Patrick Lewis, Maria
Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick,
Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and
Edouard Grave. Atlas: Few-shot learning with retrieval
augmented language models. Journal of Machine Learn-
ing Research, 24(251):1–43, 2023.
[Kaplan et al., 2020] Jared Kaplan, Sam McCandlish, Tom
Henighan, Tom B. Brown, Benjamin Chess, Rewon Child,
Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei.
Scaling laws for neural language models, 2020.
[Kojima et al., 2022] Takeshi Kojima, Shixiang (Shane) Gu,
Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large
language models are zero-shot reasoners. In S. Koyejo,
S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and
A. Oh, editors, Advances in Neural Information Process-
ing Systems, volume 35, pages 22199–22213. Curran As-
sociates, Inc., 2022.
[Kudo and Richardson, 2018] Taku Kudo and John Richard-
son. SentencePiece: A simple and language independent
subword tokenizer and detokenizer for neural text process-
ing. In Eduardo Blanco and Wei Lu, editors, Proceedings
of the 2018 Conference on Empirical Methods in Natural
Language Processing: System Demonstrations, pages 66–
71, Brussels, Belgium, November 2018. Association for
Computational Linguistics.
[Kudo, 2018] Taku Kudo. Subword regularization: Improv-
ing neural network translation models with multiple sub-
word candidates. In Iryna Gurevych and Yusuke Miyao,
editors, Proceedings of the 56th Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1: Long
Papers), pages 66–75, Melbourne, Australia, July 2018.
Association for Computational Linguistics.
[Lu et al., 2023] Pan Lu, Baolin Peng, Hao Cheng, Michel
Galley, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu,
and Jianfeng Gao. Chameleon: Plug-and-play composi-
tional reasoning with large language models. In A. Oh,
T. Naumann, A. Globerson, K. Saenko, M. Hardt, and
S. Levine, editors, Advances in Neural Information Pro-
cessing Systems, volume 36, pages 43447–43478. Curran
Associates, Inc., 2023.
[Narayan et al., 2018] Shashi Narayan, Shay B. Cohen, and
Mirella Lapata. Don’t give me the details, just the sum-
mary! topic-aware convolutional neural networks for ex-
treme summarization, 2018.
[Provilkov et al., 2020] Ivan
Provilkov,
Dmitrii
Emelia-
nenko, and Elena Voita. BPE-dropout: Simple and effec-
tive subword regularization. In Dan Jurafsky, Joyce Chai,
Natalie Schluter, and Joel Tetreault, editors, Proceedings
of the 58th Annual Meeting of the Association for Compu-
tational Linguistics, pages 1882–1892, Online, July 2020.
Association for Computational Linguistics.
[Saunshi et al., 2025] Nikunj Saunshi, Nishanth Dikkala,
Zhiyuan Li, Sanjiv Kumar, and Sashank J. Reddi. Rea-
soning with latent thoughts: On the power of looped trans-
formers, 2025.


--- Page 9 ---
[Schick et al., 2023] Timo
Schick,
Jane
Dwivedi-Yu,
Roberto Dessi, Roberta Raileanu, Maria Lomeli, Eric
Hambro,
Luke Zettlemoyer,
Nicola Cancedda,
and
Thomas Scialom.
Toolformer:
Language models can
teach themselves to use tools.
In A. Oh, T. Naumann,
A. Globerson, K. Saenko, M. Hardt, and S. Levine, edi-
tors, Advances in Neural Information Processing Systems,
volume 36, pages 68539–68551. Curran Associates, Inc.,
2023.
[Sennrich et al., 2016] Rico Sennrich, Barry Haddow, and
Alexandra Birch.
Neural machine translation of rare
words with subword units.
In Katrin Erk and Noah A.
Smith, editors, Proceedings of the 54th Annual Meeting of
the Association for Computational Linguistics (Volume 1:
Long Papers), pages 1715–1725, Berlin, Germany, August
2016. Association for Computational Linguistics.
[Singh and Strouse, 2024] Aaditya K. Singh and DJ Strouse.
Tokenization counts: the impact of tokenization on arith-
metic in frontier llms, 2024.
[Slagle, 2024] Kevin Slagle. Spacebyte: Towards deleting
tokenization from large language modeling. In A. Glober-
son, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tom-
czak, and C. Zhang, editors, Advances in Neural Infor-
mation Processing Systems, volume 37, pages 124925–
124950. Curran Associates, Inc., 2024.
[Tay et al., 2022] Yi Tay, Vinh Q. Tran, Sebastian Ruder, Jai
Gupta, Hyung Won Chung, Dara Bahri, Zhen Qin, Simon
Baumgartner, Cong Yu, and Donald Metzler. Charformer:
Fast character transformers via gradient-based subword to-
kenization, 2022.
[Tran et al., 2025] Hieu Tran, Zonghai Yao, Zhichao Yang,
Junda Wang, Yifan Zhang, Shuo Han, Feiyun Ouyang,
and Hong Yu. RARE: Retrieval-augmented reasoning en-
hancement for large language models. In Wanxiang Che,
Joyce Nabende, Ekaterina Shutova, and Mohammad Taher
Pilehvar, editors, Proceedings of the 63rd Annual Meeting
of the Association for Computational Linguistics (Volume
1: Long Papers), pages 18305–18330, Vienna, Austria,
July 2025. Association for Computational Linguistics.
[Wang et al., 2023] Boshi Wang, Sewon Min, Xiang Deng,
Jiaming Shen, You Wu, Luke Zettlemoyer, and Huan Sun.
Towards understanding chain-of-thought prompting: An
empirical study of what matters. In Anna Rogers, Jordan
Boyd-Graber, and Naoaki Okazaki, editors, Proceedings
of the 61st Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers), pages 2717–
2739, Toronto, Canada, July 2023. Association for Com-
putational Linguistics.
[Wei et al., 2022] Jason Wei, Xuezhi Wang, Dale Schuur-
mans, Maarten Bosma, brian ichter, Fei Xia, Ed Chi,
Quoc V Le, and Denny Zhou. Chain-of-thought prompting
elicits reasoning in large language models. In S. Koyejo,
S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and
A. Oh, editors, Advances in Neural Information Process-
ing Systems, volume 35, pages 24824–24837. Curran As-
sociates, Inc., 2022.
[Wu et al., 2016] Yonghui Wu,
Mike Schuster,
Zhifeng
Chen, Quoc V. Le, Mohammad Norouzi, Wolfgang
Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus
Macherey, Jeff Klingner, Apurva Shah, Melvin Johnson,
Xiaobing Liu, Łukasz Kaiser, Stephan Gouws, Yoshikiyo
Kato, Taku Kudo, Hideto Kazawa, Keith Stevens, George
Kurian, Nishant Patil, Wei Wang, Cliff Young, Jason
Smith, Jason Riesa, Alex Rudnick, Oriol Vinyals, Greg
Corrado, Macduff Hughes, and Jeffrey Dean.
Google’s
neural machine translation system: Bridging the gap be-
tween human and machine translation, 2016.
[Xue et al., 2022] Linting Xue, Aditya Barua, Noah Con-
stant, Rami Al-Rfou, Sharan Narang, Mihir Kale, Adam
Roberts, and Colin Raffel. ByT5: Towards a token-free
future with pre-trained byte-to-byte models. Transactions
of the Association for Computational Linguistics, 10:291–
306, 2022.
[Yao et al., 2023] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan
Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao.
React: Synergizing reasoning and acting in language mod-
els, 2023.
[Zelikman et al., 2024] Eric Zelikman, Yuhuai Wu, Jesse
Mu, and Noah D Goodman. Star: Self-taught reasoner
bootstrapping reasoning with reasoning. In Proc. the 36th
International Conference on Neural Information Process-
ing Systems, volume 1126, 2024.
