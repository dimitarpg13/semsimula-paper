# Continuous Learning in Semantic Simulation-Based Models vs. Transformer Models

**Technical Report — Seed Document for Section 23 (SemSimula Mega-Paper)**  
**Author:** Dimitar P. Gueorguiev  
**Date:** June 2026  
**Status:** Working Draft — speculative; several results below are stated as conjectures pending formal proof and empirical validation (Experiment G5).

---

## Abstract

This report analyzes continuous learning (CL) — the capacity of a deployed language model to
incorporate new knowledge from runtime interactions without full retraining — across two
fundamentally different architectural paradigms: standard Transformer-based language models and the
Semantic Simulation / Scalar-Potential Language Model (SPLM) family introduced in the
SemSimula framework. We first survey the most principled transformer CL approaches, organized into
three families: test-time adaptation (Test-Time Training, TTT-Layers), knowledge editing
(ROME/MEMIT), and the two strongest structural competitors for the SPLM idea —
gradient-projection methods and **parameter-isolation / modular methods** (adapters, LoRA,
Progressive Networks, PackNet). We then develop the SPLM continuous learning paradigm, grounded in
the dynamic spawning, merging, and pruning of structured potential wells in the bounded Gaussian
decomposition of $V_\theta$. Because the deployed model uses *softmax* mixing weights over wells, we
are careful to distinguish two notions of non-interference: spawning leaves existing well
*parameters* untouched (exact), but it does perturb existing well *responsibilities* through softmax
renormalization (bounded, quantified). A systematic Well Management Policy (WMP) is proposed, with
three tiers — non-RL heuristics, single-level RL (PPO and dueling per-well Q-networks), and
hierarchical RL with an Options framework — as candidate implementations. The analysis is then
lifted to the expressivity class hierarchy of Section 9 (v0–v3), conjecturing that a v0 base model
equipped with a compliant WMP attains runtime expressivity at the v1.5 ∪ v2 level. Finally, we
apply the three boundedness assumptions B1–B3 of Section 9.7 to derive candidate sufficient
conditions under which the CL-augmented system lands at Mildly Context-Sensitive (MCS) complexity,
the natural complexity class of human language. The central speculative result is Conjecture 4
(MCS-Preserving Continuous Learning), which — if its supporting closure argument can be made
rigorous — would elevate the maximum well count $K_{\max}$ from an engineering hyperparameter to a
formal architectural constant determined by the operator algebra of well management. We emphasize
throughout that this is an early-stage theoretical program: the algebraic results in Section 8 are
stated as conjectures, and all propositions require formal proof development. Section 4.5 separately
extends the spawning analysis from the one-body attractor potential $V_\theta$ to the **pairwise
relational potential** $V_\phi$, and shows — provably, and with a machine-precision numerical check —
that the continuous-learning guarantees survive if and only if $V_\phi$ is kept structured; an
unstructured MLP $V_\phi$ confines continuous learning to the attractor channel alone.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Background: The Training–Inference Dichotomy](#2-background-the-traininginference-dichotomy)
3. [Continuous Learning in Transformers](#3-continuous-learning-in-transformers)
   - 3.1 Test-Time Training (TTT)
   - 3.2 TTT-Layers
   - 3.3 Knowledge Editing: ROME and MEMIT
   - 3.4 Gradient Projection Methods
   - 3.5 Parameter-Isolation / Modular Methods (the strongest structural competitor)
   - 3.6 A Note on Retrieval Augmentation
4. [Dynamic Potential Well Spawning in SPLM](#4-dynamic-potential-well-spawning-in-splm)
   - 4.1 Structured V_theta as a Modularly Extensible Object
   - 4.2 Placing a New Well: the h-Space Center and the xi-State Address
   - 4.3 The Fock Gate Interpretation
   - 4.4 Advantages — and Their Precise Scope
   - 4.5 The Relational Channel: Continuous Learning in V_phi
5. [Systematic Comparison](#5-systematic-comparison)
6. [Well Management Strategies](#6-well-management-strategies)
   - 6.1 Formal Setup
   - 6.2 Non-RL Baselines
   - 6.3 Policy Gradient Methods
   - 6.4 Action-Value Methods
   - 6.5 Hierarchical RL and the Options Framework
   - 6.6 The Well Management Policy: Concrete Architecture
7. [Expressivity Class Analysis](#7-expressivity-class-analysis)
   - 7.1 The Class Hierarchy (v0–v3)
   - 7.2 Mapping Well Management Operations to Class Transitions
   - 7.3 Runtime vs. Architectural Expressivity
   - 7.4 The v3 Boundary and Why Spawning Cannot Cross It
8. [Boundedness Assumptions and MCS Membership](#8-boundedness-assumptions-and-mcs-membership)
   - 8.1 B1: Bounded Particle State Dimension
   - 8.2 B2: Bounded Operator Arity and the Split Decomposition Fix
   - 8.3 B3: Finite-Rank Operator Algebra and the K_max Conjecture
   - 8.4 The Candidate Algebra of Well Management Operators (speculative)
   - 8.5 The MCS Landing Conjecture
9. [Propositions and Conjectures](#9-propositions-and-conjectures)
10. [Open Questions and Future Work](#10-open-questions-and-future-work)
11. [References](#11-references)

---

## 1. Introduction

Standard large language models impose a hard architectural wall between training and inference.
Once training converges, parameter weights are frozen and the model functions as a stateless,
purely functional map from input tokens to output distributions. This wall is not a fundamental
physical necessity but an artifact of how gradient-based optimization interacts with distributed,
entangled parameter representations in Transformer architectures. The consequence is severe for
deployed systems: every new piece of world knowledge or user-specific preference must either be
re-injected via prompt (ephemeral, costly, context-limited) or incorporated via expensive
retraining or fine-tuning cycles.

The SPLM family, developed within the SemSimula framework, has a fundamentally different
architectural character. Its learned knowledge is partially localized in the structured potential
function $V_\theta$ — a sum of bounded Gaussian (or, in the legacy SQ3 variant, quadratic) wells
evaluated on the hidden state and parameterized by the streaming $\xi$-register. This locality is
the key: new knowledge can, in principle, be incorporated by adding a new well to $V_\theta$. As we
make precise in Section 4, "adding a well" leaves all existing well *parameters* untouched, but —
under the softmax mixing used by the deployed model — it does mildly renormalize existing well
*responsibilities*. The interference is therefore not zero, but it is bounded and quantifiable,
which is already a categorical improvement over the global, unbounded entanglement of a Transformer
weight update. The training-inference wall, in this architecture, is closer to a policy choice than
to a structural necessity.

This report develops the theoretical and algorithmic foundations for continuous learning in the
SPLM family, benchmarked against the most innovative transformer CL approaches, including the
parameter-isolation / modular methods that constitute the strongest structural competition.
Section 3 establishes the transformer baseline. Sections 4–5 develop and compare the SPLM paradigm.
Section 6 proposes the Well Management Policy. Sections 7–8 ground the analysis in the expressivity
class hierarchy and the MCS boundedness conditions of Section 9 of the mega-paper. We flag at the
outset that Sections 7–8 are **speculative**: the expressivity claims are conjectures, and the
operator-algebra argument underpinning the $K_{\max}$ result is, at present, a plausibility
argument rather than a theorem.

---

## 2. Background: The Training–Inference Dichotomy

### 2.1 The Standard Picture

In a standard Transformer [Vaswani et al., 2017], training minimizes a loss $L(\theta)$ over a
corpus $D$:

$$
\theta^\ast = \arg\min_\theta \mathbb{E}_{(x,y)\sim D}\big[ L(f_\theta(x), y) \big].
$$

After convergence, $\theta^\ast$ is frozen. Inference computes $f_{\theta^\ast}(x)$ for new inputs
$x$ without any modification to $\theta^\ast$. The KV cache provides in-context adaptation within a
session, but this is purely a computational efficiency mechanism — it does not alter $\theta^\ast$
and is discarded after the session ends.

### 2.2 Exceptions and Porosity

The wall between training and inference is more porous than it first appears in practice:

- **Continued pre-training / fine-tuning**: weights are unfrozen and gradient descent resumes on
  new data. This is sequential, not simultaneous with inference.
- **Online / continual learning**: gradient updates interleaved with inference on a data stream.
  The challenge is catastrophic forgetting [McCloskey & Cohen, 1989; Goodfellow et al., 2013].
- **RLHF with online PPO**: inference (generating rollouts) and weight updates are explicitly
  interleaved during the RL training phase [Ouyang et al., 2022].
- **Test-Time Training (TTT)**: lightweight gradient steps at inference time on individual inputs
  [Sun et al., 2020].
- **Parameter-isolation / modular learning**: new capacity (adapters, LoRA factors, new columns)
  is added for new tasks while old parameters are frozen [Rusu et al., 2016; Mallya & Lazebnik,
  2018; Houlsby et al., 2019; Hu et al., 2022]. This is the closest transformer analogue to
  well spawning and is treated in detail in Section 3.5.
- **In-context learning**: not a weight update, but the attention mechanism effectively encodes
  soft adaptation within the context window without modifying parameters [Brown et al., 2020].

None of these, however, achieves the goal of genuinely persistent, accumulative, *and semantically
addressed* learning from runtime interactions — the target capability analyzed in this report. The
parameter-isolation methods come closest on persistence and non-interference, and we therefore
give them careful, non-dismissive treatment.

---

## 3. Continuous Learning in Transformers

We survey the principled transformer CL approaches that do not rely on external retrieval, then
treat parameter-isolation methods as the strongest structural competitor, and finally state plainly
why retrieval augmentation is excluded from the head-to-head.

### 3.1 Test-Time Training (TTT)

**Mechanism.** At inference time, before the final prediction, the model executes a brief
gradient descent on a self-supervised auxiliary task (e.g., masked token reconstruction or
rotation prediction) using only the input instance:

$$
\theta_{\text{adapted}} = \theta^\ast - \eta \nabla_\theta L_{\text{aux}}(f_\theta(x)),
\qquad \text{prediction} = f_{\theta_{\text{adapted}}}(x).
$$

**Key properties.**
- Adaptation is input-specific and incurs a full backward pass at inference — substantial latency.
- By default, the adaptation is **ephemeral**: $\theta_{\text{adapted}}$ is discarded after each
  inference call. If the model is re-initialized to $\theta^\ast$ for each new input, no persistent
  learning occurs.
- Making TTT persistent requires explicitly checkpointing $\theta_{\text{adapted}}$ between calls,
  at which point catastrophic forgetting returns with full force because gradient updates propagate
  across all parameters.
- The auxiliary task must be carefully designed; mismatch between the auxiliary and primary
  objectives degrades adaptation quality [Sun et al., 2020].

**Computational cost.** $O(|\theta|)$ per inference call for the backward pass — the same order as a
training step. This is prohibitive for high-throughput deployment.

### 3.2 TTT-Layers

**Mechanism.** Replaces the attention mechanism entirely with layers whose hidden state $W_t$ is a
small trainable model (linear or MLP), updated via gradient descent at each token:

$$
W_t = W_{t-1} - \eta \nabla_\theta \ell(W_{t-1}; x_t),
\qquad \text{output}_t = W_t \cdot x_t.
$$

The sequence history is compressed into the evolving weight matrix $W_t$ rather than a KV cache.
This achieves $O(n)$ complexity and eliminates the cache [Sun et al., 2024].

**Key properties.**
- Structurally the most radical departure from standard attention: inference is inherently
  dynamical — the hidden state evolves by gradient descent at every token.
- Within a sequence, $W_t$ accumulates context efficiently. Across sequences, $W_t$ resets unless
  explicitly carried forward, in which case catastrophic forgetting returns.
- The inner-loop gradient steps can diverge; there is no structural stability guarantee analogous
  to a conservation law.
- Shares with SPLM the recognition that inference should be **dynamical rather than static** — but
  pays for this insight with gradient computation at every token.

**Computational cost.** Inner-loop gradient steps per token — cheaper than full TTT but
meaningfully more expensive than attention-only inference.

### 3.3 Knowledge Editing: ROME and MEMIT

**Mechanism.** ROME (Rank-One Model Editing) [Meng et al., 2022] performs closed-form rank-one
surgery on specific MLP weight matrices to insert or overwrite factual associations:

$$
W_{\text{edit}} = W + \Delta, \qquad
\Delta = \frac{(v^\ast - W k) k^\top}{(C k)^\top k},
$$

where $k$ is the key vector for the fact's subject, $v^\ast$ is the target value, and $C$ is a
pre-computed covariance matrix. MEMIT [Meng et al., 2023] extends this to simultaneous editing
of multiple facts across multiple layers.

**Key properties.**
- Updates are **permanent** — the weight matrix is overwritten for all future inputs.
- Requires explicit (subject, relation, object) factual triples — narrow applicability restricted
  to knowledge facts, not general semantic adaptation.
- "Ripple effects" [Cohen et al., 2023]: editing one fact can corrupt nearby associative
  structure in the weight matrix because the weight space is entangled.
- Relatively interpretable: the target layer and subject vector are identifiable.

**Computational cost.** Closed-form update — fast. Pre-computation of $C$ is expensive but
amortized across edits.

### 3.4 Gradient Projection Methods

**Mechanism.** Methods such as GEM [Lopez-Paz & Ranzato, 2017], OWM [Zeng et al., 2019], and
GPM [Saha et al., 2021] constrain new gradient updates to the orthogonal complement of the
gradient subspace occupied by previous tasks:

$$
g_{\text{projected}} = g_{\text{new}} - \sum_i (g_{\text{new}} \cdot e_i)  e_i,
$$

where $\lbrace e_i \rbrace$ spans the gradient directions of previous tasks.

**Key properties.**
- Prevents catastrophic forgetting **by construction** — the null-space constraint ensures
  updates to new tasks cannot interfere with previous tasks in parameter space.
- The null space shrinks monotonically as more tasks are learned; eventually the available
  gradient directions are exhausted and new learning becomes impossible.
- Requires storing the gradient bases of all previous tasks — memory grows with task count.
- No connection to the semantic geometry of the model; the projection is purely in parameter
  space.

### 3.5 Parameter-Isolation / Modular Methods (the strongest structural competitor)

This family is the closest transformer analogue to SPLM well spawning, and an honest comparison
must engage it directly rather than omit it.

**Mechanism.** New capacity is *added* for new knowledge while existing parameters are *frozen*:

- **Progressive Networks** [Rusu et al., 2016] add a new column of layers per task, with lateral
  connections from frozen old columns.
- **PackNet** [Mallya & Lazebnik, 2018] iteratively prunes and freezes a sub-network per task,
  reusing the same backbone via binary masks.
- **Adapters** [Houlsby et al., 2019] insert small bottleneck modules between frozen transformer
  layers; one adapter set per task or domain.
- **LoRA** [Hu et al., 2022] adds a low-rank update $\Delta W = BA$ to frozen weight matrices; a
  separate $(B, A)$ pair can be maintained per task and composed or switched at inference.

$$
W_{\text{effective}} = W_{\text{frozen}} + \sum_{\text{task}} m_{\text{task}} \cdot \Delta W_{\text{task}}.
$$

**Where it genuinely matches SPLM.**
- **Additive, parameter-non-interfering.** Freezing old parameters and adding new ones gives the
  *same* exact-parameter-gradient non-interference that SPLM spawning enjoys (Proposition 1).
  Adapters/LoRA do not modify the frozen backbone; old task modules are untouched by new ones.
- **Persistent and cumulative.** Task modules persist and accumulate, exactly as wells do.
- **Bounded per-module cost.** Each module is small and bounded, like a single well.

**Where SPLM differs — the honest contrast.** Parameter-isolation is local *in an arbitrary
parameter partition*, not *in semantic space*. The distinctions are:

1. **Semantic addressing.** A LoRA/adapter module is indexed by a *task ID* supplied externally;
   the system must be told which module applies. A SPLM well is *semantically addressed*: its
   region of influence is the geodesic neighborhood of its $h$-space center $\mu_k$, and it
   activates automatically when a $\xi$-trajectory enters that neighborhood. No task label is
   required at inference.
2. **Geometric locality vs. partition locality.** Modular methods are local only in the sense that
   disjoint parameter subsets are used; there is no metric notion of "near" or "far" in input
   space, and module selection/composition is a discrete routing problem. SPLM locality is metric:
   influence decays with geodesic distance under the Riemannian metric $G$, so "nearby" knowledge
   composes smoothly and distant knowledge is automatically inert.
3. **Composition.** Composing many LoRA modules requires a routing/gating mechanism (which module,
   or which mixture, for this input?) that is itself learned and can interfere. SPLM wells compose
   through the shared force field $\nabla V_\theta$; the softmax responsibilities provide the gating
   intrinsically (at the cost of the renormalization interference quantified in Section 4.1).
4. **Interpretability.** A SPLM well decodes to a semantic address ($\mu_k$ through the LM head) and
   carries a utilization history; a LoRA factor is an opaque low-rank matrix.

**Net assessment.** Parameter-isolation methods already achieve persistent, additive,
non-interfering CL in parameter space — so SPLM's contribution is **not** "additive CL is
possible" (it is, in transformers too) but rather "additive CL that is *semantically addressed and
geometrically local*, with automatic activation and metric composition, and no external task
routing." That is the defensible claim; the earlier framing that "no transformer CL approach
achieves locality" was too strong and is retracted here.

### 3.6 A Note on Retrieval Augmentation

Retrieval-augmented generation (RAG) is excluded from the head-to-head because it is
*non-parametric* memory: it stores knowledge in an external corpus/index and retrieves it into the
context window, leaving $\theta^\ast$ untouched. SPLM well spawning is *parametric* memory — it
changes the model's own potential landscape. The two are complementary rather than competing, and a
fair comparison would pit well spawning against parametric methods (Sections 3.1–3.5) and treat RAG
as an orthogonal augmentation that either paradigm could adopt. We note this explicitly so the
comparison in Section 5 is not read as a claim of superiority over retrieval.

---

## 4. Dynamic Potential Well Spawning in SPLM

### 4.1 Structured V_theta as a Modularly Extensible Object

In the SPLM family, the potential function is evaluated on the hidden state $h \in \mathbb{R}^d$ and
is parameterized by the streaming context $\xi$. In the bounded-Gaussian (mixture-PDF) form
documented in Section 20 of the mega-paper, it is

$$
V_\theta(\xi, h) = - \sum_{k=1}^{K} w_k(\xi) \exp\Big( - \frac{\lVert h - \mu_k(\xi) \rVert_G^2}{2 \sigma_k^2} \Big),
$$

with $\lVert \cdot \rVert_G$ the norm induced by the Riemannian metric tensor $G$, per-well widths
$\sigma_k$, and **softmax mixing weights**

$$
w_k(\xi) = \mathrm{softmax}_k\big( W_w \xi \big).
$$

The amplitudes are bounded by construction (the Gaussian fix that, during Phase 5 training of
Multi-$\xi$ Fock-PARFLM v2.1, brought validation perplexity down from the ~492 peak of the unstable
SQ3 configuration into the ~250 range; see the training-instabilities companion note for the exact
trajectory).

**Two distinct senses of "non-interference."** Spawning well $K{+}1$ has two effects that must not
be conflated:

1. **Parameter non-interference (exact).** The parameters $(\mu_k, \sigma_k)$ of every existing
   well $k \le K$ are untouched by the spawn. Formally, the partial derivative of any existing
   well's parameters with respect to the spawn of $K{+}1$ is zero. This is the content of
   Proposition 1.

2. **Responsibility interference via softmax (bounded, nonzero).** Because the mixing weights are a
   softmax over *all* wells, introducing a new logit $\ell_{K+1} = (W_w \xi)_{K+1}$ renormalizes
   every existing weight:

$$
w_k^{(K+1)}(\xi) = w_k^{(K)}(\xi) \cdot \frac{Z_K}{Z_{K+1}},
\qquad Z_K = \sum_{j=1}^{K} e^{\ell_j}, \quad Z_{K+1} = Z_K + e^{\ell_{K+1}}.
$$

   Every existing responsibility is scaled down by the common factor
   $\rho = Z_K / Z_{K+1} = 1 / (1 + e^{\ell_{K+1}} / Z_K) = 1 - w_{K+1}(\xi)$. Hence the *relative*
   responsibilities among the old wells are exactly preserved, and the *absolute* perturbation is

$$
\big| w_k^{(K+1)}(\xi) - w_k^{(K)}(\xi) \big| = w_k^{(K)}(\xi)  w_{K+1}(\xi) \le w_{K+1}(\xi).
$$

   The total-variation shift of the responsibility distribution restricted to the old wells is
   exactly $w_{K+1}(\xi)$. So the interference is (i) zero in *relative* terms (old wells keep their
   proportions), and (ii) bounded in *absolute* terms by the new well's own responsibility, which
   is small wherever the new well is not active — i.e., everywhere except the new well's own
   geodesic neighborhood. This is the precise, defensible form of the "locality" claim.

An optional **unnormalized variant** replaces the softmax by independent positive gates,

$$w_k(\xi) = \mathrm{softplus}\big((W_w \xi)_k\big),$$

making spawning interference-free in both senses, at the cost of losing the partition-of-unity
interpretation and requiring an explicit amplitude budget to keep $V_\theta$ bounded. We retain the
softmax form to stay consistent with the deployed model and instead bound the interference as above.

### 4.2 Placing a New Well: the h-Space Center and the xi-State Address

A new well's center lives in **hidden-state space**, $\mu_{\text{new}} \in \mathbb{R}^d$, because
$V_\theta$ is evaluated on $h$. The streaming $\xi$-state does **not** live in the same space:
$\xi \in \mathbb{R}^{K_\xi d}$ is the concatenation of $K_\xi$ causal-EMA channels and is used to
*parameterize* wells, not to position them. The earlier draft's shorthand
`mu_new = xi_t` therefore type-mismatched the two spaces and is corrected here.

The correct construction uses the current hidden state and the shared-basis treatment of the SARF
anchor-placement companion note. Because the model uses tied embeddings,
$\text{logits} = h_L E^\top$, the hidden states and token embeddings share the $\mathbb{R}^d$ basis,
and `ln_after_step=True` keeps hidden states on the shell $\lVert h_L \rVert \approx \sqrt{d}$. A
ground-truth correction or new interaction arriving at step $t$ is addressed by

$$
\mu_{\text{new}} = \mathrm{std}\big( h_L(t) \big), \qquad
\sigma_{\text{new}} = f(\text{certainty}), \qquad
A_{\text{new}} = g(\text{signal magnitude}),
$$

where the row-standardization
$\mathrm{std}(v) = (v - \mathrm{mean}(v)) / (\mathrm{sd}(v) + \varepsilon)$ is the same map applied
to anchors in the Phase-5 SARF builder, guaranteeing $\lVert \mu_{\text{new}} \rVert \approx
\sqrt{d}$ so the new well sits on the same shell as the data manifold. Here $\sigma_{\text{new}}$
encodes the specificity of the new knowledge and $A_{\text{new}}$ its (bounded) strength. The role
of the $\xi$-state is to (a) supply the *context address* used by the mixing head $w_\cdot(\xi)$ so
the well activates in the right discourse contexts, and (b) provide, through its trajectory, the
geodesic neighborhood over which the well's influence is felt. In short: hidden-state trajectories
arriving in this region of the $\sqrt{d}$ shell, in contexts resembling the current $\xi_t$, are attracted here.

Importantly, the $O(1)$ inference memory property of SPLM — achieved via the streaming
cumulative-mean $\xi$-state, which eliminates the KV cache — is unaffected by well spawning, as long
as well evaluations remain bounded (guaranteed by the Gaussian bound) and $K$ stays below
$K_{\max}$ (Section 8.3).

### 4.3 The Fock Gate Interpretation

In Multi-$\xi$ Fock-PARFLM v2.1, the correspondence between well spawning and the Fock creation
operator is suggestive and architecturally natural, though we present the operator algebra itself
as speculative (Section 8.4).

A new well $K{+}1$ addressed at the current hidden state corresponds to the action of a creation
operator $\hat{a}^\dagger_{\xi_t}$ exciting a new semantic mode:

$$
\hat{a}^\dagger_{\xi_t}  | \psi_t \rangle = | \psi_t, \mathrm{mode}_{\xi_t} \rangle.
$$

The annihilation operator $\hat{a}_{\xi_t}$ corresponds to pruning the well — removing the mode from
the active set without erasing the possibility of its re-excitation (re-promotion, in the v1.5
salience language of Section 8.8).

The learnable log-space temperature from the v2.1 fixes can be interpreted as a per-well inverse
temperature $\beta_k$ giving each spawned well its own sharpness of attraction. Higher $\beta_k$
(sharper, deeper well) is appropriate for high-confidence ground-truth corrections; lower $\beta_k$
(softer, wider well) is appropriate for uncertain or ambiguous new evidence. We treat this as an
interpretive mapping; the commutation structure that would make it a genuine Fock algebra is
examined, and flagged as unproven, in Section 8.4.

### 4.4 Advantages — and Their Precise Scope

The SPLM continuous learning paradigm has structural advantages over the transformer approaches of
Section 3, but each must be scoped carefully to remain defensible.

**Parameter non-interference (exact) and bounded responsibility interference.** Existing well
parameters are never modified by a spawn (Proposition 1); the only coupling is the softmax
renormalization of Section 4.1, which preserves relative responsibilities exactly and is bounded in
absolute terms by $w_{K+1}(\xi)$. This is stronger than gradient projection (which asymptotically
exhausts its null space) and avoids the entangled ripple effects of ROME/MEMIT. It is *comparable*
to parameter-isolation methods (Section 3.5) on the parameter-non-interference axis; the
differentiator there is semantic addressing and geometric locality, not additivity per se.

**Forgetting: by construction for spawn, policy-dependent for management.** Spawn-only learning is
non-interfering by construction. However, the *full* WMP includes prune and merge, and a poor prune
or an over-aggressive merge is forgetting. The full system therefore does not eliminate forgetting
"by construction"; it *converts* the unconstrained catastrophic-forgetting problem of gradient CL
into a *bounded, auditable policy-quality problem* — a well that is pruned can be re-promoted from
the dormant registry, and merges are constrained by the Riemannian similarity criterion. This is a
genuine improvement (forgetting becomes a controllable, reversible, inspectable decision) but it is
not the absence of forgetting. Section 5's comparison table reflects this split explicitly.

**Geometric locality.** The influence of a spawned well decays with geodesic distance from
$\mu_{\text{new}}$; semantic regions distant from the interaction are automatically inert. This is
the property that distinguishes SPLM from *all* the transformer methods, including
parameter-isolation: their locality is in a parameter partition, not in input/semantic space.

**Interpretability.** Each well has a semantic address ($\mu_k$ decoded through the LM head), a
specificity ($\sigma_k$), a strength ($A_k$), and a utilization history ($\text{hit}\_k$). The
learned potential landscape is inspectable in a way that transformer weight matrices and low-rank
adapters are not.

**Computational cost.** Spawning appends a bounded set of parameters to $V_\theta$ — $O(d)$ for the
center plus $O(1)$ for width/amplitude — with no backward pass and no gradient computation.
Evaluation of $V_\theta$ at each forward step scales as $O(K d)$ for $K$ active wells.

### 4.5 The Relational Channel: Continuous Learning in V_phi

Everything in Sections 4.1–4.4 concerns $V_\theta$, the **one-body** potential — the *attractor*
knowledge of "where, in hidden-state space, do trajectories want to settle." But the PARF arm of the
model carries a second, distinct kind of knowledge in the **pairwise** potential $V_\phi$: the
*relational* knowledge of "how does token $t$ interact with an earlier token $s$." Well spawning
says nothing about this channel. If relational knowledge is to be learned continually, we must ask
the same three questions of $V_\phi$ that Section 4 answered for $V_\theta$ — is it additively
extensible, is it semantically addressed, and is the interference of an extension bounded? This
subsection shows the answers depend sharply on whether $V_\phi$ is **structured** or an
**unstructured MLP**, and — unlike Sections 7–8 — most of the analysis here is provable or directly
measurable rather than conjectural.

**The two implemented forms.** In the deployed model the structured pair potential is

$$
V_\phi^{\text{struct}}(h_t, h_s) = - C \cdot \Theta_\phi\big(\theta(h_t), \theta(h_s)\big) \cdot \Phi_\phi\big(l(h_t), l(h_s)\big) \cdot \frac{1}{\sqrt{\lVert h_t - h_s \rVert^2 + \varepsilon^2}},
$$

with a **Gaussian type-gate** $\Phi_\phi(l_t, l_s) = \exp(-c \lVert l_t - l_s \rVert^2)$ on a
low-rank type projection $l = W_l h$, a bounded value-aligner $\Theta_\phi \in [-1, 1]$, and a
Plummer-softened $1/r$ kernel (see `StructuralVPhi` / `StructuralCompetitiveVPhi` in
`notebooks/conservative_arch/parf/model_parf.py`, and the companion note
[`On_the_MLP_Layer_modeling_pairwise_potential.md`](./On_the_MLP_Layer_modeling_pairwise_potential.md)).
The unstructured ablation `MLPVPhi` collapses all three channels into one network,
$V_\phi^{\text{mlp}}(h_t, h_s) = \mathrm{MLP}([h_t, h_s, h_t - h_s])$, with roughly $7\times$ the
parameters and no factorisation.

**The structured form is a spawnable object; the MLP is not.** The key observation is that the
type-gate $\Phi_\phi$ is a **localised, addressable** unit in type-space, structurally identical to a
Gaussian well — only it lives in the *pair*-type space rather than in $h$-space. The natural
continuous-learning generalisation makes the single gate a sum of $M$ addressable **relational
components**, each with its own type-center $\nu_m$ in the type projection,

$$
V_\phi(h_t, h_s) = - \sum_{m=1}^{M} C_m \, \Theta_m(h_t, h_s) \, \Phi_m(l_t, l_s) \, \frac{1}{r}, \qquad \Phi_m(l_t, l_s) = \exp\!\big( -c \lVert \tfrac{1}{2}(l_t + l_s) - \nu_m \rVert^2 \big),
$$

which is exactly the addressable-center version of the already-implemented `MultiHeadVPhi` (a sum of
independent structural heads — see `model_parf.py`). A *relation* is then the spawnable unit, and the
propositions of Section 4.1 transfer verbatim.

**Proposition 6 (Parameter non-interference of relational spawning).** Spawning relational component
$M{+}1$ leaves the parameters $(\nu_m, C_m, \Theta_m\text{-weights})$ of every existing component
$m \le M$ unchanged:

$$
\frac{\partial (\nu_m, C_m)}{\partial (\text{spawn of } M{+}1)} = 0 \qquad \text{for all } m = 1, \ldots, M.
$$

This is exact and holds for **both** the additive form (the current `MultiHeadVPhi`, where heads are
summed) and the softmax-competitive form. In the additive form, spawning is interference-free in
*both* the parameter and the responsibility senses, at the cost of needing an explicit amplitude
budget $\sum_m C_m \le C_{\max}$ to keep $V_\phi$ bounded (the same trade-off as the unnormalised
$V_\theta$ variant in Section 4.1).

**Proposition 7 (Bounded responsibility interference and geometric locality).** Under a softmax over
the $M$ component gates (the competitive form), spawning component $M{+}1$ rescales every existing
responsibility by the common factor $\rho = 1 - w_{M+1}$, preserving relative responsibilities
exactly and bounding the absolute perturbation by

$$
\big| w_m^{(M+1)} - w_m^{(M)} \big| = w_m^{(M)} \, w_{M+1} \le w_{M+1} \le \exp\!\big( -c\,(d_m^2 - d_{\text{near}}^2) \big),
$$

where $d_m$ is the query's type-distance to the new center and $d_{\text{near}}$ its distance to the
nearest existing center. So interference is zero in relative terms and **decays exponentially with
type-space distance** — a new relation perturbs only pairs whose type-projection is near its center.
This is identical in form to Proposition 1′ for wells.

These two propositions are validated numerically in
`notebooks/conservative_arch/parf/diagnostics/vphi_spawn_interference_demo.py`, which exercises the
exact gate arithmetic on synthetic type vectors. The bound of Proposition 7 holds to machine
precision (max violation of $|\Delta w_m| - w_{M+1}$ is $1.4 \times 10^{-16}$; max relative-weight
drift $3.3 \times 10^{-16}$), and in a controlled sweep with the partition function held fixed the
interference log-slope recovers the predicted $-c$ to three figures ($-0.999$ versus a target of
$-1.0$).

**Proposition 8 (MLP V_phi is not spawnable — global entanglement).** The unstructured
$V_\phi^{\text{mlp}}$ admits no decomposition into addressable units. Adding or revising relational
knowledge can only be done by a gradient step $\delta W$ on its shared weights, which perturbs the
pair potential at **every** pair,

$$
\delta V_\phi^{\text{mlp}}(h_t, h_s) = \big\langle \nabla_W V_\phi^{\text{mlp}}(h_t, h_s), \, \delta W \big\rangle,
$$

a quantity that is non-negligible for essentially all $(h_t, h_s)$ and carries no decay in
type-distance. The same demo confirms this: a single fixed-norm weight step moves $99.5\%$ of all
pairs, and the magnitude of the perturbation is flat in pair distance (normalised distance-slope
$\approx 0.1$, versus the exponential decay of the structured gate). This is precisely the
catastrophic-forgetting signature of entangled parameter spaces — the property that makes relational
continuous learning hard in transformers (Sections 3.3–3.4). **The cost of an MLP $V_\phi$ for
continuous learning is therefore categorical, not incremental: it removes the relational channel from
the spawnable-knowledge budget entirely**, leaving only $V_\theta$ wells learnable at deployment and
freezing all relational knowledge behind a retraining wall.

**How much is at stake? The relational fraction.** Whether any of this matters for a *given* model
is an empirical question with a directly measurable answer. Define the relational fraction

$$
\varphi_{\text{rel}} = \frac{L(V_\phi = 0) - L(\text{full})}{L(\text{full})},
$$

the fractional cross-entropy increase when the pair potential is ablated to zero (the $V_\theta$ +
Verlet dynamics still run). $\varphi_{\text{rel}} \approx 0$ means the model barely uses $V_\phi$ and
relational CL is low-value; a large $\varphi_{\text{rel}}$ means the relational channel carries real
predictive knowledge and the structured-vs-MLP choice is decisive for continuous learning. This is
read-only and runs against a saved checkpoint in a separate runtime (it does not disturb a live
training run) via
`notebooks/conservative_arch/parf/diagnostics/measure_relational_fraction.py`.

**Experiment G6 (relational continuous-learning ablation).** The propositions above are exact
statements about *interference*; the *downstream* CL behaviour is a prediction to be tested. G6 trains
two models to matched perplexity — one with mixture-structured (spawnable) $V_\phi$, one with MLP
$V_\phi$ — then injects $N$ relational facts sequentially (by component spawning for the structured
model, by gradient fine-tuning for the MLP, which has no other mechanism) and measures retention on a
held-out "old-relations" probe. The pre-registered functional forms, with analytically-derived
behaviour, are the falsifiable content. Table cells use plain Unicode per the rendering conventions.

| Observable | Structured (spawnable) prediction | MLP (gradient) prediction | Discriminating statistic |
|---|---|---|---|
| Old-relation CE after N injections, ΔL(N) | a·(1 − ρ^N), saturating at a; ρ = E[1 − w_new] | b·N + e·N² (linear-to-superlinear), unbounded | finite asymptote vs divergent |
| Interference vs type-distance | ≤ exp(−c·(d² − d²_near)) — exponential decay | ≈ constant — flat | log-slope ≈ −c vs ≈ 0 |
| New-relation acquisition | one-shot at the injection step | only after several gradient steps | steps-to-acquire: 1 vs many |
| Compute per injected fact | O(d) append, no backward pass | O(\|φ\|) backward pass(es) | wall-clock per fact |

The discriminators — bounded versus divergent retention asymptote, and negative versus zero
distance-slope — both have closed-form predictions, so the regression fits are genuinely falsifiable
rather than descriptive.

**Honest scope.** Three things are settled and need no further experiment: the interference bounds
(Propositions 6–7, validated to machine precision), the non-spawnability of the MLP form
(Proposition 8), and the measurability of $\varphi_{\text{rel}}$. Two things remain open: the
*mixture-structured spawnable* $V_\phi$ with addressable per-component type-centers is a proposed
architecture — the current Phase-5 run uses the additive `MultiHeadVPhi` *without* a spawning policy,
so it establishes only that the additive structural form trains stably, not that relational spawning
works at scale; and the G6 retention curves are predictions, not results. The defensible claim for
the paper is therefore narrow and strong: *the continuous-learning argument extends to relational
knowledge if and only if $V_\phi$ is kept structured; an MLP $V_\phi$ confines continuous learning to
the attractor channel alone.*

---

## 5. Systematic Comparison

The following table summarizes the key properties of each CL approach against the dimensions most
relevant to deployed language model systems. The catastrophic-forgetting row is split into the
*spawn-only* claim and the *fully-managed* claim, per the scoping of Section 4.4. RAG is omitted as
non-parametric (Section 3.6). Per the rendering conventions for tables, symbols here use plain
Unicode/ASCII rather than inline math.

| Property | TTT | TTT-Layers | ROME / MEMIT | Grad. Projection | Param-Isolation (Adapter/LoRA) | **SPLM Well Management** |
|---|---|---|---|---|---|---|
| Update cost | Full backward pass | Per-token grad step | Closed-form | Per-task grad basis | Per-module training | O(d) append (spawn), no grad |
| Persistence | Ephemeral by default | Ephemeral by default | Permanent | Permanent | Permanent | Permanent (dormant registry) |
| Forgetting (spawn / add only) | Returns if persistent | Returns if persistent | Ripple effects | Null-space exhaustion | None (frozen old params) | None (params); bounded softmax shift |
| Forgetting (full lifecycle) | n/a | n/a | n/a | n/a | Module collision under routing | Policy-dependent (prune/merge), reversible |
| Locality | None (global) | None | Layer-local | None | Parameter-partition only | Geodesic decay in semantic space |
| Activation | Always | Always | Always | Always | External task ID / routing | Automatic (xi enters basin) |
| Interpretability | Opaque | Opaque | Moderate | None | Low (opaque low-rank) | Full semantic address |
| Signal requirement | Self-sup task | Self-sup task | (s, r, o) triples | Task gradient | Task label + data | User interaction signal |
| Memory scaling | O(\|θ\|) | O(\|W_t\|)/layer | O(rank edit) | O(tasks × rank) | O(modules × rank) | O(K·d), K ≤ K_max |
| Inference dynamical | No | Yes (W_t evolves) | No | No | No | Yes (xi evolves) |
| Geometric awareness | None | None | None | None | None | Riemannian metric |
| O(1) memory at inference | No | ~O(n) | No | No | No | Yes (streaming xi) |

**Deepest structural contrasts.**

- *Versus TTT-Layers:* both make inference dynamical, but by opposite means. TTT-Layers make the
  hidden state a model that updates by gradient descent each token (inference becomes training).
  SPLM governs $\xi$-state evolution via Euler–Lagrange / Verlet dynamics — closed-form, governed
  by conservation laws, no gradient (learning becomes dynamics). SPLM's direction gives better
  stability guarantees (bounded trajectories) and lower inference cost.

- *Versus parameter-isolation:* both are additive and parameter-non-interfering. The SPLM
  differentiators are semantic addressing (no external task ID), metric locality and composition
  through the shared force field, and interpretability of each unit. These are real but narrower
  claims than "additivity," which transformers already have.

---

## 6. Well Management Strategies

Well spawning solves the *creation* side of continuous learning. A complete CL system requires a
policy governing the full lifecycle of wells: creation, update, dormancy, merge, and pruning.
We call this the **Well Management Policy (WMP)**. This section is design-stage; none of the
architectures below has yet been trained at scale (Section 10, Experiment G5).

### 6.1 Formal Setup

**Well state.** Each active well $k$ is characterized by the tuple of its $h$-space center
$\mu_k \in \mathbb{R}^d$ (Section 4.2), width $\sigma_k \in \mathbb{R}\_+$, amplitude
$A_k \in \mathbb{R}\_+$, age $\text{age}\_k \in \mathbb{N}$, hit count
$\text{hit}\_k \in \mathbb{N}$, and salience $\text{sal}\_k \in \mathbb{R}\_+$. Here $\text{age}\_k$
counts inference steps since spawn, $\text{hit}\_k$ counts $\xi$-trajectories that have passed within
geodesic radius $\sigma_k$ of $\mu_k$ (the utilization metric), and $\text{sal}\_k$ is the salience
score (Section 7).

**MDP formulation.**
- **State**: $s_t = (\xi_t, \lbrace w_k \rbrace_{k=1}^{K_t})$.
- **Action space (primitive)**: spawn, prune, merge, update, and no-op.
- **Reward**:

$$
R_t = \alpha \cdot \Delta\text{accuracy}_t - \beta \cdot |\text{wells}|_t - \gamma \cdot \Delta\mathrm{KL}\_t,
$$

  where $\Delta\mathrm{KL}\_t$ measures how much existing trajectory distributions shifted after the
  action (landscape stability penalty).

**Key structural challenge.** The state $s_t$ is a *set* over wells — permutation-invariant and
variable-cardinality. Standard fixed-input architectures cannot handle this directly. A
**DeepSets** [Zaheer et al., 2017] or **Set Transformer** [Lee et al., 2019] encoder over
$\lbrace w_k \rbrace$ is the natural choice:

$$
z_t = \mathrm{SetEncoder}\big( \lbrace w_k \rbrace_{k=1}^{K_t} \big) \in \mathbb{R}^m.
$$

All proximity computations within the encoder must use geodesic distances under the Riemannian
metric $G(\xi)$, not Euclidean distances. The asymmetric geodesic ratio of 1.35–1.40 measured in
the five-arm Riemannian Diagnostic Battery means Euclidean proximity is systematically miscalibrated
and would produce incorrect merge/prune decisions.

### 6.2 Non-RL Baselines

Before RL, three principled heuristic approaches provide strong baselines and correctness
guarantees:

**Sparse GP inducing point management.** Wells as inducing points in a sparse Gaussian Process
over the $h$-manifold [Titsias, 2009]. Merge when two inducing points' joint KL contribution to the
sparse posterior falls below a threshold. Principled but assumes stationarity of the kernel — not
guaranteed in a dynamically evolving landscape.

**Bayesian online learning per well.** Each well carries a posterior over $(\mu_k, \sigma_k, A_k)$.
New interactions update via Bayes' rule. Pruning criterion: the marginal likelihood of the well
given recent trajectory data falls below a model-comparison threshold (Bayes factor below 1). Fully
principled and uncertainty-aware but quadratic in $K$ at update time.

**Energy-based stability pruning.** Prune well $k$ if

$$
\big| V_\theta(\mu_k) - V_{\theta \setminus k}(\mu_k) \big| \lt \varepsilon,
$$

i.e., the well is not a genuine local minimum in the landscape when evaluated without its own
contribution. This connects directly to the Euler–Lagrange fixed-point stability condition. A well
that fails this test is not contributing a distinct attractor basin — it is a perturbation
dominated by neighboring wells. This is the cleanest heuristic and requires no additional learning.

### 6.3 Policy Gradient Methods

**REINFORCE for spawning decisions.** The simplest RL entry point. A policy
$\pi(\text{spawn} \mid \xi_t, z_t)$ gives a probability of spawning at each interaction; new-well
parameters are sampled from a conditional distribution
$\pi(\mu, \sigma, A \mid \xi_t, z_t)$. Typically $\mu$ is set at $\mathrm{std}(h_L(t))$ with a
learned offset, $\sigma$ from a learned head, and $A$ from the ground-truth signal magnitude. The
**delayed reward problem** is severe: a well spawned at step $t$ may prove useful only at step
$t+50$, and Monte Carlo REINFORCE returns have high variance under this delay. REINFORCE is
recommended only for prototyping, not production WMP.

**PPO for the full management policy.** Proximal Policy Optimization [Schulman et al., 2017] is a
natural fit. We note a *motivating analogy* (not an identity): PPO's clipped objective discourages
large policy changes, which aligns in spirit with the WMP's landscape-stability requirement.

$$
L_{\text{CLIP}} = \mathbb{E}\big[ \min\big( r_t(\theta)  \hat{A}_t,  \mathrm{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)  \hat{A}_t \big) \big].
$$

This conservatism tends to prevent drastic management actions that would destabilize existing well
basins. The PPO policy-KL trust region and the $\Delta\mathrm{KL}\_t$ landscape-stability penalty are
*different* KL divergences (one over action distributions, one over trajectory distributions); the
analogy is heuristic and we do not claim they coincide.

**Architecture:**
- **Actor**: a Set Transformer over $\lbrace w_k \rbrace$ produces action logits over
  spawn / prune / merge / update / no-op.
- **Critic**: the same Set Transformer encoder produces $V(s_t)$, the estimated long-term landscape
  quality.

For merge and prune, the action target is an element of the variable-length well set, so a
**pointer network** [Vinyals et al., 2015] over the well set is the natural output head.
Generalized Advantage Estimation (GAE) [Schulman et al., 2016] handles the multi-step credit
assignment from delayed rewards.

**PPO challenges:** variable-cardinality state complicates batching; continuous parameter sampling
$(\mu, \sigma, A)$ requires a hybrid continuous action head.

### 6.4 Action-Value Methods

**Per-well Q-networks.** Rather than a global $Q(s, a)$, decompose into per-well action values
using a cooperative value-decomposition framework [Sunehag et al., 2018]:

$$
Q_{\text{global}}(s, a) \approx V(s) + \sum_k A_k(s, a_k),
$$

where $V(s)$ is the global landscape value (from the Set Transformer) and $A_k$ is the advantage of
action $a_k$ on well $k$. Each per-well advantage head maps $(w_k, z_t, \xi_t)$ to a $Q$-value over
the per-well action set $\lbrace \text{keep, prune, merge-target, update}_+, \text{update}_- \rbrace$.

**Dueling architecture over the well set.** For each well,

$$
Q_k(s, a_k) = V(s) + A_k(s, a_k) - \mathrm{mean}_{a'}  A_k(s, a'_k).
$$

**Interpretability dividend.** The per-well Q-values provide an interpretable *well health
dashboard*:
- $A_k(\text{prune})$ strongly positive: the well is clearly redundant or harmful.
- $A_k(\text{merge with } j)$ strongly positive for specific $j$: wells $k$ and $j$ are
  semantically redundant.
- $A_k(\text{keep})$ consistently dominant: the well is earning its place.

This interpretability is unavailable in PPO's policy representation and has direct value for
monitoring and debugging the CL system in deployment.

**Hybrid discrete/continuous decomposition.** Use dueling per-well DQN for discrete management
decisions and a separate PPO head for continuous parameters:

- **Discrete (DQN)**: which action type and which well.
- **Continuous (PPO)**: exact parameter values $(\mu, \sigma, A)$ for the chosen action.

**Experience replay complication.** The well configuration in old replay entries may have very
different cardinality from the current one. The Set Transformer encoder handles variable cardinality
at training time, but care must be taken to avoid stale global embeddings $z_t$ from outdated
configurations.

### 6.5 Hierarchical RL and the Options Framework

Well management naturally operates on **two distinct timescales**:

- **Fast (reactive)**: spawn / no-spawn at every ground-truth interaction (milliseconds).
- **Slow (deliberative)**: merge / prune / split campaigns after evidence accumulates
  (minutes to hours of interactions).

The **Options framework** [Sutton et al., 1999] is the natural architecture for this two-timescale
structure.

**High-level policy (slow, possibly model-based):** decides *when* to trigger a lifecycle event
over options $\lbrace O_{\text{spawn}}, O_{\text{prune}}, O_{\text{merge}}, O_{\text{update}}
\rbrace$. The physics structure of SPLM gives a strong inductive bias for a learned world model —
the Euler–Lagrange dynamics provide analytical predictions for how $V_\theta$ changes with well
modifications:

$$
p(\Delta\mathrm{KL} \mid \text{wells}_t, \text{action}), \qquad
p(\Delta\text{accuracy} \mid \text{wells}_t, \text{action}).
$$

**Low-level options (fast, reactive):** each option $O_k$ has an initiation set $I_k$, a policy
$\pi_k$, and a termination condition $\beta_k$. For example, $O_{\text{prune}}$ initiates when
$\text{sal}\_k$ falls below threshold $\tau$, executes the prune primitive, and terminates when the
well has entered the dormant registry.

**Why hierarchy is not just a preference here:** the two timescales correspond to different
observation spaces — $(\xi_t, \text{nearest wells})$ at the fast scale, the global configuration
$\lbrace w_k \rbrace$ at the slow scale. A flat policy at both scales would require long-horizon
planning while reacting in real time. The Options framework decomposes this cleanly.

### 6.6 The Well Management Policy: Concrete Architecture Proposal

Assembling all components (design proposal, not yet validated). The schematic below is a structural
diagram, not math:

```
+---------------------------------------------------------+
|                  WELL MANAGEMENT POLICY                 |
|                                                         |
|  Encoder: DeepSets / SetTransformer                     |
|    Input:  { (mu_k, sigma_k, A_k, age_k, hit_k, sal_k) }|
|    Kernel: Riemannian geodesic distance                 |
|    Output: global embedding z_t in R^m                  |
|                                                         |
|  Fast Actor (PPO):                                      |
|    pi_fast(spawn | z_t, xi_t) + parameter head          |
|    mu-head outputs in h-space (sqrt(d) shell)           |
|    Triggered: every ground-truth interaction            |
|                                                         |
|  Per-well Q-networks (Dueling DQN):                     |
|    Q_k(s, a_k) over { keep, prune, merge, update }      |
|    Output: well health dashboard                        |
|                                                         |
|  Slow Actor (Hierarchical, model-based):                |
|    pi_slow(O | z_t) over options { prune, merge }       |
|    World model: p(dKL, d_acc | wells_t, action)         |
|    Triggered: every N interactions or |wells| > K_max   |
|                                                         |
|  Hard constraints (enforced by action masking):         |
|    |wells_active| <= K_max  (B3 compliance)             |
|    No split primitive        (B2 compliance)            |
|    All wells in R^d          (B1 compliance)            |
+---------------------------------------------------------+
```

**Key design invariants:**
- All proximity computations in the Riemannian metric (geodesic distance, not Euclidean).
- Well centers output on the $\sqrt{d}$ shell in $h$-space (Section 4.2).
- Merge criterion: $d_G(\mu_k, \mu_j) \lt \varepsilon$ **and**
  $\cos(\nabla V_k, \nabla V_j) \gt \delta$ (similar in both position and semantic gradient
  direction).
- Pruning criterion: $\text{sal}\_k \lt \tau$ **and** $Q_k(\text{prune}) \gt Q_k(\text{keep})$ for
  $N$ consecutive evaluations.
- Dormancy: pruned wells enter a dormant registry; $\text{sal}\_k$ can recover if future
  $\xi$-trajectories pass near $\mu_k$.
- Split replaced by prune + 2×spawn sequence (B2 compliance — see Section 8.2).

---

## 7. Expressivity Class Analysis

*This section is speculative. The class-transition claims are conjectures intended to guide the
formal development in Section 23, not established theorems.*

### 7.1 The Class Hierarchy (v0–v3)

Section 9 of the mega-paper defines the expressivity class hierarchy of the SPLM family:

**v0 — baseline SPLM.** Fixed particle set, fixed-cardinality well configuration, no salience
dynamics. The active semantic structure set is constant throughout inference.

**v1 — salience promotion (creation without destruction).** Wells can gain salience but cannot lose
it. A transitional class not analyzed in depth here.

**v1.5 — salience decay (destruction).** Following Section 8.8, every active semantic structure $s$
at discourse step $t$ carries a non-negative salience $\sigma_t(s) \ge 0$ that decays under absence
of content-similar evidence and is reinforced by it. Demotion below an implementation-defined
threshold suppresses the structure's participation in the force-field calculations of Sections 5
and 6 *without erasing it* (re-promotion is allowed). The salience-augmented v0+v1.5 dynamics is a
strict contraction in the salience direction.

**v2 — creation.** Following Sections 8.6 and 8.7, new semantic structures are introduced into
discourse by template instantiation and execution events. The cardinality of the active particle
set grows during inference, in contrast to v0's fixed-cast assumption. The active particle set is
treated as a state on a Fock space (Section 9.5).

**v3 — execution.** Following the template-execution duality of Section 8.6, each particle carries
(or is associated via the template channel with) an operator that acts on other particles' states.
Operator composition is in general **non-abelian**, encoding the order-dependence of meaning
composition.

### 7.2 Mapping Well Management Operations to Class Transitions

The correspondence between well lifecycle operations and expressivity class transitions is proposed
as follows (conjectural). Table cells use plain Unicode/ASCII per the rendering conventions.

**Well pruning / dormancy / salience decay → v1.5**

| v1.5 definition | Well management implementation |
|---|---|
| sigma_t(s) decays under absence of evidence | hit_k decreases as trajectories stop visiting well k |
| Demotion below threshold suppresses force-field participation | Pruning removes well k from V_theta evaluation |
| Without erasing (re-promotion allowed) | Pruned wells enter dormant registry; re-enter V_theta if hit_k recovers |
| Strict contraction in salience direction | Bounded A_k guarantees removal of low-hit wells cannot destabilize landscape |

**Well merging** is treated as a special case of v1.5: two wells whose geodesic distance falls
below $\varepsilon$ assert that their semantic structures have converged to the same evidence basin.
One's salience is absorbed into the other's rather than zeroed — cooperative demotion rather than
competitive.

**Well spawning → v2**

| v2 definition | Well management implementation |
|---|---|
| New structures introduced by template instantiation | Well parameters (Gaussian form) are the template; spawning instantiates at std(h_L(t)) |
| Introduction by execution events | Ground-truth interactions are execution events triggering spawns |
| Cardinality of active particle set grows during inference | \|wells\|_t grows as interactions accumulate |
| Active particle set as Fock space state (Section 9.5) | Fock creation operators spawn new semantic modes (interpretive; see Section 8.4) |

**Combined runtime class (conjectured):**

$$
\begin{aligned}
\text{v0} + \text{pruning/merging with dormancy} & \rightarrow  \text{v1.5} \\
\text{v0} + \text{spawning from execution events} & \rightarrow  \text{v2} \\
\text{v0} + \text{full WMP} & \rightarrow  \text{v1.5} \cup \text{v2}
\end{aligned}
$$

### 7.3 Runtime vs. Architectural Expressivity

**Definition (Runtime Expressivity).** The runtime expressivity of a deployed system
$(M, \pi_{\text{WMP}})$ is the expressivity class of the model as observed in deployment, accounting
for the management policy, as distinct from the **architectural expressivity** of the bare forward
pass.

**Pre-empting an obvious objection.** One might object that we have simply "moved the learning into
the policy," so that any expressivity gain belongs to $\pi_{\text{WMP}}$ rather than to $M$. The
distinction we draw is this: $\pi_{\text{WMP}}$ performs *structural edits under a bounded, discrete
operator set* (spawn / prune / merge / update), not gradient-based representation learning. It
changes *which* bounded wells are present, not *what* a well can represent. Consequently the
combined system inherits $M$'s MCS-class constraints (Section 8) precisely because the edits respect
B1–B3. The expressivity attribution is therefore meaningful: the WMP unlocks the model's *native*
v2/v1.5 capability across the inference boundary, rather than importing new representational power
from outside.

**Conjecture (Runtime Expressivity Lift).** For an SPLM model $M$ at architectural class v0 and a
WMP $\pi_{\text{WMP}}$ satisfying the B1–B3 conditions of Section 8,

$$
\mathrm{Expressivity}_{\text{runtime}}(M, \pi_{\text{WMP}}) = \text{v1.5} \cup \text{v2}
\supsetneq \mathrm{Expressivity}_{\text{arch}}(M) = \text{v0}.
$$

A practical corollary, stated as a *hypothesis to be tested in G5* (not as fact): a v0 model with a
sophisticated Riemannian WMP **may** outperform a v2 model with no well management on tasks
requiring cumulative semantic adaptation, because the WMP contributes runtime expressivity that the
bare v2 forward pass does not utilize when its well configuration is static. This is an empirical
conjecture, not a theorem.

### 7.4 The v3 Boundary and Why Spawning Cannot Cross It

v3 requires each particle to **carry an operator that acts on other particles' states**, with
non-abelian composition. Well spawning as defined cannot cross this boundary, for three reasons:

**The WMP acts externally on the well set.** It is an operator on the configuration space of wells,
not an operator carried by individual wells acting on each other's state vectors.

**Wells contribute scalar fields, not operator-valued interactions.** Well $k$ contributes a term
$w_k(\xi)  v_k(h; \mu_k, \sigma_k)$ to $V_\theta$ — a scalar function of position. This influences
all $\xi$-trajectories through the force field $\nabla V_\theta$, but it does not act on the state
$(\mu_j, \sigma_j, A_j)$ of well $j$. There is no mechanism by which well $k$ modifies well $j$'s
parameters.

**Spawning is order-independent at the parameter level; v3 requires non-abelian composition.**
Adding well $A$ then well $B$ contributes the same two terms to the (pre-softmax) potential sum as
adding $B$ then $A$:

$$
(\cdots + v_A + v_B) = (\cdots + v_B + v_A).
$$

(The softmax over weights is itself a symmetric function of the logit set, so the *responsibility*
assignment is also order-independent.) v3 requires that particle $A$'s operator acts on particle
$B$'s state in an order-dependent way. Crossing the v3 boundary would require wells to carry
operators that **modify each other's parameters** — a genuinely different architectural requirement,
not achievable by $V_\theta$ enrichment alone.

*(Note: a separate, weaker non-commutativity appears in iterated Riemannian merges, Section 8.4.
That is non-commutativity of a management campaign, not the operator-valued inter-particle action
that defines v3; we are careful not to conflate the two.)*

---

## 8. Boundedness Assumptions and MCS Membership

*This section is the most speculative in the report. The $K_{\max}$ result (8.3) and the algebra
(8.4) are presented as conjectures supported by plausibility arguments; making them theorems
requires a closure proof that we have not yet completed.*

### 8.1 B1: Bounded Particle State Dimension

**Statement (from Section 9.7).** Each v2 particle of type $A$ carries a state vector
$x_A \in \mathbb{R}^{k_A d}$ with $k_A$ a fixed positive integer (the particle's fan-out) and $d$ a
global state-dimension constant.

**Analysis for well spawning.** Each spawned well is a v2 particle with state
$(\mu_k, \sigma_k, A_k)$ where $\mu_k \in \mathbb{R}^d$, $\sigma_k \in \mathbb{R}\_+$,
$A_k \in \mathbb{R}\_+$. The fan-out is $k_A = 1$ (the well contributes a scalar value to $V_\theta$
regardless of $d$). The dimension $d$ is fixed by the architecture as the hidden-state ($h$-space)
dimension.

**B1 is satisfied by well spawning** provided:
1. The $h$-space dimension $d$ is held fixed — inherited from the architecture, not a WMP choice.
2. The WMP never spawns wells in an augmented space (e.g., concatenating new feature dimensions to
   accommodate a new modality). This justifies the action-space hard constraint that all wells live
   in $\mathbb{R}^d$.

**Stability interaction.** The bounded Gaussian amplitude $A_k$ together with B1 bounds each well's
contribution:

$$
\big| w_k(\xi) \cdot v_k(h; \mu_k, \sigma_k) \big| \le w_k(\xi) \cdot A_k \le A_k \lt \infty,
$$

a per-particle force-field bound independent of $K$.

### 8.2 B2: Bounded Operator Arity and the Split Decomposition Fix

**Statement (from Section 9.7).** Each v3 operator takes at most $r_{\max}$ existing particles as
inputs and produces **one output particle**.

**Analysis for well management operations.** Table cells use plain Unicode/ASCII per the rendering
conventions.

| Operation | Inputs | Outputs | B2 status |
|---|---|---|---|
| Spawn | 0 existing wells | 1 new well | OK Compliant |
| Prune / demote | 1 well | 0 (or dormant) | OK Compliant |
| Merge | 2 wells | 1 merged well | OK Compliant (r_max ≥ 2) |
| Update | 1 well | 1 modified well | OK Compliant |
| **Split** | 1 well | **2 wells** | **Violates B2** |

**The Split Decomposition Fix.** Split is replaced by the following B2-compliant primitive
sequence:

$$
\mathrm{Split}(k)  \rightarrow  \mathrm{Prune}(k) + \mathrm{Spawn}(\mu_A, \sigma_A, A_A) + \mathrm{Spawn}(\mu_B, \sigma_B, A_B),
$$

where $\mathrm{Prune}(k)$ is one-to-zero (it moves well $k$ to the dormant registry) and each
$\mathrm{Spawn}$ is zero-to-one. Here $\mu_A, \mu_B$ are the centroids of the two detected modes in
the bimodal $h$-state distribution within well $k$'s basin, and $\sigma_A, \sigma_B, A_A, A_B$ are
set proportionally to each mode's variance and mass (subject to the amplitude bound of
Section 8.1). This preserves semantic content: the original well is dormant (re-promotable), and the
two new wells capture the two subregions. B2 holds because each primitive produces at most one
output particle.

In the hierarchical framework, split lives at the slow timescale as an option $O_{\text{split}}$
that decomposes into these three B2-compliant primitives.

### 8.3 B3: Finite-Rank Operator Algebra and the K_max Conjecture

**Statement (from Section 9.7).** The group $G$ generated by the v3 operators is
finite-dimensional with finite rank.

**Analysis (plausibility argument).** We *posit* a set of management generators — spawn
$\hat{a}^\dagger_k$, prune $\hat{a}\_k$, merge $\hat{m}\_{k,j}$, update $\hat{u}\_k$ (Section 8.4) —
and ask whether the structure they generate is finite-dimensional. If the active well count $K_t$
can grow without bound, the number of candidate generators grows without bound, which would push
the system outside MCS. Bounding $K$ bounds the generator count.

**Conjecture (K_max bound).** Let $K_{\max}$ be a hard upper bound on the number of simultaneously
active wells. Then the number of *independent* management generators is at most $O(K_{\max}^2)$
(from spawn/prune over $K_{\max}$ wells and merge over ordered pairs), so — *assuming the generators
close into a finite-dimensional algebra* — its rank is finite and B3 is satisfied.

**Caveat (the missing closure step).** This is not yet a theorem. It is *not* established here that
the posited generators close: the bracket $[\hat{m}\_{k,j}, \hat{m}_{j,l}]$ (Section 8.4) produces
new elements, and we have not shown they remain within the span of finitely many generators. The
bound "$\mathrm{rank}(G) \le c \cdot K_{\max}^2$" should therefore be read as a *generator-count*
upper bound, not a proven Lie-group rank. A rigorous version requires either (i) proving closure of
the management algebra, or (ii) replacing the Lie-group language with the weaker, sufficient claim
below.

**Sufficient weaker claim (defensible now).** Independently of any algebraic closure, a hard cap
$K_t \le K_{\max}$ means the active configuration is drawn from a finite-dimensional, bounded family
of potentials, so the per-step computation and the set of reachable landscapes are bounded uniformly
in the session length. This bounded-configuration property is what the MCS argument actually needs;
the Lie-group framing is an attempt to give it algebraic content and is presently aspirational.

**Critical design implication.** Whichever framing is used, $K_{\max}$ must be enforced as a **hard
action-space constraint** via action masking, not as a soft reward penalty. The reward's $\beta$
term discourages proliferation but cannot guarantee $|\text{wells}|_t \le K_{\max}$ — a sufficiently
high-value spawn can outweigh $\beta$. Only masking (making spawn unavailable when
$|\text{wells}|_t = K_{\max}$) gives a hard guarantee.

**K_max as a structural constant (conjectural).** If the closure step can be completed, $K_{\max}$
would determine the rank bound of $G$ and hence the system's expressivity ceiling, elevating it from
a tuning knob to a formal architectural constant. We state this as the *goal* of the program, not
as an achieved result.

### 8.4 The Candidate Algebra of Well Management Operators (speculative)

We define a *candidate* set of formal management operators and examine their commutation structure.
We do **not** claim these satisfy canonical bosonic relations; the relations below are posited for
the management interpretation and require justification.

Define, acting on the Fock-space description of the active well set: $\hat{a}^\dagger_k$ (spawn well
$k$), $\hat{a}\_k$ (prune well $k$), the occupancy/salience number operator
$\hat{N}\_k = \hat{a}^\dagger_k \hat{a}\_k$, the merge operator $\hat{m}\_{k,j}$ (merge well $k$ into
well $j$), and the update operator $\hat{u}\_k$.

**Posited relations (subject to formal verification).** Adopting the standard ladder convention as
a working hypothesis,

$$
\begin{aligned}
[\hat{a}^\dagger_k, \hat{a}^\dagger_j] &= 0 \quad\text{(spawning two distinct wells commutes)} \\
[\hat{a}\_k, \hat{a}\_j] &= 0 \quad\text{(pruning two distinct wells commutes)} \\
[\hat{a}\_k, \hat{a}^\dagger_j] &= \delta_{kj}  I \quad\text{(canonical relation; nontrivial only for the same well)} \\
\hat{N}\_k &= \hat{a}^\dagger_k \hat{a}\_k \quad\text{(number operator; not itself a commutator)}.
\end{aligned}
$$

The corrected reading of the earlier draft is that spawn and prune of the *same* well fail to
commute because pruning a not-yet-spawned well is undefined; the canonical bracket
$[\hat{a}\_k, \hat{a}^\dagger_k] = I$ encodes exactly this occupancy bookkeeping, and $\hat{N}\_k$
tracks the resulting count. (The earlier draft's "$[\hat{a}^\dagger_k, \hat{a}\_k] = \hat{N}\_k$" was
an error and is withdrawn.)

**Non-commutativity of merge campaigns (the interesting, but still informal, part).** Merge uses the
Riemannian Fréchet mean to compute the merged center,
$\mu_{\text{merged}} = \mathrm{FrechetMean}\_G(\lbrace \mu_k, \mu_j \rbrace)$, so iterated merges are
order-dependent in curved space:

$$
\begin{aligned}
\text{merge}(\text{merge}(k, j), l):\quad \mu &= \mathrm{FrechetMean}\_G\big(\lbrace \mathrm{FrechetMean}\_G(\lbrace \mu_k, \mu_j \rbrace),  \mu_l \rbrace\big), \\
\text{merge}(k, \text{merge}(j, l)):\quad \mu &= \mathrm{FrechetMean}\_G\big(\lbrace \mu_k,  \mathrm{FrechetMean}\_G(\lbrace \mu_j, \mu_l \rbrace) \rbrace\big).
\end{aligned}
$$

These coincide when $G$ is Euclidean (flat) but differ under the curved SPLM metric, so the merge
sub-structure is non-abelian. The practical implication — that hierarchical merge *campaigns* are
order-dependent and the high-level policy must plan merge *sequences*, not just sets of pairs — is a
genuine consequence of the geometry. We flag, however, that we have not shown these merge operators
close into a finite-dimensional algebra (the gap noted in Section 8.3), so the "Lie algebra"
language here is aspirational.

### 8.5 The MCS Landing Conjecture

**Conjecture (MCS-Preserving Continuous Learning).** Let $M$ be an SPLM model at expressivity class
v0+v1.5+v2 under the composite dynamics of Section 9. Let $\pi_{\text{WMP}}$ be a well management
policy satisfying:

- **(C1)** Each spawned well has state in $\mathbb{R}^d$ with $d$ fixed and fan-out $k_A = 1$
  (B1 compliance).
- **(C2)** No split primitive; splits decompose into Prune + Spawn + Spawn (B2 compliance).
- **(C3)** The active well count $K_t \le K_{\max}$ at all times, enforced as a hard action-space
  constraint (B3 compliance, in the bounded-configuration sense of Section 8.3).

Then $(M, \pi_{\text{WMP}})$ satisfies B1 ∧ B2 ∧ B3 and is conjectured to land at **Mildly
Context-Sensitive (MCS) complexity** by the result of Section 9.7, with MCS membership maintained
across all inference steps of the continuous learning session. The strength of this conjecture is
currently limited by the open closure question of Section 8.3; under the weaker
bounded-configuration reading of (C3), the per-step boundedness needed for the MCS argument does
hold.

**Corollaries (contingent on the conjecture).**

1. **Polynomial parseability.** Inference time remains in P as wells accumulate up to $K_{\max}$,
   because MCS languages are polynomially parseable [Joshi, 1985].
2. **Semi-linearity.** The set of trajectories generated by $(M, \pi_{\text{WMP}})$ satisfies the
   constant growth property targeted by SPLM.
3. **Stability certificates compose.** The v1.5 strict contraction in the salience direction and
   the B1 per-well amplitude bound are independent stability certificates that reinforce each
   other: low-salience wells are actively contracted out before they can accumulate.
4. **Runtime class is maximal under B1–B3.** The deployed system operates at v1.5 ∪ v2 runtime
   expressivity — conjectured to be the highest class achievable while maintaining MCS membership,
   since v3 operator-valued inter-particle interactions would require algebraic structure whose
   rank is not bounded by $K_{\max}$ alone.

---

## 9. Propositions and Conjectures

The following summarize the main claims, separated by confidence level. Propositions are
straightforward and provable; conjectures are speculative and flagged as requiring proof and/or
empirical validation in Section 23.

**Proposition 1 (Parameter Non-Interference of Spawning).** Let
$V_\theta(\xi, h) = - \sum_{k=1}^{K} w_k(\xi)  v_k(h; \mu_k, \sigma_k)$ with softmax weights
$w_k(\xi)$. Spawning well $K{+}1$ leaves the *parameters* of every existing well unchanged:

$$
\frac{\partial (\mu_k, \sigma_k)}{\partial (\text{spawn of } K{+}1)} = 0 \qquad \text{for all } k = 1, \ldots, K.
$$

That is, no existing well's center or width is modified by the spawn. (This is the precise,
defensible statement; it concerns parameters, not responsibilities — see Proposition 1′.)

**Proposition 1′ (Bounded Responsibility Interference).** Under the softmax mixing of Section 4.1,
spawning well $K{+}1$ rescales every existing responsibility by the common factor
$\rho = 1 - w_{K+1}(\xi)$, preserving relative responsibilities exactly and bounding the absolute
perturbation by

$$
\big| w_k^{(K+1)}(\xi) - w_k^{(K)}(\xi) \big| = w_k^{(K)}(\xi)  w_{K+1}(\xi) \le w_{K+1}(\xi),
$$

with total-variation shift over the old wells equal to $w_{K+1}(\xi)$. Interference is zero in
relative terms and bounded by the new well's own (locally small) responsibility in absolute terms.

**Proposition 2 (Merging is Approximately Expressivity-Preserving under a Riemannian Criterion).**
Let wells $k$ and $j$ be merged under $d_G(\mu_k, \mu_j) \lt \varepsilon$. Define the merged well by

$$
\begin{aligned}
\mu_{k \cup j} &= \mathrm{FrechetMean}\_G(\lbrace \mu_k, \mu_j \rbrace), \\
A_{k \cup j} &= \min( A_k + A_j,  A_{\max} ), \\
\sigma_{k \cup j} &= \max(\sigma_k, \sigma_j) + d_G(\mu_k, \mu_j)/2.
\end{aligned}
$$

Then for any $\xi$-trajectory $\gamma$ that stays geodesically far from the merged basin
($d_G(\gamma(t), \mu_k) \gg \sigma$ and $d_G(\gamma(t), \mu_j) \gg \sigma$ for all $t$), the merged
well produces force-field contributions that differ from the pair $(k, j)$ by an amount that is
**exponentially small in the geodesic distance**, hence negligible — not exactly zero. *Note:* the
amplitude is explicitly capped at $A_{\max}$ to keep $V_\theta$ bounded under repeated merges
(without the cap, iterated $A_k + A_j$ would grow without bound, violating the Gaussian boundedness
the stability argument relies on). The "approximate" qualifier and the cap correct the earlier
draft's over-strong "identical" claim.

**Proposition 3 (Split Decomposition Preserves B2).** Split$(k)$ decomposed as
$\mathrm{Prune}(k) + \mathrm{Spawn}(\mu_A, \sigma_A, A_A) + \mathrm{Spawn}(\mu_B, \sigma_B, A_B)$
satisfies B2 (each primitive produces at most one output particle) and approximately preserves
attractor mass: $A_A + A_B \approx A_k$ (subject to the amplitude bound) with
$\mu_A, \mu_B \in \mathrm{basin}(k)$.

**Proposition 6 (Parameter Non-Interference of Relational Spawning).** For the mixture-structured
pair potential $V_\phi = -\sum_m C_m \Theta_m \Phi_m / r$, spawning relational component $M{+}1$
leaves the parameters of every existing component unchanged. Full statement and the additive-versus-
competitive distinction in Section 4.5.

**Proposition 7 (Bounded Responsibility Interference for Structured V_phi).** Under softmax-competitive
mixing of relational components, $|\Delta w_m| = w_m w_{M+1} \le w_{M+1} \le \exp(-c(d_m^2 -
d_{\text{near}}^2))$, so relational interference decays exponentially in type-space distance.
Validated to machine precision in `vphi_spawn_interference_demo.py` (Section 4.5).

**Proposition 8 (MLP V_phi Non-Locality).** The unstructured $V_\phi^{\text{mlp}}$ admits no
addressable decomposition; a single gradient step perturbs essentially all pairs with no distance
decay, so relational knowledge is globally entangled and not spawnable (Section 4.5).

**Conjecture 4 (MCS-Preserving Continuous Learning — speculative).** Under conditions C1–C3 of
Section 8.5, the deployed system $(M, \pi_{\text{WMP}})$ is conjectured to achieve runtime
expressivity class v1.5 ∪ v2 while maintaining MCS membership across all inference steps. The system
does not achieve v3, as operator-valued inter-particle action and non-abelian operator composition
are not properties of $V_\theta$ enrichment. *Status:* contingent on the closure question of
Section 8.3; the weaker bounded-configuration reading is defensible now, the full Lie-algebraic form
is not.

**Conjecture 5 (K_max as a Structural Constant — speculative).** If the management generators of
Section 8.4 close into a finite-dimensional algebra, then the maximum simultaneous well count
$K_{\max}$ bounds its rank,

$$
\mathrm{rank}(G) \le c \cdot K_{\max}^2 \qquad \text{(generator-count bound; rank interpretation pending a closure proof)},
$$

so that finiteness of $K_{\max}$ would make $K_{\max}$ a necessary architectural constant rather
than a tunable regularization hyperparameter. *Status:* the generator count is bounded by
$O(K_{\max}^2)$; the identification of this bound with a Lie-group rank, and hence the "structural
constant" claim, awaits the closure proof.

---

## 10. Open Questions and Future Work

**G5: Empirical validation of continuous learning via well spawning.** The framework here requires
empirical validation at scale. Proposed as Experiment G5 in the Riemannian geometry battery of
Section 23 (following G1–G4: geodesic analogical reasoning, energy-dissipation hallucination
detection, asymmetric geodesic distance, and entailment directionality). G5 would measure accuracy
on held-out interactions as a function of the number of spawned wells, comparing against
frozen-weight SPLM, fine-tuned SPLM, parameter-isolation (LoRA/adapter) transformer baselines, and
TTT-Layers.

**G6: Relational continuous learning — structured vs. MLP V_phi.** Section 4.5 shows that the
spawnability of the *relational* channel hinges on $V_\phi$ being structured (Propositions 6–8). G6
operationalises this: train matched-perplexity models with mixture-structured (spawnable) $V_\phi$ and
with MLP $V_\phi$, inject $N$ relational facts sequentially, and fit the pre-registered retention and
interference regressors (Section 4.5 table). The discriminators — bounded versus divergent retention
asymptote, and exponential versus flat interference-vs-distance — have closed-form predictions. A
prerequisite measurement, the relational fraction $\varphi_{\text{rel}} = (L(V_\phi{=}0) -
L(\text{full})) / L(\text{full})$, is available now as a read-only diagnostic
(`measure_relational_fraction.py`) and quantifies how much knowledge the relational channel even
carries before G6 is run.

**Resolve the softmax-vs-unnormalized weighting choice empirically.** Section 4.1 bounds the
softmax renormalization interference; an unnormalized-gate variant removes it entirely but
sacrifices the partition-of-unity interpretation and needs an explicit amplitude budget. Which
choice yields better cumulative-CL behavior is an open empirical question.

**Close (or replace) the management-algebra argument.** The central theoretical gap (Sections
8.3–8.4) is whether the management generators close into a finite-dimensional algebra. Either a
closure proof or a clean reformulation in terms of the bounded-configuration property would convert
Conjectures 4–5 into theorems.

**Well proliferation dynamics under the Riemannian metric.** The asymmetric geodesic ratio of
1.35–1.40 implies the effective basin volume is anisotropic. A formal characterization of basin
volume under the asymmetric metric would sharpen the merge criterion and bound the proliferation
rate as a function of discourse diversity.

**Forgetting under prune/merge.** Since the full WMP can forget through bad prune/merge decisions
(Section 4.4), bounding the *reversible* forgetting introduced by the dormant-registry mechanism —
and characterizing when re-promotion recovers prior performance — is an important open problem.

**WMP training curriculum.** The WMP needs a curriculum exposing it to discourse scenarios with
known ground-truth correction signals, covering the distribution of merge/prune challenges. This is
non-trivial and likely needs domain-specific augmentation.

**Interaction between CL and the G1–G4 experiments.** G1 (geodesic analogical reasoning) and G3
(asymmetric geodesic distance) test static properties of trained wells. Do *dynamically spawned*
wells exhibit the same geodesic structure, or do they require a "settling period" to reach geodesic
equilibrium?

**Expressivity of TTT-Layers under the Section 9 hierarchy.** TTT-Layers' hidden state $W_t$ evolves
by gradient descent — a parametric update during inference. Where does it land in v0–v3? Preliminary
assessment: $W_t$ evolution is not particle creation/destruction (not v2) and does not carry
operators over other particles' states (not v3), suggesting v0 with a rich $V_\theta$ analogue — but
this needs formal analysis.

**The v3 CL question.** What would a v3-capable CL system look like? It would require spawned
particles that carry operators over each other's state — new concepts that actively reorganize the
meaning of existing ones. This is arguably what sophisticated human learning looks like (a new
framework reinterpreting old results), but it is a substantial architectural extension beyond
$V_\theta$ enrichment.

---

## 11. References

**Architecture and foundations:**

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., &
  Polosukhin, I. (2017). *Attention is all you need.* Advances in Neural Information Processing
  Systems, 30.

- Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., ... & Amodei, D.
  (2020). *Language models are few-shot learners.* Advances in Neural Information Processing
  Systems, 33, 1877–1901.

**Test-Time Training:**

- Sun, Y., Wang, X., Liu, Z., Miller, J., Efros, A. A., & Hardt, M. (2020). *Test-time training
  with self-supervision for generalization under distribution shifts.* Proceedings of the 37th
  International Conference on Machine Learning (ICML), PMLR 119.

- Sun, Y., Li, X., Dalal, K., Xu, J., Vikram, A., Zhang, G., ... & Ré, C. (2024). *Learning to
  (learn at test time): RNNs with expressive hidden states.* arXiv preprint arXiv:2407.04620.

**Knowledge editing:**

- Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). *Locating and editing factual
  associations in GPT.* Advances in Neural Information Processing Systems, 35, 17359–17372.

- Meng, K., Sharma, A. S., Andonian, A., Belinkov, Y., & Bau, D. (2023). *Mass-editing memory
  in a transformer.* International Conference on Learning Representations (ICLR).

- Cohen, R., Biran, E., Yoran, O., Globerson, A., & Geva, M. (2023). *Evaluating the ripple
  effects of knowledge editing in language models.* Transactions of the Association for
  Computational Linguistics, 11, 1273–1288.

**Continual / lifelong learning (gradient and parameter-isolation):**

- McCloskey, M., & Cohen, N. J. (1989). *Catastrophic interference in connectionist networks:
  The sequential learning problem.* Psychology of Learning and Motivation, 24, 109–165.

- Goodfellow, I. J., Mirza, M., Xiao, D., Courville, A., & Bengio, Y. (2013). *An empirical
  investigation of catastrophic forgetting in gradient-based neural networks.* arXiv preprint
  arXiv:1312.6211.

- Lopez-Paz, D., & Ranzato, M. (2017). *Gradient episodic memory for continual task learning.*
  Advances in Neural Information Processing Systems, 30.

- Zeng, G., Chen, Y., Cui, B., & Yu, S. (2019). *Continual learning of context-dependent
  processing in neural networks.* Nature Machine Intelligence, 1(8), 364–372.

- Saha, G., Garg, I., & Roy, K. (2021). *Gradient projection memory for continual learning.*
  International Conference on Learning Representations (ICLR).

- Rusu, A. A., Rabinowitz, N. C., Desjardins, G., Soyer, H., Kirkpatrick, J., Kavukcuoglu, K.,
  Pascanu, R., & Hadsell, R. (2016). *Progressive neural networks.* arXiv preprint
  arXiv:1606.04671.

- Mallya, A., & Lazebnik, S. (2018). *PackNet: Adding multiple tasks to a single network by
  iterative pruning.* IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

- Houlsby, N., Giurgiu, A., Jastrzębski, S., Morrone, B., de Laroussilhe, Q., Gesmundo, A.,
  Attariyan, M., & Gelly, S. (2019). *Parameter-efficient transfer learning for NLP.* Proceedings
  of the 36th International Conference on Machine Learning, PMLR 97.

- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022).
  *LoRA: Low-rank adaptation of large language models.* International Conference on Learning
  Representations (ICLR).

**Reinforcement learning:**

- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal policy
  optimization algorithms.* arXiv preprint arXiv:1707.06347.

- Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). *High-dimensional
  continuous control using generalized advantage estimation.* International Conference on
  Learning Representations (ICLR).

- Sutton, R. S., Precup, D., & Singh, S. (1999). *Between MDPs and semi-MDPs: A framework for
  temporal abstraction in reinforcement learning.* Artificial Intelligence, 112(1–2), 181–211.

- Sunehag, P., Lever, G., Gruslys, A., Czarnecki, W. M., Zambaldi, V., Jaderberg, M., ... &
  Graepel, T. (2018). *Value-decomposition networks for cooperative multi-agent learning.*
  arXiv preprint arXiv:1706.05296.

- Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., ... & Lowe, R.
  (2022). *Training language models to follow instructions with human feedback.* Advances in
  Neural Information Processing Systems, 35.

**Neural architecture for sets:**

- Zaheer, M., Kottur, S., Ravanbakhsh, S., Poczos, B., Salakhutdinov, R., & Smola, A. (2017).
  *Deep sets.* Advances in Neural Information Processing Systems, 30.

- Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., & Teh, Y. W. (2019). *Set transformer:
  A framework for attention-based permutation-invariant neural networks.* Proceedings of the
  36th International Conference on Machine Learning, PMLR 97.

- Vinyals, O., Fortunato, M., & Jaitly, N. (2015). *Pointer networks.* Advances in Neural
  Information Processing Systems, 28.

**Sparse Gaussian Processes:**

- Titsias, M. K. (2009). *Variational learning of inducing variables in sparse Gaussian
  processes.* Proceedings of the 12th International Conference on Artificial Intelligence and
  Statistics (AISTATS), PMLR 5.

**Mildly Context-Sensitive languages and formal language theory:**

- Joshi, A. K. (1985). *Tree adjoining grammars: How much context-sensitivity is required to
  provide reasonable structural descriptions?* Natural Language Parsing, 206–250.
  Cambridge University Press.

- Kanazawa, M. (1998). *Learnable classes of categorial grammars.* Center for the Study of
  Language and Information (CSLI) Publications.

- Weir, D. J. (1988). *Characterizing mildly context-sensitive grammar formalisms.* PhD
  Dissertation, University of Pennsylvania.

**Riemannian geometry:**

- Do Carmo, M. P. (1992). *Riemannian geometry.* Birkhäuser, Boston.

- Pennec, X., Fillard, P., & Ayache, N. (2006). *A Riemannian framework for tensor computing.*
  International Journal of Computer Vision, 66(1), 41–66.

**SemSimula / SPLM (internal references):**

- Gueorguiev, D. P. (2026). *Semantic Simulation: A Prescriptive Lagrangian Framework for
  Efficient Semantic Inference.* [Main mega-paper; Sections 4 (Gaussian well), 6 (SARF),
  8.6–8.8, 9.1–9.9 (expressivity hierarchy and B1–B3), 20 (structured scalar potential and the
  bounded Gaussian wells).]

- Gueorguiev, D. P. (2026). *Exploiting the Riemannian Geometry of Conservative Language Models.*
  [Companion paper documenting damped Riemannian geometry in the SPLM family, five-arm Riemannian
  Diagnostic Battery, gamma_eff ≈ 0.13 vs. gamma_param = 0.93, asymmetric geodesic ratio
  1.35–1.40.]

- Companion notes: *Training Instabilities in Fock-PARFLM v2.1 with Structured V_theta* (bounded
  Gaussian fix, sigma/precision caps, watchdog) and *Placing SARF Anchors from Converged Gaussian
  Well Centres* (shared-basis row-standardization to the sqrt(d) shell, anchor-placement rules).

- HuggingFace collection: dimitarpg13/semantic-simulation-splm-model-family (CC-BY-4.0).
  Multi-xi Fock-PARFLM v2.1 (Phase 5), training on OpenWebText, A100.

---

End of Technical Report. This document is a speculative seed for Section 23 of the SemSimula
mega-paper and a companion repository markdown. Sections 7–8 are conjectural; all propositions
require formal proof development, and Conjectures 4–5 additionally require the closure argument of
Section 8.3. Internal cross-references (Sections 4, 6, 8.6–8.8, 9.1–9.9, 20) refer to the mega-paper.
