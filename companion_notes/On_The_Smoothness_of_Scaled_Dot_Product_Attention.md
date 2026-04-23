# On the Smoothness of Scaled Dot-Product Attention
### Implications for Poincaré's Lemma and the Conservativity Obstruction Theorem

> **Canonical copy (this file).** This path is the one indexed from the
> [`semsimula-paper` README](https://github.com/dimitarpg13/semsimula-paper#companion_notes--2026-companion-notes-work-in-progress)
> for readers of the paper. A second copy may exist in the
> [semsimula](https://github.com/dimitarpg13/semsimula) monorepo under `docs/`;
> if the two differ, **this** repository's file is authoritative for the paper.

**Context:** This note analyzes whether the attention force field $F(h)$ and the full
transformer block satisfy the smoothness assumptions required by Poincaré's lemma,
which underlies Theorem 46 ($K\neq V$ obstructs conservativity) in the Semantic Simulation
paper. It also catalogs the smoothness properties of each transformer component
separately, distinguishing what matters for the theorem from what does not.

---

## Table of Contents

1. [The Smoothness Question and Why It Matters](#1-the-smoothness-question-and-why-it-matters)
2. [Scaled Dot-Product Attention](#2-scaled-dot-product-attention)
3. [LayerNorm — Smooth Almost Everywhere](#3-layernorm)
4. [FFN Nonlinearities — The Real Smoothness Issue](#4-ffn-nonlinearities)
5. [Causal Masking — Not a Discontinuity in h](#5-causal-masking)
6. [Dropout — Training Only, Irrelevant at Inference](#6-dropout)
7. [What This Means for Theorem 46](#7-what-this-means-for-theorem-46)
8. [Summary Table](#8-summary-table)
9. [Technical Appendix: Smoothness Classes Defined](#9-technical-appendix)

---

## 1. The Smoothness Question and Why It Matters

### 1.1 The Role of Smoothness in Theorem 46

Theorem 46 ($K\neq V$ obstructs conservativity) establishes that the per-head attention
force $F(h)$ cannot be written as $-\nabla V$ for any scalar potential $V\colon \mathbb{R}^d \to \mathbb{R}$, unless
the key and value weight matrices are proportional.

The proof proceeds via **Poincaré's lemma**:

$$
\text{Poincaré's Lemma (smoothness form):}
$$

$$
\text{Let } U \subseteq \mathbb{R}^d \text{ be open and simply connected.}\quad
\text{Let } F\colon U \to \mathbb{R}^d \text{ be a } C^1 \text{ vector field.}
$$

$$
\begin{aligned}
\text{Then: } F = -\nabla V &\text{ for some smooth } V\colon U \to \mathbb{R} \\
&\iff \\
&\text{ the Jacobian } \frac{\partial F_i}{\partial h_j} \text{ is symmetric everywhere on } U \\
&\iff \\
&\Omega(h) := \frac{J_F(h) - J_F(h)^{\mathsf T}}{2} = 0 \quad \text{ for all } h \in U
\end{aligned}
$$

Here $J_F(h)$ denotes the **Jacobian** of the vector field $F$, with entries
$(J_F)_{ij} = \partial F_i / \partial h_j$ (equivalently written $(\nabla F)_{ij}$ in some
conventions, but not a gradient of a scalar). The matrix $\Omega$ is the antisymmetric
part of the Jacobian; vanishing of $\Omega$ is equivalent to symmetry of $J_F$.

The lemma requires $F$ to be at least $C^1$ — continuously differentiable. The
question is whether the attention force field satisfies this requirement.

If $F$ is not $C^1$, the lemma does not apply and the proof of Theorem 46 requires
modification. If $F$ is $C^\infty$ (smooth), the lemma applies with room to spare.

### 1.2 The Distinction Between the Attention Force and the Full Block

Theorem 46 analyzes a specific object:

$$
\begin{aligned}
\text{The per-head attention force:}\quad
F(h) &= \sum_{\mu=1}^{N} s_\mu(h) V^\mu - h \\[0.4em]
\text{where:}\qquad\quad
s_\mu(h) &= \mathrm{softmax}_\mu\bigl(\beta K^\mu \cdot h\bigr) \quad \text{(attention weights)} \\
K^\mu &= W_K \xi^\mu \qquad\qquad\ \text{(key for context pattern } \mu\text{)} \\
V^\mu &= W_V \xi^\mu \qquad\qquad\ \text{(value for context pattern } \mu\text{)} \\
\{\xi^\mu\} &= \text{fixed context patterns (not functions of } h \text{)}
\end{aligned}
$$

This is a function of $h$ alone, with the context treated as fixed. It is **not** the
full transformer block. The theorem does not require the FFN, LayerNorm, or
residual connection to be smooth — only this specific force field.

This distinction is critical and makes the smoothness case much simpler than
it would be for the full block.

---

## 2. Scaled Dot-Product Attention

The per-head map is $C^\infty$ (in fact $C^\omega$) in $h$ on all of $\mathbb{R}^d$; details below.

### 2.1 The Force Field

The per-head attention force is:

$$
F(h) = \sum_{\mu=1}^{N} s_\mu(h) V^\mu - h
$$

$$
s_\mu(h) = \frac{\exp\bigl(\beta K^\mu \cdot h\bigr)}{\sum_{\nu} \exp\bigl(\beta K^\nu \cdot h\bigr)} \qquad \text{(softmax)}
$$

where $\{K^\mu\}$, $\{V^\mu\}$, and $\beta$ are all fixed (functions of the weights and context,
not of the current hidden state $h$).

### 2.2 Component-by-Component Smoothness

**Step 1: Linear projections $W_Q h$, $K^\mu \cdot h$**

Each dot product $K^\mu \cdot h = \sum_i K^\mu_i h_i$ is a linear function of $h$.
Linear maps are $C^\infty$ — they are polynomials of degree 1.

**Step 2: The exponential $\exp\bigl(\beta K^\mu \cdot h\bigr)$**

Composition of $\exp\colon \mathbb{R}\to\mathbb{R}$ ($C^\infty$, analytic) with $K^\mu \cdot h$ ($C^\infty$).
Composition of smooth functions is smooth. Result: $C^\infty$, analytic.

**Step 3: The softmax $s_\mu(h)$**

$$
s_\mu(h) = \frac{\exp\bigl(\beta K^\mu \cdot h\bigr)}{\sum_{\nu} \exp\bigl(\beta K^\nu \cdot h\bigr)}
$$

This is a ratio of $C^\infty$ functions. The denominator $Z(h) = \sum_{\nu} \exp\bigl(\beta K^\nu \cdot h\bigr)$
is a sum of positive exponentials, so
$$Z(h) \ge \exp\Bigl(\beta \min_\nu (K^\nu \cdot h)\Bigr) > 0$$
for all $h \in \mathbb{R}^d$.

A $C^\infty$ function divided by a strictly positive $C^\infty$ function is $C^\infty$.

**Furthermore:** The softmax is actually **real-analytic** ($C^\omega$) — it equals
its Taylor series in a neighborhood of every point. This is strictly stronger
than $C^\infty$.

**Step 4: The weighted sum $\sum_\mu s_\mu(h) V^\mu$**

Each $V^\mu$ is a fixed vector (constant in $h$). The map $h\mapsto s_\mu(h) V^\mu$ is the
scalar function $s_\mu(h)$ times a constant vector — it is $C^\infty$ in $h$.
A finite sum of $C^\infty$ functions is $C^\infty$.

**Step 5: The full force $F(h) = \sum_\mu s_\mu(h) V^\mu - h$**

Subtracting the linear function $h$ (which is $C^\infty$) from a $C^\infty$ function gives
a $C^\infty$ function.

### 2.3 Conclusion

$$
\begin{aligned}
&\text{The per-head attention force } F(h) \text{ is } C^\infty \text{ (infinitely differentiable)} \\
&\text{and in fact } C^\omega \text{ (real-analytic) on all of } \mathbb{R}^d. \\[0.3em]
&\text{Poincaré's lemma requires } C^1. \\
&\text{The attention force satisfies } C^\omega \supset C^\infty \supset \cdots \supset C^1. \\
&\text{The smoothness assumption is satisfied with room to spare.}
\end{aligned}
$$

There are **no discontinuities, no kinks, no singularities** anywhere in the
per-head attention force as a function of $h$.

### 2.4 The Jacobian $\Omega^{\mathrm{att}}(h)$ Is Also Smooth

Since $F$ is $C^\infty$, its Jacobian $\partial F_i/\partial h_j$ is $C^\infty$. The antisymmetric part:

$$
\Omega^{\mathrm{att}}(h) = \frac{J_F(h) - J_F(h)^{\mathsf T}}{2} = \frac{\beta}{2} \sum_\mu s_\mu(h) \left( V^\mu \otimes K^\mu - K^\mu \otimes V^\mu \right)
$$

is also $C^\infty$ (so Theorem 46 is applied to this $F$; we write $\Omega^{\mathrm{att}}$ when
contrasting with Jacobians of other maps). The closed-form expression — derived by direct
differentiation of $F$ — is an attention-weighted superposition of constant
antisymmetric rank-2 matrices. It is valid everywhere on $\mathbb{R}^d$ and the theorem's
analysis of when it vanishes is rigorous.

---

## 3. LayerNorm — Smooth Almost Everywhere

### 3.1 Definition

Layer normalization:

$$
\mathrm{LayerNorm}(h) = \gamma \odot \frac{h - \mu(h)}{\sigma(h) + \varepsilon} + \beta
$$

$$
\mu(h) = \frac{1}{d}\sum_i h_i \qquad\qquad\ \text{(mean)}
$$
$$
\sigma(h) = \sqrt{ \frac{1}{d}\sum_i (h_i - \mu(h))^2 } \qquad \text{(standard deviation)}
$$
$$
\gamma, \beta \in \mathbb{R}^d\text{: learned scale and shift parameters} \qquad\quad \varepsilon > 0\text{: numerical stabilizer}
$$

### 3.2 With $\varepsilon > 0$ (Every Real Implementation)

Every practical transformer implementation uses $\varepsilon > 0$ (typical values: $10^{-5}$ or
$10^{-6}$). In that case the **denominator** in the standard formula is strictly positive:
$$\sigma(h) + \varepsilon \ge \varepsilon > 0 \quad \text{for all } h \in \mathbb{R}^d\text{.}$$
The **numerator** $h - \mu(h)\mathbf{1}$ is a polynomial in $h$; the **scale** $\sigma(h) = \sqrt{\varphi(h)}$ with
$\varphi(h) = \frac{1}{d}\|h - \mu(h)\mathbf{1}\|^{2} \ge 0$ is the Euclidean norm of a linear image of
$h$, hence $C^1$ on most of the space, but the map $\sqrt{\varphi}$ (before adding $\varepsilon$) has
the usual **non-smoothness at $\varphi = 0$** (the “cone” in the mean-zero subspace). After adding
the constant $\varepsilon$ to the denominator, the **ratio** $(h-\mu)/(\sigma+\varepsilon)$ is continuous and
$C^1$ in practice on a **full-measure open set**; a referee-tight **global** $C^\infty$ claim in one
line would be unwise without specifying an implementation.

Some codes instead replace $\sigma$ in the **denominator** by $\sqrt{\varphi(h) + \varepsilon'{}^{2}}$, which is
a composition of a **strictly positive** inner quantity with a smooth $\sqrt{\cdot}$ on
$(0,\infty)$ and is a standard way to obtain **$C^\infty$ regularity** of the output map on all of
$\mathbb{R}^{d}$ for fixed $\varepsilon' > 0$.

**Bottom line for this note:** LayerNorm (with any $\varepsilon>0$ in the **denominator path**) is as
regular as a trained pipeline needs for standard analysis; the **microscopic** global $C^\infty$
claim in older drafts is replaced here by: **$C^1$ on a dense open full-measure set** for the
literal $\sigma$ definition above, or **$C^\infty$ everywhere** for the
**$\sqrt{\varphi + (\varepsilon')^2}$-style** stabilized denominator, depending on the implementation.

### 3.3 Without $\varepsilon$ (Mathematical Idealization)

In the mathematical idealization $\varepsilon = 0$:

$$
\sigma(h) = 0 \quad \text{ when } h_1 = h_2 = \cdots = h_d = c \ \text{ for any } c \in \mathbb{R}
$$

$$
\text{Singular set: } S = \{ h \in \mathbb{R}^d : h = c \mathbf{1} \text{ for some } c \in \mathbb{R} \}
$$

$S$ is a one-dimensional subspace — a line through the origin in the direction
$(1, 1, \dots, 1)/\sqrt{d}$. On $S$, LayerNorm is undefined ($0/0$).

Properties of the singular set:

| Property | |
|---|---|
| Dimension | 1 (a line) |
| Codimension | $d-1$ (high codimension for large $d$) |
| Lebesgue measure | 0 (measure-zero in $\mathbb{R}^d$) |
| Physical relevance | negligible — random initialization and gradient updates essentially never produce $h$ with all components exactly equal |

### 3.4 Relevance to Theorem 46

LayerNorm is applied **after** the attention computation, not within the
attention force $F(h)$ analyzed by Theorem 46. The theorem states its force law
(102) explicitly and it does not include LayerNorm. Therefore the LayerNorm
smoothness question does not affect Theorem 46's proof.

For the broader dynamical analysis (full block dynamics, Appendix A), the
practical conclusion is: LayerNorm with stabilisation does **not** introduce a
**non-measure-zero** smoothness obstruction for the claims that matter; use the
$\sqrt{\varphi + \text{(small const)}^2}$ reading if a **global** $C^\infty$ block map is required.

---

## 4. FFN Nonlinearities — The Real Smoothness Issue

The feed-forward network introduces the only genuine non-smoothness in the
transformer block, and its severity depends on the activation function.

### 4.1 ReLU — $C^0$ but **not** $C^1$

$$
\mathrm{ReLU}(x) = \max(0, x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{if } x \le 0 \end{cases}
$$

**Continuity:** ReLU is continuous ($C^0$) everywhere.

**Differentiability:** ReLU has a kink at $x = 0$:

$$
\text{Left derivative: } \lim_{x\to 0^{-}} \mathrm{ReLU}'(x) = 0, \qquad
\text{Right derivative: } \lim_{x\to 0^{+}} \mathrm{ReLU}'(x) = 1
$$

The derivative is discontinuous at $x = 0$. ReLU is $C^0$ but not $C^1$ at $x = 0$.

For the full block with ReLU FFN, the set of non-differentiable points is:

$$
\text{Non-smooth set of FFN: } \{ h \in \mathbb{R}^d : (W_1 h + b_1)_i = 0 \ \text{for some } i \}
$$

This is a union of hyperplanes — a set of measure zero in $\mathbb{R}^d$.

**Consequence:** The full block with ReLU FFN is **not** $C^1$ on that union; **between**
its hyperfaces the map is piecewise-affine, hence $C^\infty$ **inside each open polyhedral
cell** (a dense open set of full measure). The complement of a union of hyperplanes
is open and dense, but is **not connected in general** (e.g. one hyperplane cuts space
in two), so Poincaré’s lemma in its usual **simply connected** form is applied
**cell-wise** (or locally) on a simply connected subregion, not as one blanket statement
on a disconnected set.

**Scope of Theorem 46 and $\Omega$:** Theorem 46 and its antisymmetric Jacobian
$\Omega^{\mathrm{att}}(h)$ are defined for the **per-head attention** force $F$ of §1.2, not
automatically for the full forward map of a ReLU+LayerNorm+residual **stack**—that map
has its own Jacobian; only use the $K\neq V$ / $\Omega^{\mathrm{att}}\neq 0$ conclusion for the object
$F$ that the paper analyses. *Architectural* claims about “generic non-conservativity” for
a **different** field require specifying that field first.

**GPT-2 does not use ReLU.** This concern is relevant to GPT-1, BERT, and
classical architectures, not to the specific model analyzed in the paper.

### 4.2 GELU — $C^\infty$ Smooth

GPT-2 uses GELU (Gaussian Error Linear Unit):

$$
\mathrm{GELU}(x) = x \Phi(x) \qquad
\text{where } \Phi(x) = \tfrac{1}{2}\bigl(1 + \mathrm{erf}(x/\sqrt{2})\bigr) \ \text{is the Gaussian CDF}
$$

**Smoothness of $\Phi$:** The error function $\mathrm{erf}$ is real-analytic ($C^\omega$) on all of $\mathbb{R}$.
It is defined by the integral of a Gaussian, which is smooth.

**Smoothness of GELU:** Product of $x$ ($C^\infty$) and $\Phi(x)$ ($C^\infty$) is $C^\infty$. In fact
GELU is $C^\omega$ (real-analytic) on all of $\mathbb{R}$.

**For GPT-2:** The FFN uses GELU — **scalar** activations are $C^\omega$. The **composed**
block (attention $+$ LayerNorm $+$ FFN) is as regular as the LayerNorm
implementation allows (§3.2, §7.3); in any case, **$C^1$** on a full-measure set and
certainly no worse than the ReLU class for standard theory.

### 4.3 SiLU / Swish — $C^\infty$ Smooth

Used in LLaMA, Mistral, and many modern architectures:

$$
\mathrm{SiLU}(x) = x \sigma(x) = \frac{x}{1 + e^{-x}}
\qquad \text{where } \sigma \text{ is the sigmoid: } C^\infty \text{, analytic}
$$

Product of $C^\infty$ functions $\Rightarrow$ $C^\infty$. No smoothness issues.

### 4.4 Summary by Architecture

| Architecture | Activation | FFN Smooth? |
|---|---|---|
| GPT-2 (analyzed in paper) | GELU | **$C^\infty$ everywhere** |
| GPT-1, BERT | ReLU | $C^0$, not $C^1$ at measure-zero set |
| LLaMA, Mistral | SiLU | **$C^\infty$ everywhere** |
| ReLU variants | ReLU | $C^0$, not $C^1$ at measure-zero set |

For the specific architecture analyzed in the Semantic Simulation paper
(GPT-2, which uses GELU), the FFN introduces no smoothness issues.

---

## 5. Causal Masking — Not a Discontinuity in h

### 5.1 The Masked Attention Formula

Causal (autoregressive) attention adds a position-dependent mask:

$$
\text{Masked attention: } \mathrm{softmax}\bigl( (QK^{\mathsf T} + M) / \sqrt{d_k} \bigr) V
$$
with the usual $Q, K$ of shapes $(n_{\text{pos}}\times d_k)$; $QK^{\mathsf T} + M$ is the logit matrix
before the row-wise $\mathrm{softmax}$ and multiplication by $V$.

$$
\text{where } M_{ij} = \begin{cases} 0 & \text{if } j \le i \ \text{(past tokens, attend normally)} \\ -\infty & \text{if } j > i \ \text{(future tokens, zeroed out)} \end{cases}
$$

### 5.2 Is This a Discontinuity?

The mask $M$ is a **fixed constant matrix** — it does not depend on the hidden
state $h$. It depends on position (which tokens attend to which), but at any
fixed token position, $M$ is a constant offset added to the attention logits.

**Effect on smoothness in $h$ (fixed token and head):** The logits (before the row-$\mathrm{softmax}$)
are affine in the relevant slice of the hidden state, hence $C^\infty$ in $h$; a standard form is
$$
\mathrm{softmax}\left( \frac{Q K^{\mathsf T} + M}{\sqrt{d_k}} \right) V, \qquad
Q\ \text{ an affine function of the query tokens in } h
$$
(aligned with the display in §5.1 with $Q$ built from the same $h$ the analysis fixes). In short:
**logits = affine in $h$**; adding constant $M$ and scaling by $1/\sqrt{d_k}$ leaves them $C^\infty$; $\mathrm{softmax} \circ$ (affine) is $C^\infty$.

The $-\infty$ entries in $M$ cause the corresponding softmax outputs to be exactly 0,
but this happens smoothly as $M_{ij}\to -\infty$ (a limit of smooth functions). In
practice, $\exp(-\infty)=0$ exactly, and the function is smooth in $h$ at each fixed
position.

**Conclusion:** Causal masking introduces no discontinuity in $h$. The attention
force $F(h)$ at any fixed token position is still $C^\infty$.

### 5.3 What Masking Does Affect

Causal masking creates a discontinuity in the **positional** structure — the
attention pattern changes discontinuously as a function of position index
(suddenly becoming 0 at the mask boundary). But this is a discrete jump in the
attention pattern over positions, not a smoothness failure in $h$.

For Theorem 46, which fixes the context and analyzes $F$ as a function of $h$ only,
causal masking is irrelevant.

---

## 6. Dropout — Training Only, Irrelevant at Inference

### 6.1 What Dropout Does

During training, dropout randomly zeros elements of hidden states:

$$
\mathrm{Dropout}(h)_i = \begin{cases} h_i / (1-p) & \text{with probability } 1-p \\ 0 & \text{with probability } p \end{cases}
$$

This is discontinuous (stochastic binary masking) and not differentiable.

### 6.2 Why It Does Not Affect the Theorem

1. Dropout is applied during training only. At inference time (evaluation), dropout is disabled: $\mathrm{Dropout}(h) = h$ (identity, $C^\infty$).
2. All experimental results in the paper — Theorem 42 verification, shared-$V_\psi$ tests, Jacobian symmetry tests, trajectory extraction — are computed on inference-time trajectories from pretrained models.
3. Theorem 46 concerns the structural properties of the force field at the trained model's inference-time computation.

Dropout is irrelevant to every smoothness claim in the paper.

---

## 7. What This Means for Theorem 46

### 7.1 The Scope of Theorem 46

Theorem 46 analyzes:

$$
F(h) = \sum_{\mu=1}^{N} s_\mu(h) V^\mu - h \qquad \text{(per-head attention force)}
$$

This is:

- The attention computation only (not LayerNorm, not FFN)
- A function of $h$ with context $\{K^\mu, V^\mu\}$ fixed
- Before any nonlinear block transformation

### 7.2 Smoothness Verdict for Theorem 46

| | |
|---|---|
| Component analyzed | $F(h) = \sum_\mu \mathrm{softmax}_\mu\bigl(\beta K^\mu \cdot h\bigr) V^\mu - h$ |
| Smoothness | $C^\omega$ (real-analytic) on all of $\mathbb{R}^d$ |
| Poincaré's lemma requires | $C^1$ |
| Conclusion | $F$ is vastly smoother than required. The lemma applies without qualification. The proof of Theorem 46 is not affected by any smoothness concern. |

### 7.3 Smoothness Verdict for the Full Block Dynamics (Appendix A)

For the broader analysis in Appendix A, which considers the full block including
LayerNorm and FFN:

| Component | |
|---|---|
| Attention (as analysed in the main paper) | $C^\omega$ in $h$ on all of $\mathbb{R}^d$ for the reduced per-head $F$ ✓ |
| LayerNorm | As in §3.2: $C^1$ on a dense open full-measure set for the **literal** $(h-\mu)/(\sigma{+}\varepsilon)$ with $\sigma=\sqrt\varphi$; **$C^\infty$ globally** for the $\sqrt{\varphi+\text{(stabiliser)}^2}$ reading ✓ (matches common implementations) |
| GELU FFN | $C^\omega$ in each scalar pre-activation; composed with $C^1$ (or $C^\infty$) sublayers ✓ |
| Residual | As smooth as the branch maps; addition preserves the minimum regularity of the addends ✓ |
| **Full block (GPT-2, typical graph)** | **At least** $C^1$ on a full-measure open set under the “smoothed denominator” reading of LayerNorm; **appendix** only needs $C^1$ of the *specified* field on its domain, not a global $C^\infty$ label for the entire stack. |

**Take-away:** A claim that the **entire** GPT-2 block is **$C^\infty$ on all of
$\mathbb{R}^d$** is stronger than this note *defends*; a claim that it is $C^1$ (or
real-analytic in sub-networks) on a **set full enough** for the dynamical theorems
is enough for Appendix~A, and the **attention** piece alone is $C^\omega$ in $h$ in the
reduced model of §1.2.

### 7.4 The One Genuine Smoothness Issue in the Broader Landscape

For architectures using ReLU activations (not GPT-2, but relevant for
architectural generality): the FFN is **$C^0$ globally, $C^1$ off a measure-zero
union of hyperplanes**, and **$C^\infty$ inside each open polyhedral cell** between
those loci. The loci themselves are not points of $C^1$ for the ReLU FFN (unless the
row happens not to straddle 0), so **Poincaré** is not applied to one
**globally** simply connected domain, but on **each** simply connected subregion
contained in a **single** cell, or in the **a.e. / local** sense used in applications.

**Do not** identify this with the **$K\neq V$** obstruction: Theorem~46 and
$\Omega^{\mathrm{att}}\neq 0$ refer to the **per-head attention** field $F$; a **full
ReLU+LayerNorm+residual** map has a **different** Jacobian, and a separate
conservativity analysis would have to be stated for **that** map.

**Mitigation for papers:** (a) state positive results (GELU, SiLU, $\tanh$) on
smooth activations explicitly; (b) for ReLU, restrict Poincaré / symmetry claims to
**a.e.** points, **local** simply connected sets, or **cell-wise** application; (c) keep
Theorem~46 (attention $F$) and any **architectural** “full block” story conceptually
separate.

---

## 8. Summary Table

| Component | Smooth in h? | Class | Affects Theorem 46? | Notes |
|---|---|---|---|---|
| Linear projection ($W_Q h$, $W_K \xi$, etc.) | Everywhere | $C^\omega$ | Yes — smooth ✓ | Polynomials of degree 1 |
| Dot product $K^\mu \cdot h$ | Everywhere | $C^\omega$ | Yes — smooth ✓ | Bilinear, analytic |
| Exponential $\exp\bigl(\beta K^\mu \cdot h\bigr)$ | Everywhere | $C^\omega$ | Yes — smooth ✓ | Analytic composition |
| Softmax $s_\mu(h)$ | Everywhere | $C^\omega$ | Yes — smooth ✓ | Denominator always $> 0$ |
| Full per-head attention $F(h)$ | Everywhere | $C^\omega$ | Yes — smooth ✓ | **Theorem 46 domain** |
| $\Omega^{\mathrm{att}}(h) = (J_F - J_F^{\mathsf T})/2$ for Thm.~46 $F$ | Everywhere | $C^\infty$ | Yes — smooth ✓ | For attention $F$; Eq.~(103) |
| LayerNorm | See §3.2–3.3 | $C^1$ a.e. (literal); $C^\infty$ global (stabilised $\sqrt{\varphi+\cdot}$ form); $\varepsilon=0$ singular on $h\propto\mathbf{1}$ | Not in T46 scope | T46 omits LayerNorm in $F$ |
| GELU FFN (GPT-2) | (scalar pre-acts) $C^\omega$; composed block: §3.2, §7.3 | — | Not in T46 scope | Used in paper’s GPT-2 |
| ReLU FFN | Complement of ReLU loci: $C^\infty$ per cell | a.e. / cell-wise | Not in T46 scope | $C^0$ through loci; not in GPT-2 |
| SiLU FFN (LLaMA etc.) | (scalar) $C^\omega$; composed as above | — | Not in T46 scope | Smooth activation ✓ |
| Causal mask | Everywhere (mask is constant in h) | $C^\infty$ | Not in T46 scope | Mask is position-fixed |
| Residual $h_{\ell+1} = h_\ell + \Delta h$ | Everywhere | $C^\infty$ | Not in T46 scope | Addition of $C^\infty$ |
| Dropout | Discontinuous (stochastic) | — | Not in T46 scope | Training only; disabled at inference |

**Key:** a.e. = almost everywhere (everywhere except a set of Lebesgue measure zero)

---

## 9. Technical Appendix: Smoothness Classes Defined

For reference, the smoothness hierarchy used throughout this note:

$$
C^\omega \ \supset\ C^\infty\ \supset\ C^k\ \supset\ \cdots\ \supset\ C^1\ \supset\ C^0
$$

| Class | Definition |
|---|---|
| $C^0$ | Continuous (no breaks or jumps) |
| $C^1$ | Continuously differentiable (derivative exists and is continuous) |
| $C^k$ | $k$-times continuously differentiable |
| $C^\infty$ | Infinitely differentiable (smooth) — every derivative exists |
| $C^\omega$ | Real-analytic — $C^\infty$ and equals its Taylor series in a neighborhood of every point. Strongest regularity class for real functions. |

**Why Poincaré's lemma needs $C^1$ and not just $C^0$:**

The lemma involves the Jacobian $J_F$ (entries $\partial F_i / \partial h_j$), which requires $F$ to
be differentiable. The symmetry condition $\Omega = 0$ for the antisymmetric part of
$J_F$ is meaningful when the partials are **continuous** — hence $C^1$ in the classical
lemma statement.

If $F$ is only $C^0$ (continuous but not differentiable), the Jacobian may not
exist, the lemma does not apply in the classical sense, and one must use
distributional/weak formulations. For the attention force $F(h)$, which is $C^\omega$,
none of these complications arise.

**Measure-zero sets and generic claims:**

A subset $S \subset \mathbb{R}^d$ has **Lebesgue measure zero** if it can be covered by open
sets of arbitrarily small total volume. Intuitively: zero probability of landing
on $S$ when drawing $h$ uniformly from any bounded region.

For ReLU networks, the non-smooth set is a union of hyperplanes
$\{ h : (W_1 h)_i = 0 \}$ — each is a $(d-1)$-dimensional flat subspace, which has
measure zero in $\mathbb{R}^d$. The smooth set is open (complement of closed hyperplanes)
and dense (every ball in $\mathbb{R}^d$ intersects it), but **need not be connected**
(one hyperplane already splits space). Generic arguments
(“for generic h”) use this measure-zero language; Poincaré is then applied **locally**
or **per cell** (§4.1, §7.4).

---

*Document version: April 2026.*  
*Prepared in connection with the smoothness analysis for Theorem 46*  
*of "Semantic Simulation: A Prescriptive Lagrangian Framework for Efficient Semantic Inference."*  
*Covers: scaled dot-product attention (GPT-2 GELU architecture), LayerNorm,*  
*FFN activations (ReLU, GELU, SiLU), causal masking, and dropout.*
