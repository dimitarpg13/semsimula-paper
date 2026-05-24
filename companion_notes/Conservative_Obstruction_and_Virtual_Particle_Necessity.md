# Conservative Obstruction and Virtual Particle Necessity

> **Summary.** We prove that no scalar potential on token particles can
> reproduce the three structural properties of scaled dot-product
> attention (asymmetric coupling, coupling-content decoupling, normalized
> budget). This conservative obstruction forces any particle-based
> framework seeking attention-like inter-token routing to extend its state
> space with auxiliary degrees of freedom. The Fock register mechanism
> (virtual particle exchange) is identified as the minimal such extension
> that restores all three properties while preserving conservative base
> dynamics.

**Paper reference:** §17 of `paper_v4/main.tex` (PARF-augmented SPLM),
subsection on the conservative obstruction theorem.

**Related documents:**
- [`Improving_the_Fock_Mechanism_to_match_Attention.md`](Improving_the_Fock_Mechanism_to_match_Attention.md)
  — design rationale for FockPARFLM v2 (Q/K/V creation protocol)
- Paper §9.4.2 — Fock space apparatus and expressivity hierarchy (v0/v1/v2/v3)

---

## 1. Setup and Motivation

The Semantic Simulation programme models language dynamics as a system of
**semantic particles** $h_1, \ldots, h_T \in \mathbb{R}^d$ evolving
under forces derived from a scalar potential $V_\theta$. In PARFLM, the
equation of motion for token $i$ is

$$
m \ddot{h}_i = -\nabla_{h_i} V_\theta(h_i) + \sum_{j \neq i} F_{ij}^{(\mathrm{PARF})} - m\gamma\dot{h}_i
$$

where $F_{ij}^{(\mathrm{PARF})} = -\nabla_{h_i} V_\phi(h_i, h_j)$ is a
conservative pairwise force derived from an inter-particle potential
$V_\phi$.

The question is: **can this conservative framework reproduce the
inter-token information routing that attention provides?** The answer is
no, and the obstruction is structural — it holds for any choice of
potentials $V_\theta$, $V_\phi$.

---

## 2. Three Structural Properties of Attention

Scaled dot-product attention computes, for each token $i$:

$$
\Delta h_i = \sum_{j=1}^{T} \alpha_{ij} W_V h_j, \qquad \alpha_{ij} = \frac{\exp(q_i \cdot k_j / \sqrt{d_k})}{\sum_{\ell} \exp(q_i \cdot k_\ell / \sqrt{d_k})}
$$

where $q_i = W_Q h_i$, $k_j = W_K h_j$, $v_j = W_V h_j$.

Three structural properties are immediate:

**(P1) Asymmetric coupling.** The influence of $j$ on $i$ (through
$\alpha_{ij}$) is generically different from the influence of $i$ on $j$
(through $\alpha_{ji}$), because $\alpha_{ij}$ depends on $q_i \cdot k_j$
while $\alpha_{ji}$ depends on $q_j \cdot k_i$, and $W_Q \neq W_K^T$ in
general.

**(P2) Coupling-content decoupling.** The coupling strength $\alpha_{ij}$
is determined by $W_Q$ and $W_K$ (query-key interaction), while the
content of the information transfer is determined by $W_V$ (value
projection). These three weight matrices are independently parameterizable.

**(P3) Normalized budget.** The softmax ensures
$\sum_{j=1}^{T} \alpha_{ij} = 1$ for every $i$, so the total influence
on each token is a convex combination — bounded independently of the
sequence length $T$.

---

## 3. The Conservative Obstruction Theorem

### Theorem (Conservative Obstruction)

> Let $h_1, \ldots, h_T$ in $\mathbb{R}^d$ be a system of
> particles with forces derived from a differentiable scalar potential
> $V : \mathbb{R}^{Td} \to \mathbb{R}$ via $F_i = -\nabla_{h_i} V$.
> Then the force map $\mathbf{F} = (F_1, \ldots, F_T)$ cannot
> simultaneously satisfy properties P1, P2, and P3.

The proof proceeds via three lemmas, each ruling out one property.

### Lemma 1 (Jacobian Symmetry — rules out P1)

> For any $V \in C^2(\mathbb{R}^{Td})$, the off-diagonal Jacobian blocks
> of the force map satisfy

$$
\frac{\partial F_i^\alpha}{\partial h_j^\beta} = \frac{\partial F_j^\beta}{\partial h_i^\alpha} \qquad \forall i \neq j, \quad \forall \alpha, \beta \in 1, \ldots, d.
$$

**Proof sketch.** The force on particle $i$ is
$F_i^\alpha = -\partial V / \partial h_i^\alpha$. Therefore

$$
\frac{\partial F_i^\alpha}{\partial h_j^\beta} = -\frac{\partial^2 V}{\partial h_j^\beta \partial h_i^\alpha} = -\frac{\partial^2 V}{\partial h_i^\alpha \partial h_j^\beta} = \frac{\partial F_j^\beta}{\partial h_i^\alpha}
$$

where the middle equality is Schwarz's theorem (equality of mixed
partial derivatives for $C^2$ functions). $\square$

**Why this rules out P1.** Consider the attention update viewed as a
force: $F_i^{(\mathrm{attn})} = \sum_j \alpha_{ij} W_V h_j$. The
off-diagonal Jacobian block is

$$
\frac{\partial F_i^{(\mathrm{attn})}}{\partial h_j} = \alpha_{ij} W_V + \sum_\ell \frac{\partial \alpha_{i\ell}}{\partial h_j} W_V h_\ell
$$

The first term involves $\alpha_{ij} W_V$, which depends on $j$'s
key and $i$'s query. The transposed block
$(\partial F_j^{(\mathrm{attn})} / \partial h_i)^T$ involves
$\alpha_{ji} W_V^T$, which depends on $i$'s key and $j$'s query. For
generic $W_Q \neq W_K^T$, these two blocks are not transposes of each
other. Therefore the attention force map violates Jacobian symmetry and
cannot be the gradient of any scalar potential.

### Lemma 2 (Coupling-Content Entanglement — rules out P2)

> Let $V = \sum_{i \lt j} V_\phi(h_i, h_j)$ be a pairwise potential.
> Then for each pair $(i, j)$, both the coupling strength (how much $j$
> influences $i$) and the content (in what direction $i$ is pushed) are
> determined by the single function $\nabla_{h_i} V_\phi(h_i, h_j)$.

**Proof sketch.** The pairwise force on $i$ from $j$ is

$$
F_{ij} = -\nabla_{h_i} V_\phi(h_i, h_j)
$$

The magnitude $\lVert F_{ij}\rVert$ (coupling strength) and the direction
$F_{ij}/\lVert F_{ij}\rVert$ (content) are both functionals of the same
gradient $\nabla_{h_i} V_\phi$. There is no free parameter to vary
the direction independently of the magnitude.

More precisely: given the potential $V_\phi$, the force $F_{ij}$ is
completely determined. Any reparameterization of $V_\phi$ that changes
the direction of $F_{ij}$ also changes its magnitude, and vice versa.
The coupling and content degrees of freedom are entangled in the
gradient.

In contrast, attention decouples these: $\alpha_{ij}$ (coupling) depends
on $W_Q$, $W_K$, while $W_V h_j$ (content) depends on $W_V$ — three
independently learnable matrices. $\square$

**Remark.** One might attempt to circumvent this by using a
vector-valued potential $\mathbf{V}: \mathbb{R}^{2d} \to \mathbb{R}^d$.
But then $F_{ij} = -\mathbf{V}(h_i, h_j)$ is no longer a conservative
force (it cannot be written as the gradient of a scalar), so one has
already left the conservative framework.

### Lemma 3 (Force Growth — rules out P3)

> Let $V_\phi : \mathbb{R}^{2d} \to \mathbb{R}$ be a pairwise potential
> with non-trivial forces. Then for generic configurations, the total
> force on each particle grows with $T$.

More precisely, if $\lVert\nabla_{h_i} V_\phi(h_i, h_j)\rVert \geq \epsilon$
for some $\epsilon \gt 0$ on a set of positive measure in
$\mathbb{R}^{2d}$, then $\lVert F_i\rVert = \Omega(T)$ as $T \to \infty$.

**Proof sketch.** Each pairwise force $F_{ij}$ contributes at least
$\epsilon$ in expectation. For generic (non-adversarial) configurations,
the forces do not cancel systematically, so
$\lVert F_i\rVert \geq c\epsilon\sqrt{T}$ by a random-walk argument, or
$\lVert F_i\rVert \geq c'\epsilon T$ if the forces are correlated in sign.

In either case, the total force on particle $i$ grows at least as
$\sqrt{T}$ with the number of particles. No scalar potential with
non-trivial pairwise interactions can produce forces that are $O(1)$
in $T$ for generic configurations.

In contrast, attention's softmax normalization guarantees

$$
\lVert\Delta h_i\rVert = \lVert\sum_j \alpha_{ij} W_V h_j\rVert \leq \lVert W_V\rVert \cdot \max_j \lVert h_j\rVert
$$

independently of $T$, since $\sum_j \alpha_{ij} = 1$ and each
$\alpha_{ij} \geq 0$. $\square$

**Remark on normalization within a potential.** One might attempt to
build softmax-like normalization into $V_\phi$ itself, for instance by
defining

$$
V_{\mathrm{norm}}(h_1, \ldots, h_T) = \sum_i \log \sum_j \exp\bigl(\phi(h_i, h_j)\bigr)
$$

This is the log-partition function of a Hopfield network and yields
forces whose magnitude is $O(1)$ in $T$. However, this construction:
(a) is not pairwise decomposable (it is a function of the entire
configuration), (b) reintroduces the Jacobian symmetry of Lemma 1,
and (c) conflates coupling and content per Lemma 2. The Hopfield
potential resolves P3 only at the cost of re-entrenching P1 and P2.

---

## 4. Corollary: State-Space Extension Is Necessary

> **Corollary.** Since the conservative obstruction holds for all
> $V \in C^2(\mathbb{R}^{Td})$, any particle-based dynamical framework
> that seeks to reproduce attention-like inter-token information routing
> must extend the state space beyond $h_1, \ldots, h_T$.

**Proof.** Direct from the theorem: if P1, P2, P3 cannot all be
satisfied by forces on the token particles alone, then the framework
must introduce additional degrees of freedom through which the
inter-token exchange is mediated. $\square$

The corollary does not yet specify what the extension should be. The
next section argues that virtual particle exchange (the Fock register
mechanism) is the natural and minimal choice.

---

## 5. Virtual Particle Exchange as the Minimal Extension

### Proposition (Fock Exchange Sufficiency and Minimality)

> The Fock register mechanism — a pool of $M$ auxiliary particles
> $r_1, \ldots, r_M$ with Q/K/V creation gates and a reverse
> channel — is a sufficient and minimal extension of the conservative
> particle framework that restores all three attention properties P1–P3.

**Sufficiency (proof sketch).**

The Fock mechanism implements a two-step information pathway:

$$
h_j \xrightarrow[\text{creation}]{\alpha_{kj},\, W_V} r_k \xrightarrow[\text{reverse channel}]{\alpha_{ik},\, W_V^{(\mathrm{reg})}} \Delta h_i
$$

- **P1 restored (asymmetry).** The creation gate uses $(q_k^{(\mathrm{reg})}, k_j^{(\mathrm{tok})})$
  while the reverse channel uses $(q_i^{(\mathrm{tok})}, k_k^{(\mathrm{reg})})$.
  These involve independent projection matrices, so the influence of $j$
  on $i$ (mediated through register $k$) is generically different from
  the influence of $i$ on $j$.

- **P2 restored (decoupling).** The coupling in each leg uses
  query-key dot products ($\alpha_{kj}$, $\alpha_{ik}$), while the content
  uses independent value projections ($W_V h_j$, $W_V^{(\mathrm{reg})} r_k$).
  Coupling and content are parameterized by separate weight matrices.

- **P3 restored (budget).** Both the creation gate and the reverse
  channel use softmax normalization: $\sum_j \alpha_{kj} = 1$ and
  $\sum_k \alpha_{ik} = 1$. The total force on each token is $O(1)$
  regardless of $T$.

**Minimality argument.**

We argue that the Fock mechanism is minimal in the sense that:

1. **Auxiliary state is necessary** (by the Corollary): some extension
   of $h_1, \ldots, h_T$ is required.

2. **The extension must carry content** (to restore P2): the auxiliary
   degrees of freedom must be able to store and transmit information
   independently of the coupling that routes it.

3. **The extension must have asymmetric ports** (to restore P1): the
   write interface (creation) must be structurally different from the
   read interface (reverse channel).

4. **The extension must be normalized** (to restore P3): both the write
   and read interfaces must include a softmax or equivalent budget
   mechanism.

Requirements 1–4 describe precisely a pool of auxiliary particles with
(a) a content vector, (b) an asymmetric write/read protocol, and
(c) normalized coupling — i.e., the Fock register mechanism. Any
simpler structure (e.g., a single shared register, or symmetric
read/write ports) would fail to restore at least one of P1–P3.

---

## 6. The QFT Correspondence

The structure uncovered by the theorem has a precise analogy in quantum
field theory, which is not coincidental but structural.

### 6.1 Virtual Photon Exchange in QED

In quantum electrodynamics (QED), two charged particles interact not by
direct action-at-a-distance but by exchanging virtual photons:

$$
e^- \xrightarrow[\text{emit}]{\text{vertex}} \gamma^* \xrightarrow[\text{propagate}]{\text{propagator}} \gamma^* \xrightarrow[\text{absorb}]{\text{vertex}} e^-
$$

The key structural features are:

| QED feature | Mathematical role |
|---|---|
| Vertex coupling | Asymmetric emission/absorption amplitudes |
| Photon polarization | Content (what is transferred) independent of coupling |
| Propagator $1/(k^2 - m^2)$ | Budget: exchange amplitude falls off with momentum transfer |
| Gauge invariance | Consistency constraint on the exchange mechanism |

### 6.2 Virtual Semantic Photon Exchange in Fock-PARFLM

The Fock register mechanism maps onto this structure:

| QED | Fock-PARFLM v2 |
|---|---|
| Charged particle $e^-$ | Token particle $h_j$ (emitter) or $h_i$ (absorber) |
| Virtual photon $\gamma^*$ | Register particle $r_k$ |
| Emission vertex | Creation gate: $r_k \leftarrow \sum_j \alpha_{kj} W_V h_j$ |
| Absorption vertex | Reverse channel: $Q_i = \sum_k \alpha_{ik} W_V^{(\mathrm{reg})} r_k$ |
| Photon polarization | Value projection $W_V h_j$ (content) |
| Vertex coupling constant | Softmax attention weight $\alpha_{kj}$ or $\alpha_{ik}$ |
| Photon lifetime | Register salience decay $\lambda$; lifetime $\tau \approx 1/(1-\lambda)$ layers |

### 6.3 Why the Analogy Is Structural, Not Metaphorical

The conservative obstruction theorem shows that the analogy is not a
post-hoc metaphor but a structural necessity:

1. **Conservative forces satisfy Newton III** (Lemma 1) — this is equivalent to
   direct electromagnetic interactions satisfying Newton III.
2. **Newton III is violated in QED** — the force between two charges is
   mediated by the electromagnetic field, which carries momentum and
   breaks the instantaneous action-reaction symmetry.
3. **The resolution in QED** is to introduce gauge bosons (photons) as
   dynamical degrees of freedom that mediate the interaction.
4. **The resolution in Semantic Simulation** is to introduce register
   particles (Fock mechanism) as dynamical degrees of freedom that
   mediate the inter-token exchange.

In both cases, the same mathematical obstruction (Jacobian symmetry of
gradient fields) forces the same structural resolution (auxiliary
mediating particles).

### 6.4 Doi-Peliti Connection

The Fock space formalism used here is the classical (Doi-Peliti)
specialization of second quantization, where:

- Creation $a_v^\dagger$ adds a particle of type $v$ to the state
- Annihilation $a_v$ removes one
- The field operator $\hat{\phi}(x) = \sum_v \phi_v(x) a_v$ represents
  the "semantic field" at position $x$

Attention can be recovered as a creation-propagation-annihilation
sequence:

$$
y_i = \sum_j \alpha_{ij} v_j \equiv \langle q_i \mid \hat{\phi}^\dagger \hat{\phi} \mid 0 \rangle
$$

where $\hat{\phi}^\dagger$ creates a virtual semantic photon from the
token values and $\hat{\phi}$ absorbs it at the receiving token. The
Fock mechanism makes this implicit exchange explicit and persistent.

---

## 7. Implications for the Semantic Simulation Programme

### 7.1 The Conservative Base Is Not Abandoned

The conservative obstruction does not invalidate the potential-based
framework. It shows that:

- The **base dynamics** (single-particle potential $V_\theta$, pairwise
  PARF forces $V_\phi$, damped Verlet integration) remains conservative
  and preserves all the geometric properties (Jacobi geodesics,
  Riemannian curvature) that the framework is built on.

- The **inter-token exchange** is a separate, non-conservative channel
  that augments the base dynamics. In the extended equation of motion:

$$
m\ddot{h}_i = \underbrace{-\nabla_{h_i} V_\theta(h_i)}_{\text{conservative}} + \underbrace{\sum_{j \neq i} F_{ij}^{(\mathrm{PARF})}}_{\text{conservative}} + \underbrace{Q_i}_{\text{non-conservative Fock exchange}} - m\gamma\dot{h}_i
$$

the Fock exchange force $Q_i$ sits in the generalized force slot of
the Euler-Lagrange equation — the same slot occupied by external driving
forces in classical mechanics.

### 7.2 The Jacobi Test as a Quantitative Diagnostic

For the conservative-by-construction architectures (SPLM, PARFLM without
Fock), the Jacobi test of §19 (Riemannian Geometry) applies directly:
trajectories are geodesics of the Jacobi metric.

For Fock-PARFLM, the Jacobi residual $\lVert R_t\rVert$ becomes a quantitative
diagnostic of how much the non-conservative exchange channel perturbs
the trajectories away from geodesic behaviour. A small residual would
indicate that the Fock channel is a perturbative correction; a large
residual would indicate that inter-token routing dominates the dynamics.

### 7.3 The Expressivity Hierarchy

The conservative obstruction theorem provides the structural motivation
for the expressivity hierarchy of §9.4.2:

| Level | Mechanism | Computational class | Conservative? |
|---|---|---|---|
| v0 | Fixed state space, $V_\theta + V_\phi$ | Regular (FA) | Yes |
| v2 | + Fock creation/destruction | Context-free (PDA) | Base: yes; exchange: no |
| v3 | + Lie-group operator actions | Mildly context-sensitive | Base: yes; actions: no |

The theorem explains why v0 cannot reach CFL: conservative forces
cannot implement the asymmetric, content-decoupled, budget-normalized
routing that pushdown operations require. The Fock mechanism provides
exactly the missing structural ingredients.

---

## 8. Summary

| Result | Statement |
|---|---|
| **Lemma 1** | Jacobian symmetry: $\partial F_i / \partial h_j = (\partial F_j / \partial h_i)^T$ for any $V \in C^2$ |
| **Lemma 2** | Coupling-content entanglement: force direction and magnitude are coupled in $\nabla V_\phi$ |
| **Lemma 3** | Force growth: $\lVert F_i\rVert = \Omega(\sqrt{T})$ for non-trivial pairwise potentials |
| **Theorem** | No $V \in C^2$ can satisfy P1 + P2 + P3 simultaneously |
| **Corollary** | State-space extension is necessary for attention-like routing |
| **Proposition** | Fock register mechanism is a sufficient and minimal extension |
| **QFT link** | The obstruction and its resolution mirror the gauge-boson mechanism of QED |
