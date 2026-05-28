# Improving the Fock Mechanism to Match Attention Expressivity in PARFLM

**Technical Report — Semantic Simulation Research Programme**  
**Status:** Active — May 2026 (updated with QFT v2.1 experiment results)  
**Relates to:** Paper v4 §§9.4.2, 17.8, 17.13, 17c; FockPARFLM Phase 1 (Dyck₂ seed 0 complete); QFT v2.1 (Q0–Q8 complete)

---

## Table of Contents

1. [Background and Motivation](#1-background-and-motivation)
2. [The PARFLM Expressivity Ceiling](#2-the-parflm-expressivity-ceiling)
3. [What Attention Actually Computes](#3-what-attention-actually-computes)
4. [Structural Properties Conservative Formalisms Cannot Replicate](#4-structural-properties-conservative-formalisms-cannot-replicate)
5. [The QFT Interpretation: Attention as Virtual Particle Exchange](#5-the-qft-interpretation-attention-as-virtual-particle-exchange)
6. [Current FockPARFLM: Architecture and Experimental Diagnosis](#6-current-fockparflm-architecture-and-experimental-diagnosis)
7. [Why the Current Fock Mechanism Fails as a PPL Lever](#7-why-the-current-fock-mechanism-fails-as-a-ppl-lever)
8. [The Core Reframe: Creation at the Wrong Level](#8-the-core-reframe-creation-at-the-wrong-level)
9. [Proposed Q/K/V-Structured Creation Protocol](#9-proposed-qkv-structured-creation-protocol)
10. [Asymmetric Non-Conservative Force Formulation](#10-asymmetric-non-conservative-force-formulation)
11. [Why This Could Improve Perplexity](#11-why-this-could-improve-perplexity)
12. [Proposed Experiments](#12-proposed-experiments)
13. [Theoretical Significance](#13-theoretical-significance)
14. [The Entropy Collapse Problem: Learnable Creation Temperature](#14-the-entropy-collapse-problem-learnable-creation-temperature)
15. [Field-Theoretic Proof of the Conservative Obstruction](#15-field-theoretic-proof-of-the-conservative-obstruction)
16. [QFT-Motivated Improvements to the Creation Gate](#16-qft-motivated-improvements-to-the-creation-gate)
17. [QFT v2.1 Experiment Results and Current Bottleneck Analysis](#17-qft-v21-experiment-results-and-current-bottleneck-analysis)

---

## 1. Background and Motivation

The **Semantic Simulation** research programme models language understanding as the motion of discrete semantic units — aspects, properties, particles, and structures — evolving through a bounded attractive scalar potential in a metric semantic space $\Sigma$. The core single-particle Lagrangian is:

$$\mathcal{L} = T - V, \qquad T = \tfrac{1}{2}m\lVert\dot{h}\rVert^2, \qquad V_\theta(h) = m\upsilon^2 \left(1 - e^{-\kappa\lVert h - h^*\rVert^2}\right)$$

where $h \in \mathbb{R}^d$ is the hidden state (position of the semantic particle in $\Sigma$), $h^*$ is its equilibrium centroid, $\upsilon$ is the characteristic speed, and $\kappa = f/\upsilon$ controls well curvature.

The **PARF-augmented SPLM (PARFLM)** enriches the single-particle picture by adding token-token pairwise interactions via a learned pair potential $V_\phi(h_t, h_s)$. This models repulsive and attractive forces between co-present semantic particles — particles that are semantically similar attract; dissimilar or contradictory particles repel. When we model the system by converting all force interactions into potential wells, this is valid **only under the assumption that all forces are conservative**:

$$F_i = -\nabla_i V, \qquad V_{\text{total}} = \sum_i V_\theta(h_i) + \sum_{i < j} V_\phi(h_i, h_j)$$

Dissipative forces (e.g. LayerNorm) and velocity-dependent forces are non-conservative and require additional machinery (Rayleigh dissipation function) beyond a simple potential. The PARFLM, as designed, assumes purely conservative PARF interactions.

Despite this enrichment, PARFLM remains bounded by a fundamental expressivity ceiling. The central motivation of this report is to understand why, to diagnose the failure of the first-generation Fock-space augmentation (FockPARFLM), and to design a principled replacement that can simulate all expressivity-enabling properties of the attention mechanism from within the particle-physics formalism.

---

## 2. The PARFLM Expressivity Ceiling

### 2.1 The v0 Ceiling Theorem

From Paper v4, §9.2 (Theorem v0-ceiling):

> PARFLM adds token-token pair interactions $V_\phi(h_t, h_s)$ to the single-particle scalar potential $V_\theta(\xi, h)$. This enriches the force law but does not escape the **v0 expressivity ceiling**:
> - The hidden state is still $h \in \mathbb{R}^d$ (fixed dimension)
> - The integrator is still a deterministic function
> - There is no mechanism for the state space to grow during inference

Consequently, PARFLM is **at most a finite automaton** (regular languages). It cannot:
- Recognise $\mathrm{Dyck}_n$ beyond the predicted collapse depth $D^*$
- Handle cross-serial dependencies ($a^n b^n c^n$)
- Reach the mildly context-sensitive (MCS) class

### 2.2 Empirical Confirmation: The P10 Ladder

The P10 ladder experiment (10 May 2026) provides decisive empirical confirmation of the ceiling. The P10h experiment — 20M tokens, 16k steps, full P5+P7+P8 stack — achieves:

$$\text{val PPL} = 26.43$$

This is **identical** to P10g (5M tokens, 16k steps, PPL = 26.42). Quadrupling the corpus produces **zero improvement**, confirming the v0 architectural ceiling. The 22M-parameter PARFLM has exhausted its representational capacity on TinyStories at approximately 26.4 PPL.

The gap to the matched attention baseline (MatchedGPT, val PPL = 7.81) is therefore **18.6 PPL** and can only be closed by escaping the expressivity class — not by scaling data, compute, or the conservative force law.

![Architecture Ladder: TinyStories PPL](images/fock_ppl_ladder.png)

---

## 3. What Attention Actually Computes

Before diagnosing what the Fock mechanism is missing, it is necessary to state precisely what attention computes. The standard scaled dot-product attention for token $i$ over a context of $T$ tokens is:

$$q_i = W_Q h_i \quad \text{(query — what token } i \text{ seeks)}$$

$$k_j = W_K h_j \quad \text{(key — what token } j \text{ advertises)}$$

$$v_j = W_V h_j \quad \text{(value — what token } j \text{ contributes when attended to)}$$

$$\alpha_{ij} = \frac{\exp\left(q_i \cdot k_j / \sqrt{d_k}\right)}{\displaystyle\sum_{k=1}^{T} \exp\left(q_i \cdot k_k / \sqrt{d_k}\right)}$$

$$y_i = \sum_{j=1}^{T} \alpha_{ij} v_j$$

The update $\Delta h_i = y_i$ is then applied to the hidden state. Three separate projection matrices ($W_Q, W_K, W_V$) decompose the role of each token into three independent functions: what it seeks, what it advertises, and what it contributes.

---

## 4. Structural Properties Conservative Formalisms Cannot Replicate

The gap between PARFLM (26 PPL) and attention (8 PPL) is not an implementation failure. It reflects three **structural** properties of attention that are categorically incompatible with any conservative pairwise force law.

### 4.1 Property 1 — Radical Asymmetry: Newton's Third Law is Broken

In any conservative pairwise potential $V_\phi(h_i, h_j)$, the forces on the two particles are:

$$F_{i \leftarrow j} = -\nabla_{h_i} V_\phi, \qquad F_{j \leftarrow i} = -\nabla_{h_j} V_\phi$$

Newton's Third Law enforces:

$$F_{i \leftarrow j} = -F_{j \leftarrow i}$$

This symmetry is **unavoidable** for any potential-derived force. Yet in attention:

$$\alpha_{ij} \neq \alpha_{ji} \quad \text{in general, and completely independently determined}$$

Token $i$ can attend overwhelmingly to $j$ while $j$ attends not at all to $i$. No conservative scalar $V_\phi$ can produce this. The asymmetry alone accounts for a substantial fraction of the expressivity gap.

### 4.2 Property 2 — Q/K/V Decoupling: Coupling ≠ Content Transferred

In PARFLM, the pair potential determines both **how strongly** particles interact and **what information** is exchanged — these are locked together:

$$F_{ij} = -\nabla_{h_i} V_\phi(h_i, h_j)$$

In attention, coupling strength and content transferred are **completely decoupled**:

- **Coupling strength:** $\alpha_{ij} = f(q_i, k_j)$ — determined by query/key inner product
- **Content transferred:** $v_j = W_V h_j$ — an entirely separate projection, independent of the coupling

Token $j$ may have high coupling with $i$ (large $\alpha_{ij}$) while transferring information from a completely different subspace (via $W_V$). This three-way decomposition is impossible to derive from a scalar potential.

### 4.3 Property 3 — Softmax: Global Competitive Normalization

The softmax normalization enforces $\sum_j \alpha_{ij} = 1$ — a **budget constraint**. Attending more to token $j$ necessarily means attending less to all other $k$. This is a non-local, competitive allocation mechanism.

Conservative potentials are additive and purely local: $V_\text{total} = \sum_{i<j} V_\phi(h_i, h_j)$. There is no global normalization and no competition between interactions. Sigmoid gates (as in the current creation gate) are independent per-register activations with no such constraint.

### 4.4 Summary Table

| Property | Conservative $V_\phi$ | Attention |
|---|---|---|
| **Asymmetry** | $\alpha_{ij} = \alpha_{ji}$ (forced) | $\alpha_{ij} \neq \alpha_{ji}$ (independent) |
| **Coupling vs. content** | Same object (gradient of $V$) | Independently parameterized ($W_Q W_K$ vs. $W_V$) |
| **Normalization** | Additive, no budget | Softmax: $\sum_j \alpha_{ij} = 1$ |
| **Expressivity class** | Regular (finite automaton) | $\mathrm{TC}^0$ (arbitrary depth) |

---

## 5. The QFT Interpretation: Attention as Virtual Particle Exchange

### 5.1 The Feynman Diagram of Attention

The three structural properties of attention map precisely onto **quantum field theory** (QFT). In quantum electrodynamics (QED), two electrons interact by exchanging a virtual photon:

$$e^- \xrightarrow{\text{emit}} \gamma_\text{virtual} \xrightarrow{\text{propagate}} \gamma_\text{virtual} \xrightarrow{\text{absorb}} e^-$$

The Feynman vertex factor at the source (emission), the propagator (what is carried), and the vertex factor at the receiver (absorption) are **three separate objects** — exactly the Q/K/V decoupling.

![Attention as Virtual Particle Exchange: QED analogy](images/fock_qed_attention_analogy.png)

### 5.2 Correspondence Table

| QFT concept | Attention equivalent |
|---|---|
| Charge / quantum number of receiver | Query $q_i = W_Q h_i$ |
| Charge / quantum number of emitter | Key $k_j = W_K h_j$ |
| Virtual particle / propagator payload | Value $v_j = W_V h_j$ |
| Vertex coupling constant | $\alpha_{ij} = \mathrm{softmax}(q_i \cdot k_j / \sqrt{d})$ |
| Feynman diagram: $i$ absorbs from $j$ | $\alpha_{ij} v_j$ contribution to $y_i$ |
| Global conservation (photon number) | Softmax budget: $\sum_j \alpha_{ij} = 1$ |

### 5.3 The Doi-Peliti Classical Specialisation

The Semantic Simulation framework commits to **classical** particles (no quantum superposition). The Doi-Peliti formalism (Doi 1976, Peliti 1985) provides exactly the required machinery: a Fock-space operator algebra for classical reaction-diffusion systems, where states are generating-function representations of configuration distributions and field equations are classical Hamilton equations on a symplectic manifold.

The full Fock-space algebraic machinery is therefore available **without invoking quantum mechanics**. The Fock space itself:

$$\mathcal{F}(\mathcal{H}) = \bigoplus_{n=0}^{\infty} \mathcal{H}^{\otimes n}$$

| v2 mechanism | Fock-space object |
|---|---|
| Introduce an entity into discourse | Creation operator $a^\dagger_v \lvert\psi\rangle$ |
| Entity drops out of discourse | Annihilation operator $a_v \lvert\psi\rangle$ |
| Count of currently-live entities | Number operator $\hat{N} = \sum_v a^\dagger_v a_v$ |
| Field at semantic position $x$ | $\hat{\phi}(x) = \sum_v \phi_v(x) a_v$ |

The key property that breaks the v0 ceiling: **the active particle count grows with input length**, so the state space is no longer fixed-dimensional.

---

## 6. Current FockPARFLM: Architecture and Experimental Diagnosis

### 6.1 Design Rationale

From Paper v4, §17.8, the FockPARFLM augments the PARFLM state with $M$ **latent register particles**:

$$\text{PARFLM:} \quad \text{state} = (h_1, \ldots, h_T) \in \mathbb{R}^{T \times d}$$
$$\text{FockPARFLM:} \quad \text{state} = (h_1, \ldots, h_T, r_1, \ldots, r_M) \in \mathbb{R}^{(T+M) \times d}$$

Registers start in a "vacuum" state (inactive). A learned creation gate activates them; a destruction gate deactivates them.

### 6.2 Forward Pass per Layer $\ell$

![FockPARFLM v1: Forward Pass per Layer](images/fock_forward_pass_v1.png)

Parameter budget at P10f scale ($d = 256$, $L = 8$, $M = 32$): **~288K overhead** (<2% of the ~13M base PARFLM).

### 6.3 Phase 1 Results: Dyck₂ Falsifier (Seed 0, 10 May 2026)

Configuration: $d = 64$, $L = 4$, $M = 16$, 4000 steps. Corpus: synthetic $\mathrm{Dyck}_2$, max nesting depth 12.

| Arm | Params | Val PPL | Deep-test acc. (depth 5–12) |
|---|---|---|---|
| F1-baseline (PARFLM, no registers) | 41,974 | 3.50 | 37.93% |
| F1-fock-nostack (bag, M=16) | 52,474 | 3.57 | 37.36% |
| **F1-fock-stack (LIFO, M=16)** | **52,474** | **3.43** | **39.22%** |

Key observations:
- **LIFO discipline is the active ingredient**: without the stack constraint, registers are inert extra parameters (bag ≈ baseline). The pushdown constraint is the critical mechanism.
- **The signal is positive but modest** (+1.3 pp). The LIFO arm overtakes baseline only after step 1600 of 4000 — gate MLPs need substantial training time to learn crisp activation timing.
- **The pre-registered success criterion** (>90% deep-test accuracy at depth 8+ with 3/3 seed consistency) is not yet met at 39.22%.

### 6.4 FockPARF Improvement Sweep: TinyShakespeare Results

From Paper v4, §17.13, five improvement strategies were tested against the FR4 baseline (190.2 PPL on TinyShakespeare):

| Arm | Strategy | Val PPL | GD convergence | Verdict |
|---|---|---|---|---|
| **P1** | Hybrid FockPARF + Attn (k=4 + m=4) | **149.2** | 100% | ✅ Matches attention baseline |
| P2 | $v_\text{hidden}=512$, 8000 steps | ~170 | 7–8% | ❌ Still 20+ PPL behind |
| P3 | $M=32$ + score entropy reg. | 224–248 | 0% | ❌ Regression |
| P4 | $d=256$ | ~174 | 7–8% | ❌ Still behind |
| P5 | Phased gate freeze | 223 | 75% | ❌ Regression |

**Structural conclusion from §17.13:**

> *"FockPARFLM's gate-based register lifecycle is not a PPL lever at TinyShakespeare scale. The only path to attention parity runs through hybridisation (P1), not through scaling standalone FockPARF. FockPARFLM's value lies in computational class (the v2 Dyck_n escape from the v0 ceiling), not in next-token perplexity."*

P1's diagnostic is especially revealing: the Hybrid model's $V_\theta$ landscape has range 0.26 (mean $\approx 0$) — an order of magnitude tighter than standalone FockPARF (range 5.3–9.5). The attention front-end contextualises so effectively that the PARF back-end operates with a near-constant potential well. This tells us the perplexity gap is almost entirely attributable to the absence of directed routing, not to any inadequacy in the conservative force law itself.

---

## 7. Why the Current Fock Mechanism Fails as a PPL Lever

The current creation gate:

$$g^{(\ell)}_\text{create} = \sigma\left(\mathrm{MLP}\left(\bar{h}_{1:T}\right)\right) \in [0,1]^M$$

is conditioned on the **mean of all tokens** — a single undifferentiated global field. This has three critical structural deficiencies that map exactly onto the three missing properties identified in Section 4:

| Missing property | Current gate failure | Consequence |
|---|---|---|
| Asymmetry | Every token contributes equally to $\bar{h}_{1:T}$ | No directional preference for any source token |
| Q/K/V decoupling | Gate determines both whether to create AND what content register holds | Coupling and content are fused |
| Competitive normalization | $M$ sigmoid gates are independent | No budget constraint; creation of register $k$ does not affect register $k'$ |

![Current FockPARFLM vs Attention: Structural Gaps](images/fock_structural_gaps.png)

The current FockPARFLM therefore uses creation/destruction to implement **auxiliary persistent memory** — registers are additional hidden states that persist across layers. This is computationally useful (it escapes the v0 ceiling via Dyck₂) but it does not implement **directed information exchange** — the mechanism that drives attention's language-modelling power.

---

## 8. The Core Reframe: Creation at the Wrong Level

The current FockPARFLM creates and destroys **input particles** (register slots as additional hidden states). The QFT analysis in Section 5 identifies that attention creates and destroys **virtual mediating particles** — semantic photons $\gamma$ that carry information from source $j$ to receiver $i$.

![FockPARFLM: Current vs Proposed mechanism](images/fock_current_vs_proposed.png)

| | Current FockPARFLM | Proposed FockPARFLM v2 |
|---|---|---|
| **What is created** | Register slot (real persistent particle) | Virtual semantic photon (transient) |
| **What is destroyed** | Register slot (when salience decays) | Virtual semantic photon (after absorption) |
| **Content at creation** | Learnable embedding (not from input) | $v_j = W_V h_j$ (drawn from source token $j$) |
| **Creation amplitude** | $\sigma(\mathrm{MLP}(\bar{h}))$ (global, symmetric) | $\alpha_{kj} = \mathrm{softmax}(q_k \cdot k_j)$ (directional) |
| **Persistence** | Across layers, salience-gated | Finite lifetime $\sim 1/(1-\lambda)$ layers |

---

## 9. Proposed Q/K/V-Structured Creation Protocol

### 9.1 Core Idea

Each register $r_k$ carries a **persistent query probe** $q_k = W_Q^{(k)} r_k \in \mathbb{R}^{d_k}$ that specifies what semantic content the register is "seeking." At each layer $\ell$, the creation event for register $k$ proceeds as a **query-driven attention readout** over the input tokens:

$$k_j = W_K h_j, \quad v_j = W_V h_j \qquad \forall j \in 1{:}T$$

$$\alpha_{kj} = \mathrm{softmax}_j\left(\frac{q_k \cdot k_j}{\sqrt{d_k}}\right) \qquad \left(\sum_j \alpha_{kj} = 1\right)$$

$$r_k \leftarrow \sum_{j=1}^{T} \alpha_{kj} \cdot v_j \qquad \text{(content of created register)}$$

$$\sigma_k \leftarrow \sigma_k \cdot \lambda + \max_j(\alpha_{kj}) \cdot (1 - \lambda) \qquad \text{(salience update)}$$

Register $k$ is **active** (created) iff $\sigma_k > \tau$; it is **destroyed** when salience decays below $\tau$.

### 9.2 How This Implements All Three Missing Properties

**Property 1 — Asymmetry.** The coupling $\alpha_{kj}$ depends on $q_k$ (what register $k$ seeks) and $k_j$ (what token $j$ advertises). Register $k$ reads from tokens; tokens do not symmetrically read back from register $k$ via the same operation. Newton's Third Law is broken by design.

**Property 2 — Q/K/V decoupling.** The coupling strength $\alpha_{kj}$ and the content transferred $v_j = W_V h_j$ are independently parameterized via separate weight matrices $W_K$ and $W_V$. A register can be strongly coupled to a token ($\alpha_{kj} \approx 1$) while drawing content from a completely different subspace of that token's representation.

**Property 3 — Competitive normalization.** The softmax over $j$ enforces $\sum_j \alpha_{kj} = 1$: attending more strongly to one source token necessarily reduces attention to all others. This is a budget constraint, not an independent gating.

### 9.3 What Is Novel Beyond Re-Implementing Attention

The critical distinction from standard attention is **temporal persistence across layers**. In standard attention, all exchange is instantaneous within a single layer — $\alpha_{ij}$ is recomputed from scratch at every layer. In the proposed protocol, a register created at layer $\ell$ with content $r_k = \sum_j \alpha_{kj} v_j$ carries that content forward until its salience decays. The characteristic memory lifetime is:

$$\tau_\text{lifetime} \approx \frac{1}{1-\lambda} \quad \text{layers}$$

For $\lambda = 0.9$, a register persists for approximately 10 layers. Standard attention corresponds formally to $\lambda = 0$ (instantaneous exchange). The destruction event is a **learned forgetting** mechanism — the register is destroyed when the context renders its content irrelevant:

$$g_\text{destroy}^{(k)} = \sigma\left(\mathrm{MLP}(r_k)\right), \qquad \sigma_k \leftarrow \sigma_k \cdot (1 - g_\text{destroy}^{(k)})$$

In QFT language: virtual photons have finite lifetime $\tau_\text{lifetime}$ governed by $\lambda$. Standard attention produces instantaneous photons; FockPARFLM v2 produces **long-lived semantic photons** that carry information across multiple integration steps, providing cross-layer working memory that standard attention lacks.

### 9.4 Architecture Specification

```python
class FockPARFLM_v2(SparsePARFLM):
    """FockPARFLM with Q/K/V-structured creation protocol."""

    def __init__(self, cfg: FockPARFConfig_v2):
        super().__init__(cfg)
        self.M = cfg.n_registers

        # Per-register query projections (what each register seeks)
        self.W_Q = nn.Parameter(torch.randn(self.M, cfg.d, cfg.d_k) * 0.02)

        # Shared key/value projections over input tokens
        self.W_K = nn.Linear(cfg.d, cfg.d_k, bias=False)
        self.W_V = nn.Linear(cfg.d, cfg.d,   bias=False)

        # Destruction gate: conditioned on register's own state
        self.destruction_gate = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.d, cfg.d // 4),
                nn.GELU(),
                nn.Linear(cfg.d // 4, 1),
                nn.Sigmoid()
            ) for _ in range(cfg.L)
        ])

        # Salience state (persistent across layers, initialized to vacuum)
        self.register_buffer('salience', torch.zeros(self.M))

    def create_registers(self, h_tokens, layer_idx):
        """Q/K/V-structured creation event."""
        T, d = h_tokens.shape

        # Key and value projections from input tokens
        K = self.W_K(h_tokens)                           # [T, d_k]
        V = self.W_V(h_tokens)                           # [T, d]

        r_new = []
        alpha_max = []
        for k in range(self.M):
            q_k = self.register_states[k] @ self.W_Q[k]  # [d_k]
            scores = (q_k @ K.T) / (self.cfg.d_k ** 0.5) # [T]
            alpha_k = F.softmax(scores, dim=-1)            # [T], sums to 1
            r_k_new = (alpha_k.unsqueeze(-1) * V).sum(0)  # [d]
            r_new.append(r_k_new)
            alpha_max.append(alpha_k.max().item())

        # Salience update: exponential decay + creation signal
        alpha_max_t = torch.tensor(alpha_max)
        self.salience = (self.salience * self.cfg.decay
                         + alpha_max_t * (1 - self.cfg.decay))

        active_mask = self.salience > self.cfg.threshold
        return torch.stack(r_new), active_mask
```

### 9.5 Full Forward Pass

![FockPARFLM v2: Full Forward Pass](images/fock_full_forward_pass.png)

---

## 10. Asymmetric Non-Conservative Force Formulation

### 10.1 Extended Equation of Motion

The Q/K/V creation protocol introduces a **generalized (non-conservative) force** on each token $i$. The extended equation of motion for semantic particle $i$ becomes:

$$\ddot{h}_i = \underbrace{-\nabla_{h_i} V_\theta(h_i)}_{\text{restoring force (conservative)}} + \underbrace{\sum_{j \neq i} F_{ij}^{(\mathrm{PARF})}}_{\text{pairwise PARF (conservative)}} + \underbrace{\sum_{k \in \mathrm{active}} \alpha_{ik} \cdot v_k^{(\mathrm{reg})}}_{\text{non-conservative Fock exchange}\ Q_i}$$

where:

$$\alpha_{ik} = \mathrm{softmax}_k\left(\frac{q_i \cdot k_k^{(\mathrm{reg})}}{\sqrt{d}}\right), \qquad v_k^{(\mathrm{reg})} = W_V^{(\mathrm{reg})} r_k$$

The third term $Q_i$ is the reverse channel: tokens read from active registers via attention-like coupling, completing the bidirectional but asymmetric exchange loop.

### 10.2 Why This Force is Non-Conservative

A force field $\mathbf{Q}(\mathbf{h})$ is conservative iff there exists a scalar $V$ such that $\mathbf{Q} = -\nabla V$. For the Fock exchange term:

$$Q_i = \sum_k \frac{\exp(q_i \cdot k_k^{(\mathrm{reg})} / \sqrt{d})}{\sum_{k'} \exp(q_i \cdot k_{k'}^{(\mathrm{reg})} / \sqrt{d})} \cdot v_k^{(\mathrm{reg})}$$

This force depends on the **relative inner products** across all active registers via the softmax — it cannot be expressed as the gradient of any scalar function of $h_i$ alone. Furthermore $Q_i \neq Q_j$ in general (asymmetry), so Newton's Third Law fails: no potential can be derived.

This is exactly the class of **non-conservative generalized forces** that the Lagrangian formalism accommodates via the generalized force term $Q_i$ in the Euler-Lagrange equations:

$$\frac{d}{dt}\frac{\partial \mathcal{L}}{\partial \dot{h}_i} - \frac{\partial \mathcal{L}}{\partial h_i} = Q_i$$

The first two terms on the right-hand side of the extended equation of motion ($V_\theta$ and $V_\phi$) remain conservative. Only the Fock exchange term breaks conservativity — and it does so in precisely the way required to match attention's three structural properties.

### 10.3 Connection to the Paper's Conservativity Obstructions (§15)

Paper v4, §15.6, catalogues six architectural obstructions that prevent attention from satisfying the shared-potential test (asymmetric couplings, multi-head split, causal mask, LayerNorm, distinct per-layer parameters, softmax). The Fock exchange force directly violates the conservativity condition: by construction, it is asymmetric ($\alpha_{ij} \neq \alpha_{ji}$) and softmax-normalized (global, non-local). This is by design — it is the precise mechanism needed to escape the v0 ceiling.

The shared-potential $R^2$ separator (SPLM median $R^2 = 0.90$ vs. GPT-2 median $R^2 = 0.19$) remains valid as a diagnostic: FockPARFLM v2, by deliberately introducing non-conservative forces, is predicted to produce $R^2$ values in the attention quadrant — confirming that the separator is mechanistically diagnostic, not merely a performance correlate.

---

## 11. Why This Could Improve Perplexity

### 11.1 The P1 Hybrid Diagnostic

The P1 result (Hybrid FockPARF+Attn, 149.2 PPL) provides the critical diagnostic. The attention front-end contextualises so effectively that the FockPARF back-end operates with a near-constant $V_\theta$ (range 0.26, mean $\approx 0$) — an order of magnitude tighter than standalone FockPARF (range 5.3–9.5). The implication: the gap between 26 PPL and 8 PPL is almost entirely attributable to the absence of directed routing, not to any inadequacy in the conservative force law itself.

By building Q/K/V-structured creation into the Fock mechanism, directed routing is injected without full attention — the register persists and decays rather than being recomputed from scratch at each layer.

### 11.2 What Remains Absent in Standalone FockPARFLM v2

| Feature | Q/K/V FockPARFLM | Standard attention |
|---|---|---|
| Asymmetric coupling | ✅ Yes (via register query) | ✅ Yes |
| Q/K/V decoupling | ✅ Yes | ✅ Yes |
| Softmax normalization | ✅ Yes (in creation step) | ✅ Yes |
| Multi-head split | ❌ No (single query per register) | ✅ Yes ($H$ heads) |
| Causal masking | ✅ Inherited from PARF | ✅ Yes |
| Per-layer reparameterization | ❌ Registers persist | ✅ Full recompute |
| Direct token-to-token exchange | ❌ Via registers only | ✅ Direct $h_i \leftarrow \alpha_{ij} v_j$ |
| Cross-layer working memory | ✅ Yes (via salience decay) | ❌ No |

The theoretical prediction: Q/K/V creation will **narrow but not close** the PPL gap to full attention. However, the mechanism should definitively move the PPL lever that the current gate mechanism cannot — producing improvement below the 26.4 PPL ceiling on TinyStories.

### 11.3 The Memory Lifetime Advantage

The proposed mechanism has one structural advantage over standard attention: cross-layer working memory. The register's information persists across layers with learned decay rate $\lambda$. Standard attention recomputes $\alpha_{ij}$ from scratch at every layer; registers carry their content forward, allowing information from early layers to influence late-layer dynamics without recomputation. Whether this advantage is empirically significant at TinyStories scale (short sequences, moderate depth) remains to be determined experimentally.

---

## 12. Proposed Experiments

![FockPARFLM v2 Experimental Programme](images/fock_experiment_gantt.png)

### 12.1 F2-qkv-creation: Dyck₂ Falsifier with Q/K/V Creation

**Goal:** Replace the mean-conditioned creation gate with the Q/K/V structured gate and re-run the Dyck₂ falsifier.

| Arm | Architecture | Expected result |
|---|---|---|
| F2-baseline | PARFLM (no registers) | ~37.9% deep-test acc (replication) |
| F2-fock-mean | FockPARFLM v1 (current gate) | ~39.2% (replication) |
| **F2-fock-qkv** | **FockPARFLM v2 (Q/K/V gate)** | **>50% deep-test acc (prediction)** |

**Success criterion:** F2-fock-qkv achieves >50% deep-test accuracy at depth 5–12, with 3/3 seed consistency.

### 12.2 F2-asymmetric-force: Bidirectional Asymmetric Exchange

Add the reverse channel — tokens read from active registers via attention-like coupling:

$$\Delta h_i^{(\text{reg})} = \sum_{k \in \text{active}} \frac{\exp(q_i \cdot k_k^{(\text{reg})} / \sqrt{d})}{\sum_{k'} \exp(q_i \cdot k_{k'}^{(\text{reg})} / \sqrt{d})} \cdot v_k^{(\text{reg})}$$

This completes the bidirectional but asymmetric exchange loop and constitutes the full non-conservative generalized force $Q_i$ in the equation of motion.

### 12.3 P11-qkv: TinyStories at P10f Scale

**Goal:** Determine whether Q/K/V-structured creation breaks the 26.4 PPL ceiling on TinyStories.

```bash
# P11-qkv: FockPARFLM v2, M=16, Q/K/V creation gate
python train_fock_parf_v2.py \
  --corpus tinystories --arch fock_v2 \
  --n-registers 16 --v-hidden 1024 \
  --qkv-creation --d-key 64 \
  --steps 16000 --seed 0

# P11-qkv-32: M=32 registers
python train_fock_parf_v2.py \
  --corpus tinystories --arch fock_v2 \
  --n-registers 32 --v-hidden 1024 \
  --qkv-creation --d-key 64 \
  --steps 16000 --seed 0
```

**Success criterion:** Any improvement below 26.4 PPL with pure FockPARFLM v2 (no hybrid attention). A result of ~20 PPL would be highly significant; ~15 PPL would indicate the mechanism genuinely captures most of attention's directed-routing expressivity.

---

## 13. Theoretical Significance

### 13.1 Attention is the Unique Minimal Mechanism

The analysis in Sections 3–5 establishes that attention is the **unique minimal mechanism** satisfying three independently necessary properties: asymmetric directional coupling (breaking Newton's Third Law), Q/K/V decoupling (coupling ≠ content transferred), and softmax global normalization (budget constraint). No conservative force law can satisfy all three. Any mechanism that does is, up to parameterization, **functionally equivalent to attention** — the "force formalism" and attention converge to the same mathematical structure from different starting points.

This is not a failure of the particle formalism; it is a **positive result**: attention can be derived from first principles within the Fock-space framework as the unique classical virtual-particle exchange mechanism satisfying the three structural requirements.

### 13.2 Formal Connection: Attention = Fock Virtual Particle Exchange

The Q/K/V creation protocol makes this equivalence precise and constructive. Attention, re-read in Fock space language, is:

$$y_i = \sum_j \alpha_{ij} v_j \equiv \int dx \langle q_i | x \rangle \hat{\phi}(x) |0\rangle$$

where $\hat{\phi}(x) = \sum_j v_j \delta(x - k_j)$ is the semantic photon field created by the source tokens, and $\langle q_i | x \rangle = \mathrm{softmax}(q_i \cdot x / \sqrt{d})$ is the absorption amplitude at receiver $i$. This is a **creation–propagation–annihilation** sequence recoverable exactly from the framework's Doi-Peliti Fock space without invoking new mathematical structures.

### 13.3 The Expressivity Hierarchy

![FockPARFLM Expressivity Hierarchy](images/fock_expressivity_hierarchy.png)

### 13.4 The Separator Remains Diagnostic

The shared-potential $R^2$ separator (Paper v4, §§13–14, TMLR submission target) is unaffected by this analysis. SPLM satisfies the conservative shared-potential condition ($R^2 = 0.90$); GPT-2 violates it ($R^2 = 0.19$). FockPARFLM v2, by deliberately introducing non-conservative forces via the Fock exchange term, is predicted to produce $R^2$ values in the attention quadrant — confirming that the separator is mechanistically diagnostic of conservativity, not merely a performance correlate.

### 13.5 Path to Full MCS Reach

From Paper v4, §9.4.3, the full mildly context-sensitive class requires a third mechanism beyond v2 (creation/destruction):

$$\text{MCS} = \underbrace{\text{v0}}_{\text{conservative dynamics}} + \underbrace{\text{v2}}_{\text{Fock creation/destruction}} + \underbrace{\text{v3}}_{\text{Lie group operator actions on register groups}}$$

The v3 mechanism maps to **non-abelian gauge theory**: register groups transform under learned group actions rather than scalar gates. This is the next major theoretical development, planned for Paper v5.

---

## 14. The Entropy Collapse Problem: Learnable Creation Temperature

### 14.1 Empirical Diagnosis: Uniform Attention in the Creation Gate

The D1–D5 TinyStories debug experiment (d=256, L=8, M=16, T=256, 2000 steps on A100) revealed a critical problem: the Q/K/V creation gate's attention is **near-uniform at every layer**. The normalised entropy (0 = peaked/useful, 1 = uniform/wasted) measured at inference after training:

| Layer | D1 (M=16) | D3 (d_k=128) |
|---|---|---|
| 0 | 1.000 | 1.000 |
| 1 | 0.996 | 0.937 |
| 2 | 0.997 | 0.951 |
| 3 | 0.995 | 0.985 |
| 4–7 | 0.995–0.997 | 0.993–0.995 |

At normalised entropy 0.996, the **effective number of attended tokens** is $n_\text{eff} = \exp(H) \approx 254$ out of $T = 256$ — the register is attending to essentially all tokens equally. The register content $r_k = \sum_j \alpha_{kj} v_j$ degenerates into $r_k \approx \frac{1}{T} \sum_j v_j$, which is identical to the mean-conditioned gate from FockPARFLM v1. Despite implementing the full Q/K/V protocol, the mechanism collapses back to mean-pooling.

Yet even this degraded form produces a genuine PPL lift: D1 (52.6 PPL) beats D5 baseline (61.0 PPL) by $-8.4$ PPL ($-13.8\%$). The Q/K/V structure helps — but the attention cannot focus.

### 14.2 Why the Standard $1/\sqrt{d_k}$ Scaling Fails for Register Queries

The temperature $1/\sqrt{d_k}$ was introduced by Vaswani et al. (2017) to prevent softmax saturation. The derivation assumes that query and key entries are independent with zero mean and unit variance, so that the dot product $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ has variance $d_k$. Dividing by $\sqrt{d_k}$ normalises the variance to 1, placing the scores in a regime where softmax is neither saturated nor uniform.

This assumption **fails categorically** for the Fock creation gate. The key difference is the **scale of the queries**:

**Standard attention (Transformer):**
- $W_Q \in \mathbb{R}^{d \times d_k}$, initialised with Xavier/Kaiming (std $\approx 1/\sqrt{d}$)
- Token embeddings $h_i$ have $\lVert h_i \rVert \sim \sqrt{d}$ after LayerNorm
- Query norm: $\lVert q_i \rVert = \lVert W_Q h_i \rVert \sim \sqrt{d_k}$ — the queries are $O(1)$ per dimension

**Fock creation gate:**
- $W_Q^{(k)} \in \mathbb{R}^{d \times d_k}$, initialised with std $= 0.02$ (`register_init_scale`)
- Register states $r_k$ initialised at std $= 0.02$
- Query norm: $\lVert q_k \rVert = \lVert W_Q^{(k)} r_k \rVert \sim d \cdot (0.02)^2 \approx 0.1$

At $d = 256$, $d_k = 64$, with `init_scale = 0.02`:

$$\text{Var}(q_k \cdot k_j) = d_k \cdot \text{Var}(q_{k,i}) \cdot \text{Var}(k_{j,i}) \approx d_k \cdot (0.02)^4 \cdot \lVert r_k \rVert^2 \cdot \lVert h_j \rVert^2$$

Empirically, the raw score standard deviation at initialisation is:

$$\sigma_s \approx 0.009$$

The standard scaling divides by $\sqrt{d_k} = 8$, yielding effective score standard deviation:

$$\sigma_\text{eff} = \frac{\sigma_s}{\sqrt{d_k}} = \frac{0.009}{8} = 0.001$$

### 14.3 The Softmax Entropy–Temperature Relationship

For scores $s_1, \ldots, s_T$ drawn from a distribution with standard deviation $\sigma_s$, the softmax at temperature $\tau$ is:

$$p_j(\tau) = \frac{\exp(s_j / \tau)}{\sum_{k=1}^T \exp(s_k / \tau)}$$

The normalised entropy $\bar{H}(\tau) = H(\tau) / \ln T$ is a monotonically increasing function of $\tau$ (monotonically decreasing in $1/\tau$). The effective score standard deviation $\sigma_\text{eff} = \sigma_s / \tau$ determines the entropy:

| $\sigma_\text{eff}$ | $\bar{H}$ (normalised entropy) | $n_\text{eff} = e^H$ | Regime |
|---|---|---|---|
| 0.001 | 1.0000 | 256.0 | Fully uniform — mean-pool |
| 0.01 | 1.0000 | 256.0 | Still uniform |
| 0.1 | 0.999 | 255 | Barely selective |
| 0.5 | 0.979 | 228 | Weakly selective |
| 1.0 | 0.907 | 153 | Moderately selective |
| 2.0 | 0.552 | 21 | Strongly selective |
| 3.0 | 0.208 | 3.2 | Highly peaked |
| 5.0 | 0.029 | 1.2 | Near one-hot |

The table reveals the core problem. With the standard $1/\sqrt{d_k}$ scaling ($\sigma_\text{eff} = 0.001$), the creation gate is **firmly in the uniform regime** — and it stays there even after training. After 2000 training steps, the weight norms grow by roughly $10\times$, pushing $\sigma_s$ to $\sim 0.09$. But $\sigma_\text{eff} = 0.09/8 = 0.011$ — still completely uniform.

The creation gate needs $\sigma_\text{eff} \geq 1$ to reach the moderately selective regime ($n_\text{eff} \lesssim 150$) and $\sigma_\text{eff} \geq 2$ for strongly selective attention ($n_\text{eff} \lesssim 20$). With the fixed $1/\sqrt{d_k}$ scaling, this requires score standard deviations of $\sigma_s \geq 8$ — an $800\times$ increase from initialization that takes far more than 2000 steps to achieve organically through gradient flow.

### 14.4 Why This Problem Does Not Arise in Standard Attention

In a Transformer, the scaling $1/\sqrt{d_k}$ works because the query and key projections are initialised with Xavier/Kaiming scaling (std $\sim 1/\sqrt{d}$), and the token embeddings are $O(\sqrt{d})$ in norm. The resulting score standard deviation at initialisation is:

$$\sigma_s \approx \sqrt{d_k} \cdot \frac{1}{\sqrt{d}} \cdot \sqrt{d} = \sqrt{d_k}$$

After dividing by $\sqrt{d_k}$: $\sigma_\text{eff} \approx 1$ — exactly in the moderately selective regime. The Transformer's $1/\sqrt{d_k}$ scaling was **calibrated for Transformer-scale initialisation**. The Fock creation gate uses a deliberately small `init_scale = 0.02` to ensure the register mechanism starts gently and does not destabilise the PARF dynamics. This conservative initialisation is correct for training stability, but it renders the standard temperature scaling useless.

### 14.5 The Fix: Learnable Log-Space Temperature

Replace the fixed $1/\sqrt{d_k}$ with a **learnable temperature** $\tau$ parameterised in log-space:

$$\alpha_{kj} = \mathrm{softmax}_j\left(\frac{q_k \cdot k_j}{\tau}\right), \qquad \tau = \exp(\theta_\tau), \qquad \theta_\tau \in \mathbb{R}$$

where $\theta_\tau$ is a learnable scalar initialised to $\ln(\tau_0)$. The log-space parameterisation ensures $\tau > 0$ without explicit clamping and provides a natural multiplicative learning dynamic: equal gradient steps produce equal proportional changes in $\tau$.

**Initialisation:** $\tau_0 = 0.1$ (i.e. $\theta_\tau = \ln 0.1 \approx -2.3$).

At $\tau_0 = 0.1$ with $\sigma_s = 0.009$: $\sigma_\text{eff} = 0.009 / 0.1 = 0.09$. This is still in the near-uniform regime at initialisation, but it is $80\times$ sharper than the standard $1/\sqrt{d_k}$ scaling. More importantly, the score $\sigma_\text{eff}$ scales as $\sigma_s / \tau$ — as the weights grow during training, the sharpening compounds. At $\sigma_s = 0.1$ (after modest training), $\sigma_\text{eff} = 1.0$ — entering the moderately selective regime.

**The critical advantage of learnability:** The model can discover the optimal temperature through backpropagation. The gradient:

$$\frac{\partial \mathcal{L}}{\partial \theta_\tau} = \frac{\partial \mathcal{L}}{\partial \tau} \cdot \tau$$

provides a natural signal. If the loss benefits from sharper attention (lower $\tau$), the gradient pushes $\theta_\tau$ negative, decreasing $\tau$ exponentially. If the model needs uniform attention at some stage (e.g. early training when scores are noisy), it can increase $\tau$. The fixed $1/\sqrt{d_k}$ scaling offers no such adaptivity.

### 14.6 Connection to Temperature Scaling in the Literature

The learnable temperature parameter connects to several established techniques:

**Gumbel-Softmax** (Jang et al. 2017; Maddison et al. 2017). In the Gumbel-Softmax framework, temperature $\tau$ controls the "hardness" of discrete selection: $\tau \to 0$ approximates argmax (hard selection), $\tau \to \infty$ produces uniform soft weighting. The creation gate temperature serves an analogous role — it controls whether registers select specific tokens (low $\tau$, hard attention) or average over all tokens (high $\tau$, soft attention).

**Cosine similarity temperature** (Radford et al. 2021, CLIP; He et al. 2020, MoCo). Contrastive learning uses a learnable temperature to scale the logits in the InfoNCE loss. CLIP's temperature starts at $\tau = 0.07$ and is learned — converging to values that maximise the mutual information between the two modality embeddings. The creation gate temperature serves the same structural role: maximising the mutual information between registers and the tokens they attend to.

**Attention temperature in efficient attention** (Zhai et al. 2021). Scaling Vision Transformers uses per-head learnable temperatures to replace $1/\sqrt{d_k}$, finding that different heads converge to different temperatures — some sharply peaked, others nearly uniform. This suggests the optimal temperature is **task-dependent and layer-dependent**, motivating learnability over any fixed value.

### 14.7 Effective Number of Attended Tokens and Information Capacity

The effective number of attended tokens $n_\text{eff} = \exp(H)$ determines the **information-theoretic capacity** of each register. Under the data processing inequality:

$$I(r_k; h_{1:T}) \leq d \cdot \ln(n_\text{eff})$$

where $I$ denotes mutual information and $d$ is the embedding dimension. At $n_\text{eff} = T = 256$ (uniform attention), the register can carry $d \ln T \approx 1408$ nats — but spread diffusely across all tokens, providing low signal-to-noise for any individual token's semantics. At $n_\text{eff} = 5$ (sharply peaked), the register carries $d \ln 5 \approx 412$ nats — concentrated on a small number of tokens, providing high signal-to-noise for specific contextual cues.

For next-token prediction, the relevant information is typically concentrated in a few key tokens (subject for verb agreement, antecedent for pronoun resolution, opening bracket for bracket matching). Registers that can **selectively attend** to these tokens are far more useful than registers that average over the entire context. The learnable temperature allows the model to discover the optimal $n_\text{eff}$ for each task.

### 14.8 D3 Evidence: Wider Queries Partially Compensate

The D3 arm ($d_k = 128$, double the baseline) provides partial corroboration. With wider queries, the score standard deviation increases by $\sqrt{2}$:

$$\sigma_s \propto \sqrt{d_k} \implies \sigma_s^{(d_k=128)} \approx \sqrt{2} \cdot \sigma_s^{(d_k=64)}$$

This produces a modest entropy reduction at layer 1 (0.937 vs D1's 0.996) and a PPL improvement from 52.6 to 49.9. The mechanism is the same — increasing $\sigma_\text{eff}$ — but achieved through a larger projection dimension (which costs more parameters) rather than through temperature (which costs a single scalar). The learnable temperature is the more efficient intervention: **one parameter instead of $M \cdot d \cdot \Delta d_k$** additional projection parameters.

### 14.9 Updated Architecture: Creation Gate with Learnable Temperature

The updated Q/K/V creation equations from §9.1 become:

$$\alpha_{kj} = \mathrm{softmax}_j\left(\frac{q_k \cdot k_j}{\tau}\right), \qquad \tau = \exp(\theta_\tau), \quad \theta_\tau \in \mathbb{R}$$

replacing the previous:

$$\alpha_{kj} = \mathrm{softmax}_j\left(\frac{q_k \cdot k_j}{\sqrt{d_k}}\right)$$

The `FockPARFConfig_v2` dataclass gains a new field:

```python
tau_create_init: Optional[float] = 0.1   # None → fallback to 1/√d_k
```

The `QKVCreationGate` stores:

```python
self.log_tau = nn.Parameter(torch.tensor(tau_create_init).log())
```

and applies it during forward:

```python
tau = self.log_tau.exp().clamp(min=1e-4)
scores = scores / tau   # replaces: scores / (d_k ** 0.5)
```

Setting `tau_create_init = None` recovers the original $1/\sqrt{d_k}$ behaviour for backward compatibility.

### 14.10 The D6 Experiment

The D6 arm of the TinyStories debug notebook tests this fix directly:

| Arm | Configuration | Purpose |
|---|---|---|
| D1 | FockPARF v2, M=16, $1/\sqrt{d_k}$ scaling | Baseline (entropy $\approx 1.0$) |
| **D6** | **FockPARF v2, M=16, $\tau_0 = 0.1$ (learnable)** | **Tests B1 fix** |

**Predictions:**
- Attention entropy at layer 1 should drop from 0.996 (D1) to $< 0.95$ (D6)
- If $\tau$ learns to decrease further, entropy could reach 0.5–0.7 range ($n_\text{eff} \sim 20\text{–}100$)
- PPL should improve beyond D1's 52.6, potentially approaching D3's 49.9 without the extra projection parameters
- If $\tau$ grows (model prefers uniform attention), the fix has no effect — ruling out B1 as the binding bottleneck

The temperature evolution curve $\tau(t)$ is logged at every training step, providing a direct readout of whether the model wants sharper or broader attention.

---

## 15. Field-Theoretic Proof of the Conservative Obstruction

This section provides a self-contained, detailed account of the **field-theoretic proof** that attention generates irreducibly interacting statistics which no conservative potential can replicate. It synthesises the dynamical-systems proof (our Conservative Obstruction Theorem, Section 4 above) with the independent QFT analysis of Ageev and Ageev (2026), showing that the two results establish the same impossibility from complementary directions.

### 15.1 Overview: Two Proofs, One Obstruction

The Conservative Obstruction Theorem (Section 4) works "from the inside" — it examines the force law $F_i = -\nabla_{h_i} V$ and shows that gradient structure imposes three constraints (Jacobian symmetry, gradient entanglement, force growth) that are individually violated by attention. The field-theoretic proof works "from the outside" — it examines the statistical correlations produced by a layer of dynamics and shows that conservative potentials produce free (Gaussian) field statistics while attention produces interacting (non-Gaussian) field statistics, and the two classes are separated by a sharp boundary.

![Two Proofs of the Conservative Obstruction](images/fock_two_proofs.png)

The two proofs are logically independent but mutually reinforcing. The dynamical-systems proof tells us WHY (which structural properties fail); the field-theoretic proof tells us that the failure is IRREDUCIBLE (no reparameterisation, resummation, or mean-field reduction can bridge the gap).

### 15.2 Setup: The NN-QFT Correspondence for Attention

Following Ageev and Ageev (2026), we construct a scalar field theory from a single attention head. The setup proceeds in three steps.

**Step 1: Define the field.** Given a trained attention head with query, key, and value weight matrices $W^Q, W^K, W^V \in \mathbb{R}^{d \times d_k}$, define a scalar field as a linear readout of the head output:

$$\phi_i(x_a) = \sum_j \alpha_{aj} (W^V x_j)_i$$

where $x_a$ is an input token at position $a$, the index $i$ selects a coordinate of the value-projected output, and the attention weights $\alpha_{aj}$ are defined by the softmax over Q/K inner products:

$$\alpha_{aj} = \mathrm{softmax}(x_a^\top W^Q {W^K}^\top x_j / \sqrt{d_k})$$

Each coordinate $i$ defines a separate scalar field.

**Step 2: Define the ensemble.** The $n$-point correlation functions are defined by averaging over Gaussian-initialised network parameters:

$$\langle \phi_{i_1}(x_{a_1}) \cdots \phi_{i_n}(x_{a_n}) \rangle = \mathbb{E}_{W^Q, W^K, W^V} \left[ \phi_{i_1}(x_{a_1}) \cdots \phi_{i_n}(x_{a_n}) \right]$$

The weights $W^Q, W^K, W^V$ are drawn i.i.d. from $\mathcal{N}(0, \sigma^2 / d)$. This is the standard NN-QFT prescription: the network parameter ensemble plays the role of the path-integral measure.

**Step 3: Compute correlators.** The central object is the **connected four-point function** (the part of the four-point correlator that cannot be decomposed into products of two-point functions):

$$G_c^{(4)}(x_{a_1}, x_{a_2}, x_{a_3}, x_{a_4}; i_1, i_2, i_3, i_4) = \langle \phi_{i_1} \phi_{i_2} \phi_{i_3} \phi_{i_4} \rangle - \langle \phi_{i_1} \phi_{i_2} \rangle \langle \phi_{i_3} \phi_{i_4} \rangle - \langle \phi_{i_1} \phi_{i_3} \rangle \langle \phi_{i_2} \phi_{i_4} \rangle - \langle \phi_{i_1} \phi_{i_4} \rangle \langle \phi_{i_2} \phi_{i_3} \rangle$$

This is the standard QFT definition. In a free (Gaussian) theory, $G_c^{(4)} = 0$ by Wick's theorem (also known as Isserlis' theorem in probability). Any nonzero $G_c^{(4)}$ signals an **interacting** theory.

### 15.3 The Two-Point Function: Gaussian Baseline

The two-point function of a single attention head is:

$$\langle \phi_i(x_a) \phi_j(x_b) \rangle = \sigma_V^2 \sum_{u,v} \langle \alpha_{au} \alpha_{bv} \rangle (x_u \cdot x_v) \delta_{ij} / d$$

where $\langle \alpha_{au} \alpha_{bv} \rangle$ denotes the average of the product of attention weights over the Q/K weight ensemble. Since $W^V$ is independent of $W^Q$ and $W^K$, the value projection contributes a simple factor proportional to $\delta_{ij}$ (Kronecker delta on the output coordinate indices). The nontrivial structure is in the attention-weight correlator.

If the attention weights were **independent** of each other (i.e. if $\alpha_{au}$ were not coupled to $\alpha_{bv}$ through the shared Q/K weights), then the four-point function would factorise exactly into products of two-point functions, and $G_c^{(4)}$ would vanish. The theory would be free.

### 15.4 The Four-Point Function: Independence-Breaking

The central result of Ageev and Ageev (2026) is that $G_c^{(4)} \neq 0$ for generic attention heads, and the nonzero contribution has a transparent origin.

The connected four-point function decomposes as:

$$G_c^{(4)} = \underbrace{I_{d_k}^{(4)}}_{\mathcal{O}(1/d_k)} + \underbrace{I_{\mathrm{IB}}^{(4)}}_{\text{finite as } d_k \to \infty}$$

Two terms contribute:

1. **The finite-width correction** $I_{d_k}^{(4)}$: This is $\mathcal{O}(1/d_k)$ and vanishes in the infinite key-dimension limit. It arises from the same mechanism as finite-width corrections in standard MLPs (non-Gaussian corrections from the central limit theorem).

2. **The independence-breaking term** $I_{\mathrm{IB}}^{(4)}$: This is the structurally novel contribution. It survives the $d_k \to \infty$ limit and has the explicit form:

$$I_{\mathrm{IB}}^{(4)} \propto \mathrm{Cov}_{W^Q, W^K}(X_{12}, X_{34})$$

where

$$X_{ab} = \frac{\sigma_V^2}{d} \sum_{u,v} \alpha_{au} \alpha_{bv} (x_u \cdot x_v)$$

The quantity $X_{ab}$ is a random variable (random because $\alpha_{au}$ depends on the random weights $W^Q, W^K$) that measures the attention-mediated inner product between positions $a$ and $b$. The independence-breaking term is the covariance of two such attention-mediated inner products, computed over the Q/K weight ensemble.

### 15.5 Why the Independence-Breaking Term is Nonzero

The key insight is structural: the softmax attention weights $\alpha_{aj}$ are **shared** across all output coordinates $i$ of the head. Different coordinates of the value-projected output pass through the **same** attention routing matrix. This shared routing creates statistical coupling between output coordinates that cannot be reduced to pairwise (two-point) correlations.

Concretely, consider two pairs of field coordinates: the pair $\phi_{i_1}(x_a)$, $\phi_{i_2}(x_b)$ and the pair $\phi_{i_3}(x_c)$, $\phi_{i_4}(x_d)$. Each pair is correlated through the attention weights. But the attention weights appearing in $X_{ab}$ and $X_{cd}$ are **the same random functions of the same Q/K weights**. Therefore $X_{ab}$ and $X_{cd}$ are not independent, and their covariance is generically nonzero:

$$\mathrm{Cov}_{W^Q, W^K}(X_{ab}, X_{cd}) \neq 0$$

This covariance vanishes **only** if the attention weights "freeze" — i.e. become deterministic functions independent of $W^Q, W^K$. But frozen attention is trivial attention (it does not attend; it applies a fixed mixing matrix). Therefore, whenever attention actually attends, $I_{\mathrm{IB}}^{(4)} \neq 0$, and the field theory is interacting.

### 15.6 Diagrammatic Interpretation: Wick Contractions

The difference between free and interacting field theories has a clean diagrammatic interpretation. In a **free (Gaussian) field theory**, Wick's theorem (Isserlis' theorem) states that all $n$-point functions decompose into products of two-point functions:

$$\langle \phi_1 \phi_2 \phi_3 \phi_4 \rangle_{\text{free}} = \langle \phi_1 \phi_2 \rangle \langle \phi_3 \phi_4 \rangle + \langle \phi_1 \phi_3 \rangle \langle \phi_2 \phi_4 \rangle + \langle \phi_1 \phi_4 \rangle \langle \phi_2 \phi_3 \rangle$$

Diagrammatically, each two-point function is a line (propagator) connecting two external points. The three terms correspond to the three possible pairings (Wick contractions) of four points into two pairs. There is no connected four-point vertex — all four-point structure reduces to products of two-point propagators.

In an **interacting (non-Gaussian) field theory**, there is an additional connected component — a four-point vertex where all four external legs meet at a common interaction point:

$$\langle \phi_1 \phi_2 \phi_3 \phi_4 \rangle_{\text{interacting}} = \underbrace{\langle \phi_1 \phi_2 \rangle \langle \phi_3 \phi_4 \rangle + \text{perms}}_{\text{disconnected: Wick contractions}} + \underbrace{G_c^{(4)}}_{\text{connected: interaction vertex}}$$

The connected vertex $G_c^{(4)} = I_{\mathrm{IB}}^{(4)}$ is the signature of irreducible interaction. In particle physics, this vertex represents virtual particle exchange: two particles scatter by exchanging a mediator that is created and destroyed during the interaction.

![Wick contraction diagrams: free (Gaussian) vs interacting (non-Gaussian) field theory](images/wick_contractions_free_vs_interacting.png)

**Figure 15.1.** Wick contraction diagrams for free vs interacting field theories. In the free (Gaussian) case, all four-point correlations decompose into products of two-point propagators. In the interacting case, an irreducible connected four-point vertex $I_{\mathrm{IB}}$ appears, representing virtual particle exchange. This is the field-theoretic signature of attention.

### 15.7 The Bridge: Conservative Potentials Produce Free Field Statistics

This is the central bridge connecting the two proofs. We now show that **any conservative scalar potential on token particles produces a free (Gaussian) field theory** in the NN-QFT sense.

Consider $T$ particles evolving under a scalar potential $V : \mathbb{R}^{Td} \to \mathbb{R}$ with forces $F_i = -\nabla_{h_i} V$. Define an output field as in Section 15.2 but replacing the attention head with the conservative dynamics. The force on particle $i$ due to particle $j$ is:

$$F_{ij}^\alpha = -\frac{\partial V}{\partial h_j^\beta} \cdot \frac{\partial^2 V}{\partial h_i^\alpha \partial h_j^\beta}$$

For pairwise potentials $V = \sum_{i \lt j} V_\phi$, the key structural constraint is that the force $F_{ij}$ and the "content" transferred are **the same object** — both are determined by the gradient of $V_\phi$. There is no independent "value" channel.

Now consider the correlator structure. In a conservative system, the coupling between particles $i$ and $j$ is mediated entirely through the gradient of the potential. The Hessian matrix

$$H_{ij}^{\alpha\beta} = \frac{\partial^2 V}{\partial h_i^\alpha \partial h_j^\beta}$$

is the fundamental two-point coupling kernel. By Schwarz's theorem, this matrix is symmetric: $H_{ij}^{\alpha\beta} = H_{ji}^{\beta\alpha}$.

The symmetry of the Hessian means that the response of particle $i$ to perturbation of particle $j$ is the transpose of the response of $j$ to perturbation of $i$. When one computes the correlation functions of the conservative system's output field (by averaging over initial conditions or parameter distributions), all higher-point correlations reduce to products of two-point correlations mediated by the Hessian. The connected four-point function vanishes:

$$G_c^{(4)}\Big|_{\text{conservative}} = 0$$

This is because the symmetric Hessian produces a Gaussian effective theory. The force map is a gradient, so there exists a partition function $Z = \int e^{-V(h)/T} dh$ (in the equilibrium or path-integral sense), and the correlations of any observable linear in the particle positions follow Wick's theorem with propagator $H^{-1}$.

### 15.8 The Impossibility: Interacting Cannot Reduce to Free

The field-theoretic proof is now a one-line consequence:

> **Conservative dynamics produces $G_c^{(4)} = 0$ (free field theory). Attention produces $G_c^{(4)} \neq 0$ (interacting field theory). Since $0 \neq \mathrm{nonzero}$, no conservative potential can replicate the statistics of attention.**

This is a **sharper** result than the dynamical-systems proof in one specific sense: it shows that the obstruction is not merely about the force law but about the **entire statistical structure** of the dynamics. Even if one could somehow construct a non-gradient force law that satisfies P1 and P2 while remaining derivable from a scalar energy, the correlator structure would still be Gaussian, and Gaussian statistics cannot reproduce the non-Gaussian correlations that attention generates.

The interacting/free boundary in QFT is rigid:

- A free theory has all connected $n$-point functions vanishing for $n \geq 3$.
- An interacting theory has at least one nonvanishing connected $n$-point function for some $n \geq 3$.
- There is no continuous deformation from free to interacting within the class of theories derivable from a quadratic (Gaussian) action. Interaction requires adding vertices (cubic, quartic, etc.) to the action — these vertices correspond to virtual particle exchange.

### 15.9 Multi-Head Suppression and the Register Pool Size

Ageev and Ageev (2026) establish an important quantitative result: multi-head averaging suppresses the non-Gaussianity. If $N_h$ attention heads are averaged, then:

$$G_c^{(4)} = \mathcal{O}(1/N_h)$$

This has direct architectural implications for FockPARFLM. Each Fock register in the register pool plays the role of a single "interaction channel" — analogous to a single attention head — that mediates non-conservative exchange forces between token particles. The NN-QFT result tells us:

1. **Each register contributes a finite non-Gaussian correction.** A single Fock register with Q/K/V gates mediates a single channel of virtual particle exchange, producing a connected four-point contribution analogous to $I_{\mathrm{IB}}^{(4)}$ from one attention head.

2. **The total interaction strength scales with register count.** To match the full expressivity of multi-head attention with $N_h$ heads, one needs a register pool whose size $M$ scales with the number of independent interaction modes the target task requires.

3. **The $1/N_h$ suppression explains why hybridisation works.** In the P1 hybrid experiment (Section 6.4), adding just 4 attention heads to FockPARFLM reduced PPL from 190 to 149 — each head contributes a large non-Gaussian correction. The FockPARF registers, by contrast, need to learn the interaction structure from scratch.

### 15.10 The Three Structural Properties Revisited Through QFT

The three structural properties identified in Section 4 can now be reinterpreted in field-theoretic language:

| Property | Dynamical-systems view | Field-theoretic view |
|---|---|---|
| **P1: Asymmetry** | Jacobian symmetry forces reciprocity: off-diagonal blocks are transposes | The propagator is symmetric; the interaction vertex $G_c^{(4)}$ can break this via directional exchange |
| **P2: Q/K/V decoupling** | Gradient entanglement locks coupling strength to content direction | In free field theory, the coupling kernel (Hessian) is a single object; interaction vertices introduce separate coupling constants at each leg |
| **P3: Normalised budget** | Additive forces grow as $\Omega(\sqrt{T})$ | Free-field propagator is $O(1)$ in particle number but does not compete; interacting vertices carry their own normalisation through the self-energy |

The field-theoretic perspective reveals that these three properties are not independent accidents — they are all consequences of the single structural distinction between free and interacting field theories. A free field theory (conservative potential) is symmetric, entangled, and unnormalised because its entire structure is determined by a single quadratic form (the Hessian). An interacting field theory (attention) breaks all three because the interaction vertices introduce genuinely new degrees of freedom (the Q/K/V weight matrices play the role of coupling constants at the vertices).

### 15.11 Implications for the Semantic Simulation Programme

![Conservative potential vs attention vs FockPARFLM in field-theoretic language](images/fock_qft_correspondence_diagram.png)

**Figure 15.2.** The three regimes in field-theoretic language. A conservative potential produces free (Gaussian) field statistics where all correlations factorise through the potential ($G_c^{(4)} = 0$). Attention produces interacting (non-Gaussian) statistics with irreducible four-point vertices ($G_c^{(4)} \neq 0$), mediated by virtual exchange through shared softmax routing. FockPARFLM's register pool is engineered to produce $G_c^{(4)} \neq 0$ by construction, via explicit creation and annihilation of exchange mediators.

The field-theoretic proof sharpens the architectural roadmap for the Semantic Simulation programme:

1. **The conservative base dynamics is necessary but inherently limited.** The PARF potential $V_\phi$ provides a well-defined energy landscape and conservative base forces. But these forces produce free-field statistics only. The base dynamics forms the "vacuum sector" of the theory.

2. **Fock registers are the interaction vertices.** Each register with Q/K/V gates acts as an interaction vertex that generates non-Gaussian connected correlations. The register's creation corresponds to emitting a virtual quantum; its annihilation (or readback via the reverse channel) corresponds to absorbing it.

3. **The register pool must be rich enough.** The $1/N_h$ scaling result means that the register pool size $M$ must grow with the complexity of the target interaction structure. For simple formal languages (Dyck$_2$), 16 registers suffice. For natural language, $M$ must scale with the effective number of independent interaction channels.

4. **The free/interacting boundary is the expressivity wall.** No amount of scaling the conservative potential (larger $d_V$, deeper MLP, richer pair potential) can cross from free to interacting statistics. The wall is topological in nature — it requires a qualitative change (adding interaction vertices = Fock registers), not a quantitative one (larger parameters).

### 15.12 Summary: The Complete Obstruction

The Conservative Obstruction Theorem and its field-theoretic proof together establish a complete no-go result:

> **Theorem (Conservative Obstruction — combined form).** Let $T$ particles $\lbrace h_1, \ldots, h_T \rbrace \subset \mathbb{R}^d$ evolve under a $C^2$ scalar potential $V : \mathbb{R}^{Td} \to \mathbb{R}$ with forces $F_i = -\nabla_{h_i} V$. Then:
>
> (a) The force map $\mathbf{F}$ cannot simultaneously satisfy the asymmetric coupling (P1), coupling-content decoupling (P2), and normalised budget (P3) properties of attention. (Dynamical-systems proof: Schwarz symmetry, gradient entanglement, force growth.)
>
> (b) The output field generated by the conservative dynamics has vanishing connected four-point function: $G_c^{(4)} = 0$. The corresponding field theory is free (Gaussian). (Field-theoretic proof: Hessian symmetry implies Wick factorisation.)
>
> (c) The output field of a single attention head has $G_c^{(4)} \neq 0$ (generically, whenever the head attends). The corresponding field theory is interacting (non-Gaussian). (Ageev and Ageev, 2026: independence-breaking through shared softmax routing.)
>
> Therefore, no conservative potential can replicate the statistics generated by attention. The resolution requires extending the state space with auxiliary degrees of freedom (Fock registers) that mediate virtual particle exchange, producing non-Gaussian connected correlations by construction.

This combined result delimits the theoretical boundary that any attention-free conservative architecture must cross. Fock-space augmentation, with its explicit creation and annihilation of exchange mediators, is the minimal mechanism that crosses this boundary within the Lagrangian framework.

---

## 16. QFT-Motivated Improvements to the Creation Gate

The field-theoretic proof of Section 15 and the NN-QFT analysis of Ageev and Ageev (2026) provide not only theoretical understanding of the conservative obstruction but also **concrete architectural prescriptions** for improving the FockPARFLM creation gate. This section translates four QFT insights into specific design changes for the Q/K/V creation protocol of Section 9 and the learnable-temperature fix of Section 14.

### 16.1 QFT Diagnosis of the Current Creation Gate

The entropy collapse problem (Section 14.1) has a precise field-theoretic interpretation. When the creation gate attention is near-uniform ($\bar{H} \approx 1$), the attention weights become effectively deterministic, frozen at $\alpha_{kj} \approx 1/T$ for all $j$. In Ageev and Ageev's framework, frozen attention produces a vanishing independence-breaking term:

$$I_{\mathrm{IB}}^{(4)} \propto \mathrm{Cov}_{W^Q, W^K}(X_{12}, X_{34}) \to 0 \quad \text{when } \alpha_{kj} \to 1/T$$

because $X_{ab} = (\sigma_V^2/d) \sum_{u,v} \alpha_{au} \alpha_{bv} (x_u \cdot x_v) \to (\sigma_V^2/d) \cdot (1/T^2) \sum_{u,v} (x_u \cdot x_v)$ becomes a **constant** independent of the Q/K weights. The covariance of a constant is zero.

In QFT language: the creation gate is operating in the **free-field regime**. It creates "virtual particles" that carry no non-Gaussian correlation; the registers mediate no genuine interaction. The register content degenerates to the mean-pool $r_k \approx (1/T) \sum_j v_j$, which is exactly the Gaussian (free-field) limit of the theory.

The learnable temperature (Section 14.9) re-introduces the independence-breaking covariance by sharpening attention. But the QFT analysis reveals three additional structural deficiencies in the creation gate that temperature alone cannot fix.

### 16.2 Improvement 1: Gumbel-Softmax Creation — Stochastic Virtual Particle Emission

**QFT motivation.** In quantum field theory, particle creation is inherently **stochastic**: the creation amplitude $\langle f | a^\dagger | i \rangle$ gives a probability amplitude, and the actual creation event is a random process governed by vacuum fluctuations. The current creation gate is entirely deterministic: given the same input tokens, every forward pass produces the same register contents. This means all $M$ registers with similar query directions will create near-identical content, wasting the register pool.

**The prescription.** Replace the deterministic softmax in the creation gate with the **Gumbel-Softmax** relaxation:

$$\alpha_{kj} = \mathrm{softmax}_j\left(\frac{q_k \cdot k_j + g_j}{\tau}\right), \qquad g_j \sim \mathrm{Gumbel}(0, 1)$$

where $g_j = -\log(-\log(u_j))$ with $u_j \sim \mathrm{Uniform}(0,1)$. The Gumbel noise serves as the **vacuum fluctuation**: it perturbs the deterministic score landscape, ensuring that:

1. Different registers, even with similar Q/K projections, **sample different tokens** during creation (diversity across the pool)
2. At training time, the stochasticity provides exploration, preventing all registers from collapsing to the same content
3. As $\tau \to 0$, creation approaches hard one-hot selection (particle created at a specific semantic location)
4. As $\tau \to \infty$, creation approaches the uniform distribution (no exchange: free-field limit)

The Gumbel noise has standard deviation $\pi/\sqrt{6} \approx 1.28$, which is comparable to the score standard deviations needed for selective attention ($\sigma_{\mathrm{eff}} \geq 1$, see Section 14.3). At initialization, when $\sigma_s \approx 0.009$, the Gumbel noise completely dominates the scores, producing random token selection. As training proceeds and Q/K projections sharpen ($\sigma_s$ grows), the learned scores increasingly dominate the noise, transitioning from **random creation** (exploration) to **informed creation** (exploitation). This annealing schedule is automatic and requires no tuning.

**Field-theoretic justification.** The independence-breaking term $I_{\mathrm{IB}}^{(4)}$ is maximized when different registers produce diverse, token-specific content. Deterministic creation with shared projections produces correlated register contents; Gumbel noise breaks this correlation, increasing the effective number of independent interaction channels per layer.

**Implementation:**

```python
def create_registers_gumbel(self, h_tokens, layer_idx, training=True):
    K = self.W_K(h_tokens)
    V = self.W_V(h_tokens)
    tau = self.log_tau.exp().clamp(min=1e-4)

    r_new = []
    for k in range(self.M):
        q_k = self.register_states[k] @ self.W_Q[k]
        scores = q_k @ K.T
        if training:
            gumbel_noise = -torch.log(-torch.log(
                torch.rand_like(scores).clamp(min=1e-8)
            ))
            scores = scores + gumbel_noise
        alpha_k = F.softmax(scores / tau, dim=-1)
        r_new.append((alpha_k.unsqueeze(-1) * V).sum(0))
    return torch.stack(r_new)
```

At inference time, the Gumbel noise can be omitted (deterministic mode) or retained with reduced amplitude for ensemble-style prediction.

### 16.3 Improvement 2: Per-Register Key Subspaces — Independent Interaction Channels

**QFT motivation.** The $1/N_h$ suppression result from Ageev and Ageev (2026) shows that each attention head contributes a **finite, independent** non-Gaussian correction to $G_c^{(4)}$. For Fock registers to serve as independent interaction channels (analogous to independent attention heads), they must attend in **different key spaces**. Currently, all $M$ registers share the same key projection $W_K \in \mathbb{R}^{d \times d_k}$ and value projection $W_V \in \mathbb{R}^{d \times d}$. Only the query projections $W_Q^{(k)}$ are per-register.

Shared keys mean all registers compute scores in the same subspace. Even with per-register queries, the attention patterns are highly correlated because the key representation is identical. From the QFT perspective, this is equivalent to $M$ interaction channels sharing the **same coupling constant** at the vertex: the effective number of independent non-Gaussian corrections is much less than $M$.

**The prescription.** Introduce per-register key projections by factoring the key space into register-specific subspaces:

$$k_j^{(k)} = W_K^{(k)} h_j, \qquad \alpha_{kj} = \mathrm{softmax}_j\left(\frac{q_k \cdot k_j^{(k)} + g_j}{\tau}\right)$$

Each register $k$ has its own $W_K^{(k)} \in \mathbb{R}^{d \times d_k}$, ensuring it attends in a different subspace of the token representations. This is structurally identical to multi-head attention's per-head Q/K projections.

**Parameter cost.** At $M = 16$, $d = 256$, $d_k = 64$: the per-register keys add $M \cdot d \cdot d_k = 16 \times 256 \times 64 \approx 262$K parameters, comparable to the existing per-register $W_Q$ cost. For value projections, the choice depends on whether per-register values are needed. The QFT analysis says the **key** subspace is what must differ (it determines the coupling), while the value projection (the content transferred) can remain shared. This parallels multi-head attention, where the key and query projections are per-head but the value projection is often shared via concatenation.

**Implementation:**

```python
# Per-register key projections
self.W_K = nn.Parameter(
    torch.randn(self.M, cfg.d, cfg.d_k) * (1.0 / cfg.d**0.5)
)

# In create_registers:
for k in range(self.M):
    q_k = self.register_states[k] @ self.W_Q[k]
    K_k = h_tokens @ self.W_K[k]       # per-register keys
    scores = q_k @ K_k.T
    ...
```

### 16.4 Improvement 3: Orthogonal Query Initialization — Maximally Spread Probes

**QFT motivation.** The independence-breaking term $\mathrm{Cov}_{W^Q, W^K}(X_{ab}, X_{cd})$ is nonzero precisely because different output coordinates are coupled through the shared attention map. For Fock registers to produce maximally diverse interaction modes, their query projections should probe **orthogonal semantic subspaces** from the first forward pass.

Currently, $W_Q \in \mathbb{R}^{M \times d \times d_k}$ is initialized with isotropic Gaussian noise at std $= 0.02$. All $M$ query directions are nearly aligned at initialization (all near zero with random small perturbations). During early training, all registers probe approximately the same direction, compounding the entropy collapse: even with a learnable temperature, the registers all attend to the same tokens.

**The prescription.** Initialize the register query projections to be **mutually orthogonal** across the register dimension. For each input dimension $i$, draw a random orthogonal matrix and assign its columns to the $M$ registers:

```python
Q_init = torch.zeros(M, d, d_k)
for i in range(d):
    U, _, _ = torch.linalg.svd(torch.randn(d_k, max(M, d_k)))
    Q_init[:, i, :] = U[:M, :d_k] * init_scale
self.W_Q = nn.Parameter(Q_init)
```

This ensures that from the very first forward pass, different registers attend to different aspects of the token representations. The orthogonality constraint is only at initialization; the projections are free to rotate during training.

**Field-theoretic justification.** In QFT, the mode expansion of a free field $\hat{\phi}(x) = \sum_k (a_k e^{ik \cdot x} + a_k^\dagger e^{-ik \cdot x})$ uses orthogonal plane-wave modes $e^{ik \cdot x}$ to ensure that each creation operator $a_k^\dagger$ contributes an independent degree of freedom. Orthogonal query initialization is the discrete analogue: each register "mode" probes an independent direction in semantic space.

### 16.5 Improvement 4: Coupled Creation-Destruction via Canonical Commutation Structure

**QFT motivation.** In quantum field theory, the creation operator $a^\dagger$ and annihilation operator $a$ are algebraically conjugate: $[a, a^\dagger] = 1$. The creation and annihilation amplitudes are not independent modules --- they are mathematical adjoints of each other. The destruction amplitude is determined by the creation amplitude.

Currently, FockPARFLM's creation gate (Q/K/V attention readout) and destruction gate (independent MLP sigmoid) are structurally decoupled. The destruction gate is an MLP conditioned on the register's own state $r_k$:

$$g_{\mathrm{destroy}}^{(k)} = \sigma(\mathrm{MLP}(r_k))$$

This has no structural relationship to the creation attention pattern $\alpha_{kj}$. The QFT canonical structure suggests they should be coupled: the destruction signal should be derived from the **same attention pattern** that created the register.

**The prescription.** Replace the MLP destruction gate with an **attention-derived destruction signal**:

$$g_{\mathrm{destroy}}^{(k)} = 1 - \max_j \alpha_{kj}$$

The logic:

- If the register's creation attention is peaked ($\max_j \alpha_{kj} \approx 1$), the register carries focused, token-specific content and should **survive** ($g_{\mathrm{destroy}} \approx 0$)
- If the creation attention is diffuse ($\max_j \alpha_{kj} \approx 1/T$), the register carries only mean-pool information (free-field content with no interaction) and should be **destroyed** ($g_{\mathrm{destroy}} \approx 1$)

The salience update with the coupled destruction becomes:

$$\sigma_k \leftarrow \sigma_k \cdot \lambda + \max_j(\alpha_{kj}) \cdot (1 - \lambda)$$

$$\sigma_k \leftarrow \sigma_k \cdot \max_j(\alpha_{kj})$$

where the first equation is the creation update (same as Section 9.1) and the second replaces the independent MLP destruction. Registers with consistently low attention peaks will see their salience decay geometrically toward zero, automatically deactivating them.

**Benefits over the MLP destruction gate:**

| Property | MLP destruction gate | Attention-derived destruction |
|---|---|---|
| Parameters | $d \cdot (d/4) + (d/4) \cdot 1 = d^2/4 + d/4$ per layer | Zero additional parameters |
| Coupling to creation | None (independent MLP) | Canonical: $g_{\mathrm{destroy}} = 1 - g_{\mathrm{create}}$ |
| QFT structure | $a$ and $a^\dagger$ independent | $a$ and $a^\dagger$ conjugate ($[a, a^\dagger] = 1$) |
| Interpretability | Opaque (learned MLP) | Transparent: destroy iff attention is diffuse |

This is also consistent with beim Graben et al.'s (2022) Fock-space framework for CFGs, where annihilation operators are the algebraic adjoints of creation operators --- a phrase is "destroyed" (decomposed) by the adjoint of the operator that "created" (composed) it.

### 16.6 Combined Architecture: QFT-Informed FockPARFLM v2.1

The four improvements compose naturally with the learnable temperature of Section 14.9. The updated creation gate becomes:

$$q_k = W_Q^{(k)} r_k, \qquad k_j^{(k)} = W_K^{(k)} h_j, \qquad v_j = W_V h_j$$

$$s_{kj} = q_k \cdot k_j^{(k)} + g_j, \qquad g_j \sim \mathrm{Gumbel}(0, 1)$$

$$\alpha_{kj} = \mathrm{softmax}_j\left(\frac{s_{kj}}{\tau}\right), \qquad \tau = \exp(\theta_\tau)$$

$$r_k \leftarrow \sum_j \alpha_{kj} \cdot v_j$$

$$\sigma_k \leftarrow \sigma_k \cdot \lambda + \max_j(\alpha_{kj}) \cdot (1 - \lambda)$$

$$\sigma_k \leftarrow \sigma_k \cdot \max_j(\alpha_{kj})$$

![QFT-Informed FockPARFLM v2.1 Creation Gate](images/fock_creation_gate_v21.png)

### 16.7 Summary: QFT-Motivated Design Principles

| Current design | QFT-motivated improvement | QFT justification |
|---|---|---|
| Deterministic softmax creation | Gumbel-Softmax (stochastic) | Virtual particle creation is probabilistic; vacuum fluctuations diversify register content |
| Shared $W_K$ across registers | Per-register $W_K^{(k)}$ subspaces | Independent interaction channels maximize $G_c^{(4)}$; shared keys correlate registers |
| Isotropic Gaussian $W_Q$ init | Orthogonal initialization | Maximally spread query probes; analogous to orthogonal plane-wave modes in field expansion |
| Independent MLP destruction | Attention-derived destruction | Canonical commutation: $a$ and $a^\dagger$ are algebraic conjugates, not independent modules |

These four changes are compatible with the learnable temperature fix (Section 14.9) and can be tested incrementally in the D6-D10 experimental series:

| Arm | Changes vs D6 | Tests |
|---|---|---|
| D6 | Learnable $\tau$ only | Baseline for temperature fix |
| D7 | D6 + Gumbel-Softmax creation | Stochastic creation effect |
| D8 | D7 + per-register $W_K^{(k)}$ | Independent interaction channels |
| D9 | D8 + orthogonal $W_Q$ init | Cold-start diversity |
| D10 | D9 + canonical destruction | Full QFT-informed gate |

---

## 17. QFT v2.1 Experiment Results and Current Bottleneck Analysis

### 17.1 Experiment Design

The QFT-motivated improvements of Section 16 were tested in a controlled 9-arm experiment (`fockparf_v2_qft_improvements.ipynb`), each arm using the P10g-equivalent architecture (d=256, L=8, M=16, v\_hidden=2048) on TinyStories with a short training budget (1M tokens, 2000 steps, seed 0) to isolate the effect of each QFT improvement before committing to full-scale runs.

| Cell | Description | Gumbel | Per-reg keys | Ortho init | Canon. destroy | tau | M |
|---|---|---|---|---|---|---|---|
| Q0 | FockPARF v2 baseline | - | - | - | - | fixed | 16 |
| Q1 | + Gumbel-Softmax creation | yes | - | - | - | fixed | 16 |
| Q2 | + per-register key subspaces | - | yes | - | - | fixed | 16 |
| Q3 | + orthogonal W\_Q init | - | - | yes | - | fixed | 16 |
| Q4 | + canonical destruction | - | - | - | yes | fixed | 16 |
| Q5 | Full QFT v2.1 (all four) | yes | yes | yes | yes | fixed | 16 |
| Q6 | Q5 + learnable tau (init 1.0) | yes | yes | yes | yes | learnable | 16 |
| Q7 | Q5 + M=32 registers | yes | yes | yes | yes | fixed | 32 |
| Q8 | PARFLM baseline (no registers) | - | - | - | - | - | 0 |

### 17.2 Results Summary

| Cell | Best PPL | Final PPL | vs Q0 | vs Q8 (PARFLM) | Entropy | Diversity |
|---|---|---|---|---|---|---|
| Q0 | 58.84 | 59.16 | -- | +4.3% | 0.134 | 0.145 |
| Q1 | 58.83 | 59.11 | -0.0% | +4.3% | 0.132 | 0.149 |
| Q2 | 59.38 | 59.60 | +0.9% | +5.2% | 0.141 | 0.347 |
| Q3 | 56.35 | 56.81 | -4.2% | -0.1% | 0.135 | 0.300 |
| Q4 | 55.89 | 56.60 | -5.0% | -1.0% | 0.134 | 0.245 |
| Q5 | 58.48 | 58.82 | -0.6% | +3.6% | 0.147 | 0.644 |
| **Q6** | **53.47** | **53.67** | **-9.1%** | **-5.2%** | **0.304** | **0.785** |
| Q7 | 122.24 | 122.80 | +107.7% | +116.6% | 0.155 | 0.489 |
| Q8 | 56.43 | 56.79 | -4.1% | -- | -- | -- |

### 17.3 Key Findings

**Finding 1: Naive Fock augmentation hurts.** Q0 (FockPARF v2 baseline, PPL 58.84) is *worse* than plain PARFLM with no registers (Q8, PPL 56.43). The register mechanism in its default configuration adds parameters without adding useful information routing. The QFT improvements are not optional refinements --- they are what makes the register mechanism beneficial.

**Finding 2: Q3 and Q4 are the individual winners.** Orthogonal query initialization (Q3, -4.2% vs Q0) and canonical destruction (Q4, -5.0% vs Q0) each independently bring FockPARFLM to parity with PARFLM. Q3 works by ensuring registers probe diverse semantic subspaces from the first forward pass. Q4 works by coupling creation and destruction --- registers carrying focused content persist; registers carrying diffuse (free-field) content are annihilated.

**Finding 3: Q5 interference resolved by learnable tau.** Combining all four improvements without learnable temperature (Q5, PPL 58.48) is worse than Q3 or Q4 alone. The four mechanisms interfere when the temperature is frozen: Gumbel noise and per-register keys increase score variance, but the fixed $1/\sqrt{d_k}$ scaling cannot compensate. Adding a learnable temperature (Q6, PPL 53.47) resolves this completely --- it is the single best arm, 9.1% below the FockPARF baseline and 5.2% below PARFLM.

**Finding 4: Q6 achieves genuine register specialization.** Q6's register diversity of 0.785 (vs Q0's 0.145) confirms that registers in Q6 specialize on different interaction channels. Its mean normalized entropy of 0.304 (vs Q0's 0.134) shows the registers are attending selectively but not collapsed --- intermediate between uniform attention (1.0) and one-hot selection (0.0). Per-layer analysis shows diversity growing smoothly from 0.985 at layer 1 to 0.864 at layer 7, indicating that later layers progressively sharpen register assignments.

**Finding 5: M=32 registers diverge.** Q7 (M=32, PPL 122.24) fails catastrophically. Diversity collapses at layers 5--7 (from 1.0 to 0.07), suggesting the model cannot coordinate 32 registers without additional stabilization (e.g., a temperature annealing schedule or per-register temperature parameters). This is consistent with the NN-QFT $1/N_h$ suppression result: each additional register contributes a smaller non-Gaussian correction, and at 32 registers the optimization landscape becomes too complex for the fixed temperature to navigate.

**Finding 6: Canonical destruction fix.** The original Q4 implementation caused catastrophic register collapse (all registers died on the first forward pass) because multiplying salience by raw $\alpha_{\max}$ (which is approximately $1/T \approx 0.004$ at initialization) produced $0.004^L \approx 10^{-18}$ after $L$ layers. The fix uses a log-normalized peakedness signal:

$$\text{survival} = \frac{\log(\alpha_{\max} \cdot T)}{\log(T)} \in [0, 1]$$

$$g_{\text{destroy}} = 0.1 + 0.4 \cdot (1 - \text{survival}) \in [0.1, 0.5]$$

This produces meaningful selectivity: uniform attention yields $g_{\text{destroy}} \approx 0.5$ (rapid annihilation), while peaked attention yields $g_{\text{destroy}} \approx 0.1$ (long persistence). The MLP destruction gates are fully replaced (not stacked on top), and a forward hook captures $\alpha_{\max}$ during the existing creation gate call without any additional forward passes.

### 17.4 Contextualizing Against the PARFLM Ceiling

The P10 ladder established the PARFLM architectural ceiling on TinyStories:

| Run | Architecture | Steps | Tokens | Best PPL |
|---|---|---|---|---|
| P10g | PARFLM (v\_hidden=2048, 22M params) | 16,000 | 5M | 26.42 |
| P10h | PARFLM (same) | 16,000 | 20M | 26.43 |
| S2 (v2) | FockPARF v1 (d=256, L=8, M=32) | 16,000 | 5M+ | 27.85 |

The Q6 result (53.47 PPL at 2000 steps / 1M tokens) is a small-budget proof-of-concept. At the same short budget, PARFLM (Q8) achieves 56.43 --- so Q6 already holds a 5.2% advantage. Power-law extrapolation of Q6's learning curve (scaling exponent $\alpha \approx 0.33$) suggests PPL $\sim$24--26 at 8000 steps with 5M tokens, which would be at or slightly below the P10g/h PARFLM ceiling. Whether Q6 can definitively break through the 26.4 PPL wall remains to be tested.

### 17.5 Current Bottleneck: The Temperature-Register Coordination Problem

The Q6 and Q7 results together reveal the current binding bottleneck. The Q6 recipe (all four QFT improvements + learnable temperature + M=16) is the clear winner, but:

1. **The global learnable temperature is a single scalar.** All 16 registers share the same temperature $\tau$. This means the model must find a single compromise temperature that works across registers with different semantic roles. At M=16 this compromise is feasible; at M=32 (Q7) it fails completely.

2. **The M=32 diversity collapse at deep layers** (layers 5--7) indicates the model needs additional structure to coordinate large register pools. In multi-head attention, each head has its own Q/K projections and effective temperature. The Fock register pool currently has per-register Q but shared temperature --- an intermediate level of independence that works at M=16 but not at M=32.

3. **The 26.4 PPL PARFLM ceiling is architectural, not data-limited** (confirmed by P10h). To break through it, the Fock mechanism must provide information routing that the conservative PARF potential cannot. The Q6 result shows the mechanism is on the right track (5.2% better than PARFLM at matched budget), but the margin is modest.

### 17.6 Next Steps

Based on the QFT v2.1 results, the following improvements are the highest-priority candidates for breaking through the 26.4 PPL ceiling:

1. **Per-register temperature.** Replace the single global $\tau = \exp(\theta_\tau)$ with per-register temperatures $\tau_k = \exp(\theta_{\tau,k})$ for $k = 1, \ldots, M$. This adds $M$ scalar parameters (16 at current scale) and allows each register to discover its own optimal attention sharpness. Hypothesis: this is necessary for M=32 to work and may further improve M=16.

2. **Temperature annealing schedule.** Initialize at a warm temperature ($\tau_0 = 1.0$) for exploration, then anneal toward a lower temperature ($\tau_{\min} = 0.1$--$0.3$) over training. This is analogous to simulated annealing and provides a more principled initialization trajectory than a fixed learnable scalar.

3. **Full-scale Q6 validation.** Run Q6 at the P10g budget (8000 steps, 5M tokens) to determine whether it breaks the 26.4 PPL ceiling. This is the critical experiment: if Q6 at full budget achieves PPL $\lt$ 26.4, it confirms that the QFT-motivated Fock mechanism provides expressivity beyond the conservative PARFLM ceiling.

4. **Per-register temperature + M=32.** If per-register temperature fixes the Q7 divergence, scaling to M=32 registers with the full QFT stack could unlock further gains. The Q7 diagnostic (diversity collapse at deep layers) is the signature of the problem; per-register temperature is the predicted fix.

5. **Deeper V\_theta.** The P10 ladder showed that V\_theta width is the dominant parameter lever for PARFLM. Whether the same holds when Fock registers provide non-conservative routing remains to be tested. Hypothesis: the Fock mechanism reduces the burden on V\_theta by providing direct information routing, so the V\_theta ceiling may be less binding.

### 17.7 Experiment Log

| Experiment | Notebook | Results directory | Date |
|---|---|---|---|
| QFT v2.1 (Q0--Q8) | `scripts/fockparf_v2_qft_improvements.ipynb` | `results/fock_v2_qft/` | May 2026 |

---

## References

- **Gueorguiev, D.** (2026). *Semantic Simulation: A Prescriptive Lagrangian Framework for Efficient Semantic Inference* (v4). arXiv / SSRN.
  — §9.2: Theorem v0-ceiling  
  — §9.4.2: v2 → Fock space and second quantisation  
  — §9.4.3: v3 → Lie groups / gauge theory  
  — §9.5: LCFRS reduction (composite reaches MCS)  
  — §15.6: Architectural obstructions to conservativity in attention  
  — §17.8: Fock-space augmentation: latent register pool  
  — §17.13: FockPARF improvement sweep  

- **Doi, M.** (1976). Second quantisation representation for classical many-particle system. *Journal of Physics A*, 9(9), 1465–1477.

- **Peliti, L.** (1985). Path integral approach to birth-death processes on a lattice. *Journal de Physique*, 46(9), 1469–1483.

- **Vaswani, A., Shazeer, N., Parmar, N., et al.** (2017). Attention is all you need. *NeurIPS 30*.

- **Jang, E., Gu, S., Poole, B.** (2017). Categorical reparameterization with Gumbel-Softmax. *ICLR 2017*.

- **Maddison, C. J., Mnih, A., Teh, Y. W.** (2017). The concrete distribution: A continuous relaxation of discrete random variables. *ICLR 2017*.

- **Radford, A., Kim, J. W., Hallacy, C., et al.** (2021). Learning transferable visual models from natural language supervision. *ICML 2021*. (CLIP — learnable temperature in contrastive loss.)

- **He, K., Fan, H., Wu, Y., Xie, S., Girshick, R.** (2020). Momentum contrast for unsupervised visual representation learning. *CVPR 2020*. (MoCo — temperature-scaled softmax.)

- **Zhai, S., Talbott, S., Srivastava, N., et al.** (2021). An attention free transformer. *arXiv:2105.14103*. (Per-head learnable temperature.)

- **Ageev, D. S., Ageeva, Y. A.** (2026). Neural Network Quantum Field Theory from Transformer Architectures. *arXiv:2602.10209*.

- **beim Graben, P., Huber, M., Meyer, W., Römer, R., Wolff, M.** (2022). Vector Symbolic Architectures for Context-Free Grammars. *Cognitive Computation*, 14, 733–748 (arXiv:2003.05171).

- `docs/Augmenting_PARFLM_to_handle_MCS_Languages.md`: FockPARFLM Phase 1 results and Phase 2 experimental plan  
- `docs/PARF-SPLM_Path_Forward_and_Experiments.md`: P10 ladder context  
- `notebooks/conservative_arch/parf/results/fockparf_improvement/`: Full P1–P5 sweep results  

---

*Report compiled: May 2026. Semantic Simulation Research Programme.*
