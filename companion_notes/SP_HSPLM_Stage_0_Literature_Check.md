# Solenoidal-Pair Hybrid SPLM (SP-HSPLM / Maxwell-PARFLM) — Stage 0 Literature Check

**Status:** Stage 0 working note for the proposed **Solenoidal-Pair Hybrid SPLM** (also called **Maxwell-PARFLM**) — an attention-free hybrid that realizes the non-conservative half of the Helmholtz decomposition by **causal pair-interaction skew/solenoidal fields** rather than by transformer attention blocks.

**Position:** companion to [`Scalar_Potential_based_Helmholtz_Architecture.md`](./Scalar_Potential_based_Helmholtz_Architecture.md) and [`On_Gumbel_softmax_sparsity_applied_to_V_phi.md`](./On_Gumbel_softmax_sparsity_applied_to_V_phi.md). This document is the literature scan that precedes the proposal write-up. Once Stage 0 is complete, Stage 1 (the leak-fixed per-token Class B/C/D rerun) and Stage 2 (the pair-skew force implementation) follow.

**Audience:** internal — collaborators and reviewers assessing whether the SP-HSPLM proposal occupies a genuinely novel point in the literature.

---

## 1. The proposal in one paragraph

PARFLM realizes routing through a **conservative pair-interaction scalar** $V_\phi(h_t, h_s)$, contributing $-\nabla_{h_t} \sum_s V_\phi$ to the force. This sits inside the autonomous conservative class $\mathcal{F}_S$ — enriched with pair structure but still curl-free in $h_t$. The v3 paper finds (E1-E5, §15.5) that **per-token** non-conservative additions (constant skew, position-dependent gauge) cannot close the SPLM-vs-attention gap on TinyStories. The conjecture explored here is that the missing ingredient is a **causal pair-interaction skew/solenoidal field**:

$$
F^{\text{rot}}_t = \sum_{s \lt t} \alpha_\phi(h_t, h_s) \cdot J_\phi(h_t - h_s)\, (\dot h_s - \dot h_t),
$$

with $J_\phi$ a learned **skew matrix function** ($J_\phi = J_+ - J_+^\top$, divergence-free by construction) and $\alpha_\phi$ a learned scalar pair affinity (the Gumbel-softmax routing already built into SparsePARFLM). Combined with PARFLM's conservative pair force, this gives a **pair-interaction Helmholtz model with no attention**:

$$
f_t = \underbrace{-\nabla_{h_t}[V_\theta + \sum_s V_\phi]}_{\text{conservative + routing}} + \underbrace{F^{\text{rot}}_t}_{\text{solenoidal + routing}} + \underbrace{\Omega(h_t)\dot h_t}_{\text{per-token gyro}} - \gamma\, \dot h_t .
$$

The Stage 0 question is: **has anyone built this object in any closely-adjacent setting, and what does the existing literature establish about the building blocks?**

---

## 2. The literature, organized by component

### 2.1 Conservative dynamics in neural networks

This is the territory PARFLM and SPLM already occupy. The literature is mature.

| Work | Year | Construction | Relevance |
|---|---|---|---|
| **HNN** (Greydanus, Dzamba, Yosinski) | 2019 | $\dot z = (\nabla S^\top - I)\nabla H_\theta(z)$, learn scalar $H_\theta$ | The canonical conservative NN; SPLM's $V_\theta$ is the symmetry-broken cousin |
| **LNN** (Cranmer, Greydanus et al.) | 2020 | Learn scalar Lagrangian $L_\theta(q,\dot q)$, derive EOM via Euler-Lagrange | Doesn't require canonical coords; SPLM-style framework |
| **SympNets** (Jin et al.) | 2020 | Symplectic feed-forward: gradient + linear-symplectic alternation | Architectural template for symplectic depth-stacks |
| **Symplectic Generative Networks (SGNs)** | 2025 | Volume-preserving generative model with Hamiltonian flow | Modern HNN extension; not a sequence model |
| **GeoHNNs** (Geometric HNNs) | 2025 | Constrained autoencoders enforcing Riemannian + symplectic geometry | Geometric prior beyond plain $H_\theta$ |
| **Hamiltonian Language Models** (Bijalwan, Medium) | 2025 | Token embeddings as phase space coordinates, semantic momentum across layers | Position match (LM + Hamiltonian) but informal blog post; not peer-reviewed; **no Helmholtz split, no pair structure**, no comparison against attention |

**Takeaway:** Conservative-only is well-charted. SPLM is a clean instance. None of these works closes the routing gap.

---

### 2.2 Conservative + dissipative split — the closest cousins

This is where SP-HSPLM lives architecturally. The most directly relevant literature.

#### 2.2.1 D-HNN — Dissipative Hamiltonian Neural Networks

**Sosanya & Greydanus, ICLR 2022.** [arXiv:2201.10085](https://arxiv.org/abs/2201.10085).

The single closest published cousin to SP-HSPLM. They parameterize **two scalar functions**:

- $H_\theta(q, p)$: the Hamiltonian (conservative, "rotational")
- $D_\theta(q, p)$: the Rayleigh dissipation function (irrotational, "friction")

and derive dynamics as

$$
\dot z = (\nabla S^\top - I) \nabla H_\theta(z) - \nabla D_\theta(z) ,
$$

i.e. **a symplectic gradient plus a regular gradient** of two distinct potentials. The authors explicitly call this an **"implicit Helmholtz decomposition"**: the $H$ branch generates the curl-bearing rotational dynamics; the $-\nabla D$ branch contributes the curl-free dissipative dynamics.

**Differences from SP-HSPLM:**
- D-HNN is for **low-dimensional scalar dynamical systems** (mass-spring with friction, ocean currents). No sequence/language application.
- Both branches are scalar potentials; **no pair-interaction structure**.
- Dissipation is purely irrotational ($-\nabla D$); **no genuine solenoidal piece** ($F_{\text{sol}}$ with $\nabla \times F \ne 0$).
- The "rotational" piece is the symplectic gradient of $H$, not a skew matrix-valued field.

**Takeaway:** D-HNN establishes that **architecturally splitting conservative and non-conservative dynamics is a published, accepted construction**. SP-HSPLM extends this in three new directions: (i) sequence models, (ii) pair-interaction routing, (iii) genuine solenoidal terms (via skew matrix parameterizations) rather than only irrotational dissipation.

#### 2.2.2 Port-Hamiltonian Neural Networks (pHNNs)

**Desai et al., Phys. Rev. E 2021** ([10.1103/PhysRevE.104.034312](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.104.034312)); **Stable PHNNs** ([arXiv:2502.02480](https://arxiv.org/abs/2502.02480)).

The **port-Hamiltonian formalism** writes dynamics as

$$
\dot z = [J(z) - R(z)] \nabla H(z) + B(z)\, u(t),
$$

where $J = -J^\top$ is a **skew matrix** (the conservative routing structure), $R \succeq 0$ is the **dissipation matrix**, and $u(t)$ is an **external forcing port**. Stable pHNNs add Lyapunov stability guarantees.

**Differences from SP-HSPLM:**
- $J$ is typically a **fixed canonical symplectic block** $\begin{pmatrix}0 & I \\ -I & 0\end{pmatrix}$, not a **learned function of $h$**. SP-HSPLM's $J_\phi(h_t - h_s)$ is fundamentally richer.
- No pair-interaction structure across many tokens; pHNN is a single-particle / single-system framework.
- Not used for sequence models.

**Takeaway:** Port-Hamiltonian work establishes the **skew-matrix-plus-dissipation parameterization as standard practice**. SP-HSPLM lifts this to pair interactions.

#### 2.2.3 GENERIC / GFINN / metriplectic networks

**Lee et al., GFINNs ([arXiv:2109.00092](https://arxiv.org/abs/2109.00092)); Hernández et al., port-metriplectic ([arXiv:2211.01873](https://arxiv.org/abs/2211.01873)); Gruber et al., neural metriplectic ([arXiv:2405.16305](https://arxiv.org/abs/2405.16305)); N-GINNs (2026, [arXiv:2605.09058](https://arxiv.org/html/2605.09058v1)).**

The **GENERIC formalism** (Öttinger 2005, *Beyond Equilibrium Thermodynamics*) prescribes

$$
\dot z = L(z)\, \nabla E(z) + M(z)\, \nabla S(z),
$$

with $L$ skew (Poisson bracket) and $M$ symmetric PSD (dissipative bracket), satisfying degeneracy conditions $L \nabla S = 0$ and $M \nabla E = 0$. GFINNs and metriplectic NNs enforce these conditions architecturally, guaranteeing energy conservation **and** entropy non-decrease by construction.

**Differences from SP-HSPLM:**
- Metriplectic NNs are for **physical thermodynamic systems** (gas heat exchange, thermoelastic dynamics, Langevin); not sequence models.
- Single-particle (or low-dimensional) state, not pair interactions.
- The skew matrix $L(z)$ is learned, but as a **per-state** object, not as a **causal pair-affinity-weighted sum** over context.

**Takeaway:** GENERIC establishes that **two-bracket structure (skew + symmetric) is a principled, peer-reviewed construction**. SP-HSPLM is essentially a **GENERIC-style architecture for language models with causal pair interactions** — a position not present in this literature.

---

### 2.3 Magnetic / gyroscopic / gauge forces in neural networks

The "magnetic-like" branch of SP-HSPLM (the $J_\phi$ skew force) has direct precedents.

| Work | Year | What is parameterized | Relation to SP-HSPLM |
|---|---|---|---|
| **Constrained HNNs (CHNN)** (Finzi et al., NeurIPS 2020) | 2020 | Cartesian coordinates with explicit Lagrange multipliers, includes magnetic field handling | Establishes magnetic-Hamiltonian NNs; per-particle, not pair |
| **Symplectic gyroceptrons** | 2022 | Structure-preserving NNs for nearly-periodic symplectic maps with adiabatic invariants | Gyroscopic systems, fixed dimensionality |
| **Hamilton-Dirac NNs (HDNNs)** ([arXiv:2401.15485](https://arxiv.org/abs/2401.15485)) | 2024 | Constrained Hamiltonian systems with magnetic fields, guiding-center motion | Strong magnetic-field NN result; not for sequence models |
| **Gauge Flow Models** ([arXiv:2507.13414](https://arxiv.org/html/2507.13414v3)) | 2025 | Neural ODE with learnable Lie-algebra-valued gauge field $A_\mu$ | Closest theoretical match for $J_\phi$; but no pair structure, not sequential |

**Takeaway:** Magnetic / gyroscopic / gauge forces in NNs are an **active, but per-particle** research area. SP-HSPLM extends this to **pair-interaction causal magnetic fields**, which is novel.

---

### 2.4 Pair-interaction physics-informed neural networks

The pair-interaction half of SP-HSPLM has natural cousins in molecular ML.

| Work | Year | Construction | Closeness |
|---|---|---|---|
| **MACE** ([arXiv:2206.07697](https://ar5iv.labs.arxiv.org/html/2206.07697)) | 2022 | Higher-order equivariant message passing, up to 4-body | Pair + many-body, but for static molecules not causal sequences |
| **PAINN** ([Schütt et al., ICML 2021](https://proceedings.mlr.press/v139/schutt21a/schutt21a.pdf)) | 2021 | Polarizable atom interaction network with vector + scalar features | Equivariant pair forces; force-as-gradient-of-energy (conservative) |
| **ENINet** | 2023 | Equivariant N-body interactions with explicit $\ell=1$ vector messages | Closest to skew pair-vector messaging |
| **Physics-informed GNN with momentum conservation** ([Nature Comm. 2025](https://www.nature.com/articles/s41467-025-67802-5)) | 2025 | Edge-local reference frames; pairwise impulses | Conservative; momentum-preserving |
| **Thermodynamics-informed GNNs (GENERIC structure)** ([arXiv:2203.01874](https://arxiv.gg/abs/2203.01874)) | 2022 | GENERIC bracket structure on graph | Conservative + dissipative on graph; not causal sequences |

**Takeaway:** Equivariant pair-interaction networks for molecules are the **closest direct analogue** for the SP-HSPLM pair force. None operate causally on sequences. MACE/PAINN/ENINet are also **conservative** (force is gradient of an energy); SP-HSPLM's pair-skew force is genuinely non-conservative.

---

### 2.5 Helmholtz decomposition in neural networks

Limited but growing.

| Work | Year | Decomposition | Relation |
|---|---|---|---|
| **HDNet** ([arXiv:2406.08570](https://arxiv.org/abs/2406.08570)) | 2024 | Decomposes flow fields into divergence-only and curl-only components | Closest "Helmholtz-named" NN; flow estimation, not LMs |
| **Project-and-Generate** ([arXiv:2603.24500](https://arxiv.org/html/2603.24500v1)) | 2026 | Differentiable Leray projection onto divergence-free subspace for incompressible flows | Hard divergence-free constraints |
| **Neural Conservation Laws** (NeurIPS 2022) | 2022 | Divergence-free vector fields via differential forms / autograd | Establishes the "guarantee divergence-free by construction" technique used in SP-HSPLM's $J_\phi$ |
| **D-HNN** (above) | 2022 | Implicit Helmholtz decomp via $H$ + $D$ | Already discussed; the closest LM-adjacent cousin |

**Takeaway:** Explicit Helmholtz decomposition has been used in **flow estimation and PINN-style PDE solvers**, but **not in sequence/language models**. The Q9d Helmholtz SPLM is already novel in this respect; SP-HSPLM extends it further by replacing the attention half with a structurally Helmholtz-faithful pair force.

---

### 2.6 Attention-free sequence models

For completeness — these are the alternatives one might compare against, but they are **not** vector-field-theoretic.

| Work | Year | Mechanism | Vector-field-theoretic? |
|---|---|---|---|
| **Mamba** (Gu & Dao, [arXiv:2312.00752](https://arxiv.org/pdf/2312.00752)) | 2023 | Selective state-space model, input-dependent SSM | No — selection is over an SSM kernel |
| **RWKV** ([arXiv:2305.13048](https://arxiv.org/pdf/2305.13048)) | 2023 | Linear attention reformulated as RNN | No |
| **RWKV-7 "Goose"** ([arXiv:2503.14456](https://arxiv.org/abs/2503.14456v2)) | 2025 | Generalized delta rule with vector-valued gating, state tracking | No |
| **TLinFormer** | 2025 | Strict $O(N)$ attention via reconfigured connections | No |

**Takeaway:** Attention-free LMs exist and are competitive (Mamba matches Transformers $\sim 2\times$ its size at small scale), but **none of them is a vector-field-theoretic construction** with explicit conservative + non-conservative components. SP-HSPLM occupies a genuinely different design point.

---

### 2.7 Physics interpretations of transformers

These works **interpret** transformer dynamics through physical lenses without changing the architecture.

| Work | Year | Physics frame | What it says |
|---|---|---|---|
| **Mean-field interacting particles** (AMS Bull. 2025) | 2025 | Tokens as particles in measure-flow PDE | Long-time behavior is clustering; explains next-token prediction concentration |
| **Continuous PDE perspective** ([arXiv:2408.09523](https://arxiv.org/html/2408.09523)) | 2024 | Self-attention as non-local operator + FFN as local reaction | Discrete depth as time discretization |
| **Non-Hermitian operator theory** | 2026 | Self-attention as non-Hermitian many-body operator | Emergent stability, representational saturation |
| **Thermodynamic Isomorphism / Lagrangian Attention** ([arXiv:2602.08216](https://arxiv.org/html/2602.08216v1)) | 2026 | Softmax as Helmholtz free-energy minimum, $QK$ as electrodynamic coupling | Explains scaling laws and grokking as phase transitions |
| **Physical Transformer** ([arXiv:2601.02433](https://arxiv.org/abs/2601.02433)) | 2026 | Attention heads as interacting Hamiltonian spins; Neural Differential Manifold | Symplectic layer that preserves invariants |
| **Momentum Attention** ([arXiv:2602.04902](https://www.arxiv.org/pdf/2602.13690)) | 2026 | Symplectic augmentation with momentum operators | Single-layer induction enabled |

**Takeaway:** A growing 2025-2026 literature **interprets** attention through physics, but does not **replace** it with a vector-field architecture. Several use Hamiltonian primitives but **on top of attention**, not as alternatives. The closest in spirit is Physical Transformer, but it keeps the attention head as the underlying operator.

---

### 2.8 Skew/antisymmetric ResNets and stable Neural ODEs

| Work | Year | Construction | Relevance |
|---|---|---|---|
| **Haber & Ruthotto, "Stable architectures for deep neural networks"** ([arXiv:1705.03341](https://arxiv.org/pdf/1705.03341)) | 2017 | Forward Euler on $\dot h = \sigma(K h + b)$ with stability via antisymmetric $K$ | Establishes the antisymmetric-weight-tying technique |
| **Reversible ResNets** (Chang et al., AAAI 2018) | 2018 | Memory-efficient reversible architectures | Borrows from stability ideas |

**Takeaway:** Antisymmetric weight tying for stability is a foundational technique; SP-HSPLM's $J_\phi = J_+ - J_+^\top$ is the same idea applied to a pair-interaction kernel.

---

### 2.9 The "Maxwell" name

For naming: a few NNs are called "Maxwell" or operate on Maxwell's equations, but none in the LM/sequence space.

- **MaxwellNet** — solves Maxwell's PDE (electromagnetic field given material distribution).
- **JefiAtten** — attention-based solver for Maxwell with charge/current sources.
- **Fourier-Helmholtz-Maxwell neural operator** — gauge-free electrodynamics PDE solver.

These **solve** Maxwell's equations as a PDE problem; they don't **architect** an NN that mimics Maxwell-like dynamics. Calling our model "Maxwell-PARFLM" is therefore unambiguous in the LM space, but the name is somewhat loaded; the more descriptive "**Solenoidal-Pair Hybrid SPLM (SP-HSPLM)**" may be safer for a paper title.

---

## 3. Originality assessment

The SP-HSPLM proposal sits in a gap that the surveyed literature does **not** occupy:

| Property | Existing work | SP-HSPLM |
|---|---|---|
| Conservative + dissipative split | D-HNN, port-Hamiltonian, GFINN | ✓ |
| Explicit Helmholtz decomposition in NN | HDNet, Neural Conservation Laws, D-HNN | ✓ |
| Block-type architectural component split | Helmholtz SPLM (Q9d) — **already in this codebase** | ✓ |
| Skew-matrix non-conservative force | Port-Hamiltonian, Constrained HNN, Gauge Flow | ✓ |
| Pair-interaction physics-informed forces | MACE, PAINN, ENINet | ✓ (in PARFLM already) |
| Causal pair interactions in sequence model | PARFLM — **already in this codebase** | ✓ |
| **Pair-interaction skew/solenoidal field** in causal sequence model | **No matching prior work** | **First** |
| **Helmholtz-decomposed sequence LM with no attention block** | Helmholtz SPLM has attention — **gap** | **First** |

Two distinct claims of novelty:

1. **Algorithmic:** A pair-interaction skew force $\sum_{s \lt t} \alpha_\phi(h_t, h_s) J_\phi(h_t - h_s)(\dot h_s - \dot h_t)$ with $J_\phi = J_+ - J_+^\top$. The skew-matrix-of-pair-difference construction does not appear in the molecular ML literature (which uses scalar-energy gradients) nor in the pHNN/GENERIC literature (per-particle, not pair).

2. **Architectural:** Helmholtz decomposition assigned to architectural blocks where the **non-conservative half is also a vector-field primitive**, not attention. The existing Helmholtz SPLM (Q9d) leaves the non-conservative half to attention; the existing PARFLM keeps everything conservative. SP-HSPLM is the first in this family to be both **block-Helmholtz** and **attention-free**.

---

## 4. Theoretical risks the literature has flagged

The literature surfaces three independent risks worth pre-registering before Stage 2.

### 4.1 Pure per-token Helmholtz components are routing-poor

The v3 paper's E1-E5 result already establishes this internally. The **external** literature corroborates it: every metriplectic / port-Hamiltonian / pHNN result we found scales tested **per-particle systems with $\le 100$ DoF**, not text. None has a **routing** mechanism. This is consistent with the v3 conclusion that the bottleneck is routing capacity. The SP-HSPLM bet is that **pair-interaction skew forces** add real routing.

### 4.2 Skew-matrix learnability

Port-Hamiltonian NNs (Desai et al., Stable PHNN) report that the skew matrix $J(z)$ is **harder to fit than the symmetric Hamiltonian gradient $\nabla H$**. The standard mitigation is to keep $J$ low-rank or constant. For SP-HSPLM, the analogue is to use **low-rank $J_\phi$**: parameterize $J_+ = U V^\top$ with $U, V \in \mathbb{R}^{d \times r}$, $r \ll d$, and project to skew. This keeps the parameter count modest and the gradients well-conditioned.

### 4.3 Velocity coupling and stability

The skew velocity term $\Omega(h)\dot h$ is energy-conserving (zero work) but can create **chaotic, long-time oscillatory dynamics**. The pHNN stability literature (Stable Port-Hamiltonian, [arXiv:2502.02480](https://arxiv.org/abs/2502.02480)) prescribes a **dissipation lower bound** as a sufficient condition for Lyapunov stability:

$$
R(z) \succeq c\, I, \quad c > 0 .
$$

In SP-HSPLM, the SPLM damping $\gamma$ already provides this; the practical requirement is to **cap** $\gamma$ from below (e.g., $\gamma \ge 0.05$) when the skew force is enabled.

---

## 5. Adjacent results to cite if SP-HSPLM works

If Stage 2 lands a quality result, the related work / discussion section of the paper should cite:

**Direct cousins (split conservative + non-conservative)**

- D-HNN (Sosanya & Greydanus, ICLR 2022) — the closest published cousin; should be cited as the published precedent for the architectural split.
- Port-Hamiltonian NNs (Desai et al., 2021; Stable PHNN, 2025) — for the $J - R$ structure.
- GFINNs (Lee et al., 2021) and metriplectic NNs (Gruber et al., 2024) — for the GENERIC formalism.

**Pair-interaction equivariant forces**

- PAINN (Schütt et al., ICML 2021), MACE (2022), ENINet (2024) — for the molecular ML precedent.
- The Nature Comms 2025 momentum-conserving GNN — for a recent peer-reviewed pair-physics example.

**Magnetic / gauge / gyroscopic NNs**

- Constrained HNN (Finzi et al., NeurIPS 2020), Hamilton-Dirac NNs (2024), Gauge Flow Models (2025) — for the magnetic-like force lineage.

**Helmholtz decomposition in NNs**

- HDNet (2024), Project-and-Generate (2026), Neural Conservation Laws (NeurIPS 2022) — for the divergence-free-by-construction technique.

**Physics-interpreted transformers (for context, not direct comparison)**

- Physical Transformer (2026), Thermodynamic Isomorphism (2026), Momentum Attention (2026) — these **interpret** attention; SP-HSPLM **replaces** it.

**Attention-free sequence models (for the falsifier baseline)**

- Mamba (Gu & Dao, 2023), RWKV-7 (2025) — non-physics-inspired attention-free LMs as the strong empirical baseline.

---

## 6. Stage 0 verdict

**Pass.** The proposed Solenoidal-Pair Hybrid SPLM occupies a genuinely novel point in the literature:

- The architectural pattern (block-type Helmholtz split) has a published cousin (D-HNN) but has never been instantiated for sequence models.
- The mechanism (causal pair-interaction skew force) has component cousins (port-Hamiltonian skew structure, equivariant molecular pair forces) but no direct prior art combining them.
- The framing (attention-free Helmholtz LM with vector-field-theoretic routing) is unique.

The literature also flags three concrete risks (routing capacity of pure per-particle non-conservative terms, skew-matrix learnability, and velocity-coupling stability), each with established mitigations.

**Recommended next step:** Stage 1 — rerun the v3 E1-E5 per-token Class B/C/D experiments on the **leak-fixed v3 codebase**. The original E1-E5 results were reported under the leak; a clean rerun is cheap and either (a) reproduces the negative result and motivates SP-HSPLM, or (b) shows that some per-token non-conservative term has been quietly competitive all along. Either way, Stage 1 is the cleanest entry point to Stage 2.

---

## 7. References

### External literature (chronological)

- Robbins, H. (1952). *Some aspects of the sequential design of experiments*. Bulletin of the AMS — bandit framing of the routing problem.
- Öttinger, H. C. (2005). *Beyond Equilibrium Thermodynamics*. Wiley — the GENERIC formalism textbook.
- Haber, E. and Ruthotto, L. (2017). *Stable architectures for deep neural networks*. [arXiv:1705.03341](https://arxiv.org/pdf/1705.03341).
- Greydanus, S., Dzamba, M., and Yosinski, J. (2019). *Hamiltonian Neural Networks*. NeurIPS. [arXiv:1906.01563](https://arxiv.org/pdf/1906.01563).
- Cranmer, M., Greydanus, S., et al. (2020). *Lagrangian Neural Networks*. ICLR Workshop. [arXiv:2003.04630](https://arxiv.org/pdf/2003.04630).
- Finzi, M., Wang, K., and Wilson, A. G. (2020). *Simplifying Hamiltonian and Lagrangian Neural Networks via Explicit Constraints*. NeurIPS. [proceedings](https://proceedings.neurips.cc/paper/2020/file/9f655cc8884fda7ad6d8a6fb15cc001e-Paper.pdf).
- Desai, S. A., Mattheakis, M., Sondak, D., Protopapas, P., and Roberts, S. J. (2021). *Port-Hamiltonian Neural Networks for Learning Explicit Time-Dependent Dynamical Systems*. Phys. Rev. E. [link](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.104.034312).
- Lee, K., Trask, N. A., and Stinis, P. (2021). *GFINNs: GENERIC Formalism Informed Neural Networks for Deterministic and Stochastic Dynamical Systems*. [arXiv:2109.00092](https://arxiv.org/abs/2109.00092).
- Sosanya, A. and Greydanus, S. (2022). *Dissipative Hamiltonian Neural Networks: Learning Dissipative and Conservative Dynamics Separately*. ICLR. [arXiv:2201.10085](https://arxiv.org/abs/2201.10085).
- Schütt, K. T., Unke, O. T., and Gastegger, M. (2021). *Equivariant message passing for the prediction of tensorial properties and molecular spectra* (PAINN). ICML. [proceedings](https://proceedings.mlr.press/v139/schutt21a/schutt21a.pdf).
- Batatia, I., Kovács, D., et al. (2022). *MACE: Higher Order Equivariant Message Passing Neural Networks*. NeurIPS. [arXiv:2206.07697](https://ar5iv.labs.arxiv.org/html/2206.07697).
- Hernández, Q. et al. (2022). *Port-metriplectic neural networks: thermodynamics-informed machine learning*. Computational Mechanics. [arXiv:2211.01873](https://arxiv.org/abs/2211.01873).
- Gu, A. and Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. [arXiv:2312.00752](https://arxiv.org/pdf/2312.00752).
- Hernández, Q. et al. (2022). *Thermodynamics-informed graph neural networks*. [arXiv:2203.01874](https://arxiv.gg/abs/2203.01874).
- Gruber, A. et al. (2024). *Efficiently Parameterized Neural Metriplectic Systems*. [arXiv:2405.16305](https://arxiv.org/html/2405.16305v3).
- Hamilton-Dirac NNs (2024). [arXiv:2401.15485](https://arxiv.org/abs/2401.15485).
- HDNet (2024). [arXiv:2406.08570](https://arxiv.org/abs/2406.08570).
- Stable Port-Hamiltonian NNs (2025). [arXiv:2502.02480](https://arxiv.org/abs/2502.02480).
- Symplectic Generative Networks (2025). [arXiv:2505.22527](https://arxiv.org/abs/2505.22527).
- Geometric HNNs (2025). [arXiv:2507.15678](https://arxiv.org/html/2507.15678v1).
- Gauge Flow Models (2025). [arXiv:2507.13414](https://arxiv.org/html/2507.13414v3).
- RWKV-7 "Goose" (2025). [arXiv:2503.14456](https://arxiv.org/abs/2503.14456v2).
- Physics-informed momentum-conserving GNN (Nature Comm. 2025). [link](https://www.nature.com/articles/s41467-025-67802-5).
- Physical Transformer (2026). [arXiv:2601.02433](https://arxiv.org/abs/2601.02433).
- Thermodynamic Isomorphism of Transformers (2026). [arXiv:2602.08216](https://arxiv.org/html/2602.08216v1).
- Momentum Attention (2026). [arXiv:2602.04902](https://www.arxiv.org/pdf/2602.13690).
- N-GINNs (2026). [arXiv:2605.09058](https://arxiv.org/html/2605.09058v1).
- Project-and-Generate (2026). [arXiv:2603.24500](https://arxiv.org/html/2603.24500v1).

### Internal documents

- [`Scalar_Potential_based_Helmholtz_Architecture.md`](./Scalar_Potential_based_Helmholtz_Architecture.md) — Q9d Helmholtz hybrid (current; uses attention for the non-conservative half).
- [`PARF_Augmented_SPLM_Architecture_v2.md`](./PARF_Augmented_SPLM_Architecture_v2.md) — PARFLM design doc (conservative pair scalar $V_\phi$).
- [`On_Gumbel_softmax_sparsity_applied_to_V_phi.md`](./On_Gumbel_softmax_sparsity_applied_to_V_phi.md) — score-head + Gumbel routing (the $\alpha_\phi$ machinery SP-HSPLM reuses).
- [`Gumbel_sparsity_method.md`](./Gumbel_sparsity_method.md) — pedagogical explainer.
- Paper v4 / v5 §15.5 (Class B-D autonomous menu), §15.6 (E1-E5 ablations), Appendix A (Eq. A.130, the non-autonomous conservative class) — the v3 negative results SP-HSPLM is responding to.

---

*Last updated: 15 May 2026. Stage 0 complete; ready to draft the SP-HSPLM design doc and pre-registered Stage 1 protocol.*
