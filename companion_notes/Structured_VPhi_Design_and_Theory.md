# Structured $V_\phi$: Design, Theory, and Analysis

**Status:** companion note to *Semantic Simulation: A Prescriptive Lagrangian Framework for Efficient
Semantic Inference* (Gueorguiev, 2026). Consolidates the structured pair-potential material that was
previously scattered across the PARF design docs, the MLP-ablation note, the Fock-mechanism levers,
and the training-instabilities note.
**Scope:** the theory, the implemented variants, the internal sub-networks, the stability levers, and
the continuous-learning properties of the **pairwise** scalar potential $V_\phi$ in the PARFLM /
Fock-PARFLM family — the analogue of
[`Structured_VTheta_Design_and_Theory.md`](./Structured_VTheta_Design_and_Theory.md) for the
*two-body* channel.
**Companion docs:**

- [`PARF_Augmented_SPLM_Architecture_v2.md`](./PARF_Augmented_SPLM_Architecture_v2.md) — the §5.1
  design doc (the prescriptive structural form) and the P1–P7 lever ladder (§10).
- [`On_the_MLP_Layer_modeling_pairwise_potential.md`](./On_the_MLP_Layer_modeling_pairwise_potential.md)
  — the unstructured MLP ablation and the OQ-1 framing.
- [`Improving_the_Fock_Mechanism_to_match_Attention.md`](./Improving_the_Fock_Mechanism_to_match_Attention.md)
  — the competitive (softmax-normalised) Lever 3 variant.
- [`On_Training_the_PARF_Force.md`](./parf/On_Training_the_PARF_Force.md) — the second-order autograd
  training pipeline (Algorithm A).
- [`PARF_Stage_1_5b_design.md`](./PARF_Stage_1_5b_design.md) — the gathered / sparse evaluation path.
- [`On_Gumbel_softmax_sparsity_applied_to_V_phi.md`](./On_Gumbel_softmax_sparsity_applied_to_V_phi.md)
  — top-$k$ Gumbel routing over pairs.
- [`Training_Instabilities_in_Fock-PARFLM_with_structured_V_theta.md`](./Training_Instabilities_in_Fock-PARFLM_with_structured_V_theta.md)
  — the $1/r$ gradient-explosion failure mode and the stability levers.
- [`Continuous_Learning_in_Semantic_Simulation-based_models_vs_with_Transformer_models.md`](./Continuous_Learning_in_Semantic_Simulation-based_models_vs_with_Transformer_models.md)
  — §4.5, the spawnability / continuous-learning properties.
- **Implementation:** [`notebooks/conservative_arch/parf/model_parf.py`](../notebooks/conservative_arch/parf/model_parf.py)
  (`StructuralVPhi`, `StructuralCompetitiveVPhi`, `MultiHeadVPhi`, `ValueTransportVPhi`,
  `CompositeVPhi`, `MLPVPhi`).

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [The pair potential in PARF dynamics](#2-the-pair-potential-in-parf-dynamics)
3. [The §5.1-faithful structural form](#3-the-51-faithful-structural-form)
4. [Anatomy: the four channels](#4-anatomy-the-four-channels)
5. [The implemented variants](#5-the-implemented-variants)
6. [Structured does not mean MLP-free: the internal sub-networks](#6-structured-does-not-mean-mlp-free-the-internal-sub-networks)
7. [Why structured: inductive biases and the force-is-gradient argument](#7-why-structured-inductive-biases-and-the-force-is-gradient-argument)
8. [Conservativity](#8-conservativity)
9. [Sparse / gathered evaluation](#9-sparse--gathered-evaluation)
10. [Stability levers](#10-stability-levers)
11. [Parameter counts and cost](#11-parameter-counts-and-cost)
12. [Empirical results and OQ-1](#12-empirical-results-and-oq-1)
13. [Continuous-learning and spawnability properties](#13-continuous-learning-and-spawnability-properties)
14. [Recommendations](#14-recommendations)
15. [Interaction with hybrid structured V_theta](#15-interaction-with-hybrid-structured-v_theta)
16. [References](#16-references)

---

## 1. Motivation

The SPLM family evolves hidden states under a conservative force $f = -\nabla_h U$ derived from a
single shared scalar $U$. In the pure SPLM, $U = V_\theta(\xi, h)$ is a **one-body** potential — it
encodes where, in hidden-state space, trajectories want to settle. The PARF arm adds a **two-body**
term,

$$
U^{(\ell)}_t = V_\theta(\xi^{(\ell)}_t, h^{(\ell)}_t) + \sum_{s \lt t} V_\phi(h^{(\ell)}_t, h^{(\ell)}_s),
$$

so the model also carries **relational** knowledge: how token $t$ interacts with an earlier token
$s$. $V_\phi$ is the architectural knob of the PARF arm — the single surface on which all of the
model's routing capacity is bought.

The same three pressures that motivated structured $V_\theta$ apply, with one extra:

1. **The force is the gradient.** The model never uses $V_\phi$ directly; it uses
   $-\nabla_{h_t} V_\phi$. Errors in a scalar are amplified in its gradient, so the *form* of $V_\phi$
   controls force quality and training stability disproportionately (Section 7).
2. **Cost.** A dense unstructured pair MLP materialises an $O(T^2 \cdot d)$ feature tensor; the
   structured form avoids it via squared-norm expansion and top-$k$ gathering (Sections 9, 11).
3. **Interpretability.** A structured $V_\phi$ exposes a type-gate, a value-aligner sign, and a
   distance kernel as separate, inspectable channels; an MLP is a black box.
4. **Continuous learning.** Only a *structured* $V_\phi$ is additively extensible with bounded
   interference — the relational analogue of well spawning (Section 13). This property has no MLP
   counterpart and is the strongest reason to keep $V_\phi$ structured.

---

## 2. The pair potential in PARF dynamics

### 2.1 The causal reduction

Each PARF layer computes a per-token energy and takes its hidden-state gradient as the force:

$$
f^{(\ell)}_t = -\nabla_{h_t} U^{(\ell)}_t = -\nabla_{h_t} V_\theta - \sum_{s \lt t} \nabla_{h_t} V_\phi(h_t, h_s).
$$

The design-doc §3 **causal reduction** treats past tokens as fixed external sources: in code
`h_src = h.detach()`, so $\nabla_{h_t}$ flows only through the *query* slot, never the *source* slot.
This is what makes the pair force strictly causal (and is verified by `causal_probe_parf.py` for every
variant below).

### 2.2 The Verlet step

The force enters the same damped velocity-Verlet update as $V_\theta$ (see
[`Structured_VTheta_Design_and_Theory.md`](./Structured_VTheta_Design_and_Theory.md) §2 and the
`_layer_step` methods of `model_parf_multixi.py` / `model_parf_sparse.py`); $V_\phi$ changes only the
force, not the integrator. Because the force is a literal gradient of a literal scalar, the dynamics
stay conservative regardless of which $V_\phi$ variant is plugged in (Section 8).

---

## 3. The §5.1-faithful structural form

Section 5.1 of the paper develops PARF as the generalisation of a central $1/r$ law to a space where
both pairwise **direction** and pairwise **type** enter. Lifting it to hidden-state level with small
learned projections $l = W_l h$ (the low-rank **type** projection, dimension `v_phi_d_type`) and
$\theta = W_\theta h$ (the low-rank **angle** projection, dimension $K$ = `v_phi_d_angle`), the
prescriptive form is

$$
V_\phi^{\text{struct}}(h_t, h_s) = - C \cdot \Theta_\phi\big(\theta(h_t), \theta(h_s)\big) \cdot \Phi_\phi\big(l(h_t), l(h_s)\big) \cdot \frac{1}{\sqrt{\lVert h_t - h_s \rVert^2 + \varepsilon^2}}.
$$

The induced pair force has the framework's central-force shape by construction:

$$
-\nabla_{h_t} V_\phi \propto \frac{\Theta_\phi \, \Phi_\phi}{\lVert h_t - h_s \rVert^2} \, \hat r_{ts} + \text{(corrections from } \nabla_{h_t}\Theta_\phi, \, \nabla_{h_t}\Phi_\phi),
$$

with $\hat r_{ts}$ the unit vector from source to query. The leading term is gravity-like; the
corrections from the learned matchers are the trainable degrees of freedom.

---

## 4. Anatomy: the four channels

Four independent multiplicative channels carry distinct semantic content. (Table cells use plain
Unicode per the rendering conventions.)

| Channel | Symbol | Codomain | What it controls |
|---|---|---|---|
| Strength | C | positive scalar | Global scale of the pair force vs. V_theta. A fixed config float (`v_phi_C`). |
| Value-aligner | Θ_φ | [−1, 1] | The **sign** of the interaction (attractive vs. repulsive — the "AR" in PARF). |
| Type-gate | Φ_φ | [0, 1] | **Which pairs interact**: a Gaussian gate exp(−c‖l_t − l_s‖²) on the type projection. |
| Distance kernel | 1/r | positive | The spatial falloff: Plummer-softened 1/sqrt(‖h_t − h_s‖² + ε²). |

The §5.1 Theorem 16 makes the type-gate a *locality* statement: pairs of close type-relatedness
(‖l_t − l_s‖ < δ) carry essentially all of the interaction; the rest contributes force exponentially
suppressed in type-distance and "can be dropped into the dissipation budget" — i.e. it is bounded
above by the energy the damping term removes each step, so it is dynamically negligible. This locality
is exactly the property that, lifted to a mixture, makes the relational channel **spawnable**
(Section 13).

---

## 5. The implemented variants

All variants share one I/O contract, so they are drop-in interchangeable via `v_phi_kind`:

- `forward(h, h_src) -> (B, T, T)` — dense.
- `forward_gathered(h, h_src_g) -> (B, T, k)` — sparse / top-$k$ gathered.

### 5.1 `StructuralVPhi` — the base §5.1 form

The direct implementation of Section 3, with the Gaussian type-gate Φ_φ unnormalised across sources.
Uses the squared-norm expansion ‖a−b‖² = ‖a‖² + ‖b‖² − 2⟨a,b⟩ to avoid the (B, T, T, d) difference
tensor, cutting intermediate memory by roughly d/2.

### 5.2 `StructuralCompetitiveVPhi` — Lever 3 (the variant in production)

This is the form the current Fock-PARFLM OpenWebText run uses (`v_phi_kind='structural_competitive'`).
It replaces the unnormalised Gaussian gate with a **causal row-softmax** across past sources:

$$
\tilde\Phi_\phi(l_t, l_s) = \mathrm{scale}(t) \cdot \mathrm{softmax}_{s \lt t}\!\Big( -c \, \lVert l_t - l_s \rVert^2 / \tau \Big).
$$

Motivation (from the dense-cell diagnostic): the unnormalised gate **saturates near 1** in
high-dimensional type space — it provides no per-pair selectivity — and the dense sum over O(T²) pairs
**interferes destructively**, washing out the directional force. Lever 3 imports softmax attention's
competitive, zero-sum selectivity (Σ_s w_ts = 1) into the structural form while preserving (i) the
AR sign decomposition through Θ_φ ∈ [−1, 1] and (ii) the gravity-like 1/r kernel. The result is a
"PARF-attention hybrid" that is still a single shared scalar U = V_θ + Σ V_φ under one Verlet step.

### 5.3 `MultiHeadVPhi` — sum of independent structural heads

$$
V_\phi(h_t, h_s) = \sum_{m=1}^{H} V^{(m)}_\phi(h_t, h_s),
$$

each head an independent instance of the configured kind, with its own W_l^(m), W_θ^(m), and
sub-networks. A sum of scalar potentials is a scalar potential, so the induced force stays
conservative (the per-pair Jacobian remains symmetric — Arm 1 of the conservativity diagnostic
passes). The heads share the *single* top-$k$ routing of the parent layer, so interaction cost stays
O(T·k); only the per-pair scalar evaluation widens by H. The production run uses
`v_phi_n_heads = 4`. `n_heads = 1` reproduces the single-head form exactly. **The heads are summed,
not softmax-mixed** — this is the additive (unnormalised) regime in the spawnability taxonomy of
Section 13.

### 5.4 `ValueTransportVPhi` and `CompositeVPhi` — directional content routing

`StructuralVPhi` produces a strictly *radial* force (along ±(h_t − h_s)). `ValueTransportVPhi` adds a
bilinear, distance-gated term

$$
V_\phi(h_t, h_s) = -\sum_{m} g(r_{ts}) \, (U_m h_t) \cdot (W_m h_s), \qquad g(r) = \exp(-r^2 / 2\sigma^2),
$$

whose gradient writes the W-transformed *source content* into the query along a learned (non-radial)
direction — the attention-like value transport the radial kernel cannot express. `CompositeVPhi` sums
heterogeneous components (e.g. a `MultiHeadVPhi` plus a `ValueTransportVPhi`); being a sum of scalars,
it too is conservative. These are optional and off in the headline run.

### 5.5 `MLPVPhi` — the unstructured ablation (the null hypothesis)

$$
V_\phi^{\text{mlp}}(h_t, h_s) = \mathrm{MLP}\big( [h_t, h_s, h_t - h_s] \big),
$$

a 3-layer GELU MLP on the concatenated pair. It collapses all four channels into one entangled
network. It is the OQ-1 comparator (Section 12), **not** a production component: it has no inductive
biases, is unbounded in magnitude, is asymmetric in (h_t, h_s) by construction, materialises a
(B, T, T, 3d) tensor that grad-checkpointing cannot intercept, and — decisively for continuous
learning — is **not spawnable** (Section 13).

---

## 6. Structured does not mean MLP-free: the internal sub-networks

A common misreading is that "structured $V_\phi$" contains no MLP. It does — but as small, bounded,
low-dimensional sub-networks *inside* the factored form, not as a monolithic pair MLP. Each structural
head contains two:

- **`phi_c_net`** — `Linear(1 → v_phi_phi_hidden) → GELU → Linear(v_phi_phi_hidden → 1)`. A 2-layer
  MLP on the **scalar** squared type-distance, producing the positive per-pair inverse bandwidth c
  via softplus. Per-pair adaptive bandwidth, not a high-dimensional map.
- **The Θ_φ value-aligner** (`theta_form='mlp'`, the default) — a small MLP on the **K-dimensional**
  angle projection (split-linear blocks W_q, W_s, W_d → GELU → Linear → 1), bounded by tanh (or
  softsign, Patch C). The alternative `theta_form='bilinear'` (Patch D) replaces this MLP entirely
  with θ_tᵀ W θ_s, removing the sub-MLP at the cost of expressivity.

Two facts make these sub-MLPs compatible with everything structured buys:

1. **They are low-dimensional and bounded.** They act on R¹ and R^K (K = 8), not R^{3d}; Θ_φ is
   tanh-bounded into [−1, 1]. They cannot produce the unbounded, high-curvature forces an
   unstructured R^{3d} MLP can.
2. **They are per-head, so they do not break spawnability.** Each head owns its own `phi_c_net` and
   Θ_φ; spawning a head leaves existing heads' sub-networks untouched (parameter non-interference,
   Section 13). The *addressability* and *locality* come from the structural Gaussian gate Φ_φ — the
   Θ_φ MLP only sets the sign/strength *within* an already-localised gate.

So "structured" is a statement about the factored form (1/r kernel, separable type/value channels,
bounded amplitude, per-head addressability), not about the literal absence of every MLP.

---

## 7. Why structured: inductive biases and the force-is-gradient argument

The single most important reason the form matters: **the training signal is the gradient of
$V_\phi$**, and gradients are sensitive to derivatives. A 1% wiggle in the scalar that the loss
tolerates can become a large error in the force at length scales below the typical pair distance. The
structural prior **locks in low-frequency structure** — it has a handful of knobs (the bandwidth c,
the angle weights, the strength C) that can wiggle, versus the MLP's tens of thousands.

| Structural prior | Mechanism (structural variant) | MLP equivalent |
|---|---|---|
| Separable selectivity channels | Θ_φ · Φ_φ / r factorisation | none — entangled in W_1 |
| Symmetric underlying form | Θ_φ tanh-MLP, Φ_φ Gaussian | none — must be learned |
| Translation-equivariant kinetic term | 1/sqrt(‖h_t − h_s‖² + ε²) | none — must be learned |
| Bounded pair force | Θ_φ ∈ [−1, 1] via tanh, Φ_φ ∈ [0, 1] | none — unbounded |
| Plummer-softened singularity | ε regulator | smooth by construction (good) |
| Low-frequency only | ~13k params/head | ~164k params/head (more capacity) |

The MLP wins on raw capacity (by universal approximation it can in principle express the structural
form); the structural variant wins on inductive bias, parameter economy, force smoothness, and
training stability. Which advantage dominates the actual perplexity is the OQ-1 question (Section 12).

---

## 8. Conservativity

A central commitment of the framework is that PARF dynamics are **conservative**: the per-token force
is the gradient of a scalar, giving the system a Lyapunov function and well-defined energy (this is
what distinguishes PARF from attention, whose per-token update is the gradient of no scalar). Every
variant in Section 5 produces a scalar $V_\phi$ from which the trainer takes the gradient via
`torch.autograd.grad`, so **all of them preserve conservativity by construction** — including the MLP.
There is no failure mode where a variant "accidentally" becomes non-conservative. What an unbounded
variant *can* do is produce high curvature in unintended places, yielding unphysically large forces;
that is a stability problem (Section 10), not a conservativity violation. The
`conservativity_diagnostic.py` arms (Jacobian symmetry, per-layer shared-potential R²) verify this
for the `baseline`, `multihead`, and `directional` configurations.

---

## 9. Sparse / gathered evaluation

The dense form is O(T²) in pairs. The production path is **sparse**: a single Gumbel-softmax top-$k$
routing (top_k = 8) selects, for each query, the k most relevant past sources, and `forward_gathered`
evaluates $V_\phi$ only on those, with intermediates of shape (B, T, k, ·) rather than (B, T, T, ·).
The gathered form is mathematically identical to `forward(h, h_src).gather(-1, idx)` but uses O(T·k)
memory instead of O(T²) (equivalence proof in
[`PARF_Stage_1_5b_design.md`](./PARF_Stage_1_5b_design.md)). This matters doubly for the MLP variant,
whose dense (B, T, T, 3d) feature tensor is the binding memory cost; gathering sidesteps it. The
top-$k$ truncation error is bounded by the §5.2 residual and, by construction, can be set below the
dissipation force — it is absorbed into the dissipation budget.

---

## 10. Stability levers

The $1/r$ kernel is the one genuinely dangerous element: when two hidden states cluster, $r \to 0$ and
the force can explode (documented in
[`Training_Instabilities_in_Fock-PARFLM_with_structured_V_theta.md`](./Training_Instabilities_in_Fock-PARFLM_with_structured_V_theta.md)).
The levers, in order of leverage:

| Lever | Config | Effect |
|---|---|---|
| Plummer softening | `v_phi_eps` (0.1 in production) | Floors r at ε, capping the 1/r force at C/ε. The primary fix for the step-43k explosion. |
| LN-before-distance (Patch A) | `ln_before_distance=True` | LayerNorms h before the distance, decoupling 1/r from absolute ‖h‖ growth across layers. |
| Per-module gradient clip | `GRAD_CLIP_VPHI` (0.3) | Pre-clips V_phi grads before the global clip, so the 1/r spike cannot dominate and over-shrink every other module. |
| Softsign value-aligner (Patch C) | `theta_activation='softsign'` | Polynomial (not exponential) approach to ±1, keeping Θ_φ gradients alive when it saturates deep in the stack. |
| Bilinear value-aligner (Patch D) | `theta_form='bilinear'` | Removes the Θ_φ sub-MLP; fewer params, recovers the canonical −sin(θ_t − θ_s) at K = 2. |
| Bounded amplitude | tanh Θ_φ, Gaussian Φ_φ | ‖V_phi^struct‖ ≤ C/ε by construction — the MLP has no such bound. |

---

## 11. Parameter counts and cost

Exact counts at the production scale (d = 384, 4 heads, `v_phi_phi_hidden = v_phi_theta_hidden = 128`,
`v_phi_d_type = 16`, `v_phi_d_angle = 8`), from instantiating the modules:

| Variant | Params (4 heads) | Per head | Bounded? | Inductive bias |
|---|---|---|---|---|
| structural_competitive | 51,720 | 12,930 | yes (≤ C/ε) | 1/r, separable, symmetric underlying |
| mlp | 656,900 | 164,225 | no | none |
| ratio | 12.7× | 12.7× | — | — |

In percentage of the full model both are small (the tied embedding dominates at ~19M), so parameter
economy is real but not decisive; the decisive deltas are stability, generalisation, and spawnability.
On wall-clock and memory the structural form is strictly cheaper: it exploits the squared-norm
expansion and (with gathering) avoids the MLP's (B, T, T, 3d) tensor that OOMs 16 GB MPS even with
grad-checkpointing.

---

## 12. Empirical results and OQ-1

**OQ-1 (the first-order question).** *Does PARF closure depend on the §5.1 structural prior?* If the
MLP variant matches structural on val PPL, the prior is pedagogical; if structural outperforms, the
prior is empirically active.

**Prototype-scale evidence (Stage 1, em-ln base, Tiny-Shakespeare-class shapes).** Table cells use
plain Unicode.

| Cell | V_phi regime | val PPL | Δ vs SPLM (no V_phi) |
|---|---|---|---|
| SPLM em-ln | none | 173.6 | — |
| P1 dense structural | −C·Θ·Φ/r, all s<t | 210.5 | +36.9 worse |
| P1.5a dense MLP (h=16) | unstructured MLP, all s<t | 297.2 | +123.6 worse |
| P1.6 dense structural (φ=θ=128) | wider structural | 207.6 | +34.0 worse |
| P5 sparse structural (top-k=4) | structural + Gumbel routing | 176.7 | +3.1 (within seed noise) |

Two readings, both important and both preliminary:

1. **The binding constraint at prototype scale was the dense aggregation regime, not the per-pair
   form.** Dense V_phi (structural or MLP) *hurt*; the top-$k$ sparse structural form (P5) closed the
   gap to within seed noise. This is why production uses sparse top-$k$ structural-competitive.
2. **The MLP did substantially worse than structural in the dense regime** (297.2 vs. 207.6), but the
   comparison is confounded: the MLP was capacity-starved at h = 16 (it OOMs at h = 64), so this is
   not a clean capacity-matched test. It is suggestive of an active structural prior, not conclusive.

**OpenWebText-scale OQ-1 is still open.** No head-to-head structural-vs-MLP PPL exists at the
production scale; the parf README lists it as TBD. Since `v_phi_kind` is a one-line flag and gathering
removes the MLP OOM, a matched-config run is cheap to launch and is the proper quantifier. A
prerequisite measurement — the relational fraction φ_rel = (L(V_φ=0) − L(full)) / L(full) — is
available now as a read-only diagnostic (`diagnostics/measure_relational_fraction.py`) and bounds the
maximum PPL swing any V_phi change can produce.

---

## 13. Continuous-learning and spawnability properties

This is the property with no MLP counterpart, developed in full in
[`Continuous_Learning_...md`](./Continuous_Learning_in_Semantic_Simulation-based_models_vs_with_Transformer_models.md)
§4.5 (Propositions 6–8). In brief:

The Gaussian type-gate Φ_φ is a localised, addressable unit in type-space — structurally identical to
a $V_\theta$ well. Generalising the single gate to a mixture of M addressable relational components
(the addressable-center version of `MultiHeadVPhi`) makes a *relation* the spawnable unit:

- **Proposition 6 (parameter non-interference).** Spawning relational component M+1 leaves every
  existing component's parameters untouched — exact, for both the additive (summed heads) and
  competitive (softmax) forms.
- **Proposition 7 (bounded responsibility interference).** Under softmax mixing, spawning rescales
  existing responsibilities by ρ = 1 − w_new, with |Δw_m| ≤ w_new ≤ exp(−c(d² − d²_near)) — relational
  interference decays exponentially in type-space distance. Validated to machine precision in
  `diagnostics/vphi_spawn_interference_demo.py` (max bound violation 1.4e-16; controlled locality
  slope −0.999 vs. target −1.0).
- **Proposition 8 (MLP non-spawnability).** The MLP admits no addressable decomposition; one gradient
  step perturbs ~99.5% of all pairs with no distance decay — the catastrophic-forgetting signature.

The current `MultiHeadVPhi` is the **additive** regime: spawning a head is interference-free in both
parameter and responsibility senses, at the cost of needing an amplitude budget Σ_m C_m ≤ C_max to
keep V_phi bounded (the same trade-off as the unnormalised V_theta variant). The
**mixture-structured spawnable** V_phi with addressable per-component type-centers, and a well
management policy over relations, is a proposed architecture (Experiment G6), not yet trained — the
production run establishes only that the additive structural form trains stably.

---

## 14. Recommendations

1. **Keep V_phi structured.** Even at PPL parity, only the structured form is cheaper (params,
   memory, wall-clock), more stable (bounded force, no high-curvature spikes), and — uniquely —
   compatible with relational continuous learning.
2. **Use sparse top-$k$ structural-competitive** (the production default). The prototype ladder shows
   the dense regime is the binding constraint; top-$k$ closes the gap.
3. **Keep the stability levers on**: `v_phi_eps ≥ 0.1`, `ln_before_distance=True`, and per-module
   V_phi clipping. They are what tamed the 1/r explosion.
4. **Run φ_rel before any OQ-1 experiment.** If the relational channel carries little predictive
   weight, the MLP-vs-structural PPL question is second-order and the decision rests entirely on the
   stability and continuous-learning axes.
5. **If running OQ-1 at OWT scale**, match parameters (small `v_phi_mlp_hidden`) and use gathered
   eval + top_k to sidestep the MLP OOM, so the comparison isolates the prior rather than capacity or
   memory.

## 15. Interaction with hybrid structured $V_\theta$

The structured V_phi's stability properties (§10) were validated in
conjunction with SQ3 quadratic wells and Gaussian wells for $V_\theta$.
A new **hybrid $V_\theta$** design — Gaussian wells with a quadratic
background ($\varepsilon \lVert h \rVert^2$) — is now under
evaluation (see
[`Structured_VTheta_Design_and_Theory.md`](./Structured_VTheta_Design_and_Theory.md) §13
and
[`colab_hybrid_gaussian_quad_vtheta.ipynb`](../notebooks/conservative_arch/scaleup/colab_hybrid_gaussian_quad_vtheta.ipynb)).

The hybrid $V_\theta$ interacts with the structured $V_\phi$ in two
relevant ways:

1. **Hidden-state scale stabilisation.** The quadratic background
   $\varepsilon \lVert h \rVert^2$ provides a global restoring force
   that prevents hidden-state drift far from the origin. This
   benefits the $V_\phi$ distance kernel directly: the Plummer-softened
   $1/r$ force (Lever 1, §10) relies on $r$ staying within a
   reasonable range.  If $V_\theta$ allows hidden states to escape to
   large $\lVert h \rVert$, pairwise distances grow and the $V_\phi$
   signal diminishes. The background quadratic ensures that $V_\phi$
   continues to operate in its designed scale regime even during early
   training when Gaussian well centres have not yet converged.

2. **Joint stability.** The SQ3 $V_\theta$ blowups documented in
   [`Training_Instabilities_in_Fock-PARFLM_with_structured_V_theta.md`](./Training_Instabilities_in_Fock-PARFLM_with_structured_V_theta.md) §2–§4
   were caused by unbounded $V_\theta$ forces — not by $V_\phi$. The
   structured $V_\phi$ was a **stabilising factor** (bounded force,
   Plummer softening) during those events. The hybrid $V_\theta$
   removes the source of the instability while preserving the structured
   $V_\phi$ pairing, so no $V_\phi$-side changes are required.

All hybrid $V_\theta$ evaluation cells (H1–H7, P1–P7) use
`structural_competitive` $V_\phi$ as the default, consistent with the
production configuration.

---

## 16. References

### Internal documents

- [`PARF_Augmented_SPLM_Architecture_v2.md`](./PARF_Augmented_SPLM_Architecture_v2.md) — §5.1
  structural form, §10 lever ladder (P1–P7), OQ-1, the prototype PPL table.
- [`On_the_MLP_Layer_modeling_pairwise_potential.md`](./On_the_MLP_Layer_modeling_pairwise_potential.md)
  — MLP ablation, inductive-bias table, memory analysis.
- [`Improving_the_Fock_Mechanism_to_match_Attention.md`](./Improving_the_Fock_Mechanism_to_match_Attention.md)
  — competitive (Lever 3) structural V_phi.
- [`On_Training_the_PARF_Force.md`](./parf/On_Training_the_PARF_Force.md) — Algorithm A, the
  second-order autograd graph, gradient checkpointing.
- [`PARF_Stage_1_5b_design.md`](./PARF_Stage_1_5b_design.md) — gathered evaluation equivalence and
  memory analysis.
- [`On_Gumbel_softmax_sparsity_applied_to_V_phi.md`](./On_Gumbel_softmax_sparsity_applied_to_V_phi.md),
  [`Gumbel_sparsity_method.md`](./Gumbel_sparsity_method.md) — top-$k$ Gumbel routing.
- [`Fock_PARFLM_Conservativity_Diagnostic.md`](./Fock_PARFLM_Conservativity_Diagnostic.md) —
  conservativity of multi-head V_phi.
- [`Training_Instabilities_in_Fock-PARFLM_with_structured_V_theta.md`](./Training_Instabilities_in_Fock-PARFLM_with_structured_V_theta.md)
  — the 1/r explosion and the stability levers.
- [`Continuous_Learning_in_Semantic_Simulation-based_models_vs_with_Transformer_models.md`](./Continuous_Learning_in_Semantic_Simulation-based_models_vs_with_Transformer_models.md)
  — §4.5 spawnability (Propositions 6–8), the G6 experiment.
- [`Structured_VTheta_Design_and_Theory.md`](./Structured_VTheta_Design_and_Theory.md) — §13: hybrid
  Gaussian + quadratic background structured $V_\theta$ design and evaluation.
- [`colab_hybrid_gaussian_quad_vtheta.ipynb`](../notebooks/conservative_arch/scaleup/colab_hybrid_gaussian_quad_vtheta.ipynb) — hybrid $V_\theta$ $\varepsilon$-sweep evaluation notebook.
- [`Structured_VTheta_Design_and_Theory.md`](./Structured_VTheta_Design_and_Theory.md) — the one-body
  analogue of this note.
- **Implementation:** [`notebooks/conservative_arch/parf/model_parf.py`](../notebooks/conservative_arch/parf/model_parf.py);
  diagnostics [`vphi_spawn_interference_demo.py`](../notebooks/conservative_arch/parf/diagnostics/vphi_spawn_interference_demo.py),
  [`measure_relational_fraction.py`](../notebooks/conservative_arch/parf/diagnostics/measure_relational_fraction.py),
  [`conservativity_diagnostic.py`](../notebooks/conservative_arch/parf/conservativity_diagnostic.py),
  [`causal_probe_parf.py`](../notebooks/conservative_arch/parf/causal_probe_parf.py).

### External literature

- Schütt, K. T. et al., **SchNet: A continuous-filter convolutional neural network for modeling
  quantum interactions**, NeurIPS 2017, arXiv:1706.08566 — pair-interaction networks with symmetry
  baked into the filters; closest external precedent for the structural variant.
- Battaglia, P. et al., **Relational inductive biases, deep learning, and graph networks**, 2018,
  arXiv:1806.01261 — inductive biases vs. unstructured MLPs in pair-interaction settings (OQ-1).
- Greydanus, S. et al., **Hamiltonian Neural Networks**, NeurIPS 2019, arXiv:1906.01563 — a learned
  scalar defines a vector field via its gradient; the same machinery as the PARF force.
- Cranmer, M. et al., **Lagrangian Neural Networks**, ICLR 2020 workshop, arXiv:2003.04630 — the
  pair-interaction generalisation.
- Hornik, K., **Approximation Capabilities of Multilayer Feedforward Networks**, Neural Networks 4(2),
  1991 — the universal-approximation basis for why an MLP V_phi can in principle express the
  structural form.

---

End of note. This is the canonical structured-$V_\phi$ reference; the one-body analogue is
`Structured_VTheta_Design_and_Theory.md`, and the continuous-learning properties are developed in
`Continuous_Learning_...md` §4.5.
