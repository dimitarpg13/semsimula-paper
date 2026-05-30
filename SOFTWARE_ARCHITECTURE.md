# Software Architecture

Mermaid-based class diagrams and sequence diagrams for the Semantic
Simulation companion codebase.  All models discussed in the paper (v4)
are included, together with their helper modules (potentials, gates,
probes, data pipeline) and training workflows.

> **Rendering**: GitHub, VS Code (with Mermaid extensions), and Cursor
> all render fenced `mermaid` blocks inline.  If your viewer does not
> support Mermaid, paste the blocks into
> [mermaid.live](https://mermaid.live).

---

## Table of Contents

1. [SPLM Family — Static Class Diagram](#1-splm-family--static-class-diagram)
2. [PARF Family — Static Class Diagram](#2-parf-family--static-class-diagram)
3. [V_θ Potential Hierarchy](#3-v_θ-potential-hierarchy)
4. [V_φ Pair-Potential Hierarchy](#4-v_φ-pair-potential-hierarchy)
5. [Fock Gate Components](#5-fock-gate-components)
6. [Context Channel (ξ) Modules](#6-context-channel-ξ-modules)
7. [Causal Probe Modules](#7-causal-probe-modules)
8. [Training Scripts & Notebooks](#8-training-scripts--notebooks)
9. [Data Pipeline](#9-data-pipeline)
10. [Attention Baseline](#10-attention-baseline)
11. [Non-Conservative Force Hierarchy](#11-non-conservative-force-hierarchy)
12. [Sequence Diagram — SPLM Training Loop](#12-sequence-diagram--splm-training-loop)
13. [Sequence Diagram — PARF/Multi-Xi Training Loop](#13-sequence-diagram--parfmulti-xi-training-loop)
14. [Sequence Diagram — Fock Multi-Xi Training Loop](#14-sequence-diagram--fock-multi-xi-training-loop)
15. [Sequence Diagram — Causal Probe (Pre-Training)](#15-sequence-diagram--causal-probe-pre-training)
16. [Sequence Diagram — Inference / Generation](#16-sequence-diagram--inference--generation)
17. [File Map](#17-file-map)

---

## 1. SPLM Family — Static Class Diagram

The vanilla Scalar-Potential Language Model (SPLM) and its successive
refinements: SARF mass, LayerNorm damping, multi-channel ξ (K-EMA,
S4D, HiPPO), Helmholtz decomposition, Hybrid SPLM, first-order
ablation, symplectic integrator variant, non-conservative extension,
and SPH-SPLM.

```mermaid
classDiagram
    direction TB

    class SPLMConfig {
        +int vocab_size
        +int d
        +int max_len
        +int L
        +int v_hidden
        +int v_depth
        +float dt
        +bool causal_force
    }

    class ScalarPotentialLM {
        +E : Embedding
        +P : Parameter
        +V_theta : ScalarPotential
        +_embed_and_pool(x) Tensor
        +forward(x, targets) logits, loss
        +generate(x, max_new_tokens) Tensor
        +num_params() int
    }

    class SPLMSARFMassConfig {
        +str mass_mode
        +bool ln_after_step
    }

    class ScalarPotentialLMSARFMass {
        +E : Embedding
        +P : Parameter
        +V_theta : ScalarPotential
        +m_global : Parameter
        +compute_mass(x, emb) Tensor
        +_embed(x) Tensor
        +_layer_step(h, h_prev, m, γ, dt) Tensor
        +_stack_forward(h0, x) h_L, traj
        +forward(x, targets) logits, loss
        +generate(x, max_new_tokens) Tensor
    }

    class SPLMSARFConfig {
        +bool causal_force
    }

    class ScalarPotentialLMSARF {
        +V_theta : ScalarPotential
        +_embed(x) Tensor
        +integrate(h0, x) h_L
        +forward(x, targets) logits, loss
    }

    class SPLMSARFMassLNConfig {
        <<extends SPLMSARFMassConfig>>
    }

    class ScalarPotentialLMSARFMassLN {
        +ln_layers : ModuleList~LayerNorm~
        +_project(h) Tensor
    }

    class SPLMSARFMassGMConfig {
        <<extends SPLMSARFMassConfig>>
    }
    class ScalarPotentialLMSARFMassGM {
        +V_theta : GaussianMixturePotential
    }

    class SPLMFirstOrderConfig {
        <<extends SPLMSARFMassLNConfig>>
    }
    class ScalarPotentialLMFirstOrder {
        +_layer_step(h, h_prev, m, γ, dt) Tensor
    }

    class SPLMSymplecticConfig {
        +float dt
    }
    class ScalarPotentialLMSymplectic {
        +_embed(x) Tensor
        +forward(x, targets) logits, loss
    }

    class SPLMSARFMassLNMultiXiConfig {
        +int xi_channels
        +List~float~ xi_alpha_inits
        +bool xi_learnable
    }
    class ScalarPotentialLMSARFMassLNMultiXi {
        +xi_module : MultiChannelXi
        +V_theta : ScalarPotentialMultiXi
        +_layer_step(h, h_prev, m, γ, dt) Tensor
    }

    class SPLMSARFMassLNMultiS4DConfig {
        +int xi_channels
    }
    class ScalarPotentialLMSARFMassLNMultiS4D {
        +xi_module : MultiChannelS4D
        +V_theta : ScalarPotentialMultiXi
    }

    class SPLMSARFMassLNMultiHiPPOConfig {
        +int xi_channels
        +int hippo_N
    }
    class ScalarPotentialLMSARFMassLNMultiHiPPO {
        +xi_module : MultiChannelHiPPO
        +V_theta : ScalarPotentialMultiXi
    }

    class HelmholtzConfig {
        +int n_heads
        +bool use_helmholtz
    }
    class HelmholtzLM {
        +V_theta : ScalarPotential
        +attn_layers : ModuleList
        +compute_mass(x) Tensor
        +_embed(x) Tensor
        +forward(x, targets) logits, loss
    }

    class HSPLMConfig {
        +int n_heads
        +float alpha_attn
    }
    class HybridSPLM {
        +V_theta : ScalarPotential
        +attn_layers : ModuleList
        +compute_mass(x) Tensor
        +_embed(x) Tensor
        +forward(x, targets) logits, loss
    }
    class _NullAttnHybrid {
        <<zero-weight attention ablation>>
    }

    class SPLMNonConservativeConfig {
        +str nc_type
        +int nc_rank
    }
    class ScalarPotentialLMNonConservative {
        +nc_force : NonConservativeForce
        +_layer_step(h, h_prev, m, γ, dt) Tensor
    }

    class SPHSPLMConfig {
        <<extends SparsePARFConfig>>
    }
    class ScalarPotentialLMSPHSPLM {
        +skew_kernel : SkewKernelLowRank
        +gyro_kernel : PerTokenGyroKernel
        +_layer_step(h, h_prev, m, γ, dt) Tensor
    }

    SPLMConfig <|-- SPLMSARFMassConfig
    SPLMSARFMassConfig <|-- SPLMSARFMassGMConfig
    SPLMSARFMassConfig <|-- SPLMSARFMassLNConfig
    SPLMSARFMassLNConfig <|-- SPLMFirstOrderConfig
    SPLMSARFMassLNConfig <|-- SPLMSARFMassLNMultiXiConfig
    SPLMSARFMassLNConfig <|-- SPLMSARFMassLNMultiS4DConfig
    SPLMSARFMassLNConfig <|-- SPLMSARFMassLNMultiHiPPOConfig
    SPLMSARFMassLNConfig <|-- SPLMNonConservativeConfig

    ScalarPotentialLM <|-- ScalarPotentialLMSARFMass
    ScalarPotentialLMSARFMass <|-- ScalarPotentialLMSARFMassGM
    ScalarPotentialLMSARFMass <|-- ScalarPotentialLMSARFMassLN
    ScalarPotentialLMSARFMassLN <|-- ScalarPotentialLMFirstOrder
    ScalarPotentialLMSARFMassLN <|-- ScalarPotentialLMSARFMassLNMultiXi
    ScalarPotentialLMSARFMassLN <|-- ScalarPotentialLMSARFMassLNMultiS4D
    ScalarPotentialLMSARFMassLN <|-- ScalarPotentialLMSARFMassLNMultiHiPPO
    ScalarPotentialLMSARFMassLN <|-- ScalarPotentialLMNonConservative

    HybridSPLM <|-- _NullAttnHybrid

    ScalarPotentialLMSARFMassLNMultiXi --> MultiChannelXi : xi_module
    ScalarPotentialLMSARFMassLNMultiS4D --> MultiChannelS4D : xi_module
    ScalarPotentialLMSARFMassLNMultiHiPPO --> MultiChannelHiPPO : xi_module
```

---

## 2. PARF Family — Static Class Diagram

The Pair-Augmented Repulsive Force (PARF) family builds on top of the
SPLM with explicit pair-interaction potentials V_φ.  This is the
primary experimental lineage of the paper.

```mermaid
classDiagram
    direction TB

    class PARFConfig {
        +int vocab_size
        +int d, L, max_len
        +int v_hidden, v_depth
        +str v_phi_kind
        +str mass_mode
        +bool causal_force
        +bool ln_after_step
        +bool use_grad_checkpoint
    }

    class PARFLM {
        +E : Embedding
        +P : Parameter
        +V_theta : ScalarPotential
        +V_phi : StructuralVPhi | MLPVPhi
        +m_global : Parameter
        +gamma : Parameter
        +ln_layers : ModuleList
        +compute_mass(x) Tensor
        +_embed(x) Tensor
        +_project(h) Tensor
        +_layer_step(h, h_prev, m, γ, dt, ℓ) Tensor
        +_stack_forward(h0, x) h_L, traj
        +forward(x, targets) logits, loss
        +generate(x, max_new) Tensor
        +num_params() int
    }

    class SparsePARFConfig {
        <<extends PARFConfig>>
        +int top_k
        +float gumbel_tau_init
        +bool gumbel_noise
        +bool use_gathered_v_phi
    }

    class ScoreHead {
        +W_q : Linear
        +W_k : Linear
        +forward(h, h_src) Tensor
    }

    class SparsePARFLM {
        +score_head : ScoreHead
        +_gumbel_tau : Buffer
        +set_gumbel_tau(τ) void
        +_sparse_mask(π, causal, T) Tensor
        +_sparse_topk_indices(π, causal, T) idx, m_g
        +_layer_step(h, h_prev, m, γ, dt, ℓ) Tensor
    }

    class MultiXiPARFConfig {
        <<extends SparsePARFConfig>>
        +int xi_channels
        +List~float~ xi_alpha_inits
        +bool xi_learnable
        +str xi_alpha_init_mode
    }

    class MultiXiPARFLM {
        +xi_module : MultiChannelXi
        +V_theta : ScalarPotentialMultiXi
        +_layer_step(h, h_prev, m, γ, dt, ℓ) Tensor
        +xi_alpha_values() List~float~
    }

    class FockPARFConfig {
        <<extends SparsePARFConfig>>
        +int n_registers
        +float register_salience_decay
    }
    class FockPARFLM {
        +register_embed : Parameter
        +creation_gates : ModuleList
        +destruction_gates : ModuleList
        +_init_registers(B, device) r, salience
        +_fock_layer_step(...) h, h_prev, r, sal
        +_stack_forward(h0, x) h_L, traj
    }

    class FockPARFConfig_v2 {
        <<extends SparsePARFConfig>>
        +int d_k
        +bool reverse_channel
    }
    class FockPARFLM_v2 {
        +creation_gate_qkv : QKVCreationGate
        +reverse_ch : ReverseChannel
        +destruction_gates : ModuleList
        +_fock_v2_layer_step(...) h, h_prev, r, sal
        +_stack_forward(h0, x) h_L, traj
    }

    class FockMultiXiPARFConfig {
        <<extends MultiXiPARFConfig>>
        +str fock_version
        +int n_registers
        +bool reverse_channel
        +bool stack_discipline
    }
    class FockMultiXiPARFLM {
        +register_embed : Parameter
        +creation_gates | creation_gate_qkv
        +destruction_gates : ModuleList
        +reverse_ch : ReverseChannel?
        +_init_registers(B, device) r, salience
        +_active_mask(salience) Tensor
        +_fock_layer_step(...) h, h_prev, r, sal
        +_stack_forward(h0, x) h_L, traj
        +get_register_overhead() int
    }

    class HybridFockPARFConfig {
        <<extends FockPARFConfig>>
        +int n_heads
    }
    class HybridFockPARF {
        +attn_layers : ModuleList
    }

    class SPHSPLMConfig_parf {
        <<extends SparsePARFConfig>>
    }
    class ScalarPotentialLMSPHSPLM_parf {
        +skew_kernel : SkewKernelLowRank
        +gyro_kernel : PerTokenGyroKernel
    }

    PARFConfig <|-- SparsePARFConfig
    SparsePARFConfig <|-- MultiXiPARFConfig
    SparsePARFConfig <|-- FockPARFConfig
    SparsePARFConfig <|-- FockPARFConfig_v2
    MultiXiPARFConfig <|-- FockMultiXiPARFConfig
    FockPARFConfig <|-- HybridFockPARFConfig
    SparsePARFConfig <|-- SPHSPLMConfig_parf

    PARFLM <|-- SparsePARFLM
    SparsePARFLM <|-- MultiXiPARFLM
    SparsePARFLM <|-- FockPARFLM
    SparsePARFLM <|-- FockPARFLM_v2
    MultiXiPARFLM <|-- FockMultiXiPARFLM
    FockPARFLM <|-- HybridFockPARF
    SparsePARFLM <|-- ScalarPotentialLMSPHSPLM_parf

    PARFLM --> ScoreHead : score_head (via Sparse)
    SparsePARFLM --> ScoreHead : score_head
    MultiXiPARFLM --> MultiChannelXi : xi_module
    FockMultiXiPARFLM --> CreationGate : v1 gates
    FockMultiXiPARFLM --> QKVCreationGate : v2 gate
    FockMultiXiPARFLM --> ReverseChannel : v2 reverse
    FockMultiXiPARFLM --> DestructionGate : v1 gates
    FockMultiXiPARFLM --> DestructionGate_v2 : v2 gates
```

---

## 3. V_θ Potential Hierarchy

The single-body scalar potential V_θ(ξ, h) drives the conservative
force on each token.  The legacy MLP version and the structured
(closed-form gradient) variants are shown.

```mermaid
classDiagram
    direction TB

    class ScalarPotential_SPLM {
        <<model.py>>
        +layers : Sequential
        +forward(ξ, h) Tensor~B,T,1~
    }

    class ScalarPotential_SARF {
        <<model_sarf_mass.py>>
        +layers : Sequential
        +forward(ξ, h) Tensor~B,T,1~
    }

    class ScalarPotential_PARF {
        <<model_parf.py, MLP>>
        +layers : Sequential
        +forward(ξ, h) Tensor~B,T,1~
    }

    class ScalarPotentialMultiXi {
        <<model_multixi.py>>
        +layers : Sequential
        +forward(xis, h) Tensor~B,T,1~
    }

    class GaussianMixturePotential {
        <<model_gm.py>>
        +mu_list : ParameterList
        +kappa_raw : Parameter
        +forward(ξ, h) Tensor~B,T,1~
    }

    class StructuredVThetaBase {
        <<abstract>>
        +forward(ξ, h) Tensor~B,T,1~
        +analytical_grad(ξ, h) Tensor~B,T,d~
    }

    class QuadraticWellVTheta {
        +W_mu : Linear
        +forward(ξ, h) Tensor
        +analytical_grad(ξ, h) Tensor
    }

    class LowRankQuadraticVTheta {
        +W_A : Linear
        +forward(ξ, h) Tensor
        +analytical_grad(ξ, h) Tensor
    }

    class MixtureQuadraticVTheta {
        +W_mu_list : ParameterList
        +gate : Linear
        +forward(ξ, h) Tensor
        +analytical_grad(ξ, h) Tensor
    }

    class HybridQuadraticVTheta {
        +quadratic : LowRankQuadraticVTheta
        +mlp_residual : Sequential
        +forward(ξ, h) Tensor
        +analytical_grad(ξ, h) Tensor
    }

    StructuredVThetaBase <|-- QuadraticWellVTheta
    StructuredVThetaBase <|-- LowRankQuadraticVTheta
    StructuredVThetaBase <|-- MixtureQuadraticVTheta
    StructuredVThetaBase <|-- HybridQuadraticVTheta

    note for ScalarPotentialMultiXi "Input is (B, T, K, d) multi-ξ\nconcatenated with h → (B, T, (K+1)d)"
```

---

## 4. V_φ Pair-Potential Hierarchy

The pair-interaction potential V_φ(h_t, h_s) mediates inter-token
forces.  Three implementations exist, all sharing the same forward
contract.

```mermaid
classDiagram
    direction TB

    class StructuralVPhi {
        +W_l : Linear
        +W_theta : Linear
        +Phi : Sequential
        +Theta : Sequential
        +forward(h, h_src) Tensor~B,T,T~
        +forward_gathered(h, h_src_g) Tensor~B,T,k~
    }

    class StructuralCompetitiveVPhi {
        <<extends StructuralVPhi>>
        +comp_scale : Parameter
        +forward(h, h_src) Tensor~B,T,T~
        +forward_gathered(h, h_src_g) Tensor~B,T,k~
    }

    class MLPVPhi {
        +layers : Sequential
        +forward(h, h_src) Tensor~B,T,T~
    }

    StructuralVPhi <|-- StructuralCompetitiveVPhi

    note for StructuralVPhi "V_φ = Φ(||l_t - l_s||²) · Θ(θ_t · θ_s)\nBilinear type × angular decomposition"
    note for StructuralCompetitiveVPhi "Adds a repulsive per-type bias\nfor competitive specialisation"
    note for MLPVPhi "Legacy unstructured MLP\nV_φ = MLP(h_t ⊕ h_s)"
```

---

## 5. Fock Gate Components

Creation and destruction gates for the Fock-space register lifecycle.
v1 uses mean-conditioned MLPs; v2 uses Q/K/V cross-attention with an
optional reverse channel.

```mermaid
classDiagram
    direction TB

    class CreationGate {
        <<v1, model_fock_parf.py>>
        +net : Sequential
        +forward(h_mean) Tensor~B,M~
    }

    class DestructionGate {
        <<v1, model_fock_parf.py>>
        +net : Sequential
        +forward(r) Tensor~B,M~
    }

    class QKVCreationGate {
        <<v2, model_fock_parf_v2.py>>
        +W_q : Linear
        +W_k : Linear
        +W_v : Linear
        +tau_create : Parameter
        +forward(h, r) r_new, α_max
    }

    class DestructionGate_v2 {
        <<v2, model_fock_parf_v2.py>>
        +net : Sequential
        +forward(r) Tensor~B,M~
    }

    class ReverseChannel {
        <<v2, model_fock_parf_v2.py>>
        +W_q : Linear
        +W_k : Linear
        +W_v : Linear
        +forward(h, r, active) Q_force~B,T,d~
    }

    note for CreationGate "σ(MLP(h̄)) → per-register\ncreation probability"
    note for QKVCreationGate "Cross-attention: tokens query\nregisters to produce new content\nand salience update"
    note for ReverseChannel "Non-conservative force Q_i on\ntokens from active registers\n(scaled by learnable tanh gate)"
```

---

## 6. Context Channel (ξ) Modules

The context channel ξ summarises past token history for V_θ.
Three implementations provide increasing representational capacity.

```mermaid
classDiagram
    direction TB

    class CausalCumulativeMean {
        <<function, not a class>>
        +causal_cumulative_mean(h) Tensor~B,T,d~
    }

    class MultiChannelXi {
        <<model_multixi.py>>
        +raw_alpha : Parameter~K~
        +forward(h) Tensor~B,T,K,d~
        +alpha_values() List~float~
    }

    class MultiChannelS4D {
        <<model_multixi_s4d.py>>
        +Lambda_re : Parameter
        +Lambda_im : Parameter
        +forward(h) Tensor~B,T,K,d~
    }

    class MultiChannelHiPPO {
        <<model_multixi_hippo.py>>
        +A : Buffer (N×N HiPPO matrix)
        +B : Parameter
        +forward(h) Tensor~B,T,K,d~
    }

    note for MultiChannelXi "K learnable EMA channels\nα_k = sigmoid(raw_alpha_k)\nξ_k[t] = α_k · ξ_k[t-1] + (1-α_k) · h[t]"
    note for MultiChannelS4D "Diagonal SSM with complex\neigenvalues (S4D kernel)"
    note for MultiChannelHiPPO "HiPPO-LegS polynomial\nprojection basis"
```

---

## 7. Causal Probe Modules

Three generations of causal-violation probes, each covering
a wider portion of the model hierarchy.  All share the same two
core tests (perturbation + gradient-Jacobian).

```mermaid
classDiagram
    direction TB

    class causal_probe {
        <<causal_probe.py>>
        +causal_violation_probe(model, ...) results
        +class_smoke(strict) int
        +natural_probe(targets, strict) int
        +ckpt_probe(ckpt_path, strict) int
        +main() int
    }

    class causal_probe_helmholtz {
        <<helmholtz/causal_probe.py>>
        +perturbation_probe(model, ...) pre, post, Δ
        +gradient_probe(model, ...) post, pre, norms
        +assert_causal(model, ...) void | RuntimeError
        +probe_one_schedule(sched, ...) ok, details
        +run_all(strict, schedules) int
    }

    class causal_probe_parf {
        <<parf/causal_probe_parf.py>>
        +perturbation_probe(model, ...) pre, post, Δ
        +gradient_probe(model, ...) post, pre, norms
        +assert_causal(model, ...) void | RuntimeError
        +probe_one_variant(vphi, ...) ok, details
        +run_all(strict, variants) int
    }

    class causal_probe_multixi {
        <<parf/causal_probe_multixi.py>>
        +perturbation_probe(model, ...) pre, post, Δ
        +gradient_probe(model, ...) post, pre, norms
        +assert_causal(model, ...) void | RuntimeError
        +probe_one_variant(key, ...) ok, details
        +run_all(strict, variants) int
    }

    note for causal_probe "Gen-1: covers vanilla SPLM,\nSARF-Mass, EM-LN variants"
    note for causal_probe_helmholtz "Gen-2: covers HelmholtzLM\n(all schedules)"
    note for causal_probe_parf "Gen-3a: covers dense PARFLM\n(structural + MLP V_φ)"
    note for causal_probe_multixi "Gen-3b: covers MultiXiPARFLM (K=2,4)\nand FockMultiXiPARFLM (v1, v2±rev)"
```

**Two causal-probe conventions are used in the codebase:**

| Convention | Where | Behaviour |
|---|---|---|
| **Startup abort** (`assert_causal()`) | Helmholtz, PARF, Multi-Xi, Fock trainers | Raises `RuntimeError` before optimiser step 0; skippable via `--skip-causal-check` |
| **In-training monitoring** (`causal_leak_probe()`) | Non-conservative, SP-HSPLM trainers | Saves JSON snapshots at init / mid / final steps; does not abort |

**Model coverage matrix:**

```mermaid
graph LR
    subgraph "causal_probe.py (Gen-1)"
        A1[ScalarPotentialLM]
        A2[SPLM SARF-Mass]
        A3[SPLM EM-LN]
        A4[Multi-Xi SPLM]
    end

    subgraph "causal_probe_helmholtz (Gen-2)"
        B1[HelmholtzLM]
    end

    subgraph "causal_probe_parf (Gen-3a)"
        C1[PARFLM dense]
    end

    subgraph "causal_probe_multixi (Gen-3b)"
        D1[MultiXiPARFLM K=2]
        D2[MultiXiPARFLM K=4]
        D3[FockMultiXiPARFLM v1]
        D4[FockMultiXiPARFLM v2+rev]
        D5[FockMultiXiPARFLM v2−rev]
    end
```

---

## 8. Training Scripts & Notebooks

```mermaid
graph TB
    subgraph "Training Scripts"
        TS1["train_splm.py\n→ ScalarPotentialLM"]
        TS2["train_splm_sarf_mass.py\n→ SPLM SARF-Mass"]
        TS3["train_splm_em_ln.py\n→ SPLM EM-LN"]
        TS4["train_helmholtz.py\n→ HelmholtzLM"]
        TS5["train_splm_hybrid.py\n→ HybridSPLM"]
        TS6["train_parf.py\n→ PARFLM (dense)"]
        TS7["train_fock_parf.py\n→ FockPARFLM v1/v2"]
    end

    subgraph "Scaleup Training Scripts"
        SS1["train_splm_em_ln_scaleup.py\n→ SPLM EM-LN H=128"]
        SS2["train_splm_em_ln_multixi_scaleup.py\n→ Multi-Xi SPLM H=128"]
        SS3["train_splm_em_ln_multixi_s4d_scaleup.py\n→ Multi-S4D SPLM"]
        SS4["train_splm_em_ln_multixi_hippo_scaleup.py\n→ Multi-HiPPO SPLM"]
        SS5["train_helmholtz_scaleup.py\n→ HelmholtzLM H=128"]
        SS6["train_hybrid_scaleup.py\n→ HybridSPLM H=128"]
        SS7["train_matched_baseline_scaleup.py\n→ MatchedGPT (attention)"]
        SS8["train_parf_scaleup.py\n→ SparsePARFLM H=128"]
        SS9["train_parf_multixi_scaleup.py\n→ MultiXiPARFLM H=128"]
        SS10["train_fock_multixi_scaleup.py\n→ FockMultiXiPARFLM H=128"]
    end

    subgraph "Colab Notebooks"
        NB1["colab_parf_multixi_h128.ipynb\n5 arms: comp_K2..comp_K8"]
        NB2["colab_fock_multixi_h128.ipynb\n8 arms: v1/v2 × M × discipline"]
    end

    NB1 -->|subprocess| SS9
    NB2 -->|subprocess| SS10
    SS9 -->|import| CP1["causal_probe_multixi\nassert_causal()"]
    SS10 -->|import| CP1
```

---

## 9. Data Pipeline

```mermaid
graph LR
    subgraph "data_module.py"
        D1["load_tiny_shakespeare()"] --> D3["get_batch(ids, B, T)"]
        D2["load_tiny_stories()"] --> D3
    end

    subgraph "Tokenisation"
        D2 -->|"HuggingFace parquet\n→ GPT-2 BPE"| T1["np.ndarray\n(train_ids, val_ids)"]
    end

    D3 -->|"random slicing\n+ device transfer"| B1["(x, y) batches\nshape (B, T)"]

    subgraph "Frequency Prior"
        FP["compute_unigram_frequencies\n_tinystories.py"] --> LF["logfreq_surprisal\n_tinystories.npy"]
    end

    LF -->|"mass_mode=\n'logfreq_surprisal'"| MASS["compute_mass(x)"]
```

---

## 10. Attention Baseline

The parameter-matched GPT-2-style transformer used as the
attention-only comparison target throughout the paper.

```mermaid
classDiagram
    direction TB

    class MatchedConfig {
        +int vocab_size
        +int n_layer
        +int n_head
        +int n_embd
        +int block_size
        +float dropout
    }

    class CausalSelfAttention {
        +c_attn : Linear
        +c_proj : Linear
        +forward(x) Tensor
    }

    class MLP {
        +c_fc : Linear
        +c_proj : Linear
        +forward(x) Tensor
    }

    class Block {
        +ln_1 : LayerNorm
        +ln_2 : LayerNorm
        +attn : CausalSelfAttention
        +mlp : MLP
        +forward(x) Tensor
    }

    class MatchedGPT {
        +transformer : ModuleDict
        +lm_head : Linear
        +forward(x, targets) logits, loss
        +generate(x, max_new) Tensor
    }

    MatchedGPT *-- Block : n_layer blocks
    Block *-- CausalSelfAttention
    Block *-- MLP
```

---

## 11. Non-Conservative Force Hierarchy

The non-conservative extension of the SPLM adds learned divergence-free
(solenoidal) or skew-symmetric forces that break the conservative
constraint.  Used for the ablation study in the paper.

```mermaid
classDiagram
    direction TB

    class NonConservativeForce {
        <<abstract, model_splm_nonconservative.py>>
        +forward(h, h_prev) Tensor~B,T,d~
    }

    class ConstantSkewForce {
        +S : Parameter~d,d~
        +forward(h, h_prev) Tensor
    }

    class AffineRank1SkewForce {
        +u : Parameter~d~
        +v : Parameter~d~
        +forward(h, h_prev) Tensor
    }

    class LowRankSkewForce {
        +U : Parameter~d,r~
        +V : Parameter~d,r~
        +forward(h, h_prev) Tensor
    }

    class LowRankSolenoidalForce {
        +U : Parameter~d,r~
        +V : Parameter~d,r~
        +W_curl : Linear
        +forward(h, h_prev) Tensor
    }

    NonConservativeForce <|-- ConstantSkewForce
    NonConservativeForce <|-- AffineRank1SkewForce
    NonConservativeForce <|-- LowRankSkewForce
    NonConservativeForce <|-- LowRankSolenoidalForce

    note for NonConservativeForce "f_nc is added to the conservative\nforce f = −∇U inside _layer_step"
```

---

## 12. Sequence Diagram — SPLM Training Loop

Core training workflow shared by all SPLM variants (vanilla, SARF-Mass,
EM-LN, Multi-Xi).  The PARF and Fock variants extend this with
additional components shown in later diagrams.

```mermaid
sequenceDiagram
    participant Main as train_splm_em_ln_scaleup.py
    participant DM as data_module
    participant Cfg as SPLMSARFMassLNConfig
    participant Model as ScalarPotentialLMSARFMassLN
    participant Optim as AdamW
    participant Sched as lr_schedule / tau_schedule

    Main->>DM: load_tiny_stories()
    DM-->>Main: train_ids, val_ids

    Main->>Cfg: build_config(args)
    Cfg-->>Main: cfg, train_cfg, tag

    Main->>Model: __init__(cfg).to(device)
    Model-->>Main: model

    Main->>Optim: AdamW(model.parameters(), lr, wd)

    loop step = 0 .. steps-1
        Main->>Sched: lr_schedule(step) → lr_now
        Main->>DM: get_batch(train_ids, B, T)
        DM-->>Main: x, y

        Main->>Model: forward(x, targets=y)
        activate Model
        Model->>Model: _embed(x) → h0
        Model->>Model: compute_mass(x) → m_b
        loop ℓ = 0 .. L-1
            Model->>Model: _layer_step(h, h_prev, m, γ, dt, ℓ)
            Note right of Model: ξ = causal_cumul_mean(h.detach())<br/>U = V_θ(ξ, h)<br/>f = −∇_h U<br/>h_new = Verlet(h, f, m, γ, dt)
        end
        Model->>Model: logits = h_L @ E.weight.T
        Model->>Model: loss = cross_entropy(logits, y)
        deactivate Model
        Model-->>Main: logits, loss

        Main->>Main: loss.backward()
        Main->>Main: clip_grad_norm_
        Main->>Optim: step()

        alt step % eval_interval == 0
            Main->>Model: evaluate(val_ids)
            Model-->>Main: val_loss, val_ppl
            Main->>Main: log to JSONL + print
        end
    end

    Main->>Main: save checkpoint, loss curve, summary
```

---

## 13. Sequence Diagram — PARF/Multi-Xi Training Loop

Extends the SPLM loop with pair-potential V_φ, Gumbel-softmax sparse
routing, and multi-channel ξ.

```mermaid
sequenceDiagram
    participant Main as train_parf_multixi_scaleup.py
    participant DM as data_module
    participant Probe as causal_probe_multixi
    participant Model as MultiXiPARFLM
    participant Xi as MultiChannelXi
    participant VT as V_theta (MultiXi)
    participant SH as ScoreHead
    participant VP as V_phi (Structural)
    participant Optim as AdamW

    Main->>DM: load_tiny_stories()
    DM-->>Main: train_ids, val_ids

    Main->>Model: MultiXiPARFLM(cfg).to(device)

    Note over Main,Probe: Pre-training causal audit
    Main->>Probe: assert_causal(model, vocab_size)
    Probe->>Model: perturbation_probe()
    Probe->>Model: gradient_probe()
    Probe-->>Main: passed ✓

    Main->>Optim: AdamW(model.parameters())

    loop step = 0 .. steps-1
        Main->>Main: lr_schedule(step), tau_schedule(step)
        Main->>Model: set_gumbel_tau(τ)
        Main->>DM: get_batch(train_ids, B, T)
        DM-->>Main: x, y

        Main->>Model: forward(x, targets=y)
        activate Model
        Model->>Model: h0 = _embed(x)
        Model->>Model: m_b = compute_mass(x)

        loop ℓ = 0 .. L-1
            Model->>Xi: xi_module(h.detach()) → xis (B,T,K,d)
            Model->>VT: V_theta(xis, h_in) → V_θ per token
            Model->>SH: score_head(h, h_src) → π (B,T,T)
            Model->>Model: sparse_topk_indices(π) → idx, m_g
            Model->>VP: V_phi.forward_gathered(h, h_src_g)
            Model->>Model: U = V_θ.sum() + (V_φ · m_g).sum()
            Model->>Model: f = −∇_h U   (autograd.grad)
            Model->>Model: h_new = Verlet(h, f, m, γ, dt)
            Model->>Model: _project(h_new)  [LayerNorm]
        end

        Model->>Model: logits = h_L @ E.weight.T
        Model-->>Main: logits, loss
        deactivate Model

        Main->>Main: loss.backward(), clip, step
    end
```

---

## 14. Sequence Diagram — Fock Multi-Xi Training Loop

Extends the Multi-Xi loop with Fock register creation/destruction
lifecycle per layer.

```mermaid
sequenceDiagram
    participant Main as train_fock_multixi_scaleup.py
    participant Probe as causal_probe_multixi
    participant Model as FockMultiXiPARFLM
    participant CG as CreationGate / QKV
    participant DG as DestructionGate
    participant RC as ReverseChannel
    participant Super as MultiXiPARFLM._layer_step

    Note over Main,Probe: Pre-training causal audit
    Main->>Probe: assert_causal(model, vocab_size)
    Probe-->>Main: passed ✓

    Main->>Model: forward(x, targets=y)
    activate Model
    Model->>Model: h0 = _embed(x)
    Model->>Model: r, salience = _init_registers(B, device)
    Model->>Model: h = h0, h_prev = h0

    loop ℓ = 0 .. L-1
        Note over Model,CG: Register Creation
        alt fock_version == "v1"
            Model->>CG: creation_gates[ℓ](h.mean(dim=1))
            CG-->>Model: g_create (B, M)
            Model->>Model: salience = decay · sal + g_create · (1−decay)
        else fock_version == "v2"
            Model->>CG: creation_gate_qkv(h, r)
            CG-->>Model: r_new_content, α_max
            Model->>Model: blend r, update salience
        end

        Model->>Model: active = _active_mask(salience)
        Model->>Model: h_ext = cat([h, r·active], dim=1)
        Model->>Model: h_prev_ext = cat([h_prev, r·active], dim=1)

        Note over Model,Super: Multi-Xi PARF Dynamics on Extended State
        Model->>Super: _layer_step(h_ext, h_prev_ext, m_ext, γ, dt, ℓ)
        Note right of Super: K-EMA ξ channels<br/>V_θ(xis, h) + V_φ sparse routing<br/>f = −∇U, Verlet step
        Super-->>Model: h_ext_new

        Model->>Model: h_new = h_ext_new[:, :T, :]
        Model->>Model: r_new = h_ext_new[:, T:, :]

        alt v2 + reverse_channel + active.any()
            Note over Model,RC: Reverse Channel Force
            Model->>RC: reverse_ch(h_new, r_new, active)
            RC-->>Model: Q_force (B, T, d)
            Model->>Model: h_new += (dt²/m) · tanh(scale) · Q_force
        end

        Note over Model,DG: Register Destruction
        Model->>DG: destruction_gates[ℓ](r_new)
        DG-->>Model: g_destroy (B, M)
        Model->>Model: salience *= (1 − g_destroy · active)

        Model->>Model: h_prev = h, h = h_new
    end

    Model->>Model: logits = h @ E.weight.T
    Model-->>Main: logits, loss
    deactivate Model
```

---

## 15. Sequence Diagram — Causal Probe (Pre-Training)

The causal-violation probe runs two complementary tests on a freshly
initialised model before the optimiser is constructed.

```mermaid
sequenceDiagram
    participant Trainer as train_*_scaleup.py
    participant Probe as assert_causal()
    participant Model as PARFLM hierarchy
    participant AG as torch.autograd

    Trainer->>Trainer: model = ModelClass(cfg).to(device)

    alt --skip-causal-check
        Trainer->>Trainer: print("SKIPPED")
    else default
        Trainer->>Probe: assert_causal(model, vocab_size, T=32, t_pert=20)

        Note over Probe,Model: Test 1: Perturbation Probe
        Probe->>Probe: x_a = random tokens (1, 32)
        Probe->>Probe: x_b = x_a with x[20] changed
        Probe->>Model: model(x_a) → logits_a
        Probe->>Model: model(x_b) → logits_b
        Probe->>Probe: Δ = |logits_a − logits_b| per position
        Probe->>Probe: assert Δ[:20].max() < 1e-6

        Note over Probe,AG: Test 2: Gradient-Jacobian Probe
        Probe->>Model: _embed(x) → emb
        Probe->>Probe: emb_in = emb.detach().requires_grad_(True)
        Probe->>Model: _stack_forward(emb_in, x) → h_L
        Probe->>Probe: target = (h_L @ E.weight.T)[0, 20, :].sum()
        Probe->>AG: autograd.grad(target, emb_in)
        AG-->>Probe: gradient norms per position
        Probe->>Probe: assert grad_norms[21:].max() < 1e-6

        alt any assertion fails
            Probe--xTrainer: RuntimeError("LEAK DETECTED")
        else both pass
            Probe-->>Trainer: return (no error)
            Trainer->>Trainer: print("causal probe passed")
        end
    end

    Trainer->>Trainer: optim = AdamW(...)
    Trainer->>Trainer: begin training loop
```

---

## 16. Sequence Diagram — Inference / Generation

Token-by-token autoregressive generation, shared by all model
families (the force-based dynamics run at each step just as in
training, but without gradient accumulation).

```mermaid
sequenceDiagram
    participant User as Caller
    participant Model as *LM.generate()
    participant Embed as _embed()
    participant Stack as _stack_forward()
    participant Layer as _layer_step() × L

    User->>Model: generate(x_prompt, max_new_tokens=N)

    loop t = 0 .. N-1
        Model->>Model: x_ctx = x[-block_size:]
        Model->>Embed: _embed(x_ctx)
        Embed-->>Stack: h0

        Stack->>Stack: m_b = compute_mass(x_ctx)
        loop ℓ = 0 .. L-1
            Stack->>Layer: _layer_step(h, h_prev, m, γ, dt, ℓ)
            Note right of Layer: Conservative force dynamics<br/>(no create_graph needed)
            Layer-->>Stack: h_new
        end
        Stack-->>Model: h_L

        Model->>Model: logits = h_L[:, -1, :] @ E.weight.T
        Model->>Model: next_token = sample(softmax(logits / τ))
        Model->>Model: x = cat([x, next_token])
    end

    Model-->>User: x (prompt + generated)
```

---

## 17. File Map

Quick reference from file path to the classes and modules defined
within.

| File | Key Exports |
|------|-------------|
| `model.py` | `SPLMConfig`, `ScalarPotential`, `ScalarPotentialLM` |
| `sarf_variant/model_sarf.py` | `SPLMSARFConfig`, `ScalarPotentialLMSARF`, `causal_cumulative_mean` |
| `sarf_mass_variant/model_sarf_mass.py` | `SPLMSARFMassConfig`, `ScalarPotentialLMSARFMass` |
| `energetic_minima/model_ln.py` | `SPLMSARFMassLNConfig`, `ScalarPotentialLMSARFMassLN` |
| `energetic_minima/model_gm.py` | `SPLMSARFMassGMConfig`, `ScalarPotentialLMSARFMassGM`, `GaussianMixturePotential` |
| `multixi/model_multixi.py` | `MultiChannelXi`, `ScalarPotentialMultiXi`, `…LNMultiXi` |
| `multixi/model_multixi_s4d.py` | `MultiChannelS4D`, `…LNMultiS4D` |
| `multixi/model_multixi_hippo.py` | `MultiChannelHiPPO`, `…LNMultiHiPPO` |
| `helmholtz/model_helmholtz.py` | `HelmholtzConfig`, `HelmholtzLM` |
| `hybrid/model_hybrid.py` | `HSPLMConfig`, `HybridSPLM`, `_NullAttnHybrid` |
| `first_order_ablation/model_first_order.py` | `SPLMFirstOrderConfig`, `…FirstOrder` |
| `symplectic_variant/model_symplectic.py` | `SPLMSymplecticConfig`, `…Symplectic` |
| `non_conservative/model_splm_nonconservative.py` | `NonConservativeForce` (+ `ConstantSkew`, `AffineRank1Skew`, `LowRankSkew`, `LowRankSolenoidal`), `…NonConservative` |
| `sphsplm/model_sphsplm.py` | `SPHSPLMConfig`, `…SPHSPLM`, skew/gyro kernels |
| `matched_baseline_model.py` | `MatchedConfig`, `MatchedGPT`, `CausalSelfAttention` |
| `parf/model_parf.py` | `PARFConfig`, `PARFLM`, `StructuralVPhi`, `MLPVPhi`, `StructuralCompetitiveVPhi` |
| `parf/model_parf_sparse.py` | `SparsePARFConfig`, `SparsePARFLM`, `ScoreHead` |
| `parf/model_parf_multixi.py` | `MultiXiPARFConfig`, `MultiXiPARFLM` |
| `parf/model_fock_parf.py` | `FockPARFConfig`, `FockPARFLM`, `CreationGate`, `DestructionGate` |
| `parf/model_fock_parf_v2.py` | `FockPARFConfig_v2`, `FockPARFLM_v2`, `QKVCreationGate`, `ReverseChannel`, `DestructionGate_v2` |
| `parf/model_fock_parf_multixi.py` | `FockMultiXiPARFConfig`, `FockMultiXiPARFLM` |
| `parf/model_hybrid_fock_parf.py` | `HybridFockPARFConfig`, `HybridFockPARF` |
| `parf/model_structured_vtheta.py` | `StructuredVThetaBase`, `QuadraticWellVTheta`, `LowRankQuadraticVTheta`, `MixtureQuadraticVTheta`, `HybridQuadraticVTheta` |
| `parf/dyck_data.py` | `DyckConfig` (synthetic Dyck language data) |
| `data_module.py` | `load_tiny_shakespeare()`, `load_tiny_stories()`, `get_batch()` |
| `causal_probe.py` | Gen-1 probe: `causal_violation_probe`, `class_smoke` |
| `helmholtz/causal_probe.py` | Gen-2 probe: `assert_causal` for HelmholtzLM |
| `parf/causal_probe_parf.py` | Gen-3a probe: `assert_causal` for dense PARFLM |
| `parf/causal_probe_multixi.py` | Gen-3b probe: `assert_causal` for MultiXi + Fock PARFLM |

All paths are relative to `notebooks/conservative_arch/`.
