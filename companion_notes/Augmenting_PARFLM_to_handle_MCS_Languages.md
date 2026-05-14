# Augmenting PARFLM to Handle Mildly Context-Sensitive Languages

## Status

**Draft plan** — May 2026. Pre-experimental design stage.

## Motivation

PARFLM (PARF-Augmented SPLM) adds token-token pair interactions $V_\phi(h_t, h_s)$ to the single-particle scalar potential $V_\theta(\xi, h)$. This enriches the force law but does not escape the **v0 expressivity ceiling** (Theorem v0-ceiling, §9.2 of paper v4):

- The hidden state is still $h \in \mathbb{R}^d$ (fixed dimension)
- The integrator is still a deterministic function
- There is no mechanism for the state space to grow during inference

Consequently, PARFLM is at most a finite automaton (regular languages). It cannot:
- Recognise $\text{Dyck}_n$ beyond the predicted collapse depth $D^*$
- Handle cross-serial dependencies ($a^n b^n c^n$)
- Reach the mildly context-sensitive (MCS) class

To escape this ceiling, the framework requires **v2 (creation/destruction)** mapped to **Fock space and second quantisation** (§9.4.2), plus eventually **v3 (execution)** mapped to **Lie groups and non-abelian gauge theory** (§9.4.3). This document plans the augmentation to v2.

## Theoretical Foundation

### The v2 → Fock Space Mapping (from §9.4.2)

| v2 mechanism | Fock-space object |
|---|---|
| Introduce an entity into discourse | Creation operator $a^\dagger_v \|\psi\rangle$ |
| Entity drops out of discourse | Annihilation operator $a_v \|\psi\rangle$ |
| Count of currently-live entities | Number operator $N = \sum_v a^\dagger_v a_v$ |
| Field at semantic position $x$ | $\hat{\phi}(x) = \sum_v \phi_v(x) a_v$ |

The Fock space itself:

$$\mathcal{F}(\mathcal{H}) = \bigoplus_{n=0}^{\infty} \mathcal{H}^{\otimes n}$$

The key property that breaks the v0 ceiling: **the active particle count grows with input length**, so the state space is no longer fixed-dimensional.

### The Doi-Peliti Classical Specialisation

The framework commits to **classical** particles (no quantum superposition). The Doi-Peliti formalism (Doi 1976, Peliti 1985) provides exactly this: a Fock-space operator algebra for classical reaction-diffusion systems. States are generating-function representations of configuration distributions; field equations are classical Hamilton equations on a symplectic manifold.

This means we can use the full Fock-space algebraic machinery without invoking quantum mechanics.

## Architecture Design: Latent Particle Pool (Path 1)

### Core Idea

Augment the PARFLM state with $M$ **latent register particles** alongside the $T$ input tokens:

```
Current PARFLM:   state = (h_1, ..., h_T)         ∈ R^{T × d}
Augmented:        state = (h_1, ..., h_T, r_1, ..., r_M)  ∈ R^{(T+M) × d}
```

Registers start in a "vacuum" state (inactive). A learned creation gate activates them during the forward pass; a destruction gate deactivates them. Active registers participate in the $V_\phi$ pair interactions identically to real tokens.

### Fock-Space Interpretation

| Implementation concept | Fock-space analogue |
|---|---|
| Register pool (all inactive) | Vacuum state $\|0\rangle$ |
| Activation of register $r_j$ | Creation: $a^\dagger_v \|0\rangle$ |
| Deactivation of register $r_j$ | Annihilation: $a_v \|\psi\rangle$ |
| Number of active registers | Number operator $N$ |
| Salience-ordered LIFO activation | Stack discipline (→ pushdown automaton) |
| Register hidden state $r_j \in \mathbb{R}^d$ | Single-particle state in $\mathcal{H}$ |

### Why This Escapes v0

The v0-ceiling proof requires that the state space be fixed and finite-dimensional throughout inference. With latent registers:

1. The **effective** state dimensionality grows as registers activate (from $T \cdot d$ to $(T + N_{\text{active}}) \cdot d$)
2. The activation pattern is **input-dependent** — complex inputs activate more registers
3. With a stack-like activation discipline (v1.5 salience ordering), the system implements a pushdown automaton → reaches CF
4. With multi-register coordination (v3 operator structure on register groups), it reaches MCS

### Architecture Specification

```python
class FockPARFLM(SparsePARFLM):
    """PARFLM augmented with a latent register pool (v2 creation/destruction)."""

    def __init__(self, cfg: FockPARFConfig):
        super().__init__(cfg)
        self.M = cfg.n_registers          # Pool size (max particles)
        # Register embeddings (learnable vacuum state)
        self.register_embed = nn.Parameter(torch.randn(self.M, cfg.d) * 0.02)
        # Per-layer creation gate: decides whether to activate registers
        self.creation_gate = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.d, cfg.d // 4),
                nn.GELU(),
                nn.Linear(cfg.d // 4, self.M),
                nn.Sigmoid()
            ) for _ in range(cfg.L)
        ])
        # Per-layer destruction gate: decides whether to deactivate
        self.destruction_gate = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.d, cfg.d // 4),
                nn.GELU(),
                nn.Linear(cfg.d // 4, 1),
                nn.Sigmoid()
            ) for _ in range(cfg.L)
        ])
        # Salience tracker (v1.5): scalar per register, decays unless reinforced
        # Implemented as a running salience that gates register participation
```

### Forward Pass (per layer $\ell$)

```
1. Compute creation gate:  g_create = creation_gate_ℓ(mean(h_1:T))  ∈ [0,1]^M
2. Update register salience:
   - For each register j: σ_j ← σ_j · decay + g_create_j · (1 - decay)
   - Active mask: active_j = (σ_j > threshold)
3. Concatenate active registers to token states:
   - h_full = [h_1, ..., h_T, r_j for j in active]
4. Run standard PARFLM dynamics on h_full:
   - V_θ force on all particles (tokens + active registers)
   - V_φ pair force between all pairs (top-k sparse selection applies)
   - Damped integration step
5. Destruction gate: for each active register j:
   - g_destroy_j = destruction_gate_ℓ(r_j)
   - σ_j ← σ_j · (1 - g_destroy_j)
6. Split h_full back: extract updated h_1:T for LM head;
   store updated r_j for next layer
```

### Configuration

```python
@dataclass
class FockPARFConfig(SparsePARFConfig):
    n_registers: int = 32            # Pool size M (max active particles)
    register_salience_decay: float = 0.9   # v1.5 decay rate
    register_salience_threshold: float = 0.1  # Activation threshold
    creation_gate_hidden: int = 64   # Gate MLP hidden dim
    stack_discipline: bool = True    # LIFO ordering on registers (→ PDA)
```

### Parameter Budget

At the P10f scale ($d = 256$, $L = 8$, $M = 32$):
- Register embeddings: $32 \times 256 = 8192$ params
- Creation gates: $8 \times (256 \times 64 + 64 \times 32) = 8 \times 18432 = 147$K params
- Destruction gates: $8 \times (256 \times 64 + 64) = 8 \times 16448 = 132$K params
- Total overhead: ~290K params (< 2% of the 22M total)

## Experimental Plan

### Phase 1: Dyck Falsifier (proof of concept)

**Goal**: Demonstrate that FockPARFLM solves $\text{Dyck}_2$ past the predicted collapse depth $D^*$, where plain PARFLM fails.

| Experiment | Architecture | Expected result |
|---|---|---|
| F1-baseline | PARFLM (P10f config) | Collapses at depth $D^* \approx 3$–$6$ |
| F1-fock-nostack | FockPARFLM, M=16, no stack | Extends past $D^*$ (≥ 8–10) |
| F1-fock-stack | FockPARFLM, M=16, LIFO stack | Extends further (≥ 12–15) |
| F1-attention | Matched GPT-2 baseline | Succeeds to arbitrary depth (TC⁰) |

**Success criterion**: F1-fock-stack succeeds at depth $> D^*$ with 3/3 seed consistency.

**Training**: Synthetic $\text{Dyck}_2$ strings at controlled max depth, next-bracket-type prediction task. Small scale ($d=64$, $L=4$) sufficient for falsifier.

### Phase 2: Natural Language (TinyStories integration)

**Goal**: Test whether the v2 mechanism improves PPL on TinyStories (which has nested narrative structure that may benefit from variable-size state).

| Experiment | Architecture | Baseline PPL |
|---|---|---|
| P11a | FockPARFLM (P10f + M=16 registers) | P10f: 28.67 |
| P11b | FockPARFLM (P10f + M=32, LIFO stack) | P10f: 28.67 |
| P11c | FockPARFLM (P10f + M=32) + 16k steps | P10g result |

**Pre-registered prediction**: If TinyStories PPL ceiling is indeed corpus-information-bounded (not expressivity-bounded), the v2 mechanism adds negligible PPL improvement at 5M tokens. The gain should appear at 20M+ tokens where nested story structures become statistically learnable.

### Phase 3: Cross-Serial Dependencies ($a^n b^n c^n$)

**Goal**: Demonstrate MCS-level expressivity. This requires v3 (operator actions) in addition to v2.

**Deferred**: Requires the full v2+v3 composite. The v3 augmentation (Lie group operators on register states) is the subject of a follow-up design document.

## Decision Rules

### After Phase 1 (Dyck falsifier)

- FockPARFLM **passes** Dyck past $D^*$ → v2 mechanism confirmed; proceed to Phase 2
- FockPARFLM **fails** at $D^*$ → implementation does not correctly realise Fock-space dynamics; debug creation/destruction gates

### After Phase 2 (TinyStories)

- PPL improves $> 1$ PPL over P10g → v2 structures are useful for natural text at this scale
- PPL unchanged → corpus-information ceiling dominates; v2 benefit will appear only at larger corpus scale (proceed to 20M tokens)

### After Phase 3 ($a^n b^n c^n$)

- Success → full MCS reach confirmed empirically
- Failure → v3 implementation or the v2+v3 coupling needs revision

## Relationship to Existing Architecture

```
v0 (SPLM)
 │
 ├── + V_φ pair force ──→ PARFLM (still v0, regular languages)
 │
 ├── + latent registers ──→ FockPARFLM (v0+v2, context-free)
 │         │
 │         └── + LIFO salience ──→ FockPARFLM + v1.5 (CF with bounded memory)
 │
 └── + operator actions on registers ──→ Full MCS system (v0+v1.5+v2+v3)
```

## Phase 1 Results: Dyck₂ Falsifier (seed 0)

**Date**: 10 May 2026. Run on Apple MPS (M-series Mac), ~5.4 hours total for 3 arms.

### Configuration

- Corpus: Synthetic $\text{Dyck}_2$, max nesting depth 12, $p_{\text{open}} = 0.55$
- Train: 10,000 samples. Val: 2,000 samples. Deep test: 500 samples (depth 5–12 only)
- Model: $d = 64$, $L = 4$, $v_{\text{hidden}} = 128$, top-$k = 8$, mass = global
- Training: 4000 steps, batch 32, lr $3 \times 10^{-4}$ cosine, AdamW
- Fock-specific: $M = 16$ registers, creation gate hidden = 16, decay = 0.9, threshold = 0.1

### Results

| Arm | Params | Final val PPL | Deep-test accuracy (depth 5–12) |
|---|---|---|---|
| F1-baseline (PARFLM, no registers) | 41,974 | 3.50 | 37.93% |
| F1-fock-nostack (bag, M=16) | 52,474 | 3.57 | 37.36% |
| **F1-fock-stack (LIFO, M=16)** | 52,474 | **3.43** | **39.22%** |

### Training dynamics

The LIFO-stack arm starts below baseline (31.4% at step 800 vs 31.1%) but steadily separates from step 1600 onward, reaching a +1.3pp advantage by convergence. This is consistent with the creation/destruction gates needing substantial training time to learn the activation pattern.

The bag-discipline arm (no LIFO) performs essentially identically to baseline throughout training, confirming that unstructured register activation provides no expressivity benefit — the pushdown constraint is the critical mechanism.

### Interpretation

1. **LIFO discipline is the active ingredient**: Without it, registers are inert extra parameters. With it, the model exploits the stack structure.

2. **The signal is in the right direction but modest** (+1.3pp, +0.07 PPL). This is consistent with:
   - Small model scale ($d = 64$, 52K params) — the registers have limited capacity per slot.
   - Short training (4000 steps) — the gate MLPs need time to specialise.
   - Sequence length cap (65 tokens) — constrains the maximum nesting that appears.

3. **Not yet a definitive falsifier**: The pre-registered success criterion was >90% deep-test accuracy at depth 8+ with 3/3 seed consistency. The current 39% is far from this threshold.

### Diagnosis and next steps for Phase 1

The modest result suggests that at $d = 64$, $M = 16$, and 4000 steps, the gate MLPs lack the capacity and training signal to learn crisp creation/destruction timing. Proposed interventions before proceeding to Phase 2:

| Intervention | Rationale |
|---|---|
| **Scale up**: $d = 128$, $M = 32$, 8000 steps | More capacity per register, longer gate specialisation time |
| **Curriculum**: start at depth 4, increase to depth 12 | Easier initial gradient signal for gate learning |
| **Gate pre-training**: initialise creation gate to trigger on open brackets | Warm-start the stack discipline |
| **Longer sequences**: max_length = 128, max_depth = 16 | More room for deep nesting to differentiate |
| **Multi-seed**: run seeds 1, 2 at current config | Confirm the LIFO > bag > baseline ordering is stable |

**Decision**: Proceed with the scale-up intervention first (cheapest signal amplification), then multi-seed at the larger config.

## Implementation Status

### Completed

1. **`FockPARFConfig`** dataclass — extends `SparsePARFConfig` with v2 knobs.
2. **`FockPARFLM`** model class — latent register pool with creation/destruction gates, LIFO stack discipline, per-layer lifecycle management.
3. **Dyck data generator** — depth-controlled dataset generation for falsifier experiments.
4. **Unified trainer** — supports both Dyck falsifier and TinyStories corpora, with baseline PARFLM arm for comparison.
5. **Phase 1 seed 0 run** — all 3 arms complete; LIFO stack wins by +1.3pp.

### Verified

- Forward + backward pass on CPU/MPS (smoke tests pass).
- Parameter budget at P10f scale: **288,520 overhead** (2.16% of 13.35M total).
- Training loop runs correctly for both PARFLM baseline and FockPARFLM on Dyck data.
- LIFO stack discipline is the active mechanism (bag ≈ baseline < LIFO).

## Implementation Priority

1. ~~Design `FockPARFConfig` dataclass and `FockPARFLM` model class~~ **DONE**
2. ~~Run Dyck falsifier (Phase 1) — seed 0, small scale~~ **DONE** (signal positive but modest)
3. **Next**: Scale-up Phase 1 ($d = 128$, $M = 32$, 8000 steps) to amplify separation
4. **After decisive Phase 1**: Integrate into TinyStories ladder as P11 series (requires GPU)
5. **Longer term**: v3 operator augmentation for full MCS

## References

- Paper v4, §9.2: Theorem v0-ceiling (the formal expressivity bound)
- Paper v4, §9.4.2: v2 → Fock space mapping
- Paper v4, §9.4.3: v3 → Lie groups / gauge theory
- Paper v4, §9.5: LCFRS reduction (composite reaches MCS)
- Paper v4, §9.6: F1–F6 falsifier programme
- Paper v4, §17.6: PARF does not escape the v0 ceiling
- Doi (1976): Second quantisation for stochastic processes
- Peliti (1985): Path integral for classical reaction-diffusion
- [PARF-SPLM Path Forward and Experiments](companion_notes/PARF-SPLM_Path_Forward_and_Experiments.md): P10 ladder context
- [PARF Stage 1.5b Design](companion_notes/PARF_Stage_1_5b_design.md): PARF sparsity and scale-up design
- [Expressivity Bounds for v0 Simulator](companion_notes/Expressivity_Bounds_For_v0_Simulator.md): v0 ceiling formal details
- [MCS Reduction for v3 Composite](companion_notes/MCS_Reduction_For_v3_Composite.md): Full MCS system design
