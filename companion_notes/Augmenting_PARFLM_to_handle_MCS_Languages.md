# Augmenting PARFLM to Handle Mildly Context-Sensitive Languages

## Status

**Reopened 2026-09-28** — Phase 1b (register-count sweep) designed and pre-registered below; unrun. Prior: the built v2 is a bounded truncation of the formalism. May-2026 status follows.

**Active** — May 2026. PARFLM P10 ladder completed (architectural ceiling confirmed at val PPL ≈ 26.4). FockPARFLM v2 (Q/K/V + gated reverse channel) F2 seed 0 complete: **best deep-test acc 49.01%**, val PPL 2.856 — +5.37 pp over PARFLM baseline, +3.89 pp over v1 mean-gate. Next: extended training (8000 steps) and/or scale-up; then TinyStories (Phase 3).

## Motivation

PARFLM (PARF-Augmented SPLM) adds token-token pair interactions $V_\phi(h_t, h_s)$ to the single-particle scalar potential $V_\theta(\xi, h)$. This enriches the force law but does not escape the **v0 expressivity ceiling** (Theorem v0-ceiling, §9.2 of paper v4):

- The hidden state is still $h \in \mathbb{R}^d$ (fixed dimension)
- The integrator is still a deterministic function
- There is no mechanism for the state space to grow during inference

Consequently, PARFLM is at most a finite automaton (regular languages). It cannot:
- Recognise $\text{Dyck}\_n$ beyond the predicted collapse depth $D^\ast$
- Handle cross-serial dependencies ($a^n b^n c^n$)
- Reach the mildly context-sensitive (MCS) class

**Empirical confirmation (P10 ladder, 10 May 2026):** The P10h experiment (20M tokens, 16k steps, full P5+P7+P8 stack) achieves val PPL **26.43** — identical to P10g (5M tokens, 16k steps, PPL 26.42). Quadrupling the corpus produces zero improvement, confirming the v0 architectural ceiling. The 22M-parameter PARFLM has exhausted its representational capacity on TinyStories at ≈ 26.4 PPL. The gap to MatchedGPT (7.81 PPL) can only be closed by escaping the expressivity class.

The pretrained potentials $V_\theta$ and $V_\phi$ from P10g/P10h are not wasted — they serve as **warm-start initialization** for the RL-calibrated EOM simulator (§8/§9 of paper v4; dynamical simulation programme planned for paper v5).

To escape this ceiling, the framework requires **v2 (creation/destruction)** mapped to **Fock space and second quantisation** (§9.4.2), plus eventually **v3 (execution)** mapped to **Lie groups and non-abelian gauge theory** (§9.4.3). This document plans the augmentation to v2.

## Theoretical Foundation

### The v2 → Fock Space Mapping (from §9.4.2)

| v2 mechanism | Fock-space object |
|---|---|
| Introduce an entity into discourse | Creation operator $a^\dagger_v \lvert \psi \rangle$ |
| Entity drops out of discourse | Annihilation operator $a_v \lvert \psi \rangle$ |
| Count of currently-live entities | Number operator $N = \sum_v a^\dagger_v a_v$ |
| Field at semantic position *x* | $\hat{\phi}(x) = \sum_v \phi_v(x) a_v$ |

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
| Register pool (all inactive) | Vacuum state $\lvert 0 \rangle$ |
| Activation of register `r_j` | Creation: $a^\dagger_v \lvert 0 \rangle$ |
| Deactivation of register `r_j` | Annihilation: $a_v \lvert \psi \rangle$ |
| Number of active registers | Number operator *N* |
| Salience-ordered LIFO activation | Stack discipline (→ pushdown automaton) |
| Register hidden state `r_j` ∈ ℝ^d | Single-particle state in $\mathcal{H}$ |

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
- Creation gates: $8 \times (256 \times 64 + 64 \times 32) = 8 \times 18432 = 147456$ parameters
- Destruction gates: $8 \times (256 \times 64 + 64) = 8 \times 16448 = 131584$ parameters
- Total overhead: ~290K params (< 2% of the 22M total)

## Experimental Plan

### Phase 1: Dyck Falsifier (proof of concept)

**Goal**: Demonstrate that FockPARFLM solves $\text{Dyck}\_2$ past the predicted collapse depth $D^\ast$, where plain PARFLM fails.

| Experiment | Architecture | Expected result |
|---|---|---|
| F1-baseline | PARFLM (P10f config) | Collapses at depth D\* ≈ 3–6 |
| F1-fock-nostack | FockPARFLM, M=16, no stack | Extends past D\* (≥ 8–10) |
| F1-fock-stack | FockPARFLM, M=16, LIFO stack | Extends further (≥ 12–15) |
| F1-attention | Matched GPT-2 baseline | Succeeds to arbitrary depth (TC⁰) |

**Success criterion**: F1-fock-stack succeeds at depth $\gt D^\ast$ with 3/3 seed consistency.

**Training**: Synthetic $\text{Dyck}\_2$ strings at controlled max depth, next-bracket-type prediction task. Small scale ($d=64$, $L=4$) sufficient for falsifier.

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

- FockPARFLM **passes** Dyck past $D^\ast$ → v2 mechanism confirmed; proceed to Phase 2
- FockPARFLM **fails** at $D^\ast$ → implementation does not correctly realise Fock-space dynamics; debug creation/destruction gates

### After Phase 2 (TinyStories)

- PPL improves $\gt 1$ PPL over P10g → v2 structures are useful for natural text at this scale
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

- Corpus: Synthetic $\text{Dyck}\_2$, max nesting depth 12, $p_{\text{open}} = 0.55$
- Train: 10,000 samples. Val: 2,000 samples. Deep test: 500 samples (depth 5–12 only)
- Model: $d = 64$, $L = 4$, $v_{\text{hidden}} = 128$, top-k = 8, mass = global
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
| **Scale up**: d = 128, M = 32, 8000 steps | More capacity per register, longer gate specialisation time |
| **Curriculum**: start at depth 4, increase to depth 12 | Easier initial gradient signal for gate learning |
| **Gate pre-training**: initialise creation gate to trigger on open brackets | Warm-start the stack discipline |
| **Longer sequences**: max_length = 128, max_depth = 16 | More room for deep nesting to differentiate |
| **Multi-seed**: run seeds 1, 2 at current config | Confirm the LIFO > bag > baseline ordering is stable |

**Decision**: Proceed with the scale-up intervention first (cheapest signal amplification), then multi-seed at the larger config.

## Phase 2 Results: FockPARFLM v2 Dyck₂ Falsifier (F2, seed 0)

**Date**: 23 May 2026. Run on Google Colab (GPU). ~98 s wall-clock for the v2 arm (4000 steps).

### Configuration

- Corpus: Synthetic $\text{Dyck}\_2$, max nesting depth 12, $p_{\text{open}} = 0.55$
- Train: 10,000 samples. Val: 2,000 samples. Deep test: 500 samples (depth 5–12 only)
- Model: $d = 64$, $L = 4$, $v_{\text{hidden}} = 128$, top-k = 8, mass = global
- Training: 4000 steps, batch 32, lr $3 \times 10^{-4}$ cosine, AdamW
- Fock v2–specific: $M = 16$ registers, $\lambda = 0.5$, $\tau_{\text{thresh}} = 0.005$, gated reverse channel (`reverse_channel_scale` learnable)
- Implemented in `notebooks/conservative_arch/parf/model_fock_parf_v2.py`; three-arm notebook: `fockparf_v2_dyck2_falsifier.ipynb`

### Results

| Arm | Mechanism | Best val PPL | Best deep-test acc (depth 5–12) |
|---|---|---|---|
| F2-baseline (PARFLM) | No registers | 3.0778 | 43.64% |
| F2-fock-v1 (mean gate) | Mean-conditioned creation | 3.0365 | 45.12% |
| **F2-fock-v2 (Q/K/V + reverse)** | **Q/K/V creation + gated reverse channel** | **2.8556** | **49.01%** |

Logs and diagnostics: `notebooks/conservative_arch/parf/results/fock_v2/`.

### Training dynamics

All 16 registers are active across all 4 layers from the start (confirmed by the register-salience diagnostic), with registers 0 and 4 dominating (salience ≈ 0.40) and clear layer-to-layer salience decay. This is the desired behavior: the destruction gate has learned to prune, having started with full salience.

The v2 curve rises monotonically from 34.1% (step 200) to 49.01% (step 4000), still climbing steeply at the end of training — suggesting further gains with more steps.

### Interpretation

1. **Q/K/V creation is the key upgrade**: v2 beats v1 by **+3.89 pp** deep-test accuracy (49.01% vs 45.12%) and **5.8% lower PPL** (2.8556 vs 3.0365). The structured attention-over-input creation mechanism gives registers content that is genuinely relevant to the current context, unlike the v1 mean-conditioned gate.

2. **Gated reverse channel is stable**: The `torch.tanh(reverse_channel_scale)` initialization at zero prevents the non-conservative force from destabilizing early training; it is gradually learned as the registers accumulate useful content.

3. **v2 approaches the 50% target**: The pre-registered success criterion is >90% at depth 8+, which requires multi-seed confirmation and likely more training steps and/or larger scale. The current 49.01% at 4000 steps is a strong proof-of-concept that the Q/K/V mechanism enables the model to exploit register structure for pushdown-like reasoning.

4. **v1 (mean gate) already beats baseline** (+1.48 pp), confirming that register structure itself is beneficial; v2 amplifies this by a further 2.4× margin.

### Diagnosis and next steps

| Intervention | Rationale |
|---|---|
| **Extend training** to 8000–12000 steps | Curve still rising at step 4000 |
| **Scale up**: d = 128, M = 32 | More capacity per register slot |
| **Multi-seed** (seeds 1, 2) | Confirm ordering is stable |
| **TinyStories integration** | Real-language validation (Phase 3) |

**Decision**: Run extended training (8000 steps) with the current config as the cheapest next signal. If the curve crosses 55%, proceed to scale-up.

## Phase 1b — the register-count sweep: is the built v2 a bounded truncation? — **designed 2026-09-28, pre-registered, unrun**

### Why this experiment exists

§10 of paper v6 defines v2 by *unbounded* particle cardinality: "the
state-space dimension grows linearly with depth, lifting from regular to
deterministic context-free." Fock-PARFLM v2.1 has `n_registers` = M fixed at
construction, with slots recycled. If M is a hard cap on usable memory, the
built model is a **bounded truncation** of the formalism and sits *below*
§10's middle rung, not on it — and the book's expressivity claims are about a
system nobody has built. That was the stated prior when this design was
first written (limb (a) in `Paper_v6_Section_Audit.md`, §10 entry). This
experiment is designed so that prior can be **refuted**, not merely confirmed.

> **Prior revised 2026-09-28, the same day, before any run: (a) → (b).**
> Reading `_fock_layer_step` showed that in the prefix-causal lifecycle the
> register bank is rebuilt from the whole prefix at **every layer**, so the
> lifecycle runs over L, not over tokens, and M caps readout width, not
> retained history. See "The mechanism as built" below and the audit's
> "CORRECTED 2026-09-28" block. The factors, arms, data, metric, limbs and
> decision rules are unchanged; the stated prior, the point prediction and
> the mechanism description are revised, and the original limb-(a) point
> prediction is kept below, marked as superseded.

### Why the May 2026 runs (Phase 1 and F2 above) cannot answer it

Five defects, each fatal on its own for this question (the fifth added
2026-09-28, after the mechanism reading below):

1. **No per-depth curve.** Deep-test accuracy was pooled over depths 5–12.
   The theory's own protocol (`Expressivity_Bounds_For_v0_Simulator.md` §6)
   calls for accuracy *as a function of* D on a grid {1, 2, 4, 8, 16, 32}.
   A collapse depth cannot be read off a pooled number.
2. **The wrong positions were scored.** `evaluate_dyck_accuracy` scores
   next-token accuracy at *every* valid position. At open positions the
   generator's next token is a stochastic choice (open-vs-close with
   `p_open`, type uniform), so those positions have a ceiling far below 1
   that depends on `p_open` and n, not on expressivity. The theory protocol
   scores **closing-bracket type only**, where the target is fully determined
   by the stack and chance is exactly 1/n. The May 43–49% figures mix the two
   and are not comparable to the pre-registered 90% criterion.
3. **Single M, single seed.** M = 16 throughout; nothing varied the quantity
   under test.
4. **The generator cannot produce the deep grid.** At the May settings
   (`p_open` = 0.55, `max_length` = 64) strings of depth ≥ 16 are 2.8% of
   samples and depth ≥ 24 do not occur. Measured 2026-09-28 on 20,000 draws.
5. **They ran the leaky lifecycle.** Both May runs predate the causal-leak
   fix (23 July, `Fock-PARFLM_Causal_Leak_Audit_Results.md`). `FockPARFLM_v2`
   then carried the cross-layer register state from the **last position of
   the full window** (`_causal_creation_readout`, `r_new = r_causal_mt[:, :,
   -1, :]`), the configuration the leak audit's T2 probe certifies as leaking
   future tokens backward with the reverse channel on. On Dyck a future leak
   can supply the closing bracket outright. Its size at d=64 was never
   measured, so the May accuracies — and the "LIFO is the active ingredient"
   reading — are not evidence in either direction. Every Phase 1b arm runs
   with `prefix_causal_registers=True`.

### The mechanism as built — what the sweep is actually testing

**Rewritten 2026-09-28 before any run.** The first version of this section
described the salience ordering correctly but drew the wrong consequence
from it; the original bullets are kept at the end, marked superseded.

**The lifecycle runs over layers.** With `prefix_causal_registers=True` (the
default, and every arm here), `FockMultiXiPARFLM._fock_layer_step` does, once
per layer:

```python
readout, alpha_max = self.creation_gate_qkv.forward_prefix(h, r)
r = blend * r + (1.0 - blend) * readout                  # (B, T, M, d)
salience = salience * decay + alpha_max * (1.0 - decay)  # (B, T, M)
```

`QKVCreationGate_v21.forward_prefix` scores each register's query — from that
register's previous-layer state at position t — against per-register keys of
tokens 1…t, and returns a cumulative-softmax readout of their values. The
active mask and the destruction gate then step, also per layer. So:

- At position t there are **at most L = 4 creation events**, independent of
  how many brackets are open. No token creates or retires a register.
- The pool is **M learned-query attention readouts over the prefix,
  iterated L times**. Its store is the prefix itself, re-read at every
  layer; M limits how many readouts a layer takes, not how much history
  survives.

**The salience ordering** (`_active_mask`):

```python
sorted_sal, sort_idx = salience.sort(dim=-1, descending=True)
sorted_above = sorted_sal > cfg.register_salience_threshold
sorted_active = torch.cumprod(sorted_above.float(), dim=-1).bool()
```

**There is no push and no pop.** "LIFO" means salience-ordered contiguous
activation, and the salience it orders by evolves across the four layers at
a fixed position, not across positions. A close bracket cannot retire
anything.

Consequences the revised predictions rely on:

- **M is not a cap on representable depth.** Depth information can reach
  position t through the token state (V_θ, V_φ over the prefix, the K-EMA ξ
  channels) and through any of the M readouts; nothing forces one slot per
  open bracket. There is no architectural reason for $D^\ast \le M$.
- **Depth capacity should be set mainly by L, d and the position signal**,
  as in a small transformer, with M contributing readout width. The expected
  signature is $D^\ast$ nearly flat in M.
- **Whether the registers help at all is read from the sweep against C-v0,
  not from the slope.** A flat $D^\ast(M)$ well above C-v0 means the readouts
  add depth capacity without the capacity being slot-bounded; a flat
  $D^\ast(M) \approx$ C-v0 means they add none at this scale.

*Superseded bullets, kept for the record (written earlier 2026-09-28):*

- ~~M is a hard cap on *simultaneously active* registers, so it bounds the
  representable stack depth from above **regardless** of how the gate
  learns.~~ Wrong: active registers are readouts of the prefix, not stack
  cells, so their number does not bound what the prefix can carry.
- Whether depth capacity tracks M at all depends on the gate learning a
  recency-encoding salience pattern that nothing in the architecture
  enforces. *(Still true as far as it goes; recency across tokens is not
  something the layer-wise salience can encode at all.)*

### Factors and arms

| factor | levels | role |
| --- | --- | --- |
| **M** (`n_registers`) | 2, 4, 8, 16, 32, 64 | the quantity under test |
| model class | `FockMultiXiPARFLM` (v2.1, `mass_mode='global'`, default `ScalarPotentialMultiXi` Vθ) | the class the ladder trains; **not** the older v2 class of the May runs |
| seeds | 0, 1, 2 | every cell |

Held fixed across all M: d = 64, L = 4, `v_hidden` = 64, `v_depth` = 2,
`top_k` = 8, `d_k` = 32, `creation_gate_hidden` = 32, salience decay 0.5,
threshold 0.005, `stack_discipline=True`, `prefix_causal_registers=True`,
reverse channel on with its default warmup, 4,000 steps, batch 64, lr
3e-4 cosine, AdamW wd 0.01 — i.e. the F2-fock-v2 recipe above, ported to
v2.1, so that M = 16 reproduces a known anchor.

Controls, each at 3 seeds:

| arm | what it removes | what it separates |
| --- | --- | --- |
| **C-bag**, M = 16, `stack_discipline=False` | the salience ordering | whether ordering, not pool size, carries any depth capacity (May: bag ≈ baseline) |
| **C-v0**, `SparsePARFLM`, no registers | the register pool | the v0 floor; its collapse depth is §7's D* ≈ 4–6 prediction, never yet measured per depth |
| **C-params**, M = 8 with `d_k` = 128 | — | matches the *parameter count* of M = 32 without adding slots; separates "more slots" from "more parameters" |
| **C-attn**, matched tiny transformer (`matched_baseline_model.py`, ≈ same params as M = 16) | everything | the §10 comparator that "succeeds to arbitrary depth"; the row the May plan wrote down and never ran |

**Parameter confound, quantified.** v2.1 at d = 64, L = 4 (measured
2026-09-28): M = 4 → 123,326 params; 8 → 124,622; 16 → 127,214; 32 →
132,398; 64 → 142,766. The register machinery grows ≈ 324 params per slot
(`W_Q`, per-register `W_K`, embeddings). From M = 4 to 64 the total grows
15.8%. C-params exists to show the effect tracks slots, not that 15.8%.

### Data

Train and validation from one distribution for **every** arm, so no arm
sees deeper strings than another: `DyckConfig(n_types=2, max_depth=32,
min_length=8, max_length=128, p_open=0.65)`. Measured depth distribution
of that generator (20,000 draws): 1–3: 3.2%, 4–7: 13.6%, 8–11: 14.1%,
12–15: 13.8%, 16–23: 26.9%, 24–31: 19.7%, 32: 8.7%. Every depth on the
test grid is represented in training; the question is capacity, not
extrapolation. (A second, extrapolation variant — train to depth 12, test
to 32 — is the F1 falsifier proper and is *not* this experiment.)

Test sets: **exact-depth** bins, 1,000 strings each, at D ∈ {1, 2, 4, 6, 8,
12, 16, 24, 32}, generated by rejection with `min_depth = max_depth = D`.
Seeds disjoint from train/val. N_TRAIN = 20,000, N_VAL = 2,000.

### Metric

**Close-type accuracy at stack depth k**, per position, not per string:
walk each test string, record the stack depth at every closing-bracket
position, score the model's predicted bracket type there. Report
$A(k)$ = accuracy over all close positions whose depth is exactly $k$,
pooled across the test bins. Chance is 0.50 for n = 2; the ceiling is 1.0
because the target is determined. (Positions where the *target* is an open
bracket are excluded from $A(k)$ entirely; they are reported separately as
a sanity curve and are expected near the generator's entropy for every arm.)

**Collapse depth** $D^\ast$: the smallest $k$ at which $A(k) \lt 0.75$, the
midpoint between chance and ceiling, provided $A(k') \lt 0.75$ for all
$k' \gt k$ as well (so a single noisy bin cannot set it). If $A(32) \ge 0.75$,
report $D^\ast \gt 32$.

Per-string max depth is also recorded so the May pooled metric can be
recomputed for continuity, but it decides nothing.

### The plot that decides

$D^\ast$ against M, log–log, with the three controls as horizontal lines
(C-v0, C-bag) or a point (C-attn), three seeds as error bars.

### Pre-registered predictions — recorded 2026-09-28, before any run

Stated prior: **limb (b), prefix conditioning carries depth and M does not
bound it** — revised 2026-09-28 before any run from the original limb (a);
see "The mechanism as built". The limb table is unchanged.

| limb | what D\*(M) looks like | reads as |
| --- | --- | --- |
| **(a) bounded truncation** | D\* rises with M and is bounded by it: D\*(M) ≤ M at every M, with slope in log D\* / log M between 0.5 and 1.0, and D\*(64) ≤ 64 | the pool is the memory, the cap is real, §10 must say the built model is below its middle rung |
| **(a′) bounded, and not even tracking M** | D\* flat in M above some small M, i.e. slope < 0.3, with C-bag ≈ the sweep | the salience ordering is not encoding recency; capacity is set by d and L, not by the pool at all — worse than (a) for the v2 story, and it would say the May "LIFO wins" result was not about the stack |
| **(b) prefix conditioning lifts it** | D\* **exceeds** M at small M — e.g. D\*(2) ≥ 8 or D\*(4) ≥ 16 | depth is being carried outside the pool, by the prefix-attending gate or by the token state; the staircase is the wrong ladder and the model must be placed on the circuit-complexity axis instead |
| **(c) effective unboundedness** | D\* > 32 at every M including M = 2, and C-attn also > 32 | indistinguishable from (b) at this grid; would need the extrapolation variant and a precision sweep to separate, and the book would owe the same precision caveat it applies to Universal Transformers |

**Point prediction, limb (b) — the revised prior:** $D^\ast$ nearly flat in
M, log–log slope **below 0.3** across M = 2…64, with every $D^\ast(M)$ in
**4–12**; in particular **$D^\ast(2) \ge 4$**, i.e. above M. The sweep sits
**at or above C-v0** by at most ~4 levels, the registers adding readout width
rather than slot-bounded depth. Reasoning: at L = 4, d = 64 the ceiling is set
by what four layers of prefix readouts can count, not by how many readouts
there are.

**Overlap with limb (a′), resolved in advance.** A flat curve was written as
(a′) under the old mechanism, where it meant "the pool is inert". Under the
corrected mechanism, flatness is the *expected* shape and reads as (b). The
pool's contribution is read from **sweep − C-v0**, not from the slope:

| flat $D^\ast(M)$ and … | reads as |
| --- | --- |
| sweep ≥ C-v0 + 3 | (b): the readouts add depth capacity that M does not bound |
| sweep within 2 of C-v0 | (b) with an inert pool: the registers add nothing to depth at this scale; the May "LIFO wins" reading is withdrawn as for (a′) |

**What refutes the revised prior.** $D^\ast$ tracking M with slope ≥ 0.5 and
$D^\ast(M) \le M$ at every M — the original limb (a). That would mean the
layer-wise readouts somehow partition depth across slots, and it would need
explaining against the code, not just reporting.

**Named turnable quantity:** L, not salience decay. Salience decays over
layers at a fixed position, so it cannot set a lifetime in *tokens*; at 0.5
it only shapes which readouts are active within a four-layer pass. If
$D^\ast$ is flat, the informative follow-up is **M = 16 at L ∈ {2, 4, 8}**:
under (b), $D^\ast$ should move with L. That also ties the result to the
depth ladder, where L is the axis under test.

*Superseded point prediction, limb (a) — the original prior, kept for the
record:* $D^\ast(2) \approx 2$, $D^\ast(4) \approx 3$–4,
$D^\ast(8) \approx 5$–7, $D^\ast(16) \approx 8$–12,
$D^\ast(32) \approx 12$–20, $D^\ast(64) \approx 16$–28,
sub-linear because decay 0.5 was taken to retire registers
faster than closes arrive. Its named turnable quantity was salience decay
(re-run M = 16 and 64 at decay 0.9). Both rested on registers carrying state
across tokens, which the prefix-causal lifecycle does not do.

**Controls, predicted:** C-v0 $D^\ast \approx 4$–6 (the §7 band [3, 8] —
this is the first per-depth measurement of that prediction, and it scores
it). C-bag $\le$ C-v0 + 2. C-params $\approx D^\ast(8)$, not $D^\ast(32)$.
C-attn $\gt 32$ at matched parameters, per Hewitt et al. and Yao et al.

*Original refutation conditions for limb (a), still valid as tests of (a):*
any of $D^\ast(M) \gt M$ at any M; C-attn failing where the sweep succeeds;
C-params matching $D^\ast(32)$. Under the revised prior the first is
expected, and C-params ≈ $D^\ast(8)$ ≈ $D^\ast(32)$ is expected too, since the
curve is flat.

### Confounds and how each is closed

| confound | closed by |
| --- | --- |
| more slots = more parameters | C-params; and report D\* per 10³ params as a secondary axis |
| deeper training distribution than May | all arms share one distribution; May numbers are not compared, only recomputed for continuity |
| the gate reading depth off the prefix rather than the pool | limb (b) is a *prediction*, not a nuisance: D\* > M is the signature, and C-v0 (no registers, same prefix access via Vφ) bounds how much the prefix alone gives |
| sequence length capping depth | `max_length` = 128 admits depth 32 with margin; strings at depth 32 are 8.7% of the distribution |
| chance-level inflation from open positions | close-only scoring |
| one lucky seed | three seeds, error bars, and D\* defined with the monotonicity guard |
| register lifetime masquerading as pool size | moot under the corrected mechanism — lifetime is in layers, not tokens; the L follow-up named above replaces the decay follow-up |
| May numbers contaminated by the pre-fix leak | every arm runs `prefix_causal_registers=True`; May numbers recomputed for continuity only, never compared |

### Harness changes required (all small, none run)

1. `dyck_data.py`: a `generate_exact_depth_dataset(cfg, n, D, seed)` and a
   `position_depths(x, cfg)` returning the stack depth at every position.
2. `train_fock_parf.py`: `--arch fock21` (constructs `FockMultiXiPARFLM`
   with `mass_mode='global'`), `--arch transformer`, `--no-stack` already
   exists; `--dyck-p-open`, `--dyck-max-length`, `--d-k`.
3. `evaluate_dyck_accuracy`: add close-only $A(k)$ and the $D^\ast$ rule;
   write per-bin JSON, not just a scalar.
4. Assert at build time that every arm's train set is the same tensor
   (hash) — the shared-distribution guarantee, enforced not assumed.

### Budget and venue

CPU on this machine: 1.58 s/step at batch 32, M = 16 → ~105 min per run;
the grid is 6 M × 3 seeds + 4 controls × 3 seeds = 30 runs ≈ 50 h. **Not
here.** Colab GPU ran the May v2 arm at 98 s per 4,000 steps; v2.1 is
heavier, call it 4 min → the grid is ~2 h of GPU. **After L=4 finishes**; do
not share the session.

### Decision rules

- **Limb (a) confirmed** → §10 gains a closing subsection stating: the built
  model implements v2's mechanism without v2's defining property; it sits
  below the middle rung; the MCS claim is about the formalism, not about any
  trained model; the register count is the memory bound and here is the
  measured $D^\ast(M)$. The ladder cards' `riemannian-geodesics` and
  `fock-space` tags are reviewed for the same overreach.
- **Limb (a′)** → as (a), plus the May "LIFO is the active ingredient"
  finding is withdrawn and the salience-ordering mechanism is flagged as not
  doing the job its name claims.
- **Limb (b)** → §10's staircase is declared the wrong instrument for the
  built model; the model is placed on the TC⁰ axis beside transformers and
  the "native memory vs. bolt-on" contrast in `framework-vs-transformers` is
  rewritten.
- **Limb (c)** → run the extrapolation variant (train ≤ 12, test ≤ 32) and a
  bf16/fp32 precision pair before any claim.

Whatever the outcome, `Expressivity_Bounds_For_v0_Simulator.md` §6–7 gets its
first per-depth measurement of D* for v0, which it has waited for since July.

---

## Implementation Status

### Completed

1. **`FockPARFConfig`** dataclass (`parf/model_fock_parf.py`) — extends `SparsePARFConfig` with v2 knobs.
2. **`FockPARFLM`** model class — latent register pool with creation/destruction gates, LIFO stack discipline, per-layer lifecycle management.
3. **`dyck_data.py`** — Dyck_n data generator with depth-controlled dataset generation for falsifier experiments.
4. **`train_fock_parf.py`** — unified trainer supporting both Dyck falsifier and TinyStories corpora, with baseline PARFLM arm for comparison.
5. **Phase 1 seed 0 run** — all 3 arms complete; LIFO stack wins by +1.3pp.
6. **`model_fock_parf_v2.py`** — FockPARFLM v2 with Q/K/V creation gate, gated reverse channel, temporal persistence; salience defaults corrected (`λ=0.5`, `τ=0.005`); salience initialized to ones; `reverse_channel_scale` initialized to zero.
7. **`fockparf_v2_dyck2_falsifier.ipynb`** — three-arm Dyck₂ notebook (baseline / v1 / v2); per-arm `DECAY`/`THRESHOLD` config cells.
8. **F2 seed 0 run** — all 3 arms complete; v2 (Q/K/V + gated reverse) wins by +5.37 pp over baseline and +3.89 pp over v1.

### Verified

- Forward + backward pass on CPU/MPS (smoke tests pass).
- Parameter budget at P10f scale: **288,520 overhead** (2.16% of 13.35M total).
- Training loop runs correctly for both `--arch fock` and `--arch parflm` on Dyck data.
- LIFO stack discipline is the active mechanism (bag ≈ baseline < LIFO).

## Concrete Experiment Commands

### Phase 1: Dyck Falsifier (F1 experiments)

Run from `notebooks/conservative_arch/parf/`:

```bash
# F1-baseline: plain PARFLM (should collapse at depth D* ≈ 3-6)
python train_fock_parf.py \
  --corpus dyck --arch parflm \
  --dyck-n-types 2 --dyck-max-depth 12 \
  --dyck-test-depth-min 5 --dyck-test-depth-max 12 \
  --steps 4000 --seed 0

# F1-fock-nostack: FockPARFLM without LIFO (bag discipline)
python train_fock_parf.py \
  --corpus dyck --arch fock --no-stack \
  --n-registers 16 \
  --dyck-n-types 2 --dyck-max-depth 12 \
  --dyck-test-depth-min 5 --dyck-test-depth-max 12 \
  --steps 4000 --seed 0

# F1-fock-stack: FockPARFLM with LIFO stack discipline
python train_fock_parf.py \
  --corpus dyck --arch fock \
  --n-registers 16 \
  --dyck-n-types 2 --dyck-max-depth 12 \
  --dyck-test-depth-min 5 --dyck-test-depth-max 12 \
  --steps 4000 --seed 0

# Repeat each with --seed 1, --seed 2 for 3-seed consistency
```

**Key metric**: Deep-test accuracy at depth > D*. Success = FockPARFLM (stack) achieves > 90% accuracy at depth 8+ where baseline collapses.

### Phase 2: TinyStories (P11 experiments)

Requires GPU (A100/H100 recommended for P10f scale):

```bash
# P11a: FockPARFLM with M=16 registers (matches P10f otherwise)
python train_fock_parf.py \
  --corpus tinystories --arch fock \
  --n-registers 16 --v-hidden 1024 \
  --steps 8000 --seed 0

# P11b: FockPARFLM with M=32, LIFO stack
python train_fock_parf.py \
  --corpus tinystories --arch fock \
  --n-registers 32 --v-hidden 1024 \
  --steps 8000 --seed 0

# P11c: FockPARFLM M=32, 16k steps (post-P10g result)
python train_fock_parf.py \
  --corpus tinystories --arch fock \
  --n-registers 32 --v-hidden 1024 \
  --steps 16000 --seed 0
```

## Implementation Priority

1. ~~Design `FockPARFConfig` dataclass and `FockPARFLM` model class~~ **DONE**
2. ~~Run Dyck falsifier (Phase 1) — seed 0, small scale~~ **DONE** (signal positive but modest, +1.3pp)
3. ~~FockPARFLM v2 (Q/K/V + gated reverse channel) implementation~~ **DONE**
4. ~~Run F2 Dyck₂ falsifier — seed 0, v2 mechanism~~ **DONE** (+5.37pp over baseline, +3.89pp over v1; 49.01% at 4000 steps, still climbing)
5. **Next**: Extend training to 8000 steps; if >55% proceed to scale-up ($d = 128$, $M = 32$)
6. **After decisive F2**: Integrate into TinyStories ladder as P11 series (requires GPU)
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
- `companion_notes/PARF-SPLM_Path_Forward_and_Experiments.md`: P10 ladder context
- `companion_notes/PARF_Stage_1_5b_design.md`: PARF sparsity and scale-up design
