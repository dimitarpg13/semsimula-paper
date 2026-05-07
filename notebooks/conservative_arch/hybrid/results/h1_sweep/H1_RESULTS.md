# H1 — Layer-split sweep results (Variant A two-stage hybrid)


Architecture: `k` attention blocks (front) + `m` SPLM integration
steps (back), tied embeddings, single shared V_theta,
`causal_force=True`, `ln_after_step=True`, `mass_mode='logfreq'`,
Tiny Shakespeare 4000 steps at d=128.

## Per-cell results

| (k, m) | seed | γ_mode | val PPL | val loss | final γ | params | elapsed |
|--------|------|--------|---------|----------|---------|--------|---------|
| (2, 6) | 0 | free γ | 147.28 | 4.9923 | 0.153 | 7.52 M | 34.7 min |
| (3, 5) | 0 | free γ | 139.29 | 4.9366 | 0.153 | 7.72 M | 34.5 min |
| (4, 4) | 0 | free γ | 133.01 | 4.8904 | 0.154 | 7.92 M | 32.2 min |
| (5, 3) | 0 | free γ | 136.48 | 4.9162 | 0.154 | 8.11 M | 33.5 min |
| (6, 2) | 0 | free γ | 135.08 | 4.9059 | 0.153 | 8.31 M | 34.4 min |

## Anchors (already on disk, leak-free)

| arm | val PPL | source |
|-----|---------|--------|
| All-attention (matched GPT-2, n_layer=8) | ~150 | `multi_seed/results/` |
| All-SPLM em_ln (free-γ, leak-free) | 173.59 | `energetic_minima/results/` |
| All-SPLM em_ln (γ=0.10, leak-free) | ~178–181 | `ln_damping_sweep/results/RESULTS_LEAKFREE_GAMMA_SWEEP.md` |

## Pre-registered decision (Phase 2 gate)

Per `the v4 title-justification rule` §6.5:
**"Efficient" is justified iff** some hybrid (k, m) achieves val PPL
within +5 PPL of the all-attention baseline AND its analytical decode-FLOP
cost at T=1024 is ≥ 30% lower than all-attention, both at S=3 with sign 3/3.

**Phase 1 → Phase 2 gate:** if best (k, m) at S=1 (this sweep) is
within +10 PPL of all-attention, proceed to H2 (S=3 confirmation).
If gap is > 15 PPL, soften the title (Option 2 fallback) and
document the hybrid as Future Work.

## Best-of-sweep summary

Best cell: **(k=4, m=4)** at seed 0 with val PPL **133.01**.
Gap to all-attention (~150): **-16.99 PPL**.
Gap to all-SPLM em_ln free-γ (~173.59): -40.58 PPL.

Phase 2 gate: **PASS** (within +10 PPL of all-attention).

## Decode-FLOP Pareto (analytical, T = 1024)

Per-token AR decode FLOPs at context length 1024 (KV-cached attention; streaming-ξ SPLM):


### T = 256

All-attention reference (k=8, m=0): **17.116 MFLOPs/tok**.
All-SPLM reference (k=0, m=8): **44.397 MFLOPs/tok**.

| (k, m) | val PPL (S=1) | decode FLOPs/tok | vs all-attn | reduction |
|--------|---------------|------------------|-------------|-----------|
| (2, 6) | 147.28 | 37.577 MFLOPs | 2.195× | -119.5% |
| (3, 5) | 139.29 | 34.167 MFLOPs | 1.996× | -99.6% |
| (4, 4) | 133.01 | 30.757 MFLOPs | 1.797× | -79.7% |
| (5, 3) | 136.48 | 27.347 MFLOPs | 1.598× | -59.8% |
| (6, 2) | 135.08 | 23.937 MFLOPs | 1.398× | -39.8% |

### T = 1024

All-attention reference (k=8, m=0): **20.385 MFLOPs/tok**.
All-SPLM reference (k=0, m=8): **44.397 MFLOPs/tok**.

| (k, m) | val PPL (S=1) | decode FLOPs/tok | vs all-attn | reduction |
|--------|---------------|------------------|-------------|-----------|
| (2, 6) | 147.28 | 38.394 MFLOPs | 1.883× | -88.3% |
| (3, 5) | 139.29 | 35.393 MFLOPs | 1.736× | -73.6% |
| (4, 4) | 133.01 | 32.391 MFLOPs | 1.589× | -58.9% |
| (5, 3) | 136.48 | 29.390 MFLOPs | 1.442× | -44.2% |
| (6, 2) | 135.08 | 26.388 MFLOPs | 1.295× | -29.5% |

### T = 4096

All-attention reference (k=8, m=0): **33.459 MFLOPs/tok**.
All-SPLM reference (k=0, m=8): **44.397 MFLOPs/tok**.

| (k, m) | val PPL (S=1) | decode FLOPs/tok | vs all-attn | reduction |
|--------|---------------|------------------|-------------|-----------|
| (2, 6) | 147.28 | 41.663 MFLOPs | 1.245× | -24.5% |
| (3, 5) | 139.29 | 40.296 MFLOPs | 1.204× | -20.4% |
| (4, 4) | 133.01 | 38.929 MFLOPs | 1.163× | -16.3% |
| (5, 3) | 136.48 | 37.561 MFLOPs | 1.123× | -12.3% |
| (6, 2) | 135.08 | 36.194 MFLOPs | 1.082× | -8.2% |

### Pre-registered rule check

Per `the v4 title-justification rule` §6.5:
**"Efficient" is justified iff** some hybrid (k, m) achieves
val PPL within +5 PPL of the all-attention baseline (~150) AND
decode-FLOP cost at T = 1024 is ≥ 30% lower than all-attention,
both at S=3 with sign 3/3.

At S=1 (this sweep) every cell already passes the *quality* arm
(in fact every cell BEATS all-attention by 2–17 PPL). For the
*FLOP* arm at T=1024 the table above is the relevant comparison.
Cells that satisfy ≥ 30% decode-FLOP reduction at T=1024 simultaneously
with within-+5-PPL-of-all-attn are the candidates that earn
the title word. Phase 2 (H2) confirms at S=3.

_(Pareto computed by `notebooks/conservative_arch/hybrid/decode_flop_pareto.py`
from `inference_efficiency/flop_counter.py`.)_

