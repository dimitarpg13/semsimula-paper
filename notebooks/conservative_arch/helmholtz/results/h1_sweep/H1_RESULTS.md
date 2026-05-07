# H1 — Helmholtz (Q9d) schedule sweep results


Architecture: layer-type Helmholtz hybrid -- single shared
`V_theta` on every S-block, per-layer attention on every A-block,
velocity-Verlet damped Euler-Lagrange S-step. Schedule `sigma: {0..L-1} -> {S, A}` controls the architectural shape.

Tied embeddings, `causal_force=True`, `ln_after_s_step=True`,
`mass_mode='logfreq'`, Tiny Shakespeare 4000 steps at d=128.

## Per-schedule results

| schedule | n_S | n_A | seed | γ_mode | val PPL | val loss | final γ | params | elapsed |
|----------|-----|-----|------|--------|---------|----------|---------|--------|---------|
| `SSSSSSSA` | 7 | 1 | 0 | free γ | 179.06 | 5.1877 | 0.152 | 7.32 M | 32.8 min |
| `AASSSSSS` | 6 | 2 | 0 | free γ | 140.60 | 4.9459 | 0.162 | 7.52 M | 37.7 min |
| `ASSSSSSA` | 6 | 2 | 0 | free γ | 200.81 | 5.3024 | 0.151 | 7.52 M | 34.8 min |
| `AAAASSSS` | 4 | 4 | 0 | free γ | 135.03 | 4.9055 | 0.163 | 7.92 M | 29.8 min |
| `SASASASA` | 4 | 4 | 0 | free γ | 189.64 | 5.2451 | 0.138 | 7.92 M | 30.2 min |
| `SSAAAASS` | 4 | 4 | 0 | free γ | 139.62 | 4.9389 | 0.114 | 7.92 M | 29.7 min |
| `SAAAAAAS` | 2 | 6 | 0 | free γ | 137.60 | 4.9243 | 0.147 | 8.31 M | 30.3 min |

## Anchors (already on disk, leak-free)

| arm | val PPL | source |
|-----|---------|--------|
| All-attention (matched GPT-2, n_layer=8) | ~150 | `multi_seed/results/` |
| All-SPLM em_ln (free-γ, leak-free) | 173.59 | `energetic_minima/results/` |
| Variant A HSPLM (k=4, m=4) at S=1 | **133.01** | `hybrid/results/h1_sweep/H1_RESULTS.md` |
| Variant A HSPLM best across (k, m) | 133.01–147.28 | `hybrid/results/h1_sweep/H1_RESULTS.md` |

## Q9d-vs-Variant-A comparison

The Helmholtz `bottom_a_LA4` schedule (`AAAASSSS`) is the closest
Q9d analogue of Variant A HSPLM (k=4, m=4): same param count,
same compute, same n_S/n_A. Two structural differences are
expected to drive any divergence in PPL:

1. The velocity proxy `h_ell - h_{ell-1}` passes *through* the
 attention stack rather than being reset to v=0 at the SPLM
 boundary.
2. `xi` is re-derived from the running `h.detach` at every
 S-block, instead of being fixed at `h_k.detach` once after
 the attention stack.

If `bottom_a_LA4` matches Variant A's 133.01 within seed noise,
the velocity-passing and xi-re-derivation are kinematically
equivalent at this scale. The interesting cells are the
schedules unreachable by Variant A: `sandwich_k1` (boundary-case
S-blocks, doc §A.5 mechanism), `interleaved` (block-type-indexed
step-function R²_ψ test, doc §4.1), `top_a_LA1` (single-attention
hybrid, the cleanest narrative cell of doc §6).

## Pre-registered decision (Phase 2 gate)

Per `the v4 title-justification rule` §6.5:
**"Efficient" is justified iff** some hybrid achieves val PPL
within +5 PPL of the all-attention baseline AND its analytical
decode-FLOP cost at T=1024 is ≥ 30% lower than all-attention,
both at S=3 with sign-consistency 3/3.

**Phase 1 → Phase 2 gate:** if best schedule at S=1 (this sweep)
is within +10 PPL of all-attention, proceed to H2 (S=3
confirmation). If gap is > 15 PPL, soften the title (Option 2
fallback) and document the schedule sweep as Future Work.

## Best-of-sweep summary

Best schedule: **`AAAASSSS`** (n_S=4, n_A=4) at seed 0 with val PPL **135.03**.
Gap to all-attention (~150): **-14.97 PPL**.
Gap to Variant A best (133.01): +2.02 PPL.

Phase 2 gate: **PASS** (within +10 PPL of all-attention).

## Decode-FLOP Pareto (analytical)

Per-token AR decode FLOPs at context length T (KV-cached attention; streaming-ξ SPLM). Cost depends only on the *count* (n_A, n_S) of each block type, so schedules with the same (n_A, n_S) report identical FLOP cost — only val PPL differentiates them.


### T = 256

All-attention reference (`AAAAAAAA`): **17.116 MFLOPs/tok**.
All-SPLM reference (`SSSSSSSS`): **44.397 MFLOPs/tok**.

| schedule | n_S | n_A | val PPL (S=1) | decode FLOPs/tok | vs all-attn | reduction |
|----------|-----|-----|---------------|------------------|-------------|-----------|
| `SSSSSSSA` | 7 | 1 | 179.06 | 40.986 MFLOPs | 2.395× | -139.5% |
| `AASSSSSS` | 6 | 2 | 140.60 | 37.576 MFLOPs | 2.195× | -119.5% |
| `ASSSSSSA` | 6 | 2 | 200.81 | 37.576 MFLOPs | 2.195× | -119.5% |
| `AAAASSSS` | 4 | 4 | 135.03 | 30.756 MFLOPs | 1.797× | -79.7% |
| `SASASASA` | 4 | 4 | 189.64 | 30.756 MFLOPs | 1.797× | -79.7% |
| `SSAAAASS` | 4 | 4 | 139.62 | 30.756 MFLOPs | 1.797× | -79.7% |
| `SAAAAAAS` | 2 | 6 | 137.60 | 23.936 MFLOPs | 1.399× | -39.9% |

### T = 1024

All-attention reference (`AAAAAAAA`): **20.384 MFLOPs/tok**.
All-SPLM reference (`SSSSSSSS`): **44.397 MFLOPs/tok**.

| schedule | n_S | n_A | val PPL (S=1) | decode FLOPs/tok | vs all-attn | reduction |
|----------|-----|-----|---------------|------------------|-------------|-----------|
| `SSSSSSSA` | 7 | 1 | 179.06 | 41.395 MFLOPs | 2.031× | -103.1% |
| `AASSSSSS` | 6 | 2 | 140.60 | 38.394 MFLOPs | 1.884× | -88.4% |
| `ASSSSSSA` | 6 | 2 | 200.81 | 38.394 MFLOPs | 1.884× | -88.4% |
| `AAAASSSS` | 4 | 4 | 135.03 | 32.391 MFLOPs | 1.589× | -58.9% |
| `SASASASA` | 4 | 4 | 189.64 | 32.391 MFLOPs | 1.589× | -58.9% |
| `SSAAAASS` | 4 | 4 | 139.62 | 32.391 MFLOPs | 1.589× | -58.9% |
| `SAAAAAAS` | 2 | 6 | 137.60 | 26.388 MFLOPs | 1.295× | -29.5% |

### T = 4096

All-attention reference (`AAAAAAAA`): **33.459 MFLOPs/tok**.
All-SPLM reference (`SSSSSSSS`): **44.397 MFLOPs/tok**.

| schedule | n_S | n_A | val PPL (S=1) | decode FLOPs/tok | vs all-attn | reduction |
|----------|-----|-----|---------------|------------------|-------------|-----------|
| `SSSSSSSA` | 7 | 1 | 179.06 | 43.029 MFLOPs | 1.286× | -28.6% |
| `AASSSSSS` | 6 | 2 | 140.60 | 41.662 MFLOPs | 1.245× | -24.5% |
| `ASSSSSSA` | 6 | 2 | 200.81 | 41.662 MFLOPs | 1.245× | -24.5% |
| `AAAASSSS` | 4 | 4 | 135.03 | 38.928 MFLOPs | 1.163× | -16.3% |
| `SASASASA` | 4 | 4 | 189.64 | 38.928 MFLOPs | 1.163× | -16.3% |
| `SSAAAASS` | 4 | 4 | 139.62 | 38.928 MFLOPs | 1.163× | -16.3% |
| `SAAAAAAS` | 2 | 6 | 137.60 | 36.194 MFLOPs | 1.082× | -8.2% |

### Pre-registered rule check

Per `the v4 title-justification rule` §6.5:
**"Efficient" is justified iff** some hybrid achieves val PPL
within +5 PPL of the all-attention baseline (~150) AND decode-FLOP
cost at T=1024 is ≥ 30% lower than all-attention, both at S=3 with
sign 3/3.

Note: at the prototype `v_hidden = 512` the SPLM step costs
~3.94 MFLOPs/tok, which is ~4× per-step heavier than an attention
block at T=1024 (~0.94 MFLOPs/tok). Schedules with high n_S
are therefore *more* expensive at T=1024 -- the same FLOP-arm
failure mode the Variant A H1 sweep documents. The H1.5 fix is
to narrow `v_hidden`; in Q9d it applies identically since V_theta
is the same module.

_(Pareto computed by
`notebooks/conservative_arch/helmholtz/decode_flop_pareto.py`
from `inference_efficiency/flop_counter.py` and the per-block
helpers in `hybrid/decode_flop_pareto.py`.)_

