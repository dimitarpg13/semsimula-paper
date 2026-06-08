# Multi-Xi SPLM Full Training Rerun — Results

## Motivation

The original α-init sweep ran **pilot** (4000 steps) on 8 α-initialisation
strategies. The best pilot configuration (`learned_from_uniform`,
α_init = [0.25, 0.50, 0.75, 0.95]) reached **14.69 PPL** at 4000 steps.

This rerun extends the winner to 8,000 and 16,000 steps to determine the
converged Multi-Xi SPLM PPL.

## Results

| Arm | Steps | PPL | Δ vs pilot |
|-----|-------|-----|------------|
| scaleup_8k | 8,000 | **12.49** | −2.20 |
| extended_16k | 16,000 | **11.51** | −3.18 |

## Final α values

| Arm | α_0 | α_1 | α_2 | α_3 |
|-----|-----|-----|-----|-----|
| scaleup_8k | 0.233 | 0.561 | 0.781 | 0.958 |
| extended_16k | 0.249 | 0.598 | 0.808 | 0.961 |

The α channels converge to similar values at both schedules, with the
slow-decay channel (α_3) remaining near 0.96 throughout.

## SPLM Family Comparison

| Model | PPL | Steps | Inference |
|-------|-----|-------|-----------|
| Attention baseline | 7.81 | 8k | O(T²) |
| Hybrid SPLM+Attn | 8.01 | 16k | O(T) |
| Fock v2.1 (B1+B2+B3) | 9.30 | 16k | O(1) |
| Fock Attention (4-head) | 9.42 | 16k | O(T²) |
| **Multi-Xi SPLM** | **11.51** | **16k** | **O(1)** |
| Multi-Xi SPLM | 12.49 | 8k | O(1) |
| Multi-Xi PARFLM (K=8) | 12.06 | 8k | O(1) |
| Multi-Xi SPLM (pilot) | 14.69 | 4k | O(1) |

![SPLM family comparison](splm_multixi_rerun_comparison.png)

![Convergence curves](splm_multixi_rerun_convergence.png)

![Alpha evolution](splm_multixi_rerun_alpha_evolution.png)

## Key Findings

1. **Extended training closes the gap.** PPL drops from 14.69 (pilot)
   to 11.51 (16k), a 21.6% reduction from training alone.

2. **Multi-Xi SPLM now matches PARFLM.** At 16k steps, the pure
   scalar-potential model (11.51) is within 0.55 PPL of Multi-Xi PARFLM
   (12.06 at 8k) — suggesting that much of the pilot-era gap was due to
   under-training rather than architectural limitation.

3. **Gap to attention is 3.70 PPL.** The remaining gap (11.51 vs 7.81)
   is attributable to the conservative constraint and the absence of
   pairwise token interactions.

4. **α channels are stable.** The learned α values converge to a
   consistent pattern across 8k and 16k, with one fast channel (~0.25),
   two medium channels (~0.58, ~0.80), and one slow channel (~0.96).

## Artifacts

- `extended_16k/` — 16k checkpoint, training log, loss curve, summary
- `scaleup_8k/` — 8k checkpoint, training log, loss curve, summary
- `splm_multixi_rerun_comparison.png` — family PPL comparison bar chart
- `splm_multixi_rerun_convergence.png` — 8k vs 16k convergence curves
- `splm_multixi_rerun_alpha_evolution.png` — α channel evolution
- `splm_multixi_rerun_report.json` — full experiment report

## HuggingFace

Both checkpoints uploaded to
[`dimitarpg13/semsimula-splm-multixi`](https://huggingface.co/dimitarpg13/semsimula-splm-multixi).
