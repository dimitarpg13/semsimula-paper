# SPLM em_ln γ-sweep diagnostic at scaleup (E9 protocol)

- seed: 0
- corpus: TinyStories (~5 M GPT-2 BPE tokens)
- block_size: 512  batch_size: 16  steps: 8000
- d=256, L=8, max_len=1024
- small-scale γ⋆ (E5 winner @ D=128, L=4): **0.166**

## γ → outcome table

| γ (fixed) | val PPL | val loss (nats) | train→val gap (nats) | wall-clock (h) |
|---:|---:|---:|---:|---:|
| 0.166 | 29.282 | 3.377 | 0.105 | 0.15 |
| 0.200 | 29.274 | 3.377 | 0.105 | 0.15 |
| 0.250 | 29.326 | 3.378 | 0.105 | 0.15 |
| 0.300 | 29.474 | 3.383 | 0.103 | 0.15 |
| 0.350 | 29.420 | 3.382 | 0.104 | 0.15 |

## γ⋆ at scaleup

- argmin γ→PPL: **γ⋆ = 0.200** (val PPL = **29.274**)
- small-scale γ⋆ (E5 winner): **0.166**
- scaleup multiplier: γ⋆_scaleup / γ⋆_small = **1.20×**

→ Multiplier deviates from the predicted **1.8×** scaleup factor by more than ±0.25.  Re-examine the colab pilot's Arm 2 free-γ trajectory and consider extending the sweep range.

**Note:** γ-sweep range = 0.200 PPL (< 0.20 PPL).  Final PPL is **insensitive** to γ in the sweep band — γ acts as an overdamped regularizer and the choice within this band does not change the architectural conclusion.

## Per-γ artifact paths

- γ=0.166: `splm_em_ln_scaleup_scaleup_g166_seed0_ckpt_latest.pt`
- γ=0.200: `splm_em_ln_scaleup_scaleup_g200_seed0_ckpt_latest.pt`
- γ=0.250: `splm_em_ln_scaleup_scaleup_g250_seed0_ckpt_latest.pt`
- γ=0.300: `splm_em_ln_scaleup_scaleup_g300_seed0_ckpt_latest.pt`
- γ=0.350: `splm_em_ln_scaleup_scaleup_g350_seed0_ckpt_latest.pt`
