# Post-completion forensics — `multixi_pilot_fixed` (leak-corrected)

**ckpt:** `/Users/dimitargueorguiev/git/ml/semsimula/notebooks/conservative_arch/scaleup/results/multixi_pilot_fixed/splm_em_ln_multixi_pilot_fixed_ckpt_latest.pt`
**training log:** `/Users/dimitargueorguiev/git/ml/semsimula/notebooks/conservative_arch/scaleup/results/multixi_pilot_fixed/splm_em_ln_multixi_pilot_fixed_training_log.jsonl`  (rows: 80)

## α_k drift (init → final)

| channel | α init | α final | drift |
|---:|---:|---:|---:|
| 0 | 0.0000 | 0.0000 | +0.0000 |
| 1 | 0.4995 | 0.5191 | +0.0195 |
| 2 | 0.9000 | 0.8547 | -0.0453 |
| 3 | 0.9900 | 0.9794 | -0.0106 |

## val PPL trajectory

| step | val_loss | val_ppl |
|---:|---:|---:|
| 200 | 4.7239 | 112.60 |
| 400 | 3.6785 | 39.59 |
| 600 | 3.4062 | 30.15 |
| 800 | 3.2134 | 24.86 |
| 1000 | 3.1095 | 22.41 |
| 1200 | 3.0093 | 20.27 |
| 1400 | 2.9579 | 19.26 |
| 1600 | 2.9130 | 18.41 |
| 1800 | 2.8722 | 17.68 |
| 2000 | 2.8144 | 16.68 |
| 2200 | 2.8106 | 16.62 |
| 2400 | 2.7416 | 15.51 |
| 2600 | 2.7338 | 15.39 |
| 2800 | 2.7295 | 15.33 |
| 3000 | 2.7259 | 15.27 |
| 3200 | 2.7168 | 15.13 |
| 3400 | 2.7202 | 15.18 |
| 3600 | 2.6818 | 14.61 |
| 3800 | 2.7026 | 14.92 |
| 4000 | 2.6934 | 14.78 |

## Causal-violation probe (same trained weights, two modes)

matched class: `multixi (K-channel ξ)`  (vocab=50257, max_len=1024, T=64, t_pert=40)

| evaluator | causal-side Δ | after-side Δ |
|---|---:|---:|
| buggy | 9.5810e-02 | 1.9539e-01 |
| fixed | 0.0000e+00 | 2.1386e-01 |

Expected: fixed-mode Δ ≈ 0 (the post-fix integrator is causal). Buggy-mode Δ may still be > 0 because the buggy integrator is forward-noncausal at inference *regardless* of how the V_θ was trained — its h-update depends on V which depends on h_{>t}. Whether that noncausality helps or hurts the loss is what discriminates buggy-trained from leak-corrected ckpts (see inflation table below).

## Val-PPL inflation (same trained weights, two evaluators)

corpus: `tinystories`   batches: 20 × 8 × 256 = 40,960 tokens

| evaluator | val_loss | val_ppl |
|---|---:|---:|
| buggy (pre-fix integrator) | 8.8840 | 7215.34 |
| fixed (post-fix integrator) | 2.6984 | 14.86 |
| **inflation factor** | | **0.00×** |

Expected:
- **buggy-trained**: inflation >> 1 (e.g., 389× on `multixi_buggy_2k`). V_θ exploits the leak; under fixed eval the leak is severed and ppl shoots up.
- **leak-corrected**: inflation << 1. V_θ never learned to use the leak. The buggy integrator still injects future info into h_t, but V_θ treats it as destructive noise → ppl explodes under buggy eval.
- **regression** (fix did not take during training): inflation ≈ 1×.

## Headline comparison

| run | final train val_ppl | fixed-eval val_ppl | inflation | causal-side Δ (fixed mode) |
|---|---:|---:|---:|---:|
| buggy multi-ξ 2k (`multixi_buggy_2k`) | 1.05 | 408.12 | 389× | 0.0000 |
| **`multixi_pilot_fixed` (leak-corrected)** | **14.78** | **14.86** | **0.00×** | **0.0000e+00** |
