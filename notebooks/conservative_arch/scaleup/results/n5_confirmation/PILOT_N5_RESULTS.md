# paper_tmlr_1 — n=5 paired confirmation results

- seeds: [0, 1, 2, 3, 4]
- corpus: TinyStories (~5 M GPT-2 BPE tokens)
- block_size: 512  batch_size: 16  steps: 8000
- d=256, L=8, max_len=1024  (matches E9 SPLM scale-up protocol; identical to colab_pilot)

## Per-arm marginal statistics

| Arm | n | val PPL  mean ± std | 95% CI (mean) | train→val gap mean ± std | wall-clock mean (h) |
|---|---:|---:|---:|---:|---:|
| matched-attn baseline | 5 | 7.790 ± 0.064 | [+7.710, +7.869] | 0.426 ± 0.011 | 0.08 |
| Helmholtz Q9d (AAAASSSS) | 5 | 8.361 ± 0.071 | [+8.272, +8.449] | 0.262 ± 0.012 | 0.11 |
| Hybrid VA (k=4, m=4) | 5 | 8.503 ± 0.035 | [+8.459, +8.546] | 0.280 ± 0.011 | 0.11 |

## Paired Δ vs matched-attention baseline

Per-seed Δ_s = PPL_arm[s] − PPL_base[s] (paired by seed; negative ⇒ arm beats baseline).  Mean Δ uses **only** seeds where both arm and baseline have a valid result.  95% CI uses Student's t with df = n − 1.

| Arm | n_paired | per-seed Δ | mean Δ | 95% CI (mean Δ) | verdict |
|---|---:|---|---:|---:|---|
| Helmholtz Q9d (AAAASSSS) | 5 | s0: +0.599, s1: +0.540, s2: +0.550, s3: +0.564, s4: +0.600 | +0.571 | [+0.536, +0.605] | _baseline wins_ (CI above 0) |
| Hybrid VA (k=4, m=4) | 5 | s0: +0.746, s1: +0.727, s2: +0.730, s3: +0.606, s4: +0.754 | +0.713 | [+0.637, +0.788] | _baseline wins_ (CI above 0) |

## Per-seed val PPL (full transparency)

| Seed | matched-attn baseline | Helmholtz Q9d (AAAASSSS) | Hybrid VA (k=4, m=4) |
|---:|---:|---:|---:|
| 0 | 7.813 | 8.412 | 8.559 |
| 1 | 7.736 | 8.276 | 8.463 |
| 2 | 7.776 | 8.327 | 8.506 |
| 3 | 7.889 | 8.454 | 8.495 |
| 4 | 7.736 | 8.335 | 8.490 |

## Generalization gap (final val − final train, in nats)

Smaller gap with comparable val PPL ⇒ tighter generalization (structural regularization).  All values reported as mean ± std across the n paired seeds.

| Arm | n | gap mean ± std |
|---|---:|---:|
| matched-attn baseline | 5 | 0.426 ± 0.011 |
| Helmholtz Q9d (AAAASSSS) | 5 | 0.262 ± 0.012 |
| Hybrid VA (k=4, m=4) | 5 | 0.280 ± 0.011 |

## Per-arm artifact paths (per-seed checkpoints)

- **matched-attn baseline**
    - seed 0: `matched_baseline_scaleup_scaleup_seed0_ckpt_latest.pt`
    - seed 1: `matched_baseline_scaleup_scaleup_seed1_ckpt_latest.pt`
    - seed 2: `matched_baseline_scaleup_scaleup_seed2_ckpt_latest.pt`
    - seed 3: `matched_baseline_scaleup_scaleup_seed3_ckpt_latest.pt`
    - seed 4: `matched_baseline_scaleup_scaleup_seed4_ckpt_latest.pt`
- **Helmholtz Q9d (AAAASSSS)**
    - seed 0: `helmholtz_AAAASSSS_L8_scaleup_scaleup_seed0_ckpt_latest.pt`
    - seed 1: `helmholtz_AAAASSSS_L8_scaleup_scaleup_seed1_ckpt_latest.pt`
    - seed 2: `helmholtz_AAAASSSS_L8_scaleup_scaleup_seed2_ckpt_latest.pt`
    - seed 3: `helmholtz_AAAASSSS_L8_scaleup_scaleup_seed3_ckpt_latest.pt`
    - seed 4: `helmholtz_AAAASSSS_L8_scaleup_scaleup_seed4_ckpt_latest.pt`
- **Hybrid VA (k=4, m=4)**
    - seed 0: `hybrid_VA_k4_m4_scaleup_scaleup_seed0_ckpt_latest.pt`
    - seed 1: `hybrid_VA_k4_m4_scaleup_scaleup_seed1_ckpt_latest.pt`
    - seed 2: `hybrid_VA_k4_m4_scaleup_scaleup_seed2_ckpt_latest.pt`
    - seed 3: `hybrid_VA_k4_m4_scaleup_scaleup_seed3_ckpt_latest.pt`
    - seed 4: `hybrid_VA_k4_m4_scaleup_scaleup_seed4_ckpt_latest.pt`
