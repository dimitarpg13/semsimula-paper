# γ\* prediction summary

- **Checkpoint:** `splm_em_ln_shakespeare_gamma0p10_seed0_ckpt_latest.pt`
- **Trained γ:** 0.1000
- **Params:** 7,123,075
- **Architecture:** d=128, L=8, v_hidden=512, max_len=256
- **Corpus mean unigram surprisal (train split):** 9.109 bits/token
- **Mean per-token mass:** 1.472 (std 0.235)
- **State samples:** 65,536 (86.6% with positive top-eigenvalue)

## §2.1 Depth-scaling closed form

| ρ | γ\*_depth |
|---:|---:|
| 0.050 | 0.5514 |
| 0.100 | 0.4238 |
| 0.150 | 0.3492 |
| 0.180 | 0.3156 |
| 0.200 | 0.2962 |
| 0.300 | 0.2216 |
| 0.500 | 0.1276 |
| 0.565 | 0.1051 |
| 0.700 | 0.0656 |

## §2.2 Hessian-spectrum critical damping

| estimator | γ\* | | mean λ | pos-fraction |
|---|---:|---:|---:|
| top-eigenvalue (positive states only) | **1.1678** | 0.5051 | 86.6% |
| average eigenvalue (Hutchinson) | 0.2281 | 0.0181 | — |

### Per-layer breakdown

| layer | n | pos % | mean λ_top | γ\*_top (pos) | mean m |
|---:|---:|---:|---:|---:|---:|
| 0 | 8,192 | 64.3 | 0.0543 | 1.0498 | 1.472 |
| 1 | 8,192 | 77.9 | 0.2229 | 1.1287 | 1.472 |
| 2 | 8,192 | 90.7 | 0.3615 | 1.1169 | 1.472 |
| 3 | 8,192 | 94.3 | 0.3765 | 1.0919 | 1.472 |
| 4 | 8,192 | 91.8 | 0.3563 | 1.0877 | 1.472 |
| 5 | 8,192 | 90.4 | 0.3685 | 1.1223 | 1.472 |
| 6 | 8,192 | 90.1 | 0.4682 | 1.2691 | 1.472 |
| 7 | 8,192 | 93.4 | 0.6563 | 1.4331 | 1.472 |

## §2.3 Corpus-surprisal scaling

- Tiny-Shakespeare reference (E5): γ\*_E5 = 0.30, S̄_E5 ≈ 9.5 bits, m̄_E5 ≈ 1.4.
- TinyStories (E10): S̄ = 9.109 bits, m̄ = 1.472.
- **γ\*_surprisal({TS}) ≈ 0.2864**

## Reconciliation

| estimator | γ\* | source |
|---|---:|---|
| §2.1 depth (ρ=0.18, buggy v2 anchor)  | 0.3156 | closed form |
| §2.2 Hessian top-eigenvalue       | 1.1678 | trained-checkpoint diagnostic |
| §2.2 Hessian avg-eigenvalue       | 0.2281 | trained-checkpoint diagnostic |
| §2.1 depth (ρ=0.565, leak-free anchor) | 0.1051 | closed form |
| §2.3 corpus-surprisal scaling     | 0.2864 | corpus statistic, no checkpoint |
| (E10 Stage 1 empirical) | TBD | live grid sweep |

All four numbers should agree to within ~10–20 % if the framework is consistent. See `docs/Determining_optimal_gamma_for_SPLM.md` §3 for the full reconciliation rules. The Stage-1 empirical γ\* will be appended to this table once E10 reports.