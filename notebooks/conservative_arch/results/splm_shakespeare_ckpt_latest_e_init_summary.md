# E-init validation -- scalar-potential LM (shakespeare_ckpt_latest)

Same protocol as `notebooks/e_init/` §1 experiments, but on trajectories of a conservative-by-construction circuit rather than GPT-2.

- Hidden dim `d = 128`; integration steps `L = 8`
- Corpus: 40 train / 10 test sentences (same as `notebooks/e_init/e_init_corpus.py`)
- Tokens: 1413 across 50 sentences

## Gaussian-well fit quality (TRAIN)

| layer | a | b | $R^{2}$ | n |
|--:|--:|--:|--:|--:|
| 1 | 10.2 | 13.8 | 0.010 | 1087 |
| 2 | 9.98 | 6.12 | 0.001 | 1087 |
| 3 | 9.93 | 5.25 | 0.000 | 1087 |
| 4 | 9.94 | 1.44 | 0.000 | 1087 |
| 5 | 9.93 | 4.37 | 0.000 | 1087 |
| 6 | 9.93 | 2.32 | 0.000 | 1087 |
| 7 | 9.93 | 1.16 | 0.000 | 1087 |
| 8 | 9.93 | 0.212 | 0.000 | 1087 |

## TRAIN / TEST residual vs damping

| $\gamma$ | TRAIN | TEST |
|--:|--:|--:|
| 0.0 | 0.4361 | 0.4406 |
| 0.1 | 0.4338 | 0.4383 |
| 0.25 | 0.4325 | 0.4379 |
| 0.5 | 0.4332 | 0.4377 |
| 1.0 | 0.4341 | 0.4382 |
| 2.0 | 0.4344 | 0.4376 |
| 5.0 | 0.4340 | 0.4372 |

Static-null baseline: TRAIN 0.4347, TEST 0.4369.

## Verdict

Best TEST residual: 0.4372 at $\gamma^{*} = 5.0$  (Δ vs null = +0.0003).

**Matches static null (no clean scalar-potential fit).**

Comparison reference: every attention-transformer fit in §1 of the Failure doc matched the static null on held-out data (Δ ≈ 0 at best). A negative Δ here of order $-0.01$ or more would constitute the quantitative positive control that v2 needs.

## Artefacts

- `splm_shakespeare_ckpt_latest_e_init_results.npz`
- `splm_shakespeare_ckpt_latest_fig_residual_vs_gamma.png`
- `splm_shakespeare_ckpt_latest_fig_residual_vs_layer.png`
