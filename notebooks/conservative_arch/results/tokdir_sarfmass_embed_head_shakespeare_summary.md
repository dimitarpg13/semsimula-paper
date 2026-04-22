# Token-direction diagnostics -- sarfmass_embed_head_shakespeare

Runs the same two diagnostics as `shared_potential_fit.py` / `jacobian_symmetry.py` but along the **token axis** at a fixed layer, instead of along the **layer axis** at a fixed token.  This tests STP's Geodesic Hypothesis in its natural coordinate system -- the sequence-time axis of autoregressive inference.

- Hidden dim `d = 128`; layers `L = 8`; samples: TRAIN 7,983 / TEST 2,034
- $V_\psi$: 2-layer MLP, hidden = 256, GELU
- PCA $k = 16$, ridge = 0.001, BOS/EOS skip = 2 tokens

## 1. Shared-$V_\psi$ fit (token direction)

- TRAIN pooled $R^2$ = **+0.667**
- TEST pooled $R^2$  = **+0.364**

| layer | A. static null | B. velocity-only | C. velocity + shared $V_\psi$ | C - B |
|--:|--:|--:|--:|--:|
| 0 | -0.000 | +0.280 | +0.283 | +0.003 |
| 1 | -0.001 | +0.257 | +0.276 | +0.019 |
| 2 | -0.001 | +0.273 | +0.407 | +0.134 |
| 3 | -0.001 | +0.237 | +0.385 | +0.148 |
| 4 | -0.001 | +0.237 | +0.364 | +0.127 |
| 5 | -0.001 | +0.233 | +0.371 | +0.137 |
| 6 | -0.001 | +0.230 | +0.374 | +0.144 |
| 7 | -0.001 | +0.230 | +0.363 | +0.133 |
| 8 | -0.001 | +0.228 | +0.346 | +0.118 |

### Learned per-layer scalars

| layer | $\alpha_\ell$ | $\beta_\ell$ |
|--:|--:|--:|
| 0 | -0.5297 | +0.0017 |
| 1 | -0.4929 | +0.0158 |
| 2 | -0.1241 | +0.3181 |
| 3 | -0.0405 | +0.4412 |
| 4 | -0.0249 | +0.5218 |
| 5 | -0.0326 | +0.5572 |
| 6 | -0.0239 | +0.6354 |
| 7 | -0.0374 | +0.7080 |
| 8 | -0.0372 | +0.7760 |

## 2. Velocity-aware Jacobian-symmetry (token direction)

| layer | POS-only $R^2_\text{full}$ | POS-only $R^2_\text{sym}$ | VEL-aug $R^2_\text{full}$ | VEL-aug $R^2_\text{sym}$ | VEL-aug gap |
|--:|--:|--:|--:|--:|--:|
| 0 | +0.585 | +0.575 | +0.592 | +0.586 | +0.006 |
| 1 | +0.611 | +0.575 | +0.618 | +0.586 | +0.031 |
| 2 | +0.555 | +0.525 | +0.546 | +0.519 | +0.026 |
| 3 | +0.551 | +0.490 | +0.552 | +0.479 | +0.073 |
| 4 | +0.542 | +0.474 | +0.543 | +0.465 | +0.078 |
| 5 | +0.527 | +0.465 | +0.527 | +0.459 | +0.067 |
| 6 | +0.528 | +0.465 | +0.526 | +0.463 | +0.063 |
| 7 | +0.537 | +0.466 | +0.538 | +0.467 | +0.071 |
| 8 | +0.533 | +0.465 | +0.538 | +0.468 | +0.070 |

### Verdict

- Velocity-aware symmetric-restricted fit tracks the unconstrained fit (max TEST gap = +0.078).  The per-token spring matrix is consistent with a symmetric Hessian at every layer: the token-direction dynamics is locally Hessian-of-scalar.

## Artefacts

- `tokdir_sarfmass_embed_head_shakespeare_results.npz`
- `tokdir_sarfmass_embed_head_shakespeare_fig.png`
