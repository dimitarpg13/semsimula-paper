# Token-direction diagnostics -- splm_shakespeare

Runs the same two diagnostics as `shared_potential_fit.py` / `jacobian_symmetry.py` but along the **token axis** at a fixed layer, instead of along the **layer axis** at a fixed token.  This tests STP's Geodesic Hypothesis in its natural coordinate system -- the sequence-time axis of autoregressive inference.

- Hidden dim `d = 128`; layers `L = 8`; samples: TRAIN 7,983 / TEST 2,034
- $V_\psi$: 2-layer MLP, hidden = 256, GELU
- PCA $k = 16$, ridge = 0.001, BOS/EOS skip = 2 tokens

## 1. Shared-$V_\psi$ fit (token direction)

- TRAIN pooled $R^2$ = **+0.683**
- TEST pooled $R^2$  = **+0.518**

| layer | A. static null | B. velocity-only | C. velocity + shared $V_\psi$ | C - B |
|--:|--:|--:|--:|--:|
| 0 | -0.000 | +0.274 | +0.466 | +0.192 |
| 1 | -0.000 | +0.271 | +0.479 | +0.208 |
| 2 | -0.000 | +0.256 | +0.508 | +0.252 |
| 3 | -0.000 | +0.258 | +0.536 | +0.278 |
| 4 | -0.000 | +0.270 | +0.551 | +0.281 |
| 5 | -0.000 | +0.248 | +0.527 | +0.279 |
| 6 | -0.000 | +0.273 | +0.506 | +0.233 |
| 7 | -0.000 | +0.273 | +0.489 | +0.216 |
| 8 | -0.000 | +0.264 | +0.533 | +0.269 |

### Learned per-layer scalars

| layer | $\alpha_\ell$ | $\beta_\ell$ |
|--:|--:|--:|
| 0 | -0.1348 | +0.2873 |
| 1 | -0.0979 | +0.2853 |
| 2 | -0.0392 | +0.3616 |
| 3 | +0.0119 | +0.3863 |
| 4 | +0.1000 | +0.4202 |
| 5 | +0.0987 | +0.4719 |
| 6 | +0.0629 | +0.5181 |
| 7 | +0.1039 | +0.5472 |
| 8 | +0.1100 | +0.5984 |

## 2. Velocity-aware Jacobian-symmetry (token direction)

| layer | POS-only $R^2_\text{full}$ | POS-only $R^2_\text{sym}$ | VEL-aug $R^2_\text{full}$ | VEL-aug $R^2_\text{sym}$ | VEL-aug gap |
|--:|--:|--:|--:|--:|--:|
| 0 | +0.580 | +0.571 | +0.589 | +0.577 | +0.011 |
| 1 | +0.585 | +0.573 | +0.592 | +0.582 | +0.010 |
| 2 | +0.573 | +0.563 | +0.582 | +0.570 | +0.012 |
| 3 | +0.582 | +0.570 | +0.587 | +0.579 | +0.008 |
| 4 | +0.587 | +0.581 | +0.582 | +0.580 | +0.002 |
| 5 | +0.586 | +0.573 | +0.591 | +0.586 | +0.005 |
| 6 | +0.605 | +0.588 | +0.609 | +0.599 | +0.011 |
| 7 | +0.601 | +0.588 | +0.608 | +0.598 | +0.010 |
| 8 | +0.598 | +0.584 | +0.608 | +0.593 | +0.015 |

### Verdict

- Velocity-aware symmetric-restricted fit tracks the unconstrained fit (max TEST gap = +0.015).  The per-token spring matrix is consistent with a symmetric Hessian at every layer: the token-direction dynamics is locally Hessian-of-scalar.

## Artefacts

- `tokdir_splm_shakespeare_results.npz`
- `tokdir_splm_shakespeare_fig.png`
