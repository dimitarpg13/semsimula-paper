# Token-direction diagnostics -- matched_baseline

Runs the same two diagnostics as `shared_potential_fit.py` / `jacobian_symmetry.py` but along the **token axis** at a fixed layer, instead of along the **layer axis** at a fixed token.  This tests STP's Geodesic Hypothesis in its natural coordinate system -- the sequence-time axis of autoregressive inference.

- Hidden dim `d = 128`; layers `L = 8`; samples: TRAIN 7,983 / TEST 2,034
- $V_\psi$: 2-layer MLP, hidden = 256, GELU
- PCA $k = 16$, ridge = 0.001, BOS/EOS skip = 2 tokens

## 1. Shared-$V_\psi$ fit (token direction)

- TRAIN pooled $R^2$ = **+0.796**
- TEST pooled $R^2$  = **+0.136**

| layer | A. static null | B. velocity-only | C. velocity + shared $V_\psi$ | C - B |
|--:|--:|--:|--:|--:|
| 0 | -0.000 | +0.294 | +0.532 | +0.238 |
| 1 | -0.001 | +0.258 | +0.436 | +0.178 |
| 2 | -0.001 | +0.244 | +0.314 | +0.071 |
| 3 | -0.002 | +0.237 | +0.159 | -0.078 |
| 4 | -0.001 | +0.238 | +0.112 | -0.126 |
| 5 | -0.001 | +0.239 | +0.085 | -0.154 |
| 6 | -0.001 | +0.239 | +0.076 | -0.164 |
| 7 | -0.001 | +0.234 | +0.087 | -0.147 |
| 8 | -0.001 | +0.234 | +0.114 | -0.121 |

### Learned per-layer scalars

| layer | $\alpha_\ell$ | $\beta_\ell$ |
|--:|--:|--:|
| 0 | -0.0272 | +0.3294 |
| 1 | -0.1294 | +0.2730 |
| 2 | -0.1035 | +0.4160 |
| 3 | -0.0624 | +0.6080 |
| 4 | -0.0397 | +0.6837 |
| 5 | -0.0339 | +0.7231 |
| 6 | -0.0383 | +0.7440 |
| 7 | -0.0293 | +0.7673 |
| 8 | -0.0397 | +0.7649 |

## 2. Velocity-aware Jacobian-symmetry (token direction)

| layer | POS-only $R^2_\text{full}$ | POS-only $R^2_\text{sym}$ | VEL-aug $R^2_\text{full}$ | VEL-aug $R^2_\text{sym}$ | VEL-aug gap |
|--:|--:|--:|--:|--:|--:|
| 0 | +0.576 | +0.576 | +0.583 | +0.583 | +0.000 |
| 1 | +0.538 | +0.530 | +0.549 | +0.537 | +0.012 |
| 2 | +0.521 | +0.511 | +0.529 | +0.517 | +0.012 |
| 3 | +0.509 | +0.491 | +0.520 | +0.505 | +0.015 |
| 4 | +0.500 | +0.487 | +0.508 | +0.497 | +0.011 |
| 5 | +0.504 | +0.489 | +0.508 | +0.498 | +0.010 |
| 6 | +0.509 | +0.496 | +0.511 | +0.504 | +0.007 |
| 7 | +0.509 | +0.494 | +0.511 | +0.504 | +0.007 |
| 8 | +0.511 | +0.496 | +0.514 | +0.507 | +0.007 |

### Verdict

- Velocity-aware symmetric-restricted fit tracks the unconstrained fit (max TEST gap = +0.015).  The per-token spring matrix is consistent with a symmetric Hessian at every layer: the token-direction dynamics is locally Hessian-of-scalar.

## Artefacts

- `tokdir_matched_baseline_results.npz`
- `tokdir_matched_baseline_fig.png`
