# Token-direction diagnostics -- sarfmass_logfreq_shakespeare

Runs the same two diagnostics as `shared_potential_fit.py` / `jacobian_symmetry.py` but along the **token axis** at a fixed layer, instead of along the **layer axis** at a fixed token.  This tests STP's Geodesic Hypothesis in its natural coordinate system -- the sequence-time axis of autoregressive inference.

- Hidden dim `d = 128`; layers `L = 8`; samples: TRAIN 7,983 / TEST 2,034
- $V_\psi$: 2-layer MLP, hidden = 256, GELU
- PCA $k = 16$, ridge = 0.001, BOS/EOS skip = 2 tokens

## 1. Shared-$V_\psi$ fit (token direction)

- TRAIN pooled $R^2$ = **+0.733**
- TEST pooled $R^2$  = **+0.329**

| layer | A. static null | B. velocity-only | C. velocity + shared $V_\psi$ | C - B |
|--:|--:|--:|--:|--:|
| 0 | -0.000 | +0.285 | +0.287 | +0.002 |
| 1 | -0.002 | +0.270 | +0.484 | +0.214 |
| 2 | -0.002 | +0.267 | +0.526 | +0.260 |
| 3 | -0.002 | +0.261 | +0.463 | +0.202 |
| 4 | -0.002 | +0.258 | +0.373 | +0.115 |
| 5 | -0.002 | +0.258 | +0.347 | +0.089 |
| 6 | -0.002 | +0.261 | +0.323 | +0.062 |
| 7 | -0.002 | +0.259 | +0.258 | -0.001 |
| 8 | -0.001 | +0.256 | +0.195 | -0.061 |

### Learned per-layer scalars

| layer | $\alpha_\ell$ | $\beta_\ell$ |
|--:|--:|--:|
| 0 | -0.5355 | +0.0021 |
| 1 | -0.0722 | +0.4673 |
| 2 | +0.0039 | +0.5831 |
| 3 | +0.0279 | +0.6499 |
| 4 | +0.0398 | +0.6957 |
| 5 | +0.0542 | +0.7359 |
| 6 | +0.0576 | +0.7907 |
| 7 | +0.0660 | +0.8418 |
| 8 | +0.0495 | +0.9050 |

## 2. Velocity-aware Jacobian-symmetry (token direction)

| layer | POS-only $R^2_\text{full}$ | POS-only $R^2_\text{sym}$ | VEL-aug $R^2_\text{full}$ | VEL-aug $R^2_\text{sym}$ | VEL-aug gap |
|--:|--:|--:|--:|--:|--:|
| 0 | +0.583 | +0.577 | +0.587 | +0.582 | +0.006 |
| 1 | +0.634 | +0.590 | +0.632 | +0.609 | +0.023 |
| 2 | +0.641 | +0.587 | +0.633 | +0.606 | +0.027 |
| 3 | +0.633 | +0.582 | +0.632 | +0.609 | +0.023 |
| 4 | +0.633 | +0.578 | +0.638 | +0.602 | +0.036 |
| 5 | +0.637 | +0.576 | +0.643 | +0.599 | +0.044 |
| 6 | +0.637 | +0.575 | +0.644 | +0.598 | +0.046 |
| 7 | +0.629 | +0.568 | +0.635 | +0.590 | +0.045 |
| 8 | +0.631 | +0.562 | +0.637 | +0.582 | +0.055 |

### Verdict

- Velocity-aware symmetric-restricted fit tracks the unconstrained fit (max TEST gap = +0.055).  The per-token spring matrix is consistent with a symmetric Hessian at every layer: the token-direction dynamics is locally Hessian-of-scalar.

## Artefacts

- `tokdir_sarfmass_logfreq_shakespeare_results.npz`
- `tokdir_sarfmass_logfreq_shakespeare_fig.png`
