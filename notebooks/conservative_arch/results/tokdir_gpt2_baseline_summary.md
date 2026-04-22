# Token-direction diagnostics -- gpt2_baseline

Runs the same two diagnostics as `shared_potential_fit.py` / `jacobian_symmetry.py` but along the **token axis** at a fixed layer, instead of along the **layer axis** at a fixed token.  This tests STP's Geodesic Hypothesis in its natural coordinate system -- the sequence-time axis of autoregressive inference.

- Hidden dim `d = 768`; layers `L = 12`; samples: TRAIN 11,531 / TEST 2,938
- $V_\psi$: 2-layer MLP, hidden = 256, GELU
- PCA $k = 16$, ridge = 0.001, BOS/EOS skip = 2 tokens

## 1. Shared-$V_\psi$ fit (token direction)

- TRAIN pooled $R^2$ = **+0.492**
- TEST pooled $R^2$  = **+0.185**

| layer | A. static null | B. velocity-only | C. velocity + shared $V_\psi$ | C - B |
|--:|--:|--:|--:|--:|
| 0 | -0.002 | +0.242 | +0.252 | +0.010 |
| 1 | -0.001 | +0.238 | +0.276 | +0.039 |
| 2 | -0.000 | +0.235 | +0.276 | +0.041 |
| 3 | -0.000 | +0.221 | +0.259 | +0.038 |
| 4 | -0.000 | +0.201 | +0.232 | +0.031 |
| 5 | -0.000 | +0.186 | +0.216 | +0.030 |
| 6 | -0.000 | +0.174 | +0.200 | +0.025 |
| 7 | -0.001 | +0.164 | +0.180 | +0.016 |
| 8 | -0.001 | +0.156 | +0.165 | +0.009 |
| 9 | -0.000 | +0.148 | +0.151 | +0.003 |
| 10 | -0.000 | +0.148 | +0.152 | +0.004 |
| 11 | -0.001 | +0.159 | +0.175 | +0.016 |
| 12 | -0.001 | +0.131 | +0.238 | +0.107 |

### Learned per-layer scalars

| layer | $\alpha_\ell$ | $\beta_\ell$ |
|--:|--:|--:|
| 0 | -0.4671 | +0.0036 |
| 1 | -0.3598 | +0.1223 |
| 2 | -0.3388 | +0.1644 |
| 3 | -0.3021 | +0.2268 |
| 4 | -0.2720 | +0.2784 |
| 5 | -0.2454 | +0.3210 |
| 6 | -0.2257 | +0.3615 |
| 7 | -0.2144 | +0.4074 |
| 8 | -0.2016 | +0.4712 |
| 9 | -0.1965 | +0.5422 |
| 10 | -0.1967 | +0.6300 |
| 11 | -0.2248 | +0.7339 |
| 12 | -0.2584 | +0.2548 |

## 2. Velocity-aware Jacobian-symmetry (token direction)

| layer | POS-only $R^2_\text{full}$ | POS-only $R^2_\text{sym}$ | VEL-aug $R^2_\text{full}$ | VEL-aug $R^2_\text{sym}$ | VEL-aug gap |
|--:|--:|--:|--:|--:|--:|
| 0 | +0.601 | +0.562 | +0.602 | +0.586 | +0.015 |
| 1 | +0.612 | +0.561 | +0.614 | +0.587 | +0.027 |
| 2 | +0.608 | +0.555 | +0.610 | +0.581 | +0.029 |
| 3 | +0.589 | +0.518 | +0.595 | +0.558 | +0.037 |
| 4 | +0.573 | +0.497 | +0.582 | +0.545 | +0.037 |
| 5 | +0.574 | +0.483 | +0.581 | +0.538 | +0.043 |
| 6 | +0.573 | +0.473 | +0.581 | +0.534 | +0.048 |
| 7 | +0.564 | +0.460 | +0.573 | +0.523 | +0.050 |
| 8 | +0.551 | +0.449 | +0.556 | +0.510 | +0.046 |
| 9 | +0.548 | +0.448 | +0.551 | +0.506 | +0.045 |
| 10 | +0.549 | +0.451 | +0.551 | +0.507 | +0.044 |
| 11 | +0.542 | +0.457 | +0.542 | +0.504 | +0.039 |
| 12 | +0.448 | +0.406 | +0.442 | +0.423 | +0.018 |

### Verdict

- Velocity-aware symmetric-restricted fit tracks the unconstrained fit (max TEST gap = +0.050).  The per-token spring matrix is consistent with a symmetric Hessian at every layer: the token-direction dynamics is locally Hessian-of-scalar.

## Artefacts

- `tokdir_gpt2_baseline_results.npz`
- `tokdir_gpt2_baseline_fig.png`
