# Token-direction diagnostics -- sarf_shakespeare

Runs the same two diagnostics as `shared_potential_fit.py` / `jacobian_symmetry.py` but along the **token axis** at a fixed layer, instead of along the **layer axis** at a fixed token.  This tests STP's Geodesic Hypothesis in its natural coordinate system -- the sequence-time axis of autoregressive inference.

- Hidden dim `d = 128`; layers `L = 8`; samples: TRAIN 7,983 / TEST 2,034
- $V_\psi$: 2-layer MLP, hidden = 256, GELU
- PCA $k = 16$, ridge = 0.001, BOS/EOS skip = 2 tokens

## 1. Shared-$V_\psi$ fit (token direction)

- TRAIN pooled $R^2$ = **+0.660**
- TEST pooled $R^2$  = **+0.406**

| layer | A. static null | B. velocity-only | C. velocity + shared $V_\psi$ | C - B |
|--:|--:|--:|--:|--:|
| 0 | -0.000 | +0.287 | +0.357 | +0.069 |
| 1 | -0.000 | +0.253 | +0.448 | +0.195 |
| 2 | -0.000 | +0.256 | +0.406 | +0.151 |
| 3 | -0.001 | +0.250 | +0.394 | +0.144 |
| 4 | -0.001 | +0.255 | +0.392 | +0.137 |
| 5 | -0.001 | +0.260 | +0.410 | +0.150 |
| 6 | -0.001 | +0.260 | +0.393 | +0.133 |
| 7 | -0.000 | +0.260 | +0.411 | +0.151 |
| 8 | -0.000 | +0.261 | +0.411 | +0.150 |

### Learned per-layer scalars

| layer | $\alpha_\ell$ | $\beta_\ell$ |
|--:|--:|--:|
| 0 | -0.4187 | -0.0095 |
| 1 | -0.1998 | -0.0498 |
| 2 | -0.1636 | -0.1010 |
| 3 | -0.1248 | -0.1537 |
| 4 | -0.0707 | -0.2234 |
| 5 | -0.0432 | -0.3031 |
| 6 | -0.0186 | -0.3936 |
| 7 | -0.0083 | -0.4911 |
| 8 | -0.0110 | -0.5588 |

## 2. Velocity-aware Jacobian-symmetry (token direction)

| layer | POS-only $R^2_\text{full}$ | POS-only $R^2_\text{sym}$ | VEL-aug $R^2_\text{full}$ | VEL-aug $R^2_\text{sym}$ | VEL-aug gap |
|--:|--:|--:|--:|--:|--:|
| 0 | +0.585 | +0.576 | +0.588 | +0.584 | +0.004 |
| 1 | +0.608 | +0.586 | +0.618 | +0.601 | +0.017 |
| 2 | +0.562 | +0.522 | +0.555 | +0.519 | +0.036 |
| 3 | +0.580 | +0.541 | +0.574 | +0.552 | +0.022 |
| 4 | +0.585 | +0.541 | +0.585 | +0.547 | +0.038 |
| 5 | +0.577 | +0.541 | +0.583 | +0.545 | +0.038 |
| 6 | +0.581 | +0.542 | +0.588 | +0.545 | +0.043 |
| 7 | +0.578 | +0.539 | +0.584 | +0.542 | +0.043 |
| 8 | +0.581 | +0.539 | +0.589 | +0.547 | +0.042 |

### Verdict

- Velocity-aware symmetric-restricted fit tracks the unconstrained fit (max TEST gap = +0.043).  The per-token spring matrix is consistent with a symmetric Hessian at every layer: the token-direction dynamics is locally Hessian-of-scalar.

## Artefacts

- `tokdir_sarf_shakespeare_results.npz`
- `tokdir_sarf_shakespeare_fig.png`
