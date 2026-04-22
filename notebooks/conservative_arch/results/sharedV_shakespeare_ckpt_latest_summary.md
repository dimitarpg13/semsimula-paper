# Shared-potential fit -- shakespeare_ckpt_latest

Strict conservative-dynamics test.  Joint fit of a *single* scalar network $V_\psi(h)$ plus per-layer scalars $\alpha_\ell, \beta_\ell$, minimising the squared residual

$$\Delta x_\ell - \alpha_\ell v_\ell + \beta_\ell \nabla V_\psi(x_\ell)$$

across **every layer $\ell \geq 1$, every token, every training sentence**.  If a single smooth $V$ can describe all layers' forces, this beats the velocity-only baseline on held-out sentences.

- Hidden dim `d = 128`; layers `L = 8`; samples per split: TRAIN 7,889 / TEST 2,002
- $V_\psi$: 2-layer MLP, hidden = 256, GELU
- Optimiser: AdamW, 4000 steps, bs=2048, lr=0.003

## Overall fit

- TRAIN pooled $R^2$ = **+0.933**
- TEST pooled $R^2$  = **+0.790**

## Per-layer TEST $R^2$

| layer | A. static null | B. velocity-only | C. velocity + shared $V_\psi$ | C - B |
|--:|--:|--:|--:|--:|
| 1 | +0.000 | +0.185 | +0.970 | +0.785 |
| 2 | +0.000 | +0.239 | +0.668 | +0.428 |
| 3 | +0.000 | +0.131 | +0.819 | +0.688 |
| 4 | +0.000 | +0.032 | +0.280 | +0.248 |
| 5 | +0.000 | +0.048 | +0.901 | +0.854 |
| 6 | +0.000 | +0.465 | +0.927 | +0.462 |
| 7 | +0.000 | +0.447 | +0.905 | +0.459 |

## Per-layer TRAIN $R^2$

| layer | A. null | B. velocity-only | C. vel + shared $V_\psi$ |
|--:|--:|--:|--:|
| 1 | +0.000 | +0.209 | +0.983 |
| 2 | +0.000 | +0.218 | +0.768 |
| 3 | +0.000 | +0.121 | +0.951 |
| 4 | +0.000 | +0.028 | +0.792 |
| 5 | +0.000 | +0.051 | +0.960 |
| 6 | +0.000 | +0.467 | +0.974 |
| 7 | +0.000 | +0.412 | +0.970 |

## Learned per-layer scalars

| layer | $\alpha_\ell$ | $\beta_\ell$ |
|--:|--:|--:|
| 1 | +0.7385 | +0.2605 |
| 2 | +0.4018 | -0.1948 |
| 3 | -0.5332 | +0.8231 |
| 4 | +0.3920 | -0.7538 |
| 5 | +0.1646 | -0.3403 |
| 6 | +0.5289 | -0.1657 |
| 7 | +0.2395 | -0.2077 |

## Artefacts

- `sharedV_shakespeare_ckpt_latest_results.npz`
- `sharedV_shakespeare_ckpt_latest_fig.png`
