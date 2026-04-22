# Shared-potential fit -- gpt2_baseline_sweep_h512_d2

Strict conservative-dynamics test.  Joint fit of a *single* scalar network $V_\psi(h)$ plus per-layer scalars $\alpha_\ell, \beta_\ell$, minimising the squared residual

$$\Delta x_\ell - \alpha_\ell v_\ell + \beta_\ell \nabla V_\psi(x_\ell)$$

across **every layer $\ell \geq 1$, every token, every training sentence**.  If a single smooth $V$ can describe all layers' forces, this beats the velocity-only baseline on held-out sentences.

- Hidden dim `d = 768`; layers `L = 12`; samples per split: TRAIN 12,397 / TEST 3,146
- $V_\psi$: 2-layer MLP, hidden = 512, GELU
- Optimiser: AdamW, 3000 steps, bs=2048, lr=0.003

## Overall fit

- TRAIN pooled $R^2$ = **+0.980**
- TEST pooled $R^2$  = **+0.969**

## Per-layer TEST $R^2$

| layer | A. static null | B. velocity-only | C. velocity + shared $V_\psi$ | C - B |
|--:|--:|--:|--:|--:|
| 1 | +0.000 | +0.254 | +0.815 | +0.560 |
| 2 | +0.000 | +0.045 | +0.997 | +0.952 |
| 3 | -0.000 | +0.718 | +0.798 | +0.080 |
| 4 | +0.000 | +0.610 | +0.731 | +0.121 |
| 5 | +0.000 | +0.376 | +0.451 | +0.075 |
| 6 | +0.000 | +0.135 | +0.193 | +0.059 |
| 7 | +0.000 | +0.051 | +0.067 | +0.016 |
| 8 | +0.000 | +0.036 | +0.039 | +0.002 |
| 9 | +0.000 | +0.057 | +0.057 | +0.000 |
| 10 | +0.000 | +0.070 | +0.081 | +0.011 |
| 11 | +0.000 | +0.000 | +0.978 | +0.978 |

## Per-layer TRAIN $R^2$

| layer | A. null | B. velocity-only | C. vel + shared $V_\psi$ |
|--:|--:|--:|--:|
| 1 | +0.000 | +0.262 | +0.975 |
| 2 | +0.000 | +0.046 | +0.998 |
| 3 | +0.000 | +0.721 | +0.805 |
| 4 | +0.000 | +0.621 | +0.739 |
| 5 | +0.000 | +0.380 | +0.458 |
| 6 | -0.000 | +0.135 | +0.195 |
| 7 | +0.000 | +0.048 | +0.067 |
| 8 | +0.000 | +0.032 | +0.035 |
| 9 | +0.000 | +0.056 | +0.056 |
| 10 | +0.000 | +0.073 | +0.084 |
| 11 | +0.000 | +0.000 | +0.988 |

## Learned per-layer scalars

| layer | $\alpha_\ell$ | $\beta_\ell$ |
|--:|--:|--:|
| 1 | +0.0765 | -0.2418 |
| 2 | -0.3510 | -0.5492 |
| 3 | -0.0281 | -0.0834 |
| 4 | +0.0835 | -0.0448 |
| 5 | +0.1186 | -0.0254 |
| 6 | +0.1205 | -0.0139 |
| 7 | +0.1695 | -0.0066 |
| 8 | +0.2014 | -0.0029 |
| 9 | +0.3008 | +0.0009 |
| 10 | +0.3859 | +0.0106 |
| 11 | -0.9047 | +1.0770 |

## Artefacts

- `sharedV_gpt2_baseline_sweep_h512_d2_results.npz`
- `sharedV_gpt2_baseline_sweep_h512_d2_fig.png`
