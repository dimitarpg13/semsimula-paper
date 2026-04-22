# Shared-potential fit -- gpt2_baseline

Strict conservative-dynamics test.  Joint fit of a *single* scalar network $V_\psi(h)$ plus per-layer scalars $\alpha_\ell, \beta_\ell$, minimising the squared residual

$$\Delta x_\ell - \alpha_\ell v_\ell + \beta_\ell \nabla V_\psi(x_\ell)$$

across **every layer $\ell \geq 1$, every token, every training sentence**.  If a single smooth $V$ can describe all layers' forces, this beats the velocity-only baseline on held-out sentences.

- Hidden dim `d = 768`; layers `L = 12`; samples per split: TRAIN 12,397 / TEST 3,146
- $V_\psi$: 2-layer MLP, hidden = 256, GELU
- Optimiser: AdamW, 4000 steps, bs=2048, lr=0.003

## Overall fit

- TRAIN pooled $R^2$ = **+0.975**
- TEST pooled $R^2$  = **+0.964**

## Per-layer TEST $R^2$

| layer | A. static null | B. velocity-only | C. velocity + shared $V_\psi$ | C - B |
|--:|--:|--:|--:|--:|
| 1 | +0.000 | +0.254 | +0.715 | +0.461 |
| 2 | +0.000 | +0.045 | +0.990 | +0.945 |
| 3 | -0.000 | +0.718 | +0.795 | +0.077 |
| 4 | +0.000 | +0.610 | +0.731 | +0.121 |
| 5 | +0.000 | +0.376 | +0.452 | +0.076 |
| 6 | +0.000 | +0.135 | +0.193 | +0.059 |
| 7 | +0.000 | +0.051 | +0.066 | +0.016 |
| 8 | +0.000 | +0.036 | +0.039 | +0.002 |
| 9 | +0.000 | +0.057 | +0.057 | +0.000 |
| 10 | +0.000 | +0.070 | +0.081 | +0.011 |
| 11 | +0.000 | +0.000 | +0.977 | +0.977 |

## Per-layer TRAIN $R^2$

| layer | A. null | B. velocity-only | C. vel + shared $V_\psi$ |
|--:|--:|--:|--:|
| 1 | +0.000 | +0.262 | +0.905 |
| 2 | +0.000 | +0.046 | +0.996 |
| 3 | +0.000 | +0.721 | +0.801 |
| 4 | +0.000 | +0.621 | +0.739 |
| 5 | +0.000 | +0.380 | +0.458 |
| 6 | -0.000 | +0.135 | +0.195 |
| 7 | +0.000 | +0.048 | +0.066 |
| 8 | +0.000 | +0.032 | +0.035 |
| 9 | +0.000 | +0.056 | +0.056 |
| 10 | +0.000 | +0.073 | +0.084 |
| 11 | +0.000 | +0.000 | +0.983 |

## Learned per-layer scalars

| layer | $\alpha_\ell$ | $\beta_\ell$ |
|--:|--:|--:|
| 1 | +0.5154 | +0.1430 |
| 2 | -0.0435 | +0.2579 |
| 3 | -0.0231 | -0.1027 |
| 4 | +0.0910 | -0.0575 |
| 5 | +0.1178 | -0.0326 |
| 6 | +0.1304 | -0.0172 |
| 7 | +0.1715 | -0.0097 |
| 8 | +0.1985 | -0.0040 |
| 9 | +0.2993 | +0.0016 |
| 10 | +0.3987 | +0.0133 |
| 11 | -0.9418 | +1.3749 |

## Artefacts

- `sharedV_gpt2_baseline_results.npz`
- `sharedV_gpt2_baseline_fig.png`
