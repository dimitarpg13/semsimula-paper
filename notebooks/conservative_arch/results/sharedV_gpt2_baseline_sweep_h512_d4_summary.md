# Shared-potential fit -- gpt2_baseline_sweep_h512_d4

Strict conservative-dynamics test.  Joint fit of a *single* scalar network $V_\psi(h)$ plus per-layer scalars $\alpha_\ell, \beta_\ell$, minimising the squared residual

$$\Delta x_\ell - \alpha_\ell v_\ell + \beta_\ell \nabla V_\psi(x_\ell)$$

across **every layer $\ell \geq 1$, every token, every training sentence**.  If a single smooth $V$ can describe all layers' forces, this beats the velocity-only baseline on held-out sentences.

- Hidden dim `d = 768`; layers `L = 12`; samples per split: TRAIN 12,397 / TEST 3,146
- $V_\psi$: 4-layer MLP, hidden = 512, GELU
- Optimiser: AdamW, 3000 steps, bs=2048, lr=0.003

## Overall fit

- TRAIN pooled $R^2$ = **+0.981**
- TEST pooled $R^2$  = **+0.972**

## Per-layer TEST $R^2$

| layer | A. static null | B. velocity-only | C. velocity + shared $V_\psi$ | C - B |
|--:|--:|--:|--:|--:|
| 1 | +0.000 | +0.254 | +0.971 | +0.717 |
| 2 | +0.000 | +0.045 | +0.997 | +0.952 |
| 3 | -0.000 | +0.718 | +0.799 | +0.081 |
| 4 | +0.000 | +0.610 | +0.731 | +0.122 |
| 5 | +0.000 | +0.376 | +0.453 | +0.077 |
| 6 | +0.000 | +0.135 | +0.193 | +0.059 |
| 7 | +0.000 | +0.051 | +0.067 | +0.017 |
| 8 | +0.000 | +0.036 | +0.039 | +0.002 |
| 9 | +0.000 | +0.057 | +0.056 | -0.000 |
| 10 | +0.000 | +0.070 | +0.081 | +0.011 |
| 11 | +0.000 | +0.000 | +0.978 | +0.978 |

## Per-layer TRAIN $R^2$

| layer | A. null | B. velocity-only | C. vel + shared $V_\psi$ |
|--:|--:|--:|--:|
| 1 | +0.000 | +0.262 | +0.982 |
| 2 | +0.000 | +0.046 | +0.998 |
| 3 | +0.000 | +0.721 | +0.806 |
| 4 | +0.000 | +0.621 | +0.740 |
| 5 | +0.000 | +0.380 | +0.461 |
| 6 | -0.000 | +0.135 | +0.196 |
| 7 | +0.000 | +0.048 | +0.068 |
| 8 | +0.000 | +0.032 | +0.036 |
| 9 | +0.000 | +0.056 | +0.055 |
| 10 | +0.000 | +0.073 | +0.084 |
| 11 | +0.000 | +0.000 | +0.989 |

## Learned per-layer scalars

| layer | $\alpha_\ell$ | $\beta_\ell$ |
|--:|--:|--:|
| 1 | +0.0661 | +0.1214 |
| 2 | -0.0489 | +0.2781 |
| 3 | -0.0283 | +0.0399 |
| 4 | +0.0747 | +0.0221 |
| 5 | +0.1098 | +0.0119 |
| 6 | +0.1190 | +0.0074 |
| 7 | +0.1662 | +0.0032 |
| 8 | +0.1977 | +0.0012 |
| 9 | +0.3011 | +0.0005 |
| 10 | +0.3864 | -0.0048 |
| 11 | -0.8542 | -0.5034 |

## Artefacts

- `sharedV_gpt2_baseline_sweep_h512_d4_results.npz`
- `sharedV_gpt2_baseline_sweep_h512_d4_fig.png`
