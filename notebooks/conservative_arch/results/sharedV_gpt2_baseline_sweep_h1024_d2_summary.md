# Shared-potential fit -- gpt2_baseline_sweep_h1024_d2

Strict conservative-dynamics test.  Joint fit of a *single* scalar network $V_\psi(h)$ plus per-layer scalars $\alpha_\ell, \beta_\ell$, minimising the squared residual

$$\Delta x_\ell - \alpha_\ell v_\ell + \beta_\ell \nabla V_\psi(x_\ell)$$

across **every layer $\ell \geq 1$, every token, every training sentence**.  If a single smooth $V$ can describe all layers' forces, this beats the velocity-only baseline on held-out sentences.

- Hidden dim `d = 768`; layers `L = 12`; samples per split: TRAIN 12,397 / TEST 3,146
- $V_\psi$: 2-layer MLP, hidden = 1024, GELU
- Optimiser: AdamW, 3000 steps, bs=2048, lr=0.003

## Overall fit

- TRAIN pooled $R^2$ = **+0.983**
- TEST pooled $R^2$  = **+0.971**

## Per-layer TEST $R^2$

| layer | A. static null | B. velocity-only | C. velocity + shared $V_\psi$ | C - B |
|--:|--:|--:|--:|--:|
| 1 | +0.000 | +0.254 | +0.917 | +0.663 |
| 2 | +0.000 | +0.045 | +0.997 | +0.952 |
| 3 | -0.000 | +0.718 | +0.799 | +0.081 |
| 4 | +0.000 | +0.610 | +0.732 | +0.122 |
| 5 | +0.000 | +0.376 | +0.454 | +0.078 |
| 6 | +0.000 | +0.135 | +0.195 | +0.061 |
| 7 | +0.000 | +0.051 | +0.068 | +0.017 |
| 8 | +0.000 | +0.036 | +0.039 | +0.003 |
| 9 | +0.000 | +0.057 | +0.057 | +0.000 |
| 10 | +0.000 | +0.070 | +0.081 | +0.011 |
| 11 | +0.000 | +0.000 | +0.978 | +0.978 |

## Per-layer TRAIN $R^2$

| layer | A. null | B. velocity-only | C. vel + shared $V_\psi$ |
|--:|--:|--:|--:|
| 1 | +0.000 | +0.262 | +0.977 |
| 2 | +0.000 | +0.046 | +0.998 |
| 3 | +0.000 | +0.721 | +0.808 |
| 4 | +0.000 | +0.621 | +0.742 |
| 5 | +0.000 | +0.380 | +0.462 |
| 6 | -0.000 | +0.135 | +0.198 |
| 7 | +0.000 | +0.048 | +0.069 |
| 8 | +0.000 | +0.032 | +0.036 |
| 9 | +0.000 | +0.056 | +0.055 |
| 10 | +0.000 | +0.073 | +0.084 |
| 11 | +0.000 | +0.000 | +0.993 |

## Learned per-layer scalars

| layer | $\alpha_\ell$ | $\beta_\ell$ |
|--:|--:|--:|
| 1 | +0.0522 | +0.1872 |
| 2 | -0.3642 | +0.4828 |
| 3 | -0.0289 | +0.0568 |
| 4 | +0.0690 | +0.0306 |
| 5 | +0.1027 | +0.0175 |
| 6 | +0.1133 | +0.0102 |
| 7 | +0.1647 | +0.0050 |
| 8 | +0.1978 | +0.0025 |
| 9 | +0.3003 | -0.0009 |
| 10 | +0.3880 | -0.0069 |
| 11 | -0.8126 | -0.7217 |

## Artefacts

- `sharedV_gpt2_baseline_sweep_h1024_d2_results.npz`
- `sharedV_gpt2_baseline_sweep_h1024_d2_fig.png`
