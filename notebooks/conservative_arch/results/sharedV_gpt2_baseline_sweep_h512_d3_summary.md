# Shared-potential fit -- gpt2_baseline_sweep_h512_d3

Strict conservative-dynamics test.  Joint fit of a *single* scalar network $V_\psi(h)$ plus per-layer scalars $\alpha_\ell, \beta_\ell$, minimising the squared residual

$$\Delta x_\ell - \alpha_\ell v_\ell + \beta_\ell \nabla V_\psi(x_\ell)$$

across **every layer $\ell \geq 1$, every token, every training sentence**.  If a single smooth $V$ can describe all layers' forces, this beats the velocity-only baseline on held-out sentences.

- Hidden dim `d = 768`; layers `L = 12`; samples per split: TRAIN 12,397 / TEST 3,146
- $V_\psi$: 3-layer MLP, hidden = 512, GELU
- Optimiser: AdamW, 3000 steps, bs=2048, lr=0.003

## Overall fit

- TRAIN pooled $R^2$ = **+0.973**
- TEST pooled $R^2$  = **+0.967**

## Per-layer TEST $R^2$

| layer | A. static null | B. velocity-only | C. velocity + shared $V_\psi$ | C - B |
|--:|--:|--:|--:|--:|
| 1 | +0.000 | +0.254 | +0.965 | +0.710 |
| 2 | +0.000 | +0.045 | +0.994 | +0.949 |
| 3 | -0.000 | +0.718 | +0.789 | +0.071 |
| 4 | +0.000 | +0.610 | +0.728 | +0.118 |
| 5 | +0.000 | +0.376 | +0.451 | +0.075 |
| 6 | +0.000 | +0.135 | +0.193 | +0.058 |
| 7 | +0.000 | +0.051 | +0.065 | +0.014 |
| 8 | +0.000 | +0.036 | +0.039 | +0.002 |
| 9 | +0.000 | +0.057 | +0.056 | -0.001 |
| 10 | +0.000 | +0.070 | +0.080 | +0.010 |
| 11 | +0.000 | +0.000 | +0.972 | +0.971 |

## Per-layer TRAIN $R^2$

| layer | A. null | B. velocity-only | C. vel + shared $V_\psi$ |
|--:|--:|--:|--:|
| 1 | +0.000 | +0.262 | +0.971 |
| 2 | +0.000 | +0.046 | +0.996 |
| 3 | +0.000 | +0.721 | +0.794 |
| 4 | +0.000 | +0.621 | +0.737 |
| 5 | +0.000 | +0.380 | +0.459 |
| 6 | -0.000 | +0.135 | +0.195 |
| 7 | +0.000 | +0.048 | +0.066 |
| 8 | +0.000 | +0.032 | +0.035 |
| 9 | +0.000 | +0.056 | +0.055 |
| 10 | +0.000 | +0.073 | +0.083 |
| 11 | +0.000 | +0.000 | +0.979 |

## Learned per-layer scalars

| layer | $\alpha_\ell$ | $\beta_\ell$ |
|--:|--:|--:|
| 1 | +0.0829 | +0.1311 |
| 2 | -0.0498 | +0.3364 |
| 3 | -0.0281 | +0.0440 |
| 4 | +0.0828 | +0.0229 |
| 5 | +0.1143 | +0.0128 |
| 6 | +0.1181 | +0.0067 |
| 7 | +0.1678 | +0.0027 |
| 8 | +0.2000 | +0.0016 |
| 9 | +0.3011 | +0.0005 |
| 10 | +0.3859 | -0.0045 |
| 11 | -0.8661 | -0.5555 |

## Artefacts

- `sharedV_gpt2_baseline_sweep_h512_d3_results.npz`
- `sharedV_gpt2_baseline_sweep_h512_d3_fig.png`
