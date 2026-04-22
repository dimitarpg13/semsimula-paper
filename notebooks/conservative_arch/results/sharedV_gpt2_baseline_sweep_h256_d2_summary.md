# Shared-potential fit -- gpt2_baseline_sweep_h256_d2

Strict conservative-dynamics test.  Joint fit of a *single* scalar network $V_\psi(h)$ plus per-layer scalars $\alpha_\ell, \beta_\ell$, minimising the squared residual

$$\Delta x_\ell - \alpha_\ell v_\ell + \beta_\ell \nabla V_\psi(x_\ell)$$

across **every layer $\ell \geq 1$, every token, every training sentence**.  If a single smooth $V$ can describe all layers' forces, this beats the velocity-only baseline on held-out sentences.

- Hidden dim `d = 768`; layers `L = 12`; samples per split: TRAIN 12,397 / TEST 3,146
- $V_\psi$: 2-layer MLP, hidden = 256, GELU
- Optimiser: AdamW, 3000 steps, bs=2048, lr=0.003

## Overall fit

- TRAIN pooled $R^2$ = **+0.972**
- TEST pooled $R^2$  = **+0.961**

## Per-layer TEST $R^2$

| layer | A. static null | B. velocity-only | C. velocity + shared $V_\psi$ | C - B |
|--:|--:|--:|--:|--:|
| 1 | +0.000 | +0.254 | +0.776 | +0.522 |
| 2 | +0.000 | +0.045 | +0.981 | +0.937 |
| 3 | -0.000 | +0.718 | +0.783 | +0.065 |
| 4 | +0.000 | +0.610 | +0.730 | +0.121 |
| 5 | +0.000 | +0.376 | +0.457 | +0.080 |
| 6 | +0.000 | +0.135 | +0.196 | +0.061 |
| 7 | +0.000 | +0.051 | +0.067 | +0.016 |
| 8 | +0.000 | +0.036 | +0.039 | +0.002 |
| 9 | +0.000 | +0.057 | +0.057 | +0.000 |
| 10 | +0.000 | +0.070 | +0.081 | +0.010 |
| 11 | +0.000 | +0.000 | +0.975 | +0.975 |

## Per-layer TRAIN $R^2$

| layer | A. null | B. velocity-only | C. vel + shared $V_\psi$ |
|--:|--:|--:|--:|
| 1 | +0.000 | +0.262 | +0.962 |
| 2 | +0.000 | +0.046 | +0.992 |
| 3 | +0.000 | +0.721 | +0.789 |
| 4 | +0.000 | +0.621 | +0.738 |
| 5 | +0.000 | +0.380 | +0.463 |
| 6 | -0.000 | +0.135 | +0.196 |
| 7 | +0.000 | +0.048 | +0.067 |
| 8 | +0.000 | +0.032 | +0.035 |
| 9 | +0.000 | +0.056 | +0.056 |
| 10 | +0.000 | +0.073 | +0.084 |
| 11 | +0.000 | +0.000 | +0.980 |

## Learned per-layer scalars

| layer | $\alpha_\ell$ | $\beta_\ell$ |
|--:|--:|--:|
| 1 | +0.0809 | -0.4729 |
| 2 | +2.8684 | +0.9793 |
| 3 | -0.0236 | -0.0877 |
| 4 | +0.1203 | -0.0475 |
| 5 | +0.1137 | -0.0268 |
| 6 | +0.1186 | -0.0157 |
| 7 | +0.1727 | -0.0075 |
| 8 | +0.2022 | -0.0032 |
| 9 | +0.3000 | +0.0018 |
| 10 | +0.3841 | +0.0119 |
| 11 | -0.9542 | +1.2045 |

## Artefacts

- `sharedV_gpt2_baseline_sweep_h256_d2_results.npz`
- `sharedV_gpt2_baseline_sweep_h256_d2_fig.png`
