# Shared-potential fit -- sarfmass_logfreq_shakespeare

Strict conservative-dynamics test.  Joint fit of a *single* scalar network $V_\psi(h)$ plus per-layer scalars $\alpha_\ell, \beta_\ell$, minimising the squared residual

$$\Delta x_\ell - \alpha_\ell v_\ell + \beta_\ell \nabla V_\psi(x_\ell)$$

across **every layer $\ell \geq 1$, every token, every training sentence**.  If a single smooth $V$ can describe all layers' forces, this beats the velocity-only baseline on held-out sentences.

- Hidden dim `d = 128`; layers `L = 8`; samples per split: TRAIN 7,889 / TEST 2,002
- $V_\psi$: 2-layer MLP, hidden = 256, GELU
- Optimiser: AdamW, 4000 steps, bs=2048, lr=0.003

## Overall fit

- TRAIN pooled $R^2$ = **+0.911**
- TEST pooled $R^2$  = **+0.837**

## Per-layer TEST $R^2$

| layer | A. static null | B. velocity-only | C. velocity + shared $V_\psi$ | C - B |
|--:|--:|--:|--:|--:|
| 1 | +0.000 | +0.227 | +0.893 | +0.666 |
| 2 | +0.000 | +0.472 | +0.876 | +0.403 |
| 3 | +0.000 | +0.526 | +0.867 | +0.341 |
| 4 | +0.000 | +0.662 | +0.883 | +0.221 |
| 5 | +0.000 | +0.718 | +0.852 | +0.134 |
| 6 | +0.000 | +0.498 | +0.751 | +0.254 |
| 7 | +0.000 | +0.319 | +0.670 | +0.351 |

## Per-layer TRAIN $R^2$

| layer | A. null | B. velocity-only | C. vel + shared $V_\psi$ |
|--:|--:|--:|--:|
| 1 | +0.000 | +0.228 | +0.964 |
| 2 | +0.000 | +0.436 | +0.923 |
| 3 | +0.000 | +0.541 | +0.913 |
| 4 | +0.000 | +0.657 | +0.912 |
| 5 | +0.000 | +0.699 | +0.899 |
| 6 | +0.000 | +0.504 | +0.874 |
| 7 | +0.000 | +0.344 | +0.835 |

## Learned per-layer scalars

| layer | $\alpha_\ell$ | $\beta_\ell$ |
|--:|--:|--:|
| 1 | +0.6619 | -0.9543 |
| 2 | +0.7024 | -0.6879 |
| 3 | +0.6983 | -0.6089 |
| 4 | +0.6737 | -0.6437 |
| 5 | +0.5752 | -0.7174 |
| 6 | +0.3571 | -0.9183 |
| 7 | +0.3122 | -0.8406 |

## Artefacts

- `sharedV_sarfmass_logfreq_shakespeare_results.npz`
- `sharedV_sarfmass_logfreq_shakespeare_fig.png`
