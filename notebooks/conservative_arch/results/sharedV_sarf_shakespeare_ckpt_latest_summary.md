# Shared-potential fit -- sarf_shakespeare_ckpt_latest

Strict conservative-dynamics test.  Joint fit of a *single* scalar network $V_\psi(h)$ plus per-layer scalars $\alpha_\ell, \beta_\ell$, minimising the squared residual

$$\Delta x_\ell - \alpha_\ell v_\ell + \beta_\ell \nabla V_\psi(x_\ell)$$

across **every layer $\ell \geq 1$, every token, every training sentence**.  If a single smooth $V$ can describe all layers' forces, this beats the velocity-only baseline on held-out sentences.

- Hidden dim `d = 128`; layers `L = 8`; samples per split: TRAIN 7,889 / TEST 2,002
- $V_\psi$: 2-layer MLP, hidden = 256, GELU
- Optimiser: AdamW, 4000 steps, bs=2048, lr=0.003

## Overall fit

- TRAIN pooled $R^2$ = **+0.812**
- TEST pooled $R^2$  = **+0.713**

## Per-layer TEST $R^2$

| layer | A. static null | B. velocity-only | C. velocity + shared $V_\psi$ | C - B |
|--:|--:|--:|--:|--:|
| 1 | +0.000 | +0.018 | +0.476 | +0.458 |
| 2 | +0.000 | +0.150 | +0.615 | +0.465 |
| 3 | +0.000 | +0.212 | +0.673 | +0.461 |
| 4 | +0.000 | +0.507 | +0.810 | +0.303 |
| 5 | +0.000 | +0.605 | +0.777 | +0.172 |
| 6 | +0.000 | +0.658 | +0.810 | +0.152 |
| 7 | +0.000 | +0.740 | +0.826 | +0.086 |

## Per-layer TRAIN $R^2$

| layer | A. null | B. velocity-only | C. vel + shared $V_\psi$ |
|--:|--:|--:|--:|
| 1 | +0.000 | +0.029 | +0.691 |
| 2 | +0.000 | +0.159 | +0.764 |
| 3 | +0.000 | +0.197 | +0.800 |
| 4 | +0.000 | +0.483 | +0.872 |
| 5 | +0.000 | +0.637 | +0.869 |
| 6 | -0.000 | +0.644 | +0.854 |
| 7 | +0.000 | +0.651 | +0.849 |

## Learned per-layer scalars

| layer | $\alpha_\ell$ | $\beta_\ell$ |
|--:|--:|--:|
| 1 | +0.5567 | +0.9986 |
| 2 | +0.5163 | +0.6837 |
| 3 | +0.5166 | +0.5412 |
| 4 | +0.5169 | +0.4255 |
| 5 | +0.5175 | +0.3458 |
| 6 | +0.5068 | +0.3527 |
| 7 | +0.5128 | +0.3297 |

## Artefacts

- `sharedV_sarf_shakespeare_ckpt_latest_results.npz`
- `sharedV_sarf_shakespeare_ckpt_latest_fig.png`
