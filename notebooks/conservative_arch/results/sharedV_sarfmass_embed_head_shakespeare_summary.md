# Shared-potential fit -- sarfmass_embed_head_shakespeare

Strict conservative-dynamics test.  Joint fit of a *single* scalar network $V_\psi(h)$ plus per-layer scalars $\alpha_\ell, \beta_\ell$, minimising the squared residual

$$\Delta x_\ell - \alpha_\ell v_\ell + \beta_\ell \nabla V_\psi(x_\ell)$$

across **every layer $\ell \geq 1$, every token, every training sentence**.  If a single smooth $V$ can describe all layers' forces, this beats the velocity-only baseline on held-out sentences.

- Hidden dim `d = 128`; layers `L = 8`; samples per split: TRAIN 7,889 / TEST 2,002
- $V_\psi$: 2-layer MLP, hidden = 256, GELU
- Optimiser: AdamW, 4000 steps, bs=2048, lr=0.003

## Overall fit

- TRAIN pooled $R^2$ = **+0.880**
- TEST pooled $R^2$  = **+0.718**

## Per-layer TEST $R^2$

| layer | A. static null | B. velocity-only | C. velocity + shared $V_\psi$ | C - B |
|--:|--:|--:|--:|--:|
| 1 | +0.000 | -0.002 | +0.753 | +0.755 |
| 2 | +0.000 | +0.159 | +0.573 | +0.414 |
| 3 | -0.000 | +0.401 | +0.721 | +0.319 |
| 4 | +0.000 | +0.508 | +0.749 | +0.241 |
| 5 | +0.000 | +0.637 | +0.768 | +0.131 |
| 6 | +0.000 | +0.643 | +0.748 | +0.105 |
| 7 | +0.000 | +0.636 | +0.724 | +0.087 |

## Per-layer TRAIN $R^2$

| layer | A. null | B. velocity-only | C. vel + shared $V_\psi$ |
|--:|--:|--:|--:|
| 1 | +0.000 | +0.001 | +0.927 |
| 2 | +0.000 | +0.207 | +0.844 |
| 3 | +0.000 | +0.455 | +0.851 |
| 4 | +0.000 | +0.539 | +0.870 |
| 5 | +0.000 | +0.628 | +0.876 |
| 6 | +0.000 | +0.658 | +0.862 |
| 7 | +0.000 | +0.632 | +0.850 |

## Learned per-layer scalars

| layer | $\alpha_\ell$ | $\beta_\ell$ |
|--:|--:|--:|
| 1 | +0.3784 | -1.2061 |
| 2 | +0.5364 | -0.8608 |
| 3 | +0.6060 | -0.4239 |
| 4 | +0.6289 | -0.4199 |
| 5 | +0.6000 | -0.4292 |
| 6 | +0.5779 | -0.4371 |
| 7 | +0.5695 | -0.4233 |

## Artefacts

- `sharedV_sarfmass_embed_head_shakespeare_results.npz`
- `sharedV_sarfmass_embed_head_shakespeare_fig.png`
