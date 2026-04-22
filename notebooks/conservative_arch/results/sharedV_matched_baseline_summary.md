# Shared-potential fit -- matched_baseline

Strict conservative-dynamics test.  Joint fit of a *single* scalar network $V_\psi(h)$ plus per-layer scalars $\alpha_\ell, \beta_\ell$, minimising the squared residual

$$\Delta x_\ell - \alpha_\ell v_\ell + \beta_\ell \nabla V_\psi(x_\ell)$$

across **every layer $\ell \geq 1$, every token, every training sentence**.  If a single smooth $V$ can describe all layers' forces, this beats the velocity-only baseline on held-out sentences.

- Hidden dim `d = 128`; layers `L = 8`; samples per split: TRAIN 7,889 / TEST 2,002
- $V_\psi$: 2-layer MLP, hidden = 256, GELU
- Optimiser: AdamW, 4000 steps, bs=2048, lr=0.003

## Overall fit

- TRAIN pooled $R^2$ = **+0.818**
- TEST pooled $R^2$  = **+0.627**

## Per-layer TEST $R^2$

| layer | A. static null | B. velocity-only | C. velocity + shared $V_\psi$ | C - B |
|--:|--:|--:|--:|--:|
| 1 | +0.000 | +0.062 | +0.809 | +0.747 |
| 2 | +0.000 | +0.174 | +0.744 | +0.570 |
| 3 | +0.000 | +0.033 | +0.659 | +0.626 |
| 4 | +0.000 | +0.079 | +0.555 | +0.476 |
| 5 | +0.000 | +0.028 | +0.510 | +0.482 |
| 6 | +0.000 | +0.044 | +0.387 | +0.344 |
| 7 | +0.000 | +0.063 | +0.267 | +0.203 |

## Per-layer TRAIN $R^2$

| layer | A. null | B. velocity-only | C. vel + shared $V_\psi$ |
|--:|--:|--:|--:|
| 1 | +0.000 | +0.067 | +0.870 |
| 2 | +0.000 | +0.178 | +0.887 |
| 3 | +0.000 | +0.039 | +0.802 |
| 4 | +0.000 | +0.082 | +0.774 |
| 5 | +0.000 | +0.027 | +0.725 |
| 6 | +0.000 | +0.050 | +0.769 |
| 7 | +0.000 | +0.067 | +0.697 |

## Learned per-layer scalars

| layer | $\alpha_\ell$ | $\beta_\ell$ |
|--:|--:|--:|
| 1 | +0.2088 | +0.6617 |
| 2 | +0.3573 | +0.6704 |
| 3 | +0.3342 | -0.3076 |
| 4 | +0.1454 | -0.3852 |
| 5 | +0.3187 | +0.2931 |
| 6 | +0.4542 | -0.5496 |
| 7 | +0.3832 | +0.4343 |

## Artefacts

- `sharedV_matched_baseline_results.npz`
- `sharedV_matched_baseline_fig.png`
