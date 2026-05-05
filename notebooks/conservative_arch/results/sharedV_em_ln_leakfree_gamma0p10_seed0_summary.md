# Shared-potential fit -- em_ln_leakfree_gamma0p10_seed0

Strict conservative-dynamics test.  Joint fit of a *single* scalar network $V_\psi(h)$ plus per-layer scalars $\alpha_\ell, \beta_\ell$, minimising the squared residual

$$\Delta x_\ell - \alpha_\ell v_\ell + \beta_\ell \nabla V_\psi(x_\ell)$$

across **every layer $\ell \geq 1$, every token, every training sentence**.  If a single smooth $V$ can describe all layers' forces, this beats the velocity-only baseline on held-out sentences.

- Hidden dim `d = 128`; layers `L = 8`; samples per split: TRAIN 3,913 / TEST 3,955
- $V_\psi$: 2-layer MLP, hidden = 256, GELU
- Optimiser: AdamW, 4000 steps, bs=2048, lr=0.003

## Overall fit

- TRAIN pooled $R^2$ = **+0.987**
- TEST pooled $R^2$  = **+0.943**

## Per-layer TEST $R^2$

| layer | A. static null | B. velocity-only | C. velocity + shared $V_\psi$ | C - B |
|--:|--:|--:|--:|--:|
| 1 | +0.000 | +0.948 | +0.960 | +0.012 |
| 2 | +0.000 | +0.887 | +0.949 | +0.063 |
| 3 | +0.000 | +0.867 | +0.950 | +0.084 |
| 4 | +0.000 | +0.858 | +0.951 | +0.092 |
| 5 | +0.000 | +0.851 | +0.948 | +0.098 |
| 6 | +0.000 | +0.829 | +0.941 | +0.113 |
| 7 | +0.000 | +0.820 | +0.925 | +0.104 |

## Per-layer TRAIN $R^2$

| layer | A. null | B. velocity-only | C. vel + shared $V_\psi$ |
|--:|--:|--:|--:|
| 1 | +0.000 | +0.944 | +0.964 |
| 2 | +0.000 | +0.883 | +0.966 |
| 3 | +0.000 | +0.870 | +0.980 |
| 4 | +0.000 | +0.851 | +0.986 |
| 5 | +0.000 | +0.843 | +0.990 |
| 6 | +0.000 | +0.824 | +0.991 |
| 7 | +0.000 | +0.821 | +0.991 |

## Learned per-layer scalars

| layer | $\alpha_\ell$ | $\beta_\ell$ |
|--:|--:|--:|
| 1 | +1.8458 | -0.0390 |
| 2 | +1.3016 | -0.1053 |
| 3 | +1.1615 | -0.1841 |
| 4 | +1.0804 | -0.2728 |
| 5 | +1.0052 | -0.3748 |
| 6 | +0.8988 | -0.4558 |
| 7 | +0.8452 | -0.4687 |

## Artefacts

- `sharedV_em_ln_leakfree_gamma0p10_seed0_results.npz`
- `sharedV_em_ln_leakfree_gamma0p10_seed0_fig.png`
