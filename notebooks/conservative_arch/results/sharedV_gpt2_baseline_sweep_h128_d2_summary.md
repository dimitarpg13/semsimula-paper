# Shared-potential fit -- gpt2_baseline_sweep_h128_d2

Strict conservative-dynamics test.  Joint fit of a *single* scalar network $V_\psi(h)$ plus per-layer scalars $\alpha_\ell, \beta_\ell$, minimising the squared residual

$$\Delta x_\ell - \alpha_\ell v_\ell + \beta_\ell \nabla V_\psi(x_\ell)$$

across **every layer $\ell \geq 1$, every token, every training sentence**.  If a single smooth $V$ can describe all layers' forces, this beats the velocity-only baseline on held-out sentences.

- Hidden dim `d = 768`; layers `L = 12`; samples per split: TRAIN 12,397 / TEST 3,146
- $V_\psi$: 2-layer MLP, hidden = 128, GELU
- Optimiser: AdamW, 3000 steps, bs=2048, lr=0.003

## Overall fit

- TRAIN pooled $R^2$ = **+0.963**
- TEST pooled $R^2$  = **+0.951**

## Per-layer TEST $R^2$

| layer | A. static null | B. velocity-only | C. velocity + shared $V_\psi$ | C - B |
|--:|--:|--:|--:|--:|
| 1 | +0.000 | +0.254 | +0.562 | +0.307 |
| 2 | +0.000 | +0.045 | +0.978 | +0.934 |
| 3 | -0.000 | +0.718 | +0.784 | +0.066 |
| 4 | +0.000 | +0.610 | +0.727 | +0.117 |
| 5 | +0.000 | +0.376 | +0.447 | +0.071 |
| 6 | +0.000 | +0.135 | +0.191 | +0.056 |
| 7 | +0.000 | +0.051 | +0.065 | +0.015 |
| 8 | +0.000 | +0.036 | +0.038 | +0.002 |
| 9 | +0.000 | +0.057 | +0.057 | +0.000 |
| 10 | +0.000 | +0.070 | +0.081 | +0.011 |
| 11 | +0.000 | +0.000 | +0.966 | +0.966 |

## Per-layer TRAIN $R^2$

| layer | A. null | B. velocity-only | C. vel + shared $V_\psi$ |
|--:|--:|--:|--:|
| 1 | +0.000 | +0.262 | +0.843 |
| 2 | +0.000 | +0.046 | +0.988 |
| 3 | +0.000 | +0.721 | +0.792 |
| 4 | +0.000 | +0.621 | +0.734 |
| 5 | +0.000 | +0.380 | +0.453 |
| 6 | -0.000 | +0.135 | +0.192 |
| 7 | +0.000 | +0.048 | +0.064 |
| 8 | +0.000 | +0.032 | +0.034 |
| 9 | +0.000 | +0.056 | +0.056 |
| 10 | +0.000 | +0.073 | +0.084 |
| 11 | +0.000 | +0.000 | +0.971 |

## Learned per-layer scalars

| layer | $\alpha_\ell$ | $\beta_\ell$ |
|--:|--:|--:|
| 1 | +0.4945 | +0.4590 |
| 2 | +2.9513 | -1.5134 |
| 3 | -0.0223 | +0.1820 |
| 4 | +0.0983 | +0.0983 |
| 5 | +0.1292 | +0.0536 |
| 6 | +0.1300 | +0.0306 |
| 7 | +0.1752 | +0.0140 |
| 8 | +0.2028 | +0.0062 |
| 9 | +0.3005 | -0.0040 |
| 10 | +0.3818 | -0.0230 |
| 11 | -0.9792 | -2.3973 |

## Artefacts

- `sharedV_gpt2_baseline_sweep_h128_d2_results.npz`
- `sharedV_gpt2_baseline_sweep_h128_d2_fig.png`
