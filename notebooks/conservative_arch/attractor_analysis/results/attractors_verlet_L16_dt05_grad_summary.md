# Semantic-attractor extraction -- verlet_L16_dt05_grad

- Variant: `symplectic`
- Model config: `d=128, L=16, mass_mode=logfreq`
- Seeds: 128 Gaussian + 128 token-embedding + 128 perturbed real-$h$
- Descent: Adam lr=0.05, 1500 steps
- Convergence filter: $\|\nabla V\| < 0.05$
- Clustering: K-means, silhouette-sweep K $\in$ [2, 8]

## Prompt (narrative): *"The old king sat on the"*

- Converged: 359/384
- $K^\ast = 2$   silhouette scores: K=2: +0.939, K=3: +0.918, K=4: +0.878, K=5: +0.867, K=6: +0.869, K=7: +0.867, K=8: +0.858

**Real next-token (tied LM head on $h_L$ of prompt):**  ` d`·0.06, `t`·0.03, ` the`·0.02, `oth`·0.01, ` king`·0.01, `\n`·0.01, `,`·0.01, ` a`·0.01, ` world`·0.01, ` blood`·0.01

| Attractor | Size | $\langle V\rangle$ | Top-10 decoded tokens |
|---|---:|---:|---|
| A0 | 358 | -297.988 | `,`·0.08, ` the`·0.06, `\n`·0.06, ` a`·0.03, `-`·0.03, ` d`·0.02, ` of`·0.01, ` that`·0.01, ` in`·0.01, ` his`·0.01 |
| A1 | 1 | -297.971 | `,`·0.08, ` the`·0.06, `\n`·0.06, ` a`·0.03, `-`·0.03, ` d`·0.02, ` of`·0.01, ` that`·0.01, ` in`·0.01, ` his`·0.01 |

## Prompt (mathematics): *"The theorem states that for every"*

- Converged: 364/384
- $K^\ast = 3$   silhouette scores: K=2: +0.940, K=3: +0.941, K=4: +0.898, K=5: +0.901, K=6: +0.903, K=7: +0.893, K=8: +0.879

**Real next-token (tied LM head on $h_L$ of prompt):**  ` the`·0.04, ` a`·0.04, `,`·0.02, `t`·0.02, `-`·0.01, `\n`·0.01, ` his`·0.01, ` d`·0.01, ` one`·0.01, ` my`·0.01

| Attractor | Size | $\langle V\rangle$ | Top-10 decoded tokens |
|---|---:|---:|---|
| A0 | 362 | -300.903 | `,`·0.08, ` the`·0.07, `\n`·0.06, ` a`·0.04, `-`·0.03, ` d`·0.02, ` of`·0.01, ` that`·0.01, ` in`·0.01, ` his`·0.01 |
| A1 | 1 | -300.891 | `,`·0.08, ` the`·0.07, `\n`·0.06, ` a`·0.04, `-`·0.03, ` d`·0.02, ` of`·0.01, ` that`·0.01, ` in`·0.01, ` his`·0.01 |
| A2 | 1 | -300.892 | `,`·0.08, ` the`·0.07, `\n`·0.06, ` a`·0.04, `-`·0.03, ` d`·0.02, ` of`·0.01, ` that`·0.01, ` in`·0.01, ` his`·0.01 |

## Prompt (scientific): *"Photosynthesis converts carbon dioxide and"*

- Converged: 360/384
- $K^\ast = 3$   silhouette scores: K=2: +0.906, K=3: +0.910, K=4: +0.905, K=5: +0.907, K=6: +0.905, K=7: +0.890, K=8: +0.873

**Real next-token (tied LM head on $h_L$ of prompt):**  `oth`·0.22, `UM`·0.21, `ER`·0.08, ` d`·0.07, `EN`·0.06, `A`·0.05, `O`·0.04, `:`·0.03, `all`·0.02, `are`·0.02

| Attractor | Size | $\langle V\rangle$ | Top-10 decoded tokens |
|---|---:|---:|---|
| A0 | 357 | -253.545 | `\n`·0.15, `,`·0.14, ` the`·0.09, ` a`·0.04, `-`·0.03, ` of`·0.02, ` d`·0.02, ` in`·0.02, ` that`·0.01, ` his`·0.01 |
| A1 | 2 | -253.548 | `\n`·0.15, `,`·0.14, ` the`·0.09, ` a`·0.04, `-`·0.03, ` of`·0.02, ` d`·0.02, ` in`·0.02, ` that`·0.01, ` his`·0.01 |
| A2 | 1 | -253.551 | `\n`·0.15, `,`·0.14, ` the`·0.09, ` a`·0.04, `-`·0.03, ` of`·0.02, ` d`·0.02, ` in`·0.02, ` that`·0.01, ` his`·0.01 |

## Prompt (dialogue): *"She whispered: I love"*

- Converged: 363/384
- $K^\ast = 2$   silhouette scores: K=2: +0.940, K=3: +0.923, K=4: +0.904, K=5: +0.862, K=6: +0.824, K=7: +0.827, K=8: +0.829

**Real next-token (tied LM head on $h_L$ of prompt):**  `\n`·0.65, ` the`·0.11, `-`·0.07, ` a`·0.02, `'s`·0.02, `,`·0.02, `'`·0.01, ` of`·0.01, ` to`·0.01, ` my`·0.00

| Attractor | Size | $\langle V\rangle$ | Top-10 decoded tokens |
|---|---:|---:|---|
| A0 | 362 | -266.078 | `,`·0.08, ` the`·0.07, `\n`·0.07, ` a`·0.04, `-`·0.03, ` d`·0.02, ` of`·0.01, ` that`·0.01, ` his`·0.01, ` in`·0.01 |
| A1 | 1 | -266.068 | `,`·0.08, ` the`·0.07, `\n`·0.07, ` a`·0.04, `-`·0.03, ` d`·0.02, ` of`·0.01, ` that`·0.01, ` his`·0.01, ` in`·0.01 |

## Prompt (code): *"def fibonacci(n): return 1 if n < 2 else"*

- Converged: 361/384
- $K^\ast = 3$   silhouette scores: K=2: +0.890, K=3: +0.892, K=4: +0.882, K=5: +0.817, K=6: +0.877, K=7: +0.879, K=8: +0.829

**Real next-token (tied LM head on $h_L$ of prompt):**  `\n`·0.91, `,`·0.04, `.`·0.02, `:`·0.01, ` of`·0.01, ` to`·0.00, `;`·0.00, ` in`·0.00, ` with`·0.00, ` and`·0.00

| Attractor | Size | $\langle V\rangle$ | Top-10 decoded tokens |
|---|---:|---:|---|
| A0 | 357 | -261.021 | `\n`·0.17, `,`·0.16, ` the`·0.09, ` a`·0.04, `-`·0.03, ` of`·0.02, ` d`·0.02, ` in`·0.02, ` that`·0.01, ` to`·0.01 |
| A1 | 3 | -261.021 | `\n`·0.17, `,`·0.16, ` the`·0.09, ` a`·0.04, `-`·0.03, ` of`·0.02, ` d`·0.02, ` in`·0.02, ` that`·0.01, ` to`·0.01 |
| A2 | 1 | -261.023 | `\n`·0.17, `,`·0.16, ` the`·0.09, ` a`·0.04, `-`·0.03, ` of`·0.02, ` d`·0.02, ` in`·0.02, ` that`·0.01, ` to`·0.01 |
