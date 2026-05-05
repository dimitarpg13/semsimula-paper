# Semantic-attractor extraction -- em_ln_leakfree_freegamma_seed0

- Variant: `sarf_mass_ln`
- Model config: `d=128, L=8, mass_mode=logfreq`
- Seeds: 256 Gaussian + 256 token-embedding + 256 perturbed real-$h$
- Descent: Adam lr=0.05, 2000 steps
- Convergence filter: $\|\nabla V\| < 0.05$
- Clustering: K-means, silhouette-sweep K $\in$ [2, 12]

## Prompt (narrative): *"The old king sat on the"*

- Converged: 0/768
- $K^\ast = 3$   silhouette scores: K=2: +0.618, K=3: +0.637, K=4: +0.586, K=5: +0.619, K=6: +0.585, K=7: +0.595, K=8: +0.563, K=9: +0.567, K=10: +0.540, K=11: +0.551, K=12: +0.561

**Real next-token (tied LM head on $h_L$ of prompt):**  ` king`·0.02, ` world`·0.01, ` the`·0.01, ` crown`·0.01, `\n`·0.01, ` people`·0.01, ` earth`·0.01, ` sun`·0.01, ` queen`·0.01, ` rest`·0.01

| Attractor | Size | $\langle V\rangle$ | Top-10 decoded tokens |
|---|---:|---:|---|
| A0 | 302 | -990.999 | `\n`·1.00, `:`·0.00, `'`·0.00, `.`·0.00, `&`·0.00, `,`·0.00, `+`·0.00, `#`·0.00, `-`·0.00, `%`·0.00 |
| A1 | 292 | -332.723 | `\n`·1.00, `:`·0.00, ` and`·0.00, ` in`·0.00, `;`·0.00, `,`·0.00, ` his`·0.00, ` of`·0.00, ` with`·0.00, ` the`·0.00 |
| A2 | 174 | -827.898 | `\n`·1.00, `:`·0.00, ` with`·0.00, ` of`·0.00, ` and`·0.00, ` the`·0.00, ` a`·0.00, ` for`·0.00, ` in`·0.00, `;`·0.00 |

## Prompt (mathematics): *"The theorem states that for every"*

- Converged: 0/768
- $K^\ast = 2$   silhouette scores: K=2: +0.714, K=3: +0.710, K=4: +0.673, K=5: +0.687, K=6: +0.687, K=7: +0.675, K=8: +0.674, K=9: +0.676, K=10: +0.682, K=11: +0.688, K=12: +0.690

**Real next-token (tied LM head on $h_L$ of prompt):**  ` day`·0.03, ` thing`·0.03, ` one`·0.02, ` word`·0.01, ` man`·0.01, ` hour`·0.01, ` every`·0.01, ` time`·0.01, ` part`·0.00, ` way`·0.00

| Attractor | Size | $\langle V\rangle$ | Top-10 decoded tokens |
|---|---:|---:|---|
| A0 | 337 | -948.848 | `\n`·1.00, `:`·0.00, `'`·0.00, ` with`·0.00, ` and`·0.00, ` of`·0.00, `,`·0.00, `.`·0.00, `&`·0.00, `-`·0.00 |
| A1 | 431 | -319.088 | `\n`·1.00, `:`·0.00, ` and`·0.00, `;`·0.00, ` in`·0.00, `,`·0.00, ` to`·0.00, ` with`·0.00, ` the`·0.00, ` of`·0.00 |

## Prompt (scientific): *"Photosynthesis converts carbon dioxide and"*

- Converged: 0/768
- $K^\ast = 2$   silhouette scores: K=2: +0.845, K=3: +0.634, K=4: +0.670, K=5: +0.681, K=6: +0.691, K=7: +0.688, K=8: +0.702, K=9: +0.634, K=10: +0.635, K=11: +0.509, K=12: +0.466

**Real next-token (tied LM head on $h_L$ of prompt):**  `\n`·0.38, ` and`·0.20, ` the`·0.01, ` I`·0.01, ` '`·0.01, `--`·0.00, ` that`·0.00, ` a`·0.00, `;`·0.00, ` their`·0.00

| Attractor | Size | $\langle V\rangle$ | Top-10 decoded tokens |
|---|---:|---:|---|
| A0 | 752 | -236.087 | `\n`·1.00, `:`·0.00, `;`·0.00, `,`·0.00, ` and`·0.00, ` to`·0.00, ` in`·0.00, `-`·0.00, ` the`·0.00, ` for`·0.00 |
| A1 | 16 | -915.713 | `\n`·1.00, `:`·0.00, `'`·0.00, ` with`·0.00, ` and`·0.00, ` of`·0.00, ` to`·0.00, `,`·0.00, `;`·0.00, ` for`·0.00 |

## Prompt (dialogue): *"She whispered: I love"*

- Converged: 0/768
- $K^\ast = 2$   silhouette scores: K=2: +0.804, K=3: +0.774, K=4: +0.750, K=5: +0.702, K=6: +0.715, K=7: +0.703, K=8: +0.684, K=9: +0.554, K=10: +0.564, K=11: +0.567, K=12: +0.563

**Real next-token (tied LM head on $h_L$ of prompt):**  `.`·0.12, `\n`·0.10, `,`·0.08, ` you`·0.04, ` him`·0.03, ` to`·0.03, `;`·0.03, ` me`·0.03, ` my`·0.03, ` it`·0.02

| Attractor | Size | $\langle V\rangle$ | Top-10 decoded tokens |
|---|---:|---:|---|
| A0 | 279 | -945.937 | `\n`·1.00, `:`·0.00, `'`·0.00, ` with`·0.00, ` and`·0.00, ` of`·0.00, `,`·0.00, ` to`·0.00, `;`·0.00, ` for`·0.00 |
| A1 | 489 | -282.527 | `\n`·1.00, `:`·0.00, `;`·0.00, ` to`·0.00, `,`·0.00, ` in`·0.00, ` and`·0.00, ` with`·0.00, ` for`·0.00, ` the`·0.00 |

## Prompt (code): *"def fibonacci(n): return 1 if n < 2 else"*

- Converged: 0/768
- $K^\ast = 2$   silhouette scores: K=2: +0.873, K=3: +0.694, K=4: +0.676, K=5: +0.665, K=6: +0.672, K=7: +0.498, K=8: +0.499, K=9: +0.504, K=10: +0.390, K=11: +0.392, K=12: +0.402

**Real next-token (tied LM head on $h_L$ of prompt):**  `\n`·0.74, `.`·0.03, `,`·0.03, ` I`·0.02, ` to`·0.01, ` you`·0.01, `?`·0.01, `;`·0.01, ` and`·0.01, ` the`·0.01

| Attractor | Size | $\langle V\rangle$ | Top-10 decoded tokens |
|---|---:|---:|---|
| A0 | 745 | -209.868 | `\n`·0.97, `:`·0.03, `-`·0.00, `;`·0.00, ` and`·0.00, ` the`·0.00, `,`·0.00, ` in`·0.00, ` to`·0.00, ` a`·0.00 |
| A1 | 23 | -881.834 | `\n`·1.00, `:`·0.00, ` with`·0.00, `'`·0.00, ` and`·0.00, ` of`·0.00, ` to`·0.00, `,`·0.00, `;`·0.00, ` for`·0.00 |
