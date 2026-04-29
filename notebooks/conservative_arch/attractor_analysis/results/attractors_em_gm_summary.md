# Semantic-attractor extraction -- em_gm

- Variant: `sarf_mass_gm`
- Model config: `d=128, L=8, mass_mode=logfreq`
- Seeds: 96 Gaussian + 96 token-embedding + 96 perturbed real-$h$
- Descent: Adam lr=0.05, 2000 steps
- Convergence filter: $\lVert\nabla V\rVert < 0.05$
- Clustering: K-means, silhouette-sweep K $\in$ [2, 10]

## Prompt (narrative): *"The old king sat on the"*

- Converged: 197/288
- $K^\ast = 2$   silhouette scores: K=2: +0.442, K=3: +0.187, K=4: +0.185, K=5: +0.102, K=6: +0.108, K=7: +0.098, K=8: +0.089, K=9: +0.069, K=10: +0.057

**Real next-token (tied LM head on $h_L$ of prompt):**  `\n`·0.12, `,`·0.06, `:`·0.03, `.`·0.02, ` the`·0.02, ` to`·0.01, `;`·0.01, ` and`·0.01, ` I`·0.01, ` of`·0.01

| Attractor | Size | $\langle V\rangle$ | Top-10 decoded tokens |
|---|---:|---:|---|
| A0 | 180 | +60.301 | `\n`·0.09, `,`·0.05, `:`·0.03, `.`·0.02, ` the`·0.01, ` to`·0.01, `;`·0.01, ` and`·0.01, ` I`·0.01, ` of`·0.01 |
| A1 | 17 | +60.306 | `\n`·0.06, `,`·0.04, `:`·0.02, `.`·0.02, ` the`·0.01, ` to`·0.01, `;`·0.01, ` and`·0.01, ` I`·0.01, ` of`·0.01 |

## Prompt (mathematics): *"The theorem states that for every"*

- Converged: 197/288
- $K^\ast = 2$   silhouette scores: K=2: +0.442, K=3: +0.187, K=4: +0.186, K=5: +0.102, K=6: +0.108, K=7: +0.098, K=8: +0.089, K=9: +0.069, K=10: +0.057

**Real next-token (tied LM head on $h_L$ of prompt):**  `\n`·0.09, `,`·0.05, `:`·0.03, `.`·0.02, ` the`·0.01, ` to`·0.01, `;`·0.01, ` and`·0.01, ` I`·0.01, ` of`·0.01

| Attractor | Size | $\langle V\rangle$ | Top-10 decoded tokens |
|---|---:|---:|---|
| A0 | 180 | +60.289 | `\n`·0.09, `,`·0.05, `:`·0.03, `.`·0.02, ` the`·0.01, ` to`·0.01, `;`·0.01, ` and`·0.01, ` I`·0.01, ` of`·0.01 |
| A1 | 17 | +60.294 | `\n`·0.06, `,`·0.04, `:`·0.02, `.`·0.02, ` the`·0.01, ` to`·0.01, `;`·0.01, ` and`·0.01, ` I`·0.01, ` of`·0.01 |

## Prompt (scientific): *"Photosynthesis converts carbon dioxide and"*

- Converged: 197/288
- $K^\ast = 2$   silhouette scores: K=2: +0.442, K=3: +0.187, K=4: +0.186, K=5: +0.102, K=6: +0.108, K=7: +0.098, K=8: +0.089, K=9: +0.069, K=10: +0.057

**Real next-token (tied LM head on $h_L$ of prompt):**  `\n`·0.06, `,`·0.03, `:`·0.02, `.`·0.02, ` the`·0.01, ` to`·0.01, `;`·0.01, ` and`·0.01, ` I`·0.01, ` of`·0.01

| Attractor | Size | $\langle V\rangle$ | Top-10 decoded tokens |
|---|---:|---:|---|
| A0 | 180 | +60.281 | `\n`·0.09, `,`·0.05, `:`·0.03, `.`·0.02, ` the`·0.01, ` to`·0.01, `;`·0.01, ` and`·0.01, ` I`·0.01, ` of`·0.01 |
| A1 | 17 | +60.286 | `\n`·0.06, `,`·0.04, `:`·0.02, `.`·0.02, ` the`·0.01, ` to`·0.01, `;`·0.01, ` and`·0.01, ` I`·0.01, ` of`·0.01 |

## Prompt (dialogue): *"She whispered: I love"*

- Converged: 196/288
- $K^\ast = 2$   silhouette scores: K=2: +0.427, K=3: +0.188, K=4: +0.184, K=5: +0.160, K=6: +0.082, K=7: +0.076, K=8: +0.098, K=9: +0.098, K=10: +0.084

**Real next-token (tied LM head on $h_L$ of prompt):**  `\n`·0.13, `,`·0.06, `:`·0.03, `.`·0.03, ` the`·0.02, ` to`·0.01, `;`·0.01, ` and`·0.01, ` I`·0.01, ` of`·0.01

| Attractor | Size | $\langle V\rangle$ | Top-10 decoded tokens |
|---|---:|---:|---|
| A0 | 180 | +60.326 | `\n`·0.09, `,`·0.05, `:`·0.03, `.`·0.02, ` the`·0.01, ` to`·0.01, `;`·0.01, ` and`·0.01, ` I`·0.01, ` of`·0.01 |
| A1 | 16 | +60.330 | `\n`·0.07, `,`·0.04, `:`·0.02, `.`·0.02, ` the`·0.01, ` to`·0.01, `;`·0.01, ` and`·0.01, ` I`·0.01, ` of`·0.01 |

## Prompt (code): *"def fibonacci(n): return 1 if n < 2 else"*

- Converged: 196/288
- $K^\ast = 2$   silhouette scores: K=2: +0.427, K=3: +0.188, K=4: +0.184, K=5: +0.160, K=6: +0.082, K=7: +0.076, K=8: +0.098, K=9: +0.098, K=10: +0.084

**Real next-token (tied LM head on $h_L$ of prompt):**  `\n`·0.09, `,`·0.05, `:`·0.03, `.`·0.02, ` the`·0.01, ` to`·0.01, `;`·0.01, ` and`·0.01, ` I`·0.01, ` of`·0.01

| Attractor | Size | $\langle V\rangle$ | Top-10 decoded tokens |
|---|---:|---:|---|
| A0 | 180 | +60.324 | `\n`·0.09, `,`·0.05, `:`·0.03, `.`·0.02, ` the`·0.01, ` to`·0.01, `;`·0.01, ` and`·0.01, ` I`·0.01, ` of`·0.01 |
| A1 | 16 | +60.329 | `\n`·0.07, `,`·0.04, `:`·0.02, `.`·0.02, ` the`·0.01, ` to`·0.01, `;`·0.01, ` and`·0.01, ` I`·0.01, ` of`·0.01 |
