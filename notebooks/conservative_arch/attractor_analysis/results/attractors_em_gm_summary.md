# Semantic-attractor extraction -- em_gm

- Variant: `sarf_mass_gm`
- Model config: `d=128, L=8, mass_mode=logfreq`
- Seeds: 96 Gaussian + 96 token-embedding + 96 perturbed real-$h$
- Descent: Adam lr=0.05, 2000 steps
- Convergence filter: $\|\nabla V\| < 0.05$
- Clustering: K-means, silhouette-sweep K $\in$ [2, 10]

## Prompt (narrative): *"The old king sat on the"*

- Converged: 10/288
- $K^\ast = 2$   silhouette scores: K=2: +0.824, K=3: +0.624, K=4: +0.129, K=5: +0.109, K=6: +0.063, K=7: +0.039, K=8: +0.034, K=9: +0.012

**Real next-token (tied LM head on $h_L$ of prompt):**  ` the`·0.02, `,`·0.02, ` a`·0.01, ` to`·0.01, `\n`·0.01, ` of`·0.01, ` my`·0.01, ` and`·0.01, ` be`·0.01, `'s`·0.01

| Attractor | Size | $\langle V\rangle$ | Top-10 decoded tokens |
|---|---:|---:|---|
| A0 | 7 | +63.651 | `\n`·0.06, `,`·0.05, ` the`·0.02, `.`·0.02, `:`·0.01, ` to`·0.01, ` I`·0.01, ` and`·0.01, ` a`·0.01, ` of`·0.01 |
| A1 | 3 | +63.652 | `\n`·0.12, `,`·0.04, `.`·0.02, `:`·0.02, ` the`·0.02, ` I`·0.01, `;`·0.01, ` to`·0.01, ` and`·0.01, ` a`·0.01 |

## Prompt (mathematics): *"The theorem states that for every"*

- Converged: 14/288
- $K^\ast = 2$   silhouette scores: K=2: +0.824, K=3: +0.514, K=4: +0.295, K=5: +0.227, K=6: +0.103, K=7: +0.079, K=8: +0.059, K=9: +0.053, K=10: +0.049

**Real next-token (tied LM head on $h_L$ of prompt):**  `,`·0.06, ` the`·0.03, `\n`·0.02, ` to`·0.02, ` of`·0.02, ` and`·0.02, ` a`·0.01, ` my`·0.01, ` I`·0.01, ` in`·0.01

| Attractor | Size | $\langle V\rangle$ | Top-10 decoded tokens |
|---|---:|---:|---|
| A0 | 13 | +63.589 | `,`·0.06, `\n`·0.03, ` the`·0.03, ` to`·0.02, ` of`·0.02, ` and`·0.02, `.`·0.01, ` a`·0.01, ` my`·0.01, ` I`·0.01 |
| A1 | 1 | +63.678 | `,`·0.24, `\n`·0.10, ` the`·0.04, ` to`·0.03, `.`·0.03, `:`·0.03, ` and`·0.03, ` of`·0.02, ` I`·0.02, `;`·0.02 |

## Prompt (scientific): *"Photosynthesis converts carbon dioxide and"*

- Converged: 29/288
- $K^\ast = 2$   silhouette scores: K=2: +0.837, K=3: +0.290, K=4: +0.205, K=5: +0.212, K=6: +0.200, K=7: +0.140, K=8: +0.111, K=9: +0.094, K=10: +0.098

**Real next-token (tied LM head on $h_L$ of prompt):**  `,`·0.16, ` to`·0.03, `\n`·0.03, ` and`·0.03, ` of`·0.03, `:`·0.02, `.`·0.02, `;`·0.02, ` the`·0.02, ` you`·0.02

| Attractor | Size | $\langle V\rangle$ | Top-10 decoded tokens |
|---|---:|---:|---|
| A0 | 27 | +64.084 | `,`·0.07, ` the`·0.03, ` to`·0.02, ` of`·0.02, `\n`·0.02, ` and`·0.02, ` a`·0.01, ` my`·0.01, ` you`·0.01, ` in`·0.01 |
| A1 | 2 | +64.133 | `,`·0.21, `\n`·0.06, ` the`·0.04, ` to`·0.03, ` and`·0.03, ` of`·0.03, `.`·0.02, `:`·0.02, `;`·0.02, ` you`·0.02 |

## Prompt (dialogue): *"She whispered: I love"*

- Converged: 14/288
- $K^\ast = 2$   silhouette scores: K=2: +0.913, K=3: +0.648, K=4: +0.492, K=5: +0.295, K=6: +0.259, K=7: +0.161, K=8: +0.090, K=9: +0.076, K=10: +0.079

**Real next-token (tied LM head on $h_L$ of prompt):**  `,`·0.07, `\n`·0.04, ` the`·0.02, ` to`·0.02, `.`·0.02, ` and`·0.01, `:`·0.01, ` of`·0.01, ` I`·0.01, ` my`·0.01

| Attractor | Size | $\langle V\rangle$ | Top-10 decoded tokens |
|---|---:|---:|---|
| A0 | 11 | +63.431 | `,`·0.06, `\n`·0.03, ` the`·0.03, ` to`·0.02, ` of`·0.02, ` and`·0.02, ` a`·0.01, ` my`·0.01, `.`·0.01, ` I`·0.01 |
| A1 | 3 | +63.530 | `,`·0.24, `\n`·0.11, ` the`·0.04, ` to`·0.03, `.`·0.03, `:`·0.03, ` and`·0.03, ` of`·0.02, ` I`·0.02, `;`·0.02 |

## Prompt (code): *"def fibonacci(n): return 1 if n < 2 else"*

- Converged: 15/288
- $K^\ast = 2$   silhouette scores: K=2: +0.207, K=3: +0.176, K=4: +0.132, K=5: +0.135, K=6: +0.127, K=7: +0.097, K=8: +0.074, K=9: +0.065, K=10: +0.056

**Real next-token (tied LM head on $h_L$ of prompt):**  `,`·0.12, `\n`·0.03, `:`·0.02, `.`·0.02, ` to`·0.02, ` and`·0.02, ` of`·0.02, `;`·0.02, ` the`·0.02, ` you`·0.02

| Attractor | Size | $\langle V\rangle$ | Top-10 decoded tokens |
|---|---:|---:|---|
| A0 | 9 | +63.784 | `,`·0.07, ` the`·0.03, ` to`·0.02, ` of`·0.02, ` and`·0.02, `\n`·0.02, ` a`·0.01, ` my`·0.01, ` you`·0.01, ` in`·0.01 |
| A1 | 6 | +63.784 | `,`·0.07, ` the`·0.03, ` to`·0.02, ` of`·0.02, ` and`·0.02, `\n`·0.02, ` a`·0.01, ` my`·0.01, ` in`·0.01, ` you`·0.01 |
