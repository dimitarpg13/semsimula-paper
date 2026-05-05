# Semantic-attractor extraction -- em_ln

- Variant: `sarf_mass_ln`
- Model config: `d=128, L=8, mass_mode=logfreq`
- Seeds: 96 Gaussian + 96 token-embedding + 96 perturbed real-$h$
- Descent: Adam lr=0.05, 2000 steps
- Convergence filter: $\|\nabla V\| < 0.05$
- Clustering: K-means, silhouette-sweep K $\in$ [2, 10]

## Prompt (narrative): *"The old king sat on the"*

- Converged: 0/288
- $K^\ast = 2$   silhouette scores: K=2: +0.636, K=3: +0.605, K=4: +0.492, K=5: +0.510, K=6: +0.447, K=7: +0.400, K=8: +0.400, K=9: +0.401, K=10: +0.400

**Real next-token (tied LM head on $h_L$ of prompt):**  ` king`·0.02, ` world`·0.01, ` the`·0.01, ` crown`·0.01, `\n`·0.01, ` people`·0.01, ` earth`·0.01, ` sun`·0.01, ` queen`·0.01, ` rest`·0.01

| Attractor | Size | $\langle V\rangle$ | Top-10 decoded tokens |
|---|---:|---:|---|
| A0 | 101 | -89.556 | `\n`·1.00, `:`·0.00, `,`·0.00, ` of`·0.00, `;`·0.00, `'`·0.00, ` and`·0.00, `.`·0.00, ` to`·0.00, `?`·0.00 |
| A1 | 187 | -68.536 | `\n`·0.78, `,`·0.03, ` the`·0.02, ` in`·0.01, `:`·0.01, ` of`·0.01, ` to`·0.01, ` with`·0.01, ` and`·0.01, `.`·0.01 |

## Prompt (mathematics): *"The theorem states that for every"*

- Converged: 0/288
- $K^\ast = 4$   silhouette scores: K=2: +0.696, K=3: +0.713, K=4: +0.726, K=5: +0.531, K=6: +0.505, K=7: +0.389, K=8: +0.404, K=9: +0.413, K=10: +0.416

**Real next-token (tied LM head on $h_L$ of prompt):**  ` day`·0.03, ` thing`·0.03, ` one`·0.02, ` word`·0.01, ` man`·0.01, ` hour`·0.01, ` every`·0.01, ` time`·0.01, ` part`·0.00, ` way`·0.00

| Attractor | Size | $\langle V\rangle$ | Top-10 decoded tokens |
|---|---:|---:|---|
| A0 | 78 | -99.989 | `\n`·1.00, `:`·0.00, ` and`·0.00, ` of`·0.00, `'`·0.00, `;`·0.00, ` to`·0.00, `,`·0.00, ` with`·0.00, ` for`·0.00 |
| A1 | 194 | -82.412 | `\n`·0.55, `,`·0.05, ` in`·0.02, ` the`·0.02, ` to`·0.01, `.`·0.01, `:`·0.01, ` so`·0.01, ` of`·0.01, ` with`·0.01 |
| A2 | 8 | -114.764 | `\n`·1.00, `:`·0.00, `,`·0.00, `.`·0.00, `;`·0.00, ` and`·0.00, ` of`·0.00, ` to`·0.00, `'`·0.00, `?`·0.00 |
| A3 | 8 | -24.635 | `\n`·1.00, `:`·0.00, `,`·0.00, `.`·0.00, `;`·0.00, ` of`·0.00, ` and`·0.00, ` to`·0.00, `?`·0.00, `!`·0.00 |

## Prompt (scientific): *"Photosynthesis converts carbon dioxide and"*

- Converged: 0/288
- $K^\ast = 2$   silhouette scores: K=2: +0.669, K=3: +0.362, K=4: +0.387, K=5: +0.350, K=6: +0.398, K=7: +0.361, K=8: +0.363, K=9: +0.380, K=10: +0.377

**Real next-token (tied LM head on $h_L$ of prompt):**  `\n`·0.38, ` and`·0.20, ` the`·0.01, ` I`·0.01, ` '`·0.01, `--`·0.00, ` that`·0.00, ` a`·0.00, `;`·0.00, ` their`·0.00

| Attractor | Size | $\langle V\rangle$ | Top-10 decoded tokens |
|---|---:|---:|---|
| A0 | 7 | -87.997 | `\n`·1.00, `:`·0.00, `,`·0.00, ` and`·0.00, ` to`·0.00, `;`·0.00, ` of`·0.00, `'`·0.00, `.`·0.00, ` with`·0.00 |
| A1 | 281 | -81.055 | `\n`·0.40, `,`·0.02, `.`·0.01, ` to`·0.01, ` the`·0.01, ` be`·0.01, `:`·0.01, ` have`·0.01, `;`·0.01, ` in`·0.01 |

## Prompt (dialogue): *"She whispered: I love"*

- Converged: 0/288
- $K^\ast = 3$   silhouette scores: K=2: +0.661, K=3: +0.670, K=4: +0.506, K=5: +0.426, K=6: +0.395, K=7: +0.396, K=8: +0.361, K=9: +0.358, K=10: +0.366

**Real next-token (tied LM head on $h_L$ of prompt):**  `.`·0.12, `\n`·0.10, `,`·0.08, ` you`·0.04, ` him`·0.03, ` to`·0.03, `;`·0.03, ` me`·0.03, ` my`·0.03, ` it`·0.02

| Attractor | Size | $\langle V\rangle$ | Top-10 decoded tokens |
|---|---:|---:|---|
| A0 | 80 | -71.342 | `\n`·1.00, `:`·0.00, `'`·0.00, `;`·0.00, ` and`·0.00, ` to`·0.00, `,`·0.00, ` of`·0.00, `.`·0.00, `?`·0.00 |
| A1 | 200 | -76.573 | `\n`·0.09, `,`·0.08, ` to`·0.03, ` in`·0.03, ` so`·0.02, `.`·0.02, ` the`·0.02, ` a`·0.01, `:`·0.01, ` not`·0.01 |
| A2 | 8 | -114.185 | `\n`·1.00, `:`·0.00, `,`·0.00, `'`·0.00, `;`·0.00, `.`·0.00, ` to`·0.00, ` and`·0.00, ` of`·0.00, `?`·0.00 |

## Prompt (code): *"def fibonacci(n): return 1 if n < 2 else"*

- Converged: 0/288
- $K^\ast = 2$   silhouette scores: K=2: +0.680, K=3: +0.298, K=4: +0.340, K=5: +0.342, K=6: +0.350, K=7: +0.344, K=8: +0.324, K=9: +0.336, K=10: +0.392

**Real next-token (tied LM head on $h_L$ of prompt):**  `\n`·0.74, `.`·0.03, `,`·0.03, ` I`·0.02, ` to`·0.01, ` you`·0.01, `?`·0.01, `;`·0.01, ` and`·0.01, ` the`·0.01

| Attractor | Size | $\langle V\rangle$ | Top-10 decoded tokens |
|---|---:|---:|---|
| A0 | 281 | -78.544 | `\n`·0.09, `,`·0.02, ` have`·0.01, `'ll`·0.01, ` be`·0.01, ` will`·0.01, ` would`·0.01, `.`·0.01, ` to`·0.01, ` so`·0.01 |
| A1 | 7 | -85.989 | `\n`·1.00, `:`·0.00, `,`·0.00, ` to`·0.00, ` and`·0.00, `;`·0.00, `'`·0.00, ` of`·0.00, ` with`·0.00, `.`·0.00 |
