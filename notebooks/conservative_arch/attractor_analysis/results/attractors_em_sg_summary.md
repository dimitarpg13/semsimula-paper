# Semantic-attractor extraction -- em_sg

- Variant: `sarf_mass`
- Model config: `d=128, L=8, mass_mode=logfreq`
- Seeds: 96 Gaussian + 96 token-embedding + 96 perturbed real-$h$
- Descent: Adam lr=0.05, 2000 steps
- Convergence filter: $\|\nabla V\| < 0.05$
- Clustering: K-means, silhouette-sweep K $\in$ [2, 10]

## Prompt (narrative): *"The old king sat on the"*

- Converged: 0/288
- $K^\ast = 2$   silhouette scores: K=2: +0.446, K=3: +0.411, K=4: +0.407, K=5: +0.424, K=6: +0.420, K=7: +0.418, K=8: +0.419, K=9: +0.412, K=10: +0.417

**Real next-token (tied LM head on $h_L$ of prompt):**  `\n`·0.80, `:`·0.09, `;`·0.05, `,`·0.04, `.`·0.00, `?`·0.00, `--`·0.00, `!`·0.00, ` the`·0.00, ` good`·0.00

| Attractor | Size | $\langle V\rangle$ | Top-10 decoded tokens |
|---|---:|---:|---|
| A0 | 109 | -789.753 | `\n`·0.59, `-`·0.32, ` very`·0.02, ` un`·0.02, ` most`·0.01, ` VI`·0.01, ` man`·0.01, ` king`·0.00, ` d`·0.00, ` noble`·0.00 |
| A1 | 179 | -851.357 | `\n`·0.97, `:`·0.02, `,`·0.01, ` the`·0.00, ` to`·0.00, ` a`·0.00, ` I`·0.00, `;`·0.00, ` in`·0.00, ` with`·0.00 |

## Prompt (mathematics): *"The theorem states that for every"*

- Converged: 0/288
- $K^\ast = 2$   silhouette scores: K=2: +0.479, K=3: +0.425, K=4: +0.406, K=5: +0.433, K=6: +0.415, K=7: +0.405, K=8: +0.393, K=9: +0.414, K=10: +0.375

**Real next-token (tied LM head on $h_L$ of prompt):**  ` the`·0.09, ` his`·0.05, `\n`·0.05, ` my`·0.03, ` their`·0.03, ` thy`·0.02, `,`·0.02, ` our`·0.02, ` in`·0.02, ` a`·0.02

| Attractor | Size | $\langle V\rangle$ | Top-10 decoded tokens |
|---|---:|---:|---|
| A0 | 111 | -898.554 | `,`·0.77, `-`·0.14, ` will`·0.03, `\n`·0.03, ` VI`·0.01, ` shall`·0.00, `'s`·0.00, ` is`·0.00, ` the`·0.00, ` cannot`·0.00 |
| A1 | 177 | -867.300 | `\n`·0.92, `:`·0.05, `,`·0.02, ` to`·0.00, ` I`·0.00, ` with`·0.00, ` the`·0.00, `;`·0.00, ` be`·0.00, ` in`·0.00 |

## Prompt (scientific): *"Photosynthesis converts carbon dioxide and"*

- Converged: 0/288
- $K^\ast = 2$   silhouette scores: K=2: +0.390, K=3: +0.339, K=4: +0.291, K=5: +0.301, K=6: +0.333, K=7: +0.348, K=8: +0.343, K=9: +0.376, K=10: +0.363

**Real next-token (tied LM head on $h_L$ of prompt):**  `:`·0.60, `,`·0.17, `\n`·0.05, `.`·0.04, `;`·0.03, `'d`·0.02, `-`·0.01, `est`·0.01, `!`·0.01, `?`·0.00

| Attractor | Size | $\langle V\rangle$ | Top-10 decoded tokens |
|---|---:|---:|---|
| A0 | 55 | -1481.982 | `\n`·1.00, `:`·0.00, `HAM`·0.00, ` I`·0.00, `ING`·0.00, ` thou`·0.00, `N`·0.00, `BR`·0.00, `EL`·0.00, ` II`·0.00 |
| A1 | 233 | -498.172 | `\n`·0.96, `:`·0.02, `,`·0.01, ` the`·0.00, `-`·0.00, ` with`·0.00, ` in`·0.00, ` his`·0.00, `;`·0.00, ` their`·0.00 |

## Prompt (dialogue): *"She whispered: I love"*

- Converged: 0/288
- $K^\ast = 2$   silhouette scores: K=2: +0.482, K=3: +0.442, K=4: +0.449, K=5: +0.462, K=6: +0.445, K=7: +0.456, K=8: +0.460, K=9: +0.449, K=10: +0.447

**Real next-token (tied LM head on $h_L$ of prompt):**  `,`·0.32, ` to`·0.05, `:`·0.05, `\n`·0.03, ` I`·0.03, `;`·0.03, ` of`·0.02, ` and`·0.01, `'s`·0.01, `.`·0.01

| Attractor | Size | $\langle V\rangle$ | Top-10 decoded tokens |
|---|---:|---:|---|
| A0 | 101 | -2332.447 | `N`·0.57, `TER`·0.14, `ENCE`·0.07, `INA`·0.06, `ROM`·0.02, `HAM`·0.02, `BR`·0.02, `D`·0.01, `MEN`·0.01, `AN`·0.01 |
| A1 | 187 | -1171.593 | `:`·0.91, `\n`·0.05, ` I`·0.01, ` thou`·0.01, ` we`·0.00, `N`·0.00, ` he`·0.00, `st`·0.00, `EL`·0.00, ` must`·0.00 |

## Prompt (code): *"def fibonacci(n): return 1 if n < 2 else"*

- Converged: 0/288
- $K^\ast = 10$   silhouette scores: K=2: +0.371, K=3: +0.314, K=4: +0.293, K=5: +0.319, K=6: +0.337, K=7: +0.354, K=8: +0.360, K=9: +0.371, K=10: +0.377

**Real next-token (tied LM head on $h_L$ of prompt):**  `:`·0.52, `,`·0.29, `;`·0.06, `\n`·0.06, `.`·0.04, `!`·0.01, `?`·0.01, ` to`·0.00, ` the`·0.00, `'s`·0.00

| Attractor | Size | $\langle V\rangle$ | Top-10 decoded tokens |
|---|---:|---:|---|
| A0 | 78 | -186.038 | `,`·0.54, `:`·0.36, `;`·0.03, `\n`·0.03, `!`·0.01, `.`·0.01, ` and`·0.00, ` in`·0.00, `?`·0.00, `'s`·0.00 |
| A1 | 24 | -1769.219 | `\n`·1.00, `HAM`·0.00, `ING`·0.00, `:`·0.00, `BR`·0.00, `EL`·0.00, ` I`·0.00, ` thou`·0.00, `N`·0.00, `YC`·0.00 |
| A2 | 17 | -969.687 | `\n`·1.00, ` the`·0.00, ` a`·0.00, ` I`·0.00, ` to`·0.00, ` his`·0.00, ` with`·0.00, `:`·0.00, `,`·0.00, ` thy`·0.00 |
| A3 | 13 | -549.405 | `-`·0.19, `ath`·0.17, ` b`·0.15, ` pl`·0.07, `EN`·0.04, ` st`·0.03, `ark`·0.03, `ill`·0.03, `is`·0.03, ` M`·0.02 |
| A4 | 48 | -368.209 | `\n`·0.43, `:`·0.23, `-`·0.16, `'d`·0.03, `ow`·0.01, `b`·0.01, `ing`·0.01, `ies`·0.01, `ish`·0.01, `,`·0.01 |
| A5 | 5 | -1442.871 | `\n`·1.00, ` the`·0.00, ` to`·0.00, ` your`·0.00, ` my`·0.00, ` a`·0.00, ` his`·0.00, ` thy`·0.00, `.`·0.00, ` our`·0.00 |
| A6 | 34 | -638.468 | `\n`·0.98, `,`·0.01, `:`·0.01, ` the`·0.00, ` to`·0.00, `.`·0.00, ` a`·0.00, ` thou`·0.00, ` I`·0.00, `;`·0.00 |
| A7 | 24 | -1112.170 | `\n`·1.00, `:`·0.00, ` I`·0.00, ` thou`·0.00, ` we`·0.00, ` he`·0.00, `BR`·0.00, `N`·0.00, ` they`·0.00, ` to`·0.00 |
| A8 | 3 | -1833.758 | ` III`·0.50, `US`·0.26, `AN`·0.08, `IO`·0.08, `OL`·0.04, ` II`·0.02, `ARD`·0.01, `CA`·0.00, `YC`·0.00, `ETH`·0.00 |
| A9 | 42 | -612.514 | `\n`·1.00, `:`·0.00, `,`·0.00, ` the`·0.00, ` in`·0.00, ` with`·0.00, ` was`·0.00, ` make`·0.00, ` be`·0.00, ` his`·0.00 |
