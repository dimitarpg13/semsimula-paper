# Energetic-minima alternatives to free $V_\theta$: cross-variant comparison

All four variants share the SARF-faithful $\xi$ re-pooling, logfreq per-token mass, Tiny Shakespeare 4000 steps, $d=128$, $L=8$, $\Delta t=1$, seed 0.

| variant | val ppl | K* per prompt (n,m,s,d,c) | V range | mean content-basin fraction |
|---|---:|---|---|---:|
| baseline SARF+mass (logfreq) | 160.55 | 9,10,8,10,8 | [-1916.6, +1444.8] | 0.58 |
| (i) LayerNorm-after-step | 88.63 | 5,9,10,2,2 | [-84.2, -60.5] | 0.23 |
| (ii) scale-gauge (lambda_V0=1e-3) | 191.00 | 2,2,2,2,10 | [-2332.4, -186.0] | 0.12 |
| (iii) Gaussian-mixture head (K=64) | 677.67 | 2,2,2,2,2 | [+60.3, +60.3] | 0.00 |

Columns:
- **val ppl**: final validation perplexity from the last   training eval step.
- **K*** per prompt: silhouette-optimal K, K∈[2,10], of   K-means on damped-flow endpoints.  Order =   (narrative, mathematics, scientific, dialogue, code).
- **V range**: global [min, max] of $\langle V\rangle$   across all basins across all five prompts (summary of   how wide the learned energy surface is).
- **mean content-basin fraction**: over the five prompts,   average fraction of basins whose largest-probability   decoded token is not a punctuation symbol.
