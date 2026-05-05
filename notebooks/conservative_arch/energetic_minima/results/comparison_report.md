# Energetic-minima alternatives to free $V_\theta$: cross-variant comparison

All four variants share the SARF-faithful $\xi$ re-pooling, logfreq per-token mass, Tiny Shakespeare 4000 steps, $d=128$, $L=8$, $\Delta t=1$, seed 0.

| variant | val ppl | K* per prompt (n,m,s,d,c) | V range | mean content-basin fraction |
|---|---:|---|---|---:|
| baseline SARF+mass (logfreq) | — | 9,10,8,10,8 | [-1916.6, +1444.8] | 0.58 |
| (i) LayerNorm-after-step | 173.59 | 2,4,2,3,2 | [-114.8, -24.6] | 0.00 |
| (ii) scale-gauge (lambda_V0=1e-3) | 244.84 | 7,5,4,5,5 | [-1698.8, -311.3] | 0.52 |
| (iii) Gaussian-mixture head (K=64) | 542.65 | 2,2,2,2,2 | [+63.4, +64.1] | 0.00 |

Columns:
- **val ppl**: final validation perplexity from the last   training eval step.
- **K*** per prompt: silhouette-optimal K, K∈[2,10], of   K-means on damped-flow endpoints.  Order =   (narrative, mathematics, scientific, dialogue, code).
- **V range**: global [min, max] of $\langle V\rangle$   across all basins across all five prompts (summary of   how wide the learned energy surface is).
- **mean content-basin fraction**: over the five prompts,   average fraction of basins whose largest-probability   decoded token is not a punctuation symbol.
