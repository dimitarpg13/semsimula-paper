# E4 overnight run — status

- Started:  2026-04-28 06:05:43
- Finished: 2026-04-28 10:16:03
- Wall clock: 15020s (4h10m)
- Phase 1 (train) rc=0  duration=13949s
- Phase 2 (diag)  rc=0  duration=1069s
- Phase 3 (plots) rc=0  duration=2s

## Per-cell status

| tag | trained | energy states | quadruples | markov decision | ppl |
|---|:-:|:-:|:-:|:-:|---:|
| `gamma0p00` | OK | OK | OK | **C** | 201.65 |
| `gamma0p10` | OK | OK | OK | **C** | 166.30 |
| `gamma0p30` | OK | OK | OK | **C** | 144.06 |
| `gamma0p85` | OK | OK | OK | **C** | 203.00 |
| `gamma2p00` | OK | OK | OK | **C** | 215.33 |
| `gamma5p00` | OK | OK | OK | **C** | 202.16 |

## Headline files

- `results/sweep_grid.csv` — one row per cell, all metrics
- `results/sweep_grid_summary.json` — JSON summary
- `results/RESULTS.md` — narrative report (read this first)
- `figures/ppl_vs_gamma.png`, `figures/drift_slope_vs_gamma.png`, `figures/markov_rho12_vs_gamma.png`, `figures/apar_negative_vs_gamma.png` — headline curves
- `results/overnight_run.log` — combined log (this run)
