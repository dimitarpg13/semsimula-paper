# `ln_damping_sweep/` — E5: LN-after-step damping sweep

> **Companion experiment**: E4 plain-Euler sweep at
> [`../damping_sweep/`](../damping_sweep/) — same six-cell gamma grid,
> same dataset and hyperparameters, **only** the model changes.

## What this experiment answers

E4 established that for the plain Euler SPLM (`splm_sarfmass_logfreq`):

- Optimal gamma is ~0.30 (PPL 144 vs. 203 at the freely-trained gamma=0.85).
- The Markov-order regression returns Decision beta at every cell, including gamma=0.
- Energy drift is near-zero across the well-damped cells.

E5 asks the same questions for the **LayerNorm-after-step** variant (`splm_em_ln`,
val ppl ~88 at its freely-trained gamma), which is the best-performing SPLM variant
identified in the E3 comparison:

1. Does the LN model also have a well-defined optimal gamma, or does LayerNorm
   stabilise training so effectively that the PPL landscape is flat across the grid?
2. Does the optimal gamma shift relative to the plain-Euler optimum (gamma*=0.30)?
   LayerNorm contracts h after every step, effectively adding a geometric-mean
   damping on top of the explicit gamma term — so the LN model may prefer a lower
   gamma* or a flatter optimum.
3. Is the Markov-order Decision beta still universal across all six cells, or does
   the LN's stronger dissipation create any underdamped cell that returns Decision alpha?
4. How do the energy-drift and bandwidth diagnostics compare with E4?

## Six-cell grid (identical to E4)

| tag | gamma | per-step damping at dt=1 | regime |
|---|---|---|---|
| `gamma0p00` | 0.00 | 0 % | undamped (ballistic floor) |
| `gamma0p10` | 0.10 | 9.5 % | very underdamped |
| `gamma0p30` | 0.30 | 25.9 % | E4 optimum — does LN shift this? |
| `gamma0p85` | 0.85 | 57.2 % | natural (freely-trained) operating point |
| `gamma2p00` | 2.00 | 86.5 % | strongly overdamped |
| `gamma5p00` | 5.00 | 99.3 % | quasi-quenched |

Everything else is held fixed at the em_ln logfreq-mass Tiny-Shakespeare
baseline (seed=0, 4000 steps, L=8, d=128, v_hidden=512, v_depth=3).

## How to reproduce

```bash
cd notebooks/conservative_arch/ln_damping_sweep

# 1. verify logfreq table exists (produced by sarf_mass_variant)
ls ../sarf_mass_variant/results/logfreq_surprisal.npy

# 2. train all six cells (~5 hours on MPS / 16-core CPU)
bash scripts/run_sweep.sh

# 3. extract energy states, trajectories, and Markov-order quadruples
python3 run_diagnostics.py

# 4. aggregate, plot, write results/RESULTS.md
python3 analyse_sweep.py
```

## Output layout (one tree per gamma cell)

```
results/<tag>/
├── splm_em_ln_shakespeare_<tag>_summary.md
├── splm_em_ln_shakespeare_<tag>_training_log.jsonl
├── splm_em_ln_shakespeare_<tag>_loss_curve.png
├── splm_em_ln_shakespeare_<tag>_ckpt_latest.pt
├── energy_states.npz
├── energy_drift_summary.json
├── trajectories.npz
├── acceleration_stats.json
└── markov_order/
    ├── quadruples.npz
    ├── primary_summary.json
    ├── primary_residuals.npz
    └── extraction_summary.json
```

After `analyse_sweep.py`:

```
results/
├── RESULTS.md
├── sweep_grid.csv
└── sweep_grid_summary.json
figures/
├── ppl_vs_gamma.png
├── drift_slope_vs_gamma.png
├── markov_rho12_vs_gamma.png
└── apar_negative_vs_gamma.png
```

## Files

- [`train_splm_em_ln.py`](train_splm_em_ln.py) — trainer for `ScalarPotentialLMSARFMassLN`
  with `--fixed-gamma` support; mirrors `sarf_mass_variant/train_splm_sarf_mass.py`.
- [`extract_ln_quadruples.py`](extract_ln_quadruples.py) — loads an em_ln checkpoint
  and extracts last-layer hidden-state quadruples + trajectories in the same format
  as `dynamics_order_test/extract_lagged_quadruples.py`.
- [`run_diagnostics.py`](run_diagnostics.py) — per-cell orchestrator: energy
  extraction (`--variant em_ln`), quadruple extraction, Markov-order regression,
  acceleration statistics.
- [`analyse_sweep.py`](analyse_sweep.py) — reads all cell outputs, produces CSV,
  JSON summary, four headline figures, and `RESULTS.md`.
- [`scripts/run_sweep.sh`](scripts/run_sweep.sh) — trains all six cells in series.

## Cross-references

- E4 plain-Euler sweep (same grid): [`../damping_sweep/`](../damping_sweep/)
- E3 energy-drift comparison (LN vs. plain Euler): [`../energy_drift/results/E3_splm_em_ln_compare/`](../energy_drift/results/E3_splm_em_ln_compare/)
- E1 multi-seed validation (em_ln freely-trained): [`../multi_seed/`](../multi_seed/)
- Markov-order regression machinery: [`../../dynamics_order_test/`](../../dynamics_order_test/)
- Overdamped synthesis document: [`companion_notes/Evidence_for_second_order_ODE_governing_evolution.md`](../../../companion_notes/Evidence_for_second_order_ODE_governing_evolution.md)
