# `damping_sweep/` — E4: how SPLM behaves as $\gamma$ is varied

> **Pre-registered protocol**: [`docs/E4_damping_sweep_pre-registered_protocol.md`](../../../docs/E4_damping_sweep_pre-registered_protocol.md)
> **Companion experiments this builds on**:
> - [`../sarf_mass_variant/`](../sarf_mass_variant/) — the trained-$\gamma$ best-PPL Euler baseline this sweep modifies one knob of.
> - [`../energy_drift/`](../energy_drift/) — the E3 energy-drift diagnostic, run per-cell as part of this sweep.
> - [`../../dynamics_order_test/`](../../dynamics_order_test/) — the Markov-order regression pipeline, reused per-cell as a positive control.

## Status

**Pre-registered, not yet executed.** All scripts in this directory
implement the protocol locked in §3 of `E4_damping_sweep_pre-registered_protocol.md`.
Running the sweep produces a six-cell × four-diagnostic grid that directly
addresses the protocol's H1–H4 hypotheses.

## What the experiment answers

The first-order ODE rejection test (`docs/first_order_ODE_rejection_pre-registered_protocol.md`)
established that natural transformer trajectories at one-token resolution
are observationally consistent with both a strict first-order ODE and an
overdamped second-order Lagrangian. This sweep asks the empirical
**inverse** question on SPLM:

1. At what $\gamma$ does SPLM's val PPL peak, and how steeply does it
   degrade off-axis?
2. Is the Markov-order regression test we built able to *reject* first-
   order on a sufficiently underdamped SPLM (positive control)?

If yes to (2), the test we ran on GPT-2 / Pythia is validated as having
real discriminative power; the C outcome on natural transformers is then
substantive evidence that those trajectories sit in the overdamped
regime, not a Type-II artefact of low test power.

## Six-cell grid (locked)

| tag | $\gamma$ | per-step damping at $\Delta t = 1$ | regime |
|---|---|---|---|
| `gamma0p00` | 0.00 | 0 % | undamped (ballistic floor) |
| `gamma0p10` | 0.10 | 9.5 % | very underdamped |
| `gamma0p30` | 0.30 | 25.9 % | mildly underdamped |
| `gamma0p85` | 0.85 | 57.2 % | natural operating point |
| `gamma2p00` | 2.00 | 86.5 % | strongly overdamped |
| `gamma5p00` | 5.00 | 99.3 % | quasi-quenched |

Everything else is held fixed at the
[`sarf_mass_variant`](../sarf_mass_variant/) `mass_mode = logfreq`
Tiny-Shakespeare baseline (seed = 0, 4000 steps, $L = 8$, $\Delta t = 1$,
$d = 128$, $v_{\text{hidden}} = 512$, $v_{\text{depth}} = 3$).

## How to reproduce

The sweep is one shell script plus three Python drivers.

```bash
cd notebooks/conservative_arch/damping_sweep

# 1. (one-shot) make sure the surprisal table is available; the
#    sarf_mass_variant produces it during its own training.
ls ../sarf_mass_variant/results/logfreq_surprisal.npy   # should exist

# 2. train all six cells (~4 hours total on MPS / 16-core CPU)
bash scripts/run_sweep.sh

# 3. extract energy states, last-layer hidden trajectories, and
#    Markov-order regression quadruples for every cell
python3 run_diagnostics.py

# 4. aggregate, plot, and emit results/RESULTS.md
python3 analyse_sweep.py
```

## Output layout (one tree per $\gamma$ cell)

```
results/<tag>/
├── splm_sarfmass_logfreq_shakespeare_<tag>_summary.md
├── splm_sarfmass_logfreq_shakespeare_<tag>_training_log.jsonl
├── splm_sarfmass_logfreq_shakespeare_<tag>_loss_curve.png
├── splm_sarfmass_logfreq_shakespeare_<tag>_ckpt_latest.pt
├── energy_states.npz                  # per-layer kinetic + potential
├── energy_drift_summary.json          # drift slope, oscillation bandwidth
├── trajectories.npz                   # last-layer hidden states, per (sentence, token)
├── acceleration_stats.json            # §14 a_par sign rate, |a_par|/|a_perp|, perm z
└── markov_order/
    ├── quadruples.npz
    ├── primary_summary.json           # mirror of dynamics_order_test/results/gpt2/
    ├── primary_residuals.npz
    └── decision_table.md
```

After `analyse_sweep.py` runs, the top-level `results/` adds:

```
results/
├── RESULTS.md                # human-readable headline + per-cell rows
├── sweep_grid.csv            # one row per cell, all numerical metrics
├── sweep_grid_summary.json   # decisions per cell + headline outcome
└── ../figures/
    ├── ppl_vs_gamma.png
    ├── drift_slope_vs_gamma.png
    ├── markov_rho12_vs_gamma.png
    ├── apar_negative_vs_gamma.png
    └── per_cell_loss_curves.png
```

## Files

- [`scripts/run_sweep.sh`](scripts/run_sweep.sh) — orchestrator that
  trains all six cells in series, each writing its own `<tag>/`
  sub-directory.
- [`run_diagnostics.py`](run_diagnostics.py) — for each cell: extract
  energy states (E3-style); run the energy-drift summary; extract last-
  layer per-token hidden states on the `dynamics_order_test/data/corpus.json`
  corpus; compute §14 acceleration statistics; run the Markov-order
  regression primary cell.
- [`analyse_sweep.py`](analyse_sweep.py) — load every per-cell output,
  build the sweep CSV/JSON/markdown, plot the four headline curves.
- [`extract_splm_quadruples.py`](extract_splm_quadruples.py) — adapter
  that mimics
  [`dynamics_order_test/extract_lagged_quadruples.py`](../../dynamics_order_test/extract_lagged_quadruples.py)
  but operates on a trained SPLM checkpoint (per-token last-layer hidden
  states), so the Markov-order regression can be run identically across
  cells.

## Cross-references

- E1 multi-seed harness: [`../multi_seed/`](../multi_seed/)
- E3 energy drift on the natural overdamped point ($\gamma\approx 0.85$):
  [`../energy_drift/results/E3_splm_em_ln_compare/`](../energy_drift/results/E3_splm_em_ln_compare/)
- The first-order ODE rejection test that motivated this sweep:
  [`../../dynamics_order_test/results/RESULTS.md`](../../dynamics_order_test/results/RESULTS.md)
- The synthesis under audit: [`docs/Evidence_for_second_order_ODE_governing_evolution.md`](../../../docs/Evidence_for_second_order_ODE_governing_evolution.md)
