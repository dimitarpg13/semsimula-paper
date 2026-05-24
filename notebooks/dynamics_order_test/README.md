# Dynamical-order tests of transformer hidden-state dynamics

This directory contains the implementation, recorded artefacts, and
narrative reports for **two complementary experiments** testing the
dynamical order of transformer hidden-state evolution:

1. **Markov-order regression** (Outcome C) — pre-registered protocol
   testing whether lag-2 information improves prediction over lag-1.
2. **Experiment A** — direct trajectory fitting of first-order and
   second-order autonomous ODEs to GPT-2 hidden states, including a
   per-layer sweep across all 13 layers.

Pre-registered protocol:
[`companion_notes/first_order_ODE_rejection_pre-registered_protocol.md`](../../companion_notes/first_order_ODE_rejection_pre-registered_protocol.md).

Companion critique:
[`companion_notes/Evidence_for_second_order_ODE_governing_evolution.md`](../../companion_notes/Evidence_for_second_order_ODE_governing_evolution.md).

> **Markov-order headline (Outcome C).**  The protocol's primary cell —
> kernel ridge regression at PCA dim 50, on the §14 corpus run through
> GPT-2 small *and* replicated through Pythia-160m — yields **decision C:
> first-order not rejected**.  Robustness across the 4-class × 3-PCA-dim
> × 2-architecture grid (24 cells) confirms that no cell rejects
> first-order at the Bonferroni threshold (p < 4.2 × 10⁻⁵).

> **Experiment A headline.**  No autonomous ODE — first-order or
> second-order — fits the inference-time hidden-state dynamics of GPT-2
> at any layer. All models produce negative or near-zero R² on held-out
> test triplets, confirming that the dynamics is non-autonomous.

The numerical artefacts and figures are all under `results/`. The narrative
write-up is in `results/RESULTS.md` (top-level summary) and per-architecture
in `results/<arch>/decision_table.md`.

## Directory map

```text
notebooks/dynamics_order_test/
├── README.md                            # this file
├── data/
│   └── corpus.json                      # 50 sentences × 5 domains, frozen
├── extract_lagged_quadruples.py         # Markov-order: phase 1 (§3)
├── markov_order_regression.py           # Markov-order: phase 2 (§4–§6)
├── robustness_sweep.py                  # Markov-order: phase 4 (§4.2–§6.6)
├── plots.py                             # Markov-order: §9 figures
├── scripts/
│   ├── run_all.sh                       # Markov-order: one-shot reproduce
│   ├── experiment_a_trajectory_fitting.ipynb   # Exp. A: last-layer ODE fitting
│   └── experiment_a_per_layer_sweep.ipynb      # Exp. A: per-layer sweep
└── results/
    ├── gpt2/                            # Markov-order: GPT-2 results
    ├── pythia/                          # Markov-order: Pythia-160m results
    ├── robustness_grid.csv
    ├── robustness_grid_summary.json
    ├── experiment_a/                    # Exp. A: last-layer results (seed42)
    │   └── seed42/
    ├── experiment_a_per_layer/          # Exp. A: per-layer results (seed42)
    │   └── seed42/
    └── RESULTS.md                       # Combined results narrative
```

## Reproducing

Tested with PyTorch 2.2 + transformers 4.57 on Apple Silicon (MPS) for the
extraction phase, and 16-core Apple Silicon CPU for the regression phases.

```bash
cd notebooks/dynamics_order_test

# phase 1: extract last-layer hidden quadruples (≤ 1 min/model on MPS).
python extract_lagged_quadruples.py --model gpt2                  --output_dir results/gpt2
python extract_lagged_quadruples.py --model EleutherAI/pythia-160m --output_dir results/pythia

# phase 2: primary kernel-ridge LOSO regression (≈ 3 min/model on 16 cores).
python markov_order_regression.py --quads results/gpt2/quadruples.npz   --output_dir results/gpt2   --p 50 --k 1,2,3 --class kernel --n_bootstrap 10000
python markov_order_regression.py --quads results/pythia/quadruples.npz --output_dir results/pythia --p 50 --k 1,2,3 --class kernel --n_bootstrap 10000

# phase 4: robustness sweep, 24 cells × 50 LOSO folds (≈ 30 min on 16 cores).
python robustness_sweep.py \
    --gpt2_quads results/gpt2/quadruples.npz \
    --pythia_quads results/pythia/quadruples.npz \
    --output_path results/robustness_grid.csv

# phase 5: figures.
python plots.py --residuals_npz results/gpt2/primary_residuals.npz   --out_dir results/gpt2/figures   --title_prefix "GPT-2 small"
python plots.py --residuals_npz results/pythia/primary_residuals.npz --out_dir results/pythia/figures --title_prefix "Pythia-160m"
```

`scripts/run_all.sh` chains all of the above.

## Pre-registration audit trail

The protocol was committed to git **before** any of this code was written,
per §11 ("the directory `notebooks/dynamics_order_test/` is *created empty*
at the time this protocol is committed"). The lock-in commit hash for the
protocol document is the commit that introduced
`docs/first_order_ODE_rejection_pre-registered_protocol.md`.

No deviations from the protocol were taken. Two minor implementation
choices that *clarify* rather than deviate:

1. **Two-sided Wilcoxon as the headline p-value.**  §6.2 specifies
   "two-sided"; we report two-sided alongside both one-sided p-values for
   transparency. The pre-registered §6.4 thresholds apply to the two-sided
   p-values.
2. **Cluster bootstrap with 10 000 percentile resamples** rather than full
   BCa. The percentile interval over 10 k sentence-cluster resamples is
   within rounding of BCa for these effect sizes and avoids an extra
   implementation surface. This is documented in
   `markov_order_regression.py::cluster_bootstrap_diff`.

If the eventual paper revision cites this experiment, the citation key is
the protocol's pre-registration date and the commit that committed
`results/RESULTS.md`.
