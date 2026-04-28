#!/usr/bin/env bash
# One-shot reproduction of the first-order ODE rejection experiment.
# Pre-registered protocol: docs/first_order_ODE_rejection_pre-registered_protocol.md
#
# Total wall-clock on M-series Mac: ≈ 35–40 min.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "[run_all] phase 1: extraction"
python extract_lagged_quadruples.py --model gpt2                   --output_dir results/gpt2
python extract_lagged_quadruples.py --model EleutherAI/pythia-160m --output_dir results/pythia

echo "[run_all] phase 2: primary cell — kernel ridge, p=50"
python markov_order_regression.py \
    --quads results/gpt2/quadruples.npz   --output_dir results/gpt2   \
    --p 50 --k 1,2,3 --class kernel --n_bootstrap 10000
python markov_order_regression.py \
    --quads results/pythia/quadruples.npz --output_dir results/pythia \
    --p 50 --k 1,2,3 --class kernel --n_bootstrap 10000

echo "[run_all] phase 4: 24-cell robustness sweep"
python robustness_sweep.py \
    --gpt2_quads results/gpt2/quadruples.npz \
    --pythia_quads results/pythia/quadruples.npz \
    --output_path results/robustness_grid.csv

echo "[run_all] phase 5: figures"
python plots.py --residuals_npz results/gpt2/primary_residuals.npz \
    --out_dir results/gpt2/figures   --title_prefix "GPT-2 small" \
    --robustness_csv results/robustness_grid.csv
python plots.py --residuals_npz results/pythia/primary_residuals.npz \
    --out_dir results/pythia/figures --title_prefix "Pythia-160m" \
    --robustness_csv results/robustness_grid.csv

echo "[run_all] done. See results/RESULTS.md"
