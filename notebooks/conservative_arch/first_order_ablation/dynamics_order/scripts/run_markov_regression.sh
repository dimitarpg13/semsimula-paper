#!/usr/bin/env bash
# Run the (kernel ridge, p=50, k=1,2,3) primary Markov-order regression
# on each of the 6 SPLM checkpoints from the SPLM-1 ablation sweep.
# Phase 2 of E7. The regression module is unmodified
# `notebooks/dynamics_order_test/markov_order_regression.py`.
set -euo pipefail

cd "$(dirname "$0")/.."

QUADS_ROOT="results/quadruples"
OUT_ROOT="results/markov_primary"
N_JOBS="${N_JOBS:-4}"
N_BOOTSTRAP="${N_BOOTSTRAP:-10000}"
mkdir -p "$OUT_ROOT"

T0_ALL="$(date +%s)"
for CELL_DIR in "$QUADS_ROOT"/*; do
    CELL="$(basename "$CELL_DIR")"
    OUT="$OUT_ROOT/$CELL"
    LOG="$OUT/markov_stdout.log"
    if [[ -f "$OUT/primary_summary.json" ]]; then
        echo "[markov] $CELL: already done, skipping"
        continue
    fi
    mkdir -p "$OUT"
    echo "[markov] $CELL  n_jobs=$N_JOBS  n_bootstrap=$N_BOOTSTRAP"
    T0="$(date +%s)"
    python3 -u ../../../dynamics_order_test/markov_order_regression.py \
        --quads "$CELL_DIR/quadruples.npz" \
        --output_dir "$OUT" \
        --p 50 --k 1,2,3 --class kernel \
        --n_jobs "$N_JOBS" --n_bootstrap "$N_BOOTSTRAP" \
        > "$LOG" 2>&1
    DUR=$(( $(date +%s) - T0 ))
    SUMMARY="$OUT/primary_summary.json"
    DECISION=$(python3 -c "import json; d=json.load(open('$SUMMARY')); print(d.get('decision','?'),'rho_12=%.3f'%d['rho_12'],'rho_23=%.3f'%d['rho_23'])" 2>/dev/null || echo "?")
    echo "[markov] $CELL: done in ${DUR}s  decision=$DECISION"
done
echo "[markov] ALL DONE in $(( $(date +%s) - T0_ALL ))s"
