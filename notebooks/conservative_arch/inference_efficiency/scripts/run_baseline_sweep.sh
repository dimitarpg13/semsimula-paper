#!/usr/bin/env bash
# 3-seed retrain of the matched-parameter GPT-2-style baseline
# (E8 Phase 1 of the inference-efficiency benchmark). Mirrors the
# SPLM-1 ablation arm B compute budget so quality and inference cost
# are measured at apples-to-apples training conditions.
set -euo pipefail

cd "$(dirname "$0")/.."

ROOT="$(pwd)/results/matched_attn"
mkdir -p "$ROOT"

START_ALL="$(date +%s)"
for SEED in 0 1 2; do
    OUT="$ROOT/seed${SEED}"
    mkdir -p "$OUT"
    LOG="$OUT/train_stdout.log"
    if [[ -f "$OUT/matched_baseline_shakespeare_seed${SEED}_ckpt_latest.pt" ]]; then
        echo "[sweep] seed=${SEED} already complete, skipping"
        continue
    fi

    echo "[sweep] seed=${SEED}  out=${OUT}  log=${LOG}"
    START="$(date +%s)"
    python3 train_matched_baseline.py \
        --mode shakespeare \
        --seed "${SEED}" \
        --tag-suffix "seed${SEED}" \
        --results-dir "${OUT}" \
        2>&1 | tee "${LOG}"
    DUR=$(( $(date +%s) - START ))
    echo "[sweep] seed=${SEED}  done in ${DUR}s"
done

DUR_ALL=$(( $(date +%s) - START_ALL ))
echo "[sweep] ALL DONE in ${DUR_ALL}s"
