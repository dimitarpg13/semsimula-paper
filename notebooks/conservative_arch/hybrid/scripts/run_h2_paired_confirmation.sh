#!/usr/bin/env bash
# H2: S=3 paired confirmation for the two-stage hybrid (Variant A) at the
# k=4, m=4 split that won H1 with val PPL 133.01.
#
# Goal: provide the multi-seed Variant A baseline that the Helmholtz H2
# aggregator (`helmholtz/aggregate_h2.py`) needs in order to compute a
# true paired-t Q9d-vs-Variant-A on quality.  Without VA at >1 seed,
# the Q9d-vs-VA arm of the H2 verdict can only be a single-point
# anchor comparison (see docs/Helmholtz-HSPLM_Path_Forward_and_
# Experiments.md §4.5 / §8.3).
#
# Cells (2 NEW = 1 split x 2 new seeds; seed 0 is already on disk
# under hybrid/results/h1_sweep/k4_m4/seed0):
#
#   00  k=4  m=4   seed=1
#   01  k=4  m=4   seed=2
#
# Output layout:
#   hybrid/results/h2_paired_confirmation/k4_m4/seed{1,2}/
#
# Wall-clock estimate (per cell on Apple MPS):
#   ~32 min/cell at the matched_baseline budget (matches the H1 seed-0
#   cell at 32.2 min) -> ~64 min total for 2 cells (~1.1 h).
#
# Resilience: idempotent.  Skip cells whose summary.md already exists
# so an interrupted sweep can be resumed by re-launching.  Continue
# past per-cell failures and record TRAINING_FAILED.txt for triage.
#
# Optional env vars:
#   CELL_LIMIT=N   -- run at most N cells (0 = unlimited; default 0).
#   START_FROM=N   -- skip the first N cells (default 0).
#   FIXED_GAMMA=x  -- if set, use fixed gamma; default = freely learned
#                    (matches the H1 seed-0 protocol).
#   SEEDS="1 2"    -- whitespace-separated seed list (default "1 2").

set -uo pipefail

# Force unbuffered Python stdio so every train log line streams live to
# the tee'd log file (otherwise Python block-buffers ~8 KB).
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HYBRID_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONS_DIR="$(cd "${HYBRID_DIR}/.." && pwd)"
SARF_MASS_DIR="${CONS_DIR}/sarf_mass_variant"
LOGFREQ_PATH="${SARF_MASS_DIR}/results/logfreq_surprisal.npy"

if [[ ! -f "${LOGFREQ_PATH}" ]]; then
    echo "[va-h2] missing logfreq surprisal table at ${LOGFREQ_PATH}" >&2
    exit 1
fi

CELL_LIMIT="${CELL_LIMIT:-0}"
START_FROM="${START_FROM:-0}"
FIXED_GAMMA="${FIXED_GAMMA:-}"
SEEDS="${SEEDS:-1 2}"

# Hard-coded to the H1 winner: k=4, m=4 (val PPL 133.01 at seed 0).
K=4
M=4

SCHEDULES=()
SEED_LIST=()
for s in ${SEEDS}; do
    SCHEDULES+=("k${K}_m${M}")
    SEED_LIST+=("${s}")
done

OUT_BASE="${HYBRID_DIR}/results/h2_paired_confirmation"
mkdir -p "${OUT_BASE}"

n_cells="${#SCHEDULES[@]}"
echo "[va-h2] cells: ${n_cells} (k=${K}, m=${M} x ${SEEDS} seeds)"
if [[ -n "${FIXED_GAMMA}" ]]; then
    echo "[va-h2] gamma: fixed at ${FIXED_GAMMA}"
else
    echo "[va-h2] gamma: freely learned (default; matches H1 seed-0)"
fi
echo "[va-h2] output base: ${OUT_BASE}"
echo "[va-h2] CELL_LIMIT=${CELL_LIMIT} START_FROM=${START_FROM}"
echo

i=0
done_count=0
for (( idx=0; idx<n_cells; idx++ )); do
    SPLIT="${SCHEDULES[$idx]}"
    SEED="${SEED_LIST[$idx]}"

    if (( i < START_FROM )); then
        i=$((i + 1)); continue
    fi
    if (( CELL_LIMIT > 0 && done_count >= CELL_LIMIT )); then
        echo "[va-h2] CELL_LIMIT reached, stopping."
        break
    fi

    if [[ -n "${FIXED_GAMMA}" ]]; then
        FG_TAG="_g${FIXED_GAMMA//./p}"
        FG_FLAG="--fixed-gamma ${FIXED_GAMMA}"
    else
        FG_TAG=""
        FG_FLAG=""
    fi

    CELL_DIR="${OUT_BASE}/${SPLIT}${FG_TAG}/seed${SEED}"
    mkdir -p "${CELL_DIR}"
    SUMMARY_GLOB="${CELL_DIR}/hybrid_${SPLIT}${FG_TAG}_shakespeare_seed${SEED}_summary.md"

    if [[ -f "${SUMMARY_GLOB}" ]]; then
        echo "[va-h2] skip cell #$(printf '%02d' $i)  ${SPLIT}${FG_TAG} seed=${SEED}  "\
             "(summary exists at ${SUMMARY_GLOB})"
        i=$((i + 1)); continue
    fi

    echo
    echo "[va-h2] cell #$(printf '%02d' $i)  ${SPLIT}${FG_TAG}  "\
         "seed=${SEED}  ->  ${CELL_DIR}"
    echo "[va-h2] $(date -u +'%Y-%m-%dT%H:%M:%SZ') start"

    pushd "${HYBRID_DIR}" >/dev/null
    if python3 train_splm_hybrid.py \
        --mode shakespeare \
        --n-attn "${K}" --n-splm "${M}" \
        ${FG_FLAG} \
        --logfreq-path "${LOGFREQ_PATH}" \
        --seed "${SEED}" \
        2>&1 | tee "${CELL_DIR}/training.log"; then
        echo "[va-h2] cell #$(printf '%02d' $i) training succeeded."
    else
        echo "[va-h2] cell #$(printf '%02d' $i) training FAILED (rc=$?)" \
            >> "${CELL_DIR}/TRAINING_FAILED.txt"
        echo "[va-h2] continuing past failed cell."
        popd >/dev/null
        i=$((i + 1)); done_count=$((done_count + 1)); continue
    fi
    popd >/dev/null

    BASE_TAG="hybrid_${SPLIT}${FG_TAG}_shakespeare_seed${SEED}"
    for ext in training_log.jsonl ckpt_latest.pt loss_curve.png summary.md; do
        SRC="${HYBRID_DIR}/results/${BASE_TAG}_${ext}"
        if [[ -f "${SRC}" ]]; then
            mv "${SRC}" "${CELL_DIR}/"
        fi
    done

    echo "[va-h2] $(date -u +'%Y-%m-%dT%H:%M:%SZ') done."
    i=$((i + 1)); done_count=$((done_count + 1))
done

echo
echo "[va-h2] sweep complete.  cells run: ${done_count}"
echo "[va-h2] outputs under: ${OUT_BASE}"
echo "[va-h2] next: re-run helmholtz/aggregate_h2.py to refresh the"
echo "        H2_RESULTS verdict with the paired-t Q9d-vs-VA arm."
