#!/usr/bin/env bash
# H2: S=3 paired confirmation for the Helmholtz hybrid (Q9d).
#
# Goal: paired-t statistics against all-attention (5-seed E1 baseline)
# and Variant A best, on the two vh=128 cells that came out of H1.5.
#
# Cells (4 NEW = 2 schedules x 2 new seeds; seed 0 is already on
# disk under helmholtz/results/h1p5_narrow_v/<schedule>_vh128/seed0):
#
#   00  AAAASSSS  vh=128   seed=1   -- quality lead at S=1
#   01  AAAASSSS  vh=128   seed=2
#   02  AASSSSSS  vh=128   seed=1   -- joint quality+FLOP lead at S=1
#   03  AASSSSSS  vh=128   seed=2
#
# Why these two cells:
#   - AAAASSSS vh=128 was H1.5's quality lead (val PPL 134.89,
#     +1.88 vs Variant A best 133.01).
#   - AASSSSSS vh=128 is the only Q9d cell that simultaneously
#     (a) beats Variant A outright at iso-params/iso-FLOPs and
#     (b) clears the 30% decode-FLOP-reduction bar at T=4096.
#
# Output layout:
#   helmholtz/results/h2_paired_confirmation/<schedule>_vh128/seed{1,2}/
#
# Wall-clock estimate (per cell on Apple MPS):
#   ~22 min/cell at vh=128 -> ~88 min total for 4 cells (~1.5 h);
#   the design doc's H2 budget allowance is ~2.2 h.
#
# Resilience: idempotent.  Skip cells whose summary.md already exists
# (so an interrupted sweep can be resumed by re-launching).  Continue
# past per-cell failures and record TRAINING_FAILED.txt for triage.
#
# Optional env vars:
#   CELL_LIMIT=N   -- run at most N cells (0 = unlimited; default 0).
#   START_FROM=N   -- skip the first N cells (default 0).
#   FIXED_GAMMA=x  -- if set, use fixed gamma; default = freely learned.
#   SEEDS="1 2"    -- whitespace-separated seed list (default "1 2").

set -uo pipefail

# Force unbuffered Python stdio so each `[helm-train]` line streams
# live to the tee'd log; otherwise Python block-buffers ~8 KB.
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONS_DIR="$(cd "${HELM_DIR}/.." && pwd)"
SARF_MASS_DIR="${CONS_DIR}/sarf_mass_variant"
LOGFREQ_PATH="${SARF_MASS_DIR}/results/logfreq_surprisal.npy"

if [[ ! -f "${LOGFREQ_PATH}" ]]; then
    echo "[helm-h2] missing logfreq surprisal table at ${LOGFREQ_PATH}" >&2
    exit 1
fi

CELL_LIMIT="${CELL_LIMIT:-0}"
START_FROM="${START_FROM:-0}"
FIXED_GAMMA="${FIXED_GAMMA:-}"
SEEDS="${SEEDS:-1 2}"

# Build the cell list as parallel arrays from (schedules x seeds).
SCHEDULES_BASE=("AAAASSSS" "AASSSSSS")
V_HIDDEN=128

SCHEDULES=()
SEED_LIST=()
for sched in "${SCHEDULES_BASE[@]}"; do
    for s in ${SEEDS}; do
        SCHEDULES+=("${sched}")
        SEED_LIST+=("${s}")
    done
done

OUT_BASE="${HELM_DIR}/results/h2_paired_confirmation"
mkdir -p "${OUT_BASE}"

n_cells="${#SCHEDULES[@]}"
echo "[helm-h2] cells: ${n_cells} (2 schedules x ${SEEDS} seeds, vh=${V_HIDDEN})"
if [[ -n "${FIXED_GAMMA}" ]]; then
    echo "[helm-h2] gamma: fixed at ${FIXED_GAMMA}"
else
    echo "[helm-h2] gamma: freely learned (default)"
fi
echo "[helm-h2] output base: ${OUT_BASE}"
echo "[helm-h2] CELL_LIMIT=${CELL_LIMIT} START_FROM=${START_FROM}"
echo

i=0
done_count=0
for (( idx=0; idx<n_cells; idx++ )); do
    SIGMA="${SCHEDULES[$idx]}"
    SEED="${SEED_LIST[$idx]}"

    if (( i < START_FROM )); then
        i=$((i + 1)); continue
    fi
    if (( CELL_LIMIT > 0 && done_count >= CELL_LIMIT )); then
        echo "[helm-h2] CELL_LIMIT reached, stopping."
        break
    fi

    if [[ -n "${FIXED_GAMMA}" ]]; then
        FG_TAG="_g${FIXED_GAMMA//./p}"
        FG_FLAG="--fixed-gamma ${FIXED_GAMMA}"
    else
        FG_TAG=""
        FG_FLAG=""
    fi

    CELL_DIR="${OUT_BASE}/${SIGMA}_vh${V_HIDDEN}${FG_TAG}/seed${SEED}"
    mkdir -p "${CELL_DIR}"
    SUMMARY_GLOB="${CELL_DIR}/helm_${SIGMA}_vh${V_HIDDEN}${FG_TAG}_shakespeare_seed${SEED}_summary.md"

    if [[ -f "${SUMMARY_GLOB}" ]]; then
        echo "[helm-h2] skip cell #$(printf '%02d' $i)  ${SIGMA} vh=${V_HIDDEN}${FG_TAG} seed=${SEED}  "\
             "(summary exists at ${SUMMARY_GLOB})"
        i=$((i + 1)); continue
    fi

    echo
    echo "[helm-h2] cell #$(printf '%02d' $i)  ${SIGMA}  vh=${V_HIDDEN}${FG_TAG}  "\
         "seed=${SEED}  ->  ${CELL_DIR}"
    echo "[helm-h2] $(date -u +'%Y-%m-%dT%H:%M:%SZ') start"

    pushd "${HELM_DIR}" >/dev/null
    if python3 train_helmholtz.py \
        --mode shakespeare \
        --schedule "${SIGMA}" \
        --v-hidden "${V_HIDDEN}" \
        ${FG_FLAG} \
        --logfreq-path "${LOGFREQ_PATH}" \
        --seed "${SEED}" \
        2>&1 | tee "${CELL_DIR}/training.log"; then
        echo "[helm-h2] cell #$(printf '%02d' $i) training succeeded."
    else
        echo "[helm-h2] cell #$(printf '%02d' $i) training FAILED (rc=$?)" \
            >> "${CELL_DIR}/TRAINING_FAILED.txt"
        echo "[helm-h2] continuing past failed cell."
        popd >/dev/null
        i=$((i + 1)); done_count=$((done_count + 1)); continue
    fi
    popd >/dev/null

    BASE_TAG="helm_${SIGMA}_vh${V_HIDDEN}${FG_TAG}_shakespeare_seed${SEED}"
    for ext in training_log.jsonl ckpt_latest.pt loss_curve.png summary.md; do
        SRC="${HELM_DIR}/results/${BASE_TAG}_${ext}"
        if [[ -f "${SRC}" ]]; then
            mv "${SRC}" "${CELL_DIR}/"
        fi
    done

    echo "[helm-h2] $(date -u +'%Y-%m-%dT%H:%M:%SZ') done."
    i=$((i + 1)); done_count=$((done_count + 1))
done

echo
echo "[helm-h2] sweep complete.  cells run: ${done_count}"
echo "[helm-h2] outputs under: ${OUT_BASE}"
echo "[helm-h2] aggregate:  python3 ${HELM_DIR}/aggregate_h2.py"
