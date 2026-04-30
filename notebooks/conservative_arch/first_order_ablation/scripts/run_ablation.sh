#!/usr/bin/env bash
# SPLM-1 ablation sweep: first-order vs. second-order at gamma*=0.30, multi-seed.
#
# Two arms, three seeds each (six cells total):
#   arm A: SPLM-1   (first-order, no v-buffer, no gamma)        seeds {0, 1, 2}
#   arm B: SPLM em_ln gamma=0.30   (E5 winner, second-order)    seeds {0, 1, 2}
#
# Arm B with seed=0 is already done (E5 ln_damping_sweep, val_ppl=87.06);
# we re-train it inside this sweep anyway so all cells share an identical
# protocol (same script invocation, same machine, same wall-clock window).
#
# Output:
#   first_order_ablation/results/<arm>/<tag>/
#     splm_first_order_shakespeare_<tag>_summary.md     (arm A)
#     splm_em_ln_shakespeare_<tag>_summary.md           (arm B)
#
# Wall-clock: ~30–60 min/cell × 6 cells ≈ 3–6 hours on MPS / 16-core CPU.
#
# Resilience: a failing cell does NOT abort the rest; a TRAINING_FAILED.txt
# marker is left for the diagnostic pass to skip cleanly.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABLATION_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONS_DIR="$(cd "${ABLATION_DIR}/.." && pwd)"
SARF_MASS_DIR="${CONS_DIR}/sarf_mass_variant"
LN_SWEEP_DIR="${CONS_DIR}/ln_damping_sweep"
LOGFREQ_PATH="${SARF_MASS_DIR}/results/logfreq_surprisal.npy"

if [[ ! -f "${LOGFREQ_PATH}" ]]; then
    echo "[SPLM-1] missing logfreq surprisal table at ${LOGFREQ_PATH}" >&2
    exit 1
fi

SEEDS=(0 1 2)
mkdir -p "${ABLATION_DIR}/results/splm1" "${ABLATION_DIR}/results/splm2_gamma0p30"
SWEEP_T0=$(date +%s)

# ---- Arm A: SPLM-1 (first-order) ----
for SEED in "${SEEDS[@]}"; do
    TAG="seed${SEED}"
    OUT_DIR="${ABLATION_DIR}/results/splm1/${TAG}"
    mkdir -p "${OUT_DIR}"
    echo
    echo "============================================================"
    echo "[SPLM-1] arm A (first-order)   seed=${SEED}   -> ${OUT_DIR}"
    echo "[SPLM-1] start  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    CELL_T0=$(date +%s)
    PYTHONUNBUFFERED=1 python3 -u "${ABLATION_DIR}/train_splm_first_order.py" \
        --mode shakespeare \
        --logfreq-path "${LOGFREQ_PATH}" \
        --seed "${SEED}" \
        --tag-suffix "${TAG}" \
        --results-dir "${OUT_DIR}" \
        2>&1 | tee "${OUT_DIR}/train_stdout.log"
    RC="${PIPESTATUS[0]}"
    CELL_DT=$(( $(date +%s) - CELL_T0 ))

    if [[ "${RC}" -ne 0 ]]; then
        echo "!! arm A seed=${SEED} FAILED with rc=${RC} after ${CELL_DT}s" \
            | tee "${OUT_DIR}/TRAINING_FAILED.txt"
    else
        echo "[SPLM-1] arm A seed=${SEED} OK in ${CELL_DT}s"
    fi
done

# ---- Arm B: SPLM em_ln, gamma=0.30 (matched second-order baseline) ----
for SEED in "${SEEDS[@]}"; do
    TAG="seed${SEED}"
    OUT_DIR="${ABLATION_DIR}/results/splm2_gamma0p30/${TAG}"
    mkdir -p "${OUT_DIR}"
    echo
    echo "============================================================"
    echo "[SPLM-1] arm B (second-order γ=0.30)   seed=${SEED}   -> ${OUT_DIR}"
    echo "[SPLM-1] start  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    CELL_T0=$(date +%s)
    PYTHONUNBUFFERED=1 python3 -u "${LN_SWEEP_DIR}/train_splm_em_ln.py" \
        --mode shakespeare \
        --logfreq-path "${LOGFREQ_PATH}" \
        --seed "${SEED}" \
        --fixed-gamma 0.30 \
        --tag-suffix "${TAG}" \
        --results-dir "${OUT_DIR}" \
        2>&1 | tee "${OUT_DIR}/train_stdout.log"
    RC="${PIPESTATUS[0]}"
    CELL_DT=$(( $(date +%s) - CELL_T0 ))

    if [[ "${RC}" -ne 0 ]]; then
        echo "!! arm B seed=${SEED} FAILED with rc=${RC} after ${CELL_DT}s" \
            | tee "${OUT_DIR}/TRAINING_FAILED.txt"
    else
        echo "[SPLM-1] arm B seed=${SEED} OK in ${CELL_DT}s"
    fi
done

SWEEP_DT=$(( $(date +%s) - SWEEP_T0 ))
echo
echo "[SPLM-1] sweep done in ${SWEEP_DT}s. Per-cell summary:"
for ARM in splm1 splm2_gamma0p30; do
    for SEED in "${SEEDS[@]}"; do
        TAG="seed${SEED}"
        DIR="${ABLATION_DIR}/results/${ARM}/${TAG}"
        if [[ -f "${DIR}/TRAINING_FAILED.txt" ]]; then
            echo "  - ${ARM}/${TAG}: FAILED"
        elif compgen -G "${DIR}/*_summary.md" > /dev/null; then
            echo "  - ${ARM}/${TAG}: OK"
        else
            echo "  - ${ARM}/${TAG}: MISSING"
        fi
    done
done
