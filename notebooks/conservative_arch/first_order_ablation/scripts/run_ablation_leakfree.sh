#!/usr/bin/env bash
# Leak-free retrain of the SPLM-1 vs SPLM-2 6-cell sweep.
#
# Identical to scripts/run_ablation.sh in every dimension EXCEPT:
#   - cfg.causal_force = True (the leak-fix; this is now the dataclass default
#     in SPLMSARFMassConfig, so no CLI flag is needed)
#   - output dirs are renamed to */leakfree/ to coexist with the original
#     buggy ckpts already on disk under results/{splm1, splm2_gamma0p30}/
#
# Two arms, three seeds each (six cells total):
#   arm A: SPLM-1   (first-order, no v-buffer, no gamma)        seeds {0, 1, 2}
#   arm B: SPLM em_ln gamma=0.30  (E5 winner, second-order)     seeds {0, 1, 2}
#
# Output:
#   first_order_ablation/results/splm1_leakfree/seed{0,1,2}/
#   first_order_ablation/results/splm2_gamma0p30_leakfree/seed{0,1,2}/
#
# Wall-clock: matched to the original sweep — ~28-34 min/cell on MPS, with one
# observed splm1/seed0 outlier of 57 min in the buggy run due to a host-sleep
# stall. Total ~3.5 h on MPS / 16-core CPU.
#
# Resilience: a failing cell does NOT abort the rest; a TRAINING_FAILED.txt
# marker is left for the diagnostic pass to skip cleanly.
#
# Decision rule for the resulting RESULTS_LEAKFREE.md:
#   - if Δ̄ = mean(SPLM-1) − mean(SPLM-2) ≥ +5 PPL with 3/3 seed sign
#     consistency: SPLM-2's published lead is qualitatively preserved under
#     the leak-fix; paper v3 §15 absolute numbers updated, conclusion stands.
#   - if Δ̄ < +5 PPL or sign-inverted: SPLM-2's published lead does NOT
#     survive a leak-free retrain; paper v3 §15 conclusion must be revised
#     (and the corresponding §17 contributions list amended).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABLATION_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONS_DIR="$(cd "${ABLATION_DIR}/.." && pwd)"
SARF_MASS_DIR="${CONS_DIR}/sarf_mass_variant"
LN_SWEEP_DIR="${CONS_DIR}/ln_damping_sweep"
LOGFREQ_PATH="${SARF_MASS_DIR}/results/logfreq_surprisal.npy"

if [[ ! -f "${LOGFREQ_PATH}" ]]; then
    echo "[SPLM-1 leakfree] missing logfreq surprisal table at ${LOGFREQ_PATH}" >&2
    exit 1
fi

SEEDS=(0 1 2)
mkdir -p "${ABLATION_DIR}/results/splm1_leakfree" \
         "${ABLATION_DIR}/results/splm2_gamma0p30_leakfree"
SWEEP_T0=$(date +%s)

echo "============================================================"
echo "[SPLM-1 leakfree] 6-cell retrain under cfg.causal_force=True"
echo "[SPLM-1 leakfree] start  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# ---- Arm A: SPLM-1 (first-order, leak-free) ----
for SEED in "${SEEDS[@]}"; do
    TAG="seed${SEED}"
    OUT_DIR="${ABLATION_DIR}/results/splm1_leakfree/${TAG}"
    mkdir -p "${OUT_DIR}"
    echo
    echo "------------------------------------------------------------"
    echo "[SPLM-1 leakfree] arm A (first-order)   seed=${SEED}   -> ${OUT_DIR}"
    echo "[SPLM-1 leakfree] start  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "------------------------------------------------------------"

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
        echo "[SPLM-1 leakfree] arm A seed=${SEED} OK in ${CELL_DT}s"
    fi
done

# ---- Arm B: SPLM em_ln, gamma=0.30 (matched second-order baseline, leak-free) ----
for SEED in "${SEEDS[@]}"; do
    TAG="seed${SEED}"
    OUT_DIR="${ABLATION_DIR}/results/splm2_gamma0p30_leakfree/${TAG}"
    mkdir -p "${OUT_DIR}"
    echo
    echo "------------------------------------------------------------"
    echo "[SPLM-1 leakfree] arm B (second-order γ=0.30)   seed=${SEED}   -> ${OUT_DIR}"
    echo "[SPLM-1 leakfree] start  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "------------------------------------------------------------"

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
        echo "[SPLM-1 leakfree] arm B seed=${SEED} OK in ${CELL_DT}s"
    fi
done

SWEEP_DT=$(( $(date +%s) - SWEEP_T0 ))
echo
echo "[SPLM-1 leakfree] sweep done in ${SWEEP_DT}s. Per-cell summary:"
for ARM in splm1_leakfree splm2_gamma0p30_leakfree; do
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
