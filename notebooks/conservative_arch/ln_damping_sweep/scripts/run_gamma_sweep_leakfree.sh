#!/usr/bin/env bash
# E5 (γ-sweep) — leak-free 3-seed retrain over γ ∈ {0.00, 0.10, 0.85}.
#
# Companion to first_order_ablation/scripts/run_ablation_leakfree.sh, which
# already retrained 3 seeds at γ=0.30 under cfg.causal_force=True. This sweep
# fills in the curve at γ ∈ {0.00, 0.10, 0.85} with 3 seeds each, so we get
# a 4-point U-curve (γ ∈ {0.00, 0.10, 0.30, 0.85}) under leak-free training,
# each point with paired-t error bars.
#
# Why these γ: the original buggy E5 sweep had γ\* = 0.30 with neighbours
# {0.10, 0.85} on the descending and ascending limbs; γ ∈ {2.0, 5.0} were
# already clearly worse in the buggy regime and are unlikely to become
# competitive under leak-free, so we omit them to fit the time budget.
#
# Output: notebooks/conservative_arch/ln_damping_sweep/results/leakfree_3seed/
#           gamma{0p00,0p10,0p85}/seed{0,1,2}/
# Wall-clock: 9 cells × ~32 min ≈ ~4.8 h on MPS / 16-core CPU.
# Resilience: a failing cell does NOT abort the rest.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SARF_MASS_DIR="$(cd "${SWEEP_DIR}/../sarf_mass_variant" && pwd)"
LOGFREQ_PATH="${SARF_MASS_DIR}/results/logfreq_surprisal.npy"

if [[ ! -f "${LOGFREQ_PATH}" ]]; then
    echo "[E5-leakfree] missing logfreq surprisal table at ${LOGFREQ_PATH}" >&2
    exit 1
fi

GAMMAS=(0.00 0.10 0.85)
TAGS=(gamma0p00 gamma0p10 gamma0p85)
SEEDS=(0 1 2)

OUT_ROOT="${SWEEP_DIR}/results/leakfree_3seed"
mkdir -p "${OUT_ROOT}"

SWEEP_T0=$(date +%s)
echo "============================================================"
echo "[E5-leakfree] 3-seed γ-sweep under cfg.causal_force=True"
echo "[E5-leakfree] start  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

for i in "${!GAMMAS[@]}"; do
    GAMMA="${GAMMAS[$i]}"
    TAG="${TAGS[$i]}"
    for SEED in "${SEEDS[@]}"; do
        SEED_TAG="seed${SEED}"
        OUT_DIR="${OUT_ROOT}/${TAG}/${SEED_TAG}"
        mkdir -p "${OUT_DIR}"
        echo
        echo "------------------------------------------------------------"
        echo "[E5-leakfree] cell ${TAG}/${SEED_TAG}  fixed-gamma=${GAMMA}"
        echo "[E5-leakfree] start  $(date '+%Y-%m-%d %H:%M:%S')   -> ${OUT_DIR}"
        echo "------------------------------------------------------------"

        CELL_T0=$(date +%s)
        PYTHONUNBUFFERED=1 python3 -u "${SWEEP_DIR}/train_splm_em_ln.py" \
            --mode shakespeare \
            --logfreq-path "${LOGFREQ_PATH}" \
            --seed "${SEED}" \
            --fixed-gamma "${GAMMA}" \
            --tag-suffix "${TAG}_${SEED_TAG}" \
            --results-dir "${OUT_DIR}" \
            2>&1 | tee "${OUT_DIR}/train_stdout.log"
        RC="${PIPESTATUS[0]}"
        CELL_DT=$(( $(date +%s) - CELL_T0 ))

        if [[ "${RC}" -ne 0 ]]; then
            echo "!! cell ${TAG}/${SEED_TAG} FAILED with rc=${RC} after ${CELL_DT}s" \
                | tee "${OUT_DIR}/TRAINING_FAILED.txt"
        else
            echo "[E5-leakfree] cell ${TAG}/${SEED_TAG} OK in ${CELL_DT}s"
        fi
    done
done

SWEEP_DT=$(( $(date +%s) - SWEEP_T0 ))
echo
echo "[E5-leakfree] sweep done in ${SWEEP_DT}s. Per-cell summary:"
for TAG in "${TAGS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        DIR="${OUT_ROOT}/${TAG}/seed${SEED}"
        if [[ -f "${DIR}/TRAINING_FAILED.txt" ]]; then
            echo "  - ${TAG}/seed${SEED}: FAILED"
        elif compgen -G "${DIR}/*_summary.md" > /dev/null; then
            echo "  - ${TAG}/seed${SEED}: OK"
        else
            echo "  - ${TAG}/seed${SEED}: MISSING"
        fi
    done
done
