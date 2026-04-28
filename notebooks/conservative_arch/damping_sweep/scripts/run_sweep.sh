#!/usr/bin/env bash
# E4 damping sweep — train six fixed-gamma SPLM cells.
#
# Per the locked protocol (docs/E4_damping_sweep_pre-registered_protocol.md
# §3), every cell uses identical architecture, optimiser, seed, data, and
# step count. The only knob that varies is --fixed-gamma.
#
# Output: notebooks/conservative_arch/damping_sweep/results/<tag>/
# Wall-clock: ~40 min/cell × 6 cells ≈ 4 hours on MPS / 16-core CPU.
#
# Resilience: a single failing cell does NOT abort the rest of the sweep;
# its stdout log + a marker file in its directory are left for the
# diagnostic / analysis pass to skip cleanly.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAINER_DIR="$(cd "${SWEEP_DIR}/../sarf_mass_variant" && pwd)"
LOGFREQ_PATH="${TRAINER_DIR}/results/logfreq_surprisal.npy"

if [[ ! -f "${LOGFREQ_PATH}" ]]; then
    echo "[E4] missing logfreq surprisal table at ${LOGFREQ_PATH}" >&2
    echo "[E4] run sarf_mass_variant/compute_unigram_frequencies.py first." >&2
    exit 1
fi

GAMMAS=(0.00 0.10 0.30 0.85 2.00 5.00)
TAGS=(gamma0p00 gamma0p10 gamma0p30 gamma0p85 gamma2p00 gamma5p00)

mkdir -p "${SWEEP_DIR}/results"
SWEEP_T0=$(date +%s)

for i in "${!GAMMAS[@]}"; do
    GAMMA="${GAMMAS[$i]}"
    TAG="${TAGS[$i]}"
    OUT_DIR="${SWEEP_DIR}/results/${TAG}"
    mkdir -p "${OUT_DIR}"
    echo
    echo "============================================================"
    echo "[E4] cell ${TAG}: fixed-gamma=${GAMMA}  -> ${OUT_DIR}"
    echo "[E4] start   $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    CELL_T0=$(date +%s)
    PYTHONUNBUFFERED=1 python3 -u "${TRAINER_DIR}/train_splm_sarf_mass.py" \
        --mode shakespeare \
        --mass-mode logfreq \
        --logfreq-path "${LOGFREQ_PATH}" \
        --seed 0 \
        --fixed-gamma "${GAMMA}" \
        --tag-suffix "${TAG}" \
        --results-dir "${OUT_DIR}" \
        2>&1 | tee "${OUT_DIR}/train_stdout.log"
    RC="${PIPESTATUS[0]}"
    CELL_DT=$(( $(date +%s) - CELL_T0 ))

    if [[ "${RC}" -ne 0 ]]; then
        echo "!! cell ${TAG} FAILED with rc=${RC} after ${CELL_DT}s" \
            | tee "${OUT_DIR}/TRAINING_FAILED.txt"
    else
        echo "[E4] cell ${TAG} OK in ${CELL_DT}s"
    fi
done

SWEEP_DT=$(( $(date +%s) - SWEEP_T0 ))
echo
echo "[E4] sweep done in ${SWEEP_DT}s. Per-cell summary:"
for TAG in "${TAGS[@]}"; do
    SUM="${SWEEP_DIR}/results/${TAG}/splm_sarfmass_logfreq_shakespeare_${TAG}_summary.md"
    if [[ -f "${SUM}" ]]; then
        echo "  - ${TAG}: OK   ${SUM}"
    elif [[ -f "${SWEEP_DIR}/results/${TAG}/TRAINING_FAILED.txt" ]]; then
        echo "  - ${TAG}: FAILED"
    else
        echo "  - ${TAG}: MISSING (no summary, no fail marker)"
    fi
done
