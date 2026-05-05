#!/usr/bin/env bash
# Confirmation sweep at S=5 narrowed to gamma in {0.05, 0.10, 0.15, 0.20}.
#
# Hypothesis under test (3-seed leak-free retrain result):
#   At the leak-corrected gamma* = 0.10, the SPLM-2 vs SPLM-1 paired delta
#   is +4.71 PPL (3/3 sign-consistent, paired-t = +2.81, d_z = +1.62,
#   p ~= 0.11), which is 0.29 PPL short of the pre-registered minimum
#   effect size Delta_min = 5.0 PPL.
#
# At S=5 the t denominator drops by sqrt(4/2) ~= 1.41x, so an unchanged
# effect size of +4.71 PPL would push p well below 0.05 -- but the
# pre-registered MAGNITUDE bar (Delta_min = 5.0 PPL) stays untouched.
# This sweep firmly establishes or refutes the +4.71 PPL paired lift.
#
# Cells to run (19 total; existing leak-free seeds 0-2 are reused):
#   - SPLM-1                seeds 3, 4         (extends seeds 0-2 already on disk)
#   - SPLM-2 gamma = 0.05   seeds 0..4         (NEW gamma value)
#   - SPLM-2 gamma = 0.10   seeds 3, 4         (extends seeds 0-2 already on disk)
#   - SPLM-2 gamma = 0.15   seeds 0..4         (NEW gamma value)
#   - SPLM-2 gamma = 0.20   seeds 0..4         (NEW gamma value)
#
# Output layout:
#   first_order_ablation/results/splm1_leakfree/seed{3,4}/                       (extension in place)
#   ln_damping_sweep/results/leakfree_5seed_confirmation/gamma{0p05,0p15,0p20}/seed{0..4}/
#   ln_damping_sweep/results/leakfree_5seed_confirmation/gamma0p10/seed{3,4}/    (extension only)
#
# Wall-clock estimate: 19 cells x ~32 min/cell ~= 10.1 h on Apple MPS / 16-core CPU.
# With one occasional outlier (the buggy run had a 57-min splm1 seed 0) the
# realistic budget is ~11 h.
#
# Resilience and idempotence:
#   - A cell is SKIPPED if it already has a *_summary.md on disk. This makes
#     the script safe to re-run after partial completion or after a smoke run.
#   - A failing cell does NOT abort the rest; a TRAINING_FAILED.txt marker
#     is left for the aggregator to skip cleanly.
#
# Optional env vars:
#   CELL_LIMIT=N   -- run at most N cells then exit (0 = unlimited; default 0).
#                     Use CELL_LIMIT=1 for the cell-00 smoke test
#                     (SPLM-1 seed 3, ~32 min, verifies the
#                     causal_force=True banner before the long sweep).
#   START_FROM=N   -- skip the first N cells in the ordered list (default 0).
#                     Useful if you want to manually re-run later cells.
#
# Decision rule for the resulting RESULTS_CONFIRMATION_S5.md:
#   PRIMARY (pre-registered, magnitude):
#     Confirmed:  paired Delta_bar(SPLM-2 vs SPLM-1) at confirmation-sweep
#                 gamma* >= +5.0 PPL with sign consistency >= 4/5 seeds.
#     Refuted:    paired Delta_bar < +5.0 PPL (or sign-inverted) at
#                 confirmation-sweep gamma*.
#   SECONDARY (sign + significance, reported separately):
#     Sign + p:   paired-t two-sided p < 0.05 with sign consistency >= 4/5.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONS_DIR="$(cd "${SWEEP_DIR}/.." && pwd)"
ABLATION_DIR="${CONS_DIR}/first_order_ablation"
SARF_MASS_DIR="${CONS_DIR}/sarf_mass_variant"
LOGFREQ_PATH="${SARF_MASS_DIR}/results/logfreq_surprisal.npy"

if [[ ! -f "${LOGFREQ_PATH}" ]]; then
    echo "[conf-S5] missing logfreq surprisal table at ${LOGFREQ_PATH}" >&2
    exit 1
fi

CELL_LIMIT="${CELL_LIMIT:-0}"
START_FROM="${START_FROM:-0}"

# ---- Build the ordered cell list ----
# Each cell: "MODEL|GAMMA|SEED|OUT_DIR"
#   MODEL is "splm1" or "splm2"
#   GAMMA is the fixed-gamma value (only consumed by the splm2 trainer)
#   OUT_DIR is the absolute path where this cell writes its training artefacts
CELLS=()

# (A) SPLM-1 extension: seeds 3, 4 in place under splm1_leakfree/
for SEED in 3 4; do
    CELLS+=("splm1|0.0|${SEED}|${ABLATION_DIR}/results/splm1_leakfree/seed${SEED}")
done

# (B) SPLM-2 gamma=0.05: full S=5 (NEW gamma value)
for SEED in 0 1 2 3 4; do
    CELLS+=("splm2|0.05|${SEED}|${SWEEP_DIR}/results/leakfree_5seed_confirmation/gamma0p05/seed${SEED}")
done

# (C) SPLM-2 gamma=0.10 extension: seeds 3, 4 (seeds 0-2 reused from leakfree_3seed/gamma0p10/)
for SEED in 3 4; do
    CELLS+=("splm2|0.10|${SEED}|${SWEEP_DIR}/results/leakfree_5seed_confirmation/gamma0p10/seed${SEED}")
done

# (D) SPLM-2 gamma=0.15: full S=5 (NEW gamma value)
for SEED in 0 1 2 3 4; do
    CELLS+=("splm2|0.15|${SEED}|${SWEEP_DIR}/results/leakfree_5seed_confirmation/gamma0p15/seed${SEED}")
done

# (E) SPLM-2 gamma=0.20: full S=5 (NEW gamma value)
for SEED in 0 1 2 3 4; do
    CELLS+=("splm2|0.20|${SEED}|${SWEEP_DIR}/results/leakfree_5seed_confirmation/gamma0p20/seed${SEED}")
done

NCELLS="${#CELLS[@]}"
SWEEP_T0=$(date +%s)

echo "============================================================"
echo "[conf-S5] confirmation sweep at S=5 -- gamma in {0.05, 0.10, 0.15, 0.20}"
echo "[conf-S5] start  $(date '+%Y-%m-%d %H:%M:%S')   total cells: ${NCELLS}"
echo "[conf-S5] CELL_LIMIT=${CELL_LIMIT}   START_FROM=${START_FROM}"
echo "[conf-S5] cfg.causal_force=True (dataclass default; verified by trainer banner)"
echo "============================================================"

NRAN=0
for IDX in $(seq 0 $((NCELLS - 1))); do
    if (( IDX < START_FROM )); then
        continue
    fi
    if (( CELL_LIMIT > 0 && NRAN >= CELL_LIMIT )); then
        echo "[conf-S5] CELL_LIMIT=${CELL_LIMIT} reached after ${NRAN} cells; stopping."
        break
    fi

    IFS='|' read -r MODEL GAMMA SEED OUT_DIR <<<"${CELLS[$IDX]}"

    # Idempotent: skip if a cell already has a summary md on disk
    if compgen -G "${OUT_DIR}/*_summary.md" > /dev/null; then
        echo "[conf-S5] cell ${IDX} (${MODEL} gamma=${GAMMA} seed=${SEED}) already has a summary.md -- SKIP"
        continue
    fi
    if [[ -f "${OUT_DIR}/TRAINING_FAILED.txt" ]]; then
        echo "[conf-S5] cell ${IDX} (${MODEL} gamma=${GAMMA} seed=${SEED}) was previously marked FAILED -- SKIP (delete the marker to retry)"
        continue
    fi

    mkdir -p "${OUT_DIR}"
    TAG_SUFFIX="confS5_${MODEL}_g${GAMMA//./p}_seed${SEED}"

    echo
    echo "------------------------------------------------------------"
    echo "[conf-S5] cell ${IDX}/${NCELLS}  ${MODEL}  gamma=${GAMMA}  seed=${SEED}"
    echo "[conf-S5] start  $(date '+%Y-%m-%d %H:%M:%S')   -> ${OUT_DIR}"
    echo "------------------------------------------------------------"

    CELL_T0=$(date +%s)
    if [[ "${MODEL}" == "splm1" ]]; then
        # SPLM-1 first-order trainer; --fixed-gamma is accepted but the integrator
        # ignores it (no v-buffer, no gamma); we still pass 0.0 for CLI compatibility.
        PYTHONUNBUFFERED=1 python3 -u "${ABLATION_DIR}/train_splm_first_order.py" \
            --mode shakespeare \
            --logfreq-path "${LOGFREQ_PATH}" \
            --seed "${SEED}" \
            --tag-suffix "seed${SEED}" \
            --results-dir "${OUT_DIR}" \
            2>&1 | tee "${OUT_DIR}/train_stdout.log"
        RC="${PIPESTATUS[0]}"
    else
        # SPLM-2 LayerNorm-after-step trainer
        PYTHONUNBUFFERED=1 python3 -u "${SWEEP_DIR}/train_splm_em_ln.py" \
            --mode shakespeare \
            --logfreq-path "${LOGFREQ_PATH}" \
            --seed "${SEED}" \
            --fixed-gamma "${GAMMA}" \
            --tag-suffix "${TAG_SUFFIX}" \
            --results-dir "${OUT_DIR}" \
            2>&1 | tee "${OUT_DIR}/train_stdout.log"
        RC="${PIPESTATUS[0]}"
    fi

    CELL_DT=$(( $(date +%s) - CELL_T0 ))
    NRAN=$(( NRAN + 1 ))

    if [[ "${RC}" -ne 0 ]]; then
        echo "!! cell ${IDX} (${MODEL} gamma=${GAMMA} seed=${SEED}) FAILED with rc=${RC} after ${CELL_DT}s" \
            | tee "${OUT_DIR}/TRAINING_FAILED.txt"
    else
        echo "[conf-S5] cell ${IDX} (${MODEL} gamma=${GAMMA} seed=${SEED}) OK in ${CELL_DT}s"
    fi

    # Smoke check after the very first cell of any invocation: was the
    # causal_force=True banner present in the trainer stdout? If not, abort.
    if (( NRAN == 1 )); then
        if grep -qE "causal_force=True" "${OUT_DIR}/train_stdout.log"; then
            echo "[conf-S5] smoke: causal_force=True banner verified on cell ${IDX}."
        else
            echo "!! smoke: causal_force=True banner NOT found in ${OUT_DIR}/train_stdout.log -- aborting." >&2
            echo "FAILED smoke check on cell ${IDX}: causal_force banner missing" \
                > "${OUT_DIR}/SMOKE_FAILED.txt"
            exit 2
        fi
    fi
done

SWEEP_DT=$(( $(date +%s) - SWEEP_T0 ))
echo
echo "[conf-S5] sweep done in ${SWEEP_DT}s. Per-cell summary:"
for IDX in $(seq 0 $((NCELLS - 1))); do
    IFS='|' read -r MODEL GAMMA SEED OUT_DIR <<<"${CELLS[$IDX]}"
    LABEL="${MODEL} g=${GAMMA} seed=${SEED}"
    if [[ -f "${OUT_DIR}/TRAINING_FAILED.txt" ]]; then
        echo "  - [${IDX}] ${LABEL}: FAILED"
    elif compgen -G "${OUT_DIR}/*_summary.md" > /dev/null; then
        echo "  - [${IDX}] ${LABEL}: OK"
    else
        echo "  - [${IDX}] ${LABEL}: MISSING"
    fi
done
