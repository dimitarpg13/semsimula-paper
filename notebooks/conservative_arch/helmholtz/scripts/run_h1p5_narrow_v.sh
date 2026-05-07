#!/usr/bin/env bash
# H1.5: V_theta-narrow ablation for the Helmholtz hybrid (Q9d).
#
# Goal: clear the FLOP arm at T=1024 by halving (and quartering)
# V_theta's hidden width.  At v_hidden=512 the SPLM step costs
# ~3.94 MFLOPs/tok and dominates an attention block (~0.94 MFLOPs/tok
# at T=1024), so every Q9d cell trained in H1 fails the +30%
# decode-FLOP-reduction arm of the pre-registered title rule.
#
# Cells (4 total = 2 schedules x 2 v_hidden):
#   00  AAAASSSS  vh=128   (n_S=4, n_A=4)  -- H1 best PPL (135.03 @ vh=512)
#   01  AAAASSSS  vh=256   (n_S=4, n_A=4)  -- H1 best PPL
#   02  AASSSSSS  vh=128   (n_S=6, n_A=2)  -- only Q9d-vs-Variant-A win in H1
#   03  AASSSSSS  vh=256   (n_S=6, n_A=2)
#
# Why these two schedules:
#   - AAAASSSS is the best PPL cell from H1 (and the cleanest direct
#     comparison vs Variant A k=4,m=4 = 133.01).
#   - AASSSSSS is the only cell where Q9d outright outperforms its
#     Variant A analogue (140.60 vs 147.28, -6.7 PPL at iso-params,
#     iso-FLOPs).  Higher n_S means V_theta dominates more strongly,
#     so it has the most to gain from narrowing v_hidden.
#
# Output layout:
#   helmholtz/results/h1p5_narrow_v/<schedule>_vh<V>/seed0/{...}
#
# Wall-clock estimate (per cell on Apple MPS):
#   ~22 min/cell at vh=128, ~26 min/cell at vh=256 -> ~96 min total.
#   Faster than H1 because the narrower V_theta is the dominant
#   per-step cost and we are halving / quartering it.
#
# Resilience: same as H1 -- skip if summary.md exists, continue past
# failure.  Idempotent.
#
# Optional env vars:
#   CELL_LIMIT=N   -- run at most N cells (0 = unlimited; default 0).
#   START_FROM=N   -- skip the first N cells (default 0).
#   FIXED_GAMMA=x  -- if set, use fixed gamma; default = freely learned.
#   SEED=N         -- seed (default 0).

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
    echo "[helm-h1p5] missing logfreq surprisal table at ${LOGFREQ_PATH}" >&2
    exit 1
fi

CELL_LIMIT="${CELL_LIMIT:-0}"
START_FROM="${START_FROM:-0}"
FIXED_GAMMA="${FIXED_GAMMA:-}"
SEED="${SEED:-0}"

# Parallel arrays: schedule[i], v_hidden[i].
SCHEDULES=(
    "AAAASSSS"   "AAAASSSS"
    "AASSSSSS"   "AASSSSSS"
)
V_HIDDENS=(
    128          256
    128          256
)

OUT_BASE="${HELM_DIR}/results/h1p5_narrow_v"
mkdir -p "${OUT_BASE}"

n_cells="${#SCHEDULES[@]}"
echo "[helm-h1p5] cells: ${n_cells}"
echo "[helm-h1p5] seed:  ${SEED}"
if [[ -n "${FIXED_GAMMA}" ]]; then
    echo "[helm-h1p5] gamma: fixed at ${FIXED_GAMMA}"
else
    echo "[helm-h1p5] gamma: freely learned (default)"
fi
echo "[helm-h1p5] output base: ${OUT_BASE}"
echo "[helm-h1p5] CELL_LIMIT=${CELL_LIMIT} START_FROM=${START_FROM}"
echo

i=0
done_count=0
for (( idx=0; idx<n_cells; idx++ )); do
    SIGMA="${SCHEDULES[$idx]}"
    VH="${V_HIDDENS[$idx]}"

    if (( i < START_FROM )); then
        i=$((i + 1)); continue
    fi
    if (( CELL_LIMIT > 0 && done_count >= CELL_LIMIT )); then
        echo "[helm-h1p5] CELL_LIMIT reached, stopping."
        break
    fi

    if [[ -n "${FIXED_GAMMA}" ]]; then
        FG_TAG="_g${FIXED_GAMMA//./p}"
        FG_FLAG="--fixed-gamma ${FIXED_GAMMA}"
    else
        FG_TAG=""
        FG_FLAG=""
    fi

    CELL_DIR="${OUT_BASE}/${SIGMA}_vh${VH}${FG_TAG}/seed${SEED}"
    mkdir -p "${CELL_DIR}"
    SUMMARY_GLOB="${CELL_DIR}/helm_${SIGMA}_vh${VH}${FG_TAG}_shakespeare_seed${SEED}_summary.md"

    if [[ -f "${SUMMARY_GLOB}" ]]; then
        echo "[helm-h1p5] skip cell #$(printf '%02d' $i)  ${SIGMA} vh=${VH}${FG_TAG}  "\
             "(summary exists at ${SUMMARY_GLOB})"
        i=$((i + 1)); continue
    fi

    echo
    echo "[helm-h1p5] cell #$(printf '%02d' $i)  ${SIGMA}  vh=${VH}${FG_TAG}  "\
         "seed=${SEED}  ->  ${CELL_DIR}"
    echo "[helm-h1p5] $(date -u +'%Y-%m-%dT%H:%M:%SZ') start"

    pushd "${HELM_DIR}" >/dev/null
    if python3 train_helmholtz.py \
        --mode shakespeare \
        --schedule "${SIGMA}" \
        --v-hidden "${VH}" \
        ${FG_FLAG} \
        --logfreq-path "${LOGFREQ_PATH}" \
        --seed "${SEED}" \
        2>&1 | tee "${CELL_DIR}/training.log"; then
        echo "[helm-h1p5] cell #$(printf '%02d' $i) training succeeded."
    else
        echo "[helm-h1p5] cell #$(printf '%02d' $i) training FAILED (rc=$?)" \
            >> "${CELL_DIR}/TRAINING_FAILED.txt"
        echo "[helm-h1p5] continuing past failed cell."
        popd >/dev/null
        i=$((i + 1)); done_count=$((done_count + 1)); continue
    fi
    popd >/dev/null

    BASE_TAG="helm_${SIGMA}_vh${VH}${FG_TAG}_shakespeare_seed${SEED}"
    for ext in training_log.jsonl ckpt_latest.pt loss_curve.png summary.md; do
        SRC="${HELM_DIR}/results/${BASE_TAG}_${ext}"
        if [[ -f "${SRC}" ]]; then
            mv "${SRC}" "${CELL_DIR}/"
        fi
    done

    echo "[helm-h1p5] $(date -u +'%Y-%m-%dT%H:%M:%SZ') done."
    i=$((i + 1)); done_count=$((done_count + 1))
done

echo
echo "[helm-h1p5] sweep complete.  cells run: ${done_count}"
echo "[helm-h1p5] outputs under: ${OUT_BASE}"
echo "[helm-h1p5] aggregate:  python3 ${HELM_DIR}/aggregate_h1p5.py"
