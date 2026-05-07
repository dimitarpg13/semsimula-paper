#!/usr/bin/env bash
# H0 + H1: schedule sweep for the Helmholtz hybrid (Q9d).
#
# H0 = first cell only (smoke at the *real* shakespeare config), CELL_LIMIT=1.
# H1 = full 7-cell schedule sweep at the matched-baseline budget.
#
# Architecture: layer-type Helmholtz hybrid, single shared V_theta on
# every S-block, per-layer attention on every A-block, total depth
# L = 8 (matches matched_baseline n_layer=8 and Variant A HSPLM
# n_attn + n_splm = 8).  All cells trained on Tiny Shakespeare for
# 4000 steps at the shakespeare config (d=128, v_hidden=512, v_depth=3,
# n_head=4, batch=16, block=128, AdamW(0.9,0.95), causal_force=True,
# ln_after_s_step=True, mass_mode='logfreq', S=1 seed).
#
# Cells in run order:
#   00  AAAASSSS    bottom_a_LA4         (Variant A (k=4, m=4) analogue, H0)
#   01  SAAAAAAS    sandwich_k1          (boundary-case mechanism, doc §6 cell 1)
#   02  SASASASA    interleaved-half     (step-function R²ψ test, doc §6 cell 2)
#   03  SSSSSSSA    top_a_LA1            (single-attention hybrid, doc §6 cell 3)
#   04  SSAAAASS    sandwich_k2          (wider sandwich)
#   05  ASSSSSSA    inverse_sandwich_k1  (routing at boundaries)
#   06  AASSSSSS    bottom_a_LA2         (Variant A (k=2, m=6) analogue)
#
# Output layout:
#   helmholtz/results/h1_sweep/<schedule>/seed0/{summary.md, ckpt.pt, log.jsonl}
#
# Wall-clock estimate (per cell on Apple MPS):
#   ~32 min/cell × 7 cells -> ~3.7 h.  Add ~30 min slack -> ~4 h total.
#
# Resilience and idempotence (matches hybrid/scripts/run_h1_layer_split_sweep.sh):
#   - A cell is SKIPPED if it already has *_summary.md on disk.
#   - A failing cell does NOT abort the rest; TRAINING_FAILED.txt marker
#     is left for follow-up review.
#
# Optional env vars:
#   CELL_LIMIT=N   -- run at most N cells then exit (0 = unlimited; default 0).
#                     Use CELL_LIMIT=1 for the H0 smoke (just AAAASSSS).
#   START_FROM=N   -- skip the first N cells (default 0).
#   FIXED_GAMMA=x  -- if set, train all cells with gamma fixed at x.
#                     If unset (default), gamma is freely learned.
#                     Recommended H0/H1 default: free gamma (no constraint).
#   SEED=N         -- training seed (default 0).

set -uo pipefail

# Force unbuffered Python stdio so each `[helm-train]` line flushes
# immediately when the trainer's stdout is piped through `tee` to a
# log file (otherwise Python block-buffers stdout to ~8 KB, hiding
# step-level progress for several minutes at a time).
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONS_DIR="$(cd "${HELM_DIR}/.." && pwd)"
SARF_MASS_DIR="${CONS_DIR}/sarf_mass_variant"
LOGFREQ_PATH="${SARF_MASS_DIR}/results/logfreq_surprisal.npy"

if [[ ! -f "${LOGFREQ_PATH}" ]]; then
    echo "[helm-h1] missing logfreq surprisal table at ${LOGFREQ_PATH}" >&2
    exit 1
fi

CELL_LIMIT="${CELL_LIMIT:-0}"
START_FROM="${START_FROM:-0}"
FIXED_GAMMA="${FIXED_GAMMA:-}"
SEED="${SEED:-0}"

# ---- Cell list (schedule strings) ----
# Order: H0 anchor first (Variant A analogue), then doc §6 cells 1-3,
# then breadth cells.
SCHEDULES=(
    "AAAASSSS"   # bottom_a_LA4         (H0, Variant A (4,4) analogue)
    "SAAAAAAS"   # sandwich_k1          (boundary-case mechanism)
    "SASASASA"   # interleaved-half     (step-function R²ψ test)
    "SSSSSSSA"   # top_a_LA1            (single-attention hybrid)
    "SSAAAASS"   # sandwich_k2          (wider sandwich)
    "ASSSSSSA"   # inverse_sandwich_k1  (routing at boundaries)
    "AASSSSSS"   # bottom_a_LA2         (Variant A (2,6) analogue)
)

OUT_BASE="${HELM_DIR}/results/h1_sweep"
mkdir -p "${OUT_BASE}"

# ---- Banner ----
echo "[helm-h1] cells: ${#SCHEDULES[@]}"
echo "[helm-h1] seed:  ${SEED}"
if [[ -n "${FIXED_GAMMA}" ]]; then
    echo "[helm-h1] gamma: fixed at ${FIXED_GAMMA}"
else
    echo "[helm-h1] gamma: freely learned (default)"
fi
echo "[helm-h1] output base: ${OUT_BASE}"
echo "[helm-h1] CELL_LIMIT=${CELL_LIMIT} START_FROM=${START_FROM}"
echo

# ---- Run cells ----
i=0
done_count=0
for SIGMA in "${SCHEDULES[@]}"; do
    if (( i < START_FROM )); then
        i=$((i + 1)); continue
    fi
    if (( CELL_LIMIT > 0 && done_count >= CELL_LIMIT )); then
        echo "[helm-h1] CELL_LIMIT reached, stopping."
        break
    fi

    if [[ -n "${FIXED_GAMMA}" ]]; then
        FG_TAG="_g${FIXED_GAMMA//./p}"
        FG_FLAG="--fixed-gamma ${FIXED_GAMMA}"
    else
        FG_TAG=""
        FG_FLAG=""
    fi

    CELL_DIR="${OUT_BASE}/${SIGMA}${FG_TAG}/seed${SEED}"
    mkdir -p "${CELL_DIR}"
    SUMMARY_GLOB="${CELL_DIR}/helm_${SIGMA}${FG_TAG}_shakespeare_seed${SEED}_summary.md"

    if [[ -f "${SUMMARY_GLOB}" ]]; then
        echo "[helm-h1] skip cell #$(printf '%02d' $i)  ${SIGMA}${FG_TAG}  "\
             "(summary exists at ${SUMMARY_GLOB})"
        i=$((i + 1)); continue
    fi

    echo
    echo "[helm-h1] cell #$(printf '%02d' $i)  ${SIGMA}${FG_TAG}  "\
         "seed=${SEED}  ->  ${CELL_DIR}"
    echo "[helm-h1] $(date -u +'%Y-%m-%dT%H:%M:%SZ') start"

    pushd "${HELM_DIR}" >/dev/null
    if python3 train_helmholtz.py \
        --mode shakespeare \
        --schedule "${SIGMA}" \
        ${FG_FLAG} \
        --logfreq-path "${LOGFREQ_PATH}" \
        --seed "${SEED}" \
        2>&1 | tee "${CELL_DIR}/training.log"; then
        echo "[helm-h1] cell #$(printf '%02d' $i) training succeeded."
    else
        echo "[helm-h1] cell #$(printf '%02d' $i) training FAILED (rc=$?)" \
            >> "${CELL_DIR}/TRAINING_FAILED.txt"
        echo "[helm-h1] continuing past failed cell."
        popd >/dev/null
        i=$((i + 1)); done_count=$((done_count + 1)); continue
    fi
    popd >/dev/null

    # Move the trainer's flat outputs into CELL_DIR.
    BASE_TAG="helm_${SIGMA}${FG_TAG}_shakespeare_seed${SEED}"
    for ext in training_log.jsonl ckpt_latest.pt loss_curve.png summary.md; do
        SRC="${HELM_DIR}/results/${BASE_TAG}_${ext}"
        if [[ -f "${SRC}" ]]; then
            mv "${SRC}" "${CELL_DIR}/"
        fi
    done

    echo "[helm-h1] $(date -u +'%Y-%m-%dT%H:%M:%SZ') done."
    i=$((i + 1)); done_count=$((done_count + 1))
done

echo
echo "[helm-h1] sweep complete.  cells run: ${done_count}"
echo "[helm-h1] outputs under: ${OUT_BASE}"
echo "[helm-h1] next step: aggregate results and compare against:"
echo "      - all-attention baseline (matched_baseline_model, val PPL ~150)"
echo "      - all-SPLM em_ln baseline (val PPL ~173.59 free-γ)"
echo "      - Variant A HSPLM best (val PPL 133.01 at (k=4, m=4))"
echo "[helm-h1] aggregate:  python3 ${HELM_DIR}/aggregate_h1.py"
echo "[helm-h1] FLOP arm:   python3 ${HELM_DIR}/decode_flop_pareto.py"
