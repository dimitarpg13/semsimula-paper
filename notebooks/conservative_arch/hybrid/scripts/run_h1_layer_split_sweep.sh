#!/usr/bin/env bash
# H0 + H1: layer-split sweep for the two-stage hybrid (Variant A).
#
# H0 = first cell only (smoke at the *real* shakespeare config, not toy
#      smoke mode), CELL_LIMIT=1.
# H1 = full 5-cell layer-split sweep at the matched-baseline budget.
#
# Architecture: two-stage hybrid (k attention blocks + m SPLM steps),
# total budget L = k + m = 8 layers (matches matched_baseline n_layer=8).
# All cells trained on Tiny Shakespeare for 4000 steps at the
# shakespeare config (d=128, L_total=8, v_hidden=512, v_depth=3,
# n_head=4, batch=16, block=128, AdamW(0.9,0.95), causal_force=True,
# ln_after_step=True, mass_mode='logfreq', S=1 seed).
#
# Cells in run order:
#   00  hybrid k=4 m=4    (H0 smoke -- balanced split, the natural anchor)
#   01  hybrid k=2 m=6    (SPLM-heavy)
#   02  hybrid k=3 m=5
#   03  hybrid k=5 m=3
#   04  hybrid k=6 m=2    (attention-heavy)
#
# Output layout:
#   hybrid/results/h1_sweep/k{k}_m{m}/seed0/{summary.md, ckpt.pt, log.jsonl}
#
# Wall-clock estimate (per cell on Apple MPS):
#   - All-attention reference cell at L=8: ~25 min in matched_baseline runs.
#   - Pure SPLM em_ln cell at L=8: ~33 min in confirmation sweep runs.
#   - Hybrids should be in between; budget 30 min/cell -> 5 cells -> ~2.5 h.
#   Add ~30 min slack -> ~3 h MPS for the full H0 + H1.
#
# Resilience and idempotence (matches run_confirmation_5seed_sweep.sh):
#   - A cell is SKIPPED if it already has *_summary.md on disk.
#   - A failing cell does NOT abort the rest; TRAINING_FAILED.txt marker
#     is left for follow-up review.
#
# Optional env vars:
#   CELL_LIMIT=N   -- run at most N cells then exit (0 = unlimited; default 0).
#                     Use CELL_LIMIT=1 for the H0 smoke test (just k=4, m=4).
#   START_FROM=N   -- skip the first N cells (default 0).
#   FIXED_GAMMA=x  -- if set, train all cells with gamma fixed at x.
#                     If unset (default), gamma is freely learned.
#                     Recommended H0/H1 default: free gamma (no constraint).
#
# Decision rule (pre-registered title-justification, see
# the v4 title-justification rule §6.5):
#   "Efficient" is justified iff some hybrid (k, m) achieves val PPL
#   within +5 PPL of the all-attention baseline (~150 on Tiny Shakespeare
#   at this width) AND its analytical decode-FLOP cost at T=1024 is
#   >= 30% lower than all-attention, both at S=3 with sign 3/3.
# H1 is the S=1 reconnaissance step toward that decision.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HYBRID_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONS_DIR="$(cd "${HYBRID_DIR}/.." && pwd)"
SARF_MASS_DIR="${CONS_DIR}/sarf_mass_variant"
LOGFREQ_PATH="${SARF_MASS_DIR}/results/logfreq_surprisal.npy"

if [[ ! -f "${LOGFREQ_PATH}" ]]; then
    echo "[h1] missing logfreq surprisal table at ${LOGFREQ_PATH}" >&2
    exit 1
fi

CELL_LIMIT="${CELL_LIMIT:-0}"
START_FROM="${START_FROM:-0}"
FIXED_GAMMA="${FIXED_GAMMA:-}"
SEED="${SEED:-0}"

# ---- Cell list (k, m) ----
# Order: H0 first, then SPLM-heavy -> attention-heavy.
KM_PAIRS=("4 4" "2 6" "3 5" "5 3" "6 2")

OUT_BASE="${HYBRID_DIR}/results/h1_sweep"
mkdir -p "${OUT_BASE}"

# ---- Banner ----
echo "[h1] cells: ${#KM_PAIRS[@]}"
echo "[h1] seed:  ${SEED}"
if [[ -n "${FIXED_GAMMA}" ]]; then
    echo "[h1] gamma: fixed at ${FIXED_GAMMA}"
else
    echo "[h1] gamma: freely learned (default)"
fi
echo "[h1] output base: ${OUT_BASE}"
echo "[h1] CELL_LIMIT=${CELL_LIMIT} START_FROM=${START_FROM}"
echo

# ---- Run cells ----
i=0
done_count=0
for KM in "${KM_PAIRS[@]}"; do
    if (( i < START_FROM )); then
        i=$((i + 1)); continue
    fi
    if (( CELL_LIMIT > 0 && done_count >= CELL_LIMIT )); then
        echo "[h1] CELL_LIMIT reached, stopping."
        break
    fi
    K="$(echo $KM | awk '{print $1}')"
    M="$(echo $KM | awk '{print $2}')"

    if [[ -n "${FIXED_GAMMA}" ]]; then
        FG_TAG="_g${FIXED_GAMMA//./p}"
        FG_FLAG="--fixed-gamma ${FIXED_GAMMA}"
    else
        FG_TAG=""
        FG_FLAG=""
    fi

    CELL_DIR="${OUT_BASE}/k${K}_m${M}${FG_TAG}/seed${SEED}"
    mkdir -p "${CELL_DIR}"
    SUMMARY_GLOB="${CELL_DIR}/hybrid_k${K}_m${M}${FG_TAG}_shakespeare_seed${SEED}_summary.md"

    if [[ -f "${SUMMARY_GLOB}" ]]; then
        echo "[h1] skip cell #$(printf '%02d' $i)  k=${K} m=${M}${FG_TAG}  "\
             "(summary exists at ${SUMMARY_GLOB})"
        i=$((i + 1)); continue
    fi

    echo
    echo "[h1] cell #$(printf '%02d' $i)  k=${K} m=${M}${FG_TAG}  "\
         "seed=${SEED}  ->  ${CELL_DIR}"
    echo "[h1] $(date -u +'%Y-%m-%dT%H:%M:%SZ') start"

    # Train.  Trainer writes its outputs into hybrid/results/ by default;
    # we move them into CELL_DIR after the run for a clean layout.
    pushd "${HYBRID_DIR}" >/dev/null
    if python3 train_splm_hybrid.py \
        --mode shakespeare \
        --n-attn "${K}" --n-splm "${M}" \
        ${FG_FLAG} \
        --logfreq-path "${LOGFREQ_PATH}" \
        --seed "${SEED}" \
        2>&1 | tee "${CELL_DIR}/training.log"; then
        echo "[h1] cell #$(printf '%02d' $i) training succeeded."
    else
        echo "[h1] cell #$(printf '%02d' $i) training FAILED (rc=$?)" \
            >> "${CELL_DIR}/TRAINING_FAILED.txt"
        echo "[h1] continuing past failed cell."
        popd >/dev/null
        i=$((i + 1)); done_count=$((done_count + 1)); continue
    fi
    popd >/dev/null

    # Move the trainer's flat outputs into CELL_DIR.
    BASE_TAG="hybrid_k${K}_m${M}${FG_TAG}_shakespeare_seed${SEED}"
    for ext in training_log.jsonl ckpt_latest.pt loss_curve.png summary.md; do
        SRC="${HYBRID_DIR}/results/${BASE_TAG}_${ext}"
        if [[ -f "${SRC}" ]]; then
            mv "${SRC}" "${CELL_DIR}/"
        fi
    done

    echo "[h1] $(date -u +'%Y-%m-%dT%H:%M:%SZ') done."
    i=$((i + 1)); done_count=$((done_count + 1))
done

echo
echo "[h1] sweep complete.  cells run: ${done_count}"
echo "[h1] outputs under: ${OUT_BASE}"
echo "[h1] next step: aggregate results and compare against:"
echo "      - all-attention baseline (matched_baseline_model.MatchedGPT, val PPL ~150)"
echo "      - all-SPLM em_ln baseline (val PPL ~173.59 free-gamma)"
echo "[h1] decision rule: the v4 title-justification rule §6.5"
