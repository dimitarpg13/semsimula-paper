#!/usr/bin/env bash
# First quality cell for the PARF-augmented SPLM (Q9c) prototype.
#
# Goal
# ----
# Place a single seed-0 datapoint for the structural V_phi cell on the
# matched_baseline / em_ln-leakfree / Q9d AAAASSSS shape (d=128, L=8,
# v_hidden=128, T=128, mass_mode=logfreq, 4000 steps), so we can read
# off PARF's val PPL against:
#
#   - all-attention 5-seed E1 baseline       (~141.80 mean)
#   - em-ln SPLM leakfree 1-seed              (~150)
#   - Helmholtz Q9d AAAASSSS vh=128 seed 0    (~134.89)
#   - Variant A k=4, m=4 seed 0               (~133.01)
#
# Wall-clock estimate
# -------------------
# At T=128 the V_phi pair sum is O(T^2) = 16384 entries per layer; on
# Apple MPS this should land at ~1.5x the Q9d AAAASSSS vh=128 wall-clock
# (~22 min/cell), so target ~35-50 min for the structural variant.  The
# MLP variant has a much larger (B, T, T, 3d) intermediate and may be
# 2-3x slower.
#
# Outputs
# -------
#   parf/results/parf_<v_phi_kind>_shakespeare_seed{seed}/
#     parf_<v_phi_kind>_shakespeare_seed{seed}_summary.md
#     parf_<v_phi_kind>_shakespeare_seed{seed}_training_log.jsonl
#     parf_<v_phi_kind>_shakespeare_seed{seed}_loss_curve.png
#     parf_<v_phi_kind>_shakespeare_seed{seed}_ckpt_latest.pt   (gitignored)
#
# Resilience: idempotent.  Skip cells whose summary.md already exists.
#
# Optional env vars:
#   V_PHI_KINDS="structural mlp"  -- whitespace-separated list (default
#                                    "structural" only).  Add "mlp" to
#                                    also run the unstructured ablation.
#   SEEDS="0"                     -- whitespace-separated seed list.
#   FIXED_GAMMA=x                 -- if set, fix gamma; default = freely
#                                    learned.
#   GRAD_CHECKPOINT=1             -- if set, gradient-checkpoint the V_phi
#                                    pair sum.  Required for the MLP V_phi
#                                    variant at B=16 (else MPS OOM).
#                                    Adds `_gc` to the output tag.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARF_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PARF_DIR}/../../.." && pwd)"

V_PHI_KINDS="${V_PHI_KINDS:-structural}"
SEEDS="${SEEDS:-0}"
FIXED_GAMMA="${FIXED_GAMMA:-}"
GRAD_CHECKPOINT="${GRAD_CHECKPOINT:-}"

start_ts="$(date +%s)"
echo "==========================================================="
echo " PARF Q9c — first quality cell (Algorithm A, NTP only)"
echo " v_phi_kinds: ${V_PHI_KINDS}"
echo " seeds:       ${SEEDS}"
echo " fixed_gamma: ${FIXED_GAMMA:-<freely learned>}"
echo "==========================================================="

cd "${REPO_ROOT}"

n_done=0
n_skipped=0
n_failed=0
for kind in ${V_PHI_KINDS}; do
    for seed in ${SEEDS}; do
        cell_dir="${PARF_DIR}/results/${kind}/seed${seed}"
        mkdir -p "${cell_dir}"
        summary="${cell_dir}/parf_${kind}_shakespeare_seed${seed}_summary.md"
        if [[ -f "${summary}" ]]; then
            echo "[wrap] SKIP kind=${kind} seed=${seed} (summary exists)"
            n_skipped=$((n_skipped + 1))
            continue
        fi
        echo "[wrap] RUN  kind=${kind} seed=${seed}  ->  ${cell_dir}"
        train_log="${cell_dir}/training.log"

        extra_args=()
        if [[ -n "${FIXED_GAMMA}" ]]; then
            extra_args+=(--fixed-gamma "${FIXED_GAMMA}")
        fi
        # Auto-enable gradient checkpointing for the MLP V_phi variant
        # (which OOMs on 16 GB MPS at B=16 without it), or whenever
        # the GRAD_CHECKPOINT env var is explicitly set.
        if [[ -n "${GRAD_CHECKPOINT}" ]] || [[ "${kind}" == "mlp" ]]; then
            extra_args+=(--grad-checkpoint)
        fi

        # Train.  Redirect stdout+stderr directly to training.log
        # (no `| tee` pipeline -- the bash subshell + pipe was
        # observed to trigger an MPS-only SIGFPE on torch 2.x at
        # process startup, before any of train_parf.py's print
        # statements ran; a clean `&> file` redirection avoids it).
        # The `tail -f` command in scripts/tail_first_quality_cell.sh
        # offers an equivalent live-watch experience for users.
        # `${extra_args[@]+"${extra_args[@]}"}` is the bash idiom that
        # protects against the empty-array unbound-variable error
        # under `set -u` when FIXED_GAMMA is unset.
        if ! python3 -u notebooks/conservative_arch/parf/train_parf.py \
                --mode shakespeare \
                --v-phi-kind "${kind}" \
                --seed "${seed}" \
                ${extra_args[@]+"${extra_args[@]}"} \
                &> "${train_log}"; then
            echo "[wrap] FAIL kind=${kind} seed=${seed}; see ${train_log}"
            echo "kind=${kind} seed=${seed} failed at $(date)" \
                > "${cell_dir}/TRAINING_FAILED.txt"
            n_failed=$((n_failed + 1))
            continue
        fi

        # The trainer writes outputs to parf/results/ (flat).  Move the
        # per-cell artifacts under the per-seed directory.
        for ext in summary.md training_log.jsonl loss_curve.png ckpt_latest.pt; do
            src="${PARF_DIR}/results/parf_${kind}_shakespeare_seed${seed}_${ext}"
            if [[ -f "${src}" ]]; then
                mv "${src}" "${cell_dir}/"
            fi
        done

        n_done=$((n_done + 1))
    done
done

elapsed=$(( $(date +%s) - start_ts ))
echo ""
echo "==========================================================="
echo " PARF first-quality-cell wrapper done in ${elapsed}s"
echo " ran=${n_done}  skipped=${n_skipped}  failed=${n_failed}"
echo "==========================================================="
