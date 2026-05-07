#!/usr/bin/env bash
# P5 -- Stage 1.5 Gumbel-softmax sparse PARF k-sweep.
#
# Goal
# ----
# Run the framework-native top-k cutoff (§5.2) on top of the P1.6
# wider-structural V_phi cell to test whether the dense O(BT^2) pair
# aggregation is the binding constraint of dense Algorithm A at this
# prototype scale.  Per the P1.6 verdict:
#
#   - dense P1.6 (top_k = T-1):  val PPL 207.58
#   - target hybrid anchors:      ~135 PPL (Q9d, Variant A)
#
# If the sparse model lands in the 150-180 range at small k, the
# framework's prescribed sparsity primitive closes (or substantially
# narrows) the architectural gap; if it lands at the same ~210 PPL
# regardless of k, the bottleneck is elsewhere (MLP V_phi capacity,
# integrator timestep, etc.) and Algorithm B / C become the primary
# next target.
#
# Cell shape
# ----------
# Same as P1.6 (em-ln vh=128, d=128, L=8, T=128, B=16) so val PPL
# is directly comparable to:
#
#   - all-attention (5-seed E1)           ~141.80 PPL
#   - Variant A HSPLM (k=4, m=4)          133.01 PPL
#   - Q9d Helmholtz AAAASSSS vh=128       134.89 PPL
#   - All-SPLM em_ln (free-gamma, v4)     173.59 PPL
#   - PARF Q9c structural (P1)            210.54 PPL
#   - PARF Q9c MLP V_phi mlp_h=16 (P1.5a) 297.22 PPL
#   - PARF Q9c wider structural (P1.6)    207.58 PPL  <-- our P5 baseline
#
# Sweep
# -----
# Default top_k in {4, 8, 16, 32}.  Set TOP_KS env var to override.
# The framework's §5.2 prescription expects a flat-then-decreasing
# curve identifying the empirical relevance threshold; the design doc
# (`docs/parf/On_Gumbel_softmax_sparsity_applied_to_V_phi.md`) recommends
# sweeping at fixed score-head width and fixed Gumbel anneal schedule
# to isolate the k-dependence.
#
# Wall-clock estimate
# -------------------
# At dense aggregation (top_k = T-1), P1.6 ran ~7.6 h MPS.  The sparse
# k-sweep adds the score head's O(BT^2 d H_s) compute (cheaper per
# pair than V_phi at the structural variant, ~half the V_phi cost),
# but the dense V_phi evaluation persists at this Stage-1.5a prototype
# (V_phi is computed at all O(T^2) pairs and then masked; the gathered
# O(T*k) optimisation is Stage-1.5b).  Net: each sweep cell should run
# in approximately the same wall-clock as P1.6, with the score-head
# cost amortised across the sweep.  Total: ~30-40 h MPS for the
# default 4-cell sweep at single seed.  Plan accordingly.
#
# Outputs
# -------
#   parf/results/structural_sparse/seed{seed}_k{K}/
#     parf_structural_sparse_k{K}_shakespeare_seed{seed}_summary.md
#     parf_structural_sparse_k{K}_shakespeare_seed{seed}_training_log.jsonl
#     parf_structural_sparse_k{K}_shakespeare_seed{seed}_loss_curve.png
#     parf_structural_sparse_k{K}_shakespeare_seed{seed}_ckpt_latest.pt   (gitignored)
#     training.log
#
# Resilience: idempotent.  Skip cells whose summary.md already exists.
#
# Optional env vars
# -----------------
#   TOP_KS="4 8 16 32"            -- whitespace-separated top-k values
#                                    to sweep.  Default "4 8 16 32".
#   SEEDS="0"                     -- whitespace-separated seed list.
#   GUMBEL_TAU_INIT=1.0           -- initial Gumbel-softmax temperature.
#   GUMBEL_TAU_MIN=0.1            -- final temperature after anneal.
#   GUMBEL_ANNEAL_FRACTION=0.8    -- fraction of total steps for the
#                                    anneal; the first 20% holds at
#                                    tau_init for score-head warm-up.
#   SCORE_HEAD_HIDDEN=32          -- score-head MLP width.
#   V_PHI_HIDDEN=128              -- V_phi internal MLP width (matches
#                                    the P1.6 wider-structural cell).
#   GRAD_CHECKPOINT=1             -- gradient-checkpoint the V_phi pair
#                                    sum.  Adds `_gc` to the output tag.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARF_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PARF_DIR}/../../.." && pwd)"

TOP_KS="${TOP_KS:-4 8 16 32}"
SEEDS="${SEEDS:-0}"
GUMBEL_TAU_INIT="${GUMBEL_TAU_INIT:-1.0}"
GUMBEL_TAU_MIN="${GUMBEL_TAU_MIN:-0.1}"
GUMBEL_ANNEAL_FRACTION="${GUMBEL_ANNEAL_FRACTION:-0.8}"
SCORE_HEAD_HIDDEN="${SCORE_HEAD_HIDDEN:-32}"
V_PHI_HIDDEN="${V_PHI_HIDDEN:-128}"
GRAD_CHECKPOINT="${GRAD_CHECKPOINT:-}"

start_ts="$(date +%s)"
echo "==========================================================="
echo " PARF Q9c -- P5 Stage 1.5 Gumbel-softmax sparse k-sweep"
echo " top_ks:        ${TOP_KS}"
echo " seeds:         ${SEEDS}"
echo " gumbel_tau:    ${GUMBEL_TAU_INIT} -> ${GUMBEL_TAU_MIN}"
echo " anneal_frac:   ${GUMBEL_ANNEAL_FRACTION}"
echo " score_head_h:  ${SCORE_HEAD_HIDDEN}"
echo " V_phi_hidden:  ${V_PHI_HIDDEN}  (matches P1.6 wider-structural)"
echo "==========================================================="

cd "${REPO_ROOT}"

n_done=0
n_skipped=0
n_failed=0
for k in ${TOP_KS}; do
    for seed in ${SEEDS}; do
        cell_dir="${PARF_DIR}/results/structural_sparse/seed${seed}_k${k}"
        mkdir -p "${cell_dir}"
        # Tag must match what train_parf.py writes; see build_config()'s
        # `_sparse_k{N}` and `_vphi{N}` suffixes.
        gc_suffix=""
        if [[ -n "${GRAD_CHECKPOINT}" ]]; then
            gc_suffix="_gc"
        fi
        tag="parf_structural_vphi${V_PHI_HIDDEN}${gc_suffix}_sparse_k${k}_shakespeare_seed${seed}"
        summary="${cell_dir}/${tag}_summary.md"
        if [[ -f "${summary}" ]]; then
            echo "[wrap] SKIP k=${k} seed=${seed} (summary exists)"
            n_skipped=$((n_skipped + 1))
            continue
        fi
        echo "[wrap] RUN  k=${k} seed=${seed}  ->  ${cell_dir}"
        train_log="${cell_dir}/training.log"

        extra_args=()
        if [[ -n "${GRAD_CHECKPOINT}" ]]; then
            extra_args+=(--grad-checkpoint)
        fi

        # Same MPS-SIGFPE workaround as run_first_quality_cell.sh: write
        # stdout+stderr directly to a file (no `| tee` pipeline).  Use
        # `tail -f training.log` for a live view.
        if ! python3 -u notebooks/conservative_arch/parf/train_parf.py \
                --mode shakespeare \
                --v-phi-kind structural \
                --v-phi-hidden "${V_PHI_HIDDEN}" \
                --top-k "${k}" \
                --score-head-hidden "${SCORE_HEAD_HIDDEN}" \
                --gumbel-tau-init "${GUMBEL_TAU_INIT}" \
                --gumbel-tau-min "${GUMBEL_TAU_MIN}" \
                --gumbel-anneal-fraction "${GUMBEL_ANNEAL_FRACTION}" \
                --seed "${seed}" \
                ${extra_args[@]+"${extra_args[@]}"} \
                &> "${train_log}"; then
            echo "[wrap] FAIL k=${k} seed=${seed}; see ${train_log}"
            echo "k=${k} seed=${seed} failed at $(date)" \
                > "${cell_dir}/TRAINING_FAILED.txt"
            n_failed=$((n_failed + 1))
            continue
        fi

        # The trainer writes outputs to parf/results/ (flat).  Move the
        # per-cell artifacts under the per-(seed, k) directory.
        for ext in summary.md training_log.jsonl loss_curve.png ckpt_latest.pt; do
            src="${PARF_DIR}/results/${tag}_${ext}"
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
echo " P5 Gumbel-softmax k-sweep wrapper done in ${elapsed}s"
echo " ran=${n_done}  skipped=${n_skipped}  failed=${n_failed}"
echo "==========================================================="
