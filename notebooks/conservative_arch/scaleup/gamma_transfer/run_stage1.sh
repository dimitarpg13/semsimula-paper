#!/usr/bin/env bash
# E10 γ-transfer experiment — Stage 1 driver.
#
# Pre-registered protocol: docs/Gamma_transfer_pre-registered_protocol.md
# Pre-registration commit: 75cad01
#
# Runs the three Stage-1 pilots sequentially, single seed, 4000 steps each,
# at γ ∈ {0.10, 0.30, 0.60}. ~19.8 h total wall-clock on MPS at the E9 rate.
#
# This driver honours the E9 Phase-1 launch convention:
#   - PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 (required for 64 GB unified memory)
#   - PYTHONUNBUFFERED=1 (real-time stdout)
#   - sequential arms, abort on first failure
#   - tees to a stage log under /tmp

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../../.. && pwd)"
cd "$REPO_ROOT"

STAGE_LOG=/tmp/gamma_transfer_stage1.log
STAGE_DIR="notebooks/conservative_arch/scaleup/gamma_transfer/results/stage1"
TRAINER="notebooks/conservative_arch/scaleup/gamma_transfer/train_splm_em_ln_gamma_sweep.py"

mkdir -p "$STAGE_DIR"

echo ""                                         | tee -a "$STAGE_LOG"
echo "=== E10 γ-transfer Stage 1 driver ===" | tee -a "$STAGE_LOG"
echo "[stage1] start $(date)"                 | tee -a "$STAGE_LOG"
echo "[stage1] repo: $REPO_ROOT"               | tee -a "$STAGE_LOG"
echo "[stage1] log:  $STAGE_LOG"               | tee -a "$STAGE_LOG"
echo ""                                         | tee -a "$STAGE_LOG"

run_pilot () {
  local GAMMA="$1"
  local TAG_GAMMA="$2"   # e.g. g0p10
  local TAG="stage1_${TAG_GAMMA}_seed0"
  local OUT_DIR="${STAGE_DIR}/${TAG_GAMMA}_seed0"

  mkdir -p "$OUT_DIR"

  echo "=== Stage 1 pilot, γ=${GAMMA} ===" | tee -a "$STAGE_LOG"
  echo "[stage1-${TAG_GAMMA}] start $(date)" | tee -a "$STAGE_LOG"
  PYTHONUNBUFFERED=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 \
    python3 "$TRAINER" \
      --mode pilot \
      --fixed-gamma "$GAMMA" \
      --seed 0 \
      --tag-suffix "$TAG" \
      --results-dir "$OUT_DIR" 2>&1 \
    | tee -a "$STAGE_LOG"
  local RC="${PIPESTATUS[0]}"
  echo "[stage1-${TAG_GAMMA}] end   $(date)   rc=${RC}" | tee -a "$STAGE_LOG"
  echo ""                                                | tee -a "$STAGE_LOG"
  if [ "$RC" -ne 0 ]; then
    echo "[stage1] pilot γ=${GAMMA} failed (rc=${RC}); aborting Stage 1." | tee -a "$STAGE_LOG"
    exit "$RC"
  fi
}

run_pilot 0.10 g0p10
run_pilot 0.30 g0p30
run_pilot 0.60 g0p60

echo "[stage1] DONE at $(date)" | tee -a "$STAGE_LOG"
