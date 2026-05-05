#!/usr/bin/env bash
# Run Tier 2a (em_ln freely-trained γ) + Tier 3 (em_sg + em_gm) under the
# leak-corrected integrator (causal_force=True is now the SPLMSARFMassConfig
# default). All three variants share the same train.py infrastructure.
#
# Output:
#   results/em_ln_shakespeare_leakfree_freegamma_*
#   results/em_sg_lam1e-03_shakespeare_leakfree_*
#   results/em_gm_K64_shakespeare_leakfree_*
#
# Wall-clock estimate (CPU): ~90 min total (3 × ~30 min).
# Wall-clock estimate (MPS): ~60 min total (3 × ~20 min if MPS warmth helps).

set -euo pipefail

cd "$(dirname "$0")/.."
HERE="$PWD"
SCRIPTS_DIR="$HERE/scripts"
RESULTS_DIR="$HERE/results"
LOGS_DIR="$RESULTS_DIR/leakfree_tiers_2_3_logs"
mkdir -p "$LOGS_DIR"

DEVICE="${DEVICE:-mps}"
SEED="${SEED:-0}"

echo "[tiers-2-3] device=$DEVICE  seed=$SEED  PWD=$HERE  LOGS=$LOGS_DIR"
echo "[tiers-2-3] starting at $(date '+%Y-%m-%d %H:%M:%S')"

# Tier 2a: em_ln with freely-trained gamma (no --fixed-gamma; cfg default).
# Note that train.py does not currently take --fixed-gamma; so SPLMSARFMassLNConfig's
# default fixed_gamma=None is used, which means gamma is a learnable parameter.
T0=$(date +%s)
echo "[tiers-2-3] === Tier 2a: em_ln freely-trained gamma ==="
python3 train.py --variant ln --mode shakespeare --device "$DEVICE" --seed "$SEED" \
  > "$LOGS_DIR/em_ln_freegamma_seed${SEED}_stdout.log" \
  2> "$LOGS_DIR/em_ln_freegamma_seed${SEED}_stderr.log"
T1=$(date +%s)
echo "[tiers-2-3] Tier 2a done in $((T1-T0))s"

# Tier 3a: em_sg with lambda_v0 = 1e-3 (the value used in v2 retrain).
echo "[tiers-2-3] === Tier 3a: em_sg lambda_v0=1e-3 ==="
python3 train.py --variant sg --mode shakespeare --lambda-v0 1e-3 \
  --device "$DEVICE" --seed "$SEED" \
  > "$LOGS_DIR/em_sg_lam1e-03_seed${SEED}_stdout.log" \
  2> "$LOGS_DIR/em_sg_lam1e-03_seed${SEED}_stderr.log"
T2=$(date +%s)
echo "[tiers-2-3] Tier 3a done in $((T2-T1))s"

# Tier 3b: em_gm with K=64 wells.
echo "[tiers-2-3] === Tier 3b: em_gm K=64 ==="
python3 train.py --variant gm --mode shakespeare --gm-K 64 \
  --device "$DEVICE" --seed "$SEED" \
  > "$LOGS_DIR/em_gm_K64_seed${SEED}_stdout.log" \
  2> "$LOGS_DIR/em_gm_K64_seed${SEED}_stderr.log"
T3=$(date +%s)
echo "[tiers-2-3] Tier 3b done in $((T3-T2))s"

echo "[tiers-2-3] === SUMMARY ==="
echo "[tiers-2-3] Tier 2a (em_ln free-γ):  $((T1-T0))s"
echo "[tiers-2-3] Tier 3a (em_sg λ=1e-3):  $((T2-T1))s"
echo "[tiers-2-3] Tier 3b (em_gm K=64):    $((T3-T2))s"
echo "[tiers-2-3] TOTAL:                    $((T3-T0))s"
echo "[tiers-2-3] finished at $(date '+%Y-%m-%d %H:%M:%S')"
