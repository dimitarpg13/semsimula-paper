#!/usr/bin/env bash
# Tier 2b: run attractor_extraction.py --mode dynamical on the leak-free
# freely-trained-γ em_ln checkpoint produced by Tier 2a (run_leakfree_tiers_2_3.sh).
#
# Pre: Tier 2a (em_ln free-γ) must have produced
#   notebooks/conservative_arch/energetic_minima/results/em_ln_shakespeare_ckpt_latest.pt
# Output:
#   notebooks/conservative_arch/attractor_analysis/results/attractors_em_ln_leakfree_freegamma_seed0_*

set -euo pipefail

CKPT="/Users/dimitargueorguiev/git/ml/semsimula/notebooks/conservative_arch/energetic_minima/results/em_ln_shakespeare_ckpt_latest.pt"
TAG="em_ln_leakfree_freegamma_seed0"

if [[ ! -f "$CKPT" ]]; then
  echo "[tier2b] ERROR: ckpt not found: $CKPT"
  echo "[tier2b]        Run scripts/run_leakfree_tiers_2_3.sh first (Tier 2a)."
  exit 1
fi

cd /Users/dimitargueorguiev/git/ml/semsimula/notebooks/conservative_arch/attractor_analysis

echo "[tier2b] starting at $(date '+%Y-%m-%d %H:%M:%S')"
echo "[tier2b] ckpt=$CKPT  tag=$TAG"

T0=$(date +%s)
python3 attractor_extraction.py \
  --ckpt "$CKPT" \
  --tag "$TAG" \
  --mode dynamical \
  --device cpu \
  --seed 0
T1=$(date +%s)

echo "[tier2b] done in $((T1-T0))s"
echo "[tier2b] finished at $(date '+%Y-%m-%d %H:%M:%S')"
