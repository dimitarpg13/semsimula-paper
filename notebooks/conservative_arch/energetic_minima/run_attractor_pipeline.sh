#!/usr/bin/env bash
#
# Orchestrates the attractor-extraction + 3D-landscape pipeline for all
# three energetic-minima variants (LN, SG, GM) plus the SARF+mass baseline.
#
# Assumes training checkpoints are already in their respective results/
# folders.  Runs every step and writes its output under
# attractor_analysis/results/ with tag = em_<variant>_<extra>.
#
# Usage:
#   bash run_attractor_pipeline.sh
#
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ATTR_DIR="${ROOT}/attractor_analysis"
EM_DIR="${ROOT}/energetic_minima"

# (ckpt_path, tag, nominal_L_train, sim_dt).
declare -a RUNS=(
  "${ROOT}/sarf_mass_variant/results/splm_sarfmass_logfreq_shakespeare_ckpt_latest.pt|em_base|8|1.0"
  "${EM_DIR}/results/em_ln_shakespeare_ckpt_latest.pt|em_ln|8|1.0"
  "${EM_DIR}/results/em_sg_lam1e-03_shakespeare_ckpt_latest.pt|em_sg|8|1.0"
  "${EM_DIR}/results/em_gm_K64_shakespeare_ckpt_latest.pt|em_gm|8|1.0"
)

echo "[pipeline] === step 1: attractor extraction (dynamical) ==="
for spec in "${RUNS[@]}"; do
  IFS='|' read -r ckpt tag Ltrain dt <<< "$spec"
  if [[ ! -f "$ckpt" ]]; then
    echo "  [skip] $tag: ckpt not found at $ckpt"
    continue
  fi
  echo "  [run]  $tag -> $ckpt"
  python3 -u "${ATTR_DIR}/attractor_extraction.py" \
    --ckpt "$ckpt" --tag "$tag" \
    --mode dynamical --n_sim_steps "$Ltrain" --sim_dt "$dt" \
    --n_gauss 96 --n_tok 96 --n_real 96 \
    --K_min 2 --K_max 10 \
    2>&1 | tail -n 20
done

echo
echo "[pipeline] === step 2: 3D landscape rendering (dialogue prompt) ==="
for spec in "${RUNS[@]}"; do
  IFS='|' read -r ckpt tag Ltrain dt <<< "$spec"
  if [[ ! -f "$ckpt" ]]; then
    echo "  [skip] $tag"
    continue
  fi
  echo "  [run]  landscape3d for $tag"
  python3 -u "${ATTR_DIR}/landscape_3d.py" \
    --ckpt "$ckpt" --tag "landscape3d_${tag}" \
    --n_sim_steps "$Ltrain" --sim_dt "$dt" \
    --prompts dialogue \
    --n_gauss 64 --n_tok 64 --n_real 64 --grid_n 50 \
    2>&1 | tail -n 10
done

echo
echo "[pipeline] done.  See ${ATTR_DIR}/results/ for outputs."
