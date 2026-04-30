#!/usr/bin/env bash
# Extract h^(L) quadruples from all 6 SPLM checkpoints (3 SPLM-1 + 3
# SPLM em_ln gamma*=0.30) of the SPLM-1 ablation sweep. Phase 1 of E7.
set -euo pipefail

cd "$(dirname "$0")/.."

ABL_RESULTS="../results"
OUT_ROOT="results/quadruples"
mkdir -p "$OUT_ROOT"

T0_ALL="$(date +%s)"
for ARM in splm1 splm2_gamma0p30; do
    case "$ARM" in
        splm1)
            CKPT_PREFIX="splm_first_order_shakespeare"
            ;;
        splm2_gamma0p30)
            CKPT_PREFIX="splm_em_ln_shakespeare"
            ;;
    esac
    for SEED in 0 1 2; do
        CKPT="${ABL_RESULTS}/${ARM}/seed${SEED}/${CKPT_PREFIX}_seed${SEED}_ckpt_latest.pt"
        OUT="${OUT_ROOT}/${ARM}__seed${SEED}"
        if [[ -f "${OUT}/quadruples.npz" ]]; then
            echo "[extract] ${ARM}/seed${SEED}: already done, skipping"
            continue
        fi
        echo "[extract] ${ARM}/seed${SEED}  ckpt=${CKPT}  out=${OUT}"
        T0="$(date +%s)"
        python3 extract_splm_quadruples.py \
            --checkpoint "${CKPT}" \
            --output_dir "${OUT}" \
            --device cpu
        DUR=$(( $(date +%s) - T0 ))
        echo "[extract] ${ARM}/seed${SEED}  done in ${DUR}s"
    done
done
echo "[extract] ALL DONE in $(( $(date +%s) - T0_ALL ))s"
