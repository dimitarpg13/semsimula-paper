#!/usr/bin/env bash
# E9 Phase 1 → E10 Stage 1 chain driver.
#
# Pre-registered protocols:
#   E9:  docs/SPLM_scaleup_pre-registered_protocol.md  (commit 17a3795)
#   E10: docs/Gamma_transfer_pre-registered_protocol.md (commit 75cad01)
#
# User decision (2026-04-30): defer E9 Phase 2 (multi-seed); run E10 first.
# This chain implements that decision:
#   1. Wait for the E9 Phase-1 matched-baseline summary file to appear.
#   2. Apply the locked E9 decision rule and write
#      notebooks/conservative_arch/scaleup/results/RESULTS.md.
#   3. Launch E10 Stage 1 (γ-grid pilot) via run_stage1.sh.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../../.. && pwd)"
cd "$REPO_ROOT"

CHAIN_LOG=/tmp/gamma_transfer_chain.log

SPLM_SUMMARY="notebooks/conservative_arch/scaleup/results/seed0_splm/splm_em_ln_scaleup_scaleup_seed0_summary.md"
ATTN_SUMMARY="notebooks/conservative_arch/scaleup/results/seed0_attn/matched_baseline_scaleup_scaleup_seed0_summary.md"
ATTN_CKPT="notebooks/conservative_arch/scaleup/results/seed0_attn/matched_baseline_scaleup_scaleup_seed0_ckpt_latest.pt"
RESULTS_OUT="notebooks/conservative_arch/scaleup/results/RESULTS.md"

WRITE_HELPER="notebooks/conservative_arch/scaleup/gamma_transfer/_write_phase1_results.py"
PREDICTOR="notebooks/conservative_arch/scaleup/gamma_transfer/predict_gamma_hessian.py"
PREDICTOR_OUT="notebooks/conservative_arch/scaleup/gamma_transfer/results/predictors"
STAGE1_DRIVER="notebooks/conservative_arch/scaleup/gamma_transfer/run_stage1.sh"

POLL_INTERVAL=300   # seconds; matched-baseline still has ~5 h to go at chain-launch time

echo ""                                                  | tee -a "$CHAIN_LOG"
echo "=== E9 Phase 1 → E10 Stage 1 chain driver ==="    | tee -a "$CHAIN_LOG"
echo "[chain] start   $(date)"                           | tee -a "$CHAIN_LOG"
echo "[chain] repo:   $REPO_ROOT"                        | tee -a "$CHAIN_LOG"
echo "[chain] poll:   ${POLL_INTERVAL}s"                 | tee -a "$CHAIN_LOG"
echo "[chain] log:    $CHAIN_LOG"                        | tee -a "$CHAIN_LOG"
echo ""                                                  | tee -a "$CHAIN_LOG"

# Sanity: SPLM arm should already be present (it ran first).
if [ ! -f "$SPLM_SUMMARY" ]; then
  echo "[chain] FATAL: SPLM Phase-1 summary missing at chain-start: $SPLM_SUMMARY" | tee -a "$CHAIN_LOG"
  echo "[chain]        E9 Phase 1's SPLM arm is the prerequisite for this chain. Aborting." | tee -a "$CHAIN_LOG"
  exit 2
fi
echo "[chain] OK: SPLM Phase-1 summary present." | tee -a "$CHAIN_LOG"

# Wait for the matched-baseline arm to finish writing both ckpt and summary.
echo "[chain] waiting for matched-baseline summary at $ATTN_SUMMARY ..." | tee -a "$CHAIN_LOG"
while [ ! -f "$ATTN_SUMMARY" ] || [ ! -f "$ATTN_CKPT" ]; do
  sleep "$POLL_INTERVAL"
done

# Small safety window to let any in-flight write flush.
sleep 30
echo "[chain] matched-baseline arm complete at $(date)." | tee -a "$CHAIN_LOG"

# Step 2: write Phase-1 RESULTS.md via the python helper.
echo ""                                                          | tee -a "$CHAIN_LOG"
echo "=== Writing E9 Phase 1 RESULTS.md ==="                    | tee -a "$CHAIN_LOG"
PYTHONUNBUFFERED=1 python3 "$WRITE_HELPER" \
  --splm-summary "$SPLM_SUMMARY" \
  --attn-summary "$ATTN_SUMMARY" \
  --out          "$RESULTS_OUT" 2>&1 \
  | tee -a "$CHAIN_LOG"
WRITE_RC="${PIPESTATUS[0]}"
if [ "$WRITE_RC" -ne 0 ]; then
  echo "[chain] FATAL: writing $RESULTS_OUT failed (rc=$WRITE_RC). Not launching E10 Stage 1." | tee -a "$CHAIN_LOG"
  exit "$WRITE_RC"
fi

echo ""                                                          | tee -a "$CHAIN_LOG"
echo "=== Running γ*-prediction diagnostic on E9 SPLM ckpt ===" | tee -a "$CHAIN_LOG"
echo "[chain] $(date) — calling $PREDICTOR"                     | tee -a "$CHAIN_LOG"
PYTHONUNBUFFERED=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 \
  python3 "$PREDICTOR" \
    --out-dir "$PREDICTOR_OUT" \
    --n-batches 4 --batch-size 2 --block-size 256 \
    --n-power-iter 20 --n-hutchinson 5 2>&1 \
  | tee -a "$CHAIN_LOG"
PRED_RC="${PIPESTATUS[0]}"
if [ "$PRED_RC" -ne 0 ]; then
  # Diagnostic failure should NOT block Stage 1; just warn loudly.
  echo "[chain] WARN: γ*-prediction diagnostic failed (rc=$PRED_RC)." | tee -a "$CHAIN_LOG"
  echo "[chain] WARN: continuing to Stage 1; rerun the predictor manually later." | tee -a "$CHAIN_LOG"
else
  echo "[chain] γ*-prediction diagnostic OK; outputs in $PREDICTOR_OUT" | tee -a "$CHAIN_LOG"
fi

echo ""                                                          | tee -a "$CHAIN_LOG"
echo "=== Launching E10 Stage 1 (γ-grid pilot) ==="              | tee -a "$CHAIN_LOG"
echo "[chain] handoff to $STAGE1_DRIVER at $(date)"              | tee -a "$CHAIN_LOG"
echo ""                                                          | tee -a "$CHAIN_LOG"

bash "$STAGE1_DRIVER"
STAGE1_RC=$?

echo "[chain] Stage 1 driver exited at $(date), rc=$STAGE1_RC" | tee -a "$CHAIN_LOG"
exit "$STAGE1_RC"
