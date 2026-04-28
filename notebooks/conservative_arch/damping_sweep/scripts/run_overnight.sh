#!/usr/bin/env bash
# E4 damping sweep — overnight meta-runner.
#
# Chains:
#   Phase 1 — train all six fixed-gamma cells (run_sweep.sh, ~4 hr)
#   Phase 2 — per-cell diagnostics (run_diagnostics.py, ~30 min)
#   Phase 3 — aggregate plots + RESULTS.md (analyse_sweep.py, <1 min)
#
# Each phase is run unconditionally (the inner scripts skip / report
# missing cells gracefully); a final STATUS.md is written that lists
# which cells made it through every phase, so the morning check is
# a single file read.
#
# Logs live at:
#   results/overnight_run.log              — combined stdout/stderr
#   results/<tag>/train_stdout.log         — per-cell training log
#   results/STATUS.md                      — overnight status report

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${SWEEP_DIR}/results"
LOG="${RESULTS_DIR}/overnight_run.log"

mkdir -p "${RESULTS_DIR}"
T0=$(date +%s)

{
    echo "============================================================"
    echo "[E4] overnight run started $(date '+%Y-%m-%d %H:%M:%S')"
    echo "[E4] sweep dir: ${SWEEP_DIR}"
    echo "============================================================"

    echo
    echo "### Phase 1 / 3 — train six cells"
    bash "${SCRIPT_DIR}/run_sweep.sh"
    P1_RC=$?
    P1_DT=$(( $(date +%s) - T0 ))
    echo "[phase1] done in ${P1_DT}s rc=${P1_RC}"

    echo
    echo "### Phase 2 / 3 — per-cell diagnostics"
    P2_T0=$(date +%s)
    python3 -u "${SWEEP_DIR}/run_diagnostics.py" || true
    P2_RC=$?
    P2_DT=$(( $(date +%s) - P2_T0 ))
    echo "[phase2] done in ${P2_DT}s rc=${P2_RC}"

    echo
    echo "### Phase 3 / 3 — aggregate + plots + RESULTS.md"
    P3_T0=$(date +%s)
    python3 -u "${SWEEP_DIR}/analyse_sweep.py" || true
    P3_RC=$?
    P3_DT=$(( $(date +%s) - P3_T0 ))
    echo "[phase3] done in ${P3_DT}s rc=${P3_RC}"

    TOTAL_DT=$(( $(date +%s) - T0 ))
    echo
    echo "============================================================"
    echo "[E4] overnight run done in ${TOTAL_DT}s ($(printf '%dh%02dm' $((TOTAL_DT/3600)) $(( (TOTAL_DT%3600)/60 ))))"
    echo "[E4] phase1 rc=${P1_RC}  phase2 rc=${P2_RC}  phase3 rc=${P3_RC}"
    echo "[E4] finished $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    STATUS_MD="${RESULTS_DIR}/STATUS.md"
    {
        echo "# E4 overnight run — status"
        echo
        echo "- Started:  $(date -r ${T0} '+%Y-%m-%d %H:%M:%S')"
        echo "- Finished: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "- Wall clock: ${TOTAL_DT}s ($(printf '%dh%02dm' $((TOTAL_DT/3600)) $(( (TOTAL_DT%3600)/60 ))))"
        echo "- Phase 1 (train) rc=${P1_RC}  duration=${P1_DT}s"
        echo "- Phase 2 (diag)  rc=${P2_RC}  duration=${P2_DT}s"
        echo "- Phase 3 (plots) rc=${P3_RC}  duration=${P3_DT}s"
        echo
        echo "## Per-cell status"
        echo
        echo "| tag | trained | energy states | quadruples | markov decision | ppl |"
        echo "|---|:-:|:-:|:-:|:-:|---:|"
        for TAG in gamma0p00 gamma0p10 gamma0p30 gamma0p85 gamma2p00 gamma5p00; do
            CD="${RESULTS_DIR}/${TAG}"
            CKPT="${CD}/splm_sarfmass_logfreq_shakespeare_${TAG}_ckpt_latest.pt"
            ENERGY="${CD}/energy_states.npz"
            QUADS="${CD}/markov_order/quadruples.npz"
            DECISION_MD="${CD}/markov_order/decision_table.md"
            SUMMARY_MD="${CD}/splm_sarfmass_logfreq_shakespeare_${TAG}_summary.md"

            CKPT_OK=$([[ -f "${CKPT}" ]] && echo "OK" || echo "—")
            ENERGY_OK=$([[ -f "${ENERGY}" ]] && echo "OK" || echo "—")
            QUADS_OK=$([[ -f "${QUADS}" ]] && echo "OK" || echo "—")
            if [[ -f "${CD}/markov_order/primary_summary.json" ]]; then
                DEC=$(python3 -c "import json; print(json.load(open('${CD}/markov_order/primary_summary.json'))['decision'])" 2>/dev/null || echo "?")
            else
                DEC="—"
            fi
            if [[ -f "${SUMMARY_MD}" ]]; then
                PPL=$(grep -oE 'ppl [0-9.]+\)' "${SUMMARY_MD}" | tail -1 | tr -d 'ppl )' || echo "?")
            else
                PPL="—"
            fi
            echo "| \`${TAG}\` | ${CKPT_OK} | ${ENERGY_OK} | ${QUADS_OK} | **${DEC}** | ${PPL} |"
        done
        echo
        echo "## Headline files"
        echo
        echo "- \`results/sweep_grid.csv\` — one row per cell, all metrics"
        echo "- \`results/sweep_grid_summary.json\` — JSON summary"
        echo "- \`results/RESULTS.md\` — narrative report (read this first)"
        echo "- \`figures/ppl_vs_gamma.png\`, \`figures/drift_slope_vs_gamma.png\`, \`figures/markov_rho12_vs_gamma.png\`, \`figures/apar_negative_vs_gamma.png\` — headline curves"
        echo "- \`results/overnight_run.log\` — combined log (this run)"
    } > "${STATUS_MD}"
    echo "[E4] wrote ${STATUS_MD}"
} 2>&1 | tee "${LOG}"
