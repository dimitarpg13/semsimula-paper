"""
Helper for the E9 Phase-1 → E10 Stage-1 chain driver.

Parses the SPLM and matched-baseline single-seed summary files written by the
E9 scale-up trainers, applies the locked E9 pre-registered decision rule,
and writes a Phase-1 `RESULTS.md` under `notebooks/conservative_arch/scaleup/results/`.

The matched-baseline summary file is the *natural* completion marker for the
E9 Phase-1 driver (it is the very last file written by the matched-baseline
trainer). Calling this helper before that file exists raises FileNotFoundError.

Pre-registered protocols:
  - E9: docs/SPLM_scaleup_pre-registered_protocol.md (commit 17a3795)
  - E10: docs/Gamma_transfer_pre-registered_protocol.md (commit 75cad01)

Phase-2 disposition is determined by §5.2 of the E9 protocol *and* the user's
April 30, 2026 decision (option `b`) to defer E9 Phase 2 in favour of running
E10 first. This is recorded explicitly in the generated RESULTS.md.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PHASE1_PPL_RE = re.compile(
    r"Final val loss:\s*([0-9eE.+-]+)\s*\(ppl\s*([0-9eE.+-]+)\)"
)


def parse_final_ppl(summary_path: Path) -> tuple[float, float]:
    """Return (final_val_loss, final_val_ppl) from a trainer's summary.md."""
    if not summary_path.exists():
        raise FileNotFoundError(f"summary file not found: {summary_path}")
    text = summary_path.read_text(encoding="utf-8")
    m = PHASE1_PPL_RE.search(text)
    if m is None:
        raise ValueError(
            f"Could not find 'Final val loss: <loss> (ppl <ppl>)' line in "
            f"{summary_path}. Contents:\n{text}"
        )
    return float(m.group(1)), float(m.group(2))


def parse_elapsed_hours(summary_path: Path) -> float | None:
    """Return wall-clock hours from the summary, or None if unparseable."""
    text = summary_path.read_text(encoding="utf-8")
    m = re.search(r"elapsed:\s*([0-9.]+)\s*s\s*\(([0-9.]+)\s*h\)", text)
    if m is None:
        return None
    return float(m.group(2))


def apply_e9_decision_rule(delta_ppl: float, defer_phase_2: bool) -> dict:
    """Implement the locked §5 decision rule of the E9 protocol.

    delta_ppl:    P_B - P_A = matched-baseline - SPLM em_ln (positive ⇒ SPLM wins)
    defer_phase_2: if True, the user has elected to defer Phase 2 in favour of
                   the E10 γ-transfer experiment (April 30, 2026 decision).
    """
    abs_delta = abs(delta_ppl)
    if delta_ppl >= 5.0:
        outcome = "A"
        outcome_text = "SPLM em_ln beats matched-attention at scale-up (gap survives)"
    elif delta_ppl <= -5.0:
        outcome = "C"
        outcome_text = (
            "Matched-attention beats SPLM em_ln at scale-up (gap inverts)"
        )
    else:
        outcome = "B"
        outcome_text = (
            "|Δ| < 5 PPL — Phase-1 result is ambiguous (gap shrinks to within "
            "single-seed measurement uncertainty inherited from E1)"
        )

    if abs_delta >= 20.0:
        phase_2_pre_reg = (
            "Phase 2 NOT triggered (|Δ⁽⁰⁾| ≥ 20 PPL — Phase-1 result decisive at S=1)"
        )
        phase_2_executed = (
            "N/A — protocol terminates at S=1 per §5.2 of the E9 pre-registration"
        )
    else:
        phase_2_pre_reg = (
            "Phase 2 triggered per §5.2 of the E9 pre-registration "
            "(|Δ⁽⁰⁾| < 20 PPL is in the ambiguous zone; seeds 1+2 of both arms "
            "are required for a paired band)"
        )
        if defer_phase_2:
            phase_2_executed = (
                "**DEFERRED.** Per the user's April 30, 2026 decision to run the "
                "E10 γ-transfer re-tuning experiment first (`option b`), the E9 "
                "Phase-2 multi-seed runs are deferred until after E10 reports. "
                "Rationale: if E10 finds a materially better γ\\* on TinyStories, "
                "running Phase-2 seeds at the pre-registered γ=0.30 would burn "
                "~26 h of compute on a now-known-suboptimal γ. Re-running Phase 2 "
                "after E10 (at the new γ\\* if the protocol's NT-material outcome "
                "fires; or at γ=0.30 if it does not) is the compute-efficient path. "
                "**This deferral is a reporting choice, not a protocol amendment.** "
                "The original Phase-2 trigger (|Δ⁽⁰⁾| < 20 PPL) is recorded above; "
                "the eventual Phase-2 result will be appended to this RESULTS.md "
                "verbatim under the seed configuration locked at the time of execution."
            )
        else:
            phase_2_executed = "Will be executed (no deferral)."

    return {
        "outcome": outcome,
        "outcome_text": outcome_text,
        "phase_2_pre_reg": phase_2_pre_reg,
        "phase_2_executed": phase_2_executed,
        "abs_delta": abs_delta,
    }


def write_results_md(
    splm_summary: Path,
    attn_summary: Path,
    out_path: Path,
    defer_phase_2: bool = True,
) -> dict:
    p_a_loss, p_a_ppl = parse_final_ppl(splm_summary)
    p_b_loss, p_b_ppl = parse_final_ppl(attn_summary)
    p_a_hours = parse_elapsed_hours(splm_summary)
    p_b_hours = parse_elapsed_hours(attn_summary)
    delta = p_b_ppl - p_a_ppl

    decision = apply_e9_decision_rule(delta, defer_phase_2=defer_phase_2)

    timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z")

    text = f"""# RESULTS — E9 SPLM scale-up de-risking experiment

**Pre-registered protocol:** [`docs/SPLM_scaleup_pre-registered_protocol.md`](../../../../docs/SPLM_scaleup_pre-registered_protocol.md)
**Pre-registration commit:** `17a3795` (April 29, 2026)
**This file generated:** {timestamp}

> This file is auto-generated by
> `notebooks/conservative_arch/scaleup/gamma_transfer/_write_phase1_results.py`,
> invoked from the E9-Phase-1 → E10-Stage-1 chain driver
> `chain_phase1_to_stage1.sh`. It applies the locked decision rule of §5 of
> the E9 pre-registration to the realised single-seed results. **No manual
> editing of the headline rows is permitted** (auxiliary commentary may be
> appended below the auto-generated section).

---

## 1. Phase 1 single-seed results (seed = 0)

| Arm | Variant | Final val loss | Final val PPL | Wall-clock |
|---|---|---:|---:|---:|
| **A — `splm_em_ln`** | SARF mass + LN-after-step, fixed γ = 0.30 | {p_a_loss:.4f} | **{p_a_ppl:.2f}** | {p_a_hours:.2f} h |
| **B — `matched_baseline`** | MatchedGPT (vanilla pre-LN GPT-2 decoder) | {p_b_loss:.4f} | **{p_b_ppl:.2f}** | {p_b_hours:.2f} h |

## 2. Realised paired Δ at seed 0

$$\\Delta^{{(0)}} = P_B - P_A = {p_b_ppl:.2f} - {p_a_ppl:.2f} = \\mathbf{{{delta:+.2f}}}\\ \\text{{PPL}}.$$

(positive ⇒ SPLM wins; negative ⇒ matched-attention wins.)

## 3. Locked decision rule (§5 of the pre-registration)

Effect-size threshold: $\\Delta_{{\\min}} = 5.0$ PPL.

**Outcome:** **{decision["outcome"]}** — {decision["outcome_text"]}.

## 4. Phase-2 disposition

**Pre-registered trigger:** {decision["phase_2_pre_reg"]}.

**This run:** {decision["phase_2_executed"]}

## 5. Auxiliary observations (recorded; not part of the decision rule)

| Quantity | Value |
|---|---:|
| Final train loss, arm A | (see `seed0_splm/splm_em_ln_scaleup_scaleup_seed0_summary.md`) |
| Final train loss, arm B | (see `seed0_attn/matched_baseline_scaleup_scaleup_seed0_summary.md`) |
| γ trajectory, arm A | constant at 0.300 (fixed_gamma) |
| Memory-usage peak | not instrumented in this run |

Loss-curve overlays and per-arm training logs live in:
- `seed0_splm/splm_em_ln_scaleup_scaleup_seed0_loss_curve.png`
- `seed0_splm/splm_em_ln_scaleup_scaleup_seed0_training_log.jsonl`
- `seed0_attn/matched_baseline_scaleup_scaleup_seed0_loss_curve.png`
- `seed0_attn/matched_baseline_scaleup_scaleup_seed0_training_log.jsonl`

---

## 6. Companion E10 γ-transfer experiment

**Pre-registered protocol:** [`docs/Gamma_transfer_pre-registered_protocol.md`](../../../../docs/Gamma_transfer_pre-registered_protocol.md)
**Pre-registration commit:** `75cad01` (April 30, 2026)

The Phase-1 SPLM result above is at the *transferred* small-scale optimum
$\\gamma=0.30$. Whether $\\gamma=0.30$ remains optimal at the E9 TinyStories
configuration is the question of E10. E10 Stage 1 (γ-grid pilot) launches
immediately after this RESULTS.md is written, per the chain driver
`chain_phase1_to_stage1.sh`.
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return {
        "p_a_ppl": p_a_ppl,
        "p_b_ppl": p_b_ppl,
        "delta": delta,
        **decision,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--splm-summary",
        type=Path,
        required=True,
        help="Path to splm_em_ln summary.md (E9 Phase 1 SPLM arm).",
    )
    ap.add_argument(
        "--attn-summary",
        type=Path,
        required=True,
        help="Path to matched_baseline summary.md (E9 Phase 1 attn arm).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Path to write the auto-generated RESULTS.md.",
    )
    ap.add_argument(
        "--no-defer-phase-2",
        action="store_true",
        help="If set, the RESULTS.md does NOT mark Phase 2 as deferred. "
        "Default behaviour (deferral noted) reflects the user's April 30, "
        "2026 decision to run E10 first.",
    )
    args = ap.parse_args()

    info = write_results_md(
        splm_summary=args.splm_summary,
        attn_summary=args.attn_summary,
        out_path=args.out,
        defer_phase_2=not args.no_defer_phase_2,
    )
    print(
        f"[write-phase1] outcome={info['outcome']}  "
        f"P_A={info['p_a_ppl']:.2f}  P_B={info['p_b_ppl']:.2f}  "
        f"Δ={info['delta']:+.2f}  abs_Δ={info['abs_delta']:.2f}",
        flush=True,
    )
    print(f"[write-phase1] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
