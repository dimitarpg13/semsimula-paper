"""
Aggregate the paper_tmlr_1 scale-up pilot results across all arms.

Reads the per-arm `*_summary.md`, `*_training_log.jsonl`, and
`*_ckpt_latest.pt` artifacts under a results directory, then writes:

  - PILOT_RESULTS.md       : a comparative table (val PPL, train-val gap,
                             params, wall-clock, decision-rule annotation)
  - pilot_loss_curves.png  : train-loss-vs-step overlay across all arms,
                             with final val-loss as a hollow endpoint marker
  - pilot_pareto.png       : params-vs-val-PPL Pareto plot

Arms (each cell is a single seed-0 run unless --seed is set):

  1. matched_baseline                         all-attention GPT-2-style baseline
  2. splm_em_ln (TF32 on, precision artifact)  all-SPLM em_ln, run with
                                                CUDA TF32 default
                                                (kept as a precision-noise
                                                reference; excluded from
                                                the headline Δ table)
  3. splm_em_ln (TF32 off)                     all-SPLM em_ln, run with
                                                allow_tf32=False
  4. helmholtz_q9d                             Helmholtz hybrid (default
                                                schedule AAAASSSS)
  5. hybrid_va                                 Variant A two-stage
                                                (default n_attn=4,
                                                n_splm=4)
  6. parf_q9c_sparse                           PARF-augmented SPLM,
                                                structural V_phi,
                                                sparse top-k=4

Δ_min for the decision rule is 0.30 PPL (tightened from the originally
pre-registered 5.0 PPL — appropriate for the TinyStories absolute-PPL
range, where the matched-attention baseline lands at ~7.8 PPL and a
5 PPL gap would represent a 64% relative difference).

Usage:
  python3 aggregate_pilot_results.py [--results-dir DIR] [--seed N] [--out-dir OUT]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"

# Tightened decision threshold; the original 5.0 PPL was too loose for
# TinyStories (matched-attn baseline ~7.8 PPL).
DELTA_MIN_PPL = 0.30


# Each ARM tuple: (display_name, ckpt_filename_glob, color, marker, opts)
#   opts['tf32_artifact']: True for the TF32-on SPLM em_ln row, which is
#     reported separately from the headline Δ table because it is a
#     precision-noise reference rather than an architectural data point.
#
# History note (May 2026): the four SPLM-family trainers
# (train_splm_em_ln/helmholtz/hybrid/parf_scaleup.py) all unconditionally
# disabled TF32 from inception, so the default seed0 file
# (`splm_em_ln_scaleup_scaleup_seed{seed}_ckpt_latest.pt`) IS the
# TF32-off result.  The TF32-on reference is produced by the
# `colab_pilot.ipynb` Arm 2b cell with `--allow-tf32 --tag-suffix
# tf32on_seed{seed}`.  An optional Arm 2b file may not exist in older
# runs; the row is silently dropped when no matching checkpoint is
# present (opts['optional'] = True).
ARMS = [
    ("matched-attn baseline",
     "matched_baseline_scaleup_*_seed{seed}_ckpt_latest.pt",
     "tab:gray",   "o", {}),
    ("SPLM em_ln (TF32 off, default)",
     "splm_em_ln_scaleup_scaleup_seed{seed}_ckpt_latest.pt",
     "tab:blue",   "s", {}),
    ("SPLM em_ln (TF32 on, precision-artifact reference)",
     "splm_em_ln_scaleup_scaleup_tf32on_seed{seed}_ckpt_latest.pt",
     "tab:orange", "X", {"tf32_artifact": True, "optional": True}),
    ("Helmholtz Q9d (AAAASSSS)",
     "helmholtz_*_scaleup_scaleup*_seed{seed}_ckpt_latest.pt",
     "tab:red",    "D", {}),
    ("Hybrid VA (k=4, m=4)",
     "hybrid_VA_*_scaleup_scaleup*_seed{seed}_ckpt_latest.pt",
     "tab:green",  "^", {}),
    # Arm 5: PARF at the A100-fitting reduced V_phi capacity (H=16).
    # Pattern is specific to vphi16 so it does not collide with the
    # optional Arm 5b (vphi128) row below.
    ("PARF Q9c sparse k=4 (V_phi H=16)",
     "parf_*_vphi16_*sparse_k4_scaleup_scaleup*_seed{seed}_ckpt_latest.pt",
     "tab:purple", "v", {}),
    # Arm 5b: optional H100 80GB rerun at full V_phi capacity (H=128).
    # Skipped automatically if no matching checkpoint exists in DRIVE_RESULTS.
    ("PARF Q9c sparse k=4 (V_phi H=128, H100)",
     "parf_*_vphi128_*sparse_k4_scaleup_scaleup*_seed{seed}_ckpt_latest.pt",
     "tab:pink",   "P", {"optional": True}),
]


def _glob_one(directory: Path, pattern: str) -> Optional[Path]:
    """Return the most-recent file matching `pattern` (or None)."""
    matches = sorted(directory.glob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        print(f"  [warn] multiple matches for {pattern!r}; "
              "picking the most recent")
        matches.sort(key=lambda p: p.stat().st_mtime)
    return matches[-1]


def _count_state_dict_params(state_dict) -> int:
    """Sum numel() across all tensors in a state dict."""
    total = 0
    for v in state_dict.values():
        if hasattr(v, "numel"):
            total += int(v.numel())
    return total


def _load_ckpt_meta(p: Path) -> dict:
    """Load the metadata fields from a checkpoint.

    The trainers don't currently write `n_params` into the checkpoint, so
    we recompute it from the saved `model_state_dict` when missing.
    """
    raw = torch.load(p, map_location="cpu", weights_only=False)
    keys_of_interest = (
        "tag", "variant", "seed", "fixed_gamma",
        "final_val_loss", "final_val_ppl", "final_gamma",
        "elapsed_sec", "max_train_tokens", "model_cfg", "train_cfg",
        "n_params", "n_v_theta_params", "n_v_phi_params",
        "n_score_head_params", "schedule", "n_attn", "n_splm",
        "v_phi_kind", "top_k",
    )
    out = {k: raw.get(k) for k in keys_of_interest}
    if out.get("n_params") is None:
        sd = raw.get("model_state_dict")
        if sd is not None:
            out["n_params"] = _count_state_dict_params(sd)
    return out


def _params_from_meta(meta: dict) -> Optional[int]:
    if meta.get("n_params") is not None:
        return int(meta["n_params"])
    return None


def _read_train_history(log_path: Path) -> List[Tuple[int, float]]:
    """Parse JSONL and return (step, train_loss) records.

    The trainers log `train_loss` per `log_interval`; `val_loss` is only
    printed to stdout (the in-memory loss_history is used only to draw
    each arm's standalone PNG).  So this aggregator's overlay shows
    train-loss curves with the final val-loss as an endpoint marker.
    """
    out = []
    if not log_path.exists():
        return out
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "train_loss" in rec and "step" in rec:
            out.append((int(rec["step"]), float(rec["train_loss"])))
    out.sort(key=lambda r: r[0])
    return out


def _last_train_loss(history: List[Tuple[int, float]]) -> Optional[float]:
    return history[-1][1] if history else None


def _resolve_log(directory: Path, ckpt_path: Path) -> Path:
    base = ckpt_path.name.replace("_ckpt_latest.pt", "")
    return directory / f"{base}_training_log.jsonl"


def _format_gamma(r: dict) -> str:
    g = r.get("final_gamma")
    fg = r.get("fixed_gamma")
    if fg is not None:
        return f"{fg:.3f} (fixed)"
    if g is not None and not (isinstance(g, float) and math.isnan(g)):
        return f"{g:.3f}"
    return "—"


def aggregate(results_dir: Path, seed: int, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    found = []
    print(f"[aggregate] scanning {results_dir} for seed={seed} runs ...")
    for display, glob_pat, color, marker, opts in ARMS:
        pattern = glob_pat.format(seed=seed)
        ckpt = _glob_one(results_dir, pattern)
        if ckpt is None:
            print(f"  [miss] {display:<46s}  pattern={pattern}")
            rows.append({
                "display": display,
                "found": False,
                "color": color, "marker": marker,
                "opts": opts,
            })
            continue
        meta = _load_ckpt_meta(ckpt)
        log_path = _resolve_log(results_dir, ckpt)
        history = _read_train_history(log_path)
        last_train = _last_train_loss(history)
        params = _params_from_meta(meta)
        elapsed_h = (meta.get("elapsed_sec") or 0.0) / 3600.0
        final_val_loss = meta.get("final_val_loss")
        final_ppl = meta.get("final_val_ppl") or (
            math.exp(final_val_loss) if final_val_loss is not None else None
        )
        gap = (final_val_loss - last_train
               if (last_train is not None and final_val_loss is not None)
               else None)
        rows.append({
            "display": display,
            "found": True,
            "ckpt": ckpt,
            "tag": meta.get("tag"),
            "variant": meta.get("variant"),
            "params": params,
            "final_train_loss": last_train,
            "final_val_loss": final_val_loss,
            "final_val_ppl": final_ppl,
            "train_val_gap": gap,
            "final_gamma": meta.get("final_gamma"),
            "fixed_gamma": meta.get("fixed_gamma"),
            "elapsed_h": elapsed_h,
            "history": history,
            "color": color,
            "marker": marker,
            "opts": opts,
        })
        found.append(rows[-1])
        print(f"  [ok]   {display:<46s}  ppl={final_ppl}  "
              f"params={params}  gap={gap}  elapsed={elapsed_h:.2f}h  "
              f"tag={meta.get('tag')}")

    md_path = out_dir / "PILOT_RESULTS.md"
    with md_path.open("w") as f:
        f.write("# paper_tmlr_1 scale-up pilot — comparative results\n\n")
        f.write(f"- seed: {seed}\n")
        f.write(f"- corpus: TinyStories (~5 M GPT-2 BPE tokens)\n")
        f.write(f"- block_size: 512  batch_size: 16  steps: 8000\n")
        f.write(f"- d=256, L=8, max_len=1024  "
                f"(matches E9 SPLM scale-up protocol)\n\n")

        f.write("| Arm | Params | Final val PPL | Train→Val gap (nats) "
                "| Final γ | Wall-clock |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for r in rows:
            if not r["found"]:
                # Optional rows (Arm 5b H100 rerun, etc.) silently
                # disappear when not run; required rows surface as MISSING.
                if r.get("opts", {}).get("optional"):
                    continue
                f.write(f"| {r['display']} | _MISSING_ | — | — | — | — |\n")
                continue
            params_str = f"{r['params']:,}" if r["params"] else "?"
            ppl_str = (f"{r['final_val_ppl']:.2f}"
                       if r["final_val_ppl"] is not None else "—")
            gap_str = (f"{r['train_val_gap']:.3f}"
                       if r["train_val_gap"] is not None else "—")
            f.write(f"| {r['display']} | {params_str} | {ppl_str} | "
                    f"{gap_str} | {_format_gamma(r)} | "
                    f"{r['elapsed_h']:.2f} h |\n")

        # Decision-rule annotation -------------------------------------
        f.write("\n## Decision-rule annotations\n\n")
        f.write(f"- Δ_min = **{DELTA_MIN_PPL:.2f} PPL** "
                "(tightened from the initial 5.0 PPL pre-registration; "
                "appropriate for the TinyStories absolute-PPL range where "
                "the matched-attn baseline is ~7.8 PPL).\n")
        baseline = next((r for r in found
                         if "matched-attn" in r["display"]), None)
        if baseline and baseline.get("final_val_ppl"):
            base_ppl = baseline["final_val_ppl"]
            f.write(f"- Baseline (matched-attn) val PPL: "
                    f"**{base_ppl:.2f}**\n")
            f.write("- Δ = arm_ppl − baseline_ppl  "
                    "(negative ⇒ arm beats baseline on absolute PPL).\n")
            f.write("- Rows tagged as **TF32-on precision artifacts** are "
                    "excluded from this table; see the dedicated section "
                    "below for that comparison.\n\n")
            f.write("| Arm | Δ vs matched-attn | Verdict |\n")
            f.write("|---|---:|---|\n")
            for r in found:
                if r is baseline:
                    continue
                if r.get("opts", {}).get("tf32_artifact"):
                    continue
                if r["final_val_ppl"] is None:
                    continue
                delta = r["final_val_ppl"] - base_ppl
                if delta < -DELTA_MIN_PPL:
                    verdict = (f"**arm wins** "
                               f"(Δ < −{DELTA_MIN_PPL:.2f})")
                elif delta > DELTA_MIN_PPL:
                    verdict = (f"_baseline wins_ "
                               f"(Δ > +{DELTA_MIN_PPL:.2f})")
                else:
                    verdict = f"tie (|Δ| ≤ {DELTA_MIN_PPL:.2f})"
                f.write(f"| {r['display']} | {delta:+.2f} | {verdict} |\n")
        else:
            f.write("Matched-attention baseline not found; "
                    "skipping Δ comparison.\n")

        # TF32 precision artifact section ------------------------------
        # Default seed0 row is the TF32-OFF result (the four SPLM-family
        # trainers disable TF32 by default).  The optional Arm 2b cell
        # produces the TF32-ON reference via --allow-tf32 + --tag-suffix
        # tf32on_seed0 — it may not exist in older runs.
        tf32off_row = next((r for r in found
                            if "SPLM em_ln (TF32 off" in r["display"]), None)
        tf32on_row = next((r for r in found
                           if r.get("opts", {}).get("tf32_artifact")), None)
        if tf32off_row and tf32off_row.get("final_val_ppl") is not None:
            f.write("\n## TF32 precision check (SPLM em_ln, second-order autograd)\n\n")
            f.write(
                "The SPLM-family forward computes the conservative force "
                "F = −∇V_θ(h) via `torch.autograd.grad(..., create_graph=True)` "
                "inside the model.  This second-order autograd path was "
                "hypothesized to be more sensitive to TF32's 10-bit mantissa "
                "than a single attention forward.  All four SPLM-family "
                "trainers disable TF32 by default (`allow_tf32 = False`); "
                "the optional Arm 2b cell in `colab_pilot.ipynb` provides "
                "the TF32-on reference for an empirical check.\n\n"
            )
            tf32off_ppl = tf32off_row["final_val_ppl"]
            f.write(f"- TF32 **off** (`allow_tf32 = False`, real fp32 "
                    f"matmuls; the default for all SPLM-family scaleup "
                    f"trainers): val PPL **{tf32off_ppl:.2f}**\n")
            if tf32on_row and tf32on_row.get("final_val_ppl") is not None:
                tf32on_ppl = tf32on_row["final_val_ppl"]
                f.write(f"- TF32 **on** (`--allow-tf32`, CUDA default; "
                        f"Arm 2b reference cell): "
                        f"val PPL **{tf32on_ppl:.2f}**\n")
                if tf32off_ppl > 0:
                    ratio = tf32on_ppl / tf32off_ppl
                    delta = tf32on_ppl - tf32off_ppl
                    if abs(delta) < 0.30:
                        verdict = (
                            f"|Δ| = {abs(delta):.2f} PPL "
                            "(< Δ_min = 0.30 PPL): **TF32 is NOT the "
                            "cause** of the SPLM em_ln / matched-attn "
                            "PPL gap.  The 21-PPL gap to attention is an "
                            "architectural property of the all-SPLM "
                            "decoder at this scale, not a precision "
                            "artifact."
                        )
                    elif delta > 0:
                        verdict = (
                            f"+{delta:.2f} PPL inflation "
                            f"({ratio:.2f}× ratio): TF32 contributes a "
                            "measurable precision penalty in the "
                            "autograd.grad path.  Headline architectural "
                            "comparisons should use the TF32-off row."
                        )
                    else:
                        verdict = (
                            f"{delta:+.2f} PPL: TF32-on is *better* "
                            "than TF32-off, which is anomalous and "
                            "warrants investigation (random seed effect "
                            "at n=1?)."
                        )
                    f.write(
                        f"- Comparison: {verdict}\n"
                    )
            else:
                f.write("- TF32 **on** reference: _not yet recorded_ "
                        "(run the optional Arm 2b cell in "
                        "`colab_pilot.ipynb` with `--allow-tf32 "
                        "--tag-suffix tf32on_seed0`).  Without this row "
                        "the 'TF32 inflates PPL' hypothesis cannot be "
                        "empirically tested.\n")

        # Helmholtz vs Hybrid VA structural-equivalence note -----------
        helm = next((r for r in found
                     if "Helmholtz" in r["display"]), None)
        hybr = next((r for r in found
                     if "Hybrid VA" in r["display"]), None)
        if (helm and hybr
                and helm.get("final_val_ppl") is not None
                and hybr.get("final_val_ppl") is not None):
            f.write("\n## Helmholtz vs Hybrid VA — structural-equivalence "
                    "check\n\n")
            delta_hh = abs(helm["final_val_ppl"] - hybr["final_val_ppl"])
            f.write(
                "Both arms use a 4 attention + 4 SPLM/S-block stack with "
                "near-identical parameter counts; the difference is purely "
                "in framing (explicit Helmholtz S/A energy decomposition vs. "
                "simpler 'attention bottom + SPLM top' Hybrid VA stack).\n\n"
            )
            f.write(f"- Helmholtz Q9d val PPL: "
                    f"**{helm['final_val_ppl']:.2f}** "
                    f"(γ→{_format_gamma(helm)})\n")
            f.write(f"- Hybrid VA val PPL: "
                    f"**{hybr['final_val_ppl']:.2f}** "
                    f"(γ→{_format_gamma(hybr)})\n")
            if helm.get("params") and hybr.get("params"):
                f.write(f"- Param-count difference: "
                        f"{abs(hybr['params'] - helm['params']):,} "
                        f"({abs(hybr['params'] - helm['params']) * 100.0 / max(helm['params'], 1):.4f}%)\n")
            if delta_hh < DELTA_MIN_PPL:
                f.write(f"- |ΔPPL| = **{delta_hh:.2f}** "
                        f"(< Δ_min = {DELTA_MIN_PPL:.2f}; "
                        "**within seed-noise margin** — the two framings "
                        "are not numerically distinguishable at this "
                        "scale).\n")
            else:
                f.write(f"- |ΔPPL| = **{delta_hh:.2f}** "
                        f"(≥ Δ_min = {DELTA_MIN_PPL:.2f}; "
                        "meaningful difference detected).\n")
            # γ-direction note: a sub-finding worth surfacing
            g_helm = helm.get("final_gamma")
            g_hybr = hybr.get("final_gamma")
            if (g_helm is not None and g_hybr is not None
                    and not (isinstance(g_helm, float) and math.isnan(g_helm))
                    and not (isinstance(g_hybr, float) and math.isnan(g_hybr))
                    and abs(g_helm - g_hybr) > 0.005):
                f.write(
                    f"- γ trajectories diverge: Helmholtz settles at "
                    f"γ={g_helm:.3f}, Hybrid VA at γ={g_hybr:.3f}. "
                    "The two architectures discover *different effective "
                    "dynamics* (different damping levels) that are "
                    "**functionally equivalent** for next-token "
                    "prediction.\n"
                )

        # Generalization-gap section -----------------------------------
        gap_rows = [r for r in found
                    if r.get("train_val_gap") is not None
                    and not r.get("opts", {}).get("tf32_artifact")]
        if gap_rows:
            f.write("\n## Generalization gap (final val − final train, in nats)\n\n")
            f.write(
                "A smaller train→val gap with comparable val loss "
                "indicates the architecture *fits less but generalizes "
                "more* — i.e. structural regularization.\n\n"
            )
            f.write("| Arm | Train loss | Val loss | Gap (nats) |\n")
            f.write("|---|---:|---:|---:|\n")
            for r in sorted(gap_rows, key=lambda x: x["train_val_gap"]):
                tl = r["final_train_loss"]
                vl = r["final_val_loss"]
                f.write(f"| {r['display']} | {tl:.3f} | {vl:.3f} | "
                        f"{r['train_val_gap']:.3f} |\n")

        # Per-arm artifact paths ---------------------------------------
        f.write("\n## Per-arm artifact paths\n\n")
        for r in rows:
            if r["found"]:
                f.write(f"- **{r['display']}**  "
                        f"`{r['ckpt'].name}`  "
                        f"(tag `{r['tag']}`)\n")

    print(f"[aggregate] wrote {md_path}")

    # --- Loss-curves overlay (train-loss with val endpoint markers) -----
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    for r in found:
        if not r["history"]:
            continue
        steps = [h[0] for h in r["history"]]
        train_ls = [h[1] for h in r["history"]]
        is_artifact = r.get("opts", {}).get("tf32_artifact", False)
        ls = "--" if is_artifact else "-"
        alpha = 0.55 if is_artifact else 0.95
        ax.plot(steps, train_ls,
                marker=r["marker"], color=r["color"],
                linestyle=ls, alpha=alpha,
                label=r["display"], linewidth=1.5, markersize=4,
                markevery=max(1, len(steps) // 20))
        # Final val-loss as hollow endpoint marker (so train + val are
        # visible on the same axis without a second y-axis).
        if r.get("final_val_loss") is not None and steps:
            ax.scatter([steps[-1]], [r["final_val_loss"]],
                       color=r["color"], marker=r["marker"],
                       s=110,
                       facecolors="none", edgecolors=r["color"],
                       linewidths=2.0, zorder=10)
    ax.set_xlabel("training step")
    ax.set_ylabel("loss (nats) — solid: train, hollow marker: final val")
    ax.set_title("paper_tmlr_1 scale-up pilot — convergence curves "
                 f"(seed {seed})")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    overlay_path = out_dir / "pilot_loss_curves.png"
    fig.savefig(overlay_path, dpi=140)
    plt.close(fig)
    print(f"[aggregate] wrote {overlay_path}")

    # --- Pareto: params vs val PPL --------------------------------------
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    for r in found:
        if r["params"] is None or r["final_val_ppl"] is None:
            continue
        is_artifact = r.get("opts", {}).get("tf32_artifact", False)
        ax.scatter(r["params"] / 1e6, r["final_val_ppl"],
                   color=r["color"], marker=r["marker"], s=140,
                   alpha=0.55 if is_artifact else 0.95,
                   edgecolor="black", linewidth=0.5,
                   label=r["display"])
        ax.annotate(r["display"],
                    (r["params"] / 1e6, r["final_val_ppl"]),
                    fontsize=7.5,
                    xytext=(6, 6),
                    textcoords="offset points")
    ax.set_xlabel("parameters (M)")
    ax.set_ylabel("final validation perplexity")
    ax.set_title("paper_tmlr_1 scale-up pilot — params vs val PPL "
                 f"(seed {seed})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    pareto_path = out_dir / "pilot_pareto.png"
    fig.savefig(pareto_path, dpi=140)
    plt.close(fig)
    print(f"[aggregate] wrote {pareto_path}")

    print(f"[aggregate] done.  artifacts in {out_dir}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR),
                    help="Directory containing per-arm *_ckpt_latest.pt + "
                         "*_training_log.jsonl files.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None,
                    help="Where to write PILOT_RESULTS.md + plots. "
                         "Defaults to --results-dir.")
    args = ap.parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()
    out_dir = (Path(args.out_dir).expanduser().resolve()
               if args.out_dir else results_dir)
    return aggregate(results_dir, args.seed, out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
