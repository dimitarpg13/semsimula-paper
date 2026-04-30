"""Aggregate the per-cell Markov-order regression summaries into the E7
arm-level outcome and produce the headline figures.

Reads:
  results/markov_primary/<arm>__seed<S>/primary_summary.json
  results/markov_primary/<arm>__seed<S>/primary_residuals.npz
for each cell. Produces:
  results/aggregate_summary.json
  results/figures/rho_per_arm.png
  results/figures/per_seed_rho_bar.png
  results/figures/residuals_by_k.png

The per-arm comparison applies the §6 decision rule of the E7 protocol
to each cell, then reports the across-seeds median rho_12, rho_23 and the
paired Wilcoxon p_12, p_23 across cells. Per E7 §6.2 of the protocol,
both arms are predicted to land in Outcome C (first-order not rejected:
rho_12 < 1.10 or p_12 > 0.01).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARMS = {
    "splm1":            ("SPLM-1 (first-order)",   "tab:red"),
    "splm2_gamma0p30":  ("SPLM γ*=0.30 (2nd-order)","tab:blue"),
}

DECISION_THRESHOLDS = {
    "rho_12_strong": 1.20,
    "p_12_strong": 1e-3,
    "rho_12_fail": 1.10,
    "p_12_fail": 0.01,
    "rho_23_keep_2nd": 1.05,
    "p_23_keep_2nd": 0.05,
    "rho_23_keep_3rd": 1.10,
    "p_23_keep_3rd": 0.05,
}


def decision_label(d: dict) -> str:
    rho_12 = d["rho_12"]
    p_12 = d["wilcoxon_p_12"]
    rho_23 = d["rho_23"]
    p_23 = d["wilcoxon_p_23"]
    if (rho_12 < DECISION_THRESHOLDS["rho_12_fail"]) or (p_12 > DECISION_THRESHOLDS["p_12_fail"]):
        return "C"
    rejects_12 = (rho_12 >= DECISION_THRESHOLDS["rho_12_strong"]) and (p_12 < DECISION_THRESHOLDS["p_12_strong"])
    if not rejects_12:
        return "D"
    if rho_23 <= DECISION_THRESHOLDS["rho_23_keep_2nd"] and p_23 > DECISION_THRESHOLDS["p_23_keep_2nd"]:
        return "A"
    if rho_23 > DECISION_THRESHOLDS["rho_23_keep_3rd"] and p_23 < DECISION_THRESHOLDS["p_23_keep_3rd"]:
        return "B"
    return "D"


def load_cells(root: Path) -> dict[str, dict[int, dict]]:
    cells: dict[str, dict[int, dict]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for seed in (0, 1, 2):
            cell = root / f"{arm}__seed{seed}" / "primary_summary.json"
            if not cell.exists():
                print(f"[aggregate] missing: {cell}")
                continue
            with cell.open() as f:
                cells[arm][seed] = json.load(f)
    return cells


def aggregate_arm(arm_cells: dict[int, dict]) -> dict:
    if not arm_cells:
        return {"n_cells": 0}
    keys = ["rho_12", "rho_23", "wilcoxon_p_12", "wilcoxon_p_23",
            "R1_mean", "R2_mean", "R3_mean"]
    out: dict = {"n_cells": len(arm_cells), "per_seed": {}}
    for seed, d in arm_cells.items():
        out["per_seed"][seed] = {k: d.get(k) for k in keys}
        out["per_seed"][seed]["decision"] = decision_label(d)

    for k in keys:
        vals = np.array([d[k] for d in arm_cells.values() if k in d], dtype=float)
        out[f"{k}_median"] = float(np.median(vals))
        out[f"{k}_min"] = float(vals.min())
        out[f"{k}_max"] = float(vals.max())

    decisions = [d["decision"] for d in out["per_seed"].values()]
    out["decisions"] = decisions
    out["unanimous_decision"] = (
        decisions[0] if (len(set(decisions)) == 1) else None
    )
    return out


def build_aggregate_summary(cells: dict[str, dict[int, dict]]) -> dict:
    summary: dict = {"arms": {}}
    for arm, _ in ARMS.items():
        summary["arms"][arm] = aggregate_arm(cells[arm])
    return summary


def make_figures(cells: dict[str, dict[int, dict]], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=False)
    metrics = [("rho_12", r"$\rho_{12} = R_1/R_2$  (k=1 vs k=2)"),
               ("rho_23", r"$\rho_{23} = R_2/R_3$  (k=2 vs k=3)")]
    for ax, (mkey, title) in zip(axes, metrics):
        x = np.arange(3)
        width = 0.36
        for off, (arm, (label, color)) in zip([-1, +1], ARMS.items()):
            ys = [cells[arm].get(s, {}).get(mkey, np.nan) for s in (0, 1, 2)]
            ax.bar(x + off * width / 2, ys, width, color=color,
                   label=label, alpha=0.85)
            for i, v in enumerate(ys):
                if not np.isnan(v):
                    ax.text(x[i] + off * width / 2, v, f"{v:.3f}",
                            ha="center", va="bottom", fontsize=8)
        ax.axhline(1.0, ls="--", color="gray", lw=0.8)
        if mkey == "rho_12":
            ax.axhline(DECISION_THRESHOLDS["rho_12_strong"], ls=":",
                       color="firebrick", alpha=0.6,
                       label=r"reject 1st-order: $\rho_{12} \geq 1.20$")
        ax.set_xticks(x)
        ax.set_xticklabels([f"seed {s}" for s in (0, 1, 2)])
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("E7 Markov-order regression — per-seed effect-size ratios "
                 r"$\rho_{12}, \rho_{23}$ for both arms")
    fig.tight_layout()
    fig.savefig(out_dir / "rho_per_arm.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for arm, (label, color) in ARMS.items():
        Rs = []
        for k in (1, 2, 3):
            vals = [cells[arm].get(s, {}).get(f"R{k}_mean", np.nan)
                    for s in (0, 1, 2)]
            Rs.append([v for v in vals if not np.isnan(v)])
        means = [np.median(r) if len(r) > 0 else np.nan for r in Rs]
        mins = [np.min(r) if len(r) > 0 else np.nan for r in Rs]
        maxs = [np.max(r) if len(r) > 0 else np.nan for r in Rs]
        x = np.array([1, 2, 3])
        ax.errorbar(x, means,
                    yerr=[np.array(means) - np.array(mins),
                          np.array(maxs) - np.array(means)],
                    color=color, marker="o", capsize=4, lw=2, label=label)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["k=1", "k=2", "k=3"])
    ax.set_xlabel("Markov order k")
    ax.set_ylabel("median residual sum-of-squares (PCA-projected)")
    ax.set_title("E7 — residuals R_k by Markov order, per-arm median across seeds\n"
                 "(error bars = min/max across the 3 seeds in that arm)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "residuals_by_k.png", dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results/markov_primary")
    ap.add_argument("--output-summary", default="results/aggregate_summary.json")
    ap.add_argument("--figures-dir", default="results/figures")
    args = ap.parse_args()

    root = Path(args.results_root)
    cells = load_cells(root)

    summary = build_aggregate_summary(cells)
    Path(args.output_summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_summary, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[aggregate] summary written to {args.output_summary}")
    for arm, info in summary["arms"].items():
        if info["n_cells"] == 0:
            print(f"  {arm}: NO CELLS YET")
            continue
        print(f"  {arm}: n_cells={info['n_cells']}  "
              f"rho_12 med={info['rho_12_median']:.3f}   "
              f"rho_23 med={info['rho_23_median']:.3f}   "
              f"decisions={info['decisions']}")

    make_figures(cells, Path(args.figures_dir))
    print(f"[aggregate] figures written to {args.figures_dir}")


if __name__ == "__main__":
    main()
