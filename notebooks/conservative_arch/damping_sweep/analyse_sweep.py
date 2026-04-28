"""E4 damping sweep — aggregate per-cell results, plot, emit RESULTS.md.

Reads the diagnostics_summary.json + per-cell training summaries +
markov_order/decision_table.md from each cell directory, builds
results/sweep_grid.csv and results/sweep_grid_summary.json, draws the
four headline curves into figures/, and writes results/RESULTS.md
following protocol \xa75 / \xa76 of E4_damping_sweep_pre-registered_protocol.md.

Run **after** train + run_diagnostics.py have completed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
FIG_DIR = SCRIPT_DIR / "figures"

DEFAULT_TAGS = (
    "gamma0p00", "gamma0p10", "gamma0p30",
    "gamma0p85", "gamma2p00", "gamma5p00",
)
GAMMA_FROM_TAG = {
    "gamma0p00": 0.00,
    "gamma0p10": 0.10,
    "gamma0p30": 0.30,
    "gamma0p85": 0.85,
    "gamma2p00": 2.00,
    "gamma5p00": 5.00,
}


@dataclass
class CellRow:
    tag: str
    gamma: float
    val_ppl: float
    val_loss: float
    train_ppl: float | None
    drift_slope_per_layer: float
    bandwidth: float
    H0: float
    HL: float
    rho_12: float
    p_12: float
    rho_23: float
    p_23: float
    decision: str
    frac_a_par_negative: float
    mean_ratio_apar_aperp: float
    permutation_z: float


def _read_cell(cell_dir: Path, tag: str) -> CellRow | None:
    if not cell_dir.exists():
        return None
    summary_md = next(
        cell_dir.glob(f"splm_sarfmass_logfreq_shakespeare_{tag}_summary.md"),
        None,
    )
    if summary_md is None:
        return None
    text = summary_md.read_text()
    m_val = re.search(r"Final val loss: ([\d.]+) \(ppl ([\d.]+)\)", text)
    val_loss = float(m_val.group(1)) if m_val else float("nan")
    val_ppl = float(m_val.group(2)) if m_val else float("nan")
    m_tr = re.search(r"Final train loss: ([\d.]+)", text)
    train_loss = float(m_tr.group(1)) if m_tr else None
    train_ppl = math.exp(train_loss) if train_loss is not None else None

    drift_path = cell_dir / "energy_drift_summary.json"
    if drift_path.exists():
        drift = json.loads(drift_path.read_text())
    else:
        drift = {}

    mo_path = cell_dir / "markov_order" / "primary_summary.json"
    if mo_path.exists():
        mo = json.loads(mo_path.read_text())
        rho_12 = float(mo.get("rho_12", float("nan")))
        rho_23 = float(mo.get("rho_23", float("nan")))
        p_12 = float(mo.get("wilcoxon_p_12", float("nan")))
        p_23 = float(mo.get("wilcoxon_p_23", float("nan")))
        decision = str(mo.get("decision", "?"))
    else:
        rho_12 = rho_23 = p_12 = p_23 = float("nan")
        decision = "?"

    accel_path = cell_dir / "acceleration_stats.json"
    if accel_path.exists():
        accel = json.loads(accel_path.read_text())
        frac_an = float(accel.get("frac_a_par_negative", float("nan")))
        ratio = float(accel.get("mean_ratio_apar_aperp", float("nan")))
        perm_z = float(accel.get("permutation_z", float("nan")))
    else:
        frac_an = ratio = perm_z = float("nan")

    return CellRow(
        tag=tag,
        gamma=GAMMA_FROM_TAG.get(tag, float("nan")),
        val_ppl=val_ppl, val_loss=val_loss, train_ppl=train_ppl,
        drift_slope_per_layer=float(
            drift.get("drift_slope_per_layer", float("nan"))
        ),
        bandwidth=float(drift.get("rolling_std_mean_normalised", float("nan"))),
        H0=float(drift.get("H0", float("nan"))),
        HL=float(drift.get("HL", float("nan"))),
        rho_12=rho_12, p_12=p_12, rho_23=rho_23, p_23=p_23, decision=decision,
        frac_a_par_negative=frac_an,
        mean_ratio_apar_aperp=ratio,
        permutation_z=perm_z,
    )


def _save_csv(rows: list[CellRow], path: Path) -> None:
    fieldnames = list(rows[0].__dict__.keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r.__dict__)


def _plot_axis_curve(rows: list[CellRow], y_attr: str, ylabel: str,
                     out_path: Path, log_y: bool = False,
                     title: str | None = None,
                     baseline_value: float | None = None,
                     baseline_label: str | None = None) -> None:
    xs = [r.gamma for r in rows]
    ys = [getattr(r, y_attr) for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(xs, ys, marker="o", linewidth=2, markersize=8)
    if baseline_value is not None:
        ax.axhline(
            baseline_value,
            linestyle="--", color="grey", alpha=0.7,
            label=baseline_label or "natural",
        )
    for r in rows:
        ax.annotate(
            r.tag.replace("gamma", "γ=").replace("p", "."),
            (r.gamma, getattr(r, y_attr)),
            textcoords="offset points", xytext=(5, 5), fontsize=8,
        )
    ax.set_xlabel(r"$\gamma$ (fixed at training)")
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    if baseline_value is not None:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _format_results_md(rows: list[CellRow], out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# RESULTS — E4 SPLM damping sweep\n")
    lines.append("> Pre-registered protocol: "
                 "[`docs/E4_damping_sweep_pre-registered_protocol.md`]"
                 "(../../../docs/E4_damping_sweep_pre-registered_protocol.md)\n")
    lines.append("")
    lines.append("## Headline grid\n")
    lines.append(
        "| tag | $\\gamma$ | val ppl | drift slope / layer | "
        "bandwidth | $\\rho_{12}$ | $p_{12}$ | Markov decision | "
        "$a_\\parallel<0$ | $\\|a_\\parallel\\|/\\|a_\\perp\\|$ | "
        "perm $z$ |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|"
    )
    for r in rows:
        lines.append(
            f"| `{r.tag}` | {r.gamma:.2f} | "
            f"{r.val_ppl:.2f} | "
            f"{r.drift_slope_per_layer:.4g} | "
            f"{r.bandwidth:.4g} | "
            f"{r.rho_12:.4f} | "
            f"{r.p_12:.2e} | "
            f"**{r.decision}** | "
            f"{r.frac_a_par_negative:.3f} | "
            f"{r.mean_ratio_apar_aperp:.3f} | "
            f"{r.permutation_z:.2f} |"
        )
    lines.append("")
    lines.append("## Decision summary (per protocol \xa76)\n")
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.decision] = counts.get(r.decision, 0) + 1
    lines.append("| decision | count |")
    lines.append("|---|---:|")
    for k, v in sorted(counts.items()):
        lines.append(f"| **{k}** | {v} |")
    lines.append("")
    lines.append("## Plots\n")
    lines.append("- ![PPL vs gamma](../figures/ppl_vs_gamma.png)")
    lines.append("- ![Drift slope vs gamma](../figures/drift_slope_vs_gamma.png)")
    lines.append("- ![Markov rho_12 vs gamma](../figures/markov_rho12_vs_gamma.png)")
    lines.append("- ![a_parallel sign rate vs gamma](../figures/apar_negative_vs_gamma.png)")
    lines.append("")
    lines.append("Refer to the protocol's outcome table (\xa76) for the headline reading.")
    out_path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_root", type=Path, default=RESULTS_DIR)
    ap.add_argument("--fig_dir", type=Path, default=FIG_DIR)
    args = ap.parse_args()

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.results_root.mkdir(parents=True, exist_ok=True)

    rows: list[CellRow] = []
    for tag in DEFAULT_TAGS:
        cell = _read_cell(args.results_root / tag, tag)
        if cell is not None:
            rows.append(cell)
    if not rows:
        print("[E4-analyse] no cell results found; nothing to do.")
        return 1
    rows.sort(key=lambda r: r.gamma)

    csv_path = args.results_root / "sweep_grid.csv"
    _save_csv(rows, csv_path)
    print(f"[E4-analyse] wrote {csv_path}")

    summary = {
        "n_cells": len(rows),
        "tags": [r.tag for r in rows],
        "gamma_grid": [r.gamma for r in rows],
        "val_ppl": [r.val_ppl for r in rows],
        "decisions": [r.decision for r in rows],
        "best_gamma": min(rows, key=lambda r: r.val_ppl).tag,
    }
    summary_path = args.results_root / "sweep_grid_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[E4-analyse] wrote {summary_path}")

    _plot_axis_curve(
        rows, "val_ppl", "val perplexity",
        args.fig_dir / "ppl_vs_gamma.png",
        title="SPLM val PPL vs damping coefficient",
    )
    _plot_axis_curve(
        rows, "drift_slope_per_layer", r"drift slope per layer (units of $|H_0|$)",
        args.fig_dir / "drift_slope_vs_gamma.png",
        title="E3 energy-drift slope vs damping",
    )
    _plot_axis_curve(
        rows, "rho_12", r"$\rho_{12} = \bar R_1 / \bar R_2$",
        args.fig_dir / "markov_rho12_vs_gamma.png",
        title=r"Markov-order regression: $\rho_{12}$ vs damping",
        baseline_value=1.0,
        baseline_label=r"$\rho_{12}=1$ (no preference)",
    )
    _plot_axis_curve(
        rows, "frac_a_par_negative",
        r"$\Pr[a_\parallel < 0]$",
        args.fig_dir / "apar_negative_vs_gamma.png",
        title=r"\xa714 trajectory shape: $a_\parallel<0$ rate vs damping",
        baseline_value=0.979,
        baseline_label="natural GPT-2 (97.9 %)",
    )
    print(f"[E4-analyse] wrote 4 figures into {args.fig_dir}")

    rmd = args.results_root / "RESULTS.md"
    _format_results_md(rows, rmd)
    print(f"[E4-analyse] wrote {rmd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
