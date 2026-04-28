"""Diagnostic plots for the first-order ODE rejection experiment.

Produces (per architecture, primary cell only):
  - R_k bar plot with cluster-bootstrap CIs.
  - Paired-residual scatter (r_1^{(i)} vs r_2^{(i)}, vs r_3^{(i)}).
  - LOSO-fold spaghetti of mean R_k per fold.

If `--robustness_csv` is given, additionally produces:
  - 4x3 (class x PCA-dim) heatmap of rho_12 across the robustness grid,
    one per architecture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_residuals(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    sids = data["sentence_idx"]
    out = {"sentence_idx": sids}
    for key in data.files:
        if key.startswith("residuals_k"):
            k = int(key.replace("residuals_k", ""))
            out[k] = data[key]
    return out


def plot_Rk_bars(residuals: dict, out_path: Path, title: str = ""):
    ks = sorted(k for k in residuals.keys() if isinstance(k, int))
    means = [float(residuals[k].mean()) for k in ks]
    sems = [float(residuals[k].std(ddof=1) / np.sqrt(len(residuals[k]))) for k in ks]
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    bars = ax.bar([str(k) for k in ks], means, yerr=sems, capsize=6,
                  color=["#4c72b0", "#55a868", "#c44e52"])
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m, f"{m:.3g}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Markov order $k$ used in $\\hat F_k$")
    ax.set_ylabel("$\\bar R_k$  (mean squared residual)")
    if title:
        ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_paired_scatter(residuals: dict, out_path: Path, title: str = ""):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4))
    pairs = [(1, 2), (2, 3)]
    for ax, (a, b) in zip(axes, pairs):
        ra = residuals[a]
        rb = residuals[b]
        ax.scatter(ra, rb, s=4, alpha=0.4, color="#4c72b0")
        lo = min(ra.min(), rb.min())
        hi = max(ra.max(), rb.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label="$y=x$")
        rho = float(ra.mean() / rb.mean())
        ax.set_xlabel(f"$r_{{{a}}}^{{(i)}}$")
        ax.set_ylabel(f"$r_{{{b}}}^{{(i)}}$")
        ax.set_title(f"$\\rho_{{{a}{b}}} = {rho:.3f}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(loc="lower right", fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_loso_spaghetti(residuals: dict, out_path: Path, title: str = ""):
    sids = residuals["sentence_idx"]
    ks = sorted(k for k in residuals.keys() if isinstance(k, int))
    sentence_ids = sorted(set(sids.tolist()))
    per_fold = {k: [] for k in ks}
    for s in sentence_ids:
        mask = sids == s
        for k in ks:
            per_fold[k].append(float(residuals[k][mask].mean()))

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.array(sentence_ids)
    colors = {1: "#4c72b0", 2: "#55a868", 3: "#c44e52"}
    for k in ks:
        ax.plot(x, per_fold[k], "o-", color=colors[k], lw=1.0,
                ms=4, alpha=0.9, label=f"$k={k}$")
    ax.set_xlabel("Held-out sentence index")
    ax.set_ylabel("Per-fold mean squared residual")
    ax.set_yscale("log")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_robustness_heatmap(csv_path: Path, out_dir: Path):
    import csv
    rows = []
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            rows.append(row)
    archs = sorted(set(r["arch"] for r in rows))
    classes = ["kernel", "linear", "poly2", "mlp"]
    pcas = sorted({int(r["p"]) for r in rows})

    for arch in archs:
        rho12 = np.full((len(classes), len(pcas)), np.nan)
        for r in rows:
            if r["arch"] != arch:
                continue
            i = classes.index(r["function_class"])
            j = pcas.index(int(r["p"]))
            rho12[i, j] = float(r["rho_12"])
        fig, ax = plt.subplots(figsize=(4.5, 3))
        im = ax.imshow(rho12, cmap="RdBu_r", vmin=0.85, vmax=1.15, aspect="auto")
        ax.set_xticks(range(len(pcas))); ax.set_xticklabels([f"p={p}" for p in pcas])
        ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
        for i in range(len(classes)):
            for j in range(len(pcas)):
                v = rho12[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            color="black" if 0.95 < v < 1.05 else "white",
                            fontsize=9)
        ax.set_title(f"$\\rho_{{12}} = R_1/R_2$  ({arch})")
        cb = fig.colorbar(im, ax=ax, shrink=0.85)
        cb.ax.axhline(1.20, color="k", lw=1, ls="--")
        fig.tight_layout()
        fig.savefig(out_dir / f"robustness_rho12_{arch}.png", dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--residuals_npz", required=True,
                    help="results/<arch>/primary_residuals.npz")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--title_prefix", default="")
    ap.add_argument("--robustness_csv", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = _load_residuals(Path(args.residuals_npz))

    plot_Rk_bars(res, out_dir / "Rk_bars.png",
                 title=f"{args.title_prefix} primary, kernel ridge, p=50")
    plot_paired_scatter(res, out_dir / "paired_scatter.png",
                        title=f"{args.title_prefix} per-quadruple paired residuals")
    plot_loso_spaghetti(res, out_dir / "loso_spaghetti.png",
                        title=f"{args.title_prefix} per-fold mean R_k")

    if args.robustness_csv:
        plot_robustness_heatmap(Path(args.robustness_csv), out_dir)

    print(f"[plots] wrote figures to {out_dir}/")


if __name__ == "__main__":
    main()
