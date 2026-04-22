"""
Capacity sweep for the shared-V_psi fit on GPT-2 trajectories.

Question this answers
---------------------
The step-2 shared-V_psi fit on GPT-2 used a fixed 2-layer MLP with
hidden=256 (~200K parameters).  Median per-layer TEST R^2 was +0.19,
with 5 of 11 middle layers below R^2 = 0.1.  Is this failure
representational (V_psi was too small to parameterise a smooth 768-dim
potential) or structural (no single V_psi whose Hessian field
matches the per-layer operators exists)?

This script runs `shared_potential_fit.py` under an increasing sequence
of V_psi capacities and reports the per-layer R^2 curves.  If the middle
layers' R^2 saturates around +0.1-0.2 as capacity grows, the failure is
structural; if it climbs monotonically toward +1, the failure was
representational.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR  = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"


SWEEP = [
    dict(hidden=128,  depth=2, steps=3000, tag="h128_d2"),
    dict(hidden=256,  depth=2, steps=3000, tag="h256_d2"),   # default (already run)
    dict(hidden=512,  depth=2, steps=3000, tag="h512_d2"),
    dict(hidden=1024, depth=2, steps=3000, tag="h1024_d2"),
    dict(hidden=512,  depth=3, steps=3000, tag="h512_d3"),
    dict(hidden=512,  depth=4, steps=3000, tag="h512_d4"),
]


TRAJ_PATH = RESULTS_DIR / "gpt2_baseline.trajectories.pkl"


def run_one(cfg):
    tag = f"gpt2_baseline_sweep_{cfg['tag']}"
    out_npz = RESULTS_DIR / f"sharedV_{tag}_results.npz"
    if out_npz.exists():
        print(f"[sweep] skip {tag} (already done)")
        return tag

    cmd = [
        sys.executable, "-u", str(SCRIPT_DIR / "shared_potential_fit.py"),
        "--traj",   str(TRAJ_PATH),
        "--tag",    tag,
        "--hidden", str(cfg["hidden"]),
        "--depth",  str(cfg["depth"]),
        "--steps",  str(cfg["steps"]),
        "--device", "cpu",
    ]
    print(f"[sweep] running {tag}: hidden={cfg['hidden']} depth={cfg['depth']} "
          f"steps={cfg['steps']}")
    subprocess.run(cmd, check=True, cwd=str(SCRIPT_DIR),
                   env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"})
    return tag


def param_count(hidden: int, depth: int, d: int = 768) -> int:
    n = d * hidden + hidden
    for _ in range(depth - 1):
        n += hidden * hidden + hidden
    n += hidden * 1 + 1
    return n


def main():
    if not TRAJ_PATH.exists():
        raise FileNotFoundError(f"Need GPT-2 trajectories at {TRAJ_PATH}")

    tags = [run_one(c) for c in SWEEP]

    curves = []
    for cfg, tag in zip(SWEEP, tags):
        z = np.load(RESULTS_DIR / f"sharedV_{tag}_results.npz")
        curves.append({
            "cfg":          cfg,
            "tag":          tag,
            "layers":       z["layers"],
            "r2_null_test": z["r2_null_test"],
            "r2_vo_test":   z["r2_vo_test"],
            "r2_shv_test":  z["r2_shv_test"],
            "r2_shv_train": z["r2_shv_train"],
            "n_params":     param_count(cfg["hidden"], cfg["depth"]),
        })

    # ---------- Sweep table ----------
    print("\n[sweep] per-layer TEST R^2 by (hidden, depth):")
    header = "layer " + "".join(f" | h{c['cfg']['hidden']:>4} d{c['cfg']['depth']}"
                                 for c in curves)
    print(header)
    L = curves[0]["layers"]
    for i, ell in enumerate(L):
        row = f"l={ell:2d}"
        for c in curves:
            row += f" | {c['r2_shv_test'][i]:+.3f}  "
        print(row)
    print("median:" + "".join(f" | {np.median(c['r2_shv_test']):+.3f}  " for c in curves))
    print("min   :" + "".join(f" | {np.min(c['r2_shv_test']):+.3f}  " for c in curves))

    # ---------- Save ----------
    out_npz = RESULTS_DIR / "sharedV_capacity_sweep_results.npz"
    np.savez(
        out_npz,
        hidden=np.array([c["cfg"]["hidden"] for c in curves]),
        depth =np.array([c["cfg"]["depth"]  for c in curves]),
        n_params=np.array([c["n_params"]    for c in curves]),
        layers=curves[0]["layers"],
        r2_shv_test=np.stack([c["r2_shv_test"]  for c in curves], axis=0),
        r2_shv_train=np.stack([c["r2_shv_train"] for c in curves], axis=0),
        r2_vo_test =np.stack([c["r2_vo_test"]   for c in curves], axis=0),
    )
    print(f"\n[sweep] saved -> {out_npz}")

    # ---------- Plot A: per-layer R^2 for each capacity setting ----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for ax, split, y_key in zip(
        axes, ["TRAIN", "TEST"], ["r2_shv_train", "r2_shv_test"],
    ):
        cmap = plt.cm.viridis(np.linspace(0, 0.9, len(curves)))
        for colour, c in zip(cmap, curves):
            label = (f"h{c['cfg']['hidden']:>4} d{c['cfg']['depth']}"
                     f"  ({c['n_params']/1e6:.2f} M)")
            ax.plot(c["layers"], c[y_key], marker="o", color=colour, label=label)
        ax.plot(curves[0]["layers"], curves[0]["r2_vo_test"],
                marker="s", color="tab:orange", linestyle="--", alpha=0.7,
                label="vel-only baseline")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axhline(0.5, color="gray", linewidth=0.3, linestyle=":")
        ax.set_xlabel("layer $\\ell$"); ax.set_ylabel(f"per-layer {split} $R^2$")
        ax.set_ylim(-0.1, 1.05); ax.grid(True, alpha=0.3)
        ax.set_title(f"GPT-2 small -- shared-$V_\\psi$ capacity sweep ({split})")
        ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig_a = RESULTS_DIR / "sharedV_capacity_sweep_per_layer.png"
    fig.savefig(fig_a, dpi=140); plt.close(fig)
    print(f"[sweep] saved -> {fig_a}")

    # ---------- Plot B: median and middle-layer R^2 vs V_psi params ----------
    params = np.array([c["n_params"] for c in curves])
    order = np.argsort(params)
    params = params[order]
    medians = np.array([np.median(curves[i]["r2_shv_test"]) for i in order])
    mins    = np.array([np.min(curves[i]["r2_shv_test"])    for i in order])
    # "Middle layers" = layers 5..10 for GPT-2 (index 4..9).
    mid = np.array([
        np.mean(curves[i]["r2_shv_test"][4:10]) for i in order
    ])

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(params, medians, marker="o", label="median over all layers")
    ax.plot(params, mid,     marker="s", label="mean over layers 5--10 (middle)")
    ax.plot(params, mins,    marker="x", label="min over all layers")
    ax.set_xscale("log"); ax.axhline(0, color="gray", linewidth=0.5)
    ax.axhline(0.5, color="gray", linewidth=0.3, linestyle=":")
    ax.set_xlabel("$V_\\psi$ parameters"); ax.set_ylabel("TEST $R^2$")
    ax.set_title("GPT-2 small -- shared-$V_\\psi$ capacity sweep")
    ax.set_ylim(-0.1, 1.05); ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    fig_b = RESULTS_DIR / "sharedV_capacity_sweep_saturation.png"
    fig.savefig(fig_b, dpi=140); plt.close(fig)
    print(f"[sweep] saved -> {fig_b}")

    # ---------- Markdown ----------
    md = RESULTS_DIR / "sharedV_capacity_sweep_summary.md"
    with md.open("w") as f:
        f.write("# Shared-$V_\\psi$ capacity sweep on GPT-2 trajectories\n\n")
        f.write("**Question.** Is the step-2 middle-layer failure on GPT-2 "
                "(layers 5--10 with per-layer TEST $R^2 \\le 0.2$) a "
                "representational limit of a 2-layer hidden-256 MLP, or a "
                "structural fact about GPT-2's per-layer operators?\n\n")
        f.write("**Method.** Re-run the step-2 shared-$V_\\psi$ fit with "
                "a sequence of MLP capacities while holding all other "
                "hyperparameters fixed.  3000 AdamW steps per run.\n\n")
        f.write("## Configuration sweep\n\n")
        f.write("| hidden | depth | params | median TEST $R^2$ | mean TEST $R^2$ (layers 5--10) | min TEST $R^2$ |\n")
        f.write("|--:|--:|--:|--:|--:|--:|\n")
        for c in curves:
            f.write(f"| {c['cfg']['hidden']} | {c['cfg']['depth']} | "
                    f"{c['n_params']/1e6:.2f} M | "
                    f"{np.median(c['r2_shv_test']):+.3f} | "
                    f"{np.mean(c['r2_shv_test'][4:10]):+.3f} | "
                    f"{np.min(c['r2_shv_test']):+.3f} |\n")
        f.write("\n## Per-layer TEST $R^2$\n\n")
        head = "| layer | " + " | ".join(
            f"h{c['cfg']['hidden']} d{c['cfg']['depth']}" for c in curves) + " |\n"
        sep  = "|--:" * (1 + len(curves)) + "|\n"
        f.write(head); f.write(sep)
        for i, ell in enumerate(curves[0]["layers"]):
            row = f"| {ell} | " + " | ".join(
                f"{c['r2_shv_test'][i]:+.3f}" for c in curves) + " |\n"
            f.write(row)
        f.write("\n## Plots\n\n")
        f.write("![per-layer](sharedV_capacity_sweep_per_layer.png)\n\n")
        f.write("![saturation](sharedV_capacity_sweep_saturation.png)\n\n")
    print(f"[sweep] saved -> {md}")


if __name__ == "__main__":
    main()
