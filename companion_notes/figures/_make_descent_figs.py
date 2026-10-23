#!/usr/bin/env python3
"""Figures for the overdamped-limit discussion (2026-10-22): paper v6 §17h (the descent paragraph), the future note §7.4.

    python3 _make_descent_figs.py      # writes figures/descent/*.png

  descent_vs_geodesic.png  -- schematic: one two-well landscape; from the same start, a lightly damped second-order
                              trajectory (a damped geodesic of the Jacobi metric) and the gradient-descent line; the
                              two basins of the descent and the separatrix between them. Computed, not measured.
  descent_readout.png      -- recorded: the descent readout of PMX and FO-b-PMX (debug/descent_readout.py), each
                              model's trained step against its own continuous law, per layer
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "descent"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 150})
C2, C1 = "#1b7837", "#d95f02"

# ---------------------------------------------------------------- 1. schematic
wells = [(np.array([-1.3, 0.0]), 1.0, 0.75), (np.array([1.3, 0.25]), 1.25, 0.85)]   # centre, depth, width
def V(x, y):
    v = 0.04 * (x ** 2 + y ** 2)
    for c, a, s in wells:
        v = v - a * np.exp(-((x - c[0]) ** 2 + (y - c[1]) ** 2) / (2 * s ** 2))
    return v
def gradV(p):
    g = 0.08 * p
    for c, a, s in wells:
        d = p - c
        g = g + a * np.exp(-(d @ d) / (2 * s ** 2)) * d / s ** 2
    return g
def descent(p0, mu=0.05, n=4000):
    p = p0.copy(); path = [p.copy()]
    for _ in range(n):
        p = p - mu * gradV(p); path.append(p.copy())
    return np.array(path)
def damped(p0, v0, gamma=0.25, dt=0.02, n=3000):
    p, v = p0.copy(), v0.copy(); path = [p.copy()]
    for _ in range(n):
        v = v - dt * gradV(p) - dt * gamma * v
        p = p + dt * v; path.append(p.copy())
    return np.array(path)

xs, ys = np.linspace(-3.2, 3.2, 260), np.linspace(-2.4, 2.4, 200)
X, Y = np.meshgrid(xs, ys); Z = V(X, Y)
# basins of the descent: run the flow from a coarse grid to convergence, label by the nearer well
bx, by = np.linspace(-3.2, 3.2, 400), np.linspace(-2.4, 2.4, 300)
P = np.stack(np.meshgrid(bx, by), -1).reshape(-1, 2)
for _ in range(800):                                   # the descent, vectorised over the grid
    G = 0.08 * P
    for c, a_, s_ in wells:
        Dd = P - c
        G = G + a_ * np.exp(-(Dd * Dd).sum(-1, keepdims=True) / (2 * s_ ** 2)) * Dd / s_ ** 2
    P = P - 0.2 * G
B = (np.linalg.norm(P - wells[0][0], axis=-1) > np.linalg.norm(P - wells[1][0], axis=-1)).astype(float).reshape(len(by), len(bx))
start = np.array([-0.55, 1.55]); v0 = np.array([1.05, -0.35])
pd_, ps_ = descent(start), damped(start, v0)

fig, ax = plt.subplots(1, 2, figsize=(13, 5.0))
for k, (path, col, title) in enumerate(((ps_, C2, "A. Second order: a damped geodesic"), (pd_, C1, "B. First order: a descent line"))):
    a = ax[k]
    a.contourf(bx, by, B, levels=[-0.5, 0.5, 1.5], colors=["#e8eef8", "#f8ece6"], alpha=0.9)
    a.contour(bx, by, B, levels=[0.5], colors="0.35", linewidths=1.6, linestyles="--")
    cs = a.contour(X, Y, Z, levels=14, colors="0.55", linewidths=0.6)
    a.plot(path[:, 0], path[:, 1], color=col, lw=2.2)
    a.plot(*start, "o", color="k", ms=7); a.annotate("start", start, textcoords="offset points", xytext=(6, 6), fontsize=9)
    for c, _, _ in wells:
        a.plot(*c, "x", color="k", ms=8, mew=2)
    a.plot(*path[-1], "s", color=col, ms=7)
    a.set_title(title); a.set_xlim(-3.2, 3.2); a.set_ylim(-2.4, 2.4); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
ax[0].text(-3.05, -2.2, "with an initial velocity, the token crosses the separatrix\nand settles in the other basin (it passes through)", fontsize=8.5)
ax[1].text(-3.05, -2.2, "the descent crosses every level set at right angles\nand settles in the basin it starts in (capture)", fontsize=8.5)
ax[0].text(0.15, 2.05, "separatrix", fontsize=8.5, color="0.3")
fig.tight_layout(); fig.savefig(OUT / "descent_vs_geodesic.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- 2. the readout (recorded)
RES = Path(__file__).resolve().parents[2] / "notebooks/conservative_arch/scaleup/results"
SFX = "_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn"
PFX = "cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_"
files = {"FO-b-PMX": RES / (PFX + "pm64_pmx16_sr2_fob" + SFX) / "descent_readout_fob-pmx_output.txt",
         "PMX": RES / (PFX + "pm64_pmx16_sr2" + SFX) / "descent_readout_pmx_output.txt"}
def parse(f):
    t = f.read_text(); out = {}
    for li, blk in enumerate(re.split(r"\nLAYER \d", t)[1:3]):
        out[li] = dict(cos=float(re.search(r"cosine\(model step, flow step\)\s+median ([\d.]+)", blk).group(1)),
                       vfall=float(re.search(r"trained step \(midpoint rule\): V falls for ([\d.]+)%", blk).group(1)),
                       rho=float(re.search(r"rank correlation of the approach over all wells: median ([\d.]+)", blk).group(1)),
                       all3=float(re.search(r"all three shared for ([\d.]+)%", blk).group(1)))
    return out
D = {k: parse(f) for k, f in files.items()}
fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
metrics = [("vfall", "potential falls along the trained step (%)", (0, 105)), ("cos", "direction of the step against its continuous law\n(median cosine)", (0.7, 1.005)),
           ("rho", "approach to the 64 wells, rank correlation\nwith the continuous law (median)", (0.7, 1.005))]
w = 0.36
for k, (m, lab, yl) in enumerate(metrics):
    a = ax[k]
    for j, (arm, col) in enumerate((("PMX", C2), ("FO-b-PMX", C1))):
        vals = [D[arm][li][m] for li in (0, 1)]
        bars = a.bar(np.arange(2) + (j - 0.5) * w, vals, w, color=col, label=arm + (" (second order)" if arm == "PMX" else " (first order)"))
        for b_, v in zip(bars, vals):
            a.text(b_.get_x() + b_.get_width() / 2, v + (yl[1] - yl[0]) * 0.01, f"{v:.0f}" if m == "vfall" else f"{v:.3f}", ha="center", fontsize=8)
    a.set_xticks([0, 1], ["layer 0", "layer 1"]); a.set_ylim(*yl); a.set_ylabel(lab, fontsize=9)
ax[0].legend(fontsize=8, loc="lower center")
ax[0].set_title("A. The potential never rises"); ax[1].set_title("B. Layer 0 follows the descent"); ax[2].set_title("C. Layer 1 is a learned map")
fig.tight_layout(); fig.savefig(OUT / "descent_readout.png"); plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png")), D)
