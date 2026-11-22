"""Figures for the joint_vtheta_arm/ notes (the joint-coupling V_theta + QK-norm d=384 arm, September 2026).

Generates into figures/joint_vtheta_arm/:

  jva_run_story.png         -- README: windowed validation PPL of the main run, its best and its stop, and what the
                               two anneal probes recovered from each branch point.
  jva_lambda_bounds.png     -- Bank geometry: the largest eigenvalue of a weighted sum of two rank-one factors against
                               the angle between them, for three weight splits, with the two bounds of T5.
  jva_predictions.png       -- Bank geometry: the quantities predicted for step 50,000, actual over predicted.
  jva_lognormal_shift.png   -- Bank geometry: a rigid shift against a widening of the omega*dt distribution, and the
                               share each puts over the wall at 2.
  jva_anneals.png           -- Anneal probes: the compressed cosine schedule, and both anneals' evaluations.
  jva_branch_health.png     -- Anneal probes: the two branch points' health readings during their anneals.
  jva_leapfrog_error.png    -- Integration refinement: leapfrog frequency error against omega*dt.
  jva_dt_wall.png           -- Integration refinement: the omega*dt distribution at four step sizes against the wall.
  jva_lowrank_vs_control.png -- Integration refinement: the exact low-rank integrator against its control.
  jva_linalg_bench.png      -- Integration refinement: time per call of the low-rank mode extraction's operations.
  jva_gpt2_budget.png       -- Matched GPT-2: tokens against steps for both runs, and where GPT-2 crossed the Fock
                               endpoint.
  jva_four_axes.png         -- Matched GPT-2: GPT-2's advantage on each axis.
  jva_two_slots.png         -- Content routing: where Family A and Alternative E act in the layer step.
  jva_alt_e.png             -- Content routing: Alternative E against its control, and the gap against the target.

Values are those recorded in the run notes (each array names its section); the anneals' full evaluation sequences
come from their run outputs (the 28,500 anneal's result log and the 50,000 anneal's notebook outputs).
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from math import erf, sqrt, log, exp, pi, sin, asin

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
})

GREEN = "#22C55E"
BLUE = "#3B82F6"
RED = "#EF4444"
PURPLE = "#8B5CF6"
GREY = "#9CA3AF"
ORANGE = "#F59E0B"
INK = "#1F2937"
MUTED = "#6B7280"

HERE = Path(__file__).resolve().parent
OUT = HERE / "joint_vtheta_arm"


def _spines(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def Phi(z):
    return 0.5 * (1 + erf(z / sqrt(2)))


def lognormal_pdf(x, med, sig):
    mu = log(med)
    return np.exp(-(np.log(x) - mu) ** 2 / (2 * sig ** 2)) / (x * sig * sqrt(2 * pi))


def over_wall(med, sig, wall=2.0):
    return 1 - Phi((log(wall) - log(med)) / sig)


# ---------------------------------------------------------------- README
def fig_run_story():
    fig, ax = plt.subplots(figsize=(11, 4.8))
    w_centres = np.array([22.5, 27.5, 32.5, 37.5, 42.5, 47.5]) * 1000          # 5,000-step windows from 20,000
    w_ppl = [89.76, 88.45, 87.88, 89.15, 88.92, 90.25]
    ax.plot(w_centres, w_ppl, "-o", color=BLUE, lw=2, label="main run, windowed mean validation PPL")
    ax.scatter([28500], [84.31], color=BLUE, marker="*", s=180, zorder=4, label="best evaluation, 84.31 at step 28,500")
    ax.scatter([50000], [93.93], color=BLUE, marker="X", s=90, zorder=4, label="stop at step 50,000, 93.93")
    for x0, y0, y1, col, txt, dy in ((50000, 93.93, 85.03, ORANGE, "4,000-step anneal from 50,000:\nsettled 85.03", 0),
                                     (28500, 84.31, 81.58, GREEN, "4,000-step anneal from 28,500:\nsettled 81.58", -1.3)):
        ax.add_patch(FancyArrowPatch((x0, y0), (x0 + 4000, y1), arrowstyle="-|>", mutation_scale=14, color=col, lw=1.8))
        ax.text(x0 + 4300, y1 + dy, txt, color=col, fontsize=9.5, va="center")
    ax.axhline(81.92, color=MUTED, ls=":", lw=1.3)
    ax.text(60800, 82.1, "additive arm's best after 100,000 steps: 81.92", color=MUTED, fontsize=9, va="bottom", ha="right")
    ax.set_xlim(19000, 61000); ax.set_ylim(79, 96)
    ax.set_xlabel("training step"); ax.set_ylabel("validation perplexity")
    ax.set_title("The joint arm: past its peak at step 28,500, recovered by a decay")
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    _spines(ax)
    fig.tight_layout(); fig.savefig(OUT / "jva_run_story.png", bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- bank geometry
def fig_lambda_bounds():
    th = np.linspace(0, pi / 2, 200)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for p, col in ((0.5, BLUE), (0.8, PURPLE), (0.95, ORANGE)):
        lam = 0.5 * (1 + np.sqrt(1 - 4 * p * (1 - p) * np.sin(th) ** 2))
        ax.plot(np.degrees(th), lam, color=col, lw=2, label=f"weights {p:.2f} / {1 - p:.2f}")
        ax.axhline(p, color=col, ls=":", lw=1)
    ax.axhline(1.0, color=INK, ls="--", lw=1.2)
    ax.text(88, 1.01, "upper bound: sum of the weighted norms", ha="right", va="bottom", fontsize=9, color=INK)
    ax.text(88, 0.505, "lower bounds (dotted): the largest single term", ha="right", va="bottom", fontsize=9, color=MUTED)
    ax.set_xlabel("angle between the two wells' directions (degrees)")
    ax.set_ylabel("largest eigenvalue of the weighted sum")
    ax.set_title("Two ways to stiffen at fixed factor norms: align, or concentrate")
    ax.set_ylim(0.45, 1.08); ax.set_xlim(0, 90)
    ax.legend(fontsize=9, frameon=False, loc="lower left")
    _spines(ax)
    fig.tight_layout(); fig.savefig(OUT / "jva_lambda_bounds.png", bbox_inches="tight"); plt.close(fig)


def fig_predictions():
    rows = [("omega*dt median", 1.16, 1.128, "held"), ("omega*dt maximum", 2.51, 2.435, "held"),
            ("log-normal sigma", 0.20, 0.217, "held"), ("over-wall share", 0.30, 0.423, "high, under falsifier"),
            ("gradient median", 1.4, 1.65, "close"), ("step where gradient p99 passes 100", 45300, 39206, "early"),
            ("validation PPL", 75, 93.93, "falsified")]
    col = {"held": GREEN, "close": BLUE, "high, under falsifier": ORANGE, "early": ORANGE, "falsified": RED}
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    y = np.arange(len(rows))[::-1]
    for yi, (name, pred, act, verdict) in zip(y, rows):
        r = act / pred
        ax.barh(yi, r - 1, left=1, color=col[verdict], height=0.55)
        ax.text(max(r, 1) + 0.01, yi, f"{act:g} against {pred:g}: {verdict}", va="center", fontsize=8.8)
    ax.axvline(1, color=INK, lw=1)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=9)
    ax.set_xlim(0.8, 1.75)
    ax.set_xlabel("actual over predicted at step 50,000")
    ax.set_title("Pre-registered calls: saturating fits held, 'keeps improving' fits failed")
    _spines(ax)
    fig.tight_layout(); fig.savefig(OUT / "jva_predictions.png", bbox_inches="tight"); plt.close(fig)


def fig_lognormal_shift():
    x = np.linspace(0.3, 3.2, 600)
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for med, sig, col, ls, lab in ((1.128, 0.217, BLUE, "-", "step 50,000: median 1.128, sigma 0.217"),
                                   (1.39, 0.217, PURPLE, "--", "rigid shift to median 1.39, sigma 0.217"),
                                   (1.128, 0.30, ORANGE, ":", "widening instead: median 1.128, sigma 0.30")):
        ax.plot(x, lognormal_pdf(x, med, sig), ls=ls, color=col, lw=2,
                label=f"{lab}  ->  {100 * over_wall(med, sig):.1f}% over the wall")
    ax.axvline(2.0, color=RED, lw=1.5)
    ax.text(2.04, 0.75, "the wall,\nomega*dt = 2", color=RED, fontsize=9.5)
    ax.set_xlabel("omega*dt per token and layer"); ax.set_ylabel("density")
    ax.set_title("Shift or widen: the over-wall share is a threshold count")
    ax.legend(fontsize=8.5, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.0))
    ax.set_ylim(0, 2.25)
    _spines(ax)
    fig.tight_layout(); fig.savefig(OUT / "jva_lognormal_shift.png", bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- anneal probes
def fig_anneals():
    S, e0, e1 = 4000, 3e-4, 1.5e-5
    s = np.linspace(0, S, 400)
    lr = e1 + 0.5 * (e0 - e1) * (1 + np.cos(pi * s / S))
    evs = np.arange(500, 4001, 500)
    a50 = [92.56, 90.60, 88.28, 86.49, 84.05, 85.42, 84.77, 84.91]          # 50,500 .. 54,000
    a28 = [88.16, 86.20, 84.56, 83.93, 80.75, 81.75, 81.53, 81.47]          # 29,000 .. 32,500
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1, 1.3]})
    a.plot(s, lr * 1e4, color=INK, lw=2)
    a.scatter(evs, (e1 + 0.5 * (e0 - e1) * (1 + np.cos(pi * evs / S))) * 1e4, color=INK, s=22, zorder=3)
    a.set_xlabel("steps into the anneal"); a.set_ylabel("learning rate (units of 1e-4)")
    a.set_title("(a) The compressed decay: cosine, 4,000 steps")
    _spines(a)
    b.plot(evs, a50, "-o", color=ORANGE, lw=2, label="from step 50,000 (settled 85.03)")
    b.plot(evs, a28, "-o", color=GREEN, lw=2, label="from step 28,500 (settled 81.58)")
    b.axhline(90.25, color=ORANGE, ls=":", lw=1.2); b.text(5550, 90.4, "50,000's stable level 90.25", color=ORANGE, fontsize=8.5, va="bottom", ha="right")
    b.axhline(84.31, color=GREEN, ls=":", lw=1.2); b.text(5550, 84.46, "28,500's level 84.31", color=GREEN, fontsize=8.5, va="bottom", ha="right")
    b.axhline(81.92, color=MUTED, ls="--", lw=1.1); b.text(5550, 82.07, "additive arm 81.92", color=MUTED, fontsize=8.5, va="bottom", ha="right")
    b.set_xlim(300, 5600); b.set_ylim(79, 94)
    b.set_xlabel("steps into the anneal"); b.set_ylabel("validation perplexity")
    b.set_title("(b) Same decay, two starting states")
    b.legend(fontsize=8.5, frameon=False, loc="upper center")
    _spines(b)
    fig.tight_layout(); fig.savefig(OUT / "jva_anneals.png", bbox_inches="tight"); plt.close(fig)


def fig_branch_health():
    rows = [("spikes in 4,000 steps", 0, 3), ("largest gradient norm", 4.22, 192.9), ("omega*dt median", 0.983, 1.128),
            ("over-wall share (%)", 0.05, 0.7), ("bproj_sig", 24.88, 36.80)]
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.4))
    for ax, (name, v28, v50) in zip(axes, rows):
        ax.bar([0, 1], [v28, v50], color=[GREEN, ORANGE], width=0.6)
        ax.set_xticks([0, 1], ["28,500", "50,000"], fontsize=9)
        ax.set_title(name, fontsize=10)
        for xi, v in ((0, v28), (1, v50)):
            ax.text(xi, v, f"{v:g}", ha="center", va="bottom", fontsize=8.5)
        ax.set_ylim(0, max(v28, v50) * 1.25 + 1e-9)
        _spines(ax)
    axes[3].text(0.5, -0.32, "upper ends of the ranges recorded", transform=axes[3].transAxes, ha="center", fontsize=8, color=MUTED)
    fig.suptitle("The two branch points during their anneals: 28,500 is healthier on every reading", fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "jva_branch_health.png", bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- integration refinement
def fig_leapfrog_error():
    x = np.linspace(0.01, 1.999, 400)
    err = 2 * np.arcsin(x / 2) / x - 1
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(x, 100 * err, color=BLUE, lw=2)
    for v, lab in ((0.564, "substeps, dt 0.5"), (1.128, "median at step 50,000"), (1.5, ""), (1.9, "")):
        e = 100 * (2 * asin(v / 2) / v - 1)
        ax.scatter([v], [e], color=INK, s=28, zorder=3)
        ax.annotate(f"{e:.1f}%" + (f"\n{lab}" if lab else ""), (v, e), textcoords="offset points", xytext=(-8, 10),
                    fontsize=8.5, ha="right")
    ax.axvline(2.0, color=RED, lw=1.5)
    ax.axvspan(2.0, 2.6, color=RED, alpha=0.08)
    ax.text(2.04, 25, "no real solution:\nunstable\n(maximum at step\n50,000: 2.435)", color=RED, fontsize=8.5)
    ax.set_xlim(0, 2.6); ax.set_ylim(0, 60)
    ax.set_xlabel("omega*dt"); ax.set_ylabel("leapfrog frequency error (%)")
    ax.set_title("An explicit kick is under-resolved long before the wall")
    _spines(ax)
    fig.tight_layout(); fig.savefig(OUT / "jva_leapfrog_error.png", bbox_inches="tight"); plt.close(fig)


def fig_dt_wall():
    x = np.linspace(0.15, 12, 1200)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for dt, col in ((0.5, GREEN), (1.0, BLUE), (2.0, ORANGE), (4.0, RED)):
        med = 1.128 * dt
        ax.plot(x, lognormal_pdf(x, med, 0.217), color=col, lw=2,
                label=f"dt {dt:g}: median {med:.2f}, {100 * over_wall(med, 0.217):.1f}% over the wall")
    ax.axvline(2.0, color=INK, lw=1.4, ls="--")
    ax.text(2.06, 3.3, "wall", fontsize=9.5)
    ax.set_xscale("log"); ax.set_xlim(0.2, 12)
    ax.set_xlabel("omega*dt (log scale)"); ax.set_ylabel("density")
    ax.set_title("omega*dt scales with dt: coarsening moves the distribution past the wall")
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    _spines(ax)
    fig.tight_layout(); fig.savefig(OUT / "jva_dt_wall.png", bbox_inches="tight"); plt.close(fig)


def fig_lowrank_vs_control():
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.4))
    st = [50000, 50500, 51000]
    a.plot(st, [93.93, 92.71, 91.67], "-o", color=PURPLE, lw=2, label="exact low-rank integrator")
    a.plot(st, [93.93, 93.27, 93.26], "-o", color=GREY, lw=2, label="control (baoab_cfc)")
    a.set_xticks(st, ["50,000", "50,500", "51,000"])
    a.set_ylabel("validation perplexity"); a.set_title("(a) Validation: ahead, but n = 2 (1.03 sigma)")
    a.legend(fontsize=9, frameon=False)
    _spines(a)
    names = ["mean", "std", "max"]; c = [3.67, 2.59, 12.16]; l = [2.34, 1.47, 7.94]
    xx = np.arange(3)
    b.bar(xx - 0.18, c, 0.34, color=GREY, label="control")
    b.bar(xx + 0.18, l, 0.34, color=PURPLE, label="exact low-rank")
    for xi, (cv, lv) in enumerate(zip(c, l)):
        b.text(xi, max(cv, lv) + 0.3, f"{cv / lv:.2f}x", ha="center", fontsize=9)
    b.set_xticks(xx, [f"gradient {n}" for n in names])
    b.set_title("(b) Gradient norms over 20 matched steps")
    b.legend(fontsize=9, frameon=False)
    _spines(b)
    fig.tight_layout(); fig.savefig(OUT / "jva_lowrank_vs_control.png", bbox_inches="tight"); plt.close(fig)


def fig_linalg_bench():
    rows = [("Gram driver, all 32 modes", 40.1, GREEN), ("small SVD, joint (20 x 32)", 12.0, BLUE),
            ("QR, randomised path (x3 per call)", 1108.7, ORANGE), ("svd driver, q = 16", 3368.9, RED),
            ("small SVD, additive (20 x 160)", 6335.4, GREY), ("full SVD, joint (384 x 32)", 9111.8, GREY)]
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    y = np.arange(len(rows))[::-1]
    for yi, (n, v, col) in zip(y, rows):
        ax.barh(yi, v, color=col, height=0.6)
        ax.text(v * 1.08, yi, f"{v:,.1f} ms", va="center", fontsize=9)
    ax.set_xscale("log"); ax.set_xlim(5, 60000)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=9)
    ax.set_xlabel("milliseconds per call (A100, 8,192 tokens; log scale)")
    ax.set_title("The Gram driver is 84 times faster than the svd driver")
    _spines(ax)
    fig.tight_layout(); fig.savefig(OUT / "jva_linalg_bench.png", bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- matched GPT-2
def fig_gpt2_budget():
    st = np.linspace(0, 32500, 500)
    fock = st * 16384 / 1e6
    gpt = np.where(st <= 14000, st * 4096, 14000 * 4096 + (st - 14000) * 16384) / 1e6
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.plot(st, fock, color=PURPLE, lw=2, label="Fock joint arm: 16,384 tokens per step (532.5M in all)")
    ax.plot(st, gpt, color=GREEN, lw=2, label="GPT-2 as run: 4,096 to step 14,000, then 16,384 (360.4M)")
    xc = 15164; yc = (14000 * 4096 + 1164 * 16384) / 1e6
    ax.scatter([xc], [yc], color=GREEN, s=50, zorder=3)
    ax.annotate(f"GPT-2 reaches Fock's settled 81.58\nat step 15,164: {yc:.1f}M tokens,\n14.3% of Fock's budget", (xc, yc),
                textcoords="data", xytext=(20500, 20), fontsize=9, arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.set_xlabel("training step"); ax.set_ylabel("tokens seen (millions)")
    ax.set_title("Equal steps were not equal tokens")
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    _spines(ax)
    fig.tight_layout(); fig.savefig(OUT / "jva_gpt2_budget.png", bbox_inches="tight"); plt.close(fig)


def fig_four_axes():
    rows = [("perplexity (settled), Fock over GPT-2", 1.49), ("total parameters", 2.28),
            ("non-embedding parameters", 2.66), ("inference cost per token", 8.84), ("training time per step", 14.4)]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], color=[RED, BLUE, BLUE, ORANGE, ORANGE], height=0.55)
    for yi, (_, v) in zip(y, rows):
        ax.text(v * 1.05, yi, f"{v:g}x", va="center", fontsize=9.5)
    ax.set_xscale("log"); ax.set_xlim(1, 25)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=9)
    ax.set_xlabel("factor in GPT-2's favour (log scale)")
    ax.set_title("Fock-PARFLM as configured lost on every axis at once")
    _spines(ax)
    fig.tight_layout(); fig.savefig(OUT / "jva_four_axes.png", bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- content routing
def _box(ax, x, y, w, h, text, fc, ec=INK, fs=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK)


def _arrow(ax, x0, y0, x1, y1, col=INK):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13, color=col, lw=1.3))


def fig_two_slots():
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    _box(ax, 0.2, 2.1, 1.6, 0.8, "past states\nh_0 .. h_t", "#F3F4F6")
    _box(ax, 2.6, 3.2, 2.6, 1.0, "context channels\n(pooling slot)", "#DBEAFE", BLUE)
    _box(ax, 2.6, 0.8, 2.6, 1.0, "pair potential\n(pair slot)", "#FEF3C7", ORANGE)
    _box(ax, 6.2, 3.2, 2.4, 1.0, "well bank V_theta", "#EDE9FE", PURPLE)
    _box(ax, 9.4, 2.0, 2.3, 1.0, "token's layer step", "#DCFCE7", GREEN)
    _arrow(ax, 1.8, 2.6, 2.6, 3.7); _arrow(ax, 1.8, 2.4, 2.6, 1.3)
    _arrow(ax, 5.2, 3.7, 6.2, 3.7); _arrow(ax, 8.6, 3.6, 9.4, 2.7); _arrow(ax, 5.2, 1.3, 9.4, 2.3)
    ax.text(2.6, 4.55, "Alternative E: content term inside the channels' softmax", ha="left", fontsize=9.5, color=BLUE)
    ax.text(3.9, 0.35, "Family A: attention routed on the channels replaces the top-k pair force", ha="center", fontsize=9.5, color=ORANGE)
    ax.text(7.4, 2.85, "87.6% of compute,\n93.9% of non-embedding parameters", ha="center", va="top", fontsize=8.5, color=PURPLE)
    ax.text(7.2, 1.05, "3.85% of compute", ha="center", fontsize=8.5, color=ORANGE)
    ax.set_title("Two slots for content addressing", loc="left")
    fig.tight_layout(); fig.savefig(OUT / "jva_two_slots.png", bbox_inches="tight"); plt.close(fig)


def fig_alt_e():
    st = np.array([29000, 29500, 30000, 30500, 31000, 31500, 32000, 32500])
    ctrl = np.array([88.16, 86.20, 84.56, 83.93, 80.75, 81.75, 81.53, 81.47])
    alte = np.array([87.36, 84.83, 83.16, 82.16, 79.17, 79.99, 79.70, 79.70])
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.5))
    a.plot(st, ctrl, "-o", color=GREY, lw=2, label="control: the same decay without the content term (settled 81.59)")
    a.plot(st, alte, "-o", color=BLUE, lw=2, label="Alternative E (settled 79.80)")
    a.axhspan(60, 71.6, color=GREEN, alpha=0.08); a.text(29100, 71.0, "band 'prediction met' (71.6 or less)", fontsize=8.5, color=GREEN, va="top")
    a.set_ylim(68, 90)
    a.set_xlabel("training step"); a.set_ylabel("validation perplexity")
    a.set_title("(a) Better at every evaluation")
    a.legend(fontsize=8, frameon=False, loc="upper right")
    _spines(a)
    gap = np.log(ctrl / alte)
    b.plot(st, gap, "-o", color=BLUE, lw=2, label="measured gap")
    b.axhline(log(81.58 / 71.58), color=GREEN, ls="--", lw=1.4, label="pre-registered: 10 PPL, 0.131 nats")
    b.axhline(0.0210, color=MUTED, ls=":", lw=1.2, label="saturated at about 0.021 nats")
    b.set_ylim(0, 0.145)
    b.set_xlabel("training step"); b.set_ylabel("gap, control over Alternative E (nats)")
    b.set_title("(b) Real, and 17% of the prediction")
    b.legend(fontsize=8.5, frameon=False, loc="center right")
    _spines(b)
    fig.tight_layout(); fig.savefig(OUT / "jva_alt_e.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for f in (fig_run_story, fig_lambda_bounds, fig_predictions, fig_lognormal_shift, fig_anneals, fig_branch_health,
              fig_leapfrog_error, fig_dt_wall, fig_lowrank_vs_control, fig_linalg_bench, fig_gpt2_budget, fig_four_axes,
              fig_two_slots, fig_alt_e):
        f()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
