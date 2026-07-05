"""Figures for section 9 (accuracy curriculum) of
Langevin_dynamics_reformulation_of_classical_damped_Lagrangian_flow.md.

Generates two PNGs into this folder:

  accuracy_curriculum_pipeline.png   -- the two-phase warm-start curriculum:
                                        Phase A (explore, sweep) -> harvest
                                        structured potentials -> Phase B
                                        (exploit, warm-start) -> optional exact
                                        simulator, with what-each-fixes notes
                                        and the pitfalls flagged on the side
  warm_start_vs_anneal_schedule.png  -- (a) two-phase step vs smooth-anneal
                                        temperature schedule; (b) cross-run
                                        transferability of the harvestable
                                        components (what to warm-start hard)

All curves/values are SCHEMATIC illustrations, not measured data.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

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
DARK = "#111827"
AMBER = "#F59E0B"
TEAL = "#0EA5E9"


def _box(ax, x, y, w, h, text, fc, ec=DARK, fs=9.0, tc="white"):
    b = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.02",
                       linewidth=1.3, edgecolor=ec, facecolor=fc)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc)


def _arrow(ax, x1, y1, x2, y2, col=DARK, style="-|>", lw=1.6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                 arrowstyle=style, mutation_scale=15, linewidth=lw, color=col))


# ---------------------------------------------------------------------------
# Figure 1 -- the two-phase warm-start curriculum pipeline
# ---------------------------------------------------------------------------
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    cx = 0.30       # centre column x
    w = 0.40
    # stages top -> bottom
    _box(ax, cx, 0.82, w, 0.13,
         "PHASE A - explore\n"
         "O-step ON, sweep (gamma, T)\n"
         "noise carves smooth, separated basins", BLUE, fs=9.3)
    _box(ax, cx, 0.635, w, 0.12,
         "SELECT (gamma*, T*)\n"
         "by val PPL + entropy band + grad stability", TEAL, fs=9.3)
    _box(ax, cx, 0.45, w, 0.12,
         "HARVEST structured potentials\n"
         "V_theta wells + V_phi kernel  (use EMA weights)", GREEN, fs=9.3)
    _box(ax, cx, 0.255, w, 0.13,
         "PHASE B - exploit\n"
         "warm-start (NO reinit) at (gamma*, T*)\n"
         "second WSD cycle, anneal T -> 0 late", PURPLE, fs=9.3)
    _box(ax, cx, 0.06, w, 0.13,
         "EXACT SIMULATOR (optional top rung)\n"
         "harvested forces + STP-BAOAB\n"
         "removes non-equilibration AND inhomogeneity", DARK, fs=9.3)

    xcen = cx + w / 2
    _arrow(ax, xcen, 0.82, xcen, 0.755)
    _arrow(ax, xcen, 0.635, xcen, 0.57)
    _arrow(ax, xcen, 0.45, xcen, 0.385)
    _arrow(ax, xcen, 0.255, xcen, 0.19)
    # feedback: harvested forces also feed the simulator directly
    _arrow(ax, cx, 0.51, 0.12, 0.51, col=GREEN, style="-|>", lw=1.4)
    ax.text(0.115, 0.47, "same forces\nfeed the DDS", ha="center", va="top",
            fontsize=8, color="#15803D")
    _arrow(ax, 0.12, 0.47, 0.30, 0.14, col=GREEN, style="-|>", lw=1.2)

    # ---- left column: what each step FIXES ----
    ax.text(0.02, 0.965, "WHAT IT FIXES", fontsize=9.2, fontweight="bold",
            color="#15803D")
    fixes = [
        (0.885, "error #4 calibration:\nthe sweep sets (gamma, T)"),
        (0.695, "picks the regulariser\noptimum, not a lucky point"),
        (0.51, "error #1 (partly): terrain is\npre-shaped; Phase B skips\nearly-basin cost"),
        (0.32, "keeps capacity; anneal gives\na clean deterministic mode\nat inference"),
        (0.125, "errors #1 AND #2: true Gibbs\nsampling on the good terrain"),
    ]
    for y, t in fixes:
        ax.text(0.02, y, t, fontsize=8.0, color="#166534", va="center")

    # ---- right column: PITFALLS ----
    ax.text(0.735, 0.965, "PITFALLS TO WATCH", fontsize=9.2,
            fontweight="bold", color="#B91C1C")
    pit = [
        (0.885, "moving target: potentials\nre-adapt per (gamma, T);\nsweep points not 1-factor"),
        (0.695, "val metric is noisy under\nnoise: select by a band /\nmany batches, not a min"),
        (0.51, "ossification: harvested wells\nlock early basins -> keep the\nOOD-gated spawn + plasticity"),
        (0.32, "T_A != T_B mismatch;\nvelocity-proxy re-scale ->\nshort re-warm-up"),
        (0.125, "harvest only conservative V;\nif Q on, terrain is a NESS,\nnot a gradient field"),
    ]
    for y, t in pit:
        ax.text(0.735, y, t, fontsize=8.0, color="#991B1B", va="center")

    fig.suptitle(
        "Improving accuracy: the explore -> harvest -> exploit curriculum "
        "(the retrofit shapes and calibrates the terrain; the simulator samples it exactly)",
        fontsize=12, y=0.99,
    )
    fig.tight_layout()
    fig.savefig("accuracy_curriculum_pipeline.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 -- schedule choice + what to warm-start hard
# ---------------------------------------------------------------------------
def fig_schedule_and_transfer():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2),
                             gridspec_kw={"width_ratios": [1.05, 1.0]})

    # ---- (a) temperature schedules ----
    ax = axes[0]
    t = np.linspace(0, 1, 400)
    Thi, Tlo = 1.0, 0.05
    # two hard phases: constant T_A, then drop to T_B, then anneal to 0 late
    step = np.where(t < 0.5, Thi, 0.35)
    step = np.where(t > 0.85, Tlo + (0.35 - Tlo) * (1 - (t - 0.85) / 0.15),
                    step)
    step = np.clip(step, Tlo, Thi)
    # smooth anneal: cosine from Thi to Tlo
    smooth = Tlo + 0.5 * (Thi - Tlo) * (1 + np.cos(np.pi * t))

    ax.plot(t, step, color=PURPLE, lw=2.6, label="two hard phases (A then B, anneal late)")
    ax.plot(t, smooth, color=TEAL, lw=2.6, ls="--", label="single-run smooth anneal T(t)")
    ax.axvspan(0.0, 0.5, color=BLUE, alpha=0.07)
    ax.axvspan(0.85, 1.0, color=GREEN, alpha=0.07)
    ax.text(0.25, 1.02, "explore", ha="center", fontsize=9, color=BLUE)
    ax.text(0.925, 1.02, "commit", ha="center", fontsize=9, color="#15803D")
    ax.axvline(0.5, ls=":", color=GREY, lw=1)
    ax.text(0.5, 0.55, "Phase A | Phase B\nwarm-start boundary", ha="center",
            fontsize=8, color=DARK, rotation=0)
    ax.set_xlabel("training progress")
    ax.set_ylabel("temperature  T  (noise amplitude)")
    ax.set_title("(a) Two hard phases vs a smooth anneal\n"
                 "smooth avoids the phase-boundary forgetting risk")
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, fontsize=8.4, loc="lower left")

    # ---- (b) transferability of harvestable components ----
    ax = axes[1]
    comps = ["V_phi\npairwise kernel", "V_theta\ncontext heads",
             "xi-routing /\ndepth-cond", "thin MLP\nresidual"]
    vals = [0.90, 0.55, 0.30, 0.20]
    cols = [GREEN, AMBER, RED, GREY]
    bars = ax.bar(comps, vals, color=cols, width=0.62, edgecolor=DARK,
                  linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", fontweight="bold", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("cross-run transferability (schematic)")
    ax.set_title("(b) What to warm-start HARD vs let ADAPT\n"
                 "run-invariant physics transfers; context does not")
    ax.annotate("warm-start / freeze\n(run-invariant physics)",
                xy=(0, 0.90), xytext=(0.15, 0.66), fontsize=8.2,
                color="#15803D",
                arrowprops=dict(arrowstyle="->", color="#15803D"))
    ax.annotate("keep plastic + OOD-gated spawn\n(context-specific)",
                xy=(2, 0.30), xytext=(1.35, 0.80), fontsize=8.2, color="#B91C1C",
                arrowprops=dict(arrowstyle="->", color="#B91C1C"))

    fig.suptitle(
        "Two improvements over a hard two-phase warm-start: anneal continuously, "
        "and transfer only the run-invariant part of the potential",
        fontsize=11.5, y=1.02,
    )
    fig.tight_layout()
    fig.savefig("warm_start_vs_anneal_schedule.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_pipeline()
    fig_schedule_and_transfer()
    print("wrote: accuracy_curriculum_pipeline.png, "
          "warm_start_vs_anneal_schedule.png")
