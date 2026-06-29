"""
Generate the two illustrative figures for
Multi_Timescale_Hierarchical_Integration_for_SPLM.md

1. mts_respa_schedule.png       -- the nested r-RESPA integration schedule
                                   (outer slow force evals, inner fast substeps)
2. mts_depth_param_decoupling.png -- flat pipeline (depth=params=time coupled)
                                   vs hierarchical pipeline (decoupled axes)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

# ── colour palette ────────────────────────────────────────────────
BLUE_DARK = "#1565C0"
BLUE_MED = "#1E88E5"
BLUE_LIGHT = "#90CAF9"
TEAL_DARK = "#00695C"
TEAL_MED = "#00897B"
ORANGE = "#EF6C00"
ORANGE_LT = "#FFB74D"
RED = "#E53935"
PURPLE = "#6A1B9A"
GREY_DASH = "#9E9E9E"
GOLD = "#F9A825"
BG = "#FAFAFA"


# ===================================================================
#  FIGURE 1 — the nested r-RESPA integration schedule
# ===================================================================
def make_schedule():
    fig, ax = plt.subplots(figsize=(13, 6.2), facecolor=BG)
    ax.set_facecolor(BG)

    n_outer = 4          # slow / outer steps
    n_inner = 4          # fast substeps per outer step
    dt_outer = 1.0
    dt_inner = dt_outer / n_inner

    y_slow = 2.6
    y_fast = 1.0
    y_axis = 0.15

    t_max = n_outer * dt_outer

    # time axis
    ax.annotate("", xy=(t_max + 0.35, y_axis), xytext=(-0.15, y_axis),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.4))
    ax.text(t_max + 0.42, y_axis, "layer / time", fontsize=10,
            va="center", ha="left")

    # ---- inner (fast) substeps : half-kick / drift / half-kick ----
    inner_times = []
    for i in range(n_outer):
        t0 = i * dt_outer
        for j in range(n_inner):
            inner_times.append(t0 + j * dt_inner)
    inner_times.append(t_max)

    # fast force evaluation ticks
    for t in inner_times:
        ax.plot([t, t], [y_fast - 0.18, y_fast + 0.18],
                color=TEAL_DARK, lw=1.4, zorder=4)
    ax.plot([0, t_max], [y_fast, y_fast], color=TEAL_MED, lw=2.0, zorder=3)
    for t in inner_times[:-1]:
        ax.plot(t + dt_inner / 2, y_fast, "o", color=TEAL_MED, ms=5, zorder=5)

    ax.text(-0.15, y_fast + 0.46,
            "FAST inner force  $f_{\\rm fast}=-\\nabla V_{\\rm local}$",
            fontsize=10.5, color=TEAL_DARK, fontweight="bold", ha="left")
    ax.text(-0.15, y_fast - 0.52,
            f"evaluated every inner substep  (dt_fast = dt_slow / {n_inner})",
            fontsize=9, color=TEAL_DARK, ha="left")

    # ---- outer (slow) steps : long-range force kicks --------------
    outer_times = [i * dt_outer for i in range(n_outer + 1)]
    for t in outer_times:
        ax.plot([t, t], [y_slow - 0.22, y_slow + 0.22],
                color=ORANGE, lw=2.6, zorder=4)
    ax.plot([0, t_max], [y_slow, y_slow], color=ORANGE_LT, lw=2.4, zorder=3)

    # half-kick markers at the slow level
    for t in outer_times:
        ax.plot(t, y_slow, "s", color=ORANGE, ms=8, zorder=5)

    ax.text(-0.15, y_slow + 0.5,
            "SLOW outer force  $f_{\\rm slow}=-\\nabla V_{\\rm global}$",
            fontsize=10.5, color=ORANGE, fontweight="bold", ha="left")
    ax.text(-0.15, y_slow - 0.62,
            "evaluated once per outer step  (the expensive O(T^2) term)",
            fontsize=9, color=ORANGE, ha="left")

    # ---- connectors: each slow half-kick brackets n_inner fast steps
    for i in range(n_outer):
        t0 = i * dt_outer
        t1 = (i + 1) * dt_outer
        # vertical dashed guides
        for t in (t0, t1):
            ax.plot([t, t], [y_axis, y_slow], color=GREY_DASH,
                    lw=0.8, ls=":", zorder=1)
        # shaded outer-step band
        ax.add_patch(mpatches.Rectangle(
            (t0, y_axis), dt_outer, y_slow - y_axis,
            facecolor=BLUE_LIGHT, alpha=0.10, zorder=0))
        ax.text(t0 + dt_outer / 2, y_axis - 0.30,
                f"outer step {i+1}", fontsize=9, ha="center",
                color=BLUE_DARK)

    # Trotter factorization annotation
    ax.text(t_max / 2, 3.55,
            "One outer step  =  half slow-kick  o  "
            "[ n inner Verlet substeps ]  o  half slow-kick",
            fontsize=10.5, ha="center", color=PURPLE, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec=PURPLE, lw=1.4))

    ax.set_xlim(-1.6, t_max + 2.3)
    ax.set_ylim(-0.7, 3.9)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(
        "Multi-timescale (r-RESPA) integration schedule for SPLM\n"
        "cheap local force on a fine grid, expensive global force on a coarse grid",
        fontsize=12.5, fontweight="bold", color=BLUE_DARK, pad=12,
        linespacing=1.4)

    out = ("/Users/dimitargueorguiev/git/ml/semsimula-paper/"
           "companion_notes/images/mts_respa_schedule.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Saved {out}")
    plt.close(fig)


# ===================================================================
#  FIGURE 2 — flat vs hierarchical: decoupling depth / params / time
# ===================================================================
def make_decoupling():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6),
                             facecolor=BG,
                             gridspec_kw={"wspace": 0.18})

    # ---------------- LEFT : flat pipeline -----------------------
    ax = axes[0]
    ax.set_facecolor(BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    n = 6
    for i in range(n):
        y = 10.2 - i * 1.6
        ax.add_patch(FancyBboxPatch(
            (3.0, y), 4.0, 1.05,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            fc=BLUE_LIGHT, ec=BLUE_DARK, lw=1.6, zorder=3))
        ax.text(5.0, y + 0.52,
                f"Verlet step {i+1}   (own params)",
                fontsize=9.5, ha="center", va="center",
                color=BLUE_DARK, zorder=4)
        if i < n - 1:
            ax.annotate("", xy=(5.0, y - 0.52), xytext=(5.0, y - 0.02),
                        arrowprops=dict(arrowstyle="-|>", color=BLUE_DARK,
                                        lw=1.6))
    ax.text(5.0, 11.6, "FLAT pipeline (current)",
            fontsize=12.5, ha="center", fontweight="bold", color=BLUE_DARK)
    ax.text(5.0, 0.55,
            "depth = #steps = #parameter sets = #timescales\n"
            "one knob controls everything",
            fontsize=10, ha="center", va="center", color=RED,
            linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="#FFEBEE",
                      ec=RED, lw=1.3))

    # ---------------- RIGHT : hierarchical pipeline --------------
    ax2 = axes[1]
    ax2.set_facecolor(BG)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(-1.2, 12)
    ax2.set_xticks([])
    ax2.set_yticks([])
    for sp in ax2.spines.values():
        sp.set_visible(False)

    n_outer = 3
    n_inner = 3
    ax2.text(5.0, 11.6, "HIERARCHICAL pipeline (proposed)",
             fontsize=12.5, ha="center", fontweight="bold", color=TEAL_DARK)

    top = 10.4
    outer_h = 3.0
    gap = 0.35
    for o in range(n_outer):
        oy = top - o * (outer_h + gap)
        # outer (slow) band
        ax2.add_patch(FancyBboxPatch(
            (1.2, oy - outer_h + 0.1), 7.6, outer_h - 0.2,
            boxstyle="round,pad=0.05,rounding_size=0.15",
            fc="#FFF3E0", ec=ORANGE, lw=2.0, zorder=2))
        ax2.text(1.5, oy - 0.12,
                 f"slow step {o+1}  (global params, shared)",
                 fontsize=9, ha="left", va="top",
                 color=ORANGE, fontweight="bold", zorder=5)
        # inner fast substeps
        for k in range(n_inner):
            ix = 2.0 + k * 2.25
            iy = oy - outer_h + 0.55
            ax2.add_patch(FancyBboxPatch(
                (ix, iy), 1.9, 1.25,
                boxstyle="round,pad=0.03,rounding_size=0.1",
                fc=TEAL_MED, ec=TEAL_DARK, lw=1.3, alpha=0.85, zorder=4))
            ax2.text(ix + 0.95, iy + 0.62, f"fast\n{k+1}",
                     fontsize=8.5, ha="center", va="center",
                     color="white", fontweight="bold", zorder=5)
            if k < n_inner - 1:
                ax2.annotate("", xy=(ix + 1.95, iy + 0.62),
                             xytext=(ix + 1.9 + 0.05, iy + 0.62),
                             arrowprops=dict(arrowstyle="-|>",
                                             color=TEAL_DARK, lw=1.2))
        if o < n_outer - 1:
            ax2.annotate("", xy=(5.0, oy - outer_h + 0.05),
                         xytext=(5.0, oy - outer_h - gap + 0.15),
                         arrowprops=dict(arrowstyle="-|>", color=ORANGE,
                                         lw=2.0))

    ax2.text(5.0, -0.55,
             "depth (steps) decoupled from parameters\n"
             "slow = global topic, fast = local syntax",
             fontsize=10, ha="center", va="center", color=TEAL_DARK,
             linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.4", fc="#E0F2F1",
                       ec=TEAL_DARK, lw=1.3))

    fig.suptitle(
        "Decoupling integration depth from parameter count",
        fontsize=13.5, fontweight="bold", y=0.99, color="#333333")

    out = ("/Users/dimitargueorguiev/git/ml/semsimula-paper/"
           "companion_notes/images/mts_depth_param_decoupling.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    make_schedule()
    make_decoupling()
