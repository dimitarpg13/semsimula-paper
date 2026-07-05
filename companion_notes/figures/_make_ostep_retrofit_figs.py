"""Figures for section 8 of
Langevin_dynamics_reformulation_of_classical_damped_Lagrangian_flow.md
(the O-step retrofit: assumptions, inaccuracies, and the two knobs).

Generates three PNGs into this folder:

  ostep_retrofit_error_taxonomy.png  -- sources of inaccuracy ranked by
                                        severity for the DEFAULT config, colour
                                        coded by how each is (or is not) fixed
  ostep_steps_scaling.png            -- will more integration steps help: the
                                        three step axes (inner O-steps, depth,
                                        exact simulator) and the equilibration
                                        floor set by inhomogeneity / NESS
  ostep_gamma_temperature_sweep.png  -- schematic (gamma, T) val-PPL landscape
                                        + the 1-D U-curve / entropy slice

All surfaces are SCHEMATIC (synthetic illustrative functions), not measured.
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


# ---------------------------------------------------------------------------
# Figure 1 -- error taxonomy
# ---------------------------------------------------------------------------
def fig_taxonomy():
    # (label, severity 0-5, colour, how-it-is-addressed tag)
    items = [
        ("Finite depth is not equilibration  (L~16 O-steps do not sample the measure)",
         5, RED, "structural  ->  exact simulator"),
        ("Inhomogeneous potential  (a different V_theta at every layer)",
         4, RED, "structural  ->  exact simulator"),
        ("Non-conservative Fock force Q  (steady state is NESS, not Gibbs)",
         4, GREY, "OFF by default (reverse channel off)"),
        ("First-order splitting, single force eval  (not palindromic BAOAB; dt=1 not small)",
         3, PURPLE, "discretisation  ->  inner O-steps / smaller dt"),
        ("Uncalibrated (gamma, T); observed spread double-counted",
         3, AMBER, "fixable  ->  (gamma, T) sweep"),
        ("LayerNorm velocity-proxy distortion  (v = (h_new - h)/dt after projection)",
         2, PURPLE, "discretisation (second order)"),
    ]

    fig, ax = plt.subplots(figsize=(12, 5.4))
    ys = np.arange(len(items))[::-1]
    for y, (label, sev, col, tag) in zip(ys, items):
        ax.barh(y, sev, color=col, alpha=0.88, height=0.62,
                edgecolor=DARK, linewidth=0.6)
        ax.text(0.08, y + 0.14, label, va="center", ha="left",
                fontsize=8.8, color="white", fontweight="bold")
        ax.text(0.08, y - 0.2, tag, va="center", ha="left",
                fontsize=8.2, color="white", style="italic")

    ax.set_xlim(0, 5.6)
    ax.set_ylim(-0.6, len(items) - 0.3)
    ax.set_yticks([])
    ax.set_xlabel("Relative severity for the DEFAULT retrofit config "
                  "(reverse channel OFF)")
    ax.set_title(
        "O-step retrofit: sources of inaccuracy, ranked\n"
        "red = structural (only the exact simulator removes it)   "
        "amber = removed by a (gamma, T) sweep   "
        "purple = discretisation (smaller steps)   grey = inactive by default",
        fontsize=10.5,
    )
    # sigma->0 reminder
    ax.text(5.5, -0.5,
            "all sources vanish as noise -> 0 (both collapse to the deterministic damped flow)",
            ha="right", va="center", fontsize=8.2, color=GREY, style="italic")
    fig.tight_layout()
    fig.savefig("ostep_retrofit_error_taxonomy.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 -- will more integration steps help
# ---------------------------------------------------------------------------
def _box(ax, x, y, w, h, text, fc, ec=DARK, fs=9.0, tc="white"):
    b = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.03",
                       linewidth=1.2, edgecolor=ec, facecolor=fc)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc)


def fig_steps():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2),
                             gridspec_kw={"width_ratios": [1.05, 1.0]})

    # ---- Left: distance-to-local-stationary vs inner O-steps ----
    ax = axes[0]
    n = np.arange(1, 33)
    floor_cons = 0.10
    floor_ness = 0.30
    kl_cons = 0.95 * np.exp(-0.30 * (n - 1)) + floor_cons
    kl_ness = 0.95 * np.exp(-0.30 * (n - 1)) + floor_ness

    ax.plot(n, kl_cons, "-o", color=GREEN, ms=3.5, lw=2,
            label="conservative drift (reverse channel OFF)")
    ax.plot(n, kl_ness, "-o", color=RED, ms=3.5, lw=2,
            label="with non-conservative Q (reverse channel ON)")
    ax.axhline(floor_cons, ls="--", color=GREEN, lw=1)
    ax.axhline(floor_ness, ls="--", color=RED, lw=1)

    ax.annotate("current retrofit:\n1 O-step / layer",
                xy=(1, kl_cons[0]), xytext=(4.5, 0.86), fontsize=9,
                color=DARK, arrowprops=dict(arrowstyle="->", color=DARK))
    ax.text(32, floor_cons + 0.015, "inhomogeneity floor",
            ha="right", va="bottom", fontsize=8.3, color="#15803D")
    ax.text(32, floor_ness + 0.015, "inhomogeneity + NESS floor",
            ha="right", va="bottom", fontsize=8.3, color="#B91C1C")

    ax.set_xlabel("inner O-steps per layer (frozen potential)")
    ax.set_ylabel("distance to the local stationary law (KL, schematic)")
    ax.set_title("(a) More inner steps thermalise the LOCAL potential\n"
                 "exponentially — down to a floor they cannot cross")
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    # ---- Right: the three step axes and what each fixes ----
    ax = axes[1]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("(b) Three 'more steps' axes — only one fixes both errors")

    _box(ax, 0.04, 0.70, 0.92, 0.20,
         "More inner O-steps per layer\n"
         "fixes: local non-equilibration    does NOT fix: inhomogeneity\n"
         "cost: linear, drop-in in the retrofit", BLUE, fs=9)
    _box(ax, 0.04, 0.42, 0.92, 0.20,
         "More layers L (depth)\n"
         "fixes: capacity / refinement    does NOT fix: sampling fidelity\n"
         "cost: linear + params, more distinct potentials", PURPLE, fs=9)
    _box(ax, 0.04, 0.10, 0.92, 0.22,
         "Exact simulator: many small steps of ONE potential\n"
         "fixes: non-equilibration AND inhomogeneity (true Gibbs sampling)\n"
         "cost: separate vehicle (STP-BAOAB), already built", GREEN, fs=9)

    fig.suptitle(
        "Will increasing the integration steps help? Inner steps buy local "
        "equilibration; only the homogeneous many-step simulator buys the theory.",
        fontsize=11.5, y=1.02,
    )
    fig.tight_layout()
    fig.savefig("ostep_steps_scaling.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 -- (gamma, T) sweep landscape
# ---------------------------------------------------------------------------
def fig_gamma_T():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})

    # ---- Left: schematic val-PPL contour over (T, gamma) ----
    ax = axes[0]
    T = np.logspace(np.log10(0.05), np.log10(3.0), 240)
    G = np.logspace(np.log10(0.05), np.log10(2.0), 240)
    TT, GG = np.meshgrid(T, G)
    Tstar, Gstar, base = 0.30, 0.30, 118.0
    J = (base
         + 26.0 * (np.log(TT / Tstar)) ** 2
         + 16.0 * (np.log(GG / Gstar)) ** 2
         + 6.0 * (np.log(TT / Tstar)) * (np.log(GG / Gstar)) * 0.0)
    cf = ax.contourf(TT, GG, J, levels=18, cmap="viridis_r")
    cs = ax.contour(TT, GG, J, levels=8, colors="white",
                    linewidths=0.4, alpha=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(Tstar, Gstar, "*", color=AMBER, ms=20, mec=DARK, mew=0.8,
            zorder=5)
    ax.text(Tstar * 1.15, Gstar * 1.15, "sweet spot", color="white",
            fontsize=9, fontweight="bold")
    ax.text(0.06, 1.6, "cold: Verlet collapse\n(low entropy, punct. basins)",
            color="white", fontsize=8, va="top")
    ax.text(2.9, 1.6, "hot: noise-dominated\n(loss up, entropy -> uniform)",
            color="white", fontsize=8, ha="right", va="top")
    ax.text(2.9, 0.055, "high gamma: overdamped\n(fast local mixing, slow transport)",
            color="white", fontsize=8, ha="right", va="bottom")
    ax.text(0.06, 0.055, "low gamma: underdamped\n(ballistic, weak noise/step)",
            color="white", fontsize=8, va="bottom")
    ax.set_xlabel("temperature  T   (noise amplitude via FDT)")
    ax.set_ylabel("friction  gamma   (mixing rate)")
    ax.set_title("(a) Schematic val-PPL over (gamma, T)\n"
                 "the sweep finds the basin; it cannot lower the whole surface")
    cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("val PPL (schematic)")

    # ---- Right: 1-D slice at gamma = gamma* : PPL U-curve + entropy ----
    ax = axes[1]
    Tt = np.logspace(np.log10(0.05), np.log10(3.0), 200)
    ppl = base + 26.0 * (np.log(Tt / Tstar)) ** 2
    ent = 2.0 + 2.1 / (1.0 + np.exp(-1.6 * np.log(Tt / 0.28)))

    l1, = ax.plot(Tt, ppl, color=BLUE, lw=2.4, label="val PPL (U-shaped)")
    ax.axvline(Tstar, ls="--", color=AMBER, lw=1.4)
    ax.set_xscale("log")
    ax.set_xlabel("temperature  T   (gamma fixed at the sweet spot)")
    ax.set_ylabel("val PPL (schematic)", color=BLUE)
    ax.tick_params(axis="y", labelcolor=BLUE)
    ax.set_title("(b) Temperature is the clean axis:\n"
                 "PPL is U-shaped, predictive entropy rises monotonically")

    ax2 = ax.twinx()
    l2, = ax2.plot(Tt, ent, color=GREEN, lw=2.4, ls="-",
                   label="predictive entropy (nats)")
    ax2.set_ylabel("predictive entropy (schematic)", color=GREEN)
    ax2.tick_params(axis="y", labelcolor=GREEN)

    ax.annotate("tie T = 1/beta = 1\nstart here, then sweep",
                xy=(Tstar, ppl.min()), xytext=(Tstar * 1.25, ppl.min() + 40),
                fontsize=8.2, color=DARK,
                arrowprops=dict(arrowstyle="->", color=DARK))
    ax.legend(handles=[l1, l2], frameon=False, fontsize=8.6, loc="upper left")

    fig.suptitle(
        "Will a (gamma, T) sweep help? It calibrates the regulariser optimum "
        "(removes the double-counting bias) — but not the structural errors.",
        fontsize=11.5, y=1.02,
    )
    fig.tight_layout()
    fig.savefig("ostep_gamma_temperature_sweep.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_taxonomy()
    fig_steps()
    fig_gamma_T()
    print("wrote: ostep_retrofit_error_taxonomy.png, ostep_steps_scaling.png, "
          "ostep_gamma_temperature_sweep.png")
