"""Figures for Portable_Learned_Potentials_and_Transplant_Map.md.

Generates three PNGs into this folder:

  portable_potentials_flavor_landscape.png -- the five Fock-PARFLM flavours
      placed on the (conservativity, ensemble-consumed) plane, annotated with
      what each one makes the readout consume and what re-fits on a transplant.
  portable_potentials_transplant_matrix.png -- 5x5 producer -> consumer
      transplant-difficulty matrix (schematic), colour + text coded.
  portable_potentials_calibration_axes.png -- why the amplitude re-fit is
      integrator-dependent: Verlet consumes minima (mode selection), Langevin
      consumes Boltzmann mass (occupancy), over the SAME learned geometry.

All surfaces / placements are SCHEMATIC (illustrative), not measured.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

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
TEAL = "#14B8A6"


# ---------------------------------------------------------------------------
def fig_landscape():
    fig, ax = plt.subplots(figsize=(11.5, 7.2))

    # flavour: (x=conservativity 0..1, y=ensemble 0..1, color, label, note)
    flavs = {
        "b": (0.10, 0.12, GREEN,
              "(b) v2.1 depth-cond\nNO reverse channel",
              "purely conservative core\nVerlet -> minima\nGRAFT-OPTIMAL baseline"),
        "d": (0.34, 0.12, TEAL,
              "(d) dense all-to-all\nconservative attention",
              "conservative pair potential\nover all pairs (Verlet)\nricher V_phi, still a gradient"),
        "a": (0.74, 0.20, AMBER,
              "(a) v2.1 depth-cond\nper-layer reverse channel",
              "directed non-conservative\nrouting (Verlet drift)\nNESS occupancy"),
        "e": (0.92, 0.34, RED,
              "(e) Fock-Attention\nnon-conservative exchange",
              "asymmetric exchange force\nno scalar generator\nNESS occupancy"),
        "c": (0.18, 0.85, PURPLE,
              "(c) v2.1 depth-cond\nO-step Langevin",
              "conservative core + thermostat\nNVT -> Boltzmann exp(-beta V)\namplitudes = occupancy"),
    }
    for key, (x, y, col, lab, note) in flavs.items():
        ax.scatter([x], [y], s=1500, color=col, alpha=0.32, zorder=2,
                   edgecolor=col, linewidth=2)
        ax.scatter([x], [y], s=90, color=col, zorder=3)
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(0, 26),
                    ha="center", fontsize=10, fontweight="bold", color=DARK)
        ax.annotate(note, (x, y), textcoords="offset points", xytext=(0, -52),
                    ha="center", fontsize=8.2, color="#374151")

    # guide bands
    ax.axhline(0.5, color=GREY, lw=1, ls="--", alpha=0.7)
    ax.text(0.015, 0.52, "Langevin / NVT  (readout consumes a thermal cloud)",
            fontsize=8.5, color=PURPLE, style="italic")
    ax.text(0.015, 0.455, "Verlet / NVE  (readout consumes a collapsed mode)",
            fontsize=8.5, color=GREEN, style="italic")

    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("force-law conservativity  (gradient of a scalar  ->  directed / asymmetric)",
                  fontsize=10.5)
    ax.set_ylabel("ensemble the readout consumes\n(mode  ->  Boltzmann occupancy)",
                  fontsize=10.5)
    ax.set_title("The five Fock-PARFLM flavours: where a transplanted potential lands")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # thesis banner
    ax.text(0.52, 1.005,
            "Geometry (V_theta, V_phi) ports across all five. What re-fits is set by the axes.",
            ha="center", fontsize=9.5, fontweight="bold", color=DARK,
            bbox=dict(boxstyle="round,pad=0.4", fc="#FEF3C7", ec=AMBER))

    fig.tight_layout()
    out = "portable_potentials_flavor_landscape.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ---------------------------------------------------------------------------
def fig_matrix():
    order = ["a", "b", "c", "d", "e"]
    names = {
        "a": "(a) reverse\nper-layer",
        "b": "(b) no\nreverse",
        "c": "(c) O-step\nLangevin",
        "d": "(d) dense\nconserv attn",
        "e": "(e) Fock\nnon-cons attn",
    }
    # difficulty[prod][cons] in {0,1,2,3}
    D = {
        "a": {"a": 0, "b": 2, "c": 3, "d": 2, "e": 3},
        "b": {"a": 3, "b": 0, "c": 2, "d": 1, "e": 3},
        "c": {"a": 3, "b": 2, "c": 0, "d": 2, "e": 3},
        "d": {"a": 3, "b": 1, "c": 2, "d": 0, "e": 3},
        "e": {"a": 3, "b": 2, "c": 3, "d": 2, "e": 0},
    }
    codes = {0: "self\nanchors+g1", 1: "geometry+\nweights port",
             2: "amplitudes\nre-fit", 3: "re-fit +\nNESS / warm"}
    palette = {0: "#DCFCE7", 1: "#BFDBFE", 2: "#FDE68A", 3: "#FECACA"}
    edge = {0: GREEN, 1: BLUE, 2: AMBER, 3: RED}

    n = len(order)
    fig, ax = plt.subplots(figsize=(10.6, 7.6))
    for i, prod in enumerate(order):        # rows top->bottom
        for j, cons in enumerate(order):
            lvl = D[prod][cons]
            y = n - 1 - i
            ax.add_patch(FancyBboxPatch(
                (j + 0.04, y + 0.04), 0.92, 0.92,
                boxstyle="round,pad=0.008", fc=palette[lvl],
                ec=edge[lvl], lw=2))
            ax.text(j + 0.5, y + 0.5, codes[lvl], ha="center", va="center",
                    fontsize=8.6, color=DARK, fontweight="bold")

    ax.set_xlim(0, n); ax.set_ylim(0, n)
    ax.set_xticks([j + 0.5 for j in range(n)])
    ax.set_xticklabels([names[c] for c in order], fontsize=9)
    ax.set_yticks([n - 1 - i + 0.5 for i in range(n)])
    ax.set_yticklabels([names[p] for p in order], fontsize=9)
    ax.set_xlabel("CONSUMER  C  (the run you are initialising)", fontsize=10.5,
                  fontweight="bold")
    ax.set_ylabel("PRODUCER  P  (the converged checkpoint)", fontsize=10.5,
                  fontweight="bold")
    ax.set_title("Transplant difficulty: producer -> consumer  (schematic)")
    ax.set_xticks(np.arange(0, n + 1), minor=True)
    ax.set_yticks(np.arange(0, n + 1), minor=True)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    # legend
    handles = [plt.Line2D([0], [0], marker="s", color="w",
                          markerfacecolor=palette[k], markeredgecolor=edge[k],
                          markeredgewidth=2, markersize=15,
                          label=f"{k}: " + codes[k].replace("\n", " "))
               for k in (0, 1, 2, 3)]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.09), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    out = "portable_potentials_transplant_matrix.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ---------------------------------------------------------------------------
def fig_calibration():
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), sharey=True)
    x = np.linspace(-6, 6, 800)

    # two wells: a wide shallow one and a narrow deep one
    def well(x, mu, sig, depth):
        return -depth * np.exp(-((x - mu) ** 2) / (2 * sig ** 2))
    V = well(x, -2.4, 1.30, 1.00) + well(x, 2.2, 0.55, 1.25)
    mins = [(-2.4, 1.30, 1.00), (2.2, 0.55, 1.25)]

    # LEFT: Verlet -> collapses to modes; bar height = mode-selection freq f_j
    ax = axes[0]
    ax.plot(x, V, color=DARK, lw=2.2, zorder=3)
    ax.fill_between(x, V, 0.6, color="#F3F4F6", zorder=1)
    fsel = [0.45, 0.55]                          # illustrative Verlet freqs
    for (mu, sig, dep), f in zip(mins, fsel):
        ax.annotate("", xy=(mu, -dep), xytext=(mu, 0.35),
                    arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.4))
        ax.scatter([mu], [-dep], s=120, color=GREEN, zorder=4)
        ax.text(mu, 0.44, f"mode\nf={f:.2f}", ha="center", fontsize=8.5,
                color=GREEN, fontweight="bold")
    ax.set_title("(A) Verlet / NVE: readout consumes the MODE")
    ax.text(0.5, -0.22, "amplitudes act as mode-selection weights;\n"
            "geometry alone suffices  ->  weights port as-is",
            transform=ax.transAxes, ha="center", fontsize=9, color="#374151")

    # RIGHT: Langevin -> Boltzmann cloud; area = occupancy w (2 pi sig^2)^{d/2} e^{b depth}
    ax = axes[1]
    ax.plot(x, V, color=DARK, lw=2.2, zorder=3)
    beta = 1.0
    rho = np.exp(-beta * V)
    rho = rho / np.trapz(rho, x)
    ax.fill_between(x, -3.2 + 2.4 * rho / rho.max(), -3.2,
                    color=PURPLE, alpha=0.35, zorder=2)
    ax.plot(x, -3.2 + 2.4 * rho / rho.max(), color=PURPLE, lw=2, zorder=3)
    # occupancy shares (schematic: wide shallow well can win mass)
    ax.text(-2.4, 0.15, "wide well\nBIG thermal mass", ha="center",
            fontsize=8.5, color=PURPLE, fontweight="bold")
    ax.text(2.2, 0.15, "narrow well\nless mass", ha="center",
            fontsize=8.5, color=PURPLE, fontweight="bold")
    ax.set_title("(B) O-step Langevin / NVT: readout consumes OCCUPANCY")
    ax.text(0.5, -0.22, "share_j ~ w_j (2 pi sig_j^2)^(d/2) exp(beta depth_j);\n"
            "same geometry, but amplitudes RE-FIT to Boltzmann mass",
            transform=ax.transAxes, ha="center", fontsize=9, color="#374151")

    for ax in axes:
        ax.set_xlim(-6, 6); ax.set_ylim(-3.3, 0.9)
        ax.set_xlabel("hidden coordinate h (schematic 1-D slice)", fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("V_theta(h)   (learned, ported unchanged)", fontsize=9.5)
    fig.suptitle("Same learned geometry, integrator-dependent calibration",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    out = "portable_potentials_calibration_axes.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    fig_landscape()
    fig_matrix()
    fig_calibration()
