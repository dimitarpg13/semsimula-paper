"""Figures for depth_ladder/Velocity_and_Pair_Force_as_Competing_Mechanisms.md.

Generates four PNGs into figures/velocity_pair_force/:

  vp_carrier.png       -- the velocity as a carrier: the two-layer stack on a
                          toy two-dimensional landscape, with the velocity
                          carried across the layer boundary and with it reset.
                          The fields are invented; the update rules are the
                          note's equations (soft directions, gamma*dt = 0.4).
  vp_weights_reach.png -- (a) the weight of layer 0's force in the output and
                          the look-ahead distance, against gamma*dt;
                          (b) how far a carried velocity moves a stiff mode,
                          sin(omega dt)/(omega dt), with the measured omega*dt
                          ranges of SR2 and SR2-NP (Cell 6b-13).
  vp_measurements.png  -- the measurements: the pair force's reliance against
                          its value; the velocity reset (Gate 1) in both
                          models; the velocity cut of the recall pathway probe
                          by class (its JSON outputs).
  vp_substitution.png  -- what stands in for what: the four corners of
                          velocity x pair force (settled perplexities, the
                          missing corner with the additive prediction), and the
                          part of the book's mechanism graph around them.

The measured numbers are read from the run outputs named beside each one.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

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
OUT = HERE / "velocity_pair_force"
RES = HERE.parent.parent / "notebooks/conservative_arch/scaleup/results"

C = float(np.exp(-0.4))        # friction factor per layer at gamma*dt = 0.4 (L = 2 ladder)


def _spines(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# ---------------------------------------------------------------------------
# Figure 1 -- the velocity as a carrier (toy landscape, real update rule)
# ---------------------------------------------------------------------------
def _wells(h, wells):
    """V(h) = -sum A exp(-|h - mu|^2 / 2 s^2) and its force -grad V."""
    V = np.zeros(h.shape[:-1])
    F = np.zeros_like(h)
    for A, mu, s in wells:
        d = h - np.asarray(mu)
        g = A * np.exp(-(d ** 2).sum(-1) / (2 * s * s))
        V -= g
        F -= (g / (s * s))[..., None] * d
    return V, F


def fig_carrier():
    W0 = [(5.0, (2.2, 1.2), 1.5)]                              # layer 0's landscape
    W1 = [(2.5, (1.0, 1.9), 0.7), (3.5, (3.1, 0.2), 0.8)]      # layer 1's landscape
    kappa = 1.0                                                # c dt^2 / 2m, absorbed into the depths
    h0 = np.array([0.0, 0.0])
    _, F0 = _wells(h0, W0)
    delta = kappa * F0
    h1 = h0 + delta
    look = h0 + 2 * delta                                      # where layer 1 samples, velocity carried
    _, F1c = _wells(look, W1)
    _, F1r = _wells(h1, W1)
    out_c = h0 + (2 + C) * delta + kappa * F1c
    out_r = h0 + delta + kappa * F1r

    xs, ys = np.meshgrid(np.linspace(-0.6, 4.6, 260), np.linspace(-1.0, 3.0, 200))
    grid = np.stack([xs, ys], -1)
    V1, _ = _wells(grid, W1)
    V0, _ = _wells(grid, W0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    for ax, carried in zip(axes, (True, False)):
        ax.contourf(xs, ys, V1, levels=14, cmap="Greys_r", alpha=0.35)
        ax.contour(xs, ys, V0, levels=6, colors=[PURPLE], linewidths=0.6, alpha=0.5)
        ax.set_aspect("equal")
        ax.set_xlim(-0.6, 4.6)
        ax.set_ylim(-1.0, 3.0)
        ax.set_xticks([])
        ax.set_yticks([])

        def arrow(p, q, col, lw=2.0, ls="-"):
            ax.add_patch(FancyArrowPatch(tuple(p), tuple(q), arrowstyle="-|>",
                                         mutation_scale=14, color=col, linewidth=lw,
                                         linestyle=ls, shrinkA=0, shrinkB=0, zorder=4))

        def dot(p, col, label, dx=0.08, dy=0.08, ha="left"):
            ax.plot(*p, "o", color=col, markersize=7, markeredgecolor="white", zorder=5)
            ax.text(p[0] + dx, p[1] + dy, label, color=INK, fontsize=10, ha=ha, zorder=6)

        arrow(h0, h1, PURPLE)
        dot(h0, INK, r"$h^{(0)}$", dx=-0.1, dy=-0.25, ha="right")
        dot(h1, PURPLE, r"$h^{(1)} = h^{(0)} + \delta$", dx=-0.05, dy=0.16, ha="right")
        if carried:
            arrow(h1, look, BLUE, ls=(0, (4, 2)))
            dot(look, BLUE, "look-ahead\n" + r"$h^{(0)} + 2\delta$", dx=-0.15, dy=0.2, ha="right")
            arrow(look, out_c, GREEN)
            ax.text(*(0.5 * (look + out_c) + np.array([0.1, 0.0])),
                    "second half-drift\n" + r"$c\delta + \kappa F^{(1)}(h^{(0)} + 2\delta)$", color=INK, fontsize=9.5)
            dot(out_c, RED, r"$h^{(2)}$, velocity carried", dx=0.1, dy=0.1)
            ax.plot([h0[0], out_c[0]], [h0[1], out_c[1]], color=RED, lw=0.8, ls=":", zorder=3)
            ax.set_title("velocity carried across the boundary")
            ax.text(-0.5, 2.75, r"$h^{(2)} = h^{(0)} + (2+c)\delta + \kappa F^{(1)}(h^{(0)} + 2\delta)$",
                    fontsize=10.5, color=INK)
        else:
            arrow(h1, out_r, GREEN)
            ax.text(*(0.5 * (h1 + out_r) + np.array([0.12, -0.05])),
                    "drift\n" + r"$\kappa F^{(1)}(h^{(0)} + \delta)$", color=INK, fontsize=9.5)
            dot(out_r, RED, r"$h^{(2)}$, velocity reset", dx=0.1, dy=0.1)
            ax.plot([h0[0], out_r[0]], [h0[1], out_r[1]], color=RED, lw=0.8, ls=":", zorder=3)
            ax.plot(*out_c, "o", color=RED, alpha=0.3, markersize=7, zorder=4)
            ax.text(out_c[0], out_c[1] - 0.3, "(carried, for comparison)", color=MUTED, fontsize=9, ha="center")
            ax.set_title("velocity reset at the boundary (Gate 1)")
            ax.text(-0.5, 2.75, r"$h^{(2)} = h^{(0)} + \delta + \kappa F^{(1)}(h^{(0)} + \delta)$",
                    fontsize=10.5, color=INK)
    handles = [Line2D([], [], color=PURPLE, lw=1, label="layer 0's landscape (contours)"),
               plt.Rectangle((0, 0), 1, 1, color="grey", alpha=0.35, label="layer 1's landscape (shaded, dark = low)"),
               Line2D([], [], color=PURPLE, lw=2, label=r"layer 0's step $\delta = \kappa F^{(0)}(h^{(0)})$"),
               Line2D([], [], color=BLUE, lw=2, ls="--", label=r"carried velocity, $\frac{\Delta t}{2} v^{(1)} = \delta$"),
               Line2D([], [], color=GREEN, lw=2, label="layer 1's second half-drift, after its kick")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / "vp_carrier.png", bbox_inches="tight")
    plt.close(fig)
    return dict(delta=delta, out_c=out_c, out_r=out_r)


# ---------------------------------------------------------------------------
# Figure 2 -- weights of the carried force, and the velocity's reach into stiff modes
# ---------------------------------------------------------------------------
OMEGA_DT = {  # Cell 6b-13, best checkpoints, 16,384 tokens: (p05, p50, p95) per layer
    "SR2": {0: (7.820, 8.927, 10.148), 1: (3.918, 5.459, 7.021)},
    "SR2-NP": {0: (6.560, 7.292, 7.982), 1: (3.078, 4.307, 5.488)},
}


def fig_weights_reach():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1, 1.35]})

    ax = axes[0]
    x = np.linspace(0, 1.5, 200)
    ax.plot(x, 2 + np.exp(-x), color=BLUE, lw=2.2, label="layer 0's force, velocity carried: $2 + c$")
    ax.plot(x, np.ones_like(x), color=GREY, lw=2.2, label="layer 0's force, velocity reset: 1")
    ax.plot(x, 2 * np.ones_like(x), color=BLUE, lw=1.4, ls="--", label=r"look-ahead distance, carried: $2\delta$ (reset: $\delta$)")
    ax.axvline(0.4, color=MUTED, lw=0.8, ls=":")
    ax.text(0.42, 2.78, r"ladder, $L = 2$: $\gamma\Delta t = 0.4$, $c = 0.670$", color=INK, fontsize=9.5)
    ax.plot([0.4], [2 + C], "o", color=BLUE)
    ax.set_xlabel(r"friction per layer, $\gamma\Delta t$")
    ax.set_ylabel("weight in the output")
    ax.set_ylim(0, 3.2)
    ax.set_title("(a) what the carried velocity re-weights", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    _spines(ax)

    pos = axes[1].get_position()
    axes[1].remove()
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.35], height_ratios=[1, 2.6], hspace=0.08)
    top = fig.add_subplot(gs[0, 1])
    ax = fig.add_subplot(gs[1, 1], sharex=top)
    rows = [("SR2", 0, BLUE, 3), ("SR2-NP", 0, ORANGE, 2), ("SR2", 1, BLUE, 1), ("SR2-NP", 1, ORANGE, 0)]
    for name, layer, col, y in rows:
        lo, med, hi = OMEGA_DT[name][layer]
        top.plot([lo, hi], [y, y], color=col, lw=6, alpha=0.6, solid_capstyle="butt")
        top.plot([med], [y], "o", color=col, markeredgecolor="white", markersize=8)
        top.text(hi + 0.2, y, f"{name}, layer {layer}: median {med:.2f}", va="center", color=INK, fontsize=9)
    top.set_ylim(-0.7, 3.7)
    top.set_yticks([])
    top.tick_params(labelbottom=False)
    for side in ("top", "right", "left"):
        top.spines[side].set_visible(False)
    top.set_title("(b) how far a carried velocity moves a stiff mode", fontsize=11)
    top.text(0.1, 3.55, "measured $\\omega\\Delta t$, 5th to 95th percentile over tokens (Cell 6b-13)",
             color=MUTED, fontsize=8.5, va="center")
    w = np.linspace(0.05, 15.5, 1000)
    ax.plot(w, np.sin(w) / w, color=INK, lw=1.8, label=r"$r(\omega\Delta t) = \sin(\omega\Delta t) / (\omega\Delta t)$")
    ax.plot(w, 1 / w, color=MUTED, lw=1, ls="--", label=r"envelope $\pm 1/(\omega\Delta t)$")
    ax.plot(w, -1 / w, color=MUTED, lw=1, ls="--")
    ax.axhline(0, color=GREY, lw=0.6)
    ax.set_xlim(0, 15.5)
    ax.set_ylim(-0.35, 1.05)
    ax.set_xlabel(r"stiffest mode, $\omega\Delta t$")
    ax.set_ylabel("displacement by a carried velocity,\nrelative to a soft direction")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _spines(ax)
    fig.savefig(OUT / "vp_weights_reach.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 -- the measurements
# ---------------------------------------------------------------------------
def _gate1(folder):
    path = next((RES / folder).glob("Cell-6b-7*"))
    for line in open(path):
        if line.strip().startswith("GATE 1"):
            p = line.split()
            return float(p[3]), float(p[5])                 # carried, reset
    raise RuntimeError(path)


SR2_DIR = ("cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_sr2_"
           "L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn")
NP_DIR = SR2_DIR.replace("_vplive_", "_nophi_")


def fig_measurements():
    rp = {k: json.load(open(RES / f"recall_pathway/recall_pathway_{k}_owt.json"))["arms"] for k in ("sr2", "sr2np")}
    g_s = _gate1(SR2_DIR)
    g_n = _gate1(NP_DIR)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), gridspec_kw={"width_ratios": [1.2, 0.8, 1.2]})

    ax = axes[0]
    vals = [100 * (59.36 / 58.28 - 1), 100 * (58.41 / 58.28 - 1), 100 * (58.51 / 58.32 - 1)]
    labels = ["removed from\ntrained SR2\n(16 windows)", "trained without\n(SR2-NP,\nsame 16 windows)", "trained without\n(SR2-NP,\nsettled)"]
    cols = [GREY, ORANGE, ORANGE]
    ax.bar(range(3), vals, color=cols, width=0.6, edgecolor="white")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.05, f"+{v:.2f}%", ha="center", color=INK, fontsize=10)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("perplexity against SR2")
    ax.set_ylim(0, 2.3)
    ax.set_title("(a) the pair force: reliance against value", fontsize=11)
    _spines(ax)

    ax = axes[1]
    gv = [100 * (g_s[1] / g_s[0] - 1), 100 * (g_n[1] / g_n[0] - 1)]
    ax.bar([0, 1], gv, color=[BLUE, ORANGE], width=0.55, edgecolor="white")
    for i, (v, g) in enumerate(zip(gv, (g_s, g_n))):
        ax.text(i, v + 0.12, f"+{v:.1f}%\n({g[0]:.2f} to {g[1]:.2f})", ha="center", color=INK, fontsize=9.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["SR2", "SR2-NP"])
    ax.set_ylabel("perplexity cost of the reset")
    ax.set_ylim(0, 7.5)
    ax.set_title("(b) velocity reset (Gate 1)", fontsize=11)
    _spines(ax)

    ax = axes[2]
    classes = ("near", "far", "control")
    for k, (key, col, mk, off) in enumerate((("sr2", BLUE, "o", -0.12), ("sr2np", ORANGE, "s", 0.12))):
        for i, cls in enumerate(classes):
            m, se = rp[key]["vel@1"][cls]
            ax.errorbar(i + off, m, yerr=se, fmt=mk, color=col, markersize=8, capsize=3,
                        markeredgecolor="white", elinewidth=1.6)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["near recall", "far recall", "control\n(recalls nothing)"])
    ax.set_ylabel("loss change, nats per token")
    ax.set_title("(c) velocity into layer 1 zeroed, by class", fontsize=11)
    ax.legend(handles=[Line2D([], [], marker="o", color=BLUE, ls="none", label="SR2"),
                       Line2D([], [], marker="s", color=ORANGE, ls="none", label="SR2-NP")],
              frameon=False, fontsize=9.5, loc="upper left")
    ax.grid(axis="y", color="#E5E7EB", lw=0.7)
    ax.set_axisbelow(True)
    _spines(ax)
    fig.tight_layout()
    fig.savefig(OUT / "vp_measurements.png", bbox_inches="tight")
    plt.close(fig)
    return gv


# ---------------------------------------------------------------------------
# Figure 4 -- what stands in for what
# ---------------------------------------------------------------------------
def fig_substitution():
    base = 58.32
    d_v = np.log(61.16 / base)
    d_p = np.log(58.51 / base)
    additive = base * np.exp(d_v + d_p)

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2), gridspec_kw={"width_ratios": [1, 1.25]})

    ax = axes[0]
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2.25)
    ax.axis("off")
    cells = {
        (0, 1): ("SR2", "velocity + pair force", "58.32", "the base", BLUE),
        (1, 1): ("SR2-NP", "velocity, no pair force", "58.51", r"$+0.3\%$", ORANGE),
        (0, 0): ("FO-b-OWT", "pair force, no velocity", "61.16", r"$+4.9\%$", PURPLE),
        (1, 0): ("not run", "neither", "?", f"additive: {additive:.2f}\nsubstitution: above", GREY),
    }
    for (i, j), (name, what, ppl, note, col) in cells.items():
        x0, y0 = i + 0.04, j + 0.04
        ls = "--" if name == "not run" else "-"
        ax.add_patch(FancyBboxPatch((x0, y0), 0.92, 0.92, boxstyle="round,pad=0.0,rounding_size=0.04",
                                    facecolor=col + "22", edgecolor=col, linewidth=1.6, linestyle=ls))
        ax.text(x0 + 0.46, y0 + 0.74, name, ha="center", fontsize=12, fontweight="bold", color=INK)
        ax.text(x0 + 0.46, y0 + 0.6, what, ha="center", fontsize=9.5, color=MUTED)
        ax.text(x0 + 0.46, y0 + 0.36, ppl, ha="center", fontsize=17, color=INK)
        ax.text(x0 + 0.46, y0 + 0.12, note, ha="center", fontsize=9.5, color=INK)
    ax.text(0.5, 2.06, "with the pair force", ha="center", color=INK, fontsize=10.5)
    ax.text(1.5, 2.06, "without the pair force", ha="center", color=INK, fontsize=10.5)
    ax.text(-0.04, 1.5, "second order\n(velocity carried)", ha="right", va="center", color=INK, fontsize=10.5)
    ax.text(-0.04, 0.5, "first order\n(no velocity)", ha="right", va="center", color=INK, fontsize=10.5)
    ax.set_title("(a) the four corners: settled perplexity, OpenWebText, 0.53B tokens", fontsize=11)
    ax.text(1.0, -0.06, "the missing corner: additive if the two only overlap, above it if each stands in for the other",
            ha="center", color=MUTED, fontsize=8.5)

    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    HW, HH = 1.35, 0.55
    nodes = {
        "xi": (5.0, 7.3, "context channels $\\xi$\n(backbone: carries recall)"),
        "phi": (1.5, 4.2, "pair force $V_{\\phi}$\n(routing family)"),
        "vel": (8.5, 4.2, "velocity\n(capacity family)"),
        "exact": (1.5, 0.8, "exact damped flow\n(integrator)"),
        "stiff": (5.0, 0.8, "a stiffer $V_{\\theta}$\n(capacity family)"),
        "wells": (8.5, 0.8, "Poisson-mode wells\n(capacity family)"),
    }
    for key, (x, y, label) in nodes.items():
        ax.add_patch(FancyBboxPatch((x - HW, y - HH), 2 * HW, 2 * HH, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor="white", edgecolor=INK, linewidth=1.2))
        ax.text(x, y, label, ha="center", va="center", fontsize=9.5, color=INK)

    def arr(p, q, col, ls="-", lw=1.6):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, color=col,
                                     linestyle=ls, linewidth=lw, shrinkA=0, shrinkB=0))

    def lab(x, y, text, ha="center"):
        ax.text(x, y, text, ha=ha, va="center", fontsize=8.6, color=INK)

    dash = (0, (3, 2))
    xr, xl = 1.5 + HW, 8.5 - HW
    arr((xr, 4.45), (xl, 4.45), PURPLE, dash)
    lab(5.0, 4.85, "pair force stands in for the velocity, partly\n(first-order twins rely on it 1.2 to 2.1 times as much)")
    arr((xl, 3.95), (xr, 3.95), ORANGE, dash)
    lab(5.0, 3.5, "velocity stands in for the pair force, partly (SR2-NP)\nreset +3.9% to +5.6%; zeroed +0.064 to +0.111 nats")
    arr((8.2, 0.8 + HH), (8.2, 4.2 - HH), GREEN, dash)
    lab(8.1, 2.5, "replace it\nfully", ha="right")
    arr((8.8, 4.2 - HH), (8.8, 0.8 + HH), RED, (0, (1, 2)))
    lab(8.9, 2.5, "cannot\nreplace them", ha="left")
    arr((5.6, 0.8 + HH), (7.4, 4.2 - HH), GREY, dash)
    lab(6.2, 2.5, "partly")
    arr((1.5, 0.8 + HH), (1.5, 4.2 - HH), INK, "-")
    lab(1.6, 2.5, "idles it\n(reliance 1.4 to 1.8%)", ha="left")
    arr((4.4, 7.3 - HH), (2.0, 4.2 + HH), INK, "-", 2.4)
    arr((5.6, 7.3 - HH), (8.0, 4.2 + HH), INK, "-", 2.4)
    ax.set_title("(b) the edges around the velocity and the pair force", fontsize=11)
    handles = [Line2D([], [], color=INK, lw=1.6, label="A crowds out B (dominance)"),
               Line2D([], [], color=INK, lw=1.6, ls=(0, (3, 2)), label="A stands in for B (substitution)"),
               Line2D([], [], color=INK, lw=2.4, label="backbone feeds")]
    ax.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    fig.savefig(OUT / "vp_substitution.png", bbox_inches="tight")
    plt.close(fig)
    return additive


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    geo = fig_carrier()
    fig_weights_reach()
    gv = fig_measurements()
    add = fig_substitution()
    print("toy:", {k: np.round(v, 3).tolist() for k, v in geo.items()})
    print("gate1 %:", [round(v, 2) for v in gv], " additive corner:", round(add, 2))
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
