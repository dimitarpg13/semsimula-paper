"""Figures for depth_ladder/Recall_Pathway_Probe.md.

Generates four PNGs into figures/recall_pathway/:

  rp_classes.png  -- one window of tokens with a near recall pair, a far
                     recall pair, a control position and the left-out case
                     p(q) = q (section 2 of the note).
  rp_cuts.png     -- where each cut acts: xi@l removes h_p from the channel
                     read at q, phi@l removes p from q's allowed sources,
                     vel@l zeroes the velocity at every position (section 3).
  rp_results.png  -- the readings of all four probed models, near / far /
                     control, one standard error over windows (section 6).
  rp_paired.png   -- SR2-NP minus SR2 on the same 16 windows, per window,
                     under vel@1 and xi@1 (section 6).

The schematics use an invented sentence; the two data figures read the
probe's own JSON outputs in notebooks/.../results/recall_pathway/ and
nothing else.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

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
OUT = HERE / "recall_pathway"
RES = HERE.parent.parent / "notebooks/conservative_arch/scaleup/results/recall_pathway"

CLASSES = ("near", "far", "control")


def _load(stem):
    with open(RES / f"recall_pathway_{stem}.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Figure 1 -- recall and control positions in one window
# ---------------------------------------------------------------------------
def fig_classes():
    # (position, token); positions 12..39 are not drawn
    toks = [(0, "Alice"), (1, "met"), (2, "Bob"), (3, "at"), (4, "noon"),
            (5, "."), (6, "Then"), (7, "Bob"), (8, "was"), (9, "very"),
            (10, "very"), (11, "late"), (40, "said"), (41, "Alice"), (42, ".")]
    xof = {s: (s if s < 12 else s - 27) for s, _ in toks}   # 40 -> 13
    W, H = 0.86, 0.62

    fig, ax = plt.subplots(figsize=(13, 4.4))
    ax.set_xlim(-0.8, 15.8)
    ax.set_ylim(-2.35, 2.55)
    ax.axis("off")

    style = {0: ("p", BLUE), 2: ("p", GREEN), 6: ("q", GREEN), 7: ("t", GREEN),
             40: ("q", BLUE), 41: ("t", BLUE), 1: ("q", GREY), 9: ("q", PURPLE),
             10: ("t", PURPLE)}
    for s, tok in toks:
        x = xof[s]
        role, col = style.get(s, (None, None))
        face, edge, ls, lw = "white", "#D1D5DB", "-", 1.0
        if role == "p":
            face, edge, lw = col + "33", col, 1.6
        elif role == "q":
            edge, lw = col, 2.2
        elif role == "t":
            edge, ls, lw = col, (0, (3, 2)), 1.6
        ax.add_patch(Rectangle((x - W / 2, -H / 2), W, H, facecolor=face,
                               edgecolor=edge, linestyle=ls, linewidth=lw))
        ax.text(x, 0, tok, ha="center", va="center", color=INK, fontsize=10)
        ax.text(x, -H / 2 - 0.2, str(s), ha="center", va="center",
                color=MUTED, fontsize=8.5)
    ax.text(12, 0, r"$\cdots$", ha="center", va="center", color=MUTED, fontsize=14)
    ax.text(12, -H / 2 - 0.2, "12 to 39", ha="center", va="center",
            color=MUTED, fontsize=8)

    def arc(x0, x1, col, h, label):
        a = FancyArrowPatch((x0, H / 2 + 0.04), (x1, H / 2 + 0.04),
                            connectionstyle=f"arc3,rad=-{h}", arrowstyle="-|>",
                            mutation_scale=13, color=col, linewidth=1.8)
        ax.add_patch(a)
        ax.text((x0 + x1) / 2, H / 2 + 0.2 + h * (x1 - x0) / 2, label,
                ha="center", va="bottom", color=INK, fontsize=9.5)

    arc(xof[2], xof[6], GREEN, 0.35, r"near: $\delta = 7 - 2 = 5$")
    arc(xof[0], xof[40], BLUE, 0.2, r"far: $\delta = 41 - 0 = 41$")

    notes = [
        (1, GREY, "control, q = 1:\ntarget Bob has\nno earlier copy"),
        (6, GREEN, "recall, q = 6:\ntarget Bob, p(6) = 2"),
        (9.5, PURPLE, "left out, q = 9:\ntarget very = x₉,\nso p(9) = 9"),
        (13, BLUE, "recall, q = 40:\ntarget Alice, p(40) = 0"),
    ]
    for x, col, txt in notes:
        ax.plot([x], [-1.05], marker="s", markersize=8, color=col)
        ax.text(x, -1.25, txt, ha="center", va="top", color=INK, fontsize=9)

    handles = [
        Rectangle((0, 0), 1, 1, facecolor=GREY + "00", edgecolor=INK, linewidth=2.0),
        Rectangle((0, 0), 1, 1, facecolor="#E5E7EB", edgecolor=INK, linewidth=1.4),
        Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=INK,
                  linestyle=(0, (3, 2)), linewidth=1.4),
    ]
    ax.legend(handles, ["position q (predicts the next token)",
                        "earlier occurrence p(q)",
                        r"target $y_q = x_{q+1}$"],
              loc="upper left", frameon=False, fontsize=9, ncol=3,
              bbox_to_anchor=(0.0, 1.04))
    fig.tight_layout()
    fig.savefig(OUT / "rp_classes.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 -- where each cut acts
# ---------------------------------------------------------------------------
def _cross(ax, x, y, size=0.13):
    ax.plot([x - size, x + size], [y - size, y + size], color=RED, linewidth=2.4)
    ax.plot([x - size, x + size], [y + size, y - size], color=RED, linewidth=2.4)


def _arrow(ax, p0, p1, col=INK, rad=0.0, ls="-", lw=1.4, alpha=1.0):
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}",
                                 arrowstyle="-|>", mutation_scale=11, color=col,
                                 linestyle=ls, linewidth=lw, alpha=alpha,
                                 shrinkA=7, shrinkB=7))


def _hop(ax, x0, x1, h, col, ls="-", lw=1.4):
    """Arc of height h from (x0, 0) to (x1, 0), arrowhead at x1."""
    u = np.linspace(0, 1, 60)
    xs, ys = x0 + (x1 - x0) * u, 4 * h * u * (1 - u)
    ax.plot(xs[:-6], ys[:-6], color=col, linestyle=ls, linewidth=lw)
    ax.add_patch(FancyArrowPatch((xs[-7], ys[-7]), (xs[-2], ys[-2]),
                                 arrowstyle="-|>", mutation_scale=11, color=col,
                                 linewidth=lw, shrinkA=0, shrinkB=4))


def fig_cuts():
    xs = np.arange(6)
    labels = [r"$s_1$", r"$p$", r"$s_2$", r"$s_3$", r"$c$", r"$q$"]
    P, C, Q = 1, 4, 5

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for ax in axes:
        ax.set_xlim(-0.6, 5.6)
        ax.set_ylim(-1.05, 2.2)
        ax.axis("off")

    def states(ax, y, name, xs_=xs):
        for x in xs_:
            ax.plot([x], [y], "o", markersize=13, markerfacecolor="white",
                    markeredgecolor=INK, markeredgewidth=1.3, zorder=3)
        ax.text(-0.55, y, name, ha="right", va="center", color=INK, fontsize=10)

    def poslabels(ax, y):
        for x, lab in zip(xs, labels):
            ax.text(x, y, lab, ha="center", va="top", color=MUTED, fontsize=11)

    # (a) xi: every s <= q feeds xi_q; the link from p is removed
    ax = axes[0]
    states(ax, 0, r"$h^{(\ell^\ast)}$")
    ax.plot([Q], [1.45], "s", markersize=13, markerfacecolor=BLUE + "33",
            markeredgecolor=BLUE, markeredgewidth=1.4, zorder=3)
    ax.text(Q + 0.25, 1.45, r"$\xi_{k,q}$", ha="left", va="center", color=INK)
    for x in xs:
        if x == P:
            _arrow(ax, (x, 0), (Q, 1.45), col=GREY, ls=(0, (3, 2)))
            _cross(ax, x + (Q - x) * 0.42, 1.45 * 0.42)
        else:
            _arrow(ax, (x, 0), (Q, 1.45), col=BLUE, lw=1.2)
    poslabels(ax, -0.3)
    ax.set_title(r"xi@$\ell^\ast$: logit of $h_p$ in $\xi_{k,q}$ set to $-\infty$",
                 fontsize=11)
    ax.text(2.5, -0.75, "the softmax renormalises over the other sources",
            ha="center", va="top", color=MUTED, fontsize=9)

    # (b) phi: q's sources come from the causal mask; p is taken out of it
    ax = axes[1]
    states(ax, 0, r"$h^{(\ell^\ast)}$")
    _hop(ax, 0, Q, 1.45, PURPLE)
    _hop(ax, P, Q, 1.1, GREY, ls=(0, (3, 2)))
    _cross(ax, (P + Q) / 2, 1.1)
    _hop(ax, 2, Q, 0.8, PURPLE, ls=(0, (1, 1.5)), lw=1.6)
    _hop(ax, 3, Q, 0.5, PURPLE)
    poslabels(ax, -0.3)
    ax.set_title(r"phi@$\ell^\ast$: $M_{qp} \to$ False", fontsize=11)
    ax.text(2.5, -0.75, "the score head picks its 16 sources among the others\n"
            "(dotted: a source picked instead; no pair force in SR2-NP)",
            ha="center", va="top", color=MUTED, fontsize=9)

    # (c) vel: the decoded velocity is zero at every position
    ax = axes[2]
    states(ax, 0, r"$h_{\mathrm{prev}}$")
    states(ax, 1.3, r"$h$")
    for x in xs:
        _arrow(ax, (x, 0), (x, 1.3), col=ORANGE, lw=1.4)
        _cross(ax, x, 0.65, size=0.11)
    poslabels(ax, -0.3)
    ax.text(5.5, 0.65, r"$v_t$", ha="left", va="center", color=INK)
    ax.set_title(r"vel@$\ell^\ast$: $h_{\mathrm{prev}} = h$, so $v_t = 0$",
                 fontsize=11)
    ax.text(2.5, -0.75, "at every position, not only along recall pairs",
            ha="center", va="top", color=MUTED, fontsize=9)

    fig.text(0.5, 0.005, r"$p$: earlier occurrence of the target of $q$;  "
             r"$c$: a position that recalls nothing;  $s_i$: other earlier positions.  "
             r"Each cut acts at one layer $\ell^\ast$, on every recall pair of the window.",
             ha="center", va="bottom", color=INK, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "rp_cuts.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 -- the readings of the four models
# ---------------------------------------------------------------------------
CUTS = [("xi@1", "xi@1"), ("phi@1", "phi@1"), ("xiphi@0", "xiphi@0"),
        ("vel@1", "vel@1")]


def fig_results():
    rows = [
        ("Code", [("SO-CODE", "so_code", BLUE, "o"),
                  ("FO-b-CODE", "fob_code", ORANGE, "s")]),
        ("OpenWebText", [("SR2", "sr2_owt", BLUE, "o"),
                         ("SR2-NP", "sr2np_owt", ORANGE, "s")]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), sharey=True)
    x = np.arange(len(CUTS))
    for r, (corpus, models) in enumerate(rows):
        for m, (name, stem, col, mk) in enumerate(models):
            arms = _load(stem)["arms"]
            off = -0.13 if m == 0 else 0.13
            for c, cls in enumerate(CLASSES):
                ax = axes[r, c]
                for i, (cut, _) in enumerate(CUTS):
                    key = cut
                    if key not in arms and cut == "xiphi@0":
                        key = "xi@0"                 # SR2-NP has no pair force
                    if key not in arms:
                        continue                     # FO-b: no vel; SR2-NP: no phi
                    mean, se = arms[key][cls]
                    ax.errorbar(x[i] + off, mean, yerr=se, fmt=mk, color=col,
                                markersize=8, markeredgecolor="white",
                                markeredgewidth=1.0, elinewidth=1.6, capsize=3,
                                zorder=3)
        handles = [Line2D([], [], marker=mk, color=col, linestyle="none",
                          markersize=8, label=name)
                   for name, _, col, mk in models]
        axes[r, 2].legend(handles=handles, loc="upper right", frameon=False,
                          fontsize=10, title=corpus, title_fontsize=10)
        axes[r, 0].set_ylabel(f"{corpus}\nchange of loss (nats per token)")

    for r in range(2):
        for c, cls in enumerate(CLASSES):
            ax = axes[r, c]
            ax.axhline(0, color=MUTED, linewidth=0.8, zorder=1)
            ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.set_xticks(x)
            ax.set_xlim(-0.5, len(CUTS) - 0.5)
            if r == 1:
                ax.set_xticklabels(["xi@1", "phi@1", "xiphi@0\n(SR2-NP: xi@0)",
                                    "vel@1"], fontsize=9.5)
            else:
                ax.set_xticklabels(["xi@1", "phi@1", "xiphi@0", "vel@1"],
                                   fontsize=9.5)
            if r == 0:
                ax.set_title({"near": r"near recall ($\delta \leq 16$)",
                              "far": r"far recall ($17 \leq \delta \leq 256$)",
                              "control": "control positions"}[cls])
    fig.text(0.5, 0.005, "Mean over 16 windows, bars one standard error over "
             "windows. Missing points: FO-b-CODE has no velocity, SR2-NP has no "
             "pair force.", ha="center", va="bottom", color=MUTED, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "rp_results.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 -- SR2-NP minus SR2, window by window
# ---------------------------------------------------------------------------
def fig_paired():
    a = _load("sr2_owt")["arms"]
    b = _load("sr2np_owt")["arms"]
    panels = [("vel@1", CLASSES, "velocity zeroed at layer 1"),
              ("xi@1", ("near", "far"), "earlier occurrence removed from the layer-1 read")]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True,
                             gridspec_kw={"width_ratios": [3, 2]})
    for ax, (cut, classes, title) in zip(axes, panels):
        for i, cls in enumerate(classes):
            d = (np.array(b[cut]["per_window"][cls])
                 - np.array(a[cut]["per_window"][cls]))
            n = len(d)
            mean, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
            jit = rng.uniform(-0.12, 0.12, n)
            ax.scatter(i + jit, d, s=26, color=GREY, edgecolor="white",
                       linewidth=0.6, zorder=2)
            ax.errorbar(i + 0.3, mean, yerr=se, fmt="D", color=INK,
                        markersize=7, elinewidth=1.8, capsize=4, zorder=3)
            ax.text(i + 0.42, mean, f"{mean:+.3f}".replace("-", "−")
                    + f"\n(se {se:.3f})\n{(d > 0).sum()}/{n} > 0",
                    ha="left", va="center", color=INK, fontsize=9)
        ax.axhline(0, color=MUTED, linewidth=0.8, zorder=1)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_xlim(-0.5, len(classes) - 0.05)
        ax.set_title(f"{cut}: {title}", fontsize=11)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("SR2-NP minus SR2\n(nats per token, per window)")
    handles = [Line2D([], [], marker="o", color=GREY, linestyle="none",
                      markersize=6, label="one window"),
               Line2D([], [], marker="D", color=INK, linestyle="none",
                      markersize=6, label="mean, one standard error")]
    axes[1].legend(handles=handles, loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "rp_paired.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig_classes()
    fig_cuts()
    fig_results()
    fig_paired()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
