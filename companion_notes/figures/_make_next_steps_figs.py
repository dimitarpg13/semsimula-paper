"""Figures for Fock-PARFLM_vs_GPT-2_on_OpenWebText_Next_Steps.md.

Generates three PNGs into the same folder:

  next_steps_capacity_asymmetry.png   -- where the params/compute go, and the
                                         distinct-function + context-mixing gap
  next_steps_force_shapes.png         -- radial vs learned-direction (value
                                         transport) vs multi-head force fields,
                                         all conservative (force = -grad V)
  next_steps_multicontext_heads.png   -- current concat-then-one-head vs
                                         per-context well-bank heads (param
                                         neutral), schematic

All force fields are computed as F = -grad(V) on a grid so the rendered arrows
are conservative by construction -- that is the whole point of the framework.
"""

import numpy as np
import matplotlib.pyplot as plt
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
DARK = "#111827"
AMBER = "#F59E0B"


# ---------------------------------------------------------------------------
# Figure 1 -- capacity asymmetry
# ---------------------------------------------------------------------------
def fig_capacity():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # Panel A: non-embedding parameter allocation (millions).
    ax = axes[0]
    cats = ["Context\nmixing", "Per-token\ntransform", "Register\noverhead"]
    fock = [0.2, 9.5, 2.3]      # V_phi, V_theta (shared), Fock registers
    gpt = [4.1, 8.3, 0.0]       # attention (7x), FFN (7x), none
    x = np.arange(len(cats))
    w = 0.38
    ax.bar(x - w / 2, fock, w, label="Fock-PARFLM", color=GREEN)
    ax.bar(x + w / 2, gpt, w, label="Matched GPT-2", color=BLUE)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("Non-embedding params (M)")
    ax.set_title("(a) Where the ~12.2M non-emb params go")
    ax.legend(frameon=False)
    for xi, (a, b) in enumerate(zip(fock, gpt)):
        ax.text(xi - w / 2, a + 0.15, f"{a:g}", ha="center", fontsize=9)
        ax.text(xi + w / 2, b + 0.15, f"{b:g}", ha="center", fontsize=9)

    # Panel B: distinct functional transformations applied per token.
    ax = axes[1]
    labels = ["Fock-PARFLM", "Matched GPT-2"]
    distinct = [2, 14]   # {V_theta, V_phi} shared vs 7 attn + 7 FFN
    bars = ax.bar(labels, distinct, color=[GREEN, BLUE], width=0.55)
    ax.set_ylabel("Distinct nonlinear transforms")
    ax.set_title("(b) Functional diversity (depth-as-refinement\nvs depth-as-composition)")
    for b, v in zip(bars, distinct):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.2, str(v),
                ha="center", fontweight="bold")
    ax.set_ylim(0, 16)
    ax.annotate("shared V_theta, V_phi\nre-used at all 16 steps",
                xy=(0, 2), xytext=(0.0, 7.5), ha="center", fontsize=8.5,
                color=DARK, arrowprops=dict(arrowstyle="->", color=GREY))

    # Panel C: context-mixing capacity deficit (log scale).
    ax = axes[2]
    labels = ["V_phi\n(1 head, k=8)", "Attention\n(6 heads, all T)"]
    vals = [0.2, 4.1]
    bars = ax.bar(labels, vals, color=[RED, BLUE], width=0.55)
    ax.set_ylabel("Pairwise-interaction params (M)")
    ax.set_title("(c) Context-mixing deficit (~20x)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:g}M",
                ha="center", fontweight="bold")
    ax.set_ylim(0, 4.8)

    fig.suptitle(
        "Capacity asymmetry at equal non-embedding params: the gap is "
        "where the budget is spent, not how much",
        fontsize=12.5, y=1.02,
    )
    fig.tight_layout()
    fig.savefig("next_steps_capacity_asymmetry.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 -- force shapes (all conservative: F = -grad V)
# ---------------------------------------------------------------------------
def _grad_field(V, dx, dy):
    gy, gx = np.gradient(V, dy, dx)
    return -gx, -gy  # F = -grad V


def fig_forces():
    n = 320
    lim = 3.0
    xs = np.linspace(-lim, lim, n)
    ys = np.linspace(-lim, lim, n)
    X, Y = np.meshgrid(xs, ys)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    src = np.array([0.0, 0.0])    # source token h_j at origin
    r2 = (X - src[0]) ** 2 + (Y - src[1]) ** 2
    eps = 0.08

    # (1) Current V_phi: radial 1/r potential -> radial force only.
    V_rad = -1.0 / np.sqrt(r2 + eps ** 2)

    # (2) Bilinear value-transport: V = -g(r) * (a . (h_i - h_j)),
    # a = learned write direction.  Force has a learned-direction component.
    a = np.array([np.cos(np.deg2rad(55)), np.sin(np.deg2rad(55))])
    g = np.exp(-r2 / (2 * 0.9 ** 2))
    proj = a[0] * (X - src[0]) + a[1] * (Y - src[1])
    V_vt = -g * proj

    # (3) Multi-head sum: two value-transport terms, different directions and
    # gates, plus a mild radial term.  Still -grad of a scalar -> conservative.
    a2 = np.array([np.cos(np.deg2rad(-25)), np.sin(np.deg2rad(-25))])
    g2 = np.exp(-((X - 0.7) ** 2 + (Y - 0.4) ** 2) / (2 * 0.8 ** 2))
    proj2 = a2[0] * (X - 0.7) + a2[1] * (Y - 0.4)
    V_multi = 0.6 * V_vt - 0.9 * g2 * proj2 - 0.25 / np.sqrt(r2 + eps ** 2)

    panels = [
        (V_rad, "(a) Current V_phi: radial force only\n"
                r"$F \propto (h_i - h_j)$, scalar magnitude", RED),
        (V_vt, "(b) Bilinear value-transport\n"
               r"$F \supset g(r)\, U^\top W h_j$ (learned direction)", PURPLE),
        (V_multi, "(c) Multi-head sum of scalar potentials\n"
                  r"$V=\sum_m V^{(m)}$  still  $F=-\nabla V$ (conservative)",
         GREEN),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (V, title, col) in zip(axes, panels):
        Fx, Fy = _grad_field(V, dx, dy)
        ax.contour(X, Y, V, levels=14, colors=GREY, linewidths=0.5, alpha=0.7)
        speed = np.sqrt(Fx ** 2 + Fy ** 2)
        ax.streamplot(X, Y, Fx, Fy, color=speed, cmap="viridis",
                      density=1.1, linewidth=1.0, arrowsize=1.0)
        ax.plot(*src, "o", color=col, ms=11, mec="k", mew=0.8,
                zorder=5, label="source $h_j$")
        ax.set_title(title)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.legend(loc="upper right", frameon=True, fontsize=8)

    fig.suptitle(
        "All three force fields are gradients of scalar potentials "
        "(curl-free / conservative). Expressivity grows left -> right "
        "WITHOUT breaking the constraint.",
        fontsize=12, y=1.04,
    )
    fig.tight_layout()
    fig.savefig("next_steps_force_shapes.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 -- multi-context heads (param-neutral restructuring)
# ---------------------------------------------------------------------------
def _box(ax, x, y, w, h, text, fc, ec=DARK, fs=9.5, tc=DARK):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                       linewidth=1.2, edgecolor=ec, facecolor=fc)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, wrap=True)


def _arrow(ax, x1, y1, x2, y2, col=DARK):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                 arrowstyle="-|>", mutation_scale=13,
                 linewidth=1.3, color=col))


def fig_multicontext():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))

    # ---- Left: current design (concat -> one head) ----
    ax = axes[0]
    ax.set_title("(a) Current: 5 contexts concatenated -> ONE well-bank")
    chan_y = np.linspace(0.62, 0.95, 5)
    for i, y in enumerate(chan_y):
        _box(ax, 0.02, y - 0.03, 0.20, 0.06, f"xi_{i}  (horizon h_{i})",
             "#E5F0FF", fs=8.5)
    _box(ax, 0.30, 0.70, 0.16, 0.18, "concat\nxi_d = 5d = 1920", "#FEF3C7", fs=9)
    _box(ax, 0.55, 0.70, 0.18, 0.18, "one mu_proj\nLinear(1920, K*d)\n~5.9M",
         "#FECACA", fs=9)
    _box(ax, 0.80, 0.72, 0.17, 0.14, "K=8 wells\n(one bank)", GREEN, fs=9,
         tc="white")
    for y in chan_y:
        _arrow(ax, 0.22, y, 0.30, 0.79)
    _arrow(ax, 0.46, 0.79, 0.55, 0.79)
    _arrow(ax, 0.73, 0.79, 0.80, 0.79)
    ax.text(0.5, 0.55,
            "Multi-resolution context is flattened before processing:\n"
            "only 1 distinct potential. The temporal structure is lost.",
            ha="center", va="center", fontsize=9.5, color=RED, style="italic")

    # ---- Right: proposed (per-context heads) ----
    ax = axes[1]
    ax.set_title("(b) Proposed: per-context well-bank heads, summed")
    for i, y in enumerate(chan_y):
        _box(ax, 0.02, y - 0.03, 0.18, 0.06, f"xi_{i}", "#E5F0FF", fs=8.5)
        _box(ax, 0.28, y - 0.03, 0.30, 0.06,
             f"head_{i}: Linear(d, K*d) ~1.18M", "#DCFCE7", fs=8)
        _arrow(ax, 0.20, y, 0.28, y)
        _arrow(ax, 0.58, y, 0.70, 0.79)
    _box(ax, 0.70, 0.72, 0.27, 0.14,
         "V = sum_m V^(m)\n(scalar -> conservative)\n5 distinct banks", GREEN,
         fs=9, tc="white")
    ax.text(0.5, 0.50,
            "Same ~5.9M params, but 5 independent context-specialised\n"
            "potentials. Sum of scalars stays conservative & bounded.",
            ha="center", va="center", fontsize=9.5, color="#15803D",
            style="italic")

    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0.42, 1.0)
        ax.axis("off")

    fig.suptitle(
        "Multi-context processing heads: a param-neutral restructuring that "
        "raises functional diversity from 1 to H",
        fontsize=12.5, y=1.0,
    )
    fig.tight_layout()
    fig.savefig("next_steps_multicontext_heads.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_capacity()
    fig_forces()
    fig_multicontext()
    print("wrote: next_steps_capacity_asymmetry.png, "
          "next_steps_force_shapes.png, next_steps_multicontext_heads.png")
