"""Generate illustrations for the Attention Optimality Conjecture companion note.

Produces three PNGs in the same directory as this script:
  pareto_frontier.png        -- attention at the corner of (entropy x score) frontier
  design_space_map.png       -- mechanisms classified by which SCN-constraint they relax
  scn_decision_tree.png      -- decision-tree view of the design choices

Run:
    .venv/bin/python _make_figures.py

Dependencies: numpy, matplotlib.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------------------
# Figure 1 - Pareto frontier in (Routing Entropy, Expected Score) space
# --------------------------------------------------------------------------------------


def make_pareto_frontier() -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    # Sweep softmax temperature beta over [0.05, 20] -> trace the entropy/score curve.
    beta = np.geomspace(0.05, 20.0, 200)
    rng = np.random.default_rng(7)
    s = rng.normal(size=64)
    s = (s - s.mean()) / s.std()

    H_vals = []
    score_vals = []
    for b in beta:
        logits = b * s
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()
        H = -(p * np.log(p + 1e-30)).sum()
        score = (p * s).sum()
        H_vals.append(H)
        score_vals.append(score)
    H_vals = np.array(H_vals)
    score_vals = np.array(score_vals)

    H_max = float(np.log(len(s)))
    score_max = float(s.max())

    # Achievable region (shaded) lies below the Pareto frontier.
    ax.fill_between(
        H_vals,
        score_vals,
        score_vals.min() - 0.5,
        color="#cfe3f3",
        alpha=0.55,
        label="Achievable region (any SCN-admissible mechanism)",
    )
    ax.plot(
        H_vals,
        score_vals,
        color="#1f4e79",
        linewidth=2.6,
        label=r"Pareto frontier: softmax with varying $\beta$",
    )

    # Mark canonical operating points along the frontier.
    def closest_idx(target: float) -> int:
        return int(np.argmin(np.abs(H_vals - target)))

    high_temp_idx = closest_idx(H_max - 0.05)
    canon_idx = closest_idx(H_max - 0.55)
    low_temp_idx = closest_idx(H_max - 2.5)

    points = [
        (high_temp_idx, "High temperature (uniform)", (-95, 18)),
        (canon_idx, "Scaled dot-product attention", (10, -28)),
        (low_temp_idx, "Low temperature (argmax)", (-110, 6)),
    ]
    for idx, label, offset in points:
        ax.plot(H_vals[idx], score_vals[idx], "o", markersize=8, color="#1f4e79")
        ax.annotate(
            label,
            xy=(H_vals[idx], score_vals[idx]),
            xytext=offset,
            textcoords="offset points",
            fontsize=10,
            arrowprops=dict(arrowstyle="-", color="#1f4e79", lw=1.0),
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#1f4e79", lw=0.9),
        )

    # Off-frontier mechanisms (illustrative placement, not measured values).
    off_mechanisms = [
        ("Conservative pairwise (PARFLM)", H_max - 0.40, score_vals[canon_idx] - 1.05, "Violates C, P1"),
        ("Linear / kernel attention", H_max - 0.65, score_vals[canon_idx] - 0.45, "Relaxes softmax shape"),
        ("Sigmoid gating (no normalisation)", H_max - 1.10, score_vals[canon_idx] - 0.65, "Violates N"),
    ]
    for name, x, y, _why in off_mechanisms:
        ax.plot(x, y, marker="X", markersize=10, color="#b9433a", linestyle="none")
        ax.annotate(
            name,
            xy=(x, y),
            xytext=(8, -18),
            textcoords="offset points",
            fontsize=9.5,
            color="#7a1d1d",
            bbox=dict(boxstyle="round,pad=0.2", fc="#fdecea", ec="#b9433a", lw=0.7),
        )

    ax.set_xlabel(
        r"Routing entropy $H(\alpha) = -\sum_j \alpha_{ij} \log \alpha_{ij}$",
        fontsize=12,
    )
    ax.set_ylabel(
        r"Expected score $\mathbb{E}_\alpha[s(h_i, h_j)]$",
        fontsize=12,
    )
    ax.set_title(
        "Pareto frontier of routing under SCN-admissibility\n"
        "Softmax attention traces the boundary as the temperature varies",
        fontsize=13,
    )

    # Annotate frontier endpoints to show the entropy and score bounds.
    ax.axhline(score_max, color="#888", linestyle=":", linewidth=1.0)
    ax.axvline(H_max, color="#888", linestyle=":", linewidth=1.0)
    ax.text(
        H_max - 0.03,
        score_max + 0.05,
        r"$H_{\max} = \log T$",
        fontsize=10,
        ha="right",
        color="#555",
    )
    ax.text(
        H_vals.min() + 0.05,
        score_max + 0.05,
        r"$\max_j s_{ij}$ (greedy selection)",
        fontsize=10,
        color="#555",
    )

    ax.set_xlim(H_vals.min() - 0.15, H_max + 0.25)
    ax.set_ylim(score_vals.min() - 0.55, score_max + 0.45)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", fontsize=9.5, framealpha=0.92)

    fig.tight_layout()
    out = HERE / "pareto_frontier.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------------------------
# Figure 2 - Design-space map: which SCN-constraint each mechanism relaxes
# --------------------------------------------------------------------------------------


def make_design_space_map() -> None:
    fig, ax = plt.subplots(figsize=(13.0, 7.4))

    # Constraint row at the top.
    constraints = [
        ("S\nSmoothness", 1.5, 6.4),
        ("C\nCoupling-content\ndecoupling", 5.0, 6.4),
        ("N\nNormalised budget\n(softmax)", 8.5, 6.4),
        ("Zero auxiliary\nstate", 12.0, 6.4),
    ]

    for name, x, y in constraints:
        ax.add_patch(
            FancyBboxPatch(
                (x - 1.20, y - 0.55),
                2.40,
                1.10,
                boxstyle="round,pad=0.08,rounding_size=0.14",
                facecolor="#dbe8f5",
                edgecolor="#1f4e79",
                linewidth=1.4,
            )
        )
        ax.text(x, y, name, ha="center", va="center", fontsize=10.0, color="#1f4e79")

    ax.text(
        6.75,
        7.55,
        "SCN-admissibility constraints (Conservative Obstruction Theorem)",
        ha="center",
        fontsize=12,
        weight="bold",
        color="#1f4e79",
    )

    # Mechanism row with non-overlapping placements; target_x = constraint centre that is relaxed.
    mechanisms = [
        # (name, x, y_mech, color, target_x, caption)
        ("Conservative pairwise\n(PARFLM, gradient flow)", 1.6, 3.3, "#b9433a", 5.0,
            "Relaxes C: coupling and content\nare tied through one gradient"),
        ("Linear / kernel\nattention", 4.4, 3.3, "#cc8d33", 8.5,
            "Relaxes the softmax shape\n(sub-exponential kernel)"),
        ("Scaled dot-product\nattention", 7.2, 3.3, "#2f7d33", None,
            "Satisfies S, C, N\nand uses no auxiliary state"),
        ("ReLU-gated routing\n(no softmax)", 10.0, 3.3, "#b9433a", 8.5,
            "Relaxes N: row sum is\nunconstrained"),
        ("Fock register mechanism\n(this programme)", 12.6, 3.3, "#7a4e9b", 12.0,
            "Relaxes auxiliary state = 0\n(introduces M register slots)"),
    ]

    box_half_w = 1.30
    box_half_h = 0.85
    for name, x, y_mech, color, target_x, caption in mechanisms:
        ax.add_patch(
            FancyBboxPatch(
                (x - box_half_w, y_mech - box_half_h),
                2 * box_half_w,
                2 * box_half_h,
                boxstyle="round,pad=0.08,rounding_size=0.14",
                facecolor="white",
                edgecolor=color,
                linewidth=1.7,
            )
        )
        ax.text(x, y_mech + 0.32, name, ha="center", va="center", fontsize=9.7, color=color)
        ax.text(
            x,
            y_mech - 0.45,
            caption,
            ha="center",
            va="center",
            fontsize=8.2,
            color="#333",
        )

        if target_x is not None:
            ax.add_patch(
                FancyArrowPatch(
                    (x, y_mech + box_half_h + 0.05),
                    (target_x, 6.4 - 0.55 - 0.05),
                    arrowstyle="->",
                    mutation_scale=14,
                    color=color,
                    linestyle="--",
                    linewidth=1.4,
                )
            )

    # Highlight attention as the unique mechanism that touches the SCN intersection.
    ax.add_patch(
        FancyArrowPatch(
            (7.2, 3.3 + box_half_h + 0.05),
            (7.2, 5.20),
            arrowstyle="->",
            mutation_scale=18,
            color="#2f7d33",
            linewidth=2.2,
        )
    )
    ax.text(
        7.2,
        4.60,
        "Lies on the\nPareto frontier",
        ha="center",
        fontsize=9.5,
        color="#2f7d33",
        style="italic",
    )

    ax.text(
        6.75,
        1.30,
        "Each non-attention mechanism relaxes exactly one SCN-constraint.\n"
        "The conjecture asserts: any such relaxation moves the mechanism off the corner of the Pareto frontier.",
        ha="center",
        fontsize=10.0,
        color="#222",
    )

    ax.set_xlim(-0.4, 14.2)
    ax.set_ylim(0.6, 8.2)
    ax.axis("off")

    fig.tight_layout()
    out = HERE / "design_space_map.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------------------------
# Figure 3 - Three justifications converging on attention
# --------------------------------------------------------------------------------------


def make_three_justifications() -> None:
    fig, ax = plt.subplots(figsize=(11.0, 6.4))

    # Three corner boxes converging on a centre. Centre placed inside the figure margins
    # so the outer boxes do not get clipped.
    centre = (5.5, 3.6)
    sources = [
        (2.1, 5.0, "Gibbs / max-entropy\n(Jaynes 1957)", "#1f4e79"),
        (8.9, 5.0, "Entropic optimal transport\n(Cuturi 2013; Sinkformer)", "#7a4e9b"),
        (5.5, 1.05, "Conservative obstruction\n(this programme)", "#b9433a"),
    ]

    ax.add_patch(
        FancyBboxPatch(
            (centre[0] - 1.95, centre[1] - 0.65),
            3.90,
            1.30,
            boxstyle="round,pad=0.10,rounding_size=0.15",
            facecolor="#e7f4ea",
            edgecolor="#2f7d33",
            linewidth=2.0,
        )
    )
    ax.text(
        centre[0],
        centre[1] + 0.30,
        "Scaled dot-product attention",
        ha="center",
        va="center",
        fontsize=12,
        weight="bold",
        color="#2f7d33",
    )
    ax.text(
        centre[0],
        centre[1] - 0.20,
        r"$\alpha^{\star}_{ij} = \mathrm{softmax}(\langle q_i, k_j\rangle / \sqrt{d_k})$",
        ha="center",
        va="center",
        fontsize=11,
        color="#2f7d33",
    )

    captions = [
        "Unique row-stochastic\nmaximiser of entropy at\nfixed expected score",
        "First Sinkhorn iterate\nof entropic OT under\nbilinear cost",
        "No conservative scalar\npotential reproduces P1+P2+P3\nwithout state extension",
    ]

    box_half_w = 1.50
    box_half_h = 0.85
    for (x, y, title, color), caption in zip(sources, captions):
        ax.add_patch(
            FancyBboxPatch(
                (x - box_half_w, y - box_half_h),
                2 * box_half_w,
                2 * box_half_h,
                boxstyle="round,pad=0.08,rounding_size=0.14",
                facecolor="white",
                edgecolor=color,
                linewidth=1.7,
            )
        )
        ax.text(x, y + 0.45, title, ha="center", va="center", fontsize=10.5, weight="bold", color=color)
        ax.text(x, y - 0.25, caption, ha="center", va="center", fontsize=8.6, color="#333")

        # Choose anchor point on the source box that faces the centre, then connect to the
        # corresponding edge of the centre box, so arrows do not enter the boxes.
        if y > centre[1]:
            src_anchor = (x, y - box_half_h - 0.02)
            tgt_anchor = (x, centre[1] + 0.65 + 0.02)
        else:
            src_anchor = (x, y + box_half_h + 0.02)
            tgt_anchor = (x, centre[1] - 0.65 - 0.02)

        ax.add_patch(
            FancyArrowPatch(
                src_anchor,
                tgt_anchor,
                arrowstyle="->",
                mutation_scale=18,
                color=color,
                linewidth=1.8,
                connectionstyle="arc3,rad=0.0",
            )
        )

    ax.text(
        centre[0],
        6.40,
        "Three independent justifications converge on softmax attention",
        ha="center",
        fontsize=12.5,
        color="#222",
        weight="bold",
    )

    ax.set_xlim(0.2, 10.8)
    ax.set_ylim(0.0, 6.8)
    ax.axis("off")

    fig.tight_layout()
    out = HERE / "three_justifications.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------------------------
# Figure 4 - Massless vs massive boson analogy
# --------------------------------------------------------------------------------------


def make_boson_analogy() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    def draw_panel(ax, title, mediator_label, mediator_color, decay_lambda, subtitle):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis("off")
        ax.set_title(title, fontsize=12, weight="bold", color="#222", pad=8)
        ax.text(5.0, 0.25, subtitle, ha="center", fontsize=9.5, color="#444", style="italic")

        # Two token particles
        for x, label in [(1.8, r"token $h_j$"), (8.2, r"token $h_i$")]:
            ax.plot(x, 4.0, "o", markersize=22, color="#1f4e79")
            ax.text(x, 3.05, label, ha="center", fontsize=10, color="#1f4e79")

        # Mediator(s)
        n_segments = 1
        x0, x1 = 1.8, 8.2
        if decay_lambda < 1.0:
            n_segments = 4

        if n_segments == 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x0 + 0.25, 4.0),
                    (x1 - 0.25, 4.0),
                    arrowstyle="-|>",
                    mutation_scale=18,
                    color=mediator_color,
                    linewidth=2.4,
                    connectionstyle="arc3,rad=-0.30",
                )
            )
            ax.text(
                5.0,
                5.30,
                mediator_label,
                ha="center",
                fontsize=10.5,
                color=mediator_color,
                weight="bold",
            )
        else:
            xs = np.linspace(x0 + 0.25, x1 - 0.25, n_segments + 1)
            for k in range(n_segments):
                alpha = max(0.18, decay_lambda ** (k + 1))
                ax.add_patch(
                    FancyArrowPatch(
                        (xs[k], 4.0),
                        (xs[k + 1], 4.0),
                        arrowstyle="-|>",
                        mutation_scale=14,
                        color=mediator_color,
                        linewidth=2.2,
                        alpha=alpha,
                        connectionstyle="arc3,rad=-0.18",
                    )
                )
                if 0 < k < n_segments:
                    ax.plot(xs[k], 4.0, "s", markersize=10, color=mediator_color, alpha=alpha)
            ax.text(
                5.0,
                5.30,
                mediator_label,
                ha="center",
                fontsize=10.5,
                color=mediator_color,
                weight="bold",
            )
            ax.text(
                5.0,
                4.85,
                rf"salience decay $\lambda = {decay_lambda:.2f}$",
                ha="center",
                fontsize=9.0,
                color=mediator_color,
            )

    draw_panel(
        axes[0],
        "Attention = massless boson",
        r"single virtual photon, range $= \infty$",
        "#2f7d33",
        decay_lambda=1.0,
        subtitle=r"infinite range within sequence; auxiliary state $\mathcal{S}=0$",
    )
    draw_panel(
        axes[1],
        "Fock register = massive boson",
        r"register particles, finite lifetime $\tau \approx 1/(1-\lambda)$",
        "#7a4e9b",
        decay_lambda=0.65,
        subtitle=r"persistence in depth at the cost of state $\mathcal{S}=M$ slots",
    )

    fig.suptitle(
        "Physical analogy: attention sits at the massless point in the design space",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    out = HERE / "boson_analogy.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    make_pareto_frontier()
    make_design_space_map()
    make_three_justifications()
    make_boson_analogy()


if __name__ == "__main__":
    main()
