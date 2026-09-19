"""Figure for Fock_Inference_Productionization_Plan.md SS6.1.

One PNG into figures/fock_inference/. Every number is MEASURED, not
modelled: the spectrum and the truncation sweep both come from
`semsimula_diag.probes.generator_rank` run against the d=384 step-28,500
checkpoint on 2026-09-19 (uniform sweep, 12 fixed batches x 4 x 512 tok,
paired -- every row scored on the same batches, so the deltas carry no
sampling noise).

  generator_truncation.png -- Panel A: validation perplexity against the
                              surviving fraction of the generator, with the
                              knee between rank 1024 and 512 and the one
                              cheap operating point marked. Panel B: per-map
                              energy rank against that map's break-even rank
                              mn/(m+n); a bar crossing the break-even line is
                              a map with NO faithful factorisation that is
                              also a saving.

Run:  python3 _make_fock_inference_truncation_figs.py
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

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

BASE_PPL = 84.96
# rank, factorised params, fraction of the 35,389,440-param generator, ppl
SWEEP = [(1024, 24_772_608, 0.700, 85.25),
         (512, 12_386_304, 0.350, 90.65),
         (256, 6_193_152, 0.175, 117.30),
         (128, 3_096_576, 0.088, 175.74),
         (64, 1_548_288, 0.044, 239.92),
         (32, 774_144, 0.022, 366.21)]
# name, (out, in), PR, r@0.9, r@0.99
MAPS = [("$W_B$", (12288, 1920), 421.9, 707, 1506),
        (r"$W_\mu$", (3072, 1920), 92.0, 668, 1483),
        ("$W_a$", (3072, 1920), 5.7, 242, 773)]


def fig_truncation():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 4.6))

    fracs = [s[2] for s in SWEEP]
    ppls = [s[3] for s in SWEEP]
    axA.axhline(BASE_PPL, color=GREY, ls="--", lw=1.4, zorder=1)
    axA.text(0.023, BASE_PPL * 1.04, f"dense baseline  {BASE_PPL:.2f}",
             color=GREY, fontsize=9.5, va="bottom")
    axA.plot(fracs, ppls, "-o", color=BLUE, lw=2, ms=7, zorder=3)
    for (r, _, f, p) in SWEEP:
        off = 1.09 if r >= 512 else 0.86
        axA.annotate(f"r={r}", (f, p), textcoords="offset points",
                     xytext=(6, 9 if r >= 512 else -16), fontsize=9.5,
                     color="#374151")
    # The one operating point that is close to free.
    axA.scatter([0.700], [85.25], s=190, facecolors="none",
                edgecolors=GREEN, lw=2.4, zorder=4)
    axA.annotate("+0.29 ppl for 30%\nof the generator",
                 (0.700, 85.25), xytext=(0.30, 62), fontsize=10,
                 color=GREEN, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.6))
    axA.set_xscale("log")
    axA.set_yscale("log")
    axA.set_xlabel("surviving fraction of the 35.4M-parameter generator")
    axA.set_ylabel("validation perplexity")
    axA.set_title("A. Truncation is not free below rank 1024")
    axA.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    axA.set_xticklabels(["2%", "5%", "10%", "20%", "50%", "100%"])
    axA.set_yticks([60, 100, 200, 400])
    axA.set_yticklabels(["60", "100", "200", "400"])
    # A log axis relabels its minor ticks too, which collides with the
    # explicit major labels above.
    axA.xaxis.set_minor_formatter(NullFormatter())
    axA.yaxis.set_minor_formatter(NullFormatter())
    axA.grid(alpha=0.25, which="both")

    ys = range(len(MAPS))
    h = 0.34
    for i, (name, shape, pr, r90, r99) in enumerate(MAPS):
        be = shape[0] * shape[1] / (shape[0] + shape[1])
        axB.barh(i + h / 2, r99, height=h, color=PURPLE, alpha=0.85,
                 zorder=2, label="r@0.99" if i == 0 else None)
        axB.barh(i - h / 2, r90, height=h, color=BLUE, alpha=0.85,
                 zorder=2, label="r@0.9" if i == 0 else None)
        axB.plot([be, be], [i - 0.46, i + 0.46], color=RED, lw=3, zorder=4)
        over = r99 > be
        # Label inside a long bar: outside, it runs into the break-even rule.
        axB.text(r99 - 40, i + h / 2, f"{r99}", va="center", ha="right",
                 fontsize=9.5, color="white", fontweight="bold")
        if over:
            axB.text(r99 + 130, i + h / 2, "over", va="center", fontsize=9,
                     color=RED, fontweight="bold")
        axB.text(r90 - 40, i - h / 2, f"{r90}", va="center", ha="right",
                 fontsize=9.5, color="white", fontweight="bold")
    axB.axvline(1920, color=GREY, ls=":", lw=1.6, zorder=1)
    axB.text(1895, -0.52, "max rank\n(xi_dim = 1920)", fontsize=9,
             color=GREY, va="top", ha="right")
    axB.set_yticks(list(ys))
    axB.set_yticklabels([m[0] for m in MAPS], fontsize=13)
    axB.set_xlim(0, 2100)
    axB.set_xlabel("rank")
    axB.set_title("B. $W_\\mu$ has no faithful factorisation that saves")
    axB.invert_yaxis()
    axB.grid(alpha=0.25, axis="x")
    handles, labels = axB.get_legend_handles_labels()
    handles.append(Line2D([], [], color=RED, lw=3))
    labels.append("break-even $mn/(m{+}n)$")
    axB.legend(handles, labels, fontsize=9, loc="lower right", framealpha=0.95)

    fig.tight_layout()
    fig.savefig("fock_inference/generator_truncation.png",
                bbox_inches="tight")
    print("wrote fock_inference/generator_truncation.png")




# Per-map allocation, measured 2026-09-19 in the same cell and on the same
# 12 fixed batches, so these are directly comparable to SWEEP above.
# label, factorised params, ppl, d ppl
PER_MAP = [("mixed  B/mu 1024, a 256", 20_938_752, 85.25, 0.30),
           ("mixed  B/mu 1024, a 64", 19_980_288, 85.33, 0.37),
           ("energy 99%, capped", 31_151_616, 85.01, 0.05)]
GEN_FULL = 35_389_440


def fig_frontier():
    """Where the allocations sit against each other, in parameters spent.

    The 99%-energy allocation is NOT dominated -- no measured point has
    both fewer parameters and a smaller loss -- so it is Pareto-optimal and
    the panel must not say otherwise. What it is, is a bad buy: 10.2M
    parameters for 0.25 ppl, about 41M per ppl, against a mixed-to-uniform
    step that moves 3.83M for 0.01. SVD energy optimises Frobenius
    fidelity, and fidelity is not the thing being purchased.
    """
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 4.6))

    FLOOR = 0.02                       # log axis needs a floor for d=0
    ux = [s[1] / 1e6 for s in SWEEP]
    uy = [max(s[3] - BASE_PPL, FLOOR) for s in SWEEP]
    axA.plot(ux, uy, "-o", color=BLUE, lw=2, ms=7, zorder=3,
             label="uniform rank")
    for (r, p, _, ppl) in SWEEP:
        if r <= 512:
            axA.annotate(f"r={r}", (p / 1e6, ppl - BASE_PPL),
                         textcoords="offset points", xytext=(7, -3),
                         fontsize=9.5, color="#374151")
    axA.scatter([GEN_FULL / 1e6], [FLOOR], marker="s", s=80, color=GREY,
                zorder=4, label="dense (35.4M)")
    for label, p, _, d in PER_MAP:
        is_energy = label.startswith("energy")
        axA.scatter([p / 1e6], [max(d, FLOOR)], marker="D" if is_energy else "*",
                    s=110 if is_energy else 260,
                    color=RED if is_energy else GREEN, zorder=5,
                    label=("99% energy" if is_energy else None)
                    if is_energy else ("per-map mixed" if d == 0.30 else None))
    axA.annotate("10.2M more than mixed,\nto buy 0.25 ppl", (31.15, 0.05),
                 xytext=(13.5, 0.09), fontsize=9.5, color=RED,
                 fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    axA.annotate("same loss as uniform r=1024,\n3.83M fewer parameters",
                 (20.94, 0.30), xytext=(2.0, 1.4), fontsize=9.5,
                 color=GREEN, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5))
    axA.set_yscale("log")
    axA.set_xlabel("generator parameters retained (millions of 35.4M)")
    axA.set_ylabel("$\\Delta$ perplexity vs dense")
    axA.set_title("A. 99% energy retention is a poor buy")
    axA.set_yticks([0.02, 0.1, 1, 10, 100])
    axA.set_yticklabels(["0", "0.1", "1", "10", "100"])
    axA.yaxis.set_minor_formatter(NullFormatter())
    axA.set_xlim(-1, 38)
    axA.grid(alpha=0.25, which="major")
    axA.legend(fontsize=9, loc="upper right", framealpha=0.95)

    bars = [("99% energy\ncapped", 31_151_616, 0.05, RED),
            ("uniform\nr = 1024", 24_772_608, 0.29, BLUE),
            ("mixed\nB/mu 1024, a 256", 20_938_752, 0.30, GREEN),
            ("mixed\nB/mu 1024, a 64", 19_980_288, 0.37, GREEN)]
    xs = range(len(bars))
    axB.bar(xs, [b[1] / 1e6 for b in bars], color=[b[3] for b in bars],
            alpha=0.85, width=0.62, zorder=2)
    for i, (lbl, p, d, _) in enumerate(bars):
        axB.text(i, p / 1e6 + 0.6, f"{p/1e6:.2f}M", ha="center",
                 fontsize=10, fontweight="bold")
        axB.text(i, p / 1e6 / 2, f"+{d:.2f}\nppl", ha="center",
                 va="center", fontsize=11, color="white", fontweight="bold")
    axB.axhline(GEN_FULL / 1e6, color=GREY, ls="--", lw=1.4, zorder=1)
    axB.text(3.42, GEN_FULL / 1e6 + 0.5, "dense 35.39M", fontsize=9,
             color=GREY, ha="right")
    axB.set_xticks(list(xs))
    axB.set_xticklabels([b[0] for b in bars], fontsize=9.5)
    axB.set_ylabel("generator parameters (M)")
    axB.set_ylim(0, 39)
    axB.set_title("B. Cost of staying under +0.5 ppl")
    axB.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig("fock_inference/generator_frontier.png", bbox_inches="tight")
    print("wrote fock_inference/generator_frontier.png")


if __name__ == "__main__":
    fig_truncation()
    fig_frontier()
