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


if __name__ == "__main__":
    fig_truncation()
