"""
Generate the illustrative figure for §21.12 of
Improving_the_Fock_Mechanism_to_match_Attention.md:

perlayer_repulsion_interaction.png — four panels:

  (A) Per-layer execution order within _fock_layer_step.  A vertical
      pipeline showing the seven stages (creation -> active mask ->
      conservative dynamics -> split -> repulsion -> reverse channel ->
      destruction) with annotations showing where B4 and per-layer gate
      act.

  (B) Gradient flow diagram.  Two independent paths from the loss to the
      register states: the NTP gradient (through the logits and reverse
      channel) pulls registers toward useful content, while the repulsion
      gradient pushes them apart.  The per-layer gate decouples the
      channel's contribution across layers.

  (C) Variance reduction: global gate vs per-layer gate.  Bar chart
      showing Var(dL/ds) proportional to L^2 sigma^2 for the global gate
      versus L independent bars each proportional to sigma^2.

  (D) Complementarity matrix: which fix addresses which failure mode.
      A 2x2 heatmap with {repulsion, per-layer gate} x {register collapse,
      gate divergence}.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe

fig = plt.figure(figsize=(18, 13), facecolor="white")

# ── Panel A: execution pipeline ──────────────────────────────────────
ax_a = fig.add_axes([0.03, 0.08, 0.23, 0.84])
ax_a.set_xlim(-0.5, 4.5)
ax_a.set_ylim(-0.5, 8.5)
ax_a.axis("off")
ax_a.set_title("(A)  Layer step execution order", fontsize=13, fontweight="bold", pad=12)

stages = [
    ("1. Creation gate", "#bde0fe", None),
    ("2. Active mask", "#bde0fe", None),
    ("3. Conservative\n    dynamics (PARF)", "#bde0fe", None),
    ("4. Split h / r", "#bde0fe", None),
    ("5. Repulsion (B4)", "#ffd6a5", "B4"),
    ("6. Reverse channel", "#caffbf", "Per-layer\ngate $s_\\ell$"),
    ("7. Destruction gate", "#bde0fe", None),
]

for i, (label, color, annot) in enumerate(stages):
    y = 7.5 - i * 1.1
    box = FancyBboxPatch((0.3, y - 0.35), 2.4, 0.7, boxstyle="round,pad=0.08",
                         facecolor=color, edgecolor="#333", linewidth=1.5)
    ax_a.add_patch(box)
    ax_a.text(1.5, y, label, ha="center", va="center", fontsize=9.5, fontweight="bold")
    if i < len(stages) - 1:
        ax_a.annotate("", xy=(1.5, y - 0.45), xytext=(1.5, y - 0.65),
                      arrowprops=dict(arrowstyle="->", lw=1.5, color="#555"))
    if annot:
        ax_a.annotate(annot, xy=(2.8, y), xytext=(3.6, y),
                      fontsize=8, color="#c1121f", fontweight="bold",
                      ha="left", va="center",
                      arrowprops=dict(arrowstyle="->", color="#c1121f", lw=1.2))

ax_a.text(1.5, -0.3, "Ordering: repulsion\nDIVERSIFIES registers\nBEFORE reverse channel\nreads from them",
          ha="center", va="top", fontsize=8.5, style="italic",
          bbox=dict(boxstyle="round,pad=0.3", fc="#fff3cd", ec="#ffc107", alpha=0.9))


# ── Panel B: gradient flow ───────────────────────────────────────────
ax_b = fig.add_axes([0.29, 0.50, 0.35, 0.42])
ax_b.set_xlim(-1, 10)
ax_b.set_ylim(-0.5, 6)
ax_b.axis("off")
ax_b.set_title("(B)  Gradient flow: two independent paths to registers",
               fontsize=13, fontweight="bold", pad=12)

node_props = dict(ha="center", va="center", fontsize=9.5, fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.35", ec="#333", lw=1.3))

ax_b.text(4.5, 5.3, "NTP Loss $\\mathcal{L}$",
          **{**node_props, "bbox": {**node_props["bbox"], "fc": "#ffccd5"}})

ax_b.text(1.5, 3.5, "Repulsion\n$\\lambda_{\\mathrm{rep}} \\cdot \\rho^2$",
          **{**node_props, "bbox": {**node_props["bbox"], "fc": "#ffd6a5"}})
ax_b.text(7.5, 3.5, "Reverse ch.\n$\\tanh(s_\\ell) \\hat{Q}_i^{(\\ell)}$",
          **{**node_props, "bbox": {**node_props["bbox"], "fc": "#caffbf"}})

ax_b.text(4.5, 1.5, "Register states\n$r_k^{(\\ell)}$",
          **{**node_props, "bbox": {**node_props["bbox"], "fc": "#a2d2ff"}})

ax_b.text(4.5, 0.0, "Token states $h_i^{(\\ell)}$",
          **{**node_props, "bbox": {**node_props["bbox"], "fc": "#e2e2e2"}})

for sx, sy, ex, ey, color, lbl in [
    (3.5, 5.0, 1.8, 4.1, "#c1121f", ""),
    (5.5, 5.0, 7.2, 4.1, "#2d6a4f", ""),
    (1.5, 2.9, 3.5, 2.0, "#c1121f", "push apart"),
    (7.5, 2.9, 5.5, 2.0, "#2d6a4f", "pull toward\nuseful content"),
    (4.5, 1.0, 4.5, 0.5, "#555", ""),
]:
    ax_b.annotate("", xy=(ex, ey), xytext=(sx, sy),
                  arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0,
                                  connectionstyle="arc3,rad=0.0"))
    if lbl:
        mx, my = (sx + ex) / 2, (sy + ey) / 2
        offset_x = -1.0 if sx < ex else 1.0
        ax_b.text(mx + offset_x, my, lbl, fontsize=8, color=color,
                  ha="center", va="center", style="italic")


# ── Panel C: variance reduction ──────────────────────────────────────
ax_c = fig.add_axes([0.68, 0.50, 0.30, 0.42])
ax_c.set_title("(C)  Gate gradient variance", fontsize=13, fontweight="bold", pad=12)

L = 16
global_var = L ** 2
per_layer_var = np.ones(L)

bars_x = np.arange(L)
ax_c.bar(bars_x, per_layer_var, color="#caffbf", edgecolor="#2d6a4f", linewidth=1.2,
         label="Per-layer gate: $\\sigma_Q^2$ each", zorder=3)
ax_c.axhline(global_var, color="#c1121f", linewidth=2.5, linestyle="--",
             label=f"Global gate: $L^2 \\sigma_Q^2 = {L}^2 \\sigma_Q^2$", zorder=4)

ax_c.set_xlabel("Layer index $\\ell$", fontsize=11)
ax_c.set_ylabel("Var($\\partial\\mathcal{L} / \\partial s$)  (units of $\\sigma_Q^2$)",
                fontsize=10)
ax_c.set_ylim(0, global_var * 1.25)
ax_c.set_xticks(range(0, L, 2))
ax_c.legend(fontsize=9, loc="upper left")
ax_c.text(L / 2, global_var * 1.1,
          f"{L}x variance\nreduction",
          ha="center", va="bottom", fontsize=11, fontweight="bold", color="#c1121f",
          path_effects=[pe.withStroke(linewidth=3, foreground="white")])
ax_c.grid(axis="y", alpha=0.3)


# ── Panel D: complementarity matrix ─────────────────────────────────
ax_d = fig.add_axes([0.29, 0.06, 0.34, 0.36])
ax_d.set_title("(D)  Fix-to-failure complementarity", fontsize=13, fontweight="bold", pad=12)

matrix = np.array([[1.0, 0.15],
                    [0.15, 1.0]])

im = ax_d.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1.0, aspect="auto")

fix_labels = ["B4 Register\nrepulsion", "Per-layer\nreverse gate"]
fail_labels = ["Register collapse\n(high $\\rho$, low $N_{\\mathrm{eff}}$)",
               "Gate divergence\n($L^2$ variance)"]

ax_d.set_xticks([0, 1])
ax_d.set_xticklabels(fail_labels, fontsize=9)
ax_d.set_yticks([0, 1])
ax_d.set_yticklabels(fix_labels, fontsize=9)

for i in range(2):
    for j in range(2):
        strength = "PRIMARY\ntarget" if matrix[i, j] > 0.5 else "indirect\nbenefit"
        color = "white" if matrix[i, j] > 0.5 else "#333"
        ax_d.text(j, i, strength, ha="center", va="center",
                  fontsize=10, fontweight="bold", color=color)

ax_d.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)


# ── Panel D side note ────────────────────────────────────────────────
ax_note = fig.add_axes([0.68, 0.06, 0.30, 0.36])
ax_note.axis("off")
ax_note.set_title("(E)  Combined causal chain", fontsize=13, fontweight="bold", pad=12)

chain = [
    ("B4 repulsion", "#ffd6a5", "lowers $\\rho$"),
    ("Higher $N_{\\mathrm{eff}}$", "#a2d2ff", "diverse keys/values"),
    ("Reverse ch. reads\nricher content", "#caffbf", "per-layer $s_\\ell$"),
    ("Lower variance\nper gate gradient", "#caffbf", "decoupled $\\partial\\mathcal{L}/\\partial s_\\ell$"),
    ("Longer stable\ntraining horizon", "#d4edda", "past 25.5k steps"),
    ("Lower PPL", "#ffccd5", ""),
]

for i, (lbl, col, annot) in enumerate(chain):
    y = 5.0 - i * 0.9
    box = FancyBboxPatch((0.2, y - 0.3), 3.0, 0.6,
                         boxstyle="round,pad=0.08", facecolor=col,
                         edgecolor="#333", linewidth=1.2)
    ax_note.add_patch(box)
    ax_note.text(1.7, y, lbl, ha="center", va="center", fontsize=9, fontweight="bold")
    if annot:
        ax_note.text(3.5, y, annot, fontsize=7.5, color="#555", va="center",
                     style="italic")
    if i < len(chain) - 1:
        ax_note.annotate("", xy=(1.7, y - 0.4), xytext=(1.7, y - 0.55),
                         arrowprops=dict(arrowstyle="->", lw=1.3, color="#555"))

ax_note.set_xlim(-0.3, 5.5)
ax_note.set_ylim(-0.8, 5.8)

plt.savefig("companion_notes/images/perlayer_repulsion_interaction.png",
            dpi=180, bbox_inches="tight", facecolor="white")
print("Saved perlayer_repulsion_interaction.png")
