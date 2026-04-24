"""
Assemble a 2x2 comparison figure of the four 3D landscape renders
(baseline, LN, SG, GM) for the dialogue prompt.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import imread

ROOT = Path(__file__).parent.parent
ATTR = ROOT / "attractor_analysis" / "results"
OUT  = Path(__file__).parent / "results" / "landscape3d_compare_four_variants_dialogue.png"

PANELS = [
    ("baseline SARF+mass (logfreq)\nval ppl 160.55",
     "landscape3d_landscape3d_em_base_dialogue.png"),
    ("(i) LayerNorm-after-step\nval ppl 88.63  (BEST)",
     "landscape3d_landscape3d_em_ln_dialogue.png"),
    ("(ii) scale-gauge $\\lambda_{V_0}{=}10^{-3}$\nval ppl 191.00",
     "landscape3d_landscape3d_em_sg_dialogue.png"),
    ("(iii) Gaussian-mixture head $K{=}64$\nval ppl 677.67  (saturated)",
     "landscape3d_landscape3d_em_gm_dialogue.png"),
]

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
for ax, (title, fname) in zip(axes.flat, PANELS):
    img = imread(str(ATTR / fname))
    ax.imshow(img)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
fig.suptitle(
    r"Learned scalar potential $V_\theta(\xi, h)$ + damped-flow trajectories "
    r"(dialogue prompt): baseline SPLM vs three energetic-minima alternatives.",
    fontsize=12, y=0.995,
)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"saved -> {OUT}")
