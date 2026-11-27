"""Figures for semantic_templates/Deploying_Semantic_Templates.md (2026-11-27).

Nothing here is read from a trained model. The first figure is a schematic; the second is computed from tensor shapes and
from two measured cache sizes quoted in the note. Writes into figures/template_deployment/:
  dep_insertion_points.png  where each deployment scenario of the note attaches to the model: the conservative ladder model
                            (part I, scenarios I.1 to I.6) and an attention-based transformer (part II, II.1 to II.9)
  dep_costs.png             (a) numbers cached per token for decoding: SR2-NP and the matched GPT-2 as measured, and the
                            particle store and virtual tokens as estimated from their shapes; (b) the number of T x T slices
                            a prefill builds per sequence, against depth: the context channels, the particle store at
                            K = 2 and 4 (every layer reads the particles of every lower layer), GPT-2's attention weights
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "figure.dpi": 150})
INK, MUTED, GREY, PALE = "#1F2937", "#6B7280", "#9CA3AF", "#F3F4F6"
C = ["#3B82F6", "#DC2626", "#059669", "#F59E0B", "#7C3AED"]          # fixed categorical order
OUT = Path(__file__).resolve().parent / "template_deployment"; OUT.mkdir(exist_ok=True)


def box(ax, x, y, w, h, text, fc=PALE, ec=GREY, fs=9.5, color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color)


def arrow(ax, x0, y0, x1, y1, color=GREY, ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, ls=ls))


def tag(ax, x, y, s, color):
    ax.add_patch(plt.Circle((x, y), 0.21, fc=color, ec="white", lw=1.5, zorder=5))
    ax.text(x, y, s, ha="center", va="center", fontsize=7.5, color="white", fontweight="bold", zorder=6)


def pipeline(ax, y, layers, title):
    """tokens -> embedding -> layers -> head; returns the x centres of the layer boxes"""
    ax.text(0.0, y + 1.55, title, fontsize=12, fontweight="bold", color=INK, ha="left")
    box(ax, 0.0, y, 1.3, 0.9, "tokens")
    box(ax, 1.8, y, 1.5, 0.9, "embedding\n$h_0$")
    xs = []
    for i, name in enumerate(layers):
        x = 3.8 + 1.75 * i
        box(ax, x, y, 1.45, 0.9, name)
        xs.append(x + 0.725)
    xh = 3.8 + 1.75 * len(layers)
    box(ax, xh, y, 1.4, 0.9, "head\nnext token")
    for x0, x1 in [(1.3, 1.8), (3.3, 3.8)] + [(3.8 + 1.75 * i + 1.45, 3.8 + 1.75 * (i + 1)) for i in range(len(layers))]:
        arrow(ax, x0, y + 0.45, x1, y + 0.45)
    return xs, xh


fig, ax = plt.subplots(figsize=(14, 9.6))
ax.set_xlim(-0.2, 16.6); ax.set_ylim(-1.9, 7.9); ax.set_aspect("equal"); ax.axis("off")
W4 = 1.75 * 4 - 0.3                                                   # width of the boxes that span the layers
RIGHT = 3.8 + W4 + 0.32                                               # tags just right of those boxes

# part I: the conservative ladder model
yI = 6.0
xs, xh = pipeline(ax, yI, ["layer 1\nwells $V_\\theta$", "layer 2\nwells $V_\\theta$", "...", "layer $L$"],
                  "Part I: a conservative ladder model (SR2-NP)")
box(ax, 3.8, yI - 1.45, W4, 0.75, "particle store: write-once slots, seen by later tokens and higher layers", fc="#ECFDF5", ec=C[2], fs=8.5)
for x in xs[:2] + xs[3:]:
    arrow(ax, x, yI - 0.7, x, yI - 0.02, color=C[2])
box(ax, 3.8, yI + 1.0, W4, 0.45, "states, displacements and well frames, read at every layer", fc="white", ec=GREY, fs=9)
box(ax, xh + 1.9, yI + 0.05, 2.6, 0.8, "external action\n(flag, checker, tool)", fc="white", ec=GREY, fs=9)
arrow(ax, xh + 1.4, yI + 0.45, xh + 1.9, yI + 0.45, color=GREY, ls="--")
tag(ax, 3.45, yI + 1.22, "I.1", C[0]); tag(ax, xh + 3.2, yI + 1.15, "I.2", C[0])
tag(ax, 3.8 + 1.45, yI + 0.9, "I.3", C[1]); tag(ax, 3.8 + 1.75 + 1.45, yI + 0.9, "I.4", C[1])
tag(ax, 3.45, yI - 1.07, "I.5", C[2]); tag(ax, RIGHT, yI - 1.07, "I.6", C[4])

# part II: an attention-based transformer
yII = 1.3
xs2, xh2 = pipeline(ax, yII, ["block 1\nattn + MLP", "block 2\nattn + MLP", "...", "block $L$"],
                    "Part II: an attention-based transformer (the matched GPT-2)")
box(ax, 3.8, yII - 1.45, W4, 0.75, "key-value cache: a key and a value per token, block and head", fc="#EFF6FF", ec=C[0], fs=9)
for x in xs2[:2] + xs2[3:]:
    arrow(ax, x, yII - 0.7, x, yII - 0.02, color=C[0])
box(ax, 3.8, yII + 1.0, W4, 0.45, "residual stream, read or written between blocks", fc="white", ec=GREY, fs=9)
ax.add_patch(FancyBboxPatch((3.7, yII - 0.1), W4 + 0.2, 1.1, boxstyle="round,pad=0.02,rounding_size=0.1", fc="none", ec=C[3], lw=1.2, ls="--"))
box(ax, xh2 + 1.9, yII + 0.05, 2.6, 0.8, "emitted text, tools,\nconstrained decoding", fc="white", ec=GREY, fs=9)
arrow(ax, xh2 + 1.4, yII + 0.45, xh2 + 1.9, yII + 0.45, color=GREY, ls="--")
tag(ax, 3.45, yII + 1.22, "II.1", C[0]); tag(ax, RIGHT, yII + 1.22, "II.2", C[1])
tag(ax, 3.7 + W4 + 0.25, yII - 0.1, "II.3", C[3]); tag(ax, 3.8 + 1.75 + 1.45, yII + 0.9, "II.4", C[1])
tag(ax, 3.8 + 1.75 * 3 + 1.45, yII + 0.9, "II.5", C[1]); tag(ax, 3.45, yII - 1.07, "II.6", C[2])
tag(ax, xh2 + 2.75, yII + 1.15, "II.7", C[2]); tag(ax, xh2 + 3.25, yII + 1.15, "II.8", C[0])
tag(ax, RIGHT, yII - 1.07, "II.9", C[4])

# the key, by the job a template does
for i, (c, s) in enumerate([(C[0], "reads only"), (C[1], "pushes the computation (a force or an added vector)"),
                            (C[2], "creates new state (particles, tokens)"), (C[3], "changes the weights of every block"),
                            (C[4], "templates made at inference")]):
    x, y = [(0.2, -1.0), (3.2, -1.0), (10.2, -1.0), (0.2, -1.6), (5.6, -1.6)][i]
    ax.add_patch(plt.Circle((x, y), 0.14, fc=c, ec="white"))
    ax.text(x + 0.3, y, s, va="center", fontsize=9.5, color=INK)
fig.savefig(OUT / "dep_insertion_points.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
print("wrote dep_insertion_points.png")

# costs from shapes
d, L_np, K_ch, dk = 384, 2, 5, 48
L_gp, H = 8, 6
np_cache = L_np * (K_ch * dk + d)                                     # 1,248, measured by the decoder note
gp_cache = 2 * L_gp * d                                               # 6,144
store_lo = np_cache + 4 * L_np                                        # payload = a state the decoder already keeps; only K existences
store_hi = np_cache + L_np * (4 + d)                                  # payload kept as one more state per layer
vt = gp_cache * 1.1                                                   # one virtual token per ten text tokens
fig, axs = plt.subplots(1, 2, figsize=(14, 4.8))
lab = ["SR2-NP\n(measured)", "SR2-NP + store\npayload shared", "SR2-NP + store\npayload kept", "GPT-2 W2\n(measured)",
       "GPT-2 W2 + 1 virtual\ntoken per 10"]
val = [np_cache, store_lo, store_hi, gp_cache, vt]
col = [C[2], C[2], C[2], C[0], C[0]]
bars = axs[0].bar(range(5), val, color=col, width=0.62, edgecolor="white", linewidth=2)
for b, hatch in zip(bars, ["", "//", "//", "", "//"]):
    b.set_hatch(hatch)
for i, v in enumerate(val):
    axs[0].text(i, v + 120, f"{v:,.0f}", ha="center", fontsize=9.5, color=INK)
axs[0].set_xticks(range(5)); axs[0].set_xticklabels(lab, fontsize=9)
axs[0].set_ylabel("numbers cached per token of context")
axs[0].set_title("(a) decoding cache per token; hatched = estimated from shapes", loc="left")
Ls = np.arange(1, 13)
axs[1].plot(Ls, K_ch * Ls, "o-", color=C[2], ms=5, lw=1.8, label="context channels, 5 per layer")
axs[1].plot(Ls, 2 * Ls * (Ls + 1) / 2, "s--", color=C[1], mfc="white", ms=5, lw=1.4, label="particle store, K = 2")
axs[1].plot(Ls, 4 * Ls * (Ls + 1) / 2, "s-", color=C[1], ms=5, lw=1.6, label="particle store, K = 4")
axs[1].plot(Ls, H * Ls, "^-", color=C[0], ms=5, lw=1.6, label="GPT-2 attention weights, 6 heads per block")
axs[1].set_xlabel("layers or blocks"); axs[1].set_ylabel("T x T slices built in one prefill")
axs[1].set_title("(b) the store's cross-layer reads grow with depth squared", loc="left")
axs[1].legend(frameon=False, fontsize=9, loc="upper left")
for ax in axs:
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "dep_costs.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
print("wrote dep_costs.png", np_cache, store_lo, store_hi, gp_cache)
