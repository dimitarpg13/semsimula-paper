"""
Regenerate ln_counter_damping_mechanism.png with correct single-headed velocity arrows.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

# ── colour palette ────────────────────────────────────────────────
BLUE_DARK   = "#1565C0"
BLUE_MED    = "#1E88E5"
BLUE_LIGHT  = "#90CAF9"
TEAL_DARK   = "#00695C"
TEAL_MED    = "#00897B"
RED         = "#E53935"
GREY_DASH   = "#9E9E9E"
GOLD        = "#F9A825"
BG          = "#FAFAFA"

fig, axes = plt.subplots(1, 2, figsize=(14, 8),
                         facecolor=BG,
                         gridspec_kw={"wspace": 0.30})

R = 1.0  # sphere radius for the right panel

# ─────────────────────────────────────────────────────────────────
#  LEFT PANEL – Without LayerNorm
# ─────────────────────────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor(BG)
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-2.4, 2.4)
ax.set_aspect("equal")
ax.axhline(0, color="k", lw=0.6, zorder=1)
ax.axvline(0, color="k", lw=0.6, zorder=1)
ax.text(2.15, 0.08, r"$h_1$", fontsize=11, ha="left")
ax.text(0.08, 2.3, r"$h_2$", fontsize=11)
ax.text(-0.12, -0.12, "0", fontsize=9, ha="right")

# dashed sphere boundary
theta_full = np.linspace(0, 2 * np.pi, 300)
ax.plot(1.7 * np.cos(theta_full), 1.7 * np.sin(theta_full),
        "--", color=GREY_DASH, lw=1.3, zorder=2)
ax.text(0.5, 1.78, r"sphere $\|\mathbf{h}\|=\sqrt{d}$",
        fontsize=8.5, color=GREY_DASH)
ax.plot(0.52, 1.7, "o", color="k", ms=5, zorder=5)

# inward spiral: r decays with angle
angles = np.linspace(0, 6 * np.pi, 800)
r_spiral = 1.65 * np.exp(-0.10 * angles)
xs = r_spiral * np.cos(angles + np.pi / 2)
ys = r_spiral * np.sin(angles + np.pi / 2)

# gradient from dark → light to show decay
from matplotlib.collections import LineCollection
points = np.array([xs, ys]).T.reshape(-1, 1, 2)
segs   = np.concatenate([points[:-1], points[1:]], axis=1)
n      = len(segs)
colors = [plt.cm.Blues(0.85 - 0.45 * i / n) for i in range(n)]
lc = LineCollection(segs, colors=colors, linewidths=1.8, zorder=3)
ax.add_collection(lc)

# velocity arrows along spiral – single-headed, tangent direction
v_indices = [80, 220, 370, 520, 660]
arrow_kw = dict(
    arrowstyle="-|>",
    mutation_scale=12,
    lw=1.4,
    zorder=6,
)
for idx in v_indices:
    if idx + 5 >= len(xs):
        continue
    x0, y0 = xs[idx], ys[idx]
    dx = xs[idx + 5] - xs[idx]
    dy = ys[idx + 5] - ys[idx]
    norm = np.hypot(dx, dy)
    dx, dy = dx / norm * 0.28, dy / norm * 0.28
    ax.annotate(
        "", xy=(x0 + dx, y0 + dy), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=BLUE_DARK,
                        mutation_scale=12, lw=1.5),
        zorder=6,
    )
    ax.text(x0 + dx * 1.5, y0 + dy * 1.5,
            r"$v$", fontsize=9, color=BLUE_DARK,
            ha="center", va="center", zorder=7)

ax.set_title(
    "Without LayerNorm\n"
    r"$\it{State\ spirals\ inward;\ forces\ vanish\ near\ origin,}$"
    "\n"
    r"$\it{\gamma_{\rm eff}=\gamma}$",
    fontsize=11, fontweight="bold", color=BLUE_DARK, pad=6,
    linespacing=1.5,
)
ax.set_xticks([])
ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)

# ─────────────────────────────────────────────────────────────────
#  CENTRE formula (axes[0] right side, overlapping into axes[1])
#  We'll place it as an inset axis
# ─────────────────────────────────────────────────────────────────
# Place the formula as a text box between the two panels
fig.text(0.515, 0.52,
         r"$\gamma_{\rm eff}=\gamma-\dfrac{\delta V}{2T\cdot\Delta t}$",
         fontsize=13, ha="center", va="center",
         bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=BLUE_DARK, lw=1.6),
         zorder=10)
fig.text(0.515, 0.38,
         r"$\uparrow$  $\delta V$" "\nenergy injected\nby LN",
         fontsize=9, ha="center", va="top",
         color=RED, zorder=10)

# ─────────────────────────────────────────────────────────────────
#  RIGHT PANEL – With LayerNorm
# ─────────────────────────────────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor(BG)
ax2.set_xlim(-1.8, 1.8)
ax2.set_ylim(-1.9, 1.9)
ax2.set_aspect("equal")
ax2.axhline(0, color="k", lw=0.6, zorder=1)
ax2.axvline(0, color="k", lw=0.6, zorder=1)
ax2.text(1.75, 0.06, r"$h_1$", fontsize=11, ha="left")
ax2.text(0.06, 1.84, r"$h_2$", fontsize=11)
ax2.text(-0.12, -0.12, "0", fontsize=9, ha="right")

# potential contours (concentric ellipses, slightly eccentric for realism)
for r_c in [0.4, 0.65, 0.9, 1.15, 1.38]:
    ax2.plot(r_c * np.cos(theta_full),
             r_c * 0.85 * np.sin(theta_full),
             color="#B0BEC5", lw=0.7, zorder=1)
ax2.text(1.35, 1.05, r"$V_\theta(h)$", fontsize=9, color="#78909C")

# sphere circle
ax2.plot(R * np.cos(theta_full), R * np.sin(theta_full),
         color=TEAL_DARK, lw=2.2, zorder=3)
ax2.text(0.55, 1.08, r"sphere $\|\mathbf{h}\|=\sqrt{d}$",
         fontsize=8.5, color=GREY_DASH)

# sample points on sphere
n_pts = 12
pt_angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False) + np.pi / 8
px = R * np.cos(pt_angles)
py = R * np.sin(pt_angles)
ax2.plot(px, py, "o", color=TEAL_MED, ms=6, zorder=5)

# tangent velocity arrows (single-headed)
for ang in pt_angles:
    x0, y0 = R * np.cos(ang), R * np.sin(ang)
    # tangent = perpendicular to radius (counter-clockwise)
    tx, ty = -np.sin(ang), np.cos(ang)
    scale = 0.22
    ax2.annotate(
        "", xy=(x0 + tx * scale, y0 + ty * scale),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=TEAL_DARK,
                        mutation_scale=11, lw=1.4),
        zorder=6,
    )
    # label a few
    if ang in pt_angles[:6:2]:
        ax2.text(x0 + tx * 0.38, y0 + ty * 0.38,
                 r"$v$", fontsize=8.5, color=TEAL_DARK,
                 ha="center", va="center", zorder=7)

# LN projection arrows (radially inward → outward back to sphere)
for ang in pt_angles[::3]:
    x0, y0 = R * np.cos(ang), R * np.sin(ang)
    # small inward displacement then LN pushes back outward
    inner = 0.78
    xi, yi = inner * np.cos(ang), inner * np.sin(ang)
    ax2.annotate(
        "", xy=(x0, y0), xytext=(xi, yi),
        arrowprops=dict(arrowstyle="-|>", color=RED,
                        mutation_scale=10, lw=1.2),
        zorder=6,
    )

ax2.set_title(
    "With LayerNorm\n"
    r"$\it{LN\ holds\ state\ on\ sphere;\ forces\ remain\ active,}$"
    "\n"
    r"$\it{\gamma_{\rm eff}<\gamma}$",
    fontsize=11, fontweight="bold", color=TEAL_DARK, pad=6,
    linespacing=1.5,
)
ax2.set_xticks([])
ax2.set_yticks([])
for sp in ax2.spines.values():
    sp.set_visible(False)

# ─────────────────────────────────────────────────────────────────
#  LEGEND
# ─────────────────────────────────────────────────────────────────
legend_elements = [
    Line2D([0], [0], color=BLUE_DARK, lw=2,
           marker=">", markersize=7, label="Trajectory (no LN)"),
    Line2D([0], [0], color=TEAL_DARK, lw=2,
           marker=">", markersize=7, label="Trajectory (with LN)"),
    Line2D([0], [0], color=RED, lw=1.5,
           marker=">", markersize=7, label="LN projection (radial)"),
    Line2D([0], [0], color=BLUE_MED, lw=1.5,
           marker=">", markersize=7, label=r"$v$  Velocity (tangent)"),
    Line2D([0], [0], color=GREY_DASH, lw=1.3, linestyle="--",
           label=r"Sphere $\|\mathbf{h}\|=\sqrt{d}$"),
    mpatches.Patch(facecolor="white", edgecolor="#B0BEC5",
                   label=r"Potential $V_\theta(h)$ contours"),
]
fig.legend(handles=legend_elements, loc="lower center",
           ncol=3, fontsize=9, frameon=True,
           bbox_to_anchor=(0.5, 0.01),
           fancybox=True, framealpha=0.9)

fig.subplots_adjust(bottom=0.14, top=0.88, left=0.03, right=0.97)

out = "/Users/dimitargueorguiev/git/ml/semsimula-paper/companion_notes/images/ln_counter_damping_mechanism.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved to {out}")
