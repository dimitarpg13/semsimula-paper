"""Figures for semantic_templates/Template_Firing_Mechanisms.md (2026-11-27).

No trained model is read here. Panels marked "schematic" are drawn from made-up clouds and trajectories chosen to show a
mechanism; the operating-point figure combines the textbook equal-variance Gaussian model with measured AUROCs and detection
rates quoted in the note. Writes into figures/template_firing/:
  fire_rule.png      the present firing rule (largest rate, rate threshold, reach) in a two-dimensional discriminant plane,
                     on verbatim premises and on premises whose form the words do not show (schematic)
  fire_bound.png     (a) detection at a fixed false-fire rate against the AUROC of the hardest rival, for 1%, 5% and 10%,
                     with the measured operating points; (b) the separation a firing rule needs
  fire_evidence.png  (a) evidence accumulated over successive readings, with two stopping bounds (schematic);
                     (b) how much accumulation gains when the readings are correlated
  fire_hazard.png    firing as a reaction with a rate: probability of at least one firing against the integration time,
                     for a valid premise and for its twin (schematic rates)
  fire_basin.png     firing inside the dynamics: a trajectory captured by a template well against one that passes through,
                     and the dwell-time criterion (schematic)
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from scipy.stats import norm

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "figure.dpi": 150})
INK, MUTED, GREY, PALE = "#1F2937", "#6B7280", "#9CA3AF", "#F3F4F6"
C = ["#3B82F6", "#DC2626", "#059669", "#F59E0B", "#7C3AED"]          # fixed categorical order
OUT = Path(__file__).resolve().parent / "template_firing"; OUT.mkdir(exist_ok=True)


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)


# 1. the present rule in a discriminant plane (schematic)
m = {"MP": np.array([0.0, 0.0]), "AC (twin)": np.array([1.6, 0.4]), "foil": np.array([-0.4, 2.6]), "ordinary text": np.array([3.4, 2.8])}
col = {"MP": C[0], "AC (twin)": C[1], "foil": C[3], "ordinary text": GREY}
xx, yy = np.meshgrid(np.linspace(-3, 6, 400), np.linspace(-3, 5.5, 400))
dens = {k: np.exp(-((xx - v[0]) ** 2 + (yy - v[1]) ** 2) / 2) for k, v in m.items()}
lam = dens["MP"] / sum(dens.values())
reach = np.sqrt(-2 * np.log(0.01))                                    # 99% reach of a unit Gaussian in two dimensions, about 3.03
rng = np.random.default_rng(3)
fig, axs = plt.subplots(1, 2, figsize=(13, 5.6))
for ax, (title, shift, spread) in zip(axs, [("(a) verbatim premises: valid MP premises sit in their cloud", np.zeros(2), 0.55),
                                            ("(b) form not shown by the words: valid premises drift toward the twin", np.array([0.8, 0.6]), 1.0)]):
    ax.contourf(xx, yy, lam, levels=[0.6, 1.01], colors=[C[0]], alpha=0.10)
    ax.contour(xx, yy, lam, levels=[0.6], colors=[C[0]], linewidths=1.4, linestyles="--")
    ax.text(-2.9, -2.85, "the rate region runs on far from every cloud:\nthe reach condition cuts it off", fontsize=8.5, color=MUTED)
    for k, v in m.items():
        ax.add_patch(Circle(v, 1.0, fc="none", ec=col[k], lw=1.6))
        ax.add_patch(Circle(v, reach, fc="none", ec=col[k], lw=0.6, ls=":", alpha=0.7))
        ax.plot(*v, "o", color=col[k], ms=7, mec="white", mew=1.5)
        ax.text(v[0] - 0.75, v[1] + 1.08, k, color=col[k] if k != "ordinary text" else MUTED, fontsize=10, fontweight="bold")
    pts = m["MP"] + shift + rng.normal(0, spread, (40, 2))
    tw = m["AC (twin)"] + rng.normal(0, 0.55 if spread < 1 else 1.0, (40, 2)) - (0 if spread < 1 else shift * 0.4)
    ax.scatter(*pts.T, s=16, color=C[0], alpha=0.8, label="valid premises (MP)", zorder=3, edgecolors="white", linewidths=0.4)
    ax.scatter(*tw.T, s=16, color=C[1], alpha=0.8, marker="s", label="their twins (AC)", zorder=3, edgecolors="white", linewidths=0.4)
    ax.set_xlim(-3, 6); ax.set_ylim(-3, 5.5); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlabel("discriminant direction 1"); ax.set_ylabel("discriminant direction 2")
    clean(ax)
axs[0].plot([], [], "--", color=C[0], label="rate of MP reaches the threshold (inside: fires)")
axs[0].plot([], [], ":", color=GREY, label="99% reach of each cloud")
fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, fontsize=9.5, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
fig.suptitle("Schematic: the present firing rule in a two-dimensional discriminant plane", x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(OUT / "fire_rule.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
print("wrote fire_rule.png")

# 2. the operating-point bound: equal-variance Gaussian scores
auc = np.linspace(0.5, 0.9999, 600)
dp = np.sqrt(2) * norm.ppf(auc)                                       # AUROC = Phi(d' / sqrt 2)
fig, axs = plt.subplots(1, 2, figsize=(13, 4.9))
for a, c, ls in [(0.01, C[0], "-"), (0.05, C[2], "--"), (0.10, C[4], ":")]:
    axs[0].plot(auc, norm.cdf(dp - norm.ppf(1 - a)), ls, color=c, lw=2, label=f"{int(a * 100)}% false fires")
meas = [("SR2-NP", 0.756), ("G2", 0.745), ("L4", 0.753), ("GPT-2 W2", 0.754)]
for name, v in meas:
    axs[0].plot(v, norm.cdf(np.sqrt(2) * norm.ppf(v) - norm.ppf(0.99)), "o", color=C[1], ms=6, mec="white", mew=1.2, zorder=4)
axs[0].annotate("task A, valid against twin:\nAUROC 0.745 to 0.756 (measured)\npredicts 8 to 9% detection at 1%;\nmeasured 0 to 2%", xy=(0.752, 0.09),
                xytext=(0.56, 0.42), fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=GREY))
axs[0].axhspan(0.92, 0.97, xmin=0.88, xmax=1.0, color=C[2], alpha=0.25)
axs[0].annotate("verbatim premises:\n92 to 97% at 1% (measured)", xy=(0.94, 0.945), xytext=(0.66, 0.80), fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="-", color=GREY))
axs[0].set_xlabel("AUROC of the template's score against its hardest rival"); axs[0].set_ylabel("detection (share of valid premises that fire)")
axs[0].set_title("(a) what a fixed false-fire rate leaves", loc="left"); axs[0].set_xlim(0.5, 1.0); axs[0].set_ylim(0, 1.02)
axs[0].legend(frameon=False, fontsize=9, loc="upper left")
need = [(a, norm.ppf(1 - a) + norm.ppf(0.9)) for a in (0.01, 0.05, 0.10)]
for i, (a, d) in enumerate(need):
    axs[1].bar(i, norm.cdf(d / np.sqrt(2)), color=[C[0], C[2], C[4]][i], width=0.45, edgecolor="white", linewidth=2)
    axs[1].text(i, norm.cdf(d / np.sqrt(2)) + 0.004, f"AUROC {norm.cdf(d / np.sqrt(2)):.3f}\n(separation {d:.2f})", ha="center", fontsize=9, color=INK)
axs[1].axhline(0.755, color=C[1], lw=1.4, ls="--"); axs[1].text(0.5, 0.758, "measured, task A\nvalid against twin", color=INK, fontsize=9, ha="center", va="bottom")
axs[1].set_xticks(range(3)); axs[1].set_xticklabels(["1% false fires", "5%", "10%"]); axs[1].set_ylim(0.7, 1.02)
axs[1].set_ylabel("AUROC needed for 90% detection"); axs[1].set_title("(b) the separation a firing rule needs", loc="left")
for ax in axs:
    clean(ax)
fig.tight_layout(); fig.savefig(OUT / "fire_bound.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
print("wrote fire_bound.png", [round(norm.cdf(np.sqrt(2) * norm.ppf(v) - norm.ppf(0.99)), 3) for _, v in meas])

# 3. evidence accumulation (schematic) and the effect of correlation
rng = np.random.default_rng(7)
alpha, beta = 0.01, 0.10
A, B = np.log((1 - beta) / alpha), np.log(beta / (1 - alpha))
d1 = 0.95                                                             # per-reading separation of the task A reader
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
for lab, mu, c, mk in [("valid premise", d1 ** 2 / 2, C[0], "-"), ("its twin", -d1 ** 2 / 2, C[1], "--")]:
    for r in range(6):
        steps = rng.normal(mu, d1, 40); path = np.concatenate([[0], np.cumsum(steps)])
        hit = np.argmax((path >= A) | (path <= B)) if np.any((path >= A) | (path <= B)) else len(path) - 1
        axs[0].plot(range(hit + 1), path[:hit + 1], mk, color=c, lw=1.2, alpha=0.85, label=lab if r == 0 else None)
        axs[0].plot(hit, path[hit], "o", color=c, ms=4)
axs[0].axhline(A, color=INK, lw=1); axs[0].axhline(B, color=INK, lw=1)
axs[0].text(0.3, A + 0.2, "fire (upper bound, false fires at most 1%)", fontsize=9, color=INK)
axs[0].text(0.3, B - 0.6, "stay silent (lower bound)", fontsize=9, color=INK)
axs[0].set_xlabel("number of readings combined (independent, schematic)"); axs[0].set_ylabel("accumulated log likelihood ratio")
axs[0].set_title("(a) firing by accumulating evidence", loc="left"); axs[0].legend(frameon=False, fontsize=9, loc="center right")
n = np.arange(1, 17)
for rho, c, ls in [(0.0, C[2], "-"), (0.5, C[3], "--"), (0.9, C[1], ":")]:
    neff = n / (1 + (n - 1) * rho)
    axs[1].plot(n, norm.cdf(d1 * np.sqrt(neff) / np.sqrt(2)), ls, color=c, lw=2, marker="o", ms=3, label=f"correlation {rho:g} between readings")
axs[1].axhline(norm.cdf((norm.ppf(0.99) + norm.ppf(0.9)) / np.sqrt(2)), color=INK, lw=1)
axs[1].text(16, 0.988, "needed for 90% at 1%", fontsize=9, color=INK, ha="right", va="top")
axs[1].set_xlabel("number of readings combined"); axs[1].set_ylabel("AUROC of the combined score")
axs[1].set_title("(b) correlated readings gain little", loc="left"); axs[1].legend(frameon=False, fontsize=9, loc="lower right")
axs[1].set_ylim(0.7, 1.0)
for ax in axs:
    clean(ax)
fig.tight_layout(); fig.savefig(OUT / "fire_evidence.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
print("wrote fire_evidence.png", round(A, 2), round(B, 2))

# 4. firing as a reaction with a rate (schematic rates)
t = np.linspace(0, 10, 300)
fig, ax = plt.subplots(figsize=(7.6, 4.6))
for lab, lamv, c, ls in [("valid premise, rate 0.60", 0.60, C[0], "-"), ("its twin, rate 0.35", 0.35, C[1], "--"), ("ordinary text, rate 0.02", 0.02, GREY, ":")]:
    ax.plot(t, 1 - np.exp(-0.5 * lamv * t), ls, color=c, lw=2, label=lab)
ax.axvline(2.0, color=INK, lw=1); ax.text(2.1, 0.93, "short window:\nfew fires, few false fires", fontsize=9, color=INK, va="top")
ax.axvline(8.0, color=INK, lw=1); ax.text(7.9, 0.12, "long window:\nboth fire", fontsize=9, color=INK, ha="right")
ax.set_xlabel("integration time, in units of the base rate"); ax.set_ylabel("probability of at least one firing")
ax.set_title("Schematic: a template that fires at a rate", loc="left"); ax.legend(frameon=False, fontsize=9, loc="center right")
clean(ax); fig.tight_layout(); fig.savefig(OUT / "fire_hazard.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
print("wrote fire_hazard.png")

# 5. firing inside the dynamics: basin capture (schematic)
xx, yy = np.meshgrid(np.linspace(-3, 3, 300), np.linspace(-2.2, 2.2, 300))
V = -1.6 * np.exp(-(xx ** 2 / 0.9 + yy ** 2 / 0.5)) + 0.08 * (xx ** 2 + yy ** 2)
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.35, 1]})
axs[0].contourf(xx, yy, V, levels=14, cmap="Greys", alpha=0.55)
axs[0].contour(xx, yy, xx ** 2 / 0.9 + yy ** 2 / 0.5, levels=[1.0], colors=[C[0]], linewidths=1.8)
tt = np.linspace(0, 1, 200)
cap = np.c_[-2.6 + 2.6 * (1 - np.exp(-5 * tt)) + 0.15 * np.sin(14 * tt) * np.exp(-3 * tt), 1.6 * np.exp(-5 * tt) + 0.12 * np.cos(14 * tt) * np.exp(-3 * tt)]
pas = np.c_[-2.7 + 5.4 * tt, 0.9 - 1.8 * tt + 0.15 * np.sin(6 * np.pi * tt)]
axs[0].plot(*cap.T, "-", color=C[0], lw=2, label="captured: stays in the basin, the template fires")
axs[0].plot(*pas.T, "--", color=C[1], lw=2, label="passes through: in the basin briefly, no firing")
axs[0].text(0.05, 0.75, "template basin", color=INK, fontsize=9)
axs[0].set_xticks([]); axs[0].set_yticks([]); axs[0].set_aspect("equal")
axs[0].set_title("(a) schematic: firing as capture by a template well", loc="left", fontsize=11)
axs[0].legend(frameon=False, fontsize=9, loc="lower left"); clean(axs[0])
inside_c = (cap[:, 0] ** 2 / 0.9 + cap[:, 1] ** 2 / 0.5) <= 1
inside_p = (pas[:, 0] ** 2 / 0.9 + pas[:, 1] ** 2 / 0.5) <= 1
axs[1].step(tt, inside_c.astype(float) + 0.04, where="post", color=C[0], lw=2, label="captured")
axs[1].step(tt, inside_p.astype(float) * 0.96, where="post", color=C[1], lw=2, ls="--", label="passes through")
axs[1].axvspan(tt[np.argmax(inside_c)], tt[np.argmax(inside_c)] + 0.45, ymin=0.05, ymax=0.95, color=C[0], alpha=0.12)
axs[1].annotate("", xy=(tt[np.argmax(inside_c)] + 0.45, 0.82), xytext=(tt[np.argmax(inside_c)], 0.82), arrowprops=dict(arrowstyle="<->", color=INK, lw=1))
axs[1].text(tt[np.argmax(inside_c)] + 0.225, 0.85, "dwell time that must pass", fontsize=9, color=INK, ha="center")
axs[1].set_yticks([0, 1]); axs[1].set_yticklabels(["outside", "inside"]); axs[1].set_xlabel("time along the trajectory (depth or substeps)")
axs[1].set_title("(b) the dwell-time criterion", loc="left", fontsize=11); axs[1].legend(frameon=False, fontsize=9, loc="lower right", bbox_to_anchor=(1.0, 0.12))
clean(axs[1]); fig.tight_layout(); fig.savefig(OUT / "fire_basin.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
print("wrote fire_basin.png")
