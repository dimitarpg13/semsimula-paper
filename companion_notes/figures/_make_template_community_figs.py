"""Figures for Community_of_Semantic_Templates_for_Native_CoT.md.

PNGs into figures/template_community/. The schematic panels draw the
constructions of the note; the measured panel reads the MT probe's lift
matrices from PMX's results folder (debug/mode_transition_probe.py).

  tc_identification.png  -- the three-way identification: a Feng DP step, a
                            template firing and a v3-operator event are one
                            object; what each formalism names
  tc_thought.png         -- a thought as a sequence of firings on a growing
                            structure: templates match slots on existing
                            particles, fire, and emit a new particle that later
                            templates match; the occupation trace beneath
  tc_mode_as_template.png -- a Poisson mode read as a learned pattern-matching
                            region: centre, width, binding radius from the
                            threshold; what it has and lacks against a template
  tc_transitions.png     -- measured: PMX's mode-to-mode lift at lag 1 against
                            the capacity-matched drift null (MT, protocol 5.24)

Run:  MPLBACKEND=Agg python3 _make_template_community_figs.py
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "figure.dpi": 150})
GREEN, BLUE, RED, PURPLE, GREY, ORANGE = "#22C55E", "#3B82F6", "#EF4444", "#8B5CF6", "#9CA3AF", "#F59E0B"
OUT = Path(__file__).parent / "template_community"
OUT.mkdir(exist_ok=True)
RES = Path(__file__).resolve().parents[2] / "notebooks/conservative_arch/scaleup/results"
PMX_RUN = ("cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_pm64_pmx16_sr2_"
           "L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn")


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight"); plt.close(fig); print("wrote", OUT / name)


def box(ax, xy, w, h, text, col, fs=9, ec="k"):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02", fc=col, ec=ec, lw=0.8, alpha=0.95))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, a, b, col="k", lw=1.2, style="->"):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle=style, lw=lw, color=col))


# --------------------------------------------------------- 1. identification
def fig_identification():
    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 5.2); ax.axis("off")
    cols = ["#DBEAFE", "#BBF7D0", "#FEF3C7"]
    heads = ["Feng et al. 2023: a DP step", "semantic template (manuscript, Def. 8)", "book §10.9.5: a v3-operator event"]
    rows = [("reads", "$K$ earlier states $\\mathrm{dp}(h(n,i))$\nand $J$ input tokens", "a pattern-matching region $\\mathfrak{P}(\\mathfrak{T})$:\na tree with $r$ dummy slots $X$", "$r$ existing particles\n(rank $r$)"),
            ("fires when", "always, in topological order", "the latch $\\mu$: overlap above $\\Theta$,\na binding radius in $x^\\ast$ units", "the operator's precondition"),
            ("writes", "one new state $\\mathrm{dp}(i)$", "the inference region $\\mathfrak{I}(\\mathfrak{T})$:\none new structure", "one new particle, typed,\nwith provenance"),
            ("depth", "$+1$ per step", "$+1$ per firing", "$+1$ per event"),
            ("order", "the DAG's topological order", "undefined: $\\mathfrak{I}$ has no body (¶185)", "non-abelian: $G$-valued tag")]
    xs = [3.0, 6.6, 10.2]
    for j, h in enumerate(heads):
        box(ax, (xs[j] - 1.65, 4.45), 3.3, 0.6, h, cols[j], fs=9.5)
    for i, (lab, *cells) in enumerate(rows):
        y = 3.75 - i * 0.78
        ax.text(1.2, y + 0.3, lab, ha="right", va="center", fontsize=9.5, fontweight="bold")
        for j, c in enumerate(cells):
            col = "#FEE2E2" if (i == 4 and j == 1) else "white"
            box(ax, (xs[j] - 1.65, y), 3.3, 0.62, c, col, fs=8.3, ec="#9CA3AF")
    ax.text(6.75, 0.1, "One object, three names. The identification is exact on the first four rows and fails on the fifth: the template's inference region, where order would live, is the part the manuscript leaves empty.",
            ha="center", fontsize=8.8)
    ax.set_title("A DP step, a template firing and a v3-operator event are the same object")
    save(fig, "tc_identification.png")


# ------------------------------------------------------------- 2. a thought
def fig_thought():
    fig, ax = plt.subplots(2, 1, figsize=(13.5, 7.2), gridspec_kw={"height_ratios": [2.1, 1]})
    a = ax[0]; a.set_xlim(0, 13.5); a.set_ylim(0, 6.2); a.axis("off")
    # input particles
    inputs = [(1.0, 1.2, "$s_1$"), (2.4, 1.2, "$s_2$"), (3.8, 1.2, "$s_3$"), (5.2, 1.2, "$s_4$")]
    for x, y, t in inputs:
        a.add_patch(Circle((x, y), 0.32, fc="#E5E7EB", ec="k")); a.text(x, y, t, ha="center", va="center", fontsize=10)
    a.text(3.1, 0.45, "input structures (the question)", ha="center", fontsize=9, color=GREY)
    # firings: template T1 matches s1,s2 -> p1 ; T2 matches s3,p1 -> p2 ; T3 matches p2,s4 -> p3 (answer)
    steps = [((1.0, 2.4), (3.0, 3.2), "$\\mathfrak{T}_1$", "$p_1$", BLUE),
             ((3.8, 3.0), (6.0, 4.1), "$\\mathfrak{T}_2$", "$p_2$", GREEN),
             ((6.0, 5.2), (9.0, 5.0), "$\\mathfrak{T}_3$", "$p_3$ = answer", ORANGE)]
    pos = {"s1": (1.0, 1.2), "s2": (2.4, 1.2), "s3": (3.8, 1.2), "s4": (5.2, 1.2)}
    # T1
    box(a, (1.6, 2.5), 1.4, 0.7, "$\\mathfrak{T}_1$ fires\nslots $X_1{=}s_1, X_2{=}s_2$", "#DBEAFE", fs=8)
    arrow(a, (1.0, 1.52), (1.9, 2.5), BLUE); arrow(a, (2.4, 1.52), (2.5, 2.5), BLUE)
    a.add_patch(Circle((3.6, 3.4), 0.34, fc="#BFDBFE", ec="k")); a.text(3.6, 3.4, "$p_1$", ha="center", va="center", fontsize=10)
    arrow(a, (3.0, 2.95), (3.35, 3.2), BLUE)
    # T2
    box(a, (4.3, 3.3), 1.5, 0.7, "$\\mathfrak{T}_2$ fires\nslots $X_1{=}s_3, X_2{=}p_1$", "#D1FAE5", fs=8)
    arrow(a, (3.8, 1.52), (4.6, 3.3), GREEN); arrow(a, (3.94, 3.5), (4.3, 3.6), GREEN)
    a.add_patch(Circle((6.5, 4.2), 0.34, fc="#BBF7D0", ec="k")); a.text(6.5, 4.2, "$p_2$", ha="center", va="center", fontsize=10)
    arrow(a, (5.8, 3.8), (6.2, 4.05), GREEN)
    # T3
    box(a, (7.3, 4.0), 1.5, 0.7, "$\\mathfrak{T}_3$ fires\nslots $X_1{=}p_2, X_2{=}s_4$", "#FEF3C7", fs=8)
    arrow(a, (6.84, 4.25), (7.3, 4.3), ORANGE); arrow(a, (5.2, 1.52), (7.6, 4.0), ORANGE)
    a.add_patch(Circle((9.6, 4.6), 0.4, fc="#FDE68A", ec="k", lw=1.5)); a.text(9.6, 4.6, "$p_3$", ha="center", va="center", fontsize=10)
    a.text(9.6, 3.95, "answer: emitted to yield", ha="center", fontsize=8.5)
    arrow(a, (8.8, 4.45), (9.2, 4.55), ORANGE)
    a.text(11.6, 5.3, "each firing:\n• reads $r$ particles (slots)\n• writes one new particle\n• raises the depth by one\n• $p_k$ becomes matchable", ha="left", va="top", fontsize=8.8)
    a.text(11.6, 2.4, "the order $\\mathfrak{T}_1 \\to \\mathfrak{T}_2 \\to \\mathfrak{T}_3$\nis forced by the slots:\n$\\mathfrak{T}_2$ needs $p_1$ to exist.\nWhether $X_1{=}s_3, X_2{=}p_1$\ndiffers from the reverse is\nthe inference region's job.", ha="left", va="top", fontsize=8.8, color=RED)
    a.set_title("A thought as a sequence of template firings on a growing structure")
    # occupation trace
    b = ax[1]
    T = 40; K = 6
    lam = 0.88
    fires = {8: 0, 14: 1, 20: 2, 24: 3, 31: 4}      # token -> mode raised (T1..T3 raise modes; answer raises mode 4)
    E = np.full((T, K), 0.03);
    for t, v in fires.items(): E[t, v] = 1.0
    n = np.zeros((T + 1, K))
    for t in range(T): n[t + 1] = lam * n[t] + E[t]
    cols = [BLUE, GREEN, ORANGE, PURPLE, RED, GREY]
    for v in range(5):
        b.plot(np.arange(1, T + 1), n[1:, v], color=cols[v], lw=2, label=f"mode {v}")
    for t in fires: b.axvline(t, color="k", ls=":", lw=0.7)
    b.set_xlabel("token position"); b.set_ylabel("occupation $\\bar n_v$")
    b.set_title("The same thought as the registers see it: a firing history, order-blind up to decay")
    b.legend(fontsize=8, ncol=5, loc="upper left")
    save(fig, "tc_thought.png")


# ------------------------------------------------- 3. a mode as a template
def fig_mode_as_template():
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [1, 1.3]})
    a = ax[0]
    a.set_aspect("equal"); a.set_xlim(-3.2, 3.2); a.set_ylim(-3.2, 3.2)
    xx, yy = np.meshgrid(np.linspace(-3.2, 3.2, 200), np.linspace(-3.2, 3.2, 200))
    a.contourf(xx, yy, np.exp(-(xx ** 2 + yy ** 2)), levels=12, cmap="Greens", alpha=0.85)
    for Th, col, lab in ((0.9, RED, "$\\Theta{=}0.9$"), (0.5, ORANGE, "$\\Theta{=}0.5$"), (0.1, BLUE, "$\\Theta{=}0.1$")):
        r = np.sqrt(np.log(1 / Th))
        a.add_patch(Circle((0, 0), r, fc="none", ec=col, lw=1.6)); a.text(-r * 0.75 - 0.9, -r * 0.75 - 0.15, lab, color=col, fontsize=8.5)
    a.plot(0, 0, "k+", ms=10); a.text(0.12, -0.35, "$\\mu_v$", fontsize=10)
    pts = [(0.5, 0.3, "fires at all three"), (1.3, -0.6, "fires at $\\Theta \\leq 0.5$"), (2.4, 1.1, "fires at $\\Theta{=}0.1$ only")]
    for x, y, t in pts:
        a.plot(x, y, "o", color="k", ms=5); a.text(x + 0.12, y + 0.12, t, fontsize=7.5)
    a.set_title("A Poisson mode as a pattern-matching region")
    a.set_xlabel("semantic space (units of $1/\\kappa_v$)")
    b = ax[1]; b.set_xlim(0, 10); b.set_ylim(0, 6); b.axis("off")
    rows = [("pattern: a density $f_p$ over $\\Sigma$", "the mode $\\mu_v, \\kappa_v$: a Gaussian cloud", GREEN),
            ("trigger: latch $\\mu$ at threshold $\\Theta$", "overlap $E_v(t) > \\Theta$; $\\Theta$ is a radius", GREEN),
            ("firing history", "the occupation $\\bar n_v(t)$, decaying with $\\lambda_v$", GREEN),
            ("learned from data", "yes: $\\mu_v, \\kappa_v, \\lambda_v$, depths", GREEN),
            ("slots: dummy particles $X$", "none: one centre, no internal structure", RED),
            ("inference region: writes a structure", "none: deepens a well (a force), writes nothing", RED),
            ("transitions between templates", "measured: MT, Figure 4", ORANGE)]
    for i, (l, r, c) in enumerate(rows):
        y = 5.3 - i * 0.78
        b.text(0.1, y, l, fontsize=9, va="center")
        box(b, (4.3, y - 0.27), 5.5, 0.56, r, {GREEN: "#BBF7D0", RED: "#FEE2E2", ORANGE: "#FEF3C7"}[c], fs=8.5, ec="#9CA3AF")
    b.text(0.1, 5.85, "a template has …", fontsize=9, fontweight="bold"); b.text(4.4, 5.85, "a trained Poisson mode has …", fontsize=9, fontweight="bold")
    save(fig, "tc_mode_as_template.png")


# ------------------------------------------------------------ 4. measured
def fig_transitions():
    f = RES / PMX_RUN / "mode_transition_probe_pmx_matrices.npz"
    j = RES / PMX_RUN / "mode_transition_probe_pmx.json"
    if not (f.exists() and j.exists()):
        print("skip tc_transitions.png: run debug/mode_transition_probe.py pmx first"); return
    import json
    d = np.load(f); mi = json.load(open(j))["mi"]
    Lf, Ld = d["lift"], d["lift_drift"]
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    for k, (M, t) in enumerate(((Lf, "measured lift $L_1(u \\to v)$"), (Ld, "capacity-matched drift null"))):
        im = ax[k].imshow(np.log2(np.clip(np.nan_to_num(M, nan=1.0), 0.25, 4)), cmap="RdBu_r", vmin=-2, vmax=2, origin="lower")
        ax[k].set_title(t); ax[k].set_xlabel("v fires at t + 1"); ax[k].set_ylabel("u fires at t")
    plt.colorbar(im, ax=ax[:2], fraction=0.025, label="$\\log_2$ lift (clipped to [1/4, 4])")
    lags = sorted(int(x) for x in mi)
    I = [mi[str(l)]["I"] for l in lags]; Id = [mi[str(l)]["I_drift"] for l in lags]; Is = [mi[str(l)]["I_shuffle"] for l in lags]
    ex = [mi[str(l)]["excess"] for l in lags]; lo = [mi[str(l)]["lo"] for l in lags]; hi = [mi[str(l)]["hi"] for l in lags]
    ax[2].plot(lags, I, "o-", color=GREEN, lw=2, label="data")
    ax[2].plot(lags, Id, "s--", color=BLUE, lw=1.5, label="drift null (64 PCs)")
    ax[2].plot(lags, Is, "x:", color=GREY, lw=1.5, label="shuffle null")
    ax[2].set_xscale("log", base=2); ax[2].set_xticks(lags); ax[2].set_xticklabels(lags)
    ax[2].set_xlabel("lag $\\Delta$ (tokens)"); ax[2].set_ylabel("summed mutual information (bits)")
    ax[2].set_title("Firing information against lag")
    ax2 = ax[2].twinx(); ax2.errorbar(lags, ex, yerr=[np.array(ex) - np.array(lo), np.array(hi) - np.array(ex)], fmt="d", color=RED, capsize=3, label="excess over drift")
    ax2.axhline(0, color=RED, lw=0.6, ls=":"); ax2.set_ylabel("excess (bits)", color=RED)
    ax[2].legend(fontsize=7.5, loc="upper right")
    save(fig, "tc_transitions.png")


if __name__ == "__main__":
    fig_identification(); fig_thought(); fig_mode_as_template(); fig_transitions()
