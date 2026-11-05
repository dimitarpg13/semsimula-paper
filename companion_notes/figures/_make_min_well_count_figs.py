#!/usr/bin/env python3
"""Figures for Minimum_Well_Count_for_Conservative_Models.md.

    python3 _make_min_well_count_figs.py        # writes figures/min_well_count/*.png

Data, read from the result files named beside each figure (no checkpoints needed):
- well probes:        notebooks/conservative_arch/scaleup/results/dead_wells/well_probe_<arm>.txt
- keep-k curves:      .../dead_wells/keep_k_<arm>_L<layer>[_renorm].txt
- temperature sweeps: .../dead_wells/vtheta_weight_logits_<arm>.txt (sweep=1)
- well-count fits:    .../well_count/well_count_fit_<arm>[_L<layer>].txt
- training curves:    the K2-L8-CODE, L8-CODE and L4-CODE Cell 6 logs in their run folders.
A figure whose data file is missing is skipped with a note.
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GREEN, BLUE, RED, PURPLE, GREY, ORANGE = "#22C55E", "#3B82F6", "#EF4444", "#8B5CF6", "#9CA3AF", "#F59E0B"
INK, MUTED = "#1F2937", "#6B7280"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 150})

HERE = Path(__file__).parent
OUT = HERE / "min_well_count"
OUT.mkdir(exist_ok=True)
RES = HERE.parents[1] / "notebooks/conservative_arch/scaleup/results"
DW, WC = RES / "dead_wells", RES / "well_count"
CORPUS_COL = {"code": BLUE, "owt": ORANGE, "ts": GREEN}


def probe(arm):
    """Per layer: median n_eff, and the share of the force weight sum by well (percent)."""
    t = (DW / f"well_probe_{arm}.txt").read_text()
    neff = []
    for line in t[t.index("medians per layer"):].splitlines()[2:]:
        p = line.split()
        if not p or not p[0].isdigit():
            break
        neff.append(float(p[5]))
    force = [[float(v) for v in m.group(2).split()] for m in re.finditer(r"^  layer (\d+): force\s+(.*?)\s+\|", t, re.M)]
    return np.array(neff), np.array(force)


def keep_k(arm, layer, renorm=False):
    f = DW / f"keep_k_{arm}_L{layer}{'_renorm' if renorm else ''}.txt"
    if not f.exists():
        return None
    rows = [l.split() for l in f.read_text().splitlines()[2:]]
    ks = [int(r[0]) for r in rows]
    ppl = [float(next(x for x in r[1:] if re.fullmatch(r"\d+\.\d+", x))) for r in rows]
    return np.array(ks), np.array(ppl)


def evals(path):
    d = {}
    for line in open(path, errors="replace"):
        m = re.search(r">>> EVAL step ([\d,]+)\s+val_loss=([\d.]+)", line)
        if m:
            d[int(m.group(1).replace(",", ""))] = float(m.group(2))
    return d


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


# ---------------------------------------------------------------- Figure 1: n_eff by layer, depth and corpus
def fig_neff():
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1, 1.25]})
    for arm, lab, col, mk in (("so_code", "SO-CODE (L = 2)", GREY, "o"), ("l4code_code_step18000", "L4-CODE (L = 4)", PURPLE, "s"),
                              ("l8code_code_step18000", "L8-CODE (L = 8)", BLUE, "D")):
        n, _ = probe(arm)
        a.plot(range(len(n)), n, marker=mk, color=col, lw=2, ms=7, label=lab)
    a.axhline(8, color=MUTED, ls=":", lw=1)
    a.text(7, 8.15, "bank size K = 8", ha="right", va="bottom", color=MUTED, fontsize=9)
    a.set_xlabel("layer")
    a.set_ylabel("effective well count $n_{\\mathrm{eff}}$ (median over tokens)")
    a.set_ylim(0, 8.9)
    a.set_title("(a) the depth ladder on code", loc="left", fontsize=11)
    a.legend(frameon=False, fontsize=9, loc="center right")

    arms = (("so_code", "SO", "code"), ("fock_code", "FOCK", "code"), ("so_owt", "SO", "owt"),
            ("f31_owt", "F3.1", "owt"), ("g2_owt", "G2", "owt"), ("so_ts", "SO", "ts"))
    x = np.arange(len(arms))
    l0 = [probe(a_)[0][0] for a_, _, _ in arms]
    l1 = [probe(a_)[0][1] for a_, _, _ in arms]
    cols = [CORPUS_COL[c] for _, _, c in arms]
    b.bar(x - 0.2, l0, 0.38, color=cols, alpha=0.35, edgecolor="white", lw=2)
    b.bar(x + 0.2, l1, 0.38, color=cols, edgecolor="white", lw=2)
    for i in range(len(arms)):
        b.text(x[i] - 0.2, l0[i] + 0.12, f"{l0[i]:.1f}", ha="center", fontsize=9, color=INK)
        b.text(x[i] + 0.2, l1[i] + 0.12, f"{l1[i]:.1f}", ha="center", fontsize=9, color=INK)
    b.set_xticks(x, [f"{n}\n{ {'code': 'code', 'owt': 'OWT', 'ts': 'TinyStories'}[c] }" for _, n, c in arms])
    b.axhline(8, color=MUTED, ls=":", lw=1)
    b.set_ylim(0, 8.9)
    b.set_title("(b) two-layer models on three corpora: layer 0 (pale), layer 1 (solid)", loc="left", fontsize=11)
    save(fig, "neff_by_layer.png")


# ---------------------------------------------------------------- Figure 2: force shares by well, L8-CODE
def fig_force():
    n, F = probe("l8code_code_step18000")
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    im = ax.imshow(F, cmap="Blues", vmin=0, vmax=60, aspect="auto")
    for i in range(F.shape[0]):
        for k in range(F.shape[1]):
            ax.text(k, i, f"{F[i, k]:.0f}", ha="center", va="center", fontsize=9,
                    color="white" if F[i, k] > 35 else INK)
        ax.text(F.shape[1] - 0.3, i, f"  $n_{{\\mathrm{{eff}}}}$ {n[i]:.1f}", ha="left", va="center", fontsize=9, color=INK)
    ax.set_xticks(range(F.shape[1]), [f"well {k}" for k in range(F.shape[1])], fontsize=9)
    ax.set_yticks(range(F.shape[0]), [f"layer {i}" for i in range(F.shape[0])], fontsize=9)
    ax.set_xlim(-0.5, F.shape[1] + 0.9)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("share of the force weight (%)")
    ax.set_title("L8-CODE: which wells pull the state, by layer", loc="left", fontsize=11)
    save(fig, "force_shares_l8code.png")


# ---------------------------------------------------------------- Figure 3: n_eff against the bank input's norm
def fig_temperature():
    f = DW / "vtheta_weight_logits_l8code_code_step18000.txt"
    t = f.read_text() if f.exists() else ""
    if "temperature sweep" not in t:
        print("skip temperature_sweep.png: no sweep in", f.name)
        return
    rows = [l.split() for l in t[t.index("temperature sweep"):].splitlines()[2:] if l.strip()]
    r = np.array([float(x[0]) for x in rows])
    N = np.array([[float(v) for v in x[1:]] for x in rows])
    med = []
    for line in t[t.index("layer  |x| med"):].splitlines()[1:]:
        p = line.split()
        if not p or not p[0].isdigit():
            break
        med.append((float(p[1]), float(p[5])))
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for g in range(N.shape[1]):
        col = BLUE if g == 0 else GREY
        ax.plot(r, N[:, g], color=col, lw=2.2 if g == 0 else 1.2, alpha=1 if g == 0 else 0.8, zorder=3 if g == 0 else 2)
        ax.plot(*med[g], marker="o", ms=8, color=col, mec="white", mew=1.5, zorder=4)
    ax.text(med[0][0] * 1.15, med[0][1] + 0.15, "layer 0 as trained", color=BLUE, fontsize=9, va="bottom")
    ax.text(med[1][0] * 1.1, med[1][1] + 0.35, "layers 1–7 as trained", color=MUTED, fontsize=9, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("norm of the bank input $|x|$ (every token rescaled to it)")
    ax.set_ylabel("effective well count $n_{\\mathrm{eff}}$ (median)")
    ax.axhline(8, color=MUTED, ls=":", lw=1)
    ax.set_ylim(0, 8.6)
    ax.set_title("L8-CODE: the same weights spread or concentrate with the input's size", loc="left", fontsize=11)
    save(fig, "temperature_sweep.png")


# ---------------------------------------------------------------- Figure 4: keep-k curves
def fig_keep_k():
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.4))
    for (arm, layer, lab, col) in (("so_code", 1, "SO-CODE layer 1", GREY), ("l8code_code_step18000", 4, "L8-CODE layer 4", BLUE)):
        kk = keep_k(arm, layer)
        if kk is None:
            continue
        ks, ppl = kk
        a.plot(ks, 100 * (ppl / ppl[-1] - 1), marker="o", color=col, lw=2, ms=6, label=lab)
    a.axhline(0, color=MUTED, lw=1)
    a.set_xlabel("wells kept $k$ (strongest by force share)")
    a.set_ylabel("perplexity change (%)")
    a.set_title("(a) deep layers: two wells suffice", loc="left", fontsize=11)
    a.legend(frameon=False, fontsize=9)
    a.set_xticks(range(1, 9))
    drew = False
    for (arm, lab, col) in (("so_code", "SO-CODE", GREY), ("l8code_code_step18000", "L8-CODE", BLUE)):
        for renorm, ls in ((False, "-"), (True, "--")):
            kk = keep_k(arm, 0, renorm)
            if kk is None:
                continue
            ks, ppl = kk
            b.plot(ks, ppl / ppl[-1], marker="o" if not renorm else "s", ls=ls, color=col, lw=2, ms=6,
                   label=f"{lab}, {'kept weights rescaled' if renorm else 'weights zeroed'}")
            drew = True
    b.set_yscale("log")
    b.axhline(1, color=MUTED, lw=1)
    b.set_xlabel("wells kept $k$ in layer 0")
    b.set_ylabel("perplexity / trained perplexity")
    b.set_title("(b) layer 0, wells removed without retraining", loc="left", fontsize=11)
    b.set_xticks(range(1, 9))
    if drew:
        b.legend(frameon=False, fontsize=9)
    save(fig, "keep_k.png")


# ---------------------------------------------------------------- Figure 6: what K gated linear maps can express
def fig_schematic():
    rng = np.random.default_rng(0)
    xi = np.linspace(-3, 3, 600)
    target = np.sin(1.3 * xi) + 0.25 * xi

    def fit(K, iters=4000, lr=0.05):
        A, c = rng.normal(0, 0.3, K), rng.normal(0, 0.3, K)
        G, g = np.linspace(-2, 2, K) * (K > 1), np.zeros(K)
        if K > 1:
            g = -G * np.linspace(-2, 2, K)
        P = np.concatenate([A, c, G, g])
        for _ in range(iters):
            A, c, G, g = np.split(P, 4)
            z = np.outer(xi, G) + g
            w = np.exp(z - z.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
            e = np.outer(xi, A) + c
            h = (w * e).sum(1)
            r = h - target
            dh = 2 * r / len(xi)
            dA = (dh[:, None] * w * xi[:, None]).sum(0); dc = (dh[:, None] * w).sum(0)
            dz = w * (e - h[:, None]) * dh[:, None]
            dG = (dz * xi[:, None]).sum(0); dg = dz.sum(0)
            P = P - lr * np.concatenate([dA, dc, dG, dg])
        return h, w, float(np.mean(r ** 2))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, K, col in zip(axes, (1, 2, 4), (GREY, PURPLE, BLUE)):
        h, w, mse = fit(K)
        ax.plot(xi, target, color=INK, lw=2.5)
        ax.plot(xi, h, color=col, lw=2, ls="--")
        ax.set_title(f"K = {K} wells: mean squared error {mse:.3f}", loc="left", fontsize=11)
        ax.set_xlabel("context $\\xi$ (one dimension)")
        ax.text(-2.9, 1.45, "target", color=INK, fontsize=9)
        ax.text(-2.9, 1.3, f"K = {K} fit", color=col, fontsize=9)
    axes[0].set_ylabel("state the wells hold")
    axes[0].set_ylim(-1.5, 1.6)
    fig.suptitle("Synthetic illustration: one well whose centre follows the context fits a line; more wells fit it piece by piece",
                 x=0.01, ha="left", fontsize=11, color=MUTED)
    save(fig, "gated_linear_schematic.png")


# ---------------------------------------------------------------- Figure 7: well-count fits against K
def fig_fits():
    arms = (("so_code", "SO-CODE", BLUE, 1), ("l8code", "L8-CODE", PURPLE, 7), ("so_owt", "SO-OWT", ORANGE, 1),
            ("so_ts", "SO-TS", GREEN, 1))
    got = {}
    for arm, _, _, last in arms:
        for layer in (0, last):
            f = WC / f"well_count_fit_{arm}{'' if layer == 0 else f'_L{layer}'}.txt"
            if not f.exists():
                continue
            rows = {}
            for line in f.read_text().splitlines():
                p = line.split()
                if len(p) > 6 and p[0].isdigit() and p[1].isdigit() and "," in p[2]:
                    rows[int(p[0]), int(p[1])] = float(p[3])
            got[arm, layer] = rows
    if not got:
        print("skip fit_vs_K.png: no well_count_fit_*.txt yet")
        return
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for arm, lab, col, last in arms:
        for ax, layer in ((a, 0), (b, last)):
            rows = got.get((arm, layer))
            if not rows:
                continue
            best = min(rows.values())
            for D, ls, mk in ((1, "-", "o"), (2, "--", "s")):
                ks = sorted(K for (d, K) in rows if d == D)
                if ks:
                    ax.plot(ks, [rows[D, K] - best for K in ks], ls=ls, marker=mk, color=col, lw=2, ms=6,
                            label=f"{lab}, depth {D}" if ax is a or D == 1 else None)
    for ax, title in ((a, "(a) on layer 0's bank input"), (b, "(b) on the last layer's bank input")):
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_xlabel("wells K in the fitted form")
        ax.set_title(title, loc="left", fontsize=11)
    a.set_ylabel("held-out loss above the best fit (nats)")
    a.legend(frameon=False, fontsize=8, ncol=2)
    save(fig, "fit_vs_K.png")


# ---------------------------------------------------------------- Figure 5: K2-L8-CODE in training
def fig_k2l8():
    k2l8 = evals(next(RES.glob("cfc_baoab_code_*dcvt5x2*L8probe*/L8_K2_SO_on_Code_19250steps_output.txt")))
    l8 = evals(next(RES.glob("cfc_baoab_code_*dcvt5x8*L8probe*/L8_SO_on_Code_19250steps_output.txt")))
    l4 = evals(next(RES.glob("cfc_baoab_code_*dcvt5x8*L4probe*/L4_SO_on_Code_19250steps_output.txt")))
    st = sorted(k2l8)
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.axvspan(1400, 3600, color=GREY, alpha=0.18, lw=0)
    ax.text(2500, 0.07, "L8-CODE's skipped\nupdates", ha="center", va="bottom", fontsize=9, color=MUTED)
    ax.axvspan(10562, 16250, color=GREY, alpha=0.08, lw=0)
    ax.text(13400, -0.15, "learning-rate\ndecay", ha="center", va="bottom", fontsize=9, color=MUTED)
    ax.axhline(0, color=MUTED, lw=1)
    ax.plot(st, [k2l8[s] - l8[s] for s in st], marker="o", color=BLUE, lw=2, ms=5, label="against L8-CODE (8 wells, 8 layers)")
    ax.plot(st, [k2l8[s] - l4[s] for s in st], marker="s", color=PURPLE, lw=2, ms=5, label="against L4-CODE (8 wells, 4 layers)")
    ax.axvline(6000, color=MUTED, ls=":", lw=1)
    ax.text(6100, 0.08, "stop rule read here\n(run to the end)", fontsize=9, color=MUTED, va="bottom")
    ax.set_ylim(-0.45, 0.15)
    ax.set_xlabel("training step")
    ax.set_ylabel("K2-L8-CODE validation loss minus the other (nats)")
    ax.set_title("Two wells at eight layers, paired against eight wells (full run)", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    save(fig, "k2l8_training.png")


if __name__ == "__main__":
    for f in (fig_neff, fig_force, fig_temperature, fig_keep_k, fig_schematic, fig_fits, fig_k2l8):
        f()
