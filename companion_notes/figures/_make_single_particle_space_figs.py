"""Figures for Single_Particle_Hilbert_Space_in_Semantic_Simulation.md.

Eight PNGs into figures/single_particle_space/. Every panel is an exact
computation from the note's own formulas and its toy embeddings
(cat = (0.90, 0.40), kitten = (0.85, 0.50), tax = (-0.30, 0.95), sigma = 0.25,
so kappa^2 = 2 and x* = 0.5). Nothing is fit to model data.

  sps_root_densities.png   -- the three root densities in Sigma = R^2 and the
                              overlap identity <phi_a, phi_b> = 1 - V/(m u^2)
                              against distance, with the three pairs marked (S1.3, S2.3).
  sps_sum_vs_mixture.png   -- the 1/sqrt2-scaled sum squared against the
                              50/50 mixture, for separated and overlapping
                              states; the gap is the cross term (S1.4).
  sps_sigma_limit.png      -- overlaps against sigma: Choice B tends to
                              Choice A as sigma -> 0; the framework's sigma = x*/2 (S2.3).
  sps_hermite.png          -- the Hermite basis around a centroid, and a shift
                              and a squeeze read off at first and second order (S2.4).
  sps_bank_correlation.png -- the correlated joint density (4.3), the product
                              of its marginals, and the conditional at y = water (S4.2).
  sps_symmetrization.png   -- the 9 ordered pairs of {cat, dog, runs} folding
                              into the 6 occupation states (S5.3).
  sps_lowdin_modes.png     -- the Lowdin-orthonormalised cat mode is not a root
                              density: it is pushed off kitten and goes negative
                              beyond it (S7.1).
  sps_vacuum_to_stationary.png -- creation at rate c, decay at rate d: the vacuum
                              relaxes to the Poisson (coherent) state of mean c/d (S6.3, S7.2).

Run:  python3 _make_single_particle_space_figs.py
"""
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12,
    "axes.titleweight": "bold", "figure.dpi": 150,
})
GREEN, BLUE, RED = "#22C55E", "#3B82F6", "#EF4444"
PURPLE, GREY, ORANGE = "#8B5CF6", "#9CA3AF", "#F59E0B"
OUT = Path(__file__).parent / "single_particle_space"
OUT.mkdir(exist_ok=True)

EMB = {"cat": np.array([0.90, 0.40]), "kitten": np.array([0.85, 0.50]),
       "tax": np.array([-0.30, 0.95])}
COL = {"cat": BLUE, "kitten": GREEN, "tax": RED}
SIGMA = 0.25
KAPPA2 = 1.0 / (8 * SIGMA ** 2)          # = 2
XSTAR = 1.0 / math.sqrt(2 * KAPPA2)      # = 0.5 = 2 sigma


def root_density(X, mu, s=SIGMA):
    """phi(x) = (2 pi s^2)^(-L/4) exp(-|x - mu|^2 / (4 s^2)), L = last axis of mu."""
    L = mu.shape[-1]
    r2 = ((X - mu) ** 2).sum(-1)
    return (2 * np.pi * s * s) ** (-L / 4) * np.exp(-r2 / (4 * s * s))


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


# ---------------------------------------------------------------------------
def fig_root_densities():
    xs = np.linspace(-1.0, 1.6, 400)
    ys = np.linspace(-0.2, 1.6, 300)
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X, Y], -1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6))
    for n, mu in EMB.items():
        phi = root_density(P, mu)
        a.contour(X, Y, phi, levels=np.array([0.25, 0.5, 0.75]) * phi.max(),
                  colors=COL[n], linewidths=1.4)
        a.plot(*mu, "o", color=COL[n])
        a.annotate(n, mu, xytext=(8, 8 if n != "kitten" else 14), textcoords="offset points",
                   color=COL[n], fontweight="bold")
    a.add_patch(plt.Circle(EMB["cat"], XSTAR, fill=False, ls="--", color=GREY))
    a.annotate("inflection radius x*\nof cat's well", EMB["cat"] + np.array([XSTAR * 0.71, -XSTAR * 0.71]),
               xytext=(14, -4), textcoords="offset points", color=GREY, fontsize=9,
               arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))
    a.set_aspect("equal"); a.set_xlim(-1.0, 1.9); a.set_ylim(-0.25, 1.6)
    a.set_title("Root densities in semantic space (sigma = x*/2 = 0.25)")
    a.set_xlabel("x1"); a.set_ylabel("x2")

    d = np.linspace(0, 3.2 * XSTAR, 400)
    ov = np.exp(-KAPPA2 * d ** 2)
    b.plot(d / XSTAR, ov, color=BLUE, lw=2, label="overlap  <phi_a, phi_b> = BC")
    b.plot(d / XSTAR, 1 - (1 - np.exp(-KAPPA2 * d ** 2)), color=ORANGE, lw=1, ls=":",
           label="1 - V(d) / (m upsilon^2)  (identical)")
    b.plot(d / XSTAR, np.sqrt(2 * (1 - ov)), color=PURPLE, lw=2, label="overlap distance  ||phi_a - phi_b||")
    b.axhline(math.sqrt(2), color=GREY, lw=0.8, ls="--")
    b.text(3.15, math.sqrt(2) + 0.03, "sqrt 2", color=GREY, ha="right", fontsize=9)
    b.axvline(1.0, color=GREY, lw=0.8, ls="--")
    b.text(1.03, 1.0, "d = x*", color=GREY, fontsize=9)
    for (p, q) in (("cat", "kitten"), ("kitten", "tax"), ("cat", "tax")):
        dd = np.linalg.norm(EMB[p] - EMB[q])
        o = math.exp(-KAPPA2 * dd ** 2)
        b.plot(dd / XSTAR, o, "o", color=BLUE)
        b.plot(dd / XSTAR, math.sqrt(2 * (1 - o)), "s", color=PURPLE)
        off = {"kitten": (6, -14), "tax": ((-62, 8) if p == "kitten" else (6, 8))}[q]
        b.annotate(f"{p}-{q}", (dd / XSTAR, o), xytext=off, textcoords="offset points", fontsize=8.5)
    b.set_xlabel("semantic distance d / x*"); b.set_ylim(-0.03, 1.55)
    b.set_title("The well fixes the inner product")
    b.legend(loc="center right", fontsize=8.5, frameon=False)
    save(fig, "sps_root_densities.png")


# ---------------------------------------------------------------------------
def fig_sum_vs_mixture():
    x = np.linspace(-2.2, 2.2, 1000)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, sep, title in ((axes[0], 4 * XSTAR, "Separated: d = 4 x*  (overlap 0.0003)"),
                           (axes[1], 0.6 * XSTAR, "Overlapping: d = 0.6 x*  (overlap 0.84)")):
        pa = root_density(x[:, None], np.array([-sep / 2])) ** 2
        pb = root_density(x[:, None], np.array([sep / 2])) ** 2
        mix = 0.5 * pa + 0.5 * pb
        summ = (np.sqrt(pa) + np.sqrt(pb)) ** 2 / 2
        ov = math.exp(-KAPPA2 * sep ** 2)
        ax.fill_between(x, mix, summ, color=ORANGE, alpha=0.35, label="cross term phi_a phi_b")
        ax.plot(x, mix, color=BLUE, lw=2, label="50/50 mixture  (p_a + p_b)/2")
        ax.plot(x, summ, color=RED, lw=1.6, ls="--", label="((phi_a + phi_b)/sqrt2)^2")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.text(0.02, 0.60, f"area under the sum\n= 1 + overlap = {1 + ov:.3f}", transform=ax.transAxes,
                fontsize=9, va="top")
    axes[0].set_ylabel("density")
    axes[0].legend(loc="upper left", fontsize=8.5, frameon=False)
    save(fig, "sps_sum_vs_mixture.png")


# ---------------------------------------------------------------------------
def fig_sigma_limit():
    s = np.logspace(-2.3, 0.3, 400)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for (p, q), c in ((("cat", "kitten"), GREEN), (("cat", "tax"), RED), (("kitten", "tax"), ORANGE)):
        d2 = ((EMB[p] - EMB[q]) ** 2).sum()
        ax.plot(s, np.exp(-d2 / (8 * s ** 2)), color=c, lw=2, label=f"{p}-{q}")
    ax.axvline(SIGMA, color=GREY, ls="--", lw=1)
    ax.text(SIGMA * 0.95, 0.5, "framework:\nsigma = x*/2", color=GREY, fontsize=9, ha="right")
    ax.set_xscale("log")
    ax.set_xlabel("width sigma (log scale)"); ax.set_ylabel("overlap  <phi_a, phi_b>")
    ax.annotate("Choice A: labels,\nevery overlap 0", (0.006, 0.02), fontsize=9, color=GREY)
    ax.annotate("everything blends", (0.9, 0.9), fontsize=9, color=GREY, ha="center")
    ax.set_title("Choice B tends to Choice A as sigma -> 0")
    ax.legend(loc="center left", fontsize=9, frameon=False)
    save(fig, "sps_sigma_limit.png")


# ---------------------------------------------------------------------------
def hermite_fn(n, u):
    """Normalised 1-D Hermite function h_n(u) (physicists' H_n), by recurrence."""
    h0 = np.pi ** -0.25 * np.exp(-u * u / 2)
    if n == 0:
        return h0
    h1 = math.sqrt(2) * u * h0
    for k in range(2, n + 1):
        h0, h1 = h1, math.sqrt(2 / k) * u * h1 - math.sqrt((k - 1) / k) * h0
    return h1


def fig_hermite():
    x = np.linspace(-1.2, 1.2, 800)
    sc = math.sqrt(2) * SIGMA
    psi = lambda n: sc ** -0.5 * hermite_fn(n, x / sc)
    dx = x[1] - x[0]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2))
    for n, c in zip(range(4), (BLUE, GREEN, ORANGE, PURPLE)):
        a.plot(x, psi(n), color=c, lw=2, label=f"psi_{n}" + ("  = phi_v" if n == 0 else ""))
    a.axhline(0, color=GREY, lw=0.6)
    a.set_title("Hermite basis around a centroid (scale sigma)")
    a.set_xlabel("x - mu_v"); a.legend(fontsize=9, frameon=False)

    phi = psi(0)
    shift = SIGMA * 0.4
    shifted = sc ** -0.5 * hermite_fn(0, (x - shift) / sc)
    c1 = (shifted * psi(1)).sum() * dx
    wide = root_density(x[:, None], np.array([0.0]), s=SIGMA * 1.3)
    c2 = (wide * psi(2)).sum() * dx
    b.plot(x, phi, color=GREY, lw=1.2, ls=":", label="phi_v")
    b.plot(x, shifted, color=BLUE, lw=2, label="shifted by 0.4 sigma")
    b.plot(x, (phi * (shifted * phi).sum() * dx) + c1 * psi(1), color=BLUE, lw=1, ls="--",
           label=f"projection on psi_0, psi_1: {c1:.2f} psi_1  (first order)")
    b.plot(x, wide, color=ORANGE, lw=2, label="widened by 30%")
    b.plot(x, (wide * phi).sum() * dx * phi + c2 * psi(2), color=ORANGE, lw=1, ls="--",
           label=f"projection on psi_0, psi_2: {c2:.2f} psi_2  (second order)")
    b.set_title("Deformations read off the basis")
    b.set_xlabel("x - mu_v"); b.legend(fontsize=8.5, frameon=False, loc="upper left")
    save(fig, "sps_hermite.png")


# ---------------------------------------------------------------------------
def fig_bank():
    s = SIGMA
    g = np.linspace(-2.2, 2.2, 500)
    X, Y = np.meshgrid(g, g)
    n1 = lambda z, m: np.exp(-(z - m) ** 2 / (2 * s * s)) / math.sqrt(2 * math.pi * s * s)
    river, finance, water, money = -1.2, 1.2, -1.2, 1.2
    P = 0.5 * n1(X, river) * n1(Y, water) + 0.5 * n1(X, finance) * n1(Y, money)
    px = 0.5 * n1(g, river) + 0.5 * n1(g, finance)
    py = 0.5 * n1(g, water) + 0.5 * n1(g, money)
    Q = np.outer(py, px)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, Z, t in ((axes[0], P, "Correlated joint P(x, y), eq. (4.3)"),
                     (axes[1], Q, "Product of its marginals")):
        ax.contourf(X, Y, Z, levels=12, cmap="Blues")
        ax.set_aspect("equal"); ax.set_title(t)
        ax.set_xlabel("x: sense of 'bank'")
        ax.set_xticks([river, finance], ["river", "finance"])
        ax.set_yticks([water, money], ["water", "money"])
    axes[0].set_ylabel("y: context word")
    axes[0].axhline(water, color=RED, lw=1.2, ls="--")
    axes[0].text(2.1, water + 0.12, "y = water", color=RED, ha="right", fontsize=9)
    iy = np.argmin(abs(g - water))
    cond = P[iy] / (P[iy].sum() * (g[1] - g[0]))
    ax = axes[2]
    ax.plot(g, px, color=GREY, lw=1.5, ls=":", label="marginal P(x): 50/50 senses")
    ax.plot(g, cond, color=RED, lw=2, label="conditional P(x | y = water)")
    ax.set_xticks([river, finance], ["river", "finance"])
    w = (cond * (g < 0)).sum() / cond.sum()
    ax.set_title("Disambiguation is conditioning")
    ax.text(0.97, 0.70, f"P(river | water) = {w:.6f}", transform=ax.transAxes, va="top", ha="right", fontsize=9)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)
    ax.set_xlabel("x: sense of 'bank'")
    save(fig, "sps_bank_correlation.png")


# ---------------------------------------------------------------------------
def fig_symmetrization():
    B = ["cat", "dog", "runs"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
    occ_col = {}
    palette = [BLUE, GREEN, ORANGE, PURPLE, RED, "#14B8A6"]
    occs = []
    for i in range(3):
        for j in range(3):
            o = tuple(sorted((i, j)))
            if o not in occ_col:
                occ_col[o] = palette[len(occ_col)]
                occs.append(o)
            a.add_patch(plt.Rectangle((j, 2 - i), 0.94, 0.94, color=occ_col[o], alpha=0.75))
            a.text(j + 0.47, 2 - i + 0.47, f"{B[i]}\n{B[j]}", ha="center", va="center", fontsize=9.5,
                   color="white", fontweight="bold")
    a.set_xlim(-0.1, 3.05); a.set_ylim(-0.1, 3.05); a.set_aspect("equal"); a.axis("off")
    a.set_title("H x H: 9 ordered pairs (particle 1, particle 2)")
    for k, o in enumerate(occs):
        n = [0, 0, 0]
        for t in o:
            n[t] += 1
        y = 5 - k
        b.add_patch(FancyBboxPatch((0.1, y + 0.1), 4.6, 0.75, boxstyle="round,pad=0.02",
                                   color=occ_col[o], alpha=0.75))
        lab = " + ".join(f"{B[t]}" for t in o) if o[0] != o[1] else f"2 x {B[o[0]]}"
        b.text(0.3, y + 0.47, f"({n[0]},{n[1]},{n[2]})", va="center", fontsize=10, color="white",
               fontweight="bold")
        b.text(1.6, y + 0.47, lab + ("   (2 orderings merged)" if o[0] != o[1] else ""), va="center",
               fontsize=9.5, color="white")
    b.set_xlim(0, 4.8); b.set_ylim(-0.1, 6.1); b.axis("off")
    b.set_title("S^2 H: 6 occupation states (n_cat, n_dog, n_runs)")
    save(fig, "sps_symmetrization.png")


# ---------------------------------------------------------------------------
def fig_lowdin():
    names = list(EMB)
    M = np.stack([EMB[n] for n in names])
    G = np.exp(-KAPPA2 * ((M[:, None] - M[None]) ** 2).sum(-1))
    w, U = np.linalg.eigh(G)
    Gmh = U @ np.diag(w ** -0.5) @ U.T
    xs = np.linspace(0.2, 1.6, 360)
    ys = np.linspace(-0.2, 1.1, 330)
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X, Y], -1)
    phi = np.stack([root_density(P, m) for m in M])
    lt = np.einsum("vw,wxy->vxy", Gmh, phi)
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"wspace": 0.35})
    vmax = abs(lt[0]).max()
    im = a.contourf(X, Y, lt[0], levels=np.linspace(-vmax, vmax, 25), cmap="RdBu_r")
    a.contour(X, Y, lt[0], levels=[0], colors="k", linewidths=0.8)
    for n in ("cat", "kitten"):
        a.plot(*EMB[n], "o", color="k")
        a.annotate(n, EMB[n], xytext=(6, -12 if n == "cat" else 6), textcoords="offset points",
                   fontweight="bold")
    a.set_aspect("equal"); fig.colorbar(im, ax=a, fraction=0.04)
    a.set_title("Lowdin cat mode: pushed off kitten, negative beyond it")
    a.set_xlabel("x1"); a.set_ylabel("x2")

    c, k = EMB["cat"], EMB["kitten"]
    t = np.linspace(-9.0, 10.0, 800)          # in units of |kitten - cat| = 0.112
    line = c[None] + t[:, None] * (k - c)[None]
    ph = np.stack([root_density(line, m) for m in M])
    lt1 = Gmh @ ph
    u = t * np.linalg.norm(k - c) / XSTAR       # distance along the line, in units of x*
    b.plot(u, ph[0], color=BLUE, lw=1.5, ls=":", label="phi_cat (root density)")
    b.plot(u, ph[1], color=GREEN, lw=1.5, ls=":", label="phi_kitten")
    b.plot(u, lt1[0], color=BLUE, lw=2, label="Lowdin cat")
    b.plot(u, lt1[1], color=GREEN, lw=2, label="Lowdin kitten")
    b.axhline(0, color=GREY, lw=0.6)
    u_k = np.linalg.norm(k - c) / XSTAR
    for x0, nm, ha in ((0.0, "cat", "right"), (u_k, "kitten", "left")):
        b.axvline(x0, color=GREY, lw=0.7, ls="--")
        b.text(x0 + (0.03 if ha == "left" else -0.03), -0.48, nm, ha=ha, fontsize=9, fontweight="bold")
    b.set_xlabel("distance along the cat -> kitten line / x*")
    b.set_title("Cut along cat -> kitten")
    print(f"   cond(G) = {w.max() / w.min():.1f}; min of Lowdin cat (2-D) = {lt[0].min():.3f}, "
          f"max {lt[0].max():.3f}; at kitten {float(np.einsum('w,w->', Gmh[0], root_density(EMB['kitten'], M))):.3f}")
    b.legend(fontsize=8.5, frameon=False, loc="upper right")
    save(fig, "sps_lowdin_modes.png")


# ---------------------------------------------------------------------------
def fig_vacuum():
    c, d = 3.0, 1.0
    n = np.arange(0, 11)
    times = (0.0, 0.3, 1.0, 4.0)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2))
    width = 0.2
    for k, (tt, col) in enumerate(zip(times, (GREY, ORANGE, GREEN, BLUE))):
        mean = (c / d) * (1 - math.exp(-d * tt))
        pn = np.array([math.exp(-mean) * mean ** m / math.factorial(m) for m in n])
        a.bar(n + (k - 1.5) * width, pn, width, color=col,
              label=f"t = {tt:g}: Poisson, mean {mean:.2f}" + ("  (vacuum)" if tt == 0 else ""))
    a.set_xlabel("occupation n"); a.set_ylabel("P(n, t)")
    a.set_title("Spontaneous creation (c = 3), decay (d = 1)")
    a.legend(fontsize=8.5, frameon=False)
    tt = np.linspace(0, 6, 300)
    b.plot(tt, (c / d) * (1 - np.exp(-d * tt)), color=BLUE, lw=2, label="mean occupation alpha(t)")
    b.plot(tt, np.exp(-(c / d) * (1 - np.exp(-d * tt))), color=RED, lw=2, label="P(empty, t)")
    b.axhline(c / d, color=GREY, ls="--", lw=0.8)
    b.text(6, c / d - 0.18, "stationary mean c/d", color=GREY, ha="right", fontsize=9)
    b.set_xlabel("time t"); b.set_title("The vacuum is a state, not the prior")
    b.legend(fontsize=9, frameon=False, loc="center right")
    save(fig, "sps_vacuum_to_stationary.png")


if __name__ == "__main__":
    fig_root_densities()
    fig_sum_vs_mixture()
    fig_sigma_limit()
    fig_hermite()
    fig_bank()
    fig_symmetrization()
    fig_lowdin()
    fig_vacuum()
