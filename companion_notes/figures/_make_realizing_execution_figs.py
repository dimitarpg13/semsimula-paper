"""Figures for Realizing_Execution_Gauge_vs_Causal_Ordering.md.

Three PNGs into figures/realizing_execution/. Every panel is an exact
computation from the note's own formulas. Nothing is fit to model data and
nothing anticipates an experimental result.

  rex_noncommutativity.png  -- why an additive channel cannot carry order and a
                            multiplicative one can: the additive parallelogram
                            closes, the rotation square does not (S2.1).
  rex_admissible_rotations.png -- the energy cost of a gauge action: rotations
                            about an attractor leave an isotropic well exactly
                            invariant and an anisotropic one invariant only in
                            the eigenspaces of its precision (S2.4).
  rex_parameter_cost.png    -- exact parameter counts of the three channels, and
                            the admissible fraction of so(d) (S2.5).

Run:  python3 _make_realizing_execution_figs.py
"""
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the projection)

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12,
    "axes.titleweight": "bold", "figure.dpi": 150,
})
GREEN, BLUE, RED = "#22C55E", "#3B82F6", "#EF4444"
PURPLE, GREY, ORANGE = "#8B5CF6", "#9CA3AF", "#F59E0B"
INK, MUTED = "#1F2937", "#6B7280"
OUT = Path(__file__).parent / "realizing_execution"
OUT.mkdir(exist_ok=True)


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def Rx(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Rz(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# ---------------------------------------------------------------------------
def fig_noncommutativity():
    """S2.1: the additive square closes; the rotation square does not."""
    fig = plt.figure(figsize=(11.4, 4.9))
    axA = fig.add_subplot(1, 2, 1)
    axB = fig.add_subplot(1, 2, 2, projection="3d")

    # --- additive: h + q1 + q2 = h + q2 + q1 -------------------------------
    h = np.array([0.0, 0.0])
    q1 = np.array([1.25, 0.28])
    q2 = np.array([0.42, 1.05])
    pts = {"h": h, "h+q1": h + q1, "h+q2": h + q2, "h+q1+q2": h + q1 + q2}
    axA.annotate("", xy=h + q1, xytext=h,
                 arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.4))
    axA.annotate("", xy=h + q1 + q2, xytext=h + q1,
                 arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=2.4))
    axA.annotate("", xy=h + q2, xytext=h,
                 arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=2.4, ls="--"))
    axA.annotate("", xy=h + q2 + q1, xytext=h + q2,
                 arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.4, ls="--"))
    for lab, p in pts.items():
        axA.plot(*p, marker="o", color=INK, ms=6, zorder=5)
    axA.text(h[0] - 0.14, h[1] - 0.17, r"$h$", fontsize=12, color=INK)
    axA.text(*(h + q1 + np.array([0.04, -0.21])), r"$h+Q_1$", fontsize=10.5,
             color=BLUE)
    axA.text(*(h + q2 + np.array([-0.52, 0.06])), r"$h+Q_2$", fontsize=10.5,
             color=PURPLE)
    axA.text(*(h + q1 + q2 + np.array([-0.30, 0.14])),
             r"$h+Q_1+Q_2 = h+Q_2+Q_1$", fontsize=10.5, color=INK,
             fontweight="bold")
    axA.set_xlim(-0.75, 2.35)
    axA.set_ylim(-0.45, 1.85)
    axA.set_aspect("equal")
    axA.set_title("Additive channel: the square closes")
    axA.set_xticks([])
    axA.set_yticks([])
    axA.text(-0.70, -0.38, "order is invisible: addition commutes",
             color=MUTED, fontsize=10)

    # --- multiplicative: U1 U2 h != U2 U1 h --------------------------------
    th = math.radians(60.0)
    U1, U2 = Rx(th), Rz(th)
    # off-axis start, so neither route has a trivial first step
    h0 = np.array([0.6, 0.0, 0.8])
    a1 = U1 @ h0
    a12 = U2 @ a1          # U2 U1 h
    b1 = U2 @ h0
    b12 = U1 @ b1          # U1 U2 h
    gap = float(np.linalg.norm(a12 - b12))

    u = np.linspace(0, 2 * np.pi, 48)
    v = np.linspace(0, np.pi, 24)
    axB.plot_wireframe(np.outer(np.cos(u), np.sin(v)),
                       np.outer(np.sin(u), np.sin(v)),
                       np.outer(np.ones_like(u), np.cos(v)),
                       color=GREY, alpha=0.16, linewidth=0.5,
                       rstride=4, cstride=6)

    def arc(P, Q, col, ls, lw=2.6):
        ts = np.linspace(0, 1, 80)
        pts = np.array([(1 - t) * P + t * Q for t in ts])
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        axB.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=col, lw=lw, ls=ls,
                 zorder=5)

    arc(h0, a1, BLUE, "-")
    arc(a1, a12, BLUE, "-")
    arc(h0, b1, GREEN, "--")
    arc(b1, b12, GREEN, "--")
    for p, col in ((a1, BLUE), (b1, GREEN)):
        axB.scatter(*p, color=col, s=34, depthshade=False, zorder=6)
    for p, col, lab, off in ((h0, INK, r"$h$", (0.10, 0.02, 0.06)),
                             (a12, BLUE, r"$U_2U_1h$", (0.04, 0.26, -0.02)),
                             (b12, GREEN, r"$U_1U_2h$", (-0.05, -0.34, 0.16))):
        axB.scatter(*p, color=col, s=62, depthshade=False, zorder=7)
        axB.text(p[0] + off[0], p[1] + off[1], p[2] + off[2], lab, color=col,
                 fontsize=11, fontweight="bold", zorder=8)
    axB.plot(*zip(a12, b12), color=RED, lw=2.4, ls=":", zorder=7)
    mid = 0.5 * (a12 + b12)
    axB.text(mid[0], mid[1] - 0.04, mid[2] - 0.34, "group commutator",
             color=RED, fontsize=10, fontweight="bold", zorder=8, ha="center")
    axB.set_box_aspect((1, 1, 1))
    axB.set_xlim(-0.85, 0.85)
    axB.set_ylim(-0.85, 0.85)
    axB.set_zlim(-0.85, 0.85)
    axB.set_axis_off()
    axB.view_init(elev=26, azim=-58)
    axB.set_title("Gauge channel: the square does not close")
    axB.text2D(0.02, 0.02, rf"two $60^\circ$ rotations, gap $\|U_1U_2h-U_2U_1h\|$"
               rf" = {gap:.3f}", transform=axB.transAxes, color=MUTED,
               fontsize=10)

    fig.suptitle("Why order needs a multiplicative channel",
                 fontsize=12.5, fontweight="bold", y=1.0)
    fig.tight_layout()
    save(fig, "rex_noncommutativity.png")
    return gap


# ---------------------------------------------------------------------------
def fig_admissible_rotations():
    """S2.4: V is invariant along a rotation orbit iff [U, Lambda] = 0."""
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.4, 4.5))
    mu = np.array([0.0, 0.0])
    r_orbit = 0.42

    cases = [("isotropic  $\\sigma_1=\\sigma_2=0.30$", 0.30, 0.30, BLUE, "-"),
             ("anisotropic  $\\sigma_1=0.20,\\ \\sigma_2=0.50$", 0.20, 0.50,
              ORANGE, "--")]

    xs = np.linspace(-0.85, 0.85, 400)
    ys = np.linspace(-0.85, 0.85, 400)
    XX, YY = np.meshgrid(xs, ys)
    s1, s2 = cases[1][1], cases[1][2]
    Vaniso = 1.0 - np.exp(-0.5 * ((XX - mu[0]) ** 2 / s1 ** 2
                                  + (YY - mu[1]) ** 2 / s2 ** 2))
    axA.contourf(xs, ys, Vaniso, levels=14, cmap="Oranges", alpha=0.5)
    axA.contour(xs, ys, Vaniso, levels=7, colors=GREY, linewidths=0.6)
    ang = np.linspace(0, 2 * np.pi, 400)
    axA.plot(mu[0] + r_orbit * np.cos(ang), mu[1] + r_orbit * np.sin(ang),
             color=INK, lw=2.2, ls=":")
    axA.plot(*mu, marker="+", color=INK, ms=12, mew=2)
    axA.annotate("rotation orbit", xy=(0.0, r_orbit),
                 xytext=(-0.80, 0.70), color=INK, fontsize=10,
                 arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
    axA.set_aspect("equal")
    axA.set_xlim(-0.85, 0.85)
    axA.set_ylim(-0.85, 0.85)
    axA.set_xticks([])
    axA.set_yticks([])
    axA.set_title("An orbit about the attractor")

    th = np.linspace(0, 2 * np.pi, 500)
    label_y = {"iso": 0.665, "aniso": 0.205}
    for lab, s1, s2, col, ls in cases:
        x = mu[0] + r_orbit * np.cos(th)
        y = mu[1] + r_orbit * np.sin(th)
        V = 1.0 - np.exp(-0.5 * (x ** 2 / s1 ** 2 + y ** 2 / s2 ** 2))
        axB.plot(np.degrees(th), V, color=col, lw=2.3, ls=ls)
        swing = V.max() - V.min()
        axB.text(95, label_y["iso"] if swing < 1e-9 else label_y["aniso"],
                 lab, color=col, fontsize=10, fontweight="bold", ha="center")
        if swing > 1e-9:
            axB.annotate("", xy=(388, V.max()), xytext=(388, V.min()),
                         arrowprops=dict(arrowstyle="<|-|>", color=col, lw=1.6))
            axB.text(398, 0.5 * (V.max() + V.min()),
                     rf"swing" "\n" rf"{swing:.2f}$\,\mathfrak{{m}}\upsilon^2$",
                     color=col, fontsize=10, va="center", ha="left")
    axB.set_xlabel("orbit angle (degrees)")
    axB.set_ylabel(r"$V/\mathfrak{m}\upsilon^2$ along the orbit")
    axB.set_title(r"Invariant iff $[U,\Lambda]=0$")
    axB.set_xlim(0, 470)
    axB.set_ylim(-0.02, 1.05)
    axB.set_xticks([0, 90, 180, 270, 360])
    axB.grid(alpha=0.25, lw=0.6)
    axB.set_axisbelow(True)

    fig.suptitle("The energy cost of a gauge action is set by the anisotropy of "
                 "the well", fontsize=12.5, fontweight="bold", y=1.01)
    fig.text(0.5, -0.05, "Left: the anisotropic case, where the circular orbit "
             "crosses contours. Right: the potential along that orbit, flat "
             "exactly when the rotation commutes with the precision.",
             ha="center", fontsize=9.5, color=MUTED)
    fig.tight_layout()
    save(fig, "rex_admissible_rotations.png")


# ---------------------------------------------------------------------------
def fig_parameter_cost():
    """S2.5: exact parameter counts and the admissible fraction of so(d)."""
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.4, 4.4))
    M, d_k, rho, r_low = 32, 64, 4, 4
    d = np.arange(128, 1153, 8)

    additive = 2 * d * d_k + d ** 2            # W_Q, W_K (d x d_k) and W_V (d x d)
    full_gauge = M * d * (d - 1) // 2          # one full so(d) generator each
    lowrank = M * 2 * rho * d                  # rank-2 rho antisymmetric each

    axA.plot(d, full_gauge, color=RED, lw=2.3)
    axA.plot(d, additive, color=BLUE, lw=2.3)
    axA.plot(d, lowrank, color=GREEN, lw=2.3, ls="--")
    axA.set_yscale("log")
    axA.text(470, 9.0e6, r"full $\mathfrak{so}(d)$ per register", color=RED,
             fontsize=10, fontweight="bold")
    axA.text(690, 7.5e5, "additive channel (today)", color=BLUE, fontsize=10,
             fontweight="bold")
    axA.text(690, 8.0e4, rf"rank-${2*rho}$ generators", color=GREEN,
             fontsize=10, fontweight="bold")
    for dd in (384,):
        i = int(np.where(d == dd)[0][0])
        for val, col in ((full_gauge[i], RED), (additive[i], BLUE),
                         (lowrank[i], GREEN)):
            axA.plot([dd], [val], marker="o", color=col, ms=7, zorder=5)
        axA.axvline(dd, color=GREY, lw=1.0, ls=":")
        axA.text(dd + 16, 1.4e7, rf"$d$ = {dd}", color=MUTED, fontsize=9.5)
    axA.set_xlabel(r"hidden dimension $d$")
    axA.set_ylabel("added parameters")
    axA.set_title(rf"Parameter cost at $M$ = {M} registers")
    axA.grid(alpha=0.25, lw=0.6, which="both")
    axA.set_axisbelow(True)

    dd = np.arange(16, 1153, 4)
    full = dd * (dd - 1) / 2
    adm = (dd - r_low - 1) * (dd - r_low - 2) / 2
    axB.plot(dd, 100 * adm / full, color=PURPLE, lw=2.4)
    axB.axhline(100, color=GREY, lw=1.2, ls=":")
    i384 = int(np.where(dd == 384)[0][0])
    axB.plot([384], [100 * adm[i384] / full[i384]], marker="o", color=RED, ms=7,
             zorder=5)
    axB.annotate(rf"$d$ = 384: {100 * adm[i384] / full[i384]:.1f}% of "
                 rf"$\mathfrak{{so}}(d)$ admissible",
                 xy=(384, 100 * adm[i384] / full[i384]),
                 xytext=(470, 92.5), fontsize=10, color=INK,
                 arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
    axB.set_xlabel(r"hidden dimension $d$")
    axB.set_ylabel("admissible fraction (%)")
    axB.set_title(rf"Generators commuting with $\Lambda$ (rank $r$ = {r_low})")
    axB.set_ylim(85, 101.5)
    axB.set_xlim(16, 1152)
    axB.grid(alpha=0.25, lw=0.6)
    axB.set_axisbelow(True)

    fig.suptitle("The gauge channel is cheaper than the additive one it replaces",
                 fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "rex_parameter_cost.png")

    i = int(np.where(d == 384)[0][0])
    print(f"  d=384, M={M}: additive={additive[i]:,}  "
          f"full_gauge={full_gauge[i]:,}  lowrank={lowrank[i]:,}")
    print(f"  admissible fraction at d=384, r={r_low}: "
          f"{100 * adm[i384] / full[i384]:.2f}%")


if __name__ == "__main__":
    g = fig_noncommutativity()
    fig_admissible_rotations()
    fig_parameter_cost()
    print(f"\ncommutator gap (two 60 deg rotations): {g:.4f}")
