"""Figures for Future_of_Semantic_Simulation_When_Second_Order_Earns_Its_Keep.md.

Six PNGs into figures/future_second_order/. Every dynamical panel is an exact
simulation of the stated equations (a damped oscillator, a gradient flow, a
Gaussian well integrated by RK4), with the ladder's own constants where they
matter: layer step dt = 4, layer time T = 8, gamma = 0.1, mobility
mu = dt / (2 m) with m = 1. One panel plots recorded measurements (Cell 6b-7
Gate 1 and the FO series' evaluations), labelled by source.

  fso_step_maps.png        -- within one layer: a stiff mode under the second-
                              order step (rotation, light damping) and under
                              the gradient flow (relaxation); the end-of-layer
                              gain as a function of omega T.
  fso_sensitivity.png      -- d z(T) / d log omega: oscillating and growing
                              with T for second order, one-signed and bounded
                              for first order.
  fso_phase_space.png      -- a cloud of initial states through one layer:
                              Liouville (area kept) against contraction.
  fso_correlated_forces.png-- the optimiser analogy: velocity carried across
                              layers helps when successive minima drift
                              coherently and hurts when they jump.
  fso_well_capture.png     -- a Gaussian well: with inertia a token can pass
                              through; without it, it is always captured.
  fso_reliance_vs_need.png -- recorded: Gate 1 by arm (reliance) against the
                              trained first-order twins (need).

Run:  python3 _make_future_second_order_figs.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parent / "future_second_order"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
                     "figure.dpi": 150, "savefig.dpi": 150, "axes.spines.top": False,
                     "axes.spines.right": False})
C_SO, C_FO, C_FOA = "#b2182b", "#2166ac", "#4dac26"
DT, T, GAMMA, M = 4.0, 8.0, 0.1, 1.0
MU = DT / (2 * M)


def so_from_rest(omega, t, gamma=GAMMA):
    """z(t) - z* over (z0 - z*) for m z'' = -m omega^2 (z - z*) - m gamma z', z'(0) = 0."""
    wd = np.sqrt(max(omega**2 - gamma**2 / 4, 1e-12))
    return np.exp(-gamma * t / 2) * (np.cos(wd * t) + gamma / (2 * wd) * np.sin(wd * t))


def fo(kappa, t, mu=MU):
    return np.exp(-mu * kappa * t)


# ---------------------------------------------------------------- 1. step maps
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
t = np.linspace(0, T, 400)
for omega, ls in ((0.3, ":"), (0.8, "--"), (1.65, "-")):
    ax[0].plot(t, so_from_rest(omega, t), color=C_SO, ls=ls, label=f"second order, ω·dt = {omega*DT:.1f}")
    ax[0].plot(t, fo(M * omega**2, t), color=C_FO, ls=ls, label=f"first order, same stiffness")
ax[0].axhline(0, color="k", lw=0.6)
ax[0].set(xlabel="time within the layer", ylabel="distance to the well, relative",
          title="A. One stiff mode through one layer (T = 8, γ = 0.1)")
ax[0].legend(fontsize=8, ncol=1, loc="upper right")
wT = np.linspace(0.05, 14, 600)
omegas = wT / T
ax[1].plot(wT, [so_from_rest(w, T) for w in omegas], color=C_SO, label="second order: e^(−γT/2)·cos(ωT), a phase")
ax[1].plot(wT, [fo(M * w**2, T) for w in omegas], color=C_FO, label="first order: e^(−μκT), a contraction")
ax[1].axhline(0, color="k", lw=0.6)
ax[1].axvline(6.6 * T / DT, color="gray", lw=0.8, ls="--")
ax[1].text(6.6 * T / DT + 0.2, 0.75, "SR2's median\nstiff mode", fontsize=8, color="gray")
ax[1].set(xlabel="ω·T (stiffness of the mode, in phase units)", ylabel="end-of-layer gain  (z(T) − z*) / (z0 − z*)",
          title="B. What the layer does to a mode, by stiffness")
ax[1].legend(fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig(OUT / "fso_step_maps.png"); plt.close(fig)

# ---------------------------------------------------------------- 2. sensitivity
fig, ax = plt.subplots(figsize=(7.5, 4.2))
dlog = 1e-4
so_sens = [(so_from_rest(w * np.exp(dlog), T) - so_from_rest(w * np.exp(-dlog), T)) / (2 * dlog) for w in omegas]
fo_sens = [(fo(M * (w * np.exp(dlog))**2, T) - fo(M * (w * np.exp(-dlog))**2, T)) / (2 * dlog) for w in omegas]
ax.plot(wT, so_sens, color=C_SO, label="second order: ≈ −ωT·e^(−γT/2)·sin(ωT)")
ax.plot(wT, fo_sens, color=C_FO, label="first order: −2μκT·e^(−μκT)")
ax.axhline(0, color="k", lw=0.6)
ax.set(xlabel="ω·T", ylabel="d z(T) / d log ω",
       title="The learning signal for the stiffness: oscillating against one-signed")
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fso_sensitivity.png"); plt.close(fig)

# ---------------------------------------------------------------- 3. phase space
rng = np.random.default_rng(0)
n = 400
z0 = rng.normal(0, 1.0, n); v0 = rng.normal(0, 0.3, n)
omega = 1.65
wd = np.sqrt(omega**2 - GAMMA**2 / 4)
e = np.exp(-GAMMA * T / 2)
c, s = np.cos(wd * T), np.sin(wd * T)
zT = e * (z0 * (c + GAMMA / (2 * wd) * s) + v0 * s / wd)
vT = e * (v0 * (c - GAMMA / (2 * wd) * s) - z0 * (omega**2 / wd) * s)
zF = z0 * fo(M * omega**2, T)
fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
for a in ax:
    a.scatter(z0, v0, s=6, color="gray", alpha=0.5, label="layer input (z, v)")
ax[0].scatter(zT, vT, s=6, color=C_SO, alpha=0.7, label="after one second-order layer")
ax[1].scatter(zF, np.zeros(n), s=6, color=C_FO, alpha=0.7, label="after one first-order layer")
for a, ttl in zip(ax, ("A. Second order: the cloud rotates,\narea kept (Liouville)", "B. First order: the cloud\ncontracts onto the well")):
    a.set(xlabel="position z − z*", ylabel="velocity v", title=ttl, xlim=(-3.2, 3.2), ylim=(-3.2, 3.2))
    a.axhline(0, color="k", lw=0.5); a.axvline(0, color="k", lw=0.5); a.legend(fontsize=8, loc="upper left")
    a.set_aspect("equal")
fig.suptitle("A cloud of initial states through one layer of a stiff mode (ω·dt = 6.6, γ = 0.1)", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "fso_phase_space.png"); plt.close(fig)

# ---------------------------------------------------------------- 4. correlated forces
def run_stack(rho, carry, L=8, kappa=1.0, eta=0.6, beta=0.6, seeds=400):
    """Each layer pulls toward its own minimum c_l with a quadratic of stiffness kappa.
    The minima drift: c_l = c_{l-1} + u_l, u_l = rho u_{l-1} + sqrt(1 - rho^2) eps_l.
    One heavy-ball step per layer, step eta, momentum beta. carry=True keeps the
    velocity across layers (second order); False resets it (first order from rest).
    Returns the mean squared distance to the last layer's minimum."""
    rng = np.random.default_rng(1)
    err = []
    for _ in range(seeds):
        c = np.zeros(2); u = rng.normal(size=2); h = np.zeros(2); v = np.zeros(2)
        for l in range(L):
            u = rho * u + np.sqrt(1 - rho**2) * rng.normal(size=2)
            c = c + u
            if not carry:
                v = np.zeros(2)
            v = beta * v - eta * kappa * (h - c)
            h = h + v
        err.append(np.sum((h - c)**2))
    return np.mean(err)


rhos = np.linspace(0, 1, 21)
fig, ax = plt.subplots(figsize=(7.5, 4.4))
ax.plot(rhos, [run_stack(r, True) for r in rhos], color=C_SO, marker="o", ms=3, label="velocity carried across layers (second order)")
ax.plot(rhos, [run_stack(r, False) for r in rhos], color=C_FO, marker="s", ms=3, label="velocity reset at each layer (first order)")
ax.set(xlabel="correlation ρ of the minima's drift from layer to layer", ylabel="mean squared distance to the last layer's minimum",
       title="Momentum across layers pays only when successive forces are correlated")
ax.legend(fontsize=9)
ax.text(0.02, 0.05, "left: each layer's potential is drawn fresh given h\n(the piecewise-autonomous stack: V_θ with a per-layer context)\nright: the potential drifts coherently in depth\n(persistent registers, an exchange field, a shared context)",
        transform=ax.transAxes, fontsize=8, va="bottom")
fig.tight_layout(); fig.savefig(OUT / "fso_correlated_forces.png"); plt.close(fig)

# ---------------------------------------------------------------- 5. well capture
a_depth, s_w = 2.0, 1.0
Vp = lambda x: a_depth * x / s_w**2 * np.exp(-x**2 / (2 * s_w**2))      # dV/dx for V = -a exp(-x^2 / 2 s^2)


def rk4(f, y, t_end, n=2000):
    h = t_end / n; ys = [y.copy()]
    for _ in range(n):
        k1 = f(y); k2 = f(y + h / 2 * k1); k3 = f(y + h / 2 * k2); k4 = f(y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); ys.append(y.copy())
    return np.array(ys)


so_f = lambda y: np.array([y[1], -Vp(y[0]) / M - GAMMA * y[1]])
fo_f = lambda y: np.array([MU * (-Vp(y[0]))])
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
tt = np.linspace(0, T, 2001)
for v0, ls in ((0.0, "-"), (1.2, "--"), (2.4, ":")):
    tr = rk4(so_f, np.array([-3.0, v0]), T)
    ax[0].plot(tt, tr[:, 0], color=C_SO, ls=ls, label=f"second order, incoming velocity {v0:.1f}")
tr = rk4(fo_f, np.array([-3.0]), T)
ax[0].plot(tt, tr[:, 0], color=C_FO, lw=2, label="first order (no velocity exists)")
ax[0].axhline(0, color="k", lw=0.6); ax[0].set(xlabel="time within the layer", ylabel="position", title="A. Approaching a well of depth 2 from x = −3")
ax[0].legend(fontsize=8)
v0s = np.linspace(0, 3.5, 120)
ax[1].plot(v0s, [rk4(so_f, np.array([-3.0, v]), T, 800)[-1, 0] for v in v0s], color=C_SO, label="second order: final position")
ax[1].axhline(tr[-1, 0], color=C_FO, lw=2, label="first order: final position")
ax[1].axhline(0, color="k", lw=0.6)
ax[1].set(xlabel="incoming velocity", ylabel="position at the end of the layer", title="B. Capture or pass-through, by incoming velocity")
ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fso_well_capture.png"); plt.close(fig)

# ---------------------------------------------------------------- 6. reliance vs need (recorded)
arms = ["SR2", "F3.1", "G2", "G3′", "G3", "L = 4", "PMX", "XRB-F"]
gate1 = [3.9, 34.3, 33, 34, 49, 55, 91, 113]
kind = ["conservative", "conservative", "Fock", "Fock", "Fock", "Fock", "wells", "Fock"]
col = {"conservative": C_FO, "Fock": "#d95f02", "wells": "#7570b3"}
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1.1, 1]})
ax[0].bar(arms, gate1, color=[col[k] for k in kind])
for i, g in enumerate(gate1):
    ax[0].text(i, g + 2, f"+{g:g}%", ha="center", fontsize=8)
ax[0].set(ylabel="Gate 1: perplexity rise when the velocity is reset", title="A. Reliance on velocity, by arm (Cell 6b-7)")
ax[0].text(0.02, 0.92, "PM1 (explicit wells): +10,600%, off the scale", transform=ax[0].transAxes, fontsize=8)
from matplotlib.patches import Patch
ax[0].legend(handles=[Patch(color=col[k], label=k) for k in col], fontsize=8, loc="upper left", bbox_to_anchor=(0, 0.9))
# panel B: trained twins
steps_ts = [4000, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 10500, 11000, 11500, 12000, 12500, 13000, 13500, 14000, 14500, 15000, 15500, 16000, 16500, 17000, 17500, 18000, 18500, 19000]
d_ts = [-0.0109, -0.0196, -0.0222, -0.0216, -0.0203, -0.0182, -0.0147, -0.0165, -0.0180, -0.0150, -0.0162, -0.0141, -0.0151, -0.0124, -0.0091, -0.0090, -0.0085, -0.0074, -0.0065, -0.0065, -0.0056, -0.0054, -0.0046, -0.0055, -0.0066, -0.0052]
steps_owt = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 10500, 11000, 11500, 12000, 12500, 13000, 13500, 14000, 14500, 15000, 15500, 16000, 16500, 17000, 17500, 18000, 18500, 19000, 19500, 20000, 20500, 21000, 21500, 22000, 22500, 23000, 23500, 24000, 24500, 25000, 25500, 26000, 26500, 27000, 27500, 28000, 28500, 29000, 29500, 30000, 30500, 31000, 31500, 32000, 32500]
d_owt = [-0.0234, -0.0206, -0.018, -0.0256, -0.0295, -0.034, -0.0436, -0.0579, -0.063, -0.0691, -0.0498, -0.0325, -0.0363, -0.0361, -0.0387, -0.0428, -0.0337, -0.0319, -0.0361, -0.0284, -0.0091, 0.0292, 0.0458, 0.037, 0.0376, 0.0414, 0.0476, 0.0396, 0.0431, 0.0341, 0.0317, 0.0325, 0.0384, 0.0349, 0.0279, 0.0305, 0.0319, 0.0294, 0.0305, 0.0229, 0.0323, 0.029, 0.0239, 0.0188, 0.03, 0.0238, 0.0286, 0.0264, 0.0237, 0.0231, 0.0326, 0.0327, 0.035, 0.0348, 0.0416, 0.0406, 0.0449, 0.0451, 0.0463, 0.0494, 0.0493, 0.0521, 0.0466, 0.0467, 0.0494]
ax[1].plot(steps_ts, [100 * (np.exp(d) - 1) for d in d_ts], color=C_FOA, marker="o", ms=3, label="TinyStories: FO-a (FO-TS) against SO-TS, same batches")
ax[1].plot(steps_owt, [100 * (np.exp(d) - 1) for d in d_owt], color=C_FO, marker="s", ms=3, label="OpenWebText: FO-b against SR2 (settled +4.9%)")
ax[1].axhline(0, color="k", lw=0.6)
ax[1].axvspan(10562, 19250, color=C_FOA, alpha=0.06); ax[1].text(10800, -4.6, "TS decay + floor", fontsize=8, color=C_FOA)
ax[1].axvspan(21125, 32500, color=C_FO, alpha=0.06); ax[1].text(27500, -1.2, "OWT decay", fontsize=8, color=C_FO)
ax[1].set(xlabel="training step", ylabel="first-order perplexity relative to second-order, %", title="B. Need: the trained first-order twins", ylim=(-7.5, 6.5))
ax[1].legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "fso_reliance_vs_need.png"); plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
