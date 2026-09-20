"""Figures for Composing_Single_Layer_Inferences_Flow_or_Maps.md.

Five PNGs into figures/flow_or_maps/. Every panel is an exact simulation
of the stated dynamics -- a damped harmonic oscillator integrated by
velocity-Verlet with an O-step multiplier, which is the structure of the
BAOAB/CfC scheme reduced to one dimension -- or a clearly-labelled
illustration of a decision rule. Nothing is fit to unpublished data.

  fom_flow_vs_maps.png     -- the central distinction. Panel A: a FIXED
                              potential, refined at constant total time
                              T = N*dt; the discretisations converge on one
                              curve. Panel B: the same refinement when the
                              potential carries a per-step depth code that
                              is CYCLED. Refinement still converges -- but
                              to the flow of the MEAN potential, by
                              homogenisation, NOT to the trained two-step
                              map. Cycling raises the alternation frequency
                              instead of resolving the same tau -> code map
                              more finely, so the N -> infinity limit is a
                              different model from the one that was trained.
  fom_carried_vs_reset.png -- what the inter-application state buys.
                              ||v|| and the h trajectory for k carried
                              steps against k reset applications.
  fom_depth_policies.png   -- the tau -> code map under cycle / hold /
                              interp for L=2 refined to N=8.
  fom_axes.png             -- the (N, dt) plane: axis 1 is the horizontal
                              ray at dt = dt_trained, axis 2 the hyperbola
                              N*dt = T_trained.
  fom_outcomes.png         -- the three pre-registered shapes of PPL(N)
                              at fixed T, illustrative only.

Run:  python3 _make_flow_or_maps_figs.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12,
    "axes.titleweight": "bold", "figure.dpi": 150,
})
GREEN, BLUE, RED = "#22C55E", "#3B82F6", "#EF4444"
PURPLE, GREY, ORANGE = "#8B5CF6", "#9CA3AF", "#F59E0B"


def integrate(h0, v0, N, dt, k_of_step, gamma=0.25, m=1.0):
    """Velocity-Verlet + O-step damping. k_of_step(j) is the spring
    constant at step j -- the one-dimensional stand-in for the depth code
    shifting V_theta's input."""
    h, v = float(h0), float(v0)
    H, V = [h], [v]
    for j in range(N):
        k = k_of_step(j)
        f = -k * h
        v = v + 0.5 * dt * f / m
        h = h + dt * v
        v = v * np.exp(-gamma * dt)          # O step
        f = -k_of_step(j) * h
        v = v + 0.5 * dt * f / m
        H.append(h); V.append(v)
    return np.array(H), np.array(V)


def fig_flow_vs_maps():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 4.7))
    # k chosen so every N on the sweep is stable for THIS illustrative
    # integrator: omega*dt = sqrt(k)*dt = 1.55 at the coarsest point, under
    # the explicit wall of 2. The deployed baoab_cfc treats the harmonic
    # part exactly and is not bound by that wall; the figure is about
    # refinement convergence, not stability.
    T, L, K0 = 8.0, 2, 0.15
    ref_H, ref_V = integrate(1.0, 0.0, 4000, T / 4000, lambda j: K0)

    axA.plot(ref_H, ref_V, color=GREY, lw=2.2, zorder=1,
             label="exact flow (N=4000)")
    for N, c in ((2, RED), (4, ORANGE), (8, BLUE)):
        H, V = integrate(1.0, 0.0, N, T / N, lambda j: K0)
        axA.plot(H, V, "-o", color=c, lw=1.6, ms=5, zorder=3,
                 label=f"N={N}, dt={T/N:g}")
    axA.set_title("A. Fixed potential: refinement CONVERGES")
    axA.set_xlabel("h"); axA.set_ylabel("v")
    axA.legend(fontsize=9, loc="lower right", framealpha=.95)
    axA.grid(alpha=.25)

    # Panel B: a depth code that CYCLES with period L. Refining N at fixed
    # T does not resolve the same tau -> k map more finely; it doubles the
    # rate at which the potential alternates.
    codes = [0.08, 0.22]   # mean = K0
    axB.plot(ref_H, ref_V, color=GREY, lw=2.2, zorder=1,
             label="flow of the MEAN potential")
    for N, c in ((2, RED), (4, ORANGE), (8, BLUE)):
        H, V = integrate(1.0, 0.0, N, T / N, lambda j: codes[j % L])
        axB.plot(H, V, "-o", color=c, lw=1.6, ms=5, zorder=3,
                 label=f"N={N}, cycled")
    axB.set_title("B. Cycled code: converges to the MEAN potential")
    axB.set_xlabel("h"); axB.set_ylabel("v")
    axB.legend(fontsize=9, loc="lower right", framealpha=.95)
    axB.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig("flow_or_maps/fom_flow_vs_maps.png", bbox_inches="tight")
    print("wrote fom_flow_vs_maps.png")


def fig_carried_vs_reset():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 4.5))
    dt, k, K = 1.0, 0.35, 8
    Hc, Vc = integrate(1.0, 0.0, K, dt, lambda j: k)
    Hr, Vr = [1.0], [0.0]
    h = 1.0
    for _ in range(K):                        # reset: v starts at 0 each time
        H1, V1 = integrate(h, 0.0, 1, dt, lambda j: k)
        h = H1[-1]; Hr.append(h); Vr.append(V1[-1])
    Hr, Vr = np.array(Hr), np.array(Vr)
    s = np.arange(K + 1)

    axA.plot(s, np.abs(Vc), "-o", color=GREEN, lw=2, ms=6,
             label="carried (h, h_prev)  ->  v accumulates")
    axA.plot(s, np.abs(Vr), "-s", color=RED, lw=2, ms=6,
             label="reset h_prev  ->  v slaved to position")
    axA.set_xlabel("application / layer index"); axA.set_ylabel("|v|")
    axA.set_title("A. Velocity is the whole difference")
    axA.legend(fontsize=9); axA.grid(alpha=.25)

    axB.plot(s, Hc, "-o", color=GREEN, lw=2, ms=6, label="carried")
    axB.plot(s, Hr, "-s", color=RED, lw=2, ms=6, label="reset")
    axB.axhline(0, color=GREY, ls=":", lw=1.2)
    axB.set_xlabel("application / layer index"); axB.set_ylabel("h")
    axB.set_title("B. and it changes where the token goes")
    axB.legend(fontsize=9); axB.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig("flow_or_maps/fom_carried_vs_reset.png", bbox_inches="tight")
    print("wrote fom_carried_vs_reset.png")


def fig_depth_policies():
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9), sharey=True)
    L, N = 2, 8
    tau = (np.arange(N) + 0.5) / N
    trained_tau = (np.arange(L) + 0.5) / L
    codes = np.array([0.0, 1.0])
    pol = {
        "cycle:  c[j mod L]": codes[np.arange(N) % L],
        "hold:  c[floor(j L / N)]": codes[(np.arange(N) * L) // N],
        "interp:  linear in tau": np.interp(tau, trained_tau, codes),
    }
    for ax, (name, vals), col in zip(axes, pol.items(), (RED, GREEN, BLUE)):
        ax.step(tau, vals, where="mid", color=col, lw=2.2, zorder=3)
        ax.plot(tau, vals, "o", color=col, ms=6, zorder=4)
        ax.step(trained_tau, codes, where="mid", color=GREY, lw=2.4,
                ls="--", zorder=2, label="trained L=2 code")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel(r"$\tau = \ell / N$")
        ax.grid(alpha=.25); ax.set_ylim(-0.35, 1.35)
    axes[0].set_ylabel("depth code (1-D stand-in)")
    axes[1].legend(fontsize=9, loc="lower right")
    fig.suptitle("Refining L=2 to N=8: only 'hold' preserves the "
                 "trained depth profile", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig("flow_or_maps/fom_depth_policies.png", bbox_inches="tight")
    print("wrote fom_depth_policies.png")


def fig_axes():
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    Ltr, dttr = 2, 4.0
    T = Ltr * dttr
    Ns = np.array([1, 2, 3, 4, 6, 8, 12, 16])
    ax.plot(Ns, np.full_like(Ns, dttr, dtype=float), "-o", color=ORANGE,
            lw=2, ms=7, label=r"axis 1: dt fixed, $T = N\Delta t$ grows")
    ax.plot(Ns, T / Ns, "-s", color=BLUE, lw=2, ms=7,
            label=r"axis 2: $T = N\Delta t$ fixed, dt shrinks")
    ax.scatter([Ltr], [dttr], s=260, marker="*", color=RED, zorder=5)
    ax.annotate("trained point\nL=2, dt=4", (Ltr, dttr), xytext=(3.3, 6.2),
                fontsize=10, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.6))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(Ns); ax.set_xticklabels([str(n) for n in Ns])
    ax.set_yticks([0.5, 1, 2, 4, 8]); ax.set_yticklabels(["0.5","1","2","4","8"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("N (integration steps at inference)")
    ax.set_ylabel(r"$\Delta t$")
    ax.set_title("Two ways to leave the trained point")
    ax.legend(fontsize=10, loc="upper right"); ax.grid(alpha=.25, which="major")
    fig.tight_layout()
    fig.savefig("flow_or_maps/fom_axes.png", bbox_inches="tight")
    print("wrote fom_axes.png")


def fig_outcomes():
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    N = np.array([1, 2, 3, 4, 6, 8, 12, 16]); base = 80.0
    flow = base + 2.2 * (2.0 / N) ** 2
    better = base + 6.0 * (2.0 / N) ** 2 - 4.0 * (1 - np.exp(-(N - 2) / 6))
    maps = base + 0.4 * np.abs(N - 2) ** 1.45
    for y, c, lab in ((flow, GREEN, "FLOW: converges as $O(\\Delta t^{p})$"),
                      (better, BLUE, "UNDER-RESOLVED: trained dt was too coarse"),
                      (maps, RED, "MAPS: valid only at the trained N")):
        ax.plot(N, y, "-o", color=c, lw=2.2, ms=7, label=lab)
    ax.axvline(2, color=GREY, ls="--", lw=1.6)
    ax.text(2.1, ax.get_ylim()[1] * 0.995, " trained N=2", color=GREY,
            fontsize=9.5, va="top")
    ax.set_xscale("log"); ax.set_xticks(N)
    ax.set_xticklabels([str(n) for n in N])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("N, at fixed $T = N\\Delta t$")
    ax.set_ylabel("validation perplexity")
    ax.set_title("Three pre-registered shapes (illustrative)")
    ax.legend(fontsize=9.5); ax.grid(alpha=.25, which="major")
    fig.tight_layout()
    fig.savefig("flow_or_maps/fom_outcomes.png", bbox_inches="tight")
    print("wrote fom_outcomes.png")


if __name__ == "__main__":
    fig_flow_vs_maps(); fig_carried_vs_reset(); fig_depth_policies()
    fig_axes(); fig_outcomes()
