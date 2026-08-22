"""Figures for the PyTorch CfC+BAOAB implementation deep dive."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.dpi": 160,
})


def sinc_unnorm(x):
    return np.where(np.abs(x) < 1e-12, 1.0, np.sin(x) / x)


def psi(x):
    return 0.5 * sinc_unnorm(x / 2.0) ** 2


def cfc_step(h, v, k, m, dt):
    omega = np.sqrt(np.maximum(k / m, 0.0))
    wt = omega * dt
    f = -k * h
    h_new = h + (dt * sinc_unnorm(wt)) * v + (dt * dt / m) * psi(wt) * f
    v_new = np.cos(wt) * v + (dt / m) * sinc_unnorm(wt) * f
    return h_new, v_new


def verlet_step(h, h_prev, k, m, dt):
    f = -k * h
    h_new = 2 * h - h_prev + (dt * dt / m) * f
    return h_new, h


def fig_sinc_psi():
    x = np.linspace(-8, 8, 800)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(x, sinc_unnorm(x), color="#1d4ed8", lw=2, label=r"$\mathrm{sinc}(x)=\sin x/x$")
    ax.plot(x, psi(x), color="#b45309", lw=2, label=r"$\psi(x)=(1-\cos x)/x^2$")
    ax.axhline(1.0, color="#1d4ed8", ls=":", lw=1)
    ax.axhline(0.5, color="#b45309", ls=":", lw=1)
    ax.axvline(0, color="#9ca3af", lw=0.8)
    ax.set_xlabel(r"$x = \omega\,\Delta t$")
    ax.set_ylabel("value")
    ax.set_title("Branch-free special functions used by cfc_substep")
    ax.legend(loc="upper right")
    ax.set_xlim(-8, 8)
    ax.set_ylim(-0.4, 1.2)
    ax.grid(True, alpha=0.3)
    fig.savefig(OUT / "cfc_sinc_psi.png")
    plt.close(fig)


def fig_phase_space():
    m, dt, steps = 1.0, 1.0, 24
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), sharey=False)

    cases = [
        (0.25, "Mild spring  (K = 0.25,  omega dt = 0.5)"),
        (1.0e4, "Stiff spring  (K = 1e4,  omega dt = 100)"),
    ]
    for ax, (k, title) in zip(axes, cases):
        h, v = 0.5, 0.0
        cfc_h, cfc_v = [h], [v]
        for _ in range(steps):
            h, v = cfc_step(h, v, k, m, dt)
            cfc_h.append(h)
            cfc_v.append(v)

        h, hp = 0.5, 0.5
        ver_h, ver_v = [h], [0.0]
        blew = False
        for _ in range(steps):
            h, hp = verlet_step(h, hp, k, m, dt)
            vel = (h - hp) / dt
            if not np.isfinite(h) or abs(h) > 20:
                blew = True
                break
            ver_h.append(h)
            ver_v.append(vel)

        ax.plot(cfc_h, cfc_v, "o-", color="#065f46", ms=3.5, lw=1.4,
                label="CfC (exact rotation)")
        if blew:
            ax.plot(ver_h, ver_v, "s--", color="#b91c1c", ms=3.5, lw=1.2,
                    label="explicit Verlet (overflow)")
            ax.annotate("diverges", xy=(ver_h[-1], ver_v[-1]),
                        xytext=(0.15, 8 if k > 1 else 0.4),
                        color="#b91c1c", fontsize=9,
                        arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1))
        else:
            ax.plot(ver_h, ver_v, "s--", color="#b91c1c", ms=3.5, lw=1.2,
                    label="explicit Verlet")

        omega = np.sqrt(k / m)
        if k < 10:
            th = np.linspace(0, 2 * np.pi, 200)
            ax.plot(0.5 * np.cos(th), -0.5 * omega * np.sin(th),
                    color="#9ca3af", lw=0.8, ls=":")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("position  h")
        ax.set_ylabel("velocity  v")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    fig.suptitle("Same spring, same dt: CfC stays on the orbit, Verlet does not",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "cfc_phase_space_verlet_vs_cfc.png")
    plt.close(fig)


def fig_aboba_timeline():
    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    boxes = [
        (0.4, "A", "cfc_substep(dt/2)\nharmonic flow at h", "#065f46"),
        (2.7, "B", "kick at h_mid\nf_theta + f_phi - f_harm", "#1d4ed8"),
        (5.0, "O", "ou_step(dt)\nexp(-gamma dt)", "#b45309"),
        (7.3, "A", "cfc_substep(dt/2)\nharmonic flow at h_mid", "#065f46"),
    ]
    for x, letter, desc, color in boxes:
        rect = plt.Rectangle((x, 1.3), 2.0, 1.7, facecolor=color,
                             edgecolor="none", alpha=0.18)
        ax.add_patch(rect)
        ax.add_patch(plt.Rectangle((x, 1.3), 2.0, 1.7, fill=False,
                                   edgecolor=color, lw=1.8))
        ax.text(x + 1.0, 2.65, letter, ha="center", va="center",
                fontsize=18, fontweight="bold", color=color)
        ax.text(x + 1.0, 1.85, desc, ha="center", va="center",
                fontsize=8, color="#111827")

    for x in (2.4, 4.7, 7.0):
        ax.annotate("", xy=(x + 0.25, 2.15), xytext=(x - 0.25, 2.15),
                    arrowprops=dict(arrowstyle="->", color="#374151", lw=1.4))

    ax.text(5.0, 3.55, "One layer: ABOBA  (one force evaluation, T = 0 default)",
            ha="center", fontsize=12, fontweight="bold")
    ax.text(5.0, 0.55,
            "Velocity is decoded from (h, h_prev) before A and encoded back after the second A.",
            ha="center", fontsize=9, color="#374151")
    fig.savefig(OUT / "cfc_aboba_timeline.png")
    plt.close(fig)


def fig_force_split():
    h = np.linspace(-3.2, 3.2, 400)
    mu, a, w = 0.0, 1.6, 1.0
    g = w * np.exp(-0.5 * a * (h - mu) ** 2)
    f_true = -(a * (h - mu)) * g
    k_diag = g * a
    s = g * a * mu
    f_harm = s - k_diag * h
    residual = f_true - f_harm  # ~0 for rank-0 / diagonal

    # add a fake residual bump to illustrate the B-kick remainder
    f_phi = 0.25 * np.exp(-0.5 * ((h - 1.2) / 0.55) ** 2) * np.sin(2.2 * h)

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.plot(h, f_true, color="#111827", lw=2.2, label=r"$f_\theta$  (true diagonal force)")
    ax.plot(h, f_harm, color="#065f46", lw=1.8, ls="--",
            label=r"$f_{\mathrm{harm}}=s-k_{\mathrm{diag}}h$  (A-substep)")
    ax.plot(h, f_phi, color="#1d4ed8", lw=1.6,
            label=r"$f_\phi$  (numerical, B-kick)")
    ax.axhline(0, color="#9ca3af", lw=0.8)
    ax.axvline(0, color="#9ca3af", lw=0.6, ls=":")
    ax.set_xlabel("hidden state  h  (one coordinate)")
    ax.set_ylabel("force")
    ax.set_title("Force split: CfC integrates f_harm exactly; B kicks the rest")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.savefig(OUT / "cfc_force_split.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_sinc_psi()
    fig_phase_space()
    fig_aboba_timeline()
    fig_force_split()
    print("wrote", list(OUT.glob("cfc_*.png")))
