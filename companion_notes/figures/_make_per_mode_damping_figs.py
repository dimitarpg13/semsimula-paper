"""Figures for Position_Dependent_Damping_and_Reinforcement_Field.md, section 12 (per-mode damping).

Six PNGs into figures/per_mode_damping/. The dynamical panels are exact solutions of the damped mode
equation z'' + gamma z' + omega^2 z = 0 over one layer step dt = 4 (the ladder's), with gamma_0 = 0.1.
One figure plots recorded perplexities, the learned-gamma(h) run's zeta medians and its gap to SR2 (labelled).

  pmd_phase_space.png     -- one stiff mode's phase-space cloud through one layer step, for five damping
                             ratios: the area kept is exp(-gamma dt), the turn is omega_d dt.
  pmd_eigen_locus.png     -- the layer map's eigenvalues in the complex plane as zeta grows: on the unit
                             circle a rotation, inside it a decaying rotation, on the real axis a contraction.
  pmd_retention_turn.png  -- amplitude kept and angle turned per layer step, by stiffness and damping ratio.
  pmd_coupling.png        -- a shared per-token zeta erases the stiff modes first; three per-mode hypotheses.
  pmd_steering.png        -- scalar damping is a throttle (force along the velocity); a damping tensor steers.
  pmd_damping_axis.png    -- recorded: perplexity against the damping of the stiff modes, and the learned
                             zeta(h) medians through training (in progress).

Run:  python3 _make_per_mode_damping_figs.py
"""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parent / "per_mode_damping"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "figure.dpi": 150,
                     "savefig.dpi": 150, "axes.spines.top": False, "axes.spines.right": False})
DT, G0 = 4.0, 0.1
CMAP = plt.get_cmap("viridis")


def layer_map(omega, gamma, t=DT):
    """2x2 matrix taking (z, v) at the start of a step to (z, v) at its end, z'' + gamma z' + omega^2 z = 0."""
    lam = np.roots([1.0, gamma, omega**2])
    if abs(lam[0] - lam[1]) < 1e-9:                      # critical
        l = lam[0].real; e = np.exp(l * t)
        return np.array([[e * (1 - l * t), e * t], [-l * l * t * e, e * (1 + l * t)]])
    l1, l2 = lam
    e1, e2 = np.exp(l1 * t), np.exp(l2 * t)
    M = np.array([[(l2 * e1 - l1 * e2) / (l2 - l1), (e2 - e1) / (l2 - l1)],
                  [l1 * l2 * (e1 - e2) / (l2 - l1), (l2 * e2 - l1 * e1) / (l2 - l1)]])
    return M.real


# ---------------------------------------------------------------- 1. phase space
rng = np.random.default_rng(0)
z0 = rng.normal(0, 1.0, 500); v0 = rng.normal(0, 1.0, 500)
w = 6.0 / DT                                             # omega*dt = 6
zetas = [0.03, 0.3, 0.6, 1.0, 3.0]
fig, ax = plt.subplots(1, 5, figsize=(16, 3.7), sharex=True, sharey=True)
for a, ze in zip(ax, zetas):
    g = G0 + 2 * ze * w if ze != 0.03 else 2 * ze * w
    M = layer_map(w, g)
    zt, vt = M @ np.vstack([z0, v0 * w])
    a.scatter(z0, v0 * w, s=4, color="0.75", label="layer input")
    a.scatter(zt, vt, s=4, color=CMAP(min(ze, 3) / 3.2), label="after one layer step")
    a.set_title(f"ζ = {ze:g}")
    a.text(0.03, 0.04, f"area kept e^(−γΔt) = {np.exp(-g * DT):.1e}\nangle turned {w * DT * np.sqrt(max(0, 1 - (g / (2 * w))**2)):.1f} rad",
           transform=a.transAxes, fontsize=8)
    a.axhline(0, color="k", lw=0.4); a.axvline(0, color="k", lw=0.4)
    a.set_xlabel("offset from the well, z")
ax[0].set_ylabel("velocity, z'"); ax[0].legend(fontsize=7, loc="upper left")
ax[0].set_xlim(-4, 4); ax[0].set_ylim(-6, 6)
fig.suptitle("One stiff mode (ω·Δt = 6) through one layer step, by damping ratio", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "pmd_phase_space.png"); plt.close(fig)

# ---------------------------------------------------------------- 2. eigenvalue locus
fig, ax = plt.subplots(figsize=(6.2, 6.2))
th = np.linspace(0, 2 * np.pi, 400)
ax.plot(np.cos(th), np.sin(th), color="0.6", lw=0.8)
zs = np.concatenate([np.linspace(0.0, 0.999, 300), np.linspace(1.001, 3, 200)])
for wdt, c in ((1.0, "#1b9e77"), (2.0, "#d95f02"), (6.0, "#7570b3")):
    pts = []
    for ze in zs:
        lam = np.roots([1.0, 2 * ze * (wdt / DT), (wdt / DT)**2])
        pts.append(np.exp(lam * DT))
    pts = np.array(pts)
    ax.plot(pts[:, 0].real, pts[:, 0].imag, color=c, lw=1.6, label=f"ω·Δt = {wdt:g}")
    ax.plot(pts[:, 1].real, pts[:, 1].imag, color=c, lw=1.6)
    for ze, mk in ((0.03, "o"), (0.6, "s"), (1.0, "D")):
        lam = np.roots([1.0, 2 * ze * (wdt / DT), (wdt / DT)**2]); e = np.exp(lam * DT)
        ax.plot(e.real, e.imag, mk, color=c, ms=5)
ax.plot([], [], "o", color="k", label="ζ = 0.03 (SR2)"); ax.plot([], [], "s", color="k", label="ζ = 0.6 (learned, layer 0)")
ax.plot([], [], "D", color="k", label="ζ = 1 (SR3-crit)")
ax.set_aspect("equal"); ax.axhline(0, color="k", lw=0.4); ax.axvline(0, color="k", lw=0.4)
ax.set(xlabel="real part", ylabel="imaginary part", xlim=(-1.1, 1.1), ylim=(-1.1, 1.1),
       title="Eigenvalues of one mode's layer map as damping grows")
ax.text(0.02, 0.02, "on the unit circle: a pure rotation\ninside: a decaying rotation\non the real axis (ζ ≥ 1): a contraction", transform=ax.transAxes, fontsize=8)
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig(OUT / "pmd_eigen_locus.png"); plt.close(fig)

# ---------------------------------------------------------------- 3. retention and turn
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
wdts = np.linspace(0.05, 12, 400)
for ze in (0.03, 0.1, 0.3, 0.6, 1.0):
    zt = ze + G0 / (2 * wdts / DT)                       # total damping ratio, gamma_0 included
    rho = np.where(zt < 1, np.exp(-zt * wdts), np.exp(-(zt - np.sqrt(np.maximum(zt**2 - 1, 0))) * wdts))
    ax[0].plot(wdts, rho, color=CMAP(min(ze, 1) / 1.1), label=f"ζ = {ze:g}")
ax[0].set(xlabel="stiffness of the mode, ω·Δt", ylabel="amplitude kept over one layer step",
          title="A. What survives a layer step", yscale="log", ylim=(1e-6, 1.2))
ax[0].axvline(7.6, color="0.5", ls="--", lw=0.8); ax[0].text(7.8, 2e-6, "median stiff mode,\nSR2 and learned γ(h)", fontsize=8, color="0.4")
ax[0].legend(fontsize=8)
zz = np.linspace(0, 1, 300)
for wdt in (1, 2, 6, 12):
    ax[1].plot(zz, wdt * np.sqrt(1 - zz**2), label=f"ω·Δt = {wdt}")
ax[1].set(xlabel="damping ratio ζ", ylabel="angle turned in one layer step (rad)", title="B. What the rotation does")
ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "pmd_retention_turn.png"); plt.close(fig)

# ---------------------------------------------------------------- 4. coupling and the per-mode hypotheses
rank = np.arange(1, 17)
wdt_r = np.geomspace(12, 0.5, 16)                        # mode 1 the stiffest
def rho_of(ze, wdt):
    zt = ze + G0 / (2 * wdt / DT)
    return np.where(zt < 1, np.exp(-zt * wdt), np.exp(-(zt - np.sqrt(np.maximum(zt**2 - 1, 0))) * wdt))
shared = np.full(16, 0.6)
h1 = np.where(rank <= 4, 0.05, 1.5)                      # carriers ring, the rest are damped
h2 = np.maximum(0.6 * np.median(wdt_r) / wdt_r - G0 * DT / (2 * wdt_r), 0.0)   # equal gamma_i: same retention
h3 = np.array([np.sqrt(max(0.0, 1 - (np.pi * max(1, round(w_ / np.pi)) / w_)**2)) if w_ > np.pi else 0.05 for w_ in wdt_r])
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
for lab, z_, c, mk in (("per token (today): one ζ for all modes", shared, "k", "o"),
                       ("H1 carriers: a few modes ring, the rest damped", h1, "#d95f02", "s"),
                       ("H2 equal retention: γᵢ the same for every mode", h2, "#1b9e77", "^"),
                       ("H3 phase code: ζᵢ sets each mode's turn to a multiple of π", h3, "#7570b3", "D")):
    ax[0].plot(rank, z_, marker=mk, color=c, ms=4, label=lab)
    ax[1].plot(rank, rho_of(z_, wdt_r), marker=mk, color=c, ms=4, label=lab)
ax[0].set(xlabel="mode rank (1 = stiffest)", ylabel="damping ratio ζᵢ", title="A. Damping by mode", ylim=(0, 3.05))
ax[1].set(xlabel="mode rank (1 = stiffest)", ylabel="amplitude kept over one layer step", title="B. What survives, by mode", yscale="log", ylim=(1e-7, 1.5))
ax[1].legend(fontsize=7, loc="lower right")
fig.suptitle("Sixteen retained modes from ω·Δt = 12 (rank 1) to 0.5 (rank 16): a shared ζ erases the stiff modes first", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "pmd_coupling.png"); plt.close(fig)

# ---------------------------------------------------------------- 5. throttle against steering
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.6))
def integrate(Gam, x0, v0, n=4000, t=8.0):
    K = np.diag([1.0, 6.0]); x, v = np.array(x0, float), np.array(v0, float); h = t / n; xs = [x.copy()]
    for _ in range(n):
        a = -K @ x - Gam @ v
        v = v + h * a; x = x + h * v; xs.append(x.copy())
    return np.array(xs)
x0, v0 = [2.0, 1.5], [0.0, 0.0]
for Gam, lab, c in ((0.3 * np.eye(2), "scalar damping, γ = 0.3 on both modes", "#2166ac"),
                    (np.diag([0.3, 3.0]), "per-mode damping, γ = 0.3 and 3.0", "#b2182b")):
    tr = integrate(Gam, x0, v0)
    ax[0].plot(tr[:, 0], tr[:, 1], color=c, label=lab)
ax[0].plot(*x0, "ko"); ax[0].plot(0, 0, "k*", ms=10)
ax[0].set(xlabel="mode 1 (soft)", ylabel="mode 2 (stiff)", title="A. Same start, same potential: different paths")
ax[0].legend(fontsize=8)
v = np.array([1.0, 1.0]) / np.sqrt(2)
for Gam, lab, c, tx in ((0.8 * np.eye(2), "scalar: −γv, along v", "#2166ac", (-1.55, -0.35)), (np.diag([0.3, 2.0]), "tensor: −Γv, off v", "#b2182b", (0.0, -1.2))):
    f = -Gam @ v
    ax[1].annotate("", xy=f, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=c, lw=2))
    ax[1].text(*tx, lab, color=c, fontsize=9)
ax[1].annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="k", lw=2)); ax[1].text(v[0], v[1] + 0.05, "velocity v", fontsize=9)
ax[1].set(xlim=(-1.6, 1.0), ylim=(-1.6, 1.0), title="B. The friction force", xlabel="mode 1", ylabel="mode 2")
ax[1].set_aspect("equal"); ax[1].axhline(0, color="k", lw=0.4); ax[1].axvline(0, color="k", lw=0.4)
fig.tight_layout(); fig.savefig(OUT / "pmd_steering.png"); plt.close(fig)

# ---------------------------------------------------------------- 6. recorded: the damping axis and the learned zeta
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
# x: total damping ratio of a median stiff mode (added ratio plus gamma_0 / (2 omega), 0.026 at SR2's median stiff mode)
pts = [(0.026, 58.32, "SR2\n(γ = 0.1, ζ ≈ 0.03)", "o"), (1.03, 63.28, "SR3-crit\n(ζ ≈ 1.03)", "D"), (30.0, 61.16, "FO-b\n(gradient flow)", "s")]
for x, y, lab, mk in pts:
    ax[0].plot(x, y, mk, color="k", ms=8); ax[0].annotate(lab, (x, y), textcoords="offset points", xytext=(8, 4), fontsize=8)
ax[0].plot([0.39, 0.59], [59.29, 59.29], color="#1b9e77", lw=4, solid_capstyle="butt")
ax[0].annotate("learned γ(h), 59.29\n(median ζ, layer 1 to layer 0)", (0.39, 59.29), textcoords="offset points", xytext=(-95, 14), fontsize=8, color="#1b9e77")
ax[0].set_xscale("log"); ax[0].set(xlabel="total damping ratio, a typical token's stiffest mode (FO-b: overdamped end)", ylabel="settled perplexity, OpenWebText",
                                   title="A. The damping axis (recorded)", xlim=(0.01, 60), ylim=(57.5, 64.8))
D = Path.home() / "Downloads"
log = D / "L2_learned_lambda_OWT_32500steps_output.txt"
sr2 = D / ("semsimula_fock_cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_sr2_L2probe_ob_untied_wsd_e5c_"
           "plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/results/L2_SR2_for_noattn_none_32500steps_output.txt")
evals = lambda f: {int(m[0].replace(",", "")): float(m[1]) for m in re.findall(r">>> EVAL step ([\d,]+)\s+val_loss=[\d.]+\s+val_ppl=([\d.]+)", f.read_text())}
if log.exists():
    rows = re.findall(r"step +(\d+)/32500 .*?zeta=\[([\d.]+)/[\d.]+ \| ([\d.]+)/[\d.]+\]", log.read_text())
    st = np.array([int(r[0]) for r in rows]); l0 = np.array([float(r[1]) for r in rows]); l1 = np.array([float(r[2]) for r in rows])
    ax[1].plot(st, l0, color="#d95f02", lw=1, label="layer 0, median ζ")
    ax[1].plot(st, l1, color="#7570b3", lw=1, label="layer 1, median ζ")
ax[1].axhline(1.0, color="k", ls="--", lw=0.8); ax[1].text(500, 1.02, "critical damping", fontsize=8)
ax[1].axhline(0.026, color="0.5", ls=":", lw=0.8); ax[1].text(500, 0.04, "SR2's effective ζ", fontsize=8, color="0.4")
ax[1].axvspan(21125, 32500, color="0.9"); ax[1].text(21400, 1.15, "decay", fontsize=8)
ax[1].set(xlabel="training step", ylabel="median added ζ (progress line, two decimals)", title="B. The learned ζ(h), through training", ylim=(0, 1.3), xlim=(0, 32500))
ax[1].legend(fontsize=8, loc="center left", bbox_to_anchor=(0.0, 0.6))
if log.exists() and sr2.exists():
    g, b = evals(log), evals(sr2); ks = sorted(set(g) & set(b))
    ax[2].plot(ks, [100 * (g[k] / b[k] - 1) for k in ks], color="#1b9e77", lw=1.5)
ax[2].axhline(0, color="k", lw=0.8); ax[2].axvspan(21125, 32500, color="0.9"); ax[2].text(21400, -4.6, "decay", fontsize=8)
ax[2].text(1500, 1.1, "behind SR2", fontsize=8, color="0.4"); ax[2].text(26500, -1.6, "ahead of SR2", fontsize=8, color="0.4")
ax[2].set(xlabel="training step", ylabel="learned γ(h) against SR2, perplexity (%)", title="C. γ(h) against SR2: lead, then deficit",
          xlim=(0, 32500), ylim=(-5.2, 2.6))
fig.tight_layout(); fig.savefig(OUT / "pmd_damping_axis.png"); plt.close(fig)

# ---------------------------------------------------------------- 7. the dial and the control (illustrative)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
c = np.geomspace(0.1, 10, 200)
for lab, curv, asym, col in (("sharp optimum: the trained wells tolerate little change", 0.06, 1.6, "#b2182b"),
                              ("broad optimum: the damping level matters little", 0.012, 1.0, "#2166ac")):
    x = np.log(c)
    y = 60 + 100 * curv * np.where(x > 0, asym * x**2, x**2)
    ax[0].plot(c, y, color=col, label=lab)
ax[0].axvline(1, color="k", ls="--", lw=0.8); ax[0].text(1.08, 66.5, "trained ζ(h)", fontsize=8)
ax[0].set_xscale("log"); ax[0].set(xlabel="dial c: ζ(h) replaced by c·ζ(h) at inference, weights fixed", ylabel="perplexity",
                                   title="A. D3, the damping dial (illustrative shapes)", ylim=(59, 68))
ax[0].text(0.11, 59.4, "c < 1: toward SR2's light damping", fontsize=8); ax[0].text(1.6, 59.4, "c > 1: toward critical", fontsize=8)
ax[0].legend(fontsize=8, loc="upper center")
lv = ["SR2\n(recorded)", "constant $\\bar{\\zeta}$\n(D4, to measure)", "learned ζ(h)\n(recorded)"]
ax[1].bar([0], [58.32], color="0.4"); ax[1].bar([2], [59.29], color="#1b9e77", alpha=0.6)
ax[1].bar([1], [59.6], color="none", edgecolor="0.4", ls="--"); ax[1].text(1, 58.75, "?", ha="center", fontsize=16, color="0.4")
for x, y in ((0, 58.32), (2, 59.29)): ax[1].text(x, y + 0.05, f"{y:.2f}", ha="center", fontsize=8)
ax[1].set_xticks([0, 1, 2], lv); ax[1].set_ylim(57.5, 60.2); ax[1].set_ylabel("settled perplexity, OpenWebText")
ax[1].text(0.5, 59.95, "SR2 → $\\bar{\\zeta}$: the value of the damping LEVEL", fontsize=8, ha="center")
ax[1].text(1.5, 59.75, "$\\bar{\\zeta}$ → ζ(h): the value of POSITION DEPENDENCE", fontsize=8, ha="center")
ax[1].set_title("B. D4: two ends recorded, the middle to measure")
fig.tight_layout(); fig.savefig(OUT / "pmd_dial_and_control.png"); plt.close(fig)

# ---------------------------------------------------------------- 8. capture at a memory well, by damping
a_d, s_w, m_ = 2.0, 1.0, 1.0
Vp = lambda x: a_d * x / s_w**2 * np.exp(-x**2 / (2 * s_w**2))
def final_x(v0, gam, t=12.0, n=3000):
    x, v = -3.0, v0; h = t / n
    for _ in range(n):
        v += h * (-Vp(x) / m_ - gam * v); x += h * v
    return x
v0s = np.linspace(0.0, 3.5, 140)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
for gam, col in ((0.02, "#b2182b"), (0.1, "#d95f02"), (0.4, "#1b9e77"), (1.2, "#2166ac")):
    ax[0].plot(v0s, [final_x(v, gam) for v in v0s], color=col, label=f"γ near the well = {gam:g}")
ax[0].axhline(0, color="k", lw=0.5)
ax[0].set(xlabel="incoming velocity", ylabel="position after the step", title="A. A memory well of depth 2: capture or pass-through", ylim=(-3.5, 12))
ax[0].legend(fontsize=8)
gs = np.geomspace(0.01, 3, 60)
vth = []
for gam in gs:
    cap = [abs(final_x(v, gam)) < 1.5 for v in v0s]
    idx = next((i for i, cpt in enumerate(cap) if not cpt and v0s[i] > 0.5), None)
    vth.append(v0s[idx] if idx is not None else np.nan)
vth = np.array(vth); end = int(np.argmax(np.isnan(vth))) if np.isnan(vth).any() else len(vth)
ax[1].plot(gs[:end], vth[:end], color="k")
if end < len(vth):
    ax[1].axvspan(gs[end - 1], gs[-1], color="0.9")
    ax[1].text(gs[end - 1] * 1.1, 1.0, "every incoming velocity\nup to 3.5 is captured", fontsize=8)
ax[1].set_xscale("log"); ax[1].set(xlabel="damping near the well, γ", ylabel="smallest incoming velocity that passes through",
                                   title="B. Damping sets the capture threshold")
ax[1].text(0.012, np.nanmax(vth[:end]) * 0.9, "above the curve: the token passes\nbelow: the well captures it", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "pmd_well_capture_damping.png"); plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
