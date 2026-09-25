"""Figures for Geodesic_Experiments_with_CfC_BAOAB.md -> figures/geodesic_cfc/.

Every curve is either an exact evaluation of a stated formula, a measured
number quoted in the document, or a clearly-labelled schematic. Nothing is
fit to data.

  gcfc_step_anatomy.png       one layer step decomposed into its substeps,
                              shaded by whether each is part of the damped
                              V_theta flow (the geodesic) or a deflection.
  gcfc_refine_vs_geodesic.png left: the MEASURED gate-3 curve (Cell 6b-7).
                              right: schematic of why a piecewise-geodesic
                              path cannot be refined -- subdividing inserts
                              punctuations, it does not resolve a curve.
  gcfc_residual_geometry.png  the E1 residual as a picture: step vector,
                              deflection vector, and the arm ladder.
  gcfc_cfc_kernels.png        psi, sinc, cos against omega*dt, with the
                              Verlet limit and the Verlet ceiling marked.
  gcfc_ln_constraint.png      LayerNorm as radial projection onto a sphere:
                              the SHAKE picture.
"""
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

OUT = Path(__file__).resolve().parent / 'geodesic_cfc'
OUT.mkdir(parents=True, exist_ok=True)
INK, GEO, DEF, MUTE, ACC = '#1b1b1b', '#2f6f9f', '#b4342a', '#9a9a9a', '#d08a1e'

# ------------------------------------------------------------ fig 1
fig, ax = plt.subplots(figsize=(11.5, 3.6))
steps = [
    ('A', 'CfC half-step\nexact harmonic\nflow of V_theta', GEO),
    ('B', 'kick\nV_theta remainder', GEO),
    ('B', 'kick\nV_phi', DEF),
    ('O', 'exact friction\nv <- exp(-gamma dt) v', GEO),
    ('A', 'CfC half-step', GEO),
    ('RC', 'reverse-channel\nkick from registers', DEF),
    ('LN', 'LayerNorm\nproject to sphere', DEF),
    ('S', 'salience decay\nregister write', DEF),
]
x = 0.0
for tag, lab, col in steps:
    w = 1.35
    ax.add_patch(FancyBboxPatch((x, 0.9), w - 0.12, 1.3, boxstyle='round,pad=0.02',
                                fc=col, ec=INK, lw=0.8, alpha=0.18))
    ax.add_patch(FancyBboxPatch((x, 0.9), w - 0.12, 1.3, boxstyle='round,pad=0.02',
                                fc='none', ec=col, lw=1.6))
    ax.text(x + (w - 0.12) / 2, 1.95, tag, ha='center', va='center', fontsize=12,
            fontweight='bold', color=col)
    ax.text(x + (w - 0.12) / 2, 1.38, lab, ha='center', va='center', fontsize=7.6, color=INK)
    x += w
ax.annotate('', xy=(x - 0.1, 0.55), xytext=(0, 0.55),
            arrowprops=dict(arrowstyle='->', color=INK, lw=1.2))
ax.text(x / 2, 0.3, 'one layer step  (h, v)  ->  (h_new, v_new)', ha='center', fontsize=9)
ax.text(0, 2.55, 'damped V_theta flow  =  the Jacobi geodesic step', color=GEO, fontsize=10, fontweight='bold')
ax.text(5.6, 2.55, 'deflections  =  what the arms of E1 remove one at a time', color=DEF, fontsize=10, fontweight='bold')
ax.set_xlim(-0.2, x + 0.1); ax.set_ylim(0.1, 2.9); ax.axis('off')
fig.tight_layout(); fig.savefig(OUT / 'gcfc_step_anatomy.png', dpi=150); plt.close(fig)

# ------------------------------------------------------------ fig 2
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
N = np.array([1, 2, 3, 4, 6, 8]); ppl = np.array([1032.73, 68.65, 133.46, 236.62, 361.09, 435.61])
a1.semilogy(N[1:], ppl[1:], 'o-', color=INK, lw=1.8, ms=6)
a1.scatter([1], [ppl[0]], s=60, facecolors='none', edgecolors=MUTE, lw=1.4)
a1.text(1.1, 780, 'N=1: static-register\nconfound, not on\nthe same curve', fontsize=7.5, color=MUTE)
a1.scatter([2], [68.65], s=110, color=GEO, zorder=5)
a1.text(2.15, 52, 'trained (N=L=2, dt=4)', color=GEO, fontsize=8.5)
a1.set_xlabel('N steps at fixed T = 8'); a1.set_ylabel('val ppl (log)')
a1.set_title('Gate 3, measured: refinement degrades monotonically')
a1.grid(alpha=0.25, which='both'); a1.set_xticks(N)

t = np.linspace(0, 1, 200)
def arc(t0, t1, k):
    tt = np.linspace(t0, t1, 40); return tt, 0.35 * np.sin(2.2 * tt + 0.3) + 0.03 * k
a2.plot(t, 0.35 * np.sin(2.2 * t + 0.3), color=GEO, lw=2.2, label='a genuine flow: one curve, refine at will')
kinks_lo, kinks_hi = [0.5], [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
for k, (kinks, off, lab, col) in enumerate(((kinks_lo, -0.25, 'N=2: 2 pieces, 1 punctuation', DEF),
                                            (kinks_hi, -0.55, 'N=8: 8 pieces, 7 punctuations', ACC))):
    bounds = [0] + kinks + [1]
    for i in range(len(bounds) - 1):
        tt = np.linspace(bounds[i], bounds[i + 1], 30)
        yy = 0.35 * np.sin(2.2 * tt + 0.3) + off + 0.06 * np.sin(9 * tt + i)
        a2.plot(tt, yy, color=col, lw=1.8)
    for kk in kinks:
        a2.scatter([kk], [0.35 * np.sin(2.2 * kk + 0.3) + off + 0.06 * np.sin(9 * kk + kinks.index(kk))],
                   s=28, color=col, zorder=4)
    a2.text(1.02, off + 0.3 * np.sin(2.2 + 0.3), lab, color=col, fontsize=8, va='center')
a2.text(1.02, 0.35 * np.sin(2.5), 'flow', color=GEO, fontsize=8, va='center')
a2.set_xlim(0, 1.45); a2.set_ylim(-1.0, 0.7); a2.axis('off')
a2.set_title('Why: subdividing inserts punctuations, it does not resolve a curve')
fig.tight_layout(); fig.savefig(OUT / 'gcfc_refine_vs_geodesic.png', dpi=150); plt.close(fig)

# ------------------------------------------------------------ fig 3
fig, ax = plt.subplots(figsize=(7.2, 5.0))
h_in = np.array([0.0, 0.0]); h_full = np.array([3.4, 1.2])
# a clean fan: geo furthest from full, each later arm closer, labels offset
arms = {'geo':     np.array([1.2, 3.3]),
        'cons':    np.array([1.9, 3.0]),
        'geo+LN':  np.array([2.6, 2.5]),
        'cons+LN': np.array([3.15, 1.75])}
ax.add_patch(FancyArrowPatch(h_in, h_full, arrowstyle='->', mutation_scale=18, color=INK, lw=2.2))
ax.text(1.5, 0.55, 'actual step  h_full - h_in', fontsize=9, color=INK, rotation=28)
for name, p in arms.items():
    col = GEO if name == 'geo' else MUTE
    ax.add_patch(FancyArrowPatch(h_in, p, arrowstyle='->', mutation_scale=14, color=col,
                                 lw=1.4 if name == 'geo' else 1.0, alpha=1 if name == 'geo' else 0.7))
    ax.scatter(*p, s=40, color=col, zorder=5)
    ax.text(p[0] - 0.05, p[1] + 0.13, name, fontsize=8.5, color=col, ha='right')
ax.add_patch(FancyArrowPatch(arms['geo'], h_full, arrowstyle='<->', mutation_scale=14, color=DEF, lw=1.6, ls='--'))
ax.text(2.55, 2.95, 'deflection\n|h_geo - h_full|', color=DEF, fontsize=9)
ax.scatter(*h_in, s=60, color=INK, zorder=6); ax.text(-0.15, -0.28, 'h_in  (captured)', fontsize=9)
ax.scatter(*h_full, s=70, color=INK, zorder=6); ax.text(3.5, 1.05, 'h_full\n(model)', fontsize=9)
ax.text(4.3, 3.35, 'R = |h_arm - h_full| / |h_full - h_in|\n'
                  'R(full) = 0 by construction (gate 0)\n'
                  'R(geo) is THE number', fontsize=9.5, ha='right',
        bbox=dict(boxstyle='round', fc='white', ec=MUTE))
ax.set_xlim(-0.4, 4.4); ax.set_ylim(-0.5, 3.9); ax.set_aspect('equal'); ax.axis('off')
ax.set_title('E1: every arm replays the SAME captured state for ONE step')
fig.tight_layout(); fig.savefig(OUT / 'gcfc_residual_geometry.png', dpi=150); plt.close(fig)

# ------------------------------------------------------------ fig 4
fig, ax = plt.subplots(figsize=(7.4, 4.2))
xx = np.linspace(1e-6, 3.5, 400)
psi = (1 - np.cos(xx)) / xx**2; sinc = np.sin(xx) / xx
ax.plot(xx, psi / 0.5, color=GEO, lw=2, label='psi(x) / (1/2)   position kernel')
ax.plot(xx, sinc, color=ACC, lw=2, label='sinc(x)           velocity kernel')
ax.plot(xx, np.cos(xx), color=DEF, lw=2, label='cos(x)            velocity decay')
ax.axhline(1, color=MUTE, ls=':', lw=1); ax.text(3.45, 1.02, 'Verlet limit', fontsize=8, color=MUTE, ha='right')
ax.axvline(2, color=INK, ls='--', lw=1); ax.text(2.04, -0.85, 'Verlet ceiling\nomega dt = 2', fontsize=8)
ax.axvspan(2, 3.5, color=DEF, alpha=0.06)
ax.text(2.7, 0.62, 'region this model\nruns in (A3: past 2)', fontsize=8, color=DEF, ha='center')
ax.set_xlabel('omega dt'); ax.set_ylabel('kernel / Verlet value'); ax.set_ylim(-1.05, 1.15)
ax.set_title('CfC A-step kernels: Verlet is the omega dt -> 0 limit')
ax.legend(frameon=False, fontsize=8.5, loc='lower left'); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(OUT / 'gcfc_cfc_kernels.png', dpi=150); plt.close(fig)

# ------------------------------------------------------------ fig 5
fig, ax = plt.subplots(figsize=(6.4, 6.2))
ax.add_patch(Circle((0, 0), 1.0, fc='none', ec=INK, lw=1.6))
ax.text(0.0, -1.13, 'LayerNorm image: |h - mean| = sqrt(d)  (a sphere)', ha='center', fontsize=8.5)
# march clockwise along the TOP of the circle so every arrow stays in frame
for k, ang in enumerate((155, 120, 85)):
    q = np.array([np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))])
    tang = np.array([q[1], -q[0]])                   # clockwise tangent
    off = q + 0.62 * tang + 0.20 * q                 # a step that leaves the sphere
    proj = off / np.linalg.norm(off)
    ax.add_patch(FancyArrowPatch(q, off, arrowstyle='->', mutation_scale=13, color=GEO, lw=1.8))
    ax.add_patch(FancyArrowPatch(off, proj, arrowstyle='->', mutation_scale=13, color=DEF, lw=1.5, ls='--'))
    ax.scatter(*q, s=40, color=INK, zorder=7)
    ax.scatter(*off, s=26, color=GEO, zorder=5)
    ax.scatter(*proj, s=34, color=DEF, zorder=6)
ax.text(-1.68, 1.44, 'flow step: leaves the sphere', color=GEO, fontsize=9)
ax.text(-1.68, 1.28, 'LN: radial projection back onto it', color=DEF, fontsize=9)
ax.text(0.0, -1.42, 'Radial projection is the SHAKE step for the constraint |Ph|^2 = d,\n'
                    'because the gradient of |h|^2 is radial. Position is constrained;\n'
                    'the velocity is NOT tangent-projected (SHAKE without RATTLE).',
        fontsize=8, ha='center', bbox=dict(boxstyle='round', fc='white', ec=MUTE))
ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.85, 1.6); ax.set_aspect('equal'); ax.axis('off')
ax.set_title('E4: LayerNorm as a holonomic constraint, not a perturbation')
fig.tight_layout(); fig.savefig(OUT / 'gcfc_ln_constraint.png', dpi=150); plt.close(fig)

# ------------------------------------------------------------ fig 6
# Why inertia constrains the next state: second-order vs first-order step.
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
h = np.array([0.0, 0.0]); v = np.array([2.2, 0.9]); F = np.array([-0.6, 1.6])
inert = h + v; forced = inert + 0.55 * F
a1.add_patch(FancyArrowPatch(h, inert, arrowstyle='->', mutation_scale=16, color=GEO, lw=2.4))
a1.add_patch(FancyArrowPatch(inert, forced, arrowstyle='->', mutation_scale=16, color=DEF, lw=2.0))
a1.add_patch(FancyArrowPatch(h, forced, arrowstyle='->', mutation_scale=14, color=INK, lw=1.2, ls='--'))
a1.scatter(*h, s=60, color=INK, zorder=6); a1.scatter(*forced, s=60, color=INK, zorder=6)
a1.text(-0.1, -0.28, '(h, v) at layer l', fontsize=9)
a1.text(forced[0] + 0.08, forced[1] + 0.05, 'h at layer l+1', fontsize=9)
a1.text(1.0, 0.15, 'dt * v   inertial part\nknown from the state', color=GEO, fontsize=9)
a1.text(2.15, 1.55, '(dt^2/2m) F\nforced part', color=DEF, fontsize=9)
a1.text(0.05, 2.35, 'SECOND ORDER: the next position is\n'
                    'constrained by the current velocity.\n'
                    'The force enters at order dt^2.', fontsize=9.5,
        bbox=dict(boxstyle='round', fc='white', ec=MUTE))
a1.set_xlim(-0.4, 3.4); a1.set_ylim(-0.5, 3.1); a1.set_aspect('equal'); a1.axis('off')
a1.set_title('inertial fraction  =  |dt v| / |step|')
h2 = np.array([0.0, 0.0])
for k, (ang, col) in enumerate(((20, MUTE), (95, MUTE), (160, MUTE), (250, MUTE), (320, INK))):
    e = 1.9 * np.array([np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))])
    a2.add_patch(FancyArrowPatch(h2, e, arrowstyle='->', mutation_scale=14, color=col,
                                 lw=2.0 if col == INK else 1.0, alpha=1 if col == INK else 0.45))
a2.scatter(*h2, s=60, color=INK, zorder=6)
a2.text(-0.15, -0.35, 'h at layer l', fontsize=9)
a2.text(1.45, -1.6, 'h at layer l+1\n= f_l(h): any direction', fontsize=9)
a2.text(-2.3, 2.35, 'FIRST ORDER (transformer): the state is h alone.\n'
                    'The next position is an arbitrary learned map.\n'
                    'Nothing carries over between layers.', fontsize=9.5,
        bbox=dict(boxstyle='round', fc='white', ec=MUTE))
a2.set_xlim(-2.5, 2.5); a2.set_ylim(-2.2, 3.1); a2.set_aspect('equal'); a2.axis('off')
a2.set_title('no inertial part: the whole step is "forced"')
fig.suptitle('What a second-order state buys, in principle -- E3 measures whether it buys it in practice', fontsize=10.5)
fig.tight_layout(); fig.savefig(OUT / 'gcfc_inertia_vs_forced.png', dpi=150); plt.close(fig)

# ------------------------------------------------------------ fig 7
# The three E3 metrics as pictures.
fig, (b1, b2, b3) = plt.subplots(1, 3, figsize=(13.5, 4.0))
# (a) direction coherence
p0, p1, p2 = np.array([0, 0]), np.array([1.6, 0.9]), np.array([2.9, 1.9])
b1.add_patch(FancyArrowPatch(p0, p1, arrowstyle='->', mutation_scale=14, color=INK, lw=2))
b1.add_patch(FancyArrowPatch(p1, p2, arrowstyle='->', mutation_scale=14, color=INK, lw=2))
b1.add_patch(FancyArrowPatch(p1, p1 + (p1 - p0) * 0.8, arrowstyle='->', mutation_scale=12, color=GEO, lw=1.4, ls=':'))
b1.text(0.5, 0.65, 's_{l-1}', fontsize=10); b1.text(2.25, 1.15, 's_l', fontsize=10)
b1.text(2.35, 0.55, 'same direction\nas before', color=GEO, fontsize=8)
b1.text(0.0, 2.35, 'coherence  c_l = cos(s_l, s_{l-1})\n1 = keeps going, 0 = unrelated, <0 = reverses', fontsize=9,
        bbox=dict(boxstyle='round', fc='white', ec=MUTE))
b1.set_xlim(-0.3, 3.6); b1.set_ylim(-0.4, 3.0); b1.set_aspect('equal'); b1.axis('off'); b1.set_title('(a) direction coherence')
# (b) velocity forecast
q0, qv, q1 = np.array([0, 0]), np.array([2.0, 1.2]), np.array([2.5, 0.5])
b2.add_patch(FancyArrowPatch(q0, qv, arrowstyle='->', mutation_scale=14, color=GEO, lw=2))
b2.add_patch(FancyArrowPatch(qv, q1, arrowstyle='->', mutation_scale=14, color=DEF, lw=1.8, ls='--'))
b2.add_patch(FancyArrowPatch(q0, q1, arrowstyle='->', mutation_scale=12, color=INK, lw=1.2))
b2.scatter(*q0, s=50, color=INK, zorder=5); b2.scatter(*qv, s=40, color=GEO, zorder=5); b2.scatter(*q1, s=50, color=INK, zorder=5)
b2.text(0.9, 1.05, 'forecast\nh + dt v', color=GEO, fontsize=9); b2.text(2.45, 0.95, 'error', color=DEF, fontsize=9)
b2.text(2.55, 0.3, 'actual h_{l+1}', fontsize=9); b2.text(-0.1, -0.3, 'h_l', fontsize=9)
b2.text(0.0, 2.35, 'forecast error  e_l = |h_{l+1} - LN(h_l + dt v_l)| / |step|\n'
                   'transformer analogue: v := h_l - h_{l-1}', fontsize=9,
        bbox=dict(boxstyle='round', fc='white', ec=MUTE))
b2.set_xlim(-0.3, 3.6); b2.set_ylim(-0.4, 3.0); b2.set_aspect('equal'); b2.axis('off'); b2.set_title('(b) velocity as forecast')
# (c) perturbation growth
t = np.linspace(0, 3, 50); mid = 0.4 * np.sin(1.3 * t) + 0.9
for sc, col, lab in ((0.12, GEO, 'contractive: |dh_L| < |dh_0|'), (0.55, DEF, 'amplifying: |dh_L| > |dh_0|')):
    w = 0.25 * (1 + (sc / 0.12 - 1) * t / 3) if sc > 0.2 else 0.25 * np.exp(-0.5 * t)
    b3.fill_between(t, mid - w, mid + w, color=col, alpha=0.18)
    b3.plot(t, mid + w, color=col, lw=1.0); b3.plot(t, mid - w, color=col, lw=1.0)
b3.plot(t, mid, color=INK, lw=1.8)
b3.text(0.05, 2.35, 'growth  g = |dh_L| / |dh_0|,  h_0 -> h_0 + delta\n'
                    'g < 1: perturbations die (predictable)\ng > 1: sensitive to initial state', fontsize=9,
        bbox=dict(boxstyle='round', fc='white', ec=MUTE))
b3.text(2.1, 1.75, 'amplifying', color=DEF, fontsize=8.5); b3.text(2.1, 0.95, 'contractive', color=GEO, fontsize=8.5)
b3.set_xlim(-0.1, 3.3); b3.set_ylim(-0.4, 3.0); b3.axis('off'); b3.set_title('(c) perturbation growth')
fig.tight_layout(); fig.savefig(OUT / 'gcfc_e3_metrics.png', dpi=150); plt.close(fig)

print('wrote:')
for f in sorted(OUT.glob('*.png')):
    print(f'   {f.relative_to(Path(__file__).resolve().parent.parent)}  {f.stat().st_size // 1024} KB')
