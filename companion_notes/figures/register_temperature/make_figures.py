"""Figures for Register_Temperature_Instability_in_the_Fock_Creation_Gate.md.

Every number here is measured, not simulated, except the two phase/Riccati
panels which are explicitly analytical illustrations of the derived ODEs.
Sources:
  - tau history: sweep_log_tau_history() over the 12 _spikebatch.pt bundles
  - per-register / per-layer gradients: probe_hot_rows() full-batch pass
  - score-bound sweep: QKVCreationGate_v21 with W_Q/W_K scaled, readout patched
"""
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = __file__.rsplit('/', 1)[0]
plt.rcParams.update({'figure.dpi': 140, 'font.size': 9,
                     'axes.grid': True, 'grid.alpha': 0.3})

# ----------------------------------------------------------------- schedule
LR, TOTAL, WU, SE = 3e-4, 100_000, 5_000, 65_000
FLOOR, WD, TAU0 = LR * 0.05, 0.01, 8.0


def lr_at(s):
    if s < WU:
        return LR * (s + 1) / WU
    if s < SE:
        return LR
    p = min((s - SE) / (TOTAL - SE), 1.0)
    return FLOOR + (LR - FLOOR) * 0.5 * (1 + math.cos(math.pi * p))


def decay_only_tau():
    l, xs, ys = math.log(TAU0), [], []
    for s in range(TOTAL):
        if s % 200 == 0:
            xs.append(s); ys.append(math.exp(l))
        l *= (1 - lr_at(s) * WD)
    return xs, ys


# measured: step, tau[14], pool median, pool max
HIST = [(68313, 5.3409, 6.5061, 8.8367), (68415, 5.3399, 6.5216, 8.8589),
        (68701, 5.3421, 6.5061, 8.8793), (70431, 5.2267, 6.4004, 8.9994),
        (70522, 5.2310, 6.3891, 9.0062), (70660, 5.2620, 6.4512, 8.9492),
        (70915, 5.2536, 6.4685, 8.9924), (71063, 5.2478, 6.4429, 9.0488),
        (71194, 5.2254, 6.4498, 9.0751), (71448, 5.2331, 6.4404, 8.9818),
        (71703, 5.2282, 6.4634, 8.9700), (71985, 5.2104, 6.4778, 8.9529)]

# ============================================ Fig 1: tau trajectory
fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.0),
                             gridspec_kw={'width_ratios': [1.35, 1]})
xs, ys = decay_only_tau()
a0.plot(xs, ys, lw=2.2, color='crimson',
        label='pure weight decay from init 8.0 (zero loss gradient)')
a0.axhline(TAU0, ls=':', c='gray', lw=1)
a0.axhline(4.0, ls='--', c='seagreen', lw=1.4, label='proposed floor tau = 4.0')
st = [h[0] for h in HIST]
a0.scatter(st, [h[1] for h in HIST], s=22, c='navy', zorder=5,
           label='observed register 14')
a0.scatter(st, [h[2] for h in HIST], s=22, c='darkorange', marker='s',
           zorder=5, label='observed pool median')
a0.scatter(st, [h[3] for h in HIST], s=22, c='purple', marker='^', zorder=5,
           label='observed pool max')
a0.axvspan(WU, SE, alpha=0.05, color='blue')
a0.text(34000, 8.6, 'WSD stable phase', fontsize=7.5, color='navy', alpha=.7)
a0.set_xlabel('training step'); a0.set_ylabel('tau')
a0.set_title('Weight decay alone reproduces the pool-wide cooling', fontsize=10)
a0.legend(fontsize=7.2, loc='lower left'); a0.set_ylim(3.5, 9.4)

a1.plot(xs, ys, lw=2.2, color='crimson')
a1.scatter(st, [h[1] for h in HIST], s=34, c='navy', zorder=5)
a1.scatter(st, [h[2] for h in HIST], s=34, c='darkorange', marker='s', zorder=5)
a1.scatter(st, [h[3] for h in HIST], s=34, c='purple', marker='^', zorder=5)
a1.annotate('decay line 5.42', (71985, 5.4154), (69200, 5.75), fontsize=7.5,
            color='crimson', arrowprops=dict(arrowstyle='->', color='crimson'))
a1.annotate('reg 14: 5.21\n(below the line)', (71985, 5.2104), (69000, 4.75),
            fontsize=7.5, color='navy',
            arrowprops=dict(arrowstyle='->', color='navy'))
a1.annotate('median 6.48\n(resists decay)', (71985, 6.4778), (68900, 6.9),
            fontsize=7.5, color='darkorange',
            arrowprops=dict(arrowstyle='->', color='darkorange'))
a1.set_xlim(68000, 72400); a1.set_ylim(4.4, 9.4)
a1.set_xlabel('training step')
a1.set_title('Detail: the 12 captured bundles', fontsize=10)
fig.tight_layout(); fig.savefig(f'{OUT}/tau_trajectory.png'); plt.close(fig)

# ============================================ Fig 2: phase portrait
# C_EFF is measured, not chosen: register 14's loss-gradient residual over
# steps 68313-71985 is -0.0073 in log tau over 3672 steps, i.e. -1.988e-6
# per step; dividing by the window-mean lr 2.86e-4 gives A = 0.00695, and
# C_eff = A * tau = 0.00695 * 5.23. Units are "per step, divided by lr",
# which is exactly the unit AdamW's DECOUPLED decay also lives in, so the
# comparison between C_eff and gamma below is apples to apples.
C_EFF, L0 = 0.036, math.log(8.0)


def gdot(x, gam, anchor):
    return -C_EFF * math.exp(-x) - gam * (x - anchor)


fig, (p0, p1) = plt.subplots(1, 2, figsize=(11.4, 4.3),
                             gridspec_kw={'width_ratios': [1.25, 1]})
ell = [0.15 + i * (2.45 - 0.15) / 600 for i in range(601)]
K = 1e3  # display in units of 1e-3
p0.axhline(0, c='k', lw=1)
p0.plot(ell, [K * -C_EFF * math.exp(-x) for x in ell], lw=2.2, c='crimson',
        label='bare dynamics, weight decay 0: no root, always sharpening')
p0.plot(ell, [K * gdot(x, 0.01, 0.0) for x in ell], lw=2.2, c='darkorange',
        label='weight decay 0.01 anchored at 0: what the run had, worse')
p0.plot(ell, [K * gdot(x, 0.01, L0) for x in ell], lw=2.0, c='steelblue',
        ls='--',
        label='weight decay 0.01 anchored at log 8: still no root, marginal')
p0.plot(ell, [K * gdot(x, 0.02, L0) for x in ell], lw=2.4, c='seagreen',
        label='weight decay 0.02 anchored at log 8: stable root')
fp = None
for i in range(600):
    a, b = gdot(ell[i], 0.02, L0), gdot(ell[i + 1], 0.02, L0)
    if a > 0 >= b:
        fp = ell[i]
if fp:
    p0.plot([fp], [0], 'o', ms=10, mfc='white', mec='seagreen', mew=2.2,
            zorder=6)
    p0.annotate(f'stable fixed point\ntau {math.exp(fp):.2f}', (fp, 0),
                (fp - 0.48, 9.0), fontsize=8, color='seagreen', ha='center',
                arrowprops=dict(arrowstyle='->', color='seagreen'))
for x, lab, c in ((math.log(5.21), 'reg 14\n5.21', 'navy'),
                  (math.log(6.48), 'median\n6.48', 'darkorange'),
                  (L0, 'init\n8.0', 'gray')):
    p0.axvline(x, ls=':', c=c, alpha=.65)
    p0.text(x, -21.0, lab, fontsize=7, color=c, ha='center')
p0.set_xlabel('log tau'); p0.set_ylabel('drift rate per step / lr  (x 1e-3)')
p0.set_title('A restoring force manufactures the equilibrium\n'
             'the bare dynamics lack', fontsize=10)
p0.legend(fontsize=7.2, loc='lower left'); p0.set_ylim(-30, 13)

# --- saddle-node: roots of gamma (L0 - l) = C exp(-l) as gamma varies
gams, up, lo = [], [], []
g = 0.0060
while g <= 0.0405:
    rs = []
    for i in range(20000):
        x = -1.2 + i * (2.4 + 1.2) / 20000
        y = -1.2 + (i + 1) * (2.4 + 1.2) / 20000
        if gdot(x, g, L0) * gdot(y, g, L0) <= 0:
            rs.append((x + y) / 2)
    if len(rs) >= 2:
        gams.append(g); lo.append(math.exp(rs[0])); up.append(math.exp(rs[-1]))
    g += 0.00025
gc = math.e * C_EFF / 8.0
p1.plot(gams, up, lw=2.4, c='seagreen', label='stable branch, the equilibrium')
p1.plot(gams, lo, lw=2.0, c='crimson', ls='--',
        label='unstable branch, the separatrix')
p1.axvline(gc, c='k', ls=':', lw=1.3)
p1.text(gc + 0.0016, 0.55,
        f'fold at weight decay {gc:.4f}\n(= e C / tau init)', fontsize=7.5)
p1.axvline(0.01, c='darkorange', lw=1.6)
p1.text(0.0104, 0.12,
        "run's WEIGHT_DECAY\n0.01: left of the fold,\nno equilibrium",
        fontsize=7.2, color='darkorange')
p1.axhline(8.0, ls=':', c='gray')
p1.text(0.033, 8.25, 'tau init 8.0', fontsize=7.5, color='gray')
p1.set_yscale('log')
p1.set_xlabel('AdamW weight decay coefficient, anchored at log 8\n'
              '(NOT the BAOAB friction FIXED_GAMMA)')
p1.set_ylabel('equilibrium tau')
p1.set_title('Saddle-node: the equilibrium exists only\n'
             'above a threshold decay strength', fontsize=10)
p1.legend(fontsize=7.5, loc='center right'); p1.set_ylim(0.1, 20)
fig.tight_layout(); fig.savefig(f'{OUT}/phase_portrait.png'); plt.close(fig)

# ============================================ Fig 3: Riccati
fig, ax = plt.subplots(figsize=(7.2, 4.0))
v0, k = 0.19, 1.0
tstar = 1 / (k * v0)
ts = [i * tstar * 0.985 / 400 for i in range(401)]
ax.plot(ts, [v0 / (1 - k * v0 * t) for t in ts], lw=2.3, c='crimson',
        label='Riccati  dv/dt = eta C v^2  (the derived dynamics)')
ax.plot(ts, [v0 * math.exp(k * v0 * t) for t in ts], lw=1.8, ls='--',
        c='darkorange', label='exponential, for comparison')
ax.plot(ts, [v0 * (1 + k * v0 * t) for t in ts], lw=1.8, ls=':', c='gray',
        label='linear, for comparison')
ax.plot(ts, [min(v0 / (1 - k * v0 * t), 1 / 4.0) for t in ts], lw=2.3,
        c='seagreen', label='same dynamics with a floor tau >= 4')
ax.axvline(tstar, ls='--', c='crimson', alpha=.5)
ax.text(tstar * 0.985, 0.55, 'finite-time\nblow-up', fontsize=8, color='crimson',
        ha='right')
ax.set_xlabel('time (arbitrary units)')
ax.set_ylabel('v = 1 / tau   (inverse temperature)')
ax.set_title('The loop is superlinear, and a bound is what terminates it',
             fontsize=10)
ax.set_ylim(0, 0.75); ax.legend(fontsize=8, loc='upper left')
fig.tight_layout(); fig.savefig(f'{OUT}/riccati.png'); plt.close(fig)

# ============================================ Fig 4: measured gradients
fig, (b0, b1) = plt.subplots(1, 2, figsize=(11, 3.9))
g522 = {14: 1.6326, 1: .2335, 15: -.0335, 13: .0267, 7: .0225, 12: .0210,
        0: -.0193, 17: .0166, 24: .0135, 11: -.0129, 4: -.0123, 20: -.0096,
        21: -.0091, 29: -.0078, 6: -.0072, 26: .0070, 2: .0067, 28: .0062,
        10: -.0062, 9: -.0052, 3: -.0049, 8: -.0040, 19: -.0035, 25: .0032,
        23: .0031, 31: -.0025, 16: -.0023, 5: .0012, 30: .0009, 27: .0005,
        22: -.0004, 18: .0001}
g194 = {14: 345.3764, 11: .1073, 15: .0575, 31: .0467, 8: .0381, 12: .0258,
        21: -.0223, 29: -.0191, 18: .0134, 27: -.0111, 6: -.0109, 24: .0108,
        1: .0090, 16: .0087, 10: .0087, 2: .0084, 17: .0081, 26: .0077,
        25: .0070, 3: -.0058, 9: .0045, 13: -.0040, 20: .0039, 4: .0038,
        28: .0031, 19: -.0026, 0: .0024, 30: .0022, 7: -.0016, 5: .0012,
        23: .0010, 22: .0002}
idx = list(range(32))
w = 0.42
b0.bar([i - w / 2 for i in idx], [abs(g522[i]) for i in idx], w,
       label='step 70522', color='steelblue')
b0.bar([i + w / 2 for i in idx], [abs(g194[i]) for i in idx], w,
       label='step 71194', color='indianred')
b0.set_yscale('log'); b0.set_xlabel('register index')
b0.set_ylabel('|grad| of log_tau  (log scale)')
b0.set_title('log_tau gradient: 97.8% then 100.0% is register 14', fontsize=10)
b0.legend(fontsize=8); b0.axvline(14, color='k', ls=':', alpha=.45)

r522 = [230.8743, -52.2892, -16.9583, .3988, .1861, .0474, .0635, .0275]
r194 = [406.1635, -10.8978, 4.4561, 1.7077, .1066, .0072, -.0080, -.0161]
lay = list(range(8))
b1.bar([i - w / 2 for i in lay], [abs(v) for v in r522], w,
       label='step 70522', color='steelblue')
b1.bar([i + w / 2 for i in lay], [abs(v) for v in r194], w,
       label='step 71194', color='indianred')
b1.set_yscale('log'); b1.set_xlabel('layer index')
b1.set_ylabel('|grad| of reverse_channel_scale  (log scale)')
b1.set_title('reverse_channel_scale: 94.6% then 99.9% is layer 0', fontsize=10)
b1.legend(fontsize=8); b1.set_xticks(lay)
fig.tight_layout(); fig.savefig(f'{OUT}/gradient_concentration.png')
plt.close(fig)

# ============================================ Fig 5: score bound
fig, ax = plt.subplots(figsize=(7.0, 4.1))
sc = [1, 3, 10, 30]
cur = [17.0, 351.2, 4303.1, 14461.0]
qk = [11.58, 12.34, 13.48, 14.70]
ax.loglog(sc, cur, 'o-', lw=2.2, c='crimson', ms=7,
          label='current gate: scores / tau')
ax.loglog(sc, qk, 's-', lw=2.2, c='seagreen', ms=7,
          label='qk_norm gate: sigma_k cos(theta)')
ax.axhline(40, ls='--', c='gray', lw=1.3)
ax.text(1.05, 44, 'readout clamp = 40', fontsize=7.5, color='gray')
ax.axhline(100, ls='--', c='seagreen', lw=1.3)
ax.text(1.05, 110, 'sigma_max = 100 (hard ceiling)', fontsize=7.5, c='seagreen')
ax.axhline(4980, ls=':', c='crimson', lw=1.3)
ax.text(1.05, 5500, 'register 14, measured on the live run: 4980',
        fontsize=7.5, color='crimson')
ax.set_xlabel('scale factor applied to W_Q and W_K only')
ax.set_ylabel('peak |scaled score|')
ax.set_title('QK-norm removes the query-key channel; nothing else does',
             fontsize=10)
ax.legend(fontsize=8, loc='center left')
fig.tight_layout(); fig.savefig(f'{OUT}/score_bound.png'); plt.close(fig)

print('wrote 5 figures to', OUT)
