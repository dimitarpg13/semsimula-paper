#!/usr/bin/env python3
"""Figure for Jacobi_Metric_from_the_Euler_Lagrange_Equations.md (§8): the example system.

A particle in the book's Gaussian well V = m u^2 (1 - exp(-k^2 |x|^2)) in two dimensions, integrated
by fourth-order Runge-Kutta, without friction and with gamma = 0.15. Along each trajectory the
Jacobi acceleration x'' + Gamma~(x', x') is computed from the Jacobi metric of the current energy and
split into its components along and across the motion.

    python3 _make_jacobi_figs.py      -> figures/jacobi/jacobi_example.png
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / 'jacobi'; OUT.mkdir(exist_ok=True)
m, ups, kap, GAM, DT, TMAX = 1.3, 1.0, 0.8, 0.15, 1e-4, 6.0
V = lambda x: m * ups ** 2 * (1 - np.exp(-kap ** 2 * (x @ x)))
gV = lambda x: 2 * m * ups ** 2 * kap ** 2 * x * np.exp(-kap ** 2 * (x @ x))

def run(g):
    acc = lambda x, v: -gV(x) / m - g * v
    x = np.array([1.2, 0.0]); v = np.array([0.1, 0.9]); X, Vv, A = [], [], []
    for _ in range(int(TMAX / DT)):
        a = acc(x, v); X.append(x.copy()); Vv.append(v.copy()); A.append(a.copy())
        k1x, k1v = v, a
        k2x, k2v = v + DT / 2 * k1v, acc(x + DT / 2 * k1x, v + DT / 2 * k1v)
        k3x, k3v = v + DT / 2 * k2v, acc(x + DT / 2 * k2x, v + DT / 2 * k2v)
        k4x, k4v = v + DT * k3v, acc(x + DT * k3x, v + DT * k3v)
        x = x + DT / 6 * (k1x + 2 * k2x + 2 * k3x + k4x); v = v + DT / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    X, Vv, A = map(np.array, (X, Vv, A))
    sp2 = (Vv ** 2).sum(1)
    E = 0.5 * m * sp2 + np.array([V(p) for p in X])
    gs = np.array([-m * gV(p) / (2 * m * (e - V(p))) for p, e in zip(X, E)])     # grad sigma, current-energy metric
    lhs = A + 2 * (gs * Vv).sum(1)[:, None] * Vv - sp2[:, None] * gs             # x'' + Gamma~(x', x')
    tang = Vv / np.sqrt(sp2)[:, None]
    along = (lhs * tang).sum(1); across = np.linalg.norm(lhs - along[:, None] * tang, axis=1)
    dln = np.gradient(np.log(sp2), DT)
    return dict(X=X, sp=np.sqrt(sp2), E=E, along=along, across=across, dln=dln, t=np.arange(len(X)) * DT,
                omega2=2 * m * (E - np.array([V(p) for p in X])))

R0, R1 = run(0.0), run(GAM)
fig, ax = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw={'wspace': 0.3})
# (a) the motion
g = np.linspace(-1.6, 1.6, 300); XX, YY = np.meshgrid(g, g)
VV = m * ups ** 2 * (1 - np.exp(-kap ** 2 * (XX ** 2 + YY ** 2)))
cs = ax[0].contourf(XX, YY, VV, levels=14, cmap='Greys', alpha=0.55)
ax[0].plot(R0['X'][:, 0], R0['X'][:, 1], color='#2563eb', lw=1.6, label='no friction')
ax[0].plot(R1['X'][:, 0], R1['X'][:, 1], color='#ea580c', lw=1.6, label=f'friction, gamma = {GAM}')
ax[0].plot(*R0['X'][0], 'ko', ms=6); ax[0].annotate('start', R0['X'][0], textcoords='offset points', xytext=(6, -12), fontsize=8.5)
for R, col in ((R0, '#2563eb'), (R1, '#ea580c')):
    for k in range(0, len(R['X']), 12000):
        p, q = R['X'][k], R['X'][min(k + 400, len(R['X']) - 1)] - R['X'][k]
        ax[0].annotate('', p + q, p, arrowprops=dict(arrowstyle='->', color=col, lw=1.4))
ax[0].set_aspect('equal'); ax[0].set_xlim(-1.6, 1.6); ax[0].set_ylim(-1.6, 1.6)
ax[0].set_title('(a) The example: a particle in the Gaussian well\n(grey: potential, darker is higher)', fontsize=10)
ax[0].legend(fontsize=8.5, loc='upper left')
# (b) the component along the motion
t, sl = R1['t'], slice(100, -100)
ax[1].plot(t[sl], R1['along'][sl], color='k', lw=3.2, alpha=0.35, label='measured along the motion')
ax[1].plot(t[sl], ((R1['dln'] + GAM) * R1['sp'])[sl], color='#16a34a', lw=1.3, ls='--', label='derived: (d/dt ln|v|^2 + gamma) |v|')
ax[1].plot(t[sl], (-GAM * R1['sp'])[sl], color='#dc2626', lw=1.3, label='book, damped: -gamma |v|')
ax[1].plot(t[sl], 0 * t[sl], color='#7c3aed', lw=1.3, label='book, undamped: 0')
ax[1].set_xlabel('time t'); ax[1].set_ylabel('component of  x\'\' + Gamma~(x\', x\')  along the motion')
ax[1].set_title('(b) Along the motion: the derived equation\nmatches; the book\'s two forms do not (gamma = 0.15)', fontsize=10)
ax[1].legend(fontsize=8)
# (c) across the motion, and the drifting metric
ax[2].semilogy(t[sl], np.maximum(R1['across'][sl], 1e-18), color='#0891b2', lw=1.2, label='across the motion (machine zero)')
ax[2].set_ylim(1e-18, 1e1); ax[2].set_xlabel('time t'); ax[2].set_ylabel('component of  x\'\' + Gamma~  across the motion', color='#0891b2')
a2 = ax[2].twinx()
a2.plot(t, R1['E'], color='#ea580c', lw=1.8, label='energy E(t)')
a2.plot(t, R1['omega2'] / (2 * m), color='#ea580c', lw=1.2, ls=':', label='conformal factor / 2m = E(t) - V')
a2.set_ylabel('energy', color='#ea580c'); a2.spines['right'].set_visible(True)
ax[2].set_title('(c) The path bends as a Jacobi geodesic at every\ninstant, while the metric drifts as E(t) falls', fontsize=10)
h1, l1 = ax[2].get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
ax[2].legend(h1 + h2, l1 + l2, fontsize=8, loc='center right')
fig.savefig(OUT / 'jacobi_example.png', dpi=150, bbox_inches='tight'); print('wrote jacobi_example.png')
