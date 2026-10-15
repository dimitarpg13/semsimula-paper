#!/usr/bin/env python3
"""Figure for the Word document Jacobi_metric_and_Geodesics_in_Riemannian_Geometry.docx
(semantic_simulation/docs/Riemannian_geodesics): Jacobi's theorem made visible.

A unit-mass particle in the Gaussian well V(q) = V_inf (1 - exp(-k^2 |q|^2)) in the plane, g = identity.
(a) E > V_inf: a fan of trajectories of one energy from one point, over the conformal factor 2(E - V).
(b) E < V_inf: a bound trajectory, confined to the accessible region M_E = {V < E}; on its boundary
    the Jacobi metric degenerates (turning points).

    python3 _make_jacobi_docx_figs.py      -> figures/jacobi/jacobi_geodesics_fixed_energy.png
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / 'jacobi'; OUT.mkdir(exist_ok=True)
VINF, K = 1.0, 1.0
V = lambda q: VINF * (1 - np.exp(-K ** 2 * (q @ q)))
gV = lambda q: 2 * VINF * K ** 2 * q * np.exp(-K ** 2 * (q @ q))

def traj(q0, v0, T, dt=2e-3):
    q, v, out = np.array(q0, float), np.array(v0, float), []
    for _ in range(int(T / dt)):
        out.append(q.copy())
        a = lambda qq: -gV(qq)
        k1q, k1v = v, a(q); k2q, k2v = v + dt / 2 * k1v, a(q + dt / 2 * k1q)
        k3q, k3v = v + dt / 2 * k2v, a(q + dt / 2 * k2q); k4q, k4v = v + dt * k3v, a(q + dt * k3q)
        q = q + dt / 6 * (k1q + 2 * k2q + 2 * k3q + k4q); v = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return np.array(out)

g = np.linspace(-2.4, 2.4, 400); X, Y = np.meshgrid(g, g)
VV = VINF * (1 - np.exp(-K ** 2 * (X ** 2 + Y ** 2)))
fig, ax = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={'wspace': 0.42})

E = 1.3
im = ax[0].contourf(X, Y, 2 * (E - VV), levels=20, cmap='YlGnBu')
plt.colorbar(im, ax=ax[0], fraction=0.046, pad=0.03, label='conformal factor 2(E - V)')
q0 = np.array([-2.0, 0.0]); sp = np.sqrt(2 * (E - V(q0)))
for ang in np.linspace(-0.55, 0.55, 11):
    tr = traj(q0, sp * np.array([np.cos(ang), np.sin(ang)]), 4.2)
    ax[0].plot(tr[:, 0], tr[:, 1], color='#b91c1c', lw=1.1)
ax[0].plot(*q0, 'ko', ms=5)
ax[0].set_title('(a) E > V∞: trajectories of one energy from one point\nare the geodesics of the Jacobi metric J = 2(E - V) g', fontsize=10)

E2 = 0.6; rstar = np.sqrt(-np.log(1 - E2 / VINF)) / K
inside = np.where(VV < E2, 2 * (E2 - VV), np.nan)
im2 = ax[1].contourf(X, Y, inside, levels=20, cmap='YlGnBu')
plt.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.03, label='conformal factor 2(E - V)')
th = np.linspace(0, 2 * np.pi, 400)
ax[1].plot(rstar * np.cos(th), rstar * np.sin(th), 'k--', lw=1.4)
ax[1].text(rstar * 0.72, rstar * 0.86, 'boundary of M_E:\nV = E, J degenerates', fontsize=8.5)
q0b = np.array([0.3, 0.0]); spb = np.sqrt(2 * (E2 - V(q0b)))
trb = traj(q0b, spb * np.array([0.35, 0.94]), 40.0)
ax[1].plot(trb[:, 0], trb[:, 1], color='#b91c1c', lw=0.7)
ax[1].text(-2.3, -2.25, 'outside M_E: classically forbidden (V > E)', fontsize=8.5, color='0.35')
ax[1].set_title('(b) E < V∞: the motion is confined to the accessible region\nM_E = {V < E}, where J is a Riemannian metric', fontsize=10)
for a in ax:
    a.set_aspect('equal'); a.set_xlim(-2.4, 2.4); a.set_ylim(-2.4, 2.4); a.set_xlabel('q¹'); a.set_ylabel('q²')
fig.savefig(OUT / 'jacobi_geodesics_fixed_energy.png', dpi=170, bbox_inches='tight'); print('wrote jacobi_geodesics_fixed_energy.png')
