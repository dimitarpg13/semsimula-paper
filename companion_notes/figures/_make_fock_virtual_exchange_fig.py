#!/usr/bin/env python3
"""The book's Fock-mechanism schematic (paper_v6 Figure 'fock-virtual-particle-exchange').

  fock_v21/fock_virtual_particle_exchange.png   tokens, registers, and the three kinds of exchange between them

Redrawn on 2026-10-30 from the earlier hand-made image, with four corrections: the PARF pair forces are one-way,
from an earlier token to a later one (V_phi runs under a causal mask, so the earlier token feels no reaction), and
a later token feels several earlier ones, not only its neighbour; creation arrows go only from tokens to registers;
V_theta is shown as each token's own one-body term, not as part of PARF; the register lifetime is written out in
words, since tau is the book's activity threshold.

The book prints it at \\textwidth (about half its drawn size), so the fonts here are 14-22 pt to print at 7-11 pt.
No title or legend: the caption names the figure, and each kind of arrow is labelled where it is drawn. Only a few
arrows of each kind are drawn; the caption says that every token writes and reads. Copy the output to
paper_v6/figures/ for the book.

    python3 _make_fock_virtual_exchange_fig.py
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MPath

HERE = Path(__file__).resolve().parent
OUT = HERE / 'fock_v21'; OUT.mkdir(exist_ok=True)
BG = '#FFFFFF'; TOK = '#1E5BD8'; TOKE = '#0D3A9A'; PARF = '#222222'; WRITE = '#138A8A'; READ = '#E0413A'
REG = '#E07B12'; REGF = '#FDD9A0'; GREY = '#555555'; VTH = '#6A3D9A'
plt.rcParams.update({'font.size': 15, 'mathtext.fontset': 'cm'})


def arrow(ax, p0, p1, c, lw=2.4, rad=0.0, ls='-', ms=22, z=2):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=ms, lw=lw, color=c, linestyle=ls,
                                 connectionstyle=f'arc3,rad={rad}', shrinkA=0, shrinkB=0, zorder=z))


def bracket(ax, x0, x1, y, c):
    ax.plot([x0, x0, x1, x1], [y + 0.2, y, y, y + 0.2], color=c, lw=2.2)


W, H = 17.0, 10.3
fig, ax = plt.subplots(figsize=(12, 12 * (H + 1.0) / W), facecolor=BG)
ax.set_xlim(0, W); ax.set_ylim(-1.0, H); ax.axis('off'); ax.set_aspect('equal')

# ------------------------------------------------------------------ tokens and the conservative forces among them
ty, tr = 5.3, 0.4
xs = [0.9, 2.55, 4.2, 5.85, 7.5]
ax.text(4.2, 9.85, 'TOKEN PARTICLES (physical)', ha='center', fontsize=17, fontweight='bold', color=TOKE)
for n, x in enumerate(xs, 1):
    ax.add_patch(Circle((x, ty), tr, fc=TOK, ec=TOKE, lw=2, zorder=4))
    ax.text(x - 0.15, ty + 1.25, f'$h_{n}$', ha='center', fontsize=22)
    arrow(ax, (x - 0.45, ty + 0.72), (x + 0.2, ty + 0.72), TOKE, lw=2, ms=16)
    ax.text(x - 0.55, ty + 0.72, f'$v_{n}$', ha='right', va='center', fontsize=18, color=TOKE)
    ax.text(x, ty - 0.92, r'$V_{\theta}$', ha='center', fontsize=19, color=VTH)          # own wells, one-body
for a, b in zip(xs[:-1], xs[1:]):                                                      # PARF: earlier -> later
    arrow(ax, (a + tr + 0.07, ty), (b - tr - 0.07, ty), PARF, lw=2.6, z=3)
for x, rad, dx in ((xs[0], 0.2, -0.22), (xs[2], 0.32, 0.12)):                           # later token feels several
    arrow(ax, (x + 0.15, ty - 1.32), (xs[4] + dx, ty - 1.32), PARF, lw=2.0, rad=rad, ms=18, z=3)
ax.text(4.35, 2.92, r'PARF: $-\nabla V_{\phi}$, earlier $\rightarrow$ later token', ha='center', fontsize=16,
        color=PARF)

# ------------------------------------------------------------------ registers
rx, rr = 11.75, 0.48
ry = [7.25, 5.3, 3.35]
sal = [0.78, 0.38, 0.86]
ax.text(rx + 1.0, 9.85, 'FOCK REGISTERS (virtual)', ha='center', fontsize=17, fontweight='bold', color=REG)
bx, bw, bh = 12.65, 1.5, 0.3
ax.text(bx + bw / 2, 8.35, 'salience', ha='center', fontsize=15)
for k, (y, s) in enumerate(zip(ry, sal), 1):
    ax.add_patch(Circle((rx, y), rr, fc=REGF, ec=REG, lw=2.4, ls=(0, (4, 2.5)), zorder=4))
    ax.text(rx - 0.8, y + 0.45, f'$r_{k}$', ha='center', fontsize=22)
    ax.add_patch(FancyBboxPatch((bx, y - bh / 2), bw, bh, boxstyle='round,pad=0.01', fc='white', ec='#333333', lw=1.4))
    ax.add_patch(FancyBboxPatch((bx, y - bh / 2), bw * s, bh, boxstyle='round,pad=0.01', fc=REG, ec='none', alpha=0.85))
    ax.text(bx + bw / 2, y + 0.42, f'$\\sigma_{k}$', ha='center', fontsize=18)
arrow(ax, (14.4, ry[2] + 0.1), (14.4, ry[0] - 0.1), REG, lw=2, rad=0.42, ms=18)
ax.text(16.2, ry[1], 'lifetime\n' + r'$\approx \dfrac{1}{1-\lambda}$' + '\nlayers', ha='center', va='center', fontsize=15)
arrow(ax, (rx, ry[2] - rr - 0.05), (rx, 1.95), REG, lw=2, ls=(0, (3, 2)), ms=18)                  # destruction
ax.add_patch(Circle((rx, 1.5), 0.4, fc='#FEF1E2', ec='#F3C590', lw=2, ls=(0, (4, 2.5))))
ax.text(rx + 0.6, 1.72, 'destruction gate', fontsize=16, fontweight='bold', color=REG)
ax.text(rx + 0.6, 1.2, r'$\sigma_k \leftarrow \sigma_k\,(1 - g_k)$', fontsize=17, color=REG)

# ------------------------------------------------------------------ creation (write): tokens -> registers only
ax.text(8.3, 9.15, 'creation gate (write)', ha='center', fontsize=17, fontweight='bold', color=WRITE)
ax.text(8.3, 8.6, r'$q_k$ meets $k_j$;  $\alpha_{kj}$ = softmax over tokens', ha='center', fontsize=15, color=WRITE)
arrow(ax, (xs[2] + 0.3, ty + 0.28), (rx - rr - 0.05, ry[0] + 0.15), WRITE, rad=-0.36)
arrow(ax, (xs[4] + 0.3, ty + 0.28), (rx - rr - 0.05, ry[0] - 0.08), WRITE, rad=-0.3)
arrow(ax, (xs[4] + tr + 0.02, ty + 0.2), (rx - rr - 0.05, ry[1] + 0.15), WRITE, rad=-0.1)

# ------------------------------------------------------------------ reverse channel (read): registers -> tokens
arrow(ax, (rx - rr - 0.05, ry[1] - 0.15), (xs[4] + tr + 0.08, ty - 0.18), READ, rad=0.06)
# the second read runs under the PARF arcs and reaches h_1 from the left, crossing nothing
p0, p1 = (rx - rr * 0.9, ry[2] - rr * 0.4), (xs[0] - tr - 0.06, ty - 0.12)
ax.add_patch(FancyArrowPatch(path=MPath([p0, (8.5, -0.3), (-1.0, 0.0), p1], [MPath.MOVETO] + [MPath.CURVE4] * 3),
                             arrowstyle='-|>', mutation_scale=22, lw=2.4, color=READ, zorder=2))
ax.text(5.3, 2.25, r'reverse channel (read), $Q_i$', ha='center', fontsize=17, fontweight='bold', color=READ)
ax.text(5.3, 1.75, 'non-conservative, after the step', ha='center', fontsize=15, color=READ)

# ------------------------------------------------------------------ divider and brackets
ax.plot([9.75, 9.75], [0.5, 6.6], color=GREY, lw=1.3, ls=(0, (4, 4)))
bracket(ax, 0.4, 8.0, 0.2, TOKE)
ax.text(4.2, -0.5, r'conservative subsystem: $V_{\theta} + V_{\phi}$', ha='center', fontsize=17, fontweight='bold', color=TOKE)
bracket(ax, 10.9, 14.0, 0.2, REG)
ax.text(12.45, -0.5, 'extended state space', ha='center', fontsize=17, fontweight='bold', color=REG)

fig.savefig(OUT / 'fock_virtual_particle_exchange.png', dpi=200, facecolor=BG, bbox_inches='tight')
plt.close(fig); print('wrote fock_virtual_particle_exchange.png')
