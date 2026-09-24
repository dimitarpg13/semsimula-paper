"""Figures for Fock_Mechanism_Efficiency_Across_Layer_Depth.md.

Three PNGs into figures/fock_depth/. Every number plotted is either an
exact evaluation of the code's own recurrence (the salience/blend
schedule) or a measurement recorded in the document -- nothing is fit.

  fock_depth_blend.png     -- the admittance (1 - blend) per layer for
                              L in {1,2,4,8}. This is the fraction of the
                              creation readout that survives into the
                              register bank at each layer, and it is
                              EXACTLY ZERO at layer 0 for every depth.
  fock_depth_gate.png      -- the reverse-channel gate,
                              tanh(scale) * min(1, step/warmup), which is
                              the only route from registers to tokens.
                              Zero on a freshly built model; the shaded
                              region is where every naive probe reports
                              "registers do nothing" for reasons that have
                              nothing to do with the architecture.
  fock_depth_gradients.png -- measured next-token gradient reaching
                              creation_gate_qkv, with the reverse gate at
                              its trained value.
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / 'fock_depth'
OUT.mkdir(parents=True, exist_ok=True)

DECAY = 0.5          # cfg.register_salience_decay, live value
ALPHA = 0.30         # representative alpha_max; the qualitative shape is
                     # independent of it, only the asymptote moves
INK, ACC, WARN, MUTE = '#1b1b1b', '#2f6f9f', '#b4342a', '#9a9a9a'


def salience_schedule(L, s0=1.0, decay=DECAY, alpha=ALPHA):
    """salience entering the blend at each layer, from the code's recurrence
       s_{l+1} = s_l * decay + alpha_max * (1 - decay)."""
    s, out = s0, []
    for _ in range(L):
        out.append(s)
        s = s * decay + alpha * (1.0 - decay)
    return np.array(out)


# ---------------------------------------------------------------- fig 1
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
# one schedule: every depth walks the same curve, just not as far. Drawing
# four overlapping lines hides that, so draw the curve once and mark reach.
adm8 = 1.0 - salience_schedule(8)
ax.plot(range(8), adm8, '-', color=MUTE, lw=1.4, zorder=1)
ax.scatter(range(8), adm8, s=34, color=MUTE, zorder=2)
for L, c, dy in ((1, WARN, 0.06), (2, ACC, 0.06), (4, INK, 0.06)):
    ax.scatter([L - 1], [adm8[L - 1]], s=95, color=c, zorder=3)
    ax.annotate(f'L={L} stops here', xy=(L - 1, adm8[L - 1]),
                xytext=(L - 1 + 0.25, adm8[L - 1] - dy - 0.09), color=c,
                fontsize=9, arrowprops=dict(arrowstyle='->', color=c, lw=1.1))
ax.axhline(0, color=INK, lw=0.8)
ax.set_xlabel('layer index')
ax.set_ylabel(r'admittance  $1-\mathrm{blend}$')
ax.set_title('Fraction of the creation readout that enters the bank')
ax.annotate('layer 0 admits exactly 0\nat every depth', xy=(0, 0),
            xytext=(1.6, 0.10), color=WARN, fontsize=9,
            arrowprops=dict(arrowstyle='->', color=WARN, lw=1.1))
ax.set_ylim(-0.08, 0.80)
ax.grid(alpha=0.25)

Ls = [1, 2, 4, 8]
tot = [float((1.0 - salience_schedule(L)).sum()) for L in Ls]
ax2.bar([str(L) for L in Ls], tot,
        color=[WARN, ACC, INK, MUTE], width=0.6)
for i, v in enumerate(tot):
    ax2.text(i, v + 0.03, f'{v:.2f}', ha='center', fontsize=9, color=INK)
ax2.set_xlabel('L'); ax2.set_ylabel('total admittance over the stack')
ax2.set_title('Capacity to write content, summed over layers')
ax2.grid(alpha=0.25, axis='y')
fig.suptitle('The register bank can only be written where salience has decayed',
             fontsize=11)
fig.tight_layout()
fig.savefig(OUT / 'fock_depth_blend.png', dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- fig 2
fig, ax = plt.subplots(figsize=(7.4, 4.0))
warm_steps = 4000
step = np.arange(0, 12000)
for scale, c, lab in ((0.017, ACC, r'trained, $\tanh(s)=0.017$'),
                      (0.000, WARN, r'fresh init, $\tanh(s)=0$')):
    gate = scale * np.minimum(1.0, step / warm_steps)
    ax.plot(step, gate, color=c, lw=2.0, label=lab)
ax.axvspan(0, warm_steps, color=WARN, alpha=0.07)
ax.text(warm_steps * 0.5, 0.0125, 'warmup\ngate < trained value',
        ha='center', color=WARN, fontsize=9)
ax.axvline(warm_steps, color=MUTE, ls='--', lw=1.0)
ax.text(warm_steps + 200, 0.0005, 'REVERSE_CHANNEL_WARMUP_STEPS = 4000',
        fontsize=8, color=MUTE)
ax.set_xlabel('forward passes'); ax.set_ylabel('effective gate')
ax.set_title(r'The only register $\to$ token path:  '
             r'$(\Delta t^2/m)\cdot\tanh(s)\cdot\mathrm{warm}\cdot Q_{\rm force}$')
ax.legend(frameon=False, loc='center right')
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / 'fock_depth_gate.png', dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- fig 3
fig, ax = plt.subplots(figsize=(7.8, 4.0))
labs = ['L=1\n$s_0$=1.0', 'L=1\n$s_0$=0.9', 'L=1\n$s_0$=0.5', 'L=2\n$s_0$=1.0']
vals = [0.0, 5.503e-4, 2.900e-3, 2.815e-3]
cols = [WARN, ACC, ACC, INK]
bars = ax.bar(labs, vals, color=cols, width=0.62)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2,
            v + 1.2e-4, ('0  (severed)' if v == 0 else f'{v:.2e}'),
            ha='center', fontsize=9,
            color=(WARN if v == 0 else INK))
ax.axhline(2.815e-3, color=INK, ls=':', lw=1.1)
ax.text(-0.42, 2.815e-3 + 6e-5, 'L=2 reference level', fontsize=8,
        color=INK, ha='left', va='bottom')
ax.set_ylim(0, 3.7e-3)
ax.set_ylabel(r'max $|\nabla|$ reaching creation_gate_qkv')
ax.set_title('Next-token gradient to the creation gate, reverse gate open')
ax.grid(alpha=0.25, axis='y')
fig.tight_layout()
fig.savefig(OUT / 'fock_depth_gradients.png', dpi=150)
plt.close(fig)

print('wrote:')
for f in sorted(OUT.glob('*.png')):
    print(f'   {f.relative_to(Path(__file__).resolve().parent.parent)}  '
          f'{f.stat().st_size//1024} KB')
