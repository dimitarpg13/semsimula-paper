#!/usr/bin/env python3
"""Figures for Fock_v2.1_Mechanism_Principle_of_Operation.md.

  fock_v21/fock_two_routes.png    the two exchange routes of a Gen 3 layer (schematic)
  fock_v21/fock_layer_step.png    one layer step, in the order the code executes it (schematic)
  fock_v21/fock_g3p_lifecycle.png measured on the trained G3' (8 validation sequences): the register
                                  lifecycle, the read port's focus and the size of each exchange route

The measured numbers are cached in fock_v21/_numbers.json; delete it to recompute (needs the local
G3' checkpoint and the sync/ replay helpers).

    python3 _make_fock_v21_figs.py
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse

HERE = Path(__file__).resolve().parent
OUT = HERE / 'fock_v21'; OUT.mkdir(exist_ok=True)
SYNC = HERE.parents[1] / 'notebooks/conservative_arch/scaleup/sync'
BG = '#FAFAFA'; WRITE = '#1565C0'; READ = '#C62828'; REG = '#2E7D32'; EXCH = '#E65100'; GREY = '#555555'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})


def box(ax, x, y, w, h, text, ec, fs=10.5, fc='white', bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04,rounding_size=0.1', fc=fc, ec=ec, lw=1.8))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, fontweight='bold' if bold else 'normal')


def arrow(ax, x0, y0, x1, y1, c, lw=2.0, ls='-'):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=18, lw=lw, color=c, linestyle=ls))


# ---------------------------------------------------------------- figure 1: two exchange routes
fig, ax = plt.subplots(figsize=(13, 6.2), facecolor=BG); ax.set_xlim(0, 13); ax.set_ylim(0, 6.2); ax.axis('off')
ax.text(6.5, 5.85, 'Two ways one token reaches another in a Gen 3 layer', ha='center', fontsize=14, fontweight='bold')
box(ax, 0.3, 2.6, 1.9, 1.0, 'token  j\n(earlier)', '#333333', fs=11.5, bold=True)
box(ax, 10.8, 2.6, 1.9, 1.0, 'token  i', '#333333', fs=11.5, bold=True)
# direct route (exchange field), top
arrow(ax, 2.25, 3.45, 10.75, 3.45, EXCH, lw=2.4)
ax.text(6.5, 4.95, 'DIRECT: the exchange field (one hop, inside the integrator)', ha='center', fontsize=11.5, fontweight='bold', color=EXCH)
ax.text(6.5, 4.45, r'$f^{\mathrm{ex}}_i = \lambda_{\mathrm{ex}} \sum_{\mathrm{heads}} \sum_{j < i} a_{ij}\, U^{\top} W h_j$,    '
        r'$a_{ij} = \mathrm{softmax}_{j < i}(\tilde q_i \cdot \tilde k_j)$', ha='center', fontsize=11)
ax.text(6.5, 3.75, 'a force: it accelerates token i during the step; recomputed every layer; no memory', ha='center', fontsize=9.5, color=GREY)
# mediated route (registers), bottom
ax.add_patch(Ellipse((6.5, 1.55), 2.6, 1.15, fc='#F1F8E9', ec=REG, lw=1.8, ls='--'))
ax.text(6.5, 1.72, r'register $r_k$', ha='center', fontsize=11.5, fontweight='bold')
ax.text(6.5, 1.3, 'persists across layers', ha='center', fontsize=9, color=GREY)
arrow(ax, 2.25, 2.75, 5.25, 1.75, WRITE, lw=2.2)
arrow(ax, 7.75, 1.75, 10.75, 2.75, READ, lw=2.2)
ax.text(3.2, 1.55, r'write  $\alpha_{kj}$' + '\n(softmax over tokens)', ha='center', fontsize=10, color=WRITE)
ax.text(9.8, 1.55, r'read  $\beta_{ik}$' + '\n(softmax over registers)', ha='center', fontsize=10, color=READ)
ax.text(6.5, 0.3, r'MEDIATED: the Fock registers (two hops, read after the step):   $A_{ij} = \sum_k \beta_{ik}\,\alpha_{kj}$,  rank $\leq M$',
        ha='center', fontsize=11, fontweight='bold', color=REG)
fig.savefig(OUT / 'fock_two_routes.png', dpi=150, facecolor=BG, bbox_inches='tight'); plt.close(fig); print('wrote fock_two_routes.png')

# ---------------------------------------------------------------- figure 2: one layer step
fig, ax = plt.subplots(figsize=(13, 8.6), facecolor=BG); ax.set_xlim(0, 13); ax.set_ylim(0, 8.6); ax.axis('off')
ax.text(6.5, 8.3, 'One Gen 3 layer step, in the order the code executes it (prefix-causal registers)', ha='center', fontsize=13.5, fontweight='bold')
steps = [
    ('1  WRITE', r'readout$_k^{(t)} = \sum_{j \leq t} \alpha^{(t)}_{kj} v_j$' + '\n' + r'$r_k \leftarrow \sigma_k r_k + (1-\sigma_k)\,$readout$_k$', WRITE),
    ('2  SALIENCE', r'$\sigma_k \leftarrow \lambda\sigma_k + (1-\lambda)\max_{j \leq t}\alpha^{(t)}_{kj}$' + '\n' + r'active iff $\sigma_k > \tau$', REG),
    ('3  INTEGRATE', r'BAOAB step: $V_\theta$, $V_\phi$' + '\n' + r'and the exchange field $f^{\mathrm{ex}}$', EXCH),
    ('4  NORMALISE', r'$h \leftarrow \mathrm{LN}(h)$', '#444444'),
    ('5  READ', r'$h_i \leftarrow h_i + \frac{dt^2}{m_i}\tanh(s_\ell)\,\widetilde{Q}_i$' + '\n' + r'$Q_i = \sum_k \beta_{ik} W^{\mathrm{rev}}_V r_k$', READ),
    ('6  NORMALISE', r'$h \leftarrow \mathrm{LN}(h)$', '#444444'),
    ('7  DESTROY', r'$\sigma_k \leftarrow \sigma_k\,(1 - g_k)$' + '\n' + r'$g_k = \mathrm{sigmoid}(\mathrm{MLP}(r_k))$', REG),
]
w, h, gap = 2.85, 1.75, 0.35
pos = [(0.3 + n * (w + gap), 5.85) for n in range(4)] + [(0.3 + n * (w + gap), 3.75) for n in (3, 2, 1)]
for n, ((title, body, c), (x, y)) in enumerate(zip(steps, pos)):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04,rounding_size=0.1', fc='white', ec=c, lw=2.0))
    ax.text(x + w / 2, y + h - 0.3, title, ha='center', fontsize=11, fontweight='bold', color=c)
    ax.text(x + w / 2, y + h / 2 - 0.2, body, ha='center', va='center', fontsize=10)
for n in range(3):                                   # top row, left to right
    (x, y) = pos[n]; arrow(ax, x + w + 0.02, y + h / 2, x + w + gap - 0.02, y + h / 2, '#333333', lw=1.6)
(x3, y3) = pos[3]; arrow(ax, x3 + w / 2, y3 - 0.02, x3 + w / 2, y3 - 0.35 + 0.02, '#333333', lw=1.6)   # 4 -> 5, straight down
for n in (4, 5):                                     # bottom row, right to left
    (x, y) = pos[n]; arrow(ax, x - 0.02, y + h / 2, x - gap + 0.02, y + h / 2, '#333333', lw=1.6)
ax.text(0.3 + w / 2, 3.75 + h / 2, 'next layer:\nback to step 1', ha='center', va='center', fontsize=10, color=GREY, style='italic')
notes = [
    (WRITE, 'Steps 1, 2, 7 act on the registers; steps 3 and 5 act on the tokens.'),
    (EXCH, 'Step 3 is the direct route: the exchange field is a force inside the integrator, beside the conservative potentials.'),
    (READ, 'Step 5 is the mediated route\'s only read port: without it the registers cannot affect a prediction.'),
    (REG, 'Layer 0: registers start at their learned embeddings with salience 1, so step 1 writes nothing; step 5 reads the embeddings.'),
    (REG, 'From layer 1 on, the blend in step 1 writes the token content, weighted by how much salience step 7 removed.'),
    (GREY, 'G3′: M = 32, λ = 0.5 (lifetime 1/(1-λ) = 2 layers), τ = 0.005, per-layer read scales s, exchange gate λ_ex = 1.'),
]
for n, (c, s) in enumerate(notes):
    ax.text(0.4, 2.95 - n * 0.47, '■', color=c, fontsize=11, va='center')
    ax.text(0.75, 2.95 - n * 0.47, s, fontsize=10, va='center')
fig.savefig(OUT / 'fock_layer_step.png', dpi=150, facecolor=BG, bbox_inches='tight'); plt.close(fig); print('wrote fock_layer_step.png')


# ---------------------------------------------------------------- measured on G3'
cache = OUT / '_numbers.json'
if cache.exists():
    NUM = json.loads(cache.read_text())
else:
    import torch
    sys.path.insert(0, str(SYNC))
    from sy_replay import Replayer, CACHE
    R = Replayer('G3p'); m = R.model
    X = np.load(CACHE / 'sy_tokens.npz')['x'][:8]
    rec = {k: {0: [], 1: []} for k in ('g', 'sal_pre', 'active', 'rev_ent', 'rev_peak', 'cre_ent', 'Qrms', 'ex', 'cons', 'mass')}
    cur = {'l': 0}
    step0 = m._fock_layer_step
    def step(h, h_prev, r, sal, m_b, gamma, dt, layer_idx=0):
        cur['l'] = layer_idx; rec['mass'][layer_idx].append(m_b.detach().reshape(-1).numpy())
        return step0(h, h_prev, r, sal, m_b, gamma, dt, layer_idx=layer_idx)
    m._fock_layer_step = step
    for l, dg in enumerate(m.destruction_gates):
        f0 = dg.forward
        def fw(r, f0=f0):
            out = f0(r); rec['g'][cur['l']].append(out.detach().reshape(-1).numpy()); return out
        dg.forward = fw
    am0 = m._active_mask
    def am(s):
        out = am0(s); rec['sal_pre'][cur['l']].append(s.detach().reshape(-1).numpy())
        rec['active'][cur['l']].append(out.detach().float().reshape(-1).numpy()); return out
    m._active_mask = am
    rc = m.reverse_ch; rc.capture_stats = True; rf0 = rc.forward
    def rfw(h, r, a):
        Q = rf0(h, r, a); rec['rev_ent'][cur['l']].append(rc.last_entropy); rec['rev_peak'][cur['l']].append(rc.last_alpha_max)
        rec['Qrms'][cur['l']].append(Q.detach().pow(2).mean(-1).sqrt().reshape(-1).numpy()); return Q
    rc.forward = rfw
    cg = m.creation_gate_qkv; cg.capture_stats = True; cf0 = cg.forward_prefix
    def cfw(h, r):
        o = cf0(h, r); rec['cre_ent'][cur['l']].append(cg.last_entropy); return o
    cg.forward_prefix = cfw
    fl0 = m.relax_field.force_live
    def flw(*a, **kw):
        f = fl0(*a, **kw); rec['ex'][cur['l']].append(f.detach().norm(dim=-1).reshape(-1).numpy()); return f
    m.relax_field.force_live = flw
    lf0 = m._layer_forces
    def lfw(*a, **kw):
        out = lf0(*a, **kw)
        if kw.get('split'):
            ft, fp = out; tot = fp if ft is None else ft + fp
            rec['cons'][cur['l']].append(tot.detach().norm(dim=-1).reshape(-1).numpy())
        return out
    m._layer_forces = lfw
    for b in range(0, len(X), 2):
        R.forward(torch.from_numpy(X[b:b + 2]).long())
    cat = lambda v: np.concatenate(v) if len(v) else np.array([])
    NUM = {}
    s_l = torch.tanh(m.reverse_channel_scale.detach()).numpy()
    for l in (0, 1):
        g = cat(rec['g'][l]); mass = cat(rec['mass'][l]); act = cat(rec['active'][l])
        ex = cat(rec['ex'][l]); cons = cat(rec['cons'][l])
        NUM[f'L{l}'] = {
            'g_hist': np.histogram(g, bins=20, range=(0, 1))[0].tolist(), 'g_median': float(np.median(g)),
            'g_p10': float(np.percentile(g, 10)), 'g_p90': float(np.percentile(g, 90)),
            'active_frac': float(act.mean()), 'sal_pre_median': float(np.median(cat(rec['sal_pre'][l]))),
            'rev_entropy': float(np.mean(rec['rev_ent'][l])), 'rev_peak': float(np.mean(rec['rev_peak'][l])),
            'cre_entropy': float(np.mean(rec['cre_ent'][l])) if rec['cre_ent'][l] else None,
            'Q_rms_median': float(np.median(cat(rec['Qrms'][l]))),
            'tanh_s': float(s_l[l]), 'mass_median': float(np.median(mass)),
            'read_coeff': float(16.0 / np.median(mass) * s_l[l]),
            'ex_force_median': float(np.median(ex)), 'total_force_median': float(np.median(cons)),
        }
    NUM['M'] = int(m.cfg.n_registers); NUM['ln_M'] = float(np.log(m.cfg.n_registers))
    cache.write_text(json.dumps(NUM, indent=1))

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={'wspace': 0.32})
edges = np.linspace(0, 1, 21); ctr = 0.5 * (edges[1:] + edges[:-1])
for l, c in ((0, '#7c3aed'), (1, '#0891b2')):
    hgt = np.array(NUM[f'L{l}']['g_hist'], float); hgt /= hgt.sum()
    ax[0].bar(ctr + (l - 0.5) * 0.02, hgt, 0.02, color=c, label=f'layer {l} (median {NUM[f"L{l}"]["g_median"]:.2f})')
ax[0].set_xlabel('destruction gate g (fraction of salience removed)'); ax[0].set_ylabel('share of register-positions')
ax[0].set_title('(a) Destruction gate: bimodal at layer 0; at layer 1 it stays\nat its initial 0.5 (nothing reads the registers after it)', fontsize=10)
ax[0].legend(fontsize=8.5)
lay = np.arange(2); w = 0.35
ent = [NUM[f'L{l}']['rev_entropy'] for l in (0, 1)]; peak = [NUM[f'L{l}']['rev_peak'] for l in (0, 1)]
ax[1].bar(lay - w / 2, np.exp(ent), w, color=READ, label='effective registers read, exp(entropy)')
ax[1].axhline(NUM['M'], color='0.5', ls=':', lw=1); ax[1].text(1.45, NUM['M'] - 2.2, f'M = {NUM["M"]}', ha='right', fontsize=8.5, color='0.4')
a2 = ax[1].twinx(); a2.bar(lay + w / 2, peak, w, color='#f59e0b', label='peak read weight')
a2.set_ylim(0, 1.37); a2.spines['right'].set_visible(True); a2.set_ylabel('peak read weight', color='#b45309')
ax[1].set_xticks(lay); ax[1].set_xticklabels(['layer 0', 'layer 1']); ax[1].set_ylim(0, 44); ax[1].set_ylabel('effective number of registers read', color=READ)
ax[1].set_title('(b) How focused the read port is', fontsize=10)
h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels(); ax[1].legend(h1 + h2, l1 + l2, fontsize=8, loc='upper left')
exr = [NUM[f'L{l}']['ex_force_median'] / NUM[f'L{l}']['total_force_median'] for l in (0, 1)]
rd = [NUM[f'L{l}']['read_coeff'] * NUM[f'L{l}']['Q_rms_median'] for l in (0, 1)]
ax[2].bar(lay - w / 2, exr, w, color=EXCH, label='exchange force / total integrator force')
ax[2].bar(lay + w / 2, rd, w, color=READ, label='register read increment, RMS per coordinate\n(the normalised state has RMS 1)')
ax[2].set_xticks(lay); ax[2].set_xticklabels(['layer 0', 'layer 1']); ax[2].set_ylabel('relative size (medians)')
ax[2].set_title('(c) The size of each exchange route', fontsize=10); ax[2].set_ylim(0, 1.55); ax[2].legend(fontsize=8, loc='upper left')
fig.suptitle('Measured on the trained G3′ (8 validation sequences of 512 tokens)', fontsize=11.5, y=1.02)
fig.savefig(OUT / 'fock_g3p_lifecycle.png', dpi=150, facecolor=BG, bbox_inches='tight'); plt.close(fig); print('wrote fock_g3p_lifecycle.png')
print(json.dumps(NUM, indent=1))
