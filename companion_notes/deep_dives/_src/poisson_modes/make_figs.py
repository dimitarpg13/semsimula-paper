#!/usr/bin/env python3
"""Figures for the deep dive on the Poisson-mode registers, wells and semantic templates (2026-10-23).

    python3 make_figs.py            # writes figures/*.png next to this file

  pm_principle.png       schematic: one layer step with the Poisson-mode wells (computed layout, no data)
  pm_occupations.png     recorded: PMX-K16's overlaps and occupations of three modes along two different sequences
  pm_geometry.png        recorded: where the centres sit against the token clouds; selectivity against matching rank
  pm_halflife.png        recorded: the occupations' position share against the initial half-life range
  pm_static.png          recorded: the cost of fixing the occupations, and of removing the wells, by position
  pm_shorttrain.png      recorded: the short CPU training run, isotropic against rank-8 matching
The recorded numbers are read from the outputs in notebooks/conservative_arch/scaleup/{debug,results}.
"""
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
OUT = HERE / 'figures'; OUT.mkdir(exist_ok=True)
REPO = HERE.parents[3]
DBG = REPO / 'notebooks/conservative_arch/scaleup/debug'
RES = REPO / 'notebooks/conservative_arch/scaleup/results'
PF = 'cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_'
SF = '_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 200,
                     'font.family': 'sans-serif'})
C = dict(blue='#156082', orange='#E97132', green='#196B24', teal='#0F9ED5', purple='#A02B93', grey='#7f7f7f')

# ---------------------------------------------------------------- 1. principle (schematic)
fig, ax = plt.subplots(figsize=(10, 4.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis('off')
def box(x, y, w, h, text, fc, ec='0.3', fs=8.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2', fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs)
def arrow(x0, y0, x1, y1, text='', dy=1.2):
    ax.annotate('', (x1, y1), (x0, y0), arrowprops=dict(arrowstyle='->', lw=1.1, color='0.25'))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, text, ha='center', fontsize=7.2, color='0.25')
box(1, 31, 17, 9, 'earlier tokens\n$h_s^{(\\ell)}$,  $s < t$', '#E8E8E8')
box(24, 31, 20, 9, 'overlaps\n$E_v(s)=e^{-\\kappa_v^2\\|h_s-\\mu_v\\|^2}$', '#DCEBF5')
box(50, 31, 22, 9, 'occupations (strict past)\n$\\bar n_v(t)=\\sum_{s<t}\\lambda_v^{t-1-s}E_v(s)$', '#DCEBF5')
box(78, 31, 21, 9, 'Gaussian wells\n$V_{pm}=-\\sum_v \\bar n_v\\,\\varepsilon_{\\ell,v}e^{-\\kappa_v^2\\|h-\\mu_v\\|^2}$', '#FCE4D6', fs=7.6)
arrow(18.6, 35.5, 23.6, 35.5); arrow(44.6, 35.5, 49.6, 35.5, 'decay $\\lambda_v$'); arrow(72.6, 35.5, 77.6, 35.5, 'depth $\\varepsilon_{\\ell,v}$')
box(1, 13, 17, 9, 'token $t$ at layer $\\ell$\n$h_t^{(\\ell)}$', '#E8E8E8')
box(24, 13, 26, 9, 'force $f_{pm}=-\\nabla_h V_{pm}$\n(exact gradient: $\\bar n_v(t)$\ndoes not depend on $h_t$)', '#FCE4D6', fs=7.8)
box(56, 17.5, 20, 7.5, 'PM1: explicit kick\n(beside $V_\\phi$)', '#E2F0D9', fs=7.8)
box(56, 8, 20, 7.5, 'PMX: quadratic in the\nexact damped flow', '#E2F0D9', fs=7.8)
box(82, 13, 17, 9, 'next state\n$h_t^{(\\ell+1)}$', '#E8E8E8')
arrow(88, 30.4, 40, 22.6); arrow(18.6, 17.5, 23.6, 17.5); arrow(50.6, 18.5, 55.6, 21); arrow(50.6, 16.5, 55.6, 12)
arrow(76.6, 21, 81.6, 18.5); arrow(76.6, 12, 81.6, 16.5)
ax.text(1, 4.6, 'The occupation is the exact Poisson mean of an immigration-death process: token $s$ creates Poisson($E_v(s)$) '
        'particles in mode $v$,\neach survives a token with probability $\\lambda_v$. Linear in $\\bar n$, so carrying the mean is exact. '
        'Schematic; no data.', fontsize=7.4, color='0.3')
fig.savefig(OUT / 'pm_principle.png', bbox_inches='tight'); plt.close(fig)

# ---------------------------------------------------------------- 2. occupations along two sequences (recorded, PMX-K16)
try:
    import torch
    sys.argv = [sys.argv[0], str(HERE / '_harness')]
    sys.path.insert(0, str(DBG))
    import verify_pm_switch as P
    torch.set_num_threads(16)
    repl = P.CONFIGS['F3.1'][1] + (("POISSON_MODES        = 0", "POISSON_MODES        = 16"),
                                   ("LOWRANK_DAMPED_FLOW  = False", "LOWRANK_DAMPED_FLOW  = True"),
                                   ("POISSON_WELLS_EXACT  = False", "POISSON_WELLS_EXACT  = True"))
    model, g, _ = P.build('semsimula_' + PF.replace('cfc_baoab_owt', 'fock_cfc_baoab_owt') + 'pm16_pmx16_sr2' + SF, repl)
    model.to('cpu'); model.eval(); model.cfg.use_layer_checkpoint = False
    rng = np.random.default_rng(20261009)
    xb, _ = g['get_batch'](g['val_ids'], 2, 512, rng)
    caps = {}
    orig = model._fock_layer_step
    def rec(h, h_prev, r, sal, m_b, gamma, dt, layer_idx=0):
        caps.setdefault(int(layer_idx), h.detach().clone()); return orig(h, h_prev, r, sal, m_b, gamma, dt, layer_idx=layer_idx)
    model._fock_layer_step = rec
    with torch.enable_grad():
        model(torch.from_numpy(xb))
    with torch.no_grad():
        E, phi = model.poisson_mode_occupation(caps[1])
    hl = (np.log(0.5) / torch.nn.functional.logsigmoid(model.pm_logit_lambda.detach()).numpy())
    pick = [int(np.argmin(np.abs(hl - q))) for q in (np.quantile(hl, 0.1), np.median(hl), np.quantile(hl, 0.9))]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
    for j, v in enumerate(pick):
        col = (C['teal'], C['blue'], C['purple'])[j]
        for b, ls in ((0, '-'), (1, '--')):
            ax[0].plot(E[b, :, v].numpy(), ls, color=col, lw=0.8, alpha=0.9)
            ax[1].plot(phi[b, :, v].numpy(), ls, color=col, lw=1.2, label=f'mode {v}: half-life {hl[v]:.1f} tok' if b == 0 else None)
    ax[0].set_title('A. Overlaps $E_v(s)$ at layer 1 (two sequences: solid, dashed)', fontsize=9)
    ax[0].set_xlabel('position $s$'); ax[0].set_ylabel('$E_v(s)$')
    ax[1].set_title('B. Occupations $\\bar n_v(t)$: the two sequences coincide', fontsize=9)
    ax[1].set_xlabel('position $t$'); ax[1].set_ylabel('$\\bar n_v(t)$'); ax[1].legend(fontsize=7, frameon=False)
    for a in ax:
        a.set_xlim(0, 200)
    fig.tight_layout(); fig.savefig(OUT / 'pm_occupations.png'); plt.close(fig)
    print('pm_occupations: modes', pick, 'half-lives', [round(float(hl[v]), 1) for v in pick])
except Exception as e:                                                   # the other figures do not need the model
    print('pm_occupations skipped:', repr(e)[:200])

# ---------------------------------------------------------------- 3. geometry and rank (recorded, narrow_wells_design_output.txt)
t = (DBG / 'narrow_wells_design_output.txt').read_text()
geo = re.findall(r'layer (\d): \|h\| p10/p50/p90\s+[\d.]+\s+([\d.]+)\s+[\d.]+\s+token-token distance p50\s+([\d.]+)\s+'
                 r'mode-token distance: nearest token p50\s+([\d.]+)', t)
rk = {}
for l, r, sp, row in re.findall(r'layer (\d), r = +(\d+): spread ([\d.]+)  (.*)', t):
    c2 = re.search(r'c=2\s+sel\s+([\d.]+) reach\s+(\d+)% cover\s+(\d+)%', row)
    rk.setdefault(int(l), []).append((int(r), float(sp), float(c2.group(1)), int(c2.group(2)), int(c2.group(3))))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
labels = ['L0 init', 'L1 init', 'L0 trained', 'L1 trained']
x = np.arange(4); w = 0.38
ax[0].bar(x - w / 2, [float(gg[2]) for gg in geo], w, color=C['teal'], label='token-token distance (median)')
ax[0].bar(x + w / 2, [float(gg[3]) for gg in geo], w, color=C['orange'], label='centre to its nearest token (median)')
ax[0].set_xticks(x, labels); ax[0].set_ylabel('distance (units of $h$)'); ax[0].legend(fontsize=7, frameon=False, loc='upper left'); ax[0].set_ylim(0, 36)
ax[0].set_title('A. Token clouds and centres: at layer 0 the centres are far outside', fontsize=9)
for l, col in ((0, C['blue']), (1, C['purple'])):
    rows = sorted(rk[l]); rr = [r_[0] for r_ in rows]
    ax[1].plot(rr, [r_[1] for r_ in rows], 'o-', color=col, label=f'layer {l}: spread of distances')
    ax[1].plot(rr, [r_[4] / 100 for r_ in rows], 's--', color=col, alpha=0.6, label=f'layer {l}: coverage at c = 2')
ax[1].set_xscale('log'); ax[1].set_xticks([4, 8, 16, 64, 384], ['4', '8', '16', '64', '384']); ax[1].set_xlabel('matching rank $r$')
ax[1].set_title('B. Low-rank matching grades; full-rank does not', fontsize=9); ax[1].legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(OUT / 'pm_geometry.png'); plt.close(fig)

# ---------------------------------------------------------------- 4. half-life range (recorded)
t = (DBG / 'pm_halflife_range_output.txt').read_text()
fig, ax = plt.subplots(figsize=(6.4, 3.0))
for cloud, l, col, ls in (('init', 0, C['teal'], '--'), ('init', 1, C['purple'], '--'), ('trained', 0, C['blue'], '-'), ('trained', 1, C['orange'], '-')):
    row = re.search(rf'{cloud}\s+layer {l} .*?:  (.*)', t).group(1)
    pts = re.findall(r'(\d+)-(\d+): (-?[\d.]+)', row)
    ax.plot(range(len(pts)), [max(0.0, float(p[2])) for p in pts], ls, marker='o', color=col, label=f'{cloud} states, layer {l}')
ax.set_xticks(range(len(pts)), [f'{p[0]}-{p[1]}' for p in pts]); ax.set_xlabel('initial half-life range (tokens)')
ax.set_ylabel('occupations explained\nby position ($R^2$)'); ax.set_ylim(0, 1.02); ax.legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(OUT / 'pm_halflife.png'); plt.close(fig)

# ---------------------------------------------------------------- 5. static landscape (recorded)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
buckets = ['0-15', '16-63', '64-127', '128-255', '256-511']
for arm, folder, kind, col in (('PMX (K=64)', 'pm64_pmx16_sr2', 'pmx', C['blue']), ('FO-b-PMX (K=64)', 'pm64_pmx16_sr2_fob', 'fob-pmx', C['orange']),
                               ('PMX-K16', 'pm16_pmx16_sr2', 'pmx', C['green'])):
    t = (RES / (PF + folder + SF) / f'pm_static_landscape_{kind}_output.txt').read_text()
    for j, var in enumerate(('static', 'off')):
        row = re.search(rf'^\s+{var}\s+(.*)$', t, re.M).group(1)
        vals = [float(v) for v in re.findall(r'([+-][\d.]+)%', row)][1:]
        ax[j].plot(range(5), vals, 'o-', color=col, label=arm)
ax[0].set_title('A. Occupations fixed at their late value: cost', fontsize=9); ax[0].set_ylabel('perplexity change (%)')
ax[1].set_title('B. Wells removed: cost', fontsize=9)
for a in ax:
    a.set_xticks(range(5), buckets); a.set_xlabel('position bucket'); a.legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(OUT / 'pm_static.png'); plt.close(fig)

# ---------------------------------------------------------------- 6. short training (recorded)
t = (DBG / 'pm_match_short_train_output.txt').read_text()
vals = {}
for arm in ('iso', 'lr8'):
    blk = t.split(f'PM-MATCH SHORT TRAIN: {arm}')[1].split('PM-MATCH')[0]
    vals[arm] = [(float(a), float(b)) for a, b in re.findall(r'selectivity ([\d.]+)\s+occupations explained by position R\^2 ([\d.]+)', blk)]
fig, ax = plt.subplots(1, 2, figsize=(8.4, 2.9)); x = np.arange(2); w = 0.36
for j, (arm, col) in enumerate((('iso', C['grey']), ('lr8', C['blue']))):
    ax[0].bar(x + (j - 0.5) * w, [v[0] for v in vals[arm]], w, color=col, label='isotropic' if arm == 'iso' else 'rank 8')
    ax[1].bar(x + (j - 0.5) * w, [v[1] for v in vals[arm]], w, color=col)
ax[0].set_title('A. Selectivity after 300 steps', fontsize=9); ax[0].set_yscale('log'); ax[0].legend(fontsize=7, frameon=False)
ax[1].set_title('B. Occupations explained by position', fontsize=9); ax[1].set_ylim(0.7, 1.01)
for a in ax:
    a.set_xticks(x, ['layer 0', 'layer 1'])
fig.tight_layout(); fig.savefig(OUT / 'pm_shorttrain.png'); plt.close(fig)
print('wrote', sorted(p.name for p in OUT.glob('*.png')))
