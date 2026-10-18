#!/usr/bin/env python3
"""Figure for the XR comparison (protocol §5.25): the exchange field against the register path,
in the trained G2, G3, G3′ (part A) and the field-only XRB-F (part B).

Reads notebooks/conservative_arch/scaleup/xr/xr_<ARM>.json and writes figures/xr/xr_routes.png:
  (a) each route's removal cost by kind of position (induction, repeated, novel);
  (b) how much of each route's value survives when only its most important units are kept;
  (c) where each route's kernel puts its weight: on induction targets, and beyond 64 tokens back.

    python3 _make_xr_figs.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
XR = HERE.parents[1] / 'notebooks/conservative_arch/scaleup/xr'
OUT = HERE / 'xr'; OUT.mkdir(exist_ok=True)
J = {a: json.loads((XR / f'xr_{a}.json').read_text()) for a in ('G2', 'G3', 'G3p', 'XRBF')}
NAME = {'G2': 'G2', 'G3': 'G3', 'G3p': 'G3′', 'XRBF': 'XRB-F'}
REG, EX = '#2E7D32', '#E65100'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(1, 3, figsize=(18.5, 5.1), gridspec_kw={'wspace': 0.32, 'width_ratios': [1.25, 1, 1.1]})

# (a) removal cost by kind of position
bars = [('G2\nregisters\n(alone)', J['G2']['classes']['read'], REG, 1.0), ('XRB-F\nfield\n(alone)', J['XRBF']['classes']['field'], EX, 1.0),
        ('G3\nregisters', J['G3']['classes']['read'], REG, 1.0),
        ('G3\nfield', J['G3']['classes']['field'], EX, 1.0), ('G3′\nregisters', J['G3p']['classes']['read'], REG, 0.55),
        ('G3′\nfield', J['G3p']['classes']['field'], EX, 0.55)]
kinds = [('induction', 'induction (copyable)', '//'), ('repeated', 'repeated', ''), ('novel', 'novel', '..')]
w = 0.26
for i, (name, row, c, al) in enumerate(bars):
    for j, (k, lab, hatch) in enumerate(kinds):
        v = row[k]
        ax[0].bar(i + (j - 1) * w, v[0], w, yerr=[[v[0] - v[1]], [v[2] - v[0]]], capsize=2, color=c, alpha=al,
                  hatch=hatch, edgecolor='white', label=lab if i == 0 else None)
ax[0].set_xticks(range(len(bars))); ax[0].set_xticklabels([b[0] for b in bars], fontsize=9)
ax[0].set_ylabel('loss increase when the route is removed (nats per token)')
ax[0].axvline(1.5, color='0.6', lw=0.8, ls=':')
ax[0].set_title('(a) Alone, either route carries loss everywhere; beside the\nregister path, the field serves copyable (induction) positions', fontsize=10)
ax[0].legend(fontsize=8, title='kind of position', title_fontsize=8)

# (b) capacity
for arm, ls in (('G2', ':'), ('G3', '-'), ('G3p', '--'), ('XRBF', '-.')):
    cap = J[arm]['capacity']
    if 'read' in cap:
        r = cap['read']['top']; ks = sorted(int(k) for k in r)
        ax[1].plot([k / 32 for k in ks], [r[str(k)] for k in ks], ls, marker='o', color=REG, lw=1.8, label=f'{NAME[arm]} registers (of 32)')
    if 'field' in cap:
        f = cap['field']['top']; kh = sorted(int(k) for k in f)
        ax[1].plot([k / 8 for k in kh], [f[str(k)] for k in kh], ls, marker='s', color=EX, lw=1.8,
                   mfc='white' if arm == 'XRBF' else EX, label=f'{NAME[arm]} heads (of 8)')
ax[1].axhline(0.9, color='0.5', ls=':', lw=1); ax[1].text(0.02, 0.92, '90% retained', fontsize=8, color='0.4')
ax[1].set_xlabel('share of the route\'s units kept (most important first)'); ax[1].set_ylabel('share of the route\'s value retained')
ax[1].set_ylim(-0.15, 1.05)
ax[1].set_title('(b) Both routes are wide: keeping 90% of a route\'s\nvalue takes about half its units or more', fontsize=10)
ax[1].legend(fontsize=8)

# (c) kernel scores at the last layer
labels, ind, far, cols = [], [], [], []
for arm in ('G2', 'G3', 'G3p', 'XRBF'):
    for route, c in (('reg', REG), ('field', EX)):
        k = J[arm]['kernels'].get(f'{route}/L1')
        if k:
            labels.append(f'{NAME[arm]}\n{"registers" if route == "reg" else "field"}')
            ind.append(k['induction'][0]); far.append(k['far64'][0]); cols.append(c)
x = np.arange(len(labels))
ax[2].bar(x - 0.2, ind, 0.4, color=cols, label='weight on induction targets')
a2 = ax[2].twinx(); a2.bar(x + 0.2, far, 0.4, color=cols, alpha=0.35, label='share of weight beyond 64 tokens back')
a2.set_ylim(0, 1); a2.spines['right'].set_visible(True); a2.set_ylabel('share of weight beyond 64 tokens back (pale)')
ax[2].set_xticks(x); ax[2].set_xticklabels(labels, fontsize=9); ax[2].set_ylabel('weight on induction targets (solid)')
ax[2].set_title('(c) The field puts 2–5× more weight on induction\ntargets and reaches slightly further back', fontsize=10)
fig.suptitle('The exchange field against the register path (trained G2, G3, G3′ and the field-only XRB-F; protocol §5.25)', fontsize=11.5, y=1.02)
fig.savefig(OUT / 'xr_routes.png', dpi=150, bbox_inches='tight'); print('wrote xr_routes.png')
