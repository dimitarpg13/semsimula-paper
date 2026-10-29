#!/usr/bin/env python3
"""Figures for companion_notes/Dead_Wells_in_the_Joint_V_theta_Bank.md (2026-10-28, protocol p-dead-wells).

    python3 dead_wells_figures.py

Reads only filed results: the code arms' training logs, results/dead_wells/ (vtheta_well_probe.py),
results/copy_split_code/ (copy split, xi_reach). Writes companion_notes/figures/dead_wells/fig*.png.
"""
import json, re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE.parent / 'results'
DW = RES / 'dead_wells'
CS = RES / 'copy_split_code'
FIG = HERE.parents[3] / 'companion_notes/figures/dead_wells'
FIG.mkdir(parents=True, exist_ok=True)

_T = 'topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_sr2_st16250_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn'
RUN = {'SO-CODE': RES / f'cfc_baoab_code_xi5long_{_T}',
       'XI6-CODE': RES / f'cfc_baoab_code_xi6long_{_T.replace("dcvt5x8", "dcvt6x8")}',
       'FOCK-CODE': RES / f'cfc_baoab_code_xi5long_{_T.replace("_norc", "")}'}
GPT2 = RES / 'gpt2_baseline_wsd_lr0p0012_st19250_sch16250_code'

# reference palette (dataviz skill, light mode); colour follows the arm in every figure
C = {'SO-CODE': '#2a78d6', 'XI6-CODE': '#eb6834', 'FOCK-CODE': '#1baf7a', 'SO-OWT': '#eda100',
     'SO-TS': '#e87ba4', 'F3.1-OWT': '#008300', 'G2-OWT': '#6250d6'}
INK, INK2, MUTED, GRID, AXIS, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
plt.rcParams.update({
    'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'font.size': 9.5,
    'text.color': INK, 'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.edgecolor': AXIS, 'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'axes.titlecolor': INK,
    'legend.frameon': False, 'lines.linewidth': 1.6, 'lines.solid_capstyle': 'round'})


def evals(folder):
    p = next(f for f in folder.glob('*.txt') if re.match(r'L2_.*_output\.txt|.*gpt2.*\.txt|.*output.*\.txt', f.name)
             and 'Cell' not in f.name)
    d = {}
    for line in open(p, errors='replace'):
        m = re.search(r'EVAL step ([\d,]+)\s+(?:val_loss=|loss@1024=[\d.]+ ppl@1024=[\d.]+\s+loss@512=)([\d.]+)', line)   # GPT-2: the 512 context
        if m:
            d[int(m.group(1).replace(',', ''))] = float(m.group(2))
    return d


def resonance(folder):
    p = next(f for f in folder.glob('L2_*_output.txt'))
    d = {}
    for line in open(p, errors='replace'):
        m = re.search(r'\[resonance\] step\s+(\d+)\s+omega\*dt p50=([\d.]+)', line)
        if m:
            d[int(m.group(1))] = float(m.group(2))
    return d


def vreg(folder):
    r = [json.loads(l) for l in open(folder / 'training_log.jsonl')]
    r = [x for x in r if 'v_reg' in x and 'event' not in x]
    return np.array([x['step'] for x in r]), np.array([x['v_reg'] for x in r])


def label_end(ax, x, y, text, dy=0):
    ax.annotate(text, (x, y), xytext=(4, dy), textcoords='offset points', va='center', fontsize=8.5, color=INK2)


def save(fig, name):
    fig.savefig(FIG / name, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print('wrote', FIG / name)


# --- Figure 1: validation perplexity and the paired gap --------------------------------------------------
E = {k: evals(v) for k, v in RUN.items()}
g2 = evals(GPT2)
fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={'wspace': 0.32})
for k in ('XI6-CODE', 'SO-CODE', 'FOCK-CODE'):
    s = sorted(E[k]); a.plot(s, np.exp([E[k][t] for t in s]), color=C[k], label=k)
s = sorted(g2); a.plot(s, np.exp([g2[t] for t in s]), color=MUTED, lw=1.2, ls=(0, (4, 3)), label='GPT-2-CODE')
a.legend(fontsize=8.5, loc='upper right')
a.set_yscale('log'); a.set_ylim(4, 50); a.set_yticks([5, 7, 10, 15, 20, 30, 50]); a.set_yticklabels([5, 7, 10, 15, 20, 30, 50])
a.set_xlabel('training step'); a.set_ylabel('validation perplexity (log scale)')
a.set_title('Validation perplexity on code'); a.set_xlim(0, 19500)
s = sorted(set(E['XI6-CODE']) & set(E['SO-CODE']))
gap = [100 * (np.exp(E['XI6-CODE'][t] - E['SO-CODE'][t]) - 1) for t in s]
b.axhline(0, color=AXIS, lw=0.8); b.plot(s, gap, color=C['XI6-CODE'])
b.axvline(2000, color=MUTED, lw=0.8, ls=(0, (2, 2)))
b.annotate('level until step 2,000', (2000, 3), xytext=(6, 0), textcoords='offset points', fontsize=8.5, color=INK2)
b.set_xlabel('training step'); b.set_ylabel('XI6-CODE vs SO-CODE, same step (%)')
b.set_title('Paired gap: a fixed deficit from step 5,500'); b.set_xlim(0, 19500)
save(fig, 'fig1_training_curves.png')

# --- Figure 2: the early signals, v_reg and omega*dt -----------------------------------------------------
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4))
for k in ('SO-CODE', 'XI6-CODE'):
    st, v = vreg(RUN[k]); m = st <= 6000
    a.plot(st[m], v[m], color=C[k], lw=1.3); label_end(a, st[m][-1], v[m][-1], k)
a.set_yscale('log'); a.axvline(2000, color=MUTED, lw=0.8, ls=(0, (2, 2)))
a.set_xlabel('training step'); a.set_ylabel('v_reg = mean V² at the final state')
a.set_title('v_reg separates first, from step ~1,000'); a.set_xlim(0, 7000)
for k in ('SO-CODE', 'FOCK-CODE', 'XI6-CODE'):
    r = resonance(RUN[k]); s = [t for t in sorted(r) if t <= 10000]
    b.plot(s, [r[t] for t in s], color=C[k], marker='o', ms=3.5, mec=SURF, mew=1); label_end(b, s[-1], r[s[-1]], k)
b.axvline(2000, color=MUTED, lw=0.8, ls=(0, (2, 2)))
b.set_xlabel('training step'); b.set_ylabel('median ω·Δt (both layers)')
b.set_title('SO-CODE stiffens from step 2,000; XI6-CODE does not'); b.set_xlim(0, 12500)
save(fig, 'fig2_early_signals.png')

# --- Figure 3: nearest-well distance relative to the state ----------------------------------------------
ARMS = [('so_code', 'SO-CODE'), ('fock_code', 'FOCK-CODE'), ('so_owt', 'SO-OWT'), ('f31_owt', 'F3.1-OWT'),
        ('g2_owt', 'G2-OWT'), ('so_ts', 'SO-TS'), ('xi6_code', 'XI6-CODE')]
P = {name: np.load(DW / f'well_probe_{f}.npz') for f, name in ARMS}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for l, ax in enumerate(axes):
    for f, name in ARMS:
        r = np.sort(P[name][f'L{l}_near_ratio']); y = np.arange(1, len(r) + 1) / len(r)
        healthy = name != 'XI6-CODE'
        ax.plot(r, y, color=(MUTED if healthy else C[name]), lw=(1.0 if healthy else 2.0), alpha=(0.8 if healthy else 1))
    ax.axvline(3, color=INK2, lw=0.9, ls=(0, (3, 2)))
    ax.annotate('ρ = 3', (3, 0.06), xytext=(4, 0), textcoords='offset points', fontsize=8.5, color=INK2)
    ax.set_xlim(0, 5.5); ax.set_xlabel('nearest-well distance / |h|')
    ax.set_title(f'Layer {l}')
axes[0].set_ylabel('share of tokens at or below')
r = np.sort(P['XI6-CODE']['L1_near_ratio']); axes[1].annotate('XI6-CODE', (np.median(r), 0.5), xytext=(8, 0),
                                                             textcoords='offset points', fontsize=8.5, color=INK2)
axes[1].annotate('six healthy five-channel arms\n(code, OpenWebText, TinyStories)', (1.6, 0.85), fontsize=8.5, color=INK2)
save(fig, 'fig3_nearest_well.png')

# --- Figure 4: the layer-1 well centres split into channels, depth code, bias -----------------------------
def split(f):
    t = (DW / f'well_probe_{f}.txt').read_text()  # [|mu|, channel term, depth-code term, bias, cos(code term, mu)]
    m = re.search(r'layer 1: \|mu\|\s+([\d.]+)\s+channels\s+([\d.]+)\s+depth code\s+([\d.]+)\s+bias\s+([\d.]+)\s+'
                  r'cos\(code term, mu\)\s+([-\d.]+)', t)
    return [float(v) for v in m.groups()]
SP = {name: split(f) for f, name in (('so_code', 'SO-CODE'), ('fock_code', 'FOCK-CODE'), ('xi6_code', 'XI6-CODE'))}
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={'width_ratios': [1.25, 1.4], 'wspace': 0.25})
names = list(SP); x = np.arange(len(names)); w = 0.26
for i, (lab, idx, col) in enumerate((('|μ| (well centre)', 0, INK2), ('channel term', 1, MUTED), ('depth-code term', 2, AXIS))):
    v = [SP[n][idx] for n in names]
    bars = a.bar(x + (i - 1) * w, v, width=w * 0.9, color=col, edgecolor=SURF, linewidth=1)
    a.bar_label(bars, fmt='%.0f', fontsize=8, color=INK2, padding=2)
a.set_xticks(x, names); a.set_ylabel('norm (median token)'); a.grid(axis='x', visible=False)
a.legend(['|μ| (well centre)', 'channel term', 'depth-code term'], fontsize=8.5, loc='upper left')
a.set_title('Layer-1 well centres, split'); a.set_ylim(0, 110)
ALL = [('so_code', 'SO-CODE'), ('fock_code', 'FOCK-CODE'), ('so_owt', 'SO-OWT'), ('f31_owt', 'F3.1-OWT'),
       ('g2_owt', 'G2-OWT'), ('so_ts', 'SO-TS'), ('xi6_code', 'XI6-CODE')]
rt = [split(f)[0] / split(f)[1] for f, _ in ALL]
xb = np.arange(len(ALL))
bars = b.bar(xb, rt, width=0.55, color=[C[n] for _, n in ALL], edgecolor=SURF, linewidth=1)
b.bar_label(bars, fmt='%.2f', fontsize=8.5, color=INK2, padding=2)
b.axhline(1, color=INK2, lw=0.9, ls=(0, (3, 2)))
b.set_xticks(xb, [n.replace('-', '-\n', 1) for _, n in ALL], fontsize=8.5); b.set_ylim(0, 1.15); b.grid(axis='x', visible=False)
b.set_ylabel('|μ| ÷ channel term'); b.set_title('The depth code shrinks the centres, except in XI6')
save(fig, 'fig4_well_centres.png')

# --- Figure 5: removing wells from trained models, one layer at a time ------------------------------------
def ppl(f):
    m = re.search(r'loss [\d.]+\s+ppl ([\d.]+)', (DW / f'well_probe_{f}.txt').read_text())
    return float(m.group(1))
cond = [('all 8', ''), ('layer 1:\n2 kept', '_L1_keep2'), ('layer 1:\n1 kept', '_L1_keep1'), ('layer 1:\nnone', '_L1_none'),
        ('layer 0:\n2 kept', '_L0_keep2')]
fig, ax = plt.subplots(figsize=(8.5, 3.4))
x = np.arange(len(cond)); w = 0.34
for i, (arm, f) in enumerate((('SO-CODE', 'so_code'), ('SO-OWT', 'so_owt'))):
    base = ppl(f); xs, vs = [], []
    for j, (_, suf) in enumerate(cond):
        if (DW / f'well_probe_{f}{suf}.txt').exists():
            xs.append(j); vs.append(ppl(f + suf) / base)
    bars = ax.bar(np.array(xs) + (i - 0.5) * w, vs, width=w * 0.9, color=C[arm], edgecolor=SURF, linewidth=1, label=arm)
    ax.bar_label(bars, labels=[f'{v:.2f}×' if v < 10 else f'{v:.0f}×' for v in vs], fontsize=8, color=INK2, padding=2)
ax.set_yscale('log'); ax.set_ylim(0.8, 60); ax.set_yticks([1, 2, 5, 10, 20, 50]); ax.set_yticklabels(['1×', '2×', '5×', '10×', '20×', '50×'])
ax.set_xticks(x, [c for c, _ in cond]); ax.grid(axis='x', visible=False); ax.legend(fontsize=8.5, loc='upper left')
ax.set_ylabel('perplexity relative to all wells'); ax.set_title('Layer 1 runs on one or two wells; layer 0 needs all eight')
save(fig, 'fig5_well_masking.png')

# --- Figure 6: the copy split ----------------------------------------------------------------------------
J = json.load(open(CS / 'copy_split_report.json'))['models']
groups = [('first occurrence', 'first occurrence'), ('repeat (not induction)', 'repeat'), ('induction', 'induction'),
          ('repeat/ind., dist 1-16', 'last seen 1–16 back'), ('repeat/ind., dist 17-64', '17–64 back'),
          ('repeat/ind., dist 65-256', '65–256 back'), ('repeat/ind., dist 257-512', '257–512 back')]
fig, ax = plt.subplots(figsize=(8.5, 3.6))
y = np.arange(len(groups))[::-1]; h = 0.36
for i, (arm, key) in enumerate((('SO-CODE', 'so'), ('XI6-CODE', 'xi6'))):
    v = [J[key][g]['gap'] for g, _ in groups]
    bars = ax.barh(y + (0.5 - i) * h, v, height=h * 0.9, color=C[arm], edgecolor=SURF, linewidth=1, label=arm)
    ax.bar_label(bars, fmt='%+.2f', fontsize=8, color=INK2, padding=2)
ax.set_yticks(y, [lab for _, lab in groups]); ax.grid(axis='y', visible=False); ax.set_xlim(0, 1.45)
ax.set_xlabel('loss above GPT-2-CODE on the same targets (nats per token)'); ax.legend(fontsize=8.5, loc='upper right')
ax.set_title('Where XI6-CODE loses: copyable and long-range targets first')
save(fig, 'fig6_copy_split.png')

# --- Figure 7: what the layer-1 context channels retrieve -------------------------------------------------
def reach(f):
    lift, peak = {}, {}
    for line in open(CS / f):
        p = line.split()
        if len(p) == 9 and p[0] == '1' and p[2] == 'far':
            lift[int(p[1])] = float(p[6])
        if len(p) == 6 and p[0] == '1' and p[-1].endswith('%'):
            peak[int(p[1])] = float(p[2])
    return lift, peak
R = {'SO-CODE': reach('xi_reach_so.txt'), 'XI6-CODE': reach('xi_reach_xi6.txt')}
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.2))
for i, arm in enumerate(R):
    lift, peak = R[arm]; ch = sorted(lift)
    bars = a.bar(np.array(ch) + (i - 0.5) * 0.36, [lift[c] for c in ch], width=0.32, color=C[arm], edgecolor=SURF, linewidth=1, label=arm)
    a.bar_label(bars, fmt='%.1f', fontsize=7.5, color=INK2, padding=2)
    if peak:
        bars = b.bar(np.array(sorted(peak)) + (i - 0.5) * 0.36, [peak[c] for c in sorted(peak)], width=0.32, color=C[arm],
                     edgecolor=SURF, linewidth=1, label=arm)
        b.bar_label(bars, fmt='%.2f', fontsize=7.5, color=INK2, padding=2)
a.set_xticks(range(6), [f'ch {c + 1}' for c in range(6)]); a.grid(axis='x', visible=False); a.legend(fontsize=8.5)
a.set_ylabel('lift on earlier copies 65+ back'); a.set_title('Layer 1: does the channel find earlier copies?')
b.set_xticks(range(6), [f'ch {c + 1}' for c in range(6)]); b.grid(axis='x', visible=False); b.set_ylim(0, 1.12)
b.set_ylabel('median peak weight on one position'); b.set_title('Layer 1: how peaked is its lookup?')
save(fig, 'fig7_xi_reach.png')
