#!/usr/bin/env python3
"""Figures for the results part of Spontaneous_Synchronization_and_Semantic_Binding_Test_Programme.md.

Reads the SY outputs in notebooks/conservative_arch/scaleup/debug/ and writes
figures/sy_results/*.png. The statistics (with their bootstrap intervals) are cached in
figures/sy_results/_numbers.json; delete it to recompute.

    python3 _make_sy_results_figs.py
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

HERE = Path(__file__).resolve().parent
DBG = HERE.parents[1] / 'notebooks/conservative_arch/scaleup/sync'
OUT = HERE / 'sy_results'; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(DBG))
import sy_instruments as S
from sy_routes_stats import enrichment_ci

ARMS = ['F3.1', 'SR2', 'G2', 'PM1', 'PMX', 'L4F']
LABEL = {'F3.1': 'F3.1', 'SR2': 'SR2', 'G2': 'G2', 'PM1': 'PM1', 'PMX': 'PMX', 'L4F': 'L = 4'}
NL = {'F3.1': 2, 'SR2': 2, 'G2': 2, 'PM1': 2, 'PMX': 2, 'L4F': 4}
REF = {'F3.1': 'fcom', 'L4F': 'fcom', 'SR2': 'dcom', 'G2': 'dcom', 'PM1': 'dcom', 'PMX': 'dcom'}
COUP = {'F3.1': ['vphi'], 'SR2': ['vphi'], 'G2': ['vphi', 'fock'], 'PM1': ['vphi', 'wells'], 'PMX': ['vphi', 'wells'], 'L4F': ['vphi', 'fock']}
LING = ['coref', 'dep', 'chunk', 'ent', 'pos']
GNAME = {'coref': 'coreference', 'dep': 'head-dependent', 'chunk': 'noun chunk', 'ent': 'named entity', 'pos': 'part of speech'}
CCOL = {'vphi': '#3b82f6', 'wells': '#22c55e', 'fock': '#ef4444', 'xi': '#a855f7'}
CNAME = {'vphi': 'pair potential', 'wells': 'Poisson wells', 'fock': 'Fock registers', 'xi': 'context field'}
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})


# ---------------------------------------------------------------- numbers (cached)
cache = OUT / '_numbers.json'
if cache.exists():
    N = json.loads(cache.read_text())
else:
    N = {}
def num(key, fn):
    if key not in N:
        N[key] = fn()
        cache.write_text(json.dumps(N))
    return N[key]

def load(kind, arm):
    return np.load(DBG / f'sy_{kind}_{arm}.npz')

for a in ARMS:
    m = load('measure', a); last = NL[a] - 1
    for c in COUP[a]:
        for l in range(NL[a]):
            num(f'{a}/S/{c}/L{l}', lambda: S.boot_mean(m[f'L{l}/{c}/{REF[a]}/Gamma']))
        for g in LING:
            num(f'{a}/Bspec/{c}/{g}', lambda: S.boot_contrast_diff(m[f'L{last}/{g}/B_on'], m[f'L{last}/{g}/{c}/{REF[a]}/B_ref']))
    for g in ('chunk', 'dep', 'coref'):
        num(f'{a}/Bdepth/{g}', lambda: [S.boot_contrast(m[f'L{l}/{g}/B_in'])[0] for l in range(NL[a])] + [S.boot_contrast(m[f'L{last}/{g}/B_on'])[0]])
    x = load('xi_pred', a)
    for l in range(NL[a]):
        num(f'{a}/S/xi/L{l}', lambda: S.boot_mean(x[f'L{l}/common/Gamma']))
    for g in LING:
        num(f'{a}/Bspec/xi/{g}', lambda: S.boot_contrast_diff(x[f'L{last}/{g}/B_on'], x[f'L{last}/{g}/common/B_ref']))
    for g in ('chunk', 'dep'):
        for k in ('B_state', 'B_resid', 'B_target', 'B_pred'):
            num(f'{a}/pred/{g}/{k}', lambda: S.boot_contrast(x[f'pred/{g}/{k}']))
    r = load('routes', a)
    for g in LING:
        num(f'{a}/enrich/{g}', lambda: enrichment_ci(r[f'L{last}/{g}/sel']))
    N[f'{a}/probe'] = json.loads((DBG / f'sy_coref_probe_{a}.json').read_text())
cache.write_text(json.dumps(N))
sy0 = json.loads((DBG / 'sy0.json').read_text()); sy0v1 = json.loads((DBG / 'sy0_v1.json').read_text())


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches='tight'); plt.close(fig); print('wrote', name)


# ---------------------------------------------------------------- Figure 1: the measuring tools
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
k1, k1v = sy0['checks']['1_kuramoto'], sy0v1['checks']['1_kuramoto']
fr = sorted(k1, key=float); xs = [float(f) for f in fr]
r_on = [k1[f]['r_on'] for f in fr]
G = np.array([k1[f]['Gamma_spec'] for f in fr]); R = np.array([k1v[f]['dR_c'] for f in fr])
a0 = ax[0]
a0.plot(xs, r_on, 'k-o', lw=2, label='true order parameter r (known answer)')
a0.plot(xs, G[:, 0] / G[:, 0].max(), '-s', color='#a855f7', lw=2, label='contraction S (adopted), scaled to 1')
a0.plot(xs, R[:, 0] / R[:, 0].max(), '--^', color='#94a3b8', lw=2, label='centred coherence (rejected), scaled to 1')
a0.axvline(1, color='0.6', ls=':', lw=1); a0.text(1.03, 0.05, 'threshold', color='0.4', fontsize=8)
a0.set_xlabel('coupling strength K / K_c'); a0.set_ylabel('value (each scaled to its maximum)')
a0.set_title('(a) Coupled oscillators: which measure\nfollows the true synchronisation?', fontsize=10)
a0.legend(fontsize=7.5, loc='center right')
a1 = ax[1]
rows = [('shared drift', sy0['checks']['4a']['384']), ('drift +\nnormalisation', sy0['checks']['4b']['384']), ('one fixed\nattractor', sy0['checks']['5']['384'])]
xx = np.arange(len(rows)); w = 0.36
a1.bar(xx - w / 2, [r[1]['Gamma_tot'][0] for r in rows], w, color='#94a3b8', label='against coupling switched off')
a1.bar(xx + w / 2, [r[1]['Gamma_spec'][0] for r in rows], w, color='#a855f7', label='against common-mode reference')
a1.set_xticks(xx); a1.set_xticklabels([r[0] for r in rows], fontsize=8.5)
a1.set_ylabel('contraction S'); a1.legend(fontsize=7.5)
a1.set_title('(b) A shared push reads zero against the common-mode\nreference; a fixed attractor still reads as contraction', fontsize=10)
for i, r in enumerate(rows):
    a1.text(i + w / 2, r[1]['Gamma_spec'][0] + 0.03, f"{r[1]['Gamma_spec'][0]:+.2f}", ha='center', fontsize=8)
a2 = ax[2]
c2 = sy0['checks']['2_attention']; T = [1, 2, 4, 8]
b = np.array([c2[str(t)]['B_spec'] for t in T])
a2.errorbar(T, b[:, 0], yerr=[b[:, 0] - b[:, 1], b[:, 2] - b[:, 0]], fmt='-o', color='#a855f7', lw=2, capsize=3, label='planted groups')
sh = c2['shuffled_T8']['B_spec']
a2.errorbar([8.3], [sh[0]], yerr=[[sh[0] - sh[1]], [sh[2] - sh[0]]], fmt='s', color='0.4', capsize=3, label='same, labels shuffled')
a2.axhline(0.02, color='0.6', ls=':', lw=1); a2.text(1, 0.06, 'threshold 0.02', color='0.4', fontsize=8)
a2.set_xlabel('depth (integration time)'); a2.set_ylabel('binding contrast B')
a2.set_title('(c) Groups planted only through the coupling\nare recovered; scrambled labels read zero', fontsize=10)
a2.legend(fontsize=7.5)
save(fig, 'sy_r1_instruments.png')


# ---------------------------------------------------------------- Figure 2: what each coupling does to the whole set
fig, ax = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw={'width_ratios': [1.6, 1]})
a0 = ax[0]
two = [a for a in ARMS if a != 'L4F']
cs = ['vphi', 'wells', 'fock', 'xi']; w = 0.2
for k, c in enumerate(cs):
    vals, xsb = [], []
    for i, a in enumerate(two):
        if c == 'xi' or c in COUP[a]:
            vals.append(N[f'{a}/S/{c}/L1'][0]); xsb.append(i + (k - 1.5) * w)
    a0.bar(xsb, vals, w, color=CCOL[c], label=CNAME[c])
a0.axhline(0, color='k', lw=0.8)
a0.set_yscale('symlog', linthresh=0.05); a0.set_ylim(-1.2, 2.5)
a0.set_xticks(range(len(two))); a0.set_xticklabels([LABEL[a] for a in two])
a0.set_ylabel('contraction S at the last layer\n(> 0: pulled together, < 0: spread apart)')
a0.set_title('(a) Each coupling\'s effect on the token set as a whole (two-layer models)', fontsize=10)
a0.legend(fontsize=8, ncol=4, loc='lower left')
a1 = ax[1]
for c in ('vphi', 'fock', 'xi'):
    v = [N[f'L4F/S/{c}/L{l}'][0] for l in range(4)]
    a1.plot(range(4), v, '-o', color=CCOL[c], lw=2, label=CNAME[c])
a1.axhline(0, color='k', lw=0.8); a1.set_yscale('symlog', linthresh=0.02)
a1.set_xticks(range(4)); a1.set_xlabel('layer'); a1.set_ylabel('contraction S')
a1.set_title('(b) The four-layer model, layer by layer:\nno build-up with depth', fontsize=10)
a1.legend(fontsize=8)
save(fig, 'sy_r2_couplings.png')


# ---------------------------------------------------------------- Figure 3: the pair potential does not choose related tokens
fig, ax = plt.subplots(figsize=(10.5, 4.2))
w = 0.13
for k, a in enumerate(ARMS):
    e = np.array([N[f'{a}/enrich/{g}'] for g in LING])
    xsb = np.arange(len(LING)) + (k - 2.5) * w
    ax.bar(xsb, e[:, 0], w, yerr=[e[:, 0] - e[:, 1], e[:, 2] - e[:, 0]], capsize=2, label=LABEL[a],
           color=plt.cm.viridis(k / len(ARMS)))
ax.axhline(1.0, color='k', lw=1); ax.text(4.45, 1.02, 'chance', fontsize=8)
ax.axhline(1.2, color='#ef4444', ls='--', lw=1); ax.text(4.45, 1.22, 'required 1.2', fontsize=8, color='#ef4444')
ax.set_xticks(range(len(LING))); ax.set_xticklabels([GNAME[g] for g in LING])
ax.set_ylabel('selection rate of related sources\n/ rate for unrelated ones at the same distance')
ax.set_title('Which earlier tokens does the pair potential choose to interact with? (last layer)', fontsize=10)
ax.legend(fontsize=8, ncol=6, loc='upper left')
save(fig, 'sy_r3_selection.png')


# ---------------------------------------------------------------- Figure 4: the pattern, layer by layer
fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
for j, g in enumerate(('chunk', 'dep')):
    for k, a in enumerate(ARMS):
        v = N[f'{a}/Bdepth/{g}']
        x = np.arange(len(v))
        ax[j].plot(x, v, '-o', lw=2.2 if a == 'L4F' else 1.6, color=plt.cm.viridis(k / len(ARMS)), label=LABEL[a])
    ax[j].axhline(0, color='k', lw=0.8)
    ax[j].axhspan(-0.02, 0.02, color='0.92', zorder=0)
    ax[j].set_xticks(range(5)); ax[j].set_xticklabels(['embedding', 'after 1', 'after 2', 'after 3', 'after 4'])
    ax[j].set_ylabel('binding contrast B')
ax[0].set_title('(a) Words of one noun chunk: drawn together', fontsize=10)
ax[1].set_title('(b) A word and its syntactic head: pushed apart', fontsize=10)
ax[0].legend(fontsize=8, ncol=2)
ax[0].text(0.05, 0.022, 'grey band: below the 0.02 threshold', fontsize=7.5, color='0.4')
save(fig, 'sy_r4_depth.png')


# ---------------------------------------------------------------- Figure 5: which coupling makes it
fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
for j, g in enumerate(('chunk', 'dep')):
    w = 0.2
    for k, c in enumerate(['vphi', 'wells', 'fock', 'xi']):
        vals, xsb, err = [], [], []
        for i, a in enumerate(ARMS):
            if c == 'xi' or c in COUP[a]:
                b = N[f'{a}/Bspec/{c}/{g}']
                vals.append(b[0]); xsb.append(i + (k - 1.5) * w); err.append([b[0] - b[1], b[2] - b[0]])
        err = np.array(err).T
        ax[j].bar(xsb, vals, w, yerr=err, capsize=2, color=CCOL[c], label=CNAME[c])
    ax[j].axhline(0, color='k', lw=0.8)
    ax[j].axhline(0.02 if g == 'chunk' else -0.02, color='0.5', ls=':', lw=1)
    ax[j].set_xticks(range(len(ARMS))); ax[j].set_xticklabels([LABEL[a] for a in ARMS])
    ax[j].set_ylabel('binding made by the coupling (last layer)')
ax[0].set_title('(a) Noun chunks: what each coupling contributes', fontsize=10)
ax[1].set_title('(b) Head-dependent: what each coupling contributes', fontsize=10)
ax[0].legend(fontsize=8, ncol=2)
save(fig, 'sy_r5_attribution.png')


# ---------------------------------------------------------------- Figure 6: not the geometry of prediction
fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
for j, g in enumerate(('chunk', 'dep')):
    w = 0.26
    for k, (key, col, nm) in enumerate([('B_state', '#64748b', 'in the states'),
                                         ('B_resid', '#a855f7', 'left after removing prediction similarity'),
                                         ('B_target', '#f59e0b', 'in the true next words')]):
        b = np.array([N[f'{a}/pred/{g}/{key}'] for a in ARMS])
        ax[j].bar(np.arange(len(ARMS)) + (k - 1) * w, b[:, 0], w, yerr=[b[:, 0] - b[:, 1], b[:, 2] - b[:, 0]],
                  capsize=2, color=col, label=nm)
    ax[j].axhline(0, color='k', lw=0.8)
    ax[j].set_xticks(range(len(ARMS))); ax[j].set_xticklabels([LABEL[a] for a in ARMS])
    ax[j].set_ylabel('binding contrast B (last layer)')
ax[0].set_title('(a) Noun chunks: survives; the data has none of it', fontsize=10)
ax[1].set_title('(b) Head-dependent: explained by prediction', fontsize=10)
ax[0].legend(fontsize=8)
save(fig, 'sy_r6_prediction.png')


# ---------------------------------------------------------------- Figure 7: clustered, not global (schematic + numbers)
rng = np.random.default_rng(3)
fig, ax = plt.subplots(1, 4, figsize=(18, 4.4), gridspec_kw={'width_ratios': [1, 1, 0.95, 0.95], 'wspace': 0.42})
cols = ['#3b82f6', '#22c55e', '#ef4444', '#f59e0b', '#a855f7']
centres = np.array([[-2.2, 1.2], [1.8, 1.8], [2.2, -1.6], [-1.6, -1.9], [0.1, 0.0]])
for gi, c in enumerate(centres):
    pts = c + rng.normal(0, 0.3, (5, 2))
    ax[0].scatter(pts[:, 0], pts[:, 1], s=42, color=cols[gi], edgecolor='k', lw=0.4, zorder=3)
    ax[0].add_patch(Ellipse(c, 1.5, 1.5, fill=False, ec=cols[gi], lw=1, ls='--'))
    pts = 0.3 * c + rng.normal(0, 0.45, (5, 2))
    ax[1].scatter(pts[:, 0], pts[:, 1], s=42, color=cols[gi], edgecolor='k', lw=0.4, zorder=3)
for p, title, note in ((0, '(a) each token reads its own context', 'spread out overall,\ntight within each noun chunk'),
                       (1, '(b) every token reads the same context', 'pulled together overall,\nnoun chunks mixed')):
    ax[p].set_xlim(-3.6, 3.6); ax[p].set_ylim(-3.2, 3.2); ax[p].set_aspect('equal')
    ax[p].set_xticks([]); ax[p].set_yticks([]); ax[p].set_title(title, fontsize=10)
    ax[p].text(-3.4, -3.05, note, fontsize=7.8)
ax[0].text(-3.4, 2.75, 'colour = one noun chunk', fontsize=7.8)
xx = np.arange(len(ARMS))
sx = np.array([N[f'{a}/S/xi/L{NL[a] - 1}'] for a in ARMS])
ax[2].bar(xx, sx[:, 0], 0.6, yerr=[sx[:, 0] - sx[:, 1], sx[:, 2] - sx[:, 0]], capsize=2, color='#94a3b8')
ax[2].axhline(0, color='k', lw=0.8)
ax[2].set_xticks(xx); ax[2].set_xticklabels([LABEL[a] for a in ARMS], fontsize=8.5)
ax[2].set_ylabel('contraction S, own vs same context\n(< 0: own context spreads tokens apart)')
ax[2].set_title('(c) Measured, whole token set', fontsize=10)
bx = np.array([N[f'{a}/Bspec/xi/chunk'] for a in ARMS])
ax[3].bar(xx, bx[:, 0], 0.6, yerr=[bx[:, 0] - bx[:, 1], bx[:, 2] - bx[:, 0]], capsize=2, color='#a855f7')
ax[3].axhline(0.02, color='0.5', ls=':', lw=1); ax[3].text(-0.4, 0.022, 'threshold', fontsize=7.5, color='0.4')
ax[3].set_xticks(xx); ax[3].set_xticklabels([LABEL[a] for a in ARMS], fontsize=8.5)
ax[3].set_ylabel('noun-chunk binding made by the context')
ax[3].set_title('(d) Measured, within noun chunks', fontsize=10)
fig.text(0.26, -0.03, '(a) and (b) are schematic illustrations of the two outcomes, not data; (c) and (d) are measured at the last layer',
         ha='left', fontsize=8, color='0.35')
save(fig, 'sy_r7_clustered.png')


# ---------------------------------------------------------------- Figure 8: coreference
cc = {a: json.loads((DBG / f'sy_coref_subspace_check_{a}.json').read_text())
      for a in ['PMX', 'PM1', 'F3.1', 'G2', 'SR2'] if (DBG / f'sy_coref_subspace_check_{a}.json').exists()}
fig, ax = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw={'width_ratios': [1.2, 1]})
a0 = ax[0]
for k, a in enumerate(ARMS):
    acc = N[f'{a}/probe']['acc']
    ys = [acc['emb']] + [acc[str(l)] for l in range(NL[a])]
    a0.plot(range(len(ys)), ys, '-o', color=plt.cm.viridis(k / len(ARMS)), lw=1.8, label=LABEL[a])
pr = N['PMX/probe']
a0.axhline(pr['nearest'], color='#ef4444', ls='--'); a0.text(3.05, pr['nearest'] + 0.008, 'pick the nearest mention', fontsize=8, color='#ef4444')
a0.axhline(pr['chance'], color='0.5', ls=':'); a0.text(3.05, pr['chance'] + 0.008, 'chance', fontsize=8, color='0.4')
a0.set_xticks(range(5)); a0.set_xticklabels(['embedding', 'after 1', 'after 2', 'after 3', 'after 4'])
a0.set_ylabel('antecedent found (test accuracy)'); a0.set_ylim(0.1, 0.65)
a0.set_title('(a) Can a pronoun\'s antecedent be read from its state?', fontsize=10)
a0.legend(fontsize=8, ncol=3, loc='lower center')
a1 = ax[1]
if cc:
    variants = [('all', 'all coreferent pairs'), ('nonident', 'identical tokens removed'),
                ('nonword', 'same words removed'), ('pron', 'pronoun-antecedent only')]
    w = 0.16
    for k, a in enumerate(cc):
        res = cc[a]['results']
        vals = [res[v]['B_spec'][0] if v in res else np.nan for v, _ in variants]
        err = np.array([[res[v]['B_spec'][0] - res[v]['B_spec'][1], res[v]['B_spec'][2] - res[v]['B_spec'][0]] if v in res else [0, 0] for v, _ in variants]).T
        a1.bar(np.arange(len(variants)) + (k - 2) * w, vals, w, yerr=err, capsize=2, color=plt.cm.viridis(ARMS.index(a) / len(ARMS)), label=LABEL[a])
    a1.axhline(0, color='k', lw=0.8); a1.axhline(0.02, color='0.5', ls=':', lw=1)
    a1.axvspan(2.55, 3.45, color='#fef3c7', zorder=0)
    a1.text(3.0, -0.075, 'shuffled controls not null here:\nread with the probe, not alone', ha='center', fontsize=7.5, color='#92400e')
    a1.set_xticks(range(len(variants))); a1.set_xticklabels([v[1] for v in variants], fontsize=8, rotation=12)
    a1.set_ylabel('binding made by the context\n(top 16 directions, first layer)')
    a1.legend(fontsize=8, ncol=3)
else:
    a1.text(0.5, 0.5, 'coreference check pending', ha='center', transform=a1.transAxes)
a1.set_title('(b) The one coreference signal: what is it made of?', fontsize=10)
save(fig, 'sy_r8_coreference.png')
