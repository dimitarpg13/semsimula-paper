#!/usr/bin/env python3
"""
SY binding in a subspace: is binding by closeness hidden under each token's own
identity in the whole-vector cosine? (protocol §5.22; criteria fixed
2026-10-14 before the first run.)

    python3 sy_subspace.py ARM [n_seq=200]      ARM in F3.1 SR2 G2 PM1 PMX L4F

B_spec (trained step against the coupling's common-mode reference, the one
sy_measure.py adopted for the arm) recomputed after projecting the states onto
a fixed family of subspaces of the layer's output, fitted on sequences 0-49
and measured on 50-199:
  pc1-16     top 16 principal components (where token identity mostly lives)
  pc17-64    components 17-64
  pc65+      the remainder (low-variance directions)
  wells      span of the 64 mode centres (PM1, PMX)
  probe      the 64 directions of the coreference probe's query and key maps on
             the last layer (sy_coref_probe.py), last layer only
  rand16, rand64, rand320   random orthonormal subspaces of matched dimension (controls)
  whole      no projection (must reproduce sy_measure's stored statistics exactly)
Couplings as in sy_measure, plus the context xi (common-mode replay of xi).

A hidden binding signal: for a linguistic grouping (coref, dep, chunk, ent,
pos), a coupling and a subspace, B_spec is positive, passes the floor, its
shuffled control is null, and it exceeds the matched-dimension random
subspace's B_spec by at least the floor; replicated in PMX and in at least one
other arm with the same route (xi: any two arms including PMX).

Output beside this script: sy_subspace_<ARM>.npz and _output.txt.
"""
import json, sys, time
from pathlib import Path

import numpy as np
import torch

ARM = sys.argv[1]
N_SEQ = next((int(a[6:]) for a in sys.argv[2:] if a.startswith('n_seq=')), 200)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent / 'debug'))   # shared model-building helpers (verify_pm_switch, ...)
from sy_replay import Replayer, CACHE
import sy_instruments as S
from sy_groupings import Groupings, bin_stats_rel

LING = ('coref', 'dep', 'chunk', 'ent', 'pos')
N_FIT = 50 if N_SEQ >= 200 else max(2, N_SEQ // 4)
lines = []
say = lambda s='': (print(s, flush=True), lines.append(s))

R = Replayer(ARM)
m, L = R.model, R.L
X = np.load(CACHE / 'sy_tokens.npz')['x'][:N_SEQ]
GR = Groupings(np.load(CACHE / 'sy_annotation.npz'), X)
say(f'SY binding in a subspace: {ARM} (L = {L}; couplings {R.couplings} + xi; common reference {R.ref})')
R.check(torch.from_numpy(X[:2]), say)

# ---------------------------------------------------------------- subspaces, fitted on sequences 0..N_FIT-1
fit = {l: [] for l in range(L)}
for b0 in range(0, N_FIT, 4):
    R.forward(torch.from_numpy(X[b0:min(b0 + 4, N_FIT)]))
    for l in range(L):
        fit[l].append(R.records[l]['out'].double().numpy().reshape(-1, m.cfg.d))
rng0 = np.random.default_rng(7)
def orth(A):
    Q, _ = np.linalg.qr(A); return Q
SUB = {}
for l in range(L):
    Hf = np.concatenate(fit[l]); Hf = Hf - Hf.mean(0)
    _, _, Vt = np.linalg.svd(Hf, full_matrices=False)
    SUB[l] = {'pc1-16': Vt[:16].T, 'pc17-64': Vt[16:64].T, 'pc65+': Vt[64:].T,
              'rand16': orth(rng0.normal(size=(m.cfg.d, 16))), 'rand64': orth(rng0.normal(size=(m.cfg.d, 64))),
              'rand320': orth(rng0.normal(size=(m.cfg.d, 320)))}
    if getattr(m, 'pm_mu', None) is not None:
        SUB[l]['wells'] = orth(m.pm_mu.detach().double().numpy().T)
pf = HERE / f'sy_coref_probe_{ARM}_subspace.npz'
if pf.exists():
    SUB[L - 1]['probe'] = orth(np.load(pf)['basis'])
MATCH = {'pc1-16': 'rand16', 'pc17-64': 'rand64', 'pc65+': 'rand320', 'wells': 'rand64', 'probe': 'rand64'}
say(f'   subspaces: ' + ', '.join(f'{k} ({v.shape[1]})' for k, v in SUB[L - 1].items()) + ('' if 'probe' in SUB[L - 1] else '   [no probe subspace: sy_coref_probe has not saved one]'))

def proj_cos(H, Q):
    return S.cos_matrix(S.centred(H if Q is None else H @ Q))

# ---------------------------------------------------------------- measurement on sequences N_FIT..N_SEQ-1
ref_meas = dict(np.load(HERE / f'sy_measure_{ARM}.npz')) if (HERE / f'sy_measure_{ARM}.npz').exists() else None
store = {}
def put(k, v): store.setdefault(k, []).append(v)
rng = np.random.default_rng(20261014)
t0 = time.time(); checked = False
for b0 in range(N_FIT, N_SEQ, 2):
    xb = torch.from_numpy(X[b0:b0 + 2])
    R.forward(xb)
    vers = {}
    for l in range(L):
        on = R.replay(l, None)
        vers[(l, 'on')] = on
        for c in R.couplings:
            vers[(l, c)] = R.common_ref(l, c, on)
        vers[(l, 'xi')] = R.replay(l, ('xi', 'common'))
    for j in range(xb.shape[0]):
        i = b0 + j
        G, Gs = GR(i, rng)
        for l in range(L):
            post_on = vers[(l, 'on')][1][j]
            subs = dict(SUB[l]); subs['whole'] = None
            for sname, Q in subs.items():
                cos_on = proj_cos(post_on, Q)
                for name in LING:
                    put(f'L{l}/{sname}/{name}/B_on', bin_stats_rel(cos_on, *G[name]))
                    put(f'L{l}/{sname}/{name}/shuf/B_on', bin_stats_rel(cos_on, *Gs[name]))
                if sname == 'whole' and ref_meas is not None and not checked and i < 200 and 'L0/coref/B_on' in ref_meas:
                    d = np.abs(bin_stats_rel(cos_on, *G['coref']) - ref_meas[f'L{l}/coref/B_on'][i]).max()
                    say(f'CHECK  whole-space statistics vs sy_measure_{ARM}.npz (sequence {i}, layer {l}, coref): max |d| {d:.1e}')
                    assert d < 1e-9
                    checked = l == L - 1
                for c in R.couplings + ['xi']:
                    cos_r = proj_cos(vers[(l, c)][1][j], Q)
                    for name in LING:
                        put(f'L{l}/{sname}/{name}/{c}/B_ref', bin_stats_rel(cos_r, *G[name]))
                        put(f'L{l}/{sname}/{name}/{c}/shuf/B_ref', bin_stats_rel(cos_r, *Gs[name]))
    if ((b0 - N_FIT) // 2) % 10 == 0:
        print(f'   {b0 + xb.shape[0] - N_FIT}/{N_SEQ - N_FIT} sequences, {(time.time() - t0) / 60:.1f} min', flush=True)
arrs = {k: np.asarray(v) for k, v in store.items()}
np.savez_compressed(HERE / f'sy_subspace_{ARM}.npz', **arrs)

# ---------------------------------------------------------------- summary and per-arm hits
D = lambda a, b: S.boot_contrast_diff(arrs[a], arrs[b])
say()
say(f'B_spec by subspace ({N_SEQ - N_FIT} sequences; 95% cluster-bootstrap intervals; floor {S.FLOOR}); a hit needs: passes, shuffled null, '
    f'and above the matched random subspace by the floor')
hits = []
for l in range(L):
    for c in R.couplings + ['xi']:
        for name in LING:
            row = []
            rnd = {}
            for sname in ('rand16', 'rand64', 'rand320'):
                rnd[sname] = D(f'L{l}/{sname}/{name}/B_on', f'L{l}/{sname}/{name}/{c}/B_ref')
            for sname in [s for s in SUB[l]] + ['whole']:
                bs = D(f'L{l}/{sname}/{name}/B_on', f'L{l}/{sname}/{name}/{c}/B_ref')
                bss = D(f'L{l}/{sname}/{name}/shuf/B_on', f'L{l}/{sname}/{name}/{c}/shuf/B_ref')
                mt = MATCH.get(sname)
                above = mt is not None and bs[0] - rnd[mt][0] >= S.FLOOR
                hit = sname not in ('rand16', 'rand64', 'rand320', 'whole') and bs[1] > 0 and bs[0] >= S.FLOOR and S.is_null(bss) and above
                if hit: hits.append([l, c, name, sname, bs[0]])
                row.append(f'{sname} {bs[0]:+.4f}{"*" if hit else ""}')
            say(f'   L{l} {c:<5} {name:<6} ' + '  '.join(row))
say(f'   hits in this arm (layer, coupling, grouping, subspace, B_spec): {hits if hits else "none"}')
(HERE / f'sy_subspace_{ARM}_output.txt').write_text('\n'.join(lines) + '\n')
(HERE / f'sy_subspace_{ARM}_hits.json').write_text(json.dumps(hits))
print(f'wrote sy_subspace_{ARM}.npz, _output.txt, _hits.json')
