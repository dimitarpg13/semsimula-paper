#!/usr/bin/env python3
"""Two-part templates: node states for the matcher test (2026-11-25, author's question: how accurately does it fire?).

    python3 template_matcher_test.py OUTDIR FOLDER name=<stem> [ckpt=best|step500] [threads=8] [limit=N]

Premises only, up to "Therefore". Two frames (conditional, disjunctive) in several wordings; premise 2 is one of: a clause
of premise 1 affirmed or negated, an unrelated clause, a near miss, a paraphrase, or a clause denied in another wording.
Fitting items: 140 contents from the 80 clauses of logic_figures.py, two wordings per frame. Test items: 60 contents from
the 60 clauses of template_matcher_clauses.py (none shared), in the fitting wordings and in new ones. Plus ordinary text
in the same token layouts. Saves the mean state of five nodes (connective, the two clauses by role, premise 2,
"Therefore") at h0, h1, h2. The analysis is template_matcher_analysis.py.
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
OUT = Path(sys.argv[1]).resolve(); FOLDER = sys.argv[2]; OPTS = dict(a.split('=', 1) for a in sys.argv[3:] if '=' in a)
DBG = Path(__file__).resolve().parent; sys.path.insert(0, str(DBG)); OUT.mkdir(parents=True, exist_ok=True)
from template_matcher_clauses import TEST
import tiktoken
enc = tiktoken.get_encoding('gpt2')
t_ = (DBG / 'logic_figures.py').read_text(); ns = {}; exec(t_[t_.index("C = [\n"):t_.index("cap = lambda")], ns)
FIT = [dict(aff=c[k], neg=c[k + 2]) for c in ns['C'] for k in (0, 1)]
cap = lambda s: s[0].upper() + s[1:]
# wordings of premise 1: pieces (text with {P}/{Q}, role); fit wordings first
W = {'cond': [('if-then', [("If", 'c'), (" {P}", 'a1'), (",", None), (" then", 'c'), (" {Q}", 'a2'), (".", None)], 'fit'),
              ('when', [("When", 'c'), (" {P}", 'a1'), (",", None), (" {Q}", 'a2'), (".", None)], 'fit'),
              ('whenever', [("Whenever", 'c'), (" {P}", 'a1'), (",", None), (" {Q}", 'a2'), (".", None)], 'new'),
              ('q-if-p', [("{Q^}", 'a2'), (" if", 'c'), (" {P}", 'a1'), (".", None)], 'new'),
              ('assuming', [("Assuming that", 'c'), (" {P}", 'a1'), (",", None), (" {Q}", 'a2'), (".", None)], 'new')],
     'disj': [('either-or', [("Either", 'c'), (" {P}", 'a1'), (" or", 'c'), (" {Q}", 'a2'), (".", None)], 'fit'),
              ('or', [("{P^}", 'a1'), (" or", 'c'), (" {Q}", 'a2'), (".", None)], 'fit'),
              ('at-least-one', [("At least one is true:", 'c'), (" {P}", 'a1'), (" or", 'c'), (" {Q}", 'a2'), (".", None)], 'new'),
              ('or-else', [("{P^}", 'a1'), (",", None), (" or else", 'c'), (" {Q}", 'a2'), (".", None)], 'new')]}
def premise2(kind, P, Q, R):
    """(text, template that should fire or '', denied or affirmed clause for DS)"""
    no = lambda x: 'it is not true that ' + x['aff']
    if kind[0] == 'c':
        return {'c:MP': (P['aff'], 'MP'), 'c:MT': (Q['neg'], 'MT'), 'c:AC': (Q['aff'], ''), 'c:DA': (P['neg'], ''),
                'c:unrel+': (R['aff'], ''), 'c:unrel-': (R['neg'], ''), 'c:near+': (P.get('near'), ''), 'c:near-': (Q.get('near_neg'), ''),
                'c:paraMP': (P.get('para'), 'MP'), 'c:paraMT': (Q.get('para_neg'), 'MT'), 'c:notMT': (no(Q), 'MT')}[kind]
    return {'d:DS1': (P['neg'], 'DS'), 'd:DS2': (Q['neg'], 'DS'), 'd:P': (P['aff'], ''), 'd:Q': (Q['aff'], ''),
            'd:unrel+': (R['aff'], ''), 'd:unrel-': (R['neg'], ''), 'd:near-': (P.get('near_neg'), ''), 'd:paraDS': (P.get('para_neg'), 'DS'),
            'd:notDS': (no(P), 'DS')}[kind]
KF = ['c:MP', 'c:MT', 'c:AC', 'c:DA', 'c:unrel+', 'c:unrel-', 'd:DS1', 'd:DS2', 'd:P', 'd:Q', 'd:unrel+', 'd:unrel-']
KT = KF + ['c:near+', 'c:near-', 'c:paraMP', 'c:paraMT', 'c:notMT', 'd:near-', 'd:paraDS', 'd:notDS']
rng = np.random.default_rng(20261125)
def contents(bank, n):
    out = []
    while len(out) < n:
        for i in rng.permutation(len(bank)):
            j, k = rng.choice([x for x in range(len(bank)) if x // 2 != i // 2], 2, replace=False) if bank is FIT else rng.choice([x for x in range(len(bank)) if x != i], 2, replace=False)
            out.append((int(i), int(j), int(k)))
    return out[:n]
items = []
def add(split, bank, trip, ci, frame, w, kind):
    name, pcs, wkind = w; P, Q, R = (bank[t] for t in trip); p2, want = premise2(kind, P, Q, R)
    sub = lambda s: s.replace('{P}', P['aff']).replace('{Q}', Q['aff']).replace('{P^}', cap(P['aff'])).replace('{Q^}', cap(Q['aff']))
    segs = [(sub(t), r) for t, r in pcs] + [(" " + cap(p2), 'p2'), (".", None), (" Therefore", 'th')]
    ids, role = [], []
    for t, r in segs:
        e = enc.encode(t); ids += e; role += [r] * len(e)
    items.append(dict(split=split, content=ci, frame=frame, wording=name, wkind=wkind, kind=kind, want=want, ids=ids, role=role,
                      text=''.join(t for t, _ in segs), P=P['aff'], Q=Q['aff'], p2=p2))
for split, bank, n, kinds in (('fit', FIT, 140, KF), ('test', TEST, 60, KT)):
    for ci, trip in enumerate(contents(bank, n)):
        for frame in W:
            for w in W[frame]:
                if split == 'fit' and w[2] != 'fit':
                    continue
                for kind in kinds:
                    if kind[0] == frame[0]:
                        add(split, bank, trip, ci, frame, w, kind)
import verify_pm_switch as P_
val = np.load(P_.G.VAL).astype(np.int64)
base = list(items)
for split, n in (('fit', 300), ('test', 300)):
    pool = [x for x in base if x['split'] == split]
    for i in range(n):
        it = pool[rng.integers(len(pool))]; s = int(rng.integers(200_000, 900_000) if split == 'fit' else rng.integers(1_100_000, 1_900_000))
        items.append(dict(split=split, content=1000 + i, frame=it['frame'], wording='text', wkind='text', kind='text', want='',
                          ids=[int(v) for v in val[s:s + len(it['ids'])]], role=it['role'], text='', P='', Q='', p2=''))
if 'limit' in OPTS:
    items = [items[i] for i in sorted(rng.choice(len(items), int(OPTS['limit']), replace=False))]
pre = [int(v) for v in val[1_000_000:1_000_048]] + enc.encode("\n\n")
meta = [{k: v for k, v in it.items() if k not in ('ids', 'role')} for it in items]
(OUT / 'items.json').write_text(json.dumps(meta))
# model
src = (DBG / 'copy_split.py').read_text(); src = src[:src.index("\nif MODE == 'nll':")]
sys.argv = ['copy_split.py', str(OUT / 'harness'), 'probe', 'windows=1', f"threads={OPTS.get('threads', 8)}"]
CS = {'__name__': 'copy_split', '__file__': str(DBG / 'copy_split.py')}; exec(compile(src, 'copy_split.py', 'exec'), CS)
_b = P_.build; got = {}
def build(*a, **k):
    r = _b(*a, **k); got['m'] = r[0]; return r
P_.build = build
ck = OPTS.get('ckpt', 'best'); D = Path.home() / 'Downloads' / FOLDER / 'checkpoints'
cf = [p.name for p in D.glob('fock_cfc_*_best.pt') if ('_step' not in p.name if ck == 'best' else f'_{ck}_best' in p.name)]; assert len(cf) == 1, cf
CS['ladder_model'](FOLDER, cf[0]); m = got['m']; m.eval()
rec = {}; os_ = m._layer_step_langevin
def step(h, h_prev, m_b, gamma, dt_, layer_idx=0):
    o = os_(h, h_prev, m_b, gamma, dt_, layer_idx)
    if int(layer_idx) == 0: rec[0] = h.detach().float()
    rec[int(layer_idx) + 1] = o[0].detach().float(); return o
m._layer_step_langevin = step
ROLES = ['c', 'a1', 'a2', 'p2', 'th']; N = np.zeros((len(items), 3, 5, 384), np.float16)
order = sorted(range(len(items)), key=lambda i: len(items[i]['ids'])); t0 = time.time()
for b in range(0, len(order), 32):
    idx = order[b:b + 32]; L = len(pre) + max(len(items[i]['ids']) for i in idx)
    x = torch.tensor([pre + items[i]['ids'] + [50256] * (L - len(pre) - len(items[i]['ids'])) for i in idx])
    with torch.enable_grad(): m(x)
    for r, i in enumerate(idx):
        for k, ro in enumerate(ROLES):
            p = [len(pre) + j for j, q in enumerate(items[i]['role']) if q == ro]
            for l in (0, 1, 2): N[i, l, k] = rec[l][r, p].mean(0).numpy()
    if b % 1600 == 0: print(f'{b}/{len(order)} {time.time() - t0:.0f}s', flush=True)
np.savez(OUT / f'nodes_{OPTS["name"]}.npz', N=N); print('done', len(items), f'{time.time() - t0:.0f}s')
