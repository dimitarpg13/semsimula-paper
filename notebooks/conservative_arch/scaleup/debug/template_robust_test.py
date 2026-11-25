#!/usr/bin/env python3
"""The robust two-part templates: features for the firing test (2026-11-25).
    python3 template_robust_test.py OUTDIR FOLDER name=<stem> [ckpt=best|step500] [limit=N] [threads=8]
Premises up to "Therefore", twelve wordings (six conditional, six disjunctive). Development items come from the 60 clauses
that shaped the design, test items from 60 fresh ones. Saved per item and per state: the cosines between all the premise's particles
(token states, centred and whitened by ordinary text), their norms, and the connective's mean state. No token identity is
saved or used by the matcher, which sees h1 and h2; h0 is kept as a reference. Scored by template_geometry_analysis.py."""
import json, pickle, sys, time
from pathlib import Path
import numpy as np, torch
OUT = Path(sys.argv[1]).resolve(); FOLDER = sys.argv[2]; OPTS = dict(a.split('=', 1) for a in sys.argv[3:] if '=' in a)
DBG = Path(__file__).resolve().parent; sys.path.insert(0, str(DBG)); OUT.mkdir(parents=True, exist_ok=True)
from template_robust_clauses import TEST, DEV
import tiktoken
enc = tiktoken.get_encoding('gpt2'); cap = lambda s: s[0].upper() + s[1:]
pq = lambda lead: [(lead, 'c'), (" {P}", 'a1'), (",", None), (" {Q}", 'a2'), (".", None)]
W = {'cond': {'if-then': [("If", 'c'), (" {P}", 'a1'), (",", None), (" then", 'c'), (" {Q}", 'a2'), (".", None)], 'when': pq("When"), 'whenever': pq("Whenever"),
              'q-if-p': [("{Q^}", 'a2'), (" if", 'c'), (" {P}", 'a1'), (".", None)], 'assuming': pq("Assuming that"), 'provided': pq("Provided that")},
     'disj': {'either-or': [("Either", 'c'), (" {P}", 'a1'), (" or", 'c'), (" {Q}", 'a2'), (".", None)], 'or': [("{P^}", 'a1'), (" or", 'c'), (" {Q}", 'a2'), (".", None)],
              'at-least-one': [("At least one is true:", 'c'), (" {P}", 'a1'), (" or", 'c'), (" {Q}", 'a2'), (".", None)],
              'or-else': [("{P^}", 'a1'), (",", None), (" or else", 'c'), (" {Q}", 'a2'), (".", None)], 'unless': pq("Unless"),
              'one-holds': [("One of these holds:", 'c'), (" {P}", 'a1'), (" or", 'c'), (" {Q}", 'a2'), (".", None)]}}
def p2s(frame, P, Q, R):
    no = lambda x: 'it is not true that ' + x['aff']; sw = lambda x, k: f"{R['subj']} {x[k]}"
    d = dict(U=(R['aff'], ''), Un=(R['neg'], ''), near=(P['near'], ''), nearn=(P['near_neg'], ''), nearQ=(Q['near'], ''), nearQn=(Q['near_neg'], ''),
             swap=(sw(P, 'pred'), ''), swapn=(sw(P, 'npred'), ''), swapQ=(sw(Q, 'pred'), ''), swapQn=(sw(Q, 'npred'), ''))
    if frame == 'cond':
        d.update(MP=(P['aff'], 'MP'), MT=(Q['neg'], 'MT'), AC=(Q['aff'], ''), DA=(P['neg'], ''), MTnot=(no(Q), 'MT'), DAnot=(no(P), ''))
        if 'syn' in P: d.update(MPsyn=(P['syn'], 'MP'), MTsyn=(Q['syn_neg'], 'MT'), ACsyn=(Q['syn'], ''), DAsyn=(P['syn_neg'], ''))
    else:
        d.update(DS1=(P['neg'], 'DS1'), DS2=(Q['neg'], 'DS2'), P=(P['aff'], ''), Q=(Q['aff'], ''), DS1not=(no(P), 'DS1'))
        if 'syn' in P: d.update(DS1syn=(P['syn_neg'], 'DS1'), DS2syn=(Q['syn_neg'], 'DS2'), Psyn=(P['syn'], ''))
    return d
items = []; rng = np.random.default_rng(7)
for split, bank in (('dev', DEV), ('test', TEST)):
    n = len(bank); q, r = np.roll(np.arange(n), 17), np.roll(np.arange(n), 31)
    for i in range(n):
        P, Q, R = bank[i], bank[q[i]], bank[r[i]]
        for frame in W:
            for wn, pcs in W[frame].items():
                for kind, (p2, want) in p2s(frame, P, Q, R).items():
                    if split == 'dev' and (i % 2) != (list(W[frame]).index(wn) % 2) and kind not in ('MP', 'MT', 'DS1', 'DS2'):
                        continue          # development: every wording, half of the other items, to save time
                    sub = lambda s: s.replace('{P}', P['aff']).replace('{Q}', Q['aff']).replace('{P^}', cap(P['aff'])).replace('{Q^}', cap(Q['aff']))
                    segs = [(sub(t), ro) for t, ro in pcs] + [(" " + cap(p2), 'p2'), (".", None), (" Therefore", 'th')]; ids, role = [], []
                    for t, ro in segs:
                        e = enc.encode(t); ids += e; role += [ro] * len(e)
                    items.append(dict(split=split, content=i, frame=frame, wording=wn, kind=kind, want=want, ids=ids, role=role, text=''.join(t for t, _ in segs), P=P['aff'], Q=Q['aff'], p2=p2))
import verify_pm_switch as P_
val = np.load(P_.G.VAL).astype(np.int64); base = list(items)
for split, lo, hi in (('dev', 200_000, 900_000), ('test', 1_100_000, 1_900_000)):
    pool = [x for x in base if x['split'] == split]
    for i in range(600):
        it = pool[rng.integers(len(pool))]; s = int(rng.integers(lo, hi)); ids = [int(v) for v in val[s:s + len(it['ids'])]]
        dec = lambda ro: enc.decode([t for t, q_ in zip(ids, it['role']) if q_ == ro])
        items.append(dict(split=split, content=1000 + i, frame=it['frame'], wording='text', kind='text', want='', ids=ids, role=it['role'], text=enc.decode(ids), P=dec('a1'), Q=dec('a2'), p2=dec('p2')))
if 'limit' in OPTS: items = [items[i] for i in sorted(rng.choice(len(items), int(OPTS['limit']), replace=False))]
pre = [int(v) for v in val[1_000_000:1_000_048]] + enc.encode("\n\n")
src = (DBG / 'copy_split.py').read_text(); src = src[:src.index("\nif MODE == 'nll':")]
sys.argv = ['copy_split.py', str(OUT / 'harness'), 'probe', 'windows=1', f"threads={OPTS.get('threads', 8)}"]
CS = {'__name__': 'copy_split', '__file__': str(DBG / 'copy_split.py')}; exec(compile(src, 'copy_split.py', 'exec'), CS)
_b = P_.build; got = {}
def build(*a, **k):
    r = _b(*a, **k); got['m'] = r[0]; return r
P_.build = build; ck = OPTS.get('ckpt', 'best'); D = Path.home() / 'Downloads' / FOLDER / 'checkpoints'
cf = [p.name for p in D.glob('fock_cfc_*_best.pt') if ('_step' not in p.name if ck == 'best' else f'_{ck}_best' in p.name)]; assert len(cf) == 1, cf
CS['ladder_model'](FOLDER, cf[0]); m = got['m']; m.eval(); rec = {}; os_ = m._layer_step_langevin
def step(h, h_prev, m_b, gamma, dt_, layer_idx=0):
    o = os_(h, h_prev, m_b, gamma, dt_, layer_idx)
    if int(layer_idx) == 0: rec[0] = h.detach().float()
    rec[int(layer_idx) + 1] = o[0].detach().float(); return o
m._layer_step_langevin = step
H = {0: [], 1: [], 2: []}; ws = torch.tensor(np.stack([val[300_000 + 700 * i:300_000 + 700 * i + 160] for i in range(96)])); Wt = {}
for i in range(0, 96, 16):
    with torch.enable_grad(): m(ws[i:i + 16])
    for l in H: H[l].append(rec[l][:, 32:].reshape(-1, 384).numpy())
for l in H:
    x = np.concatenate(H[l]); mu = x.mean(0); e, U = np.linalg.eigh(np.cov((x - mu).T)); e = np.maximum(e, 0); Wt[l] = (mu, U / np.sqrt(e + .1 * e.mean()))
res = [None] * len(items); order = sorted(range(len(items)), key=lambda i: len(items[i]['ids'])); t0 = time.time()
un = lambda a: a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
for b in range(0, len(order), 32):
    idx = order[b:b + 32]; L = len(pre) + max(len(items[i]['ids']) for i in idx)
    x = torch.tensor([pre + items[i]['ids'] + [50256] * (L - len(pre) - len(items[i]['ids'])) for i in idx])
    with torch.enable_grad(): m(x)
    for r, i in enumerate(idx):
        it = items[i]; pos = {ro: [len(pre) + j for j, q_ in enumerate(it['role']) if q_ == ro] for ro in ('c', 'a1', 'a2', 'p2', 'th')}; o = {}
        for l in (0, 1, 2):
            s = rec[l][r].numpy(); mu, U = Wt[l]; w = {ro: (s[p] - mu) @ U for ro, p in pos.items()}
            ro_ = [ro for ro in ('c', 'a1', 'a2', 'p2', 'th') for _ in pos[ro]]; Z = np.concatenate([w[ro] for ro in ('c', 'a1', 'a2', 'p2', 'th')]); nm = np.linalg.norm(Z, axis=1); Zu = Z / (nm[:, None] + 1e-9)
            o[l] = dict(G=(Zu @ Zu.T).astype(np.float16), norm=nm.astype(np.float16), c=w['c'].mean(0).astype(np.float16))
        res[i] = dict(roles=ro_, st=o)
    if b % 3200 == 0: print(f'{b}/{len(order)} {time.time() - t0:.0f}s', flush=True)
meta = [{k: v for k, v in it.items() if k not in ('ids', 'role')} for it in items]
pickle.dump(dict(meta=meta, res=res), open(OUT / f'robust_{OPTS["name"]}.pkl', 'wb')); print('done', len(items), f'{time.time() - t0:.0f}s')
