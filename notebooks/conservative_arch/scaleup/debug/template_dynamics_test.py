#!/usr/bin/env python3
"""The template matchers on the dynamics: what the particles of a premise did (2026-11-25).
    python3 template_dynamics_test.py OUTDIR FOLDER name=<stem> [ckpt=best|step500] [limit=N] [threads=8]
Same premises as template_robust_test.py; development items as there, test items from 60 clauses seen by no earlier matcher.
Saved per premise particle: the raw states h0 (enters layer 0), pre1 (leaves layer 0, before its LayerNorm), h1, pre2, h2, and per layer, for each well of the bank at the
context the layer used: weight w, squared radius q and stiff coordinates u = B^T (h - mu), at the state that enters the layer,
that leaves it before the LayerNorm (u too) and after it (q only). Scored by template_dynamics_analysis.py. Also a short settling table on ordinary text."""
import json, pickle, sys, time
from pathlib import Path
import numpy as np, torch
OUT = Path(sys.argv[1]).resolve(); FOLDER = sys.argv[2]; OPTS = dict(a.split('=', 1) for a in sys.argv[3:] if '=' in a)
DBG = Path(__file__).resolve().parent; sys.path.insert(0, str(DBG)); OUT.mkdir(parents=True, exist_ok=True)
from template_robust_clauses import DEV
from template_fresh_clauses import FRESH as TEST
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
items = []; rng = np.random.default_rng(11)
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
CS['ladder_model'](FOLDER, cf[0]); m = got['m']; m.eval(); rec = {}; os_ = m._layer_step_langevin; bank = m.V_theta.bank.bank if hasattr(m.V_theta.bank, 'bank') else m.V_theta.bank
cur = {}; olr = bank.harmonic_terms_lowrank
def lr(xi, h, *, comps=None):
    if cur.get('l') is not None and cur['l'] not in cur['done']:          # the first evaluation of the layer: at the state that enters it
        cur['done'].add(cur['l'])
        with torch.no_grad(): rec['c', cur['l']] = [v.detach().float() for v in (comps if comps is not None else bank._components(xi))]
    return olr(xi, h, comps=comps)
bank.harmonic_terms_lowrank = lr
def step(h, h_prev, m_b, gamma, dt_, layer_idx=0):
    l = int(layer_idx); cur.update(l=l); cur.setdefault('done', set())
    if l == 0: cur['done'] = set(); rec['in', 0] = h.detach().float(); rec['pn', 0] = rec['pn', 1] = 0
    else: rec['in', l] = h.detach().float()
    o = os_(h, h_prev, m_b, gamma, dt_, layer_idx); rec['out', l] = o[0].detach().float(); cur['l'] = None; return o
m._layer_step_langevin = step; opj = m._project
def proj(h):
    if cur.get('l') is not None: rec['pre', cur['l']] = h.detach().float(); rec['pn', cur['l']] = rec.get(('pn', cur['l']), 0) + 1
    return opj(h)
m._project = proj
def wells(l, st, pos):
    """w (K), q (K), u (K, r) of the layer's wells at the states st[pos] of one sequence (pos: (b, t) index arrays)."""
    mu, a, w, B = (v[pos] for v in rec['c', l]); d = st[pos].unsqueeze(1) - mu; u = torch.einsum('nkd,nkdr->nkr', d, B)
    return w, (a * d * d).sum(-1) + (u * u).sum(-1), u
def pack(pos):
    """(particles, 5*384 states h0, pre1, h1, pre2, h2: pre = before the layer-end LayerNorm; per layer: w 8, q in 8, u in 32,
    q pre 8, u pre 32, q out 8) as two float32 arrays."""
    assert rec['pn', 0] == rec['pn', 1] == 1, (rec['pn', 0], rec['pn', 1])
    S = torch.cat([rec['in', 0][pos], rec['pre', 0][pos], rec['out', 0][pos], rec['pre', 1][pos], rec['out', 1][pos]], 1); out = []
    for l in (0, 1):
        w, qi, ui = wells(l, rec['in', l], pos); _, qo, uo = wells(l, rec['pre', l], pos); _, qp, _ = wells(l, rec['out', l], pos); out += [w, qi, ui.flatten(1), qo, uo.flatten(1), qp]
    return S.numpy(), torch.cat(out, 1).numpy()
# ordinary text: the states and well readings at 96 windows of 160, for whitening and the settling table
ws = torch.tensor(np.stack([val[300_000 + 700 * i:300_000 + 700 * i + 160] for i in range(96)])); TS, TW, sj = [], [], []
for i in range(0, 96, 16):
    with torch.enable_grad(): m(ws[i:i + 16])
    b, t_ = np.meshgrid(np.arange(16), np.arange(32, 160), indexing='ij'); pos = (torch.tensor(b.ravel()), torch.tensor(t_.ravel())); sj.append(float((rec['out', 0] - rec['in', 1]).abs().max())); S, Wl = pack(pos); TS.append(S); TW.append(Wl)
print('h1 is layer 1 input: max |out0 - in1| =', max(sj)); np.savez(OUT / f'dyntext_{OPTS["name"]}.npz', S=np.concatenate(TS), W=np.concatenate(TW))
offs, Ss, Ws, roles = [0] * (len(items) + 1), [None] * len(items), [None] * len(items), [None] * len(items); order = sorted(range(len(items)), key=lambda i: len(items[i]['ids'])); t0 = time.time()
for b in range(0, len(order), 32):
    idx = order[b:b + 32]; L = len(pre) + max(len(items[i]['ids']) for i in idx); x = torch.tensor([pre + items[i]['ids'] + [50256] * (L - len(pre) - len(items[i]['ids'])) for i in idx])
    with torch.enable_grad(): m(x)
    for r, i in enumerate(idx):
        it = items[i]; ro_ = [ro for ro in ('c', 'a1', 'a2', 'p2', 'th') for q_ in it['role'] if q_ == ro]; pp = [len(pre) + j for ro in ('c', 'a1', 'a2', 'p2', 'th') for j, q_ in enumerate(it['role']) if q_ == ro]
        S, Wl = pack((torch.full((len(pp),), r), torch.tensor(pp))); Ss[i], Ws[i], roles[i] = S, Wl, ro_
    if b % 3200 == 0: print(f'{b}/{len(order)} {time.time() - t0:.0f}s', flush=True)
meta = [{k: v for k, v in it.items() if k not in ('ids', 'role')} for it in items]; n = [len(s) for s in Ss]; off = np.r_[0, np.cumsum(n)]
np.save(OUT / f'dynS_{OPTS["name"]}.npy', np.concatenate(Ss)); np.save(OUT / f'dynW_{OPTS["name"]}.npy', np.concatenate(Ws)); pickle.dump(dict(meta=meta, roles=roles, off=off), open(OUT / f'dynmeta_{OPTS["name"]}.pkl', 'wb')); print('done', len(items), f'{time.time() - t0:.0f}s')
