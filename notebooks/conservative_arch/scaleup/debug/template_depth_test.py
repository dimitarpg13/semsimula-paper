#!/usr/bin/env python3
"""Template matchers on the dynamics at any depth, and how close each layer is to its own flow (2026-11-25).
    python3 template_depth_test.py OUTDIR FOLDER name=<stem> [ckpt=best|step500] [limit=N] [threads=8]
Premises as template_dynamics_test.py (the 60 fresh test clauses, the old 60 for development). For every vector kind the
cosines and norms between the whitened particles and the connective's mean vector are saved, not the states:
  h<l>      the state that enters layer l (l = 0..L; h<L> is the state that leaves the last layer, after the reverse channel if any)
  D<l>      the conservative step's displacement: state before the layer-end LayerNorm minus the state that enters
  W<l>      the well-frame vector of the layer's bank (shares, radii in and their change, stiff coordinates in and their change)
Whitening by 12,288 ordinary-text tokens per kind. Scored by template_depth_analysis.py.

flow_<stem>.json: each layer step replayed from its captured inputs (ordinary text and premises) with k = 1, 2, 4, 8, 16 substeps
of dt/k (context, low-rank quadratic and wells frozen at the first substep, one LayerNorm at the end: the per-layer flow of
protocol SS5.19). gap_k = |pre_k - pre_16| / |pre_16 - h_in|, median over tokens: k = 1 is the trained step. Also the cosine of the
trained step's displacement with the flow's, the size of the displacements along depth, and the cosine of successive ones. For a
model with a reverse channel only the conservative step is replayed."""
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
cur = {}; olr = bank.harmonic_terms_lowrank; L_ = m.cfg.L
def lr(xi, h, *, comps=None):
    if cur.get('l') is not None and cur['l'] not in cur['done']:          # the first evaluation of the layer: at the state that enters it
        cur['done'].add(cur['l'])
        with torch.no_grad(): rec['c', cur['l']] = [v.detach().float() for v in (comps if comps is not None else bank._components(xi))]
    return olr(xi, h, comps=comps)
bank.harmonic_terms_lowrank = lr
keep = {'args': None}
def step(h, h_prev, m_b, gamma, dt_, layer_idx=0):
    l = int(layer_idx); cur['l'] = l
    if l == 0: cur['done'] = set()
    rec['in', l] = h.detach().float()
    if keep['args'] is not None: keep['args'][l] = (h.detach(), h_prev.detach(), m_b, gamma, dt_)
    o = os_(h, h_prev, m_b, gamma, dt_, layer_idx); cur['l'] = None; return o
m._layer_step_langevin = step; opj = m._project
def proj(h):
    if cur.get('l') is not None: rec['pre', cur['l']] = h.detach().float(); rec['pn', cur['l']] = rec.get(('pn', cur['l']), 0) + 1
    if keep.get('replay') is not None: keep['replay'] = h.detach().float()
    return opj(h)
m._project = proj; osf = m._stack_forward
def sf(h0, x, return_trajectory=False):
    for l in range(L_): rec['pn', l] = 0
    o = osf(h0, x, return_trajectory); rec['in', L_] = o[0].detach().float(); return o
m._stack_forward = sf; KINDS = [f'h{l}' for l in range(L_ + 1)] + [f'D{l}' for l in range(L_)] + [f'W{l}' for l in range(L_)]
def vecs(pos):
    """kind -> (particles, dim) float32 at the (b, t) index arrays, for the sequences just run"""
    assert all(rec['pn', l] == 1 for l in range(L_)), [rec['pn', l] for l in range(L_)]; o = {f'h{l}': rec['in', l][pos].numpy() for l in range(L_ + 1)}
    for l in range(L_):
        h, p = rec['in', l][pos], rec['pre', l][pos]; o[f'D{l}'] = (p - h).numpy(); mu, a, w, B = (v[pos] for v in rec['c', l]); out = []
        for st in (h, p):
            d = st.unsqueeze(1) - mu; u = torch.einsum('nkd,nkdr->nkr', d, B); out.append(((a * d * d).sum(-1) + (u * u).sum(-1), u.flatten(1)))
        (qi, ui), (qo, uo) = out; lg = torch.log(w.clamp_min(1e-12)) - .5 * qi; lp = lg - torch.logsumexp(lg, 1, keepdim=True); ri, ro = qi.sqrt(), qo.sqrt()
        o[f'W{l}'] = torch.cat([lp.clamp_min(-20), ri, ro - ri, ui, uo - ui], 1).numpy()
    return o
def replay(l, k):
    h, hp, mb, ga, dt_ = keep['args'][l]
    if hasattr(m.V_theta, 'set_active_layer'): m.V_theta.set_active_layer(l)
    m._flow_ctx = {} if k > 1 else None
    try:
        for j in range(k):
            if k > 1: m._flow_ctx['project'] = j == k - 1
            with torch.enable_grad(): keep['replay'] = None if j < k - 1 else True; o = os_(h, hp, mb, ga, dt_ / k, l)
            h, hp = o[0].detach(), o[1].detach()
    finally: m._flow_ctx = None
    r, keep['replay'] = keep['replay'], None; return r
def flow_rows(mask_pos):
    rows = []
    for l in range(L_):
        h = keep['args'][l][0].float(); pre = {k: replay(l, k) for k in (1, 2, 4, 8, 16)}; ref = pre[16]; dmax = float((pre[1] - rec['pre', l]).abs().max()); assert dmax < 1e-4, (l, dmax); D16 = (ref - h)[mask_pos]; D1 = (pre[1] - h)[mask_pos]; nD = D16.norm(dim=1)
        rows.append(dict(layer=l, gap={k: float(((pre[k][mask_pos] - ref[mask_pos]).norm(dim=1) / nD).median()) for k in (1, 2, 4, 8)}, cos=float((D1 * D16).sum(1).div(D1.norm(dim=1) * nD).median()), disp=float(D1.norm(dim=1).median()), state=float(h[mask_pos].norm(dim=1).median())))
    return rows
ws = torch.tensor(np.stack([val[300_000 + 700 * i:300_000 + 700 * i + 160] for i in range(96)])); TV = {k: [] for k in KINDS}; flow = {}; jump = []
for i in range(0, 96, 16):
    if i == 0: keep['args'] = {}
    with torch.enable_grad(): m(ws[i:i + 16])
    b, t_ = np.meshgrid(np.arange(16), np.arange(32, 160), indexing='ij'); pos = (torch.tensor(b.ravel()), torch.tensor(t_.ravel())); v = vecs(pos)
    for k in KINDS: TV[k].append(v[k])
    if i == 0: flow['text'] = flow_rows(pos); keep['args'] = None
W = {}
for k in KINDS:
    x = np.concatenate(TV[k]); mu = x.mean(0); e, U = np.linalg.eigh(np.cov((x - mu).T)); e = np.maximum(e, 0); W[k] = (mu, U / np.sqrt(e + .1 * e.mean()))
D_ = [np.concatenate(TV[f'D{l}']) for l in range(L_)]; n = lambda a: np.linalg.norm(a, axis=1); flow['cos successive D (text)'] = [float(np.median((D_[l] * D_[l + 1]).sum(1) / (n(D_[l]) * n(D_[l + 1])))) for l in range(L_ - 1)]; flow['|D| (text)'] = [float(np.median(n(d))) for d in D_]
res = [None] * len(items); order = sorted(range(len(items)), key=lambda i: len(items[i]['ids'])); t0 = time.time()
for b in range(0, len(order), 32):
    idx = order[b:b + 32]; Ln = len(pre) + max(len(items[i]['ids']) for i in idx); x = torch.tensor([pre + items[i]['ids'] + [50256] * (Ln - len(pre) - len(items[i]['ids'])) for i in idx])
    if b == 0 and len(order) >= 32: keep['args'] = {}
    with torch.enable_grad(): m(x)
    bi, ti, cut = [], [], [0]
    for r, i in enumerate(idx):
        it = items[i]; pp = [len(pre) + j for ro in ('c', 'a1', 'a2', 'p2', 'th') for j, q_ in enumerate(it['role']) if q_ == ro]; bi += [r] * len(pp); ti += pp; cut.append(len(ti))
    pos = (torch.tensor(bi), torch.tensor(ti)); V = vecs(pos)
    if b == 0 and keep['args'] is not None: flow['premises'] = flow_rows(pos); keep['args'] = None
    Z = {k: (V[k] - W[k][0]) @ W[k][1] for k in KINDS}
    for r, i in enumerate(idx):
        it = items[i]; a, e_ = cut[r], cut[r + 1]; ro_ = [ro for ro in ('c', 'a1', 'a2', 'p2', 'th') for q_ in it['role'] if q_ == ro]; c = np.array([q == 'c' for q in ro_]); st = {}
        for k in KINDS:
            z = Z[k][a:e_]; nm = n(z); zu = z / (nm[:, None] + 1e-9); st[k] = dict(G=(zu @ zu.T).astype(np.float16), norm=nm.astype(np.float16), c=z[c].mean(0).astype(np.float16))
        res[i] = dict(roles=ro_, st=st)
    if b % 1600 == 0: print(f'{b}/{len(order)} {time.time() - t0:.0f}s', flush=True)
meta = [{k: v for k, v in it.items() if k not in ('ids', 'role')} for it in items]
(OUT / f'flow_{OPTS["name"]}.json').write_text(json.dumps(flow, indent=1)); pickle.dump(dict(meta=meta, res=res, L=L_, kinds=KINDS), open(OUT / f'depth_{OPTS["name"]}.pkl', 'wb')); print('done', len(items), f'{time.time() - t0:.0f}s')
