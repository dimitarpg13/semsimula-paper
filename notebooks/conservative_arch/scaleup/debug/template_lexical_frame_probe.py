#!/usr/bin/env python3
"""How does token identity tell the frame on a wording left out of fitting? (2026-11-26, written after template_lexical_baseline.py)
    python3 template_lexical_frame_probe.py RUNDIR OUTDIR stem wording
For the lexical reading at the m and d of its primary run, with `wording` left out of fitting: the connective coordinate of its test
premises, the threshold rho, and per kind (MP, DA, AC, the near miss) the rates of the four templates and the nearest clouds. Then the
same with the connective coordinate replaced by its development mean (no frame information), and the reading's R1-type detection."""
import json, sys
from pathlib import Path
import numpy as np
DBG = Path(__file__).resolve().parent; sys.path.insert(0, str(DBG))
W_ = sys.argv[4]; sys.argv = sys.argv[:4]
_L = (DBG / 'template_lexical_baseline.py').read_text(); _L = _L[:_L.index("res_all = {}")]; exec(_L)
s = STEMS_.split(',')[0]; J = json.loads((OUT / f'template_lexical_{s}.json').read_text())['lex']; m, d = J['m'], J['d']
M = pickle.load(open(RUN / f'depth_{s}.pkl', 'rb')); meta, res = M['meta'], M['res']; R = [x['roles'] for x in res]; del M
ids_list = [(particle_ids(it), R[i]) for i, it in enumerate(ITEMS)]
K = np.array([x['kind'] for x in meta]); want = np.array([x['want'] for x in meta], object); wi = np.array([x['wording'] for x in meta]); sp = np.array([x['split'] for x in meta]); fr = np.array([x['frame'] for x in meta]); g = np.array([x['content'] for x in meta])
lab = np.array(['text' if k == 'text' else f'{f}:{k}' for k, f in zip(K, fr)])
voc = sorted({t for (ids, ro), k_ in zip(ids_list, K) if k_ != 'text' for t, r in zip(ids, ro) if r == 'c'}); cv = {v: j for j, v in enumerate(voc)}
P = lexical(ids_list, lambda t: t, cv); WC.clear(); WC.update({k: [w for w in v if w != 'unless'] for k, v in ALLW.items()}); keep = wi != 'unless'
X = np.stack([coords(x, r, m) for x, r in zip(P, R)]); C = np.stack([x['c'] for x in P]); C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
dev = (sp == 'dev') & keep & (wi != W_); te = (sp == 'test') & (wi == W_)
F_ = Frame([C[dev]], wi[dev], fr[dev]); fc = F_([C])[:, 0]
print(f'{s} lex m{m} d{d}; wording left out: {W_}; connective coordinate on its test premises: {np.unique(np.round(fc[te], 4))}')
print('connective coordinate, development, by frame:', {f: (round(float(fc[dev & (fr == f)].mean()), 3), round(float(fc[dev & (fr == f)].std()), 3)) for f in ('cond', 'disj')})
for tag, Y in (('as fitted', np.c_[X, fc]), ('frame coordinate set to the development mean', np.c_[X, np.full(len(X), fc[dev].mean())])):
    c = Clouds(np.c_[X[dev], fc[dev]], lab[dev], want[dev], d); rho = cvf(X[dev], [C[dev]], lab[dev], want[dev], g[dev], wi[dev], fr[dev], d)[0]
    Rt, t, ins, _ = c.rates(Y); Fi = c.fire(Y, rho); Yp = ((Y - c.mu0) / c.sd) @ c.B; Dd = ((Yp[:, None] - c.C[None]) ** 2).sum(2)
    print(f'\n  {tag}: rho {rho:.3f}')
    for k_ in ('MP', 'MT', 'DA', 'AC', 'near'):
        q = te & (K == k_)
        if not q.any(): continue
        near2 = [', '.join(f'{c.cl[j]} {Dd[i, j]:.1f}' for j in np.argsort(Dd[i])[:3]) for i in np.flatnonzero(q)[:2]]
        print(f'    {k_:5s} n {q.sum():4d} fires: ' + ', '.join(f'{x or "none"} {n_}' for x, n_ in zip(*np.unique(Fi[q].astype(str), return_counts=True))) + f' | mean template rates MP {Rt[q, 0].mean():.2f} MT {Rt[q, 1].mean():.2f} DS1 {Rt[q, 2].mean():.2f} DS2 {Rt[q, 3].mean():.2f} | nearest clouds of two premises: {near2}')
# which clauses does token identity miss? (known wordings, all development premises fitted)
from template_fresh_clauses import FRESH
infl = {i for i, P_c in enumerate(FRESH) if not set(enc.encode(' ' + P_c['aff'])) <= set(enc.encode(' ' + P_c['neg']))}
devall = (sp == 'dev') & keep; Fa = Frame([C[devall]], wi[devall], fr[devall]); ca = Clouds(np.c_[X[devall], Fa([C[devall]])], lab[devall], want[devall], d)
rho_a = cvf(X[devall], [C[devall]], lab[devall], want[devall], g[devall], wi[devall], fr[devall], d)[0]; Fk = ca.fire(np.c_[X, Fa([C])], rho_a); tk = (sp == 'test') & keep
for k_, f_ in (('MT', 'cond'), ('DS1', 'disj'), ('DS2', 'disj')):
    q = tk & (K == k_) & (fr == f_); miss = sorted({int(x) for x in g[q & (Fk != k_)]}); src = 'P' if k_ == 'DS1' else 'Q'
    # DS2 and MT deny clause 2 (Q = bank[q[i]]), DS1 clause 1 (P = bank[i]); map the content index to the denied clause's bank index
    n = len(FRESH); qi = np.roll(np.arange(n), 17); denied = {int(qi[c]) if src == 'Q' else c for c in miss}
    print(f'{k_}: missed contents {len(miss)} of {len(set(g[q]))}; the denied clause is inflected for {len(denied & infl)} of them; inflected clauses {len(infl)}')
