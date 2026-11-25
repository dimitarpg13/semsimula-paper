#!/usr/bin/env python3
"""How accurately do the two-part templates fire? (2026-11-25; written before any result was seen)

    python3 template_matcher_analysis.py RUNDIR OUTDIR stem[,stem...]

Gates (a template fires when all three pass):
  frame     the distances between connective, clause 1, clause 2 and "Therefore" (unit-size figure, 6 numbers) against the
            frozen mean of the frame, each distance scaled by its spread; variant 'state': the connective's own state,
            cosine to the frame's centroid minus cosine to the other frame's
  binding   cosine distance of premise 2 to the other clause minus to the required one; DS: the absolute difference, the
            nearer clause being the denied one; variant 'close' also requires premise 2 close to the required clause
  polarity  (premise 2 - required clause) along the frozen negation direction, signed
States are centred on the mean fitting node. Thresholds: every gate at the same quantile alpha of the template's own fitting
positives, alpha the smallest with a joint false-fire rate of at most 1% on the fitting negatives (mean over their types).
Nothing from the test clauses or the new wordings enters. References: the same on h0; step-500 checkpoints; a string
matcher (frame words of the fitting wordings; content-word overlap >= 0.5 and larger than with the other clause; "not"/"no").
"""
import json, sys
from pathlib import Path
import numpy as np
RUN, OUT = Path(sys.argv[1]), Path(sys.argv[2]); STEMS = sys.argv[3].split(','); OUT.mkdir(parents=True, exist_ok=True)
I = json.loads((RUN / 'items.json').read_text()); n = len(I); A = lambda k: np.array([x[k] for x in I])
split, frame, want, kind, wk, cont = A('split'), A('frame'), A('want'), A('kind'), A('wkind'), A('content')
TP = {'MP': ('cond', 'a1', +1), 'MT': ('cond', 'a2', -1), 'DS': ('disj', None, -1)}
NEG_P2 = np.isin(kind, ['c:MT', 'c:DA', 'c:unrel-', 'd:DS1', 'd:DS2', 'd:unrel-']); FITT = split == 'fit'
def figure(X):
    Y = X[:, [0, 1, 2, 4]]; Y = Y - Y.mean(1, keepdims=True); Y = Y / np.linalg.norm(Y, axis=(1, 2), keepdims=True)
    i, j = np.triu_indices(4, 1); return np.linalg.norm(Y[:, i] - Y[:, j], axis=-1)
cos = lambda a, b: (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)
def freeze(X):
    p = {}; f = FITT & (kind != 'text'); p['mu'] = X[f].reshape(-1, 384).mean(0); Z = X - p['mu']; D = figure(X)
    for c in ('cond', 'disj'):
        k = f & (frame == c); p['fig_' + c] = (D[k].mean(0), D[k].std(0) + 1e-6); p['con_' + c] = Z[k, 0].mean(0)
    same = {'c:MP': 1, 'c:MT': 2, 'c:AC': 2, 'c:DA': 1, 'd:DS1': 1, 'd:DS2': 2, 'd:P': 1, 'd:Q': 2}
    dif = np.stack([Z[i, 3] - Z[i, same[kind[i]]] if kind[i] in same else np.zeros(384) for i in range(n)])
    ok = f & np.isin(kind, list(same)); p['w'] = dif[ok & NEG_P2].mean(0) - dif[ok & ~NEG_P2].mean(0); p['w'] /= np.linalg.norm(p['w']); return p
def scores(X, p):
    Z = X - p['mu']; D = figure(X); d1, d2 = 1 - cos(Z[:, 3], Z[:, 1]), 1 - cos(Z[:, 3], Z[:, 2]); S = {}
    for t, (c, req, sg) in TP.items():
        o = 'disj' if c == 'cond' else 'cond'; mu, sd = p['fig_' + c]
        first = d1 <= d2 if req is None else np.full(n, req == 'a1')
        dr, do = np.where(first, d1, d2), np.where(first, d2, d1); ra = np.where(first[:, None], Z[:, 1], Z[:, 2])
        S[t] = dict(frame=-np.linalg.norm((D - mu) / sd, axis=1), state=cos(Z[:, 0], p['con_' + c]) - cos(Z[:, 0], p['con_' + o]),
                    bind=do - dr, close=-dr, pol=-sg * ((Z[:, 3] - ra) @ p['w']), first=first)
    return S
NTYPE = [k for k in ('c:MP', 'c:MT', 'c:AC', 'c:DA', 'c:unrel+', 'c:unrel-', 'd:DS1', 'd:DS2', 'd:P', 'd:Q', 'd:unrel+', 'd:unrel-', 'text')]
def calibrate(S, gates):
    T = {}
    for t in TP:
        pos = FITT & (want == t); best = None
        for a in np.r_[np.arange(0, .1, .005), np.arange(.1, .96, .02)]:
            th = {g: np.quantile(S[t][g][pos], a) for g in gates}
            fire = np.all([S[t][g] >= th[g] for g in gates], axis=0)
            ff = np.mean([fire[FITT & (kind == k)].mean() for k in NTYPE if not (want[FITT & (kind == k)] == t).any()])
            if best is None or ff < best[1]: best = (a, ff, th)
            if ff <= .01: best = (a, ff, th); break
        T[t] = best
    return T
def fires(S, T, gates):
    return {t: np.all([S[t][g] >= T[t][2][g] for g in gates], axis=0) for t in TP}
CELLS = {'new contents, known wordings': (split == 'test') & (wk == 'fit'), 'new contents, new wordings': (split == 'test') & (wk == 'new')}
def table(F, S=None):
    R = {}
    for cn, m in CELLS.items():
        r = R[cn] = {}
        for nm, ks in (('exact', ['c:MP', 'c:MT', 'd:DS1', 'd:DS2']), ('paraphrase', ['c:paraMP', 'c:paraMT', 'd:paraDS']), ('denied otherwise', ['c:notMT', 'd:notDS'])):
            for t in TP:
                k = m & np.isin(kind, ks) & (want == t)
                if k.any(): r[f'detect {t} {nm}'] = float(F[t][k].mean())
        anyf = F['MP'] | F['MT'] | F['DS']
        for k in ('c:AC', 'c:DA', 'd:P', 'd:Q', 'c:unrel+', 'c:unrel-', 'd:unrel+', 'd:unrel-', 'c:near+', 'c:near-', 'd:near-'):
            r[f'false {k}'] = float(anyf[m & (kind == k)].mean())
        pos = m & (want != ''); wrong = np.any([F[t] & (want != t) for t in TP], axis=0); r['wrong template on a valid argument'] = float(wrong[pos].mean())
        r['two templates at once'] = float(((F['MP'].astype(int) + F['MT'] + F['DS']) > 1)[m].mean())
        if S is not None:
            k = m & np.isin(kind, ['d:DS1', 'd:DS2']) & F['DS']
            r['DS names the right clause'] = float((S['DS']['first'] == (kind == 'd:DS1'))[k].mean()) if k.any() else float('nan')
    R['ordinary text'] = {t: float(F[t][(split == 'test') & (kind == 'text')].mean()) for t in TP}
    return R
def auroc(a, b):
    r = np.argsort(np.argsort(np.r_[a, b])); return float((r[:len(a)].sum() - len(a) * (len(a) - 1) / 2) / (len(a) * len(b)))
# string matcher
STOP = set('the a an is are was not does do did has have had been it that no to of in on for and or if then when either at with too'.split())
words = lambda s: {w.strip('.,').lower() for w in s.split()} - STOP
stem = lambda W: {w[:-1] if w.endswith('s') and len(w) > 3 else w for w in W}
def string():
    ov = lambda a, b: len(stem(words(a)) & stem(words(b))) / max(1, len(stem(words(a)) | stem(words(b))))
    o1, o2 = np.array([ov(x['p2'], x['P']) for x in I]), np.array([ov(x['p2'], x['Q']) for x in I])
    pos = lambda t: [set(x['text'].split('.')[0].lower().replace(',', '').split()) for x in I][t]
    fw = [pos(i) for i in range(n)]; cond = np.array([bool(f & {'if', 'when', 'then'}) for f in fw]); disj = np.array([bool(f & {'or', 'either'}) for f in fw])
    ng = np.array([bool({'not', 'no'} & set(x['p2'].lower().split())) for x in I]); t = kind != 'text'
    return ({'MP': t & cond & ~disj & (o1 >= .5) & (o1 > o2) & ~ng, 'MT': t & cond & ~disj & (o2 >= .5) & (o2 > o1) & ng,
             'DS': t & disj & (np.maximum(o1, o2) >= .5) & (o1 != o2) & ng}, o1, o2)
res = {}; SF, o1, o2 = string(); res['string'] = table(SF)
mt = split == 'test'
def decisive(f1, f2, m):
    """para against near miss: premise 2's closeness to the clause it should (or only seems to) restate"""
    out = {}
    for nm, a, b, f in (('MP', 'c:paraMP', 'c:near+', f1), ('MT', 'c:paraMT', 'c:near-', f2), ('DS', 'd:paraDS', 'd:near-', f1)):
        out[nm] = auroc(f[m & (kind == a)], f[m & (kind == b)])
    for nm, a, b, f in (('MP', 'c:paraMP', 'c:unrel+', f1), ('MT', 'c:paraMT', 'c:unrel-', f2), ('DS', 'd:paraDS', 'd:unrel-', f1)):
        out[nm + ' against unrelated'] = auroc(f[m & (kind == a)], f[m & (kind == b)])
    return out
res['string']['decisive'] = decisive(o1, o2, mt)
V = {'as designed': ('frame', 'bind', 'pol'), 'with closeness': ('frame', 'bind', 'close', 'pol'), 'connective state': ('state', 'bind', 'close', 'pol')}
for s in STEMS:
    N = np.load(RUN / f'nodes_{s}.npz')['N'].astype(np.float32); assert len(N) == n
    for l in (0, 1, 2):
        X = N[:, l]; p = freeze(X); S = scores(X, p); q = res[f'{s} h{l}'] = {}
        for v, g in V.items():
            T = calibrate(S, g); F = fires(S, T, g); q[v] = table(F, S)
            q[v]['alpha, fitting false fires'] = {t: (float(T[t][0]), float(T[t][1])) for t in TP}
            q[v]['gates alone'] = {t: {gg: {k: float((S[t][gg] >= T[t][2][gg])[CELLS['new contents, known wordings'] & (kind == k)].mean())
                                  for k in ('c:MP', 'c:MT', 'c:AC', 'c:DA', 'c:unrel+', 'c:near+', 'c:paraMP', 'c:paraMT', 'd:DS1', 'd:P', 'd:near-', 'd:paraDS')} for gg in g} for t in TP}
        Z = X - p['mu']; q['decisive'] = decisive(cos(Z[:, 3], Z[:, 1]), cos(Z[:, 3], Z[:, 2]), mt)
(OUT / 'template_matcher.json').write_text(json.dumps(res, indent=1))
def show(nm, R):
    print('==', nm)
    for c in CELLS:
        r = R[c]; print('  ', c); print('     detect  ' + ' '.join(f"{k[7:]} {v:.2f};" for k, v in r.items() if k.startswith('detect')))
        print('     false   ' + ' '.join(f"{k[6:]} {v:.2f};" for k, v in r.items() if k.startswith('false')), f"| wrong {r['wrong template on a valid argument']:.2f} double {r['two templates at once']:.2f} DSclause {r.get('DS names the right clause', float('nan')):.2f}")
    print('   text', R['ordinary text'], R.get('alpha, fitting false fires', ''))
show('string matcher', res['string']); print('   AUROC', {k: round(v, 2) for k, v in res['string']['decisive'].items()})
for s in STEMS:
    for l in (0, 1, 2):
        print('######', s, f'h{l}', 'AUROC para/near', {k: round(v, 2) for k, v in res[f'{s} h{l}']['decisive'].items()})
        for v in V: show(f'{s} h{l} {v}', res[f'{s} h{l}'][v])
