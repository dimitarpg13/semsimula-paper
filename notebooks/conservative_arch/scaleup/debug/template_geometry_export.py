#!/usr/bin/env python3
"""The fitted template clouds written out (2026-11-25): the explicit expressions and the subspace that encloses them.
    python3 template_geometry_export.py RUNDIR OUTDIR stem
Clouds of the geometry matcher with the connective's position (template_geometry_frame.py), states h1 and h2 together, m = 4,
d = 10, fitted on all development rows (the known-wording fit). Writes OUTDIR/template_cloud_<stem>.json: the standardisation and
the projection, class means, 99% reach, firing threshold, the leading loadings of every direction, pair distances and overlaps,
the chi-squared check and the projected test points of the main kinds."""
import sys
from pathlib import Path
_A = Path(__file__).with_name('template_geometry_analysis.py').read_text(); _A = _A[:_A.index("res = {}; np.random.seed(0)")]; exec(_A)
_F = Path(__file__).with_name('template_geometry_frame.py').read_text(); exec(_F[_F.index("class Frame:"):_F.index("\nout = {}\nfor s in STEMS")])
from scipy import stats
s = STEMS[0]; m_, d_, lv = 4, 10, (1, 2)
D_ = pickle.load(open(RUN / f'robust_{s}.pkl', 'rb')); M, Rr = D_['meta'], D_['res']; K = np.array([x['kind'] for x in M]); want = np.array([x['want'] for x in M], object); wi = np.array([x['wording'] for x in M]); sp = np.array([x['split'] for x in M]); fr = np.array([x['frame'] for x in M]); g = np.array([x['content'] for x in M])
lab = np.array(['text' if k == 'text' else f'{f}:{k}' for k, f in zip(K, fr)]); dev, te = sp == 'dev', sp == 'test'
X = np.concatenate([np.stack([coords(r['st'][l], r['roles'], m_) for r in Rr]) for l in lv], 1); Cs = []
for l in lv: C = np.stack([r['st'][l]['c'].astype(np.float64) for r in Rr]); Cs.append(C / np.linalg.norm(C, axis=1, keepdims=True))
F_ = Frame([C[dev] for C in Cs], wi[dev], fr[dev]); XF = np.c_[X, F_(Cs)]; c = Clouds(XF[dev], lab[dev], want[dev], d_); rho = cvf(X[dev], [C[dev] for C in Cs], lab[dev], want[dev], g[dev], wi[dev], fr[dev], d_)[0]
Y = ((XF - c.mu0) / c.sd) @ c.B; nms = sum([[f'h{l} {x}' for x in names(m_)] for l in lv], []) + [f'h{l} connective position (or to if)' for l in lv]
ids = {k: i for i, k in enumerate(c.cl)}; out = dict(stem=s, m=m_, d=d_, rho=rho, classes={k: dict(template=c.tpl[k], mean=c.C[ids[k]].tolist(), reach=float(c.reach[k])) for k in c.cl})
corr = np.array([[np.corrcoef(XF[dev][:, j], Y[dev][:, i])[0, 1] for j in range(XF.shape[1])] for i in range(d_)]); out['loadings'] = [[(nms[j], float(corr[i, j])) for j in np.argsort(-abs(corr[i]))[:4]] for i in range(d_)]; out['singular'] = c.sv[:d_].tolist(); out['coordinates'] = XF.shape[1]
main = ['cond:MP', 'cond:MT', 'disj:DS1', 'disj:DS2', 'cond:DA', 'cond:AC', 'disj:P', 'disj:Q', 'cond:near', 'cond:swap', 'cond:U', 'text']; D = np.array([[np.linalg.norm(c.C[ids[a]] - c.C[ids[b]]) for b in main] for a in main]); out['main'] = main; out['dist'] = D.tolist(); out['bc'] = np.exp(-D ** 2 / 8).tolist()
pr = []
for a, b in (('cond:MP', 'cond:DA'), ('cond:MP', 'disj:P'), ('cond:MP', 'cond:near'), ('cond:MT', 'cond:AC'), ('cond:MT', 'cond:nearQn'), ('cond:MT', 'cond:swapQn'), ('disj:DS1', 'disj:P'), ('disj:DS1', 'disj:nearn'), ('disj:DS2', 'disj:Q'), ('cond:MP', 'disj:DS1'), ('disj:DS1', 'disj:DS2'), ('cond:MT', 'disj:DS1')):
    ma, mb = c.C[ids[a]], c.C[ids[b]]; w = ma - mb; dl = np.linalg.norm(w); t_ = (ma @ ma - mb @ mb) / 2; ea, eb = te & (lab == a), te & (lab == b); pr.append((a, b, float(dl), float(stats.norm.cdf(-dl / 2)), float(.5 * ((Y[ea] @ w < t_).mean() + (Y[eb] @ w >= t_).mean()))))
out['pairs'] = pr; q = {}
for k in ('cond:MP', 'cond:MT', 'disj:DS1', 'disj:DS2', 'cond:MTnot', 'disj:DS1not'):
    v = ((Y[te & (lab == k)] - c.C[ids[k]]) ** 2).sum(1); q[k] = dict(mean=float(v.mean()), median=float(np.median(v)), beyond99=float((v > stats.chi2.ppf(.99, d_)).mean()), q=np.quantile(v, [.1, .25, .5, .75, .9, .99]).tolist())
out['chi2'] = q; W = np.concatenate([Y[te & (lab == k)] - c.C[ids[k]] for k in ('cond:MP', 'cond:MT', 'disj:DS1', 'disj:DS2')]); ev = np.linalg.eigvalsh(np.cov(W.T)); out['within-cloud covariance eigenvalues, test'] = ev.tolist()
rng = np.random.default_rng(0); out['points'] = {k: Y[te & (lab == k)][rng.permutation(int((te & (lab == k)).sum()))[:120]].tolist() for k in ('cond:MP', 'cond:MT', 'disj:DS1', 'disj:DS2', 'cond:DA', 'disj:P', 'cond:near', 'text')}
np.savez(OUT / f'template_cloud_{s}.npz', mu0=c.mu0, sd=c.sd, B=c.B, C=c.C, classes=np.array(c.cl), axis=np.array([a for a, _ in F_.ax]), axis_mid=np.array([m for _, m in F_.ax]), names=np.array(nms)); (OUT / f'template_cloud_{s}.json').write_text(json.dumps(out, indent=1))
print('rho', rho, 'coords', XF.shape[1], 'd', d_); print('reach', {k: round(v['reach'], 1) for k, v in out['classes'].items() if k in main[:8]})
for p in pr: print(p[0], p[1], 'dist %.1f predicted %.4f measured %.4f' % p[2:])
print('chi2', {k: (round(v['mean'], 1), round(v['beyond99'], 3)) for k, v in q.items()}); print('within eig', np.round(ev, 2))
for i, l in enumerate(out['loadings']): print(i, [(n, round(x, 2)) for n, x in l])
