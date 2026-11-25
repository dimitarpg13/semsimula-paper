#!/usr/bin/env python3
"""Geometry-only two-part templates: clouds of premise figures, how many particles they need, and the smallest subspace
that holds them (2026-11-25; written before any result of this version was seen).
    python3 template_geometry_analysis.py RUNDIR OUTDIR stem[,stem...]

Input per premise and state: the cosines between all particles (token states centred and whitened by ordinary text) and
their norms. No token identity, no word list, no corpus frequency. h1 and h2 are the matcher; h0 is a reference only.
Relational coordinates r (dimension 10 m + 10):
  per clause k of premise 1: the m smallest best-cosines of premise 2's particles to the clause; the m smallest of the
  clause's particles to premise 2; the cosine of the two last particles; the cosine of the two mean states
  per particle among the m of premise 2 worst matched to its nearer clause: log norm, cosine to "Therefore", to the
  connective, to its neighbours in premise 2, and to premise 2's mean state
  the six cosines among the mean states of connective, clause 1, clause 2, "Therefore"
Clouds: one Gaussian per kind of development premise (mean; one Ledoit-Wolf covariance for all), ordinary text included.
The rate of a template is the sum of its clouds' densities over the sum of all clouds' densities. It fires when its rate
is the largest and at least rho, and the premise lies within the 99% reach of its cloud. rho: the smallest value with a
false-fire rate of at most 1% in content-wise five-fold cross-validation on development items (mean over the rival kinds).
An unseen wording is tested with clouds frozen on the other wordings. m and the subspace dimension d (linear discriminant
directions of the clouds) are the smallest within one standard error of the best cross-validated development score.
Score: mean over kinds of correct decisions (fires the right template, or stays silent)."""
import json, pickle, sys
from pathlib import Path
import numpy as np
from sklearn.covariance import LedoitWolf
RUN, OUT = Path(sys.argv[1]), Path(sys.argv[2]); STEMS = sys.argv[3].split(','); OUT.mkdir(parents=True, exist_ok=True)
WC = {'cond': ['if-then', 'when', 'whenever', 'q-if-p', 'assuming', 'provided'], 'disj': ['either-or', 'or', 'at-least-one', 'or-else', 'unless', 'one-holds']}
DIFF = {'unrelated': ('U', 'Un'), 'near miss': ('near', 'nearn', 'nearQ', 'nearQn'), 'other subject': ('swap', 'swapn', 'swapQ', 'swapQn')}
FALL = {'AC': ('AC',), 'DA': ('DA', 'DAnot'), 'affirmed disjunct': ('P', 'Q')}; SYN_V = ('MPsyn', 'MTsyn', 'DS1syn', 'DS2syn'); SYN_F = ('ACsyn', 'DAsyn', 'Psyn')
MS = (1, 2, 3, 4, 5)
def coords(st, roles, m):
    G, nm = st['G'].astype(np.float64), st['norm'].astype(np.float64); R = np.array(roles); ix = {k: np.flatnonzero(R == k) for k in ('c', 'a1', 'a2', 'p2', 'th')}
    I = G * nm[:, None] * nm[None]; nodes = ('c', 'a1', 'a2', 'p2', 'th'); A = np.stack([(R == k) / max(1, (R == k).sum()) for k in nodes]); N = A @ I @ A.T; dn = np.sqrt(np.maximum(np.diag(N), 1e-9)); Nc = N / dn[:, None] / dn[None]
    srt = lambda v: np.sort(v)[np.minimum(np.arange(m), len(v) - 1)]; p = ix['p2']; f = []; best = []
    for j, k in ((1, 'a1'), (2, 'a2')):
        S = G[np.ix_(p, ix[k])]; best.append(S.max(1)); f += [srt(S.max(1)), srt(S.max(0)), [G[p[-1], ix[k][-1]], Nc[3, j]]]
    near = best[0] if best[0].mean() >= best[1].mean() else best[1]; o = np.argsort(near)[np.minimum(np.arange(m), len(p) - 1)]; w = p[o]; pm = A[3]; tomean = (I @ pm) / nm / dn[3]; toc = (I @ A[0]) / nm / dn[0]
    prev, nxt = np.maximum(o - 1, 0), np.minimum(o + 1, len(p) - 1)
    f += [np.log(nm[w]), G[w, ix['th'][-1]], toc[w], G[w, p[prev]], G[w, p[nxt]], tomean[w]]
    return np.concatenate([np.atleast_1d(v) for v in f] + [Nc[[0, 0, 0, 1, 1, 2], [1, 2, 4, 2, 4, 4]]])
def names(m):
    n = []
    for k in ('clause 1', 'clause 2'):
        n += [f'premise 2 to {k}, worst {i + 1}' for i in range(m)] + [f'{k} to premise 2, worst {i + 1}' for i in range(m)] + [f'last particles, {k}', f'mean states, {k}']
    for q in ('log norm', 'to Therefore', 'to connective', 'to previous', 'to next', 'to premise-2 mean'): n += [f'odd particle {i + 1}: {q}' for i in range(m)]
    return n + ['connective-clause 1', 'connective-clause 2', 'connective-Therefore', 'clause 1-clause 2', 'clause 1-Therefore', 'clause 2-Therefore']
class Clouds:
    def __init__(s, X, lab, want, d=None):
        s.mu0, s.sd = X.mean(0), X.std(0) + 1e-9; Z = (X - s.mu0) / s.sd; s.cl = sorted(set(lab)); s.tpl = {c: want[np.flatnonzero(lab == c)[0]] for c in s.cl}
        M = np.stack([Z[lab == c].mean(0) for c in s.cl]); Rz = np.concatenate([Z[lab == c] - M[i] for i, c in enumerate(s.cl)]); Sg = LedoitWolf(assume_centered=True).fit(Rz).covariance_
        e, U = np.linalg.eigh(Sg); s.Wh = U / np.sqrt(np.maximum(e, 1e-9)); Mw = M @ s.Wh; Mc = Mw - Mw.mean(0); _, sv, Vt = np.linalg.svd(Mc, full_matrices=False); s.sv = sv; s.V = Vt[:d].T if d else Vt[:min(len(s.cl) - 1, Vt.shape[0])].T
        s.B = s.Wh @ s.V; s.C = Mw @ s.V; Y = Z @ s.B; s.reach = {c: np.quantile(((Y[lab == c] - s.C[i]) ** 2).sum(1), .99) for i, c in enumerate(s.cl)}
    def rates(s, X):
        Y = ((X - s.mu0) / s.sd) @ s.B; D = ((Y[:, None] - s.C[None]) ** 2).sum(2); L = -.5 * D; L = L - L.max(1, keepdims=True); P = np.exp(L); P /= P.sum(1, keepdims=True)
        T = ['MP', 'MT', 'DS1', 'DS2']; Rt = np.stack([P[:, [i for i, c in enumerate(s.cl) if s.tpl[c] == t]].sum(1) for t in T], 1); b = D.argmin(1); inside = D[np.arange(len(D)), b] <= np.array([s.reach[c] for c in s.cl])[b]
        # log rate of the best template against the best rival cloud, for threshold-free readings
        vt = np.array([s.tpl[c] != '' for c in s.cl]); llr = L[:, vt].max(1) - L[:, ~vt].max(1)
        return Rt, np.array(T, object)[Rt.argmax(1)], inside, llr
    def fire(s, X, rho):
        Rt, t, inside, _ = s.rates(X); return np.where((Rt.max(1) >= rho) & inside, t, '')
def decide_rho(R, t, inside, want, lab):
    riv = [k for k in sorted(set(lab)) if want[np.flatnonzero(lab == k)[0]] == '']
    for rho in np.r_[.3:.9:.02, .9:.999:.005, .999, .9995, .9999]:
        f = (R.max(1) >= rho) & inside
        if np.mean([f[lab == k].mean() for k in riv]) <= .01: return float(rho)
    return 1.01
def cv(X, lab, want, g, d=None):
    R, t, ins = np.zeros((len(X), 4)), np.zeros(len(X), object), np.zeros(len(X), bool); fold = (g % 1000) % 5
    for k in range(5):
        a, b = fold != k, fold == k; c = Clouds(X[a], lab[a], want[a], d); R[b], t[b], ins[b], _ = c.rates(X[b])
    rho = decide_rho(R, t, ins, want, lab); F = np.where((R.max(1) >= rho) & ins, t, ''); ok = F == want; kinds = sorted(set(lab))
    per = np.array([np.mean([ok[(lab == k) & (fold == q)].mean() for k in kinds]) for q in range(5)]); return rho, float(np.mean([ok[lab == k].mean() for k in kinds])), float(per.std(ddof=1) / np.sqrt(5))
def ci(x, g, B=1000):
    x = np.asarray(x, float); _, gi = np.unique(g, return_inverse=True); s, n = np.bincount(gi, x), np.bincount(gi); i = np.random.default_rng(0).integers(len(n), size=(B, len(n))); return [float(x.mean())] + [float(v) for v in np.percentile(s[i].sum(1) / n[i].sum(1), [2.5, 97.5])]
def auroc(a, b):
    v = np.r_[a, b]; _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True); r = (np.cumsum(cnt) - (cnt - 1) / 2)[inv]; return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))
def report(F, K, want, g, llr, m):
    r = {}
    for nm, ks in (('MP', ['MP']), ('MT', ['MT']), ('DS', ['DS1', 'DS2']), ('reworded denial', ['MTnot', 'DS1not']), ('paraphrase', SYN_V)):
        k = m & np.isin(K, ks); r['detect ' + nm] = ci(F[k] == want[k], g[k])
    for nm, ks in {**FALL, **DIFF, 'paraphrased fallacy': SYN_F, 'ordinary text': ('text',)}.items():
        k = m & np.isin(K, ks)
        if k.any(): r['false ' + nm] = ci(F[k] != '', g[k])
    k = m & np.isin(K, ['MP', 'MT', 'DS1', 'DS2', 'MTnot', 'DS1not']); r['wrong template'] = float(((F[k] != '') & (F[k] != want[k])).mean())
    rival = m & np.isin(K, sum(DIFF.values(), ()) + sum(FALL.values(), ())); r['any false fire'] = ci(F[rival] != '', g[rival]); r['AUROC exact / rivals'] = auroc(llr[k], llr[rival]); sy = m & np.isin(K, SYN_V)
    r['AUROC paraphrase / near miss'] = auroc(llr[sy], llr[m & np.isin(K, DIFF['near miss'])]); r['AUROC paraphrase / other subject'] = auroc(llr[sy], llr[m & np.isin(K, DIFF['other subject'])]); r['AUROC paraphrase / unrelated'] = auroc(llr[sy], llr[m & np.isin(K, DIFF['unrelated'])]); return r
res = {}; np.random.seed(0)
for s in STEMS:
    d = pickle.load(open(RUN / f'robust_{s}.pkl', 'rb')); M, Rr = d['meta'], d['res']; K = np.array([x['kind'] for x in M]); want = np.array([x['want'] for x in M], object); wi = np.array([x['wording'] for x in M]); sp = np.array([x['split'] for x in M]); fr = np.array([x['frame'] for x in M]); g = np.array([x['content'] for x in M])
    lab = np.array(['text' if k == 'text' else f'{f}:{k}' for k, f in zip(K, fr)]); dev, te = sp == 'dev', sp == 'test'; stw = np.array([w == 'text' for w in wi])
    for lv in ((1,), (2,), (1, 2), (0,)):
        nm = f"{s} {'+'.join(f'h{l}' for l in lv)}"; q = res[nm] = {}; Xm = {m: np.concatenate([np.stack([coords(r['st'][l], r['roles'], m) for r in Rr]) for l in lv], 1) for m in MS}
        sc = {m: cv(Xm[m][dev], lab[dev], want[dev], g[dev]) for m in MS}; bm = max(sc, key=lambda m: sc[m][1]); m_ = min(m for m in MS if sc[m][1] >= sc[bm][1] - sc[bm][2]); q['particles'] = {str(m): dict(dev=sc[m][1], se=sc[m][2]) for m in MS}; q['m'] = m_; X = Xm[m_]
        full = len(set(lab[dev])) - 1; full = min(full, X.shape[1]); ds = [x for x in (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25) if x < full] + [full]; sd = {x: cv(X[dev], lab[dev], want[dev], g[dev], x) for x in ds}; bd = max(sd, key=lambda x: sd[x][1]); d_ = min(x for x in ds if sd[x][1] >= sd[bd][1] - sd[bd][2]); q['dims'] = {str(x): dict(dev=sd[x][1], se=sd[x][2]) for x in ds}; q['d'], q['full'], q['D'] = d_, full, X.shape[1]
        def run(Xa, dim, unseen):
            F, L = np.full(len(K), '', object), np.zeros(len(K))
            for h in (sum(WC.values(), []) if unseen else [None]):
                a = dev & (wi != h); c = Clouds(Xa[a], lab[a], want[a], dim); rho = cv(Xa[a], lab[a], want[a], g[a], dim)[0]; b = te & ((wi == h) | (stw & (h == 'if-then')) if unseen else True); _, _, _, l_ = c.rates(Xa[b]); F[b] = c.fire(Xa[b], rho); L[b] = l_
            return F, L, (c, rho)
        for tn, dim in (('minimal subspace', d_), ('full', None)):
            for un in (False, True):
                F, L, (c, rho) = run(X, dim, un); q[f'{tn}, {"unseen" if un else "known"} wording'] = report(F, K, want, g, L, te)
                if un: q[f'{tn}, by wording'] = {w: [float(np.mean(F[te & (wi == w) & np.isin(K, ['MP', 'MT', 'DS1', 'DS2'])] == want[te & (wi == w) & np.isin(K, ['MP', 'MT', 'DS1', 'DS2'])])), float(np.mean(F[te & (wi == w) & np.isin(K, sum(DIFF.values(), ()) + sum(FALL.values(), ()))] != ''))] for w in sum(WC.values(), [])}
                if not un and tn == 'minimal subspace':
                    Y = ((X[dev] - c.mu0) / c.sd) @ c.B; corr = np.array([[np.corrcoef(X[dev][:, j], Y[:, i])[0, 1] for j in range(X.shape[1])] for i in range(dim)]); nms = sum([[f'h{l} {x}' for x in names(m_)] for l in lv], [])
                    q['cloud'] = dict(rho=rho, classes={k: dict(template=c.tpl[k], mean=c.C[i].tolist(), reach=float(c.reach[k])) for i, k in enumerate(c.cl)}, singular=c.sv.tolist(), directions=[[(nms[j], float(corr[i, j])) for j in np.argsort(-abs(corr[i]))[:5]] for i in range(dim)], within_cov_test=np.cov((((X[te & ~stw & ~np.char.endswith(lab.astype(str), 'syn')] - c.mu0) / c.sd) @ c.B - np.stack([c.C[c.cl.index(k)] for k in lab[te & ~stw & ~np.char.endswith(lab.astype(str), 'syn')]])).T).tolist())
                    np.savez(OUT / f'cloud_{nm.replace(" ", "_")}.npz', mu0=c.mu0, sd=c.sd, B=c.B, C=c.C, classes=np.array(c.cl), Y_test=(((X[te] - c.mu0) / c.sd) @ c.B), lab_test=lab[te])
        pq = [np.linalg.eigvalsh((lambda I_: I_)(np.array([r['st'][lv[-1]]['G'].astype(float) * np.outer(r['st'][lv[-1]]['norm'], r['st'][lv[-1]]['norm'])])[0])) for r in [Rr[i] for i in np.flatnonzero(te & ~stw)[::40]]]; q['state dims of one premise (90%)'] = float(np.median([int((np.cumsum(np.sort(e)[::-1]) / e.sum() < .9).sum() + 1) for e in pq])); q['particles in one premise'] = float(np.median([len(e) for e in pq]))
        v = lambda r, k: f"{r[k][0] if isinstance(r[k], list) else r[k]:.2f}" if k in r else '-'
        print(f"##### {nm}: m {m_}, d {d_} of {full} (coordinates {X.shape[1]}); dev by m " + ' '.join(f'{m}:{sc[m][1]:.3f}' for m in MS) + ' | by d ' + ' '.join(f'{x}:{sd[x][1]:.3f}' for x in ds) + f" | state dims {q['state dims of one premise (90%)']:.0f} of {q['particles in one premise']:.0f} particles")
        for tn in ('minimal subspace, known wording', 'full, known wording', 'minimal subspace, unseen wording', 'full, unseen wording'):
            r = q[tn]; print(f'   {tn:33s} detect ' + ' '.join(f'{k} {v(r, "detect " + k)}' for k in ('MP', 'MT', 'DS', 'reworded denial', 'paraphrase')) + ' | false ' + ' '.join(f'{k} {v(r, "false " + k)}' for k in ('AC', 'DA', 'affirmed disjunct', 'unrelated', 'near miss', 'other subject', 'ordinary text')) + f" | any {v(r, 'any false fire')} wrong {v(r, 'wrong template')} | AUROC exact/rivals {r['AUROC exact / rivals']:.3f} para/near {r['AUROC paraphrase / near miss']:.2f} para/subj {r['AUROC paraphrase / other subject']:.2f} para/unrel {r['AUROC paraphrase / unrelated']:.2f}", flush=True)
        json.dump(res, open(OUT / 'template_geometry.json', 'w'), indent=1)
