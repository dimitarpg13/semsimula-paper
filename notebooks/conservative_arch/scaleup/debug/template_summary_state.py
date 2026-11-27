#!/usr/bin/env python3
"""Task A: the logical form read from the state at "Therefore" alone (2026-11-27; written before any result of it).
    python3 template_summary_state.py VECDIR OUTDIR stem[,stem...] [analogy=ANALOGYDIR] [items=template_blogic_items] [tag=blogic] [dry=1]
VECDIR holds depth_<stem>.pkl of template_vectors_extract.py (the depth premises with each particle's whitened state vector, kinds
h0 ... hL). Calls: companion_notes/semantic_templates/Summary_State_Test.md. Confirmed by the author on 2026-11-27, before any score.

"Therefore" is the same token in every premise, so its embedding differs between premises only by the position vector, that is by
the premise's length. Whatever else the state at "Therefore" carries about the premise, the layers put there. Readings, each a
vector per premise fed to the clouds of the geometry matcher (one Gaussian cloud per kind of premise, the discriminant subspace, the
1% false-fire rule; no relational coordinates and no connective position, since there is one particle):
  length    the premise's length in tokens (what the position vector can carry)
  h<l>      the state at "Therefore" entering layer l (h0 the embedding; h1 the early state)
  states    the states at "Therefore" after every layer, side by side
Each vector reading is reduced to 64 principal components fitted on the development premises of the fit. The subspace dimension is
chosen by the five-fold content-wise development score over 6, 10, 15, 20, 25, 31 (1 for length). "Unless" is left out.
Scores, on the 60 test clauses, intervals from 4,000 resamples of the clauses, differences paired by clause:
  R1u    unseen wordings: half the detection of MP, MT, DS1, DS2 (right template fires), half the silence on AC, DA, DAnot, P, Q
  rank   unseen wordings: on valid premises the right template has the largest density of the four (chance 0.25)
  twin   known wordings: AUROC of the valid template's log odds, valid form against its twin (MP-AC, MT-DA, DS1-P, DS2-Q)
  para   known wordings: paraphrases caught (right template, log odds above the 99th percentile of the near misses)
analogy=: the clouds fitted on all development premises read the Test 2 premises (analogy_<stem>.pkl, extracted with vec=1):
foil and twin pair readings per level as in template_analogy_analysis.py; exploratory.
Writes OUTDIR/template_summary_state_<stem>.json."""
import json, pickle, sys
from pathlib import Path
import numpy as np
DBG = Path(__file__).resolve().parent; sys.path.insert(0, str(DBG))
OPTS = dict(a.split('=', 1) for a in sys.argv[4:] if '=' in a)
ITEMS_MOD = OPTS.get('items', 'template_analogy_items')
VEC, OUT_, STEMS_ = sys.argv[1:4]; sys.argv = [sys.argv[0], VEC, OUT_, STEMS_]
_S = (DBG / 'template_depth_analysis.py').read_text(); exec(_S[:_S.rindex("if __name__ == '__main__':")])   # Clouds, cv, decide_rho, auroc, WC, ...
import tiktoken
enc = tiktoken.get_encoding('gpt2'); RNG = np.random.default_rng(4); T4 = ['MP', 'MT', 'DS1', 'DS2']; GRID = (6, 10, 15, 20, 25, 31)
HS_ = [('cond', 'AC'), ('cond', 'DA'), ('cond', 'DAnot'), ('disj', 'P'), ('disj', 'Q')]; TWINS = (('MP', 'AC', 'cond'), ('MT', 'DA', 'cond'), ('DS1', 'P', 'disj'), ('DS2', 'Q', 'disj'))
WC.clear(); WC.update({k: [w for w in v if w != 'unless'] for k, v in ALLW.items()}); WS = sum(WC.values(), [])


def bootc(per):
    i = RNG.integers(len(per), size=(4000, len(per))); b = per[i].mean(1); return [float(per.mean()), *[float(x) for x in np.percentile(b, [2.5, 97.5])]]


class Reader:
    """PCA (vector readings) + clouds with d chosen by content-wise CV; log odds per template"""
    def __init__(s, X, lab, want, g, pca=64):
        s.mu = X.mean(0)
        if X.shape[1] > 1:
            _, _, Vt = np.linalg.svd(X - s.mu, full_matrices=False); s.P = Vt[:pca].T
        else: s.P = np.eye(1)
        Z = s.z(X); grid = (1,) if Z.shape[1] == 1 else [d for d in GRID if d <= min(Z.shape[1], len(set(lab)) - 1)]
        best = max(((cv(Z, lab, want, g, d)[1], -d, d) for d in grid))[2]; s.d = best
        s.rho = cv(Z, lab, want, g, best)[0]; s.c = Clouds(Z, lab, want, best)
    def z(s, X): return (X - s.mu) @ s.P
    def read(s, X):
        Z = s.z(X); Y = ((Z - s.c.mu0) / s.c.sd) @ s.c.B; L = -.5 * ((Y[:, None] - s.c.C[None]) ** 2).sum(2); tpl = np.array([s.c.tpl[x] for x in s.c.cl], object)
        lse = lambda m: np.logaddexp.reduce(np.where(m[None], L, -np.inf), axis=1)
        S = np.stack([lse(tpl == t) for t in T4], 1); LO = np.stack([S[:, j] - lse(tpl != t) for j, t in enumerate(T4)], 1)
        _, _, _, llr = s.c.rates(Z); return LO, np.array(T4, object)[S.argmax(1)], s.c.fire(Z, s.rho), llr


def score(X, M):
    K, want, wi, sp, fr, g, lab = M; dev, te = (sp == 'dev') & (wi != 'unless'), (sp == 'test') & (wi != 'unless'); cont = np.unique(g[te & (K != 'text')])
    q = {}; R = Reader(X[dev], lab[dev], want[dev], g[dev]); LO, top, F, llr = R.read(X); q['d known'] = R.d
    # known wordings: twin AUROC per clause, paraphrases at 1% near misses
    tw = []
    for c in cont:
        a = []
        for v, t_, f_ in TWINS:
            j = T4.index(v); x, y = LO[te & (g == c) & (K == v) & (fr == f_), j], LO[te & (g == c) & (K == t_) & (fr == f_), j]
            if len(x) and len(y): a.append(auroc(x, y))
        tw.append(np.mean(a))
    q['twin'] = bootc(np.array(tw)); sy, nr = te & np.isin(K, SYN_V), te & np.isin(K, DIFF['near miss'])
    hit = (top[sy] == want[sy]) & (llr[sy] > np.quantile(llr[nr], .99)); q['para'] = bootc(np.array([hit[g[sy] == c].mean() for c in cont])) if sy.any() else None
    q['known detect'] = {t: float((F[te & (K == t)] == t).mean()) for t in T4}; q['known fires on fallacies'] = float((F[te & np.isin(K, ['AC', 'DA', 'DAnot', 'P', 'Q'])] != '').mean())
    # unseen wordings: each wording left out of the fit
    Fu, Tu = np.full(len(K), '', object), np.full(len(K), '', object); ds = []
    for w in WS:
        a = dev & (wi != w); R = Reader(X[a], lab[a], want[a], g[a]); b = te & (wi == w); _, Tu[b], Fu[b], _ = R.read(X[b]); ds.append(R.d)
    r1 = np.array([.5 * np.mean([(Fu[te & (g == c) & (K == t) & (fr == VAL[t][0])] == t).mean() for t in T4]) + .5 * np.mean([(Fu[te & (g == c) & (K == k) & (fr == f)] == '').mean() for f, k in HS_]) for c in cont])
    rk = np.array([np.mean([(Tu[te & (g == c) & (K == t) & (fr == VAL[t][0])] == t).mean() for t in T4]) for c in cont])
    q['R1u'], q['rank'], q['d unseen'] = bootc(r1), bootc(rk), ds
    q['unseen detect'] = {t: float((Fu[te & (K == t)] == t).mean()) for t in T4}; q['unseen fires on fallacies'] = float((Fu[te & np.isin(K, ['AC', 'DA', 'DAnot', 'P', 'Q'])] != '').mean())
    return q, dict(R1u=r1, rank=rk, twin=np.array(tw)), R


if __name__ == '__main__':
    out = {}
    for s in STEMS:
        M_ = pickle.load(open(Path(VEC) / f'depth_{s}.pkl', 'rb')); meta, res, L_ = M_['meta'], M_['res'], M_['L']; del M_
        th = {k: np.stack([r['vec'][k][r['roles'].index('th')].astype(np.float64) for r in res]) for k in res[0]['vec']}; del res
        ln = np.array([[len(enc.encode(m['text']))] for m in meta], float)
        K = np.array([x['kind'] for x in meta]); want = np.array([x['want'] for x in meta], object); wi = np.array([x['wording'] for x in meta]); sp = np.array([x['split'] for x in meta]); fr = np.array([x['frame'] for x in meta])
        g = np.array([x['content'] for x in meta]); lab = np.array(['text' if k == 'text' else f'{f}:{k}' for k, f in zip(K, fr)]); Mt = (K, want, wi, sp, fr, g, lab)
        READ = [('length', ln)] + [(k, th[k]) for k in sorted(th, key=lambda k: int(k[1:]))] + [('states', np.concatenate([th[f'h{l}'] for l in range(1, L_ + 1)], 1))]
        if OPTS.get('dry'):
            keep = np.flatnonzero((np.arange(len(K)) % 7 == 0) | (sp == 'dev')); print(s, 'dry run on', len(keep), 'premises', {k: v.shape for k, v in READ}, flush=True)
            Mt = tuple(a[keep] for a in Mt); READ = [(k, v[keep]) for k, v in READ[:2]]
        q = out[s] = {}; per = {}; readers = {}
        for name, X in READ:
            q[name], per[name], readers[name] = score(X, Mt)
            if OPTS.get('dry'): print('  ran', name, 'd', q[name]['d known'], flush=True); continue
            print(f"{s} {name:7s} R1u {q[name]['R1u'][0]:.3f} rank {q[name]['rank'][0]:.3f} twin {q[name]['twin'][0]:.3f} para {q[name]['para'][0] if q[name]['para'] else float('nan'):.3f} | known det " + ' '.join(f'{v:.2f}' for v in q[name]['known detect'].values()) + f" fires {q[name]['known fires on fallacies']:.3f} | unseen det " + ' '.join(f'{v:.2f}' for v in q[name]['unseen detect'].values()) + f" fires {q[name]['unseen fires on fallacies']:.3f}", flush=True)
        if OPTS.get('dry'): continue
        for name in per:
            for ref in ('length', 'h0'):
                if name != ref: q[name][f'minus {ref}'] = {k: bootc(per[name][k] - per[ref][k]) for k in per[name]}
        if 'analogy' in OPTS:                                     # exploratory: the Test 2 premises read by the clouds fitted on all development premises
            from importlib import import_module as _im
            _IT = _im(ITEMS_MOD); PAIRS, LEVEL_ORDER = _IT.PAIRS, _IT.LEVEL_ORDER
            A_ = pickle.load(open(Path(OPTS['analogy']) / f'analogy_{s}.pkl', 'rb')); am = [m for m in A_['meta'] if m['split'] == 'analogy']; ar = [r for m, r in zip(A_['meta'], A_['res']) if m['split'] == 'analogy']; del A_
            ath = {k: np.stack([r['vec'][k][r['roles'].index('th')].astype(np.float64) for r in ar]) for k in ar[0]['vec']}; aln = np.array([[len(enc.encode(m['text']))] for m in am], float)
            AX = {'length': aln, **ath, 'states': np.concatenate([ath[f'h{l}'] for l in range(1, L_ + 1)], 1)}
            lv = np.array([m['level'] for m in am]); ac = np.array([m['content'] for m in am]); aw = np.array([m['wording'] for m in am]); ak = np.array([m['kind'] for m in am])
            for name, R in readers.items():
                LO = R.read(AX[name])[0]; q[name]['analogy'] = {}
                for L in LEVEL_ORDER:
                    for rd in sorted({p[0] for p in PAIRS[L]}):
                        w = {c: [] for c in range(20)}
                        for _, a, b, t in [p for p in PAIRS[L] if p[0] == rd]:
                            j = T4.index(t); ia = {(ac[i], aw[i]): i for i in np.flatnonzero((lv == L) & (ak == a))}; ib = {(ac[i], aw[i]): i for i in np.flatnonzero((lv == L) & (ak == b))}
                            for key, i in ia.items():
                                x, y = LO[i, j], LO[ib[key], j]; w[key[0]].append(1. if x > y + 1e-9 else .5 if abs(x - y) <= 1e-9 else 0.)
                        q[name]['analogy'][f'{L} {rd}'] = bootc(np.array([np.mean(w[c]) for c in range(20)]))
        (Path(OUT_) / f'template_summary_state{"_" + OPTS["tag"] if "tag" in OPTS else ""}_{s}.json').write_text(json.dumps(q, indent=1))
