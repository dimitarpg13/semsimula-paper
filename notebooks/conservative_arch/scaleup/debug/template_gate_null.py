#!/usr/bin/env python3
"""Hybrid step 1b (2026-11-26): the null check for the residual separation of step 0, adjusted after step 1a. Written before any
result of it was seen.
    python3 template_gate_null.py CACHEDIR OUTDIR stem[,stem...] [draws]
CACHEDIR holds cache_<stem>.npz of template_gate_fit.py. For each paraphrase pair (MPsyn against near, MTsyn against nearQn, DS1syn
against nearn, DS2syn against nearQn, in the template's frame) and step 0's control (MP against the affirmed disjunct P), the layer
block b (the best single state, or all states) is joined to the embedding's coordinates a, and three things are read:
  pooled     step 0's measure, cross-fitted: delta_A' S^-1 delta_B in (a, b) minus the same in a, S the pooled within-kind
             covariance of all test premises but ordinary text, shrunk 10% toward its diagonal, the means from the two halves
  along      the squared separation the two kinds have along a direction fitted on one half, read on the other half with the
             other half's own spread ((mean gap / average sd)^2), averaged over the two swaps; in (a, b) minus in a
  auroc      the held-out AUROC of that direction, in (a, b) minus in a
Each is read with the real block b and with two null blocks of the same dimension: independent standard normal noise, and the
real block with its rows shuffled at random over all premises (it keeps b's own distribution and drops its tie to the premise).
`draws` null draws of each (default 20). The halves are the even and odd clauses, as steps 0 and 1a.
Expectation, stated before the run: under both nulls, the pooled residual is within 0.5 squared widths of zero on average, the along
and AUROC residuals are zero or negative (a fit with more coordinates and no information can only lose); with all states, the real
residual is above every null draw on every paraphrase pair, for all three readings and models.
Writes OUTDIR/template_gate_null_<stem>.json and prints a table."""
import json, sys
from pathlib import Path
import numpy as np
CACHE, OUT, STEMS = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3].split(',')
DRAWS = int(sys.argv[4]) if len(sys.argv) > 4 else 20
PAIRS = [('MP', ('cond', 'MPsyn'), ('cond', 'near')), ('MT', ('cond', 'MTsyn'), ('cond', 'nearQn')), ('DS1', ('disj', 'DS1syn'), ('disj', 'nearn')),
         ('DS2', ('disj', 'DS2syn'), ('disj', 'nearQn')), ('control', ('cond', 'MP'), ('disj', 'P'))]


def auroc(a, b):
    r = np.argsort(np.argsort(np.r_[a, b])) + 1; return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def pooled_cov(Z, grp, rows):
    W = np.concatenate([Z[rows & (grp == k)] - Z[rows & (grp == k)].mean(0) for k in np.unique(grp[rows])]); S = np.cov(W.T); return .9 * S + .1 * np.diag(np.diag(S))


def read(X, na, K, fr, grp, nt, te, halves):
    """per pair: (pooled, along, auroc) for the first na coordinates and for all of X."""
    Z = (X - X[te].mean(0)) / (X[te].std(0) + 1e-9); S = pooled_cov(Z, grp, nt); out = {}
    Zh = [(X - X[h].mean(0)) / (X[h].std(0) + 1e-9) for h in halves]; Sh = [pooled_cov(Zh[i], grp, halves[i]) for i in (0, 1)]
    for t, (fp, kp), (fq, kq) in PAIRS:
        P, Q = (K == kp) & (fr == fp), (K == kq) & (fr == fq); r = {}
        for tag, idx in (('a', np.arange(na)), ('ab', np.arange(X.shape[1]))):
            d = [Z[P & h][:, idx].mean(0) - Z[Q & h][:, idx].mean(0) for h in halves]; pooled = float(d[0] @ np.linalg.solve(S[np.ix_(idx, idx)], d[1]))
            al, au = [], []
            for i in (0, 1):
                f, s_ = halves[i], halves[1 - i]; Zf = Zh[i][:, idx]; w = np.linalg.solve(Sh[i][np.ix_(idx, idx)], Zf[P & f].mean(0) - Zf[Q & f].mean(0)); v = Zf @ w
                al.append(float((v[P & s_].mean() - v[Q & s_].mean()) ** 2 / (.5 * (v[P & s_].var() + v[Q & s_].var())))); au.append(auroc(v[P & s_], v[Q & s_]))
            r[tag] = (pooled, float(np.mean(al)), float(np.mean(au)))
        out[t] = dict(pooled=r['ab'][0] - r['a'][0], along=r['ab'][1] - r['a'][1], auroc=r['ab'][2] - r['a'][2], a=r['a'])
    return out


rng = np.random.default_rng(0)
for s in STEMS:
    C = np.load(CACHE / f'cache_{s}.npz', allow_pickle=True); best = str(C['best']); K, wi, sp, fr, g = (C[k] for k in ('K', 'wi', 'sp', 'fr', 'g'))
    te = (sp == 'test') & (wi != 'unless'); nt = te & (K != 'text'); grp = np.char.add(np.char.add(fr.astype(str), ':'), K.astype(str))
    cs = np.unique(g[nt]); halves = [nt & np.isin(g, cs[::2]), nt & np.isin(g, cs[1::2])]; Xa = C['X_h0']; na = Xa.shape[1]; o = {}
    print(f'\n{s}: {DRAWS} draws of each null; residual = with the block minus the embedding alone (pooled | along | AUROC)')
    for bn in (best, 'states'):
        Xb = C[f'X_{bn}']; real = read(np.c_[Xa, Xb], na, K, fr, grp, nt, te, halves); nulls = {'noise': [], 'shuffled': []}
        for _ in range(DRAWS):
            nulls['noise'].append(read(np.c_[Xa, rng.standard_normal(Xb.shape)], na, K, fr, grp, nt, te, halves))
            nulls['shuffled'].append(read(np.c_[Xa, Xb[rng.permutation(len(Xb))]], na, K, fr, grp, nt, te, halves))
        o[bn] = r = dict(dim=int(Xb.shape[1]), pairs={})
        for t, *_ in PAIRS:
            q = r['pairs'][t] = dict(real={m: real[t][m] for m in ('pooled', 'along', 'auroc')}, embedding_alone=dict(zip(('pooled', 'along', 'auroc'), real[t]['a'])))
            for nn, L in nulls.items():
                q[nn] = {m: dict(mean=float(np.mean([x[t][m] for x in L])), sd=float(np.std([x[t][m] for x in L])), max=float(np.max([x[t][m] for x in L]))) for m in ('pooled', 'along', 'auroc')}
            print(f'   {bn:6s} {t:7s} real {q["real"]["pooled"]:+6.2f} | {q["real"]["along"]:+5.2f} | {q["real"]["auroc"]:+.3f}   '
                  + '   '.join(f'{nn} {q[nn]["pooled"]["mean"]:+5.2f} (max {q[nn]["pooled"]["max"]:+5.2f}) | {q[nn]["along"]["mean"]:+5.2f} (max {q[nn]["along"]["max"]:+5.2f}) | {q[nn]["auroc"]["mean"]:+.3f} (max {q[nn]["auroc"]["max"]:+.3f})' for nn in nulls), flush=True)
    (OUT / f'template_gate_null_{s}.json').write_text(json.dumps(o, indent=1))
