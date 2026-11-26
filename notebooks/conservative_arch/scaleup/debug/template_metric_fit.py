#!/usr/bin/env python3
"""A metric fitted for each reading (2026-11-26; written before any result of it was seen).
    python3 template_metric_fit.py VECDIR OUTDIR stem[,stem...]
VECDIR holds depth_<stem>.pkl of template_vectors_extract.py (the particles' whitened state vectors). The question: the matcher's
coordinates were designed on the embedding, around exact repetition. Does a metric fitted for each reading, instead of the plain
whitening by ordinary text, let a layer state show paraphrase identity that the embedding cannot?
The metric (approach G of Hybrid_Template_Matchers.md, closed form, no tuning). On the fitting half of the test clauses, pairs of the
last particle of the second sentence and the last particle of the clause it restates (or nearly restates):
  same       MP, MT, DS1, DS2 and their paraphrases MPsyn, MTsyn, DS1syn, DS2syn
  different  the near misses near, nearn, nearQ, nearQn
with d = u - v the difference of the two whitened vectors, S_same and S_diff the second moments of d over the two sets, each shrunk
by adding its mean eigenvalue times the identity, M = S_same^-1 - S_diff^-1 (KISSME), its negative eigenvalues set to zero, scaled to
trace 384, and the metric used is I + M: the plain whitening plus an equal share of the fitted part. Every particle of every premise
(development, test, ordinary text) is mapped by the square root of I + M, and its cosines and lengths are recomputed; the connective's
vector is left as it was. The matcher is otherwise unchanged (template_depth_analysis.py: "unless" left out, the same search for m and
d on the development clauses, which have no paraphrases).
Cross-fitting: the metric is fitted on the even test clauses and the matcher scored on the odd ones, then the reverse; the per-clause
scores of the two halves are joined. Each reading (every state h0 ... hL) is scored with the plain whitening and with the metric, on
the same halves, so all differences are paired by clause.
Expectations, stated before the run:
  M1  the metric raises the paraphrases caught at 1% near misses for every reading on every model, by at least 0.05
  M2  the gain is larger for the best early state than for the embedding on at least two of the three models
  M3  the embedding's R1 (unseen wordings) moves by at most 0.02 under the metric
  M4  under the metric, the AUROC of paraphrase against near miss exceeds 0.85 for at least one reading on each model
Writes OUTDIR/template_metric_<stem>.json and prints a table."""
import json, pickle, sys
from pathlib import Path
import numpy as np
DBG = Path(__file__).resolve().parent; sys.path.insert(0, str(DBG))
RUN_, OUT_, STEMS_ = sys.argv[1:4]
_S = (DBG / 'template_depth_analysis.py').read_text(); _S = _S[:_S.rindex("if __name__ == '__main__':")]; exec(_S)
RUN, OUT = Path(RUN_), Path(OUT_)
SAME = {'MP': 'a1', 'MPsyn': 'a1', 'MT': 'a2', 'MTsyn': 'a2', 'DS1': 'a1', 'DS1syn': 'a1', 'DS2': 'a2', 'DS2syn': 'a2'}
DIFFK = {'near': 'a1', 'nearn': 'a1', 'nearQ': 'a2', 'nearQn': 'a2'}


def last(roles, ro):
    return max(i for i, r in enumerate(roles) if r == ro)


def fit_metric(V, R, K, rows):
    dS, dD = [], []
    for i in np.flatnonzero(rows):
        k = K[i]
        if k in SAME or k in DIFFK:
            ro = SAME.get(k) or DIFFK[k]; d = V[i][last(R[i], 'p2')].astype(np.float64) - V[i][last(R[i], ro)].astype(np.float64)
            (dS if k in SAME else dD).append(d)
    mom = lambda D: (lambda S: S + np.trace(S) / len(S) * np.eye(len(S)))(np.asarray(D).T @ np.asarray(D) / len(D))
    M = np.linalg.inv(mom(dS)) - np.linalg.inv(mom(dD)); e, U = np.linalg.eigh(M); e = np.maximum(e, 0); e *= len(e) / max(e.sum(), 1e-12)
    return U * np.sqrt(1 + e), len(dS), len(dD)          # z -> z @ T gives the metric I + M


def remap(V, st, T):
    out = []
    for v, s_ in zip(V, st):
        z = v.astype(np.float64) @ T; nm = np.linalg.norm(z, axis=1); zu = z / (nm[:, None] + 1e-9); out.append(dict(G=zu @ zu.T, norm=nm, c=s_['c']))
    return out


for s in STEMS_.split(','):
    M_ = pickle.load(open(RUN / f'depth_{s}.pkl', 'rb')); meta, res = M_['meta'], M_['res']; R = [x['roles'] for x in res]; HK = sorted(res[0]['vec'], key=lambda k: int(k[1:]))
    K = np.array([x['kind'] for x in meta]); want = np.array([x['want'] for x in meta], object); wi = np.array([x['wording'] for x in meta]); sp = np.array([x['split'] for x in meta]); fr = np.array([x['frame'] for x in meta]); g = np.array([x['content'] for x in meta])
    lab = np.array(['text' if k == 'text' else f'{f}:{k}' for k, f in zip(K, fr)]); te = sp == 'test'
    out = {}
    for k in HK:
        V = [x['vec'][k] for x in res]; base = [x['st'][k] for x in res]; per = {'plain': [], 'metric': []}; au = {'plain': [], 'metric': []}; info = []
        for h in (0, 1):
            fit = te & (g % 2 == h) & (wi != 'unless'); sp2 = np.where(te & (g % 2 == h), 'fit', sp); L_meta = (K, want, wi, sp2, fr, g, lab)
            T, nS, nD = fit_metric(V, R, K, fit); info.append(dict(half=h, same_pairs=nS, different_pairs=nD))
            for cond, P in (('plain', {k: base}), ('metric', {k: remap(V, base, T)})):
                q, vec = score(k, P, (k,), R, L_meta); per[cond].append(vec); au[cond].append(q['AUROC para/near']); info[-1][cond] = dict(m=q['m'], d=q['d'], R1=q['R1 unseen'], para01=q['para at near-miss rate']['0.01'], auroc=q['AUROC para/near'])
                print(f"{s} {k} half {h} {cond:6s} m{q['m']} d{q['d']:2d} R1 {q['R1 unseen']:.3f} para 1% {q['para at near-miss rate']['0.01']:.3f} AUROC {q['AUROC para/near']:.3f} (pairs {nS}/{nD})", flush=True)
        J = {c: {m: np.concatenate([v[m] for v in per[c]]) for m in ('r1', 'p0.01', 'p0.02')} for c in per}
        out[k] = dict(halves=info, **{c: dict(R1=float(J[c]['r1'].mean()), para01=float(J[c]['p0.01'].mean()), para02=float(J[c]['p0.02'].mean()), auroc=float(np.mean(au[c]))) for c in J},
                      metric_minus_plain={m: boot(J['metric'][m] - J['plain'][m]) for m in ('r1', 'p0.01', 'p0.02')}, vectors={c: {m: J[c][m].tolist() for m in J[c]} for c in J})
        print(f"{s} {k}: metric minus plain R1 {out[k]['metric_minus_plain']['r1']} para 1% {out[k]['metric_minus_plain']['p0.01']}", flush=True)
    e0 = {c: {m: np.array(out['h0']['vectors'][c][m]) for m in ('r1', 'p0.01')} for c in ('plain', 'metric')}
    for k in HK:
        if k != 'h0': out[k]['minus h0'] = {c: {m: boot(np.array(out[k]['vectors'][c][m]) - e0[c][m]) for m in ('r1', 'p0.01')} for c in ('plain', 'metric')}
    (OUT / f'template_metric_{s}.json').write_text(json.dumps(out, indent=1)); del M_, res
