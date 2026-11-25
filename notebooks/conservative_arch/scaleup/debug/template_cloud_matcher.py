#!/usr/bin/env python3
"""Cloud matcher (2026-11-25; POST HOC, designed after template_matcher_analysis.py's first results on SR2-2B).

    python3 template_cloud_matcher.py RUNDIR OUTDIR stem[,stem...]

Each premise pattern of the fitting items is frozen as a Gaussian cloud of relational features (mean and a covariance
shared by the patterns, shrunk); rejection patterns (AC, DA, P, Q, unrelated, ordinary text) are clouds too. A template
fires when one of its clouds is the most likely and the argument lies within that cloud's reach (99th percentile of its
own fitting Mahalanobis distances). No threshold is tuned on the test clauses or the new wordings.
Feature sets:
  figure   the 10 distances of the five-node unit figure (rotation discarded)
  relation figure + cosines of premise 2 to each clause + its offset from each clause along the frozen negation direction
           + the connective's cosine to the two frozen connective centroids
"""
import json, sys
from pathlib import Path
import numpy as np
A_ = Path(__file__).with_name('template_matcher_analysis.py').read_text(); exec(A_[:A_.index("res = {}; SF")])
PAT = {'c:MP': 'MP', 'c:MT': 'MT', 'd:DS1': 'DS', 'd:DS2': 'DS'}
CLS = ['c:MP', 'c:MT', 'c:AC', 'c:DA', 'c:unrel+', 'c:unrel-', 'd:DS1', 'd:DS2', 'd:P', 'd:Q', 'd:unrel+', 'd:unrel-', 'text']
def feats(X, p, which):
    Y = X - X.mean(1, keepdims=True); Y = Y / np.linalg.norm(Y, axis=(1, 2), keepdims=True); i, j = np.triu_indices(5, 1)
    f = [np.linalg.norm(Y[:, i] - Y[:, j], axis=-1)]
    if which == 'relation':
        Z = X - p['mu']; f.append(np.stack([cos(Z[:, 3], Z[:, 1]), cos(Z[:, 3], Z[:, 2]), (Z[:, 3] - Z[:, 1]) @ p['w'], (Z[:, 3] - Z[:, 2]) @ p['w'],
                                            cos(Z[:, 0], p['con_cond']), cos(Z[:, 0], p['con_disj'])], 1))
    return np.concatenate(f, 1)
def fit(Fx):
    M = {c: Fx[FITT & (kind == c)].mean(0) for c in CLS}; R = np.concatenate([Fx[FITT & (kind == c)] - M[c] for c in CLS if c != 'text'])
    Sg = np.cov(R.T); Sg = .9 * Sg + .1 * np.diag(np.diag(Sg)); t = Fx[FITT & (kind == 'text')] - M['text']; St = np.cov(t.T); St = .9 * St + .1 * np.diag(np.diag(St))
    P = {c: np.linalg.inv(St if c == 'text' else Sg) for c in CLS}; ld = {c: np.linalg.slogdet(P[c])[1] for c in CLS}
    md = lambda x, c: np.einsum('ni,ij,nj->n', x - M[c], P[c], x - M[c])
    reach = {c: np.quantile(md(Fx[FITT & (kind == c)], c), .99) for c in CLS}
    return md, ld, reach
def fire(Fx, model):
    md, ld, reach = model; D = np.stack([md(Fx, c) for c in CLS], 1); L = -.5 * D + .5 * np.array([ld[c] for c in CLS]); b = L.argmax(1)
    inside = D[np.arange(len(D)), b] <= np.array([reach[c] for c in CLS])[b]; name = np.array([PAT.get(c, '') for c in CLS])[b]
    F = {t: inside & (name == t) for t in TP}; any_ = {t: name == t for t in TP}
    return F, any_, np.array(CLS)[b]
if __name__ == '__main__':
    out = {}
    for s in STEMS:
        N = np.load(RUN / f'nodes_{s}.npz')['N'].astype(np.float32)
        for l in (0, 1, 2):
            X = N[:, l]; p = freeze(X)
            for which in ('figure', 'relation'):
                Fx = feats(X, p, which); F, Fa, b = fire(Fx, fit(Fx)); T = table(F); Ta = table(Fa); out[f'{s} h{l} {which}'] = dict(reach=T, no_reach=Ta)
                for nm, R in (('', T), (' (no reach limit)', Ta)):
                    for c in CELLS:
                        r = R[c]; print(f'{s} h{l} {which:8s}{nm:18s} {c[13:]:15s} exact ' + ' '.join(f"{t} {r[f'detect {t} exact']:.2f}" for t in TP)
                              + ' | para ' + ' '.join(f"{t} {r[f'detect {t} paraphrase']:.2f}" for t in TP) + ' | false ' + ' '.join(f"{k[6:]} {v:.2f}" for k, v in r.items() if k.startswith('false'))
                              + f" | wrong {r['wrong template on a valid argument']:.2f} text {max(R['ordinary text'].values()):.2f}")
    (OUT / 'template_cloud.json').write_text(json.dumps(out, indent=1))
