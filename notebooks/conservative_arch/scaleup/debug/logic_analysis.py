#!/usr/bin/env python3
"""Analysis of the saved node states of logic_figures.py and logic_controls.py (2026-11-25; CPU, seconds).
    python3 logic_analysis.py RESULTS_DIR
Every classifier is a nearest-centroid rule with one content (all its forms) held out at a time."""
import json, sys
from pathlib import Path
import numpy as np
D0 = Path(sys.argv[1]); CL = [1, 3, 4, 6]; CON = [0, 2, 5]; ALL = list(range(7)); n = 40
def unit(X):
    X = X - X.mean(-2, keepdims=True); return X / np.linalg.norm(X, axis=(-2, -1), keepdims=True)
def dv(X, idx=ALL):
    U = unit(X[:, idx]); i, j = np.triu_indices(len(idx), 1); return np.linalg.norm(U[:, i] - U[:, j], axis=-1)
def loo(A, B):
    r = []
    for i in range(n):
        k = np.arange(n) != i; a, b = A[k].mean(0), B[k].mean(0)
        r.append([np.linalg.norm(A[i] - a) < np.linalg.norm(A[i] - b), np.linalg.norm(B[i] - b) < np.linalg.norm(B[i] - a)])
    return float(np.mean(np.array(r, float)))
def transfer(V, I, V2, I2):
    """direction from invalid I to valid V, learned without content i, applied to the pair (V2, I2): share of contents
    whose valid member lies further along it"""
    return float(np.mean([((V2[i] - I2[i]) @ (V - I)[np.arange(n) != i].mean(0)) > 0 for i in range(n)]))
def contrast(mp, da, mt, ac, B=2000):
    """The validity contrast V_i = (MP - DA + MT - AC)/2 per content, in which an additive negation effect cancels, against
    the negation contrast N_i = (MP - DA - MT + AC)/2. Norms of the content means, and a sign-flip null for V. The null
    tests any interaction of negation with the recurrence pattern, of which validity is one reading."""
    V, N = (mp - da + mt - ac) / 2, (mp - da - mt + ac) / 2; rng = np.random.default_rng(0); G = V @ V.T
    obs = G.sum(); sg = rng.choice([-1., 1.], (B, n)); null = np.einsum('bi,ij,bj->b', sg, G, sg)
    k = lambda i: np.arange(n) != i
    acc = np.mean([(V[i] @ V[k(i)].mean(0)) > 0 for i in range(n)])
    return dict(V_norm=float(np.linalg.norm(V.mean(0))), N_norm=float(np.linalg.norm(N.mean(0))), p=float((1 + (null >= obs).sum()) / (B + 1)), loo=float(acc))
def d90(F):
    s = np.linalg.svd(F, compute_uv=False) ** 2; return int((np.cumsum(s) / s.sum() < .9).sum() + 1)
res = {}
for m in ('sr2np', 'fobk16np', 'sr2', 'sr2_2b'):
    if not (D0 / f'logic_controls_{m}.npz').exists():
        continue
    Z = dict(np.load(D0 / f'logic_figures_{m}.npz')); Z.update(np.load(D0 / f'logic_controls_{m}.npz')); R = {}
    for l in (0, 1, 2):
        X = {k: Z[f'nodes_h{l}_{k}'] for k in ('MP', 'MT', 'DS', 'AC', 'DA', 'NR', 'TXT')}
        raw = {k: v[:, [4, 6]].reshape(n, -1) for k, v in X.items()}; cn = {k: v[:, 6] for k, v in X.items()}
        q = R[f'h{l}'] = {}
        for a, b in (('MP', 'DA'), ('MT', 'AC'), ('MP', 'MT'), ('MP', 'NR')):
            q[f'{a}-{b}'] = dict(shape=loo(dv(X[a]), dv(X[b])), clause_shape=loo(dv(X[a], CL), dv(X[b], CL)), raw=loo(raw[a], raw[b]), cn=loo(cn[a], cn[b]))
        q['transfer_MPDA_to_MTAC'] = transfer(raw['MP'], raw['DA'], raw['MT'], raw['AC'])
        q['transfer_MTAC_to_MPDA'] = transfer(raw['MT'], raw['AC'], raw['MP'], raw['DA'])
        q['validity_contrast'] = dict(p2_cn=contrast(raw['MP'], raw['DA'], raw['MT'], raw['AC']), cn=contrast(cn['MP'], cn['DA'], cn['MT'], cn['AC']))
        for k in ('MP', 'NR', 'TXT'):
            F = X[k] - X[k].mean(1, keepdims=True); e = (F ** 2).sum((1, 2)).mean()
            v = dv(X[k]); c = dv(X[k], CL)
            q[f'desc_{k}'] = dict(inst_d90=float(np.median([d90(f) for f in F])), form_d90={str(t): d90(F[:t].reshape(-1, 384)) for t in (5, 10, 20, 40)},
                connective_energy=float((F[:, CON] ** 2).sum() / (F ** 2).sum()), shape_cv=float(((v - v.mean(0)) ** 2).sum(1).mean() / (v.mean(0) ** 2).sum()),
                clause_cv=float(((c - c.mean(0)) ** 2).sum(1).mean() / (c.mean(0) ** 2).sum()))
        # restatement distances inside the clause-only figure
        U = {k: unit(X[k][:, CL]) for k in X}; dd = lambda k, i, j: float(np.linalg.norm(U[k][:, i] - U[k][:, j], axis=-1).mean())
        q['restate'] = {k: dict(p2_a1=dd(k, 2, 0), p2_a2=dd(k, 2, 1), cn_a1=dd(k, 3, 0), cn_a2=dd(k, 3, 1)) for k in X}
    res[m] = R; print('##', m)
    for l, q in R.items():
        print(' ', l, '; '.join(f"{p} shape {q[p]['shape']:.2f} clauses {q[p]['clause_shape']:.2f} raw {q[p]['raw']:.2f} cn {q[p]['cn']:.2f}" for p in ('MP-DA', 'MT-AC', 'MP-MT', 'MP-NR')))
        print('     validity direction: MP/DA applied to MT/AC %.2f, MT/AC applied to MP/DA %.2f' % (q['transfer_MPDA_to_MTAC'], q['transfer_MTAC_to_MPDA']))
        print('     validity contrast (negation cancelled), conclusion state:', {a: round(b, 4) for a, b in q['validity_contrast']['cn'].items()})
        for k in ('MP', 'NR', 'TXT'): print('     ', k, q[f'desc_{k}'])
    print('  restate h2', {k: {a: round(b, 2) for a, b in v.items()} for k, v in R['h2']['restate'].items()})
(D0 / 'logic_analysis.json').write_text(json.dumps(res, indent=1))

# ---- can a frozen figure (the mean inter-node distances, 21 numbers) identify its form? ----
FM = ('MP', 'MT', 'DS', 'AC', 'DA'); MS = [m for m in ('sr2np', 'fobk16np', 'sr2', 'sr2_2b') if m in res]; fr = {}
for l in (0, 1, 2):
    V = {}
    for m in MS:
        Z = dict(np.load(D0 / f'logic_figures_{m}.npz')); Z.update(np.load(D0 / f'logic_controls_{m}.npz'))
        V[m] = {f: dv(Z[f'nodes_h{l}_{f}']) for f in FM + ('NR', 'TXT')}
        V[m].update({f + '_c': dv(Z[f'nodes_h{l}_{f}'], CL) for f in FM})
    fr[f'h{l}'] = {}
    for a in MS:                                          # frozen on a (content i held out), read on b
        for b in MS:
            for suf in ('', '_c'):
                ok = np.zeros((5, 5))
                for i in range(n):
                    k = np.arange(n) != i; M = np.stack([V[a][f + suf][k].mean(0) for f in FM])
                    for j, f in enumerate(FM):
                        ok[j, np.linalg.norm(M - V[b][f + suf][i], axis=1).argmin()] += 1
                fr[f'h{l}'][f'{a}->{b}{suf}'] = dict(acc=float(np.trace(ok) / (5 * n)), confusion=ok.tolist())
    for m in MS:                                          # rejection: distance to the nearest frozen figure, held out
        M = np.stack([V[m][f].mean(0) for f in FM]); near = lambda x: np.linalg.norm(x[:, None] - M, axis=-1).min(1)
        own = np.concatenate([near(V[m][f]) for f in FM])
        fr[f'h{l}'][f'reject_{m}'] = {k: float((near(V[m][k]) > np.percentile(own, 95)).mean()) for k in ('NR', 'TXT')}
        same = np.mean([np.linalg.norm(V[m][f].mean(0) - V[o][f].mean(0)) for f in FM for o in MS if o != m])
        diff = np.mean([np.linalg.norm(V[m][f].mean(0) - V[m][g].mean(0)) for f in FM for g in FM if f != g])
        fr[f'h{l}'][f'agree_{m}'] = dict(other_models_same_form=float(same), same_model_other_forms=float(diff))
res['frozen'] = fr; (D0 / 'logic_analysis.json').write_text(json.dumps(res, indent=1))
for l, q in fr.items():
    print('FROZEN', l, 'five-way accuracy, frozen on row, read on column (all nodes | clause nodes)')
    for a in MS: print('   ', a, ' '.join(f"{q[f'{a}->{b}']['acc']:.2f}|{q[f'{a}->{b}_c']['acc']:.2f}" for b in MS), q[f'reject_{a}'], {k: round(v, 3) for k, v in q[f'agree_{a}'].items()})
print('confusion h2, sr2_2b (rows true, columns read as MP MT DS AC DA):', fr['h2']['sr2_2b->sr2_2b']['confusion'])
