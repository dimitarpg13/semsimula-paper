#!/usr/bin/env python3
"""The geometry-only templates with the connective's position as one more coordinate (2026-11-25, POST HOC).
    python3 template_geometry_frame.py RUNDIR OUTDIR stem[,stem...]
The frame coordinate: the connective's whitened mean state (unit length) along the axis from the disjunctive wordings'
centroid to the conditional ones', both the mean of the wordings' centroids on the training rows, one axis per state.
Everything else as template_geometry_analysis.py; the axis is refitted in every fold and with the tested wording left out."""
import sys
from pathlib import Path
A_ = Path(__file__).with_name('template_geometry_detail.py').read_text(); A_ = A_[:A_.index("out = {}\n")]; exec(A_)
class Frame:
    def __init__(s, Cs, w, f):
        s.ax = []
        for C in Cs:
            u = lambda v: v / np.linalg.norm(v); cen = {k: u(np.mean([u(C[w == x].mean(0)) for x in WC[k] if (w == x).any()], axis=0)) for k in WC}; a = u(cen['cond'] - cen['disj']); s.ax.append((a, (cen['cond'] + cen['disj']) / 2))
    def __call__(s, Cs): return np.stack([(C - m) @ a for C, (a, m) in zip(Cs, s.ax)], 1)
def cvf(X, Cs, lab, want, g, w, f, d):
    R, t, ins = np.zeros((len(X), 4)), np.zeros(len(X), object), np.zeros(len(X), bool); fold = (g % 1000) % 5
    for k in range(5):
        a, b = fold != k, fold == k; F_ = Frame([C[a] for C in Cs], w[a], f[a]); c = Clouds(np.c_[X[a], F_([C[a] for C in Cs])], lab[a], want[a], d); R[b], t[b], ins[b], _ = c.rates(np.c_[X[b], F_([C[b] for C in Cs])])
    rho = decide_rho(R, t, ins, want, lab); F = np.where((R.max(1) >= rho) & ins, t, ''); ok = F == want; return rho, float(np.mean([ok[lab == k].mean() for k in sorted(set(lab))]))
out = {}
for s in STEMS:
    D_ = pickle.load(open(RUN / f'robust_{s}.pkl', 'rb')); M, Rr = D_['meta'], D_['res']; K = np.array([x['kind'] for x in M]); want = np.array([x['want'] for x in M], object); wi = np.array([x['wording'] for x in M]); sp = np.array([x['split'] for x in M]); fr = np.array([x['frame'] for x in M]); g = np.array([x['content'] for x in M])
    lab = np.array(['text' if k == 'text' else f'{f}:{k}' for k, f in zip(K, fr)]); dev, te = sp == 'dev', sp == 'test'; stw = wi == 'text'
    for lv, m_ in (((1,), 6), ((2,), 4), ((1, 2), 4), ((0,), 8)):
        nm = f"{s} {'+'.join(f'h{l}' for l in lv)}"; X = np.concatenate([np.stack([coords(r['st'][l], r['roles'], m_) for r in Rr]) for l in lv], 1); Cs = []
        for l in lv: C = np.stack([r['st'][l]['c'].astype(np.float64) for r in Rr]); Cs.append(C / np.linalg.norm(C, axis=1, keepdims=True))
        sc = {d: cvf(X[dev], [C[dev] for C in Cs], lab[dev], want[dev], g[dev], wi[dev], fr[dev], d)[1] for d in (6, 10, 15, 20, 25, 31)}; d = max(sc, key=sc.get); q = out[nm] = dict(m=m_, d=d, dev=sc)
        for un in (False, True):
            F, L = np.full(len(K), '', object), np.zeros(len(K))
            for h in (sum(WC.values(), []) if un else [None]):
                a = dev & (wi != h); F_ = Frame([C[a] for C in Cs], wi[a], fr[a]); tr = np.c_[X[a], F_([C[a] for C in Cs])]; c = Clouds(tr, lab[a], want[a], d); rho = cvf(X[a], [C[a] for C in Cs], lab[a], want[a], g[a], wi[a], fr[a], d)[0]
                b = te & ((wi == h) | (stw & (h == 'if-then')) if un else True); Y = np.c_[X[b], F_([C[b] for C in Cs])]; L[b] = c.rates(Y)[3]; F[b] = c.fire(Y, rho)
            r = q['unseen' if un else 'known'] = dict(report=report(F, K, want, g, L, te)); cond = lambda t, kk: te & (K == kk) & (fr == VAL[t][0])
            r['detect'] = {t: ci(F[cond(t, t)] == t, g[cond(t, t)]) for t in VAL}; r['para'] = {t: ci(F[cond(t, SY[t])] == t, g[cond(t, SY[t])]) for t in VAL}
            for n_, mp in HARD.items(): r[f'matched false, {n_}'] = {t: ci(F[cond(t, kk)] != '', g[cond(t, kk)]) for t, kk in mp.items()}; r[f'AUROC para / {n_}'] = {t: auroc(L[cond(t, SY[t])], L[cond(t, kk)]) for t, kk in mp.items()}
            r['by wording'] = {w: [float((F[te & (wi == w) & np.isin(K, list(VAL))] == want[te & (wi == w) & np.isin(K, list(VAL))]).mean()), float((F[te & (wi == w) & np.isin(K, ['AC', 'DA', 'P', 'Q'])] != '').mean())] for w in sum(WC.values(), [])}
            p = r['report']; print(f"{nm:14s} d{d} {'unseen' if un else 'known ':6s} detect " + ' '.join(f'{t} {v[0]:.2f}' for t, v in r['detect'].items()) + ' | para ' + ' '.join(f'{v[0]:.2f}' for v in r['para'].values()) + f" | denied {p['detect reworded denial'][0]:.2f} | false any {p['any false fire'][0]:.3f} AC {p['false AC'][0]:.2f} DA {p['false DA'][0]:.2f} disj {p['false affirmed disjunct'][0]:.2f} unrel {p['false unrelated'][0]:.2f} near {p['false near miss'][0]:.3f} subj {p['false other subject'][0]:.3f} text {p['false ordinary text'][0]:.2f} wrong {p['wrong template']:.2f} | AUROC para/near " + ' '.join(f'{v:.2f}' for v in r['AUROC para / near miss'].values()) + ' /subj ' + ' '.join(f'{v:.2f}' for v in r['AUROC para / other subject'].values()), flush=True)
            if un: print('     by unseen wording (detect, false on fallacies):', ' '.join(f'{w} {a_:.2f}/{b_:.2f}' for w, (a_, b_) in r['by wording'].items()))
        json.dump(out, open(OUT / 'template_geometry_frame.json', 'w'), indent=1)
