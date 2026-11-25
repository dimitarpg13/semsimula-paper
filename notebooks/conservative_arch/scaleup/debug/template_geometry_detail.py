#!/usr/bin/env python3
"""Detail readings of the geometry-only templates (2026-11-25, POST HOC: written after the first SR2-2B results).
    python3 template_geometry_detail.py RUNDIR OUTDIR stem[,stem...]
Matched comparisons (a rival is compared with the valid premise whose slot and polarity it imitates), more particles
(m up to 8), the confusion of kinds, predicted against measured pair errors, and the chi-squared law of the distances."""
import json, sys, pickle
from pathlib import Path
import numpy as np
A_ = Path(__file__).with_name('template_geometry_analysis.py').read_text(); exec(A_[:A_.index("res = {}; np.random.seed")])
from scipy import stats
VAL = {'MP': ('cond', 'MP'), 'MT': ('cond', 'MT'), 'DS1': ('disj', 'DS1'), 'DS2': ('disj', 'DS2')}
HARD = {'near miss': {'MP': 'near', 'MT': 'nearQn', 'DS1': 'nearn', 'DS2': 'nearQn'}, 'other subject': {'MP': 'swap', 'MT': 'swapQn', 'DS1': 'swapn', 'DS2': 'swapQn'}, 'unrelated': {'MP': 'U', 'MT': 'Un', 'DS1': 'Un', 'DS2': 'Un'}}
SY = {'MP': 'MPsyn', 'MT': 'MTsyn', 'DS1': 'DS1syn', 'DS2': 'DS2syn'}
out = {}
for s in STEMS:
    d = pickle.load(open(RUN / f'robust_{s}.pkl', 'rb')); M, Rr = d['meta'], d['res']; K = np.array([x['kind'] for x in M]); want = np.array([x['want'] for x in M], object); wi = np.array([x['wording'] for x in M]); sp = np.array([x['split'] for x in M]); fr = np.array([x['frame'] for x in M]); g = np.array([x['content'] for x in M])
    lab = np.array(['text' if k == 'text' else f'{f}:{k}' for k, f in zip(K, fr)]); dev, te = sp == 'dev', sp == 'test'; stw = wi == 'text'
    for lv in ((1,), (2,), (1, 2), (0,)):
        nm = f"{s} {'+'.join(f'h{l}' for l in lv)}"; q = out[nm] = {}; Xm = {m: np.concatenate([np.stack([coords(r['st'][l], r['roles'], m) for r in Rr]) for l in lv], 1) for m in range(1, 9)}
        sc = {m: cv(Xm[m][dev], lab[dev], want[dev], g[dev])[1:] for m in Xm}; q['particles'] = {m: v for m, v in sc.items()}; bm = max(sc, key=lambda m: sc[m][0]); m_ = min(m for m in sc if sc[m][0] >= sc[bm][0] - sc[bm][1]); X = Xm[m_]; q['m'] = m_
        ds = (2, 4, 6, 8, 10, 12, 15, 20, 25, 31); sd = {x: cv(X[dev], lab[dev], want[dev], g[dev], x)[1:] for x in ds}; bd = max(sd, key=lambda x: sd[x][0]); d_ = min(x for x in ds if sd[x][0] >= sd[bd][0] - sd[bd][1]); q['d'] = d_; q['dims'] = sd
        for un in (False, True):
            F, L = np.full(len(K), '', object), np.zeros(len(K))
            for h in (sum(WC.values(), []) if un else [None]):
                a = dev & (wi != h); c = Clouds(X[a], lab[a], want[a], d_); rho = cv(X[a], lab[a], want[a], g[a], d_)[0]; b = te & ((wi == h) | (stw & (h == 'if-then')) if un else True); L[b] = c.rates(X[b])[3]; F[b] = c.fire(X[b], rho)
            r = q['unseen' if un else 'known'] = dict(report=report(F, K, want, g, L, te)); cond = lambda t, kk: te & (K == kk) & (fr == VAL[t][0])
            for n_, mp in HARD.items():
                r[f'matched false, {n_}'] = {t: ci(F[cond(t, kk)] != '', g[cond(t, kk)]) for t, kk in mp.items()}
                r[f'matched AUROC, exact / {n_}'] = {t: auroc(L[cond(t, t)], L[cond(t, kk)]) for t, kk in mp.items()}
                r[f'matched AUROC, paraphrase / {n_}'] = {t: auroc(L[cond(t, SY[t])], L[cond(t, kk)]) for t, kk in mp.items()}
            r['detect by kind'] = {t: {kk: ci(F[cond(t, kk)] == (t if kk[-3:] != 'not' else t), g[cond(t, kk)]) for kk in (t, SY[t]) + ((t + 'not',) if t + 'not' in set(K) else ())} for t in VAL}
            r['detect by wording'] = {w: [float((F[te & (wi == w) & np.isin(K, list(VAL))] == want[te & (wi == w) & np.isin(K, list(VAL))]).mean()), float((F[te & (wi == w) & np.isin(K, ['AC', 'DA', 'P', 'Q'])] != '').mean())] for w in sum(WC.values(), [])}
            if not un:                                     # predicted against measured; chi-squared
                Y = ((X - c.mu0) / c.sd) @ c.B; ids = {k: i for i, k in enumerate(c.cl)}; pr = []
                for a_, b_ in (('cond:MP', 'cond:DA'), ('cond:MP', 'cond:near'), ('cond:MP', 'cond:swap'), ('cond:MT', 'cond:AC'), ('cond:MT', 'cond:nearQn'), ('cond:MT', 'cond:swapQn'), ('disj:DS1', 'disj:P'), ('disj:DS1', 'disj:nearn'), ('disj:DS2', 'disj:Q'), ('disj:DS1', 'disj:DS2'), ('cond:MP', 'disj:P')):
                    ma, mb = c.C[ids[a_]], c.C[ids[b_]]; dl = np.linalg.norm(ma - mb); w = (ma - mb); t_ = (ma @ ma - mb @ mb) / 2; ea, eb = te & (lab == a_), te & (lab == b_); pr.append((a_, b_, float(dl), float(stats.norm.cdf(-dl / 2)), float(((Y[ea] @ w < t_).mean() + (Y[eb] @ w >= t_).mean()) / 2), float(np.exp(-dl ** 2 / 8))))
                q['pairs'] = pr; D2 = {k: ((Y[te & (lab == k)] - c.C[ids[k]]) ** 2).sum(1) for k in ('cond:MP', 'cond:MT', 'disj:DS1', 'disj:DS2')}; q['chi2'] = {k: dict(d=int(d_), mean=float(v.mean()), median=float(np.median(v)), chi2_median=float(stats.chi2.ppf(.5, d_)), beyond_99=float((v > stats.chi2.ppf(.99, d_)).mean())) for k, v in D2.items()}
        f1 = lambda r: ' '.join(f'{t} {v[0]:.2f}' for t, v in r.items()); f2 = lambda r: ' '.join(f'{t} {v:.2f}' for t, v in r.items())
        print(f"##### {nm} m {m_} d {d_}; dev by m " + ' '.join(f'{m}:{v[0]:.3f}' for m, v in sc.items()))
        for w_ in ('known', 'unseen'):
            r = q[w_]; print(f"  {w_:6s} detect " + ' '.join(f'{t} {v[t][0]:.2f}' for t, v in r['detect by kind'].items()) + ' | para ' + ' '.join(f'{t} {v[SY[t]][0]:.2f}' for t, v in r['detect by kind'].items()) + ' | any false %.2f AC %.2f DA %.2f disj %.2f wrong %.2f' % (r['report']['any false fire'][0], r['report']['false AC'][0], r['report']['false DA'][0], r['report']['false affirmed disjunct'][0], r['report']['wrong template']))
            for n_ in HARD: print(f'     matched {n_:13s} false {f1(r["matched false, " + n_])} | AUROC exact {f2(r["matched AUROC, exact / " + n_])} | AUROC paraphrase {f2(r["matched AUROC, paraphrase / " + n_])}')
        print('  pairs (distance, predicted error, measured):', ' ; '.join(f'{a.split(":")[1]}/{b.split(":")[1]} {x:.1f} {p:.3f} {e:.3f}' for a, b, x, p, e, _ in q['pairs']), '\n  chi2', {k: (round(v['mean'], 1), v['d'], round(v['beyond_99'], 2)) for k, v in q['chi2'].items()}, flush=True)
        json.dump(out, open(OUT / 'template_geometry_detail.json', 'w'), indent=1)
