#!/usr/bin/env python3
"""Templates read from the dynamics at any depth, against the embedding (2026-11-25; written before any result of the depth arms).
    python3 template_depth_analysis.py RUNDIR OUTDIR stem[,stem...]
RUNDIR holds depth_<stem>.pkl and flow_<stem>.json of template_depth_test.py. Scoring as template_dynamics_analysis.py, with two
changes decided on the SR2-2B report: the primary reading leaves the wording "unless" out of fitting and scoring (the twelve-wording
numbers are kept as a secondary reading, at the same m and d; a third reading, added at the author's request before any depth result
was read, keeps "unless" with clouds of its own that feed the existing templates), and paraphrases are read at a fixed rate of near misses (1%) as well
as at the rule's threshold. Matchers: h0 (the reference), every single layer's h, D, W, and
  path    the displacements of all layers        wells   the well frames of all layers        states  the states after all layers
Primary: path against h0 on R1 (unseen wordings, without "unless") and on the paraphrases caught at 1% near misses (known wordings).
Between two models the differences are compared clause by clause."""
import sys
from pathlib import Path
_D = Path(__file__).with_name('template_dynamics_analysis.py').read_text(); _D = _D[:_D.index("if __name__ == '__main__':")]; exec(_D)
HS = [('cond', 'AC'), ('cond', 'DA'), ('cond', 'DAnot'), ('disj', 'P'), ('disj', 'Q')]; ALLW = {k: list(v) for k, v in WC.items()}; RATES = (.0025, .01, .02, .05)
def boot(d, B=4000):
    i = np.random.default_rng(1).integers(len(d), size=(B, len(d))); b = d[i].mean(1); return [float(d.mean()), *[float(x) for x in np.percentile(b, [2.5, 97.5])]]
def score(name, P, ks, R, L_meta, fixed=None, drop='unless', own=False):
    """drop: a wording left out of fitting and scoring. own: "unless" kept, with clouds of its own (kind 'unl:<kind>') that feed the
    existing templates, the connective axis fitted on the other eleven wordings; R1, R2 and the rates over the other eleven."""
    K, want, wi, sp, fr, g, lab = L_meta; WC.clear(); WC.update({k: [w for w in v if w != drop and not (own and w == 'unless')] for k, v in ALLW.items()}); WS = sum(WC.values(), []) + (['unless'] if own else [])
    keep = wi != drop
    if own: lab = np.where(wi == 'unless', np.char.add('unl:', K.astype(str)), lab)
    dev, te_all = (sp == 'dev') & keep, (sp == 'test') & keep; te = te_all & (wi != 'unless') if own else te_all; stw = wi == 'text'; contents = np.unique(g[te & ~stw])
    Cs = []
    for k in ks: C = np.stack([x['c'] for x in P[k]]).astype(np.float64); Cs.append(C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9))
    X_ = lambda m: np.concatenate([np.stack([coords(x, r, m) for x, r in zip(P[k], R)]) for k in ks], 1)
    if fixed is None: Xm = {m: X_(m) for m in GRID_M}; sc, m, d = tune(Xm, Cs, dev, lab, want, g, wi, fr); X = Xm[m]; del Xm
    else: (sc, m, d), X = fixed, None
    X = X_(m) if X is None else X; q = dict(dev=sc, m=m, d=d); Fs = {}
    for un in (False, True):
        F, Lr, T_ = np.full(len(K), '', object), np.zeros(len(K)), np.full(len(K), '', object)
        for h in (WS if un else [None]):
            a = dev & (wi != h); F_ = Frame([C[a] for C in Cs], wi[a], fr[a]); c = Clouds(np.c_[X[a], F_([C[a] for C in Cs])], lab[a], want[a], d); rho = cvf(X[a], [C[a] for C in Cs], lab[a], want[a], g[a], wi[a], fr[a], d)[0]
            b = te_all & ((wi == h) | (stw & (h == 'if-then')) if un else True); Y = np.c_[X[b], F_([C[b] for C in Cs])]; _, t, _, Lr[b] = c.rates(Y); T_[b] = t; F[b] = c.fire(Y, rho)
        Fs[un] = F, Lr, T_; tag = 'unseen' if un else 'known'; q[f'{tag} detect'] = {t: float((F[te & (K == t)] == t).mean()) for t in VAL}; q[f'{tag} fires on'] = {f'{f}:{kk}': float((F[te & (K == kk) & (fr == f)] != '').mean()) for f, kk in HS}
        if own: u = te_all & (wi == 'unless'); q[f'{tag} unless'] = dict(detect={t: float((F[u & (K == t)] == t).mean()) for t in ('DS1', 'DS2')}, fires_on={kk: float((F[u & (K == kk)] != '').mean()) for kk in ('P', 'Q')}, which_on={kk: {str(x): int(n_) for x, n_ in zip(*np.unique(F[u & (K == kk)].astype(str), return_counts=True))} for kk in ('DS1', 'DS2', 'P', 'Q')}, conditional_DA={str(x): int(n_) for x, n_ in zip(*np.unique(F[te & (K == 'DA') & (fr == 'cond')].astype(str), return_counts=True))}, conditional_DA_by_wording={w: float((F[te & (K == 'DA') & (wi == w)] != '').mean()) for w in WC['cond']})
        q[f'{tag} by wording'] = {w: float((F[te & (wi == w) & np.isin(K, list(VAL))] == want[te & (wi == w) & np.isin(K, list(VAL))]).mean()) for w in WS}
    F, Lr, T_ = Fs[False]; sy, nr = te & np.isin(K, SYN_V), te & np.isin(K, DIFF['near miss']); Fu = Fs[True][0]
    r1 = np.array([.5 * np.mean([(Fu[te & (g == c) & (K == t) & (fr == VAL[t][0])] == t).mean() for t in VAL]) + .5 * np.mean([(Fu[te & (g == c) & (K == kk) & (fr == f)] == '').mean() for f, kk in HS]) for c in contents])
    r2 = np.array([np.mean([(F[te & (g == c) & (K == SY[t]) & (fr == VAL[t][0])] == t).mean() for t in VAL]) for c in contents]); hit = {a_: (T_[sy] == want[sy]) & (Lr[sy] > np.quantile(Lr[nr], 1 - a_)) for a_ in RATES}; pf = {a_: np.array([hit[a_][g[sy] == c].mean() for c in contents]) for a_ in RATES}
    q['R1 unseen'], q['R2 rule'], q['near fires'] = float(r1.mean()), float(r2.mean()), float((F[nr] != '').mean()); q['para at near-miss rate'] = {str(a_): float(pf[a_].mean()) for a_ in RATES}; q['AUROC para/near'] = auroc(Lr[sy], Lr[nr])
    return q, dict(r1=r1, r2=r2, **{f'p{a_}': pf[a_] for a_ in RATES})
if __name__ == '__main__':
    out, vec = {}, {}
    for s in STEMS:
        M = pickle.load(open(RUN / f'depth_{s}.pkl', 'rb')); meta, res, L_ = M['meta'], M['res'], M['L']; R = [x['roles'] for x in res]; kinds = M['kinds']; P = {k: [x['st'][k] for x in res] for k in kinds}
        K = np.array([x['kind'] for x in meta]); want = np.array([x['want'] for x in meta], object); wi = np.array([x['wording'] for x in meta]); sp = np.array([x['split'] for x in meta]); fr = np.array([x['frame'] for x in meta]); g = np.array([x['content'] for x in meta]); lab = np.array(['text' if k == 'text' else f'{f}:{k}' for k, f in zip(K, fr)]); L_meta = (K, want, wi, sp, fr, g, lab)
        q = out[s] = dict(L=L_, flow=json.load(open(RUN / f'flow_{s}.json'))) if (RUN / f'flow_{s}.json').exists() else dict(L=L_); vec[s] = {}; MATCH = [('h0', ('h0',))] + [(k, (k,)) for k in kinds if k != 'h0'] + [('path', tuple(f'D{l}' for l in range(L_))), ('wells', tuple(f'W{l}' for l in range(L_))), ('states', tuple(f'h{l}' for l in range(1, L_ + 1)))]
        q['primary'], q['twelve wordings'], q['unless own clouds'] = {}, {}, {}
        for name, ks in MATCH:
            q['primary'][name], vec[s][name] = score(name, P, ks, R, L_meta); fx = (q['primary'][name]['dev'], q['primary'][name]['m'], q['primary'][name]['d'])
            q['twelve wordings'][name], vec[s][name + '|12'] = score(name, P, ks, R, L_meta, fx, drop=None); p, t = q['primary'][name], q['twelve wordings'][name]
            q['unless own clouds'][name], vec[s][name + '|own'] = score(name, P, ks, R, L_meta, fx, drop=None, own=True); o_ = q['unless own clouds'][name]
            print(f"{s} {name:7s} m{p['m']} d{p['d']:2d} dev {p['dev']:.3f} | R1 {p['R1 unseen']:.3f} (12 wordings {t['R1 unseen']:.3f}) | para rule {p['R2 rule']:.3f} at 1% near {p['para at near-miss rate']['0.01']:.3f} near {p['near fires']:.4f} AUROC {p['AUROC para/near']:.3f} | unseen det " + ' '.join(f'{v:.2f}' for v in p['unseen detect'].values()) + ' fires ' + ' '.join(f'{v:.2f}' for v in p['unseen fires on'].values()) + f" | own clouds: R1 {o_['R1 unseen']:.3f} para 1% {o_['para at near-miss rate']['0.01']:.3f} unless known det " + ' '.join(f'{v:.2f}' for v in o_['known unless']['detect'].values()) + ' fires ' + ' '.join(f'{v:.2f}' for v in o_['known unless']['fires_on'].values()) + ' unseen det ' + ' '.join(f'{v:.2f}' for v in o_['unseen unless']['detect'].values()), flush=True)
            if name != 'h0':
                q['primary'][name]['minus h0'] = {k: boot(vec[s][name][k] - vec[s]['h0'][k]) for k in ('r1', 'r2', *(f'p{a_}' for a_ in RATES))}
                q['twelve wordings'][name]['minus h0'] = {k: boot(vec[s][name + '|12'][k] - vec[s]['h0|12'][k]) for k in ('r1', 'r2')}
                q['unless own clouds'][name]['minus h0'] = {k: boot(vec[s][name + '|own'][k] - vec[s]['h0|own'][k]) for k in ('r1', 'r2', 'p0.01')}
        (OUT / f'template_depth_{s}.json').write_text(json.dumps(q, indent=1))
    if len(STEMS) > 1:                                    # gap to h0 at each depth, compared clause by clause, every later model against the first
        a = STEMS[0]; q = out['depth comparison'] = {}
        for b in STEMS[1:]:
            q[f'{b} minus {a}'] = {name: {k: boot((vec[b][name][k] - vec[b]['h0'][k]) - (vec[a][name][k] - vec[a]['h0'][k])) for k in ('r1', 'r2', 'p0.01')} for name in ('path', 'wells', 'states')}
            for name, v in q[f'{b} minus {a}'].items(): print(f'{b} minus {a} {name}', {k: [round(x, 3) for x in vv] for k, vv in v.items()}, flush=True)
        (OUT / 'template_depth_comparison.json').write_text(json.dumps(q, indent=1))
