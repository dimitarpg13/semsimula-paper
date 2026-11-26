#!/usr/bin/env python3
"""Step 0 of companion_notes/semantic_templates/Hybrid_Template_Matchers.md (2026-11-26): do the embedding reading and the layer
readings catch different paraphrases?
    python3 template_hybrid_check.py RUNDIR OUTDIR stem[,stem...]
RUNDIR holds depth_<stem>.pkl of template_depth_test.py. Each matcher is fitted exactly as the primary reading of
template_depth_analysis.py (without "unless", known wordings, its own (m, d) from the development search); this script only
reads what that fit already computes, per test premise. Matchers: the embedding (h0) and three layer readings: the single state
that catches the most paraphrases at 1% near misses, the joint states, the path. For each layer reading against h0:
  1. paraphrases (MPsyn, MTsyn, DS1syn, DS2syn) caught at each matcher's own 1% near-miss threshold (right template and log-odds
     above the 99th percentile of the near misses): embedding only, layer reading only, both, neither;
  2. the near misses that pass those thresholds, and how many pass both;
  3. "either fires" at a matched rate: both thresholds set at one common near-miss rate a, a chosen so that 1% of near misses
     pass at least one; the paraphrases caught then (score fusion by the larger of two, Hybrid note approach A);
  4. the Spearman rank correlation of the two log-odds over the paraphrases and near misses (the note's s0 against s_l rank
     test, read on the matchers' scores instead of on one coordinate);
  5. the residual separation (note §3.3) of the layer reading's relational coordinates given the embedding's, for each
     paraphrase kind against its near miss in the same frame, and for MP against the affirmed disjunct as a control. The
     separation is cross-fitted: the class-mean differences from two halves of the test clauses, multiplied through the pooled
     within-kind covariance (shrunk 10% toward its diagonal), so the estimate has no upward bias from the noise of one mean.
Writes OUTDIR/template_hybrid_check_<stem>.json and prints a table."""
import json, pickle, sys
from pathlib import Path
_S = Path(__file__).with_name('template_depth_analysis.py').read_text(); _S = _S[:_S.rindex("if __name__ == '__main__':")]
_HOOK = "q['AUROC para/near'] = auroc(Lr[sy], Lr[nr])"
assert _S.count(_HOOK) == 1
_S = _S.replace(_HOOK, _HOOK + "\n    if drop == 'unless' and not own: CAP[name] = dict(Lr=Lr, T=T_, X=X, m=m, d=d)")
CAP = {}
exec(_S)
from scipy.stats import spearmanr

PAIRS = [('MP', 'cond', 'MPsyn', 'near'), ('MT', 'cond', 'MTsyn', 'nearQn'), ('DS1', 'disj', 'DS1syn', 'nearn'), ('DS2', 'disj', 'DS2syn', 'nearQn')]


def residual_sep(Xa, Xb, K, fr, g, sel_p, sel_q, te, kinds_mask):
    """Cross-fitted squared Mahalanobis separation of p against q in a, in (a, b), and the residual (a, b) minus a."""
    Z = np.c_[Xa, Xb]; mu, sd = Z[te].mean(0), Z[te].std(0) + 1e-9; Z = (Z - mu) / sd
    grp = np.array([f'{f}:{k}' for f, k in zip(fr, K)]); W = []
    for k in np.unique(grp[kinds_mask]):
        r = kinds_mask & (grp == k); W.append(Z[r] - Z[r].mean(0))
    W = np.concatenate(W); S = np.cov(W.T); S = .9 * S + .1 * np.diag(np.diag(S))
    cs = np.unique(g[te]); half = np.isin(g, cs[::2])
    da = [Z[sel_p & h].mean(0) - Z[sel_q & h].mean(0) for h in (half, ~half)]
    na = Xa.shape[1]
    def sep(idx):
        Si = np.linalg.inv(S[np.ix_(idx, idx)]); return float(da[0][idx] @ Si @ da[1][idx])
    a, ab = sep(np.arange(na)), sep(np.arange(Z.shape[1]))
    return a, ab, ab - a


if __name__ == '__main__':
    out = {}
    for s in STEMS:
        M = pickle.load(open(RUN / f'depth_{s}.pkl', 'rb')); meta, res, L_ = M['meta'], M['res'], M['L']; R = [x['roles'] for x in res]; kinds = M['kinds']; P = {k: [x['st'][k] for x in res] for k in kinds}
        K = np.array([x['kind'] for x in meta]); want = np.array([x['want'] for x in meta], object); wi = np.array([x['wording'] for x in meta]); sp = np.array([x['split'] for x in meta]); fr = np.array([x['frame'] for x in meta]); g = np.array([x['content'] for x in meta]); lab = np.array(['text' if k == 'text' else f'{f}:{k}' for k, f in zip(K, fr)]); L_meta = (K, want, wi, sp, fr, g, lab)
        te = (sp == 'test') & (wi != 'unless'); sy, nr = te & np.isin(K, SYN_V), te & np.isin(K, DIFF['near miss'])
        singles = [f'h{l}' for l in range(1, L_ + 1)]; MATCH = {'h0': ('h0',), 'states': tuple(singles), 'path': tuple(f'D{l}' for l in range(L_))}
        MATCH.update({k: (k,) for k in singles})
        q = {}
        for name, ks in MATCH.items():
            q[name] = score(name, P, ks, R, L_meta)[0]; print(f'   [{s} {name} fitted: m {CAP[name]["m"]} d {CAP[name]["d"]}, paraphrases at 1% {q[name]["para at near-miss rate"]["0.01"]:.3f}]', flush=True)
        best = max(singles, key=lambda k: q[k]['para at near-miss rate']['0.01'])
        def caught(name, a_=.01):
            c = CAP[name]; thr = np.quantile(c['Lr'][nr], 1 - a_); return (c['T'] == want) & (c['Lr'] > thr), c['Lr'] > thr
        o = out[s] = dict(L=L_, best_single=best, readings={})
        h0p, h0n = caught('h0')
        print(f'\n{s} (L = {L_}): paraphrases {sy.sum()}, near misses {nr.sum()}, test clauses {len(np.unique(g[te]))}')
        print(f'   {"reading":22s} {"emb only":>9s} {"layer only":>10s} {"both":>7s} {"neither":>8s} | {"near: emb":>9s} {"layer":>7s} {"both":>6s} | {"either @1%":>10s} {"emb @1%":>8s} {"layer @1%":>9s} | {"Spearman":>8s}')
        for name in (best, 'states', 'path'):
            lp, ln_ = caught(name)
            e_only, l_only, both = (h0p & ~lp)[sy].mean(), (lp & ~h0p)[sy].mean(), (h0p & lp)[sy].mean()
            # either fires at a matched 1% near-miss rate
            lo_, hi_ = .0, .01
            for _ in range(40):
                a_ = (lo_ + hi_) / 2; (pa, na_), (pb, nb_) = caught('h0', a_), caught(name, a_)
                lo_, hi_ = (a_, hi_) if (na_ | nb_)[nr].mean() < .01 else (lo_, a_)
            (pa, na_), (pb, nb_) = caught('h0', lo_), caught(name, lo_); either = (pa | pb)[sy].mean()
            rho = spearmanr(CAP['h0']['Lr'][sy | nr], CAP[name]['Lr'][sy | nr]).statistic
            seps = {}
            kmask = te & (K != 'text')
            for t, f, ks_, kq in PAIRS:
                seps[t] = residual_sep(CAP['h0']['X'], CAP[name]['X'], K, fr, g, te & (K == ks_) & (fr == f), te & (K == kq) & (fr == f), te, kmask)
            seps['control MP vs P'] = residual_sep(CAP['h0']['X'], CAP[name]['X'], K, fr, g, te & (K == 'MP') & (fr == 'cond'), te & (K == 'P') & (fr == 'disj'), te, kmask)
            o['readings'][name] = dict(m=CAP[name]['m'], d=CAP[name]['d'], para_1pct=q[name]['para at near-miss rate']['0.01'], emb_only=float(e_only), layer_only=float(l_only), both=float(both),
                                       neither=float(1 - e_only - l_only - both), near_emb=float(h0n[nr].mean()), near_layer=float(ln_[nr].mean()), near_both=float((h0n & ln_)[nr].mean()),
                                       either_matched=float(either), either_rate_each=float(lo_), spearman=float(rho),
                                       residual_separation={k: dict(emb=v[0], joint=v[1], residual=v[2]) for k, v in seps.items()})
            print(f'   {name + (" (best single)" if name == best else ""):22s} {e_only:9.3f} {l_only:10.3f} {both:7.3f} {1 - e_only - l_only - both:8.3f} | {h0n[nr].mean():9.4f} {ln_[nr].mean():7.4f} {(h0n & ln_)[nr].mean():6.4f} | '
                  f'{either:10.3f} {q["h0"]["para at near-miss rate"]["0.01"]:8.3f} {q[name]["para at near-miss rate"]["0.01"]:9.3f} | {rho:8.3f}')
        print('   residual separation, squared Mahalanobis, cross-fitted (embedding / embedding + layer / residual):')
        for name in (best, 'states', 'path'):
            rs = o['readings'][name]['residual_separation']
            print(f'   {name:10s} ' + '  '.join(f'{k}: {v["emb"]:.1f} / {v["joint"]:.1f} / {v["residual"]:+.1f}' for k, v in rs.items()))
        (OUT / f'template_hybrid_check_{s}.json').write_text(json.dumps(o, indent=1))
