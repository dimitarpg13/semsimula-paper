#!/usr/bin/env python3
"""Hybrid step 1a of companion_notes/semantic_templates/Hybrid_Template_Matchers.md (2026-11-26): a paraphrase gate fitted across
clauses (approach G, linear), on the saved extractions. Written before any result of it was seen.
    python3 template_gate_fit.py RUNDIR OUTDIR stem[,stem...] [CACHEDIR]
RUNDIR holds depth_<stem>.pkl of template_depth_test.py and OUTDIR template_hybrid_check_<stem>.json (step 0), whose (m, d) per
matcher are reused, so nothing is tuned here. The relational coordinates X of each matcher are rebuilt by template_depth_analysis.py's
own score (primary reading, "unless" left out) and, with CACHEDIR, saved as cache_<stem>.npz for the null check.

Coordinates of the gate: the embedding's X (h0), the best single state's (from step 0), the embedding and that state stacked, the
embedding and all states stacked. Each block is standardised on the fitting half. Test clauses are split by content into two halves
(the even and odd contents, as step 0's cross-fit); the gate is fitted on one half and scored on the other, then the halves swap.
The covariance is the pooled within-kind covariance of the fitting half (frame and kind, all kinds but ordinary text), shrunk 10%
toward its diagonal, as step 0. For each template t the direction is w = S^-1 (mean of p minus mean of q) on the fitting half.
  1. per pair (MPsyn against near, MTsyn against nearQn, DS1syn against nearn, DS2syn against nearQn, in the template's frame): the
     share of held-out paraphrases above the 99th percentile of the held-out near misses of that kind, and the held-out AUROC;
     beside it the Gaussian prediction from step 0's cross-fitted separation, Phi(Delta - 2.326), and the in-sample catch of the
     direction fitted on all test clauses (both use the test labels, so they bound the held-out number from above);
  2. pooled, comparable with the matcher's "paraphrases at 1% near misses": a premise is caught when the embedding matcher's
     template T is the one wanted and the gate score of T, z_T (w_T x standardised by the near misses of T on the fitting half), is
     above the 99th percentile of the held-out near misses' own z_T (all four near-miss kinds, both frames). Reported with the share
     of paraphrases whose T is right (the ceiling of this rule), and the pass rates at the same threshold of the exact forms (in
     frame), the fallacies (AC, DA, DAnot, P, Q and their paraphrases), the other-subject kinds and the unrelated kinds;
  3. two gates: narrow (p = the paraphrase kind, q = its near miss: what step 0 measured) and broad (p = the exact form and its
     paraphrase, q = every other kind of the frame but ordinary text: an identity gate).
Expectation, stated before the run: the held-out catch per pair keeps at least half of the Gaussian prediction; the states raise it
on every model; the pooled narrow catch with the embedding alone is at least 0.40 (the matcher 0.18 to 0.30); the narrow gate lets
more than 10% of the fallacies through, the broad gate fewer.
Writes OUTDIR/template_gate_fit_<stem>.json and prints a table."""
import json, pickle, sys
from pathlib import Path
_S = Path(__file__).with_name('template_depth_analysis.py').read_text(); _S = _S[:_S.rindex("if __name__ == '__main__':")]
_HOOK = "q['AUROC para/near'] = auroc(Lr[sy], Lr[nr])"
assert _S.count(_HOOK) == 1
_S = _S.replace(_HOOK, _HOOK + "\n    if drop == 'unless' and not own: CAP[name] = dict(Lr=Lr, T=T_, X=X, m=m, d=d)")
CAP = {}
CACHE = Path(sys.argv[4]) if len(sys.argv) > 4 else None
sys.argv = sys.argv[:4]
exec(_S)
from scipy.stats import norm

PAIRS = [('MP', 'cond', 'MPsyn', 'near'), ('MT', 'cond', 'MTsyn', 'nearQn'), ('DS1', 'disj', 'DS1syn', 'nearn'), ('DS2', 'disj', 'DS2syn', 'nearQn')]
FAMILY = {'exact form': None, 'fallacy': ('AC', 'DA', 'DAnot', 'P', 'Q', 'ACsyn', 'DAsyn', 'Psyn'), 'other subject': DIFF['other subject'], 'unrelated': DIFF['unrelated']}
Z99 = norm.ppf(.99)


def stand(X, fit):
    mu, sd = X[fit].mean(0), X[fit].std(0) + 1e-9; return (X - mu) / sd


def pooled_cov(Z, grp, fit):
    W = [Z[fit & (grp == k)] - Z[fit & (grp == k)].mean(0) for k in np.unique(grp[fit])]
    S = np.cov(np.concatenate(W).T); return .9 * S + .1 * np.diag(np.diag(S))


def above(s, pos, neg, a_=.01):
    return float((s[pos] > np.quantile(s[neg], 1 - a_)).mean())


if __name__ == '__main__':
    out = {}
    for s in STEMS:
        M = pickle.load(open(RUN / f'depth_{s}.pkl', 'rb')); meta, res, L_ = M['meta'], M['res'], M['L']; R = [x['roles'] for x in res]; P = {k: [x['st'][k] for x in res] for k in M['kinds']}
        K = np.array([x['kind'] for x in meta]); want = np.array([x['want'] for x in meta], object); wi = np.array([x['wording'] for x in meta]); sp = np.array([x['split'] for x in meta]); fr = np.array([x['frame'] for x in meta]); g = np.array([x['content'] for x in meta]); lab = np.array(['text' if k == 'text' else f'{f}:{k}' for k, f in zip(K, fr)]); L_meta = (K, want, wi, sp, fr, g, lab)
        s0 = json.loads((OUT / f'template_hybrid_check_{s}.json').read_text()); best = s0['best_single']
        h0md = json.loads((OUT / f'template_depth_{s}.json').read_text())['primary']['h0']
        fixed = {'h0': (h0md['m'], h0md['d']), best: (s0['readings'][best]['m'], s0['readings'][best]['d']), 'states': (s0['readings']['states']['m'], s0['readings']['states']['d'])}
        ks_of = {'h0': ('h0',), best: (best,), 'states': tuple(f'h{l}' for l in range(1, L_ + 1))}
        for name, (m, d) in fixed.items():
            score(name, P, ks_of[name], R, L_meta, (float('nan'), m, d)); print(f'   [{s} {name} rebuilt: m {m} d {d}, X {CAP[name]["X"].shape[1]} coordinates]', flush=True)
        if CACHE:
            CACHE.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(CACHE / f'cache_{s}.npz', best=best, **{f'X_{n}': CAP[n]['X'] for n in fixed}, **{f'T_{n}': CAP[n]['T'].astype(str) for n in fixed}, **{f'Lr_{n}': CAP[n]['Lr'] for n in fixed}, K=K, want=want.astype(str), wi=wi, sp=sp, fr=fr, g=g)
        te = (sp == 'test') & (wi != 'unless'); nt = te & (K != 'text'); grp = np.array([f'{f}:{k}' for f, k in zip(fr, K)])
        cs = np.unique(g[nt]); halves = [nt & np.isin(g, cs[::2]), nt & np.isin(g, cs[1::2])]
        sy, nr = te & np.isin(K, SYN_V), te & np.isin(K, DIFF['near miss']); T0 = CAP['h0']['T']
        READ = {'h0': ['h0'], best: [best], f'h0+{best}': ['h0', best], 'h0+states': ['h0', 'states']}
        o = out[s] = dict(L=L_, best_single=best, clauses=int(len(cs)), right_template_on_paraphrases=float((T0[sy] == want[sy]).mean()), matcher_para_1pct={n: s0['readings'][n]['para_1pct'] for n in (best, 'states')}, readings={})
        print(f'\n{s} (L = {L_}): {len(cs)} test clauses, halves {halves[0].sum()} / {halves[1].sum()} premises; embedding matcher T right on {o["right_template_on_paraphrases"]:.3f} of paraphrases')
        for rn, blocks in READ.items():
            Xr = np.concatenate([CAP[b]['X'] for b in blocks], 1); r = o['readings'][rn] = dict(dim=int(Xr.shape[1]), pairs={}, pooled={})
            # direction fitted on all test clauses, scored in sample: the label-using bound
            Zall = stand(Xr, nt); Sall = pooled_cov(Zall, grp, nt)
            for gate in ('narrow', 'broad'):
                zT = {h: np.zeros(len(K)) for h in (0, 1)}; per = {t: dict(held=[], auroc=[]) for t, *_ in PAIRS}
                for h in (0, 1):
                    fit, sc_ = halves[h], halves[1 - h]; Z = stand(Xr, fit); S = pooled_cov(Z, grp, fit); Si = np.linalg.inv(S); zt = np.zeros((len(K), 4))
                    for j, (t, f, kp, kq) in enumerate(PAIRS):
                        if gate == 'narrow': pos, neg = (K == kp) & (fr == f), (K == kq) & (fr == f)
                        else: pos, neg = np.isin(K, (t, kp)) & (fr == f), ~np.isin(K, (t, kp, 'text')) & (fr == f)
                        w = Si @ (Z[fit & pos].mean(0) - Z[fit & neg].mean(0)); v = Z @ w; qv = v[fit & (K == kq) & (fr == f)]; zt[:, j] = (v - qv.mean()) / (qv.std() + 1e-9)
                        if gate == 'narrow':
                            p_, n_ = sc_ & (K == kp) & (fr == f), sc_ & (K == kq) & (fr == f); per[t]['held'].append(above(v, p_, n_)); per[t]['auroc'].append(auroc(v[p_], v[n_]))
                    ti = {t: j for j, (t, *_) in enumerate(PAIRS)}; idx = np.array([ti.get(x, 0) for x in T0]); zT[h] = zt[np.arange(len(K)), idx]   # T is blank on "unless", never scored
                if gate == 'narrow':
                    for j, (t, f, kp, kq) in enumerate(PAIRS):
                        pos, neg = (K == kp) & (fr == f), (K == kq) & (fr == f); w = np.linalg.solve(Sall, Zall[nt & pos].mean(0) - Zall[nt & neg].mean(0)); v = Zall @ w
                        rs = s0['readings'][best if rn in (best, f'h0+{best}') else 'states']['residual_separation'][t]; d2 = rs['emb'] if rn == 'h0' else (np.nan if rn == best else rs['joint'])
                        r['pairs'][t] = dict(held_1pct=float(np.mean(per[t]['held'])), held_auroc=float(np.mean(per[t]['auroc'])), in_sample_1pct=above(v, nt & pos, nt & neg), gauss_pred=float(norm.cdf(np.sqrt(max(d2, 0)) - Z99)) if np.isfinite(d2) else None, step0_sq_sep=None if not np.isfinite(d2) else float(d2))
                # pooled: each half scored by the gate fitted on the other
                zs = np.where(halves[0], zT[1], np.where(halves[1], zT[0], np.nan)); thr = {h: np.quantile(zs[halves[h] & nr], .99) for h in (0, 1)}
                pas = np.zeros(len(K), bool)
                for h in (0, 1): pas |= halves[h] & (zs > thr[h])
                inf = lambda m_: float(pas[m_].mean()); ex = te & np.isin(K, list(VAL)) & np.array([VAL.get(k, ('', ''))[0] == f for k, f in zip(K, fr)])
                r['pooled'][gate] = dict(para_1pct=float(((T0 == want) & pas)[sy].mean()), near_pass=inf(nr), exact_pass=inf(ex), exact_caught=float(((T0 == want) & pas)[ex].mean()),
                                         **{f'{k}_pass': inf(te & np.isin(K, v_)) for k, v_ in FAMILY.items() if v_ is not None},
                                         by_kind={kp: float(((T0 == want) & pas)[te & (K == kp)].mean()) for kp in SYN_V})
            pr = r['pairs']; pn, pb = r['pooled']['narrow'], r['pooled']['broad']
            print(f'   {rn:12s} dim {r["dim"]:4d} | per pair held-out @1% (AUROC) [in-sample, Gauss]: ' + '  '.join(f'{t} {v["held_1pct"]:.2f} ({v["held_auroc"]:.3f}) [{v["in_sample_1pct"]:.2f}, ' + ('-' if v['gauss_pred'] is None else f'{v["gauss_pred"]:.2f}') + ']' for t, v in pr.items()))
            for gn, pg in (('narrow', pn), ('broad', pb)):
                print(f'   {"":12s} pooled {gn:6s}: para @1% {pg["para_1pct"]:.3f} | pass: near {pg["near_pass"]:.4f} exact {pg["exact_pass"]:.3f} fallacy {pg["fallacy_pass"]:.3f} other subj {pg["other subject_pass"]:.3f} unrelated {pg["unrelated_pass"]:.3f}', flush=True)
        (OUT / f'template_gate_fit_{s}.json').write_text(json.dumps(out[s], indent=1))
