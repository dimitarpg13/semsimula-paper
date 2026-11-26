#!/usr/bin/env python3
"""Why the fitted gate (template_gate_fit.py) falls short of step 0's Gaussian prediction (2026-11-26, written after its result).
    python3 template_gate_diag.py CACHEDIR OUTDIR stem[,stem...]
CACHEDIR holds cache_<stem>.npz written by template_gate_fit.py. For each paraphrase pair (as template_gate_fit.py) and reading,
the direction w = S^-1 delta fitted on all test clauses (pooled within-kind covariance S shrunk 10% toward its diagonal, as step 0):
  sq_sep        delta' S^-1 delta, in sample (step 0 reported the cross-fitted value of the same quantity)
  ratio_p/q     the variance of the paraphrases / near misses along w, over w' S w: 1 if the two kinds spread like the pooled kinds
  sq_sep_emp    the squared separation the two kinds actually have along w: (mean gap / pooled sd of the two kinds)^2
  z99_q         the 99th percentile of the near misses along w, in their own sd above their mean (2.326 for a Gaussian)
  by_content    the share of the two kinds' variance along w that lies between clauses (content means), the rest within
and for step 0's control pair (MP against the affirmed disjunct P); and the matcher's own per pair reading from its log-odds: AUROC, and the share caught at 1% near misses of that kind (right
template and log-odds above the 99th percentile). Writes OUTDIR/template_gate_diag_<stem>.json and prints a table."""
import json, sys
from pathlib import Path
import numpy as np
CACHE, OUT, STEMS = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3].split(',')
PAIRS = [('MP', 'cond', 'MPsyn', 'near'), ('MT', 'cond', 'MTsyn', 'nearQn'), ('DS1', 'disj', 'DS1syn', 'nearn'), ('DS2', 'disj', 'DS2syn', 'nearQn')]
CONTROL = ('control', ('cond', 'MP'), ('disj', 'P'))               # step 0's control pair: modus ponens against the affirmed disjunct


def auroc(a, b):
    r = np.argsort(np.argsort(np.r_[a, b])) + 1; return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def between_share(v, g):
    m = np.array([v[g == c].mean() for c in g]); return float(m.var() / v.var())


for s in STEMS:
    C = np.load(CACHE / f'cache_{s}.npz', allow_pickle=True); best = str(C['best']); K, want, wi, sp, fr, g = (C[k] for k in ('K', 'want', 'wi', 'sp', 'fr', 'g'))
    te = (sp == 'test') & (wi != 'unless'); nt = te & (K != 'text'); grp = np.char.add(np.char.add(fr.astype(str), ':'), K.astype(str))
    READ = {'h0': ['h0'], best: [best], f'h0+{best}': ['h0', best], 'h0+states': ['h0', 'states']}; o = {'matcher': {}, 'readings': {}}
    for t, f, kp, kq in PAIRS:
        p, q = nt & (K == kp) & (fr == f), nt & (K == kq) & (fr == f); Lr, T = C['Lr_h0'], C['T_h0']
        o['matcher'][t] = dict(auroc=auroc(Lr[p], Lr[q]), caught_1pct=float(((T == t) & (Lr > np.quantile(Lr[q], .99)))[p].mean()))
    print(f'\n{s}: matcher (embedding) per pair  ' + '  '.join(f'{t} AUROC {v["auroc"]:.3f} @1% {v["caught_1pct"]:.2f}' for t, v in o['matcher'].items()))
    for rn, blocks in READ.items():
        X = np.concatenate([C[f'X_{b}'] for b in blocks], 1); Z = (X - X[nt].mean(0)) / (X[nt].std(0) + 1e-9)
        W = np.concatenate([Z[nt & (grp == k)] - Z[nt & (grp == k)].mean(0) for k in np.unique(grp[nt])]); S = np.cov(W.T); S = .9 * S + .1 * np.diag(np.diag(S))
        r = o['readings'][rn] = {}
        for t, (fp, kp), (fq, kq) in [(t, (f, kp), (f, kq)) for t, f, kp, kq in PAIRS] + [CONTROL]:
            p, q = nt & (K == kp) & (fr == fp), nt & (K == kq) & (fr == fq); dl = Z[p].mean(0) - Z[q].mean(0); w = np.linalg.solve(S, dl); v = Z @ w; ws = float(w @ S @ w)
            sd2 = .5 * (v[p].var() + v[q].var()); vq = (v[q] - v[q].mean()) / v[q].std()
            r[t] = dict(sq_sep=float(dl @ w), ratio_p=float(v[p].var() / ws), ratio_q=float(v[q].var() / ws), sq_sep_emp=float((v[p].mean() - v[q].mean()) ** 2 / sd2),
                        z99_q=float(np.quantile(vq, .99)), by_content_p=between_share(v[p], g[p]), by_content_q=between_share(v[q], g[q]), auroc=auroc(v[p], v[q]))
        print(f'   {rn:11s} ' + '  '.join(f'{t}: sep {v["sq_sep"]:5.1f} emp {v["sq_sep_emp"]:4.1f} ratio p {v["ratio_p"]:4.1f} q {v["ratio_q"]:4.1f} z99 {v["z99_q"]:.2f} between p {v["by_content_p"]:.2f} q {v["by_content_q"]:.2f}' for t, v in r.items()))
    (OUT / f'template_gate_diag_{s}.json').write_text(json.dumps(o, indent=1))
