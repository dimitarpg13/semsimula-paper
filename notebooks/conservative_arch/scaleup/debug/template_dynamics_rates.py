#!/usr/bin/env python3
"""Why the displacement matcher fires on more paraphrases than the embedding (2026-11-25, POST HOC, no call rests on it).
    python3 template_dynamics_rates.py RUNDIR OUTDIR stem DYNAMICS_JSON
Known wordings, every matcher at the (m, d) it chose. For the paraphrases: which gate stops them (no template is the nearest
cloud's, wrong template, rate under rho, outside the reach), and how many fire the right template when the threshold is set on
the test near misses instead (fires on near misses at 0.25%, 1%, 2% and 5%), by the log rate of the best template cloud
against the best rival cloud, without the reach and without rho. Paired over contents against h0."""
import sys
from pathlib import Path
_D = Path(__file__).with_name('template_dynamics_readings.py').read_text(); _D = _D[:_D.index("out, fire = {}, {}")]; exec(_D)
J = json.load(open(STORED))[s]; out = {}; A = {}; sy = te & np.isin(K, SYN_V); nr = te & np.isin(K, DIFF['near miss'])
for name, ks in MATCH[:10]:
    Cs = []
    for k in ks: C = np.stack([x['c'] for x in P[k]]); Cs.append(C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9))
    m, d = J[name]['m'], J[name]['d']; X = np.concatenate([np.stack([coords(x, r, m) for x, r in zip(P[k], R)]) for k in ks], 1)
    F_ = Frame([C[dev] for C in Cs], wi[dev], fr[dev]); c = Clouds(np.c_[X[dev], F_([C[dev] for C in Cs])], lab[dev], want[dev], d); rho = cvf(X[dev], [C[dev] for C in Cs], lab[dev], want[dev], g[dev], wi[dev], fr[dev], d)[0]
    Y = np.c_[X, F_(Cs)]; Rt, t, ins, llr = c.rates(Y); mx = Rt.max(1); Dm = ((((Y - c.mu0) / c.sd) @ c.B)[:, None] - c.C[None]) ** 2; Dm = Dm.sum(2); near_cl = np.array(c.cl, object)[Dm.argmin(1)]
    q = out[name] = dict(rho=rho); w = want[sy]; right = t[sy] == w
    q['paraphrase gates'] = dict(right_template=float(right.mean()), right_and_rho=float((right & (mx[sy] >= rho)).mean()), right_and_reach=float((right & ins[sy]).mean()), all=float((right & (mx[sy] >= rho) & ins[sy]).mean()), inside=float(ins[sy].mean()))
    q['near miss gates'] = dict(rho=float((mx[nr] >= rho).mean()), reach=float(ins[nr].mean()), both=float(((mx[nr] >= rho) & ins[nr]).mean())); q['fallacy of the nearest cloud'] = {k_: float((near_cl[sy] == k_).mean()) for k_ in sorted(set(near_cl[sy])) if (near_cl[sy] == k_).mean() > .05}
    q['at near-miss rate'] = {}
    for a_ in (.0025, .01, .02, .05):
        th = np.quantile(llr[nr], 1 - a_); hit = right & (llr[sy] > th); ca = np.array([hit[g[sy] == x].mean() for x in contents]); A[name, a_] = ca; q['at near-miss rate'][str(a_)] = dict(threshold=float(th), paraphrase=float(hit.mean()), fire_no_reach=float((llr[nr] > th).mean()), by_form={T_: float(hit[w == T_].mean()) for T_ in VAL})
    th = {}; q['by wording known'] = {wd: {T_: float((c.fire(Y[te & (wi == wd) & (K == T_)], rho) == T_).mean()) for T_ in VAL if (te & (wi == wd) & (K == T_)).any()} for wd in WS}
    q['by wording known fallacy'] = {wd: {kk: float((c.fire(Y[te & (wi == wd) & (K == kk)], rho) != '').mean()) for kk in ('AC', 'DA', 'DAnot', 'P', 'Q') if (te & (wi == wd) & (K == kk)).any()} for wd in WS}
    print(name, rho, q['paraphrase gates'], q['near miss gates'], {k_: round(v['paraphrase'], 3) for k_, v in q['at near-miss rate'].items()}, flush=True)
pairs = out['paired to h0'] = {}
for name, _ in MATCH[1:10]:
    for a_ in (.0025, .01, .02, .05):
        pairs[f'{name} at {a_}'] = boot(A[name, a_] - A['h0', a_])[:3]
json.dump(out, open(OUT / f'template_dynamics_rates_{s}.json', 'w'), indent=1)
for k_, v in pairs.items(): print(k_, [round(x, 3) for x in v])
for n in out:
    if n != 'paired to h0' and n in ('h0', 'D1+D2', 'h2', 'h1+h2'): print(n, 'nearest clouds of paraphrases', out[n]['fallacy of the nearest cloud']); print('  DS/cond by wording', out[n]['by wording known']['unless'], out[n]['by wording known']['provided'], out[n]['by wording known fallacy']['unless'])
