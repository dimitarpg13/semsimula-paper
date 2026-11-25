#!/usr/bin/env python3
"""The dynamics test with the wording "unless" left out of fitting and scoring (2026-11-25, POST HOC, no call rests on it).
    python3 template_dynamics_without_unless.py RUNDIR OUTDIR stem DYNAMICS_JSON
Every matcher keeps the (m, d) it chose; clouds, axis and threshold are refitted on the eleven other wordings."""
import sys
from pathlib import Path
_D = Path(__file__).with_name('template_dynamics_readings.py').read_text(); _D = _D[:_D.index("out, fire = {}, {}")]; exec(_D)
WC['disj'].remove('unless'); WS = sum(WC.values(), []); dev &= wi != 'unless'; te &= wi != 'unless'; J = json.load(open(STORED))[s]; out, fire = {}, {}
for name, ks in MATCH[:10]:
    Cs = []
    for k in ks: C = np.stack([x['c'] for x in P[k]]); Cs.append(C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9))
    m, d = J[name]['m'], J[name]['d']; X = np.concatenate([np.stack([coords(x, r, m) for x, r in zip(P[k], R)]) for k in ks], 1); q = out[name] = dict(m=m, d=d); fire[name] = {}
    for un in (False, True):
        F = np.full(len(K), '', object)
        for h in (WS if un else [None]):
            a = dev & (wi != h); F_ = Frame([C[a] for C in Cs], wi[a], fr[a]); c = Clouds(np.c_[X[a], F_([C[a] for C in Cs])], lab[a], want[a], d); rho = cvf(X[a], [C[a] for C in Cs], lab[a], want[a], g[a], wi[a], fr[a], d)[0]
            b = te & ((wi == h) | (stw & (h == 'if-then')) if un else True); F[b] = c.fire(np.c_[X[b], F_([C[b] for C in Cs])], rho)
        if not un: q['rho'] = rho; q['near fires'] = float((F[te & np.isin(K, DIFF['near miss'])] != '').mean())
        fire[name][un] = F; tag = 'unseen' if un else 'known'; q[f'{tag} detect'] = {t: float((F[te & (K == t)] == t).mean()) for t in VAL}; q[f'{tag} fires on'] = {f'{f}:{kk}': float((F[te & (K == kk) & (fr == f)] != '').mean()) for f, kk in HS}
    q['R1 unseen'], q['R2 known'] = float(r1(fire[name][True]).mean()), float(r2(fire[name][False]).mean())
    if name != 'h0': q['R1 minus h0'], q['R2 minus h0'] = boot(r1(fire[name][True]) - r1(fire['h0'][True]))[:3], boot(r2(fire[name][False]) - r2(fire['h0'][False]))[:3]
    print(name, 'rho', round(q['rho'], 3), 'near', round(q['near fires'], 4), 'R1', round(q['R1 unseen'], 3), 'R2', round(q['R2 known'], 3), 'known', {t: round(v, 2) for t, v in q['known detect'].items()}, 'unseen', {t: round(v, 2) for t, v in q['unseen detect'].items()}, {k_: round(v, 3) for k_, v in q['unseen fires on'].items()}, q.get('R1 minus h0'), q.get('R2 minus h0'), flush=True)
json.dump(out, open(OUT / f'template_dynamics_without_unless_{s}.json', 'w'), indent=1)
