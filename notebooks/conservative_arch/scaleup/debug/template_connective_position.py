#!/usr/bin/env python3
"""Where the connective of each wording sits between the disjunctive and the conditional wordings (2026-11-25, POST HOC).
    python3 template_connective_position.py RUNDIR OUTDIR stem DYNAMICS_JSON
Per particle vector: the connective's mean whitened unit vector along the axis from the disjunctive to the conditional centroid, the
axis fitted on the development premises without "unless"; 0 is the disjunctive centroid, 1 the conditional one. Test premises."""
import sys
from pathlib import Path
_D = Path(__file__).with_name('template_dynamics_readings.py').read_text(); _D = _D[:_D.index("out, fire = {}, {}")]; exec(_D)
out = {}; WD = [w for w in WC['disj'] if w != 'unless']; u = lambda v: v / np.linalg.norm(v)
for k in ('h0', 'D1', 'h1', 'D2', 'h2', 'W1', 'W2'):
    C = np.stack([x['c'] for x in P[k]]); C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9); cen = {f_: u(np.mean([u(C[dev & (wi == w)].mean(0)) for w in ws_], axis=0)) for f_, ws_ in (('cond', WC['cond']), ('disj', WD))}
    a = u(cen['cond'] - cen['disj']); lo, hi = cen['disj'] @ a, cen['cond'] @ a; pos = (C @ a - lo) / (hi - lo); out[k] = {w: [float(x) for x in np.percentile(pos[te & (wi == w)], [25, 50, 75])] for w in WS}
    # held out: leave each other wording out of the axis and read its position
    loo = {}
    for w in WS:
        if w == 'unless': loo[w] = out[k][w][1]; continue
        cc = {f_: u(np.mean([u(C[dev & (wi == q_)].mean(0)) for q_ in ws_ if q_ != w], axis=0)) for f_, ws_ in (('cond', WC['cond']), ('disj', WD))}; b = u(cc['cond'] - cc['disj']); l_, h_ = cc['disj'] @ b, cc['cond'] @ b; loo[w] = float(np.median((C[te & (wi == w)] @ b - l_) / (h_ - l_)))
    out[k]['left out of the axis'] = loo; print(k, {w: round(v[1], 2) for w, v in out[k].items() if w in WS}, '| left out', {w: round(v, 2) for w, v in loo.items()}, flush=True)
json.dump(out, open(OUT / f'template_connective_position_{s}.json', 'w'), indent=1)
