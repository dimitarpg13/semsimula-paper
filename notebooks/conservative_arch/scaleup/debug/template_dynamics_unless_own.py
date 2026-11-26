#!/usr/bin/env python3
"""The SR2-2B dynamics test with "unless" given clouds of its own that feed the existing templates (2026-11-25, POST HOC for SR2-2B;
the same reading as the third one of template_depth_analysis.py, so the depth arms can be compared with it).
    python3 template_dynamics_unless_own.py RUNDIR OUTDIR stem DYNAMICS_JSON
Every matcher keeps the (m, d) it chose. Writes OUTDIR/template_dynamics_unless_own_<stem>.json."""
import sys
from pathlib import Path
_D = Path(__file__).with_name('template_dynamics_readings.py').read_text(); _D = _D[:_D.index("out, fire = {}, {}")]; exec(_D)
_S = Path(__file__).with_name('template_depth_analysis.py').read_text(); exec(_S[_S.index("HS = [('cond', 'AC')"):_S.rindex("if __name__ == '__main__':")])
J = json.load(open(STORED))[s]; LM = (K, want, wi, sp, fr, g, lab); out, vec = {}, {}
ONLY = sys.argv[5].split(',') if len(sys.argv) > 5 else None
for name, ks in [x for x in MATCH[:10] if ONLY is None or x[0] in ONLY or x[0] == 'h0']:
    fx = (J[name]['dev'], J[name]['m'], J[name]['d']); out[name], vec[name] = score(name, P, ks, R, LM, fx, drop=None, own=True)
    if name != 'h0': out[name]['minus h0'] = {k: boot(vec[name][k] - vec['h0'][k]) for k in ('r1', 'r2', 'p0.01')}
    o = out[name]; print(name, 'R1', round(o['R1 unseen'], 3), 'para rule', round(o['R2 rule'], 3), '1%', round(o['para at near-miss rate']['0.01'], 3), 'near', round(o['near fires'], 4), '| unseen fires', {k: round(v, 3) for k, v in o['unseen fires on'].items()}, '| unless known', o['known unless']['detect'], o['known unless']['fires_on'], 'unseen', o['unseen unless']['detect'], o['unseen unless']['which_on'], o.get('minus h0', {}).get('r1'), flush=True)
    json.dump(out, open(OUT / f'template_dynamics_unless_own_{s}.json', 'w'), indent=1)
