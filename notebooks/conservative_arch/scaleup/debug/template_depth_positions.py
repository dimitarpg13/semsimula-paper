#!/usr/bin/env python3
"""Where each wording's connective sits, at every depth (2026-11-25). The reading of template_connective_position.py on the
files of template_depth_test.py: per vector kind, the connective's mean whitened unit vector along the axis from the disjunctive
to the conditional centroid, fitted on the development premises without "unless"; 0 is the disjunctive, 1 the conditional
centroid; test premises; also each wording left out of its own axis.
    python3 template_depth_positions.py RUNDIR OUTDIR stem"""
import json, pickle, sys
from pathlib import Path
import numpy as np
RUN, OUT, s = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
WC = {'cond': ['if-then', 'when', 'whenever', 'q-if-p', 'assuming', 'provided'], 'disj': ['either-or', 'or', 'at-least-one', 'or-else', 'unless', 'one-holds']}
M = pickle.load(open(RUN / f'depth_{s}.pkl', 'rb')); meta, res = M['meta'], M['res']; wi = np.array([x['wording'] for x in meta]); sp = np.array([x['split'] for x in meta]); dev, te = sp == 'dev', sp == 'test'
WD = [w for w in WC['disj'] if w != 'unless']; WS = WC['cond'] + WC['disj']; u = lambda v: v / np.linalg.norm(v); out = {}
for k in M['kinds']:
    C = np.stack([x['st'][k]['c'].astype(np.float64) for x in res]); C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
    def pos(skip=None):
        cen = {f: u(np.mean([u(C[dev & (wi == w)].mean(0)) for w in ws if w != skip], axis=0)) for f, ws in (('cond', WC['cond']), ('disj', WD))}; a = u(cen['cond'] - cen['disj']); lo, hi = cen['disj'] @ a, cen['cond'] @ a; return (C @ a - lo) / (hi - lo)
    p = pos(); out[k] = {w: float(np.median(p[te & (wi == w)])) for w in WS}; out[k]['left out of the axis'] = {w: float(np.median(pos(w)[te & (wi == w)])) for w in WS}
    print(k, {w: round(v, 2) for w, v in out[k].items() if w in WS}, '| left out', {w: round(v, 2) for w, v in out[k]['left out of the axis'].items()}, flush=True)
(OUT / f'template_depth_positions_{s}.json').write_text(json.dumps(out, indent=1))
