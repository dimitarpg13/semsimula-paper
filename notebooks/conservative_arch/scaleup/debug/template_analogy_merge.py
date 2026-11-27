#!/usr/bin/env python3
"""Replace the premises of some levels in analogy_<stem>.pkl by a re-extraction of those levels (2026-11-26).
    python3 template_analogy_merge.py FULL.pkl PART.pkl OUT.pkl
PART.pkl comes from template_analogy_test.py with levels=...; its check premises must equal FULL's (same model, same extraction).
The levels present in PART replace FULL's premises of those levels; everything else is kept as it was."""
import pickle, sys
import numpy as np
F, P = (pickle.load(open(x, 'rb')) for x in sys.argv[1:3])
assert F['kinds'] == P['kinds'] and F['L'] == P['L']
cf = {m['depth_index']: r for m, r in zip(F['meta'], F['res']) if m['split'] == 'check'}
for m, r in zip(P['meta'], P['res']):
    if m['split'] != 'check': continue
    d = max(float(np.abs(r['st'][k]['G'].astype(float) - cf[m['depth_index']]['st'][k]['G'].astype(float)).max()) for k in F['kinds'])
    assert d < 1e-3, (m['depth_index'], d)
lv = {m['level'] for m in P['meta'] if m['split'] == 'analogy'}
keep = [i for i, m in enumerate(F['meta']) if m['level'] not in lv]; add = [i for i, m in enumerate(P['meta']) if m['split'] == 'analogy']
out = dict(meta=[F['meta'][i] for i in keep] + [P['meta'][i] for i in add], res=[F['res'][i] for i in keep] + [P['res'][i] for i in add], L=F['L'], kinds=F['kinds'])
pickle.dump(out, open(sys.argv[3], 'wb')); print('replaced levels', sorted(lv), '| premises', len(F['meta']), '->', len(out['meta']))
