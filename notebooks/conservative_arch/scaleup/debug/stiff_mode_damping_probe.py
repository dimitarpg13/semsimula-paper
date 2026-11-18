#!/usr/bin/env python3
"""Does the oscillation of the stiff modes carry information? Damp it at evaluation, keep the pull (2026-11-16,
protocol p-bank-memory-conjecture).

    python3 stiff_mode_damping_probe.py OUT FOLDER [ckpt=<file>] [arms=<arm>;<arm>...] [windows=16] [threads=8]
                                        [val=<tokens.npy>] [name=<stem>]

An arm is <zeta>@<layers>: zeta a damping ratio or `none`, layers `all` or a comma list (`0`, `1`). On those layers
the run sets cfg.stiff_mode_damping_ratio = zeta for the layer step (SR3-crit's switch, read at every forward):
the retained stiff modes, span(U) of V_theta's low-rank Hessian, get gamma_i = gamma_0 + 2 zeta omega_i inside
the exact damped flow; the force and the complement are unchanged. The other layers keep the trained value.
`base` is the model as trained. Default arms: SR2's dose and layer split; for SR3-crit pass none@all;none@0;none@1
(the off-distribution control: the oscillation put back on a model trained without it).

Writes OUT/damping_probe_<stem>.txt and .json: per arm the loss, perplexity, the change against base in nats
with its standard error over the windows (paired), and the largest logit change against base.
"""
import json, sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(sys.argv[1]).resolve(); FOLDER = sys.argv[2]
OPTS = dict(a.split('=', 1) for a in sys.argv[3:] if '=' in a)
DBG = Path(__file__).resolve().parent
sys.path.insert(0, str(DBG))

# copy_split's own Cell-0-from-tag build and validation windows, without its command dispatch (as vtheta_well_probe)
src = (DBG / 'copy_split.py').read_text()
src = src[:src.index("\nif MODE == 'nll':")]
sys.argv = ['copy_split.py', str(OUT), 'probe', f"windows={OPTS.get('windows', 16)}", f"threads={OPTS.get('threads', 8)}"] \
    + ([f"val={OPTS['val']}"] if 'val' in OPTS else [])
CS = {'__name__': 'copy_split', '__file__': str(DBG / 'copy_split.py')}
exec(compile(src, str(DBG / 'copy_split.py'), 'exec'), CS)

import verify_pm_switch as P
_build = P.build; got = {}
def build(*a, **k):
    r = _build(*a, **k); got['model'] = r[0]; return r
P.build = build
_, desc = CS['ladder_model'](FOLDER, OPTS.get('ckpt'))
model = got['model']
print('model:', desc, flush=True)
assert getattr(model.cfg, 'lowrank_damped_flow', False) and not getattr(model.cfg, 'first_order_overdamped', False), \
    'needs a second-order arm with the exact damped flow'
assert getattr(model, 'damping_head', None) is None, 'learned gamma(h) overrides the ratio'
TRAINED = model.cfg.stiff_mode_damping_ratio

DEFAULT = 'base;0@all;0.1@all;0.3@all;1@all;0.1@0;0.3@0;1@0;0.1@1;0.3@1;1@1'
ARMS = [a for a in OPTS.get('arms', DEFAULT).split(';') if a]
if 'base' not in ARMS:
    ARMS = ['base'] + ARMS

cur = {'zeta': TRAINED, 'layers': set()}
orig_step = model._layer_step_langevin
def step(h, h_prev, m_b, gamma, dt_, layer_idx=0):
    model.cfg.stiff_mode_damping_ratio = cur['zeta'] if int(layer_idx) in cur['layers'] else TRAINED
    seen.add(int(layer_idx))
    try:
        return orig_step(h, h_prev, m_b, gamma, dt_, layer_idx)
    finally:
        model.cfg.stiff_mode_damping_ratio = TRAINED
model._layer_step_langevin = step

x, y = CS['windows']()
res = {}; logits0 = {}
for arm in ARMS:
    seen = set()
    if arm == 'base':
        cur.update(zeta=TRAINED, layers=set())
    else:
        z, ls = arm.split('@')
        cur['zeta'] = None if z == 'none' else float(z)
        cur['layers'] = set(range(64)) if ls == 'all' else {int(t) for t in ls.split(',')}
    per_win, dmax = [], 0.0
    for i in range(0, len(x), 4):
        with torch.enable_grad():         # the V_phi pair force is an autograd gradient
            lg = model(torch.from_numpy(x[i:i + 4]))[0].detach()
        ce = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(),
                                               torch.from_numpy(y[i:i + 4]).reshape(-1), reduction='none')
        per_win += ce.view(lg.shape[0], -1).mean(1).tolist()
        if arm == 'base':
            logits0[i] = lg
        else:
            dmax = max(dmax, float((lg - logits0[i]).abs().max()))
    assert model.cfg.stiff_mode_damping_ratio == TRAINED
    res[arm] = {'per_window': per_win, 'loss': float(np.mean(per_win)), 'max_logit_change': dmax,
                'layers_seen': sorted(seen)}
    print(f"{arm:>10}: loss {res[arm]['loss']:.4f}  ppl {np.exp(res[arm]['loss']):.2f}  max|dlogit| {dmax:.2e}", flush=True)

b = np.array(res['base']['per_window'])
lines = [f'Stiff-mode damping at evaluation: {FOLDER}', f'model {desc}',
         f'{len(x)} validation windows x 512 ({Path(CS["VAL"]).name}); trained stiff_mode_damping_ratio = {TRAINED}; '
         f'layers seen {res["base"]["layers_seen"]}', '',
         f'{"arm":>10}  {"loss":>7}  {"ppl":>7}  {"vs base":>8}  {"d nats":>8}  {"se":>7}  {"max|dlogit|":>11}']
for arm, r in res.items():
    d = np.array(r['per_window']) - b
    r.update(d_nats=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float('nan'),
             ppl=float(np.exp(r['loss'])), rel=float(np.exp(r['loss'] - res['base']['loss']) - 1))
    lines.append(f"{arm:>10}  {r['loss']:7.4f}  {r['ppl']:7.2f}  {100 * r['rel']:+7.2f}%  {r['d_nats']:+8.4f}  "
                 f"{r['se']:7.4f}  {r['max_logit_change']:11.2e}")
stem = OPTS.get('name', FOLDER[-40:])
OUT.mkdir(parents=True, exist_ok=True)
(OUT / f'damping_probe_{stem}.txt').write_text('\n'.join(lines) + '\n')
(OUT / f'damping_probe_{stem}.json').write_text(json.dumps({'folder': FOLDER, 'model': desc, 'trained_zeta': TRAINED,
                                                             'arms': res}, indent=1))
print('\n'.join(lines))
