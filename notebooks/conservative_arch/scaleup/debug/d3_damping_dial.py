#!/usr/bin/env python3
"""
D3, the damping dial (tech report §7.3; checklist D3; 2026-10-22): the learned gamma(h) arm's trained
zeta(h) replaced by c * zeta(h) at inference, weights fixed, and the validation perplexity measured.

    python3 d3_damping_dial.py OUT_DIR FOLDER [threads=8]

zeta(h) = zeta_0 exp(min(s(h), M)), so setting the model's zeta_0 to c * zeta_0 (cap M unchanged) gives
exactly c * zeta(h). c = 0 is SR2's damping (gamma_0 alone) on the learned arm's weights. Two dials:
  joint      c on both layers;
  per layer  c on one layer, the other at its trained zeta (c = 1).
Evaluation: Cell 6b-7's 12 fixed batches (seed 20260920) of 4 x 512 validation tokens, so c = 1 must
reproduce Cell 6b-7's unpatched perplexity (58.02) on the same checkpoint.

Reading fixed in the report before the run: the curve's minimum lies near c = 1 by construction (the wells
were trained with this damping); its width and asymmetry are the result; if c below 1 lowers perplexity,
the trained head added more damping than its own wells need. Reliance, not need.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

OUT = Path(sys.argv[1]); FOLDER = sys.argv[2]
THREADS = int(next((a[8:] for a in sys.argv[3:] if a.startswith('threads=')), 8))
sys.argv = [sys.argv[0], str(OUT / 'harness')]
sys.path.insert(0, str(Path(__file__).parent))
import verify_pm_switch as P
torch.set_num_threads(THREADS)

repl = P.CONFIGS['F3.1'][1] + (("LOWRANK_DAMPED_FLOW  = False", "LOWRANK_DAMPED_FLOW  = True"),
                               ("LEARNED_DAMPING_RATIO = False", "LEARNED_DAMPING_RATIO = True"))
model, g, _ = P.build(FOLDER, repl)
model.to('cpu'); model.eval()
assert model.damping_head is not None
Z0 = model._zeta0

rng = np.random.default_rng(20260920)                      # Cell 6b-7's batches
batches = [g['get_batch'](g['val_ids'], 4, 512, rng) for _ in range(12)]

scale = {0: 1.0, 1: 1.0}
orig_core = model._layer_step_langevin_core
def core(h, h_prev, m_b, gamma, dt, layer_idx, *a, **k):
    model._zeta0 = Z0 * scale[layer_idx]
    try:
        return orig_core(h, h_prev, m_b, gamma, dt, layer_idx, *a, **k)
    finally:
        model._zeta0 = Z0
model._layer_step_langevin_core = core


def ppl(c0, c1):
    scale[0], scale[1] = c0, c1
    tot = 0.0
    for xb, yb in batches:
        with torch.enable_grad():
            _, loss = model(torch.from_numpy(xb), torch.from_numpy(yb))
        tot += float(loss)
    return float(np.exp(tot / len(batches)))


lines = [f'D3, the damping dial: learned gamma(h) arm  ...{FOLDER[-60:]}',
         '   12 x 4 x 512 tokens (Cell 6b-7\'s batches); zeta(h) -> c * zeta(h) at inference, weights fixed']
say = lambda s='': (print(s, flush=True), lines.append(s))
res = {'joint': {}, 'layer0': {}, 'layer1': {}}
t0 = time.time()
base = ppl(1.0, 1.0); res['joint'][1.0] = base
say(f'   c = 1 (trained): PPL {base:.3f}  (Cell 6b-7 unpatched: 58.02)   [{time.time() - t0:.0f}s per evaluation]')
say('\nJOINT DIAL (both layers)')
for c in (0.0, 0.1, 0.25, 0.5, 0.75, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0):
    res['joint'][c] = ppl(c, c)
    say(f'   c = {c:5.2f}   PPL {res["joint"][c]:8.3f}   {100 * (res["joint"][c] / base - 1):+7.2f}%')
for li in (0, 1):
    say(f'\nPER-LAYER DIAL (layer {li} scaled, the other at its trained zeta)')
    for c in (0.0, 0.5, 2.0):
        cc = (c, 1.0) if li == 0 else (1.0, c)
        res[f'layer{li}'][c] = ppl(*cc)
        say(f'   c = {c:5.2f}   PPL {res[f"layer{li}"][c]:8.3f}   {100 * (res[f"layer{li}"][c] / base - 1):+7.2f}%')
cs = sorted(res['joint']); best = min(cs, key=lambda c: res['joint'][c])
say(f'\nminimum of the joint dial at c = {best} (PPL {res["joint"][best]:.3f}); total {time.time() - t0:.0f}s')
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'd3_damping_dial_output.txt').write_text('\n'.join(lines) + '\n')
(OUT / 'd3_damping_dial.json').write_text(json.dumps({k: {str(c): v for c, v in d.items()} for k, d in res.items()}, indent=1))
print(f'wrote {OUT}/d3_damping_dial_output.txt')
