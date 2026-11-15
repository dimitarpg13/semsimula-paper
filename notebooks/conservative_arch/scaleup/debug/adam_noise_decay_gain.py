#!/usr/bin/env python3
"""Does AdamW's second moment predict the decay gain? (2026-11-15, protocol p-decay-noise-cpu)

    python3 -I adam_noise_decay_gain.py OUT label=<checkpoint.pt> [label=<checkpoint.pt> ...]

Writes OUT/adam_noise_decay_gain.txt. At a constant peak rate the optimizer bounces in the steep directions, and the
loss carries an excess the decay drains. For one coordinate of curvature lambda and gradient noise sigma, with the noise
dominating, AdamW's second moment is v ~ sigma^2, the update is SGD at step eta/sigma, the stationary variance
eta*sigma/(2*lambda), and the excess loss eta*sigma/4, whatever lambda is. Summed: G_pred = eta_peak/4 * sum_i sqrt(v_i).
With beta2 = 0.95, exp_avg_sq averages about the last 20 steps, so it is a local reading of sigma_i^2. Reported per
checkpoint: the sum over all coordinates, over the vocabulary-sized matrices and over the rest, G_pred, the median of
|m|/sqrt(v) (about 0.15 for pure noise at beta1 = 0.9; the coordinates are noise-dominated when it is that small or
smaller), and the sum by tensor shape, so two checkpoints can be compared block by block.
"""
import sys
from collections import defaultdict
from pathlib import Path

import torch

OUT = Path(sys.argv[1]).resolve()
ETA = 1.2e-3
lines = []; shape_cols = []
for arg in sys.argv[2:]:
    lab, p = arg.split('=', 1)
    c = torch.load(p, map_location='cpu', weights_only=False)
    o = c['optimizer_state_dict']
    b1, b2 = o['param_groups'][0]['betas']
    tot = defaultdict(float); by_shape = defaultdict(float); ratios = []
    for st in o['state'].values():
        if 'exp_avg_sq' not in st:
            continue
        t = float(st['step']); sh = tuple(st['exp_avg_sq'].shape)
        s = (st['exp_avg_sq'].double() / (1 - b2 ** t)).sqrt()
        m = st['exp_avg'].double() / (1 - b1 ** t)
        tot['vocab' if 50257 in sh else 'rest'] += float(s.sum()); by_shape[sh] += float(s.sum())
        g = torch.Generator().manual_seed(0)
        idx = torch.randint(0, s.numel(), (min(20000, s.numel()),), generator=g)
        ratios.append(m.flatten()[idx].abs() / s.flatten()[idx].clamp(min=1e-30))
    a = tot['vocab'] + tot['rest']
    lines += [f'{lab}: step {c.get("step")}  lr now {o["param_groups"][0]["lr"]:.2e}  val ppl {c.get("val_ppl", float("nan")):.2f}',
              f'   sum sqrt(v): all {a:.1f}  vocab-sized {tot["vocab"]:.1f}  rest {tot["rest"]:.1f}   '
              f'G_pred = eta/4 * sum: all {ETA / 4 * a:.3f}  rest {ETA / 4 * tot["rest"]:.3f}   '
              f'median |m|/sqrt(v) {float(torch.cat(ratios).median()):.3f}']
    shape_cols.append((lab, by_shape))
    del c, o
lines += ['', 'sum sqrt(v) by tensor shape:']
shapes = sorted(set().union(*[d for _, d in shape_cols]), key=lambda s: -max(d.get(s, 0) for _, d in shape_cols))
lines.append(f'  {"shape":>16} ' + ' '.join(f'{lab:>18}' for lab, _ in shape_cols))
for s in shapes:
    v = [d.get(s, 0.0) for _, d in shape_cols]
    if max(v) >= 1:
        lines.append(f'  {str(s):>16} ' + ' '.join(f'{x:18.1f}' for x in v))
txt = '\n'.join(lines); print(txt)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'adam_noise_decay_gain.txt').write_text(txt + '\n')
