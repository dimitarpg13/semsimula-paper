#!/usr/bin/env python3
"""
Float64 check that the analytic V_phi force IS autograd's force as a function (2026-10-18).

In float32 the training gradients of the two paths differ by ~5e-5 of the whole gradient, most of it in
the score head, whose gradient comes from second-order router paths where y is nearly one-hot. That is
either rounding or a missing term. In float64 a formula that matches only to first order shows up at
once: here the trained model is cast to float64 and, at real layer inputs, compared are
  (a) the force, closed form against -autograd.grad(U_pair, h, create_graph=True);
  (b) the derivatives of a fixed random functional L = <f, R> of the force with respect to every
      parameter and to the input states: the quantity training differentiates.
Both must agree to ~1e-10 relative.

    python3 verify_vphi_analytic_f64.py [ARM ...] [T=128]
"""
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'sync'))
from sy_replay import Replayer, CACHE

args = [a for a in sys.argv[1:] if '=' not in a] or ['F3.1', 'G2']
T = next((int(a[2:]) for a in sys.argv[1:] if a.startswith('T=')), 128)
torch.set_num_threads(8)
X = torch.from_numpy(np.load(CACHE / 'sy_tokens.npz')['x'][:2, :T]).long()
lines = []
say = lambda s='': (print(s, flush=True), lines.append(s))
say(f'Analytic V_phi force against autograd in float64, at real layer inputs (2 x {T} tokens)')
ok_all = True
for arm in args:
    R = Replayer(arm); m = R.model; m.eval(); m.cfg.use_layer_checkpoint = False
    with torch.enable_grad():
        m(X)                                                   # records each layer's input
    m.double()
    live = getattr(m.cfg, 'vphi_grad_path', 'default') == 'live'
    params = [p for p in m.parameters() if p.requires_grad]
    names = [n for n, p in m.named_parameters() if p.requires_grad]
    say(f'\n{arm} (V_phi sources {"live" if live else "detached"})')
    for l in range(m.cfg.L):
        h = R.records[l]['args'][0].double().detach().requires_grad_(True)
        q = h.view_as(h) if live else h
        src = h if live else None
        U = m._pair_potential(q, l, h_src_live=src)
        fa = -torch.autograd.grad(U, q, create_graph=True)[0]
        fc = m._pair_force_analytic(q, l, h_src_live=src)
        rf = float((fc - fa).norm() / fa.norm())
        g = torch.Generator().manual_seed(l)
        Rn = torch.randn(fa.shape, generator=g, dtype=fa.dtype)
        ga = torch.autograd.grad((fa * Rn).sum(), params + [h], allow_unused=True, retain_graph=True)
        gc = torch.autograd.grad((fc * Rn).sum(), params + [h], allow_unused=True, retain_graph=True)
        rows = []
        for n, a, c in zip(names + ['<input states h>'], ga, gc):
            if a is None and c is None:
                continue
            a = torch.zeros_like(c) if a is None else a
            c = torch.zeros_like(a) if c is None else c
            if float(a.norm()) == 0 and float(c.norm()) == 0:
                continue
            rows.append((float((c - a).norm() / max(float(a.norm()), 1e-300)), n, float(a.norm())))
        # a gradient that is zero in exact arithmetic (e.g. score_head.w2.bias: a uniform shift of every
        # score leaves the softmax unchanged) reads as rounding residue in both paths; compare it absolutely
        gmax = max(r_[2] for r_ in rows)
        zero = [r_ for r_ in rows if r_[2] < 1e-12 * gmax]
        rows = [r_ for r_ in rows if r_[2] >= 1e-12 * gmax]
        rows.sort(reverse=True)
        worst = rows[0]
        ok = rf < 1e-9 and worst[0] < 1e-8
        ok_all &= ok
        say(f'   layer {l}: force relative error {rf:.1e}; d<f,R>/d(parameters, h): worst {worst[0]:.1e} '
            f'({worst[1]}, |grad| {worst[2]:.1e}); {len(rows)} tensors'
            + (f'; zero in exact arithmetic: {", ".join(n for _, n, _ in zero)} (|grad| below {1e-12 * gmax:.0e} in both)' if zero else '')
            + f'  -> {"PASS" if ok else "FAIL"}')
        if not ok:
            for r_, n, a in rows[:5]:
                say(f'      {r_:.1e}  {n}  |grad| {a:.1e}')
say(f'\nALL {"PASS" if ok_all else "FAIL"}')
(HERE / 'verify_vphi_analytic_f64_output.txt').write_text('\n'.join(lines) + '\n')
