#!/usr/bin/env python3
"""
Verify the analytic V_phi force (cfg.vphi_analytic_force, 2026-10-18) against the autograd force on
trained checkpoints, and time it.

    python3 verify_vphi_analytic.py [ARM ...] [T=256]      ARM in F3.1 SR2 G2 PM1 PMX G3p (default F3.1 SR2 G2)

Per arm, on validation tokens (2 sequences of T):
  1. the force: at every layer step, the closed-form f_phi against torch.autograd.grad at the same
     input (relative error per token, and the largest absolute difference);
  2. eval logits and loss, analytic against autograd;
  3. inference under torch.no_grad() with the analytic force: it runs, and equals the enable_grad result;
  4. training (train mode, layer checkpointing on, the same Gumbel draws): the loss and every parameter
     gradient, analytic against autograd (relative error per tensor);
  5. CPU timing: one train step (forward + backward) and one eval forward, autograd against analytic
     (the eval forward of the analytic path under no_grad).
Equal up to float rounding is the criterion, not bit-identity: the closed form orders the arithmetic
differently from autograd.
"""
import sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'sync'))
from sy_replay import Replayer, CACHE

args = [a for a in sys.argv[1:] if '=' not in a] or ['F3.1', 'SR2', 'G2']
T = next((int(a[2:]) for a in sys.argv[1:] if a.startswith('T=')), 256)
torch.set_num_threads(8)
X = torch.from_numpy(np.load(CACHE / 'sy_tokens.npz')['x'][:2, :T + 1]).long()
x, y = X[:, :T], X[:, 1:]
lines = []
say = lambda s='': (print(s, flush=True), lines.append(s))
say(f'Analytic V_phi force against autograd; 2 x {T} validation tokens')
ok_all = True


def logits(m):
    with torch.enable_grad():
        return m(x)[0].detach()


for arm in args:
    R = Replayer(arm); m = R.model; m.eval()
    if m.V_phi is None:
        say(f'\n{arm}: no V_phi, skipped'); continue
    say(f'\n{arm}')
    ok = []
    # ---- 1. the force at every layer step, same input
    m.cfg.use_layer_checkpoint = False
    lf0 = m._layer_forces
    errs = []

    def lf(h_in, xis, layer_idx, *, split=False, vtheta_comps=None):
        m.cfg.vphi_analytic_force = False
        ft, fa = lf0(h_in, xis, layer_idx, split=True, vtheta_comps=vtheta_comps)
        m.cfg.vphi_analytic_force = True
        try:
            _, fc = lf0(h_in, xis, layer_idx, split=True, vtheta_comps=vtheta_comps)
        finally:
            m.cfg.vphi_analytic_force = False
        dn = (fc - fa).detach().norm(dim=-1); an = fa.detach().norm(dim=-1)
        errs.append((layer_idx, float((dn / an.clamp_min(1e-12)).median()), float((dn / an.clamp_min(1e-12)).max()),
                     float((fc - fa).detach().abs().max()), float(an.median())))
        return (ft, fa) if split else (fa if ft is None else ft + fa)
    m._layer_forces = lf
    logits(m)
    m._layer_forces = lf0
    worst = max(e[3] for e in errs)
    rel_med = max(e[1] for e in errs)
    for l, med, mx, ab, fn in errs:
        say(f'   1. layer {l}: |f_closed - f_autograd| / |f_autograd| per token: median {med:.1e}, max {mx:.1e}; '
            f'max abs {ab:.1e} (median |f_phi| {fn:.2e})')
    ok.append(rel_med < 1e-4)
    # ---- 2. eval logits
    m.cfg.vphi_analytic_force = False; la = logits(m)
    m.cfg.vphi_analytic_force = True; lc = logits(m)
    ce = lambda lo: float(torch.nn.functional.cross_entropy(lo.reshape(-1, lo.shape[-1]), y.reshape(-1)))
    d_lo = float((lc - la).abs().max())
    say(f'   2. eval logits max |d| {d_lo:.1e}; loss autograd {ce(la):.6f}, analytic {ce(lc):.6f}')
    ok.append(d_lo < 1e-2 and abs(ce(la) - ce(lc)) < 1e-4)
    # ---- 3. no_grad inference
    try:
        with torch.no_grad():
            ln = m(x)[0]
        d_ng = float((ln - lc).abs().max())
        say(f'   3. inference under torch.no_grad(): runs; max |d logit| against the enable_grad analytic run {d_ng:.1e}')
        ok.append(d_ng < 1e-4)
    except Exception as e:  # noqa: BLE001
        say(f'   3. inference under torch.no_grad() FAILED: {type(e).__name__}: {str(e)[:160]}')
        ok.append(False)
    # ---- 4. training gradients, same Gumbel draws, layer checkpointing as trained
    m.cfg.use_layer_checkpoint = True
    grads, losses = {}, {}
    for flag in (False, True):
        m.cfg.vphi_analytic_force = flag
        m.train(); m.zero_grad(set_to_none=True)
        torch.manual_seed(0)
        _, loss = m(x, targets=y)
        loss.backward()
        losses[flag] = float(loss)
        grads[flag] = {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}
    m.eval(); m.cfg.vphi_analytic_force = False
    same = grads[False].keys() == grads[True].keys()
    rel = {n: float((grads[True][n] - grads[False][n]).norm() / grads[False][n].norm().clamp_min(1e-30))
           for n in grads[False] if n in grads[True]}
    big = {n: v for n, v in rel.items() if grads[False][n].norm() > 1e-6 * max(g.norm() for g in grads[False].values())}
    wn = max(big, key=big.get)
    say(f'   4. training: loss autograd {losses[False]:.6f}, analytic {losses[True]:.6f}; same parameters with a gradient '
        f'{same} ({len(rel)}); worst relative gradient error {big[wn]:.1e} ({wn}); median {np.median(list(big.values())):.1e}')
    gtot = torch.sqrt(sum(g.norm() ** 2 for g in grads[False].values()))
    for n in sorted(big, key=big.get, reverse=True)[:4]:
        say(f'      {n}: |grad| {float(grads[False][n].norm()):.2e} ({float(grads[False][n].norm() / gtot):.1e} of the total), '
            f'|d grad| {float((grads[True][n] - grads[False][n]).norm()):.2e}, relative {big[n]:.1e}')
    dtot = torch.sqrt(sum((grads[True][n] - grads[False][n]).norm() ** 2 for n in rel))
    say(f'      whole gradient: |d grad| / |grad| = {float(dtot / gtot):.1e}')
    ok.append(same and abs(losses[False] - losses[True]) < 1e-4 and float(dtot / gtot) < 1e-4)
    # ---- 5. timing
    def t_train(flag, reps=2):
        m.cfg.vphi_analytic_force = flag; m.train(); ts = []
        for _ in range(reps):
            m.zero_grad(set_to_none=True); torch.manual_seed(0); t0 = time.perf_counter()
            _, loss = m(x, targets=y); loss.backward(); ts.append(time.perf_counter() - t0)
        m.eval(); return min(ts)
    def t_eval(flag, reps=2):
        m.cfg.vphi_analytic_force = flag; ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            if flag:
                with torch.no_grad():
                    m(x)
            else:
                with torch.enable_grad():
                    m(x)
            ts.append(time.perf_counter() - t0)
        return min(ts)
    ta, tc = t_train(False), t_train(True)
    ea, ec = t_eval(False), t_eval(True)
    m.cfg.vphi_analytic_force = False
    say(f'   5. CPU: train step autograd {ta:.2f} s, analytic {tc:.2f} s (x{ta / tc:.2f}); '
        f'eval forward autograd {ea:.2f} s, analytic under no_grad {ec:.2f} s (x{ea / ec:.2f})')
    say(f'   -> {"PASS" if all(ok) else "FAIL"}')
    ok_all &= all(ok)
say(f'\nALL {"PASS" if ok_all else "FAIL"}')
(HERE / 'verify_vphi_analytic_output.txt').write_text('\n'.join(lines) + '\n')
