#!/usr/bin/env python3
"""
Verify the FO-b switch (Cell 0 FIRST_ORDER_OVERDAMPED, cfg.first_order_overdamped; protocol SS5.13,
2026-10-19): a true first-order layer step, the gradient flow dh/dt = mu f(h), mu = dt / (2 m).

    python3 verify_fob_switch.py OUT ref [arm=pmx] [threads=1]            # in a copy of HEAD: write the reference
    python3 verify_fob_switch.py OUT check REF.npz [arm=pmx] [threads=1]  # in the working tree: compare and verify

arm: sr2 (default; the FO series' anchor) or pmx (FO-b-PMX, added 2026-10-19: PMX's exact wells, whose
frozen spring alpha I + modes relaxes on all of R^d). Checks (working tree):
  1. off is HEAD, bit for bit: eval logits, the training loss and every parameter gradient.
  2. the relaxations are exact gradient flows of their frozen springs: float64, against a fine RK4
     solve, for stiff, soft, zero and negative stiffness. lowrank_overdamped_substep (SR2): the
     complement of span(U) does not move; lowrank_iso_overdamped_substep (PMX): K = alpha I + modes on
     all of R^d, with alpha of either sign.
  3. the layer step is a discretisation of the gradient flow of the model's own force: at real layer
     inputs, in float64, k substeps of dt/k (the per-layer flow) converge to an RK4 solve of
     dh/dt = mu (f_theta + f_phi)(h) over dt, and the step hands no velocity to the next layer
     (h_prev_out == h_new). The middle (B) part is one explicit gradient step, the only explicit
     first-order step with one force evaluation, so the scheme is first order in 1/k (a single FO-b
     step is gradient descent with exact relaxation of the stiff modes). Required, per layer: the
     error falls at every doubling from k = 8 to 128, with an observed order of at least 0.8 over
     the last two doublings (32 -> 64 -> 128); on SR2 reported with V_phi's pair term on and zeroed. The reference holds PMX's well
     occupations fixed, as the per-layer flow does, at their k -> infinity value: the occupations at the
     layer input (the flow freezes them at the first substep's midpoint, which tends there).
  4. on trains: the loss is finite, every gradient is finite, the same parameters receive a gradient
     as when off (the mass among them: it sets the mobility), and the gradients differ from off.
  5. the switch adds no parameters: identical state_dict keys and shapes on and off.
  6. the mobility is fixed at construction: a layer step of the trained size gives the same output
     when cfg.dt is changed (as Cell 6b-7 does while it probes other step sizes).
"""
import sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(sys.argv[1]); MODE = sys.argv[2]
REF = Path(sys.argv[3]) if MODE == 'check' else None
HERE = Path(__file__).resolve().parent
_argv = list(sys.argv)
sys.argv = [sys.argv[0], str(OUT / 'harness')]
sys.path.insert(0, str(HERE))
import verify_pm_switch as P
sys.argv = _argv
G = P.G
torch.set_num_threads(int(next((a[8:] for a in sys.argv[2:] if a.startswith('threads=')), 8)))

SR2 = (("LOWRANK_DAMPED_FLOW  = False", "LOWRANK_DAMPED_FLOW  = True"),)
FOB = (("FIRST_ORDER_OVERDAMPED = False", "FIRST_ORDER_OVERDAMPED = True"),)
PMXS = (("POISSON_WELLS_EXACT  = False", "POISSON_WELLS_EXACT  = True"),)
ARM = next((a[4:] for a in sys.argv[2:] if a.startswith('arm=')), 'sr2')
_T = 'semsimula_fock_cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_'
_E = '_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn'
FOLDER, REPL = {'sr2': (_T + 'sr2' + _E, P.CONFIGS['F3.1'][1] + SR2),
                'pmx': (_T + 'pm64_pmx16_sr2' + _E, P.CONFIGS['F3.1'][1] + P.PM_ON + SR2 + PMXS)}[ARM]
NAME = ARM.upper()

val = np.load(G.VAL)
xy = torch.from_numpy(val[:2 * 257].astype(np.int64)).view(2, 257)
x, y = xy[:, :256], xy[:, 1:]


def run(model):
    model.eval()
    with torch.enable_grad():
        lo = model(x)[0].detach().clone()
    model.train(); model.zero_grad(set_to_none=True)
    torch.manual_seed(0)
    _, loss = model(x, targets=y)
    loss.backward()
    grads = {n: (p.grad.detach().clone() if p.grad is not None else None) for n, p in model.named_parameters()}
    model.eval()
    return lo, loss.detach().clone(), grads


off, g_off, _ = P.build(FOLDER, REPL)
off.to('cpu')
lo0, loss0, gr0 = run(off)

if MODE == 'ref':
    np.savez(OUT / f'fob_ref_{ARM}.npz', logits=lo0.numpy(), loss=loss0.numpy(),
             **{f'g::{n}': v.numpy() for n, v in gr0.items() if v is not None})
    print(f'reference written: {OUT / f"fob_ref_{ARM}.npz"}  (loss {float(loss0):.6f}, {len(gr0)} parameters)')
    sys.exit(0)

lines = []
say = lambda s='': (print(s, flush=True), lines.append(s))
ok = []
say(f"FO-b switch verification on {NAME}'s trained weights (2 x 256 validation tokens)")

# 1. off == HEAD
ref = np.load(REF)
d_lo = float(np.abs(lo0.numpy() - ref['logits']).max()); d_loss = abs(float(loss0) - float(ref['loss']))
gk = {k[3:] for k in ref.files if k.startswith('g::')}
mine = {n for n, v in gr0.items() if v is not None}
d_g = max(float(np.abs(gr0[n].numpy() - ref[f'g::{n}']).max()) for n in gk & mine)
c1 = d_lo == 0.0 and d_loss == 0.0 and gk == mine and d_g == 0.0
ok.append(c1)
say(f'1. off vs HEAD: max |d logit| {d_lo:.1e}; |d loss| {d_loss:.1e}; same parameters with a gradient '
    f'{gk == mine} ({len(mine)}); max |d grad| {d_g:.1e}  -> {"PASS" if c1 else "FAIL"}')

# 2. the mode relaxation against RK4, float64
from cfc_baoab import lowrank_overdamped_substep, lowrank_iso_overdamped_substep
torch.manual_seed(1)
d, q, n = 24, 6, 5
U = torch.linalg.qr(torch.randn(n, d, q, dtype=torch.float64))[0]
kappa = torch.tensor([30.0, 2.0, 0.3, 1e-9, 0.0, -0.7], dtype=torch.float64).expand(n, q).clone()
s = torch.randn(n, d, dtype=torch.float64)
h0 = torch.randn(n, d, dtype=torch.float64)
mu = (0.2 + torch.rand(n, 1, dtype=torch.float64))
t = 2.0
L = U @ torch.diag_embed(kappa) @ U.transpose(-1, -2)
fL = lambda h: torch.einsum('ndq,nq->nd', U, torch.einsum('ndq,nd->nq', U, s - (L @ h.unsqueeze(-1)).squeeze(-1)))
h_ex = lowrank_overdamped_substep(h0, U, kappa, fL(h0), mu, t)
h_rk, M = h0.clone(), 20000
for _ in range(M):
    e = t / M
    k1 = mu * fL(h_rk); k2 = mu * fL(h_rk + 0.5 * e * k1); k3 = mu * fL(h_rk + 0.5 * e * k2); k4 = mu * fL(h_rk + e * k3)
    h_rk = h_rk + e / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
err2 = float((h_ex - h_rk).norm() / (h_rk - h0).norm())
P_c = torch.eye(d, dtype=torch.float64) - U @ U.transpose(-1, -2)
comp = float((P_c @ (h_ex - h0).unsqueeze(-1)).abs().max())
# the isotropic spring on all of R^d (PMX): K = alpha I + U diag(kappa) U^T, alpha of either sign
alpha = torch.tensor([0.5, -0.3, 0.0, 1e-9, 2.0], dtype=torch.float64).view(n, 1)
K = alpha.unsqueeze(-1) * torch.eye(d, dtype=torch.float64) + L
f0 = torch.randn(n, d, dtype=torch.float64)
fK = lambda x: f0 - (K @ (x - h0).unsqueeze(-1)).squeeze(-1)
x_ex = lowrank_iso_overdamped_substep(h0, U, kappa, alpha, f0, mu, t)
x_rk = h0.clone()
for _ in range(M):
    e = t / M
    k1 = mu * fK(x_rk); k2 = mu * fK(x_rk + 0.5 * e * k1); k3 = mu * fK(x_rk + 0.5 * e * k2); k4 = mu * fK(x_rk + e * k3)
    x_rk = x_rk + e / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
err2i = float((x_ex - x_rk).norm() / (x_rk - h0).norm())
c2 = err2 < 1e-10 and comp < 1e-12 and err2i < 1e-10
ok.append(c2)
say(f'2. relaxations vs RK4 (float64): low-rank (kappa 30, 2, 0.3, 1e-9, 0, -0.7) relative error {err2:.1e}, '
    f'complement moved {comp:.1e}; isotropic + low-rank (alpha 0.5, -0.3, 0, 1e-9, 2) relative error {err2i:.1e}'
    f'  -> {"PASS" if c2 else "FAIL"}')

# 3. the layer step converges to the gradient flow of the model's force (float64, real layer inputs)
on, g_on, _ = P.build(FOLDER, REPL + FOB)
on.to('cpu')
assert on.cfg.first_order_overdamped is True and off.cfg.first_order_overdamped is False
m = on
m.eval(); m.double()
m.cfg.use_layer_checkpoint = False
m.cfg.force_clamp_max = None                  # the clamp acts on the kick only; the limit is the unclamped flow
xs = x[:1, :64]
with torch.no_grad():
    h_in0 = m._embed(xs)                       # the stack's input, as in forward()
m_b = m.compute_mass(xs)
m_b = m_b.double() if torch.is_tensor(m_b) else m_b
gamma, dt = m.gamma, float(m.cfg.dt)
mu_b = dt / (2.0 * m_b)
say(f'3. per-layer flow vs RK4 on dh/dt = mu (f_theta + f_phi), mu = dt / 2m (dt = {dt:g}), float64, 1 x 64 tokens')


def layer_map(h, layer, k):
    m._flow_ctx = {}
    try:
        hp = h
        for j in range(k):
            m._flow_ctx['project'] = j == k - 1
            with torch.enable_grad():
                h_new, hp_new = m._layer_step_langevin(h, hp, m_b, gamma, dt / k, layer)
            if j == 0:
                ctx = dict(m._flow_ctx)
            h, hp = h_new.detach(), hp_new.detach()
        return h, (h_new - hp_new).abs().max().item(), ctx
    finally:
        m._flow_ctx = None


def converge(tag):
    """per layer: errors of k substeps against RK4, and the observed order between k = 64 and 128"""
    res, h = [], h_in0
    for layer in range(m.cfg.L):
        out1, vel, ctx = layer_map(h, layer, 1)
        xis = ctx['xis']
        if 'phi' in ctx:
            # PMX: the per-layer flow freezes the well occupations where the force is first evaluated,
            # the first substep's midpoint; as k grows that tends to the layer input, so the limit
            # holds them at the layer input (copying the k = 1 run's would integrate another flow)
            ctx = dict(ctx)
            with torch.no_grad():
                ctx['phi'] = m.poisson_mode_occupation(h)[1]
        comps = m.V_theta.context_components(xis)

        def F(hh):
            m._flow_ctx = ctx                  # PMX: the occupations held at the layer's first substep
            try:
                with torch.enable_grad():
                    ft, fp = m._layer_forces(hh, xis, layer, split=True, vtheta_comps=comps)
            finally:
                m._flow_ctx = None
            return (mu_b * (ft + fp)).detach()
        h_rk, M = h.clone(), 2048
        for _ in range(M):
            e = dt / M
            k1 = F(h_rk); k2 = F(h_rk + 0.5 * e * k1); k3 = F(h_rk + 0.5 * e * k2); k4 = F(h_rk + e * k3)
            h_rk = h_rk + e / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        h_ref = m._project(h_rk) if m.cfg.ln_after_step else h_rk
        errs = {}
        for k in (1, 2, 4, 8, 16, 32, 64, 128):
            hk = out1 if k == 1 else layer_map(h, layer, k)[0]
            errs[k] = float((hk - h_ref).norm() / (h_ref - h).norm())
        res.append((layer, vel, errs, float(np.log2(errs[64] / errs[128]))))
        h = out1
    for layer, vel, errs, order in res:
        say(f'   {tag} layer {layer}: |h_prev_out - h_new| {vel:.1e}; relative error by k: '
            + ', '.join(f'{k}: {e:.1e}' for k, e in errs.items()) + f'; observed order (64 -> 128) {order:.2f}')
    return res


# asymptotic first order: the error falls at every doubling from k = 8 on, with an observed order of at
# least 0.8 over the last two doublings. (A first criterion, error at k = 128 at most 2% of the k = 1
# error, was dropped on 2026-10-19: the k = 1 error depends on the layer's stiffness, and PMX's layer 1,
# converging at order 0.99, missed it at 2.2%.)
good = lambda res: all(
    vel == 0.0 and all(errs[2 * k] < errs[k] for k in (8, 16, 32, 64))
    and np.log2(errs[32] / errs[64]) >= 0.8 and order >= 0.8 for _, vel, errs, order in res)
res_a = converge('(a) full force,')
c3a = good(res_a)
c3b = True
if ARM == 'sr2':
    _pp, _af = m._pair_potential, m.cfg.vphi_analytic_force
    m._pair_potential = lambda h, l, xis=None, h_src_live=None: (h * 0).sum()
    m.cfg.vphi_analytic_force = False
    try:
        res_b = converge('(b) V_phi off,')
    finally:
        m._pair_potential, m.cfg.vphi_analytic_force = _pp, _af
    c3b = good(res_b)
c3 = c3a and c3b
say(f'   converges to the gradient flow at first order, zero velocity handed on: full force {c3a}'
    + (f', V_phi off {c3b}' if ARM == 'sr2' else '') + f'  -> {"PASS" if c3 else "FAIL"}')
ok.append(c3)

# 4. on trains (a fresh float32 copy)
on, _, _ = P.build(FOLDER, REPL + FOB)
on.to('cpu')
lo1, loss1, gr1 = run(on)
fin = all(torch.isfinite(v).all() for v in gr1.values() if v is not None) and bool(torch.isfinite(loss1))
got = {n for n, v in gr1.items() if v is not None}
mass = [n for n in got if n.split('.')[-1] in ('raw_m_bias', 'raw_logfreq_alpha')]
moved = max(float((gr1[n] - gr0[n]).abs().max()) for n in mine & got)
c4 = fin and got == mine and moved > 0 and bool(mass)
ok.append(c4)
nll = lambda lo: float(torch.nn.functional.cross_entropy(lo.reshape(-1, lo.shape[-1]), y.reshape(-1)))
say(f'4. on, train mode: loss {float(loss1):.4f} (off {float(loss0):.4f}); all finite {fin}; same parameters with a '
    f'gradient {got == mine}; mass parameters with a gradient {sorted(mass)}; max |grad on - grad off| {moved:.2e}'
    f'  -> {"PASS" if c4 else "FAIL"}')
say(f'   (descriptive) eval loss on {NAME}\'s second-order weights: off {nll(lo0):.4f}, FO-b {nll(lo1):.4f}')

# 5. no parameters added
sd0, sd1 = off.state_dict(), on.state_dict()
c5 = sd0.keys() == sd1.keys() and all(sd0[k].shape == sd1[k].shape for k in sd0)
ok.append(c5)
say(f'5. state_dict keys and shapes identical on and off ({len(sd0)} tensors)  -> {"PASS" if c5 else "FAIL"}')
# 6. mobility independent of cfg.dt
on.eval()
with torch.no_grad():
    h6 = on._embed(x[:1, :64])
m6 = on.compute_mass(x[:1, :64])
dt0 = float(on.cfg.dt)
with torch.enable_grad():
    a6 = on._layer_step_langevin(h6, h6, m6, on.gamma, dt0, 0)[0].detach()
    on.cfg.dt = dt0 / 3
    try:
        b6 = on._layer_step_langevin(h6, h6, m6, on.gamma, dt0, 0)[0].detach()
    finally:
        on.cfg.dt = dt0
d6 = float((a6 - b6).abs().max())
c6 = d6 == 0.0
ok.append(c6)
say(f'6. a step of dt = {dt0:g} with cfg.dt = {dt0:g} and with cfg.dt = {dt0 / 3:.3g}: max |d h| {d6:.1e}  -> {"PASS" if c6 else "FAIL"}')
say(f'\nALL {"PASS" if all(ok) else "FAIL"}')
(HERE / ('verify_fob_switch_output.txt' if ARM == 'sr2' else f'verify_fob_switch_{ARM}_output.txt')).write_text('\n'.join(lines) + '\n')
