"""PMX's faster basis: indefinite_lowrank_modes_split against the original (protocol SS5.15, 2026-10-10).

The split routine builds the eigenbasis from Vtheta's already orthonormal
retained modes plus the part of the 16 well directions outside their span (a
16-wide Gram solve instead of a 32-wide one). Same operator, same span: the
model must be unchanged up to float rounding.

  1. operator level (float64, random inputs with an inert column): both
     routines reconstruct the same L; both bases orthonormal
  2. model level, PM1's weights, PMX on: logits, train loss and every
     gradient with the split routine against the original one patched in
  3. the layer-1 kick share and the per-layer flow at k = 4, both routines
  4. CPU wall time of the spring routine and of one train step, both routines

Usage: python3 verify_pmx_split.py OUT_DIR
"""
import sys, time
from pathlib import Path

import numpy as np
import torch

OUT = Path(sys.argv[1])
sys.argv = [sys.argv[0], str(OUT / 'harness')]
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / 'parf'))
import verify_pm_switch as P

lines = []
say = lambda s='': (print(s, flush=True), lines.append(s))
ok_all = True

import cfc_baoab as C
torch.manual_seed(0)
d, q, M = 48, 6, 5
A = torch.randn(3, d, q, dtype=torch.float64)
U0 = torch.linalg.qr(A).Q; U0[1, :, 2] = 0.0                  # an inert Vtheta column
k0 = torch.rand(3, q, dtype=torch.float64) * 3; k0[1, 2] = 0.0
R = torch.randn(3, d, M, dtype=torch.float64); R[2, :, 4] = U0[2] @ torch.randn(q, dtype=torch.float64)   # inside span(U0)
dW = torch.randn(3, M, dtype=torch.float64)
L = U0 @ torch.diag_embed(k0) @ U0.transpose(-1, -2) - R @ torch.diag_embed(dW) @ R.transpose(-1, -2)
Us, ks = C.indefinite_lowrank_modes_split(U0, k0, R, dW)
Uo, ko = C.indefinite_lowrank_modes(torch.cat([U0 * k0.sqrt().unsqueeze(-2), R], -1), torch.cat([torch.ones_like(k0), -dW], -1))
rec = lambda U, k: float((U @ torch.diag_embed(k) @ U.transpose(-1, -2) - L).norm() / L.norm())
orth = lambda U: float((U.transpose(-1, -2) @ U - torch.diag_embed((U.norm(dim=-2) > 0.5).double())).abs().max())
ok = rec(Us, ks) < 1e-10 and rec(Uo, ko) < 1e-10 and orth(Us) < 1e-10
ok_all &= ok
say(f'1. operator level: reconstruction split {rec(Us, ks):.1e}, original {rec(Uo, ko):.1e}; orthonormality split {orth(Us):.1e} -> {ok}')

PM1F = ('semsimula_fock_cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_pm64'
        '_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn')
SR2 = (("LOWRANK_DAMPED_FLOW  = False", "LOWRANK_DAMPED_FLOW  = True"),)
PMX = (("POISSON_WELLS_EXACT  = False", "POISSON_WELLS_EXACT  = True"),)
model, g, _ = P.build(PM1F, P.CONFIGS['F3.1'][1] + P.PM_ON + SR2 + PMX)
model.to('cpu')
import model_parf_multixi as MPX
split_fn = MPX.indefinite_lowrank_modes_split
def original_fn(U0, k0, R, dW, floor=1e-10):
    return C.indefinite_lowrank_modes(torch.cat([U0 * k0.clamp(min=0).sqrt().unsqueeze(-2), R], -1),
                                      torch.cat([torch.ones_like(k0), -dW], -1), floor)
spring_t = {'t': 0.0}
def timed(fn):
    def w(*a, **k):
        t0 = time.perf_counter(); out = fn(*a, **k); spring_t['t'] += time.perf_counter() - t0
        return out
    return w

x, y = P.tokens(n_seq=2, T=256)
def run(fn):
    MPX.indefinite_lowrank_modes_split = timed(fn)
    spring_t['t'] = 0.0
    try:
        model.eval()
        with torch.enable_grad():
            ev, _ = model(x)
        model.train(); torch.manual_seed(7)
        t0 = time.perf_counter()
        _, loss = model(x, y); loss.backward()
        t_step = time.perf_counter() - t0
        grads = {k: p.grad.detach().clone() for k, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        model.eval(); model.cfg.substeps_per_layer = 4
        with torch.enable_grad():
            ev4, _ = model(x)
        model.cfg.substeps_per_layer = 1
        return ev.detach(), float(loss), grads, ev4.detach(), t_step, spring_t['t']
    finally:
        MPX.indefinite_lowrank_modes_split = split_fn
S_, O_ = run(split_fn), run(original_fn)
d_log = float((S_[0] - O_[0]).abs().max() / O_[0].abs().max())
d_loss = abs(S_[1] - O_[1])
rel = {k: (float((S_[2][k] - O_[2][k]).norm() / O_[2][k].norm().clamp_min(1e-30)), float(O_[2][k].norm())) for k in O_[2]}
for k, (r_, n_) in sorted(rel.items(), key=lambda kv: -kv[1][0])[:6]:
    print(f'   grad {k:<50} rel |d| {r_:.2e}   |grad| {n_:.2e}')
gtot = sum(n_ ** 2 for _, n_ in rel.values()) ** 0.5
d_grad = max(r_ for r_, n_ in rel.values() if n_ > 1e-6 * gtot)   # tensors carrying gradient
d_k4 = float((S_[3] - O_[3]).abs().max() / O_[3].abs().max())
ok = d_log < 1e-4 and d_loss < 1e-4 and d_grad < 1e-3 and d_k4 < 1e-4; ok_all &= ok
say(f'2. model, PM1 weights, PMX on: logits max rel |d| {d_log:.1e}; train loss |d| {d_loss:.1e}; '
    f'worst gradient rel |d| {d_grad:.1e} (tensors above 1e-6 of the total gradient norm) -> {ok}')
say(f'3a. per-layer flow at k = 4: logits max rel |d| {d_k4:.1e}')

cap = []
orig_step = model._fock_layer_step
def capture(h, h_prev, r, sal, m_b, gamma, dt, layer_idx=0):
    cap.append((h.detach().clone(), h_prev.detach().clone(), r.clone(), sal.clone(), m_b, gamma, dt, layer_idx))
    return orig_step(h, h_prev, r, sal, m_b, gamma, dt, layer_idx=layer_idx)
model._fock_layer_step = capture
try:
    model.eval()
    with torch.enable_grad():
        model(x)
finally:
    model._fock_layer_step = orig_step
def kick_share(fn, item):
    MPX.indefinite_lowrank_modes_split = fn
    try:
        h, hp, r, s, mb, gmm, dt, li = item
        with torch.enable_grad():
            full = model._fock_layer_step(h, hp, r, s, mb, gmm, dt, layer_idx=li)[0].detach()
            model.cfg.force_clamp_max = 0.0
            try:
                nok = model._fock_layer_step(h, hp, r, s, mb, gmm, dt, layer_idx=li)[0].detach()
            finally:
                model.cfg.force_clamp_max = None
        return float(((nok - full).norm(dim=-1) / (full - h).norm(dim=-1).clamp_min(1e-12)).median())
    finally:
        MPX.indefinite_lowrank_modes_split = split_fn
ks_s, ks_o = kick_share(split_fn, cap[-1]), kick_share(original_fn, cap[-1])
ok = abs(ks_s - ks_o) < 1e-3; ok_all &= ok
say(f'3b. layer-1 kick share: split {ks_s:.4f}, original {ks_o:.4f} -> {ok}')
say(f'4. CPU, one train step (2 x 256): split {S_[4]:.1f} s (spring {S_[5]:.2f} s), original {O_[4]:.1f} s (spring {O_[5]:.2f} s); '
    f'spring routine {O_[5] / max(S_[5], 1e-9):.2f}x faster')
say(f'\n-> {"ALL PASS" if ok_all else "FAILURES ABOVE"}')
(HERE / 'verify_pmx_split_output.txt').write_text('\n'.join(lines) + '\n')
