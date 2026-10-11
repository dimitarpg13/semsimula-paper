"""PMX's identical-math speed-ups, checked against the split path (protocol SS5.15, 2026-10-11).

The A100 profile (Cell 6b-16) put a third of PMX's extra time in its batched
SVDs (each ~20 ms whatever the width up to 32) and a quarter in the exact
substeps, where damped_mode_coefficients sums a 40-term series for every
element. Two changes, both the same mathematics:
  (a) indefinite_lowrank_modes_chol: the well basis by two Cholesky QR passes
      instead of a Gram SVD (one batched SVD fewer per PMX layer step);
  (b) series_terms_for: the coefficient series cut to the terms that matter at
      PMX's gamma * dt (12 at 0.2), for PMX's substep only. SR2's default
      40-term path is untouched.

  1. basis: Cholesky QR orthonormal to rounding, inert columns zeroed; on real
     states the well directions' conditioning (they must stay far from
     dependent for (a))
  2. operator: the Cholesky routine reconstructs the same L as the split one
  3. coefficients: the short series equals the 40-term sum to float64 rounding
     on omega0^2 in [-1e-4, 1e-4] t^-2 (the series branch), at gamma t 0.2,
     0.1, 0.05; the default call is unchanged (40 terms)
  4. model, PM1's weights, PMX on: logits, train loss and every gradient,
     new path against the split path patched back in; the per-layer flow at
     k = 4; the layer-1 kick share
  5. CPU wall time of one train step, both paths

Usage: python3 verify_pmx_fast.py OUT_DIR
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
import cfc_baoab as C

lines = []
say = lambda s='': (print(s, flush=True), lines.append(s))
ok_all = True

# 1. basis
torch.manual_seed(0)
Rp = torch.randn(4, 64, 16, dtype=torch.float64)
Rp[1, :, 5] = 0.0                                           # an inert column
Rp[2, :, 7] = Rp[2, :, 6] * (1 + 1e-4) + 1e-4 * torch.randn(64, dtype=torch.float64)   # nearly dependent
Q, inert = C._cholqr_basis(Rp, 1e-10)
nz = ~inert
orth = float((Q.transpose(-1, -2) @ Q - torch.diag_embed(nz.double())).abs().max())
span = float(((Rp - Q @ (Q.transpose(-1, -2) @ Rp)).norm(dim=-2) / Rp.norm(dim=-2).clamp_min(1e-30))[nz].max())
ok = orth < 1e-12 and span < 1e-12 and bool(inert[1, 5]) and float(Q[1, :, 5].abs().max()) == 0.0
ok_all &= ok
say(f'1a. Cholesky QR: orthonormality {orth:.1e}; columns reproduced to {span:.1e}; inert column zeroed: {bool(inert[1, 5])} -> {ok}')

PM1F = ('semsimula_fock_cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_pm64'
        '_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn')
SR2 = (("LOWRANK_DAMPED_FLOW  = False", "LOWRANK_DAMPED_FLOW  = True"),)
PMX = (("POISSON_WELLS_EXACT  = False", "POISSON_WELLS_EXACT  = True"),)
model, g, _ = P.build(PM1F, P.CONFIGS['F3.1'][1] + P.PM_ON + SR2 + PMX)
model.to('cpu')
import model_parf_multixi as MPX
x, y = P.tokens(n_seq=2, T=256)
seen = []
chol_fn = MPX.indefinite_lowrank_modes_chol
def spy(U0, k0, R, dW, *a, **k):
    seen.append((U0.detach().double(), k0.detach().double(), R.detach().double(), dW.detach().double()))
    return chol_fn(U0, k0, R, dW, *a, **k)
MPX.indefinite_lowrank_modes_chol = spy
try:
    model.eval()
    with torch.enable_grad():
        model(x)
finally:
    MPX.indefinite_lowrank_modes_chol = chol_fn
conds = []
for U0, k0, R, dW in seen:
    Rp_ = R - U0 @ (U0.transpose(-1, -2) @ R)
    sv = torch.linalg.svdvals(Rp_)
    conds.append(float((sv[..., 0] / sv[..., -1]).max()))
say(f'1b. real states (PM1 weights, every PMX call of one forward): worst condition number of the well directions outside span(U0) {max(conds):.2f}')

# 2. operator: the inputs are the model's float32 tensors, so U0 is orthonormal
#    only to ~1e-7 and both routines reconstruct L to that floor alike; what
#    must agree to float64 rounding is the two routines' operators
worst_rec, worst_mut = 0.0, 0.0
for U0, k0, R, dW in seen:
    L = U0 @ torch.diag_embed(k0.clamp(min=0)) @ U0.transpose(-1, -2) - R @ torch.diag_embed(dW) @ R.transpose(-1, -2)
    Uc, kc = C.indefinite_lowrank_modes_chol(U0, k0, R, dW)
    Us, ks = C.indefinite_lowrank_modes_split(U0, k0, R, dW)
    Lc = Uc @ torch.diag_embed(kc) @ Uc.transpose(-1, -2)
    Ls = Us @ torch.diag_embed(ks) @ Us.transpose(-1, -2)
    worst_rec = max(worst_rec, float((Lc - L).norm() / L.norm()), float((Ls - L).norm() / L.norm()))
    worst_mut = max(worst_mut, float((Lc - Ls).norm() / L.norm()))
ok = worst_mut < 1e-10 and worst_rec < 1e-4; ok_all &= ok
say(f'2. operator on the real states: Cholesky vs split routine {worst_mut:.1e}; each against L {worst_rec:.1e} '
    f'(the float32 floor of U0, alike for both) -> {ok}')

# 3. coefficients
worst = 0.0
for gt in (0.2, 0.1, 0.05):
    t = 2.0; gm = gt / t
    n = C.series_terms_for(gt)
    w2 = torch.linspace(-0.99e-4, 0.99e-4, 401, dtype=torch.float64) / t ** 2
    full = C.damped_mode_coefficients(w2, gm, t)
    short = C.damped_mode_coefficients(w2, gm, t, n)
    worst = max(worst, max(float(((a - b).abs() / a.abs().clamp_min(1e-300)).max()) for a, b in zip(full, short)))
d_default = max(float((a - b).abs().max()) for a, b in zip(C.damped_mode_coefficients(w2, gm, t), C.damped_mode_coefficients(w2, gm, t, 40)))
ok = worst < 1e-14 and d_default == 0.0; ok_all &= ok
say(f'3. coefficients: short series ({C.series_terms_for(0.2)} terms at gamma t 0.2) vs 40 terms on the series branch: worst rel |d| {worst:.1e}; '
    f'default call = 40 terms exactly: {d_default == 0.0} -> {ok}')

# 4. model
split_fn = MPX.indefinite_lowrank_modes_split
orig_terms = MPX.series_terms_for
def run(new):
    if not new:
        MPX.indefinite_lowrank_modes_chol = split_fn
        MPX.series_terms_for = lambda gt, tol=1e-18: 40
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
        return ev.detach(), float(loss), grads, ev4.detach(), t_step
    finally:
        MPX.indefinite_lowrank_modes_chol = chol_fn
        MPX.series_terms_for = orig_terms
N_, O_ = run(True), run(False)
d_log = float((N_[0] - O_[0]).abs().max() / O_[0].abs().max())
d_loss = abs(N_[1] - O_[1])
gtot = sum(float(v.norm()) ** 2 for v in O_[2].values()) ** 0.5
d_grad = max(float((N_[2][k] - O_[2][k]).norm() / O_[2][k].norm().clamp_min(1e-30)) for k in O_[2]
             if float(O_[2][k].norm()) > 1e-6 * gtot)
d_k4 = float((N_[3] - O_[3]).abs().max() / O_[3].abs().max())
ok = d_log < 1e-4 and d_loss < 1e-4 and d_grad < 1e-3 and d_k4 < 1e-4; ok_all &= ok
say(f'4a. model, PM1 weights, PMX on, new vs split path: logits max rel |d| {d_log:.1e}; train loss |d| {d_loss:.1e}; '
    f'worst gradient rel |d| {d_grad:.1e} (tensors above 1e-6 of the total norm); per-layer flow k = 4 logits {d_k4:.1e} -> {ok}')

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
def kick_share(new, item):
    if not new:
        MPX.indefinite_lowrank_modes_chol = split_fn
        MPX.series_terms_for = lambda gt, tol=1e-18: 40
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
        MPX.indefinite_lowrank_modes_chol = chol_fn
        MPX.series_terms_for = orig_terms
ks_n, ks_o = kick_share(True, cap[-1]), kick_share(False, cap[-1])
ok = abs(ks_n - ks_o) < 1e-3; ok_all &= ok
say(f'4b. layer-1 kick share: new {ks_n:.4f}, split {ks_o:.4f} -> {ok}')
say(f'5. CPU, one train step (2 x 256): new {N_[4]:.1f} s, split {O_[4]:.1f} s')
say(f'\n-> {"ALL PASS" if ok_all else "FAILURES ABOVE"}')
(HERE / 'verify_pmx_fast_output.txt').write_text('\n'.join(lines) + '\n')
