"""PMX check: the Poisson-mode wells integrated exactly (protocol SS5.15, 2026-10-10).

OFF (POISSON_WELLS_EXACT = False) must be bit-identical to HEAD, from a
`git archive HEAD` copy: eval logits, train-mode loss and every gradient on
PM1's configuration and weights and on SR2's (SR2 exercises the patched
damped_mode_coefficients), and damped_mode_coefficients itself on a grid of
omega0^2 >= 0.

  python3 verify_pm_wells_exact.py OUT_DIR --dump     # run from the HEAD copy
  python3 verify_pm_wells_exact.py OUT_DIR            # working copy: compare + ON checks

ON (F3.1's Cell 0 + POISSON_MODES = 64 + LOWRANK_DAMPED_FLOW + POISSON_WELLS_EXACT),
on PM1's trained weights:
  0. unit: damped_mode_coefficients for omega0^2 < 0 against RK4; the
     indefinite modes reconstruct B diag(s) B^T; the full-space exact
     substep against RK4; two half steps equal one (frozen quadratic)
  1. tag 'pmx' and the 5b banner; the model carries the switch
  2. poisson_mode_quadratic's force equals poisson_mode_force's
  3. consistency: PMX and the explicit-wells step integrate the same ODE --
     their relative difference falls at second order (ratio at dt 0.5 and
     0.25 >= 3.5; smaller dt hits the float32 floor of the wells code)
  4. kick share at layer 1 (|h(no kick) - h| / |h - h_in|), PMX against the
     explicit wells: the explicit part of the step should collapse
  5. causality: tokens replaced at t >= 128 leave t < 128 unchanged
  6. gradients finite and reaching pm_depth, pm_mu, pm_log_kappa2
  7. per-layer flow: substeps_per_layer = 4 takes one quadratic per layer;
     k = 1 equals the default
  8. guards: PMX without SR2, with langevin_T > 0, and with an eigensolve wider
     than 32 (lowrank_max_modes + poisson_wells_exact_modes), refuse to build
  9. the eigensolve is exactly 32 columns wide (the GPU's batched fast path),
     and the wall time of one eval forward at the real size, 16 x 512, against SR2
"""
import sys, time
from pathlib import Path

import numpy as np
import torch

OUT = Path(sys.argv[1]); DUMP = '--dump' in sys.argv
sys.argv = [sys.argv[0], str(OUT / 'harness')]
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import verify_pm_switch as P

_B = 'semsimula_fock_cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_'
_E = 'L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn'
SR2_ON = (("LOWRANK_DAMPED_FLOW  = False", "LOWRANK_DAMPED_FLOW  = True"),)
PMX_ON = (("POISSON_WELLS_EXACT  = False", "POISSON_WELLS_EXACT  = True"),)
PM1F, SR2F = _B + 'pm64_' + _E, _B + 'sr2_' + _E
F31 = P.CONFIGS['F3.1'][1]
OFF_ARMS = {'PM1': (PM1F, F31 + P.PM_ON), 'SR2': (SR2F, F31 + SR2_ON)}
W2_GRID = torch.cat([torch.zeros(1), torch.logspace(-12, 2, 57, dtype=torch.float64)])


def record(model, x, y):
    model.eval()
    with torch.enable_grad():
        ev, _ = model(x)
    model.train(); torch.manual_seed(7)
    tr, loss = model(x, y)
    loss.backward()
    grads = {k: p.grad.detach().clone() for k, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    return {'eval': ev.detach(), 'train': tr.detach(), 'loss': loss.item(), 'grads': grads}


def coeffs():
    import cfc_baoab as C
    return {(g, t): C.damped_mode_coefficients(W2_GRID, g, t) for g in (0.0, 0.1, 3.0) for t in (0.5, 2.0)}


x, y = P.tokens(n_seq=2, T=256)
if DUMP:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (folder, repl) in OFF_ARMS.items():
        model, g, _ = P.build(folder, repl)
        torch.save(record(model, x, y), OUT / f'head_{name}.pt')
        print(f'dumped HEAD {name}')
    torch.save(coeffs(), OUT / 'head_coeffs.pt')
    print('dumped HEAD damped_mode_coefficients grid')
    raise SystemExit(0)

lines = []
say = lambda s='': (print(s, flush=True), lines.append(s))
ok_all = True

# ---- OFF ----
say('OFF (POISSON_WELLS_EXACT = False): working copy vs HEAD, on the trained weights')
for name, (folder, repl) in OFF_ARMS.items():
    model, g, _ = P.build(folder, repl)
    assert not model.cfg.poisson_wells_exact
    head = torch.load(OUT / f'head_{name}.pt', weights_only=False)
    cur = record(model, x, y)
    d_eval = float((cur['eval'] - head['eval']).abs().max())
    d_train = float((cur['train'] - head['train']).abs().max())
    d_grad = max(float((cur['grads'][k] - head['grads'][k]).abs().max()) for k in head['grads'])
    same = set(cur['grads']) == set(head['grads'])
    ok = d_eval == 0 and d_train == 0 and d_grad == 0 and cur['loss'] == head['loss'] and same
    ok_all &= ok
    say(f'   {name}: eval {d_eval:.1e}  train {d_train:.1e}  grads {d_grad:.1e}  ({len(head["grads"])} params) -> {"IDENTICAL" if ok else "DIFFERS"}')
hc, cc = torch.load(OUT / 'head_coeffs.pt', weights_only=False), coeffs()
d = max(float((a - b).abs().max()) for k in hc for a, b in zip(hc[k], cc[k]))
ok_all &= d == 0.0
say(f'   damped_mode_coefficients on omega0^2 in {{0}} U [1e-12, 1e2], 3 dampings x 2 steps: max |d| {d:.1e} -> {"IDENTICAL" if d == 0 else "DIFFERS"}')

# ---- ON ----
say('\nON, unit tests (float64)')
import cfc_baoab as C
torch.manual_seed(0)
def rk4_scalar(w2, gm, t, v0, a, n=20000):
    xx, vv, hh = 0.0, v0, t / n
    f = lambda xx, vv: (vv, a - gm * vv - w2 * xx)
    for _ in range(n):
        k1 = f(xx, vv); k2 = f(xx + hh/2*k1[0], vv + hh/2*k1[1]); k3 = f(xx + hh/2*k2[0], vv + hh/2*k2[1]); k4 = f(xx + hh*k3[0], vv + hh*k3[1])
        xx += hh/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0]); vv += hh/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
    return xx, vv
worst = 0.0
for w2 in (-0.6, -0.05, -1e-3):
    for gm in (0.0, 0.1, 3.0):
        E12, E22, Pc, Qc = C.damped_mode_coefficients(torch.tensor([w2], dtype=torch.float64), gm, 2.0)
        xr, vr = rk4_scalar(w2, gm, 2.0, 0.7, 1.3)
        worst = max(worst, max(abs(float(E12*0.7 + Qc*1.3) - xr), abs(float(E22*0.7 + Pc*1.3) - vr)) / (abs(xr) + abs(vr)))
ok = worst < 1e-8; ok_all &= ok
say(f'0a. damped_mode_coefficients, omega0^2 < 0, vs RK4: worst rel err {worst:.1e} -> {ok}')
d_, n_ = 40, 9
Bm = torch.randn(3, d_, n_, dtype=torch.float64); Bm[1, :, 4] = Bm[1, :, 3]
sg = torch.randn(3, n_, dtype=torch.float64)
U, kap = C.indefinite_lowrank_modes(Bm, sg)
Lm = Bm @ torch.diag_embed(sg) @ Bm.transpose(-1, -2)
Lr = U @ torch.diag_embed(kap) @ U.transpose(-1, -2)
rec = float((Lr - Lm).norm() / Lm.norm())
nz = U.norm(dim=-2) > 0.5
orth = float((U.transpose(-1, -2) @ U - torch.diag_embed(nz.double())).abs().max())
ok = rec < 1e-9 and orth < 1e-9 and bool((kap < 0).any()); ok_all &= ok
say(f'0b. indefinite modes: reconstruction {rec:.1e}, orthonormality {orth:.1e}, negative modes present -> {ok}')
K = 0.3 * torch.eye(d_, dtype=torch.float64) + Lr[0]; al = torch.tensor([[0.3]], dtype=torch.float64)
h0 = torch.randn(d_, dtype=torch.float64); v0 = torch.randn(d_, dtype=torch.float64); f0 = torch.randn(d_, dtype=torch.float64)
mm = torch.tensor([1.7], dtype=torch.float64); gm, Tt = 0.1, 2.0
hn, vn = C.lowrank_iso_damped_substep(h0[None], v0[None], U[0][None], kap[0][None], al, f0[None], mm, gm, Tt)
xx, vv, N = h0.clone(), v0.clone(), 20000; dd = Tt / N
acc = lambda xx, vv: (f0 - K @ (xx - h0)) / mm - gm * vv
for _ in range(N):
    k1x, k1v = vv, acc(xx, vv); k2x, k2v = vv + dd/2*k1v, acc(xx + dd/2*k1x, vv + dd/2*k1v)
    k3x, k3v = vv + dd/2*k2v, acc(xx + dd/2*k2x, vv + dd/2*k2v); k4x, k4v = vv + dd*k3v, acc(xx + dd*k3x, vv + dd*k3v)
    xx = xx + dd/6*(k1x+2*k2x+2*k3x+k4x); vv = vv + dd/6*(k1v+2*k2v+2*k3v+k4v)
e_sub = max(float((hn[0]-xx).norm()/xx.norm()), float((vn[0]-vv).norm()/vv.norm()))
h1, v1 = C.lowrank_iso_damped_substep(h0[None], v0[None], U[0][None], kap[0][None], al, f0[None], mm, gm, Tt/2)
h2, _ = C.lowrank_iso_damped_substep(h1, v1, U[0][None], kap[0][None], al, (f0 - K @ (h1[0] - h0))[None], mm, gm, Tt/2)
e_grp = float((h2 - hn).norm() / hn.norm())
ok = e_sub < 1e-10 and e_grp < 1e-12; ok_all &= ok
say(f'0c. full-space exact substep vs RK4: {e_sub:.1e}; two half steps vs one: {e_grp:.1e} -> {ok}')

say('\nON (F3.1 Cell 0 + POISSON_MODES = 64 + LOWRANK_DAMPED_FLOW + POISSON_WELLS_EXACT), on PM1\'s trained weights')
model, g, out5b = P.build(PM1F, F31 + P.PM_ON + SR2_ON + PMX_ON)
tag = g['_variant_tag']
ok = 'pmx16' in tag.split('_') and 'sr2' in tag.split('_') and model.cfg.poisson_wells_exact and model.cfg.poisson_wells_exact_modes == 16 and 'WELLS INTEGRATED EXACTLY' in out5b
ok_all &= ok
say(f'1. tag ...{tag[tag.find("pm64"):tag.find("_ob_")]}; switch on: {model.cfg.poisson_wells_exact}; 5b banner: {"WELLS INTEGRATED EXACTLY" in out5b} -> {ok}')

# capture layer inputs on the trained weights
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
assert model._fock_layer_step is orig_step

h1_, _, _, _, mb1, gm1, dt1, l1 = cap[-1]
with torch.enable_grad():
    Fq, _, _, _ = model.poisson_mode_quadratic(h1_, l1)
    Fe = model.poisson_mode_force(h1_, l1).float()
d = float((Fq - Fe).abs().max() / Fe.abs().max())
ok = d < 1e-6; ok_all &= ok
say(f'2. poisson_mode_quadratic force vs poisson_mode_force at layer {l1}: max rel |d| {d:.1e} -> {ok}')

def layer_step(cfg_pmx, h, hp, li, dt, mb, gmm):
    old = model.cfg.poisson_wells_exact
    model.cfg.poisson_wells_exact = cfg_pmx
    try:
        with torch.enable_grad():
            return model._layer_step_langevin(h, hp, mb, gmm, dt, layer_idx=li)[0].detach()
    finally:
        model.cfg.poisson_wells_exact = old
hh, hp, _, _, mb, gmm, _, li = cap[-1]
diffs = []
for dt_ in (0.5, 0.25):        # above the float32 floor (|h| ~ 20); second order -> ratio 4
    hp_ = hh - dt_ * (hh - hp) / 4.0                          # same velocity at the smaller dt
    a_ = layer_step(True, hh, hp_, li, dt_, mb, gmm); b_ = layer_step(False, hh, hp_, li, dt_, mb, gmm)
    diffs.append(float((a_ - b_).norm() / (b_ - hh).norm()))
ratio = diffs[0] / max(diffs[1], 1e-30)
ok = ratio >= 3.5; ok_all &= ok
say(f'3. consistency, PMX vs explicit wells, |d h| / |step| at dt 0.5, 0.25: {diffs[0]:.2e}, {diffs[1]:.2e}; ratio {ratio:.1f} (>= 3.5) -> {ok}')

def kick_share(cfg_pmx, item):
    h, hp, r, s, mb, gmm, dt, li = item
    old = model.cfg.poisson_wells_exact
    model.cfg.poisson_wells_exact = cfg_pmx
    try:
        with torch.enable_grad():
            full = model._fock_layer_step(h, hp, r, s, mb, gmm, dt, layer_idx=li)[0].detach()
            model.cfg.force_clamp_max = 0.0
            try:
                nok = model._fock_layer_step(h, hp, r, s, mb, gmm, dt, layer_idx=li)[0].detach()
            finally:
                model.cfg.force_clamp_max = None
    finally:
        model.cfg.poisson_wells_exact = old
    return float(((nok - full).norm(dim=-1) / (full - h).norm(dim=-1).clamp_min(1e-12)).median())
ks = {l: (kick_share(True, it), kick_share(False, it)) for it in cap for l in [it[7]]}
say('4. kick share, median |h(no kick) - h| / |h - h_in| (PMX against explicit wells, both on SR2):')
for l, (a_, b_) in sorted(ks.items()):
    say(f'      layer {l}: PMX {a_:.3f}   explicit wells {b_:.3f}')
ok = ks[max(ks)][0] < 0.5 * ks[max(ks)][1]; ok_all &= ok
say(f'   the explicit part at layer {max(ks)} at least halves -> {ok}')

model.eval()
with torch.enable_grad():
    base, _ = model(x)
xc = x.clone(); xc[:, 128:] = torch.randint(0, model.cfg.vocab_size, xc[:, 128:].shape, generator=torch.Generator().manual_seed(1))
with torch.enable_grad():
    pert, _ = model(xc)
d = float((pert[:, :128] - base[:, :128]).abs().max())
ok = d == 0.0; ok_all &= ok
say(f'5. causality: tokens replaced at t >= 128, max |d logit| at t < 128 = {d:.1e} -> {ok}')

model.train(); torch.manual_seed(7)
t0 = time.time()
_, loss = model(x, y); loss.backward()
t_pmx = time.time() - t0
gn = {k: (float(p.grad.norm()) if p.grad is not None else None) for k, p in model.named_parameters() if k.startswith('pm_')}
fin = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
ok = fin and all(gn.get(k) for k in ('pm_depth', 'pm_mu', 'pm_log_kappa2')); ok_all &= ok
say(f'6. gradients finite: {fin}; pm_ grad norms ' + ', '.join(f'{k} {v:.2e}' for k, v in gn.items() if v is not None) + f' -> {ok}')
model.zero_grad(set_to_none=True)
model.cfg.poisson_wells_exact = False
model.train(); torch.manual_seed(7); t0 = time.time()
_, loss = model(x, y); loss.backward(); t_exp = time.time() - t0
model.zero_grad(set_to_none=True); model.cfg.poisson_wells_exact = True
say(f'   CPU cost of one train step, 2 x 256 tokens: PMX {t_pmx:.1f} s, explicit wells {t_exp:.1f} s ({t_pmx / t_exp:.2f}x)')

calls = {'n': 0}
orig_q = model.poisson_mode_quadratic
def countq(h, layer_idx, **kw):
    calls['n'] += 1
    return orig_q(h, layer_idx, **kw)
model.poisson_mode_quadratic = countq
try:
    model.eval(); model.cfg.substeps_per_layer = 4
    with torch.enable_grad():
        lo4, _ = model(x)
    n4 = calls['n']
    model.cfg.substeps_per_layer = 1; calls['n'] = 0
    with torch.enable_grad():
        lo1, _ = model(x)
    n1 = calls['n']
finally:
    model.__dict__.pop('poisson_mode_quadratic', None)
d = float((lo1 - base).abs().max())
ok = n4 == model.cfg.L and d == 0.0 and bool(torch.isfinite(lo4).all()); ok_all &= ok
say(f'7. per-layer flow: quadratics taken per forward at k = 4: {n4} (one per layer, L = {model.cfg.L}); '
    f'at k = 1: {n1} (the layer-checkpoint wrapper runs each step twice, as for every model); k = 1 vs default max |d| {d:.1e} -> {ok}')

refused = []
import copy
for why in ('no SR2', 'T > 0', 'wider than 32'):
    try:
        if why == 'no SR2':
            P.build(PM1F, F31 + P.PM_ON + PMX_ON)
        elif why == 'wider than 32':
            cfg2 = copy.deepcopy(model.cfg); cfg2.poisson_wells_exact_modes = 24
            type(model)(cfg2)
        else:
            cfg2 = copy.deepcopy(model.cfg); cfg2.langevin_T = 0.1
            type(model)(cfg2)
        refused.append(False)
    except Exception as e:                       # a refusal only if it names PMX
        refused.append('PMX' in str(e) or 'poisson_wells_exact' in str(e))
ok = all(refused); ok_all &= ok
say(f'8. guards: refuses without SR2 {refused[0]}, with langevin_T > 0 {refused[1]}, wider than 32 {refused[2]} -> {ok}')

import model_parf_multixi as MPX
widths = []
orig_ilm = MPX.indefinite_lowrank_modes_split             # the routine PMX calls (2026-10-10)
def ilm(U0, k0, R, dW, *a, **k):
    widths.append(U0.shape[-1] + R.shape[-1]); return orig_ilm(U0, k0, R, dW, *a, **k)
MPX.indefinite_lowrank_modes_split = ilm
try:
    xb, _ = g['get_batch'](g['val_ids'], 16, 512, np.random.default_rng(20260920))
    xf = torch.from_numpy(xb)
    model.eval(); t0 = time.time()
    with torch.enable_grad():
        model(xf)
    t_pmx_full = time.time() - t0
finally:
    MPX.indefinite_lowrank_modes_split = orig_ilm
model.cfg.poisson_wells_exact = False; model.cfg.lowrank_damped_flow = True
t0 = time.time()
with torch.enable_grad():
    model(xf)
t_sr2_full = time.time() - t0
model.cfg.poisson_wells_exact = True
ok = len(widths) > 0 and set(widths) == {32}; ok_all &= ok
say(f'9. eigensolve widths seen: {sorted(set(widths))} (must be 32) -> {ok}; '
    f'eval forward 16 x 512 on the CPU: PMX {t_pmx_full:.1f} s, explicit wells on SR2 {t_sr2_full:.1f} s ({t_pmx_full / t_sr2_full:.2f}x)')

say(f'\n-> {"ALL PASS" if ok_all else "FAILURES ABOVE"}')
(HERE / 'verify_pm_wells_exact_output.txt').write_text('\n'.join(lines) + '\n')
