#!/usr/bin/env python3
"""V_theta's stiffest-mode omega*dt with ONE definition for both orders (2026-10-23; M3 of the FO-b K = 16 twin).

    python3 vtheta_omega_dt.py OUT_DIR {pmx|fob-pmx} [threads=16] [folder=<run folder>]

Cell 6b-13 reads different quantities in the two orders: for second order the semsimula-diag monitor reads V_theta's
stiffest eigenvalue per token (opened by harmonic_terms_lowrank, closed by the damped substep); for FO-b on the well
base the cell records kappa + alpha, V_theta's retained stiffness plus the wells' isotropic curvature. Here both
arms use the monitor itself: under FO-b its record is closed from lowrank_iso_overdamped_substep, with the token's
mass recovered from FO-b's mobility, m = dt_trained / (2 mu), and the same kick step (2 x dt/2 = dt) as the
second-order wrapper uses. Cell 6b-13's batches: seed 20260928, 8 x 4 x 512 validation tokens. For the second-order
arm the result must equal Cell 6b-13's (PMX-K16: p50 3.651).
"""
import sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(sys.argv[1]); KIND = sys.argv[2]
THREADS = int(next((a[8:] for a in sys.argv[3:] if a.startswith('threads=')), 16))
FOLDER_ARG = next((a[7:] for a in sys.argv[3:] if a.startswith('folder=')), None)
assert KIND in ('pmx', 'fob-pmx')
sys.argv = [sys.argv[0], str(OUT / 'harness')]
sys.path.insert(0, str(Path(__file__).parent))
import verify_pm_switch as P
sys.path.insert(0, str(Path(__file__).resolve().parents[5] / 'semsimula-diag' / 'src'))
from semsimula_diag.probes import resonance
torch.set_num_threads(THREADS)

PFX = ('semsimula_fock_cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_')
SFX = '_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn'
FOLDER = FOLDER_ARG or (PFX + ('pm64_pmx16_sr2_fob' if KIND == 'fob-pmx' else 'pm64_pmx16_sr2') + SFX)
repl = P.CONFIGS['F3.1'][1] + P.PM_ON + (("LOWRANK_DAMPED_FLOW  = False", "LOWRANK_DAMPED_FLOW  = True"),
                                         ("POISSON_WELLS_EXACT  = False", "POISSON_WELLS_EXACT  = True"))
_k = next((int(t[2:]) for t in FOLDER.split('_') if t.startswith('pm') and t[2:].isdigit()), 64)
if _k != 64:
    repl = repl + (("POISSON_MODES        = 64", f"POISSON_MODES        = {_k}"),)
if KIND == 'fob-pmx':
    repl = repl + (("FIRST_ORDER_OVERDAMPED = False", "FIRST_ORDER_OVERDAMPED = True"),)
model, g, _ = P.build(FOLDER, repl)
model.to('cpu'); model.eval(); model.cfg.use_layer_checkpoint = False
assert bool(model.cfg.first_order_overdamped) == (KIND == 'fob-pmx') and model.pm_depth.shape[1] == _k
mod = sys.modules[type(model)._layer_step_langevin.__module__]
dt = float(model.cfg.dt); dtt = float(getattr(model, '_fob_dt', dt))

rng = np.random.default_rng(20260928)
batches = [tuple(torch.from_numpy(a) for a in g['get_batch'](g['val_ids'], 4, 512, rng)) for _ in range(8)]
per_layer = {}
for xb, yb in batches:
    with resonance.observe(model, mod, wall=2.0) as mon:
        if KIND == 'fob-pmx':
            orig = mod.lowrank_iso_overdamped_substep
            def rec(hh, U, kappa, alpha, f, mu, dt_sub, *a, **k):
                m = dtt / (2.0 * (mu if torch.is_tensor(mu) else torch.tensor(float(mu))))
                mon._finish(m, dt / 2.0)
                return orig(hh, U, kappa, alpha, f, mu, dt_sub, *a, **k)
            mod.lowrank_iso_overdamped_substep = rec
        try:
            with torch.enable_grad():
                model(xb, yb)
        finally:
            if KIND == 'fob-pmx':
                mod.lowrank_iso_overdamped_substep = orig
    for lay, rows in mon.per_layer.items():
        per_layer.setdefault(lay, []).extend(rows)
assert per_layer, 'no reading: the record was never closed'

print(f'V_THETA STIFFEST-MODE OMEGA*DT: {KIND}  K = {_k}  (one definition for both orders; Cell 6b-13 batches, '
      f'seed 20260928, 8 x 4 x 512 tokens; dt = {dt:g})')
print(f'   {"layer":>6} {"p05":>8} {"p50":>8} {"p95":>8} {"max":>8} {"over 2.0":>9}')
allv = []
for lay in sorted(per_layer):
    t = torch.cat(per_layer[lay]); allv.append(t)
    print(f'   {lay:6d} {float(t.quantile(.05)):8.3f} {float(t.quantile(.5)):8.3f} {float(t.quantile(.95)):8.3f} '
          f'{float(t.max()):8.3f} {100 * float((t > 2.0).float().mean()):8.2f}%')
t = torch.cat(allv)
print(f'   {"ALL":>6} {float(t.quantile(.05)):8.3f} {float(t.quantile(.5)):8.3f} {float(t.quantile(.95)):8.3f} '
      f'{float(t.max()):8.3f} {100 * float((t > 2.0).float().mean()):8.2f}%   ({t.numel():,} readings)')
