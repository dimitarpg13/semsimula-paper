#!/usr/bin/env python3
"""Cell 6b-13's reading for FO-b-PMX, on CPU (2026-10-22): omega*dt of the retained modes at the endpoint.

    python3 fobpmx_omega_dt.py OUT_DIR [threads=16]

The FO-b-on-PMX step calls lowrank_iso_overdamped_substep with PMX's frozen spring K = alpha I + U diag(kappa) U^T,
so the modes in span(U) have stiffness kappa + alpha; omega*dt = sqrt((kappa + alpha) / m) * dt with
1/m = 2 mu / dt_trained (FO-b's mobility mu = dt_trained / 2m), the first call of each layer step kept, as the
notebook's recorder of 2026-10-22 does. Cell 6b-13's batches: seed 20260928, 8 x 4 x 512 validation tokens.
Also: the relaxation exponent mu * kappa_eff * dt per layer step (above 3 a mode is fully relaxed within the step).
"""
import sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(sys.argv[1])
THREADS = int(next((a[8:] for a in sys.argv[2:] if a.startswith('threads=')), 16))
sys.argv = [sys.argv[0], str(OUT / 'harness')]
sys.path.insert(0, str(Path(__file__).parent))
import verify_pm_switch as P
torch.set_num_threads(THREADS)

FOLDER = ('semsimula_fock_cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_'
          'pm64_pmx16_sr2_fob_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn')
repl = P.CONFIGS['F3.1'][1] + P.PM_ON + (("LOWRANK_DAMPED_FLOW  = False", "LOWRANK_DAMPED_FLOW  = True"),
                                         ("POISSON_WELLS_EXACT  = False", "POISSON_WELLS_EXACT  = True"),
                                         ("FIRST_ORDER_OVERDAMPED = False", "FIRST_ORDER_OVERDAMPED = True"))
model, g, _ = P.build(FOLDER, repl)
model.to('cpu'); model.eval(); model.cfg.use_layer_checkpoint = False
assert model.cfg.first_order_overdamped and model.cfg.poisson_wells_exact
dt = float(model.cfg.dt); dtt = float(getattr(model, '_fob_dt', dt))
rng = np.random.default_rng(20260928)
batches = [tuple(torch.from_numpy(a) for a in g['get_batch'](g['val_ids'], 4, 512, rng)) for _ in range(8)]

mod = sys.modules[type(model)._layer_step_langevin.__module__]
orig_iso = mod.lowrank_iso_overdamped_substep
orig_step = model._layer_step_langevin
cur = [0, 0]; per_layer = {}; relax = {}
def step(h, h_prev, m_b, gamma, dt_, layer_idx=0):
    cur[0], cur[1] = int(layer_idx), 0
    return orig_step(h, h_prev, m_b, gamma, dt_, layer_idx)
def rec_iso(hh, U, kappa, alpha, f, mu, dt_):
    first = cur[1] == 0; cur[1] += 1
    if first:
        with torch.no_grad():
            ke = kappa + (alpha.unsqueeze(-1) if torch.is_tensor(alpha) and alpha.dim() == kappa.dim() - 1 else alpha)
            kp = ke.detach().float().clamp(min=0.0)
            mu_ = mu.detach().float() if torch.is_tensor(mu) else torch.tensor(float(mu))
            wdt = (kp * (2.0 * mu_ / dtt)).sqrt() * dt
            per_layer.setdefault(cur[0], []).append(wdt.flatten())
            relax.setdefault(cur[0], []).append((mu_ * kp * dt).flatten())
    return orig_iso(hh, U, kappa, alpha, f, mu, dt_)
mod.lowrank_iso_overdamped_substep = rec_iso
model._layer_step_langevin = step
try:
    for i, (xb, yb) in enumerate(batches):
        with torch.enable_grad():
            model(xb, yb)
        print(f'batch {i + 1}/8', flush=True)
finally:
    mod.lowrank_iso_overdamped_substep = orig_iso
    del model._layer_step_langevin
q = lambda t, p: float(t.quantile(p))
lines = [f'FO-b-PMX omega*dt at the endpoint (Cell 6b-13 logic on CPU), best checkpoint; 8 x 4 x 512 tokens, seed 20260928',
         f'   stiffness kappa + alpha of the modes in span(U) (V_theta modes and the top-16 wells, 32 per token); dt = {dt:g}',
         f'\nFO-b relaxation exponent mu*kappa*dt per layer step\n   {"layer":>6} {"p05":>8} {"p50":>8} {"p95":>8} {"relaxed":>9}']
for l in sorted(relax):
    t = torch.cat(relax[l]); lines.append(f'   {l:6d} {q(t, .05):8.3f} {q(t, .5):8.3f} {q(t, .95):8.3f} {100 * float((t > 3).float().mean()):8.2f}%')
flat = torch.cat([torch.cat(v) for v in per_layer.values()])
lines.append(f'\nomega*dt over {flat.numel():,} (token, mode, layer) readings\n   {"layer":>6} {"p05":>8} {"p50":>8} {"p95":>8} {"max":>8} {"over 2.0":>9}')
for l in sorted(per_layer):
    t = torch.cat(per_layer[l]); lines.append(f'   {l:6d} {q(t, .05):8.3f} {q(t, .5):8.3f} {q(t, .95):8.3f} {float(t.max()):8.3f} {100 * float((t > 2).float().mean()):8.2f}%')
lines.append(f'   {"ALL":>6} {q(flat, .05):8.3f} {q(flat, .5):8.3f} {q(flat, .95):8.3f} {float(flat.max()):8.3f} {100 * float((flat > 2).float().mean()):8.2f}%')
lines.append('\n   NOTE: PMX and the SR2-base arms report the stiffest mode per token (lambda_max); these are all 32 retained modes per token.')
print('\n'.join(lines))
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'fobpmx_omega_dt_output.txt').write_text('\n'.join(lines) + '\n')
