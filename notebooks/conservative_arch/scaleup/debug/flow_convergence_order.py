#!/usr/bin/env python3
"""Convergence order of the per-layer flow, second order against first order (2026-10-24).

    python3 flow_convergence_order.py OUT_DIR {pmx|fob-pmx} {trained|init} [folder=<run folder>] [threads=16] [kmax=64] [chunk=4] [double=1]

Why: every second-order model's per-layer flow (FLOW-R) converges by N = 16, and no first-order model's does. The
candidate explanation is numerical: the second-order layer step (palindromic A-B-O-A, kick at the midpoint) is a
second-order-accurate splitting, while FO-b's step (exact relaxation, ONE explicit Euler step for everything else,
relaxation) is first order in the step, so its per-layer flow converges as 1/k (model_parf_multixi._overdamped_step's
docstring) where the second-order flow converges as 1/k^2.

Method: the model's own per-layer flow (cfg.substeps_per_layer = k, identical to FLOW-R's per-layer flow, checked
there to 0.0 in the logits), k = 1, 2, 4, ..., kmax substeps per trained layer, against a reference at 2 kmax. For each
k: the relative error of the logits, ||logits_k - logits_ref|| / ||logits_ref - logits_k=1|| (so 1.0 at k = 1), and the
perplexity. The observed order between k and 2k is log2(error_k / error_2k): about 1 for a first-order integrator,
about 2 for a second-order one, once k is in the asymptotic range.

  trained   the best checkpoint of the run folder (PMX-K16 or FO-b-PMX-K16 by default)
  init      the same configuration at initialisation, no checkpoint: the integrator's own order, with no learned
            compensation. The Poisson-mode depths start at zero, so at init the wells exert no force; the landscape
            is V_theta's and V_phi's initial one.
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(sys.argv[1]); KIND = sys.argv[2]; WHICH = sys.argv[3]
THREADS = int(next((a[8:] for a in sys.argv[4:] if a.startswith('threads=')), 16))
KMAX = int(next((a[5:] for a in sys.argv[4:] if a.startswith('kmax=')), 64))
DOUBLE = 'double=1' in sys.argv[4:]
CHUNK = int(next((a[6:] for a in sys.argv[4:] if a.startswith('chunk=')), 4))
FOLDER_ARG = next((a[7:] for a in sys.argv[4:] if a.startswith('folder=')), None)
assert KIND in ('pmx', 'fob-pmx') and WHICH in ('trained', 'init')
sys.argv = [sys.argv[0], str(OUT / f'harness_{KIND}_{WHICH}')]
sys.path.insert(0, str(Path(__file__).parent))
import verify_pm_switch as P
torch.set_num_threads(THREADS)

PFX = 'semsimula_fock_cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_'
SFX = '_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn'
FOLDER = FOLDER_ARG or (PFX + ('pm16_pmx16_sr2_fob' if KIND == 'fob-pmx' else 'pm16_pmx16_sr2') + SFX)
_k = next((int(t[2:]) for t in FOLDER.split('_') if t.startswith('pm') and t[2:].isdigit()), 64)
repl = P.CONFIGS['F3.1'][1] + (("POISSON_MODES        = 0", f"POISSON_MODES        = {_k}"),
                               ("LOWRANK_DAMPED_FLOW  = False", "LOWRANK_DAMPED_FLOW  = True"),
                               ("POISSON_WELLS_EXACT  = False", "POISSON_WELLS_EXACT  = True"))
if KIND == 'fob-pmx':
    repl = repl + (("FIRST_ORDER_OVERDAMPED = False", "FIRST_ORDER_OVERDAMPED = True"),)


class _Stop(Exception):
    pass


if WHICH == 'trained':
    model, g, _ = P.build(FOLDER, repl)
else:
    load = torch.load
    torch.load = lambda *a, **k: (_ for _ in ()).throw(_Stop())
    torch.manual_seed(0); np.random.seed(0)
    try:
        P.build(FOLDER, repl)
    except _Stop:
        tb = sys.exc_info()[2]
        while tb:
            fl = tb.tb_frame.f_locals
            if tb.tb_frame.f_code.co_name == 'build' and 'model' in fl:
                model, g = fl['model'], fl['g']
            tb = tb.tb_next
    finally:
        torch.load = load
model.to('cpu'); model.eval(); model.cfg.use_layer_checkpoint = False
if DOUBLE:  # float64 throughout: separates a round-off floor from the flow's own error
    model.double(); torch.set_default_dtype(torch.float64)
assert bool(model.cfg.first_order_overdamped) == (KIND == 'fob-pmx')

rng = np.random.default_rng(20261009)
xb, yb = g['get_batch'](g['val_ids'], 4, 256, rng)
x, y = torch.from_numpy(xb), torch.from_numpy(yb)

def run(k):
    # One sequence at a time when chunk=1: under enable_grad the forward keeps its graph through every substep
    # until the logits are detached, so peak memory grows with k and with the batch; sequences are independent.
    model.cfg.substeps_per_layer = k
    los = []
    for i in range(0, x.shape[0], CHUNK):
        with torch.enable_grad():
            los.append(model(x[i:i + CHUNK])[0].detach().double())
    lo = torch.cat(los)
    ppl = math.exp(float(torch.nn.functional.cross_entropy(lo.reshape(-1, lo.shape[-1]).float(), y.reshape(-1))))
    return lo, ppl

ks = [2 ** i for i in range(int(math.log2(KMAX)) + 1)]
out = {k: run(k) for k in ks}
ref, ppl_ref = run(2 * KMAX)
model.cfg.substeps_per_layer = 1
scale = (ref - out[1][0]).norm()
err = {k: float((out[k][0] - ref).norm() / scale) for k in ks}
print(f'FLOW CONVERGENCE ORDER: {KIND} ({"second order" if KIND == "pmx" else "first order, FO-b"}), {WHICH}, '
      f'K = {_k}; 4 x 256 validation tokens; reference k = {2 * KMAX} substeps per layer (perplexity {ppl_ref:.2f})')
print(f'   {"k":>4} {"N = 2k":>7} {"perplexity":>11} {"rel. error":>11} {"order k -> 2k":>14}')
for i, k in enumerate(ks):
    order = math.log2(err[k] / err[ks[i + 1]]) if i + 1 < len(ks) and err[ks[i + 1]] > 0 else float('nan')
    print(f'   {k:4d} {2 * k:7d} {out[k][1]:11.2f} {err[k]:11.4f} {order:14.2f}')
