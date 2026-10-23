#!/usr/bin/env python3
"""Do the Poisson-mode wells remember content, or only position? (2026-10-23, after pm_mode_usage.py.)

    python3 pm_content_swap.py OUT_DIR {pmx|fob-pmx} [batches=4] [threads=16] [ckpt=<file>] [folder=<run folder>]

Inference only, best checkpoint. The occupations phi_v(t) = sum_{s<t} lambda_v^(t-1-s) E_v(s) are replaced, wherever
the model computes them (the wells' force and the integrator's quadratic), by:

  swap      the phi of another sequence in the batch (the batch rolled by one): same positions, different content
  posmean   the mean phi over the batch's sequences at each position: position only, no content
  off       zero: no wells at all (the size of the wells' whole contribution, for scale)

and the perplexity on the same tokens is compared with the unmodified model, paired by sequence. The token's own
overlap E_v(t) is left as it is. If swap and posmean cost nothing while off costs a lot, the wells act as a learned
position-dependent potential and carry no memory of what the earlier tokens were.
"""
import sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(sys.argv[1]); KIND = sys.argv[2]
NB_ = int(next((a[8:] for a in sys.argv[3:] if a.startswith('batches=')), 4))
THREADS = int(next((a[8:] for a in sys.argv[3:] if a.startswith('threads=')), 16))
CKPT = next((a[5:] for a in sys.argv[3:] if a.startswith('ckpt=')), None)
FOLDER_ARG = next((a[7:] for a in sys.argv[3:] if a.startswith('folder=')), None)
assert KIND in ('pmx', 'fob-pmx')
sys.argv = [sys.argv[0], str(OUT / 'harness')]
sys.path.insert(0, str(Path(__file__).parent))
import verify_pm_switch as P
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
model, g, _ = P.build(FOLDER, repl, ckpt=CKPT)
model.to('cpu'); model.eval(); model.cfg.use_layer_checkpoint = False
assert bool(model.cfg.first_order_overdamped) == (KIND == 'fob-pmx') and model.cfg.poisson_wells_exact

rng = np.random.default_rng(20261009)
batches = [g['get_batch'](g['val_ids'], 4, 512, rng) for _ in range(NB_)]
orig_occ = model.poisson_mode_occupation
MODE = {'v': 'base'}
def occ(h):
    E, phi = orig_occ(h)
    if MODE['v'] == 'swap':
        phi = phi.roll(1, dims=0)
    elif MODE['v'] == 'posmean':
        phi = phi.mean(0, keepdim=True).expand_as(phi)
    elif MODE['v'] == 'off':
        phi = torch.zeros_like(phi)
    return E, phi
model.poisson_mode_occupation = occ

nll = {}
for v in ('base', 'swap', 'posmean', 'off'):
    MODE['v'] = v; rows = []
    for xb, yb in batches:
        with torch.enable_grad():
            lo = model(torch.from_numpy(xb))[0].detach()
        ce = torch.nn.functional.cross_entropy(lo.reshape(-1, lo.shape[-1]), torch.from_numpy(yb).reshape(-1), reduction='none')
        rows.append(ce.reshape(xb.shape).mean(1).numpy())          # per-sequence mean NLL
    nll[v] = np.concatenate(rows)
print(f'POISSON-MODE CONTENT SWAP: {KIND}  K = {_k}  ({CKPT or "best checkpoint"}; {NB_} x 4 x 512 tokens, batch seed 20261009)')
print(f'   base perplexity {np.exp(nll["base"].mean()):.2f}')
for v, what in (('swap', "another sequence's occupations (same positions, other content)"),
                ('posmean', 'occupations averaged over sequences at each position (position only)'),
                ('off', 'no wells (occupations zero)')):
    d = nll[v] - nll['base']
    print(f'   {v:8s} perplexity {np.exp(nll[v].mean()):8.2f}   change {np.expm1(d.mean()) * 100:+7.2f}%  '
          f'(per-sequence spread {d.std(ddof=1) * 100:.2f}%, n = {len(d)})   {what}')
