#!/usr/bin/env python3
"""Clock or fixed landscape? What the position-only wells contribute, by position (2026-10-23).

    python3 pm_static_landscape.py OUT_DIR {pmx|fob-pmx} [batches=4] [threads=16] [ckpt=<file>] [folder=<run folder>]

After pm_mode_usage.py and pm_content_swap.py (the occupations are a function of position, and the models do not use
their content). An occupation phi_v(t) ~ E_v (1 - lambda_v^t) / (1 - lambda_v) rises over a few half-lives (trained
3-20 tokens) and then levels off, so beyond the first ~100 tokens the wells may be a FIXED extra landscape. Inference
only, best checkpoint; the perplexity is compared with the unmodified model per position bucket, paired by sequence:

  static        every phi_v(t) replaced by its late value, the mean over the batch and positions 256-511: a
                position-independent landscape at every position
  static_128    the real phi for t < 128, the late value from t = 128 on
  off           no wells

If `static` costs only in the first buckets, the wells are a fixed landscape plus a start-of-sequence clock.
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
assert bool(model.cfg.first_order_overdamped) == (KIND == 'fob-pmx') and model.pm_depth.shape[1] == _k

rng = np.random.default_rng(20261009)
batches = [g['get_batch'](g['val_ids'], 4, 512, rng) for _ in range(NB_)]
MODE = {'v': 'base'}
orig_occ = model.poisson_mode_occupation
def occ(h):
    E, phi = orig_occ(h)
    v = MODE['v']
    if v in ('static', 'static_128'):
        late = phi[:, 256:].mean((0, 1), keepdim=True)                # (1, 1, K)
        t0 = 0 if v == 'static' else 128
        phi = torch.cat([phi[:, :t0], late.expand(phi.shape[0], phi.shape[1] - t0, -1)], dim=1)
    elif v == 'off':
        phi = torch.zeros_like(phi)
    return E, phi
model.poisson_mode_occupation = occ

BUCKETS = [(0, 16), (16, 64), (64, 128), (128, 256), (256, 512)]
nll = {}
for v in ('base', 'static', 'static_128', 'off'):
    MODE['v'] = v; rows = []
    for xb, yb in batches:
        with torch.enable_grad():
            lo = model(torch.from_numpy(xb))[0].detach()
        ce = torch.nn.functional.cross_entropy(lo.reshape(-1, lo.shape[-1]), torch.from_numpy(yb).reshape(-1), reduction='none')
        rows.append(ce.reshape(xb.shape).numpy())                     # (4, 512) per-token NLL
    nll[v] = np.concatenate(rows)
MODE['v'] = 'base'

print(f'WELLS: CLOCK OR FIXED LANDSCAPE? {KIND}  K = {_k}  ({CKPT or "best checkpoint"}; {NB_} x 4 x 512 tokens, '
      f'batch seed 20261009)')
print(f'   base perplexity {np.exp(nll["base"].mean()):.2f}')
print(f'   perplexity change by position bucket (paired by sequence; per-sequence spread in brackets)')
print(f'   {"variant":11s} {"all":>17s}' + ''.join(f'{f"t {a}-{b - 1}":>17s}' for a, b in BUCKETS))
for v in ('static', 'static_128', 'off'):
    cells = []
    for a, b in [(0, 512)] + BUCKETS:
        d = nll[v][:, a:b].mean(1) - nll['base'][:, a:b].mean(1)
        cells.append(f'{np.expm1(d.mean()) * 100:+8.2f}% ({d.std(ddof=1) * 100:5.2f})')
    print(f'   {v:11s} ' + ' '.join(f'{c:>16s}' for c in cells))
print(f'   base perplexity by bucket: ' + ', '.join(f't {a}-{b - 1}: {np.exp(nll["base"][:, a:b].mean()):.1f}' for a, b in BUCKETS))
