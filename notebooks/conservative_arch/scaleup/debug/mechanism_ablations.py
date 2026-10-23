#!/usr/bin/env python3
"""Where does a model keep what its velocity would carry? Inference ablations of the content paths (2026-10-23).

    python3 mechanism_ablations.py OUT_DIR {pmx|fob-pmx} [batches=4] [threads=16] [ckpt=<file>] [folder=<run folder>]
                                   [variants=base,xi_ctx_swap,...]

Registered with the FO-b K = 16 twin (protocol §5.13, "what replaces the velocity", M1-M4), before its result. Best
checkpoint, inference only; each variant changes one path and the perplexity on the same tokens is compared with the
unmodified model, paired by sequence:

  xi_swap       the context channels xi (all five, wherever the model reads them: V_theta's wells, the CfC
                linearisation) taken from another sequence in the batch: same positions, other content
  xi_posmean    xi averaged over the batch's sequences at each position: no content
  xi_ctx_swap   only the three CONTEXT channels of xi (channels 2-4, trained alpha ~0.88-0.997) from another
                sequence; the two short channels (alpha ~0.3-0.5), which mostly carry the current token itself
                (xi includes s = t), are kept. Added 2026-10-23: the full xi_swap costs ~10,000% in every model
                because it replaces the token's identity, so it cannot compare how much two models keep in context
  xi_ctx_posmean  the context channels averaged over sequences at each position; the short channels kept
  vphi_off      the pair force f_phi = -grad V_phi set to zero: no token-token coupling
  wells_swap    the Poisson-mode occupations from another sequence (as pm_content_swap.py)
  wells_off     no Poisson-mode wells
  content_swap  xi_swap and wells_swap together

The model's content enters through xi and V_phi only (no positional encoding; the wells carry position, see
pm_mode_usage.py), so a model that stores more in content than its twin pays more for xi_swap and vphi_off.
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
ONLY = next((a[9:].split(',') for a in sys.argv[3:] if a.startswith('variants=')), None)
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
ON = set()

alphas = [float(a) for a in model.xi_module.alpha.detach()]
CTX = 2
assert len(alphas) == 5 and max(alphas[:CTX]) < 0.7 < min(alphas[CTX:]), alphas   # short channels first
orig_xi = model.xi_module.forward
def xi_fwd(*a, **k):
    xis = orig_xi(*a, **k)
    if 'xi_swap' in ON:
        xis = xis.roll(1, dims=0)
    elif 'xi_posmean' in ON:
        xis = xis.mean(0, keepdim=True).expand_as(xis)
    elif 'xi_ctx_swap' in ON:                      # xis: (B, T, channels, d); channels ordered by alpha
        xis = torch.cat([xis[:, :, :CTX], xis[:, :, CTX:].roll(1, dims=0)], dim=2)
    elif 'xi_ctx_posmean' in ON:
        xis = torch.cat([xis[:, :, :CTX], xis[:, :, CTX:].mean(0, keepdim=True).expand_as(xis[:, :, CTX:])], dim=2)
    return xis
model.xi_module.forward = xi_fwd

orig_occ = model.poisson_mode_occupation
def occ(h):
    E, phi = orig_occ(h)
    if 'wells_swap' in ON:
        phi = phi.roll(1, dims=0)
    elif 'wells_off' in ON:
        phi = torch.zeros_like(phi)
    return E, phi
model.poisson_mode_occupation = occ

orig_pf, orig_pp = model._pair_force_analytic, model._pair_potential
def pf(h_in, *a, **k):
    return torch.zeros_like(h_in) if 'vphi_off' in ON else orig_pf(h_in, *a, **k)
def pp(h_in, *a, **k):
    return torch.zeros((), device=h_in.device) if 'vphi_off' in ON else orig_pp(h_in, *a, **k)
model._pair_force_analytic, model._pair_potential = pf, pp

VARIANTS = {'base': (), 'xi_swap': ('xi_swap',), 'xi_posmean': ('xi_posmean',), 'vphi_off': ('vphi_off',),
            'wells_swap': ('wells_swap',), 'wells_off': ('wells_off',), 'content_swap': ('xi_swap', 'wells_swap'),
            'xi_ctx_swap': ('xi_ctx_swap',), 'xi_ctx_posmean': ('xi_ctx_posmean',)}
if ONLY:
    VARIANTS = {v: VARIANTS[v] for v in ['base'] + [x for x in ONLY if x != 'base']}
nll = {}
for v, sw in VARIANTS.items():
    ON.clear(); ON.update(sw); rows = []
    for xb, yb in batches:
        with torch.enable_grad():
            lo = model(torch.from_numpy(xb))[0].detach()
        ce = torch.nn.functional.cross_entropy(lo.reshape(-1, lo.shape[-1]), torch.from_numpy(yb).reshape(-1), reduction='none')
        rows.append(ce.reshape(xb.shape).mean(1).numpy())
    nll[v] = np.concatenate(rows)
    print(f'   [{v} done]', flush=True)
ON.clear()

print(f'MECHANISM ABLATIONS: {KIND}  K = {_k}  ({FOLDER.split("xilive_")[1].split("_L2probe")[0]}, '
      f'{CKPT or "best checkpoint"}; {NB_} x 4 x 512 tokens, batch seed 20261009)')
print(f'   base perplexity {np.exp(nll["base"].mean()):.2f}   xi channel alphas {[round(a, 3) for a in alphas]}')
for v in list(VARIANTS)[1:]:
    d = nll[v] - nll['base']
    print(f'   {v:13s} perplexity {np.exp(nll[v].mean()):9.2f}   change {np.expm1(d.mean()) * 100:+8.2f}%  '
          f'(per-sequence spread {d.std(ddof=1) * 100:.2f}%, n = {len(d)})')
