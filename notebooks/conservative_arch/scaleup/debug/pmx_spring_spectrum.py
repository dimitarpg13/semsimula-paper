"""The frozen spring PMX integrates, per token and layer, on the trained weights (2026-10-11).

Records, at every exact half-step (lowrank_iso_damped_substep) of one eval
forward, the per-token spring K = alpha I + U diag(kappa) U^T: the 32 mode
stiffnesses kappa_i + alpha, the complement's alpha, the mass m and the
substep length, tagged with the layer from the poisson_mode_quadratic call
that precedes them. For the figures of Exact_Integration_of_Poisson_Mode_Wells_PMX.md.

  omega_i^2 = (kappa_i + alpha) / m   (mode i),   omega_c^2 = alpha / m   (complement)

Usage: python3 pmx_spring_spectrum.py OUT_DIR FOLDER   (FOLDER: a PMX run, or PM1's with PMX switched on)
Writes OUT_DIR/pmx_spring_spectrum.npz and a short summary.
"""
import sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(sys.argv[1]); FOLDER = sys.argv[2]
sys.argv = [sys.argv[0], str(OUT / 'harness')]
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / 'parf'))
import verify_pm_switch as P
import model_parf_multixi as MPX

repl = (P.CONFIGS['F3.1'][1] + P.PM_ON
        + (("LOWRANK_DAMPED_FLOW  = False", "LOWRANK_DAMPED_FLOW  = True"),
           ("POISSON_WELLS_EXACT  = False", "POISSON_WELLS_EXACT  = True")))
model, g, _ = P.build(FOLDER, repl)
model.to('cpu'); model.eval()
assert getattr(model.cfg, 'poisson_wells_exact', False)

x, _ = P.tokens(n_seq=1, T=256)
rec, layer = [], [None]
orig_q, orig_sub = model.poisson_mode_quadratic, MPX.lowrank_iso_damped_substep


def spy_q(h, layer_idx, top=None):
    layer[0] = layer_idx
    return orig_q(h, layer_idx, top=top)


def spy_sub(h, v, U, kappa, alpha, f, m, gamma, dt, n_terms=40):
    rec.append(dict(layer=layer[0], kappa=kappa.detach().double().numpy()[0],
                    alpha=alpha.detach().double().numpy()[0, :, 0],
                    m=m.detach().double().numpy()[0, :, 0], dt=float(dt),
                    alive=(U.detach().norm(dim=-2) > 0.5).numpy()[0]))
    return orig_sub(h, v, U, kappa, alpha, f, m, gamma, dt, n_terms)


model.poisson_mode_quadratic = spy_q
MPX.lowrank_iso_damped_substep = spy_sub
try:
    with torch.enable_grad():
        model(x)
finally:
    MPX.lowrank_iso_damped_substep = orig_sub
    del model.__dict__['poisson_mode_quadratic']

# the first exact half-step of each layer (the spring is the same for both halves)
first = {}
for r in rec:
    first.setdefault(r['layer'], r)
out, lines = {}, [f'PMX spring spectrum  ...{FOLDER[-80:]}', f'   1 x 256 tokens; layer dt = {model.cfg.dt:g}']
DT = float(model.cfg.dt)
for li, r in sorted(first.items()):
    w2 = (r['kappa'] + r['alpha'][:, None]) / r['m'][:, None]             # (T, 32)
    wc2 = r['alpha'] / r['m']                                               # (T,)
    w2 = np.where(r['alive'], w2, np.nan)
    sgn = lambda a: np.sign(a) * np.sqrt(np.abs(a)) * DT                    # signed omega * dt
    out[f'l{li}_modes'], out[f'l{li}_comp'] = sgn(w2), sgn(wc2)
    neg = (w2 < 0).sum(1)
    lines.append(f'   layer {li}: modes alive {int(r["alive"].sum(1).mean())}/32 per token; '
                 f'negative modes per token median {int(np.median(neg))} (min {neg.min()}, max {neg.max()}); '
                 f'complement alpha < 0 for {100 * (wc2 < 0).mean():.0f}% of tokens; '
                 f'|omega dt| p50 modes {np.nanmedian(np.abs(sgn(w2))):.2f}, complement {np.median(np.abs(sgn(wc2))):.2f}')
np.savez(OUT / 'pmx_spring_spectrum.npz', **out)
(OUT / 'pmx_spring_spectrum_output.txt').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
