#!/usr/bin/env python3
"""
D1, what sets zeta(h) (tech report §7.1; checklist D1; protocol GH2; 2026-10-22): the learned gamma(h) arm's
trained damping head read token by token, on fixed weights.

    python3 d1_zeta_readout.py OUT_DIR FOLDER [batches=4] [threads=16]

Per token and layer, from one eval-mode forward pass on Cell 6b-7's first batches (seed 20260920, 4 x 512
validation tokens each):
  zeta      the ratio the head produced (captured from the head itself during the forward pass)
  eps       the local anharmonic fraction of M1 (protocol SS5.13; stage0_models.m1, copied below):
            ||f(h+dh) - f(h) + H dh|| / ||f(h+dh) - f(h)||, f = -grad V_theta, dh the layer's actual step
  omega     the median and the maximum frequency sqrt(kappa_i / m) of the token's 16 retained stiff modes
            (captured from the exact damped flow's inputs)
  unigram   the token's unigram surprisal, -log p(token), from the run's frequency table
  position  the token's position in its 512-token sequence
  nll       the model's next-token loss at that position
GH2 (called 55%): "the correlation between zeta(h) and eps is below 0.2 in magnitude". Fixed here, before
the run: the correlation is Spearman's rank correlation over all tokens of a layer, and GH2 hits if
|rho| < 0.2 at both layers. Pearson on log zeta is reported alongside, not scored.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(sys.argv[1]); FOLDER = sys.argv[2]
NB = int(next((a[8:] for a in sys.argv[3:] if a.startswith('batches=')), 4))
THREADS = int(next((a[8:] for a in sys.argv[3:] if a.startswith('threads=')), 16))
sys.argv = [sys.argv[0], str(OUT / 'harness')]
sys.path.insert(0, str(Path(__file__).parent))
import verify_pm_switch as P
torch.set_num_threads(THREADS)

repl = P.CONFIGS['F3.1'][1] + (("LOWRANK_DAMPED_FLOW  = False", "LOWRANK_DAMPED_FLOW  = True"),
                               ("LEARNED_DAMPING_RATIO = False", "LEARNED_DAMPING_RATIO = True"))
model, g, _ = P.build(FOLDER, repl)
model.to('cpu'); model.eval(); model.cfg.use_layer_checkpoint = False
assert model.damping_head is not None
L = model.cfg.L
uni = np.load(Path.home() / 'Downloads' / FOLDER / 'results' / 'logfreq_surprisal_openwebtext.npy')

rng = np.random.default_rng(20260920)
batches = [g['get_batch'](g['val_ids'], 4, 512, rng) for _ in range(12)][:NB]

# ---- captures: the head's output and the exact flow's stiffnesses, in call order ----
cap_s, cap_k = [], []
hh = model.damping_head.register_forward_hook(lambda m, i, o: cap_s.append(o.detach().float()))
import model_parf_multixi as MPX
_f0 = MPX.lowrank_damped_substep
def _wrap(h, v, U, kappa, f_lr, m, gamma, dt):
    cap_k.append((kappa.detach().float(), (m.detach().float() if torch.is_tensor(m) else torch.tensor(float(m)))))
    return _f0(h, v, U, kappa, f_lr, m, gamma, dt)
MPX.lowrank_damped_substep = _wrap


def m1_eps(traj):
    """stage0_models.m1, eps only."""
    out = []
    vt = model.V_theta
    lnv = getattr(model, 'ln_before_v', None)
    for ell in range(len(traj) - 1):
        h, dh = traj[ell], traj[ell + 1] - traj[ell]
        with torch.no_grad():
            xis = model.xi_module(h)
        if hasattr(vt, 'set_active_layer'):
            vt.set_active_layer(ell)
        V = lambda z: vt(xis, lnv(z) if lnv is not None else z).sum()
        z = h.clone().requires_grad_(True)
        g1, = torch.autograd.grad(V(z), z, create_graph=True)
        hdh, = torch.autograd.grad((g1 * dh).sum(), z)
        z2 = (h + dh).clone().requires_grad_(True)
        g2, = torch.autograd.grad(V(z2), z2)
        df = -(g2 - g1.detach())
        out.append(((df + hdh).norm(dim=-1) / df.norm(dim=-1).clamp(min=1e-12)).detach())
    return out


rows = {ell: {k: [] for k in ('zeta', 'eps', 'om_med', 'om_max', 'uni', 'pos', 'nll')} for ell in range(L)}
check = []
for bi, (xb, yb) in enumerate(batches):
    x, y = torch.from_numpy(xb), torch.from_numpy(yb)
    cap_s.clear(); cap_k.clear()
    with torch.enable_grad():
        logits = model(x)[0].detach()
    assert len(cap_s) == L and len(cap_k) == 2 * L, (len(cap_s), len(cap_k))
    zeta = [(model._zeta0 * torch.exp(torch.clamp(s, max=model._zeta_head_max))).squeeze(-1) for s in cap_s]
    nll = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction='none').view_as(y)
    with torch.enable_grad():
        _, traj = model._stack_forward(model._embed(x), x, return_trajectory=True)
    traj = [t.detach().float() for t in traj]
    eps = m1_eps(traj)
    for ell in range(L):
        # the head's input is the layer input: check it against the trajectory
        z_traj = (model._zeta0 * torch.exp(torch.clamp(model.damping_head(traj[ell]).float(), max=model._zeta_head_max))).squeeze(-1).detach()
        check.append(float((z_traj - zeta[ell]).abs().max() / zeta[ell].abs().max()))
        kappa, m = cap_k[2 * ell]
        om = torch.sqrt(torch.clamp(kappa / m, min=0.0))                    # (B, T, q)
        r = rows[ell]
        r['zeta'].append(zeta[ell].flatten().numpy()); r['eps'].append(eps[ell].flatten().numpy())
        r['om_med'].append(om.median(dim=-1).values.flatten().numpy()); r['om_max'].append(om.max(dim=-1).values.flatten().numpy())
        r['uni'].append(uni[xb].flatten()); r['pos'].append(np.tile(np.arange(xb.shape[1]), xb.shape[0]))
        r['nll'].append(nll.flatten().numpy())
    print(f'batch {bi + 1}/{NB} done', flush=True)
hh.remove(); MPX.lowrank_damped_substep = _f0


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


lines = [f'D1, what sets zeta(h): learned gamma(h) arm  ...{FOLDER[-60:]}',
         f'   {NB} x 4 x 512 tokens (Cell 6b-7\'s first batches); eval mode; fixed weights',
         f'   check: zeta from the head on the trajectory\'s layer inputs vs the captured zeta, max rel |d| {max(check):.1e}']
say = lambda s='': (print(s, flush=True), lines.append(s))
for l in lines: print(l)
res = {}
names = {'eps': 'anharmonic fraction eps (M1)', 'om_med': 'median mode frequency omega', 'om_max': 'largest mode frequency',
         'uni': 'unigram surprisal -log p(token)', 'pos': 'position in the sequence', 'nll': 'next-token loss'}
for ell in range(L):
    r = {k: np.concatenate(v) for k, v in rows[ell].items()}
    z = r['zeta']; lz = np.log(np.clip(z, 1e-12, None))
    q = np.quantile(z, [0.1, 0.5, 0.9])
    say(f'\nLAYER {ell}: {len(z)} tokens; zeta p10 / median / p90 = {q[0]:.3f} / {q[1]:.3f} / {q[2]:.3f}; above 1: {100 * (z > 1).mean():.2f}%;'
        f' eps median {np.median(r["eps"]):.3f}; omega*dt median {4 * np.median(r["om_med"]):.2f}')
    say(f'   {"against":<34} Spearman rho   Pearson(log zeta)')
    res[ell] = {'n': int(len(z)), 'zeta_q': q.tolist(), 'corr': {}}
    for k, nm in names.items():
        rho = spearman(z, r[k]); pr = float(np.corrcoef(lz, r[k])[0, 1])
        res[ell]['corr'][k] = {'spearman': rho, 'pearson_logzeta': pr}
        say(f'   {nm:<34} {rho:+.3f}         {pr:+.3f}')
    say('   median zeta by decile of:')
    for k in ('eps', 'om_med', 'uni', 'pos', 'nll'):
        edges = np.quantile(r[k], np.linspace(0, 1, 11)); idx = np.clip(np.searchsorted(edges, r[k], side='right') - 1, 0, 9)
        med = [float(np.median(z[idx == d])) if (idx == d).any() else float('nan') for d in range(10)]
        res[ell].setdefault('deciles', {})[k] = {'edges': edges.tolist(), 'median_zeta': med}
        say(f'     {k:<7} ' + ' '.join(f'{m:.3f}' for m in med))
gh2 = all(abs(res[ell]['corr']['eps']['spearman']) < 0.2 for ell in range(L))
say(f'\nGH2 (|Spearman(zeta, eps)| < 0.2 at both layers): {"HIT" if gh2 else "MISS"}  '
    f'({", ".join(f"layer {e}: {res[e]["corr"]["eps"]["spearman"]:+.3f}" for e in range(L))})')
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'd1_zeta_readout_output.txt').write_text('\n'.join(lines) + '\n')
(OUT / 'd1_zeta_readout.json').write_text(json.dumps(res, indent=1))
print(f'wrote {OUT}/d1_zeta_readout_output.txt')
