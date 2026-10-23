#!/usr/bin/env python3
"""Can a token's layer step be read off the landscape? (2026-10-22, the first-order "descent lines" question.)

    python3 descent_readout.py OUT_DIR {pmx|fob-pmx} [batches=2] [threads=16] [ckpt=<file>] [folder=<run folder>]

  ckpt=: a checkpoint in the run's checkpoints/ other than <prefix>_best.pt (a <prefix>_step<N>_snap.pt snapshot);
  folder=: another run folder of the same kind (the 2B runs, the dose-response arms).

For each trained layer, with the context frozen at the layer's input (xi, V_theta's quadratic and the wells' quadratic,
as FLOW-R's per-layer flow freezes them), compares the model's own layer step (N = 2, the trained map) with the
continuous flow on that frozen landscape (N = 16: eight substeps per layer), per token:

  direction   cosine between the model's displacement and the flow's displacement, per layer
  distance    |flow endpoint - model endpoint| / |model displacement|
  Lyapunov    the change of the potential along the path, dV = -sum f . dh, from the forces the step itself evaluates
              (one midpoint sample for the trained step, eight for the flow): the share of tokens whose V falls
  basin       whether the model's endpoint and the flow's endpoint overlap the same Poisson-mode well most (argmax E_v)
  readout     top-1 agreement and KL(model || flow) of the next-token distributions, and both perplexities

Under first order (fob-pmx) the step is a descent and V should fall for nearly every token; under second order (pmx)
inertia can carry a token up the potential and the direction need not follow the gradient.
"""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(sys.argv[1]); KIND = sys.argv[2]
NB_ = int(next((a[8:] for a in sys.argv[3:] if a.startswith('batches=')), 2))
THREADS = int(next((a[8:] for a in sys.argv[3:] if a.startswith('threads=')), 16))
CKPT = next((a[5:] for a in sys.argv[3:] if a.startswith('ckpt=')), None)
FOLDER_ARG = next((a[7:] for a in sys.argv[3:] if a.startswith('folder=')), None)
assert KIND in ('pmx', 'fob-pmx')
sys.argv = [sys.argv[0], str(OUT / 'harness')]
sys.path.insert(0, str(Path(__file__).parent))
import verify_pm_switch as P
from refinement_ln_linearisation_split import split_stack
torch.set_num_threads(THREADS)

PFX = ('semsimula_fock_cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_')
SFX = '_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn'
FOLDER = FOLDER_ARG or (PFX + ('pm64_pmx16_sr2_fob' if KIND == 'fob-pmx' else 'pm64_pmx16_sr2') + SFX)
repl = P.CONFIGS['F3.1'][1] + P.PM_ON + (("LOWRANK_DAMPED_FLOW  = False", "LOWRANK_DAMPED_FLOW  = True"),
                                         ("POISSON_WELLS_EXACT  = False", "POISSON_WELLS_EXACT  = True"))
if KIND == 'fob-pmx':
    repl = repl + (("FIRST_ORDER_OVERDAMPED = False", "FIRST_ORDER_OVERDAMPED = True"),)
model, g, _ = P.build(FOLDER, repl, ckpt=CKPT)
model.to('cpu'); model.eval(); model.cfg.use_layer_checkpoint = False
assert bool(model.cfg.first_order_overdamped) == (KIND == 'fob-pmx') and model.cfg.poisson_wells_exact

nb = json.load(open(P.G.NB))
src = [''.join(c['source']) for c in nb['cells'] if ''.join(c['source']).startswith('# == Cell 6b-7:')][0]
tree = ast.parse(src)
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))], type_ignores=[]), 'Cell6b7_defs', 'exec'), g)
policy_index = g['_fom_policy_index']
L, DT = model.cfg.L, model.cfg.dt; T = L * DT
FLOW = dict(ln_once=True, freeze=True, xi_freeze=True, phi_freeze=False, pmq_freeze=True)
TRAINED = dict(ln_once=False, freeze=False, xi_freeze=False, phi_freeze=False)
rng = np.random.default_rng(20261009)
batches = [g['get_batch'](g['val_ids'], 4, 512, rng) for _ in range(NB_)]

# ---- captures: per layer step (h_in, h_out); per force call (h, f) for the line integral
cap = {'steps': [], 'forces': []}
orig_step = model._fock_layer_step
def rec_step(h, h_prev, r, sal, m_b, gamma, dt, layer_idx=0):
    out = orig_step(h, h_prev, r, sal, m_b, gamma, dt, layer_idx=layer_idx)
    cap['steps'].append((int(layer_idx), h.detach().clone(), out[0].detach().clone()))
    return out
orig_forces = model._layer_forces
def rec_forces(h, xis, layer_idx, *a, **k):
    out = orig_forces(h, xis, layer_idx, *a, **k)
    f = (out[0] + out[1]) if isinstance(out, tuple) else out
    cap['forces'].append((int(layer_idx), h.detach().clone(), f.detach().clone()))
    return out
# _layer_forces already includes the Poisson-mode wells' force (model_parf_multixi: f_phi + poisson_mode_force), so it
# is the full force f = -grad(V_theta + V_phi + V_pm) at the point it is evaluated: one sample per (sub)step, at h_mid.
model._fock_layer_step = rec_step; model._layer_forces = rec_forces

def run(xb, arm, n):
    cap['steps'].clear(); cap['forces'].clear()
    with split_stack(model, n, T / n, policy_index, **arm), torch.enable_grad():
        logits = model(torch.from_numpy(xb))[0].detach()
    # per layer: first input, last output; and the (sub)steps with their one force sample each, in call order
    assert len(cap['steps']) == len(cap['forces']), (len(cap['steps']), len(cap['forces']))
    per, sub = {}, {}
    for (li, hi, ho), (lf, hm, ff) in zip(cap['steps'], cap['forces']):
        assert li == lf
        if li not in per: per[li] = [hi, ho]
        else: per[li][1] = ho
        sub.setdefault(li, []).append((hi, ho, ff))
    return logits, per, sub

def dV_along(subs):
    """-sum_k f(h_mid,k) . (h_out,k - h_in,k): the midpoint rule for the change of V along the (sub)steps of a layer."""
    tot = torch.zeros(subs[0][0].shape[:-1])
    for hi, ho, f in subs:
        tot = tot - (f * (ho - hi)).sum(-1)
    return tot

def pm_basin(h, h_ctx, layer_idx):
    """The well pulling hardest on the token at h: argmax_v |w_v| |r_v|, with w_v = 2 kappa_v^2 a_v phi_v E_v(h) and
    r_v = h - mu_v, the integrator's own ranking (poisson_mode_quadratic). The occupations phi are taken from the layer
    input h_ctx and held, as the integrator holds them over the step. Replaces the overlap argmax, which proved
    degenerate (one broad mode is the most-overlapped for every PMX token)."""
    with torch.no_grad():
        _, phi = model.poisson_mode_occupation(h_ctx)
        E, _ = model.poisson_mode_occupation(h)
        k2 = model.pm_log_kappa2.float().exp(); a = model.pm_effective_depth(layer_idx)
        mu = model.pm_mu.float(); hf = h.float()
        r = ((hf * hf).sum(-1, keepdim=True) + (mu * mu).sum(-1) - 2.0 * hf @ mu.t()).clamp_min(0.0).sqrt()
        w = phi * E * (2.0 * k2 * a)
        return w.abs() * r                                                     # (B, T, K): the pull of each well

res = {li: dict(cos=[], rel=[], dV_tr=[], dV_fl=[], basin=[]) for li in range(L)}
agree, kl, nll_m, nll_f = [], [], [], []
for bi, (xb, yb) in enumerate(batches):
    lo_m, per_m, fo_m = run(xb, TRAINED, 2)
    lo_f, per_f, fo_f = run(xb, FLOW, 16)
    for li in range(L):
        hi, ho = per_m[li]; _, hf = per_f[li]
        dm, df = ho - hi, hf - hi
        res[li]['cos'].append(torch.nn.functional.cosine_similarity(dm, df, dim=-1).flatten())
        res[li]['rel'].append(((hf - ho).norm(dim=-1) / dm.norm(dim=-1).clamp_min(1e-9)).flatten())
        # Lyapunov: the trained step (one force sample at its midpoint) and the flow (eight substeps)
        res[li]['dV_tr'].append(dV_along(fo_m[li]).flatten())
        res[li]['dV_fl'].append(dV_along(fo_f[li]).flatten())
        pm_, pf_ = pm_basin(ho, hi, li), pm_basin(hf, hi, li)
        bm, bf = pm_.argmax(-1), pf_.argmax(-1)
        res[li]['basin'].append((bm == bf).float().flatten())
        res[li].setdefault('nb', []).append(bm.flatten())
        # finer, on the motion itself: approach_v = log E_v(end) - log E_v(start) = -kappa_v^2 (|end - mu_v|^2 - |start - mu_v|^2),
        # how much closer the step brought the token to each well (occupation and depth, shared by both endpoints, drop out)
        def approach(h_end):
            k2 = model.pm_log_kappa2.float().exp(); mu = model.pm_mu.float()
            d2 = lambda x: ((x * x).sum(-1, keepdim=True) + (mu * mu).sum(-1) - 2.0 * x @ mu.t()).clamp_min(0.0)
            return -k2 * (d2(h_end.float()) - d2(hi.float()))
        pm_, pf_ = approach(ho), approach(hf)
        t3m, t3f = pm_.topk(3, dim=-1).indices, pf_.topk(3, dim=-1).indices
        inter = (t3m.unsqueeze(-1) == t3f.unsqueeze(-2)).any(-1).float().sum(-1)
        res[li].setdefault('top3', []).append((inter / 3.0).flatten())
        rk = lambda x: x.argsort(-1).argsort(-1).float()
        rm, rf = rk(pm_), rk(pf_); rm = rm - rm.mean(-1, keepdim=True); rf = rf - rf.mean(-1, keepdim=True)
        res[li].setdefault('rho', []).append(((rm * rf).sum(-1) / (rm.norm(dim=-1) * rf.norm(dim=-1)).clamp_min(1e-12)).flatten())
    pm_, pf_ = torch.log_softmax(lo_m, -1), torch.log_softmax(lo_f, -1)
    agree.append((lo_m.argmax(-1) == lo_f.argmax(-1)).float().flatten())
    kl.append((pm_.exp() * (pm_ - pf_)).sum(-1).flatten())
    y = torch.from_numpy(yb)
    nll_m.append(torch.nn.functional.cross_entropy(lo_m.reshape(-1, lo_m.shape[-1]), y.reshape(-1), reduction='none'))
    nll_f.append(torch.nn.functional.cross_entropy(lo_f.reshape(-1, lo_f.shape[-1]), y.reshape(-1), reduction='none'))
    print(f'batch {bi + 1}/{NB_} done', flush=True)
model._fock_layer_step = orig_step; model._layer_forces = orig_forces

q = lambda t, p: float(torch.cat(t).quantile(p))
lines = [f'DESCENT READOUT: {KIND}  ({CKPT or "best checkpoint"}; {NB_} x 4 x 512 tokens, batch seed 20261009)',
         '   the model\'s trained layer step (N = 2) against the continuous flow on the frozen landscape (N = 16, FLOW-R\'s per-layer flow)', '']
for li in range(L):
    r = res[li]; cos = torch.cat(r['cos']); rel = torch.cat(r['rel']); dtr = torch.cat(r['dV_tr']); dfl = torch.cat(r['dV_fl']); bas = torch.cat(r['basin'])
    lines += [f'LAYER {li}',
              f'   direction: cosine(model step, flow step)  median {q(r["cos"], .5):.3f}  p10 {q(r["cos"], .1):.3f}  p90 {q(r["cos"], .9):.3f}   share above 0.9: {100 * float((cos > 0.9).float().mean()):.1f}%',
              f'   distance:  |flow end - model end| / |model step|  median {q(r["rel"], .5):.3f}  p90 {q(r["rel"], .9):.3f}',
              f'   Lyapunov:  dV along the trained step (midpoint rule): V falls for {100 * float((dtr < 0).float().mean()):.1f}% of tokens (median dV {float(dtr.median()):+.3f})',
              f'              dV along the flow path (8 samples):      V falls for {100 * float((dfl < 0).float().mean()):.1f}% of tokens (median dV {float(dfl.median()):+.3f})',
              f'   basin:     same strongest-pulling well at the model\'s and the flow\'s endpoint: {100 * float(bas.mean()):.1f}%  '
              f'(the model\'s endpoints use {len(torch.cat(r["nb"]).unique())} distinct wells of {model.pm_mu.shape[0]})',
              f'   approach:  the three wells the step moved the token most toward, shared by model and flow: mean {float(torch.cat(r["top3"]).mean()):.3f} of 3/3; all three shared for {100 * float((torch.cat(r["top3"]) == 1).float().mean()):.1f}% of tokens',
              f'              rank correlation of the approach over all wells: median {q(r["rho"], .5):.3f}  p10 {q(r["rho"], .1):.3f}']
lines += ['', 'READOUT (after layer 1)',
          f'   top-1 agreement model vs flow: {100 * float(torch.cat(agree).mean()):.1f}%   mean KL(model || flow): {float(torch.cat(kl).mean()):.3f} nats',
          f'   perplexity: model {float(torch.exp(torch.cat(nll_m).mean())):.2f}   flow {float(torch.exp(torch.cat(nll_f).mean())):.2f}']
print('\n'.join(lines))
OUT.mkdir(parents=True, exist_ok=True)
_tag = ('_' + CKPT.split('_')[-2]) if CKPT and CKPT.endswith('_snap.pt') else ''
(OUT / f'descent_readout_{KIND}{_tag}_output.txt').write_text('\n'.join(lines) + '\n')
