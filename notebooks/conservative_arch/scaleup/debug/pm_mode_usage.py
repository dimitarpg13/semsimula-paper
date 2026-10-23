#!/usr/bin/env python3
"""How many Poisson-mode wells does a trained model use? (2026-10-23, D-e of the TL plan; PMX-K16 settled level with PMX.)

    python3 pm_mode_usage.py OUT_DIR {pmx|fob-pmx} [batches=2] [threads=16] [ckpt=<file>] [folder=<run folder>]

  ckpt=: a checkpoint in the run's checkpoints/ other than <prefix>_best.pt (a <prefix>_step<N>_snap.pt snapshot);
  folder=: another run folder of the same kind (the 2B runs, the dose-response arms, e.g. the pm16 folders).

At each layer's input h, with the occupations phi taken from that input (as the integrator's quadratic takes them),
mode v pulls token t with f_v = w_v (mu_v - h_t), w_v = 2 kappa_v^2 a_v phi_v(t) E_v(t), of size |w_v| |h_t - mu_v|.
Per layer:

  p_v      the mode's share of the wells' force over all tokens, sum_t |f_v(t)| / sum_t sum_u |f_u(t)|
  N_eff    exp(entropy of p): K if every mode carries the same share, 1 if one mode carries it all
  N_80     the fewest modes that carry 80% of the force
  per-token N_eff   median over tokens of exp(entropy) of the token's own shares (wells acting on one token at once)

and, for reading N_eff, the attractive share of the depths, the force share of the attractive modes, and the
half-lives of the modes that carry 80% of the force. Added after the first PMX pass (per-token N_eff 39.5-40.6, every
token feeling nearly the same mix): token specificity, the total-variation distance of a token's shares from the
average shares; the coefficient of variation of each overlap E_v across tokens; the exponent kappa_v^2 |h - mu_v|^2;
and the position-only reconstruction: each w_v(t) replaced by its mean over the other sequences at the same position
(the token's own h kept), the share of the wells' force and of the occupations it explains. The ablation that asks
whether the model USES the content is pm_content_swap.py.
"""
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
K = model.pm_depth.shape[1]; assert K == _k, (K, _k)

rng = np.random.default_rng(20261009)
batches = [g['get_batch'](g['val_ids'], 4, 512, rng) for _ in range(NB_)]
inputs = []
orig_step = model._fock_layer_step
def rec_step(h, h_prev, r, sal, m_b, gamma, dt, layer_idx=0):
    inputs.append((int(layer_idx), h.detach().clone()))
    return orig_step(h, h_prev, r, sal, m_b, gamma, dt, layer_idx=layer_idx)
model._fock_layer_step = rec_step
with torch.enable_grad():                    # the conservative forces need autograd even at eval
    for xb, _ in batches:
        model(torch.from_numpy(xb))
model._fock_layer_step = orig_step

def neff(p, axis=-1):
    p = p / p.sum(axis, keepdims=True).clip(1e-30)
    return np.exp(-(p * np.log(p.clip(1e-30))).sum(axis))

halflife = (np.log(0.5) / torch.nn.functional.logsigmoid(model.pm_logit_lambda.detach().float()).numpy())
print(f'POISSON-MODE USAGE: {KIND}  K = {K}  ({FOLDER.split("xilive_")[1].split("_L2probe")[0]}, '
      f'{CKPT or "best checkpoint"}; {NB_} x 4 x 512 tokens, batch seed 20261009)')
for li in range(model.cfg.L):
    mags, ovl, expo = [], [], []
    for l_, h in inputs:
        if l_ != li:
            continue
        with torch.no_grad():
            E, phi = model.poisson_mode_occupation(h)
            k2 = model.pm_log_kappa2.float().exp()
            a = model.pm_effective_depth(li)
            w = phi * E * (2.0 * k2 * a)
            hf = h.float(); mu = model.pm_mu.float()
            d2 = ((hf * hf).sum(-1, keepdim=True) + (mu * mu).sum(-1) - 2.0 * hf @ mu.t()).clamp_min(0.0)
            mags.append((w.abs() * d2.sqrt()).reshape(-1, K).numpy())
            ovl.append(E.reshape(-1, K).numpy()); expo.append((k2 * d2).reshape(-1, K).numpy())
    M = np.concatenate(mags)                                   # (tokens, K)
    Eo, X = np.concatenate(ovl), np.concatenate(expo)
    p = M.sum(0) / M.sum()
    order = np.argsort(-p); cum = np.cumsum(p[order])
    n80 = int(np.searchsorted(cum, 0.8) + 1)
    tok = neff(M)
    a = model.pm_effective_depth(li).detach().numpy()
    att = a > 0
    print(f'LAYER {li}')
    print(f'   N_eff {neff(p):5.1f} of {K}   N_80 {n80:2d}   per-token N_eff median {np.median(tok):4.1f} '
          f'(p10 {np.percentile(tok, 10):4.1f}, p90 {np.percentile(tok, 90):4.1f})')
    pt = M / M.sum(1, keepdims=True).clip(1e-30)
    tv = 0.5 * np.abs(pt - p).sum(1)
    cv = Eo.std(0) / Eo.mean(0).clip(1e-30)
    print(f'   token specificity: total-variation distance of a token\'s mix from the average mix, median {np.median(tv):.3f} '
          f'(p90 {np.percentile(tv, 90):.3f}; 0 = every token feels the same mix, 1 = disjoint)')
    print(f'   overlap E_v across tokens: coefficient of variation, median over modes {np.median(cv):.3f}; '
          f'exponent kappa^2 |h - mu|^2 median {np.median(X):.2f} (p10 {np.percentile(X, 10):.2f}, p90 {np.percentile(X, 90):.2f})')
    print(f'   top-5 modes carry {cum[min(4, K - 1)]:.0%}; top-16 carry {cum[min(15, K - 1)]:.0%}')
    print(f'   attractive depths {att.mean():.0%} of modes, {p[att].sum():.0%} of the force')
    # position only: rebuild the wells' force with each w_v(t) replaced by its mean over the OTHER sequences at the
    # same position (leave one sequence out); the token's own h is kept, so only the content carried by w is removed
    with torch.no_grad():
        H = torch.cat([h for l_, h in inputs if l_ == li]).float()          # (sequences, T, d)
        E, phi = model.poisson_mode_occupation(H)
        k2 = model.pm_log_kappa2.float().exp(); mu = model.pm_mu.float()
        W = phi * E * (2.0 * k2 * model.pm_effective_depth(li))
        F = -(W.sum(-1, keepdim=True) * H - W @ mu)
        Wl = (W.sum(0, keepdim=True) - W) / (W.shape[0] - 1)
        Fl = -(Wl.sum(-1, keepdim=True) * H - Wl @ mu)
        r2 = 1 - float(((F - Fl) ** 2).sum() / ((F - F.mean((0, 1))) ** 2).sum())
        pl = (phi.sum(0, keepdim=True) - phi) / (phi.shape[0] - 1)
        r2p = 1 - float(((phi - pl) ** 2).sum() / ((phi - phi.mean((0, 1))) ** 2).sum())
        cs = torch.nn.functional.cosine_similarity(F, Fl, dim=-1)
    print(f'   position only (leave one sequence out): the wells\' force explained R^2 {r2:.4f}, median cosine {float(cs.median()):.4f}; '
          f'the occupations explained R^2 {r2p:.4f}')
    hl = halflife[order[:n80]]
    print(f'   half-lives of the N_80 modes: median {np.median(hl):.1f} tokens (range {hl.min():.1f}-{hl.max():.1f}); '
          f'all modes median {np.median(halflife):.1f}')
