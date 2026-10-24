#!/usr/bin/env python3
"""Stage 0' (make matching real), step 0'a: why the trained Poisson modes do not discriminate, and which
initialisation would make them (2026-10-23; Implementing_Semantic_Templates_A_Path.md §7, protocol §5.13).

    python3 narrow_wells_design.py OUT_DIR [threads=16]

Part 1, geometry. At each layer's input, for the model at initialisation (built from the notebook's cells, PMX-K16's
Cell 0, no checkpoint) and for the trained PMX-K16: the token states' norm |h|, the centres' norm |mu|, the typical
token-token distance, and each mode's distance to the tokens (nearest, median), all relative to the width 1/kappa.

Part 2, candidates, scored on the states at initialisation (what a run starts from) and on the trained PMX-K16
states (a proxy for where the cloud goes). A candidate is a choice of centres and a width:
  centres: 'randn'  the current init, N(0, I) in R^d (|mu| ~ sqrt(d))
           'tok0'   K token states sampled from layer 0's input
           'tok1'   K token states sampled from layer 1's input
  width:   kappa^2 = c / D2, with D2 the median squared token-token distance at the layer, c in {0.5, 2, 8, 32}
           (the current init is kappa^2 = 1/d)
For each (layer, candidate): the median over modes of the overlap's coefficient of variation across tokens
(selectivity; the trained models read 0.01-0.04), the median share of tokens a mode reaches (E above half the mode's
maximum), and the share of dead modes (no token with E above 0.01).
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

OUT = Path(sys.argv[1])
THREADS = int(next((a[8:] for a in sys.argv[2:] if a.startswith('threads=')), 16))
sys.argv = [sys.argv[0], str(OUT / 'harness')]
sys.path.insert(0, str(Path(__file__).parent))
import verify_pm_switch as P
torch.set_num_threads(THREADS)

PFX = ('semsimula_fock_cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_')
SFX = '_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn'
K = 16
REPL = P.CONFIGS['F3.1'][1] + (("POISSON_MODES        = 0", f"POISSON_MODES        = {K}"),
                               ("LOWRANK_DAMPED_FLOW  = False", "LOWRANK_DAMPED_FLOW  = True"),
                               ("POISSON_WELLS_EXACT  = False", "POISSON_WELLS_EXACT  = True"))


class _Stop(Exception):
    pass


def build_init():
    """The notebook's model at initialisation: P.build up to the checkpoint load (as verify_dose_modes.py)."""
    load = torch.load
    torch.load = lambda *a, **k: (_ for _ in ()).throw(_Stop())
    try:
        P.build(PFX + f'pm{K}_pmx16_sr2' + SFX, REPL)
    except _Stop:
        tb = sys.exc_info()[2]; m = g = None
        while tb:
            fl = tb.tb_frame.f_locals
            if tb.tb_frame.f_code.co_name == 'build' and 'model' in fl:
                m, g = fl['model'], fl['g']
            tb = tb.tb_next
        return m, g
    finally:
        torch.load = load


def layer_inputs(model, xs):
    caps = {}
    orig = model._fock_layer_step
    def rec(h, h_prev, r, sal, m_b, gamma, dt, layer_idx=0):
        caps.setdefault(int(layer_idx), []).append(h.detach().reshape(-1, h.shape[-1]).float().clone())
        return orig(h, h_prev, r, sal, m_b, gamma, dt, layer_idx=layer_idx)
    model._fock_layer_step = rec
    model.eval(); model.cfg.use_layer_checkpoint = False
    with torch.enable_grad():
        for x in xs:
            model(torch.from_numpy(x))
    model._fock_layer_step = orig
    return {l: torch.cat(v) for l, v in caps.items()}


def sqdist(a, b):
    return ((a * a).sum(-1, keepdim=True) + (b * b).sum(-1) - 2.0 * a @ b.t()).clamp_min(0.0)


def q(t, p):
    return float(torch.quantile(t.flatten()[:1_000_000], p))


t0 = time.time()
m_init, g = build_init()
rng = np.random.default_rng(20261009)
xs = [g['get_batch'](g['val_ids'], 4, 512, rng)[0] for _ in range(2)]
H_init = layer_inputs(m_init, xs)
mu_init = m_init.pm_mu.detach().float(); k2_init = float(m_init.pm_log_kappa2.detach().float().exp().median())
m_tr, _, _ = P.build(PFX + f'pm{K}_pmx16_sr2' + SFX, REPL)
m_tr.to('cpu')
H_tr = layer_inputs(m_tr, xs)
mu_tr = m_tr.pm_mu.detach().float(); k2_tr = m_tr.pm_log_kappa2.detach().float().exp()
d = mu_init.shape[1]
gen = torch.Generator().manual_seed(0)

print(f'NARROW-WELLS DESIGN, step 0\'a  (K = {K}; {len(xs)} x 4 x 512 validation tokens, batch seed 20261009; '
      f'built in {time.time() - t0:.0f}s)')
print('\nPART 1: GEOMETRY  (distances are Euclidean, in the units of h)')
for name, H, mu, k2 in (('init', H_init, mu_init, torch.full((K,), k2_init)), ('trained PMX-K16', H_tr, mu_tr, k2_tr)):
    print(f'  {name}:  |mu| median {float(mu.norm(dim=1).median()):.1f}   1/kappa median {float(k2.rsqrt().median()):.1f}'
          f'   kappa^2 x d median {float((k2 * d).median()):.2f}')
    for l in sorted(H):
        h = H[l]
        sub = h[torch.randperm(len(h), generator=gen)[:2000]]
        dtt = sqdist(sub, sub).sqrt()
        dtt = dtt[~torch.eye(len(sub), dtype=bool)]
        dm = sqdist(h, mu).sqrt()                                     # (tokens, K)
        print(f'    layer {l}: |h| p10/p50/p90 {q(h.norm(dim=1), .1):6.1f} {q(h.norm(dim=1), .5):6.1f} {q(h.norm(dim=1), .9):6.1f}'
              f'   token-token distance p50 {q(dtt, .5):6.1f}   mode-token distance: nearest token p50 '
              f'{float(dm.min(0).values.median()):6.1f}, median token {float(dm.median(0).values.median()):6.1f}, '
              f'spread (p90 - p10) over tokens {float((dm.quantile(.9, 0) - dm.quantile(.1, 0)).median()):5.2f}')


def score(H, mu, k2):
    E = torch.exp(-k2 * sqdist(H, mu))                                # (tokens, K)
    cv = (E.std(0) / E.mean(0).clamp_min(1e-30))
    reach = (E > 0.5 * E.max(0, keepdim=True).values).float().mean(0)
    dead = (E.max(0).values < 0.01).float().mean()
    return float(cv.median()), float(reach.median()), float(dead)


print('\nPART 2: CANDIDATES  (selectivity = median over modes of the overlap\'s coefficient of variation across tokens;'
      '\n  reach = median share of tokens with E above half the mode\'s maximum; dead = share of modes with max E < 0.01)')
print(f'  the current init (randn centres, kappa^2 = 1/d) on the init states, for reference:')
for l in sorted(H_init):
    cv, reach, dead = score(H_init[l], mu_init, torch.tensor(1.0 / d))
    print(f'    layer {l}: selectivity {cv:.3f}  reach {reach:.0%}  dead {dead:.0%}')
for cname in ('randn', 'tok0', 'tok1'):
    print(f'  centres = {cname}')
    if cname == 'randn':
        mu_c = torch.randn(K, d, generator=gen)
    else:
        src = H_init[0 if cname == 'tok0' else 1]
        mu_c = src[torch.randperm(len(src), generator=gen)[:K]].clone()
    for c in (0.5, 2.0, 8.0, 32.0):
        row = []
        for l in sorted(H_init):
            src = H_init[l][torch.randperm(len(H_init[l]), generator=gen)[:2000]]
            D2 = float(sqdist(src, src)[~torch.eye(len(src), dtype=bool)].median())
            k2c = torch.tensor(c / D2)
            for tag, H in (('init', H_init[l]), ('trained', H_tr[l])):
                cv, reach, dead = score(H, mu_c, k2c)
                row.append(f'L{l} {tag:7s} sel {cv:6.3f} reach {reach:4.0%} dead {dead:4.0%}')
        print(f'    c = {c:4g}:  ' + '  |  '.join(row))


def score2(H, mu, k2):
    """reach: median share of tokens a mode reaches (E > 0.5, absolute); coverage: share of tokens that reach at least
    one mode (E > 0.5); per-token modes: median number of modes a token reaches."""
    E = torch.exp(-k2 * sqdist(H, mu))
    cv = (E.std(0) / E.mean(0).clamp_min(1e-30))
    hit = E > 0.5
    return float(cv.median()), float(hit.float().mean(0).median()), float(hit.any(1).float().mean()), float(hit.sum(1).float().median())


print('\nPART 3: REACHABLE SELECTIVITY  (centres sampled per layer from the TRAINED states, scored on the trained states;'
      '\n  reach and coverage with the absolute criterion E > 0.5; kappa^2 = c / D2 as in part 2)')
for l in sorted(H_tr):
    src = H_tr[l][torch.randperm(len(H_tr[l]), generator=gen)[:2000]]
    D2 = float(sqdist(src, src)[~torch.eye(len(src), dtype=bool)].median())
    for Kc in (16, 64):
        mu_c = H_tr[l][torch.randperm(len(H_tr[l]), generator=gen)[:Kc]].clone()
        row = []
        for c in (0.25, 0.5, 1.0, 2.0, 4.0):
            cv, reach, cover, per = score2(H_tr[l], mu_c, torch.tensor(c / D2))
            row.append(f'c={c:<4g} sel {cv:5.2f} reach {reach:4.0%} cover {cover:4.0%} modes/token {per:4.0f}')
        print(f'  layer {l}, K = {Kc:2d} (D2 = {D2:.1f}):  ' + '  |  '.join(row))


print('\nPART 4: LOW-RANK MATCHING  (distance measured in an r-dimensional subspace: the top-r principal directions of the'
      '\n  layer\'s trained states, shared by the modes; centres sampled from the trained states; kappa^2 = c / D2_r with D2_r'
      '\n  the median squared token-token distance in the subspace; the relative spread of token-mode distances is'
      '\n  (p90 - p10) / median, over tokens, median over modes)')
for l in sorted(H_tr):
    H = H_tr[l]
    Hc = H - H.mean(0, keepdim=True)
    _, _, V = torch.linalg.svd(Hc[torch.randperm(len(Hc), generator=gen)[:4000]], full_matrices=False)
    mu_idx = torch.randperm(len(H), generator=gen)[:64]
    for r in (d, 64, 16, 8, 4):
        Pr = V[:r].t()                                                 # (d, r)
        Z = Hc @ Pr
        Zm = Z[mu_idx]
        src = Z[torch.randperm(len(Z), generator=gen)[:2000]]
        D2r = float(sqdist(src, src)[~torch.eye(len(src), dtype=bool)].median())
        dist = sqdist(Z, Zm).sqrt()
        spread = float(((dist.quantile(.9, 0) - dist.quantile(.1, 0)) / dist.median(0).values).median())
        row = []
        for c in (0.5, 1.0, 2.0, 4.0):
            cv, reach, cover, per = score2(Z, Zm, torch.tensor(c / D2r))
            row.append(f'c={c:<3g} sel {cv:5.2f} reach {reach:4.0%} cover {cover:4.0%} modes/tok {per:3.0f}')
        print(f'  layer {l}, r = {r:3d}: spread {spread:4.2f}  ' + '  |  '.join(row))
