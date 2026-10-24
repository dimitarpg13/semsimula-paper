#!/usr/bin/env python3
"""How much content do the occupations keep, by half-life range? (2026-10-23; Stage 0', before a 2B template arm.)

    python3 pm_halflife_range.py OUT_DIR [threads=16]

An occupation phi_v(t) = sum_{s<t} lambda_v^(t-1-s) E_v(s) averages a mode's overlaps over its half-life window, and
the average washes token-level variation out: in the short CPU run (pm_match_short_train.py) the rank-8 matcher's
overlaps were selective while layer 1's occupations stayed 98% position. Here, for the rank-8 matcher (K = 16) and
several half-life ranges (K half-lives log-spaced from the minimum to the maximum, as the model initialises them),
the share of the occupations' variance explained by position alone (leave one sequence out; 8 x 512 validation
tokens), per layer, on two clouds:
  init     the model's own layer inputs at initialisation, matcher initialised on them (the model's routine)
  trained  the trained PMX-K16's layer inputs, a rank-8 matcher initialised on them by the same routine
           (a proxy for the trained geometry; PMX-K16's own wells are isotropic)
1 - R^2 is the share left for content. Only the occupations change between ranges; the overlaps E are the same.
"""
import math
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

PFX = 'semsimula_fock_cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_'
SFX = '_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn'
BASE = P.CONFIGS['F3.1'][1] + (("POISSON_MODES        = 0", "POISSON_MODES        = 16"),
                               ("LOWRANK_DAMPED_FLOW  = False", "LOWRANK_DAMPED_FLOW  = True"))
LR8 = BASE + (("POISSON_MATCH_RANK   = 0", "POISSON_MATCH_RANK   = 8"),)
RANGES = [(4, 128), (2, 64), (2, 32), (1, 32), (1, 16), (1, 8), (1, 4)]


class _Stop(Exception):
    pass


def build_init(repl):
    load = torch.load
    torch.load = lambda *a, **k: (_ for _ in ()).throw(_Stop())
    torch.manual_seed(0); np.random.seed(0)
    try:
        P.build(P.F31_F, repl)
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
        caps.setdefault(int(layer_idx), []).append(h.detach().clone())
        return orig(h, h_prev, r, sal, m_b, gamma, dt, layer_idx=layer_idx)
    model._fock_layer_step = rec
    with torch.enable_grad():
        for x in xs:
            model(torch.from_numpy(x))
    model._fock_layer_step = orig
    return {l: torch.cat(v) for l, v in caps.items()}                  # (8, 512, d)


def occupation(E, hl_min, hl_max):
    K, T = E.shape[-1], E.shape[1]
    hl = torch.logspace(math.log10(hl_min), math.log10(hl_max), K)
    log_lam = torch.log(0.5 ** (1.0 / hl))
    t = torch.arange(T)
    lag = t[:, None] - t[None, :] - 1
    Lam = torch.where(lag >= 0, torch.exp(lag.clamp_min(0)[None].float() * log_lam[:, None, None]), torch.zeros(()))
    return torch.einsum('kts,bsk->btk', Lam, E)


def position_r2(phi):
    loo = (phi.sum(0, keepdim=True) - phi) / (phi.shape[0] - 1)
    return 1 - float(((phi - loo) ** 2).sum() / ((phi - phi.mean((0, 1))) ** 2).sum())


m_lr, g = build_init(LR8)
m_lr.to('cpu'); m_lr.cfg.use_layer_checkpoint = False
rng = np.random.default_rng(20261009)
xs = [g['get_batch'](g['val_ids'], 4, 512, rng)[0] for _ in range(2)]
m_lr.eval()
H_init = layer_inputs(m_lr, xs)                                       # states at initialisation (matcher not yet set)
m_tr, _, _ = P.build(PFX + 'pm16_pmx16_sr2' + SFX, BASE + (("POISSON_WELLS_EXACT  = False", "POISSON_WELLS_EXACT  = True"),))
m_tr.to('cpu'); m_tr.eval(); m_tr.cfg.use_layer_checkpoint = False
H_tr = layer_inputs(m_tr, xs)

print('OCCUPATIONS: SHARE OF VARIANCE EXPLAINED BY POSITION, BY HALF-LIFE RANGE  (rank-8 matcher, K = 16; '
      '8 x 512 validation tokens; 1 - R^2 is left for content)')
for cloud, H in (('init', H_init), ('trained', H_tr)):
    for l in sorted(H):
        m_lr.pm_match_ready[l] = 0.0
        m_lr.train()
        with torch.no_grad():
            E = m_lr._pm_match(H[l], l)[0]                            # initialises the matcher on this cloud
        m_lr.eval()
        Ef = E.reshape(-1, E.shape[-1])
        sel = float((Ef.std(0) / Ef.mean(0).clamp_min(1e-30)).median())
        cells = []
        for lo, hi in RANGES:
            cells.append(f'{lo}-{hi}: {position_r2(occupation(E, lo, hi)):.3f}')
        print(f'  {cloud:7s} layer {l} (overlap selectivity {sel:.2f}):  ' + '   '.join(cells), flush=True)
    m_lr.pm_match_ready.zero_()
