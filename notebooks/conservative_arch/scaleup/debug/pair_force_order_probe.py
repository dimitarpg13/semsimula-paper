#!/usr/bin/env python3
"""How the pair force acts in a first-order step against a second-order one (2026-11-24, after FO-b-PMX-K16-NP).

    python3 pair_force_order_probe.py OUT_DIR {pmx|fob-pmx} folder=<run folder> [ckpt=<file>] [name=<stem>]
                                      [batches=2] [threads=8] [eps=0.01] [refine=2,4]

FO-b-PMX-K16 trained without the pair force V_phi settles 0.6% BELOW the same model trained with it, while SR2
trained without it settles 0.3% above SR2. The hypothesis under test: in both orders V_phi enters the layer step only
through its one explicit kick (beside the soft modes of V_theta), but the kick lands in different places. In the
first-order step (FO-b, `_overdamped_step`) it moves the state directly, h_mid += c f; in the second-order step it
changes the velocity, v_mid += (dt / m) f, which the exact damped flow then carries and damps, and which passes into
the next layer. The coefficient of the force in the state is the same in both: c = mu dt = dt^2 / (2 m) in first
order, (dt / 2)(dt / m) in the second-order half substep. A rough, top-k-routed force applied as one explicit
position update is the suspect.

Readings per layer, at the point h_mid where the step evaluates its kick (best or given checkpoint, inference only):
  share     |c f_phi| / |h_out - h_in|: the pair force's displacement against the whole layer step
  ratio     |f_phi| / |f_rest|, f_rest = the conservative force without V_phi (V_theta + wells), and
  cos       cos(f_phi, f_rest): does the pair force oppose the rest of the force?
  rough     (|f(h + d) - f(h)| / |f(h)|) / eps, |d| = eps |h| per token, Gaussian direction: the relative change
            of the force per relative change of the state, for f_phi and for f_rest
  switch    the share of each position's top-k pair-force sources that change under the same perturbation
  drift     |f_phi(h_mid + c f_phi) - f_phi(h_mid)| / |f_phi(h_mid)|: how much the pair force changes over the
            displacement its own explicit update produces (the update is self-consistent when this is small), and
  switch_own  the share of top-k sources re-selected after that displacement
  step, h_in  |h_out - h_in| and |h_in|; h_out is after the LayerNorm projection, so at layer 0 (h_in = the
            embedding) the step is mostly the projection, and disp_phi_rel_h = |c f_phi| / |h_in| is read at layer 1
and over the batches:
  vphi_off  the loss change when V_phi is switched off at inference (reliance), per sequence
  refine    first order only: the explicit kick split into k sub-kicks of c / k, with (a) only V_phi re-evaluated
            at each sub-kick, (b) the whole explicit force re-evaluated; the loss change against the trained
            single kick. A map whose V_phi part does work only as one Euler step changes more under (a).
For a model without V_phi only the geometry readings that do not need it are written.
"""
import json, sys, time
from pathlib import Path

import numpy as np
import torch

OUT = Path(sys.argv[1]); KIND = sys.argv[2]
OPTS = dict(a.split('=', 1) for a in sys.argv[3:] if '=' in a)
FOLDER = OPTS['folder']; CKPT = OPTS.get('ckpt'); NAME = OPTS.get('name', KIND)
NB, THREADS, EPS = int(OPTS.get('batches', 2)), int(OPTS.get('threads', 8)), float(OPTS.get('eps', 0.01))
REFINE = [int(k) for k in OPTS.get('refine', '2,4').split(',')]
assert KIND in ('pmx', 'fob-pmx')
sys.argv = [sys.argv[0], str(OUT / 'harness')]
sys.path.insert(0, str(Path(__file__).parent))
import verify_pm_switch as P
torch.set_num_threads(THREADS)

# the build of mechanism_ablations.py
repl = P.CONFIGS['F3.1'][1] + P.PM_ON + (("LOWRANK_DAMPED_FLOW  = False", "LOWRANK_DAMPED_FLOW  = True"),
                                         ("POISSON_WELLS_EXACT  = False", "POISSON_WELLS_EXACT  = True"))
_k = next((int(t[2:]) for t in FOLDER.split('_') if t.startswith('pm') and t[2:].isdigit()), 64)
if _k != 64:
    repl = repl + (("POISSON_MODES        = 64", f"POISSON_MODES        = {_k}"),)
if KIND == 'fob-pmx':
    repl = repl + (("FIRST_ORDER_OVERDAMPED = False", "FIRST_ORDER_OVERDAMPED = True"),)
if 'nophi' in FOLDER.split('_'):
    repl = tuple(r for r in repl if 'VPHI_GRAD_PATH' not in r[0]) + \
        (("PAIR_POTENTIAL  = 'sparse_topk'", "PAIR_POTENTIAL  = 'none'"),)
model, g, _ = P.build(FOLDER, repl, ckpt=CKPT)
m = model; m.to('cpu'); m.eval(); m.cfg.use_layer_checkpoint = False
FOB = bool(m.cfg.first_order_overdamped)
assert FOB == (KIND == 'fob-pmx') and m.pm_depth.shape[1] == _k
PHI = getattr(m, 'V_phi', None) is not None
MOD = sys.modules[type(m)._layer_step_langevin.__module__]

rng = np.random.default_rng(20261009)                  # mechanism_ablations.py's batches (its first NB)
batches = [g['get_batch'](g['val_ids'], 4, 512, rng) for _ in range(NB)]
gen = torch.Generator().manual_seed(20261124)

st = {'phi_off': False, 'measure': False, 'refine': None, 'topk': None}
if PHI:
    orig_pf, orig_pp = m._pair_force_analytic, m._pair_potential
    m._pair_force_analytic = lambda h_in, *a, **k: torch.zeros_like(h_in) if st['phi_off'] else orig_pf(h_in, *a, **k)
    m._pair_potential = lambda h_in, *a, **k: torch.zeros((), device=h_in.device) if st['phi_off'] else orig_pp(h_in, *a, **k)
    orig_topk = m._sparse_topk_indices
    def topk(*a, **k):
        r = orig_topk(*a, **k)
        if st['topk'] is not None:
            st['topk'].append(r[0].detach())
        return r
    m._sparse_topk_indices = topk

orig_forces = m._layer_forces
def forces(h, xis, layer_idx, phi_off=False, **kw):
    """(f_theta + f_phi) of the conservative force at h, with V_phi on or off; detached."""
    prev = st['phi_off']; st['phi_off'] = phi_off
    hh = h.detach().requires_grad_(True)
    with torch.enable_grad():
        ft, fp = orig_forces(hh, xis, layer_idx, split=True, **kw)
    st['phi_off'] = prev
    return (ft + fp).detach()

nrm = lambda x: x.norm(dim=-1)
rec = {}                                               # layer -> reading -> list of per-token arrays
cur = {}
def put(l, key, v):
    rec.setdefault(l, {}).setdefault(key, []).append(v.detach().reshape(-1).numpy())

def lf_wrap(h_in, xis, layer_idx, *, split=False, vtheta_comps=None):
    out = orig_forces(h_in, xis, layer_idx, split=split, vtheta_comps=vtheta_comps)
    if not (st['measure'] and split and cur.get('layer') == layer_idx and not cur.get('done')):
        return out
    cur['done'] = True
    kw = {'vtheta_comps': vtheta_comps}
    f_all = (out[0] + out[1]).detach()
    f_rest = forces(h_in, xis, layer_idx, phi_off=True, **kw) if PHI else f_all
    f_phi = f_all - f_rest
    c = cur['c']
    cur['c_fphi'] = c * f_phi
    put(layer_idx, 'f_rest', nrm(f_rest))
    hn = h_in.detach()
    d = torch.randn(hn.shape, generator=gen); d = d / nrm(d).unsqueeze(-1) * (EPS * nrm(hn).unsqueeze(-1))
    if PHI:
        put(layer_idx, 'f_phi', nrm(f_phi))
        put(layer_idx, 'ratio', nrm(f_phi) / nrm(f_rest).clamp_min(1e-30))
        put(layer_idx, 'cos', torch.nn.functional.cosine_similarity(f_phi, f_rest, dim=-1))
        st['topk'] = []
        f_all_d = forces(hn, xis, layer_idx, **kw)            # records the base selection again
        sel0 = st['topk'][-1] if st['topk'] else None
        st['topk'] = []
        f_all_p = forces(hn + d, xis, layer_idx, **kw)
        sel1 = st['topk'][-1] if st['topk'] else None
        st['topk'] = None
        f_rest_p = forces(hn + d, xis, layer_idx, phi_off=True, **kw)
        f_phi_p = f_all_p - f_rest_p
        put(layer_idx, 'rough_phi', nrm(f_phi_p - f_phi) / nrm(f_phi).clamp_min(1e-30) / EPS)
        put(layer_idx, 'rough_rest', nrm(f_rest_p - f_rest) / nrm(f_rest).clamp_min(1e-30) / EPS)
        put(layer_idx, 'base_repeat', nrm(f_all_d - f_all) / nrm(f_all).clamp_min(1e-30))
        if sel0 is not None and sel1 is not None:
            k_ = sel0.shape[-1]
            same = (sel0.unsqueeze(-1) == sel1.unsqueeze(-2)).any(-1).float().sum(-1) / k_
            put(layer_idx, 'switch', 1.0 - same)
        # drift of f_phi over its own displacement c f_phi (the same coefficient in both orders): how far one
        # explicit update of the pair force is from self-consistent, and how many sources it re-selects
        hs = hn + c * f_phi
        st['topk'] = []
        f_all_s = forces(hs, xis, layer_idx, **kw)
        sel2 = st['topk'][-1] if st['topk'] else None
        st['topk'] = None
        f_phi_s = f_all_s - forces(hs, xis, layer_idx, phi_off=True, **kw)
        put(layer_idx, 'drift_phi', nrm(f_phi_s - f_phi) / nrm(f_phi).clamp_min(1e-30))
        if sel0 is not None and sel2 is not None:
            same = (sel0.unsqueeze(-1) == sel2.unsqueeze(-2)).any(-1).float().sum(-1) / sel0.shape[-1]
            put(layer_idx, 'switch_own', 1.0 - same)
    else:
        f_rest_p = forces(hn + d, xis, layer_idx, **kw)
        put(layer_idx, 'rough_rest', nrm(f_rest_p - f_rest) / nrm(f_rest).clamp_min(1e-30) / EPS)
    return out
m._layer_forces = lf_wrap

orig_core = m._layer_step_langevin_core
def core(h, h_prev, m_b, gamma, dt, layer_idx, *a, **k):
    if st['measure']:
        cur.clear(); cur.update(layer=layer_idx, c=dt * dt / (2.0 * m_b))
    out = orig_core(h, h_prev, m_b, gamma, dt, layer_idx, *a, **k)
    if st['measure'] and cur.get('done'):
        step = nrm((out[0] - h).detach())
        put(layer_idx, 'step', step)
        put(layer_idx, 'h_in', nrm(h.detach()))
        if PHI:
            put(layer_idx, 'share', nrm(cur['c_fphi']) / step.clamp_min(1e-30))
            put(layer_idx, 'disp_phi_rel_h', nrm(cur['c_fphi']) / nrm(h.detach()).clamp_min(1e-30))
    return out
m._layer_step_langevin_core = core

# first order: the explicit kick split into k sub-kicks (a: only V_phi re-evaluated; b: the whole kick re-evaluated)
if FOB:
    orig_od = m._overdamped_step
    def od(h_in, xis, vtheta_comps, layer_idx, dt, m_b, lr_U, lr_kappa, alpha, a_force, kick_sub, flow_ctx):
        if st['refine'] is None:
            return orig_od(h_in, xis, vtheta_comps, layer_idx, dt, m_b, lr_U, lr_kappa, alpha, a_force, kick_sub, flow_ctx)
        mode, K = st['refine']; cfg = m.cfg; half = 0.5 * dt
        mu = getattr(m, '_fob_dt', cfg.dt) / (2.0 * m_b)
        def relax(hh):
            if alpha is None:
                return MOD.lowrank_overdamped_substep(hh, lr_U, lr_kappa, a_force(hh), mu, half)
            return MOD.lowrank_iso_overdamped_substep(hh, lr_U, lr_kappa, alpha, a_force(hh), mu, half)
        def kick(hh, phi_off=False):
            hh = hh.detach().requires_grad_(True)
            prev = st['phi_off']; st['phi_off'] = phi_off
            with torch.enable_grad():
                ft, fp = orig_forces(hh, xis, layer_idx, split=True, vtheta_comps=vtheta_comps)
                f = (ft + fp - kick_sub(hh)).detach()
            st['phi_off'] = prev
            return f
        clamp = lambda f: f.clamp(-cfg.force_clamp_max, cfg.force_clamp_max) if cfg.force_clamp_max is not None else f
        h_mid = relax(h_in).detach()
        if mode == 'phi':
            f_rest = kick(h_mid, phi_off=True)
            hh = h_mid
            for _ in range(K):
                f_phi = kick(hh) - kick(hh, phi_off=True)
                hh = hh + (mu * dt / K) * clamp(f_rest + f_phi)
        else:
            hh = h_mid
            for _ in range(K):
                hh = hh + (mu * dt / K) * clamp(kick(hh))
        h_new = relax(hh)
        if cfg.ln_after_step and (flow_ctx is None or flow_ctx.get("project", True)):
            h_new = m._project(h_new)
        return h_new, h_new
    m._overdamped_step = od

def nll_of(xb, yb):
    with torch.enable_grad():
        lo = m(torch.from_numpy(xb))[0].detach()
    ce = torch.nn.functional.cross_entropy(lo.reshape(-1, lo.shape[-1]), torch.from_numpy(yb).reshape(-1), reduction='none')
    ce = ce.reshape(xb.shape).numpy()
    pos_nll.setdefault(st.get('arm'), []).append(ce)
    return ce.mean(1)

t0 = time.time()
pos_nll = {}                                           # arm -> list of (B, T) per-token losses
BUCKETS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]  # position buckets [a, b)
arms = {'base': [], 'base_measured': []}
if PHI:
    arms['vphi_off'] = []
if FOB:
    for K in REFINE:
        if PHI:
            arms[f'refine_phi_k{K}'] = []
        arms[f'refine_all_k{K}'] = []
for xb, yb in batches:
    st['arm'] = 'base'; arms['base'].append(nll_of(xb, yb))
    st['arm'] = 'base_measured'; st['measure'] = True; arms['base_measured'].append(nll_of(xb, yb)); st['measure'] = False
    if PHI:
        st['arm'] = 'vphi_off'; st['phi_off'] = True; arms['vphi_off'].append(nll_of(xb, yb)); st['phi_off'] = False
    if FOB:
        for K in REFINE:
            for mode in (('phi', 'all') if PHI else ('all',)):
                st['arm'] = f'refine_{mode}_k{K}'; st['refine'] = (mode, K); arms[f'refine_{mode}_k{K}'].append(nll_of(xb, yb)); st['refine'] = None
    print(f'   batch done ({time.time() - t0:.0f}s)', flush=True)
arms = {a: np.concatenate(v) for a, v in arms.items()}
assert np.allclose(arms['base'], arms['base_measured'], atol=1e-6), 'the measurement hooks changed the forward'

def q(v):
    v = np.concatenate(v); return [float(np.percentile(v, p)) for p in (5, 50, 95)] + [float(v.mean())]
lines = [f'PAIR FORCE IN THE STEP: {KIND}  {FOLDER.split("xilive_")[1].split("_L2probe")[0]}  ({CKPT or "best checkpoint"}; '
         f'{NB} x 4 x 512 tokens, batch seed 20261009; perturbation eps = {EPS})',
         f'   first order: {FOB}   pair force: {PHI}   base perplexity {np.exp(arms["base"].mean()):.2f}',
         '   per-token readings at the kick point, p05 / p50 / p95 (mean)']
summ = {}
for l in sorted(rec):
    summ[l] = {key: q(v) for key, v in rec[l].items()}
    for key, (a, b, c_, mn) in summ[l].items():
        lines.append(f'   layer {l}  {key:15s} {a:10.4g} {b:10.4g} {c_:10.4g}   ({mn:.4g})')
lines.append('   loss change against the trained step, nats per token (mean over sequences; se over sequences)')
dl = {}
for a in arms:
    if a in ('base', 'base_measured'):
        continue
    d = arms[a] - arms['base']; dl[a] = [float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))]
    lines.append(f'   {a:16s} {d.mean():+.5f} ({dl[a][1]:.5f})   perplexity change {np.expm1(d.mean()) * 100:+.2f}%')
T_ = 512
def by_pos(v):
    a = np.concatenate([x.reshape(-1, T_) for x in v])
    return [float(a[:, lo:hi].mean()) for lo, hi in zip(BUCKETS[:-1], BUCKETS[1:])]
prof = {l: {key: by_pos(v) for key, v in rec[l].items()} for l in rec}
pnll = {a: by_pos(v) for a, v in pos_nll.items() if a is not None}
lines.append('   by position bucket [a, b): ' + '  '.join(f'{lo}-{hi - 1}' for lo, hi in zip(BUCKETS[:-1], BUCKETS[1:])))
for a in pnll:
    lines.append(f'   loss {a:16s} ' + ' '.join(f'{x:7.4f}' for x in pnll[a]))
for l in sorted(prof):
    for key in ('ratio', 'cos', 'disp_phi_rel_h', 'rough_phi', 'drift_phi', 'f_phi', 'f_rest'):
        if key in prof[l]:
            lines.append(f'   layer {l} {key:15s}   ' + ' '.join(f'{x:7.4g}' for x in prof[l][key]))
lines.append(f'   ({time.time() - t0:.0f}s)')
print('\n'.join(lines))
OUT.mkdir(parents=True, exist_ok=True)
(OUT / f'pair_force_order_{NAME}.txt').write_text('\n'.join(lines) + '\n')
(OUT / f'pair_force_order_{NAME}.json').write_text(json.dumps(
    {'folder': FOLDER, 'ckpt': CKPT, 'kind': KIND, 'first_order': FOB, 'pair_force': PHI, 'eps': EPS, 'batches': NB,
     'readings_p05_p50_p95_mean': summ, 'loss_change_nats_mean_se': dl,
     'position_buckets': BUCKETS, 'readings_by_position': prof, 'loss_by_position': pnll,
     'per_sequence_nll': {a: v.tolist() for a, v in arms.items()}}, indent=1))
