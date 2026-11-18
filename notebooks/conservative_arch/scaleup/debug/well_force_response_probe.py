#!/usr/bin/env python3
"""Concentration or sensitivity? Each well's share of the actual force, and what removing it costs (2026-11-17,
protocol p-bank-memory-conjecture).

    python3 well_force_response_probe.py OUT FOLDER [ckpt=<file>] [layers=0,1] [delta=1e-3] [windows=16] [threads=8]
                                         [val=<tokens.npy>] [name=<stem>] [check=1] [double=1] [drops=0]

At the trained state of every token and layer, well k's force is its term of the bank's analytical force,
    f_k = g_k [ a_k * (mu_k - h) + B_k B_k^T (mu_k - h) ],   g_k = w_k occ_k,
(model_aniso_gaussian_vtheta.harmonic_terms_lowrank: the terms sum to -grad V_theta). Its force share is
E_t |f_k| / sum_j E_t |f_j| over the tokens; its weight share, as in vtheta_well_probe, is E_t g_k / E_t sum_j g_j.
Then, one well and one layer at a time, w_k is scaled by (1 - delta) on that layer:
  - delta = 1 (drop one well): the change of the loss in nats, of the centred logits and of the layer's output;
  - delta small (linear response): the same changes divided by delta.
Removal cost = force share x response per unit share: the first factor is concentration, the second sensitivity.
Writes OUT/force_response_<stem>.txt and .json.
"""
import json, sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(sys.argv[1]).resolve(); FOLDER = sys.argv[2]
OPTS = dict(a.split('=', 1) for a in sys.argv[3:] if '=' in a)
DBG = Path(__file__).resolve().parent
sys.path.insert(0, str(DBG))

# copy_split's own Cell-0-from-tag build and validation windows, without its command dispatch (as vtheta_well_probe)
src = (DBG / 'copy_split.py').read_text()
src = src[:src.index("\nif MODE == 'nll':")]
sys.argv = ['copy_split.py', str(OUT), 'probe', f"windows={OPTS.get('windows', 16)}", f"threads={OPTS.get('threads', 8)}"] \
    + ([f"val={OPTS['val']}"] if 'val' in OPTS else [])
CS = {'__name__': 'copy_split', '__file__': str(DBG / 'copy_split.py')}
exec(compile(src, str(DBG / 'copy_split.py'), 'exec'), CS)

import verify_pm_switch as P
_build = P.build; got = {}
def build(*a, **k):
    r = _build(*a, **k); got['model'] = r[0]; return r
P.build = build
_, desc = CS['ladder_model'](FOLDER, OPTS.get('ckpt'))
model = got['model']
print('model:', desc, flush=True)
if OPTS.get('double', '1') == '1':
    model.double()
FOB = bool(getattr(model.cfg, 'first_order_overdamped', False))
LAYERS = [int(t) for t in OPTS.get('layers', '0,1').split(',')]
DELTA = float(OPTS.get('delta', 1e-3)); CHECK = OPTS.get('check', '0') == '1'
DROPS = OPTS.get('drops', '1') == '1'      # drops=0: the force factors only, no removals

VT = model.V_theta
bank = VT.bank.bank if hasattr(VT.bank, 'bank') else VT.bank      # depth-conditioned -> joint -> mixture
K = bank.K

_mod = sys.modules[type(model)._layer_step_langevin.__module__]
_orig_modes = _mod.lowrank_modes
N_RETRY = [0]
def _modes(G, *a, **k):               # 2026-11-17: a non-converging CPU SVD (code 'gram' driver): jittered retry, counted
    try:
        return _orig_modes(G, *a, **k)
    except RuntimeError:
        N_RETRY[0] += 1
        gj = torch.Generator().manual_seed(N_RETRY[0])
        sc = G.detach().abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        return _orig_modes(G + 1e-10 * sc * torch.randn(G.shape, generator=gj, dtype=G.dtype), *a, **k)
_mod.lowrank_modes = _modes

st = {'layer': None, 'scale': None, 'lr': set(), 'force': None, 'out': {}}
orig_step = model._layer_step_langevin
def step(h, h_prev, m_b, gamma, dt_, layer_idx=0):
    st['layer'] = int(layer_idx)
    h_new, h_prev_out = orig_step(h, h_prev, m_b, gamma, dt_, layer_idx)
    st['out'][int(layer_idx)] = h_new.detach()
    return h_new, h_prev_out
model._layer_step_langevin = step

_orig_comp = bank._components
def comp(xi):
    mu, a, w, B = _orig_comp(xi)
    if st['scale'] is not None and st['scale'][0] == st['layer']:
        _, k, f = st['scale']
        s = torch.ones(K, dtype=w.dtype); s[k] = f
        w = w * s
    return mu, a, w, B
bank._components = comp

orig_lr = bank.harmonic_terms_lowrank
def lr(xi, h, *, comps=None):
    out = orig_lr(xi, h, comps=comps)
    L = st['layer']
    if st['force'] is not None and L not in st['lr']:      # the unperturbed run: each well's force, first call per layer
        st['lr'].add(L)
        with torch.no_grad():
            mu, a, w, B = (c.detach() for c in (comps if comps is not None else bank._components(xi)))
            dmu = mu - h.detach().unsqueeze(-2)                                         # (..., K, d)
            quad = (a * dmu * dmu).sum(-1) + (torch.einsum('...kd,...kdr->...kr', dmu, B) ** 2).sum(-1)
            g = w * torch.exp(-0.5 * quad)
            fk = g.unsqueeze(-1) * (a * dmu + torch.einsum('...kdr,...kr->...kd', B, torch.einsum('...kd,...kdr->...kr', dmu, B)))
            r = st['force'].setdefault(L, {'fnorm': [], 'g': [], 'ftot': [], 'lg': [], 'ls': [], 'ld': [], 'occ': []})
            r['fnorm'].append(fk.norm(dim=-1).reshape(-1, K)); r['g'].append(g.reshape(-1, K))
            # 2026-11-17: the factors of each well's force, |f_k| = g_k s_k d_k (logs, per token and well)
            Pd = a * dmu + torch.einsum('...kdr,...kr->...kd', B, torch.einsum('...kd,...kdr->...kr', dmu, B))
            dn = dmu.norm(dim=-1).clamp_min(1e-30)
            r['lg'].append(torch.log(g.clamp_min(1e-300)).reshape(-1, K)); r['ld'].append(torch.log(dn).reshape(-1, K))
            r['ls'].append(torch.log((Pd.norm(dim=-1) / dn).clamp_min(1e-300)).reshape(-1, K))
            r['occ'].append(torch.exp(-0.5 * quad).reshape(-1, K))
            r['ftot'].append(fk.sum(-2).norm(dim=-1).flatten())
            # check: the wells' terms add up to the model's own frozen force at h, f_a(h) + f_L(h)
            kd, sa, G, Gmu = (t.detach() for t in out)
            hh = h.detach()
            f_model = sa - kd * hh + torch.einsum('...dp,...p->...d', G, Gmu - torch.einsum('...dp,...d->...p', G, hh))
            err = float((fk.sum(-2) - f_model).norm() / f_model.norm().clamp_min(1e-30))
            st['check'] = max(st.get('check', 0.0), err)
    return out
bank.harmonic_terms_lowrank = lr


def forward(xb, yb):
    st['out'] = {}
    with torch.enable_grad():         # the V_phi pair force is an autograd gradient
        lg = model(torch.from_numpy(xb))[0].detach()
    ce = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), torch.from_numpy(yb).reshape(-1))
    return lg - lg.mean(-1, keepdim=True), float(ce), dict(st['out'])


def rms(t):
    return float(t.norm(dim=-1).pow(2).mean().sqrt())


x, y = CS['windows']()
force = {}; acc = {}; base_ce = []
for i in range(0, len(x), 4):
    xb, yb = x[i:i + 4], y[i:i + 4]
    st['scale'] = None; st['force'] = force; st['lr'] = set()
    lg0, ce0, out0 = forward(xb, yb); st['force'] = None
    assert st['lr'] >= set(LAYERS), st['lr']
    base_ce.append(ce0)
    for L in (LAYERS if DROPS else []):
        for k in range(K):
            ds = [1.0, DELTA] + ([DELTA / 2] if CHECK and k == 0 else [])
            for dl in ds:
                st['scale'] = (L, k, 1.0 - dl)
                lg, ce, out = forward(xb, yb)
                tag = 'drop' if dl == 1.0 else ('lin' if dl == DELTA else 'half')
                a = acc.setdefault((L, k, tag), {'dce': [], 'logit': [], 'hout': []})
                a['dce'].append((ce - ce0) / dl); a['logit'].append(rms((lg - lg0) / dl))
                a['hout'].append(rms((out[L] - out0[L]) / dl))
    st['scale'] = None
    print(f'windows {i + 4}/{len(x)}', flush=True)

res = {'base_ce': float(np.mean(base_ce)), 'layers': {}, 'force_check': st.get('check')}
lines = [f'Well force shares and removal response: {FOLDER}', f'model {desc}',
         f'{len(x)} validation windows x 512 ({Path(CS["VAL"]).name}); first order {FOB}; K {K}; delta {DELTA}; '
         f'base loss {res["base_ce"]:.4f} (ppl {np.exp(res["base_ce"]):.2f}); mode-solver retries {N_RETRY[0]}',
         'force share = E|f_k| / sum_j E|f_j|; weight share = E g_k / E sum_j g_j; drop = w_k -> 0 on that layer '
         '(loss change in nats, centred-logit RMS, layer-output RMS); lin = the same per unit delta of w_k -> (1-delta) w_k']
for L in LAYERS:
    fn = torch.cat(force[L]['fnorm']); g = torch.cat(force[L]['g']); ft = torch.cat(force[L]['ftot'])
    fs = (fn.mean(0) / fn.mean(0).sum()).tolist(); ws = (g.mean(0) / g.mean(0).sum()).tolist()
    cancel = float(ft.mean() / fn.sum(-1).mean())       # |sum_k f_k| / sum_k |f_k|: 1 = the wells pull together
    rows = []
    lg_, ls_, ld_ = (torch.cat(force[L][k]) for k in ('lg', 'ls', 'ld')); occ_ = torch.cat(force[L]['occ'])
    wf = fn / fn.sum(0, keepdim=True).clamp_min(1e-300)        # force-weighted: the tokens where the well actually pulls
    fac = {k: {'log_g': float((wf[:, k] * lg_[:, k]).sum()), 'log_s': float((wf[:, k] * ls_[:, k]).sum()),
               'log_d': float((wf[:, k] * ld_[:, k]).sum()),
               'log_s_unweighted': float(ls_[:, k].mean()), 'log_d_unweighted': float(ld_[:, k].mean()),
               'occ_median': float(occ_[:, k].median()), 'force_share': fs[k], 'weight_share': ws[k]} for k in range(K)}
    res.setdefault('factors', {})[L] = fac
    order = sorted(range(K), key=lambda k: -ws[k]); top, light = order[0], order[1:]
    lines += ['', f'layer {L}: force factors, |f_k| = g_k s_k d_k (force-weighted means over tokens of the logs; s = |P(mu-h)|/|mu-h|, '
              'the effective stiffness along the displacement; d = |mu-h|)',
              f'{"well":>4}  {"weight":>7}  {"force":>7}  {"<log g>":>8}  {"<log s>":>8}  {"<log d>":>8}  {"occ p50":>8}']
    for k in order:
        f_ = fac[k]
        lines.append(f'{k:>4}  {100 * ws[k]:6.1f}%  {100 * fs[k]:6.1f}%  {f_["log_g"]:8.3f}  {f_["log_s"]:8.3f}  {f_["log_d"]:8.3f}  {f_["occ_median"]:8.3f}')
    dS = float(np.mean([fac[k]['log_s'] for k in light]) - fac[top]['log_s'])
    dD = float(np.mean([fac[k]['log_d'] for k in light]) - fac[top]['log_d'])
    gm_s = float(np.exp(np.mean([fac[k]['log_s'] for k in light])))
    res.setdefault('light_vs_top', {})[L] = {'d_log_s': dS, 'd_log_d': dD, 'stiffness_part': dS / (dS + dD) if dS + dD != 0 else float('nan'),
                                              'light_s_geomean': gm_s, 'top_s_geomean': float(np.exp(fac[top]['log_s']))}
    lines.append(f'  light wells against the heaviest ({top}): force over g larger by log {dS + dD:+.3f} = stiffness {dS:+.3f} + distance '
                 f'{dD:+.3f} (stiffness part {res["light_vs_top"][L]["stiffness_part"]:.2f}); light wells\' geometric-mean s {gm_s:.4g}, '
                 f'heaviest {np.exp(fac[top]["log_s"]):.4g}')
    if not DROPS:
        continue
    lines += ['', f'layer {L}: |sum f_k| / sum |f_k| = {cancel:.3f}; check, sum_k f_k against the model force: '
                  f'largest relative difference {st.get("check", float("nan")):.1e} (all layers)',
              f'{"well":>4}  {"weight":>7}  {"force":>7}  {"drop nats":>9}  {"drop logit":>10}  {"drop h_out":>10}  '
              f'{"lin nats":>9}  {"lin logit":>9}  {"lin h_out":>9}']
    for k in sorted(range(K), key=lambda k: -fs[k]):
        d_, l_ = acc[(L, k, 'drop')], acc[(L, k, 'lin')]
        r = {'weight_share': ws[k], 'force_share': fs[k],
             'drop_nats': float(np.mean(d_['dce'])), 'drop_logit': float(np.sqrt(np.mean(np.square(d_['logit'])))),
             'drop_hout': float(np.sqrt(np.mean(np.square(d_['hout'])))),
             'lin_nats': float(np.mean(l_['dce'])), 'lin_logit': float(np.sqrt(np.mean(np.square(l_['logit'])))),
             'lin_hout': float(np.sqrt(np.mean(np.square(l_['hout']))))}
        if (L, k, 'half') in acc:
            h_ = acc[(L, k, 'half')]
            r['half_logit'] = float(np.sqrt(np.mean(np.square(h_['logit'])))); r['half_nats'] = float(np.mean(h_['dce']))
        rows.append((k, r))
        lines.append(f'{k:>4}  {100 * ws[k]:6.1f}%  {100 * fs[k]:6.1f}%  {r["drop_nats"]:+9.4f}  {r["drop_logit"]:10.4f}  '
                     f'{r["drop_hout"]:10.4f}  {r["lin_nats"]:+9.4f}  {r["lin_logit"]:9.4f}  {r["lin_hout"]:9.4f}')
    # response per unit force share: least squares through the origin over the wells
    fsv = np.array([r['force_share'] for _, r in rows])
    for key in ('drop_nats', 'lin_logit', 'lin_hout'):
        v = np.array([r[key] for _, r in rows])
        res.setdefault('slopes', {})[f'{L}:{key}_per_force_share'] = float((fsv * v).sum() / (fsv * fsv).sum())
    res['layers'][L] = {'cancel': cancel, 'wells': {str(k): r for k, r in rows}}
    lines.append(f'  per unit force share (fit through 0 over the wells): drop nats '
                 f'{res["slopes"][f"{L}:drop_nats_per_force_share"]:.3f}, lin logit '
                 f'{res["slopes"][f"{L}:lin_logit_per_force_share"]:.3f}, lin h_out {res["slopes"][f"{L}:lin_hout_per_force_share"]:.3f}')
    for k, r in rows:
        if 'half_logit' in r:
            lines.append(f'  linearity, well {k}: lin logit {r["lin_logit"]:.5f} at delta, {r["half_logit"]:.5f} at delta/2; '
                         f'nats {r["lin_nats"]:+.5f}, {r["half_nats"]:+.5f}')
stem = OPTS.get('name', FOLDER[-40:])
OUT.mkdir(parents=True, exist_ok=True)
(OUT / f'force_response_{stem}.txt').write_text('\n'.join(lines) + '\n')
(OUT / f'force_response_{stem}.json').write_text(json.dumps({'folder': FOLDER, 'model': desc, 'first_order': FOB,
                                                              'delta': DELTA, **res}, indent=1))
print('\n'.join(lines))
