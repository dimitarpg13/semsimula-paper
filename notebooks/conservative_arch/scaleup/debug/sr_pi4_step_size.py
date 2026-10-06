"""SR-pi.4: are the layer steps finite jumps that the projection rewrites? (protocol SS5.9)

Pre-registered 2026-10-05 before any measurement (60%): across F3.1, L=4 Fock
live, G2 and G3, the rank order of Gate 3 at 1.5x refinement matches the rank
order of the projection nonlinearity, averaged over layers.

Per arm, layer and token (4 x 512 validation tokens, eval, best checkpoint):
  cons_rel  ||x0 - h|| / ||h||: the conservative (BAOAB) step before its
            projection, relative to the layer-input state h
  inc_rel   ||delta|| / ||h||: the register increment (dt^2/m) s warm Q
  nonlin    ||P(x) - x|| / ||x - h||, for the layer's LAST projection: the share
            of the layer's displacement that the projection rewrites
  nonlin0   the same for the projection that ends the BAOAB step

Usage: python3 sr_pi4_step_size.py OUT_DIR
"""
import json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import gradcheck_vphi_xi_paths as G
import verify_pm_switch as VP
from exchange_field_probe import ARMS as EF_ARMS

L4_F = VP._P + 'vplive_xilive_L4probe' + VP._S.replace('idt4', 'idt2') + 'noattn'
ARMS = {  # name: (folder, Cell 0 replacements, Gate 3 at 1.5x from 6b-7, %)
    'F3.1': (VP.F31_F, VP.CONFIGS['F3.1'][1], 143),
    'L=4 Fock live': (L4_F, VP.LIVE + (("LADDER_L         = 2 ", "LADDER_L         = 4 "),), 216),
    'G2': (VP.G2_F, VP.CONFIGS['G2'][1], 1274),
    'G3': (EF_ARMS['G3'][0], EF_ARMS['G3'][1], 1342),
    "G3'": (EF_ARMS['G3p'][0], EF_ARMS['G3p'][1], 115),   # SR-pi.4b, added 2026-10-06
}
if len(sys.argv) > 2:                                      # e.g. G3' alone
    ARMS = {k: v for k, v in ARMS.items() if k in sys.argv[2:]}


def measure(model, x):
    cfg = model.cfg
    rec = {}
    cur = {}
    orig_step = model._fock_layer_step
    had_proj = '_project' in vars(model)
    orig_proj = model._project

    def step(h, h_prev, r, salience, m_b, gamma, dt, layer_idx=0, *a, **k):
        cur.clear(); cur.update(h=h.detach(), m_b=m_b, dt=dt, layer=layer_idx, proj=[], inc=None)
        out = orig_step(h, h_prev, r, salience, m_b, gamma, dt, layer_idx, *a, **k)
        rec.setdefault(layer_idx, []).append(dict(cur))
        return out

    def proj(xx):
        y = orig_proj(xx)
        if 'proj' in cur:
            cur['proj'].append((xx.detach(), y.detach()))
        return y

    def rev_hook(mod, args, out):
        l = cur['layer']
        raw = (model.reverse_channel_scale[l] if model.reverse_channel_scale.numel() > 1
               else model.reverse_channel_scale)
        sc = torch.tanh(raw)
        if cfg.reverse_channel_warmup_steps > 0:
            sc = sc * (model.reverse_warmup_step.float() / float(cfg.reverse_channel_warmup_steps)).clamp(max=1.0)
        cur['inc'] = ((cur['dt'] * cur['dt'] / cur['m_b']) * sc * out.detach())

    model._fock_layer_step = step
    model._project = proj
    hk = model.reverse_ch.register_forward_hook(rev_hook) if model.reverse_ch is not None else None
    try:
        model.eval()
        with torch.enable_grad():
            model(x)
    finally:
        model._fock_layer_step = orig_step          # assignment: the routing lives on the instance
        if had_proj:
            model._project = orig_proj
        else:
            del model._project                      # was a class method, never an instance attribute
        if hk is not None:
            hk.remove()
    out = {}
    for l, items in rec.items():
        it = items[0]
        h = it['h']; nh = h.norm(dim=-1).clamp(min=1e-12)
        (x0, y0) = it['proj'][0]
        (xl, yl) = it['proj'][-1]
        r = dict(cons_rel=(x0 - h).norm(dim=-1) / nh,
                 nonlin0=(y0 - x0).norm(dim=-1) / (x0 - h).norm(dim=-1).clamp(min=1e-12),
                 nonlin=(yl - xl).norm(dim=-1) / (xl - h).norm(dim=-1).clamp(min=1e-12),
                 n_proj=len(it['proj']))
        if it['inc'] is not None:
            r['inc_rel'] = it['inc'].norm(dim=-1) / nh
        out[l] = r
    return out


if __name__ == '__main__':
    val = np.load(G.VAL)
    st = np.random.default_rng(20261005).integers(0, len(val) - 513, size=4)
    x = torch.from_numpy(np.stack([val[s:s + 512] for s in st]).astype(np.int64))
    res = {}
    for name, (folder, repl, g3) in ARMS.items():
        model, g, _ = VP.build(folder, repl)
        model.cfg.use_layer_checkpoint = False
        m = measure(model, x)
        row = {'gate3': g3, 'layers': {}}
        print(f"\n== {name}  (L={model.cfg.L}, dt={model.cfg.dt:g}; Gate 3 at 1.5x: +{g3}%)")
        print(f"   {'layer':>5} {'cons_rel':>9} {'inc_rel':>8} {'nonlin0':>8} {'nonlin':>8}  (medians; projections per layer)")
        for l in sorted(m):
            r = m[l]
            med = {k: float(v.median()) for k, v in r.items() if torch.is_tensor(v)}
            row['layers'][l] = med
            print(f"   {l:5d} {med['cons_rel']:9.3f} {med.get('inc_rel', float('nan')):8.3f} "
                  f"{med['nonlin0']:8.3f} {med['nonlin']:8.3f}  ({r['n_proj']})")
        row['nonlin_mean'] = float(np.mean([v['nonlin'] for v in row['layers'].values()]))
        row['step_mean'] = float(np.mean([v['cons_rel'] + v.get('inc_rel', 0.0) for v in row['layers'].values()]))
        res[name] = row
        print(f"   mean over layers: nonlin {row['nonlin_mean']:.3f}; cons_rel + inc_rel {row['step_mean']:.3f}")
        del model
    by_g = sorted(res, key=lambda k: res[k]['gate3'])
    by_n = sorted(res, key=lambda k: res[k]['nonlin_mean'])
    by_s = sorted(res, key=lambda k: res[k]['step_mean'])
    print(f"\nrank by Gate 3:          {by_g}")
    print(f"rank by nonlinearity:    {by_n}   -> {'MATCH' if by_g == by_n else 'NO MATCH'} (pre-registered, 60%)")
    print(f"rank by relative step:   {by_s}   -> {'match' if by_g == by_s else 'no match'} (descriptive)")
    res['verdict'] = 'HIT' if by_g == by_n else 'MISS'
    (G.OUT / 'sr_pi4.json').write_text(json.dumps(res, indent=1))
    print(f"\n-> SR-pi.4 {res['verdict']}")
