"""SR-pi.3: is G2's refinement failure concentrated where stiff modes cross pi? (protocol SS5.9)

Pre-registered 2026-10-04 (55%): on G2's checkpoint, split the Gate 3 loss
rise by token according to whether the token's stiff modes cross pi under
refinement; crossing tokens' mean loss rise is >= 2x that of non-crossing ones.

  - tokens: Cell 6b-13's draw (seed 20260928, 8 x 4 x 512), from the local
    validation cache
  - trained pass (N = L = 2, dt = 4): per-token NLL, and theta = omega*dt per
    token and layer from the 6b-13 resonance monitor (semsimula-diag)
  - refined pass (N = 3, dt = T/3, 'hold' policy): per-token NLL, through
    Cell 6b-7's own _fom_stack, so the refinement is Gate 3's exactly
  - a token crosses pi at a layer if theta in (pi, 1.5 pi): refinement scales
    dt by L/N = 2/3, so theta * 2/3 < pi < theta. Classes per token: 'below'
    (every layer below pi), 'crossing' (some layer crosses), 'past' (some layer
    stays past pi after refinement, none crosses)

Usage: python3 sr_pi3_crossing.py OUT_DIR
"""
import json, math, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path.home() / 'git/ml/semsimula-diag/src'))
import gradcheck_vphi_xi_paths as G
import verify_pm_switch as VP
from semsimula_diag.probes import resonance

N_REF, SEED, NB, BS, BLOCK = 3, 20260928, 8, 4, 512


def nll_tokens(logits, y):
    return torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), y.flatten(),
                                             reduction='none').view(y.shape)


if __name__ == '__main__':
    model, g, _ = VP.build(VP.G2_F, VP.CONFIGS['G2'][1])
    model.cfg.use_layer_checkpoint = False
    nb = json.load(open(G.NB))
    c19 = [''.join(c['source']) for c in nb['cells'] if ''.join(c['source']).startswith('# == Cell 6b-7')][0]
    exec(compile(G.strip(c19[:c19.index('_f7_saved =')]), 'Cell6b7defs', 'exec'), g)
    _fom_stack = g['_fom_stack']
    mpx = sys.modules[type(model)._layer_step_langevin.__module__]
    L, dt = model.cfg.L, float(model.cfg.dt)
    T_int = L * dt
    rng = np.random.default_rng(SEED)
    val = np.load(G.VAL)
    batches = [tuple(torch.from_numpy(a) for a in g['get_batch'](val, BS, BLOCK, rng)) for _ in range(NB)]
    th_all, d_all, n0_all = [], [], []
    model.eval()
    for xb, yb in batches:
        with resonance.observe(model, mpx, wall=2.0) as mon:
            with torch.enable_grad():
                lo0, _ = model(xb)
        th = torch.stack([torch.cat(mon.per_layer[l]).view(xb.shape) for l in range(L)], -1)   # (B, T, L)
        with _fom_stack(model, N_REF, T_int / N_REF, 'hold'):
            with torch.enable_grad():
                lo1, _ = model(xb)
        n0, n1 = nll_tokens(lo0, yb), nll_tokens(lo1, yb)
        th_all.append(th.flatten(0, 1)); d_all.append((n1 - n0).flatten()); n0_all.append(n0.flatten())
        print(f"  batch: PPL trained {math.exp(n0.mean()):.2f}  refined {math.exp(n1.mean()):.2f}", flush=True)
    th = torch.cat(th_all); dn = torch.cat(d_all); n0 = torch.cat(n0_all)
    pi, scale = math.pi, L / N_REF
    cross_l = (th > pi) & (th * scale < pi)                    # (Ntok, L)
    past_l = th * scale >= pi
    crossing = cross_l.any(-1)
    past = ~crossing & past_l.any(-1)
    below = ~crossing & ~past
    gate3 = math.exp(dn.mean())
    res = {'n_tokens': int(dn.numel()), 'gate3_ratio': gate3,
           'theta_median_per_layer': [float(th[:, l].median()) for l in range(L)],
           'theta_median_pooled': float(th.flatten().median())}
    print(f"\nG2, {dn.numel():,} tokens: refined/trained PPL = {gate3:.2f} (Gate 3 = {100 * (gate3 - 1):+.0f}%)")
    print(f"theta median per layer {', '.join(f'{v:.2f}' for v in res['theta_median_per_layer'])}; "
          f"pooled {res['theta_median_pooled']:.2f}")
    print(f"\n{'class':>10} {'share':>7} {'mean dNLL':>10} {'median':>8} {'trained NLL':>12}")
    for name, m in (('below', below), ('crossing', crossing), ('past', past)):
        res[name] = {'share': float(m.float().mean()), 'mean_dnll': float(dn[m].mean()) if m.any() else None,
                     'median_dnll': float(dn[m].median()) if m.any() else None,
                     'trained_nll': float(n0[m].mean()) if m.any() else None}
        r = res[name]
        print(f"{name:>10} {100 * r['share']:6.1f}% {r['mean_dnll'] if r['mean_dnll'] is not None else float('nan'):10.3f} "
              f"{r['median_dnll'] if r['median_dnll'] is not None else float('nan'):8.3f} "
              f"{r['trained_nll'] if r['trained_nll'] is not None else float('nan'):12.3f}")
    ncr = ~crossing
    ratio = float(dn[crossing].mean() / dn[ncr].mean())
    res['ratio_crossing_vs_noncrossing'] = ratio
    print(f"\nmean dNLL, crossing / non-crossing = {ratio:.2f}   (pre-registered: >= 2, called 55%)")
    for l in range(L):
        rl = float(dn[cross_l[:, l]].mean() / dn[~cross_l[:, l]].mean())
        res[f'ratio_layer{l}'] = rl
        print(f"   crossing at layer {l} only counted: ratio {rl:.2f}  (share {100 * cross_l[:, l].float().mean():.1f}%)")
    # dose-response in the largest theta
    tmax = th.max(-1).values
    edges = [0, 2, 2.5, 3, pi, 3.6, 4.2, 1.5 * pi, 6, 99]
    print(f"\nmean dNLL by the token's largest theta (pi = {pi:.2f}, 1.5 pi = {1.5 * pi:.2f}):")
    res['dose'] = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (tmax >= a) & (tmax < b)
        if m.sum() > 20:
            res['dose'].append({'lo': a, 'hi': b, 'share': float(m.float().mean()), 'mean_dnll': float(dn[m].mean())})
            print(f"   [{a:4.2f}, {b:5.2f})  share {100 * m.float().mean():5.1f}%  mean dNLL {float(dn[m].mean()):.3f}")
    # control for difficulty: crossing vs not within trained-NLL quartiles
    q = torch.quantile(n0, torch.tensor([0.25, 0.5, 0.75]))
    bins = torch.bucketize(n0, q)
    print("\nwithin quartiles of trained NLL (controls for token difficulty):")
    res['by_difficulty'] = []
    for k in range(4):
        m = bins == k
        c, nc = m & crossing, m & ~crossing
        if c.sum() > 20 and nc.sum() > 20:
            rk = float(dn[c].mean() / dn[nc].mean())
            res['by_difficulty'].append(rk)
            print(f"   quartile {k + 1}: crossing {float(dn[c].mean()):.3f}  non-crossing {float(dn[nc].mean()):.3f}  ratio {rk:.2f}")
    verdict = 'HIT' if ratio >= 2 else 'MISS'
    print(f"\n-> SR-pi.3 {verdict}")
    res['verdict'] = verdict
    (G.OUT / 'sr_pi3_crossing.json').write_text(json.dumps(res, indent=1))
