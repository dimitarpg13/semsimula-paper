"""C1 and C2 of the clip-confound series on Gen 3 G2 (protocol, "The C-series").

C1  the reverse-channel slider extended ABOVE lambda = 1 (the trained gate):
    tanh(gate) -> lambda * tanh(gate) per layer, set on the gate parameter
    (atanh), with the warmup buffer left at its trained value (warm = 1). The
    Cell 6b-11 buffer method cannot exceed lambda = 1, because warm is clamped.
    Same tokens as 6b-11 (seed 20260925, 4 x 4 x 512). Pre-registered (from C0,
    which Gen 3 reproduces: the gate falls in training): PPL rises
    monotonically above lambda = 1.
C2  the Adam realised-step audit: |exp_avg| / sqrt(exp_avg_sq) per clip group,
    from the optimizer state in G2's best checkpoint, with the EXACT
    parameter -> optimizer-slot map: Cell 6's own split_decay_params ordering
    (decay tensors, then no-decay tensors), checked shape by shape.

Usage: python3 c1_c2_g2.py OUT_DIR
"""
import json, math, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import gradcheck_vphi_xi_paths as G
import verify_pm_switch as VP

if __name__ == '__main__':
    model, g, _ = VP.build(VP.G2_F, VP.CONFIGS['G2'][1])
    model.cfg.use_layer_checkpoint = False
    nb = json.load(open(G.NB))
    c6 = [''.join(c['source']) for c in nb['cells'] if ''.join(c['source']).startswith('# == Cell 6: Training loop')][0]
    res = {}

    # ---------------- C1
    rng = np.random.default_rng(20260925)
    val = np.load(G.VAL)
    xy = [tuple(torch.from_numpy(a) for a in g['get_batch'](val, 4, 512, rng)) for _ in range(4)]
    s0 = model.reverse_channel_scale.detach().clone()
    lams = (3.0, 2.0, 1.5, 1.25, 1.1, 1.0, 0.9, 0.75, 0.5, 0.25, 0.0)
    model.eval()
    ppl = {}
    try:
        for lam in lams:
            with torch.no_grad():
                model.reverse_channel_scale.copy_(torch.atanh((lam * torch.tanh(s0)).clamp(-1 + 1e-6, 1 - 1e-6)))
            L = 0.0
            for x, y in xy:
                with torch.enable_grad():
                    L += float(model(x, y)[1])
                model.zero_grad(set_to_none=True)
            ppl[lam] = math.exp(L / len(xy))
    finally:
        with torch.no_grad():
            model.reverse_channel_scale.copy_(s0)
    above = [ppl[l] for l in sorted(l for l in lams if l >= 1.0)]
    mono = all(b > a for a, b in zip(above, above[1:]))
    print(f"C1. G2 best, tanh(gate) per layer {[round(v, 4) for v in torch.tanh(s0).tolist()]}; 4 x 4 x 512 tokens")
    print(f"   {'lambda':>7} {'PPL':>9} {'vs lambda=1':>12}")
    for lam in lams:
        print(f"   {lam:7.2f} {ppl[lam]:9.2f} {100 * (ppl[lam] / ppl[1.0] - 1):+11.1f}%")
    print(f"   -> PPL rises monotonically above lambda = 1: {mono}  (pre-registered: yes)")
    res['C1'] = {'ppl': ppl, 'monotone_above_1': mono, 'gate_tanh': torch.tanh(s0).tolist()}

    # ---------------- C2
    a = c6.index('def split_decay_params')
    b = c6.index('\n\n\n', a) if '\n\n\n' in c6[a:] else c6.index('\nif ', a)
    exec(compile(c6[a:b], 'split', 'exec'), g)
    decay, no_decay = g['split_decay_params'](model)
    pid = {id(p): n for n, p in model.named_parameters()}
    order = [pid[id(p)] for p in decay] + [pid[id(p)] for p in no_decay]
    pfx = VP.G2_F[len('semsimula_'):].replace('fock_cfc_baoab_owt', 'fock_cfc_owt')
    ck = torch.load(g['CKPT_DIR'] / f'{pfx}_best.pt', map_location='cpu', weights_only=False)
    ost = ck['optimizer_state_dict']
    flat = [i for grp in ost['param_groups'] for i in grp['params']]
    assert len(flat) == len(order), (len(flat), len(order))
    params = dict(model.named_parameters())
    stateless = [n for i, n in zip(flat, order) if i not in ost['state']]
    for i, n in zip(flat, order):
        if i in ost['state']:
            assert ost['state'][i]['exp_avg'].shape == params[n].shape, (n, ost['state'][i]['exp_avg'].shape, params[n].shape)
    i0 = c6.index('GRAD_CLIP_OVERRIDES = {')
    i1 = c6.index("    GRAD_CLIP_OVERRIDES['pm_'] = POISSON_MODE_CLIP\n") + len("    GRAD_CLIP_OVERRIDES['pm_'] = POISSON_MODE_CLIP\n")
    ns = {'GRAD_CLIP_VPHI': 0.3, 'RELAX_FIELD_CLIP': None, 'POISSON_MODES': 0, 'POISSON_MODE_CLIP': 0.3}
    exec(c6[i0:i1], ns)
    from grad_clip_utils import GradClipConfig, assign_clip_group
    cfg = GradClipConfig(default_clip=1.0, overrides=ns['GRAD_CLIP_OVERRIDES'])
    groups = {}
    for i, n in zip(flat, order):
        if i not in ost['state']:
            continue
        st = ost['state'][i]
        r = (st['exp_avg'].abs() / (st['exp_avg_sq'].sqrt() + 1e-8)).flatten()
        key, thr = assign_clip_group(n, cfg)
        groups.setdefault((key, thr), []).append((n, r))
    print(f"\nC2. parameters with NO optimizer state (never received a gradient): {stateless}")
    res['C2_stateless'] = stateless
    print(f"C2. |m| / sqrt(v) per clip group, G2 best (step {ck['step']}); exact map, {len(order)} tensors checked by shape")
    print(f"   {'clip group':<38} {'thr':>5} {'tensors':>7} {'elements':>9} {'p10':>7} {'median':>7} {'p90':>7}")
    res['C2'] = {}
    for (key, thr), items in sorted(groups.items(), key=lambda kv: kv[0][1]):
        allr = torch.cat([r for _, r in items])
        q = [float(v) for v in np.quantile(allr.numpy(), (0.1, 0.5, 0.9))]
        res['C2'][key] = dict(threshold=thr, tensors=len(items), elements=int(allr.numel()), p10=q[0], median=q[1], p90=q[2])
        print(f"   {key[:38]:<38} {thr:5.2f} {len(items):7d} {allr.numel():9,d} {q[0]:7.3f} {q[1]:7.3f} {q[2]:7.3f}")
    rc = [r for (key, _), items in groups.items() for n, r in items if n == 'reverse_channel_scale'][0]
    print(f"   reverse_channel_scale elements: {[round(v, 4) for v in rc.tolist()]}")
    res['C2_reverse_channel_scale'] = rc.tolist()
    (G.OUT / 'c1_c2_g2.json').write_text(json.dumps(res, indent=1, default=str))
