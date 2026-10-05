"""CB0: the free baseline for the CB series, on G2 (protocol SS5.11).

A. eta per token and layer: ||register increment|| / ||conservative step||,
   with the increment (dt^2/m) tanh(scale_l) warm Q_force and the conservative
   step d_cons = h_new - h exactly as _fock_layer_step defines them (the CB
   code path's own definitions, recomputed here because cb_stats keeps only
   the mean, p90 and max). Reported: quantiles, and the share of tokens above
   rho = 0.3 and 1.0, i.e. how hard each CB2 cap will bind.
B. Cell 6b-11 (E5, the reverse-channel slider) run as written, on CPU, with
   R11_NBATCH = 4 (8,192 tokens) and the figure redirected to OUT_DIR.

Usage: python3 cb0_g2.py OUT_DIR
"""
import contextlib, io, json, math, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import gradcheck_vphi_xi_paths as G
import verify_pm_switch as VP

if __name__ == '__main__':
    model, g, _ = VP.build(VP.G2_F, VP.CONFIGS['G2'][1])
    model.cfg.use_layer_checkpoint = False
    cfg = model.cfg
    rng = np.random.default_rng(20260928)
    val = np.load(G.VAL)
    batches = [tuple(torch.from_numpy(a) for a in g['get_batch'](val, 4, 512, rng)) for _ in range(8)]

    # ---- A. eta per token
    cur, etas = {}, {l: [] for l in range(cfg.L)}
    orig_step = model._fock_layer_step

    def step(h, h_prev, r, salience, m_b, gamma, dt, layer_idx=0, *a, **k):
        cur.update(h=h.detach(), m_b=m_b, dt=dt, layer=layer_idx)
        return orig_step(h, h_prev, r, salience, m_b, gamma, dt, layer_idx, *a, **k)

    def rev_hook(mod, args, out):
        h_new = args[0].detach()
        l = cur['layer']
        raw = (model.reverse_channel_scale[l] if model.reverse_channel_scale.numel() > 1
               else model.reverse_channel_scale)
        scale = torch.tanh(raw)
        if cfg.reverse_channel_warmup_steps > 0:
            scale = scale * (model.reverse_warmup_step.float() / float(cfg.reverse_channel_warmup_steps)).clamp(max=1.0)
        inc = (cur['dt'] * cur['dt'] / cur['m_b']) * scale * out.detach()
        d_cons = h_new - cur['h']
        etas[l].append((inc.norm(dim=-1) / d_cons.norm(dim=-1).clamp(min=1e-12)).flatten())

    model._fock_layer_step = step
    hk = model.reverse_ch.register_forward_hook(rev_hook)
    model.eval()
    try:
        for xb, _ in batches:
            with torch.enable_grad():
                model(xb)
    finally:
        # Restore by ASSIGNMENT, never del: the notebook installs the anisotropic
        # depth routing as an INSTANCE attribute _fock_layer_step, so a del would
        # strip it and leave the unrouted class method (a different model).
        hk.remove(); model._fock_layer_step = orig_step
    res = {'eta': {}}
    print(f"A. eta = ||register increment|| / ||conservative step||, G2 best, {8 * 4 * 512:,} tokens")
    print(f"   {'layer':>5} {'p10':>7} {'p50':>7} {'mean':>7} {'p90':>7} {'max':>8} {'> 0.3':>7} {'> 1.0':>7}")
    for l in range(cfg.L):
        e = torch.cat(etas[l])
        q = [float(e.quantile(p)) for p in (0.1, 0.5, 0.9)]
        r = dict(p10=q[0], p50=q[1], mean=float(e.mean()), p90=q[2], max=float(e.max()),
                 above_0p3=float((e > 0.3).float().mean()), above_1=float((e > 1.0).float().mean()))
        res['eta'][l] = r
        print(f"   {l:5d} {r['p10']:7.2f} {r['p50']:7.2f} {r['mean']:7.2f} {r['p90']:7.2f} {r['max']:8.2f} "
              f"{100 * r['above_0p3']:6.1f}% {100 * r['above_1']:6.1f}%")

    # ---- B. Cell 6b-11 as written
    nb = json.load(open(G.NB))
    c23 = [''.join(c['source']) for c in nb['cells'] if ''.join(c['source']).startswith('# == Cell 6b-11')][0]
    for old, new in (("R11_NBATCH = 8", "R11_NBATCH = 4"),
                     ("R11_FIG = CKPT_DIR / 'e5_reverse_channel_slider.png'",
                      f"R11_FIG = Path({str(G.OUT / 'e5_reverse_channel_slider_G2.png')!r})")):
        assert c23.count(old) == 1, old
        c23 = c23.replace(old, new)
    g['DEVICE'] = torch.device('cpu')
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(G.strip(c23), 'Cell6b11', 'exec'), g)
    print("\nB. Cell 6b-11 (reverse-channel slider), R11_NBATCH = 4:")
    print(buf.getvalue())
    (G.OUT / 'cb0_g2.json').write_text(json.dumps(res, indent=1))
