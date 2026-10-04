"""SR5 temperature calibration on the F3.1 checkpoint (protocol SS5.9, SR5).

The O-step thermostat is FDT-locked, v <- c v + sqrt((T/m)(1 - c^2)) xi with
c = exp(-gamma dt), so at equilibrium each velocity component has variance
T/m and the thermal speed is ||v_th|| = sqrt(d T / m). A temperature is
therefore only meaningful against the speeds the trained model actually
uses. This records, at every O-step of an eval forward (no noise), the
velocity entering it and the token's semantic mass, and reports the
kinetic temperature per token,

    T_kin = m ||v||^2 / d,

so that T_r = r^2 * median(T_kin) gives an equilibrium thermal speed equal
to a fraction r of the typical speed. It also reports what a single O-step
injects at that T, sqrt(d (T/m)(1 - c^2)), relative to ||c v||, because the
stack runs 2 O-steps and never reaches equilibrium.

Usage: python3 calibrate_langevin_T.py OUT_DIR
"""
import contextlib, io, json, math, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import gradcheck_vphi_xi_paths as G

FOLDER = ('semsimula_fock_cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_'
          'vplive_xilive_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_'
          'idt4_lr0p0012_noattn')
c5b = [''.join(c['source']) for c in json.load(open(G.NB))['cells']
       if ''.join(c['source']).startswith('# == Cell 5b')][0]


def build():
    import os
    os.chdir(G.REPO / 'notebooks/conservative_arch/scaleup')
    g = {'__name__': '__main__'}
    c0 = G.cells['Cell 0:']
    for old, new in (('REVERSE_CHANNEL              = True', 'REVERSE_CHANNEL              = False'),
                     ("VPHI_GRAD_PATH         = 'default'", "VPHI_GRAD_PATH         = 'live'"),
                     ("XI_GRAD_PATH           = 'default'", "XI_GRAD_PATH           = 'live'")):
        assert c0.count(old) == 1, old
        c0 = c0.replace(old, new)
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(G.strip(c0), 'Cell0', 'exec'), g)
        exec(compile(G.strip(G.cells['Cell 1:']), 'Cell1', 'exec'), g)
        g['CKPT_DIR'] = Path.home() / 'Downloads' / FOLDER / 'checkpoints'
        g['RESULTS_DIR'] = G.OUT / 'results'; g['RESULTS_DIR'].mkdir(parents=True, exist_ok=True)
        g['DATA_DIR'] = G.OUT / 'data'; g['GDRIVE_ROOT'] = G.OUT
        exec(compile(G.strip(G.cells['Cell 1b:']), 'Cell1b', 'exec'), g)
        g['DATA_DIR'] = G.OUT / 'data'
        exec(compile(G.strip(G.cells['Cell 2:']), 'Cell2', 'exec'), g)
        exec('from data_module import get_batch', g)
        g['val_ids'] = np.load(G.VAL); g['train_ids'] = g['val_ids']
        exec(compile(G.strip(G.cells['Cell 4:']), 'Cell4', 'exec'), g)
        exec(compile(G.strip(G.cells['Cell 5:']), 'Cell5', 'exec'), g)
        g['PROBE_MAX_STEPS'] = g.get('PROBE_MAX_STEPS')
        exec(compile(G.strip(c5b), 'Cell5b', 'exec'), g)
    assert FOLDER.endswith(g['_variant_tag']), g['_variant_tag']
    model = g['model']
    ck = torch.load(g['CKPT_DIR'] / f"{g['CKPT_PREFIX']}_best.pt", map_location='cpu', weights_only=False)
    r = model.load_state_dict(ck['model_state_dict'], strict=False)
    assert not r.missing_keys and not r.unexpected_keys, r
    return model


if __name__ == '__main__':
    model = build()
    model.eval()
    cfg = model.cfg
    d, dt, L = cfg.d, cfg.dt, cfg.L
    gamma = float(model.gamma)
    c = math.exp(-gamma * dt)
    M = sys.modules['model_parf_multixi']
    orig = M.ou_step
    rec = []

    def spy(v, gam, dt_, m=None, T=0.0, training=True, noise_eval=False):
        rec.append((v.detach().clone(), m.detach().clone() if isinstance(m, torch.Tensor) else m))
        return orig(v, gam, dt_, m=m, T=T, training=training, noise_eval=noise_eval)

    M.ou_step = spy
    val = np.load(G.VAL)
    rng = np.random.default_rng(20260928)          # the 6b-13 token seed
    starts = rng.integers(0, len(val) - 513, size=8)
    per_layer = [[] for _ in range(L)]
    try:
        for b in range(0, 8, 2):
            x = torch.from_numpy(np.stack([val[s:s + 512] for s in starts[b:b + 2]]).astype(np.int64))
            rec.clear()
            with torch.enable_grad():
                model(x)
            assert len(rec) == L, len(rec)
            for ell, (v, m) in enumerate(rec):
                mm = m.expand(v.shape[:-1] + (1,)) if isinstance(m, torch.Tensor) else torch.full(v.shape[:-1] + (1,), float(m))
                per_layer[ell].append(torch.cat([(v.pow(2).sum(-1, keepdim=True) * mm / d).flatten()[:, None],
                                                 v.norm(dim=-1).flatten()[:, None],
                                                 mm.flatten()[:, None]], 1))
    finally:
        M.ou_step = orig

    out = []
    p = lambda t, q: float(torch.quantile(t, q))
    print(f"F3.1 (L={L}, dt={dt:g}, gamma={gamma:g}): O-step decay c = exp(-gamma dt) = {c:.4f}, "
          f"1 - c^2 = {1 - c * c:.4f}; d = {d}; 8 x 512 validation tokens, eval (no noise)\n")
    print("kinetic temperature T_kin = m ||v||^2 / d of the velocity entering each O-step:")
    allk = []
    for ell in range(L):
        a = torch.cat(per_layer[ell]); k, vn, mm = a[:, 0], a[:, 1], a[:, 2]
        allk.append(k)
        print(f"   layer {ell}: T_kin median {p(k, .5):.4e}  p10 {p(k, .1):.4e}  p90 {p(k, .9):.4e}   "
              f"||v|| median {p(vn, .5):.4f}   m median {p(mm, .5):.4f} (p10 {p(mm, .1):.4f}, p90 {p(mm, .9):.4f})")
    K = torch.cat(allk)
    Kmed = p(K, .5)
    print(f"   pooled over layers: T_kin median {Kmed:.4e}\n")
    print("SR5 temperatures, T_r = r^2 * median T_kin (equilibrium thermal speed = r x typical speed):")
    for r in (0.1, 0.3):
        T = r * r * Kmed
        inj = []
        for ell in range(L):
            a = torch.cat(per_layer[ell]); vn, mm = a[:, 1], a[:, 2]
            inj.append(torch.sqrt(d * (T / mm) * (1 - c * c)) / (c * vn).clamp(min=1e-12))
        inj = torch.cat(inj)
        print(f"   r = {r:.1f}:  LANGEVIN_T = {T:.3g}   one O-step injects {p(inj, .5):.3f} x ||c v|| "
              f"(median; p10 {p(inj, .1):.3f}, p90 {p(inj, .9):.3f})")
        out.append({'r': r, 'T': T, 'inject_median': p(inj, .5)})
    json.dump({'T_kin_median': Kmed, 'c': c, 'arms': out}, open(G.OUT / 'sr5_calibration.json', 'w'), indent=1)
