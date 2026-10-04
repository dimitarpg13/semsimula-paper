"""FO series, Stage 0, model side (protocol SS5.13, M1-M2).

Models, each on its own corpus's validation tokens (eval mode, 2 x 512 tokens,
fixed seed):
  TS-SO   TinyStories second-order anchor (Verlet, d=256, L=8, gamma*=0.3; 9.04)
  TS-FO   TinyStories Fock-G1 (first order, same architecture; 8.95)
  OWT-F31 F3.1: conservative-only, live gradients (CfC low-rank, d=384, L=2)
  OWT-L4  the L=4 live-gradient Fock arm (CfC low-rank, d=384, L=4)
built through each notebook's own cells.

M1  anharmonic fraction per token and layer,
        eps = ||f(h+dh) - f(h) + H(h) dh|| / ||f(h+dh) - f(h)||,
    f = -grad_h V_theta(xi, h) with xi frozen at the layer's own value, H its
    Hessian (one Hessian-vector product), dh the layer's actual step (LayerNorm
    and every other force included). eps << 1: the force is linear across the
    step, the regime in which second order is absorbable (the note's
    anharmonicity gate).
M2  inertial share of each layer step,
        ||Phi(h, v) - Phi(h, 0)|| / ||Phi(h, v) - h||,
    re-running the stack with the incoming velocity reset at that layer only.

Usage: python3 stage0_models.py OUT_DIR
"""
import contextlib, io, json, math, sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
REPO = Path(__file__).resolve().parents[4]
CA = REPO / 'notebooks/conservative_arch'
DL = Path.home() / 'Downloads'
strip = lambda s: '\n'.join(l for l in s.splitlines() if not l.lstrip().startswith(('!', '%')))


def cells(nb):
    return [''.join(c['source']) for c in json.load(open(nb))['cells'] if c['cell_type'] == 'code']


def build_ts(kind):
    for sub in ['', 'parf', 'multixi', 'scaleup', 'sarf_mass_variant', 'energetic_minima']:
        p = str(CA / sub) if sub else str(CA)
        if p not in sys.path:
            sys.path.insert(0, p)
    if kind == 'SO':
        cs = cells(CA / 'scaleup/colab_fock_aniso_gaussian_fockreg_tinystories.ipynb')
        use = [0, 2, 4, 5]          # Cell 0, Cell 2 (imports), Cell 4 (V_theta), Cell 5 (builder)
        ck = DL / 'semsimula_fock_aniso_gaussian_fockreg_tinystories/results/seed0_gamma=0.3/ckpt_best.pt'
    else:
        cs = cells(CA / 'first_order_ablation/colab_fock_g1_aniso_gaussian_fockreg_tinystories.ipynb')
        use = [0, 2, 4, 5, 6]       # ... + Cell 5b (Fock-G1 class)
        ck = DL / 'semsimula_fock_g1_aniso_gaussian_fockreg_tinystories/results/seed0/ckpt_best.pt'
    g = {'__name__': '__main__', 'CA_DIR': CA, 'RESULTS_DIR': OUT / 'ts_results', 'REPO_ROOT': REPO}
    (OUT / 'ts_results').mkdir(exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        for i in use:
            src = strip(cs[i])
            if i == 5 and kind == 'FO':
                src = src.split('# ── Smoke test')[0]
            exec(compile(src, f'TS{kind}-cell{i}', 'exec'), g)
            g['DEVICE'] = 'cpu'
            if i == 4:
                # The notebook's routing patch targets the August _layer_step
                # signature (layer_idx, x_tokens); the current one is
                # (h, h_prev, m_b, gamma, dt, layer_idx). Same semantics: set
                # the active depth code, then take the step unchanged.
                def install_aniso_depth_routing(model):
                    import types
                    if not hasattr(model.V_theta, 'set_active_layer'):
                        return
                    orig = model._layer_step.__func__
                    def _patched(self, h, h_prev, m_b, gamma, dt, layer_idx=0):
                        self.V_theta.set_active_layer(layer_idx)
                        return orig(self, h, h_prev, m_b, gamma, dt, layer_idx)
                    model._layer_step = types.MethodType(_patched, model)
                g['install_aniso_depth_routing'] = install_aniso_depth_routing
    model = g['model']
    sd = torch.load(ck, map_location='cpu', weights_only=False)
    r = model.load_state_dict(sd['model_state_dict'], strict=False)
    assert not r.missing_keys and not r.unexpected_keys, r
    val = np.load(DL / 'semsimula_fock_g1_aniso_gaussian_fockreg_tinystories/data/'
                       'tinystories_gpt2_1files_5000000toks.npz')['val']
    return model, val, float(sd.get('val_ppl', float('nan')))


def build_owt(folder, rc, L):
    sys.path.insert(0, str(REPO / 'notebooks/conservative_arch/scaleup/debug'))
    sys.argv = sys.argv[:1] + [str(OUT / 'owt_scratch')]
    import gradcheck_vphi_xi_paths as G
    import os
    c5b = [c for c in cells(G.NB) if c.startswith('# == Cell 5b')][0]
    os.chdir(REPO / 'notebooks/conservative_arch/scaleup')
    g = {'__name__': '__main__'}
    c0 = G.cells['Cell 0:']
    for old, new in (('REVERSE_CHANNEL              = True', f'REVERSE_CHANNEL              = {rc}'),
                     ("VPHI_GRAD_PATH         = 'default'", "VPHI_GRAD_PATH         = 'live'"),
                     ("XI_GRAD_PATH           = 'default'", "XI_GRAD_PATH           = 'live'"),
                     ('LADDER_L         = 2 ', f'LADDER_L         = {L} ')):
        assert c0.count(old) == 1, old
        c0 = c0.replace(old, new)
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(G.strip(c0), 'Cell0', 'exec'), g)
        exec(compile(G.strip(G.cells['Cell 1:']), 'Cell1', 'exec'), g)
        g['CKPT_DIR'] = DL / folder / 'checkpoints'
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
    assert folder.endswith(g['_variant_tag']), (folder, g['_variant_tag'])
    model = g['model']
    sd = torch.load(g['CKPT_DIR'] / f"{g['CKPT_PREFIX']}_best.pt", map_location='cpu', weights_only=False)
    r = model.load_state_dict(sd['model_state_dict'], strict=False)
    assert not r.missing_keys and not r.unexpected_keys, r
    return model, np.load(G.VAL), float('nan')


def trajectory(model, x):
    model.eval()
    with torch.enable_grad():
        _, traj = model._stack_forward(model._embed(x), x, return_trajectory=True)
    return [t.float() for t in traj]


def m1(model, traj):
    out = []
    vt = model.V_theta
    lnv = getattr(model, 'ln_before_v', None)
    for ell in range(len(traj) - 1):
        h, dh = traj[ell], traj[ell + 1] - traj[ell]
        with torch.no_grad():
            xis = model.xi_module(h)
        if hasattr(vt, 'set_active_layer'):
            vt.set_active_layer(ell)
        V = lambda z: vt(xis, lnv(z) if lnv is not None else z).sum()
        z = h.clone().requires_grad_(True)
        g1, = torch.autograd.grad(V(z), z, create_graph=True)
        hdh, = torch.autograd.grad((g1 * dh).sum(), z)
        z2 = (h + dh).clone().requires_grad_(True)
        g2, = torch.autograd.grad(V(z2), z2)
        df = -(g2 - g1.detach())                 # f(h+dh) - f(h)
        eps = (df + hdh).norm(dim=-1) / df.norm(dim=-1).clamp(min=1e-12)
        rel = df.norm(dim=-1) / g1.detach().norm(dim=-1).clamp(min=1e-12)
        out.append({'layer': ell, 'eps': eps.flatten(), 'force_change': rel.flatten(),
                    'step': dh.norm(dim=-1).flatten()})
    return out


def m2(model, x, traj):
    orig = model.__dict__.get('_fock_layer_step')
    bound = model._fock_layer_step
    res = []
    for target in range(1, len(traj) - 1):
        def patched(*a, **k):
            a = list(a)
            ell = k.get('layer_idx', a[7] if len(a) > 7 else None)
            if ell == target:
                a[1] = a[0]                        # h_prev := h at this layer only
            return bound(*a, **k)
        model._fock_layer_step = patched
        try:
            tr = trajectory(model, x)
        finally:
            if orig is None:
                del model._fock_layer_step
            else:
                model._fock_layer_step = orig
        num = (traj[target + 1] - tr[target + 1]).norm(dim=-1)
        den = (traj[target + 1] - traj[target]).norm(dim=-1).clamp(min=1e-12)
        res.append({'layer': target, 'share': (num / den).flatten()})
    return res


if __name__ == '__main__':
    q = lambda t, p: float(torch.quantile(t, p))
    summary = {}
    specs = [
        ('TS-SO', lambda: build_ts('SO')),
        ('TS-FO', lambda: build_ts('FO')),
        ('OWT-F31', lambda: build_owt('semsimula_fock_cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn', False, 2)),
        ('OWT-L4', lambda: build_owt('semsimula_fock_cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_vplive_xilive_L4probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt2_lr0p0012_noattn', True, 4)),
    ]
    print('FO Stage 0, model side (protocol SS5.13): each model on its own corpus, eval, 2 x 512 tokens\n')
    for name, mk in specs:
        torch.manual_seed(0)
        model, val, ppl = mk()
        rng = np.random.default_rng(20261004)
        st = rng.integers(0, len(val) - 513, size=2)
        x = torch.from_numpy(np.stack([val[s:s + 512] for s in st]).astype(np.int64))
        # Reproduction guard: the model must score its own validation tokens
        # as its checkpoint did, or the readings below are not of that model.
        model.eval(); tot = 0.0
        r2 = np.random.default_rng(7)
        for _ in range(6):
            s2 = r2.integers(0, len(val) - 513, size=2)
            xb = torch.from_numpy(np.stack([val[s:s + 512] for s in s2]).astype(np.int64))
            yb = torch.from_numpy(np.stack([val[s + 1:s + 513] for s in s2]).astype(np.int64))
            with torch.enable_grad():
                _, l = model(xb, yb)
            tot += l.item() / 6
        ppl_now = math.exp(tot)
        traj = trajectory(model, x)
        a = m1(model, traj)
        b = m2(model, x, traj) if name != 'TS-FO' else []
        cfg = model.cfg
        print(f"== {name}: d={cfg.d} L={cfg.L} dt={cfg.dt:g} integrator={getattr(cfg, 'integrator', 'verlet')}"
              + f"  val PPL here {ppl_now:.2f} (6 x 2 x 512)"
              + (f", checkpoint {ppl:.2f}" if not math.isnan(ppl) else ''))
        print('   M1 anharmonic fraction eps (median / p10 / p90), force change ||df||/||f||, step ||dh||:')
        for r in a:
            print(f"     layer {r['layer']}: eps {q(r['eps'], .5):.3f} / {q(r['eps'], .1):.3f} / {q(r['eps'], .9):.3f}"
                  f"   ||df||/||f|| {q(r['force_change'], .5):.3f}   ||dh|| {q(r['step'], .5):.3f}")
        allE = torch.cat([r['eps'] for r in a])
        print(f"     pooled: eps median {q(allE, .5):.3f}, p90 {q(allE, .9):.3f}")
        if b:
            print('   M2 inertial share (median / p10 / p90):')
            for r in b:
                print(f"     layer {r['layer']}: {q(r['share'], .5):.3f} / {q(r['share'], .1):.3f} / {q(r['share'], .9):.3f}")
        elif name == 'TS-FO':
            print('   M2: 0 by construction (first order: no velocity crosses a layer)')
        summary[name] = {'val_ppl_here': ppl_now, 'eps_median_by_layer': [q(r['eps'], .5) for r in a],
                         'eps_p90_by_layer': [q(r['eps'], .9) for r in a],
                         'eps_pooled_median': q(allE, .5), 'eps_pooled_p90': q(allE, .9),
                         'inertial_share_median_by_layer': [q(r['share'], .5) for r in b]}
        print()
        del model
    json.dump(summary, open(OUT / 'stage0_models.json', 'w'), indent=1)
