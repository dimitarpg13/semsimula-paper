"""Tier 1 check of vphi_grad_path / xi_grad_path (gradient-starvation programme).

For the no-exchange and conservative-only L=2 arms, each built through the
ladder notebook's own Cells 0-5b with the switches set, on the PARENT's
_best.pt and real validation tokens:

  1. tag and Cell 5b: the run gets its own Drive folder and the banner prints
  2. forward: logits and loss against the parent (eval, and train with the
     same Gumbel seed) -- must be bit-identical or float-reordering only
  3. one real layer step (_layer_step_ex, train mode): gradient from the
     last position into EARLIER tokens -- exactly 0 as trained, >0 live
  4. a full train-mode loss.backward() runs, and its wall time

plus a small random model on the Verlet integrator with a non-analytic V_theta,
where xi live needs the alias node (the force is differentiated w.r.t. h).

Usage: python3 verify_vphi_xi_grad_path.py OUT_DIR [--exchange]

--exchange runs the same checks on the exchange-field arms instead
('attention', 'attention_potential', and 'attention_potential' with
relax_grad_path='live'), the ones a live-gradient ladder needs.
"""
import contextlib, io, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import gradcheck_vphi_xi_paths as G   # notebook cells, arms, VAL, build helpers

_PFX = 'semsimula_fock_cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_'
_SFX = 'L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_'
EXCHANGE_ARMS = {  # name: (REVERSE_CHANNEL, folder, LADDER_MECHANISM, RELAX_GRAD_PATH)
    'attention': (True, _PFX + _SFX + 'attn', 'attention', 'default'),
    'attention_potential': (True, _PFX + _SFX + 'attnpot', 'attention_potential', 'default'),
    'attention_potential + rglive': (True, _PFX + 'rglive_' + _SFX + 'attnpot',
                                     'attention_potential', 'live'),
}
COMBOS = (('default', 'default'), ('live', 'default'), ('default', 'live'), ('live', 'live'))
c5b = [''.join(c['source']) for c in json.load(open(G.NB))['cells']
       if ''.join(c['source']).startswith('# == Cell 5b')][0]


def build(rc, folder, vp, xi, mech='none', rg='default'):
    import os
    os.chdir(G.REPO / 'notebooks/conservative_arch/scaleup')
    g = {'__name__': '__main__'}
    c0 = G.cells['Cell 0:']
    for old, new in (('REVERSE_CHANNEL              = True', f'REVERSE_CHANNEL              = {rc}'),
                     ("VPHI_GRAD_PATH         = 'default'", f"VPHI_GRAD_PATH         = {vp!r}"),
                     ("XI_GRAD_PATH           = 'default'", f"XI_GRAD_PATH           = {xi!r}"),
                     ("LADDER_MECHANISM = 'none'", f"LADDER_MECHANISM = {mech!r}"),
                     ("RELAX_GRAD_PATH        = 'default'", f"RELAX_GRAD_PATH        = {rg!r}")):
        assert c0.count(old) == 1, old
        c0 = c0.replace(old, new)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(G.strip(c0), 'Cell0', 'exec'), g)
        exec(compile(G.strip(G.cells['Cell 1:']), 'Cell1', 'exec'), g)
        g['CKPT_DIR'] = Path.home() / 'Downloads' / folder / 'checkpoints'      # the PARENT's weights
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
    model = g['model']
    assert model.cfg.vphi_grad_path == vp and model.cfg.xi_grad_path == xi
    assert model.cfg.force_relaxation == (mech if mech != 'none' else model.cfg.force_relaxation)
    # The PARENT's prefix (its folder name), not this tag's.
    parent = folder[len('semsimula_'):].replace('fock_cfc_baoab_owt', 'fock_cfc_owt')
    ck = torch.load(g['CKPT_DIR'] / f'{parent}_best.pt', map_location='cpu', weights_only=False)
    r = model.load_state_dict(ck['model_state_dict'], strict=False)
    assert not r.missing_keys and not r.unexpected_keys, r
    banner = [l.strip() for l in out.getvalue().splitlines() if 'SOURCE-GRADIENT PROBE' in l]
    return model, g['_variant_tag'], banner


def step_source_grad(model, x, layer):
    """Cotangent on the last position of one real layer step's output;
    norm of the gradient reaching the step input's EARLIER positions."""
    model.train()
    torch.manual_seed(1)
    with torch.enable_grad():
        _, traj = model._stack_forward(model._embed(x), x, return_trajectory=True)
        h = traj[layer].clone().float().requires_grad_(True)
        h_new, _ = model._layer_step_ex(h, h.detach().clone(), model.compute_mass(x),
                                        model.gamma, model.cfg.dt, layer_idx=layer)
        cot = torch.zeros_like(h_new)
        cot[:, -1] = torch.randn(h.shape[0], h.shape[2], generator=torch.Generator().manual_seed(0))
        gh, = torch.autograd.grad((cot * h_new).sum(), h)
    return gh[:, :-1].norm().item(), gh[:, -1].norm().item()


def arm_report(name, rc, folder, x, y, mech='none', rg='default'):
    print(f"\n== {name}")
    ref = None
    for vp, xi in COMBOS:
        model, tag, banner = build(rc, folder, vp, xi, mech, rg)
        model.eval()
        with torch.enable_grad():
            lo, l = model(x, y)
        model.train(); torch.manual_seed(7)
        t0 = time.time()
        lo_tr, l_tr = model(x, y)
        l_tr.backward()
        secs = time.time() - t0
        g_par = sum(p.grad.pow(2).sum().item() for p in model.parameters() if p.grad is not None) ** 0.5
        src = [step_source_grad(model, x, l_) for l_ in range(model.cfg.L)]
        if ref is None:
            ref = (lo.detach(), l.item(), lo_tr.detach(), l_tr.item())
        d_eval = (lo.detach() - ref[0]).abs().max().item()
        d_tr = (lo_tr.detach() - ref[2]).abs().max().item()
        print(f"   vphi={vp:7s} xi={xi:7s}  tag ...{tag[tag.find('cgqk'):][:60]}")
        print(f"      5b banner : {banner[0] if banner else '(none: ladder point)'}")
        print(f"      forward   : eval max|dlogit| {d_eval:.2e} (loss {l.item():.6f})   "
              f"train max|dlogit| {d_tr:.2e} (loss {l_tr.item():.6f})")
        print(f"      backward  : train fwd+bwd {secs:5.1f}s, |grad params| {g_par:.4e}")
        for l_, (e, s) in enumerate(src):
            print(f"      layer {l_} step: grad -> earlier tokens {e:.4e}   -> own token {s:.4e}")


def verlet_smoke():
    sys.path.insert(0, str(G.REPO / 'notebooks/conservative_arch/parf'))
    from model_parf_multixi import MultiXiPARFConfig, MultiXiPARFLM
    print("\n== small random model, integrator='verlet', autograd V_theta (alias path)")
    x = torch.randint(0, 257, (2, 16))
    ref = None
    for vp, xi in COMBOS:
        cfg = MultiXiPARFConfig(
            vocab_size=257, d=16, max_len=64, L=2, v_hidden=32, v_depth=2,
            v_phi_d_type=4, v_phi_d_angle=2, v_phi_phi_hidden=8, v_phi_theta_hidden=8,
            v_phi_mlp_hidden=16, mass_mode="global", top_k=8, score_head_hidden=8,
            xi_channels=4, xi_alpha_inits=[0.0, 0.5, 0.9, 0.99], xi_learnable=True,
            use_gathered_v_phi=True, vphi_grad_path=vp, xi_grad_path=xi)
        torch.manual_seed(0)
        m = MultiXiPARFLM(cfg)
        assert not m._use_analytic_vtheta()
        m.eval()
        with torch.enable_grad():
            lo, _ = m(x)
        ref = lo.detach() if ref is None else ref
        m.train()
        h = m._embed(x).detach().requires_grad_(True)
        h_new = m._layer_step(h, h.detach().clone(), m.compute_mass(x), m.gamma, cfg.dt, 0)
        cot = torch.zeros_like(h_new); cot[:, -1] = 1.0
        gh, = torch.autograd.grad((cot * h_new).sum(), h)
        print(f"   vphi={vp:7s} xi={xi:7s}  max|dlogit| {(lo.detach() - ref).abs().max().item():.2e}"
              f"   step grad -> earlier {gh[:, :-1].norm().item():.4e}")


if __name__ == '__main__':
    val = np.load(G.VAL)
    rng = np.random.default_rng(20261001)
    starts = rng.integers(0, len(val) - 513, size=2)
    x = torch.from_numpy(np.stack([val[s:s + 512] for s in starts]).astype(np.int64))
    y = torch.from_numpy(np.stack([val[s + 1:s + 513] for s in starts]).astype(np.int64))
    if '--exchange' in sys.argv:
        for name, (rc, folder, mech, rg) in EXCHANGE_ARMS.items():
            arm_report(name, rc, folder, x, y, mech, rg)
    else:
        verlet_smoke()
        for name, (rc, folder) in G.ARMS.items():
            arm_report(name, rc, folder, x, y)
