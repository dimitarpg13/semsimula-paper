"""Tier 0 of the gradient-starvation programme: do V_phi and xi starve the sources?

For the ladder's no-exchange ('none') and conservative-only ('none' + norc) L=2
arms, on their trained _best.pt and real validation tokens:

  G0.1  gradient reaching EARLIER tokens through the V_phi pair force, as trained
  G0.2  the same through the V_theta(xi, h) force via the xi context channels
  G0.3  the score-head configuration (the only other route V_phi could open)
  G0.4  RMS of the V_theta, V_phi and reverse-channel forces per layer

plus, for V_phi and xi, a LIVE-SOURCE COUNTERFACTUAL: the same forward force with
the sources left live for backprop (the Tier 1 construction), checked to be
forward-identical, reporting how much gradient it would carry.

Usage: python3 gradcheck_vphi_xi_paths.py OUT_DIR
Needs the ladder checkpoints mirrored under ~/Downloads/<drive folder>/checkpoints
and the validation cache named in VAL. Writes only under OUT_DIR.
"""
import contextlib, io, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
NB = REPO / 'notebooks/conservative_arch/scaleup/colab_fock_cfc_baoab_lowrank_depth_ladder_openwebtext_d384.ipynb'
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
VAL = Path.home() / 'Downloads/semsimula_fock_gamma_sweep_aniso_gaussian_fockreg_d384/data/openwebtext_val_2M.npy'
ARMS = {  # name: (REVERSE_CHANNEL, drive folder)
    'no-exchange': (True, 'semsimula_fock_cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_'
                          'L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn'),
    'conservative-only': (False, 'semsimula_fock_cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_'
                                 'norc_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn'),
}

cells = {}
for c in json.load(open(NB))['cells']:
    s = ''.join(c['source'])
    for key in ('Cell 0:', 'Cell 1:', 'Cell 1b:', 'Cell 2:', 'Cell 4:', 'Cell 5:'):
        if s.split('\n', 1)[0].startswith('# == ' + key):
            cells[key] = s
strip = lambda s: '\n'.join(l for l in s.splitlines() if not l.lstrip().startswith(('!', '%')))


def build(rc, folder):
    os.chdir(REPO / 'notebooks/conservative_arch/scaleup')
    g = {'__name__': '__main__'}
    c0 = cells['Cell 0:']
    assert c0.count("LADDER_MECHANISM = 'none'") == 1 and c0.count('REVERSE_CHANNEL              = True') == 1
    c0 = c0.replace('REVERSE_CHANNEL              = True', f'REVERSE_CHANNEL              = {rc}')
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(strip(c0), 'Cell0', 'exec'), g)
        exec(compile(strip(cells['Cell 1:']), 'Cell1', 'exec'), g)
        g['CKPT_DIR'] = Path.home() / 'Downloads' / folder / 'checkpoints'
        g['RESULTS_DIR'] = OUT / 'results'; g['RESULTS_DIR'].mkdir(parents=True, exist_ok=True)
        g['DATA_DIR'] = OUT / 'data'; g['GDRIVE_ROOT'] = OUT
        exec(compile(strip(cells['Cell 1b:']), 'Cell1b', 'exec'), g)
        g['DATA_DIR'] = OUT / 'data'
        exec(compile(strip(cells['Cell 2:']), 'Cell2', 'exec'), g)
        sys.path.insert(0, str(REPO / 'notebooks/conservative_arch'))
        exec('from data_module import get_batch', g)
        g['val_ids'] = np.load(VAL); g['train_ids'] = g['val_ids']
        exec(compile(strip(cells['Cell 4:']), 'Cell4', 'exec'), g)
        exec(compile(strip(cells['Cell 5:']), 'Cell5', 'exec'), g)
    assert folder.endswith(g['_variant_tag']), (folder, g['_variant_tag'])
    model = g['model']
    ck = torch.load(g['CKPT_DIR'] / f"{g['CKPT_PREFIX']}_best.pt", map_location='cpu', weights_only=False)
    r = model.load_state_dict(ck['model_state_dict'], strict=False)
    assert not r.missing_keys and not r.unexpected_keys, r
    model.eval()
    return model, ck


def pair_potential_live(model, h_tgt, h_src, layer_idx):
    """U_pair exactly as _pair_potential's sparse path, but with a separate LIVE source tensor.
    Differentiating w.r.t. h_tgt only gives the same forward force; h_src carries backprop."""
    cfg = model.cfg
    B, T, d = h_tgt.shape
    src_score = h_src.detach() if cfg.score_head_use_detached_h_src else h_src
    pi = model.score_head(h_tgt, src_score)
    causal = model._pair_mask_for(T, h_tgt.device)
    assert cfg.use_gathered_v_phi and model.V_attn is None
    idx, m_g = model._sparse_topk_indices(pi, causal, T)
    h_src_g = h_src.unsqueeze(1).expand(-1, T, -1, -1).gather(2, idx.unsqueeze(-1).expand(-1, -1, -1, d))
    U = (model.V_phi.forward_gathered(h_tgt, h_src_g) * m_g).sum()
    s_ell = model.per_layer_scale(layer_idx)
    return U * s_ell if s_ell is not None else U


def probe_arm(name, rc, folder, x):
    t0 = time.time()
    model, ck = build(rc, folder)
    cfg = model.cfg
    rows = []
    rev = {}
    if model.reverse_ch is not None:                        # capture the reverse-channel force per layer
        orig = model.reverse_ch.forward
        calls = []
        def spy(*a, **k):
            out = orig(*a, **k); calls.append(out.detach()); return out
        model.reverse_ch.forward = spy
    with torch.enable_grad():
        _, traj = model._stack_forward(model._embed(x), x, return_trajectory=True)
    if model.reverse_ch is not None:
        model.reverse_ch.forward = orig
        for l, q in enumerate(calls[:cfg.L]):
            scale = torch.tanh(model.reverse_channel_scale[l] if model.reverse_channel_scale.numel() > 1
                               else model.reverse_channel_scale).item()
            rev[l] = scale * q.pow(2).mean().sqrt().item()
    gen = torch.Generator().manual_seed(0)
    for l in range(cfg.L):
        h = traj[l].clone().float().requires_grad_(True)
        B, T, d = h.shape
        cot = torch.zeros(B, T, d); cot[:, -1] = torch.randn(B, d, generator=gen)

        # ---- G0.1: V_phi as trained ----
        xis_tr = model.xi_module(h.detach() if cfg.causal_force else h)
        U = model._pair_potential(h, l, xis=xis_tr)
        gU, = torch.autograd.grad(U, h, create_graph=True)
        F_phi = -gU
        g_phi, = torch.autograd.grad((cot * F_phi).sum(), h, allow_unused=True)
        g_phi = torch.zeros_like(h) if g_phi is None else g_phi
        # ---- V_phi live-source counterfactual ----
        h_t = h.detach().clone().requires_grad_(True)
        h_s = h.detach().clone().requires_grad_(True)
        Ul = pair_potential_live(model, h_t, h_s, l)
        gUl, = torch.autograd.grad(Ul, h_t, create_graph=True)
        F_phi_live = -gUl
        g_s, = torch.autograd.grad((cot * F_phi_live).sum(), h_s, allow_unused=True)
        g_s = torch.zeros_like(h) if g_s is None else g_s

        # ---- G0.2: V_theta(xi, h) force via xi, as trained ----
        h2 = h.detach().clone().requires_grad_(True)
        xis_tr2 = model.xi_module(h2.detach() if cfg.causal_force else h2)
        F_th = -model.V_theta.analytical_grad(xis_tr2, h2)
        g_th, = torch.autograd.grad((cot * F_th).sum(), h2, allow_unused=True)
        g_th = torch.zeros_like(h) if g_th is None else g_th
        # ---- xi live-source counterfactual: xi from a live copy, force still the partial in h ----
        h3 = h.detach().clone().requires_grad_(True)
        h_xi = h.detach().clone().requires_grad_(True)
        F_th_live = -model.V_theta.analytical_grad(model.xi_module(h_xi), h3)
        g_xi, = torch.autograd.grad((cot * F_th_live).sum(), h_xi, allow_unused=True)
        g_xi = torch.zeros_like(h) if g_xi is None else g_xi

        rows.append(dict(
            layer=l,
            h_rms=h.detach().pow(2).mean().sqrt().item(),
            F_theta_rms=F_th.detach().pow(2).mean().sqrt().item(),
            F_phi_rms=F_phi.detach().pow(2).mean().sqrt().item(),
            F_rev_rms=rev.get(l),
            vphi_trained_to_earlier=g_phi[:, :-1].norm().item(),
            vphi_trained_to_self=g_phi[:, -1].norm().item(),
            vphi_live_forward_diff=(F_phi.detach() - F_phi_live.detach()).abs().max().item(),
            vphi_live_to_earlier=g_s[:, :-1].norm().item(),
            xi_trained_to_earlier=g_th[:, :-1].norm().item(),
            xi_live_forward_diff=(F_th.detach() - F_th_live.detach()).abs().max().item(),
            xi_live_to_earlier=g_xi[:, :-1].norm().item(),
            xi_live_to_self_via_xi=g_xi[:, -1].norm().item()))
    flags = dict(causal_force=cfg.causal_force,
                 score_head_use_detached_h_src=cfg.score_head_use_detached_h_src,
                 use_gathered_v_phi=cfg.use_gathered_v_phi, reverse_channel=rc)
    return dict(arm=name, step=ck['step'], ppl=ck['val_ppl'], flags=flags, rows=rows, secs=round(time.time() - t0))


if __name__ == '__main__':
    val = np.load(VAL)
    rng = np.random.default_rng(20260930)
    starts = rng.integers(0, len(val) - 513, size=2)
    x = torch.from_numpy(np.stack([val[s:s + 512] for s in starts]).astype(np.int64))
    report = []
    for name, (rc, folder) in ARMS.items():
        rep = probe_arm(name, rc, folder, x)
        report.append(rep)
        print(f"\n== {name}  (_best.pt step {rep['step']:,}, PPL {rep['ppl']:.2f})  flags {rep['flags']}  [{rep['secs']}s]")
        print(f"{'layer':>5} {'|h|':>7} {'F_theta':>9} {'F_phi':>9} {'F_rev':>9} | "
              f"{'Vphi->earlier':>13} {'Vphi->self':>11} {'LIVE dF':>9} {'LIVE->earlier':>13} | "
              f"{'xi->earlier':>11} {'LIVE dF':>9} {'LIVE->earlier':>13}")
        for r in rep['rows']:
            fr = f"{r['F_rev_rms']:9.4f}" if r['F_rev_rms'] is not None else f"{'—':>9}"
            print(f"{r['layer']:>5} {r['h_rms']:7.3f} {r['F_theta_rms']:9.4f} {r['F_phi_rms']:9.4f} {fr} | "
                  f"{r['vphi_trained_to_earlier']:13.4e} {r['vphi_trained_to_self']:11.4e} "
                  f"{r['vphi_live_forward_diff']:9.1e} {r['vphi_live_to_earlier']:13.4e} | "
                  f"{r['xi_trained_to_earlier']:11.4e} {r['xi_live_forward_diff']:9.1e} {r['xi_live_to_earlier']:13.4e}")
    json.dump(report, open(OUT / 'gradcheck_vphi_xi.json', 'w'), indent=1)
