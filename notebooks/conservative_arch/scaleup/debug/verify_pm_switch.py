"""PM1 switch check: bosonic Poisson-mode registers (protocol SS5.15).

OFF (POISSON_MODES = 0) must be bit-identical to the committed code on every
configuration that may run while it is pushed: G2, F3.1 and G3' (the arm
training now). Built through the ladder notebook's own Cells 0-5b on fixed
trained weights and real validation tokens; records eval logits, train-mode
logits (same seed) and every parameter gradient.

  python3 verify_pm_switch.py OUT_DIR --dump      # with HEAD files in place
  python3 verify_pm_switch.py OUT_DIR             # working copy: compare + ON checks

ON (F3.1's Cell 0 plus POISSON_MODES = 64), on F3.1's trained weights:
  1. tag: F3.1's tag plus 'pm64'; Cell 5b banner; the only new parameters are pm_*
  2. depths at 0: logits bit-identical to the model without modes
  3. depths set non-zero: the force equals -autograd.grad of the potential at
     fixed occupations (the explicit formula is the gradient in h_t)
  4. occupations are the Poisson means: a Monte Carlo immigration-death
     simulation (Poisson(E) arrivals per token, binomial survival lambda)
     reproduces phi
  5. causality with non-zero depths: replacing tokens at t >= 256 leaves
     logits at t < 256 unchanged
  6. gradients reach pm_depth at depth 0, and every pm_ parameter once non-zero
  7. clip groups: 'pm_' captures exactly the four pm_ parameters
"""
import contextlib, io, json, os, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import gradcheck_vphi_xi_paths as G
import verify_vphi_xi_grad_path as V

_P = 'semsimula_fock_cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_'
_S = '_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_'
G2_F = _P + 'vplive_xilive_L2probe' + _S + 'noattn'
F31_F = _P + 'norc_vplive_xilive_L2probe' + _S + 'noattn'
G3_F = _P + 'rglive_vplive_xilive_L2probe' + _S + 'attnpot'
LIVE = (("VPHI_GRAD_PATH         = 'default'", "VPHI_GRAD_PATH         = 'live'"),
        ("XI_GRAD_PATH           = 'default'", "XI_GRAD_PATH           = 'live'"))
CONFIGS = {  # name: (weights folder, Cell 0 replacements)
    'G2': (G2_F, LIVE),
    'F3.1': (F31_F, LIVE + (('REVERSE_CHANNEL              = True', 'REVERSE_CHANNEL              = False'),)),
    "G3'": (G3_F, LIVE + (("LADDER_MECHANISM = 'none'", "LADDER_MECHANISM = 'attention_potential'"),
                          ("RELAX_GRAD_PATH        = 'default'", "RELAX_GRAD_PATH        = 'live'"),
                          ("RELAX_ATTN_QK_NORM          = False", "RELAX_ATTN_QK_NORM          = True"),
                          ("RELAX_FIELD_CLIP            = None", "RELAX_FIELD_CLIP            = 0.3"))),
}
PM_ON = (("POISSON_MODES        = 0", "POISSON_MODES        = 64"),)


def build(folder, repl):
    c0 = G.cells['Cell 0:']
    for old, new in repl:
        assert c0.count(old) == 1, old
        c0 = c0.replace(old, new)
    os.chdir(G.REPO / 'notebooks/conservative_arch/scaleup')
    g = {'__name__': '__main__'}
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(G.strip(c0), 'Cell0', 'exec'), g)
        exec(compile(G.strip(G.cells['Cell 1:']), 'Cell1', 'exec'), g)
        g['CKPT_DIR'] = Path.home() / 'Downloads' / folder / 'checkpoints'
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
        exec(compile(G.strip(V.c5b), 'Cell5b', 'exec'), g)
    model = g['model']
    pfx = folder[len('semsimula_'):].replace('fock_cfc_baoab_owt', 'fock_cfc_owt')
    ck = torch.load(g['CKPT_DIR'] / f'{pfx}_best.pt', map_location='cpu', weights_only=False)
    r = model.load_state_dict(ck['model_state_dict'], strict=False)
    assert not r.unexpected_keys, r.unexpected_keys
    for k in r.missing_keys:      # new parameters: deterministic fill so HEAD and working copy agree
        assert k in ('relax_field.logit_scale',) or k.startswith('pm_'), k
    with torch.no_grad():
        if getattr(model, 'relax_field', None) is not None and getattr(model.relax_field, 'logit_scale', None) is not None:
            model.relax_field.logit_scale.fill_(np.log(1 / 0.07))
    return model, g, out.getvalue()


def record(model, x, y):
    model.eval()
    with torch.enable_grad():
        ev, _ = model(x)
    model.train(); torch.manual_seed(7)
    tr, loss = model(x, y)
    loss.backward()
    grads = {k: p.grad.detach().clone() for k, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    return {'eval': ev.detach(), 'train': tr.detach(), 'loss': loss.item(), 'grads': grads}


def tokens(n_seq=2, T=256, seed=20261005):
    val = np.load(G.VAL)
    st = np.random.default_rng(seed).integers(0, len(val) - T - 1, size=n_seq)
    x = torch.from_numpy(np.stack([val[s:s + T] for s in st]).astype(np.int64))
    y = torch.from_numpy(np.stack([val[s + 1:s + T + 1] for s in st]).astype(np.int64))
    return x, y


if __name__ == '__main__':
    x, y = tokens()
    ref = G.OUT / 'pm_head_ref.pt'
    if '--dump' in sys.argv:
        recs = {}
        for name, (folder, repl) in CONFIGS.items():
            model, g, _ = build(folder, repl)
            recs[name] = record(model, x, y)
            print(f"  dumped {name}: tag ...{g['_variant_tag'][g['_variant_tag'].find('cgqk'):]}  loss {recs[name]['loss']:.6f}")
        torch.save(recs, ref)
        sys.exit(0)

    ok = []
    h = torch.load(ref)
    print("OFF (POISSON_MODES = 0): HEAD vs working copy, fixed trained weights")
    for name, (folder, repl) in CONFIGS.items():
        model, g, _ = build(folder, repl)
        rec = record(model, x, y)
        assert set(h[name]['grads']) == set(rec['grads']), name
        de = (h[name]['eval'] - rec['eval']).abs().max().item()
        dt = (h[name]['train'] - rec['train']).abs().max().item()
        dg = max((h[name]['grads'][k] - rec['grads'][k]).abs().max().item() for k in rec['grads'])
        ok.append(de == dt == dg == 0)
        print(f"  {name:5s} eval {de:.1e}  train {dt:.1e}  grads {dg:.1e}  ({len(rec['grads'])} params)  -> "
              f"{'IDENTICAL' if ok[-1] else 'DIFFERS'}")

    print("\nON (F3.1's Cell 0 + POISSON_MODES = 64), on F3.1's trained weights")
    off_model, g_off, _ = build(F31_F, CONFIGS['F3.1'][1])
    model, g, log = build(F31_F, CONFIGS['F3.1'][1] + PM_ON)
    tag, tag_off = g['_variant_tag'], g_off['_variant_tag']
    new = sorted(set(dict(model.named_parameters())) - set(dict(off_model.named_parameters())))
    ok.append(tag == tag_off.replace('_L2probe', '_pm64_L2probe') and new ==
              ['pm_depth', 'pm_log_kappa2', 'pm_logit_lambda', 'pm_mu'] and 'POISSON-MODE REGISTERS' in log)
    print(f"1. tag ...{tag[tag.find('cgqk'):]}\n   new parameters {new}; banner printed: {'POISSON-MODE' in log} -> {ok[-1]}")
    hl = (-np.log(2) / torch.nn.functional.logsigmoid(model.pm_logit_lambda.detach())).numpy()
    print(f"   half-lives {hl.min():.1f}-{hl.max():.1f} tokens; |mu| mean {model.pm_mu.norm(dim=-1).mean():.1f} "
          f"(sqrt d = {np.sqrt(model.cfg.d):.1f})")

    a = record(model, x, y); b = record(off_model, x, y)
    d0 = max((a['eval'] - b['eval']).abs().max().item(), (a['train'] - b['train']).abs().max().item())
    ok.append(d0 == 0.0 and a['grads'].get('pm_depth') is not None and a['grads']['pm_depth'].abs().sum() > 0)
    print(f"2. depths 0: max |dlogit| vs no modes {d0:.1e}; grad reaches pm_depth "
          f"(norm {a['grads']['pm_depth'].norm():.2e}) -> {ok[-1]}")

    torch.manual_seed(3)
    with torch.no_grad():
        model.pm_depth.normal_(0, 0.5)
    hh = torch.randn(2, 64, model.cfg.d) * 0.3 + model.pm_mu[:1].detach() * 0.8
    hh.requires_grad_(True)
    E, phi = model.poisson_mode_occupation(hh)
    k2 = model.pm_log_kappa2.exp()
    d2 = ((hh[..., None, :] - model.pm_mu) ** 2).sum(-1)
    U = -(phi.detach() * model.pm_depth[1] * torch.exp(-k2 * d2)).sum()
    gU, = torch.autograd.grad(U, hh)
    f = model.poisson_mode_force(hh, 1)
    rel = ((f + gU).norm() / gU.norm()).item()
    ok.append(rel < 1e-5)
    print(f"3. force vs -grad U at fixed phi: relative error {rel:.1e} (|F| {f.norm():.3e}) -> {ok[-1]}")

    rng = np.random.default_rng(0)
    Es = E[0].detach().numpy(); lam = torch.sigmoid(model.pm_logit_lambda).detach().numpy()
    n = np.zeros((20000, Es.shape[1])); means = np.zeros_like(Es)
    for t in range(Es.shape[0]):
        means[t] = n.mean(0)                       # occupation seen by token t: arrivals from s < t
        n = rng.binomial(n.astype(int), lam) + rng.poisson(Es[t], size=n.shape)
    err = np.abs(means - phi[0].detach().numpy()).max() / (phi[0].detach().numpy().max())
    disp = (n.var(0) / n.mean(0).clip(1e-9))
    ok.append(err < 0.02 and abs(np.median(disp) - 1) < 0.05)
    print(f"4. Monte Carlo immigration-death (20,000 runs): max |mean - phi| / max phi = {err:.4f}; "
          f"variance/mean median {np.median(disp):.3f} (Poisson: 1) -> {ok[-1]}")

    model.eval()
    x2 = x.clone(); x2[:, 128:] = torch.from_numpy(rng.integers(0, 50257, size=(2, x.shape[1] - 128)))
    with torch.enable_grad():
        l1, _ = model(x); l2, _ = model(x2)
    dc = (l1[:, :128] - l2[:, :128]).abs().max().item()
    ok.append(dc == 0.0)
    print(f"5. causality, depths non-zero: tokens replaced at t >= 128, max |dlogit| at t < 128 = {dc:.1e} -> {ok[-1]}")

    r = record(model, x, y)
    gn = {k: r['grads'][k].norm().item() for k in r['grads'] if k.startswith('pm_')}
    ok.append(len(gn) == 4 and all(v > 0 for v in gn.values()))
    print("6. grads with depths non-zero: " + ', '.join(f'{k} {v:.2e}' for k, v in gn.items()) + f" -> {ok[-1]}")

    c6 = [''.join(c['source']) for c in json.load(open(G.NB))['cells']
          if ''.join(c['source']).startswith('# == Cell 6: Training loop')][0]
    i0 = c6.index('GRAD_CLIP_OVERRIDES = {')
    end = "    GRAD_CLIP_OVERRIDES['pm_'] = POISSON_MODE_CLIP\n"
    i1 = c6.index(end) + len(end)
    from grad_clip_utils import GradClipConfig, assign_clip_group
    ns = {'GRAD_CLIP_VPHI': 0.3, 'RELAX_FIELD_CLIP': None, 'POISSON_MODES': 64, 'POISSON_MODE_CLIP': 0.3}
    exec(c6[i0:i1], ns)
    cfg = GradClipConfig(default_clip=1.0, overrides=ns['GRAD_CLIP_OVERRIDES'])
    grp = {n_: assign_clip_group(n_, cfg)[0] for n_, _ in model.named_parameters()}
    caught = sorted(n_ for n_, gk in grp.items() if gk == 'override:pm_')
    ok.append(caught == ['pm_depth', 'pm_log_kappa2', 'pm_logit_lambda', 'pm_mu'])
    print(f"7. clip group 'pm_' catches {caught} -> {ok[-1]}")
    print(f"\n-> {'ALL PASS' if all(ok) else 'FAIL ' + str(ok)}")
