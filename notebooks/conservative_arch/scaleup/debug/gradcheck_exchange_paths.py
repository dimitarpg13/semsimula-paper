"""Does the exchange field send any gradient into the hidden states?

For the 'attention' and 'attention_potential' L=2 arms, on their trained _best.pt
and real validation tokens: compute the exchange force exactly as the model does,
back-propagate a cotangent from the LAST position only, and report how much
gradient reaches h -- at earlier positions (the source / value path) and at the
last position itself (the query path). Controls: the force's own magnitude (a
zero gradient from a zero force would mean nothing) and the gradient reaching the
routing weights.

Model built by the notebook's own Cells 0, 1, 1b, 2, 4, 5 with LADDER_MECHANISM
substituted; nothing written outside the scratchpad.
"""
import json, sys, os, time
from pathlib import Path
import numpy as np
import torch

REPO = Path('/Users/dimitargueorguiev/git/ml/semsimula-paper')
NB = REPO / 'notebooks/conservative_arch/scaleup/colab_fock_cfc_baoab_lowrank_depth_ladder_openwebtext_d384.ipynb'
SCR = Path(sys.argv[1])
VAL = Path.home() / 'Downloads/semsimula_fock_gamma_sweep_aniso_gaussian_fockreg_d384/data/openwebtext_val_2M.npy'
TAG = ('semsimula_fock_cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_'
       'L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_')

cells = {}
for c in json.load(open(NB))['cells']:
    s = ''.join(c['source'])
    for key in ('Cell 0:', 'Cell 1:', 'Cell 1b:', 'Cell 2:', 'Cell 4:', 'Cell 5:'):
        if s.split('\n', 1)[0].startswith('# == ' + key):
            cells[key] = s
strip = lambda s: '\n'.join(l for l in s.splitlines() if not l.lstrip().startswith(('!', '%')))
os.chdir(REPO / 'notebooks/conservative_arch/scaleup')


def build(mech, suffix):
    g = {'__name__': '__main__'}
    c0 = cells['Cell 0:']
    assert c0.count("LADDER_MECHANISM = 'none'") == 1
    c0 = c0.replace("LADDER_MECHANISM = 'none'", f"LADDER_MECHANISM = {mech!r}")
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(strip(c0), 'Cell0', 'exec'), g)
        exec(compile(strip(cells['Cell 1:']), 'Cell1', 'exec'), g)
        g['CKPT_DIR'] = Path.home() / 'Downloads' / (TAG + suffix) / 'checkpoints'
        g['RESULTS_DIR'] = SCR / suffix; g['RESULTS_DIR'].mkdir(parents=True, exist_ok=True)
        g['DATA_DIR'] = SCR / 'data'; g['GDRIVE_ROOT'] = SCR
        exec(compile(strip(cells['Cell 1b:']), 'Cell1b', 'exec'), g)
        g['DATA_DIR'] = SCR / 'data'
        exec(compile(strip(cells['Cell 2:']), 'Cell2', 'exec'), g)
        sys.path.insert(0, str(REPO / 'notebooks/conservative_arch'))
        exec('from data_module import get_batch', g)
        g['val_ids'] = np.load(VAL); g['train_ids'] = g['val_ids']   # Cell 3 stand-in
        exec(compile(strip(cells['Cell 4:']), 'Cell4', 'exec'), g)
        exec(compile(strip(cells['Cell 5:']), 'Cell5', 'exec'), g)
    model = g['model']
    ck = torch.load(g['CKPT_DIR'] / f"{g['CKPT_PREFIX']}_best.pt", map_location='cpu', weights_only=False)
    res = model.load_state_dict(ck['model_state_dict'], strict=False)
    assert not res.missing_keys and not res.unexpected_keys, res
    assert g['_variant_tag'].endswith(suffix), g['_variant_tag']
    model.eval()
    return model, ck


def exchange_force(model, h, layer):
    """The exchange field's force at state h, computed the way the model does."""
    mode = model.cfg.force_relaxation
    lam = model._relax_gate(layer)
    if mode == 'attention':                         # _layer_forces, arm N
        return lam * model.relax_field(h)
    assert mode == 'attention_potential'            # _add_relax_potential
    T = h.shape[1]
    h_src = h.detach() if model.cfg.causal_force else h
    route = h.detach() if getattr(model.relax_field, 'route_from', 'xi') == 'h' else None
    assert route is not None
    U = lam * model.relax_field.potential(h, h_src, route, model._pair_mask_for(T, h.device))
    gU, = torch.autograd.grad(U, h, create_graph=True)
    return -gU


def probe(model, traj, routing_params):
    out = []
    gen = torch.Generator().manual_seed(0)
    for layer in range(model.cfg.L):
        h = traj[layer].clone().float().requires_grad_(True)
        F = exchange_force(model, h, layer)
        B, T, d = F.shape
        cot = torch.zeros_like(F)
        cot[:, -1, :] = torch.randn(B, d, generator=gen)          # last position only
        params = [p for p in routing_params if p.requires_grad]
        grads = torch.autograd.grad((cot * F).sum(), [h] + params, allow_unused=True)
        gh = grads[0] if grads[0] is not None else torch.zeros_like(h)
        src = gh[:, :-1, :].norm().item()                         # earlier positions: value/key path
        qry = gh[:, -1, :].norm().item()                          # the position itself: query path
        pg = sum((x.norm().item() ** 2 if x is not None else 0.0) for x in grads[1:]) ** 0.5
        out.append(dict(layer=layer, force_rms=F.detach().pow(2).mean().sqrt().item(),
                        h_rms=h.detach().pow(2).mean().sqrt().item(),
                        grad_to_sources=src, grad_to_query=qry, grad_to_routing=pg,
                        grad_is_none=grads[0] is None))
    return out


if __name__ == '__main__':
    val = np.load(VAL)
    rng = np.random.default_rng(20260929)
    starts = rng.integers(0, len(val) - 513, size=2)
    x = torch.from_numpy(np.stack([val[s:s + 512] for s in starts]).astype(np.int64))
    report = {}
    for mech, suffix, rparams in (('attention', 'attn', ('W_Q', 'W_K')),
                                  ('attention_potential', 'attnpot', ('W_q', 'W_k'))):
        t = time.time()
        model, ck = build(mech, suffix)
        rp = [getattr(model.relax_field, n).weight for n in rparams]
        with torch.enable_grad():
            h0 = model._embed(x)
            _, traj = model._stack_forward(h0, x, return_trajectory=True)
            rows = probe(model, traj, rp)
        report[mech] = dict(step=ck['step'], ppl=ck['val_ppl'], rows=rows,
                            gate=[float(model._relax_gate(l)) for l in range(model.cfg.L)])
        print(f"\n== {mech}  (_best.pt step {ck['step']:,}, PPL {ck['val_ppl']:.2f}; "
              f"gate per layer {report[mech]['gate']})  [{time.time()-t:.0f}s]")
        print(f"{'layer':>5} {'|F| rms':>9} {'|h| rms':>9} {'grad->sources':>14} {'grad->query':>12} {'grad->routing W':>16}")
        for r in rows:
            print(f"{r['layer']:>5} {r['force_rms']:9.4f} {r['h_rms']:9.4f} {r['grad_to_sources']:14.4e} "
                  f"{r['grad_to_query']:12.4e} {r['grad_to_routing']:16.4e}"
                  + ("   (autograd: no path to h at all)" if r['grad_is_none'] else ""))
    json.dump(report, open(SCR / 'gradcheck_exchange.json', 'w'), indent=1)
