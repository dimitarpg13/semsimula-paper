"""G3' switch check: QK-normalised exchange-field routing + relax_field clip (protocol SS5.10).

The OFF path is covered by verify_head_equiv_g3.py (HEAD vs working copy on
G3's configuration: bit-identical). This checks the ON path, built through the
ladder notebook's own Cells 0-5b with G3's Cell 0 plus
RELAX_ATTN_QK_NORM = True and RELAX_FIELD_CLIP = 0.3:

  1. tag: G3's tag with 'rfqk_rfclip0p3' after 'xilive' (its own Drive folder);
     Cell 5b's G3' banner prints and its asserts pass
  2. parameters: G3's set plus relax_field.logit_scale (one per head), at
     log(1/0.07); G3's trained weights load into everything else
  3. routing: |score| <= sigma_h <= logit_scale_max per head, also with the
     logit scale pushed past the ceiling; rows still sum to 1
  4. a train-mode loss.backward() reaches logit_scale, W_q and W_k
  5. causality: perturbing tokens at t >= 256 leaves logits at t < 256 unchanged
  6. clip groups (Cell 6's own override block): relax_field.* -> its own group
     at 0.3; every other parameter's group is unchanged

Usage: python3 verify_g3prime_switch.py OUT_DIR
"""
import contextlib, io, json, math, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import gradcheck_vphi_xi_paths as G
import verify_vphi_xi_grad_path as V

G3 = ('semsimula_fock_cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_rglive_vplive_'
      'xilive_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_attnpot')
G3_TAG = G3[len('semsimula_fock_cfc_baoab_owt_'):]
G3P_TAG = G3_TAG.replace('_xilive_', '_xilive_rfqk_rfclip0p3_')

G3_SWITCHES = (("VPHI_GRAD_PATH         = 'default'", "VPHI_GRAD_PATH         = 'live'"),
               ("XI_GRAD_PATH           = 'default'", "XI_GRAD_PATH           = 'live'"),
               ("LADDER_MECHANISM = 'none'", "LADDER_MECHANISM = 'attention_potential'"),
               ("RELAX_GRAD_PATH        = 'default'", "RELAX_GRAD_PATH        = 'live'"))
ON = (("RELAX_ATTN_QK_NORM          = False", "RELAX_ATTN_QK_NORM          = True"),
      ("RELAX_FIELD_CLIP            = None", "RELAX_FIELD_CLIP            = 0.3"))


if __name__ == '__main__':
    # V.build's strict key check refuses the new logit_scale, so this is its
    # cell sequence by hand, with G3's switches plus the two G3' lines.
    c0 = G.cells['Cell 0:']
    for old, new in G3_SWITCHES + ON:
        assert c0.count(old) == 1, old
        c0 = c0.replace(old, new)
    import os
    os.chdir(G.REPO / 'notebooks/conservative_arch/scaleup')
    g = {'__name__': '__main__'}
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(G.strip(c0), 'Cell0', 'exec'), g)
        exec(compile(G.strip(G.cells['Cell 1:']), 'Cell1', 'exec'), g)
        g['CKPT_DIR'] = Path.home() / 'Downloads' / G3 / 'checkpoints'
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
    log = out.getvalue()
    model, tag = g['model'], g['_variant_tag']
    rf = model.relax_field
    ok = []

    # 1. tag and banner
    print("G3' switch check (RELAX_ATTN_QK_NORM = True, RELAX_FIELD_CLIP = 0.3, on G3's Cell 0)")
    print(f"1. tag ...{tag[tag.find('cgqk'):]}")
    ok.append(tag == G3P_TAG)
    print(f"   = G3's tag with rfqk_rfclip0p3 after xilive: {ok[-1]}")
    ban = [l.strip() for l in log.splitlines() if "G3'" in l or 'routing: cosine' in l or 'clip: relax_field' in l]
    print('   Cell 5b banner:'); [print('     ' + b) for b in ban]
    ok.append(any('HARDENED EXCHANGE FIELD' in b for b in ban))

    # 2. parameters
    pfx = G3[len('semsimula_'):].replace('fock_cfc_baoab_owt', 'fock_cfc_owt')
    ck = torch.load(g['CKPT_DIR'] / f'{pfx}_best.pt', map_location='cpu', weights_only=False)
    r = model.load_state_dict(ck['model_state_dict'], strict=False)
    new = set(model.state_dict()) - set(ck['model_state_dict'])
    ok.append(r.missing_keys == ['relax_field.logit_scale'] and not r.unexpected_keys and new == {'relax_field.logit_scale'})
    sig0 = rf.logit_scale.detach().exp()
    ok.append(rf.logit_scale.numel() == rf.H and torch.allclose(sig0, torch.full_like(sig0, 1 / 0.07)))
    print(f"2. new parameters vs G3: {sorted(new)} ({rf.logit_scale.numel()} heads, sigma init "
          f"{float(sig0[0]):.4f} = 1/0.07); G3 weights load into the rest: {ok[-2]}")

    # 3. routing bound
    val = np.load(G.VAL)
    rng = np.random.default_rng(20261005)
    st = rng.integers(0, len(val) - 513, size=2)
    x = torch.from_numpy(np.stack([val[s:s + 512] for s in st]).astype(np.int64))
    y = torch.from_numpy(np.stack([val[s + 1:s + 513] for s in st]).astype(np.int64))
    T = x.shape[1]
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool), diagonal=-1)
    h = model._embed(x).detach()

    def scores_and_alpha():
        with torch.no_grad():
            q = torch.nn.functional.normalize(rf.W_q(h).view(2, T, rf.H, rf.d_k).transpose(1, 2), dim=-1)
            k = torch.nn.functional.normalize(rf.W_k(h).view(2, T, rf.H, rf.d_k).transpose(1, 2), dim=-1)
            sig = rf.logit_scale.exp().clamp(max=rf.logit_scale_max)
            sc = (q @ k.transpose(-1, -2)) * sig.view(1, -1, 1, 1)
            a = rf._routing(h, causal)
        return sc.masked_select(causal.expand_as(sc)).abs().max().item(), sig, a

    m, sig, a = scores_and_alpha()
    rows = a[..., 1:, :].sum(-1)
    ok.append(m <= float(sig.max()) + 1e-4 and torch.allclose(rows, torch.ones_like(rows), atol=1e-5))
    print(f"3. routing at init: max|score| {m:.3f} <= sigma {float(sig.max()):.3f}; rows sum to 1: {ok[-1]}")
    with torch.no_grad():
        rf.logit_scale.fill_(math.log(1e4))          # far past the ceiling
    m2, sig2, _ = scores_and_alpha()
    ok.append(float(sig2.max()) == rf.logit_scale_max and m2 <= rf.logit_scale_max + 1e-3)
    print(f"   logit scale pushed to 1e4: sigma clamps to {float(sig2.max()):g}, max|score| {m2:.2f} "
          f"<= {rf.logit_scale_max:g}: {ok[-1]}")
    with torch.no_grad():
        rf.logit_scale.fill_(math.log(1 / 0.07))

    # 4. gradients
    model.train(); torch.manual_seed(7)
    _, loss = model(x, y)
    loss.backward()
    gn = {n: float(p.grad.norm()) for n, p in model.named_parameters()
          if n.startswith('relax_field.') and p.grad is not None}
    ok.append(all(gn.get(n, 0) > 0 for n in ('relax_field.logit_scale', 'relax_field.W_q.weight',
                                               'relax_field.W_k.weight')))
    print(f"4. train loss {loss.item():.4f}; grad norms " +
          ', '.join(f"{n.split('.', 1)[1]} {v:.2e}" for n, v in gn.items()) + f": {ok[-1]}")
    model.zero_grad(set_to_none=True)

    # 5. causality
    model.eval()
    x2 = x.clone(); x2[:, 256:] = torch.from_numpy(rng.integers(0, 50257, size=(2, T - 256)))
    with torch.enable_grad():
        l1, _ = model(x); l2, _ = model(x2)
    d = (l1[:, :256] - l2[:, :256]).abs().max().item()
    ok.append(d == 0.0)
    print(f"5. causality: future tokens replaced at t >= 256, max|dlogit| at t < 256 = {d:.1e}: {ok[-1]}")

    # 6. clip groups, from Cell 6's own override block
    c6 = [''.join(c['source']) for c in json.load(open(G.NB))['cells']
          if ''.join(c['source']).startswith('# == Cell 6: Training loop')][0]
    i0 = c6.index('GRAD_CLIP_OVERRIDES = {')
    i1 = c6.index("    GRAD_CLIP_OVERRIDES['relax_field'] = RELAX_FIELD_CLIP\n") + len(
        "    GRAD_CLIP_OVERRIDES['relax_field'] = RELAX_FIELD_CLIP\n")
    from grad_clip_utils import GradClipConfig, assign_clip_group
    grp = {}
    for clip in (None, 0.3):
        ns = {'GRAD_CLIP_VPHI': 0.3, 'RELAX_FIELD_CLIP': clip}
        exec(c6[i0:i1], ns)
        cfg = GradClipConfig(default_clip=1.0, overrides=ns['GRAD_CLIP_OVERRIDES'])
        grp[clip] = {n: assign_clip_group(n, cfg) for n, _ in model.named_parameters()}
    moved = {n for n in grp[None] if grp[None][n] != grp[0.3][n]}
    rfn = {n for n in grp[None] if n.startswith('relax_field.')}
    ok.append(moved == rfn and all(grp[0.3][n] == ('override:relax_field', 0.3) for n in rfn))
    print(f"6. clip groups: {len(rfn)} relax_field parameters move from "
          f"{sorted({grp[None][n] for n in rfn})} to {sorted({grp[0.3][n] for n in rfn})}; "
          f"no other parameter changes group: {ok[-1]}")

    print(f"\n-> {'ALL PASS' if all(ok) else 'FAIL: ' + str(ok)}")
