"""Check the CB-series switches (protocol SS5.11) before any CB run.

Builds G2's configuration (Fock-PARFLM 'none', L=2, VPHI/XI live) through the
ladder notebook's own Cells 0-5b, with and without each CB switch, on the
trained Gen 2 no-exchange _best.pt (the same parameter set as G2) and real
validation tokens, and reports:

  1. tags: each CB arm gets its own Drive folder; G2's tag is unchanged
  2. parameters: CB1/CB2 add none; CB3 adds exactly 2*L*d + L, and building
     it consumes no RNG (fresh inits of the shared parameters bit-identical)
  3. bit-identity: CB2 with an unbounded budget, CB3 with the gate pinned at
     1, and CB1 past its ramp give logits identical to G2's, eval and train
     (same Gumbel seed)
  4. the switches act: CB2 at rho=0.1 bounds every token's eta; CB1 ramps
     slower; CB3 starts at g=0.957, its L1 term reaches the gate's weights
  5. CB3 stays causal: future perturbation leaves earlier logits bit-exact
  0. the default build on this model file is bit-identical to the committed
     (HEAD) model file, eval and train -- a running G2 can resume on it

Usage: python3 verify_cb_switches.py OUT_DIR
  Step 0 needs OUT_DIR/head_ref.pt, written by a first run with --dump-ref
  while parf/model_fock_parf_multixi.py is the committed (HEAD) version.
"""
import contextlib, io, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import gradcheck_vphi_xi_paths as G

FOLDER = G.ARMS['no-exchange'][1]          # Gen 2 L=2 no-exchange: G2's parameter set
G2_TAG = ('xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_vplive_xilive_L2probe_ob_untied_'
          'wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn')
c5b = [''.join(c['source']) for c in json.load(open(G.NB))['cells']
       if ''.join(c['source']).startswith('# == Cell 5b')][0]


def build(budget=None, gate_l1=None, warm=4000, load=True, seed=None):
    import os
    os.chdir(G.REPO / 'notebooks/conservative_arch/scaleup')
    g = {'__name__': '__main__'}
    c0 = G.cells['Cell 0:']
    for old, new in (("VPHI_GRAD_PATH         = 'default'", "VPHI_GRAD_PATH         = 'live'"),
                     ("XI_GRAD_PATH           = 'default'", "XI_GRAD_PATH           = 'live'"),
                     ('REVERSE_CHANNEL_WARMUP_STEPS = 4000', f'REVERSE_CHANNEL_WARMUP_STEPS = {warm}'),
                     ('FOCK_BUDGET  = None', f'FOCK_BUDGET  = {budget!r}'),
                     ('FOCK_GATE_L1 = None', f'FOCK_GATE_L1 = {gate_l1!r}')):
        assert c0.count(old) == 1, old
        c0 = c0.replace(old, new)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
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
        if seed is not None:
            torch.manual_seed(seed)
        exec(compile(G.strip(G.cells['Cell 5:']), 'Cell5', 'exec'), g)
        g['PROBE_MAX_STEPS'] = g.get('PROBE_MAX_STEPS')
        exec(compile(G.strip(c5b), 'Cell5b', 'exec'), g)
    model = g['model']
    if load:
        parent = FOLDER[len('semsimula_'):].replace('fock_cfc_baoab_owt', 'fock_cfc_owt')
        ck = torch.load(g['CKPT_DIR'] / f'{parent}_best.pt', map_location='cpu', weights_only=False)
        r = model.load_state_dict(ck['model_state_dict'], strict=False)
        assert not r.unexpected_keys, r
        assert set(r.missing_keys) <= {'fock_gate_w', 'fock_gate_b'}, r
    banner = [l.strip() for l in out.getvalue().splitlines() if l.strip().startswith(('CB', 'CB SERIES'))]
    return model, g['_variant_tag'], banner


def logits(model, x, train):
    if train:
        model.train(); torch.manual_seed(7)
        lo, _ = model(x)
    else:
        model.eval()
        with torch.enable_grad():
            lo, _ = model(x)
    return lo.detach()


if __name__ == '__main__':
    val = np.load(G.VAL)
    rng = np.random.default_rng(20261003)
    starts = rng.integers(0, len(val) - 513, size=2)
    x = torch.from_numpy(np.stack([val[s:s + 512] for s in starts]).astype(np.int64))
    y = torch.from_numpy(np.stack([val[s + 1:s + 513] for s in starts]).astype(np.int64))

    ref, tag, _ = build()
    print(f"G2 reference: tag ...{tag[tag.find('xi5long'):]}")
    assert tag.endswith(G2_TAG), 'G2 tag changed -- a running G2 could not resume'
    n_ref = sum(p.numel() for p in ref.parameters())
    ref_ev, ref_tr = logits(ref, x, False), logits(ref, x, True)
    L, d = ref.cfg.L, ref.cfg.d

    ref.train(); torch.manual_seed(7); _, lr_ = ref(x, y); lr_.backward()
    grads = {k: p.grad.clone() for k, p in ref.named_parameters() if p.grad is not None}
    ref.zero_grad(set_to_none=True)
    if '--dump-ref' in sys.argv:
        # Run with the committed (HEAD) model file in place; see usage.
        torch.save({'eval': ref_ev, 'train': ref_tr, 'grads': grads}, G.OUT / 'head_ref.pt')
        print(f"   dumped HEAD reference to {G.OUT / 'head_ref.pt'}")
        sys.exit(0)
    print("\n0. DEFAULT BUILD AGAINST THE COMMITTED MODEL FILE")
    hr = torch.load(G.OUT / 'head_ref.pt')
    de = (hr['eval'] - ref_ev).abs().max().item(); dt_ = (hr['train'] - ref_tr).abs().max().item()
    gd = max((hr['grads'][k] - grads[k]).abs().max().item() for k in grads)
    assert set(hr['grads']) == set(grads)
    print(f"   HEAD vs working file, G2 config: eval max|dlogit| {de:.1e}   train {dt_:.1e}   "
          f"grads max|diff| {gd:.1e}   -> {'IDENTICAL' if de == dt_ == gd == 0 else 'DIFFERS'}")

    print("\n1-3. TAGS, PARAMETERS, BIT-IDENTITY AT THE NEUTRAL SETTING")
    cases = (('CB1 rcw20000 (past its ramp)', dict(warm=20000), lambda m: None),
             ('CB2 budget 1e6 (unbounded)', dict(budget=1e6), lambda m: None),
             ('CB3 gate pinned at 1', dict(gate_l1=0.02),
              lambda m: setattr(m.cfg, 'fock_gate_pin', True)))
    for name, kw, prep in cases:
        m, t, banner = build(**kw)
        prep(m)
        n = sum(p.numel() for p in m.parameters())
        de = (logits(m, x, False) - ref_ev).abs().max().item()
        dt = (logits(m, x, True) - ref_tr).abs().max().item()
        comp = t[t.find('xilive_') + 7:t.find('_L2probe')] or '(none)'
        print(f"   {name:30s} tag component {comp:10s} params {n - n_ref:+,}   "
              f"eval max|dlogit| {de:.1e}   train {dt:.1e}   -> "
              f"{'IDENTICAL' if de == 0 and dt == 0 else 'DIFFERS'}")
        for b in banner:
            print(f"      5b: {b}")
        assert t != tag
        assert n - n_ref == (2 * L * d + L if 'gate_l1' in kw else 0)

    fresh_ref, _, _ = build(load=False, seed=0)
    fresh_cb3, _, _ = build(gate_l1=0.02, load=False, seed=0)
    sd_r, sd_c = fresh_ref.state_dict(), fresh_cb3.state_dict()
    same = all(torch.equal(sd_r[k], sd_c[k]) for k in sd_r)
    print(f"   CB3 fresh init, shared params bit-identical to G2's fresh init: {same}")

    print("\n4. THE SWITCHES ACT")
    m, _, _ = build(budget=0.1)
    logits(m, x, False)
    print("   CB2 rho=0.1, eval: per layer eta mean / max = "
          + ', '.join(f"{c['eta']:.4f} / {c['eta_max']:.4f}" for c in m.cb_stats)
          + f"   -> {'BOUND HOLDS' if all(c['eta_max'] <= 0.1 + 1e-5 for c in m.cb_stats) else 'BOUND VIOLATED'}")
    ref.set_fock_capture(True) if hasattr(ref, 'set_fock_capture') else None
    ref._fock_capture = []
    logits(ref, x, False)
    print("   G2 config uncapped, eval: per layer eta mean / p90 / max = "
          + ', '.join(f"{c['eta']:.3f} / {c['eta_p90']:.3f} / {c['eta_max']:.3f}" for c in ref.cb_stats))
    ref._fock_capture = None
    m, _, _ = build(warm=20000)
    print(f"   CB1 ramp at 4,000 forwards: {min(1.0, 4000 / 20000):.2f} (G2: 1.00); "
          f"cfg.reverse_channel_warmup_steps = {m.cfg.reverse_channel_warmup_steps}")
    m, _, _ = build(gate_l1=0.02)
    m.train(); torch.manual_seed(7)
    lo, l = m(x, y)
    pen = m.pop_fock_gate_mean()
    gpen = torch.autograd.grad(pen, m.fock_gate_b, retain_graph=True)[0].norm().item()
    n_stats_fwd = len(m.cb_stats)
    (l + 0.02 * pen).backward()
    print(f"   CB3 readings: {n_stats_fwd} layers after forward, {len(m.cb_stats)} after backward "
          f"(recompute must not touch them); |d L1 / d b| alone {gpen:.3e}")
    assert n_stats_fwd == len(m.cb_stats) == L and gpen > 0
    gw = m.fock_gate_w.grad.norm().item(); gb = m.fock_gate_b.grad.norm().item()
    print(f"   CB3 at init, train: gate mean per layer {[round(c['gate_mean'], 4) for c in m.cb_stats]}, "
          f"g==0 {[c['gate_zero'] for c in m.cb_stats]}, L1 term {pen.item():.4f}, "
          f"|grad w| {gw:.3e}, |grad b| {gb:.3e}")
    assert gw > 0 and gb > 0

    print("\n5. CB3 CAUSALITY (gate made content-dependent; future perturbation)")
    with torch.no_grad():
        m.fock_gate_w.normal_(0, 0.05, generator=torch.Generator().manual_seed(3))
        m.fock_gate_b.fill_(0.5)
    base = logits(m, x, False)
    print(f"   gate mean / g==0 with random weights: "
          f"{[(round(c['gate_mean'], 3), round(c['gate_zero'], 3)) for c in m.cb_stats]}")
    worst = 0.0
    gen = torch.Generator().manual_seed(1)
    for p in (63, 255, 450):
        xc = x.clone(); xc[:, p + 1:] = torch.randint(0, ref.cfg.vocab_size, xc[:, p + 1:].shape, generator=gen)
        worst = max(worst, (logits(m, xc, False)[:, :p + 1] - base[:, :p + 1]).abs().max().item())
    print(f"   max |dlogit| at positions <= p over p in (63, 255, 450): {worst:.1e}  -> "
          f"{'CLEAN' if worst == 0 else 'LEAK'}")
