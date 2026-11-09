"""RO-COST: what does V_phi's straight-through router term do to a trained model at its own step count?
(protocol p-router-off-cost, 2026-11-08; descriptive, CPU)

The pair force has two parts (model_parf_multixi._pair_force_analytic): the pair
term, the gradient of the gated V_phi sum at fixed selection, and the router
term, the gate's gradient through the score head. The router term is the
measured exception to per-token conservativity at fixed context. TKF-A
(p-tkf-attribution-result) found that dropping it costs K2-L8-R6-CODE and
K2-L8-R6-VS-CODE nothing at N = L on Cell 6b-7's 12 batches. This script
measures the same on more tokens and more models.

The model runs its own forward (N = L, the trained step), with the analytic
V_phi force as in training (cfg.vphi_analytic_force = True), twice per
sequence: router on (the trained model) and router off (the gate detached and
the soft weights zero, so the router term is exactly zero; the pair term and
the selection are unchanged). Per batch: the loss of each, the mean KL(on || off)
per token, the share of tokens whose top-1 prediction changes, the largest
|d logit|.

Checks: the analytic and autograd paths give the same loss on one sequence
(relative 1e-5); at one layer's input, in float64, the router-off switch keeps
the selection and the gate's value, and the analytic and autograd pair forces
agree with the router term on, off, and in their difference (the router term
itself); the float64 round trip leaves the model unchanged.

Usage: python3 router_off_cost.py OUT_DIR FOLDER name=<stem> [ckpt=<file>] [batches=<n>]
  FOLDER: a code arm's run folder in ~/Downloads (built from its tag as FLOW-R's 'code' kind).
  ckpt: default the folder's single best checkpoint without a step in its name.
  batches: default 48, drawn at Cell 6b-7's seed (the first 12 are its batches).
"""
import contextlib, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BATCH, BLOCK = 4, 512


@contextlib.contextmanager
def router_off(model):
    """Detach the gate and zero the soft weights: the router term vanishes in the
    autograd and the analytic force alike; selection and gate value are kept."""
    orig = model._sparse_topk_indices

    def topk(pi, causal, T, return_soft=False):
        idx, m_g, y = orig(pi, causal, T, return_soft=True)
        m_g, y = m_g.detach(), torch.zeros_like(y)
        return (idx, m_g, y) if return_soft else (idx, m_g)
    model._sparse_topk_indices = topk
    try:
        yield
    finally:
        del model.__dict__['_sparse_topk_indices']
        assert model._sparse_topk_indices == orig


if __name__ == '__main__':
    OUT = Path(sys.argv[1]); FOLDER = sys.argv[2]
    NAME = next(a[5:] for a in sys.argv[3:] if a.startswith('name='))
    CKPT = next((a[5:] for a in sys.argv[3:] if a.startswith('ckpt=')), None)
    NB = next((int(a[8:]) for a in sys.argv[3:] if a.startswith('batches=')), 48)
    sys.argv = [sys.argv[0], str(OUT / 'harness')]
    sys.path.insert(0, str(Path(__file__).parent))
    import verify_pm_switch as P
    import types
    _src = (Path(__file__).parent / 'copy_split.py').read_text()
    _argv = list(sys.argv); sys.argv = ['copy_split.py', str(OUT / 'harness'), 'none']
    CS = types.SimpleNamespace(**(lambda g: (exec(compile(_src[:_src.index("\nif MODE == 'nll':")], 'copy_split.py', 'exec'), g), g)[1])(
        {'__name__': 'copy_split', '__file__': str(Path(__file__).parent / 'copy_split.py')}))
    sys.argv = _argv
    repl, _tag = CS.tag_repl(FOLDER)
    if not CKPT:
        _ck = [p.name for p in (Path.home() / 'Downloads' / FOLDER / 'checkpoints').glob('fock_cfc_*_best.pt')
               if '_step' not in p.name]
        assert len(_ck) == 1, _ck
        CKPT = _ck[0]
    model, g, _ = P.build(FOLDER, repl, ckpt=CKPT)
    assert g.get('_variant_tag', _tag) == _tag, (g.get('_variant_tag'), _tag)
    g['val_ids'] = np.load(CS.VAL)
    model.to('cpu'); model.eval()
    cfg = model.cfg
    assert cfg.use_gathered_v_phi and model.V_attn is None and model.V_phi is not None
    assert model._use_analytic_vtheta()
    L = cfg.L
    rng = np.random.default_rng(20260920)                    # Cell 6b-7's seed: its 12 batches come first
    batches = [g['get_batch'](g['val_ids'], BATCH, BLOCK, rng) for _ in range(NB)]

    lines = [f'RO-COST, the router term at the trained step: {NAME}  ...{FOLDER[-70:]}  (L = {L}, dt = {cfg.dt:g})',
             f'   checkpoint {CKPT.split("_")[-2] if "_step" in CKPT else "best"}; {NB} x {BATCH} x {BLOCK} tokens '
             f'(Cell 6b-7\'s seed; its 12 batches first); the model\'s own forward at N = L; analytic V_phi force']
    say = lambda s='': (print(s, flush=True), lines.append(s))

    # ---- checks ----
    ok = True
    xb, yb = batches[0]
    x, y = torch.from_numpy(xb[:1]), torch.from_numpy(yb[:1])
    losses = {}
    for flag in (False, True):
        cfg.vphi_analytic_force = flag
        with torch.enable_grad():
            losses[flag] = float(model(x, y)[1])
    cfg.vphi_analytic_force = True
    d = abs(losses[True] - losses[False]) / abs(losses[False]); ok &= d < 1e-5
    say(f'CHECK  analytic vs autograd V_phi force, loss on one sequence: {losses[True]:.6f} vs {losses[False]:.6f} (rel {d:.1e})')
    # at one layer's input, float64: router off = analytic pair force minus its router term, same selection and gate
    pts = []
    orig_lf = model._layer_forces
    def lf_capture(h_in, xis, layer_idx, **kw):
        pts.append((h_in.detach().clone(), layer_idx))
        return orig_lf(h_in, xis, layer_idx, **kw)
    model._layer_forces = lf_capture
    try:
        with torch.enable_grad():
            model(x[:, :256])
    finally:
        del model.__dict__['_layer_forces']
    live = getattr(cfg, 'vphi_grad_path', 'default') == 'live'
    h, li = pts[0]
    model.double()
    try:
        hs = h.double()
        hq = hs.view_as(hs) if live else hs
        seen = []
        orig_topk = model._sparse_topk_indices
        def topk_rec(pi, causal, T, return_soft=False):
            out = orig_topk(pi, causal, T, return_soft=True)
            seen.append(out)
            return out if return_soft else out[:2]
        model._sparse_topk_indices = topk_rec
        try:
            f_on = model._pair_force_analytic(hq, li, h_src_live=hs if live else None)
        finally:
            del model.__dict__['_sparse_topk_indices']
        with router_off(model):
            orig_off = model._sparse_topk_indices
            def topk_rec_off(pi, causal, T, return_soft=False):
                out = orig_off(pi, causal, T, return_soft=True)
                seen.append(out)
                return out if return_soft else out[:2]
            model._sparse_topk_indices = topk_rec_off
            f_off = model._pair_force_analytic(hq, li, h_src_live=hs if live else None)
        (i_on, m_on, _), (i_off, m_off, y_off) = seen
        same_sel = bool((i_on == i_off).all()) and float((m_on - m_off).abs().max()) == 0.0 and float(y_off.abs().max()) == 0.0
        ok &= same_sel
        say(f'CHECK  router off keeps the selection and the gate value, zeroes the soft weights: {same_sel}')
        # the same two forces by autograd of the pair potential; their difference is the router term
        hq2 = hs.clone().requires_grad_(True)
        hq2v = hq2.view_as(hq2) if live else hq2
        with torch.enable_grad():
            U_on = model._pair_potential(hq2v, li, h_src_live=hq2 if live else None)
            g_on = -torch.autograd.grad(U_on, hq2v)[0]
            with router_off(model):
                U_off = model._pair_potential(hq2v, li, h_src_live=hq2 if live else None)
                g_off = -torch.autograd.grad(U_off, hq2v)[0]
        rel = lambda a, b: float((a - b).abs().max() / b.abs().max())
        d1, d2 = rel(f_on, g_on), rel(f_off, g_off)
        d3 = rel(f_on - f_off, g_on - g_off) if float((g_on - g_off).abs().max()) > 0 else 0.0
        ok &= d1 < 1e-9 and d2 < 1e-9 and d3 < 1e-7
        say(f'CHECK  layer {li}, float64: analytic vs autograd, router on {d1:.1e}, router off {d2:.1e}, '
            f'router term itself {d3:.1e}; router term / pair force max {float((f_on - f_off).abs().max() / f_off.abs().max()):.2e}')
    finally:
        model.float()
    with torch.enable_grad():
        d = abs(float(model(x, y)[1]) - losses[True]); ok &= d == 0.0
    say(f'CHECK  loss after the float64 round trip unchanged: |d| {d:.1e}')
    say(f'-> checks {"PASS" if ok else "FAIL"}')
    if not ok:
        raise SystemExit(1)

    # ---- router on vs off, sequence by sequence ----
    per = []
    t0 = time.time()
    for b, (xb, yb) in enumerate(batches):
        rec = dict(on=0.0, off=0.0, kl=0.0, top1=0.0, maxd=0.0)
        for i in range(BATCH):
            x, y = torch.from_numpy(xb[i:i + 1]), torch.from_numpy(yb[i:i + 1])
            with torch.enable_grad():
                lo_on, l_on = model(x, y)
                with router_off(model):
                    lo_off, l_off = model(x, y)
            lo_on, lo_off = lo_on.detach().float(), lo_off.detach().float()
            lp_on, lp_off = F.log_softmax(lo_on, -1), F.log_softmax(lo_off, -1)
            rec['on'] += float(l_on) / BATCH
            rec['off'] += float(l_off) / BATCH
            rec['kl'] += float((lp_on.exp() * (lp_on - lp_off)).sum(-1).mean()) / BATCH
            rec['top1'] += float((lo_on.argmax(-1) != lo_off.argmax(-1)).float().mean()) / BATCH
            rec['maxd'] = max(rec['maxd'], float((lo_on - lo_off).abs().max()))
        per.append(rec)
        if (b + 1) % 4 == 0:
            dl = np.array([r['off'] - r['on'] for r in per])
            print(f'      {b + 1}/{NB} batches: d loss {dl.mean():+.5f} (SE {dl.std(ddof=1) / np.sqrt(len(dl)):.5f})  '
                  f'({time.time() - t0:.0f} s)', flush=True)

    on = np.array([r['on'] for r in per]); off = np.array([r['off'] for r in per]); dl = off - on
    kl = np.array([r['kl'] for r in per]); t1 = np.array([r['top1'] for r in per])
    se = dl.std(ddof=1) / np.sqrt(NB)
    say(f'\nROUTER OFF at N = {L} ({NB} batches, {NB * BATCH * BLOCK:,} tokens), paired by batch:')
    say(f'   loss on {on.mean():.5f} (PPL {np.exp(on.mean()):.4f}), off {off.mean():.5f} (PPL {np.exp(off.mean()):.4f})')
    say(f'   d loss (off - on) {dl.mean():+.5f} nats, SD over batches {dl.std(ddof=1):.5f}, SE {se:.5f}, '
        f'95% interval {dl.mean() - 1.96 * se:+.5f} .. {dl.mean() + 1.96 * se:+.5f}; worst batch {dl.max():+.5f}')
    say(f'   d loss on Cell 6b-7\'s 12 batches {dl[:12].mean():+.5f} (SD {dl[:12].std(ddof=1):.5f})')
    say(f'   KL(on || off) per token: mean {kl.mean():.2e}, worst batch {kl.max():.2e}; '
        f'top-1 prediction changed on {100 * t1.mean():.3f}% of tokens; largest |d logit| {max(r["maxd"] for r in per):.2f}')

    OUT.mkdir(parents=True, exist_ok=True)
    stem = f'router_off_cost_{NAME}'
    (OUT / f'{stem}_output.txt').write_text('\n'.join(lines) + '\n')
    (OUT / f'{stem}.json').write_text(json.dumps({'checkpoint': CKPT, 'per_batch': per}, indent=1))
    print(f'\nwrote {OUT}/{stem}_output.txt')
