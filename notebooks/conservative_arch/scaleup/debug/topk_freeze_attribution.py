"""TKF-A: how much of the whole-stack refinement penalty is V_phi re-selecting its sources?
(protocol p-tkf-attribution, calls registered 2026-11-08 before the run)

The pair potential sums V_phi over each token's top-k sources, selected by
the score head at every force evaluation. Refined to N > L steps, a trained
layer's share of the interval re-selects at each substep; the potential the
substeps integrate then changes piecewise. Gate 3 (policy 'hold', T = L dt,
N = L, 1.5L, 2L, 3L, 4L) is re-run under five arms, all through
refinement_ln_linearisation_split.split_stack:

  as trained         every substep re-selects (= Cell 6b-7's Gate 3)
  top-k frozen       the substeps of one trained layer's share reuse the
                     selection made at the share's first substep; the
                     sources' positions, V_phi and the soft weights stay live
  top-k frozen, router off
                     the same, and the force drops the straight-through
                     router term: at fixed context and selection it is then
                     an exact gradient. Changes the model at N = L too
  per-layer flow     FLOW-R's arm (LN once, quadratic and xi frozen per
                     share; top-k live), on these batches
  all frozen         per-layer flow + top-k frozen

Also, at the trained N = L on the first two sequences: the router term's
size, |f(router on) - f(router off)| / |f(router on)| per token at every
force evaluation point, with the router-off force checked between the
autograd and analytic paths; and the selection's churn: at N = 2L and 4L
(as trained), the fraction of a query's k sources that were also selected
at its share's first substep.

Checks first, all exact unless stated: switches off = 6b-7's _fom_stack at
N = L, L+1; top-k frozen, per-layer flow and all frozen = trained at N = L;
one fresh selection per trained layer at N = 4L; the rebuilt gate equals
_sparse_topk_indices' own; the pair force agrees between the autograd and
analytic paths, router on and off, in float64 (relative 1e-9).

Usage: python3 topk_freeze_attribution.py OUT_DIR FOLDER ckpt=<file> name=<stem> [micro=<k>] [checks]
  FOLDER: a code arm's run folder in ~/Downloads (built from its tag as FLOW-R's 'code' kind).
"""
import ast, json, sys, time
from pathlib import Path

import numpy as np
import torch

NS_FACTORS = (1, 1.5, 2, 3, 4)
N_BATCH, BATCH, BLOCK = 12, 4, 512

if __name__ == '__main__':
    OUT = Path(sys.argv[1]); FOLDER = sys.argv[2]
    CKPT = next(a[5:] for a in sys.argv[3:] if a.startswith('ckpt='))
    NAME = next(a[5:] for a in sys.argv[3:] if a.startswith('name='))
    MICRO = next((int(a[6:]) for a in sys.argv[3:] if a.startswith('micro=')), BATCH)
    CHECKS_ONLY = 'checks' in sys.argv[3:]
    assert BATCH % MICRO == 0, MICRO
    sys.argv = [sys.argv[0], str(OUT / 'harness')]
    sys.path.insert(0, str(Path(__file__).parent))
    import verify_pm_switch as P
    from refinement_ln_linearisation_split import split_stack, gate
    # copy_split's definitions without its command dispatch, as refinement_flow_confirmation.py loads them
    import types
    _src = (Path(__file__).parent / 'copy_split.py').read_text()
    _argv = list(sys.argv); sys.argv = ['copy_split.py', str(OUT / 'harness'), 'none']
    CS = types.SimpleNamespace(**(lambda g: (exec(compile(_src[:_src.index("\nif MODE == 'nll':")], 'copy_split.py', 'exec'), g), g)[1])(
        {'__name__': 'copy_split', '__file__': str(Path(__file__).parent / 'copy_split.py')}))
    sys.argv = _argv
    repl, _tag = CS.tag_repl(FOLDER)
    assert 'sr2' in _tag.split('_') and 'fob' not in _tag.split('_'), _tag
    model, g, _ = P.build(FOLDER, repl, ckpt=CKPT)
    assert g.get('_variant_tag', _tag) == _tag, (g.get('_variant_tag'), _tag)
    g['val_ids'] = np.load(CS.VAL)
    model.to('cpu'); model.eval()
    cfg = model.cfg
    assert cfg.integrator == 'baoab_cfc_lowrank' and model.reverse_ch is None and cfg.ln_after_step
    assert cfg.force_clamp_max is None and cfg.use_gathered_v_phi and model.V_attn is None
    assert getattr(model, 'pm_mu', None) is None
    print(f'checkpoint: {CKPT}  (code validation tokens {CS.VAL}); vphi_analytic_force={cfg.vphi_analytic_force}, '
          f'vphi_grad_path={cfg.vphi_grad_path}, top_k={cfg.top_k}', flush=True)

    nb = json.load(open(P.G.NB))
    src = [''.join(c['source']) for c in nb['cells'] if ''.join(c['source']).startswith('# == Cell 6b-7:')][0]
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    exec(compile(ast.Module(body=keep, type_ignores=[]), 'Cell6b7_defs', 'exec'), g)
    policy_index, fom_stack = g['_fom_policy_index'], g['_fom_stack']

    rng = np.random.default_rng(20260920)                    # Cell 6b-7's seed and batches
    batches = [g['get_batch'](g['val_ids'], BATCH, BLOCK, rng) for _ in range(12)][:N_BATCH]
    L, DT = cfg.L, cfg.dt
    T = L * DT
    NS = tuple(int(f * L) for f in NS_FACTORS)
    OFF = dict(ln_once=False, freeze=False)
    FLOW = dict(ln_once=True, freeze=True, xi_freeze=True)
    ARMS = {'as trained': OFF,
            'top-k frozen': dict(OFF, topk_freeze=True),
            'top-k frozen, router off': dict(OFF, topk_freeze=True, router_off=True),
            'per-layer flow': FLOW,
            'all frozen': dict(FLOW, topk_freeze=True)}

    lines = [f'TKF-A, top-k freeze attribution: {NAME}  ...{FOLDER[-70:]}  (L = {L}, dt = {DT:g})',
             f'   checkpoint {CKPT.split("_")[-2]}; {N_BATCH} x {BATCH} x {BLOCK} tokens (Cell 6b-7\'s batches); '
             f'policy hold; T = {T:g}' + ('' if MICRO == BATCH else f'; Gate 3 in forwards of {MICRO}')
             + f'; force path {"analytic" if cfg.vphi_analytic_force else "autograd"}']
    say = lambda s='': (print(s, flush=True), lines.append(s))

    def stack(n, arm):
        return split_stack(model, n, T / n, policy_index, **arm)

    # ---- checks ----
    xb, _ = batches[0]
    x = torch.from_numpy(xb[:2, :256])
    def logits(ctx):
        with ctx, torch.enable_grad():
            return model(x)[0].detach()
    ok = True
    for n in (L, L + 1):
        d = float((logits(fom_stack(model, n, T / n, 'hold')) - logits(stack(n, OFF))).abs().max()); ok &= d == 0.0
        say(f'CHECK  N={n}: as trained vs Cell 6b-7 _fom_stack, max |d logit| {d:.1e}')
    base = logits(stack(L, OFF))
    for name in ('top-k frozen', 'per-layer flow', 'all frozen'):
        d = float((logits(stack(L, ARMS[name])) - base).abs().max()); ok &= d == 0.0
        say(f'CHECK  N={L}: {name} vs the trained model, max |d logit| {d:.1e}')
    d_ro = float((logits(stack(L, ARMS['top-k frozen, router off'])) - base).abs().max())
    say(f'       N={L}: router off vs the trained model, max |d logit| {d_ro:.2e} (a different model; measured below)')
    for name in ('top-k frozen', 'all frozen'):
        with stack(4 * L, ARMS[name]) as st, torch.enable_grad():
            model(x)
        good = st['tfresh'] == L and st['tlate'] == 3 * L
        ok &= good
        say(f'CHECK  N={4 * L} {name}: fresh / reused selections = {st["tfresh"]} / {st["tlate"]} (expected {L} / {3 * L})')

    # the rebuilt gate is _sparse_topk_indices' own, bit for bit
    orig_topk = model._sparse_topk_indices
    gd = []
    def topk_probe(pi, causal, T_, return_soft=False):
        idx, m_g, y = orig_topk(pi, causal, T_, return_soft=True)
        gd.append(float((gate(idx, y, causal) - m_g).abs().max()))
        return (idx, m_g, y) if return_soft else (idx, m_g)
    model._sparse_topk_indices = topk_probe
    try:
        logits(stack(L, OFF))
    finally:
        del model.__dict__['_sparse_topk_indices']
    ok &= len(gd) == L and max(gd) == 0.0
    say(f'CHECK  rebuilt gate vs _sparse_topk_indices over {len(gd)} calls: max |d| {max(gd):.1e}')

    # force evaluation points at the trained N = L (as trained), for the router term's size
    pts = []
    orig_lf = model._layer_forces
    def lf_capture(h_in, xis, layer_idx, **kw):
        pts.append((h_in.detach().clone(), layer_idx))
        return orig_lf(h_in, xis, layer_idx, **kw)
    model._layer_forces = lf_capture
    try:
        with stack(L, OFF), torch.enable_grad():
            model(torch.from_numpy(xb[:2]))
    finally:
        del model.__dict__['_layer_forces']
    live = getattr(cfg, 'vphi_grad_path', 'default') == 'live'

    def pair_forces(h, li):
        hs = h.clone().requires_grad_(True)
        hq = hs.view_as(hs) if live else hs
        with torch.enable_grad():
            U = model._pair_potential(hq, li, h_src_live=hs if live else None)
            f_auto = -torch.autograd.grad(U, hq)[0]
            f_an = model._pair_force_analytic(hq, li, h_src_live=hs if live else None).detach()
        return f_auto, f_an

    router = {}
    rel = lambda a, b: float((a - b).abs().max() / b.abs().max())
    for h, li in pts:
        f_on, _ = pair_forces(h, li)
        with stack(L, dict(OFF, router_off=True)):
            f_off, _ = pair_forces(h, li)
        # the two paths in float64 (float32 rounds the router term's softmax derivative to ~1e-5)
        model.double()
        try:
            a_on, n_on = pair_forces(h.double(), li)
            with stack(L, dict(OFF, router_off=True)):
                a_off, n_off = pair_forces(h.double(), li)
        finally:
            model.float()
        d_on, d_off = rel(n_on, a_on), rel(n_off, a_off)
        ok &= d_on < 1e-9 and d_off < 1e-9
        say(f'CHECK  layer {li}: autograd vs analytic pair force in float64, router on {d_on:.1e}, router off {d_off:.1e} (max rel)')
        share = ((f_on - f_off).norm(dim=-1) / f_on.norm(dim=-1).clamp_min(1e-12)).flatten()
        router[li] = dict(median=float(share.median()), p90=float(share.quantile(0.9)),
                          pair_norm_median=float(f_off.norm(dim=-1).median()),
                          router_norm_median=float((f_on - f_off).norm(dim=-1).median()))
    d = float((logits(stack(L, OFF)) - base).abs().max()); ok &= d == 0.0
    say(f'CHECK  N={L}: the trained model after the float64 round trip, max |d logit| {d:.1e}')
    say(f'-> checks {"PASS" if ok else "FAIL"}')
    if not ok:
        raise SystemExit(1)
    if CHECKS_ONLY:
        print('checks only: stopping before the churn and Gate 3', flush=True)
        raise SystemExit(0)

    say(f'\nROUTER TERM at the trained N = {L} (2 x {BLOCK} tokens, at each force evaluation point): '
        '|f(router on) - f(router off)| / |f(router on)| of the V_phi force, per token')
    for li, r in router.items():
        say(f'   layer {li}: median {r["median"]:.3f} (p90 {r["p90"]:.3f}); |f_pair| median {r["pair_norm_median"]:.3g}, '
            f'|f_router| median {r["router_norm_median"]:.3g}')

    churn = {}
    say(f'\nCHURN, as trained (2 x {BLOCK} tokens): fraction of a query\'s {cfg.top_k} sources also selected at its '
        'share\'s first substep, mean over queries, per later substep')
    for n in (2 * L, 4 * L):
        with stack(n, dict(OFF, topk_churn=True)) as st, torch.enable_grad():
            model(torch.from_numpy(xb[:2]))
        per = {}
        for li, f in st['churn']:
            per.setdefault(li, []).append(f)
        churn[n] = per
        say(f'   N={n}: ' + '  '.join(f'L{li} ' + '/'.join(f'{v:.3f}' for v in fs) for li, fs in per.items()))

    # ---- Gate 3, five arms ----
    def ev(arm, n):
        per = []
        with stack(n, arm):
            for xb_, yb_ in batches:
                part = []
                for i in range(0, BATCH, MICRO):
                    with torch.enable_grad():
                        _, loss = model(torch.from_numpy(xb_[i:i + MICRO]), torch.from_numpy(yb_[i:i + MICRO]))
                    part.append(float(loss))
                per.append(sum(part) / len(part))
        return per

    out = {'checkpoint': CKPT, 'router': router, 'churn': churn, 'gate3': {}}
    say(f'\nGATE 3: PPL at N steps of dt = T/N, then the penalty ln(PPL_N / PPL_{L}) in nats (against the arm\'s own N = {L})')
    t0 = time.time()
    for name, arm in ARMS.items():
        per = {}
        for n in NS:
            if n == L and name in ('top-k frozen', 'per-layer flow', 'all frozen'):
                per[n] = out['gate3']['as trained'][L]          # identical at N = L (checked above)
            else:
                per[n] = ev(arm, n)
            print(f'      {name} N={n}: {np.exp(np.mean(per[n])):.3f}  ({time.time() - t0:.0f} s)', flush=True)
        out['gate3'][name] = per
        loss = {n: float(np.mean(v)) for n, v in per.items()}
        say(f'   {name:<26} PPL ' + '  '.join(f'N={n}:{np.exp(loss[n]):.3f}' for n in NS))
        say(f'   {"":<26} pen ' + '  '.join(f'N={n}:{loss[n] - loss[L]:+.3f}' for n in NS if n != L))

    loss = {a: {n: np.array(v) for n, v in per.items()} for a, per in out['gate3'].items()}
    pen = {a: {n: loss[a][n] - loss[a][L] for n in NS if n != L} for a in loss}
    say(f'\nPAIRED against as trained: penalty difference (arm minus as trained), mean and SD over the {N_BATCH} batches; '
        'share of the as-trained penalty removed')
    for a in ('top-k frozen', 'top-k frozen, router off'):
        say(f'   {a:<26} ' + '  '.join(
            f'N={n}: {pen[a][n].mean() - pen["as trained"][n].mean():+.3f} (SD {np.std(pen[a][n] - pen["as trained"][n], ddof=1):.3f}, '
            f'{100 * (1 - pen[a][n].mean() / pen["as trained"][n].mean()):+.0f}%)' for n in NS if n != L))
    say('PAIRED against the per-layer flow (all frozen minus per-layer flow):')
    say(f'   {"all frozen":<26} ' + '  '.join(
        f'N={n}: {pen["all frozen"][n].mean() - pen["per-layer flow"][n].mean():+.3f} '
        f'(SD {np.std(pen["all frozen"][n] - pen["per-layer flow"][n], ddof=1):.3f})' for n in NS if n != L))
    d8 = loss['top-k frozen, router off'][L] - loss['as trained'][L]
    say(f'ROUTER OFF at the trained N = {L}: loss {d8.mean():+.4f} nats against the trained model (SD over batches {np.std(d8, ddof=1):.4f})')

    txt = '\n'.join(lines)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f'topk_freeze_attribution_{NAME}'
    (OUT / f'{stem}_output.txt').write_text(txt + '\n')
    (OUT / f'{stem}.json').write_text(json.dumps(out, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
    print(f'\nwrote {OUT}/{stem}_output.txt')
