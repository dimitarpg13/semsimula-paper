"""FLOW-C: confirmation of the per-layer flow (protocol SS5.19, pre-registered 2026-10-09).

The per-layer flow: each trained layer's share of the interval T is integrated
with N/L substeps, with the context xi, V_theta's low-rank quadratic and (PM1)
the occupation phi taken at the share's first substep, the explicit kick at
the real force, and one LayerNorm projection at the end of the share. At N = L
it is the trained model. The stack is refinement_ln_linearisation_split's,
with every switch on.

  1. GATE 3 under 'as trained' (Cell 6b-7's) and 'per-layer flow', all 12 of
     6b-7's batches, policy hold, N in NS at fixed T. Penalty ln(PPL_N/PPL_2).
  2. KICK SHARE at the trained N = 2: every layer step of the first batch is
     replayed from its captured inputs with the explicit kick exactly zero
     (force_clamp_max = 0 clamps the kick, and only the kick: the split force
     path returns before its own clamp). |h(no kick) - h| / |h - h_in| per
     token, median per layer; velocity likewise.

Checks first, all exact: switches off = 6b-7's _fom_stack; per-layer flow at
N = 2 = trained; one fresh xi / quadratic (/ phi) per trained layer at N = 8;
the replay with the normal clamp reproduces the captured output.

Usage: python3 refinement_flow_confirmation.py OUT_DIR FOLDER {f31|sr2|pm1} [seed=<int>]
  seed=<int> (FLOW-R, 2026-10-09): draw the 12 batches with this seed instead
  of Cell 6b-7's 20260920, and run 'as trained' at N = 2 and 8 only.
"""
import ast, json, sys
from pathlib import Path

import numpy as np
import torch

NS = (2, 3, 4, 6, 8, 12, 16)
N_BATCH, BATCH, BLOCK = 12, 4, 512

if __name__ == '__main__':
    OUT = Path(sys.argv[1]); FOLDER = sys.argv[2]; KIND = sys.argv[3]
    SEED = next((int(a[5:]) for a in sys.argv[4:] if a.startswith('seed=')), None)
    assert KIND in ('f31', 'sr2', 'pm1'), KIND
    sys.argv = [sys.argv[0], str(OUT / 'harness')]
    sys.path.insert(0, str(Path(__file__).parent))
    import verify_pm_switch as P
    from refinement_ln_linearisation_split import split_stack
    repl = P.CONFIGS['F3.1'][1]
    if KIND == 'sr2':
        repl = repl + (("LOWRANK_DAMPED_FLOW  = False", "LOWRANK_DAMPED_FLOW  = True"),)
    if KIND == 'pm1':
        repl = repl + P.PM_ON
    model, g, _ = P.build(FOLDER, repl)
    model.to('cpu'); model.eval()
    assert bool(getattr(model.cfg, 'lowrank_damped_flow', False)) == (KIND == 'sr2')
    assert (getattr(model, 'pm_mu', None) is not None) == (KIND == 'pm1')
    assert model.cfg.integrator == 'baoab_cfc_lowrank' and model.reverse_ch is None and model.cfg.ln_after_step
    assert model.cfg.force_clamp_max is None

    nb = json.load(open(P.G.NB))
    src = [''.join(c['source']) for c in nb['cells'] if ''.join(c['source']).startswith('# == Cell 6b-7:')][0]
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    exec(compile(ast.Module(body=keep, type_ignores=[]), 'Cell6b7_defs', 'exec'), g)
    policy_index, fom_stack = g['_fom_policy_index'], g['_fom_stack']

    rng = np.random.default_rng(20260920 if SEED is None else SEED)   # the cell's seed, or FLOW-R's
    batches = [g['get_batch'](g['val_ids'], BATCH, BLOCK, rng) for _ in range(12)][:N_BATCH]
    L, DT = model.cfg.L, model.cfg.dt
    T = L * DT
    PM = KIND == 'pm1'
    FLOW = dict(ln_once=True, freeze=True, xi_freeze=True, phi_freeze=PM)
    TRAINED = dict(ln_once=False, freeze=False, xi_freeze=False, phi_freeze=False)

    lines = [f'{"FLOW-C" if SEED is None else "FLOW-R"}, the per-layer flow: {KIND}  ...{FOLDER[-70:]}',
             f'   {N_BATCH} x {BATCH} x {BLOCK} tokens (' + ("Cell 6b-7's batches" if SEED is None else f'batch seed {SEED}')
             + f'); policy hold; T = {T:g}']
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
    for n in (2, 3):
        d = float((logits(fom_stack(model, n, T / n, 'hold')) - logits(stack(n, TRAINED))).abs().max()); ok &= d == 0.0
        say(f'CHECK  N={n}: as trained vs Cell 6b-7 _fom_stack, max |d logit| {d:.1e}')
    d = float((logits(stack(2, FLOW)) - logits(stack(2, TRAINED))).abs().max()); ok &= d == 0.0
    say(f'CHECK  N=2: per-layer flow vs the trained model, max |d logit| {d:.1e}')
    with stack(8, FLOW) as st, torch.enable_grad():
        model(x)
    fresh = (st['fresh'], st['xfresh']) + ((st['pfresh'],) if PM else ())
    ok &= all(f == L for f in fresh)
    say(f'CHECK  N=8 per-layer flow: fresh quadratic / xi{" / phi" if PM else ""} = {fresh} (one per trained layer: {L})')

    # ---- kick share at the trained N = 2 ----
    cap = []
    orig_step = model._fock_layer_step
    def capture(h, h_prev, r, sal, m_b, gamma, dt, layer_idx=0):
        args = (h.detach().clone(), h_prev.detach().clone(), r.clone(), sal.clone())
        out = orig_step(h, h_prev, r, sal, m_b, gamma, dt, layer_idx=layer_idx)
        h, h_prev, r, sal = args
        cap.append(((h.detach().clone(), h_prev.detach().clone(), r.clone(), sal.clone(), m_b, gamma, dt, layer_idx),
                    (out[0].detach().clone(), out[1].detach().clone())))
        return out
    model._fock_layer_step = capture
    try:
        with torch.enable_grad():
            model(torch.from_numpy(xb))
    finally:
        # restore by ASSIGNMENT: the instance attribute is the ladder's depth
        # routing wrapper (install_aniso_depth_routing); popping it would
        # leave V_theta's active layer stuck at the last value
        model._fock_layer_step = orig_step
    assert model._fock_layer_step is orig_step and getattr(model, '_depth_routing_installed', False)
    kick = {}
    for (h, hp, r, sal, m_b, gamma, dt, li), (h_out, hp_out) in cap:
        with torch.enable_grad():
            rep, rep_p, _, _ = model._fock_layer_step(h, hp, r, sal, m_b, gamma, dt, layer_idx=li)
            model.cfg.force_clamp_max = 0.0
            try:
                nok, nok_p, _, _ = model._fock_layer_step(h, hp, r, sal, m_b, gamma, dt, layer_idx=li)
            finally:
                model.cfg.force_clamp_max = None
        d = float((rep.detach() - h_out).abs().max()); ok &= d == 0.0
        step = (h_out - h).norm(dim=-1)
        v_out, v_nok = (h_out - hp_out) / dt, (nok.detach() - nok_p.detach()) / dt
        kick[li] = dict(
            h_share_median=float(((nok.detach() - h_out).norm(dim=-1) / step.clamp_min(1e-12)).median()),
            h_share_p90=float(((nok.detach() - h_out).norm(dim=-1) / step.clamp_min(1e-12)).quantile(0.9)),
            v_share_median=float(((v_nok - v_out).norm(dim=-1) / v_out.norm(dim=-1).clamp_min(1e-12)).median()),
            replay_max_abs_diff=d)
        say(f'CHECK  layer {li}: replay with the normal clamp vs captured output, max |d| {d:.1e}')
    say(f'-> checks {"PASS" if ok else "FAIL"}')
    if not ok:
        raise SystemExit(1)

    say('\nKICK SHARE at the trained N = 2 (first batch, 4 x 512 tokens): |h(no kick) - h| / |h - h_in|')
    for li, k in kick.items():
        say(f'   layer {li}: position median {k["h_share_median"]:.3f} (p90 {k["h_share_p90"]:.3f}); velocity median {k["v_share_median"]:.3f}')

    # ---- Gate 3, both arms ----
    def ev(arm, n):
        tot = 0.0
        with stack(n, arm):
            for xb_, yb_ in batches:
                with torch.enable_grad():
                    _, loss = model(torch.from_numpy(xb_), torch.from_numpy(yb_))
                tot += float(loss)
        return tot / len(batches)

    out = {'kick_share': kick, 'gate3': {}}
    say('\nGATE 3: PPL at N steps of dt = T/N, then the penalty ln(PPL_N / PPL_2) in nats')
    for name, arm in (('as trained', TRAINED), ('per-layer flow', FLOW)):
        loss = {}
        for n in (NS if (SEED is None or name == 'per-layer flow') else (2, 8)):
            loss[n] = loss[2] if (n == 2 and 2 in loss) else ev(arm, n)
            print(f'      {name} N={n}: {np.exp(loss[n]):.2f}', flush=True)
        out['gate3'][name] = dict(loss)
        say(f'   {name:<15} PPL ' + '  '.join(f'N={n}:{np.exp(loss[n]):.2f}' for n in loss))
        say(f'   {"":<15} pen ' + '  '.join(f'N={n}:{loss[n] - loss[2]:.3f}' for n in loss if n != 2))

    txt = '\n'.join(lines)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f'refinement_flow_confirmation_{KIND}' + ('' if SEED is None else f'_seed{SEED}')
    (OUT / f'{stem}_output.txt').write_text(txt + '\n')
    (OUT / f'{stem}.json').write_text(json.dumps(out, indent=1))
    print(f'\nwrote {OUT}/{stem}_output.txt')
