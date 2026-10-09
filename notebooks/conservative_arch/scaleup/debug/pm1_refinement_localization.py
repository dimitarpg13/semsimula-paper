"""Where does PM1's refinement failure come from? (protocol SS5.15, SS5.19)

Written 2026-10-08 after Cell 6b-7 on the PM1 full run: Gate 3 (refinement at
fixed T) +149% at N = 3 but +16,700% at N = 4; Gate 2 +951%; Gate 1 (velocity
reset) +10,600%, against F3.1's +143% / +374% / +92% / +34%. PM1 has the lowest
stiffness of the L = 2 arms, so the integrator is not the obvious suspect; the
layer-1 wells (about 4.7x the conservative force, acting on every token) are.

Evaluation only, CPU, on the full run's best checkpoint, built through the
ladder notebook's Cells 0-5b. Cell 6b-7's own refinement functions
(_fom_policy_index, _fom_stack) are executed verbatim from the notebook, with
its 'hold' policy and batch seed (the first 4 of its 12 batches).

  1. WELL STRENGTH. All well depths scaled by alpha in {1, 0.75, 0.5, 0}; Gate 3
     at N = 2 (trained), 3, 4. If the N = 3, 4 penalties shrink as alpha falls,
     the wells drive the failure. alpha = 1 must reproduce Colab's ratios.
     alpha < 1 is not a trained model; it previews what capping the wells might
     buy, and what it costs at the trained N = 2.
     With an option cap<c> (added 2026-10-08, before the PM1-cap arm), the bounded
     depth a = c tanh(a_raw / c) is applied to the trained depths instead of
     alpha: a post-hoc preview of the PM1-cap reparameterisation, not that arm.
  2. GATE 1 PER LAYER. The velocity entering layer 0 only, layer 1 only, or
     both set to zero (h_prev := h), wells on (alpha = 1) and off (alpha = 0).
     Says whose momentum the model depends on.

Usage: python3 pm1_refinement_localization.py OUT_DIR FOLDER [cap<c>]
"""
import ast, contextlib, io, json, sys
from pathlib import Path

import numpy as np
import torch

N_BATCH, BATCH, BLOCK = 4, 4, 512
ALPHAS = (1.0, 0.75, 0.5, 0.0)
NS = (2, 3, 4)

if __name__ == '__main__':
    OUT = Path(sys.argv[1]); FOLDER = sys.argv[2]
    CAP = float(sys.argv[3][3:].replace('p', '.')) if len(sys.argv) > 3 else None
    SUF = f'_cap{CAP:g}'.replace('.', 'p') if CAP is not None else ''
    sys.argv = [sys.argv[0], str(OUT / 'harness')]
    sys.path.insert(0, str(Path(__file__).parent))
    import verify_pm_switch as P
    _, repl = P.CONFIGS['F3.1']
    model, g, _ = P.build(FOLDER, repl + P.PM_ON)       # loads FOLDER's _best.pt
    model.to('cpu'); model.eval()
    assert getattr(model, 'pm_mu', None) is not None

    # Cell 6b-7's refinement machinery, verbatim
    nb = json.load(open(P.G.NB))
    src = [''.join(c['source']) for c in nb['cells'] if ''.join(c['source']).startswith('# == Cell 6b-7:')][0]
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    exec(compile(ast.Module(body=keep, type_ignores=[]), 'Cell6b7_defs', 'exec'), g)
    _fom_stack = g['_fom_stack']

    rng = np.random.default_rng(20260920)                # the cell's seed
    val = g['val_ids']; get_batch = g['get_batch']
    batches = [get_batch(val, BATCH, BLOCK, rng) for _ in range(12)][:N_BATCH]

    def ev():
        tot = 0.0
        for xb, yb in batches:
            x, y = torch.from_numpy(xb), torch.from_numpy(yb)
            with torch.enable_grad():                       # the forces use autograd
                _, loss = model(x, y)
            tot += float(loss)
        return tot / len(batches)

    L, DT = model.cfg.L, model.cfg.dt
    T = L * DT
    depth0 = model.pm_depth.detach().clone()
    out = {'refinement': {}, 'gate1': {}}
    if CAP is not None:                                  # preview: the cap replaces alpha
        model.cfg.poisson_depth_cap = CAP
        ALPHAS = (1.0,)
    lines = [f'PM1 refinement localization: {FOLDER[-60:]}' + (f'; depths capped post hoc, a = {CAP:g} tanh(a_raw / {CAP:g})' if CAP is not None else ''),
             f'   {N_BATCH} x {BATCH} x {BLOCK} tokens (the first {N_BATCH} of Cell 6b-7\'s batches); policy hold; T = {T:g}']

    lines.append('\n1. WELL STRENGTH x alpha: PPL at N steps of dt = T/N, and the change from N = 2')
    lines.append(f'   {"alpha":>6}' + ''.join(f'{"N="+str(n):>12}' for n in NS) + ''.join(f'{"d N="+str(n):>12}' for n in NS[1:]))
    try:
        for a in ALPHAS:
            with torch.no_grad():
                model.pm_depth.copy_(depth0 * a)
            ppl = {}
            for n in NS:
                with _fom_stack(model, n, T / n, 'hold'):
                    ppl[n] = float(np.exp(ev()))
            out['refinement'][a] = ppl
            lines.append(f'   {a:>6.2f}' + ''.join(f'{ppl[n]:>12.2f}' for n in NS)
                         + ''.join(f'{100*(ppl[n]/ppl[2]-1):>+11.0f}%' for n in NS[1:]))
            print(lines[-1], flush=True)

        lines.append('\n2. GATE 1 PER LAYER: velocity entering the layer set to zero (h_prev := h), at the trained N = 2')
        orig_step = model._fock_layer_step
        for a in ALPHAS[:1] + ALPHAS[-1:] if CAP is None else ALPHAS:
            with torch.no_grad():
                model.pm_depth.copy_(depth0 * a)
            row = {}
            for name, layers in (('none', ()), ('layer 0', (0,)), ('layer 1', (1,)), ('both', (0, 1))):
                def step(h, h_prev, r, sal, m_b, gamma, dt, layer_idx=0, _ls=layers, **kw):
                    if layer_idx in _ls:
                        h_prev = h
                    return orig_step(h, h_prev, r, sal, m_b, gamma, dt, layer_idx=layer_idx, **kw)
                model._fock_layer_step = step
                try:
                    row[name] = float(np.exp(ev()))
                finally:
                    model._fock_layer_step = orig_step
            out['gate1'][a] = row
            lines.append(f'   wells x{a:g}: ' + '   '.join(f'{k} {v:.2f}' + (f' ({100*(v/row["none"]-1):+.0f}%)' if k != 'none' else '')
                                                       for k, v in row.items()))
            print(lines[-1], flush=True)
    finally:
        with torch.no_grad():
            model.pm_depth.copy_(depth0)

    txt = '\n'.join(lines)
    print('\n' + txt)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'pm1_refinement_localization{SUF}_output.txt').write_text(txt + '\n')
    (OUT / f'pm1_refinement_localization{SUF}.json').write_text(json.dumps(out, indent=1))
