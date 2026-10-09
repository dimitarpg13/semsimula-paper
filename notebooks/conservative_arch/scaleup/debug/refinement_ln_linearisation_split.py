"""What fails at fine steps: LayerNorm placement or re-linearisation? (protocol SS5.19)

Written 2026-10-09 after SR2 on F3.1. SR2 halves F3.1's Gate 3 penalty at N = 3
(0.405 against 0.889 nats) but barely at N = 8 (2.110 against 2.275): the
fine-step penalty belongs to something both integrators share. Two candidates,
one from each descriptive cell:

  (a) LN PLACEMENT (6b-9). The step ends in one LayerNorm projection after a
      move of about 6 x |h_in| at layer 0. Refined to N steps, the projection
      runs N times after shorter moves.
  (b) RE-LINEARISATION (6b-13). The A substeps integrate the low-rank
      quadratic of V_theta frozen at the step's starting state, exactly. With
      omega*dt near 9 (SR2, layer 0), the trained map leans on one frozen
      quadratic per step; refined, the quadratic is re-taken N times.

Gate 3 is re-run (policy 'hold', T fixed, N in NS) under four arms:

  as trained   every substep projects and re-linearises (= Cell 6b-7's Gate 3)
  LN once      the projection runs only at the end of each trained layer's
               share of the interval ([0,0,0,0,1,1,1,1] at N = 8 projects
               after substeps 4 and 8)
  freeze       the substeps of one trained layer's share reuse the quadratic
               (G, G mu, hence U, kappa) taken at the share's first substep;
               the force itself (B kick, analytic V_theta at h_mid) is the
               real one, so only the split between exact flow and kick moves
  both         LN once + freeze
  xi frozen    (option 'xi', added 2026-10-09 after the first F3.1 pass found
               neither (a) nor (b) explains the fine-step penalty) the
               context xi -- which feeds V_theta's wells and V_phi -- is
               taken once per trained layer's share, at its first substep;
               what is then left to change with N is the force's nonlinear
               residual, evaluated at each substep's midpoint
  all three    LN once + freeze + xi frozen

At N = 2 each share is one substep, so every arm IS the trained model.
Evaluation only, CPU, on FOLDER's _best.pt, built through the ladder
notebook's Cells 0-5b. Cell 6b-7's policy function is executed verbatim; the
stack loop mirrors its _fom_stack (Fock branch, no layer checkpointing, which
changes memory, not values) and is checked against it bit for bit.

Usage: python3 refinement_ln_linearisation_split.py OUT_DIR FOLDER [sr2] [xi]
  'xi' runs 'as trained', 'xi frozen' and 'all three' instead of the four arms.
"""
import ast, contextlib, json, sys
from pathlib import Path

import numpy as np
import torch

N_BATCH, BATCH, BLOCK = 4, 4, 512
NS = (2, 3, 4, 8)
ARMS = {'as trained': (False, False, False), 'LN once': (True, False, False),
        'freeze': (False, True, False), 'both': (True, True, False)}
ARMS_XI = {'as trained': (False, False, False), 'xi frozen': (False, False, True),
           'all three': (True, True, True)}


@contextlib.contextmanager
def split_stack(model, N, dt, policy_index, ln_once, freeze, xi_freeze=False, phi_freeze=False):
    """6b-7's _fom_stack (policy 'hold'), plus the switches. phi_freeze (PM1,
    added 2026-10-09 for FLOW-C): the occupation phi is taken at the share's
    first force evaluation; the token's own overlap E stays live."""
    L = model.cfg.L
    lis = [policy_index(j, N, L, 'hold') for j in range(N)]
    st = {'proj': True, 'start': True, 'calls': 0, 'cache': {}, 'fresh': 0,
          'inloop': False, 'xcalls': 0, 'xcache': {}, 'xfresh': 0,
          'pcalls': 0, 'pcache': {}, 'pfresh': 0}
    has_pm = getattr(model, 'pm_mu', None) is not None
    orig_occ = model.poisson_mode_occupation if has_pm else None

    def occupation(h):
        E, phi = orig_occ(h)
        if not (phi_freeze and st['inloop']):
            return E, phi
        k = st['pcalls']; st['pcalls'] += 1
        if st['start'] or k not in st['pcache']:
            st['pcache'][k] = phi
            st['pfresh'] += 1
        return E, st['pcache'][k]
    orig_project = model._project
    vt = model.V_theta
    orig_htl = vt.harmonic_terms_lowrank

    def project(h):
        return orig_project(h) if st['proj'] else h

    def htl(xis, h, *, comps=None):
        k = st['calls']; st['calls'] += 1
        if not freeze or st['start'] or k not in st['cache']:
            st['cache'][k] = orig_htl(xis, h, comps=comps)
            st['fresh'] += 1
        return st['cache'][k]

    def xi_hook(module, inp, out):
        if not (xi_freeze and st['inloop']):
            return None
        k = st['xcalls']; st['xcalls'] += 1
        if st['start'] or k not in st['xcache']:
            st['xcache'][k] = out
            st['xfresh'] += 1
            return None
        return st['xcache'][k]

    def stack(h0, x, return_trajectory=False):
        m_b = model.compute_mass(x)
        gamma = model.gamma
        h = h_prev = h0
        r, sal = model._init_registers(h0.shape[0], h0.device)
        for j, li in enumerate(lis):
            st['start'] = j == 0 or lis[j - 1] != li
            st['proj'] = (not ln_once) or j == N - 1 or lis[j + 1] != li
            st['calls'] = st['xcalls'] = st['pcalls'] = 0
            st['inloop'] = True
            h, h_prev, r, sal = model._fock_layer_step(h, h_prev, r, sal, m_b, gamma, dt, layer_idx=li)
            st['inloop'] = False
        st['proj'] = True
        return h, None

    orig_sf, orig_dt = model._stack_forward, model.cfg.dt
    model._stack_forward, model.cfg.dt = stack, dt
    saved = {(o, k): o.__dict__[k] for o, k in ((model, '_project'), (vt, 'harmonic_terms_lowrank'),
                                                 (model, 'poisson_mode_occupation')) if k in o.__dict__}
    model._project, vt.harmonic_terms_lowrank = project, htl
    hook = model.xi_module.register_forward_hook(xi_hook)
    if has_pm:
        model.poisson_mode_occupation = occupation
    try:
        yield st
    finally:
        hook.remove()
        model._stack_forward, model.cfg.dt = orig_sf, orig_dt
        for o, k in ((model, '_project'), (vt, 'harmonic_terms_lowrank'), (model, 'poisson_mode_occupation')):
            if (o, k) in saved:                     # an instance wrapper was there: put it back
                setattr(o, k, saved[(o, k)])
            else:
                o.__dict__.pop(k, None)
        assert model._project == orig_project and vt.harmonic_terms_lowrank == orig_htl


if __name__ == '__main__':
    OUT = Path(sys.argv[1]); FOLDER = sys.argv[2]; SR2 = 'sr2' in sys.argv[3:]; XI = 'xi' in sys.argv[3:]
    sys.argv = [sys.argv[0], str(OUT / 'harness')]
    sys.path.insert(0, str(Path(__file__).parent))
    import verify_pm_switch as P
    repl = P.CONFIGS['F3.1'][1] + ((("LOWRANK_DAMPED_FLOW  = False", "LOWRANK_DAMPED_FLOW  = True"),) if SR2 else ())
    model, g, _ = P.build(FOLDER, repl)
    model.to('cpu'); model.eval()
    assert bool(getattr(model.cfg, 'lowrank_damped_flow', False)) == SR2
    assert model.cfg.integrator == 'baoab_cfc_lowrank' and model.reverse_ch is None and model.cfg.ln_after_step

    nb = json.load(open(P.G.NB))
    src = [''.join(c['source']) for c in nb['cells'] if ''.join(c['source']).startswith('# == Cell 6b-7:')][0]
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    exec(compile(ast.Module(body=keep, type_ignores=[]), 'Cell6b7_defs', 'exec'), g)
    policy_index, fom_stack = g['_fom_policy_index'], g['_fom_stack']

    rng = np.random.default_rng(20260920)                    # the cell's seed
    batches = [g['get_batch'](g['val_ids'], BATCH, BLOCK, rng) for _ in range(12)][:N_BATCH]

    def ev(n_batches=N_BATCH):
        tot = 0.0
        for xb, yb in batches[:n_batches]:
            with torch.enable_grad():                        # the forces use autograd
                _, loss = model(torch.from_numpy(xb), torch.from_numpy(yb))
            tot += float(loss)
        return tot / n_batches

    L, DT = model.cfg.L, model.cfg.dt
    T = L * DT
    lines = [f'Refinement split, LN placement vs re-linearisation: {"SR2" if SR2 else "F3.1"}  ...{FOLDER[-70:]}',
             f'   {N_BATCH} x {BATCH} x {BLOCK} tokens (the first {N_BATCH} of Cell 6b-7\'s batches); policy hold; T = {T:g}']
    say = lambda s='': (print(s, flush=True), lines.append(s))

    # ---- checks: the loop is 6b-7's; at N = 2 every arm is the trained model ----
    xb, yb = batches[0]
    x = torch.from_numpy(xb[:2, :256])
    def logits(ctx):
        with ctx, torch.enable_grad():
            return model(x)[0].detach()
    ok = True
    for n in (2, 3):
        a = logits(fom_stack(model, n, T / n, 'hold'))
        b = logits(split_stack(model, n, T / n, policy_index, False, False))
        d = float((a - b).abs().max()); ok &= d == 0.0
        say(f'CHECK  N={n}: switches off vs Cell 6b-7 _fom_stack, max |d logit| {d:.1e}')
    base = logits(split_stack(model, 2, T / 2, policy_index, False, False))
    for name, (lo, fr, xf) in {**ARMS, **ARMS_XI}.items():
        d = float((logits(split_stack(model, 2, T / 2, policy_index, lo, fr, xf)) - base).abs().max()); ok &= d == 0.0
        say(f'CHECK  N=2 {name:<10}: max |d logit| against the trained model {d:.1e}')
    with split_stack(model, 8, T / 8, policy_index, False, True) as st, torch.enable_grad():
        model(x)
    ok &= st['fresh'] == L
    say(f'CHECK  freeze at N=8 takes {st["fresh"]} fresh linearisations (one per trained layer: {L})')
    with split_stack(model, 8, T / 8, policy_index, False, False) as st, torch.enable_grad():
        model(x)
    ok &= st['fresh'] == 8
    say(f'CHECK  as trained at N=8 takes {st["fresh"]} (one per substep: 8)')
    with split_stack(model, 8, T / 8, policy_index, False, False, True) as st, torch.enable_grad():
        model(x)
    ok &= st['xfresh'] == L
    say(f'CHECK  xi frozen at N=8 takes {st["xfresh"]} fresh xi (one per trained layer: {L})')
    say(f'-> checks {"PASS" if ok else "FAIL"}')
    if not ok:
        raise SystemExit(1)

    # ---- Gate 3 under the four arms ----
    arms = ARMS_XI if XI else ARMS
    say('\nGATE 3 per arm: PPL at N steps of dt = T/N, and the penalty ln(PPL_N / PPL_2) in nats')
    say(f'   {"arm":<12}' + ''.join(f'{"N="+str(n):>10}' for n in NS) + ''.join(f'{"pen N="+str(n):>12}' for n in NS[1:]))
    out = {}
    for name, (lo, fr, xf) in arms.items():
        loss = {}
        for n in NS:
            with split_stack(model, n, T / n, policy_index, lo, fr, xf):
                loss[n] = ev()
        out[name] = {n: {'loss': loss[n], 'ppl': float(np.exp(loss[n]))} for n in NS}
        say(f'   {name:<12}' + ''.join(f'{np.exp(loss[n]):>10.2f}' for n in NS)
            + ''.join(f'{loss[n] - loss[2]:>12.3f}' for n in NS[1:]))

    say('\nSHARE of the as-trained penalty removed by each arm')
    for name in [a for a in arms if a != 'as trained']:
        say(f'   {name:<12}' + ''.join(
            f'   N={n}: {100 * (1 - (out[name][n]["loss"] - out[name][2]["loss"]) / (out["as trained"][n]["loss"] - out["as trained"][2]["loss"])):>+5.0f}%'
            for n in NS[1:]))

    txt = '\n'.join(lines)
    tag = ('sr2' if SR2 else 'f31') + ('_xi' if XI else '')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'refinement_ln_linearisation_split_{tag}_output.txt').write_text(txt + '\n')
    (OUT / f'refinement_ln_linearisation_split_{tag}.json').write_text(json.dumps(out, indent=1))
    print(f'\nwrote {OUT}/refinement_ln_linearisation_split_{tag}_output.txt')
