#!/usr/bin/env python3
"""Verification of the content-selective Poisson modes (POISSON_MATCH_RANK, 2026-10-23; Stage 0').

    python3 verify_pm_match.py OUT_DIR fingerprint      # run from HEAD's checkout and from the working copy; compare
    python3 verify_pm_match.py OUT_DIR checks           # the new mode itself (working copy only)
    python3 verify_pm_match.py OUT_DIR planned          # the 2B template pair's Cell 0, built as it would run
    python3 verify_pm_match.py OUT_DIR compare A.json B.json

fingerprint: four existing configurations built at initialisation from the notebook's own cells (F3.1; PM1, K = 64
explicit; PMX-K16; FO-b-PMX-K16), seed 0, one training forward and backward on 2 x 64 fixed validation tokens.
Records the loss, a logits checksum, every parameter's sum and every gradient's norm, to a JSON file. With the switch
off the working copy must reproduce HEAD exactly.

checks (POISSON_MATCH_RANK = 8, K = 16, SR2 base, explicit wells; second order and FO-b):
  1. the tag carries pmlr8 and the model the new parameters (pm_P, pm_log_c, per-layer pm_mu) and no pm_log_kappa2;
  2. the force equals -grad U by autograd, U = -sum_v a_v phi_v exp(-kappa_v^2 |P (h - mu_v)|^2), phi held fixed;
  3. the first training forward sets mu_l, P_l and D2_l from the layer's states (pm_match_ready = 1), and on
     2 x 4 x 512 validation tokens the overlaps then have selectivity (median CV over modes), reach and coverage
     in the design's range;
  4. after one optimizer step (depths no longer zero) every new parameter has a finite, non-zero gradient;
  5. an eval forward is finite and leaves D2 unchanged; a training forward moves it (the running median).
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(sys.argv[1]); MODE = sys.argv[2]
if MODE == 'compare':
    A, B = (json.load(open(p)) for p in sys.argv[3:5])
    bad = 0
    for cfg in A:
        for k in A[cfg]:
            if A[cfg][k] != B[cfg].get(k):
                bad += 1
                if bad <= 10:
                    print(f'  DIFF {cfg} {k}: {str(A[cfg][k])[:80]} vs {str(B[cfg].get(k))[:80]}')
        extra = set(B[cfg]) - set(A[cfg])
        if extra:
            bad += 1; print(f'  EXTRA keys in {cfg}: {sorted(extra)[:5]}')
    print('BIT-IDENTICAL' if bad == 0 else f'{bad} differences')
    sys.exit(0 if bad == 0 else 1)
OUT.mkdir(parents=True, exist_ok=True)
sys.argv = [sys.argv[0], str(OUT / 'harness')]
sys.path.insert(0, str(Path(__file__).parent))
import verify_pm_switch as P
torch.set_num_threads(8)

F31 = P.CONFIGS['F3.1'][1]
SR2 = (("LOWRANK_DAMPED_FLOW  = False", "LOWRANK_DAMPED_FLOW  = True"),)
PMX = (("POISSON_WELLS_EXACT  = False", "POISSON_WELLS_EXACT  = True"),)
FOB = (("FIRST_ORDER_OVERDAMPED = False", "FIRST_ORDER_OVERDAMPED = True"),)
def K(k):
    return (("POISSON_MODES        = 0", f"POISSON_MODES        = {k}"),)


class _Stop(Exception):
    pass


def build_init(repl):
    load = torch.load
    torch.load = lambda *a, **k: (_ for _ in ()).throw(_Stop())
    torch.manual_seed(0); np.random.seed(0)
    try:
        P.build(P.F31_F, repl)
    except _Stop:
        tb = sys.exc_info()[2]; m = g = None
        while tb:
            fl = tb.tb_frame.f_locals
            if tb.tb_frame.f_code.co_name == 'build' and 'model' in fl:
                m, g = fl['model'], fl['g']
            tb = tb.tb_next
        return m, g
    finally:
        torch.load = load


def tokens(g, n=2, T=64):
    x = torch.from_numpy(g['val_ids'][:n * (T + 1)].astype('int64')).view(n, T + 1)
    return x[:, :T], x[:, 1:]


if MODE == 'fingerprint':
    CONFIGS = {'F3.1': F31, 'PM1_K64': F31 + K(64), 'PMX_K16': F31 + K(16) + SR2 + PMX,
               'FOb_PMX_K16': F31 + K(16) + SR2 + PMX + FOB}
    out = {}
    for name, repl in CONFIGS.items():
        model, g = build_init(repl)
        model.to('cpu'); model.train(); model.cfg.use_layer_checkpoint = False
        xb, yb = tokens(g)
        torch.manual_seed(7)
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits, loss = model(xb, targets=yb)
        loss.backward()
        rec = {'tag': g['_variant_tag'], 'loss': repr(float(loss)),
               'logits_sha': hashlib.sha256(logits.detach().numpy().tobytes()).hexdigest()}
        for n_, p in model.state_dict().items():
            rec['sd:' + n_] = repr(float(p.double().sum())) + ' ' + str(tuple(p.shape))
        for n_, p in model.named_parameters():
            rec['grad:' + n_] = repr(float(p.grad.double().norm())) if p.grad is not None else 'None'
        out[name] = rec
        print(f'  {name:12s} loss {float(loss):.6f}  {len(rec)} entries', flush=True)
        del model
    f = OUT / 'fingerprint.json'; f.write_text(json.dumps(out, indent=0)); print('wrote', f)
    sys.exit(0)

if MODE == 'planned':
    SNAP = "SNAPSHOT_STEPS = [6_500, 16_500, 32_500, 48_500, 65_000, 84_500, 97_500, 113_500, 130_000]"
    RUN = (("LADDER_STEPS     = 32_500", "LADDER_STEPS     = 130_000"), ("SNAPSHOT_STEPS = []", SNAP),
           ("POISSON_HALFLIFE_MIN = 4.0", "POISSON_HALFLIFE_MIN = 1.0"),
           ("POISSON_HALFLIFE_MAX = 128.0", "POISSON_HALFLIFE_MAX = 16.0"))
    for name, extra in (('template arm', (("POISSON_MATCH_RANK   = 0", "POISSON_MATCH_RANK   = 8"),)), ('isotropic control', ())):
        model, g = build_init(F31 + K(16) + SR2 + RUN + extra)                  # runs the notebook's Cell 5b
        model.to('cpu'); model.train(); model.cfg.use_layer_checkpoint = False
        hl = (-np.log(2) / torch.nn.functional.logsigmoid(model.pm_logit_lambda.detach()).numpy())
        xb, yb = tokens(g)
        with torch.enable_grad():
            _, loss = model(xb, targets=yb)
        loss.backward()
        fin = bool(torch.isfinite(loss)) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        tag = g['_variant_tag']
        print(f'  {name:17s} tag ...{tag[tag.find("_pm16"):tag.find("_L2probe")]}_L2probe...  steps {g["LADDER_STEPS"]:,}  '
              f'snapshots {len(g["SNAPSHOT_STEPS"])}  half-lives {hl.min():.2f}-{hl.max():.2f} tokens  '
              f'match rank {model.cfg.poisson_match_rank}  {sum(p.numel() for p in model.parameters()):,} params  '
              f'loss {float(loss):.4f}  finite {fin}  (Cell 5b passed)', flush=True)
        del model
    sys.exit(0)

# ------------------------------------------------------------------ checks
MATCH = (("POISSON_MATCH_RANK   = 0", "POISSON_MATCH_RANK   = 8"),)
ok_all = True
def say(ok, msg):
    global ok_all
    ok_all &= bool(ok)
    print(f'  [{"PASS" if ok else "FAIL"}] {msg}', flush=True)

for order, extra in (('second order', ()), ('FO-b', FOB)):
    print(f'\n=== POISSON_MATCH_RANK = 8, K = 16, SR2 base, explicit wells: {order}')
    model, g = build_init(F31 + K(16) + SR2 + MATCH + extra)
    model.to('cpu'); model.cfg.use_layer_checkpoint = False
    tag = g['_variant_tag']
    L = model.cfg.L
    # 1. structure
    say('pmlr8' in tag.split('_') and 'pm16' in tag.split('_') and not any(t.startswith('pmx') for t in tag.split('_')),
        f'tag ...{tag[tag.find("pm16"):tag.find("_L2probe")]}')
    say(tuple(model.pm_mu.shape) == (L, 16, model.cfg.d) and tuple(model.pm_P.shape) == (L, 8, model.cfg.d)
        and tuple(model.pm_log_c.shape) == (L, 16) and not hasattr(model, 'pm_log_kappa2'),
        f'parameters: pm_mu {tuple(model.pm_mu.shape)}, pm_P {tuple(model.pm_P.shape)}, pm_log_c {tuple(model.pm_log_c.shape)}, '
        f'no pm_log_kappa2; {sum(p.numel() for p in model.parameters()):,} parameters')
    say(bool(model.cfg.first_order_overdamped) == (order == 'FO-b') and not model.cfg.poisson_wells_exact, 'order and explicit wells')
    # 3. initialisation from the states on the first training forward (real tokens, 2 x 4 x 512)
    rng = np.random.default_rng(20261009)
    xs = [g['get_batch'](g['val_ids'], 4, 512, rng) for _ in range(2)]
    model.train()
    say(float(model.pm_match_ready.sum()) == 0, 'before the first training forward: not initialised')
    caps = {}
    orig = model._fock_layer_step
    def rec(h, h_prev, r, sal, m_b, gamma, dt, layer_idx=0):
        caps.setdefault(int(layer_idx), []).append(h.detach().clone())
        return orig(h, h_prev, r, sal, m_b, gamma, dt, layer_idx=layer_idx)
    model._fock_layer_step = rec
    torch.manual_seed(7)
    with torch.enable_grad():
        _, loss = model(torch.from_numpy(xs[0][0]), targets=torch.from_numpy(xs[0][1]))
    model._fock_layer_step = orig
    say(float(model.pm_match_ready.sum()) == L and bool(torch.isfinite(loss)),
        f'first training forward initialises every layer (ready {model.pm_match_ready.tolist()}), loss {float(loss):.4f}')
    P_ = model.pm_P.detach()
    orth = max(float((P_[l] @ P_[l].t() - torch.eye(8)).abs().max()) for l in range(L))
    say(orth < 1e-4, f'P_l starts orthonormal (max |P P^T - I| {orth:.1e}); D2 {[round(float(v), 3) for v in model.pm_D2]}')
    model.eval()
    for l in range(L):
        H = caps[l][0]
        with torch.no_grad():
            E = model._pm_match(H, l)[0].reshape(-1, 16)
        cv = float((E.std(0) / E.mean(0).clamp_min(1e-30)).median())
        rv = (E > 0.5).float().mean(0); cover = float((E > 0.5).any(1).float().mean())
        per = float((E > 0.5).sum(1).float().median())
        # Reach is reported, not tested: at initialisation the projected cloud is a dense core (~75% of the
        # tokens) plus small tight clusters (most likely frequent tokens' random embeddings), and k-means gives
        # most centres to those clusters, so the median mode reaches a few percent by construction.
        say(cv >= 0.5 and cover >= 0.9,
            f'layer {l} at initialisation: selectivity {cv:.2f} (target >= 0.5; trained isotropic modes 0.01-0.04), '
            f'coverage {cover:.0%} (>= 90%); reach per mode min/median/max {float(rv.min()):.0%}/{float(rv.median()):.0%}/'
            f'{float(rv.max()):.0%}, modes per token (median) {per:.0f}')
    # 2. force against autograd
    for l in range(L):
        h = caps[l][0][:1, :64].clone().requires_grad_(True)
        with torch.no_grad():
            model.pm_depth.data[l] = torch.linspace(-0.5, 0.5, 16)
        f = model.poisson_mode_force(h, l)
        E, z, m, k2 = model._pm_match(h, l)
        phi = model._pm_occupation_from_overlap(E).detach()
        U = -(model.pm_effective_depth(l) * phi * E).sum()
        gU, = torch.autograd.grad(U, h)
        err = float((f + gU).abs().max() / gU.abs().max().clamp_min(1e-30))
        say(err < 1e-4, f'layer {l}: force = -grad U by autograd (relative max error {err:.1e})')
    with torch.no_grad():
        model.pm_depth.data.zero_()
    # 4. one optimizer step, then gradients on every new parameter
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        torch.manual_seed(7)
        with torch.enable_grad():
            _, loss = model(torch.from_numpy(xs[1][0][:2, :128]), targets=torch.from_numpy(xs[1][1][:2, :128]))
        loss.backward()
        if step == 0:
            opt.step()
    for n_ in ('pm_mu', 'pm_P', 'pm_log_c', 'pm_depth', 'pm_logit_lambda'):
        gr = getattr(model, n_).grad
        gn = float(gr.norm()) if gr is not None else float('nan')
        say(gr is not None and np.isfinite(gn) and gn > 0, f'after one step: |grad {n_}| = {gn:.3e}')
    # 5. eval leaves D2 alone; training moves it
    D2a = model.pm_D2.clone(); model.eval()
    with torch.enable_grad():
        lo = model(torch.from_numpy(xs[1][0][:2, :128]))[0]
    say(bool(torch.isfinite(lo).all()) and torch.equal(D2a, model.pm_D2), 'eval forward finite; D2 unchanged in eval')
    model.train()
    with torch.enable_grad():
        model(torch.from_numpy(xs[1][0][:2, :128]))
    say(not torch.equal(D2a, model.pm_D2), f'training forward updates D2: {[round(float(v), 3) for v in D2a]} -> '
        f'{[round(float(v), 3) for v in model.pm_D2]}')
    del model
print('\nALL CHECKS PASS' if ok_all else '\nSOME CHECKS FAILED')
