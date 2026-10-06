"""SR2 switch check: exact damped stiff-mode flow (book Prop 45; protocol SS5.19 Test 2).

OFF (LOWRANK_DAMPED_FLOW = False) must be bit-identical to the committed code on
every configuration that may run while it is pushed: G2, F3.1 and G3'. Built
through the ladder notebook's own Cells 0-5b on fixed trained weights and real
validation tokens; records eval logits, train-mode logits (same seed) and every
parameter gradient.

  python3 verify_sr2_switch.py OUT_DIR --dump     # with HEAD files in place
  python3 verify_sr2_switch.py OUT_DIR            # working copy: compare + ON checks

ON, unit level (float64 unless stated):
  1. gamma = 0: lowrank_damped_substep equals lowrank_cfc_substep
  2. one mode against a fine RK4 solution of x'' + g x' + w0^2 x = a, across
     underdamped, critical, overdamped, w0 = 0 and the small-w0 series branch
  3. group property: two half steps (force re-evaluated at the midpoint, as the
     layer step does) equal one full step -- the linear stiff dynamics do not
     depend on how T is cut
ON, model level (F3.1's Cell 0 plus LOWRANK_DAMPED_FLOW = True, F3.1's weights):
  4. tag: F3.1's tag plus 'sr2'; Cell 5b banner; no new parameters
  5. gamma set to 0 on both models: flag ON equals flag OFF (float32 tolerance)
  6. gamma = 0.1 (trained): finite logits and gradients; size of the change
     from the split scheme (descriptive)
  7. causality: replacing tokens at t >= 128 leaves logits at t < 128 unchanged
  8. the model refuses SR2 without a fixed gamma or with another integrator
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import gradcheck_vphi_xi_paths as G
import verify_pm_switch as VP

SR2_ON = (("LOWRANK_DAMPED_FLOW  = False", "LOWRANK_DAMPED_FLOW  = True"),)


def rk4_mode(w2, g, a, w0, t, n=20000):
    """x'' + g x' + w2 x = a, x(0) = 0, x'(0) = w0, by RK4 in float64."""
    h = t / n
    x, v = 0.0, w0
    f = lambda x, v: (v, a - g * v - w2 * x)
    for _ in range(n):
        k1 = f(x, v); k2 = f(x + h / 2 * k1[0], v + h / 2 * k1[1])
        k3 = f(x + h / 2 * k2[0], v + h / 2 * k2[1]); k4 = f(x + h * k3[0], v + h * k3[1])
        x += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        v += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return x, v


if __name__ == '__main__':
    x, y = VP.tokens()
    ref = G.OUT / 'sr2_head_ref.pt'
    if '--dump' in sys.argv:
        recs = {}
        for name, (folder, repl) in VP.CONFIGS.items():
            model, g, _ = VP.build(folder, repl)
            recs[name] = VP.record(model, x, y)
            print(f"  dumped {name}: tag ...{g['_variant_tag'][g['_variant_tag'].find('cgqk'):]}  loss {recs[name]['loss']:.6f}")
        torch.save(recs, ref)
        sys.exit(0)

    ok = []
    h = torch.load(ref)
    print("OFF (LOWRANK_DAMPED_FLOW = False): HEAD vs working copy, fixed trained weights")
    for name, (folder, repl) in VP.CONFIGS.items():
        model, g, _ = VP.build(folder, repl)
        rec = VP.record(model, x, y)
        assert set(h[name]['grads']) == set(rec['grads']), name
        de = (h[name]['eval'] - rec['eval']).abs().max().item()
        dt = (h[name]['train'] - rec['train']).abs().max().item()
        dg = max((h[name]['grads'][k] - rec['grads'][k]).abs().max().item() for k in rec['grads'])
        ok.append(de == dt == dg == 0)
        print(f"  {name:5s} eval {de:.1e}  train {dt:.1e}  grads {dg:.1e}  ({len(rec['grads'])} params)  -> "
              f"{'IDENTICAL' if ok[-1] else 'DIFFERS'}")
        del model

    from cfc_baoab import lowrank_cfc_substep, lowrank_damped_substep, damped_mode_coefficients
    torch.manual_seed(0)
    B, T, d, q = 2, 8, 384, 16
    U, _ = torch.linalg.qr(torch.randn(B, T, d, q, dtype=torch.float64))
    kappa = torch.rand(B, T, q, dtype=torch.float64) * 0.21
    kappa[..., 0] = 0.0                                      # a soft mode, as trained
    m = 0.5 + 1.5 * torch.rand(B, T, 1, dtype=torch.float64)
    hh = torch.randn(B, T, d, dtype=torch.float64); vv = torch.randn(B, T, d, dtype=torch.float64) * 0.1
    sL = torch.randn(B, T, d, dtype=torch.float64) * 0.05
    Lmat = torch.einsum('...dq,...q,...eq->...de', U, kappa, U)
    force = lambda hx: sL - torch.einsum('...de,...e->...d', Lmat, hx)

    print("\nON, unit level (float64; q = 16 modes, kappa in [0, 0.21], m in [0.5, 2], dt = 2)")
    a0 = lowrank_cfc_substep(hh, vv, U, kappa, force(hh), m, 2.0)
    a1 = lowrank_damped_substep(hh, vv, U, kappa, force(hh), m, 0.0, 2.0)
    e1 = max((a0[0] - a1[0]).abs().max().item(), (a0[1] - a1[1]).abs().max().item())
    ok.append(e1 < 1e-9)
    print(f"1. gamma = 0 vs lowrank_cfc_substep: max |diff| {e1:.1e} -> {ok[-1]}")

    cases = [('underdamped', 0.2, 0.1), ('critical', 0.0025, 0.1), ('overdamped', 1e-3, 0.5),
             ('omega0 = 0', 0.0, 0.1), ('omega0 = 0, strong damping', 0.0, 2.0),
             ('series branch', 1e-6, 0.1), ('stiff, undamped', 0.21 / 0.5, 0.0)]
    worst = 0.0
    for lab, w2, gm in cases:
        for t in (2.0, 4.0):
            E12, E22, P, Q = damped_mode_coefficients(torch.tensor([w2], dtype=torch.float64), gm, t)
            for a, w0 in ((0.3, 0.0), (0.0, 0.7), (-0.2, 0.4)):
                xr, vr = rk4_mode(w2, gm, a, w0, t)
                xe = float(E12 * w0 + Q * a); ve = float(E22 * w0 + P * a)
                worst = max(worst, abs(xe - xr) / max(1.0, abs(xr)), abs(ve - vr) / max(1.0, abs(vr)))
    ok.append(worst < 1e-8)
    print(f"2. exact coefficients vs RK4 (20,000 steps), {len(cases)} regimes x 2 horizons x 3 initial "
          f"conditions: max error {worst:.1e} -> {ok[-1]}")

    gm = 0.1
    full = lowrank_damped_substep(hh, vv, U, kappa, force(hh), m, gm, 4.0)
    hm, vm = lowrank_damped_substep(hh, vv, U, kappa, force(hh), m, gm, 2.0)
    two = lowrank_damped_substep(hm, vm, U, kappa, force(hm), m, gm, 2.0)
    e3 = max((full[0] - two[0]).abs().max().item(), (full[1] - two[1]).abs().max().item())
    sp0 = lowrank_cfc_substep(hh, vv, U, kappa, force(hh), m, 4.0)
    ok.append(e3 < 1e-10)
    print(f"3. group property, gamma = 0.1: one step of 4 vs two steps of 2, max |diff| {e3:.1e} -> {ok[-1]}")

    print("\nON, model level (F3.1's Cell 0 + LOWRANK_DAMPED_FLOW = True, F3.1's trained weights)")
    off_model, g_off, _ = VP.build(VP.F31_F, VP.CONFIGS['F3.1'][1])
    model, g, log = VP.build(VP.F31_F, VP.CONFIGS['F3.1'][1] + SR2_ON)
    tag, tag_off = g['_variant_tag'], g_off['_variant_tag']
    new = sorted(set(dict(model.named_parameters())) ^ set(dict(off_model.named_parameters())))
    ok.append(tag == tag_off.replace('_L2probe', '_sr2_L2probe') and not new
              and 'EXACT DAMPED STIFF-MODE FLOW' in log and model.cfg.lowrank_damped_flow)
    print(f"4. tag ...{tag[tag.find('cgqk'):]}\n   parameter-set difference {new}; banner printed: "
          f"{'EXACT DAMPED' in log}; gamma {model.cfg.fixed_gamma:g} -> {ok[-1]}")

    for mm in (model, off_model):
        mm._gamma_value = 0.0
        mm.cfg.use_layer_checkpoint = False
    a = VP.record(model, x, y); b = VP.record(off_model, x, y)
    d5 = max((a['eval'] - b['eval']).abs().max().item(), (a['train'] - b['train']).abs().max().item())
    sc = b['eval'].abs().max().item()
    ok.append(d5 / sc < 1e-4)
    print(f"5. gamma = 0: max |dlogit| flag ON vs OFF {d5:.1e} (max |logit| {sc:.1f}) -> {ok[-1]}")
    for mm in (model, off_model):
        mm._gamma_value = float(mm.cfg.fixed_gamma)

    a = VP.record(model, x, y); b = VP.record(off_model, x, y)
    fin = bool(torch.isfinite(a['eval']).all() and all(torch.isfinite(v).all() for v in a['grads'].values()))
    ok.append(fin and set(a['grads']) == set(b['grads']))
    dl = (a['eval'] - b['eval']).abs()
    print(f"6. gamma = {model.cfg.fixed_gamma:g}: finite logits and grads {fin}; same {len(a['grads'])} params get grads; "
          f"loss ON {a['loss']:.4f} vs OFF {b['loss']:.4f}; |dlogit| median {dl.median():.2e} max {dl.max():.2e} "
          f"(descriptive: the weights were trained under the split scheme) -> {ok[-1]}")

    model.eval()
    rng = np.random.default_rng(0)
    x2 = x.clone(); x2[:, 128:] = torch.from_numpy(rng.integers(0, 50257, size=(2, x.shape[1] - 128)))
    with torch.enable_grad():
        l1, _ = model(x); l2, _ = model(x2)
    dc = (l1[:, :128] - l2[:, :128]).abs().max().item()
    ok.append(dc == 0.0)
    print(f"7. causality: tokens replaced at t >= 128, max |dlogit| at t < 128 = {dc:.1e} -> {ok[-1]}")

    import copy
    refused = []
    for field, val in (('fixed_gamma', None), ('integrator', 'baoab_cfc')):
        c = copy.deepcopy(model.cfg); setattr(c, field, val)
        try:
            type(model)(c)
            refused.append(False)
        except ValueError:
            refused.append(True)
    ok.append(all(refused))
    print(f"8. refuses learned gamma {refused[0]}, refuses integrator='baoab_cfc' {refused[1]} -> {ok[-1]}")

    print(f"\n{'ALL CHECKS PASS' if all(ok) else 'FAILURES: ' + str([i for i, v in enumerate(ok) if not v])}")
