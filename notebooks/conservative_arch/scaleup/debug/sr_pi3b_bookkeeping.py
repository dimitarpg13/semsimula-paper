"""SR-pi.3b: is the register bookkeeping what refinement breaks? (protocol SS5.9)

Pre-registered 2026-10-05 before any measurement: refine G2's token dynamics as
Cell 6b-7 does (N = 3, dt = T/3, 'hold' -> layer codes [0, 0, 1]) but run the
register bookkeeping once per TRAINED layer code. Prediction: Gate 3 falls to
<= +300% (from +1,446% on SR-pi.3's tokens), called 50%.

Three arms on the same tokens (SR-pi.3's draw):
  trained   N = 2, the model as trained
  refined   N = 3, Cell 6b-7's refinement exactly (bookkeeping every step)
  held      N = 3, bookkeeping held: at the repeated step 1 the creation gate
            is the identity (readout = the step-0 content, peak weight = the
            step-0 salience, so blend and refresh change nothing), the
            destruction gate is off, and the register state carried into
            step 2 is exactly step 0's output. The token step at step 1 sees
            the same register content and mask as step 0 did.

Usage: python3 sr_pi3b_bookkeeping.py OUT_DIR
"""
import contextlib, json, math, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import gradcheck_vphi_xi_paths as G
import verify_pm_switch as VP
from sr_pi3_crossing import nll_tokens, SEED, NB, BS, BLOCK


@contextlib.contextmanager
def held_stack(model, N, dt):
    """N refined steps, 'hold' policy, register bookkeeping once per trained code."""
    L = model.cfg.L
    orig_sf, orig_dt = model._stack_forward, model.cfg.dt
    cg, dgs = model.creation_gate_qkv, model.destruction_gates

    def patched(h0, x, return_trajectory=False):
        m_b, gamma = model.compute_mass(x), model.gamma
        h, h_prev = h0, h0
        r, s = model._init_registers(h0.shape[0], h0.device)
        last_li, keep = None, {}
        for j in range(N):
            li = min((j * L) // N, L - 1)
            if li != last_li:                      # first step of a trained code: real bookkeeping
                seen = {}
                orig_mask = model._active_mask

                def mask_hook(sal):
                    seen['s_pre'] = sal
                    return orig_mask(sal)
                model._active_mask = mask_hook
                try:
                    h_new, h_prev_out, r, s = model._fock_layer_step(h, h_prev, r, s, m_b, gamma, dt, layer_idx=li)
                finally:
                    del model._active_mask
                keep = dict(r=r, s_pre=seen['s_pre'], s_post=s)
            else:                                  # repeated code: hold the bookkeeping
                r0, s_pre0 = keep['r'], keep['s_pre']
                orig_fp, orig_dg = cg.forward_prefix, dgs[li].forward
                cg.forward_prefix = lambda hh, rr: (r0, s_pre0)
                dgs[li].forward = lambda rr: torch.zeros(rr.shape[:-1], device=rr.device, dtype=rr.dtype)
                try:
                    h_new, h_prev_out, _r, _s = model._fock_layer_step(h, h_prev, r0, s_pre0, m_b, gamma, dt,
                                                                       layer_idx=li)
                finally:
                    cg.forward_prefix = orig_fp; dgs[li].forward = orig_dg
                r, s = keep['r'], keep['s_post']
            last_li = li
            h_prev, h = h_prev_out, h_new
        return h, None

    model._stack_forward = patched
    model.cfg.dt = dt
    try:
        yield
    finally:
        model._stack_forward = orig_sf
        model.cfg.dt = orig_dt


if __name__ == '__main__':
    # RR-B (protocol SS5.18): optional argument "G3'" runs the hardened model
    if len(sys.argv) > 2 and sys.argv[2] == "G3'":
        from exchange_field_probe import ARMS as _EF
        _NAME = "G3'"
        model, g, _ = VP.build(*_EF['G3p'])
    else:
        _NAME = 'G2'
        model, g, _ = VP.build(VP.G2_F, VP.CONFIGS['G2'][1])
    model.cfg.use_layer_checkpoint = False
    nb = json.load(open(G.NB))
    c19 = [''.join(c['source']) for c in nb['cells'] if ''.join(c['source']).startswith('# == Cell 6b-7')][0]
    exec(compile(G.strip(c19[:c19.index('_f7_saved =')]), 'Cell6b7defs', 'exec'), g)
    L, dt = model.cfg.L, float(model.cfg.dt)
    T_int = L * dt
    rng = np.random.default_rng(SEED)
    val = np.load(G.VAL)
    batches = [tuple(torch.from_numpy(a) for a in g['get_batch'](val, BS, BLOCK, rng)) for _ in range(NB)]
    model.eval()
    tot = {'trained': [], 'refined': [], 'held': []}
    # sanity: held_stack at N = L must reproduce the trained model exactly (no repeated codes)
    with held_stack(model, L, dt):
        with torch.enable_grad():
            a, _ = model(batches[0][0][:1, :128])
    with torch.enable_grad():
        b, _ = model(batches[0][0][:1, :128])
    print(f"sanity: held stack at N = L vs the model, max |dlogit| = {(a - b).abs().max().item():.1e}")
    for xb, yb in batches:
        with torch.enable_grad():
            lo, _ = model(xb)
        tot['trained'].append(nll_tokens(lo, yb).flatten())
        with g['_fom_stack'](model, 3, T_int / 3, 'hold'):
            with torch.enable_grad():
                lo, _ = model(xb)
        tot['refined'].append(nll_tokens(lo, yb).flatten())
        with held_stack(model, 3, T_int / 3):
            with torch.enable_grad():
                lo, _ = model(xb)
        tot['held'].append(nll_tokens(lo, yb).flatten())
        print('  batch ' + '  '.join(f"{k} {math.exp(v[-1].mean()):.1f}" for k, v in tot.items()), flush=True)
    ppl = {k: math.exp(torch.cat(v).mean()) for k, v in tot.items()}
    g_ref = 100 * (ppl['refined'] / ppl['trained'] - 1)
    g_held = 100 * (ppl['held'] / ppl['trained'] - 1)
    print(f"\n{_NAME}, {sum(v.numel() for v in tot['trained']):,} tokens: PPL trained {ppl['trained']:.2f}  "
          f"refined {ppl['refined']:.2f} (Gate 3 {g_ref:+.0f}%)  held {ppl['held']:.2f} (Gate 3 {g_held:+.0f}%)")
    verdict = 'HIT' if g_held <= 300 else 'MISS'
    print(f"-> SR-pi.3b {verdict} (prediction: held Gate 3 <= +300%, called 50%)")
    (G.OUT / f"sr_pi3b_{_NAME.replace(chr(39), 'p')}.json").write_text(json.dumps({'ppl': ppl, 'gate3_refined': g_ref, 'gate3_held': g_held,
                                                    'verdict': verdict}, indent=1))
