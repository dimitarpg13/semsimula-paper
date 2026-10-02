"""Check pair_potential='none' (protocol runs 10 and 11, the V_phi x Fock factorial).

Builds multi-xi SPLM (REVERSE_CHANNEL=False) and Fock-SPLM (True) through the
ladder notebook's own Cells 0-5b at fresh init, each with XI_GRAD_PATH default
and live, and reports:

  1. tag, Cell 5b banner, parameter count against the PARF arm of the same
     config (V_phi, the score head and the per-layer scale must be gone)
  2. eval and train forward run; train loss.backward() runs
  3. forward identical between xi default and live (same init seed)
  4. one real layer step: gradient into EARLIER tokens -- in multi-xi SPLM,
     xi is the only inter-token channel, so it must be exactly 0 as default
     and > 0 live

Usage: python3 verify_pair_potential_none.py OUT_DIR
"""
import contextlib, io, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import gradcheck_vphi_xi_paths as G
from verify_vphi_xi_grad_path import c5b, step_source_grad


def build(rc, pp, xi):
    import os
    os.chdir(G.REPO / 'notebooks/conservative_arch/scaleup')
    g = {'__name__': '__main__'}
    c0 = G.cells['Cell 0:']
    for old, new in (('REVERSE_CHANNEL              = True', f'REVERSE_CHANNEL              = {rc}'),
                     ("PAIR_POTENTIAL  = 'sparse_topk'", f"PAIR_POTENTIAL  = {pp!r}"),
                     ("XI_GRAD_PATH           = 'default'", f"XI_GRAD_PATH           = {xi!r}")):
        assert c0.count(old) == 1, old
        c0 = c0.replace(old, new)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(G.strip(c0), 'Cell0', 'exec'), g)
        exec(compile(G.strip(G.cells['Cell 1:']), 'Cell1', 'exec'), g)
        g['CKPT_DIR'] = G.OUT / 'ckpt_empty'; g['CKPT_DIR'].mkdir(parents=True, exist_ok=True)
        g['RESULTS_DIR'] = G.OUT / 'results'; g['RESULTS_DIR'].mkdir(parents=True, exist_ok=True)
        g['DATA_DIR'] = G.OUT / 'data'; g['GDRIVE_ROOT'] = G.OUT
        exec(compile(G.strip(G.cells['Cell 1b:']), 'Cell1b', 'exec'), g)
        g['DATA_DIR'] = G.OUT / 'data'
        exec(compile(G.strip(G.cells['Cell 2:']), 'Cell2', 'exec'), g)
        exec('from data_module import get_batch', g)
        g['val_ids'] = np.load(G.VAL); g['train_ids'] = g['val_ids']
        exec(compile(G.strip(G.cells['Cell 4:']), 'Cell4', 'exec'), g)
        torch.manual_seed(0)
        exec(compile(G.strip(G.cells['Cell 5:']), 'Cell5', 'exec'), g)
        g['PROBE_MAX_STEPS'] = g.get('PROBE_MAX_STEPS')
        exec(compile(G.strip(c5b), 'Cell5b', 'exec'), g)
    banner = [l.strip() for l in out.getvalue().splitlines()
              if 'NO PAIR POTENTIAL' in l or 'parameters.' in l or 'SOURCE-GRADIENT' in l]
    return g['model'], g['_variant_tag'], banner


if __name__ == '__main__':
    val = np.load(G.VAL)
    rng = np.random.default_rng(20261001)
    starts = rng.integers(0, len(val) - 513, size=2)
    x = torch.from_numpy(np.stack([val[s:s + 512] for s in starts]).astype(np.int64))
    y = torch.from_numpy(np.stack([val[s + 1:s + 513] for s in starts]).astype(np.int64))
    for rc, name in ((False, 'multi-xi SPLM (run 10)'), (True, 'Fock-SPLM (run 11)')):
        parf, _, _ = build(rc, 'sparse_topk', 'default')
        n_parf = parf.num_params()
        print(f"\n== {name}   (PARF arm of the same config: {n_parf:,} params)")
        ref = None
        for xi in ('default', 'live'):
            m, tag, banner = build(rc, 'none', xi)
            assert m.V_phi is None and m.score_head is None and m.raw_v_phi_scale is None
            assert not any(k.startswith(('V_phi.', 'score_head.', 'raw_v_phi_scale')) for k in m.state_dict())
            m.eval()
            with torch.enable_grad():
                lo, l = m(x, y)
            m.train(); torch.manual_seed(7)
            lo_tr, l_tr = m(x, y); l_tr.backward()
            ref = lo.detach() if ref is None else ref
            src = [step_source_grad(m, x, l_) for l_ in range(m.cfg.L)]
            print(f"   xi={xi:7s} tag ...{tag[tag.find('cgqk'):][:58]}")
            for b in banner:
                print(f"      5b: {b}")
            print(f"      params {m.num_params():,} ({m.num_params() - n_parf:+,} vs PARF)   "
                  f"eval loss {l.item():.4f}  train loss {l_tr.item():.4f}  "
                  f"max|dlogit| vs xi=default {(lo.detach() - ref).abs().max().item():.1e}")
            for l_, (e, s_) in enumerate(src):
                print(f"      layer {l_} step: grad -> earlier tokens {e:.4e}   -> own token {s_:.4e}")
