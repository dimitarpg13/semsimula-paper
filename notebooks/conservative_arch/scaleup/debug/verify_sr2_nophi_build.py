"""Build check for SR2 trained without the pair force (2026-11-21; stage 3, lever T1 of
companion_notes/productization/Conservative_Core_Productization_Plan.md).

Builds SR2 and the new arm through the ladder notebook's own Cells 0-5b at fresh initialisation and reports, for each:

  1. the tag, the Cell 5b lines about the pair potential and the parameter count; for the new arm, that V_phi, its score
     head and its per-layer scale are gone
  2. eval and train forward, train loss.backward(), all finite
  3. CPU time of one forward and backward on the same batch (the ratio is a first estimate of the GPU step's)
  4. one real layer step's gradient into earlier tokens: with the pair force gone, the context channels (live) are the
     only pathway between tokens, so it must stay > 0

The new arm's Cell 0 is SR2's six deltas with PAIR_POTENTIAL = 'none', except VPHI_GRAD_PATH, which stays 'default' as
Cell 0 requires when there is no V_phi to make live.

Usage: python3 verify_sr2_nophi_build.py OUT_DIR
"""
import contextlib, io, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import gradcheck_vphi_xi_paths as G
from verify_vphi_xi_grad_path import c5b, step_source_grad

SR2 = [('REVERSE_CHANNEL              = True', 'REVERSE_CHANNEL              = False'),
       ("XI_GRAD_PATH           = 'default'", "XI_GRAD_PATH           = 'live'"),
       ('LOWRANK_DAMPED_FLOW  = False', 'LOWRANK_DAMPED_FLOW  = True'),
       ('VPHI_ANALYTIC_FORCE = False', 'VPHI_ANALYTIC_FORCE = True'),
       ('LAYER_CHECKPOINT = True', 'LAYER_CHECKPOINT = False')]
ARMS = {'SR2': SR2 + [("VPHI_GRAD_PATH         = 'default'", "VPHI_GRAD_PATH         = 'live'")],
        'SR2, pair force off': SR2 + [("PAIR_POTENTIAL  = 'sparse_topk'", "PAIR_POTENTIAL  = 'none'")]}


def build(repl):
    import os
    os.chdir(G.REPO / 'notebooks/conservative_arch/scaleup')
    g = {'__name__': '__main__'}
    c0 = G.cells['Cell 0:']
    for old, new in repl:
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
              if 'PAIR POTENTIAL' in l or 'parameters.' in l or 'lowrank' in l.lower() and 'driver' in l.lower()]
    return g['model'], g['_variant_tag'], banner


if __name__ == '__main__':
    torch.set_num_threads(8)
    val = np.load(G.VAL)
    rng = np.random.default_rng(20261121)
    starts = rng.integers(0, len(val) - 513, size=2)
    x = torch.from_numpy(np.stack([val[s:s + 512] for s in starts]).astype(np.int64))
    y = torch.from_numpy(np.stack([val[s + 1:s + 513] for s in starts]).astype(np.int64))
    lines, times = [], {}
    for name, repl in ARMS.items():
        m, tag, banner = build(repl)
        if 'off' in name:
            assert m.V_phi is None and m.score_head is None and m.raw_v_phi_scale is None
            assert not any(k.startswith(('V_phi.', 'score_head.', 'raw_v_phi_scale')) for k in m.state_dict())
        m.eval()
        with torch.enable_grad():
            lo, l = m(x, y)
        m.train(); torch.manual_seed(7)
        lo_tr, l_tr = m(x, y); l_tr.backward()
        assert torch.isfinite(lo).all() and torch.isfinite(l_tr)
        gn = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in m.parameters() if p.grad is not None)).item()
        ts = []
        for _ in range(3):
            m.zero_grad(set_to_none=True); torch.manual_seed(7)
            t0 = time.perf_counter(); _, l2 = m(x, y); l2.backward(); ts.append(time.perf_counter() - t0)
        times[name] = float(np.median(ts))
        src = [step_source_grad(m, x, l_) for l_ in range(m.cfg.L)]
        lines += [f'== {name}', f'   tag {tag}', *[f'   5b: {b}' for b in banner],
                  f'   params {m.num_params():,}   eval loss {l.item():.4f}   train loss {l_tr.item():.4f}   '
                  f'grad norm {gn:.3f}   forward+backward {times[name]:.2f} s (median of 3, CPU, 2 x 512)']
        lines += [f'   layer {i} step: grad -> earlier tokens {e:.4e}   -> own token {s_:.4e}' for i, (e, s_) in enumerate(src)]
        print('\n'.join(lines[-(4 + len(banner) + len(src)):]), flush=True)
    a, b = times['SR2'], times['SR2, pair force off']
    lines.append(f'\nforward+backward time, pair force off over SR2: {b / a:.2f}')
    print(lines[-1])
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    (out / 'verify_sr2_nophi_build_output.txt').write_text('\n'.join(lines) + '\n')
