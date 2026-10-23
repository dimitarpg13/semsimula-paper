#!/usr/bin/env python3
"""Run a ladder-notebook diagnostic cell, unchanged, on CPU against a downloaded run folder (2026-10-23).

    python3 run_diag_cell_cpu.py OUT_DIR {pmx|fob-pmx} CELL [threads=16] [folder=<run folder>]

  CELL: the cell's id as its header writes it, e.g. 6b-13 or 6b-15.
  folder=: the run folder in ~/Downloads (default PMX's K = 64 folder of the kind); K is read from its pm<K> tag.

The model is built and loaded by the shared harness (verify_pm_switch.build: the notebook's Cells 0-5 with the
kind's Cell 0 replacements, the folder's _best.pt). The cell's source is then executed in the same namespace, so it
reads its own checkpoint, batches and settings exactly as on Colab. For Cell 6b-13 the resonance monitor is taken
from the local semsimula-diag checkout (../../../../../semsimula-diag/src) instead of a clone into /content. The
cell's own jsonl records go to OUT_DIR/harness/results.
"""
import io
import contextlib
import json
import sys
from pathlib import Path

import torch

OUT = Path(sys.argv[1]); KIND = sys.argv[2]; CELL = sys.argv[3]
THREADS = int(next((a[8:] for a in sys.argv[4:] if a.startswith('threads=')), 16))
FOLDER_ARG = next((a[7:] for a in sys.argv[4:] if a.startswith('folder=')), None)
assert KIND in ('pmx', 'fob-pmx')
sys.argv = [sys.argv[0], str(OUT / 'harness')]
sys.path.insert(0, str(Path(__file__).parent))
import verify_pm_switch as P
torch.set_num_threads(THREADS)

PFX = ('semsimula_fock_cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_')
SFX = '_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn'
FOLDER = FOLDER_ARG or (PFX + ('pm64_pmx16_sr2_fob' if KIND == 'fob-pmx' else 'pm64_pmx16_sr2') + SFX)
repl = P.CONFIGS['F3.1'][1] + P.PM_ON + (("LOWRANK_DAMPED_FLOW  = False", "LOWRANK_DAMPED_FLOW  = True"),
                                         ("POISSON_WELLS_EXACT  = False", "POISSON_WELLS_EXACT  = True"))
_k = next((int(t[2:]) for t in FOLDER.split('_') if t.startswith('pm') and t[2:].isdigit()), 64)
if _k != 64:
    repl = repl + (("POISSON_MODES        = 64", f"POISSON_MODES        = {_k}"),)
if KIND == 'fob-pmx':
    repl = repl + (("FIRST_ORDER_OVERDAMPED = False", "FIRST_ORDER_OVERDAMPED = True"),)
model, g, _ = P.build(FOLDER, repl)
model.to('cpu'); model.cfg.use_layer_checkpoint = False
assert bool(model.cfg.first_order_overdamped) == (KIND == 'fob-pmx') and model.pm_depth.shape[1] == _k

nb = json.load(open(P.G.NB))
src = [''.join(c['source']) for c in nb['cells'] if ''.join(c['source']).startswith(f'# == Cell {CELL}:')]
assert len(src) == 1, f'no unique cell {CELL}'
if CELL == '6b-13':
    diag = Path(__file__).resolve().parents[5] / 'semsimula-diag' / 'src'
    sys.path.insert(0, str(diag))
    from semsimula_diag.probes import resonance
    g['_RESONANCE_AVAILABLE'] = True
    g['_resonance'] = resonance
    g['_resonance_mpx'] = sys.modules[type(model)._layer_step_langevin.__module__]
g['DEVICE'] = 'cpu'
print(f'[cpu] Cell {CELL} on {FOLDER}  (K = {_k}, {KIND})', flush=True)
exec(compile(P.G.strip(src[0]), f'Cell{CELL}', 'exec'), g)
