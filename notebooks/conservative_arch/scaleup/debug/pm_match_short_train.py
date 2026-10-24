#!/usr/bin/env python3
"""Does the low-rank matcher's selectivity survive training? A short CPU training run (2026-10-23; Stage 0').

    python3 pm_match_short_train.py OUT_DIR {iso|lr8} [steps=400] [batch=8] [threads=16]

Both arms: the notebook's model from its own cells (F3.1 base, SR2, K = 16 Poisson modes, explicit wells), seed 0,
trained from scratch on CPU: AdamW, lr 1.2e-3 with a 50-step linear warm-up then constant, weight decay 0.01,
gradient clip 1.0, batches of `batch` (default 8) x 256 tokens from the first 1.5M tokens of the validation cache. The arms differ only
in the matcher: 'iso' the default isotropic modes, 'lr8' POISSON_MATCH_RANK = 8. Evaluated at the end on 8 x 256
held-out tokens (the cache's last 0.5M), per layer, with the readings used on the trained models:
  selectivity   median over modes of the overlap's coefficient of variation across tokens
  position R^2  share of the occupations' variance explained by position (leave one sequence out)
  content swap  perplexity change when each sequence gets another sequence's occupations (all layers)
A few hundred steps cannot show whether a full run USES the selectivity; they show whether it collapses early,
while the cloud reorganises from the embeddings' initial geometry.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

OUT = Path(sys.argv[1]); ARM = sys.argv[2]
STEPS = int(next((a[6:] for a in sys.argv[3:] if a.startswith('steps=')), 400))
BATCH = int(next((a[6:] for a in sys.argv[3:] if a.startswith('batch=')), 8))
THREADS = int(next((a[8:] for a in sys.argv[3:] if a.startswith('threads=')), 16))
assert ARM in ('iso', 'lr8')
OUT.mkdir(parents=True, exist_ok=True)
sys.argv = [sys.argv[0], str(OUT / f'harness_{ARM}')]
sys.path.insert(0, str(Path(__file__).parent))
import verify_pm_switch as P
torch.set_num_threads(THREADS)

REPL = P.CONFIGS['F3.1'][1] + (("POISSON_MODES        = 0", "POISSON_MODES        = 16"),
                               ("LOWRANK_DAMPED_FLOW  = False", "LOWRANK_DAMPED_FLOW  = True"))
if ARM == 'lr8':
    REPL = REPL + (("POISSON_MATCH_RANK   = 0", "POISSON_MATCH_RANK   = 8"),)


class _Stop(Exception):
    pass


def build_init():
    load = torch.load
    torch.load = lambda *a, **k: (_ for _ in ()).throw(_Stop())
    torch.manual_seed(0); np.random.seed(0)
    try:
        P.build(P.F31_F, REPL)
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


model, g = build_init()
model.to('cpu'); model.cfg.use_layer_checkpoint = False
ids = g['val_ids']; n_tr = 1_500_000
rng = np.random.default_rng(0)
def batch(lo, hi, B, T):
    s = rng.integers(lo, hi - T - 1, B)
    x = np.stack([ids[i:i + T + 1] for i in s]).astype('int64')
    return torch.from_numpy(x[:, :-1]), torch.from_numpy(x[:, 1:])

decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
nodecay = [p for n, p in model.named_parameters() if p.dim() < 2]
opt = torch.optim.AdamW([{'params': decay, 'weight_decay': 0.01}, {'params': nodecay, 'weight_decay': 0.0}], lr=1.2e-3)
log = []
t0 = time.time()
model.train()
for step in range(1, STEPS + 1):
    for gr in opt.param_groups:
        gr['lr'] = 1.2e-3 * min(1.0, step / 50)
    xb, yb = batch(0, n_tr, BATCH, 256)
    torch.manual_seed(step)
    opt.zero_grad(set_to_none=True)
    with torch.enable_grad():
        _, loss = model(xb, targets=yb)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 25 == 0 or step == 1:
        log.append((step, float(loss)))
        print(f'  [{ARM}] step {step:4d}  loss {float(loss):.4f}  ({time.time() - t0:.0f}s)', flush=True)

# ---- evaluation on held-out tokens
model.eval()
rng = np.random.default_rng(12345)
xe, ye = batch(n_tr, len(ids), 8, 256)
caps = {}
orig = model._fock_layer_step
def rec(h, h_prev, r, sal, m_b, gamma, dt, layer_idx=0):
    caps.setdefault(int(layer_idx), h.detach().clone())
    return orig(h, h_prev, r, sal, m_b, gamma, dt, layer_idx=layer_idx)
model._fock_layer_step = rec
with torch.enable_grad():
    lo = model(xe)[0].detach()
model._fock_layer_step = orig
base = torch.nn.functional.cross_entropy(lo.reshape(-1, lo.shape[-1]), ye.reshape(-1), reduction='none').reshape(xe.shape).mean(1)
res = {'arm': ARM, 'steps': STEPS, 'loss_log': log, 'layers': {}}
for l, h in caps.items():
    with torch.no_grad():
        E, phi = model.poisson_mode_occupation(h, l) if ARM == 'lr8' else model.poisson_mode_occupation(h)
    Ef = E.reshape(-1, E.shape[-1])
    cv = float((Ef.std(0) / Ef.mean(0).clamp_min(1e-30)).median())
    loo = (phi.sum(0, keepdim=True) - phi) / (phi.shape[0] - 1)
    r2 = 1 - float(((phi - loo) ** 2).sum() / ((phi - phi.mean((0, 1))) ** 2).sum())
    res['layers'][l] = {'selectivity': cv, 'position_R2': r2}
orig_occ = model._pm_occupation_from_overlap
model._pm_occupation_from_overlap = lambda E: orig_occ(E).roll(1, dims=0)
with torch.enable_grad():
    lo2 = model(xe)[0].detach()
model._pm_occupation_from_overlap = orig_occ
sw = torch.nn.functional.cross_entropy(lo2.reshape(-1, lo2.shape[-1]), ye.reshape(-1), reduction='none').reshape(xe.shape).mean(1)
d = (sw - base).numpy()
res['content_swap_pct'] = float(np.expm1(d.mean()) * 100); res['content_swap_spread_pct'] = float(d.std(ddof=1) * 100)
res['heldout_ppl'] = float(np.exp(base.mean()))
print(f'PM-MATCH SHORT TRAIN: {ARM}, {STEPS} steps of {BATCH} x 256 tokens; held-out perplexity {res["heldout_ppl"]:.1f}')
for l, v in sorted(res['layers'].items()):
    print(f'   layer {l}: selectivity {v["selectivity"]:.3f}   occupations explained by position R^2 {v["position_R2"]:.4f}')
print(f'   content swap: {res["content_swap_pct"]:+.2f}% (per-sequence spread {res["content_swap_spread_pct"]:.2f}%, n = 8)')
(OUT / f'short_train_{ARM}.json').write_text(json.dumps(res, indent=1))
