#!/usr/bin/env python3
"""Does the particle store survive training, and does the model use it? A short CPU run
(2026-11-01; Stage 0'' of Implementing_Semantic_Templates_A_Path.md SS7).

    python3 ps_short_train.py OUT_DIR {off|decay|nodecay} [steps=300] [batch=4] [threads=8]

Three arms, identical but for the store, each trained from scratch on CPU from the notebook's
own cells (F3.1 base, SR2, explicit kick, no Poisson modes), seed 0, AdamW lr 1.2e-3 with a
50-step warm-up, weight decay 0.01, clip 1.0, batches of `batch` x 256 tokens from the first
1.5M tokens of the validation cache:
    off      PARTICLE_STORE = 0                      the control
    decay    PARTICLE_STORE = 4, rank 8, half-lives 1-16 tokens   bounded memory
    nodecay  PARTICLE_STORE = 4, rank 8, no spontaneous death     the live count grows with t
Evaluated at the end on 8 x 256 held-out tokens (the cache's last 0.5M):
    selectivity    median over types of the existence's coefficient of variation across tokens
    position R^2   share of the existences' variance explained by position (leave one sequence out)
    live count     particles a token sees at t = 32 / 128 / 255
    removal cost   perplexity change when every well depth is set to 0 (the store's force off),
                   the analogue of the wells-removal cost -- how much the model leans on it
    depths         how far the depths moved from their zero initialisation
A few hundred CPU steps cannot say whether a full run uses the store; they say whether the
existences stay content-dependent while the cloud reorganises, and whether the force is used
at all. Companion to pm_match_short_train.py, which asked the same of the Stage 0' matcher.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

OUT = Path(sys.argv[1]); ARM = sys.argv[2]
STEPS = int(next((a[6:] for a in sys.argv[3:] if a.startswith('steps=')), 300))
BATCH = int(next((a[6:] for a in sys.argv[3:] if a.startswith('batch=')), 4))
THREADS = int(next((a[8:] for a in sys.argv[3:] if a.startswith('threads=')), 8))
assert ARM in ('off', 'decay', 'nodecay')
OUT.mkdir(parents=True, exist_ok=True)
sys.argv = [sys.argv[0], str(OUT / f'harness_{ARM}')]
sys.path.insert(0, str(Path(__file__).parent))
import verify_pm_switch as P
torch.set_num_threads(THREADS)

REPL = P.CONFIGS['F3.1'][1] + (("LOWRANK_DAMPED_FLOW  = False", "LOWRANK_DAMPED_FLOW  = True"),)
if ARM != 'off':
    REPL += (("PARTICLE_STORE       = 0", "PARTICLE_STORE       = 4"),)
    if ARM == 'nodecay':
        REPL += (("PARTICLE_STORE_DECAY = True", "PARTICLE_STORE_DECAY = False"),)


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
tag = g['_variant_tag']
print(f'[{ARM}] tag ...{tag[tag.find("sr2"):tag.find("_L2probe")]}  '
      f'{sum(p.numel() for p in model.parameters()):,} parameters', flush=True)
ids = g['val_ids']; n_tr = 1_500_000
rng = np.random.default_rng(0)


def batch(lo, hi, B, T=256):
    s = rng.integers(lo, hi - T - 1, B)
    x = np.stack([ids[i:i + T + 1] for i in s]).astype('int64')
    return torch.from_numpy(x[:, :-1]), torch.from_numpy(x[:, 1:])


dec = [p for n, p in model.named_parameters() if p.dim() >= 2]
nod = [p for n, p in model.named_parameters() if p.dim() < 2]
opt = torch.optim.AdamW([{'params': dec, 'weight_decay': 0.01}, {'params': nod, 'weight_decay': 0.0}], lr=1.2e-3)
log = []
t0 = time.time()
model.train()
for step in range(1, STEPS + 1):
    for gr in opt.param_groups:
        gr['lr'] = 1.2e-3 * min(1.0, step / 50)
    xb, yb = batch(0, n_tr, BATCH)
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

# ---------------------------------------------------------------- evaluation
model.eval()
xe, ye = batch(n_tr, len(ids), 8)
res = {'arm': ARM, 'tag': tag, 'steps': STEPS, 'batch': BATCH, 'loss_log': log,
       'train_seconds': round(time.time() - t0, 1), 'layers': {}}


def ppl(x, y):
    with torch.enable_grad():
        lo = model(x)[0].detach()
    nll = torch.nn.functional.cross_entropy(
        lo.reshape(-1, lo.shape[-1]), y.reshape(-1), reduction='none').reshape(x.shape).mean(1)
    return nll


base = ppl(xe, ye)
res['heldout_ppl'] = float(np.exp(base.mean()))

if ARM != 'off':
    store = {l: (e.detach(), h.detach()) for l, (e, h) in model._pstore.items()}
    for l, (E, _h) in store.items():
        Ef = E.reshape(-1, E.shape[-1])
        cv = float((Ef.std(0) / Ef.mean(0).clamp_min(1e-30)).median())
        loo = (E.sum(0, keepdim=True) - E) / (E.shape[0] - 1)
        r2 = 1 - float(((E - loo) ** 2).sum() / ((E - E.mean((0, 1))) ** 2).sum())
        T, K = E.shape[1], E.shape[2]
        t = torch.arange(T); lag = t[:, None] - t[None, :] - 1; past = lag >= 0
        if model.pstore_logit_lambda is None:
            Lam = past.float().unsqueeze(-1).expand(T, T, K)
        else:
            ll = torch.nn.functional.logsigmoid(model.pstore_logit_lambda.detach().float())
            Lam = torch.where(past, torch.exp(lag.clamp_min(0)[None].float() * ll[:, None, None]),
                              torch.zeros(())).permute(1, 2, 0)
        n = torch.einsum('tsk,bsk->btk', Lam, E).sum(-1).mean(0)
        res['layers'][l] = {'selectivity': cv, 'position_R2': r2,
                            'mean_existence': float(E.mean()),
                            'live_at_32_128_255': [round(float(n[i]), 1) for i in (32, 128, 255)]}
    d = model.pstore_depth.detach()
    res['depths'] = {'abs_mean': float(d.abs().mean()), 'min': float(d.min()), 'max': float(d.max()),
                     'per_layer_abs_mean': [round(float(v), 4) for v in d.abs().mean(-1)]}
    res['force_share_per_layer'] = [round(float(v), 4) for v in model.pstore_share]
    # removal cost: the store's force switched off at evaluation (depths -> 0)
    keep = model.pstore_depth.detach().clone()
    with torch.no_grad():
        model.pstore_depth.zero_()
    off = ppl(xe, ye)
    with torch.no_grad():
        model.pstore_depth.copy_(keep)
    dd = (off - base).numpy()
    res['removal_cost_pct'] = float(np.expm1(dd.mean()) * 100)
    res['removal_spread_pct'] = float(dd.std(ddof=1) * 100)

print(f'\nPARTICLE STORE SHORT TRAIN: {ARM}, {STEPS} steps of {BATCH} x 256 tokens, '
      f'{res["train_seconds"]:.0f}s; held-out perplexity {res["heldout_ppl"]:.1f}')
for l, v in sorted(res['layers'].items()):
    print(f'   layer {l}: selectivity {v["selectivity"]:.3f}   existences by position R^2 {v["position_R2"]:.4f}   '
          f'mean e {v["mean_existence"]:.3f}   live at t=32/128/255 {v["live_at_32_128_255"]}')
if ARM != 'off':
    print(f'   depths |a| mean {res["depths"]["abs_mean"]:.4f} (per layer {res["depths"]["per_layer_abs_mean"]}), '
          f'range {res["depths"]["min"]:+.3f}..{res["depths"]["max"]:+.3f}')
    print(f'   force share of the store per layer {res["force_share_per_layer"]}')
    print(f'   removal cost (depths -> 0): {res["removal_cost_pct"]:+.2f}% '
          f'(per-sequence spread {res["removal_spread_pct"]:.2f}%, n = 8)')
(OUT / f'ps_short_train_{ARM}.json').write_text(json.dumps(res, indent=1))
print('wrote', OUT / f'ps_short_train_{ARM}.json')
