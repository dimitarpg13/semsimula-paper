"""Independent causality check of a trained ladder checkpoint, before publication.

Complements the in-flight SCAF monitor with three direct tests on the final
weights, in eval mode (the mode every reported PPL uses), on the model built
through the ladder notebook's own Cells 0-5b:

  1. future perturbation: for several cut points p, replace tokens > p with
     random tokens (several draws); logits at positions <= p must be
     bit-identical
  2. batch independence: replace one sequence of the batch entirely; the
     other sequences' logits must be bit-identical
  3. prefix-only ("honest") scoring: at sampled positions t, feed only
     x[:t+1] and compare the logit at t with the full-sequence logit at t;
     report max |d logit| and the leak tax in nats (float noise expected,
     since the sequence length changes the reduction order)

plus validation PPL on the local cache, as a load check.

Usage: python3 causality_check_checkpoint.py OUT_DIR FOLDER [REVERSE_CHANNEL VPHI XI]
  e.g. ... OUT semsimula_..._norc_vplive_xilive_..._noattn False live live
"""
import math, sys
from pathlib import Path

import numpy as np
import torch

sys.argv = sys.argv[:1] + [sys.argv[1]] + sys.argv[2:]
sys.path.insert(0, str(Path(__file__).parent))
import gradcheck_vphi_xi_paths as G
from verify_vphi_xi_grad_path import build

FOLDER = sys.argv[2]
RC = (sys.argv[3] == 'True') if len(sys.argv) > 3 else True
VP = sys.argv[4] if len(sys.argv) > 4 else 'default'
XI = sys.argv[5] if len(sys.argv) > 5 else 'default'


def logits_of(model, x):
    with torch.enable_grad():          # the force needs autograd.grad even in eval
        lo, _ = model(x)
    return lo.detach()


if __name__ == '__main__':
    torch.manual_seed(0)
    model, tag, _ = build(RC, FOLDER, VP, XI)
    model.eval()
    V = model.cfg.vocab_size
    val = np.load(G.VAL)
    rng = np.random.default_rng(20261002)
    starts = rng.integers(0, len(val) - 513, size=3)
    x = torch.from_numpy(np.stack([val[s:s + 512] for s in starts]).astype(np.int64))
    base = logits_of(model, x)
    print(f"model: ...{tag[tag.find('cgqk'):]}\n")

    print("1. FUTURE PERTURBATION  (max |d logit| at positions <= p; must be exactly 0)")
    worst = 0.0
    g = torch.Generator().manual_seed(1)
    for p in (31, 127, 255, 383, 500):
        for draw in range(3):
            xc = x.clone()
            xc[:, p + 1:] = torch.randint(0, V, xc[:, p + 1:].shape, generator=g)
            d = (logits_of(model, xc)[:, :p + 1] - base[:, :p + 1]).abs().max().item()
            worst = max(worst, d)
        print(f"   cut p={p:>3}: max |d| over 3 draws = {d:.3e}")
    print(f"   -> {'CLEAN' if worst == 0.0 else 'LEAK'} (worst {worst:.3e})\n")

    print("2. BATCH INDEPENDENCE  (replace sequence 2 entirely; sequences 0,1 must not move)")
    xc = x.clone()
    xc[2] = torch.randint(0, V, (512,), generator=g)
    d = (logits_of(model, xc)[:2] - base[:2]).abs().max().item()
    print(f"   max |d| on the untouched sequences = {d:.3e}  -> {'CLEAN' if d == 0.0 else 'CROSS-SEQUENCE LEAK'}\n")

    print("3. PREFIX-ONLY SCORING  (feed x[:t+1] only; compare the logit at t)")
    y = torch.from_numpy(np.stack([val[s + 1:s + 513] for s in starts]).astype(np.int64))
    lp_full = torch.log_softmax(base.double(), -1)
    nll_full, nll_hon, dmax = [], [], 0.0
    for t in (0, 7, 63, 128, 200, 311, 400, 511):
        hon = logits_of(model, x[:, :t + 1])[:, t]
        dmax = max(dmax, (hon - base[:, t]).abs().max().item())
        lp_h = torch.log_softmax(hon.double(), -1)
        nll_full += (-lp_full[torch.arange(3), t, y[:, t]]).tolist()
        nll_hon += (-lp_h[torch.arange(3), y[:, t]]).tolist()
    tax = float(np.mean(nll_hon) - np.mean(nll_full))
    print(f"   positions 0..511 (8 sampled x 3 seqs): max |d logit| = {dmax:.3e}, "
          f"leak tax = {tax:+.2e} nats  -> {'CLEAN' if abs(tax) < 1e-3 else 'CHECK'}\n")

    print("4. VALIDATION PPL on the local cache  (load check; 12 batches x 4 x 512)")
    r2 = np.random.default_rng(7)
    tot, n = 0.0, 0
    for _ in range(12):
        st = r2.integers(0, len(val) - 513, size=4)
        xb = torch.from_numpy(np.stack([val[s:s + 512] for s in st]).astype(np.int64))
        yb = torch.from_numpy(np.stack([val[s + 1:s + 513] for s in st]).astype(np.int64))
        with torch.enable_grad():
            _, loss = model(xb, yb)
        tot += loss.item(); n += 1
    print(f"   val loss {tot/n:.4f}  PPL {math.exp(tot/n):.2f}")
