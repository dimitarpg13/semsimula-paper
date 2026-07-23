#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fock_trained_leak_probe.py
==========================

Measures the reverse-channel causal leak of Fock-PARFLM v2.1 **on the trained
checkpoint** — the certification step the causal-leak audit
(companion_notes/Fock-PARFLM_Causal_Leak_Audit_Results.md, section 11, step 1)
left open. The audit measured the leak at INITIALIZATION scale (~1e-5 logit
shift) and explicitly warned that gradient flows through the channel, so
training can amplify it. This script replaces the init-scale bound with
numbers from the actual trained weights.

Why this matters: every PPL number reported so far (in-loop `ntp`, the 40-batch
in-loop eval, and the full-set sliding-window eval) is computed by
teacher-forced FULL-WINDOW forwards. In such a forward the Fock register
summary is built from the ENTIRE 512-token window — including tokens at and
after the position being scored. If the trained model has learned to route a
"window vocabulary/topic" digest through the reverse channel, all of those
numbers are inflated by the same mechanism, coherently (train loss and val
loss together), and the inflation GROWS over training as the channel
strengthens. A leak of this kind is invisible to a WikiText-103 cross-check
(it is architectural, not data contamination).

Two independent measurements:

PART 1 — future-perturbation probe at trained scale (float64, CPU-exact)
    For window pairs (x, x') identical up to position t_p and different after,
    measure at positions < t_p:
      (a) max |Δlogit|                      (init-scale reference: ~1.1e-5)
      (b) mean Δnll of the TRUE past targets when the future is swapped —
          this is the leak's predictive value in nats/token, directly
          comparable to the ~0.5-0.7 nat gap that needs explaining.
    Controls: determinism (same input twice → 0), gate zeroed (→ exactly 0).

PART 2 — honest (leak-free) PPL via next-token scoring
    Standard protocol scores token t from a window that CONTAINS token t and
    its future. Honest protocol: feed ONLY tokens < t and read the logits of
    the LAST input position (whose target lies outside the window — nothing
    in the forward can have seen it). For the same K target tokens:
      PPL_A  : mid-window scoring (in-window index 256; ≈ the protocol behind
               every number reported so far)
      PPL_B  : last-position scoring, target outside window (leak-free), with
               511 tokens of left context (MORE than A — for a causal model
               PPL_B ≤ PPL_A must hold, up to noise)
    If PPL_B >> PPL_A, PPL_B is the model's real perplexity and the gap
    measures the leak's contribution. Also prints the within-window NLL
    position profile (for a causal model it decreases toward the window end;
    rising NLL toward the end = future context was doing the work).

Usage (Colab CPU, same env as eval_ppl_proper.py; model code on sys.path):

    !python fock_trained_leak_probe.py \
        --ckpt "/content/drive/MyDrive/.../..._step103500_best.pt" \
        --val  "/content/drive/MyDrive/.../openwebtext_val_2M.npy" \
        --k 1024 --batch 8

Or from the eval_ppl_debug notebook with `model` and `val_tokens` in memory:

    from fock_trained_leak_probe import probe_trained_leak, honest_ppl_test
    probe_res  = probe_trained_leak(model, val_tokens, device=DEVICE)
    honest_res = honest_ppl_test(model, val_tokens, k=1024, batch=4,
                                 device=DEVICE)
"""

import argparse
import gc
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from eval_ppl_proper import (  # noqa: E402
    build_model, load_tokens, _forward_logits, _nll_from_logits,
)


# =========================================================================
# PART 1 — future-perturbation probe at trained scale
# =========================================================================
def probe_trained_leak(model, val_tokens, device="cpu", context=512,
                       t_perturb=None, n_pairs=4, use_float64=True, seed=7):
    """
    Perturb the future half of real validation windows and measure the effect
    on past-position logits and past-target NLL, using the trained weights and
    the trained (as-loaded) reverse-channel gate values.
    """
    t_p = t_perturb if t_perturb is not None else context // 2
    toks = np.asarray(val_tokens).reshape(-1)
    rng = np.random.default_rng(seed)

    orig_dtype = next(model.parameters()).dtype
    if use_float64:
        model.double()
    model.eval()
    flag = [True]

    def fwd(x_np):
        x = torch.from_numpy(x_np[None]).long().to(device)
        logits = _forward_logits(model, x, flag).float().cpu()
        del x
        gc.collect()
        return logits

    gate = None
    if getattr(model, "reverse_channel_scale", None) is not None:
        gate = torch.tanh(model.reverse_channel_scale.detach()).float().cpu()
        print(f"[probe] reverse gate tanh(scale) per layer: "
              f"{[round(v, 4) for v in gate.tolist()]}")
        print(f"[probe] gate |mean| = {gate.abs().mean().item():.4f}")

    max_dlogit, mean_dnll_list, det_list = [], [], []
    print(f"[probe] {n_pairs} window pairs, context={context}, t_p={t_p}, "
          f"dtype={'float64' if use_float64 else str(orig_dtype)}")

    for p in range(n_pairs):
        s = int(rng.integers(0, len(toks) - 2 * context))
        d = int(rng.integers(0, len(toks) - context))
        x1 = toks[s:s + context].astype(np.int64)
        x2 = x1.copy()
        x2[t_p:] = toks[d:d + context].astype(np.int64)[t_p:]  # real-text swap

        la = fwd(x1)
        if p == 0:  # determinism control on the first pair only
            lb = fwd(x1)
            det = (la - lb).abs().max().item()
            det_list.append(det)
            print(f"  [T0] determinism (same input twice): "
                  f"max|dlogit| = {det:.3e}")
            del lb
        lc = fwd(x2)

        d_logit = (la[:, :t_p] - lc[:, :t_p]).abs().max().item()
        max_dlogit.append(d_logit)

        # NLL of the TRUE past targets under both futures
        tgt = torch.from_numpy(x1[1:t_p])[None]
        nll_a = _nll_from_logits(la[:, :t_p - 1], tgt)
        nll_c = _nll_from_logits(lc[:, :t_p - 1], tgt)
        d_nll = (nll_c - nll_a).mean().item()   # >0: original future HELPED
        mean_dnll_list.append(d_nll)
        print(f"  pair {p}: max|dlogit|(past) = {d_logit:.3e}   "
              f"mean dNLL(past targets) = {d_nll:+.4f} nats")
        del la, lc
        gc.collect()

    # control: gate zeroed must give EXACTLY zero (float64)
    d_ctrl = None
    if getattr(model, "reverse_channel_scale", None) is not None:
        with torch.no_grad():
            saved = model.reverse_channel_scale.detach().clone()
            model.reverse_channel_scale.zero_()
        s = int(rng.integers(0, len(toks) - 2 * context))
        x1 = toks[s:s + context].astype(np.int64)
        x2 = x1.copy()
        x2[t_p:] = toks[s + context:s + 2 * context].astype(np.int64)[t_p:]
        la0, lc0 = fwd(x1), fwd(x2)
        d_ctrl = (la0[:, :t_p] - lc0[:, :t_p]).abs().max().item()
        print(f"  [ctrl] gate zeroed: max|dlogit|(past) = {d_ctrl:.3e} "
              f"(must be ~0)")
        with torch.no_grad():
            model.reverse_channel_scale.copy_(saved)
        del la0, lc0
        gc.collect()

    model.to(orig_dtype)
    gc.collect()

    res = {
        "max_dlogit_past": max(max_dlogit),
        "mean_dnll_past": float(np.mean(mean_dnll_list)),
        "dnll_per_pair": mean_dnll_list,
        "determinism": det_list[0] if det_list else None,
        "gate_zero_control": d_ctrl,
        "gate_abs_mean": gate.abs().mean().item() if gate is not None else None,
        "t_p": t_p, "context": context, "n_pairs": n_pairs,
    }
    print("\n[probe] SUMMARY")
    print(f"  trained-scale leak:  max|dlogit| at past positions = "
          f"{res['max_dlogit_past']:.3e}   (init-scale reference ~1.1e-05)")
    print(f"  predictive value:    mean dNLL of past targets when future "
          f"swapped = {res['mean_dnll_past']:+.4f} nats/token")
    print(f"  interpretation:      dNLL ~ +0.3..0.7 nats -> the leak carries "
          f"most of the apparent PPL gain;")
    print(f"                       dNLL ~ +0.00x nats     -> leak still "
          f"negligible at trained scale.")
    return res


# =========================================================================
# PART 2 — honest (leak-free) PPL vs standard full-window PPL
# =========================================================================
def honest_ppl_test(model, val_tokens, k=1024, context=512, batch=8,
                    device="cpu", seed=123, profile_bins=16):
    """
    Score the SAME k target tokens two ways:
      A) mid-window: target inside the window at index context//2 (the window
         also contains the target itself and its future — this is what the
         training loss, in-loop eval and full-set sliding eval all do);
      B) last-position: input is exactly the `context` tokens BEFORE the
         target; the target is read from the final position's logits and is
         never part of the forward. Leak-free by construction.
    """
    toks = np.asarray(val_tokens).reshape(-1)
    N = len(toks)
    half = context // 2
    rng = np.random.default_rng(seed)
    # target token index a: window B needs a-context >= 0; window A needs
    # a-half+context <= N
    anchors = rng.integers(context, N - half - 1, size=k).astype(np.int64)

    model.eval()
    flag = [True]
    nll_A = np.zeros(k, dtype=np.float64)
    nll_B = np.zeros(k, dtype=np.float64)
    prof_sum = torch.zeros(context - 1, dtype=torch.float64)
    prof_cnt = 0

    for off in range(0, k, batch):
        idx = anchors[off:off + batch]

        # --- A: mid-window scoring (standard protocol) ---
        xa = np.stack([toks[a - half:a - half + context] for a in idx]
                      ).astype(np.int64)
        xA = torch.from_numpy(xa).to(device)
        logits = _forward_logits(model, xA, flag)
        lp = F.log_softmax(logits[:, half - 1].float(), dim=-1)
        nll_A[off:off + len(idx)] = (
            -lp.gather(-1, xA[:, half:half + 1]).squeeze(-1)
        ).cpu().numpy()
        del logits, lp, xA
        gc.collect()

        # --- B: last-position scoring (leak-free) ---
        xb = np.stack([toks[a - context:a] for a in idx]).astype(np.int64)
        yb = toks[idx].astype(np.int64)
        xB = torch.from_numpy(xb).to(device)
        logits = _forward_logits(model, xB, flag)
        lp = F.log_softmax(logits[:, -1].float(), dim=-1)  # target OUTSIDE window
        nll_B[off:off + len(idx)] = (
            -lp.gather(-1, torch.from_numpy(yb)[:, None].to(device))
            .squeeze(-1)
        ).cpu().numpy()
        # free position profile from the same forward (in-window targets,
        # i.e. the LEAKY regime, as a function of position)
        prof = _nll_from_logits(logits[:, :-1].float().cpu(), xB[:, 1:].cpu())
        prof_sum += prof.sum(dim=0).double()
        prof_cnt += prof.shape[0]
        del logits, lp, xB, prof
        gc.collect()

        done = min(off + batch, k)
        print(f"\r[honest-ppl] targets {done}/{k}", end="", flush=True)
    print()

    m_A, m_B = nll_A.mean(), nll_B.mean()
    diff = nll_B - nll_A
    se = diff.std(ddof=1) / math.sqrt(k)
    res = {
        "k": k,
        "nll_mid_window": m_A, "ppl_mid_window": math.exp(m_A),
        "nll_last_pos": m_B, "ppl_last_pos": math.exp(m_B),
        "paired_diff_nats": diff.mean(), "paired_diff_se": se,
    }

    print("\n" + "=" * 64)
    print("HONEST vs STANDARD PPL (same target tokens)")
    print("=" * 64)
    print(f"  A mid-window  (256 left ctx, target+future IN window)   : "
          f"NLL {m_A:.4f}  PPL {math.exp(m_A):8.2f}")
    print(f"  B last-pos    (511 left ctx, target OUTSIDE window)     : "
          f"NLL {m_B:.4f}  PPL {math.exp(m_B):8.2f}   <-- leak-free")
    print(f"  paired diff (B - A) = {diff.mean():+.4f} +/- {se:.4f} nats")
    print(f"  causal model expectation: B <= A (B has MORE left context).")
    print(f"  B >> A  ==> reported PPLs are leak-inflated; "
          f"PPL_B is the honest number.")

    # position profile (in-window targets = leaky regime)
    prof_mean = (prof_sum / prof_cnt).numpy()
    bsz = (context - 1) // profile_bins
    print(f"\n  within-window NLL profile ({prof_cnt} windows; "
          f"position -> mean NLL):")
    for b in range(profile_bins):
        lo, hi = b * bsz, (b + 1) * bsz if b < profile_bins - 1 else context - 1
        seg = prof_mean[lo:hi]
        print(f"    pos {lo + 1:3d}-{hi:3d}: {seg.mean():.4f}")
    print(f"  (causal: decreasing toward the end; RISING toward the end = "
          f"future context was doing the work)")
    res["position_profile"] = prof_mean.tolist()
    return res


# =========================================================================
# CLI
# =========================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Trained-checkpoint causal-leak probe for Fock-PARFLM.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--logfreq", default=None)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--n-pairs", type=int, default=4)
    ap.add_argument("--k", type=int, default=1024,
                    help="target tokens for the honest-PPL test")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--skip-probe", action="store_true")
    ap.add_argument("--skip-honest", action="store_true")
    args = ap.parse_args()

    if args.logfreq is None:
        import tempfile
        _dummy = np.full(50257, -np.log(1.0 / 50257), dtype=np.float32)
        _tmpf = os.path.join(tempfile.gettempdir(), "dummy_logfreq.npy")
        np.save(_tmpf, _dummy)
        args.logfreq = _tmpf

    print(f"[load] checkpoint: {args.ckpt}")
    ckpt_state = torch.load(args.ckpt, map_location="cpu")
    model = build_model(ckpt_state, args.device, logfreq_path=args.logfreq)
    del ckpt_state
    gc.collect()

    val = load_tokens(args.val)

    if not args.skip_probe:
        print("\n" + "#" * 64)
        print("# PART 1: future-perturbation probe at TRAINED scale")
        print("#" * 64)
        probe_trained_leak(model, val, device=args.device,
                           context=args.context, n_pairs=args.n_pairs)

    if not args.skip_honest:
        print("\n" + "#" * 64)
        print("# PART 2: honest (leak-free) PPL vs standard protocol")
        print("#" * 64)
        honest_ppl_test(model, val, k=args.k, context=args.context,
                        batch=args.batch, device=args.device)


if __name__ == "__main__":
    main()
