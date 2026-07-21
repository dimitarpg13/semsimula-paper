#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_ppl_proper.py
==================

A one-off, *trustworthy* perplexity evaluation for a Fock-PARFLM v2.1
checkpoint. Its purpose is to answer the question "is my validation metric
believable?" independently of the training loop's cheap 5-batch estimate.

It differs from the in-loop eval in three deliberate ways:

  1. It scores the *entire* held-out set (e.g. the full 2M-token
     `openwebtext_val_2M.npy`), not 5 random batches (~40k tokens).
  2. It uses a *sliding window with stride* so that (almost) every scored
     token sees a full CONTEXT-token left context, instead of penalising
     tokens near window starts.
  3. It reports PPL as an aggregate AND as mean +/- std across chunks, so
     you can see the confidence spread.

--------------------------------------------------------------------------
IMPORTANT GOTCHAS (read before running)
--------------------------------------------------------------------------

* EVAL AT THE NATIVE CONTEXT (512), *NOT* 1024.
  The positional embedding matrix has max_len=1024 rows, but training used
  block_size=512, so positions 512..1023 NEVER received a gradient. Scoring
  at 1024 feeds the model untrained position embeddings and yields a
  garbage / inflated PPL. Keep CONTEXT=512. (Sliding-window at 512 is the
  honest way to approximate long-context conditioning.)

* DO NOT WRAP THE FORWARD IN torch.no_grad() BLINDLY.
  Fock-PARF computes its conservative force via autograd.grad(U, h)
  INTERNALLY, even at inference. If the model does not itself wrap that in
  torch.enable_grad(), an outer torch.no_grad() will raise
  ("element 0 ... does not require grad") or silently break the force.
  This script auto-detects: it tries a no_grad forward once and falls back
  to grad-enabled (with manual detach) if that fails. No backward() is ever
  called and create_graph=False in eval, so memory stays bounded either way.

* THE CONFIG MUST MATCH TRAINING EXACTLY.
  load_state_dict is unforgiving. Fill in build_model() below by importing
  YOUR model module (v2.1: model_fock_parf_v2.py) and constructing the SAME
  config you trained with (d=384, L=16, M=32, xi-heads=5, wells/head=8,
  gamma=0.30, untied embeddings, depth-conditioned V_theta,
  use_reverse_channel=True, block_size=512, max_len=1024, vocab=50257).

--------------------------------------------------------------------------
USAGE (Colab, recommended)
--------------------------------------------------------------------------

    from google.colab import drive
    drive.mount('/content/drive')
    # make the model code importable:
    import sys; sys.path.append('/content/semsimula-paper/notebooks/conservative_arch/parf')
    # then run:
    !python eval_ppl_proper.py \
        --ckpt   "/content/drive/MyDrive/.../fock_..._step77500_best.pt" \
        --val    "/content/drive/MyDrive/.../openwebtext_val_2M.npy" \
        --context 512 --stride 256 --batch 16

Or import and call run_eval(...) directly from a notebook cell.
"""

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F


# =========================================================================
# 1. MODEL CONSTRUCTION  --  *** EDIT THIS TO MATCH YOUR TRAINING ***
# =========================================================================
def build_model(ckpt_state, device, logfreq_path='/content/dummy_logfreq.npy'):
    """
    Build the Fock-PARFLM v2.1 model with the EXACT config used in training,
    then load weights from `ckpt_state`.

    This reproduces the model construction from
    colab_fock_depthcond_vtheta_openwebtext_ext2.ipynb:
      1. Build FockMultiXiPARFLM with FockMultiXiPARFConfig (fock_version='v2')
      2. Replace V_theta with DepthConditionedMultiContextGaussianVTheta
      3. Install depth routing on the per-layer step
      4. Patch reverse_channel_scale shape if needed (per-layer vs scalar)
      5. Load state_dict
    """
    import math
    from model_fock_parf_multixi import FockMultiXiPARFLM, FockMultiXiPARFConfig
    from model_gaussian_vtheta import (
        DepthConditionedMultiContextGaussianVTheta,
        install_depth_routing,
    )

    # --- resolved constants from notebook (XI_OVERRIDE='5long') ---
    XI_ALPHA_INITS = [0.50, 0.75, 0.95, 0.99, 0.995]
    XI_CHANNELS = 5
    V_THETA_N_HEADS = XI_CHANNELS       # 5
    V_THETA_WELLS_PER_HEAD = 8
    V_THETA_DEPTH_CODE_INIT_STD = 0.02
    W_SCALE = 1.0
    D = 384
    L = 16

    print("[build_model] constructing FockMultiXiPARFLM with fock_version='v2'")
    config = FockMultiXiPARFConfig(
        vocab_size=50257, d=D, max_len=1024,
        L=L, v_hidden=1024, v_depth=3, dt=1.0,
        mass_mode='logfreq',
        logfreq_path=logfreq_path,
        logfreq_init_alpha=0.1,
        init_gamma=1.0,
        fixed_gamma=0.30,
        causal_force=True,
        ln_after_step=True,
        xi_channels=XI_CHANNELS,
        xi_alpha_inits=XI_ALPHA_INITS,
        xi_learnable=True,
        xi_alpha_init_mode='explicit',
        v_phi_kind='structural_competitive',
        v_phi_d_type=32,
        v_phi_d_angle=16,
        v_phi_eps=0.1,
        v_phi_phi_hidden=128,
        v_phi_theta_hidden=128,
        v_phi_mlp_hidden=128,
        top_k=16,
        v_phi_n_heads=4,
        use_output_bias=True,
        tie_embeddings=False,
        score_head_hidden=32,
        gumbel_tau_init=1.0,
        gumbel_tau_min=0.3,
        gumbel_noise=True,
        use_gathered_v_phi=True,
        use_layer_checkpoint=True,
        ln_before_distance=True,
        per_layer_v_phi_scale=True,
        fock_version='v2',
        n_registers=32,
        register_salience_decay=0.5,
        register_salience_threshold=0.005,
        creation_gate_hidden=64,
        stack_discipline=True,
        d_k=64,
        tau_create_init=8.0,
        reverse_channel=True,
        reverse_channel_stable=True,
        reverse_channel_pre_ln=True,
        reverse_channel_soft_norm=True,
        reverse_channel_warmup_steps=4000,
        reverse_channel_per_layer=True,
        per_register_tau=True,
        per_register_keys=True,
        ortho_register_init=True,
        register_repulsion=True,
        register_repulsion_coeff=0.05,
        register_repulsion_kind='gram',
    )
    model = FockMultiXiPARFLM(config)

    # --- replace V_theta with depth-conditioned Gaussian ---
    _init_log_prec = -math.log(D)
    _prec_max = 2.0 / D
    print(f"[build_model] replacing V_theta -> DepthConditionedMultiContextGaussian"
          f"({V_THETA_N_HEADS}h x {V_THETA_WELLS_PER_HEAD}w, L={L})")
    model.V_theta = DepthConditionedMultiContextGaussianVTheta(
        d=D, K=V_THETA_WELLS_PER_HEAD, n_ctx=V_THETA_N_HEADS,
        n_layers=L,
        w_scale=W_SCALE,
        init_log_precision=_init_log_prec,
        precision_max=_prec_max,
        code_init_std=V_THETA_DEPTH_CODE_INIT_STD,
    ).to(device)
    install_depth_routing(model)
    print("[build_model] depth routing installed")

    # --- patch reverse_channel_scale shape if checkpoint is per-layer ---
    sd = ckpt_state["model_state_dict"] if "model_state_dict" in ckpt_state else ckpt_state
    if "reverse_channel_scale" in sd:
        ckpt_shape = sd["reverse_channel_scale"].shape
        if hasattr(model, "reverse_channel_scale") and model.reverse_channel_scale.shape != ckpt_shape:
            print(f"[build_model] patching reverse_channel_scale: "
                  f"{model.reverse_channel_scale.shape} -> {ckpt_shape}")
            model.reverse_channel_scale = torch.nn.Parameter(torch.zeros(ckpt_shape))

    # --- load weights ---
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # reverse_warmup_step is a training-only counter, safe to ignore
    unexpected = [k for k in unexpected if k != "reverse_warmup_step"]
    if missing:
        print(f"[build_model] MISSING keys ({len(missing)}): {missing[:10]}")
    if unexpected:
        print(f"[build_model] UNEXPECTED keys ({len(unexpected)}): {unexpected[:10]}")
    if missing or unexpected:
        print("[build_model] ^ config does NOT match training. Fix build_model().")
    else:
        print("[build_model] load_state_dict: perfect match (no missing/unexpected keys)")

    model.to(device)
    model.eval()
    return model


# =========================================================================
# 2. DATA
# =========================================================================
def load_tokens(path):
    """Load a 1-D array of token ids saved as .npy (uint16/int32/…)."""
    arr = np.load(path, mmap_mode="r")
    arr = np.asarray(arr).reshape(-1)
    print(f"[data] {os.path.basename(path)}: {arr.shape[0]:,} tokens "
          f"(dtype={arr.dtype})")
    return arr


def load_wikitext103_tokens(split="test"):
    """
    Optional cross-corpus contamination check. Tokenises WikiText-103 with the
    SAME GPT-2 BPE as OpenWebText (via tiktoken) so PPLs are comparable.
    Requires: `pip install datasets tiktoken`.
    """
    from datasets import load_dataset
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = enc.encode_ordinary(text)
    arr = np.asarray(ids, dtype=np.int32)
    print(f"[data] wikitext-103[{split}]: {arr.shape[0]:,} tokens")
    return arr


# =========================================================================
# 3. EVAL CORE
# =========================================================================
def _forward_logits(model, x, use_no_grad_flag):
    """
    Return logits (B, T, V). Handles models that return either `logits` or
    `(logits, loss)`. Never calls backward. See the no_grad gotcha above.
    """
    def _call():
        out = model(x)
        return out[0] if isinstance(out, (tuple, list)) else out

    if use_no_grad_flag[0]:
        try:
            with torch.no_grad():
                return _call().detach()
        except RuntimeError as e:
            # Force needs autograd internally; fall back permanently.
            print(f"[eval] torch.no_grad() forward failed ({e}); "
                  f"falling back to grad-enabled forward with manual detach.")
            use_no_grad_flag[0] = False
    # grad-enabled path (safe default for autograd-force models)
    return _call().detach()


@torch.no_grad()
def _nll_from_logits(logits, targets):
    """Per-position next-token NLL. logits:(B,T,V) already shifted-aligned."""
    logp = F.log_softmax(logits.float(), dim=-1)
    return -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B, T)


def run_eval(model, tokens, context=512, stride=256, batch=16,
             device="cuda", n_chunks=20):
    """
    Strided sliding-window PPL over a 1-D token array.

    Every window is `context` long; consecutive windows advance by `stride`.
    For the FIRST window we score all context-1 targets; for every later
    window we score only the last `stride` targets (they each have >= (context
    - stride) tokens of left context), so no token is double-counted and
    almost all scored tokens are fully conditioned.
    """
    assert 0 < stride <= context
    tokens = np.asarray(tokens).reshape(-1)
    N = tokens.shape[0]

    # window start positions
    starts = list(range(0, N - context + 1, stride))
    if not starts:
        raise ValueError(f"val set too short ({N} tokens) for context={context}")

    use_no_grad_flag = [True]  # mutable; may flip to False on first failure
    all_nll = []               # list of 1-D tensors of scored per-token NLL

    model.eval()
    for bstart in range(0, len(starts), batch):
        wstarts = starts[bstart:bstart + batch]
        # build (B, context) input of token ids
        xb = np.stack([tokens[s:s + context] for s in wstarts]).astype(np.int64)
        x = torch.from_numpy(xb).to(device)

        logits = _forward_logits(model, x, use_no_grad_flag)  # (B, context, V)
        logits = logits[:, :-1, :]          # predict pos t+1 from <= t
        tgt = x[:, 1:]                      # (B, context-1)
        nll = _nll_from_logits(logits, tgt)  # (B, context-1)

        for i, s in enumerate(wstarts):
            if s == 0:
                scored = nll[i]                      # score everything once
            else:
                scored = nll[i, -stride:]            # only the new tokens
            all_nll.append(scored.detach().float().cpu())

        done = min(bstart + batch, len(starts))
        print(f"\r[eval] windows {done}/{len(starts)}", end="", flush=True)
    print()

    all_nll = torch.cat(all_nll)           # (num_scored_tokens,)
    n_tok = all_nll.numel()
    agg_loss = all_nll.mean().item()
    agg_ppl = math.exp(agg_loss)

    # chunked spread: split scored tokens into n_chunks contiguous chunks
    chunk_ppls = []
    csz = max(1, n_tok // n_chunks)
    for c in range(0, n_tok, csz):
        seg = all_nll[c:c + csz]
        if seg.numel() >= 32:
            chunk_ppls.append(math.exp(seg.mean().item()))
    chunk_ppls = torch.tensor(chunk_ppls)
    ppl_mean = chunk_ppls.mean().item()
    ppl_std = chunk_ppls.std(unbiased=True).item() if chunk_ppls.numel() > 1 else 0.0

    return {
        "tokens_scored": n_tok,
        "windows": len(starts),
        "context": context,
        "stride": stride,
        "agg_loss": agg_loss,
        "agg_ppl": agg_ppl,
        "ppl_mean_over_chunks": ppl_mean,
        "ppl_std_over_chunks": ppl_std,
        "n_chunks": int(chunk_ppls.numel()),
    }


def _report(name, r):
    print("\n" + "=" * 60)
    print(f"RESULT: {name}")
    print("=" * 60)
    print(f"  context / stride      : {r['context']} / {r['stride']}")
    print(f"  windows               : {r['windows']:,}")
    print(f"  tokens scored         : {r['tokens_scored']:,}")
    print(f"  aggregate val_loss    : {r['agg_loss']:.4f}")
    print(f"  aggregate PPL         : {r['agg_ppl']:.2f}   <-- headline")
    print(f"  PPL mean +/- std      : {r['ppl_mean_over_chunks']:.2f} "
          f"+/- {r['ppl_std_over_chunks']:.2f}  (over {r['n_chunks']} chunks)")
    print("=" * 60)


# =========================================================================
# 4. CLI
# =========================================================================
def main():
    ap = argparse.ArgumentParser(description="Proper PPL eval for Fock-PARFLM.")
    ap.add_argument("--ckpt", required=True, help="path to *_best.pt")
    ap.add_argument("--val", required=True, help="path to openwebtext_val_2M.npy")
    ap.add_argument("--context", type=int, default=512,
                    help="MUST be 512 (native trained context); see gotchas.")
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n-chunks", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--logfreq", default=None,
                    help="path to logfreq .npy (surprisal array); "
                         "if omitted, creates a dummy uniform one")
    ap.add_argument("--wikitext", action="store_true",
                    help="also eval WikiText-103 test as a contamination cross-check")
    args = ap.parse_args()

    if args.context != 512:
        print(f"[warn] context={args.context} != 512. Positions >=512 were never "
              f"trained; results will be invalid. Proceed only if you know why.")

    # --- prepare logfreq file ---
    if args.logfreq is None:
        import tempfile
        _dummy = np.full(50257, -np.log(1.0 / 50257), dtype=np.float32)
        _tmpf = os.path.join(tempfile.gettempdir(), "dummy_logfreq.npy")
        np.save(_tmpf, _dummy)
        _logfreq_path = _tmpf
        print(f"[logfreq] no --logfreq given; using uniform dummy: {_tmpf}")
    else:
        _logfreq_path = args.logfreq
        print(f"[logfreq] using: {_logfreq_path}")

    print(f"[load] checkpoint: {args.ckpt}")
    ckpt_state = torch.load(args.ckpt, map_location="cpu")
    model = build_model(ckpt_state, args.device, logfreq_path=_logfreq_path)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] parameters: {n_params:,}  device: {args.device}")

    owt = load_tokens(args.val)
    r_owt = run_eval(model, owt, context=args.context, stride=args.stride,
                     batch=args.batch, device=args.device, n_chunks=args.n_chunks)
    _report("OpenWebText held-out (full)", r_owt)

    if args.wikitext:
        try:
            wt = load_wikitext103_tokens("test")
            r_wt = run_eval(model, wt, context=args.context, stride=args.stride,
                            batch=args.batch, device=args.device,
                            n_chunks=args.n_chunks)
            _report("WikiText-103 test (contamination cross-check)", r_wt)
            print("\n[interpret] If OWT PPL << WikiText PPL by much more than a "
                  "GPT-2-small-class model would show, suspect OWT train/val "
                  "contamination in the 4B pool.")
        except Exception as e:
            print(f"[wikitext] skipped ({e}). `pip install datasets tiktoken`.")


if __name__ == "__main__":
    main()
