"""
Training loop for the **paper_tmlr_1 scale-up pilot** (PARF-augmented SPLM arm).

This trainer is a hard-fork of `parf/train_parf.py` adapted for the
scale-up configuration locked by the existing E9 SPLM scale-up
protocol (`companion_notes/SPLM_scaleup_pre-registered_protocol.md`):
  - corpus      : TinyStories (~5 M GPT-2 BPE tokens)
  - max_len     : 1024
  - block_size  : 512
  - d / L / v_h : 256 / 8 / 1024
  - V_phi       : structural (P5 §5.1-faithful), per-pair Gumbel-softmax
                  top-k=4 sparse routing (P5 winner: 176.65 PPL on Shakespeare)
  - V_phi inner : phi_hidden=128, theta_hidden=128, d_type=32, d_angle=16
  - mass        : logfreq, alpha-init 0.1, surprisal computed on TinyStories
  - damping     : free gamma, init 0.15
  - steps       : 8000   batch 16   lr 5e-4 cosine, 400-step warmup
  - eval        : every 400 steps, 40 batches × batch 16 × block 512
  - Gumbel τ    : 1.0 -> 0.1 over the last 80% of training (P5 default)

Modes
-----
  --mode smoke    : 300-step pipeline-correctness verification (no PPL claim).
  --mode scaleup  : full pilot run (8000 steps, 400 warmup, 400 eval).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Set the CUDA allocator config BEFORE torch creates a CUDA context.
# expandable_segments=True allows the caching allocator to grow/shrink
# segments to reduce fragmentation; harmless when memory is genuinely
# over-subscribed but useful as defense in depth for the PARF backward
# pass which allocates many same-shape (B, T, T, H) gradient tensors.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).parent
PARENT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PARF_DIR = PARENT_DIR / "parf"
sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(PARF_DIR))

from data_module import get_batch, load_tiny_stories  # noqa: E402
from model_parf import PARFConfig, PARFLM  # noqa: E402
from model_parf_sparse import SparsePARFConfig, SparsePARFLM  # noqa: E402

# Re-assert PARF_DIR at the front because the model imports above mutate
# sys.path[0] = PARENT_DIR (see _PARENT_DIR insert in model_parf.py).
sys.path.insert(0, str(PARF_DIR))
from causal_probe_parf import assert_causal  # noqa: E402

DEFAULT_LOGFREQ_PATH = SCRIPT_DIR / "results" / "logfreq_surprisal_tinystories.npy"


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_config(
    mode: str,
    v_phi_kind: str,
    logfreq_path: str | None,
    fixed_gamma: float | None = None,
    use_grad_checkpoint: bool = False,
    sparse_top_k: int | None = 4,
    sparse_score_head_hidden: int = 32,
    sparse_gumbel_tau_init: float = 1.0,
    sparse_gumbel_tau_min: float = 0.1,
    sparse_gumbel_noise: bool = True,
    v_phi_phi_hidden: int | None = None,
    v_phi_theta_hidden: int | None = None,
    v_phi_mlp_hidden: int | None = None,
    competitive_temp: float = 1.0,
    competitive_scale: str = "row",
    ln_before_distance: bool = False,
    per_layer_v_phi_scale: bool = False,
    per_layer_scale_init: float = -3.0,
    theta_activation: str = "tanh",
    theta_form: str = "mlp",
    v_hidden: int | None = None,
    v_depth: int | None = None,
) -> tuple[PARFConfig | SparsePARFConfig, dict, str]:
    # ------------------------------------------------------------------
    # V_phi memory budget (read this before changing scaleup defaults!)
    # ------------------------------------------------------------------
    # Inside V_phi.forward the structural variant materialises TWO
    # (B, T, T, H) tensors: phi_c_net hidden (line 286 of model_parf.py)
    # and theta_w hidden (line 311).  At scaleup (B=16, T=512) each
    # 4-byte (B, T, T, H) costs 16 * 512 * 512 * H * 4 bytes.
    #
    #   H = 128 -> 2.0 GiB per tensor, so ~4 GiB per layer (peak)
    #   H =  32 -> 0.5 GiB per tensor, so ~1 GiB per layer (peak)
    #
    # Crucially, --grad-checkpoint does NOT reduce the steady-state
    # footprint here.  V_phi is wrapped in checkpoint(use_reentrant=False)
    # but the SPLM layer-step computes the force via
    #   torch.autograd.grad(U, h_in, create_graph=True)
    # The create_graph=True forces the V_phi forward graph to be
    # retained for the second-order backward through loss.backward(),
    # so the intermediates that checkpoint normally discards are
    # re-created and held alive anyway.  This combination is a known
    # corner case in the PyTorch checkpoint docs.
    #
    # The validated small-scale PARF runs used H=128 at B=8, T=256
    # (per-layer V_phi peak ~268 MiB).  Scaleup increases B*T^2 by 8x,
    # so at H=128 the per-layer peak grows to ~2.0 GiB and 8 layers
    # blow past the 40 GiB A100 budget.
    #
    # H=32 already cleared the forward OOM, but the BACKWARD path still
    # ran out of room: torch.autograd.grad(create_graph=True) needs to
    # allocate gradient tensors the same shape as every saved (B, T, T, H)
    # intermediate, AND it retains a second-order graph that doubles the
    # forward state.  At H=32 / B=16 / T=512 / L=8 this came out to
    # ~38.8 GiB allocated when trying to fit a 0.5 GiB grad buffer (~350
    # MiB short).  We drop to H=16 to halve that and leave ~5 GiB of
    # headroom on a 40 GiB A100.
    # ------------------------------------------------------------------
    base_kw = dict(
        vocab_size=50257,
        mass_mode="logfreq",
        logfreq_init_alpha=0.1,
        logfreq_path=logfreq_path,
        v_phi_kind=v_phi_kind,
        causal_force=True,
        ln_after_step=True,
        fixed_gamma=fixed_gamma,
        use_grad_checkpoint=use_grad_checkpoint,
        # ----- P7 (Lever 3) and P8 (Levers 1.5/4/5) knobs -----
        # All default OFF/identity so a `--v-phi-kind structural --top-k 4`
        # command (P10a) is byte-identical to the pre-P7/P8 trainer.
        v_phi_competitive_temp=competitive_temp,
        v_phi_competitive_scale=competitive_scale,
        ln_before_distance=ln_before_distance,
        per_layer_v_phi_scale=per_layer_v_phi_scale,
        per_layer_scale_init=per_layer_scale_init,
        theta_activation=theta_activation,
        theta_form=theta_form,
    )
    if mode == "smoke":
        # Smoke fits at H=128 because B=8, T=256 keeps (B, T, T, H) small.
        mode_phi_hidden = 128
        mode_theta_hidden = 128
        mode_mlp_hidden = 256
        base_kw.update(
            d=256, max_len=1024, L=8,
            v_hidden=1024, v_depth=3,
            v_phi_d_type=32, v_phi_d_angle=16,
        )
        train_cfg = dict(
            batch_size=8, block_size=256,
            steps=300, lr=5e-4, weight_decay=0.01,
            warmup_steps=20, grad_clip=1.0,
            eval_interval=100, eval_iters=10,
            log_interval=10,
        )
    elif mode == "scaleup":
        # H=16 keeps per-layer V_phi peak ~256 MiB (forward) + same-shape
        # gradient buffers, which leaves ~5 GiB of headroom on a 40 GiB
        # A100 once the create_graph=True second-order graph is included.
        # Override at your own peril -- and bring extra GPU memory.
        mode_phi_hidden = 16
        mode_theta_hidden = 16
        mode_mlp_hidden = 32
        base_kw.update(
            d=256, max_len=1024, L=8,
            v_hidden=1024, v_depth=3,
            v_phi_d_type=32, v_phi_d_angle=16,
        )
        train_cfg = dict(
            batch_size=16, block_size=512,
            steps=8000, lr=5e-4, weight_decay=0.01,
            warmup_steps=400, grad_clip=1.0,
            eval_interval=400, eval_iters=40,
            log_interval=50,
        )
    else:
        raise ValueError(f"unknown mode: {mode}")

    # CLI overrides win over mode defaults (set to None to use defaults).
    phi_hidden = mode_phi_hidden if v_phi_phi_hidden is None else int(v_phi_phi_hidden)
    theta_hidden = mode_theta_hidden if v_phi_theta_hidden is None else int(v_phi_theta_hidden)
    mlp_hidden = mode_mlp_hidden if v_phi_mlp_hidden is None else int(v_phi_mlp_hidden)
    base_kw.update(
        v_phi_phi_hidden=phi_hidden,
        v_phi_theta_hidden=theta_hidden,
        v_phi_mlp_hidden=mlp_hidden,
    )
    # ----- V_theta width / depth overrides (P10f and beyond) -----
    # The mode block has already set v_hidden/v_depth in base_kw above; if
    # the caller supplied an override, replace those.  Useful for the
    # V_theta-ceiling ablation (P10f doubles v_hidden 1024 -> 2048).
    if v_hidden is not None:
        base_kw["v_hidden"] = int(v_hidden)
    if v_depth is not None:
        base_kw["v_depth"] = int(v_depth)

    gc_tag = "_gc" if use_grad_checkpoint else ""

    # ----- P7 competitive-Φ tag -----
    if v_phi_kind == "structural_competitive":
        ct_tag = (
            f"_ct{competitive_temp:g}".rstrip("0").rstrip(".")
            if abs(competitive_temp - 1.0) > 1e-9 else ""
        )
        cs_tag = "" if competitive_scale == "row" else f"_cs-{competitive_scale}"
        comp_tag = ct_tag + cs_tag
    else:
        comp_tag = ""

    # ----- P8 patch tag suffix (composable, alphabetical for stability) -----
    p8_parts = []
    if ln_before_distance:
        p8_parts.append("lnD")
    if per_layer_v_phi_scale:
        p8_parts.append("pls")
    if str(theta_activation).lower() == "softsign":
        p8_parts.append("\u03B8ss")  # 'θss' — softsign Θ
    if str(theta_form).lower() == "bilinear":
        p8_parts.append("\u03B8bl")  # 'θbl' — bilinear Θ
    p8_tag = ("_" + "-".join(p8_parts)) if p8_parts else ""

    if sparse_top_k is not None:
        cfg = SparsePARFConfig(
            **base_kw,
            top_k=int(sparse_top_k),
            score_head_hidden=int(sparse_score_head_hidden),
            gumbel_tau_init=float(sparse_gumbel_tau_init),
            gumbel_tau_min=float(sparse_gumbel_tau_min),
            gumbel_noise=bool(sparse_gumbel_noise),
        )
        sparse_tag = f"_sparse_k{sparse_top_k}"
    else:
        cfg = PARFConfig(**base_kw)
        sparse_tag = ""

    fg_tag = "" if fixed_gamma is None else f"_g{fixed_gamma:.3f}"
    tag = (f"parf_{v_phi_kind}_vphi{phi_hidden}{gc_tag}"
           f"{fg_tag}{comp_tag}{p8_tag}{sparse_tag}_scaleup_{mode}")
    return cfg, train_cfg, tag


def lr_schedule(step: int, lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def tau_schedule(step: int, tau_init: float, tau_min: float,
                 total_steps: int, anneal_fraction: float = 0.8) -> float:
    """Linear anneal of the Gumbel-softmax temperature; matches train_parf.py."""
    warm = int((1.0 - anneal_fraction) * total_steps)
    if step < warm:
        return tau_init
    if step >= total_steps:
        return tau_min
    progress = (step - warm) / max(total_steps - warm, 1)
    return tau_init + (tau_min - tau_init) * min(progress, 1.0)


@torch.no_grad()
def evaluate(model, ids: np.ndarray, iters: int,
             batch_size: int, block_size: int,
             rng: np.random.Generator, device: str) -> float:
    model.eval()
    losses = []
    for _ in range(iters):
        xb, yb = get_batch(ids, batch_size, block_size, rng)
        x = torch.from_numpy(xb).to(device)
        y = torch.from_numpy(yb).to(device)
        with torch.enable_grad():
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "scaleup"], default="smoke")
    ap.add_argument("--v-phi-kind",
                    choices=["structural", "structural_competitive", "mlp"],
                    default="structural", dest="v_phi_kind",
                    help="Inner shape of V_phi: 'structural' is the "
                         "§5.1-faithful default; 'structural_competitive' "
                         "is the Lever-3 row-softmax variant (P7); 'mlp' "
                         "is the unstructured MLP ablation.")
    ap.add_argument("--fixed-gamma", type=float, default=None,
                    dest="fixed_gamma")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    dest="grad_checkpoint",
                    help="Gradient-checkpoint the V_phi pair sum.")
    ap.add_argument("--logfreq-path", dest="logfreq_path",
                    default=str(DEFAULT_LOGFREQ_PATH))
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train-tokens", dest="max_train_tokens",
                    type=int, default=5_000_000)
    ap.add_argument("--tag-suffix", dest="tag_suffix", type=str, default="")
    ap.add_argument("--results-dir", dest="results_dir", type=str, default=None)
    ap.add_argument("--skip-causal-check", action="store_true")
    ap.add_argument("--top-k", type=int, default=4, dest="top_k",
                    help="Top-k Gumbel-softmax sparse pair routing. "
                         "Default 4 = P5 winner at small scale. "
                         "Pass --top-k 0 to disable sparsity (dense fallback).")
    ap.add_argument("--score-head-hidden", type=int, default=32,
                    dest="score_head_hidden")
    ap.add_argument("--gumbel-tau-init", type=float, default=1.0,
                    dest="gumbel_tau_init")
    ap.add_argument("--gumbel-tau-min", type=float, default=0.1,
                    dest="gumbel_tau_min")
    ap.add_argument("--gumbel-anneal-fraction", type=float, default=0.8,
                    dest="gumbel_anneal_fraction")
    ap.add_argument("--no-gumbel-noise", action="store_false",
                    dest="gumbel_noise", default=True)
    ap.add_argument("--v-phi-phi-hidden", type=int, default=None,
                    dest="v_phi_phi_hidden",
                    help="Override V_phi inner phi_c_net hidden width "
                         "(controls (B,T,T,H) phi-side intermediate). "
                         "Scaleup default is 32 to keep memory in budget.")
    ap.add_argument("--v-phi-theta-hidden", type=int, default=None,
                    dest="v_phi_theta_hidden",
                    help="Override V_phi inner theta_w hidden width "
                         "(controls (B,T,T,H) theta-side intermediate). "
                         "Scaleup default is 32.")
    ap.add_argument("--v-phi-mlp-hidden", type=int, default=None,
                    dest="v_phi_mlp_hidden",
                    help="Override V_phi MLP hidden (used by the MLP V_phi "
                         "ablation; scaleup default is 32).")
    ap.add_argument("--v-hidden", type=int, default=None, dest="v_hidden",
                    help="Override V_theta MLP hidden width (the single "
                         "shared ScalarPotential).  Scaleup default is 1024; "
                         "P10f bumps to 2048 to test the V_theta-ceiling "
                         "hypothesis.  V_theta params scale ~ v_hidden^2.")
    ap.add_argument("--v-depth", type=int, default=None, dest="v_depth",
                    help="Override V_theta MLP depth.  Scaleup default is 3.")
    ap.add_argument("--grad-accum", type=int, default=1, dest="grad_accum",
                    help="Number of micro-batches per optimiser step.  "
                         "batch_size is split into N equal-sized micro-"
                         "batches, each forward+backward is performed "
                         "independently, and gradients are accumulated "
                         "before optim.step().  Effective batch size is "
                         "preserved (= train_cfg.batch_size).  Use this "
                         "to fit the PARF outer loss.backward() on GPUs "
                         "with limited memory.  batch_size must be "
                         "divisible by grad_accum.")
    # ----- P7 (Lever 3) competitive-Φ knobs -----
    ap.add_argument("--v-phi-competitive-temp", type=float, default=1.0,
                    dest="v_phi_competitive_temp",
                    help="Softmax temperature τ for competitive Φ̃_φ "
                         "(only used with --v-phi-kind structural_competitive).")
    ap.add_argument("--v-phi-competitive-scale",
                    choices=["row", "mean", "none"], default="row",
                    dest="v_phi_competitive_scale",
                    help="Post-softmax rescale of Φ̃_φ: 'row' multiplies "
                         "by per-row causal count (default; preserves the "
                         "scale of the unnormalised dense sum); 'mean' "
                         "leaves Σ Φ̃ = 1 per row; 'none' is the "
                         "diagnostic-only no-rescale case.")
    # ----- P8 cell knobs (compose with structural / structural_competitive) -----
    ap.add_argument("--ln-before-distance", action="store_true",
                    dest="ln_before_distance",
                    help="P8 patch A: replace ‖h_t-h_s‖ with "
                         "‖LN(h_t)-LN(h_s)‖ inside V_phi.  Decouples 1/r "
                         "from per-layer ‖h‖ growth.  Tag gains 'lnD'.")
    ap.add_argument("--per-layer-v-phi-scale", action="store_true",
                    dest="per_layer_v_phi_scale",
                    help="P8 patch B: learnable s_ℓ = softplus(σ_ℓ) "
                         "per integrator layer multiplies the V_phi "
                         "contribution to U.  Tag gains 'pls'.")
    ap.add_argument("--per-layer-scale-init", type=float, default=-3.0,
                    dest="per_layer_scale_init",
                    help="Initial logit σ_ℓ for the per-layer V_phi scale "
                         "(softplus(-3) ≈ 0.0486).  Only consulted with "
                         "--per-layer-v-phi-scale.")
    ap.add_argument("--theta-activation",
                    choices=["tanh", "softsign"], default="tanh",
                    dest="theta_activation",
                    help="P8 patch C: bounded activation for Θ_φ.  "
                         "'tanh' (default) saturates exponentially; "
                         "'softsign' x/(1+|x|) saturates polynomially.  "
                         "Tag gains 'θss' for softsign.")
    ap.add_argument("--theta-form", choices=["mlp", "bilinear"],
                    default="mlp", dest="theta_form",
                    help="P8 patch D: parameterisation of Θ_φ.  'mlp' "
                         "(default) is the 3K→H→1 GELU MLP; 'bilinear' "
                         "is θ_t^T W θ_s + b — K^2+1 params, gradient-"
                         "bounded.  Tag gains 'θbl' for bilinear.")
    args = ap.parse_args()

    device = args.device or _pick_device()
    results_dir = (
        Path(args.results_dir).expanduser().resolve()
        if args.results_dir is not None else RESULTS_DIR
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    # Disable TF32 for SPLM-family models on CUDA: the PARF forward uses
    # torch.autograd.grad through V_phi to compute the property/attractive/
    # repulsive force, and that second-order path is more sensitive to TF32's
    # 10-bit-mantissa reduction than a single attention forward.  Disabling
    # TF32 forces real fp32 matmuls (23-bit mantissa) at the cost of ~2x
    # slower matmul.  Worth it.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print("[scaleup-parf] TF32 disabled for PARF autograd.grad "
              "numerical stability (CUDA matmuls in true fp32)")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train_ids, val_ids = load_tiny_stories(
        max_train_tokens=args.max_train_tokens,
    )
    print(f"[scaleup-parf] tokens: train={len(train_ids):,}  "
          f"val={len(val_ids):,}")

    logfreq_path = args.logfreq_path
    if not Path(logfreq_path).exists():
        raise FileNotFoundError(
            f"logfreq surprisal file not found at {logfreq_path}.  "
            "Run scaleup/compute_unigram_frequencies_tinystories.py first."
        )

    sparse_top_k = args.top_k if args.top_k and args.top_k > 0 else None
    cfg, train_cfg, base_tag = build_config(
        args.mode, args.v_phi_kind, logfreq_path,
        fixed_gamma=args.fixed_gamma,
        use_grad_checkpoint=args.grad_checkpoint,
        sparse_top_k=sparse_top_k,
        sparse_score_head_hidden=args.score_head_hidden,
        sparse_gumbel_tau_init=args.gumbel_tau_init,
        sparse_gumbel_tau_min=args.gumbel_tau_min,
        sparse_gumbel_noise=args.gumbel_noise,
        v_phi_phi_hidden=args.v_phi_phi_hidden,
        v_phi_theta_hidden=args.v_phi_theta_hidden,
        v_phi_mlp_hidden=args.v_phi_mlp_hidden,
        competitive_temp=args.v_phi_competitive_temp,
        competitive_scale=args.v_phi_competitive_scale,
        ln_before_distance=args.ln_before_distance,
        per_layer_v_phi_scale=args.per_layer_v_phi_scale,
        per_layer_scale_init=args.per_layer_scale_init,
        theta_activation=args.theta_activation,
        theta_form=args.theta_form,
        v_hidden=args.v_hidden,
        v_depth=args.v_depth,
    )
    tag = base_tag
    if args.tag_suffix:
        tag = f"{tag}_{args.tag_suffix}"
    is_sparse = isinstance(cfg, SparsePARFConfig)
    print(f"[scaleup-parf] device={device}  tag={tag}  "
          f"variant={'Q9c-sparse-stage1.5' if is_sparse else 'Q9c-dense-stage1'}")
    print(f"[scaleup-parf] arch: V_phi={cfg.v_phi_kind!r}  L={cfg.L}  "
          f"d={cfg.d}  v_hidden={cfg.v_hidden}  v_depth={cfg.v_depth}  "
          f"max_len={cfg.max_len}  fixed_gamma={cfg.fixed_gamma}  "
          f"use_grad_checkpoint={cfg.use_grad_checkpoint}")
    print(f"[scaleup-parf] V_phi widths: phi_hidden={cfg.v_phi_phi_hidden}  "
          f"theta_hidden={cfg.v_phi_theta_hidden}  "
          f"mlp_hidden={cfg.v_phi_mlp_hidden}  "
          f"d_type={cfg.v_phi_d_type}  d_angle={cfg.v_phi_d_angle}")
    if is_sparse:
        print(f"[scaleup-parf] sparse: top_k={cfg.top_k}  "
              f"gumbel_tau {cfg.gumbel_tau_init} -> {cfg.gumbel_tau_min}  "
              f"anneal_fraction={args.gumbel_anneal_fraction}")

    if is_sparse:
        model = SparsePARFLM(cfg).to(device)
    else:
        model = PARFLM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_v_phi = sum(p.numel() for p in model.V_phi.parameters())
    n_v_theta = sum(p.numel() for p in model.V_theta.parameters())
    n_score = (
        sum(p.numel() for p in model.score_head.parameters())
        if is_sparse else 0
    )
    score_str = f"  score_head={n_score:,}" if is_sparse else ""
    print(f"[scaleup-parf] params: {n_params:,}  "
          f"V_theta={n_v_theta:,}  V_phi={n_v_phi:,}{score_str}")

    if not args.skip_causal_check:
        print(f"[scaleup-parf] running causal-violation probe...")
        try:
            assert_causal(
                model, vocab_size=cfg.vocab_size,
                T=32, t_pert=20, seed=args.seed,
            )
            print(f"[scaleup-parf] causal probe PASSED")
        except RuntimeError as exc:
            print(f"[scaleup-parf] causal probe FAILED — aborting.")
            print(f"[scaleup-parf] {exc}")
            raise SystemExit(2)

    optim = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"], betas=(0.9, 0.95),
    )

    # Gradient accumulation: split batch_size into N equal micro-batches.
    # Effective batch is preserved.  Memory per micro-batch scales 1/N.
    grad_accum = max(1, int(args.grad_accum))
    if train_cfg["batch_size"] % grad_accum != 0:
        raise ValueError(
            f"batch_size {train_cfg['batch_size']} not divisible by "
            f"--grad-accum {grad_accum}."
        )
    micro_batch = train_cfg["batch_size"] // grad_accum
    if grad_accum > 1:
        print(f"[scaleup-parf] grad-accum: {grad_accum} micro-batches of "
              f"size {micro_batch} per optim step "
              f"(effective batch = {train_cfg['batch_size']})")

    log_path = results_dir / f"{tag}_training_log.jsonl"
    log_f = log_path.open("w")
    loss_history: list[tuple[int, float, float]] = []

    t0 = time.time()
    model.train()
    running = 0.0
    n_run = 0

    for step in range(train_cfg["steps"]):
        lr_now = lr_schedule(step, train_cfg["lr"],
                             train_cfg["warmup_steps"], train_cfg["steps"])
        for g in optim.param_groups:
            g["lr"] = lr_now

        if is_sparse:
            tau_now = tau_schedule(
                step,
                tau_init=cfg.gumbel_tau_init,
                tau_min=cfg.gumbel_tau_min,
                total_steps=train_cfg["steps"],
                anneal_fraction=args.gumbel_anneal_fraction,
            )
            model.set_gumbel_tau(tau_now)

        optim.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _micro in range(grad_accum):
            xb, yb = get_batch(train_ids, micro_batch,
                               train_cfg["block_size"], rng)
            x = torch.from_numpy(xb).to(device)
            y = torch.from_numpy(yb).to(device)

            _, loss = model(x, y)
            # Average gradients across micro-batches so the effective
            # gradient matches a single full-batch step.
            (loss / grad_accum).backward()
            accum_loss += loss.item()

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(),
                                             train_cfg["grad_clip"])
        optim.step()

        # Report the average loss across micro-batches (matches the
        # full-batch loss in expectation).
        running += accum_loss / grad_accum
        n_run += 1

        if (step + 1) % train_cfg["log_interval"] == 0:
            avg = running / n_run
            running, n_run = 0.0, 0
            elapsed = time.time() - t0
            tau_str = (f"   tau={model.gumbel_tau:.3f}"
                       if is_sparse else "")
            print(f"[scaleup-parf] step {step+1:5d}/{train_cfg['steps']}   "
                  f"train {avg:.4f}   lr {lr_now:.2e}   "
                  f"grad {grad_norm:.2f}   "
                  f"gamma={model.gamma.item():.3f}{tau_str}   "
                  f"elapsed {elapsed:.0f}s")
            log_record = {
                "step": step + 1, "train_loss": avg,
                "lr": lr_now, "grad_norm": float(grad_norm),
                "gamma": model.gamma.item(),
                "elapsed_s": elapsed,
            }
            if is_sparse:
                log_record["gumbel_tau"] = model.gumbel_tau
            log_f.write(json.dumps(log_record) + "\n")
            log_f.flush()

        if ((step + 1) % train_cfg["eval_interval"] == 0
                or step + 1 == train_cfg["steps"]):
            val_loss = evaluate(
                model, val_ids, train_cfg["eval_iters"],
                train_cfg["batch_size"], train_cfg["block_size"],
                rng, device,
            )
            ppl = math.exp(val_loss)
            print(f"[scaleup-parf] >>> eval @ {step+1}: "
                  f"val {val_loss:.4f}   ppl {ppl:.2f}")
            log_f.write(json.dumps({
                "step": step + 1,
                "val_loss": val_loss, "val_ppl": ppl,
            }) + "\n")
            log_f.flush()
            loss_history.append((step + 1, avg, val_loss))

    log_f.close()
    final_val = evaluate(model, val_ids,
                         train_cfg["eval_iters"],
                         train_cfg["batch_size"],
                         train_cfg["block_size"], rng, device)
    final_ppl = math.exp(final_val)
    final_gamma = float(model.gamma.item())
    total_elapsed = time.time() - t0
    print(f"\n[scaleup-parf] DONE  val_loss={final_val:.4f}  "
          f"val_ppl={final_ppl:.2f}  gamma={final_gamma:.4f}  "
          f"elapsed={total_elapsed:.0f}s")

    ckpt_path = results_dir / f"{tag}_ckpt_latest.pt"
    ckpt = {
        "model_state_dict": model.state_dict(),
        "model_cfg": asdict(cfg),
        "train_cfg": train_cfg,
        "loss_history": loss_history,
        "final_val_loss": final_val,
        "final_val_ppl": final_ppl,
        "final_gamma": final_gamma,
        "fixed_gamma": args.fixed_gamma,
        "max_train_tokens": args.max_train_tokens,
        "logfreq_path": str(logfreq_path),
        "variant": "parf_q9c_sparse_stage1.5" if is_sparse else "parf_q9c_dense",
        "v_phi_kind": cfg.v_phi_kind,
        "experiment": "tmlr1_scaleup_pilot",
        "tag": tag,
        "seed": args.seed,
        "n_params": n_params,
        "n_v_theta_params": n_v_theta,
        "n_v_phi_params": n_v_phi,
        "elapsed_sec": total_elapsed,
    }
    if is_sparse:
        ckpt["n_score_head_params"] = n_score
        ckpt["final_gumbel_tau"] = model.gumbel_tau
        ckpt["top_k"] = cfg.top_k
    torch.save(ckpt, ckpt_path)
    print(f"[scaleup-parf] saved checkpoint -> {ckpt_path}")

    fig, ax = plt.subplots(figsize=(7, 4))
    if loss_history:
        steps_e = [e[0] for e in loss_history]
        va_e = [e[2] for e in loss_history]
        ax.plot(steps_e, [math.exp(v) for v in va_e],
                marker="s", label="val ppl", color="purple")
    ax.set_xlabel("step")
    ax.set_ylabel("perplexity")
    sparse_str = (f"sparse k={cfg.top_k}" if is_sparse else "dense")
    ax.set_title(f"PARF Q9c V_phi={cfg.v_phi_kind} {sparse_str} scale-up "
                 f"— {args.mode} — seed={args.seed}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    png_path = results_dir / f"{tag}_loss_curve.png"
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    print(f"[scaleup-parf] saved loss curve -> {png_path}")

    summary_path = results_dir / f"{tag}_summary.md"
    with summary_path.open("w") as f:
        f.write(f"# Training summary — {tag}\n\n")
        f.write(f"- experiment: paper_tmlr_1 scale-up pilot (PARF Q9c arm)\n")
        f.write(f"- model: {'SparsePARFLM' if is_sparse else 'PARFLM'}\n")
        f.write(f"- v_phi_kind: {cfg.v_phi_kind}\n")
        if is_sparse:
            f.write(f"- top_k (Gumbel-softmax sparse): {cfg.top_k}\n")
            f.write(f"- gumbel_tau: {cfg.gumbel_tau_init} -> "
                    f"{cfg.gumbel_tau_min}  (anneal "
                    f"{args.gumbel_anneal_fraction})\n")
        f.write(f"- mode: {args.mode}\n")
        f.write(f"- corpus: TinyStories (cap {args.max_train_tokens:,} train tokens)\n")
        f.write(f"- fixed_gamma: {args.fixed_gamma}\n")
        f.write(f"- params: {n_params:,}  V_theta={n_v_theta:,}  "
                f"V_phi={n_v_phi:,}\n")
        f.write(f"- d={cfg.d}  L={cfg.L}  v_hidden={cfg.v_hidden}  "
                f"v_depth={cfg.v_depth}  max_len={cfg.max_len}\n")
        f.write(f"- v_phi inner: phi_hidden={cfg.v_phi_phi_hidden}  "
                f"theta_hidden={cfg.v_phi_theta_hidden}  "
                f"d_type={cfg.v_phi_d_type}  d_angle={cfg.v_phi_d_angle}\n")
        f.write(f"- block_size: {train_cfg['block_size']}  "
                f"batch_size: {train_cfg['batch_size']}  "
                f"steps: {train_cfg['steps']}\n")
        f.write(f"- seed: {args.seed}\n")
        f.write(f"- elapsed: {total_elapsed:.0f} s "
                f"({total_elapsed/3600:.2f} h)\n")
        f.write(f"\nFinal val loss: {final_val:.6f} (ppl {final_ppl:.2f})\n")
        f.write(f"Final gamma: {final_gamma:.4f}\n")
    print(f"[scaleup-parf] summary written to {summary_path}")


if __name__ == "__main__":
    main()
