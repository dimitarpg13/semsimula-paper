"""
Training loop for the **multi-channel ξ PARF** scale-up experiment.

This trainer is a hard-fork of `train_parf_scaleup.py` with the model
class swapped:

    SparsePARFLM          (single causal mean ξ)
    →
    MultiXiPARFLM         (K-channel K-EMA ξ  +  sparse PARF pair forces)

All PARF hyperparameters (V_φ, score head, Gumbel-softmax, P8 patches)
are inherited unchanged.  The only additions are the K-EMA α-init flags
from the multi-ξ SPLM family (--xi-channels, --xi-alpha-inits, etc.).

Hypothesis
----------
The multi-channel K-EMA ξ alone (without PARF) reaches ~14.78 PPL at
4000 steps.  The existing PARF + single ξ reaches ~28 PPL at 8000 steps.
If the ξ-bottleneck was the binding constraint on PARF, this hybrid
should materially beat 14.78 PPL — the pair-exchange forces would add
information that K-EMA ξ alone cannot capture.

Modes
-----
  --mode smoke    : 300-step pipeline-correctness verification.
  --mode scaleup  : full 8000-step run (matches E9 schedule).
  --mode pilot    : 4000-step half-schedule for quick comparison.
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
MULTIXI_DIR = PARENT_DIR / "multixi"
sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(PARF_DIR))
sys.path.insert(0, str(MULTIXI_DIR))

from data_module import get_batch, load_tiny_stories  # noqa: E402
from model_parf_multixi import (  # noqa: E402
    MultiXiPARFConfig,
    MultiXiPARFLM,
)

DEFAULT_LOGFREQ_PATH = SCRIPT_DIR / "results" / "logfreq_surprisal_tinystories.npy"


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _parse_alpha_list(s: str) -> list[float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out = []
    for p in parts:
        try:
            v = float(p)
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                f"--xi-alpha-inits got non-float component {p!r}"
            ) from e
        if not (0.0 <= v <= 1.0):
            raise argparse.ArgumentTypeError(
                f"--xi-alpha-inits component {v} not in [0, 1]"
            )
        out.append(v)
    if not out:
        raise argparse.ArgumentTypeError("--xi-alpha-inits is empty")
    return out


def build_config(
    mode: str,
    v_phi_kind: str,
    logfreq_path: str | None,
    *,
    fixed_gamma: float | None = None,
    use_grad_checkpoint: bool = False,
    use_layer_checkpoint: bool = False,
    use_gathered_v_phi: bool = False,
    sparse_top_k: int = 4,
    score_head_hidden: int = 32,
    gumbel_tau_init: float = 1.0,
    gumbel_tau_min: float = 0.1,
    gumbel_noise: bool = True,
    v_phi_phi_hidden: int | None = None,
    v_phi_theta_hidden: int | None = None,
    competitive_temp: float = 1.0,
    competitive_scale: str = "row",
    ln_before_distance: bool = False,
    per_layer_v_phi_scale: bool = False,
    per_layer_scale_init: float = -3.0,
    theta_activation: str = "tanh",
    theta_form: str = "mlp",
    xi_channels: int = 4,
    xi_alpha_inits: list[float] | None = None,
    xi_learnable: bool = True,
    xi_alpha_init_mode: str = "explicit",
    xi_tau_max: float = 100.0,
) -> tuple[MultiXiPARFConfig, dict, str]:

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
        use_layer_checkpoint=use_layer_checkpoint,
        v_phi_competitive_temp=competitive_temp,
        v_phi_competitive_scale=competitive_scale,
        ln_before_distance=ln_before_distance,
        per_layer_v_phi_scale=per_layer_v_phi_scale,
        per_layer_scale_init=per_layer_scale_init,
        theta_activation=theta_activation,
        theta_form=theta_form,
        # Sparse PARF knobs
        top_k=sparse_top_k,
        score_head_hidden=score_head_hidden,
        gumbel_tau_init=gumbel_tau_init,
        gumbel_tau_min=gumbel_tau_min,
        gumbel_noise=gumbel_noise,
        use_gathered_v_phi=use_gathered_v_phi,
        # Multi-xi knobs
        xi_channels=xi_channels,
        xi_learnable=xi_learnable,
        xi_alpha_init_mode=xi_alpha_init_mode,
        xi_tau_max=xi_tau_max,
    )

    if xi_alpha_init_mode == "explicit":
        if xi_alpha_inits is None:
            xi_alpha_inits = [0.0, 0.5, 0.9, 0.99]
        base_kw["xi_alpha_inits"] = xi_alpha_inits
    else:
        base_kw["xi_alpha_inits"] = [0.0] * xi_channels

    if mode == "smoke":
        mode_phi_hidden = 128
        mode_theta_hidden = 128
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
        mode_phi_hidden = 16
        mode_theta_hidden = 16
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
    elif mode == "pilot":
        mode_phi_hidden = 16
        mode_theta_hidden = 16
        base_kw.update(
            d=256, max_len=1024, L=8,
            v_hidden=1024, v_depth=3,
            v_phi_d_type=32, v_phi_d_angle=16,
        )
        train_cfg = dict(
            batch_size=16, block_size=512,
            steps=4000, lr=5e-4, weight_decay=0.01,
            warmup_steps=200, grad_clip=1.0,
            eval_interval=200, eval_iters=40,
            log_interval=50,
        )
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    phi_hidden = mode_phi_hidden if v_phi_phi_hidden is None else int(v_phi_phi_hidden)
    theta_hidden = mode_theta_hidden if v_phi_theta_hidden is None else int(v_phi_theta_hidden)
    base_kw.update(
        v_phi_phi_hidden=phi_hidden,
        v_phi_theta_hidden=theta_hidden,
        v_phi_mlp_hidden=32,
    )

    cfg = MultiXiPARFConfig(**base_kw)

    p8_parts = []
    if ln_before_distance:
        p8_parts.append("lnD")
    if per_layer_v_phi_scale:
        p8_parts.append("pls")
    if str(theta_activation).lower() == "softsign":
        p8_parts.append("\u03B8ss")
    if str(theta_form).lower() == "bilinear":
        p8_parts.append("\u03B8bl")
    p8_tag = ("_" + "-".join(p8_parts)) if p8_parts else ""

    fg_tag = "" if fixed_gamma is None else f"_g{fixed_gamma:.3f}"
    gc_tag = "_gc" if use_grad_checkpoint else ""
    lc_tag = "_lc" if use_layer_checkpoint else ""
    gv_tag = "_gv" if use_gathered_v_phi else ""
    tag = (f"parf_multixi_K{xi_channels}_{v_phi_kind}_vphi{phi_hidden}"
           f"{gc_tag}{lc_tag}{gv_tag}{fg_tag}{p8_tag}"
           f"_sparse_k{sparse_top_k}_{mode}")
    return cfg, train_cfg, tag


def lr_schedule(step: int, lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def tau_schedule(step: int, tau_init: float, tau_min: float,
                 total_steps: int, anneal_fraction: float = 0.8) -> float:
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
    ap = argparse.ArgumentParser(
        description="Multi-channel ξ PARF scale-up trainer"
    )
    ap.add_argument("--mode", choices=["smoke", "scaleup", "pilot"],
                    default="smoke")
    ap.add_argument("--v-phi-kind",
                    choices=["structural", "structural_competitive", "mlp"],
                    default="structural_competitive", dest="v_phi_kind")
    ap.add_argument("--fixed-gamma", type=float, default=0.30,
                    dest="fixed_gamma")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    dest="grad_checkpoint")
    ap.add_argument("--use-layer-checkpoint", action="store_true",
                    dest="use_layer_checkpoint",
                    help="Level-2 per-layer-step gradient checkpointing.")
    ap.add_argument("--use-gathered-v-phi", action="store_true",
                    dest="use_gathered_v_phi",
                    help="Stage-1.5b gathered V_phi (O(T*k) instead of O(T^2)).")
    ap.add_argument("--logfreq-path", dest="logfreq_path",
                    default=str(DEFAULT_LOGFREQ_PATH))
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train-tokens", dest="max_train_tokens",
                    type=int, default=5_000_000)
    ap.add_argument("--tag-suffix", dest="tag_suffix", type=str, default="")
    ap.add_argument("--results-dir", dest="results_dir", type=str,
                    default=None)
    ap.add_argument("--top-k", type=int, default=8, dest="top_k")
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
                    dest="v_phi_phi_hidden")
    ap.add_argument("--v-phi-theta-hidden", type=int, default=None,
                    dest="v_phi_theta_hidden")
    # P7 / P8 knobs
    ap.add_argument("--v-phi-competitive-temp", type=float, default=1.0,
                    dest="v_phi_competitive_temp")
    ap.add_argument("--v-phi-competitive-scale",
                    choices=["row", "mean", "none"], default="row",
                    dest="v_phi_competitive_scale")
    ap.add_argument("--ln-before-distance", action="store_true",
                    dest="ln_before_distance")
    ap.add_argument("--per-layer-v-phi-scale", action="store_true",
                    dest="per_layer_v_phi_scale")
    ap.add_argument("--per-layer-scale-init", type=float, default=-3.0,
                    dest="per_layer_scale_init")
    ap.add_argument("--theta-activation",
                    choices=["tanh", "softsign"], default="tanh",
                    dest="theta_activation")
    ap.add_argument("--theta-form", choices=["mlp", "bilinear"],
                    default="mlp", dest="theta_form")
    ap.add_argument("--grad-accum", type=int, default=1, dest="grad_accum")
    ap.add_argument("--max-steps", dest="max_steps", type=int, default=None)
    # ── Multi-xi knobs ──
    ap.add_argument("--xi-channels", dest="xi_channels", type=int, default=4)
    ap.add_argument("--xi-alpha-inits", dest="xi_alpha_inits",
                    type=_parse_alpha_list, default=None)
    ap.add_argument("--xi-frozen", dest="xi_learnable", action="store_false",
                    default=True)
    ap.add_argument("--xi-alpha-init-mode", dest="xi_alpha_init_mode",
                    choices=["explicit", "log_spaced"], default="explicit")
    ap.add_argument("--xi-tau-max", dest="xi_tau_max", type=float,
                    default=100.0)
    args = ap.parse_args()

    device = args.device or _pick_device()
    results_dir = (
        Path(args.results_dir).expanduser().resolve()
        if args.results_dir is not None else RESULTS_DIR
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print("[multixi-parf] TF32 disabled for autograd.grad stability")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train_ids, val_ids = load_tiny_stories(
        max_train_tokens=args.max_train_tokens,
    )
    print(f"[multixi-parf] tokens: train={len(train_ids):,}  "
          f"val={len(val_ids):,}")

    logfreq_path = args.logfreq_path
    if not Path(logfreq_path).exists():
        raise FileNotFoundError(
            f"logfreq surprisal file not found at {logfreq_path}.  "
            "Run scaleup/compute_unigram_frequencies_tinystories.py first."
        )

    cfg, train_cfg, base_tag = build_config(
        args.mode, args.v_phi_kind, logfreq_path,
        fixed_gamma=args.fixed_gamma,
        use_grad_checkpoint=args.grad_checkpoint,
        use_layer_checkpoint=args.use_layer_checkpoint,
        use_gathered_v_phi=args.use_gathered_v_phi,
        sparse_top_k=args.top_k,
        score_head_hidden=args.score_head_hidden,
        gumbel_tau_init=args.gumbel_tau_init,
        gumbel_tau_min=args.gumbel_tau_min,
        gumbel_noise=args.gumbel_noise,
        v_phi_phi_hidden=args.v_phi_phi_hidden,
        v_phi_theta_hidden=args.v_phi_theta_hidden,
        competitive_temp=args.v_phi_competitive_temp,
        competitive_scale=args.v_phi_competitive_scale,
        ln_before_distance=args.ln_before_distance,
        per_layer_v_phi_scale=args.per_layer_v_phi_scale,
        per_layer_scale_init=args.per_layer_scale_init,
        theta_activation=args.theta_activation,
        theta_form=args.theta_form,
        xi_channels=args.xi_channels,
        xi_alpha_inits=args.xi_alpha_inits,
        xi_learnable=args.xi_learnable,
        xi_alpha_init_mode=args.xi_alpha_init_mode,
        xi_tau_max=args.xi_tau_max,
    )
    if args.max_steps is not None:
        train_cfg["steps"] = args.max_steps
    tag = base_tag
    if args.tag_suffix:
        tag = f"{tag}_{args.tag_suffix}"

    model = MultiXiPARFLM(cfg).to(device)
    n_params = model.num_params()
    n_v_phi = sum(p.numel() for p in model.V_phi.parameters())
    n_v_theta = sum(p.numel() for p in model.V_theta.parameters())
    n_score = sum(p.numel() for p in model.score_head.parameters())
    n_xi = sum(p.numel() for p in model.xi_module.parameters())
    alpha_str = ", ".join(f"{a:.3f}" for a in model.xi_alpha_values())

    print(f"[multixi-parf] device={device}  tag={tag}")
    print(f"[multixi-parf] params: {n_params:,}  V_theta={n_v_theta:,}  "
          f"V_phi={n_v_phi:,}  score_head={n_score:,}  xi_module={n_xi:,}")
    print(f"[multixi-parf] arch: d={cfg.d}  L={cfg.L}  "
          f"v_hidden={cfg.v_hidden}  V_phi={cfg.v_phi_kind!r}  "
          f"top_k={cfg.top_k}  fixed_gamma={cfg.fixed_gamma}")
    print(f"[multixi-parf] xi: K={cfg.xi_channels}  "
          f"learnable={cfg.xi_learnable}  "
          f"mode={cfg.xi_alpha_init_mode}  α=[{alpha_str}]")
    print(f"[multixi-parf] schedule: {args.mode}  "
          f"steps={train_cfg['steps']}  batch={train_cfg['batch_size']}  "
          f"block={train_cfg['block_size']}")

    optim = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"], betas=(0.9, 0.95),
    )

    grad_accum = max(1, int(args.grad_accum))
    if train_cfg["batch_size"] % grad_accum != 0:
        raise ValueError(
            f"batch_size {train_cfg['batch_size']} not divisible by "
            f"--grad-accum {grad_accum}."
        )
    micro_batch = train_cfg["batch_size"] // grad_accum
    if grad_accum > 1:
        print(f"[multixi-parf] grad-accum: {grad_accum} micro-batches "
              f"of size {micro_batch}")

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
            (loss / grad_accum).backward()
            accum_loss += loss.item()

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(),
                                             train_cfg["grad_clip"])
        optim.step()

        running += accum_loss / grad_accum
        n_run += 1

        if (step + 1) % train_cfg["log_interval"] == 0:
            avg = running / n_run
            running, n_run = 0.0, 0
            elapsed = time.time() - t0
            alphas = model.xi_alpha_values()
            alpha_log_str = ",".join(f"{a:.3f}" for a in alphas)
            print(
                f"[multixi-parf] step {step+1:5d}/{train_cfg['steps']}   "
                f"train {avg:.4f}   lr {lr_now:.2e}   "
                f"grad {grad_norm:.2f}   "
                f"gamma={model.gamma.item():.3f}   "
                f"tau={model.gumbel_tau:.3f}   "
                f"α=[{alpha_log_str}]   "
                f"elapsed {elapsed:.0f}s"
            )
            log_f.write(json.dumps({
                "step": step + 1, "train_loss": avg,
                "lr": lr_now, "grad_norm": float(grad_norm),
                "gamma": model.gamma.item(),
                "gumbel_tau": model.gumbel_tau,
                "xi_alphas": alphas,
                "elapsed_s": elapsed,
            }) + "\n")
            log_f.flush()

        if ((step + 1) % train_cfg["eval_interval"] == 0
                or step + 1 == train_cfg["steps"]):
            val_loss = evaluate(
                model, val_ids, train_cfg["eval_iters"],
                train_cfg["batch_size"], train_cfg["block_size"],
                rng, device,
            )
            ppl = math.exp(val_loss)
            print(f"[multixi-parf] >>> eval @ {step+1}: "
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
    final_alphas = model.xi_alpha_values()
    total_elapsed = time.time() - t0
    print(
        f"\n[multixi-parf] DONE  val_loss={final_val:.4f}  "
        f"val_ppl={final_ppl:.2f}  gamma={final_gamma:.4f}  "
        f"α_final={final_alphas}  elapsed={total_elapsed:.0f}s"
    )

    ckpt_path = results_dir / f"{tag}_ckpt_latest.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_cfg": asdict(cfg),
        "train_cfg": train_cfg,
        "loss_history": loss_history,
        "final_val_loss": final_val,
        "final_val_ppl": final_ppl,
        "final_gamma": final_gamma,
        "final_xi_alphas": final_alphas,
        "fixed_gamma": args.fixed_gamma,
        "max_train_tokens": args.max_train_tokens,
        "logfreq_path": str(logfreq_path),
        "variant": "parf_multixi_sparse",
        "experiment": "multixi_parf_scaleup",
        "tag": tag,
        "seed": args.seed,
        "n_params": n_params,
        "n_v_theta_params": n_v_theta,
        "n_v_phi_params": n_v_phi,
        "n_score_head_params": n_score,
        "n_xi_module_params": n_xi,
        "elapsed_sec": total_elapsed,
        "final_gumbel_tau": model.gumbel_tau,
        "top_k": cfg.top_k,
    }, ckpt_path)
    print(f"[multixi-parf] checkpoint -> {ckpt_path}")

    fig, ax = plt.subplots(figsize=(7, 4))
    if loss_history:
        steps_e = [e[0] for e in loss_history]
        va_e = [e[2] for e in loss_history]
        ax.plot(steps_e, [math.exp(v) for v in va_e],
                marker="s", label="val ppl", color="darkorange")
    ax.set_xlabel("step")
    ax.set_ylabel("perplexity")
    ax.set_title(
        f"PARF+K-EMA K={cfg.xi_channels} {cfg.v_phi_kind} "
        f"k={cfg.top_k} — {args.mode} — seed={args.seed}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    png_path = results_dir / f"{tag}_loss_curve.png"
    fig.savefig(png_path, dpi=130)
    plt.close(fig)

    summary_path = results_dir / f"{tag}_summary.md"
    with summary_path.open("w") as f:
        f.write(f"# Training summary — {tag}\n\n")
        f.write("- experiment: multi-channel ξ PARF scale-up\n")
        f.write("- model: MultiXiPARFLM (K-EMA ξ + sparse PARF)\n")
        f.write(f"- v_phi_kind: {cfg.v_phi_kind}\n")
        f.write(f"- top_k: {cfg.top_k}\n")
        f.write(f"- gumbel_tau: {cfg.gumbel_tau_init} -> "
                f"{cfg.gumbel_tau_min}\n")
        f.write(f"- mode: {args.mode}\n")
        f.write(f"- corpus: TinyStories "
                f"(cap {args.max_train_tokens:,} train tokens)\n")
        f.write(f"- fixed_gamma: {args.fixed_gamma}\n")
        f.write(f"- params: {n_params:,}  V_theta={n_v_theta:,}  "
                f"V_phi={n_v_phi:,}  score_head={n_score:,}  "
                f"xi_module={n_xi:,}\n")
        f.write(f"- d={cfg.d}  L={cfg.L}  v_hidden={cfg.v_hidden}  "
                f"v_depth={cfg.v_depth}  max_len={cfg.max_len}\n")
        f.write(f"- xi: K={cfg.xi_channels}  learnable={cfg.xi_learnable}  "
                f"α_init_mode={cfg.xi_alpha_init_mode}  "
                f"α_init={cfg.xi_alpha_inits}\n")
        f.write(f"- block_size: {train_cfg['block_size']}  "
                f"batch_size: {train_cfg['batch_size']}  "
                f"steps: {train_cfg['steps']}\n")
        f.write(f"- seed: {args.seed}\n")
        f.write(f"- elapsed: {total_elapsed:.0f} s "
                f"({total_elapsed/3600:.2f} h)\n")
        f.write(f"\nFinal val loss: {final_val:.6f} (ppl {final_ppl:.2f})\n")
        f.write(f"Final gamma: {final_gamma:.4f}\n")
        f.write(f"Final α_k: {final_alphas}\n")
    print(f"[multixi-parf] summary -> {summary_path}")


if __name__ == "__main__":
    main()
