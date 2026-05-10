"""
Training loop for the **paper_tmlr_1 scale-up pilot** (Helmholtz Q9d arm).

This trainer is a hard-fork of `helmholtz/train_helmholtz.py` adapted for
the scale-up configuration locked by the existing E9 SPLM scale-up
protocol (`docs/SPLM_scaleup_pre-registered_protocol.md`):
  - corpus      : TinyStories (~5 M GPT-2 BPE tokens)
  - max_len     : 1024
  - block_size  : 512
  - d / L / v_h : 256 / 8 / 1024
  - schedule    : AAAASSSS  (Variant-A-like Q9d cell, H1 winner at small scale)
  - mass        : logfreq, alpha-init 0.1, surprisal computed on TinyStories
  - damping     : free gamma, init 0.15  (matches H1 leak-free Q9d cells)
  - steps       : 8000   batch 16   lr 5e-4 cosine, 400-step warmup
  - eval        : every 400 steps, 40 batches × batch 16 × block 512

Modes
-----
  --mode smoke    : 300-step pipeline-correctness verification (no PPL claim).
  --mode scaleup  : full pilot run (8000 steps, 400 warmup, 400 eval).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

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

HELMHOLTZ_DIR = PARENT_DIR / "helmholtz"
sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(HELMHOLTZ_DIR))

from data_module import get_batch, load_tiny_stories  # noqa: E402
from model_helmholtz import HelmholtzConfig, HelmholtzLM  # noqa: E402

# `model_helmholtz` mutates sys.path[0] = PARENT_DIR, which would shadow the
# Helmholtz-local `causal_probe.py` with the repo-wide one.  Re-assert
# HELMHOLTZ_DIR at the front so the local import wins.
sys.path.insert(0, str(HELMHOLTZ_DIR))
from causal_probe import assert_causal  # noqa: E402

DEFAULT_LOGFREQ_PATH = SCRIPT_DIR / "results" / "logfreq_surprisal_tinystories.npy"


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_config(
    mode: str,
    schedule: str,
    logfreq_path: str | None,
    fixed_gamma: float | None = None,
) -> tuple[HelmholtzConfig, dict, str]:
    base_kw = dict(
        vocab_size=50257,
        mass_mode="logfreq",
        logfreq_init_alpha=0.1,
        logfreq_path=logfreq_path,
        n_head=4,
        mlp_mult=4,
        v_depth=3,
        dt=1.0,
        init_m=1.0,
        init_gamma=0.15,
        learn_mgamma=True,
        fixed_gamma=fixed_gamma,
        ln_after_s_step=True,
        causal_force=True,
        tie_embeddings=True,
        schedule=schedule,
    )
    L = len(schedule)
    if mode == "smoke":
        model_cfg = HelmholtzConfig(
            d=256, max_len=1024, v_hidden=1024,
            **base_kw,
        )
        train_cfg = dict(
            batch_size=8, block_size=256,
            steps=300, lr=5e-4, weight_decay=0.01,
            warmup_steps=20, grad_clip=1.0,
            eval_interval=100, eval_iters=10,
            log_interval=10,
        )
    elif mode == "scaleup":
        model_cfg = HelmholtzConfig(
            d=256, max_len=1024, v_hidden=1024,
            **base_kw,
        )
        train_cfg = dict(
            batch_size=16, block_size=512,
            steps=8000, lr=5e-4, weight_decay=0.01,
            warmup_steps=400, grad_clip=1.0,
            eval_interval=400, eval_iters=40,
            log_interval=50,
        )
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    fg_tag = "" if fixed_gamma is None else f"_g{fixed_gamma:.3f}"
    tag = f"helmholtz_{schedule}_L{L}{fg_tag}_scaleup_{mode}"
    return model_cfg, train_cfg, tag


def lr_schedule(step: int, lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


@torch.no_grad()
def evaluate(model: HelmholtzLM, ids: np.ndarray, iters: int,
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
    ap.add_argument(
        "--schedule", default="AAAASSSS",
        help="Helmholtz schedule string of A/S characters; "
             "default 'AAAASSSS' = H1 winner at small scale (Variant-A-like).",
    )
    ap.add_argument("--fixed-gamma", dest="fixed_gamma", type=float, default=None,
                    help="If set, fix the damping coefficient at this value. "
                         "Default None = freely learned (matches H1 cells).")
    ap.add_argument("--logfreq-path", dest="logfreq_path",
                    default=str(DEFAULT_LOGFREQ_PATH))
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train-tokens", dest="max_train_tokens",
                    type=int, default=5_000_000)
    ap.add_argument("--tag-suffix", dest="tag_suffix", type=str, default="",
                    help="Optional suffix appended to the output tag, e.g. 'seed0'.")
    ap.add_argument("--results-dir", dest="results_dir", type=str, default=None)
    ap.add_argument("--skip-causal-check", action="store_true",
                    help="Skip the startup causal-violation probe.")
    args = ap.parse_args()

    device = args.device or _pick_device()
    results_dir = (
        Path(args.results_dir).expanduser().resolve()
        if args.results_dir is not None else RESULTS_DIR
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    # Disable TF32 for SPLM-family models on CUDA: the forward pass uses
    # torch.autograd.grad to compute -grad V_theta(h), and that second-order
    # path is more sensitive to TF32's 10-bit-mantissa reduction than a
    # single attention forward.  Disabling TF32 forces real fp32 matmuls
    # (23-bit mantissa) at the cost of ~2x slower matmul.  Worth it.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print("[scaleup-helmholtz] TF32 disabled for SPLM autograd.grad "
              "numerical stability (CUDA matmuls in true fp32)")

    print(f"[scaleup-helmholtz] device={device}  mode={args.mode}  "
          f"schedule={args.schedule!r}  fixed_gamma={args.fixed_gamma!r}  "
          f"seed={args.seed}  results_dir={results_dir}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train_ids, val_ids = load_tiny_stories(
        max_train_tokens=args.max_train_tokens,
    )
    print(f"[scaleup-helmholtz] tokens: train={len(train_ids):,}  val={len(val_ids):,}")

    logfreq_path = args.logfreq_path
    if not Path(logfreq_path).exists():
        raise FileNotFoundError(
            f"logfreq surprisal file not found at {logfreq_path}.  "
            "Run scaleup/compute_unigram_frequencies_tinystories.py first."
        )

    model_cfg, train_cfg, base_tag = build_config(
        args.mode, args.schedule, logfreq_path, fixed_gamma=args.fixed_gamma,
    )
    model = HelmholtzLM(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[scaleup-helmholtz] params: {n_params:,}   d={model_cfg.d}  "
          f"L={len(model_cfg.schedule)}  schedule={model_cfg.schedule}  "
          f"v_hidden={model_cfg.v_hidden}  max_len={model_cfg.max_len}")

    if not args.skip_causal_check:
        print("[scaleup-helmholtz] running causal-violation probe...")
        try:
            assert_causal(
                model, vocab_size=model_cfg.vocab_size,
                T=32, t_pert=20, seed=args.seed,
            )
            print("[scaleup-helmholtz] causal probe PASSED "
                  "(perturbation + gradient-Jacobian, both modes < 1e-6)")
        except RuntimeError as exc:
            print("[scaleup-helmholtz] causal probe FAILED — aborting "
                  "before any compute is wasted.")
            print(f"[scaleup-helmholtz] {exc}")
            raise SystemExit(2)

    optim = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"], betas=(0.9, 0.95),
    )

    tag = base_tag
    if args.tag_suffix:
        tag = f"{tag}_{args.tag_suffix}"
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

        xb, yb = get_batch(train_ids, train_cfg["batch_size"],
                           train_cfg["block_size"], rng)
        x = torch.from_numpy(xb).to(device)
        y = torch.from_numpy(yb).to(device)

        _, loss = model(x, y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(),
                                             train_cfg["grad_clip"])
        optim.step()

        running += loss.item()
        n_run += 1

        if (step + 1) % train_cfg["log_interval"] == 0:
            avg = running / n_run
            running, n_run = 0.0, 0
            elapsed = time.time() - t0
            gamma_val = (model.gamma.item() if hasattr(model, "gamma")
                         else float("nan"))
            print(f"[scaleup-helmholtz] step {step+1:5d}/{train_cfg['steps']}   "
                  f"train {avg:.4f}   lr {lr_now:.2e}   "
                  f"grad {grad_norm:.2f}   gamma={gamma_val:.3f}   "
                  f"elapsed {elapsed:.0f}s")
            log_f.write(json.dumps({
                "step": step + 1, "train_loss": avg,
                "lr": lr_now, "grad_norm": float(grad_norm),
                "gamma": gamma_val,
                "elapsed_sec": elapsed,
            }) + "\n")
            log_f.flush()

        if (step + 1) % train_cfg["eval_interval"] == 0:
            val_loss = evaluate(model, val_ids,
                                train_cfg["eval_iters"],
                                train_cfg["batch_size"],
                                train_cfg["block_size"], rng, device)
            val_ppl = math.exp(val_loss)
            print(f"[scaleup-helmholtz] step {step+1:5d}   val_loss={val_loss:.4f}  "
                  f"val_ppl={val_ppl:.2f}")
            loss_history.append((step + 1,
                                  avg if n_run == 0 else running / max(n_run, 1),
                                  val_loss))

    log_f.close()
    final_val = evaluate(model, val_ids,
                         train_cfg["eval_iters"],
                         train_cfg["batch_size"],
                         train_cfg["block_size"], rng, device)
    final_ppl = math.exp(final_val)
    final_gamma = model.gamma.item() if hasattr(model, "gamma") else float("nan")
    total_elapsed = time.time() - t0
    print(f"\n[scaleup-helmholtz] DONE  val_loss={final_val:.4f}  "
          f"val_ppl={final_ppl:.2f}  gamma={final_gamma:.4f}  "
          f"elapsed={total_elapsed:.0f}s")

    ckpt_path = results_dir / f"{tag}_ckpt_latest.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_cfg": asdict(model_cfg),
        "train_cfg": train_cfg,
        "final_val_loss": final_val,
        "final_val_ppl": final_ppl,
        "final_gamma": final_gamma,
        "fixed_gamma": args.fixed_gamma,
        "max_train_tokens": args.max_train_tokens,
        "logfreq_path": str(logfreq_path),
        "variant": "helmholtz_q9d",
        "experiment": "tmlr1_scaleup_pilot",
        "tag": tag,
        "seed": args.seed,
        "schedule": args.schedule,
        "elapsed_sec": total_elapsed,
    }, ckpt_path)
    print(f"[scaleup-helmholtz] checkpoint saved to {ckpt_path}")

    fig, ax = plt.subplots(figsize=(7, 4))
    if loss_history:
        steps_v, _train_vs, val_vs = zip(*loss_history)
        ax.plot(steps_v, [math.exp(v) for v in val_vs],
                label="val ppl", color="darkred")
    ax.set_xlabel("step")
    ax.set_ylabel("perplexity")
    gamma_str = (f"fixed γ={args.fixed_gamma}" if args.fixed_gamma is not None
                 else "free γ")
    ax.set_title(f"Helmholtz {args.schedule} scale-up — {args.mode} "
                 f"({gamma_str}) — seed={args.seed}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / f"{tag}_loss_curve.png", dpi=120)
    plt.close(fig)

    summary_path = results_dir / f"{tag}_summary.md"
    with summary_path.open("w") as f:
        f.write(f"# Training summary — {tag}\n\n")
        f.write(f"- experiment: paper_tmlr_1 scale-up pilot (Helmholtz Q9d arm)\n")
        f.write(f"- model: HelmholtzLM\n")
        f.write(f"- schedule: {args.schedule}  (L={len(args.schedule)})\n")
        f.write(f"- mode: {args.mode}\n")
        f.write(f"- corpus: TinyStories (cap {args.max_train_tokens:,} train tokens)\n")
        f.write(f"- fixed_gamma: {args.fixed_gamma}\n")
        f.write(f"- params: {n_params:,}\n")
        f.write(f"- d={model_cfg.d}  L={len(model_cfg.schedule)}  "
                f"v_hidden={model_cfg.v_hidden}  v_depth={model_cfg.v_depth}  "
                f"n_head={model_cfg.n_head}  mlp_mult={model_cfg.mlp_mult}  "
                f"max_len={model_cfg.max_len}  ln_after_s_step=True\n")
        f.write(f"- block_size: {train_cfg['block_size']}  "
                f"batch_size: {train_cfg['batch_size']}  "
                f"steps: {train_cfg['steps']}\n")
        f.write(f"- seed: {args.seed}\n")
        f.write(f"- elapsed: {total_elapsed:.0f} s ({total_elapsed/3600:.2f} h)\n")
        f.write(f"\nFinal val loss: {final_val:.6f} (ppl {final_ppl:.2f})\n")
        f.write(f"Final gamma: {final_gamma:.4f}\n")
    print(f"[scaleup-helmholtz] summary written to {summary_path}")


if __name__ == "__main__":
    main()
