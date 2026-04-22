"""
Training loop for the scalar-potential conservative-by-construction LM.

Modes
-----
  --mode smoke        : 300-step smoke test on Tiny Shakespeare with a
                        miniature model; verifies the pipeline end-to-end.
  --mode shakespeare  : real training on Tiny Shakespeare (~10 min on MPS,
                        default config, one convergence run).
  --mode tinystories  : real training on Tiny Stories (one parquet
                        shard, ~300 MB after tokenisation; longer run).

Outputs
-------
  results/
    splm_<mode>_ckpt_latest.pt        model + optimizer checkpoint
    splm_<mode>_training_log.jsonl    per-step training metrics (jsonl)
    splm_<mode>_summary.md            human-readable convergence report
    splm_<mode>_loss_curve.png        train / val loss curve
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from data_module import (
    DATA_DIR, get_batch, load_tiny_shakespeare, load_tiny_stories
)
from model import ScalarPotentialLM, SPLMConfig


# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
def build_config(mode: str) -> tuple[SPLMConfig, dict]:
    """Return (model_cfg, train_cfg) for a given training mode."""
    if mode == "smoke":
        model_cfg = SPLMConfig(
            vocab_size=50257, d=64, max_len=128,
            v_hidden=128, v_depth=2, L=4,
            init_m=1.0, init_gamma=1.0,
        )
        train_cfg = dict(
            batch_size=8, block_size=64,
            steps=300, lr=1e-3, weight_decay=0.0,
            warmup_steps=20, grad_clip=1.0,
            eval_interval=50, eval_iters=20,
            log_interval=10,
        )
    elif mode == "shakespeare":
        model_cfg = SPLMConfig(
            vocab_size=50257, d=128, max_len=256,
            v_hidden=512, v_depth=3, L=8,
            init_m=1.0, init_gamma=1.0,
        )
        train_cfg = dict(
            batch_size=16, block_size=128,
            steps=4000, lr=5e-4, weight_decay=0.01,
            warmup_steps=200, grad_clip=1.0,
            eval_interval=200, eval_iters=40,
            log_interval=50,
        )
    elif mode == "tinystories":
        model_cfg = SPLMConfig(
            vocab_size=50257, d=192, max_len=256,
            v_hidden=768, v_depth=3, L=8,
            init_m=1.0, init_gamma=1.0,
        )
        train_cfg = dict(
            batch_size=32, block_size=128,
            steps=10000, lr=3e-4, weight_decay=0.01,
            warmup_steps=400, grad_clip=1.0,
            eval_interval=500, eval_iters=50,
            log_interval=50,
        )
    else:
        raise ValueError(f"unknown mode: {mode}")
    return model_cfg, train_cfg


def lr_schedule(step: int, lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


@torch.no_grad()
def evaluate(model: ScalarPotentialLM, ids: np.ndarray,
             iters: int, batch_size: int, block_size: int,
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
    ap.add_argument("--mode", choices=["smoke", "shakespeare", "tinystories"],
                    default="smoke")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_train_tokens", type=int, default=None,
                    help="for tinystories mode: cap on tokenised train tokens")
    args = ap.parse_args()

    device = args.device or _pick_device()
    print(f"[train] device={device}  mode={args.mode}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # --- Data ---
    if args.mode == "tinystories":
        train_ids, val_ids = load_tiny_stories(
            n_train_files=1,
            max_train_tokens=args.max_train_tokens or 5_000_000,
        )
    else:
        train_ids, val_ids = load_tiny_shakespeare()
    print(f"[train] tokens: train={len(train_ids):,}  val={len(val_ids):,}")

    # --- Model ---
    model_cfg, train_cfg = build_config(args.mode)
    model = ScalarPotentialLM(model_cfg).to(device)
    n_params = model.num_params()
    print(f"[train] params: {n_params:,}   d={model_cfg.d}  L={model_cfg.L}  "
          f"v_hidden={model_cfg.v_hidden}")

    optim = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"], betas=(0.9, 0.95),
    )

    # --- Logging ---
    log_path = RESULTS_DIR / f"splm_{args.mode}_training_log.jsonl"
    log_f = log_path.open("w")
    loss_history: list[tuple[int, float, float]] = []   # (step, train_loss, val_loss)

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
            print(f"[train] step {step+1:5d}/{train_cfg['steps']}   "
                  f"train {avg:.4f}   lr {lr_now:.2e}   "
                  f"grad {grad_norm:.2f}   m={model.m.item():.3f}   "
                  f"gamma={model.gamma.item():.3f}   "
                  f"elapsed {elapsed:.0f}s")
            log_f.write(json.dumps({
                "step": step + 1, "train_loss": avg,
                "lr": lr_now, "grad_norm": float(grad_norm),
                "m": model.m.item(), "gamma": model.gamma.item(),
                "elapsed_s": elapsed,
            }) + "\n")
            log_f.flush()

        if (step + 1) % train_cfg["eval_interval"] == 0 or step + 1 == train_cfg["steps"]:
            val_loss = evaluate(
                model, val_ids, train_cfg["eval_iters"],
                train_cfg["batch_size"], train_cfg["block_size"], rng, device,
            )
            train_loss_eval = evaluate(
                model, train_ids, train_cfg["eval_iters"],
                train_cfg["batch_size"], train_cfg["block_size"], rng, device,
            )
            ppl = math.exp(val_loss)
            print(f"[train] >>> eval @ {step+1}: train {train_loss_eval:.4f}   "
                  f"val {val_loss:.4f}   ppl {ppl:.2f}")
            log_f.write(json.dumps({
                "step": step + 1, "train_loss_eval": train_loss_eval,
                "val_loss": val_loss, "val_ppl": ppl,
            }) + "\n")
            log_f.flush()
            loss_history.append((step + 1, train_loss_eval, val_loss))

    log_f.close()
    elapsed = time.time() - t0
    print(f"[train] done in {elapsed:.0f}s")

    # --- Save checkpoint ---
    ckpt_path = RESULTS_DIR / f"splm_{args.mode}_ckpt_latest.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_cfg": asdict(model_cfg),
        "train_cfg": train_cfg,
        "loss_history": loss_history,
        "final_m": float(model.m.item()),
        "final_gamma": float(model.gamma.item()),
    }, ckpt_path)
    print(f"[train] saved checkpoint -> {ckpt_path}")

    # --- Loss curve ---
    if loss_history:
        steps_e = [e[0] for e in loss_history]
        tr_e    = [e[1] for e in loss_history]
        va_e    = [e[2] for e in loss_history]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(steps_e, tr_e, marker="o", label="train (eval)")
        ax.plot(steps_e, va_e, marker="s", label="val")
        ax.set_xlabel("step")
        ax.set_ylabel("cross-entropy")
        ax.set_title(f"Scalar-potential LM -- {args.mode}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        png_path = RESULTS_DIR / f"splm_{args.mode}_loss_curve.png"
        fig.savefig(png_path, dpi=130)
        plt.close(fig)
        print(f"[train] saved loss curve -> {png_path}")

    # --- Summary ---
    summary_path = RESULTS_DIR / f"splm_{args.mode}_summary.md"
    with summary_path.open("w") as f:
        f.write(f"# Scalar-potential LM -- {args.mode} training summary\n\n")
        f.write(f"- Device: `{device}`\n")
        f.write(f"- Parameters: **{n_params:,}**\n")
        f.write(f"- Model config: `{asdict(model_cfg)}`\n")
        f.write(f"- Train config: `{train_cfg}`\n")
        f.write(f"- Tokens: train={len(train_ids):,}, val={len(val_ids):,}\n")
        f.write(f"- Wall-clock time: {elapsed:.0f}s\n")
        if loss_history:
            final_train, final_val = loss_history[-1][1], loss_history[-1][2]
            f.write(f"- Final train loss: {final_train:.4f}\n")
            f.write(f"- Final val loss: {final_val:.4f} "
                    f"(ppl {math.exp(final_val):.2f})\n")
        f.write(f"- Final m = {model.m.item():.4f}, "
                f"gamma = {model.gamma.item():.4f}\n")
        f.write(f"- Loss curve: `splm_{args.mode}_loss_curve.png`\n")
        f.write(f"- Checkpoint: `splm_{args.mode}_ckpt_latest.pt`\n")
        f.write(f"- Log: `splm_{args.mode}_training_log.jsonl`\n")
    print(f"[train] saved summary -> {summary_path}")


if __name__ == "__main__":
    main()
