"""
Training loop for the matched-parameter GPT-2-style baseline.

This mirrors `train_splm.py` on the same dataset, same tokenizer,
same token budget and same optimiser, so that SPLM-vs-MatchedGPT
comparisons are controlled for scale and data.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from data_module import get_batch, load_tiny_shakespeare
from matched_baseline_model import MatchedConfig, MatchedGPT


SCRIPT_DIR  = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_config(mode: str) -> tuple[MatchedConfig, dict]:
    """Matched training configs that mirror train_splm.py line-for-line."""
    if mode == "smoke":
        model_cfg = MatchedConfig(d=64, max_len=128, n_layer=4, n_head=4)
        train_cfg = dict(
            batch_size=8, block_size=64, steps=300,
            lr=1e-3, weight_decay=0.0,
            warmup_steps=20, grad_clip=1.0,
            eval_interval=50, eval_iters=20, log_interval=10,
        )
    elif mode == "shakespeare":
        model_cfg = MatchedConfig(d=128, max_len=256, n_layer=8, n_head=4)
        train_cfg = dict(
            batch_size=16, block_size=128, steps=4000,
            lr=5e-4, weight_decay=0.01,
            warmup_steps=200, grad_clip=1.0,
            eval_interval=200, eval_iters=40, log_interval=50,
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
def evaluate(model, ids, iters, batch_size, block_size, rng, device):
    model.eval()
    losses = []
    for _ in range(iters):
        xb, yb = get_batch(ids, batch_size, block_size, rng)
        x = torch.from_numpy(xb).to(device)
        y = torch.from_numpy(yb).to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "shakespeare"], default="shakespeare")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = args.device or _pick_device()
    print(f"[train-matched] device={device}  mode={args.mode}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train_ids, val_ids = load_tiny_shakespeare()
    print(f"[train-matched] tokens: train={len(train_ids):,}  val={len(val_ids):,}")

    model_cfg, train_cfg = build_config(args.mode)
    model = MatchedGPT(model_cfg).to(device)
    n_params = model.num_params()
    print(f"[train-matched] params: {n_params:,}   d={model_cfg.d}  "
          f"n_layer={model_cfg.n_layer}  n_head={model_cfg.n_head}")

    optim = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"], betas=(0.9, 0.95),
    )

    log_path = RESULTS_DIR / f"matched_{args.mode}_training_log.jsonl"
    log_f = log_path.open("w")
    loss_history = []

    t0 = time.time()
    model.train()
    running, n_run = 0.0, 0

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

        running += loss.item(); n_run += 1

        if (step + 1) % train_cfg["log_interval"] == 0:
            avg = running / n_run
            running, n_run = 0.0, 0
            elapsed = time.time() - t0
            print(f"[train-matched] step {step+1:5d}/{train_cfg['steps']}   "
                  f"train {avg:.4f}   lr {lr_now:.2e}   grad {grad_norm:.2f}   "
                  f"elapsed {elapsed:.0f}s")
            log_f.write(json.dumps({
                "step": step + 1, "train_loss": avg,
                "lr": lr_now, "grad_norm": float(grad_norm),
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
            print(f"[train-matched] >>> eval @ {step+1}: "
                  f"train {train_loss_eval:.4f}   val {val_loss:.4f}   ppl {ppl:.2f}")
            log_f.write(json.dumps({
                "step": step + 1, "train_loss_eval": train_loss_eval,
                "val_loss": val_loss, "val_ppl": ppl,
            }) + "\n")
            log_f.flush()
            loss_history.append((step + 1, train_loss_eval, val_loss))

    log_f.close()
    elapsed = time.time() - t0
    print(f"[train-matched] done in {elapsed:.0f}s")

    ckpt_path = RESULTS_DIR / f"matched_{args.mode}_ckpt_latest.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_cfg": asdict(model_cfg),
        "train_cfg": train_cfg,
        "loss_history": loss_history,
    }, ckpt_path)
    print(f"[train-matched] saved checkpoint -> {ckpt_path}")

    if loss_history:
        steps_e = [e[0] for e in loss_history]
        tr_e    = [e[1] for e in loss_history]
        va_e    = [e[2] for e in loss_history]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(steps_e, tr_e, marker="o", label="train (eval)")
        ax.plot(steps_e, va_e, marker="s", label="val")
        ax.set_xlabel("step"); ax.set_ylabel("cross-entropy")
        ax.set_title(f"Matched GPT-2-style baseline -- {args.mode}")
        ax.grid(True, alpha=0.3); ax.legend()
        fig.tight_layout()
        png_path = RESULTS_DIR / f"matched_{args.mode}_loss_curve.png"
        fig.savefig(png_path, dpi=130)
        plt.close(fig)
        print(f"[train-matched] saved loss curve -> {png_path}")

    summary_path = RESULTS_DIR / f"matched_{args.mode}_summary.md"
    with summary_path.open("w") as f:
        f.write(f"# Matched GPT-2-style baseline -- {args.mode}\n\n")
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
        f.write(f"- Loss curve: `matched_{args.mode}_loss_curve.png`\n")
        f.write(f"- Checkpoint: `matched_{args.mode}_ckpt_latest.pt`\n")
    print(f"[train-matched] saved summary -> {summary_path}")


if __name__ == "__main__":
    main()
