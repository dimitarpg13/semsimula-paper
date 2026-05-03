"""
Training loop for the **E9 SPLM scale-up de-risking experiment**
(matched-attention baseline arm).

Pre-registered protocol:
  docs/SPLM_scaleup_pre-registered_protocol.md
Pre-registration commit: 17a3795

This trainer is a hard-fork of `inference_efficiency/train_matched_baseline.py`
adapted for the scale-up configuration locked in the protocol:
  - corpus      : TinyStories (~5 M GPT-2 BPE tokens)
  - max_len     : 1024
  - block_size  : 512
  - d / L / heads : 256 / 8 / 4   (mlp_mult=4, tied embeddings; ~19.45 M params)
  - steps       : 8000  batch 16  lr 5e-4 cosine, 400-step warmup
  - eval        : every 400 steps, 40 batches × batch 16 × block 512

Modes
-----
  --mode smoke    : 300-step pipeline-correctness verification (no PPL claim).
  --mode scaleup  : full pre-registered run.
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

sys.path.insert(0, str(PARENT_DIR))

from data_module import get_batch, load_tiny_stories  # noqa: E402
from matched_baseline_model import MatchedConfig, MatchedGPT  # noqa: E402


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_config(mode: str) -> tuple[MatchedConfig, dict]:
    if mode == "smoke":
        # Same model size as scaleup, smaller block + step budget so we verify
        # the pipeline at scale-up parameter count without burning the night.
        model_cfg = MatchedConfig(
            vocab_size=50257, d=256, max_len=1024,
            n_layer=8, n_head=4, mlp_mult=4, tie_embeddings=True,
        )
        train_cfg = dict(
            batch_size=8, block_size=256,
            steps=300, lr=5e-4, weight_decay=0.01,
            warmup_steps=20, grad_clip=1.0,
            eval_interval=100, eval_iters=10,
            log_interval=10,
        )
    elif mode == "scaleup":
        model_cfg = MatchedConfig(
            vocab_size=50257, d=256, max_len=1024,
            n_layer=8, n_head=4, mlp_mult=4, tie_embeddings=True,
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
    return model_cfg, train_cfg


def lr_schedule(step: int, lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


@torch.no_grad()
def evaluate(model: MatchedGPT, ids: np.ndarray,
             iters: int, batch_size: int, block_size: int,
             rng: np.random.Generator, device: str) -> float:
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
    ap.add_argument("--mode", choices=["smoke", "scaleup"], default="smoke")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train-tokens", dest="max_train_tokens",
                    type=int, default=5_000_000)
    ap.add_argument("--tag-suffix", dest="tag_suffix", type=str, default="",
                    help="Optional suffix appended to the output tag, e.g. 'seed0'.")
    ap.add_argument("--results-dir", dest="results_dir", type=str, default=None)
    args = ap.parse_args()

    device = args.device or _pick_device()
    results_dir = (
        Path(args.results_dir).expanduser().resolve()
        if args.results_dir is not None else RESULTS_DIR
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"[scaleup-attn] device={device}  mode={args.mode}  "
          f"seed={args.seed}  results_dir={results_dir}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train_ids, val_ids = load_tiny_stories(
        max_train_tokens=args.max_train_tokens,
    )
    print(f"[scaleup-attn] tokens: train={len(train_ids):,}  val={len(val_ids):,}")

    model_cfg, train_cfg = build_config(args.mode)
    model = MatchedGPT(model_cfg).to(device)
    n_params = model.num_params()
    print(f"[scaleup-attn] params: {n_params:,}   d={model_cfg.d}  "
          f"L={model_cfg.n_layer}  n_head={model_cfg.n_head}  "
          f"mlp_mult={model_cfg.mlp_mult}  max_len={model_cfg.max_len}  "
          f"tie={model_cfg.tie_embeddings}")

    optim = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"], betas=(0.9, 0.95),
    )

    tag = f"matched_baseline_scaleup_{args.mode}"
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
            print(f"[scaleup-attn] step {step+1:5d}/{train_cfg['steps']}   "
                  f"train {avg:.4f}   lr {lr_now:.2e}   "
                  f"grad {grad_norm:.2f}   elapsed {elapsed:.0f}s")
            log_f.write(json.dumps({
                "step": step + 1, "train_loss": avg,
                "lr": lr_now, "grad_norm": float(grad_norm),
                "elapsed_sec": elapsed,
            }) + "\n")
            log_f.flush()

        if (step + 1) % train_cfg["eval_interval"] == 0:
            val_loss = evaluate(model, val_ids,
                                train_cfg["eval_iters"],
                                train_cfg["batch_size"],
                                train_cfg["block_size"], rng, device)
            val_ppl = math.exp(val_loss)
            print(f"[scaleup-attn] step {step+1:5d}   val_loss={val_loss:.4f}  "
                  f"val_ppl={val_ppl:.2f}")
            loss_history.append((step + 1, avg if n_run == 0 else running / max(n_run, 1),
                                  val_loss))

    log_f.close()
    final_val = evaluate(model, val_ids,
                         train_cfg["eval_iters"],
                         train_cfg["batch_size"],
                         train_cfg["block_size"], rng, device)
    final_ppl = math.exp(final_val)
    total_elapsed = time.time() - t0
    print(f"\n[scaleup-attn] DONE  val_loss={final_val:.4f}  "
          f"val_ppl={final_ppl:.2f}  (matched attention baseline)  "
          f"elapsed={total_elapsed:.0f}s")

    ckpt_path = results_dir / f"{tag}_ckpt_latest.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_cfg": asdict(model_cfg),
        "train_cfg": train_cfg,
        "final_val_loss": final_val,
        "final_val_ppl": final_ppl,
        "max_train_tokens": args.max_train_tokens,
        "variant": "matched_attention_baseline",
        "experiment": "E9_scaleup",
        "tag": tag,
        "seed": args.seed,
        "elapsed_sec": total_elapsed,
    }, ckpt_path)
    print(f"[scaleup-attn] checkpoint saved to {ckpt_path}")

    steps_v, train_vs, val_vs = [], [], []
    if loss_history:
        steps_v, train_vs, val_vs = zip(*loss_history)
    fig, ax = plt.subplots(figsize=(7, 4))
    if steps_v:
        ax.plot(steps_v, [math.exp(v) for v in val_vs],
                label="val ppl", color="forestgreen")
    ax.set_xlabel("step")
    ax.set_ylabel("perplexity")
    ax.set_title(f"Matched attention baseline scale-up — {args.mode} — seed={args.seed}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / f"{tag}_loss_curve.png", dpi=120)
    plt.close(fig)

    summary_path = results_dir / f"{tag}_summary.md"
    with summary_path.open("w") as f:
        f.write(f"# Training summary — {tag}\n\n")
        f.write(f"- experiment: E9 scale-up (matched-attention arm)\n")
        f.write(f"- model: MatchedGPT\n")
        f.write(f"- mode: {args.mode}\n")
        f.write(f"- corpus: TinyStories (cap {args.max_train_tokens:,} train tokens)\n")
        f.write(f"- variant: matched_attention_baseline\n")
        f.write(f"- params: {n_params:,}\n")
        f.write(f"- d={model_cfg.d}  L={model_cfg.n_layer}  "
                f"n_head={model_cfg.n_head}  mlp_mult={model_cfg.mlp_mult}  "
                f"max_len={model_cfg.max_len}  "
                f"tie_embeddings={model_cfg.tie_embeddings}\n")
        f.write(f"- block_size: {train_cfg['block_size']}  "
                f"batch_size: {train_cfg['batch_size']}  "
                f"steps: {train_cfg['steps']}\n")
        f.write(f"- seed: {args.seed}\n")
        f.write(f"- elapsed: {total_elapsed:.0f} s ({total_elapsed/3600:.2f} h)\n")
        f.write(f"\nFinal val loss: {final_val:.6f} (ppl {final_ppl:.2f})\n")
    print(f"[scaleup-attn] summary written to {summary_path}")


if __name__ == "__main__":
    main()
