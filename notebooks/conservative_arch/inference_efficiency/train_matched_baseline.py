"""
Training loop for the matched-parameter GPT-2-style transformer baseline.

This is **Phase 1** of the E8 inference-efficiency benchmark. The protocol
is pre-registered in
`companion_notes/SPLM_inference_efficiency_pre-registered_protocol.md`, §3.

The training protocol is identical to the SPLM-1 ablation arm B (which
trained SPLM em_ln at γ*=0.30 on Tiny Shakespeare): 4 000 steps, batch 16,
block 128, AdamW lr=5e-4 with cosine decay and 200-step warmup,
weight_decay=0.01, betas=(0.9, 0.95), grad_clip=1.0, eval_iters=40 every
200 steps. Three seeds {0, 1, 2}.

The model is `MatchedGPT` from
`notebooks/conservative_arch/matched_baseline_model.py` — a vanilla
pre-LN GPT-2 decoder with d=128, n_layer=8, n_head=4, mlp_mult=4, tied
embeddings (~8 M params, vs SPLM em_ln's ~7 M).
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

from data_module import get_batch, load_tiny_shakespeare  # noqa: E402
from matched_baseline_model import MatchedConfig, MatchedGPT  # noqa: E402


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_config(mode: str) -> tuple[MatchedConfig, dict]:
    if mode == "smoke":
        model_cfg = MatchedConfig(
            vocab_size=50257, d=64, max_len=128,
            n_layer=4, n_head=4, mlp_mult=4, tie_embeddings=True,
        )
        train_cfg = dict(
            batch_size=8, block_size=64,
            steps=300, lr=1e-3, weight_decay=0.0,
            warmup_steps=20, grad_clip=1.0,
            eval_interval=50, eval_iters=20,
            log_interval=10,
        )
    elif mode == "shakespeare":
        model_cfg = MatchedConfig(
            vocab_size=50257, d=128, max_len=256,
            n_layer=8, n_head=4, mlp_mult=4, tie_embeddings=True,
        )
        train_cfg = dict(
            batch_size=16, block_size=128,
            steps=4000, lr=5e-4, weight_decay=0.01,
            warmup_steps=200, grad_clip=1.0,
            eval_interval=200, eval_iters=40,
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
    ap.add_argument("--mode", choices=["smoke", "shakespeare"], default="smoke")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
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
    print(f"[train-attn] device={device}  mode={args.mode}  "
          f"seed={args.seed}  results_dir={results_dir}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train_ids, val_ids = load_tiny_shakespeare()
    print(f"[train-attn] tokens: train={len(train_ids):,}  val={len(val_ids):,}")

    model_cfg, train_cfg = build_config(args.mode)
    model = MatchedGPT(model_cfg).to(device)
    n_params = model.num_params()
    print(f"[train-attn] params: {n_params:,}   d={model_cfg.d}  "
          f"L={model_cfg.n_layer}  n_head={model_cfg.n_head}  "
          f"mlp_mult={model_cfg.mlp_mult}  tie={model_cfg.tie_embeddings}")

    optim = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"], betas=(0.9, 0.95),
    )

    tag = f"matched_baseline_{args.mode}"
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
            print(f"[train-attn] step {step+1:5d}/{train_cfg['steps']}   "
                  f"train {avg:.4f}   lr {lr_now:.2e}   "
                  f"grad {grad_norm:.2f}   elapsed {elapsed:.0f}s")
            log_f.write(json.dumps({
                "step": step + 1, "train_loss": avg,
                "lr": lr_now, "grad_norm": float(grad_norm),
            }) + "\n")
            log_f.flush()

        if (step + 1) % train_cfg["eval_interval"] == 0:
            val_loss = evaluate(model, val_ids,
                                train_cfg["eval_iters"],
                                train_cfg["batch_size"],
                                train_cfg["block_size"], rng, device)
            val_ppl = math.exp(val_loss)
            print(f"[train-attn] step {step+1:5d}   val_loss={val_loss:.4f}  "
                  f"val_ppl={val_ppl:.2f}")
            loss_history.append((step + 1, avg if n_run == 0 else running / max(n_run, 1),
                                  val_loss))

    log_f.close()
    final_val = evaluate(model, val_ids,
                         train_cfg["eval_iters"],
                         train_cfg["batch_size"],
                         train_cfg["block_size"], rng, device)
    final_ppl = math.exp(final_val)
    print(f"\n[train-attn] DONE  val_loss={final_val:.4f}  "
          f"val_ppl={final_ppl:.2f}  (matched attention baseline)")

    ckpt_path = results_dir / f"{tag}_ckpt_latest.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_cfg": asdict(model_cfg),
        "final_val_loss": final_val,
        "final_val_ppl": final_ppl,
        "variant": "matched_attention_baseline",
        "tag": tag,
        "seed": args.seed,
    }, ckpt_path)
    print(f"[train-attn] checkpoint saved to {ckpt_path}")

    steps_v, train_vs, val_vs = [], [], []
    if loss_history:
        steps_v, train_vs, val_vs = zip(*loss_history)
    fig, ax = plt.subplots(figsize=(7, 4))
    if steps_v:
        ax.plot(steps_v, [math.exp(v) for v in val_vs],
                label="val ppl", color="forestgreen")
    ax.set_xlabel("step")
    ax.set_ylabel("perplexity")
    ax.set_title(f"Matched attention baseline — {args.mode} — seed={args.seed}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / f"{tag}_loss_curve.png", dpi=120)
    plt.close(fig)

    summary_path = results_dir / f"{tag}_summary.md"
    with summary_path.open("w") as f:
        f.write(f"# Training summary — {tag}\n\n")
        f.write(f"- Model: MatchedGPT (E8 Phase 1 attention baseline)\n")
        f.write(f"- mode: {args.mode}\n")
        f.write(f"- variant: matched_attention_baseline\n")
        f.write(f"- params: {n_params:,}\n")
        f.write(f"- d={model_cfg.d}  L={model_cfg.n_layer}  "
                f"n_head={model_cfg.n_head}  mlp_mult={model_cfg.mlp_mult}  "
                f"tie_embeddings={model_cfg.tie_embeddings}\n")
        f.write(f"- seed: {args.seed}\n")
        f.write(f"\nFinal val loss: {final_val:.6f} (ppl {final_ppl:.2f})\n")
    print(f"[train-attn] summary written to {summary_path}")


if __name__ == "__main__":
    main()
