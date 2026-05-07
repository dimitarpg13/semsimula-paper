"""
Hybrid SPLM + Attention trainer (HSPLM, Variant A).

Mirrors notebooks/conservative_arch/energetic_minima/train.py:
  - Tiny Shakespeare data, GPT-2 BPE tokens
  - logfreq mass mode (matches leak-free SPLM em_ln cells)
  - 4000 steps, batch_size=16, block_size=128, AdamW(0.9, 0.95)
  - cosine LR schedule with 200 warmup steps
  - val perplexity logged every eval_interval steps
  - causal_force=True (leak-fix invariant always on)

Adds two CLI flags:
  --n-attn k   number of attention blocks in the front stack
  --n-splm m   number of SPLM integration steps in the back stack

Outputs (under hybrid/results/):
  - hybrid_k{n_attn}_m{n_splm}_<mode>_seed{seed}_training_log.jsonl
  - hybrid_k{n_attn}_m{n_splm}_<mode>_seed{seed}_ckpt_latest.pt
  - hybrid_k{n_attn}_m{n_splm}_<mode>_seed{seed}_loss_curve.png
  - hybrid_k{n_attn}_m{n_splm}_<mode>_seed{seed}_summary.md

This trainer is the H0 + H1 driver of:
  the v4 title-justification rule §6 + §7.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Tuple

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

sys.path.insert(0, str(SCRIPT_DIR))
from model_hybrid import HSPLMConfig, HybridSPLM  # noqa: E402


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_config(
    mode: str,
    logfreq_path: str,
    n_attn: int,
    n_splm: int,
    fixed_gamma_arg,
) -> Tuple[HSPLMConfig, dict, str]:
    """Return (model_cfg, train_cfg, tag)."""

    base_kw = dict(
        vocab_size=50257,
        mass_mode="logfreq",
        logfreq_init_alpha=0.1,
        logfreq_path=logfreq_path,
        n_attn=n_attn,
        n_splm=n_splm,
        causal_force=True,
        ln_after_step=True,
        fixed_gamma=fixed_gamma_arg,
    )

    if mode == "smoke":
        base_kw.update(d=64, max_len=128, v_hidden=128, v_depth=2)
        train_cfg = dict(
            batch_size=8, block_size=64,
            steps=300, lr=1e-3, weight_decay=0.0,
            warmup_steps=20, grad_clip=1.0,
            eval_interval=50, eval_iters=20,
            log_interval=10,
        )
    elif mode == "shakespeare":
        base_kw.update(d=128, max_len=256, v_hidden=512, v_depth=3)
        train_cfg = dict(
            batch_size=16, block_size=128,
            steps=4000, lr=5e-4, weight_decay=0.01,
            warmup_steps=200, grad_clip=1.0,
            eval_interval=200, eval_iters=40,
            log_interval=50,
        )
    else:
        raise ValueError(f"unknown mode: {mode}")

    cfg = HSPLMConfig(**base_kw)
    fg_tag = "" if fixed_gamma_arg is None else f"_g{fixed_gamma_arg:.3f}"
    tag = f"hybrid_k{n_attn}_m{n_splm}{fg_tag}"
    return cfg, train_cfg, tag


def lr_schedule(step: int, lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


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
    ap.add_argument("--mode", choices=["smoke", "shakespeare"],
                    default="shakespeare")
    ap.add_argument("--n-attn", type=int, required=True,
                    help="Number of attention blocks (front stack).")
    ap.add_argument("--n-splm", type=int, required=True,
                    help="Number of SPLM integration steps (back stack).")
    ap.add_argument("--fixed-gamma", type=float, default=None,
                    help="If set, use this fixed gamma for the SPLM stage. "
                         "If None (default), gamma is freely learned.")
    ap.add_argument("--logfreq-path",
                    default=str(PARENT_DIR / "sarf_mass_variant" /
                                "results" / "logfreq_surprisal.npy"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.n_attn < 1 or args.n_splm < 1:
        # We allow n_attn=0 or n_splm=0 only via dedicated baselines
        # (matched_baseline_model.py for all-attn; energetic_minima/train.py
        # for all-SPLM em_ln); refuse them in the hybrid trainer to avoid
        # accidentally re-running a baseline under a hybrid tag.
        raise SystemExit(
            f"Hybrid trainer requires n_attn>=1 and n_splm>=1; got "
            f"n_attn={args.n_attn}, n_splm={args.n_splm}. For all-attention "
            f"or all-SPLM cells use matched_baseline_model.py or "
            f"energetic_minima/train.py respectively.")

    device = args.device or _pick_device()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train_ids, val_ids = load_tiny_shakespeare()

    cfg, train_cfg, tag = build_config(
        args.mode, args.logfreq_path,
        args.n_attn, args.n_splm,
        args.fixed_gamma,
    )
    full_tag = f"{tag}_{args.mode}_seed{args.seed}"
    print(f"[hybrid-train] device={device}  tag={full_tag}")
    print(f"[hybrid-train] tokens: train={len(train_ids):,}  "
          f"val={len(val_ids):,}")
    print(f"[hybrid-train] arch: n_attn={cfg.n_attn}, n_splm={cfg.n_splm}, "
          f"d={cfg.d}, v_hidden={cfg.v_hidden}, v_depth={cfg.v_depth}, "
          f"n_head={cfg.n_head}, fixed_gamma={cfg.fixed_gamma}, "
          f"causal_force={cfg.causal_force}, ln_after_step={cfg.ln_after_step}")

    model = HybridSPLM(cfg).to(device)
    n_params = model.num_params()
    print(f"[hybrid-train] params: {n_params:,}  "
          f"(target ~8.0 M, delta {(n_params - 8_000_000) / 1e6:+.3f} M)")

    optim = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"], betas=(0.9, 0.95),
    )

    log_path = RESULTS_DIR / f"{full_tag}_training_log.jsonl"
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
            print(f"[hybrid-train] step {step+1:5d}/{train_cfg['steps']}   "
                  f"train {avg:.4f}   lr {lr_now:.2e}   "
                  f"grad {grad_norm:.2f}   "
                  f"gamma={model.gamma.item():.3f}   "
                  f"elapsed {elapsed:.0f}s")
            log_f.write(json.dumps({
                "step": step + 1, "train_loss": avg,
                "lr": lr_now, "grad_norm": float(grad_norm),
                "gamma": model.gamma.item(),
                "elapsed_s": elapsed,
            }) + "\n")
            log_f.flush()

        if ((step + 1) % train_cfg["eval_interval"] == 0
                or step + 1 == train_cfg["steps"]):
            val_loss = evaluate(
                model, val_ids, train_cfg["eval_iters"],
                train_cfg["batch_size"], train_cfg["block_size"], rng, device,
            )
            train_loss_eval = evaluate(
                model, train_ids, train_cfg["eval_iters"],
                train_cfg["batch_size"], train_cfg["block_size"], rng, device,
            )
            ppl = math.exp(val_loss)
            print(f"[hybrid-train] >>> eval @ {step+1}: "
                  f"train {train_loss_eval:.4f}   val {val_loss:.4f}   "
                  f"ppl {ppl:.2f}")
            log_f.write(json.dumps({
                "step": step + 1, "train_loss_eval": train_loss_eval,
                "val_loss": val_loss, "val_ppl": ppl,
            }) + "\n")
            log_f.flush()
            loss_history.append((step + 1, train_loss_eval, val_loss))

    log_f.close()
    elapsed = time.time() - t0
    print(f"[hybrid-train] done in {elapsed:.0f}s")

    ckpt_path = RESULTS_DIR / f"{full_tag}_ckpt_latest.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_cfg": asdict(cfg),
        "train_cfg": train_cfg,
        "loss_history": loss_history,
        "final_gamma": float(model.gamma.item()),
        "variant": "hybrid_a_two_stage",
        "tag": full_tag,
        "n_params": n_params,
    }, ckpt_path)
    print(f"[hybrid-train] saved checkpoint -> {ckpt_path}")

    if loss_history:
        steps_e = [e[0] for e in loss_history]
        tr_e    = [e[1] for e in loss_history]
        va_e    = [e[2] for e in loss_history]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(steps_e, tr_e, marker="o", label="train (eval)")
        ax.plot(steps_e, va_e, marker="s", label="val")
        ax.set_xlabel("step")
        ax.set_ylabel("cross-entropy")
        ax.set_title(f"hybrid k={cfg.n_attn} m={cfg.n_splm}  "
                     f"({args.mode}, seed {args.seed})")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        png_path = RESULTS_DIR / f"{full_tag}_loss_curve.png"
        fig.savefig(png_path, dpi=130)
        plt.close(fig)
        print(f"[hybrid-train] saved loss curve -> {png_path}")

    summary_path = RESULTS_DIR / f"{full_tag}_summary.md"
    with summary_path.open("w") as f:
        f.write(f"# Hybrid SPLM (Variant A) -- {args.mode} training summary\n\n")
        f.write(f"- Tag: `{full_tag}`\n")
        f.write(f"- Architecture: n_attn={cfg.n_attn} attention blocks + "
                f"n_splm={cfg.n_splm} SPLM integration steps "
                f"(shared V_theta, leak-fixed ξ).\n")
        f.write(f"- Device: `{device}`\n")
        f.write(f"- Parameters: **{n_params:,}** "
                f"(target ~8.0 M, delta "
                f"{(n_params - 8_000_000) / 1e6:+.3f} M)\n")
        f.write(f"- Model config: `{asdict(cfg)}`\n")
        f.write(f"- Train config: `{train_cfg}`\n")
        f.write(f"- Tokens: train={len(train_ids):,}, val={len(val_ids):,}\n")
        f.write(f"- Wall-clock time: {elapsed:.0f}s\n")
        if loss_history:
            final_train, final_val = loss_history[-1][1], loss_history[-1][2]
            f.write(f"- Final train loss: {final_train:.4f}\n")
            f.write(f"- Final val loss: {final_val:.4f} "
                    f"(ppl {math.exp(final_val):.2f})\n")
        f.write(f"- Final gamma: {model.gamma.item():.4f}\n")
        f.write(f"- Loss curve: `{full_tag}_loss_curve.png`\n")
        f.write(f"- Checkpoint: `{full_tag}_ckpt_latest.pt`\n")
    print(f"[hybrid-train] saved summary -> {summary_path}")


if __name__ == "__main__":
    main()
