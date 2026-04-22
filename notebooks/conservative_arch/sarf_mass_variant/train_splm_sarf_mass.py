"""
Training loop for SARF-faithful SPLM with per-token semantic mass.

Mirrors sarf_variant/train_splm_sarf.py but:
  - imports ScalarPotentialLMSARFMass / SPLMSARFMassConfig from model_sarf_mass
  - adds `--mass-mode {global, embed_head, logfreq}`
  - logs mass statistics (mean/std/min/max) at every eval step
  - writes outputs under sarf_mass_variant/results/ with
    `splm_sarfmass_<mass_mode>_<mode>_` prefix

Hyperparameters (batch, block, steps, lr, ...) are identical to the SARF
baseline so any difference in val loss / diagnostics is attributable to the
mass parameterisation alone.

Modes
-----
  --mode smoke        : 300-step sanity check.
  --mode shakespeare  : full convergence run on Tiny Shakespeare.
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

SCRIPT_DIR   = Path(__file__).parent
PARENT_DIR   = SCRIPT_DIR.parent
RESULTS_DIR  = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(PARENT_DIR))
from data_module import get_batch, load_tiny_shakespeare, load_tiny_stories  # noqa: E402

from model_sarf_mass import ScalarPotentialLMSARFMass, SPLMSARFMassConfig  # noqa: E402


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_config(mode: str, mass_mode: str,
                 logfreq_path: str | None) -> tuple[SPLMSARFMassConfig, dict]:
    if mode == "smoke":
        model_cfg = SPLMSARFMassConfig(
            vocab_size=50257, d=64, max_len=128,
            v_hidden=128, v_depth=2, L=4,
            init_m=1.0, init_gamma=1.0,
            mass_mode=mass_mode,
            logfreq_init_alpha=0.1,
            logfreq_path=logfreq_path,
        )
        train_cfg = dict(
            batch_size=8, block_size=64,
            steps=300, lr=1e-3, weight_decay=0.0,
            warmup_steps=20, grad_clip=1.0,
            eval_interval=50, eval_iters=20,
            log_interval=10,
        )
    elif mode == "shakespeare":
        model_cfg = SPLMSARFMassConfig(
            vocab_size=50257, d=128, max_len=256,
            v_hidden=512, v_depth=3, L=8,
            init_m=1.0, init_gamma=1.0,
            mass_mode=mass_mode,
            logfreq_init_alpha=0.1,
            logfreq_path=logfreq_path,
        )
        train_cfg = dict(
            batch_size=16, block_size=128,
            steps=4000, lr=5e-4, weight_decay=0.01,
            warmup_steps=200, grad_clip=1.0,
            eval_interval=200, eval_iters=40,
            log_interval=50,
        )
    elif mode == "tinystories":
        model_cfg = SPLMSARFMassConfig(
            vocab_size=50257, d=192, max_len=256,
            v_hidden=768, v_depth=3, L=8,
            init_m=1.0, init_gamma=1.0,
            mass_mode=mass_mode,
            logfreq_init_alpha=0.1,
            logfreq_path=logfreq_path,
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
def evaluate(model: ScalarPotentialLMSARFMass, ids: np.ndarray,
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
    ap.add_argument("--mass-mode", dest="mass_mode",
                    choices=["global", "embed_head", "logfreq"],
                    default="embed_head")
    ap.add_argument("--logfreq-path", dest="logfreq_path",
                    default=str(RESULTS_DIR / "logfreq_surprisal.npy"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_train_tokens", type=int, default=None)
    args = ap.parse_args()

    device = args.device or _pick_device()
    print(f"[train-mass] device={device}  mode={args.mode}  "
          f"mass_mode={args.mass_mode}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    if args.mode == "tinystories":
        train_ids, val_ids = load_tiny_stories(
            n_train_files=1,
            max_train_tokens=args.max_train_tokens or 5_000_000,
        )
    else:
        train_ids, val_ids = load_tiny_shakespeare()
    print(f"[train-mass] tokens: train={len(train_ids):,}  val={len(val_ids):,}")

    logfreq_path = None
    if args.mass_mode == "logfreq":
        if not Path(args.logfreq_path).exists():
            raise FileNotFoundError(
                f"logfreq surprisal file not found at {args.logfreq_path}.  "
                f"Run `python3 compute_unigram_frequencies.py` first."
            )
        logfreq_path = args.logfreq_path

    model_cfg, train_cfg = build_config(args.mode, args.mass_mode, logfreq_path)
    model = ScalarPotentialLMSARFMass(model_cfg).to(device)
    n_params = model.num_params()
    print(f"[train-mass] params: {n_params:,}   d={model_cfg.d}  L={model_cfg.L}  "
          f"v_hidden={model_cfg.v_hidden}")

    xb0, _ = get_batch(train_ids, train_cfg["batch_size"],
                       train_cfg["block_size"], rng)
    x0 = torch.from_numpy(xb0).to(device)
    init_mass = model.mass_stats(x0)
    print(f"[train-mass] init mass: mean={init_mass['mean']:.3f}  "
          f"std={init_mass['std']:.3f}  "
          f"min={init_mass['min']:.3f}  max={init_mass['max']:.3f}")

    optim = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"], betas=(0.9, 0.95),
    )

    tag = f"splm_sarfmass_{args.mass_mode}_{args.mode}"
    log_path = RESULTS_DIR / f"{tag}_training_log.jsonl"
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
            mstats = model.mass_stats(x)
            print(f"[train-mass] step {step+1:5d}/{train_cfg['steps']}   "
                  f"train {avg:.4f}   lr {lr_now:.2e}   "
                  f"grad {grad_norm:.2f}   "
                  f"m[mean {mstats['mean']:.3f} std {mstats['std']:.3f}]   "
                  f"gamma={model.gamma.item():.3f}   elapsed {elapsed:.0f}s")
            log_f.write(json.dumps({
                "step": step + 1, "train_loss": avg,
                "lr": lr_now, "grad_norm": float(grad_norm),
                "mass_mean": mstats["mean"], "mass_std": mstats["std"],
                "mass_min":  mstats["min"],  "mass_max": mstats["max"],
                "gamma": model.gamma.item(),
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
            print(f"[train-mass] >>> eval @ {step+1}: train {train_loss_eval:.4f}   "
                  f"val {val_loss:.4f}   ppl {ppl:.2f}")
            log_f.write(json.dumps({
                "step": step + 1, "train_loss_eval": train_loss_eval,
                "val_loss": val_loss, "val_ppl": ppl,
            }) + "\n")
            log_f.flush()
            loss_history.append((step + 1, train_loss_eval, val_loss))

    log_f.close()
    elapsed = time.time() - t0
    print(f"[train-mass] done in {elapsed:.0f}s")

    final_mstats = model.mass_stats(x)

    ckpt_path = RESULTS_DIR / f"{tag}_ckpt_latest.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_cfg": asdict(model_cfg),
        "train_cfg": train_cfg,
        "loss_history": loss_history,
        "final_mass_stats": final_mstats,
        "final_gamma": float(model.gamma.item()),
        "variant": "sarf_mass",
        "mass_mode": args.mass_mode,
    }, ckpt_path)
    print(f"[train-mass] saved checkpoint -> {ckpt_path}")

    if loss_history:
        steps_e = [e[0] for e in loss_history]
        tr_e    = [e[1] for e in loss_history]
        va_e    = [e[2] for e in loss_history]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(steps_e, tr_e, marker="o", label="train (eval)")
        ax.plot(steps_e, va_e, marker="s", label="val")
        ax.set_xlabel("step")
        ax.set_ylabel("cross-entropy")
        ax.set_title(f"SARF+mass ({args.mass_mode}) -- {args.mode}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        png_path = RESULTS_DIR / f"{tag}_loss_curve.png"
        fig.savefig(png_path, dpi=130)
        plt.close(fig)
        print(f"[train-mass] saved loss curve -> {png_path}")

    summary_path = RESULTS_DIR / f"{tag}_summary.md"
    with summary_path.open("w") as f:
        f.write(f"# SARF+mass({args.mass_mode}) -- {args.mode} training summary\n\n")
        f.write(f"Variant: **SARF-faithful with mass_mode = `{args.mass_mode}`**.\n\n")
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
        f.write(f"- Final mass: mean={final_mstats['mean']:.4f}, "
                f"std={final_mstats['std']:.4f}, "
                f"min={final_mstats['min']:.4f}, "
                f"max={final_mstats['max']:.4f}\n")
        f.write(f"- Final gamma: {model.gamma.item():.4f}\n")
        f.write(f"- Loss curve: `{tag}_loss_curve.png`\n")
        f.write(f"- Checkpoint: `{tag}_ckpt_latest.pt`\n")
        f.write(f"- Log: `{tag}_training_log.jsonl`\n")
    print(f"[train-mass] saved summary -> {summary_path}")


if __name__ == "__main__":
    main()
