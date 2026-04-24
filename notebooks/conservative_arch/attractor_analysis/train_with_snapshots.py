"""
Re-train the SARF-mass SPLM (Euler, L=8, logfreq, Tiny Shakespeare) while
saving intermediate checkpoints at log-spaced training steps, so we can
visualise how the V_theta landscape evolves during training.

Outputs go to results/snapshots/<tag>/ckpt_step<step>.pt
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).parent
PARENT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
SNAP_ROOT = RESULTS_DIR / "snapshots"
SNAP_ROOT.mkdir(exist_ok=True)
sys.path.insert(0, str(PARENT_DIR))

from data_module import get_batch, load_tiny_shakespeare  # noqa: E402
from sarf_mass_variant.model_sarf_mass import (           # noqa: E402
    ScalarPotentialLMSARFMass, SPLMSARFMassConfig,
)


def lr_schedule(step, lr, warmup, total):
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
        with torch.enable_grad():
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def save_snapshot(model, cfg_dict, snap_dir: Path, step: int,
                  val_loss: float | None):
    path = snap_dir / f"ckpt_step{step:05d}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_cfg": cfg_dict,
        "variant": "sarf_mass",
        "mass_mode": cfg_dict.get("mass_mode"),
        "step": int(step),
        "val_loss": val_loss,
    }, path)
    print(f"  snapshot -> {path}  (val_loss={val_loss})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="euler_shakespeare_snapshots")
    ap.add_argument("--snapshot_steps", default="0,50,200,500,1000,2000,4000",
                    help="Comma-separated training-step indices to snapshot.")
    ap.add_argument("--total_steps", type=int, default=4000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--eval_interval", type=int, default=500)
    ap.add_argument("--eval_iters",    type=int, default=40)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = args.device or (
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"[train-snap] device={device}  tag={args.tag}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    logfreq_path = str(PARENT_DIR / "results" / "logfreq_surprisal.npy")
    if not Path(logfreq_path).exists():
        logfreq_path = str(PARENT_DIR / "sarf_mass_variant"
                                      / "results" / "logfreq_surprisal.npy")

    model_cfg = SPLMSARFMassConfig(
        vocab_size=50257, d=128, max_len=256,
        v_hidden=512, v_depth=3, L=8,
        init_m=1.0, init_gamma=1.0,
        mass_mode="logfreq",
        logfreq_init_alpha=0.1,
        logfreq_path=logfreq_path,
    )
    train_ids, val_ids = load_tiny_shakespeare()

    snap_dir = SNAP_ROOT / args.tag
    snap_dir.mkdir(exist_ok=True)
    snapshot_steps = sorted({int(s) for s in args.snapshot_steps.split(",")})
    print(f"[train-snap] snapshot_steps = {snapshot_steps}")

    model = ScalarPotentialLMSARFMass(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train-snap] params: {n_params:,}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)

    cfg_dict = asdict(model_cfg)

    if 0 in snapshot_steps:
        save_snapshot(model, cfg_dict, snap_dir, 0, val_loss=None)

    t0 = time.time()
    model.train()
    running, n_run = 0.0, 0

    for step in range(1, args.total_steps + 1):
        lr_now = lr_schedule(step, args.lr, args.warmup_steps, args.total_steps)
        for g in optim.param_groups:
            g["lr"] = lr_now

        xb, yb = get_batch(train_ids, args.batch_size, args.block_size, rng)
        x = torch.from_numpy(xb).to(device)
        y = torch.from_numpy(yb).to(device)
        _, loss = model(x, y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        running += loss.item(); n_run += 1

        if step % 100 == 0:
            avg = running / n_run
            print(f"[train-snap] step {step:5d}/{args.total_steps}  "
                  f"train {avg:.4f}  lr {lr_now:.2e}  "
                  f"elapsed {time.time()-t0:.0f}s")
            running, n_run = 0.0, 0

        if step in snapshot_steps:
            val_loss = evaluate(model, val_ids, args.eval_iters,
                                args.batch_size, args.block_size,
                                rng, device)
            save_snapshot(model, cfg_dict, snap_dir, step, val_loss)

    print(f"[train-snap] done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
