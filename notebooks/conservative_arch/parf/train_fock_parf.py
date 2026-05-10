"""
FockPARFLM trainer — Phase 1 (Dyck falsifier) and Phase 2 (TinyStories).

This trainer supports two corpus modes:

  --corpus dyck       : Synthetic Dyck_n for expressivity falsification.
                        Tests whether FockPARFLM can recognise CF languages
                        past the v0-ceiling collapse depth D*.
  --corpus tinystories: TinyStories (5M tokens).  Same as P10/P11 ladder.

Architecture selection:

  --arch parflm       : Baseline PARFLM (SparsePARFLM, v0-only).
  --arch fock         : FockPARFLM (v0 + v2 creation/destruction).

This script is designed to run the F1 falsifier experiments and the P11
TinyStories integration experiments described in
companion_notes/Augmenting_PARFLM_to_handle_MCS_Languages.md.

Outputs (under parf/results/fock/):
  - {tag}_training_log.jsonl
  - {tag}_ckpt_latest.pt
  - {tag}_loss_curve.png
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
RESULTS_DIR = SCRIPT_DIR / "results" / "fock"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from data_module import get_batch, load_tiny_stories  # noqa: E402
from model_parf_sparse import SparsePARFConfig, SparsePARFLM  # noqa: E402
from model_fock_parf import FockPARFConfig, FockPARFLM  # noqa: E402
from dyck_data import (  # noqa: E402
    DyckConfig,
    generate_dyck_dataset,
    generate_depth_controlled_dataset,
    get_dyck_batch,
)

DEFAULT_LOGFREQ_PATH = (
    PARENT_DIR / "scaleup" / "results" / "logfreq_surprisal_tinystories.npy"
)


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
def evaluate_dyck(model, x_val, y_val, batch_size, rng, device) -> float:
    """Evaluate on Dyck validation set; returns mean loss."""
    model.eval()
    losses = []
    n_batches = max(1, len(x_val) // batch_size)
    for i in range(min(n_batches, 40)):
        xb, yb = get_dyck_batch(x_val, y_val, batch_size, rng)
        x = torch.from_numpy(xb).to(device)
        y = torch.from_numpy(yb).to(device)
        with torch.enable_grad():
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


@torch.no_grad()
def evaluate_tinystories(model, val_ids, batch_size, block_size, rng, device) -> float:
    model.eval()
    losses = []
    for _ in range(40):
        xb, yb = get_batch(val_ids, batch_size, block_size, rng)
        x = torch.from_numpy(xb).to(device)
        y = torch.from_numpy(yb).to(device)
        with torch.enable_grad():
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def evaluate_dyck_accuracy(model, x_test, y_test, cfg_dyck, device) -> dict:
    """Compute per-position accuracy on Dyck test set.

    Returns dict with overall accuracy and breakdown by whether the target
    was an open or close bracket.
    """
    model.eval()
    x = torch.from_numpy(x_test).to(device)
    y_true = torch.from_numpy(y_test).to(device)

    with torch.enable_grad():
        logits, _ = model(x, y_true)

    preds = logits.argmax(dim=-1)  # (N, T)
    valid = y_true != -100

    correct = (preds == y_true) & valid
    total_correct = correct.sum().item()
    total_valid = valid.sum().item()

    results = {
        "accuracy": total_correct / max(total_valid, 1),
        "n_correct": total_correct,
        "n_total": total_valid,
    }

    model.train()
    return results


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

def build_dyck_config(
    arch: str,
    n_types: int = 2,
    max_depth: int = 8,
    d: int = 64,
    L: int = 4,
    v_hidden: int = 128,
    n_registers: int = 16,
    stack_discipline: bool = True,
) -> tuple:
    """Build config for Dyck falsifier experiments (small scale)."""
    dyck_cfg = DyckConfig(
        n_types=n_types,
        max_depth=max_depth,
        min_length=8,
        max_length=64,
        p_open=0.55,
    )

    base_kw = dict(
        vocab_size=dyck_cfg.vocab_size,
        d=d,
        max_len=66,  # max_length + BOS + EOS
        L=L,
        v_hidden=v_hidden,
        v_depth=2,
        v_phi_kind="structural",
        v_phi_d_type=8,
        v_phi_d_angle=4,
        v_phi_phi_hidden=16,
        v_phi_theta_hidden=16,
        v_phi_mlp_hidden=32,
        mass_mode="global",
        causal_force=True,
        ln_after_step=True,
        use_grad_checkpoint=False,
        top_k=8,
        score_head_hidden=16,
        gumbel_tau_init=1.0,
        gumbel_tau_min=0.1,
        gumbel_noise=True,
    )

    train_cfg = dict(
        batch_size=32,
        steps=4000,
        lr=3e-4,
        weight_decay=0.01,
        warmup_steps=200,
        grad_clip=1.0,
        eval_interval=200,
        log_interval=50,
        n_train_samples=10000,
        n_val_samples=2000,
    )

    if arch == "fock":
        cfg = FockPARFConfig(
            **base_kw,
            n_registers=n_registers,
            creation_gate_hidden=max(d // 4, 16),
            stack_discipline=stack_discipline,
            register_salience_decay=0.9,
            register_salience_threshold=0.1,
        )
    else:
        cfg = SparsePARFConfig(**base_kw)

    return cfg, train_cfg, dyck_cfg


def build_tinystories_config(
    arch: str,
    logfreq_path: str | None = None,
    n_registers: int = 32,
    stack_discipline: bool = True,
    v_hidden: int = 1024,
) -> tuple:
    """Build config for TinyStories experiments (P11 ladder, P10f-scale)."""
    base_kw = dict(
        vocab_size=50257,
        d=256,
        max_len=1024,
        L=8,
        v_hidden=v_hidden,
        v_depth=3,
        v_phi_kind="structural",
        v_phi_d_type=32,
        v_phi_d_angle=16,
        v_phi_phi_hidden=16,
        v_phi_theta_hidden=16,
        v_phi_mlp_hidden=32,
        mass_mode="logfreq" if logfreq_path else "global",
        logfreq_path=logfreq_path,
        logfreq_init_alpha=0.1,
        causal_force=True,
        ln_after_step=True,
        use_grad_checkpoint=False,
        top_k=4,
        score_head_hidden=32,
        gumbel_tau_init=1.0,
        gumbel_tau_min=0.1,
        gumbel_noise=True,
    )

    train_cfg = dict(
        batch_size=16,
        block_size=512,
        steps=8000,
        lr=5e-4,
        weight_decay=0.01,
        warmup_steps=400,
        grad_clip=1.0,
        eval_interval=400,
        eval_iters=40,
        log_interval=50,
    )

    if arch == "fock":
        cfg = FockPARFConfig(
            **base_kw,
            n_registers=n_registers,
            creation_gate_hidden=64,
            stack_discipline=stack_discipline,
            register_salience_decay=0.9,
            register_salience_threshold=0.1,
        )
    else:
        cfg = SparsePARFConfig(**base_kw)

    return cfg, train_cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="FockPARFLM experiment trainer")
    ap.add_argument("--corpus", choices=["dyck", "tinystories"], default="dyck")
    ap.add_argument("--arch", choices=["parflm", "fock"], default="fock")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag-suffix", dest="tag_suffix", default="")

    # Dyck-specific
    ap.add_argument("--dyck-n-types", type=int, default=2, dest="dyck_n_types")
    ap.add_argument("--dyck-max-depth", type=int, default=8, dest="dyck_max_depth")
    ap.add_argument("--dyck-test-depth-min", type=int, default=None,
                    dest="dyck_test_depth_min",
                    help="If set, generate a depth-controlled test set at "
                         "[min, max] to probe expressivity at specific depths.")
    ap.add_argument("--dyck-test-depth-max", type=int, default=None,
                    dest="dyck_test_depth_max")

    # Architecture
    ap.add_argument("--d", type=int, default=None)
    ap.add_argument("--L", type=int, default=None)
    ap.add_argument("--v-hidden", type=int, default=None, dest="v_hidden")
    ap.add_argument("--n-registers", type=int, default=None, dest="n_registers")
    ap.add_argument("--no-stack", action="store_true", dest="no_stack")

    # Training
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    ap.add_argument("--results-dir", dest="results_dir", default=None)
    ap.add_argument("--logfreq-path", dest="logfreq_path", default=None)
    ap.add_argument("--max-train-tokens", type=int, default=5_000_000,
                    dest="max_train_tokens")

    args = ap.parse_args()

    device = args.device or _pick_device()
    results_dir = (
        Path(args.results_dir) if args.results_dir else RESULTS_DIR
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # --- Build config ---
    if args.corpus == "dyck":
        n_reg = args.n_registers if args.n_registers is not None else 16
        d_val = args.d if args.d is not None else 64
        L_val = args.L if args.L is not None else 4
        vh_val = args.v_hidden if args.v_hidden is not None else 128
        cfg, train_cfg, dyck_cfg = build_dyck_config(
            arch=args.arch,
            n_types=args.dyck_n_types,
            max_depth=args.dyck_max_depth,
            d=d_val, L=L_val, v_hidden=vh_val,
            n_registers=n_reg,
            stack_discipline=not args.no_stack,
        )
    else:
        logfreq_path = args.logfreq_path or (
            str(DEFAULT_LOGFREQ_PATH)
            if DEFAULT_LOGFREQ_PATH.exists() else None
        )
        n_reg = args.n_registers if args.n_registers is not None else 32
        vh_val = args.v_hidden if args.v_hidden is not None else 1024
        cfg, train_cfg = build_tinystories_config(
            arch=args.arch,
            logfreq_path=logfreq_path,
            n_registers=n_reg,
            stack_discipline=not args.no_stack,
            v_hidden=vh_val,
        )
        dyck_cfg = None

    # CLI overrides on train_cfg.
    if args.steps is not None:
        train_cfg["steps"] = args.steps
    if args.lr is not None:
        train_cfg["lr"] = args.lr
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size

    # --- Build tag ---
    arch_tag = "fock" if args.arch == "fock" else "parflm"
    if args.arch == "fock":
        stack_tag = "stack" if (not args.no_stack) else "bag"
        arch_tag = f"fock_M{cfg.n_registers}_{stack_tag}"
    corpus_tag = args.corpus
    if args.corpus == "dyck":
        corpus_tag = f"dyck{args.dyck_n_types}_d{args.dyck_max_depth}"
    tag = f"F1_{arch_tag}_{corpus_tag}_seed{args.seed}"
    if args.tag_suffix:
        tag = f"{tag}_{args.tag_suffix}"

    # --- Load data ---
    if args.corpus == "dyck":
        print(f"[fock-trainer] Generating Dyck_{args.dyck_n_types} data "
              f"(max_depth={args.dyck_max_depth})...")
        x_train, y_train = generate_dyck_dataset(
            dyck_cfg, n_samples=train_cfg["n_train_samples"], seed=args.seed,
        )
        x_val, y_val = generate_dyck_dataset(
            dyck_cfg, n_samples=train_cfg["n_val_samples"],
            seed=args.seed + 1000,
        )
        print(f"[fock-trainer] Train: {len(x_train)} samples, "
              f"Val: {len(x_val)} samples, "
              f"Seq len: {x_train.shape[1]}")

        # Optional depth-controlled test set.
        x_test_deep = y_test_deep = None
        if args.dyck_test_depth_min is not None:
            dmin = args.dyck_test_depth_min
            dmax = args.dyck_test_depth_max or args.dyck_max_depth
            print(f"[fock-trainer] Generating depth-controlled test set "
                  f"(depth {dmin}-{dmax})...")
            x_test_deep, y_test_deep, test_depths = \
                generate_depth_controlled_dataset(
                    dyck_cfg, n_samples=500,
                    min_depth=dmin, max_depth=dmax, seed=args.seed + 2000,
                )
            print(f"[fock-trainer] Deep test set: {len(x_test_deep)} samples")
        train_ids = val_ids = None
    else:
        train_ids, val_ids = load_tiny_stories(
            max_train_tokens=args.max_train_tokens,
        )
        print(f"[fock-trainer] TinyStories tokens: train={len(train_ids):,}  "
              f"val={len(val_ids):,}")
        x_train = y_train = x_val = y_val = None
        x_test_deep = y_test_deep = None

    # --- Build model ---
    if isinstance(cfg, FockPARFConfig):
        model = FockPARFLM(cfg).to(device)
    else:
        model = SparsePARFLM(cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    fock_overhead = (
        model.get_register_overhead()
        if isinstance(model, FockPARFLM) else 0
    )
    print(f"[fock-trainer] tag={tag}  device={device}")
    print(f"[fock-trainer] arch={args.arch}  params={n_params:,}  "
          f"fock_overhead={fock_overhead:,}")
    print(f"[fock-trainer] cfg: d={cfg.d} L={cfg.L} "
          f"v_hidden={cfg.v_hidden} top_k={cfg.top_k}")
    if isinstance(cfg, FockPARFConfig):
        print(f"[fock-trainer] fock: M={cfg.n_registers} "
              f"stack={cfg.stack_discipline} "
              f"decay={cfg.register_salience_decay}")

    # --- Optimiser ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        betas=(0.9, 0.95),
    )

    # --- Training loop ---
    steps = train_cfg["steps"]
    warmup = train_cfg["warmup_steps"]
    log_path = results_dir / f"{tag}_training_log.jsonl"
    loss_history = []
    val_history = []
    t0 = time.time()

    model.train()
    print(f"[fock-trainer] Starting training: {steps} steps, "
          f"lr={train_cfg['lr']}, batch={train_cfg['batch_size']}")

    for step in range(steps):
        # LR schedule.
        current_lr = lr_schedule(step, train_cfg["lr"], warmup, steps)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        # Gumbel tau anneal.
        if hasattr(model, "set_gumbel_tau"):
            tau = tau_schedule(step, cfg.gumbel_tau_init, cfg.gumbel_tau_min, steps)
            model.set_gumbel_tau(tau)

        # Get batch.
        if args.corpus == "dyck":
            xb, yb = get_dyck_batch(
                x_train, y_train, train_cfg["batch_size"], rng
            )
            x = torch.from_numpy(xb).to(device)
            y = torch.from_numpy(yb).to(device)
        else:
            xb, yb = get_batch(
                train_ids, train_cfg["batch_size"],
                train_cfg["block_size"], rng,
            )
            x = torch.from_numpy(xb).to(device)
            y = torch.from_numpy(yb).to(device)

        # Forward + backward.
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        # Logging.
        if step % train_cfg["log_interval"] == 0:
            elapsed = time.time() - t0
            print(f"  step {step:5d}/{steps}  loss={loss_val:.4f}  "
                  f"lr={current_lr:.2e}  "
                  f"elapsed={elapsed:.1f}s")

        # Eval.
        if step > 0 and step % train_cfg["eval_interval"] == 0:
            if args.corpus == "dyck":
                val_loss = evaluate_dyck(
                    model, x_val, y_val,
                    train_cfg["batch_size"], rng, device,
                )
                val_ppl = math.exp(min(val_loss, 20.0))
                entry = {
                    "step": step, "val_loss": val_loss,
                    "val_ppl": val_ppl, "train_loss": loss_val,
                }
                # Accuracy on deep test set if available.
                if x_test_deep is not None:
                    acc_info = evaluate_dyck_accuracy(
                        model, x_test_deep, y_test_deep, dyck_cfg, device,
                    )
                    entry["deep_test_accuracy"] = acc_info["accuracy"]
                    print(f"  [eval] step={step} val_loss={val_loss:.4f} "
                          f"val_ppl={val_ppl:.2f} "
                          f"deep_acc={acc_info['accuracy']:.4f}")
                else:
                    print(f"  [eval] step={step} val_loss={val_loss:.4f} "
                          f"val_ppl={val_ppl:.2f}")
            else:
                val_loss = evaluate_tinystories(
                    model, val_ids, train_cfg["batch_size"],
                    train_cfg["block_size"], rng, device,
                )
                val_ppl = math.exp(min(val_loss, 20.0))
                entry = {
                    "step": step, "val_loss": val_loss,
                    "val_ppl": val_ppl, "train_loss": loss_val,
                }
                print(f"  [eval] step={step} val_loss={val_loss:.4f} "
                      f"val_ppl={val_ppl:.2f}")

            val_history.append(entry)
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

    # --- Final eval ---
    elapsed = time.time() - t0
    if args.corpus == "dyck":
        final_val_loss = evaluate_dyck(
            model, x_val, y_val, train_cfg["batch_size"], rng, device,
        )
    else:
        final_val_loss = evaluate_tinystories(
            model, val_ids, train_cfg["batch_size"],
            train_cfg["block_size"], rng, device,
        )
    final_val_ppl = math.exp(min(final_val_loss, 20.0))
    print(f"\n[fock-trainer] FINAL: val_loss={final_val_loss:.4f}  "
          f"val_ppl={final_val_ppl:.2f}  elapsed={elapsed:.1f}s")

    # Final deep-test accuracy for Dyck.
    if args.corpus == "dyck" and x_test_deep is not None:
        final_acc = evaluate_dyck_accuracy(
            model, x_test_deep, y_test_deep, dyck_cfg, device,
        )
        print(f"[fock-trainer] Deep test accuracy: "
              f"{final_acc['accuracy']:.4f} "
              f"({final_acc['n_correct']}/{final_acc['n_total']})")

    # --- Save checkpoint ---
    ckpt_path = results_dir / f"{tag}_ckpt_latest.pt"
    ckpt = {
        "model_state_dict": model.state_dict(),
        "model_cfg": asdict(cfg),
        "train_cfg": train_cfg,
        "tag": tag,
        "arch": args.arch,
        "corpus": args.corpus,
        "seed": args.seed,
        "n_params": n_params,
        "fock_overhead": fock_overhead,
        "final_val_loss": final_val_loss,
        "final_val_ppl": final_val_ppl,
        "elapsed_sec": elapsed,
        "loss_history": loss_history,
        "val_history": val_history,
    }
    if args.corpus == "dyck":
        ckpt["dyck_cfg"] = asdict(dyck_cfg)
    torch.save(ckpt, ckpt_path)
    print(f"[fock-trainer] Checkpoint saved to {ckpt_path}")

    # --- Loss curve ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(loss_history, alpha=0.3, label="train loss")
    if val_history:
        val_steps = [e["step"] for e in val_history]
        val_losses = [e["val_loss"] for e in val_history]
        ax.plot(val_steps, val_losses, "r-o", label="val loss", markersize=4)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(f"{tag}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / f"{tag}_loss_curve.png", dpi=150)
    plt.close(fig)
    print(f"[fock-trainer] Loss curve saved.")


if __name__ == "__main__":
    main()
