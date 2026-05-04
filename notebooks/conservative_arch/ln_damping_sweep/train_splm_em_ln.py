"""
Training loop for the E5 LN-after-step damping sweep.

Identical to sarf_mass_variant/train_splm_sarf_mass.py except it uses
ScalarPotentialLMSARFMassLN (LayerNorm after every damped Euler step)
instead of the plain Euler integrator.  The --fixed-gamma flag is
inherited from SPLMSARFMassConfig (base class of SPLMSARFMassLNConfig),
so the sweep harness can call this script identically to the E4 trainer.

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

SCRIPT_DIR  = Path(__file__).parent
PARENT_DIR  = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# --- model imports ---
EM_MINIMA_DIR = PARENT_DIR / "energetic_minima"
SARF_MASS_DIR = PARENT_DIR / "sarf_mass_variant"
sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(EM_MINIMA_DIR))
sys.path.insert(0, str(SARF_MASS_DIR))

from data_module import get_batch, load_tiny_shakespeare  # noqa: E402
from model_ln import ScalarPotentialLMSARFMassLN, SPLMSARFMassLNConfig  # noqa: E402


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_config(
    mode: str,
    logfreq_path: str | None,
    fixed_gamma: float | None = None,
) -> tuple[SPLMSARFMassLNConfig, dict]:
    base_kw = dict(
        vocab_size=50257,
        mass_mode="logfreq",
        logfreq_init_alpha=0.1,
        logfreq_path=logfreq_path,
        ln_after_step=True,
        fixed_gamma=fixed_gamma,
    )
    if mode == "smoke":
        model_cfg = SPLMSARFMassLNConfig(
            d=64, max_len=128, v_hidden=128, v_depth=2, L=4,
            init_m=1.0, init_gamma=1.0,
            **base_kw,
        )
        train_cfg = dict(
            batch_size=8, block_size=64,
            steps=300, lr=1e-3, weight_decay=0.0,
            warmup_steps=20, grad_clip=1.0,
            eval_interval=50, eval_iters=20,
            log_interval=10,
        )
    elif mode == "shakespeare":
        model_cfg = SPLMSARFMassLNConfig(
            d=128, max_len=256, v_hidden=512, v_depth=3, L=8,
            init_m=1.0, init_gamma=1.0,
            **base_kw,
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
def evaluate(model: ScalarPotentialLMSARFMassLN, ids: np.ndarray,
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
    ap.add_argument("--mode", choices=["smoke", "shakespeare"], default="smoke")
    ap.add_argument("--logfreq-path", dest="logfreq_path",
                    default=str(SARF_MASS_DIR / "results" / "logfreq_surprisal.npy"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--fixed-gamma", dest="fixed_gamma", type=float, default=None,
        help="Fix the damping coefficient at this value (disables learning of gamma). "
             "Used by the E5 LN damping sweep.",
    )
    ap.add_argument(
        "--tag-suffix", dest="tag_suffix", type=str, default="",
        help="Optional suffix appended to the output tag, e.g. 'gamma0p10'.",
    )
    ap.add_argument(
        "--results-dir", dest="results_dir", type=str, default=None,
        help="Override output directory (default: ln_damping_sweep/results/). "
             "Used by the sweep to write into ln_damping_sweep/results/<tag>/.",
    )
    args = ap.parse_args()

    device = args.device or _pick_device()
    results_dir = (
        Path(args.results_dir).expanduser().resolve()
        if args.results_dir is not None else RESULTS_DIR
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train-em-ln] device={device}  mode={args.mode}  "
          f"fixed_gamma={args.fixed_gamma!r}  results_dir={results_dir}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train_ids, val_ids = load_tiny_shakespeare()
    print(f"[train-em-ln] tokens: train={len(train_ids):,}  val={len(val_ids):,}")

    logfreq_path = args.logfreq_path
    if not Path(logfreq_path).exists():
        raise FileNotFoundError(
            f"logfreq surprisal file not found at {logfreq_path}.  "
            "Run sarf_mass_variant/compute_unigram_frequencies.py first."
        )

    model_cfg, train_cfg = build_config(
        args.mode, logfreq_path, fixed_gamma=args.fixed_gamma,
    )
    model = ScalarPotentialLMSARFMassLN(model_cfg).to(device)
    n_params = model.num_params()
    print(f"[train-em-ln] params: {n_params:,}   d={model_cfg.d}  L={model_cfg.L}  "
          f"v_hidden={model_cfg.v_hidden}  ln_after_step={model_cfg.ln_after_step}")
    print(f"[train-em-ln] causal_force={model_cfg.causal_force}  "
          f"({'FIXED (post-bug)' if model_cfg.causal_force else 'BUGGY (pre-fix)'})  "
          f"steps={train_cfg['steps']}")

    xb0, _ = get_batch(train_ids, train_cfg["batch_size"],
                       train_cfg["block_size"], rng)
    x0 = torch.from_numpy(xb0).to(device)
    init_mass = model.mass_stats(x0)
    print(f"[train-em-ln] init mass: mean={init_mass['mean']:.3f}  "
          f"std={init_mass['std']:.3f}  "
          f"min={init_mass['min']:.3f}  max={init_mass['max']:.3f}")

    optim = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"], betas=(0.9, 0.95),
    )

    tag = f"splm_em_ln_{args.mode}"
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
            mstats = model.mass_stats(x)
            gamma_val = model.gamma.item()
            print(f"[train-em-ln] step {step+1:5d}/{train_cfg['steps']}   "
                  f"train {avg:.4f}   lr {lr_now:.2e}   "
                  f"grad {grad_norm:.2f}   "
                  f"m[mean {mstats['mean']:.3f} std {mstats['std']:.3f}]   "
                  f"gamma={gamma_val:.3f}   elapsed {elapsed:.0f}s")
            log_f.write(json.dumps({
                "step": step + 1, "train_loss": avg,
                "lr": lr_now, "grad_norm": float(grad_norm),
                "mass_mean": mstats["mean"], "mass_std": mstats["std"],
                "mass_min":  mstats["min"],  "mass_max": mstats["max"],
                "gamma": gamma_val,
            }) + "\n")
            log_f.flush()

        if (step + 1) % train_cfg["eval_interval"] == 0:
            val_loss = evaluate(model, val_ids,
                                train_cfg["eval_iters"],
                                train_cfg["batch_size"],
                                train_cfg["block_size"], rng, device)
            val_ppl = math.exp(val_loss)
            print(f"[train-em-ln] step {step+1:5d}   val_loss={val_loss:.4f}  "
                  f"val_ppl={val_ppl:.2f}")
            loss_history.append((step + 1, avg if n_run == 0 else running / max(n_run, 1),
                                  val_loss))

    log_f.close()
    final_val = evaluate(model, val_ids,
                         train_cfg["eval_iters"],
                         train_cfg["batch_size"],
                         train_cfg["block_size"], rng, device)
    final_ppl = math.exp(final_val)
    final_gamma = model.gamma.item()
    print(f"\n[train-em-ln] DONE  val_loss={final_val:.4f}  "
          f"val_ppl={final_ppl:.2f}  gamma={final_gamma:.4f}")

    ckpt_path = results_dir / f"{tag}_ckpt_latest.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_cfg": asdict(model_cfg),
        "final_val_loss": final_val,
        "final_val_ppl": final_ppl,
        "final_gamma": final_gamma,
        "fixed_gamma": args.fixed_gamma,
        "variant": "sarf_mass_ln",
        "tag": tag,
    }, ckpt_path)
    print(f"[train-em-ln] checkpoint saved to {ckpt_path}")

    # loss curve
    steps_v, train_vs, val_vs = [], [], []
    if loss_history:
        steps_v, train_vs, val_vs = zip(*loss_history)
    fig, ax = plt.subplots(figsize=(7, 4))
    if steps_v:
        ax.plot(steps_v, [math.exp(v) for v in val_vs],
                label="val ppl", color="steelblue")
    ax.set_xlabel("step")
    ax.set_ylabel("perplexity")
    gamma_str = (f"fixed γ={args.fixed_gamma}" if args.fixed_gamma is not None
                 else "free γ")
    ax.set_title(f"SPLM em_ln — {args.mode} ({gamma_str})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / f"{tag}_loss_curve.png", dpi=120)
    plt.close(fig)

    # summary md
    summary_path = results_dir / f"{tag}_summary.md"
    with summary_path.open("w") as f:
        f.write(f"# Training summary — {tag}\n\n")
        f.write(f"- Model: ScalarPotentialLMSARFMassLN (em_ln)\n")
        f.write(f"- mode: {args.mode}\n")
        f.write(f"- fixed_gamma: {args.fixed_gamma}\n")
        f.write(f"- params: {n_params:,}\n")
        f.write(f"- d={model_cfg.d}  L={model_cfg.L}  "
                f"v_hidden={model_cfg.v_hidden}  ln_after_step=True\n")
        f.write(f"- seed: {args.seed}\n")
        f.write(f"\nFinal val loss: {final_val:.6f} (ppl {final_ppl:.2f})\n")
        f.write(f"Final gamma: {final_gamma:.4f}\n")
    print(f"[train-em-ln] summary written to {summary_path}")


if __name__ == "__main__":
    main()
