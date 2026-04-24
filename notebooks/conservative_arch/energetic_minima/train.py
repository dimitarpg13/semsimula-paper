"""
Unified trainer for the three energetic-minima variants:

  --variant ln : LayerNorm-after-step (option C, model_ln.py).
  --variant sg : Scale-gauge -- uses the unchanged sarf_mass model and
                 adds a loss-side regulariser
                      lambda_v0 * mean_over_tokens V_theta(xi_0, h_0)^2
                 anchoring V's absolute scale at the *input* embedding
                 (so it does not conflict with the LM objective at h_L).
  --variant gm : Gaussian-mixture head (option B, model_gm.py).

All three variants are trained with the same optimizer, schedule, and
mode hyper-parameters as sarf_mass_variant/train_splm_sarf_mass.py
(logfreq mass, Tiny Shakespeare, 4000 steps), so any difference in
val perplexity is attributable to the variant alone.

Example:
  python3 train.py --variant ln --mode shakespeare
  python3 train.py --variant sg --mode shakespeare --lambda-v0 1e-3
  python3 train.py --variant gm --mode shakespeare --gm-K 64
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple

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

sys.path.insert(0, str(PARENT_DIR))
from data_module import get_batch, load_tiny_shakespeare  # noqa: E402
from sarf_mass_variant.model_sarf_mass import (  # noqa: E402
    ScalarPotentialLMSARFMass,
    SPLMSARFMassConfig,
)

sys.path.insert(0, str(SCRIPT_DIR))
from model_ln import (  # noqa: E402
    ScalarPotentialLMSARFMassLN,
    SPLMSARFMassLNConfig,
)
from model_gm import (  # noqa: E402
    ScalarPotentialLMSARFMassGM,
    SPLMSARFMassGMConfig,
)


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_config(variant: str, mode: str, logfreq_path: str,
                 lambda_v0: float, gm_K: int) -> Tuple[object, dict, str]:
    """Return (model_cfg, train_cfg, variant_tag)."""

    base_kw = dict(
        vocab_size=50257,
        mass_mode="logfreq",
        logfreq_init_alpha=0.1,
        logfreq_path=logfreq_path,
    )
    if mode == "smoke":
        base_kw.update(d=64, max_len=128, v_hidden=128, v_depth=2, L=4)
        train_cfg = dict(
            batch_size=8, block_size=64,
            steps=300, lr=1e-3, weight_decay=0.0,
            warmup_steps=20, grad_clip=1.0,
            eval_interval=50, eval_iters=20,
            log_interval=10,
            lambda_v0=lambda_v0,
        )
    elif mode == "shakespeare":
        base_kw.update(d=128, max_len=256, v_hidden=512, v_depth=3, L=8)
        train_cfg = dict(
            batch_size=16, block_size=128,
            steps=4000, lr=5e-4, weight_decay=0.01,
            warmup_steps=200, grad_clip=1.0,
            eval_interval=200, eval_iters=40,
            log_interval=50,
            lambda_v0=lambda_v0,
        )
    else:
        raise ValueError(f"unknown mode: {mode}")

    if variant == "ln":
        cfg = SPLMSARFMassLNConfig(**base_kw, ln_after_step=True)
        tag = "em_ln"
    elif variant == "sg":
        cfg = SPLMSARFMassConfig(**base_kw)
        tag = f"em_sg_lam{lambda_v0:.0e}"
    elif variant == "gm":
        cfg = SPLMSARFMassGMConfig(**base_kw, gm_K=gm_K)
        tag = f"em_gm_K{gm_K}"
    else:
        raise ValueError(f"unknown variant: {variant}")

    return cfg, train_cfg, tag


def build_model(variant: str, cfg) -> nn.Module:
    if variant == "ln":
        return ScalarPotentialLMSARFMassLN(cfg)
    if variant == "sg":
        return ScalarPotentialLMSARFMass(cfg)
    if variant == "gm":
        return ScalarPotentialLMSARFMassGM(cfg)
    raise ValueError(f"unknown variant: {variant}")


def lr_schedule(step: int, lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def compute_v0_penalty(model, x: torch.Tensor) -> torch.Tensor:
    """Evaluate V_theta(xi_0, h_0) at the input embedding, averaged.

    Small quantity: V(xi_0, h_0) for every (batch, token).  We return
    the mean of V**2 so gradient steps toward V(xi_0, h_0) = 0 for
    every token.  This anchors the absolute scale of V_theta without
    interfering with the cross-entropy objective at h_L.
    """
    emb = model._embed(x)
    from sarf_mass_variant.model_sarf_mass import causal_cumulative_mean
    xi0 = causal_cumulative_mean(emb)
    V0 = model.V_theta(xi0, emb)
    return V0.pow(2).mean()


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
    ap.add_argument("--variant", choices=["ln", "sg", "gm"], required=True)
    ap.add_argument("--mode", choices=["smoke", "shakespeare"],
                    default="smoke")
    ap.add_argument("--logfreq-path",
                    default=str(PARENT_DIR / "sarf_mass_variant" /
                                "results" / "logfreq_surprisal.npy"))
    ap.add_argument("--lambda-v0", type=float, default=0.0,
                    help="Weight of V_theta(xi_0, h_0)^2 scale anchor. "
                         "Only effective for --variant sg (or any variant).")
    ap.add_argument("--gm-K", type=int, default=64,
                    help="Number of Gaussian wells in --variant gm.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = args.device or _pick_device()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train_ids, val_ids = load_tiny_shakespeare()

    cfg, train_cfg, tag = build_config(
        args.variant, args.mode, args.logfreq_path,
        lambda_v0=args.lambda_v0, gm_K=args.gm_K,
    )
    print(f"[em-train] device={device}  variant={args.variant}  "
          f"mode={args.mode}  tag={tag}")
    print(f"[em-train] tokens: train={len(train_ids):,}  "
          f"val={len(val_ids):,}")

    model = build_model(args.variant, cfg).to(device)
    n_params = model.num_params()
    print(f"[em-train] params: {n_params:,}  "
          f"cfg: d={cfg.d}, L={cfg.L}, v_hidden={cfg.v_hidden}")

    optim = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"], betas=(0.9, 0.95),
    )

    log_path = RESULTS_DIR / f"{tag}_{args.mode}_training_log.jsonl"
    log_f = log_path.open("w")
    loss_history: list[tuple[int, float, float]] = []

    t0 = time.time()
    model.train()
    running = 0.0
    running_v0 = 0.0
    n_run = 0
    lam_v0 = train_cfg["lambda_v0"]

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
        if lam_v0 > 0.0:
            v0_pen = compute_v0_penalty(model, x)
            total_loss = loss + lam_v0 * v0_pen
        else:
            v0_pen = torch.zeros((), device=device)
            total_loss = loss

        optim.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(),
                                             train_cfg["grad_clip"])
        optim.step()

        running += loss.item()
        running_v0 += float(v0_pen.item())
        n_run += 1

        if (step + 1) % train_cfg["log_interval"] == 0:
            avg = running / n_run
            avg_v0 = running_v0 / n_run
            running, running_v0, n_run = 0.0, 0.0, 0
            elapsed = time.time() - t0
            print(f"[em-train] step {step+1:5d}/{train_cfg['steps']}   "
                  f"train {avg:.4f}   lr {lr_now:.2e}   "
                  f"grad {grad_norm:.2f}   v0_sq {avg_v0:.3e}   "
                  f"gamma={model.gamma.item():.3f}   elapsed {elapsed:.0f}s")
            log_f.write(json.dumps({
                "step": step + 1, "train_loss": avg,
                "lr": lr_now, "grad_norm": float(grad_norm),
                "v0_penalty_sq": avg_v0,
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
            print(f"[em-train] >>> eval @ {step+1}: "
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
    print(f"[em-train] done in {elapsed:.0f}s")

    ckpt_path = RESULTS_DIR / f"{tag}_{args.mode}_ckpt_latest.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_cfg": asdict(cfg),
        "train_cfg": train_cfg,
        "loss_history": loss_history,
        "final_gamma": float(model.gamma.item()),
        "variant": {"ln": "sarf_mass_ln",
                    "sg": "sarf_mass",
                    "gm": "sarf_mass_gm"}[args.variant],
        "em_variant": args.variant,
        "tag": tag,
    }, ckpt_path)
    print(f"[em-train] saved checkpoint -> {ckpt_path}")

    if loss_history:
        steps_e = [e[0] for e in loss_history]
        tr_e    = [e[1] for e in loss_history]
        va_e    = [e[2] for e in loss_history]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(steps_e, tr_e, marker="o", label="train (eval)")
        ax.plot(steps_e, va_e, marker="s", label="val")
        ax.set_xlabel("step")
        ax.set_ylabel("cross-entropy")
        ax.set_title(f"energetic-minima variant `{args.variant}` -- {args.mode}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        png_path = RESULTS_DIR / f"{tag}_{args.mode}_loss_curve.png"
        fig.savefig(png_path, dpi=130)
        plt.close(fig)
        print(f"[em-train] saved loss curve -> {png_path}")

    summary_path = RESULTS_DIR / f"{tag}_{args.mode}_summary.md"
    with summary_path.open("w") as f:
        f.write(f"# energetic-minima variant `{args.variant}` "
                f"-- {args.mode} training summary\n\n")
        f.write(f"- Tag: `{tag}`\n")
        f.write(f"- Device: `{device}`\n")
        f.write(f"- Parameters: **{n_params:,}**\n")
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
        if args.variant == "gm":
            f.write(f"- GM wells: K={cfg.gm_K}, "
                    f"mean amplitude={model.V_theta.amplitudes.mean().item():.4f}, "
                    f"mean kappa={model.V_theta.kappas.mean().item():.4f}\n")
        if lam_v0 > 0:
            f.write(f"- Scale-gauge lambda_v0: {lam_v0}\n")
        f.write(f"- Loss curve: `{tag}_{args.mode}_loss_curve.png`\n")
        f.write(f"- Checkpoint: `{tag}_{args.mode}_ckpt_latest.pt`\n")
    print(f"[em-train] saved summary -> {summary_path}")


if __name__ == "__main__":
    main()
