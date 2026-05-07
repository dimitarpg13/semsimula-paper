"""
Helmholtz hybrid (Q9d) trainer.

Mirrors notebooks/conservative_arch/hybrid/train_splm_hybrid.py one-to-one:
  - Tiny Shakespeare data, GPT-2 BPE tokens
  - logfreq mass mode (matches leak-free SPLM em_ln cells and Variant A HSPLM)
  - 4000 steps, batch_size=16, block_size=128, AdamW(0.9, 0.95)
  - cosine LR schedule with 200 warmup steps
  - val perplexity logged every eval_interval steps
  - causal_force=True (leak-fix invariant always on)

The only architectural difference from the Variant A trainer is that
the model takes a `--schedule` string (e.g. "AAAASSSS", "SASASASA",
"SAAAAAAS") instead of `--n-attn k --n-splm m`.  This lets us cover
the full design space of doc section 6 with a single trainer.

Outputs (under helmholtz/results/):
  - helm_<schedule>_<mode>_seed{seed}_training_log.jsonl
  - helm_<schedule>_<mode>_seed{seed}_ckpt_latest.pt
  - helm_<schedule>_<mode>_seed{seed}_loss_curve.png
  - helm_<schedule>_<mode>_seed{seed}_summary.md
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
from model_helmholtz import (  # noqa: E402
    HelmholtzConfig,
    HelmholtzLM,
    parse_schedule,
    schedule_counts,
    make_schedule,
)
# `model_helmholtz` re-inserts PARENT_DIR at sys.path[0] when it loads,
# which would shadow this directory's local `causal_probe` with the
# repo-wide one in `notebooks/conservative_arch/causal_probe.py`.
# Re-assert SCRIPT_DIR at the front of the path so the local import wins.
sys.path.insert(0, str(SCRIPT_DIR))
from causal_probe import assert_causal  # noqa: E402


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_schedule(arg: str, L: int = 8, k: int = 1, LA: int = 1) -> str:
    """Accept either an explicit schedule string or a registry name.

    If `arg` matches one of the registry names from
    `model_helmholtz.make_schedule`, expand it; otherwise return `arg`
    verbatim after validating it's a parseable schedule.
    """
    name = arg.lower()
    registry = {"all_s", "all_a", "sandwich", "inverse_sandwich",
                "interleaved", "top_a", "bottom_a"}
    if name in registry:
        return make_schedule(name, L=L, k=k, LA=LA)
    parse_schedule(arg)   # validates
    return arg.upper()


def build_config(
    mode: str,
    logfreq_path: str,
    schedule_str: str,
    fixed_gamma_arg,
    v_hidden_arg: int | None = None,
) -> Tuple[HelmholtzConfig, dict, str]:
    """Return (model_cfg, train_cfg, tag).

    `v_hidden_arg`: override the mode default for `V_theta`'s hidden width.
    Used by H1.5 to sweep `v_hidden ∈ {128, 256}` and clear the FLOP arm
    at T=1024 (see companion_notes/Helmholtz-HSPLM_Path_Forward_and_Experiments.md
    §7).  When supplied, the tag gains a `_vhN` suffix so H1.5 cells do
    not overwrite the H1 cells trained at the mode default of 512.
    """

    base_kw = dict(
        vocab_size=50257,
        mass_mode="logfreq",
        logfreq_init_alpha=0.1,
        logfreq_path=logfreq_path,
        schedule=schedule_str,
        causal_force=True,
        ln_after_s_step=True,
        fixed_gamma=fixed_gamma_arg,
    )

    if mode == "smoke":
        default_v_hidden = 128
        base_kw.update(d=64, max_len=128, v_hidden=default_v_hidden, v_depth=2)
        train_cfg = dict(
            batch_size=8, block_size=64,
            steps=300, lr=1e-3, weight_decay=0.0,
            warmup_steps=20, grad_clip=1.0,
            eval_interval=50, eval_iters=20,
            log_interval=10,
        )
    elif mode == "shakespeare":
        default_v_hidden = 512
        base_kw.update(d=128, max_len=256, v_hidden=default_v_hidden, v_depth=3)
        train_cfg = dict(
            batch_size=16, block_size=128,
            steps=4000, lr=5e-4, weight_decay=0.01,
            warmup_steps=200, grad_clip=1.0,
            eval_interval=200, eval_iters=40,
            log_interval=50,
        )
    else:
        raise ValueError(f"unknown mode: {mode}")

    if v_hidden_arg is not None:
        base_kw["v_hidden"] = v_hidden_arg
        vh_tag = f"_vh{v_hidden_arg}"
    else:
        vh_tag = ""

    cfg = HelmholtzConfig(**base_kw)
    fg_tag = "" if fixed_gamma_arg is None else f"_g{fixed_gamma_arg:.3f}"
    tag = f"helm_{schedule_str}{vh_tag}{fg_tag}"
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
    ap.add_argument("--schedule", required=True,
                    help="Either an explicit schedule string (e.g. "
                         "'AAAASSSS', 'SASASASA', 'SAAAAAAS') or a "
                         "registry name expanded by --L / --k / --LA: "
                         "{all_s, all_a, sandwich, inverse_sandwich, "
                         "interleaved, top_a, bottom_a}.")
    ap.add_argument("--L", type=int, default=8,
                    help="Stack depth (used when --schedule is a registry "
                         "name; defaults to 8 to match the matched_baseline "
                         "n_layer).")
    ap.add_argument("--k", type=int, default=1,
                    help="Sandwich half-width (only used by --schedule "
                         "sandwich / inverse_sandwich).")
    ap.add_argument("--LA", type=int, default=1,
                    help="A-block count (only used by --schedule "
                         "top_a / bottom_a).")
    ap.add_argument("--fixed-gamma", type=float, default=None,
                    help="If set, use this fixed gamma for the SPLM phase. "
                         "If None (default), gamma is freely learned.")
    ap.add_argument("--v-hidden", type=int, default=None,
                    dest="v_hidden",
                    help="Override V_theta hidden width.  Mode defaults: "
                         "smoke=128, shakespeare=512.  Use 128 or 256 for "
                         "the H1.5 narrow-V FLOP-arm ablation.  When set, "
                         "the output tag gains a `_vh{N}` suffix so H1.5 "
                         "cells do not overwrite H1.")
    ap.add_argument("--logfreq-path",
                    default=str(PARENT_DIR / "sarf_mass_variant" /
                                "results" / "logfreq_surprisal.npy"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--skip-causal-check", action="store_true",
        help="Skip the startup causal-violation probe.  Default behaviour "
             "is to abort training BEFORE any optimisation step if the "
             "model leaks future-position information (see "
             "`helmholtz/causal_probe.py` for the test definition and "
             "`companion_notes/Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md` for "
             "the historical bug this guards against).",
    )
    args = ap.parse_args()

    schedule_str = resolve_schedule(
        args.schedule, L=args.L, k=args.k, LA=args.LA,
    )
    sigma = parse_schedule(schedule_str)
    nS, nA = schedule_counts(sigma)
    if nS < 1 and nA < 1:
        raise SystemExit("schedule must contain at least one block.")

    device = args.device or _pick_device()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train_ids, val_ids = load_tiny_shakespeare()

    cfg, train_cfg, tag = build_config(
        args.mode, args.logfreq_path,
        schedule_str, args.fixed_gamma,
        v_hidden_arg=args.v_hidden,
    )
    full_tag = f"{tag}_{args.mode}_seed{args.seed}"
    print(f"[helm-train] device={device}  tag={full_tag}")
    print(f"[helm-train] tokens: train={len(train_ids):,}  "
          f"val={len(val_ids):,}")
    print(f"[helm-train] arch: schedule={schedule_str!r}  "
          f"(n_S={nS}, n_A={nA}, L={len(sigma)})  "
          f"d={cfg.d}, v_hidden={cfg.v_hidden}, v_depth={cfg.v_depth}, "
          f"n_head={cfg.n_head}, fixed_gamma={cfg.fixed_gamma}, "
          f"causal_force={cfg.causal_force}, "
          f"ln_after_s_step={cfg.ln_after_s_step}")

    model = HelmholtzLM(cfg).to(device)
    n_params = model.num_params()
    print(f"[helm-train] params: {n_params:,}  "
          f"(target ~8.0 M, delta {(n_params - 8_000_000) / 1e6:+.3f} M)")

    # Causal-violation probe.  Runs perturbation + gradient-Jacobian
    # tests on the actual on-device model BEFORE any optimisation step.
    # Aborts training (RuntimeError) if either test detects a leak —
    # cheaper to fail fast than to discover a causality bug after a
    # 30-minute MPS run.  Tolerance is the strict 1e-6 used by the
    # repo-wide probe in notebooks/conservative_arch/causal_probe.py.
    if not args.skip_causal_check:
        print(f"[helm-train] running causal-violation probe...")
        try:
            assert_causal(
                model, vocab_size=cfg.vocab_size,
                T=32, t_pert=20, seed=args.seed,
            )
            print(f"[helm-train] causal probe PASSED "
                  f"(perturbation + gradient-Jacobian, both modes < 1e-6)")
        except RuntimeError as exc:
            print(f"[helm-train] causal probe FAILED — aborting before any "
                  f"compute is wasted.")
            print(f"[helm-train] {exc}")
            raise SystemExit(2)
    else:
        print(f"[helm-train] WARNING: --skip-causal-check is set; "
              f"the model will train without the startup causality guard.")

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
            print(f"[helm-train] step {step+1:5d}/{train_cfg['steps']}   "
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
            print(f"[helm-train] >>> eval @ {step+1}: "
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
    print(f"[helm-train] done in {elapsed:.0f}s")

    ckpt_path = RESULTS_DIR / f"{full_tag}_ckpt_latest.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_cfg": asdict(cfg),
        "train_cfg": train_cfg,
        "loss_history": loss_history,
        "final_gamma": float(model.gamma.item()),
        "variant": "helmholtz_q9d",
        "schedule": schedule_str,
        "n_s_blocks": nS,
        "n_a_blocks": nA,
        "tag": full_tag,
        "n_params": n_params,
    }, ckpt_path)
    print(f"[helm-train] saved checkpoint -> {ckpt_path}")

    if loss_history:
        steps_e = [e[0] for e in loss_history]
        tr_e    = [e[1] for e in loss_history]
        va_e    = [e[2] for e in loss_history]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(steps_e, tr_e, marker="o", label="train (eval)")
        ax.plot(steps_e, va_e, marker="s", label="val")
        ax.set_xlabel("step")
        ax.set_ylabel("cross-entropy")
        ax.set_title(f"helmholtz {schedule_str}  "
                     f"(n_S={nS}, n_A={nA}, {args.mode}, seed {args.seed})")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        png_path = RESULTS_DIR / f"{full_tag}_loss_curve.png"
        fig.savefig(png_path, dpi=130)
        plt.close(fig)
        print(f"[helm-train] saved loss curve -> {png_path}")

    summary_path = RESULTS_DIR / f"{full_tag}_summary.md"
    with summary_path.open("w") as f:
        f.write(f"# Helmholtz hybrid (Q9d) -- {args.mode} training summary\n\n")
        f.write(f"- Tag: `{full_tag}`\n")
        f.write(f"- Schedule: `{schedule_str}` "
                f"(n_S={nS}, n_A={nA}, L={len(sigma)})\n")
        f.write(f"- Architecture: layer-type Helmholtz hybrid -- single shared "
                f"V_theta on every S-block, per-layer attention on every "
                f"A-block, velocity-Verlet damped Euler-Lagrange S-step.\n")
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
    print(f"[helm-train] saved summary -> {summary_path}")


if __name__ == "__main__":
    main()
