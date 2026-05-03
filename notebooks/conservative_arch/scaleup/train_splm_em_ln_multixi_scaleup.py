"""
Training loop for the **E11 multi-channel-ξ SPLM scale-up experiment**.

Pre-registered protocol (drafted alongside this trainer):
  docs/SPLM_multichannel_xi_pre-registered_protocol.md

This trainer is a hard-fork of `train_splm_em_ln_scaleup.py` with the only
substantive change being the model class:

    ScalarPotentialLMSARFMassLN          (E9 baseline; single causal mean ξ)
    →
    ScalarPotentialLMSARFMassLNMultiXi   (E11; K causal weighted-EMA ξ-channels)

Configuration is otherwise identical to the locked E9 scale-up (same
corpus, same data caps, same LR schedule, same logfreq mass), so any val-PPL
delta against E9 is attributable to the ξ-multiplexing and the widened V_θ
input layer.

Modes
-----
  --mode smoke     : 300-step pipeline-correctness verification.
  --mode scaleup   : full E11 run (8000 steps).  Same schedule as E9.
  --mode pilot     : 4000-step half-schedule pilot (used for fast comparison
                     against E9 at the same compute fraction).
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

EM_MINIMA_DIR = PARENT_DIR / "energetic_minima"
SARF_MASS_DIR = PARENT_DIR / "sarf_mass_variant"
MULTIXI_DIR = PARENT_DIR / "multixi"
sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(EM_MINIMA_DIR))
sys.path.insert(0, str(SARF_MASS_DIR))
sys.path.insert(0, str(MULTIXI_DIR))

from data_module import get_batch, load_tiny_stories  # noqa: E402
from model_multixi import (  # noqa: E402
    ScalarPotentialLMSARFMassLNMultiXi,
    SPLMSARFMassLNMultiXiConfig,
)

DEFAULT_LOGFREQ_PATH = SCRIPT_DIR / "results" / "logfreq_surprisal_tinystories.npy"


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _parse_alpha_list(s: str) -> list[float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out = []
    for p in parts:
        try:
            v = float(p)
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                f"--xi-alpha-inits got non-float component {p!r}"
            ) from e
        if not (0.0 <= v <= 1.0):
            raise argparse.ArgumentTypeError(
                f"--xi-alpha-inits component {v} not in [0, 1]"
            )
        out.append(v)
    if not out:
        raise argparse.ArgumentTypeError("--xi-alpha-inits is empty")
    return out


def build_config(
    mode: str,
    logfreq_path: str | None,
    fixed_gamma: float | None = None,
    xi_channels: int = 4,
    xi_alpha_inits: list[float] | None = None,
    xi_learnable: bool = True,
    causal_force: bool = True,
    xi_alpha_init_mode: str = "explicit",
    xi_tau_max: float = 100.0,
) -> tuple[SPLMSARFMassLNMultiXiConfig, dict]:
    if xi_alpha_init_mode == "log_spaced":
        # α is computed inside the model from K + tau_max; pass a
        # placeholder of the right length so the dataclass field validates.
        # The model overrides cfg.xi_alpha_inits at __init__ and persists the
        # resolved list back to cfg, so the ckpt records the actual values.
        xi_alpha_inits = [0.0] * xi_channels
    elif xi_alpha_init_mode == "explicit":
        if xi_alpha_inits is None:
            xi_alpha_inits = [0.0, 0.5, 0.9, 0.99]
        if len(xi_alpha_inits) != xi_channels:
            raise ValueError(
                f"len(xi_alpha_inits)={len(xi_alpha_inits)} != "
                f"xi_channels={xi_channels}"
            )
    else:
        raise ValueError(
            f"unknown xi_alpha_init_mode={xi_alpha_init_mode!r} "
            "(expected 'explicit' or 'log_spaced')"
        )

    base_kw = dict(
        vocab_size=50257,
        mass_mode="logfreq",
        logfreq_init_alpha=0.1,
        logfreq_path=logfreq_path,
        ln_after_step=True,
        fixed_gamma=fixed_gamma,
        xi_channels=xi_channels,
        xi_alpha_inits=xi_alpha_inits,
        xi_learnable=xi_learnable,
        causal_force=causal_force,
        xi_alpha_init_mode=xi_alpha_init_mode,
        xi_tau_max=xi_tau_max,
    )
    if mode == "smoke":
        model_cfg = SPLMSARFMassLNMultiXiConfig(
            d=256, max_len=1024, v_hidden=1024, v_depth=3, L=8,
            init_m=1.0, init_gamma=1.0,
            **base_kw,
        )
        train_cfg = dict(
            batch_size=8, block_size=256,
            steps=300, lr=5e-4, weight_decay=0.01,
            warmup_steps=20, grad_clip=1.0,
            eval_interval=100, eval_iters=10,
            log_interval=10,
        )
    elif mode == "scaleup":
        model_cfg = SPLMSARFMassLNMultiXiConfig(
            d=256, max_len=1024, v_hidden=1024, v_depth=3, L=8,
            init_m=1.0, init_gamma=1.0,
            **base_kw,
        )
        train_cfg = dict(
            batch_size=16, block_size=512,
            steps=8000, lr=5e-4, weight_decay=0.01,
            warmup_steps=400, grad_clip=1.0,
            eval_interval=400, eval_iters=40,
            log_interval=50,
        )
    elif mode == "pilot":
        model_cfg = SPLMSARFMassLNMultiXiConfig(
            d=256, max_len=1024, v_hidden=1024, v_depth=3, L=8,
            init_m=1.0, init_gamma=1.0,
            **base_kw,
        )
        train_cfg = dict(
            batch_size=16, block_size=512,
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
def evaluate(
    model: ScalarPotentialLMSARFMassLNMultiXi,
    ids: np.ndarray,
    iters: int,
    batch_size: int,
    block_size: int,
    rng: np.random.Generator,
    device: str,
) -> float:
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
    ap.add_argument(
        "--mode",
        choices=["smoke", "scaleup", "pilot"],
        default="smoke",
    )
    ap.add_argument("--logfreq-path", dest="logfreq_path",
                    default=str(DEFAULT_LOGFREQ_PATH))
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--fixed-gamma", dest="fixed_gamma", type=float, default=0.30,
        help="Fix the damping coefficient at this value. Default 0.30 = E5 winner.",
    )
    ap.add_argument("--max-train-tokens", dest="max_train_tokens",
                    type=int, default=5_000_000)
    ap.add_argument(
        "--tag-suffix", dest="tag_suffix", type=str, default="",
        help="Optional suffix appended to the output tag, e.g. 'seed0'.",
    )
    ap.add_argument(
        "--results-dir", dest="results_dir", type=str, default=None,
    )
    ap.add_argument(
        "--xi-channels", dest="xi_channels", type=int, default=4,
        help="K — number of multi-resolution ξ-channels.",
    )
    ap.add_argument(
        "--xi-alpha-inits", dest="xi_alpha_inits", type=_parse_alpha_list,
        default=None,
        help=(
            "Comma-separated initial decay values α_k ∈ [0, 1] "
            "(length must equal --xi-channels). "
            "Default: 0.0, 0.5, 0.9, 0.99."
        ),
    )
    ap.add_argument(
        "--xi-frozen", dest="xi_learnable", action="store_false",
        default=True,
        help="If passed, freeze the α_k values at their initialisation "
             "(otherwise they are learned).",
    )
    ap.add_argument(
        "--xi-alpha-init-mode", dest="xi_alpha_init_mode",
        choices=["explicit", "log_spaced"], default="explicit",
        help="α-init source. 'explicit' uses --xi-alpha-inits (default, "
             "matches K-EMA pilot). 'log_spaced' computes α_k = "
             "1 - 1/τ_max^(k/(K-1)) from --xi-channels and --xi-tau-max "
             "(R6.h.1 / Fix 2 from §4.2 of "
             "docs/Reducing_Information_Bottleneck_In_Multi-Channel_Xi_SPLM.md).",
    )
    ap.add_argument(
        "--xi-tau-max", dest="xi_tau_max", type=float, default=100.0,
        help="Max effective horizon (in tokens) for log_spaced α-init. "
             "Default 100. Only used when --xi-alpha-init-mode=log_spaced.",
    )
    ap.add_argument(
        "--causal-force", dest="causal_force", default="true",
        choices=["true", "false"],
        help="When 'true' (default) the integrator uses h.detach() before "
             "computing ξ, severing the anti-causal autograd leak documented "
             "in docs/Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md. "
             "Set to 'false' ONLY for forensic reproduction of the pre-fix "
             "buggy integrator.",
    )
    ap.add_argument(
        "--max-steps", dest="max_steps", type=int, default=None,
        help="If set, cap training at this many steps (overrides the mode's "
             "default schedule but keeps the warmup / eval cadence).",
    )
    args = ap.parse_args()
    args.causal_force = (args.causal_force.lower() == "true")

    device = args.device or _pick_device()
    results_dir = (
        Path(args.results_dir).expanduser().resolve()
        if args.results_dir is not None else RESULTS_DIR
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[multixi-splm] device={device}  mode={args.mode}  "
        f"fixed_gamma={args.fixed_gamma!r}  seed={args.seed}  "
        f"results_dir={results_dir}"
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train_ids, val_ids = load_tiny_stories(
        max_train_tokens=args.max_train_tokens,
    )
    print(
        f"[multixi-splm] tokens: train={len(train_ids):,}  "
        f"val={len(val_ids):,}"
    )

    logfreq_path = args.logfreq_path
    if not Path(logfreq_path).exists():
        raise FileNotFoundError(
            f"logfreq surprisal file not found at {logfreq_path}.  "
            "Run scaleup/compute_unigram_frequencies_tinystories.py first."
        )

    model_cfg, train_cfg = build_config(
        args.mode,
        logfreq_path,
        fixed_gamma=args.fixed_gamma,
        xi_channels=args.xi_channels,
        xi_alpha_inits=args.xi_alpha_inits,
        xi_learnable=args.xi_learnable,
        causal_force=args.causal_force,
        xi_alpha_init_mode=args.xi_alpha_init_mode,
        xi_tau_max=args.xi_tau_max,
    )
    if args.max_steps is not None:
        train_cfg["steps"] = args.max_steps
    model = ScalarPotentialLMSARFMassLNMultiXi(model_cfg).to(device)
    n_params = model.num_params()
    alpha_init_str = ",".join(f"{a:.3f}" for a in model.xi_alpha_values())
    print(
        f"[multixi-splm] params: {n_params:,}   d={model_cfg.d}  "
        f"L={model_cfg.L}  v_hidden={model_cfg.v_hidden}  "
        f"max_len={model_cfg.max_len}  ln_after_step={model_cfg.ln_after_step}"
    )
    init_mode_tag = (
        f"log_spaced(τ_max={model_cfg.xi_tau_max:g})"
        if model_cfg.xi_alpha_init_mode == "log_spaced" else "explicit"
    )
    print(
        f"[multixi-splm] xi: K={model_cfg.xi_channels}  "
        f"learnable={model_cfg.xi_learnable}  "
        f"α_init_mode={init_mode_tag}  α_init=[{alpha_init_str}]"
    )
    print(
        f"[multixi-splm] causal_force={model_cfg.causal_force}  "
        f"({'FIXED (post-bug)' if model_cfg.causal_force else 'BUGGY (pre-fix forensic)'})"
        f"  steps={train_cfg['steps']}"
    )

    xb0, _ = get_batch(train_ids, train_cfg["batch_size"],
                       train_cfg["block_size"], rng)
    x0 = torch.from_numpy(xb0).to(device)
    init_mass = model.mass_stats(x0)
    print(
        f"[multixi-splm] init mass: mean={init_mass['mean']:.3f}  "
        f"std={init_mass['std']:.3f}  "
        f"min={init_mass['min']:.3f}  max={init_mass['max']:.3f}"
    )

    optim = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"], betas=(0.9, 0.95),
    )

    tag = f"splm_em_ln_multixi_{args.mode}"
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
        lr_now = lr_schedule(
            step, train_cfg["lr"],
            train_cfg["warmup_steps"], train_cfg["steps"],
        )
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
            alphas = model.xi_alpha_values()
            alpha_str = ",".join(f"{a:.3f}" for a in alphas)
            print(
                f"[multixi-splm] step {step+1:5d}/{train_cfg['steps']}   "
                f"train {avg:.4f}   lr {lr_now:.2e}   "
                f"grad {grad_norm:.2f}   "
                f"m[mean {mstats['mean']:.3f} std {mstats['std']:.3f}]   "
                f"gamma={gamma_val:.3f}   "
                f"α=[{alpha_str}]   elapsed {elapsed:.0f}s"
            )
            log_f.write(json.dumps({
                "step": step + 1, "train_loss": avg,
                "lr": lr_now, "grad_norm": float(grad_norm),
                "mass_mean": mstats["mean"], "mass_std": mstats["std"],
                "mass_min":  mstats["min"],  "mass_max": mstats["max"],
                "gamma": gamma_val,
                "xi_alphas": alphas,
                "elapsed_sec": elapsed,
            }) + "\n")
            log_f.flush()

        if (step + 1) % train_cfg["eval_interval"] == 0:
            val_loss = evaluate(model, val_ids,
                                train_cfg["eval_iters"],
                                train_cfg["batch_size"],
                                train_cfg["block_size"], rng, device)
            val_ppl = math.exp(val_loss)
            print(
                f"[multixi-splm] step {step+1:5d}   "
                f"val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}"
            )
            loss_history.append(
                (step + 1, avg if n_run == 0 else running / max(n_run, 1),
                 val_loss)
            )

    log_f.close()
    final_val = evaluate(model, val_ids,
                         train_cfg["eval_iters"],
                         train_cfg["batch_size"],
                         train_cfg["block_size"], rng, device)
    final_ppl = math.exp(final_val)
    final_gamma = model.gamma.item()
    final_alphas = model.xi_alpha_values()
    total_elapsed = time.time() - t0
    print(
        f"\n[multixi-splm] DONE  val_loss={final_val:.4f}  "
        f"val_ppl={final_ppl:.2f}  gamma={final_gamma:.4f}  "
        f"α_final={final_alphas}  elapsed={total_elapsed:.0f}s"
    )

    ckpt_path = results_dir / f"{tag}_ckpt_latest.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_cfg": asdict(model_cfg),
        "train_cfg": train_cfg,
        "final_val_loss": final_val,
        "final_val_ppl": final_ppl,
        "final_gamma": final_gamma,
        "final_xi_alphas": final_alphas,
        "fixed_gamma": args.fixed_gamma,
        "max_train_tokens": args.max_train_tokens,
        "logfreq_path": str(logfreq_path),
        "variant": "sarf_mass_ln_multixi",
        "experiment": "E11_multixi_scaleup",
        "tag": tag,
        "seed": args.seed,
        "elapsed_sec": total_elapsed,
    }, ckpt_path)
    print(f"[multixi-splm] checkpoint saved to {ckpt_path}")

    steps_v, train_vs, val_vs = [], [], []
    if loss_history:
        steps_v, train_vs, val_vs = zip(*loss_history)
    fig, ax = plt.subplots(figsize=(7, 4))
    if steps_v:
        ax.plot(steps_v, [math.exp(v) for v in val_vs],
                label="val ppl (E11 multi-ξ)", color="darkorange")
    ax.set_xlabel("step")
    ax.set_ylabel("perplexity")
    gamma_str = (f"fixed γ={args.fixed_gamma}" if args.fixed_gamma is not None
                 else "free γ")
    ax.set_title(
        f"SPLM em_ln multi-ξ — {args.mode} ({gamma_str}) — "
        f"K={model_cfg.xi_channels} — seed={args.seed}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / f"{tag}_loss_curve.png", dpi=120)
    plt.close(fig)

    summary_path = results_dir / f"{tag}_summary.md"
    with summary_path.open("w") as f:
        f.write(f"# Training summary — {tag}\n\n")
        f.write("- experiment: E11 multi-channel-ξ SPLM scale-up\n")
        f.write("- model: ScalarPotentialLMSARFMassLNMultiXi (em_ln + multi-ξ)\n")
        f.write(f"- mode: {args.mode}\n")
        f.write(
            f"- corpus: TinyStories "
            f"(cap {args.max_train_tokens:,} train tokens)\n"
        )
        f.write(f"- fixed_gamma: {args.fixed_gamma}\n")
        f.write(f"- params: {n_params:,}\n")
        f.write(
            f"- d={model_cfg.d}  L={model_cfg.L}  "
            f"v_hidden={model_cfg.v_hidden}  max_len={model_cfg.max_len}  "
            f"ln_after_step=True\n"
        )
        f.write(
            f"- xi_channels: {model_cfg.xi_channels}  "
            f"learnable={model_cfg.xi_learnable}  "
            f"α_init={model_cfg.xi_alpha_inits}\n"
        )
        f.write(
            f"- block_size: {train_cfg['block_size']}  "
            f"batch_size: {train_cfg['batch_size']}  "
            f"steps: {train_cfg['steps']}\n"
        )
        f.write(f"- seed: {args.seed}\n")
        f.write(
            f"- elapsed: {total_elapsed:.0f} s "
            f"({total_elapsed/3600:.2f} h)\n"
        )
        f.write(f"\nFinal val loss: {final_val:.6f} (ppl {final_ppl:.2f})\n")
        f.write(f"Final gamma: {final_gamma:.4f}\n")
        f.write(f"Final α_k: {final_alphas}\n")
    print(f"[multixi-splm] summary written to {summary_path}")


if __name__ == "__main__":
    main()
