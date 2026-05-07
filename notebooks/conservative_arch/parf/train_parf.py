"""
PARF-augmented SPLM (Q9c) trainer — Algorithm A reference.

Mirrors notebooks/conservative_arch/helmholtz/train_helmholtz.py one-to-one:
  - Tiny Shakespeare data, GPT-2 BPE tokens
  - logfreq mass mode (matches em_ln leakfree SPLM and Q9d S-blocks)
  - 4000 steps, batch_size=16, block_size=128, AdamW(0.9, 0.95)
  - cosine LR schedule with 200 warmup steps
  - val perplexity logged every eval_interval steps
  - causal_force=True (leak-fix invariant always on)
  - startup causal-violation probe (gates training before any optimiser
    step; aborts on leak)

PARF-specific
-------------
The model takes a `--v-phi-kind {structural, mlp}` instead of a schedule;
otherwise the trainer is identical.  Algorithm A means: pure NTP cross-
entropy backprop through both V_θ and V_φ.  No Gumbel-softmax sparsity
yet — that's a Stage 1.5 add-on per the design doc; here we just
establish the baseline numbers (PARF vs SPLM em_ln vs all-attn vs Q9d
vs VA).

Outputs (under parf/results/):
  - parf_<v_phi_kind>_<mode>_seed{seed}_training_log.jsonl
  - parf_<v_phi_kind>_<mode>_seed{seed}_ckpt_latest.pt
  - parf_<v_phi_kind>_<mode>_seed{seed}_loss_curve.png
  - parf_<v_phi_kind>_<mode>_seed{seed}_summary.md
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
from model_parf import PARFConfig, PARFLM  # noqa: E402
# `model_parf` re-inserts PARENT_DIR at sys.path[0] when it loads,
# which would shadow this directory's local `causal_probe_parf` if a
# similarly named module exists upstream.  Re-assert SCRIPT_DIR so the
# local import wins.
sys.path.insert(0, str(SCRIPT_DIR))
from causal_probe_parf import assert_causal  # noqa: E402


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_config(
    mode: str,
    logfreq_path: str,
    v_phi_kind: str,
    fixed_gamma_arg,
    v_hidden_arg: int | None = None,
    v_phi_hidden_arg: int | None = None,
    use_grad_checkpoint: bool = False,
) -> Tuple[PARFConfig, dict, str]:
    """Return (model_cfg, train_cfg, tag).

    `v_hidden_arg`  : override the mode default for V_θ's hidden width
                       (for parity with Q9d's H1.5 narrow-V FLOP arm).
    `v_phi_hidden_arg`: override the mode default for V_φ's hidden width.
                        For 'structural' it sets `v_phi_phi_hidden` and
                        `v_phi_theta_hidden` jointly; for 'mlp' it sets
                        `v_phi_mlp_hidden`.
    """

    base_kw = dict(
        vocab_size=50257,
        mass_mode="logfreq",
        logfreq_init_alpha=0.1,
        logfreq_path=logfreq_path,
        v_phi_kind=v_phi_kind,
        causal_force=True,
        ln_after_step=True,
        fixed_gamma=fixed_gamma_arg,
        use_grad_checkpoint=use_grad_checkpoint,
    )

    if mode == "smoke":
        default_v_hidden = 128
        base_kw.update(d=64, max_len=128, L=4,
                       v_hidden=default_v_hidden, v_depth=2,
                       v_phi_d_type=8, v_phi_d_angle=4,
                       v_phi_phi_hidden=16, v_phi_theta_hidden=16,
                       v_phi_mlp_hidden=32)
        train_cfg = dict(
            batch_size=8, block_size=64,
            steps=300, lr=1e-3, weight_decay=0.0,
            warmup_steps=20, grad_clip=1.0,
            eval_interval=50, eval_iters=20,
            log_interval=10,
        )
    elif mode == "shakespeare":
        # Match the Helmholtz Q9d AAAASSSS vh=128 cell shape so PARF
        # numbers are directly comparable to that quality lead.
        default_v_hidden = 128
        base_kw.update(d=128, max_len=256, L=8,
                       v_hidden=default_v_hidden, v_depth=3,
                       v_phi_d_type=16, v_phi_d_angle=8,
                       v_phi_phi_hidden=32, v_phi_theta_hidden=32,
                       v_phi_mlp_hidden=64)
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

    if v_phi_hidden_arg is not None:
        if v_phi_kind == "mlp":
            base_kw["v_phi_mlp_hidden"] = v_phi_hidden_arg
        else:
            base_kw["v_phi_phi_hidden"] = v_phi_hidden_arg
            base_kw["v_phi_theta_hidden"] = v_phi_hidden_arg
        vph_tag = f"_vphi{v_phi_hidden_arg}"
    else:
        vph_tag = ""

    gc_tag = "_gc" if use_grad_checkpoint else ""

    cfg = PARFConfig(**base_kw)
    fg_tag = "" if fixed_gamma_arg is None else f"_g{fixed_gamma_arg:.3f}"
    tag = f"parf_{v_phi_kind}{vh_tag}{vph_tag}{gc_tag}{fg_tag}"
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
        # The model uses torch.autograd.grad inside forward (V_θ + V_φ),
        # which requires enable_grad even at eval time.
        with torch.enable_grad():
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "shakespeare"],
                    default="shakespeare")
    ap.add_argument("--v-phi-kind", choices=["structural", "mlp"],
                    default="structural", dest="v_phi_kind",
                    help="Inner shape of V_phi: 'structural' is the "
                         "§5.1-faithful pair potential (default); 'mlp' "
                         "is the unstructured MLP ablation.")
    ap.add_argument("--fixed-gamma", type=float, default=None,
                    dest="fixed_gamma",
                    help="If set, use this fixed gamma for the integrator. "
                         "If None (default), gamma is freely learned.")
    ap.add_argument("--v-hidden", type=int, default=None,
                    dest="v_hidden",
                    help="Override V_theta hidden width.  Mode defaults: "
                         "smoke=128, shakespeare=128.  When set, the "
                         "output tag gains a `_vh{N}` suffix.")
    ap.add_argument("--v-phi-hidden", type=int, default=None,
                    dest="v_phi_hidden",
                    help="Override V_phi hidden width (phi_hidden + "
                         "theta_hidden for 'structural'; mlp_hidden for "
                         "'mlp').  When set, the output tag gains a "
                         "`_vphi{N}` suffix.")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    dest="grad_checkpoint",
                    help="Gradient-checkpoint the V_phi pair sum.  Trades "
                         "~15-25%% wall-clock for ~50%% lower per-layer "
                         "activation memory.  Required for the MLP V_phi "
                         "variant at B=16 (else MPS OOM).  When set, the "
                         "output tag gains a `_gc` suffix.")
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
             "`parf/causal_probe_parf.py` for the test definition).",
    )
    args = ap.parse_args()

    device = args.device or _pick_device()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train_ids, val_ids = load_tiny_shakespeare()

    cfg, train_cfg, tag = build_config(
        args.mode, args.logfreq_path,
        args.v_phi_kind, args.fixed_gamma,
        v_hidden_arg=args.v_hidden,
        v_phi_hidden_arg=args.v_phi_hidden,
        use_grad_checkpoint=args.grad_checkpoint,
    )
    full_tag = f"{tag}_{args.mode}_seed{args.seed}"
    print(f"[parf-train] device={device}  tag={full_tag}")
    print(f"[parf-train] tokens: train={len(train_ids):,}  "
          f"val={len(val_ids):,}")
    print(f"[parf-train] arch: V_phi={cfg.v_phi_kind!r}  L={cfg.L}  "
          f"d={cfg.d}, v_hidden={cfg.v_hidden}, v_depth={cfg.v_depth}, "
          f"v_phi_d_type={cfg.v_phi_d_type}, "
          f"v_phi_d_angle={cfg.v_phi_d_angle}, "
          f"fixed_gamma={cfg.fixed_gamma}, "
          f"causal_force={cfg.causal_force}, "
          f"ln_after_step={cfg.ln_after_step}, "
          f"use_grad_checkpoint={cfg.use_grad_checkpoint}")

    model = PARFLM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_v_phi = sum(p.numel() for p in model.V_phi.parameters())
    n_v_theta = sum(p.numel() for p in model.V_theta.parameters())
    print(f"[parf-train] params: {n_params:,}  "
          f"(target ~8.0 M, delta {(n_params - 8_000_000) / 1e6:+.3f} M)  "
          f"V_theta={n_v_theta:,}  V_phi={n_v_phi:,}")

    # Causal-violation probe.  Runs perturbation + gradient-Jacobian on
    # the actual on-device model BEFORE any optimisation step.  Aborts
    # training on leak — cheaper to fail fast than to discover a
    # causality bug after a 6-hour MPS run.  Tolerance: strict 1e-6.
    if not args.skip_causal_check:
        print(f"[parf-train] running causal-violation probe...")
        try:
            assert_causal(
                model, vocab_size=cfg.vocab_size,
                T=32, t_pert=20, seed=args.seed,
            )
            print(f"[parf-train] causal probe PASSED "
                  f"(perturbation + gradient-Jacobian, both V_phi paths, "
                  f"both modes < 1e-6)")
        except RuntimeError as exc:
            print(f"[parf-train] causal probe FAILED — aborting before any "
                  f"compute is wasted.")
            print(f"[parf-train] {exc}")
            raise SystemExit(2)
    else:
        print(f"[parf-train] WARNING: --skip-causal-check is set; "
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
            print(f"[parf-train] step {step+1:5d}/{train_cfg['steps']}   "
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
            print(f"[parf-train] >>> eval @ {step+1}: "
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
    print(f"[parf-train] done in {elapsed:.0f}s")

    ckpt_path = RESULTS_DIR / f"{full_tag}_ckpt_latest.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_cfg": asdict(cfg),
        "train_cfg": train_cfg,
        "loss_history": loss_history,
        "final_gamma": float(model.gamma.item()),
        "variant": "parf_q9c",
        "v_phi_kind": cfg.v_phi_kind,
        "tag": full_tag,
        "n_params": n_params,
        "n_v_theta_params": n_v_theta,
        "n_v_phi_params": n_v_phi,
    }, ckpt_path)
    print(f"[parf-train] saved checkpoint -> {ckpt_path}")

    if loss_history:
        steps_e = [e[0] for e in loss_history]
        tr_e    = [e[1] for e in loss_history]
        va_e    = [e[2] for e in loss_history]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(steps_e, tr_e, marker="o", label="train (eval)")
        ax.plot(steps_e, va_e, marker="s", label="val")
        ax.set_xlabel("step")
        ax.set_ylabel("cross-entropy")
        ax.set_title(f"PARF Q9c V_phi={cfg.v_phi_kind}  "
                     f"(L={cfg.L}, {args.mode}, seed {args.seed})")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        png_path = RESULTS_DIR / f"{full_tag}_loss_curve.png"
        fig.savefig(png_path, dpi=130)
        plt.close(fig)
        print(f"[parf-train] saved loss curve -> {png_path}")

    summary_path = RESULTS_DIR / f"{full_tag}_summary.md"
    with summary_path.open("w") as f:
        f.write(f"# PARF Q9c -- {args.mode} training summary\n\n")
        f.write(f"- Tag: `{full_tag}`\n")
        f.write(f"- V_phi kind: `{cfg.v_phi_kind}` "
                f"(structural = §5.1-faithful; mlp = unstructured ablation)\n")
        f.write(f"- Architecture: PARF-augmented SPLM (Q9c) -- single shared "
                f"V_theta (4-layer MLP) plus single shared V_phi pair "
                f"interaction with causal reduction (past tokens = fixed "
                f"external sources via .detach()), velocity-Verlet damped "
                f"Euler-Lagrange step at every layer, Algorithm A "
                f"(NTP-only backprop, no Gumbel sparsity).\n")
        f.write(f"- L (depth): {cfg.L}\n")
        f.write(f"- Device: `{device}`\n")
        f.write(f"- Parameters: **{n_params:,}** "
                f"(target ~8.0 M, delta "
                f"{(n_params - 8_000_000) / 1e6:+.3f} M; "
                f"V_theta={n_v_theta:,}, V_phi={n_v_phi:,})\n")
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
    print(f"[parf-train] saved summary -> {summary_path}")


if __name__ == "__main__":
    main()
