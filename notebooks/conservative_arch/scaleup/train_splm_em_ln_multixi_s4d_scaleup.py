"""
Training loop for the **E13 S4D multi-channel-ξ SPLM scale-up experiment**.

Architectural change vs E12 (HiPPO-LegT)
----------------------------------------
This trainer is a hard-fork of `train_splm_em_ln_multixi_hippo_scaleup.py`
with a single substantive change: the model class

    ScalarPotentialLMSARFMassLNMultiHiPPO     (E12; structured-A LegT, fixed basis)
    →
    ScalarPotentialLMSARFMassLNMultiS4D       (E13; diagonal-complex-A, learned basis)

The K "ξ-channels" are now produced by a **diagonal complex-valued
state-space ODE** with K trainable eigenvalues {λ_k} (Re < 0 enforced
structurally) and K trainable input gains {B_k}. The basis itself is
gradient-discovered. See `Reducing_Information_Bottleneck_*.md` §11
for the theoretical motivation and the experimental hierarchy R6.i / j / k.

Configuration is otherwise identical to the locked E11/E12 scale-up
(same corpus, same data caps, same LR schedule, same logfreq mass), so
any val-PPL delta against E12 is attributable to the basis-learning step.

Causal-leak compatibility
-------------------------
Identical semantics to `train_splm_em_ln_multixi_hippo_scaleup.py`:
`--causal-force true` (default) detaches `h` before the S4D state update,
severing the autograd path from `c_t` back to `h_t`. The S4D recurrence
is strictly causal by construction. See
`docs/Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md`.

Modes
-----
  --mode smoke   : 300-step pipeline-correctness verification.
  --mode pilot   : 4000-step half-schedule pilot for direct comparison
                   against the E12 HiPPO-LegT pilot (multihippo_pilot_fixed
                   = R6.a, val_ppl 19.82) and the E12 learnable-Δ pilot
                   (multihippo_pilot_learndt = R6.e, in flight).
  --mode scaleup : full 8000-step run for the head-to-head against E11/E12.
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
from model_multixi_s4d import (  # noqa: E402
    ScalarPotentialLMSARFMassLNMultiS4D,
    SPLMSARFMassLNMultiS4DConfig,
)

DEFAULT_LOGFREQ_PATH = SCRIPT_DIR / "results" / "logfreq_surprisal_tinystories.npy"


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
    xi_channels: int = 4,
    xi_theta: float = 200.0,
    xi_eigval_init: str = "legt",
    xi_learnable_dt: bool = True,
    xi_learnable_B: bool = True,
    causal_force: bool = True,
) -> tuple[SPLMSARFMassLNMultiS4DConfig, dict]:
    base_kw = dict(
        vocab_size=50257,
        mass_mode="logfreq",
        logfreq_init_alpha=0.1,
        logfreq_path=logfreq_path,
        ln_after_step=True,
        fixed_gamma=fixed_gamma,
        xi_channels=xi_channels,
        xi_theta=xi_theta,
        xi_eigval_init=xi_eigval_init,
        xi_learnable_dt=xi_learnable_dt,
        xi_learnable_B=xi_learnable_B,
        causal_force=causal_force,
    )
    if mode == "smoke":
        model_cfg = SPLMSARFMassLNMultiS4DConfig(
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
        model_cfg = SPLMSARFMassLNMultiS4DConfig(
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
        model_cfg = SPLMSARFMassLNMultiS4DConfig(
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
    model: ScalarPotentialLMSARFMassLNMultiS4D,
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


def _format_eig_summary(diag: dict) -> str:
    re = diag["eigvals_re"]
    im = diag["eigvals_im"]
    parts = [f"({r:+.3f}{i:+.3f}j)" for r, i in zip(re, im)]
    return " ".join(parts)


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
        help="Fix the damping coefficient at this value. Default 0.30.",
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
        help="K — number of S4D channels (= number of complex eigenvalues).",
    )
    ap.add_argument(
        "--xi-theta", dest="xi_theta", type=float, default=200.0,
        help="Initial discretisation horizon (Δt = 1/theta). Default 200.",
    )
    ap.add_argument(
        "--xi-eigval-init", dest="xi_eigval_init",
        choices=["legt", "s4d_lin"], default="legt",
        help="How to initialise the diagonal eigenvalues of A. 'legt' "
             "(default) uses the HiPPO-LegT spectrum so R6.i is a strict "
             "generalisation of R6.a/R6.e; 's4d_lin' uses S4D-Lin from "
             "Gu, Goel, Gu, Re 2022.",
    )
    ap.add_argument(
        "--xi-fixed-dt", dest="xi_fixed_dt", action="store_true",
        default=False,
        help="If set, freeze Δt at its init value (1/theta). Default is "
             "learnable Δt — matches R6.e's setup carried over to S4D.",
    )
    ap.add_argument(
        "--xi-fixed-B", dest="xi_fixed_B", action="store_true", default=False,
        help="If set, freeze the per-channel input gains B at their init "
             "values. Default is learnable B (the natural S4D parameterisation).",
    )
    ap.add_argument(
        "--causal-force", dest="causal_force", default="true",
        choices=["true", "false"],
        help="When 'true' (default) the integrator uses h.detach() before "
             "computing the S4D state, severing the anti-causal autograd "
             "leak documented in docs/Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md.",
    )
    ap.add_argument(
        "--max-steps", dest="max_steps", type=int, default=None,
        help="If set, cap training at this many steps (overrides the mode's "
             "default schedule but keeps the warmup / eval cadence).",
    )
    args = ap.parse_args()
    args.causal_force = (args.causal_force.lower() == "true")
    xi_learnable_dt = not args.xi_fixed_dt
    xi_learnable_B = not args.xi_fixed_B

    device = args.device or _pick_device()
    results_dir = (
        Path(args.results_dir).expanduser().resolve()
        if args.results_dir is not None else RESULTS_DIR
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[multixi-s4d-splm] device={device}  mode={args.mode}  "
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
        f"[multixi-s4d-splm] tokens: train={len(train_ids):,}  "
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
        xi_theta=args.xi_theta,
        xi_eigval_init=args.xi_eigval_init,
        xi_learnable_dt=xi_learnable_dt,
        xi_learnable_B=xi_learnable_B,
        causal_force=args.causal_force,
    )
    if args.max_steps is not None:
        train_cfg["steps"] = args.max_steps
    model = ScalarPotentialLMSARFMassLNMultiS4D(model_cfg).to(device)
    n_params = model.num_params()
    diag = model.s4d_diagnostics()
    print(
        f"[multixi-s4d-splm] params: {n_params:,}   d={model_cfg.d}  "
        f"L={model_cfg.L}  v_hidden={model_cfg.v_hidden}  "
        f"max_len={model_cfg.max_len}  ln_after_step={model_cfg.ln_after_step}"
    )
    print(
        f"[multixi-s4d-splm] s4d: K={diag['K']}  init={diag['eigval_init']}  "
        f"dt={diag['dt']:.4g}  learnable_dt={diag['learnable_dt']}  "
        f"learnable_B={diag['learnable_B']}"
    )
    print(f"[multixi-s4d-splm] eigvals_init = {_format_eig_summary(diag)}")
    print(f"[multixi-s4d-splm] B_proj_init  = "
          f"{['{:+.3f}'.format(b) for b in diag['b_proj']]}")
    print(
        f"[multixi-s4d-splm] causal_force={model_cfg.causal_force}  "
        f"({'FIXED (post-bug)' if model_cfg.causal_force else 'BUGGY (pre-fix forensic)'})"
        f"  steps={train_cfg['steps']}"
    )

    xb0, _ = get_batch(train_ids, train_cfg["batch_size"],
                       train_cfg["block_size"], rng)
    x0 = torch.from_numpy(xb0).to(device)
    init_mass = model.mass_stats(x0)
    print(
        f"[multixi-s4d-splm] init mass: mean={init_mass['mean']:.3f}  "
        f"std={init_mass['std']:.3f}  "
        f"min={init_mass['min']:.3f}  max={init_mass['max']:.3f}"
    )

    optim = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"], betas=(0.9, 0.95),
    )

    tag = f"splm_em_ln_multixi_s4d_{args.mode}"
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
            d_now = model.s4d_diagnostics()
            dt_val = d_now["dt"]
            eig_re_max = max(d_now["eigvals_re"])
            eig_re_min = min(d_now["eigvals_re"])
            eig_im_absmax = max(abs(z) for z in d_now["eigvals_im"])
            print(
                f"[multixi-s4d-splm] step {step+1:5d}/{train_cfg['steps']}   "
                f"train {avg:.4f}   lr {lr_now:.2e}   "
                f"grad {grad_norm:.2f}   "
                f"m[mean {mstats['mean']:.3f} std {mstats['std']:.3f}]   "
                f"gamma={gamma_val:.3f}   "
                f"dt={dt_val:.4g}   "
                f"eig.re=[{eig_re_min:+.2f},{eig_re_max:+.2f}]   "
                f"|eig.im|max={eig_im_absmax:.2f}   "
                f"elapsed {elapsed:.0f}s"
            )
            log_f.write(json.dumps({
                "step": step + 1, "train_loss": avg,
                "lr": lr_now, "grad_norm": float(grad_norm),
                "mass_mean": mstats["mean"], "mass_std": mstats["std"],
                "mass_min":  mstats["min"],  "mass_max": mstats["max"],
                "gamma": gamma_val,
                "s4d_dt": dt_val,
                "s4d_eigvals_re": d_now["eigvals_re"],
                "s4d_eigvals_im": d_now["eigvals_im"],
                "s4d_b_proj":     d_now["b_proj"],
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
                f"[multixi-s4d-splm] step {step+1:5d}   "
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
    final_diag = model.s4d_diagnostics()
    final_dt = final_diag["dt"]
    total_elapsed = time.time() - t0
    print(
        f"\n[multixi-s4d-splm] DONE  val_loss={final_val:.4f}  "
        f"val_ppl={final_ppl:.2f}  gamma={final_gamma:.4f}  "
        f"dt={final_dt:.4g}  elapsed={total_elapsed:.0f}s"
    )
    print(
        f"[multixi-s4d-splm] DONE  eigvals_final = "
        f"{_format_eig_summary(final_diag)}"
    )
    print(
        f"[multixi-s4d-splm] DONE  B_proj_final  = "
        f"{['{:+.3f}'.format(b) for b in final_diag['b_proj']]}"
    )

    ckpt_path = results_dir / f"{tag}_ckpt_latest.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_cfg": asdict(model_cfg),
        "train_cfg": train_cfg,
        "final_val_loss": final_val,
        "final_val_ppl": final_ppl,
        "final_gamma": final_gamma,
        "final_s4d_dt": final_dt,
        "final_s4d_eigvals_re": final_diag["eigvals_re"],
        "final_s4d_eigvals_im": final_diag["eigvals_im"],
        "final_s4d_b_proj":     final_diag["b_proj"],
        "fixed_gamma": args.fixed_gamma,
        "max_train_tokens": args.max_train_tokens,
        "logfreq_path": str(logfreq_path),
        "variant": "sarf_mass_ln_multixi_s4d",
        "experiment": "E13_multixi_s4d_scaleup",
        "tag": tag,
        "seed": args.seed,
        "elapsed_sec": total_elapsed,
    }, ckpt_path)
    print(f"[multixi-s4d-splm] checkpoint saved to {ckpt_path}")

    steps_v, train_vs, val_vs = [], [], []
    if loss_history:
        steps_v, train_vs, val_vs = zip(*loss_history)
    fig, ax = plt.subplots(figsize=(7, 4))
    if steps_v:
        ax.plot(steps_v, [math.exp(v) for v in val_vs],
                label="val ppl (E13 multi-ξ S4D)", color="purple")
    ax.set_xlabel("step")
    ax.set_ylabel("perplexity")
    gamma_str = (f"fixed γ={args.fixed_gamma}" if args.fixed_gamma is not None
                 else "free γ")
    ax.set_title(
        f"SPLM em_ln multi-ξ S4D — {args.mode} ({gamma_str}) — "
        f"K={model_cfg.xi_channels} init={args.xi_eigval_init} "
        f"seed={args.seed}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / f"{tag}_loss_curve.png", dpi=120)
    plt.close(fig)

    summary_path = results_dir / f"{tag}_summary.md"
    with summary_path.open("w") as f:
        f.write(f"# Training summary — {tag}\n\n")
        f.write("- experiment: E13 S4D multi-channel-ξ SPLM scale-up\n")
        f.write("- model: ScalarPotentialLMSARFMassLNMultiS4D "
                "(em_ln + multi-ξ S4D)\n")
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
            f"init={model_cfg.xi_eigval_init}  "
            f"theta(init)={model_cfg.xi_theta}  "
            f"learnable_dt={model_cfg.xi_learnable_dt}  "
            f"learnable_B={model_cfg.xi_learnable_B}\n"
        )
        f.write(
            f"- block_size: {train_cfg['block_size']}  "
            f"batch_size: {train_cfg['batch_size']}  "
            f"steps: {train_cfg['steps']}\n"
        )
        f.write(f"- causal_force: {model_cfg.causal_force}\n")
        f.write(f"- final_val_loss: {final_val:.4f}\n")
        f.write(f"- final_val_ppl:  {final_ppl:.2f}\n")
        f.write(f"- final_gamma:    {final_gamma:.4f}\n")
        f.write(f"- final_dt:       {final_dt:.4g}\n")
        f.write(f"- final_eigvals:  {_format_eig_summary(final_diag)}\n")
        f.write(f"- final_b_proj:   "
                f"{['{:+.4f}'.format(b) for b in final_diag['b_proj']]}\n")
        f.write(f"- elapsed_sec:    {total_elapsed:.0f}\n")
        f.write(f"- ckpt:           {ckpt_path.name}\n")
    print(f"[multixi-s4d-splm] summary written to {summary_path}")


if __name__ == "__main__":
    main()
