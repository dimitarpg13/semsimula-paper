"""Training loop for SP-HSPLM Stage 1: SPLM em_ln + per-token non-conservative force.

Pre-registered protocol:
  docs/SP_HSPLM_Stage_1_pre-registered_protocol.md

This trainer is a hard-fork of train_splm_em_ln_scaleup.py adapted for the
Stage 1 cells. The model is ScalarPotentialLMNonConservative (em_ln + per-
token non-conservative force g_l) and the schedule is matched to P10g for
forward compatibility with Stage 2.

Cells (selected by --cell)
--------------------------
  e0_baseline         : g_l = 0 (matched Cell 0 baseline at the 16k schedule).
  e1_const_skew       : Class B, constant skew Omega = J - J^T.
  e2_affine_rank1     : Class B, affine-rank-1 skew Omega(h) = u h^T - h u^T.
  e3_lowrank_rank2    : Class B, low-rank skew (r=2) of h-dependent type.
  e4_solenoidal_rank4 : Class C, position-only solenoidal (r=4, h_rho=64).
  e5_lowrank_rank4    : Class B, low-rank skew (r=4) of h-dependent type.

Modes (selected by --mode)
--------------------------
  smoke               : 300-step pipeline-correctness verification (no PPL claim).
  scaleup             : Stage 1 protocol run (16k steps, 800 warmup, 800 eval,
                        d=256, L=8, v_hidden=1024, max_len=1024, block=512,
                        batch=16, lr=5e-4 cosine, AdamW). Matched to P10g.

Diagnostics
-----------
  causal-leak probe   : run at steps {1, mid (~8k), final (~16k)}; saves
                        causal_probe.json. Mandatory: leak floor must be
                        <= 1e-6 at every checkpoint per protocol section 4.3.
  nonconservative norms : ||g_l|| / ||f_l|| per layer, saved every eval
                          interval to nonconservative_norms.json. Used to
                          detect "non-conservative coefficient collapsed
                          to zero" (Outcome DELTA).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).parent
PARENT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results" / "sp_hsplm" / "stage1"
RESULTS_DIR.mkdir(exist_ok=True)

EM_MINIMA_DIR = PARENT_DIR / "energetic_minima"
SARF_MASS_DIR = PARENT_DIR / "sarf_mass_variant"
sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(EM_MINIMA_DIR))
sys.path.insert(0, str(SARF_MASS_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from data_module import get_batch, load_tiny_stories  # noqa: E402

from model_splm_nonconservative import (  # noqa: E402
    CELLS,
    ScalarPotentialLMNonConservative,
    SPLMNonConservativeConfig,
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


def build_config(
    mode: str,
    cell: str,
    logfreq_path: str | None,
    fixed_gamma: float | None = None,
) -> tuple[SPLMNonConservativeConfig, dict]:
    base_kw = dict(
        vocab_size=50257,
        mass_mode="logfreq",
        logfreq_init_alpha=0.1,
        logfreq_path=logfreq_path,
        ln_after_step=True,
        fixed_gamma=fixed_gamma,
        cell=cell,
    )
    if mode == "smoke":
        model_cfg = SPLMNonConservativeConfig(
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
            probe_steps=(1, 150, 300),
        )
    elif mode == "scaleup":
        # Stage 1 locked configuration (protocol section 4.1):
        # 16 000 steps matched to P10g for forward compatibility with Stage 2.
        model_cfg = SPLMNonConservativeConfig(
            d=256, max_len=1024, v_hidden=1024, v_depth=3, L=8,
            init_m=1.0, init_gamma=1.0,
            **base_kw,
        )
        train_cfg = dict(
            batch_size=16, block_size=512,
            steps=16000, lr=5e-4, weight_decay=0.01,
            warmup_steps=800, grad_clip=1.0,
            eval_interval=800, eval_iters=40,
            log_interval=100,
            probe_steps=(1, 8000, 16000),
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
def evaluate(model: ScalarPotentialLMNonConservative, ids: np.ndarray,
             iters: int, batch_size: int, block_size: int,
             rng: np.random.Generator, device: str) -> float:
    model.eval()
    losses: List[float] = []
    for _ in range(iters):
        xb, yb = get_batch(ids, batch_size, block_size, rng)
        x = torch.from_numpy(xb).to(device)
        y = torch.from_numpy(yb).to(device)
        with torch.enable_grad():
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def causal_leak_probe(
    model: ScalarPotentialLMNonConservative,
    val_ids: np.ndarray,
    block_size: int,
    rng: np.random.Generator,
    device: str,
    n_pairs: int = 4,
) -> Dict[str, float]:
    """Probe: perturb one token at t_pert and verify no logit at t < t_pert
    changes. Returns the maximum absolute logit deviation across n_pairs
    sampled perturbation pairs.
    """
    was_training = model.training
    model.eval()

    max_delta = 0.0
    for _ in range(n_pairs):
        xb, _ = get_batch(val_ids, 1, block_size, rng)
        x_a = torch.from_numpy(xb).to(device)
        T = x_a.shape[1]
        t_pert = T // 2
        x_b = x_a.clone()
        x_b[0, t_pert] = (int(x_a[0, t_pert].item()) + 7) % model.cfg.vocab_size
        with torch.enable_grad():
            logits_a, _ = model(x_a)
            logits_b, _ = model(x_b)
        delta = (
            (logits_a[:, :t_pert, :] - logits_b[:, :t_pert, :])
            .abs().max().item()
        )
        max_delta = max(max_delta, float(delta))

    if was_training:
        model.train()
    return {"max_logit_delta_past": max_delta, "n_pairs": n_pairs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["smoke", "scaleup"],
        default="smoke",
    )
    ap.add_argument(
        "--cell", choices=list(CELLS), required=True,
        help="Stage 1 cell identifier; selects the non-conservative force.",
    )
    ap.add_argument("--logfreq-path", dest="logfreq_path",
                    default=str(DEFAULT_LOGFREQ_PATH))
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--fixed-gamma", dest="fixed_gamma", type=float, default=None,
        help=(
            "Fix the damping coefficient at this value. Default None "
            "= gamma is *learned* (matches Stage 1 protocol section 4.1)."
        ),
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
        "--allow-tf32", dest="allow_tf32", action="store_true",
        default=False,
        help=(
            "Leave TF32 enabled (CUDA default) instead of forcing real "
            "fp32 matmuls. Defaults to False (TF32 disabled) per protocol "
            "section 4.1; the SPLM forward uses torch.autograd.grad with "
            "create_graph=True and is sensitive to TF32's 10-bit mantissa "
            "reduction."
        ),
    )
    args = ap.parse_args()

    device = args.device or _pick_device()
    results_dir = (
        Path(args.results_dir).expanduser().resolve()
        if args.results_dir is not None else RESULTS_DIR
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        if args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("[stage1-splm] TF32 ENABLED (--allow-tf32): producing "
                  "precision-artifact reference run; expect numerical drift "
                  "in the autograd.grad path")
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            print("[stage1-splm] TF32 disabled (default per Stage 1 "
                  "protocol section 4.1) for SPLM autograd.grad numerical "
                  "stability (CUDA matmuls in true fp32)")

    print(f"[stage1-splm] device={device}  mode={args.mode}  "
          f"cell={args.cell}  fixed_gamma={args.fixed_gamma!r}  "
          f"seed={args.seed}  results_dir={results_dir}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train_ids, val_ids = load_tiny_stories(
        max_train_tokens=args.max_train_tokens,
    )
    print(f"[stage1-splm] tokens: train={len(train_ids):,}  "
          f"val={len(val_ids):,}")

    logfreq_path = args.logfreq_path
    if not Path(logfreq_path).exists():
        raise FileNotFoundError(
            f"logfreq surprisal file not found at {logfreq_path}. "
            "Run scaleup/compute_unigram_frequencies_tinystories.py first."
        )

    model_cfg, train_cfg = build_config(
        args.mode, args.cell, logfreq_path, fixed_gamma=args.fixed_gamma,
    )
    model = ScalarPotentialLMNonConservative(model_cfg).to(device)
    n_params = model.num_params()
    n_force_params = sum(
        p.numel() for p in model.nonconservative.parameters()
        if p.requires_grad
    )
    print(f"[stage1-splm] params: total={n_params:,}  "
          f"force={n_force_params:,}  d={model_cfg.d}  L={model_cfg.L}  "
          f"v_hidden={model_cfg.v_hidden}  max_len={model_cfg.max_len}  "
          f"ln_after_step={model_cfg.ln_after_step}  cell={model_cfg.cell}")

    xb0, _ = get_batch(train_ids, train_cfg["batch_size"],
                       train_cfg["block_size"], rng)
    x0 = torch.from_numpy(xb0).to(device)
    init_mass = model.mass_stats(x0)
    print(f"[stage1-splm] init mass: mean={init_mass['mean']:.3f}  "
          f"std={init_mass['std']:.3f}  "
          f"min={init_mass['min']:.3f}  max={init_mass['max']:.3f}")

    optim = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"], betas=(0.9, 0.95),
    )

    tag = f"splm_nonconservative_{args.cell}_{args.mode}"
    if args.tag_suffix:
        tag = f"{tag}_{args.tag_suffix}"
    log_path = results_dir / f"{tag}_training_log.jsonl"
    log_f = log_path.open("w")
    loss_history: List[Tuple[int, float, float]] = []

    probe_records: List[Dict] = []
    norms_records: List[Dict] = []

    def run_probe(step: int, label: str):
        rec = causal_leak_probe(
            model, val_ids, train_cfg["block_size"], rng, device,
        )
        rec_full = {"step": step, "label": label, **rec}
        probe_records.append(rec_full)
        delta = rec["max_logit_delta_past"]
        status = "leak-clean" if delta <= 1e-6 else "LEAK!"
        print(f"[stage1-splm] causal-leak probe @ step {step} "
              f"({label}): max_logit_delta_past={delta:.2e} [{status}]")

    def run_norms_eval(step: int):
        stats = model.nonconservative_norm_stats(x0)
        rec = {"step": step, **stats}
        norms_records.append(rec)
        max_ratio = max(stats["ratio"]) if stats["ratio"] else 0.0
        mean_ratio = (
            sum(stats["ratio"]) / len(stats["ratio"])
            if stats["ratio"] else 0.0
        )
        print(f"[stage1-splm] nonconservative norms @ step {step}: "
              f"mean(||g||/||f||)={mean_ratio:.3f}  "
              f"max(||g||/||f||)={max_ratio:.3f}")

    t0 = time.time()
    model.train()
    running = 0.0
    n_run = 0

    for step in range(train_cfg["steps"]):
        if (step + 1) in train_cfg["probe_steps"]:
            label = (
                "init" if step + 1 == 1 else
                ("mid" if step + 1 < train_cfg["steps"] else "final")
            )
            run_probe(step + 1, label)
            run_norms_eval(step + 1)

        lr_now = lr_schedule(step, train_cfg["lr"],
                             train_cfg["warmup_steps"], train_cfg["steps"])
        for g_param in optim.param_groups:
            g_param["lr"] = lr_now

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
            print(f"[stage1-splm] step {step+1:5d}/{train_cfg['steps']}   "
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
                "elapsed_sec": elapsed,
            }) + "\n")
            log_f.flush()

        if (step + 1) % train_cfg["eval_interval"] == 0:
            val_loss = evaluate(model, val_ids,
                                train_cfg["eval_iters"],
                                train_cfg["batch_size"],
                                train_cfg["block_size"], rng, device)
            val_ppl = math.exp(val_loss)
            print(f"[stage1-splm] step {step+1:5d}   val_loss={val_loss:.4f}  "
                  f"val_ppl={val_ppl:.2f}")
            loss_history.append((
                step + 1,
                avg if n_run == 0 else running / max(n_run, 1),
                val_loss,
            ))
            run_norms_eval(step + 1)

    # Final probe at the last step (always, regardless of probe_steps).
    if train_cfg["steps"] not in train_cfg["probe_steps"]:
        run_probe(train_cfg["steps"], "final")

    log_f.close()
    final_val = evaluate(model, val_ids,
                         train_cfg["eval_iters"],
                         train_cfg["batch_size"],
                         train_cfg["block_size"], rng, device)
    final_ppl = math.exp(final_val)
    final_gamma = model.gamma.item()
    total_elapsed = time.time() - t0
    print(f"\n[stage1-splm] DONE  cell={args.cell}  "
          f"val_loss={final_val:.4f}  val_ppl={final_ppl:.2f}  "
          f"gamma={final_gamma:.4f}  elapsed={total_elapsed:.0f}s")

    ckpt_path = results_dir / f"{tag}_ckpt_latest.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_cfg": asdict(model_cfg),
        "train_cfg": train_cfg,
        "final_val_loss": final_val,
        "final_val_ppl": final_ppl,
        "final_gamma": final_gamma,
        "fixed_gamma": args.fixed_gamma,
        "max_train_tokens": args.max_train_tokens,
        "logfreq_path": str(logfreq_path),
        "variant": "sarf_mass_ln_nonconservative",
        "experiment": "SP_HSPLM_Stage_1",
        "cell": args.cell,
        "tag": tag,
        "seed": args.seed,
        "elapsed_sec": total_elapsed,
        "n_params": n_params,
        "n_force_params": n_force_params,
    }, ckpt_path)
    print(f"[stage1-splm] checkpoint saved to {ckpt_path}")

    # Save Stage 1 diagnostics: causal probe, non-conservative norms.
    probe_path = results_dir / f"{tag}_causal_probe.json"
    with probe_path.open("w") as f:
        json.dump(probe_records, f, indent=2)
    print(f"[stage1-splm] causal probe history -> {probe_path}")

    norms_path = results_dir / f"{tag}_nonconservative_norms.json"
    with norms_path.open("w") as f:
        json.dump(norms_records, f, indent=2)
    print(f"[stage1-splm] nonconservative norms history -> {norms_path}")

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
    ax.set_title(f"SP-HSPLM Stage 1 — {args.cell} — {args.mode} "
                 f"({gamma_str}) — seed={args.seed}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / f"{tag}_loss_curve.png", dpi=120)
    plt.close(fig)

    summary_path = results_dir / f"{tag}_summary.md"
    with summary_path.open("w") as f:
        f.write(f"# Training summary — {tag}\n\n")
        f.write(f"- experiment: SP-HSPLM Stage 1 "
                f"(per-token Class B/C rerun on leak-fixed v3 codebase)\n")
        f.write(f"- protocol: docs/SP_HSPLM_Stage_1_pre-registered_protocol.md\n")
        f.write(f"- model: ScalarPotentialLMNonConservative (em_ln + per-token g_l)\n")
        f.write(f"- cell: {args.cell}\n")
        f.write(f"- mode: {args.mode}\n")
        f.write(f"- corpus: TinyStories (cap {args.max_train_tokens:,} train tokens)\n")
        f.write(f"- fixed_gamma: {args.fixed_gamma}\n")
        f.write(f"- params: total {n_params:,}  force {n_force_params:,}\n")
        f.write(f"- d={model_cfg.d}  L={model_cfg.L}  "
                f"v_hidden={model_cfg.v_hidden}  max_len={model_cfg.max_len}  "
                f"ln_after_step=True\n")
        f.write(f"- block_size: {train_cfg['block_size']}  "
                f"batch_size: {train_cfg['batch_size']}  "
                f"steps: {train_cfg['steps']}\n")
        f.write(f"- seed: {args.seed}\n")
        f.write(f"- elapsed: {total_elapsed:.0f} s ({total_elapsed/3600:.2f} h)\n")
        f.write(f"\nFinal val loss: {final_val:.6f} (ppl {final_ppl:.2f})\n")
        f.write(f"Final gamma: {final_gamma:.4f}\n")
        f.write(f"\n## Causal-leak probe history\n\n")
        f.write("| step | label | max_logit_delta_past | verdict |\n")
        f.write("|---|---|---:|---|\n")
        for rec in probe_records:
            verdict = (
                "leak-clean" if rec["max_logit_delta_past"] <= 1e-6 else "LEAK"
            )
            f.write(
                f"| {rec['step']} | {rec['label']} | "
                f"{rec['max_logit_delta_past']:.2e} | {verdict} |\n"
            )
        f.write(f"\n## Nonconservative norms (final)\n\n")
        if norms_records:
            last = norms_records[-1]
            f.write("| layer | ||f|| | ||g|| | ||g||/||f|| |\n")
            f.write("|---:|---:|---:|---:|\n")
            for ell, (fn, gn, r) in enumerate(zip(
                last["f_norms"], last["g_norms"], last["ratio"],
            )):
                f.write(f"| {ell} | {fn:.3f} | {gn:.3f} | {r:.3f} |\n")
    print(f"[stage1-splm] summary written to {summary_path}")


if __name__ == "__main__":
    main()
