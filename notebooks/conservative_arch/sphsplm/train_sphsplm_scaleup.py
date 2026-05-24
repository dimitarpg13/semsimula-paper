"""Training loop for SP-HSPLM Stage 2: Q9(e) pair-skew cell ladder.

Pre-registered protocol:
  docs/SP_HSPLM_Stage_2_pre-registered_protocol.md

This trainer is a hard-fork of train_splm_nonconservative_scaleup.py
adapted for the Stage 2 SP-HSPLM cells. The model is
ScalarPotentialLMSPHSPLM (SparsePARFLM S-block + low-rank pair-skew
C-block, dispatched per layer by `cfg.schedule`) and the schedule is
matched to P10g for forward compatibility with the H1/H2 baseline
comparisons.

Cells (selected by --cell)
--------------------------
Mechanism-2 ladder (autonomous force law; SPLM "shared across ell"
commitment preserved):
  q9e_a : interleaved (SCSC...), k=4, r=16, no per-token gyro.
  q9e_b : interleaved, k=8, r=16 -- routing-density sweep.
  q9e_c : interleaved, k=4, r=32 -- kernel-rank sweep.
  q9e_d : interleaved, k=4, r=16, with per-token gyro Omega.
  q9e_e : bottom_c (CCCC SSSS), k=4, r=16 -- C-then-S ordering.
  q9e_f : top_c (SSSS CCCC), k=4, r=16 -- S-then-C ordering.
  q9e_g : sandwich (SSCCCCSS), k=4, r=16 -- conservative on edges.

Mechanism-1 extension (per-layer-indexed force law; lifts the SPLM
"shared across ell" commitment one submodule at a time; everything
else matches q9e_a):
  q9e_h : per-layer J_phi^(ell) (L_C independent skew kernels).
  q9e_i : per-layer V_phi^(ell) (L_S independent pair scalars).
  q9e_j : per-layer alpha_phi^(ell) (L independent score heads).
  q9e_k : per-layer J_phi + V_phi + alpha_phi (joint Mechanism-1).

Mechanism-1 x Mechanism-2 additivity cell (H6):
  q9e_l : q9e_d (gyro Omega on, shared) + q9e_h (per-layer J_phi).
          Tests whether the q9e_d and q9e_h improvements add.

Maximal-Mechanism-1 cell (contingent next-step after q9e_l):
  q9e_m : q9e_l + per-layer Omega^(ell) (L_C independent gyro
          kernels).  Lifts the last shared non-conservative submodule.

Full-Class-F test (intentionally non-iso-parameter-count vs P10g):
  q9e_n : q9e_m + per-layer V_theta^(ell) + per-layer V_phi^(ell)
          + per-layer alpha_phi^(ell).  Every SP-HSPLM force-law
          module is per-layer; V_theta dominates the parameter
          overhead (~10M extra params at d=256, v_hidden=1024,
          v_depth=3, L_S=4).  Tests H8: does the residual ~17 PPL
          gap to MatchedGPT collapse once V_theta is no longer
          shared across layers?

Modes (selected by --mode)
--------------------------
  smoke   : 300-step pipeline-correctness verification (no PPL claim).
  scaleup : Stage 2 protocol run (16k steps, 800 warmup, 800 eval,
            d=256, L=8, v_hidden=1024, max_len=1024, block=512,
            batch=16, lr=5e-4 cosine, AdamW). Matched to P10g.

Diagnostics
-----------
  causal-leak probe   : run at steps {1, mid (~8k), final (~16k)};
                        saves causal_probe.json. Mandatory: leak floor
                        must be <= 1e-6 at every checkpoint per
                        protocol section 4.3.
  pair_kernel_norms   : ||J_phi||_F, ||U||_F, ||V||_F per eval step,
                        saves pair_kernel_norms.json. Used for H4 and
                        Outcome DELTA detection (kernel collapse).
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
RESULTS_DIR = SCRIPT_DIR / "results" / "sp_hsplm" / "stage2"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

PARF_DIR = PARENT_DIR / "parf"
sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(PARF_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from data_module import get_batch, load_tiny_stories  # noqa: E402

from model_sphsplm import (  # noqa: E402
    CELLS,
    ScalarPotentialLMSPHSPLM,
    SPHSPLMConfig,
    _stage2_cell_kwargs,
    pair_kernel_norms,
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
) -> tuple[SPHSPLMConfig, dict]:
    """Build SP-HSPLM config + train config for the given Stage 2 cell.

    The base config matches the SparsePARFLM P10g locked configuration
    (16k-step scaleup) so the H1 PPL comparison is apples-to-apples
    with no parameter or schedule asterisk.
    """
    base_kw = dict(
        vocab_size=50257,
        mass_mode="logfreq",
        logfreq_init_alpha=0.1,
        logfreq_path=logfreq_path,
        ln_after_step=True,
        causal_force=True,
        score_head_hidden=32,
        score_head_init_scale=0.02,
        score_head_use_detached_h_src=True,
        gumbel_tau_init=1.0,
        gumbel_tau_min=0.1,
        gumbel_noise=True,
        kernel_init_scale=0.02,
        gyro_init_scale=0.02,
        gamma_min=0.05,
    )
    if mode == "smoke":
        base = dict(
            d=256, max_len=1024, v_hidden=1024, v_depth=3, L=8,
            init_m=1.0, init_gamma=1.0,
            v_phi_d_type=8, v_phi_d_angle=4,
            v_phi_phi_hidden=32, v_phi_theta_hidden=32,
            v_phi_mlp_hidden=32,
            **base_kw,
        )
        kwargs = _stage2_cell_kwargs(cell, base)
        model_cfg = SPHSPLMConfig(**kwargs)
        train_cfg = dict(
            batch_size=8, block_size=256,
            steps=300, lr=5e-4, weight_decay=0.01,
            warmup_steps=20, grad_clip=1.0,
            eval_interval=100, eval_iters=10,
            log_interval=10,
            probe_steps=(1, 150, 300),
            tau_anneal_steps=200,
            skew_warmup_steps=200,
            skew_warmup_lambda=1e-2,
        )
    elif mode == "scaleup":
        base = dict(
            d=256, max_len=1024, v_hidden=1024, v_depth=3, L=8,
            init_m=1.0, init_gamma=1.0,
            v_phi_d_type=8, v_phi_d_angle=4,
            v_phi_phi_hidden=32, v_phi_theta_hidden=32,
            v_phi_mlp_hidden=32,
            **base_kw,
        )
        kwargs = _stage2_cell_kwargs(cell, base)
        model_cfg = SPHSPLMConfig(**kwargs)
        train_cfg = dict(
            batch_size=16, block_size=512,
            steps=16000, lr=5e-4, weight_decay=0.01,
            warmup_steps=800, grad_clip=1.0,
            eval_interval=800, eval_iters=40,
            log_interval=100,
            probe_steps=(1, 8000, 16000),
            tau_anneal_steps=8000,
            skew_warmup_steps=200,
            skew_warmup_lambda=1e-2,
        )
    else:
        raise ValueError(f"unknown mode: {mode!r}")
    return model_cfg, train_cfg


def lr_schedule(step: int, lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def tau_schedule(
    step: int, tau_init: float, tau_min: float, anneal_steps: int,
) -> float:
    """Linear anneal of Gumbel temperature from tau_init to tau_min."""
    if anneal_steps <= 0:
        return tau_init
    progress = step / max(anneal_steps, 1)
    return float(tau_init + (tau_min - tau_init) * min(progress, 1.0))


def skew_warmup_lambda(
    step: int, lam: float, warmup_steps: int,
) -> float:
    """Frobenius warm-up regulariser per protocol section 4.1.

    Returns lambda * max(0, 1 - step / warmup_steps).
    """
    if warmup_steps <= 0:
        return 0.0
    decay = max(0.0, 1.0 - step / warmup_steps)
    return float(lam * decay)


def evaluate(
    model: ScalarPotentialLMSPHSPLM, ids: np.ndarray,
    iters: int, batch_size: int, block_size: int,
    rng: np.random.Generator, device: str,
) -> float:
    model.eval()
    losses: List[float] = []
    for _ in range(iters):
        xb, yb = get_batch(ids, batch_size, block_size, rng)
        x = torch.from_numpy(xb).to(device)
        y = torch.from_numpy(yb).to(device)
        # SPLM forward uses autograd.grad inside; cannot wrap in no_grad.
        out = model(x, y)
        loss = out[1]
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def causal_leak_probe(
    model: ScalarPotentialLMSPHSPLM,
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
        x_b[0, t_pert] = (
            int(x_a[0, t_pert].item()) + 7
        ) % model.cfg.vocab_size
        logits_a = model(x_a)[0]
        logits_b = model(x_b)[0]
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
        "--mode", choices=["smoke", "scaleup"], default="smoke",
    )
    ap.add_argument(
        "--cell", choices=list(CELLS), required=True,
        help="Stage 2 cell identifier; selects the SP-HSPLM C-block "
             "schedule, top_k, and kernel_rank.",
    )
    ap.add_argument("--logfreq-path", dest="logfreq_path",
                    default=str(DEFAULT_LOGFREQ_PATH))
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
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
            "fp32 matmuls. Defaults to False (TF32 disabled) per "
            "protocol section 4.1; the SPLM autograd.grad path is "
            "sensitive to TF32's 10-bit mantissa reduction."
        ),
    )
    args = ap.parse_args()

    device = args.device or _pick_device()
    results_dir = (
        Path(args.results_dir).expanduser().resolve()
        if args.results_dir is not None else RESULTS_DIR
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    # TF32 is EXPLICITLY DISABLED by default per Stage 2 protocol
    # section 4.1. Three independent knobs are set + read back:
    #   1. torch.backends.cuda.matmul.allow_tf32  (cuBLAS path)
    #   2. torch.backends.cudnn.allow_tf32        (cuDNN path)
    #   3. torch.set_float32_matmul_precision('highest')  (modern API,
    #      equivalent to (1) but explicit at the matmul-precision layer)
    # The --allow-tf32 flag flips all three knobs to the TF32-enabled
    # state for explicit-benchmarking runs only.
    if torch.cuda.is_available():
        if args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            print("[stage2-sphsplm] TF32 ENABLED (--allow-tf32): "
                  "expect numerical drift in the autograd.grad path; "
                  "matmul.allow_tf32=True, cudnn.allow_tf32=True, "
                  "float32_matmul_precision='high'")
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
            assert torch.backends.cuda.matmul.allow_tf32 is False
            assert torch.backends.cudnn.allow_tf32 is False
            assert torch.get_float32_matmul_precision() == "highest"
            print("[stage2-sphsplm] TF32 EXPLICITLY DISABLED "
                  "(default per Stage 2 protocol section 4.1): "
                  "matmul.allow_tf32=False, cudnn.allow_tf32=False, "
                  "float32_matmul_precision='highest' "
                  "(true fp32 matmuls for autograd.grad numerical "
                  "stability)")
    else:
        torch.set_float32_matmul_precision("highest")
        print("[stage2-sphsplm] CPU run; float32_matmul_precision="
              f"{torch.get_float32_matmul_precision()!r}")

    print(f"[stage2-sphsplm] device={device}  mode={args.mode}  "
          f"cell={args.cell}  seed={args.seed}  results_dir={results_dir}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train_ids, val_ids = load_tiny_stories(
        max_train_tokens=args.max_train_tokens,
    )
    print(f"[stage2-sphsplm] tokens: train={len(train_ids):,}  "
          f"val={len(val_ids):,}")

    logfreq_path = args.logfreq_path
    if not Path(logfreq_path).exists():
        raise FileNotFoundError(
            f"logfreq surprisal file not found at {logfreq_path}. "
            "Run scaleup/compute_unigram_frequencies_tinystories.py "
            "first."
        )

    model_cfg, train_cfg = build_config(
        args.mode, args.cell, logfreq_path,
    )
    model = ScalarPotentialLMSPHSPLM(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_skew = sum(p.numel() for p in model.skew_kernel.parameters())
    n_gyro = (
        sum(p.numel() for p in model.gyro_kernel.parameters())
        if model.gyro_kernel is not None else 0
    )
    nS, nC = model.schedule_counts()
    print(f"[stage2-sphsplm] params: total={n_params:,}  "
          f"skew={n_skew:,}  gyro={n_gyro:,}  d={model_cfg.d}  "
          f"L={model_cfg.L}  v_hidden={model_cfg.v_hidden}  "
          f"max_len={model_cfg.max_len}  schedule={model_cfg.schedule}  "
          f"(nS={nS} nC={nC})  k={model_cfg.top_k}  "
          f"r={model_cfg.kernel_rank}  "
          f"gyro={model_cfg.use_pertoken_gyro}")

    optim = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"], betas=(0.9, 0.95),
    )

    tag = f"sphsplm_{args.cell}_{args.mode}"
    if args.tag_suffix:
        tag = f"{tag}_{args.tag_suffix}"
    log_path = results_dir / f"{tag}_training_log.jsonl"

    # Also mirror the JSONL log to a fast local directory on /content/ so it
    # is never hostage to the GDrive FUSE write-back buffer. The local mirror
    # gets every flush() instantly; the Drive copy is the canonical artefact
    # but may lag in the GDrive web UI by minutes. We tee writes to both;
    # at end-of-run, copy the local file over the Drive one to guarantee a
    # complete, byte-exact, atomic Drive artefact.
    local_mirror_dir = Path("/content") / "sphsplm_local" / tag
    try:
        local_mirror_dir.mkdir(parents=True, exist_ok=True)
        local_log_path = local_mirror_dir / f"{tag}_training_log.jsonl"
    except Exception:
        local_log_path = None
        print("[stage2-sphsplm] could not create /content/sphsplm_local "
              "mirror; JSONL log lives on Drive only.")

    class _TeeWriter:
        """File-like that mirrors writes to two underlying file handles."""
        def __init__(self, primary, mirror):
            self._primary = primary
            self._mirror = mirror

        def write(self, data):
            self._primary.write(data)
            if self._mirror is not None:
                self._mirror.write(data)
            return len(data)

        def flush(self):
            self._primary.flush()
            if self._mirror is not None:
                self._mirror.flush()

        def close(self):
            self._primary.close()
            if self._mirror is not None:
                self._mirror.close()

    drive_log_f = log_path.open("w")
    local_log_f = (
        local_log_path.open("w") if local_log_path is not None else None
    )
    log_f = _TeeWriter(drive_log_f, local_log_f)
    if local_log_path is not None:
        print(f"[stage2-sphsplm] JSONL log -> {log_path}")
        print(f"[stage2-sphsplm] JSONL log (local mirror, no FUSE lag) "
              f"-> {local_log_path}")
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
        print(f"[stage2-sphsplm] causal-leak probe @ step {step} "
              f"({label}): max_logit_delta_past={delta:.2e} "
              f"[{status}]")

    def run_norms_eval(step: int):
        norms = pair_kernel_norms(model)
        rec = {"step": step, **norms}
        norms_records.append(rec)
        print(f"[stage2-sphsplm] pair-kernel norms @ step {step}: "
              f"||J_phi||_F={norms['J_phi_fro']:.4f}  "
              f"||U||={norms['U_fro']:.4f}  "
              f"||V||={norms['V_fro']:.4f}"
              + (
                  f"  ||Omega||_F={norms['Omega_fro']:.4f}"
                  if "Omega_fro" in norms else ""
              )
              )

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

        lr_now = lr_schedule(
            step, train_cfg["lr"],
            train_cfg["warmup_steps"], train_cfg["steps"],
        )
        for g_param in optim.param_groups:
            g_param["lr"] = lr_now

        tau_now = tau_schedule(
            step, model_cfg.gumbel_tau_init, model_cfg.gumbel_tau_min,
            train_cfg["tau_anneal_steps"],
        )
        model.set_gumbel_tau(tau_now)

        xb, yb = get_batch(
            train_ids, train_cfg["batch_size"],
            train_cfg["block_size"], rng,
        )
        x = torch.from_numpy(xb).to(device)
        y = torch.from_numpy(yb).to(device)

        _, loss = model(x, y)

        # Optional Frobenius warm-up regulariser on the skew kernel.
        # Decays linearly from `skew_warmup_lambda` at step 0 to 0 at
        # step `skew_warmup_steps`, then identically zero. The
        # default lambda 1e-2 keeps ||J_phi||_F suppressed during the
        # earliest steps when the SPLM autograd.grad path is most
        # sensitive to large velocity-coupled forces.
        lam_skew = skew_warmup_lambda(
            step,
            train_cfg["skew_warmup_lambda"],
            train_cfg["skew_warmup_steps"],
        )
        if lam_skew > 0.0:
            # Frobenius warm-up regulariser, summed over all skew
            # kernels (one shared kernel in q9e_a..g; L_C per-layer
            # kernels in q9e_h/k).  See
            # ScalarPotentialLMSPHSPLM.skew_kernel_frobenius_squared.
            reg = lam_skew * model.skew_kernel_frobenius_squared()
            loss = loss + reg

        optim.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(), train_cfg["grad_clip"],
        )
        optim.step()

        running += loss.item()
        n_run += 1

        if (step + 1) % train_cfg["log_interval"] == 0:
            avg = running / n_run
            running, n_run = 0.0, 0
            elapsed = time.time() - t0
            gamma_val = float(model.gamma.item())
            print(f"[stage2-sphsplm] step {step+1:5d}/"
                  f"{train_cfg['steps']}   train {avg:.4f}   "
                  f"lr {lr_now:.2e}   tau {tau_now:.3f}   "
                  f"grad {grad_norm:.2f}   gamma={gamma_val:.3f}   "
                  f"elapsed {elapsed:.0f}s")
            log_f.write(json.dumps({
                "step": step + 1, "train_loss": avg,
                "lr": lr_now, "tau": tau_now, "lam_skew": lam_skew,
                "grad_norm": float(grad_norm),
                "gamma": gamma_val,
                "elapsed_sec": elapsed,
            }) + "\n")
            log_f.flush()

        if (step + 1) % train_cfg["eval_interval"] == 0:
            val_loss = evaluate(
                model, val_ids,
                train_cfg["eval_iters"],
                train_cfg["batch_size"],
                train_cfg["block_size"], rng, device,
            )
            val_ppl = math.exp(val_loss)
            print(f"[stage2-sphsplm] step {step+1:5d}   "
                  f"val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}")
            loss_history.append((
                step + 1,
                avg if n_run == 0 else running / max(n_run, 1),
                val_loss,
            ))
            log_f.write(json.dumps({
                "step": step + 1,
                "val_loss": float(val_loss),
                "val_ppl": float(val_ppl),
                "elapsed_sec": time.time() - t0,
            }) + "\n")
            log_f.flush()
            run_norms_eval(step + 1)

    if train_cfg["steps"] not in train_cfg["probe_steps"]:
        run_probe(train_cfg["steps"], "final")

    log_f.close()
    # Atomic-copy the local JSONL mirror over the Drive copy so the Drive
    # artefact is byte-exact even if FUSE buffering dropped intermediate
    # writes during training. The local file is always the source of truth.
    if local_log_path is not None and local_log_path.exists():
        try:
            import shutil as _shutil
            _shutil.copy2(local_log_path, log_path)
            print(f"[stage2-sphsplm] JSONL log atomically synced "
                  f"local -> Drive  (source: {local_log_path})")
        except Exception as _e:
            print(f"[stage2-sphsplm] warning: final local->Drive sync "
                  f"failed: {_e}")
    final_val = evaluate(
        model, val_ids,
        train_cfg["eval_iters"],
        train_cfg["batch_size"],
        train_cfg["block_size"], rng, device,
    )
    final_ppl = math.exp(final_val)
    final_gamma = float(model.gamma.item())
    total_elapsed = time.time() - t0
    print(f"\n[stage2-sphsplm] DONE  cell={args.cell}  "
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
        "max_train_tokens": args.max_train_tokens,
        "logfreq_path": str(logfreq_path),
        "variant": "sp_hsplm_stage2",
        "experiment": "SP_HSPLM_Stage_2",
        "cell": args.cell,
        "tag": tag,
        "seed": args.seed,
        "elapsed_sec": total_elapsed,
        "n_params": n_params,
        "n_skew_params": n_skew,
        "n_gyro_params": n_gyro,
    }, ckpt_path)
    print(f"[stage2-sphsplm] checkpoint saved to {ckpt_path}")

    probe_path = results_dir / f"{tag}_causal_probe.json"
    with probe_path.open("w") as f:
        json.dump(probe_records, f, indent=2)
    print(f"[stage2-sphsplm] causal probe history -> {probe_path}")

    norms_path = results_dir / f"{tag}_pair_kernel_norms.json"
    with norms_path.open("w") as f:
        json.dump(norms_records, f, indent=2)
    print(f"[stage2-sphsplm] pair-kernel norms history -> {norms_path}")

    steps_v, train_vs, val_vs = [], [], []
    if loss_history:
        steps_v, train_vs, val_vs = zip(*loss_history)
    fig, ax = plt.subplots(figsize=(7, 4))
    if steps_v:
        ax.plot(
            steps_v, [math.exp(v) for v in val_vs],
            label="val ppl", color="steelblue",
        )
    ax.set_xlabel("step")
    ax.set_ylabel("perplexity")
    ax.set_title(
        f"SP-HSPLM Stage 2 — {args.cell} — {args.mode} — "
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
        f.write(
            "- experiment: SP-HSPLM Stage 2 (Q9(e) pair-skew cell ladder)\n"
        )
        f.write(
            "- protocol: docs/SP_HSPLM_Stage_2_pre-registered_protocol.md\n"
        )
        f.write(
            "- model: ScalarPotentialLMSPHSPLM (SparsePARFLM S-block + "
            "low-rank pair-skew C-block)\n"
        )
        f.write(f"- cell: {args.cell}  schedule: {model_cfg.schedule}\n")
        f.write(f"- mode: {args.mode}\n")
        f.write(
            f"- corpus: TinyStories (cap {args.max_train_tokens:,} train "
            "tokens)\n"
        )
        f.write(
            f"- params: total {n_params:,}  skew {n_skew:,}  "
            f"gyro {n_gyro:,}\n"
        )
        f.write(
            f"- d={model_cfg.d}  L={model_cfg.L}  "
            f"v_hidden={model_cfg.v_hidden}  "
            f"max_len={model_cfg.max_len}  ln_after_step=True  "
            f"k={model_cfg.top_k}  r={model_cfg.kernel_rank}  "
            f"gyro={model_cfg.use_pertoken_gyro}\n"
        )
        per_layer_flags: list[str] = []
        if not model_cfg.share_skew_kernel_across_layers:
            per_layer_flags.append("J_phi")
        if (not model_cfg.share_gyro_kernel_across_layers
                and model_cfg.use_pertoken_gyro):
            per_layer_flags.append("Omega")
        if not model_cfg.share_v_theta_across_layers:
            per_layer_flags.append("V_theta")
        if not model_cfg.share_v_phi_across_layers:
            per_layer_flags.append("V_phi")
        if not model_cfg.share_score_head_across_layers:
            per_layer_flags.append("alpha_phi")
        per_layer_str = (
            ", ".join(per_layer_flags) if per_layer_flags else "none "
            "(SPLM autonomous commitment)"
        )
        n_v_theta = sum(p.numel() for p in model.V_theta.parameters())
        n_v_phi = sum(p.numel() for p in model.V_phi.parameters())
        n_score_head_p = sum(p.numel() for p in model.score_head.parameters())
        f.write(
            f"- mechanism-1 per-layer modules: {per_layer_str}\n"
        )
        f.write(
            f"- module param breakdown: V_theta={n_v_theta:,}  "
            f"V_phi={n_v_phi:,}  alpha_phi={n_score_head_p:,}  "
            f"skew={n_skew:,}  gyro={n_gyro:,}\n"
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
        f.write(
            f"\nFinal val loss: {final_val:.6f} (ppl {final_ppl:.2f})\n"
        )
        f.write(f"Final gamma: {final_gamma:.4f}\n")

        f.write("\n## Causal-leak probe history\n\n")
        f.write("| step | label | max_logit_delta_past | verdict |\n")
        f.write("|---|---|---:|---|\n")
        for rec in probe_records:
            verdict = (
                "leak-clean"
                if rec["max_logit_delta_past"] <= 1e-6 else "LEAK"
            )
            f.write(
                f"| {rec['step']} | {rec['label']} | "
                f"{rec['max_logit_delta_past']:.2e} | {verdict} |\n"
            )

        f.write("\n## Pair-kernel norms (final)\n\n")
        if norms_records:
            last = norms_records[-1]
            # Scalar row first (J_phi_fro / U_fro / V_fro / Omega_*).
            f.write("| quantity | value |\n")
            f.write("|---|---:|\n")
            for k, v in last.items():
                if k == "step":
                    continue
                if isinstance(v, (list, tuple)):
                    continue
                f.write(f"| {k} | {v:.4f} |\n")
            # Per-layer block follows, when Mechanism-1 cells expose
            # per-layer kernel lists.
            per_layer_keys = [
                k for k, v in last.items() if isinstance(v, (list, tuple))
            ]
            if per_layer_keys:
                f.write("\n### Per-layer pair-kernel norms\n\n")
                f.write(
                    "These rows are populated only when the cell config "
                    "has at least one `share_*_across_layers=False` flag, "
                    "i.e. the Mechanism-1 cells (q9e_h/i/j/k).  The "
                    "indices below run over the relevant block-type "
                    "positions: J_phi/Omega keys index C-blocks; "
                    "V_phi keys index S-blocks.\n\n"
                )
                for k in per_layer_keys:
                    vals = last[k]
                    if not vals:
                        f.write(f"- `{k}`: (empty)\n")
                        continue
                    rendered = ", ".join(f"{float(v):.4f}" for v in vals)
                    f.write(f"- `{k}` (len {len(vals)}): [{rendered}]\n")
    print(f"[stage2-sphsplm] summary written to {summary_path}")


if __name__ == "__main__":
    main()
