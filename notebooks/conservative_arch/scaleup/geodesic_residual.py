#!/usr/bin/env python3
"""
Geodesic residual analysis for Fock-PARFLM gamma sweep checkpoints.

Computes the damped-geodesic residual R_bar(gamma) from retained gamma-sweep
checkpoints and overlays it against PPL(gamma) to test the mechanistic claim:
"the damping that minimises perplexity is the one that makes the dynamics
most geodesic."

This is an inference-only script — no training, no gradient accumulation.
All heavy math (Christoffel symbols, conformal factor gradients) is
closed-form via V_theta.analytical_grad(); no autograd is needed.

Theory: see docs/geodesic_preservation_experiment_proposal.md sections 2-4.

Usage:
  # Diagonal overlay (one R_bar per trained gamma):
  python geodesic_residual.py \\
      --sweep_dir ~/runs/sweep_sweep-d768_.../gamma_sweep \\
      --preset sweep-d768 \\
      --data_dir ~/data \\
      --output_dir ~/runs/geodesic_d768

  # Off-diagonal heatmap (vary gamma_eval independently of gamma_train):
  python geodesic_residual.py \\
      --sweep_dir ~/runs/sweep_sweep-d768_.../gamma_sweep \\
      --preset sweep-d768 --data_dir ~/data \\
      --output_dir ~/runs/geodesic_d768 \\
      --eval_gammas 0.01,0.02,0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50

  # With null controls (shuffled-Gamma, random-v):
  python geodesic_residual.py \\
      --sweep_dir ~/runs/sweep_sweep-d768_.../gamma_sweep \\
      --preset sweep-d768 --data_dir ~/data \\
      --output_dir ~/runs/geodesic_d768 --controls
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Bootstrap imports from train_fock (reuse model builders, data loading, etc.)
# ---------------------------------------------------------------------------

def _bootstrap():
    """Add the same sys.path entries that train_fock uses."""
    ca_dir = SCRIPT_DIR.parent
    for sub in ["", "parf", "multixi", "scaleup",
                "sarf_mass_variant", "energetic_minima"]:
        d = str(ca_dir / sub) if sub else str(ca_dir)
        if d not in sys.path:
            sys.path.insert(0, d)

_bootstrap()

from train_fock import (  # noqa: E402
    TrainConfig, PRESETS, load_data, build_fock_model, ensure_logfreq,
    get_batch,
)


# ---------------------------------------------------------------------------
# Model loading from checkpoint
# ---------------------------------------------------------------------------

def load_checkpoint_model(
    ckpt_path: str,
    cfg: TrainConfig,
    device: str,
    logfreq_path: str,
) -> Tuple[torch.nn.Module, object, float, float]:
    """Load a gamma-sweep checkpoint and reconstruct the model.

    Returns (model, model_cfg, gamma_train, val_ppl).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    gamma_train = ckpt.get("gamma", cfg.fixed_gamma)
    val_ppl = ckpt.get("val_ppl", float("inf"))

    cfg.fixed_gamma = gamma_train
    model, model_cfg, _ = build_fock_model(cfg, device, logfreq_path)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    del ckpt
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model, model_cfg, gamma_train, val_ppl


# ---------------------------------------------------------------------------
# Trajectory collection
# ---------------------------------------------------------------------------

def collect_trajectory(
    model: torch.nn.Module,
    x: torch.Tensor,
) -> List[torch.Tensor]:
    """Run a forward pass and return per-layer hidden states.

    Returns a list of L+1 tensors [h_0, h_1, ..., h_L], each (B, T, d),
    detached on CPU (as returned by _stack_forward with return_trajectory=True).

    Note: torch.enable_grad() is required because _layer_step uses
    autograd.grad(U, h_in, create_graph=training) internally for the
    conservative force, even in eval mode.
    """
    with torch.enable_grad():
        h0 = model._embed(x)
        _, traj = model._stack_forward(h0, x, return_trajectory=True)
    return traj


# ---------------------------------------------------------------------------
# V_theta evaluation (closed-form, no autograd)
# ---------------------------------------------------------------------------

@torch.no_grad()
def vtheta_value_and_grad(
    model: torch.nn.Module,
    h_ell: torch.Tensor,
    layer_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate V_theta and its analytical gradient at layer `layer_idx`.

    Returns (V, grad_V) where V: (B, T) and grad_V: (B, T, d).
    """
    xis = model.xi_module(h_ell)
    model.V_theta.set_active_layer(layer_idx)
    V = model.V_theta(xis, h_ell).squeeze(-1)         # (B, T)
    grad_V = model.V_theta.analytical_grad(xis, h_ell)  # (B, T, d)
    return V, grad_V


# ---------------------------------------------------------------------------
# Conformal-factor gradient and Gamma(v,v) contraction
# ---------------------------------------------------------------------------

def conformal_grad(
    grad_V: torch.Tensor,
    E_minus_V: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Gradient of the Jacobi conformal factor phi.

    phi = 0.5 * log(2*(E - V_theta))
    => d_i phi = -d_i V_theta / (2*(E - V_theta))

    Args:
        grad_V: (B, T, d) — analytical gradient of V_theta w.r.t. h
        E_minus_V: (B, T) — E - V_theta(h), clamped > 0 externally

    Returns:
        phi_grad: (B, T, d)
    """
    denom = 2.0 * E_minus_V.unsqueeze(-1).clamp(min=epsilon)
    return -grad_V / denom


def christoffel_vv(
    phi_grad: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Contract Christoffel symbols with velocity: Gamma^k_ij v^i v^j.

    For a conformally flat metric g_ij = exp(2*phi) * delta_ij:
      Gamma(v,v)^k = 2 (grad_phi . v) v^k  -  ||v||^2 grad_phi^k

    Args:
        phi_grad: (B, T, d)
        v: (B, T, d)

    Returns:
        Gamma_vv: (B, T, d)
    """
    phi_dot_v = (phi_grad * v).sum(dim=-1, keepdim=True)  # (B, T, 1)
    v_sq = (v * v).sum(dim=-1, keepdim=True)              # (B, T, 1)
    return 2.0 * phi_dot_v * v - v_sq * phi_grad


# ---------------------------------------------------------------------------
# Core: compute R_ell per layer
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_residual(
    traj: List[torch.Tensor],
    model: torch.nn.Module,
    gamma_eval: float,
    device: str,
    epsilon: float = 1e-6,
    ref_layer: int = 0,
) -> Dict:
    """Compute the damped-geodesic residual for one checkpoint.

    Args:
        traj: list of L+1 tensors (B, T, d) on CPU
        model: loaded FockPARFLM (eval mode, on `device`)
        gamma_eval: damping coefficient for the residual equation
        device: cuda device string
        epsilon: threshold for classically-allowed region
        ref_layer: reference layer for per-token energy E

    Returns dict with keys:
        R_bar: float — mean residual over layers and tokens
        per_layer_R: list[float] — per-layer mean residual
        excluded_frac: float — fraction of tokens excluded (E - V < eps)
        gamma_geo: float — closed-form best-fit gamma (section 4.2)
    """
    L = len(traj) - 1  # number of layers

    # Compute reference energy E at ref_layer (per-token):
    #   v_ref = h[ref+1] - h[ref]
    #   E = 0.5 * ||v_ref||^2 + V_theta(h[ref])
    h_ref = traj[ref_layer].to(device)
    v_ref = (traj[ref_layer + 1].to(device) - h_ref)
    KE_ref = 0.5 * (v_ref * v_ref).sum(dim=-1)   # (B, T)
    V_ref, _ = vtheta_value_and_grad(model, h_ref, ref_layer)
    E = KE_ref + V_ref                             # (B, T)
    del h_ref, v_ref, KE_ref, V_ref

    per_layer_R = []
    total_excluded = 0
    total_tokens = 0

    # Accumulators for gamma_geo closed form:
    #   gamma_geo = - <a + Gamma(v,v), v> / ||v||^2
    num_sum = 0.0
    den_sum = 0.0

    for ell in range(1, L):
        h_prev = traj[ell - 1].to(device)
        h_curr = traj[ell].to(device)
        h_next = traj[ell + 1].to(device)

        v_ell = h_curr - h_prev         # velocity at layer ell
        v_next = h_next - h_curr        # velocity at layer ell+1
        a_ell = v_next - v_ell           # acceleration

        V_ell, grad_V_ell = vtheta_value_and_grad(model, h_curr, ell)

        E_minus_V = E - V_ell            # (B, T)
        allowed = E_minus_V > epsilon     # classically allowed mask
        n_excluded = int((~allowed).sum().item())
        n_total = allowed.numel()
        total_excluded += n_excluded
        total_tokens += n_total

        phi_g = conformal_grad(grad_V_ell, E_minus_V, epsilon)
        Gamma_vv = christoffel_vv(phi_g, v_ell)

        # Residual: || a + Gamma(v,v) + gamma*v || / (||a|| + eps)
        residual_vec = a_ell + Gamma_vv + gamma_eval * v_ell
        residual_norm = residual_vec.norm(dim=-1)     # (B, T)
        a_norm = a_ell.norm(dim=-1)                   # (B, T)
        R_ell = residual_norm / (a_norm + epsilon)    # (B, T)

        # Mask out excluded tokens
        R_ell = R_ell * allowed.float()
        n_allowed = allowed.float().sum().clamp(min=1.0)
        mean_R = R_ell.sum() / n_allowed

        per_layer_R.append(float(mean_R.item()))

        # gamma_geo accumulators (over allowed tokens only):
        #   gamma_geo = - <a + Gamma(v,v), v> / ||v||^2
        a_plus_Gamma = a_ell + Gamma_vv
        dot_num = (a_plus_Gamma * v_ell).sum(dim=-1)  # (B, T)
        v_sq = (v_ell * v_ell).sum(dim=-1)             # (B, T)
        num_sum += float((dot_num * allowed.float()).sum().item())
        den_sum += float((v_sq * allowed.float()).sum().item())

        del h_prev, h_curr, h_next, v_ell, v_next, a_ell
        del V_ell, grad_V_ell, phi_g, Gamma_vv

    R_bar = float(np.mean(per_layer_R)) if per_layer_R else float("inf")
    excluded_frac = total_excluded / max(total_tokens, 1)
    gamma_geo = -num_sum / max(den_sum, 1e-12)

    return {
        "R_bar": R_bar,
        "per_layer_R": per_layer_R,
        "excluded_frac": excluded_frac,
        "gamma_geo": gamma_geo,
    }


# ---------------------------------------------------------------------------
# Null controls (section 6)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_null_controls(
    traj: List[torch.Tensor],
    model: torch.nn.Module,
    gamma_eval: float,
    device: str,
    epsilon: float = 1e-6,
) -> Dict[str, float]:
    """Run shuffled-Gamma and random-v null controls on one checkpoint.

    Both should produce large R_bar; if they don't, the implementation
    has a normalisation bug.
    """
    L = len(traj) - 1

    h_ref = traj[0].to(device)
    v_ref = (traj[1].to(device) - h_ref)
    KE_ref = 0.5 * (v_ref * v_ref).sum(dim=-1)
    V_ref, _ = vtheta_value_and_grad(model, h_ref, 0)
    E = KE_ref + V_ref
    del h_ref, v_ref, KE_ref, V_ref

    R_shuffled_layers = []
    R_random_layers = []

    for ell in range(1, L):
        h_prev = traj[ell - 1].to(device)
        h_curr = traj[ell].to(device)
        h_next = traj[ell + 1].to(device)

        v_ell = h_curr - h_prev
        v_next = h_next - h_curr
        a_ell = v_next - v_ell

        V_ell, grad_V_ell = vtheta_value_and_grad(model, h_curr, ell)
        E_minus_V = E - V_ell
        allowed = E_minus_V > epsilon
        n_allowed = allowed.float().sum().clamp(min=1.0)

        # --- Shuffled-Gamma null: use Gamma from a DIFFERENT token's geometry.
        # Roll the token dimension by a random offset so each token's
        # Christoffel symbols come from a different token's local potential.
        B, T, d = h_curr.shape
        shift = max(1, T // 3)
        grad_V_shuffled = grad_V_ell.roll(shifts=shift, dims=1)
        E_minus_V_shuffled = E_minus_V.roll(shifts=shift, dims=1)
        phi_g_shuf = conformal_grad(grad_V_shuffled, E_minus_V_shuffled, epsilon)
        Gamma_vv_shuf = christoffel_vv(phi_g_shuf, v_ell)
        res_shuf = (a_ell + Gamma_vv_shuf + gamma_eval * v_ell).norm(dim=-1)
        a_norm = a_ell.norm(dim=-1)
        R_shuf = res_shuf / (a_norm + epsilon)
        R_shuf = (R_shuf * allowed.float()).sum() / n_allowed
        R_shuffled_layers.append(float(R_shuf.item()))

        # --- Random-v null: replace v with a norm-matched random vector.
        v_norm = v_ell.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        v_random = torch.randn_like(v_ell)
        v_random = F.normalize(v_random, dim=-1) * v_norm

        phi_g = conformal_grad(grad_V_ell, E_minus_V, epsilon)
        Gamma_vv_rand = christoffel_vv(phi_g, v_random)
        res_rand = (a_ell + Gamma_vv_rand + gamma_eval * v_random).norm(dim=-1)
        R_rand = res_rand / (a_norm + epsilon)
        R_rand = (R_rand * allowed.float()).sum() / n_allowed
        R_random_layers.append(float(R_rand.item()))

        del h_prev, h_curr, h_next

    return {
        "R_bar_shuffled_gamma": float(np.mean(R_shuffled_layers)),
        "R_bar_random_v": float(np.mean(R_random_layers)),
    }


# ---------------------------------------------------------------------------
# Discover gamma directories in sweep_dir
# ---------------------------------------------------------------------------

def discover_gamma_dirs(sweep_dir: str) -> List[Tuple[float, str]]:
    """Find gamma_X.XXX/checkpoints/ckpt_best.pt under sweep_dir.

    Returns sorted list of (gamma_value, ckpt_path).
    """
    pattern = os.path.join(sweep_dir, "gamma_*/checkpoints/ckpt_best.pt")
    hits = sorted(glob.glob(pattern))
    results = []
    for path in hits:
        gamma_dir = Path(path).parent.parent.name
        try:
            gamma_val = float(gamma_dir.replace("gamma_", ""))
            results.append((gamma_val, path))
        except ValueError:
            print(f"  [warn] Skipping unrecognised dir: {gamma_dir}")
    results.sort(key=lambda x: x[0])
    return results


# ---------------------------------------------------------------------------
# Read sweep_summary.json for PPL data
# ---------------------------------------------------------------------------

def load_sweep_ppl(sweep_dir: str) -> Dict[float, float]:
    """Load PPL results from sweep_summary.json."""
    summary_path = os.path.join(sweep_dir, "sweep_summary.json")
    if not os.path.exists(summary_path):
        return {}
    with open(summary_path) as f:
        data = json.load(f)
    return {r["gamma"]: r["best_ppl"] for r in data.get("results", [])}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_overlay(
    results: List[Dict],
    output_path: str,
    title: str = "",
):
    """Dual-axis overlay: PPL(gamma) and R_bar(gamma)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gammas = [r["gamma_train"] for r in results]
    ppls = [r["val_ppl"] for r in results]
    r_bars = [r["R_bar"] for r in results]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_ppl = "#2563eb"
    color_r = "#dc2626"

    ax1.set_xlabel(r"$\gamma_{\mathrm{train}}$", fontsize=13)
    ax1.set_ylabel("Perplexity (PPL)", color=color_ppl, fontsize=13)
    ax1.plot(gammas, ppls, "o-", color=color_ppl, linewidth=2,
             markersize=7, label="PPL")
    ax1.tick_params(axis="y", labelcolor=color_ppl)

    ax2 = ax1.twinx()
    ax2.set_ylabel(r"$\bar{R}(\gamma)$  (geodesic residual)",
                    color=color_r, fontsize=13)
    ax2.plot(gammas, r_bars, "s--", color=color_r, linewidth=2,
             markersize=7, label=r"$\bar{R}$")
    ax2.tick_params(axis="y", labelcolor=color_r)

    # Mark minima
    best_ppl_idx = int(np.argmin(ppls))
    best_r_idx = int(np.argmin(r_bars))
    ax1.axvline(gammas[best_ppl_idx], color=color_ppl, linestyle=":",
                alpha=0.5, label=f"PPL min @ {gammas[best_ppl_idx]:.3f}")
    ax2.axvline(gammas[best_r_idx], color=color_r, linestyle=":",
                alpha=0.5, label=rf"$\bar{{R}}$ min @ {gammas[best_r_idx]:.3f}")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
               fontsize=10)

    default_title = r"PPL vs Geodesic Residual $\bar{R}(\gamma)$"
    ax1.set_title(title or default_title, fontsize=14, pad=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Overlay figure saved: {output_path}")


def plot_heatmap(
    heatmap_data: Dict,
    output_path: str,
):
    """Off-diagonal heatmap R_bar(gamma_eval; theta_{gamma_train})."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_gammas = sorted(heatmap_data.keys())
    eval_gammas = sorted(heatmap_data[train_gammas[0]].keys())
    matrix = np.array([
        [heatmap_data[gt][ge] for ge in eval_gammas]
        for gt in train_gammas
    ])

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(eval_gammas)))
    ax.set_xticklabels([f"{g:.2f}" for g in eval_gammas], rotation=45)
    ax.set_yticks(range(len(train_gammas)))
    ax.set_yticklabels([f"{g:.3f}" for g in train_gammas])
    ax.set_xlabel(r"$\gamma_{\mathrm{eval}}$", fontsize=13)
    ax.set_ylabel(r"$\gamma_{\mathrm{train}}$ (checkpoint)", fontsize=13)
    ax.set_title(
        r"$\bar{R}(\gamma_{\mathrm{eval}};\;\theta_{\gamma_{\mathrm{train}}})$",
        fontsize=14, pad=12,
    )
    fig.colorbar(im, ax=ax, label=r"$\bar{R}$")

    # Mark diagonal
    for i in range(min(len(train_gammas), len(eval_gammas))):
        gt = train_gammas[i]
        if gt in eval_gammas:
            j = eval_gammas.index(gt)
            ax.plot(j, i, "wx", markersize=10, markeredgewidth=2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap figure saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Geodesic residual analysis for Fock-PARFLM gamma sweeps")
    parser.add_argument("--sweep_dir", type=str, required=True,
                        help="Path to gamma_sweep directory containing "
                             "gamma_X.XXX/ subdirs with checkpoints")
    parser.add_argument("--preset", type=str, default="sweep-d768",
                        choices=list(PRESETS.keys()),
                        help="Model preset (must match the sweep)")
    parser.add_argument("--data_dir", type=str, default="",
                        help="Directory containing cached OWT .npy files")
    parser.add_argument("--output_dir", type=str, default="",
                        help="Where to write results JSON and figures")
    parser.add_argument("--n_batches", type=int, default=10,
                        help="Number of validation batches to average over")
    parser.add_argument("--batch_size", type=int, default=0,
                        help="Override batch size (0 = use preset)")
    parser.add_argument("--eval_gammas", type=str, default="",
                        help="Comma-separated gamma values for off-diagonal "
                             "heatmap (empty = diagonal only)")
    parser.add_argument("--controls", action="store_true",
                        help="Run shuffled-Gamma and random-v null controls")
    parser.add_argument("--bf16", type=lambda x: x.lower() == "true",
                        default=False,
                        help="Use bf16 for model inference")
    parser.add_argument("--epsilon", type=float, default=1e-6,
                        help="Threshold for classically-allowed region")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for validation batches")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Config from preset ──
    cfg = TrainConfig()
    if args.preset in PRESETS:
        for k, v in PRESETS[args.preset].items():
            setattr(cfg, k, v)
    cfg.resolve_xi_override()
    cfg.resolve_wsd_lr_floor()

    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.batch_size > 0:
        cfg.batch_size = args.batch_size

    if not args.output_dir:
        args.output_dir = os.path.join(
            os.path.dirname(args.sweep_dir), "geodesic_analysis")
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}  ({props.total_memory / 1e9:.1f} GB)")

    # ── Paths ──
    script_dir = SCRIPT_DIR
    ca_dir = script_dir.parent
    for sub in ["", "parf", "multixi", "scaleup",
                "sarf_mass_variant", "energetic_minima"]:
        d = str(ca_dir / sub) if sub else str(ca_dir)
        if d not in sys.path:
            sys.path.insert(0, d)

    if not cfg.data_dir:
        cfg.data_dir = str(ca_dir / "data")
    Path(cfg.data_dir).mkdir(parents=True, exist_ok=True)

    cfg.output_dir = args.output_dir
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.output_dir) / "checkpoints").mkdir(parents=True, exist_ok=True)

    # ── Data ──
    print("Loading validation data...")
    train_ids, val_ids = load_data(cfg)
    logfreq_path = ensure_logfreq(train_ids, cfg, ca_dir)

    # ── Discover checkpoints ──
    gamma_dirs = discover_gamma_dirs(args.sweep_dir)
    if not gamma_dirs:
        print(f"ERROR: No gamma_*/checkpoints/ckpt_best.pt found in "
              f"{args.sweep_dir}")
        sys.exit(1)

    ppl_from_summary = load_sweep_ppl(args.sweep_dir)

    print(f"\n{'='*60}")
    print(f"  GEODESIC RESIDUAL ANALYSIS")
    print(f"  Preset: {args.preset}  d={cfg.d}  L={cfg.L}")
    print(f"  Sweep dir: {args.sweep_dir}")
    print(f"  Checkpoints found: {len(gamma_dirs)}")
    print(f"  Gammas: {[g for g, _ in gamma_dirs]}")
    print(f"  Validation batches: {args.n_batches}")
    print(f"  Controls: {'yes' if args.controls else 'no'}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*60}\n")

    # ── Fixed validation batches (same for ALL checkpoints — section 5b) ──
    rng = np.random.default_rng(args.seed)
    val_batches = []
    for _ in range(args.n_batches):
        xb, _ = get_batch(val_ids, cfg.batch_size, cfg.block_size, rng)
        val_batches.append(torch.from_numpy(xb))

    # ── Off-diagonal eval gammas ──
    eval_gammas: Optional[List[float]] = None
    if args.eval_gammas:
        eval_gammas = [float(x) for x in args.eval_gammas.split(",")]

    # ── Main loop ──
    results = []
    heatmap_data: Dict[float, Dict[float, float]] = {}

    for gi, (gamma_val, ckpt_path) in enumerate(gamma_dirs):
        t0 = time.time()
        print(f"--- [{gi+1}/{len(gamma_dirs)}] gamma={gamma_val:.3f} ---")
        print(f"  Loading: {ckpt_path}")

        import copy
        cfg_copy = copy.deepcopy(cfg)
        model, model_cfg, gamma_train, val_ppl = load_checkpoint_model(
            ckpt_path, cfg_copy, device, logfreq_path,
        )

        # Use PPL from summary if available (more accurate than ckpt snapshot)
        if gamma_val in ppl_from_summary:
            val_ppl = ppl_from_summary[gamma_val]

        # Collect trajectories and compute residual, averaged over batches
        batch_results = []
        for bi, x_batch in enumerate(val_batches):
            x = x_batch.to(device)
            traj = collect_trajectory(model, x)
            res = compute_residual(
                traj, model, gamma_train, device,
                epsilon=args.epsilon,
            )
            batch_results.append(res)
            del traj

        R_bar = float(np.mean([r["R_bar"] for r in batch_results]))
        gamma_geo = float(np.mean([r["gamma_geo"] for r in batch_results]))
        excluded_frac = float(np.mean(
            [r["excluded_frac"] for r in batch_results]))
        per_layer_R = [
            float(np.mean([br["per_layer_R"][i]
                           for br in batch_results]))
            for i in range(len(batch_results[0]["per_layer_R"]))
        ]

        entry = {
            "gamma_train": gamma_train,
            "val_ppl": val_ppl,
            "R_bar": R_bar,
            "gamma_geo": gamma_geo,
            "excluded_frac": excluded_frac,
            "per_layer_R": per_layer_R,
        }

        # Null controls
        if args.controls:
            null_results = []
            for x_batch in val_batches[:3]:
                x = x_batch.to(device)
                traj = collect_trajectory(model, x)
                nulls = compute_null_controls(
                    traj, model, gamma_train, device,
                    epsilon=args.epsilon,
                )
                null_results.append(nulls)
                del traj
            entry["R_bar_shuffled_gamma"] = float(np.mean(
                [n["R_bar_shuffled_gamma"] for n in null_results]))
            entry["R_bar_random_v"] = float(np.mean(
                [n["R_bar_random_v"] for n in null_results]))

        # Off-diagonal heatmap
        if eval_gammas is not None:
            heatmap_data[gamma_train] = {}
            x = val_batches[0].to(device)
            traj = collect_trajectory(model, x)
            for ge in eval_gammas:
                res_ge = compute_residual(
                    traj, model, ge, device,
                    epsilon=args.epsilon,
                )
                heatmap_data[gamma_train][ge] = res_ge["R_bar"]
            del traj

        results.append(entry)
        elapsed = time.time() - t0
        print(f"  R_bar={R_bar:.4f}  gamma_geo={gamma_geo:.4f}  "
              f"excluded={excluded_frac:.4f}  PPL={val_ppl:.2f}  "
              f"({elapsed:.1f}s)")
        if args.controls:
            print(f"  Nulls: shuffled={entry.get('R_bar_shuffled_gamma', 0):.4f}"
                  f"  random_v={entry.get('R_bar_random_v', 0):.4f}")

        del model, model_cfg
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Summary ──
    results.sort(key=lambda r: r["gamma_train"])

    print(f"\n{'='*60}")
    print(f"  GEODESIC RESIDUAL RESULTS  d={cfg.d}  L={cfg.L}")
    print(f"{'='*60}")
    hdr = (f"  {'gamma':>8s}  {'PPL':>8s}  {'R_bar':>8s}  "
           f"{'gamma_geo':>10s}  {'excl%':>6s}")
    print(hdr)
    print(f"  {'-----':>8s}  {'---':>8s}  {'-----':>8s}  "
           f"{'--------':>10s}  {'-----':>6s}")
    best_r_idx = int(np.argmin([r["R_bar"] for r in results]))
    best_ppl_idx = int(np.argmin([r["val_ppl"] for r in results]))
    for i, r in enumerate(results):
        markers = []
        if i == best_r_idx:
            markers.append("R*")
        if i == best_ppl_idx:
            markers.append("PPL*")
        marker = "  <-- " + ", ".join(markers) if markers else ""
        print(f"  {r['gamma_train']:8.3f}  {r['val_ppl']:8.2f}  "
              f"{r['R_bar']:8.4f}  {r['gamma_geo']:10.4f}  "
              f"{r['excluded_frac']*100:5.1f}%{marker}")

    coincidence = (results[best_r_idx]["gamma_train"]
                   == results[best_ppl_idx]["gamma_train"])
    if coincidence:
        print(f"\n  >>> MINIMA COINCIDE at gamma="
              f"{results[best_r_idx]['gamma_train']:.3f}")
        print(f"      Mechanistic claim supported: the damping that minimises")
        print(f"      PPL also minimises the geodesic residual.")
    else:
        print(f"\n  >>> PPL minimum at gamma="
              f"{results[best_ppl_idx]['gamma_train']:.3f}, "
              f"R_bar minimum at gamma="
              f"{results[best_r_idx]['gamma_train']:.3f}")
        print(f"      Minima do NOT coincide (gap = "
              f"{abs(results[best_ppl_idx]['gamma_train'] - results[best_r_idx]['gamma_train']):.3f})")

    # ── Save results ──
    json_path = os.path.join(args.output_dir, "geodesic_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "d": cfg.d, "L": cfg.L,
            "preset": args.preset,
            "n_batches": args.n_batches,
            "seed": args.seed,
            "epsilon": args.epsilon,
            "results": results,
        }, f, indent=2)
    print(f"\n  Results saved: {json_path}")

    # ── Plot overlay ──
    overlay_path = os.path.join(args.output_dir, "geodesic_overlay.png")
    plot_overlay(
        results, overlay_path,
        title=f"d={cfg.d}  L={cfg.L}  |  PPL vs Geodesic Residual",
    )

    # ── Plot heatmap (if off-diagonal was computed) ──
    if heatmap_data:
        heatmap_path = os.path.join(args.output_dir, "geodesic_heatmap.png")
        plot_heatmap(heatmap_data, heatmap_path)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
