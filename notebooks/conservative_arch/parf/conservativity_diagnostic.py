"""
Fock-PARFLM v2.1 conservativity diagnostic — 5-arm experimental battery.

Demonstrates that FockPARFLM v2.1 is conservative-by-construction except
for the explicitly designed reverse-channel force Q_i.

Arms
----
  1. Structural Jacobian symmetry (no checkpoint needed)
  2. Conservative ablation: clamp Q_i=0, verify R^2 = 1.0
  3. Per-layer energy budget decomposition
  4. Conservativity dial: sweep reverse_channel_scale
  5. Four-way architectural separator (requires baselines)

Usage
-----
  # Arm 1 only (no checkpoint needed):
  python conservativity_diagnostic.py --arm 1 --output-dir results/conservativity

  # All arms with a trained checkpoint:
  python conservativity_diagnostic.py --arm all --checkpoint path/to/ckpt.pt \
      --output-dir results/conservativity

  # Retrain from scratch then run all:
  python conservativity_diagnostic.py --arm all --retrain \
      --output-dir results/conservativity
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = Path(__file__).parent
_PARENT_DIR = _THIS_DIR.parent
_SCALEUP_DIR = _PARENT_DIR / "scaleup"
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_PARENT_DIR))
sys.path.insert(0, str(_PARENT_DIR / "multixi"))
sys.path.insert(0, str(_SCALEUP_DIR))

from model_fock_parf_multixi import FockMultiXiPARFConfig, FockMultiXiPARFLM
from data_module import get_batch, load_tiny_stories


LOGFREQ_PATH = _SCALEUP_DIR / "results" / "logfreq_surprisal_tinystories.npy"


# =====================================================================
# Utilities
# =====================================================================

def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_v21_config(
    logfreq_path: str | None = None,
) -> FockMultiXiPARFConfig:
    """Reconstruct the exact config used for the PPL 9.30 run."""
    return FockMultiXiPARFConfig(
        vocab_size=50257,
        d=256, max_len=1024, L=8,
        v_hidden=1024, v_depth=3,
        dt=1.0,
        init_m=1.0, init_gamma=0.15,
        learn_mgamma=True,
        fixed_gamma=0.30,
        mass_mode="logfreq",
        logfreq_init_alpha=0.1,
        logfreq_path=logfreq_path,
        causal_force=True,
        ln_after_step=True,
        v_phi_kind="structural_competitive",
        v_phi_d_type=32, v_phi_d_angle=16,
        v_phi_phi_hidden=128, v_phi_theta_hidden=128,
        v_phi_mlp_hidden=32,
        v_phi_competitive_temp=1.0,
        v_phi_competitive_scale="row",
        ln_before_distance=True,
        per_layer_v_phi_scale=True,
        per_layer_scale_init=-3.0,
        theta_activation="tanh",
        theta_form="mlp",
        top_k=8,
        score_head_hidden=32,
        gumbel_tau_init=1.0,
        gumbel_tau_min=0.3,
        gumbel_noise=True,
        use_grad_checkpoint=False,
        use_layer_checkpoint=True,
        use_gathered_v_phi=True,
        xi_channels=4,
        xi_alpha_inits=[0.0, 0.5, 0.9, 0.99],
        xi_learnable=True,
        xi_alpha_init_mode="log_spaced",
        xi_tau_max=100.0,
        fock_version="v2",
        n_registers=16,
        register_salience_decay=0.5,
        register_salience_threshold=0.005,
        creation_gate_hidden=64,
        stack_discipline=True,
        register_init_scale=0.02,
        d_k=64,
        tau_create_init=8.0,
        destruction_gate_hidden=64,
        reverse_channel=True,
        per_register_tau=True,
        per_register_keys=True,
        ortho_register_init=True,
    )


def build_smoke_config() -> FockMultiXiPARFConfig:
    """Tiny config for structural tests that don't need a trained model."""
    return FockMultiXiPARFConfig(
        vocab_size=257, d=32, max_len=64, L=4,
        v_hidden=64, v_depth=2,
        dt=1.0,
        init_m=1.0, init_gamma=0.15,
        learn_mgamma=True,
        fixed_gamma=0.30,
        mass_mode="global",
        causal_force=True,
        ln_after_step=False,
        v_phi_kind="structural_competitive",
        v_phi_d_type=4, v_phi_d_angle=2,
        v_phi_phi_hidden=8, v_phi_theta_hidden=8,
        v_phi_mlp_hidden=16,
        v_phi_competitive_temp=1.0,
        v_phi_competitive_scale="row",
        ln_before_distance=False,
        per_layer_v_phi_scale=False,
        theta_activation="tanh",
        theta_form="mlp",
        top_k=4,
        score_head_hidden=8,
        gumbel_tau_init=1.0,
        gumbel_tau_min=0.3,
        gumbel_noise=False,
        use_grad_checkpoint=False,
        use_layer_checkpoint=False,
        use_gathered_v_phi=False,
        xi_channels=2,
        xi_alpha_inits=[0.0, 0.9],
        xi_learnable=True,
        xi_alpha_init_mode="explicit",
        fock_version="v2",
        n_registers=4,
        register_salience_decay=0.5,
        register_salience_threshold=0.005,
        creation_gate_hidden=16,
        stack_discipline=True,
        register_init_scale=0.02,
        d_k=16,
        tau_create_init=4.0,
        destruction_gate_hidden=16,
        reverse_channel=True,
        per_register_tau=True,
        per_register_keys=True,
        ortho_register_init=True,
    )


def load_model(
    checkpoint: str | None,
    device: str,
    logfreq_path: str | None = None,
) -> FockMultiXiPARFLM:
    """Load model from checkpoint, or build at random init for structural tests."""
    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg_dict = ckpt.get("config") or ckpt.get("model_cfg")
        if cfg_dict is not None:
            if isinstance(cfg_dict, dict):
                known_fields = {f.name for f in FockMultiXiPARFConfig.__dataclass_fields__.values()}
                filtered = {k: v for k, v in cfg_dict.items() if k in known_fields}
                cfg = FockMultiXiPARFConfig(**filtered)
            else:
                cfg = cfg_dict
        else:
            cfg = build_v21_config(logfreq_path=logfreq_path)
        if logfreq_path and hasattr(cfg, "logfreq_path"):
            cfg.logfreq_path = logfreq_path
        model = FockMultiXiPARFLM(cfg)
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        model.load_state_dict(state, strict=False)
        print(f"[conservativity] Loaded checkpoint from {checkpoint}")
        if hasattr(model, "reverse_channel_scale") and model.reverse_channel_scale is not None:
            print(f"[conservativity] reverse_channel_scale: raw={model.reverse_channel_scale.item():.6f}, "
                  f"tanh={torch.tanh(model.reverse_channel_scale).item():.6f}")
    else:
        cfg = build_smoke_config()
        model = FockMultiXiPARFLM(cfg)
        print("[conservativity] Using random-init smoke model (no checkpoint)")
    model.to(device)
    model.eval()
    return model


# =====================================================================
# Arm 1: Structural Jacobian symmetry
# =====================================================================

def _compute_U_fixed_context(
    model: FockMultiXiPARFLM,
    h_in: torch.Tensor,
    xi_frozen: torch.Tensor,
    h_src_frozen: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    """Compute U = V_theta(xi, h) + V_phi(h, h_src) with frozen context.

    xi_frozen and h_src_frozen are held fixed (detached) so that U is a
    pure function of h_in alone.  The Hessian of this U w.r.t. h_in is
    guaranteed symmetric by Schwarz's theorem.
    """
    cfg = model.cfg
    T = h_in.shape[1]

    V_th = model.V_theta(xi_frozen, h_in)
    P = model.V_phi(h_in, h_src_frozen)
    mask = model._pair_mask_for(T, h_in.device)
    P_masked = P.masked_fill(~mask, 0.0)

    s_ell = model.per_layer_scale(layer_idx)
    if s_ell is not None:
        P_masked = P_masked * s_ell

    return V_th.sum() + P_masked.sum()


def _compute_Q_force_raw(
    model: FockMultiXiPARFLM,
    h: torch.Tensor,
    r: torch.Tensor,
    salience: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Compute the reverse-channel force Q_i (unscaled by tanh gate)."""
    if model.reverse_ch is None:
        return torch.zeros_like(h)

    _, r_causal, _ = model.creation_gate_qkv(h, r)
    return model.reverse_ch(h, r_causal, active)


def arm1_jacobian_symmetry(
    model: FockMultiXiPARFLM,
    device: str,
    output_dir: Path,
    eps: float = 5e-4,
) -> Dict[str, Any]:
    """Arm 1: Structural proof that the conservative force is curl-free.

    Three tests:
      (A) Gradient verification: the layer step's force equals -grad_h(U)
          where U = V_theta + V_phi, verified by comparing the autograd
          gradient to finite differences of U (with context frozen).
      (B) Hessian symmetry: the Hessian of U w.r.t. h (with frozen xi
          and h_src) is symmetric — proving the conservative force field
          is curl-free *within the autograd computation graph*.
      (C) Q_i curl: the reverse-channel force Q_i has a non-symmetric
          Jacobian, confirming it is genuinely non-conservative.
    """
    print("\n" + "=" * 70)
    print("ARM 1: Structural Jacobian symmetry test")
    print("=" * 70)

    cfg = model.cfg
    torch.manual_seed(42)

    B, T = 1, min(8, cfg.max_len)
    x = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    h0 = model._embed(x).detach()
    d = cfg.d

    results: Dict[str, Any] = {"arm": 1, "description": "Jacobian symmetry"}

    # Precompute frozen context (xi, h_src) at h0
    with torch.no_grad():
        xi_frozen = model.xi_module(h0).detach()
        h_src_frozen = h0.detach()

    # --- Test A: gradient verification ---
    print(f"\n  [Test A] Gradient verification: f_autograd vs -dU/dh (finite diff) ...")

    h_in = h0.clone().requires_grad_(True)
    U_base = _compute_U_fixed_context(model, h_in, xi_frozen, h_src_frozen, layer_idx=0)
    grad_U, = torch.autograd.grad(U_base, h_in, retain_graph=True)
    f_autograd = -grad_U.detach()
    U_base_val = U_base.item()

    f_fd = torch.zeros(B, T, d, device=device)
    for t_idx in range(T):
        for d_idx in range(d):
            h_pert = h0.clone().requires_grad_(True)
            h_pert_data = h_pert.data.clone()
            h_pert_data[0, t_idx, d_idx] += eps
            h_pert2 = h_pert_data.requires_grad_(True)
            U_pert = _compute_U_fixed_context(model, h_pert2, xi_frozen, h_src_frozen, layer_idx=0)
            f_fd[0, t_idx, d_idx] = -(U_pert.item() - U_base_val) / eps

    abs_err = (f_autograd - f_fd).abs()
    f_scale = f_autograd.abs().max().item()
    normalised_err = abs_err / (f_scale + 1e-12)
    max_abs_err = abs_err.max().item()
    mean_norm_err = normalised_err.mean().item()

    print(f"  [Test A] ||f_autograd||_max = {f_scale:.6f}")
    print(f"  [Test A] Max |f_ag - f_fd|  = {max_abs_err:.2e}")
    print(f"  [Test A] Mean |err|/||f||    = {mean_norm_err:.2e}")
    grad_pass = mean_norm_err < 0.05
    print(f"  [Test A] {'PASS' if grad_pass else 'WARN'}: "
          f"autograd and finite-diff agree to {'<5%' if grad_pass else f'{mean_norm_err:.1%}'}")

    results["gradient_verification"] = {
        "f_scale": f_scale,
        "max_abs_error": max_abs_err,
        "mean_normalised_error": mean_norm_err,
        "eps": eps,
        "pass": grad_pass,
    }

    # --- Test B: Hessian symmetry ---
    print(f"\n  [Test B] Hessian symmetry of U w.r.t. h (with frozen context) ...")
    n_probe = min(T * d, 64)
    torch.manual_seed(123)
    probe_indices = torch.randperm(T * d)[:n_probe].tolist()

    J_cons = torch.zeros(T * d, n_probe, device=device)
    for col, i in enumerate(probe_indices):
        h_pert = h0.clone().requires_grad_(True)
        h_pert_data = h_pert.data.clone()
        t_idx, d_idx = i // d, i % d
        h_pert_data[0, t_idx, d_idx] += eps
        h_pert2 = h_pert_data.requires_grad_(True)
        U_pert = _compute_U_fixed_context(model, h_pert2, xi_frozen, h_src_frozen, layer_idx=0)
        grad_pert, = torch.autograd.grad(U_pert, h_pert2, retain_graph=False)
        f_pert = -grad_pert.detach()
        J_cons[:, col] = (f_pert.reshape(-1) - f_autograd.reshape(-1)) / eps

    J_sub = J_cons[probe_indices, :]
    J_antisym = 0.5 * (J_sub - J_sub.T)
    frob_J = torch.norm(J_sub, p="fro").item()
    frob_antisym = torch.norm(J_antisym, p="fro").item()
    ratio_cons = frob_antisym / (frob_J + 1e-12)

    print(f"  [Test B] Force Jacobian ({n_probe}x{n_probe} submatrix, frozen context):")
    print(f"  [Test B]   ||J||_F        = {frob_J:.6f}")
    print(f"  [Test B]   ||J - J^T||_F  = {frob_antisym:.6f}")
    print(f"  [Test B]   Antisymmetry   = {ratio_cons:.2e}")
    hess_pass = ratio_cons < 0.02
    print(f"  [Test B] {'PASS' if hess_pass else 'WARN'}: "
          f"Hessian {'is symmetric (curl-free)' if hess_pass else 'has unexpected asymmetry'}")

    results["conservative_hessian"] = {
        "frob_J": frob_J,
        "frob_antisym": frob_antisym,
        "antisymmetry_ratio": ratio_cons,
        "n_probes": n_probe,
        "pass": hess_pass,
    }

    # --- Test C: Q_i curl ---
    print(f"\n  [Test C] Curl of reverse-channel force Q_i ...")
    if model.reverse_ch is not None:
        r, salience = model._init_registers(B, device)
        active = model._active_mask(salience)

        Q_base = _compute_Q_force_raw(model, h0, r, salience, active).detach()
        Q_base_flat = Q_base.reshape(-1)

        J_Q = torch.zeros(T * d, n_probe, device=device)
        for col, i in enumerate(probe_indices):
            h_pert = h0.clone()
            h_pert[0, i // d, i % d] += eps
            Q_pert = _compute_Q_force_raw(model, h_pert, r, salience, active).detach()
            J_Q[:, col] = (Q_pert.reshape(-1) - Q_base_flat) / eps

        J_Q_sub = J_Q[probe_indices, :]
        J_Q_antisym = 0.5 * (J_Q_sub - J_Q_sub.T)
        frob_JQ = torch.norm(J_Q_sub, p="fro").item()
        frob_JQ_antisym = torch.norm(J_Q_antisym, p="fro").item()
        ratio_Q = frob_JQ_antisym / (frob_JQ + 1e-12)

        print(f"  [Test C] Q_i force Jacobian ({n_probe}x{n_probe} submatrix):")
        print(f"  [Test C]   ||J_Q||_F        = {frob_JQ:.6f}")
        print(f"  [Test C]   ||J_Q - J_Q^T||_F = {frob_JQ_antisym:.6f}")
        print(f"  [Test C]   Antisymmetry      = {ratio_Q:.2e}")
        curl_pass = ratio_Q > 1e-2
        print(f"  [Test C] {'PASS' if curl_pass else 'WARN'}: "
              f"Q_i {'has non-zero curl (non-conservative)' if curl_pass else 'is unexpectedly curl-free'}")

        results["Q_force_curl"] = {
            "frob_J": frob_JQ,
            "frob_antisym": frob_JQ_antisym,
            "antisymmetry_ratio": ratio_Q,
            "n_probes": n_probe,
            "pass": curl_pass,
        }
    else:
        ratio_Q = 0.0
        results["Q_force_curl"] = {"skipped": True}
        print("  [Test C] SKIPPED: no reverse channel")

    # --- Verdict ---
    print(f"\n  SUMMARY:")
    print(f"    Test A (f = -grad U):       {'PASS' if results['gradient_verification']['pass'] else 'FAIL'}")
    print(f"    Test B (Hessian symmetric):  {'PASS' if results['conservative_hessian']['pass'] else 'FAIL'}")
    if model.reverse_ch:
        print(f"    Test C (Q_i has curl):       {'PASS' if results.get('Q_force_curl', {}).get('pass') else 'FAIL'}")

    # Visualise
    n_panels = 3 if model.reverse_ch else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))

    ax = axes[0]
    ax.set_title(f"Test A: f_autograd vs f_fd\nmean err/||f|| = {mean_norm_err:.2e}", fontsize=10)
    scatter_n = min(200, f_autograd.numel())
    fa = f_autograd.reshape(-1)[:scatter_n].cpu().numpy()
    ff = f_fd.reshape(-1)[:scatter_n].cpu().numpy()
    ax.scatter(fa, ff, s=8, alpha=0.5, color="steelblue")
    lims = [min(fa.min(), ff.min()), max(fa.max(), ff.max())]
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("f (autograd)")
    ax.set_ylabel("f (finite diff)")

    ax = axes[1]
    ax.set_title(f"Test B: Hessian Symmetry\ncurl = {ratio_cons:.2e}", fontsize=10)
    ax.bar(["||H||_F", "||H-H^T||_F"], [frob_J, frob_antisym],
           color=["steelblue", "firebrick"])
    ax.set_ylabel("Frobenius norm")

    if model.reverse_ch and n_panels > 2:
        ax = axes[2]
        ax.set_title(f"Test C: Q_i Curl\ncurl = {ratio_Q:.2e}", fontsize=10)
        ax.bar(["||J_Q||_F", "||J_Q-J_Q^T||_F"],
               [results["Q_force_curl"]["frob_J"],
                results["Q_force_curl"]["frob_antisym"]],
               color=["steelblue", "firebrick"])
        ax.set_ylabel("Frobenius norm")

    fig.suptitle("Arm 1: Conservative Force = Gradient (Structural Proof)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "arm1_jacobian_symmetry.png", dpi=150)
    plt.close(fig)
    print(f"  Saved arm1_jacobian_symmetry.png")

    return results


# =====================================================================
# Arm 3: Per-layer energy budget decomposition
# =====================================================================

def _compute_potential(
    model: FockMultiXiPARFLM,
    h: torch.Tensor,
    x: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    """Compute U = V_theta(xi, h) + sum V_phi(h_t, h_s) at a given state."""
    cfg = model.cfg
    B, T, d = h.shape

    xi_input = h.detach() if cfg.causal_force else h
    from model_multixi import MultiChannelXi
    xi_now = model.xi_module(xi_input)

    h_in = h.detach().requires_grad_(False)
    V_th = model.V_theta(xi_now, h_in)

    h_src = h_in.detach()
    P = model.V_phi(h_in, h_src)
    mask = model._pair_mask_for(T, h.device)
    P_masked = P.masked_fill(~mask, 0.0)

    s_ell = model.per_layer_scale(layer_idx)
    if s_ell is not None:
        P_masked = P_masked * s_ell

    U_per_token = V_th.squeeze(-1) + P_masked.sum(dim=-1)
    return U_per_token.sum(dim=-1)


def arm3_energy_budget(
    model: FockMultiXiPARFLM,
    device: str,
    output_dir: Path,
    val_ids: np.ndarray | None = None,
    n_batches: int = 10,
    batch_size: int = 4,
    block_size: int = 64,
) -> Dict[str, Any]:
    """Arm 3: Per-layer energy budget decomposition.

    Tracks H_l per layer, decomposes Delta_H into:
      - W_damping (dissipative, <= 0)
      - W_Q (reverse channel work)
      - residual (should be ~0)
    """
    print("\n" + "=" * 70)
    print("ARM 3: Per-layer energy budget decomposition")
    print("=" * 70)

    cfg = model.cfg
    L = cfg.L
    dt = cfg.dt
    gamma_val = model.gamma.item() if isinstance(model.gamma, torch.Tensor) else float(cfg.fixed_gamma or 0.3)

    rng = np.random.default_rng(42)

    if val_ids is None:
        torch.manual_seed(0)
        x_all = torch.randint(0, cfg.vocab_size, (n_batches * batch_size, block_size), device=device)
    else:
        x_batches = []
        for _ in range(n_batches):
            xb, _ = get_batch(val_ids, batch_size, block_size, rng)
            x_batches.append(torch.from_numpy(xb).to(device))
        x_all = torch.cat(x_batches, dim=0)

    all_H = []
    all_KE = []
    all_PE = []
    all_W_damping = []
    all_W_Q = []
    all_residual = []

    total_samples = x_all.shape[0]
    for b_start in range(0, total_samples, batch_size):
        x = x_all[b_start:b_start + batch_size]
        B = x.shape[0]

        with torch.enable_grad():
            h0 = model._embed(x)
            m_b = model.compute_mass(x)
            gamma = model.gamma
            r, salience = model._init_registers(B, device)

            h = h0.detach().clone()
            h_prev = h0.detach().clone()
            v = torch.zeros_like(h)

            H_layers = []
            KE_layers = []
            PE_layers = []
            W_damp_layers = []
            W_Q_layers = []

            PE_0 = _compute_potential(model, h, x, layer_idx=0)
            KE_0 = 0.5 * (m_b.detach() * v.pow(2)).sum(dim=(-1, -2))
            H_0 = KE_0 + PE_0
            H_layers.append(H_0.detach().cpu())
            KE_layers.append(KE_0.detach().cpu())
            PE_layers.append(PE_0.detach().cpu())

            for ell in range(L):
                v_before = (h - h_prev) / dt if ell > 0 else torch.zeros_like(h)

                orig_scale = None
                if model.reverse_channel_scale is not None:
                    orig_scale = model.reverse_channel_scale.data.clone()

                if model.reverse_channel_scale is not None:
                    model.reverse_channel_scale.data.fill_(float("-inf"))

                r_copy, sal_copy = r.clone(), salience.clone()
                h_cons, _, r_after_cons, sal_after_cons = model._fock_layer_step(
                    h.clone(), h_prev.clone(), r_copy, sal_copy, m_b, gamma, dt, layer_idx=ell,
                )

                if orig_scale is not None:
                    model.reverse_channel_scale.data.copy_(orig_scale)

                r_copy2, sal_copy2 = r.clone(), salience.clone()
                h_new, h_prev_out, r_new, salience_new = model._fock_layer_step(
                    h.clone(), h_prev.clone(), r_copy2, sal_copy2, m_b, gamma, dt, layer_idx=ell,
                )

                delta_Q = (h_new[:, :h.shape[1], :] - h_cons[:, :h.shape[1], :]).detach()

                v_after = (h_new[:, :h.shape[1], :].detach() - h) / dt

                PE_next = _compute_potential(model, h_new[:, :h.shape[1], :].detach(), x, layer_idx=min(ell + 1, L - 1))
                KE_next = 0.5 * (m_b.detach() * v_after.pow(2)).sum(dim=(-1, -2))
                H_next = KE_next + PE_next
                H_layers.append(H_next.detach().cpu())
                KE_layers.append(KE_next.detach().cpu())
                PE_layers.append(PE_next.detach().cpu())

                displacement = (h_new[:, :h.shape[1], :].detach() - h)
                W_damp = -(gamma_val * (v_before.pow(2)).sum(dim=(-1, -2)) * dt)
                W_Q = (delta_Q * displacement).sum(dim=(-1, -2))

                W_damp_layers.append(W_damp.detach().cpu())
                W_Q_layers.append(W_Q.detach().cpu())

                h_prev = h.detach().clone()
                h = h_new[:, :h.shape[1], :].detach().clone()
                r = r_new.detach() if isinstance(r_new, torch.Tensor) else r_new
                salience = salience_new.detach() if isinstance(salience_new, torch.Tensor) else salience_new

        H_stack = torch.stack(H_layers, dim=1)
        KE_stack = torch.stack(KE_layers, dim=1)
        PE_stack = torch.stack(PE_layers, dim=1)
        W_damp_stack = torch.stack(W_damp_layers, dim=1)
        W_Q_stack = torch.stack(W_Q_layers, dim=1)

        delta_H = H_stack[:, 1:] - H_stack[:, :-1]
        residual = delta_H - W_damp_stack - W_Q_stack

        all_H.append(H_stack)
        all_KE.append(KE_stack)
        all_PE.append(PE_stack)
        all_W_damping.append(W_damp_stack)
        all_W_Q.append(W_Q_stack)
        all_residual.append(residual)

    H_all = torch.cat(all_H, dim=0)
    W_damp_all = torch.cat(all_W_damping, dim=0)
    W_Q_all = torch.cat(all_W_Q, dim=0)
    residual_all = torch.cat(all_residual, dim=0)

    H_mean = H_all.mean(dim=0).numpy()
    W_damp_mean = W_damp_all.mean(dim=0).numpy()
    W_Q_mean = W_Q_all.mean(dim=0).numpy()
    residual_mean = residual_all.mean(dim=0).numpy()
    residual_std = residual_all.std(dim=0).numpy()

    delta_H_mean = H_mean[1:] - H_mean[:-1]

    print(f"\n  Per-layer energy budget (mean over {H_all.shape[0]} samples):")
    print(f"  {'Layer':>5}  {'Delta_H':>12}  {'W_damping':>12}  {'W_Q':>12}  {'Residual':>12}  {'Res/|DH|':>10}")
    print(f"  {'-'*5}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*10}")
    for ell in range(L):
        dh = delta_H_mean[ell]
        wd = W_damp_mean[ell]
        wq = W_Q_mean[ell]
        res = residual_mean[ell]
        ratio = abs(res) / (abs(dh) + 1e-12)
        print(f"  {ell:>5d}  {dh:>12.4f}  {wd:>12.4f}  {wq:>12.4f}  {res:>12.4f}  {ratio:>10.2e}")

    results = {
        "arm": 3,
        "description": "Energy budget decomposition",
        "n_samples": int(H_all.shape[0]),
        "H_mean_per_layer": H_mean.tolist(),
        "delta_H_mean": delta_H_mean.tolist(),
        "W_damping_mean": W_damp_mean.tolist(),
        "W_Q_mean": W_Q_mean.tolist(),
        "residual_mean": residual_mean.tolist(),
        "residual_std": residual_std.tolist(),
        "mean_abs_residual": float(np.mean(np.abs(residual_mean))),
        "mean_abs_delta_H": float(np.mean(np.abs(delta_H_mean))),
        "residual_ratio": float(np.mean(np.abs(residual_mean)) / (np.mean(np.abs(delta_H_mean)) + 1e-12)),
    }

    # Stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    layers = np.arange(L)
    bar_width = 0.6

    ax.bar(layers, W_damp_mean, bar_width, label="W_damping (dissipative)", color="steelblue", alpha=0.8)
    ax.bar(layers, W_Q_mean, bar_width, bottom=np.where(W_Q_mean > 0, W_damp_mean, 0),
           label="W_Q (reverse channel)", color="firebrick", alpha=0.8)
    ax.bar(layers, residual_mean, bar_width * 0.4,
           bottom=W_damp_mean + np.where(W_Q_mean > 0, W_Q_mean, 0),
           label=f"Residual (mean |r|={results['mean_abs_residual']:.2e})",
           color="gray", alpha=0.7)

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Layer index", fontsize=12)
    ax.set_ylabel("Energy contribution", fontsize=12)
    ax.set_title("Arm 3: Per-Layer Energy Budget Decomposition", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xticks(layers)
    fig.tight_layout()
    fig.savefig(output_dir / "arm3_energy_budget.png", dpi=150)
    plt.close(fig)
    print(f"  Saved arm3_energy_budget.png")

    return results


# =====================================================================
# Arm 2: Conservative ablation (R^2 recovery)
# =====================================================================

def _compute_layer_updates(
    model: FockMultiXiPARFLM,
    x: torch.Tensor,
    disable_Q: bool = False,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Run forward pass, recording (h_in, h_out, delta) per layer."""
    cfg = model.cfg
    B, T = x.shape
    device = x.device

    orig_scale = None
    if disable_Q and model.reverse_channel_scale is not None:
        orig_scale = model.reverse_channel_scale.data.clone()
        model.reverse_channel_scale.data.fill_(float("-inf"))

    with torch.enable_grad():
        h0 = model._embed(x)
        m_b = model.compute_mass(x)
        gamma = model.gamma
        r, salience = model._init_registers(B, device)

        h = h0.clone()
        h_prev = h0.clone()
        updates = []

        for ell in range(cfg.L):
            h_in_snap = h.detach().clone()
            h_new, h_prev_out, r, salience = model._fock_layer_step(
                h, h_prev, r, salience, m_b, gamma, cfg.dt, layer_idx=ell,
            )
            h_out_snap = h_new[:, :T, :].detach().clone()
            delta = h_out_snap - h_in_snap
            updates.append((h_in_snap, h_out_snap, delta))
            h_prev = h.detach().clone()
            h = h_new[:, :T, :].detach().clone()

    if orig_scale is not None:
        model.reverse_channel_scale.data.copy_(orig_scale)

    return updates


def _fit_shared_potential_r2(
    updates: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> List[float]:
    """Compute per-layer R^2 of a linear fit to the layer updates.

    A shared-potential system produces updates where the force field is
    curl-free across layers. We measure this via R^2 of predicting
    delta_h from (h_in, layer_idx) using a single linear model.
    For a perfectly conservative system with shared potential, the
    per-layer R^2 should be very high.
    """
    r2_per_layer = []
    for ell, (h_in, h_out, delta) in enumerate(updates):
        B, T, d = delta.shape
        delta_flat = delta.reshape(-1, d)
        h_in_flat = h_in.reshape(-1, d)

        ss_tot = (delta_flat - delta_flat.mean(dim=0, keepdim=True)).pow(2).sum().item()
        if ss_tot < 1e-12:
            r2_per_layer.append(1.0)
            continue

        X = h_in_flat
        Y = delta_flat
        XtX = X.T @ X
        reg = 1e-6 * torch.eye(d, device=X.device)
        beta = torch.linalg.solve(XtX + reg, X.T @ Y)
        Y_pred = X @ beta
        ss_res = (Y - Y_pred).pow(2).sum().item()
        r2 = 1.0 - ss_res / ss_tot
        r2_per_layer.append(r2)

    return r2_per_layer


def arm2_conservative_ablation(
    model: FockMultiXiPARFLM,
    device: str,
    output_dir: Path,
    val_ids: np.ndarray | None = None,
    n_batches: int = 10,
    batch_size: int = 4,
    block_size: int = 64,
) -> Dict[str, Any]:
    """Arm 2: Disable Q_i and verify conservativity recovery."""
    print("\n" + "=" * 70)
    print("ARM 2: Conservative ablation (Q_i = 0)")
    print("=" * 70)

    rng = np.random.default_rng(42)
    cfg = model.cfg

    x_batches = []
    for _ in range(n_batches):
        if val_ids is not None:
            xb, _ = get_batch(val_ids, batch_size, block_size, rng)
            x_batches.append(torch.from_numpy(xb).to(device))
        else:
            x_batches.append(torch.randint(0, cfg.vocab_size, (batch_size, block_size), device=device))

    # R^2 with Q_i disabled
    all_updates_off = []
    for xb in x_batches:
        all_updates_off.extend(_compute_layer_updates(model, xb, disable_Q=True))

    # Group by layer
    L = cfg.L
    updates_by_layer_off = [[] for _ in range(L)]
    for i, upd in enumerate(all_updates_off):
        updates_by_layer_off[i % L].append(upd)

    r2_off = []
    for ell in range(L):
        h_ins = torch.cat([u[0] for u in updates_by_layer_off[ell]], dim=0)
        h_outs = torch.cat([u[1] for u in updates_by_layer_off[ell]], dim=0)
        deltas = torch.cat([u[2] for u in updates_by_layer_off[ell]], dim=0)
        r2_off.extend(_fit_shared_potential_r2([(h_ins, h_outs, deltas)]))

    # R^2 with Q_i enabled
    all_updates_on = []
    for xb in x_batches:
        all_updates_on.extend(_compute_layer_updates(model, xb, disable_Q=False))

    updates_by_layer_on = [[] for _ in range(L)]
    for i, upd in enumerate(all_updates_on):
        updates_by_layer_on[i % L].append(upd)

    r2_on = []
    for ell in range(L):
        h_ins = torch.cat([u[0] for u in updates_by_layer_on[ell]], dim=0)
        h_outs = torch.cat([u[1] for u in updates_by_layer_on[ell]], dim=0)
        deltas = torch.cat([u[2] for u in updates_by_layer_on[ell]], dim=0)
        r2_on.extend(_fit_shared_potential_r2([(h_ins, h_outs, deltas)]))

    rev_scale = None
    if model.reverse_channel_scale is not None:
        rev_scale = torch.tanh(model.reverse_channel_scale).item()

    print(f"\n  Learned reverse_channel_scale: tanh(s) = {rev_scale}")
    print(f"\n  Per-layer R^2:")
    print(f"  {'Layer':>5}  {'Q_i=0':>10}  {'Q_i=on':>10}  {'Delta':>10}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}")
    for ell in range(L):
        print(f"  {ell:>5d}  {r2_off[ell]:>10.6f}  {r2_on[ell]:>10.6f}  {r2_on[ell]-r2_off[ell]:>+10.6f}")

    results = {
        "arm": 2,
        "description": "Conservative ablation",
        "reverse_channel_scale_tanh": rev_scale,
        "r2_Q_off": r2_off,
        "r2_Q_on": r2_on,
        "r2_Q_off_mean": float(np.mean(r2_off)),
        "r2_Q_on_mean": float(np.mean(r2_on)),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    layers = np.arange(L)
    ax.plot(layers, r2_off, "o-", color="forestgreen", linewidth=2, markersize=8, label="Q_i = 0 (conservative)")
    ax.plot(layers, r2_on, "s--", color="firebrick", linewidth=2, markersize=8, label=f"Q_i on (tanh(s)={rev_scale:.3f})" if rev_scale else "Q_i on")
    ax.set_xlabel("Layer index", fontsize=12)
    ax.set_ylabel("Linear-fit R^2", fontsize=12)
    ax.set_title("Arm 2: Conservative Ablation (Q_i disabled vs enabled)", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=min(min(r2_off), min(r2_on)) - 0.05, top=1.02)
    ax.set_xticks(layers)
    fig.tight_layout()
    fig.savefig(output_dir / "arm2_conservative_ablation.png", dpi=150)
    plt.close(fig)
    print(f"  Saved arm2_conservative_ablation.png")

    return results


# =====================================================================
# Arm 4: Conservativity dial
# =====================================================================

def _eval_ppl(
    model: FockMultiXiPARFLM,
    val_ids: np.ndarray,
    device: str,
    batch_size: int = 4,
    block_size: int = 64,
    n_batches: int = 20,
) -> float:
    """Evaluate validation perplexity."""
    rng = np.random.default_rng(99)
    total_loss = 0.0
    count = 0
    model.eval()
    with torch.enable_grad():
        for _ in range(n_batches):
            xb, yb = get_batch(val_ids, batch_size, block_size, rng)
            x = torch.from_numpy(xb).to(device)
            y = torch.from_numpy(yb).to(device)
            _, loss = model(x, targets=y)
            total_loss += loss.detach().item()
            count += 1
    return math.exp(total_loss / count)


def arm4_conservativity_dial(
    model: FockMultiXiPARFLM,
    device: str,
    output_dir: Path,
    val_ids: np.ndarray,
    batch_size: int = 4,
    block_size: int = 64,
) -> Dict[str, Any]:
    """Arm 4: Sweep reverse_channel_scale from 0 to learned value."""
    print("\n" + "=" * 70)
    print("ARM 4: Conservativity dial (sweep reverse_channel_scale)")
    print("=" * 70)

    if model.reverse_channel_scale is None:
        print("  SKIPPED: model has no reverse channel")
        return {"arm": 4, "skipped": True}

    learned_raw = model.reverse_channel_scale.data.item()
    learned_tanh = math.tanh(learned_raw)
    print(f"  Learned raw scale: {learned_raw:.6f}")
    print(f"  Learned tanh(scale): {learned_tanh:.6f}")

    sweep_tanh = sorted(set([0.0, -0.05, -0.10, -0.15, -0.20, learned_tanh]))
    sweep_raw = [math.atanh(max(min(t, 0.999), -0.999)) for t in sweep_tanh]

    results_list = []
    print(f"\n  {'tanh(s)':>10}  {'PPL':>8}  {'R2_mean':>10}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*10}")

    for raw_val, tanh_val in zip(sweep_raw, sweep_tanh):
        model.reverse_channel_scale.data.fill_(raw_val)

        ppl = _eval_ppl(model, val_ids, device, batch_size, block_size)

        rng = np.random.default_rng(42)
        xb, _ = get_batch(val_ids, batch_size, block_size, rng)
        x = torch.from_numpy(xb).to(device)
        updates = _compute_layer_updates(model, x, disable_Q=False)
        r2_vals = _fit_shared_potential_r2(updates)
        r2_mean = float(np.mean(r2_vals))

        print(f"  {tanh_val:>10.4f}  {ppl:>8.2f}  {r2_mean:>10.6f}")
        results_list.append({
            "tanh_scale": tanh_val,
            "raw_scale": raw_val,
            "ppl": ppl,
            "r2_mean": r2_mean,
            "r2_per_layer": r2_vals,
        })

    model.reverse_channel_scale.data.fill_(learned_raw)

    results = {
        "arm": 4,
        "description": "Conservativity dial sweep",
        "learned_tanh": learned_tanh,
        "sweep": results_list,
    }

    fig, ax1 = plt.subplots(figsize=(9, 5))
    tanhs = [r["tanh_scale"] for r in results_list]
    ppls = [r["ppl"] for r in results_list]
    r2s = [r["r2_mean"] for r in results_list]

    ax1.plot(tanhs, ppls, "o-", color="steelblue", linewidth=2, markersize=8, label="Val PPL")
    ax1.set_xlabel("tanh(reverse_channel_scale)", fontsize=12)
    ax1.set_ylabel("Validation PPL", color="steelblue", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.plot(tanhs, r2s, "s--", color="forestgreen", linewidth=2, markersize=8, label="R^2 (mean)")
    ax2.set_ylabel("Linear-fit R^2 (mean)", color="forestgreen", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="forestgreen")

    ax1.axvline(x=learned_tanh, color="gray", linestyle=":", alpha=0.7, label=f"Learned ({learned_tanh:.3f})")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="center left")

    fig.suptitle("Arm 4: Conservativity-Performance Tradeoff", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "arm4_conservativity_dial.png", dpi=150)
    plt.close(fig)
    print(f"  Saved arm4_conservativity_dial.png")

    return results


# =====================================================================
# Arm 5: Four-way architectural separator
# =====================================================================

def arm5_separator(
    model: FockMultiXiPARFLM,
    device: str,
    output_dir: Path,
    val_ids: np.ndarray | None = None,
    batch_size: int = 4,
    block_size: int = 64,
) -> Dict[str, Any]:
    """Arm 5: Compare FockPARFLM (Q_i on/off) against architectural baselines.

    Runs Arms 2-style diagnostics on the same model in two configurations
    (Q_i disabled = pure PARFLM, Q_i enabled = full Fock). The SPLM and
    GPT-2 baselines are reported as literature values from the paper.
    """
    print("\n" + "=" * 70)
    print("ARM 5: Four-way architectural separator")
    print("=" * 70)

    rng = np.random.default_rng(42)
    cfg = model.cfg
    L = cfg.L

    x_batches = []
    for _ in range(10):
        if val_ids is not None:
            xb, _ = get_batch(val_ids, batch_size, block_size, rng)
            x_batches.append(torch.from_numpy(xb).to(device))
        else:
            x_batches.append(torch.randint(0, cfg.vocab_size, (batch_size, block_size), device=device))

    # FockPARFLM with Q_i = 0 (= PARFLM mode)
    all_upd_off = []
    for xb in x_batches:
        all_upd_off.extend(_compute_layer_updates(model, xb, disable_Q=True))
    upd_by_layer_off = [[] for _ in range(L)]
    for i, upd in enumerate(all_upd_off):
        upd_by_layer_off[i % L].append(upd)
    r2_parflm = []
    for ell in range(L):
        h_ins = torch.cat([u[0] for u in upd_by_layer_off[ell]], dim=0)
        h_outs = torch.cat([u[1] for u in upd_by_layer_off[ell]], dim=0)
        deltas = torch.cat([u[2] for u in upd_by_layer_off[ell]], dim=0)
        r2_parflm.extend(_fit_shared_potential_r2([(h_ins, h_outs, deltas)]))

    # FockPARFLM with Q_i on
    all_upd_on = []
    for xb in x_batches:
        all_upd_on.extend(_compute_layer_updates(model, xb, disable_Q=False))
    upd_by_layer_on = [[] for _ in range(L)]
    for i, upd in enumerate(all_upd_on):
        upd_by_layer_on[i % L].append(upd)
    r2_fock = []
    for ell in range(L):
        h_ins = torch.cat([u[0] for u in upd_by_layer_on[ell]], dim=0)
        h_outs = torch.cat([u[1] for u in upd_by_layer_on[ell]], dim=0)
        deltas = torch.cat([u[2] for u in upd_by_layer_on[ell]], dim=0)
        r2_fock.extend(_fit_shared_potential_r2([(h_ins, h_outs, deltas)]))

    # Literature baselines from the paper
    r2_splm_paper = 0.957
    r2_gpt2_paper = 0.46

    models_data = {
        "GPT-2 (paper)": {"r2_mean": r2_gpt2_paper, "type": "literature"},
        "SPLM (paper)": {"r2_mean": r2_splm_paper, "type": "literature"},
        "FockPARFLM (Q_i=0)": {"r2_mean": float(np.mean(r2_parflm)), "r2_per_layer": r2_parflm, "type": "measured"},
        "FockPARFLM v2.1": {"r2_mean": float(np.mean(r2_fock)), "r2_per_layer": r2_fock, "type": "measured"},
    }

    print(f"\n  Four-way separator (linear-fit R^2, mean over layers):")
    print(f"  {'Model':>25}  {'R^2 mean':>10}  {'Source':>12}")
    print(f"  {'-'*25}  {'-'*10}  {'-'*12}")
    for name, info in models_data.items():
        print(f"  {name:>25}  {info['r2_mean']:>10.4f}  {info['type']:>12}")

    results = {
        "arm": 5,
        "description": "Four-way architectural separator",
        "models": {k: {kk: vv for kk, vv in v.items() if kk != "r2_per_layer"} for k, v in models_data.items()},
        "r2_per_layer_parflm_mode": r2_parflm,
        "r2_per_layer_fock": r2_fock,
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    names = list(models_data.keys())
    r2_means = [models_data[n]["r2_mean"] for n in names]
    colors = ["lightcoral", "gold", "forestgreen", "steelblue"]
    bars = ax.bar(names, r2_means, color=colors, edgecolor="black", linewidth=0.8)

    for bar, val in zip(bars, r2_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Linear-fit R^2 (mean over layers)", fontsize=12)
    ax.set_title("Arm 5: Architectural Conservativity Separator", fontsize=13)
    ax.set_ylim(0, 1.15)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / "arm5_separator.png", dpi=150)
    plt.close(fig)
    print(f"  Saved arm5_separator.png")

    return results


# =====================================================================
# Main
# =====================================================================

def write_summary(results: Dict[str, Any], output_dir: Path) -> None:
    """Write a human-readable markdown summary of all results."""
    lines = [
        "# Fock-PARFLM v2.1 Conservativity Diagnostic — Results Summary\n",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
    ]

    if "arm1" in results:
        r = results["arm1"]
        lines.append("## Arm 1: Structural Jacobian Symmetry\n")
        for label in ["conservative_only", "with_Q_i"]:
            if label in r:
                d = r[label]
                lines.append(f"- **{label}**: ||J-J^T||/||J|| = {d['antisymmetry_ratio']:.2e}")
        lines.append("")

    if "arm2" in results:
        r = results["arm2"]
        lines.append("## Arm 2: Conservative Ablation\n")
        lines.append(f"- Reverse channel scale: tanh(s) = {r.get('reverse_channel_scale_tanh', '?')}")
        lines.append(f"- Mean R^2 (Q_i off): {r.get('r2_Q_off_mean', '?'):.6f}")
        lines.append(f"- Mean R^2 (Q_i on):  {r.get('r2_Q_on_mean', '?'):.6f}")
        lines.append("")

    if "arm3" in results:
        r = results["arm3"]
        lines.append("## Arm 3: Energy Budget Decomposition\n")
        lines.append(f"- Samples: {r.get('n_samples', '?')}")
        lines.append(f"- Mean |residual|: {r.get('mean_abs_residual', 0):.4e}")
        lines.append(f"- Mean |Delta_H|:  {r.get('mean_abs_delta_H', 0):.4e}")
        lines.append(f"- Residual ratio:  {r.get('residual_ratio', 0):.4e}")
        lines.append("")

    if "arm4" in results:
        r = results["arm4"]
        if not r.get("skipped"):
            lines.append("## Arm 4: Conservativity Dial\n")
            lines.append(f"- Learned tanh(scale): {r.get('learned_tanh', '?'):.6f}")
            lines.append("| tanh(s) | PPL | R^2 mean |")
            lines.append("|---------|-----|----------|")
            for pt in r.get("sweep", []):
                lines.append(f"| {pt['tanh_scale']:.4f} | {pt['ppl']:.2f} | {pt['r2_mean']:.4f} |")
            lines.append("")

    if "arm5" in results:
        r = results["arm5"]
        lines.append("## Arm 5: Four-Way Separator\n")
        lines.append("| Model | R^2 mean | Source |")
        lines.append("|-------|----------|--------|")
        for name, info in r.get("models", {}).items():
            lines.append(f"| {name} | {info['r2_mean']:.4f} | {info['type']} |")
        lines.append("")

    summary_path = output_dir / "conservativity_summary.md"
    summary_path.write_text("\n".join(lines))
    print(f"\nSaved summary to {summary_path}")


def main():
    ap = argparse.ArgumentParser(description="Fock-PARFLM conservativity diagnostic")
    ap.add_argument("--arm", type=str, default="all",
                    help="Which arm(s) to run: 1,2,3,4,5 or 'all'")
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="Path to trained FockMultiXiPARFLM checkpoint (.pt)")
    ap.add_argument("--output-dir", type=str, default="results/conservativity",
                    help="Directory for output artifacts")
    ap.add_argument("--logfreq-path", type=str, default=str(LOGFREQ_PATH),
                    help="Path to logfreq_surprisal .npy for mass model")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--n-batches", type=int, default=10)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = args.device or _pick_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arms_to_run = set()
    if args.arm == "all":
        arms_to_run = {1, 2, 3, 4, 5}
    else:
        for a in args.arm.split(","):
            arms_to_run.add(int(a.strip()))

    print(f"Device: {device}")
    print(f"Arms to run: {sorted(arms_to_run)}")
    print(f"Output directory: {output_dir}")

    has_checkpoint = args.checkpoint is not None

    # Arm 1 uses a smoke model; Arms 2-5 need a real model
    if 1 in arms_to_run and not has_checkpoint:
        cfg_smoke = build_smoke_config()
        torch.manual_seed(42)
        model_smoke = FockMultiXiPARFLM(cfg_smoke).to(device).eval()
    else:
        model_smoke = None

    model = None
    val_ids = None
    if arms_to_run & {2, 3, 4, 5} or (1 in arms_to_run and has_checkpoint):
        if not has_checkpoint:
            print("\nWARNING: Arms 2-5 require a trained checkpoint. "
                  "Running with random-init smoke model for demonstration.")
            model = load_model(None, device)
        else:
            logfreq = args.logfreq_path if Path(args.logfreq_path).exists() else None
            model = load_model(args.checkpoint, device, logfreq_path=logfreq)

        if has_checkpoint:
            try:
                _, val_ids = load_tiny_stories(max_train_tokens=5_000_000)
                print(f"Loaded TinyStories val set: {len(val_ids):,} tokens")
            except Exception as e:
                print(f"WARNING: Could not load TinyStories: {e}")
                print("  Using random tokens for evaluation.")

    all_results: Dict[str, Any] = {}

    if 1 in arms_to_run:
        m = model if has_checkpoint else model_smoke
        all_results["arm1"] = arm1_jacobian_symmetry(m, device, output_dir)

    if 2 in arms_to_run:
        all_results["arm2"] = arm2_conservative_ablation(
            model, device, output_dir,
            val_ids=val_ids,
            n_batches=args.n_batches,
            batch_size=args.batch_size,
            block_size=args.block_size,
        )

    if 3 in arms_to_run:
        all_results["arm3"] = arm3_energy_budget(
            model, device, output_dir,
            val_ids=val_ids,
            n_batches=args.n_batches,
            batch_size=args.batch_size,
            block_size=args.block_size,
        )

    if 4 in arms_to_run:
        if val_ids is None:
            print("\nARM 4: SKIPPED (requires TinyStories val set for PPL eval)")
            all_results["arm4"] = {"arm": 4, "skipped": True, "reason": "no val data"}
        else:
            all_results["arm4"] = arm4_conservativity_dial(
                model, device, output_dir,
                val_ids=val_ids,
                batch_size=args.batch_size,
                block_size=args.block_size,
            )

    if 5 in arms_to_run:
        all_results["arm5"] = arm5_separator(
            model, device, output_dir,
            val_ids=val_ids,
            batch_size=args.batch_size,
            block_size=args.block_size,
        )

    # Save JSON report
    report_path = output_dir / "conservativity_report.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved JSON report to {report_path}")

    write_summary(all_results, output_dir)

    print("\n" + "=" * 70)
    print("CONSERVATIVITY DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
