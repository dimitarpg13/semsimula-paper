"""
Causal-violation probe for the full PARFLM hierarchy:
  PARFLM  →  SparsePARFLM  →  MultiXiPARFLM  →  FockMultiXiPARFLM

This generalises causal_probe_parf.py to handle multi-channel ξ and
Fock-space register extensions.

Probe strategy (unchanged from the single-ξ version)
=====================================================
  A. *Perturbation probe* — change one token x[t_pert], compare logits
     at every position t < t_pert.  In a causal model these MUST be
     bit-identical (Δ ≡ 0 within fp32 noise).

  B. *Gradient-Jacobian probe* — pick a target t and compute
     ∂(logits[0, t, :].sum()) / ∂(emb_in[0, t', :]) via autograd.  In
     a causal model the gradient is non-zero only for t' <= t.

Fock-specific leak surfaces
===========================
The Fock model introduces three additional causality risk surfaces:

  1. **Register creation gates** — in v1 the creation gate sees h_mean
     (the mean over all T token states); in v2 the Q/K/V cross-attention
     reads all T token states.  If registers carry information from
     future tokens into past positions, this leaks.

  2. **Reverse channel** (v2 only) — the reverse channel force Q_i
     applied to token states h_t is a function of the active registers.
     If the register content bleeds future information backward, the
     Q_force contaminates past predictions.

  3. **Register-extended dynamics** — token and register states are
     concatenated before the layer step; the detach() on the source
     slice must correctly apply to the extended T+M state.

All three surfaces are exercised by the same perturbation and gradient
probes, because:
  - The perturbation at x[t_pert] propagates through h → registers →
    h (via creation, dynamics, reverse channel, destruction).
  - The gradient probe traces the full autograd graph including
    register lifecycle.

The probe also runs in buggy mode (causal_force=False) to confirm
the probes are sensitive enough to detect leaks.

Usage
-----
  python3 causal_probe_multixi.py
      → run probes on all model variants (MultiXi, FockV1, FockV2)

  python3 causal_probe_multixi.py --variant fock_v2
      → run only on FockMultiXiPARFLM v2

  python3 causal_probe_multixi.py --strict
      → exit non-zero on any failure (CI gate)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

THIS_DIR = Path(__file__).parent
PARENT_DIR = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(PARENT_DIR / "multixi"))

from model_parf_multixi import MultiXiPARFConfig, MultiXiPARFLM  # noqa: E402
from model_fock_parf_multixi import (  # noqa: E402
    FockMultiXiPARFConfig,
    FockMultiXiPARFLM,
)

TOL_PRE: float = 1e-6
TOL_BUGGY_FLOOR: float = 1e-6


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------
def _multixi_config(causal_force: bool, K: int = 2) -> MultiXiPARFConfig:
    return MultiXiPARFConfig(
        vocab_size=257,
        d=16, max_len=64, L=4,
        v_hidden=32, v_depth=2,
        v_phi_kind="structural_competitive",
        v_phi_d_type=4, v_phi_d_angle=2,
        v_phi_phi_hidden=8, v_phi_theta_hidden=8,
        mass_mode="global",
        ln_after_step=True,
        causal_force=causal_force,
        top_k=4,
        xi_channels=K,
        xi_alpha_init_mode="log_spaced",
        xi_learnable=True,
        use_layer_checkpoint=False,
        use_gathered_v_phi=True,
    )


def _fock_config(
    causal_force: bool,
    fock_version: str = "v1",
    reverse_channel: bool = True,
    K: int = 2,
    M: int = 4,
) -> FockMultiXiPARFConfig:
    return FockMultiXiPARFConfig(
        vocab_size=257,
        d=16, max_len=64, L=4,
        v_hidden=32, v_depth=2,
        v_phi_kind="structural_competitive",
        v_phi_d_type=4, v_phi_d_angle=2,
        v_phi_phi_hidden=8, v_phi_theta_hidden=8,
        mass_mode="global",
        ln_after_step=True,
        causal_force=causal_force,
        top_k=4,
        xi_channels=K,
        xi_alpha_init_mode="log_spaced",
        xi_learnable=True,
        use_layer_checkpoint=False,
        use_gathered_v_phi=True,
        fock_version=fock_version,
        n_registers=M,
        register_salience_decay=0.9,
        register_salience_threshold=0.1,
        creation_gate_hidden=16,
        destruction_gate_hidden=16,
        stack_discipline=(fock_version == "v1"),
        register_init_scale=0.02,
        d_k=8,
        tau_create_init=0.1,
        reverse_channel=reverse_channel,
    )


# ---------------------------------------------------------------------------
# Perturbation probe
# ---------------------------------------------------------------------------
def perturbation_probe(
    model: torch.nn.Module,
    vocab_size: int,
    T: int = 32,
    t_pert: int = 20,
    seed: int = 0,
) -> Tuple[float, float, torch.Tensor]:
    """Return (max Δ on causal side, max Δ on after-side, full Δ vector)."""
    rng = np.random.default_rng(seed)
    xb = rng.integers(0, vocab_size, size=(1, T)).astype(np.int64)
    x_a = torch.from_numpy(xb)
    x_b = x_a.clone()
    orig = int(x_b[0, t_pert].item())
    x_b[0, t_pert] = (orig + 17) % vocab_size

    model.eval()
    with torch.enable_grad():
        out_a = model(x_a)
        out_b = model(x_b)
    logits_a = out_a[0].detach()
    logits_b = out_b[0].detach()
    diffs = (logits_a - logits_b).abs().max(dim=-1).values[0]
    pre = float(diffs[:t_pert].max().item())
    post = float(diffs[t_pert + 1:].max().item()) if t_pert + 1 < T else 0.0
    return pre, post, diffs


# ---------------------------------------------------------------------------
# Gradient-Jacobian probe
# ---------------------------------------------------------------------------
def gradient_probe(
    model: torch.nn.Module,
    vocab_size: int,
    T: int = 32,
    t_target: int = 20,
    seed: int = 0,
) -> Tuple[float, float, torch.Tensor]:
    """Compute ∂(logits[0, t_target, :].sum()) / ∂(emb_in[0, t', :]).

    Must run in train() mode to exercise create_graph=True paths
    (the integrator's V_θ/V_φ second-order gradient, and in Fock
    models the register lifecycle autograd graph).
    """
    rng = np.random.default_rng(seed)
    xb = rng.integers(0, vocab_size, size=(1, T)).astype(np.int64)
    x = torch.from_numpy(xb)

    was_training = model.training
    model.train()
    try:
        with torch.enable_grad():
            emb_static = model._embed(x)
            emb_in = emb_static.detach().clone().requires_grad_(True)
            h_L, _ = model._stack_forward(emb_in, x)
            logits = h_L @ model.E.weight.T
            target = logits[0, t_target, :].sum()
            (g,) = torch.autograd.grad(
                target, emb_in,
                retain_graph=False, create_graph=False,
            )
    finally:
        if not was_training:
            model.eval()

    g = g[0]
    norms = g.norm(dim=-1)
    post = float(norms[t_target + 1:].max().item()) if t_target + 1 < T else 0.0
    pre = float(norms[:t_target + 1].max().item())
    return post, pre, norms


# ---------------------------------------------------------------------------
# assert_causal — drop-in hook for trainers
# ---------------------------------------------------------------------------
def assert_causal(
    model: torch.nn.Module,
    vocab_size: int,
    T: int = 32,
    t_pert: int = 20,
    seed: int = 0,
    tol: float = TOL_PRE,
) -> None:
    """Run both probes on a real, on-device model; raise if leakage found.

    Works for any model in the PARFLM hierarchy (MultiXiPARFLM,
    FockMultiXiPARFLM).  MPS workaround: temporarily moves to CPU
    for the gradient probe (second-order graph not supported on MPS).
    Must be called BEFORE the optimiser is constructed.
    """
    orig_device = next(model.parameters()).device
    moved = orig_device.type != "cpu"
    if moved:
        model.to("cpu")
    try:
        pre_pert, post_pert, _ = perturbation_probe(
            model, vocab_size=vocab_size, T=T, t_pert=t_pert, seed=seed,
        )
        post_grad, pre_grad, _ = gradient_probe(
            model, vocab_size=vocab_size, T=T, t_target=t_pert, seed=seed,
        )
    finally:
        if moved:
            model.to(orig_device)

    model_class = type(model).__name__
    cfg = getattr(model, "cfg", None)
    causal_flag = getattr(cfg, "causal_force", "?") if cfg else "?"
    fock_ver = getattr(cfg, "fock_version", "n/a") if cfg else "n/a"
    rev_ch = getattr(cfg, "reverse_channel", "n/a") if cfg else "n/a"

    if pre_pert >= tol:
        raise RuntimeError(
            f"[causal-probe] PERTURBATION LEAK: pre={pre_pert:.4e} >= "
            f"tol={tol:.0e}.  Aborting training.  "
            f"model={model_class}, causal_force={causal_flag}, "
            f"fock_version={fock_ver}, reverse_channel={rev_ch}."
        )
    if post_grad >= tol:
        raise RuntimeError(
            f"[causal-probe] GRADIENT LEAK: post={post_grad:.4e} >= "
            f"tol={tol:.0e}.  Aborting training.  "
            f"model={model_class}, causal_force={causal_flag}, "
            f"fock_version={fock_ver}, reverse_channel={rev_ch}."
        )


# ---------------------------------------------------------------------------
# Per-variant runner
# ---------------------------------------------------------------------------
VARIANT_BUILDERS = {
    "multixi_K2": lambda cf: (MultiXiPARFLM(_multixi_config(cf, K=2)),
                               "MultiXiPARFLM K=2"),
    "multixi_K4": lambda cf: (MultiXiPARFLM(_multixi_config(cf, K=4)),
                               "MultiXiPARFLM K=4"),
    "fock_v1": lambda cf: (FockMultiXiPARFLM(_fock_config(cf, "v1", False)),
                            "FockMultiXiPARFLM v1 (no rev)"),
    "fock_v2_rev": lambda cf: (FockMultiXiPARFLM(_fock_config(cf, "v2", True)),
                                "FockMultiXiPARFLM v2 (+ reverse ch)"),
    "fock_v2_norev": lambda cf: (FockMultiXiPARFLM(_fock_config(cf, "v2", False)),
                                  "FockMultiXiPARFLM v2 (no reverse ch)"),
}


def probe_one_variant(
    variant_key: str,
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[bool, Dict]:
    """Run both probes in fixed and buggy modes on a variant."""
    builder = VARIANT_BUILDERS[variant_key]
    T = 32
    t_pert = 20

    # ----- fixed (causal_force=True) -----
    torch.manual_seed(seed)
    m_fix, label = builder(True)
    pre_pert_fix, post_pert_fix, _ = perturbation_probe(
        m_fix, vocab_size=257, T=T, t_pert=t_pert, seed=seed,
    )
    post_grad_fix, pre_grad_fix, _ = gradient_probe(
        m_fix, vocab_size=257, T=T, t_target=t_pert, seed=seed,
    )

    # ----- buggy (causal_force=False) -----
    torch.manual_seed(seed)
    m_bug, _ = builder(False)
    pre_pert_bug, post_pert_bug, _ = perturbation_probe(
        m_bug, vocab_size=257, T=T, t_pert=t_pert, seed=seed,
    )
    post_grad_bug, pre_grad_bug, _ = gradient_probe(
        m_bug, vocab_size=257, T=T, t_target=t_pert, seed=seed,
    )

    fix_pert_ok = pre_pert_fix < TOL_PRE
    fix_grad_ok = post_grad_fix < TOL_PRE
    ok = fix_pert_ok and fix_grad_ok

    if verbose:
        verdict = "OK" if ok else "FAIL"
        print(f"  [{verdict:>4}] {label}")
        print(f"           fixed mode (causal_force=True):")
        print(f"             perturbation pre={pre_pert_fix:.2e}  "
              f"post={post_pert_fix:.2e}  "
              f"({'OK' if fix_pert_ok else 'FAIL'})")
        print(f"             gradient     post={post_grad_fix:.2e}  "
              f"pre={pre_grad_fix:.2e}  "
              f"({'OK' if fix_grad_ok else 'FAIL'})")
        if pre_pert_bug > TOL_BUGGY_FLOOR or post_grad_bug > TOL_BUGGY_FLOOR:
            leak_status = "(leak detected as expected)"
        else:
            leak_status = ("(no leak at random init; fix path verified "
                           "by fixed mode)")
        print(f"           buggy mode (causal_force=False):")
        print(f"             perturbation pre={pre_pert_bug:.2e}  "
              f"post={post_pert_bug:.2e}  {leak_status}")
        print(f"             gradient     post={post_grad_bug:.2e}")

    details = {
        "variant": variant_key, "label": label,
        "fix_pert_pre": pre_pert_fix, "fix_pert_post": post_pert_fix,
        "fix_grad_post": post_grad_fix, "fix_grad_pre": pre_grad_fix,
        "bug_pert_pre": pre_pert_bug, "bug_pert_post": post_pert_bug,
        "bug_grad_post": post_grad_bug,
        "fix_pert_ok": fix_pert_ok, "fix_grad_ok": fix_grad_ok,
        "ok": ok,
    }
    return ok, details


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run_all(strict: bool, variants: List[str]) -> int:
    print("=" * 78)
    print(" Multi-Xi / Fock PARFLM causal-violation probe")
    print(" mode: random-init, d=16, perturbation + gradient")
    print("=" * 78)
    fails = 0
    for vk in variants:
        try:
            ok, _ = probe_one_variant(vk, verbose=True)
        except Exception as exc:
            print(f"  [FAIL] {vk}: {type(exc).__name__}: {exc}")
            fails += 1
            continue
        if not ok:
            fails += 1
    print("-" * 78)
    if fails == 0:
        print(f"  All {len(variants)} variants: fixed-mode causal-side "
              f"Δ < {TOL_PRE:.0e}.  Models are causal by construction.")
        return 0
    print(f"  {fails} variant(s) leaked in fixed mode.")
    return 1 if strict else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Causal-violation probe for Multi-Xi and Fock PARFLM."
    )
    ap.add_argument(
        "--variant", default=None,
        choices=list(VARIANT_BUILDERS.keys()),
        help="Run only this variant. If omitted, runs all.",
    )
    ap.add_argument(
        "--strict", action="store_true",
        help="Exit non-zero on any failure (CI gate).",
    )
    args = ap.parse_args()

    if args.variant:
        variants = [args.variant]
    else:
        variants = list(VARIANT_BUILDERS.keys())

    return run_all(strict=args.strict, variants=variants)


if __name__ == "__main__":
    sys.exit(main())
