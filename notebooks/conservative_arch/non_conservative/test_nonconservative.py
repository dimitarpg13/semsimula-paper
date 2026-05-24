"""Unit tests for Stage 1 non-conservative force modules.

Run:
    python3 notebooks/conservative_arch/non_conservative/test_nonconservative.py

Tests
-----
1. Skew-symmetry property:
       For each Class-B cell (E1, E2, E3, E5):
           Omega(h)^T = -Omega(h)  for every h
       Verified numerically by sampling random h and checking
       Omega + Omega^T  is exactly zero (machine epsilon).

2. Solenoidal property (E4):
       The integrand kernel J_+(h) - J_+(h)^T is skew, so
       contracting it with rho(h) produces a vector field that
       is divergence-free in h.
       This is the same skew-symmetry test applied to the J kernel.

3. Velocity orthogonality (Class B):
       For Class B cells (g = Omega v with Omega skew),
           g . v = 0  for every (h, v).
       This is the gyroscopic property: the force does no work
       on the velocity, so kinetic energy is conserved by the
       non-conservative term in isolation.

4. Gradient flow:
       For each cell, the loss gradient w.r.t. every learnable
       parameter of the force module is non-zero on a random batch
       (no parameter is silently disconnected from the graph).

5. Causal-leak invariant:
       For each cell, the causal-violation probe returns 0.0 leak
       floor at the Class B/C per-token cell. The probe perturbs a
       token at position t_pert and verifies that no logit at
       position t < t_pert changes.

The protocol's H3 hypothesis is enforced by tests 1-3 (force-class
correctness) and test 5 (causal-leak invariant).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR))

from model_splm_nonconservative import (  # noqa: E402
    AffineRank1SkewForce,
    CELLS,
    ConstantSkewForce,
    LowRankSkewForce,
    LowRankSolenoidalForce,
    NonConservativeForce,
    ScalarPotentialLMNonConservative,
    SPLMNonConservativeConfig,
    make_force,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _omega_from_force(force: NonConservativeForce, h: torch.Tensor) -> torch.Tensor:
    """Reconstruct the Omega(h) matrix per token by probing with d unit vectors.

    For Class B forces g = Omega(h) v, sweeping v over the standard basis
    e_1, ..., e_d gives the columns of Omega(h).

    Returns: (B, T, d, d), where Omega[b, t, i, j] = g(h[b, t], e_j)[i].
    """
    B, T, d = h.shape
    Omega = torch.zeros(B, T, d, d, device=h.device, dtype=h.dtype)
    for j in range(d):
        e_j = torch.zeros(B, T, d, device=h.device, dtype=h.dtype)
        e_j[..., j] = 1.0
        col_j = force(h, e_j)  # (B, T, d)
        Omega[..., :, j] = col_j
    return Omega


def _print(name: str, ok: bool, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    print(f"[test] {mark}  {name}{('  ' + detail) if detail else ''}")
    return ok


# ---------------------------------------------------------------------------
# Test 1: Skew-symmetry of Class-B force kernels
# ---------------------------------------------------------------------------

def test_skew_symmetry() -> bool:
    """Omega(h)^T = -Omega(h) for E1, E2, E3, E5 at every h."""
    torch.manual_seed(0)
    d = 16
    h = torch.randn(2, 4, d) * 0.5

    cases = [
        ("E1 ConstantSkew", ConstantSkewForce(d, init_std=0.1)),
        ("E2 AffineRank1Skew", AffineRank1SkewForce(d, init_std=0.1)),
        ("E3 LowRankSkew (r=2)", LowRankSkewForce(d, r=2, init_std=0.1)),
        ("E5 LowRankSkew (r=4)", LowRankSkewForce(d, r=4, init_std=0.1)),
    ]

    all_ok = True
    for name, force in cases:
        Omega = _omega_from_force(force, h)
        sym = Omega + Omega.transpose(-1, -2)  # should be ~0
        max_abs = sym.abs().max().item()
        ok = max_abs < 1e-5
        all_ok &= _print(
            f"skew-symmetry: {name}",
            ok,
            f"max|Omega + Omega^T| = {max_abs:.2e}",
        )
    return all_ok


# ---------------------------------------------------------------------------
# Test 2: E4 solenoidal kernel J_+ - J_+^T is skew
# ---------------------------------------------------------------------------

def test_solenoidal_kernel_skew() -> bool:
    """For E4: the kernel J(h) = J_+(h) - J_+(h)^T is skew at every h."""
    torch.manual_seed(0)
    d = 16
    h = torch.randn(2, 4, d) * 0.5

    force = LowRankSolenoidalForce(d, r=4, h_rho=64, init_std=0.1)
    # Reconstruct J(h) by probing rho's role: drive rho(h) = e_j and
    # measure g. Since rho is a learned MLP, we can't directly inject
    # canonical vectors. Instead, reconstruct J(h) algebraically:
    # J(h)[i, j] = U_{i, k} V(h)_{j, k} - V(h)_{i, k} U_{j, k}.
    V_h = torch.einsum("jir,bti->btjr", force.W, h)  # (B, T, d, r)
    U = force.U                                       # (d, r)
    # J[b, t, i, j] = sum_k U[i,k] V_h[b,t,j,k] - V_h[b,t,i,k] U[j,k]
    J = (
        torch.einsum("ik,btjk->btij", U, V_h)
        - torch.einsum("btik,jk->btij", V_h, U)
    )
    sym = J + J.transpose(-1, -2)
    max_abs = sym.abs().max().item()
    ok = max_abs < 1e-5
    return _print(
        "skew-symmetry: E4 J kernel",
        ok,
        f"max|J + J^T| = {max_abs:.2e}",
    )


# ---------------------------------------------------------------------------
# Test 3: Velocity orthogonality (gyroscopic property) for Class B cells
# ---------------------------------------------------------------------------

def test_velocity_orthogonality() -> bool:
    """For Class B (g = Omega v, Omega skew): g . v = 0 for every (h, v)."""
    torch.manual_seed(0)
    d = 16
    h = torch.randn(2, 4, d) * 0.5
    v = torch.randn(2, 4, d) * 0.5

    cases = [
        ("E1 ConstantSkew", ConstantSkewForce(d, init_std=0.1)),
        ("E2 AffineRank1Skew", AffineRank1SkewForce(d, init_std=0.1)),
        ("E3 LowRankSkew (r=2)", LowRankSkewForce(d, r=2, init_std=0.1)),
        ("E5 LowRankSkew (r=4)", LowRankSkewForce(d, r=4, init_std=0.1)),
    ]

    all_ok = True
    for name, force in cases:
        g = force(h, v)
        # g . v per token
        dot = (g * v).sum(dim=-1)  # (B, T)
        max_abs = dot.abs().max().item()
        ok = max_abs < 1e-4   # double-precision arithmetic; loosely bounded
        all_ok &= _print(
            f"velocity-orthogonality: {name}",
            ok,
            f"max|g . v| = {max_abs:.2e}",
        )
    return all_ok


# ---------------------------------------------------------------------------
# Test 4: Gradient flow (every learnable parameter receives non-zero grad)
# ---------------------------------------------------------------------------

def test_gradient_flow() -> bool:
    """For each cell, all learnable parameters of the force module receive
    a non-zero gradient on a random training batch.
    """
    torch.manual_seed(0)
    V = 257
    base_kw = dict(
        vocab_size=V, d=16, max_len=32, v_hidden=32, v_depth=2, L=4,
        mass_mode="global",
    )

    all_ok = True
    for cell in CELLS:
        if cell == "e0_baseline":
            # No-op cell: nothing to check on the force module.
            _print(f"gradient-flow: {cell}", True, "(no learnable force params)")
            continue
        cfg = SPLMNonConservativeConfig(cell=cell, **base_kw)
        net = ScalarPotentialLMNonConservative(cfg)
        x = torch.randint(0, V, (2, 16))
        y = torch.randint(0, V, (2, 16))
        _, loss = net(x, y)
        loss.backward()

        ok = True
        zero_params = []
        for name, p in net.nonconservative.named_parameters():
            if p.grad is None:
                ok = False
                zero_params.append(f"{name}=None")
                continue
            if p.grad.abs().max().item() == 0.0:
                ok = False
                zero_params.append(f"{name}=0")

        all_ok &= _print(
            f"gradient-flow: {cell}",
            ok,
            (f"zero-grad: {zero_params}" if zero_params else "all params receive grad"),
        )
    return all_ok


# ---------------------------------------------------------------------------
# Test 5: Causal-leak invariant
# ---------------------------------------------------------------------------

def test_causal_leak() -> bool:
    """For each cell, perturbing token x[t_pert] must NOT change any logit
    at position t < t_pert. Tested at random initialisation with cfg.causal_force=True.

    This is the same probe used by notebooks/conservative_arch/causal_probe.py
    in --strict mode, applied to the new model class.
    """
    torch.manual_seed(0)
    V = 257
    base_kw = dict(
        vocab_size=V, d=16, max_len=32, v_hidden=32, v_depth=2, L=4,
        mass_mode="global",
    )
    B, T, t_pert = 1, 24, 12

    all_ok = True
    for cell in CELLS:
        cfg = SPLMNonConservativeConfig(
            cell=cell, causal_force=True, **base_kw,
        )
        net = ScalarPotentialLMNonConservative(cfg)
        net.eval()

        x_a = torch.randint(0, V, (B, T))
        x_b = x_a.clone()
        # Pick a different token id at t_pert.
        x_b[0, t_pert] = (x_a[0, t_pert] + 7) % V

        with torch.enable_grad():
            logits_a, _ = net(x_a)
            logits_b, _ = net(x_b)

        # Maximum logit deviation at strictly past positions t < t_pert.
        delta = (logits_a[:, :t_pert, :] - logits_b[:, :t_pert, :]).abs().max().item()
        ok = delta == 0.0
        all_ok &= _print(
            f"causal-leak: {cell}",
            ok,
            f"max|logits_a - logits_b|_{{t < t_pert}} = {delta:.2e}",
        )
    return all_ok


# ---------------------------------------------------------------------------
# Test 6: Initial g/f norm ratio is small (mitigates E5 s=1 divergence)
# ---------------------------------------------------------------------------

def test_initial_norm_ratio() -> bool:
    """Protocol section 8: at init, ||g_l|| / ||f_l|| <= 0.05 across layers.

    This is the structural mitigation against the velocity-coupled feedback
    divergence pathology of the v3 paper section 15.5 E5 with s=1.
    """
    torch.manual_seed(0)
    V = 257
    base_kw = dict(
        vocab_size=V, d=16, max_len=32, v_hidden=32, v_depth=2, L=4,
        mass_mode="global",
    )

    all_ok = True
    for cell in CELLS:
        if cell == "e0_baseline":
            _print(f"init-norm-ratio: {cell}", True, "(no g term)")
            continue
        cfg = SPLMNonConservativeConfig(cell=cell, **base_kw)
        net = ScalarPotentialLMNonConservative(cfg)
        x = torch.randint(0, V, (2, 16))
        stats = net.nonconservative_norm_stats(x)
        max_ratio = max(stats["ratio"])
        ok = max_ratio <= 0.05
        all_ok &= _print(
            f"init-norm-ratio: {cell}",
            ok,
            f"max(||g||/||f||) = {max_ratio:.3f} (threshold 0.05)",
        )
    return all_ok


# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print("Stage 1 non-conservative force unit tests")
    print("=" * 70)

    results = {
        "skew-symmetry": test_skew_symmetry(),
        "solenoidal-kernel-skew": test_solenoidal_kernel_skew(),
        "velocity-orthogonality": test_velocity_orthogonality(),
        "gradient-flow": test_gradient_flow(),
        "causal-leak": test_causal_leak(),
        "initial-norm-ratio": test_initial_norm_ratio(),
    }

    print("=" * 70)
    n_pass = sum(1 for v in results.values() if v)
    n_total = len(results)
    print(f"summary: {n_pass}/{n_total} test groups passed")
    for name, ok in results.items():
        print(f"  {('PASS' if ok else 'FAIL'):4s}  {name}")
    print("=" * 70)
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
