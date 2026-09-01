"""Correctness tests for the joint-context anisotropic Gaussian V_theta.

This is "option A" of the analytic multi-channel integration programme
(see companion_notes/Analytic_Multi_Channel_Integration_in_Structured_Vtheta.md):
a single well bank whose wells are joint functions of *all* K-EMA context
channels, replacing the additive per-channel banks.  The point of the class
is to restore cross-horizon coupling while keeping the three properties that
make structured V_theta worth having:

  * the analytic force ``-grad_h V`` matches autograd exactly, and
  * the CfC-BAOAB harmonic split (diagonal + PSD low-rank) is still exact, and
  * the well parameters remain independent of ``h`` (so both of the above hold).

The load-bearing test is :func:`test_joint_couples_channels`: the additive
bank's potential is separable across channels (its channel-input Hessian is
block-diagonal), while the joint bank's is not -- that off-block-diagonal
curvature is precisely the cross-horizon conjunction the additive form cannot
represent.

Run directly (CPU, a few seconds)::

    python test_joint_context_vtheta.py

or under pytest::

    pytest test_joint_context_vtheta.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from model_aniso_gaussian_vtheta import (                       # noqa: E402
    AnisotropicMultiContextGaussianVTheta,
    JointContextAnisotropicGaussianVTheta,
    AnisotropicDepthConditionedGaussianVTheta,
)

TOL = 1e-4


def _randomise_projections(bank: JointContextAnisotropicGaussianVTheta,
                           seed: int = 0) -> None:
    """Give a-proj / w-proj nonzero weights so a_k and w_k also depend on the
    context (their default init is zero-weight, which would make the coupling
    test rely on mu alone).  Exercises the full joint dependence."""
    torch.manual_seed(seed)
    inner = bank.bank
    with torch.no_grad():
        torch.nn.init.normal_(inner.a_proj.weight, std=0.05)
        torch.nn.init.normal_(inner.w_proj.weight, std=0.05)
        torch.nn.init.normal_(inner.B_proj.weight, std=0.05)


# ---------------------------------------------------------------------------
# 1. Analytic force == autograd grad
# ---------------------------------------------------------------------------
def test_joint_analytic_force_matches_autograd():
    torch.manual_seed(0)
    d, K, n_ctx, rank = 16, 8, 5, 4
    bank = JointContextAnisotropicGaussianVTheta(d=d, K=K, n_ctx=n_ctx, rank=rank)
    _randomise_projections(bank)

    xis = torch.randn(3, 5, n_ctx, d)
    h = torch.randn(3, 5, d, requires_grad=True)

    V = bank(xis, h)
    assert V.shape == (3, 5, 1)
    # analytical_grad returns +grad_h V (the force is its negative); autograd
    # of V.sum() is also +grad_h V, so the two compare directly.
    g_auto, = torch.autograd.grad(V.sum(), h)
    g_ana = bank.analytical_grad(xis, h)
    assert g_ana.shape == h.shape
    max_err = (g_ana - g_auto).abs().max().item()
    assert max_err < TOL, f"analytic grad mismatch: {max_err:.2e}"


# ---------------------------------------------------------------------------
# 2. CfC-BAOAB low-rank harmonic split stays exact
# ---------------------------------------------------------------------------
def test_joint_harmonic_lowrank_split_exact():
    torch.manual_seed(0)
    d, K, n_ctx, rank = 16, 6, 4, 4
    bank = JointContextAnisotropicGaussianVTheta(d=d, K=K, n_ctx=n_ctx, rank=rank)
    _randomise_projections(bank)

    xis = torch.randn(3, 5, n_ctx, d)
    h = torch.randn(3, 5, d)

    k_diag_a, s_a, G, Gmu = bank.harmonic_terms_lowrank(xis, h)
    f_a = s_a - k_diag_a * h
    s_L = torch.einsum('...dp,...p->...d', G, Gmu)
    Gt_h = torch.einsum('...dp,...d->...p', G, h)
    f_L = s_L - torch.einsum('...dp,...p->...d', G, Gt_h)
    f_true = -bank.analytical_grad(xis, h)
    err = (f_a + f_L - f_true).abs().max().item()
    assert err < TOL, f"low-rank split not exact: {err:.2e}"

    # low-rank operator L = G G^T must be PSD; footprint is K*rank modes.
    assert G.shape[-1] == K * rank
    Lmat = torch.einsum('...dp,...ep->...de', G, G)
    eig_min = torch.linalg.eigvalsh(Lmat).min().item()
    assert eig_min > -1e-4, f"L not PSD: {eig_min:.2e}"


# ---------------------------------------------------------------------------
# 3. rank=0 -> harmonic_terms reproduces the FULL force (diagonal precision)
# ---------------------------------------------------------------------------
def test_joint_harmonic_diag_rank0_exact():
    torch.manual_seed(0)
    d, K, n_ctx = 12, 6, 3
    bank = JointContextAnisotropicGaussianVTheta(d=d, K=K, n_ctx=n_ctx, rank=0)
    xis = torch.randn(3, 5, n_ctx, d)
    h = torch.randn(3, 5, d)

    k_diag, s = bank.harmonic_terms(xis, h)
    f_harm = s - k_diag * h
    f_true = -bank.analytical_grad(xis, h)
    err = (f_harm - f_true).abs().max().item()
    assert err < 1e-5, f"rank=0 harmonic model must be exact: {err:.2e}"
    assert (k_diag >= 0).all(), "stiffness must be non-negative"


# ---------------------------------------------------------------------------
# 4. The joint bank couples channels; the additive bank does not (load-bearing)
# ---------------------------------------------------------------------------
def _channel_hessian(bank, n_ctx, d, seed=0):
    """Hessian of V(h; xi) w.r.t. the flattened context, at one token."""
    torch.manual_seed(seed)
    h = torch.randn(1, 1, d)
    xi0 = torch.randn(n_ctx * d)

    def f(xi_flat):
        xis = xi_flat.view(1, 1, n_ctx, d)
        return bank(xis, h).sum()

    return torch.autograd.functional.hessian(f, xi0)  # (n_ctx*d, n_ctx*d)


def test_joint_couples_channels():
    d, K, n_ctx, rank = 4, 3, 2, 2

    joint = JointContextAnisotropicGaussianVTheta(d=d, K=K, n_ctx=n_ctx, rank=rank)
    _randomise_projections(joint, seed=1)
    Hj = _channel_hessian(joint, n_ctx, d, seed=2)
    cross_j = Hj[0:d, d:2 * d].abs().max().item()

    additive = AnisotropicMultiContextGaussianVTheta(
        d=d, K=K, n_ctx=n_ctx, rank=rank,
    )
    # even with randomised weights, the additive form is structurally separable
    with torch.no_grad():
        for b in additive.banks:
            torch.nn.init.normal_(b.mu_proj.weight, std=0.1)
            torch.nn.init.normal_(b.a_proj.weight, std=0.1)
            torch.nn.init.normal_(b.w_proj.weight, std=0.1)
            torch.nn.init.normal_(b.B_proj.weight, std=0.1)
    Ha = _channel_hessian(additive, n_ctx, d, seed=2)
    cross_a = Ha[0:d, d:2 * d].abs().max().item()

    assert cross_a < 1e-8, (
        f"additive bank must be channel-separable (cross block ~0), "
        f"got {cross_a:.2e}"
    )
    assert cross_j > 1e-6, (
        f"joint bank must couple channels (cross block != 0), got {cross_j:.2e}"
    )


# ---------------------------------------------------------------------------
# 5. Drop-in through the depth-conditioned wrapper via coupling='joint'
# ---------------------------------------------------------------------------
def test_depthcond_joint_flag_matches_autograd():
    torch.manual_seed(0)
    d, K, n_ctx, L, rank = 16, 8, 5, 6, 4
    dcv = AnisotropicDepthConditionedGaussianVTheta(
        d=d, K=K, n_ctx=n_ctx, n_layers=L, rank=rank, coupling="joint",
    )
    assert dcv.coupling == "joint"
    # joint bank exposes exactly one underlying bank (clamp / ablation compat)
    assert len(dcv.banks) == 1

    xis = torch.randn(2, 8, n_ctx, d)
    h = torch.randn(2, 8, d, requires_grad=True)

    dcv.set_active_layer(2)
    g_ana = dcv.analytical_grad(xis, h)
    V = dcv(xis, h).sum()
    g_auto, = torch.autograd.grad(V, h)
    max_err = (g_ana - g_auto).abs().max().item()
    assert max_err < TOL, f"depthcond joint analytic grad mismatch: {max_err:.2e}"

    # depth code still differentiates layers
    dcv.set_active_layer(0)
    V0 = dcv(xis, h)
    dcv.set_active_layer(L - 1)
    VL = dcv(xis, h)
    assert not torch.allclose(V0, VL)


def test_depthcond_additive_is_default():
    d, K, n_ctx, L, rank = 8, 4, 3, 4, 2
    dcv = AnisotropicDepthConditionedGaussianVTheta(
        d=d, K=K, n_ctx=n_ctx, n_layers=L, rank=rank,
    )
    assert dcv.coupling == "additive"
    assert len(dcv.banks) == n_ctx  # one bank per channel


# ---------------------------------------------------------------------------
# 6. precision_lr_max is settable on the joint bank (ablation compatibility)
# ---------------------------------------------------------------------------
def test_precision_lr_max_settable_on_joint_banks():
    d, K, n_ctx, rank = 12, 6, 4, 4
    bank = JointContextAnisotropicGaussianVTheta(d=d, K=K, n_ctx=n_ctx, rank=rank)
    for b in bank.banks:                 # the notebook ablation iterates this
        b._precision_lr_max = 2.0
    assert bank.bank._precision_lr_max == 2.0

    # the cap actually bounds sigma_max(B_k)^2 <= precision_lr_max
    torch.manual_seed(0)
    with torch.no_grad():
        bank.bank.B_proj.weight.mul_(50.0)
        bank.bank.B_proj.bias.add_(5.0)
    xis = torch.randn(2, 3, n_ctx, d)
    _, _, _, B = bank.context_components(xis)
    gram = torch.einsum('...dr,...ds->...rs', B, B)
    sig_max_sq = torch.linalg.eigvalsh(gram)[..., -1].max().item()
    assert sig_max_sq <= 2.0 + 1e-5, sig_max_sq


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  {t.__name__}  OK")
    print(f"\nAll {len(tests)} joint-context V_theta tests passed.")
