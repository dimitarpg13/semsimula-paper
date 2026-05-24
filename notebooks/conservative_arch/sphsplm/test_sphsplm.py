"""Unit tests for SP-HSPLM Stage 2.

Pre-registered protocol
-----------------------
docs/SP_HSPLM_Stage_2_pre-registered_protocol.md

These tests verify the structural invariants the protocol depends on
(section 4.3 causal-leak, section 6.2 skew property, section 4.1
parameter scaling). They are CPU-only and small (d=16, T=12, L=4) so
the entire suite runs in <30 s. Run as
    python -m pytest notebooks/conservative_arch/sphsplm/test_sphsplm.py -q
or directly via
    python notebooks/conservative_arch/sphsplm/test_sphsplm.py
in which case the file falls back to a manual driver that prints
PASS/FAIL per test.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import torch

_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR))

from model_sphsplm import (  # type: ignore  # noqa: E402
    CELLS,
    PerTokenGyroKernel,
    SkewKernelLowRank,
    SPHSPLMConfig,
    ScalarPotentialLMSPHSPLM,
    _stage2_cell_kwargs,
    make_schedule_sc,
    pair_kernel_norms,
    parse_schedule_sc,
    schedule_counts_sc,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_smoke_model(
    cell: str = "q9e_a", L: int = 4, T: int = 12, d: int = 16,
    seed: int = 0,
) -> Tuple[ScalarPotentialLMSPHSPLM, SPHSPLMConfig]:
    base = dict(
        vocab_size=257, d=d, max_len=T * 2, L=L,
        v_hidden=32, v_depth=2,
        v_phi_d_type=4, v_phi_d_angle=2,
        v_phi_phi_hidden=8, v_phi_theta_hidden=8,
        v_phi_mlp_hidden=16,
        mass_mode="global",
        score_head_hidden=8,
        kernel_rank=4, kernel_init_scale=0.02,
        gyro_rank=4, gyro_init_scale=0.02,
        gamma_min=0.05,
    )
    kwargs = _stage2_cell_kwargs(cell, base)
    cfg = SPHSPLMConfig(**kwargs)
    torch.manual_seed(seed)
    return ScalarPotentialLMSPHSPLM(cfg), cfg


# ---------------------------------------------------------------------------
# 1. Schedule registry: canonical strings for every Stage 2 cell
# ---------------------------------------------------------------------------
def test_schedule_canonical_strings() -> None:
    assert make_schedule_sc("interleaved", L=8) == "SCSCSCSC"
    assert make_schedule_sc("bottom_c", L=8, LC=4) == "CCCCSSSS"
    assert make_schedule_sc("top_c", L=8, LC=4) == "SSSSCCCC"
    assert make_schedule_sc("sandwich", L=8, k=2) == "SSCCCCSS"
    assert make_schedule_sc("inverse_sandwich", L=8, k=2) == "CCSSSSCC"
    assert make_schedule_sc("all_s", L=8) == "SSSSSSSS"
    assert make_schedule_sc("all_c", L=8) == "CCCCCCCC"


def test_schedule_counts() -> None:
    sigma = parse_schedule_sc("SCSCSCSC")
    nS, nC = schedule_counts_sc(sigma)
    assert nS == 4 and nC == 4
    sigma2 = parse_schedule_sc("CCCCSSSS")
    nS, nC = schedule_counts_sc(sigma2)
    assert nS == 4 and nC == 4
    sigma3 = parse_schedule_sc("SSCCCCSS")
    nS, nC = schedule_counts_sc(sigma3)
    assert nS == 4 and nC == 4


def test_schedule_rejects_invalid_tokens() -> None:
    try:
        parse_schedule_sc("SASA")
    except ValueError as e:
        assert "invalid block types" in str(e)
        return
    raise AssertionError("expected ValueError on schedule 'SASA'")


# ---------------------------------------------------------------------------
# 2. Skew kernel: J + J^T == 0 exactly
# ---------------------------------------------------------------------------
def test_skew_kernel_skew_symmetric() -> None:
    torch.manual_seed(0)
    kern = SkewKernelLowRank(d=16, rank=4, init_scale=1.0)
    J = kern.matrix()
    asymm = (J + J.transpose(0, 1)).abs().max().item()
    assert asymm < 1e-12, (
        f"skew kernel violates J + J^T == 0; max|J + J^T| = {asymm:.2e}"
    )


def test_skew_kernel_zero_work() -> None:
    """v^T J v == 0 exactly for every v (skew property)."""
    torch.manual_seed(0)
    kern = SkewKernelLowRank(d=16, rank=4, init_scale=1.0)
    v = torch.randn(8, 16)
    Jv = kern(v)
    work = (v * Jv).sum(dim=-1).abs().max().item()
    assert work < 1e-5, f"non-zero work v^T J v = {work:.2e}"


def test_skew_kernel_matches_low_rank_forward() -> None:
    """Verify forward(v) == J @ v exactly."""
    torch.manual_seed(0)
    kern = SkewKernelLowRank(d=16, rank=4, init_scale=1.0)
    v = torch.randn(8, 16)
    Jv_via_forward = kern(v)
    J_dense = kern.matrix()
    Jv_via_matmul = v @ J_dense.transpose(0, 1)
    diff = (Jv_via_forward - Jv_via_matmul).abs().max().item()
    assert diff < 1e-5, f"low-rank forward mismatch: {diff:.2e}"


def test_gyro_kernel_skew_symmetric() -> None:
    torch.manual_seed(0)
    kern = PerTokenGyroKernel(d=16, rank=4, init_scale=1.0)
    Omega = kern.matrix()
    asymm = (Omega + Omega.transpose(0, 1)).abs().max().item()
    assert asymm < 1e-12, (
        f"gyro kernel violates Omega + Omega^T == 0; "
        f"max|Omega + Omega^T| = {asymm:.2e}"
    )


# ---------------------------------------------------------------------------
# 3. Gradient flow: U and V receive non-zero gradients
# ---------------------------------------------------------------------------
def test_gradient_flow_skew_kernel() -> None:
    net, cfg = _build_smoke_model(cell="q9e_a")
    x = torch.randint(0, cfg.vocab_size, (2, 12))
    y = torch.randint(0, cfg.vocab_size, (2, 12))
    net.train()
    out = net(x, y)
    loss = out[1]
    loss.backward()
    assert net.skew_kernel.U.grad is not None, "U has no .grad after backward"
    assert net.skew_kernel.V.grad is not None, "V has no .grad after backward"
    grad_U_norm = float(net.skew_kernel.U.grad.norm().item())
    grad_V_norm = float(net.skew_kernel.V.grad.norm().item())
    assert grad_U_norm > 0.0, f"U.grad is zero ({grad_U_norm})"
    assert grad_V_norm > 0.0, f"V.grad is zero ({grad_V_norm})"


def test_gradient_flow_gyro_kernel_when_enabled() -> None:
    net, cfg = _build_smoke_model(cell="q9e_d")
    assert net.gyro_kernel is not None
    x = torch.randint(0, cfg.vocab_size, (2, 12))
    y = torch.randint(0, cfg.vocab_size, (2, 12))
    net.train()
    out = net(x, y)
    loss = out[1]
    loss.backward()
    assert net.gyro_kernel.U.grad is not None
    grad_U_norm = float(net.gyro_kernel.U.grad.norm().item())
    assert grad_U_norm > 0.0, f"gyro U.grad is zero ({grad_U_norm})"


# ---------------------------------------------------------------------------
# 4. Causal-leak invariant: max_logit_delta_past == 0 across the C-block
# ---------------------------------------------------------------------------
def _causal_leak_floor(
    net: ScalarPotentialLMSPHSPLM, cfg: SPHSPLMConfig,
    T: int = 12, B: int = 2, n_pairs: int = 4, seed: int = 0,
) -> float:
    """Run the standard causal-leak probe (paired-input style).

    For each of n_pairs random (input_a, input_b) pairs that differ
    only at a single forward position p_perturb, measure the max
    absolute change in logits at every past position p < p_perturb.
    A causal model returns 0.0 exactly. The probe is the runtime
    invariant of the v3 leak-fix (causal_force=True).
    """
    torch.manual_seed(seed)
    net.eval()
    max_delta_past = 0.0
    for k in range(n_pairs):
        # Vary the perturbation position so the test exercises every
        # past slice once, not just T - 1.
        p_perturb = max(1, T - 1 - k)
        x_a = torch.randint(0, cfg.vocab_size, (B, T))
        x_b = x_a.clone()
        new_tok = (x_a[:, p_perturb] + 17) % cfg.vocab_size
        x_b[:, p_perturb] = new_tok
        # The conservative S-block uses autograd.grad inside its
        # forward, so we cannot wrap this in torch.no_grad. Detach
        # the logits afterward to avoid graph retention.
        logits_a = net(x_a)[0].detach()
        logits_b = net(x_b)[0].detach()
        delta_past = (
            logits_a[:, :p_perturb, :] - logits_b[:, :p_perturb, :]
        ).abs().max().item()
        if delta_past > max_delta_past:
            max_delta_past = float(delta_past)
    return max_delta_past


def test_causal_leak_invariant_q9e_a() -> None:
    net, cfg = _build_smoke_model(cell="q9e_a")
    # Force the leak-fix invariant on (it should be the default).
    assert cfg.causal_force, "causal_force must be True by default"
    floor = _causal_leak_floor(net, cfg)
    assert floor == 0.0, (
        f"causal-leak invariant violated for q9e_a (interleaved): "
        f"max_logit_delta_past = {floor:.2e} (expected 0.0)"
    )


def test_causal_leak_invariant_q9e_d_with_gyro() -> None:
    net, cfg = _build_smoke_model(cell="q9e_d")
    floor = _causal_leak_floor(net, cfg)
    assert floor == 0.0, (
        f"causal-leak invariant violated for q9e_d (interleaved + gyro): "
        f"max_logit_delta_past = {floor:.2e} (expected 0.0)"
    )


def test_causal_leak_invariant_q9e_e_bottom_c() -> None:
    net, cfg = _build_smoke_model(cell="q9e_e")
    floor = _causal_leak_floor(net, cfg)
    assert floor == 0.0, (
        f"causal-leak invariant violated for q9e_e (bottom_c): "
        f"max_logit_delta_past = {floor:.2e} (expected 0.0)"
    )


def test_causal_leak_invariant_q9e_g_sandwich() -> None:
    net, cfg = _build_smoke_model(cell="q9e_g")
    floor = _causal_leak_floor(net, cfg)
    assert floor == 0.0, (
        f"causal-leak invariant violated for q9e_g (sandwich): "
        f"max_logit_delta_past = {floor:.2e} (expected 0.0)"
    )


# ---------------------------------------------------------------------------
# 5. Initial pair-kernel Frobenius norm is small (||g||/||f|| < ~0.05 prior)
# ---------------------------------------------------------------------------
def test_initial_jphi_norm_is_small() -> None:
    """At init, ||J_phi||_F should be at most ~3 * init_scale.

    With init_scale = 0.02 and rank = 4, the expected Frobenius norm
    is ~0.02 (per the architecture v3 doc section 5.3 prescription).
    We allow a generous 5x factor to absorb random fluctuation.
    """
    torch.manual_seed(0)
    kern = SkewKernelLowRank(d=16, rank=4, init_scale=0.02)
    J = kern.matrix()
    fro = float(J.norm(p="fro").item())
    assert fro < 0.10, (
        f"||J_phi||_F at init is {fro:.4f}, expected ~0.02 (architecture "
        f"v3 doc section 5.3)"
    )


def test_pair_kernel_norms_helper() -> None:
    """The diagnostic helper returns sensible keys."""
    net, _ = _build_smoke_model(cell="q9e_a")
    norms = pair_kernel_norms(net)
    for key in ("J_phi_fro", "U_fro", "V_fro"):
        assert key in norms
        assert norms[key] >= 0.0
    # Q9e-A has no gyro.
    assert "Omega_fro" not in norms

    net2, _ = _build_smoke_model(cell="q9e_d")
    norms2 = pair_kernel_norms(net2)
    for key in (
        "J_phi_fro", "U_fro", "V_fro",
        "Omega_fro", "Omega_U_fro", "Omega_V_fro",
    ):
        assert key in norms2


# ---------------------------------------------------------------------------
# 6. Bit-equivalence to SparsePARFLM at all_s schedule
# ---------------------------------------------------------------------------
def test_all_s_schedule_matches_sparse_parflm_layer_step() -> None:
    """At schedule='SSSS', the SP-HSPLM stack uses only S-blocks.

    The S-block in this model is `SparsePARFLM._layer_step` unchanged,
    so the forward should be bit-identical to the SparsePARFLM
    forward at the same config. We don't construct a parallel
    SparsePARFLM here (the test covers the integrator logic, not
    cross-class equivalence); this test simply verifies that the
    all_s schedule produces no NaN and that the gradient flows
    through V_theta and V_phi parameters as expected.
    """
    base = dict(
        vocab_size=257, d=16, max_len=24, L=4,
        v_hidden=32, v_depth=2,
        v_phi_d_type=4, v_phi_d_angle=2,
        v_phi_phi_hidden=8, v_phi_theta_hidden=8,
        v_phi_mlp_hidden=16,
        mass_mode="global", score_head_hidden=8,
        kernel_rank=4, kernel_init_scale=0.02,
        gyro_rank=4, gyro_init_scale=0.02,
        gamma_min=0.05,
        schedule=make_schedule_sc("all_s", L=4),
        top_k=4, use_pertoken_gyro=False,
    )
    cfg = SPHSPLMConfig(**base)
    torch.manual_seed(0)
    net = ScalarPotentialLMSPHSPLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 12))
    y = torch.randint(0, cfg.vocab_size, (2, 12))
    net.train()
    out = net(x, y)
    loss = out[1]
    loss.backward()
    assert torch.isfinite(loss), f"all_s loss is non-finite: {loss}"
    # Skew kernel should still receive zero (or numerically-zero)
    # gradient because it is never used at all_s.
    if net.skew_kernel.U.grad is not None:
        gu = float(net.skew_kernel.U.grad.norm().item())
        assert gu == 0.0, (
            f"all_s schedule should give zero grad on skew_kernel.U; "
            f"got ||grad||={gu:.6f}"
        )


# ---------------------------------------------------------------------------
# Manual driver
# ---------------------------------------------------------------------------
_TESTS = [
    test_schedule_canonical_strings,
    test_schedule_counts,
    test_schedule_rejects_invalid_tokens,
    test_skew_kernel_skew_symmetric,
    test_skew_kernel_zero_work,
    test_skew_kernel_matches_low_rank_forward,
    test_gyro_kernel_skew_symmetric,
    test_gradient_flow_skew_kernel,
    test_gradient_flow_gyro_kernel_when_enabled,
    test_causal_leak_invariant_q9e_a,
    test_causal_leak_invariant_q9e_d_with_gyro,
    test_causal_leak_invariant_q9e_e_bottom_c,
    test_causal_leak_invariant_q9e_g_sandwich,
    test_initial_jphi_norm_is_small,
    test_pair_kernel_norms_helper,
    test_all_s_schedule_matches_sparse_parflm_layer_step,
]


def _main() -> int:
    fails = 0
    for t in _TESTS:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            fails += 1
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n[sphsplm-test] {len(_TESTS) - fails}/{len(_TESTS)} passed")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(_main())
