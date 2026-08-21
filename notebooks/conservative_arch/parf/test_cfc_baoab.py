"""Correctness tests for the CfC / BAOAB integrator path.

Run directly (CPU, a few seconds)::

    python test_cfc_baoab.py

The tests are ordered from "the maths is right" to "the model still
trains", and three of them are load-bearing for interpreting any
CfC-vs-Verlet training comparison:

  * :func:`test_analytic_vtheta_equivalence` -- switching V_theta's force
    from autograd to its closed form must leave *every parameter
    gradient* unchanged.  If it did not, the "analytic V_theta" arm would
    be a different model, not a different implementation.
  * :func:`test_cfc_force_preservation` -- the CfC split must reproduce
    the same force field as the unsplit integrator, with the two agreeing
    to third order in dt.  A second-order discrepancy would mean the
    propagator changed the physics rather than the integration.
  * :func:`test_cfc_stiffness_immunity` -- the reason for the whole
    exercise: a well sharp enough to blow up the explicit step must leave
    the CfC step bounded.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from cfc_baoab import cfc_substep, ou_step                      # noqa: E402
from model_aniso_gaussian_vtheta import (                       # noqa: E402
    AnisotropicDepthConditionedGaussianVTheta,
    install_aniso_depth_routing,
)
from model_fock_parf_multixi import (                           # noqa: E402
    FockMultiXiPARFConfig,
    FockMultiXiPARFLM,
)

TOL = 1e-5


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _make_model(integrator="verlet", analytic=False, d=16, L=3, seed=0,
                **cfg_kwargs):
    """A small CPU Fock-PARFLM with the production aniso-Gaussian V_theta."""
    torch.manual_seed(seed)
    cfg = FockMultiXiPARFConfig(
        vocab_size=97, d=d, max_len=32, L=L, v_hidden=32, v_depth=2,
        dt=1.0, fixed_gamma=0.10, init_gamma=1.0,
        mass_mode="global",
        causal_force=True, ln_after_step=True,
        xi_channels=3, xi_alpha_inits=[0.5, 0.9, 0.99],
        xi_learnable=True, xi_alpha_init_mode="explicit",
        v_phi_kind="structural_competitive",
        v_phi_d_type=8, v_phi_d_angle=4,
        v_phi_phi_hidden=16, v_phi_theta_hidden=16, v_phi_mlp_hidden=16,
        v_phi_n_heads=2, top_k=4, score_head_hidden=8,
        use_gathered_v_phi=True, use_layer_checkpoint=False,
        ln_before_distance=True, per_layer_v_phi_scale=True,
        fock_version="v2", n_registers=4, d_k=8,
        creation_gate_hidden=8, stack_discipline=True,
        reverse_channel=True, reverse_channel_stable=True,
        reverse_channel_per_layer=True, reverse_channel_warmup_steps=0,
        register_repulsion=False,
        prefix_causal_registers=True,
        use_output_bias=True, tie_embeddings=False,
        integrator=integrator, vtheta_analytic_force=analytic,
        **cfg_kwargs,
    )
    torch.manual_seed(seed)
    model = FockMultiXiPARFLM(cfg)
    torch.manual_seed(seed)
    model.V_theta = AnisotropicDepthConditionedGaussianVTheta(
        d=d, K=4, n_ctx=cfg.xi_channels, n_layers=L, rank=2,
        init_log_precision=-math.log(d), precision_max=2.0 / d,
    )
    install_aniso_depth_routing(model)
    model.eval()
    return model, cfg


def _layer_inputs(model, cfg, B=2, T=6, seed=1):
    torch.manual_seed(seed)
    d = cfg.d
    h = torch.randn(B, T, d, requires_grad=True)
    v = torch.randn(B, T, d) * 0.1
    m_b = torch.ones(B, T, 1)
    return h, v, m_b


# ---------------------------------------------------------------------------
# 1. Propagator maths
# ---------------------------------------------------------------------------
def test_propagator_exactness():
    """cfc_substep reproduces the analytic harmonic solution."""
    torch.manual_seed(0)
    B, T, d = 2, 3, 4
    m = torch.ones(B, T, 1)
    k = torch.rand(B, T, d) * 9.0 + 1.0
    mu = torch.randn(B, T, d)
    h, v = torch.randn(B, T, d), torch.randn(B, T, d)
    dt = 0.83

    h1, v1 = cfc_substep(h, v, -k * (h - mu), k, m, dt)

    omega = (k / m).sqrt()
    x = h - mu
    h_exact = mu + x * torch.cos(omega * dt) + (v / omega) * torch.sin(omega * dt)
    v_exact = -omega * x * torch.sin(omega * dt) + v * torch.cos(omega * dt)
    assert (h1 - h_exact).abs().max() < TOL
    assert (v1 - v_exact).abs().max() < TOL
    print("  [ok] propagator matches the analytic harmonic solution")


def test_ou_exact_decay():
    """The O-step decays by exp(-gamma dt), not the Verlet 1/(1+gamma dt)."""
    v = torch.randn(2, 3, 4)
    gamma, dt = 0.30, 1.0
    out = ou_step(v, gamma, dt, T=0.0)
    assert (out - math.exp(-gamma * dt) * v).abs().max() < TOL
    verlet_factor = 1.0 / (1.0 + gamma * dt)
    rel = abs(math.exp(-gamma * dt) - verlet_factor) / verlet_factor
    assert rel > 0.03, "expected the two friction models to differ measurably"
    print(f"  [ok] O-step friction exact  (differs from Verlet's by {rel:.1%} "
          f"at gamma={gamma})")


# ---------------------------------------------------------------------------
# 2. The analytic V_theta force is the same model, not a different one
# ---------------------------------------------------------------------------
def test_analytic_vtheta_equivalence():
    """Closed-form V_theta force must give identical parameter gradients."""
    grads = {}
    for analytic in (False, True):
        model, cfg = _make_model(integrator="verlet", analytic=analytic)
        model.train()
        torch.manual_seed(7)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        y = torch.randint(0, cfg.vocab_size, (2, 8))
        _, loss = model(x, y)
        loss.backward()
        grads[analytic] = {
            n: p.grad.detach().clone()
            for n, p in model.named_parameters() if p.grad is not None
        }
        grads[f"loss_{analytic}"] = float(loss)

    assert abs(grads["loss_False"] - grads["loss_True"]) < 1e-6, (
        grads["loss_False"], grads["loss_True"])

    common = set(grads[False]) & set(grads[True])
    assert len(common) > 20, f"suspiciously few shared params: {len(common)}"
    worst, worst_name = 0.0, ""
    for n in common:
        a, b = grads[False][n], grads[True][n]
        denom = max(a.abs().max().item(), 1e-8)
        err = (a - b).abs().max().item() / denom
        if err > worst:
            worst, worst_name = err, n
    assert worst < 1e-4, f"gradient mismatch {worst:.2e} at {worst_name}"

    # ...and V_theta really is being trained through the closed form.
    vt = [n for n in common if n.startswith("V_theta")]
    assert vt, "no V_theta parameters received gradients"
    assert any(grads[True][n].abs().max() > 0 for n in vt)
    print(f"  [ok] analytic V_theta force: identical loss and gradients "
          f"(worst rel. err {worst:.1e} over {len(common)} tensors)")


# ---------------------------------------------------------------------------
# 3. Verlet path is untouched by the refactor
# ---------------------------------------------------------------------------
def test_verlet_formula_unchanged():
    """_layer_step still computes the historical damped-Verlet update."""
    model, cfg = _make_model(integrator="verlet")
    h, v, m_b = _layer_inputs(model, cfg)
    dt, gamma = cfg.dt, model.gamma
    h_prev = (h - dt * v).detach()

    model.V_theta.set_active_layer(0)
    got = model._layer_step(h, h_prev, m_b, gamma, dt, layer_idx=0)

    # Reference: force from a single joint autograd call, then the
    # historical update, written out longhand.
    h_ref = h.detach().clone().requires_grad_(True)
    xis = model.xi_module(h_ref.detach())
    model.V_theta.set_active_layer(0)
    U = model.V_theta(xis, h_ref).sum() + model._pair_potential(h_ref, 0)
    grad_U, = torch.autograd.grad(U, h_ref)
    f = -grad_U
    denom = 1.0 + dt * gamma
    expect = h_ref + (h_ref - h_prev) / denom + (dt * dt / (m_b * denom)) * f
    expect = model._project(expect)

    err = (got - expect).abs().max().item()
    assert err < 1e-5, err

    h_new, h_prev_out = model._layer_step_ex(h, h_prev, m_b, gamma, dt, 0)
    assert torch.equal(h_prev_out, h), "Verlet must pass h through as h_prev"
    print(f"  [ok] Verlet update unchanged (max dev {err:.1e}) and "
          f"_layer_step_ex passes h through")


# ---------------------------------------------------------------------------
# 4. The CfC split preserves the force field
# ---------------------------------------------------------------------------
def _step_positions(model, cfg, integrator, dt):
    """One layer step from a fixed (h, v), returning (h, v, h_new)."""
    model.cfg.integrator = integrator
    h, v, m_b = _layer_inputs(model, cfg)
    h, v, m_b = h.double(), v.double(), m_b.double()
    h = h.detach().requires_grad_(True)
    h_prev = (h - dt * v).detach()
    model.V_theta.set_active_layer(0)
    h_new, _ = model._layer_step_langevin(h, h_prev, m_b, 0.0, dt, layer_idx=0)
    return h, v, h_new


def test_cfc_force_preservation():
    """CfC and plain BAOAB must agree to O(dt^3): same forces, different flow.

    Both schemes expand to ``h + dt v + (dt^2/2m) f + O(dt^3)`` with the
    *same* f.  So their difference must fall by 8x when dt is halved.  If
    the CfC propagator had changed the force field rather than only the
    way it is integrated, the leading discrepancy would be O(dt^2) and the
    ratio would be ~4.

    Run in float64: at float32 the true dt^3 term is below the rounding
    floor and this test would just measure ulp noise.
    """
    model, cfg = _make_model(integrator="baoab")
    model.cfg.ln_after_step = False        # LN would mask the scaling law
    model.double()

    devs, dts = [], (0.2, 0.1)
    for dt in dts:
        _, _, h_plain = _step_positions(model, cfg, "baoab", dt)
        _, _, h_cfc = _step_positions(model, cfg, "baoab_cfc", dt)
        devs.append((h_plain - h_cfc).abs().max().item())

    # Normalise by dt^2 so the comparison is against the force term both
    # schemes are supposed to share, not against the raw displacement.
    rel = devs[0] / (dts[0] ** 2)
    ratio = (devs[0] / dts[0] ** 2) / max(devs[1] / dts[1] ** 2, 1e-300)

    assert rel < 1e-3, f"CfC force term deviates by {rel:.2e} (absolute)"
    assert 1.6 < ratio < 2.6, (
        f"force-term deviation scales as dt^{math.log2(ratio):.2f}, expected "
        f"dt^1 after normalising (i.e. dt^3 raw); a flat ratio (~1) would "
        f"mean the force field itself changed")
    print(f"  [ok] CfC preserves the force field: raw deviation "
          f"{devs[0]:.2e} -> {devs[1]:.2e} when dt halves "
          f"(dt^{math.log2(devs[0]/devs[1]):.2f}, expect dt^3)")


# ---------------------------------------------------------------------------
# 5. The point of the exercise: stiffness immunity
# ---------------------------------------------------------------------------
def test_cfc_stiffness_immunity():
    """A well sharp enough to blow up the explicit step stays bounded."""
    B, T, d = 1, 4, 8
    m = torch.ones(B, T, 1)
    mu = torch.zeros(B, T, d)
    h0 = torch.full((B, T, d), 0.5)
    v0 = torch.zeros(B, T, d)
    dt = 1.0

    # The explicit step is stable only for omega*dt < 2, i.e. K < 4/dt^2 = 4
    # here; the CfC step has no such limit.
    for k_val, tag in ((0.25, "mild"), (1e4, "stiff")):
        k = torch.full((B, T, d), k_val)
        # Explicit (Verlet-style) integration of the same spring.
        h, h_prev = h0.clone(), h0.clone()
        for _ in range(12):
            f = -k * (h - mu)
            h_next = 2 * h - h_prev + (dt * dt / m) * f
            h_prev, h = h, h_next
        verlet_amp = h.abs().max().item()

        # CfC integration of the same spring.
        hc, vc = h0.clone(), v0.clone()
        for _ in range(12):
            hc, vc = cfc_substep(hc, vc, -k * (hc - mu), k, m, dt)
        cfc_amp = hc.abs().max().item()

        if tag == "mild":
            assert verlet_amp < 10.0 and cfc_amp < 10.0
        else:
            # 12 explicit steps at omega*dt = 100 amplify by ~1e48, which
            # overflows float32 outright -- exactly the runaway that shows
            # up in training as a gradient spike.
            assert not (verlet_amp < 1e6), (
                f"expected the explicit step to diverge at K={k_val}, "
                f"got {verlet_amp:.2e}")
            assert cfc_amp <= 0.5 + 1e-4, cfc_amp
        blew_up = not math.isfinite(verlet_amp)
        print(f"  [ok] K={k_val:<8g} ({tag:>5}): explicit -> "
              f"{'overflow (non-finite)' if blew_up else f'{verlet_amp:.3e}'}, "
              f"CfC -> {cfc_amp:.3e}")


# ---------------------------------------------------------------------------
# 6. End-to-end: every integrator trains
# ---------------------------------------------------------------------------
def test_end_to_end_all_integrators():
    """Forward + backward through the full Fock stack, for each integrator."""
    results = {}
    for integrator in ("verlet", "baoab", "baoab_cfc"):
        for ckpt in (False, True):
            model, cfg = _make_model(integrator=integrator, analytic=True)
            model.cfg.use_layer_checkpoint = ckpt
            model.train()
            torch.manual_seed(11)
            x = torch.randint(0, cfg.vocab_size, (2, 8))
            y = torch.randint(0, cfg.vocab_size, (2, 8))
            _, loss = model(x, y)
            loss.backward()
            assert torch.isfinite(loss), f"{integrator}: non-finite loss"
            n_grad = sum(
                1 for p in model.parameters()
                if p.grad is not None and torch.isfinite(p.grad).all()
            )
            n_param = sum(1 for _ in model.parameters())
            assert n_grad > 0.8 * n_param, (
                f"{integrator}: only {n_grad}/{n_param} params got finite grads")
            if not ckpt:
                results[integrator] = float(loss)
        print(f"  [ok] {integrator:<10} loss={results[integrator]:.4f}  "
              f"(with and without layer checkpointing)")

    assert abs(results["verlet"] - results["baoab"]) > 1e-6, (
        "baoab produced a bit-identical result to verlet -- the integrator "
        "flag is probably not reaching the layer step")


def test_velocity_encoding_roundtrip():
    """The BAOAB step's h_prev_out really carries the outgoing velocity."""
    model, cfg = _make_model(integrator="baoab_cfc", analytic=True)
    h, v, m_b = _layer_inputs(model, cfg)
    dt = cfg.dt
    h_prev = (h - dt * v).detach()
    model.V_theta.set_active_layer(0)
    h_new, h_prev_out = model._layer_step_langevin(
        h, h_prev, m_b, model.gamma, dt, layer_idx=0,
    )
    v_out = (h_new - h_prev_out) / dt
    assert torch.isfinite(v_out).all()
    assert not torch.equal(h_prev_out, h), (
        "h_prev_out should encode a velocity, not just pass h through")
    print(f"  [ok] velocity encoded in h_prev_out "
          f"(|v_out|={v_out.abs().mean():.4f})")


# ---------------------------------------------------------------------------
def main():
    tests = [
        test_propagator_exactness,
        test_ou_exact_decay,
        test_analytic_vtheta_equivalence,
        test_verlet_formula_unchanged,
        test_cfc_force_preservation,
        test_cfc_stiffness_immunity,
        test_velocity_encoding_roundtrip,
        test_end_to_end_all_integrators,
    ]
    for t in tests:
        print(f"\n{t.__name__}:")
        t()
    print("\nAll CfC/BAOAB tests passed.")


if __name__ == "__main__":
    main()
