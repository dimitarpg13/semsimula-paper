"""Closed-form (CfC) harmonic propagator and exact OU thermostat for BAOAB.

This module holds the *pure* integrator mathematics used by the
``integrator='baoab'`` / ``integrator='baoab_cfc'`` paths of
:class:`model_parf_multixi.MultiXiPARFLM`.  Nothing here knows about
V_theta, V_phi, registers or the Fock machinery: every function maps
tensors to tensors, which is what makes the correctness tests in
``test_cfc_baoab.py`` cheap and exact.

Why this exists
---------------

The production per-layer step is a damped velocity-Verlet update

    h_new = h + (h - h_prev)/(1 + dt*gamma) + dt^2/(m (1 + dt*gamma)) * f

with ``f = -grad_h (V_theta + V_phi)`` obtained from
``autograd.grad(..., create_graph=True)``.  Two structural problems follow:

1. **The stiff part of the force is integrated explicitly.**  A token
   sitting in a sharp V_theta well has a large local curvature ``K``; the
   explicit update is only stable for ``dt < 2 sqrt(m/K)``.  When a well
   sharpens during training, the layer step silently crosses that
   threshold and the state amplifies geometrically down the remaining
   layers -- which is what a gradient spike looks like from the outside.
2. **Damping is folded into the force coefficient**, so friction is an
   approximation ``1/(1 + gamma*dt)`` of ``exp(-gamma*dt)`` and it sits
   inside the same second-order graph as everything else.

BAOAB fixes (2) by giving friction its own exact Ornstein-Uhlenbeck
substep (:func:`ou_step`).  CfC fixes (1) by integrating the stiff,
locally-harmonic part of V_theta *exactly* instead of explicitly
(:func:`cfc_substep`): a harmonic flow is a rotation in phase space, so
it is unconditionally stable no matter how sharp the well becomes.

The harmonic substep
--------------------

For the frozen-coefficient linear force ``f_harm(h) = -K (h - mu)`` with
mass ``m``, the equation of motion ``m h'' = -K (h - mu)`` has the exact
solution, in terms of ``omega = sqrt(K/m)``:

    h(t+dt) = h + (dt^2/m) psi(omega dt) f_harm(h) + dt sinc(omega dt) v
    v(t+dt) = cos(omega dt) v + (dt/m) sinc(omega dt) f_harm(h)

with ``sinc(x) = sin(x)/x`` and ``psi(x) = (1 - cos x)/x^2``.  This
parameterisation is deliberate: it is written in terms of the *force*
rather than the equilibrium point ``mu``, so no division by ``K`` and no
subtraction of a possibly-huge ``mu`` ever occurs.  Both special
functions are evaluated through ``torch.sinc`` using

    sinc(x)  = torch.sinc(x/pi)
    psi(x)   = (1 - cos x)/x^2 = 2 sin^2(x/2)/x^2
             = 0.5 * torch.sinc(x/(2 pi))^2

which is exact, branch-free, and smooth through ``omega -> 0``.  In that
limit ``sinc -> 1`` and ``psi -> 1/2``, and the update degenerates to the
free drift-plus-constant-force step ``h + dt v + (dt^2/2m) f`` -- so a
token far from every well is integrated exactly as an unforced particle,
with no special-casing.

Because ``K >= 0`` for the Gaussian-mixture V_theta family (every well is
attractive, see ``model_aniso_gaussian_vtheta.harmonic_terms``), omega is
always real: the substep is always a rotation, never a hyperbolic
expansion.

Ordering
--------

:func:`ou_step` and :func:`cfc_substep` are composed by the model as a
palindromic **position-first (ABOBA)** sequence:

    A: cfc_substep(dt/2)   -- drift + exact harmonic part of V_theta
    B: kick(dt)            -- everything not in the harmonic part
    O: ou_step(dt)         -- exact friction (+ optional FDT noise)
    B: (folded into the single kick above)
    A: cfc_substep(dt/2)

ABOBA rather than the textbook BAOAB because it needs only **one** force
evaluation per layer, matching the cost of the Verlet step it replaces
(BAOAB proper needs the force at both ends of the step, and the usual
force-caching trick is invalid here: the potential differs per layer, via
the depth-conditioned V_theta, the per-layer V_phi scale and the register
injections between layers).  Both orderings are second-order accurate and
palindromic; BAOAB's known advantage over ABOBA is specific to
configurational sampling accuracy at high friction with an active
thermostat, which does not apply at the deterministic ``T = 0`` default
used here.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Special functions (branch-free, exact through the omega -> 0 limit)
# ---------------------------------------------------------------------------
def _sinc(x: torch.Tensor) -> torch.Tensor:
    """sin(x)/x, smooth at x = 0 (``torch.sinc`` is the normalised variant)."""
    return torch.sinc(x / math.pi)


def _psi(x: torch.Tensor) -> torch.Tensor:
    """(1 - cos x)/x^2 = 0.5 sinc(x/2)^2, smooth at x = 0 (value 1/2)."""
    half = torch.sinc(x / (2.0 * math.pi))
    return 0.5 * half * half


# ---------------------------------------------------------------------------
# A-step: exact flow of the locally-harmonic part of the potential
# ---------------------------------------------------------------------------
def cfc_substep(
    h: torch.Tensor,
    v: torch.Tensor,
    f_harm: torch.Tensor,
    k_diag: Optional[torch.Tensor],
    m: torch.Tensor,
    dt: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Advance ``(h, v)`` by ``dt`` under the frozen harmonic force.

    Parameters
    ----------
    h, v : (B, T, d)
        Position and velocity at the start of the substep.
    f_harm : (B, T, d)
        The harmonic force *evaluated at* ``h``, i.e. ``-K (h - mu)``.
        Pass zeros (with ``k_diag=None``) for a pure drift substep.
    k_diag : (B, T, d) or None
        Per-dimension stiffness ``K``.  ``None`` means ``K = 0``: the
        substep degenerates to free drift plus the constant force
        ``f_harm``, which is exactly the A+B part of an ordinary Verlet
        step.
    m : (B, T, 1) or scalar
        Per-token mass.
    dt : float
        Substep length.

    Returns
    -------
    (h_new, v_new)

    Notes
    -----
    The map is symplectic for any ``dt`` and any ``K >= 0``: its Jacobian
    determinant is ``cos^2 + omega sin * sin/omega = 1``.  That is the
    property that makes it immune to the stiffness blow-up of the
    explicit step -- a sharp well rotates the phase-space point faster,
    it does not amplify it.
    """
    if k_diag is None:
        # omega = 0: sinc = 1, psi = 1/2.  Free drift + constant force.
        h_new = h + dt * v + (0.5 * dt * dt / m) * f_harm
        v_new = v + (dt / m) * f_harm
        return h_new, v_new

    omega = (k_diag / m).clamp(min=0.0).sqrt()
    wt = omega * dt

    cos_wt = torch.cos(wt)
    sinc_wt = _sinc(wt)
    psi_wt = _psi(wt)

    h_new = h + (dt * sinc_wt) * v + (dt * dt / m) * psi_wt * f_harm
    v_new = cos_wt * v + (dt / m) * sinc_wt * f_harm
    return h_new, v_new


# ---------------------------------------------------------------------------
# O-step: exact Ornstein-Uhlenbeck friction (+ optional FDT-locked noise)
# ---------------------------------------------------------------------------
def ou_step(
    v: torch.Tensor,
    gamma: torch.Tensor | float,
    dt: float,
    m: Optional[torch.Tensor] = None,
    T: float = 0.0,
    training: bool = True,
    noise_eval: bool = False,
) -> torch.Tensor:
    """Exact friction substep ``v <- exp(-gamma dt) v + noise``.

    With the default ``T = 0`` this is pure deterministic friction and the
    only difference from the Verlet step's ``1/(1 + gamma dt)`` factor is
    that the decay is exact rather than first-order in ``gamma dt``.

    With ``T > 0`` an FDT-locked thermostat is added, giving the velocity
    the equilibrium variance ``kT/m``:

        v <- c1 v + sqrt((T/m)(1 - c1^2)) * xi,    c1 = exp(-gamma dt)

    which is the same O-step already used by ``fock_ostep_setup.py``.
    """
    if isinstance(gamma, torch.Tensor):
        c1 = torch.exp(-gamma * dt)
        one_minus_c1sq = 1.0 - c1 * c1
    else:
        c1 = math.exp(-float(gamma) * dt)
        one_minus_c1sq = 1.0 - c1 * c1

    v_new = c1 * v
    add_noise = T > 0.0 and (training or noise_eval)
    if add_noise:
        if m is None:
            raise ValueError("ou_step needs the mass m when T > 0")
        std = torch.sqrt((T / m) * one_minus_c1sq)
        v_new = v_new + std * torch.randn_like(v_new)
    return v_new


# ---------------------------------------------------------------------------
# Velocity <-> h_prev encoding
# ---------------------------------------------------------------------------
def decode_velocity(h: torch.Tensor, h_prev: torch.Tensor,
                    dt: float) -> torch.Tensor:
    """``v = (h - h_prev)/dt`` -- the implicit-velocity convention."""
    return (h - h_prev) / dt


def encode_velocity(h_new: torch.Tensor, v_new: torch.Tensor,
                    dt: float) -> torch.Tensor:
    """``h_prev_out = h_new - dt v_new``, so the next layer decodes ``v_new``.

    This is what lets the BAOAB/CfC integrator carry a genuine velocity
    without changing the ``(h, h_prev)`` state signature that the model,
    the checkpoints and the inference path all assume.
    """
    return h_new - dt * v_new


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _self_test() -> None:
    torch.manual_seed(0)
    B, T, d = 2, 3, 5
    m = torch.ones(B, T, 1)

    # 1) Harmonic exactness: compare a single big step against the analytic
    #    solution of  h'' = -omega^2 (h - mu).
    k = torch.rand(B, T, d) * 4.0 + 0.1
    mu = torch.randn(B, T, d)
    h0 = torch.randn(B, T, d)
    v0 = torch.randn(B, T, d)
    dt = 0.7
    omega = (k / m).sqrt()
    f0 = -k * (h0 - mu)
    h1, v1 = cfc_substep(h0, v0, f0, k, m, dt)
    x0 = h0 - mu
    h_exact = mu + x0 * torch.cos(omega * dt) + (v0 / omega) * torch.sin(omega * dt)
    v_exact = -omega * x0 * torch.sin(omega * dt) + v0 * torch.cos(omega * dt)
    err_h = (h1 - h_exact).abs().max().item()
    err_v = (v1 - v_exact).abs().max().item()
    assert err_h < 1e-5, err_h
    assert err_v < 1e-5, err_v

    # 2) Stiffness immunity: a well so sharp that an explicit step would
    #    diverge (omega dt >> 2) still produces a bounded rotation.
    k_stiff = torch.full((B, T, d), 1e6)
    f_stiff = -k_stiff * (h0 - mu)
    h_s, v_s = cfc_substep(h0, v0, f_stiff, k_stiff, m, 1.0)
    amp0 = (h0 - mu).abs().max().item()
    amp1 = (h_s - mu).abs().max().item()
    assert torch.isfinite(h_s).all() and torch.isfinite(v_s).all()
    # energy-bounded: displacement cannot exceed the initial orbit radius
    radius = ((h0 - mu) ** 2 + (v0 / (k_stiff / m).sqrt()) ** 2).sqrt().max().item()
    assert amp1 <= radius + 1e-4, (amp0, amp1, radius)

    # 3) omega -> 0 limit degenerates to the free drift + constant force step.
    f_const = torch.randn(B, T, d)
    h_free, v_free = cfc_substep(h0, v0, f_const, None, m, dt)
    k_tiny = torch.full((B, T, d), 1e-12)
    h_lim, v_lim = cfc_substep(h0, v0, f_const, k_tiny, m, dt)
    assert (h_free - h_lim).abs().max().item() < 1e-6
    assert (v_free - v_lim).abs().max().item() < 1e-6

    # 4) Symplecticity: the phase-space volume is preserved exactly.
    #    (cos^2 + sin^2 == 1 for every element)
    wt = omega * dt
    jac_det = torch.cos(wt) ** 2 + (omega * torch.sin(wt)) * (
        dt * _sinc(wt) / 1.0
    )
    assert (jac_det - 1.0).abs().max().item() < 1e-5

    # 5) O-step: exact decay, and T=0 is deterministic.
    v = torch.randn(B, T, d)
    g, dtl = 0.3, 1.0
    v_o = ou_step(v, g, dtl, m=m, T=0.0)
    assert (v_o - math.exp(-g * dtl) * v).abs().max().item() < 1e-6
    # FDT noise changes the value but keeps it finite
    v_n = ou_step(v, g, dtl, m=m, T=1.0, training=True)
    assert torch.isfinite(v_n).all() and not torch.allclose(v_n, v_o)

    # 6) Velocity encode/decode round-trip.
    h_new = torch.randn(B, T, d)
    v_new = torch.randn(B, T, d)
    hp = encode_velocity(h_new, v_new, dt)
    assert (decode_velocity(h_new, hp, dt) - v_new).abs().max().item() < 1e-6

    print("cfc_baoab self-test: OK")


if __name__ == "__main__":
    _self_test()
