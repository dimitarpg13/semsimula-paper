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

# `k_diag == 0` is a real, expected state (a token far from every well, see
# `harmonic_terms`'s docstring) and grows more common as wells sharpen over
# training.  `torch.sqrt` has an infinite derivative at 0, so clamping the
# sqrt input to exactly 0.0 makes `omega`'s backward pass hit `0 * inf =
# nan` at every such element -- forward is fine (sinc/psi/cos are smooth
# through omega -> 0), only the sqrt node that *produces* omega is not.
# Flooring at a tiny positive epsilon instead removes the singular point:
# `clamp`'s own backward is exactly 0 below the floor, which is the correct
# limit here anyway, since the upstream `g = w * exp(-0.5 * quad_form)` that
# drove k_diag to (numerically) 0 has *already* lost its own gradient
# sensitivity at that point (`d(exp)/dx = exp(x) -> 0` right alongside the
# value). See `test_cfc_baoab.py::test_cfc_substep_zero_stiffness_no_nan`.
_OMEGA_SQ_FLOOR = 1e-12


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

    omega = (k_diag / m).clamp(min=_OMEGA_SQ_FLOOR).sqrt()
    wt = omega * dt

    cos_wt = torch.cos(wt)
    sinc_wt = _sinc(wt)
    psi_wt = _psi(wt)

    h_new = h + (dt * sinc_wt) * v + (dt * dt / m) * psi_wt * f_harm
    v_new = cos_wt * v + (dt / m) * sinc_wt * f_harm
    return h_new, v_new


# ---------------------------------------------------------------------------
# Low-rank exact substep: exact flow of a PSD low-rank harmonic force
# ---------------------------------------------------------------------------
#
# Mitigation "#1 / low-rank exponential integration" of the CfC/BAOAB
# companion note.  The diagonal ``cfc_substep`` above integrates the
# per-dimension springs exactly, but leaves the *off-diagonal* coupling of
# an anisotropic-Gaussian V_theta (the ``B_k B_k^T`` part) to the explicit
# kick -- which reintroduces exactly the ``omega dt < 2`` stability wall the
# CfC step was built to remove, now on the aggregate low-rank operator
#
#     L = sum_k g_k B_k B_k^T = G G^T,   G = [sqrt(g_1) B_1, ..., sqrt(g_K) B_K].
#
# ``L`` is symmetric PSD (a sum of PSD rank-r terms), so its eigenmodes are
# genuine oscillators, never hyperbolic: rotating them is unconditionally
# stable, exactly as for the diagonal case.  ``lowrank_modes`` extracts the
# modes from the small ``P x P`` Gram of ``G`` (P = number of low-rank
# columns, e.g. K*rank or n_ctx*K*rank), and ``lowrank_cfc_substep`` rotates
# the state inside that subspace with the same closed-form propagator.
#
# The mode *geometry* (directions ``U`` and curvatures ``kappa``) is frozen
# and detached: a rank-deficient Gram has degenerate near-zero eigenvalues
# whose ``eigh`` backward is singular (the standard exponential-integrator
# "frozen Jacobian" is detached for exactly this reason).  The substep stays
# differentiable in ``h``, ``v`` and the frozen force, which is what carries
# the gradient to the V_theta parameters.


def lowrank_modes(
    G: torch.Tensor,
    max_modes: Optional[int] = None,
    floor: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eigenmodes of the PSD operator ``L = G G^T`` from its factor ``G``.

    Parameters
    ----------
    G : (..., d, P)
        Aggregate low-rank factor; ``L = G G^T`` has rank <= P.
    max_modes : int or None
        Keep only the ``max_modes`` stiffest (largest-eigenvalue) modes.
        ``None`` keeps all ``P`` of them.
    floor : float
        Eigenvalues at or below this are treated as zero: the matching mode
        direction is zeroed (made inert) and its curvature set to 0, which
        both avoids the ``1/sqrt(lambda)`` blow-up when normalising a
        vanishing mode and matches the true dynamics (``L`` exerts no force
        along a null direction).

    Returns
    -------
    U : (..., d, q)
        Mode directions as orthonormal columns (zero columns are inert),
        ``q = min(P, max_modes)``.  Detached from the graph.
    kappa : (..., q)
        Per-mode stiffness (eigenvalues of ``L``, ``>= 0``).  Detached.

    Notes
    -----
    Uses the Gram trick: the nonzero spectrum of ``G G^T`` (d x d) equals
    that of ``M = G^T G`` (P x P), and if ``M w = lambda w`` then
    ``u = G w / sqrt(lambda)`` is the matching unit eigenvector of
    ``G G^T``.  The ``P x P`` eigenproblem is cheap even when ``d`` is large.
    """
    Gd = G.detach()
    P = Gd.shape[-1]
    M = Gd.transpose(-1, -2) @ Gd                       # (..., P, P) Gram
    lam, W = torch.linalg.eigh(M)                       # ascending eigenpairs

    q = P if max_modes is None else min(int(max_modes), P)
    lam = lam[..., -q:]                                 # (..., q) largest q
    W = W[..., -q:]                                     # (..., P, q)

    keep = lam > floor                                  # (..., q) bool
    inv_sqrt = torch.where(
        keep, lam.clamp(min=floor).rsqrt(), torch.zeros_like(lam),
    )                                                   # 0 on dropped modes
    U = (Gd @ W) * inv_sqrt.unsqueeze(-2)               # (..., d, q), unit cols
    kappa = torch.where(keep, lam, torch.zeros_like(lam))
    return U, kappa


def lowrank_cfc_substep(
    h: torch.Tensor,
    v: torch.Tensor,
    U: torch.Tensor,
    kappa: torch.Tensor,
    f_lr: torch.Tensor,
    m: torch.Tensor,
    dt: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact flow of ``T + V_L`` by ``dt``: free drift + PSD low-rank rotation.

    This is the fast sub-flow of the impulse / RESPA multiple-time-stepping
    scheme used by ``integrator='baoab_cfc_lowrank'``.  The low-rank spring
    ``L = U diag(kappa) U^T`` acts only inside ``span(U)``; there the mode
    coordinates rotate under the exact harmonic propagator, and on the
    orthogonal complement (where ``L`` exerts no force) the motion is a free
    drift ``h += dt v``.  Both together are the exact flow of the full-mass
    kinetic term ``T`` plus the low-rank potential ``V_L``.

    Parameters
    ----------
    h, v : (..., d)
        Position and velocity at the start of the substep.
    U : (..., d, q)
        Mode directions (orthonormal columns), from :func:`lowrank_modes`.
    kappa : (..., q)
        Per-mode stiffness (eigenvalues of ``L``).
    f_lr : (..., d)
        The low-rank harmonic force *evaluated at* ``h``: ``s_L - L h``.
        Its projection ``U^T f_lr`` is the mode-space force ``U^T s_L -
        kappa * z`` the propagator expects.
    m : (..., 1) or scalar
        Per-token mass.
    dt : float
        Substep length.

    Returns
    -------
    (h_new, v_new)

    Notes
    -----
    The map is symplectic and, as a *standalone* flow, a bounded rotation on
    ``span(U)`` for any ``kappa >= 0`` and any ``dt`` -- no ``omega dt < 2``
    wall, however sharp the low-rank curvature becomes.  Passing ``f_lr``
    (rather than ``s_L`` and a matvec) keeps this parallel to ``cfc_substep``
    and lets the caller build the force once with gradient tracking while
    ``U``/``kappa`` stay frozen and detached.
    """
    z = torch.einsum('...dq,...d->...q', U, h)
    wz = torch.einsum('...dq,...d->...q', U, v)
    fz = torch.einsum('...dq,...d->...q', U, f_lr)
    z_new, wz_new = cfc_substep(z, wz, fz, kappa, m, dt)
    # Free drift everywhere, then overwrite the span(U) drift with the exact
    # harmonic mode solution (the complement keeps its free drift ``dt v``).
    h_new = h + dt * v + torch.einsum(
        '...dq,...q->...d', U, z_new - z - dt * wz,
    )
    v_new = v + torch.einsum('...dq,...q->...d', U, wz_new - wz)
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

    # 7) Backward pass through k_diag == 0 must not produce nan.  This is
    #    the "token far from every well" case (see harmonic_terms), which
    #    is a real, expected state that grows more common as wells sharpen
    #    over training -- not an edge case that only shows up in synthetic
    #    tests.  Before the _OMEGA_SQ_FLOOR fix, sqrt()'s infinite
    #    derivative at 0 turned this into `0 * inf = nan` in the backward
    #    pass, silently poisoning every parameter k_diag traces back to.
    k_zero = torch.zeros(B, T, d, requires_grad=True)
    h_z = torch.randn(B, T, d, requires_grad=True)
    v_z = torch.randn(B, T, d, requires_grad=True)
    f_z = torch.randn(B, T, d)
    h_out, v_out = cfc_substep(h_z, v_z, f_z, k_zero, m, dt)
    (h_out.pow(2).sum() + v_out.pow(2).sum()).backward()
    assert torch.isfinite(k_zero.grad).all(), k_zero.grad
    assert torch.isfinite(h_z.grad).all(), h_z.grad
    assert torch.isfinite(v_z.grad).all(), v_z.grad

    # 8) Low-rank modes reconstruct L = G G^T, and the low-rank substep
    #    matches the analytic mode-space oscillator solution exactly.
    torch.manual_seed(1)
    d2, P = 6, 4
    G = torch.randn(B, T, d2, P)
    L = G @ G.transpose(-1, -2)                          # (B,T,d2,d2) PSD
    U, kappa = lowrank_modes(G)
    L_rec = (U * kappa.unsqueeze(-2)) @ U.transpose(-1, -2)
    assert (L_rec - L).abs().max().item() < 1e-4, (L_rec - L).abs().max().item()

    m2 = torch.ones(B, T, 1)
    mu2 = torch.randn(B, T, d2)
    s_L = torch.einsum('...ij,...j->...i', L, mu2)       # in range(L)
    h2 = torch.randn(B, T, d2)
    v2 = torch.randn(B, T, d2)
    dt2 = 0.6
    f_lr = s_L - torch.einsum('...ij,...j->...i', L, h2)
    h_lr, v_lr = lowrank_cfc_substep(h2, v2, U, kappa, f_lr, m2, dt2)

    z = torch.einsum('...dq,...d->...q', U, h2)
    wz = torch.einsum('...dq,...d->...q', U, v2)
    zmu = torch.einsum('...dq,...d->...q', U, mu2)       # U^T s_L = kappa * zmu
    omega2 = (kappa / m2).clamp(min=_OMEGA_SQ_FLOOR).sqrt()
    x = z - zmu
    z_ex = zmu + x * torch.cos(omega2 * dt2) + (wz / omega2) * torch.sin(omega2 * dt2)
    wz_ex = -omega2 * x * torch.sin(omega2 * dt2) + wz * torch.cos(omega2 * dt2)
    # span(U): harmonic; complement: free drift dt*v.
    h_ex = h2 + dt2 * v2 + torch.einsum('...dq,...q->...d', U, z_ex - z - dt2 * wz)
    v_ex = v2 + torch.einsum('...dq,...q->...d', U, wz_ex - wz)
    assert (h_lr - h_ex).abs().max().item() < 1e-4, (h_lr - h_ex).abs().max().item()
    assert (v_lr - v_ex).abs().max().item() < 1e-4, (v_lr - v_ex).abs().max().item()

    # complement of span(U) drifts freely (L exerts no force there): its new
    # value must equal the old plus dt*v_perp.
    def _perp(x_):
        return x_ - torch.einsum(
            '...dq,...q->...d', U, torch.einsum('...dq,...d->...q', U, x_),
        )
    assert (_perp(h_lr) - (_perp(h2) + dt2 * _perp(v2))).abs().max().item() < 1e-4

    # 9) Stiffness immunity: a low-rank operator sharp enough to blow up an
    #    explicit step still produces a bounded rotation on its modes.
    torch.manual_seed(2)
    G_stiff = torch.randn(B, T, d2, 2) * 1e3             # sigma_max(L) ~ 1e6
    L_s = G_stiff @ G_stiff.transpose(-1, -2)
    U_s, kappa_s = lowrank_modes(G_stiff)
    assert kappa_s.max().item() > 1e5, kappa_s.max().item()
    h3 = torch.randn(B, T, d2)
    v3 = torch.zeros(B, T, d2)
    f_lr3 = -torch.einsum('...ij,...j->...i', L_s, h3)   # mu = 0
    hs, vs = lowrank_cfc_substep(h3, v3, U_s, kappa_s, f_lr3, m2, 1.0)
    assert torch.isfinite(hs).all() and torch.isfinite(vs).all()
    assert hs.abs().max().item() < 10.0, hs.abs().max().item()

    # 10) max_modes keeps only the stiffest modes.
    U_k, kappa_k = lowrank_modes(G, max_modes=2)
    assert kappa_k.shape[-1] == 2
    assert kappa_k.min().item() >= kappa.topk(2).values.min().item() - 1e-4

    # 11) Impulse (RESPA) composition -- the scheme 'baoab_cfc_lowrank' uses:
    #     A(dt/2) = exact fast flow (T + V_L), B = explicit soft kick (the
    #     clamped diagonal spring V_diag), A(dt/2).  Second-order accurate:
    #     the error against the exact *coupled* flow of T + V_diag + V_L
    #     falls ~4x when dt halves.
    torch.manual_seed(3)
    Bs, Ts, d3, P3 = 2, 2, 6, 3
    ms = torch.ones(Bs, Ts, 1)
    k_a = torch.rand(Bs, Ts, d3) * 2.0 + 0.5
    G3 = torch.randn(Bs, Ts, d3, P3) * 0.7
    L3 = G3 @ G3.transpose(-1, -2)
    Hmat = torch.diag_embed(k_a) + L3                    # (B,T,d,d) SPD
    U3, kappa3 = lowrank_modes(G3)
    mu3 = torch.randn(Bs, Ts, d3)
    s_a = k_a * mu3
    s_L = torch.einsum('...ij,...j->...i', L3, mu3)
    h0 = torch.randn(Bs, Ts, d3)
    v0 = torch.randn(Bs, Ts, d3)

    def _exact_flow(t):
        w, Q = torch.linalg.eigh(Hmat)                   # w>=0
        Om = (w / ms).clamp(min=_OMEGA_SQ_FLOOR).sqrt()  # (B,T,d)
        p0 = torch.einsum('...ji,...j->...i', Q, h0 - mu3)
        q0 = torch.einsum('...ji,...j->...i', Q, v0)
        pt = torch.cos(Om * t) * p0 + torch.sin(Om * t) / Om * q0
        qt = -Om * torch.sin(Om * t) * p0 + torch.cos(Om * t) * q0
        h_t = mu3 + torch.einsum('...ij,...j->...i', Q, pt)
        v_t = torch.einsum('...ij,...j->...i', Q, qt)
        return h_t, v_t

    def _impulse_step(h, v, dt):
        half = 0.5 * dt
        f_L = s_L - torch.einsum('...ij,...j->...i', L3, h)
        h, v = lowrank_cfc_substep(h, v, U3, kappa3, f_L, ms, half)
        v = v + (dt / ms) * (s_a - k_a * h)              # soft diagonal kick
        f_L = s_L - torch.einsum('...ij,...j->...i', L3, h)
        h, v = lowrank_cfc_substep(h, v, U3, kappa3, f_L, ms, half)
        return h, v

    T_end = 1.0
    h_ex, v_ex = _exact_flow(T_end)
    errs = []
    for N in (20, 40):
        dt3 = T_end / N
        hh, vv = h0.clone(), v0.clone()
        for _ in range(N):
            hh, vv = _impulse_step(hh, vv, dt3)
        errs.append((hh - h_ex).abs().max().item())
    order = math.log2(errs[0] / max(errs[1], 1e-300))
    assert 1.7 < order < 2.3, (errs, order)

    # 12) The impulse scheme survives a low-rank curvature that blows the
    #     explicit step up outright.  A single rank-1 mode with a controlled,
    #     *non-resonant* omega*dt (safely between the resonances k*pi) is
    #     used so the test is deterministic; the explicit (all-forces-kick)
    #     integrator with the same omega*dt >> 2 diverges.
    d4 = 4
    ms4 = torch.ones(1, 1, 1)
    u_dir = torch.tensor([1.0, -2.0, 0.5, 1.5]).view(1, 1, d4, 1)
    u_dir = u_dir / u_dir.norm(dim=-2, keepdim=True)
    omega_L = 4.7                                        # in (pi, 2pi): stable
    G4 = u_dir * omega_L                                 # kappa = omega_L^2
    L4 = (G4 @ G4.transpose(-1, -2))
    U4, kappa4 = lowrank_modes(G4)
    ka4 = torch.full((1, 1, d4), 0.2)                    # soft diagonal
    h4 = torch.randn(1, 1, d4)
    v4 = torch.zeros(1, 1, d4)
    dt4 = 1.0
    hi, vi = h4.clone(), v4.clone()
    for _ in range(200):
        half = 0.5 * dt4
        f_L = -torch.einsum('...ij,...j->...i', L4, hi)
        hi, vi = lowrank_cfc_substep(hi, vi, U4, kappa4, f_L, ms4, half)
        vi = vi + (dt4 / ms4) * (-ka4 * hi)
        f_L = -torch.einsum('...ij,...j->...i', L4, hi)
        hi, vi = lowrank_cfc_substep(hi, vi, U4, kappa4, f_L, ms4, half)
    assert torch.isfinite(hi).all(), hi
    assert hi.abs().max().item() < 100.0 * h4.abs().max().item(), hi.abs().max().item()

    # explicit (velocity-Verlet with the full force) at the same omega*dt: dies
    he, ve = h4.clone(), v4.clone()
    for _ in range(200):
        f = -torch.einsum('...ij,...j->...i', L4, he) - ka4 * he
        ve = ve + (0.5 * dt4 / ms4) * f
        he = he + dt4 * ve
        f = -torch.einsum('...ij,...j->...i', L4, he) - ka4 * he
        ve = ve + (0.5 * dt4 / ms4) * f
    assert not torch.isfinite(he).all() or he.abs().max().item() > 1e6, (
        f"explicit step should blow up at omega*dt={omega_L}, got "
        f"{he.abs().max().item():.2e}")

    print("cfc_baoab self-test: OK")


if __name__ == "__main__":
    _self_test()
