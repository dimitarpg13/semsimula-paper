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
functions are evaluated through ``_sinc`` below, using

    sinc(x)  = sin(x)/x, with a Taylor branch under 1e-3
    psi(x)   = (1 - cos x)/x^2 = 2 sin^2(x/2)/x^2
             = 0.5 * sinc(x/2)^2

which is exact to the last fp32 bit and smooth through ``omega -> 0``.
``torch.sinc`` is deliberately NOT used -- see ``_sinc``'s docstring: it is
a jiterator op and needs a working NVRTC at runtime, which a host may not
have.  In that
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

Low-rank exact off-diagonal arm (``baoab_cfc_lowrank``) -- status
------------------------------------------------------------------

``cfc_substep`` above only integrates the *diagonal* part of V_theta's
local curvature exactly; ``harmonic_terms_lowrank`` + :func:`lowrank_modes`
+ :func:`lowrank_cfc_substep` extend this to the anisotropic Gaussian
family's off-diagonal precision ``B_k B_k^T`` as well, via an
impulse/RESPA split (see ``model_parf_multixi._layer_step_langevin``'s
``use_lowrank`` branch).  Mathematically correct and unconditionally
stable, as of 2026-08-30 (commits ``82e84ef``, ``e6445d2`` and the
``harmonic_terms_lowrank`` fix below), after three bugs surfaced getting it
running at the L=8, d=384 OWT scale:

1. **``torch.svd_lowrank`` is not safe under gradient checkpointing.**  Its
   global-RNG projection and internal ``try/except`` fallback let the
   backward recompute take a different branch (and even a different
   *device*, via the CPU last resort) than the forward, tripping
   ``CheckpointError: recomputed values ... have different metadata``.
   Fixed by :func:`_randomised_svd_det`: a local-generator, branch-free,
   GPU-only randomised SVD whose output shape/dtype/device never varies.
2. **NaN modes from a degenerate GPU SVD poisoned the whole gradient.**  A
   fully degenerate token can make cusolver return NaN singular
   vectors/values *without raising*; the old ``U * keep`` masking computed
   ``NaN * 0 = NaN``.  Fixed with ``torch.where``-based masking (picks the
   zero branch regardless of the NaN) plus a final ``nan_to_num`` scrub.
3. **``sqrt(g)`` in ``harmonic_terms_lowrank`` had an infinite backward at
   ``g = 0``** -- the same ``inf * 0`` class of bug as the
   ``_OMEGA_SQ_FLOOR`` comment above, but for the well weight rather than
   the stiffness: any well far enough from ``h`` underflows ``g -> 0``
   (forward-safe), and ``g.sqrt()``'s ``1/(2 sqrt g)`` backward times
   ``dg/de = g = 0`` gave NaN.  Fixed by computing ``sqrt(g) = sqrt(w) *
   exp(-0.25*e)`` directly (identical value, analytic gradient).

**Verdict: correct but not production-feasible at this scale.**  The
batched per-token SVD is the entire extra cost over ``baoab_cfc``.
Measured on the L=8, d=384, B=4 (block=512) OWT run: **~120 s/step** with
all 8 layers exactly integrated (``lowrank_max_modes=16``), or **~50
s/step** restricted to the 2 stiffest layers via ``lowrank_layers`` plus
cheaper randomised-SVD knobs (``lowrank_niter=1``, ``lowrank_oversample=2``)
-- vs. ``baoab_cfc``'s ~10-15 s/step.  Even the cheapest configuration is
~3-4x too slow to finish a 100k-step run in practical time (the 2-layer
figure alone is ~22 days for the last 62.5k steps).  It is also aimed at
the wrong target: the companion note's stiffness-bracket result (see
``semsimula-paper/companion_notes/CfC_BAOAB_Integrator_and_Mitigations.md``,
S:33) already showed ``sigma_max(B_k)^2`` -- the very quantity this arm
integrates exactly -- is only a weak correlate of the observed gradient
spikes, not their driver (``E``, ``P`` and ``depth_code`` lead the hard
triggers instead).  So this arm is **retained in the codebase for
completeness and for future / smaller-scale use** (e.g. smaller ``d``/``L``,
or once the batched-SVD cost is amortised differently), but production
training uses ``integrator='baoab_cfc'``.
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

# Bump this whenever the low-rank / checkpoint behaviour changes.  Print
# ``cfc_baoab.__revision__`` in Colab to confirm the *loaded* module is the one
# you think it is -- a stale import is the usual reason a "fixed" bug persists.
__revision__ = "2026-08-30-randomised-svd-det-checkpoint-safe"

# Robustness / cost knobs for `lowrank_modes` (see its docstring).
#
# The jitter path keeps a single ill-conditioned batch element from dragging
# the *entire* batched SVD onto the CPU LAPACK fallback -- which, called per
# layer per step, is the dominant wall-clock cost of the `baoab_cfc_lowrank`
# arm.  The seed is fixed so both the jitter and the randomised truncated SVD
# are deterministic across replays; both use RNG that is isolated from the
# global stream, which the Langevin thermostat and the Phase-1 spike-replay
# rely on being reproducible.
_LOWRANK_SVD_SEED = 1234567
_LOWRANK_SVD_JITTER = 1e-6          # relative to the batch element's max |G_ij|
_LOWRANK_SVD_MAX_TRIES = 3


# ---------------------------------------------------------------------------
# Special functions (branch-free, exact through the omega -> 0 limit)
# ---------------------------------------------------------------------------
# Below this, sin(x)/x is replaced by its Taylor series. The truncation
# error there is x^4/120 <= 8.3e-15 at the cutoff, ~7 orders below fp32
# epsilon, so the two branches agree to the last representable bit.
_SINC_TAYLOR_EPS = 1e-3


def _sinc(x: torch.Tensor) -> torch.Tensor:
    """sin(x)/x, smooth at x = 0.

    NOT ``torch.sinc``. That op is implemented through PyTorch's
    **jiterator**: it has no prebuilt CUDA kernel and is NVRTC-compiled at
    runtime, so it fails outright on any host whose ``libnvrtc-builtins``
    does not match the build. A Colab image refresh on 2026-09-23 did
    exactly that --

        nvrtc: error: failed to open libnvrtc-builtins.so.13.0

    -- killing Cell 5 on a machine where three previous runs of this same
    code had been fine. The hand-rolled version below uses only ``sin``,
    ``where`` and division, all of which ship as prebuilt kernels, so the
    integrator no longer depends on the runtime's ability to compile CUDA.

    ``safe`` exists for the gradient, not the value: dividing by the raw
    ``x`` would make the unused branch ``0/0 -> nan``, and ``where``'s
    backward propagates that nan even though the forward discarded it.
    Substituting 1.0 inside the masked region keeps both branches finite.
    This is the same failure class as the ``_OMEGA_SQ_FLOOR`` fix that
    ``test_cfc_substep_zero_stiffness_no_nan`` guards.
    """
    small = x.abs() < _SINC_TAYLOR_EPS
    safe = torch.where(small, torch.ones_like(x), x)
    return torch.where(small, 1.0 - x * x / 6.0, torch.sin(safe) / safe)


def _psi(x: torch.Tensor) -> torch.Tensor:
    """(1 - cos x)/x^2 = 0.5 sinc(x/2)^2, smooth at x = 0 (value 1/2)."""
    half = _sinc(0.5 * x)
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


def _svd_stable(Gd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Thin SVD ``G = U diag(S) V^T`` of a detached ``G``, robust to cusolver
    convergence failures without the whole-batch CPU cliff.

    ``torch.linalg.svd`` is a single *batched* call, so on GPU one
    degenerate / repeated-spectrum batch element can make the entire call
    raise (``_LinAlgError ... error code: N``).  The naive remedy -- retry
    the whole batch on the CPU LAPACK driver -- is catastrophic when it runs
    per layer per step, because it serialises thousands of small SVDs on the
    CPU.  Instead we first retry on the GPU with a tiny rank-preserving
    jitter added to ``G``: enough to unstick the Jacobi driver on the
    offending element while leaving the stiff modes (the only ones the
    integrator uses) unchanged to ~jitter relative precision.  The CPU path
    stays only as a genuine last resort, now rarely reached.

    The jitter is drawn from a *private* fixed-seed generator, so the global
    RNG stream is never perturbed.
    """
    try:
        U, S, _ = torch.linalg.svd(Gd, full_matrices=False)
        return U, S
    except RuntimeError:
        pass
    scale = Gd.abs().amax(dim=(-2, -1), keepdim=True).clamp_(min=1e-12)
    gen = torch.Generator(device=Gd.device)
    for _i in range(_LOWRANK_SVD_MAX_TRIES):
        gen.manual_seed(_LOWRANK_SVD_SEED + _i)
        eps = _LOWRANK_SVD_JITTER * (8.0 ** _i)         # escalate if it re-fails
        noise = torch.randn(
            Gd.shape, generator=gen, device=Gd.device, dtype=Gd.dtype,
        )
        try:
            U, S, _ = torch.linalg.svd(Gd + eps * scale * noise,
                                       full_matrices=False)
            return U, S
        except RuntimeError:
            continue
    # Genuine last resort: CPU LAPACK (gesdd/gesvd), steadier than the GPU
    # driver on the truly pathological remainder.
    U_cpu, S_cpu, _ = torch.linalg.svd(Gd.cpu(), full_matrices=False)
    return U_cpu.to(Gd), S_cpu.to(Gd)


def _randomised_svd_det(
    Gd: torch.Tensor, q_over: int, niter: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministic, branch-free, device-stable truncated SVD of a detached
    ``G`` -- the top-``q_over`` left singular vectors / values of ``G``.

    This exists because ``torch.svd_lowrank`` is NOT safe inside a
    gradient-checkpointed region: it draws its projection from the *global*
    RNG and wraps internal linalg in fallbacks, so the backward recompute can
    take a different branch / device than the forward and trip
    ``CheckpointError: recomputed values ... have different metadata`` (shape/
    dtype/device).  ``torch.utils.checkpoint`` only requires the recompute to
    match *metadata*, not values -- so the fix is a single code path that

      * uses a *local* fixed-seed generator (never the global stream, so the
        Langevin thermostat and Phase-1 replay stay reproducible), and
      * has no ``try/except`` and never moves data to another device,

    which guarantees identical output shapes/dtypes/device on every call,
    forward or recompute.  It is the standard randomised range-finder
    (Halko-Martinsson-Tropp): sketch the range of ``G`` with a random
    projection, orthonormalise, refine with ``niter`` subspace iterations,
    then take the SVD of the small projected factor.  Cost ``O(d P q_over)``,
    the whole point of the truncation.
    """
    lead = Gd.shape[:-2]
    d, P = Gd.shape[-2], Gd.shape[-1]
    gen = torch.Generator(device=Gd.device)
    gen.manual_seed(_LOWRANK_SVD_SEED)
    omega = torch.randn(*lead, P, q_over, generator=gen,
                        device=Gd.device, dtype=Gd.dtype)
    q_basis, _ = torch.linalg.qr(Gd @ omega)            # (..., d, q_over)
    g_t = Gd.transpose(-1, -2)
    for _ in range(max(0, niter)):                      # subspace iterations
        q_basis, _ = torch.linalg.qr(Gd @ (g_t @ q_basis))
    small = q_basis.transpose(-1, -2) @ Gd              # (..., q_over, P)
    u_small, s, _ = torch.linalg.svd(small, full_matrices=False)
    u_full = q_basis @ u_small                          # (..., d, q_over)
    return u_full, s


def _gram_eigh(Gd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Left singular vectors / values of ``G`` via the ``P x P`` Gram matrix.

    The nonzero eigenvalues of ``L = G G^T`` (``d x d``) are exactly those of
    ``M = G^T G`` (``P x P``), and if ``M v = lam v`` then ``u = G v /
    sqrt(lam)`` is the matching unit eigenvector of ``L``.  For ``P << d``
    this replaces the ``(d x P)`` SVD -- or the range-finder's three
    ``(d x q)`` QRs -- with a single ``P x P`` symmetric eigensolve.

    **Why this is worth a second driver.**  Measured on an A100 at this
    model's shapes (8,192 tokens, ``d=384``, ``P=32``, ``q_over=20``):

        3 x QR (384x20)          1126.4 ms each  -> 3379 ms   (99.6% of cost)
        small SVD (20x32)          12.0 ms
        ------------------------------------------------------------------
        randomised path total    3391.2 ms
        GRAM+EIGH (32x32)          14.1 ms       -> 241x faster

    At 16 calls per optimiser step, roughly doubled by gradient-checkpoint
    recompute, that is the difference between ~112 s/step and ~4.65 s/step,
    i.e. between an unrunnable arm and an ~11% overhead on ``baoab_cfc``.
    It also returns **all** ``P`` modes, so truncation stops being necessary.

    **The tradeoff, stated because :func:`_svd_stable` deliberately avoids
    it.**  Forming ``G^T G`` squares the condition number.  That degrades the
    *small* eigenvalues -- but this integrator wants the *stiffest* modes,
    and :func:`lowrank_modes` already zeroes anything under ``floor`` as
    inert, so the damaged end of the spectrum is the end already discarded.
    Validate against ``_svd_stable`` on real ``G`` before trusting it on a
    new configuration.

    Branch-free and device-stable (matmul -> svd -> matmul, no try/except,
    no CPU fallback), so unlike ``_svd_stable`` it is safe to recompute
    inside a gradient-checkpointed layer step.

    **2026-09-17.** A first version used ``eigh`` in fp32 and died on real
    ``G`` with ``_LinAlgError: ... ill-conditioned or has too many repeated
    eigenvalues`` at batch element 5121 of the very first step. The three
    hardenings below (symmetrise, float64, deterministic ramp) and the move
    from ``eigh`` to ``svd`` are the response; see the inline notes.
    """
    dt_in, P = Gd.dtype, Gd.shape[-1]
    M = Gd.transpose(-1, -2) @ Gd                       # (..., P, P)

    # -- three deterministic hardenings, all branch-free ----------------
    # (1) Symmetrise. G^T G is symmetric in exact arithmetic but not in fp,
    #     and the asymmetry is what tips a Jacobi driver into non-convergence.
    # (2) float64. Forming the Gram SQUARES the condition number, which is
    #     precisely the cost _svd_stable avoids by never forming it. 32x32
    #     doubles are cheap; fp32 is not enough headroom for that squaring.
    # (3) A deterministic diagonal ramp, ~1e-12 relative, to split exactly
    #     repeated eigenvalues -- the failure cuSOLVER reports as "too many
    #     repeated eigenvalues". Within a degenerate block ANY orthonormal
    #     basis reconstructs the same operator U diag(lam) U^T, so choosing
    #     one deterministically is harmless to the dynamics.
    M = (0.5 * (M + M.transpose(-1, -2))).double()
    scale = M.diagonal(dim1=-2, dim2=-1).amax(-1).clamp(min=1e-300)
    ramp = torch.arange(P, device=M.device, dtype=M.dtype) * (1e-12 / max(P, 1))
    M = M + torch.diag_embed(scale.unsqueeze(-1) * ramp)

    # SVD rather than eigh: for a PSD matrix the two coincide, but the
    # Jacobi SVD driver is markedly steadier than the symmetric-eigen one on
    # repeated spectra -- and it returns DESCENDING values, which is the
    # order every caller here expects.
    V, lam, _ = torch.linalg.svd(M)                     # lam desc, = sigma^2(G)
    lam = lam.clamp(min=0.0)

    # u_i = G v_i / sigma_i.  Near-null directions would blow up, so they are
    # zeroed here; lowrank_modes' own ``floor`` then marks them inert.
    inv_sqrt = torch.where(lam > 0, lam.clamp(min=1e-300).rsqrt(),
                           torch.zeros_like(lam))
    U = (Gd.double() @ V) * inv_sqrt.unsqueeze(-2)      # (..., d, P)
    return U.to(dt_in), lam.sqrt().to(dt_in)


def lowrank_modes(
    G: torch.Tensor,
    max_modes: Optional[int] = None,
    floor: float = 1e-10,
    *,
    niter: int = 2,
    oversample: int = 4,
    driver: str = "svd",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eigenmodes of the PSD operator ``L = G G^T`` from its factor ``G``.

    Parameters
    ----------
    G : (..., d, P)
        Aggregate low-rank factor; ``L = G G^T`` has rank <= P.
    max_modes : int or None
        Keep only the ``max_modes`` stiffest (largest-eigenvalue) modes.
        ``None`` keeps all ``P`` of them (exact full-SVD path).
    floor : float
        Eigenvalues at or below this are treated as zero: the matching mode
        direction is zeroed (made inert) and its curvature set to 0, which
        both avoids the ``1/sqrt(lambda)`` blow-up when normalising a
        vanishing mode and matches the true dynamics (``L`` exerts no force
        along a null direction).
    niter, oversample : int
        Randomised-SVD controls, used only on the truncated path: the
        projection uses ``max_modes + oversample`` probes and ``niter``
        subspace iterations.  Ignored when ``max_modes`` is ``None``.

    Returns
    -------
    U : (..., d, q)
        Mode directions as orthonormal columns (zero columns are inert),
        ``q = min(P, max_modes)``.  Detached from the graph.
    kappa : (..., q)
        Per-mode stiffness (eigenvalues of ``L``, ``>= 0``).  Detached.

    Notes
    -----
    Full path (``max_modes is None``): computed from the SVD
    ``G = U_full diag(S) V^T`` -- the left singular vectors are the
    eigenvectors of ``L = G G^T`` and ``S**2`` are its eigenvalues.  This
    avoids forming the Gram ``G^T G`` (which squares the condition number
    and can make eigh's divide-and-conquer driver fail to converge on
    degenerate spectra), and uses :func:`_svd_stable` so a single bad batch
    element does not force the whole batch onto the CPU.

    Truncated path (``max_modes = q < P``): the full ``P``-mode SVD is
    wasteful when only the ``q`` *stiffest* modes threaten the
    ``omega dt < 2`` stability wall and the rest are handled by the caller's
    explicit kick.  A randomised truncated SVD (:func:`_randomised_svd_det`
    with ``q + oversample`` probes) returns just those modes at
    ``O(d P q)`` rather than ``O(d P^2)`` cost, with a proportionally
    smaller memory footprint.  It is deterministic, branch-free and
    device-stable (a *local* fixed-seed generator, no ``try/except``, no CPU
    fallback), so its output metadata is identical on the forward and on the
    gradient-checkpoint backward recompute -- ``torch.svd_lowrank`` is not,
    and trips ``CheckpointError`` when used here.

    IMPORTANT: with truncation the caller MUST demote the dropped modes to
    its explicit kick -- subtract only ``P_U f_L`` (the retained-mode
    projection of the low-rank force), not the full ``f_L`` -- otherwise the
    soft modes' restoring force is silently cancelled out of the dynamics.
    See ``model_parf_multixi._layer_step_langevin``'s ``use_lowrank`` kick.
    """
    Gd = G.detach()
    P = Gd.shape[-1]

    if driver == "gram":
        # One P x P eigensolve instead of a (d x P) SVD or three (d x q) QRs.
        # Returns all P modes, so max_modes only trims afterwards.
        q_req = P if max_modes is None else min(int(max_modes), P)
        U_full, S = _gram_eigh(Gd)
    elif max_modes is not None and int(max_modes) < P:
        q_req = int(max_modes)
        q_over = min(q_req + max(0, oversample), Gd.shape[-2], Gd.shape[-1])
        # Deterministic, branch-free, device-stable randomised SVD -- safe to
        # recompute inside a gradient-checkpointed layer step (torch.svd_lowrank
        # is not; see _randomised_svd_det).
        U_full, S = _randomised_svd_det(Gd, q_over, niter)
    else:
        q_req = P if max_modes is None else min(int(max_modes), P)
        U_full, S = _svd_stable(Gd)

    # Both drivers return singular values in descending order, so the leading
    # columns / values are already the stiffest modes.
    q = min(q_req, S.shape[-1])
    lam = S[..., :q] ** 2                               # (..., q) largest q (desc)
    U = U_full[..., :, :q]                              # (..., d, q), unit cols

    # A fully degenerate token can make the GPU SVD/QR emit NaN singular
    # values / vectors *without* raising.  ``lam > floor`` is then False, so
    # the mode is meant to be dropped -- but ``U * keep`` would compute
    # ``NaN * 0 = NaN`` and inject it into the differentiable substep,
    # NaN-poisoning the whole gradient (seen only on CUDA; LAPACK stays
    # clean).  Use ``torch.where`` (picks the zero branch regardless of the
    # NaN) and a final scrub for any surviving NaN in a *kept* column.
    keep = torch.isfinite(lam) & (lam > floor)         # (..., q) bool
    zeros_U = torch.zeros_like(U)
    U = torch.where(keep.unsqueeze(-2), U, zeros_U)     # zero inert directions
    U = torch.nan_to_num(U, nan=0.0, posinf=0.0, neginf=0.0)
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
# SR2: exact DAMPED flow of the low-rank stiff modes (book Prop 45)
# ---------------------------------------------------------------------------
#
# The split A(dt/2) O(dt) A(dt/2) rotates each stiff mode, applies friction,
# and rotates again; the friction is therefore sampled at the phases the
# rotation happens to reach, and the energy a mode dissipates depends on the
# step count (book Prop 44, the theta/sin theta term). The friction and the
# low-rank rotation are diagonal in the same basis, so each stiff mode can
# instead be integrated as ONE forced damped oscillator,
#
#     x'' + gamma x' + omega0^2 x = a,   x = z - z0,  omega0^2 = kappa/m,
#     a = (U^T f_lr)/m (the frozen affine mode force at the start),
#
# whose flow is a linear autonomous map: e^{sM} e^{tM} = e^{(s+t)M}, so the
# linear stiff dynamics no longer depend on how T is cut (book Prop 45). The
# caller removes the O-step's action on span(U) so friction is not applied
# twice there; the complement keeps its free drift and its O-step.

def _sinhc(x: torch.Tensor) -> torch.Tensor:
    """sinh(x)/x, smooth at x = 0 (nan-safe in the unused branch)."""
    small = x.abs() < 1e-4
    safe = torch.where(small, torch.ones_like(x), x)
    return torch.where(small, 1.0 + x * x / 6.0, torch.sinh(safe) / safe)


def damped_mode_coefficients(
    omega0_sq: torch.Tensor, gamma: torch.Tensor | float, t: float,
    n_terms: int = 40,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(E12, E22, P, Q) of the exact flow of x'' + g x' + w0^2 x = a over t.

    With x(0) = 0, x'(0) = w0:  x(t) = E12 w0 + Q a,  x'(t) = E22 w0 + P a.
    (E11 is not needed because x(0) = 0 in the caller's coordinates.)
    Evaluated in float64 for every damping regime: underdamped (cos/sin),
    overdamped (cosh/sinh), critical and omega0 = 0, without dividing by a
    vanishing omega0^2 (series branch) and nan-free in every unused branch.
    """
    w2 = omega0_sq.double()
    g = torch.as_tensor(gamma, dtype=torch.float64, device=w2.device)
    wd2 = w2 - 0.25 * g * g                                    # omega_d^2, either sign
    r = wd2.abs().sqrt()
    under = wd2 >= 0
    C = torch.where(under, torch.cos(r * t), torch.cosh(torch.where(under, torch.zeros_like(r), r) * t))
    S = t * torch.where(under, _sinc(r * t), _sinhc(torch.where(under, torch.zeros_like(r), r) * t))
    e = torch.exp(-0.5 * g * t)
    E11 = e * (C + 0.5 * g * S)
    E12 = e * S
    E22 = e * (C - 0.5 * g * S)
    P = E12                                                    # velocity response to unit forcing
    # Position response Q = (1 - E11)/omega0^2. Where omega0^2 t^2 is tiny the
    # division cancels catastrophically, so Q is summed from the exact Taylor
    # recurrence of x'' = 1 - g x' - w0^2 x, x(0) = x'(0) = 0, in the terms
    # T_n = c_n t^n:  T_{n+2} = -(g t (n+1) T_{n+1} + w0^2 t^2 T_n)/((n+2)(n+1)),
    # T_2 = t^2/2 (40 terms: converged to round-off for g t < 4). For g t >= 4
    # with tiny omega0^2, the omega0 = 0 damped drift (relative error ~ w0^2 t/2g)
    # or the division (~ eps g/(w0^2 t)), whichever is smaller (both <= 1e-8).
    # |w2| and w2 != 0 (2026-10-10, PMX): negative omega0^2 (an unstable,
    # hyperbolic mode) takes the division branch like a positive one. For
    # w2 >= 0 both are the same expressions as before, bit for bit.
    small = (w2.abs() * t * t) < 1e-4
    safe_w2 = torch.where(w2 != 0, w2, torch.ones_like(w2))
    Q_div = (1.0 - E11) / safe_w2
    gt = g * t
    Tm, Tn = torch.zeros_like(w2), torch.full_like(w2, 0.5 * t * t)     # T_1, T_2
    Q_ser = Tn
    for k in range(2, 2 + n_terms):                            # T_3 .. T_41 at the default 40
        Tm, Tn = Tn, -(gt * k * Tn + w2 * t * t * Tm) / ((k + 1) * k)
        Q_ser = Q_ser + Tn
    g_safe = torch.where(gt > 0, g, torch.ones_like(g))
    Q_zero = (gt - 1.0 + torch.exp(-gt)) / (g_safe * g_safe)
    Q_large = torch.where(w2.abs() * t < 1e-8 * g_safe, Q_zero.expand_as(Q_div), Q_div)
    Q = torch.where(small, torch.where(gt < 4.0, Q_ser, Q_large), Q_div)
    return E12, E22, P, Q


def lowrank_damped_substep(
    h: torch.Tensor,
    v: torch.Tensor,
    U: torch.Tensor,
    kappa: torch.Tensor,
    f_lr: torch.Tensor,
    m: torch.Tensor,
    gamma: torch.Tensor | float,
    dt: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``lowrank_cfc_substep`` with the friction integrated exactly on span(U).

    Same arguments plus ``gamma`` (a scalar, constant damping). On span(U)
    each mode follows the exact forced damped oscillator over ``dt``; on the
    complement the motion is the same free drift ``h += dt v``. With
    ``gamma = 0`` this equals ``lowrank_cfc_substep`` (up to its omega^2
    floor). The caller must NOT also apply the O-step to span(U).
    """
    z = torch.einsum('...dq,...d->...q', U, h)
    wz = torch.einsum('...dq,...d->...q', U, v)
    fz = torch.einsum('...dq,...d->...q', U, f_lr)
    mq = m                                                    # (..., 1) broadcasts over the q modes
    w0sq = kappa / mq
    E12, E22, P, Q = damped_mode_coefficients(w0sq, gamma, dt)
    E12, E22, P, Q = (c.to(h.dtype) for c in (E12, E22, P, Q))
    a = fz / mq
    z_new = z + E12 * wz + Q * a
    wz_new = E22 * wz + P * a
    h_new = h + dt * v + torch.einsum('...dq,...q->...d', U, z_new - z - dt * wz)
    v_new = v + torch.einsum('...dq,...q->...d', U, wz_new - wz)
    return h_new, v_new


# ---------------------------------------------------------------------------
# PMX (protocol SS5.15, 2026-10-10): the Poisson-mode wells integrated exactly
# ---------------------------------------------------------------------------
def indefinite_lowrank_modes(
    B: torch.Tensor, signs: torch.Tensor, floor: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eigenmodes of the symmetric, possibly INDEFINITE ``L = B diag(s) B^T``.

    ``B`` is (..., d, n), ``signs`` (..., n). Returns orthonormal mode columns
    ``U`` (..., d, n) spanning range(B) (zero columns where B is rank-deficient,
    inert) and eigenvalues ``kappa`` (..., n) of either sign, both detached,
    as :func:`lowrank_modes` returns for the PSD case. ``L`` acts as zero on
    the complement of range(B).

    Route: an orthonormal basis Q of range(B) from :func:`_gram_eigh` (the
    hardened Gram path), the n x n restriction M = (Q^T B) diag(s) (Q^T B)^T,
    shifted by c >= |lambda_min| to be PSD so that the same Jacobi SVD can
    diagonalise it (it returns |lambda| otherwise), then shifted back. Inert
    basis directions are decoupled by a large distinct diagonal value, so
    their eigenvectors are unit vectors and their mode columns exactly zero.
    """
    Bd, sd = B.detach(), signs.detach()
    dt_in, n = Bd.dtype, Bd.shape[-1]
    Q, sv = _gram_eigh(Bd)                                   # (..., d, n), (..., n) desc
    inert = ~(torch.isfinite(sv) & (sv * sv > floor))
    Q = torch.where(inert.unsqueeze(-2), torch.zeros_like(Q), Q)
    Q = torch.nan_to_num(Q, nan=0.0, posinf=0.0, neginf=0.0).double()
    Pm = Q.transpose(-1, -2) @ Bd.double()                   # (..., n, n)
    M = (Pm * sd.double().unsqueeze(-2)) @ Pm.transpose(-1, -2)
    M = 0.5 * (M + M.transpose(-1, -2))
    c = M.abs().sum(-1).amax(-1, keepdim=True).clamp(min=1e-300)   # >= spectral radius
    big = 10.0 * c + 1.0
    eye = torch.eye(n, dtype=M.dtype, device=M.device)
    Ms = M + c.unsqueeze(-1) * eye + torch.diag_embed(inert.double() * big)
    ramp = torch.arange(n, device=M.device, dtype=M.dtype) * (1e-12 / max(n, 1))
    Ms = Ms + torch.diag_embed(c * ramp)
    Y, lam, _ = torch.linalg.svd(Ms)                         # PSD: SVD = eigen, desc
    kappa = lam - c - (Y * Y * (inert.double() * big).unsqueeze(-1)).sum(-2)
    U = Q @ Y                                                # (..., d, n)
    dead = U.norm(dim=-2) < 0.5                              # an inert direction's column
    U = torch.where(dead.unsqueeze(-2), torch.zeros_like(U), U)
    kappa = torch.where(dead, torch.zeros_like(kappa), kappa)
    return U.to(dt_in), kappa.to(dt_in)


def indefinite_lowrank_modes_split(
    U0: torch.Tensor, kappa0: torch.Tensor, R: torch.Tensor, dW: torch.Tensor,
    floor: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """:func:`indefinite_lowrank_modes` for ``L = U0 diag(kappa0) U0^T - R diag(dW) R^T``.

    The same operator and span as ``B = [U0 sqrt(kappa0), R]`` with signs
    ``[+1, -dW]``, but ``U0`` (..., d, q) is already orthonormal (or zero
    columns), as :func:`lowrank_modes` returns it, so the basis only needs the
    part of ``R`` (..., d, M) outside span(U0): a M x M Gram solve instead of a
    (q + M) x (q + M) one (PMX, 2026-10-10; about 8x less basis work at
    q = M = 16). The restriction is then diagonalised exactly as in
    :func:`indefinite_lowrank_modes`. Returns ``U`` (..., d, q + M) and
    ``kappa`` (..., q + M), detached.
    """
    U0d, k0, Rd, sd = U0.detach(), kappa0.detach(), R.detach(), dW.detach()
    dt_in = Rd.dtype
    q, M = U0d.shape[-1], Rd.shape[-1]
    Rp = Rd - U0d @ (U0d.transpose(-1, -2) @ Rd)               # R outside span(U0)
    Qr, sv = _gram_eigh(Rp)                                    # (..., d, M)
    inert_r = ~(torch.isfinite(sv) & (sv * sv > floor))
    Qr = torch.where(inert_r.unsqueeze(-2), torch.zeros_like(Qr), Qr)
    Qr = torch.nan_to_num(Qr, nan=0.0, posinf=0.0, neginf=0.0)
    inert = torch.cat([U0d.norm(dim=-2) < 0.5, inert_r], dim=-1)    # (..., q + M)
    Q = torch.cat([U0d, Qr], dim=-1).double()                  # (..., d, q + M)
    Bd = torch.cat([U0d * k0.clamp(min=0).sqrt().unsqueeze(-2), Rd], dim=-1).double()
    sg = torch.cat([torch.ones_like(k0), -sd], dim=-1).double()
    n = q + M
    Pm = Q.transpose(-1, -2) @ Bd                              # (..., n, n)
    Mm = (Pm * sg.unsqueeze(-2)) @ Pm.transpose(-1, -2)
    Mm = 0.5 * (Mm + Mm.transpose(-1, -2))
    c = Mm.abs().sum(-1).amax(-1, keepdim=True).clamp(min=1e-300)
    big = 10.0 * c + 1.0
    eye = torch.eye(n, dtype=Mm.dtype, device=Mm.device)
    Ms = Mm + c.unsqueeze(-1) * eye + torch.diag_embed(inert.double() * big)
    ramp = torch.arange(n, device=Mm.device, dtype=Mm.dtype) * (1e-12 / max(n, 1))
    Ms = Ms + torch.diag_embed(c * ramp)
    Y, lam, _ = torch.linalg.svd(Ms)
    kappa = lam - c - (Y * Y * (inert.double() * big).unsqueeze(-1)).sum(-2)
    U = Q @ Y
    dead = U.norm(dim=-2) < 0.5
    U = torch.where(dead.unsqueeze(-2), torch.zeros_like(U), U)
    kappa = torch.where(dead, torch.zeros_like(kappa), kappa)
    return U.to(dt_in), kappa.to(dt_in)


def series_terms_for(gamma_t: float, tol: float = 1e-18) -> int:
    """Taylor terms :func:`damped_mode_coefficients` needs at gamma * t (host float).

    Its series branch is used only where |omega0^2| t^2 < 1e-4, so the terms
    fall at least as fast as (gamma t + 0.01)^n / n!. Returns the smallest
    count, at least 8 and at most the default 40, that brings a term below
    ``tol`` (relative to T_2): 12 at PMX's gamma t = 0.2. Identical to the
    40-term sum to float64 rounding (PMX, 2026-10-11).
    """
    x, term, n = float(gamma_t) + 0.01, 1.0, 0
    while n < 40:
        n += 1
        term *= x / (n + 2)
        if n >= 8 and term < tol:
            break
    return n


def _cholqr_basis(Rp: torch.Tensor, floor: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Orthonormal columns spanning Rp's (float64), by Cholesky QR in two passes.

    Columns with |r|^2 <= floor are inert: their Gram rows are decoupled to the
    identity and their output columns zeroed. Two passes make the result
    orthonormal to float64 rounding for condition numbers far beyond what PMX
    meets (at most 2.1 on PM1's states, debug/verify_pmx_fast.py). One batched
    Cholesky and one triangular solve per pass replace a Gram SVD, which on the
    GPU costs a fixed 32 x 16 Jacobi tile whatever the width.
    """
    Rd = Rp.double()
    M = Rd.shape[-1]
    G = Rd.transpose(-1, -2) @ Rd
    inert = G.diagonal(dim1=-2, dim2=-1) <= floor                  # (..., M)
    eye = torch.eye(M, dtype=G.dtype, device=G.device)
    pair = inert.unsqueeze(-1) | inert.unsqueeze(-2)
    G = torch.where(pair, eye.expand_as(G), G)
    Q = Rd
    for _ in range(2):
        L = torch.linalg.cholesky(G)
        Q = torch.linalg.solve_triangular(L, Q.transpose(-1, -2), upper=False).transpose(-1, -2)
        Q = torch.where(inert.unsqueeze(-2), torch.zeros_like(Q), Q)
        G = Q.transpose(-1, -2) @ Q
        G = torch.where(pair, eye.expand_as(G), G)
    return Q, inert


def indefinite_lowrank_modes_chol(
    U0: torch.Tensor, kappa0: torch.Tensor, R: torch.Tensor, dW: torch.Tensor,
    floor: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """:func:`indefinite_lowrank_modes_split` with the well basis by Cholesky QR.

    Same operator, same span, same restriction and shifted-SVD diagonalisation;
    only the orthonormal basis of R's part outside span(U0) is built by two
    Cholesky passes instead of a Gram SVD (PMX, 2026-10-11). Removes one of
    the three batched SVDs per PMX layer step.
    """
    U0d, k0, Rd, sd = U0.detach(), kappa0.detach(), R.detach(), dW.detach()
    dt_in = Rd.dtype
    q, M = U0d.shape[-1], Rd.shape[-1]
    Rp = Rd - U0d @ (U0d.transpose(-1, -2) @ Rd)
    Qr, inert_r = _cholqr_basis(Rp, floor)
    inert = torch.cat([U0d.norm(dim=-2) < 0.5, inert_r], dim=-1)
    Q = torch.cat([U0d.double(), Qr], dim=-1)
    Bd = torch.cat([U0d * k0.clamp(min=0).sqrt().unsqueeze(-2), Rd], dim=-1).double()
    sg = torch.cat([torch.ones_like(k0), -sd], dim=-1).double()
    n = q + M
    Pm = Q.transpose(-1, -2) @ Bd
    Mm = (Pm * sg.unsqueeze(-2)) @ Pm.transpose(-1, -2)
    Mm = 0.5 * (Mm + Mm.transpose(-1, -2))
    c = Mm.abs().sum(-1).amax(-1, keepdim=True).clamp(min=1e-300)
    big = 10.0 * c + 1.0
    eye = torch.eye(n, dtype=Mm.dtype, device=Mm.device)
    Ms = Mm + c.unsqueeze(-1) * eye + torch.diag_embed(inert.double() * big)
    ramp = torch.arange(n, device=Mm.device, dtype=Mm.dtype) * (1e-12 / max(n, 1))
    Ms = Ms + torch.diag_embed(c * ramp)
    Y, lam, _ = torch.linalg.svd(Ms)
    kappa = lam - c - (Y * Y * (inert.double() * big).unsqueeze(-1)).sum(-2)
    U = Q @ Y
    dead = U.norm(dim=-2) < 0.5
    U = torch.where(dead.unsqueeze(-2), torch.zeros_like(U), U)
    kappa = torch.where(dead, torch.zeros_like(kappa), kappa)
    return U.to(dt_in), kappa.to(dt_in)


def lowrank_iso_damped_substep(
    h: torch.Tensor,
    v: torch.Tensor,
    U: torch.Tensor,
    kappa: torch.Tensor,
    alpha: torch.Tensor,
    f: torch.Tensor,
    m: torch.Tensor,
    gamma: torch.Tensor | float,
    dt: float,
    n_terms: int = 40,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact flow over ``dt`` of  m x'' = f - K (x - h) - gamma m x',  x(0) = h, x'(0) = v,

    with the frozen spring ``K = alpha I + U diag(kappa) U^T`` on ALL of R^d:
    modes in span(U) have stiffness ``kappa + alpha`` (either sign), the
    complement has ``alpha`` (per token, either sign), and the damping is
    integrated exactly everywhere. ``f`` is the force at ``h`` (the start of
    the substep), as in :func:`lowrank_damped_substep`. The caller must apply
    no O-step at all: friction is inside this flow on the whole space.
    ``alpha`` is (..., 1); ``U`` has orthonormal or zero columns.
    """
    vz = torch.einsum('...dq,...d->...q', U, v)
    fz = torch.einsum('...dq,...d->...q', U, f)
    vc = v - torch.einsum('...dq,...q->...d', U, vz)
    fc = f - torch.einsum('...dq,...q->...d', U, fz)
    E12s, E22s, Ps, Qs = (c.to(h.dtype) for c in damped_mode_coefficients((kappa + alpha) / m, gamma, dt, n_terms))
    E12c, E22c, Pc, Qc = (c.to(h.dtype) for c in damped_mode_coefficients(alpha / m, gamma, dt, n_terms))
    dz = E12s * vz + Qs * fz / m
    vz_new = E22s * vz + Ps * fz / m
    dc = E12c * vc + Qc * fc / m
    vc_new = E22c * vc + Pc * fc / m
    h_new = h + torch.einsum('...dq,...q->...d', U, dz) + dc
    v_new = torch.einsum('...dq,...q->...d', U, vz_new) + vc_new
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

    # 13) Truncated modes (randomised svd_lowrank path) recover the stiffest q
    #     modes of the full SVD, are deterministic across calls, and leave the
    #     global RNG stream untouched -- the last two are what make the
    #     baoab_cfc_lowrank arm reproducible for Phase-1 spike-replay.
    torch.manual_seed(7)
    Bt, Tt, dt_d, Pt = 2, 3, 12, 8
    # Planted spectrum with a clean gap: 3 stiff modes, 5 soft ones.
    Ug, _ = torch.linalg.qr(torch.randn(Bt, Tt, dt_d, Pt))       # orthonormal cols
    svals = torch.tensor([10.0, 8.0, 6.0, 0.3, 0.2, 0.1, 0.05, 0.02])
    Gt = Ug * svals                                              # (B,T,d,P)
    _, kap_full = lowrank_modes(Gt)                              # exact full SVD
    U_full3, _ = lowrank_modes(Gt)
    q3 = 3
    U_tr, kap_tr = lowrank_modes(Gt, max_modes=q3)
    assert kap_tr.shape[-1] == q3, kap_tr.shape
    # stiffest-3 curvatures match the full decomposition
    assert (kap_tr - kap_full[..., :q3]).abs().max().item() < 1e-2, (
        kap_tr[0, 0], kap_full[0, 0, :q3])
    # retained subspace agrees with the full top-3 (clean gap -> no rotation
    # ambiguity): each truncated column aligns with one full column.
    olap = torch.einsum('...dq,...dr->...qr', U_tr, U_full3[..., :q3]).abs()
    assert (olap.amax(dim=-1) > 0.99).all(), olap[0, 0]
    # determinism: a second call is bit-identical (fixed-seed forked RNG)
    U_tr2, kap_tr2 = lowrank_modes(Gt, max_modes=q3)
    assert torch.equal(U_tr, U_tr2) and torch.equal(kap_tr, kap_tr2)
    # global RNG untouched: the next global draw is exactly what it would have
    # been had lowrank_modes never run (fork_rng must fully restore state).
    torch.manual_seed(99)
    ref = torch.randn(5)
    torch.manual_seed(99)
    _ = lowrank_modes(Gt, max_modes=q3)
    got = torch.randn(5)
    assert torch.equal(ref, got), (ref, got)

    # 14) Truncated impulse step -- the RESPA scheme baoab_cfc_lowrank runs WITH
    #     lowrank_max_modes set: retain only the stiff modes in the exact fast
    #     flow and demote the rest to the explicit kick via the retained-mode
    #     projection P_U f_L (exactly as model_parf_multixi's use_lowrank kick
    #     now does).  This must stay 2nd-order accurate against the exact
    #     coupled flow.  Subtracting the *full* f_L instead of P_U f_L would
    #     cancel the dropped modes' restoring force and break this.
    torch.manual_seed(4)
    Bq, Tq, dq, Pq = 2, 2, 6, 5
    msq = torch.ones(Bq, Tq, 1)
    Uq0, _ = torch.linalg.qr(torch.randn(Bq, Tq, dq, Pq))        # orthonormal cols
    sv = torch.tensor([3.0, 2.5, 0.4, 0.3, 0.2])                 # 2 stiff, 3 soft
    Gq = Uq0 * sv                                                # (B,T,d,P)
    Lq = Gq @ Gq.transpose(-1, -2)
    ka_q = torch.rand(Bq, Tq, dq) * 0.5 + 0.2                    # soft diagonal
    Hq = torch.diag_embed(ka_q) + Lq                            # SPD
    muq = torch.randn(Bq, Tq, dq)
    s_Lq = torch.einsum('...ij,...j->...i', Lq, muq)
    s_aq = ka_q * muq
    h0q = torch.randn(Bq, Tq, dq)
    v0q = torch.randn(Bq, Tq, dq)
    Uq, kapq = lowrank_modes(Gq, max_modes=2)                    # keep 2 stiff modes

    def _exact_flow_q(t):
        w, Q = torch.linalg.eigh(Hq)
        Om = (w / msq).clamp(min=_OMEGA_SQ_FLOOR).sqrt()
        p0 = torch.einsum('...ji,...j->...i', Q, h0q - muq)
        q0 = torch.einsum('...ji,...j->...i', Q, v0q)
        pt = torch.cos(Om * t) * p0 + torch.sin(Om * t) / Om * q0
        return muq + torch.einsum('...ij,...j->...i', Q, pt)

    def _impulse_trunc(h, v, dt):
        half = 0.5 * dt
        f_L = s_Lq - torch.einsum('...ij,...j->...i', Lq, h)
        h, v = lowrank_cfc_substep(h, v, Uq, kapq, f_L, msq, half)
        # kick: total force minus the retained-mode projection of f_L, so the
        # soft (dropped) modes' force stays in the explicit kick.
        f_L = s_Lq - torch.einsum('...ij,...j->...i', Lq, h)
        PUf = torch.einsum('...dq,...q->...d', Uq,
                           torch.einsum('...dq,...d->...q', Uq, f_L))
        f_tot = (s_aq - ka_q * h) + f_L
        v = v + (dt / msq) * (f_tot - PUf)
        f_L = s_Lq - torch.einsum('...ij,...j->...i', Lq, h)
        h, v = lowrank_cfc_substep(h, v, Uq, kapq, f_L, msq, half)
        return h, v

    h_exq = _exact_flow_q(1.0)
    errs_q = []
    for N in (20, 40):
        dtq = 1.0 / N
        hh, vv = h0q.clone(), v0q.clone()
        for _ in range(N):
            hh, vv = _impulse_trunc(hh, vv, dtq)
        errs_q.append((hh - h_exq).abs().max().item())
    order_q = math.log2(errs_q[0] / max(errs_q[1], 1e-300))
    assert 1.7 < order_q < 2.3, (errs_q, order_q)

    print("cfc_baoab self-test: OK")


if __name__ == "__main__":
    _self_test()
