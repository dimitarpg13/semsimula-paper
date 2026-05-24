"""SP-HSPLM (Q9(e)) — pair-skew + conservative-pair Helmholtz model.

Pre-registered protocol
-----------------------
docs/SP_HSPLM_Stage_2_pre-registered_protocol.md (Stage 2)

This module implements the attention-free Helmholtz hybrid of the v3
architecture doc section 9.2. The forward at layer ell dispatches on
the schedule sigma to either:

  - S-block: SparsePARFLM force (conservative pair scalar V_phi over
             top-k routed sources, plus per-token V_theta gradient).
             Bit-identical to SparsePARFLM._layer_step.

  - C-block: pair-skew force using a learned low-rank skew kernel
             J_phi = U V^T - V U^T over the same Gumbel top-k mask.
             Optionally augmented with a per-token gyroscopic kernel
             Omega(h_t) delta_t.

Both block types share:
  - the SPLM damped-Verlet integrator shell,
  - the learnable per-token mass m_t,
  - the learnable global damping gamma (floored at gamma_min),
  - the SparsePARFLM score head and Gumbel top-k mask
    (recomputed per layer; both branches consume it when active).

Causal-leak invariant (mandatory; see docs/Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md)
---------------------------------------------------------------------------------------
Three new .detach() points relative to dense PARFLM and SparsePARFLM:

  1. Source-side hidden state h_s entering the score head -- inherited
     from SparsePARFLM unchanged.
  2. Source-side velocity proxy delta_s = h_s - h_prev_s entering the
     C-block kernel multiplication J_phi (delta_s - delta_t). The
     entire delta tensor is .detach()-ed before the masked sum over
     sources; only the query-side delta_t (the diagonal-position part
     of the (delta_s - delta_t) expression) carries gradient back to
     h_t. This is enforced inside _c_block_step.
  3. xi-pool detach inside any S-block step -- inherited from
     SparsePARFLM (the xi = causal_cumulative_mean(h.detach()) line of
     the leak-fix patch).

The standalone causal probe
(notebooks/conservative_arch/causal_probe.py) is the runtime invariant;
the structural argument above is what makes the invariant hold.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

_THIS_DIR = Path(__file__).parent
_PARENT_DIR = _THIS_DIR.parent
_PARF_DIR = _PARENT_DIR / "parf"
sys.path.insert(0, str(_PARENT_DIR))
sys.path.insert(0, str(_PARF_DIR))

from model_parf import (  # type: ignore  # noqa: E402
    MLPVPhi,
    StructuralCompetitiveVPhi,
    StructuralVPhi,
    causal_cumulative_mean,
)
from model_parf_sparse import (  # type: ignore  # noqa: E402
    ScoreHead,
    SparsePARFConfig,
    SparsePARFLM,
)

_SARF_DIR = _PARENT_DIR / "sarf_mass_variant"
sys.path.insert(0, str(_SARF_DIR))
from model_sarf_mass import ScalarPotential  # type: ignore  # noqa: E402


# ---------------------------------------------------------------------------
# Mechanism-1 module factories (per-layer-indexed force-law support)
# ---------------------------------------------------------------------------
def _build_v_phi(cfg: SparsePARFConfig) -> nn.Module:
    """Mirror of `PARFLM.__init__`'s V_phi dispatch; returns one fresh module.

    Used when SPHSPLMConfig.share_v_phi_across_layers is False to
    populate `nn.ModuleList([_build_v_phi(cfg) for _ in range(L_S)])`.
    The dispatch must stay bit-identical to the parent class's
    construction so the shared-flag and per-layer-flag cells differ
    only in identity (one instance vs. many), not in parametric form.
    """
    if cfg.v_phi_kind == "structural":
        return StructuralVPhi(cfg)
    if cfg.v_phi_kind == "structural_competitive":
        return StructuralCompetitiveVPhi(cfg)
    if cfg.v_phi_kind == "mlp":
        return MLPVPhi(cfg)
    raise ValueError(
        f"unknown v_phi_kind={cfg.v_phi_kind!r}; "
        "expected 'structural', 'structural_competitive', or 'mlp'."
    )


def _build_score_head(cfg: SparsePARFConfig) -> nn.Module:
    """Build one fresh SparsePARFLM-style ScoreHead.

    Used when SPHSPLMConfig.share_score_head_across_layers is False to
    populate `nn.ModuleList([_build_score_head(cfg) for _ in range(L)])`.
    """
    return ScoreHead(cfg)


def _build_v_theta(cfg: SparsePARFConfig) -> nn.Module:
    """Mirror of `PARFLM.__init__`'s V_theta construction; one fresh module.

    Used when SPHSPLMConfig.share_v_theta_across_layers is False to
    populate `nn.ModuleList([_build_v_theta(cfg) for _ in range(L_S)])`.
    V_theta is the **dominant** force in SP-HSPLM (the scalar
    Hopfield-style potential consumed at every S-block layer); per
    paper Appendix A Eq. A.130, Class F prescribes a per-layer
    parametric family theta_ell for the scalar potential.  Lifting
    V_theta to per-layer is the largest single Mechanism-1 move
    available -- at v_hidden=1024, v_depth=3 each V_theta copy is
    ~2.6M params, so L_S copies dominate the model parameter budget.
    Exercised by cell q9e_n (full-Class-F test).
    """
    return ScalarPotential(cfg.d, cfg.v_hidden, cfg.v_depth)


# ---------------------------------------------------------------------------
# Schedule registry (S/C tokens, distinct from helmholtz S/A registry)
# ---------------------------------------------------------------------------
def make_schedule_sc(
    name: str, L: int = 8, k: int = 1, LC: int = 1
) -> str:
    """Construct a length-L schedule string of S/C tokens.

    Parallels notebooks/conservative_arch/helmholtz/model_helmholtz.py
    `make_schedule`, but with the non-conservative block type renamed
    from 'A' (attention) to 'C' (circulation, the SP-HSPLM C-block).

    Parameters
    ----------
    name : str
        One of {"all_s", "all_c", "sandwich", "inverse_sandwich",
                "interleaved", "top_c", "bottom_c"}.
    L : int
        Total stack depth.
    k : int
        Sandwich half-width (only used by sandwich / inverse_sandwich).
    LC : int
        Number of C-blocks (only used by top_c / bottom_c).

    Returns
    -------
    str
        A length-L schedule string of 'S' and 'C' characters.
    """
    name = name.lower()
    if name == "all_s":
        return "S" * L
    if name == "all_c":
        return "C" * L
    if name == "sandwich":
        if 2 * k > L:
            raise ValueError(f"sandwich requires 2*k <= L; got k={k}, L={L}")
        return "S" * k + "C" * (L - 2 * k) + "S" * k
    if name == "inverse_sandwich":
        if 2 * k > L:
            raise ValueError(
                f"inverse_sandwich requires 2*k <= L; got k={k}, L={L}"
            )
        return "C" * k + "S" * (L - 2 * k) + "C" * k
    if name == "interleaved":
        if L % 2 != 0:
            raise ValueError(f"interleaved requires even L; got L={L}")
        return "SC" * (L // 2)
    if name == "top_c":
        if LC > L:
            raise ValueError(f"top_c requires LC <= L; got LC={LC}, L={L}")
        return "S" * (L - LC) + "C" * LC
    if name == "bottom_c":
        if LC > L:
            raise ValueError(f"bottom_c requires LC <= L; got LC={LC}, L={L}")
        return "C" * LC + "S" * (L - LC)
    raise ValueError(
        f"unknown schedule name {name!r}; valid: all_s, all_c, sandwich, "
        f"inverse_sandwich, interleaved, top_c, bottom_c."
    )


def parse_schedule_sc(schedule: str) -> List[str]:
    """Validate and parse a schedule string of 'S' and 'C' tokens."""
    sigma = list(schedule.upper())
    bad = sorted(set(sigma) - {"S", "C"})
    if bad:
        raise ValueError(
            f"schedule {schedule!r} contains invalid block types {bad}; "
            f"only 'S' and 'C' are allowed."
        )
    return sigma


def schedule_counts_sc(sigma: List[str]) -> Tuple[int, int]:
    """Return (n_S_blocks, n_C_blocks) for a parsed schedule."""
    nS = sum(1 for c in sigma if c == "S")
    nC = sum(1 for c in sigma if c == "C")
    return nS, nC


# ---------------------------------------------------------------------------
# Skew kernel modules
# ---------------------------------------------------------------------------
class SkewKernelLowRank(nn.Module):
    """Constant low-rank skew matrix kernel J_phi = U V^T - V U^T.

    The J_phi map is fixed (does not depend on its position-difference
    argument); the pair structure of the C-block enters only through
    the Gumbel top-k routing mask m_{ts}. The kernel is enforced
    skew-symmetric by construction (J_phi + J_phi^T = 0 exactly), so
    the velocity-coupled term J_phi v has zero work and zero divergence
    in position regardless of optimisation state.

    Parameters
    ----------
    d : int
        Hidden dimension (the dimension of the velocity proxy).
    rank : int
        Low-rank factor; J_+ = U V^T with U, V in R^{d x rank}.
    init_scale : float
        Frobenius scale at initialisation. The U and V columns are
        drawn from N(0, init_scale^2 / sqrt(rank)) so the initial
        ||J_phi||_F is approximately `init_scale`. Default 0.02
        matches the architecture v3 doc section 5.3.
    """

    def __init__(self, d: int, rank: int, init_scale: float = 0.02):
        super().__init__()
        self.d = d
        self.rank = rank
        std = init_scale / max(1.0, float(rank) ** 0.5)
        self.U = nn.Parameter(torch.randn(d, rank) * std)
        self.V = nn.Parameter(torch.randn(d, rank) * std)

    def matrix(self) -> torch.Tensor:
        """Materialise J_phi = U V^T - V U^T as a (d, d) tensor.

        Used by diagnostics (Frobenius norm, Jacobian-symmetry probe).
        Not used in the forward path; the forward avoids materialising
        the (d, d) matrix, taking O(d * rank) per token instead of
        O(d^2).
        """
        UV = self.U @ self.V.transpose(0, 1)
        return UV - UV.transpose(0, 1)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """Apply J_phi to v.

        v has shape (..., d); returns (..., d). Computed as
            J_phi v = U (V^T v) - V (U^T v),
        which is O(d * rank) per token versus O(d^2) for the full
        matrix multiply.
        """
        Vt_v = v @ self.V
        Ut_v = v @ self.U
        return Vt_v @ self.U.transpose(0, 1) - Ut_v @ self.V.transpose(0, 1)


class PerTokenGyroKernel(nn.Module):
    """Per-token gyroscopic kernel Omega = Omega_+ - Omega_+^T low-rank.

    Architecturally identical to SkewKernelLowRank; the distinction is
    that this kernel is applied to the query-side delta_t alone (no
    pair structure), giving f_t^gyro = Omega @ delta_t. Used only when
    cfg.use_pertoken_gyro = True (Q9e-D cell).
    """

    def __init__(self, d: int, rank: int, init_scale: float = 0.02):
        super().__init__()
        self.d = d
        self.rank = rank
        std = init_scale / max(1.0, float(rank) ** 0.5)
        self.U = nn.Parameter(torch.randn(d, rank) * std)
        self.V = nn.Parameter(torch.randn(d, rank) * std)

    def matrix(self) -> torch.Tensor:
        UV = self.U @ self.V.transpose(0, 1)
        return UV - UV.transpose(0, 1)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        Vt_v = v @ self.V
        Ut_v = v @ self.U
        return Vt_v @ self.U.transpose(0, 1) - Ut_v @ self.V.transpose(0, 1)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class SPHSPLMConfig(SparsePARFConfig):
    """SP-HSPLM configuration extending SparsePARFConfig.

    SP-HSPLM-specific knobs:

      schedule : str
        The block-type schedule of length L over {'S', 'C'}. Built via
        `make_schedule_sc` for canonical ladder cells.
      kernel_rank : int
        Low-rank factor of the C-block skew kernel J_phi.
      kernel_init_scale : float
        Frobenius initialisation scale of J_phi. Default 0.02.
      use_pertoken_gyro : bool
        If True, the C-block also applies the per-token gyroscopic
        kernel Omega(h_t) delta_t on top of the pair-skew force.
      gyro_rank : int
        Low-rank factor for Omega when use_pertoken_gyro is True.
      gamma_min : float
        Lower bound on the learnable damping gamma; enforced as
        gamma = gamma_min + softplus(raw_gamma - softplus_inverse(gamma_min)).
        Architecture v3 doc section 8.3 prescribes gamma_min = 0.05.

    Mechanism-1 (per-layer-indexed force law) knobs.  Defaults
    preserve the SPLM autonomous commitment (one parametric family
    shared across all layers); flipping any flag to False replaces the
    shared submodule with an nn.ModuleList of independent copies
    indexed by the relevant block-type schedule position.  Cells
    q9e_h/q9e_i/q9e_j/q9e_k/q9e_n of the Stage 2 ladder exercise these
    flags.  See paper Appendix A (Eq. A.130) for the theoretical
    Class-F prescription this targets, and architecture v3 doc §3.2
    for the corresponding "shared across ell" SPLM commitment.

      share_skew_kernel_across_layers : bool
        If True (default), one J_phi is shared across all C-blocks.
        If False, build nn.ModuleList(L_C) of independent J_phi^(ell)
        kernels, one per C-block layer index.
      share_gyro_kernel_across_layers : bool
        If True (default), one Omega is shared across all C-blocks
        (only relevant when use_pertoken_gyro=True).  If False, build
        nn.ModuleList(L_C) of independent Omega^(ell) kernels.
      share_v_phi_across_layers : bool
        If True (default), one V_phi is shared across all S-blocks
        (the parent PARFLM behaviour).  If False, build
        nn.ModuleList(L_S) of independent V_phi^(ell) modules, one per
        S-block layer index.
      share_score_head_across_layers : bool
        If True (default), one alpha_phi (ScoreHead) is shared across
        all layers (the SparsePARFLM behaviour).  If False, build
        nn.ModuleList(L) of independent alpha_phi^(ell) heads, one per
        layer; the same per-layer head feeds both the S-block V_phi
        routing and the C-block J_phi routing at that layer.
      share_v_theta_across_layers : bool
        If True (default), one V_theta is shared across all S-blocks
        (the parent PARFLM behaviour).  If False, build
        nn.ModuleList(L_S) of independent V_theta^(ell) modules, one
        per S-block layer index.  This is the **dominant** per-layer
        knob -- V_theta is the largest single force-law module
        (~2.6M params at v_hidden=1024, v_depth=3), so flipping it
        roughly doubles the model parameter budget at the canonical
        Stage 2 (d=256, L=8) scale.  Exercised by cell q9e_n
        (full-Class-F test); breaks iso-parameter-count comparability
        with the P10g baseline by design.
    """

    schedule: str = "SCSCSCSC"
    kernel_rank: int = 16
    kernel_init_scale: float = 0.02
    use_pertoken_gyro: bool = False
    gyro_rank: int = 16
    gyro_init_scale: float = 0.02
    gamma_min: float = 0.05

    # Mechanism-1 (layer-indexed force law) per-layer-isation flags;
    # defaults preserve the SPLM autonomous commitment.
    share_skew_kernel_across_layers: bool = True
    share_gyro_kernel_across_layers: bool = True
    share_v_phi_across_layers: bool = True
    share_score_head_across_layers: bool = True
    share_v_theta_across_layers: bool = True


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class ScalarPotentialLMSPHSPLM(SparsePARFLM):
    """SP-HSPLM (Q9(e)) attention-free Helmholtz model.

    Inherits the SparsePARFLM forward shell (embedding -> integrator
    over L layers -> tied LM head), the SparsePARFLM score head and
    Gumbel mask, and the SparsePARFLM dense `_layer_step` for S-blocks.
    Adds a `_c_block_step` for C-blocks and dispatches per layer index
    based on `cfg.schedule`.

    The integrator is iterated explicitly in `integrate(...)` rather
    than via the SparsePARFLM L-layer loop, because we need per-layer
    schedule lookup. This is the only structural deviation from the
    parent class.
    """

    def __init__(self, cfg: SPHSPLMConfig):
        super().__init__(cfg)
        self.cfg = cfg
        sigma = parse_schedule_sc(cfg.schedule)
        if len(sigma) != cfg.L:
            raise ValueError(
                f"schedule length {len(sigma)} != cfg.L {cfg.L}; "
                f"got schedule {cfg.schedule!r}."
            )
        self._sigma = sigma

        # Schedule-position maps:
        #   layer_idx -> position-in-S-block-list (or -1 if not S)
        #   layer_idx -> position-in-C-block-list (or -1 if not C)
        # Used by per-layer module dispatch to skip allocating dead
        # weights at the non-matching block-type layers.
        s_indices = [i for i, b in enumerate(sigma) if b == "S"]
        c_indices = [i for i, b in enumerate(sigma) if b == "C"]
        self._s_block_indices: List[int] = s_indices
        self._c_block_indices: List[int] = c_indices
        s_pos = {i: p for p, i in enumerate(s_indices)}
        c_pos = {i: p for p, i in enumerate(c_indices)}
        self._layer_to_s_pos: List[int] = [
            s_pos.get(ell, -1) for ell in range(cfg.L)
        ]
        self._layer_to_c_pos: List[int] = [
            c_pos.get(ell, -1) for ell in range(cfg.L)
        ]
        n_S = len(s_indices)
        n_C = len(c_indices)

        # ----- C-block skew kernel J_phi -----
        if cfg.share_skew_kernel_across_layers:
            self.skew_kernel: nn.Module = SkewKernelLowRank(
                d=cfg.d, rank=cfg.kernel_rank,
                init_scale=cfg.kernel_init_scale,
            )
        else:
            # nn.ModuleList of size L_C; one J_phi^(ell) per C-block.
            # If the schedule has no C-blocks (e.g. all_s smoke), the
            # ModuleList is empty and _c_block_step will never be hit.
            self.skew_kernel = nn.ModuleList([
                SkewKernelLowRank(
                    d=cfg.d, rank=cfg.kernel_rank,
                    init_scale=cfg.kernel_init_scale,
                ) for _ in range(max(n_C, 0))
            ])

        # ----- C-block gyroscopic kernel Omega -----
        if cfg.use_pertoken_gyro:
            if cfg.share_gyro_kernel_across_layers:
                self.gyro_kernel: Optional[nn.Module] = PerTokenGyroKernel(
                    d=cfg.d, rank=cfg.gyro_rank,
                    init_scale=cfg.gyro_init_scale,
                )
            else:
                self.gyro_kernel = nn.ModuleList([
                    PerTokenGyroKernel(
                        d=cfg.d, rank=cfg.gyro_rank,
                        init_scale=cfg.gyro_init_scale,
                    ) for _ in range(max(n_C, 0))
                ])
        else:
            self.gyro_kernel = None

        # ----- S-block scalar potential V_theta (dominant force) -----
        # Parent PARFLM has already built `self.V_theta` as a single
        # shared ScalarPotential.  V_theta is consumed only at S-block
        # layers in the current SP-HSPLM design (C-blocks apply
        # skew + optional gyro, no scalar potential), so per-layer
        # V_theta is indexed by S-block position (length L_S), not by
        # raw layer index.  This matches the V_phi convention below.
        # Replacing the attribute correctly de-registers the parent's
        # V_theta and registers the new ModuleList in self._modules.
        if not cfg.share_v_theta_across_layers:
            self.V_theta = nn.ModuleList([
                _build_v_theta(cfg) for _ in range(max(n_S, 0))
            ])

        # ----- S-block conservative pair scalar V_phi -----
        # Parent SparsePARFLM has already built `self.V_phi` as a
        # single shared module.  If the user asked for per-layer
        # V_phi, replace it with an nn.ModuleList(L_S) of fresh copies
        # built via the same factory the parent uses.  Reassigning the
        # attribute correctly de-registers the parent's V_phi and
        # registers the new ModuleList in `self._modules` (standard
        # nn.Module behaviour).
        if not cfg.share_v_phi_across_layers:
            self.V_phi = nn.ModuleList([
                _build_v_phi(cfg) for _ in range(max(n_S, 0))
            ])

        # ----- ScoreHead alpha_phi (shared between S- and C-branches
        # at any layer where both are active; architecture v3 doc
        # section 4.1) -----
        # Parent SparsePARFLM has already built `self.score_head`.
        # When per-layer, replace with nn.ModuleList(L) since the
        # score head is consumed at every layer (S or C).
        if not cfg.share_score_head_across_layers:
            self.score_head = nn.ModuleList([
                _build_score_head(cfg) for _ in range(cfg.L)
            ])

    @property
    def sigma(self) -> List[str]:
        """The parsed schedule list, length L, entries in {'S', 'C'}."""
        return list(self._sigma)

    def schedule_counts(self) -> Tuple[int, int]:
        """Return (n_S_blocks, n_C_blocks)."""
        return schedule_counts_sc(self._sigma)

    # ------------------------------------------------------------------
    # Per-layer module dispatch (Mechanism-1 support)
    # ------------------------------------------------------------------
    def _skew_kernel_at(self, layer_idx: int) -> nn.Module:
        """Return the J_phi module to apply at the given C-block layer.

        When `share_skew_kernel_across_layers=True` this is the single
        shared `SkewKernelLowRank` for every C-block.  When False this
        is the `layer_idx`-th C-block's dedicated `J_phi^(ell)`.
        """
        if isinstance(self.skew_kernel, nn.ModuleList):
            pos = self._layer_to_c_pos[layer_idx]
            if pos < 0:
                raise RuntimeError(
                    f"_skew_kernel_at called at non-C layer {layer_idx} "
                    f"(schedule={self.cfg.schedule!r})"
                )
            return self.skew_kernel[pos]
        return self.skew_kernel

    def _gyro_kernel_at(self, layer_idx: int) -> Optional[nn.Module]:
        """Return the Omega module to apply at this C-block layer, or None."""
        if self.gyro_kernel is None:
            return None
        if isinstance(self.gyro_kernel, nn.ModuleList):
            pos = self._layer_to_c_pos[layer_idx]
            if pos < 0:
                raise RuntimeError(
                    f"_gyro_kernel_at called at non-C layer {layer_idx}"
                )
            return self.gyro_kernel[pos]
        return self.gyro_kernel

    def _v_phi_at(self, layer_idx: int) -> nn.Module:
        """Return the V_phi module to apply at this S-block layer."""
        if isinstance(self.V_phi, nn.ModuleList):
            pos = self._layer_to_s_pos[layer_idx]
            if pos < 0:
                raise RuntimeError(
                    f"_v_phi_at called at non-S layer {layer_idx} "
                    f"(schedule={self.cfg.schedule!r})"
                )
            return self.V_phi[pos]
        return self.V_phi

    def _v_theta_at(self, layer_idx: int) -> nn.Module:
        """Return the V_theta module to apply at this S-block layer.

        When `share_v_theta_across_layers=True` this is the single
        shared `ScalarPotential` inherited from PARFLM.  When False
        this is the `layer_idx`-th S-block's dedicated
        `V_theta^(ell)` from the nn.ModuleList constructed in
        `__init__`.  Like `_v_phi_at`, the index is the S-block
        position, not the raw layer index, since V_theta is consumed
        only at S-block layers in the current SP-HSPLM design.
        """
        if isinstance(self.V_theta, nn.ModuleList):
            pos = self._layer_to_s_pos[layer_idx]
            if pos < 0:
                raise RuntimeError(
                    f"_v_theta_at called at non-S layer {layer_idx} "
                    f"(schedule={self.cfg.schedule!r})"
                )
            return self.V_theta[pos]
        return self.V_theta

    def _score_head_at(self, layer_idx: int) -> nn.Module:
        """Return the alpha_phi score head to apply at this layer.

        The score head is consumed at every layer regardless of block
        type (architecture v3 doc section 4.1: shared between S- and
        C-branches), so the per-layer variant is indexed directly by
        `layer_idx` rather than by S- or C-block position.
        """
        if isinstance(self.score_head, nn.ModuleList):
            return self.score_head[layer_idx]
        return self.score_head

    # ------------------------------------------------------------------
    # Skew-kernel Frobenius-squared aggregator (training-loop helper)
    # ------------------------------------------------------------------
    def skew_kernel_frobenius_squared(self) -> torch.Tensor:
        """Sum of ||J_phi||_F^2 over all (shared or per-layer) kernels.

        Used by the Stage 2 trainer's optional warm-up regulariser
        (`lam_skew * sum_ell ||J_phi^(ell)||_F^2`).  When the kernel is
        shared this collapses to a single ||J_phi||_F^2; when
        per-layer it sums over all L_C kernels, so the warm-up
        suppression intensity scales linearly with the kernel count
        (matching the per-cell regularisation budget of the shared
        case at one-per-block-type granularity).
        """
        if isinstance(self.skew_kernel, nn.ModuleList):
            if len(self.skew_kernel) == 0:
                return torch.zeros(
                    (), device=next(self.parameters()).device,
                )
            total = None
            for k in self.skew_kernel:
                J = k.matrix()
                term = (J * J).sum()
                total = term if total is None else total + term
            return total
        J = self.skew_kernel.matrix()
        return (J * J).sum()

    # ------------------------------------------------------------------
    # C-block step
    # ------------------------------------------------------------------
    def _c_block_step(
        self,
        h: torch.Tensor,
        h_prev: torch.Tensor,
        m_b: torch.Tensor,
        gamma: torch.Tensor,
        dt: float,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """One velocity-Verlet step of the SP-HSPLM C-block.

        Force at position t is the masked pair-skew sum
            f_t^sol = sum_{s<t} m_{ts} * J_phi (delta_s - delta_t)
        plus, optionally, the per-token gyro term
            f_t^gyro = Omega(h_t) delta_t,
        where the routing mask m_{ts} is the same Gumbel top-k mask
        used by the SparsePARFLM S-block (`_sparse_mask`).

        Causal-leak invariant
        ---------------------
        The .detach() of the source-side delta_s is the load-bearing
        invariant of this step. We compute
            delta_src = delta.detach()
            weighted_src = m @ delta_src
            arg = weighted_src - mask_sum * delta
        where the un-detached `delta` is used only for the diagonal
        delta_t component (the t-th row of `delta` is the query-side
        velocity). The weighted_src contains only s < t entries because
        m is causal, so no future-position information enters f_t.
        """
        cfg = self.cfg
        B, T, d = h.shape

        delta = h - h_prev

        # Score head -> sparse mask (inherits SparsePARFLM's
        # source-side detach via cfg.score_head_use_detached_h_src and
        # the strict causal mask).  In the per-layer score-head cell
        # the head is selected by layer_idx (see _score_head_at).
        h_in = h
        if not h_in.requires_grad:
            h_in = h_in.requires_grad_(True)

        h_src_for_score = (
            h_in.detach() if cfg.score_head_use_detached_h_src else h_in
        )
        score_head = self._score_head_at(layer_idx)
        pi = score_head(h_in, h_src_for_score)
        causal = self._pair_mask_for(T, h_in.device)
        tilde_m = self._sparse_mask(pi, causal, T)

        # Pair-skew force: f_t^sol = sum_s m_ts * J_phi (delta_s - delta_t)
        #               = J_phi * (sum_s m_ts delta_s - mask_sum_t * delta_t)
        # delta_src is detached so future-position information cannot
        # leak into the gradient w.r.t. the embedding inputs.
        delta_src = delta.detach()
        weighted_src = torch.matmul(tilde_m, delta_src)
        mask_sum = tilde_m.sum(dim=-1, keepdim=True)
        arg = weighted_src - mask_sum * delta

        skew = self._skew_kernel_at(layer_idx)
        f_sol = skew(arg)

        f_total = f_sol
        gyro = self._gyro_kernel_at(layer_idx)
        if gyro is not None:
            f_gyro = gyro(delta)
            f_total = f_total + f_gyro

        denom = 1.0 + dt * gamma
        h_new = h_in + delta / denom + (dt * dt / (m_b * denom)) * f_total

        if cfg.ln_after_step:
            h_new = self._project(h_new)
        return h_new

    # ------------------------------------------------------------------
    # _layer_step override: dispatch to per-layer S-block path when the
    # parent's `self.V_phi` or `self.score_head` is a per-layer ModuleList
    # ------------------------------------------------------------------
    def _layer_step(
        self,
        h: torch.Tensor,
        h_prev: torch.Tensor,
        m_b: torch.Tensor,
        gamma: torch.Tensor,
        dt: float,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """S-block step dispatcher.

        When `share_v_theta_across_layers`,
        `share_v_phi_across_layers`, and
        `share_score_head_across_layers` are all True (the SPLM
        autonomous commitment, default), the parent's `_layer_step`
        is bit-equivalent to what we want -- delegate to it.

        When any of those flags is False, the corresponding
        attribute (`self.V_theta`, `self.V_phi`, or
        `self.score_head`) is an `nn.ModuleList` and the parent's
        direct module-call would fail.  Route through
        `_layer_step_per_layer` instead.
        """
        per_layer_v_theta = isinstance(self.V_theta, nn.ModuleList)
        per_layer_v_phi = isinstance(self.V_phi, nn.ModuleList)
        per_layer_score = isinstance(self.score_head, nn.ModuleList)
        if per_layer_v_theta or per_layer_v_phi or per_layer_score:
            return self._layer_step_per_layer(
                h, h_prev, m_b, gamma, dt, layer_idx,
            )
        return super()._layer_step(h, h_prev, m_b, gamma, dt, layer_idx)

    def _layer_step_per_layer(
        self,
        h: torch.Tensor,
        h_prev: torch.Tensor,
        m_b: torch.Tensor,
        gamma: torch.Tensor,
        dt: float,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """Per-layer-dispatched copy of SparsePARFLM._layer_step.

        Hard-forked from `model_parf_sparse.SparsePARFLM._layer_step`
        (kept in sync with that source) with the only difference being
        the replacement of `self.V_theta`, `self.V_phi`, and
        `self.score_head` by their per-layer dispatchers.  Used by
        the Mechanism-1 cells (q9e_i/j/k/n) which promote any of
        V_theta / V_phi / alpha_phi to `nn.ModuleList`s indexed by
        S-block / layer position.
        """
        cfg = self.cfg
        B, T, d = h.shape
        delta = h - h_prev

        xi_input = h.detach() if cfg.causal_force else h
        xi_now = causal_cumulative_mean(xi_input)

        h_in = h
        if not h_in.requires_grad:
            h_in = h_in.requires_grad_(True)

        h_src = h_in.detach() if cfg.causal_force else h_in
        h_src_for_score = (
            h_in.detach() if cfg.score_head_use_detached_h_src else h_in
        )

        # 1. V_theta evaluation (per-S-block when share_v_theta_across_layers=False).
        v_theta = self._v_theta_at(layer_idx)
        V_th_per_token = v_theta(xi_now, h_in)

        # 2. Per-layer score head -> straight-through composite mask.
        score_head = self._score_head_at(layer_idx)
        pi = score_head(h_in, h_src_for_score)
        causal = self._pair_mask_for(T, h_in.device)
        tilde_m = self._sparse_mask(pi, causal, T)

        # 3. Per-layer V_phi -> dense eval, sparse aggregation.
        v_phi = self._v_phi_at(layer_idx)
        if cfg.use_grad_checkpoint and self.training:
            P = torch.utils.checkpoint.checkpoint(
                v_phi, h_in, h_src, use_reentrant=False,
            )
        else:
            P = v_phi(h_in, h_src)
        P_masked = (P * tilde_m).masked_fill(~causal, 0.0)
        s_ell = self.per_layer_scale(layer_idx)
        if s_ell is not None:
            P_masked = P_masked * s_ell
        U = V_th_per_token.sum() + P_masked.sum()

        grad_U, = torch.autograd.grad(
            U, h_in,
            create_graph=self.training,
            retain_graph=True,
        )
        f = -grad_U

        denom = 1.0 + dt * gamma
        h_new = h_in + delta / denom + (dt * dt / (m_b * denom)) * f

        if cfg.ln_after_step:
            h_new = self._project(h_new)
        return h_new

    # ------------------------------------------------------------------
    # Per-layer dispatch
    # ------------------------------------------------------------------
    def _layer_step_sc(
        self,
        h: torch.Tensor,
        h_prev: torch.Tensor,
        m_b: torch.Tensor,
        gamma: torch.Tensor,
        dt: float,
        layer_idx: int,
    ) -> torch.Tensor:
        """Dispatch to S-block or C-block based on the schedule."""
        block = self._sigma[layer_idx]
        if block == "S":
            return self._layer_step(h, h_prev, m_b, gamma, dt, layer_idx)
        if block == "C":
            return self._c_block_step(h, h_prev, m_b, gamma, dt, layer_idx)
        raise ValueError(
            f"invalid block token {block!r} at layer {layer_idx}; "
            f"valid: 'S', 'C'."
        )

    # ------------------------------------------------------------------
    # Override _stack_forward(): per-layer schedule dispatch
    # ------------------------------------------------------------------
    def _stack_forward(
        self,
        h0: torch.Tensor,
        x: torch.Tensor,
        return_trajectory: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Walk the L SP-HSPLM layers with per-layer S/C dispatch.

        Parallels PARFLM._stack_forward but consults `self._sigma` to
        choose the layer step. The kinematic memory `h_prev` is
        propagated across both block types so the velocity proxy
        delta = h - h_prev is well-defined regardless of the schedule.
        """
        cfg = self.cfg

        gamma = self.gamma
        gamma = torch.clamp(gamma, min=float(cfg.gamma_min))
        dt = float(cfg.dt)
        m_b = self.compute_mass(x)

        h = h0
        h_prev = h0

        traj: Optional[List[torch.Tensor]] = None
        if return_trajectory:
            traj = [h.detach().cpu()]

        for ell in range(cfg.L):
            h_new = self._layer_step_sc(
                h, h_prev, m_b, gamma, dt, layer_idx=ell,
            )
            h_prev = h
            h = h_new
            if traj is not None:
                traj.append(h.detach().cpu())

        return h, traj


# ---------------------------------------------------------------------------
# Stage 2 cell registry
# ---------------------------------------------------------------------------
def _stage2_cell_kwargs(cell: str, base_kwargs: dict) -> dict:
    """Return the SPHSPLMConfig kwargs for the given Stage 2 cell name.

    Cells q9e_a..q9e_g are the original Mechanism-2 ladder (autonomous
    force law; the SPLM "shared across ell" commitment is preserved):
      q9e_a : interleaved, k=4,  r=16, no gyro.  Central bet.
      q9e_b : interleaved, k=8,  r=16, no gyro.  Routing-density sweep.
      q9e_c : interleaved, k=4,  r=32, no gyro.  Kernel-rank sweep.
      q9e_d : interleaved, k=4,  r=16, with per-token gyro Omega.
      q9e_e : bottom_c    (CCCCSSSS), k=4, r=16, no gyro.
      q9e_f : top_c       (SSSSCCCC), k=4, r=16, no gyro.
      q9e_g : sandwich    (SSCCCCSS), k=4, r=16, no gyro.

    Cells q9e_h..q9e_k are the Mechanism-1 extension (per-layer-indexed
    force law; lifts the SPLM "shared across ell" commitment one
    submodule at a time, otherwise matching q9e_a's config):
      q9e_h : q9e_a + per-layer J_phi^(ell) (L_C independent skew
              kernels).  Tests whether per-layer skew kernel parameters
              recover the gap that capacity sweeps in q9e_b/c could not.
      q9e_i : q9e_a + per-layer V_phi^(ell) (L_S independent
              conservative pair scalars).
      q9e_j : q9e_a + per-layer alpha_phi^(ell) (L independent
              score heads driving the Gumbel top-k routing).
      q9e_k : q9e_a + ALL of {J_phi, V_phi, alpha_phi} per-layer.  The
              joint Mechanism-1 cell; the cleanest test of the paper
              Appendix A Class-F prescription.

    Cell q9e_l is the Mechanism-1 x Mechanism-2 additivity cell:
      q9e_l : q9e_d (per-token gyro Omega ON, shared) + q9e_h
              (per-layer J_phi^(ell)).  Tests the H6 additivity
              hypothesis -- if q9e_d's -0.69 PPL Omega gain and
              q9e_h's -0.90 PPL Mechanism-1 gain are additive, q9e_l
              lands at PPL ~ 25.99, clearing the P10g 26.42 ceiling
              for Outcome ALPHA confirmation.

    Cell q9e_m is the maximal-Mechanism-1 cell built on top of q9e_l:
      q9e_m : q9e_l (per-layer J_phi^(ell) + per-token gyro Omega ON)
              + per-layer Omega^(ell) (L_C independent gyro kernels).
              Lifts the last shared submodule that still carried
              non-conservative work in q9e_l (Omega).  Tests whether
              the q9e_l synergy can be pushed further by letting each
              C-block carry its own Omega.  Contingent next-step
              after q9e_l multi-seed confirms the seed-1 25.11 PPL.

    Cell q9e_n is the full-Class-F test built on top of q9e_m:
      q9e_n : q9e_m + per-layer V_theta^(ell) + per-layer V_phi^(ell)
              + per-layer alpha_phi^(ell).  Promotes every
              SP-HSPLM-side force-law module to per-layer.  The
              dominant new piece is V_theta (~2.6M params per copy at
              v_hidden=1024, v_depth=3), so q9e_n breaks the
              iso-parameter-count comparability with P10g and roughly
              doubles the model.  This is the test of the user's
              standing question: does the residual ~17 PPL gap between
              q9e_l (PPL 25.11) and MatchedGPT (~8 PPL) collapse once
              V_theta is no longer the autonomous-in-ell bottleneck?
              See H8 in
              docs/SP_HSPLM_Stage_2_pre-registered_protocol.md.

    See docs/SP_HSPLM_Stage_2_pre-registered_protocol.md section 3 and
    docs/Scalar_Potential_based_Helmholtz_Architecture_v3.md section 9.2
    for the per-cell hypothesis and decision rule.
    """
    L = base_kwargs.get("L", 8)
    overrides: dict
    if cell == "q9e_a":
        overrides = {
            "schedule": make_schedule_sc("interleaved", L=L),
            "top_k": 4, "kernel_rank": 16, "use_pertoken_gyro": False,
        }
    elif cell == "q9e_b":
        overrides = {
            "schedule": make_schedule_sc("interleaved", L=L),
            "top_k": 8, "kernel_rank": 16, "use_pertoken_gyro": False,
        }
    elif cell == "q9e_c":
        overrides = {
            "schedule": make_schedule_sc("interleaved", L=L),
            "top_k": 4, "kernel_rank": 32, "use_pertoken_gyro": False,
        }
    elif cell == "q9e_d":
        overrides = {
            "schedule": make_schedule_sc("interleaved", L=L),
            "top_k": 4, "kernel_rank": 16, "use_pertoken_gyro": True,
            "gyro_rank": 16,
        }
    elif cell == "q9e_e":
        overrides = {
            "schedule": make_schedule_sc("bottom_c", L=L, LC=L // 2),
            "top_k": 4, "kernel_rank": 16, "use_pertoken_gyro": False,
        }
    elif cell == "q9e_f":
        overrides = {
            "schedule": make_schedule_sc("top_c", L=L, LC=L // 2),
            "top_k": 4, "kernel_rank": 16, "use_pertoken_gyro": False,
        }
    elif cell == "q9e_g":
        overrides = {
            "schedule": make_schedule_sc("sandwich", L=L, k=L // 4),
            "top_k": 4, "kernel_rank": 16, "use_pertoken_gyro": False,
        }
    elif cell == "q9e_h":
        # Mechanism-1: per-layer J_phi^(ell), everything else == q9e_a.
        overrides = {
            "schedule": make_schedule_sc("interleaved", L=L),
            "top_k": 4, "kernel_rank": 16, "use_pertoken_gyro": False,
            "share_skew_kernel_across_layers": False,
        }
    elif cell == "q9e_i":
        # Mechanism-1: per-layer V_phi^(ell), everything else == q9e_a.
        overrides = {
            "schedule": make_schedule_sc("interleaved", L=L),
            "top_k": 4, "kernel_rank": 16, "use_pertoken_gyro": False,
            "share_v_phi_across_layers": False,
        }
    elif cell == "q9e_j":
        # Mechanism-1: per-layer alpha_phi^(ell), everything else == q9e_a.
        overrides = {
            "schedule": make_schedule_sc("interleaved", L=L),
            "top_k": 4, "kernel_rank": 16, "use_pertoken_gyro": False,
            "share_score_head_across_layers": False,
        }
    elif cell == "q9e_k":
        # Mechanism-1 (joint): per-layer J_phi + V_phi + alpha_phi.
        # The cleanest test of the paper Appendix A two-mechanism
        # decomposition: Mechanism-2 (context, prefix) was carried by
        # q9e_a; q9e_k adds Mechanism-1 (per-layer-indexed theta_ell)
        # without instantiating any attention layer.
        overrides = {
            "schedule": make_schedule_sc("interleaved", L=L),
            "top_k": 4, "kernel_rank": 16, "use_pertoken_gyro": False,
            "share_skew_kernel_across_layers": False,
            "share_v_phi_across_layers": False,
            "share_score_head_across_layers": False,
        }
    elif cell == "q9e_l":
        # Mechanism-1 x Mechanism-2 additivity cell: q9e_d (per-token
        # gyro Omega ON, shared) + q9e_h (per-layer J_phi^(ell)).
        # H6 additivity test -- the natural extension after q9e_d
        # (PPL 26.89, -0.69 vs q9e_a) and q9e_h (PPL 26.68, -0.90 vs
        # q9e_a) both individually showed measurable improvement.
        # EXECUTED at seed 0: PPL 25.11 (-2.47 vs q9e_a, -1.31 vs
        # P10g) -- synergy bonus 0.88 PPL over the 25.99 additive
        # prediction.  First SP-HSPLM cell below the P10g ceiling.
        overrides = {
            "schedule": make_schedule_sc("interleaved", L=L),
            "top_k": 4, "kernel_rank": 16,
            "use_pertoken_gyro": True, "gyro_rank": 16,
            "share_skew_kernel_across_layers": False,
        }
    elif cell == "q9e_m":
        # Maximal-Mechanism-1 cell on top of q9e_l: q9e_l + per-layer
        # Omega^(ell).  Lifts the last shared submodule still carrying
        # non-conservative work (Omega ceded only ~16% of its norm to
        # the per-layer J_phi kernels in q9e_l; the rest is still
        # shared).  Tests whether the q9e_l synergy can be pushed
        # further by letting each C-block carry its own Omega.
        # Contingent next-step after q9e_l multi-seed confirmation.
        overrides = {
            "schedule": make_schedule_sc("interleaved", L=L),
            "top_k": 4, "kernel_rank": 16,
            "use_pertoken_gyro": True, "gyro_rank": 16,
            "share_skew_kernel_across_layers": False,
            "share_gyro_kernel_across_layers": False,
        }
    elif cell == "q9e_n":
        # Full-Class-F test on top of q9e_m: q9e_m + per-layer V_theta
        # + per-layer V_phi + per-layer alpha_phi.  Every force-law
        # module that can be per-layer-ised in the current SP-HSPLM
        # design is per-layer-ised.  V_theta is the dominant new
        # piece (~2.6M params per S-block copy at v_hidden=1024,
        # v_depth=3 -> ~10M extra params for L_S=4 S-blocks, roughly
        # doubling the model from ~15.8M to ~26M).  This intentionally
        # breaks the iso-parameter-count comparability with P10g and
        # is the cleanest test of the paper Appendix A Eq. A.130
        # Class-F prescription within the SP-HSPLM architecture.
        # See H8 in the pre-registered protocol.
        overrides = {
            "schedule": make_schedule_sc("interleaved", L=L),
            "top_k": 4, "kernel_rank": 16,
            "use_pertoken_gyro": True, "gyro_rank": 16,
            "share_skew_kernel_across_layers": False,
            "share_gyro_kernel_across_layers": False,
            "share_v_theta_across_layers": False,
            "share_v_phi_across_layers": False,
            "share_score_head_across_layers": False,
        }
    else:
        raise ValueError(
            f"unknown Stage 2 cell {cell!r}; valid: "
            f"q9e_a, q9e_b, q9e_c, q9e_d, q9e_e, q9e_f, q9e_g, "
            f"q9e_h, q9e_i, q9e_j, q9e_k, q9e_l, q9e_m, q9e_n."
        )

    out = dict(base_kwargs)
    out.update(overrides)
    return out


CELLS: Tuple[str, ...] = (
    "q9e_a", "q9e_b", "q9e_c", "q9e_d", "q9e_e", "q9e_f", "q9e_g",
    "q9e_h", "q9e_i", "q9e_j", "q9e_k", "q9e_l", "q9e_m", "q9e_n",
)


# ---------------------------------------------------------------------------
# Diagnostic: pair kernel norms (for H4 + Outcome DELTA detection)
# ---------------------------------------------------------------------------
def _kernel_norms_one(kernel: nn.Module) -> dict:
    """Return {J_fro, U_fro, V_fro} for one SkewKernelLowRank or PerTokenGyro."""
    return {
        "J_fro": float(kernel.matrix().norm(p="fro").item()),
        "U_fro": float(kernel.U.norm(p="fro").item()),
        "V_fro": float(kernel.V.norm(p="fro").item()),
    }


def pair_kernel_norms(
    model: ScalarPotentialLMSPHSPLM,
) -> dict:
    """Per-block-type Frobenius norms of the SP-HSPLM kernels.

    Returns a dict containing both scalar aggregates (for the existing
    summary table) and per-layer lists (for the Mechanism-1 cells).

    Scalar keys (always present):
      - `J_phi_fro`: in the shared cell, the single ||J_phi||_F.
                    In a per-layer cell, the **mean** over all L_C
                    C-block kernels (so the headline summary number
                    is directly comparable to the shared case).
      - `U_fro`, `V_fro`: same convention.
      - `Omega_fro` etc.: present only when use_pertoken_gyro=True.

    Per-layer keys (present only when the kernel is `nn.ModuleList`):
      - `J_phi_fro_per_layer`: list of length L_C of ||J_phi^(ell)||_F.
      - `U_fro_per_layer`, `V_fro_per_layer`: ditto.
      - `J_phi_fro_total`: sqrt(sum_ell ||J_phi^(ell)||_F^2), the
        Frobenius norm of the stacked kernel block-diagonal -- the
        natural "total kernel work" across the C-block stack.
      - Analogous keys for Omega when use_pertoken_gyro=True.
    """
    out: dict = {}
    with torch.no_grad():
        if isinstance(model.skew_kernel, nn.ModuleList):
            per = [_kernel_norms_one(k) for k in model.skew_kernel]
            if not per:
                out["J_phi_fro"] = 0.0
                out["U_fro"] = 0.0
                out["V_fro"] = 0.0
                out["J_phi_fro_per_layer"] = []
                out["U_fro_per_layer"] = []
                out["V_fro_per_layer"] = []
                out["J_phi_fro_total"] = 0.0
            else:
                jf = [p["J_fro"] for p in per]
                uf = [p["U_fro"] for p in per]
                vf = [p["V_fro"] for p in per]
                out["J_phi_fro"] = sum(jf) / len(jf)
                out["U_fro"] = sum(uf) / len(uf)
                out["V_fro"] = sum(vf) / len(vf)
                out["J_phi_fro_per_layer"] = jf
                out["U_fro_per_layer"] = uf
                out["V_fro_per_layer"] = vf
                out["J_phi_fro_total"] = float(
                    sum(x * x for x in jf) ** 0.5
                )
        else:
            single = _kernel_norms_one(model.skew_kernel)
            out["J_phi_fro"] = single["J_fro"]
            out["U_fro"] = single["U_fro"]
            out["V_fro"] = single["V_fro"]

        if model.gyro_kernel is not None:
            if isinstance(model.gyro_kernel, nn.ModuleList):
                per = [_kernel_norms_one(k) for k in model.gyro_kernel]
                if not per:
                    out["Omega_fro"] = 0.0
                    out["Omega_U_fro"] = 0.0
                    out["Omega_V_fro"] = 0.0
                    out["Omega_fro_per_layer"] = []
                    out["Omega_U_fro_per_layer"] = []
                    out["Omega_V_fro_per_layer"] = []
                    out["Omega_fro_total"] = 0.0
                else:
                    of = [p["J_fro"] for p in per]
                    ouf = [p["U_fro"] for p in per]
                    ovf = [p["V_fro"] for p in per]
                    out["Omega_fro"] = sum(of) / len(of)
                    out["Omega_U_fro"] = sum(ouf) / len(ouf)
                    out["Omega_V_fro"] = sum(ovf) / len(ovf)
                    out["Omega_fro_per_layer"] = of
                    out["Omega_U_fro_per_layer"] = ouf
                    out["Omega_V_fro_per_layer"] = ovf
                    out["Omega_fro_total"] = float(
                        sum(x * x for x in of) ** 0.5
                    )
            else:
                single = _kernel_norms_one(model.gyro_kernel)
                out["Omega_fro"] = single["J_fro"]
                out["Omega_U_fro"] = single["U_fro"]
                out["Omega_V_fro"] = single["V_fro"]
    return out


# ---------------------------------------------------------------------------
# Smoke entry point (cheap sanity check)
# ---------------------------------------------------------------------------
def _smoke_test() -> None:
    """Minimal one-step round-trip on CPU for every Stage 2 cell.

    Not the real smoke test (that runs at the protocol-locked d/T/L on
    GPU). This just verifies that every cell builds, runs forward, and
    backwards without NaN.
    """
    base = dict(
        vocab_size=257, d=16, max_len=64, L=4,
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
    for cell in CELLS:
        torch.manual_seed(0)
        kwargs = _stage2_cell_kwargs(cell, base)
        cfg = SPHSPLMConfig(**kwargs)
        net = ScalarPotentialLMSPHSPLM(cfg)
        nS, nC = net.schedule_counts()
        n_params = sum(p.numel() for p in net.parameters())
        n_skew = sum(p.numel() for p in net.skew_kernel.parameters())
        n_gyro = (
            sum(p.numel() for p in net.gyro_kernel.parameters())
            if net.gyro_kernel is not None else 0
        )
        n_v_phi = sum(p.numel() for p in net.V_phi.parameters())
        n_v_theta = sum(p.numel() for p in net.V_theta.parameters())
        n_score_head = sum(p.numel() for p in net.score_head.parameters())
        per_layer_flags = []
        if not cfg.share_skew_kernel_across_layers:
            per_layer_flags.append("J_phi")
        if not cfg.share_gyro_kernel_across_layers and cfg.use_pertoken_gyro:
            per_layer_flags.append("Omega")
        if not cfg.share_v_theta_across_layers:
            per_layer_flags.append("V_theta")
        if not cfg.share_v_phi_across_layers:
            per_layer_flags.append("V_phi")
        if not cfg.share_score_head_across_layers:
            per_layer_flags.append("alpha_phi")
        per_layer_str = (
            ",".join(per_layer_flags) if per_layer_flags else "none"
        )
        print(
            f"[sphsplm-smoke] cell={cell:6s}  schedule={cfg.schedule}  "
            f"nS={nS} nC={nC}  k={cfg.top_k}  r={cfg.kernel_rank}  "
            f"gyro={cfg.use_pertoken_gyro}  per_layer={per_layer_str}  "
            f"params={n_params:,}  skew={n_skew}  "
            f"V_theta={n_v_theta}  V_phi={n_v_phi}  "
            f"alpha_phi={n_score_head}  gyro_params={n_gyro}"
        )

        x = torch.randint(0, cfg.vocab_size, (2, 16))
        y = torch.randint(0, cfg.vocab_size, (2, 16))
        net.train()
        out = net(x, y)
        loss = out[1]
        loss.backward()

        # Gradient-norm read-out: works for both shared (single kernel)
        # and per-layer (ModuleList) skew kernels.
        if isinstance(net.skew_kernel, nn.ModuleList):
            ker_iter = list(net.skew_kernel)
        else:
            ker_iter = [net.skew_kernel]
        kernel_grad_norm = 0.0
        for k in ker_iter:
            if k.U.grad is not None:
                kernel_grad_norm += float(k.U.grad.norm().item())
            if k.V.grad is not None:
                kernel_grad_norm += float(k.V.grad.norm().item())

        norms = pair_kernel_norms(net)
        per_layer_suffix = ""
        if "J_phi_fro_per_layer" in norms:
            per_layer_suffix = (
                f"  per_layer_J_phi="
                f"{[f'{v:.4f}' for v in norms['J_phi_fro_per_layer']]}"
            )
        print(
            f"[sphsplm-smoke]   loss={float(loss.item()):.4f}  "
            f"||J_phi||_F={norms['J_phi_fro']:.4f}"
            f"  ||grad_UV||={kernel_grad_norm:.6f}{per_layer_suffix}"
        )


if __name__ == "__main__":
    _smoke_test()
