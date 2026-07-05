"""Shared setup for the O-step Langevin Fock-PARFLM experiments.

This module factors the *build + calibrate + evaluate* logic that the two
O-step notebooks share:

  - ``colab_fock_ostep_langevin_openwebtext.ipynb``  (the long production run)
  - ``colab_fock_ostep_gammaT_sweep_openwebtext.ipynb`` (the Phase-A sweep)

There is **no new model class here**.  The "O-step Langevin" model is the
existing :class:`FockMultiXiPARFLM` (``model_fock_parf_multixi.py``) with the
per-layer damped Verlet step replaced by an *instance-level monkey-patch*
(:func:`install_ostep`) that turns each layer into a BAOAB step: the
conservative drift is run with the Rayleigh damping folded out, then a single
exact Ornstein-Uhlenbeck (O) substep carries BOTH the friction and an
FDT-locked thermostat noise.  It adds **zero parameters**; the only new degree
of freedom is the temperature ``T``.

Theory: ``companion_notes/
Langevin_dynamics_reformulation_of_classical_damped_Lagrangian_flow.md``
(section 7 O-step, section 8 retrofit, section 9 accuracy curriculum) and the
paper section ``sec:langevin-completion``.

IMPORTANT — keep in sync with the main notebook.  The defaults in
:class:`FockSetupConfig` mirror the config cell of
``colab_fock_ostep_langevin_openwebtext.ipynb`` so that a ``(gamma, T)``
selected by the sweep transfers directly to the production run.  The
``variant_tag`` helper reproduces that notebook's tag exactly; the module's
self-test asserts the equality.

All heavy imports (torch, the model modules) are done lazily inside the
functions so that ``import fock_ostep_setup`` works for a light self-test even
when the model modules are not yet on ``sys.path``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence


# ---------------------------------------------------------------------------
# Xi channel presets
# ---------------------------------------------------------------------------
XI_PRESETS = {
    5:       [0.25, 0.50, 0.75, 0.95, 0.99],
    "5long": [0.50, 0.75, 0.95, 0.99, 0.995],
    6:       [0.25, 0.50, 0.75, 0.95, 0.99, 0.995],
    "4long": [0.50, 0.75, 0.95, 0.995],
}


def resolve_xi_preset(xi_override):
    """Return (xi_alpha_inits, xi_channels) for a preset key.

    Mirrors the resolution in both notebooks' config/build cells.
    """
    if xi_override is None:
        alphas = [0.25, 0.50, 0.75, 0.95]
    elif xi_override in XI_PRESETS:
        alphas = list(XI_PRESETS[xi_override])
    else:
        raise ValueError(
            f"Unsupported xi_override={xi_override!r}; use None, 5, 6, "
            f'"5long", or "4long"'
        )
    return alphas, len(alphas)


# ---------------------------------------------------------------------------
# Configuration (mirrors the main notebook's Cell 0)
# ---------------------------------------------------------------------------
@dataclass
class FockSetupConfig:
    """All knobs needed to build an identically-configured Fock-PARFLM.

    Defaults reproduce the production O-step notebook.  Override only the
    fields you need (the sweep, for example, drives ``langevin_gamma`` and
    ``langevin_T`` per grid point).
    """

    # ── V_theta architecture ────────────────────────────────────────
    v_theta_variant: str = "gaussian"          # 'gaussian'|'sarf'|'sq3'|'mlp'
    w_scale: float = 1.0
    bg_quad_eps: float = 0.0
    sq3_tau: float = 1.0
    sq3_curv_max: Optional[float] = 2.0
    sarf_n_anchors: int = 64
    v_theta_wells_per_head: int = 8
    v_theta_depth_condition: bool = True
    v_theta_depth_code_init_std: float = 0.02
    xi_override: object = "5long"
    v_theta_n_heads: Optional[int] = None      # None -> xi_channels

    # ── PARF V_phi ───────────────────────────────────────────────────
    v_phi_kind: str = "structural_competitive"
    v_phi_mlp_hidden: int = 128
    top_k: int = 16
    v_phi_n_heads: int = 4
    v_phi_d_type: int = 32
    v_phi_d_angle: int = 16

    # ── Fock reverse channel ─────────────────────────────────────────
    reverse_channel: bool = False
    reverse_channel_stable: bool = True
    reverse_channel_pre_ln: bool = True
    reverse_channel_soft_norm: bool = True
    reverse_channel_warmup_steps: int = 4000
    reverse_channel_per_layer: bool = False

    # ── Register repulsion (B4) ──────────────────────────────────────
    register_repulsion: bool = True
    register_repulsion_coeff: float = 0.05
    register_repulsion_kind: str = "gram"

    # ── Read-out head ────────────────────────────────────────────────
    use_output_bias: bool = True
    tie_embeddings: bool = False

    # ── O-step Langevin retrofit ─────────────────────────────────────
    langevin_ostep: bool = True
    langevin_gamma: Optional[float] = None     # None -> reuse model.gamma
    langevin_T: float = 1.0
    langevin_tie_T_to_beta: bool = True
    langevin_noise_train: bool = True
    langevin_noise_eval: bool = False

    # ── Optimisation-shape knobs used by build/eval ──────────────────
    lambda_v: float = 1e-2
    block_size: int = 512
    vocab_size: int = 50257
    max_len: int = 1024

    # ── Architecture tiers (d, L, n_registers) to try in order ───────
    arch_tiers: Sequence = field(default_factory=lambda: (
        (384, 16, 32), (384, 12, 16), (256, 16, 16), (256, 8, 16),
    ))

    # ── Fixed model hyperparameters (rarely changed) ─────────────────
    v_hidden: int = 1024
    v_depth: int = 3
    dt: float = 1.0
    init_gamma: float = 1.0
    fixed_gamma: float = 0.30

    def __post_init__(self):
        self.xi_alpha_inits, self.xi_channels = resolve_xi_preset(self.xi_override)
        if self.v_theta_n_heads is None:
            self.v_theta_n_heads = self.xi_channels
        if self.langevin_tie_T_to_beta:
            # beta_readout = 1 in this readout parameterisation.
            self.langevin_T = 1.0

    # ── Derived helpers ──────────────────────────────────────────────
    @property
    def is_structured(self) -> bool:
        return self.v_theta_variant in ("gaussian", "sarf", "sq3")

    def effective_gamma(self, model_gamma: float) -> float:
        """The friction the O-step uses (explicit override or model gamma)."""
        return float(model_gamma) if self.langevin_gamma is None \
            else float(self.langevin_gamma)

    def variant_tag(self) -> str:
        """Reproduce the main notebook's ``_variant_tag`` string exactly."""
        parts = []
        if self.v_theta_variant == "mlp":
            parts.append("mlp_vtheta")
        elif self.v_theta_variant == "sq3":
            parts.append(f"sq3_k{self.v_theta_wells_per_head}")
        elif self.v_theta_variant == "gaussian" and self.v_theta_wells_per_head != 8:
            parts.append(f"k{self.v_theta_wells_per_head}")
        if self.v_phi_kind == "mlp":
            parts.append(f"mlp_vphi_h{self.v_phi_mlp_hidden}")
        elif self.v_phi_kind == "structural":
            parts.append("struct_vphi")
        if self.xi_override is not None:
            parts.append(f"xi{self.xi_override}")
        if self.top_k != 8:
            parts.append(f"topk{self.top_k}")
        if self.v_phi_d_type != 16 or self.v_phi_d_angle != 8:
            parts.append(f"dt{self.v_phi_d_type}da{self.v_phi_d_angle}")
        if self.v_phi_n_heads != 1:
            parts.append(f"mh{self.v_phi_n_heads}")
        if self.v_theta_n_heads > 1:
            if self.v_theta_depth_condition:
                parts.append(f"dcvt{self.v_theta_n_heads}x{self.v_theta_wells_per_head}")
            else:
                parts.append(f"mcvt{self.v_theta_n_heads}x{self.v_theta_wells_per_head}")
        if self.use_output_bias:
            parts.append("ob")
        if not self.tie_embeddings:
            parts.append("untied")
        # optimizer / grad-centralization tags are training-side; the sweep
        # and the main run both use adamw with GC off, so they add nothing.
        # LR schedule is a training knob; the main notebook appends 'wsd'.
        parts.append("wsd")
        if not self.reverse_channel:
            parts.append("e5a")
        elif self.reverse_channel_stable:
            parts.append("e5c")
        if self.reverse_channel and self.reverse_channel_per_layer:
            parts.append("plgate")
        if self.register_repulsion:
            parts.append(f"rep{self.register_repulsion_coeff:g}")
        if self.langevin_ostep:
            parts.append("ostep")
            parts.append(f"T{self.langevin_T:g}")
            if self.langevin_gamma is not None:
                parts.append(f"og{self.langevin_gamma:g}")
            if not self.langevin_noise_train:
                parts.append("ntr0")
            if self.langevin_noise_eval:
                parts.append("nev1")
        return "_".join(parts)


# ---------------------------------------------------------------------------
# O-step Langevin retrofit (instance-level monkey-patch, +0 params)
# ---------------------------------------------------------------------------
def install_ostep(model, gamma=None, T=1.0, noise_train=True, noise_eval=False,
                  verbose=True):
    """Retrofit the per-layer damped Verlet step into a BAOAB Langevin step.

    Verbatim behaviour of the production notebook's ``install_ostep``: an
    instance-level monkey-patch of ``model._fock_layer_step`` that (1) runs the
    conservative drift with the Rayleigh damping folded OUT (gamma -> 0 inside
    the Verlet), then (2) applies one exact Ornstein-Uhlenbeck (O) substep to
    the implicit velocity, carrying BOTH the friction and the FDT-locked
    thermal noise, and re-encodes the thermostatted velocity into ``h_prev``
    for the next layer.  Composes on top of the depth-routing wrapper, so
    install it AFTER :func:`build_structured_vtheta`.
    """
    import torch
    import types

    if getattr(model, "_ostep_installed", False):
        return
    prev = model._fock_layer_step            # bound (may already be depth-routed)
    dt = float(model.cfg.dt)
    g = float(model.gamma) if gamma is None else float(gamma)
    c1 = math.exp(-g * dt)                    # exact OU decay over one layer step
    one_minus_c1sq = 1.0 - c1 * c1
    # Mutable so a (gamma, T) sweep / anneal can retune T without reinstalling.
    model._ostep_cfg = {
        "gamma": g, "dt": dt, "c1": c1, "one_minus_c1sq": one_minus_c1sq,
        "T": float(T), "noise_train": bool(noise_train),
        "noise_eval": bool(noise_eval),
    }

    def _ostep_fock_layer_step(self, h, h_prev, r, salience, m_b, gamma, dt,
                               layer_idx):
        oc = self._ostep_cfg
        zero = torch.zeros((), device=h.device, dtype=h.dtype)
        # (1) Conservative drift (B-A ... A of BAOAB): damping folded out so
        #     the O-step is the sole friction; Q stays a post-Verlet correction.
        h_new, _h_prev_out, r_new, sal = prev(
            h, h_prev, r, salience, m_b, zero, dt, layer_idx=layer_idx,
        )
        # (2) Exact O-step on the implicit velocity v = (h_new - h)/dt.
        v = (h_new - h) / oc["dt"]
        v = oc["c1"] * v
        add_noise = ((self.training and oc["noise_train"])
                     or ((not self.training) and oc["noise_eval"]))
        if add_noise and oc["T"] > 0.0:
            # FDT: <v_i^2>_eq = kT/m  =>  per-step velocity noise var =
            #      (kT/m)(1 - e^{-2 gamma dt}).  m_b is (B,T,1); broadcasts.
            std = torch.sqrt((oc["T"] / m_b) * oc["one_minus_c1sq"])
            v = v + std * torch.randn_like(v)
        # Re-encode the thermostatted velocity for the next layer's leapfrog.
        h_prev_ret = h_new - oc["dt"] * v
        return h_new, h_prev_ret, r_new, sal

    model._fock_layer_step = types.MethodType(_ostep_fock_layer_step, model)
    model._ostep_installed = True
    if verbose:
        print(f"O-step installed: BAOAB thermostat on _fock_layer_step  "
              f"(gamma={g:.3f}, c1=exp(-g.dt)={c1:.3f}, T={T:g}, "
              f"noise train={noise_train}/eval={noise_eval}, +0 params)")


def set_ostep(model, T=None, gamma=None, noise_train=None, noise_eval=None):
    """Retune an already-installed O-step in place (no reinstall).

    This is the sweep's cheap knob: changing ``gamma`` recomputes the exact
    OU decay ``c1 = exp(-gamma*dt)`` and the FDT noise factor, and changing
    ``T`` rescales the thermostat amplitude.  Raises if the O-step is not
    installed.
    """
    if not getattr(model, "_ostep_installed", False):
        raise RuntimeError("set_ostep called before install_ostep")
    oc = model._ostep_cfg
    if gamma is not None:
        g = float(gamma)
        c1 = math.exp(-g * oc["dt"])
        oc["gamma"] = g
        oc["c1"] = c1
        oc["one_minus_c1sq"] = 1.0 - c1 * c1
    if T is not None:
        oc["T"] = float(T)
    if noise_train is not None:
        oc["noise_train"] = bool(noise_train)
    if noise_eval is not None:
        oc["noise_eval"] = bool(noise_eval)
    return dict(oc)


# ---------------------------------------------------------------------------
# Model config + structured V_theta + model build
# ---------------------------------------------------------------------------
def build_fock_config(cfg: FockSetupConfig, d, L, n_registers, logfreq_file):
    """Return a FockMultiXiPARFConfig for one architecture tier.

    Mirrors ``make_config`` in the production notebook.
    """
    from model_fock_parf_multixi import FockMultiXiPARFConfig
    return FockMultiXiPARFConfig(
        vocab_size=cfg.vocab_size, d=d, max_len=cfg.max_len,
        L=L, v_hidden=cfg.v_hidden, v_depth=cfg.v_depth, dt=cfg.dt,
        mass_mode="logfreq",
        logfreq_path=str(logfreq_file),
        logfreq_init_alpha=0.1,
        init_gamma=cfg.init_gamma,
        fixed_gamma=cfg.fixed_gamma,
        causal_force=True,
        ln_after_step=True,
        xi_channels=cfg.xi_channels,
        xi_alpha_inits=cfg.xi_alpha_inits,
        xi_learnable=True,
        xi_alpha_init_mode="explicit",
        v_phi_kind=cfg.v_phi_kind,
        v_phi_d_type=cfg.v_phi_d_type,
        v_phi_d_angle=cfg.v_phi_d_angle,
        v_phi_eps=0.1,
        v_phi_phi_hidden=128,
        v_phi_theta_hidden=128,
        v_phi_mlp_hidden=cfg.v_phi_mlp_hidden,
        top_k=cfg.top_k,
        v_phi_n_heads=cfg.v_phi_n_heads,
        use_output_bias=cfg.use_output_bias,
        tie_embeddings=cfg.tie_embeddings,
        score_head_hidden=32,
        gumbel_tau_init=1.0,
        gumbel_tau_min=0.3,
        gumbel_noise=True,
        use_gathered_v_phi=True,
        use_layer_checkpoint=True,
        ln_before_distance=True,
        per_layer_v_phi_scale=True,
        fock_version="v2",
        n_registers=n_registers,
        register_salience_decay=0.5,
        register_salience_threshold=0.005,
        creation_gate_hidden=64,
        stack_discipline=True,
        d_k=64,
        tau_create_init=8.0,
        reverse_channel=cfg.reverse_channel,
        reverse_channel_stable=cfg.reverse_channel_stable,
        reverse_channel_pre_ln=cfg.reverse_channel_pre_ln,
        reverse_channel_soft_norm=cfg.reverse_channel_soft_norm,
        reverse_channel_warmup_steps=cfg.reverse_channel_warmup_steps,
        reverse_channel_per_layer=cfg.reverse_channel_per_layer,
        per_register_tau=True,
        per_register_keys=True,
        ortho_register_init=True,
        register_repulsion=cfg.register_repulsion,
        register_repulsion_coeff=cfg.register_repulsion_coeff,
        register_repulsion_kind=cfg.register_repulsion_kind,
    )


def build_structured_vtheta(model, cfg: FockSetupConfig, d, device,
                            train_ids=None, verbose=True):
    """Swap ``model.V_theta`` with the configured structured variant.

    Mirrors ``build_structured_vtheta`` in the production notebook.  ``train_ids``
    is required only for the 'sarf' variant (PMI anchor selection).
    """
    import numpy as np
    import torch

    variant = cfg.v_theta_variant
    if variant == "mlp":
        return
    xi_d = cfg.xi_channels * d
    from model_gaussian_vtheta import (
        MixtureGaussianVTheta, SARFGaussianVTheta,
        GaussianVThetaMultiXiAdapter, MultiContextGaussianVTheta,
        DepthConditionedMultiContextGaussianVTheta, install_depth_routing,
    )

    if variant == "gaussian":
        _init_log_prec = -math.log(d)
        _prec_max = 2.0 / d

        if cfg.v_theta_n_heads > 1 and cfg.v_theta_depth_condition:
            n_layers = model.cfg.L
            model.V_theta = DepthConditionedMultiContextGaussianVTheta(
                d=d, K=cfg.v_theta_wells_per_head, n_ctx=cfg.v_theta_n_heads,
                n_layers=n_layers,
                w_scale=cfg.w_scale,
                init_log_precision=_init_log_prec,
                precision_max=_prec_max,
                code_init_std=cfg.v_theta_depth_code_init_std,
            ).to(device)
            install_depth_routing(model)
            if verbose:
                _n_code = model.V_theta.depth_code.numel()
                print(f"V_theta -> DepthConditionedMultiContextGaussian("
                      f"{cfg.v_theta_n_heads} heads x {cfg.v_theta_wells_per_head} wells, "
                      f"L={n_layers} layers, code_params={_n_code:,})")
                print("  depth routing installed on _fock_layer_step")
        elif cfg.v_theta_n_heads > 1:
            model.V_theta = MultiContextGaussianVTheta(
                d=d, K=cfg.v_theta_wells_per_head, n_ctx=cfg.v_theta_n_heads,
                w_scale=cfg.w_scale,
                init_log_precision=_init_log_prec,
                precision_max=_prec_max,
            ).to(device)
            if verbose:
                print(f"V_theta -> MultiContextGaussian({cfg.v_theta_n_heads} heads x "
                      f"{cfg.v_theta_wells_per_head} wells)")
        else:
            inner = MixtureGaussianVTheta(
                d=d, K=cfg.v_theta_wells_per_head, w_scale=cfg.w_scale, xi_d=xi_d,
                init_log_precision=_init_log_prec,
                precision_max=_prec_max,
            )
            model.V_theta = GaussianVThetaMultiXiAdapter(
                inner, K=cfg.xi_channels, d=d,
            ).to(device)
            if verbose:
                print(f"V_theta -> ConcatGaussian(K={cfg.v_theta_wells_per_head})")

    elif variant == "sarf":
        if train_ids is None:
            raise ValueError("build_structured_vtheta(variant='sarf') needs train_ids")
        WINDOW = 5
        TOP_V = 8192
        token_counts = np.bincount(train_ids.astype(np.int64), minlength=cfg.vocab_size)
        top_v_ids = np.argsort(-token_counts)[:TOP_V]
        id_to_local = np.full(cfg.vocab_size, -1, dtype=np.int64)
        id_to_local[top_v_ids] = np.arange(TOP_V)

        cooc = np.zeros((TOP_V, TOP_V), dtype=np.float64)
        local_ids = id_to_local[train_ids.astype(np.int64)]
        for offset in range(1, WINDOW + 1):
            a = local_ids[:-offset]
            b = local_ids[offset:]
            valid = (a >= 0) & (b >= 0)
            np.add.at(cooc, (a[valid], b[valid]), 1.0)
        cooc = cooc + cooc.T

        row_sums = cooc.sum(axis=1, keepdims=True)
        total = cooc.sum()
        expected = row_sums * row_sums.T / total
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log(cooc / np.maximum(expected, 1e-12))
        pmi = np.nan_to_num(pmi, nan=0.0, posinf=0.0, neginf=-20.0)

        np.fill_diagonal(pmi, -np.inf)
        pmi_peaks = pmi.max(axis=1)
        anchor_local_ids = np.argsort(-pmi_peaks)[:cfg.sarf_n_anchors]
        anchor_token_ids = top_v_ids[anchor_local_ids]
        anchor_positions = model.E.weight.data[anchor_token_ids].detach().clone()

        _log_sigma_max = 0.5 * math.log(d) + 1.0
        _init_log_sigma = math.log(d) / 2.0

        inner = SARFGaussianVTheta(
            d=d, anchor_positions=anchor_positions, xi_d=xi_d, w_scale=cfg.w_scale,
            init_log_sigma=_init_log_sigma,
            log_sigma_max=_log_sigma_max,
        )
        model.V_theta = GaussianVThetaMultiXiAdapter(inner, K=cfg.xi_channels, d=d).to(device)

        with torch.no_grad():
            a = model.V_theta.inner.anchors
            a = (a - a.mean(dim=-1, keepdim=True)) / (a.std(dim=-1, keepdim=True) + 1e-5)
            model.V_theta.inner.anchors.copy_(a)
        if verbose:
            print(f"V_theta -> SARF Gaussian(N_S={cfg.sarf_n_anchors})")

    elif variant == "sq3":
        from model_structured_vtheta import MixtureQuadraticVTheta
        from model_structured_vtheta_multixi import StructuredVThetaMultiXiAdapter

        inner = MixtureQuadraticVTheta(
            d=d, K=cfg.v_theta_wells_per_head, tau=cfg.sq3_tau, init_a_bias=0.0,
            xi_d=xi_d,
        )
        if cfg.sq3_curv_max is not None:
            _orig_components = inner._components

            def _clamped_components(xi, _fn=_orig_components, _cmax=cfg.sq3_curv_max):
                mu, a, log_pi = _fn(xi)
                return mu, a.clamp(max=_cmax), log_pi
            inner._components = _clamped_components

        model.V_theta = StructuredVThetaMultiXiAdapter(
            inner, K=cfg.xi_channels, d=d,
        ).to(device)
        if verbose:
            print(f"V_theta -> SQ3 Mixture(K={cfg.v_theta_wells_per_head}, tau={cfg.sq3_tau})")

    else:
        raise ValueError(f"Unknown V_theta variant: {variant}")


def build_fock_model(cfg: FockSetupConfig, device, get_batch, train_ids,
                     logfreq_file, oom_probe=True, verbose=True):
    """Build the Fock-PARFLM model, trying architecture tiers in order.

    Mirrors the tier-probe loop in the production notebook.  Returns
    ``(model, model_cfg)``.  Does NOT install the O-step or the output bias
    (the caller does those, so the sweep can vary the O-step per grid point).
    """
    import gc
    import numpy as np
    import torch
    from model_fock_parf_multixi import FockMultiXiPARFLM

    model = None
    model_cfg = None
    for d, L, M in cfg.arch_tiers:
        try:
            fcfg = build_fock_config(cfg, d, L, M, logfreq_file)
            mdl = FockMultiXiPARFLM(fcfg).to(device)
            n_v_theta_mlp = sum(p.numel() for p in mdl.V_theta.parameters())
            build_structured_vtheta(mdl, cfg, d, device, train_ids=train_ids,
                                    verbose=verbose)
            n = mdl.num_params()
            n_v_theta = sum(p.numel() for p in mdl.V_theta.parameters())
            if verbose:
                print(f"Trying d={d} L={L} M={M} -> {n:,} params "
                      f"(V_theta {n_v_theta_mlp:,} MLP -> {n_v_theta:,})")
            if oom_probe and device == "cuda":
                _rng = np.random.default_rng(42)
                _xb, _yb = get_batch(train_ids, 2, cfg.block_size, _rng)
                _x = torch.from_numpy(_xb).to(device)
                _y = torch.from_numpy(_yb).to(device)
                _, _loss = mdl(_x, _y)
                _loss.backward()
                mdl.zero_grad(set_to_none=True)
                del _x, _y, _xb, _yb, _loss
                torch.cuda.empty_cache()
                if verbose:
                    print("OOM probe passed (batch=2)")
            model = mdl
            model_cfg = fcfg
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if verbose:
                    print(f"  OOM at d={d} L={L} M={M} — trying next tier ...")
                del mdl
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                continue
            raise
    if model is None:
        raise RuntimeError("All architecture tiers OOMed.")
    return model, model_cfg


def init_output_bias(model, train_ids, vocab_size, verbose=True):
    """Initialise the output bias from the log-unigram frequency."""
    import numpy as np
    counts = np.bincount(train_ids.astype(np.int64), minlength=vocab_size)
    model.init_output_bias_from_logfreq(counts)
    if verbose:
        print(f"Output bias <- log-unigram-freq  "
              f"(b range [{model.out_bias.min().item():.2f}, "
              f"{model.out_bias.max().item():.2f}])")


def auto_batch_size(model, get_batch, train_ids, device, block_size,
                    grad_accum, default=4, verbose=True):
    """VRAM-aware batch probe (mirrors the production notebook)."""
    import numpy as np
    import torch
    bs_out = default
    if device == "cuda":
        _vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        _probe_sizes = [16, 12, 8, 6, 4] if _vram_gb >= 70 else [12, 8, 6, 4]
        for bs in _probe_sizes:
            try:
                _rng = np.random.default_rng(42)
                _xb, _yb = get_batch(train_ids, bs, block_size, _rng)
                _x = torch.from_numpy(_xb).to(device)
                _y = torch.from_numpy(_yb).to(device)
                _, _loss = model(_x, _y)
                _loss.backward()
                model.zero_grad(set_to_none=True)
                del _x, _y, _xb, _yb, _loss
                torch.cuda.empty_cache()
                bs_out = bs
                break
            except RuntimeError:
                torch.cuda.empty_cache()
                continue
    if verbose:
        print(f"Auto batch: {bs_out} x accum={grad_accum} (eff={bs_out*grad_accum})")
    return bs_out


# ---------------------------------------------------------------------------
# LR schedule + evaluation
# ---------------------------------------------------------------------------
def make_lr_schedule(schedule, total_steps, peak_lr, warmup_frac=0.05,
                     stable_frac=0.60, lr_floor=None, warmup_steps=4000):
    """Return an ``lr(step)`` callable for 'wsd' or 'cosine'."""
    if lr_floor is None:
        lr_floor = peak_lr * 0.05

    def lr_at(step):
        if schedule == "wsd":
            warmup_end = int(warmup_frac * total_steps)
            stable_end = int((warmup_frac + stable_frac) * total_steps)
            if step < warmup_end:
                return peak_lr * (step + 1) / max(warmup_end, 1)
            elif step < stable_end:
                return peak_lr
            else:
                decay_steps = total_steps - stable_end
                progress = (step - stable_end) / max(decay_steps, 1)
                cos_decay = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
                return lr_floor + (peak_lr - lr_floor) * cos_decay
        else:
            if step < warmup_steps:
                return peak_lr * (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return peak_lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return lr_at


def make_evaluate(model, val_ids, get_batch, device, block_size, batch_size,
                  eval_iters, rng):
    """Return an ``evaluate()`` -> (mean_loss, mean_entropy_in_nats).

    Entropy is the basin-coarseness proxy of Lessons_from_AlphaFold.md: the
    Verlet 'commits harder' collapse shows up as LOW entropy, which the O-step
    thermostat is meant to relax.  The forward runs under enable_grad (the
    model needs grad to compute -grad V); logits are detached before the
    entropy reduction so no graph is retained across iterations.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    def evaluate():
        model.eval()
        losses, entropies = [], []
        for _ in range(eval_iters):
            xb, yb = get_batch(val_ids, batch_size, block_size, rng)
            x = torch.from_numpy(xb).to(device)
            y = torch.from_numpy(yb).to(device)
            with torch.enable_grad():
                out = model(x, y)
            logits = out[0].detach().float()
            losses.append(float(out[1].item()))
            del out
            logp = F.log_softmax(logits, dim=-1)
            ent = -(logp.exp() * logp).sum(dim=-1).mean()
            entropies.append(float(ent.item()))
            del logits, logp, ent
        model.train()
        return float(np.mean(losses)), float(np.mean(entropies))

    return evaluate


# ---------------------------------------------------------------------------
# Short proxy training loop (the sweep's reusable core)
# ---------------------------------------------------------------------------
def train_proxy(model, cfg: FockSetupConfig, model_cfg, optim, get_batch,
                train_ids, val_ids, device, *, steps, batch_size, grad_accum,
                peak_lr, weight_decay, eval_interval, eval_iters, seed=0,
                lr_schedule="wsd", warmup_frac=0.05, stable_frac=0.60,
                lr_floor=None, grad_clip=1.0, grad_clip_vphi=0.3,
                log_interval=200, band_last_k=5, verbose=True):
    """Run a short training run and return a metrics history.

    A stripped-but-faithful version of the production training loop: WSD LR
    schedule, gradient accumulation, V_reg, register repulsion, tight V_phi
    clip + global clip, non-finite skip, structured clamp_params after each
    step, and periodic (loss, entropy) eval.  It deliberately omits the
    Drive-checkpointing / resume / watchdog machinery that only the long run
    needs.

    Returns a dict with ``history`` (list of eval records) and ``summary``
    (best_ppl, band_ppl = median of the last ``band_last_k`` evals, final
    entropy, and a grad-stability fraction).
    """
    import time
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    vocab = model_cfg.vocab_size
    is_sq3 = (cfg.v_theta_variant == "sq3")
    lr_at = make_lr_schedule(lr_schedule, steps, peak_lr, warmup_frac,
                             stable_frac, lr_floor)
    evaluate = make_evaluate(model, val_ids, get_batch, device, cfg.block_size,
                             batch_size, eval_iters, rng)

    def forward_with_vreg(x, targets, lambda_v):
        h0 = model._embed(x)
        h_L, _ = model._stack_forward(h0, x, return_trajectory=False)
        logits = model.compute_logits(h_L)
        loss_ntp = F.cross_entropy(
            logits.reshape(-1, vocab), targets.reshape(-1),
        )
        v_reg_value = torch.tensor(0.0, device=x.device)
        if lambda_v > 0:
            xis = model.xi_module(h_L.detach())
            V_vals = model.V_theta(xis, h_L)
            if is_sq3:
                v_reg_value = torch.log1p(V_vals ** 2).mean()
            else:
                v_reg_value = (V_vals ** 2).mean()
            if cfg.bg_quad_eps > 0:
                bg = cfg.bg_quad_eps * (h_L ** 2).sum(dim=-1, keepdim=True).mean()
                loss = loss_ntp + lambda_v * v_reg_value + bg
            else:
                loss = loss_ntp + lambda_v * v_reg_value
        else:
            loss = loss_ntp
        return loss, loss_ntp, v_reg_value

    history = []
    best_ppl = float("inf")
    n_steps_done = 0
    n_finite = 0
    t0 = time.time()
    model.train()
    for step in range(steps):
        lr_now = lr_at(step)
        for g in optim.param_groups:
            g["lr"] = lr_now
        optim.zero_grad(set_to_none=True)
        accum_ntp = 0.0
        for _acc in range(grad_accum):
            xb, yb = get_batch(train_ids, batch_size, cfg.block_size, rng)
            x = torch.from_numpy(xb).to(device)
            y = torch.from_numpy(yb).to(device)
            loss, loss_ntp, v_reg = forward_with_vreg(x, y, cfg.lambda_v)
            if cfg.register_repulsion:
                loss = loss + model.pop_repulsion_loss()
            (loss / grad_accum).backward()
            accum_ntp += loss_ntp.item() / grad_accum

        if model.V_phi is not None:
            nn.utils.clip_grad_norm_(model.V_phi.parameters(), grad_clip_vphi)
        grad_norm = nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], grad_clip,
        )
        if torch.isfinite(grad_norm) and math.isfinite(accum_ntp):
            optim.step()
            n_finite += 1
            if cfg.is_structured and cfg.v_theta_n_heads == 1:
                _inner = getattr(model.V_theta, "inner", None)
                if hasattr(_inner, "clamp_params"):
                    model.V_theta.inner.clamp_params()
            elif cfg.is_structured and cfg.v_theta_n_heads > 1:
                for bank in model.V_theta.banks:
                    if hasattr(bank, "clamp_params"):
                        bank.clamp_params()
        else:
            optim.zero_grad(set_to_none=True)
        n_steps_done += 1

        if verbose and (step + 1) % log_interval == 0:
            print(f"    step {step+1:6d}/{steps}  ntp={accum_ntp:.4f}  "
                  f"lr={lr_now:.2e}  grad={float(grad_norm):.2f}  "
                  f"({time.time()-t0:.0f}s)")

        if (step + 1) % eval_interval == 0 or (step + 1) == steps:
            val_loss, val_ent = evaluate()
            val_ppl = math.exp(val_loss)
            best_ppl = min(best_ppl, val_ppl)
            history.append({
                "step": step + 1, "val_loss": val_loss, "val_ppl": val_ppl,
                "val_entropy": val_ent, "grad_norm": float(grad_norm),
            })
            if verbose:
                print(f"    >>> eval {step+1:6d}  ppl={val_ppl:.2f}  "
                      f"best={best_ppl:.2f}  ent={val_ent:.3f}")

    ppls = [h["val_ppl"] for h in history]
    ents = [h["val_entropy"] for h in history]
    band = ppls[-band_last_k:] if len(ppls) >= 1 else [float("inf")]
    summary = {
        "best_ppl": best_ppl,
        "band_ppl": float(np.median(band)),
        "final_ppl": ppls[-1] if ppls else float("inf"),
        "final_entropy": ents[-1] if ents else float("nan"),
        "grad_finite_frac": n_finite / max(n_steps_done, 1),
        "n_evals": len(history),
    }
    return {"history": history, "summary": summary}


# ---------------------------------------------------------------------------
# Self-test (light; no torch model build required)
# ---------------------------------------------------------------------------
def _self_test():
    # 1) xi presets
    a, n = resolve_xi_preset("5long")
    assert n == 5 and a[0] == 0.50, (a, n)

    # 2) variant tag reproduces the production notebook's default tag.
    cfg = FockSetupConfig()
    expected = ("xi5long_topk16_dt32da16_mh4_dcvt5x8_ob_untied_wsd_"
                "e5a_rep0.05_ostep_T1")
    assert cfg.variant_tag() == expected, cfg.variant_tag()

    # 3) lr schedule shape
    lr = make_lr_schedule("wsd", 1000, 3e-4)
    assert lr(0) < lr(50) <= 3e-4 + 1e-12          # warmup rising
    assert abs(lr(500) - 3e-4) < 1e-12             # stable at peak
    assert lr(999) < 3e-4                           # decaying

    # 4) install_ostep / set_ostep on a dummy module (no real model needed).
    import torch
    import torch.nn as nn

    class _Cfg:
        dt = 1.0

    class _Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.cfg = _Cfg()
            self.gamma = torch.tensor(0.30)
            self.calls = 0

        def _fock_layer_step(self, h, h_prev, r, salience, m_b, gamma, dt,
                             layer_idx):
            self.calls += 1
            # trivial "drift": advance h by a fixed velocity encoded in h_prev
            h_new = h + (h - h_prev)
            return h_new, h_new, r, salience

    m = _Dummy()
    install_ostep(m, gamma=None, T=1.0, noise_train=True, noise_eval=False,
                  verbose=False)
    assert m._ostep_installed
    assert abs(m._ostep_cfg["c1"] - math.exp(-float(m.gamma))) < 1e-6
    B, T, d = 2, 3, 4
    h = torch.zeros(B, T, d)
    h_prev = -torch.ones(B, T, d)          # implies v = (h_new - h)/dt
    m_b = torch.ones(B, T, 1)
    m.train()
    out = m._fock_layer_step(h, h_prev, None, None, m_b, None, 1.0, layer_idx=0)
    assert out[0].shape == (B, T, d)
    # retune via set_ostep: gamma -> 0 makes c1 = 1 (no OU decay)
    set_ostep(m, gamma=0.0, T=2.0)
    assert abs(m._ostep_cfg["c1"] - 1.0) < 1e-12 and m._ostep_cfg["T"] == 2.0
    # eval + noise_eval False -> deterministic
    m.eval()
    o1 = m._fock_layer_step(h, h_prev, None, None, m_b, None, 1.0, layer_idx=0)[0]
    o2 = m._fock_layer_step(h, h_prev, None, None, m_b, None, 1.0, layer_idx=0)[0]
    assert torch.allclose(o1, o2)
    print("fock_ostep_setup self-test: OK")


if __name__ == "__main__":
    _self_test()
