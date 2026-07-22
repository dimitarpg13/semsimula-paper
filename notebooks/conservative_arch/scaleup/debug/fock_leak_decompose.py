#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fock_leak_decompose.py — attribute the reverse-channel acausality.

Candidate channels (all weights-only; token values entering any position t
are always restricted to tokens <= t):

  C1  salience/active-mask: alpha_max is computed from the FULL-sequence
      creation softmax; it updates salience, which gates the active mask and
      the next layer's register blend.
  C2  cross-layer register content: r_new_content (full-sequence readout)
      is blended into the global register state r; the NEXT layer's creation
      gate computes Q = f(r), so attention scores (and hence the causal
      cumulative-softmax weights at every position t) shift.

Decomposition runs (eval mode, float64, reverse gate fully open):

  D1  baseline L=4                          -> total leak
  D2  L=4, _active_mask forced all-True     -> removes C1's mask component
  D3  L=4, salience frozen at 1.0           -> removes C1 entirely (blend + mask)
  D4  L=1                                   -> removes C2 (no cross-layer r reuse)
  D5  L=1 + salience frozen                 -> should be exactly 0
"""

import math
import sys
import types
from pathlib import Path

import numpy as np
import torch

_THIS = Path(__file__).resolve()
_PARF = _THIS.parent.parent.parent / "parf"
sys.path.insert(0, str(_PARF))

from model_fock_parf_multixi import FockMultiXiPARFLM, FockMultiXiPARFConfig
from model_gaussian_vtheta import (
    DepthConditionedMultiContextGaussianVTheta,
    install_depth_routing,
)

VOCAB, D, T, M, XI, WELLS = 101, 32, 48, 8, 3, 4
LOGFREQ = Path("/tmp/probe_logfreq.npy")
np.save(LOGFREQ, np.full(VOCAB, 5.0, dtype=np.float32))


def build(L):
    cfg = FockMultiXiPARFConfig(
        vocab_size=VOCAB, d=D, max_len=64, L=L,
        v_hidden=64, v_depth=3, dt=1.0,
        mass_mode="logfreq", logfreq_path=str(LOGFREQ),
        logfreq_init_alpha=0.1,
        init_gamma=1.0, fixed_gamma=0.30,
        causal_force=True, ln_after_step=True,
        xi_channels=XI, xi_alpha_inits=[0.5, 0.9, 0.99],
        xi_learnable=True, xi_alpha_init_mode="explicit",
        v_phi_kind="structural_competitive",
        v_phi_d_type=8, v_phi_d_angle=4,
        v_phi_eps=0.1, v_phi_phi_hidden=16, v_phi_theta_hidden=16,
        v_phi_mlp_hidden=16, top_k=8, v_phi_n_heads=2,
        use_output_bias=True, tie_embeddings=False,
        score_head_hidden=8,
        gumbel_tau_init=1.0, gumbel_tau_min=0.3, gumbel_noise=True,
        use_gathered_v_phi=True, use_layer_checkpoint=False,
        ln_before_distance=True, per_layer_v_phi_scale=True,
        fock_version="v2", n_registers=M,
        register_salience_decay=0.5, register_salience_threshold=0.005,
        creation_gate_hidden=16, stack_discipline=True,
        d_k=16, tau_create_init=8.0,
        reverse_channel=True, reverse_channel_stable=True,
        reverse_channel_pre_ln=True, reverse_channel_soft_norm=True,
        reverse_channel_warmup_steps=4000, reverse_channel_per_layer=True,
        per_register_tau=True, per_register_keys=True,
        ortho_register_init=True, register_repulsion=False,
    )
    torch.manual_seed(1234)
    m = FockMultiXiPARFLM(cfg)
    m.V_theta = DepthConditionedMultiContextGaussianVTheta(
        d=D, K=WELLS, n_ctx=XI, n_layers=L, w_scale=1.0,
        init_log_precision=-math.log(D), precision_max=2.0 / D,
        code_init_std=0.02,
    )
    install_depth_routing(m)
    m.double()
    m.eval()
    with torch.no_grad():
        m.reverse_channel_scale.fill_(1.0)
        m.reverse_warmup_step.fill_(4000)
    return m


def force_all_active(m):
    def _all_active(self, salience):
        return torch.ones_like(salience, dtype=torch.bool)
    m._active_mask = types.MethodType(_all_active, m)


def freeze_salience(m, value=1.0):
    """Wrap _fock_layer_step so salience in/out is pinned to `value`.

    value=1.0 -> blend=1 -> r_new_content never enters r (kills C2a AND the
                 alpha_max-carried salience signal C1).
    value=0.5 -> constant blend (kills C1 only; r_new_content still flows).
    """
    orig = m._fock_layer_step
    def _pinned(h, h_prev, r, salience, m_b, gamma, dt, layer_idx):
        pinned = torch.full_like(salience, value)
        h_new, h_out, r_new, _sal = orig(h, h_prev, r, pinned, m_b, gamma,
                                         dt, layer_idx)
        return h_new, h_out, r_new, torch.full_like(_sal, value)
    m._fock_layer_step = _pinned


def leak(m, x1, x2, t_p):
    with torch.enable_grad():
        a = m(x1)[0].detach()
        b = m(x2)[0].detach()
    return (a[:, :t_p] - b[:, :t_p]).abs().max().item()


def main():
    t_p = T // 2
    rng = np.random.default_rng(7)
    x1 = torch.from_numpy(rng.integers(0, VOCAB, (2, T))).long()
    x2 = x1.clone()
    x2[:, t_p:] = torch.from_numpy(rng.integers(0, VOCAB, (2, T - t_p))).long()

    m = build(L=4)
    print(f"D1 L=4 baseline (rev ON)                    {leak(m, x1, x2, t_p):.3e}")

    m = build(L=4); force_all_active(m)
    print(f"D2 L=4 active mask forced all-True          {leak(m, x1, x2, t_p):.3e}")

    m = build(L=4); freeze_salience(m, 1.0); force_all_active(m)
    print(f"D3 L=4 salience pinned 1.0 + mask all-True  {leak(m, x1, x2, t_p):.3e}")

    m = build(L=4); freeze_salience(m, 0.5); force_all_active(m)
    print(f"D6 L=4 salience pinned 0.5 + mask all-True  {leak(m, x1, x2, t_p):.3e}")

    m = build(L=1)
    print(f"D4 L=1 (no cross-layer register reuse)      {leak(m, x1, x2, t_p):.3e}")

    m = build(L=1); freeze_salience(m, 1.0); force_all_active(m)
    print(f"D5 L=1 + salience pinned + mask all-True    {leak(m, x1, x2, t_p):.3e}")


if __name__ == "__main__":
    main()
