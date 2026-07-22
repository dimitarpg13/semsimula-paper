#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fock_causality_probe.py
=======================

Empirical causal-leak probe for Fock-PARFLM v2.1 (the architecture trained by
colab_fock_depthcond_vtheta_openwebtext_ext2.ipynb).

Method: build a small model with the SAME structural features as the training
config (fock_version='v2', gathered structural_competitive V_phi, multi-head
V_phi, depth-conditioned Gaussian V_theta with install_depth_routing,
per-register keys/tau, stable reverse channel with pre-LN + soft-norm,
logfreq mass, stack discipline), run it in float64 on CPU, and measure:

    max_t<t_p | logits(x)[t] - logits(x')[t] |

where x' differs from x ONLY at positions >= t_p. For a strictly causal model
this must be exactly 0 (float64, deterministic CPU kernels, eval mode: no
gumbel noise).

Tests
-----
  T0  determinism        same input twice                        -> delta == 0
  T1  rev channel OFF    perturb future, scale=0 (init)          -> delta == 0 ?
  T2  rev channel ON     perturb future, scale=1, warmup done    -> quantify
  T3  positive control   causal_force=False (detach removed)     -> delta > 0
  T4  past sensitivity   perturb PAST token                      -> delta > 0
  T5  train-mode STE     same as T2 but training path + seeded   -> quantify
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch

# make the parf package importable
_THIS = Path(__file__).resolve()
_PARF = _THIS.parent.parent.parent / "parf"
sys.path.insert(0, str(_PARF))

from model_fock_parf_multixi import FockMultiXiPARFLM, FockMultiXiPARFConfig
from model_gaussian_vtheta import (
    DepthConditionedMultiContextGaussianVTheta,
    install_depth_routing,
)

torch.manual_seed(0)
np.random.seed(0)

VOCAB = 101
D = 32
L = 4
T = 48
M = 8
XI = 3
WELLS = 4

# dummy logfreq file (uniform surprisal)
LOGFREQ = Path("/tmp/probe_logfreq.npy")
np.save(LOGFREQ, np.full(VOCAB, 5.0, dtype=np.float32))


def build(causal_force=True):
    cfg = FockMultiXiPARFConfig(
        vocab_size=VOCAB, d=D, max_len=64, L=L,
        v_hidden=64, v_depth=3, dt=1.0,
        mass_mode="logfreq", logfreq_path=str(LOGFREQ),
        logfreq_init_alpha=0.1,
        init_gamma=1.0, fixed_gamma=0.30,
        causal_force=causal_force,
        ln_after_step=True,
        xi_channels=XI, xi_alpha_inits=[0.5, 0.9, 0.99],
        xi_learnable=True, xi_alpha_init_mode="explicit",
        v_phi_kind="structural_competitive",
        v_phi_d_type=8, v_phi_d_angle=4,
        v_phi_eps=0.1, v_phi_phi_hidden=16, v_phi_theta_hidden=16,
        v_phi_mlp_hidden=16,
        top_k=8, v_phi_n_heads=2,
        use_output_bias=True, tie_embeddings=False,
        score_head_hidden=8,
        gumbel_tau_init=1.0, gumbel_tau_min=0.3, gumbel_noise=True,
        use_gathered_v_phi=True,
        use_layer_checkpoint=False,
        ln_before_distance=True,
        per_layer_v_phi_scale=True,
        fock_version="v2",
        n_registers=M,
        register_salience_decay=0.5,
        register_salience_threshold=0.005,
        creation_gate_hidden=16,
        stack_discipline=True,
        d_k=16, tau_create_init=8.0,
        reverse_channel=True,
        reverse_channel_stable=True,
        reverse_channel_pre_ln=True,
        reverse_channel_soft_norm=True,
        reverse_channel_warmup_steps=4000,
        reverse_channel_per_layer=True,
        per_register_tau=True, per_register_keys=True,
        ortho_register_init=True,
        register_repulsion=False,
    )
    torch.manual_seed(1234)  # identical weights across builds
    m = FockMultiXiPARFLM(cfg)
    m.V_theta = DepthConditionedMultiContextGaussianVTheta(
        d=D, K=WELLS, n_ctx=XI, n_layers=L,
        w_scale=1.0,
        init_log_precision=-math.log(D),
        precision_max=2.0 / D,
        code_init_std=0.02,
    )
    install_depth_routing(m)
    m.double()
    m.eval()
    return m


def logits_of(model, x, train_mode=False, seed=None):
    if train_mode:
        model.train()
    else:
        model.eval()
    if seed is not None:
        torch.manual_seed(seed)
    with torch.enable_grad():
        out = model(x)
    logits = out[0].detach()
    model.eval()
    return logits


def report(name, delta, expect_zero):
    status = "LEAK" if (delta > 0 and expect_zero) else "ok"
    if not expect_zero and delta > 0:
        status = "signal (expected)"
    if not expect_zero and delta == 0:
        status = "NO SIGNAL (unexpected)"
    print(f"  {name:<58s} max|dlogit| = {delta:.3e}   [{status}]")
    return delta


def main():
    t_p = T // 2
    rng = np.random.default_rng(7)
    x1 = torch.from_numpy(rng.integers(0, VOCAB, (2, T))).long()
    x2 = x1.clone()
    x2[:, t_p:] = torch.from_numpy(rng.integers(0, VOCAB, (2, T - t_p))).long()
    x3 = x1.clone()
    x3[:, 3] = (x3[:, 3] + 17) % VOCAB  # past perturbation

    print(f"probe: B=2 T={T} t_p={t_p} d={D} L={L} M={M} float64 cpu\n")

    # ---- baseline model, reverse gate at init (tanh(0)=0 -> OFF) ----
    m = build(causal_force=True)

    la = logits_of(m, x1)
    lb = logits_of(m, x1)
    report("T0 determinism (same input twice)",
           (la - lb).abs().max().item(), expect_zero=True)

    lc = logits_of(m, x2)
    report("T1 future perturb, reverse gate OFF (init)",
           (la[:, :t_p] - lc[:, :t_p]).abs().max().item(), expect_zero=True)

    # ---- open the reverse channel fully ----
    with torch.no_grad():
        m.reverse_channel_scale.fill_(1.0)          # tanh(1) ~ 0.76 per layer
        m.reverse_warmup_step.fill_(4000)           # warmup complete
    la2 = logits_of(m, x1)
    lc2 = logits_of(m, x2)
    d2 = report("T2 future perturb, reverse channel ON",
                (la2[:, :t_p] - lc2[:, :t_p]).abs().max().item(),
                expect_zero=True)
    rms = la2[:, :t_p].pow(2).mean().sqrt().item()
    print(f"      (logit rms = {rms:.3f}; relative = {d2 / rms:.3e})")
    # per-position profile of the delta
    prof = (la2[:, :t_p] - lc2[:, :t_p]).abs().amax(dim=(0, 2))
    nz = (prof > 0).sum().item()
    print(f"      (positions with nonzero delta: {nz}/{t_p}; "
          f"mean over positions = {prof.mean().item():.3e})")

    # ---- T3 positive control: remove the causal detach ----
    m_leaky = build(causal_force=False)
    with torch.no_grad():
        m_leaky.reverse_channel_scale.fill_(0.0)    # isolate the detach effect
    ld_a = logits_of(m_leaky, x1)
    ld_b = logits_of(m_leaky, x2)
    report("T3 positive control: causal_force=False (no detach)",
           (ld_a[:, :t_p] - ld_b[:, :t_p]).abs().max().item(),
           expect_zero=False)

    # ---- T4 sanity: perturbing the PAST must change later logits ----
    le = logits_of(m, x3)
    report("T4 past perturb (position 3) affects later logits",
           (la2[:, 4:t_p] - le[:, 4:t_p]).abs().max().item(),
           expect_zero=False)

    # ---- T5 train mode (STE gumbel path), seeded identically ----
    lt_a = logits_of(m, x1, train_mode=True, seed=99)
    lt_b = logits_of(m, x2, train_mode=True, seed=99)
    report("T5 future perturb, train mode + gumbel (seeded), rev ON",
           (lt_a[:, :t_p] - lt_b[:, :t_p]).abs().max().item(),
           expect_zero=True)

    print("\ndone.")


if __name__ == "__main__":
    main()
