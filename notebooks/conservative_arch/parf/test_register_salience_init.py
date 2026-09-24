"""``register_salience_init`` — the knob that makes a single layer's Fock
registers trainable.

WHY THIS EXISTS. The creation step is

    blend = salience
    r = blend * r + (1 - blend) * readout

and ``salience`` initialised to exactly 1.0 makes ``(1 - blend) == 0``, so at
layer 0 the creation gate's output is multiplied by zero. The only other route
out of the gate is ``alpha_max -> salience -> active``, and ``_active_mask`` is
a boolean comparison with no gradient; registers reach the loss solely through
the repulsion term, which never enters the forward logits. So **layer 0 cannot
train the creation gate at any depth**. At L>=2 later layers (whose salience
has decayed) train the shared module and the effect is invisible. At L=1 it is
the whole model, and the Fock mechanism is inert — which is how it was found:
a live L=1 run held ``sig_max`` at its initialisation value for hundreds of
steps while both L=2 arms differentiated within 50.

See companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md §6.1.
"""
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import model_fock_parf_multixi as MF  # noqa: E402
from model_aniso_gaussian_vtheta import (  # noqa: E402
    AnisotropicDepthConditionedGaussianVTheta,
    install_aniso_depth_routing,
)

V, D = 97, 32


def _build(L, dt, salience_init=1.0, decay=0.5, thresh=0.005, tmpdir=None):
    import numpy as np
    lf = Path(tmpdir or ".") / "_test_logfreq.npy"
    np.save(lf, np.random.RandomState(0).uniform(2, 12, V).astype("float32"))
    cfg = MF.FockMultiXiPARFConfig(
        vocab_size=V, d=D, max_len=16, L=L, dt=dt,
        integrator="baoab_cfc_lowrank", vtheta_analytic_force=True,
        lowrank_driver="gram", fixed_gamma=0.1, causal_force=True,
        logfreq_path=str(lf), xi_channels=5,
        xi_alpha_inits=(0.5, 0.75, 0.95, 0.99, 0.995),
        xi_content_route=True, xi_content_d_k=8,
        fock_version="v2", n_registers=32, creation_qk_norm=True,
        prefix_causal_registers=True,
        register_repulsion=True, register_repulsion_coeff=0.05,
        register_salience_decay=decay, register_salience_threshold=thresh,
        register_salience_init=salience_init,
    )
    torch.manual_seed(0)
    m = MF.FockMultiXiPARFLM(cfg)
    m.V_theta = AnisotropicDepthConditionedGaussianVTheta(
        d=D, K=8, n_ctx=5, n_layers=L, rank=4, w_scale=1.0,
        init_log_precision=-math.log(D), precision_max=2.0 / D,
        precision_lr_max=1.0, code_init_std=0.02, coupling="joint")
    install_aniso_depth_routing(m)
    return m


def _creation_grad(L, dt, salience_init, tmpdir):
    m = _build(L, dt, salience_init, tmpdir=tmpdir)
    m.train()
    x = torch.randint(0, V, (2, 16))
    _, loss = m(x, x)
    loss = loss + m.pop_repulsion_loss()
    loss.backward()
    return max(
        0.0 if p.grad is None else float(p.grad.abs().max())
        for n, p in m.named_parameters() if "creation_gate_qkv" in n
    )


def test_default_leaves_layer_zero_severed(tmp_path):
    """The default MUST reproduce the historical behaviour exactly.

    Three completed 32,500-step arms were trained under it; if this changes,
    their checkpoints no longer correspond to the code that made them.
    """
    assert _creation_grad(1, 8.0, 1.0, tmp_path) == 0.0
    assert _creation_grad(2, 4.0, 1.0, tmp_path) > 0.0


@pytest.mark.parametrize("s0", [0.9, 0.5, 0.25])
def test_below_one_makes_a_single_layer_trainable(s0, tmp_path):
    """Opening ``(1 - blend)`` is the whole fix, and it is monotone in s0."""
    assert _creation_grad(1, 8.0, s0, tmp_path) > 0.0


def test_gradient_grows_as_salience_init_falls(tmp_path):
    g = [_creation_grad(1, 8.0, s, tmp_path) for s in (0.9, 0.5, 0.25)]
    assert g[0] < g[1] < g[2], g


@pytest.mark.parametrize("bad", [0.005, 0.0, -0.1, 1.5])
def test_out_of_range_raises_at_construction(bad, tmp_path):
    """At or below the threshold every register starts inactive.

    Raised in ``__init__`` rather than on the first forward, so a bad config
    costs a second instead of a batch.
    """
    with pytest.raises(ValueError, match="register_salience_init"):
        _build(1, 8.0, bad, tmpdir=tmp_path)
