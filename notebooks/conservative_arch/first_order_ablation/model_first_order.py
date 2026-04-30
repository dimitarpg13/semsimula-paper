"""
SPLM-1: first-order gradient-flow ablation of the SPLM em_ln architecture.

Architectural change vs. ScalarPotentialLMSARFMassLN
----------------------------------------------------
The damped semi-implicit Euler step

    v_{l+1} = (v_l + dt * f / m) / (1 + dt * gamma)
    h_{l+1} = h_l + dt * v_{l+1}

is replaced by the first-order gradient-flow step (eq. 6 of
companion_notes/Replacing_The_Conservative_Mechanism_of_SPLM_with_First_Order.md):

    h_{l+1} = h_l - dt * grad_V / m

There is no velocity buffer, no damping coefficient, and no inertial dynamics.
Everything else — V_theta, the causal context pool xi_t, the per-token
semantic mass, LayerNorm-after-step, the tied-embedding readout, the loss —
is identical to the SPLM em_ln baseline at the corresponding gamma.

The model intentionally retains the gamma parameter in its state dict (as a
non-trainable buffer set to 0.0) so that downstream diagnostics (Markov-order
extractor, energy-drift report, attractor extractor) can load this checkpoint
without code changes. The integrate() method does not consult gamma.

Pre-registered prediction
-------------------------
At matched architecture, matched data, matched training budget, and matched
multi-seed protocol, this model should reach a strictly worse validation
perplexity than its second-order counterpart at gamma* = 0.30 (the E5 minimum,
val ppl 87.06). See companion_notes/SPLM-1_ablation_pre-registered_protocol.md for the
locked decision rule and effect-size thresholds.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = Path(__file__).parent
_PARENT_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_PARENT_DIR))
sys.path.insert(0, str(_PARENT_DIR / "energetic_minima"))
sys.path.insert(0, str(_PARENT_DIR / "sarf_mass_variant"))

from sarf_mass_variant.model_sarf_mass import causal_cumulative_mean  # noqa: E402
from energetic_minima.model_ln import (  # noqa: E402
    SPLMSARFMassLNConfig,
    ScalarPotentialLMSARFMassLN,
)


@dataclass
class SPLMFirstOrderConfig(SPLMSARFMassLNConfig):
    """Config for the first-order ablation. Same fields as parent.

    The fixed_gamma field is ignored at integration time but kept here for
    state-dict compatibility with the LN second-order baseline.
    """
    pass


class ScalarPotentialLMFirstOrder(ScalarPotentialLMSARFMassLN):
    """SPLM em_ln stripped of the velocity buffer and damping term.

    Layer update:
        h_{l+1} = LN( h_l - dt * grad_V(xi_l, h_l) / m )
    """

    def __init__(self, cfg: SPLMFirstOrderConfig):
        super().__init__(cfg)
        self.cfg: SPLMFirstOrderConfig = cfg
        self._gamma_value = 0.0
        with torch.no_grad():
            self.raw_gamma.copy_(torch.tensor(0.0))
        self.raw_gamma.requires_grad_(False)

    def integrate(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        return_trajectory: bool = False,
        return_xi_trajectory: bool = False,
    ) -> Tuple[torch.Tensor,
               Optional[List[torch.Tensor]],
               Optional[List[torch.Tensor]]]:
        cfg = self.cfg
        h = self._project(emb) if cfg.ln_after_step else emb
        dt = cfg.dt

        m = self.compute_mass(x, emb)
        m_b = m

        traj_h: Optional[List[torch.Tensor]] = None
        traj_xi: Optional[List[torch.Tensor]] = None
        if return_trajectory:
            traj_h = [h.detach().cpu()]
        if return_xi_trajectory:
            traj_xi = []

        for _ in range(cfg.L):
            xi_now = causal_cumulative_mean(h)
            if return_xi_trajectory:
                assert traj_xi is not None
                traj_xi.append(xi_now.detach().cpu())

            h_in = h
            if not h_in.requires_grad:
                h_in = h_in.requires_grad_(True)
            V = self.V_theta(xi_now, h_in).sum()
            grad_V, = torch.autograd.grad(
                V, h_in,
                create_graph=self.training,
                retain_graph=True,
            )
            f = -grad_V
            h_new = h_in + dt * f / m_b
            if cfg.ln_after_step:
                h_new = self._project(h_new)
            h = h_new
            if return_trajectory:
                assert traj_h is not None
                traj_h.append(h.detach().cpu())

        return h, traj_h, traj_xi


def smoke_test():
    torch.manual_seed(0)
    V = 257
    cfg = SPLMFirstOrderConfig(
        vocab_size=V, d=16, max_len=32, v_hidden=32, v_depth=2, L=4,
        mass_mode="global",
    )
    net = ScalarPotentialLMFirstOrder(cfg)
    x = torch.randint(0, V, (2, 16))
    y = torch.randint(0, V, (2, 16))
    logits, loss = net(x, y)
    loss.backward()

    with torch.enable_grad():
        _, _, traj_h, _ = net(x, y, return_trajectory=True,
                              return_xi_trajectory=True)
    last = traj_h[-1]
    per_tok_norm = last.reshape(-1, cfg.d).pow(2).mean(dim=-1).sqrt()
    print(f"[FO smoke]  params={net.num_params():,}  loss={loss.item():.4f}  "
          f"gamma_value={net._gamma_value}")
    print(f"[FO smoke]  h_L per-token RMS (should be ~1): "
          f"mean={per_tok_norm.mean().item():.4f}  "
          f"std={per_tok_norm.std().item():.4e}")

    has_v_buffer = any(
        n.endswith("v") or "velocity" in n.lower()
        for n, _ in net.named_buffers()
    )
    print(f"[FO smoke]  velocity buffer in state_dict: {has_v_buffer} "
          f"(should be False — first-order has no v)")


if __name__ == "__main__":
    smoke_test()
