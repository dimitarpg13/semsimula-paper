"""
SPLM variant (i) of the energetic-minima study: LayerNorm-after-step.

Architectural idea
------------------
The only change from the SARF-faithful SPLM with per-token semantic mass
(the `sarf_mass_variant` logfreq variant, val ppl 160.55) is that after
each semi-implicit damped integration step

  v_{l+1} = (v_l + dt * f_l / m) / (1 + dt * gamma)
  h_{l+1} = h_l + dt * v_{l+1}

we project h_{l+1} back to the unit-LayerNorm shell

  h_{l+1} <- (h_{l+1} - mu_{l+1}) / (sigma_{l+1} + eps),     mu, sigma per-token.

i.e. the layer-normalisation that all attention transformers apply.
V_theta is otherwise unchanged (a free MLP head, no structural bound).

Rationale
---------
Compactness of S^{d-1} (up to the mean-shift) delivers a finite minimum of
any continuous V_theta on the shell by the extreme-value theorem.  This
is the cheapest way to buy a finite minimum of V without changing V's
functional form or its loss-side gauge.

Empirically this should either (a) leave val perplexity essentially
unchanged while producing a narrower V_theta range and crisper basins,
or (b) damage expressivity (by over-constraining h).

Checkpoints tag themselves with variant="sarf_mass_ln" so the
attractor-extraction dispatcher can load them correctly.
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

from sarf_mass_variant.model_sarf_mass import (  # noqa: E402
    SPLMSARFMassConfig,
    ScalarPotentialLMSARFMass,
    causal_cumulative_mean,
)


@dataclass
class SPLMSARFMassLNConfig(SPLMSARFMassConfig):
    """Extends SPLMSARFMassConfig with LayerNorm-after-step switches."""
    ln_eps: float = 1e-5
    ln_after_step: bool = True
    ln_affine: bool = False


class ScalarPotentialLMSARFMassLN(ScalarPotentialLMSARFMass):
    """SARF-faithful SPLM with mandatory LayerNorm after every damped step."""

    def __init__(self, cfg: SPLMSARFMassLNConfig):
        super().__init__(cfg)
        self.cfg: SPLMSARFMassLNConfig = cfg
        if cfg.ln_affine:
            self.post_ln = nn.LayerNorm(cfg.d, eps=cfg.ln_eps)
        else:
            self.post_ln = None

    def _project(self, h: torch.Tensor) -> torch.Tensor:
        if self.post_ln is not None:
            return self.post_ln(h)
        return F.layer_norm(h, (self.cfg.d,), eps=self.cfg.ln_eps)

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
        v = torch.zeros_like(h)
        gamma, dt = self.gamma, cfg.dt

        m = self.compute_mass(x, emb)
        m_b = m

        traj_h: Optional[List[torch.Tensor]] = None
        traj_xi: Optional[List[torch.Tensor]] = None
        if return_trajectory:
            traj_h = [h.detach().cpu()]
        if return_xi_trajectory:
            traj_xi = []

        for _ in range(cfg.L):
            xi_input = h.detach() if cfg.causal_force else h
            xi_now = causal_cumulative_mean(xi_input)
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
            v = (v + dt * f / m_b) / (1.0 + dt * gamma)
            h_new = h_in + dt * v
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
    cfg = SPLMSARFMassLNConfig(
        vocab_size=V, d=16, max_len=32, v_hidden=32, v_depth=2, L=4,
        mass_mode="global",
    )
    net = ScalarPotentialLMSARFMassLN(cfg)
    x = torch.randint(0, V, (2, 16))
    y = torch.randint(0, V, (2, 16))
    logits, loss = net(x, y)
    loss.backward()

    with torch.enable_grad():
        _, _, traj_h, _ = net(x, y, return_trajectory=True,
                              return_xi_trajectory=True)
    last = traj_h[-1]
    per_tok_norm = last.reshape(-1, cfg.d).pow(2).mean(dim=-1).sqrt()
    print(f"[LN smoke]  params={net.num_params():,}  loss={loss.item():.4f}")
    print(f"[LN smoke]  h_L per-token RMS (should be ~1): "
          f"mean={per_tok_norm.mean().item():.4f}  "
          f"std={per_tok_norm.std().item():.4e}")


if __name__ == "__main__":
    smoke_test()
