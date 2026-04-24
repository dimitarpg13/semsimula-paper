"""
SPLM variant (iii) of the energetic-minima study: Gaussian-mixture head.

Architectural idea
------------------
The only change from the SARF-faithful SPLM with per-token semantic mass
is that the free MLP scalar potential

  V_theta(xi, h) = MLP([xi; h])

is replaced by the framework-prescribed Gaussian-well form
(Section `sec:well` of the paper), summed over K learnable wells:

  V_theta(xi, h) = sum_k  m_k * upsilon_k^2 *
                          (1 - exp(-kappa_k^2 * || [xi;h] - c_k ||^2))

where c_k in R^{2d} is the learnable centre of the k-th well,
m_k > 0, upsilon_k > 0, kappa_k > 0 are learnable positive parameters
(parameterised via softplus).

This V is:
  * bounded below by 0 at each centre c_k (up to cross-well leakage),
  * bounded above by sum_k m_k * upsilon_k^2,
  * everywhere smooth, attractive, and structurally containing finite
    local minima.

Rationale
---------
This is the "honest test" of option B in the paper's rationale:
can a physics-prescribed, bounded-below V reach the SARF-faithful
val ppl 160.55 when used as a full SPLM head, when the same scalar
forms failed on real GPT-2 trajectories in Section
`subsec:five-negatives`?

Checkpoints tag themselves with variant="sarf_mass_gm" so the
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
)


@dataclass
class SPLMSARFMassGMConfig(SPLMSARFMassConfig):
    """SPLMSARFMassConfig with mixture-of-Gaussian-wells head."""
    gm_K: int = 64
    gm_init_kappa: float = 0.25
    gm_init_amp: float = 1.0
    gm_center_std: float = 0.5


def _raw_from_positive(y: float) -> float:
    import math
    return math.log(math.expm1(max(y, 1e-3)))


class GaussianMixturePotential(nn.Module):
    """Physics-prescribed scalar potential: sum of K Gaussian wells.

    The framework's well is V(x) = m * upsilon^2 * (1 - exp(-kappa^2 x^2)).
    Here we sum K such wells in the (xi, h) concatenation space (R^{2d}),
    so:

        V(xi, h) = sum_k  amp_k * (1 - exp(-kappa_k^2 * ||z - c_k||^2))

    with z = cat([xi, h], dim=-1) and amp_k = m_k * upsilon_k^2 a single
    learnable positive amplitude per well (the individual m_k and
    upsilon_k are not separately identifiable here, so we collapse them).
    """

    def __init__(self, d: int, K: int,
                 init_kappa: float = 0.25,
                 init_amp: float = 1.0,
                 center_std: float = 0.5):
        super().__init__()
        self.K = K
        self.input_dim = 2 * d
        self.centers = nn.Parameter(
            torch.randn(K, self.input_dim) * center_std
        )
        self.raw_amp = nn.Parameter(
            torch.full((K,), _raw_from_positive(init_amp))
        )
        self.raw_kappa = nn.Parameter(
            torch.full((K,), _raw_from_positive(init_kappa))
        )

    @property
    def amplitudes(self) -> torch.Tensor:
        return F.softplus(self.raw_amp) + 1e-4

    @property
    def kappas(self) -> torch.Tensor:
        return F.softplus(self.raw_kappa) + 1e-4

    def forward(self, xi: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Return V(xi, h) with shape (..., 1)."""
        z = torch.cat([xi, h], dim=-1)
        z_e = z.unsqueeze(-2)
        diff = z_e - self.centers
        sqdist = (diff * diff).sum(dim=-1)
        amp = self.amplitudes
        kappa = self.kappas
        per_well = amp * (1.0 - torch.exp(-kappa.pow(2) * sqdist))
        V = per_well.sum(dim=-1, keepdim=True)
        return V

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ScalarPotentialLMSARFMassGM(ScalarPotentialLMSARFMass):
    """SARF-faithful SPLM with a Gaussian-mixture V_theta head."""

    def __init__(self, cfg: SPLMSARFMassGMConfig):
        super().__init__(cfg)
        self.cfg: SPLMSARFMassGMConfig = cfg
        self.V_theta = GaussianMixturePotential(
            d=cfg.d,
            K=cfg.gm_K,
            init_kappa=cfg.gm_init_kappa,
            init_amp=cfg.gm_init_amp,
            center_std=cfg.gm_center_std,
        )


def smoke_test():
    torch.manual_seed(0)
    V = 257
    cfg = SPLMSARFMassGMConfig(
        vocab_size=V, d=16, max_len=32, L=4,
        mass_mode="global",
        gm_K=16, gm_init_kappa=0.5, gm_init_amp=1.0,
    )
    net = ScalarPotentialLMSARFMassGM(cfg)
    x = torch.randint(0, V, (2, 16))
    y = torch.randint(0, V, (2, 16))
    logits, loss = net(x, y)
    loss.backward()

    with torch.no_grad():
        xi = torch.randn(4, cfg.d)
        h = torch.randn(4, cfg.d)
        V_vals = net.V_theta(xi, h)
        upper = net.V_theta.amplitudes.sum().item()
    print(f"[GM smoke]  params={net.num_params():,}  loss={loss.item():.4f}")
    print(f"[GM smoke]  V range on random z: "
          f"[{V_vals.min().item():.4f}, {V_vals.max().item():.4f}]  "
          f"structural upper bound = sum amp_k = {upper:.4f}")
    print(f"[GM smoke]  V >= 0 everywhere: "
          f"{(V_vals >= -1e-6).all().item()}")


if __name__ == "__main__":
    smoke_test()
