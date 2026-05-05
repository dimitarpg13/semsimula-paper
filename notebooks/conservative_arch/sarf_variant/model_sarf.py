"""
SARF-faithful scalar-potential conservative-by-construction LM.

This is the baseline `model.py` with ONE architectural change: the causal
cumulative-mean context pool xi_t^{(ell)} is recomputed at every integration
step from the *current* hidden states h^{(ell)}, instead of being computed
once from the layer-0 embeddings and held fixed across all L steps.

This matches the definition in paper_v2/sections/14_conservative_architectures.tex:

    xi_t^{(ell)} := (1/t) * sum_{s <= t} h_s^{(ell)}, computed anew at every layer.

See docs/Training_and_Inference_with_SPLM.md §4 and the conversation that
produced this branch for the design rationale and the comparison protocol.

All other architecture choices (shared V_theta, learned scalar m and gamma,
damped Euler-Lagrange integrator, tied-embedding readout) are identical to
the baseline model.  The change is one-liner in scope and preserves
conservativity, positivity of m and gamma, and the paper's update formula.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SPLMSARFConfig:
    vocab_size: int = 50257
    d: int = 128
    max_len: int = 512
    v_hidden: int = 512
    v_depth: int = 3
    L: int = 8
    dt: float = 1.0
    init_m: float = 1.0
    init_gamma: float = 1.0
    learn_mgamma: bool = True

    # See SPLMSARFMassConfig.causal_force for full documentation.
    # When True (default), ξ is computed from h.detach() inside the
    # integration loop, severing the autograd path from ξ back to h
    # and restoring the physics-correct per-token Euler-Lagrange force.
    causal_force: bool = True


class ScalarPotential(nn.Module):
    """MLP  (xi, h) in R^(2d)  ->  scalar energy.  Identical to baseline."""

    def __init__(self, d: int, hidden: int, depth: int):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(2 * d, hidden), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        last = [m for m in self.net.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.normal_(last.weight, std=0.002)
        nn.init.zeros_(last.bias)

    def forward(self, xi: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([xi, h], dim=-1))


def causal_cumulative_mean(h: torch.Tensor) -> torch.Tensor:
    """xi_t = (1/t) * sum_{s <= t} h_s, along the token dim (dim=1).

    h:  (B, T, d)  ->  xi: (B, T, d)
    """
    T = h.shape[1]
    cumsum = h.cumsum(dim=1)
    denom = torch.arange(1, T + 1, device=h.device, dtype=h.dtype).view(1, T, 1)
    return cumsum / denom


class ScalarPotentialLMSARF(nn.Module):
    """SARF-faithful variant of ScalarPotentialLM.

    Difference from baseline: xi is recomputed at every integration step
    from the current hidden states h, instead of being computed once from
    layer-0 embeddings.
    """

    def __init__(self, cfg: SPLMSARFConfig):
        super().__init__()
        self.cfg = cfg
        self.E = nn.Embedding(cfg.vocab_size, cfg.d)
        self.P = nn.Parameter(torch.zeros(cfg.max_len, cfg.d))
        nn.init.normal_(self.E.weight, std=0.02)
        nn.init.normal_(self.P, std=0.01)

        self.V_theta = ScalarPotential(cfg.d, cfg.v_hidden, cfg.v_depth)

        def _raw_from_positive(y: float) -> float:
            import math
            return math.log(math.expm1(max(y, 1e-3)))
        self.raw_m     = nn.Parameter(torch.tensor(_raw_from_positive(cfg.init_m)),
                                      requires_grad=cfg.learn_mgamma)
        self.raw_gamma = nn.Parameter(torch.tensor(_raw_from_positive(cfg.init_gamma)),
                                      requires_grad=cfg.learn_mgamma)

    @property
    def m(self) -> torch.Tensor:
        return F.softplus(self.raw_m) + 1e-3

    @property
    def gamma(self) -> torch.Tensor:
        return F.softplus(self.raw_gamma)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        pos = self.P[:T].unsqueeze(0)
        return self.E(x) + pos

    def integrate(
        self,
        emb: torch.Tensor,
        return_trajectory: bool = False,
        return_xi_trajectory: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
        """Damped Lagrangian flow with per-layer re-pooled xi.

        emb: (B, T, d)  initial h_0 = E(x) + P

        Returns (h_L, traj_h?, traj_xi?).  traj_h is the list of CPU
        snapshots [h_0, h_1, ..., h_L]; traj_xi is [xi_0, xi_1, ..., xi_{L-1}]
        (only the xis actually used in force evaluations).
        """
        cfg = self.cfg
        h = emb
        v = torch.zeros_like(h)
        m, gamma, dt = self.m, self.gamma, cfg.dt

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
                V, h_in, create_graph=self.training, retain_graph=True,
            )
            f = -grad_V
            v = (v + dt * f / m) / (1.0 + dt * gamma)
            h = h_in + dt * v
            if return_trajectory:
                assert traj_h is not None
                traj_h.append(h.detach().cpu())

        return h, traj_h, traj_xi

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
        return_xi_trajectory: bool = False,
    ):
        emb = self._embed(x)
        h_L, traj_h, traj_xi = self.integrate(
            emb,
            return_trajectory=return_trajectory,
            return_xi_trajectory=return_xi_trajectory,
        )
        logits = h_L @ self.E.weight.T

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.cfg.vocab_size),
                targets.reshape(-1),
            )
        out = [logits, loss]
        if return_trajectory:
            out.append(traj_h)
        if return_xi_trajectory:
            out.append(traj_xi)
        return tuple(out) if len(out) > 2 else (out[0], out[1])

    @torch.no_grad()
    def generate(self, x: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            x_cond = x[:, -self.cfg.max_len:]
            logits, _ = self.forward(x_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, nxt], dim=1)
        return x

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = SPLMSARFConfig(vocab_size=257, d=16, max_len=32, v_hidden=32, v_depth=2, L=4)
    net = ScalarPotentialLMSARF(cfg)
    print(f"[sarf] params: {net.num_params():,}   m={net.m.item():.3f}  "
          f"gamma={net.gamma.item():.3f}")

    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = net(x, y)
    loss.backward()
    print(f"[sarf] loss: {loss.item():.4f}  logits: {tuple(logits.shape)}  "
          f"grad_norm(E): {net.E.weight.grad.norm().item():.4f}  "
          f"grad_norm(V_theta.last): "
          f"{[p for p in net.V_theta.net.parameters()][-2].grad.norm().item():.4f}")

    net.eval()
    with torch.enable_grad():
        out = net(x, y, return_trajectory=True, return_xi_trajectory=True)
        _, _, traj_h, traj_xi = out
    assert len(traj_h) == cfg.L + 1, (len(traj_h), cfg.L + 1)
    assert len(traj_xi) == cfg.L, (len(traj_xi), cfg.L)
    assert traj_h[0].shape == (2, 16, cfg.d)
    assert traj_xi[0].shape == (2, 16, cfg.d)
    print(f"[sarf] traj_h: {len(traj_h)} layers x {tuple(traj_h[0].shape)}")
    print(f"[sarf] traj_xi: {len(traj_xi)} layers x {tuple(traj_xi[0].shape)}")

    import sys
    sys.path.insert(0, "..")
    from model import ScalarPotentialLM, SPLMConfig
    torch.manual_seed(0)
    baseline_cfg = SPLMConfig(vocab_size=257, d=16, max_len=32, v_hidden=32, v_depth=2, L=4)
    baseline = ScalarPotentialLM(baseline_cfg)
    torch.manual_seed(0)
    logits_b, _ = baseline(x, y)
    torch.manual_seed(0)
    logits_s, _ = ScalarPotentialLMSARF(SPLMSARFConfig(
        vocab_size=257, d=16, max_len=32, v_hidden=32, v_depth=2, L=4,
    ))(x, y)
    assert logits_b.shape == logits_s.shape
    print(f"[sarf] baseline logits norm: {logits_b.norm().item():.4f}   "
          f"sarf logits norm: {logits_s.norm().item():.4f}   "
          f"(different by construction at L=1+)")
