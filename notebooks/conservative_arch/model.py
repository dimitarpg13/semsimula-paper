"""
Scalar-potential conservative-by-construction language model.

Reference:
  docs/Conservative_by_Construction_Language_Models.md §3.1

Architecture
------------
  Token embedding E (V, d) + positional embedding P (T_max, d).
  Causal cumulative-mean pool of (emb(x) + P) gives per-position
  context vector xi_t in R^d.
  Shared scalar energy V_theta(xi, h): R^(2d) -> R, realised as a small
  MLP with GELU activations and a final 1-unit linear head.
  Per-position inference runs a damped Euler-Lagrange flow on the
  learned energy:

      h_0 = emb(x) + P            (contextualised start)
      v_0 = 0
      for i = 1..L:
          f = -grad_h V_theta(xi, h_{i-1})
          v_i = (v_{i-1} + dt * f / m) / (1 + dt * gamma)
          h_i = h_{i-1} + dt * v_i

  Output logits_t = h_L_t @ E^T (tied embeddings), cross-entropy loss
  against the next-token target.

All "layers" (integration steps) share the same V_theta, m, gamma.  The
resulting flow is conservative on h by construction and the depth of
the network is the number of integration steps on a single shared
force law.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SPLMConfig:
    vocab_size: int = 50257
    d: int = 128
    max_len: int = 512
    v_hidden: int = 512
    v_depth: int = 3         # number of hidden layers in V_theta MLP
    L: int = 8               # integration steps (shared-weight "depth")
    dt: float = 1.0
    init_m: float = 1.0
    init_gamma: float = 1.0
    learn_mgamma: bool = True


class ScalarPotential(nn.Module):
    """MLP  (xi, h) in R^(2d)  ->  scalar energy."""

    def __init__(self, d: int, hidden: int, depth: int):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(2 * d, hidden), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
        # Gentle init so gradients are small at start -- integrator stays stable.
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Final layer even smaller -- the initial force should be tiny.
        last = [m for m in self.net.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.normal_(last.weight, std=0.002)
        nn.init.zeros_(last.bias)

    def forward(self, xi: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """xi: (..., d), h: (..., d)  ->  V: (..., 1)"""
        return self.net(torch.cat([xi, h], dim=-1))


class ScalarPotentialLM(nn.Module):
    def __init__(self, cfg: SPLMConfig):
        super().__init__()
        self.cfg = cfg
        self.E = nn.Embedding(cfg.vocab_size, cfg.d)
        self.P = nn.Parameter(torch.zeros(cfg.max_len, cfg.d))
        nn.init.normal_(self.E.weight, std=0.02)
        nn.init.normal_(self.P, std=0.01)

        self.V_theta = ScalarPotential(cfg.d, cfg.v_hidden, cfg.v_depth)

        # Positive-definite mass / damping via softplus.
        # raw values chosen so F.softplus(raw) ~ init_m / init_gamma.
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

    # ------------------------------------------------------------------
    def _embed_and_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T) token ids -> (emb, xi)"""
        B, T = x.shape
        pos = self.P[:T].unsqueeze(0)        # (1, T, d)
        emb = self.E(x) + pos                # (B, T, d)
        cumsum = emb.cumsum(dim=1)           # (B, T, d)
        denom = torch.arange(1, T + 1, device=x.device,
                             dtype=emb.dtype).view(1, T, 1)
        xi = cumsum / denom                  # causal cumulative mean
        return emb, xi

    # ------------------------------------------------------------------
    def integrate(
        self,
        emb: torch.Tensor,   # (B, T, d)  initial h_0
        xi:  torch.Tensor,   # (B, T, d)  context (held fixed along integration)
        return_trajectory: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Damped Lagrangian flow starting from h_0 = emb, v_0 = 0.

        Returns h_L (and optionally the full per-step trajectory
        [h_0, h_1, ..., h_L] as a list of detached tensors for analysis).
        """
        cfg = self.cfg
        h = emb
        v = torch.zeros_like(h)
        m, gamma, dt = self.m, self.gamma, cfg.dt

        traj: Optional[List[torch.Tensor]] = None
        if return_trajectory:
            traj = [h.detach().cpu()]

        for _ in range(cfg.L):
            # Force f = -grad_h V_theta(xi, h), differentiable through V_theta.
            h_in = h
            if not h_in.requires_grad:
                h_in = h_in.requires_grad_(True)
            V = self.V_theta(xi, h_in).sum()
            grad_V, = torch.autograd.grad(
                V, h_in, create_graph=self.training, retain_graph=True,
            )
            f = -grad_V
            v = (v + dt * f / m) / (1.0 + dt * gamma)
            h = h_in + dt * v
            if return_trajectory:
                assert traj is not None
                traj.append(h.detach().cpu())

        return h, traj

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
    ):
        """x: (B, T) token ids, targets: (B, T) next-token ids (or None).

        Returns (logits, loss, traj?).  traj is a list of (B, T, d)
        tensors on CPU only when return_trajectory=True.
        """
        emb, xi = self._embed_and_pool(x)
        h_L, traj = self.integrate(emb, xi, return_trajectory=return_trajectory)

        logits = h_L @ self.E.weight.T                                # (B, T, V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.cfg.vocab_size),
                targets.reshape(-1),
            )
        if return_trajectory:
            return logits, loss, traj
        return logits, loss

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, x: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """Autoregressive greedy / sampled generation.  x: (1, T0)."""
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

    # ------------------------------------------------------------------
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Self-test: tiny forward+backward works, mass/damping are positive, trajectory
# shape is correct.  Run with: python3 model.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = SPLMConfig(vocab_size=257, d=16, max_len=32, v_hidden=32, v_depth=2, L=4)
    net = ScalarPotentialLM(cfg)
    print(f"params: {net.num_params():,}   m={net.m.item():.3f}  "
          f"gamma={net.gamma.item():.3f}")

    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = net(x, y)
    loss.backward()
    print(f"loss: {loss.item():.4f}  logits: {tuple(logits.shape)}  "
          f"grad_norm(E): {net.E.weight.grad.norm().item():.4f}  "
          f"grad_norm(V_theta.last): "
          f"{[p for p in net.V_theta.net.parameters()][-2].grad.norm().item():.4f}")

    # Trajectory extraction sanity check
    net.eval()
    with torch.enable_grad():
        _, _, traj = net(x, y, return_trajectory=True)
    assert len(traj) == cfg.L + 1, (len(traj), cfg.L + 1)
    assert traj[0].shape == (2, 16, cfg.d)
    print(f"trajectory: {len(traj)} layers x {tuple(traj[0].shape)}")
