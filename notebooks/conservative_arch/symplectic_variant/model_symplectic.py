"""
SARF-faithful scalar-potential LM with per-token semantic mass AND
a symplectic (velocity-Verlet + Strang-split damping) integrator.

Extends sarf_mass_variant/model_sarf_mass.py with exactly one structural
change: the damped-Euler integrator is replaced by a second-order,
symmetric kick-drift-kick velocity-Verlet scheme with Strang-split
(analytically-integrated) damping.  Everything else is identical:

  - SARF-faithful xi re-pooling per layer
  - shared V_theta(xi, h)
  - per-token mass m_t (global / embed_head / logfreq)
  - learnable scalar damping gamma
  - tied-embedding readout

The continuous equation of motion is unchanged,

    m_t * h_ddot = - grad_h V_theta(xi, h) - gamma * m_t * h_dot,

but the discretisation is now symmetric:

    Strang splitting of  dx/dt = A(x) + B(x),
       where  A = Hamiltonian flow  (force -grad V / m)
              B = damping flow      (v' = -gamma v,
                                     integrable as v <- v * exp(-gamma dt)).

One full step (kick-drift-kick velocity-Verlet for A, plus half-steps of B):

    # half-step damping
    v <- v * exp(-gamma dt / 2)
    # kick 1 (half force)
    v <- v + (dt/2) * f(h) / m
    # drift
    h <- h + dt * v
    # recompute xi at new h, and the force there
    f(h) <- - grad V_theta(cumul_mean(h), h)
    # kick 2 (half force)
    v <- v + (dt/2) * f(h) / m
    # half-step damping
    v <- v * exp(-gamma dt / 2)

With the KDK optimisation, adjacent `kick 2 -> kick 1` and
`damp_half -> damp_half` pairs fuse into a single full-dt update, so
the cost per L-step stack is  L + 1  force evaluations (vs Euler's L),
which is a ~12.5% overhead at L=8 for O(dt^2) accuracy instead of O(dt).

No new learnable parameters.  This is a pure integrator swap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SPLMSymplecticConfig:
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

    # Mass parameterisation: identical semantics to sarf_mass_variant.
    mass_mode: str = "global"
    logfreq_init_alpha: float = 0.0
    logfreq_path: Optional[str] = None


class ScalarPotential(nn.Module):
    """MLP (xi, h) -> scalar energy. Identical to SARF baseline."""

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
    T = h.shape[1]
    cumsum = h.cumsum(dim=1)
    denom = torch.arange(1, T + 1, device=h.device, dtype=h.dtype).view(1, T, 1)
    return cumsum / denom


def _raw_from_positive(y: float) -> float:
    import math
    return math.log(math.expm1(max(y, 1e-3)))


def _force_and_xi(V_theta: ScalarPotential,
                  h: torch.Tensor,
                  create_graph: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (f = -grad_h V(xi(h), h), xi(h)).

    The input tensor `h` must have requires_grad=True (the caller is
    responsible, to avoid leaf-tensor surprises inside the integrator loop).
    """
    xi = causal_cumulative_mean(h)
    V = V_theta(xi, h).sum()
    grad_V, = torch.autograd.grad(
        V, h, create_graph=create_graph, retain_graph=True,
    )
    return -grad_V, xi


class ScalarPotentialLMSymplectic(nn.Module):
    """SARF-faithful SPLM with per-token mass + velocity-Verlet integrator."""

    def __init__(self, cfg: SPLMSymplecticConfig):
        super().__init__()
        self.cfg = cfg
        self.E = nn.Embedding(cfg.vocab_size, cfg.d)
        self.P = nn.Parameter(torch.zeros(cfg.max_len, cfg.d))
        nn.init.normal_(self.E.weight, std=0.02)
        nn.init.normal_(self.P, std=0.01)

        self.V_theta = ScalarPotential(cfg.d, cfg.v_hidden, cfg.v_depth)

        self.raw_m_bias = nn.Parameter(
            torch.tensor(_raw_from_positive(cfg.init_m)),
            requires_grad=cfg.learn_mgamma,
        )
        self.raw_gamma = nn.Parameter(
            torch.tensor(_raw_from_positive(cfg.init_gamma)),
            requires_grad=cfg.learn_mgamma,
        )

        if cfg.mass_mode == "global":
            pass
        elif cfg.mass_mode == "embed_head":
            self.mass_head = nn.Linear(cfg.d, 1, bias=True)
            nn.init.zeros_(self.mass_head.weight)
            nn.init.zeros_(self.mass_head.bias)
        elif cfg.mass_mode == "logfreq":
            if cfg.logfreq_path is None:
                raise ValueError(
                    "mass_mode='logfreq' requires cfg.logfreq_path (a .npy "
                    "file with one surprisal value per vocabulary id)."
                )
            surprisal = torch.from_numpy(_load_npy(cfg.logfreq_path)).float()
            if surprisal.numel() != cfg.vocab_size:
                raise ValueError(
                    f"logfreq vector length {surprisal.numel()} != "
                    f"vocab_size {cfg.vocab_size}"
                )
            self.register_buffer("logfreq_surprisal", surprisal)
            self.raw_logfreq_alpha = nn.Parameter(
                torch.tensor(_raw_from_positive(max(cfg.logfreq_init_alpha, 1e-3))),
                requires_grad=True,
            )
        else:
            raise ValueError(f"unknown mass_mode: {cfg.mass_mode!r}")

    @property
    def gamma(self) -> torch.Tensor:
        return F.softplus(self.raw_gamma)

    @property
    def m_global(self) -> torch.Tensor:
        return F.softplus(self.raw_m_bias) + 1e-3

    def compute_mass(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        if cfg.mass_mode == "global":
            return self.m_global
        if cfg.mass_mode == "embed_head":
            raw = self.mass_head(self.E(x))
            return F.softplus(raw + self.raw_m_bias) + 1e-3
        if cfg.mass_mode == "logfreq":
            surprisal = self.logfreq_surprisal[x]
            alpha = F.softplus(self.raw_logfreq_alpha)
            scaled = alpha * surprisal.unsqueeze(-1)
            return F.softplus(self.raw_m_bias + scaled) + 1e-3
        raise RuntimeError("unreachable")

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        pos = self.P[:T].unsqueeze(0)
        return self.E(x) + pos

    def integrate(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        return_trajectory: bool = False,
        return_xi_trajectory: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
        """Damped velocity-Verlet + Strang-split damping.

        Same signature and trajectory semantics as sarf_mass_variant so that
        the paper's downstream analysis scripts (shared_potential_fit.py,
        token_direction_fit.py, compare.py) can consume trajectories from
        this variant without modification:

          - traj_h:  L+1 tensors (initial + state after each of the L steps)
          - traj_xi: L tensors  (xi used in the first force kick of each step)
        """
        cfg = self.cfg
        dt = cfg.dt
        gamma = self.gamma

        m_b = self.compute_mass(x, emb)

        damp_half = torch.exp(-0.5 * dt * gamma)
        damp_full = torch.exp(-dt * gamma)

        h = emb
        v = torch.zeros_like(h)

        traj_h: Optional[List[torch.Tensor]] = None
        traj_xi: Optional[List[torch.Tensor]] = None
        if return_trajectory:
            traj_h = [h.detach().cpu()]
        if return_xi_trajectory:
            traj_xi = []

        create_graph = self.training

        h_rg = h.detach().clone().requires_grad_(True) \
            if not h.requires_grad else h
        f, xi = _force_and_xi(self.V_theta, h_rg, create_graph=create_graph)
        h = h_rg
        if return_xi_trajectory:
            assert traj_xi is not None
            traj_xi.append(xi.detach().cpu())

        v = v * damp_half

        for step in range(cfg.L):
            v = v + 0.5 * dt * f / m_b

            h_new = h + dt * v

            if not h_new.requires_grad:
                h_new = h_new.requires_grad_(True)
            f_new, xi_new = _force_and_xi(
                self.V_theta, h_new, create_graph=create_graph,
            )

            v = v + 0.5 * dt * f_new / m_b

            if step == cfg.L - 1:
                v = v * damp_half
            else:
                v = v * damp_full

            h = h_new
            f = f_new

            if return_trajectory:
                assert traj_h is not None
                traj_h.append(h.detach().cpu())
            if return_xi_trajectory and step < cfg.L - 1:
                assert traj_xi is not None
                traj_xi.append(xi_new.detach().cpu())

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
            x, emb,
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
                 temperature: float = 1.0,
                 top_k: Optional[int] = None) -> torch.Tensor:
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

    @torch.no_grad()
    def mass_stats(self, x: torch.Tensor) -> dict:
        emb = self._embed(x)
        m = self.compute_mass(x, emb)
        if m.dim() == 0:
            return {"mean": float(m.item()), "std": 0.0,
                    "min": float(m.item()), "max": float(m.item())}
        m_flat = m.reshape(-1)
        return {
            "mean": float(m_flat.mean().item()),
            "std":  float(m_flat.std().item()),
            "min":  float(m_flat.min().item()),
            "max":  float(m_flat.max().item()),
        }


def _load_npy(path: str):
    import numpy as np
    return np.load(path)


if __name__ == "__main__":
    import tempfile
    import numpy as np

    torch.manual_seed(0)
    V = 257

    print("=" * 60)
    print("[symplectic] smoke test: global mass")
    cfg = SPLMSymplecticConfig(
        vocab_size=V, d=16, max_len=32, v_hidden=32, v_depth=2, L=4,
        mass_mode="global",
    )
    net = ScalarPotentialLMSymplectic(cfg)
    x = torch.randint(0, V, (2, 16))
    y = torch.randint(0, V, (2, 16))
    logits, loss = net(x, y)
    loss.backward()
    stats = net.mass_stats(x)
    print(f"  params={net.num_params():,}  loss={loss.item():.4f}  "
          f"mass mean={stats['mean']:.3f}  std={stats['std']:.3f}")

    print("=" * 60)
    print("[symplectic] smoke test: embed_head mass")
    cfg = SPLMSymplecticConfig(
        vocab_size=V, d=16, max_len=32, v_hidden=32, v_depth=2, L=4,
        mass_mode="embed_head",
    )
    net = ScalarPotentialLMSymplectic(cfg)
    logits, loss = net(x, y)
    loss.backward()
    stats = net.mass_stats(x)
    print(f"  params={net.num_params():,}  loss={loss.item():.4f}  "
          f"mass mean={stats['mean']:.3f}  std={stats['std']:.3f}  "
          f"min={stats['min']:.3f}  max={stats['max']:.3f}")

    print("=" * 60)
    print("[symplectic] smoke test: logfreq mass")
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tf:
        rng = np.random.default_rng(0)
        surp = rng.uniform(low=1.0, high=12.0, size=V).astype(np.float32)
        np.save(tf.name, surp)
        logfreq_path = tf.name
    cfg = SPLMSymplecticConfig(
        vocab_size=V, d=16, max_len=32, v_hidden=32, v_depth=2, L=4,
        mass_mode="logfreq", logfreq_init_alpha=0.1, logfreq_path=logfreq_path,
    )
    net = ScalarPotentialLMSymplectic(cfg)
    logits, loss = net(x, y)
    loss.backward()
    stats = net.mass_stats(x)
    print(f"  params={net.num_params():,}  loss={loss.item():.4f}  "
          f"mass mean={stats['mean']:.3f}  std={stats['std']:.3f}  "
          f"min={stats['min']:.3f}  max={stats['max']:.3f}")
    print("[symplectic] all three mass modes work end-to-end with Verlet.")
