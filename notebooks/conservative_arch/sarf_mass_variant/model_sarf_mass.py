"""
SARF-faithful scalar-potential LM with per-token semantic mass.

Extends sarf_variant/model_sarf.py with three mass parameterisations,
selected by `cfg.mass_mode`:

  - "global"     : single learnable scalar, identical to the SARF baseline.
                   Included so this module can reproduce the baseline in a
                   single codebase and act as a null control.
  - "embed_head" : m_t = softplus(<w_m, E[x_t]> + b_m) + eps, i.e. a cheap
                   linear head on the token embedding.  Learned, position-
                   invariant, content-dependent.  Variant (A) of the plan.
  - "logfreq"    : m_t = 1 + alpha * (-log p_hat(x_t)), a frozen unigram-
                   surprisal prior with a single learnable scale alpha >= 0.
                   Variant (B) of the plan.

Everything else (SARF-faithful xi re-pooling, shared V_theta, damped
Euler-Lagrange integrator with learnable gamma, tied-embedding readout)
is identical to the SARF baseline.  The integrator only changes by one
broadcast: `m` can now be a tensor of shape (B, T, 1) instead of a 0-d
scalar.

Mass is computed once per forward pass from the first-layer input and
held fixed across the L integration steps, matching the framework's
per-particle-scalar prescription (no state-dependence; no layer drift).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SPLMSARFMassConfig:
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

    mass_mode: str = "global"
    logfreq_init_alpha: float = 0.0
    logfreq_path: Optional[str] = None

    fixed_gamma: Optional[float] = None

    # When True (default), compute ξ from h.detach() inside the integration
    # loop. This severs the autograd path from ξ back to h, eliminating an
    # anti-causal leak in the conservative force where ∂V[t']/∂h[t] ≠ 0 for
    # t' > t (because ξ[t'] is a causal weighted average that includes h[t]).
    # The fix restores the physics-correct Euler-Lagrange equation
    #     m · ḧ_t = -∂V(ξ_t, h_t)/∂h_t
    # as the per-token dynamics. See docs/Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md
    # for the full bug-and-fix writeup. Set causal_force=False ONLY to
    # bit-exactly reproduce pre-fix experiments (e.g. E9 forensics).
    causal_force: bool = True


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


class ScalarPotentialLMSARFMass(nn.Module):
    """SARF-faithful SPLM with pluggable per-token mass."""

    def __init__(self, cfg: SPLMSARFMassConfig):
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
        if cfg.fixed_gamma is not None:
            self.raw_gamma = nn.Parameter(
                torch.tensor(0.0),
                requires_grad=False,
            )
            self._gamma_value: Optional[float] = float(cfg.fixed_gamma)
        else:
            self.raw_gamma = nn.Parameter(
                torch.tensor(_raw_from_positive(cfg.init_gamma)),
                requires_grad=cfg.learn_mgamma,
            )
            self._gamma_value = None

        if cfg.mass_mode == "global":
            pass
        elif cfg.mass_mode == "embed_head":
            self.mass_head = nn.Linear(cfg.d, 1, bias=True)
            nn.init.zeros_(self.mass_head.weight)
            nn.init.zeros_(self.mass_head.bias)
        elif cfg.mass_mode == "logfreq":
            if cfg.logfreq_path is None:
                raise ValueError(
                    "mass_mode='logfreq' requires cfg.logfreq_path (a .npy file "
                    "with one surprisal value per vocabulary id)."
                )
            surprisal = torch.from_numpy(
                _load_npy(cfg.logfreq_path)
            ).float()
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
        if self._gamma_value is not None:
            return torch.full(
                (), self._gamma_value,
                device=self.raw_gamma.device, dtype=self.raw_gamma.dtype,
            )
        return F.softplus(self.raw_gamma)

    @property
    def m_global(self) -> torch.Tensor:
        return F.softplus(self.raw_m_bias) + 1e-3

    def compute_mass(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Return m as a tensor broadcastable over (B, T, d).

        x:   (B, T)         token ids
        emb: (B, T, d)      first-layer hidden state E[x] + P
        """
        cfg = self.cfg
        if cfg.mass_mode == "global":
            return self.m_global

        if cfg.mass_mode == "embed_head":
            raw = self.mass_head(self.E(x))
            return (F.softplus(raw + self.raw_m_bias) + 1e-3)

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
        """SARF-faithful damped EL with per-token mass.

        x:    (B, T)
        emb:  (B, T, d)   first-layer hidden state (with position embed)
        """
        cfg = self.cfg
        h = emb
        v = torch.zeros_like(h)
        gamma, dt = self.gamma, cfg.dt

        m = self.compute_mass(x, emb)
        if m.dim() == 0:
            m_b = m
        else:
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
                V, h_in, create_graph=self.training, retain_graph=True,
            )
            f = -grad_V
            v = (v + dt * f / m_b) / (1.0 + dt * gamma)
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

    @torch.no_grad()
    def mass_stats(self, x: torch.Tensor) -> dict:
        """Diagnostic summary of the per-token mass on a batch of tokens."""
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
    print("[mass] smoke test: global mass (baseline-equivalent)")
    cfg = SPLMSARFMassConfig(
        vocab_size=V, d=16, max_len=32, v_hidden=32, v_depth=2, L=4,
        mass_mode="global",
    )
    net = ScalarPotentialLMSARFMass(cfg)
    x = torch.randint(0, V, (2, 16))
    y = torch.randint(0, V, (2, 16))
    logits, loss = net(x, y)
    loss.backward()
    stats = net.mass_stats(x)
    print(f"  params={net.num_params():,}  loss={loss.item():.4f}  "
          f"mass mean={stats['mean']:.3f}  std={stats['std']:.3f}")

    print("=" * 60)
    print("[mass] smoke test: embed_head mass")
    cfg = SPLMSARFMassConfig(
        vocab_size=V, d=16, max_len=32, v_hidden=32, v_depth=2, L=4,
        mass_mode="embed_head",
    )
    net = ScalarPotentialLMSARFMass(cfg)
    logits, loss = net(x, y)
    loss.backward()
    stats = net.mass_stats(x)
    print(f"  params={net.num_params():,}  loss={loss.item():.4f}  "
          f"mass mean={stats['mean']:.3f}  std={stats['std']:.3f}  "
          f"min={stats['min']:.3f}  max={stats['max']:.3f}")

    print("=" * 60)
    print("[mass] smoke test: logfreq mass")
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tf:
        rng = np.random.default_rng(0)
        surp = rng.uniform(low=1.0, high=12.0, size=V).astype(np.float32)
        np.save(tf.name, surp)
        logfreq_path = tf.name
    cfg = SPLMSARFMassConfig(
        vocab_size=V, d=16, max_len=32, v_hidden=32, v_depth=2, L=4,
        mass_mode="logfreq", logfreq_init_alpha=0.1, logfreq_path=logfreq_path,
    )
    net = ScalarPotentialLMSARFMass(cfg)
    logits, loss = net(x, y)
    loss.backward()
    stats = net.mass_stats(x)
    print(f"  params={net.num_params():,}  loss={loss.item():.4f}  "
          f"mass mean={stats['mean']:.3f}  std={stats['std']:.3f}  "
          f"min={stats['min']:.3f}  max={stats['max']:.3f}")
    print("[mass] all three mass modes work end-to-end.")
