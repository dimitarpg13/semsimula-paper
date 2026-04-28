"""Extract per-layer energy states from an SPLM checkpoint.

Re-runs the SPLM integration loop on the §1 e-init test corpus and
saves, for every sentence and every layer, the four quantities needed
for the energy-drift diagnostic:

  - kinetic_per_token   = (1/2) m_t ||v_l||^2          shape (n_sent, L+1, T_max)
  - potential_per_token = V_theta(xi_l, h_l)           shape (n_sent, L+1, T_max)
  - kinetic             = mean_t kinetic_per_token     shape (n_sent, L+1)
  - potential           = mean_t potential_per_token   shape (n_sent, L+1)
  - H                   = kinetic + potential          shape (n_sent, L+1)

Plus a per-sentence ``T`` array so downstream code knows how much of
the T_max axis is real per sentence.

Three SPLM variants are supported:

    --variant euler       parent-SPLM (notebooks/conservative_arch/model.py),
                          fixed-xi damped-Euler integrator, scalar mass
    --variant sarfmass    sarf_mass_variant (SARF-faithful re-pool +
                          per-token mass + Euler integrator)
    --variant symplectic  symplectic_variant (SARF-faithful re-pool +
                          per-token mass + velocity-Verlet integrator with
                          Strang-split damping)

Usage::

    python3 extract_energy_states.py \\
        --variant sarfmass \\
        --ckpt path/to/ckpt.pt \\
        --label splm_sarfmass_L8 \\
        --logfreq path/to/logfreq_surprisal.npy \\
        --out_npz results/splm_sarfmass_L8.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONSERVATIVE_ARCH_DIR = SCRIPT_DIR.parent
SARF_MASS_DIR = CONSERVATIVE_ARCH_DIR / "sarf_mass_variant"
SYMPLECTIC_DIR = CONSERVATIVE_ARCH_DIR / "symplectic_variant"
ENERGETIC_MINIMA_DIR = CONSERVATIVE_ARCH_DIR / "energetic_minima"

REPO_ROOT = SCRIPT_DIR.parent.parent.parent

# Importing the upstream model packages requires their directories on sys.path
# (they expose modules like ``model``, ``model_sarf_mass``, ``model_symplectic``,
# ``model_ln``, ``e_init_corpus``).
for _p in (CONSERVATIVE_ARCH_DIR, SARF_MASS_DIR, SYMPLECTIC_DIR,
           ENERGETIC_MINIMA_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from e_init_corpus import CORPUS  # noqa: E402  (import after sys.path edit)


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# variant adapters

class _BaseAdapter:
    """One adapter per SPLM variant, re-implementing its integrator with
    full per-layer state capture (h, v, V_theta, kinetic)."""

    label: str = "base"

    def __init__(self, ckpt_path: Path, device: str,
                 logfreq_path: Optional[Path] = None) -> None:
        self.ckpt_path = ckpt_path
        self.device = device
        self.logfreq_path = logfreq_path

    def load(self) -> None:
        raise NotImplementedError

    def integrate_with_capture(
        self, x: torch.Tensor,
    ) -> Dict[str, np.ndarray]:
        """Return per-layer (kin_pt, pot_pt) of shape (L+1, T) each.

        Returns numpy arrays already on CPU.  Layer 0 is the initial
        state (kinetic = 0, potential = V_theta at h_0).
        """
        raise NotImplementedError

    @property
    def L(self) -> int:
        raise NotImplementedError

    @property
    def d(self) -> int:
        raise NotImplementedError


class EulerAdapter(_BaseAdapter):
    """Parent-SPLM, fixed-xi, scalar mass, damped Euler."""

    label = "euler"

    def load(self) -> None:
        from model import ScalarPotentialLM, SPLMConfig
        ck = torch.load(self.ckpt_path, map_location=self.device,
                        weights_only=False)
        cfg = SPLMConfig(**ck["model_cfg"])
        model = ScalarPotentialLM(cfg).to(self.device)
        model.load_state_dict(ck["model_state_dict"])
        model.eval()
        self.model = model
        self.cfg = cfg

    @property
    def L(self) -> int:
        return self.cfg.L

    @property
    def d(self) -> int:
        return self.cfg.d

    def integrate_with_capture(
        self, x: torch.Tensor,
    ) -> Dict[str, np.ndarray]:
        m = self.model
        cfg = self.cfg
        with torch.enable_grad():
            emb, xi = m._embed_and_pool(x)
            h = emb
            v = torch.zeros_like(h)
            mass = m.m  # scalar
            gamma = m.gamma
            dt = cfg.dt

            kin_pt: List[torch.Tensor] = []
            pot_pt: List[torch.Tensor] = []
            kin_pt.append(
                (0.5 * mass * (v * v).sum(dim=-1)).detach()
            )
            with torch.no_grad():
                pot_pt.append(m.V_theta(xi, h).squeeze(-1).detach())

            for _ in range(cfg.L):
                h_in = h.detach().clone().requires_grad_(True)
                V = m.V_theta(xi, h_in).sum()
                grad_V, = torch.autograd.grad(V, h_in, retain_graph=False)
                f = -grad_V
                v = (v + dt * f / mass) / (1.0 + dt * gamma)
                h = h_in + dt * v
                kin_pt.append(
                    (0.5 * mass * (v * v).sum(dim=-1)).detach()
                )
                with torch.no_grad():
                    pot_pt.append(m.V_theta(xi, h).squeeze(-1).detach())

        kin_arr = torch.stack(kin_pt, dim=0).squeeze(1).cpu().numpy()
        pot_arr = torch.stack(pot_pt, dim=0).squeeze(1).cpu().numpy()
        return dict(kinetic_per_token=kin_arr, potential_per_token=pot_arr)


class SarfMassAdapter(_BaseAdapter):
    """sarf_mass_variant: SARF-faithful xi re-pool, per-token mass, Euler."""

    label = "sarfmass"

    def load(self) -> None:
        from model_sarf_mass import (
            ScalarPotentialLMSARFMass, SPLMSARFMassConfig
        )
        ck = torch.load(self.ckpt_path, map_location=self.device,
                        weights_only=False)
        cfg_dict = dict(ck["model_cfg"])
        if cfg_dict.get("mass_mode") == "logfreq":
            if self.logfreq_path is None:
                raise ValueError(
                    "logfreq mass_mode requires --logfreq <path>"
                )
            cfg_dict["logfreq_path"] = str(self.logfreq_path)
        cfg = SPLMSARFMassConfig(**cfg_dict)
        model = ScalarPotentialLMSARFMass(cfg).to(self.device)
        model.load_state_dict(ck["model_state_dict"])
        model.eval()
        self.model = model
        self.cfg = cfg

    @property
    def L(self) -> int:
        return self.cfg.L

    @property
    def d(self) -> int:
        return self.cfg.d

    def integrate_with_capture(
        self, x: torch.Tensor,
    ) -> Dict[str, np.ndarray]:
        m = self.model
        cfg = self.cfg
        from model_sarf_mass import causal_cumulative_mean
        with torch.enable_grad():
            emb = m._embed(x)
            mass = m.compute_mass(x, emb)
            mass_b = mass.detach()
            if mass_b.dim() < 3:
                mass_scalar = mass_b
            else:
                mass_scalar = mass_b.squeeze(-1)
            h = emb
            v = torch.zeros_like(h)
            gamma = m.gamma
            dt = cfg.dt

            kin_pt: List[torch.Tensor] = []
            pot_pt: List[torch.Tensor] = []

            xi0 = causal_cumulative_mean(h)
            kin_pt.append(
                (0.5 * mass_scalar * (v * v).sum(dim=-1)).detach()
            )
            with torch.no_grad():
                pot_pt.append(m.V_theta(xi0, h).squeeze(-1).detach())

            for _ in range(cfg.L):
                xi = causal_cumulative_mean(h)
                h_in = h.detach().clone().requires_grad_(True)
                xi_d = xi.detach()
                V = m.V_theta(xi_d, h_in).sum()
                grad_V, = torch.autograd.grad(V, h_in, retain_graph=False)
                f = -grad_V
                v = (v + dt * f / mass_b) / (1.0 + dt * gamma)
                h = h_in + dt * v
                kin_pt.append(
                    (0.5 * mass_scalar * (v * v).sum(dim=-1)).detach()
                )
                with torch.no_grad():
                    xi_after = causal_cumulative_mean(h)
                    pot_pt.append(m.V_theta(xi_after, h).squeeze(-1).detach())

        kin_arr = torch.stack(kin_pt, dim=0).squeeze(1).cpu().numpy()
        pot_arr = torch.stack(pot_pt, dim=0).squeeze(1).cpu().numpy()
        return dict(kinetic_per_token=kin_arr, potential_per_token=pot_arr)


class SymplecticAdapter(_BaseAdapter):
    """symplectic_variant: SARF-faithful, per-token mass, velocity-Verlet."""

    label = "symplectic"

    def load(self) -> None:
        from model_symplectic import (
            ScalarPotentialLMSymplectic, SPLMSymplecticConfig
        )
        ck = torch.load(self.ckpt_path, map_location=self.device,
                        weights_only=False)
        cfg_dict = dict(ck["model_cfg"])
        if cfg_dict.get("mass_mode") == "logfreq":
            if self.logfreq_path is None:
                raise ValueError(
                    "logfreq mass_mode requires --logfreq <path>"
                )
            cfg_dict["logfreq_path"] = str(self.logfreq_path)
        cfg = SPLMSymplecticConfig(**cfg_dict)
        model = ScalarPotentialLMSymplectic(cfg).to(self.device)
        model.load_state_dict(ck["model_state_dict"])
        model.eval()
        self.model = model
        self.cfg = cfg

    @property
    def L(self) -> int:
        return self.cfg.L

    @property
    def d(self) -> int:
        return self.cfg.d

    def integrate_with_capture(
        self, x: torch.Tensor,
    ) -> Dict[str, np.ndarray]:
        m = self.model
        cfg = self.cfg
        from model_symplectic import causal_cumulative_mean
        dt = cfg.dt
        gamma = m.gamma
        damp_half = torch.exp(-0.5 * dt * gamma)
        damp_full = torch.exp(-dt * gamma)

        with torch.enable_grad():
            emb = m._embed(x)
            mass = m.compute_mass(x, emb)
            mass_b = mass.detach()
            mass_scalar = mass_b if mass_b.dim() < 3 else mass_b.squeeze(-1)

            h = emb
            v = torch.zeros_like(h)

            kin_pt: List[torch.Tensor] = []
            pot_pt: List[torch.Tensor] = []

            h_rg = h.detach().clone().requires_grad_(True)
            xi0 = causal_cumulative_mean(h_rg)
            V0_for_grad = m.V_theta(xi0, h_rg).sum()
            grad_V0, = torch.autograd.grad(
                V0_for_grad, h_rg, retain_graph=False,
            )
            f = -grad_V0
            h = h_rg

            kin_pt.append(
                (0.5 * mass_scalar * (v * v).sum(dim=-1)).detach()
            )
            with torch.no_grad():
                pot_pt.append(m.V_theta(xi0, h).squeeze(-1).detach())

            v = v * damp_half

            for step in range(cfg.L):
                v = v + 0.5 * dt * f / mass_b
                h_new = h.detach() + dt * v
                h_new_rg = h_new.detach().clone().requires_grad_(True)
                xi_new = causal_cumulative_mean(h_new_rg)
                V_for_grad = m.V_theta(xi_new, h_new_rg).sum()
                grad_V_new, = torch.autograd.grad(
                    V_for_grad, h_new_rg, retain_graph=False,
                )
                f_new = -grad_V_new

                v = v + 0.5 * dt * f_new / mass_b

                if step == cfg.L - 1:
                    v = v * damp_half
                else:
                    v = v * damp_full

                h = h_new_rg
                f = f_new

                kin_pt.append(
                    (0.5 * mass_scalar * (v * v).sum(dim=-1)).detach()
                )
                with torch.no_grad():
                    pot_pt.append(
                        m.V_theta(xi_new, h).squeeze(-1).detach()
                    )

        kin_arr = torch.stack(kin_pt, dim=0).squeeze(1).cpu().numpy()
        pot_arr = torch.stack(pot_pt, dim=0).squeeze(1).cpu().numpy()
        return dict(kinetic_per_token=kin_arr, potential_per_token=pot_arr)


class EmLnAdapter(_BaseAdapter):
    """energetic_minima/model_ln.py: SARF-faithful xi re-pool, per-token mass,
    damped Euler integrator, with a LayerNorm projection inserted after each
    position update (and once on the initial embedding, matching
    ``model_ln.py:integrate``)::

        h_0    = LN(emb)                          if cfg.ln_after_step
        v_l+1  = (v_l + dt * f_l / m) / (1 + dt * gamma)
        h_l+1  = LN(h_l + dt * v_l+1)             if cfg.ln_after_step

    The kinetic energy is computed before the LN projection (the velocity is
    not touched by LN); the potential is evaluated *at* the LN-projected h, so
    the reported H_l reflects the same h that the model's loss head sees.
    """

    label = "em_ln"

    def load(self) -> None:
        from model_ln import (
            ScalarPotentialLMSARFMassLN, SPLMSARFMassLNConfig,
        )
        ck = torch.load(self.ckpt_path, map_location=self.device,
                        weights_only=False)
        cfg_dict = dict(ck["model_cfg"])
        if cfg_dict.get("mass_mode") == "logfreq":
            if self.logfreq_path is None:
                raise ValueError(
                    "logfreq mass_mode requires --logfreq <path>"
                )
            cfg_dict["logfreq_path"] = str(self.logfreq_path)
        cfg = SPLMSARFMassLNConfig(**cfg_dict)
        model = ScalarPotentialLMSARFMassLN(cfg).to(self.device)
        model.load_state_dict(ck["model_state_dict"])
        model.eval()
        self.model = model
        self.cfg = cfg

    @property
    def L(self) -> int:
        return self.cfg.L

    @property
    def d(self) -> int:
        return self.cfg.d

    def integrate_with_capture(
        self, x: torch.Tensor,
    ) -> Dict[str, np.ndarray]:
        m = self.model
        cfg = self.cfg
        from model_sarf_mass import causal_cumulative_mean
        with torch.enable_grad():
            emb = m._embed(x)
            mass = m.compute_mass(x, emb)
            mass_b = mass.detach()
            mass_scalar = mass_b if mass_b.dim() < 3 else mass_b.squeeze(-1)

            h = m._project(emb) if cfg.ln_after_step else emb
            v = torch.zeros_like(h)
            gamma = m.gamma
            dt = cfg.dt

            kin_pt: List[torch.Tensor] = []
            pot_pt: List[torch.Tensor] = []

            xi0 = causal_cumulative_mean(h)
            kin_pt.append(
                (0.5 * mass_scalar * (v * v).sum(dim=-1)).detach()
            )
            with torch.no_grad():
                pot_pt.append(m.V_theta(xi0, h).squeeze(-1).detach())

            for _ in range(cfg.L):
                xi = causal_cumulative_mean(h)
                h_in = h.detach().clone().requires_grad_(True)
                xi_d = xi.detach()
                V = m.V_theta(xi_d, h_in).sum()
                grad_V, = torch.autograd.grad(V, h_in, retain_graph=False)
                f = -grad_V
                v = (v + dt * f / mass_b) / (1.0 + dt * gamma)
                h_new = h_in + dt * v
                if cfg.ln_after_step:
                    h_new = m._project(h_new)
                h = h_new
                kin_pt.append(
                    (0.5 * mass_scalar * (v * v).sum(dim=-1)).detach()
                )
                with torch.no_grad():
                    xi_after = causal_cumulative_mean(h)
                    pot_pt.append(m.V_theta(xi_after, h).squeeze(-1).detach())

        kin_arr = torch.stack(kin_pt, dim=0).squeeze(1).cpu().numpy()
        pot_arr = torch.stack(pot_pt, dim=0).squeeze(1).cpu().numpy()
        return dict(kinetic_per_token=kin_arr, potential_per_token=pot_arr)


VARIANTS: Dict[str, type] = {
    "euler": EulerAdapter,
    "sarfmass": SarfMassAdapter,
    "symplectic": SymplecticAdapter,
    "em_ln": EmLnAdapter,
}


# ---------------------------------------------------------------------------

def _gather_corpus(n_test_per_domain: int, seed: int) -> List[Tuple[str, str, str]]:
    rng = np.random.default_rng(seed)
    out: List[Tuple[str, str, str]] = []
    for domain, sents in CORPUS.items():
        idx = np.arange(len(sents))
        rng.shuffle(idx)
        for i in idx[:n_test_per_domain]:
            out.append((sents[i], domain, "test"))
        for i in idx[n_test_per_domain:]:
            out.append((sents[i], domain, "train"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS.keys()))
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--out_npz", required=True, type=Path)
    ap.add_argument("--label", required=True, type=str,
                    help="Short label for this run (used in plots).")
    ap.add_argument("--logfreq", type=Path, default=None,
                    help="Path to logfreq_surprisal.npy "
                         "(required for sarfmass/symplectic when mass_mode=logfreq).")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--n_test_per_domain", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = args.device or _pick_device()
    print(f"[extract] variant={args.variant}  ckpt={args.ckpt}  device={device}")

    AdapterCls = VARIANTS[args.variant]
    adapter = AdapterCls(
        ckpt_path=args.ckpt, device=device, logfreq_path=args.logfreq,
    )
    adapter.load()
    L = adapter.L
    d = adapter.d
    print(f"[extract] loaded {args.variant}  L={L}  d={d}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    sentences = _gather_corpus(args.n_test_per_domain, args.seed)
    print(f"[extract] {len(sentences)} sentences from {len(CORPUS)} domains.")

    n_sent = len(sentences)
    T_lens: List[int] = []
    kin_per_layer: List[np.ndarray] = []
    pot_per_layer: List[np.ndarray] = []
    domains: List[str] = []
    splits: List[str] = []

    t0 = time.time()
    for k, (sent, domain, split) in enumerate(sentences):
        ids = tokenizer.encode(sent)[:args.max_len]
        x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        out = adapter.integrate_with_capture(x)
        kin_pt = out["kinetic_per_token"]      # (L+1, T)
        pot_pt = out["potential_per_token"]    # (L+1, T)
        kin_per_layer.append(kin_pt)
        pot_per_layer.append(pot_pt)
        T_lens.append(kin_pt.shape[1])
        domains.append(domain)
        splits.append(split)
        if (k + 1) % 10 == 0 or k + 1 == n_sent:
            elapsed = time.time() - t0
            print(f"  [{k+1:3d}/{n_sent}] T={kin_pt.shape[1]:3d}  "
                  f"H_0={pot_pt[0].mean():.4f}  H_L={(kin_pt[-1]+pot_pt[-1]).mean():.4f}  "
                  f"elapsed {elapsed:.1f}s")

    T_max = max(T_lens)
    K_arr = np.full((n_sent, L + 1, T_max), np.nan, dtype=np.float32)
    P_arr = np.full((n_sent, L + 1, T_max), np.nan, dtype=np.float32)
    for i, (k, p) in enumerate(zip(kin_per_layer, pot_per_layer)):
        K_arr[i, :, : k.shape[1]] = k
        P_arr[i, :, : p.shape[1]] = p

    kinetic = np.nanmean(K_arr, axis=2)        # (n_sent, L+1)
    potential = np.nanmean(P_arr, axis=2)
    H = kinetic + potential

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "label": args.label,
        "variant": args.variant,
        "ckpt": str(args.ckpt),
        "L": int(L),
        "d": int(d),
        "n_sent": int(n_sent),
        "n_test_per_domain": int(args.n_test_per_domain),
        "max_len": int(args.max_len),
        "seed": int(args.seed),
        "T_lens": [int(t) for t in T_lens],
        "domains": domains,
        "splits": splits,
    }
    np.savez_compressed(
        args.out_npz,
        kinetic=kinetic.astype(np.float32),
        potential=potential.astype(np.float32),
        H=H.astype(np.float32),
        kinetic_per_token=K_arr,
        potential_per_token=P_arr,
        T_lens=np.asarray(T_lens, dtype=np.int32),
        meta=np.asarray(json.dumps(meta), dtype=object),
    )
    print(f"[extract] saved -> {args.out_npz}  "
          f"({n_sent} sent x {L+1} layers x up to {T_max} tokens)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
