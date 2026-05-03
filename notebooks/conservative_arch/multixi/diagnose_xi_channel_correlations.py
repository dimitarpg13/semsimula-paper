"""
R6.c diagnostic — empirical pairwise channel correlation on a trained SPLM ckpt.

Background
----------
Section 5.4 of `docs/Reducing_Information_Bottleneck_In_Multi-Channel_Xi_SPLM.md`
predicts, under the Gaussian-stationary approximation:

  - K-EMA at α = (0, 0.5, 0.9, 0.99): pairwise correlation ≈ 0.94 (adjacent),
    mean off-diagonal |corr| ≈ 0.5.
  - HiPPO-LegT (orthogonal Legendre basis): pairwise correlation ≈ 0,
    mean off-diagonal |corr| ≈ 0.

The smoke-test on **white noise** measured 0.481 (K-EMA) vs 0.214 (HiPPO),
qualitatively in the predicted direction. This script measures the same
quantity on **real trained-network ξ trajectories** — i.e. what V_θ
actually sees during training. The §5.4 bound is informative iff this
empirical correlation stays close to its white-noise value; if learned
hidden states pull HiPPO channels into correlation (or, conversely, leave
K-EMA channels less redundant than predicted), the bound is uninformative.

What this script does
---------------------
1. Loads a SPLM ckpt (HiPPO or K-EMA), dispatching by config shape.
2. Runs the forward pass on `--n-batches` of TinyStories validation data
   with `return_xi_trajectory=True`, collecting ξ at every integrator step.
3. Computes the K × K pairwise Pearson correlation matrix on the
   final-integrator-step ξ (i.e. what V_θ saw at the last conservative-flow
   step).
4. Prints the correlation matrix, mean off-diagonal |corr|, and the
   "effective channel count" K_eff = K - log(det(R)) / log(2) where R is
   the correlation matrix (≈ K under perfect orthogonality, ≈ 1 under
   perfect correlation).
5. Optionally saves a JSON report.

Usage
-----
  python3 notebooks/conservative_arch/multixi/diagnose_xi_channel_correlations.py \
      <ckpt_path> [--n-batches N] [--output-json PATH]

Reference
---------
- docs/Reducing_Information_Bottleneck_In_Multi-Channel_Xi_SPLM.md §5.4, §10.
- docs/Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md §4.6 (K-EMA pilot ckpt).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

THIS_DIR = Path(__file__).parent
PARENT = THIS_DIR.parent
for sub in ("", "energetic_minima", "sarf_mass_variant", "multixi", "scaleup"):
    sys.path.insert(0, str(PARENT / sub))

# Defer model imports until ckpt loaded — needed config-class match.


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _instantiate_from_ckpt(
    ckpt_path: Path,
    device: str,
) -> Tuple[torch.nn.Module, dict, str]:
    """Load a multi-ξ SPLM ckpt and return (model, model_cfg_dict, label)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ck.get("model_cfg") or ck.get("model_config")
    if cfg_dict is None:
        raise ValueError(f"ckpt {ckpt_path} has no 'model_cfg' field")
    variant = ck.get("variant", "")

    # Dispatch order: S4D variant-tag first (most-specific), then HiPPO
    # (xi_basis), then K-EMA (xi_alpha_inits / xi_alphas). S4D and HiPPO
    # ckpts both populate `xi_channels` / `xi_theta`, so we must distinguish
    # by either the `variant` tag or the S4D-specific `xi_eigval_init` field.
    is_s4d = (
        variant == "sarf_mass_ln_multixi_s4d"
        or "xi_eigval_init" in cfg_dict
    )
    if is_s4d:
        from model_multixi_s4d import (
            ScalarPotentialLMSARFMassLNMultiS4D,
            SPLMSARFMassLNMultiS4DConfig,
        )
        keep = {k: v for k, v in cfg_dict.items()
                if k in SPLMSARFMassLNMultiS4DConfig.__dataclass_fields__}
        keep["causal_force"] = True
        cfg = SPLMSARFMassLNMultiS4DConfig(**keep)
        model = ScalarPotentialLMSARFMassLNMultiS4D(cfg)
        label = (
            f"S4D (K={cfg.xi_channels}, init={cfg.xi_eigval_init}, "
            f"theta_init={cfg.xi_theta}, "
            f"learnable_dt={cfg.xi_learnable_dt}, "
            f"learnable_B={cfg.xi_learnable_B})"
        )
    elif "xi_basis" in cfg_dict:
        from model_multixi_hippo import (
            ScalarPotentialLMSARFMassLNMultiHiPPO,
            SPLMSARFMassLNMultiHiPPOConfig,
        )
        keep = {k: v for k, v in cfg_dict.items()
                if k in SPLMSARFMassLNMultiHiPPOConfig.__dataclass_fields__}
        keep["causal_force"] = True
        cfg = SPLMSARFMassLNMultiHiPPOConfig(**keep)
        model = ScalarPotentialLMSARFMassLNMultiHiPPO(cfg)
        learn_tag = (
            f", learnable_dt={cfg.xi_learnable_dt}"
            if hasattr(cfg, "xi_learnable_dt") else ""
        )
        label = (
            f"HiPPO-{cfg.xi_basis} (K={cfg.xi_channels}, "
            f"theta={cfg.xi_theta if cfg.xi_basis == 'legt' else 'n/a'}"
            f"{learn_tag})"
        )
    elif "xi_alpha_inits" in cfg_dict or "xi_alphas" in cfg_dict:
        from model_multixi import (
            ScalarPotentialLMSARFMassLNMultiXi,
            SPLMSARFMassLNMultiXiConfig,
        )
        keep = {k: v for k, v in cfg_dict.items()
                if k in SPLMSARFMassLNMultiXiConfig.__dataclass_fields__}
        keep["causal_force"] = True
        cfg = SPLMSARFMassLNMultiXiConfig(**keep)
        model = ScalarPotentialLMSARFMassLNMultiXi(cfg)
        alphas = ck.get("final_xi_alphas", cfg.xi_alpha_inits)
        label = (
            f"K-EMA (K={cfg.xi_channels}, "
            f"α=[{', '.join(f'{a:.3f}' for a in alphas)}])"
        )
    else:
        raise ValueError(
            f"ckpt {ckpt_path} ({variant=!r}) is not a multi-ξ SPLM"
        )

    model.load_state_dict(ck["model_state_dict"])
    model = model.to(device).eval()
    return model, cfg_dict, label


def _collect_xi_trajectory(
    model: torch.nn.Module,
    val_ids: np.ndarray,
    n_batches: int,
    batch_size: int,
    block_size: int,
    rng: np.random.Generator,
    device: str,
) -> torch.Tensor:
    """Run the model on n_batches of validation data and stack the
    integrator-final ξ tensors. Returns a (∑B, T, K, d) tensor on CPU.
    """
    from data_module import get_batch  # noqa: E402

    chunks = []
    for i in range(n_batches):
        xb, _ = get_batch(val_ids, batch_size, block_size, rng)
        x = torch.from_numpy(xb).to(device)
        with torch.enable_grad():
            out = model(x, None,
                        return_xi_trajectory=True)
        # `forward` returns (logits, loss, [traj_h], [traj_xi]) — variable layout.
        # We requested return_xi_trajectory only.
        traj_xi = out[-1]
        # traj_xi is a list of L tensors of shape (B, T, K, d).
        # Take the final integrator step (matches what V_θ saw last).
        last = traj_xi[-1].detach().cpu()
        chunks.append(last)
    return torch.cat(chunks, dim=0)


def _channel_corr(xi: torch.Tensor) -> Tuple[torch.Tensor, dict]:
    """xi: (B, T, K, d) → (K x K corr matrix, summary dict).

    Summary fields:
      - mean_off_abs_corr  : mean |R[j, k]| for j ≠ k.
      - total_correlation  : TC(c) = -½ log det R (nats; redundancy under
                             the Gaussian-stationary approx).
      - k_eff_entropy_power: K_eff = exp(H(λ̃)) where λ̃_i = λ_i / Σλ are
                             the normalised eigenvalues of R. K_eff = K
                             when R = I (orthogonal); K_eff = 1 when R
                             is rank-1 (fully correlated).
      - k_eff_logdet       : K + log det R / log 2  (= K - 2·TC / log 2);
                             a different but related "effective channels"
                             definition. Quoted for cross-reference with
                             §5.4 of the design doc, which uses this form.
    """
    B, T, K, d = xi.shape
    flat = xi.reshape(-1, K, d).permute(0, 2, 1).reshape(-1, K)  # (B·T·d, K)
    flat = flat - flat.mean(dim=0, keepdim=True)
    cov = flat.T @ flat / max(flat.shape[0] - 1, 1)
    std = cov.diag().sqrt().clamp(min=1e-12)
    corr = cov / (std[:, None] * std[None, :])
    off_mask = ~torch.eye(K, dtype=torch.bool)
    mean_off = float(corr[off_mask].abs().mean().item())

    # Eigen-decomposition for entropy-power K_eff.
    eigvals = torch.linalg.eigvalsh(corr).clamp(min=1e-12)
    p = eigvals / eigvals.sum()
    h = -(p * torch.log(p)).sum()
    k_eff_ep = float(torch.exp(h).item())

    sign, logdet = torch.slogdet(corr)
    if sign.item() <= 0 or torch.isnan(logdet) or torch.isinf(logdet):
        tc = float("inf")
        k_eff_ld = 1.0
    else:
        tc = -0.5 * float(logdet.item())
        # K_eff_logdet = K + logdet/log(2) ; equivalent to K - 2 TC / log 2
        k_eff_ld = float(K) + float(logdet.item()) / math.log(2.0)

    return corr, {
        "mean_off_abs_corr": mean_off,
        "total_correlation_nats": tc,
        "k_eff_entropy_power": k_eff_ep,
        "k_eff_logdet": k_eff_ld,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ckpt", help="Path to a multi-ξ SPLM .pt checkpoint.")
    ap.add_argument(
        "--n-batches", type=int, default=10,
        help="Number of validation batches to collect ξ over. Default 10.",
    )
    ap.add_argument(
        "--batch-size", type=int, default=8,
        help="Validation batch size. Default 8 (memory-friendly on MPS).",
    )
    ap.add_argument(
        "--block-size", type=int, default=512,
        help="Sequence length per batch. Default 512.",
    )
    ap.add_argument(
        "--max-train-tokens", type=int, default=5_000_000,
        help="Cache key for load_tiny_stories. Default 5_000_000.",
    )
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--output-json", default=None,
        help="Optional path to dump the diagnostic report as JSON.",
    )
    args = ap.parse_args()

    device = args.device or _pick_device()
    ckpt_path = Path(args.ckpt).resolve()
    if not ckpt_path.exists():
        print(f"[diag] ckpt not found: {ckpt_path}", file=sys.stderr)
        return 2

    print(f"[diag] loading: {ckpt_path}")
    model, cfg_dict, label = _instantiate_from_ckpt(ckpt_path, device)
    print(f"[diag] matched: {label}")

    from data_module import load_tiny_stories  # noqa: E402
    _, val_ids = load_tiny_stories(max_train_tokens=args.max_train_tokens)
    print(f"[diag] val tokens: {len(val_ids):,}  device: {device}")

    rng = np.random.default_rng(args.seed)
    print(
        f"[diag] collecting ξ at integrator step L "
        f"on {args.n_batches} batches × B={args.batch_size} × "
        f"T={args.block_size} = "
        f"{args.n_batches * args.batch_size * args.block_size:,} samples..."
    )
    xi = _collect_xi_trajectory(
        model, val_ids, args.n_batches,
        args.batch_size, args.block_size, rng, device,
    )
    K = int(xi.shape[2])
    corr, summary = _channel_corr(xi)

    np.set_printoptions(precision=3, suppress=True)
    print(
        f"\n[diag] {label}\n"
        f"  ξ trajectory shape: {tuple(xi.shape)}  (B_total, T, K, d)\n"
        f"  K = {K}    samples per channel = {xi.shape[0]*xi.shape[1]*xi.shape[3]:,}\n"
    )
    print("  Pairwise Pearson correlation matrix R[j, k]:")
    print("  ", str(corr.cpu().numpy()).replace("\n", "\n   "))
    print()
    print(f"  mean |off-diagonal corr|        = {summary['mean_off_abs_corr']:.4f}")
    print(f"  total correlation TC(c) (nats)  = {summary['total_correlation_nats']:.4f}")
    print(f"  K_eff (entropy power)           = {summary['k_eff_entropy_power']:.3f}   "
          f"(K = {K} ⇔ orthogonal,  K = 1 ⇔ rank-1)")
    print(f"  K_eff (logdet form, §5.4)       = {summary['k_eff_logdet']:.3f}")

    if args.output_json:
        out = {
            "ckpt": str(ckpt_path),
            "label": label,
            "K": K,
            "n_samples": int(xi.shape[0] * xi.shape[1] * xi.shape[3]),
            "corr_matrix": corr.cpu().numpy().tolist(),
            **summary,
        }
        Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"\n[diag] JSON report saved to {args.output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
