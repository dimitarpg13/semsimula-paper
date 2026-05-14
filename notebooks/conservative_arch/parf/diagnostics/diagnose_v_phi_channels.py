"""
Channel diagnostic for the §5.1-faithful structural V_phi (Eq. (131)).

Purpose
-------
The Stage-1 PARF cells (P1 dense val PPL 210.5, P1.6 dense wider 207.6,
P5 sparse k=4 val PPL 176.7) localise the binding constraint on dense
PARF to the aggregation regime, not to V_phi capacity.  The natural
follow-up is to inspect what each multiplicative channel of

    V_phi^struct(h_t, h_s)
        = - C * Theta_phi(theta(h_t), theta(h_s))
              * Phi_phi(l(h_t), l(h_s))
              / sqrt(||h_t - h_s||^2 + eps^2)

actually does at convergence.  This script loads a trained PARF
checkpoint and emits, per layer, the empirical distributions of:

    1. ||h_t - h_s||           pairwise hidden distance
    2. Phi_phi(l_t, l_s)        type-gate (saturation toward 1?)
    3. Theta_phi(theta_t, theta_s)  value-aligner sign distribution
    4. |V_phi(h_t, h_s)|        per-pair potential magnitude
    5. signed V_phi(h_t, h_s)   per-pair signed potential
                                (destructive cancellation across s?)

plus a per-layer scalar:

    R(ell) = ||grad_h sum_{s<t} V_phi||  /  ||grad_h V_theta||

which quantifies whether the pair force is a perturbation on the
SPLM single-particle force (R << 1) or comparable in scale (R ~ 1).

Usage
-----
    python diagnose_v_phi_channels.py \
        --ckpt path/to/parf_..._ckpt_latest.pt \
        --out  path/to/output_dir/

    python diagnose_v_phi_channels.py --all
        # runs against every PARF checkpoint under parf/results/ and
        # writes results to parf/diagnostics/results/<tag>/.

The script is tolerant of both PARFLM and SparsePARFLM checkpoints;
for SparsePARFLM it bypasses the score head and runs V_phi at all
O(T^2) pairs, so the histograms are of the underlying potential
(which is what we want to diagnose), not of the post-mask aggregation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).parent
PARF_DIR = SCRIPT_DIR.parent
PARENT_DIR = PARF_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(PARENT_DIR))
from data_module import get_batch, load_tiny_shakespeare  # noqa: E402

sys.path.insert(0, str(PARF_DIR))
from model_parf import (  # noqa: E402
    PARFConfig,
    PARFLM,
    StructuralCompetitiveVPhi,
    StructuralVPhi,
    causal_cumulative_mean,
)
from model_parf_sparse import SparsePARFConfig, SparsePARFLM  # noqa: E402


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def _filter_cfg(cls, cfg_dict: dict) -> dict:
    """Return only the fields of `cfg_dict` that are dataclass fields of cls."""
    import dataclasses

    fields = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in cfg_dict.items() if k in fields}


def load_ckpt(
    ckpt_path: Path,
    device: str,
    logfreq_path_override: Path | None = None,
) -> Tuple[nn.Module, dict, str]:
    """Reconstruct the PARFLM (or SparsePARFLM) from a training checkpoint.

    Returns (model, model_cfg_dict, variant_tag).

    `logfreq_path_override`: if given, replaces the `logfreq_path` field of the
    saved cfg before model instantiation. Use this when a Colab-trained ckpt
    baked in an absolute `/content/...` surprisal path that does not exist
    locally; pass the local equivalent instead.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = dict(ckpt["model_cfg"])  # mutable copy
    if logfreq_path_override is not None:
        cfg_dict["logfreq_path"] = str(logfreq_path_override)
    variant = ckpt.get("variant", "parf_q9c")

    is_sparse = (
        "sparse" in variant
        or "top_k" in cfg_dict
        or "score_head_hidden" in cfg_dict
    )
    if is_sparse:
        cfg_kw = _filter_cfg(SparsePARFConfig, cfg_dict)
        cfg = SparsePARFConfig(**cfg_kw)
        model = SparsePARFLM(cfg)
    else:
        cfg_kw = _filter_cfg(PARFConfig, cfg_dict)
        cfg = PARFConfig(**cfg_kw)
        model = PARFLM(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, cfg_dict, variant


# ---------------------------------------------------------------------------
# Internals capture for StructuralVPhi
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_v_phi_internals(
    v_phi: StructuralVPhi,
    h: torch.Tensor,
    h_src: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Mirror StructuralVPhi.forward step-by-step and return intermediates.

    Handles both `StructuralVPhi` (unnormalised Gaussian Φ) and
    `StructuralCompetitiveVPhi` (softmax-normalised Φ̃, Lever 3) by
    branching on the dynamic type of `v_phi`.

    Used under no_grad: we only need the values of the channels, not their
    gradient.  All returned tensors have batch+pair shape (B, T, T) except
    the projections (B, T, *) which are not directly returned.

    Returns dict with keys (all detached, on the input device):
        l_dist2 : (B, T, T)  squared type distance
        c       : (B, T, T)  learned per-pair inverse bandwidth
        Phi     : (B, T, T)  type-gate value (raw Gaussian for base class,
                              softmax-normalised + rescaled for competitive)
        Theta   : (B, T, T)  value-aligner output, in [-1, 1]
        h_dist2 : (B, T, T)  squared hidden-state distance
        r       : (B, T, T)  hidden distance with Plummer softening
        V_phi   : (B, T, T)  full V_phi value (signed)
    """
    B, T, _ = h.shape

    l_q = v_phi.W_l(h)
    l_s = v_phi.W_l(h_src)
    th_q = v_phi.W_theta(h)
    th_s = v_phi.W_theta(h_src)

    l_dist2 = StructuralVPhi._pair_dist2(l_q, l_s)
    c = F.softplus(v_phi.phi_c_net(l_dist2.unsqueeze(-1)).squeeze(-1))

    if isinstance(v_phi, StructuralCompetitiveVPhi):
        # Competitive Φ̃: softmax over s of -c·d²/τ, masked to s < t,
        # then rescaled per cfg.v_phi_competitive_scale.
        tau = max(float(v_phi.competitive_temp), 1e-6)
        logit = -(c * l_dist2) / tau
        causal = torch.tril(
            torch.ones(T, T, device=logit.device, dtype=torch.bool),
            diagonal=-1,
        )
        logit = logit.masked_fill(~causal[None, ...], -1e9)
        Phi = torch.softmax(logit, dim=-1)
        row_has_valid = causal.any(dim=-1)
        Phi = Phi * row_has_valid[None, :, None].to(Phi.dtype)
        if v_phi.competitive_scale == "row":
            row_count = causal.sum(dim=-1).to(Phi.dtype)
            Phi = Phi * row_count[None, :, None]
        # 'mean' / 'none' need no further rescale.
    else:
        Phi = torch.exp(-c * l_dist2)

    # P8 patch D: bilinear Θ vs MLP Θ (instance-attr dispatch).
    theta_form = getattr(v_phi, "theta_form", "mlp")
    if theta_form == "bilinear":
        tmp = th_q @ v_phi.theta_W
        score = tmp @ th_s.transpose(-2, -1) + v_phi.theta_b
    else:
        proj_q = v_phi.theta_w_q(th_q)
        proj_s = v_phi.theta_w_s(th_s)
        proj_qd = v_phi.theta_w_d(th_q)
        proj_sd = v_phi.theta_w_d(th_s)
        proj_t = proj_q + proj_qd + v_phi.theta_b1
        proj_u = proj_s - proj_sd
        hidden = proj_t.unsqueeze(2) + proj_u.unsqueeze(1)
        hidden = F.gelu(hidden)
        score = v_phi.theta_w2(hidden).squeeze(-1)
    # P8 patch C: softsign vs tanh (instance-attr dispatch).
    theta_act = getattr(v_phi, "theta_activation", "tanh")
    Theta = F.softsign(score) if theta_act == "softsign" else torch.tanh(score)

    # P8 patch A: LN-before-distance (instance-attr dispatch).
    if getattr(v_phi, "ln_before_distance", False):
        h_for_dist = F.layer_norm(h, (h.shape[-1],))
        hs_for_dist = F.layer_norm(h_src, (h_src.shape[-1],))
    else:
        h_for_dist = h
        hs_for_dist = h_src
    h_dist2 = StructuralVPhi._pair_dist2(h_for_dist, hs_for_dist)
    r = torch.sqrt(h_dist2 + v_phi.eps2)

    V_phi = -v_phi.C * Theta * Phi / r

    return {
        "l_dist2": l_dist2.detach(),
        "c": c.detach(),
        "Phi": Phi.detach(),
        "Theta": Theta.detach(),
        "h_dist2": h_dist2.detach(),
        "r": r.detach(),
        "V_phi": V_phi.detach(),
    }


# ---------------------------------------------------------------------------
# Per-layer gradient ratio (uses autograd, lives outside no_grad)
# ---------------------------------------------------------------------------
def per_layer_force_norms(
    model: nn.Module,
    h_layer: torch.Tensor,
    layer_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (theta_force_norm, pair_force_norm) per (B, T) for one layer.

    Mirrors the V_theta and V_phi gradient calls of `_layer_step`, but
    splits them so we can measure their magnitudes independently.  No
    integrator update is applied; this is purely a force-magnitude probe.

    When the model has the P8 per-layer V_φ scale enabled, the returned
    `pair_force_norm` is multiplied by softplus(σ_ℓ) so that R(ℓ)
    reflects the *effective* pair-force contribution that the model
    integrator actually feels.
    """
    cfg = model.cfg
    h_in = h_layer.detach().clone().requires_grad_(True)
    h_src = h_in.detach() if cfg.causal_force else h_in

    xi_input = h_in.detach() if cfg.causal_force else h_in
    xi_now = causal_cumulative_mean(xi_input)

    # V_theta: scalar per token, gradient w.r.t. h_in.
    V_th = model.V_theta(xi_now, h_in)  # (B, T, 1)
    grad_theta, = torch.autograd.grad(
        V_th.sum(), h_in, retain_graph=True, create_graph=False,
    )
    theta_force_norm = grad_theta.norm(dim=-1)  # (B, T)

    # V_phi pair sum, masked strict lower triangular (s < t), gradient
    # w.r.t. h_in only (h_src is detached).
    P = model.V_phi(h_in, h_src)  # (B, T, T)
    T_ = P.shape[1]
    mask = torch.tril(
        torch.ones(T_, T_, device=P.device, dtype=torch.bool), diagonal=-1
    )
    P_masked = P.masked_fill(~mask, 0.0)
    grad_pair, = torch.autograd.grad(
        P_masked.sum(), h_in, retain_graph=False, create_graph=False,
    )
    pair_force_norm = grad_pair.norm(dim=-1)  # (B, T)

    # Apply the P8 per-layer scale if active (mirrors `_layer_step`).
    if getattr(model, "raw_v_phi_scale", None) is not None:
        s_ell = F.softplus(model.raw_v_phi_scale[layer_idx]).detach()
        pair_force_norm = pair_force_norm * s_ell

    return theta_force_norm.detach(), pair_force_norm.detach()


# ---------------------------------------------------------------------------
# Causal-mask helpers
# ---------------------------------------------------------------------------
def _strict_lower_indices(T: int) -> torch.Tensor:
    """Bool mask of shape (T, T) with True for s < t (the causal pair slots)."""
    return torch.tril(torch.ones(T, T, dtype=torch.bool), diagonal=-1)


# ---------------------------------------------------------------------------
# Run the diagnostic on one checkpoint
# ---------------------------------------------------------------------------
def diagnose(
    ckpt_path: Path,
    out_dir: Path,
    n_batches: int = 4,
    batch_size: int = 16,
    block_size: int = 128,
    seed: int = 0,
    device: Optional[str] = None,
    logfreq_path_override: Optional[Path] = None,
) -> Dict[str, object]:
    """Load a PARF checkpoint and emit per-layer channel diagnostics.

    Aggregates over `n_batches` Tiny-Shakespeare val batches (typically
    4 x 16 x 128 = 8,192 tokens; ~520 K causal pairs at L=8).  Writes
    out_dir/{channels.png, gradient_ratio.png, summary.json,
    summary.md}.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"[diag] loading checkpoint -> {ckpt_path}")
    model, cfg_dict, variant = load_ckpt(
        ckpt_path, device, logfreq_path_override=logfreq_path_override,
    )
    cfg = model.cfg
    L, T_block, B = cfg.L, block_size, batch_size

    if not isinstance(model.V_phi, StructuralVPhi):
        raise NotImplementedError(
            f"diagnose() currently only supports StructuralVPhi-family "
            f"models (StructuralVPhi, StructuralCompetitiveVPhi); got "
            f"{type(model.V_phi).__name__} for ckpt {ckpt_path.name}."
        )

    # -------- data --------
    train_ids, val_ids = load_tiny_shakespeare()
    rng = np.random.default_rng(seed)

    # -------- collectors --------
    # Per-layer flat tensors on CPU; we cap at 2 M values to keep RAM sane.
    cap_per_layer = 2_000_000
    coll = {
        "h_dist": [[] for _ in range(L)],
        "Phi": [[] for _ in range(L)],
        "Theta": [[] for _ in range(L)],
        "abs_V_phi": [[] for _ in range(L)],
        "signed_V_phi": [[] for _ in range(L)],
        "c_inverse_bw": [[] for _ in range(L)],
    }
    grad_ratio_per_layer = [[] for _ in range(L)]
    theta_norm_per_layer = [[] for _ in range(L)]
    pair_norm_per_layer = [[] for _ in range(L)]
    layer_signed_sum_over_s = [[] for _ in range(L)]  # mean over t of |sum_s V_phi|

    causal_mask_T = _strict_lower_indices(T_block).to(device)

    for batch_idx in range(n_batches):
        xb, yb = get_batch(val_ids, B, T_block, rng)
        x = torch.from_numpy(xb).to(device)

        # Get the full per-layer trajectory; this runs the actual integrator
        # but with create_graph=False at eval-time, so it's cheap.
        with torch.enable_grad():
            _, _, traj = model(x, return_trajectory=True)

        # traj is a list of L+1 tensors of shape (B, T, d) on CPU.
        for ell in range(L):
            h_in_layer = traj[ell].to(device)  # input to layer (ell+1)

            # ----- channel internals (no_grad) -----
            internals = compute_v_phi_internals(
                model.V_phi, h_in_layer, h_in_layer.detach()
            )
            mask_flat = causal_mask_T  # (T, T) bool
            for key_src, key_dst, transform in [
                ("h_dist2", "h_dist", lambda x: x.sqrt()),
                ("Phi", "Phi", lambda x: x),
                ("Theta", "Theta", lambda x: x),
                ("V_phi", "abs_V_phi", lambda x: x.abs()),
                ("V_phi", "signed_V_phi", lambda x: x),
                ("c", "c_inverse_bw", lambda x: x),
            ]:
                vals = transform(internals[key_src])  # (B, T, T)
                # Apply causal mask: only s < t pairs.
                masked = vals.masked_select(mask_flat[None].expand_as(vals))
                # Subsample if too many.
                if masked.numel() > cap_per_layer // n_batches:
                    idx = torch.randperm(
                        masked.numel(), device=masked.device
                    )[: cap_per_layer // n_batches]
                    masked = masked[idx]
                coll[key_dst][ell].append(masked.cpu())

            # ----- per-token signed sum over s, then |.|, then mean over t -----
            V_phi_signed = internals["V_phi"]  # (B, T, T)
            # Causal sum over s: only s < t.  At t=0 the sum is empty -> 0.
            V_phi_signed_masked = V_phi_signed.masked_fill(~mask_flat[None], 0.0)
            sum_over_s = V_phi_signed_masked.sum(dim=-1)  # (B, T)
            layer_signed_sum_over_s[ell].append(sum_over_s.abs().mean().cpu())

            # ----- gradient ratio -----
            # Reuse h_in_layer.  Build a fresh leaf for each layer.
            theta_force, pair_force = per_layer_force_norms(
                model, h_in_layer, layer_idx=ell,
            )
            theta_norm_per_layer[ell].append(theta_force.mean().cpu())
            pair_norm_per_layer[ell].append(pair_force.mean().cpu())
            ratio = pair_force / (theta_force + 1e-12)
            grad_ratio_per_layer[ell].append(ratio.mean().cpu())

        # Free the trajectory CPU tensors (they hold (B, T, d) per layer).
        del traj

    # -------- aggregate / summarise --------
    summary: Dict[str, object] = {
        "checkpoint": str(ckpt_path),
        "variant": variant,
        "v_phi_kind": cfg.v_phi_kind,
        "L": L,
        "block_size": T_block,
        "n_batches": n_batches,
        "batch_size": B,
        "device": device,
        "model_cfg": cfg_dict,
        "per_layer_stats": [],
        "global_stats": {},
    }
    for ell in range(L):
        layer_stats = {"layer": ell + 1}  # 1-indexed for human readability
        for key in [
            "h_dist", "Phi", "Theta",
            "abs_V_phi", "signed_V_phi", "c_inverse_bw",
        ]:
            cat = torch.cat(coll[key][ell]).numpy()
            layer_stats[key] = {
                "mean": float(np.mean(cat)),
                "std": float(np.std(cat)),
                "median": float(np.median(cat)),
                "p05": float(np.percentile(cat, 5)),
                "p95": float(np.percentile(cat, 95)),
                "min": float(np.min(cat)),
                "max": float(np.max(cat)),
                "count": int(cat.size),
            }
        layer_stats["theta_force_norm_mean"] = float(
            np.mean([x.item() for x in theta_norm_per_layer[ell]])
        )
        layer_stats["pair_force_norm_mean"] = float(
            np.mean([x.item() for x in pair_norm_per_layer[ell]])
        )
        layer_stats["grad_ratio_mean"] = float(
            np.mean([x.item() for x in grad_ratio_per_layer[ell]])
        )
        layer_stats["mean_abs_signed_sum"] = float(
            np.mean([x.item() for x in layer_signed_sum_over_s[ell]])
        )
        summary["per_layer_stats"].append(layer_stats)

    # -------- plots --------
    _plot_channels(coll, out_dir / "channels.png", L, ckpt_path.stem)
    _plot_force_ratio(summary, out_dir / "gradient_ratio.png", ckpt_path.stem)

    # -------- write outputs --------
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    _write_summary_md(summary, out_dir / "summary.md", ckpt_path)

    print(f"[diag] wrote -> {out_dir}/  "
          f"(channels.png, gradient_ratio.png, summary.json, summary.md)")
    return summary


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_channels(
    coll: Dict[str, List[List[torch.Tensor]]],
    out_path: Path,
    L: int,
    title_tag: str,
) -> None:
    """6-panel figure with per-layer histograms of each channel."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))
    cmap = plt.get_cmap("viridis")
    layer_colors = [cmap(ell / max(L - 1, 1)) for ell in range(L)]

    panels = [
        ("h_dist", "$\\|h_t - h_s\\|$  (hidden distance)", axes[0, 0], "linear"),
        ("Phi", "$\\Phi_\\phi(l_t, l_s) \\in [0, 1]$  (type-gate)", axes[0, 1], "linear"),
        ("Theta", "$\\Theta_\\phi(\\theta_t, \\theta_s) \\in [-1, 1]$  (value-aligner, signed)",
         axes[0, 2], "linear"),
        ("c_inverse_bw", "$c$ (learned inv-bandwidth, softplus-positive)",
         axes[1, 0], "log"),
        ("abs_V_phi", "$|V_\\phi(h_t, h_s)|$  (per-pair magnitude)",
         axes[1, 1], "log"),
        ("signed_V_phi", "$V_\\phi(h_t, h_s)$  (signed; AR cancellation?)",
         axes[1, 2], "linear"),
    ]
    for key, label, ax, yscale in panels:
        for ell in range(L):
            data = torch.cat(coll[key][ell]).numpy()
            # Robust binning per panel: clip to [p1, p99] of layer 0 values
            # to keep the x-axis stable across layers.
            if ell == 0:
                lo, hi = np.percentile(data, [1, 99])
                if lo == hi:
                    lo, hi = data.min(), data.max() + 1e-6
            ax.hist(
                data,
                bins=80,
                range=(lo, hi),
                histtype="step",
                color=layer_colors[ell],
                alpha=0.85,
                label=f"L{ell+1}" if ell in {0, L // 2, L - 1} else None,
            )
        ax.set_title(label)
        ax.set_xlabel(label.split("$")[1] if "$" in label else label)
        ax.set_ylabel("count")
        ax.set_yscale(yscale)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"PARF V_phi channel distributions  ({title_tag})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_force_ratio(
    summary: dict,
    out_path: Path,
    title_tag: str,
) -> None:
    """Per-layer bar/line plot of grad ratio + force norms."""
    L = summary["L"]
    layers = np.arange(1, L + 1)
    th_norm = [s["theta_force_norm_mean"] for s in summary["per_layer_stats"]]
    pair_norm = [s["pair_force_norm_mean"] for s in summary["per_layer_stats"]]
    ratio = [s["grad_ratio_mean"] for s in summary["per_layer_stats"]]
    abs_sum = [s["mean_abs_signed_sum"] for s in summary["per_layer_stats"]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    ax.plot(layers, th_norm, marker="o", label="$\\|\\nabla_h V_\\theta\\|$")
    ax.plot(layers, pair_norm, marker="s",
            label="$\\|\\nabla_h \\sum V_\\phi\\|$")
    ax.set_xlabel("layer")
    ax.set_ylabel("mean per-token force norm")
    ax.set_yscale("log")
    ax.set_title("Force-norm magnitudes per layer")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.bar(layers, ratio, color="C2", alpha=0.85)
    ax.axhline(1.0, color="black", linestyle=":", alpha=0.5)
    ax.set_xlabel("layer")
    ax.set_ylabel("$R(\\ell) = \\|\\nabla V_\\phi\\| / \\|\\nabla V_\\theta\\|$")
    ax.set_title("Pair-vs-single force-norm ratio")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[2]
    ax.plot(layers, abs_sum, marker="^", color="C3")
    ax.set_xlabel("layer")
    ax.set_ylabel("$\\langle |\\sum_{s<t} V_\\phi(h_t, h_s)| \\rangle_t$")
    ax.set_yscale("log")
    ax.set_title("Per-token signed pair-sum magnitude\n(small ⇒ destructive cancellation)")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"PARF gradient and aggregation diagnostic  ({title_tag})",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _write_summary_md(summary: dict, out_path: Path, ckpt_path: Path) -> None:
    """Render a human-readable per-layer table to markdown."""
    L = summary["L"]
    lines = []
    lines.append(f"# PARF V_phi channel diagnostic — `{ckpt_path.name}`\n")
    lines.append(f"- variant: `{summary['variant']}`")
    lines.append(f"- v_phi_kind: `{summary['v_phi_kind']}`")
    lines.append(f"- L: {L}")
    lines.append(f"- batches × batch_size × block_size: "
                 f"{summary['n_batches']} × {summary['batch_size']} × {summary['block_size']}")
    lines.append("")
    lines.append("## Channel summary table\n")
    header_cols = [
        "L", "‖h_t-h_s‖ med", "Φ med", "Φ p95",
        "|Θ| med", "|V_φ| med", "Σ_s V_φ |·|",
        "‖∇V_θ‖", "‖∇ΣV_φ‖", "R(ℓ)",
    ]
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("|" + "|".join("---" for _ in header_cols) + "|")
    for s in summary["per_layer_stats"]:
        row = [
            str(s["layer"]),
            f"{s['h_dist']['median']:.3f}",
            f"{s['Phi']['median']:.3f}",
            f"{s['Phi']['p95']:.3f}",
            f"{abs(s['Theta']['median']):.3f}",
            f"{s['abs_V_phi']['median']:.3e}",
            f"{s['mean_abs_signed_sum']:.3e}",
            f"{s['theta_force_norm_mean']:.3e}",
            f"{s['pair_force_norm_mean']:.3e}",
            f"{s['grad_ratio_mean']:.3f}",
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Failure-mode read-off\n")
    lines.append(_diagnose_failure_modes(summary))
    out_path.write_text("\n".join(lines) + "\n")


def _diagnose_failure_modes(summary: dict) -> str:
    """Heuristic narrative interpretation of the per-layer table.

    The five hypotheses below correspond to the levers in
    `companion_notes/PARF_Augmented_SPLM_Architecture.md` §10.  They are
    pattern-matched on the per-layer statistics; the script is honest
    about which signals are weak (so the user knows when to inspect
    the histograms manually).
    """
    layers = summary["per_layer_stats"]
    L = summary["L"]
    out: List[str] = []

    # Phi saturation check.
    phi_med = np.array([s["Phi"]["median"] for s in layers])
    phi_p95 = np.array([s["Phi"]["p95"] for s in layers])
    phi_saturated = (phi_med.mean() > 0.85) and (phi_p95.mean() > 0.95)

    # Theta collapse check.
    abs_theta_med = np.array([abs(s["Theta"]["median"]) for s in layers])
    theta_collapsed = abs_theta_med.mean() < 0.05

    # Concentration of measure: small spread in ||h_t - h_s|| across pairs.
    h_p05 = np.array([s["h_dist"]["p05"] for s in layers])
    h_p95 = np.array([s["h_dist"]["p95"] for s in layers])
    h_med = np.array([s["h_dist"]["median"] for s in layers])
    rel_spread = (h_p95 - h_p05) / (h_med + 1e-9)
    concentration = rel_spread.mean() < 0.3

    # Destructive cancellation: per-token signed sum << per-pair magnitude * T.
    abs_v_med = np.array([s["abs_V_phi"]["median"] for s in layers])
    abs_sum = np.array([s["mean_abs_signed_sum"] for s in layers])
    # Expected magnitude if all pairs added constructively: ~ T_avg * |V_phi|.
    # Tiny Shakespeare T=128, average causal pairs per token ~ 64.
    expected_constructive = 64.0 * abs_v_med
    cancellation = (abs_sum < 0.2 * expected_constructive).mean() > 0.5

    # Force ratio: pair force should be a perturbation (R << 1) at start of
    # training; at convergence, R ~ O(1) is healthy, R >> 1 means V_phi is
    # drowning out V_theta.
    r = np.array([s["grad_ratio_mean"] for s in layers])
    pair_dominates = r.mean() > 1.5
    pair_negligible = r.mean() < 0.05

    lines = []
    if phi_saturated:
        lines.append(
            f"- **[Φ_φ saturated near 1]** Median Φ across layers is "
            f"{phi_med.mean():.3f} (p95 = {phi_p95.mean():.3f}).  The "
            f"type-gate is *not* selecting; nearly every pair contributes "
            f"close to its full distance-and-Θ value.  Lever 3 (softmax-"
            f"normalised Φ) directly targets this."
        )
    else:
        lines.append(
            f"- Φ_φ has working dynamic range "
            f"(median {phi_med.mean():.3f}, p95 {phi_p95.mean():.3f}); "
            f"selectivity is at least partially active."
        )
    if theta_collapsed:
        lines.append(
            f"- **[Θ_φ collapsed near 0]** Mean of |Θ_φ| median across "
            f"layers is {abs_theta_med.mean():.3f}; the value-aligner has "
            f"trained toward zero output, so V_φ ≈ 0 per pair.  Curriculum "
            f"on C (Lever 6) and bilinear Θ (Lever 4) target this."
        )
    else:
        lines.append(
            f"- Θ_φ retains nontrivial sign structure "
            f"(mean |Θ_φ| median {abs_theta_med.mean():.3f})."
        )
    if concentration:
        lines.append(
            f"- **[Concentration of measure on ‖h_t-h_s‖]** Relative "
            f"spread (p95-p05)/median = {rel_spread.mean():.3f} across "
            f"layers; pairs are nearly equidistant in d=128, so 1/r barely "
            f"varies.  Lever 1 (Yukawa / learned-exponent kernel) and "
            f"Lever 2 (learned distance projection) target this."
        )
    else:
        lines.append(
            f"- ‖h_t-h_s‖ has healthy spread "
            f"(rel (p95-p05)/median = {rel_spread.mean():.3f})."
        )
    if cancellation:
        lines.append(
            f"- **[Destructive cancellation across s]** Per-token signed "
            f"sum is much smaller than the per-pair magnitude × pair count; "
            f"the dense aggregation is washing out the sign.  Confirms the "
            f"P5 sparsity-helps reading.  Levers 3 (competitive Φ) and 5 "
            f"(per-layer V_φ) target this."
        )
    else:
        lines.append(
            f"- Signed pair-sum magnitudes are consistent with constructive "
            f"interference; aggregation is *not* obviously destructive."
        )
    if pair_dominates:
        lines.append(
            f"- **[V_φ force dominates V_θ]** Mean R(ℓ) = {r.mean():.3f} > "
            f"1.5; the pair force is overwhelming the SPLM single-particle "
            f"force.  Curriculum on C and reduced init scale (Lever 6) "
            f"target this."
        )
    elif pair_negligible:
        lines.append(
            f"- **[V_φ force negligible vs V_θ]** Mean R(ℓ) = {r.mean():.3f} "
            f"<< 1; the pair force is not contributing to the dynamics in a "
            f"meaningful way.  Either Θ has collapsed (see above) or C is "
            f"too small.  Lever 6 (warm-up curriculum) targets this."
        )
    else:
        lines.append(
            f"- Force-norm ratio R(ℓ) = {r.mean():.3f} is in the "
            f"perturbation-but-non-trivial regime; the pair force is "
            f"plausibly active but not dominant."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _find_all_ckpts() -> List[Path]:
    return sorted((PARF_DIR / "results").rglob("*ckpt_latest.pt"))


def _tag_for_ckpt(ckpt: Path) -> str:
    """Build a short tag for the output subdir from the checkpoint path."""
    parts = ckpt.parent.parts[-2:]
    return "_".join(parts)


def _safe_relative_to(p: Path, root: Path) -> str:
    """Return p.relative_to(root) if possible, else the absolute path."""
    p = p.resolve()
    root = root.resolve()
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=str, default=None,
                    help="Path to a single PARF checkpoint .pt file.")
    ap.add_argument("--out", type=str, default=None,
                    help="Output directory (default: parf/diagnostics/results/<tag>/).")
    ap.add_argument("--all", action="store_true",
                    help="Run diagnostic against every PARF ckpt under "
                         "parf/results/.")
    ap.add_argument("--n-batches", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument(
        "--logfreq-path", type=str, default=None,
        help=("Override the saved cfg's logfreq_path. Useful for ckpts trained "
              "on Colab (which bake in /content/... paths) being diagnosed "
              "locally."),
    )
    args = ap.parse_args()
    logfreq_override = (
        Path(args.logfreq_path).expanduser().resolve()
        if args.logfreq_path else None
    )

    targets: List[Tuple[Path, Path]] = []
    if args.all:
        for ckpt in _find_all_ckpts():
            tag = _tag_for_ckpt(ckpt)
            targets.append((ckpt, RESULTS_DIR / tag))
    else:
        if args.ckpt is None:
            raise SystemExit("--ckpt or --all is required")
        ckpt = Path(args.ckpt)
        if not ckpt.exists():
            raise SystemExit(f"checkpoint not found: {ckpt}")
        out = Path(args.out) if args.out else (RESULTS_DIR / _tag_for_ckpt(ckpt))
        targets.append((ckpt, out))

    print(f"[diag] {len(targets)} target(s):")
    for ckpt, out in targets:
        ckpt_str = _safe_relative_to(ckpt, PARF_DIR)
        out_str = _safe_relative_to(out, SCRIPT_DIR)
        print(f"        {ckpt_str} -> {out_str}")

    summaries = []
    for ckpt, out in targets:
        try:
            summary = diagnose(
                ckpt, out,
                n_batches=args.n_batches,
                batch_size=args.batch_size,
                block_size=args.block_size,
                seed=args.seed,
                device=args.device,
                logfreq_path_override=logfreq_override,
            )
            summaries.append(summary)
        except NotImplementedError as exc:
            print(f"[diag] SKIP {ckpt.name}: {exc}")

    if args.all and len(summaries) > 1:
        # Cross-checkpoint comparison plot.
        _plot_cross_ckpt(summaries, RESULTS_DIR / "cross_ckpt_summary.png")
        print(f"[diag] wrote cross-ckpt summary -> "
              f"{(RESULTS_DIR / 'cross_ckpt_summary.png').relative_to(SCRIPT_DIR)}")


def _plot_cross_ckpt(summaries: List[dict], out_path: Path) -> None:
    """Compare R(ℓ), median Φ, |Θ| median across all diagnosed checkpoints."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for s in summaries:
        L = s["L"]
        layers = np.arange(1, L + 1)
        tag = Path(s["checkpoint"]).parent.name
        ratio = [ls["grad_ratio_mean"] for ls in s["per_layer_stats"]]
        phi_med = [ls["Phi"]["median"] for ls in s["per_layer_stats"]]
        theta_med = [abs(ls["Theta"]["median"]) for ls in s["per_layer_stats"]]
        axes[0].plot(layers, ratio, marker="o", label=tag, alpha=0.85)
        axes[1].plot(layers, phi_med, marker="s", label=tag, alpha=0.85)
        axes[2].plot(layers, theta_med, marker="^", label=tag, alpha=0.85)
    for ax, label in zip(
        axes,
        ["R(ℓ) = ‖∇V_φ‖ / ‖∇V_θ‖",
         "median Φ_φ per layer",
         "median |Θ_φ| per layer"],
    ):
        ax.set_xlabel("layer")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_yscale("log")
    fig.suptitle("Cross-checkpoint comparison of V_phi channel signatures",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
