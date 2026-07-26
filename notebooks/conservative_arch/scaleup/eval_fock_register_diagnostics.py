"""
Register diversity and attention entropy eval for Fock Multi-Xi PARFLM checkpoints.

Loads a saved checkpoint and computes per-layer diagnostic metrics from §17.2
of the companion report (Improving_the_Fock_Mechanism_to_match_Attention.md):

  1. **Normalised attention entropy** — H_norm = -sum_j(a_kj log a_kj) / log(T).
     0 = peaked (useful routing), 1 = uniform (wasted mean-pool).
  2. **Register content diversity** — 1 - mean off-diagonal cosine similarity
     of the creation readout r_new_content.
     0 = identical registers, 1 = orthogonal specialisation.
  3. **alpha_max** — max_j(a_kj) per register (creation signal strength).

Reference values (from Q0/Q6 sweep on standalone FockPARFLM_v2):
  - Genuine routing (Q6):  diversity ~ 0.785, entropy ~ 0.304
  - Inert mean-pool (Q0):  diversity ~ 0.145, entropy ~ 1.0

Under the prefix-causal register fix the creation gate holds one bank per
position rather than one per sequence, so there is no single sequence-wide
attention distribution to report.  The metrics are read at the final position,
where the prefix window covers the whole sequence — that is the distribution
the legacy path reported, so the reference values above still apply.

Usage:
  python eval_fock_register_diagnostics.py --checkpoint /path/to/ckpt.pt
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).parent
PARENT_DIR = SCRIPT_DIR.parent
PARF_DIR = PARENT_DIR / "parf"
MULTIXI_DIR = PARENT_DIR / "multixi"
sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(PARF_DIR))
sys.path.insert(0, str(MULTIXI_DIR))

from data_module import get_batch, load_tiny_stories  # noqa: E402
from model_fock_parf_multixi import (  # noqa: E402
    FockMultiXiPARFConfig,
    FockMultiXiPARFLM,
)
from model_fock_parf_v2 import _prefix_causal_creation_readout  # noqa: E402

DEFAULT_LOGFREQ_PATH = SCRIPT_DIR / "results" / "logfreq_surprisal_tinystories.npy"

# Score ceiling used by the prefix-causal readout, read off the model so the
# diagnostic cannot silently drift from the arithmetic it is measuring.
PREFIX_SCORE_CLAMP = (
    inspect.signature(_prefix_causal_creation_readout).parameters["clamp"].default
)


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_from_checkpoint(
    ckpt_path: str, device: str,
) -> tuple[FockMultiXiPARFLM, FockMultiXiPARFConfig, dict]:
    """Load model from checkpoint, fixing logfreq_path for local execution."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_cfg_dict = ckpt["model_cfg"]

    local_logfreq = str(DEFAULT_LOGFREQ_PATH)
    if not Path(local_logfreq).exists():
        raise FileNotFoundError(
            f"logfreq surprisal file not found at {local_logfreq}. "
            "Run scaleup/compute_unigram_frequencies_tinystories.py first."
        )
    model_cfg_dict["logfreq_path"] = local_logfreq

    cfg = FockMultiXiPARFConfig(**model_cfg_dict)
    model = FockMultiXiPARFLM(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model, cfg, ckpt


def compute_creation_alpha(
    model: FockMultiXiPARFLM,
    h: torch.Tensor,
    r: torch.Tensor,
    cfg: FockMultiXiPARFConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recompute full creation attention weights (alpha) from the QKV gate.

    Returns (alpha, r_new_content, alpha_max) where alpha is (B, M, T).
    Handles both v2 (shared W_K, scalar tau) and v2.1 (per-register W_K, per-register tau).

    Two register layouts are supported:

      * legacy — ``r`` is (B, M, d), one bank per sequence.  Every register
        queries every token, giving the (B, M, T) score matrix directly.
      * prefix-causal (``cfg.prefix_causal_registers``) — ``r`` is
        (B, Tr, M, d) with Tr in {1, T}, one bank per position.  The gate
        scores token t with the query as of position t, so the scores are
        the DIAGONAL of the full matrix.  The prefix softmax at the final
        position spans the whole sequence, which is what the legacy path
        reports, so evaluating there keeps the metrics on the same scale as
        the Q0/Q6 reference values.
    """
    gate = model.creation_gate_qkv
    B, T, d = h.shape
    M = cfg.n_registers
    d_k = cfg.d_k

    per_register_keys = getattr(gate, "per_register_keys", False)
    prefix_causal = r.dim() == 4

    if prefix_causal:
        Q = torch.einsum("btmd,mdk->btmk", r, gate.W_Q)  # (B, Tr, M, d_k)
        if Q.shape[1] == 1:
            Q = Q.expand(B, T, M, d_k)
        if per_register_keys:
            K = torch.einsum("btd,mdk->btmk", h, gate.W_K)   # (B, T, M, d_k)
            scores = (Q * K).sum(-1).permute(0, 2, 1)        # (B, M, T)
        else:
            K = gate.W_K(h)                                   # (B, T, d_k)
            scores = torch.einsum("btmk,btk->btm", Q, K).permute(0, 2, 1)
    else:
        Q = torch.einsum("bmd,mdk->bmk", r, gate.W_Q)  # (B, M, d_k)
        if per_register_keys:
            K = torch.einsum("btd,mdk->bmtk", h, gate.W_K)  # (B, M, T, d_k)
            scores = torch.einsum("bmk,bmtk->bmt", Q, K)
        else:
            K = gate.W_K(h)  # (B, T, d_k)
            scores = torch.bmm(
                Q.reshape(B * M, 1, d_k),
                K.unsqueeze(1).expand(B, M, T, d_k).reshape(B * M, d_k, T),
            ).reshape(B, M, T)

    if gate.log_tau is not None:
        tau = gate.log_tau.exp().clamp(min=1e-4)
        if tau.dim() >= 1:
            scores = scores / tau.unsqueeze(0).unsqueeze(-1)
        else:
            scores = scores / tau
    else:
        scores = scores / (d_k ** 0.5)

    if prefix_causal:
        # Mirror the saturating constant shift the model applies before the
        # exp.  The shift itself cancels in the softmax, but the clamp does
        # not: it ties scores above the ceiling, and that shows up as higher
        # entropy exactly in the sharply-routed regime this eval is looking
        # for.  Reporting the distribution the model really uses beats
        # reporting a cleaner one it does not.
        scores = scores.float().clamp(max=PREFIX_SCORE_CLAMP)

    alpha = F.softmax(scores, dim=-1)  # (B, M, T)

    V = gate.W_V(h).to(alpha.dtype)  # (B, T, d)
    r_new = torch.bmm(
        alpha.reshape(B * M, 1, T),
        V.unsqueeze(1).expand(B, M, T, d).reshape(B * M, T, d),
    ).reshape(B, M, d)

    alpha_max = alpha.max(dim=-1).values  # (B, M)

    return alpha, r_new, alpha_max


def compute_diagnostics(
    model: FockMultiXiPARFLM,
    cfg: FockMultiXiPARFConfig,
    val_ids: np.ndarray,
    *,
    diag_batches: int = 10,
    batch_size: int = 16,
    block_size: int = 512,
    device: str = "cpu",
    seed: int = 0,
) -> dict:
    """Run the diagnostic eval loop across val batches.

    Returns a dict with per-layer entropy, alpha_max, and diversity arrays.
    """
    rng = np.random.default_rng(seed)
    model.eval()

    all_entropy = []
    all_alpha_max = []
    all_diversity = []

    for b_idx in range(diag_batches):
        xb, _ = get_batch(val_ids, batch_size, block_size, rng)
        x = torch.from_numpy(xb).to(device)

        with torch.enable_grad():
            h0 = model._embed(x)
            r, salience = model._init_registers(batch_size, h0.device)
            h, h_prev = h0, h0
            m_b = model.compute_mass(x)
            gamma, dt = model.gamma, cfg.dt

            batch_entropy = []
            batch_alpha_max = []
            batch_diversity = []

            for ell in range(cfg.L):
                B, T, d = h.shape
                M = cfg.n_registers

                alpha, r_new_content, alpha_max_diag = compute_creation_alpha(
                    model, h, r, cfg,
                )

                # Normalised entropy: 0 = peaked, 1 = uniform
                log_alpha = torch.log(alpha + 1e-12)
                entropy = -(alpha * log_alpha).sum(dim=-1)  # (B, M)
                norm_entropy = entropy / math.log(T)

                batch_entropy.append(
                    norm_entropy.detach().mean(0).cpu().numpy()
                )
                batch_alpha_max.append(
                    alpha_max_diag.detach().mean(0).cpu().numpy()
                )

                # Diversity: 1 - mean off-diagonal cosine similarity
                r_flat = r_new_content.detach().mean(0)  # (M, d)
                cos_matrix = F.cosine_similarity(
                    r_flat.unsqueeze(0), r_flat.unsqueeze(1), dim=-1,
                )
                mask = ~torch.eye(M, dtype=torch.bool, device=device)
                mean_off_diag_cos = cos_matrix[mask].mean().item()
                batch_diversity.append(1.0 - mean_off_diag_cos)

                # Step the layer (detach to avoid OOM on long sequences)
                h_new, h_prev_out, r, salience = model._fock_layer_step(
                    h, h_prev, r, salience, m_b, gamma, dt, layer_idx=ell,
                )
                h = h_new.detach().requires_grad_(True)
                h_prev = h_prev_out.detach().requires_grad_(True)
                r = r.detach()
                salience = salience.detach()

        all_entropy.append(np.stack(batch_entropy))      # (L, M)
        all_alpha_max.append(np.stack(batch_alpha_max))   # (L, M)
        all_diversity.append(batch_diversity)              # list of L floats

        print(f"  batch {b_idx+1}/{diag_batches} done")

    mean_entropy = np.mean(all_entropy, axis=0)       # (L, M)
    mean_alpha_max = np.mean(all_alpha_max, axis=0)   # (L, M)
    mean_diversity = np.mean(all_diversity, axis=0)    # (L,)

    return {
        "entropy_per_layer": mean_entropy,
        "alpha_max_per_layer": mean_alpha_max,
        "diversity_per_layer": mean_diversity,
    }


def print_summary(diag: dict, cfg: FockMultiXiPARFConfig) -> None:
    """Print per-layer summary and overall verdict."""
    entropy = diag["entropy_per_layer"]
    alpha_max = diag["alpha_max_per_layer"]
    diversity = diag["diversity_per_layer"]
    L = cfg.L

    prefix_causal = (
        cfg.fock_version == "v2"
        and getattr(cfg, "prefix_causal_registers", False)
    )

    print("\n" + "=" * 78)
    print(" Fock v2 Register Diagnostics — Per-Layer Summary")
    if prefix_causal:
        print(" (prefix-causal registers: metrics read at the final position)")
    print("=" * 78)
    print(f"{'Layer':>6s}  {'Entropy':>10s}  {'alpha_max':>10s}  {'Diversity':>10s}")
    print("-" * 78)
    for ell in range(L):
        print(
            f"{ell:6d}  "
            f"{entropy[ell].mean():10.4f}  "
            f"{alpha_max[ell].mean():10.4f}  "
            f"{diversity[ell]:10.4f}"
        )
    print("-" * 78)

    overall_entropy = entropy.mean()
    overall_alpha_max = alpha_max.mean()
    overall_diversity = diversity.mean()

    print(f"\n  Overall mean entropy:   {overall_entropy:.4f}")
    print(f"  Overall mean alpha_max: {overall_alpha_max:.4f}")
    print(f"  Overall mean diversity: {overall_diversity:.4f}")

    print(f"\n  Reference — Q6 (genuine routing):  "
          f"entropy=0.304  diversity=0.785")
    print(f"  Reference — Q0 (inert mean-pool):  "
          f"entropy=1.000  diversity=0.145")

    # Verdict
    print("\n" + "-" * 78)
    if overall_diversity >= 0.6 and 0.1 <= overall_entropy <= 0.5:
        verdict = "ROUTING"
        desc = ("Genuine register specialisation detected. "
                "Resolution hierarchy: Step 3 (scale M with sparse routing).")
    elif overall_diversity < 0.3 and overall_entropy > 0.8:
        verdict = "MEAN-POOL"
        desc = ("Inert capacity — registers are mean-pooling. "
                "Resolution hierarchy: Step 1 (fix creation gate: "
                "temperature, per-register keys, orthogonal init).")
    else:
        verdict = "MIXED"
        desc = ("Partial routing detected. "
                "Resolution hierarchy: Step 1 first, then reassess.")
    print(f"  VERDICT: {verdict}")
    print(f"  {desc}")
    print("=" * 78)


def save_outputs(
    diag: dict,
    cfg: FockMultiXiPARFConfig,
    tag: str,
    output_dir: Path,
) -> None:
    """Save JSON diagnostics and 3-panel PNG figure."""
    output_dir.mkdir(parents=True, exist_ok=True)

    entropy = diag["entropy_per_layer"]
    alpha_max = diag["alpha_max_per_layer"]
    diversity = diag["diversity_per_layer"]
    L = cfg.L

    # JSON
    json_path = output_dir / f"{tag}_register_diagnostics.json"
    with json_path.open("w") as f:
        json.dump({
            "entropy_per_layer": entropy.tolist(),
            "alpha_max_per_layer": alpha_max.tolist(),
            "diversity_per_layer": diversity.tolist(),
            "overall_mean_entropy": float(entropy.mean()),
            "overall_mean_alpha_max": float(alpha_max.mean()),
            "overall_mean_diversity": float(diversity.mean()),
            "model_tag": tag,
            "n_registers": cfg.n_registers,
            "fock_version": cfg.fock_version,
            "xi_channels": cfg.xi_channels,
            "d_k": cfg.d_k,
            "prefix_causal_registers": bool(
                getattr(cfg, "prefix_causal_registers", False)
            ),
        }, f, indent=2)
    print(f"Saved: {json_path}")

    # PNG
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    im = ax.imshow(
        entropy, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1,
    )
    ax.set_xlabel("Register index")
    ax.set_ylabel("Layer")
    ax.set_title("Normalised attention entropy\n(0=peaked, 1=uniform)")
    plt.colorbar(im, ax=ax)

    ax = axes[1]
    im = ax.imshow(
        alpha_max, aspect="auto", cmap="viridis", vmin=0,
    )
    ax.set_xlabel("Register index")
    ax.set_ylabel("Layer")
    ax.set_title("max_j(alpha_kj)\n(creation signal strength)")
    plt.colorbar(im, ax=ax)

    ax = axes[2]
    ax.plot(range(L), diversity, "o-", linewidth=2, color="steelblue")
    ax.axhline(y=0.785, color="green", linestyle="--", alpha=0.5,
               label="Q6 reference (0.785)")
    ax.axhline(y=0.145, color="red", linestyle="--", alpha=0.5,
               label="Q0 reference (0.145)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Register diversity (1 - mean cos sim)")
    ax.set_title("Register content diversity")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    png_path = output_dir / f"{tag}_register_diagnostics.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Register diversity/entropy eval for Fock Multi-Xi PARFLM"
    )
    ap.add_argument("--checkpoint", required=True,
                    help="Path to the .pt checkpoint file.")
    ap.add_argument("--diag-batches", type=int, default=10,
                    dest="diag_batches",
                    help="Number of val batches to average over.")
    ap.add_argument("--batch-size", type=int, default=16,
                    dest="batch_size")
    ap.add_argument("--block-size", type=int, default=512,
                    dest="block_size")
    ap.add_argument("--output-dir", dest="output_dir", default=None,
                    help="Output directory (default: same dir as checkpoint).")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-train-tokens", type=int, default=5_000_000,
                    dest="max_train_tokens")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = args.device or _pick_device()
    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir else ckpt_path.parent
    )

    print(f"[register-diag] checkpoint: {ckpt_path}")
    print(f"[register-diag] device: {device}")

    model, cfg, ckpt = load_model_from_checkpoint(str(ckpt_path), device)
    tag = ckpt.get("tag", ckpt_path.stem)
    n_params = sum(p.numel() for p in model.parameters())
    n_fock = model.get_register_overhead()

    print(f"[register-diag] model: FockMultiXiPARFLM "
          f"(fock={cfg.fock_version}, K={cfg.xi_channels}, "
          f"M={cfg.n_registers}, d_k={cfg.d_k})")
    print(f"[register-diag] params: {n_params:,}  fock_oh: {n_fock:,}")
    print(f"[register-diag] final_val_ppl: {ckpt.get('final_val_ppl', '?')}")

    _, val_ids = load_tiny_stories(max_train_tokens=args.max_train_tokens)
    print(f"[register-diag] val tokens: {len(val_ids):,}")

    print(f"\n[register-diag] Running diagnostics "
          f"({args.diag_batches} batches, bs={args.batch_size}, "
          f"block={args.block_size})...")

    diag = compute_diagnostics(
        model, cfg, val_ids,
        diag_batches=args.diag_batches,
        batch_size=args.batch_size,
        block_size=args.block_size,
        device=device,
        seed=args.seed,
    )

    print_summary(diag, cfg)
    save_outputs(diag, cfg, tag, output_dir)


if __name__ == "__main__":
    main()
