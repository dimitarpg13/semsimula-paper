r"""
γ-prediction diagnostic for SPLM (E10 companion).

Loads a trained SPLM (sarf_mass_ln / em_ln) checkpoint, samples typical
hidden states by re-running the model on validation data, and emits four
predictions of the optimal damping coefficient γ* per the framework in
docs/Determining_optimal_gamma_for_SPLM.md:

  §2.1  γ*_depth          (closed form, depth scaling)
  §2.2  γ*_Hessian        (Hessian-spectrum critical-damping estimate;
                          UPPER BOUND on optimal-PPL γ — see doc §3 caveat)
  §2.3  γ*_surprisal      (corpus-conditional surprisal scaling)
  §2.4  γ*_per_layer      (Hessian estimate stratified by integrator layer)

Output: a `predict_gamma.json` (machine-readable) and a `predict_gamma_summary.md`
(human-readable) under the requested output directory.

Companion experiment: E10 γ-transfer (docs/Gamma_transfer_pre-registered_protocol.md).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
SCALEUP_DIR = SCRIPT_DIR.parent
NOTEBOOKS_DIR = SCALEUP_DIR.parent
sys.path.insert(0, str(NOTEBOOKS_DIR))
sys.path.insert(0, str(NOTEBOOKS_DIR / "energetic_minima"))
sys.path.insert(0, str(NOTEBOOKS_DIR / "sarf_mass_variant"))

from data_module import get_batch, load_tiny_stories, load_tiny_shakespeare  # noqa: E402
from model_ln import ScalarPotentialLMSARFMassLN, SPLMSARFMassLNConfig  # noqa: E402
from model_sarf_mass import causal_cumulative_mean  # noqa: E402


# ---------------------------------------------------------------------------
# Hessian-vector primitives
# ---------------------------------------------------------------------------

def hvp(V_theta, xi: torch.Tensor, h: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Per-token Hessian-vector product H @ v where H = ∇²_h V_θ(xi, h).

    All tensors are (B, T, d). xi is held fixed; the Hessian is taken w.r.t. h.
    """
    xi = xi.detach()
    h_in = h.detach().clone().requires_grad_(True)
    V = V_theta(xi, h_in).sum()
    g, = torch.autograd.grad(V, h_in, create_graph=True)
    Hv, = torch.autograd.grad((g * v).sum(), h_in)
    return Hv.detach()


def power_iter_signed_lambda(
    V_theta, xi: torch.Tensor, h: torch.Tensor, n_iter: int = 20
) -> torch.Tensor:
    """Per-token signed Rayleigh quotient of the dominant H eigenvector.

    Power iteration on H converges to the eigenvector with the largest |λ|.
    Returns the signed Rayleigh quotient ⟨v, Hv⟩, which equals λ_dominant.
    Sign tells us whether the local landscape is convex (+; attractor) or
    has a saddle/maximum direction (−).
    """
    v = torch.randn_like(h)
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    for _ in range(n_iter):
        Hv = hvp(V_theta, xi, h, v)
        v = Hv / Hv.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    Hv = hvp(V_theta, xi, h, v)
    rayleigh = (v * Hv).sum(dim=-1)  # (B, T)
    return rayleigh


def hutchinson_trace_per_dim(
    V_theta, xi: torch.Tensor, h: torch.Tensor, n_probes: int = 5
) -> torch.Tensor:
    """Per-token Hutchinson estimate of tr(H)/d, the *mean* H eigenvalue."""
    d = h.shape[-1]
    accum = torch.zeros(h.shape[0], h.shape[1], device=h.device, dtype=h.dtype)
    for _ in range(n_probes):
        v = (torch.randint(0, 2, h.shape, device=h.device, dtype=h.dtype) * 2.0
             - 1.0)
        Hv = hvp(V_theta, xi, h, v)
        accum += (v * Hv).sum(dim=-1)
    return accum / (n_probes * d)


# ---------------------------------------------------------------------------
# Forward-pass collection
# ---------------------------------------------------------------------------

def _broadcast_mass_2d(m_full: torch.Tensor, B: int, T: int) -> torch.Tensor:
    """Reduce the model.compute_mass output to a (B, T) per-token mass tensor.

    Handles the three legitimate dtypes: scalar (global), (B, T, 1) (logfreq),
    or (B, T) (already broadcast).
    """
    if m_full.dim() == 0:
        return m_full.expand(B, T)
    if m_full.dim() == 2:
        return m_full
    if m_full.dim() == 3:
        return m_full.squeeze(-1)
    raise RuntimeError(f"Unexpected mass tensor shape {tuple(m_full.shape)}")


@torch.enable_grad()
def collect_hessian_stats(
    model: ScalarPotentialLMSARFMassLN,
    val_ids: np.ndarray,
    n_batches: int,
    batch_size: int,
    block_size: int,
    rng: np.random.Generator,
    device: str,
    n_power_iter: int = 20,
    n_hutchinson: int = 5,
) -> dict:
    """Run forward integration; collect per-layer Hessian eigenvalue stats."""
    model.eval()
    L = model.cfg.L

    # Per-(batch × layer) flat numpy arrays of length B*T
    lam_top_per_layer: list[list[np.ndarray]] = [[] for _ in range(L)]
    lam_avg_per_layer: list[list[np.ndarray]] = [[] for _ in range(L)]
    mass_per_layer:    list[list[np.ndarray]] = [[] for _ in range(L)]

    for batch_idx in range(n_batches):
        xb, _ = get_batch(val_ids, batch_size, block_size, rng)
        x = torch.from_numpy(xb).to(device)
        B, T = x.shape

        with torch.no_grad():
            emb = model._embed(x)
            m_full = model.compute_mass(x, emb)
        m_2d = _broadcast_mass_2d(m_full, B, T)

        # Reproduce the integrator forward pass, capturing (xi, h) at each layer.
        h = model._project(emb) if model.cfg.ln_after_step else emb
        h = h.detach()
        v_kin = torch.zeros_like(h)
        gamma_val = model.gamma.detach()
        dt = model.cfg.dt

        for layer in range(L):
            xi_now = causal_cumulative_mean(h).detach()
            h_now = h.detach()

            # Hessian stats at this (xi, h)
            lam_top = power_iter_signed_lambda(
                model.V_theta, xi_now, h_now, n_iter=n_power_iter
            )                                                        # (B, T) signed
            lam_avg = hutchinson_trace_per_dim(
                model.V_theta, xi_now, h_now, n_probes=n_hutchinson
            )                                                        # (B, T) signed

            lam_top_per_layer[layer].append(
                lam_top.detach().cpu().numpy().reshape(-1)
            )
            lam_avg_per_layer[layer].append(
                lam_avg.detach().cpu().numpy().reshape(-1)
            )
            mass_per_layer[layer].append(
                m_2d.detach().cpu().numpy().reshape(-1)
            )

            # Advance h via the trained-γ integrator (no grad through model params)
            h_in = h_now.clone().requires_grad_(True)
            V = model.V_theta(xi_now, h_in).sum()
            grad_V, = torch.autograd.grad(V, h_in, create_graph=False)
            f = -grad_V
            # m_full has the dimensionality the integrator expects
            v_kin = (v_kin + dt * f / m_full) / (1.0 + dt * gamma_val)
            h_new = h_now + dt * v_kin
            if model.cfg.ln_after_step:
                h_new = model._project(h_new)
            h = h_new.detach()
            v_kin = v_kin.detach()

        print(
            f"[predict-gamma] batch {batch_idx + 1}/{n_batches} done "
            f"(processed {B * T:,} tokens × {L} layers = "
            f"{B * T * L:,} state samples)",
            flush=True,
        )

    # Aggregate
    per_layer = []
    for layer in range(L):
        lam_top = np.concatenate(lam_top_per_layer[layer])
        lam_avg = np.concatenate(lam_avg_per_layer[layer])
        m       = np.concatenate(mass_per_layer[layer])
        pos = lam_top > 0
        per_layer.append({
            "layer": layer,
            "n_states": int(len(lam_top)),
            "pos_fraction": float(pos.mean()),
            "lam_top_mean":  float(lam_top.mean()),
            "lam_top_std":   float(lam_top.std()),
            "lam_top_p50":   float(np.median(lam_top)),
            "lam_top_p10":   float(np.percentile(lam_top, 10)),
            "lam_top_p90":   float(np.percentile(lam_top, 90)),
            "lam_avg_mean":  float(lam_avg.mean()),
            "m_mean":        float(m.mean()),
            "m_std":         float(m.std()),
            "gamma_top_pos_mean":
                float(2.0 * np.mean(np.sqrt(lam_top[pos] / m[pos])))
                if pos.any() else float("nan"),
        })

    # Aggregate over all layers
    lam_top_all = np.concatenate(
        [np.concatenate(lst) for lst in lam_top_per_layer]
    )
    lam_avg_all = np.concatenate(
        [np.concatenate(lst) for lst in lam_avg_per_layer]
    )
    m_all       = np.concatenate(
        [np.concatenate(lst) for lst in mass_per_layer]
    )

    pos_mask = lam_top_all > 0
    pos_avg_mask = lam_avg_all > 0
    overall = {
        "n_states_total":      int(len(lam_top_all)),
        "pos_fraction_total":  float(pos_mask.mean()),
        "lam_top_mean":        float(lam_top_all.mean()),
        "lam_top_pos_mean":    float(lam_top_all[pos_mask].mean())
                                if pos_mask.any() else float("nan"),
        "lam_avg_mean":        float(lam_avg_all.mean()),
        "m_mean":              float(m_all.mean()),
        "m_std":               float(m_all.std()),
        "gamma_hessian_top":
            float(2.0 * np.mean(np.sqrt(lam_top_all[pos_mask]
                                        / m_all[pos_mask])))
            if pos_mask.any() else float("nan"),
        "gamma_hessian_avg":
            float(2.0 * np.mean(np.sqrt(lam_avg_all[pos_avg_mask]
                                        / m_all[pos_avg_mask])))
            if pos_avg_mask.any() else float("nan"),
    }

    return {"per_layer": per_layer, "overall": overall}


# ---------------------------------------------------------------------------
# Closed-form predictors (§2.1, §2.3)
# ---------------------------------------------------------------------------

def gamma_depth_closed_form(
    L: int, dt: float, m_mean: float, rho: float
) -> float:
    """§2.1 of docs/Determining_optimal_gamma_for_SPLM.md."""
    return (m_mean / (L * dt)) * math.log(1.0 / rho)


def gamma_surprisal_scaling(
    gamma_e5: float,
    s_bar_corpus: float, m_bar_corpus: float,
    s_bar_e5: float = 9.5, m_bar_e5: float = 1.4,
) -> float:
    """§2.3 of docs/Determining_optimal_gamma_for_SPLM.md.

    Default (s_bar_e5, m_bar_e5) hold the Tiny-Shakespeare reference at
    E5 fingerprint values. Can be overridden if a more accurate Tiny-Shakespeare
    surprisal estimate is available.
    """
    ratio = math.sqrt(
        (s_bar_corpus / m_bar_corpus) / (s_bar_e5 / m_bar_e5)
    )
    return gamma_e5 * ratio


def estimate_corpus_surprisal_bits(
    val_ids: np.ndarray, vocab_size: int = 50257
) -> float:
    """Rough mean unigram surprisal of a token stream, in bits/token."""
    counts = np.bincount(val_ids, minlength=vocab_size)
    p = counts / counts.sum()
    nz = p > 0
    return float(-(p[nz] * np.log2(p[nz])).sum())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Four-estimator γ\\*-prediction diagnostic for SPLM.",
    )
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "notebooks/conservative_arch/scaleup/results/seed0_splm/"
            "splm_em_ln_scaleup_scaleup_seed0_ckpt_latest.pt"
        ),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "notebooks/conservative_arch/scaleup/gamma_transfer/results/predictors"
        ),
    )
    ap.add_argument("--logfreq-path",
                    type=Path,
                    default=Path("notebooks/conservative_arch/scaleup/results/"
                                 "logfreq_surprisal_tinystories.npy"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-batches", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--n-power-iter", type=int, default=20)
    ap.add_argument("--n-hutchinson", type=int, default=5)
    ap.add_argument("--max-train-tokens", type=int, default=5_000_000)
    ap.add_argument("--mode", choices=["stories", "shakespeare"], default="stories",
                    help="Validation-data corpus for hidden-state sampling.")
    args = ap.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    device = args.device
    print(f"[predict-gamma] device={device}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # --- Load checkpoint ---
    print(f"[predict-gamma] loading checkpoint {args.checkpoint}", flush=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg_dict = ckpt["model_cfg"]
    # logfreq_path is required for logfreq mass mode; respect arg if available
    if cfg_dict.get("mass_mode") == "logfreq":
        cfg_dict["logfreq_path"] = str(args.logfreq_path)
    model_cfg = SPLMSARFMassLNConfig(**cfg_dict)
    model = ScalarPotentialLMSARFMassLN(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(
        f"[predict-gamma] checkpoint loaded; params={model.num_params():,}  "
        f"d={model_cfg.d}  L={model_cfg.L}  v_hidden={model_cfg.v_hidden}  "
        f"trained_gamma={ckpt.get('final_gamma', 'unknown')}",
        flush=True,
    )

    # --- Load val data ---
    if args.mode == "shakespeare":
        print(f"[predict-gamma] mode=shakespeare", flush=True)
        train_ids, val_ids = load_tiny_shakespeare()
    else:
        print(f"[predict-gamma] mode=stories  max_train_tokens={args.max_train_tokens:,}",
              flush=True)
        train_ids, val_ids = load_tiny_stories(
            max_train_tokens=args.max_train_tokens,
        )
    print(f"[predict-gamma] tokens: train={len(train_ids):,}  val={len(val_ids):,}",
          flush=True)

    # --- Compute corpus unigram surprisal (in bits) ---
    s_bar_train = estimate_corpus_surprisal_bits(train_ids,
                                                 vocab_size=model_cfg.vocab_size)
    print(f"[predict-gamma] mean unigram surprisal of train split: "
          f"{s_bar_train:.3f} bits/token", flush=True)

    # --- §2.2 Hessian-spectrum diagnostics ---
    print("[predict-gamma] running power iter + Hutchinson over Hessian "
          "(this takes a few minutes)...", flush=True)
    hess_stats = collect_hessian_stats(
        model, val_ids,
        n_batches=args.n_batches, batch_size=args.batch_size,
        block_size=args.block_size, rng=rng, device=device,
        n_power_iter=args.n_power_iter, n_hutchinson=args.n_hutchinson,
    )

    overall = hess_stats["overall"]
    m_mean = overall["m_mean"]

    # --- §2.1 depth-scaling closed form ---
    L = model_cfg.L
    dt = model_cfg.dt
    gamma_depth = {
        f"rho_{rho:.3f}": gamma_depth_closed_form(L, dt, m_mean, rho)
        for rho in [0.05, 0.10, 0.15, 0.18, 0.20, 0.30, 0.50, 0.565, 0.70]
    }

    # --- §2.3 corpus-surprisal scaling ---
    # Tiny-Shakespeare reference (E5 corpus): ~9.5 bits/token at the BPE level
    # (estimated; can be re-measured if a corpus snapshot is available).
    gamma_surprisal = gamma_surprisal_scaling(
        gamma_e5=0.30,
        s_bar_corpus=s_bar_train,
        m_bar_corpus=m_mean,
        s_bar_e5=9.5,
        m_bar_e5=1.4,
    )

    # --- Aggregate output ---
    output = {
        "input": {
            "checkpoint":    str(args.checkpoint),
            "logfreq_path":  str(args.logfreq_path),
            "model_cfg":     {k: v for k, v in cfg_dict.items()
                              if not isinstance(v, (Path,))},
            "trained_gamma": ckpt.get("final_gamma", None),
            "params":        model.num_params(),
        },
        "diagnostics": {
            "n_batches":             args.n_batches,
            "batch_size":            args.batch_size,
            "block_size":            args.block_size,
            "n_power_iter":          args.n_power_iter,
            "n_hutchinson":          args.n_hutchinson,
            "n_states_total":        overall["n_states_total"],
            "pos_eigenvalue_fraction_total":  overall["pos_fraction_total"],
            "mass_mean":             overall["m_mean"],
            "mass_std":              overall["m_std"],
            "lam_top_mean":          overall["lam_top_mean"],
            "lam_top_pos_mean":      overall["lam_top_pos_mean"],
            "lam_avg_mean":          overall["lam_avg_mean"],
            "corpus_surprisal_bits": s_bar_train,
        },
        "predictions": {
            "gamma_depth":     gamma_depth,
            "gamma_surprisal": gamma_surprisal,
            "gamma_hessian_top":  overall["gamma_hessian_top"],
            "gamma_hessian_avg":  overall["gamma_hessian_avg"],
        },
        "per_layer": hess_stats["per_layer"],
    }

    out_json = args.out_dir / "predict_gamma.json"
    out_md   = args.out_dir / "predict_gamma_summary.md"

    out_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"[predict-gamma] wrote {out_json}", flush=True)

    # Human-readable summary
    md_lines: list[str] = []
    md_lines.append("# γ\\* prediction summary")
    md_lines.append("")
    md_lines.append(f"- **Checkpoint:** `{args.checkpoint.name}`")
    md_lines.append(f"- **Trained γ:** {ckpt.get('final_gamma', '?'):.4f}")
    md_lines.append(f"- **Params:** {model.num_params():,}")
    md_lines.append(f"- **Architecture:** d={model_cfg.d}, L={model_cfg.L}, "
                    f"v_hidden={model_cfg.v_hidden}, max_len={model_cfg.max_len}")
    md_lines.append(f"- **Corpus mean unigram surprisal (train split):** "
                    f"{s_bar_train:.3f} bits/token")
    md_lines.append(f"- **Mean per-token mass:** {overall['m_mean']:.3f} "
                    f"(std {overall['m_std']:.3f})")
    md_lines.append(f"- **State samples:** {overall['n_states_total']:,} "
                    f"({overall['pos_fraction_total'] * 100:.1f}% with positive "
                    f"top-eigenvalue)")
    md_lines.append("")
    md_lines.append("## §2.1 Depth-scaling closed form")
    md_lines.append("")
    md_lines.append("| ρ | γ\\*_depth |")
    md_lines.append("|---:|---:|")
    for rho in [0.05, 0.10, 0.15, 0.18, 0.20, 0.30, 0.50, 0.565, 0.70]:
        md_lines.append(f"| {rho:.3f} | {gamma_depth[f'rho_{rho:.3f}']:.4f} |")
    md_lines.append("")
    md_lines.append("## §2.2 Hessian-spectrum critical damping")
    md_lines.append("")
    md_lines.append(
        "| estimator | γ\\* |"
        " | mean λ | pos-fraction |"
    )
    md_lines.append("|---|---:|---:|---:|")
    md_lines.append(
        f"| top-eigenvalue (positive states only) | "
        f"**{overall['gamma_hessian_top']:.4f}** | "
        f"{overall['lam_top_pos_mean']:.4f} | "
        f"{overall['pos_fraction_total'] * 100:.1f}% |"
    )
    md_lines.append(
        f"| average eigenvalue (Hutchinson) | "
        f"{overall['gamma_hessian_avg']:.4f} | "
        f"{overall['lam_avg_mean']:.4f} | — |"
    )
    md_lines.append("")
    md_lines.append("### Per-layer breakdown")
    md_lines.append("")
    md_lines.append(
        "| layer | n | pos % | mean λ_top | γ\\*_top (pos) | mean m |"
    )
    md_lines.append("|---:|---:|---:|---:|---:|---:|")
    for entry in hess_stats["per_layer"]:
        md_lines.append(
            f"| {entry['layer']} | {entry['n_states']:,} | "
            f"{entry['pos_fraction'] * 100:.1f} | "
            f"{entry['lam_top_mean']:.4f} | "
            f"{entry['gamma_top_pos_mean']:.4f} | "
            f"{entry['m_mean']:.3f} |"
        )
    md_lines.append("")
    md_lines.append("## §2.3 Corpus-surprisal scaling")
    md_lines.append("")
    md_lines.append(
        f"- Tiny-Shakespeare reference (E5): γ\\*_E5 = 0.30, "
        f"S̄_E5 ≈ 9.5 bits, m̄_E5 ≈ 1.4."
    )
    md_lines.append(
        f"- TinyStories (E10): S̄ = {s_bar_train:.3f} bits, m̄ = {overall['m_mean']:.3f}."
    )
    md_lines.append(f"- **γ\\*_surprisal({{TS}}) ≈ {gamma_surprisal:.4f}**")
    md_lines.append("")
    md_lines.append("## Reconciliation")
    md_lines.append("")
    md_lines.append(
        "| estimator | γ\\* | source |"
    )
    md_lines.append("|---|---:|---|")
    md_lines.append(
        f"| §2.1 depth (ρ=0.18, buggy v2 anchor)  | "
        f"{gamma_depth['rho_0.180']:.4f} | closed form |"
    )
    md_lines.append(
        f"| §2.2 Hessian top-eigenvalue       | "
        f"{overall['gamma_hessian_top']:.4f} | trained-checkpoint diagnostic |"
    )
    md_lines.append(
        f"| §2.2 Hessian avg-eigenvalue       | "
        f"{overall['gamma_hessian_avg']:.4f} | trained-checkpoint diagnostic |"
    )
    md_lines.append(
        f"| §2.1 depth (ρ=0.565, leak-free anchor) | "
        f"{gamma_depth['rho_0.565']:.4f} | closed form |"
    )
    md_lines.append(
        f"| §2.3 corpus-surprisal scaling     | "
        f"{gamma_surprisal:.4f} | corpus statistic, no checkpoint |"
    )
    md_lines.append(
        "| (E10 Stage 1 empirical) | TBD | live grid sweep |"
    )
    md_lines.append("")
    md_lines.append(
        "All four numbers should agree to within ~10–20 % if the framework is "
        "consistent. See `docs/Determining_optimal_gamma_for_SPLM.md` §3 for the "
        "full reconciliation rules. The Stage-1 empirical γ\\* will be appended "
        "to this table once E10 reports."
    )

    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[predict-gamma] wrote {out_md}", flush=True)

    print("")
    print("=== γ\\* PREDICTIONS ===")
    print(f"  §2.1 depth (ρ=0.18,  buggy)     : {gamma_depth['rho_0.180']:.4f}")
    print(f"  §2.1 depth (ρ=0.565, leak-free) : {gamma_depth['rho_0.565']:.4f}")
    print(f"  §2.2 Hessian top                 : {overall['gamma_hessian_top']:.4f}")
    print(f"  §2.2 Hessian avg                 : {overall['gamma_hessian_avg']:.4f}")
    print(f"  §2.3 surprisal                   : {gamma_surprisal:.4f}")
    print(f"  trained γ (locked)               : {ckpt.get('final_gamma', '?')}")
    print(f"  E10 Stage 1 (TBD)                : awaiting grid sweep")

    return 0


if __name__ == "__main__":
    sys.exit(main())
