"""
Post-training causal leak probe for Fock Multi-Xi PARFLM checkpoints.

The at-init causal probe (causal_probe_multixi.py) has a blind spot:
the reverse channel scale is initialised to 0, so tanh(0) = 0 zeroes
the only path from registers to tokens.  As training pushes the scale
away from 0, a leak path opens if the creation gate attends to future
tokens (which it does — softmax over all T positions, no causal mask).

This script loads a TRAINED checkpoint and runs three diagnostics:

  1. **Perturbation probe** — change x[t_pert], check if logits at
     positions < t_pert change.  Any non-zero pre-perturbation delta
     is a causal violation.

  2. **Gradient probe** — compute ∂logits[t_target] / ∂emb[t'] for
     all t'.  Non-zero gradients for t' > t_target indicate a leak.

  3. **Reverse-channel ablation** — eval PPL with the reverse channel
     scale clamped to 0.  The PPL difference vs the original model
     upper-bounds the leak contribution (some of the reverse channel's
     value may be legitimate causal information).

Usage:
  python eval_causal_leak_trained.py --checkpoint /path/to/ckpt.pt
  python eval_causal_leak_trained.py --checkpoint /path/to/ckpt.pt --eval-batches 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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

DEFAULT_LOGFREQ_PATH = SCRIPT_DIR / "results" / "logfreq_surprisal_tinystories.npy"


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_from_checkpoint(
    ckpt_path: str, device: str,
) -> tuple[FockMultiXiPARFLM, FockMultiXiPARFConfig, dict]:
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


# ── Perturbation probe ──────────────────────────────────────────────
def perturbation_probe(
    model: torch.nn.Module,
    vocab_size: int,
    device: str,
    T: int = 128,
    n_probes: int = 5,
) -> dict:
    """Run multiple perturbation probes with different seeds and positions."""
    results = []
    for seed in range(n_probes):
        rng = np.random.default_rng(seed)
        xb = rng.integers(0, vocab_size, size=(1, T)).astype(np.int64)
        t_pert = T // 2 + seed  # vary the perturbation position

        x_a = torch.from_numpy(xb).to(device)
        x_b = x_a.clone()
        orig = int(x_b[0, t_pert].item())
        x_b[0, t_pert] = (orig + 17) % vocab_size

        with torch.enable_grad():
            out_a = model(x_a)
            out_b = model(x_b)
        logits_a = out_a[0].detach()
        logits_b = out_b[0].detach()

        diffs = (logits_a - logits_b).abs().max(dim=-1).values[0]
        pre_max = float(diffs[:t_pert].max().item())
        post_max = float(diffs[t_pert + 1:].max().item()) if t_pert + 1 < T else 0.0
        pre_mean = float(diffs[:t_pert].mean().item())

        results.append({
            "seed": seed,
            "t_pert": t_pert,
            "pre_max": pre_max,
            "pre_mean": pre_mean,
            "post_max": post_max,
            "leak_detected": pre_max > 1e-6,
        })

    max_pre = max(r["pre_max"] for r in results)
    mean_pre = np.mean([r["pre_max"] for r in results])
    any_leak = any(r["leak_detected"] for r in results)

    return {
        "n_probes": n_probes,
        "T": T,
        "max_pre_across_probes": max_pre,
        "mean_pre_across_probes": float(mean_pre),
        "any_leak_detected": any_leak,
        "per_probe": results,
    }


# ── Gradient probe ──────────────────────────────────────────────────
def gradient_probe(
    model: torch.nn.Module,
    vocab_size: int,
    device: str,
    T: int = 64,
    n_probes: int = 3,
) -> dict:
    """Gradient-Jacobian probe on trained model."""
    results = []

    cpu_model = model.to("cpu") if device != "cpu" else model
    try:
        for seed in range(n_probes):
            rng = np.random.default_rng(seed + 100)
            xb = rng.integers(0, vocab_size, size=(1, T)).astype(np.int64)
            x = torch.from_numpy(xb)
            t_target = T // 2 + seed

            cpu_model.train()
            with torch.enable_grad():
                emb_static = cpu_model._embed(x)
                emb_in = emb_static.detach().clone().requires_grad_(True)
                h_L, _ = cpu_model._stack_forward(emb_in, x)
                logits = h_L @ cpu_model.E.weight.T
                target = logits[0, t_target, :].sum()
                (g,) = torch.autograd.grad(
                    target, emb_in,
                    retain_graph=False, create_graph=False,
                )

            g = g[0]
            norms = g.norm(dim=-1)
            future_max = float(norms[t_target + 1:].max().item()) if t_target + 1 < T else 0.0
            past_max = float(norms[:t_target + 1].max().item())

            results.append({
                "seed": seed,
                "t_target": t_target,
                "future_grad_max": future_max,
                "past_grad_max": past_max,
                "leak_detected": future_max > 1e-6,
            })
    finally:
        if device != "cpu":
            model.to(device)
        model.eval()

    max_future = max(r["future_grad_max"] for r in results)
    any_leak = any(r["leak_detected"] for r in results)

    return {
        "n_probes": n_probes,
        "T": T,
        "max_future_grad": max_future,
        "any_leak_detected": any_leak,
        "per_probe": results,
    }


# ── Reverse channel ablation ────────────────────────────────────────
def eval_ppl(
    model: FockMultiXiPARFLM,
    val_data: np.ndarray,
    device: str,
    n_batches: int = 20,
    batch_size: int = 16,
    seq_len: int = 256,
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i in range(n_batches):
            xb = get_batch(val_data, batch_size=batch_size, seq_len=seq_len)
            xb = torch.from_numpy(xb).to(device)
            with torch.enable_grad():
                _, loss = model(xb)
            total_loss += loss.item()
    return float(np.exp(total_loss / n_batches))


def reverse_channel_ablation(
    model: FockMultiXiPARFLM,
    val_data: np.ndarray,
    device: str,
    n_batches: int = 20,
) -> dict:
    """Eval PPL with and without the reverse channel to bound leak contribution."""
    ppl_original = eval_ppl(model, val_data, device, n_batches=n_batches)

    rev_scale = model.reverse_channel_scale
    if rev_scale is None:
        return {
            "ppl_original": ppl_original,
            "ppl_rev_zeroed": ppl_original,
            "delta_ppl": 0.0,
            "original_rev_scale": None,
            "note": "No reverse channel in this model",
        }

    original_val = rev_scale.data.clone()
    original_tanh = float(torch.tanh(original_val).item())

    rev_scale.data.fill_(0.0)
    ppl_zeroed = eval_ppl(model, val_data, device, n_batches=n_batches)

    rev_scale.data.copy_(original_val)

    return {
        "ppl_original": round(ppl_original, 3),
        "ppl_rev_zeroed": round(ppl_zeroed, 3),
        "delta_ppl": round(ppl_zeroed - ppl_original, 3),
        "original_rev_scale_raw": float(original_val.item()),
        "original_rev_scale_tanh": round(original_tanh, 4),
    }


# ── Main ────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Post-training causal leak probe for Fock PARFLM checkpoints."
    )
    ap.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    ap.add_argument("--eval-batches", type=int, default=20,
                    help="Number of val batches for PPL ablation (default: 20)")
    ap.add_argument("--skip-gradient", action="store_true",
                    help="Skip gradient probe (slow on CPU)")
    ap.add_argument("--device", default=None,
                    help="Force device (cpu/cuda/mps)")
    args = ap.parse_args()

    device = args.device or _pick_device()
    print(f"[causal-leak-probe] device: {device}")
    print(f"[causal-leak-probe] checkpoint: {args.checkpoint}")

    model, cfg, ckpt = load_model_from_checkpoint(args.checkpoint, device)

    rev_scale = model.reverse_channel_scale
    if rev_scale is not None:
        rs_val = float(rev_scale.data.item())
        rs_tanh = float(torch.tanh(rev_scale.data).item())
        print(f"[causal-leak-probe] reverse_channel_scale: {rs_val:.4f} "
              f"(tanh = {rs_tanh:.4f})")
    else:
        print("[causal-leak-probe] no reverse channel")

    fock_ver = getattr(cfg, "fock_version", "?")
    per_reg_tau = getattr(cfg, "per_register_tau", False)
    per_reg_keys = getattr(cfg, "per_register_keys", False)
    print(f"[causal-leak-probe] fock_version={fock_ver}, "
          f"per_register_tau={per_reg_tau}, per_register_keys={per_reg_keys}")

    report = {
        "checkpoint": str(args.checkpoint),
        "fock_version": fock_ver,
        "per_register_tau": per_reg_tau,
        "per_register_keys": per_reg_keys,
    }

    # 1. Perturbation probe
    print("\n" + "=" * 60)
    print("  PERTURBATION PROBE (trained weights)")
    print("=" * 60)
    pert_results = perturbation_probe(model, cfg.vocab_size, device)
    report["perturbation_probe"] = pert_results

    if pert_results["any_leak_detected"]:
        print(f"  *** LEAK DETECTED ***")
        print(f"  max pre-perturbation delta: {pert_results['max_pre_across_probes']:.4e}")
        print(f"  mean pre-perturbation delta: {pert_results['mean_pre_across_probes']:.4e}")
    else:
        print(f"  No leak detected (all deltas < 1e-6)")
        print(f"  max pre-perturbation delta: {pert_results['max_pre_across_probes']:.4e}")

    for p in pert_results["per_probe"]:
        status = "LEAK" if p["leak_detected"] else "ok"
        print(f"    seed={p['seed']}  t_pert={p['t_pert']}  "
              f"pre_max={p['pre_max']:.4e}  post_max={p['post_max']:.4e}  [{status}]")

    # 2. Gradient probe
    if not args.skip_gradient:
        print("\n" + "=" * 60)
        print("  GRADIENT PROBE (trained weights)")
        print("=" * 60)
        grad_results = gradient_probe(model, cfg.vocab_size, device)
        report["gradient_probe"] = grad_results

        if grad_results["any_leak_detected"]:
            print(f"  *** LEAK DETECTED ***")
            print(f"  max future gradient norm: {grad_results['max_future_grad']:.4e}")
        else:
            print(f"  No leak detected (all future grads < 1e-6)")

        for p in grad_results["per_probe"]:
            status = "LEAK" if p["leak_detected"] else "ok"
            print(f"    seed={p['seed']}  t_target={p['t_target']}  "
                  f"future_max={p['future_grad_max']:.4e}  "
                  f"past_max={p['past_grad_max']:.4e}  [{status}]")
    else:
        print("\n  (gradient probe skipped)")

    # 3. Reverse channel ablation
    print("\n" + "=" * 60)
    print("  REVERSE CHANNEL PPL ABLATION")
    print("=" * 60)
    print(f"  Loading validation data ...")
    _, val_data = load_tiny_stories()
    abl_results = reverse_channel_ablation(
        model, val_data, device, n_batches=args.eval_batches,
    )
    report["reverse_channel_ablation"] = abl_results

    print(f"  PPL (original):          {abl_results['ppl_original']:.3f}")
    print(f"  PPL (rev channel = 0):   {abl_results['ppl_rev_zeroed']:.3f}")
    print(f"  ΔPPL (zeroed - orig):    +{abl_results['delta_ppl']:.3f}")
    if "original_rev_scale_tanh" in abl_results:
        print(f"  reverse_channel_scale:   {abl_results['original_rev_scale_raw']:.4f} "
              f"(tanh = {abl_results['original_rev_scale_tanh']})")

    # Summary verdict
    print("\n" + "=" * 60)
    print("  VERDICT")
    print("=" * 60)
    pert_leak = pert_results["any_leak_detected"]
    grad_leak = report.get("gradient_probe", {}).get("any_leak_detected", None)

    if pert_leak or grad_leak:
        print("  ⚠  CAUSAL LEAK CONFIRMED in trained model.")
        print(f"     Perturbation leak: {pert_leak}")
        print(f"     Gradient leak: {grad_leak}")
        print(f"     Reverse channel contribution: {abl_results['delta_ppl']:.2f} PPL")
        print(f"     PPL without reverse channel: {abl_results['ppl_rev_zeroed']:.3f}")
        print()
        print("     The creation gate attends to all T positions without a")
        print("     causal mask. Future-token information leaks through the")
        print("     reverse channel into past-position logits.")
    else:
        print("  ✓  No causal leak detected at this checkpoint.")
        print(f"     Reverse channel contribution: {abl_results['delta_ppl']:.2f} PPL")

    # Save report
    out_path = Path(args.checkpoint).with_name(
        Path(args.checkpoint).stem + "_causal_leak_report.json"
    )
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
