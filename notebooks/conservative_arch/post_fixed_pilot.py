"""Post-completion forensics driver for the fixed multi-ξ pilot.

When the leak-corrected multi-channel ξ pilot finishes (currently running
as PID 72285 with ETA ~6 AM EDT), this driver runs the standard
forensic pipeline in one shot:

  1.  Causal-violation probe in BOTH modes (buggy and fixed integrators
      applied to the same trained weights). For a leak-corrected ckpt we
      expect causal-side Δ ≈ 0 in *both* modes — the trained V_θ never
      learned a leak channel because the integrator severed it.

  2.  Val-PPL inflation measurement (40 K val tokens, same batches under
      both evaluators). For a leak-corrected ckpt we expect inflation
      factor ≈ 1×, in stark contrast to the 389× we measured on the
      buggy multi-ξ ckpt (`docs/Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md`
      §4.5).

  3.  α_k drift extraction from the training log. The final α_k vector
      is compared to its initialisation to quantify how aggressively the
      optimiser pushed the K = 4 EMA decay rates downward toward shorter
      horizons. Under the fix the *expected* drift is much smaller than
      the buggy run's drift (channel 1 dropped 0.50 -> 0.41, channel 2
      dropped 0.90 -> 0.85), because there is no leak channel to harvest.

  4.  A markdown report block (suitable for paste into §4.6 of the bug
      doc) is printed to stdout and saved to disk; a parallel JSON dump
      is also emitted for downstream tabulation.

Usage
-----
  python3 notebooks/conservative_arch/post_fixed_pilot.py
      --> auto-discovers the most recent ckpt under
          `notebooks/conservative_arch/scaleup/results/multixi_pilot_fixed/`,
          runs probe + inflation, parses the training log, writes the
          report.

  python3 notebooks/conservative_arch/post_fixed_pilot.py --ckpt PATH \
      [--n-batches 20] [--batch 8] [--block 256] [--seed 0] \
      [--out-md PATH] [--out-json PATH]

The default eval geometry (20 × 8 × 256 = 40 960 tokens) matches the
forensic settings used on the buggy multi-ξ ckpt for direct comparison.

References
----------
  - docs/Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md
  - notebooks/conservative_arch/causal_probe.py
  - notebooks/conservative_arch/eval_ppl_under_fix.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

THIS_DIR = Path(__file__).parent
for sub in ("", "sarf_mass_variant", "energetic_minima", "multixi", "scaleup"):
    sys.path.insert(0, str(THIS_DIR / sub))

# Reuse existing forensic helpers.
from causal_probe import causal_violation_probe  # type: ignore  # noqa: E402
from eval_ppl_under_fix import (  # type: ignore  # noqa: E402
    _dispatch, _resolve_corpus, _load_val_ids,
)
from data_module import get_batch  # type: ignore  # noqa: E402


DEFAULT_DIR = (
    THIS_DIR / "scaleup" / "results" / "multixi_pilot_fixed"
)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def find_default_ckpt() -> Optional[Path]:
    """Return the *_ckpt_latest.pt under DEFAULT_DIR, if exactly one exists."""
    if not DEFAULT_DIR.exists():
        return None
    cands = sorted(DEFAULT_DIR.glob("*_ckpt_latest.pt"))
    if len(cands) == 1:
        return cands[0]
    if len(cands) > 1:
        # Pick newest by mtime.
        return max(cands, key=lambda p: p.stat().st_mtime)
    return None


def find_training_log(ckpt_path: Path) -> Optional[Path]:
    """Find the training_log.jsonl that pairs with `ckpt_path`."""
    stem = ckpt_path.name.replace("_ckpt_latest.pt", "")
    cand = ckpt_path.parent / f"{stem}_training_log.jsonl"
    if cand.exists():
        return cand
    # Fallback: any *_training_log.jsonl in the dir.
    candidates = list(ckpt_path.parent.glob("*_training_log.jsonl"))
    return candidates[0] if candidates else None


def find_stdout_log(ckpt_path: Path) -> Optional[Path]:
    """Find the captured train_stdout.log next to the ckpt (used as fallback
    source of val_loss/val_ppl since the JSONL training log only stores
    train rows in the multi-ξ trainer)."""
    cand = ckpt_path.parent / "train_stdout.log"
    return cand if cand.exists() else None


_VAL_RE = None  # lazy-compiled


def parse_stdout_val_curve(stdout_path: Path) -> List[Dict[str, Any]]:
    """Parse lines like
        [multixi-splm] step  2000   val_loss=0.0440  val_ppl=1.05
    out of train_stdout.log into a list of {step, val_loss, val_ppl} dicts.
    """
    import re
    global _VAL_RE
    if _VAL_RE is None:
        _VAL_RE = re.compile(
            r"step\s+(\d+)\s+val_loss=([-+0-9.eE]+)\s+val_ppl=([-+0-9.eE]+)"
        )
    out: List[Dict[str, Any]] = []
    try:
        for line in stdout_path.read_text(encoding="utf-8",
                                          errors="replace").splitlines():
            m = _VAL_RE.search(line)
            if m:
                out.append({
                    "step": int(m.group(1)),
                    "val_loss": float(m.group(2)),
                    "val_ppl": float(m.group(3)),
                })
    except OSError:
        pass
    return out


# ---------------------------------------------------------------------------
# Probe + inflation (in-process; no subprocess)
# ---------------------------------------------------------------------------

def run_probe(ckpt_path: Path) -> Dict[str, Any]:
    """Load the ckpt under both causal_force={False, True}, run the
    causal-violation probe in each, and return the (pre, post) Δ values.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    label, cfg_cls, model_cls, keep = _dispatch(ck)
    vocab = int(keep.get("vocab_size", 257))
    max_len = int(keep.get("max_len", 64))
    T = min(64, max_len)
    t_pert = min(40, T - 4)

    out: Dict[str, Any] = {
        "label": label, "vocab_size": vocab, "max_len": max_len,
        "T": T, "t_pert": t_pert,
    }
    for mode_label, causal in [("buggy", False), ("fixed", True)]:
        cfg = cfg_cls(**keep)
        cfg.causal_force = causal
        m = model_cls(cfg)
        m.load_state_dict(ck["model_state_dict"])
        pre, post, _ = causal_violation_probe(
            m, vocab_size=vocab, T=T, t_pert=t_pert, seed=7,
        )
        out[mode_label] = {"causal_delta": pre, "after_delta": post}
        del m
    return out


def run_inflation(
    ckpt_path: Path, n_batches: int, batch: int, block: int, seed: int,
) -> Dict[str, Any]:
    """Eval val loss / PPL under both modes on identical batches; return
    a dict with buggy / fixed loss + ppl and the inflation factor."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    label, cfg_cls, model_cls, keep = _dispatch(ck)
    max_len = int(keep.get("max_len", block))
    if block > max_len:
        block = max_len
    corpus = _resolve_corpus("auto", ckpt_path, max_len)
    val_ids = _load_val_ids(corpus)

    rng = np.random.default_rng(seed)
    batches = [get_batch(val_ids, batch, block, rng) for _ in range(n_batches)]

    out: Dict[str, Any] = {
        "label": label, "device": device, "corpus": corpus,
        "n_batches": n_batches, "batch": batch, "block": block,
        "seed": seed, "val_tokens": len(val_ids),
    }
    for run_label, causal in [("buggy", False), ("fixed", True)]:
        cfg = cfg_cls(**keep)
        cfg.causal_force = causal
        m = model_cls(cfg).to(device)
        m.load_state_dict(ck["model_state_dict"])
        m.eval()
        losses = []
        for xb, yb in batches:
            x = torch.from_numpy(xb).to(device)
            y = torch.from_numpy(yb).to(device)
            _, loss = m(x, y)
            losses.append(loss.item())
        l = float(np.mean(losses))
        out[run_label] = {"loss": l, "ppl": math.exp(l)}
        del m
    out["inflation"] = out["fixed"]["ppl"] / out["buggy"]["ppl"]
    return out


# ---------------------------------------------------------------------------
# Training-log parsing
# ---------------------------------------------------------------------------

def parse_training_log(log_path: Path,
                       stdout_path: Optional[Path] = None) -> Dict[str, Any]:
    """Read a training_log.jsonl and extract α-trajectory, val_loss curve,
    final values, etc. If `stdout_path` is given and the JSONL has no
    val_loss rows, backfill the val curve from the stdout log."""
    rows: List[Dict[str, Any]] = []
    with log_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return {"log_path": str(log_path), "n_rows": 0}

    first = rows[0]
    last = rows[-1]
    alphas_first = first.get("xi_alphas")
    alphas_last = last.get("xi_alphas")
    val_rows = [r for r in rows if "val_loss" in r]
    val_curve = [
        {"step": r["step"], "val_loss": r["val_loss"],
         "val_ppl": math.exp(r["val_loss"])}
        for r in val_rows
    ]
    val_source = "jsonl"
    if not val_curve and stdout_path is not None:
        val_curve = parse_stdout_val_curve(stdout_path)
        if val_curve:
            val_source = "stdout"
    final_val = val_curve[-1] if val_curve else None

    drift = None
    if alphas_first is not None and alphas_last is not None:
        drift = [
            {"channel": k, "init": float(a0), "final": float(a1),
             "drift": float(a1 - a0)}
            for k, (a0, a1) in enumerate(zip(alphas_first, alphas_last))
        ]

    return {
        "log_path": str(log_path), "n_rows": len(rows),
        "first_step": int(first.get("step", 0)),
        "last_step": int(last.get("step", 0)),
        "alphas_first": alphas_first,
        "alphas_last": alphas_last,
        "alpha_drift": drift,
        "val_curve": val_curve,
        "val_source": val_source,
        "stdout_path": str(stdout_path) if stdout_path else None,
        "final_val": final_val,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _classify_ckpt(probe: Dict[str, Any], infl: Dict[str, Any]) -> str:
    """Return 'leak-corrected', 'buggy', 'regression', or 'ambiguous'.

    Discriminator keys (corrected after observing the leak-free K=4 multi-ξ
    pilot results, May 2 2026):

    1. **fixed-mode causal-side Δ** is the primary correctness check on the
       integrator: if it is > 1e-4 the fix did not take and the run is a
       *regression*.

    2. **inflation factor (= fixed_ppl / buggy_ppl)** is the primary
       discriminator between buggy-trained and leak-corrected ckpts:

       - **buggy-trained**: V_θ learned to route prediction signal through
         the leak channel. The buggy integrator gives it future info AND it
         knows how to use it → buggy-eval ppl << fixed-eval ppl →
         inflation >> 1 (we saw 389× on `multixi_buggy_2k`).

       - **leak-corrected**: V_θ never learned to use the leak channel.
         The buggy integrator still feeds future info into h_t mechanically
         (because the autograd update of h depends on V which depends on
         h_{>t}), but V_θ can't exploit it → that future info is destructive
         noise → buggy-eval ppl >> fixed-eval ppl → inflation << 1 (we saw
         0.002× / "0.00×" on the K=4 pilot below).

    3. The buggy-mode causal-side Δ is *not* a reliable training-mode
       discriminator: the buggy integrator is forward-noncausal at inference
       *regardless* of whether the trained V_θ uses the leak. The
       discriminator is whether that noncausality helps or hurts the loss.
    """
    pre_f = float(probe["fixed"]["causal_delta"])
    if pre_f > 1e-4:
        return "regression"
    inflation = float(infl.get("inflation", 1.0))
    if inflation > 10.0:
        return "buggy"
    if inflation < 0.5:
        return "leak-corrected"
    return "ambiguous"


def render_report(
    ckpt_path: Path, probe: Dict[str, Any], infl: Dict[str, Any],
    log: Dict[str, Any],
) -> str:
    """Return a markdown report block matching the §4.5 format in the bug doc."""
    cls = _classify_ckpt(probe, infl)
    cls_label = {
        "leak-corrected": "leak-corrected",
        "buggy": "buggy-trained",
        "regression": "REGRESSION (fix did not take)",
        "ambiguous": "ambiguous (inflation ≈ 1×)",
    }.get(cls, cls)
    run_dir = ckpt_path.parent.name

    lines: List[str] = []
    lines.append(f"# Post-completion forensics — `{run_dir}` ({cls_label})")
    lines.append("")
    lines.append(f"**ckpt:** `{ckpt_path}`")
    lines.append(f"**training log:** `{log.get('log_path', '?')}`  "
                 f"(rows: {log.get('n_rows', 0)})")
    lines.append("")

    # -- α-drift table --
    lines.append("## α_k drift (init → final)")
    lines.append("")
    drift = log.get("alpha_drift")
    if drift:
        lines.append("| channel | α init | α final | drift |")
        lines.append("|---:|---:|---:|---:|")
        for d in drift:
            lines.append(
                f"| {d['channel']} | {d['init']:.4f} | {d['final']:.4f} | {d['drift']:+.4f} |"
            )
    else:
        lines.append("_(no `xi_alphas` field in the training log)_")
    lines.append("")

    # -- val PPL trajectory --
    lines.append("## val PPL trajectory")
    lines.append("")
    vc = log.get("val_curve") or []
    if vc:
        lines.append("| step | val_loss | val_ppl |")
        lines.append("|---:|---:|---:|")
        for r in vc:
            lines.append(
                f"| {r['step']} | {r['val_loss']:.4f} | {r['val_ppl']:.2f} |"
            )
    else:
        lines.append("_(no val rows in training log)_")
    lines.append("")

    # -- causal probe --
    lines.append("## Causal-violation probe (same trained weights, two modes)")
    lines.append("")
    lines.append(f"matched class: `{probe['label']}`  "
                 f"(vocab={probe['vocab_size']}, max_len={probe['max_len']}, "
                 f"T={probe['T']}, t_pert={probe['t_pert']})")
    lines.append("")
    lines.append("| evaluator | causal-side Δ | after-side Δ |")
    lines.append("|---|---:|---:|")
    for k in ("buggy", "fixed"):
        lines.append(
            f"| {k} | {probe[k]['causal_delta']:.4e} | {probe[k]['after_delta']:.4e} |"
        )
    lines.append("")
    lines.append(
        "Expected: fixed-mode Δ ≈ 0 (the post-fix integrator is causal). "
        "Buggy-mode Δ may still be > 0 because the buggy integrator is "
        "forward-noncausal at inference *regardless* of how the V_θ was "
        "trained — its h-update depends on V which depends on h_{>t}. "
        "Whether that noncausality helps or hurts the loss is what "
        "discriminates buggy-trained from leak-corrected ckpts (see "
        "inflation table below)."
    )
    lines.append("")

    # -- inflation --
    lines.append("## Val-PPL inflation (same trained weights, two evaluators)")
    lines.append("")
    lines.append(f"corpus: `{infl['corpus']}`   "
                 f"batches: {infl['n_batches']} × {infl['batch']} × {infl['block']}"
                 f" = {infl['n_batches'] * infl['batch'] * infl['block']:,} tokens")
    lines.append("")
    lines.append("| evaluator | val_loss | val_ppl |")
    lines.append("|---|---:|---:|")
    lines.append(
        f"| buggy (pre-fix integrator) | {infl['buggy']['loss']:.4f} | "
        f"{infl['buggy']['ppl']:.2f} |"
    )
    lines.append(
        f"| fixed (post-fix integrator) | {infl['fixed']['loss']:.4f} | "
        f"{infl['fixed']['ppl']:.2f} |"
    )
    lines.append(f"| **inflation factor** | | **{infl['inflation']:.2f}×** |")
    lines.append("")
    lines.append(
        "Expected:\n"
        "- **buggy-trained**: inflation >> 1 (e.g., 389× on `multixi_buggy_2k`). "
        "V_θ exploits the leak; under fixed eval the leak is severed and "
        "ppl shoots up.\n"
        "- **leak-corrected**: inflation << 1. V_θ never learned to use the "
        "leak. The buggy integrator still injects future info into h_t, but "
        "V_θ treats it as destructive noise → ppl explodes under buggy eval.\n"
        "- **regression** (fix did not take during training): inflation ≈ 1×."
    )
    lines.append("")

    # -- direct comparison to buggy multi-ξ --
    lines.append("## Headline comparison")
    lines.append("")
    final_val = log.get("final_val")
    fv_str = f"{final_val['val_ppl']:.2f}" if final_val else "?"
    lines.append("| run | final train val_ppl | fixed-eval val_ppl | "
                 "inflation | causal-side Δ (fixed mode) |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        "| buggy multi-ξ 2k (`multixi_buggy_2k`) | 1.05 | 408.12 | "
        "389× | 0.0000 |"
    )
    lines.append(
        f"| **`{run_dir}` ({cls_label})** | **{fv_str}** | "
        f"**{infl['fixed']['ppl']:.2f}** | **{infl['inflation']:.2f}×** | "
        f"**{probe['fixed']['causal_delta']:.4e}** |"
    )
    lines.append("")
    return "\n".join(lines)


def to_json(
    ckpt_path: Path, probe: Dict[str, Any], infl: Dict[str, Any],
    log: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "ckpt": str(ckpt_path),
        "probe": probe,
        "inflation": infl,
        "training_log": log,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Post-completion forensics for the fixed multi-ξ pilot."
    )
    ap.add_argument(
        "--ckpt", type=Path, default=None,
        help=f"Path to the *_ckpt_latest.pt to forensically evaluate. "
             f"If omitted, auto-discovers in {DEFAULT_DIR}.",
    )
    ap.add_argument("--n-batches", type=int, default=20,
                    help="Number of val batches (default 20).")
    ap.add_argument("--batch", type=int, default=8,
                    help="Batch size (default 8).")
    ap.add_argument("--block", type=int, default=256,
                    help="Block length, capped at ckpt max_len (default 256).")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for batch sampling (default 0).")
    ap.add_argument("--out-md", type=Path, default=None,
                    help="Write the markdown report to this path. "
                         "Default: <ckpt-dir>/post_fixed_pilot_report.md")
    ap.add_argument("--out-json", type=Path, default=None,
                    help="Write the JSON dump to this path. "
                         "Default: <ckpt-dir>/post_fixed_pilot_report.json")
    args = ap.parse_args()

    ckpt_path = args.ckpt
    if ckpt_path is None:
        ckpt_path = find_default_ckpt()
        if ckpt_path is None:
            print(f"[post-pilot] no ckpt under {DEFAULT_DIR}; pass --ckpt PATH.",
                  file=sys.stderr)
            return 2
    if not ckpt_path.exists():
        print(f"[post-pilot] ckpt not found: {ckpt_path}", file=sys.stderr)
        return 2

    log_path = find_training_log(ckpt_path)
    print(f"[post-pilot] ckpt: {ckpt_path}")
    print(f"[post-pilot] log : {log_path}")

    print("[post-pilot] running causal-violation probe (both modes)...")
    probe = run_probe(ckpt_path)
    for k in ("buggy", "fixed"):
        print(f"  {k:>5}: causal Δ={probe[k]['causal_delta']:.4e}  "
              f"after Δ={probe[k]['after_delta']:.4e}")

    print(f"[post-pilot] running inflation eval ({args.n_batches} × "
          f"{args.batch} × {args.block} = "
          f"{args.n_batches * args.batch * args.block:,} tokens)...")
    infl = run_inflation(
        ckpt_path, args.n_batches, args.batch, args.block, args.seed,
    )
    print(f"  buggy: loss={infl['buggy']['loss']:.4f}  ppl={infl['buggy']['ppl']:.2f}")
    print(f"  fixed: loss={infl['fixed']['loss']:.4f}  ppl={infl['fixed']['ppl']:.2f}")
    print(f"  inflation: {infl['inflation']:.2f}×")

    log_data: Dict[str, Any] = {}
    if log_path is not None and log_path.exists():
        stdout_path = find_stdout_log(ckpt_path)
        print(f"[post-pilot] parsing training log..."
              f"{' (with stdout fallback)' if stdout_path else ''}")
        log_data = parse_training_log(log_path, stdout_path=stdout_path)

    md = render_report(ckpt_path, probe, infl, log_data)
    blob = to_json(ckpt_path, probe, infl, log_data)

    out_md = args.out_md or (ckpt_path.parent / "post_fixed_pilot_report.md")
    out_json = args.out_json or (ckpt_path.parent / "post_fixed_pilot_report.json")
    out_md.write_text(md, encoding="utf-8")
    out_json.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    print(f"[post-pilot] wrote: {out_md}")
    print(f"[post-pilot] wrote: {out_json}")
    print()
    print("=" * 76)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
