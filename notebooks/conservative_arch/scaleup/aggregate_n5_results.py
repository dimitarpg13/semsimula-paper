"""
Aggregate the paper_tmlr_1 n=5 paired confirmation runs.

The n=5 confirmation pass repeats three arms five times each (one seed per
run, same hyperparameters) so that the seed-0 pilot finding can be
upgraded from a single point estimate to a paired comparison with
quantified uncertainty:

  1. matched-attn baseline                         (`train_matched_baseline_scaleup.py`)
  2. Helmholtz Q9d AAAASSSS                        (`train_helmholtz_scaleup.py`)
  3. Hybrid VA (k=4, m=4)                          (`train_hybrid_scaleup.py`)

For each arm we read all matching `*_seed{S}_ckpt_latest.pt` files in
`--results-dir`, extract the final val PPL / train-val gap / wall-clock,
and write:

  - PILOT_N5_RESULTS.md  : per-arm mean ± std + 95% CI tables, plus
                            **paired** Δ vs matched-attention baseline
                            (paired by seed: Δ_s = ppl_arm[s] − ppl_base[s];
                            mean Δ and 95% CI from the t-distribution with
                            df = n − 1).
  - n5_paired_strip.png  : per-arm strip plot of per-seed val PPL with
                            mean + 95% CI overlay (one panel for absolute
                            PPL, one panel for paired Δ vs baseline).

Statistical convention (n = 5):
  - mean ± std reported per arm
  - 95% CI half-width = t_{0.025, df=n-1} · s / √n
  - paired Δ uses the per-seed pairing (the same seed produces matched
    micro-batch ordering across arms, so most of the data noise is
    differenced out)

Usage:
  python3 aggregate_n5_results.py \
      [--results-dir DIR] [--seeds 0,1,2,3,4] [--out-dir OUT]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"

# Two-sided 95% t-critical values, indexed by degrees of freedom (n-1).
# Hardcoded so the script has no scipy dependency at runtime.
T_CRIT_95: Dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447,  7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    19: 2.093, 24: 2.064, 29: 2.045, 49: 2.010, 99: 1.984,
}


def _t_crit_95(df: int) -> float:
    if df in T_CRIT_95:
        return T_CRIT_95[df]
    if df < 1:
        return float("nan")
    candidates = sorted(T_CRIT_95.keys())
    return T_CRIT_95[min(candidates, key=lambda k: abs(k - df))]


# Each ARM tuple: (display_name, ckpt_filename_glob, color, marker)
ARMS = [
    ("matched-attn baseline",
     "matched_baseline_scaleup_*_seed{seed}_ckpt_latest.pt",
     "tab:gray",   "o"),
    ("Helmholtz Q9d (AAAASSSS)",
     "helmholtz_*_scaleup_scaleup*_seed{seed}_ckpt_latest.pt",
     "tab:red",    "D"),
    ("Hybrid VA (k=4, m=4)",
     "hybrid_VA_*_scaleup_scaleup*_seed{seed}_ckpt_latest.pt",
     "tab:green",  "^"),
]
BASELINE_DISPLAY = "matched-attn baseline"


def _glob_one(directory: Path, pattern: str) -> Optional[Path]:
    matches = sorted(directory.glob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        matches.sort(key=lambda p: p.stat().st_mtime)
    return matches[-1]


def _count_state_dict_params(state_dict) -> int:
    total = 0
    for v in state_dict.values():
        if hasattr(v, "numel"):
            total += int(v.numel())
    return total


def _load_ckpt_meta(p: Path) -> dict:
    raw = torch.load(p, map_location="cpu", weights_only=False)
    keys_of_interest = (
        "tag", "variant", "seed", "fixed_gamma",
        "final_val_loss", "final_val_ppl", "final_gamma",
        "elapsed_sec", "n_params",
    )
    out = {k: raw.get(k) for k in keys_of_interest}
    if out.get("n_params") is None:
        sd = raw.get("model_state_dict")
        if sd is not None:
            out["n_params"] = _count_state_dict_params(sd)
    if (out.get("final_val_ppl") is None
            and out.get("final_val_loss") is not None):
        out["final_val_ppl"] = math.exp(out["final_val_loss"])
    out["final_train_loss"] = _read_final_train_loss(p)
    return out


def _read_final_train_loss(ckpt_path: Path) -> Optional[float]:
    """Pull the last `train_loss` record from the JSONL training log
    that lives next to the checkpoint."""
    base = ckpt_path.name.replace("_ckpt_latest.pt", "")
    log_path = ckpt_path.parent / f"{base}_training_log.jsonl"
    if not log_path.exists():
        return None
    last: Optional[float] = None
    last_step = -1
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "train_loss" in rec and "step" in rec:
            step = int(rec["step"])
            if step >= last_step:
                last_step = step
                last = float(rec["train_loss"])
    return last


def _summarize(values: List[float]) -> Tuple[float, float, float, float]:
    """Return (mean, std (n-1), ci_half, df).  Returns NaN if n < 2."""
    n = len(values)
    if n == 0:
        return (float("nan"),) * 4
    mean = float(np.mean(values))
    if n == 1:
        return (mean, float("nan"), float("nan"), 0.0)
    std = float(np.std(values, ddof=1))
    df = n - 1
    ci_half = _t_crit_95(df) * std / math.sqrt(n)
    return (mean, std, ci_half, float(df))


def aggregate(results_dir: Path, seeds: List[int], out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)

    # arm_data[display] = {seed: meta_dict}
    arm_data: Dict[str, Dict[int, dict]] = {a[0]: {} for a in ARMS}
    arm_specs = {a[0]: a for a in ARMS}

    print(f"[aggregate] scanning {results_dir} for seeds={seeds} ...")
    for display, glob_pat, _color, _marker in ARMS:
        for seed in seeds:
            pattern = glob_pat.format(seed=seed)
            ckpt = _glob_one(results_dir, pattern)
            if ckpt is None:
                print(f"  [miss] {display:<30s}  seed={seed}  pattern={pattern}")
                continue
            meta = _load_ckpt_meta(ckpt)
            meta["_ckpt"] = ckpt
            arm_data[display][seed] = meta
            ppl = meta.get("final_val_ppl")
            print(f"  [ok]   {display:<30s}  seed={seed}  ppl={ppl}")

    md_path = out_dir / "PILOT_N5_RESULTS.md"
    with md_path.open("w") as f:
        f.write("# paper_tmlr_1 — n=5 paired confirmation results\n\n")
        f.write(f"- seeds: {seeds}\n")
        f.write("- corpus: TinyStories (~5 M GPT-2 BPE tokens)\n")
        f.write("- block_size: 512  batch_size: 16  steps: 8000\n")
        f.write("- d=256, L=8, max_len=1024  "
                "(matches E9 SPLM scale-up protocol; identical to colab_pilot)\n\n")

        # Per-arm marginal table -----------------------------------------
        f.write("## Per-arm marginal statistics\n\n")
        f.write(
            "| Arm | n | val PPL  mean ± std | 95% CI (mean) | "
            "train→val gap mean ± std | wall-clock mean (h) |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|\n")
        per_arm_summary: Dict[str, dict] = {}
        for display, _g, _c, _m in ARMS:
            seed_to_meta = arm_data[display]
            ppls = [m["final_val_ppl"]
                    for m in seed_to_meta.values()
                    if m.get("final_val_ppl") is not None]
            gaps = []
            for m in seed_to_meta.values():
                tl = m.get("final_train_loss")
                vl = m.get("final_val_loss")
                if tl is not None and vl is not None:
                    gaps.append(vl - tl)
            elapsed = [(m.get("elapsed_sec") or 0.0) / 3600.0
                       for m in seed_to_meta.values()]
            ppl_mean, ppl_std, ppl_ci, _ = _summarize(ppls)
            gap_mean, gap_std, _, _ = _summarize(gaps)
            wc_mean = float(np.mean(elapsed)) if elapsed else float("nan")
            n = len(ppls)
            per_arm_summary[display] = dict(
                ppls=ppls, ppl_mean=ppl_mean, ppl_std=ppl_std,
                ppl_ci=ppl_ci, n=n,
                gap_mean=gap_mean, gap_std=gap_std,
                wc_mean=wc_mean,
            )
            f.write(
                f"| {display} | {n} | "
                f"{_fmt_pm(ppl_mean, ppl_std)} | "
                f"{_fmt_ci(ppl_mean, ppl_ci)} | "
                f"{_fmt_pm(gap_mean, gap_std)} | "
                f"{wc_mean:.2f} |\n"
            )

        # Paired Δ vs matched-attention baseline -------------------------
        f.write("\n## Paired Δ vs matched-attention baseline\n\n")
        f.write(
            "Per-seed Δ_s = PPL_arm[s] − PPL_base[s] (paired by seed; "
            "negative ⇒ arm beats baseline).  Mean Δ uses **only** seeds "
            "where both arm and baseline have a valid result.  95% CI uses "
            "Student's t with df = n − 1.\n\n"
        )
        baseline_data = arm_data[BASELINE_DISPLAY]
        f.write(
            "| Arm | n_paired | per-seed Δ | mean Δ | 95% CI (mean Δ) | verdict |\n"
        )
        f.write("|---|---:|---|---:|---:|---|\n")
        for display, _g, _c, _m in ARMS:
            if display == BASELINE_DISPLAY:
                continue
            paired = []
            for seed in seeds:
                bm = baseline_data.get(seed)
                am = arm_data[display].get(seed)
                if (bm is None or am is None
                        or bm.get("final_val_ppl") is None
                        or am.get("final_val_ppl") is None):
                    continue
                paired.append((seed,
                               am["final_val_ppl"] - bm["final_val_ppl"]))
            if not paired:
                f.write(f"| {display} | 0 | — | — | — | _no paired runs_ |\n")
                continue
            deltas = [d for _s, d in paired]
            d_mean, _d_std, d_ci, _df = _summarize(deltas)
            per_seed_str = ", ".join(
                f"s{s}: {d:+.3f}" for s, d in paired
            )
            verdict = _verdict(d_mean, d_ci)
            f.write(
                f"| {display} | {len(paired)} | {per_seed_str} | "
                f"{d_mean:+.3f} | {_fmt_ci(d_mean, d_ci)} | {verdict} |\n"
            )

        # Per-seed table for full transparency --------------------------
        f.write("\n## Per-seed val PPL (full transparency)\n\n")
        f.write("| Seed | " +
                " | ".join(a[0] for a in ARMS) + " |\n")
        f.write("|---:|" + "---:|" * len(ARMS) + "\n")
        for seed in seeds:
            row = [str(seed)]
            for display, _g, _c, _m in ARMS:
                m = arm_data[display].get(seed)
                if m is None or m.get("final_val_ppl") is None:
                    row.append("—")
                else:
                    row.append(f"{m['final_val_ppl']:.3f}")
            f.write("| " + " | ".join(row) + " |\n")

        # Generalization gap section ------------------------------------
        f.write("\n## Generalization gap (final val − final train, in nats)\n\n")
        f.write(
            "Smaller gap with comparable val PPL ⇒ tighter generalization "
            "(structural regularization).  All values reported as mean ± std "
            "across the n paired seeds.\n\n"
        )
        f.write("| Arm | n | gap mean ± std |\n")
        f.write("|---|---:|---:|\n")
        for display, _g, _c, _m in ARMS:
            s = per_arm_summary[display]
            f.write(f"| {display} | {s['n']} | "
                    f"{_fmt_pm(s['gap_mean'], s['gap_std'])} |\n")

        # Per-arm artifact paths ----------------------------------------
        f.write("\n## Per-arm artifact paths (per-seed checkpoints)\n\n")
        for display, _g, _c, _m in ARMS:
            f.write(f"- **{display}**\n")
            for seed in seeds:
                m = arm_data[display].get(seed)
                if m is None:
                    f.write(f"    - seed {seed}: _missing_\n")
                else:
                    f.write(f"    - seed {seed}: `{m['_ckpt'].name}`\n")

    print(f"[aggregate] wrote {md_path}")

    # --- Strip plot (left: absolute PPL, right: paired Δ vs base) -------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 5.0))
    base_per_seed = {s: m.get("final_val_ppl")
                     for s, m in baseline_data.items()
                     if m.get("final_val_ppl") is not None}

    x_ticks_left, labels_left = [], []
    x_ticks_right, labels_right = [], []
    for i, (display, _g, color, marker) in enumerate(ARMS):
        s = per_arm_summary[display]
        ppls = s["ppls"]
        rng = np.random.default_rng(seed=i)
        jitter = rng.normal(0.0, 0.04, size=len(ppls))
        x = np.full(len(ppls), float(i)) + jitter
        axL.scatter(x, ppls, color=color, marker=marker, s=70,
                    edgecolor="black", linewidth=0.5,
                    alpha=0.85, label=display)
        if not (math.isnan(s["ppl_mean"]) or math.isnan(s["ppl_ci"])):
            axL.errorbar([float(i)], [s["ppl_mean"]],
                         yerr=[s["ppl_ci"]],
                         fmt="_", color=color, capsize=8, lw=2.0,
                         markersize=24)
        x_ticks_left.append(i)
        labels_left.append(display.replace(" ", "\n", 1))

    # Right panel: paired Δ vs baseline
    j = 0
    for display, _g, color, marker in ARMS:
        if display == BASELINE_DISPLAY:
            continue
        deltas = []
        for seed in sorted(arm_data[display].keys()):
            bm = baseline_data.get(seed)
            am = arm_data[display][seed]
            if (bm is None or am.get("final_val_ppl") is None
                    or bm.get("final_val_ppl") is None):
                continue
            deltas.append(am["final_val_ppl"] - bm["final_val_ppl"])
        if not deltas:
            j += 1
            continue
        rng = np.random.default_rng(seed=100 + j)
        jitter = rng.normal(0.0, 0.04, size=len(deltas))
        x = np.full(len(deltas), float(j)) + jitter
        axR.scatter(x, deltas, color=color, marker=marker, s=70,
                    edgecolor="black", linewidth=0.5, alpha=0.85,
                    label=display)
        d_mean, _d_std, d_ci, _df = _summarize(deltas)
        if not (math.isnan(d_mean) or math.isnan(d_ci)):
            axR.errorbar([float(j)], [d_mean], yerr=[d_ci],
                         fmt="_", color=color, capsize=8, lw=2.0,
                         markersize=24)
        x_ticks_right.append(j)
        labels_right.append(display.replace(" ", "\n", 1))
        j += 1

    axL.set_xticks(x_ticks_left)
    axL.set_xticklabels(labels_left, fontsize=9)
    axL.set_ylabel("final validation perplexity")
    axL.set_title("absolute val PPL (per-seed, mean ± 95% CI)")
    axL.grid(True, axis="y", alpha=0.3)

    axR.axhline(0.0, color="black", lw=1.0, alpha=0.6)
    axR.set_xticks(x_ticks_right)
    axR.set_xticklabels(labels_right, fontsize=9)
    axR.set_ylabel("Δ PPL (arm − matched-attn) [paired by seed]")
    axR.set_title("paired Δ vs matched-attn baseline (mean ± 95% CI)")
    axR.grid(True, axis="y", alpha=0.3)

    fig.suptitle("paper_tmlr_1 n=5 paired confirmation",
                 fontsize=11)
    fig.tight_layout()
    out_png = out_dir / "n5_paired_strip.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"[aggregate] wrote {out_png}")

    print(f"[aggregate] done.  artifacts in {out_dir}")
    return 0


def _fmt_pm(mean: float, std: float) -> str:
    if math.isnan(mean):
        return "—"
    if math.isnan(std):
        return f"{mean:.3f} ± —"
    return f"{mean:.3f} ± {std:.3f}"


def _fmt_ci(mean: float, half: float) -> str:
    if math.isnan(mean) or math.isnan(half):
        return "—"
    return f"[{mean - half:+.3f}, {mean + half:+.3f}]"


def _verdict(mean_delta: float, ci_half: float) -> str:
    if math.isnan(mean_delta) or math.isnan(ci_half):
        return "—"
    lo = mean_delta - ci_half
    hi = mean_delta + ci_half
    if hi < 0.0:
        return "**arm wins** (CI below 0)"
    if lo > 0.0:
        return "_baseline wins_ (CI above 0)"
    return "tie (CI straddles 0)"


def _parse_seeds(spec: str) -> List[int]:
    return [int(x) for x in spec.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR),
                    help="Directory containing per-seed *_ckpt_latest.pt files.")
    ap.add_argument("--seeds", default="0,1,2,3,4",
                    help="Comma-separated seed list (default 0,1,2,3,4).")
    ap.add_argument("--out-dir", default=None,
                    help="Where to write PILOT_N5_RESULTS.md + plots. "
                         "Defaults to --results-dir.")
    args = ap.parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()
    out_dir = (Path(args.out_dir).expanduser().resolve()
               if args.out_dir else results_dir)
    seeds = _parse_seeds(args.seeds)
    return aggregate(results_dir, seeds, out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
