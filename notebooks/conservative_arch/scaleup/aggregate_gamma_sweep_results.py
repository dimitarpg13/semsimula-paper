"""
Aggregate the SPLM em_ln γ-sweep diagnostic at scaleup.

The γ-sweep tests the hypothesis that the small-scale γ⋆ ≈ 0.166 (E5
winner) inflates by ~1.8× at scaleup to ≈ 0.30 (the colab_pilot Arm 2
winner).  We sweep γ ∈ {0.166, 0.20, 0.25, 0.30, 0.35} (one seed per γ
by default) and read the per-γ artifacts from `--results-dir`, expecting
filenames produced by `train_splm_em_ln_scaleup.py` with a tag suffix of
the form `g{int(gamma*1000):03d}_seed{seed}`, e.g.

    splm_em_ln_scaleup_scaleup_g300_seed0_summary.md
    splm_em_ln_scaleup_scaleup_g300_seed0_ckpt_latest.pt
    splm_em_ln_scaleup_scaleup_g300_seed0_training_log.jsonl

Outputs:

  - GAMMA_SWEEP_RESULTS.md  : table of γ → val PPL / train-val gap /
                              wall-clock, plus a paragraph identifying
                              the γ⋆ at scaleup (argmin val PPL) and the
                              implied scaleup multiplier vs the small-
                              scale γ⋆ ≈ 0.166.
  - gamma_sweep.png          : two panels — (left) val PPL vs γ;
                              (right) train→val gap vs γ.

Usage:
  python3 aggregate_gamma_sweep_results.py \
      [--results-dir DIR] [--seed 0] [--out-dir OUT] \
      [--gammas 0.166,0.20,0.25,0.30,0.35]
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
SMALL_SCALE_GAMMA_STAR = 0.166  # E5 winner @ small scale (D=128, L=4)

# γ values reported by the colab pilot's Arm 2 (E9 scale-up) as the
# default for the sweep.  Override on the CLI via --gammas if desired.
DEFAULT_GAMMAS = [0.166, 0.20, 0.25, 0.30, 0.35]


def _gamma_tag(gamma: float) -> str:
    """Encode a γ value as the suffix used by the trainer's tag.

    e.g. 0.30 -> 'g300', 0.166 -> 'g166', 0.205 -> 'g205'.
    """
    return f"g{int(round(gamma * 1000)):03d}"


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


def _read_final_train_loss(ckpt_path: Path) -> Optional[float]:
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


def aggregate(results_dir: Path, seed: int, gammas: List[float],
              out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Tuple[float, Optional[dict]]] = []
    print(f"[aggregate] scanning {results_dir} for γ-sweep "
          f"(seed={seed}, γ ∈ {gammas}) ...")
    for gamma in gammas:
        gtag = _gamma_tag(gamma)
        pattern = (f"splm_em_ln_scaleup_scaleup_{gtag}_seed{seed}"
                   "_ckpt_latest.pt")
        ckpt = _glob_one(results_dir, pattern)
        if ckpt is None:
            print(f"  [miss] γ={gamma:.3f}  pattern={pattern}")
            rows.append((gamma, None))
            continue
        meta = _load_ckpt_meta(ckpt)
        meta["_ckpt"] = ckpt
        ppl = meta.get("final_val_ppl")
        print(f"  [ok]   γ={gamma:.3f}  ppl={ppl}")
        rows.append((gamma, meta))

    md_path = out_dir / "GAMMA_SWEEP_RESULTS.md"
    with md_path.open("w") as f:
        f.write("# SPLM em_ln γ-sweep diagnostic at scaleup (E9 protocol)\n\n")
        f.write(f"- seed: {seed}\n")
        f.write("- corpus: TinyStories (~5 M GPT-2 BPE tokens)\n")
        f.write("- block_size: 512  batch_size: 16  steps: 8000\n")
        f.write("- d=256, L=8, max_len=1024\n")
        f.write(f"- small-scale γ⋆ (E5 winner @ D=128, L=4): "
                f"**{SMALL_SCALE_GAMMA_STAR:.3f}**\n\n")

        f.write("## γ → outcome table\n\n")
        f.write("| γ (fixed) | val PPL | val loss (nats) | "
                "train→val gap (nats) | wall-clock (h) |\n")
        f.write("|---:|---:|---:|---:|---:|\n")
        present: List[Tuple[float, dict]] = []
        for gamma, meta in rows:
            if meta is None:
                f.write(f"| {gamma:.3f} | _MISSING_ | — | — | — |\n")
                continue
            ppl = meta.get("final_val_ppl")
            vl = meta.get("final_val_loss")
            tl = meta.get("final_train_loss")
            gap = (vl - tl) if (vl is not None and tl is not None) else None
            wc_h = (meta.get("elapsed_sec") or 0.0) / 3600.0
            f.write(
                f"| {gamma:.3f} | "
                f"{ppl:.3f} | {vl:.3f} | "
                f"{gap:.3f} | "
                f"{wc_h:.2f} |\n"
                if (ppl is not None and vl is not None and gap is not None)
                else
                f"| {gamma:.3f} | "
                f"{('—' if ppl is None else f'{ppl:.3f}')} | "
                f"{('—' if vl is None else f'{vl:.3f}')} | "
                f"{('—' if gap is None else f'{gap:.3f}')} | "
                f"{wc_h:.2f} |\n"
            )
            present.append((gamma, meta))

        # γ⋆ identification + scaleup-multiplier interpretation ---------
        if present:
            valid = [(g, m) for g, m in present
                     if m.get("final_val_ppl") is not None]
            if valid:
                g_star, m_star = min(
                    valid,
                    key=lambda gm: gm[1]["final_val_ppl"],
                )
                ppl_star = m_star["final_val_ppl"]
                multiplier = g_star / SMALL_SCALE_GAMMA_STAR
                f.write("\n## γ⋆ at scaleup\n\n")
                f.write(
                    f"- argmin γ→PPL: **γ⋆ = {g_star:.3f}** "
                    f"(val PPL = **{ppl_star:.3f}**)\n"
                )
                f.write(
                    f"- small-scale γ⋆ (E5 winner): "
                    f"**{SMALL_SCALE_GAMMA_STAR:.3f}**\n"
                )
                f.write(
                    f"- scaleup multiplier: γ⋆_scaleup / γ⋆_small = "
                    f"**{multiplier:.2f}×**\n\n"
                )
                if abs(multiplier - 1.8) < 0.25:
                    f.write(
                        "→ Multiplier is within ±0.25 of the predicted "
                        "**1.8×** scaleup factor (predicted by the colab "
                        "pilot's Arm 2 finding that γ=0.30 dominated the "
                        "free-γ run at this scale).  "
                        "**Hypothesis confirmed.**\n"
                    )
                else:
                    f.write(
                        "→ Multiplier deviates from the predicted **1.8×** "
                        "scaleup factor by more than ±0.25.  Re-examine the "
                        "colab pilot's Arm 2 free-γ trajectory and consider "
                        "extending the sweep range.\n"
                    )

                # Sensitivity diagnosis
                ppls = [m["final_val_ppl"] for _g, m in valid]
                rng = max(ppls) - min(ppls)
                if rng < 0.20:
                    f.write(
                        f"\n**Note:** γ-sweep range = {rng:.3f} PPL "
                        f"(< 0.20 PPL).  Final PPL is **insensitive** to γ "
                        "in the sweep band — γ acts as an overdamped "
                        "regularizer and the choice within this band does "
                        "not change the architectural conclusion.\n"
                    )
                else:
                    f.write(
                        f"\n**Note:** γ-sweep range = {rng:.3f} PPL.  γ has "
                        "a measurable but modest effect on final val PPL "
                        "at scaleup.\n"
                    )

        f.write("\n## Per-γ artifact paths\n\n")
        for gamma, meta in rows:
            if meta is None:
                f.write(f"- γ={gamma:.3f}: _missing_\n")
            else:
                f.write(f"- γ={gamma:.3f}: `{meta['_ckpt'].name}`\n")

    print(f"[aggregate] wrote {md_path}")

    # --- Plot: val PPL vs γ + train-val gap vs γ ----------------------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.5))

    g_arr, ppl_arr, gap_arr, wc_arr = [], [], [], []
    for gamma, meta in rows:
        if meta is None or meta.get("final_val_ppl") is None:
            continue
        g_arr.append(gamma)
        ppl_arr.append(meta["final_val_ppl"])
        vl = meta.get("final_val_loss")
        tl = meta.get("final_train_loss")
        gap_arr.append((vl - tl) if (vl is not None and tl is not None)
                       else float("nan"))
        wc_arr.append((meta.get("elapsed_sec") or 0.0) / 3600.0)

    if g_arr:
        axL.plot(g_arr, ppl_arr, marker="o", color="tab:blue",
                 lw=1.6, markersize=8)
        for g, p in zip(g_arr, ppl_arr):
            axL.annotate(f"  γ={g:.3f}\n  PPL={p:.2f}",
                         (g, p), fontsize=8,
                         xytext=(6, 6), textcoords="offset points")
        if SMALL_SCALE_GAMMA_STAR is not None:
            axL.axvline(SMALL_SCALE_GAMMA_STAR, color="tab:gray",
                        ls="--", alpha=0.7,
                        label=f"E5 small-scale γ⋆={SMALL_SCALE_GAMMA_STAR:.3f}")
            axL.legend(fontsize=8, loc="best")

    axL.set_xlabel("γ (fixed damping coefficient)")
    axL.set_ylabel("final val PPL (TinyStories)")
    axL.set_title("γ-sweep — val PPL vs γ")
    axL.grid(True, alpha=0.3)

    if g_arr:
        axR.plot(g_arr, gap_arr, marker="s", color="tab:red",
                 lw=1.6, markersize=8)
        for g, gap in zip(g_arr, gap_arr):
            if math.isnan(gap):
                continue
            axR.annotate(f"  γ={g:.3f}\n  gap={gap:.3f}",
                         (g, gap), fontsize=8,
                         xytext=(6, 6), textcoords="offset points")
    axR.axhline(0.0, color="black", lw=1.0, alpha=0.6)
    axR.set_xlabel("γ (fixed damping coefficient)")
    axR.set_ylabel("train→val gap (nats)")
    axR.set_title("γ-sweep — generalization gap vs γ")
    axR.grid(True, alpha=0.3)

    fig.suptitle("SPLM em_ln γ-sweep diagnostic at scaleup",
                 fontsize=11)
    fig.tight_layout()
    out_png = out_dir / "gamma_sweep.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"[aggregate] wrote {out_png}")

    print(f"[aggregate] done.  artifacts in {out_dir}")
    return 0


def _parse_gammas(spec: str) -> List[float]:
    return [float(x) for x in spec.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR),
                    help="Directory containing per-γ *_ckpt_latest.pt files.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gammas",
                    default=",".join(f"{g:.3f}" for g in DEFAULT_GAMMAS),
                    help="Comma-separated γ list.")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()
    out_dir = (Path(args.out_dir).expanduser().resolve()
               if args.out_dir else results_dir)
    return aggregate(results_dir, args.seed,
                     _parse_gammas(args.gammas), out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
