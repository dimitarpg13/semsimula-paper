"""
Cross-variant comparison for the energetic-minima study.

For each of the four variants
    (baseline SARF+mass, LN, SG, GM)
this script reads
    - the training summary (val_loss, val_ppl, param_count)
    - the attractor-extraction summary (per-prompt K*, V range, basin
      size distribution, top-1 decoded token of the largest basin)
and writes a single comparison_report.md + comparison_table.json under
energetic_minima/results/.

Run AFTER run_attractor_pipeline.sh.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

SCRIPT_DIR = Path(__file__).parent
ROOT       = SCRIPT_DIR.parent
EM_RESULTS = SCRIPT_DIR / "results"
ATTR_RES   = ROOT / "attractor_analysis" / "results"
BASE_RES   = ROOT / "sarf_mass_variant" / "results"

# variant tag -> (training_log.jsonl, attractor_summary.md, pretty-name, param_label)
VARIANTS = [
    ("em_base",
     BASE_RES / "splm_sarfmass_logfreq_shakespeare_training_log.jsonl",
     ATTR_RES / "attractors_em_base_summary.md",
     "baseline SARF+mass (logfreq)"),
    ("em_ln",
     EM_RESULTS / "em_ln_shakespeare_training_log.jsonl",
     ATTR_RES / "attractors_em_ln_summary.md",
     "(i) LayerNorm-after-step"),
    ("em_sg",
     EM_RESULTS / "em_sg_lam1e-03_shakespeare_training_log.jsonl",
     ATTR_RES / "attractors_em_sg_summary.md",
     "(ii) scale-gauge (lambda_V0=1e-3)"),
    ("em_gm",
     EM_RESULTS / "em_gm_K64_shakespeare_training_log.jsonl",
     ATTR_RES / "attractors_em_gm_summary.md",
     "(iii) Gaussian-mixture head (K=64)"),
]

PROMPTS = ["narrative", "mathematics", "scientific", "dialogue", "code"]


def parse_training_log(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    last_val_ppl = None
    last_val_loss = None
    last_train_loss = None
    with path.open() as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "val_ppl" in d:
                last_val_ppl = d["val_ppl"]
            if "val_loss" in d:
                last_val_loss = d["val_loss"]
            if "train_loss_eval" in d:
                last_train_loss = d["train_loss_eval"]
    return {
        "val_ppl": last_val_ppl,
        "val_loss": last_val_loss,
        "train_loss": last_train_loss,
    }


def parse_attractor_summary(path: Path) -> Dict:
    """Per-prompt K*, V range across basins, top-1 token of biggest basin."""
    if not path.exists():
        return {}
    text = path.read_text()
    out: Dict[str, Dict] = {}
    current_prompt: Optional[str] = None
    for line in text.splitlines():
        m = re.match(r"^##\s+Prompt\s+\(([^)]+)\):", line)
        if m:
            current_prompt = m.group(1)
            out[current_prompt] = {"K_star": None, "basins": [],
                                   "V_min": None, "V_max": None}
            continue
        if current_prompt is None:
            continue
        m = re.match(r"^-\s+\$K\^\\ast\s*=\s*(\d+)", line)
        if m:
            out[current_prompt]["K_star"] = int(m.group(1))
            continue
        m = re.match(r"^\|\s*A(\d+)\s*\|\s*(\d+)\s*\|\s*([+-]?\d+\.\d+)"
                     r"\s*\|\s*(.+?)\s*\|$", line)
        if m:
            basin = {
                "id": int(m.group(1)),
                "size": int(m.group(2)),
                "V_mean": float(m.group(3)),
                "top_tokens_raw": m.group(4),
            }
            first_tok_match = re.match(r"`([^`]+)`.*?(\d+\.\d+)",
                                       m.group(4))
            if first_tok_match:
                basin["top_token"] = first_tok_match.group(1)
                basin["top_prob"] = float(first_tok_match.group(2))
            out[current_prompt]["basins"].append(basin)
    for pname, info in out.items():
        vs = [b["V_mean"] for b in info["basins"]]
        if vs:
            info["V_min"] = min(vs)
            info["V_max"] = max(vs)
    return out


def pick_content_ratio(basins: List[Dict]) -> float:
    """Fraction of basins whose largest-prob token is content (not punctuation).

    A crude proxy for 'meaningful attractors': any token not in the
    small punctuation/whitespace set.
    """
    PUNCT = {",", ".", ";", ":", "?", "!", "-", "--", "'",
             "\\n", "``", "''"}
    if not basins:
        return float("nan")
    n_content = 0
    for b in basins:
        tok = b.get("top_token", "").strip()
        if tok and tok not in PUNCT:
            n_content += 1
    return n_content / max(len(basins), 1)


def main():
    rows = []
    for tag, log_path, attr_path, pretty in VARIANTS:
        train = parse_training_log(log_path)
        attr = parse_attractor_summary(attr_path)

        ks = [attr[p]["K_star"] for p in PROMPTS if p in attr]
        k_tuple = tuple(ks) if ks else None
        n_basins = [len(attr[p]["basins"]) for p in PROMPTS if p in attr]
        vmin = min((attr[p]["V_min"] for p in PROMPTS
                    if p in attr and attr[p]["V_min"] is not None),
                   default=None)
        vmax = max((attr[p]["V_max"] for p in PROMPTS
                    if p in attr and attr[p]["V_max"] is not None),
                   default=None)
        content_ratios = [
            pick_content_ratio(attr[p]["basins"])
            for p in PROMPTS if p in attr
        ]
        mean_content = (sum(c for c in content_ratios
                            if not math.isnan(c))
                        / max(len([c for c in content_ratios
                                   if not math.isnan(c)]), 1)
                        ) if content_ratios else float("nan")

        row = {
            "tag": tag,
            "pretty": pretty,
            "val_ppl": train.get("val_ppl"),
            "val_loss": train.get("val_loss"),
            "train_loss": train.get("train_loss"),
            "K_star_per_prompt": k_tuple,
            "V_range_global": (vmin, vmax),
            "mean_content_basin_frac": mean_content,
            "attr_found": bool(attr),
        }
        rows.append(row)

    out_md = EM_RESULTS / "comparison_report.md"
    out_json = EM_RESULTS / "comparison_table.json"

    with out_md.open("w") as f:
        f.write("# Energetic-minima alternatives to free $V_\\theta$: "
                "cross-variant comparison\n\n")
        f.write("All four variants share the SARF-faithful $\\xi$ "
                "re-pooling, logfreq per-token mass, Tiny Shakespeare "
                "4000 steps, $d=128$, $L=8$, $\\Delta t=1$, seed 0.\n\n")
        f.write("| variant | val ppl | K* per prompt (n,m,s,d,c) | "
                "V range | mean content-basin fraction |\n")
        f.write("|---|---:|---|---|---:|\n")
        for r in rows:
            ppl = (f"{r['val_ppl']:.2f}" if r['val_ppl']
                   is not None else "—")
            k = (",".join(str(k) for k in r['K_star_per_prompt'])
                 if r['K_star_per_prompt'] else "—")
            vr = (f"[{r['V_range_global'][0]:+.1f}, "
                  f"{r['V_range_global'][1]:+.1f}]"
                  if r['V_range_global'][0] is not None else "—")
            c = (f"{r['mean_content_basin_frac']:.2f}"
                 if not math.isnan(r['mean_content_basin_frac'])
                 else "—")
            f.write(f"| {r['pretty']} | {ppl} | {k} | {vr} | {c} |\n")
        f.write("\n"
                "Columns:\n"
                "- **val ppl**: final validation perplexity from the last "
                "  training eval step.\n"
                "- **K*** per prompt: silhouette-optimal K, K∈[2,10], of "
                "  K-means on damped-flow endpoints.  Order = "
                "  (narrative, mathematics, scientific, dialogue, code).\n"
                "- **V range**: global [min, max] of $\\langle V\\rangle$ "
                "  across all basins across all five prompts (summary of "
                "  how wide the learned energy surface is).\n"
                "- **mean content-basin fraction**: over the five prompts, "
                "  average fraction of basins whose largest-probability "
                "  decoded token is not a punctuation symbol.\n")

    with out_json.open("w") as f:
        json.dump(rows, f, indent=2, default=str)

    print(f"[compare] wrote {out_md}")
    print(f"[compare] wrote {out_json}")

    print()
    print("=== Summary ===")
    for r in rows:
        ppl = f"{r['val_ppl']:.2f}" if r['val_ppl'] is not None else "—"
        print(f"  {r['tag']:10s}  ppl={ppl:>8s}  "
              f"K*={r['K_star_per_prompt']}  "
              f"V∈{r['V_range_global']}  "
              f"content={r['mean_content_basin_frac']:.2f}"
              if not math.isnan(r['mean_content_basin_frac'])
              else f"  {r['tag']:10s}  ppl={ppl:>8s}  (no attractors)")


if __name__ == "__main__":
    main()
