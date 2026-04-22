"""
Four-way comparison:
  - fixed-xi SPLM (baseline SPLM)                 ../results/splm_shakespeare_*
  - SARF-faithful SPLM                            ../sarf_variant/results/splm_sarf_shakespeare_*
  - SARF + embed_head mass (variant A)            results/splm_sarfmass_embed_head_shakespeare_*
  - SARF + logfreq   mass (variant B)             results/splm_sarfmass_logfreq_shakespeare_*

Inputs (per variant):
  training summary md
  training jsonl log
  checkpoint (for final mass stats; optional)
  sharedV_*_results.npz  (depth-axis shared-potential fit)
  tokdir_*_results.npz    (token-axis shared-potential fit)

Outputs:
  comparison_report.md
  comparison_loss_curve.png
  comparison_sharedV_r2.png
  comparison_tokdir_r2.png
  comparison_mass_hist.png         (histogram / bar plot of learned mass)
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR     = Path(__file__).parent
PARENT_DIR     = SCRIPT_DIR.parent
BASE_RESULTS   = PARENT_DIR / "results"
SARF_RESULTS   = PARENT_DIR / "sarf_variant" / "results"
MASS_RESULTS   = SCRIPT_DIR / "results"


@dataclass
class Variant:
    tag: str
    label: str
    color: str
    marker: str
    summary_md: Path
    training_log: Path
    ckpt: Path
    sharedV_npz: Path
    tokdir_npz: Path


VARIANTS: List[Variant] = [
    Variant(
        tag="fixed_xi",
        label="fixed-$\\xi$ SPLM",
        color="tab:blue",
        marker="o",
        summary_md=BASE_RESULTS / "splm_shakespeare_summary.md",
        training_log=BASE_RESULTS / "splm_shakespeare_training_log.jsonl",
        ckpt=BASE_RESULTS / "splm_shakespeare_ckpt_latest.pt",
        sharedV_npz=BASE_RESULTS / "sharedV_shakespeare_ckpt_latest_results.npz",
        tokdir_npz=BASE_RESULTS / "tokdir_splm_shakespeare_results.npz",
    ),
    Variant(
        tag="sarf",
        label="SARF-faithful SPLM",
        color="tab:red",
        marker="s",
        summary_md=SARF_RESULTS / "splm_sarf_shakespeare_summary.md",
        training_log=SARF_RESULTS / "splm_sarf_shakespeare_training_log.jsonl",
        ckpt=SARF_RESULTS / "splm_sarf_shakespeare_ckpt_latest.pt",
        sharedV_npz=BASE_RESULTS / "sharedV_sarf_shakespeare_ckpt_latest_results.npz",
        tokdir_npz=BASE_RESULTS / "tokdir_sarf_shakespeare_results.npz",
    ),
    Variant(
        tag="sarf_mass_embed",
        label="SARF + embed-head mass (A)",
        color="tab:green",
        marker="^",
        summary_md=MASS_RESULTS / "splm_sarfmass_embed_head_shakespeare_summary.md",
        training_log=MASS_RESULTS / "splm_sarfmass_embed_head_shakespeare_training_log.jsonl",
        ckpt=MASS_RESULTS / "splm_sarfmass_embed_head_shakespeare_ckpt_latest.pt",
        sharedV_npz=BASE_RESULTS / "sharedV_sarfmass_embed_head_shakespeare_results.npz",
        tokdir_npz=BASE_RESULTS / "tokdir_sarfmass_embed_head_shakespeare_results.npz",
    ),
    Variant(
        tag="sarf_mass_logfreq",
        label="SARF + logfreq mass (B)",
        color="tab:purple",
        marker="v",
        summary_md=MASS_RESULTS / "splm_sarfmass_logfreq_shakespeare_summary.md",
        training_log=MASS_RESULTS / "splm_sarfmass_logfreq_shakespeare_training_log.jsonl",
        ckpt=MASS_RESULTS / "splm_sarfmass_logfreq_shakespeare_ckpt_latest.pt",
        sharedV_npz=BASE_RESULTS / "sharedV_sarfmass_logfreq_shakespeare_results.npz",
        tokdir_npz=BASE_RESULTS / "tokdir_sarfmass_logfreq_shakespeare_results.npz",
    ),
]


def load_training_summary(p: Path) -> Dict[str, str]:
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        m = re.match(r"- (.+?): (.+)", line)
        if m:
            out[m.group(1).strip().lower()] = m.group(2).strip(" `*")
    return out


def load_final_loss_from_log(p: Path) -> Tuple[Optional[float], Optional[float]]:
    if not p.exists():
        return None, None
    tr, va = None, None
    with p.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "val_loss" in r:
                va = float(r["val_loss"])
                tr = float(r.get("train_loss_eval", tr)) if "train_loss_eval" in r else tr
    return tr, va


def load_loss_curve(p: Path) -> Tuple[list, list, list]:
    steps, tr, va = [], [], []
    if not p.exists():
        return steps, tr, va
    with p.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "val_loss" in r and "train_loss_eval" in r:
                steps.append(int(r["step"]))
                tr.append(float(r["train_loss_eval"]))
                va.append(float(r["val_loss"]))
    return steps, tr, va


def load_final_mass_from_ckpt(p: Path) -> Optional[dict]:
    if not p.exists():
        return None
    try:
        import torch
        ck = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"[compare] could not load {p}: {e}")
        return None
    if "final_mass_stats" in ck:
        return ck["final_mass_stats"]
    if "final_m" in ck:
        return {"mean": float(ck["final_m"]), "std": 0.0,
                "min":  float(ck["final_m"]), "max": float(ck["final_m"])}
    return None


def load_npz(p: Path) -> Optional[dict]:
    if not p.exists():
        return None
    return dict(np.load(p))


def variant_layers(d: dict) -> Optional[np.ndarray]:
    return d.get("layers", d.get("layers_shv"))


def r2_overall(d: Optional[dict], split: str) -> Optional[float]:
    if d is None:
        return None
    k = f"r2_shv_{split}_overall"
    if k in d:
        return float(d[k][0])
    return None


def r2_per_layer(d: Optional[dict], split: str) -> Optional[np.ndarray]:
    if d is None:
        return None
    k = f"r2_shv_{split}"
    if k in d:
        return np.asarray(d[k], dtype=np.float64)
    return None


def median_r2(d: Optional[dict], split: str) -> Optional[float]:
    arr = r2_per_layer(d, split)
    if arr is None:
        return None
    return float(np.median(arr))


def plot_per_layer_all(variants: List[Variant],
                       results: Dict[str, dict],
                       key_npz: str, key_metric: str,
                       outpath: Path, title: str):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    plotted = 0
    for v in variants:
        d = results.get(f"{v.tag}_{key_npz}")
        if d is None:
            continue
        layers = variant_layers(d)
        y = r2_per_layer(d, key_metric)
        if layers is None or y is None:
            continue
        ax.plot(layers, y, marker=v.marker, label=v.label, color=v.color)
        plotted += 1
    if plotted == 0:
        print(f"[compare] skip plot {title}: no data")
        plt.close(fig)
        return
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("layer $\\ell$")
    ax.set_ylabel(f"{key_metric.upper()} $R^2$ (shared $V_\\psi$)")
    ax.set_ylim(-0.2, 1.05)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print(f"[compare] saved {outpath}")


def plot_loss_curves_all(variants: List[Variant], outpath: Path):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    plotted = 0
    for v in variants:
        steps, tr, va = load_loss_curve(v.training_log)
        if not steps:
            continue
        ax.plot(steps, va, marker=v.marker, label=f"{v.label} val",
                color=v.color)
        ax.plot(steps, tr, marker=v.marker, linestyle="--",
                color=v.color, alpha=0.4)
        plotted += 1
    if plotted == 0:
        print(f"[compare] skip loss-curve plot: no logs")
        plt.close(fig)
        return
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy")
    ax.set_title("Tiny Shakespeare -- training val (solid) / train-eval (dashed)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print(f"[compare] saved {outpath}")


def plot_mass_stats(variants: List[Variant], outpath: Path):
    """Bar chart of mean mass with error bars (= std) for each variant."""
    stats = []
    for v in variants:
        s = load_final_mass_from_ckpt(v.ckpt)
        stats.append((v, s))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(stats))
    means = [(s["mean"] if s is not None else 0.0) for _, s in stats]
    stds = [(s["std"] if s is not None else 0.0) for _, s in stats]
    colors = [v.color for v, _ in stats]
    ax.bar(x, means, yerr=stds, color=colors, alpha=0.75, capsize=4)
    for i, (v, s) in enumerate(stats):
        if s is None:
            continue
        ax.text(i, means[i] + stds[i] + 0.02,
                f"mean={means[i]:.3f}\nstd={stds[i]:.3f}\n"
                f"min={s['min']:.3f}\nmax={s['max']:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([v.label for v, _ in stats], rotation=15, ha="right",
                       fontsize=9)
    ax.set_ylabel("final semantic mass $m_t$")
    ax.set_title("Learned mass at end of training (error bar = std over tokens)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print(f"[compare] saved {outpath}")


def write_lm_quality_table(f, variants, results):
    f.write("## 1. Language-modelling quality\n\n")
    f.write("| variant | params | wall-clock | final train CE | "
            "final val CE | val ppl | mass mean (std) |\n")
    f.write("|---|--:|--:|--:|--:|--:|--:|\n")
    for v in variants:
        summ = load_training_summary(v.summary_md)
        tr, va = load_final_loss_from_log(v.training_log)
        mass = load_final_mass_from_ckpt(v.ckpt)
        row = [v.label]
        row.append(summ.get("parameters", "--"))
        row.append(summ.get("wall-clock time", "--"))
        row.append(f"{tr:.4f}" if tr is not None else "--")
        row.append(f"{va:.4f}" if va is not None else "--")
        row.append(f"{math.exp(va):.2f}" if va is not None else "--")
        if mass is not None:
            row.append(f"{mass['mean']:.3f} ({mass['std']:.3f})")
        else:
            row.append("--")
        f.write("| " + " | ".join(row) + " |\n")
    f.write("\nFigure: `comparison_loss_curve.png`\n")
    f.write("Mass stats: `comparison_mass_hist.png`\n\n")


def write_shv_section(f, variants, results, npz_key: str, caption: str,
                      fig_name: str):
    f.write(f"## {caption}\n\n")
    f.write("| variant | pooled train | pooled test | median-per-layer test | "
            "min test | layers $\\ge 0.5$ |\n")
    f.write("|---|--:|--:|--:|--:|--:|\n")
    for v in variants:
        d = results.get(f"{v.tag}_{npz_key}")
        if d is None:
            f.write(f"| {v.label} | -- | -- | -- | -- | -- |\n")
            continue
        ptr = r2_overall(d, "train")
        pte = r2_overall(d, "test")
        med_te = median_r2(d, "test")
        arr_te = r2_per_layer(d, "test")
        min_te = float(np.min(arr_te)) if arr_te is not None else None
        n_good = int(np.sum(arr_te >= 0.5)) if arr_te is not None else 0
        n_tot = int(len(arr_te)) if arr_te is not None else 0
        f.write(f"| {v.label} "
                f"| {ptr:+.3f} "
                f"| {pte:+.3f} "
                f"| {med_te:+.3f} "
                f"| {min_te:+.3f} "
                f"| {n_good} / {n_tot} |\n")
    f.write(f"\nFigure: `{fig_name}`\n\n")


def write_per_layer_table(f, variants, results, npz_key: str, caption: str):
    f.write(f"### {caption} -- per-layer TEST $R^2$\n\n")
    ds = {v.tag: results.get(f"{v.tag}_{npz_key}") for v in variants}
    all_layers = set()
    for v in variants:
        d = ds[v.tag]
        if d is None:
            continue
        ly = variant_layers(d)
        if ly is not None:
            all_layers.update(int(x) for x in ly)
    if not all_layers:
        f.write("_(missing data)_\n\n")
        return
    layers = sorted(all_layers)
    header = "| layer "
    sep = "|--:"
    for v in variants:
        header += f"| {v.label} "
        sep += "|--:"
    f.write(header + "|\n")
    f.write(sep + "|\n")
    per_layer = {}
    for v in variants:
        d = ds[v.tag]
        if d is None:
            per_layer[v.tag] = {}
            continue
        ly = variant_layers(d)
        arr = r2_per_layer(d, "test")
        if ly is None or arr is None:
            per_layer[v.tag] = {}
        else:
            per_layer[v.tag] = {int(l): float(a) for l, a in zip(ly, arr)}
    for l in layers:
        row = f"| {l} "
        for v in variants:
            val = per_layer[v.tag].get(l)
            row += f"| {val:+.3f} " if val is not None else "| -- "
        row += "|\n"
        f.write(row)
    f.write("\n")


def main():
    results: Dict[str, Optional[dict]] = {}
    for v in VARIANTS:
        results[f"{v.tag}_shV"]    = load_npz(v.sharedV_npz)
        results[f"{v.tag}_tdir"]   = load_npz(v.tokdir_npz)
        print(f"[compare] {v.tag}: shV={'OK' if results[f'{v.tag}_shV'] is not None else 'MISSING'}  "
              f"tdir={'OK' if results[f'{v.tag}_tdir'] is not None else 'MISSING'}")

    plot_per_layer_all(
        VARIANTS, results, "shV", "test",
        SCRIPT_DIR / "comparison_sharedV_r2.png",
        "Shared-potential fit (DEPTH axis) -- per-layer TEST $R^2$")
    plot_per_layer_all(
        VARIANTS, results, "tdir", "test",
        SCRIPT_DIR / "comparison_tokdir_r2.png",
        "Shared-potential fit (TOKEN axis) -- per-layer TEST $R^2$")
    plot_loss_curves_all(
        VARIANTS, SCRIPT_DIR / "comparison_loss_curve.png")
    plot_mass_stats(
        VARIANTS, SCRIPT_DIR / "comparison_mass_hist.png")

    md_path = SCRIPT_DIR / "comparison_report.md"
    with md_path.open("w") as f:
        f.write("# SARF + per-token mass -- four-way comparison report\n\n")
        f.write("Auto-generated by `compare.py`.  Compares:\n\n")
        for v in VARIANTS:
            f.write(f"- **{v.label}** ({v.tag})\n")
        f.write(
            "\nAll four share identical $(d, L, d_V)$, data, batch, optimiser,\n"
            "seed and step budget on Tiny Shakespeare.  The only systematic\n"
            "differences are (i) whether $\\xi_t^{(\\ell)}$ is fixed at\n"
            "$\\ell=0$ (variant 1) or recomputed per layer (variants 2--4),\n"
            "and (ii) the shape of the per-token mass $\\semm_t$.\n\n"
        )

        write_lm_quality_table(f, VARIANTS, results)

        write_shv_section(
            f, VARIANTS, results, "shV",
            "2. Strict shared-potential fit (paper §14.2, DEPTH axis)",
            "comparison_sharedV_r2.png")
        write_per_layer_table(f, VARIANTS, results, "shV",
                              "Depth-axis shared-$V_\\psi$")

        write_shv_section(
            f, VARIANTS, results, "tdir",
            "3. Token-direction shared-potential fit (paper §14.5)",
            "comparison_tokdir_r2.png")
        write_per_layer_table(f, VARIANTS, results, "tdir",
                              "Token-axis shared-$V_\\psi$")

        f.write("## 4. Interpretation\n\n")
        f.write(
            "### 4.1 LM quality: the surprisal prior wins decisively\n\n"
            "At identical parameter count (within $d+1=129$), identical\n"
            "optimiser, schedule, data and seed, the four variants land at\n"
            "val perplexities `287 -> 192 -> 223 -> 161`.  SARF + logfreq\n"
            "mass (B) is the best configuration by a clear margin:\n"
            "~**44%** ppl reduction relative to fixed-$\\xi$ SPLM, and\n"
            "~**17%** ppl reduction relative to the SARF-faithful variant\n"
            "that already won the previous ablation round.  By contrast,\n"
            "SARF + embed-head mass (A) *loses* ~16% ppl to plain SARF.\n\n"
            "The asymmetry between (A) and (B) is the real finding.\n"
            "Both variants have essentially the same expressive capacity\n"
            "in principle (the embed-head linear projection can implement\n"
            "any per-token scalar function of the embedding, and should be\n"
            "strictly richer than a frozen surprisal lookup).  Yet at\n"
            "Tiny Shakespeare scale the theoretically-motivated prior\n"
            "beats the data-driven learner.  The most natural reading is\n"
            "an **inductive-bias-vs-data-efficiency** story: logfreq tells\n"
            "the optimiser \"rare tokens are heavier\" from step 0, fixing\n"
            "the *shape* of the mass so only a single scale $\\alpha$ has\n"
            "to be learned; embed-head must discover both the shape and\n"
            "the scale from 300 K tokens of Shakespeare.  The prior\n"
            "happens to be approximately right -- content/rare tokens\n"
            "*are* the ones you want to anchor -- so the prior does most\n"
            "of the work and the remaining tunable $\\alpha \\approx 0.1$\n"
            "suffices.\n\n"
            "This is the clearest quantitative evidence so far that the\n"
            "framework's information-theoretic reading of semantic mass\n"
            "($\\semm_t \\propto -\\log p(x_t)$) is not just allegorical.\n"
            "It is *prescriptive*: the observed 17% ppl gain over the\n"
            "single-scalar SARF baseline comes for a single extra scalar\n"
            "parameter and a frozen vocabulary-sized lookup table.\n\n"
            "### 4.2 Depth-axis shared-$V_\\psi$: logfreq also closes the\n"
            "gap that SARF had opened\n\n"
            "The SARF-faithful ablation had introduced a modest structural\n"
            "cost: pooled depth-direction TEST $R^2$ dropped from +0.79\n"
            "(fixed-$\\xi$) to +0.71 (SARF), with the framework reading\n"
            "that a pointwise $V_\\psi(h)$ cannot capture the sequence-\n"
            "level functional SARF implements.  Per-token mass changes\n"
            "this picture:\n\n"
            "- SARF + embed-head (A): pooled TEST $R^2$ = +0.72, essentially\n"
            "  the same as plain SARF, but the worst-layer $R^2$ climbs from\n"
            "  +0.48 to +0.57, i.e. the profile becomes more uniform.\n"
            "- SARF + logfreq (B): pooled TEST $R^2$ = **+0.84**, *better\n"
            "  than fixed-$\\xi$ SPLM* (+0.79).  No layer drops below +0.67;\n"
            "  median per-layer +0.87.\n\n"
            "Variant (B) is therefore doing two things at once: (i) lower\n"
            "validation cross-entropy, and (ii) higher pointwise shared-$V_\\psi$\n"
            "fidelity along the depth axis.  The combination is remarkable\n"
            "because the two metrics a priori trade off against each other\n"
            "-- LM quality usually pulls toward expressive, non-conservative\n"
            "dynamics, while shared-$V_\\psi$ fidelity pulls toward strict\n"
            "pointwise conservativity.  The logfreq mass finds a regime where\n"
            "both improve simultaneously.\n\n"
            "Framework reading: the surprisal-shaped mass gives different\n"
            "tokens different effective step sizes in the integrator\n"
            "($v_t = (v_t + dt f_t / m_t) / (1 + dt\\gamma)$), and this\n"
            "factorises cleanly into a per-token scale that the shared\n"
            "$V_\\psi(h)$ fit can absorb through its per-layer $\\alpha_\\ell, \\beta_\\ell$\n"
            "constants (the fit is still strictly pointwise in $h$).\n"
            "In other words, logfreq mass does not add sequence-level\n"
            "coupling (that is what SARF $\\xi$ re-pooling does); it just\n"
            "*tilts* the integrator toward heavier anchoring of rare tokens,\n"
            "which happens to regularise the dynamics back into the shared-\n"
            "scalar family.\n\n"
            "### 4.3 Token-direction: gradual softening, no collapse\n\n"
            "Along the token axis the picture is monotonic in the other\n"
            "direction: pooled TEST $R^2$ drops `0.52 -> 0.41 -> 0.36 -> 0.33`\n"
            "across fixed-$\\xi$ / SARF / SARF+A / SARF+B.  Adding SARF\n"
            "$\\xi$ re-pooling costs ~0.11, per-token mass costs an\n"
            "additional ~0.05--0.08.  No variant collapses, and no layer\n"
            "in any variant goes below +0.195.  All four stay firmly in\n"
            "SPLM-like territory, at least an order of magnitude above\n"
            "the GPT-2 middle-band near-zero values from §14.4.\n\n"
            "This is exactly the pattern we would expect theoretically:\n"
            "every step that introduces more cross-token coupling (xi\n"
            "re-pooling; per-token mass) makes the pointwise shared-$V_\\psi$\n"
            "fit harder along the token axis where the coupling lives.\n"
            "Closing these gaps is the job of a $\\xi$- and $m$-aware\n"
            "$V_\\psi(\\xi, h, m)$ fit (Q8 in the paper).\n\n"
            "### 4.4 What we learnt about semantic mass\n\n"
            "1. **Per-token mass has empirical content on top of SARF\n"
            "    $\\xi$ re-pooling.**  One of the two variants beats SARF on\n"
            "    LM quality, and *both* variants produce non-trivial per-\n"
            "    token dispersion (std 0.16--0.21 around means 0.84--1.28).\n"
            "2. **The framework's surprisal prior is better than a free\n"
            "    learned head at this scale.**  The gain from fixing the\n"
            "    shape of $\\semm_t$ to $-\\log\\hat p(x_t)$ is ~17% ppl over\n"
            "    SARF while adding a single scalar parameter.  The free\n"
            "    linear head underperforms, presumably because 4000 steps\n"
            "    on 300 K tokens is not enough to learn the right shape\n"
            "    from scratch.\n"
            "3. **Per-token mass improves the depth-axis shared-$V_\\psi$\n"
            "    separator** (logfreq reaches +0.84 > fixed-$\\xi$ +0.79),\n"
            "    showing that the conservative-by-construction narrative\n"
            "    *tightens* when we implement more of the framework, not\n"
            "    loosens.  This is the opposite of what we saw when moving\n"
            "    from fixed-$\\xi$ to SARF-$\\xi$ alone.\n"
            "4. **The concatenation of SARF $\\xi$ and logfreq mass is\n"
            "    where the framework lives.**  On Tiny Shakespeare,\n"
            "    implementing both prescriptions simultaneously gives the\n"
            "    best LM quality, the cleanest depth-axis conservativity,\n"
            "    and still only ~0.19 lower token-axis $R^2$ than the\n"
            "    fixed-$\\xi$ variant -- at ~44% lower perplexity.\n\n"
            "### 4.5 Open questions\n\n"
            "- **Does embed-head (A) eventually match logfreq (B)?**  Either\n"
            "  via longer training, larger corpus, or mass-head\n"
            "  regularisation toward the surprisal profile?  If yes, the\n"
            "  prior is a useful accelerator but not a structural choice;\n"
            "  if no, the prior encodes something the linear head cannot\n"
            "  learn from a single scalar of signal per token.\n"
            "- **Context-aware $V_\\psi(\\xi, h, m)$ fit.**  The pointwise\n"
            "  shared-$V_\\psi(h)$ used here cannot see $\\xi$ or $m$\n"
            "  directly.  A context-aware oracle should drive the token-axis\n"
            "  R^2 back toward fixed-$\\xi$ levels; quantifying the gap\n"
            "  closes the Q8 storyline.\n"
            "- **Scale-up.**  All the above is at 7.1 M params on 300 K\n"
            "  tokens.  The framework's claim is that the conservative-\n"
            "  by-construction circuit is *inference-efficient*, not that\n"
            "  it wins at every corpus size.  The next controlled step is\n"
            "  to repeat the four-way on TinyStories and WikiText.\n"
        )

    print(f"[compare] saved {md_path}")


if __name__ == "__main__":
    main()
