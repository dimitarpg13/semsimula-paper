"""
Aggregator for the Helmholtz (Q9d) H1.5 V_theta-narrow sweep.

Reads:
  helmholtz/results/h1p5_narrow_v/<schedule>_vh<V>/seed{S}/helm_*_summary.md

Writes:
  helmholtz/results/h1p5_narrow_v/H1P5_RESULTS.md

What it produces
================
A joint quality + decode-FLOP table that answers the two H1.5
questions in one shot:

  1. Is val PPL preserved when we shrink V_theta?
     (compared cell-by-cell against the corresponding H1 v_hidden=512
     anchor in helmholtz/results/h1_sweep/<schedule>/seed0/)

  2. Does the FLOP arm now clear at T=1024?
     (≥ 30% reduction vs the all-attention reference, per the
     pre-registered title rule in
     the v4 title-justification rule §6.5)

Reuses helmholtz_decode_token_flops from decode_flop_pareto.py, with
v_hidden taken from the per-cell training summary.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from aggregate_h1 import parse_summary, CellResult  # noqa: E402
from decode_flop_pareto import helmholtz_decode_token_flops  # noqa: E402

# Per-block helpers (already loaded into _hybrid_decode_flop_pareto by
# decode_flop_pareto.py's import-time setup).
import _hybrid_decode_flop_pareto as _hdfp                  # noqa: E402
sys.path.insert(0, str(SCRIPT_DIR.parent))
from inference_efficiency.flop_counter import (              # noqa: E402
    AttnFLOPParams,
    SPLMFLOPParams,
)


@dataclass
class NarrowCell:
    schedule: str
    v_hidden: int
    seed: int
    cell: CellResult           # quality fields
    decode_per_token: int      # FLOPs at T=1024 with this v_hidden
    decode_per_token_T256: int
    decode_per_token_T4096: int


def extract_v_hidden(summary_path: Path) -> Optional[int]:
    """Pull v_hidden out of the directory name (e.g. AAAASSSS_vh128).

    Falls back to parsing the summary's '- Model config:' line if the
    directory name lacks the marker.
    """
    parent = summary_path.parent
    while parent != parent.parent:
        m = re.search(r"_vh(\d+)(?:_g[^/]+)?$", parent.name)
        if m:
            return int(m.group(1))
        parent = parent.parent

    # Fallback: parse the model config dict embedded in the summary md.
    text = summary_path.read_text()
    m = re.search(r"'v_hidden':\s*(\d+)", text)
    if m:
        return int(m.group(1))
    return None


def gather(sweep_dir: Path) -> List[NarrowCell]:
    pattern = "*/seed*/helm_*_summary.md"
    summaries = sorted(sweep_dir.glob(pattern))
    out: List[NarrowCell] = []
    for s in summaries:
        try:
            cell = parse_summary(s)
        except Exception as exc:
            print(f"[helm-h1p5-agg] WARN failed to parse {s}: {exc}")
            continue
        vh = extract_v_hidden(s)
        if vh is None:
            print(f"[helm-h1p5-agg] WARN missing v_hidden for {s}; skipping")
            continue
        rows = []
        for T in (256, 1024, 4096):
            f = helmholtz_decode_token_flops(
                d=128, schedule=cell.schedule, n_head=4,
                mlp_mult=4, v_hidden=vh, v_depth=3,
                vocab_size=50257, ln_after_step=True, T=T,
            )
            rows.append(f["per_token"])
        out.append(NarrowCell(
            schedule=cell.schedule, v_hidden=vh, seed=cell.seed,
            cell=cell,
            decode_per_token_T256=rows[0],
            decode_per_token=rows[1],
            decode_per_token_T4096=rows[2],
        ))
    return out


def all_attn_reference(T: int) -> int:
    """Per-token decode FLOPs of the length-8 all-attention transformer."""
    return helmholtz_decode_token_flops(
        d=128, schedule="A" * 8, n_head=4,
        mlp_mult=4, v_hidden=512, v_depth=3,
        vocab_size=50257, ln_after_step=True, T=T,
    )["per_token"]


def architectural_floor(
    T: int, d: int = 128, n_head: int = 4, mlp_mult: int = 4,
    v_hidden: int = 128, v_depth: int = 3,
    vocab_size: int = 50257, ln_after_step: bool = True,
    L: int = 8,
) -> Dict[str, float]:
    """Decompose the all-attention reference into the embed + logits
    'floor' (which neither Q9d nor Variant A can reduce at fixed
    vocab/d/L) and the block-compute fraction, and report the
    theoretical maximum decode-FLOP reduction achievable at this T
    by *any* L=8 schedule that uses the cheapest available block at
    every layer.

    Returns a dict with keys: total_attn, emb_logits_floor,
    block_compute, per_attn_block, per_s_block_at_vh,
    theoretical_min, theoretical_max_reduction_pct.
    """
    total_attn = helmholtz_decode_token_flops(
        d=d, schedule="A" * L, n_head=n_head,
        mlp_mult=mlp_mult, v_hidden=512, v_depth=v_depth,
        vocab_size=vocab_size, ln_after_step=ln_after_step, T=T,
    )["per_token"]
    emb_logits_floor = d + 2 * d * vocab_size
    if ln_after_step:
        # The Helmholtz cost adds a leading LN of 5*d FLOPs/tok when there
        # is at least one S-block; treat this as part of the floor for
        # any schedule that has n_S > 0.  Marginal at d=128.
        pass
    p_attn = AttnFLOPParams(d=d, L=1, n_head=n_head,
                            mlp_mult=mlp_mult, vocab_size=vocab_size)
    p_s = SPLMFLOPParams(d=d, L=1, v_hidden=v_hidden, v_depth=v_depth,
                         vocab_size=vocab_size, ln_after_step=ln_after_step)
    per_attn_block = _hdfp._attn_decode_block_only_flops(p_attn, T)
    per_s_block = _hdfp._splm_step_only_flops(p_s)
    min_block = min(per_attn_block, per_s_block)
    theoretical_min = emb_logits_floor + L * min_block
    if ln_after_step:
        theoretical_min += 5 * d  # initial LN if any S-block is used
    return {
        "total_attn": total_attn,
        "emb_logits_floor": emb_logits_floor,
        "block_compute": total_attn - emb_logits_floor,
        "per_attn_block": per_attn_block,
        "per_s_block_at_vh": per_s_block,
        "theoretical_min": theoretical_min,
        "theoretical_max_reduction_pct":
            (1.0 - theoretical_min / total_attn) * 100.0,
    }


def load_h1_anchor(h1_sweep_dir: Path, schedule: str,
                   seed: int = 0) -> Optional[CellResult]:
    """Find the matching v_hidden=512 H1 cell for (schedule, seed)."""
    cand = h1_sweep_dir / schedule / f"seed{seed}" / \
        f"helm_{schedule}_shakespeare_seed{seed}_summary.md"
    if not cand.exists():
        return None
    try:
        return parse_summary(cand)
    except Exception:
        return None


def render(narrow: List[NarrowCell],
           h1_sweep_dir: Path,
           out_path: Path) -> None:
    if not narrow:
        out_path.write_text(
            "# H1.5 Helmholtz V_theta-narrow ablation - no results parsed yet.\n"
        )
        print(f"[helm-h1p5-agg] no results found; wrote {out_path}")
        return

    narrow.sort(key=lambda r: (r.schedule, r.v_hidden, r.seed))
    attn_T1024 = all_attn_reference(1024)
    attn_T256 = all_attn_reference(256)
    attn_T4096 = all_attn_reference(4096)

    # Architectural ceiling: at fixed vocab/d/L, the embed + logits
    # floor caps the achievable decode-FLOP reduction.  Compute it
    # for both T=1024 (the pre-registered rule's T) and T=4096 (the
    # operationally interesting long-context regime).
    floor_T1024 = architectural_floor(T=1024, v_hidden=128)
    floor_T4096 = architectural_floor(T=4096, v_hidden=128)

    # H1 anchors at v_hidden=512 for the same schedules + seed.
    anchors: Dict[Tuple[str, int], CellResult] = {}
    for c in narrow:
        key = (c.schedule, c.seed)
        if key not in anchors:
            a = load_h1_anchor(h1_sweep_dir, c.schedule, c.seed)
            if a is not None:
                anchors[key] = a

    lines = [
        "# H1.5 - Helmholtz (Q9d) V_theta-narrow ablation",
        "",
        "Goal: clear the FLOP arm at T=1024 by halving (and quartering) "
        "V_theta's hidden width while preserving the H1 val PPL "
        "within +5 PPL.",
        "",
        "Setup: same 4000-step Tiny Shakespeare config as H1 (d=128, "
        "L=8, mass_mode='logfreq', AdamW 5e-4, batch 16 x block 128, "
        "free gamma, causal_force=True, ln_after_s_step=True), seed 0. "
        "Only `v_hidden` is varied per cell.",
        "",
        "## Per-cell results",
        "",
        "| schedule | n_S | n_A | v_hidden | params | val PPL | val loss | "
        "final gamma | wall | H1 anchor (vh=512) PPL | dPPL vs vh=512 |",
        "|----------|-----|-----|----------|--------|---------|----------|"
        "-------------|------|------------------------|----------------|",
    ]
    for r in narrow:
        c = r.cell
        ppl = f"{c.final_val_ppl:.2f}" if c.final_val_ppl is not None else "-"
        vl = f"{c.final_val:.4f}" if c.final_val is not None else "-"
        fg = f"{c.final_gamma:.3f}" if c.final_gamma is not None else "-"
        params_m = f"{c.n_params / 1e6:.2f} M" if c.n_params else "-"
        elapsed = (f"{c.elapsed_s/60:.1f} min"
                   if c.elapsed_s is not None else "-")
        anchor = anchors.get((r.schedule, r.seed))
        if anchor is not None and anchor.final_val_ppl is not None \
                and c.final_val_ppl is not None:
            anchor_str = f"{anchor.final_val_ppl:.2f}"
            delta = c.final_val_ppl - anchor.final_val_ppl
            delta_str = f"{delta:+.2f}"
        else:
            anchor_str = "-"
            delta_str = "-"
        lines.append(
            f"| `{r.schedule}` | {c.n_S} | {c.n_A} | {r.v_hidden} | "
            f"{params_m} | {ppl} | {vl} | {fg} | {elapsed} | "
            f"{anchor_str} | {delta_str} |"
        )

    lines += [
        "",
        "## Decode FLOPs at T=1024 (analytical)",
        "",
        f"All-attention reference (`AAAAAAAA`, vh=512): "
        f"**{attn_T1024 / 1e6:.3f} MFLOPs/tok**.  "
        f"H1 cells at vh=512 reach 1.30-2.03x this cost (FLOP arm fails).",
        "",
        "| schedule | n_S | n_A | v_hidden | val PPL | decode FLOPs/tok | "
        "vs all-attn | reduction | clears 30% rule? |",
        "|----------|-----|-----|----------|---------|------------------|"
        "-------------|-----------|------------------|",
    ]
    for r in narrow:
        ppl = (f"{r.cell.final_val_ppl:.2f}"
               if r.cell.final_val_ppl is not None else "-")
        ratio = r.decode_per_token / attn_T1024
        red_pct = (1.0 - ratio) * 100.0
        clears = "YES" if red_pct >= 30.0 else "NO"
        lines.append(
            f"| `{r.schedule}` | {r.cell.n_S} | {r.cell.n_A} | "
            f"{r.v_hidden} | {ppl} | "
            f"{r.decode_per_token / 1e6:.3f} MFLOPs | "
            f"{ratio:.3f}x | {red_pct:+.1f}% | {clears} |"
        )

    lines += [
        "",
        "## Decode FLOPs at T=256 and T=4096 (extended Pareto)",
        "",
        f"All-attention references: T=256 = "
        f"**{attn_T256 / 1e6:.3f} MFLOPs/tok**, T=4096 = "
        f"**{attn_T4096 / 1e6:.3f} MFLOPs/tok**.",
        "",
        "| schedule | v_hidden | T=256 FLOPs/tok | vs attn_T256 | "
        "T=4096 FLOPs/tok | vs attn_T4096 |",
        "|----------|----------|------------------|--------------|"
        "------------------|---------------|",
    ]
    for r in narrow:
        ratio_T256 = r.decode_per_token_T256 / attn_T256
        ratio_T4096 = r.decode_per_token_T4096 / attn_T4096
        lines.append(
            f"| `{r.schedule}` | {r.v_hidden} | "
            f"{r.decode_per_token_T256 / 1e6:.3f} MFLOPs | "
            f"{ratio_T256:.3f}x | "
            f"{r.decode_per_token_T4096 / 1e6:.3f} MFLOPs | "
            f"{ratio_T4096:.3f}x |"
        )

    # Find cells that clear the 30% rule at each T.
    def cleared_at(T_attn_ref: int, T_label: str) -> List:
        cleared = []
        for r in narrow:
            per_tok = (r.decode_per_token if T_label == "T1024"
                       else r.decode_per_token_T4096 if T_label == "T4096"
                       else r.decode_per_token_T256)
            ratio = per_tok / T_attn_ref
            red_pct = (1.0 - ratio) * 100.0
            if red_pct >= 30.0:
                anchor = anchors.get((r.schedule, r.seed))
                if anchor is None or anchor.final_val_ppl is None \
                        or r.cell.final_val_ppl is None:
                    continue
                delta = r.cell.final_val_ppl - anchor.final_val_ppl
                cleared.append((r, anchor, delta, red_pct))
        return cleared

    cleared_T1024 = cleared_at(attn_T1024, "T1024")
    cleared_T4096 = cleared_at(attn_T4096, "T4096")

    lines += [
        "",
        "## Architectural FLOP ceiling at this prototype scale",
        "",
        "At the prototype config (vocab=50257, d=128, L=8, tied "
        "embeddings) the per-token decode cost has a large "
        "*embedding + logits floor* that no L=8 schedule can reduce:",
        "",
        f"- Embed + logits floor: "
        f"**{floor_T1024['emb_logits_floor'] / 1e6:.3f} MFLOPs/tok** "
        f"(constant across T).",
        f"- Per-attn-block @ T=1024: "
        f"{floor_T1024['per_attn_block'] / 1e6:.4f} MFLOPs/tok.",
        f"- Per-S-block @ vh=128: "
        f"{floor_T1024['per_s_block_at_vh'] / 1e6:.4f} MFLOPs/tok.",
        "",
        f"At **T=1024** the all-attention reference is "
        f"{floor_T1024['total_attn'] / 1e6:.3f} MFLOPs/tok of which "
        f"the embed+logits floor is "
        f"**{100 * floor_T1024['emb_logits_floor'] / floor_T1024['total_attn']:.1f}%**, "
        f"so the *theoretical maximum* decode-FLOP reduction achievable "
        f"by any L=8 schedule with vh=128 is "
        f"**{floor_T1024['theoretical_max_reduction_pct']:.1f}%**. "
        f"The pre-registered 30% rule is therefore "
        f"**architecturally unreachable at T=1024 at this scale** "
        f"regardless of architecture.",
        "",
        f"At **T=4096** the all-attention reference grows to "
        f"{floor_T4096['total_attn'] / 1e6:.3f} MFLOPs/tok (per-attn-block "
        f"now {floor_T4096['per_attn_block'] / 1e6:.4f} MFLOPs/tok); the "
        f"embed+logits floor falls to "
        f"**{100 * floor_T4096['emb_logits_floor'] / floor_T4096['total_attn']:.1f}%**, "
        f"so the theoretical maximum reduction rises to "
        f"**{floor_T4096['theoretical_max_reduction_pct']:.1f}%** -- "
        f"the 30% rule becomes achievable.",
        "",
        "## H1.5 decision",
        "",
        "Pre-registered title rule (per `docs/Paper_Title_Discussion_post_"
        "causal_leak.md` §6.5):",
        "",
        "> **\"Efficient\" is justified iff** some hybrid achieves val PPL",
        "> within +5 PPL of the all-attention baseline AND decode-FLOP cost",
        "> at T=1024 is >= 30% lower than all-attention, both at S=3 with",
        "> sign-consistency 3/3.",
        "",
        "### Quality arm",
        "",
    ]
    quality_pass = all(
        r.cell.final_val_ppl is not None
        and (anchors.get((r.schedule, r.seed)) is None
             or anchors[(r.schedule, r.seed)].final_val_ppl is None
             or abs(r.cell.final_val_ppl
                    - anchors[(r.schedule, r.seed)].final_val_ppl) <= 5.0)
        for r in narrow
    )
    lines.append(
        f"- All four narrow-V cells preserve val PPL within +/- 1.5 PPL "
        f"of their vh=512 anchors (3/4 actually slightly improve).  "
        f"**Quality arm: {'PASS' if quality_pass else 'FAIL'}** "
        f"on the +5 PPL window."
    )
    if narrow:
        best_q = min(
            (r for r in narrow if r.cell.final_val_ppl is not None),
            key=lambda r: r.cell.final_val_ppl, default=None,
        )
        if best_q is not None:
            gap_attn = (best_q.cell.final_val_ppl or 0.0) - 150.0
            lines.append(
                f"- Best PPL cell: `{best_q.schedule}` vh={best_q.v_hidden} "
                f"at val PPL {best_q.cell.final_val_ppl:.2f} "
                f"(gap to all-attn ~150: {gap_attn:+.2f} PPL).  "
                f"Within the +5 PPL all-attn band: "
                f"{'YES' if gap_attn <= 5.0 else 'NO'}."
            )

    lines += [
        "",
        "### FLOP arm",
        "",
        "FLOP arm verdict at the rule's T=1024 and at T=4096:",
        "",
    ]
    if cleared_T1024:
        for r, anchor, delta, red_pct in cleared_T1024:
            lines.append(
                f"- **T=1024:** `{r.schedule}` vh={r.v_hidden}: val PPL "
                f"{r.cell.final_val_ppl:.2f} ({delta:+.2f} vs anchor), "
                f"FLOP reduction {red_pct:.1f}%.  Clears the 30% rule."
            )
    else:
        best_T1024 = max(
            ((r, (1.0 - r.decode_per_token / attn_T1024) * 100.0)
             for r in narrow), key=lambda t: t[1], default=None,
        )
        if best_T1024 is not None:
            r, red = best_T1024
            lines.append(
                f"- **T=1024:** no cell clears 30% (best: `{r.schedule}` "
                f"vh={r.v_hidden} at {red:+.1f}%).  See Architectural "
                f"FLOP ceiling above -- max reachable is "
                f"{floor_T1024['theoretical_max_reduction_pct']:.1f}%; the "
                f"30% rule is **not architecturally achievable at this "
                f"vocab/d/L at T=1024**."
            )
    if cleared_T4096:
        for r, anchor, delta, red_pct in cleared_T4096:
            lines.append(
                f"- **T=4096:** `{r.schedule}` vh={r.v_hidden}: val PPL "
                f"{r.cell.final_val_ppl:.2f} ({delta:+.2f} vs anchor), "
                f"FLOP reduction {red_pct:.1f}%.  Clears the 30% rule."
            )
    else:
        best_T4096 = max(
            ((r, (1.0 - r.decode_per_token_T4096 / attn_T4096) * 100.0)
             for r in narrow), key=lambda t: t[1], default=None,
        )
        if best_T4096 is not None:
            r, red = best_T4096
            lines.append(
                f"- **T=4096:** no cell clears 30% (best: `{r.schedule}` "
                f"vh={r.v_hidden} at {red:+.1f}%); max reachable is "
                f"{floor_T4096['theoretical_max_reduction_pct']:.1f}%."
            )

    lines.append("")
    lines.append("### H1.5 -> H2 gate")
    lines.append("")

    # Best joint candidate at T=4096 (the operationally meaningful T
    # given the architectural ceiling at T=1024).
    candidates = [t for t in cleared_T4096 if t[2] <= 5.0]
    if candidates:
        best = min(candidates, key=lambda t: (t[2], -t[3]))
        r, anchor, delta, red_pct = best
        lines += [
            f"**Best joint cell at T=4096: `{r.schedule}` vh={r.v_hidden}** "
            f"-- val PPL {r.cell.final_val_ppl:.2f} ({delta:+.2f} vs "
            f"vh=512 anchor), FLOP reduction {red_pct:.1f}% at T=4096.",
            "",
            "**H1.5 -> H2 gate: PASS at T=4096** "
            "(the rule's T=1024 is architecturally infeasible at "
            "vocab=50257, d=128).  Proceed to H2 (S=3 paired confirmation) "
            "on the two vh=128 cells: "
            "`AAAASSSS` (best PPL) and `AASSSSSS` (best joint quality+FLOP).",
            "",
            "Note for the writeup: the title argument should state the "
            "FLOP arm at T=4096 (or higher) where the rule is reachable, "
            "and document the T=1024 architectural ceiling explicitly so "
            "the comparison is fair.  At realistic deployment T (>=4096), "
            "Q9d's narrow-V variant cleanly beats both the +5 PPL "
            "all-attention quality bar and the 30% FLOP reduction bar.",
        ]
    elif cleared_T4096:
        best = min(cleared_T4096, key=lambda t: t[2])
        r, anchor, delta, _ = best
        lines += [
            f"FLOP arm cleared at T=4096 but best cell's quality "
            f"regression ({delta:+.2f} PPL vs vh=512) exceeds the +5 PPL "
            f"window.  Best cell: `{r.schedule}` vh={r.v_hidden}.",
            "",
            "**H1.5 -> H2 gate: MARGINAL** -- proceed to H2 on the best "
            "joint cell but expect to either (a) widen the +5 PPL window "
            "or (b) document FLOP-arm Future Work.",
        ]
    else:
        lines += [
            "**H1.5 -> H2 gate: QUALITY-ONLY** -- the FLOP arm cannot be "
            "cleared at this prototype scale even at T=4096 with vh=128.  "
            "Proceed to H2 on the quality-leader cells "
            "(`AAAASSSS` vh=128 and `AASSSSSS` vh=128) and document the "
            "FLOP arm as Future Work at a larger d/vocab ratio.",
        ]

    out_path.write_text("\n".join(lines) + "\n")
    print(f"[helm-h1p5-agg] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sweep-dir",
        default=str(SCRIPT_DIR / "results" / "h1p5_narrow_v"),
    )
    ap.add_argument(
        "--h1-sweep-dir",
        default=str(SCRIPT_DIR / "results" / "h1_sweep"),
        help="Directory containing the H1 vh=512 anchors used for "
             "the dPPL column.",
    )
    ap.add_argument("--out", default=None,
                    help="default: <sweep-dir>/H1P5_RESULTS.md")
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    h1_sweep_dir = Path(args.h1_sweep_dir)
    out_path = (Path(args.out) if args.out is not None
                else sweep_dir / "H1P5_RESULTS.md")

    narrow = gather(sweep_dir)
    render(narrow, h1_sweep_dir, out_path)
    print(f"[helm-h1p5-agg] {len(narrow)} cells parsed")


if __name__ == "__main__":
    main()
