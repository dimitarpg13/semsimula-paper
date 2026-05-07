"""
Aggregator for the Helmholtz (Q9d) H2 paired-confirmation sweep.

Reads (3 seeds per schedule):
  helmholtz/results/h1p5_narrow_v/<schedule>_vh128/seed0/helm_*_summary.md  # reused
  helmholtz/results/h2_paired_confirmation/<schedule>_vh128/seed{1,2}/helm_*_summary.md

Reads baselines:
  notebooks/conservative_arch/multi_seed/results/E1_shakespeare/matched_baseline/seed_{0..4}/
    matched_shakespeare_summary.md                                          # 5-seed all-attn E1
  notebooks/conservative_arch/hybrid/results/h1_sweep/k4_m4/seed0/
    hybrid_k4_m4_shakespeare_seed0_summary.md                               # Variant A best, seed 0

Writes:
  helmholtz/results/h2_paired_confirmation/H2_RESULTS.md

Decision rule (per companion_notes/Helmholtz-HSPLM_Path_Forward_and_Experiments.md §8.3,
mirroring the v4 title-justification rule §6.5):

  PASS quality arm iff at the best Q9d schedule:
    mean Δ̄ = mean(Q9d_PPL - attn_PPL) ≥ -5 PPL vs all-attention
    AND sign-consistency 3/3 across paired seeds
    AND paired-t two-sided p < 0.05.

  Q9d-vs-Variant-A win iff at the best Q9d schedule:
    mean Δ̄ = mean(Q9d_PPL - VariantA_best_PPL) ≤ -5 PPL
    AND sign-consistency 3/3.

Note: Variant A H2 (3 seeds) is not yet on disk; we report the
single-point delta vs Variant A best (k=4, m=4) seed 0 = 133.01
and flag that the strict 3/3 sign-consistency arm against
Variant A awaits the matching Variant A H2 run.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scipy.stats import ttest_rel  # paired t-test

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from aggregate_h1 import parse_summary, CellResult  # noqa: E402

REPO_ROOT = SCRIPT_DIR.parent.parent.parent
ATTN_BASELINE_DIR = (
    REPO_ROOT
    / "notebooks/conservative_arch/multi_seed/results/E1_shakespeare/matched_baseline"
)
VARIANT_A_BEST_PATH = (
    REPO_ROOT
    / "notebooks/conservative_arch/hybrid/results/h1_sweep/k4_m4/seed0/"
      "hybrid_k4_m4_shakespeare_seed0_summary.md"
)
VARIANT_A_H2_DIR = (
    REPO_ROOT
    / "notebooks/conservative_arch/hybrid/results/h2_paired_confirmation/k4_m4"
)


# -----------------------------------------------------------------------
# Cell loading
# -----------------------------------------------------------------------

@dataclass
class H2Cell:
    schedule: str
    v_hidden: int
    seed: int
    cell: CellResult


def _extract_v_hidden(summary_path: Path) -> Optional[int]:
    """Pull v_hidden out of the directory name (e.g. AAAASSSS_vh128)."""
    parent = summary_path.parent
    while parent != parent.parent:
        m = re.search(r"_vh(\d+)(?:_g[^/]+)?$", parent.name)
        if m:
            return int(m.group(1))
        parent = parent.parent
    return None


def gather_q9d_cells(
    h2_dir: Path, h1p5_dir: Path,
    schedules: Tuple[str, ...] = ("AAAASSSS", "AASSSSSS"),
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4),
) -> List[H2Cell]:
    """Gather Q9d cells across (schedule, seed), pulling seed 0 from H1.5
    and seeds >= 1 from H2.

    Default seeds = (0, 1, 2, 3, 4) supports both the n=3 H2 paired
    confirmation and the n=5 power-up.  Cells whose summary.md does
    not yet exist on disk are silently skipped.
    """
    out: List[H2Cell] = []
    for sched in schedules:
        for seed in seeds:
            base = h1p5_dir if seed == 0 else h2_dir
            cell_dir = base / f"{sched}_vh128" / f"seed{seed}"
            cand = cell_dir / (
                f"helm_{sched}_vh128_shakespeare_seed{seed}_summary.md"
            )
            if not cand.exists():
                continue
            try:
                cell = parse_summary(cand)
            except Exception as exc:
                print(f"[helm-h2-agg] WARN failed to parse {cand}: {exc}")
                continue
            vh = _extract_v_hidden(cand) or 128
            out.append(H2Cell(schedule=sched, v_hidden=vh,
                              seed=seed, cell=cell))
    return out


def load_attn_baseline_5seed(base_dir: Path) -> Dict[int, float]:
    """Returns {seed: val_ppl} for the all-attention 5-seed E1 baseline."""
    ppl_by_seed: Dict[int, float] = {}
    for seed in range(5):
        cand = base_dir / f"seed_{seed}" / "matched_shakespeare_summary.md"
        if not cand.exists():
            continue
        text = cand.read_text()
        m = re.search(r"Final val loss:.*\(ppl\s*([0-9.]+)\)", text)
        if m:
            ppl_by_seed[seed] = float(m.group(1))
    return ppl_by_seed


def load_variant_a_best(path: Path) -> Optional[float]:
    """Returns Variant A best (k=4, m=4) seed 0 val PPL = 133.01."""
    if not path.exists():
        return None
    text = path.read_text()
    m = re.search(r"Final val loss:.*\(ppl\s*([0-9.]+)\)", text)
    return float(m.group(1)) if m else None


def load_variant_a_3seed(seed0_path: Path,
                         h2_dir: Path,
                         seeds: Tuple[int, ...] = (1, 2, 3, 4)) -> Dict[int, float]:
    """Pull Variant A k=4, m=4 val PPL for seeds {0, …}.

    seed 0 lives under hybrid/results/h1_sweep/k4_m4/seed0/.
    seeds >= 1 live under hybrid/results/h2_paired_confirmation/k4_m4/seed{N}/.
    Default `seeds = (1, 2, 3, 4)` supports both the n=3 H2 paired
    confirmation and the n=5 power-up; missing seeds are silently
    skipped.  Function name retained for backward compatibility.
    """
    out: Dict[int, float] = {}

    if seed0_path.exists():
        text = seed0_path.read_text()
        m = re.search(r"Final val loss:.*\(ppl\s*([0-9.]+)\)", text)
        if m:
            out[0] = float(m.group(1))

    if h2_dir.exists():
        for seed in seeds:
            cand = (
                h2_dir / f"seed{seed}"
                       / f"hybrid_k4_m4_shakespeare_seed{seed}_summary.md"
            )
            if not cand.exists():
                continue
            text = cand.read_text()
            m = re.search(r"Final val loss:.*\(ppl\s*([0-9.]+)\)", text)
            if m:
                out[seed] = float(m.group(1))

    return out


# -----------------------------------------------------------------------
# Paired statistics
# -----------------------------------------------------------------------

@dataclass
class PairedStat:
    n: int
    mean_delta: float
    std_delta: float
    sign_consistency: str       # e.g. "3/3"
    t_stat: float
    p_value: float


def paired_t(q9d_by_seed: Dict[int, float],
             attn_by_seed: Dict[int, float]) -> Optional[PairedStat]:
    """Pair Q9d seed s with attention seed s; compute paired-t."""
    common = sorted(set(q9d_by_seed) & set(attn_by_seed))
    if len(common) < 2:
        return None
    a = [q9d_by_seed[s] for s in common]
    b = [attn_by_seed[s] for s in common]
    deltas = [x - y for x, y in zip(a, b)]
    n = len(deltas)
    mean_d = sum(deltas) / n
    var_d = sum((d - mean_d) ** 2 for d in deltas) / max(n - 1, 1)
    std_d = var_d ** 0.5
    res = ttest_rel(a, b)
    n_neg = sum(1 for d in deltas if d < 0)
    return PairedStat(
        n=n,
        mean_delta=mean_d,
        std_delta=std_d,
        sign_consistency=f"{n_neg}/{n}",
        t_stat=float(res.statistic),
        p_value=float(res.pvalue),
    )


# -----------------------------------------------------------------------
# Render
# -----------------------------------------------------------------------

def _ppl_table_per_seed(cells: List[H2Cell],
                        attn_by_seed: Dict[int, float],
                        va_best: Optional[float],
                        va_by_seed: Optional[Dict[int, float]] = None
                        ) -> List[str]:
    """Per-cell table.  When VA multi-seed is available, show Δ against
    paired VA seed in addition to Δ vs VA best (single seed)."""
    has_va_paired = va_by_seed is not None and len(va_by_seed) >= 2
    n_q9d = len({c.seed for c in cells})
    n_sched = len({c.schedule for c in cells})
    header_label = (
        f"## Per-cell results ({n_q9d} seeds × {n_sched} schedules, vh=128)"
    )
    if has_va_paired:
        lines = [
            header_label,
            "",
            "| schedule | v_hidden | seed | val PPL | val loss | "
            "Δ vs all-attn(seed) | Δ vs VA(seed) | Δ vs VA best |",
            "|----------|----------|------|---------|----------|"
            "---------------------|---------------|--------------|",
        ]
    else:
        lines = [
            header_label,
            "",
            "| schedule | v_hidden | seed | val PPL | val loss | "
            "Δ vs all-attn(seed) | Δ vs Variant A best |",
            "|----------|----------|------|---------|----------|"
            "---------------------|---------------------|",
        ]
    cells_sorted = sorted(cells, key=lambda c: (c.schedule, c.seed))
    for c in cells_sorted:
        ppl = c.cell.final_val_ppl
        if ppl is None:
            continue
        attn = attn_by_seed.get(c.seed)
        d_attn = f"{ppl - attn:+.2f}" if attn is not None else "—"
        d_va = f"{ppl - va_best:+.2f}" if va_best is not None else "—"
        vl = (f"{c.cell.final_val:.4f}"
              if c.cell.final_val is not None else "—")
        if has_va_paired:
            va_seed = va_by_seed.get(c.seed) if va_by_seed else None
            d_va_seed = (f"{ppl - va_seed:+.2f}"
                         if va_seed is not None else "—")
            lines.append(
                f"| `{c.schedule}` | {c.v_hidden} | {c.seed} | "
                f"{ppl:.2f} | {vl} | {d_attn} | {d_va_seed} | {d_va} |"
            )
        else:
            lines.append(
                f"| `{c.schedule}` | {c.v_hidden} | {c.seed} | "
                f"{ppl:.2f} | {vl} | {d_attn} | {d_va} |"
            )
    lines += [""]
    return lines


def _per_schedule_paired(cells: List[H2Cell],
                         attn_by_seed: Dict[int, float],
                         va_best: Optional[float],
                         va_by_seed: Optional[Dict[int, float]] = None
                         ) -> Tuple[List[str], Dict[str, PairedStat],
                                    Dict[str, PairedStat]]:
    """For each schedule, paired-t Q9d N-seed vs:
       (a) all-attn 5-seed (paired by seed index), and
       (b) Variant A multi-seed (paired by seed index, when available).
    Returns (lines, attn_stats, va_stats).
    """
    by_sched: Dict[str, Dict[int, float]] = {}
    for c in cells:
        if c.cell.final_val_ppl is not None:
            by_sched.setdefault(c.schedule, {})[c.seed] = c.cell.final_val_ppl

    lines = [
        "## Paired-t statistics vs all-attention 5-seed E1 baseline",
        "",
        "All-attention E1 PPL by seed: "
        + ", ".join(f"seed{s}={p:.2f}" for s, p in
                    sorted(attn_by_seed.items()))
        + (f" (mean {sum(attn_by_seed.values())/len(attn_by_seed):.2f})"
           if attn_by_seed else ""),
        "",
        "| schedule | n pairs | Q9d mean | Δ̄ vs attn | std Δ | "
        "sign (Δ<0) | paired-t | two-sided p | meets +5 PPL bar? |",
        "|----------|---------|----------|------------|-------|"
        "------------|----------|-------------|--------------------|",
    ]
    attn_stats: Dict[str, PairedStat] = {}
    for sched in sorted(by_sched):
        q9d = by_sched[sched]
        ps = paired_t(q9d, attn_by_seed)
        if ps is None:
            lines.append(f"| `{sched}` | — | — | — | — | — | — | — | — |")
            continue
        attn_stats[sched] = ps
        q_mean = sum(q9d.values()) / len(q9d)
        meets = "YES" if ps.mean_delta <= 5.0 else "NO"
        sig = "*" if ps.p_value < 0.05 else ""
        lines.append(
            f"| `{sched}` | {ps.n} | {q_mean:.2f} | "
            f"{ps.mean_delta:+.2f} | {ps.std_delta:.2f} | "
            f"{ps.sign_consistency} | {ps.t_stat:+.3f}{sig} | "
            f"{ps.p_value:.4f} | {meets} |"
        )

    # Variant A vs all-attn for context: this answers the question
    # "does VA itself beat all-attn at iso-budget?" using the SAME
    # paired-by-seed-index protocol.  Without this row the table can
    # be misread as "Q9d marginal therefore both arms tied" when in
    # fact VA is also directionally better than all-attn by ~6 PPL
    # but constrained by the same n=3 sample-size limitation.
    has_va_paired_local = va_by_seed is not None and len(va_by_seed) >= 2
    if has_va_paired_local:
        ps_va = paired_t(va_by_seed, attn_by_seed)
        if ps_va is not None:
            va_mean = sum(va_by_seed.values()) / len(va_by_seed)
            meets = "YES" if ps_va.mean_delta <= 5.0 else "NO"
            sig = "*" if ps_va.p_value < 0.05 else ""
            lines.append(
                f"| `k4_m4` (VA) | {ps_va.n} | {va_mean:.2f} | "
                f"{ps_va.mean_delta:+.2f} | {ps_va.std_delta:.2f} | "
                f"{ps_va.sign_consistency} | {ps_va.t_stat:+.3f}{sig} | "
                f"{ps_va.p_value:.4f} | {meets} |"
            )
            attn_stats["__va__"] = ps_va

    lines += [""]

    va_stats: Dict[str, PairedStat] = {}
    has_va_paired = va_by_seed is not None and len(va_by_seed) >= 2
    if has_va_paired:
        n_va_seeds = len(va_by_seed)
        lines += [
            f"## Paired-t statistics vs Variant A {n_va_seeds}-seed H2 "
            "baseline (k=4, m=4)",
            "",
            "Variant A k=4, m=4 PPL by seed: "
            + ", ".join(f"seed{s}={p:.2f}" for s, p in
                        sorted(va_by_seed.items()))
            + (f" (mean {sum(va_by_seed.values())/len(va_by_seed):.2f})"
               if va_by_seed else ""),
            "",
            "| schedule | n pairs | Q9d mean | Δ̄ vs VA | std Δ | "
            "sign (Δ<0) | paired-t | two-sided p | "
            "Q9d-vs-VA win? |",
            "|----------|---------|----------|---------|-------|"
            "------------|----------|-------------|---------------|",
        ]
        for sched in sorted(by_sched):
            q9d = by_sched[sched]
            ps = paired_t(q9d, va_by_seed)
            if ps is None:
                lines.append(f"| `{sched}` | — | — | — | — | — | — | — | — |")
                continue
            va_stats[sched] = ps
            q_mean = sum(q9d.values()) / len(q9d)
            sig = "*" if ps.p_value < 0.05 else ""
            sign_n = int(ps.sign_consistency.split("/")[0])
            sign_d = int(ps.sign_consistency.split("/")[1])
            full_sign = (sign_n == sign_d) and (sign_d >= 3)
            win_strict = (ps.mean_delta <= -5.0 and full_sign)
            win_status = ("STRICT WIN" if win_strict
                          else "weak win" if ps.mean_delta < 0.0
                          else "ON PAR" if ps.mean_delta <= 5.0
                          else "LOSS")
            lines.append(
                f"| `{sched}` | {ps.n} | {q_mean:.2f} | "
                f"{ps.mean_delta:+.2f} | {ps.std_delta:.2f} | "
                f"{ps.sign_consistency} | {ps.t_stat:+.3f}{sig} | "
                f"{ps.p_value:.4f} | {win_status} |"
            )
        lines += [""]
    elif va_best is not None:
        lines += [
            f"## Single-point comparison vs Variant A best "
            f"(k=4, m=4, seed 0 = {va_best:.2f} PPL)",
            "",
            "Variant A multi-seed H2 not on disk; this is the best "
            "available comparator until the matching Variant A H2 run "
            "completes.  Sign-consistency vs Variant A is computed "
            "against the single seed-0 anchor.",
            "",
            "| schedule | n Q9d seeds | Q9d mean | Δ̄ vs VA best | "
            "Q9d outperforms VA on n/n seeds | meets -5 PPL bar? |",
            "|----------|-------------|----------|---------------|"
            "-----------------------------|--------------------|",
        ]
        for sched in sorted(by_sched):
            q9d = by_sched[sched]
            n = len(q9d)
            mean = sum(q9d.values()) / n
            d = mean - va_best
            n_better = sum(1 for v in q9d.values() if v < va_best)
            meets = "YES" if d <= -5.0 else "NO"
            lines.append(
                f"| `{sched}` | {n} | {mean:.2f} | {d:+.2f} | "
                f"{n_better}/{n} | {meets} |"
            )
        lines += [""]
    return lines, attn_stats, va_stats


def render(cells: List[H2Cell],
           attn_by_seed: Dict[int, float],
           va_best: Optional[float],
           out_path: Path,
           va_by_seed: Optional[Dict[int, float]] = None) -> None:
    if not cells:
        out_path.write_text(
            "# H2 Helmholtz paired-confirmation - no results parsed yet.\n"
        )
        print(f"[helm-h2-agg] no results found; wrote {out_path}")
        return

    n_cells = len(cells)
    schedules = sorted({c.schedule for c in cells})
    has_va_paired = va_by_seed is not None and len(va_by_seed) >= 2
    n_q9d_seeds = len({c.seed for c in cells})
    n_va_seeds = len(va_by_seed) if va_by_seed else 0
    seeds_present_q9d = sorted({c.seed for c in cells})
    seeds_present_va = sorted(va_by_seed.keys()) if va_by_seed else []

    lines = [
        "# H2 - Helmholtz (Q9d) S=3 paired confirmation",
        "",
        f"Setup: {n_q9d_seeds} seeds × {len(schedules)} schedules at "
        f"vh=128, 4000-step Tiny Shakespeare config (d=128, L=8, "
        f"mass_mode='logfreq', AdamW 5e-4, batch 16 × block 128, "
        f"free gamma, causal_force=True, ln_after_s_step=True).  "
        f"Seeds present (Q9d): {seeds_present_q9d}.  Seed 0 sourced from "
        f"H1.5; seeds ≥ 1 from H2 / H2 power-up.  "
        f"Cells parsed: **{n_cells}**."
        + (f"  Variant A H2 paired baseline at seeds {seeds_present_va} "
           f"({n_va_seeds} cells)." if has_va_paired else ""),
        "",
        "Schedules under test:",
        "",
        "- `AAAASSSS` vh=128 - **quality lead** (best PPL at S=1; "
        "+1.88 PPL vs Variant A best at seed 0).",
        "- `AASSSSSS` vh=128 - **joint quality+FLOP lead** (only Q9d "
        "cell that beats Variant A outright AND clears the 30% "
        "decode-FLOP-reduction bar at T=4096).",
        "",
    ]
    lines += _ppl_table_per_seed(cells, attn_by_seed, va_best, va_by_seed)
    paired_lines, attn_stats, va_stats = _per_schedule_paired(
        cells, attn_by_seed, va_best, va_by_seed
    )
    lines += paired_lines
    stats = attn_stats  # alias; verdict block below uses attn-arm stats

    # ------------------------------------------------------------
    # Decision verdict
    # ------------------------------------------------------------
    lines += [
        "## H2 decision verdict",
        "",
        "Pre-registered title rule (per `docs/Helmholtz-HSPLM_Path_Forward"
        "_and_Experiments.md` §8.3):",
        "",
        "- **PASS quality arm** iff at the best Q9d schedule: "
        "mean Δ̄ ≥ -5 PPL vs all-attention AND sign-consistency n/n AND "
        "paired-t two-sided p < 0.05.",
        "- **Q9d-vs-Variant-A win** iff at the best Q9d schedule: "
        "mean Δ̄ ≤ -5 PPL vs Variant A best AND sign-consistency n/n.",
        f"- Current sample sizes: Q9d n = {n_q9d_seeds}, "
        f"all-attn n = {len(attn_by_seed)}"
        + (f", VA n = {n_va_seeds}." if has_va_paired else "."),
        "",
    ]

    # Per-schedule per-arm verdict.
    if not stats:
        lines += [
            "**Insufficient cells for paired-t**: at least 2 paired "
            "seeds are needed.  Re-run aggregator after the H2 sweep "
            "completes.",
            "",
        ]
    else:
        # Quality arm: PASS iff mean delta <= +5 PPL AND p < 0.05 AND
        # the cell beats all 3 attention seeds (sign 3/3) — the rule
        # phrases this as "Δ̄ ≥ -5 PPL of all-attention" meaning Q9d is
        # not worse by more than 5 PPL.
        lines += ["### Quality arm (vs all-attention 5-seed E1)", ""]
        for sched, ps in stats.items():
            sign_n = int(ps.sign_consistency.split("/")[0])
            sign_d = int(ps.sign_consistency.split("/")[1])
            quality_within_band = ps.mean_delta <= 5.0
            beats_attn = ps.mean_delta < 0.0
            sig = ps.p_value < 0.05
            full_sign = (sign_n == sign_d) and (sign_d >= 3)
            if quality_within_band and full_sign and sig:
                verdict = "**PASS**"
            elif beats_attn and full_sign:
                verdict = (
                    f"**MARGINAL on strict rule (sample-size limited): "
                    f"Δ̄ = {ps.mean_delta:+.2f} PPL beats attn on every "
                    f"paired seed (sign {ps.sign_consistency}), "
                    f"p = {ps.p_value:.4f}**"
                )
            elif quality_within_band and full_sign:
                verdict = "**MARGINAL**"
            elif quality_within_band:
                verdict = "**MARGINAL**"
            else:
                verdict = "**FAIL**"
            arm_label = ("Variant A (k=4, m=4)" if sched == "__va__"
                         else f"Q9d `{sched}`")
            lines.append(
                f"- {arm_label}: Δ̄ = {ps.mean_delta:+.2f} PPL, "
                f"sign {ps.sign_consistency}, "
                f"p = {ps.p_value:.4f} → {verdict}"
            )
        lines += [""]
        # Pull VA-vs-attn for the summary line below if present.
        ps_va_attn = stats.get("__va__")
        if ps_va_attn is not None:
            n_pairs_va = ps_va_attn.n
            sign_va = ps_va_attn.sign_consistency
            sn_va, sd_va = (int(x) for x in sign_va.split("/"))
            full_sign_va = (sn_va == sd_va)
            best_q9d_arm = min(stats.items(),
                               key=lambda kv: kv[1].mean_delta
                               if kv[0] != "__va__" else 1e9)
            best_q9d_sched, ps_best = best_q9d_arm
            sign_q = ps_best.sign_consistency
            sn_q, sd_q = (int(x) for x in sign_q.split("/"))
            full_sign_q = (sn_q == sd_q)
            both_sig = (ps_va_attn.p_value < 0.05 and
                        ps_best.p_value < 0.05)
            both_full_sign = full_sign_va and full_sign_q
            both_directional = (ps_va_attn.mean_delta < 0.0 and
                                ps_best.mean_delta < 0.0)
            if both_sig:
                summary = ("Both arms statistically beat all-attention "
                           "(p < 0.05 paired-t, "
                           f"Q9d Δ̄ = {ps_best.mean_delta:+.1f} PPL, "
                           f"VA Δ̄ = {ps_va_attn.mean_delta:+.1f} PPL).")
            elif both_directional and both_full_sign:
                summary = (
                    "Q9d's best schedule and Variant A are both "
                    "directionally better than all-attention "
                    f"(Δ̄ ≈ {ps_va_attn.mean_delta:+.1f} PPL for VA, "
                    f"sign {sign_va}; "
                    f"Δ̄ ≈ {ps_best.mean_delta:+.1f} PPL for Q9d "
                    f"`{best_q9d_sched}`, sign {sign_q}). "
                    "At this sample size the paired-t cannot reach "
                    "p < 0.05 against the observed per-seed std on the "
                    "observed effect; this is a sample-size constraint, "
                    "not an absence of signal."
                )
            elif both_directional:
                summary = (
                    f"Q9d's best schedule (`{best_q9d_sched}`) and Variant "
                    "A are both directionally better than all-attention "
                    f"(Δ̄ = {ps_best.mean_delta:+.1f} PPL for Q9d, "
                    f"sign {sign_q}; Δ̄ = {ps_va_attn.mean_delta:+.1f} PPL "
                    f"for VA, sign {sign_va}), but the per-seed dispersion "
                    "is high enough that **at least one seed reverses "
                    "the sign on at least one hybrid arm**.  The earlier "
                    "n=3 \"3/3\" sign-consistency claim was a "
                    "small-sample artifact: with seeds 3 and 4 added, "
                    "the hybrid advantage shrinks and is not robust "
                    "across the full seed panel.  Treat the apparent "
                    "PPL gap as a noisy small-effect signal, not a "
                    "reliable architectural win at this scale."
                )
            else:
                summary = (
                    "Effect of hybridisation on quality is **not "
                    "directionally consistent across seeds**.  "
                    f"Q9d best (`{best_q9d_sched}`) Δ̄ = "
                    f"{ps_best.mean_delta:+.2f} PPL, sign {sign_q}; "
                    f"VA Δ̄ = {ps_va_attn.mean_delta:+.2f} PPL, sign "
                    f"{sign_va}.  Neither arm meets either the "
                    "directional or the strict statistical bar; the "
                    "hybrid advantage seen at n=3 does not survive at n=5."
                )
            lines += ["**Effect-size summary**: " + summary, ""]

        # Q9d-vs-Variant-A win arm.
        if has_va_paired:
            va_mean = sum(va_by_seed.values()) / len(va_by_seed)
            lines += [
                f"### Q9d-vs-Variant-A arm "
                f"(paired-t vs VA k=4, m=4 {n_va_seeds}-seed mean "
                f"{va_mean:.2f})",
                "",
            ]
            for sched, ps in va_stats.items():
                sign_n = int(ps.sign_consistency.split("/")[0])
                sign_d = int(ps.sign_consistency.split("/")[1])
                full_sign = (sign_n == sign_d) and (sign_d >= 3)
                strict_win = (
                    ps.mean_delta <= -5.0 and full_sign
                    and ps.p_value < 0.05
                )
                weak_win = ps.mean_delta < 0.0 and full_sign
                near_par = -5.0 < ps.mean_delta <= 5.0
                if strict_win:
                    verdict = (
                        f"**WIN (strict, p<0.05, sign "
                        f"{ps.sign_consistency})**"
                    )
                elif ps.mean_delta <= -5.0 and full_sign:
                    verdict = (
                        f"**WIN (sign {ps.sign_consistency} + "
                        "≥5 PPL gap, p ≥ 0.05)**"
                    )
                elif weak_win:
                    verdict = (
                        f"**WIN (weak: sign {ps.sign_consistency}, "
                        "<5 PPL gap)**"
                    )
                elif near_par:
                    verdict = "**ON PAR**"
                else:
                    verdict = "**LOSS**"
                lines.append(
                    f"- `{sched}`: Δ̄ = {ps.mean_delta:+.2f} PPL, "
                    f"sign {ps.sign_consistency} (Q9d better than VA), "
                    f"p = {ps.p_value:.4f} → {verdict}"
                )
            lines += [""]
        elif va_best is not None:
            lines += [
                "### Q9d-vs-Variant-A arm "
                f"(vs Variant A best = {va_best:.2f} PPL, seed 0)",
                "",
            ]
            by_sched_dict: Dict[str, Dict[int, float]] = {}
            for c in cells:
                if c.cell.final_val_ppl is not None:
                    by_sched_dict.setdefault(c.schedule, {})[c.seed] = (
                        c.cell.final_val_ppl
                    )
            for sched in sorted(by_sched_dict):
                q9d = by_sched_dict[sched]
                n = len(q9d)
                mean = sum(q9d.values()) / n
                d = mean - va_best
                n_better = sum(1 for v in q9d.values() if v < va_best)
                strong_win = (d <= -5.0) and (n_better == n) and (n >= 3)
                weak_win = (d < 0.0) and (n_better == n)
                near_par = (-5.0 < d <= 5.0)
                if strong_win:
                    verdict = "**WIN (strong)**"
                elif weak_win:
                    verdict = "**WIN (weak: <5 PPL)**"
                elif near_par:
                    verdict = "**ON PAR**"
                else:
                    verdict = "**LOSS**"
                lines.append(
                    f"- `{sched}`: Q9d mean {mean:.2f}, Δ̄ = {d:+.2f} PPL, "
                    f"Q9d outperforms VA on {n_better}/{n} seeds → "
                    f"{verdict}"
                )
            lines += [""]
            lines += [
                "Note: Variant A multi-seed H2 not yet on disk. "
                "When it lands, this aggregator computes a true "
                "paired-t against Variant A as well; the current sign "
                "column is computed against the single seed-0 anchor.",
                "",
            ]

    out_path.write_text("\n".join(lines) + "\n")
    print(f"[helm-h2-agg] wrote {out_path}  ({n_cells} cells, "
          f"{len(schedules)} schedules)")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--h2-dir",
        default=str(SCRIPT_DIR / "results" / "h2_paired_confirmation"),
    )
    ap.add_argument(
        "--h1p5-dir",
        default=str(SCRIPT_DIR / "results" / "h1p5_narrow_v"),
    )
    ap.add_argument(
        "--attn-baseline-dir",
        default=str(ATTN_BASELINE_DIR),
    )
    ap.add_argument(
        "--variant-a-best",
        default=str(VARIANT_A_BEST_PATH),
    )
    ap.add_argument(
        "--variant-a-h2-dir",
        default=str(VARIANT_A_H2_DIR),
        help="Directory holding Variant A H2 cells "
             "hybrid_k4_m4_shakespeare_seed{1,2}_summary.md.  When at "
             "least 2 seeds are present, the aggregator switches to "
             "true paired-t Q9d-vs-VA in addition to the all-attn arm.",
    )
    ap.add_argument("--out", default=None,
                    help="default: <h2-dir>/H2_RESULTS.md")
    args = ap.parse_args()

    h2_dir = Path(args.h2_dir)
    h1p5_dir = Path(args.h1p5_dir)
    h2_dir.mkdir(parents=True, exist_ok=True)
    out_path = (Path(args.out) if args.out is not None
                else h2_dir / "H2_RESULTS.md")

    cells = gather_q9d_cells(h2_dir, h1p5_dir)
    attn_by_seed = load_attn_baseline_5seed(Path(args.attn_baseline_dir))
    va_best = load_variant_a_best(Path(args.variant_a_best))
    va_by_seed = load_variant_a_3seed(
        Path(args.variant_a_best), Path(args.variant_a_h2_dir),
    )

    print(f"[helm-h2-agg] Q9d cells: {len(cells)}")
    print(f"[helm-h2-agg] attn baseline: {len(attn_by_seed)} seeds")
    if va_best is not None:
        print(f"[helm-h2-agg] variant A best: {va_best:.2f} PPL")
    else:
        print("[helm-h2-agg] variant A best: not found")
    print(f"[helm-h2-agg] variant A H2: {len(va_by_seed)} seeds "
          f"({sorted(va_by_seed)})")

    render(cells, attn_by_seed, va_best, out_path, va_by_seed=va_by_seed)


if __name__ == "__main__":
    main()
