#!/usr/bin/env python3
"""Aggregate the confirmation sweep at S=5 narrowed to gamma in {0.05, 0.10, 0.15, 0.20}.

Reads every `*_summary.md` produced by the per-cell trainers, parses the
final val_ppl, builds per-(gamma, seed) tables, computes paired-t and
related statistics for SPLM-2 vs SPLM-1 at each gamma, identifies the
confirmation-sweep gamma*, applies the pre-registered Delta_min = 5.0
PPL decision rule, and writes RESULTS_CONFIRMATION_S5.md.

Run from anywhere:
    python3 aggregate_confirmation_5seed.py

Output:
    notebooks/conservative_arch/ln_damping_sweep/results/leakfree_5seed_confirmation/RESULTS_CONFIRMATION_S5.md
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CONS_DIR = Path(__file__).resolve().parent.parent
ABL_RESULTS = CONS_DIR / "first_order_ablation" / "results"
SWEEP_RESULTS = CONS_DIR / "ln_damping_sweep" / "results"
CONF_RESULTS = SWEEP_RESULTS / "leakfree_5seed_confirmation"

# ---- Pre-registered constants ------------------------------------------
DELTA_MIN_PPL = 5.0          # pre-registered minimum effect size
SIGN_THRESHOLD = 4           # >= 4/5 seeds sign-consistent for the secondary verdict
ALPHA = 0.05                 # significance level for the secondary verdict

# ---- Cell ledger -------------------------------------------------------
# For each (model, gamma) we list the per-seed cell directories. Seeds 0-2
# at SPLM-1 and gamma=0.10 are the reused leak-free 3-seed cells; seeds
# 3-4 are the new confirmation cells. New gamma values (0.05, 0.15, 0.20)
# are full S=5 in the confirmation tree.
SPLM1_CELLS: Dict[int, Path] = {
    0: ABL_RESULTS / "splm1_leakfree" / "seed0",
    1: ABL_RESULTS / "splm1_leakfree" / "seed1",
    2: ABL_RESULTS / "splm1_leakfree" / "seed2",
    3: ABL_RESULTS / "splm1_leakfree" / "seed3",  # new in confirmation sweep (in place)
    4: ABL_RESULTS / "splm1_leakfree" / "seed4",  # new in confirmation sweep (in place)
}

# SPLM-2 ledger: gamma_value -> {seed -> path}
SPLM2_CELLS: Dict[float, Dict[int, Path]] = {
    0.05: {s: CONF_RESULTS / "gamma0p05" / f"seed{s}" for s in range(5)},
    0.10: {
        0: SWEEP_RESULTS / "leakfree_3seed" / "gamma0p10" / "seed0",  # reused
        1: SWEEP_RESULTS / "leakfree_3seed" / "gamma0p10" / "seed1",  # reused
        2: SWEEP_RESULTS / "leakfree_3seed" / "gamma0p10" / "seed2",  # reused
        3: CONF_RESULTS / "gamma0p10" / "seed3",                       # new
        4: CONF_RESULTS / "gamma0p10" / "seed4",                       # new
    },
    0.15: {s: CONF_RESULTS / "gamma0p15" / f"seed{s}" for s in range(5)},
    0.20: {s: CONF_RESULTS / "gamma0p20" / f"seed{s}" for s in range(5)},
}

# ---- val_ppl extraction ------------------------------------------------
PPL_RE = re.compile(r"Final\s+val\s+loss[^()]*\(\s*ppl\s+([0-9.]+)\s*\)")


def parse_val_ppl(cell_dir: Path) -> Optional[float]:
    """Find the *_summary.md inside cell_dir and parse its val_ppl.

    Returns None if no summary.md is present (cell still running or never ran).
    """
    if not cell_dir.exists():
        return None
    summaries = sorted(cell_dir.glob("*_summary.md"))
    if not summaries:
        return None
    text = summaries[0].read_text()
    m = PPL_RE.search(text)
    if not m:
        return None
    return float(m.group(1))


# ---- Statistics --------------------------------------------------------
def mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return (math.nan, math.nan)
    n = len(xs)
    mu = sum(xs) / n
    if n < 2:
        return (mu, 0.0)
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return (mu, math.sqrt(var))


def paired_t(a: List[float], b: List[float]) -> Dict[str, float]:
    """Paired-t for arms a vs b on the same seed indexing.

    Returns {n, delta_bar, delta_std, t, df, p_two_sided, d_z, sign_pos, sign_total}.
    """
    assert len(a) == len(b), f"length mismatch: {len(a)} vs {len(b)}"
    deltas = [ai - bi for ai, bi in zip(a, b)]
    n = len(deltas)
    if n < 2:
        return {
            "n": n,
            "delta_bar": deltas[0] if deltas else math.nan,
            "delta_std": math.nan,
            "t": math.nan,
            "df": max(n - 1, 0),
            "p_two_sided": math.nan,
            "d_z": math.nan,
            "sign_pos": int(deltas and deltas[0] > 0),
            "sign_total": n,
        }
    mu, sd = mean_std(deltas)
    if sd == 0:
        t = math.inf if mu != 0 else 0.0
        d_z = math.inf if mu != 0 else 0.0
    else:
        t = mu / (sd / math.sqrt(n))
        d_z = mu / sd
    df = n - 1
    p_two = student_t_two_sided_p(abs(t), df)
    return {
        "n": n,
        "delta_bar": mu,
        "delta_std": sd,
        "t": t,
        "df": df,
        "p_two_sided": p_two,
        "d_z": d_z,
        "sign_pos": sum(1 for d in deltas if d > 0),
        "sign_total": n,
    }


def student_t_two_sided_p(abs_t: float, df: int) -> float:
    """Two-sided p-value for a Student-t statistic with `df` degrees of freedom.

    Pure-Python via the regularised incomplete beta function:
        P(|T| > |t|) = I_{x}(df/2, 1/2)  with  x = df / (df + t^2)
    """
    if not math.isfinite(abs_t):
        return 0.0
    if df <= 0:
        return math.nan
    x = df / (df + abs_t * abs_t)
    return _betainc(df / 2.0, 0.5, x)


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a,b) via continued fraction.

    Adapted from Numerical Recipes, eq. 6.4.9. Sufficient for df >= 1, |t| < ~50.
    """
    if x < 0.0 or x > 1.0:
        raise ValueError(f"betainc x out of [0,1]: {x}")
    if x == 0.0 or x == 1.0:
        bt = 0.0
    else:
        bt = math.exp(
            math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
            + a * math.log(x) + b * math.log(1.0 - x)
        )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _betacf(a: float, b: float, x: float, eps: float = 3.0e-7, max_iter: int = 200) -> float:
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1.0e-30:
        d = 1.0e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1.0e-30:
            d = 1.0e-30
        c = 1.0 + aa / c
        if abs(c) < 1.0e-30:
            c = 1.0e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1.0e-30:
            d = 1.0e-30
        c = 1.0 + aa / c
        if abs(c) < 1.0e-30:
            c = 1.0e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    return h


# ---- Aggregation -------------------------------------------------------
@dataclass
class CellRow:
    seed: int
    val_ppl: Optional[float]
    cell_dir: Path

    @property
    def status(self) -> str:
        return "OK" if self.val_ppl is not None else "MISSING"


@dataclass
class GammaArm:
    gamma: float
    cells: List[CellRow] = field(default_factory=list)

    @property
    def ppls(self) -> List[float]:
        return [c.val_ppl for c in self.cells if c.val_ppl is not None]

    @property
    def n_complete(self) -> int:
        return len(self.ppls)


def collect() -> Tuple[List[CellRow], Dict[float, GammaArm]]:
    splm1: List[CellRow] = []
    for seed, cell in sorted(SPLM1_CELLS.items()):
        splm1.append(CellRow(seed=seed, val_ppl=parse_val_ppl(cell), cell_dir=cell))
    splm2: Dict[float, GammaArm] = {}
    for gamma in sorted(SPLM2_CELLS.keys()):
        arm = GammaArm(gamma=gamma)
        for seed, cell in sorted(SPLM2_CELLS[gamma].items()):
            arm.cells.append(CellRow(seed=seed, val_ppl=parse_val_ppl(cell), cell_dir=cell))
        splm2[gamma] = arm
    return splm1, splm2


def fmt_ppl(p: Optional[float]) -> str:
    return "—" if p is None else f"{p:.2f}"


def render_table_per_seed(splm1: List[CellRow], splm2: Dict[float, GammaArm]) -> str:
    seeds_sorted = sorted({c.seed for c in splm1} | {s for arm in splm2.values() for s in (c.seed for c in arm.cells)})
    header_seeds = "".join(f"| seed {s} " for s in seeds_sorted)
    sep_seeds = "".join("|---:" for _ in seeds_sorted)
    lines: List[str] = []
    lines.append(f"| arm | gamma {header_seeds}| **mean** | std | n |")
    lines.append(f"|---|---:{sep_seeds}|---:|---:|---:|")
    cells_by_seed = {c.seed: c.val_ppl for c in splm1}
    row = "| SPLM-1 | — | " + " ".join(f"{fmt_ppl(cells_by_seed.get(s))} |" for s in seeds_sorted)
    mu, sd = mean_std([p for p in cells_by_seed.values() if p is not None])
    n_done = sum(1 for p in cells_by_seed.values() if p is not None)
    row += f" **{mu:.2f}** | {sd:.2f} | {n_done} |"
    lines.append(row)
    for gamma in sorted(splm2.keys()):
        arm = splm2[gamma]
        cs = {c.seed: c.val_ppl for c in arm.cells}
        row = f"| SPLM-2 | {gamma:.2f} | " + " ".join(f"{fmt_ppl(cs.get(s))} |" for s in seeds_sorted)
        mu, sd = mean_std(arm.ppls)
        row += f" **{mu:.2f}** | {sd:.2f} | {arm.n_complete} |"
        lines.append(row)
    return "\n".join(lines)


def render_paired_table(splm1: List[CellRow], splm2: Dict[float, GammaArm]) -> Tuple[str, Dict[float, Dict[str, float]]]:
    splm1_by_seed = {c.seed: c.val_ppl for c in splm1}
    lines: List[str] = []
    lines.append("| gamma | n paired | SPLM-1 mean | SPLM-2 mean | Δ̄ (1 − 2) | std(Δ) | paired-t | p (two-sided) | d_z | sign-consistent |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    stats_by_gamma: Dict[float, Dict[str, float]] = {}
    for gamma in sorted(splm2.keys()):
        arm = splm2[gamma]
        # Pair seed-by-seed; only seeds present in BOTH arms are used
        a, b = [], []
        for c in arm.cells:
            p1 = splm1_by_seed.get(c.seed)
            p2 = c.val_ppl
            if p1 is None or p2 is None:
                continue
            a.append(p1)
            b.append(p2)
        if not a:
            lines.append(f"| {gamma:.2f} | 0 | — | — | — | — | — | — | — | — |")
            continue
        mu1, _ = mean_std(a)
        mu2, _ = mean_std(b)
        s = paired_t(a, b)
        stats_by_gamma[gamma] = s
        lines.append(
            f"| {gamma:.2f} | {len(a)} | {mu1:.2f} | {mu2:.2f} | "
            f"**{s['delta_bar']:+.2f}** | {s['delta_std']:.2f} | "
            f"{s['t']:+.2f} | {s['p_two_sided']:.3f} | {s['d_z']:+.2f} | "
            f"{s['sign_pos']}/{s['sign_total']} |"
        )
    return "\n".join(lines), stats_by_gamma


def render_results_md(splm1: List[CellRow], splm2: Dict[float, GammaArm]) -> str:
    n_splm1 = sum(1 for c in splm1 if c.val_ppl is not None)
    n_splm2_total = sum(arm.n_complete for arm in splm2.values())
    n_splm2_expected = sum(len(arm.cells) for arm in splm2.values())

    table_seeds = render_table_per_seed(splm1, splm2)
    table_paired, paired_stats = render_paired_table(splm1, splm2)

    # Pick gamma* = argmin SPLM-2 mean val_ppl among gammas with full S=5 paired data
    gamma_star: Optional[float] = None
    gamma_star_mu = math.inf
    for gamma, arm in splm2.items():
        if gamma not in paired_stats:
            continue
        if paired_stats[gamma]["sign_total"] < 5:
            continue
        mu, _ = mean_std(arm.ppls)
        if mu < gamma_star_mu:
            gamma_star_mu = mu
            gamma_star = gamma

    # Decision rule on the pre-registered Delta_min = 5.0 PPL at gamma*
    primary_verdict_lines: List[str] = []
    secondary_verdict_lines: List[str] = []
    if gamma_star is None:
        primary_verdict_lines.append(
            "Sweep is incomplete (no gamma cell has all 5 seeds with paired SPLM-1). "
            "Re-run this aggregator after the sweep finishes."
        )
    else:
        s = paired_stats[gamma_star]
        delta = s["delta_bar"]
        sign = f"{s['sign_pos']}/{s['sign_total']}"
        primary_pass = (delta >= DELTA_MIN_PPL) and (s["sign_pos"] >= SIGN_THRESHOLD)
        secondary_pass = (s["sign_pos"] >= SIGN_THRESHOLD) and (s["p_two_sided"] < ALPHA) and (delta > 0)

        if primary_pass:
            primary_verdict_lines.append(
                f"**CONFIRMED.** At confirmation-sweep γ\\* = {gamma_star:.2f}, the paired Δ̄ "
                f"= +{delta:.2f} PPL meets the pre-registered Δ\\_min = {DELTA_MIN_PPL:.1f} PPL "
                f"({sign} sign-consistent). The +4.71 PPL second-order lift from the 3-seed "
                f"retrain is **firmly established** at S=5."
            )
        elif delta < 0:
            primary_verdict_lines.append(
                f"**REFUTED (sign-inverted).** At confirmation-sweep γ\\* = {gamma_star:.2f}, "
                f"the paired Δ̄ = {delta:+.2f} PPL is sign-inverted relative to the +4.71 PPL "
                f"3-seed point estimate ({sign} sign-consistent). The published +23.18 PPL "
                f"second-order lift does NOT survive a leak-free S=5 retrain."
            )
        else:
            primary_verdict_lines.append(
                f"**REFUTED (magnitude).** At confirmation-sweep γ\\* = {gamma_star:.2f}, the "
                f"paired Δ̄ = +{delta:.2f} PPL falls short of the pre-registered Δ\\_min = "
                f"{DELTA_MIN_PPL:.1f} PPL by {DELTA_MIN_PPL - delta:.2f} PPL ({sign} "
                f"sign-consistent). The +4.71 PPL second-order lift is *suggestive in "
                f"direction* but does NOT clear the pre-registered minimum effect size at S=5."
            )

        if secondary_pass:
            secondary_verdict_lines.append(
                f"**Secondary verdict (sign + significance):** PASS. {sign} seeds favour "
                f"SPLM-2, paired-t two-sided p = {s['p_two_sided']:.3f} < {ALPHA}. The "
                f"second-order direction is statistically distinguishable from zero at S=5 "
                f"under causally honest training, even if the magnitude bar is not cleared."
            )
        else:
            secondary_verdict_lines.append(
                f"**Secondary verdict (sign + significance):** {sign} sign-consistent, "
                f"paired-t two-sided p = {s['p_two_sided']:.3f}. The direction does not "
                f"clear the {SIGN_THRESHOLD}/{s['sign_total']} sign-threshold and "
                f"α = {ALPHA} significance simultaneously."
            )

    verdict_block = "\n".join(primary_verdict_lines + ["", *secondary_verdict_lines])

    body = f"""# RESULTS — Confirmation sweep at S=5, γ ∈ {{0.05, 0.10, 0.15, 0.20}}

> **Pre-registered confirmation** of the +4.71 PPL paired SPLM-2 vs SPLM-1
> lift reported by the 3-seed leak-free retrain at γ\\*=0.10
> (`RESULTS_LEAKFREE_GAMMA_SWEEP.md`, §3 and §4 item 6). At S=5 the t
> denominator drops by √(4/2) ≈ 1.41×, so the same point estimate of
> Δ̄ ≈ +4.71 PPL would push the paired-t p well below 0.05; but the
> pre-registered MAGNITUDE bar (Δ\\_min = 5.0 PPL) stays untouched.
>
> All cells trained under `cfg.causal_force = True`; identical
> hyperparameters to the leak-free 3-seed γ-sweep. SPLM-1 baseline is
> extended in place at `first_order_ablation/results/splm1_leakfree/seed{{0..4}}/`;
> γ=0.10 reuses seeds 0–2 from `leakfree_3seed/gamma0p10/`; new γ values
> live under `leakfree_5seed_confirmation/`.
>
> Sweep launcher: `scripts/run_confirmation_5seed_sweep.sh`.
> Status at write time: SPLM-1 cells {n_splm1}/5 complete, SPLM-2 cells
> {n_splm2_total}/{n_splm2_expected} complete.

---

## 1. Per-(γ, seed) final validation perplexity

{table_seeds}

## 2. Paired SPLM-2 vs SPLM-1 at each γ

The `Δ̄ (1 − 2)` column is the paired (SPLM-1 − SPLM-2) difference,
matching the +4.71 PPL convention from the 3-seed RESULTS file: positive
Δ̄ means SPLM-2 has lower PPL (i.e. is *better*).

{table_paired}

## 3. Pre-registered decision rule (Δ\\_min = 5.0 PPL)

{verdict_block}

---

## 4. Implications for paper v3

To be filled in based on the verdict above. Likely §15 / §17 surgical
edits will follow the same shape as the 3-seed leak-free retrain
(see `RESULTS_LEAKFREE_GAMMA_SWEEP.md` §4).

---

## 5. Compute summary

Per-cell wall-clock and PER-CELL train_stdout.log entries are not
re-aggregated here; see each `*/train_stdout.log` for the per-cell
elapsed line, and `sweep_full.log` for the launcher's overall timing.
"""
    return body


def main() -> int:
    splm1, splm2 = collect()

    # Print a quick sanity summary on stderr-style stdout so we know
    # whether any cell is still missing.
    splm1_done = sum(1 for c in splm1 if c.val_ppl is not None)
    splm2_done = sum(arm.n_complete for arm in splm2.values())
    splm2_total = sum(len(arm.cells) for arm in splm2.values())
    print(f"[aggregate] SPLM-1 cells:   {splm1_done}/{len(splm1)}")
    print(f"[aggregate] SPLM-2 cells:   {splm2_done}/{splm2_total}")
    for gamma in sorted(splm2.keys()):
        arm = splm2[gamma]
        ppls = ", ".join(f"{p:.2f}" if p is not None else "—" for p in [c.val_ppl for c in arm.cells])
        print(f"[aggregate]   gamma={gamma:.2f}  ({arm.n_complete}/{len(arm.cells)})  ppls=[{ppls}]")

    md = render_results_md(splm1, splm2)
    out_md = CONF_RESULTS / "RESULTS_CONFIRMATION_S5.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md)
    print(f"[aggregate] wrote {out_md}")

    # Also dump a machine-readable sidecar for the paper-update script.
    splm1_by_seed = {c.seed: c.val_ppl for c in splm1}
    payload = {
        "splm1_val_ppl": {str(s): v for s, v in splm1_by_seed.items()},
        "splm2_val_ppl": {
            f"{gamma:.2f}": {str(c.seed): c.val_ppl for c in arm.cells}
            for gamma, arm in splm2.items()
        },
        "delta_min_ppl": DELTA_MIN_PPL,
        "sign_threshold": SIGN_THRESHOLD,
        "alpha": ALPHA,
    }
    out_json = CONF_RESULTS / "results_confirmation_s5.json"
    out_json.write_text(json.dumps(payload, indent=2, default=lambda x: None if isinstance(x, float) and math.isnan(x) else x))
    print(f"[aggregate] wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
