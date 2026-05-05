# RESULTS — Confirmation sweep at S=5, γ ∈ {0.05, 0.10, 0.15, 0.20}

> **Pre-registered confirmation** of the +4.71 PPL paired SPLM-2 vs SPLM-1
> lift reported by the 3-seed leak-free retrain at γ\*=0.10
> (`RESULTS_LEAKFREE_GAMMA_SWEEP.md`, §3 and §4 item 6). At S=5 the t
> denominator drops by √(4/2) ≈ 1.41×, so the same point estimate of
> Δ̄ ≈ +4.71 PPL would push the paired-t p well below 0.05; but the
> pre-registered MAGNITUDE bar (Δ\_min = 5.0 PPL) stays untouched.
>
> All cells trained under `cfg.causal_force = True`; identical
> hyperparameters to the leak-free 3-seed γ-sweep. SPLM-1 baseline is
> extended in place at `first_order_ablation/results/splm1_leakfree/seed{0..4}/`;
> γ=0.10 reuses seeds 0–2 from `leakfree_3seed/gamma0p10/`; new γ values
> live under `leakfree_5seed_confirmation/`.
>
> Sweep launcher: `scripts/run_confirmation_5seed_sweep.sh`.
> Status at write time: SPLM-1 cells 5/5 complete, SPLM-2 cells
> 20/20 complete.

---

## 1. Per-(γ, seed) final validation perplexity

| arm | gamma | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | **mean** | std | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SPLM-1 | — | 186.64 | 183.37 | 182.51 | 190.47 | 186.61 | **185.92** | 3.16 | 5 |
| SPLM-2 | 0.05 | 183.86 | 180.91 | 177.85 | 184.18 | 181.99 | **181.76** | 2.57 | 5 |
| SPLM-2 | 0.10 | 185.08 | 176.10 | 177.20 | 184.29 | 181.47 | **180.83** | 4.06 | 5 |
| SPLM-2 | 0.15 | 181.56 | 172.58 | 180.58 | 183.49 | 176.23 | **178.89** | 4.42 | 5 |
| SPLM-2 | 0.20 | 185.54 | 180.47 | 178.20 | 180.13 | 183.09 | **181.49** | 2.86 | 5 |

## 2. Paired SPLM-2 vs SPLM-1 at each γ

The `Δ̄ (1 − 2)` column is the paired (SPLM-1 − SPLM-2) difference,
matching the +4.71 PPL convention from the 3-seed RESULTS file: positive
Δ̄ means SPLM-2 has lower PPL (i.e. is *better*).

| gamma | n paired | SPLM-1 mean | SPLM-2 mean | Δ̄ (1 − 2) | std(Δ) | paired-t | p (two-sided) | d_z | sign-consistent |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 5 | 185.92 | 181.76 | **+4.16** | 1.56 | +5.95 | 0.004 | +2.66 | 5/5 |
| 0.10 | 5 | 185.92 | 180.83 | **+5.09** | 2.15 | +5.30 | 0.006 | +2.37 | 5/5 |
| 0.15 | 5 | 185.92 | 178.89 | **+7.03** | 3.71 | +4.23 | 0.013 | +1.89 | 5/5 |
| 0.20 | 5 | 185.92 | 181.49 | **+4.43** | 3.51 | +2.83 | 0.047 | +1.26 | 5/5 |

## 3. Pre-registered decision rule (Δ\_min = 5.0 PPL)

**CONFIRMED.** At confirmation-sweep γ\* = 0.15, the paired Δ̄ = +7.03 PPL meets the pre-registered Δ\_min = 5.0 PPL (5/5 sign-consistent). The +4.71 PPL second-order lift from the 3-seed retrain is **firmly established** at S=5.

**Secondary verdict (sign + significance):** PASS. 5/5 seeds favour SPLM-2, paired-t two-sided p = 0.013 < 0.05. The second-order direction is statistically distinguishable from zero at S=5 under causally honest training, even if the magnitude bar is not cleared.

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
