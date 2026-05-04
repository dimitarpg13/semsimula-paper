# RESULTS — SPLM-1 first-order ablation, **leak-free retrain**

> **What changed vs. `RESULTS.md`**: every cell of the original 6-cell sweep
> was retrained from scratch under `cfg.causal_force = True` (the leak-fix
> documented in `docs/Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md`). All
> other dimensions of the protocol — architecture, optimiser, schedule,
> seeds, data, eval procedure, machine — are bit-for-bit identical to the
> original buggy run. Sweep launcher: `scripts/run_ablation_leakfree.sh`;
> sweep wall-clock: 11{,}519 s (~3 h 12 min) on MPS, 2026-05-03/04.
>
> Forensic re-evaluation companion: `LEAKFREE_RE_EVAL.md` (loaded the buggy
> ckpts under `causal_force = True` and reported a 3.89× / 14.83× asymmetric
> inflation; predicted outcome (c) — that this retrain would invert or null
> the published gap. The retrain confirms the prediction).

---

## Headline

**Outcome A (the original "second-order beats first-order on TinyShakespeare"
verdict) does NOT replicate under a leak-free retrain.**

At matched architecture, matched data, matched optimiser, matched training
budget, and across the same three random seeds, the second-order SPLM em\_ln
at $\gamma^{\ast} = 0.30$ no longer reaches a strictly better validation
perplexity than its structurally first-order ablation (SPLM-1, no $v$-buffer,
no $\gamma$) when *both arms are trained under the leak-free integrator*.
The mean per-seed perplexity gap is $\overline{\Delta} = +1.27$ PPL — **18×
weaker than the buggy $+23.18$ PPL, well below the pre-registered minimum
effect size $\Delta_{\min} = 5.0$ PPL, and outside the pre-registered
prediction interval $[10, 30]$**.

The locked decision rule of §5 of the protocol is **failed** on every
substantive criterion (Δ̄ < 5 PPL, only 2/3 seeds sign-consistent in the
predicted direction). The paired $t$-test on the three differences yields
$t = 1.00$, $\mathrm{df} = 2$, two-sided $p \approx 0.42$. Cohen's paired
$d_z = 0.58$ — a "medium" effect size that is not significant at $S = 3$.

The paper-side consequence: the v2 §15 paragraph claiming a
"$+23.18$-PPL second-order architectural lift" must be retired. The
quantitative claim becomes: under a leak-free integrator, second-order and
first-order SPLMs are statistically indistinguishable on TinyShakespeare.

---

## 1. Per-seed and per-arm final validation perplexity

The numbers below are the final post-training validation losses
(`evaluate(model, val_ids, eval_iters=40, …)`) reported by each cell as its
`[train-fo] DONE` / `[train-em-ln] DONE` line. Both arms used identical
evaluation infrastructure: same validation tokens, same `eval_iters = 40`,
same batch size, same block size — exactly the protocol of the original
sweep.

| Seed | Arm A — SPLM-1 (first-order) | Arm B — SPLM em\_ln $\gamma^{\ast}=0.30$ | $\Delta_s = \mathrm{A} - \mathrm{B}$ |
|---:|---:|---:|---:|
| 0 | 186.64 | 187.43 | **−0.79** |
| 1 | 183.37 | 179.78 | +3.59 |
| 2 | 182.51 | 181.50 | +1.01 |
| **mean** | **184.17** | **182.90** | **+1.27** |
| **std** | 2.18 | 4.01 | 2.20 |
| **min** | 182.51 | 179.78 | −0.79 |
| **max** | 186.64 | 187.43 | +3.59 |

Three properties of note:

- **Sign mixing.** Δ < 0 at seed 0 (SPLM-1 ahead by 0.79 PPL),
  Δ > 0 at seeds 1–2 (SPLM-2 ahead by 3.59 / 1.01 PPL). The 3/3 sign
  consistency criterion of §5 of the protocol is **failed**.
- **Cross-seed variance is now asymmetric.** $\sigma_{\mathrm{A}} = 2.18$
  (essentially unchanged from buggy 2.25), but $\sigma_{\mathrm{B}} = 4.01$
  (~ 2× the buggy 2.03). The leak appears to have been silently shrinking
  cross-seed variance in the second-order arm — without it, SPLM-2's
  per-seed dispersion roughly doubles.
- **Worst-case cross-arm pair.** $\min(\mathrm{Arm\,A}) - \max(\mathrm{Arm\,B})
  = 182.51 - 187.43 = -4.92$ PPL: under the worst pairing, SPLM-1 is
  *ahead* of SPLM-2 by ~5 PPL.

---

## 2. Inferential statistics

### 2.1 Pre-registered substantive criteria (§5 of the protocol)

| Criterion | Locked threshold | Observed | Status |
|---|---|---|---|
| $\overline{\Delta} \ge 5.0$ | 5.0 PPL | 1.27 PPL | **FAIL** (4× short) |
| Per-seed sign consistency | 3 / 3 in the predicted direction | 2 / 3 positive | **FAIL** |
| Pre-registered effect-size interval | $\overline{\Delta} \in [10, 30]$ | 1.27 | **outside** (below interval) |

### 2.2 Inferential tests on the paired sample of three differences

| Test | Statistic | one-sided $p$ | Interpretation |
|---|---|---|---|
| Paired $t$-test | $t = 1.00$, df = 2 | ~0.21 | far from any conventional threshold |
| Wilcoxon signed-rank | $W^{+} = 5$, $S = 3$ | ≥ 0.125 (floor) | floor case; uninformative at $S = 3$ |
| Sign test | 2 / 3 positive | ~0.5 | ≥ 0.125 floor; non-significant |
| Cohen's $d_z$ (paired) | $\overline{\Delta} / \sigma_{\Delta}$ | $d_z = 0.58$ | "medium" — not significant at $S = 3$ |

### 2.3 Direct comparison to the original (buggy) run

| Quantity | Original (buggy) | This run (leak-free) | Change |
|---|---:|---:|---:|
| SPLM-1 mean val\_ppl | 111.50 | 184.17 | +72.67 (1.65× worse) |
| SPLM-2 mean val\_ppl | 88.32 | 182.90 | +94.58 (2.07× worse) |
| Δ̄ (A−B) | **+23.18** | **+1.27** | **−21.91** |
| paired-t (df=2) | 10.09 | 1.00 | collapsed |
| Cohen's $d_z$ | 5.83 | 0.58 | collapsed |
| 3/3 sign consistency | yes | no (2/3) | failed |

The +23.18-PPL gap was **18× weaker** under the leak-free retrain. The leak
disproportionately helped the second-order arm (SPLM-2 inflated 2.07× vs.
SPLM-1 inflated 1.65×), consistent with the forensic re-eval finding of
asymmetric inflation factors (3.89× SPLM-1 vs. 14.83× SPLM-2 on the
trained-buggy ckpts; see `LEAKFREE_RE_EVAL.md`).

---

## 3. Pre-registered decision-rule outcome

The leak-free retrain pre-registered three coarse outcomes for the
substantive criterion test (encoded in
`scripts/run_ablation_leakfree.sh` decision-rule comment block):

- **(a) $\overline{\Delta} \ge +5$ PPL with 3/3 seed sign consistency.**
  SPLM-2's published lead survives; only the absolute numbers shift in
  paper v3 §15.
- **(b) $\overline{\Delta} < +5$ PPL or sign mixing.** SPLM-2's published
  lead does NOT survive; v2 §15 conclusion is retracted.
- **(c) Δ̄ sign-inverted (significantly).** Stronger version of (b);
  SPLM-1 is significantly better in the leak-free regime.

This run lands cleanly in **(b)**: $\overline{\Delta} = +1.27$ (sub-5),
sign-mixed across seeds, paired-$t$ non-significant. The qualitative
conclusion is therefore: **second-order and first-order SPLMs are
statistically indistinguishable on TinyShakespeare under a leak-free
integrator**.

---

## 4. Compute summary

| Cell | Wall-clock | Notes |
|---|---:|---|
| splm1\_leakfree/seed0 | ~1{,}900 s | clean |
| splm1\_leakfree/seed1 | ~1{,}900 s | clean |
| splm1\_leakfree/seed2 | ~1{,}900 s | clean |
| splm2\_gamma0p30\_leakfree/seed0 | ~1{,}900 s | clean |
| splm2\_gamma0p30\_leakfree/seed1 | ~1{,}900 s | clean |
| splm2\_gamma0p30\_leakfree/seed2 | 1{,}929 s | clean |
| **Total** | **11{,}519 s ≈ 3 h 12 min** | |

All six cells converged cleanly. No `TRAINING_FAILED.txt` markers.

---

## 5. Implications for paper v3

This result triggers the following actions on `paper_v3`:

1. **§15 (`subsec:e1-multi-seed`)** — replace the "$+23.18$ PPL with
   paired-t = 10.09, $p = 0.005$" claim with the leak-free numbers above
   ($+1.27$ PPL, $t = 1.00$, sign-mixed). Reframe the multi-seed result
   from "second-order architectural lift" to "second-order does not lift
   over first-order on TinyShakespeare under a leak-free integrator".
2. **§16 (conclusion)** — update the "second-order is the value-add"
   sentence to be measurement-framed: *the inertial term reproducibly
   improves training stability and matches the empirical second-order
   ODE fit (§17), but does not produce a measurable val-PPL gap on
   TinyShakespeare under causally honest training*.
3. **§17 contributions list** — leave the C-list unchanged (the
   contributions were already framed as measurement-framed in v3),
   but the C8 ("first-order ablation establishes second-order as the
   architectural value-add") item is **withdrawn** outright.
4. **§18 (causal-leak audit methodology, planned)** — this RESULTS file
   becomes the worked example: a published, statistically-strong
   architectural claim that did not survive the audit.
5. **§A.3 (paper TMLR-1 leak-correction note)** — the TMLR-1 SPLM
   positive-control regression should also be re-run on the leak-free
   ckpts (the new ckpts under `splm1_leakfree/seed0` and
   `splm2_gamma0p30_leakfree/seed0` are immediate replacements).

---

## 6. Pre-registration deviation note (mirroring §3 of the original RESULTS.md)

This sweep does NOT alter the pre-registered protocol; it re-runs it under
a corrected integrator. The original §3 deviation note (Wilcoxon $p$-floor
at $S = 3$) carries forward unchanged: the paired $t$-test remains the
appropriate primary inferential statistic at $S = 3$.

The §5 decision rule was applied as written. It was **failed** on every
substantive criterion. This is the pre-registered consequence: the original
Outcome A is withdrawn, and §15 of `paper_v3` is updated accordingly.

---

## 7. Reporting plan

Per the §12.5 strategy of `docs/Restructuring_paper_v3_after_causal_leak_bug.md`,
this result triggers:

- (a) This `RESULTS_LEAKFREE.md` is committed alongside the per-cell training
  logs, checkpoints, and figures under
  `notebooks/conservative_arch/first_order_ablation/results/{splm1,splm2_gamma0p30}_leakfree/`.
- (b) `paper_v3/sections/15_conservative_architectures.tex`'s
  `subsec:e1-multi-seed` is rewritten to cite this protocol's outcome
  (Δ̄ = 1.27 PPL across $S = 3$ seeds, paired-$t$ $p \approx 0.4$,
  sign-mixed).
- (c) The companion repo (`semsimula-paper`) gets this RESULTS file under
  `companion_notes/RESULTS_LEAKFREE_first_order_ablation.md` and the
  `paper_v3` PDF re-rendered with the updated §15 numbers.
