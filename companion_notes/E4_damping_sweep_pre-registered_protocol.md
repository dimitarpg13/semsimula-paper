# E4 — SPLM damping sweep: pre-registered protocol

> **Status**: Pre-registered, not yet executed.
> **Date locked**: 2026-04-28.
> **Author**: Dimitar Gueorguiev (with Claude as scribe / sanity-checker).
> **Companion documents**: this protocol; the executed write-up will live
> at `notebooks/conservative_arch/damping_sweep/results/RESULTS.md` once
> complete.
> **Prior protocols this builds on**:
> - [`first_order_ODE_rejection_pre-registered_protocol.md`](./first_order_ODE_rejection_pre-registered_protocol.md)
>   — established the Markov-order regression test on natural transformer
>   trajectories.
> - [`Evidence_for_second_order_ODE_governing_evolution.md`](./Evidence_for_second_order_ODE_governing_evolution.md)
>   — reframed (post-experiment banner) to the **overdamped synthesis**
>   that motivates this sweep.

---

## 1. One-paragraph motivation

The first-order ODE rejection test (`docs/first_order_ODE_rejection_pre-registered_protocol.md`,
outcome **C — first-order not rejected** on GPT-2 small and Pythia-160m,
2026-04-27) established that natural transformer hidden-state trajectories
at one-token resolution are observationally consistent with both a
strict first-order ODE and an **overdamped second-order Lagrangian** in
the limit $\gamma \gg \omega_0$. Two questions follow naturally:

1. **Where on the damping axis does SPLM's best-PPL configuration sit?**
   At what trained $\gamma$ does perplexity peak, and how steeply does
   it degrade as we move away from that operating point?
2. **Does decreasing the damping push SPLM into the regime the rejection
   test *can* reject?** I.e., does an underdamped SPLM produce hidden-
   state trajectories on which $\bar R_2 < \bar R_1$ at the pre-registered
   thresholds, providing a *positive control* for the test?

This protocol locks in the design that answers both.

## 2. Hypotheses

Stated as falsifiable predictions, against which the executed run will
be compared in the §6 decision matrix.

**H1 (LM-quality monotonicity).** Among configurations that complete
training, val perplexity is **monotonically non-increasing as $\gamma$
increases from 0 toward the natural operating point** $\gamma \approx
0.85$, then approximately flat or slightly increasing past it. Formally:
$\mathrm{PPL}(\gamma=0)>\mathrm{PPL}(\gamma=0.1)>
\mathrm{PPL}(\gamma=0.3)\ge\mathrm{PPL}(\gamma=0.85)\le
\mathrm{PPL}(\gamma=2.0)\le\mathrm{PPL}(\gamma=5.0)$.

**H2 (Energy-drift signature).** The E3 energy-drift slope and oscillation
bandwidth are monotone in $\gamma$: small $\gamma$ ⇒ large drift / large
bandwidth; large $\gamma$ ⇒ small drift / small bandwidth. The boundary
between "drift-dominated" and "bounded-oscillation" behaviour locates the
underdamped → overdamped transition empirically.

**H3 (Markov-order positive control).** At sufficiently small $\gamma$
(the "ballistic" / underdamped regime), the §6.4 primary cell of the
Markov-order regression test (kernel ridge, PCA-50, LOSO) on the SPLM's
last-layer hidden states will return $\bar R_2 < \bar R_1$ with
$\rho_{12} \ge 1.20$ and Wilcoxon two-sided $p_{12} < 10^{-3}$ — i.e.,
an **outcome A or B** in the locked decision matrix. At large $\gamma$
the same test will return outcome C (consistent with the natural-
transformer baseline).

**H4 (Trajectory-shape correlation).** §14 acceleration statistics
($a_\parallel<0$ rate, $\lVerta_\parallel\rVert/\lVerta_\perp\rVert$, permutation-null
$z$) are monotone in $\gamma$ in the same direction as PPL: at low
$\gamma$, oscillatory trajectories degrade these signatures; at high
$\gamma$ they recover the natural-transformer values.

## 3. Locked configuration grid

A **single-axis sweep over $\gamma$**, with everything else held fixed
to the existing best-PPL Euler baseline (`sarf_mass_variant`, mass_mode
= `logfreq`):

| Cell | $\gamma$ | per-step damping $1-e^{-\gamma\Delta t}$ at $\Delta t = 1$ | regime |
|---|---|---|---|
| C0 (ballistic floor) | 0.00 | 0 % | undamped |
| C1 (very underdamped) | 0.10 | 9.5 % | underdamped |
| C2 (mildly underdamped) | 0.30 | 25.9 % | mildly underdamped |
| C3 (natural) | 0.85 | 57.2 % | overdamped (matches the trained baseline's converged $\gamma\approx 0.83$) |
| C4 (strongly overdamped) | 2.00 | 86.5 % | overdamped |
| C5 (extreme overdamped) | 5.00 | 99.3 % | quasi-quenched |

All other knobs (locked):

| Axis | Value |
|---|---|
| Architecture | SARF-faithful $\xi$ re-pool, per-token mass `logfreq`, shared $V_\theta$, tied-embedding readout, $L=8$, $\Delta t=1$, $d=128$, $v_{\text{hidden}}=512$, $v_{\text{depth}}=3$ |
| Trainer | `train_splm_sarf_mass.py --mode shakespeare --mass-mode logfreq` (frozen) |
| `--fixed-gamma <γ>` | the only varying argument |
| Optimiser, LR, schedule, weight decay, grad clip, batch, block, seed | identical to the existing `sarf_mass_variant` Tiny Shakespeare baseline (seed = 0) |
| Steps | 4000 |
| Per-token mass `m_t` | learnable, initialised at 1.0 (same as baseline) |
| Per-token logfreq scale $\alpha$ | learnable, initialised at 0.1 (same as baseline) |
| `init_m` | 1.0 |
| Logfreq surprisal source | `notebooks/conservative_arch/sarf_mass_variant/results/logfreq_surprisal.npy` |
| Reference (do not retrain) | existing `splm_sarfmass_logfreq_shakespeare` checkpoint with **freely-learned** $\gamma$ — used as a sanity check (its converged $\gamma \approx 0.83$ should produce PPL close to C3). |

The rationale for picking $\gamma$ values that are not log-spaced is that
we want one cell on each side of the qualitative regime boundaries
(ballistic / underdamped / overdamped / quasi-quenched), with the natural
operating point sitting inside the grid for direct comparison.

## 4. Outputs per cell

For each of the six cells, after training, the experiment produces a
named-tag output bundle:

```
notebooks/conservative_arch/damping_sweep/results/<tag>/
  splm_sarfmass_logfreq_shakespeare_<tag>_summary.md
  splm_sarfmass_logfreq_shakespeare_<tag>_training_log.jsonl
  splm_sarfmass_logfreq_shakespeare_<tag>_loss_curve.png
  splm_sarfmass_logfreq_shakespeare_<tag>_ckpt_latest.pt

  energy_states.npz                 # E3 input (kinetic, potential per layer per sentence)
  energy_drift_summary.json         # drift slope, oscillation bandwidth, exit-vs-mean

  trajectories.pkl                  # per-token last-layer hidden states for §14 + Markov-order
  acceleration_stats.json           # a_parallel sign rate, |a_par|/|a_perp|, permutation z
  markov_order/
    quadruples.npz
    primary_summary.json            # mirror of dynamics_order_test/results/gpt2/
    decision_table.md
```

with `<tag>` $\in$ {`gamma0p00`, `gamma0p10`, `gamma0p30`, `gamma0p85`,
`gamma2p00`, `gamma5p00`}.

## 5. Statistical and decision protocol

### 5.1 PPL — single-seed first cut, multi-seed if borderline

We train **one seed (= 0)** per cell. If the resulting PPL curve is
strictly monotone over the grid we accept H1 with that single seed. If
adjacent cells differ by less than the **2-sigma E1 single-variant
seed variance** (from `multi_seed/results/E1_shakespeare/E1_report.md`,
flagship `splm_em_ln` row), we run two additional seeds (1, 2) on the
two ambiguous adjacent cells before deciding.

E1's reported single-variant seed-to-seed val-PPL std is **σ ≈ 1.8 ppl**
(`splm_em_ln`, 5 seeds). The 2-σ ambiguity threshold therefore sits at
**Δppl < 3.6** between adjacent cells.

### 5.2 Markov-order regression (positive control)

Run the **identical primary-cell pipeline** from
`notebooks/dynamics_order_test/markov_order_regression.py`:

- function class: kernel ridge (RBF), $\alpha$ and $\gamma$ inner-CV-
  selected on the 5-fold inner grid;
- PCA dim: **50** (same as the GPT-2 / Pythia primary cell);
- outer CV: **leave-one-sentence-out**, 50 folds;
- bootstrap: sentence-cluster percentile, $B = 10000$;
- decision matrix: **identical to the locked §6.4 matrix** of the
  first-order rejection protocol.

This gives one decision letter (A / B / C / D) per $\gamma$ cell. The
positive-control prediction is:

| $\gamma$ | predicted decision |
|---|---|
| 0.00 | A or B (test rejects first-order) |
| 0.10 | A or B |
| 0.30 | A or B (likely) |
| 0.85 | C (matches natural-transformer behaviour) |
| 2.00 | C |
| 5.00 | C |

### 5.3 Energy-drift slope

The E3 diagnostic (`energy_drift/energy_drift_diagnostic.py`) computes:

- `drift_slope` — linear regression slope of mean Hamiltonian
  $H_\ell = T_\ell + V_\ell$ across the $L+1$ layer index, normalised by
  `H_0`;
- `bandwidth` — rolling-std of $H_\ell$ across layers, in units of `|H_0|`.

Reported per cell. No formal threshold — these are descriptive.

### 5.4 §14 acceleration statistics

For each trained checkpoint, run the model on the same 50-sentence
$\times$ 5-domain corpus from `dynamics_order_test/data/corpus.json`,
extract per-token last-layer hidden states (the last layer, $h_L$, is
the one the readout sees), and compute on inside-sentence triplets:

- `frac_a_par_negative` — fraction of triplets with $a_\parallel < 0$
  (paper §14: 97.9 % on natural GPT-2);
- `mean_ratio_apar_aperp` — mean $\lVerta_\parallel\rVert / \lVerta_\perp\rVert$
  (paper §14: ~2.0 on natural GPT-2);
- `permutation_z` — z-score of natural-ordering acceleration vs. random-
  permutation null (paper §14: ~23 on natural GPT-2; this is a weak
  diagnostic on small $L$ but we report it for completeness).

### 5.5 Decision matrix (per cell)

For *each* $\gamma$ cell, three independent verdicts are reported:

| Diagnostic | Verdict |
|---|---|
| Markov-order test | A / B / C / D per the locked §6.4 matrix |
| Energy-drift slope | "drifting" if slope $> 0.05$ per layer in $\lVertH_0\rVert$, "bounded" otherwise |
| Trajectory-shape match to natural | "match" if `frac_a_par_negative` $\ge 0.85$ AND `mean_ratio_apar_aperp` $\in [1.0, 3.0]$, "mismatch" otherwise |

The headline reading of the sweep is the *correlation* of these three
verdicts with $\gamma$, not any single cell's pass / fail.

## 6. What the sweep can show

Three qualitative outcomes are possible at the level of the entire grid.

**Outcome α — overdamped synthesis confirmed.** PPL is monotone (best at
$\gamma \approx 0.85$, worsens both directions); Markov-order test
returns A/B at small $\gamma$ and C at large $\gamma$; energy-drift and
trajectory-shape co-vary in the predicted direction. This sweep provides quantitative evidence for the overdamped synthesis.

**Outcome β — partial confirmation.** PPL is monotone but Markov-order
test does *not* reject first-order even at $\gamma = 0$. This is
*possible* if the SPLM's intrinsic representational capacity at $L=8$ is
small enough that even with no damping the trajectories are too short to
encode detectable inertia, or if the LOSO-CV over only 50 sentences with
the small SPLM hidden dim ($d=128$) lacks statistical power. The
sweep is still useful as a damping-axis PPL curve, but the positive-
control claim becomes weaker.

**Outcome γ — non-monotone PPL.** PPL is non-monotone, e.g. C2 wins or
C5 wins. This would falsify H1 and require retraction of the overdamped
synthesis at SPLM scale. This is unexpected based on the existing
symplectic-variant data but is a logically possible outcome of the
sweep.

In every case the sweep is decided by what the data says, with the
verdict pre-committed in this protocol.

## 7. Reproducibility lock-in

Locked-in values that may **not** change between protocol publication
and execution without a follow-up amendment to this document:

| Field | Locked value |
|---|---|
| Six-cell $\gamma$ grid | {0.00, 0.10, 0.30, 0.85, 2.00, 5.00} |
| Architecture | `sarf_mass_variant` `logfreq`, $L=8$, $\Delta t=1$, $d=128$ |
| Per-cell training | seed = 0, 4000 steps, default optimiser (matching the existing baseline) |
| PCA dim for Markov-order regression | 50 |
| Function class for Markov-order regression | kernel ridge (RBF), inner-CV-selected $(\alpha, \gamma_{\text{kernel}})$ |
| Bootstrap | sentence-cluster, $B=10000$ |
| Decision matrix | the one locked in §6.4 of the first-order rejection protocol, applied per cell |
| Energy-drift threshold | `slope > 0.05/layer` for "drifting" |
| Trajectory-shape thresholds | `frac_a_par_negative ≥ 0.85`, `|a_par|/|a_perp| ∈ [1.0, 3.0]` for "match" |
| Compute target | one workstation, 16-core CPU + MPS, ~4–5 wall-clock hours |

If during execution we discover a bug (e.g. NaN at $\gamma = 0$, or a
training run fails to converge for a reason other than the underlying
dynamics), we will document the bug in the executed write-up and
re-run that cell after the fix; we will not silently amend any locked
field.

## 8. Pre-registration audit table

| field | value |
|---|---|
| protocol committed | (pending git commit; date of `git add` of this file) |
| author | Dimitar Gueorguiev |
| scribe | Claude |
| executed by | (TBD; pending) |
| reviewer | (TBD; pending) |
| executed write-up | `notebooks/conservative_arch/damping_sweep/results/RESULTS.md` (pending) |
| status | **Pre-registered, not yet executed.** |

## 9. Forward links

- [`notebooks/conservative_arch/damping_sweep/README.md`](../notebooks/conservative_arch/damping_sweep/README.md)
  — once the experiment runs, this README points at the results tree.
- [`notebooks/conservative_arch/damping_sweep/results/RESULTS.md`](../notebooks/conservative_arch/damping_sweep/results/RESULTS.md)
  — the executed write-up.
- [`notebooks/dynamics_order_test/`](../notebooks/dynamics_order_test/)
  — the pipeline this sweep reuses for its Markov-order positive control.
- [`notebooks/conservative_arch/energy_drift/`](../notebooks/conservative_arch/energy_drift/)
  — the E3 pipeline this sweep reuses for energy-drift readout.
