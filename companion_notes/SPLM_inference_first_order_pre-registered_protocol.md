# Pre-Registered Protocol — SPLM Inference Markov-Order Test

> Pre-registration document, drafted **April 29, 2026**, by Dimitar Gueorguiev with Claude.
> Companion to:
> *Semantic Simulation: A Prescriptive Lagrangian Framework for Efficient Semantic Inference* (Gueorguiev, 2026), v3.
> Companion experiments:
> - First-order rejection on trained transformers: [`companion_notes/first_order_ODE_rejection_pre-registered_protocol.md`](./first_order_ODE_rejection_pre-registered_protocol.md) — Outcome **C** (first-order not rejected) on GPT-2 small and Pythia-160m.
> - SPLM-1 first-order ablation: [`companion_notes/SPLM-1_ablation_pre-registered_protocol.md`](./SPLM-1_ablation_pre-registered_protocol.md) — pre-registered concurrently; in flight at the time of this draft, projected Outcome **A** (training-time value of the inertial term) on seed 0.

> **Status.** Pre-registered, not yet executed. This document fixes the experimental design, the analysis pipeline, and the decision rule **before** any hidden-state extraction or regression is run on SPLM checkpoints. The committing commit hash is the timestamp of pre-registration.

---

## 1. Question

the paper advances a **two-tier** dynamical claim about SPLM:

- **Tier 1 — generative second-order.** SPLM is induced by the damped Lagrangian $\mathcal{L} = \tfrac{1}{2} m_t \lVert \dot h_t \rVert^2 - V_\theta(\xi_t, h_t)$. The integration graph is structurally second-order; the velocity buffer and the damping coefficient $\gamma$ are integral parts of the **training-time** computation. (Empirically supported by the in-flight SPLM-1 ablation, projected Outcome A.)
- **Tier 2 — observational first-order at the trained inference fixed point.** Once training has converged, the inference dynamics on the attractor manifold reduce, in the heavy-mass / overdamped regime, to an effectively first-order gradient flow $\gamma \dot h_t \approx -\nabla V_\theta$. The trained SPLM should therefore look — to a Markov-order regression test of its hidden-state trajectory — *exactly like* a normal trained transformer (GPT-2 / Pythia, Outcome C).

Tier 1 is now empirically supported. Tier 2 is currently a **theoretical prediction only**, and is the only un-tested claim in the paper.

This experiment closes that gap. It runs the **identical** Markov-order regression test described in `companion_notes/first_order_ODE_rejection_pre-registered_protocol.md` (the protocol used for GPT-2 small and Pythia-160m), but on hidden-state trajectories produced by trained SPLM and SPLM-1 checkpoints. The question being asked is exactly the same:

> Is $h_t$ alone a sufficient statistic for predicting $h_{t+1}$ at the inference-time hidden state of a trained SPLM, or is at least one extra lag $h_{t-1}$ required?

---

## 2. Hypotheses

Let $\hat F_k$ denote the best predictor in a fixed function class for $h_{t+1}^{(L)}$ given the lag-$k$ history $(h_t^{(L)}, h_{t-1}^{(L)}, \ldots, h_{t-k+1}^{(L)})$, where $h_t^{(L)}$ is the SPLM's final-integration-step hidden state at token position $t$. Define per-fold mean-squared residuals $R_k$ exactly as in §2 of the original protocol.

| Hypothesis | Operational form | Theoretical reading |
|---|---|---|
| $H_0$ (first-order suffices) | $R_1 \approx R_2 \approx R_3$ | confirms paper §12 last paragraph: trained SPLM is observationally first-order at its inference fixed point |
| $H_1$ (second-order required) | $R_1 > R_2 \approx R_3$ | refutes paper §12 last paragraph: trained SPLM is observationally second-order, despite identical Markov signature on standard transformers |
| $H_2$ (higher-than-second order) | $R_1 > R_2 > R_3$ | refutes paper §12 last paragraph more strongly |

the paper's prediction is $H_0$ (the same prediction that held for GPT-2 / Pythia in the prior experiment). The pre-registered prediction in this protocol is therefore:

- **SPLM em\_ln at $\gamma^{\ast} = 0.30$:** Outcome **C** (first-order not rejected), $\rho_{12} \in [0.97, 1.05]$, mirroring the GPT-2 small / Pythia-160m bands ($0.979$ and $0.985$ respectively).
- **SPLM-1 (positive control):** Outcome **C**. SPLM-1 *is* first-order by construction, so anything else would indicate a bug in the regression pipeline.
- **Comparison SPLM vs. SPLM-1:** their Markov-order signatures should be **indistinguishable** (overlapping 95 % cluster-bootstrap CIs on $\bar R_1 - \bar R_2$), despite the SPLM-1 PPL deficit measured by the SPLM-1 ablation.

The third prediction is the *theoretically interesting* one. It says: the training-time inertial term is **not** detectable as inertia at the inference fixed point. The Lagrangian shapes the **learning trajectory**, not the trained-state evolution. If both arms test as Outcome C with overlapping effect sizes, this is a strong empirical confirmation of the paper's "generative ≠ observational" framing.

---

## 3. Data

### 3.1 Primary corpus — Tiny Shakespeare validation split (in-distribution)

The trained SPLM and SPLM-1 checkpoints were optimised on Tiny Shakespeare; their inference fixed point is defined relative to that distribution. We therefore use the **held-out Tiny Shakespeare validation split** (the same 16 901-token split used for the PPL evaluation in the SPLM-1 ablation) as the primary corpus for this test.

Sentence segmentation. The Shakespeare validation text is segmented into "sentences" by splitting on the regex `r"[.!?]\s+|\n\s*\n"`, then filtered to sentences with ≥ 16 BPE tokens (so each sentence yields ≥ 13 inside-sentence quadruples). Sample size target: the **first 100** such filtered sentences in document order, deterministically.

Expected yield: ~2 200 quadruples (cf. ~1 200 in the GPT-2 small primary cell), giving comparable or higher statistical power.

### 3.2 Secondary corpus — same 50-sentence, 5-domain corpus as GPT-2 / Pythia (out-of-distribution)

For direct head-to-head comparability with the GPT-2 / Pythia results, we additionally run the test on `notebooks/dynamics_order_test/data/corpus.json`, BPE-tokenised through the GPT-2 tokeniser (the same tokeniser SPLM was trained with, so the token IDs are exactly aligned). This is **out-of-distribution** for the Shakespeare-trained SPLM, so the residuals will be larger, but the *Markov order* of the residuals is the question — and that is dimensionally invariant to the underlying scale.

The secondary corpus is a robustness probe, not the primary decision cell.

### 3.3 Checkpoints

The 6 checkpoints produced by the SPLM-1 ablation sweep (`notebooks/conservative_arch/first_order_ablation/results/`):

| Arm | Checkpoint | Purpose |
|---|---|---|
| SPLM em\_ln $\gamma^{\ast} = 0.30$ | `splm2_gamma0p30/seed0/...ckpt_latest.pt` | primary cell, seed 0 |
| SPLM em\_ln $\gamma^{\ast} = 0.30$ | `splm2_gamma0p30/seed1/...ckpt_latest.pt` | primary cell, seed 1 |
| SPLM em\_ln $\gamma^{\ast} = 0.30$ | `splm2_gamma0p30/seed2/...ckpt_latest.pt` | primary cell, seed 2 |
| SPLM-1 | `splm1/seed0/...ckpt_latest.pt` | positive control, seed 0 |
| SPLM-1 | `splm1/seed1/...ckpt_latest.pt` | positive control, seed 1 |
| SPLM-1 | `splm1/seed2/...ckpt_latest.pt` | positive control, seed 2 |

All six checkpoints are produced by the same machine, same Python environment, and same data pipeline as each other; the only differences are the integrator (first- vs. second-order) and the seed.

### 3.4 Hidden state choice

For each (checkpoint, sentence) pair, we run a forward pass with `return_trajectory=True` and extract the **final integration step** hidden state $h_t^{(L)}$ at every token position $t$. This is the SPLM analogue of GPT-2 small's last-layer state $h_t^{(12)}$. Intermediate integration steps $h_t^{(0)}, \ldots, h_t^{(L-1)}$ are recorded as a sensitivity probe (§4.4) but are not part of the primary cell.

### 3.5 What is **not** an input

Identical to the original protocol §3.3: token IDs, token embeddings, and positional encodings are excluded from the regression input feature set. Only $h_t^{(L)}$ vectors are used.

---

## 4. Function class

### 4.1 Primary class

Identical to the original protocol §4.1 — **Gaussian RBF kernel ridge regression**, with PCA dimensionality $p = 50$ on the training fold, $\alpha$ and $\gamma$ cross-validated separately for each $k$.

The only adjustment for the SPLM cells is that the SPLM hidden state has $d = 128$, not $d = 768$ as in GPT-2 small. PCA dim $p = 50$ retains $> 99 \%$ of the variance in our smoke probes, so the same value is reused for direct comparability.

### 4.2 Capacity-matched alternatives (sensitivity sweep)

Identical to original §4.2: linear ridge, polynomial-2 ridge, kernel ridge (primary), 2-layer MLP. The same headline-rejection-requires-three-of-four rule applies.

### 4.3 PCA dimensionality sweep

We use $p \in \{20, 50, 100\}$ as in the original. Note: $p = 100$ is close to the SPLM hidden dim ($d = 128$), so this cell is essentially "no PCA"; we keep it for direct comparability.

### 4.4 Layer-axis robustness probe

In addition to the per-token $h_t^{(L)}$ regression, we run the same Markov-order test on the **per-layer** trajectory $h_t^{(\ell)}$ for $\ell \in \{1, \ldots, L\}$ at a fixed token position. This is the SPLM analogue of the Lu et al. layer-axis first-order reading mentioned but not pre-registered in the original protocol. It tests whether the SPLM's **integration-axis** dynamics (i.e., the L=8 Euler steps) are themselves first-order at the inference fixed point.

The layer-axis cell is a robustness probe; it does not enter the primary decision but is reported as part of the §9 RESULTS.md.

---

## 5. Cross-validation

Identical to the original protocol §5: leave-one-sentence-out (LOSO) outer cross-validation, nested 5-fold inner cross-validation for hyperparameter selection, per-quadruple residual recorded for paired comparison.

With 100 sentences in the primary cell, this gives 100 LOSO folds (vs. 50 in GPT-2 / Pythia primary). The 50-sentence secondary corpus retains 50 LOSO folds for direct comparability.

---

## 6. Statistical decision rule (locked at pre-registration)

The decision rule is **identical** to the original protocol §6.4. Thresholds are committed by reference to that document and are not re-derived here:

| Outcome | $\rho_{12}$ | $p_{12}$ (Wilcoxon, two-sided) | $\rho_{23}$ | $p_{23}$ | Conclusion (this protocol) |
|---|---|---|---|---|---|
| **A** | $\ge 1.20$ | $< 10^{-3}$ | $\le 1.05$ | $> 0.05$ | First-order rejected; trained SPLM is observationally second-order. **Refutes the paper §12 last paragraph.** |
| **B** | $\ge 1.20$ | $< 10^{-3}$ | $> 1.10$ | $< 0.05$ | First-order rejected; second-order also insufficient. Refutes §12 last paragraph in a *stronger* way (trained SPLM is even higher-order). |
| **C** | $< 1.10$ | OR $> 0.01$ | — | — | First-order not rejected. **Confirms the paper §12 last paragraph.** |
| **D** | otherwise | otherwise | — | — | Inconclusive; report verbatim, do not change paper rhetoric. |

The decision is taken **per checkpoint** on the primary cell (kernel ridge, $p = 50$, all 100 LOSO folds, primary corpus).

### 6.1 Per-arm aggregation rule

Each arm has $S = 3$ seeds. The arm-level verdict is determined by:

- **Unanimous C across all 3 seeds** → arm verdict is **C** (paper claim confirmed for the arm).
- **Unanimous A or B across all 3 seeds** → arm verdict is **A** or **B** correspondingly (paper claim refuted for the arm).
- **Mixed** → arm verdict is **D** (inconclusive; report verbatim, soften paper rhetoric defensively).

The headline conclusion is determined by the **SPLM em\_ln $\gamma^{\ast} = 0.30$ arm** (the paper's actual model). The SPLM-1 arm is a positive-control sanity check; if the SPLM-1 arm is **not** unanimous C, the regression pipeline is presumed buggy on SPLM checkpoints and the entire experiment is paused for debugging before the SPLM verdict is reported.

### 6.2 Required confirmations beyond the primary

Identical to original §6.5 — function-class robustness ($\ge 3$ of 4 classes agree), PCA-dim robustness (same conclusion at $p \in \{20, 50, 100\}$), corpus robustness (the secondary 5-domain corpus must yield the same arm-level verdict as the primary Shakespeare corpus).

If any required confirmation fails, the per-checkpoint conclusion is automatically downgraded one row (A → D, B → D, C unchanged).

### 6.3 Multiple-comparison correction

The full robustness sweep is now $4 \text{ classes} \times 3 \text{ PCA dims} \times 6 \text{ checkpoints} \times 2 \text{ corpora} = 144$ cells. The Bonferroni-corrected per-cell threshold for the robustness audit is $p_{12} < 10^{-3} / 144 \approx 6.9 \times 10^{-6}$. The **primary cell** is fixed in advance (kernel ridge, $p = 50$, primary Shakespeare corpus, per-checkpoint) and uses the un-corrected $10^{-3}$ threshold from §6 row A.

---

## 7. Pitfalls and how each is handled

| Pitfall | Mitigation |
|---|---|
| SPLM hidden dim is much smaller than GPT-2's ($d = 128$ vs $d = 768$); the test might be noisier. | Larger sample (100 sentences vs 50), $\ge 99 \%$ variance retained at $p = 50$, BCa cluster bootstrap with $B = 10\,000$. |
| Out-of-distribution corpus might bias residuals. | Primary cell is in-distribution Shakespeare validation; out-of-distribution 5-domain corpus is robustness only. |
| LayerNorm-after-step constrains $\lVert h_t \rVert \approx 1$, possibly trivialising or hardening the Markov-order signature. | Both SPLM em\_ln $\gamma^{\ast} = 0.30$ and SPLM-1 use the same LN-after-step projection, so any LN artefact appears in both arms equally; the comparison cancels it out. |
| SPLM-1 is structurally first-order; if it fails the Outcome C check, the test is broken. | Arm-level rule §6.1 pauses the SPLM verdict in that case. |
| Within-sentence statistical dependence | LOSO + cluster bootstrap by sentence (same as original §5, §6). |
| Researcher degrees of freedom | All thresholds locked to the original protocol; pre-registration committed before any extraction code is run on SPLM checkpoints. |
| Multiple comparisons across the 144-cell robustness grid | Bonferroni correction §6.3; primary cell fixed in advance. |

---

## 8. What this experiment is **not** claiming

- **It is not a re-test of the dynamical class of standard transformers.** That question was settled in the original protocol. This experiment tests the dynamical class of **trained SPLM** specifically.
- **It is not a calibration of the U-shape damping curve.** $\gamma^{\ast} = 0.30$ is taken as fixed from E5; we do not re-sweep $\gamma$ here.
- **It is not a test of inference *speed*.** This protocol is an empirical test of the §12 last-paragraph claim about *dynamical order at inference*. Inference-time speed advantages (which the paper's STP / convection-diffusion limit predicts) require a separate timing benchmark; we flag this as a follow-up but do not pre-register it.
- **It is not a test of generalisation.** All cells use the SPLM checkpoints' in-distribution validation split (or a proxy 5-domain corpus). The order question is invariant to whether the prediction is *correct*; it is about whether one lag suffices given the data the model actually generates.

---

## 9. Reporting plan

After the test runs, the following artefacts are committed:

- `notebooks/conservative_arch/first_order_ablation/dynamics_order/RESULTS.md` — per-checkpoint $\rho_{12}, p_{12}, \rho_{23}, p_{23}$, BCa CIs, per-arm verdict, headline conclusion.
- `notebooks/conservative_arch/first_order_ablation/dynamics_order/decision_table.md` — formatted decision table (one row per checkpoint × cell).
- `notebooks/conservative_arch/first_order_ablation/dynamics_order/figures/` — at minimum: per-arm $\bar R_k$ bar plot with per-seed error bars, paired $r_1^{(i)}$ vs $r_2^{(i)}$ scatter for the primary cell, layer-axis sensitivity plot.
- `notebooks/conservative_arch/first_order_ablation/dynamics_order/robustness_grid.csv` — full 144-cell sweep.

### 9.1 Paper consequences

| SPLM em\_ln $\gamma^{\ast}$ verdict | SPLM-1 verdict | Paper consequence |
|---|---|---|
| **C** | **C** | Paper §12 last paragraph empirically confirmed. Add a new sub-section in §15 (or §16) titled "Inference-time first-order behaviour of the trained SPLM", inlining the decision table. The headline narrative becomes: *"Train with a coherent second-order Lagrangian, get a model whose inference dynamics are indistinguishable in Markov order from standard transformers."* |
| **A** or **B** | **C** | Paper §12 last paragraph **falsified**. The trained SPLM is observationally second-order, unlike GPT-2 / Pythia. This is itself a publishable finding — it would mean SPLM is genuinely a *different* dynamical object from standard transformers, not "the same observationally with a coherent generative theory". The §12 last paragraph is rewritten as such; the abstract and intro are revised to describe SPLM as a second-order *both* generatively and observationally. |
| any | **A** or **B** | Pipeline bug. Pause publication, debug, re-run before reporting any SPLM verdict. |
| **D** (mixed seeds) | **C** | Soften §12 last paragraph: "the inference-time Markov order is consistent with first-order on a majority of seeds; the cross-seed variance suggests the observational reduction is sensitive to initialisation and warrants further study." |

---

## 10. Pre-registered authors' beliefs (separate from the decision rule)

The author of this protocol believes the most likely outcome is:

- SPLM em\_ln $\gamma^{\ast} = 0.30$: **Outcome C**, with $\rho_{12}$ in the same band as GPT-2 small ($\sim 0.98$) — possibly a touch closer to 1 because the LN-after-step projection enforces a tighter manifold structure.
- SPLM-1: **Outcome C**, with $\rho_{12}$ also in the $\sim 0.98$ band; not perfectly $1.00$ because the kernel-ridge regression has finite-sample noise even when the underlying generative process is first-order by construction.
- The two arms' $\rho_{12}$ values should overlap within the 95 % BCa CI.

This belief does not modify the decision rule — the locked thresholds of §6 stand regardless. Recording the belief here makes it a public commitment, so that any actual outcome other than the predicted Outcome C results in the explicit rewrite of the paper §12 described in §9.1, rather than an unflagged re-framing.

---

## 11. Compute estimate

| Step | Hardware | Time |
|---|---|---|
| Hidden-state extraction (6 checkpoints × 100 sentences + 6 × 50 sentences = 900 forward passes) | local MPS | ~5 min |
| Quadruple construction + PCA fitting | local CPU | ~2 min |
| Primary kernel-ridge regression with LOSO + nested 5-fold (6 checkpoints × 2 corpora × 100 LOSO folds = 1 200 fits) | local CPU | ~30–60 min |
| Robustness sweep (144 cells, lighter classes than primary) | local CPU | ~60–90 min |
| BCa cluster bootstrap ($B = 10\,000$ × 6 checkpoints × 2 corpora = 12 cells) | local CPU | ~10–20 min |
| Plot generation + RESULTS.md | local CPU | ~5 min |

**Total wall-clock: ~2–3 hours.** No GPU required after the extraction step.

---
