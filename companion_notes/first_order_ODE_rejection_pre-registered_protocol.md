# Pre-Registered Protocol — First-Order ODE Rejection Test for Transformer Hidden-State Dynamics

> Pre-registration document, drafted **April 27, 2026**, by Dimitar Gueorguiev with Claude.
> Companion to:
> *Semantic Simulation: A Prescriptive Lagrangian Framework for Efficient Semantic Inference* (Gueorguiev, 2026).
> Companion critique: [`Evidence_for_second_order_ODE_governing_evolution.md`](./Evidence_for_second_order_ODE_governing_evolution.md).

> **Status.** Pre-registered, not yet executed. This document fixes the experimental design, the analysis pipeline, and the decision rule **before** any new data extraction or regression is run. Once committed to git, any post-hoc deviation must be flagged explicitly in the eventual write-up. The committing commit hash is the timestamp of pre-registration.
>
> **Executed 2026-04-27 (same-day).** Outcome **C — first-order not
> rejected** at the primary cell on GPT-2 small ($\rho_{12}=0.979$,
> two-sided $p_{12}=0.124$), confirmed on Pythia-160m
> ($\rho_{12}=0.985$, two-sided $p_{12}=7.9\times10^{-7}$ but
> with the *opposite* direction $\bar R_1<\bar R_2$), and robust across
> the 24-cell sweep (21 / 24 cells return C; the 3 non-C cells are poly-2
> ridge over-fitting artefacts at $p\ge50$). All implementation lives
> in [`notebooks/dynamics_order_test/`](../notebooks/dynamics_order_test/);
> the headline write-up is
> [`results/RESULTS.md`](../notebooks/dynamics_order_test/results/RESULTS.md).
> No deviations were taken from this protocol; two minor implementation
> notes are recorded in
> [`notebooks/dynamics_order_test/README.md`](../notebooks/dynamics_order_test/README.md)
> (Wilcoxon two-sided per §6.2; cluster bootstrap as percentile-CI rather
> than full BCa).

---

## 1. Question

> **Update 2026-04-27 (post-outcome C) — clarification of the paper's claim.**
> At the time this protocol was pre-registered, the working assumption was that the
> paper asserted the *observational* second-order claim: that transformer hidden-state
> evolution is governed by a second-order ODE whose second-order character would
> be detectable by a Markov-order regression test. Outcome C (first-order not
> rejected) prompted a clarification of the paper's actual position, which the
> current version of the paper (v3, §12 "Generative second-order, observational
> first-order") now makes explicit.
>
> **What the current paper claims.** The framework commits to the *generative*
> second-order claim: the underlying Lagrangian is genuinely second-order — the
> EL equation $w_t\ddot{h}_t + \gamma\dot{h}_t = -\nabla V(h_t)$ contains an
> inertial term, the kinematic state is $(h_t, \dot{h}_t)$, and Theorem 49
> algebraically identifies acceleration with the STP loss. This claim is supported
> throughout the paper and is not affected by the present experiment.
>
> **What the current paper does not claim.** The paper explicitly does *not*
> assert the *observational* second-order claim — that the inertial term carries
> detectable predictive information beyond $h_t$ alone at inference. The current
> paper acknowledges that trained models inhabit the overdamped regime
> $\gamma \gg \omega_0$, where the EL equation reduces to the first-order gradient
> flow $\dot{h} \approx -\nabla V/\gamma$, making the Markov-order test unable to
> distinguish the full second-order Lagrangian from a genuinely first-order ODE.
>
> **Historical context.** The pre-registration below was drafted under the prior
> belief that the observational claim was operative. Outcome C confirmed the
> overdamped interpretation rather than refuting the second-order framework.
> The §2 hypothesis $H_1$ ("second-order suffices, but not first") was the
> claim the pre-registration was designed to test; the experiment returned $H_0$
> (Outcome C), which the current paper explains as the predicted overdamped
> reduction of the same Lagrangian.

At the time of pre-registration, the paper asserted that transformer hidden-state evolution along token position is governed (in its effective dynamics on the attractor manifold) by a **second-order** ordinary differential equation, of which the STP first-order flow (Huang et al.) and the Lu et al. convection–diffusion flow are shallow-limit specialisations. The companion document `Evidence_for_second_order_ODE_governing_evolution.md` argues for this through Theorem 49 (the STP–acceleration identity), the "promotion" argument of §12, and the §14 empirical results (≈2× tangential dominance, 97.9 % systematic deceleration, permutation null).

Those arguments establish **second-order as a sufficient and natural framing**, but — as audited internally — they do **not** rigorously exclude every first-order ODE. Specifically:

- **Theorem 49 is kinematic.** It is an algebraic identity over any three vectors and is silent on the dynamical class that produced them.
- **The 97.9 % tangential-deceleration statistic** can be reproduced by a pure first-order gradient flow $\dot{x} = -\nabla V(x)$ on a locally convex potential.

The present experiment is designed to close this gap with a **distribution-free Markov-order regression test** that asks the question directly: is $h_t$ alone a sufficient statistic for predicting $h_{t+1}$, or is at least one extra lag $h_{t-1}$ required?

---

## 2. Hypotheses

Let $\hat F_k$ denote the best predictor in a fixed function class for $h_{t+1}$ given the lag-$k$ history $(h_t, h_{t-1}, \ldots, h_{t-k+1})$. Define the per-fold mean-squared residual:

$$R_k = \mathbb{E}_{(h_{t-k+1}, \ldots, h_{t+1})}\left[\big\lVert h_{t+1} - \hat F_k(h_t, \ldots, h_{t-k+1}) \big\rVert^2\right].$$

| Hypothesis | Operational form |
|---|---|
| $H_0$ (first-order suffices) | $R_1 = R_2 = R_3$ |
| $H_1$ (second-order suffices, but not first) | $R_1 > R_2 \approx R_3$ |
| $H_2$ (higher than second order) | $R_1 > R_2 > R_3$ |

At the time of pre-registration, the paper's working claim was $H_1$. The headline rejection of *any* first-order ODE in the chosen class would have corresponded to rejecting $H_0$ in favour of $H_1$ or $H_2$.

> **Update 2026-04-27 (post-outcome C).** The experiment returned Outcome C ($H_0$
> not rejected), meaning the lag-1 representation $h_t \to h_{t+1}$ was preferred
> over the lag-2 representation $\{h_t, h_{t-1}\} \to h_{t+1}$ across all tested
> architectures and function classes. The current paper (v3, §12) no longer asserts
> $H_1$ as its claim. Instead, the paper explains Outcome C as the expected
> consequence of the overdamped regime: the full second-order EL equation reduces
> to a first-order gradient flow at inference, so a Markov-order regression at
> one-token resolution cannot distinguish the two — making $H_0$ the
> *observationally predicted* outcome of the second-order framework in the
> overdamped limit, not a refutation of it.

A first-order ODE $\dot{h} = f(h, t)$, in its discrete one-step embedding, must factorise as $h_{t+1} = F_1(h_t)$ — the next state is a function of the current state alone. Any deterministic dynamical system with this property satisfies $R_1 = R_2 = R_3$ at the level of its true conditional expectation, regardless of the form of $f$. This is what makes the test *distribution-free over $f$*: we never need to fit $f$, we only need to fit the conditional expectation of $h_{t+1}$ given various-length histories and compare them.

---

## 3. Data

### 3.1 Primary corpus (GPT-2 small)

Reuse the existing 50-sentence, 5-domain corpus from §14 of the paper, extracted in `notebooks/stp_loss/energy_landscape_validation.ipynb`. Hidden states are last-layer ($\ell = 12$) GPT-2 small activations $h_t^{(12)} \in \mathbb{R}^{768}$.

**Lag extension required.** The current pipeline emits 1,314 consecutive triplets $(h_{t-1}, h_t, h_{t+1})$. The rejection test additionally needs $h_{t-2}$ for the $k = 3$ regression. Re-run the extraction to emit *quadruples* $(h_{t-2}, h_{t-1}, h_t, h_{t+1})$ inside-sentence. Expected yield: ~1,200 quadruples (loses one position per sentence relative to the triplet count; exact number to be reported in `results/gpt2/extraction_summary.json`).

### 3.2 Replication corpus (Pythia-160M)

Same protocol, executed via `notebooks/cross_model/pythia_tangential_acceleration.ipynb` extended for an additional lag. Pythia uses rotary positional encodings and was trained on The Pile, so a result that replicates here is not a GPT-2-specific artefact.

### 3.3 What is **not** an input

Inputs to every $\hat F_k$ are **strictly hidden-state vectors**. The following are excluded from the input feature set:

- token IDs and token embeddings,
- positional encodings (raw),
- attention weights,
- any feature derived from the token identity or its surface form.

This is essential. The autoregressive transformer's per-position state $h_t^{(12)}$ is, in principle, computable from the entire prefix of token IDs, so providing token identity as input would trivialise the prediction task and would be testing token-level next-token prediction rather than the dynamical-order question. We test only the *dynamical reading* of the trajectory at a fixed layer.

---

## 4. Function class

### 4.1 Primary class — Gaussian RBF kernel ridge regression

For each $k \in \{1, 2, 3\}$:

```python
X_k = concat([h_t, h_{t-1}, ..., h_{t-k+1}])           # shape (N, k*p)
F_k = KernelRidge(kernel='rbf', alpha=alpha_k, gamma=gamma_k)
F_k.fit(X_k_train, h_{t+1}_train)
```

where the regression target is itself reduced to PCA dimension $p$ on the training fold's $h_{t+1}$ values, and the input lags are reduced to PCA dimension $p$ in the same basis.

**Why kernel ridge as the primary class.** It is a universal approximator with capacity controlled by the single regulariser $\alpha$. We **cross-validate $\alpha$ and $\gamma$ separately for each $k$** so that the first-order model receives the optimal regularisation for its input dimensionality. This neutralises the obvious "maybe a bigger first-order model would win" objection: if the best regularised universal approximator for $k = 1$ still loses materially to that for $k = 2$, the loss is not a capacity artefact.

### 4.2 Capacity-matched alternatives (sensitivity sweep)

The headline claim must be **robust across function classes**, not specific to kernel ridge. We additionally evaluate:

- Linear ridge regression (baseline / sanity check).
- Polynomial ridge, degree 2, with cross-validated $\alpha$.
- 2-layer MLP, hidden width chosen so that the parameter count of the $k = 1$ MLP matches the parameter count of the $k = 2$ MLP at the *same hidden width* (i.e., the input layer absorbs the difference in lag dimensionality). Trained with Adam, early stopping on a held-out validation fold.

Failure to reject in any one of these four classes will be reported. The pre-registered headline rejection requires success across **all four**.

### 4.3 PCA dimensionality sweep

PCA component count $p \in \{20, 50, 100\}$. The PCA basis is fitted on the *training* fold of each cross-validation split and applied to the held-out fold (no leakage). The headline rejection must hold at $p = 50$ and be qualitatively present at all three $p$ values.

---

## 5. Cross-validation

### 5.1 Outer split — leave-one-sentence-out (LOSO)

Triplets and quadruples within a single sentence share a prefix and are not independent. The resampling unit is therefore the **sentence**, not the quadruple. With 50 sentences, we run 50 outer folds, holding out one sentence's quadruples per fold and training on the other 49 sentences' quadruples.

### 5.2 Inner split — nested 5-fold for hyperparameter selection

Inside each outer fold we run a 5-fold inner cross-validation on the training sentences only, used to select $(\alpha_k, \gamma_k)$ via grid search. Search ranges:

- $\alpha \in \{10^{-3}, 10^{-2}, 10^{-1}, 1, 10\}$
- $\gamma$: median heuristic at $\{0.5, 1, 2\}$ × the median pairwise distance in the training set.

The selected hyperparameters are then refit on the full outer-training fold and evaluated on the outer-held-out fold.

### 5.3 Per-quadruple residual

For each quadruple $i$ in the outer-held-out fold, we record the squared error vector

$$r_k^{(i)} = \big\lVert h_{t+1}^{(i)} - \hat F_k\left(h_t^{(i)}, \ldots, h_{t-k+1}^{(i)}\right) \big\rVert^2.$$

These per-quadruple residuals are the unit of paired statistical comparison.

---

## 6. Statistical decision rule

### 6.1 Effect-size ratios

$$\rho_{12} = \frac{\bar R_1}{\bar R_2}, \qquad \rho_{23} = \frac{\bar R_2}{\bar R_3},$$

where $\bar R_k$ is the mean per-quadruple squared residual, averaged across all LOSO folds.

### 6.2 Paired Wilcoxon signed-rank tests

On the paired sequences $\{r_1^{(i)} - r_2^{(i)}\}_i$ and $\{r_2^{(i)} - r_3^{(i)}\}_i$, two-sided.

### 6.3 Bootstrap confidence intervals

Cluster-bootstrap by **sentence** (not by quadruple) over the 50 sentences, $B = 10{,}000$ resamples, BCa 95 % CI on $\bar R_1 - \bar R_2$ and on $\bar R_2 - \bar R_3$.

### 6.4 Decision matrix (pre-registered, locked numbers)

The decision is taken on the **GPT-2 primary** result (kernel ridge, $p = 50$, all 50 LOSO folds). The Pythia replication and the function-class sweep serve as required confirmations.

| Outcome | $\rho_{12}$ | $p_{12}$ (Wilcoxon) | $\rho_{23}$ | $p_{23}$ (Wilcoxon) | Conclusion |
|---|---|---|---|---|---|
| **A** | $\ge 1.20$ | $< 10^{-3}$ | $\le 1.05$ | $> 0.05$ | **First-order rejected; second-order sufficient.** The second-order dynamical-class claim is empirically supported. |
| **B** | $\ge 1.20$ | $< 10^{-3}$ | $> 1.10$ | $< 0.05$ | **First-order rejected; second-order *also* insufficient.** The dynamics are at least second-order; the truncation order remains to be determined. |
| **C** | $< 1.10$ | OR $> 0.01$ | — | — | **First-order not rejected.** The second-order Lagrangian provides a unifying theoretical account; rigorous dynamical-class exclusion at this temporal resolution remains open. |
| **D** | otherwise | otherwise | — | — | **Inconclusive.** The boundary case should be documented alongside a power analysis identifying what sample size would resolve the ambiguity. |

The thresholds 1.20, 1.05, 1.10, $10^{-3}$, 0.01, 0.05 are committed in this document; they are not to be adjusted after observing the data. If the headline result lands at the edge (e.g., $\rho_{12} = 1.18$), the conclusion is **D**, not **A**.

### 6.5 Required confirmations beyond the primary

The headline conclusion (A, B, or C) must additionally satisfy:

- **Architecture independence.** The same conclusion holds on Pythia-160M with the identical pipeline.
- **Function-class robustness.** The same conclusion holds for at least 3 of the 4 classes in §4.2 (linear, polynomial-2, kernel ridge, MLP). If exactly one class disagrees, this is reported but does not invalidate the headline.
- **PCA-dim robustness.** The same conclusion holds at $p \in \{20, 50, 100\}$.

If any of these required confirmations fails, the published conclusion is automatically downgraded one row in the table (A → D, B → D, C unchanged since C is already negative).

### 6.6 Multiple-comparison correction

The robustness sweep amounts to $4 \text{ classes} \times 3 \text{ PCA dims} \times 2 \text{ architectures} = 24$ cells. We Bonferroni-correct the per-cell Wilcoxon thresholds: each cell must satisfy $p_{12} < 10^{-3} / 24 \approx 4.2 \times 10^{-5}$ to count as "rejecting first-order in that cell". The headline still requires the *primary* cell at the un-corrected $10^{-3}$ threshold, but the robustness audit uses the corrected threshold.

---

## 7. Pitfalls and how each is handled

| Pitfall | Mitigation |
|---|---|
| Universal-approximator confound: "maybe a bigger first-order net would win" | Kernel ridge is universal; $\alpha$ is cross-validated; first-order receives the same hyperparameter search as second-order. |
| Token-identity leakage | Inputs are strictly hidden-state vectors; no token IDs, embeddings, or positional encodings (§3.3). |
| Within-sentence statistical dependence | Leave-one-sentence-out CV; cluster bootstrap by sentence (§5.1, §6.3). |
| Curse of dimensionality at $d = 768$ with ~1,200 examples | PCA to $p \in \{20, 50, 100\}$ on training fold only; sensitivity sweep (§4.3). |
| Architecture-specific artefact | Mandatory replication on Pythia-160M (§3.2, §6.5). |
| Researcher degrees of freedom | Pre-registration committed in git before any extraction or regression code is run; thresholds locked (§6.4). |
| Multiple comparisons across the robustness grid | Bonferroni correction over 24 cells for the robustness audit (§6.6); the primary cell is fixed in advance. |
| "Two lags is just one extra dimension" — the test might pass trivially | Required: $\rho_{12} \ge 1.20$ AND Wilcoxon $p < 10^{-3}$. A trivial dimensionality bump cannot pass both jointly given LOSO + cluster bootstrap. |

---

## 8. What we are explicitly **not** claiming

- We do not claim to identify the specific functional form of the second-order ODE.
- We do not claim the dynamics is *exactly* Markov-2 in the underlying continuous flow; the pre-registration claimed that, at the temporal-resolution probed (one token position), two lags are sufficient to predict the next state to within the noise floor of the kernel-ridge predictor. Outcome C (first-order not rejected) demonstrated that this claim does not hold at inference-fixed-point resolution, consistent with the overdamped reduction of the full EL equation.
- We do not claim the $a_\parallel < 0$ statistic of §14 is, by itself, exclusive of first-order. This experiment was intended to deliver that exclusion or to fail to do so; it returned Outcome C — first-order not excluded — and the current paper accounts for this as the predicted observational signature of the overdamped regime.
- We do not claim that rejection of first-order is rejection of *all* finite-order ODEs other than second; outcome **B** was explicitly allowed and would have itself been a publishable, framework-relevant finding.
- We do not claim our pipeline tests the Lu et al. *layer-axis* first-order reading, only the STP / position-axis first-order reading. The Lu axis would require a separate experiment (regress $h_t^{(\ell+1)}$ on $\{h_t^{(\ell)}, h_t^{(\ell-1)}\}$); this remains flagged as a follow-up.

---

## 9. Reporting plan

Independent of the outcome, the following artefacts will be produced and committed:

- `notebooks/dynamics_order_test/results/gpt2/decision_table.md` — primary decision (A/B/C/D), with $\rho_{12}, \rho_{23}$, both Wilcoxon $p$-values, both BCa CIs.
- `notebooks/dynamics_order_test/results/pythia/decision_table.md` — same, replication.
- `notebooks/dynamics_order_test/results/robustness_grid.csv` — 24-cell sweep with per-cell $\rho_{12}, p_{12}, \rho_{23}, p_{23}$.
- `notebooks/dynamics_order_test/results/figures/` — at minimum: per-class $R_k$ bar plot with error bars; $r_1^{(i)}$ vs $r_2^{(i)}$ paired scatter; LOSO-fold spaghetti plot of $\bar R_k$ across folds.
- A self-contained `RESULTS.md` summarising primary, replication, robustness, decision row, and any deviations from this protocol with explicit justification.

**Outcome A** was not realised. Had it been, the result would have constituted quantitative empirical support for the observational second-order claim, with the decision table and protocol reference forming the complete documentation of the test.

**Outcome C was returned (executed 2026-04-27).** The second-order Lagrangian provides the unifying theoretical account for the generative dynamics, while the observational question of whether some first-order ODE of arbitrary form could match the observed trajectories at this temporal resolution remains empirically open. The current paper explains this as the predicted consequence of the overdamped regime rather than as a refutation of the framework. The result and its theoretical synthesis are documented in `Evidence_for_second_order_ODE_governing_evolution.md`, in [`notebooks/dynamics_order_test/results/RESULTS.md`](../notebooks/dynamics_order_test/results/RESULTS.md), and in the paper's §12 paragraph "Generative second-order, observational first-order."

---

## 10. Compute estimate

| Step | Hardware | Time |
|---|---|---|
| Lag-extension extraction (GPT-2) | MPS, batch eval | ≤ 10 min |
| Lag-extension extraction (Pythia) | MPS, batch eval | ≤ 10 min |
| Kernel ridge primary, 50 LOSO folds | CPU | ≤ 5 min |
| Robustness grid (24 cells × 50 folds) | CPU | ≤ 60 min |
| Bootstrap (10,000 resamples) | CPU | ≤ 5 min |
| Plotting + report | CPU | ≤ 30 min |

**Total wall-clock: ≤ 2 h.** Modest enough that there is no resource excuse to defer.

---

## 11. Implementation plan

```
notebooks/dynamics_order_test/
├── README.md                            # how to reproduce
├── extract_lagged_quadruples.py         # GPT-2 + Pythia 4-tuples
├── markov_order_regression.py           # F_1, F_2, F_3 with LOSO + nested CV
├── robustness_sweep.py                  # 24-cell grid (4 classes × 3 PCA × 2 arch)
├── statistical_tests.py                 # Wilcoxon + cluster bootstrap
├── plots.py                             # R_k bar charts, paired scatters, fold spaghetti
├── results/
│   ├── gpt2/
│   │   ├── extraction_summary.json
│   │   ├── decision_table.md
│   │   └── figures/
│   ├── pythia/
│   │   ├── extraction_summary.json
│   │   ├── decision_table.md
│   │   └── figures/
│   ├── robustness_grid.csv
│   └── RESULTS.md
└── scripts/
    └── run_all.sh
```

The directory `notebooks/dynamics_order_test/` is **created empty** at the time this protocol is committed; no implementation files exist yet. This is an explicit invariant of the pre-registration: implementation begins after this document is in `git log`.

---

## 12. Lock-in

| Field | Value |
|---|---|
| Pre-registration date | 2026-04-27 |
| Pre-registered by | Dimitar Gueorguiev (with Claude) |
| Companion document | [`Evidence_for_second_order_ODE_governing_evolution.md`](./Evidence_for_second_order_ODE_governing_evolution.md) |
| Paper version this targets | v3 (Zenodo DOI [10.5281/zenodo.19819861](https://doi.org/10.5281/zenodo.19819861)) |
| Status | **Executed 2026-04-27.** Outcome **C — first-order not rejected.** |
| Execution gating | E3 production complete; ran on 16-core CPU after Pythia and GPT-2 extraction on MPS. |
| Lock-in commit | filled by the commit creating this file (in `git log` of `docs/first_order_ODE_rejection_pre-registered_protocol.md`) |
| Result write-up | [`notebooks/dynamics_order_test/results/RESULTS.md`](../notebooks/dynamics_order_test/results/RESULTS.md) |

Any deviation from the protocol after the lock-in commit must be:

1. Documented in a `DEVIATIONS.md` file in `notebooks/dynamics_order_test/`.
2. Justified explicitly in the eventual `RESULTS.md`.
3. Reflected in any paper text citing this experiment.

A rerun under a *different* protocol is not a deviation — it is a separate experiment, and must have its own pre-registration document.
