# Side probes: CG8, DP, F0, RF, MT and BF

**What this note is.** A readable summary of six small pre-registered programmes in §5 of the depth-ladder [protocol](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md), written 2026-11-17. Each asks one narrow question beside the main ladder runs: §5.12 CG8, §5.14 the DP series, §5.16 F0, §5.23 RF, §5.24 MT and §5.26 BF. For each, the note gives the question, the method, the calls with their stated probabilities, the result and verdict as recorded, and the status. Every number, verdict and date is copied from the protocol, which stays the record of truth.

**Status in one line.** DP, F0, RF and MT were run and scored (2026-10-05, 2026-10-06, 2026-10-12, 2026-10-12); CG8 has no recorded result; BF is a draft that was deferred on 2026-10-18 and never run.

**Contents**

1. [§5.12 CG8: does the geometry predict the model's own errors?](#cg8)
2. [§5.14 DP series: what Doi–Peliti process do the trained registers implement?](#dp)
3. [§5.16 F0: is there a shared floor near 50 PPL?](#f0)
4. [§5.23 RF: do the Poisson-mode registers carry forward-looking state?](#rf)
5. [§5.24 MT: do the Poisson modes have transitions?](#mt)
6. [§5.26 BF: selective bf16 training (draft, not binding)](#bf)
7. [Scorecard](#scorecard)

**Terms used throughout.**

- A **call** is a prediction written down before the measurement, with the probability the author gave it. **HIT** and **MISS** are the protocol's scores.
- PPL is validation perplexity; lower is better. **Settled** PPL is the mean of the last three evaluations.
- The arms named here, as the protocol uses them: **G2** is the $L = 2$ Fock arm ("live"); **G3** is $L = 2$ with the exchange field; **G3′** is a pre-registered follow-up built on G3's configuration; **L=4 live** is the four-layer Fock arm; **F3.1** is the model trained without the register path; **PM1** and **PMX** are the two Poisson-mode arms, PM1 with explicitly integrated wells and PMX with exact integration (see [Poisson_Mode_Wells_PM1_PMX.md](Poisson_Mode_Wells_PM1_PMX.md)).
- The scripts named below are in [`notebooks/conservative_arch/scaleup/debug/`](../../notebooks/conservative_arch/scaleup/debug/).

---

<a name="cg8"></a>
## §5.12 CG8: does the geometry predict the model's own errors?

**Pre-registered 2026-10-03, before any measurement.**

**Question.** Book §18d (`subsec:geom-requirements`) separates three properties: a conservative step (C), refinement invariance (R) and settling (S). A geometric capability belongs to the model only if the properties it reads are present. The SR1–SR4 arms target (R) and (S); CB1–CB3 target (C). CG8 asks whether securing them is worth anything: does the geometry of the trajectory carry information about the model's own errors that the output distribution does not? It is a token-level stand-in for the hallucination detector of §18d Experiment G2, and it needs evaluation only.

**Method** (a new cell, planned as 6b-14).

- **Tokens:** $32 \times 512$ validation tokens, with one fixed seed shared by every arm.
- **Event:** the top-1 prediction at token $t$ is wrong. A secondary event: the loss at $t$ is above the arm's median.
- **Signals per token** (summed over layers unless stated):
  - $s_E$, the energy anomaly $\lvert \Delta E_{\mathrm{obs}} - \Delta E_{\mathrm{expected}} \rvert$, with $H = \tfrac{1}{2} m \lVert v \rVert^2 + V_\theta(\xi, h)$ (plus $V_\varphi$ when present) and $\Delta E_{\mathrm{expected}} = \gamma \lVert v \rVert^2 \Delta t$;
  - $s_G$, how far the full step deflects from the damped $V_\theta$ geodesic step (as in 6b-12);
  - $s_\eta$, Fock arms only: the ratio of the reverse-channel increment's norm to the conservative step's norm, maximum over layers;
  - the reference: the softmax entropy $H_{\mathrm{soft}}$ at $t$.
- **Score:** the gain in AUROC (the area under the error-detection curve) from adding a signal to the entropy, $\Delta\mathrm{AUROC}(s) = \mathrm{AUROC}(H_{\mathrm{soft}}, s) - \mathrm{AUROC}(H_{\mathrm{soft}})$, both by logistic regression. Five-fold cross-validation over sequences (not tokens), with a 1,000-sample bootstrap interval.
- **Arms:** the baselines F3.1, G2 and L=4 live first, then SR1–SR4 against F3.1 and CB1–CB3 against G2.

**Calls.**

| arm(s) | signal | prediction | called |
| --- | --- | --- | --- |
| baselines (F3.1, G2, L=4 live) | s_E, s_G | ΔAUROC < 0.01: at the trained step size the geometry adds nothing to entropy | 60% |
| G2, L=4 live | s_η | ΔAUROC < 0.01 | 55% |
| SR2 (exact damped flow) vs F3.1 | s_E | ΔAUROC higher by ≥ 0.01, because the phase noise of Prop 44 is removed | 35% |
| SR4a (variable N at fixed T) vs F3.1 | s_G | ΔAUROC ≥ 0.02 | 35% |
| CB2b (ρ = 0.3) vs G2 | s_G | ΔAUROC higher by ≥ 0.01 | 40% |
| CB3 | s_G on tokens with g = 0 vs g > 0 | AUROC(s_G) higher on the exactly-conservative tokens | 50% |
| SR5 (added in the SR5 pre-registration) | s_G | ΔAUROC < 0.01 | 60% |

**Decision rule.** Any arm with a gain of at least 0.02 (interval excluding 0) for a geometry signal is the first evidence that the Lagrangian geometry carries information the output distribution does not; the property that arm holds is the one worth paying for. If the baselines already reach 0.02, the geometry is informative even at the trained step size, and (C), (R) and (S) matter for interpretation only. If no arm reaches 0.01, the §18d capability claims remain unshown on the CfC/BAOAB family, and the book says so in §18d and in the abstract's scope.

**Caveat recorded in advance.** Next-token top-1 errors on OpenWebText are mostly ambiguity, not hallucination. CG8 tests whether geometry tracks error at all, not factuality. Factuality needs the QA protocol of §18d Experiment G2.

**Result.** No result is recorded. The protocol contains no 6b-14 run and no scoring of any CG8 call.

**Status.** Pre-registered, not run (as far as the protocol records).

<a name="dp"></a>
## §5.14 DP series: what Doi–Peliti process do the trained registers implement?

**Pre-registered 2026-10-05, before any measurement. Scored 2026-10-05.**

**Question.** Book §10.5.2 makes the Fock registers a bosonic Doi–Peliti system (a field theory of particles that are created and destroyed), and says "a continuous salience variable plays the role of a Poisson mean." The code shows four facts that bear on that reading, stated before any measurement:

1. **Time is depth.** Salience is updated once per layer, separately at each position.
2. **The process starts full.** Every register starts at salience 1, not empty. With decay 0.5, every register is necessarily active at layer 0.
3. **Salience is bounded and deterministic.** Each update is the old salience times 0.5 plus the peak creation weight times 0.5, then multiplied by the destruction factor $(1 - g)$. Salience stays in $[0, 1]$ and has no fluctuation.
4. **The force sees a yes/no mask, not the salience.** The reverse channel receives only the mask (salience above 0.005, with last-in-first-out stack discipline). Otherwise salience only weights how much of a register's old content is kept.

Each register is one slot holding one content vector: an exclusion (hard-core) object. A bosonic Poisson reading would put up to 26% of a register's probability on double occupancy, which one slot cannot represent. The companion note is [Doi_Peliti_Dynamics_of_Semantic_Particles_and_Registers.md](../Doi_Peliti_Dynamics_of_Semantic_Particles_and_Registers.md).

**Method.** Evaluation only, on the local CPU, with [`dp_register_statistics.py`](../../notebooks/conservative_arch/scaleup/debug/dp_register_statistics.py). $8 \times 512$ validation tokens, seed 20261005. Arms G2, L=4 live and G3, each on its best checkpoint. Hooks record, for every layer, position and register: the salience that sets that layer's mask, the destruction gate $g$, the mask, and the content passed to the reverse channel.

- **DP1, is particle number dynamic at all?** From layer 1 up, the active fraction (active registers out of $M = 32$), the fraction of cells below threshold, and the distribution of $g$.
- **DP2, do registers share content?** Among active registers, the salience-weighted mean absolute cosine between their contents, and the fraction of pairs with cosine above 0.9 (near-duplicates). Earlier positions and the last position are reported apart, because the 0.05 repulsion penalty acts in training only on the last.
- **DP3, does salience act as intensity?** The Spearman rank correlation between each active register's salience and its leave-one-out contribution to the reverse-channel force.

**The three statistics, written out.** The wording above is the intent; this is what the script computes, so that a reader can reproduce the numbers rather than approximate them.

At layer $\ell$, in sequence $b$ at position $t$, let $r_k \in \mathbb{R}^d$ be the content vector of register $k$ and $\sigma_k \in [0,1]$ its salience. The book defines a register as *active* by a threshold test on salience (§20, p. 296), which this note writes as an indicator, since neither the book nor the code gives it a symbol:

$$
a_k = \mathbf{1}\lbrack \sigma_k \gt \tau \rbrack, \qquad \tau = 0.005.
$$

**DP1** is then the mean of that indicator over the $M = 32$ registers, reported per layer and position,

$$
A(b,t) = \frac{1}{M}\sum_{k=1}^{M} a_k .
$$

**DP2** normalises the contents, forms the pairwise cosine matrix, and weights each pair by the product of the two masked saliences:

$$
\hat{r}_k = \frac{r_k}{\lVert r_k\rVert}, \qquad C_{kj} = \hat{r}_k \cdot \hat{r}_j, \qquad W_{kj} = (\sigma_k a_k)(\sigma_j a_j).
$$

Writing $\mathcal{P} = \lbrace (k,j) : k \neq j \rbrace$ for the ordered pairs of distinct registers, the first statistic is the weighted mean of $\lvert C_{kj}\rvert$ over a set $S$ of positions. Numerator and denominator are pooled across $S$ before dividing, rather than averaging per-position ratios:

$$
\bar{c}(S) = \frac{\sum_{(b,t)\in S} \sum_{(k,j)\in\mathcal{P}} W_{kj}^{(b,t)} \lvert C_{kj}^{(b,t)}\rvert}{\sum_{(b,t)\in S} \sum_{(k,j)\in\mathcal{P}} W_{kj}^{(b,t)}} .
$$

The mask enters only through the weights: an inactive register contributes zero to both sums, so pairs are excluded by weight rather than filtered out, and the result is a genuine weighted mean over active pairs. The absolute value matters because two registers holding opposite content span the same direction, and without it aligned and anti-aligned pairs would cancel and hide duplication.

With $L = \lbrace (b,t) : t = T-1 \rbrace$ the last position of each sequence and $E$ its complement, the reported quantity is the difference

$$
\Delta_\ell = \bar{c}(E) - \bar{c}(L),
$$

which measures how much of the registers' distinctness is owed to the repulsion penalty, since that penalty acts in training only on $L$. The second statistic counts near-duplicates instead of weighting them, and uses the signed cosine, because two slots holding opposite content are not duplicates:

$$
\mathrm{dup}(S) = \frac{\lvert \lbrace (b,t,k,j) \in S\times\mathcal{P} : C_{kj} \gt 0.9, a_k a_j = 1 \rbrace\rvert}{\lvert \lbrace (b,t,k,j) \in S\times\mathcal{P} : a_k a_j = 1 \rbrace\rvert} .
$$

**DP3** removes one register at a time from the mask and measures how far the reverse-channel force moves. With $Q(a)$ the force the channel returns under mask $a$, and $a^{(-k)}$ that mask with register $k$ switched off,

$$
\delta_k = \lVert Q(a) - Q(a^{(-k)}) \rVert ,
$$

and the reported number is the Spearman rank correlation $\rho(\sigma, \delta)$ taken across the active registers at one position, then averaged over positions. Positions with fewer than four active registers are skipped. A positive $\rho$ would mean that a more salient register contributes more force, which is what an intensity reading predicts.

**Calls.**

| test | arm(s) | prediction | called |
| --- | --- | --- | --- |
| DP1 | G2, G3 | active fraction ≥ 0.95 at layer 1: no number dynamics in practice | 65% |
| DP1 | L=4 live | active fraction ≥ 0.95 at every layer from 1 to 3 | 55% |
| DP2 | all | near-duplicate pairs (cosine above 0.9) under 1% of active pairs, at every position | 55% |
| DP2 | all | mean absolute cosine at earlier positions above the last position's by at least 0.05 | 50% |
| DP3 | all | Spearman ρ below 0.3: salience does not act as intensity | 60% |

**Decision rule.** If DP1 holds, the trained models implement no particle-number dynamics, and the Fock/Doi–Peliti language describes the architecture's *capacity*, not its trained behaviour. If DP3 is weak (Spearman below 0.3), "salience plays the role of a Poisson mean" is withdrawn and salience is read as the retention probability of a hard-core slot; if strong (at least 0.5), the sentence stays with a qualification. If near-duplicates are common (at least 5%), registers are exclusive per slot but bosonic in content, and the book says that.

**Caveat recorded in advance.** Salience is a deterministic mean-field quantity, so no measurement on these models can tell bosonic from exclusion fluctuations: there are none. The tests read how the models *use* the variables.

**Result** (output `debug/dp_register_statistics_output.txt` and its JSON; figure [dp_register_diagnostics.png](../figures/doi_peliti/dp_register_diagnostics.png)). Layer checkpointing was off; the logits are bit-identical with and without it.

| test | G2 | L=4 live | G3 | prediction | outcome |
| --- | --- | --- | --- | --- | --- |
| DP1, active fraction from layer 1 | 0.9995 | 1.0000 (layers 1–3) | 1.0000 | at least 0.95 (65%, 55%) | **hit** |
| DP2, near-duplicates at earlier positions | 1.30% | 0.66–0.94% | 0.98% | under 1% everywhere (55%) | **miss**, narrowly (G2) |
| DP2, absolute cosine earlier minus last | 0.037 | 0.054–0.070 | 0.051 | at least 0.05 everywhere (50%) | **miss** (G2); hit elsewhere from layer 1 |
| DP3, Spearman(salience, own force) | −0.005, −0.061 | −0.022, −0.321, −0.144, −0.407 | −0.022, −0.172 | below 0.3 (60%) | **hit**, and negative |

**Verdict.** DP1 holds: the trained models implement no particle-number dynamics. DP3 is weak, in fact negative: "salience plays the role of a Poisson mean" is withdrawn, and salience is read as the retention probability of a hard-core slot. DP2 is far below the 5% line, so registers do not share content and the hybrid reading does not apply either. DP1 was nearly forced. Where the mask is taken, salience is at least $(1 - \lambda)$ divided by the number of prefix positions. At decay 0.5 and threshold 0.005, no register can be inactive in the first 99 positions.

**Found without pre-registration.**

- The destruction gate is switch-like at layer 0 (G2 median 0.994, 10th percentile 0.016). It acts as a content reset.
- The last layer's destruction gate receives exactly zero gradient in every model (its output is never read) and stays at its initialisation, 0.46–0.54.
- The median weight of the initial salience after the last layer is 0.0007 (G2), 0.0001 (L=4) and 0.0027 (G3).
- DP3's negative sign has a mechanism: freshly reset registers carry the current context and dominate the force.

**Later uses.** The DP result is why PM1 was built (§5.15). DP3 was rerun on PM1's modes as a sanity check (called 80%) and scored **HIT**, +0.815. A later refinement test reports G3′'s DP3 correlation at layer 1 as $-0.63$, stronger than G2's (see [Refinement_Readiness_RR.md](Refinement_Readiness_RR.md)). A different, later probe also uses the labels DP1 to DP3 (the 2026-11-16 damping probe in [Second_Order_Memory_Probes.md](Second_Order_Memory_Probes.md)); it is unrelated.

**Status.** Scored 2026-10-05. The book changes (§10.5.2 and §20) are listed in the companion note's §8.1, for the next edition.

<a name="f0"></a>
## §5.16 F0: is there a shared floor near 50 PPL? The stable-phase extrapolation

**Pre-registered 2026-10-06, before any fit. Scored 2026-10-06.**

**Question.** Among the $L = 2$ models with the register path, G2 (53.12), G3 (54.21) and G3′ (52.90) settle within 2.5% of each other. L=4 Fock (50.10) and the 8-layer matched GPT-2 (49.81) end near 50. Is the floor set by depth, or by what every model shares: width $d = 384$, the untied head and 532M tokens?

**Method** ([`f0_floor_fit.py`](../../notebooks/conservative_arch/scaleup/debug/f0_floor_fit.py), CPU).

Fit $L(t) = L_\infty + A t^{-\alpha}$, in validation loss (nats), to each model's evaluations in the constant-learning-rate stable phase: steps 3,000 to 21,000, every 500 steps, ending before the WSD decay at 21,125. (WSD is the warmup, stable, decay schedule.) The models are F3.1, G2, G3, G3′ and L=4 Fock, which share one schedule, learning rate, batches and token budget. A 90% interval on $L_\infty$ comes from 2,000 residual-bootstrap refits; a fit is **unidentified** if that interval is wider than 0.3 nats, or if $\alpha$ hits a bound. The matched GPT-2 used a cosine schedule with no constant-rate phase, so it is reported descriptively only. These are constant-rate asymptotes, not decayed ones; the comparison is between models.

**Calls.**

| | prediction | called |
| --- | --- | --- |
| P1 | G2, G3 and G3′ have L∞ within 0.05 nats (about 5% in PPL) of each other | 65% |
| P2 | the L=2 Fock models' L∞ are above L=4's by more than their combined 90% intervals (a depth-set floor) | 45% |
| P3 | F3.1's L∞ is above the L=2 Fock models' by more than the intervals | 60% |
| P4 | at least one fit is unidentified (the window is short and the eval noise is about ±2 PPL) | 50% |

**Decision rule.** P2 holds: the floor is set by depth, and F1 (the doubled token budget) asks whether it moves with data. P2 fails with identified fits: the floor is shared, and F1 and F2 test which cause. Fits unidentified: F0 is inconclusive, and F1 is the test.

**Result** (37 evaluations per model).

| model | L∞ (nats) | 90% interval | PPL∞ | PPL interval | α | identified | PPL at step 21,000 |
| --- | ---: | --- | ---: | --- | ---: | --- | ---: |
| F3.1 | 3.256 | [2.209, 3.636] | 25.9 | [9.1, 37.9] | 0.24 | no | 71.81 |
| G2 | 3.339 | [2.965, 3.549] | 28.2 | [19.4, 34.8] | 0.33 | no | 63.61 |
| G3 | 3.898 | [3.805, 3.967] | 49.3 | [44.9, 52.8] | 0.61 | yes | 65.22 |
| G3′ | 3.875 | [3.772, 3.944] | 48.2 | [43.5, 51.6] | 0.62 | yes | 63.45 |
| L=4 Fock | 3.763 | [3.661, 3.839] | 43.1 | [38.9, 46.5] | 0.60 | yes | 60.80 |
| GPT-2, cosine (descriptive only) | 3.557 | [3.434, 3.647] | 35.0 | [31.0, 38.4] | 0.45 | yes | 56.35 |

| call | outcome |
| --- | --- |
| P1 | MISS: spread 0.56 nats, driven by G2's unidentified fit; G3 and G3′ agree within 0.023 |
| P2 | MISS: the identified L=2 intervals overlap L=4's, narrowly (G3's lower bound 3.805 against L=4's upper bound 3.839) |
| P3 | MISS: F3.1's fit is unidentified |
| P4 | HIT: F3.1's and G2's intervals are 1.4 and 0.58 nats wide |

**Verdict: inconclusive, and F1 is the test.**

- **Only a relative reading holds.** G3 and G3′ extrapolate to a constant-rate asymptote about 11% above L=4's (about 48–49 against about 43 PPL), with intervals that just overlap. That hints at a depth dependence. It is not a floor.
- **No absolute floor for $L = 2$ is supported**, for four reasons (corrected 2026-10-06 after an overstatement in discussion): only G3 and G3′ are identified; $L_\infty$ is the constant-rate asymptote and the final decay goes well below it (G3′ went from 63.45 at step 21,000 to 52.90 settled); extrapolating a three-parameter fit far beyond its sixfold window is unreliable; and a much longer run would be re-tuned and reach a regime these runs never see.

**Follow-up in W1 (§5.17, scored 2026-10-06).** The same fit on the matched GPT-2 retrained on the WSD schedule was identified (call W1.3, 75%: HIT), with an asymptote of 4.038 nats (56.74 PPL). Yet that run settled at 48.82, 0.15 nats (8 PPL) *below* its own stable-phase asymptote. The protocol concludes that **a stable-phase $L_\infty$ is not a floor**, confirming point 2 of F0's correction, and that it is unfit for comparing architectures trained at different peak learning rates. See [Matched_GPT2_Baselines.md](Matched_GPT2_Baselines.md).

**Status.** Scored 2026-10-06; inconclusive by its own rule; F1 named as the real test.

<a name="rf"></a>
## §5.23 RF: do the Poisson-mode registers carry forward-looking state?

**Pre-registered 2026-10-12, before any measurement. Scored 2026-10-12.**

**Question.** The note [Chain_of_Thought_Substrate_in_Lagrangian_Models.md](../Chain_of_Thought_Substrate_in_Lagrangian_Models.md) finds that the Poisson-mode registers are the conservative scratchpad a native chain of thought would write to, and that the missing piece is a training signal. Before any such training, RF asks: do the registers of the published PMX model already carry state about what comes next, beyond what the hidden state carries? If not, the scratchpad is a memory of the past only.

**Method** ([`register_forward_probe.py`](../../notebooks/conservative_arch/scaleup/debug/register_forward_probe.py), CPU, about an hour per arm). 200 validation sequences of 512 tokens, seed 20261012. At each position: the per-token loss, the layer-1 occupation vector $\bar{n}(t)$ (64 numbers, one per mode) and the layer-1 hidden state $h_t$ (384). The target is the mean loss over the next $w$ tokens, $w \in \lbrace 1, 4, 16 \rbrace$. Ridge-regression probes, scored by $R^2$ on 50 held-out sequences, with bootstrap intervals over sequences.

The probes are $\bar{n}$ alone, $h$ alone, and both together. The controls are $\bar{n}$ with modes shuffled across positions, position alone, and $h$ cut to 64 principal components (matched in size to $\bar{n}$). The primary statistic is $\Delta R^2 = R^2(\bar{n}, h) - R^2(h)$ at $w = 16$. Arms: PMX (primary), PM1, and G2's slot registers as comparison.

**Calls.**

| id | prediction | called at |
| --- | --- | ---: |
| RF-a | n̄ alone beats the shuffle and position at every w (intervals disjoint) | 80% |
| RF-b | ΔR² ≥ 0.02 at w = 16 | 45% |
| RF-c | ΔR² increases from w = 1 to w = 16 | 40% |
| RF-d | the capacity-matched h (64 PCs) beats n̄ alone at w = 16 | 60% |

**Reading rule.** RF-b holding: the registers carry forward-looking state the hidden state lacks, the precondition for a scratchpad. RF-b failing with RF-a holding: a memory of the past, not a working memory.

**Result.** Both exactness checks pass on every arm.

| R² on held-out sequences, pooled | PMX w=1 | w=4 | w=16 | PM1 w=1 | w=4 | w=16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| occupation n̄ | 0.003 | 0.007 | 0.024 | 0.001 | −0.002 | −0.017 |
| hidden state h | 0.036 | 0.017 | 0.023 | 0.037 | 0.024 | 0.013 |
| n̄ + h | 0.035 | 0.015 | 0.032 | 0.035 | 0.016 | −0.005 |
| n̄ shuffled | 0.000 | −0.002 | −0.006 | 0.000 | 0.000 | 0.001 |
| h, 64 PCs | 0.031 | 0.017 | 0.012 | 0.031 | 0.017 | 0.010 |

| call | PMX | PM1 |
| --- | --- | --- |
| RF-a (80%) | MISS: beats the shuffle at w = 4 and 16 ([+0.003, +0.021] and [+0.008, +0.064]) but not at w = 1, and never position by a disjoint interval | MISS: beats neither at any w |
| RF-b (45%) | MISS: +0.011 [−0.006, +0.028] | MISS: −0.028 [−0.056, −0.002] |
| RF-c (40%) | HIT: −0.001 → +0.011 | MISS |
| RF-d (60%) | MISS: [−0.044, +0.013] | MISS: [−0.000, +0.070] |

**PMX's occupation carries a small, real signal about the next 16 tokens' loss:** $R^2$ 0.024 alone, equal to the full hidden state's 0.023. It adds +0.011 to the hidden state, below the pre-registered 0.02, with an interval crossing 0. **PM1's occupation carries none** at any horizon. Both signals are weak: the best $R^2$ anywhere is 0.04. **The G2 probe is invalid and not scored.** Its register bank is 12,288 features against 125 training sequences; the probe overfits to $R^2$ of $-1$ to $-4$. This is not a result about G2's registers.

**Follow-up, content or difficulty?** Adding the mean loss over the *previous* 16 tokens (past16) as a competing feature, for PMX:

| R², w = 16 | past16 | n̄ | n̄ + past16 | h + past16 | n̄ + h + past16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PMX | 0.022 | 0.024 | 0.051 | 0.045 | 0.052 |

The occupation's increment over $h$ plus past16 is +0.004 to +0.014 (95% interval, disjoint from 0). The occupation and past16 are nearly additive. So the signal is about **what** the recent context was, not how predictable it was.

**Verdict, revised.** RF-b misses: the strong precondition fails at the pre-registered size. The weak precondition holds, for PMX only: a small, real, content-bearing signal about what comes next. The registers are a memory of the past with a faint forward component. The scratchpad design is not refuted, since nothing has been trained to write to it.

**Status.** Scored 2026-10-12, with the follow-up.

<a name="mt"></a>
## §5.24 MT: do the Poisson modes have transitions?

**Pre-registered 2026-10-12, before any measurement. Scored 2026-10-12.**

**Question.** The note [Community_of_Semantic_Templates_for_Native_CoT.md](../Community_of_Semantic_Templates_for_Native_CoT.md) reads each Poisson mode as a learned pattern-matching region (a *template*) and its occupation as a firing history. A template bank that carries a chain of thought needs *transitions*: which template fires next, given which fired. RF asked whether the occupation predicts future loss; MT asks whether it predicts future firings.

**Method** ([`mode_transition_probe.py`](../../notebooks/conservative_arch/scaleup/debug/mode_transition_probe.py), CPU, the same capture as RF).

A mode $v$ **fires** on token $t$ if its overlap $E_v(t) = \exp(-\kappa_v^2 \lVert h_t - \mu_v \rVert^2)$ exceeds a threshold set at its 90th percentile, so every mode fires on 10% of tokens. For lags $\Delta \in \lbrace 1, 2, 4, 8 \rbrace$ the probe builds the $64 \times 64$ matrix of the chance that $v$ fires at $t + \Delta$ given that $u$ fired at $t$, and its **lift** over $v$'s base rate.

- **The drift null.** Neighbouring states are similar, so firings co-occur with no transition structure. The null replaces $h_{t+\Delta}$ by its best linear prediction from $h_t$: the transitions a smooth drift would produce.
- **Primary statistic:** the mutual information $I_\Delta$ (bits) between "$u$ fires at $t$" and "$v$ fires at $t + \Delta$", and its **excess over the drift null**. Secondary: whether top transitions are **asymmetric**, which a chain needs and a similarity structure lacks.

**Calls.**

| id | prediction | called at |
| --- | --- | ---: |
| MT-a | I₁ exceeds the shuffle null (modes permuted within each sequence) | 90% |
| MT-b | the excess ΔI₁ over the drift null is positive, interval disjoint from 0 | 55% |
| MT-c | ΔI_Δ decays with Δ but is still positive at Δ = 4 | 40% |
| MT-d | at least 20 of the 64 × 63 ordered pairs have asymmetric lift (L(u→v) ≥ 2 L(v→u), both beyond the drift null) | 35% |

**Result.** The exactness check passes on both arms. Firing rate per mode on the test split 0.099; no mode silent.

| summed MI, bits | PMX Δ=1 | Δ=2 | Δ=4 | Δ=8 | PM1 Δ=1 | Δ=2 | Δ=4 | Δ=8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| data | 1.646 | 0.779 | 0.458 | 0.354 | 3.369 | 2.362 | 0.832 | 0.407 |
| drift null (64 PCs) | 1.274 | 0.588 | 0.375 | 0.329 | 2.888 | 2.282 | 0.639 | 0.314 |
| shuffle null | 0.550 | 0.231 | 0.191 | 0.171 | 1.279 | 0.919 | 0.435 | 0.228 |
| excess over drift | +0.373 | +0.191 | +0.083 | +0.025 | +0.482 | +0.079 | +0.192 | +0.093 |
| 95% CI | [+0.31, +0.57] | [+0.13, +0.36] | [+0.04, +0.23] | [−0.01, +0.14] | [+0.17, +0.96] | [−0.22, +0.54] | [+0.07, +0.42] | [+0.04, +0.27] |

| call | PMX | PM1 |
| --- | --- | --- |
| MT-a (90%) | HIT, 3.0× | HIT, 2.6× |
| MT-b (55%) | HIT, +0.37 [+0.31, +0.57] | HIT, +0.48 [+0.17, +0.96] |
| MT-c (40%) | HIT, +0.08 [+0.04, +0.23] | HIT, +0.19 [+0.07, +0.42] |
| MT-d (35%) | MISS, 0 (no off-diagonal pair reaches lift 2; top lift 1.9, matched by the null's 1.7) | MISS, 3 (16 pairs at lift ≥ 2, 7 beyond the null, 3 asymmetric) |

**Verdict.** **The modes have transitions beyond the state's drift**: about 0.37 bits at lag 1 in PMX (a quarter of the total), decaying over 4–8 tokens. **But the transitions are symmetric:** "$u$ then $v$" is as likely as "$v$ then $u$". That is co-occurrence, which a similarity structure gives, not sequence, which a chain needs.

- PM1's transitions are stronger and more concentrated than PMX's (16 strong pairs against 0; summed MI 3.4 against 1.6 bits). Neither direction of this bears on order.
- The weak form of the author's claim (a bank of learned regions with transition structure) is supported; the strong form (directed transitions) is not.
- A reported-only column printed `inf` at lag 1 (a saturated fit); the probe now clips probabilities, and the recorded calls are unaffected.

**Status.** Scored 2026-10-12.

<a name="bf"></a>
## §5.26 BF: selective bf16 training (draft, not binding)

**Drafted 2026-10-18 for the author's review; never made binding. Deferred the same day.**

**Question.** Can the 2B-token runs train with selective bf16 (16-bit "brain float" arithmetic in parts of the forward pass, Cell 0 `AMP_BF16`) without changing what the model learns? Those runs take about 60 h per arm on the A100 (PMX about 100 h), and bf16 was expected to be the largest single speed-up.

**What bf16 covers.** The well generator for $V_\theta$, the context channels, the output head and the Fock gates run in bf16. The layer step from the linearisation on (eigensolves, exact substeps, force assembly, the LayerNorm projection) stays in float32. Evaluation is float32. Checked on the CPU with [`verify_bf16_switch.py`](../../notebooks/conservative_arch/scaleup/debug/verify_bf16_switch.py): with autocast off, bit-identical to HEAD.

**Method (planned).** SR2's Cell 0 plus `AMP_BF16 = True`, `PROBE_MAX_STEPS = 8_000`, seed 0, about 2.5–4 h. The reference is SR2's float32 run (92.93 at step 8,000). Measure evaluations, gradient norms, spikes, seconds per step and peak memory.

**Calls (draft).**

| | prediction | called |
| --- | --- | --- |
| BF1 | bf16 tracks float32: the mean of ln(PPL_bf16 / PPL_fp32) over the evals at steps 4,000–8,000 lies within ±0.02 (±2%) | 70% |
| BF2 | at step 8,000 the ratio lies within ±3% | 70% |
| BF3 | no watchdog trigger and no spike capture | 85% |
| BF4 | at least 1.3× faster per training step than float32 | 50% |

**Why ±2%.** Consecutive evaluations of one run differ by a median 1.7% in SR2's constant-rate phase (90th percentile 4.9%). A ±2% band on a mean of nine evaluations is well above that noise and well below the closest pre-registered gaps (2–3%).

**Decision rule (draft).** BF1–BF3 hit: every 2B-token Fock arm trains in bf16, except runs compared against an existing float32 run. Any of BF1–BF3 misses: train in float32 and look for the cause. BF4 decides only whether the switch is worth using.

**Result.** None; the probe was not run. On 2026-10-18, after an A100 measurement, bf16 made a training step only 2% faster. The analytic $V_\varphi$ force with layer checkpointing off made it $1.62\times$ faster at 29.9 GiB, with identical arithmetic, and bf16 on top added 2%. Later arms record that selective bf16 is not used for that reason.

**Status.** Deferred. The draft is kept for a later scale ($d = 768$), where matrix multiplies weigh more. Two questions were left open for the author: whether to probe PMX as well, and whether to run the probe with the analytic $V_\varphi$ force on.

<a name="scorecard"></a>
## Scorecard

| programme | date scored | calls | hits | misses | verdict as recorded |
| --- | --- | ---: | ---: | ---: | --- |
| CG8 (§5.12) | not scored | 6 (+1 on SR5) | — | — | no result recorded |
| DP (§5.14) | 2026-10-05 | 5 | 3 (both DP1 calls, DP3) | 2 (both DP2 calls, on G2) | no number dynamics; salience is retention, not a Poisson mean |
| F0 (§5.16) | 2026-10-06 | 4 | 1 (P4) | 3 | inconclusive; F1 is the test |
| RF (§5.23) | 2026-10-12 | 4 per arm | PMX 1 (RF-c), PM1 0 | PMX 3, PM1 4 | weak forward signal in PMX only; G2 probe invalid |
| MT (§5.24) | 2026-10-12 | 4 per arm | 3 per arm (MT-a, b, c) | 1 per arm (MT-d) | transitions exist but have no direction |
| BF (§5.26) | not run | 4 (draft) | — | — | deferred 2026-10-18 |

The DP row counts the five calls of the protocol's call table; its result table scores the two DP1 calls together.
