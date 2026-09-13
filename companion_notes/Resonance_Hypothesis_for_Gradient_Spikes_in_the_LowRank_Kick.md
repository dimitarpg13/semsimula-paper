# Resonance, Not Magnitude: a Threshold Hypothesis for Gradient Spikes in the Low-Rank Kick

**Status: FALSIFIED, 2026-09-13. Superseded by
[Gradient_Spikes_as_Routing_Conjunctions.md](Gradient_Spikes_as_Routing_Conjunctions.md),
which establishes the actual cause.**

The mechanism proposed here is not present. Direct measurement (§5) finds
the stiffest low-rank mode at $\omega \Delta t = 1.50$ against a wall at
2, with **0 of 32** (microbatch, layer) cells crossing it; the quantity
does not track spike magnitude; truncation moves it the wrong way; and the
coherent tail the explanation required is absent.

The answer turned out to be that a spike is a **conjunction** between one
microbatch and the routing draw it receives — neither sufficient alone,
and resetting the RNG per microbatch collapses every captured spike to
baseline. That is written up in the note above. Read this one only for the
refutation itself, which is recorded in §4 and §5 and remains valid.

---

## 1. The observation

Stage 2 of the rank-selection procedure (see
`Curvature_Diagnostics_and_Rank_Selection_for_Aniso_Gaussian_Vtheta.md` §7.3)
was run against the step-87196 spikebatch capture with
`replay_rank_truncation_ablation`. Each arm replays the identical captured
batch with every realised $B_k$ replaced by its rank-$r'$ SVD truncation.

| arm | pre-clip grad norm | ntp | batch PPL |
|---|---|---|---|
| untruncated | 2539.20 | 4.3385 | 76.6 |
| rank = 4 (no-op) | 2539.20 | 4.3385 | 76.6 |
| rank = 3 | 2.83 | 4.3808 | 79.9 |
| rank = 2 | 4.35 | 4.4682 | 87.2 |
| rank = 1 | 2.87 | 4.6609 | 105.7 |

The `rank = 4` arm is a built-in fidelity check — the truncation is a true
no-op at full rank — and it reproduces the untruncated arm exactly, so the
harness is sound. The untruncated arm also reproduces the value training
recorded live, 2539.2.

![Measured at step 87196: the gradient norm falls off a cliff at the first truncation and stays on a floor, while batch perplexity moves smoothly and in order](figures/rh_observed_collapse.png)

Three things about this table are hard to explain with the account of spikes
we currently hold:

1. **Dropping only the smallest of four directions collapses the gradient by
   a factor of about 900**, from 2539 to 2.83. It does not attenuate it
   proportionately; it returns the network to an ordinary, healthy gradient
   norm.
2. **The loss barely notices.** That same truncation costs 4.3% batch
   perplexity. A change that annihilates the spike is nearly free in the
   objective.
3. **The collapse does not deepen with more aggressive truncation.** Ranks
   3, 2 and 1 give 2.83, 4.35 and 2.87 — noise around a common floor, not a
   trend. Whatever is switched off is switched off entirely by the first
   step.

Meanwhile `ntp` behaves perfectly smoothly across the same arms (4.3385,
4.3808, 4.4682, 4.6609), so the forward pass is responding continuously to
truncation. Only the gradient is discontinuous.

> Note on the fourth column of the probe's output. Its
> `relative_force_error` reads 0.9999, 1.0002 and 0.9998 for ranks 3, 2 and
> 1, and those numbers are uninformative here. Writing $g$ for the full
> gradient and $g'$ for a truncated one, the metric is
> $\lVert g' - g \rVert / \lVert g \rVert$; once $g'$ collapses to about
> 0.1% of $g$, the triangle inequality pins the ratio into
> $[0.9989, 1.0011]$ regardless of which directions were removed. All three
> observed values sit inside that band. Read `ntp` and the gradient norm
> instead.

---

## 2. Why the magnitude account does not fit

The working account of these spikes has been a magnitude one: the low-rank
curvature channel grows too sharp, and sharpness drives the gradient. That
account has never sat comfortably with the bracket measurement in
`CfC_BAOAB_Integrator_and_Mitigations.md` §33, which found
$\sigma_{\max}(B_k)^2$ — the very quantity the sharpness story indicts —
elevated by only +1% to +24%, and non-monotonically, between a healthy
checkpoint and both hard-trigger snapshots. §34.4 drew the reasonable
conclusion that the off-diagonal channel is a weak lever and that `E`, `P`
and `depth_code` lead the real triggers.

The table above is hard to square with "weak lever". A 4.3% perturbation of
the objective, applied to that same off-diagonal channel, removes the spike
completely.

Both results are explained at once if the spike is **not a smooth function
of curvature magnitude but a threshold crossing**. A threshold correlates
weakly with the magnitude of its argument — the argument only has to cross,
and by how much is nearly irrelevant — while remaining exquisitely sensitive
to any change that moves it back across.

*The figure below illustrates the rejected hypothesis, not a measured
result.*

![Under a magnitude model the response is proportional and a weak elevation implies a weak effect; under a threshold model the same input range spans a cliff, and a deterministic mechanism still leaves the magnitude a poor predictor](figures/rh_threshold_vs_magnitude.png)

The right-hand panel is the point worth dwelling on. It is generated by a
*purely deterministic* threshold, and yet the correlation between spike size
and $\sigma_{\max}(B_k)^2$ comes out at 0.34, with the median elevation at
spikes only about +8% — the same order §33 measured. Weak correlation is not
evidence against the mechanism here; it is what the mechanism predicts. The
reason is in the next section: $\omega \Delta t$ is set by the *summed*
operator, so any single well's $\sigma_{\max}^2$ is only one of its inputs.

---

## 3. The mechanism this would be

`baoab_cfc` splits $V_\theta$'s local curvature into two channels and treats
them differently, as described in
`Analytic_Multi_Channel_Integration_in_Structured_Vtheta.md` and in the
integrator note. The diagonal part is integrated **exactly** by the
closed-form harmonic propagator and carries no stability constraint. The
off-diagonal part,

$$\mathcal{L} = \sum_k g_k B_k B_k^\top, \qquad g_k = w_k \exp\left(-\frac{1}{2} e_k\right),$$

is integrated as an **explicit kick**. Explicit integration of a harmonic
mode of frequency $\omega$ at step $\Delta t$ is stable only while

$$\omega \Delta t \lt 2,$$

and the relevant frequencies are $\omega_i = \sqrt{\lambda_i}$ for the
eigenvalues $\lambda_i$ of $\mathcal{L}$. So the governing quantity is

$$\omega_{\max} \Delta t = \Delta t \sqrt{\lambda_{\max}(\mathcal{L})}.$$

This is a hard threshold. Below it the mode oscillates and the kick is
merely inaccurate; above it the mode amplifies geometrically, once per
application. Through $L = 8$ layers an amplification factor $\rho \gt 1$
compounds as $\rho^8$.

*The wall in this figure is real and the formula exact; what §5 refutes is
that the model ever reaches it.*

![Leapfrog amplification is identically 1 below the wall and rises steeply above it; compounded through 8 layers, a 9% overshoot produces the observed 900x spike](figures/rh_stability_wall.png)

### A number this predicts

The amplification per application is exact, not a modelling choice. The
leapfrog recurrence for a harmonic mode is

$$x_{n+1} - (2 - h^2) x_n + x_{n-1} = 0, \qquad h = \omega \Delta t,$$

whose roots have modulus 1 while $h \lt 2$. Beyond the wall one root is real
and larger, giving the per-layer amplification

$$\rho = |a| + \sqrt{a^2 - 1}, \qquad a = 1 - \frac{h^2}{2}.$$

Inverting $\rho^8$ against the measured ratio $2539.20 / 2.83 = 897$ gives
$\rho = 2.34$ per layer and

$$\omega \Delta t \approx 2.18,$$

about 9% past the wall. That is a genuine prediction for D2 below, and a
sharp one: the hypothesis does not merely say "something exceeded a
threshold", it says the excursion should be *small*. If a direct measurement
finds the top mode at $\omega \Delta t$ of 5 or 10, the compounding
arithmetic does not work and the mechanism is wrong. Three caveats — the
estimate assumes undamped leapfrog (BAOAB's friction adds margin), a uniform
excursion across layers, and amplification as the sole source of the 897x
ratio — so treat it as an order-of-magnitude consistency check rather than a
precise target.

Four consequences, matching the four puzzles above:

**It explains the weak magnitude correlation.** If the model normally
operates just below the wall, a 10% rise in $\sigma_{\max}^2$ is enough to
carry $\omega \Delta t$ from 1.95 to 2.05 and flip stability. Correlation
between spike size and curvature size is then expected to be poor, which is
what §33 measured.

**It explains why removing the smallest direction is enough.** $\lambda_{\max}(\mathcal{L})$
is the top eigenvalue of a *sum* over $K = 8$ wells and $n_c = 5$ channels —
up to 160 rank-one contributions in a $d = 384$ space. The relevant quantity
is therefore not any single well's spectrum but the alignment structure of
the sum, and those two can come apart completely.

The sharp case is a **coherent tail**: a direction that is individually
negligible in every single well, the smallest singular direction of each
$B_k$, yet placed similarly by all 40 factors. Individually it is beneath
notice. Collectively it adds as $40 c^2$ while the incoherent bulk adds only
as $\sqrt{40}$, so it can dominate $\lambda_{\max}$ outright. Truncation
removes it from all 40 factors simultaneously — which is exactly what
truncating each well by one direction does.

![With independent wells, removing the weakest direction costs 4% of the top eigenvalue; with a coherent tail it costs 56%, and the curve then flattens](figures/rh_sum_alignment.png)

A synthetic construction at the deployed shape puts numbers on the gap. With
independent wells, truncating rank 4 to rank 3 costs 4% of
$\lambda_{\max}(\mathcal{L})$ — the gentle, proportional loss one would expect. With a
coherent tail it costs **56%**, and the curve then flattens, with ranks 2 and
1 taking it only a little further. That shape — one cliff, then a floor — is
the shape the measured gradient column has, and nothing in the construction
was fitted to it.

This also reframes what a "small" direction means. Under the participation
ratio, a well spreading its budget evenly across four directions
(`pr_p50 = 3.68/4`) looks healthy and unconcentrated. Coherence across wells
is invisible to that statistic, because it is a per-well measure. Two wells
can each be perfectly well-conditioned while their fourth directions point
the same way.

**It explains why the loss barely moves.** The potential's *value* is
dominated by the leading directions; the integrator's *stability* is
governed by the operator norm of the summed curvature. These are different
functionals of the same $B_k$, and one can be changed sharply while the
other barely moves.

**It explains the all-or-nothing floor.** A spike under this account is an
amplification event, not a large-but-ordinary gradient. Switch the
amplification off and what remains is the baseline the network would have
produced anyway — about 3 — with no dependence on how far past the wall the
system had been.

It also explains the otherwise awkward fact that the spike's largest
contributors are `register`, `depth_code` and `creation_gate` rather than
$V_\theta$ itself. Amplification in $h$ propagates backward into every
parameter upstream of the hidden state, so the groups that *report* the
spike need not be the ones that *cause* it.

---

## 4. What would falsify it

The hypothesis is worth exactly as much as the tests that could kill it.

![The D1 control splits the outcome: if the spike also dies under matched noise the truncation says nothing about rank, and if it survives the rank reading stands](figures/rh_falsification_map.png)

**F1. The perturbation control collapses too.** The truncation confounds two
things: it removes specific directions, and it changes $B$ by a particular
magnitude. `replay_rank_perturbation_control` holds the magnitude and drops
the specificity — isotropic noise carrying exactly the discarded energy,
leaving $B$ full rank. If the spike also dies under matched noise, then
*any* perturbation of that size defuses it, the truncation result says
nothing about rank or direction, and the resonance reading needs the
independent evidence of F2 and F3 to survive at all. **This has not been
run. It should be run first.**

**F2. No token or layer is over the wall. — THIS IS WHAT HAPPENED (§5).**
If a direct measurement of
$\Delta t \sqrt{\lambda_{\max}(\mathcal{L})}$ at the spike checkpoint finds nothing
near 2, the mechanism is simply absent and the hypothesis is dead.

**F3. The transition is smooth, not sharp.** Not reached: F2 fired first,
and with no crossing there is no knee to look for. A threshold implies a knee.
Scaling $B \mapsto \alpha B$ continuously should show the gradient norm
jumping at some $\alpha^\ast \lt 1$, not ramping.

Three alternatives to keep live:

- **A1. Generic fragility.** The spike is sensitive to any perturbation of
  $V_\theta$, with no threshold involved. F1 addresses this.
- **A2. Well-weight reshuffling.** Truncation changes $B_k$, hence the
  exponent $e_k$, hence $g_k$ — so which wells are active shifts, and the
  spike may be a property of one particular well-weight pattern rather than
  of stiffness. Distinguishable by recording the $g_k$ distribution beside each arm
  and checking whether the defused arms differ in *which* wells fire.
- **A3. Something outside the well potential.** The spike is genuinely a
  register or
  depth-code phenomenon and $V_\theta$ truncation disturbs it only
  incidentally. The layer-profile prediction in D4 separates this.

---

## 5. Result: the wall is real, and the model is not near it

Measured on the A100, 2026-09-13, with `probes.resonance` (D2, D2b).

### 5.1 Nothing crosses the wall, and nothing is close

| step | pre-clip grad | predicted $\omega \Delta t$ | measured p95 | measured max | % over wall |
|---|---|---|---|---|---|
| 87196 | 2539.2 | 2.183 | 1.277 | **1.500** | 0.000 |
| 86201 | 685.6 | 2.119 | 1.276 | **1.524** | 0.000 |
| 90360 | 567.3 | 2.111 | 1.303 | **1.522** | 0.000 |

The model runs at about 75% of the stability limit with zero crossings.
That is a reasonable operating point and, as far as these notes record, it
had never been measured — the wall was reasoned about but never observed.

### 5.2 The quantity does not track what it was meant to explain

Across a 4.5x range of spike magnitudes the measured band is
$[1.500, 1.524]$, a width of 0.024, against a predicted band of
$[2.111, 2.183]$. Worse for the hypothesis, the ordering is *inverted*:
step 87196, the largest spike in the set at 2539.2, has the **lowest**
maximum at 1.500, while 90360, the smallest at 567.3, reads 1.522. The
quantity is essentially constant and slightly anti-correlated with the
thing it was supposed to drive.

### 5.3 The within-bundle control fails in the opposite direction

The decisive test was whether the truncation that annihilates the gradient
also carries $\omega \Delta t$ under the wall. It does the reverse:

| arm | p50 | max | % over wall |
|---|---|---|---|
| untruncated | 1.025 | 1.500 | 0.000 |
| rank = 3 | 1.050 | 1.539 | 0.000 |
| rank = 1 | 1.053 | 1.590 | 0.000 |

Truncation *raises* the stiffest mode. The reason is an effect this note
failed to anticipate, and it matters beyond the refutation. The well weight
is

$$g_k = w_k \exp\left(-\frac{1}{2} e_k\right),$$

and the exponent $e_k$ contains the low-rank term, so truncating $B_k$
**lowers the exponent and raises the well weight**.
Less curvature per well, but every well firing harder, and here the weight
wins. A rank truncation is therefore not a clean "less curvature"
intervention on $\mathcal{L}$ at all — the two effects compete, and which one
dominates is an empirical matter rather than something to be assumed.

### 5.4 There is no coherent tail

§3 explained the puzzle by supposing the wells' weakest directions point the
same way. They do not. Measured `tail_pr` is **7.525 out of 8**, i.e. 94% of
the maximum, meaning the tails are close to mutually orthogonal. The mean
absolute cosine is 0.0765, only about 1.9x the random-vector baseline for
$d = 384$, which is near $\sqrt{2 / (\pi d)} \approx 0.041$. Slightly more
aligned than chance, nowhere near coherent.

### 5.5 The one loophole, and why it does not save the account

Power iteration returns a Rayleigh quotient and therefore *under*-states
$\lambda_{\max}$. Reaching the wall from 1.500 would require the true value
to be $(2 / 1.5)^2 \approx 1.78$ times larger, which 24 iterations should
not miss on a well-separated spectrum; re-running at a much higher iteration
count settles it. Two things hold regardless of the absolute level: the same
estimator was used everywhere, so the *comparisons* in §5.2 and §5.3 stand;
and the step used is the full layer $\Delta t$, the largest defensible
choice, so the reported figure is already an upper bound on that axis.

### 5.6 What this leaves

The observation in §1 is untouched and still unexplained. What §5 removes is
one candidate explanation, and it removes the specifically *structural* one —
the appeal of the resonance account was that it was a property of the
integrator rather than an accident of one step, and that is now excluded.

Of the alternatives in §4, **A2 (well-weight reshuffling)** is considerably
more plausible than it was: §5.3 shows directly that truncating $B_k$ moves
the well weights $g_k$, which is the mechanism A2 names.

A terminology caution, since the two are easy to run together and this note
originally did. The weight $g_k$ — defined in §3 — is a per-well,
*unnormalised* value at a single token. "Exponent occupancy" already means
something else in these notes — `live_frac`, the fraction of token-slots per
bank whose exponent clears an underflow cutoff, measured in
`CfC_BAOAB_Integrator_and_Mitigations.md` §39.3. That is a count over a
population, not a weight. Both derive from $e_k$, and neither is the
normalised *responsibility* of the log-sum-exp quadratic family. All three
are now defined and contrasted in
`deep_dives/Structured_Scalar_Potential_Design_and_Theory.docx`, under
"Three weights that are easy to confuse". **A1 (generic
fragility)** remains untested, and D1 is still the experiment that
distinguishes them.

## 6. Diagnostic programme, cost-ordered

> **D2 and D2b are done and returned negative (§5); D3, D5 and D6 are
> moot — they all test consequences of a wall crossing that does not
> occur. D1 remains outstanding and is now the only live item here.**

**D1. Run the perturbation control. — STILL THE NEXT STEP.** `replay_rank_perturbation_control` at
step 87196 with `ranks=(1, 2, 3)`, matching the truncation run. Minutes, same
harness, settles F1. Everything below is worth doing only if the spike
survives matched noise.

**D2. Measure the wall directly — done, negative (§5.1).** Implemented as
`probes.resonance.omega_dt_report` in `semsimula-diag`. It does not use
`lowrank_modes`, because the full eigendecomposition is exactly what makes
the exact arm unaffordable: only $\lambda_{\max}(\mathcal{L})$ is needed, and that
comes from power iteration on $\mathcal{L} v = G(G^\top v)$ — thin matmuls, no
eigensolver. $G$ is reconstructed by calling `harmonic_terms_lowrank` on
precisely the $(\xi, h)$ the layer already linearised at, so the model's own
computation is untouched.

Two properties to keep in mind when reading its output. The estimate is
**one-sided**: a Rayleigh quotient of an unconverged vector always
*under*-states $\lambda_{\max}$, so a reading near the wall means "at least
this large". And the step that matters is the **kick's** $\Delta t$, not the
half-step handed to `cfc_substep`, since under `baoab_cfc` the low-rank part
rides the kick — a factor of 2 that would silently halve every number.

Compare the spike capture against a healthy checkpoint; the contrast is the
result, not either number alone.

**D2b. Cross-well tail coherence — done, negative (§5.4).**
Implemented as `probes.resonance.tail_coherence_report`. Implied by the
coherent-tail mechanism in §3 and not covered by any existing statistic: for
each layer and channel it takes the weakest $h$-space direction of every
well's $B_k$ — obtained without a $d \times r$ SVD, as $B v_{\min} / s_{\min}$
from the $r \times r$ Gram matrix — and reports both the mean pairwise cosine
across wells and the participation ratio of the stacked tails, which runs
from 1 (every tail identical) to the well count (mutually orthogonal). The
prediction is that this coherence is elevated at the spike capture relative
to a healthy checkpoint, *while per-well participation ratios stay flat*.
That combination is the signature, and it is a weight-only measurement
needing no replay at all — the same cost class as `stiffness_report`. It is
also the measurement most likely to be *newly* informative, since every
statistic collected so far has been per-well and is blind to this by
construction.

**D3. Sweep the scale for a knee. — moot, see §5.** Replay with $B \mapsto \alpha B$ for
$\alpha$ across roughly 0.90 to 1.00. A sharp transition confirms a
threshold; a smooth ramp refutes it. A small variant of the truncation
harness, and it also locates how far past the wall the spike sits.

**D4. Per-layer amplification profile.** Compounding predicts geometric
growth of the boundary gradient through the stack at a spike and a flat
profile when healthy. `replay_spike_batch(per_layer=True)` already records
`per_layer_h_grad`, so this may be answerable from captures already in hand.
A flat profile at the spike would also support A3 over the resonance story.

**D5. Halve the step. — moot, see §5.** The wall is on $\omega \Delta t$, not on $\omega$.
Replaying the identical weights and batch at reduced $\Delta t$ should defuse
the spike at some $\Delta t^\ast$, with no change to the model at all. This
is as close to a controlled experiment on the mechanism as the setup allows.

**D6. Replay the spike batch under `baoab_cfc_lowrank`. — moot, see §5.** That arm
integrates the low-rank channel exactly and therefore has no
$\omega \Delta t$ wall, so under this hypothesis it should defuse the spike
outright. Note the reframing this implies: §34 retired the arm as too
expensive for *training* — about 120 s/step against 10-15 s/step, see
§7 of `Anisotropic_Rank_r_Frozen_Force_and_LowRank_CfC_BAOAB.docx` for the
cost analysis — but its cost is irrelevant for a *single replayed step*. It
becomes a perfectly practical diagnostic instrument. §34.4's separate
judgement that the arm is "aimed at the wrong target" rests on
$\sigma_{\max}(B_k)^2$ being a weak correlate, which is precisely the
inference a threshold mechanism would invalidate. The cost verdict stands
regardless; the targeting verdict may not.

---

## 7. What survives

The mechanism is gone; three things outlast it.

**A standing fact about the integrator.** The stiffest low-rank mode sits at
roughly 75% of the explicit kick's stability limit, with no crossings
observed in any capture examined. The wall was reasoned about at length in
the integrator note and never measured. It is measured now, and there is
real headroom — which also means the `omega dt < 2` constraint is not
currently what bounds this configuration, and arguments resting on its
tightness need re-examining.

**A measurement that stays cheap.** `probes.resonance` computes
$\lambda_{\max}(\mathcal{L})$ by power iteration on $\mathcal{L} v = G(G^\top v)$, so the wall
can be monitored during training for a small fraction of what the exact
low-rank arm costs. `observe()` logs it without altering the model. That is
worth keeping switched on: the headroom measured here is a property of *this*
configuration, and a rank-8 pilot or a longer schedule could consume it
without anything else giving warning.

**A caution about rank truncation as an instrument.** §5.3 showed that
truncating $B_k$ raises the well weight $g_k$, because the truncated
directions were contributing to the exponent that suppresses it. Any
experiment that truncates $B$ is therefore changing two things at once, and
the weight term can dominate. That applies to the Stage 2 ablation in the
rank-selection note as much as it does here.

**What does not survive** is the mitigation sketch this section previously
carried: gating exact integration on a measured $\omega \Delta t$ solves a
problem the model does not have. The narrower point it rested on does stand,
independent of any mechanism — an *unconditional* "remove the weakest
direction" would be a permanent capability tax, costing 4.3% batch
perplexity at step 87196 while the `ntp` column shows every direction
carrying real signal. That is a rank-3 model paying rank-4 parameter costs,
and it was never the right shape for a mitigation.

---

## 8. Provenance

The observation arose from the first successful GPU run of
`replay_rank_truncation_ablation` (step 87196, A100, 2026-09-13), built to
answer Stage 2 of the rank-selection procedure. It does not answer that
question: with any truncation defusing the spike wholesale, the ablation
cannot report what the tail directions contribute. The `ntp` column still
carries a usable Stage 2 reading — each discarded direction costs real
perplexity, 4.3% for the smallest alone, which argues the rank budget is not
idle and is consistent with Stage 1's saturated `pr_p50 = 3.68/4` — but the
rank question should be re-asked at a healthy checkpoint, since step 87196 is
an outlier by construction and the worst possible place to ask what the model
does ordinarily.

The probe implementations live in `semsimula-diag`
(`probes/precision_cap.py`); `MIGRATION.md` there records the measurement and
the several implementation hazards met on the way to it.

Provenance: figures generated by
`companion_notes/figures/_make_resonance_hypothesis_figs.py`. Figure 1 and
the $\omega \Delta t \approx 2.18$ estimate are exact evaluations of the
leapfrog stability formula; Figure 3 reproduces the measured step-87196
table; Figures 2 and 4 are clearly-labelled synthetic illustrations at the
deployed shape, fitted to nothing. Integrator code in
`notebooks/conservative_arch/parf/cfc_baoab.py`
(`lowrank_modes`, `harmonic_terms_lowrank`); well parameters in
`notebooks/conservative_arch/parf/model_aniso_gaussian_vtheta.py`.

Measured results in §5 come from `probes.resonance` in `semsimula-diag`
(`omega_dt_report`, `omega_dt_under_truncation`, `tail_coherence_report`),
run against the step-87196, 86201 and 90360 spikebatch captures on an A100.

Last updated: September 2026. Initial version stated the threshold
mechanism, the compounding estimate, the coherent-tail explanation and the
D1-D6 programme. Revised the same month to record that D2 and D2b came back
negative on all three legs and the hypothesis is refuted; §5 added, §7
rewritten around what outlasts it.
