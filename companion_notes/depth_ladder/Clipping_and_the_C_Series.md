# Gradient clipping and the C-series probes

**What this note is.** A short, readable summary of one episode in the depth ladder, written 2026-11-17. On 2026-09-28 the question "are the clips layer-depth dependent and mistuned?" led to four protocol blocks: a measurement of how hard each clip binds, a reviewer-facing answer on what the clipping costs, an audit of the older model cards, and a plan of cheap probes (the C-series, C0 to C7) on the clip confound. This note gathers them in one place, with the probe outcomes recorded later in the log. Every number, verdict and date is copied from the [protocol](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md), which stays the full running log. The ladder's rules are in [Ladder_Protocol_Overview.md](Ladder_Protocol_Overview.md); the arms named here are described in [Ladder_Early_Arms_Results.md](Ladder_Early_Arms_Results.md).

**Status.** C0, C1 and C2 are run, and their readings say the clip is cosmetic for the trained endpoint; C3 to C7 have no result recorded, and C4 and C5 are not licensed.

**A naming warning.** The labels C1 to C5 are reused elsewhere in the protocol for unrelated things: the corpus statistics of §5.13, the FLOW-C calls of §5.19, and the matched GPT-2 runs on TinyStories (C1), code (C3) and the code pair PM2-CODE (C5). None of those is part of this series.

**Contents**

1. [Terms](#c-terms)
2. [The clip thresholds and why they matter](#c-thresholds)
3. [What the clipping costs](#c-cost)
4. [The model-card audit](#c-cards)
5. [The C-series: plan and outcomes](#c-series)
6. [Side findings from the same block](#c-side)
7. [What stands and what is open](#c-open)

---

<a name="c-terms"></a>
## 1. Terms

- **Gradient clipping.** Before each optimizer step, if the gradient's norm exceeds a threshold, the gradient is scaled down so its norm equals the threshold. It caps the size of one update; it does not change the update's sign.
- **Per-group clips.** Training does not use one global clip. It uses a global clip of 1.0 plus nine per-group overrides. The function `clip_grads_per_group` puts parameters into named groups and clips each group **jointly**: one norm per group, one scalar rescale per group. The tightest override is 0.1, on `reverse_channel_scale` and `reverse_ch`.
- **Pre-clip norm.** The group's gradient norm before clipping. Each run log's `top[...]` line reports the largest group and its pre-clip norm every 50 steps.
- **Clip-hit rate.** The fraction of logged steps on which a clip fires. From $L = 4$ onward every ladder point must report it.
- **The reverse channel and its gate.** The reverse channel is the path by which the register bank (the Fock mechanism) reaches the tokens. `reverse_channel_scale` is its gate: one scalar per layer. An earlier experiment, E5, found it sets the layer-1 output direction outright, and the book's central claim rests on it.
- **Adam's update.** Adam moves a parameter by $\mathrm{lr}\cdot\hat{m}/\sqrt{\hat{v}}$, where $m$ is a running mean of the gradient and $v$ a running mean of its square. The ratio $|m|/\sqrt{v}$ is the fraction of the full learning rate the parameter actually moves at.

<a name="c-thresholds"></a>
## 2. The clip thresholds and why they matter

**Structure.** The model was built at $L = 1, 2, 4, 8$ and every parameter bucketed by its clip group. Exactly three groups grow linearly in $L$: `creation_gate`, `destruction_gate` and `depth_code`. `reverse_channel_scale` is per-layer by config. Everything else (`V_theta`, `V_phi`, `register`, `score_head`, embeddings) is depth-invariant. For a group that grows with $L$, the joint norm sums $L$ contributions. It should grow as $\sqrt{L}$ if they are independent, or as $L$ if they are correlated. The threshold it is compared against never moves.

**Provenance.** `depth_code: 0.25` was tightened from 0.5 on 2026-08-23, against the $L = 16$ g0.1 OpenWebText run. Every other override traces to $L = 8$ and $L = 16$ forensics (step 6435, step 71194). None was revisited when the ladder dropped to $L = 1$, 2 and 4.

**Measured on the completed rungs**, from the pre-clip `top[...]` lines:

| arm | group that dominates | share of logged steps | median pre-clip | threshold | over by |
| --- | --- | ---: | ---: | ---: | ---: |
| L=2 `none` | `reverse_channel_scale` | 98.5% | 2.00 | 0.1 | 20× |
| L=2 `attention_potential` | `reverse_channel_scale` | 88.9% | 1.00 | 0.1 | 10× |
| L=2 `attention` | `relax_field` / `reverse_channel_scale` | 59% / 34% | 0.90 / 1.00 | — / 0.1 | — / 10× |
| L=2 `none` no-RC | `depth_code` | 90.9% | 0.20 | 0.25 | not clipped |
| L=4 `none` (to step 8,350) | `reverse_channel_scale` | 77.2% overall, 100% after step 4,000 | 1.20 (1.60 steady) | 0.1 | 12–16× |

**The headline is the reverse channel, not the depth code.** Its gate is clipped by a median factor of 10 to 20, on essentially every step, on every arm that has a reverse channel, including the flagship. Its maximum reaches 8.8 against a 0.1 ceiling, that is 88 times. The `norc` arm (no reverse channel) is the only completed arm where nothing is meaningfully clipped.

**The update from $L = 4$ at step 8,350 (2026-09-28).** The dominant group by 1,000-step block:

| steps | dominant group | share | median pre-clip |
| --- | --- | ---: | ---: |
| 0–1,999 | `depth_code` | 63–80% | 0.40–0.50 |
| 2,000–3,999 | `reverse_channel_scale` | 85–95% | 1.00 |
| 4,000–8,350 | `reverse_channel_scale` | 100% | 1.60 |

- **The onset is explained.** The handover at about step 2,000 is the reverse channel's warmup completing: 4,000 forwards at grad-accum 2 is exactly 2,000 optimizer steps. From then on the gate is clipped by a median 16 times (maximum 45) on every step.
- **`depth_code` is not the problem.** It dominates only during warmup, peaks at twice its threshold, and is a warmup transient at $L = 4$. The first worry is withdrawn.
- **The $\sqrt{L}$ prediction is refuted.** On the one window both logs share (steps 6,050–8,350, 47 logged steps each):

| arm | dominant | share | median pre-clip | over threshold |
| --- | --- | ---: | ---: | ---: |
| L=2 `none` | `reverse_channel_scale` | 100% | 1.50 | 15× |
| L=4 `none` | `reverse_channel_scale` | 100% | 1.60 | 16× |

Doubling the depth moved the group norm by 7%, not the 41% that $\sqrt{2}$ predicts. The per-layer gate gradients largely cancel, or shrink with depth. The group-membership observation stands; the inference to a per-rung difference in clip strength does not.

**What remains.** The gate is clipped 10 to 20 times on essentially every step, at both depths. That is a **uniform** confound, not a per-rung one. It is harmless for the ladder's internal comparisons and no help at all for absolute claims about what the reverse channel learns. The log's verdict: **a confound of unknown magnitude, not a demonstrated error.**

<a name="c-cost"></a>
## 3. What the clipping costs

The protocol wrote this to be quotable, for a reader of the book or a referee of the TMLR paper. The disclosure comes first: the global clip plus nine overrides, the gate at a median 10 to 20 times its threshold (15 at $L = 2$, 16 at $L = 4$, maximum 88), and thresholds set from $L = 8$ and $L = 16$ forensics in August–September 2026 and never re-derived for the ladder.

**Three reasons it is not thought to cost measurable perplexity**, two of them measured:

1. **The gate is falling, not held against a ceiling.** On the $L = 2$ flagship it reads 0.0612/0.0600 at step 500 and 0.0174/0.0148 at step 31,500, a fall of 3.5 to 4 times. A clip limits step size in either direction, so all it can have done is slow a decline.
2. **Adam largely absorbs a persistent rescale.** Scaling every gradient in a group by one factor scales $m$ and $\sqrt{v}$ alike and leaves the ratio unchanged. On the endpoint's optimizer state the gate's $|m|/\sqrt{v}$ is 0.007 to 0.27: what limits its movement is step-to-step gradient cancellation, not the clip.
3. **For this group the clip cannot distort direction.** The group is just the per-layer gate scalars, so one scalar rescale multiplies every element by the same number. It cannot change which layer gets more gate, only how fast the whole vector moves.

**Where the argument is soft.** The clip factor is not constant: the pre-clip norm ranges 1.0 to 4.5 against a 0.1 threshold. So the clip compresses large-gradient steps harder than small ones. That is variance reduction on the gate's trajectory, not a uniform learning-rate cut, and it can move where the gate settles. Its direction and size are unmeasured, and reason 2 does not defend against it.

**The direction of any residual bias.** If the clip binds at all, it has slowed a shrinking gate, so the unclipped model has a **weaker** reverse channel. Any bias makes the mechanism look more important than it is, not less.

**What would settle it.** E5 swept the gate multiplier $\lambda$ from 1 down to 0 and found PPL rising monotonically with no knee. Nobody had looked above 1; that became C1. C1 is an upper bound, not a prediction: in this volume inference-time ablation overstated this mechanism 3.91 times, against 1.31 times for trained-without.

**Status as written on 2026-09-28: open, bounded, and biased in the direction that does not flatter the thesis.**

<a name="c-cards"></a>
## 4. The model-card audit

On 2026-09-28 all fifteen cards in the older (SPLM-family) collection were audited against what the CfC+BAOAB work had since measured. Two classes of error were fixed; a third was left standing.

1. **Reverse-channel models called "purely conservative".** The Fock-PARFLM card said so in its opening sentence and derived a Riemannian geometry from the premise that all forces come from one scalar potential. Its own `config.json` has `use_reverse_channel: true`. E1 later measured the deflection of the layer step from the damped $V_\theta$ geodesic at 1.09 with the channel on against 0.0003 with it off, so the non-gradient term carries most of the step. The `fock-attention` and `hybrid-splm` cards listed Fock-PARFLM among "the purely conservative variants". Corrected on four cards, with the removal from the list stated rather than silent. `depthcond-vtheta` and `anisogaussian-vtheta` needed nothing.
2. **A small residual read as proof of geodesy.** Three gamma-sweep cards said a residual $R \approx 0$ means the trajectory is a damped geodesic. The calibration withdrew that inference: at one step per layer the residual cannot separate a geodesic from a forced trajectory. Scoped in place. What survives is the comparative use across $\gamma$, including the coincidence of the PPL and $\bar{R}$ minima. The `hybrid-splm` card's diagnostic-battery claim was scoped the same way.
3. **Left standing: about 8% redundancy.** 21 verbatim blocks are shared by three or more cards, about 37 KB of 461 KB. Cosmetic. The duplicated correction notes are deliberate: each card carries its own.

Verified after upload: 0 of 15 cards still carry any of the three corrected claims.

<a name="c-series"></a>
## 5. The C-series: plan and outcomes

Agreed on 2026-09-28: study the confound with targeted probes, ordered by cost, rather than one big re-run, and let $L = 4$ finish first. The probes C1 and C2 were scored on G2, the Gen 3 $L = 2$ Fock model with live gradients.

| probe | question | cost | prediction | outcome |
| --- | --- | --- | --- | --- |
| C0 | which way is the gate moving? | free | none recorded | run 2026-09-28: gate falls 3.5–4× |
| C1 | is the trained gate below its inference optimum? | free, minutes | PPL rises monotonically above λ = 1 | HIT, 2026-10-05; question closed |
| C2 | does Adam absorb the clip? | free, minutes | (reading stated, no probability) | finished 2026-10-05: clip cosmetic for the endpoint |
| C3 | what can a joint clip do to a group of L scalars? | free, code read | none | no result recorded |
| C4 | does a 500-step unclipped continuation differ? | about 30 min GPU | gate ends lower by under 20%; val PPL within ±1.5 | not run: not licensed by C1 |
| C5 | full re-run with the threshold raised | 11.7 h | none | not run: only if C1 or C4 separate |
| C6 | is the watchdog blind to this parameter? | free, documentation | none | no result recorded |
| C7 | does clipping slow convergence? | free, no GPU | none | no result recorded |

No probe in the series carries a stated probability.

**C0, which way is the gate moving?** The gate read from the $L = 2$ `none` checkpoints:

| checkpoint | step | raw gate (per layer) | tanh |
| --- | ---: | --- | --- |
| `_step500_best` | 500 | 0.0612, 0.0600 | 0.0611, 0.0599 |
| `_best` | 31,500 | 0.0174, 0.0148 | 0.0174, 0.0148 |

The gate falls over training. The starting hypothesis, "the clip suppressed a gate that wanted to grow", is **refuted in its stated direction**. The mirror image survives: the unclipped model would have a gate that falls further and faster.

**C1, the gate above $\lambda = 1$.** The E5 slider (`R11_LAMBDAS` in Cell 6b-11) stopped at 1.0. C1 extends it upward. A PPL fall above 1 would be the signature of a binding clip. Scored with `debug/c1_c2_g2.py`, scaling the gate through its parameter, on the tokens of 6b-11 ($4 \times 4 \times 512$):

| λ | PPL | against λ = 1 |
| --- | ---: | ---: |
| 3.0 | 3,019.20 | +5,255% |
| 2.0 | 1,172.45 | +1,980% |
| 1.5 | 233.60 | +314% |
| 1.25 | 78.56 | +39% |
| 1.1 | 58.64 | +4.0% |
| 1.0 | 56.38 | 0 |
| 0.9 | 58.08 | +3.0% |
| 0.5 | 102.94 | +83% |
| 0.0 | 246.72 | +338% |

PPL rises on both sides of 1: the trained gate sits at a sharp inference optimum. The $\lambda = 1$ and $\lambda = 0$ values reproduce 6b-11's exactly. The log calls this **the null that closes the question**: the clip did not hold the gate below its preferred value, so C4 and C5 are not licensed.

**C2, the Adam realised-step audit.** The partial result (two optimizer states, mapped to names by element count) gave 0.007 to 0.27. The finished version maps parameters exactly, by Cell 6's `split_decay_params` ordering over 95 tensors:

- The gate reads 0.0446 and 0.0204 (median 0.033), against about 0.15 for other groups. Its gradient cancels step to step; it is not being throttled.
- The heavily clipped reverse-channel weights (threshold 0.1) read a median 0.153, the same as the unclipped $V_\theta$ (0.154), $\xi$ (0.153) and score head (0.164). Adam absorbs the rescale.
- Side finding: `destruction_gates.1`, the last layer's destruction gate, has no optimizer state. It never received a gradient, consistent with DP (§5.14).

**C3, what joint clipping can do.** For a group of $L$ scalars, one rescale is exactly a learning-rate cut for that group: the per-layer allocation is untouched. The plan was to verify this once by assertion. The argument appears as reason 3 in §3 above; no assertion run is recorded.

**C4, the paired short run.** From the $L = 2$ `none` `_best` checkpoint, 500 steps twice with identical data order: as-is, and with the gate's threshold at 2.0 (above its 1.50 median pre-clip norm). A larger gate in the unclipped arm would contradict C0. A later note adds that a clip-threshold tag defect blocks any run that varies a threshold, C4 and C5 included, and must be fixed first. Not run; C1's null removed the licence.

**C5, the full re-run.** Re-train the $L = 2$ flagship with the threshold raised. Only if C1 or C4 separate. Not run.

**C6, the watchdog's blind spot.** Cell 6 excludes `reverse_channel_scale` and `reverse_ch` from the watchdog aggregate, so their warmup ramp does not trigger it. Its own comment says this leaves both watchdog layers "structurally blind to them", and they were in the top 4 of nearly every spike in the 2026-08-23 burst. The action is documentation: say so wherever run health is claimed. No further result recorded.

**C7, what the clips cost in convergence.** Every probe above asks where training ends; none asks how long it took. Variance reduction can change the second without moving the first. The plan: per arm, the fraction of steps each group exceeds its threshold, binned by 1,000 steps, against the loss curve. The usable signal is within an arm, since `norc` is also a different model: does the loss bend where the clip-hit fraction changes, for example at the $L = 4$ handover near step 2,000? No bend across five arms would be a strong null. No result recorded.

<a name="c-side"></a>
## 6. Side findings from the same block

- **`dc_ratio` is not predictive on its own.** Of 7 captured spike events, every smooth cascade had `dc_ratio` below 1.8 and both localized blowups above 2.2. $L = 4$ sat at 2.1 to 3.0 through completely healthy training (no spikes, no watchdog, grad norm about 0.5).
- **The resonance monitor is blind under `baoab_cfc_lowrank`** (diagnosed 2026-09-28, a library bug in `semsimula-diag`). The monitor opens a record on `harmonic_terms`, which this integrator never calls; it calls `harmonic_terms_lowrank`. The 2026-09-26 fix added only the closing half. So the stiffness reading had been dark for every run of the entire ladder, and the fix had to be written. This block does not record the fix landing; Cell 6b-13, written the same day, asserts the clone is at 409175b or later before measuring.

<a name="c-open"></a>
## 7. What stands and what is open

**What stands.**

- The reverse-channel gate is clipped by a median 10 to 20 times on essentially every step of every arm that has it, at both depths measured. The confound is uniform across rungs, not depth-dependent.
- The gate falls over training (C0), sits at a sharp inference optimum (C1), and its gradient cancels step to step while Adam absorbs the rescale on the clipped weights (C2). For the trained endpoint the clip is cosmetic.
- Any residual bias makes the reverse channel look stronger, not weaker.
- The fifteen older cards no longer claim conservativity for reverse-channel models or read a small residual as geodesy.

**What is open.**

- The variance-reduction effect of a non-constant clip factor on where the gate settles is unmeasured; C1 and C2 bound it at the endpoint but do not measure it.
- Convergence cost (C7) has no result recorded.
- C3's assertion and C6's documentation have no completion recorded.
- The clip-threshold tag defect is recorded as blocking threshold-varying runs; no fix is recorded in these blocks.
- **Unclear:** C0, C2's partial result and the planned C4 use Gen 2 $L = 2$ `none` checkpoints, while C1 and C2 were scored on Gen 3 G2. The log does not say whether this change of model matters for the readings.
