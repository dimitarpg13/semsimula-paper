# Post-100K analysis checklist — CfC/BAOAB d384 run

Personal working note, not for publication. What to run now that the
100,000-step run (`fock_cfc_owt_..._baoab_cfc`) has finished, before deciding
on the joint-coupling/QK-norm pilot vs. continuing the additive arm.

**Run finished 2026-09-12.** Best PPL **81.92 at step 96,000** (`_best.pt`).
Final health checks clean: causal probe `[PASS]` (max|dlogit|=0), trained
leak probe `[CLEAN]` (honest PPL 53.59 vs. standard 51.14, diff +0.0469
nats — consistent with the +0.0136 nats seen at step 90,000, not a drift).

Full background lives alongside this file in `companion_notes/`:
[`Diagnostic_Programme_in_CfC_BAOAB_Integrator.md`](Diagnostic_Programme_in_CfC_BAOAB_Integrator.md) §17,
[`Curvature_Diagnostics_and_Rank_Selection_for_Aniso_Gaussian_Vtheta.md`](Curvature_Diagnostics_and_Rank_Selection_for_Aniso_Gaussian_Vtheta.md),
[`CfC_BAOAB_Integrator_and_Mitigations.md`](CfC_BAOAB_Integrator_and_Mitigations.md) §41-42, 49-51.

---

## 0. Before running anything

- **The run has finished** — this is safe to run in a fresh session now.
- **Notebook cell order**: run Cell 0 → 1 → 1b → 1c → 2 → 3 → 4 → 5 → 6d →
  6b → 6b-2 → 6b-3 → 6b-4, in that order. **Skip Cell 6** (training loop) —
  none of this needs it. Cell 1c also re-syncs the archive on this fresh
  session, cheap and harmless to run.
- **Verify what's actually loaded** before trusting any live-model probe —
  Cell 5 builds the model but doesn't load a checkpoint by itself; confirm
  the resume step printed by Cell 2 matches what you expect. (This bit us
  once already — Mitigations §42.5.)
- All spikebatch/prereload bundles below are already copied to
  `spikebatch_archive`/`prereload_archive` — `spectrum_across_checkpoints`
  and the `replay_*` helpers fall back there automatically if the live ring
  has rotated them out, no extra step needed.

---

## 1. Priority checkpoints

**Tier A — the original set** (from the 83K-90K window, all GPU-verified
for replay fidelity on the A100: 0.0002%/0.0003% for 87196/90360 via
`layer_profile.replay_spike_batch`):

| step | pre-clip grad | leading group | why it matters |
|---|---|---|---|
| **87196** | 2539.2 | `register` (85% of total) | most extreme event **in this window**; register/V_theta ratio 8.59 |
| **85885** | 2090.8 | `reverse_channel_scale` | most extreme `reverse_channel_scale`-led event |
| **90360** | 567.3 | `reverse_channel_scale` | same mechanism as 85885, but **shrinking** (2014.1 → 490.8) — the contrast case |
| **86201** | 685.6 | `register` | register-led; use alongside 87196 |

**Tier B — the late-run register cluster (new, 2026-09-12).** Four
register-decoupled events inside a ~1,300-step span (95068 → 96410) — the
densest such clustering seen anywhere in the run, versus isolated single
events everywhere else. Ratios below are **manually computed from pasted
log excerpts**, not yet from a full-log sweep — see the note in §2.6 about
re-running the sweep on the complete `training_log.jsonl` before treating
these as final:

| step | pre-clip grad | register | V_theta | ratio | note |
|---|---|---|---|---|---|
| **95068** | 418.8 | 311.6 | 50.1 | **6.22** | opens the cluster |
| **95091** | 216.2 | 148.5 | 35.9 | **4.14** | |
| **95280** | 659.8 | 527.8 | 75.3 | **7.01** | **also the hard-trigger event** (only hard-trigger after 87196-era) |
| **96410** | 129.8 | 96.3 | 14.4 | **6.69** | **at risk — see below** |

**Secondary set** (lower severity, wider spread — useful only if Tier A/B
don't give a clean read): **81647** (269.6), **82660** (220.5), **81393**
(173.6).

**From the §2.6 ratio sweep over the first 88 archived spikes (steps
50–56,300 only): step 47142** is the strongest register/V_theta decoupling
found so far (ratio 20.03, z=+3.92 — more than twice 87196's 8.59), with
**41824** (9.05) next. Both predate `clip_then_sum`/`NO_DECAY_1D`, so they
are a different regime, but if the register question survives §2.6 these
are the sharpest examples available — and step 47116, 26 steps away in the
same cluster, already has archived `replay_spike_batch` /
`attribute_spike_rows` golden outputs to compare against. **These 88
events are a small prefix of the finished run — re-running the sweep on
the complete log (§2.6) will very likely surface a sharper picture than
either this fit or the Tier B ratios above.**

**Bundle availability, cross-checked against the actual archive contents,
2026-09-12 (run finished).** The live ring evicted everything before step
77,223 — including 70,522/71,194/71,703, so the §16 golden outputs for
those steps can no longer be reproduced, only read. Of the checkpoints
above:

| step | spikebatch (replayable) | prereload (weights only) |
|---|---|---|
| 87196 | yes | yes |
| 90360 | yes | yes |
| 86201 | yes | yes |
| 85885 | **NO — evicted** | yes |
| 95068 | yes | no |
| 95091 | yes | no |
| 95280 | yes | **yes** (in `prereload_archive/`, confirmed) |
| **96410** | **NOT ARCHIVED** | n/a |
| 74870 | n/a | yes (prereload only; unexamined — no `top_groups` on record for it) |

**96410 is genuinely at risk.** Its pre-clip norm (129.8) never cleared the
`ARCHIVE_MIN_GRAD_NORM = 200.0` auto-archive gate, so it was never copied
out of the live ring. If the Colab runtime is still warm, check
`checkpoints/` directly for `..._step96410_spikebatch.pt` and copy it to
`spikebatch_archive/` by hand; if the runtime has already recycled, it's
gone, and the Tier B analysis proceeds on the other three.

So **85885 can only be used by `spectrum_across_checkpoints` (§2.2)**; it
cannot take part in §2.3 / §2.4 / §2.5, which all need the batch and RNG.
Use 90360 (or a Tier B member) as the `reverse_channel_scale` case for
those.

**Weights-only checkpoints** (no batch/RNG, `spectrum_across_checkpoints`
only, not the `replay_*` ablations):
- `_best.pt` — **final, step 96,000, PPL 81.92.**
- `_prereload.pt` for 74870 / 85885 / 86201 / 87196 / 90360 / 95280 (same
  step numbers, saved automatically at each hard-trigger).

---

## 2. Run in this order

> **Except**: §2.6's `register_vtheta_ratio_sweep.py` needs no GPU, no
> bundles and no checkpoints — only `training_log.jsonl`. Run it before
> anything else here; its result decides whether the register branch is
> worth any of the work below.

### 2.1 Rank decision (free, no bundle needed) — **done, 2026-09-13**

```python
spectrum_across_checkpoints()   # defaults to _best.pt
```

**Result, at the final `_best.pt` (step 96,000):**

| statistic | value | reading |
|---|---|---|
| `fro_p50` | 0.9999999814 | cap binding almost exactly at `sqrt(precision_lr_max)=1.0` — Stage 0 passes cleanly |
| `pr_p50` | 3.68 / 4 | see the correction below |
| `pr_p05` | 2.50 | |
| `pr_p95` | 3.91 | |

Stable to 0.2% across seven checkpoints from 86201 to 96410, so this is a
converged property rather than a spike-time artifact.

> **Correction (2026-09-13). The "saturated → rank 8" reading here was wrong
> on two counts, and the rank-8 experiment is no longer indicated.**
>
> **1. Wrong null.** `pr_p50 = 3.68` was read as saturated because it clears
> $0.75r = 3.0$. But a random $d \times r$ Gaussian is already near-flat: an
> **untrained** model on this architecture scores **3.96**. Measured
> accidentally, when a silent auto-resume failure left the model at
> initialisation. Training moves PR *down* from that null, so 3.68 is a
> concentration of 0.28 away from unstructured, not evidence of a budget
> straining against its ceiling.
>
> **2. Pooling hid the structure.** The per-site view
> (`stiffness.sigma_lr_spectrum_by_site`, step 96410, 40 sites) gives
> `between_frac = 0.39` — substantial heterogeneity — carried by **channel**
> (means 3.18, 3.74, 3.82, 3.64, 3.61; range 0.65) far more than by layer
> (range 0.33). And $\lVert B_k \rVert_F$ reads **1.000 at every one of the
> 40 sites**, so the cap binds uniformly and those PR differences are pure
> redistribution — no channel is idle.
>
> A global rank 8 would raise every channel alike, leaving roughly five
> directions idle in channel 0 to serve channel 2, at **+33%** of total
> parameters. A per-channel allocation targets the same hypothesis for about
> **+1.7%**. See Stage 4 in
> `Curvature_Diagnostics_and_Rank_Selection_for_Aniso_Gaussian_Vtheta.md`,
> now indicated rather than hypothetical.
>
> **Methodology note.** Run the by-site probe, not the pooled one, and load
> the checkpoint's weights explicitly with `_load_weights_into` rather than
> relying on whatever is in `model` — an entire afternoon of measurements in
> this session was taken on randomly-initialised weights after Cell 2's
> auto-resume failed without erroring.

**Stage 2 — rank-truncation ablation, built 2026-09-13** (was `PROPOSED`
in the companion note's §7.3; now implemented as
`precision_cap.replay_rank_truncation_ablation`):

```python
replay_rank_truncation_ablation(87196, ranks=(1, 2, 3, 4))
```

**Result (A100, 2026-09-13).** `rank=4` reproduced the untruncated
reference exactly (2539.20, error 0.0000), so the built-in fidelity check
passes and the harness is sound.

| arm | pre-clip grad norm | ntp | batch PPL |
|---|---|---|---|
| untruncated | 2539.20 | 4.3385 | 76.6 |
| rank = 4 | 2539.20 | 4.3385 | 76.6 |
| rank = 3 | 2.83 | 4.3808 | 79.9 |
| rank = 2 | 4.35 | 4.4682 | 87.2 |
| rank = 1 | 2.87 | 4.6609 | 105.7 |

Two readings, and only the first is about rank.

**On rank:** read the `ntp` column, not `relative_force_error` — the latter
saturated at 1.0 across all three arms because the gradient collapsed, at
which point it is fixed by the norm ratio alone and carries no directional
information. `ntp` orders correctly and shows every direction paying its
way: discarding even the *smallest* of four costs 4.3% batch perplexity,
rising to 13.8% and 38.0%. The budget is not idle, consistent with Stage
1's saturated `pr_p50 = 3.68/4`. This argues against reclaiming parameters
by cutting rank, and weakly for Stage 3.

**On the spike:** any truncation collapses the gradient from 2539 to about
3 — a normal, healthy value — for a 4.3% loss cost, and no further with
deeper truncation. That looked like threshold behaviour, and was tracked as a resonance
hypothesis — **now refuted, and the actual cause established
(2026-09-13): a spike is a conjunction between one microbatch and the
routing draw it receives.** Resetting the RNG per microbatch collapses
every captured spike to baseline (87196: 2539→2.71; 86201: 686→1.19;
90360: 567→2.15), and 20 alternative draws on the spiking batch all land
160x below the real one. Containment (clip + watchdog) is therefore the
correct response and the rank question is decoupled from spikes entirely.
See [Gradient_Spikes_as_Routing_Conjunctions.md](Gradient_Spikes_as_Routing_Conjunctions.md).
The superseded resonance analysis is in
[Resonance_Hypothesis_for_Gradient_Spikes_in_the_LowRank_Kick.md](Resonance_Hypothesis_for_Gradient_Spikes_in_the_LowRank_Kick.md),
which carries the mechanism and its diagnostic programme.

**Caveat — run the control before drawing either conclusion too hard.**
Truncation confounds removing specific directions with changing `B` by a
given magnitude. `precision_cap.replay_rank_perturbation_control(ctx,
87196, ranks=(1, 2, 3))` holds the magnitude and drops the specificity
(isotropic noise of matched energy, rank left intact). If the spike dies
under that too, the collapse is generic fragility and says nothing about
direction. Also note step 87196 is an outlier by construction, so the rank
question itself wants re-asking at a healthy checkpoint.

### 2.2 Spectral-collapse test + mechanism comparison

```python
spectrum_across_checkpoints(step_tags=(87196, 85885, 90360, 95068, 95091, 95280))
```

Does PR drop (spectral collapse) at the spike bundles vs. the final
`_best.pt` (step 96,000)? Does it drop differently for the register-led
events (87196, and the Tier B cluster) vs. the `reverse_channel_scale`-led
ones (85885, 90360)?

### 2.3 Precision cap ablation — does capping the low-rank channel alone collapse it?

```python
for s in (87196, 85885, 90360, 86201, 95068, 95091, 95280):
    replay_precision_cap_ablation(s, budgets=(1.0, 0.25, 0.1, None))
```

`1.0` is the as-trained baseline (fidelity check: should match the
recorded pre-clip norm almost exactly). Watch `vtheta_exponent_min` too,
not just the collapsed norm — that's the un-saturation read. If 96410 was
rescued per §1, add it here too — it's the cleanest single-mechanism case
in the Tier B cluster.

### 2.4 Curvature rebalance — diagonal vs. low-rank channel

```python
for s in (87196, 85885, 90360, 95068, 95280):
    replay_curvature_rebalance_ablation(s)
    # sweeps precision_max x precision_lr_max, default grids
```

### 2.5 Classify each event (smooth cascade vs. localized blow-up)

```python
for s in (87196, 85885, 90360, 86201, 95068, 95091, 95280):
    replay_spike_batch(s)
```

### 2.6 Register mechanism — decoupled from V_theta, or just riding it?

**Do this FIRST. It is free, offline, and covers every spike in the whole
run** — not just the four replayable bundles. `training_log.jsonl` records
`top_groups` on every `grad_spike` (top 8) and `watchdog_hard_reload`
(top 5) event, so the question is answerable by parsing the log: no
bundles, no GPU, no checkpoint loads, and it still works for spikes whose
bundles rotated out of the ring long ago.

**The discriminator.** In a V_theta-driven cascade `override:register` comes
out roughly V_theta-sized — that is mechanism B riding mechanism A
(Mitigations §42). A genuinely register-specific event breaks that coupling.

**Calibrated 2026-09-11 against 88 real spikes** (the archived
`training_log.jsonl`, steps 50–56,300 of this same run). This replaced an
earlier eyeballed rule that was measured on only 10 hand-transcribed
captures and turned out to be badly miscalibrated — see the warning below.

**Re-run this now that the run is finished.** The fit below covers only
the first 56,300 of 100,000 steps and predates `clip_then_sum`/
`NO_DECAY_1D` (see the regime caveat further down) — it is not the right
distribution to score the Tier B cluster (§1) against. The manually
computed ratios for 95068/95091/95280/96410 are a strong hint, not a
verified z-score. Re-running
`register_vtheta_ratio_sweep.py` against the **complete** final
`training_log.jsonl` gives the authoritative fit and should be the actual
first thing run in this section, ahead of everything below it.

| statistic over 88 events | value |
|---|---|
| median | **0.65** |
| p75 / p90 / p95 | 1.16 / 2.50 / 2.88 |
| geometric mean, log-sd | **0.81**, **0.82** |
| full range | 0.05 – 20.03 |

The ratio is **log-normally distributed with a fat right tail**, so score it
in log space:

| threshold | value | events flagged |
|---|---|---|
| +2sd | 4.15 | 3 of 88 (3.4%) |
| +3sd | 9.42 | 1 of 88 (1.1%) |

> **Do not use a multiple of the median.** The rule originally written here
> — "more than 3x the median" — is *not* an outlier test on this
> distribution: 3 x 0.65 = 1.95 lands between p75 (1.16) and p90 (2.50), and
> flagged **12 of 88 events (14%)**. That is the fat tail, not an anomaly.

**What the real data showed:**

| step | reg / V_theta | z |
|---|---|---|
| **47142** | **20.03** | **+3.92** — the strongest register decoupling on record |
| 41824 | 9.05 | +2.95 |
| 51426 | 4.78 | +2.17 |
| *87196 (later window)* | *8.59* | *+2.42 against this fit* |

**This weakens the "87196 is the register event" framing.** It is a genuine
tail event, but it is **not unprecedented** — step 47142 was more than twice
as extreme, in an earlier regime, and 47116 (26 steps away, same cluster)
already has archived golden outputs. Treat 87196 as *one* member of a
recurring tail, not as a unique signature.

**Caveat on regime.** Those 88 events predate three mitigations that landed
later: `clip_then_sum` (Mitigations §45.4), the `log_tau` clip-group split
(§49.8, around step 71K) and `NO_DECAY_1D` (§51.4, around step 72K). The central
tendency did shift — median 0.65 there vs ~1.0 across the 10 captures in
86,500–90,450 — so **re-fit on the full post-100K log** rather than reusing
these constants. The log-space method is what carries over; the numbers are
a baseline, not a law.

```bash
python3 ../notebooks/conservative_arch/scaleup/debug/register_vtheta_ratio_sweep.py \
    <RESULTS_DIR>/training_log.jsonl
```

([`register_vtheta_ratio_sweep.py`](../notebooks/conservative_arch/scaleup/debug/register_vtheta_ratio_sweep.py)
lives in the notebook's own `debug/` folder alongside its sibling diagnostic
scripts, not next to this note. It fits the log-normal and reports z-scores
itself, and de-duplicates steps that emit both a `grad_spike` and a
`watchdog_hard_reload` record — counting those twice skews the fit.)

**Decision rule:**

- **Nothing exceeds +2sd** → register is V_theta-slaved everywhere, there is
  no register-specific mechanism, and no register knob worth tuning. Skip the
  probes below; put the effort into 2.3/2.4.
- **Some events exceed +2sd** → those steps *are* the register
  investigation. Run the probes below on those steps only, and prefer the
  highest-z one — which on current evidence may well not be 87196: the
  Tier B cluster (95068/95091/95280/96410, §1) is denser than any single
  earlier event and should re-rank near the top once scored against the
  full-run fit.

**Check this claim while you are here:** §1's table says the register-led
magnitude is escalating (567.6 → 2169.2 from 86201 to 87196). That rests on
raw magnitude alone — V_theta for 86201 was never recorded in the pasted
window, so its ratio is unknown. If 86201 comes back near ~1.0, then it was
an ordinary V_theta cascade, 87196 is a lone outlier rather than the second
point of a trend, and the "escalating register mechanism" reading is wrong.
The Tier B cluster is independent evidence either way: four events in
1,300 steps is a *frequency* signal, not just a magnitude one, and doesn't
depend on how 86201/87196 resolve.

**Then, only for the steps the ratio flagged**, read across registers —
not rows (§49.4's lesson, don't repeat the row-axis mistake):

```python
for s in (87196, 95068, 95091, 95280):
    probe_gate_saturation(s)
attribute_spike_rows(95280)   # caveat: unreliable on chronic mechanism-A
                               # events, SS41.5 — ranking survives, magnitude doesn't
```

Note: `probe_gate_saturation`'s `focus=` param (default `FOCUS_REGISTER=14`,
a leftover from the log_tau investigation) only adds a highlighted line for
that one register — it does not restrict the computation. The printed
top-8 ranking already covers whichever register(s) actually matter here;
don't assume it's still register 14 without checking that ranking.

If this doesn't resolve it, the register mechanism may need its own
`sweep_register_embed_history`-style tool, mirroring `sweep_log_tau_history`.

**Knobs, if and only if the ratio test says register is real** (none of
these is `LAMBDA_FOCK_REG` — that is a log-barrier on the 5 xi-channel EMA
decay rates and touches nothing in the register path):
`REGISTER_REPULSION_COEFF` (0.05) and `REGISTER_REPULSION_KIND`
('gram' → 'coulomb'); register capacity `n_registers`=32 / `TOP_K`=16;
`GRAD_CLIP_OVERRIDES['register']` (0.3). Current repulsion health: logged
`rep` 0.0036–0.0047 implies RMS inter-register cosine ~0.27–0.31, against
the model's own collapse flag at 0.6 — registers are **not** collapsing, so
raising the repulsion coefficient would be treating an absent disease.

### 2.7 Only if 2.3 is ambiguous (expensive — minutes per call)

```python
replay_integrator_ablation(87196, lowrank_layers=frozenset({0, 1, 2}))
```

---

## 3. What the results decide

- **2.3 collapses all four events** → confirms mechanism A (chronic
  low-rank stiffness) as the shared root cause behind both named
  mechanisms, same as §42's finding that "mechanism B rides on mechanism
  A." Prioritize the curvature-rebalance / flatness-incentive work over a
  register-specific fix.
- **2.6's ratio sweep finds nothing above 3x median** → register is
  V_theta-slaved across the entire run; there is no third mechanism and no
  register knob worth turning. Strongest single piece of evidence available,
  and it costs nothing to get.
- **2.6 flags 87196 (± others) AND 2.3 collapses `register`-led events less
  cleanly than `reverse_channel_scale`-led ones** → the two independent
  tests agree, and register is a genuinely separate third mechanism. Only
  then is the register_embed/`REGISTER_REPULSION` deep-dive justified.
- **The two disagree** (ratio flags 87196 but the cap collapses it anyway,
  or vice versa) → trust the cap ablation: it is causal (re-runs the step
  under a changed constraint), the ratio is only correlational.
- **2.1's rank verdict is in: saturated** (`pr_p50=3.68/4`, `pr_p05=2.50`).
  The rank-truncation ablation (also §2.1) is the tie-breaker on whether
  that's functionally real: if `relative_force_error` stays near 0 well
  below rank 4, the saturation is geometric only and rank should stay at 4
  (or drop); if it grows steadily down to rank 1, the joint-coupling pilot
  (already built, `K=8` parameter-matched) should carry a rank increase
  too, not just the coupling change.

---

## 4. Lower priority / optional

- `bracket_precision_lr_max()` — cross-check ambient stiffness, healthy vs.
  spike-regime, if 2.3's picture is unclear.
- `_load_bottleneck_mod` (Cell 6c) — GPU-compute vs. CPU/launch step-time
  profiling. Unrelated to the stability question; only relevant if
  wall-clock/step suddenly looks off.
- Cell 7 component-health probe (`_mk`/`_eval_on`) — general "is each Fock
  piece being used well" sanity check, not spike-specific.

---

## 5. After this analysis

If 2.1–2.7 don't change the plan: proceed to the **joint-coupling + QK-norm
pilot**
([`colab_fock_cfc_baoab_joint_vtheta_qknorm_openwebtext_d384.ipynb`](../notebooks/conservative_arch/scaleup/colab_fock_cfc_baoab_joint_vtheta_qknorm_openwebtext_d384.ipynb)),
15,000-step probe stop already set, before committing the 150K continuation
on either arm.
