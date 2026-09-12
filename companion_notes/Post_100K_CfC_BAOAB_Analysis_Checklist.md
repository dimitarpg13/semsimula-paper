# Post-100K analysis checklist — CfC/BAOAB d384 run

Personal working note, not for publication. What to run once the current
100,000-step run (`fock_cfc_owt_..._baoab_cfc`) finishes, before deciding on
the joint-coupling/QK-norm pilot vs. continuing the additive arm.

Full background lives alongside this file in `companion_notes/`:
[`Diagnostic_Programme_in_CfC_BAOAB_Integrator.md`](Diagnostic_Programme_in_CfC_BAOAB_Integrator.md) §17,
[`Curvature_Diagnostics_and_Rank_Selection_for_Aniso_Gaussian_Vtheta.md`](Curvature_Diagnostics_and_Rank_Selection_for_Aniso_Gaussian_Vtheta.md),
[`CfC_BAOAB_Integrator_and_Mitigations.md`](CfC_BAOAB_Integrator_and_Mitigations.md) §41-42, 49-51.

---

## 0. Before running anything

- **Run only after the run ends** — don't interrupt it for this.
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

| step | pre-clip grad | leading group | why it matters |
|---|---|---|---|
| **87196** | 2539.2 | `register` (85% of total) | most extreme event on record; `register`-led magnitude is **growing** (567.6 → 2169.2, 3.8x, vs step 86201) |
| **85885** | 2090.8 | `reverse_channel_scale` | most extreme `reverse_channel_scale`-led event |
| **90360** | 567.3 | `reverse_channel_scale` | same mechanism as 85885, but **shrinking** (2014.1 → 490.8) — the contrast case |
| **86201** | 685.6 | `register` | first of the register-led pair; use alongside 87196 for the escalation check |

Secondary set (lower severity, wider spread — useful only if the above four
don't give a clean read): **81647** (269.6), **82660** (220.5), **81393**
(173.6).

**Added 2026-09-11 from the §2.6 ratio sweep over 88 archived spikes:
step 47142** is the strongest register/V_theta decoupling on record
(ratio 20.03, z=+3.92 — more than twice 87196's 8.59), with **41824**
(9.05) next. Both predate the later mitigations, so they are a different
regime, but if the register question survives §2.6 these are the sharpest
examples available — and step 47116, 26 steps away in the same cluster,
already has archived `replay_spike_batch` / `attribute_spike_rows` golden
outputs to compare against.

**Bundle availability, checked 2026-09-11.** The live ring has rotated far
enough that the oldest surviving archived spikebatch is **step 77,223** —
everything older is gone, including 70,522 / 71,194 / 71,703 (so the §16
golden outputs for those steps can no longer be reproduced, only read).
Of the §1 set:

| step | spikebatch (replayable) | prereload (weights only) |
|---|---|---|
| 87196 | **yes** | yes |
| 90360 | **yes** | yes |
| 86201 | yes | yes |
| 85885 | **NO — evicted** | yes |

So **85885 can only be used by `spectrum_across_checkpoints` (§2.2)**; it
cannot take part in §2.3 / §2.4 / §2.5, which all need the batch and RNG.
Use 90360 as the `reverse_channel_scale` case for those.

**Weights-only checkpoints** (no batch/RNG, `spectrum_across_checkpoints`
only, not the `replay_*` ablations):
- `_best.pt` — whatever the final best is when the run ends (currently step
  83,000 / PPL 82.87, may improve further before 100K).
- `_prereload.pt` for 85885 / 86201 / 87196 / 90360 (same step numbers,
  saved automatically at each hard-trigger).

---

## 2. Run in this order

> **Except**: §2.6's `register_vtheta_ratio_sweep.py` needs no GPU, no
> bundles and no checkpoints — only `training_log.jsonl`. Run it before
> anything else here; its result decides whether the register branch is
> worth any of the work below.

### 2.1 Rank decision (free, no bundle needed)

```python
spectrum_across_checkpoints()   # defaults to _best.pt
```

Read `fro_p50` first — confirms the Frobenius cap is actually binding
(should sit near `sqrt(precision_lr_max) = 1.0`). If it doesn't, stop: the
rank argument below doesn't apply yet.

Then read `pr_p50` (participation ratio, out of rank=4):
- **≥ 3.0** → budget saturated, rank 8 has a real case
- **≤ 2.0** → budget unused, rank 8 would be wasted params — prefer a
  flatness incentive or drop to rank 2 instead
- **2.0–3.0** → ambiguous, weigh against step 2.2 below

### 2.2 Spectral-collapse test + mechanism comparison

```python
spectrum_across_checkpoints(step_tags=(87196, 85885, 90360))
```

Does PR drop (spectral collapse) at the spike bundles vs. `_best.pt`? Does
it drop differently for the escalating (`register`) vs. non-escalating
(`reverse_channel_scale`) mechanism?

### 2.3 Precision cap ablation — does capping the low-rank channel alone collapse it?

```python
for s in (87196, 85885, 90360, 86201):
    replay_precision_cap_ablation(s, budgets=(1.0, 0.25, 0.1, None))
```

`1.0` is the as-trained baseline (fidelity check: should match the
recorded pre-clip norm almost exactly). Watch `vtheta_exponent_min` too,
not just the collapsed norm — that's the un-saturation read.

### 2.4 Curvature rebalance — diagonal vs. low-rank channel

```python
for s in (87196, 85885, 90360):
    replay_curvature_rebalance_ablation(s)
    # sweeps precision_max x precision_lr_max, default grids
```

### 2.5 Classify each event (smooth cascade vs. localized blow-up)

```python
for s in (87196, 85885, 90360, 86201):
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
  highest-z one — which on current evidence may well not be 87196.

**Check this claim while you are here:** §1's table says the register-led
magnitude is escalating (567.6 → 2169.2 from 86201 to 87196). That rests on
raw magnitude alone — V_theta for 86201 was never recorded in the pasted
window, so its ratio is unknown. If 86201 comes back near ~1.0, then it was
an ordinary V_theta cascade, 87196 is a lone outlier rather than the second
point of a trend, and the "escalating register mechanism" reading is wrong.

**Then, only for the steps the ratio flagged**, read across registers —
not rows (§49.4's lesson, don't repeat the row-axis mistake):

```python
probe_gate_saturation(87196)
attribute_spike_rows(87196)   # caveat: unreliable on chronic mechanism-A
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
- **2.1's rank verdict** feeds directly into whether the joint-coupling
  pilot (already built, `K=8` parameter-matched) should also carry a rank
  change, or whether rank should be left alone and only coupling tested.

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
