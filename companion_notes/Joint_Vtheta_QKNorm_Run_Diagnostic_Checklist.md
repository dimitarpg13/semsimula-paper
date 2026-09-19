# Diagnostic checklist — joint-V_theta + QK-norm d384 run

Personal working note, not for publication. What to run against the
joint-coupling arm (`fock_cfc_owt_..._vtjoint_cgqk_L8probe_..._baoab_cfc`)
once GPU time is free. Deliberately deferred: every probe here is a
deterministic function of saved checkpoints, so none of it competes with
training for a Colab session.

**Run status (updated 2026-09-16).** The 50,000-step probe **concluded**.
`TOTAL_STEPS = 100,000` (WSD: warmup 0→5,000, stable 5,000→65,000, decay
65,000→100,000, floor 1.50e-05). Best PPL **84.31 at step 28,500**; final
eval **93.93 at step 50,000**. Four `[spike]` events fired (steps 39,206 /
39,521 / 40,075 / 41,135); **zero `[watchdog]` or reload events**.

**The run stopped improving and began degrading.** Windowed mean PPL:
89.76 → 88.45 → 87.88 → 89.15 → 88.92 → 90.25 across 5,000-step windows
from 20,000 — bottoming at 30-35K and rising after. Slope from step 34,000
is **+0.159 ± 0.065 PPL / 1,000 steps (t = +2.45)**: significant
degradation, not noise. Diagnosis, from the telemetry:

- **Not overfitting.** Train `ntp` and val loss agree within ±0.01 nats in
  every window, and *both* stopped falling. Nothing to overfit to at 0.41
  epochs of a 2B pool.
- **Not capacity.** 76.8M params × 0.819B tokens = **10.7 tokens/param**
  against Chinchilla-optimal ≈20 — 53% of compute-optimal, under-trained.
- **Not data volume.** Train loss is flat, so more data cannot help a model
  that is not fitting what it already has.
- **Optimization.** Median grad norm doubled (0.82 → 1.65), p99 grew 14x
  (5.73 → 80.27), `bproj_sig` kept climbing (22.12 → 34.10) — all while
  `lr` sat at 3.00e-04 for 15,400 steps. A fixed step size gone too hot for
  a sharpening landscape.

**Next action: the anneal probe** (see §6), branching from
`_step50000_probe_stop.pt` to test whether decaying the LR unsticks it.

Background lives alongside this file in `companion_notes/`:
[`Post_100K_CfC_BAOAB_Analysis_Checklist.md`](Post_100K_CfC_BAOAB_Analysis_Checklist.md)
(the additive arm's equivalent — many probes here are its counterparts),
[`Gradient_Spikes_as_Routing_Conjunctions.md`](Gradient_Spikes_as_Routing_Conjunctions.md),
[`Analytic_Multi_Channel_Integration_in_Structured_Vtheta.md`](Analytic_Multi_Channel_Integration_in_Structured_Vtheta.md) §5,
[`Register_Temperature_Instability_in_the_Fock_Creation_Gate.md`](Register_Temperature_Instability_in_the_Fock_Creation_Gate.md).

---

## 0. Before running anything

- **Cell order for a diagnostics-only session**: Cell 0 → 1 → 1c → 2 → 3 →
  4 → 5 → 6b → 6b-2 → 6b-3 → 6b-4. Cell 1c must run (it archives the
  spikebatch/prereload rings before rotation; cheap, idempotent). Cells
  6d-2/6d-3/6d-4 were **deleted from this notebook on 2026-09-14** — they
  were hardcoded to the additive arm's steps 70522/71194/71703, register 14,
  and `model_fock_parf_v2`, and 6d-4 probed `log_tau`, which does not exist
  under QK-norm (`[tau-floor] no creation_gate_qkv.log_tau on this model`).
- **Cell 6d's helpers need Cell 6 to have at least started.** Its own
  comment is explicit: `replay_spike_batch` / `attribute_spike_rows` look up
  Cell 6 globals (`forward_with_vreg`, `LAMBDA_V`, `_GRAD_CLIP_CFG`,
  `WATCHDOG_EXCLUDE_GROUPS`, `CKPT_DIR`/`CKPT_PREFIX`) **at call time**. A
  session that skips Cell 6 entirely can *define* them but not *call* them.
  Verify this before relying on it — the additive arm's checklist §0 says to
  skip Cell 6, which may only have worked because nothing there called 6d.
- **Verify what is actually loaded before trusting any live-model probe.**
  Cell 5 builds the model but does not load weights by itself. Confirm the
  resume step printed by Cell 2. This has already cost a full round of
  measurements once (a silent Cell 2 resume failure produced a
  random-init model that read as "homogeneous" — Mitigations §42.5).
- **Remember the PR null.** Participation ratio must be read against the
  measured random-init null (**3.96** at rank 4), not against `r`. An
  untrained model already scores near the ceiling.

---

## 1. What exists, and what is blocked

**Checkpoints available** (Drive, `checkpoints/`): every 500-step eval that
set a new best, plus `_step15000.pt`, `_step15000_probe_stop.pt`, the 23.5h
autosave at **step 34,156**, and `_best.pt` (PPL 84.31, step 28,500).

**Spikebatch bundles now EXIST — the bundle-dependent probe family is
unblocked.** Four captures, but only two are permanent:

| step | pre-clip grad | top group | archived? | availability |
|---|---|---|---|---|
| 39,206 | 202.2 | `reverse_channel_scale` 286.6 | yes | **permanent** (`spikebatch_archive`) |
| 39,521 | **479.9** | `reverse_channel_scale` 321.2 | yes | **permanent** (`spikebatch_archive`) |
| 40,075 | 102.4 | `reverse_channel_scale` 125.9 | no | **transient** — live ring only |
| 41,135 | 163.6 | — | no | **transient** — live ring only |

**Why only two archived.** `ARCHIVE_MIN_GRAD_NORM` is
`2 * CAPTURE_SPIKE_THRESHOLD` = 200.0, which deliberately filters the
"routine 100-190 band" to bound Drive usage. This is by design, not a fault. The two
unarchived bundles are still in `CKPT_DIR` (the live ring keeps 12 and only
4 were captured) but **will rotate out once 12+ more captures accumulate** — roughly 30,000 further steps at the observed rate of 1 per
≈3,850. If 40,075 or 41,135 is wanted, copy it out of the live ring before
then, or lower `ARCHIVE_MIN_GRAD_NORM` before the next long run.

All four are `reverse_channel_scale`-led, which is itself a finding: the
earlier grad-norm cluster (steps 23,000-28,700) was `depth_code`-led, so the
dominant group **changed** as the run progressed.

**Also runnable** (take a batch `x` or checkpoint tags, not bundles):
`sigma_lr_spectrum_by_site`, `sigma_lr_spectrum_report`, `sigma_lr_report`,
`stiffness_report`, `spectrum_across_checkpoints`, `bracket_precision_lr_max`.

**Note on the anneal probe and the ring.** The anneal redirects `CKPT_DIR`
to `anneal_probe/`, so captures during it land there and cannot evict the
main ring's four bundles. They are safe for the duration of that probe.

---

## 2. Run in this order

### 2.1 Is `||B_k||_F` pinned at the `PRECISION_LR_MAX = 1.0` cap? — **PROPOSED**

Highest priority, free, no bundle. This decides whether cap-tightening is
even an available lever.

```python
sigma_lr_spectrum_by_site(ctx, x)     # prints per-site ||B||_F and PR
```

**Why it matters.** `omega*dt = sqrt(lambda_max(L)/m)` with
`L = sum_k g_k B_k B_k^T`. If `||B_k||_F` is pinned at 1.0 everywhere, B's
scale is *fixed*, and the rigid upward shift of the `omega*dt` distribution
(§3) cannot be coming from B's magnitude — it must come from `g_k`
concentration, from `B_k` alignment across wells, or from `m`. In that case
lowering `PRECISION_LR_MAX` buys nothing, and the lever has to act on the
well weights or the bank geometry instead.

**Prior.** On the additive arm this read **1.000 at every one of 40 sites** —
the cap binding uniformly. This arm has **8 sites, not 40** (one joint bank
of K=8, versus five additive banks of 8), so do not assume the same answer.

**Decision rule:**
- all sites ≈ 1.000 → cap is binding; `PRECISION_LR_MAX` is NOT the lever;
  go to §2.5 (decomposition) to find what is.
- sites below 1.000 → cap is slack; tightening it directly targets the
  `omega*dt` shift, and `bank._precision_lr_max` is hot-swappable mid-run
  (a live Python attribute, not part of `state_dict`).

### 2.2 Per-site rank structure of the joint bank — **PROPOSED**

Same call as §2.1; read the `between_frac` and the channel-vs-layer range.

**What is being tested.** On the additive arm, `between_frac ≈ 0.39-0.41`
("structured"), carried by **channel** (range 0.65) more than layer (range
0.33). The joint bank has no per-channel sites at all — each well is a joint
function of all 5 channels — so the channel axis of that heterogeneity
*cannot* exist here by construction. Whether the layer axis alone still
produces structure is a direct, architectural test of what the additive
arm's heterogeneity was actually made of.

Read PR against the **3.96** random-init null, not against `r = 4`.

### 2.3 Trajectory across checkpoints — **PROPOSED**

```python
spectrum_across_checkpoints(ctx, step_tags=(15000, 20000, 25000, 28500, 34156))
```

When did `||B||_F` reach the cap, and does PR move as the `omega*dt`
distribution shifts? If PR falls while `||B||_F` stays pinned, the shift is
alignment/concentration — which §2.1's decision rule routes to.

### 2.4 `bracket_precision_lr_max` — **PROPOSED**

What budget would actually bind, given current ambient `sigma_max(B_k)^2`.
On the additive arm the healthy and spike-regime checkpoints had
*statistically similar* ambient values (p50 ≈280-310 across all four), so no
tight tail-only budget existed there. Worth re-checking here, because this
arm has run with the cap on from step 0 rather than switching it on at 47K.

### 2.5 `omega*dt` decomposition — **NEEDS NEW TOOLING**

Not currently in `semsimula-diag`. The question §2.1 routes to: of the three
factors in `omega*dt = sqrt(lambda_max(sum_k g_k B_k B_k^T)/m)`, which one
is moving? Candidates, in order of suspicion:

1. **`g_k` concentration** — well weights piling onto fewer wells, so the
   sum stops averaging and starts adding coherently.
2. **`B_k` alignment** — the per-well factors rotating into a shared
   subspace, so `lambda_max` of the sum grows even at fixed `||B_k||` and
   fixed `g_k`.
3. **`m`** — mass shrinking. Cheapest to rule out; check first.

A minimal probe logs, at one checkpoint: the `g_k` distribution (entropy /
top-1 share), the pairwise principal angles between the `B_k` column spaces,
and `m`'s percentiles. Compare against the step-15,000 checkpoint to get a
direction of travel.

### 2.6 Bundle-dependent probes — **UNBLOCKED, 2026-09-16**

Use `step_tag=39521` first (pre-clip 479.9, the most severe) and `39206`
(202.2) as the confirmation; both are permanently archived. `40075`/`41135`
are transient — see §1 before relying on them.

Run in this order, highest value first:

1. **`replay_rank_perturbation_control`** — the RNG-reset test. **This is
   the one genuinely open question.** The routing-conjunction finding
   (spikes are a microbatch × RNG-draw coincidence, not curvature) was
   established *on the additive arm*. If the joint arm's first spike
   collapses under RNG reset the same way (additive saw 264x-937x across
   three checkpoints), the finding generalises and joint coupling introduces
   no new mechanism. If it does *not* collapse, joint coupling has
   introduced a genuinely curvature-driven spike — which would be the first
   evidence for anything like the (otherwise falsified) resonance
   hypothesis, and would change what the `omega*dt` wall crossings mean.
2. **`omega_dt_report`** on the bundle — was the spiking step's `omega*dt`
   distribution anomalous versus ambient, or ordinary? Given the rigid-shift
   finding (§3), the prediction is **ordinary**.
3. **`omega_dt_under_truncation`** — does the truncation that kills the
   spike also carry `omega*dt` back under the wall? (D2b.)
4. **`replay_precision_cap_ablation`** — does the cap collapse this arm's
   spike, as it did all three of the additive arm's?
5. **`tail_coherence_report`**.

---

## 3. Predictions recorded in advance, for the 50,000-step checkpoint

Stated **before** the data, so the next log is a test rather than a story.
All fits are on steps 34,600 and earlier.

| quantity | prediction at 50,000 | **actual** | verdict |
|---|---|---|---|
| `omega*dt` p50 | 1.16 | **1.128** | held |
| `omega*dt` max | 2.51 | **2.435** | held |
| `over_wall` | ≈0.3% | **0.423%** | high, under the 0.5% falsifier |
| log-normal σ | ≈0.20 | **0.217** | held (under the 0.22 falsifier) |
| grad p99 crosses 100 | step ≈45,300 | **step 39,206** | ≈6,000 steps early |
| grad median | ≈1.4 | **1.65** | close |
| val PPL | **≈75** | **93.93 (best 84.31)** | **FALSIFIED** |

**Scored 2026-09-16. The pattern in the misses is the lesson.** Every
*saturating* (logarithmic) fit held — the `omega*dt` family was predicted
within 3%. Both fits that assumed *continued improvement* failed, and the
PPL one failed badly because it extrapolated a monotonic power law straight
through a turning point that the windowed means place at step 30-35K. The
grad-tail exponential was directionally right but too slow, meaning real
growth in that window was faster than exponential.

**Rule for next time:** before extrapolating any trend here, check whether
its growth rate is decaying. Saturating fits have been reliable; "this keeps
going" fits have not.

**The load-bearing claim, and how to falsify it.** `omega*dt` is *saturating*
(logarithmic, growth rate already decayed 4.4x from +0.034 to +0.0077 per
1,000 steps) and the distribution is shifting **rigidly**, not widening
(σ +3.7% over 11,000 steps). Under that reading, p50 reaches only ≈1.39 by
step 100,000 and would not touch 2.0 until **step ≈663,000** — and the rising
`over_wall` is a pure threshold-crossing artifact of a rigid shift past a
fixed cut, not a runaway.

**This is falsified if**, at 50,000: σ has grown materially past ≈0.22 (the
distribution is widening, not just shifting), or p50 overshoots ≈1.2 (the
log fit is wrong and growth is not saturating), or `over_wall` substantially
exceeds ≈0.5%.

**The grad-norm tail is the trend that does NOT saturate**, and is the real
thing to watch. Note these are *pre-clip* norms — the per-group clips
(0.1-0.3 on the implicated groups) are what protect the update, and they are
evidently working: PPL is on trend with zero watchdog events. Two distinct
signatures so far, not one: `depth_code`-led (steps 23,000-28,700, up to
9.43) and `reverse_channel_scale`-led (steps 33,200-34,600, up to 15.36).

---

## 4. What the results decide

- **§2.1 = pinned** → `PRECISION_LR_MAX` is spent as a lever; any mitigation
  has to act on `g_k` or bank geometry (§2.5). Do not waste a run lowering
  the cap.
- **§2.1 = slack** → lowering the cap is available and hot-swappable
  mid-run; bracket it with §2.4 before choosing a value.
- **§2.6.1 collapses under RNG reset** → joint coupling introduced no new
  spike mechanism; the routing-conjunction finding generalises; the
  `omega*dt` crossings are a stability curiosity rather than a cause, and
  the grad tail is the only thing worth managing.
- **§2.6.1 does NOT collapse** → this is the significant outcome. Joint
  coupling has a curvature-driven spike mechanism the additive arm did not,
  and §5 of the multi-channel note (fusion concentrates precision into
  exactly the operator the `omega*dt` wall constrains) is vindicated as a
  live risk rather than a hedge.
- **Predictions in §3 hold** → the saturating reading is right; the run can
  go to 100,000 without `omega*dt` intervention.
- **Predictions in §3 fail** → re-derive before acting; do not patch the fit.

---

## 5. Lower priority / optional

- Register dominance is **not** the clean lock reported after the 15K pilot.
  Mapped precisely: r14 held 7,350→19,300, then **r9 took over for
  19,900→22,950** (≈3,050 steps), oscillating between the two in 150-400
  step bursts since, with an r18 blip at 34,450. Worth a proper
  per-register gate-scale attribution if it ever settles permanently, but
  handing off is the healthy behaviour and it is handing off.
- `bproj_sig` is growing but **decelerating** (7.29 → 25.77 across the run;
  increments +6.33, +4.17, +2.20, +2.13, +1.88, +1.77 per 5,000-step
  window). Consistent with the saturating picture; no action.
- Cosmetic: the final log line prints
  `next_step=150000 (== TOTAL_STEPS means training is fully done)` after a
  `PROBE_MAX_STEPS` stop, which is misleading — the halt is correctly
  reported one line earlier. Harmless; fix if the cell is being edited
  anyway.

---

## 6. The anneal probe — **RUN, SUCCEEDED** (2026-09-17)

**Notebook:** [`colab_fock_cfc_baoab_joint_vtheta_qknorm_annealed_after_50K_openwebtext_d384.ipynb`](../notebooks/conservative_arch/scaleup/colab_fock_cfc_baoab_joint_vtheta_qknorm_annealed_after_50K_openwebtext_d384.ipynb)
(committed 2026-09-17). Run order: Cell 0 → 1 → 1c → 2 → 3 → 4 → 5 →
the `import shutil` redirect cell → Cell 6. Skip 1b (no-op) and 6d.

Tests whether the plateau is an LR/curvature mismatch or something deeper.
Branches from `_step50000_probe_stop.pt` and runs a **compressed** version
of the real decay — same start, same floor, same cosine shape, 8.75x faster.

```python
ANNEAL_PROBE     = True        # short-circuits lr_schedule entirely
ANNEAL_FROM_STEP = 50_000
ANNEAL_STEPS     = 4_000       # ~4.7h at 4.2 s/step
ANNEAL_LR_START  = 3e-4
ANNEAL_LR_END    = 1.5e-5      # == WSD_LR_FLOOR
PROBE_MAX_STEPS  = 54_000
```

`TOTAL_STEPS` stays 100,000, so the WSD windows are untouched and the main
run stays resumable.

**Three traps, all found by inspection before running:**

1. **Cell 2 will silently resume from `_step45000.pt`.** It matches
   `f'{CKPT_PREFIX}_step{s}.pt'` for `s in CKPT_STEPS`, and at
   `CKPT_INTERVAL = 7,500` **50,000 is not in `CKPT_STEPS`** — nor does the
   pattern match `_step50000_probe_stop.pt`. The probe cell must set
   `resume_ckpt` / `resume_step` explicitly.
2. **Outputs must be redirected** to `GDRIVE_ROOT / 'anneal_probe'`.
   `RESULTS_DIR` is where `training_log.jsonl` is opened in *append* mode,
   and the probe's steps 50,001-54,000 would otherwise interleave with the
   main run's future steps at identical step numbers.
3. **Seed the probe folder's `_best.pt`.** `_reload_best` resolves
   `CKPT_DIR / f'{CKPT_PREFIX}_best.pt'` and returns early if absent, so the
   watchdog is a **silent no-op** until the first eval at 50,500. Copy
   `_step50000_probe_stop.pt` in as `_best.pt`: it closes that window *and*
   makes the rollback target the anneal's own starting state rather than the
   main run's step-28,500 best, 21,500 steps back.

**Do not add an anneal marker to `_variant_parts`** — that would change
`CKPT_PREFIX` and break the resume path. The probe reuses the tag on purpose
and isolates via the explicit redirect instead.

**Do not judge it before step ≈52,500.** Cosine is nearly flat at its start:
lr is still 96.4% of its starting value at the first eval (50,500) and 86.1%
at 51,000. The informative evals are 52,500 on; decisive are 53,500/54,000.

| step | lr | % of start |
|---|---|---|
| 50,500 | 2.892e-04 | 96.4% |
| 52,000 | 1.575e-04 | 52.5% |
| 53,000 | 5.674e-05 | 18.9% |
| 54,000 | 1.500e-05 | 5.0% |

**Reading the result** (gate as recorded in advance):
- **PPL into the 70s** → confirmed; the LR was the block.
- **PPL barely moves (high 80s)** → the block is deeper than LR.

### 6.1 RESULT, 2026-09-17 — the LR was the block

| step | lr | val_ppl |
|---|---|---|
| 50,000 | 3.00e-04 | 93.93 (branch point) |
| 51,000 | 2.58e-04 | 90.60 |
| 52,000 | 1.58e-04 | 86.49 |
| **52,500** | 1.03e-04 | **84.05** |
| 53,000 | 5.68e-05 | 85.42 |

Monotone throughout, and **84.05 beats the main run's all-time best of
84.31** (step 28,500) — from a start 10 PPL worse, in 2,500 steps. The
53,000 uptick is +1.37 against eval noise σ≈1.54, i.e. noise.

**Scored honestly: a partial hit, not a clean one.** 93.93 was a high draw
(the 45-50K windowed mean was 90.25), so the real gain is **90.25 → 84.05,
−6.2 PPL (−6.9%)** — between the two outcomes stated above, not "into the
70s". Direction unambiguous; magnitude below the recorded threshold. Also
note the anneal was a *compressed* 4,000-step decay against a real one of
≈27,000, and compressed anneals typically underperform, so **84.05 is a
lower bound**.

**The finding that matters most, and it falsifies §7's original premise.**
`omega*dt` was **flat** across the whole anneal — p50 ranged 1.097-1.160
while the LR fell 5.3x and PPL improved 7%; `over_wall` stayed 0.2-0.7% and
`bproj_sig` 36.7-36.9. So the curvature situation did not change at all and
PPL improved anyway. **Integration error was therefore not what capped PPL.**
See §7's revised premise.

**Spikes continued: 3 in 3,050 steps** (101.2, 164.2, 192.9) against 4 in
the previous 15,400. Do not read this as a rate increase — pre-clip gradient
norm does not depend on LR (it is computed before the optimiser step), so
the anneal was never expected to suppress spikes, and n=3 is too small.
The useful negative: **lowering LR does not touch the spike mechanism**,
consistent with routing conjunctions.

### 6.2 Next action

Return to the **main** folder at step 50,000 (not the anneal's checkpoint —
that state is already annealed to a low LR and is not a valid start for a
full decay). Set `ANNEAL_PROBE = False` and **`TOTAL_STEPS = 77,000`**, so
`stable_end = int(0.65 x 77,000) = 50,050` and the real decay begins
immediately rather than burning 15,000 more flat steps that measurably
degrade (+0.159 PPL/1k). ≈27,000 steps ≈ 31h, against ≈58h to finish at
100,000 with 15,000 of those spent going backwards.

### 6.3 Companion probe: anneal from the run's BEST (step 28,500) — **RUN, SUCCEEDED**

**Notebook:** [`colab_fock_cfc_baoab_joint_vtheta_qknorm_annealed_after_28500_openwebtext_d384.ipynb`](../notebooks/conservative_arch/scaleup/colab_fock_cfc_baoab_joint_vtheta_qknorm_annealed_after_28500_openwebtext_d384.ipynb)
(2026-09-17). Identical recipe, length and LR-at-every-eval to §6's
notebook — **only the starting weights differ** — so the two are directly
comparable. Diffs against it are confined to three cells: the config
(`ANNEAL_FROM_STEP = 28_500`, `PROBE_MAX_STEPS = 32_500`), the redirect
(source `_step28500_best.pt`, output folder `anneal_probe_28500`, plus a
hard assertion that the loaded checkpoint's `step` really is 28,500), and
a markdown note. Same run order as §6.

**Why.** §6.1's anneal settled at ≈85.0 from a stable-phase level of
≈90.25, i.e. **≈−5.25 PPL of decay gain**. The main run's stable-phase best
was **84.31 at step 28,500** — so a decayed step-50,000 model is worth
about the same as an *undecayed* step-28,500 model, implying the step-
28,500 weights are better by roughly one decay's worth. The 25,500
stable-phase steps between them bought nothing, and train `ntp` agrees
(4.4733 in 25-30K → 4.5000 in 45-50K): the model genuinely regressed.

**Prediction, recorded before running: settles near 79.**

- **≈79** → confirmed. Set `TOTAL_STEPS ≈ 44,000` (`stable_end = 28,600`)
  and run the real decay from step 28,500. That would beat the additive
  baseline's **81.92** at under half its compute, and reframes the arm as
  "reached its peak 3x faster, then was trained past it on a schedule that
  decayed ≈36,500 steps too late."
- **≈84 again** → the step-28,500 advantage is illusory, §6.2's
  `TOTAL_STEPS = 77,000` plan stands, and the joint-vs-additive gap
  (+2.6% best / +3.8% settled, at 54% of the steps) is real.

Read the **settled level** (mean of the last three evals), not the best:
§6.1 settled at 85.03 while its best, 84.05, was a 1.37-point lucky draw.

#### RESULT, 2026-09-17 — it beat the additive baseline

| eval | lr | from 50,000 (§6.1) | from 28,500 | Δ |
|---|---|---|---|---|
| 1 | 2.89e-04 | 92.56 | 88.16 | −4.40 |
| 3 | 2.12e-04 | 88.28 | 84.56 | −3.72 |
| 5 | 1.03e-04 | 84.05 | **80.75** | −3.30 |
| 8 | 1.50e-05 | 84.91 | 81.47 | −3.44 |
| **settled** | | **85.03** | **81.58** | **−3.45** |

Better at **every** eval, by 2.5-4.4 PPL.

**Against the additive baseline (81.92 best, 100,000 steps): joint wins.**
Best **80.75** (−1.4%), settled **81.58** (−0.4%), using **32,500 steps =
32.5% of the compute**.

**Prediction scored: near miss.** Recorded "settles near 79"; actual 81.58
— off by +2.58. Direction strongly confirmed, magnitude optimistic. Same
failure mode as §3's PPL extrapolation: direction reliable, magnitude not.

**The sharpest form of the finding.** Training 28,500 → 50,000 did not
merely buy nothing — it cost **3.45 PPL permanently**, measured after both
states were given an identical decay. The step-28,500 state is also
healthier on every axis:

| | at 28,500 | at 50,000 |
|---|---|---|
| spikes in 4,000 steps | **0** | 3 |
| max grad norm | **4.22** | 192.9 |
| `omega*dt` p50 | **0.983** | 1.128 |
| `over_wall` | **0.011-0.050%** | 0.2-0.7% |
| `bproj_sig` | **24.88** | 36.80 |

`bproj_sig` grew 48% across that window *while the model got worse*, so the
curvature accumulated there was harmful rather than productive. Consistent
with §6.1: the anneal at 50,000 improved PPL without moving `omega*dt`, so
curvature does not cap PPL *directly* — but a state that accumulated it
anneals to a worse place.

**This reframes the arm.** The joint-vs-additive comparison in §6.1 read
"+2.6% worse at 54% of the steps". With the right branch point it is
**−1.4% better at 32.5%**. Joint coupling was never underperforming; it was
being trained ≈36,500 steps past its peak on a schedule that decayed far
too late.

Leak probe at step 30,000: **[CLEAN]** (honest 49.86 vs standard 48.29,
+0.0319 ± 0.0358 nats).

### 6.4 Production decay from step 28,500 — **NEXT**

**Notebook:** `colab_fock_cfc_baoab_joint_vtheta_qknorm_decay_from_28500_openwebtext_d384.ipynb`

`TOTAL_STEPS = 44_000` → `stable_end = int(0.65 x 44,000) = 28,600`, so a
resume at 28,500 runs 100 stable steps and then a real **15,400-step
decay**. `ANNEAL_PROBE = False`, `PROBE_MAX_STEPS = None`. ≈18h. The
compressed 4,000-step version reached 81.58; a decay ≈4x longer should do
better.

**Two traps this notebook handles explicitly:**

1. **Cell 2 would resume from `_step30000.pt`, not 28,500.** With
   `TOTAL_STEPS = 44,000` the checkpoint steps are 7,500 / 15,000 /
   22,500 / 30,000 / 37,500, and `_step30000.pt` already exists in the
   main folder from the abandoned trajectory. Step 30,000 evaluated at 87.73 against 28,500's
   84.31 — a materially worse start. The notebook sets `resume_ckpt`
   explicitly and asserts the loaded `step`.
2. **Writes must NOT go to the main folder.** The main run already covers
   steps 28,501-50,000, so a second trajectory over 28,501-44,000 would
   duplicate step numbers in `training_log.jsonl` and overwrite
   `_step30000.pt` / `_step37500.pt` from the original run. Output is
   redirected to `GDRIVE_ROOT / 'decay_from_28500'`.

**Do not reuse `anneal_probe/`.** Its `_best.pt` is now the step-52,500
model at 84.05, which would suppress checkpointing, restore the wrong
`best_val_ppl`, and give the watchdog a rollback target from a different
trajectory. The notebook already redirects to `anneal_probe_28500`.

---

## 7. Integration refinement: does the over-wall tail actually cost anything?

**Premise revised 2026-09-17, after §6.1.** This section originally said
"run only if the anneal does not unstick the plateau." The anneal **did**
unstick it — and in doing so falsified the motivating hypothesis, because
`omega*dt` stayed flat (p50 1.097-1.160) while PPL improved 7%. Integration
error is therefore **not** what caps PPL today.

**What survives is forward-looking**, and it is still worth doing:

- `omega*dt` is climbing logarithmically — projected p50 ≈1.39 and
  `over_wall` ≈3.3% by step 100,000.
- The tail is **genuinely unstable**: max 2.2-2.5, and above 2.0 the
  explicit kick has no real solution at all.
- Phase error at the median is ≈6% and rising (§7.6's table).
- All of it worsens at the next scale-up, where the additive arm's d=768
  blowup at step ≈37,000 is already on the record.

So the justification moved from *"rescue this run"* to *"remove a ceiling
while it is still cheap to measure, rather than mid-crisis at the next
scale."* Treat §7 as de-risking work, not as the fix for the plateau.

**The idea.** `omega*dt` scales linearly with `dt`, so halving the step size
halves the whole distribution and moves it back under the stability wall:

| | dt=1.0 (now, step 50,000) | dt=0.5 |
|---|---|---|
| `omega*dt` p50 | 1.128 | **0.564** |
| `omega*dt` max | 2.435 | **≈1.22** |
| `over_wall` | 0.423% | **≈0%** |

This is a **stability** experiment, not a capacity one. It does not add
parameters; it integrates the same dynamics more accurately.

**Order to run these in.** §7.0 comes first and costs two minutes; it
decides whether §7.6 is cheap, and therefore the order of everything below.

**Ordering revised 2026-09-17.** §7.2 was previously called "cheapest".
That was written when §7.6 was believed to need implementation work. It does
not — §7.6 is **zero code**, a config switch (see §7.6). So:

0. **§7.0 — benchmark the linalg first.** Two minutes, no model, no
   training. Determines whether `baoab_cfc_lowrank` is affordable on *this*
   arm, which decides the order of everything below.
1. **§7.6 if §7.0 says it is affordable** — **zero code**, and the better
   fix: exact on the off-diagonal channel rather than merely shrinking the
   step. Cheapest *and* best when the benchmark permits it.
2. **§7.2 as the independent cross-check** — ≈12 lines (see its revised
   estimate). Worth running even if §7.6 works, because it measures the same
   quantity through a different mechanism: **two mechanisms agreeing on "no
   PPL change" is far stronger than either alone**, and agreeing on
   "improvement" would be decisive.
3. **§7.2 first instead** if §7.0 says §7.6 is expensive and the speed
   fixes are not worth building yet.
4. **§7.3 only if you also want the extra gate applications**, a larger
   change than either.

§7.4 and §7.7 record what NOT to do, and why.

### 7.0 Benchmark the linalg before choosing — **RUN, 2026-09-17**

#### RESULT (A100, CUDA 12.8, 8,192 tokens)

| op | shape | ms/call |
|---|---|---|
| QR (randomised path, x3 per call) | `384x20` | 1,108.7 |
| SVD small JOINT | `20x32` | 12.0 |
| SVD small ADDITIVE | `20x160` | **6,335.4** |
| SVD full JOINT (`max_modes=None`) | `384x32` | 9,111.8 |
| **`lowrank_modes` svd driver, q=16** | `384x32` | **3,368.9** |
| **`lowrank_modes` gram driver, all 32** | `384x32` | **40.1** |

**The 32x32 cliff is real — 532x** between `(20,160)` and `(20,32)`. **But it
is not this arm's problem.** Three QRs = 3,326 ms against the measured
svd-path total of 3,368.9 — **99% of the cost is the QR**, and the QR shape
is `P`-independent, so joint coupling gives no relief. The "joint sidesteps
the cliff for free" hypothesis is closed: wrong branch.

**Gram is 84x faster than the svd path** (40.1 vs 3,368.9), taking the arm
from **112 s/step to 5.48** predicted / **5.74 measured**. It also returns
all 32 modes, so the exact treatment is now the *cheap* one and truncation
is obsolete.

**Two process notes, both self-inflicted.** The first benchmark hardcoded a
clone path and skipped the only row that mattered; the second had an
unasserted string replacement that silently no-op'd. Assert every
replacement, and prefer a shallow clone over path discovery.


[`notebooks/conservative_arch/scaleup/debug/bench_lowrank.py`](../notebooks/conservative_arch/scaleup/debug/bench_lowrank.py)
(added 2026-09-17) times the ops `lowrank_modes`
actually performs, at the shapes this arm produces, with no model and no
training. It auto-scales the token batch to fit smaller cards and
normalises every timing against a reference batched matmul so results
compare across GPUs.

**The hypothesis it tests.** cuSOLVER's `gesvdjBatched` handles matrices
only up to **32×32**; above that PyTorch loops over the batch — 8,192
sequential kernel launches per call. The randomised path's small SVD has
shape `(q_over, P)`, so:

| arm | small-SVD shape | inside 32×32? |
|---|---|---|
| additive (P=160) | `(20, 160)` | no → **loops** |
| **joint (P=32, this arm)** | `(20, 32)` | **yes → batched** |

The QR in the same routine is `(B·T, d, q_over)` — **independent of P**, so
identical on both arms. That gives a clean discriminator: **if the SVD is
the bottleneck, the joint arm sidesteps it for free; if the QR is, joint
changes nothing.** The decisive pair of rows is `SVD small JOINT (20x32)`
versus `ADDITIVE (20x160)`.

**Unverified assumption that carries this whole reading.** The ≈60 s/step
figure is remembered from an earlier attempt, and it was **never confirmed
which arm it was measured on**. The cliff explanation only works if it was
the additive arm (P=160). If it was the joint arm (P=32), the mechanism
above does not explain it and the diagnosis must be redone. **Establish
this before interpreting the benchmark.**

A T4 is adequate — the 32×32 limit is a cuSOLVER API constraint, not a
hardware one — but a T4 *understates* the gap (looping is launch-bound and
roughly GPU-independent, while the batched path is throughput-bound and
slower there), so a cliff seen on a T4 is a lower bound on the A100's.

### 7.1 What refines for free, and what does not

Checked against `_fock_layer_step` / `_layer_step_langevin`, 2026-09-16:

- **Forces refine for free.** They enter the update multiplied by `dt`
  (`v_mid + (dt / m_b) * f_kick`), so two half-steps deliver the same
  impulse as one whole step.
  V_theta and the reverse channel need no correction.
- **Register blends do NOT.** The register update is
  `r = blend * r + (1.0 - blend) * readout` — applied once per layer with
  **no `dt` factor**. Doubling the layer count applies it twice as often
  (`blend^16` vs `blend^8`), halving the register's effective memory
  horizon in layer units. Any depth change must either hold the gate
  schedule fixed or correct `blend -> sqrt(blend)` (right for a geometric
  blend, but `blend` is gate-computed per token, not a stored parameter,
  so this is the harder path).

That asymmetry is what makes 7.2 preferable to 7.3.

### 7.2 Substeps at dt=0.5 — **LARGELY SUPERSEDED by §7.6, 2026-09-18**

§7.6 answered the same question more directly and with a proper control:
exact integration of the stiff modes gives **equal train loss** (4.5115 vs
4.5120), a **1.03-sigma** val-PPL edge on n=2, and **1.57x calmer
gradients**. Halving `dt` is a weaker version of that test — it shrinks the
error rather than removing it — so this is now only worth building if you
want a second, independent mechanism to confirm §7.6's stability result.
The ≈12-line implementation below still stands if so.


Keep `L = 8`, `depth_code`, every gate and every weight **exactly as they
are**; split each layer's integration into two substeps of `dt = 0.5`.

- **No checkpoint surgery.** Shapes are unchanged, so the resume path is
  untouched.
- **Gate schedule preserved exactly** — gates still fire once per layer.
- **Total integration time per layer unchanged** (2 × 0.5 = 1 × 1.0).
- Compute rises only on the integrator, not the gates/attention, so expect
  well under 2x per step.

**The zero-training read.** Branch from the **anneal's best,
`anneal_probe/checkpoints/..._step52500_best.pt` (PPL 84.05)** — *not*
step 50,000. Revised 2026-09-17: that is a **settled** state at low LR
rather than one bouncing at 3e-4, so a PPL change is attributable to
integration accuracy instead of being confounded with plateau dynamics.
`omega*dt` there is still ≈1.12 and `bproj_sig` ≈36.8 (both flat through
the anneal), so the error under test is present at full magnitude.

Evaluate with the refined integrator and compare against **84.05**, with no
training at all:

- **PPL improves** → integration error was genuinely costing accuracy, the
  over-wall tail is doing damage, and reducing `dt` (or capping curvature)
  is a real lever.
- **PPL unchanged** → the wall crossings are harmless at this magnitude,
  and `omega*dt` can be retired as a concern for this run. This would also
  retire §2.6.2's open question.
- **PPL degrades** → **inconclusive, not a negative result.** The model was
  *trained* with dt=1.0 explicit kicks, so its parameters partly compensate
  for that integration error; refining the integrator removes an error it
  had adapted to. Distinguishing "the refinement is worse" from "the
  compensation was removed" needs a few hundred steps of re-adaptation
  before the comparison means anything.

The gate is therefore **asymmetric**: an immediate improvement is strong
evidence, an immediate regression is not evidence either way. The first read
still costs **minutes**.

**Implementation — ≈12 lines, revised down 2026-09-17.** An earlier estimate
here said 30-60 lines because the `h_prev` bookkeeping looked risky: velocity
is carried implicitly and the substeps must decode it at the inner `dt`, not
the outer one. Checked, and both conversions are linear and trivially
invertible (`v = (h - h_prev)/dt`, `h_prev_out = h_new - dt*v`), so the
boundary re-encoding is just a rescale. Wrap the existing `_layer_step_ex`
call in `_fock_layer_step` (**two** call sites — the `prefix_causal` branch
and the extended one):

```python
_n = getattr(cfg, 'integrator_substeps', 1)
if _n > 1:
    _dt  = dt / _n
    v_in = decode_velocity(h, h_prev, dt)
    h_cur, hp = h, encode_velocity(h, v_in, _dt)
    for _ in range(_n):
        h_cur, hp = super()._layer_step_ex(
            h_cur, hp, m_b, gamma, _dt, layer_idx=layer_idx)
    h_new, h_prev_out = h_cur, encode_velocity(
        h_cur, decode_velocity(h_cur, hp, _dt), dt)
else:
    h_new, h_prev_out = super()._layer_step_ex(
        h, h_prev, m_b, gamma, dt, layer_idx=layer_idx)
```

The gates stay **outside** this loop, which is what preserves the schedule
(§7.1). **Validate first:** `integrator_substeps=1` must be bit-identical to
the current path — that catches any boundary-handling error for free before
`=2` is trusted.

### 7.3 Fallback: L=16 at dt=0.5 with checkpoint surgery — **only if §7.2 or §7.6 says yes**

Genuinely doubles the gate applications as well as the integration
resolution, so it is a different (and larger) change than 7.2. Requires
surgery, because three tensors are L-shaped:

| parameter | L=8 | L=16 |
|---|---|---|
| `depth_code` | `[8, 5, 384]` | `[16, 5, 384]` |
| `reverse_channel_scale` | `[8]` | `[16]` |
| `destruction_gates` | 8 modules | 16 modules |

(`creation_gates` is an **empty** ModuleList under `fock_version='v2'` — the
creation gate is the shared `creation_gate_qkv` — and `reverse_ch` and
V_theta are shared, so only these three break.)

**It will not load without surgery, and it fails loudly**, verified
2026-09-16: `load_state_dict(..., strict=False)` suppresses missing and
unexpected keys but **never** shape mismatches, so `depth_code` and
`reverse_channel_scale` raise `RuntimeError` outright; the 8 new
`destruction_gates.{8..15}.*` would additionally trip the
`_n_loaded < 0.9 * _n_model` resume guard. Two independent safety nets.

Surgery: interpolate `depth_code` along the L axis (new layer ℓ' sits at old
time ℓ'/2), interpolate `reverse_channel_scale` likewise, duplicate each
`destruction_gate` into the two layers subdividing it — **and then deal with
the blend problem in §7.1**, which duplication alone does not solve.

Same zero-training go/no-go gate: evaluate immediately against 93.93. If PPL
collapses, the surgery did not preserve the function and anything measured
afterward is confounded.

### 7.4 Do NOT run L=16 at dt=1.0

Confounded. Layers here are **integration steps**, so L=16 at dt=1.0
integrates for twice as long — a different trajectory, not more capacity.
The run's own `L_PROBE_OVERRIDE` comment already isolates this: *"not
conflated with the separate 'fewer L, bigger dt for the same total
integration time' question, which is a different experiment."*

### 7.5 Standing facts about depth in this architecture

- **Depth is nearly free in parameters, expensive in compute.** Embeddings +
  untied head (38.6M, 50%) and the **shared** V_theta bank (35.4M, 46%) are
  both L-independent; only ≈2.69M is per-layer, i.e. **337K/layer**. So
  L=8→16 is **+3.5% params** but **≈2x compute/step** (≈8.4 s/step). Adding
  wells is the opposite trade: K=8→16 is +46% params at ≈1x compute.
- **L=8 is a mitigation, not a preference.** The ARCH_TIERS ladder picks
  `d=384, L=16, M=32` by default. L=8 was pinned because the L=16 run hit a
  grad-clip spike burst (steps 6297-6676) with a real PPL hit (176.88 →
  207.11), the stated mechanism being that depth lengthens the compounding
  chain a spike propagates through. **All four of this run's spikes are
  `reverse_channel_scale`-led**, and `reverse_ch` is a single weight-tied
  module reused at every layer — so its gradient accumulates once per layer.
  Depth amplifies the exact signature currently escalating.
- No evidence L=16 was ever better at matched steps: at step 6,000 the L=16
  run was at 176.88 PPL (pre-burst) against this L=8 arm's **170.98**.
  Different arms, so not clean — but it was not ahead, and it was spikier.

---

### 7.6 `baoab_cfc_lowrank` — **RUN AND CLOSED, 2026-09-18**

#### VERDICT

**It works, it is affordable, and it buys stability rather than loss.**
Use `baoab_cfc` for the production decay; keep lowrank for the next
scale-up. Reasoning below.

**Notebooks:** `colab_..._lowrank_from_50K_...ipynb` and its matched
control `colab_..._control_flat_from_50K_...ipynb` — identical branch
point (step 50,000), flat `lr=3e-4`, 1,000 steps, same seed and data
order, differing only in `INTEGRATOR` and the output folder.

**PPL — lowrank ahead at both evals, and widening:**

| step | lowrank | control | edge |
|---|---|---|---|
| 50,000 | 93.93 | 93.93 | *(shared start)* |
| 50,500 | 92.71 | 93.27 | +0.56 |
| 51,000 | **91.67** | 93.26 | **+1.59** |

Control went flat (−0.67 then nothing); lowrank kept descending (−2.26,
still falling at the stop).

**But train loss is a dead heat**, and it is the more direct measurement:
control **4.5120** vs lowrank **4.5115** over 20 matched step-lines — a
difference of 0.0005. A 1.59 val gap is **1.03 sigma on n=2**. Treat the
PPL edge as suggestive, not established.

**What IS solid — gradient stability, at n=20:**

| | control | lowrank | |
|---|---|---|---|
| grad mean | 3.67 | 2.34 | **1.57x** |
| grad std | 2.59 | 1.47 | **1.76x** |
| grad max | 12.16 | 7.94 | |
| spikes | **2** (343.9, 258.7) | **1** (145.3) | `creation_gate` vs `depth_code` |

(Called at n=7, collapsed at n=10, recovered at n=20. Magnitude is loose;
direction has now survived a real sample.)

**Cost: 1.36x time (5.74 vs 4.21 s/step) and +6.9 GB** — eval peak 75.86 GB
against `baoab_cfc`'s 68.96, i.e. **89% of 85.1 GB**. Tight on an 80 GB
card, would not fit a smaller one.

**Why `baoab_cfc` for the production decay anyway:**

1. Known quantity, 18h vs 24.6h, with the anneal's 81.58 already as a
   reference point.
2. The PPL edge was measured at flat LR from the *degraded* step-50,000
   state. No reason to assume it transfers to a decay from healthy 28,500.
3. The decay is where the final number comes from — not the place for a
   code path that raised `_LinAlgError` on its first contact with real `G`.

**Why keep it for the next scale-up:** `omega*dt` is climbing
logarithmically, `over_wall` projects to ≈3.3% by step 100,000, and the
d=768 blowup at step ≈37,000 is on the record. 1.57x calmer gradients and
half the spikes is the right insurance *there* — and it is now a working,
benchmarked, config-switchable option rather than a 112 s/step non-starter.

#### OPEN ITEM — the resonance monitor is blind under this integrator

Zero `[resonance]` lines across the entire lowrank pilot, where every
`baoab_cfc` run emits one per 500 steps. `resonance.observe()` patches
`vt.harmonic_terms` and `integrator_module.cfc_substep`; the lowrank path
calls `harmonic_terms_lowrank` and `lowrank_cfc_substep`, so the monitor
records nothing and warns about nothing. **Fix before any long lowrank
run** — that arm is precisely the one whose `omega*dt` we would want to
watch. The control supplies the reference meanwhile: p50 1.108/1.118, max
2.418/2.344, `over_wall` 0.200%/0.284% at steps 50,500/51,000.

#### The driver, as built


**Nothing needed building — verified 2026-09-17.** Targeting only the
stiffest modes is already in the code, and correctly:

- `lowrank_modes(G, max_modes=...)` has both paths — `_svd_stable` (full,
  all `P` modes) and `_randomised_svd_det` (truncated, top `q`).
- `_randomised_svd_det` is a hand-rolled Halko-Martinsson-Tropp
  range-finder written **branch-free with a local fixed-seed generator**,
  specifically because `torch.svd_lowrank` trips `CheckpointError` under
  gradient checkpointing. Do not replace it with a library call.
- The caller demotes dropped modes correctly: `_layer_step_langevin`
  subtracts only `P_U f_L = U(U^T f_L)`, so the soft modes stay in the
  stable explicit kick instead of being silently cancelled — exactly what
  `lowrank_modes`' docstring requires of it.
- `lowrank_max_modes`, `lowrank_niter` and `lowrank_oversample` are all
  live config knobs read via `getattr(cfg, ...)`.

So this is a **configuration change, not an implementation task**. The open
question is cost, not correctness — see §7.0.

**Keep `LOWRANK_MAX_MODES = 16`, not `None`, on this arm.** `16 < P = 32`
selects the randomised path, whose small SVD is `(20, 32)` and fits inside
cuSOLVER's batched limit; `None` takes `_svd_stable` on `(384, 32)`, which
does not. `_svd_stable` also carries `try/except`, a jitter-retry escalation
and a CPU fallback — metadata still matches, so it likely will not raise
under checkpointing, but a forward and a recompute could take different
branches and compute gradients against slightly different activations. The
truncated path avoids that by construction.

**The defect this targets.** `baoab_cfc` is only *partially* exact. The
closed-form harmonic propagator handles the **diagonal** part
(`diag(a_k)`); the anisotropic **off-diagonal** part `B_k B_k^T` is still an
explicit kick, which is where the `omega*dt < 2` wall comes from. So the
off-diagonal channel is not merely near a stability limit — it is
*under-resolved*. For leapfrog on a harmonic mode the numerical frequency
satisfies `sin(w_num*dt/2) = w*dt/2`, giving:

| `omega*dt` | frequency error | where |
|---|---|---|
| 0.564 | 1.4% | under §7.2 substeps |
| **1.128** | **6.2%** | **current p50** |
| 1.500 | 13.1% | |
| 1.900 | 31.9% | |
| 2.000 | unstable | the wall |
| **2.435** | **unstable** | **current max** |

The median mode carries ≈6% phase error and the 0.423% over the wall have no
stable solution. `INTEGRATOR = 'baoab_cfc_lowrank'` integrates
`L = sum_k g_k B_k B_k^T` **exactly** on its stiffest modes, so that channel
has no hard wall at all (only narrow damped resonances at `omega*dt ≈ k·pi`).
It fixes the cause rather than shrinking the step.

**Why it may be affordable here when it was not for additive — CONFIRMED
2026-09-16.** The arm was rejected at ≈120 s/step against `baoab_cfc`'s
10-15. But `G` has `K·rank` columns **per bank**, and the number of banks
differs by coupling. Traced through all three implementations in
`model_aniso_gaussian_vtheta.py`:

| class | `G` width | modes |
|---|---|---|
| `AnisotropicMixtureGaussianVTheta` (one bank) | `d × K·rank` | 32 |
| `AnisotropicMultiContextGaussianVTheta` (additive, `torch.cat(Gs)`) | `d × n_ctx·K·rank` | **160** |
| `JointContextAnisotropicGaussianVTheta` (**this arm**, delegates to `self.bank`) | `d × K·rank` | **32** |

So this arm's low-rank operator is **5x smaller** than the additive arm's —
an incidental payoff of joint coupling that had not been noticed. If the
cost is dominated by the `O(d·P^2)` SVD, `P` going 160 → 32 is a **25x**
reduction in that term, which would put the exact arm in the neighbourhood
of the current 4.2 s/step rather than 120. **Measure it; do not assume it** —
the cost model may not be SVD-dominated, and the 120 s/step figure was taken
on a different configuration.

Note also that `LOWRANK_MAX_MODES = 16` covers **half** of this arm's 32
modes, where it covered only 10% of additive's 160. With `over_wall` at
0.423%, the set genuinely needing exact treatment is far smaller than 16, so
the truncated path should be ample — and `None` (all 32, full SVD) may now
be viable too.

**Plumbing already exists.** `INTEGRATOR` is part of `_variant_tag`, so
switching it would normally orphan this arm's checkpoints into a fresh empty
folder. **Cell 1b** (`RESUME_VARIANT_TAG_OVERRIDE`) was built for exactly
this case, in its own words: *"the goal is to keep training the SAME run and
only change which integrator it uses from here on (e.g. baoab_cfc ->
baoab_cfc_lowrank after a hard-watchdog burst)."* Set it to this arm's tag.
Re-read that cell's warning first — pointed at the *wrong* tag it silently
loads an incompatible V_theta.

**Same zero-training go/no-go as §7.2, including its asymmetry:** load
step-50,000, switch the integrator, evaluate without training, compare
against **93.93**. An immediate improvement is strong evidence; an immediate
regression is inconclusive, because the trained weights partly compensate
for the integration error being removed. Then measure s/step over a few
hundred steps before committing — the cost is an open question (see the
`gesvdjBatched` 32x32 batching cliff), not a settled one.

### 7.7 Do NOT coarsen (L=4 at dt=2.0, etc.)

`omega*dt` scales linearly with `dt`, so coarsening moves the whole
distribution *past* the wall. Using the measured p50 = 1.128 and
log-normal σ = 0.217 at step 50,000:

| config | dt | `omega*dt` p50 | max | `over_wall` |
|---|---|---|---|---|
| L=16, or L=8 with 2 substeps | 0.5 | 0.564 | 1.22 | ≈0% |
| **L=8 (current)** | **1.0** | **1.128** | **2.44** | **0.4%** |
| L=4 | 2.0 | 2.256 | 4.87 | **71%** |
| L=2 | 4.0 | 4.512 | 9.74 | **100%** |

At `L=4, dt=2.0` the **median** token-slot is over the wall. Beyond the
stability argument, two structural reasons make "exactness ⇒ fewer layers"
not apply here:

1. **The flow is non-autonomous.** `depth_code` is `[L, n_ctx, d]` and
   `V_THETA_DEPTH_CONDITION = True`, so each layer applies a *different*
   conditioned potential. Exact integration lets you take larger steps
   through a *fixed* vector field; it cannot merge steps of a field that
   changes every layer. Dropping layers discards learned potentials, not
   redundant Euler steps.
2. **Gates are not integration.** `r = blend * r + (1 - blend) * readout`
   fires once per layer with no `dt` factor (§7.1), so halving `L` halves
   the register updates — halving the depth of a recurrent channel.

§7.2 bounds this question from both sides as a by-product: if PPL is
unchanged at dt=0.5 then integration error at dt=1.0 is already negligible
and coarsening buys nothing while costing the wall; if PPL improves, then
coarsening is worse still.

---

## 8. After this analysis

Decide `PROBE_MAX_STEPS`: clear to `None` for the full 100,000-step arm, or
set another stop. The WSD schedule is a pure function of `TOTAL_STEPS` and
is unaffected by either choice; the decay phase does not begin until step
65,000, so any stop before then costs nothing but a pause.

---

## 9. The matched GPT-2 baseline — **RUNNING, 2026-09-18**

**Notebook:** [`colab_matched_gpt2_baseline_openwebtext.ipynb`](../notebooks/conservative_arch/scaleup/colab_matched_gpt2_baseline_openwebtext.ipynb)

This is the experiment that answers the programme's standing question: where
is the bottleneck in the current Fock architecture?

### 9.1 Why this comparison is valid

Four things had to line up, and all four were verified in the source rather
than assumed:

| | Fock joint arm | GPT-2 baseline |
| --- | --- | --- |
| val set | `openwebtext_val_2M.npy` | **same file** — same 2B stream, same `all_ids[-2M:]` slice |
| tokens/step | 16 x accum 2 x 512 = 16,384 | 32 x 512 = 16,384 |
| train pool | 2B, sampled with replacement | **same 2B pool** |
| endpoint | step 32,500, lr at floor 1.50e-05 | step 32,500, cosine at `LR_MIN` |

Because tokens/step match, **steps map 1:1** and equal steps are equal tokens.
Four notebook bugs had to be fixed first: a 200M token budget (32.5 epochs), a
`total_mem` typo, a stale cache-name list that re-streamed 2B tokens, and an
`EVAL_INTERVAL` of 2000 which does not divide 32,500 — that last one would
have skipped the final decayed eval entirely and produced exactly the
schedule-position error this section exists to avoid.

### 9.2 Result

**GPT-2 crossed Fock's fully-decayed endpoint at ≈step 15,200.** Interpolating
between the step-15,000 eval (82.80) and step-15,500 (79.07):

| Fock reference | GPT-2 reaches it at | fraction of budget |
| --- | ---: | ---: |
| 81.58 (settled, §6.3) | step ≈15,164 | **46.7%** |
| 80.75 (best, §6.3) | step ≈15,275 | 47.0% |

At step 20,500 GPT-2 stood at **64.60** and was still falling ≈1 PPL per 500
steps with 12,000 steps of cosine decay remaining.

**FINAL ENDPOINT AT STEP 32,500: `val_ppl_512` = 54.59** (settled 54.67),
recorded 2026-09-19.

**Prediction scored: the extrapolation was wrong again, in the same
direction.** A log fit over steps 17,000-20,500 projected ≈44-48; the
saturating read said 55-60. Actual **54.59**. That is the third time in this
programme a log-linear extrapolation has overshot and a saturating fit has
held (§3 recorded the first two). Treat log-linear extrapolation of PPL as
refuted for this programme, not merely unreliable.

Note GPT-2 also missed its own Kaplan prediction of 30-35, which is expected:
0.53B tokens against 33.7M parameters is 15.7 tokens/param, below the
Chinchilla-optimal 20.

### 9.3 The four-axis comparison

| | Fock | GPT-2 | |
| --- | ---: | ---: | --- |
| PPL, final decayed | 81.47 | **54.59** | −26.88 |
| PPL, settled (last 3) | 81.58 | **54.67** | −26.91, a ratio of **1.49x** |
| PPL, best | 80.75 | 54.59 | −26.16 |
| total parameters | 76,745,698 | 33,691,776 | GPT-2 uses **2.28x fewer** |
| non-embedding parameters | 37,760,000 | 14,196,480 | GPT-2 uses **2.66x fewer** |
| inference MMAC/token | 323.6 | 36.6 | GPT-2 is **8.84x cheaper** |
| training s/step | 4.18 | 0.29 | GPT-2 is **14.4x faster** |

**The comparison is decided.** A plain transformer with 2.66x fewer
non-embedding parameters reaches 33% lower perplexity on identical data at
identical tokens, while costing 8.84x less to serve and 14.4x less to train.
Fock-PARFLM as configured loses on every axis simultaneously, and not
narrowly.

Step times measured from the logs: Fock steps 28,550 to 28,600 took 209s for
50 steps; GPT-2 steps 14,200 to 14,400 took 58s for 200.

The one caveat runs in Fock's favour: 81.58 came from a *compressed*
4,000-step anneal, and §6.1 notes compressed anneals typically underperform a
real decay. Credit Fock a generous 2 PPL and the conclusion does not move.

### 9.4 The bottleneck diagnosis

Full analysis in
[`Fock_Inference_Productionization_Plan.md`](Fock_Inference_Productionization_Plan.md) §7.
In brief: the context reaches the token update only through 5 fixed-decay
EMAs, whose weights depend on distance `t-s` and never on content. 93.9% of
Fock's non-embedding parameters sit in `V_theta`'s well-parameter generator,
which is **downstream** of that compression and can only reshape what survived
it. Every prior negative ablation in this checklist is consistent with an
upstream bottleneck: more wells did not help (§ joint K=8 beat additive K=40),
more depth did not help (§7.5), and rank is already fully used (PR 3.68
against rank 4).

### 9.5 Pre-registered predictions

Recorded before running, to be scored the way §3 was.

1. **Factorizing `V_theta` at bottleneck width 256** (Phase 2 of the
   productionization plan) cuts its parameters ≈6.5x. **Prediction: PPL moves
   by less than 2.** If it holds, those parameters were not doing work and the
   bottleneck is definitively upstream in `xi`. If PPL degrades sharply, the
   capacity was being used and this diagnosis is wrong.
2. **Raising `XI_CHANNELS` from 5 to 10.** **Prediction: gain under 2 PPL.**
   More fixed taps do not fix content-independence.
3. **A content-addressed pooling variant** (paper `17f` family A or C).
   **Prediction: gain of 10 PPL or more**, and much larger than 2.

Predictions 2 and 3 together separate "too few taps" from "wrong kind of
pooling". Prediction 1 is the cheapest and sharpest, and is already scheduled
for other reasons.

---

## 10. Family A: xi-routed conservative attention — **GATES 0-2 PASSED, 2026-09-18**

Tests prediction 3 of §9.5. Motivation and the bottleneck argument are in
[`Fock_Inference_Productionization_Plan.md`](Fock_Inference_Productionization_Plan.md) §7.

### 10.1 What already existed

`model_xi_attention.py` (committed 2026-07-02, `8fbf4fe`) already implements
family A as `XiRoutedConservativeAttention`, and it is conservative **by
construction**: routing weights are read off the detached `xi` summary, so
`alpha(t,s)` is a constant with respect to `h_t` and the induced force stays
the gradient of a scalar potential. No straight-through estimator is needed.

Two things blocked using it directly:

1. `colab_xi_attention_openwebtext.ipynb` **has never been run** — zero output
   cells.
2. `XiAttnPARFLM` subclasses `MultiXiPARFLM`, **not** `FockMultiXiPARFLM`. It
   therefore has no registers, no QK-norm, no CfC/BAOAB integrator and no
   anisotropic joint `V_theta`. Training it as-is would confound
   content-addressed routing with four other absent mechanisms.

### 10.2 The port (Gate 0) — **DONE**, committed in `661d959`

`_pair_potential` in `model_parf_multixi.py` is a clean seam: it returns a
**scalar** and is called once from `_layer_forces`, which already holds `xis`.
`FockMultiXiPARFLM` overrides neither. So the port is a branch, not a rewrite.

Four edits to `model_parf_multixi.py`, all strictly opt-in:

| edit | what |
| ---- | ---- |
| `MultiXiPARFConfig` | added `pair_potential` (default `'sparse_topk'`) plus the `attn_*` knobs |
| `MultiXiPARFLM.__init__` | builds `V_attn` when selected and sets `V_phi`/`score_head` to `None`; imports `model_xi_attention` **lazily**, since that module imports from this one and a module-level import would be circular |
| `_pair_potential` | new leading branch; signature gains `xis=None`, so the old two-argument call still works |
| `_layer_forces` | passes `xis=xis` |

The default path is untouched: `pair_potential` defaults to `'sparse_topk'`,
`V_attn` is `None`, and nothing new is constructed. Selection is read with
`getattr`, so a checkpoint or notebook built against an older config still
loads and still takes the sparse path.

### 10.3 Gate results

| gate | test | result |
| ---- | ---- | ------ |
| 0a | default config unchanged, old two-arg `_pair_potential` still callable | PASS |
| 0b | `xi_attention` builds; zero `V_phi`/`score_head` keys left in `state_dict` | PASS |
| 0c | calling without `xis` raises rather than silently mis-computing | PASS |
| 0d | forward and backward for both variants; all 4 `V_attn` tensors receive gradient | PASS |
| **1** | force equals minus the finite-difference gradient, context frozen | **PASS**, rel err 1.4e-09 (`dot`), 4.8e-09 (`rbf`) |
| **2** | perturb tokens at `t >= 4`; force at `t < 4` must not move | **PASS**, change 0.0e+00 exactly, against a 2.9e-03 control at `t >= 4` |

### 10.4 Two findings from running the gates

**The Hessian-symmetry test is VACUOUS for the `dot` kernel.** Its potential is
bilinear in `h_t` with `h_s` detached, hence **linear** in `h_t`, so its Hessian
is identically zero and symmetry proves nothing. The finite-difference test is
the real proof and is valid for either kernel. Under `rbf` the Hessian is
genuinely non-trivial (max magnitude 2.0) and exactly symmetric.

This has a direct consequence for Gate 3. A linear potential contributes **no
curvature**, so the `dot` kernel cannot move `omega*dt` at all. The `rbf` kernel
is quadratic in `h_t` and **can** push it toward the wall. Run `dot` first.

**A finite-difference conservativity check must freeze `h_src`.** Because
`h_src = h_in.detach()`, a naive perturbation moves the query and the source
together and the test fails with relative error near 1.0 — a broken test, not a
broken model. This is what `arm1_jacobian_symmetry`'s docstring means by "with
context frozen".

### 10.5 Cost and parameters at d=384

| | parameters | MMAC/token/layer |
| --- | ---: | ---: |
| `sparse_topk`: `V_phi` + `score_head` | 137,801 | 1.557 |
| `xi_attention`: `V_attn` (dot, 4 heads, d_k=d_v=48) | 884,736 | 1.081 |

All-to-all routing is **cheaper** than top-k here, because the routing
projections from the 1920-dimensional `xi` dominate while the T-dependent term
is only `n_heads * T * d_k`. The parameter count rises 6.42x but that is
+747K against a 76.7M model, under 1% of the total.

### 10.6 Two mechanisms, two slots

Family A as written replaces `V_phi` — the **pair** path, 3.85% of compute. The
bottleneck diagnosed in §9.4 is the **`xi` to `V_theta`** path: 87.6% of
compute, 93.9% of parameters. Running only family A risks a false negative on
the whole hypothesis.

**Naming.** An earlier draft of this section called these "A1" and "A2". That
was a bad choice twice over: `Context_Mixing_Mechanisms_in_the_Conservative_Framework.md`
§4.4 already uses **Option A1** and **Option A2** for family A's own *kernel*
choices (squared-norm and dot-product), and paper v3 uses §A1/§A2 for
appendices. The canonical names below follow that document's
`Alternative X` convention and are used everywhere from here on.

| mechanism | slot | change | tests |
| --------- | ---- | ------ | ----- |
| **Family A** — xi-routed conservative attention (note §4) | pair | `V_attn` replaces top-k `V_phi` | content-addressing on the pair path. **Ported, gated, ready.** |
| **Alternative E** — content-addressed xi pooling (note §8) | pooling | content term added inside the EMA softmax, feeding `V_theta` | content-addressing on the diagnosed path. **Implemented and gate-verified.** |

The two occupy different slots and compose. Outcomes: both help means
content-addressing helps generally; only Alternative E confirms §9.4; only
family A means the diagnosis is wrong and the paper's framing is right;
neither means content-addressing is not the issue and the second-order flow
itself is the suspect.

### 10.7 Remaining gates

The two mechanisms need **different ladders**, because only Alternative E
warm-starts. Family A removes trained `V_phi` and `score_head` and
initialises `V_attn` at `attn_init_scale = 0.02`, so it perturbs the model at
step 0 and cannot resume; Alternative E is bit-identical at initialisation
(§10.4) and can be dropped onto a trained checkpoint.

#### Alternative E — warm-start probe (RECOMMENDED NEXT)

| gate | what | cost | criterion |
| ---- | ---- | ---- | --------- |
| 3 | `omega*dt` and step-0 PPL after loading `_step28500_best.pt` | minutes | PPL must read **84.31** exactly, the checkpoint's own value. Anything else means the warm start is not bit-identical and the run is invalid. **Confirm the resonance monitor arms** — §7.6 recorded it going silently blind under the lowrank path |
| 3b | `scaf.audit(...).assert_causal()` | minutes | no leak. The local check in §10.4 is a single future-perturbation probe, which is exactly what certified a 7.69-PPL checkpoint whose honest value was 258.07 |
| 4 | the §6.3 anneal, unchanged: 4,000-step decay from step 28,500 | ≈5h | compare **settled against 81.58** |

This is an exactly controlled comparison: same checkpoint, same schedule,
same step count, same eval grid as §6.3, with the content term as the only
difference. It needs no new control run.

**Pre-registered.** §9.5 prediction 3 says 10 PPL or more.

| settled result | reading |
| -------------- | ------- |
| **≤ 71.6** | prediction met; §9.4's diagnosis confirmed; proceed to a full run |
| 73.6 to 79.6 | real but under-predicted; worth the full run, prediction scored as a miss |
| **≥ 79.6** | within roughly 2 PPL of 81.58, i.e. inside §6.3's own eval noise. **This refutes the diagnosis** — the parameters downstream of `xi` were not starved of context, and the second-order flow itself becomes the suspect (§7.4 of the plan) |

Fold `bench_inference.py` into this notebook after Cell 5. It yields **Part B**
— `V_theta` measured on GPU — at zero marginal cost, which plan §8a.3 requires
before anyone acts on the Phase 2 roadmap.

#### Family A — from-scratch ladder (only if worth running)

| gate | what | cost | criterion |
| ---- | ---- | ---- | --------- |
| 3 | `omega*dt` at init; step-0 PPL | minutes | use the `dot` kernel first: §10.4 shows it adds no curvature and so cannot move `omega*dt`; `rbf` can |
| 4 | 2,000-step pilot | ≈2.5h | no watchdog trips; grad norms comparable |
| 5 | 5,000-step pilot | ≈6h | see below |
| 6 | full 32,500 | ≈38h | beat 81.58 and 54.67 |

**Gate 5 uses a control that already exists.** Run on the *same* schedule as
the 150,000-step run (warmup 7,500, lr 3.0e-04) and compare step-for-step
against its logged head. Do not generate a fresh control, and do not change
`TOTAL_STEPS` — the WSD schedule is a pure function of it.

| step | current arm `val_ppl` |
| ---: | ---: |
| 2,000 | 473.12 |
| 3,000 | 332.90 |
| 4,000 | 253.93 |
| 5,000 | **201.56** |

This matched-step comparison is valid where the GPT-2 early-curve comparison
was not, and for a specific reason: identical schedule, warmup, learning rate,
initialisation recipe and data order, with the pair term as the only
difference. The GPT-2 comparison failed that test and had to be restricted to
the decayed endpoint.

At step 5,000, where PPL is near 200, a real mechanism gain should appear as
**15 PPL or more**, far outside the 1.5 eval noise. **Under 5 PPL at step
5,000 does not justify Gate 6.**
