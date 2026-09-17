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

## 6. The anneal probe (next action, 2026-09-16)

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

**Reading the result:**
- **PPL into the 70s** → confirmed; the LR was the block and the
  architecture was never the problem. Then set `TOTAL_STEPS ≈ 77,000` so
  `stable_end = 50,050` and the real decay starts immediately instead of
  burning 15,000 more flat steps.
- **PPL barely moves (high 80s)** → the block is deeper than LR. §2.1's cap
  question becomes the priority, with the four bundles to diagnose against.

---

## 7. Integration refinement: does the over-wall tail actually cost anything?

**Run only if the anneal (§6) does NOT unstick the plateau.** If lowering the
LR fixes it, this question is moot for now.

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

0. **§7.0 — benchmark the linalg first.** Two minutes, no model, no
   training. Determines whether `baoab_cfc_lowrank` is affordable on *this*
   arm, which reorders the rest.
1. **§7.2** — the cheapest *diagnostic*: zero surgery, zero training,
   minutes. Answers "does integration error cost anything at all?" If PPL
   is unchanged at dt=0.5, stop — the wall crossings are harmless and
   §7.3/§7.6 are unnecessary.
2. **§7.6 if §7.2 says yes** — the better *fix*: it removes the wall on the
   off-diagonal channel outright rather than shrinking the step. **If §7.0
   shows it is cheap on this arm, it likely outranks §7.2**, being exact
   rather than approximate.
3. **§7.3 only if you also want the extra gate applications**, a larger
   change than either.

§7.4 and §7.7 record what NOT to do, and why.

### 7.0 Benchmark the linalg before choosing — **RUN FIRST**

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

### 7.2 Cheapest diagnostic: same L=8, two substeps of dt=0.5 — **PROPOSED**

Keep `L = 8`, `depth_code`, every gate and every weight **exactly as they
are**; split each layer's integration into two substeps of `dt = 0.5`.

- **No checkpoint surgery.** Shapes are unchanged, so the resume path is
  untouched.
- **Gate schedule preserved exactly** — gates still fire once per layer.
- **Total integration time per layer unchanged** (2 × 0.5 = 1 × 1.0).
- Compute rises only on the integrator, not the gates/attention, so expect
  well under 2x per step.

**The zero-training read.** Load the step-50,000 weights, evaluate with the
refined integrator, and compare against **93.93** with no training at all:

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

**Implementation note.** `_layer_step_ex` can be called twice with `dt/2`,
but velocity is carried
implicitly — `decode_velocity(h_in, h_prev, dt)` — so the `h_prev`
bookkeeping between the two substeps must be right or the refinement is
silently wrong. Validate by setting substeps=1 and confirming the result is
bit-identical to the current path before trusting substeps=2.

### 7.3 Fallback: L=16 at dt=0.5 with checkpoint surgery — **only if 7.2 says yes**

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

### 7.6 `baoab_cfc_lowrank` — the principled fix, already implemented

**Nothing needs building — verified 2026-09-17.** Targeting only the
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
