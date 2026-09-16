# Diagnostic checklist — joint-V_theta + QK-norm d384 run

Personal working note, not for publication. What to run against the
joint-coupling arm (`fock_cfc_owt_..._vtjoint_cgqk_L8probe_..._baoab_cfc`)
once GPU time is free. Deliberately deferred: every probe here is a
deterministic function of saved checkpoints, so none of it competes with
training for a Colab session.

**Run status at time of writing (2026-09-15).** Paused at **step 34,600**,
resuming to `PROBE_MAX_STEPS = 50,000` under `TOTAL_STEPS = 100,000`
(WSD: warmup 0→5,000, stable 5,000→65,000, decay 65,000→100,000, floor
1.50e-05). Best PPL **84.31 at step 28,500**; latest eval 87.95 at 34,500.
**Zero `[spike]`, `[watchdog]` or reload events across all 34,600 steps.**

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

**No spikebatch bundles exist for this arm.** Nothing has cleared
`capture_threshold = 100.0` — the largest pre-clip grad norm in 34,600 steps
is **15.36** (step 34,600). This blocks, entirely:

| probe | module | why blocked |
|---|---|---|
| `omega_dt_report` | `resonance` | takes `step_tag`, loads a bundle |
| `omega_dt_under_truncation` | `resonance` | same |
| `tail_coherence_report` | `resonance` | same |
| `replay_precision_cap_ablation` | `precision_cap` | same |
| `replay_rank_truncation_ablation` | `precision_cap` | same |
| `replay_rank_perturbation_control` | `precision_cap` | same |
| `replay_curvature_rebalance_ablation` | `precision_cap` | same |

**Runnable now** (take a batch `x` or checkpoint tags, not bundles):
`sigma_lr_spectrum_by_site`, `sigma_lr_spectrum_report`, `sigma_lr_report`,
`stiffness_report`, `spectrum_across_checkpoints`, `bracket_precision_lr_max`.

**Expected unblock: step ≈45,000.** The pre-clip grad-norm p99 is growing
exponentially with a ≈5,200-step doubling time (§3), projecting to cross
`capture_threshold = 100` around step 45,300 — i.e. the run itself should
produce the first bundle shortly before the 50,000-step stop. **This is the
main reason training was prioritised over these diagnostics.**

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

### 2.6 The moment a bundle exists (expected step ≈45,000) — **BLOCKED**

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

| quantity | fit | prediction at step 50,000 |
|---|---|---|
| `omega*dt` p50 | `-2.3531 + 0.3248·ln(step)`, R²=0.9904 | **1.16** |
| `omega*dt` max | `-5.6224 + 0.7514·ln(step)`, R²=0.9759 | **2.51** |
| `over_wall` | log-normal, σ held at 0.199 | **≈0.3%** |
| log-normal σ | 0.191 → 0.198 over 11,000 steps | **stays ≈0.20** |
| grad p99 | exponential, doubling 5,193 steps, R²=0.983 | crosses **50 at step ≈40,100**, **100 at ≈45,300** |
| grad p90 | doubling 11,581 steps | ≈4.5 |
| grad median | doubling 23,548 steps | ≈1.4 |
| val PPL | `loss ~ step^-0.0581`, n=54 | **≈75** |

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

## 6. After this analysis

Decide `PROBE_MAX_STEPS`: clear to `None` for the full 100,000-step arm, or
set another stop. The WSD schedule is a pure function of `TOTAL_STEPS` and
is unaffected by either choice; the decay phase does not begin until step
65,000, so any stop before then costs nothing but a pause.
