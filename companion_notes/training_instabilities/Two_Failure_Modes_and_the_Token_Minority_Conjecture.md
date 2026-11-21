# Two Failure Modes, and the Token-Minority Conjecture Falsified

**What this note is.** §38–§39 of the training-instabilities record: seven replays showing two distinct failure modes, and the falsification of the token-minority conjecture (the localized mode is batch-wide). It was split on 2026-11-21 from [`CfC_BAOAB_Integrator_and_Mitigations.md`](https://github.com/dimitarpg13/semsimula-paper/blob/5551315/companion_notes/CfC_BAOAB_Integrator_and_Mitigations.md) (written August to October 2026). The text is that document's, unchanged except for its links, and keeps its section numbers, so a reference to "§N" anywhere in the record or in other notes finds its note through [the index](README.md).

## Contents

- [38. Seven Replays In: Two Distinct Failure Modes, Not One -- and Only the Localized One Has Crossed 500](#38-seven-replays-in-two-distinct-failure-modes-not-one----and-only-the-localized-one-has-crossed-500)
- [39. The Token-Minority Conjecture Is Falsified Twice Over: the Localized Mode Is Batch-Wide, Not Batch-Specific](#39-the-token-minority-conjecture-is-falsified-twice-over-the-localized-mode-is-batch-wide-not-batch-specific)

---

## 38. Seven Replays In: Two Distinct Failure Modes, Not One -- and Only the Localized One Has Crossed 500

### 38.1 The first genuine `watchdog-hard` reload, replayed

Step 41,837 (pre-clip 528.0) is this run's first `GRAD_NORM_HARD_TRIGGER`
event since Phase 1/2 went live in §35, replayed alongside the also-newly-
captured step 41,824 (265.8, under the reload line but over
`CAPTURE_SPIKE_THRESHOLD`). Both replayed with the same excellent fidelity
as the earlier five (all seven now <0.0013% gap), so the comparison below
draws on n=7, not n=1:

| step | pre-clip | leader (norm) | layer 0 -> 3 ($h$-grad) | layer0/layer3 | profile |
|---|---|---|---|---|---|
| 37,763 | 160.4 | `depth_code`=88.3 | 0.169, 0.102, 0.044, 0.027 | 6.3x | smooth cascade |
| 40,043 | 379.8 | `reverse_channel_scale`=274.9 | 0.081, 0.064, 0.040, 0.025 | 3.2x | smooth cascade |
| 40,387 | 369.3 | `depth_code`=258.8 | 0.082, 0.058, 0.033, 0.021 | 3.9x | smooth cascade |
| 41,318 | 432.8 | `E`=266.7 | 0.054, 0.042, 0.028, 0.021 | 2.6x | smooth cascade |
| 41,824 | 265.8 | `register`=231.8 | 0.087, 0.070, 0.047, 0.032 | 2.7x | smooth cascade |
| 39,983 | 235.5 | `depth_code`=208.9 | 16.45, 12.20, 6.08, 0.33 | 50x | **localized** |
| 41,837 | 528.0 | `depth_code`=424.6 | 31.07, 12.61, 1.46, 0.17 | 177x | **localized** |

### 38.2 Two failure modes, discriminated cleanly by the per-layer profile

Five of the seven replays reproduce the smooth, monotone cascade signature
from §33.4/§35: layer 0's $h$-gradient is only 2.6-6.3x layer 3's, decaying
gently across all 8 layers, consistent with a single gain compounding as
it backpropagates the full stack. The other two -- 39,983 and 41,837 -- are
categorically different: layers 0-2 are **50-300x larger** than the same
layers in every one of the other five replays, with a sharp cliff by layer
3, after which they rejoin the same gentle tail everyone else has. This is
not a one-off artifact of a single replay (the possibility flagged when
39,983 was first seen): it has now recurred, at roughly double the
severity (layer 0: 16.45 -> 31.07; total: 235.5 -> 528.0), a second
independent instance of the same distinctive shape.

**Leadership does not predict the mode.** `depth_code` leads both localized
events, but it also leads 40,387 (258.8, its second-largest reading of all
seven) -- a perfectly smooth cascade, not localized. The per-layer profile,
not which group tops the leaderboard, is the reliable discriminator between
the two modes.

**`register` leading 41,824 (231.8) is itself a new observation** -- no
prior replay in this run had `register` anywhere near the top of the
per-group breakdown; it is otherwise a normal smooth-cascade event.

### 38.3 Working hypothesis: the localized mode is what actually forces a reload

Of the five smooth-cascade events, none exceeds 432.8 pre-clip. The one
event that has crossed `GRAD_NORM_HARD_TRIGGER=500` in this run's entire
Phase-1/2-instrumented history is a localized event (41,837). Read
together with 39,983 being the second-most-severe localized case and also
comfortably under the reload line at the time, the working hypothesis is:
the smooth cascade self-limits under per-group clipping (each group's own
threshold caps its own contribution before the aggregate can compound much
past ~400), while the layer-0-2 localized blowup does not follow the same
per-group-clip-bounded dynamics and is the actual driver of this run's rare
genuine reloads.

**Caveat: this is n=2 for the localized mode.** It is a real, recurring,
now-twice-confirmed signature, and the only one of the two modes to have
reached the reload threshold so far -- but two events is not enough to
rule out coincidence in which mode happens to cross 500 first. The
falsification test is straightforward: keep harvesting captures via
`replay_all_captures()`; if a *smooth-cascade* event eventually also
crosses 500, or a third localized event stays comfortably under it, the
hypothesis needs revising.

### 38.4 What is not yet known

Nothing in the per-parameter or activation-extreme breakdown (V_theta bank
exponents, `xi_shifted_norm`, destruction-gate outputs, creation-gate tau)
cleanly separates the two localized events from the five smooth-cascade
ones on inspection so far -- the layer-0-2 $h$-gradient profile is the only
discriminator found. What specifically differs about the forward pass at
layers 0-2 on a localized-mode step (a particular token pattern, a
register-creation event, a V_theta well boundary crossing, ...) is open;
answering it would need comparing the *forward* activations (not just
backward gradients) between a localized and a smooth-cascade capture at
matched layers, which the current instrumentation does not yet do.

### 38.5 Status

**Descriptive finding, not yet a mitigation.** Documents the pattern
across the 7 captures replayed to date; no code change proposed here.
Next step: keep accumulating captures under `CAPTURE_SPIKE_THRESHOLD=200`
and re-run this comparison at larger n to firm up (or break) both the
mode-discrimination claim (§38.2) and the reload-driver hypothesis (§38.3).

### 38.6 Deepening forensics on the two existing localized captures, without waiting for new ones

§38.4 flagged that nothing in the *current* forward-activation breakdown
discriminates the two modes. Before spending wall-clock time waiting for a
third localized event, it's worth exploiting a fact already true today:
every `_spikebatch.pt` bundle pins the exact pre-step weights, the exact
offending microbatch tokens, and the RNG state, and Cell 6d's replay is
already proven bit-exact (<0.0013% fidelity gap on all 7). That means
steps 39,983 and 41,837 can be re-interrogated with *more* instrumentation
right now, with zero new training time -- so Cell 6d gained three
additions the same day, rather than waiting on more captures:

1. **Fixed a real per-layer attribution bug.** `creation_gate_qkv`,
   `reverse_ch`, and each `V_theta` bank are single shared modules called
   once per `_fock_layer_step` invocation (i.e. once per layer, L times
   per forward) -- unlike `destruction_gates`, a genuine per-layer
   `ModuleList`. The old `_record()` hook overwrote the same dict key on
   every layer's call, so `activation_extremes` silently only ever
   reflected **layer 7's** values -- the layers *not* implicated in the
   localized mode. `activation_extremes` is now `{name: {layer: stats}}`,
   layer-resolved for every hooked op. This plausibly explains why §38.4
   found "nothing" in the existing breakdown: it was looking at the wrong
   layers the whole time.
2. **Wired in the model's own per-layer diagnostic buffer.**
   `FockMultiXiPARFLM.set_fock_capture(True)` (the same API the eval-time
   causal/register diagnostics already use) makes `_fock_layer_step`
   append a cheap `@torch.no_grad()` dict per layer per forward:
   `active_frac`, `salience_mean/std`, `reg_cos_sim` (register-collapse
   diagnostic), `destroy_mean`, `create_alpha_max`, `create_entropy`,
   `rev_entropy`, `rev_alpha_max`, `rev_scale`, `qforce_ratio`. `replay_spike_batch`
   now enables it for the replayed forward+backward and reports
   `fock_capture_per_microbatch` (full per-layer table) in the returned
   report, printed as a table in verbose mode.
3. **Added `inspect_spike_tokens(step_tag)`**, pure CPU bookkeeping (no
   model, GPU, or RNG state touched) that decodes the offending
   microbatch's GPT-2 tokens via Cell 3's `tok`, ranks rows by longest
   same-token repeat run and unique-token ratio, and prints a decoded
   snippet of the most degenerate rows -- testing whether a boilerplate
   OpenWebText passage (long whitespace/punctuation/header repeats) is
   what triggers the localized layers-0-2 blowup, independent of anything
   on the gradient side.

**Status: implemented, syntax-checked (`ast.parse` on every notebook
cell), existing `grad_clip_utils`/CfC-BAOAB test suites still pass
unchanged, not yet run against a live GPU session.** Next step: run
`replay_spike_batch(39983)` / `replay_spike_batch(41837)` and
`inspect_spike_tokens(39983)` / `inspect_spike_tokens(41837)` against the
already-captured bundles and see whether either new signal (per-layer
register/gate health, or offending-token degeneracy) separates from the
five smooth-cascade replays.

### 38.7 The deepened forensics discriminate nothing -- but a number already sitting in the Phase-0 data does

§38.6's three additions were run against the two localized captures
(39,983 / 41,837), then, for a genuine control group, against two of the
five smooth-cascade captures (37,763 / 41,318). Both localized events
replayed with the same 0.0% fidelity as before. Raw combined
`replay_spike_batch` + `inspect_spike_tokens` output for 39,983/41,837:
[replay_spike_batch_and_inspect_spike_tokens_39983_41837_output.txt](../../notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_L8probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc/replay_spike_batch_and_inspect_spike_tokens_39983_41837_output.txt).

**Negative result 1 -- token degeneracy, ruled out.** `inspect_spike_tokens`
found nothing: across all 64 rows examined (32 per capture), every single
row has `max_repeat_run = 2` -- the floor value for ordinary English text
(no run of 3+ identical tokens anywhere) -- and unique-token ratios in the
unremarkable 0.29-0.63 range. The decoded snippets are routine OpenWebText
prose (and, incidentally, one Solidity smart-contract snippet mixed into
the 41,837 batch, but that's normal corpus composition, not a cause --
grad norms aggregate over the whole 32-row batch, so nothing here singles
out *which* row would even matter). Boilerplate/degenerate input text is
not the trigger.

**Negative result 2 -- the layer-resolved activation extremes and
`set_fock_capture` gate-health table, ruled out.** Comparing all four
deep replays side by side, the numbers are essentially indistinguishable
between modes. Layer 0's `create_alpha_max` mean is 0.0036-0.0041 and its
`create_entropy` is 6.215-6.224 in *all four* events, localized or smooth
-- to three significant figures, regardless of which failure mode
happened. `V_theta.bank[*].exponent` is in the tens-of-thousands-negative
range in all four. `reverse_ch.Q_force` and `destruction_gates[*].output`
per-layer curves are likewise the same shape in all four. This appears to
be a property of the current model weights at this point in training, not
of the batch or the outcome -- so this round of instrumentation, while a
useful sanity check, was measuring the wrong thing.

**Correction to §38.6 item 1:** `V_theta` banks turned out *not* to share
the per-layer-overwrite bug after all. Re-inspecting real replay output
showed every `V_theta.bank[*]` stat reported at "layer 7" in *every*
capture, localized or smooth -- because `V_theta` is called once per
forward pass on the full depth-stacked `xi` tensor, not once per
`_fock_layer_step`. `_record()` now takes an explicit `layer=-1` override
for these three stats (matching the existing `reverse_ch.logit_scale`
convention) instead of trusting `_current_layer`, which would otherwise
report whichever layer happened to run last.

**Positive result -- it was already sitting in the Phase-0 data,
uncomputed.** Going back to the original 7-event `spike_replay_reports`
JSON ([spike_replay_reports.json](../../notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_L8probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc/spike_replay_reports.json), no
new replay needed) and computing one ratio per event --
`override:depth_code`'s captured group norm divided by the next-largest
group's norm ("`dc_ratio`") -- produces a clean split:

| step | mode | pre-clip | `depth_code` | 2nd-largest group | `dc_ratio` | L0/L3 |
|---|---|---|---|---|---|---|
| 37,763 | smooth | 160.4 | 88.3 | E (81.0) | 1.09 | 6.2 |
| 40,043 | smooth | 379.8 | 163.2 | reverse_channel_scale (274.9) | 0.59 | 3.2 |
| 40,387 | smooth | 369.3 | 258.8 | P (144.8) | 1.79 | 3.9 |
| 41,318 | smooth | 432.8 | 31.2 | E (266.7) | 0.12 | 2.6 |
| 41,824 | smooth | 265.8 | 73.1 | register (231.8) | 0.32 | 2.7 |
| 39,983 | **localized** | 235.5 | 208.9 | P (67.5) | **3.09** | 50.1 |
| 41,837 | **localized** | 528.0 | 424.6 | E (190.7) | **2.23** | 177.6 |

Every smooth-cascade event has `dc_ratio < 1.8`; both localized events
have `dc_ratio > 2.2`. `dc_ratio` is computable straight from data the
watchdog already gathers every step (`per_group_grad_norms`/
`clip_grads_per_group`'s per-group breakdown) with zero replay cost, and
it is mechanistically plausible as the actual signature of the localized
mode: `depth_code` is what shifts `xi` per depth position, so a runaway
gradient there concentrates in the early layers where that shift is
applied, rather than spreading evenly like an ordinary cascade -- which
is exactly the L0/L3 pattern that originally defined the split.

**Caveat -- this cannot yet be validated as a *leading* indicator.**
`training_log.jsonl`'s periodic entries only ever carried the aggregate
`grad_norm`, never the per-group breakdown; the breakdown was only ever
persisted to disk at the 7 spike-capture moments. So there is no way to
retroactively check what `dc_ratio` looks like on ordinary, uneventful
steps, or whether it climbs for several steps before a localized-mode
hard-trigger (a real early-warning signal) versus jumping simultaneously
with no lead time (useful for post hoc labeling, not prevention). Cell
6's periodic `LOG_INTERVAL` log line now also writes `dc_ratio` (the
per-group breakdown needed for it is already computed in memory every
step when `PER_GROUP_CLIP` is on, for clipping itself -- this just
persists one extra derived number, no new computation), so a future
mining pass over an extended run has the data to answer this.

**Status: negative results (token degeneracy, layer-resolved activation
extremes) and the `dc_ratio` correlation established from the existing 7
captures; `V_theta` layer-label correction and `dc_ratio` logging landed
in Cell 6/6d; not yet validated against live per-step data.** Next step:
let training accumulate steps with `dc_ratio` now being logged, then mine
`training_log.jsonl` for its distribution on ordinary steps and its
behavior in the run-up to any future hard-trigger.

## 39. The Token-Minority Conjecture Is Falsified Twice Over: the Localized Mode Is Batch-Wide, Not Batch-Specific

§38.4 established that nothing in the forward-activation breakdown
discriminates the two modes, but flagged that *density*-style measurements
(as opposed to the min/max/mean extremes already tried) had not been ruled
out, and that comparing forward activations at matched layers between the
two modes had not been done. Two such measurements were designed and run
against all four Phase-1/2-instrumented events with a per-layer $h$-gradient
profile on record (37,763 / 41,318 smooth; 39,983 / 41,837 localized). Both
came back negative -- one of them decisively in the *opposite* direction
from the working conjecture.

### 39.1 The conjecture, stated before either result came back

Per-group clipping caps `depth_code` at 0.25 on *every* step, quiet or
spiking (§28's clamp, tightened 0.5 → 0.25 on 2026-08-23 precisely because
`depth_code` was already saturating its old ceiling on every quiet step --
see the `GRAD_CLIP_OVERRIDES` comment in Cell 6). So a localized event's
*applied* `depth_code` update is exactly the same size as a quiet step's --
0.25 either way, whether the pre-clip norm was 88 or 425. Its damage
therefore cannot be magnitude; it has to be direction. The conjecture: the
direction is dictated by a small minority of rows in the batch whose tokens
land near a sharp `V_theta` well at layers 0-2 (the only layers carrying
meaningful salience, §38.7), while the other ~30 rows contribute little.
Two tests were designed to check this *before* either was run, so the
result couldn't be retrofitted to the story:

1. **Per-row gradient attribution** (`attribute_spike_rows`, new in Cell
   6d): replay the captured batch one row at a time (RNG reset to the
   capture's pinned state before every row, so all rows see an identical
   noise draw and are comparable to each other) and measure what fraction
   of the summed `depth_code` row-norms a single row accounts for. Flat
   batch of 32 rows -> baseline 1/32 = 0.031; conjecture predicts the two
   localized events sit well above that and the two smooth events sit near
   it.
2. **`V_theta` exponent occupancy histogram** (added to `replay_spike_batch`):
   per-bank counts of how many token-slots have exponent > -10 (i.e.
   `exp()` >= 4.5e-5, the well still contributes gradient) versus
   underflowed to numerically dead. Conjecture predicts localized events
   show a denser live band than smooth events.

### 39.2 Result 1: per-row attribution -- concentration is *higher* in the smooth mode

| step | mode | pre-clip | `depth_code` top-1-row share | top-3-rows share | layer-0 $h$-grad top-1 share |
|---|---|---|---|---|---|
| 37,763 | smooth | 160.4 | 0.292 | 0.558 | 0.238 |
| 41,318 | smooth | 432.8 | **0.387** | 0.585 | **0.429** |
| 39,983 | localized | 235.5 | 0.217 | 0.424 | 0.149 |
| 41,837 | localized | 528.0 | **0.095** | 0.248 | **0.115** |

(uniform-batch baseline for 32 rows: 0.031)

Raw `attribute_spike_rows` output for step 37,763 (one of the four
events above): [attributes_spike_batch_37763_output.txt](../../notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_L8probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc/attributes_spike_batch_37763_output.txt).

This is the opposite ranking from the conjecture. The two localized events
are the *flattest* of the four -- 41,837, the single most extreme event on
record (layer0/layer3 $h$-gradient ratio 177x, the only one to cross the
500 hard-trigger), has its `depth_code` gradient spread almost evenly
across all 32 rows, barely 3x the flat-batch baseline. The two smooth
events are the more row-concentrated ones, with 41,318 putting 58% of the
`depth_code` gradient on three rows. There is also a clean monotonic
anti-correlation: as layer-0-2 localization gets more severe (39,983's 50x
-> 41,837's 177x layer0/layer3 ratio), the per-row concentration gets
*less* severe (0.217 -> 0.095 top-1 share) in lockstep.

### 39.3 Result 2: exponent occupancy -- no separation in either direction

| bank | 37,763 (smooth) | 41,318 (smooth) | 39,983 (localized) | 41,837 (localized) |
|---|---|---|---|---|
| 0 | 7.6e-6 | 3.8e-5 | 3.1e-5 | 3.1e-5 |
| 1 | 1.5e-5 | 3.1e-5 | 7.6e-6 | 2.3e-5 |
| 2 | 1.4e-4 | 1.4e-4 | 1.4e-4 | 1.4e-4 |
| 3 | 1.4e-3 | 8.8e-4 | 9.3e-4 | 8.7e-4 |
| 4 | 2.4e-3 | 1.1e-3 | 1.6e-3 | 8.6e-4 |

(`live_frac`: fraction of token-slots per bank with exponent > -10)

Bank 2 is 1.4e-4 in all four events; the others are within the same order
of magnitude across modes with no consistent direction. Worth noting on
its own, independent of the mode question: **at every capture, regardless
of mode, over 99.9% of token-well pairs are numerically dead** (exponent
< -10, contributing exactly zero gradient) -- the anisotropic-Gaussian
`V_theta` wells operate in this extremely sparse regime universally, not
just during spikes.

### 39.4 Revised picture: localized is layer-localized but batch-wide

"Localized" in §38 was always a statement about *layers* (0-2 blow up, 3-7
don't), never about *rows*. §39.2 shows those are not the same thing: the
localized mode is layer-localized but **batch-wide** -- something makes
nearly every row in the batch simultaneously hypersensitive at layers 0-2,
rather than one bad token dominating. The smooth mode, perhaps counter to
its name, is the one with real row-level structure.

Combined with §38.4/§38.6/§38.7's prior negative results (token
degeneracy, layer-resolved activation extremes, per-layer gate health all
failed to discriminate the modes), the pattern is now consistent across
five independent forward-pass-side measurements: **nothing about what is
in this particular batch, on this particular forward pass, discriminates
the two modes.** The only thing that ever has is a backward/parameter-
magnitude quantity -- `dc_ratio` (§38.7) -- which is really a statement
about where in *parameter* space the gradient concentrates
(`depth_code`+`V_theta` vs. everywhere else), not about the data.

A batch-wide effect that gets *more* uniform across rows as it gets *more*
severe by layer smells like a property of the shared weights/dynamics
state the model is in going into that step, not of which tokens happen to
be present. This reopens, in sharper form, the `sigma_max(B_k)` /
precision-matrix stiffness line from §28/§31 (previously bracketed and
downgraded as "not the primary driver" for *overall* run stability, a
different question). The sharper form: does a raw weight-space stiffness
proxy -- e.g. `V_theta.bank.banks[k].B_proj`'s spectral or Frobenius norm,
computed from the *parameters themselves*, not their per-token forward
evaluation (which §39.3 just showed doesn't discriminate) -- drift upward
over the steps immediately preceding a localized event, before any
particular batch is even drawn? The `_spikebatch.pt` bundles are single-
instant snapshots and cannot answer this; it needs a trajectory.

### 39.5 Status

**Two forward-pass hypotheses falsified (token-minority via per-row
attribution; well-density via exponent occupancy), both tested against
all four Phase-1/2-instrumented events with a recorded layer profile
before either was run.** Sharpens rather than resolves the open question
from §38.4: the discriminator is confirmed to live on the parameter side,
not the data side. Next step, mirroring the `dc_ratio` pattern of §38.7:
add a weight-space stiffness proxy for the `V_theta` low-rank precision
factor to Cell 6's periodic `LOG_INTERVAL` logging, then mine it
retroactively once more localized events accumulate, checking specifically
whether it trends upward in the steps before a localized-mode hard-trigger
(a real leading indicator) or only coincides with one (useful for labeling,
not prevention).
