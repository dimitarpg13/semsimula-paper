# A Lost Session, clip_then_sum in Production, and a Replay Fidelity Bug

**What this note is.** §46–§48 of the training-instabilities record: a checkpoint-recompute divergence that cost a session, the first production validation of `clip_then_sum`, and the replay-fidelity bug that predated it. It was split on 2026-11-21 from [`CfC_BAOAB_Integrator_and_Mitigations.md`](https://github.com/dimitarpg13/semsimula-paper/blob/5551315/companion_notes/CfC_BAOAB_Integrator_and_Mitigations.md) (written August to October 2026). The text is that document's, unchanged except for its links, and keeps its section numbers, so a reference to "§N" anywhere in the record or in other notes finds its note through [the index](README.md).

## Contents

- [46. A Checkpoint-Recompute Divergence Took Down the Whole Session: 4,252 Steps Lost, and a Wall-Clock Autosave Added](#46-a-checkpoint-recompute-divergence-took-down-the-whole-session-4252-steps-lost-and-a-wall-clock-autosave-added)
- [47. First Production Validation of clip_then_sum (§45.4): Plateau Broken, No New Spikes Through Step 61,650](#47-first-production-validation-of-clip_then_sum-454-plateau-broken-no-new-spikes-through-step-61650)
- [48. replay_spike_batch Predates clip_then_sum: a Diagnostic Fidelity Bug, Found and Fixed, and What the Corrected Replays of Steps 70522, 71194, and 71703 Show](#48-replay_spike_batch-predates-clip_then_sum-a-diagnostic-fidelity-bug-found-and-fixed-and-what-the-corrected-replays-of-steps-70522-71194-and-71703-show)

---

## 46. A Checkpoint-Recompute Divergence Took Down the Whole Session: 4,252 Steps Lost, and a Wall-Clock Autosave Added

Attempting `replay_clip_ablation(52940)` (§45) hit
`torch.utils.checkpoint`'s `CheckpointError` ("Recomputed values for the
following tensors have different metadata than during the forward
pass"). Ruling out `replay_clip_ablation` itself as the cause took one
step: `replay_spike_batch(52940)`, an unrelated, already-working replay
path, failed with the same error immediately after. Both walk the
identical captured-bundle -> forward/backward-under-checkpoint code
path, so the divergence lives in something shared -- most likely the
gumbel-softmax top-`k` routing in `model_parf_sparse.py`, which draws
fresh randomness on every call and sits inside nested
`torch.utils.checkpoint` regions in `_fock_layer_step`
(`model_fock_parf_multixi.py`): a checkpoint's recompute pass must
retrace *exactly* the same control flow as its original forward, and a
discrete routing decision that differs between the two (even from the
same nominal RNG state, if consumed in a different order or count due to
some other nondeterminism upstream) changes downstream tensor shapes --
consistent with the actual mismatches observed (e.g. `[8,512]` int64 vs.
`[8,512,384]` float32, a shape/dtype pair that looks like a routing
index tensor being compared against a routed activation tensor).
Attempting `torch.utils.checkpoint.set_checkpoint_debug_enabled(True)`
to localize it further made the process hang instead of erroring, and a
third, different bundle (`replay_spike_batch(55919)`) then also hung
with no output at all -- at that point the working hypothesis shifted
from "a latent model bug" to "this specific CUDA session/process is
corrupted" (the debug-mode hang in particular is not explicable by a
routing bug alone).

### 46.1 The corrupted session compounded the loss

Recovering from a corrupted session should just mean "disconnect and
resume from the last checkpoint" -- costly (checkpoints save only every
`EVAL_INTERVAL`-gated best-val-improvement events, not on a fixed step
cadence, so the gap since the last one can be large) but bounded and
recoverable. Two things made it worse here. First, interrupting the
hung `replay_spike_batch(55919)` cell left the live training loop
sitting at step 56,752 with no recent checkpoint, so a manual
`save_manual_checkpoint(step + 1)` was attempted to avoid losing more
than necessary -- but the same session corruption meant `torch.save`
itself hung (1m50s with no completion), including for a bare
optimizer-state save to local disk, i.e. this was not a Drive I/O issue
but something wrong with the CUDA context itself (most likely: reading
a tensor's data off a corrupted CUDA context blocks even when the write
target is healthy local disk). Second, interrupting *that* hung save
left a partially-written file, `_step56753_manual.pt`, on disk --
`torch.load` verification later failed on it ("missing data file"),
confirming the interruption had caught `torch.save` mid-write rather
than after a slow-but-complete write. Net result: the file that was
specifically created to avoid losing progress could not itself be
trusted, and the actual last verified-good checkpoint was step 52,500 --
a loss of 4,252 steps, not just "since the last periodic save" but
inflated further by the failed manual-save attempt consuming the window
during which the corruption was discovered.

### 46.2 Mitigation: a wall-clock autosave independent of step count or val-improvement

None of the existing checkpoint triggers (`EVAL_INTERVAL`-gated
best-val-improvement, manual on-demand) are tied to wall-clock time, so
none of them protect against Colab's hard ~24h runtime cutoff landing in
an unlucky gap -- which is exactly what happened one session later: this
one ran to the 24h limit and was torn down mid-session with no warning,
independent of the corruption incident above. `AUTOSAVE_WALLCLOCK_HOURS
= 23.5` (Cell 6, alongside the other interval configs) adds a
per-process, fire-once safety net: every training step reads
`/proc/uptime` (actual VM boot-relative uptime, read fresh from the
kernel every call -- deliberately *not* `t0`/`time.time()`-based, since
`t0` is local to a single `run_training()` call and resets on every
interrupt-and-resume within the same still-alive session, which would
make a `time.time()`-based check blind to elapsed wall-clock time across
exactly the kind of interruption this feature exists to survive), and
once uptime crosses the threshold, calls `save_manual_checkpoint(step +
1)` exactly once for the rest of the process. The save is wrapped in a
bare `try/except` that logs and continues rather than propagating: given
§46.1's evidence that `torch.save` can itself hang or silently corrupt
its output under session-level failure, a save-failure here must not be
allowed to take an otherwise-healthy training loop down with it, and a
printed traceback is enough to prompt a manual intervention if it ever
fires. 23.5h (vs. the ~24h limit) is deliberately conservative -- it
needs enough margin for the save itself to complete (observed up to
~2min under healthy conditions, more under duress) plus whatever step
happens to be running when the check trips.

## 47. First Production Validation of `clip_then_sum` (§45.4): Plateau Broken, No New Spikes Through Step 61,650

The clean session resumed from the step-52,500 restore (§46.1) with
`clip_then_sum` active for `E`/`P` from the very first step
(`[clip-then-sum] active for groups ['E', 'P'] (2 param(s) total),
threshold=0.3`). Across the ≈9,100 steps and ~21.8h of continuous
wall-clock logged since (step 52,550 at 438s of process uptime through
step 61,650 at 78,962s), two signals together give this mitigation its
first real production evidence, not just the offline/numeric validation
§45.4 already had going in.

### 47.1 Signal 1: `E`/`P` have not led a single step in ~9,100 steps

No `top[...]` entry across this entire window names `E` or `P` -- every
logged leader is one of the `override:*` groups, `V_theta`, or
`raw_m_bias`, exactly the set §45.4 predicted would remain once `E`/`P`'s
own contribution is capped at the microbatch level before it can ever
dominate a step's `top[...]` ranking. Only one `[spike]` bundle capture
fired in the whole window (step 55,405, pre-clip total grad=140.6,
led by `override:creation_gate=112.1` with `override:depth_code`,
`override:register`, `reverse_ch`, and `V_theta` following -- an
already-known non-`E`/`P` mechanism, not a new one), and the last
~1,600+ steps of the window (60,050 through 61,650) produced zero
`[spike]` captures at all, the quietest stretch logged since before the
§46 incident. This is exactly the side effect §45.4 flagged as "not yet
observed running live": `E`/`P`'s post-clip group norm now sits around
the ~0.5 order that offline validation predicted, well under
`CAPTURE_SPIKE_THRESHOLD=100`, so the failure mode this mitigation
targets appears to have simply stopped firing rather than merely
shrinking.

### 47.2 Signal 2: `val_ppl` broke the long-standing ~98-100 plateau

`>>> EVAL step 60,500  val_loss=4.5205  val_ppl=91.88  best=91.88  ***
NEW BEST ***` -- a decisive break below the plateau this run had been
stuck at since long before §46: §44 had already called 98.45 (step
52,500, the checkpoint this session resumed from) "the first genuine
improvement on the 100.47 record since it was set," and 91.88 is a
further ~6.7 percent reduction from that mark, clearly outside the
92-104 band the run's evals had been oscillating in previously. Both the
step-specific (`..._step60500_best.pt`) and canonical (`..._best.pt`)
checkpoints saved successfully, so the gain is durable independent of
anything that happens later in the session. The two evals immediately
after (step 61,000: val_ppl=94.11; step 61,500: val_ppl=95.25) sit back
inside the old noise band, which reads as normal WSD stable-phase
variance around a now-lower center rather than a regression -- consistent
with `EVAL_ITERS=40` giving each eval a low-noise, not resampled-per-call,
estimate. The periodic causal-leak probe fired `[CLEAN]` at step 60,000
(honest_PPL=62.06, standard_PPL=59.62, diff=+0.0401 nats), corroborating
that the gain is a genuine capability improvement rather than a leak
artifact.

### 47.3 What this does and does not confirm

`bproj_sig` continued its pre-existing slow climb across the same window
(28.45 at step 52,550 to 30.35 at step 61,650) at an unchanged rate --
expected, since `clip_then_sum` bounds `E`/`P`'s *applied step*, not the
chronic low-rank stiffness (mechanism A, §41.2) that drives `bproj_sig`'s
drift; the two are orthogonal by design and this window offers no
evidence either way on mechanism A. One eval point beating the plateau by
this much, one quiet ~9,100-step window, and one clean leak probe are a
genuinely strong first signal, but still a single window: the schedule's
decay phase starts at step 65,000, and a few more evals through that
transition (plus continued absence of `E`/`P`-led `[spike]` events) would
turn this from "first quantitative validation" into "confirmed fix."

## 48. `replay_spike_batch` Predates `clip_then_sum`: a Diagnostic Fidelity Bug, Found and Fixed, and What the Corrected Replays of Steps 70522, 71194, and 71703 Show

Two new `watchdog-hard` reloads fired in the ~700 steps after §47's window
closed -- step 70,522 (pre-clip total grad=887.9, led by
`override:depth_code=752.2`) and step 71,194 (691.0, a flatter three-way
tie between `override:register=441.5`, `override:creation_gate=432.2`, and
`override:reverse_channel_scale=406.3`) -- plus a second, larger
reproduction of the `reverse_channel_scale`-exceeds-total anomaly at step
71,703 (361.3 vs. a reported total of 233.1). Replaying all three with
`replay_spike_batch`/`attribute_spike_rows` to dig into them surfaced a
diagnostic-tool bug, not a training-run problem -- worth documenting
precisely because it looked, at first, exactly like the kind of
non-determinism/session-corruption signature §46 had just been caused by.

### 48.1 The symptom: `E`/`P` inflate to 1,000+ in replay, an 80-105% fidelity gap

All three replays failed `replay_spike_batch`'s own fidelity check by a wide
margin -- 105.1% (70522: replayed 1821.4 vs. recorded 887.9), 90.5% (71194:
1316.6 vs. 691.0), and 77.5% (71703: 413.7 vs. 233.1) -- each one dwarfing
the 5% warning threshold that check exists to catch. The signature that
ruled out nondeterminism, though, was immediate: for all three steps,
*every group other than `E`/`P`* matched the bundle's captured
`top_groups` **exactly**, to displayed precision (e.g. 70522:
`override:depth_code` 752.20 replayed vs. 752.2 captured,
`override:creation_gate` 429.52 vs. 429.52, `V_theta` 170.85 vs. 170.85,
down to `raw_logfreq_alpha` at 32.02 vs. 32.02). `E` and `P` were the only
groups off, and not subtly -- they replayed at 1124.55 / 792.43 / 241.74
respectively, an order of magnitude above everything else and, at 70522
and 71194, larger than the entire recorded total. A real nondeterminism
bug does not reproduce seven other groups bit-for-bit while blowing up
exactly two specific ones by 3-4 orders of magnitude; a missing piece of
training-loop logic that happens to touch only those two groups does.

### 48.2 Root cause: the replay helpers predate `clip_then_sum` (§45.4)

`clip_then_sum` (implemented live for `CLIP_THEN_SUM_GROUPS = {'E', 'P'}`,
Cell 6, threshold 0.3) clips each microbatch's own `E`/`P` gradient
*before* folding it into the running total across `GRAD_ACCUM`, replacing
the normal successive-`.backward()`-into-the-same-`.grad`-tensor
accumulation for just those two groups. That is precisely why a captured
step's `top_groups` never lists `E`/`P` at all (their post-clip group norm
sits around the ~0.5 order this bounds it to, invisible next to
`override:depth_code`-class numbers in the hundreds) -- and precisely why
`replay_spike_batch`, whose per-microbatch loop was written before
`clip_then_sum` existed and just does a plain `(loss / grad_accum
).backward()` with no clip in between, reconstructs `E`/`P`'s *raw,
unclipped, fully-summed* gradient instead. For a token/positional
embedding table, that raw per-microbatch gradient is large by nature
(concentrated in the handful of rows the microbatch's tokens actually
touch), and without the per-microbatch clip, four microbatches' worth
accumulate freely -- landing at ≈1,100+ instead of the ~0.5-order value the
live optimizer actually saw and applied.

### 48.3 The fix: mirroring the exact splice for `replay_spike_batch`; a documented caveat instead for `attribute_spike_rows`

Three small helpers were added to Cell 6d, reproducing Cell 6's own
`clip_then_sum` mechanics exactly (same `nn.utils.clip_grad_norm_` call,
same running-total-keyed-by-`id(param)` accumulation, same post-loop
splice into `.grad`):

- `_cts_group_params(mdl)` -- resolves `CLIP_THEN_SUM_GROUPS` to actual
  parameters via `assign_clip_group`, fresh against whichever `mdl` is
  passed in (not the live `_CLIP_THEN_SUM_PARAMS` global, which is bound
  to `model` at Cell 6 run time). Returns `{}` -- a true no-op -- if
  `CLIP_THEN_SUM_GROUPS`/`PER_GROUP_CLIP` aren't configured at all, so
  this stays safe against pre-§45.4 notebooks or bundles.
- `_cts_apply_microbatch(cts_params, cts_running)` -- called right after
  each microbatch's `.backward()`, before the next one's; jointly clips,
  accumulates, and zeros `.grad` for the tracked groups.
- `_cts_splice_back(cts_params, cts_running)` -- called once after the
  full `GRAD_ACCUM` loop, replacing the (already-zeroed) `.grad` with the
  accumulated, per-microbatch-clipped total.

`replay_spike_batch` now calls these at exactly the two points Cell 6
does. `attribute_spike_rows` deliberately does **not** get the same
splice: it replays one row completely alone (each row's own share of the
aggregate, RNG reset before every row -- see its own docstring caveats),
so there is no multi-row microbatch gradient for a joint, threshold-based
clip to act on in the first place. Reproducing `clip_then_sum` per single
row would be a different, not-live-representative computation, not a
fidelity fix. Its docstring got a third caveat instead, spelling out that
`norm_of_summed_grad` for `E`/`P` (or any future `CLIP_THEN_SUM_GROUPS`
member) is expected to diverge from `replay_spike_batch`'s figure for
that reason, and that its raw per-row numbers remain exactly what the
`top1_share`/`top3_share` concentration metrics need.

### 48.4 Corrected fidelity: 0.0% across all three replays

| step | pre-fix fidelity gap | post-fix fidelity gap |
|---|---|---|
| 70522 | 105.1% | **0.0%** (887.9 vs. 887.9) |
| 71194 | 90.5% | **0.0%** (691.0 vs. 691.0) |
| 71703 | 77.5% | **0.0%** (233.1 vs. 233.1) |

`E`/`P` are absent from all three top-parameter lists post-fix, consistent
with the live logs' own `top[...]` entries and with §47.1's finding that
`E`/`P` have not led a single event since `clip_then_sum` went live.

### 48.5 The 71703 anomaly, corrected: the blind spot is 202.3, not 139.8

§44.1 first identified that `WATCHDOG_EXCLUDE_GROUPS` (excluding
`reverse_channel_scale`/`reverse_ch` from the aggregate both the EMA and
`GRAD_NORM_HARD_TRIGGER` read) leaves the watchdog structurally blind to
those two groups; this is now the third reproduction of that exact
mechanism, and -- once the `E`/`P` contamination above is removed from the
numbers -- the largest gap observed yet:

| | watchdog-visible total | true total (incl. reverse groups) | gap |
|---|---|---|---|
| 70660 (§ earlier) | 118.8 | 242.1 (`reverse_channel_scale` alone) | -- |
| 71703, pre-fix (wrong) | 413.7 | 553.5 | 139.8 |
| 71703, post-fix (correct) | **233.1** | **435.4** | **202.3** |

`reverse_channel_scale` alone replays at 361.3, `reverse_ch` at 68.3;
`233.1` also now matches `bundle['pre_clip_grad_norm']` exactly, as it
should (`WATCHDOG_EXCLUDE_GROUPS` is exactly what `pre_clip_grad_norm`
excludes at capture time too).

### 48.6 Step 70522: `depth_code` and `reverse_channel_scale` are two different mechanisms riding together, not one

With `E`/`P` no longer swamping the top-parameter list, `attribute_spike_rows`
was run against the four real top params
(`V_theta.depth_code`, `creation_gate_qkv.W_V.weight`,
`reverse_channel_scale`, `V_theta.bank.banks.0.B_proj.bias`). Row
`mb=3,row=2` is the top row for every one of them, but by very different
margins:

- `reverse_channel_scale`: **top1_share = 0.57** -- this one row
  contributes 7.15 of a 12.49 row-sum total; the next-highest row is only
  0.66. A genuine single-row blowup.
- `V_theta.depth_code`: top1_share = 0.34 -- elevated, but far more
  broadly spread across the batch, consistent with `depth_code`'s usual
  chronic, distributed contribution (§41.2) rather than a single outlier.

The same row's `layer0_h_grad` (0.65) is 6-10x every other row's
(typically 0.01-0.1), while its `ntp` loss (4.37) sits unremarkably inside
the batch's normal 4.2-5.2 range -- i.e. this is not simply "the hardest
example in the batch," it is specifically a row whose gradient the
reverse channel (and, to a lesser extent, `depth_code`) reacts to sharply.
Read together, step 70,522 looks like two co-occurring mechanisms rather
than one: `depth_code`'s usual broad elevation (the reason this event's
`top_groups` calls it "depth_code-dominant" by raw magnitude) plus a
genuinely localized, single-row `reverse_channel_scale` spike riding along
in the same step.

### 48.7 Step 71194: two distinct hot rows, not one shared cause -- and a new `creation_gate_qkv.log_tau` signal

> **Corrected 2026-09-09, see §49.4.** The per-row concentration figures in
> this subsection are measurements of `attribute_spike_rows`' *isolated-row
> surrogate*, which at this event reconstructs well under 1% of the real
> gradient (register 14's true batch gradient is 345.38; the entire row-sum
> for `log_tau` is 2.10). This is the same §41.5 reconstruction gap that
> invalidated the magnitude claims at step 47,116, recurring on a different
> parameter. The *ranking* of which rows are implicated survives and is what
> the bullets below support; the original reading -- "81% of `log_tau`'s
> gradient comes from one row" -- does not. §49 replaces it with a
> per-register decomposition taken from the full-batch pass, where no such
> gap exists.

Tracking the real top-4 params (`register_embed`, `reverse_channel_scale`,
`creation_gate_qkv.log_tau`, `creation_gate_qkv.W_K`) shows the "three-way"
shape in the live log (`register`/`creation_gate`/`reverse_channel_scale`
all comparably large) is actually **two separate localized triggers
landing in the same optimizer step**, not one shared root cause:

- `mb=2,row=2` dominates `register_embed` (top1_share=0.51),
  `creation_gate_qkv.W_K` (0.34), and especially
  `creation_gate_qkv.log_tau` -- **top1_share=0.81**, the sharpest
  single-row concentration seen in any replay to date (`log_tau` controls
  the creation gate's cumulative-softmax temperature). This row barely
  registers for `reverse_channel_scale` (1.87, well behind that group's
  own top row).
- `mb=1,row=5` dominates `reverse_channel_scale` instead (top1_share=0.22
  overall, but its own largest single value at 2.94, ahead of `mb=2,row=2`'s
  1.87) while contributing almost nothing to `log_tau` (0.038 vs. row 2's
  1.71).

`creation_gate_qkv.log_tau` reaching 345.4 is a genuinely new signal, and
it was visible only once the `E`/`P` contamination was removed -- it did
not appear in any earlier, pre-fix top-parameter list. Two distinct
implicated rows, two distinct downstream groups, coinciding in the same
step is what actually produced 71194's "three-way tie" appearance in the
coarse per-group log -- not a single row destabilizing three mechanisms at
once.

What the row axis cannot tell us, given the reconstruction gap above, is
*why* `log_tau` is large. That turned out to be a question about the
**register** axis, not the row axis, and §49 answers it: 100.0% of the
345.4 is a single register.

### 48.8 Proposed (not yet implemented) mitigation: the creation gate never got the reverse channel's QK-norm hardening

Reading the two implicated modules side by side turns up a structural
asymmetry that was not deliberate, and that predicts exactly the
`log_tau` behaviour §48.7 measured. `ReverseChannel` was hardened under
`stable=True` (§10.12 of the design doc / the E5c arm) with three
explicit bounding devices: QK-normalisation of $q$ and $k$ plus a
*clamped* learnable `logit_scale` (so logits stay bounded regardless of
$\lVert q \rVert$, $\lVert k \rVert$, and the softmax cannot saturate into
spiky-gradient regimes), per-token RMS-normalisation of the output
`Q_force`, and an optional pre-LayerNorm on its inputs. This run has all
of that switched on.

`QKVCreationGate_v21` has none of it. It computes raw
`scores = Q · K` with no normalisation of either factor and then divides
by the learned per-register temperature, `scores = scores / tau` with
`tau = log_tau.exp().clamp(min=1e-4)`. Writing $\tilde{s} = s/\tau$ for
the scaled scores, the temperature gradient is

$$\frac{\partial L}{\partial \log \tau}  = -\sum \tilde{s} \odot \frac{\partial L}{\partial \tilde{s}}$$

-- that is, **proportional to the scaled score magnitudes themselves**,
which are unbounded because $\lVert Q \rVert \lVert K \rVert$ is
unbounded, and which are additionally divided by a *learned, per-register*
$\tau$ that nothing constrains to stay comparable across registers.
Anything that produces large scaled scores therefore lands directly, and
linearly, on `log_tau`'s gradient.

This derivative is the one part of this subsection that survived contact
with the data unchanged. What did not survive is the resolution at which
it was tested: the paragraph below guessed that the offending scores would
belong to a single *row*, and §49 shows they belong to a single
*register*.

Two supporting observations from the same replays. First,
`create_entropy` sits at roughly 6.0 at layer 0 but collapses to
~0.47-0.74 at layers 2-7 -- i.e. the creation gate is running
near-saturated exactly where these events originate, the regime the
reverse channel's device 1 exists to prevent. Second, the prefix-causal
readout stabilises its cumulative softmax with a *constant* shift
(`clamp=40.0`) rather than a running max, deliberately, because a
constant shift is what makes each position's output a bit-exact function
of positions $\le t$ and the causality probe return literal 0.0. A useful
property, but it also means the absolute score scale matters numerically
in a way a max-shifted softmax would hide: the usable band is fixed, so
any drift in $\lVert Q \rVert \lVert K \rVert$ or in $\tau$ moves the whole
distribution around inside it.

The proposed change is to mirror `reverse_channel_stable` for the
creation gate, behind its own opt-in config flag: L2-normalise $Q$ and
$K$, and replace the bare learned `tau` divisor with a clamped learnable
scale, keeping the existing per-register granularity (`log_tau` is shape
$(M,)$, one entry per register, and that granularity is worth preserving
-- it is also what makes the per-element probe in §48.9 able to say
*which* register is responsible). This is deferred rather than
implemented for two reasons: it changes the forward function, so unlike
`clip_then_sum` it cannot be picked up by an in-place
`run_training(next_step, ...)` resume and needs a fresh arm or a
checkpoint-compatible flag; and the mechanism is not yet confirmed.

The confirming test is cheap and stated in advance, so it can falsify as
well as confirm: replay the capture and record the creation gate's
pre-softmax scores per row. **If** the gradient-hot row's scores are
outliers against the other 31 rows, the score-magnitude mechanism holds
and QK-norm addresses it at source. **If** the hot row's scores are
unremarkable and only the upstream $\partial L / \partial \tilde{s}$ is
large, then QK-norm would not fix anything here, and the real lever is
the shared early-layer backward amplification those rows have in common
(§48.6-48.7) rather than anything specific to the creation gate.

> **Outcome, 2026-09-09: the stated test returned the second branch, but
> the test itself was mis-specified and neither branch's conclusion
> follows.** The peak $\lvert\tilde{s}\rvert$ per row is a maximum taken *over all 32
> registers*, and register 14 -- present in every row -- dominates that
> maximum everywhere, so the statistic was very nearly constant across
> rows and had no power to separate them. The gradient-hot row duly came
> back unremarkable (rank 8/32 at 70522, rank 31/32 at 71194), which the
> test's own wording would have read as "QK-norm would not fix anything
> here." Re-running the same measurement resolved **per register** instead
> of per row reverses that: the score-magnitude mechanism is confirmed,
> and QK-norm does address it at source. §49 has the corrected test and
> the resulting mechanism.

### 48.9 Status and next steps

The fix is live in Cell 6d; every future replay of a `clip_then_sum`-era
capture now gets a correct fidelity check automatically, with no per-call
action needed. Two follow-ups noted but not yet done:

- `replay_clip_ablation`'s own docstring (§45) still describes
  `sum_then_clip` as "the live training loop's current order, Cell 6" --
  that was true when it was written but is now stale for `E`/`P`
  specifically, the same class of staleness this section just fixed
  elsewhere. Low risk (it doesn't affect that helper's actual arithmetic,
  only a comment), but worth correcting alongside any other doc pass.
- `creation_gate_qkv.log_tau` at 345.4 was invisible until per-parameter
  replay attribution looked underneath the group-level clip:
  `override:creation_gate=432.16` alone gives no hint that 345.4 of it is
  one single parameter. `log_tau` is not unprotected -- it shares a
  ceiling with `creation_gate_qkv`'s `W_Q`/`W_K`/`W_V` via the
  `creation_gate` override group (`GRAD_CLIP_OVERRIDES`, 0.3) -- but §49.8
  shows that sharing is itself a problem. Pursued in §49.
