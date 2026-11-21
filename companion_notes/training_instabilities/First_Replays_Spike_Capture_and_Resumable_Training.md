# The First Spike Replays, Spike Capture, and Resumable Training

**What this note is.** §35–§37 of the training-instabilities record: the first `replay_spike_batch` result, decoupling spike capture from the reload trigger, and making Cell 6 resumable in place with `grad_clip_utils.py`. It was split on 2026-11-21 from [`CfC_BAOAB_Integrator_and_Mitigations.md`](https://github.com/dimitarpg13/semsimula-paper/blob/5551315/companion_notes/CfC_BAOAB_Integrator_and_Mitigations.md) (written August to October 2026). The text is that document's, unchanged except for its links, and keeps its section numbers, so a reference to "§N" anywhere in the record or in other notes finds its note through [the index](README.md).

## Contents

- [35. Phase 1/2 Validated: First replay_spike_batch Result (Step 37,763) and a Clean Cascade Signature](#35-phase-12-validated-first-replay_spike_batch-result-step-37763-and-a-clean-cascade-signature)
- [36. Decoupling Spike Capture from the Reload Trigger: the Plateau, Not Just the Rare Crisis, Is the Real Target](#36-decoupling-spike-capture-from-the-reload-trigger-the-plateau-not-just-the-rare-crisis-is-the-real-target)
- [37. Making Cell 6 Resumable In-Place, and Extracting grad_clip_utils.py](#37-making-cell-6-resumable-in-place-and-extracting-grad_clip_utilspy)

---

## 35. Phase 1/2 Validated: First `replay_spike_batch` Result (Step 37,763) and a Clean Cascade Signature

### 35.1 A bug fixed first: `torch.load`'s `map_location=DEVICE` broke RNG restore

Running `replay_spike_batch` for the first time against a real capture failed
with `TypeError: RNG state must be a torch.ByteTensor`.
`torch.load(path, map_location=DEVICE, weights_only=False)` remaps every
tensor in the bundle to `DEVICE` (cuda) -- including `rng_state_cpu` and
`rng_state_cuda`'s per-GPU tensors, which `torch.set_rng_state()` /
`torch.cuda.set_rng_state_all()` both require to stay plain CPU
`ByteTensor`s. The model-weights tensors never needed this either:
`replay_spike_batch` already re-maps each one explicitly
(`{k: v.to(DEVICE) for k, v in bundle['model_state_dict'].items()}`) a few
lines later. Fixed by loading with `map_location='cpu'` instead.

### 35.2 Fidelity check: 0.0012% -- bit-exact reproduction confirmed

With the fix in place, the Stage-1 smoke test (`GRAD_NORM_HARD_TRIGGER=110.0`,
§33.5/§34) produced its first `_spikebatch.pt` at step 37,763 (pre-clip total
grad 160.4), and `replay_spike_batch(37763)` reproduced it with:

```
pre_clip_grad_norm_recorded  = 160.39
pre_clip_grad_norm_replayed  = 160.39   (matching groups)
fidelity_gap_pct             = 0.0012%
```

Raw `replay_spike_batch` output (this event and step 41,318, discussed
together with it in §38-§39): [replay_spike_batch_37763_41318_output.txt](../../notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_L8probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc/replay_spike_batch_37763_41318_output.txt).

This settles the open question from §33.5: the RNG/microbatch/weight capture
and the isolated replay reproduce a real training step's forward+backward
essentially exactly, not just "close enough." Phase 1 and Phase 2 are
validated end to end.

### 35.3 The result: a clean cascade-amplification signature, not an isolated spike

The per-layer hidden-state gradient norm (the gradient flowing into each
layer boundary -- i.e. how much of the backward pass's magnitude has reached
that point) is monotone across all 8 layers:

| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| grad norm | 0.169 | 0.102 | 0.044 | 0.027 | 0.019 | 0.013 | 0.008 | 0.005 |

That is roughly a 37x growth from layer 7 down to layer 0, with no bump or
plateau anywhere in the middle. This is exactly the signature §33.4's
cascade-amplification hypothesis predicted (a per-layer $h$-gradient-norm
profile that grows from layer $L-1$ toward layer 0, not an isolated
single-layer spike), and it argues against a single-layer localized cause
(one bad gate or one saturated softmax firing at one specific depth).

### 35.4 `depth_code`'s structural role

The largest single per-parameter gradient at this event is
`V_theta.depth_code` (88.3), narrowly ahead of `E.weight` (81.0) and `P`
(80.9). `depth_code` is used at every layer, so it directly sums the
disturbance from all 8 layers as it accumulates during backprop, rather than
having to survive the full backward pass the way `E`/`P` do (both of which
only see the *cumulative* post-cascade signal once it reaches $h_0$ --
consistent with them landing close behind, not ahead). This is compatible
with either reading: `depth_code` as the stack-wide cascade's natural
aggregation point, or the more specific §33.4 variant where `depth_code`
pushes $V_\theta$ into a locally stiff regime that itself carries the
cascade forward.

### 35.5 Secondary signals: mostly clean, nothing else implicated at this severity

- `creation_gate.tau` sits at 5.6-7.5 (not small/sharp), so the
  "sharp-softmax Jacobian" candidate does not look active here -- though
  `creation_gate.alpha_max` reaching exactly 1.0 for at least one token in
  the batch shows some individual token is winner-take-all even at this
  $\tau$.
- `reverse_ch.Q_force` stays inside [-9.4, 7.6], no outlier blow-up -- the
  "non-conservative reverse-channel kick" candidate is not implicated in
  this event either. `reverse_ch.logit_scale` is a fixed 17.93 across every
  token (it is a single global scalar); worth a follow-up check on whether
  that sits at or near its configured clamp ceiling.
- Several `destruction_gates` swing from near 0 to near 1 across the batch
  (e.g. banks 2, 4, 5, 6), consistent with -- but not confirming -- the
  "hard salience/threshold gating" candidate.
- `V_theta.bank[k].exponent` reaching into the $-10^4$ to $-2\times10^5$
  range is the already-documented underflow regime (most wells are far from
  most tokens); expected, not new information.

### 35.6 Caveat: this event is not yet the target population

Step 37,763's pre-clip total (160.4) sits in the "<200" severity band, where
Phase 0's log-mining (§33.3) found `depth_code` leading 76% of the time --
not the `E`/`P`-dominated severe (≥500) `watchdog_hard_reload` regime the
whole investigation exists to explain. The three top groups here (88.3 /
81.0 / 80.9) are close enough to read as a blend right at that boundary,
which is itself informative (§34's Stage-1 design rationale), but §35.3's
cascade signature and §35.4's `depth_code` attribution should be treated as
a first bit-exact look, not a closed case, until a genuine severe event is
replayed the same way.

### 35.7 Status and next step

**Phase 1/2 fully validated (§35.1-§35.2); one modest-severity replay
analyzed (§35.3-§35.5), with a caveat (§35.6).** `GRAD_NORM_HARD_TRIGGER` is
being restored to `500.0` and `baoab_cfc` production training resumed, per
the two-stage plan (§33.5). The next actionable step is unchanged from
before: replay the next genuine `watchdog_hard_reload` (pre-clip ≥500, the
`E`/`P`-dominated regime per §33.3 Phase 0) the same way, and check whether
the per-layer growth profile holds or sharpens further, and whether `E`/`P`
overtake `depth_code` at the top the way the mined log predicts. (See §36:
the plan below is revised once more real training data came in.)

---

## 36. Decoupling Spike Capture from the Reload Trigger: the Plateau, Not Just the Rare Crisis, Is the Real Target

### 36.1 A ~1,800-step window with zero hard triggers, still stuck on the plateau

After §35's Stage-1 validation, `GRAD_NORM_HARD_TRIGGER` was restored to
`500.0` and `baoab_cfc` production training resumed from step 37,500.
Over the following ≈1,800 steps (37,501-39,350), evals came back 107.85 /
105.59 / 105.84 -- no new best, still stuck on the §32 plateau -- while
**zero** `watchdog_hard_reload` events fired. Four ordinary `[spike]`
events did fire in that window (`GRAD_SPIKE_THRESHOLD=100`, pre-clip 198 to
449), none of them close to the 500 reload line. This directly
demonstrates that the rare ≥500 events are not what is capping this run's
convergence -- whatever is stalling it at ~105 is happening through the far
more frequent, individually-clipped 100-450 band instead (or through
something unrelated to spikes at all, e.g. the WSD schedule not entering
its decay phase until step 65,000 -- §36.4 flags this as still open).

That reframes the diagnostic goal. §33-§35 were built to explain the rare
catastrophic reloads (question A); the run's actual practical problem --
what is capping convergence (question B) -- may have a different answer,
and this window is evidence that (A) does not answer (B).

### 36.2 Why Phase 1 could not see this band: capture and reload shared one knob

Cell 6's Phase 1 capture condition was gated on the same variable as the
watchdog-hard reload:

```python
# capture (old):
if CAPTURE_SPIKE_BATCH and GRAD_NORM_HARD_TRIGGER is not None
        and _raw_gn_pre > GRAD_NORM_HARD_TRIGGER: ...
# reload:
elif GRAD_NORM_HARD_TRIGGER is not None and _raw_gn > GRAD_NORM_HARD_TRIGGER:
    _reload_best(step + 1)
```

Lowering that one threshold to harvest the frequent moderate band would
also have made the watchdog reload on every one of those events --
discarding progress constantly, which is the opposite of what a
convergence-focused investigation wants. This coupling, not a deliberate
choice, is the only reason Phase 1 had only ever captured rare, possibly
unrepresentative catastrophic events.

### 36.3 The fix: a separate `CAPTURE_SPIKE_THRESHOLD`, plus a batch-replay helper

Three changes to
`colab_fock_cfc_baoab_aniso_gaussian_openwebtext_d384.ipynb`:

1. **New `CAPTURE_SPIKE_THRESHOLD = 200.0`** (Cell 6), independent of
   `GRAD_NORM_HARD_TRIGGER = 500.0`. The capture condition now reads
   `_raw_gn_pre > CAPTURE_SPIKE_THRESHOLD`; the reload condition is
   untouched. Lowering the capture bar can no longer cause an extra
   reload -- the two questions (A) and (B) from §36.1 now have
   independently tunable instrumentation.
2. **`SPIKEBATCH_SNAPSHOT_MAX_KEEP` raised 5 -> 12**, so the run
   accumulates a small *library* of moderate-band captures to replay
   across, rather than only ever holding the single latest one.
3. **New `replay_all_captures()` in Cell 6d**, which replays every
   `_spikebatch.pt` currently on disk and prints one summary table (step,
   pre-clip norm, fidelity gap, leading group, next two groups) instead of
   requiring a separate `replay_spike_batch(step_tag)` call per capture.
   `verbose=True` still prints each individual replay's full per-parameter
   / per-layer / activation-extreme report.

### 36.4 What this buys, and the honest caveat

A consistent leading-group and per-layer signature across many
moderate-band replays is materially stronger evidence than the single
step-37,763 data point in §35 -- and, being harvested every ~1-2K steps
instead of waiting an unpredictable number of hours for a ≥500 event, it
arrives fast enough to actually inform this run rather than only
postmortem it.

**Caveat.** It is not yet established that these moderate spikes (rather
than, say, the stable-phase learning rate not having entered its WSD decay
until step 65,000) are what is capping convergence -- each one is clipped,
so any single update is bounded. `replay_all_captures()`'s output is what
will discriminate this: a coherent, fixable mechanism across the library
(the cascade profile sharpening, a specific gate saturating, etc.) argues
for (B) being spike-driven; captures that look like benign clipped noise
argue for the LR-schedule explanation instead.

### 36.5 Status and next step

**Implemented, not yet run against real data.** The notebook changes
landed in this section; the live run has `CAPTURE_SPIKE_THRESHOLD=200.0`
and `SPIKEBATCH_SNAPSHOT_MAX_KEEP=12` armed going forward, but no capture
under the new, lower threshold has fired yet as of this writing. Next
step: once several moderate-band captures accumulate, run
`replay_all_captures()` and compare the resulting per-event leading-group
column against §35's single depth_code-led data point and the severity
trend noted after §35 (`depth_code` leading the smaller of a set of
spikes, `E`/`P` overtaking it in the largest) -- looking specifically for
whether that trend holds up over a larger n.

## 37. Making Cell 6 Resumable In-Place, and Extracting `grad_clip_utils.py`

### 37.1 The problem: inspecting captures cost a full restart

§36's `replay_all_captures()` needs Cell 6d's functions, which (before this
section) required Cell 6 to have already run at least once, and running
`replay_all_captures()` at all meant *interrupting* Cell 6's training loop.
Two practical problems followed directly from Cell 6 being a bare
top-level `for step in range(resume_step, TOTAL_STEPS):` loop:

1. **Resuming training after an interrupt required a full restart.** Cell 5
   unconditionally rebuilds `model` from scratch and Cell 6's own resume
   block always reloads from whatever checkpoint is on disk, so continuing
   training after an inspection break meant: force a manual checkpoint of
   the in-memory state (`save_checkpoint(step + 1, ..., tag_suffix='_manual')`
   from a scratch cell), restart the Colab runtime, re-run Cells 0-5, add a
   one-off override cell to point the resume logic at the manual
   checkpoint, then re-run Cell 6 -- several minutes of ceremony every time,
   purely to look at a JSON file.
2. **Interrupting reliably caused a `CUDA out of memory` on the very next
   `replay_all_captures()` call.** A `KeyboardInterrupt` landing mid-step
   left that iteration's `x`, `y`, `loss`, and the full forward/backward
   graph alive as ordinary top-level notebook globals -- nothing ever went
   out of scope, since there was no enclosing function frame to tear down.
   On this run that pinned 78+ GiB of *allocated* (not just
   reserved-but-cached) GPU memory until a manual
   `gc.collect()`+`torch.cuda.empty_cache()` cell cleared it -- see the
   error/fix pair earlier in this session's log; every one of the first
   `replay_all_captures()` attempts against a freshly-interrupted run OOM'd
   before that fix.

### 37.2 The fix: `run_training()`, a real function instead of a bare loop

The training loop in `colab_fock_cfc_baoab_aniso_gaussian_openwebtext_d384.ipynb`'s
Cell 6 is now:

```python
def run_training(start_step, total_steps):
    global run_ntp, run_vreg, run_fock_reg, n_run, n_skipped, best_val_ppl
    global _grad_norm_ema, _grad_norm_above_thresh, steps_this_session
    global _last_pg_norms, _last_spike_step, step
    for step in range(start_step, total_steps):
        ...  # unchanged body
    _log_fh[0].close()
    print(f'\nTraining complete. Best PPL: {best_val_ppl:.2f}')
    return total_steps


try:
    next_step = run_training(resume_step, TOTAL_STEPS)
except KeyboardInterrupt:
    print(f'\n[run_training] interrupted at step {step + 1:,}. Call '
          f'run_training({step + 1}, TOTAL_STEPS) to resume in-place ...')
    next_step = step + 1
```

Two things fall out of this shape almost for free:

* **Interrupting no longer OOMs the next replay.** `x`, `y`, `loss`, `xb`,
  `yb`, and every other per-step temporary are now locals of
  `run_training`'s stack frame. When a `KeyboardInterrupt` propagates out
  of that frame to the `try/except` around the call, the frame is torn
  down and those locals go with it -- the bare `except KeyboardInterrupt:`
  (no `as e:`, nothing stored) means CPython drops the exception's
  traceback, and every frame it referenced, as soon as the except block
  finishes, typically well before the very next cell (a
  `replay_all_captures()` call) runs.
* **Resuming after an interrupt is just calling the function again.**
  `run_ntp`, `best_val_ppl`, `_grad_norm_ema`, and friends are declared
  `global` and initialized exactly once, outside and before the function
  (unchanged from before); `run_training()` only mutates them, so a second
  call picks up exactly where the first left off -- watchdog EMA state,
  best-PPL tracking, and the open log file handle all carry over. The new
  contract is:

  ```python
  next_step = run_training(next_step, TOTAL_STEPS)   # re-run this line to resume
  ```

  with no rebuild, no checkpoint reload, and no runtime restart, unless you
  actually want to pick up new *code* (e.g. a fresh `git pull` of this
  notebook) -- `save_manual_checkpoint(next_step)` (a thin wrapper over the
  existing `save_checkpoint(..., tag_suffix='_manual')`) is still there for
  that case.

`evaluate()` got the same treatment for a narrower failure mode: its
`model.eval()`/`model.train()` pair is now a `try/finally`, so an interrupt
landing mid-eval can no longer leave `model` stuck in eval mode for
whatever `run_training()` call resumes after it.

### 37.3 `grad_clip_utils.py`: the first functionality actually pushed to a module

`assign_clip_group`, `per_group_grad_norms`, and `clip_grads_per_group`
moved out of Cell 6 into
`notebooks/conservative_arch/scaleup/grad_clip_utils.py`, parameterized by
a small `GradClipConfig` dataclass (`default_clip`, `overrides`,
`watchdog_exclude_groups`) instead of reading Cell 6's globals directly.
This was the one piece of Cell 6 judged low-risk enough to extract in this
pass: the three functions are pure functions of `(model, config)` with no
hidden state, and moving them fixes a real, pre-existing wart --
Cell 6d's `replay_spike_batch`/`replay_all_captures` silently depended on
Cell 6 having already executed at least once, purely so these three names
existed as globals. `grad_clip_utils.py` ships with
`test_grad_clip_utils.py` (4 CPU-only tests against a toy module, run via
`python test_grad_clip_utils.py` or `pytest`), and Cell 6d now moves to
*before* Cell 6 in the notebook's cell order, since its functions no
longer need Cell 6 to have run first to be *defined* (only to be *called*,
once `_GRAD_CLIP_CFG` and a handful of other Cell-6 config globals exist).

**Deliberately not extracted in this pass:** `evaluate`, the two probes
(`run_causal_probe`, `run_trained_leak_probe`), `save_checkpoint`,
`forward_with_vreg`, and the watchdog/reload logic all stay inline in
Cell 6. They are tightly coupled to `model`/`model_cfg` and several dozen
live-tunable config constants, and there is no way to validate a full
extraction against the actual GPU+OpenWebText pipeline outside Colab --
moving them now would be a much larger, harder-to-verify change for
comparatively little benefit to the actual pain point (§37.1). Candidate
for a later, separate pass once this run's current phase is less
time-critical.

### 37.4 Corollary bug found the same day: `replay_all_captures()` had the identical leak

Confirmed live on this run: interrupting Cell 6 to inspect the run's first
genuine `watchdog-hard` reload (step 41,837, pre-clip 528.0 -- analysis
pending, to be added as a future section once replayed) reproduced the
exact §37.1 OOM symptom (`allocated` stuck near 80 GB even
after the memory-cleanup cell) from *inside* `replay_all_captures()`
itself, not just from interrupting Cell 6's training step. Root cause:
`KeyboardInterrupt` derives from `BaseException`, not `Exception`, so
`replay_all_captures`'s per-capture `except Exception as e:` never caught
it -- an interrupt landing mid-replay propagated uncaught to IPython's
top-level handler, which pins the interrupted replay's forward/backward
graph via `sys.last_traceback` exactly like an interrupted training step
does. Fixed by adding an explicit `except KeyboardInterrupt:` clause that
stops the loop (keeping whatever reports already replayed) instead of
letting the exception escape. `replay_spike_batch`'s own `finally` block
was never affected by this -- `finally` runs for any `BaseException`, so
the live model's weights/grads/hooks were always correctly restored even
when this bug pinned memory; only the *cleanup of the replay's own
temporaries* was missing.

### 37.5 Status

**Implemented, syntax-checked, not yet exercised against a live GPU
session.** Every notebook cell was re-parsed with `ast.parse()` after the
edit (no `SyntaxError`s), `grad_clip_utils.py`'s 4 unit tests pass, and the
full CfC/BAOAB test suite (`test_cfc_baoab.py`) still passes unchanged.
`git diff` on the notebook is limited to Cell 6d's relocation, Cell 6's
edits, and a trailing-newline addition -- no unrelated cells were touched.
Next real test is the first live interrupt-and-resume cycle on the running
L=8 job.
