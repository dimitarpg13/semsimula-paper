# precision_lr_max and baoab_cfc_lowrank Validated on Three Replays

**What this note is.** §42 of the training-instabilities record: step 1 validated: `precision_lr_max` and `baoab_cfc_lowrank` both collapse all three replays, with a hook bug found on the way. It was split on 2026-11-21 from [`CfC_BAOAB_Integrator_and_Mitigations.md`](https://github.com/dimitarpg13/semsimula-paper/blob/5551315/companion_notes/CfC_BAOAB_Integrator_and_Mitigations.md) (written August to October 2026). The text is that document's, unchanged except for its links, and keeps its section numbers, so a reference to "§N" anywhere in the record or in other notes finds its note through [the index](README.md).

## Contents

- [42. Step 1 Validated: precision_lr_max and baoab_cfc_lowrank Both Collapse All Three Replays, a Hook Bug and a Checkpoint-Loading Pitfall Found Along the Way, and the Cap Switched On](#42-step-1-validated-precision_lr_max-and-baoab_cfc_lowrank-both-collapse-all-three-replays-a-hook-bug-and-a-checkpoint-loading-pitfall-found-along-the-way-and-the-cap-switched-on)

---

## 42. Step 1 Validated: `precision_lr_max` and `baoab_cfc_lowrank` Both Collapse All Three Replays, a Hook Bug and a Checkpoint-Loading Pitfall Found Along the Way, and the Cap Switched On

§41.7 item 1's offline ablation ran the same evening against all three
captures (47,116 / 48,507 / 48,917), alongside `replay_integrator_ablation`
(§40) on the same three. The result is unambiguous, and one part of it is
a genuine surprise that reframes §41.4's two-mechanism picture.

### 42.1 A latent bug in both new ablation helpers, found before any real data came out

Both `replay_precision_cap_ablation` and `replay_integrator_ablation`
crashed on first use with `TypeError: expected Variable, but hook returned
'float'`. Their per-layer hook was written as

```python
h_new.register_hook(
    lambda g, li=layer_idx: _layer_grad.setdefault(
        li, float(g.detach().norm())))
```

`dict.setdefault(key, value)` *returns* the resulting value (the new one,
or the existing one if `key` was already set) -- unlike the
no-`return`-statement hook in `replay_spike_batch`'s `_make_layer_hook`
that this was modelled on. PyTorch treats any non-`None` value returned
from a tensor hook as a proposed replacement gradient; a bare `float`
fails that check immediately. Fixed by moving the `setdefault` call into a
named function's body as a statement, so it returns `None`:

```python
def _layer_hook(g, li=layer_idx):
    _layer_grad.setdefault(li, float(g.detach().norm()))
h_new.register_hook(_layer_hook)
```

Semantics (first microbatch per budget/arm wins, per §41.3's caveat) are
unchanged; only the crash is gone. Worth remembering for any future
hook-based instrumentation in this notebook: a tensor hook must return
`None` or an actual replacement `Tensor`, never a bookkeeping value.

### 42.2 The ablation result: both budgets, and the low-rank integrator, collapse all three replays -- including the reverse-channel-led one

| step | recorded pre-clip | replayed (as trained) | `precision_lr_max=1.0` | `precision_lr_max=4.0` | `baoab_cfc_lowrank` L0-2 |
|---|---|---|---|---|---|
| 47,116 | 13,139.5 | 13,033.0 | 3.98 | 4.09 | 2.65 |
| 48,507 | 203.1 | 203.9 | 2.14 | 2.95 | 1.42 |
| 48,917 | 202.0 | 199.6 | 2.79 | 1.90 | 1.34 |

All three "as trained" replays reproduce the recorded pre-clip norm within
about 1 percent, so this is trustworthy, not an artefact of drift between
capture and replay. Raw combined `replay_precision_cap_ablation` +
`replay_integrator_ablation` output for all three steps:
[replay_precision_cap_and_integration_ablations_47116_48507_48917_output.txt](../../notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_L8probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc/replay_precision_cap_and_integration_ablations_47116_48507_48917_output.txt). Both `precision_lr_max` budgets and the low-rank
integrator collapse every one of the three captures down to a pre-clip
norm of about 1-4 -- squarely in quiet-step territory -- regardless of
whether the starting severity was 202 or 13,139.5.

**The surprise is 48,917.** §41.4 classified it as mechanism B -- an
episodic, `reverse_channel_scale`-led cascade, mechanistically distinct
from mechanism A's chronic $V_\theta$ stiffness, on the grounds that
neither `precision_lr_max` nor the low-rank integrator touches
`reverse_channel_scale` or `reverse_ch` at all. Yet both fully suppress
it too. The leading groups collapse in lockstep even though only the
$V_\theta$ low-rank term was touched:

| group | uncapped | `precision_lr_max=1.0` | ratio |
|---|---|---|---|
| `reverse_channel_scale` | 130.7 | 1.94 | 67x |
| `depth_code` | 122.8 | 0.87 | 141x |

**Finding: mechanism B rides on mechanism A, it is not independent of
it.** The most coherent reading is that `reverse_channel_scale`'s cascade
was never an independent trigger -- it was amplifying the same
catastrophically sharp $V_\theta$ force mechanism A produces. Once that
force is bounded, whatever propagates through `reverse_ch` in the early
layers has nothing extreme left to amplify. §41.4's "two coexisting
mechanisms" framing should be read as two *symptoms* of one root cause
(unbounded low-rank curvature), not two independent failure modes that
happen to share a coincidental remediation.

Per-bank exponent minima confirm the cap is doing exactly what §41.6
predicted, at both budgets, e.g. at 47,116: uncapped bank minima range
from -257,630 to -150,076 (numerically annihilated); at
`precision_lr_max=1.0` they range from -146.8 to -51.8; at
`precision_lr_max=4.0`, a looser cap as expected, from -579.3 to -186.6 --
both drastic improvements, with `1.0` giving a firmer margin.

### 42.3 Corroborating evidence: the low-rank integrator's own SVD chokes on the uncapped matrices

`replay_integrator_ablation`'s `baoab_cfc_lowrank` arm triggered a PyTorch
warning during the 47,116 replay: 119 of its batched $B_k$ matrices (5
named plus "114 batches" more) failed to converge under the standard
cuSOLVER driver and fell back to a slower, more accurate method. This is
independent numerical evidence, from a completely different code path
(exact SVD rather than the exponent computation), that the raw
(uncapped) $B_k$ at this snapshot are severely ill-conditioned -- not
just numerically large in the exponent sense, but pathological enough
that a generic dense-SVD routine chokes on them. It is also a fresh, very
concrete illustration of why `baoab_cfc_lowrank` remains
production-infeasible (§34) even though it is the single best-performing
arm in the table above: it is exposed to these same matrices via a more
expensive, less numerically robust code path.

### 42.4 Bracketing $\sigma_{\max}(B_k)^2$ directly: healthy and spike-regime checkpoints look statistically similar under a neutral batch

A companion tool, `bracket_precision_lr_max` (`Cell 6b-3`), implements
§31.2-§31.4's bracketing protocol properly: rather than reading
`sigma_max(B_k)^2` off "whatever's currently loaded," it loads the
healthy `_best.pt` (step 27,000) and each spike bundle's own
`model_state_dict` in turn, on a fixed generic batch, restoring the live
model afterward.

| state | p50 | p90 | p99 | p99.9 | max |
|---|---|---|---|---|---|
| healthy (step 27,000) | 283.2 | 669.9 | 1054.0 | 2329.5 | 6362.1 |
| spike step 47,116 | 310.6 | 714.0 | 1136.1 | 2900.6 | 8065.6 |
| spike step 48,507 | 284.2 | 683.4 | 1058.3 | 2305.3 | 6386.8 |
| spike step 48,917 | 281.6 | 687.4 | 1054.9 | 2426.9 | 6682.8 |

Raw `bracket_precision_lr_max` output:
[bracket_precision_lr_max_47116_48507_48917_output.txt](../../notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_L8probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc/bracket_precision_lr_max_47116_48507_48917_output.txt).

**Finding: on a neutral batch, all four states look almost the same.**
This refines §33's own bracket (which also found the shift between
checkpoints "modest") with a wider, four-way comparison: the ambient
sigma_max(B_k)^2 distribution has apparently been elevated since at
least step 27,000 and stayed roughly flat since, rather than escalating
toward the spikes. §31.4's own rule -- "set the budget above the healthy
row's p90-p99, below the spike rows' p99/max" -- does not resolve cleanly
here, because the healthy and spike percentile bands overlap almost
entirely.

This means there is no tight, tail-only budget available: §41.2's
"chronic, not transient" finding already implied this, and this
four-checkpoint bracket confirms it directly. What actually
distinguishes a quiet forward pass from a catastrophic one is not
`sigma_max(B_k)^2` alone but its *product* with $\lVert h - \mu_k
\rVert^2$ on the specific tokens in play -- which is exactly why the
direct ablation in §42.2 (replaying the real offending batch) is more
decisive evidence than this bracket, and why the chosen budget (§42.6)
follows the ablation, not the bracket.

### 42.5 A checkpoint-loading pitfall: Cell 5 never loads a checkpoint, and an over-eager interrupt on Cell 6 can silently leave weights at random init

While chasing an apparent contradiction (`Cell 6b-2`'s live-model reading
of `sigma_max(B_k)^2` was 400-600x smaller than the same quantity measured
moments later on the same nominal checkpoint via `bracket_precision_lr_max`),
the actual root cause turned out to be a workflow hazard, not a
measurement bug. `Cell 5`'s own `_rebuild_model()` docstring says so
explicitly: *"Weights are still random at this point (the training cell
loads any checkpoint), so this does not discard trained state."* The
checkpoint load is a top-level block inside `Cell 6`, executed
synchronously before `run_training()` is even called:

```python
if resume_ckpt is not None and resume_step < TOTAL_STEPS:
    ckpt_data = torch.load(resume_ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt_data['model_state_dict'], strict=False)
    ...
```

`torch.load` on a large checkpoint over Google Drive's FUSE mount is not
instantaneous. If `Cell 6` is interrupted immediately after being run
(the documented pattern for defining its globals without committing to a
long training run, §37), the interrupt can land *before* this block
finishes, leaving `model` on fresh random-init weights while still
printing as if the session were live. A direct parameter-norm check
caught this cleanly:

```python
_live_w = dict(model.named_parameters())['V_theta.bank.banks.0.B_proj.weight']
_bundle_w = bundle['model_state_dict']['V_theta.bank.banks.0.B_proj.weight']
# broken session: live norm 21.72 vs bundle norm 118.90 (random-init magnitude)
# after waiting for the "Model loaded." print before interrupting:
# live norm 118.9039 vs bundle norm 118.8997 (diff 0.0012 -- correct)
```

`bracket_precision_lr_max`'s own four measurements were never affected --
it reloads each state's weights from disk internally, independent of
whatever `model` held beforehand -- so §42.4's table stands as reported.
Only a bare, no-reload reading of "the live model" was ever at risk.

**Operational rule going forward.** Before calling
`run_training(next_step, TOTAL_STEPS)` to resume real training, always
run a quick parameter-norm check like the one above against the
checkpoint that was supposed to load. Silently resuming from random-init
weights would be a far worse failure than anything else diagnosed in this
note, and the only symptom is a live-model diagnostic reading that looks
implausibly tame -- exactly what happened here before the cause was
understood.

### 42.6 The decision: `PRECISION_LR_MAX = 1.0`, switched on, training resumed from step 47,121

Both evidenced-safe budgets (`1.0` and `4.0`) fully suppress all three
replays; `1.0` was chosen as the live value: it is the more conservative
of the two (firmer exponent-tail margin, §42.2), and `_precision_lr_max`
is a live, hot-swappable Python attribute rather than part of
`state_dict()`, so it is cheap to loosen later and expensive to have
picked too loose a value now (another catastrophic reload). `Cell 0`'s
default was updated from `None` to `1.0` for any future fresh session,
and the live session (already confirmed, via §42.5's check, to be
correctly holding the step-47,116 weights) was patched in place:

```python
for b in model.V_theta.banks:
    b._precision_lr_max = 1.0
next_step = run_training(47121, TOTAL_STEPS)
```

No checkpoint surgery, no rebuild, no restart -- exactly as §41.6
anticipated. `PRECISION_LR_MAX` is not expected to touch anything the
low-rank $V_\theta$ channel does not feed (§42.2's finding says that is,
in practice, everything captured to date, mechanism B included), so this
single change is the resumption strategy in full, pending the monitoring
in §42.7.

### 42.7 Status and next steps

1. **Done.** `replay_precision_cap_ablation` / `replay_integrator_ablation`
   hook bug fixed (§42.1); both validated against all three captures
   (§42.2); `bracket_precision_lr_max` added and run (§42.4); the
   checkpoint-loading pitfall found and worked around (§42.5);
   `PRECISION_LR_MAX = 1.0` switched on in both `Cell 0`'s default and the
   live session; training resumed from step 47,121.
2. **Monitor the resumed run.** Watch `dc_ratio` and `b_proj_sigma_max` in
   the periodic log for a plateau rather than continued growth, watch for
   the absence of further `[watchdog-hard]` reloads near the old
   47,116-49,095 territory, and watch val_ppl against the restored best
   of 100.47 (step 27,000) to confirm the cap is not visibly hampering
   learning. Loosen to `4.0` if it is; no rebuild needed either way.
3. **§41.7 items 2-4 remain open**: the per-layer hook's overwrite-not-accumulate
   behaviour across `grad_accum` microbatches, a `reverse_ch`-side
   stiffness proxy now that §42.2 suggests it is downstream of mechanism
   A rather than an independent trigger, and a surgical row-attribution
   fix for `attribute_spike_rows`.
4. **Fix the two hook-registration call sites' habit of re-registering a
   hook on every microbatch's forward** (`replay_precision_cap_ablation`
   and `replay_integrator_ablation` both do this, inherited from
   `replay_spike_batch`'s pattern) if a future variant needs to
   distinguish per-microbatch contributions rather than "first microbatch
   per arm wins."
