# Revision History — CfC/BAOAB Integrator and Mitigations

Extracted from
[`CfC_BAOAB_Integrator_and_Mitigations.md`](CfC_BAOAB_Integrator_and_Mitigations.md)
on 11 September 2026 (see that note's §52.2 for the reasoning). The trailer had
grown to 440 lines and 26 chained "Previously updated" entries in a single
paragraph — 7.6% of the parent document — and contains no section anchors, so
nothing links into it and moving it breaks no cross-reference.

The **current** revision entry stays in the parent note. Everything below is the
prior history, newest first, exactly as it was written.

---

Previously updated 9 September 2026, late evening (§50: worked out the
temperature dynamics analytically, which changed the mitigation design.
the temperature gradient as the negative covariance of the scaled scores with the utility, taken under the register's own attention distribution
-- a covariance under the register's own attention distribution -- which
has no interior zero: it vanishes only if the scores carry no information
about utility or if attention has collapsed to a point mass, so the
temperature is always driven to one of two degenerate ends. In the
diffuse regime the inverse temperature obeys $\dot v = \eta C v^2$, a
Riccati equation with finite-time blow-up, so the drift is superlinear
rather than exponential. Crucially, **QK-normalisation does not change
these dynamics at all** -- the identical Riccati equation reappears in
$\sigma$ -- and QK-norm applied *on top of* the $1/\tau$ divisor bounds
only the numerator of a ratio whose denominator is still free to fall,
leaving the runaway completely intact; the first implementation did
exactly that and was corrected so the clamped per-register scale
*replaces* `log_tau`. What QK-norm actually buys is that the loop's
endpoint becomes a chosen constant ($\lvert\tilde s\rvert \le
\sigma_{\max}$) instead of an unbounded accident, and that bounding one
scalar becomes *sufficient*, which it is not while
$\lVert q\rVert\lVert k\rVert$ is free. §50.5's table separates the two
multiplicative channels and shows the $\tau$ floor and QK-norm are
complementary, not alternative. Implemented: the §49.8 floor as a
post-`optim.step()` *projection* (`TAU_CREATE_MIN = 4.0`, a no-op at
today's $\tau_{14}=5.21$, and a projection rather than a forward clamp
specifically so the gradient is not zeroed at the boundary), plus
`creation_qk_norm` as an opt-in, non-retrofittable config flag for a
fresh arm. §50.7 states four predictions in advance, of which channel
migration -- register 14's raw $\lvert q\cdot k\rvert$ should start
growing once the floor engages -- is the discriminating one. Previously
updated 9 September 2026, evening (§49: two probes
(`probe_gate_saturation`, `sweep_log_tau_history`) falsified both §48.8's
stated row-level test *and* the clamp-gating hypothesis raised to explain
that first failure, then located the real mechanism one axis over:
`creation_gate_qkv.log_tau`'s gradient is 97.8%/100.0% a *single
register*, number 14, whose learned temperature has drifted to the
coldest in the pool (5.21 vs. a flat pool median of 6.45, rank 30 → 32
over 3,672 steps) so its scaled scores run an order of magnitude above
every other register's -- which by §48.8's own derivative
$\partial L/\partial\log\tau = -\sum\tilde{s}\odot\partial L/\partial\tilde{s}$
is exactly the observed gradient, confirming the score-magnitude
mechanism at register rather than row granularity and vindicating the
proposed QK-norm hardening. The drift is self-reinforcing (both measured
gradients positive, so descent lowers $\tau_{14}$ further). §49.4 records
why the row-level test had no power (its statistic maxed over registers,
so register 14 dominated it in *every* row), §49.7 falsifies token
degeneracy as a necessary condition while showing `max_repeat_run` is
blind to template-level repetition that `unique_token_ratio` catches, and
§49.8 adds a config-only, resume-safe mitigation: split `log_tau` into
its own clip group, since at 71194 it consumed enough of the joint
`creation_gate` budget to cut `W_Q`/`W_K`/`W_V`'s effective update to 60%
of normal. §48.7's "81% single-row" figure is corrected -- it measured an
isolated-row surrogate that reconstructs under 1% of the real gradient,
§41.5's pathology recurring. Previously updated 9 September 2026 (§48:
`replay_spike_batch`/
`attribute_spike_rows` turned out to predate `clip_then_sum` (§45.4),
which made all three new post-§47 replays -- steps 70522, 71194, 71703 --
fail the fidelity check by 77-105%, an `E`/`P` blowup to 1,000+ that
looked at first like another §46-style corruption but was confined
entirely to the replay helpers, not the training run (every other group
matched its captured value exactly); fixed by splicing `clip_then_sum`'s
own per-microbatch-clip-then-accumulate mechanics into
`replay_spike_batch` (three new helpers, `_cts_group_params` /
`_cts_apply_microbatch` / `_cts_splice_back`, a true no-op against
pre-§45.4 bundles), restoring 0.0% fidelity on all three; corrected the
71703 `WATCHDOG_EXCLUDE_GROUPS` blind-spot figure to a 202.3 gap (not the
E/P-contaminated 139.8 first reported), the largest reproduction of that
mechanism yet; and, with `E`/`P` no longer swamping the top-parameter
list, found two new row-level results: 70522's `depth_code` and
`reverse_channel_scale` are two different mechanisms riding together in
one step (`top1_share` 0.34 broad vs. 0.57 single-row), and 71194's
apparent three-way tie is actually two distinct localized rows each
driving a different group, one of them (`creation_gate_qkv.log_tau`) at
an 81% single-row concentration -- the sharpest localization number seen
in this investigation to date; §48.8 then specs, without implementing, a
candidate fix traced from the model code itself: the creation gate never
received the QK-norm hardening the reverse channel got under
`stable=True`, leaving `log_tau`'s gradient proportional to unbounded raw
score magnitudes, with a falsifiable score-magnitude test stated up front
so the mechanism can be ruled out before any forward-function change is
made). Previously updated 8 September 2026,
latest (§47: first production evidence for `clip_then_sum`, ~9,100 steps
and ~21.8h after the clean resume from §46.1 -- `E`/`P` have not led a
single `top[...]` entry in that window, `[spike]` captures dropped to one
non-`E`/`P` event followed by a fully quiet 1,600+-step stretch, and
`val_ppl` broke the long-standing ~98-100 plateau with a new best of
91.88 at step 60,500 (vs. 98.45 at the step-52,500 resume point itself
already a record per §44), corroborated by a clean causal-leak probe at
step 60,000; `bproj_sig` kept drifting at its pre-existing rate, as
expected since this mitigation doesn't touch mechanism A's chronic
stiffness (§41.2) -- one window is a strong first signal, not yet a
confirmed fix, with the WSD decay phase (step 65,000) still to come).
Previously updated 7 September 2026, latest (§45.4:
`clip_then_sum` implemented
and wired live for `E`/`P` (`CLIP_THEN_SUM_GROUPS`, Cell 6, threshold=0.3
chosen directly off §45.3's replay table rather than a magnitude-parity
calculation), immediately rather than deferred, since the session had
only just resumed from the step-52,500 restore (§46) -- the cheapest
possible point to take one more short interrupt/restart. Implementation
clips each microbatch's own `E`/`P` gradient jointly (via the same
`nn.utils.clip_grad_norm_` mechanics `clip_grads_per_group` already uses
post-hoc) before folding it into a running total, replacing the normal
full-`GRAD_ACCUM`-sum accumulation for just these two groups; validated
offline via a standalone numeric check confirming the live per-microbatch
algorithm is bit-identical to the closed-form computation
`replay_clip_ablation` used to produce §45.3's results, so what's running
live is provably the same arithmetic already validated, not just an
assumed equivalent). Previously updated 7 September 2026 (§45.3-45.4:
`replay_clip_ablation` run against both live bundles (52940, 55919) --
clip order (`sum_then_clip` vs. `clip_then_sum`) already disagrees on
`E`/`P`'s applied-update *direction* by ~46-47 degrees at the threshold
currently used in production (`default_clip=1.0`), worsening to ~60
degrees once tightened, and the entire pattern reproduces near-
identically across both bundles despite their opposite layer-profile
shapes (§44) -- strong evidence it's a structural microbatch-outlier
effect, not spike-specific noise). Previously updated 7 September 2026 (§46: a `torch.utils.checkpoint`
`CheckpointError` while replaying bundle 52940 turned out to affect
every replay path in the session, not just the new one, pointing to
either gumbel-softmax routing nondeterminism inside nested checkpoint
regions or session-level CUDA corruption -- the latter confirmed when
`torch.save` itself started hanging and a manual safety checkpoint was
caught mid-write and left unloadable; net loss 4,252 steps, recovered
from the last verified-good checkpoint at step 52,500. Added
`AUTOSAVE_WALLCLOCK_HOURS = 23.5`, a fire-once-per-process wall-clock
safety net (Cell 6) that saves a manual checkpoint based on
`/proc/uptime` rather than `t0`/`time.time()`, specifically so it
survives interrupt-and-resume within a session and isn't blind to
elapsed wall-clock time the way a `run_training()`-local timer would
be). Previously updated 7 September 2026 (§45: a direct `precision_lr_max`
-style clip-threshold ablation turns out to be a non-experiment --
`clip_grad_norm_` returns its pre-clip norm independent of the threshold
passed in, so the "replayed" gradient is identical in every arm by
construction and the applied update is just `min(threshold, raw_norm)`
arithmetic; the real, replay-worthy question is clip *order* --
`sum_then_clip` (live loop: accumulate raw grads across microbatches,
clip once) vs. `clip_then_sum` (clip each microbatch's `E`/`P` grad
before accumulating), since only the latter can bound an outlier row's
influence on the final update at the source. `replay_clip_ablation` (Cell
6d) implements and compares both orders via applied-norm ratio and
cosine-of-direction per threshold; not yet run against the 52940/55919
bundles). Previously updated 7 September 2026 (adds §44: two E/P-led near-trigger
replays -- 441.9 at step 52,940 and 446.3 at step 55,919, both 91-94% of
`hard_trigger=500` once the reverse-channel gap is included -- show
identical named-group leadership (`E`/`P`) but opposite layer-profile
shapes (3.2x smooth cascade vs. 136x localized blowup), confirming
layer-profile shape rather than group identity is the real discriminator
from §38; extends §39's row-concentration anti-correlation into this new
regime (the smoother event is the more row-concentrated one); flags a
worsening $V_\theta$ bank-3 saturation side-observation; not yet acted
on). Previously updated 6 September 2026, latest (§43.5: five consecutive clean
evals through step 50,000 confirm the fix is durable, not a one-off; the
run's periodic causal-leak probe also fired clean at step 50,000
(diff=+0.0186 nats, `[CLEAN]`) on the live post-churn weights; flags two
larger-than-recent `[spike]` events (224.8, 261.8) led by groups outside
the capped low-rank $V_\theta$ channel as worth tracking, not yet acting
on). Previously updated 6 September 2026, later (§43.4: the eval `.backward()`
fix validated live on the first step-47,500 eval to run after the
fresh-session resume -- `mem_alloc` held flat at 2.22GB across all 40
iterations instead of climbing from 47.47GB, `peak_during=55.04GB`
confirming the memory was always a reclaimable peak rather than a leak,
and `evaluate()` completed cleanly for the first time past this step,
reporting val_ppl=115.00 against the restored best of 100.47). Previously
updated 6 September 2026 (adds §43: a fourth repeat of the
step-47,500 eval-time OOM, this time *after* patching `evaluate()` to
call `gc.collect()` every iteration, hit the identical `mem_alloc=47.47GB`
at iteration 0, falsifying the "uncollected `create_graph=True` reference
cycle" theory behind that patch -- the code confirms `create_graph` is
correctly gated `False` in eval. The real cause is the analytic
$V_\theta$ force's ordinary (non-`autograd.grad`, hence
create_graph-flag-immune) tensor graph chaining across all $L$
checkpointed layers, which only gets torn down by an actual
`.backward()` call -- exactly what training does every step and eval
never does. Fix: give `evaluate()` a real, weight-inert
`loss.backward()` + `zero_grad(set_to_none=True)` per iteration so
PyTorch's own checkpoint-teardown machinery runs). Previously updated 5 September 2026 (adds §42: the §41.7 item-1 ablation ran
against all three captures and validates `precision_lr_max` at both
1.0 and 4.0, and `baoab_cfc_lowrank`, against all of them -- including
the reverse-channel-led 48,917, which implies mechanism B rides on
mechanism A rather than being independent of it; fixes a hook-return-value
bug in the two new ablation helpers; adds `bracket_precision_lr_max` and
finds healthy and spike-regime checkpoints look statistically similar
under a neutral batch; finds and works around a checkpoint-loading
pitfall where `Cell 5` never loads a checkpoint and an over-eager
interrupt on `Cell 6` can silently leave weights at random init; switches
`PRECISION_LR_MAX` on, at 1.0, in both `Cell 0`'s default and the live
session, and resumes training from step 47,121). Previously updated 31
August 2026 (§41.7 item 1 implemented: `Cell 6d`'s
`replay_precision_cap_ablation(step_tag, budgets=(1.0, 4.0, None))`,
mirroring §40's `replay_integrator_ablation` but swapping
`bank._precision_lr_max` across arms; not yet run against a live GPU
session). Previously updated 31 August 2026 (adds §41: three new replays -- steps 47,116,
48,507, 48,917 -- show the low-rank channel is chronically, not
transiently, dominant across every capture, refining §33.1's own caveat;
falsify §38.3's smooth-cascade self-limiting hypothesis outright (47,116
crosses the hard trigger 26x over while smooth-shaped); expose a `dc_ratio`
blind spot for a reverse-channel-led second mechanism at 48,917; flag
`attribute_spike_rows` as unreliable on this failure mode; and motivate
turning on the already-implemented `precision_lr_max` cap, with a starting
value and an offline ablation plan). Previously updated 31 August 2026
(adds §40: does `baoab_cfc_lowrank` address the
localized mode; reconciles §33's negative verdict, which was measured against
a structurally different, earlier crisis signature, with the new
layer-0-2-cliff taxonomy, and proposes a targeted `lowrank_layers={0,1,2}`
offline ablation test against the existing spikebatch bundles rather than a
full-cost live retraining trial). Previously updated 31 August 2026 (adds §39: two forward-pass hypotheses for
the localized mode's mechanism -- a token-minority driving `depth_code`'s
direction, and denser `V_theta` well occupancy -- were each designed
before being tested and both came back falsified against all four
Phase-1/2-instrumented events with a recorded layer profile; per-row
attribution shows the localized events are the *flattest* across rows,
the opposite of the conjecture, with a clean monotonic anti-correlation
between layer-0-2 severity and row concentration; exponent occupancy
shows no separation at all, and additionally reveals over 99.9% of
token-well pairs are numerically dead in every capture regardless of
mode. Revises the working picture: the localized mode is layer-localized
but batch-wide, not batch-specific, reopening the `sigma_max(B_k)`
precision-matrix stiffness line in sharper, trajectory-based form. Next
step: add a weight-space stiffness proxy to Cell 6's periodic logging,
same pattern as `dc_ratio`.)

Previously updated 30 August 2026 (adds §38.7: the §38.6 deep-dive
against the two localized captures, and a matching control-group replay
of two smooth-cascade captures, found that the layer-resolved activation
extremes and `set_fock_capture` gate-health table are statistically
indistinguishable between modes (a property of current model weights, not
batch/outcome) and token degeneracy is ruled out entirely -- but a number
already present in the original 7-event Phase-0 data, `dc_ratio`
(`depth_code`'s group norm over the next-largest group's), splits cleanly:
<1.8 for every smooth-cascade event, >2.2 for both localized events.
Corrects §38.6: `V_theta` banks don't share the per-layer-overwrite bug,
they're called once per forward pass on the depth-stacked `xi` tensor, now
reported at `layer=-1`. `dc_ratio` logging added to Cell 6's periodic log
line so a future mining pass can check whether it leads a hard-trigger or
only coincides with one).
Previously updated the same day (adds §38.6: rather than waiting on
new localized captures, Cell 6d's replay was deepened to re-interrogate
the two already-on-disk localized bundles (steps 39,983/41,837) for free
-- fixes a real bug where `activation_extremes` silently only ever
reflected the LAST layer (7) for `creation_gate_qkv`/`reverse_ch`/`V_theta`
banks, since they're shared modules called once per layer and the old hook
overwrote its own dict key each time; wires in the model's own
`set_fock_capture` per-layer register/gate health buffer; and adds
`inspect_spike_tokens()` to decode and rank the offending microbatch's
tokens by degeneracy, independent of the gradient side entirely).
Previously updated the same day (adds §38: with 7 spike replays now
on hand -- including this run's first genuine `watchdog-hard` reload,
step 41,837 at 528.0 pre-clip -- the per-layer $h$-gradient profile splits
cleanly into two failure modes, a smooth 8-layer cascade (5 of 7 events,
layer0/layer3 ratio 2.6-6.3x) and a localized layer-0-2 blowup with a sharp
cliff by layer 3 (2 of 7, ratio 50x and 177x); leadership by parameter
group does not discriminate the two modes (`depth_code` leads one of each),
but only the localized mode has crossed 500 so far, giving a working
(n=2, not yet confirmed) hypothesis that it -- not the self-limiting smooth
cascade -- is what actually forces this run's rare reloads. Previously
updated the same day, earlier (adds §37: Cell 6's training loop is
now a real function, `run_training(start_step, total_steps)`, instead of a
bare top-level `for` loop -- interrupting it to inspect spike captures no
longer OOMs the next `replay_all_captures()` call (per-step locals now die
with the function's stack frame instead of lingering as notebook globals)
and no longer requires a checkpoint-save/restart/override dance to resume
(`run_training(next_step, TOTAL_STEPS)` just picks up where it left off,
since its accumulator/watchdog state is `global` and initialized once
outside the function). `evaluate()` got a `try/finally` so an interrupt
mid-eval can't leave `model` stuck in eval mode. `assign_clip_group` /
`per_group_grad_norms` / `clip_grads_per_group` moved out to a new
`grad_clip_utils.py` module (with its own unit tests), the first
Cell-6 functionality actually pushed to a module rather than left inline;
Cell 6d moved before Cell 6 in the notebook's cell order now that its
functions no longer need Cell 6 to have run first just to be defined.
Same day, confirmed live on the run: `replay_all_captures()` had the
identical leak for the identical reason -- `except Exception` doesn't
catch `KeyboardInterrupt` (a `BaseException`), so an interrupt landing
mid-replay pinned ~80GB via `sys.last_traceback` just like an interrupted
training step did; fixed with an explicit `except KeyboardInterrupt:`
that stops the loop instead of letting it escape uncaught).
Previously updated 30 August 2026, later (adds §36: after §35's fix,
`GRAD_NORM_HARD_TRIGGER` was restored to 500.0 and production resumed --
over the next ≈1,800 steps (37,501-39,350) zero hard reloads fired, yet
the run stayed stuck on the same ~105 plateau, with four ordinary
100-450 `[spike]` events in that window. Since Phase 1 capture had been
gated on the same `GRAD_NORM_HARD_TRIGGER` as the reload, it could never
harvest that far more frequent moderate band. Fix: a new, independent
`CAPTURE_SPIKE_THRESHOLD=200.0` that gates capture without touching the
reload condition, `SPIKEBATCH_SNAPSHOT_MAX_KEEP` raised 5->12, and a new
`replay_all_captures()` helper in Cell 6d that replays every captured
`_spikebatch.pt` and prints one summary attribution table instead of one
call per capture. Reframes the diagnostic goal from "explain the rare
catastrophic reload" to "explain the plateau," with the caveat that the
moderate band's causal role vs. the not-yet-decayed WSD learning-rate
schedule is still open, to be discriminated once captures accumulate).
Previously updated 30 August 2026 (adds §35: fixed a `TypeError` in
`replay_spike_batch` -- `torch.load`'s `map_location=DEVICE` was remapping
`rng_state_cpu`/`rng_state_cuda` off the CPU, which `torch.set_rng_state()`
rejects; fixed via `map_location='cpu'`. With the fix, the Stage-1 smoke
test's first capture (step 37,763, pre-clip 160.4) replayed with a
0.0012% fidelity gap -- Phase 1/2 fully validated end to end. The
per-layer $h$-gradient profile came back monotone across all 8 layers
(≈37x growth from layer 7 to layer 0, no isolated bump), matching §33.4's
cascade-amplification signature rather than a single-layer cause;
`depth_code` led the per-parameter attribution narrowly ahead of `E`/`P`.
Flagged as a first look, not a closed case, since this event sits in the
"<200" band rather than the `E`/`P`-dominated severe regime; next step is
the same replay against a genuine ≥500 `watchdog_hard_reload`).
Previously updated 30 August 2026 (adds §34: `baoab_cfc_lowrank` was run
end-to-end at the live L=8/$d$=384/OWT scale from the §32 checkpoint,
surfacing and fixing three bugs — a `CheckpointError` from
`torch.svd_lowrank`'s non-deterministic branching under gradient
checkpointing (fixed via the new branch-free `_randomised_svd_det`), NaN
modes from a degenerate GPU SVD poisoning the gradient (fixed via
`torch.where` masking + `nan_to_num`), and an infinite backward in
`harmonic_terms_lowrank`'s `sqrt(g)` at $g=0$ (fixed via the analytic
$\sqrt w\exp(-e/4)$ form) — before measuring ≈120 s/step at full width and
≈50 s/step even restricted to 2 layers (vs. ≈10-15 s/step for
`baoab_cfc`), which would take ≈36 days to finish the run. Verdict: correct
and stable but not production-feasible at this scale, and independently
aimed at a quantity §33 already showed to be a weak driver; the code is
retained for future/smaller-scale use but production reverts to
`baoab_cfc`, resuming the §32 trajectory. `cfc_baoab.py`'s module
docstring and `model_parf_multixi.PARFConfig.integrator`'s docstring now
carry the same status note). Previously updated 28 August 2026, latest
(§33.3 Phase 0 done: mined the L=8
run's `training_log.jsonl` (964 lines, steps 50-40,900, downloaded from
Drive) for `grad_spike`/`watchdog_hard_reload` events. `depth_code` leads
76% of smaller (pre-clip < 200) spikes but 0% of the two severe hard
triggers, where `E`/`P` (token/positional embedding, tied via
$h_0 = E(x) + P$) lead 100% of the time, agreeing to 4 significant figures
at the larger trigger -- read together as one cascade-amplification
mechanism crossing a severity threshold, not two independent culprits.
§33.3/§33.4/§33.5 updated with the finding, the E/P mechanism (via
`model_parf.py`), and the resulting per-layer-growth prediction for Phase
2 to test against the run's next hard trigger). Previously updated 28
August 2026, later night (implements §33.3 Phases 1 and
2 in `colab_fock_cfc_baoab_aniso_gaussian_openwebtext_d384.ipynb`: `Cell 6`
gains `CAPTURE_SPIKE_BATCH`/`SPIKEBATCH_SNAPSHOT_MAX_KEEP` plus a
pre-`optim.step()` capture of RNG state, exact microbatches, and (only on a
`GRAD_NORM_HARD_TRIGGER`) a CPU clone of the pre-step weights, into a new
`_spikebatch.pt` sidecar; new `Cell 6d` adds `replay_spike_batch(step_tag)`,
a non-polluting isolated replay with per-parameter grad norms, a per-layer
`h`-tensor hook composed with the existing depth-routing patch, best-effort
forward-activation-extreme hooks on the creation gate/reverse
channel/V_theta/destruction gates, and a built-in fidelity check against
the originally recorded `pre_clip_grad_norm`. Scoped to hard triggers only
— the EMA path has no single offending batch. §33.3/§33.5 text and the
Phase-0-3 mermaid diagram updated to mark Phases 1/2 implemented; no
`_spikebatch.pt` exists yet for the two already-passed hard triggers
(32,139/34,091), so the harness awaits the run's next one). Previously
updated 28 August 2026, night (adds §31.7: ran the L=8 bracket
through §31.4's own budget-selection recipe and it self-terminates at
step 1 — no valid `precision_lr_max` window exists, because the
"runaway" tail is already present in the best/healthy checkpoint
(`max` 6,364.81, within 1.0% of spike_34091's 6,427.16 and only 14.5%
below spike_32139's 7,285.88). Recommendation: leave
`PRECISION_LR_MAX = None` on the live run; a non-targeted diagnostic
value (~2,500) is noted for §31.5's optional falsification A/B only.
§31.6's status line updated to match). Previously updated 28 August
2026, evening (§33.1 revised with the second bracket point: the
step-32,139 prereload snapshot came back +3% to +24% above healthy —
more elevated than step-34,091, despite firing first — so the bracket is
a modest, non-monotonic elevation rather than a perfectly flat null
result. The magnitude ($\lesssim 1.1\times$ in $\omega$) still rules out
$B_k$ as the primary driver of the $\gt 100\times$ grad-norm spikes;
§33.1/§33.2 text, table, and figure updated accordingly). Previously
updated 28 August 2026 (adds §33: the L=8 `precision_lr_max` bracket pair
was measured — $\sigma_{\max}(B_k)^2$ was within +8% at every percentile
between the healthy step-27,000 checkpoint and the step-34,091 spike-regime
prereload snapshot — confirming §31.4 step 1's escape hatch that $B_k$
growth is not the primary driver of these bursts; `precision_lr_max` is not
the primary lever, and §33 lays out the cheapest-first Phase 0–3 root-cause
workflow for the non-$V_\theta$ spike groups plus a pointer to the new SCAF
`GradientSpikeProbe` design doc. Also fixes `lowrank_modes` to decompose $G$
by SVD instead of eigh on the Gram $G^\top G$, removing an
ill-conditioned-Gram convergence failure surfaced while running the
diagnostic). Previously updated 27
August 2026, evening (adds §32: the L=8 probe's
extended 27,000–39,867 trajectory is a noisy plateau — no new best,
spike rate not decaying — and the resulting decision to switch to
`baoab_cfc_lowrank` mid-run on single-GPU compute rather than complete
the plain-`baoab_cfc` arm to 100,000). Earlier the same day: implements
§31.3's SCAF `sigma_lr_*` percentiles and adds the equivalent
`sigma_lr_report` notebook diagnostic; broadens §31's scope to the L=8
probe after its own escalating 32,139/34,091 hard-watchdog bursts, with a
bracket checkpoint pair already on hand. Previously updated 24 August
2026 (adds §24.4's depth-probe update and §31's SCAF audit plan). Split
out of the parent note (former §24-§28,
content unchanged) for maintainability. Sections 29-30 record the principled
directions beyond the $B_k$ clamp and the concrete low-rank exponential
substep. This revision marks §29.2 (`integrator='baoab_cfc_lowrank'`) and
§29.3 (`precision_lr_max`) as **implemented and unit-tested** (the pre-existing
`baoab_cfc` path is byte-for-byte unchanged; both are opt-in via notebook
config) and corrects two errors uncovered during implementation:
(1) the exact rotation must act on the **PSD** low-rank $L = G G^\top$ with the
diagonal precision $D_a$ split off, not on the **indefinite** off-diagonal
$G G^\top - \mathrm{diag}$ (§30.1 proves the off-diagonal is indefinite via a
zero-trace argument and shows why PSD-ness of the sum $H$ does not survive an
indefinite split); and (2) the composed scheme is an **impulse / RESPA**
multiple-time-step method, which is stable at any stiffness *between*
resonances $\omega_L \Delta t \approx k\pi$ but is **not** unconditionally
A-stable — replacing the earlier overstated "no wall at all" claim. §30 is now
the as-built description (frozen/detached mode geometry, mode curvature
$\kappa_j = \lambda_j$ of $L$ rather than a Rayleigh quotient of $H$, and the
soft $D_a$ + nonlinear residual demoted to the explicit kick).
