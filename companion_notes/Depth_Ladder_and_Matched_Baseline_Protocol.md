# Depth ladder and the matched baseline — protocol and running ledger

> **Scope.** The from-scratch programme: `L` swept at matched integration
> time on `baoab_cfc_lowrank`, plus the token-matched GPT-2 rerun that every
> comparison depends on. Distinct from
> [`Joint_Vtheta_QKNorm_Run_Diagnostic_Checklist.md`](Joint_Vtheta_QKNorm_Run_Diagnostic_Checklist.md),
> which covers the **warm-started L=8** arm and shares neither checkpoint,
> depth, integrator nor notebook with anything here.
>
> **Notebook:** [`colab_fock_cfc_baoab_lowrank_depth_ladder_openwebtext_d384.ipynb`](../notebooks/conservative_arch/scaleup/colab_fock_cfc_baoab_lowrank_depth_ladder_openwebtext_d384.ipynb)
> **Design:** [`Composing_Single_Layer_Inferences_Flow_or_Maps.md`](Composing_Single_Layer_Inferences_Flow_or_Maps.md)
> **Cost model:** [`Fock_Inference_Productionization_Plan.md`](Fock_Inference_Productionization_Plan.md)
> **Conservativity arms:** [`Measuring_the_Price_of_Conservativity.md`](Measuring_the_Price_of_Conservativity.md)

---

## 1. Why this programme exists

Two separate failures of attribution made it necessary.

**The warm-start arms could only screen.** Every relaxation arm grafted onto
the step-28,500 checkpoint pays a consolidation cost it cannot be separated
from — §7.3 of the conservativity note derives this and measures it at about
1 PPL. Those arms answer "does adding X during the final anneal help", never
"is X worth anything here".

**The published baseline was not token-matched.** §9 of the checklist claimed
equal steps meant equal tokens. It did not: GPT-2 ran its first 14,000 steps
at batch 8, so it saw **360.4M tokens** against Fock's **532.5M**. Every
budget-fraction figure downstream inherited that.

Running from scratch fixes the first. Re-running GPT-2 at a constant batch
fixes the second.

---

## 2. Fixed configuration

Held identical across every ladder point, so `L` and the added mechanism are
the only variables.

| | value |
| --- | --- |
| d / block / vocab | 384 / 512 / 50257 |
| V&#95;theta | joint anisotropic Gaussian, K=8, `ANISO_RANK=4` |
| xi channels | 5, `XI_CONTENT_ROUTE=True` (Alternative E on) |
| V&#95;phi | sparse top-k, `TOP_K=16` |
| integrator | `baoab_cfc_lowrank`, `lowrank_driver='gram'` |
| total integration time | `T = L * dt = 8`, held fixed |
| schedule | WSD, warmup 5%, stable 60%, decay to a 1.50e-05 floor |
| steps / tokens | 32,500 / **532.5M** at 16,384 tok/step |
| lr | 3.0e-04 |
| exchange field, when present | 8 heads x d&#95;k 48, lambda pinned at 1.0, live readout |

**`lowrank_driver='gram'` is mandatory and is not the library default.**
`MultiXiPARFConfig.lowrank_driver` defaults to `'svd'`, measured at 3,368.9 ms
per call against gram's 40.1 ms. Cell 5 asserts the built config carries it.

---

## 3. The queue

| # | run | cost | what it settles | state |
| --- | --- | ---: | --- | --- |
| 1 | **matched GPT-2**, batch 32 throughout | 2.7h | removes the 1.48x token confound from every comparison in the programme | **DONE**, §5.4 |
| 2 | L=2, `'attention'` **@3e-04** | 14.5h | — | **DONE**, §5.1 — but at the *old* LR; see run 9 |
| 3 | L=2, `'none'` **@3e-04, then @1.2e-03** | 14.5h each | what the exchange field contributes at fixed depth | **DONE**, §5.2 (3e-04: 75.09) and §5.4a (1.2e-03: **66.98**) |
| 4 | **L=4**, matched | ~27h | hops, versus "the L=8 arm was handicapped" | queued |
| 5 | L=2, `'attention_potential'` **@1.2e-03** | 13.9h | the price of conservativity, from scratch, parameter-matched | **DONE**, §5.8: **80.90** — conservativity costs +27.4%, and the conservative field is *worse than no field at all* |
| 6 | L=2, `'nonconservative'`, lambda pinned **@1.2e-03** | ~14h | an unconstrained pointwise map, the one function class Fock has nowhere | queued — **Cell 0 could not launch it until 2026-09-25**: the variant-tag dict had no `'nonconservative'` entry and raised `KeyError`; fixed, tag is `noncons` |
| 7 | L=1, `'none'` | ~7h | one hop; the structural floor where the Jacobi metric ceases to exist — **but see §6.1: it also silently disables the Fock registers** | **DONE**, §5.5 |
| 9 | **L=2, `'attention'` @1.2e-03 — a RE-RUN** | 13.6h | **repairs the ladder's central comparison.** Run 2 measured `'attention'` at 3e-04; `'none'` has since been retuned to 1.2e-03. The two arms are currently at different learning rates, which is what §5.2's SUSPENDED banner records. Until this runs, "what the exchange field contributes" has no answer at the ladder LR. Pre-registered **63, band 59–68** | **DONE**, §5.7: **63.51** (band hit, error +0.51) |
| 8 | **L=2, `'none'`, `REVERSE_CHANNEL = False`** — *not a ladder point; an architecture control* | 14.5h | **the conservative-only baseline**: what PARFLM reaches with the Fock mechanism off and every parameter free to compensate. The three existing numbers (+275% ablation A, 3.91x E5 at λ=0, +1226% ablation B) are all inference-time removals from a trained model and are upper bounds. Pre-registered **105, band 85–140**; design and reasoning in [`Forced_Lagrangian_Reformulation.md`](Forced_Lagrangian_Reformulation.md) §3.5 | **DONE**, §5.6: **87.93** |

| 10 | **L=2, multi-ξ SPLM** — V_θ(ξ, h) + ξ routing, **no V_φ**, no Fock | ~12h | the PARF rung: **V_φ has never been removed from a trained model**, in a family named PARFLM. Pairs with run 11 as a 2×2 (§3.2) | queued — **unblocked 2026-10-01**: `PAIR_POTENTIAL = 'none'` (tag `nophi`; −137,803 params, 0.18%). Under the live convention: run with `XI_GRAD_PATH = 'live'` (gradient-starvation note, live ladder) |
| 11 | **L=2, Fock-SPLM** — as run 10 but with the Fock mechanism on | ~13h | replicates the Fock price on a V_φ-free base and supplies the 2×2's interaction term (§3.2) | queued — **unblocked 2026-10-01**, as run 10 with `REVERSE_CHANNEL = True` |

Run 1 first regardless of ordering elsewhere: it is 2.7 hours and it makes
every other number defensible.

### 3.2 Runs 10 and 11 — the V_φ × Fock factorial

Runs 10 and 11 are queued as a **pair**, because separately they are two
more points and together they are a 2×2 factorial with two cells already
measured:

| | with V_φ (PARF) | without V_φ |
| --- | ---: | ---: |
| **with Fock** | Fock-PARFLM (run 3) **66.98** | Fock-SPLM (run 11) — ? |
| **without Fock** | PARFLM (run 8) **87.93** | multi-ξ SPLM (run 10) — ? |

Three things follow that no single arm gives:

1. **A replication of the Fock price on a different base.** +31.3% rests on
   one pair. Run 11 against run 10 measures the same quantity
   independently. Agreement makes the number robust; disagreement is the
   finding.
2. **V_φ's worth, measured twice** — once with the register mechanism
   present, once without.
3. **The interaction term.** Are V_φ and the register bank *substitutes*
   (each covering context the other would) or *complements*?

**Pre-registered**, from E1's measurement that V_φ moves the step by
−0.0002 with the Fock mechanism and −0.0015 without it (master doc §4.7,
§4.9): **near-zero interaction, and a near-flat right-hand column.**
Concretely, run 10 within 5% of 87.93 and run 11 within 5% of 66.98. If
either misses by much, V_φ matters through *training dynamics* in a way
the per-step deflection cannot see — which would be a more interesting
result than confirmation.

> **Priority revised 2026-09-27 (§3.3).** The paragraph below ranks these
> runs third, reasoning from E1's small Vφ deflection to a predicted
> near-null. That inference is unsound — §5.6 establishes that a per-step
> or ablation measure does not predict a trained-without outcome — and the
> runs are promoted to follow run 4. The screening advice still holds as a
> cheap sanity check, but it is no longer a gate on whether to run them.

**Screen before spending 29 hours.** Zero `f_phi` at inference on the run 3
and run 8 checkpoints and read the PPL. F5 established that an
inference-time ablation overstates by roughly 3×, so it is a poor price
and a serviceable **upper bound**: if the ablation costs ~2%, the trained
arms are predictable and the factorial is confirmatory; if it costs ~40%,
run it.

**Blocker.** There is no clean switch. `pair_potential='xi_attention'` sets
`V_phi = None` but *replaces* it with the routed attention rather than
removing it, and pinning `raw_v_phi_scale` leaves the parameters in the
optimiser. Both runs need a `pair_potential='none'` branch in
`model_parf_multixi.py` — small, but it is shared model code touching every
other arm, so it needs tests. The arms will not be parameter-matched (V_φ's
four heads plus the score head leave), and their cards must say so.

**Not a rung: vanilla SPLM.** A truly pointwise V_θ(h) is not a config
flag — the anisotropic-Gaussian V_θ computes its well centres, amplitudes,
widths and low-rank factors *from* ξ, so removing ξ means a different
potential family. It would be a floor rather than a ladder step, and the
paper already records vanilla SPLM's ceiling from earlier work.

### 3.3 Completion plan — **agreed 2026-09-27**

Four runs remain. Two of them address **single-point-of-failure headline
claims**; two add rungs. The sequencing below is by *risk*, not value,
because the two priorities cost the same.

**Why these two first.** Each of the book's two strongest numbers rests on
a single measurement:

| headline claim | rests on | what removes the single point of failure |
| --- | --- | --- |
| `geo+LN` = **0.0003** — geodesics are exact on the conservative arm | **one** dynamically clean layer (at L=2, layer 0 is the embedding-to-sphere projection at 5.4–9.1 × \|h_in\|) | **run 4, L=4** — three clean layers, so a point becomes a trend |
| the Fock mechanism costs **+31.3%** | **one** pair (`'none'` vs no-reverse-channel) | **runs 11 vs 10** — the same quantity measured independently on a Vφ-free base |

**A correction to §3.2's earlier priority note.** Runs 10/11 were ranked
third there, on the grounds that E1 measures Vφ at −0.0002 and −0.0015 and
so predicts a near-null. **That inference is unsound and this document
already says why:** §5.6 establishes that a per-step or ablation measure
does not predict a trained-without outcome — the reverse channel's
ablation said 3.75x where the trained arm said 1.31x. A small deflection
is no more predictive than a large ablation was. Runs 10/11 are promoted.

**Sequencing: start the zero-risk run, build the risky code while it
trains.** Run 4 needs one Cell 0 knob. Runs 10/11 need a
`pair_potential='none'` branch in shared model code that every other arm
executes, so it must be written and tested before it lands.

---

#### Checklist

**A. Run 4 — L=4, `'none'`, @1.2e-03 — start immediately**

- [ ] Cell 0: `LADDER_L = 4`. Leave `LADDER_T = 8.0` (so `LADDER_DT`
      derives to 2.0), `LADDER_MECHANISM = 'none'`, `LADDER_LR = 1.2e-3`,
      `REVERSE_CHANNEL = True`, `RELAX_GATE = 'scalar'`,
      `RELAX_INIT_SCALE = 0.02`, `PROBE_MAX_STEPS = None`
- [ ] Confirm the tag carries `L4probe` and Cell 2 trains from scratch
- [x] Pre-register the settled PPL **before launch**, naming the quantity
      that could turn it — **done 2026-09-27**, band below. §5.3's "no strong
      prior; this is the point of running it" is superseded now that four
      arms are measured
- [ ] Record the clip-hit rate: §6 requires it from L=4 onward
- [ ] Run ~27h
- [ ] Probes afterwards: 6b-7, **6b-9** (three clean layers for the first
      time), **6b-10** (E3 was structurally invalid at L=2 and becomes
      measurable), 6b-12
- [ ] File to `results/`, write §5.9, score the forecast, update the HF
      card and `ladder.json`

#### Pre-registration for run 4 — recorded **2026-09-27, before launch**

Drafted by Claude; overrule before starting if you read it differently.

**Point 59, band 55–65 settled.**

Built from the one measured transfer available, and discounted for why that
transfer overstates: L=1 → L=2 at this learning rate gave 87.09 → 66.98, a
23.1% improvement. **That number is contaminated and must not be applied
again as-is.** At L=1 the register bank is read but never updated (§6.1), so
part of the L=1 → L=2 gain was a mechanism switching on, not depth. From L=2
to L=4 the mechanism is live at both ends, so the depth-only component is
what remains. Halving the contaminated gain gives 66.98 × 0.88 ≈ 59; the band
spans 0.82 (55) to 0.97 (65).

**The quantity that could turn it: the timestep, not the depth.** `LADDER_T`
is held at 8, so L=4 means dt = 2 and ω·Δt halves. Gate 3 has already shown
the stack is fitted to its own dt — re-running the *trained* L=2 model at
N=4, dt=2 gave 236.62 against 68.65. Training at dt=2 is not that experiment,
but it is the same warning: this rung changes the operating point as well as
the depth, and the two cannot be separated within this run.

**What would show it turning.** Clip-hit rate against L=2's 0.0%, and
`bproj_sig`. If clipping appears where L=2 had none, the smaller timestep is
binding and the settled value should be read as an operating-point result,
not a depth result. §6 requires the clip-hit rate from this rung onward for
exactly this reason.

**If it lands above 66.98** — worse than L=2 — the honest reading is that
matched-T depth scaling has turned, and the next rung should hold dt fixed
and let T grow instead, which is a different ladder.

**Launched 2026-09-27.** Clean start: schedule reads warmup 0→1,625, stable
1,625→21,125, decay 21,125→32,500, matching every other arm (stable_end =
warmup + 0.60 × total). Watchdog, spike capture, per-group clips and the
resonance monitor all armed. `[tau-floor] skipped` is expected under
`CREATION_QK_NORM = True`, which does not register `log_tau`.

**It will need one resume.** 3.12 s/step against L=2's 1.29, i.e. **2.41×,
not the 2× that doubling the layer count suggests** — the extra 20% is the
per-step overhead that does not halve with dt. Projected wall clock **28.2 h**
against the 23.5 h autosave, so the autosave fires near **step 27,100** with
about **4.7 h** left to run. Scheduled checkpoints at 7,500 / 15,000 / 22,500
/ 30,000 are the coarser safety net underneath it. To resume: re-run Cells
0→5, then Cell 6; Cell 2 picks up the tagged snapshot.

**Autosave mechanics, verified in the committed notebook.**
`AUTOSAVE_WALLCLOCK_HOURS = 23.5` is checked every step and fires **once per
process**, calling `save_manual_checkpoint`, which writes
`{CKPT_PREFIX}_step{N}_manual.pt` with model *and* optimizer state. Cell 2's
resume regex is `_step(\d+)_(?:manual|prereload|probe_stop)\.pt`, so that file
is picked up automatically and Cell 2 prints that it is resuming from it
whenever its step is ahead of `_best.pt`. The save is wrapped in try/except:
a failure warns and training continues rather than dying.

Two details that matter operationally. It reads **VM uptime**, not training
time, so it fires earlier in the run by however long the VM was alive before
Cell 6 started — deliberate, since the thing it guards against is Colab's
~24 h VM lifetime. And the once-per-process flag is reset by re-running
Cell 6; calling `run_training(...)` again by hand in a still-live session
leaves it armed-and-spent.

**The exposure is small even without it.** `_best.pt` is written on every
improving eval, i.e. potentially every 500 steps, and at L=2 the best landed
at step 31,500 of 32,500, so improvements continued almost to the end. The
periodic saves at 7,500 / 15,000 / 22,500 / 30,000 are the floor. Worst case
is a stalled-PPL stretch ending at a hard cutoff, which would cost back to
the last periodic save; the realistic case is under 500 steps.

Memory is not the constraint: peak 30.9 GB of 85.1 GB at batch 16. The batch
is held at 16 × 2 for token-budget parity with the other arms, not because
the device is full.

**Reference curve for reading L=4 while it runs.** L=2 at the same LR and
budget, so the two are directly comparable step for step:

| step | L=2 ppl | local movement per 500-step eval |
| ---: | ---: | ---: |
| 500 | 478.33 | −46.1% |
| 2,000 | 167.87 | −11.8% |
| 5,000 | 118.05 | −5.2% |
| 10,000 | 97.14 | −3.8% |
| 15,000 | 86.57 | −0.5% |
| 21,000 | 81.09 | +2.4% (decay begins 21,125) |
| 25,000 | 75.19 | −0.0% |
| 30,000 | 69.79 | −2.7% |
| 32,500 | 67.63 | — |

**Running log, L=4 against the L=2 curve.**

| step | L=2 | L=4 | gap |
| ---: | ---: | ---: | ---: |
| 500 | 478.33 | 495.47 | +3.6% |
| 1,000 | 257.74 | 260.64 | +1.1% |
| **1,500** | 197.04 | 195.16 | **−1.0%** (crossover) |
| 2,500 | 148.02 | 145.57 | −1.7% |
| 5,000 | 118.05 | 115.63 | −2.1% (widest lead) |
| 7,000 | 104.21 | 103.06 | −1.1% |
| 8,000 | 100.77 | 100.52 | −0.2% |

**L=4 is not trailing; it has been ahead since step 1,500.** The early deficit
was the warmup artefact it looked like. It crossed over at 1,500, led by 1–2%
through step 7,000, and is now level. Do not read the lead as a result either:
decay does not begin until 21,125, and the pre-registered band is scored on
the settled value alone.

**`dc_ratio` is NOT a clip ratio — correction to a first reading.** It is
`||grad(depth_code)|| / max(||grad|| of every OTHER group)`, i.e. how far
`depth_code` dominates the gradient landscape, and it says nothing directly
about clipping. The clip ratio is the printed `top[...]` value, which
`clip_grads_per_group` returns **pre-clip**, divided by the group's threshold.

Two separate facts, then, from the first 1,000 steps at L=4:

1. **`depth_code` is clipped, mildly.** Pre-clip norms of 0.2–0.6 against its
   0.25 threshold: between not clipped at all and **2.4×**. Its update is
   scaled to roughly 42–83% of what the gradient asked. Real, not draconian,
   and the printed value carries only one decimal so the precision is poor.
2. **`depth_code` now *dominates*, which it did not at L=2.** `dc_ratio`
   1.2–3.0 here, against 0.13–0.42 in the L=2 log at step 6,050+. At L=4 the
   code is twice the size (shape `[L, n_ctx, d]`) and must differentiate four
   layers instead of two, so dominance is plausible rather than surprising.

**Whether the clip is "too harsh" is NOT established by this.** The threshold
was tuned on earlier depths and nobody has checked it against L=4's larger
code. What to check in the completed log: whether the pre-clip norm decays
after warmup or stays above 0.25, and whether the trained depth codes actually
separate four layers. If the norm stays pinned above threshold for most of the
run AND the codes do not separate, then the clip is binding and this rung is
not a clean depth measurement.

### Clip thresholds are depth-dependent in effect and were never re-tuned — **2026-09-28**

Prompted by the question "are the clips layer-depth dependent and mistuned?".
They are, and the biggest offender is not `depth_code`.

**Structure.** Building the model at L = 1, 2, 4, 8 and bucketing every
parameter by its clip group: exactly three groups grow **linearly in L** —
`creation_gate`, `destruction_gate`, `depth_code` — plus
`reverse_channel_scale`, which is per-layer by config. Everything else
(`V_theta`, `V_phi`, `register`, `score_head`, embeddings) is depth-invariant.
`clip_grads_per_group` clips each group **jointly**, so for the L-scaling
groups the norm being compared against a fixed threshold aggregates L
contributions and grows as √L (independent) to L (correlated), while the
number it is compared against never moves.

**Provenance.** `depth_code: 0.25` was tightened from 0.5 on **2026-08-23**
against the **L=16** g0.1 OpenWebText run. Every other override traces to
L=8/L=16 forensics (step 6435, step 71194). None was revisited when the
ladder dropped to L=1, 2 and 4. So the ladder — whose whole claim is that
rungs differ by *exactly one mechanism* — has been running each rung at a
different effective clip strength.

**Measured on the completed rungs** (from each run's `top[...]` line, which
`clip_grads_per_group` reports **pre-clip**):

| arm | group that dominates | share of logged steps | median pre-clip | threshold | over by |
| --- | --- | ---: | ---: | ---: | ---: |
| L=2 `none` | `reverse_channel_scale` | **98.5%** | 2.00 | 0.1 | **20×** |
| L=2 `attention_potential` | `reverse_channel_scale` | 88.9% | 1.00 | 0.1 | **10×** |
| L=2 `attention` | `relax_field` / `reverse_channel_scale` | 59% / 34% | 0.90 / 1.00 | — / 0.1 | — / **10×** |
| L=2 `none` no-RC | `depth_code` | 90.9% | 0.20 | 0.25 | not clipped |
| L=4 `none` (to step 8,350) | `reverse_channel_scale` | 77.2% overall, **100%** after step 4,000 | 1.20 (1.60 steady) | 0.1 | **12–16×** |

**The headline is the reverse channel, not the depth code.**
`reverse_channel_scale` is clipped by a **median factor of 10–20, on
essentially every step, on every arm that has a reverse channel** — including
the flagship. Its maximum reaches 8.8 against a 0.1 ceiling, i.e. 88×. The
`norc` arm, which has no reverse channel, is the only completed arm where
nothing is meaningfully clipped.

**Why this matters more than a tuning nit.** E5 measured the reverse channel
as *setting the layer-1 output direction outright*, and the book's central
claim rests on it. If its gate parameter's gradient has been scaled down by
10–20× at every step of training, the trained gate is not obviously where an
unclipped run would have put it.

**The argument on the other side, and it is the notebook's own.** The
`log_tau` comment in Cell 6 states it directly: *"Adam is close to
scale-invariant per parameter in steady state, so clipping log_tau's gradient
changes its own step size far less than the factor suggests."* The same
applies here. A group clipped 20× does not get a 20× smaller update; the
second-moment estimate absorbs most of a persistent rescale, and what survives
is the transient and the change in relative direction within the group.

#### Update from L=4 at step 8,350 — **2026-09-28**

The confound survives, one half of it is resolved, and the depth-scaling
prediction is **refuted**.

**1. The reverse-channel clip is confirmed, and its onset is explained.**
Dominant group by 1,000-step block:

| steps | dominant group | share | median pre-clip |
| --- | --- | ---: | ---: |
| 0–1,999 | `depth_code` | 63–80% | 0.40–0.50 |
| 2,000–3,999 | `reverse_channel_scale` | 85–95% | 1.00 |
| 4,000–8,350 | `reverse_channel_scale` | **100%** | **1.60** |

The handover at ~step 2,000 is the reverse channel's own warmup completing:
4,000 forwards at grad-accum 2 is exactly 2,000 optimiser steps. From then on
the gate parameter is clipped by a **median 16×** (max 45×) on *every* step.

**2. `depth_code` is not the problem, and my first worry is withdrawn.** It
dominates only during warmup, peaks at 2× its threshold, and stops being the
top group once the reverse channel comes online. It is a warmup transient at
L=4, not a persistent constraint.

**3. The √L prediction is refuted.** `reverse_channel_scale` is per-layer, so
its joint group norm should grow with depth. On the only like-for-like window
the two logs share (steps 6,050–8,350, 47 logged steps each):

| arm | dominant | share | median pre-clip | over threshold |
| --- | --- | ---: | ---: | ---: |
| L=2 `none` | `reverse_channel_scale` | 100% | 1.50 | 15× |
| L=4 `none` | `reverse_channel_scale` | 100% | 1.60 | 16× |

**1.50 against 1.60.** Doubling the depth moved the group norm by 7%, not the
41% that √2 predicts, let alone the 100% that full correlation would give.
So the per-layer gate gradients are not adding up across layers in the way
the parameter-count argument assumed — they are largely cancelling, or the
per-layer magnitudes shrink as depth grows. **Whatever the clip confound is,
it is not depth-dependent in the way I argued.** The structural observation
about group membership stands; the inference from it to a per-rung difference
in clip strength does not.

**What remains true, and it is the part that matters.** The gate parameter of
the mechanism this book is built on is clipped by a factor of 10–20 on
essentially every step of every arm that has it, at **both** depths measured.
That is a uniform confound rather than a per-rung one, which is better for
the ladder's internal comparisons and no better at all for the absolute
claims about what the reverse channel learns.

**So this is a confound of unknown magnitude, not a demonstrated error.**

### What the clipping costs — the reviewer-facing answer, **2026-09-28**

Written to be quotable. If a reader of the book or a referee of the TMLR
paper asks "your gate is clipped on every step; what does that cost you?",
this is the answer, with the evidence for each clause and the parts still
open named as open.

**The disclosure first.** Training does not use a single global clip. It uses
a global clip of 1.0 **plus nine per-group overrides**, of which the tightest
is 0.1 on `reverse_channel_scale` and `reverse_ch`. On every arm that has a
reverse channel, `reverse_channel_scale` is the largest pre-clip group on
essentially every logged step, at a median **10–20× its threshold** (15× at
L=2, 16× at L=4, maximum 88×). The thresholds were set from L=8/L=16
forensics in August–September 2026 and were never re-derived for the ladder.

**Why we nevertheless do not think it costs measurable perplexity.** Three
independent reasons, two of them measured:

1. **The gate is not being held up against a ceiling — it is falling.**
   Measured at the two ends of the L=2 flagship: 0.0612/0.0600 at step 500,
   0.0174/0.0148 at step 31,500, a fall of 3.5–4×. A clip limits step size in
   either direction, so what it can have done here is *slow a decline*. The
   unclipped counterfactual has a gate that falls **further and faster**, not
   one that grows.
2. **Adam largely absorbs a persistent rescale.** The update is
   `lr · m̂/√v̂`; scaling every gradient for a group by the same factor scales
   `m` and `√v` alike and leaves the ratio unchanged. Measured on the
   endpoint's optimizer state, this parameter's `|m|/√v` is **0.007–0.27**:
   what limits its movement is step-to-step gradient cancellation, not the
   clip.
3. **For this group the clip cannot distort direction at all.**
   `clip_grads_per_group` rescales a group by one scalar, and this group is
   just the per-layer gate scalars, so clipping multiplies every element by
   the same number. It cannot change which layer gets more gate — only how
   fast the whole vector moves. That removes a whole class of confound by
   construction.

**What we do NOT claim, and where the argument is still soft.** The clip
factor is not constant: the pre-clip norm ranges 1.0–4.5 against a 0.1
threshold, so clipping compresses large-gradient steps harder than small ones.
That is **not** a uniform learning-rate cut, it is variance reduction on the
gate's trajectory, and it can move where the gate settles. Direction and
magnitude are unmeasured. Reason 2 defends against a *persistent* rescale, not
against this.

**And the direction of any residual bias is the uncomfortable one.** If the
clip binds at all, it has slowed a shrinking gate, so the unclipped model has
a **weaker** reverse channel. Any bias makes the mechanism look more important
than it is, not less. Nothing here inflates the case against the reverse
channel; if anything it inflates the case for it.

**What would settle it, and what it would be worth.** The one untested
direction is a gate *larger* than the trained value: E5 swept λ from 1 down to
0 and found PPL rising monotonically with no knee, so the trained gate is best
among all smaller gates, and nobody has looked above 1. That is C1 below, free
and unrun. Note its status carefully — it is an **upper bound**, not a
prediction: this volume's own measurement is that inference-time ablation
overstated this mechanism 3.91× against trained-without's 1.31×, so whatever
C1 puts on the table is the most retraining could recover and probably several
times more than it would.

**Status: open, bounded, and biased in the direction that does not flatter the
thesis.** That is the honest position, and it is the one to quote.

---

### SPLM-family card audit — corrections applied **2026-09-28**

Audited all fifteen cards in the older collection against what the CfC+BAOAB
work has since measured. Two classes of error, both fixed; a third left
standing deliberately.

**1. Cards called reverse-channel models "purely conservative".** The
Fock-PARFLM card's opening sentence said so outright, and its capabilities
section derived a Riemannian geometry from the premise that *"all forces
derive from the gradient of a scalar potential"*. That card's own
`config.json` carries `use_reverse_channel: true`, and E1 later measured the
deflection of the layer step from the damped Vθ geodesic at **1.09 with the
channel on against 0.0003 with it off** — the non-gradient term carries most
of the step. Two further cards (`fock-attention`, `hybrid-splm`) propagated
the error by naming Fock-PARFLM in a list of "the purely conservative
variants". Corrected on four cards; the removal from the list is stated
rather than silent. `depthcond-vtheta` and `anisogaussian-vtheta` needed
nothing — they make no conservativity claim in prose, and their architecture
diagrams already label the reverse channel non-conservative.

**2. Three gamma-sweep cards read a small residual as proof of geodesy.**
The exact sentence: *"R ≈ 0 means the trajectory is a damped geodesic of the
metric induced by the model's own learned potential"*. That is the inference
the calibration withdrew — at one step per layer the residual cannot separate
a geodesic from a forced trajectory, and returns ≈1 for motion that is
geodesic by construction. Scoped in place, with a note saying what survives:
the **comparative** use across γ at otherwise fixed settings, including the
coincidence of the PPL and R̄ minima, which is what the sweep actually rests
on. The `hybrid-splm` card's claim that a diagnostic battery *"confirmed the
metric validity and characterised the damping-dominated dynamics"* was scoped
the same way — both of its headline readings have since been narrowed.

**3. Left standing: ~8% redundancy.** 21 verbatim blocks shared by three or
more cards, about 37 KB of 461 KB — two rival citation blocks, a 1.8 KB
capabilities bullet list on three cards, a 0.9 KB comparison table on three.
Cosmetic, and not worth churning the cards a third time in two days. The
duplicated correction notes are deliberate: each card must carry its own.

Verified after upload: **0 of 15** cards still carry a conservativity claim
alongside an enabled reverse channel, still list Fock-PARFLM as purely
conservative, or still state the unscoped geodesic reading.

---

### The C-series — probes on the clip confound, planned **2026-09-28**

Agreed: study it with targeted probes rather than one big re-run, and let L=4
finish first. Ordered by cost. **C0 and C2 are already run** — both are reads
of checkpoints we hold, needing no GPU.

---

#### C0 — which way is the gate moving? **RUN 2026-09-28. Result reframes the whole question.**

Read `reverse_channel_scale` out of the L=2 `none` checkpoints at both ends of
training:

| checkpoint | step | raw gate (per layer) | tanh |
| --- | ---: | --- | --- |
| `_step500_best` | 500 | 0.0612, 0.0600 | 0.0611, 0.0599 |
| `_best` | 31,500 | 0.0174, 0.0148 | 0.0174, 0.0148 |

**The gate FALLS by 3.5–4× over training.** It is not straining upward against
a ceiling the clip denies it. The hypothesis I started from — "the clip
suppressed a gate that wanted to grow" — is **refuted in its stated
direction**.

What survives is the mirror image, and it is the more awkward one: a clip
limits step size in *either* direction, so the unclipped counterfactual has a
gate that falls **further and faster**, i.e. an even weaker reverse channel.
If the clip has biased anything, it has biased the mechanism to look
**stronger** than it is, not weaker. Every downstream claim that leans on the
reverse channel's size inherits that direction.

---

#### C1 — extend the E5 slider above λ = 1 (FREE, minutes, no training)

`R11_LAMBDAS` in Cell 6b-11 stops at 1.0, so the sweep has only ever asked
what happens when the gate is turned **down**. The clip question is about
whether the trained gate is below where it would otherwise sit, and that is a
question about λ > 1, which has never been measured.

- **Change:** `R11_LAMBDAS = (2.0, 1.5, 1.25, 1.1, 1.0, 0.9, ..., 0.0)`.
- **Reading:** PPL falling above λ = 1 means the trained gate is below the
  inference-optimal value — the signature a binding clip would leave. PPL
  rising immediately means the gate is at its optimum and the clip did not
  bind the endpoint.
- **Pre-registered, from C0:** **PPL rises monotonically above λ = 1.** C0
  shows the model spent training *reducing* this gate, so a value above the
  trained one should be worse. A fall above λ = 1 would contradict C0 and
  would be the single most interesting outcome in this series.
- **C1 is an UPPER BOUND, not a prediction.** It scales the gate at
  inference on a model trained with the clip. By this volume's own rule —
  inference ablation overstated the reverse channel 3.91× against
  trained-without's 1.31× — whatever PPL C1 shows on the table is the most
  that retraining without the clip could recover, and probably several times
  more than it would. A null in C1 closes the question; a hit in C1 only
  licenses C4 and then C5.

---

#### C2 — the Adam realised-step audit (FREE, minutes; partial result in hand)

Tests the scale-invariance defence directly instead of arguing it. Adam's
update is `lr · m̂/√v̂`; a *persistent* k× clip scales `m` and `√v` alike, so
the ratio — and therefore the update — is unchanged. The measurable is
`|exp_avg| / √exp_avg_sq`, the fraction of full learning rate a parameter
actually moves at, read straight from the optimizer state the unstripped
checkpoints carry.

**Partial result:** the two 2-element optimizer states in the L=2 `none`
endpoint give `|m|/√v` of **0.007–0.27**. Far below 1, so this parameter's
gradient largely cancels step to step; it is not a consistent push being
throttled. Consistent with the clip being cosmetic, but **not yet decisive**:
the parameter-index-to-name map was inferred by numel and two states match.

- **To finish:** build the model, zip `named_parameters()` against the
  optimizer's `param_groups` ordering for an exact map, then report `|m|/√v`
  per clip group.
- **Reading:** if the heavily-clipped groups show the same `|m|/√v` as
  unclipped ones, Adam is absorbing the rescale and the clip is cosmetic for
  the endpoint.

---

#### C3 — what joint clipping can and cannot do (FREE, code reading + one assertion)

`clip_grads_per_group` rescales each group by a single scalar. For a group
that is **L scalars** — which `reverse_channel_scale` is — that is *exactly* a
learning-rate reduction for that group and nothing else: the per-layer
allocation is untouched, because every element is multiplied by the same
number. This **bounds** what the confound can be. It cannot have changed which
layer gets more gate, only how fast the whole vector moved. Verify and state
once; it removes a whole class of worry.

---

#### C4 — the paired short run (CHEAP, ~30 min GPU, after L=4)

From the L=2 `none` `_best` checkpoint, 500 steps twice with identical data
order: once as-is, once with `reverse_channel_scale`'s threshold at 2.0 (above
its 1.50 median pre-clip norm, so the clip stops binding). Compare the val-PPL
trajectory and the gate value.

- **Pre-registered, from C0 + C2:** gate ends **lower** in the unclipped arm,
  by less than 20%; val PPL within eval noise (±1.5). A larger gate in the
  unclipped arm would contradict C0.

---

#### C5 — the full re-run (11.7 h, ONLY if C1 or C4 separate)

Re-train the L=2 flagship with the threshold raised, compare settled PPL and
final gate. Do not spend this until a cheaper probe says it is warranted.

---

#### C6 — the watchdog is structurally blind to this parameter (FREE, documentation)

Not a clip question but the same parameter, and it belongs in the same place.
Cell 6's own comment: excluding `reverse_channel_scale` and `reverse_ch` from
the watchdog aggregate keeps it from false-triggering on their warmup ramp,
**"but it also means BOTH watchdog layers below are structurally blind to
them"** — and they appeared in the top-4 of nearly every spike in the
2026-08-23 burst. State this wherever run health is claimed.

---

#### C7 — what the clips cost in CONVERGENCE, not accuracy (FREE, no GPU)

**The whole C-series so far asks where training *ends*. Nothing in it asks
how long it took to get there** — and variance reduction on a parameter's
updates is precisely the kind of intervention that changes the second without
moving the first. Raised 2026-09-28; the gap was real and unexamined.

Free, because the data is already on disk: every run log prints the top
pre-clip group and its norm every 50 steps, for all five completed arms plus
L=4 in flight.

- **Measure:** per arm, the fraction of steps on which each group exceeds its
  threshold, binned by 1,000 steps, plotted against that arm's loss curve.
- **The comparison that does the work:** the `norc` arm is the one completed
  arm with nothing meaningfully clipped. If heavy clipping slows convergence,
  the arms with a reverse channel should reach a given loss later *in steps*
  than their curves' shape otherwise predicts — and `norc` is the control
  that has no such brake.
- **Confound to respect:** `norc` is also a different model, so a raw
  step-to-loss comparison is not clean. The usable signal is *within* an arm:
  does the loss curve bend where the clip-hit fraction changes? The
  reverse-channel warmup gives a natural discontinuity — at L=4 the dominant
  group hands over from `depth_code` to `reverse_channel_scale` at step
  ~2,000, and L=4 crossed ahead of L=2 at step 1,500. Whether those are
  related is exactly this probe's question.
- **Reading:** a bend in the loss curve coincident with the clip-hit handover
  is evidence the clip shapes convergence. No bend across five arms is a
  strong null.

---

**Sequencing.** C0 done. C1, C2 and C7 are free and can run the moment a GPU
session is spare — C7 needs no GPU at all — C1 needs one tuple edit and a probe cell, C2 needs no GPU
at all. C3 is a code read. C4 waits for L=4 to finish. C5 only on evidence.
**Do not start C1 or C4 while L=4 is training.**

**A second, free result.** The notebook logs `dc_ratio` precisely to find out
whether it is a leading indicator of spikes or a spike-time coincidence: of 7
captured spike events, every smooth cascade had `dc_ratio < 1.8` and both
localized blowups had `> 2.2`. L=4 is sitting at **2.1–3.0 through completely
healthy training** — no spikes, no watchdog, grad ≈ 0.5. That is evidence
against `dc_ratio > 2.2` being predictive on its own, from ordinary training
rather than from capture-time forensics, which is exactly the data the note
asked for.

**The resonance monitor is blind under `baoab_cfc_lowrank` — diagnosed
2026-09-28, and it is a library bug, not a stale clone.** The `EMPTY SUMMARY`
at step 500 was first blamed on a Colab session that cloned
`semsimula-diag` before the 2026-09-26 fix. That was wrong. Reproduced
locally against `origin/main` at `036d529`, on a toy Fock model with the
anisotropic depth-conditioned Vθ and `integrator='baoab_cfc_lowrank'`.

Call counts on one forward:

| hook | fires |
| --- | ---: |
| `_fock_layer_step` | 2 (one per layer) |
| `harmonic_terms_lowrank` | 2 |
| `lowrank_cfc_substep` | 4 |
| **`harmonic_terms`** | **0** |
| `cfc_substep` | 0 |

`observe()` opens a record in its wrapper on **`harmonic_terms`** (which
calls `mon._stash`) and closes it in the substep wrapper (`mon._finish`).
`_finish` returns immediately when `_pending is None`. Under the low-rank
integrator `harmonic_terms` is never called — the path calls
`harmonic_terms_lowrank` — so nothing is ever stashed, every `_finish` is a
no-op, and `summary()` is empty. The 2026-09-26 commit added
`_wrap_lowrank_substep`, the *closing* half, but no wrapper on
`harmonic_terms_lowrank`, the *opening* half. Half a fix.

**Consequence: the stiffness reading has been dark for every
`baoab_cfc_lowrank` run — the entire ladder.** Pulling the library at the
resume will not help; the fix has to be written. The ingredient exists
already: `omega_dt_report` reconstructs `G` via
`vt.harmonic_terms_lowrank(xis, h, comps=comps)`, so `observe()` needs the
same call stashing `λ_max(G Gᵀ)`.

### L=4 is genuinely behind, and it is NOT overfitting — diagnosis **2026-09-28, step 15,500**

The crossover reversed. L=4 led from step 1,500 to 8,500, then fell behind and
the gap is widening: +4.7% at 11,500, +6.5% at 12,500, +9.0% at 14,500, +7.7%
at 15,500. Its own validation curve now oscillates upward (88.68 → 91.81 →
92.60 → 90.57 → 92.76), which is not a plateau.

**The decisive measurement: L=4 is worse on TRAIN as well.** Next-token loss,
L=4 minus L=2, on the steps the two logs share:

| step | Δ ntp (nats) |
| ---: | ---: |
| 6,050 | −0.009 |
| 8,050 | −0.001 |
| 9,050 | **+0.044** |
| 11,050 | +0.044 |
| 13,050 | **+0.064** |
| 15,050 | +0.037 |

It crosses at the same place the validation curve does and stays positive. At
the last shared step L=4 is **+4.7% worse in perplexity terms on the training
objective**. So this is **not** a generalisation failure — the deeper model is
*fitting worse*. Overfitting is ruled out; every "more layers, more capacity"
intuition is ruled out with it.

**Curvature is higher at L=4 and rising faster.** `bproj_sig`, the σ_max
proxy on the low-rank factor:

| step | L=2 | L=4 |
| ---: | ---: | ---: |
| 6,500 | 51.7 | **54.3** |
| 10,500 | 64.0 | 67.8 |
| 15,500 | 74.6 | **83.1** |

and `sig_max` reaches 42.7 at L=4 by step 6,500, a level L=2 does not reach
until step ~9,500. L=4 is running a stiffer potential, earlier.

**Health is otherwise clean.** Zero `[spike]` events, zero watchdog triggers,
grad norm steady at 0.28–0.33, learning rate correct, dominant register index
rotating rather than locking. Nothing is diverging; it is simply losing.

#### Two candidate mechanisms, both consistent with everything above

**(1) Curvature compensation — the dt story.** `LADDER_T` is held at 8, so
L=4 means dt = 2. If the model grows ω to recover the displacement the smaller
step costs it, then ω·dt is unchanged and the finer integration buys nothing
while costing 2.41× the compute. `bproj_sig` rising faster at L=4 is exactly
what that compensation looks like. **This is the quantity the pre-registration
named as able to turn the result**, and it has turned.

**(2) Weight tying — the depth story.** L=4 adds only **0.069%** more
parameters (§ above). That is not a happy accident, it is the mechanism:
`reverse_ch` is one module reused at every layer, and V_θ, V_φ, the register
bank and the score head are all depth-invariant. Only `depth_code`, the
creation/destruction gates and the per-layer scalars can differentiate layers
at all. So L=4 asks the *same weights* to serve twice as many layer-roles,
with almost no new capacity to do it. A model asked to do more with the same
parameters fits worse — which is what the train loss says.

**These are distinguishable, and cheaply.** (1) predicts ω·dt is
approximately equal across L=2 and L=4 — the compensation is exact. (2)
predicts it is not, and that the per-layer parameters are the binding
constraint.

**The fix is now in `semsimula-diag` main** (cherry-picked 2026-09-28,
`409175b`). Verified against a fresh clone of the published branch: 134 tests
pass, and an end-to-end run on the real `FockMultiXiPARFLM` with the
anisotropic depth-conditioned Vθ under `baoab_cfc_lowrank` returns a summary
covering every layer at both L=2 and L=4. Colab will pick it up on the
restart.

**But the in-flight readings will NOT settle the question, and it is worth
being clear why.** The autosave fires near step 28,300, leaving ~4,200 steps
and about **8 resonance readings**, all inside the deep decay phase. Worse,
**there is no L=2 baseline to compare them against**: the L=2 run was trained
with the same broken hook, so its ω·dt was never recorded either. Eight
readings from one arm compare with nothing.

**The comparison that does settle it is an offline probe, not in-flight
logging.** `omega_dt_report` replays a captured batch under `observe()`, so
it runs against a *checkpoint*. Both `_best.pt` files exist. After L=4
finishes, run it on the L=2 and L=4 endpoints at matched batch and seed and
compare the ω·dt distributions directly. That is an apples-to-apples endpoint
comparison rather than eight samples from one arm, it costs minutes, and it
is the measurement that chooses between mechanisms (1) and (2).

Treat the in-flight readings as a smoke test that the fix works in Colab, and
nothing more.

**Do not stop the run.** It is the pre-registered rung and the settled value
is what scores the band; a partial curve scores nothing. Current trajectory
puts the settled value near **88–93**, i.e. *above* L=2's 66.98 and far above
the 55–65 band — a clean, large miss, and the most informative outcome the
rung could produce, because it says matched-T depth scaling has turned.

> **Corrected 2026-09-29.** Run 4 settled at **71.75**, not 88–93. The
> projection extrapolated the stable phase and ignored the WSD decay. The band
> was still missed (above 65), but the gap to L=2 is 7.1%, not about 30%. See
> "Run 4 settled at 71.75, and the ω·dt comparison scored" below.

**If it lands there, the next rung is not L=8.** It is L=4 at fixed dt,
letting T grow, which separates the two mechanisms above by construction.

#### Interrupted and resumed with the monitor live — **2026-09-28, step 20,000**

Interrupted at step 20,160 and resumed from `_best.pt` at step 20,000 (160
steps redone, ≈8 min). The interrupt was worth taking because the best
checkpoint had *just* been written at 20,000 — an hour earlier it was
stranded at 13,500 while the curve oscillated, and the same interrupt would
have cost 6,500 steps.

**The trap that nearly defeated it.** A Colab *runtime* restart does not wipe
`/content`, and the notebook's resonance block clones `semsimula-diag` only
`if not exists` — it never pulls. A restart alone would have re-imported the
stale clone and produced the same `EMPTY SUMMARY`. The clone had to be
removed by hand first. Only the ~24 h VM teardown avoids this, because that
gives a fresh disk.

**Verified live, from the Colab Terminal** (which runs while the training
cell is busy, unlike notebook cells, which queue):

    git -C /content/semsimula-diag log -1 --oneline
      -> 409175b resonance: open a record on the low-rank path too
    grep -c _wrap_harmonic_lowrank .../probes/resonance.py
      -> 2

Resume is clean: 106/106 tensors, optimizer state restored, best PPL 85.81
carried over, WSD windows unchanged (decay still begins at 21,125). So the
monitor covers **the entire decay phase**, ~25 readings, rather than the 8 at
the tail that waiting would have given.

**What the in-flight readings can and cannot do.** They confirm the fix works
under Colab and show how ω·dt moves through decay. They **cannot** settle
mechanism (1) vs (2): there is still no L=2 in-flight counterpart, because
that arm trained under the same broken hook. The discriminator remains the
offline `omega_dt_report` on both arms' checkpoints after this finishes.

**Operational note, learned the hard way.** The console output is the *only*
home of the per-step telemetry, the clip-group lines and the resonance
readings — `training_log.jsonl` on Drive does not carry them. The pre-restart
console was lost and survives only because it had been saved to
`~/Downloads/lowrank_depth_ladder_L=4_no_attention_LR=1.2e-03_32500_training_output.txt`.
Save the tab before every restart.

#### First ω·dt reading ever taken on a CfC+BAOAB ladder arm — **step 20,500**

    [resonance] step 20500  omega*dt p50=2.236  max=4.398  over_wall=60.938%

The hook fix works. Two things follow, one immediate and one pre-registered.

**1. The propagator is not a convenience, it is load-bearing.** The median
token-layer pair sits at **ω·dt = 2.24**, above the Störmer/leapfrog wall of
2, and **60.9% of all pairs are past it**. Under an explicit integrator the
majority of this model would be provably amplifying at every step. The
closed-form A-substep integrates those modes exactly, so this is a
*stiffness* reading and not an instability — but it is the direct measurement
that the integrator change was necessary rather than precautionary. **This
has never been measured on a CfC+BAOAB arm before** (the hook has been broken
for the entire ladder), and it retires the caveat in the Verlet-instability
card that the ω·dt evidence was Verlet-era only.

**2. It sets up a decisive, falsifiable test of the L=4 deficit.** At L=4,
dt = 2, so the measured ω·dt = 2.236 implies **ω ≈ 1.118**. At L=2, dt = 4.
The two competing mechanisms predict different L=2 readings:

| hypothesis | what it says | predicted ω·dt at L=2 |
| --- | --- | ---: |
| **(1) exact curvature compensation** | the model doubles ω when dt halves, so the operating point is depth-invariant and finer integration buys nothing | **≈ 2.2** |
| **(2) no compensation** | ω is a property of the learned potential, unchanged by depth; halving dt genuinely halved the product | **≈ 4.5** |

**The weight-space proxy already favours (2), and not weakly.** `bproj_sig`
at the matched step 20,450: **97.35 at L=4 against 81.77 at L=2**, a ratio of
**1.19×** — nowhere near the 2× that exact compensation requires. Scaling ω by
that measured ratio predicts **ω·dt(L=2) ≈ 3.8**, much closer to (2) than to
(1).

If that holds, the reading is: **halving the timestep really did move the
operating point, the model only partly compensated, and L=4 is a different
dynamical system rather than a finer integration of the same one.** The
deficit then belongs to mechanism (2), weight tying across depth — the same
weights serving twice as many layer-roles with 0.069% more parameters.

**Pre-registered before the measurement: ω·dt(L=2) lands in [3.3, 4.2].**
Below 2.6 refutes this and hands it back to compensation; above 4.7 says ω
*fell* with depth, which neither mechanism predicts and would need its own
explanation.

**The measurement.** `omega_dt_report` on the L=2 `_best.pt` and the L=4
`_best.pt`, same batch, same seed, once this run finishes. Minutes of work.
The local L=2 checkpoints already carry everything needed; the only thing
missing here is an OpenWebText batch, so it runs in Colab, not on the laptop.

#### Nine readings after the fix — steps 20,500–24,500, **recorded 2026-09-28**

Source: the appended run-4 log
(`lowrank_depth_ladder_L=4_no_attention_LR=1.2e-03_32500_training_output.txt`),
resumed with `semsimula-diag` at 409175b. Every `[resonance]` line from
20,500 on is populated; every one before the restart reads `EMPTY SUMMARY`.

| step | ω·dt p50 | ω·dt max | over wall (2) | `bproj_sig` | val PPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20,500 | 2.236 | 4.398 | 60.9% | 97.48 | 87.05 |
| 21,000 | 2.346 | 4.573 | 65.6% | 98.88 | 85.97 |
| 21,500 | 2.318 | 4.579 | 63.5% | 100.10 | 85.83 |
| 22,000 | 2.271 | 4.589 | 61.9% | 101.40 | 85.62 |
| 22,500 | 2.283 | 4.661 | 63.1% | 102.64 | **82.91** |
| 23,000 | 2.296 | 4.692 | 63.4% | 103.76 | 84.08 |
| 23,500 | 2.262 | 4.503 | 62.8% | 104.85 | 83.99 |
| 24,000 | 2.258 | 4.627 | 62.6% | 105.67 | 83.97 |
| 24,500 | 2.293 | **4.945** | 64.1% | 106.25 | **81.41** |

`bproj_sig` is read from the training line of the same step. Linear fits
over the nine points, slope per 1,000 steps against the fit's residual std:

| series | mean | slope / 1k | residual std | change 20,500 → 24,500 |
| --- | ---: | ---: | ---: | ---: |
| ω·dt p50 | 2.285 | −0.004 | 0.035 | +2.5% |
| ω·dt max | 4.619 | +0.077 | 0.115 | +12.4% |
| over wall | 63.1% | +0.13 pp | 1.41 pp | +3.2 pp |
| `bproj_sig` | 102.3 | +2.24 | 0.33 | +9.0% |

**1. The median is stationary, and past the wall.** p50 = 2.285 ± 0.033
with no slope distinguishable from noise, so ω ≈ 1.14 at dt = 2. About 63%
of token-layer pairs sit above ω·dt = 2 at every reading. The step-20,500
reading was representative, not a transient. The WSD decay began near step
21,700 and has not moved the median, while val PPL keeps improving (new
bests at 22,500 and 24,500): the operating point is holding while the loss
falls.

**2. The max drifts up, mildly.** 4.40 → 4.95, a fitted rise of about 0.31
over the window against 0.115 residual scatter, and roughly half of the
total change is the last reading alone. Stiffness, not instability — the
A-substep integrates every pair exactly — but worth watching through the
rest of the decay phase.

**3. Caveat on the pre-registered band: `bproj_sig` tracks the tail, not
the median.** Over the same nine readings `bproj_sig` rose a smooth 9%
(residual std 0.33) while p50 did not move: corr(p50, `bproj_sig`) = −0.16,
corr(max, `bproj_sig`) = +0.67. `bproj_sig` behaves like a σ_max-type
quantity. The [3.3, 4.2] band was derived by scaling the **median** by the
L=4 / L=2 `bproj_sig` ratio (97.35 / 81.77 = 1.19×, giving ≈ 3.8), so its
derivation leaned on a proxy that, on this evidence, does not follow the
statistic it was used to scale.

What this does and does not change:

- **The band stands as pre-registered.** It was recorded before the L=2
  measurement and Cell 6b-13 scores it on p50 exactly as written. It is not
  moved.
- **A hit is less confirmatory than it looked, and a miss is less
  damning.** The two mechanisms still predict different L=2 medians (≈ 2.2
  vs ≈ 4.5), and that discrimination is unaffected; what weakened is only
  the route to the narrow band inside it. A miss inside [2.6, 4.7] should be
  read against the proxy before it is read against mechanism (2).
- **Exploratory secondary comparison, labelled as added after these
  readings:** the ratio max(L=2) / max(L=4). That is the comparison
  `bproj_sig` actually speaks to. Cell 6b-13 already writes `max` and
  per-layer p50 to `results/omega_dt_endpoint.jsonl`, so no cell change is
  needed; it is reported beside the scored p50, never in place of it.

**4. The L=4 endpoint reading should be unremarkable.** With p50 this
stationary, 6b-13 on the final L=4 checkpoint should land near 2.29. If it
does not, the batch or the checkpoint is the first suspect, not the model.

**Plan unchanged:** run 4 to 32,500; 6b-13 on both arms; then D1.

#### Incident: Cell 6 resumed the FINISHED L=2 arm — **2026-09-28/29**, repaired

**What happened.** A laptop reboot made the browser restore an older copy
of this notebook: the committed `LADDER_L = 2` default (run 4's
`LADDER_L = 4` existed only as an uncommitted edit in the Colab tab) and a
version from before Cell 6b-13 was pushed (`39539e1`). Cell 0 therefore
resolved run 3's folder,
`…_cgqk_L2probe_…_idt4_lr0p0012_noattn`; Cell 2 picked its `_best.pt`
(step 31,500, val PPL 66.56); and Cell 6 resumed the completed arm and
trained **31,501 → 32,183** before being interrupted. Nothing in the
notebook checked that the resolved arm was the one intended.

**Damage and repair — verified byte-exact.**

| artefact | state after the accident | repair | verified |
| --- | --- | --- | --- |
| `checkpoints/…_best.pt` | untouched: the one eval reached (step 32,000) gave 68.30, not a new best, and no checkpoint was written | none | md5 `4ea4dd87cec6958a06a5540bdb3d277f`, matching the local copy (step 31,500, PPL 66.56) |
| `results/training_log.jsonl` | 15 lines appended (737 total): per-step records 31,550–32,150 and the 32,000 eval | trimmed to the first 722 lines on 2026-09-29 | 722 lines, 482,567 bytes, md5 `ad29260a02d13b01c3a4d75697943719`, identical to the local copy |
| any other file in the folder | none modified (`find -mmin -180` listed only the log) | none | — |

Run 3's folder is byte-identical to its pre-accident state, and every
number recorded for run 3 (settled 66.98) stands. The appended lines were
kept off-Drive as `/content/training_log.damaged.jsonl` (VM-local).

**Prevention — completed-run guard, added to the notebook 2026-09-29.**
Cell 0 gains `ALLOW_EXTEND_COMPLETED_RUN = False` and a helper,
`_ladder_final_eval()`, that finds an eval record at
`step >= TOTAL_STEPS` in the folder's `training_log.jsonl`.

- **Cell 6 refuses** to start, before it loads a checkpoint or opens the log,
  when that record exists, unless the flag is set.
- **Cell 5b only warns**, so probe-only sessions (0 → … → 5b → 6b-*) on a
  finished arm keep working.
- **Tested against real logs:**
  - run 3's pristine log, and the same log with the accident's 15 lines,
    both refuse;
  - a log ending at 28,500 (run 4's situation), a missing log and a
    truncated last line all proceed;
  - the flag overrides as intended.
- **Not covered:** calling `run_training(...)` again inside a session whose
  Cell 6 already passed the guard.

**Unplanned preview of the L=2 ω·dt reading — NOT the scored measurement.**
The accidental run printed one resonance line from the L=2 arm:

    [resonance] step 32000  omega*dt p50=3.797  max=6.926  over_wall=98.334%

It lands **inside the pre-registered band [3.3, 4.2]**, near the ≈ 3.8
point estimate. Against L=4's stationary p50 ≈ 2.25–2.29, the ratio is
**≈ 1.69**: neither exact compensation (≈ 1) nor none (≈ 2). As ω, that is
≈ 0.95 at L=2 against ≈ 1.13 at L=4, so halving dt raised ω by only ≈ 18%.
That is partial compensation, which leans towards mechanism (2).

Why it is not the result, and how it must be reported:

- **Wrong weights.** They are 500 steps past `_best.pt`, trained at LR ≈ 6.5e-5.
  That is tiny this late in decay (`bproj_sig` sat at 84.45 ± 0.03 throughout),
  but it is not the checkpoint the band scores.
- **Wrong batches.** It used training batches, not 6b-13's fixed seed
  `20260928`.
- **The L=2 6b-13 run is no longer blind.** The band itself was fixed before
  either reading, so the pre-registration stands, but the scored L=2 value
  must be reported as **"measured after an unplanned preview of 3.80"**.
- **The L=2 max (6.93) and share over the wall (98.3%) are far above
  L=4's** (≈ 4.4–4.9 and ≈ 62%). The exploratory max ratio is therefore
  ≈ 1.5, to be confirmed by 6b-13 on both endpoints.

#### Run 4 settled at 71.75, and the ω·dt comparison scored — **2026-09-29**

**The rung.** Run 4 finished cleanly at 32,500. Best 70.19 (step 31,000);
last three evals 71.47, 71.93, 71.84, so **settled 71.75**. Log:
[`results/.../L4_idt2_lr0p0012_noattn_altE_fromscratch_32500_result.txt`](../notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_L4probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt2_lr0p0012_noattn/L4_idt2_lr0p0012_noattn_altE_fromscratch_32500_result.txt).

| L | dt | T | settled | against L=2 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 8 | 87.09 | +30.0% |
| **2** | **4** | **8** | **66.98** | — |
| 4 | 2 | 8 | **71.75** | **+7.1%** (best against best: +5.5%) |

- **Pre-registration scored: MISS.** The band was 55–65, and 71.75 lands
  above it, on the side where depth does not pay.
- **The interior optimum at fixed T stands, but narrowly.** L=4 is 7% behind
  L=2, not the ~28% the stable-phase readings suggested.
- **The "88–93" projection above was wrong by about 20 PPL.** It
  extrapolated the stable phase and ignored the WSD decay, which took L=4
  from 85.8 (step 21,125) to 71.8. Most of the gain arrived in the last
  third, as it does on every rung.
- **The gap is not a trend.** It was about 9% at step 25,000, about 3.5% over
  the 28,000–28,500 window, and 7.1% settled. The last 1,500 steps moved L=4
  slightly the wrong way (70.19, then 71.47, 71.93, 71.84), which is within
  eval noise but is why best and settled differ.

**The ω·dt endpoint comparison (Cell 6b-13).** Both arms were measured at
`_best.pt` on the same seed (20260928) and the same tokens.

| arm | step | ω·dt p50 | ω | max | over the wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| L=2, dt = 4 | 31,500 | **3.796** | 0.949 | 6.766 | 98.4% |
| L=4, dt = 2 | 31,000 | **2.292** | 1.146 | 4.484 | 63.9% |

- **Pre-registered band for L=2, [3.3, 4.2]: HIT** at 3.796.
- **Recorded before reading this:** the L=2 value was previewed twice — once
  in-flight by accident (3.797, the incident entry), and once by a local dry
  run (3.796) — so the scored run was not blind. The band itself was fixed
  before either.
- **The route to the band was partly a proxy mismatch.** `bproj_sig` tracks
  the tail, not the median (nine-readings entry). So the hit confirms the
  hypothesis less strongly than the band's width suggests.
- **Exploratory, added after the in-flight readings:** max(L=2) / max(L=4) =
  1.51.

**What it says.** Halving dt raised ω by only **21%** (0.949 → 1.146); full
compensation would have doubled it. The ω·dt ratio is 1.66, against 1 for
exact compensation and 2 for none. In log terms the model compensated about
27% of the way, so **the smaller step genuinely moved the operating point.**
Of the two mechanisms, this is mechanism (2).

**The per-layer breakdown locates the compensation.** Aligned by integration
time:

| time window | L=2 layer : ω | L=4 layers : ω |
| --- | --- | --- |
| t ∈ [0, 4] | 0 : **0.84** | 0 : **1.68**; 1 : 1.00 |
| t ∈ [4, 8] | 1 : 1.13 | 2 : 1.25; 3 : 0.74 |

- **At the first layer the compensation is full.** ω doubles (0.84 → 1.68),
  so ω·dt is held (3.34 against 3.37).
- **Beyond the first layer there is none.** L=4's later layers run no stiffer
  than L=2's second layer, and its last layer is the softest in either model:
  23% over the wall, against 97–100% at every L=2 layer.
- **So L=4 is a different dynamical system, not a finer integration of the
  same one.** The first layer adapts to the smaller step and the rest do not.
  That is the pattern weight tying predicts: everything except `depth_code`,
  the gates and the per-layer scalars is shared across layers, so only
  limited per-layer freedom exists to re-tune the curvature at each depth.
  It is **consistent with** tying rather than a demonstration of it.

**Still open, and what separates the candidates.** The ω·dt result narrows
the question but does not close it:

- **D1** (L=4 at dt = 4) decides between "the step size" and "the depth".
- **Offline per-layer diagnostics on both checkpoints** — knocking out one
  layer at a time, and comparing 6b-8 and 6b-9 per layer — can tell whether
  layers 2–3 at L=4 contribute anything, at no training cost.
- **The per-group clip difference** (the reverse-channel gate is clipped
  12–16× at L=4 against 20× at L=2) and **an LR that was tuned at L=2** remain
  unexamined.

Results filed with both logs, as `D2_6b13_omega_dt_endpoint.txt` in the L=2
and L=4 results folders.

#### L=4 with live gradients (Gen 3), 3,000-step probe — pre-registered **2026-10-02, at launch, before any eval**

**Run.**

- Tag: `…cgqk_vplive_xilive_L4probe…idt2_lr0p0012_noattn`.
- Config: run 4's config plus `VPHI_GRAD_PATH = XI_GRAD_PATH = 'live'` and `PROBE_MAX_STEPS = 3_000`. The reverse channel is on (the Fock arm), with L=4, dt=2 and T=8.
- The forward force is identical to run 4's. Verified locally at L=4 from fresh init through the notebook's cells:
  - max |Δlogit| = 0;
  - the gradient from a layer step into earlier tokens goes from exactly 0 at every layer to 0.51 / 0.069 / 0.069 / 0.070.

**Scored at step 3,000** against run 4's own evals (1,000: 260.64; 2,000: 165.58; 3,000: **138.94**). Noise band ±2.5%.

| outcome | step-3,000 PPL | reading |
| --- | --- | --- |
| **move** | **≤ 135.5** | the starvation fix helps at L=4 too; extend to 32,500 |
| null | 135.5 – 142.4 | no early effect at L=4. Extending is still informative, since F3.1's gap widened through training |
| worse | > 142.4 | check clip-hit and grad-norm before reading it |

- **Point estimate: about 125** (−10%), band 112–135.
  - The basis is F3.1's −19.2% at 3k, discounted because this arm also carries the register path, which was never starved.
  - More starved layers (four, not two) could push the other way. That direction is recorded so it can be scored.
- **What this run can and cannot answer.**
  - On its own it prices the starvation fix at L=4 (against 71.75 settled).
  - Whether L=4 beats L=2 *on the live convention* needs the L=2 Fock-live arm (P2.2), which has not run. Comparing it with the Gen 2 L=2 arm (66.98) mixes conventions and will not be read as the depth answer.

#### L=4 live probe scored at step 3,000: **119.77 — MOVE, and a band HIT** — **2026-10-02**

Log: [`results/…cgqk_vplive_xilive_L4probe…idt2…_noattn/L4_idt2_lr0p0012_vplive_xilive_noattn_probe3000_result.txt`](../notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_vplive_xilive_L4probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt2_lr0p0012_noattn/L4_idt2_lr0p0012_vplive_xilive_noattn_probe3000_result.txt)

| step | run 4 (L=4, Gen 2) | **L=4 live** | Δ |
| ---: | ---: | ---: | ---: |
| 500 | 495.47 | **475.12** | −4.1% |
| 1,000 | 260.64 | **242.99** | −6.8% |
| 1,500 | 195.16 | **178.94** | −8.3% |
| 2,000 | 165.58 | **147.42** | −11.0% |
| 2,500 | 145.57 | **127.44** | −12.5% |
| **3,000** | **138.94** | **119.77** | **−13.8%** |

- **Scored: MOVE** (criterion ≤ 135.5) **and HIT** (band 112–135; the point estimate of about 125 was 5 too high).
- The gap widened at every eval, as F3.1's did.
- The gain is smaller than F3.1's −19.2%, as predicted: this arm also has the register path, which was never starved.
- **The best step-3,000 number in the programme.** For comparison at 3k:
  - F3.1 (L=2 conservative-only, live) 127.73;
  - `attention` 131.20; `rglive` 133.74;
  - L=2 no-exchange (Gen 2) 140.61.
- **Health:**
  - 0 clip-hits in 60 logged steps (max grad-norm 0.79; run 4 was also 0 in its first 3k);
  - memory 30.9 GB and 3.14 s/step, both as run 4.
- **Resonance:** ω·dt median 1.22 at step 3,000 with 3.5% past the wall, rising from 0.63. That is far below F3.1's 2.16 at the same step, since dt is 2 here against 4 there. Run 4's monitor recorded nothing (hook bug), so there is no Gen 2 comparison.

#### L=4 live extended to 32,500 — pre-registered **2026-10-02, at step 3,000, before step 3,001**

**Basis.** The ratio of settled PPL to step-3,000 PPL across the finished arms:

| arm | ratio |
| --- | ---: |
| run 4 (L=4, Gen 2) | 0.516 |
| L=2 no-exchange | 0.476 |
| `attention` | 0.484 |
| `rglive` | 0.457 |
| F3.1 | 0.452 |

Applied to 119.77, the two extremes give 54.1 (F3.1's ratio) and 61.8 (run 4's).

**Prediction:**

- **Settled: point 57, band 52–63.**
- **Against run 4 (71.75):** below it, called near-certain. This is the starvation fix at L=4.
- **Against F3.1 (57.76, L=2 conservative-only live):** even odds.
- **Against the Gen 2 L=2 Fock arm (66.98):** below it, very likely. **This is not the depth answer**: it mixes conventions.
- **What would answer the depth question:** whether L=4 beats L=2 under live gradients needs P2.2 (L=2 Fock live). That run is still unrun and is now the decisive next run.

**Health, scored:**

- clip-hit rate below run 4's full-run rate;
- SCAF CLEAN at every audit;
- no watchdog trigger.

The ω·dt median is to be reported, not scored.

**Cost:** about 25.5 h for the remaining 29,500 steps at 3.1 s/step, so at least one session break and a resume.

#### CG3 (forecastability, Cell 6b-10) on the L=4 live arm — pre-registered **2026-10-02, before the run finishes and before 6b-10 is run on it**

Book: §18 "The replay instrument and the forecastability test" (CG3, formerly note-label E3). At L=2 the test cannot be read: the first step projects the embedding onto the LayerNorm sphere (the book's sphere-obstruction proposition). **L=4 is the first arm where it can.**

**Rules fixed in advance:**

- **Read only layers ℓ ≥ 2.** At ℓ = 1, s₀ is still the embedding projection, and its radial fraction is printed by the cell. The cell's own "ℓ ≥ 1" summary line is therefore *not* the scored number; the per-layer rows at ℓ = 2, 3 are.
- **Use the tangential coherence (a⊥), not the raw value.**
- **Paired control: run 6b-10 on run 4** (the Gen 2 L=4 `_best.pt`, same tokens). The live-minus-detached difference is the gradient-convention effect on forecastability.
- **Use the same matched GPT-2 and the same ε values** {10⁻², 3·10⁻², 10⁻¹}.

**Predictions** (L=4 live; Gen 2 run 4 expected to sit on the same side but weaker):

| metric (ℓ ≥ 2) | point | band | "forecastable" requires |
| --- | ---: | --- | --- |
| (a⊥) tangential coherence, mean | +0.10 | −0.10 to +0.30 | > 0; "beats GPT-2" requires > +0.195 (called about 1 in 3) |
| (b) true-velocity forecast error | 0.95 | 0.85–1.05 | **≤ 0.90**, i.e. 10% below the stay-put null |
| (b) finite-difference error | 1.00 | 0.90–1.15 | ≤ 0.90, and below GPT-2's per-layer value |
| (c) per-step growth, every ε | 0.92 | 0.88–0.97 | ≤ GPT-2's (0.979–0.982), with Fock drift across ε ≤ 0.02 |

**Verdict rule:**

- **Forecastable:** (b) true-velocity ≤ 0.90 **and** (a⊥) > 0.
- **Not forecastable:** (b) ≥ 1.0 at both ℓ = 2 and ℓ = 3, i.e. on the stay-put null.
- **Indeterminate:** anything in between.
- Metric (c) is reported separately. It speaks to contraction, not to forecasting.

#### L=4 live scored: **50.10 settled — within 0.6% of the matched GPT-2** — **2026-10-03**

Log: [`results/…cgqk_vplive_xilive_L4probe…idt2…_noattn/L4_idt2_lr0p0012_vplive_xilive_noattn_32500_result.txt`](../notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_vplive_xilive_L4probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt2_lr0p0012_noattn/L4_idt2_lr0p0012_vplive_xilive_noattn_32500_result.txt)

| | **L=4 Fock, live V_φ + ξ** | run 4 (L=4, detached) | F3.1 (L=2 cons.-only, live) | matched GPT-2 |
| --- | ---: | ---: | ---: | ---: |
| **settled** (last 3: 50.05, 50.77, 49.48) | **50.10** | 71.75 | 57.76 | 49.81 |
| best | 49.48 (32,500 = final) | 70.19 | 57.35 | 49.76 |
| ratio to GPT-2 | **1.006×** (+0.006 nats) | 1.441× | 1.160× | 1 |
| parameters | 76.8M | 76.8M | 76.6M | **33.7M** |

**Scored against the pre-registration (point 57, band 52–63, frozen at step 3,000):**

- **MISS on the good side.** 50.10 is 1.9 below the band and 6.9 below the point.
- The settled-to-3k ratio is 0.418, the lowest of any arm (previous range 0.452–0.556). Every live arm has consolidated more than its predecessors, and every forecast built on them has been too pessimistic.
- **Below run 4 (71.75), called near-certain: YES**, by −30.2%. This is the starvation fix at L=4, against −34.3% at L=2 for the conservative-only arm.
- **Below F3.1 (57.76), called even odds: YES**, by −13.3%.

**Health, scored:**

- Clip-hit 0.5% (3 of 650, max 1.96), against run 4's 0.0%: a MISS on the letter of the criterion, though at a trivially low rate.
- SCAF CLEAN at all seven audits. Future perturbation was exactly 0.0, and the leak tax was at most 2×10⁻⁴ nats where measured (at 32.5k the honest stage was skipped, as before).
- **The first in-flight Tier B readings on a joint-bank model**, after the SCAF fix: Tier A 0.0 and Tier B 0.0 at steps 30,000 and 32,500.
- No watchdog trigger.
- The ω·dt median at the endpoint was 2.34 in flight (run 4's 6b-13: 2.29).

**What it means, and what it does not:**

- **Parity with the matched GPT-2 at equal tokens, within eval noise.** The 0.29-PPL gap is smaller than the 1–3 PPL step-to-step bounce of these evals.
- **The parameter count is not matched**: 76.8M against 33.7M. Roughly 19M of the difference is the untied output head; the rest is V_θ's bank, V_φ and the registers.
- The baseline is untuned (one LR, separately chosen) and twice as deep (L=8). The run is one seed.
- The fair statement is therefore: *the first conservative-forward model in the programme to reach the matched transformer's perplexity at the same token budget, with 2.3× its parameters and half its depth*. A parameter-matched baseline (a wider or deeper GPT-2 at 76.8M, or a slimmer Fock arm) is needed before anything stronger is said.
- **Framing, agreed 2026-10-03.** Neither side is tuned, and the conservative family is immature: a month of development (joint V_θ, exact low-rank integration, the gradient fix), no LR or width tuning for this architecture, and V_φ still in the explicit kick. 50.10 is therefore a lower bound on the family, not its ceiling. The parameter caveat is about the *comparison*, not the model. The cheap control is a GPT-2 at about 77M parameters on the same notebook (about 3 h). It decides whether "parity" holds on a parameter-matched footing.
- **Headroom noted:** with sources frozen for the layer, V_φ is a function of h_t alone. If built from Gaussian-type pair terms, it satisfies the analyticity hinge (book Prop 80), so its stiff part could join the exact flow (Props 97–100) rather than ride in the kick, removing the last second-order autograd chain in the force.
- **Headroom, also noted:** position-dependent damping. Every arm runs a constant γ = 0.1, and the stiff modes sit at a damping ratio ζ ≈ 0.05 (book §8.8). The exact friction step already accepts a per-token γ, and the precondition (a clean constant-γ baseline) is met.
  - A scalar γ(h), or the mode-resolved friction Γ = 2ζ√(L/m), is a local change.
  - So is the palindromic step order (O half-steps around the kick), which the same section shows is second-order at no extra cost.
- **Depth:** L=4 live beats L=2 conservative-only live by 13.3%. That comparison mixes depth with the Fock mechanism, so the depth answer still needs P2.2 (L=2 Fock live).
- Run 4's L=4 < L=2 inversion, the original question that opened the depth investigation, does not survive the gradient fix: L=4 live (50.10) is far below every L=2 arm measured so far.

#### L=4 live: post-training probes, and CG3 scored — **2026-10-03**

Outputs filed with the run (`results/…cgqk_vplive_xilive_L4probe…/Cell-6b-*`). All gates passed bit-exactly. On the probe batches the model scores PPL 48.02, against GPT-2's 47.42.

**CG3 (forecastability, 6b-10), scored by the rule frozen on 2026-10-02:** layers ℓ ≥ 2 only, tangential coherence.

The geometry is clean exactly as the sphere proposition requires: the radial fraction of s_{ℓ−1} is **0.86 at ℓ = 1 and 0.00 at ℓ = 2, 3**.

| metric (ℓ ≥ 2) | ℓ = 2 | ℓ = 3 | mean | pre-registered point / band | GPT-2 (per layer) | verdict |
| --- | ---: | ---: | ---: | --- | --- | --- |
| (a⊥) tangential coherence | +0.749 | +0.302 | **+0.526** | +0.10 / −0.10 to +0.30 | +0.22 (ℓ = 2), +0.21 (ℓ = 3) | > 0 ✓; > GPT-2's +0.195 ✓ (called 1 in 3); band MISS, good side |
| (b) true-velocity forecast error | 0.583 | 0.935 | **0.759** | 0.95 / 0.85–1.05 | — | ≤ 0.90 ✓ (layer 3 alone does not) |
| (b) finite-difference error | 0.664 | 0.969 | 0.817 | 1.00 / 0.90–1.15 | 1.878, 1.331 | ≤ 0.90 ✓; below GPT-2 at both layers ✓ |
| (c) per-step growth, ε = 10⁻² / 3·10⁻² / 10⁻¹ | | | 0.930 / 0.936 / 0.950 | 0.92 / 0.88–0.97 | 0.979–0.982 | ≤ GPT-2 at every ε ✓; drift 0.020, at the 0.02 limit |

- **Verdict: FORECASTABLE**, by the frozen rule: (b) true-velocity ≤ 0.90 and (a⊥) > 0. This is the first positive forecastability result in the programme.
- The integrator's own velocity beats the finite-difference momentum at both layers (0.583 vs 0.664, 0.935 vs 0.969). So the second-order state carries forecast information beyond the position sequence.
- The effect is concentrated at ℓ = 2, where the steps are small (|s|/|h| = 0.14–0.17). At ℓ = 3, the large output step (|s|/|h| = 1.24), forecasting is near the stay-put null.
- **The inertial fraction |Δt·v|/|step| is 1.41.** The velocity alone would overshoot the step, and the forces brake it; this is consistent with the stiff-mode rotation (ω·dt ≈ 2.3).
- **Not yet separable from the gradient fix:** the paired control, 6b-10 on run 4, has not run. Until it does, this says the L=4 live model forecasts, not that live gradients made it so.

**CG1 (6b-9):**

- R(geo) = 1.369; LayerNorm −0.918; **V_φ −0.0035**; reverse channel −0.449; interaction I = +0.001.
- **V_φ is inert again**, with live gradients, in the Fock arm. In the conservative-only live arm (F3.1) it woke up (−0.291).
- So V_φ and the register path behave as **substitutes**: with the register path available, V_φ is not used.
- This is the V_φ × Fock interaction the factorial (runs 10/11) was designed to measure, seen here first, across a depth difference.
- **Predictions it makes:**
  - P2.2 (L=2 Fock live) will also show V_φ inert;
  - removing V_φ will cost little with the Fock mechanism on (run 11) and much without it (run 10).
- The reverse channel's deflection is 0.449, against 0.90 in the Gen 2 L=2 Fock arm (step sizes differ). R_v(geo+LN) = R_v(geo) to every digit at all four layers.

**CG2 (6b-7):**

- Inertia worth **+26.30 PPL (+54.8%)**, against run 4's in-flight-only record, which has no 6b-7.
- Refinement still reads MAPS (N=6: +104; N=8: +201). Coarsening is far worse than refining (N=3: +799).
- **Extra hops at fixed dt degrade gently:** 1.5× the trained depth costs +40.6%, against +92% (F3.1), +341% (`rglive`) and +331% (Gen 2 `attention_potential`) at 1.5× for the L=2 arms. The Gen 2 conservative-only arm, at +26%, is the only one gentler.

**CG6 (6b-12):** uniform forcing again. At layer 3, 99.8% of tokens have R > 0.75 against the conservative+LN step.

**CG7 (6b-13):**

- ω·dt median **2.347**, against run 4's 2.292: overall stiffness unchanged by the gradient fix at L=4, unlike L=2 conservative-only (4.32).
- It is redistributed across layers (layer 0: 2.69 vs 3.37; layer 3: 2.27 vs 1.49), and the tail is longer (max 6.15 vs 4.48).

#### "Can we fix L=4 by adding parameters?" — **analysed 2026-09-28**

Short answer: **partly, and the cheap part is worth trying; the expensive part
is not a fix but a different thesis.**

**What is actually shared.** Bucketing every parameter by whether it grows
with L (build at d=384, n_registers=32, ladder shape):

| group | tied across layers? | share |
| --- | --- | ---: |
| **V_θ** | **tied** | **~59% of non-embedding** |
| V_φ, register bank, reverse channel | tied | small |
| `depth_code` | per-layer | **6,144 params at L=4 (0.013%)** |
| creation / destruction gates | per-layer | 205,700 at L=4 |

Untying every tied group per layer at L=4 costs **2.76× the parameters**.

**Why untying V_θ is not a bug fix.** The framework's claim is that inference
is the integration of **one** potential over L steps: dt and L are integrator
settings, not capacity. Give each layer its own V_θ and the model is no longer
one dynamical system integrated L times — it is L different systems stacked,
which is a transformer with unusual blocks. The Jacobi metric, the geodesic
results of §27, and the "structured memory natively, not as a retrofit"
argument all rest on the potential being shared. **Untying it would trade the
thesis for the perplexity.**

**The legitimate, cheap part of the space.** The architecture already has a
per-layer conditioning channel, and it is extraordinarily narrow:
`depth_code`, shape `[L, n_ctx, d]`, **6,144 parameters at L=4** — 0.013% of
the model, and the *only* thing that distinguishes layer 3 from layer 1 inside
the shared potential. We measured it clipped at 0.25 and dominating the
gradient landscape through warmup. Two interventions preserve the thesis
entirely:

- **C-clip: loosen `depth_code`'s clip** (0.25 → 1.0). **Zero** new
  parameters. Tests whether the existing channel is throttled rather than
  too small.
- **C-width: widen the depth conditioning.** A few thousand parameters,
  still one potential, still depth-modulated.

**A standing warning against "just add capacity".** This architecture has
already been shown to get *worse* with more parameters: the conservative-twin
arm added **589,825** parameters to the L=2 model and settled at **80.90
against 66.98**, i.e. **20.8% worse**. Capacity is not a free axis here.

**The experiment that should come first changes no parameters at all.**
**L=4 at fixed dt = 4** (so T grows to 16 instead of being held at 8). That
holds the operating point and varies only depth. If depth pays once dt is
held, there is nothing for parameters to fix and the matched-T protocol was
the confound. If it still loses at fixed dt, depth genuinely does not pay in
this architecture at this budget — and only then is the capacity question
live.

**Order:** L=4 at fixed dt → C-clip (free) → C-width → and only if all three
fail, the untying question, which should be framed as a new architecture with
its own name rather than as a repair of this one.

**Do not read the early steps.** L=4's first eval came in at 495.47 against
L=2's 478.33, which looks like a 3.6% deficit and is not one: the curve is
falling **46% per 500-step eval** there, so 495.47 sits **28 steps** behind
L=2's own trajectory — under two minutes of equivalent training, measured at
1.5% of the run with the learning rate still in warmup at 31% of peak.
Nothing about depth or timestep is visible yet. The first eval where local
movement is small enough for a gap to mean anything is around **step 15,000**
(−0.5% per eval); before the decay phase any ordering is noise. The
pre-registration is about the settled value, and only the settled value
scores it.

**Depth is nearly free in parameters — recorded at launch, 2026-09-27.**
Cell 5 built L=4 at **76,823,511** parameters against L=2's **76,770,256**:
a difference of **53,255**, or **0.069%**, about 26,600 per added layer. The
architecture is almost entirely weight-tied across depth — `reverse_ch` is one
module reused at every layer, and only `depth_code`, the creation/destruction
gates and the per-layer scalars grow with L. For comparison the exchange field
costs 589,825 parameters, eleven times more than two extra layers.

Two consequences, both good for this rung:

1. **Capacity is not a confound.** Whatever L=4 settles at, it cannot be
   attributed to a bigger model. That is a stronger control than the ladder's
   other rungs enjoy, where the mechanism under test does change the count.
2. **It sharpens the dt caveat below.** With capacity held flat, the two
   candidate explanations for any change are depth and the timestep, and only
   those two. The clip-hit rate is what separates them.

Batch resolved to 16 × accum 2 = 32, identical to every other arm, so the
token budget is again exactly 532,480,000.

**Caveat to state when it lands:** at fixed `LADDER_T = 8`, L=4 means
dt = 2, so ω·Δt halves. That is a different operating point, not only more
layers — and Gate 3 shows the model is fitted to its dt. Some of what L=4
shows will be dt, not depth.

**B. `pair_potential='none'` — write while A trains**

- [ ] Add the branch to `model_parf_multixi.py`: null `V_phi` **and**
      `score_head`
- [ ] Guard the call sites in `_pair_potential` — the gathered path, the
      dense path, and the checkpointed path — in **both**
      `model_parf_multixi.py` and `model_parf_sparse.py`
- [ ] Decide what `_add_relax_potential` returns when there is no `U_pair`
      to add to (the `'attention_potential'` arm depends on this)
- [ ] Tests: forward/backward clean; `V_phi` and `score_head` absent from
      `state_dict`; no gradient path to them; parameter count drops by the
      expected amount; **existing arms bit-identical** — the regression
      that matters, since this is shared code
- [ ] Expose `PAIR_POTENTIAL` in Cell 0 and thread it through
      `make_config`
- [ ] **Add it to the variant tag** — the fourth time this has been needed
      (`norc`, `ris`, `zro`); an untagged arm shares a Drive folder and
      silently resumes from the wrong checkpoint
- [ ] Cell 5b guard + NOT-A-LADDER-POINT banner, matching the existing three

**C. Runs 10 and 11 — back to back after A**

- [ ] Run 10: multi-ξ SPLM — `pair_potential='none'`,
      `REVERSE_CHANNEL = False`, `LADDER_MECHANISM = 'none'`, L=2
- [ ] Run 11: Fock-SPLM — as run 10 but `REVERSE_CHANNEL = True`
- [ ] Pre-registrations already recorded in §3.2: each **within 5%** of its
      Vφ-carrying sibling (87.93 and 66.98 respectively)
- [ ] Report the **2×2 interaction** explicitly, not just the four cells
- [ ] Report the **Fock-price replication**: run 11 vs run 10 against
      run 3 vs run 8's +31.3%. Agreement makes the abstract's number
      robust; disagreement is the finding
- [ ] Note in the cards that these arms are **not parameter-matched** —
      Vφ's four heads and the score head leave

**D. Deferred**

- [ ] Run 6 (`'nonconservative'`, λ pinned) — a function-class bound;
      unblocks nothing. Cell 0's tag dict now accepts it (`noncons`)
- [ ] F1 on the `'none'` and conservative arms — minutes each, and
      **confirmatory rather than decisive**: at RMS 0.0003 a single token
      at R = 0.3 among 16,384 would give ≥ 0.0023, so the average already
      constrains every token. Worth running for the distribution shape

---

### 3.1 Why L=4 rather than L=8

L=2 reached 68.33 where the warm-started L=8 reached 81.58, which looks like
depth hurting. It is not evidence of that: the two differ in schedule,
Alternative E, the exchange field **and** depth. L=4 at matched everything is
the only cheap way to separate "hops do not help in this architecture" from
"the L=8 arm was handicapped by its schedule". At 27h it is half of L=8 and
answers the same question.

### 3.2 Runs 5 and 6 are parameter-matched by construction

`XiRoutedConservativeAttention` and `DirectExchangeForce` both carry exactly
**589,824** parameters — four 384x384 matrices each. So arm N minus arm C is
the price of conservativity at matched parameters, matched routing shape and
matched budget, with none of §7.3's bias.

---

## 3b. The D-series — why depth does not pay, and what to do about it

Opened **2026-09-28**, after L=4 at matched T came in behind L=2 **on the
training objective as well as on validation**. Ordered so that the cheapest
experiment that could make the others unnecessary runs first.

### The constraint that shapes all of it

$T = L \cdot dt$. You cannot hold depth, timestep and integration time fixed
at once — varying $L$ forces a choice of which of $dt$ or $T$ moves with it.
The ladder so far has held **T = 8** and let dt fall. That is one arm of the
fork and it has now produced a non-monotonic result:

| arm | L | dt | T | settled |
| --- | ---: | ---: | ---: | ---: |
| run 7 | 1 | 8 | 8 | 87.09 |
| **run 3** | **2** | **4** | **8** | **66.98** |
| run 4 | 4 | 2 | 8 | **71.75** (settled 2026-09-29) |

**At fixed T the optimum is interior.** Depth 2 with dt = 4 beats both its
neighbours, by 23% over L=1 and by **7.1%** over L=4. (Written 2026-09-28 as
"an apparent ~28%" from a stable-phase projection of 85–90; corrected when run
4 settled.) Nothing in the framework predicted an interior optimum, and it is
the single most interesting thing the ladder has produced.

### D1 — L=4 at **fixed dt**, letting T grow (one full run, ~27 h)

The other arm of the fork. Holds the operating point that worked and asks
whether depth itself pays.

- **Cell 0:** `LADDER_L = 4`, **`LADDER_T = 16.0`** (so `LADDER_DT` derives to
  4.0). Everything else exactly as run 4.
- **Tag check:** derives to `...cgqk_L4probe_ob_..._idt4_lr0p0012_noattn` —
  `L4probe` + `idt4`, distinct from both run 3 (`L2probe`+`idt4`) and run 4
  (`L4probe`+`idt2`). No collision, trains from scratch.
- **What it confounds, stated up front:** depth *and* integration time move
  together. That is unavoidable, and it is the complement of run 4's confound
  rather than a defect.

**Pre-registered, recorded before launch: settles in 62–70.**
Reasoning: dt = 4 is the operating point that produced the ladder's best Fock
result, and D1 keeps it while adding depth and integration time. The band is
centred slightly *better* than run 3's 66.98 but deliberately spans it,
because the plausible failure is that **T = 8 already saturates the dynamics**
and the extra integration time buys nothing.
**Named turnable quantity:** whether T = 8 is saturated. The tell is D1's
train loss against run 3's — if D1 fits *better* but generalises the same, the
extra time is being spent on the training distribution.
**If D1 lands near run 4's value** — 71.75 settled; this read "near 85–90"
until run 4 settled on 2026-09-29 — depth is the problem, not the timestep,
and the capacity question below becomes live. With run 4 at 71.75, this
branch and the top of D1's 62–70 band are under 2 PPL apart, so D1 has to be
read against eval noise rather than by eye.
**If D1 beats 66.98 materially**, the matched-T protocol was the confound, and
the headline ladder should be re-stated at fixed dt.

### D2 — loosen the `depth_code` clip (FREE in parameters; one run)

`depth_code` is the **only** per-layer channel inside the shared potential:
`[L, n_ctx, d]` = 6,144 parameters at L=4, 0.013% of the model. It is clipped
at **0.25**, the second-tightest override in the table, and it dominates the
gradient landscape through warmup.

- **Change:** `GRAD_CLIP_OVERRIDES['depth_code'] = 1.0` (the global default).
- **⚠ TAG TRAP — must be fixed before this runs.** `GRAD_CLIP_OVERRIDES`
  lives in **Cell 6**, not Cell 0, and **does not reach `_variant_tag`**. As
  written, D2 resolves to the *same Drive folder as D1/run 4* and Cell 2 would
  silently resume from its checkpoint. Add a tag component first, on the
  `ris`/`zro` precedent:

      if GRAD_CLIP_OVERRIDES.get('depth_code', 0.25) != 0.25:
          _variant_parts.append(f"dcclip{...:g}".replace('.', 'p'))

  and a Cell 5b guard asserting the tag carries it. **This is the same class
  of bug the `norc`, `ris` and `zro` components exist to prevent, and it is
  currently live for every clip threshold.**
- **Pre-registered:** if the clip is binding, the pre-clip norm falls below
  threshold within ~2,000 steps and settled PPL improves by >2%. If nothing
  moves, the channel is too *small*, not too throttled → D3.

### D3 — widen the depth conditioning (a few thousand parameters)

Only if D2 is null. Give the depth code more capacity, or let it modulate more
of V_θ, while keeping **one** potential. Preserves the thesis; changes
parameter shapes, so it reaches the tag automatically.

### D4 — untying (CONTINGENT, and it is a new architecture, not a repair)

Only if D1–D3 all fail. Untying every tied group at L=4 costs **2.76×** the
parameters, and **~59% of the non-embedding model is V_θ alone**. Untying
V_θ means each layer has its own potential, i.e. L stacked dynamical systems
rather than one integrated L times — which forfeits the Jacobi metric, §27's
geodesic results, and the "native memory, not a retrofit" argument. **If it is
ever run, it gets its own name and its own claims; it does not enter this
ladder.**

### Standing caution

More parameters have already made this architecture *worse* once: the
conservative twin added **589,825** parameters and settled at **80.90 against
66.98**, 20.8% worse. Capacity is not a free axis here.

### Order, cost, and the decision taken **2026-09-28**

| step | cost | |
| --- | --- | --- |
| **D1** | ~27 h | the only rung authorised now |
| D2 | ~27 h, 0 new params | **parked** behind runs 10 & 11 |
| D3 | ~27 h, few k params | parked |
| D4 | new architecture | contingent, and renamed if ever run |

**Decision: run D1, then return to the factorial unless D1 comes back
positive.** Each D rung is a full 27 h run and competes directly with runs 10
and 11, which would close the V_φ × Fock 2×2 that is already half-measured
and fully pre-registered. Completing a factorial beats chasing a repair for a
rung that may simply be a true negative.

**Three branches, not two.**

| D1 settles | reading | next |
| --- | --- | --- |
| **materially below 66.98** | matched-T *was* the confound; depth pays once dt is held | **stop and re-state the headline ladder at fixed dt.** Bigger than D2/D3 and it comes first |
| **≈ 66.98 (null)** | depth does not pay even at a held operating point | **runs 10 & 11.** The interior optimum at fixed T stands as the result |
| **≈ run 4's 71.75 (negative)** — read "≈ 85–90" until run 4 settled, 2026-09-29 | depth actively hurts at this budget | **runs 10 & 11.** Strengthens the interior-optimum finding |

So D2 and D3 are reached only through the *positive* branch, and even then
only after the ladder is re-stated. The prior from run 4 is that the null or
negative branch is the likely one.

**The clip-threshold tag defect drops in priority with D2, but does not go
away.** It blocks any run that varies a threshold, which includes C4 and C5
of the clip series. Fix it before either, not before D1 — D1 changes no
thresholds.

### What to run in the gaps, since none of it needs a 27 h slot

Ordered by when the GPU is free:

1. **The ω·dt offline comparison — immediately when run 4 finishes, before
   D1 starts.** It needs the run-4 endpoint checkpoint and a short GPU
   window, and it scores the pre-registered band [3.3, 4.2]. Do not let D1
   occupy the machine first.

   **Cell 6b-13 is written and in the notebook** (inserted after 6b-12,
   2026-09-28). Two things about it worth knowing before running:

   - **Cell 6b is NOT this measurement.** 6b forces `integrator='baoab_cfc'`
     and records `k_diag` from `harmonic_terms` — the *diagonal* curvature,
     i.e. the proxy the Verlet-instability audit found "stays under 1
     throughout and would have reported no problem at all". 6b-13 runs the
     low-rank monitor, the same one the in-flight `[resonance]` line reports.
   - **Run order is 0 → 1 → 1b → 2 → 3 → 4 → 5 → 6b-13. Do NOT run Cell 6.**
     On a completed arm Cell 6 resumes training from `_best.pt` and can
     overwrite a published checkpoint. 6b-13 therefore imports the monitor
     itself if Cell 6 has not, and writes to its own
     `results/omega_dt_endpoint.jsonl` rather than the training log.

   It asserts the clone is at 409175b or later before measuring, so a stale
   checkout fails loudly instead of returning an empty summary. Batches are
   drawn with a fixed seed that does not depend on L, so both arms see the
   same tokens. It scores the band automatically when `model.cfg.L == 2`.

   **Two sessions, one per arm**, because L is baked into the built model:
   the L=4 session that is live now, and a fresh L=2-configured session.
2. **C1** (extend the E5 slider above λ = 1) — one tuple edit, minutes.
3. **C2** and **C7** — no GPU at all; they read optimizer state and existing
   logs. Can run while D1 trains.

That ordering costs nothing and closes three pre-registered questions during
time the machine is busy or idle anyway.

---

## 4. The token-budget correction

| | steps | batch | tokens |
| --- | ---: | ---: | ---: |
| Fock, any ladder point | 32,500 | 32 throughout | **532.5M** |
| GPT-2 as published | 32,500 | **8 then 32** | **360.4M** |
| GPT-2, matched rerun | 32,500 | 32 throughout | **532.5M** |

Consequences for figures already in circulation:

- GPT-2 crossed Fock's settled endpoint at **14.3%** of Fock's token budget,
  not the 46.7% recorded in checklist §9.2. That figure was a step fraction.
- Fock used **1.48x** the tokens to reach 81.58 and still lost. Correcting
  the budget makes the architecture comparison **worse** for Fock.
- `training_log.jsonl` cannot detect this: it computes `tokens` as
  `step * 16,384` regardless of the batch in use. Only the console banner
  records the truth, which is why the logs are archived alongside the jsonl.

The notebook now **raises** on a batch mismatch rather than printing a
warning, with an explicit `ALLOW_BATCH_MISMATCH` escape hatch, and prints a
token-budget line on every run.

**Before starting run 1:** move the existing GPT-2 Drive folder aside rather
than deleting it. It is the provenance for the 54.59 currently quoted in the
paper, and a fresh run must not resume its step-32,500 checkpoint.

---

## 5. Results

### 5.1 L=2, `'attention'` — **DONE 2026-09-20**

Log: [`results/.../L2_idt4_attn_altE_fromscratch_32500_result.txt`](../notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_attn/L2_idt4_attn_altE_fromscratch_32500_result.txt)

| | value |
| --- | ---: |
| final (step 32,500) | 68.44 |
| best (step 31,000) | **66.03** |
| **settled** (mean of last three) | **68.33** |

Against the references: 0.838x the warm-started L=8's 81.58, 0.856x
Alternative E's 79.80, and **1.252x** GPT-2's 54.67 settled — down from
1.49x, the largest narrowing the programme has produced.

**Prediction scored: MISS.** Recorded 65, band 63-68; actual 68.33, error
+3.33 and outside the band. The run flattened at 66.03 then **reversed** —
68.23, 68.31, 68.44 — while the lr was still falling. The saturating read
(64.6) beat the log-linear one (59.9), consistent with §9's finding that
log-linear extrapolation is refuted for this programme, but it was still
optimistic. **Saturating is better, not conservative.**

Health: `share_max` peaked at 16.71 near step 800 and fell monotonically to
~3.3; `bproj_sig` reached 26.07, above the L=8 trained ~24.9.

**Attribution: none.** Four things differ from the 81.58 reference —
schedule, Alternative E, the exchange field, depth 8 to 2 — so no part of the
13.25-point gain is assignable from this run alone.

### 5.2 L=2, `'none'` — **DONE 2026-09-21**

Log: [`results/.../L2_idt4_noattn_altE_fromscratch_32500_result.txt`](../notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_noattn/L2_idt4_noattn_altE_fromscratch_32500_result.txt)

Single-variable contrast against §5.1: **589,824 fewer parameters**, four
fewer tensors, nothing else changed, paired on identical batches.

| | `'attention'` | `'none'` | delta |
| --- | ---: | ---: | ---: |
| final (32,500) | 68.44 | 75.48 | +7.04 |
| best | 66.03 (31,000) | **73.20** (31,000) | +7.17 |
| **settled** | **68.33** | **75.09** | **+6.76** |

**+9.9%, +0.0943 nats.**

#### The gap saturated; it did not widen

| step | 5,000 | 8,000 | 11,000 | 20,000-32,500 |
| --- | ---: | ---: | ---: | ---: |
| gap | 3.7% | 7.1% | 10.0% | **8.9-10.9%, mean 9.8%** |

It reached ~10% by step 11,000 and then held flat for the remaining 21,500
steps, through the entire decay.

**Prediction scored: MISS.** The band recorded in §5.3 was 20-30%, from a
linear fit to the nats gap over steps 4,000-11,000 which projected +19% to
+40% depending on the window. The gap was **saturating**, not linear, and the
later windows I weighted most heavily were the steepest part of an S-curve.

This is the third miss in a row in this programme, and all three share one
shape: **a trend was extrapolated through its growth phase and the quantity
saturated.** §9 of the checklist already recorded log-linear PPL
extrapolation as refuted. The generalisation is broader — *any* linear
extrapolation of a monotone quantity here has overshot, including the
saturating-read correction in §5.1, which was closer but still optimistic.
The working rule should now be: fit a saturating form, then treat even that
as an upper bound on the improvement.

#### The attribution, which is the point of the run

> **SUSPENSION LIFTED 2026-09-26 — both arms are now tuned.** This
> subsection was suspended on 2026-09-22 because it compared two arms at
> 3e-04 while only `'none'` had been retuned to 1.2e-03, at which point
> `'none'` (66.98) *beat* the untuned `'attention'` (68.33) and the credited
> attribution pointed the wrong way. `'attention'` has now run at 1.2e-03
> (§5.7): **63.51**.
>
> | comparison | exchange field is worth |
> | --- | --- |
> | both arms @ 3e-04 | 75.09 vs 68.33 — **+9.9%** |
> | `'none'` tuned only (the suspension) | ~~`'none'` better by 2.0%~~ — an artefact |
> | **both arms @ 1.2e-03** | **66.98 vs 63.51 — +5.2%** |
>
> **The corrected figure is +5.2%, not the +9.9% this subsection credits
> below.** Retuning nearly halved the exchange field's measured
> contribution: `'none'` gained 10.8% from the LR change and `'attention'`
> only 7.1%, so most of what the 3e-04 comparison attributed to the
> mechanism was really the untuned arm's headroom. Every number in the rest
> of this subsection is a 3e-04 number and must be labelled as such.


| component | PPL | share |
| --- | ---: | ---: |
| exchange field | **6.76** | 51% |
| schedule + Alternative E + depth 8 to 2 | **6.49** | 49% |
| **total gain over the 81.58 reference** | **13.25** | |

The 13.25-point gain splits almost exactly in half. **This is the first real
attribution the programme has produced** — every previous number was
confounded four ways.

The 6.49 remains confounded three ways and needs runs 4 and a schedule
control to split further. The 6.76 is clean.

#### What the conservative number is worth on its own

75.09 is a **fully conservative** model, kappa identically zero, and it is
the best conservative result in the programme:

| conservative arm | settled |
| --- | ---: |
| L=8 warm-start | 81.58 |
| L=8 warm-start + Alternative E | 79.80 |
| **L=2 from scratch + Alternative E** | **75.09** |

At L=2 the velocity is still an independent state variable, so
`Omega^2 = 2 T m` does not collapse and the Jacobi metric, its geodesics and
parallel transport all survive — see
[`Composing_Single_Layer_Inferences_Flow_or_Maps.md`](Composing_Single_Layer_Inferences_Flow_or_Maps.md)
§3.1 for why L=1 would not. So the geometric capabilities of paper section
18d are available on this checkpoint, at roughly a third of the L=8
inference cost.

Against GPT-2 the ratio is **1.373** (75.09 vs 54.67 settled), and that
comparison still carries the token confound of §4 until run 1 lands.

### 5.3 Pre-registered bands, recorded before the runs

**The expected ladder at the tuned LR (1.2e-03), stated as an ordering
before any of the remaining arms runs:**

| arm | status | settled PPL |
| --- | --- | ---: |
| matched GPT-2 L=8 | measured | **49.81** |
| L=2 `'attention'` | run 9, **DONE** | **63.51** |
| L=2 `'attention_potential'` | run 5, **DONE** | **80.90** |
| L=2 `'none'` | measured | **66.98** |
| L=2 `'none'`, reverse channel off | run 8, **DONE** | **87.93** (pre-reg 105, band 85–140: band hit) |

The ordering itself is the prediction: each arm removes one mechanism from
the one above it, and the gaps price them. Any inversion is a result.

| run | prediction | reasoning |
| --- | --- | --- |
| L=2 `'none'` | ~~gap holds at 20-30%~~ **MISS: actual 9.9%** | see §5.2; the gap saturated rather than growing |
| L=2 `'attention_potential'` | **75-82, point 78** | arm C detaches both alpha and `h_src`, so its Jacobian is block-diagonal and there is no inter-token coupling in the dynamics. Below 72 would be a genuine surprise. |
| L=4 | no strong prior | this is the point of running it |
| L=2 `'attention'` **@1.2e-03** (run 9) | **63, band 59–68 — HIT, actual 63.51, error +0.51 (+0.8%)** | recorded 2026-09-25. At 3e-04 `'attention'` led `'none'` 68.33 to 75.09 (9.0%); retuning `'none'` to 1.2e-03 bought 10.8%. A like-for-like gain puts `'attention'` near 61, but its optimum may sit lower than `'none'`'s (quadratic vertex 1.13e-03) because it carries more parameters — so the band is widened upward. **The quantity that could turn:** whether 1.2e-03 is already past `'attention'`'s own optimum, as 2.4e-03 was past `'none'`'s. If it is, the gain shrinks or reverses and the result lands above 68 |
| L=2 `'attention_potential'` **@1.2e-03** (run 5) | ~~70, band 65–77~~ **66, band 62–72 — MISS, actual 80.90, error +14.90 (+22.6%)**; the *superseded* 75–82 band would have hit, see §5.3a | recorded 2026-09-25, **revised the same day after re-reading the code** — see §5.3a. The 3e-04 band below (75–82, point 78) stands as recorded. |
| L=1 `'none'` @1.2e-03 | ~~74-80~~ **MISS: actual 87.09 settled** | forecast made in conversation from the L=2 curve shape; see §5.5 — the L=1/L=2 gap did not saturate, it kept widening through the decay |
| matched GPT-2 | ~~below 54.59~~ **HIT: 49.76 final, 49.81 settled** | predicted 49.5 band 49.0-50.0 from the published run's behaviour over the same lr range; error +0.26 |

### 5.3a Correcting the `attention_potential` reasoning — **2026-09-25**

The band recorded in §5.3 for arm C rested on the sentence *"arm C detaches
both alpha and `h_src`, so its Jacobian is block-diagonal and there is no
inter-token coupling in the dynamics."* The first half is right and the
second half is wrong, and the difference changes which side of `'none'`
the arm should be expected to land.

What the detaches actually do, from
[`model_parf_multixi.py`](../notebooks/conservative_arch/parf/model_parf_multixi.py)
`_add_relax_potential` and
[`model_xi_attention.py`](../notebooks/conservative_arch/parf/model_xi_attention.py):

```python
h_src  = h_in.detach() if self.cfg.causal_force else h_in
_route = h_in.detach()          # route_from='h'
add    = self.relax_field.potential(h_in, h_src, _route, causal_mask)
```

- The routing weights and the source slice are constants with respect to
  $h_t$. **That is what makes the force a gradient** — the point of the
  arm.
- The Jacobian is therefore block-diagonal: $\partial F_t / \partial h_s = 0$
  for $s \neq t$. **True.**
- But the potential still *reads* $h_s$ numerically. Context flows
  **forward**: the force on token $t$ depends on what the other tokens
  are. What is missing is the **backward** path — no gradient reaches the
  source through this term, so the mechanism can *use* source
  representations but cannot *shape* them.

Two further facts the earlier reasoning did not weigh:

- The term **adds to** $V_\phi$, it does not replace it (the code comment
  is explicit: `pair_potential='xi_attention'` would discard a trained
  component, and arm C must not). So this arm is `'none'` **plus** a
  second conservative context mechanism — strictly more capacity.
- `RELAX_LAMBDA_FIXED = 1.0` **pins** λ. The model cannot turn the term
  down if it is a poor inductive bias.

Those two point in opposite directions, which is why the revised band
straddles `'none'`'s 66.98 rather than sitting cleanly on one side:

**Revised pre-registration: 66, band 62–72.** *The quantity that decides:*
whether the pinned-at-1.0 conservative attention term is extra capacity or
a forced burden. If it helps, this lands just below `'none'` and the
expected ordering holds; if the pin makes it fight $V_\phi$, it lands
above 66.98 and the original §5.3 intuition is vindicated for a reason
nobody wrote down.

This is a revision of a pre-registered band *before* the run, on the
grounds that the stated mechanism was factually wrong — not a re-basing
after seeing a result. Both bands stay on the record.

### 5.7 L=2, `'attention'` @1.2e-03 — **DONE 2026-09-26** (run 9)

Log: [`results/.../L2_idt4_lr0p0012_attn_altE_fromscratch_32500_result.txt`](../notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_attn/L2_idt4_lr0p0012_attn_altE_fromscratch_32500_result.txt)

The re-run that repairs §5.2. Identical to run 2 except the learning rate —
77,360,081 parameters both times — so LR is the only variable and the
3e-04 to 1.2e-03 transfer is measured, not inferred.

| | @3e-04 (run 2) | @1.2e-03 (run 9) | gain |
| --- | ---: | ---: | ---: |
| final (32,500) | 68.44 | 63.42 | |
| best | 66.03 (31,000) | **61.49** (31,000) | |
| **settled** (last 3) | **68.33** | **63.51** | **+7.1%** |

Ratio to the matched GPT-2 (49.81): **1.275**, the closest any arm in this
programme has come.

#### What it settles: the exchange field is worth +5.2%, not +9.9%

| both arms at | `'none'` | `'attention'` | exchange field |
| --- | ---: | ---: | ---: |
| 3e-04 | 75.09 | 68.33 | +9.9% |
| **1.2e-03** | **66.98** | **63.51** | **+5.2%** |

Retuning nearly halved it. `'none'` gained 10.8% from the LR change,
`'attention'` 7.1%, so roughly half of what the 3e-04 comparison credited
to the mechanism was the untuned arm's headroom. This is the third time in
the programme that a contribution measured at an untuned operating point
shrank once both sides were tuned, and it belongs with the ablation lesson
of §5.6: **a gap measured off-optimum is an upper bound.**

**Prediction scored: HIT — the first point-accurate one.** Pre-registered
63, band 59–68; actual 63.51, error +0.51 (+0.8%). The reasoning that
earned it is worth keeping: the forecast was built from the *transfer*
`'none'` had already shown (10.8%) discounted for the extra parameters
`'attention'` carries, with "is 1.2e-03 past this arm's own optimum?"
named as the quantity that could turn it. It was — partially. The arm
gained 7.1% rather than 10.8%, which is exactly the discount the band was
widened for.

#### The clip rate is the caveat, and it is large

**29.2%** of logged steps hit the gradient clip (190 of 650, max norm
1.42), against **7.5%** for the same arm at 3e-04 and **0.0%** for
`'none'` at this LR. That is close to the 36% L=8 regime §6 warns about.

Two things follow. First, the two arms being compared are not equally well
optimised at 1.2e-03: `'attention'` is training under heavy clipping and
`'none'` under none, so **+5.2% is a lower bound on the exchange field** —
at its own optimal LR, which the 7.1% transfer and the clip rate both
suggest is below 1.2e-03, this arm would likely do better. Second, the
ladder's discipline is one LR for all arms, so 63.51 is the correct ladder
number; the caveat is recorded rather than corrected for.

#### The gate confound, eliminated — **probe run 2026-09-27**

This arm runs with `RELAX_GATE = 'scalar'` and `RELAX_LAMBDA_FIXED = 1.0`,
which the model config's own docstring marks **superseded**: *"lambda can
scale that field but not orient it, and a random direction in d dimensions
overlaps the useful one by only about 1/sqrt(d) with arbitrary per-batch
sign."* Measured at initialisation, the exchange force is **0.616** of the
entire conservative force here — the term is live at full random strength
from step 0. `'zero_readout'` instead zeroes the output projection, so the
term starts at exactly zero and grows through a gradient that can
*orient* it, not merely scale it. Since this is the ladder's best arm, a
gain would move both the exchange-field attribution and the conservativity
price.

A 3,000-step probe, `RELAX_GATE = 'zero_readout'`, everything else
identical, against this arm's own evals:

| step | zero_readout | scalar (this arm) | delta |
| ---: | ---: | ---: | ---: |
| 500 | **467.17** | 481.47 | **−14.30** |
| 1,000 | 252.12 | 248.81 | +3.31 |
| 2,000 | 161.62 | 158.76 | +2.86 |
| **3,000** | **133.37** | **131.20** | **+2.17** |

**Null: +1.65% at step 3,000**, inside the ±5% band, and marginally the
wrong way.

**The share trace says why, and it is the more interesting half.** The
zeroed readout does not stall — it climbs from 0.569 at step 50 to
**12.58 by step 350**, reaching the scalar arm's operating point within a
few hundred steps, and the two track each other from there (12.77 against
13.75 at step 2,750). **The gate changes how the term starts, not where it
ends up.**

That also explains the one real difference: at step 500 the zero-readout
arm is **3.0% ahead**. Starting a random-direction field at full strength
*is* a genuine early handicap, exactly as the docstring argues. The
advantage simply does not survive the term finding its own scale, and by
step 1,000 it has reversed.

**Taken with the init-scale probe of §5.8, both initialisation confounds
are eliminated.** Neither a 2.75x larger starting magnitude on the
conservative arm nor a zero start with an orientable readout on this one
moves the curve outside noise. The exchange field converges to its
operating point regardless of how it is initialised, so the differences
between these arms are about what each term **can express**, not how it
begins. The +5.2% exchange-field attribution and the +27.4% conservativity
price both stand.

Log: [`results/.../L2_idt4_lr0p0012_attn_zro_probe3000_result.txt`](../notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_zro_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_attn/L2_idt4_lr0p0012_attn_zro_probe3000_result.txt)

**Caveat.** 3,000 steps is early, and this arm's clip rate reaches 29.2%
only over the full run (5.0% in the probe window). A full-length
`'zero_readout'` run could still differ. The null is recorded as *no
evidence of a gate effect at the point where half the eventual
conservativity gap has already opened*, not as proof of none.

#### Run health

0 watchdog triggers, 0 spike captures. `share_max` — the exchange field's
share of the force — peaked at 15.68 early and fell monotonically to 3.93,
the same shape as the 3e-04 run (16.71 to ~3.3), so the mechanism is not
running away at the higher LR. `bproj_sig` saturated at 84.25 (85.4 for
`'none'`). Decay gain 17.1% (76.62 at 21,500 to 63.51), against `'none'`'s
19.3%.

### 5.8 L=2, `'attention_potential'` @1.2e-03 — **DONE 2026-09-27** (run 5)

Log: [`results/.../L2_idt4_lr0p0012_attnpot_altE_fromscratch_32500_result.txt`](../notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_attnpot/L2_idt4_lr0p0012_attnpot_altE_fromscratch_32500_result.txt)

The conservative twin of run 9, **parameter-matched to it exactly**
(77,360,081 both): same routing source (`h`), same 8 heads, same
`d_k = 48`, same λ pinned at 1.0. The single difference is whether the
xi-routed exchange field enters as a **force** or as the **gradient of a
scalar potential**.

| | final | best | **settled** | vs GPT-2 |
| --- | ---: | ---: | ---: | ---: |
| `'attention'` (run 9) | 63.42 | 61.49 | **63.51** | 1.275 |
| **`'attention_potential'`** | 81.50 | 79.14 (31,000) | **80.90** | **1.624** |

#### The price of conservativity is +27.4%, and it inverts the mechanism's sign

| comparison | delta |
| --- | ---: |
| `'attention'` → `'attention_potential'` | **+17.39 PPL, +27.4%** — the price of conservativity |
| `'none'` → `'attention'` | −3.47 PPL, **−5.2%** — a non-conservative exchange field *helps* |
| `'none'` → `'attention_potential'` | **+13.92 PPL, +20.8%** — a conservative one *hurts* |

**The conservative exchange field is worse than having no exchange field at
all.** Adding it to `'none'` costs 20.8%; adding the non-conservative twin
of the same size gains 5.2%. Conservativity here is not a tax on a
mechanism — it reverses the mechanism's sign at matched capacity.

The predicted ordering (matched GPT-2 ≥ attention ≥ attention_potential >
none > no-RC) is therefore **wrong in one place**: `attention_potential`
does not sit between `attention` and `none`, it sits between `none` and
the arm with no Fock mechanism at all.

| arm | settled | vs GPT-2 |
| --- | ---: | ---: |
| matched GPT-2 | 49.81 | 1.000 |
| L=2 `'attention'` | 63.51 | 1.275 |
| L=2 `'none'` | 66.98 | 1.345 |
| **L=2 `'attention_potential'`** | **80.90** | **1.624** |
| L=2 `'none'`, no reverse channel | 87.93 | 1.765 |

#### It is not an optimisation failure

**Clip-hit 0.0%** (0 of 650 logged steps, max grad-norm 0.93), against
`'attention'`'s **29.2%** at the same LR. 0 watchdog, 0 spikes.
`bproj_sig` reached 100.46, the highest of any arm. Decay gain 15.5%
(95.70 at 21,500 to 80.90), the lowest of the L=2 arms.

This arm trained smoothly, unclipped, and converged cleanly to a worse
place. It is not struggling; the ceiling is lower. **And the clip
asymmetry cuts against `'attention'`, not for it** — the better arm is the
one training under heavy clipping, so the +27.4% is if anything an
underestimate of the gap at each arm's own best LR.

#### The mechanism, from the code

> **Withdrawn 2026-09-29.** The "weld" reading below does not survive a
> second look at the code: W_uq is an output projection, and the routing
> query is the separate W_q. The two arms' forces have the same form; they
> differ in the backward pass. See "The 'welded projection' reading withdrawn"
> at the end of this section, and the two probes pre-registered there.

With α detached (which is what makes the force a gradient) and `h_src`
detached (which is what keeps it causal), the potential is **bilinear**,
so the force is

$$F_t = \sum_s \alpha(t,s) W_{uq}^{\top} W_v h_s$$

— structurally the same shape as `DirectExchangeForce`'s
$\sum_s \alpha W_V h_s$, except that taking the gradient **welds the
output projection to the query projection**. The non-conservative twin has
three free matrices (W_Q, W_K, W_V) and can choose *where* to push
independently of *what it reads*; the conservative twin cannot. That is
the expressivity conservativity costs here, and it is evidently large.

#### Prediction scored: MISS, and the revision made it worse

Pre-registered 66, band 62–72; actual **80.90**, error +14.90 (+22.6%).

The uncomfortable part: **the band this one replaced would have hit.** The
original §5.3 pre-registration was 75–82, point 78, on the reasoning that
arm C's detaches leave "no inter-token coupling in the dynamics". §5.3a
revised it to 66 (62–72) after re-reading the code, on the grounds that
the stated mechanism was factually wrong — context *does* flow forward,
only the backward path is cut — and that the term *adds* to V_φ and so is
strictly more capacity.

Both halves of that correction are still true. The conclusion drawn from
them was not. **A flawed argument can support a correct prediction, and
repairing the argument without re-deriving the magnitude can make the
prediction worse.** The revision also moved the band toward an expected
ordering and away from the answer, which is the shape of anchoring.

The rule this adds to §9's forecast discipline: *when correcting the
reasoning behind a pre-registered band, re-derive the number from the
corrected reasoning rather than adjusting the old number — and record what
the superseded band would have predicted.*

#### The initialisation confound, eliminated — **probe run 2026-09-27**

The conservative arm's exchange term starts **7x weaker** than the
non-conservative twin's: measured at init, the exchange force is 0.088 of
the conservative force for `'attention_potential'` against **0.616** for
`'attention'`. The cause is structural — the potential is bilinear, so its
induced force scales as `init_scale^2` where the direct force scales
linearly. That is an obvious candidate explanation for the gap that has
nothing to do with conservativity, so it was tested.

A 3,000-step probe at `RELAX_INIT_SCALE = 0.055` (chosen to put the term's
starting magnitude at ~0.62, matching `'attention'`), everything else
identical, against the completed run's own evals:

| step | probe @0.055 | arm @0.02 | delta | `'attention'` |
| ---: | ---: | ---: | ---: | ---: |
| 500 | 475.66 | 477.72 | −2.06 | 481.47 |
| 1,000 | 257.44 | 258.18 | −0.74 | 248.81 |
| 2,000 | 174.28 | 173.22 | +1.06 | 158.76 |
| **3,000** | **149.84** | **148.99** | **+0.85** | **131.20** |

**Null, and decisively so.** +0.57% at step 3,000 — well inside the ±5%
band and an order of magnitude smaller than the 13.6% gap to
`'attention'` it was meant to explain. A 2.75x larger starting magnitude
moves the curve by under 1 PPL at any point. The gap to `'attention'`
is 13.6% at 0.02 and 14.2% at 0.055: unchanged.

Log: [`results/.../L2_idt4_lr0p0012_attnpot_ris0p055_probe3000_result.txt`](../notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_ris0p055_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_attnpot/L2_idt4_lr0p0012_attnpot_ris0p055_probe3000_result.txt)

**The result was predicted, by the code and by three measurements.** The
model config's docstring states it outright: *"Raising `relax_init_scale`
does not help: it scales signal and noise together, and Adam is
scale-invariant in the gradient."* Three independent observations agreed
before the probe ran — the two arms are **tied at step 500** (477.72
against 481.47) and diverge only afterwards, which is the signature of an
expressivity deficit rather than a starting-magnitude one; the term
**grew large enough by convergence to destabilise a no-LayerNorm replay**
(master doc §4.11), so it plainly did get going; and Adam normalises by
recent gradient RMS, erasing a constant scale within a few hundred steps.

**So the +27.4% is not an initialisation artefact.** What remains is the
expressivity reading: taking the gradient welds the output projection to
the query projection, so the conservative twin cannot choose where to push
independently of what it reads. That is the mechanism, and the obvious
confound is now eliminated rather than argued away.

#### One caveat that limits how strongly this can be stated

Both arms ran with `RELAX_GATE = 'scalar'` and `RELAX_LAMBDA_FIXED = 1.0`.
The model config's own docstring marks `'scalar'` **superseded**: *"lambda
can scale that field but not orient it, and a random direction in d
dimensions overlaps the useful one by only about 1/sqrt(d) with arbitrary
per-batch sign, so lambda random-walks instead of growing. Kept to
reproduce the 2026-09-19 run."* So the honest claim is **conservativity
costs 27.4% *as instantiated here***, with a random-direction field forced
on at full strength from step 0. Whether a zero-readout gate
(`RELAX_GATE = 'zero_readout'`, which is symmetric across both arms)
recovers it is the obvious follow-up, and is tracked as a separate
question rather than a ladder rung.

#### The "welded projection" reading withdrawn: the conservative arm is gradient-starved — **2026-09-29**

**The two forces have the same form.** Read side by side,
`DirectExchangeForce` (`'attention'`) and
`XiRoutedConservativeAttention.potential` (`'attention_potential'`) compute:

| | `'attention'` | `'attention_potential'` |
| --- | --- | --- |
| routing α(t,s) | softmax of (W_Q h_t)·(W_K h_s) | softmax of (W_q h_t)·(W_k h_s), `route_from='h'` |
| force on token t | W_O Σ_s α W_V h_s | Σ_s α W_uqᵀ W_v h_s / √d_v |
| matrices | W_Q, W_K, W_V, W_O | W_q, W_k, W_v, W_uq, with the same shapes |

- **There is no output-to-query weld.** W_uq is not the routing query, which
  is the separate W_q; W_uqᵀ plays exactly the role of W_O. The mechanism
  paragraph above ("taking the gradient welds the output projection to the
  query projection") misread this, and **the expressivity reading is
  withdrawn.**
- **The conservativity here is close to trivial.** With α and h_src detached,
  the potential is *linear* in h_t, so its force does not depend on h_t, and
  any such field is a gradient.

**What differs is the backward pass.** Detaching changes gradients, never
forward values:

- In `'attention'`, `relax_field(h_in)` receives the live h_in, so the loss
  trains every token's state through q, k and v. Earlier tokens learn to be
  useful values for later ones.
- In `'attention_potential'`, the routing input and h_src are detached, and
  the force is constant in h_t. **The exchange field sends no gradient into
  any hidden state.** Its own four matrices still learn, through
  `create_graph`.

**Measured offline, on both `_best.pt` checkpoints** (step 31,000), with real
validation tokens and a cotangent at the last position only:

| arm | layer | force RMS | gradient → earlier tokens | gradient → the token itself | gradient → routing W |
| --- | ---: | ---: | ---: | ---: | ---: |
| `'attention'` | 0 | 0.0070 | 0.271 | 0.047 | 0.056 |
| `'attention'` | 1 | 0.0714 | **5.28** | **4.28** | 52.1 |
| `'attention_potential'` | 0 | 0.0099 | **0** | **0** | 0.038 |
| `'attention_potential'` | 1 | 0.0323 | **0** | **0** | 10.6 |

For `'attention_potential'`, autograd finds no path from the force to h at
all. The controls rule out a trivial zero: the force is live, and the routing
weights receive gradient. Output and scripts:
[`gradient_path_check_output.txt`](../notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_attnpot/gradient_path_check_output.txt),
`scaleup/debug/gradcheck_exchange_paths.py` and
`verify_relax_grad_path.py`.

**New leading hypothesis, H_s: gradient starvation.** It fits what was
already measured. The arms are tied at step 500 and diverge afterwards, a
learning deficit rather than a starting one. Neither the init-scale nor the
gate probe moved the curve. And it would explain "worse than no field at
all": a force the representations cannot adapt to is noise the rest of the
model must work around.

This confirms the **mechanism**. It does not yet show that the mechanism
**causes** the +27.4%, which needs training.

#### Pre-registered: the two gradient-path probes — **recorded 2026-09-29, before either run**

One switch, `RELAX_GRAD_PATH` in Cell 0, passed as `relax_grad_path` to the
model. **The forward force is unchanged in every setting; only the backward
pass differs.** Verified on the checkpoints before any run:

| probe | Cell 0 | tag component | forward vs parent | gradient into earlier tokens |
| --- | --- | --- | --- | --- |
| **(a)** | `LADDER_MECHANISM='attention'`, `RELAX_GRAD_PATH='detached'` | `rgdet` | identical (Δlogit 0) | 5.28 → **0** |
| **(b)** | `LADDER_MECHANISM='attention_potential'`, `RELAX_GRAD_PATH='live'` | `rglive` | identical (Δlogit 3e-5, ΔF 1e-7) | 0 → **0.80** |

- **(a)** detaches the field's inputs, which puts `'attention'` in exactly
  the starved class `'attention_potential'` trains in.
- **(b)** writes the same conservative force out explicitly
  (`XiRoutedConservativeAttention.force_live`, Σ_s α W_uqᵀ W_v h_s / √d_v
  over live tensors), so its forward dynamics stay conservative and causal
  while the learning signal returns.
- Both runs: `PROBE_MAX_STEPS = 3_000`, everything else at the ladder
  defaults (`RELAX_GATE='scalar'`, λ pinned at 1.0, LR 1.2e-3). They are
  **not ladder points**; Cell 5b says so and asserts the tag. About 1.5 h
  each.

**What the parents recorded** (a 3,000-step window is enough, because the gap
was already 13.6% at step 3,000):

| step | `'attention'` | `'attention_potential'` |
| ---: | ---: | ---: |
| 500 | 481.47 | 477.72 |
| 1,000 | 248.81 | 258.18 |
| 2,000 | 158.76 | 173.22 |
| **3,000** | **131.20** | **148.99** |

**Predictions under H_s, scored at step 3,000** with a ±3% band: the two
earlier probes on this pair moved step 3,000 by +0.57% and +1.65%.

- **(a) `attention` + `rgdet` lands within ±3% of 148.99, i.e. in [144.5,
  153.5].** Within ±3% of 131.20, i.e. [127.3, 135.1], **refutes** H_s.
- **(b) `attention_potential` + `rglive` lands within ±3% of 131.20, i.e. in
  [127.3, 135.1].** Within ±3% of 148.99 **refutes** H_s.
- Anywhere in between reads as **partial**: starvation explains part of the
  gap, and the rest is something else.

> **Revised for probe (b) during the run — 2026-09-29, at step 1,300.** This
> revision was made after seeing (b)'s evals at step 500 (472.41) and step
> 1,000 (255.98), and before any later eval. Those two evals are
> uninformative: every arm sits within 4% of every other at both steps.
>
> **Why the band above is too strict for (b).** It asks the probe to match
> `'attention'` fully within 3,000 steps. Even if starvation is the whole
> story, (b) starts with a field about 7× weaker than `'attention'`'s (the
> potential's force carries 1/√d_v and a product of two init-scale
> matrices). Restored gradients let it grow, but need not let it catch up by
> step 3,000. The band also counts a partial explanation as a failure.
>
> **The sharper yardstick is the no-exchange arm** (`'none'`). The symptom that
> matters is that the conservative field is *worse than no field at all*:
>
> | step | `'attention'` | no-exchange | `'attention_potential'` |
> | ---: | ---: | ---: | ---: |
> | 1,000 | 248.81 | 257.74 | 258.18 |
> | 2,000 | 158.76 | 167.87 | 173.22 |
> | 3,000 | 131.20 | **140.61** | 148.99 |
>
> **Revised criterion for (b), scored at step 3,000** against no-exchange's
> 140.61, ±2.5% for noise:
>
> - **≤ 137.1: the field now helps.** Starvation explains the sign flip. This
>   is strong support for H_s, without a full match to `'attention'`.
> - **137.1–144.1: the field is neutral.** Starvation explains the harm but
>   not the benefit, so H_s is partial.
> - **≥ 144.5: refuted.** Restoring gradients does not rescue the field.
>
> The original [127.3, 135.1] stays on the record as the **full-recovery**
> criterion, a (b) that matches `'attention'`. Probe (a) keeps its
> pre-registration unchanged.

#### Probe (b) extended to a full run — pre-registered **2026-09-29, at step ~1,300, before the step-3,000 eval**

(b) continues past its 3,000-step stop to 32,500 steps in the same session:
`PROBE_MAX_STEPS = None; run_training(3000, TOTAL_STEPS)`. The WSD schedule is
a function of step and `TOTAL_STEPS` only, and the probe stop reuses the
step-3,000 eval, so the continuation is identical to an uninterrupted
from-scratch run. It becomes a **complete, comparable arm**, and its settled
PPL is what the model cards will report. The step-3,000 value is still
scored against the revised criterion above.

**Expectation recorded by the author before the step-3,000 eval:** about
136–137 at step 3,000, which would sit just inside "the field now helps".

**Prediction for the settled value.** From step 3,000 to settled, the arms
improved by these factors: no-exchange ×0.476, `'attention'` ×0.484,
`'attention_potential'` ×0.543 (the starved arm degraded relative to the
others as it trained). A step-3,000 value near 136.5 maps to about 65–66 if
(b) trains like the healthy arms, and about 74 if it trains like its parent.

- **Pre-registered: settled 63–72, point 67.**
- **≤ 66.98** (at or below no-exchange): the conservative field genuinely
  *helps* once it can train the representations. The "worse than no field"
  result was starvation.
- **≥ 76: refutes H_s at full length.** Restored gradients do not carry the
  conservative field to a useful place.
- **Between 72 and 76: partial.** The pre-registration misses on the high
  side without a clean refutation.

**What would change on the cards.** A settled value in the band supports
restating the +27.4% as "the price of detaching the exchange field's inputs"
rather than "the price of conservativity", and adding (b) to the collection
as a named arm: a conservative forward pass with a transformer-like learning
signal. Probe (a) is still needed as the mirror-image check.

#### Probe (b) scored at step 3,000: **133.74 — HIT on both criteria** — **2026-09-29**

Log:
[`results/.../L2_idt4_lr0p0012_attnpot_rglive_probe3000_result.txt`](../notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_rglive_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_attnpot/L2_idt4_lr0p0012_attnpot_rglive_probe3000_result.txt).
The run was clean: no spikes, no watchdog events, and the same parameter
count and step-50 line as its parent.

| step | `'attention'` | **(b) live** | no-exchange | `'attention_potential'` |
| ---: | ---: | ---: | ---: | ---: |
| 500 | 481.47 | 472.41 | 478.33 | 477.72 |
| 1,000 | 248.81 | 255.98 | 257.74 | 258.18 |
| 1,500 | 188.01 | **192.73** | 197.04 | 200.79 |
| 2,000 | 158.76 | **162.27** | 167.87 | 173.22 |
| 2,500 | 140.11 | **142.28** | 148.02 | 155.64 |
| **3,000** | **131.20** | **133.74** | **140.61** | **148.99** |

- **The original pre-registration, recorded before the run** (full recovery,
  [127.3, 135.1]): **HIT.**
- **The revised criterion, recorded at step 1,300** (≤ 137.1, "the field now
  helps"): **HIT.** Both agree, so the mid-run revision did not decide the
  verdict. The author's expectation of 136–137 was on the cautious side.
- **Against its parent:** restoring the gradients closes **86% of the gap** to
  `'attention'`, from 148.99 to 133.74 against 131.20. (b) is 1.9% behind
  `'attention'`.
- **Against no-exchange:** the conservative field now **helps**, 4.9% better
  than no field, where the starved version was 6.0% worse. **The sign flip is
  gone.**
- **It is not one noisy eval.** The separation from the parent grows steadily:
  4.0% at step 1,500, 6.3% at 2,000, 8.6% at 2,500, 10.2% at 3,000.

**What this establishes at 3,000 steps.** The same conservative forward force,
given back the learning signal it was denied, behaves almost exactly like
`'attention'`. The ladder's "+27.4% price of conservativity" was **mostly the
price of detaching the exchange field's inputs**, which is gradient
starvation, and not of conservativity itself.

**Not yet established.**

- **The settled value.** (b) continues to 32,500 steps under the full-run
  pre-registration (63–72, point 67).
- **Probe (a), the mirror image.** `'attention'` with detached inputs should
  fall to about 149 if starvation is the mechanism.
- **The reverse-channel confound.** The reverse channel is still on in (b), as
  in every exchange arm. That question stands, but it no longer carries the
  "worse than no field" result.

The model cards keep their current wording until the full run settles.

**Decision rules.**

| (a) | (b) | reading | next |
| --- | --- | --- | --- |
| confirms | confirms | the +27.4% is the price of **detaching**, not of conservativity | train (b) to 32,500 steps as a new, named arm; **correct the cards** and the book's conservativity claim |
| confirms | refutes | starving a field hurts, but the conservative arm has a further deficit | look for it: the per-head 1/√d_v scale, the sign convention, delivery through the potential path |
| refutes | confirms | restoring gradients helps the conservative arm, while starving attention does not hurt it | attention's own backward path is not what makes it good; re-examine the routing difference |
| refutes | refutes | gradient flow is not the explanation | the reverse-channel interference test (the arm with the reverse channel off, against `none-norc`) moves to first |

**Order: (b) first.** It is the one that could yield a better conservative
model, and a (b) confirmation alone already overturns the current reading.

---

#### Probe (b), full run: post-training probes (6b-7/9/12/13) and publication — **2026-10-02**

On `_best.pt` (step 31,000, PPL 59.09), against its starved parent on the same architecture. E1 values are comparable within this pair only; other arms carry different potentials.

| reading | parent (starved) | probe (b), `rglive` |
| --- | ---: | ---: |
| 6b-7 gate 1: PPL rise when the velocity is reset | +18.80 | **+32.66** |
| 6b-7 gate 3: refinement at fixed T | MAPS | MAPS |
| 6b-9 R(geo) | 1.274 | 1.144 |
| 6b-9 cons+LN (everything except the reverse channel) | 1.021 | 0.877 |
| 6b-9 reverse-channel effective gate | 0.024 | 0.034 |
| 6b-12 layer 1, share of tokens strongly forced (R > 0.75) | 100% | 100% (uniform) |
| 6b-13 ω·dt median (share past 2) | not run | 1.784 (44%); layer 0 1.61, layer 1 2.82 |

- **Live gradients make the model lean more on inertia,** and its step sits somewhat closer to the damped Vθ geodesic. It is still not one, and it still reads MAPS.
- 6b-13 printed MISS against [3.3, 4.2]. That band was pre-registered for the no-exchange arm and does not apply here; the reading is recorded, not scored.
- **Published 2026-10-02** as a Gen 2 probe, NOT a ladder arm: `dimitarpg13/semsimula-ladder-owt-d384-l2-attention-potential-rglive`. It is the last item of the Gen 2 collection, and the four Gen 2 banners link it where they cite 61.11.

### 5.4 Matched GPT-2 baseline — **DONE 2026-09-21**

Log: [`results/gpt2_baseline_d384_L8/matched_batch32_32500_result.txt`](../notebooks/conservative_arch/scaleup/results/gpt2_baseline_d384_L8/matched_batch32_32500_result.txt)

Batch 32 throughout, 32,500 steps, **532.5M tokens** — token-matched to every
ladder point for the first time.

| | value |
| --- | ---: |
| final (step 32,500) | **49.76** |
| settled (mean of last three) | **49.81** |

**Prediction scored: HIT**, the first in this programme. Recorded 49.5 with a
band of 49.0-50.0; actual 49.76, error +0.26. The method that worked was not
a fit: it was the **analogue** — the published run's behaviour across the
same lr range (8.4e-05 to the 6e-05 floor, a factor of 0.9726) applied to the
current value. The three log(lr) fits gave 38.87, 47.32 and 49.00 depending
on the window, i.e. the fit was less informative than the matched-schedule
comparison. Worth remembering: where an analogue with the same architecture
and schedule exists, prefer it to extrapolating the run's own curve.

#### Consequences

The extra 172M tokens are worth **4.86 PPL** to GPT-2, so every ratio widens:

| | tokens | settled | ratio |
| --- | ---: | ---: | ---: |
| **GPT-2 matched** | 532.5M | **49.81** | 1.000 |
| Fock L=2 + exchange force | 532.5M | 68.33 | **1.372** |
| Fock L=2, conservative only | 532.5M | 75.09 | **1.508** |
| Fock L=8 warm-start (reference) | 532.5M | 81.58 | 1.638 |

The **1.49x** figure in circulation for the L=8 arm was computed against the
360.4M-token baseline. Against a token-matched one it is **1.638x**. Every
Fock-versus-GPT-2 claim in the paper needs this denominator.

#### These ratios are tuned against untuned, and must be quoted that way

The comparison is now matched on **data** and on **evaluation**. It is not
matched on **hyperparameter effort**, and the asymmetry is large.

| | GPT-2 | Fock ladder |
| --- | --- | --- |
| learning rate | 6e-04, nanoGPT default | 3e-04, **inherited from the L=8 arm, never swept at this depth** |
| schedule | cosine, warmup 2,000 | WSD 5/60/35, inherited |
| lambda on the exchange field | n/a | **1.0, chosen on structural argument, not evidence** |
| heads x d&#95;k | 6 x 64, standard | 8 x 48, chosen for width parity |
| T = L*dt, gamma, clip overrides | n/a | inherited or conventional, all unswept |

GPT-2's settings are not optimal for this budget either -- nanoGPT defaults
target a different regime -- but they are **known-good across a wide range**,
whereas the Fock side has never been checked at all. One side sits at a
community-validated point; the other sits where a previous experiment left
it.

So the honest form of every ratio here is **"at these hyperparameters"**, and
the numbers are an **upper bound on the gap**, not a measurement of it. The
LR probe in §3 is the first step at closing that, and until it lands no ratio
in this document should be quoted without the qualifier.

**The full inventory of what is unswept now lives in
[`Hyperparameter_Tuning_Checklist.md`](Hyperparameter_Tuning_Checklist.md)**,
together with the evidence for which knobs actually bind.

**The LR sweep has landed, and it was worth far more than this section
assumed.** L=2 `'none'` at 1.2e-03 settles at **66.98** against 75.09 —
**+10.8%**, moving the ratio from 1.507 to **1.345** on a single knob. The
paragraph above guessed "5-15%" for LR tuning and that was right; the guess
that tuning "is very unlikely to close this on its own" still stands, but
with much less room than it had. **The sweep is now closed**: 2.4e-03 came
back at 69.59, worse by 3.9%, bracketing the optimum at 1.2e-03 (a quadratic
through the three points puts the vertex at 1.13e-03). **`LADDER_LR =
1.2e-03` is the ladder learning rate**, and LR was the largest knob
available.

So this section's qualifier does not merely stand, it **binds harder**: every
ratio in this document was measured at 3e-04, every one of them is now known
to be pessimistic by roughly a tenth, and none should be quoted without
saying so.

What the qualifier does **not** license: closing 68.33 to 49.81 needs **27%**,
and LR tuning on a well-behaved setup typically buys 5-15%. Tuning is very
unlikely to close this on its own, and claiming otherwise in advance would be
the mirror image of the error this section exists to correct.

### 5.5 L=1, `'none'` @1.2e-03 — **DONE 2026-09-24**

Log: [`results/.../L1_idt8_lr0p0012_noattn_altE_fromscratch_32500_result.txt`](../notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_L1probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt8_lr0p0012_noattn/L1_idt8_lr0p0012_noattn_altE_fromscratch_32500_result.txt)

One hop, `LADDER_T` held (dt = 8), same LR, same schedule, same batches as
the L=2 `'none'` @1.2e-03 run. **Read with the static-bank caveat of §6.1:**
at L=1 the register bank is read by the reverse channel but never updated,
so this arm is "one hop with a frozen Fock bank", not "one hop with the
Fock mechanism".

| | L=2 `'none'` @1.2e-03 | L=1 `'none'` @1.2e-03 | delta |
| --- | ---: | ---: | ---: |
| final (32,500) | 67.63 | 87.94 | +20.31 |
| best | 66.56 (31,500) | 85.49 (31,000) | +18.93 |
| **settled** (last 3) | **66.98** | **87.09** | **+20.11** |

**+30.0%, +0.263 nats.** Against the matched GPT-2 (49.81 settled) the
ratio is **1.748**, versus 1.345 at L=2.

#### The gap widened through the decay; it did not saturate

| step | 8,000 | 11,000 | 15,000 | 20,000 | 25,000 | 30,000 | 32,500 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L=1 / L=2 gap | 12.9% | 14.7% | 19.5% | 22.1% | 27.1% | 24.2% | 30.0% |

This is the opposite shape from the `'attention'`/`'none'` gap of §5.2,
which reached ~10% by step 11,000 and held. Here the gap was still growing
at the end of the stable phase (22% at 20,000) and the decay then opened it
further: the decay bought L=2 **19.3%** (83.05 at 21,500 → 66.98) but L=1
only **11.2%** (98.07 → 87.09). Whatever the decay phase consolidates,
one hop consolidates less of it.

**Prediction scored: MISS, on the low side.** The forecast (74-80) was
made from the L=2 curve shape and assumed the L=1/L=2 gap would saturate
the way the attention gap had. It kept widening, and the decay gain — the
quantity that could turn, and did — was smaller at L=1. Third distinct
failure mode in the forecast record (checklist §8): this one extrapolated
a *saturation* that did not happen.

#### Run health, reported per §6's rule

- **Clip-hit rate 2.6%** (17 of 650 logged steps, max grad-norm 1.87)
  against **0.0%** at L=2 @1.2e-03 (max 0.42). Small, but non-zero at the
  same LR — the clip is a depth-dependent intervention, as §6 says, and it
  fires *more* with fewer layers, not fewer. Noted, not corrected.
- 0 watchdog triggers, 0 spike captures. `bproj_sig` saturated at 80.4
  (85.4 at L=2). `sig_max` frozen at 14.286 for the entire run — the static
  bank of §6.1, visible in the log.
- Resonance monitor: empty summaries throughout (the known
  `semsimula_diag` patch mismatch; missing diagnostic, not a passing one).

#### What this run settles, and what it cannot

It puts a number on the one-hop floor at the ladder LR: **87.09**, 30%
behind two hops. It cannot say how much of that 30% is depth and how much
is the frozen bank, because L=1 changes both at once (§6.1, and
[`Fock_Mechanism_Efficiency_Across_Layer_Depth.md`](Fock_Mechanism_Efficiency_Across_Layer_Depth.md)
§6, where the register-to-token path ablates to +52% at L=1 and +275% at
L=2). The decomposition is the L=1 run at `register_salience_init = 0.5`,
which is outside the ladder by design.

### 5.6 L=2, `'none'`, reverse channel off — the conservative-only control, **DONE 2026-09-25**

Log: [`results/.../L2_idt4_lr0p0012_norc_noattn_altE_fromscratch_32500_result.txt`](../notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/L2_idt4_lr0p0012_norc_noattn_altE_fromscratch_32500_result.txt)

Single-variable control against run 3: `REVERSE_CHANNEL = False`, nothing
else changed. The register bank is still created and destroyed but has no
path to the tokens, so the Fock mechanism is off and the model is the
conservative architecture — V_θ, V_φ and the ξ content routing.

| | L=2 `'none'` (run 3) | L=2 `'none'`, no reverse channel | delta |
| --- | ---: | ---: | ---: |
| final (32,500) | 67.63 | 88.82 | +21.19 |
| best | 66.56 (31,500) | 85.90 (31,000) | +19.34 |
| **settled** (last 3) | **66.98** | **87.93** | **+20.95** |

**+31.3%, +0.272 nats.** Ratio to the matched GPT-2 (49.81): **1.765**,
against 1.345 with the mechanism on.

#### The ablations overstated the mechanism by roughly threefold

This is the headline, and it is methodological:

| estimate | method | ratio to its own baseline |
| --- | --- | ---: |
| Cell 6b-8, ablation A | inference-time removal from a trained model | 3.75x |
| Cell 6b-11 (E5), λ = 0 | the same removal, reached continuously | 3.91x |
| **run 8, trained without** | **from scratch, every parameter free to compensate** | **1.31x** |

The two ablations overstate the mechanism's value by **2.9x in PPL ratio
and 4.9x in nats**. Both were already labelled upper bounds
([`Fock_Mechanism_Efficiency_Across_Layer_Depth.md`](Fock_Mechanism_Efficiency_Across_Layer_Depth.md)
§6.1, master doc §11.6); this run says how loose those bounds are. **No
"+275%" or "3.9x" figure may be quoted as the price of the Fock
mechanism.** The price is **+31.3%**.

#### The coincidence with L=1

The L=1 `'none'` run (§5.5), whose register bank is read but never
updated, settled at **87.09**. This arm, with two layers and no register
path at all, settles at **87.93** — within 1% of it. Two quite different
mutilations of the same architecture land in the same place: one hop with
a frozen bank, and two hops with no bank. Whether that is coincidence or
a ceiling imposed by what V_θ, V_φ and ξ-routing can do alone is not
settled by these two points; F5's from-scratch reading and the L=4 rung
would speak to it.

#### The gap widened through training

| step | 8,000 | 11,000 | 15,000 | 20,000 | 25,000 | 32,500 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gap vs run 3 | 19.1% | 20.0% | 23.9% | 25.4% | 29.7% | 31.3% |

Same shape as the L=1 comparison and the opposite of the saturating
`'attention'`/`'none'` gap of §5.2. The decay bought this arm **13.6%**
(101.81 at 21,500 to 87.93) against run 3's 19.3% — consolidating less,
as the L=1 arm also did at 11.2%.

**Prediction scored: BAND HIT, point high.** Pre-registered 105, band
85–140 (F5 design, reformulation §3.5); actual 87.93, which is 3% above
the lower edge and 16% below the point. **The first band hit in the
programme's forecast record**, and the pre-registration earned it for the
right reason: the named quantity that could turn — whether V_φ's share of
the step grows once the reverse channel is not competing — was recorded
with "if it grows into the tens of percent, the run lands at the low end
or below," and it landed at the low end. The band was deliberately wide
on the high side; the answer came in at the bottom. **Open follow-up:**
run Cell 6b-11 or a V_φ attribution probe on *this* checkpoint to confirm
V_φ actually grew, rather than inferring it from the PPL.

#### Run health, per §6's rule

- **Clip-hit rate 2.6%** (17 of 650 logged steps, max grad-norm 2.05),
  identical to the L=1 rate and against **0.0%** for run 3. Both arms with
  the Fock mechanism degraded clip at 2.6%; the intact arm does not clip
  at all. Noted, not corrected.
- `top[...]` never named `reverse_channel_scale` or `reverse_ch` in the
  whole run — those parameters do not exist in this arm, which is the
  visible signature that the mechanism is genuinely absent.
- `sig_max` drifted **14.296 → 60.11**, driven entirely by `fock_reg` and
  the register repulsion, since the LM loss reaches no register parameter.
  Harmless — nothing downstream reads them — and both auxiliary terms are
  identically weighted in run 3, so the paired comparison is unaffected.
- 0 watchdog triggers, 0 spike captures. `bproj_sig` saturated at 88.7
  (85.4 in run 3).

#### The V_φ follow-up, answered offline: V_φ did not grow — **2026-09-30**

Gradient-starvation Tier 0
([`Gradient_Starvation_Investigation.md`](Gradient_Starvation_Investigation.md))
measured the forces on this run's `_best.pt`:

- V_φ's RMS force is **9% of F_θ at layer 0** and **effectively zero at layer 1** (below 5e-5).
- It did not grow "into the tens of percent". The band hit above therefore did not come about for the reason the pre-registration named; that part of the reading is withdrawn.
- The same measurement shows why V_φ could not have grown much: **no loss gradient reaches an earlier token through V_φ or through ξ**. Both detach their sources under causal force.
- So in this arm the conservative architecture's two inter-token channels trained only from the target side.
- The +31.3% "price of the Fock mechanism" therefore includes an unknown share of starvation. That is what P2.1 below measures.

#### P2.1 pre-registered: conservative-only with live V_φ and ξ sources — **frozen 2026-10-01, at launch, before any eval**

**Run.**

- Tag: `…cgqk_norc_vplive_xilive_L2probe…`.
- Config: this arm's config plus `VPHI_GRAD_PATH = XI_GRAD_PATH = 'live'`, with `PROBE_MAX_STEPS = 3_000`. Everything else is unchanged.
- The forward force is bit-identical to this arm's: verified on its `_best.pt`, max |Δlogit| = 0 (Tier 1). Only the backward pass reaches earlier tokens.

**Scored at step 3,000** against this arm's own trajectory (1,000: 267.41; 2,000: 183.10; 3,000: **158.09**). The noise band is ±2.5%:

| outcome | step-3,000 PPL | reading |
| --- | --- | --- |
| **move** | **≤ 154.1** | the detach cost something; run P2.3 / P2.4 (V_φ alone, ξ alone) to split it |
| null | 154.1 – 162.0 | the detach costs nothing measurable at 3k; on its own this does not stop the programme — P2.2 still runs |
| worse | > 162.0 | live sources hurt early training; check the clip-hit rate and grad norm before reading it |

Secondary readings (not scored):

- Distance to no-exchange's 140.61. Closing more than half the 17.5-point gap would mean most of the early "Fock price" is starvation.
- Clip-hit rate, against 2.6% for the parent.
- GPU memory on the first log lines. This is not measured anywhere yet.

Tier 0 suggests that any effect will be mostly ξ: live ξ carries 4–20× more source gradient than live V_φ. That expectation is recorded here so that P2.4 can test it.

#### P2.1 scored at step 3,000: **127.73 — MOVE, by eight times the threshold** — **2026-10-01**

Log: [`results/.../L2_idt4_lr0p0012_norc_vplive_xilive_noattn_probe3000_result.txt`](../notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/L2_idt4_lr0p0012_norc_vplive_xilive_noattn_probe3000_result.txt)

| step | conservative-only (parent) | **P2.1: same, V_φ + ξ live** | Δ vs parent | no-exchange (run 3) | `attention` | `rglive` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 490.68 | **463.41** | −5.6% | — | 481.47 | 472.41 |
| 1,000 | 267.41 | **241.67** | −9.6% | 257.74 | 248.81 | 255.98 |
| 1,500 | 210.60 | **179.65** | −14.7% | — | 188.01 | 192.73 |
| 2,000 | 183.10 | **151.89** | −17.0% | 167.87 | 158.76 | 162.27 |
| 2,500 | 164.90 | **134.33** | −18.5% | — | 140.11 | 142.28 |
| **3,000** | **158.09** | **127.73** | **−19.2%** (0.213 nats) | 140.61 | 131.20 | 133.74 |

**Scored: MOVE.**

- The criterion was ≤ 154.1, i.e. 2.5% below the parent. P2.1 is **19.2%** below.
- The gap widened at every eval: 5.6% → 9.6% → 14.7% → 17.0% → 18.5% → 19.2%. It is not an early-training transient.
- The forward force is bit-identical to the parent's (Tier 1). The only thing changed is which tokens the loss gradient reaches.

**What it means at 3,000 steps:**

1. **Opening the V_φ and ξ source gradients is worth more than adding the Fock mechanism.**
   - P2.1 has no reverse channel, so no register ever reaches a token. Yet it is **9.2% below no-exchange** (127.73 vs 140.61), which has the reverse channel but starved V_φ and ξ.
   - The parent's 17.5-point gap to no-exchange was not just closed; it is **overshot by 12.9 points**.
   - At step 3,000, the "price of the Fock mechanism" measured in §5.6 (+31.3% settled) is **entirely an artefact of the gradient convention**.
2. **It is the best L=2 step-3,000 number in the programme.** It beats `attention` (131.20, −2.6%) and `rglive` (133.74, −4.5%), with **no non-conservative force and no register path**.
3. **The arm also trains more stably.** Clip-hits in steps 1–3,000: **2 of 60** logged steps (max grad-norm 1.16), against **17 of 60** for the parent (max 2.05).
   - All of the parent's 17 clip-hits (its "2.6%, 17 of 650" above) fell in these first 3,000 steps.
   - Training loss agrees with the evals: ntp 4.875 against 5.087 at step 3,000.
   - Peak GPU memory 22.2 GB, the same as the parent; wall time +3%.
4. ~~**ξ is consistent with "learning to read further back".**~~ **Corrected 2026-10-01:** that reading had the sign backwards. ξ weights source s by α^(t−s), so a *smaller* α means a *shorter* memory. The fastest channel moved 0.500 → 0.412 by step 3,000 (parent: 0.428): it became **more local**, slightly more than the parent's. The other four track the parent's.

**What it does not show yet:**

- **A settled number.** At 3,000 steps the `attention`/no-exchange gap was 6.7% and settled at 5.2%; the `rglive`/`attention_potential` gap was 10.2% and *grew* to 24.5%. Neither direction is safe to assume.
- **Which channel carries the effect.** P2.3 (V_φ only) and P2.4 (ξ only) split it. Tier 0 and Tier 1 predict mostly ξ.
- **Whether the Fock arm gains as much** (P2.2). If it does, the ladder shifts but keeps its order. If it gains less, the mechanism's value was partly compensating for starved channels.
- **The SCAF audit.** The first audit was due at step 5,000, so none ran. The forward pass is bit-identical, so causality is unchanged by construction, but the full run will audit it.
- **Seeds.** This is one seed, as for every ladder arm.

**Consequences, effective now:**

- Every statement that prices the Fock mechanism or the conservative architecture is **suspended** until P2.2 and a full run land. That covers the +31.3% above, the "ablations overstated by 3×" table, the F5 reading, and the model cards and book wherever they quote these.
- The §5.6 headline numbers stand as measurements of the **starved** convention. They are no longer readings of the architecture.

#### P2.1 extended to the full 32,500 steps (F3.1) — pre-registered **2026-10-01, at step 3,000, before step 3,001**

The run continues from `_step3000_probe_stop.pt` (model and optimizer state) on the WSD schedule it has used from step 1. Nothing else changes. This replaces P2.2 as the next GPU job, at the author's call. P2.2, P2.3 and P2.4 stay queued.

**Basis.** The ratio of settled PPL to step-3,000 PPL in the four finished L=2 arms:

| arm | 3,000 | settled | ratio |
| --- | ---: | ---: | ---: |
| conservative-only (parent) | 158.09 | 87.93 | 0.556 |
| no-exchange | 140.61 | 66.98 | 0.476 |
| `attention` | 131.20 | 63.51 | 0.484 |
| `rglive` | 133.74 | 61.11 | 0.457 |

127.73 × 0.556 = **71.0** (if it consolidates like its parent) and 127.73 × 0.476 = **60.8** (if it consolidates like no-exchange). The upper edge allows for the 3k gap shrinking to −10% of the parent: 0.90 × 87.93 = **79.1**.

**Prediction:**

- **Settled (mean of the last three evals): point 67, band 60–79.**
- **The question this run exists to answer:** does conservative-only with live gradients settle **below no-exchange's 66.98**? That would mean the conservative architecture alone, trained properly, beats the Fock arm as trained.
  - The point sits on that line, so call it **even odds**.
  - Below 63.51 would also beat `attention`. I don't expect that (about 1 in 4).
- **Settled above 79.1** means the early advantage mostly washed out in the decay. The detach would then cost little at convergence, and the 3k reading would be withdrawn as transient.

**Health, also scored:**

- Clip-hit rate below the parent's 2.6%.
- All seven SCAF audits CLEAN (5k, 10k, …, 30k, 32.5k). The forward pass is bit-identical, so anything else would be a harness or code fault, not a leak.
- No watchdog trigger.

#### F3.1 mid-run reading at step 15,000 — **2026-10-01, not a revision of the prediction above**

Log (in progress, not yet filed): `~/Downloads/L2_none_norc_32500steps_output.txt`. Validation PPL at matched steps:

| step | **F3.1: conservative-only, V_φ + ξ live** | conservative-only (parent) | no-exchange | `attention` | `rglive` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3,000 | **127.73** | 158.09 | 140.61 | 131.20 | 133.74 |
| 5,000 | **106.71** | 136.05 | — | 107.62 | 106.77 |
| 7,500 | **93.69** | 120.86 | 102.08 | 93.85 | 91.60 |
| 10,000 | **85.13** | 114.31 | 97.14 | 86.97 | 83.74 |
| 12,500 | **81.97** | 114.28 | 90.62 | 86.28 | 82.64 |
| 15,000 | **77.31** | 107.30 | 86.57 | 81.27 | 77.46 |
| Δ vs parent at 15,000 | **−27.9%** | | −19.3% | −24.3% | −27.8% |

No-exchange's log in the repo starts at step 6,500, so it has no 5,000 eval.

- **The gap to the parent kept widening:** −19.2% at 3,000 and −27.9% at 15,000.
- **It is now below no-exchange by 10.7%** and below `attention` by 4.9%. It is level with `rglive`, the best L=2 arm, which has an exchange field and a reverse channel. This model has neither.
- **What it implies for the settled score, by the same method as the pre-registration.** The finished arms settled at 0.77–0.82× their 15,000-step PPL: parent 0.819, no-exchange 0.774, `attention` 0.781, `rglive` 0.789.
  - Applied to 77.31, that gives **59.8–63.3**.
  - That is below the pre-registered point (67), below no-exchange (66.98) by a clear margin, and probably below `attention` (63.51).
  - It sits near the band's lower edge (60). The run is outperforming its own prediction on the low/good side.
  - The prediction stands as frozen; it is scored at 32,500.
- **Health:**
  - Clip-hits: 2 of 300 logged steps (0.7%), both before step 3,000; max grad-norm 1.16.
  - SCAF CLEAN at 5k, 10k and 15k; leak tax ≤ 1.1e-4 nats.
  - No watchdog trigger; memory flat at 22.2 GB.
- **ξ:** the two fast channels kept shortening: α 0.335 and 0.570 at 15,000, against the parent's 0.371 and 0.616. The slow channels hold near 0.91 and 0.99.
  - With live gradients, ξ makes its fast channels **more local**, not longer-range.
- **Resonance: much stiffer than any arm so far.** The in-flight ω·dt median rose steadily: 2.16 at 3,000, 2.72 at 5,000, 4.88 at 10,000 and **6.41 at 15,000**, with 99% of tokens over the wall and max 9.6–10.0.
  - For comparison: `rglive` held at about 1.6 (33–40% over), and no-exchange **ended** at 3.80 (6b-13, `_best.pt`). The parent's in-flight monitor was not working (the pre-409175b hook bug).
  - Under `baoab_cfc_lowrank` those modes are integrated exactly, so this is a **stiffness** reading, not an instability. Loss and grad-norm are clean.
  - But it is the largest stiffness in the programme, and it is still rising, though more slowly (6.17 → 6.41 over the last 2,000 steps).
  - The plausible mechanism: live ξ lets V_θ's context-dependent wells be **trained from the source side**, and they are getting sharper.
  - **Watch item:** if the median keeps climbing through the decay, or the loss shows spikes, run 6b-13 on the parent's `_best.pt` to see whether the detached arm was stiff too. Its monitor never read it.

#### F3.1 second mid-run reading at step 29,500 — **2026-10-01, still not the score**

| step | **F3.1** | parent | no-exchange | `attention` | `rglive` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 25,000 | **67.46** | 97.50 | 75.19 | 72.49 | 68.47 |
| 27,500 | **61.29** | 91.34 | 71.66 | 66.91 | 63.68 |
| 28,500 | **59.35** | 90.96 | 68.95 | 66.36 | 63.50 |
| 29,000 | **58.84** (best) | 89.55 | 68.82 | 65.02 | 62.11 |
| 29,500 | **59.38** | 88.39 | 72.47 | 64.11 | 61.31 |
| *settled* | *pending* | 87.93 | 66.98 | 63.51 | 61.11 |

- **Ahead of every L=2 arm at every matched step since 25,500.**
  - At step 29,000 it is 5.3% below `rglive`, 9.5% below `attention`, 14.5% below no-exchange, and **34.3% below its own detached-gradient parent**.
- **Projected settled score: about 57–59.**
  - The finished arms settled at 0.973–0.984× their step-29,000 value. Applied to 58.84, that gives 57.3–57.9.
  - The noisier step-29,500 ratios give 54.8–59.2.
  - Either way it lands **below the pre-registered band (60–79)**, a miss on the good side. It would also be below `rglive`'s 61.11, the best settled L=2 number so far.
  - Ratio to the matched GPT-2 (49.81): about 1.16, against 1.345 for no-exchange.
- **Stiffness eased through the decay.** The ω·dt median peaked at 6.41 (step 15,000), then fell: 5.95 → 5.10 → 4.79 → 4.54 → **4.45** at 27,500. That is still above no-exchange's 3.80 endpoint, but the watch item is resolving.
- **Health:**
  - Two isolated clip-hits in the stable phase, at steps 17,700 (1.51) and 19,250 (2.19). Total 4 of 593 logged (0.7%), against the parent's 2.6%. No watchdog trigger.
  - SCAF CLEAN at 5k, 10k, 15k, 20k and 25k.

#### F3.1 scored: **57.76 settled — the best L=2 model in the programme** — **2026-10-02**

Log and probe outputs: [`results/…cgqk_norc_vplive_xilive_L2probe…_noattn/`](../notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_vplive_xilive_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn/).

| | F3.1: conservative-only, Vφ + ξ live | conservative-only (parent, detached) | no-exchange | `attention` | `rglive` | matched GPT-2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **settled** (last 3) | **57.76** | 87.93 | 66.98 | 63.51 | 61.11 | 49.81 |
| best | 57.35 (32,500) | 85.90 | 66.56 | | 59.09 | |
| final | 57.35 | 88.82 | 67.63 | | 61.06 | |
| ratio to GPT-2 | **1.160×** | 1.765× | 1.345× | 1.275× | 1.227× | 1 |

The last three evals were 58.39, 57.54 and 57.35. The run was still improving at the last eval, so best = final.

**Scored against the frozen pre-registration (point 67, band 60–79):**

- **MISS on the good side.** 57.76 is 2.24 below the band's lower edge and 9.24 below the point. The basis, the finished arms' settled-to-3k ratio, underestimated how much more this arm consolidates (0.452×, below every finished arm's 0.457–0.556×).
- **The question the run existed to answer: YES, by a clear margin.**
  - It settles **13.8% below no-exchange** (57.76 vs 66.98). The conservative architecture alone, with live gradients, beats the Fock arm as trained.
  - It is also 9.1% below `attention` and 5.5% below `rglive`. I had called beating `attention` about a 1-in-4 chance.
- **Against its own parent: −34.3%** (−0.420 nats). The forward pass is identical; only the gradient convention changed.

**Health, scored:**

- **Clip-hit 0.6%** (4 of 650 logged steps, max 2.19) against the parent's 2.6%: HIT.
- **SCAF CLEAN at all seven audits.** Future perturbation was exactly 0.0, and the leak tax was at most 2.8e-4 nats where measured. At 32.5k the honest stage was skipped (prints nan), which is the known cosmetic print issue: HIT.
- **No watchdog trigger:** HIT.

**Probe readings, against the parent on the same architecture.** E1 compares like with like only within this pair.

| reading | parent (detached) | **F3.1 (live)** |
| --- | ---: | ---: |
| 6b-7 gate 1: PPL rise when the velocity is reset | +4.72 | **+19.33** |
| 6b-7 gate 3: refinement at fixed T | MAPS | MAPS |
| 6b-9 R(geo), distance from the Vθ-only geodesic | 0.742 | 0.665 |
| 6b-9 **Vφ attribution** (geo → cons) | **−0.0015** | **−0.291** |
| 6b-9 LN attribution | −0.669 | −0.267 |
| 6b-9 layer 1: R_h geo → cons | 0.642 → 0.642 (Vφ absent) | **0.772 → 0.240** |
| 6b-12 layer 1, Vθ geodesic + LN: median R_tok (Vφ's per-token share) | 0.000 | **0.669** (85.5% of tokens in 0.25–0.75) |
| 6b-13 ω·dt median (share past 2) | not run | 4.32 (99.8%) |

1. **Vφ woke up.** Its attribution grew about 190×, from −0.0015 to −0.291.
   - At layer 1, where it was effectively off in the parent (Tier 0), it now carries most of the step's departure from the Vθ geodesic, and it does so for almost every token.
   - **"Vφ is inert" (E1, F5) is withdrawn.** It was a property of the detached convention, exactly as the investigation predicted.
   - This also reverses Tier 0's expectation that ξ would carry most of the effect; P2.3 and P2.4 now matter more, not less.
2. **The step is a geodesic of the full conservative potential, up to LayerNorm.** With no reverse channel this holds by construction (cons+LN = full).
   - What is new is that Vθ alone no longer describes it: R(geo) = 0.665, with Vφ making up the difference.
   - The Riemannian reading, a damped geodesic of V = Vθ + Vφ at the trained step, holds for this model, piecewise: 6b-7 still reads MAPS, so the steps cannot be subdivided.
3. **Inertia is worth 4× more:** +19.33 against +4.72 when the velocity is reset.
4. **E3 (6b-10) against the matched GPT-2 on the same batches.** The cell's "GATE SHUT, results void" line is a false alarm for an arm with no reverse channel, fixed in the notebook on 2026-10-02.
   - Velocity forecast error: 1.008 vs 1.314. **Fock lower, as pre-registered.**
   - Perturbation growth per step: 0.920 vs 0.979. **Fock at or below, as pre-registered.**
   - Tangential direction coherence: −0.126 vs +0.195. **Opposite to the pre-registration.**
   - Two of three, against a null at L=2 for no-exchange (§6.8 of the geodesic note). But the cell itself warns that at L=2, (a) and (b) are dominated by the embedding-to-sphere step and need L ≥ 3. **Not to be quoted as a forecastability result until an L ≥ 3 live arm runs.**
5. **The stiffest arm at the endpoint:** ω·dt median 4.32, with 99.8% past the wall, against no-exchange's 3.80. Integrated exactly, so not an instability.
   - The in-flight median peaked at 6.4 (step 15,000) and settled to 4.3 through the decay.
   - 6b-13 printed MISS against [3.3, 4.2]. That band was pre-registered for no-exchange and does not apply to this arm.

**Consequences:**

- The §5.6 Fock price (+31.3%) and the F5 reading are confirmed withdrawn. Under the live convention, removing the Fock mechanism from a starved model and opening Vφ/ξ instead buys −13.8% against the starved Fock arm.
- **The Fock mechanism's real value is now unknown.** It needs no-exchange with live gradients (P2.2, then the full run), which is now the most important open run.
- Single seed.

---

### 5.9 SR1–SR4: settling, refinement and depth extension — **pre-registered 2026-10-03, before any run**

Book §8.9 (`ssec:settling-refinement`, Props 44–45) gives the theory: the split A·O·A step makes a stiff mode's dissipation depend on the step count through θ/sin θ (θ = ω·Δt), proportional to the damping ratio; the exact damped-mode flow removes that for the linear stiff part; settling needs ‖v_T‖ small; a common cause, single-discretisation training with weak damping, sits behind all three failures.

**Baseline** (all four arms): the L=2 live-gradient conservative-only configuration of F3.1, **retrained from scratch per arm, full 32,500 steps**. The comparison is against F3.1's own probe readings:

| baseline reading (F3.1) | value |
| --- | ---: |
| settled PPL | 57.76 |
| Gate 1, velocity reset | +34.3% |
| Gate 2, 1.5× steps at fixed Δt (N=3) | +92% |
| Gate 3, 1.5× steps at fixed T (N=3) | +143% |
| CG1 V_φ attribution | −0.291 |
| ω·Δt median | 4.32 |

**Arms and predictions:**

| arm | intervention | Gate 3 (1.5×) | Gate 2 (1.5×) | Gate 1 | settled PPL |
| --- | --- | --- | --- | --- | --- |
| **SR1** | palindromic order A·O½·B·O½·A | within ±25% of +143% | within ±25% of +92% | within ±25% of +34% | within ±2% |
| **SR2** | exact damped-mode flow on span(U), constant γ = 0.1 | **≤ +100%** | not predicted | within ±25% | within ±3% |
| **SR3** | SR2 + constant-ratio friction ζ* = 1 on the stiff modes | **≤ +70%** | **≤ +50%** | **≤ +25%** (inertia falls) | within +5% |
| **SR4a** | train with N ~ U{2, 3, 4} at T = 8 (depth code by time, 6b-7 `hold` policy) | **≤ +10% on N ∈ {2, 3, 4}** | not predicted | **≥ +27%** (≥ 80% of baseline: inertia kept) | within +5% |
| **SR4b** | train with N ~ U{2, 3} at Δt = 4 | not predicted | **≤ +10% at N = 3** | not predicted | within +5% |
| **SR5a** | thermal training, `LANGEVIN_T = 0.00025` (r ≈ 0.1; added 2026-10-03, see below) | within ±25% of +143% | within ±25% of +92% | within ±25% of +34% | within +2% |
| **SR5b** | thermal training, `LANGEVIN_T = 0.0022` (r = 0.3) | **≤ +100%**, called at 45% | **≤ +70%**, called at 40% | **≤ +25%**, called at 60%: inertia falls | **+2% to +8%**, point +4% |

**Decision rule** (scored after SR2–SR4a are in):

- **SR2 or SR3 meets its Gate 3 threshold and its PPL band:** the split, or the underdamping, is a cause. Adopt it in the integrator.
- **Only SR4a meets it:** the single-discretisation objective is the cause. Variable-step training becomes the default for any model meant to have depth as an inference-time knob.
- **Both meet it:** the effects add. Run SR2 + SR4a together.
- **Neither meets it:** the cause lies in what no arm touches (per-step refreezing of the occupancies, the kick, per-layer context, the projection). Record this as a negative result.
- **The common-cause check:** across the baseline and SR1–SR4a, Gate 1 and Gate 3 have been perfectly rank-ordered over the four L=2 arms so far. SR4a is predicted to **break** that ordering, keeping inertia while being refinable; SR3 is predicted to **keep** it, losing both together.

**SR-π: refinement cost and the π crossing (added 2026-10-04, before any run that tests it).**

Proposition 44 amplifies the phase-sampled dissipation of the split step by θ/sin θ (θ = ω·Δt), and the amplification is sharpest near θ = π, where the sampled phase aliases. Three Gen 3 arms now have both the trained θ (6b-13 median) and Gate 3 at 1.5× the steps:

| arm | trained θ | θ/sin θ | θ after 1.5× refinement | crosses π? | Gate 3, 1.5× |
| --- | ---: | ---: | ---: | --- | ---: |
| L=4 Fock live | 2.35 | +3.3 | 1.57 | no | +216% |
| F3.1 (L=2 conservative-only) | 4.32 | −4.7 | 2.88 | yes | +143% |
| G2 (L=2 Fock live) | 3.40 | −13.3 | 2.27 | yes | **+1,274%** |

**Reading.**

- **The Fock pair agrees with the mechanism.** G2 trains its stiff modes just past π, at the largest |θ/sin θ| of the three, and refinement carries them back across π, flipping the sign of the phase term. L=4 trains below π and stays below it.
- **F3.1 does not fit cleanly.** It crosses π but fails much less, so the crossing is not sufficient alone. The register path (present in G2 and L=4, absent in F3.1) amplifies whatever the crossing does.
- **Weaknesses:** one seed each, medians over wide spreads (G2's θ runs from 2.6 to 5.5 between p05 and p95), and "1.5×" means N = 3 at L=2 against N = 6 at L=4.

**Predictions, pre-registered.**

| id | prediction | called |
| --- | --- | --- |
| SR-π.1 | Across Fock arms at matched mechanism and convention, Gate 3 at 1.5× falls monotonically as the trained median θ moves below π. An **L=8 Fock live** arm (Δt = 1; expected θ ≈ 1.2) reads Gate 3 ≤ +100%. | 60%; **revised to about 40% on 2026-10-05** after G3 (θ 2.07, no crossing, Gate 3 +1,342%) |
| SR-π.2 | On G2's configuration, **SR2** (the exact damped-mode flow, which removes the phase term) cuts Gate 3 by more than half, a far larger relative cut than on F3.1's configuration. | 55% |
| SR-π.3 | A **diagnostic, free:** on G2's checkpoint, split the Gate 3 loss rise by token according to whether the token's stiff modes cross π under refinement (from the 6b-13 per-coordinate θ). Tokens whose modes cross contribute disproportionately: their mean loss rise is ≥ 2× that of non-crossing tokens. | 55% |

**Design lever if these hold.** Choose L, or Δt, so the stiff modes train below π. Report the trained θ distribution with every Gate 3 reading.

**Order.** SR-π.3 is free and runs first. SR-π.2 rides on SR2 (adding G2's configuration as a second SR2 arm). SR-π.1 needs an L=8 Fock live run: about 6 s/step, so about 55 h. It is scheduled after the CB series.

**SR5, thermal training (added 2026-10-03, before any run).** `LANGEVIN_T > 0` turns the O-step into an FDT-locked thermostat. The noise is applied in training only (`noise_eval=False`), so eval PPL and every 6b probe stay deterministic. The cell-0 switch and its tag already exist; the tag now renders `T0p00025` / `T0p0022`.

- **Why it is in the SR series.** It attacks the common cause from a different side than SR4.
  - SR4 randomises the discretisation, so the model must make its momentum work at every step count.
  - Velocity noise randomises the phase and momentum detail of the one discrete trajectory (Prop 44's mechanism), so the readout cannot rely on it.
- **Why it is not a settling route at inference.** The FDT stationary state holds ⟨m‖v‖²⟩ = dT. In addition, with γT = 0.8 the relaxation time (1/γ = 10) exceeds the trajectory (T = 8), so 2–4 steps never anneal into a basin.
- **Temperatures are calibrated, not guessed** (`debug/calibrate_langevin_T.py` on F3.1's checkpoint, eval, 8 × 512 validation tokens; output saved next to it). The kinetic temperature T_kin = m‖v‖²/d entering the O-step has a median of 1.05e-2 at layer 0 and 5.19e-2 at layer 1, pooled 2.43e-2. The median speed rises from 6.7 to 15.0 across the two layers: the trajectory is accelerating, not settling, inside the trained window. T_r = r²·median(T_kin) gives an equilibrium thermal speed of r × the typical speed:

  | arm | r | `LANGEVIN_T` | one O-step injects (median, relative to ‖c v‖, c = e^{−0.4}) |
  | --- | --- | --- | --- |
  | SR5a | 0.1 | 0.00025 (calibrated 2.43e-4) | 0.11 (p10 0.07, p90 0.19) |
  | SR5b | 0.3 | 0.0022 (calibrated 2.19e-3) | 0.33 (p10 0.21, p90 0.58) |

- **The frontier prediction** (the point of the arm). The four L=2 arms have Gate 1 and Gate 3 perfectly rank-ordered. **SR5 is predicted to move along that ordering, not off it:** lower Gate 3 comes with lower Gate 1 and a PPL cost. Called at 65%. SR4a is predicted to break the ordering (above).
- **Decision rule.**
  - **SR5b meets Gate 3 ≤ +100% at PPL ≤ +3%:** thermal training is a cheap repair. Adopt it as a default training regulariser and run it together with SR4a.
  - **It lowers Gate 3 only with Gate 1 ≤ +25% and PPL ≥ +5%:** it trades along the frontier. That confirms the momentum tension, and SR4 remains the route.
  - **SR5a and SR5b both within noise of the baseline:** phase-reliance is not what velocity noise at these scales reaches. Record it as a negative result.
- **CG8 is read on SR5 as well.** Predicted s_G ΔAUROC < 0.01, called at 60%.
- **SR5c, annealed thermostat: conditional on SR4b, not scheduled.** A per-layer temperature, hot early and zero at the last step, combined with stronger late damping (γ(h) or SR3's ζ* = 1): anneal into the attractor, then come to rest. With 2–4 steps there is no time for barrier crossing (1/γ > T), so it is meaningful only on a model trained over longer horizons. It runs only if SR4b passes its Gate 2 threshold. It needs a per-layer temperature in `ou_step`, which does not exist yet.

**Order and cost:** SR1 (cheapest; code change only), then SR2, then SR4a, then SR3, then SR4b. SR5a and SR5b need no code and can run in any free slot; SR5b first. About 14 GPU-h each at L=2. Implementation note: SR2/SR3 need a joint damped-mode substep on span(U) in `cfc_baoab.py`; SR4 needs the step count sampled per batch and the depth code indexed by time. Neither exists yet.

**Caveat recorded in advance:** with γ·T = 0.8, Prop 44 is first-order and qualitative. Its prediction is the *mechanism* (SR2 helps Gate 3), not a magnitude.

### 5.10 The v6 abstract-gating runs — **pre-registered 2026-10-03, before any run**

> **Label note (2026-10-03).** The book already uses "Experiment G1–G4" for the geometric capability experiments (§18d: analogy, energy anomaly, geodesic distance, asymmetry). These runs therefore appear in the book as **AG1–AG4** (abstract-gating). In these notes and in conversation they stay G1–G4.

The v6 abstract (book stage 2, `Paper_v6_Section_Audit.md`) will state three things the published Gen 3 models do not yet settle: what the Fock register mechanism is worth under live gradients, whether depth helps under live gradients, and whether parity with the matched GPT-2 survives a parameter-matched baseline. Each run below decides one of them. All three run on the same notebooks, data and 32,500-step schedule as the published arms. Settled = the mean of the last three evals.

**G1. Parameter-matched GPT-2 at the same width: scheduled, deferred until G2–G4 are in** (`colab_matched_gpt2_baseline_openwebtext.ipynb`). Revised 2026-10-03, before any run.

- **Configuration.** Cell 0: `N_LAYERS = 22`, `TIE_EMBEDDINGS = False`. Everything else is unchanged: `D_MODEL = 384`, `N_HEADS = 6`, LR 6e-4 with a cosine to 6e-5, 16,384 tokens/step.
- **Parameter count:** 77,832,960, which is +1.3% against the L=4 live arm's 76,823,510. L=22 is chosen over L=21 (76,058,496, −1.0%) so that the baseline is never the smaller model.
- **Output folders:** `checkpoints_d384_L22_h6_untied/` and `results_d384_L22_h6_untied/`, via the `VARIANT_TAG` guard. The published 33.7M baseline's files are not touched.
- **Cost:** about 2.7× the baseline's time per step.
- **Width is held at d = 384 by design.** All comparisons are constrained to the same semantic-space dimension. A wider GPT-2 (d = 512, L = 8, 76.9M) was considered and rejected: the conservative model at d = 512 would be a different dynamical system, so matching parameters by width compares across semantic spaces.
- **Author's reservation, recorded in advance.** Matching the parameter count by stacking depth ignores the physics and dynamics of the model, and a reviewer may question it on those grounds. The run is kept for completeness; on its own it is not expected to carry much weight. The primary comparison stays the width-matched 33.7M baseline, with the parameter counts stated in the same sentence.

- **Prediction:** settled **45**, band **41–48**. Adding depth at fixed width and a fixed 0.53B tokens (about 7 tokens per parameter) gains less than widening would.
- **Key line:** settled ≤ 47.6 (5% below 50.10) means parity does not hold on a parameter-matched footing at the same width. Settled > 47.6 means it does.
- **Called:** about 65% that parity does not hold.
- **Abstract, until G1 runs:** the parity sentence quotes the width-matched baseline (49.81, 33.7M) against the L=4 live arm (50.10, 76.8M), with both parameter counts.

**G2 (= P2.2 → full). L=2 Fock-PARFLM `none`, live** (ladder notebook). Cell 0: `LADDER_L = 2`, `LADDER_MECHANISM = 'none'`, `REVERSE_CHANNEL = True`, `VPHI_GRAD_PATH = 'live'`, `XI_GRAD_PATH = 'live'`, `PROBE_MAX_STEPS = None`. Expected tag: `…cgqk_vplive_xilive_L2probe_…_idt4_lr0p0012_noattn`. The Gen 2 twin is the published no-exchange arm (66.98). The 3k criterion in `Gradient_Starvation_Investigation.md` (≤ 137.1 against 140.61) is read from this run's step-3,000 eval. The run does not stop there.

- **Prediction:** settled **53**, band **50–57**.
- **Key line 1, the register mechanism's value under live gradients:** settled < 57.76 (F3.1, the same depth without the Fock mechanism). Called at about 80%.
- **Key line 2, depth under live gradients:** settled > 50.10 means L=4 beats L=2 on one mechanism and one convention, which is the first clean depth answer. Called at about 75%. If it is ≤ 50.10, the L=2 < L=4 inversion survives the gradient fix, and depth is not what the L=4 run bought.
- **CG1 forecast:** V_φ attribution |·| < 0.05 (inert, as at L=4), from the substitution reading. Above 0.15 refutes the substitution account at L=2.

**G2 scored, 2026-10-04.**

| measure | value |
| --- | --- |
| settled (last three evals: 53.01, 53.29, 53.07) | **53.12** |
| best | 51.27 at step 31,000 |
| parameters | 76,770,256, identical to the Gen 2 twin |
| speed | 1.64 s/step |
| ω·Δt at the endpoint | p50 3.41, 99.7% past the wall |
| SCAF audits (5k, 10k, …, 32.5k) | CLEAN at all seven: future perturbation 0.0, Tier A and Tier B both 0 |

The step-32,500 audit printed honest/standard PPL as nan, while its future perturbation is exactly 0. The independent causality check on the final weights is still to run.

- **Point 53, band 50–57: HIT, on the point.**
- **Step-3,000 criterion: HIT.** 123.59 against ≤ 137.1, which is −12.1% vs the Gen 2 twin's 140.61.
- **Key line 1, the register mechanism under live gradients: YES.** 53.12 vs F3.1's 57.76, −8.0%.
- **Key line 2, depth under live gradients: YES.** The L=4 live arm, at 50.10, is 5.7% better (a 3.02 PPL gap, larger than either arm's last-three spread). It costs 1.9× per step (3.12 vs 1.64 s/step). This is the first clean depth answer: one mechanism, one convention.
- **Against the Gen 2 twin (66.98): −20.7%.** Against the matched GPT-2 (49.81): 1.067×.
- **CB stop rule:** G2 < 56.6, so the CB series is **live**.
- **CG1 V_φ forecast: HIT.** 6b-9 gives a V_φ share of **−0.0002** against the threshold |·| < 0.05. V_φ is inert in the L=2 Fock arm, as at L=4 (−0.0035), against −0.291 in F3.1. The substitution account now holds at both depths.

**G2 6b readings, 2026-10-04** (best checkpoint, step 31,000; files in G2's results folder):

| reading | F3.1 (L=2 conservative-only live) | **G2 (L=2 Fock live)** | L=4 Fock live |
| --- | ---: | ---: | ---: |
| CG1 V_φ share | −0.291 | **−0.0002** | −0.0035 |
| CG1 reverse-channel share (cons+LN → full) | — | −0.694 | −0.449 |
| CG1 R(geo) | 0.665 | 0.911 | 1.369 |
| Gate 1, velocity reset | +34.3% | **+32.7%** (51.99 → 68.97) | +55% |
| Gate 2, 1.5× steps at the trained Δt | +92% | +50.7% (N=3); +14.7% at N=4 | +41% |
| Gate 3, 1.5× steps at fixed T | +143% | **+1,274%** (N=3); +611% at N=4 | +216% (N=6) |
| CG6 forcing, layer 1 | — | uniform: 100% of tokens above 0.75 | — |
| CG7 ω·Δt p50 | 4.32 | **3.40**: in the pre-registered L=2 band [3.3, 4.2], HIT | 2.35 |
| 6b-8, reverse channel off (inference ablation) | — | **+300%** (51.17 → 204.84) | — |
| 6b-8, bank frozen at init | — | +24.0% | — |

**Two findings.**

1. **The momentum–refinement rank order breaks.** Book §8.9 states that, across four L=2 arms, Gate 1 (reliance on velocity) and Gate 3 (refinement failure) are perfectly rank-ordered. G2 relies on velocity *slightly less* than F3.1 (+32.7% vs +34.3%), yet fails refinement **nine times worse** (+1,274% vs +143%). The non-conservative register path drives refinement failure independently of momentum. That is consistent with the Gen 1 finding, "how badly refinement fails tracks non-conservative content". §8.9's common-cause paragraph needs this fifth arm (stage 2). For the CB series it raises a direct prediction: capping the Fock path (CB2, CB3) should cut Gate 3 sharply.
2. **Ablation overstates trained value by far more under live gradients.**
   - **Under live gradients:** switching the reverse channel off at inference costs +300%, yet the model trained without it (F3.1, 57.76) is only 8.7% worse than G2.
   - **In Gen 2:** the same pair read about +275–291% against 31.3%.

   The book's methodological sentence ("ablation indicates 3.75–3.91× against 1.31× trained without") gains a live-gradient row: **4.00× against 1.087×**.

**G3 (= F3.2). L=2 `attention_potential`, everything live** (ladder notebook). Cell 0: `LADDER_L = 2`, `LADDER_MECHANISM = 'attention_potential'`, `REVERSE_CHANNEL = True`, `RELAX_GRAD_PATH = 'live'`, `VPHI_GRAD_PATH = 'live'`, `XI_GRAD_PATH = 'live'`, `PROBE_MAX_STEPS = None`. The tag carries `rglive`, `vplive` and `xilive` and ends `_attnpot`. Its partial-live parent is the published `rglive` model (61.11).

- **Prediction:** settled **51**, band **46–56**. The live V_φ/ξ gain is smaller than the conservative-only arm's 34%, because the exchange field already carries some of what the starved channels could not.
- **Key line, the conservative exchange field's value under live gradients:** settled below G2's settled value by more than 2% (beyond eval noise). Called at about 60%.
- **Mid-run reading, 2026-10-04, step 21,700, recorded before the result.** The run is not stopped; the pre-registered scoring stands.

  | | G3 | G2 | the parent probe |
  | --- | --- | --- | --- |
  | PPL, steps 3,000–8,000 | led G2 by **5–7 PPL** | — | — |
  | PPL, step 21,500 | 64.18; crossed G2 near 13,000, now trails by about 1.6 (+2.5%) | 62.61 | G3 holds a steady −11% against it throughout |
  | global gradient norm, constant-LR phase | rose 0.43 → **0.88** | flat at about 0.35 | rose 0.42 → 0.64 |
  | clip hits | 5.9% of steps in the last window | 0 | 0 |

  - **Where the growth is.** When `relax_field` tops the per-group gradient table (35 logged steps; it never appears in G2's log), its gradient alone rises from about 0.3 to about 0.8, almost the whole model's norm. The growth is the exchange field's.
  - **Mechanism, a hypothesis.** The field's routing is unnormalised: scores = q·k/√d_k with free W_q and W_k (`model_xi_attention.py`, `XiRoutedConservativeAttention._routing`), and its bilinear value kernel is unbounded too. It has no per-group clip override either; it shares the global 1.0.
    - In Gen 2 its routing received no gradient, so it could not sharpen. Neither the hardening the creation gate got (QK-norm, `cgqk`) nor the reverse channel's (stable QK-norm + soft-norm) was ever needed or applied.
    - Under live gradients the field learns, and its gradients grow through the stable phase. That is the signature of logit growth (sharpening routing), the failure the programme met and fixed in the creation gate.
  - **Not yet shown.** Adam normalises per parameter, so growing gradients alone do not shrink the other groups' steps until global clipping binds, which happened only in the last 2,500 steps. The PPL crossover (about 10k–13k) predates it. The check is 6b-6 part B (routing entropy: has α collapsed?), plus the W_q/W_k norms and `relax_share` from the checkpoint and `training_log.jsonl`.
  - **Projection.** G2 fell 16.5% from step 21,000 to settled. The same decay would put G3 near 54.4, missing the key line (≤ 52.1) and landing a few percent behind G2.
  - **Pre-registered follow-up arm, G3′ (F3.2b).** G3's configuration plus:
    - QK-normalised routing for the exchange field, with a clamped learnable logit scale (the creation gate's `cgqk` scheme);
    - a per-group clip override for `relax_field` at 0.3, like the other gates.

    Predicted: the gradient norm stays flat (within ±25% of its 5,000-step value) through the stable phase, called at 70%. Settled ≤ 52.1, i.e. the exchange field adds value once hardened, called at 50%. Code: a QK-norm switch on `XiRoutedConservativeAttention`, off by default and verified bit-identical. It is not gating the v6 abstract.

    **Implemented 2026-10-05.**
    - **Code.**
      - `XiRoutedConservativeAttention(qk_norm=...)`: q and k L2-normalised over d_k, times a clamped per-head σ_h = min(exp λ_h, 100), with λ initialised at log(1/0.07).
      - Config `relax_attn_qk_norm`; Cell 0 `RELAX_ATTN_QK_NORM` (tag `rfqk`) and `RELAX_FIELD_CLIP` (tag `rfclip0p3`, its own clip group in Cell 6), plus a Cell 5b guard.
    - **Off: bit-identical to HEAD on G3's configuration** (eval and train logits, loss and all 95 gradients; `debug/verify_head_equiv_g3.py`).
    - **On: all six checks pass** (`debug/verify_g3prime_switch.py` and its output):
      - the tag is G3's plus `rfqk_rfclip0p3`;
      - the only new parameter is `relax_field.logit_scale`, one per head;
      - |score| ≤ σ ≤ 100;
      - gradients reach σ, W_q and W_k;
      - the run is causal;
      - only the five `relax_field` parameters change clip group.
    - **Watch item.** On G3's trained weights, W_v carries most of the field's gradient (4.0 against 0.1–0.4 for the others). The 0.3 group clip will be set by W_v. Per-group norms are logged, so it can be read in the run.

- **G3 scored, 2026-10-05.**

  | measure | value |
  | --- | --- |
  | settled (last three evals: 54.17, 54.35, 54.12) | **54.21** |
  | best | 52.44 at step 31,000 |
  | speed | 1.68 s/step |
  | clip hits | 9 of 650 logged steps (1.4%), max norm 1.14 |
  | SCAF audits | CLEAN at all seven; Tier A and Tier B 0. The final audit printed nan for honest/standard PPL, as G2's did |

  - **Point 51, band 46–56:** in the band; the point missed by 3.2, on the bad side.
  - **Key line, the exchange field's value under live gradients (≤ 52.1): NO.** G3 is **2.1% worse** than G2 (53.12), the same model without the exchange field.
  - **Against its parent probe** (only the field live, 61.11): −11.3%. **Against F3.1** (57.76): −6.1%. **Against GPT-2:** 1.088×.
  - **The mid-run projection** (about 54.4) was close.
  - **Gradients:** the gradient norm peaked at 0.91 (steps 22.5k–25k) and fell to 0.75 in the decay; clip hits were confined to steps 20k–27.5k.
  - **Reading.** As trained, the conservative exchange field adds nothing over no exchange field at all under live gradients. It led by 5–7 PPL early, and lost the lead as its gradient grew. Whether that is the field or its untuned routing is what **G3′** (QK-norm routing plus a 0.3 clip override) decides. The price of conservativity is still undetermined: it needs `attention` live.

- **G3 diagnostics, 2026-10-05.** Outputs are in G3's results folder; the routing probe is `debug/g3_routing_probe.py` with its output.
  - **Independent causality check on the final weights: CLEAN.** Future perturbation is 0 at five cut points, batch independence is 0, and the prefix-only leak tax is −9.5e-5 nats.
  - **6b-6 A, λ-ablation: the field is load-bearing.** λ = 0 costs **+29.46 PPL (+56%)**, λ = 0.125 costs +19.3 and λ = 0.5 costs +4.3. Yet the model trained without the field (G2) is 2.1% *better*: the cleanest ablation-against-trained-without gap yet. The field substitutes for work the model otherwise does elsewhere.
  - **6b-6 B and its force share do not read this field.** Part B (routing entropy) is written only for `DirectExchangeForce`. The "force share 0.0" is a measurement gap: `relax_share` is computed only for the non-conservative modes, and the live `attention_potential` force goes through `force_live`.
  - **The routing probe, step 500 against best (31,000), on the same tokens:**

    | | step 500 | step 31,000 |
    | --- | --- | --- |
    | W_q, W_k spectral norms per head | about 0.9–1.4 | **5.0–6.3** |
    | logit bound σ(W_q)σ(W_k)/√d_k per unit input | 0.12–0.22 | 3.7–5.6 (**about 28×**) |
    | layer 1 score std / max abs score | 1.5–3.8 / 9.5–24 | 3.9–5.1 / 23–37 |
    | layer 1 routing entropy (as a share of uniform) | 0.57–0.86 | 0.60–0.69 |
    | layer 1 max routing weight | 0.06–0.21 | 0.19–0.28 |
    | layer 0 routing | uniform (1.00) | near-uniform (0.94–0.98) |
    | field share of the total conservative force, layer 1 (median) | 1.00 | **0.92** |
    | the same, layer 0 | 0.05 | 0.32 |

    **Reading.** The logit-growth hypothesis is confirmed in the weights: the routing projections grew about 5–6× unchecked, and the attainable logit scale about 28×. The routing sharpened only moderately, though: layer 1 entropy stays near 0.6 of uniform. It did not collapse. The field carries about 92% of layer 1's conservative force, so the model became attention-dominated at its output layer, while still conservative. QK-norm (G3′) caps exactly the growth measured here.
  - **6b-9 (CG1):** R(geo) is 1.068. The reverse channel moves R by −0.759, and V_φ plus the exchange field by only −0.073. **Checked 2026-10-05: consistent.** The replay's "cons" arm keeps `f_phi`, which includes the live field's `force_live`; "geo" zeroes it. And G3's η (reverse-channel increment / conservative step) is **1.16 at layer 0 and 2.75 at layer 1** (`debug/g3_eta_output.txt`). The register path dominates the step's displacement. The field dominates the smaller conservative part (about 92% of it at layer 1), so its displacement attribution is small. Both readings stand.
  - **6b-7:** Gate 1 +25.8 (+49%). Gate 2 at N=3 **+325%** (G2: +51%), so extension fails far worse. Gate 3 at N=3 **+1,342%** (G2: +1,274%).
  - **6b-13:** θ = ω·Δt p50 is **2.07** (layer 0 1.91, layer 1 3.00). The pre-registered L=2 band [3.3, 4.2] is a **MISS**: the exchange field takes stiffness off V_θ.
  - **SR-π, a fourth point that goes against the π-crossing account.** G3's median θ (2.07, and 3.00 at layer 1) does not cross π under 1.5× refinement, yet Gate 3 is as bad as G2's. So the π crossing is not sufficient to explain refinement failure in the Fock arms; the register path, and here the field, drive it. The SR-π.1 prediction (L=8, θ ≈ 1.2, Gate 3 ≤ +100%) stays registered but is now less likely. My call drops from 60% to about 40%.
  - **6b-8:** reverse channel off **+393.7%** (G2: +300%); bank frozen +16.7%.
  - **6b-12:** uniform forcing (100% of tokens strongly forced at layer 1).

- **Not decided by this run:** the price of conservativity under live gradients needs `attention` live against it. That run is not gating. Until it runs, the abstract states the price as measured under Gen 2 only (`attention` 63.51 against `attention_potential` 80.9 detached and 61.11 with `rglive`).

**G4 (cheap, gates only a forecastability sentence). Cell 6b-10 on run 4**, the Gen 2 twin of the L=4 live arm, with the same layer 2–3 rule as the L=4 scoring. Prediction: tangential coherence below +0.526, and the integrator-velocity forecast error above 0.759. Either one meeting its threshold would credit part of the forecastability to the live gradients. If run 4 is also forecastable, the property belongs to the L=4 architecture and the abstract says so.

**What the abstract says in each case** is fixed now, so that the results fill in numbers rather than choose the story. The parity sentence quotes both GPT-2 baselines with parameter counts. The register sentence quotes G2 against F3.1. The depth sentence quotes the L=4 live arm against G2. The exchange-field sentence quotes G3 against G2. Each claim carries one seed and an untuned baseline in the same sentence.

**Order and cost (author's call, 2026-10-03):** G2 first (it gates two claims and is the longest run, about 12 h at 1.29 s/step), with G4 (minutes) alongside. G3 next, or at the same time if a second session is free (about 13–14 h). G1 is deferred until all three are in. With concurrent Colab sessions, all three trainings can run at once. They share no files: separate tags and, for G1, separate folders.

### 5.11 CB1–CB3: balancing the conservative and Fock paths — **pre-registered 2026-10-03, before any run**

**Question.** Is the pair potential (PARF's V_φ) starved when the Fock register path is present? And can a model keep most of the Fock gain while staying mostly conservative? (Author's hypothesis, 2026-10-03.)

**Evidence so far.** At L=4, V_φ's 6b-9 share of the step is −0.0035 in the Fock arm, against −0.291 in the conservative-only arm, so the two read as substitutes. G2's 6b-9 tests this at L=2: the pre-registered value in §5.10 is |share| < 0.05.

**Measure of "mostly conservative".** Any non-zero Fock force breaks strict conservativity, so the claim needs a number.

- **η** (per token, per layer) = ‖Fock increment‖ / ‖conservative step‖. The conservative step is h_new − h before the increment, LayerNorm included. Reported as the mean, p90 and max over tokens.
- **ν** (CB3 only) = the fraction of tokens whose gate is exactly 0. For those tokens the layer step is exactly the conservative step.
- η measures the magnitude of the non-gradient increment, not its curl. It is therefore an upper bound on the non-conservative share of the step.

**Baseline η, measured 2026-10-03** (`debug/verify_cb_switches_output.txt`, on the trained Gen 2 no-exchange L=2 weights, eval, 2 × 512 validation tokens):

| layer | mean | p90 | max |
| --- | ---: | ---: | ---: |
| 0 | 1.62 | 1.70 | 2.13 |
| 1 | 3.05 | 3.45 | 3.84 |

The Fock increment is larger than the conservative step at both layers, and three times larger at the output layer. The Gen 2 model is Fock-dominated by this measure. G2's own η is read from its checkpoint before any CB arm starts.

**Arms.** Each is G2's configuration (L=2, `none`, Fock on, V_φ and ξ live) plus one switch, trained from scratch for 32,500 steps. The switches were verified on 2026-10-03 (`debug/verify_cb_switches.py`):

- At the neutral setting, every switch gives logits bit-identical to G2's, in eval and in train with the same Gumbel seed.
- G2's default build is bit-identical to the committed model file, including gradients, so a running G2 can resume on the new code.
- CB3 adds 1,538 parameters and consumes no RNG.
- The CB2 bound holds for every token.
- CB3 passes the future-perturbation test.

| arm | Cell 0 | tag | what it tests |
| --- | --- | --- | --- |
| **CB1** | `REVERSE_CHANNEL_WARMUP_STEPS = 20000` (forwards; 10,000 steps at accum 2, against 2,000) | `rcw20000` | Fock arrives late. Does V_φ stay awake when it matures first? |
| **CB2a** | `FOCK_BUDGET = 1.0` | `fb1` | per token, the Fock increment is no larger than the conservative step |
| **CB2b** | `FOCK_BUDGET = 0.3` | `fb0p3` | the conservative step is at least 3.3× the Fock increment: **mostly conservative** |
| **CB3** | `FOCK_GATE_L1 = 0.02` | `fg0p02` | a learned per-token gate, g = clamp(1.2σ(w·[h, Q] + b) − 0.1, 0, 1), with an L1 penalty; the Fock path acts only where it pays |

**Fixed details.**

- **CB2's cap factor is detached.** The model gets no gradient for inflating its conservative step to buy Fock budget.
- **CB3's gate starts at g = 0.957** (b = 2, w = 0), inside the stretch, so it has gradient from step 0.
- **The budgets are fixed now** and are not chosen after G2's η is known. 0.3 is the operational threshold for "mostly conservative".

**Predictions.** "Gap" means F3.1 (57.76) minus G2's settled value. "Recovered" means the share of the gap an arm keeps, so 100% means as good as G2.

| arm | settled PPL | gap recovered | η mean (both layers) | V_φ 6b-9 share |
| --- | --- | --- | --- | --- |
| CB1 | within 3% of G2 | ≥ 80% | within 25% of G2's | \|·\| ≥ 0.15, called at **35%**: I expect the substitution to be structural, not a matter of order |
| CB2a | within 3% of G2 | ≥ 70% | ≤ 1.0 by construction | \|·\| ≥ 0.05, called at 50% |
| CB2b | — | **≥ 50%**, called at **50%** | ≤ 0.3 by construction | \|·\| ≥ 0.15, called at **65%**: V_φ wakes when Fock is capped |
| CB3 | — | ≥ 50% | ≤ 0.5 | ν ≥ 0.3, called at 40%. CB3 Pareto-dominates CB2 (lower PPL at equal η), called at 55% |

**Success criterion, the claim the book could make.** At least one arm has η mean ≤ 0.3 at both layers and recovers ≥ 50% of the gap. If so: "a model whose every layer step is at least 70% conservative by magnitude keeps half or more of the register mechanism's gain."

**Stop rule.** If G2 settles at or above 56.6 (within 2% of F3.1), the Fock path adds nothing under live gradients, and the series does not run.

**Readout per arm:**
- settled PPL;
- η per layer (mean, p90, max), logged every 50 steps as `cb=[...]` and in `training_log.jsonl`;
- ν (CB3);
- 6b-9 (V_φ share), 6b-7 (the gates) and 6b-13;
- SCAF at every 5k audit, and the independent causality check before any publication.

**Order and cost:** after G2, G3 and G4.
1. η and 6b-11 (the inference-time slider) on G2's checkpoint, which are free. The slider understates a trained arm's gain (the Gen 2 ablation overstated the Fock path's value by 3–4×), so it is read only as the shape of the curve.
2. CB2b, the key arm.
3. CB1.
4. CB3.
5. CB2a.

About 15 h each at L=2.

### 5.12 CG8: does the geometry predict the model's own errors? — **pre-registered 2026-10-03, before any measurement**

**Why.** Book §18d's new subsection (`subsec:geom-requirements`) separates three properties: a conservative step (C), refinement invariance (R) and settling (S). Each geometric capability is the model's own only if the properties it reads are present. SR1–SR4 target (R) and (S); CB1–CB3 target (C). CG8 asks whether securing them is worth anything: does the trajectory's geometry carry information about the model's own errors that the output distribution does not? It is a token-level proxy for the hallucination detector (§18d Experiment G2), and it needs evaluation only.

**Measurement** (one new 6b cell, planned as 6b-14):

- **Tokens:** 32 × 512 validation tokens, with a fixed seed shared by every arm (as in 6b-13).
- **Event:** the top-1 prediction at token t is wrong. Secondary event: the NLL at t is above the arm's median.
- **Signals per token** (summed over layers unless stated):
  - **s_E**, the energy anomaly |ΔE_obs − ΔE_expected|. Here H = ½ m‖v‖² + V_θ(ξ, h) (+ V_φ when present), and ΔE_expected = γ‖v‖²Δt (§18d eq. energy-anomaly).
  - **s_G**, the deflection of the full step from the damped V_θ geodesic step, as in CG6 (6b-12).
  - **s_η**, Fock arms only: the per-token η = ‖reverse-channel increment‖ / ‖conservative step‖, maximum over layers.
  - **Reference:** the softmax entropy H_soft at t.
- **Score:** ΔAUROC(s) = AUROC(logistic[H_soft, s]) − AUROC(logistic[H_soft]), by 5-fold cross-validation over sequences, not tokens, with a 1,000-sample bootstrap CI. AUROC(s) alone is also reported.
- **Arms:**
  - the baselines F3.1, G2 and the L=4 live arm (measured first, free);
  - SR1–SR4 against F3.1;
  - CB1–CB3 against G2.

**Predictions:**

| arm(s) | signal | prediction | called |
| --- | --- | --- | --- |
| baselines (F3.1, G2, L=4 live) | s_E, s_G | ΔAUROC < 0.01: at the trained step size the geometry adds nothing to entropy | 60% |
| G2, L=4 live | s_η | ΔAUROC < 0.01 | 55% |
| SR2 (exact damped flow) vs F3.1 | s_E | ΔAUROC higher by ≥ 0.01, because the phase noise of Prop 44 is removed | 35% |
| SR4a (variable N at fixed T) vs F3.1 | s_G | ΔAUROC ≥ 0.02 | 35% |
| CB2b (ρ = 0.3) vs G2 | s_G | ΔAUROC higher by ≥ 0.01 | 40% |
| CB3 | s_G on tokens with g = 0 vs g > 0 | AUROC(s_G) higher on the exactly-conservative tokens | 50% |

**Decision rule.**

- **Any arm with ΔAUROC ≥ 0.02 (CI excluding 0)** for a geometry signal: this is the first evidence that the Lagrangian geometry carries information the output distribution does not. Name the property that arm holds; that property is the one worth paying for.
- **The baselines already ≥ 0.02:** the geometry is informative even at the trained step size. (C)/(R)/(S) then matter for interpretation, not for the signal.
- **No arm reaches 0.01:** the §18d capability claims remain unshown on the CfC/BAOAB family. The book says so in §18d and in the abstract's scope.

**Caveat recorded in advance.** Next-token top-1 errors on OpenWebText are mostly ambiguity, not hallucination. CG8 tests whether geometry tracks error at all, not factuality. Factuality needs the QA protocol of §18d Experiment G2.

### 5.13 FO series: does OpenWebText need second-order training where TinyStories did not? — **pre-registered 2026-10-03, before any measurement**

**Question** (author's, 2026-10-03). On TinyStories with Verlet, Fock-PARFLM trained first-order matched second order: Fock-G1 8.95 against 9.04, d=256, L=8, one seed. Does the same hold on OpenWebText, or does that corpus need the second-order dynamics? Which corpus statistics separate the two cases?

**Theory.** `Corpus_Statistics_and_the_First_vs_Second_Order_Well_Gap.md`, the order-gap master inequality, with three corpus channels:

- **(A) predictive information I_pred**, through the per-layer step and the anharmonicity A = s̄·√λ_max;
- **(B) long-range dependence**, through the conditioning κ of the ξ filter bank and the force variation across the momentum window;
- **(C) Zipf**, which affects the noise floor only.

Its prediction #1 orders the gap: Markov < TinyStories < OpenWebText < code.

**What has changed since that note.**

- Its §11 explained the TinyStories null mainly by a realized damping of γ_geo ≈ 0.965, worth about 26× suppression. γ_geo has since been withdrawn as an artefact of the residual (book Remark 102).
- The CfC models measure the opposite regime: Gate 1 +34% (F3.1) and +55% (L=4), inertial fraction 1.41, speeds rising from 6.7 to 15.0 across layers, and stiff modes at ζ ≈ 0.05.
- The TinyStories null also carries three confounds besides the corpus: Verlet, detached gradients (Gate 1 there would be about +5%), and d=256.

**Scope.** Every reading in this series is **at the current damping regime**: constant γ = 0.1 on the CfC/BAOAB low-rank propagator, stiff modes at ζ ≈ 0.05, live gradients, L=2 with T = 8. The master inequality scales the gap as about γ_eff⁻³, so heavier damping (γ(h), SR3's ζ* = 1) would shrink it. Results are reported with the realized damping alongside and are not extrapolated past it.

#### Stage 0, free (no training). Corpus side, with identical procedures on both corpora

Data: TinyStories `tinystories_gpt2_1files_5000000toks.npz` (GPT-2 BPE), and 5M contiguous tokens of OpenWebText train (GPT-2 BPE). Blocks of 512, as in training. One fixed embedding for both corpora (GPT-2 `wte`), so the comparison is about the corpus, not a model.

| id | statistic | channel | prediction (OWT relative to TinyStories) | called |
| --- | --- | --- | --- | --- |
| C1 | token types used; Zipf exponent α (ranks 10–10⁴) | C | OWT ≥ 3× the types; α within ±0.15 | 80% |
| C2 | **I_pred proxy** = H_unigram − H_model, in bits/token. H_unigram is the plug-in unigram entropy; H_model is the best trained model's val loss on that corpus (OWT: matched GPT-2, 49.81; TinyStories: the second-order anchor, 9.04) | A | OWT higher, by ≥ 0.5 bit | **55%**: TinyStories' low conditional entropy may make the two comparable, contrary to the note's assumption |
| C3 | token-repetition autocorrelation R(τ) = P(x_t = x_{t+τ}) − Σp², τ = 1…256, within blocks | B | **TinyStories higher** at τ ≤ 64 (names and phrases repeat within a story) | 70% |
| C4 | embedded-stream autocovariance C_e(τ)/C_e(0) (centred `wte`); half-decay lag and tail slope over τ = 8–256 | B | TinyStories decays more slowly over τ ≤ 64 | 60% |
| C5 | ξ filter-bank Gram G for the ladder's α = (0.5, 0.75, 0.95, 0.99, 0.995) on the embedded stream: κ(G) and effective rank (participation ratio) | B | TinyStories κ ≥ OWT κ | 55% |

The C3–C5 predictions run **against** the note's simple ordering. If they hold, Channel B does not favour an OWT gap, and any OWT-specific gap must come through Channel A or through the model's operating point.

#### Stage 0, free. Model side, on the trained second-order models of each corpus

Models:
- **TinyStories:** the second-order anchor (`semsimula_fock_aniso_gaussian_fockreg_tinystories/results/seed0_gamma=0.3`, 9.04) and the Fock-G1 checkpoint (8.95).
- **OWT:** F3.1 (conservative-only live, L=2) and the L=4 live Fock arm, plus G2 when it lands.

Measured on each model's own validation tokens, with ξ frozen at the layer's own value:

| id | statistic | prediction | called |
| --- | --- | --- | --- |
| M1 | **anharmonic fraction** per token and layer, ε = ‖f(h+Δh) − f(h) − H(h)Δh‖ / ‖f(h+Δh) − f(h)‖. Here f = −∇_h V_θ, H is its Hessian (one Hessian-vector product), and Δh is the layer's actual step. This is the model-agnostic form of the note's anharmonicity gate: ε ≪ 1 means the force is linear across a step, so second order is absorbable | median ε: OWT models ≥ 2× the TinyStories anchor | 65% |
| M1′ | the TinyStories anchor alone: median ε < 0.2 (structural-sufficiency regime) | — | 55% |
| M2 | **inertial share** of each layer step, ‖Φ(h, v) − Φ(h, 0)‖ / ‖Φ(h, v) − h‖, using each model's own layer step with and without the incoming velocity | OWT CfC models ≥ 0.5 at the last layer; TinyStories anchor ≤ 0.3 | 60% |

**Caveat recorded in advance.** M1 and M2 compare the operating points of the two model families: Verlet, d=256, L=8, Δt=1 against CfC, d=384, L=2, Δt=4. They are not a pure corpus comparison, which is why Stage 1 exists.

**Stage-0 decision rule (§12 of the note).**

- **ε ≪ 1 on the OWT models** (median < 0.1, p90 < 0.3): first order is structurally sufficient at this operating point. Stage 1 runs only FO-OWT, as confirmation.
- **Otherwise:** the full Stage-1 2×2 runs.

#### Stage 1, trained: the 2×2 that separates corpus from setup

| | second order | first order (FO-a, memoryless) |
| --- | --- | --- |
| OWT, Fock 'none', L=2, CfC, live (G2's configuration) | **G2** (running) | **FO-OWT** |
| TinyStories, the same architecture, propagator, convention, d and L; **16,250 steps** (amended 2026-10-04, below) | **SO-TS** | **FO-TS** |

**FO-a** is the Fock-G1 definition carried over unchanged: h_prev := h at every layer, so no velocity crosses a layer boundary. Every other channel (V_θ, V_φ, ξ, registers, reverse channel, regularisers) is identical. On the CfC propagator each step then starts from rest. It is the trained counterpart of Gate 1, and it needs one Cell-0 switch with the same bit-identity verification as the CB switches (identical to G2 when off).

**FO-b** is a true overdamped first-order Langevin step, with exact exponential relaxation of the low-rank stiff modes. It is a second, conditional arm: run only if FO-a shows a gap, to separate inter-layer memory from within-step inertia. It needs new integrator code.

**Statistic.** Δ_corpus = (settled PPL_FO − settled PPL_SO) / settled PPL_SO, per corpus. The test is the interaction Δ_OWT − Δ_TS.

| reading | prediction | called |
| --- | --- | --- |
| Δ_OWT | ≥ +5% (second order earns it on OWT) | 60% |
| Δ_TS | within ±3% (the Fock-G1 null reproduces under CfC and live gradients) | 55% |
| interaction Δ_OWT − Δ_TS | ≥ 4 points | 50% |
| FO arms, Gates 2 and 3 | pass by construction (no momentum, so no phase-dependent dissipation); reported as the perplexity cost of perfect settling | — |

**Outcomes.**

- **OWT gap without a TinyStories gap:** the corpus separates them, and Stage 0 says which statistics carry it.
- **Gaps on both:** CfC and live gradients, not the corpus, make second order matter. The TinyStories null was a property of the Gen 1 setup.
- **No gap on either:** first-order training suffices even on OWT at this damping regime. That supports a train-first-order / infer-second-order recipe, and bears on the book's "minimal structural commitment" claim for training.

**Future prediction, the γ(h) link** (recorded now, to be tested only once a trained γ(h) arm exists). Where learned damping is high, a token is effectively overdamped (first-order); where it is low, the token keeps its inertia. So a trained γ(h) is a local measurement of where second order earns its keep. Predicted:

- the learned γ(h) is higher on tokens whose local ε (M1) is low and whose context mixes fast;
- the fraction of locally overdamped tokens (ζ_local > 1) is lower on OWT than on TinyStories.

#### Stage 0 results, 2026-10-04 (scored against the predictions above)

Outputs: `notebooks/conservative_arch/first_order_ablation/fo_series/results/` (`stage0_corpus_output.txt`, `stage0_c4_corrected_output.txt`, `stage0_models_output.txt`, plus the JSON files).

**Disclosed correction.** C4 as specified centred the GPT-2 embedding on the *vocabulary* mean. It measured a constant offset (flat at about 0.28 for TinyStories and 0.22 for OWT from τ = 1 to 256), not dependence. It was re-run, centred on each corpus's own frequency-weighted mean, and only that version is scored. C5 centres each channel on its own sample mean and was unaffected.

**Corpus side:**

| id | TinyStories | OpenWebText | prediction | scored |
| --- | --- | --- | --- | --- |
| C1 types | 12,742 | 47,787 (3.75×) | OWT ≥ 3× | **hit** |
| C1 Zipf α | 1.98 | 1.01 | within ±0.15 | **miss**: TinyStories' frequency distribution is far steeper |
| C2 I_pred proxy (H_uni − H_model) | 5.38 bits | 5.24 bits | OWT higher by ≥ 0.5 | **miss**: equal within 0.15 bit |
| C2 held-out bigram drop (H_uni − H_bi) | 3.10 bits | 1.72 bits | (not predicted) | — |
| C3 repetition R(τ ≤ 64) | lower or equal at most lags | higher or equal | TinyStories higher | **miss** |
| C4 (corrected) embedded autocorrelation | ≈ 0 at τ ≤ 4 (slightly negative), 0.008 at τ = 8–32, 0.0016 at 256; tail slope −0.44 | 0.012 at τ = 1, 0.008 at 256; tail slope −0.19 | TinyStories decays more slowly | **miss**: OWT carries long-range positive correlation. Removing each block's own mean erases most of it, so it is topic persistence |
| C5 κ(G) / κ(corr) / effective rank | 1,022 / 133 / 2.21 | 1,243 / 173 / 2.06 | TinyStories κ ≥ OWT | **miss**: OWT's channels are more collinear |

**Reading.** My contrarian predictions for Channel B (C3–C5) all failed. The note's original expectation holds: OpenWebText has more long-range dependence and a worse-conditioned ξ filter bank. Channel A, as I operationalised it (total predictive information), does **not** separate the corpora. Where that information lives does, though this is post hoc and needs confirming:

- **TinyStories:** 58% of its predictive information is available from the previous token alone (3.10 / 5.38).
- **OWT:** only 33% (1.72 / 5.24).

OWT's predictability sits at longer range. That is exactly what must be carried across positions and layers by the ξ channels and, in the second-order models, by the momentum.

**Model side** (the reproduction guard passed; on 6 × 2 × 512 validation tokens):

| model | PPL here | its checkpoint |
| --- | --- | --- |
| TinyStories second-order | 8.68 | 9.04 |
| TinyStories Fock-G1 | 8.79 | 8.95 |
| OWT F3.1 | 65.5 | 57.76 |
| OWT L=4 | 55.1 | 50.10 |

The OWT figures sit about 12% high on 12 short sequences, consistent with eval-sample noise at this tiny size.

| model | M1 ε, interior layers (median) | ‖Δf‖/‖f‖ across a step | step ‖Δh‖ | M2 inertial share (median, per layer from 1) |
| --- | --- | --- | --- | --- |
| TinyStories second-order (Verlet, d=256, L=8) | **0.000** at layers 1–7 (p90 ≤ 0.49) | **0.000** | 0.9–2.3 | 0.92, 0.59, 0.56, 0.62, 0.67, 0.69, 0.70 |
| TinyStories Fock-G1 | 0.000 at layers 1–7 | 0.000 | 1.4–2.3 | 0 by construction |
| OWT F3.1 (CfC, L=2) | **6.13** (layer 1, the output step) | 1.01 | 20.7 | 0.50 |
| OWT L=4 live (CfC) | **0.32, 0.46** (layers 1–2); 6.17 (layer 3, the output step) | 0.62, 0.97 | 2.6, 3.2 | 0.68, 0.59, 0.24 |

Layer 0 reads ε ≈ 6–15 in every model. That step is the projection of the embedding onto the LayerNorm sphere (‖Δh‖ ≈ 15–18), not dynamics, so it is excluded from the comparison.

**Scored:**

- **M1: hit.** The OWT interior median ε is 0.3–6 against 0.000 for TinyStories, far beyond 2×.
- **M1′: hit.** The TinyStories anchor's interior ε is below 0.2.
- **M2: miss**, and the miss is informative. The TinyStories second-order model carries a *large* inertial share, 0.56–0.92 at every layer, larger than the OWT models (0.24–0.68), and yet first order matched it.

**What Stage 0 says.**

1. **On TinyStories, the V_θ force does not change across a layer step**: ‖Δf‖/‖f‖ ≈ 0. A constant force makes the inertial displacement exactly reproducible by a first-order step with a rescaled step size. That is the note's absorbable case (§6.2), seen directly, and it explains the Fock-G1 null *mechanistically*, not as a power failure. A large inertial share does not imply that second order is needed. M2 is not the discriminator; M1 is.
2. **On the OWT CfC models, the force changes by 60–100% across a step** (ε 0.3–6). Second order is not absorbable at this operating point, and the §12 sufficiency criterion fails.
3. **The confound is architectural as well as corpus-level.** Both families cap the diagonal well precision at 2/d. The OWT ladder adds a joint V_θ bank and a stiff low-rank precision channel (`PRECISION_LR_MAX = 1.0`), integrated exactly by the CfC propagator, and its L=2/L=4 steps are larger. Stage 0 cannot tell whether OWT *requires* stiff, anharmonic wells or whether this architecture merely *permits* them.

**Stage-0 decision.** The OWT ε median is far above 0.1, so **the full Stage-1 2×2 runs.** One reading is added to it, pre-registered now: M1 and the corpus statistics on **SO-TS** (TinyStories trained in the CfC architecture).

- If SO-TS also learns anharmonic wells (interior ε ≥ 0.3), the architecture, not the corpus, produced the OWT stiffness. FO-TS then predicts a gap as well, called at 50%.
- If SO-TS stays near-linear (ε < 0.1) under the same architecture, the corpus is the separator, and the **long-range share of predictive information** (post-hoc C2b) and the long-range embedded correlation (C4) are the candidate statistics. A third corpus would confirm them.

**Order and cost.**

1. Stage 0 now (local, minutes).
2. Stage 1 after G2–G4 and ahead of the CB and SR series: FO-OWT at 32,500 steps (about 13–15 h); SO-TS and FO-TS at 16,250 steps (about 7 h each). That is about 28 GPU-h serial, or about 15 h wall time with three sessions in parallel. SO-TS and FO-TS reuse the ladder notebook with a TinyStories data path.

**Amendment, 2026-10-04, before any Stage-1 run: the TinyStories schedule and a settling check.**

- **Schedule.** SO-TS and FO-TS train for **16,250 steps** (266M tokens, about 0.6 epoch of the full TinyStories training set), not 32,500. The WSD schedule keeps its fractions of the run.
  - **Why.** The test statistic compares first order with second order *within* each corpus, so the two TinyStories arms need to match each other, not G2's budget. TinyStories also saturates far sooner: the August second-order model reached 9.04 on 164M tokens.
  - **Data.** 266M tokens still needs the full TinyStories token cache. The local 5M-token cache, which the August runs repeated about 33 times, is not enough.
- **Settling check, applied to both TinyStories arms before scoring.** Fit a line to the evals over the last 3,000 steps. Both arms are "settled" if the fitted change over that window is within the eval noise, estimated as the standard deviation of the residuals about the fit.
  - **If either arm fails:** extend **both** by the same number of steps, from their checkpoints. Continue the decay at the floor learning rate, so the schedule for the extra steps is identical in both. Re-check.
  - **If both arms settle at 16,250:** the arms are scored there.
- **Disclosure.** With different budgets per corpus, the interaction Δ_OWT − Δ_TS compares relative gaps measured at different training lengths. First-order arms can lag at short budgets, which would inflate Δ_TS. The settling check is what guards against that. If an extension is needed, both budgets are reported with the result.
3. The FO-a switch is implemented and verified before Stage 1.

## 6. Open risks

### 6.1 At L=1 the register bank is read but never updated — **2026-09-24**

> Full treatment, with figures and the depth-dependence of the mechanism's
> capacity, in
> [`Fock_Mechanism_Efficiency_Across_Layer_Depth.md`](Fock_Mechanism_Efficiency_Across_Layer_Depth.md).

**Scope, corrected.** An earlier draft of this section claimed the whole Fock
mechanism is inert at L=1 and, in a further draft, at L=2 as well. Both were
wrong, and wrong the same way: they rested on gradient measurements taken on
a **freshly built** model, where the register-to-token path is gated shut by
construction. The correct, narrower finding is below.

Registers reach the tokens by exactly one route,

```python
Q_force   = self.reverse_ch(h_new, r_rev, active)
increment = (dt*dt / m_b) * tanh(reverse_channel_scale) * warm * Q_force
```

and **both gate factors are zero at initialisation**:
`reverse_channel_scale` is `nn.Parameter(torch.zeros(...))` so
`tanh(.) == 0`, and `warm = reverse_warmup_step / 4000 == 0`. Any
gradient probe on an untrained model therefore reports "registers do nothing"
at every depth. With both gates set to their trained values
(`tanh(scale) ~ 0.017`, warmup complete):

| | `register_embed` | `creation_gate_qkv` |
| --- | ---: | ---: |
| L=1, gate shut (fresh init) | 0.000e+00 | 0.000e+00 |
| **L=1, gate open (trained)** | **3.370e-02** | **0.000e+00** |
| L=2, gate shut (fresh init) | 0.000e+00 | 0.000e+00 |
| **L=2, gate open (trained)** | **2.185e-02** | **2.815e-03** |

**At L=2 the mechanism works.** Both the bank and the creation gate receive
next-token gradient. Nothing in §5 needs re-describing.

**At L=1 the bank is read but never updated.** `register_embed` takes
gradient — the reverse channel reads it and it does affect predictions — but
`creation_gate_qkv` sits at exactly zero. The cause is separate from
the gate and survives: `_init_registers` sets `salience = 1.0`, so

```python
blend = salience.unsqueeze(-1)               # 1.0 at layer 0
r = blend * r + (1.0 - blend) * readout      # (1 - 1.0) == 0
```

annihilates the creation readout at layer 0. The gate's only other exit,
`alpha_max -> salience -> active`, runs through `_active_mask`,
a boolean comparison with no gradient. At L>=2 later layers (whose salience
has decayed) train the shared module; **at L=1 there is no later layer.**

So L=1 runs with a **static** register bank: read by the reverse channel,
frozen at `register_embed`, with the creation gate untrainable.

#### This is broader than L=1

**Layer 0 never trains the creation gate at any depth.** The gate is one
shared module, so at L>=2 the later layers cover for it and the effect is
invisible. L=1 removes the cover rather than introducing the problem.

#### A knob exists, opt-in and default-inert

`register_salience_init` (added 2026-09-24, default **1.0**) sets the
starting salience. At the default it reproduces the historical behaviour
bit-exactly — verified against pre-change measurements, so the three
completed arms and their checkpoints still correspond to the code that made
them. Below 1.0 it opens `(1 - blend)` and the creation gate becomes
trainable in a single layer: 9.11e-06 at 0.9, 8.52e-05 at 0.5, 3.41e-04 at
0.25. `0.5` is the principled choice — one decay step from 1.0 at the live
`register_salience_decay = 0.5`, i.e. the floor of what layer 1 sees
at L=2 — and it stays well clear of the 0.005 activity threshold. Range is
checked in `__init__`; nine tests in
`test_register_salience_init.py` pin both the default and the fix.

It is **not** a causality risk: it scales a position-independent mixing
coefficient, touching no mask and no readout path.

#### What it costs this programme

**Run 7 still cannot cleanly answer the question it was queued for.** L=1 was
meant to isolate whether the second-order velocity state is load-bearing —
`h_prev = h0` gives `v == 0` at the only layer, verified in running code
(`decode_velocity` called once, `max|h - h_prev| = 0.000e+00`, against
L=2's second layer at 9.748e-02). It also has a frozen creation gate, so it differs from
L=2 in **two** ways and no endpoint can be attributed to either. The second
difference is narrower than the earlier draft claimed — a static bank, not an
absent mechanism — but it is still a second difference.

Let it finish — it is still a legitimate ladder point, and "what does this
architecture do at depth 1" is a question the ladder wants answered. Just do
not read it as a velocity result.

**The clean velocity test is gate 1 of
[`Composing_Single_Layer_Inferences_Flow_or_Maps.md`](Composing_Single_Layer_Inferences_Flow_or_Maps.md)**
— reset-versus-carried at N=L, which sets `h_prev = h` so `v == 0` with
registers working and depth unchanged. Minutes of evaluation against seven
hours of training.

**A comparable L=1 arm** can now be had with
`register_salience_init = 0.5`, but it is a *different architecture*
from the L>=2 rungs and voids §2 if placed on the ladder. Use it outside the
ladder — as the instrument for
[`Composing_Single_Layer_Inferences_Flow_or_Maps.md`](Composing_Single_Layer_Inferences_Flow_or_Maps.md),
which is about composing a single trained layer — and keep run 7 as the
ladder point, interpreted with the static-bank caveat.


- **`GRAD_CLIP = 1.0` is a depth-dependent intervention, and it is not
  recorded.** It fires on 4.2% of steps at L=2 `'none'` and 36.2% at L=8 —
  nine times the rate, at a *lower* LR — so part of what this ladder
  attributes to depth is the clip. §7 of
  [`Hyperparameter_Tuning_Checklist.md`](Hyperparameter_Tuning_Checklist.md)
  has the table, the caveats and a pre-registered rule; **from L=4 onward
  every ladder point must report its clip-hit rate** in §5.

- **lambda is uncalibrated for the potential arm.** For `'attention'` it
  scales a force; for `'attention_potential'` a potential whose gradient is
  the force. Different units, no principled match, and `relax_share` is not
  recorded on the potential path — `bproj_sig` is the only proxy, and Cell
  6b-6's lambda-ablation at the endpoint is the real measurement.
- **The resonance monitor reports nothing under this integrator.** It patches
  `cfc_substep`; `baoab_cfc_lowrank` calls `lowrank_cfc_substep`. Patched in
  `semsimula-diag` 2026-09-20, but a Colab session that cloned earlier will
  print the EMPTY SUMMARY warning instead. Pull before the next run.
- **No stability check for the explicit terms.** Even once the monitor
  reports, under `baoab_cfc_lowrank` its `omega*dt` is a stiffness reading,
  not a stability wall — the low-rank modes are integrated exactly. `V_phi`
  and any relaxation kick stay explicit and this probe does not see them.
- **`hold` subsamples when coarsening.** Relevant to Cell 6b-7's N < L
  direction: L=8 to N=4 visits codes `[0,2,4,6]` and never 1,3,5,7.
  Refinement is a true refinement; coarsening is not its mirror image.

## 7. Publication readiness of the five completed arms — audit **2026-09-27**

Audited the five local experiment folders that hold the L=2 ladder plus the
matched baseline, against what a HuggingFace repo per arm needs. Folder names
in `hf_model_cards/_ladder/ladder.json` (`gdrive` field) match the folders on
disk exactly for all five; the two queued arms (runs 10 and 11, §3.2) have no
data yet.

### 7.1 What is complete

**Every arm's headline number is recomputable from the artefacts it ships.**
Recomputing `mean(last three val_ppl)` from each `results/training_log.jsonl`
reproduces `ladder.json`'s settled value to the second decimal for all five:

| arm | best step / PPL in ckpt | last three evals | settled | ladder.json |
| --- | --- | --- | ---: | ---: |
| `gpt2-matched` | 32,500 / 49.76 | 49.87, 49.80, 49.76 | 49.81 | 49.81 |
| `attention` | 31,000 / 61.49 | 63.52, 63.58, 63.42 | 63.51 | 63.51 |
| `attention_potential` | 31,000 / 79.14 | 80.74, 80.46, 81.50 | 80.90 | 80.90 |
| `none` | 31,500 / 66.56 | 66.56, 66.75, 67.63 | 66.98 | 66.98 |
| `none` no-RC | 31,000 / 85.90 | 87.53, 87.44, 88.82 | 87.93 | 87.93 |

Also verified present and correct:

- **Two stripped checkpoints per arm** — `_best.published.pt` and
  `_step500_best.published.pt`, the pairing agreed earlier (endpoint plus the
  earliest usable point, so a reader can see where the run started). No
  optimizer state in any of them; Fock arms 293–295 MB, GPT-2 137 MB.
- **Full config inside every Fock checkpoint** under `model_cfg` (112–113
  fields) plus `train_cfg` (25). The arm-defining knobs read correctly:
  `force_relaxation` is `none`/`attention`/`attention_potential` on the three
  arms that vary it, and `reverse_channel` is the *only* model-config
  difference between the `none` and no-RC folders (`True` vs `False`, plus a
  `register_salience_init` default and the logfreq path). `d=384`, `L=2`,
  `xi_channels=5`, `top_k=16`, `integrator='baoab_cfc_lowrank'`, lr 1.2e-3,
  batch 16 × grad-accum 2 on all four.
- **Complete training logs** — 721/722 JSONL rows spanning steps 50–32,500,
  65–66 evals, carrying `relax_lambda` and `relax_share` alongside PPL. The
  GPT-2 log has 227 rows over 200–32,500 with both context lengths.
- **Console logs** for all four Fock arms and, after this audit, for GPT-2
  (`matched_gpt2_training_output.txt` was loose in `~/Downloads` and is now
  filed into that arm's `results/`).
- **Probe logs** for three arms: 6b-7, 6b-9 and 6b-12 outputs sit in the
  `attention`, `attention_potential` and no-RC folders.

**Parameter counts in `ladder.json` are right and should not be "corrected".**
Summing `numel` over each `model_state_dict` exceeds the card's count by a
fixed amount because state dicts carry buffers: +50,257 (`logfreq_surprisal`)
plus three scalars on every Fock arm, and +2,097,152 on GPT-2 (eight
262,144-element causal masks, one per block). Net of buffers the counts match.

### 7.2 The four gaps

> **CLOSED 2026-09-27.** Gaps 1 and 2 below were filled in one session of the
> shape §7.2a describes. Five logs now sit in that arm's `results/`
> (6b-7, 6b-9, 6b-10, 6b-11, 6b-12) plus the two figures in `checkpoints/`.
> Every header reads `step 31500  ppl 66.56  missing 0  unexpected 0`. 6b-7
> and 6b-9 reproduce their recorded numbers exactly (Gate 1 +50.5%, Gate 3
> 435.61 at N=8, R(geo) = 1.0855). 6b-10 and 6b-11 produced the better
> readings §7.2a predicted they would, and one recorded verdict changed as a
> result: E3's coherence metric moves from "invalid at L=2" to **refuted**
> (geodesic notes §6.8). 6b-12 was a first run and **met its pre-registered
> band on all three criteria**. Gaps 3 and 4 stand. The text below is kept as
> written, since it is what the audit found.

1. **The `none` arm ships no probe logs.** 6b-7 and 6b-9 were run on it
   (R(geo) = 1.09, gates 4.32×/6.35×, recorded in
   `Geodesic_Experiments_with_CfC_BAOAB.md` §4.7 and
   `Composing_Single_Layer_Inferences_Flow_or_Maps.md` §8.2) but the console
   output was never filed; 6b-12 (F1) has never been run on this arm. The
   numbers are recorded in the notes, so the card can cite them, but the arm
   with the *most* prose written about it is the one whose probe output cannot
   be re-read. Remedy and its true cost in §7.2a: not 20 minutes, one session
   of 45 to 75 minutes that closes this gap and the next one together.
2. **E3 (6b-10) and E5 (6b-11) outputs are not filed anywhere.** Both ran and
   both are written up (§6.8 and §11.6 of the geodesic notes). Both ran on
   *this same* `none` arm, which is why §7.2a treats gaps 1 and 2 as one job.
   Note that neither cell is the cell that produced those write-ups any more.
3. **`gpt2_baseline_summary.json` carries a stale cross-reference.** Its four
   `fock_*` fields and `ratio_fock_to_gpt2_settled = 1.6378` are not a ladder
   arm: the file was written 2026-09-21, before any L=2 arm existed, and those
   numbers come from the pre-ladder **L=8** run
   (`fock_cfc_baoab_joint_vtheta_qknorm_annealed_after_28500_...`, lr 3e-4,
   read at step 32,500 of a 100k schedule), whose own last-three mean is
   81.49. The ratio is close to the `attention_potential` arm's 1.624 by
   coincidence, which makes it a trap rather than a harmless leftover. A
   `NOTE_summary_json_fock_fields.md` now sits beside it in that folder,
   relabelling the fields as a matched-depth side note. Worth keeping *as*
   that side note: at matched depth L=8 the ratio was 1.638, against the L=2
   `attention` arm's 1.275 — depth 8 at lr 3e-4 was further from GPT-2 than
   depth 2 at lr 1.2e-3, which is why the ladder was rebuilt at L=2.

4. **The second checkpoint is not the same kind of checkpoint on every arm.**
   The four Fock arms ship step 500 (PPL 478–491, essentially untrained — an
   initialisation reference, not a learning-curve point). The GPT-2 folder has
   no step-500 checkpoint at all; its second file is step 25,000 at PPL 52.87,
   already within 6% of its endpoint. So "the early checkpoint" means two
   different things across the collection, and anyone diffing first-checkpoints
   arm-by-arm will be comparing an init to a near-converged model. Either state
   the asymmetry on the GPT-2 card, or drop its step-25,000 file and ship the
   baseline as endpoint-only.

Also quarantined: the superseded `f1_forcing_per_token_WRONG.png` in the no-RC
folder's `checkpoints/` (from F1's first, tautological run) was moved to
`results/` under a `_DO_NOT_PUBLISH_` prefix, and the corrected plot renamed
`F1_6b12_per_token_deflection.png` to match its log.

### 7.2a What closing gaps 1 and 2 actually costs

Both gaps are the same job. E3 (6b-10) and E5 (6b-11) were run on the `none`
arm, not on some other arm, so all four missing logs belong to one
configuration and one Colab session closes them.

**Nothing is retrained.** Every probe resolves its own checkpoint as
`CKPT_DIR / f'{CKPT_PREFIX}_best.pt'`, loads it into the built model, and
restores the previous weights on exit. Cell 6 is not needed. Cell 2 is
read-only. Run order:

    Cell 0 (config for this arm)  ->  1  ->  1b  ->  2  ->  3  ->  4  ->  5
    then 6b-9, 6b-7, 6b-12, 6b-11, 6b-10

Probes in that order rather than numeric order: 6b-9 is the cheapest and
verifies the load, the tag and the gate readings before anything expensive
runs, and 6b-10 is last because it is the only one that also needs the matched
GPT-2 checkpoint present on Drive. 6b-11 and 6b-12 write their figures into
`CKPT_DIR`, so collect those alongside the logs.

Skip 1c and 1d (no-ops here: nothing they assign is read by any later cell),
skip 5b (the from-scratch guard protects a training run), skip 6.

**Cell 0 as committed already IS this arm.** Reconstructing `_variant_tag`
from the committed values reproduces the Drive folder's tag character for
character, so the work is confirming the Colab copy has not drifted from the
`attention_potential` run and the two gate/init-scale probes that followed it.
Values to verify, most-recently-disturbed first:

| knob | value | tag effect if wrong |
| --- | --- | --- |
| `LADDER_MECHANISM` | `'none'` | `noattn` becomes `attnpot`/`attn` — **was last set to `'attention_potential'`** |
| `RELAX_INIT_SCALE` | `0.02` | adds `ris0p055` — the init-scale probe set this |
| `RELAX_GATE` | `'scalar'` | **silent on this arm**: the `zro` token is gated on `FORCE_RELAXATION != 'none'`, so a stale `'zero_readout'` neither shows in the tag nor changes the model here. Harmless, but the tag cannot warn you |
| `REVERSE_CHANNEL` | `True` | adds `norc` and resolves the *other* folder |
| `LADDER_L` | `2` | `L2probe` |
| `LADDER_T` / `LADDER_DT` | `8.0` / `None` | dt = 4, giving `idt4` |
| `LADDER_LR` | `1.2e-3` | `lr0p0012` |
| `LADDER_STEPS` | `32_500` | not in the tag, but sets the WSD windows |
| `WSD_STABLE_FRAC` | `0.60` | adds `sf...` |
| `LADDER_SALIENCE_INIT` | `1.0` | adds `sal...` |
| `LANGEVIN_T` | `0.0` | adds `T...` |

`PROBE_MAX_STEPS` and `ANNEAL_PROBE` do not matter: nothing trains.

The two lines to read back before running any probe:

    [variant] tag=xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn
    CKPT_DIR    = /content/drive/MyDrive/semsimula_fock_cfc_baoab_owt_<that same tag>/checkpoints

**This is the only thing that can silently go wrong.** The probes locate the
checkpoint through the tag and load it with `strict=False`, so a drifted Cell 0
finds either nothing or the wrong file and reports a partial load rather than
failing. The guard is already printed: every probe header must read
`step 31500  ppl 66.56  missing 0  unexpected 0`.

No edits are needed inside any probe. Each defaults its own checkpoint to
`CKPT_DIR / f'{CKPT_PREFIX}_best.pt'`, and 6b-12's `R12_PRIMARY` auto-selects
`conservative + LN` when a reverse channel exists, which is correct here.

| cell | on this arm | work | estimate |
| --- | --- | --- | ---: |
| setup, 0–5 | | Drive mount, OWT cache load, model build | 10–15 min |
| 6b-7 flow or maps | re-file | ~14 evals x 12 x 4 x 512 | 3–7 min |
| 6b-9 E1 | re-file | 3 x 4 x 512, three arms | 1–2 min |
| 6b-10 E3 | **improved run** | builds GPT-2 too; fp32; 3 perturbation sizes | 5–15 min |
| 6b-11 E5 | **improved run** | 12 lambda points x 8 x 4 x 512 | 10–20 min |
| 6b-12 F1 | **first run** | 8 x 4 x 512, three arms | 2–5 min |

One session, roughly 45 to 75 minutes of GPU, no training. 6b-10 additionally
needs the matched GPT-2 checkpoint present on Drive at the path hard-coded in
`R10_GPT2_CKPT`.

**Two of the four are not re-filings.** The cells changed after those runs:

- **6b-10** now uses `R10_EPS = (1e-2, 3e-2, 1e-1)`. The run written up in
  `Geodesic_Experiments_with_CfC_BAOAB.md` §6.8 reports an epsilon = 1e-3 row
  that sat on the TF32 rounding floor and was discarded, and has no 3e-2 or
  1e-1 rows at all. A re-run replaces that table's (c) block with three usable
  rows.
- **6b-11** now prints the maximum *and* the minimum of the per-layer V_phi
  change across the sweep. The run in §11.6 was read off a version that only
  checked increases, which is the bug that let layer 1's ninefold *fall* go
  unmentioned. A re-run puts both directions in the log.

So budget a short edit to §6.8 and §11.6 afterwards. Neither is expected to
move a conclusion; both make the recorded evidence match the current cell.

**6b-12 on this arm is a first run, so pre-register before launching it.**
E1 puts the arm's average deflection at R(geo) = 1.09 with the reverse channel
carrying about 90% of it, and the forced arm already measured (`attention`)
read UNIFORM at 93.8% of tokens above 0.75. The band to record: primary arm
`conservative + LN`, layer 1, **above 80% of tokens over 0.75, under 5% below
0.25, bimodality under 0.6** — that is, uniform forcing, the mirror image of
the no-RC arm's 100% below 0.25.

**A wider point the same work exposes.** The F1 logs already held for
`attention` and `attention_potential` came from the two-arm version of 6b-12,
before the third arm and before the partial correlations that control for step
size. On the attention arm the raw correlation between deflection and step
size is **-0.936**, so every other correlate in that log is read through a
confound the current cell removes. Making all four arms comparable under one
cell version costs two more sessions of the same shape, about 25 minutes each,
because Cell 0 defines the architecture and each arm needs its own build.

### 7.3a Uploaded — **2026-09-27**

Done, public, five repos plus the collection.

| repo (under `dimitarpg13/`) | files | contents |
| --- | ---: | --- |
| `semsimula-ladder-owt-d384-l2-gpt2-matched` | 9 | 2 checkpoints, 4 results, 1 notebook |
| `semsimula-ladder-owt-d384-l2-attention` | 26 | 2 checkpoints, 6 results, 16 code |
| `semsimula-ladder-owt-d384-l2-attention-potential` | 26 | as above |
| `semsimula-ladder-owt-d384-l2-none` | 28 | 2 checkpoints, 8 results, 16 code |
| `semsimula-ladder-owt-d384-l2-none-norc` | 27 | 2 checkpoints, 7 results, 16 code |

Collection: **Semantic Simulation — CfC+BAOAB Mechanism Ladder**, six items,
the Verlet-instability repo first as the prologue. Both URL forms resolve,
and the book uses the suffix-free one to match the existing family link:
`https://huggingface.co/collections/dimitarpg13/semantic-simulation-cfcbaoab-mechanism-ladder`

**Code shipped, and why that subset.** The SPLM family's precedent is that a
repo carries the model source it was trained with, so nobody has to guess
which revision the weights belong to. These repos do that and add the
notebook, because an arm here is defined by Cell 0 knobs rather than by a
script. The Python set is the *transitive import closure* of what the
notebook imports, computed by walking the ASTs rather than chosen by hand:
12 modules plus `data_module.py`, `parf/__init__.py` and
`parf/test_cfc_baoab.py`, which Cell 4 runs as a gate. Layout under `code/`
mirrors the repository so the notebook's own `sys.path` lines resolve
unchanged. The GPT-2 arm ships one notebook and nothing else, because it
defines its model in Cell 3 and imports nothing from the repository.

**Three Hub limits found the hard way**, recorded so the next collection does
not rediscover them: a collection description is capped at 150 characters, an
item note at 500, and the collection slug gets a 24-hex suffix appended to a
slugified title (`CfC+BAOAB` becomes `cfcbaoab`, the em dash is dropped). The
suffix-free URL redirects, which is why guessing the slug in advance would
have half-worked and been fragile.

**Verified after upload**, not merely reported by the uploader: every repo is
public with the expected file count, no `*.published.*` name and no
`_DO_NOT_PUBLISH_*` file leaked, and the flagship checkpoint was downloaded
back from the Hub, loaded, and confirmed at step 31,500, PPL 66.56, 98
tensors, no optimizer state.

Gaps 3 and 4 of §7.2 were handled on the cards rather than in the data: the
GPT-2 card now carries its own caveat block naming both the stale `fock_*`
fields and the step-25,000 asymmetry, and `NOTE_summary_json_fock_fields.md`
ships beside the summary it corrects.

### 7.3b The "Verlet-era" label is too narrow — **2026-09-27**

Checking whether the SPLM Model Family collection should be marked as
Verlet-based turned up that it is not. Integrator declared by each of its
fifteen items:

| integrator | items |
| --- | ---: |
| `semi_implicit_euler` (config) plus the three gamma sweeps, whose READMEs document a damped Euler step `v += dt*f/m; v /= (1 + dt*gamma); h += dt*v` | 13 |
| `first_order_gradient_flow` (the Fock-G1 ablation — no velocity at all) | 1 |
| `verlet` (the instability repo, which is also item 0 of the ladder collection) | 1 |

So **fourteen of fifteen published models in that family are not Verlet.**
What the family actually shares is being *pre-CfC*: explicit integrators,
one force evaluation per step, chosen before the closed-form propagator
existed. Its description now says so and its title is unchanged, because the
title generates the slug and paper v6 cites the suffix-free form of it.

**This reaches further than the collection.** This programme has been saying
"Verlet-era" for the whole pre-CfC period, here and in the scoping remarks
added to paper v6 §8 and §27 (`rem:riemannian-verlet-scope`,
`rem:gamma-eff-scope`). On the published evidence that phrase is too narrow:
the geodesic-residual work whose conclusions those remarks scope was done on
damped-Euler runs, and only the instability that ended the era was Verlet.
The remarks are not *wrong* — a damped explicit step has the same defect the
calibration exposed — but they name the wrong integrator. **Unreviewed; no
text has been changed.** Check before the Verlet containment pass of §2a.

### 7.3 Upload shape

Total for five repos, publishing only the stripped checkpoints and the
results: **≈2.6 GB** (587 / 595 / 592 / 586 MB for the four Fock arms, 273 MB
for GPT-2). The `collection_cfc` design in `_ladder/ladder.json` stands: a
**separate** collection, *Semantic Simulation — CfC+BAOAB Mechanism Ladder*,
with the Verlet-instability repo as item 0 (prologue) and the two queued arms
added when they run rather than created empty. Separate rather than folded
into *SPLM Model Family* because every repo in that family is Verlet-era and
the integrator is exactly what this collection holds fixed.

**The paper's URL waits on the collection.** HuggingFace derives the
collection slug from the title, and it is not obvious what it does with the
em dash and the `+` in `CfC+BAOAB`. The existing family URL resolves without
any id suffix, so a slug-only URL is safe once the real slug is known. Create
the collection first, read its URL, then add one sentence to `main.tex`'s
*Code and supporting material* paragraph next to the existing family link.
Guessing the slug into a 481-page book is the one step not worth saving time
on.
