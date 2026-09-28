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

| 10 | **L=2, multi-ξ SPLM** — V_θ(ξ, h) + ξ routing, **no V_φ**, no Fock | ~12h | the PARF rung: **V_φ has never been removed from a trained model**, in a family named PARFLM. Pairs with run 11 as a 2×2 (§3.2) | queued — **blocked on `pair_potential='none'`** |
| 11 | **L=2, Fock-SPLM** — as run 10 but with the Fock mechanism on | ~13h | replicates the Fock price on a V_φ-free base and supplies the 2×2's interaction term (§3.2) | queued — same blocker |

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

---

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

---

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
