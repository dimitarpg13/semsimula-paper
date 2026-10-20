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

> **Superseded 2026-10-11 by §5.21 (VF),** the same factorial under live gradients and extended by a wells row. The predictions below were made on Gen 2 numbers and are not scored.

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

**C1 scored on Gen 3 G2, 2026-10-05: prediction HIT; the question is closed.** Script `debug/c1_c2_g2.py` and its output. The gate is scaled through its parameter, since 6b-11's buffer method cannot exceed λ = 1. The tokens are 6b-11's (4 × 4 × 512).

| λ | PPL | against λ = 1 |
| --- | ---: | ---: |
| 3.0 | 3,019.20 | +5,255% |
| 2.0 | 1,172.45 | +1,980% |
| 1.5 | 233.60 | +314% |
| 1.25 | 78.56 | +39% |
| 1.1 | 58.64 | +4.0% |
| **1.0** | **56.38** | 0 |
| 0.9 | 58.08 | +3.0% |
| 0.5 | 102.94 | +83% |
| 0.0 | 246.72 | +338% |

- **Perplexity rises monotonically above λ = 1,** and also below it: the trained gate sits at a sharp inference optimum.
- **Consistency check:** the λ = 1 and λ = 0 values reproduce 6b-11's exactly.
- **This is the null that closes the question.** The clip did not hold the gate below the value the model prefers, so C4 and C5 are not licensed.

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

**C2 finished on Gen 3 G2, 2026-10-05: the clip is cosmetic for the endpoint.** The map is exact: Cell 6's `split_decay_params` ordering, checked shape by shape over 95 tensors.
- **The gate (`reverse_channel_scale`) reads |m|/√v of 0.0446 and 0.0204,** median 0.033, against a median of about 0.15 for the other groups. Its gradient cancels from step to step, so it is not being throttled.
- **The heavily clipped reverse-channel weights** (threshold 0.1) read a median of 0.153, the same as the unclipped V_θ (0.154), ξ (0.153) and the score head (0.164). Adam absorbs the rescale.
- **Side finding:** the last layer's destruction gate (`destruction_gates.1`) has no optimizer state at all. It never received a gradient, consistent with DP (§5.14).

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
| SR-π.2 | On G2's configuration, **SR2** (the exact damped-mode flow, which removes the phase term) cuts Gate 3 by more than half, a far larger relative cut than on F3.1's configuration. | 55%; **withdrawn untested on 2026-10-05** (its arm was dropped after SR-π.3) |
| SR-π.3 | A **diagnostic, free:** on G2's checkpoint, split the Gate 3 loss rise by token according to whether the token's stiff modes cross π under refinement (from the 6b-13 per-coordinate θ). Tokens whose modes cross contribute disproportionately: their mean loss rise is ≥ 2× that of non-crossing tokens. | 55% |

**Design lever if these hold.** Choose L, or Δt, so the stiff modes train below π. Report the trained θ distribution with every Gate 3 reading.

**SR-π.3 scored, 2026-10-05: MISS.** Script `debug/sr_pi3_crossing.py` and its output. It runs on G2's best checkpoint, with 6b-13's token draw (seed 20260928, 16,384 tokens) and 6b-7's own refinement (N = 3, hold policy).
- **Gate 3 on these tokens:** +1,446%.
- **θ:** median 2.94 at layer 0 and 4.62 at layer 1, pooled 3.40 (as in 6b-13).
- **The prediction fails.** Crossing tokens' mean loss rise is 2.93 nats, against 2.42 for the rest: a ratio of **1.21**, against the predicted 2 or more. Within each quartile of trained token loss the ratio is 1.14–1.29.
- **The dependence on θ runs the wrong way.** The loss rise *falls* as the token's largest θ rises:

  | θ range of the token's largest mode | share of tokens | mean loss rise (nats) |
  | --- | --- | --- |
  | 2.5–3.14 | 2.9% | 3.66–3.81 |
  | 3.14–3.6 | 7.5% | 3.62 |
  | 3.6–4.2 | 19.8% | 3.17 |
  | 4.2–4.71 | 24.5% | 2.61 |
  | above 4.71 (stays past π) | 45.2% | 2.40 |

  Tokens that never pass π fail worst (3.73).
- **Reading.** In G2, refinement failure is broad, 2.3–3.7 nats for every class of token, and is not a π-phase effect.
- **Consequences:**
  - **SR-π.2** (SR2 on G2's configuration, predicted to cut Gate 3 by more than half) loses its rationale. **That second SR2 arm is dropped (author's decision, 2026-10-05); SR2 runs on F3.1's configuration only, and SR-π.2 is withdrawn untested.**
  - **SR-π.1** is weakened further; my call moves from about 40% to about 30%.
  - **The design lever** (choose L so stiff modes train below π) is not supported by G2.

**SR-π.3b, pre-registered 2026-10-05 before any measurement: is the register bookkeeping what refinement breaks?**
- **Mechanism.** Under 6b-7's refinement the register updates (creation, blend, refresh, destruction) run once per *step*. At N = 3 with the hold policy, steps 0 and 1 both use layer 0's codes, so layer 0's destruction gate, switch-like in the trained models (median about 0.99, DP §5.14), fires twice.
- **Test.** This is the "per-step refreezing of the occupancies" listed under the SR decision rule. Refine the token dynamics as in 6b-7, but run the register bookkeeping once per trained layer code (at steps 0 and 2), carrying the register state through step 1 unchanged. Evaluation only, on G2's checkpoint and SR-π.3's tokens.
- **Prediction:** Gate 3 on G2 falls to +300% or below (from +1,446% on these tokens), called **50%**.
- **Decision.** If it does, the Fock arms' refinement failure is mostly register bookkeeping, not integration. The fix is a time-consistent register update (per-step decay λ^(L/N), with destruction applied once per trained interval), a code change that can be tested before any SR arm runs. If it does not, the failure lies in the token step itself, and SR2 and SR4 remain the right tests.

**SR-π.3b scored, 2026-10-05: MISS.** Script `debug/sr_pi3b_bookkeeping.py` and its output, on SR-π.3's tokens. The sanity check passes: the held stack at N = L reproduces the model exactly.
- **Gate 3:** +1,446% with Gate 3's own refinement, and **+994%** with the register bookkeeping held to once per trained layer. The prediction was +300% or below.
- **In loss terms,** holding the bookkeeping removes about 13% of the refinement penalty: ln 15.46 = 2.74 nats falls to ln 10.94 = 2.39.
- **Reading.** The repeated register update is a real but minor part. The bulk of the Fock models' refinement failure lies in the token step under the register path, not in the register bookkeeping.
- **Open.** G2's register increment exceeds its conservative step on every token (CB0: η 1.5–3.9). It is applied as (dt²/m)·scale·Q per step and then passes into the next step's velocity encoding. How its integrated effect scales with N is the next thing to derive, before any SR GPU run.

**SR-π.4: derivation and pre-registration, 2026-10-05, before any measurement.**
- **The derivation.** In one Fock layer, the BAOAB step returns a projected state h₀ = P(·) and encodes h_prev_out = h₀ − Δt·v₀. The register increment δ = (Δt²/m)·s·Q is then added and projected, h_new = P(h₀ + δ), and the next layer decodes v_next = (P(h₀ + δ) − h₀ + Δt·v₀)/Δt.
  - If P were linear, v_next = v₀ + (Δt/m)·s·Q. The increment is then a velocity kick proportional to Δt plus a displacement proportional to Δt²: a kick-then-drift splitting of a force s·Q, whose total impulse over N steps of Δt = T/N is independent of N.
  - **So the register push is refinement-consistent by construction, and rescaling it by N/L would break that, not fix it.** The suggested rescaled-push fix is withdrawn on this derivation.
- **What can break refinement** is the LayerNorm projection after a finite jump. The continuum argument needs ‖δ‖ ≪ ‖h‖. 6b-11 measured the increment at 1.37× (layer 0) and 1.15× (layer 1) the state's RMS before projection. A projected O(1) jump leaves a velocity footprint (P(h₀ + δ) − h₀)/Δt that does not scale as a force does. The same holds for the conservative step when it is large.
- **Measurement** (`debug/sr_pi4_step_size.py`, evaluation only, CPU). On F3.1, G2, G3 and the L=4 Fock live arm, per layer:
  - the conservative step's size relative to the state, before projection;
  - the register increment's size relative to the state;
  - the projection nonlinearity of the layer's last projection, ‖P(x) − x‖ / ‖x − h‖, the share of the step the projection rewrites.
- **Prediction** (called **60%**). Across the four arms, the rank order of Gate 3 at 1.5× refinement matches the rank order of the projection nonlinearity, averaged over layers. Gate 3: F3.1 +143%, L=4 +216%, G2 +1,274%, G3 +1,342%.
- **Decision.**
  - **If it holds,** the refinement failure is the finite-jump regime: steps comparable to the state, rewritten by the projection. The remedies are smaller per-layer steps (more layers or smaller Δt, which SR-π.1's L=8 arm also tests), or a projection that is not applied to the whole jump. Neither is a rescaled push.
  - **If it does not,** the cause is elsewhere and SR1–SR4 remain the tests.

**SR-π.4 scored, 2026-10-05: MISS.** `debug/sr_pi4_step_size.py` and its output; 4 × 512 tokens, medians per layer.

| arm | Gate 3 | layer | conservative step / state | increment / state | projection rewrites (last) |
| --- | ---: | --- | ---: | ---: | ---: |
| F3.1 | +143% | 0 / 1 | 3.21 / 0.96 | — | 1.01 / 0.24 |
| L=4 Fock live | +216% | 0 / 1 / 2 / 3 | 2.41 / 0.28 / 0.35 / 0.45 | 1.34 / 0.24 / 0.25 / 1.40 | 0.19 / 0.84 / 0.80 / 0.79 |
| G2 | +1,274% | 0 / 1 | 1.68 / 0.39 | **9.22** / 1.15 | 0.59 / 0.76 |
| G3 | +1,342% | 0 / 1 | 1.23 / 0.45 | **8.30** / 0.95 | 0.46 / 0.67 |

- **The prediction fails.** The rank by mean projection nonlinearity (G3, F3.1, L=4, G2) does not match the rank by Gate 3 (F3.1, L=4, G2, G3).
- **Every arm takes finite jumps that the projection substantially rewrites,** including F3.1, which refines best and has the largest conservative steps. So the finite-jump regime is universal, not what separates the arms. Layer 0's ratios presumably also include the lift from the embedding onto the LayerNorm sphere.
- **What does separate the two L=2 Fock arms is their layer-0 register increment,** 8–9× the state, against 1.3× at L=4. Part of that is Δt²: Δt is 4 at L=2 against 2 at L=4. This is descriptive, not a tested prediction.
- **Conclusions:**
  - The derivation stands: the push is a correctly scaled force, and a rescaled push is not a fix.
  - The finite-jump hypothesis does not explain the cross-arm ordering.
  - The cause of the Fock arms' refinement failure stays open. SR1–SR4 (GPU) remain the tests, with the L=2 register increment as the leading descriptive suspect.



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

**Order and cost:** SR1 (cheapest; code change only), then SR2, then SR4a, then SR3, then SR4b. SR5a and SR5b need no code and can run in any free slot; SR5b first. About 14 GPU-h each at L=2. Implementation note: SR2/SR3 need a joint damped-mode substep on span(U) in `cfc_baoab.py`; SR4 needs the step count sampled per batch and the depth code indexed by time. *(2026-10-06: the SR2 substep now exists, `lowrank_damped_flow`, §5.19; SR3's constant-ratio friction and SR4 do not.)*

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
- **CG1 V_φ forecast: HIT.** 6b-9 gives a V_φ share of **−0.0002** against the threshold |·| < 0.05. V_φ is inert in the L=2 Fock arm, as at L=4 (−0.0035), against −0.291 in F3.1. The substitution account now holds at both depths. *Amended 2026-10-11 (§5.20 CD2): "inert" is a displacement reading. Removed at inference, V_φ costs G2 0.037 nats (+3.7%): subordinate, not inert; the Fock channel costs 1.486.*

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

    **G3′ scored, 2026-10-06.** Log `~/Downloads/L2_arm_attnpot_rfqk_32500steps_output.txt`.

    | measure | value |
    | --- | --- |
    | settled (last three evals: 52.83, 53.05, 52.81) | **52.90** |
    | best | 50.97 at step 31,000 |
    | SCAF | CLEAN at all seven audits (the final audit prints nan, as G2's and G3's did) |
    | against G2 (53.12) | −0.4%, within eval noise |
    | against G3 (54.21) | −2.4% |

    - **Settled ≤ 52.1 (50%): MISS.** The mid-run projection, 52.4–52.7, was close.
    - **The key line, whether the exchange field adds value once hardened: NO.** QK-norm and the clip recover G3's loss against G2, but the field then adds nothing measurable over the model without it.
    - **Gradient flat within ±25% of the step-5,000 value (70%): MISS.** The field's group norm, read whenever it is the largest group, rises from 0.43 (steps 7,500–10,000) to 0.82 (22,500–25,000), about +90% across the stable phase, then eases to 0.65–0.74 in the decay. This is consistent with the observation recorded before the result: QK-norm caps the routing logits but not W_v's growth.
    - **Eval noise near the end** is about ±2 PPL between consecutive evals (50.97 at step 31,000, then 52.83–53.05). The three L=2 Fock arms, G2, G3 and G3′, settle within 2.5% of each other.
    - **G3′ diagnostics, 2026-10-06.** Outputs are in `~/Downloads/Cell-6b-*`, all on G3′'s best checkpoint.

      | measure | G3′ | G3 | G2 | F3.1 | L=4 |
      | --- | ---: | ---: | ---: | ---: | ---: |
      | 6b-6, exchange field off (λ = 0) | +26.0 PPL (+51%) | +29.5 (+56%) | — | — | — |
      | 6b-7 Gate 1, velocity reset | +34% | +49% | +33% | +34% | +55% |
      | 6b-7 Gate 2, N = 3 at fixed Δt | +441% | +325% | +51% | +92% | +41% |
      | **6b-7 Gate 3, 1.5× refinement** | **+115%** | +1,342% | +1,274% | +143% | +216% |
      | 6b-8, reverse channel off | +291% | +394% | +300% | — | — |
      | 6b-8, register bank frozen at initialisation | **+2,280%** | +16.7% | +24.0% | — | — |
      | 6b-9, R(geo) | 1.050 | 1.068 | — | — | — |
      | 6b-12, near-geodesic tokens | 0.0% | 0.0% | — | — | — |
      | 6b-13, θ median: layer 0 / layer 1 / pooled | 2.01 / 3.40 / 2.29 | 1.91 / 3.00 / 2.07 | 2.94 / 4.62 / 3.40 | — | — |

    - **The headline is refinement.** G3′ is the most refinement-ready model in the programme: +115% at 1.5× the steps, against +1,274–1,342% for its L=2 Fock siblings, and better than both F3.1 (+143%) and L=4 (+216%). Its layer-1 θ (3.40) still crosses π under refinement, which is one more case against the π account (SR-π.3).
    - **It is also a different solution.** Freezing the register bank at initialisation costs +2,280%, against +17–24% for G3 and G2: G3′ relies on accumulated register content far more than its siblings. Extension (Gate 2) is worse than G3's.
    - **The exchange field remains load-bearing at inference** (+51% when removed), while adding nothing measurable over G2 when trained in (§ G3′ scored).
    - **One seed.** The routing hardening (QK-norm plus the 0.3 field clip) is the only difference from G3, so the change in refinement is attributable to it within this pair. It may not survive replication.
    - **Pre-registered 2026-10-06, before the checkpoints are downloaded: SR-π.4b.** The leading descriptive suspect for the Fock arms' refinement failure is the size of the layer-0 register increment (SR-π.4: 8–9× the state in G2 and G3, 1.3× at L=4). If it is the cause, G3′, which refines well, should have a small one.
      - **Prediction:** G3′'s layer-0 increment / state is 3.0 or below (`debug/sr_pi4_step_size.py` run on G3′), called **55%**.
      - **Miss:** a refinement-ready model with a large increment rules out increment size as the cause.
    - **Local checks on the downloaded folder, 2026-10-06:**
      - **Independent causality check: CLEAN.** Future perturbation is exactly 0 at five cut points, batch independence is exact, and the prefix-only leak tax is −1.4e-4 nats. The local validation PPL is 52.17 (`causality_check_checkpoint.py … rfqk`; output in the run's results folder).
      - **Exchange-field probe** (`debug/exchange_field_probe_G3prime_output.txt`), step 500 → best:
        - the per-head logit scales settled at 11.6–21.7 (init 14.3), so no head approached the ceiling of 100;
        - layer-1 routing entropy is 0.55–0.75 of uniform, and max |score| is 10–20 against G3's 23–37: QK-norm bounded the logits as designed;
        - the field's share of layer 1's conservative force is 0.90 (G3: 0.92);
        - the field's offline gradient rose 0.36 → 2.46, 6.9× (G3: 12×), dominated by W_v (0.11 → 2.41).
        - The W_q/W_k norms still grew, but under QK-norm the scores no longer depend on them.
      - **SR-π.4b: MISS.** G3′'s layer-0 register increment is **7.8×** the state (layer 1: 0.98×), as large as G2's 9.2× and G3's 8.3×. Yet G3′ refines best of all the arms. **The size of the register increment is ruled out as the cause of the Fock arms' refinement failure.** The remaining difference between G3′ and G3 is the routing hardening. Its measured effects are bounded routing logits and a far larger dependence on accumulated register content (frozen bank +2,280%). Which of these makes the trajectory refinable is the open question.

    **Mid-run reading, step 14,500 of 32,500 (2026-10-05, about 8.3 h left).**

    | step | G3′ | G3 | G2 | G3′ / G3 | G3′ / G2 |
    | --- | ---: | ---: | ---: | ---: | ---: |
    | 3,000 | 113.52 | 118.14 | 123.59 | 0.961 | 0.919 |
    | 6,000 | 84.28 | 87.51 | 93.11 | 0.963 | 0.905 |
    | 10,000 | 71.98 | 74.22 | 76.26 | 0.970 | 0.944 |
    | 12,000 | 69.80 | 72.05 | 72.20 | 0.969 | 0.967 |
    | 14,500 | 67.51 | 69.50 | 69.14 | 0.971 | 0.976 |

    - **Against G3,** G3′ holds a steady 3–4% lead (0.961–0.971 since step 3,000). The hardening helps, and the help is not fading.
    - **Against G2,** the lead is shrinking, from about 10% at steps 4,000–8,000 to 2.4% now. That is the pattern G3 followed before it fell behind G2 at about step 13,000.
    - **Projection.** If the 0.965–0.971 ratio to G3 holds through the decay, G3′ settles at about 52.4–52.7, against G3's 54.21. That would beat G2's 53.12 by about 1%, but miss the pre-registered ≤ 52.1 narrowly. The ratio could still move in the decay phase.
    - **Gradient prediction (flat within ±25% of the step-5,000 value): trending to a miss.** The field's group norm appears in the log only when it is the largest group. It was never largest before step 7,500, so its step-5,000 value is not logged. Its readings then rise from 0.43 (steps 7,500–10,000) to 0.52 and 0.57, at least +33% within the stable phase. G3's matching readings were 0.49, 0.57 and 0.67: the same growth, about 15% lower. This is consistent with the W_v observation below, since QK-norm caps the routing logits but not W_v's growth. The offline probe will score it checkpoint for checkpoint.
    - **Health:** SCAF CLEAN at steps 5,000 and 10,000 (leak tax −1.7e-5 and +1.6e-4 nats); no spikes or watchdog triggers. The total gradient norm tracks G3's (0.43 → 0.70 against 0.43 → 0.72).

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
  - **Observation recorded before G3′ completes (2026-10-05), not a revision of any call.** `debug/exchange_field_probe.py` (it supersedes `g3_routing_probe.py` and reads either routing mode) measured the offline LM-loss gradient on G3's checkpoints.
    - **The field's gradient grows 12× from step 500 to best,** 0.21 to 2.63.
    - **Almost all of that is W_v:** 0.11 to 2.58, 24×. W_q goes 0.12 to 0.22 and W_k 0.04 to 0.11.
    - **QK-norm acts on the routing (W_q, W_k), not on W_v.** So G3′'s "gradient stays flat" (70%) rests on the 0.3 group clip and on whether W_v's growth followed from the routing sharpening.
    - The probe reproduces the earlier G3 routing figures (layer 1 entropy 0.60–0.69, max |score| 23–37).
    - It runs on G3′ the moment its folder is downloaded, checkpoint for checkpoint.
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

**CB0 scored, 2026-10-05.** Script `debug/cb0_g2.py` and its output. It runs on G2's best checkpoint, on CPU.
- **η per token** (16,384 tokens) is ‖register increment‖ / ‖conservative step‖, computed with the code's own definitions:
  - layer 0: median **1.47**, p90 1.52, max 1.71;
  - layer 1: median **3.88**, p90 4.64, max 9.64;
  - 100% of tokens are above both 0.3 and 1.0, at both layers.
- **The register path is larger than the conservative step on every token.** So CB2's caps are drastic:
  - ρ = 1.0 binds on every token, cutting the increment by about 1.5× at layer 0 and 3.9× at layer 1;
  - ρ = 0.3 cuts it by about 5× and 13×.

  CB2a and CB2b test a different regime from the one G2 trained in, and their predictions should be read with that in mind.
- **6b-11, the inference slider, run as written** (4 × 4 × 512 tokens per point):
  - PPL goes from 56.38 at λ = 1 to **246.72 at λ = 0, a factor of 4.38**;
  - there is no knee: λ* = 0.9, and every notch costs perplexity;
  - R(geo) at λ = 1 is 0.44 (layer 0) and 0.95 (layer 1);
  - V_φ's direct share of the step rises from 2.6% to 7.0% at layer 0 as the register path is removed;
  - consecutive steps are anti-aligned at λ = 1, with coherence −0.64.
- **Compared with training.** The model trained without the register path, F3.1, is only 1.087× worse than G2. So the slider overstates the path's value about 4×, as the Gen 2 ablation did.
- **Harness note.** A first local run reported 1,114 at λ = 1. That was a bug in my wrapper, not in the cell. The notebook installs the depth routing as an *instance* attribute `_fock_layer_step`, and my script removed its own wrapper with `del`, which stripped the routing too. The script now restores by assignment, and the cell itself (which assigns) is unaffected. The other local scripts were checked: only `_fock_layer_step` is instance-level, and none of them deletes it.


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

- Its §11 explained the TinyStories null mainly by a realized damping of γ_geo ≈ 0.965, worth about 26× suppression. γ_geo has since been withdrawn as an artefact of the residual (book Remark 103).
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
   - *Done 2026-10-18* (`debug/verify_fo_switch.py`, output `debug/verify_fo_switch_output.txt`), on SR2's weights, since the anchor moved to SR2 on 2026-10-11. Off: bit-identical to HEAD in logits, loss and every gradient. On: equal, bit for bit, to Cell 6b-7's Gate 1 reset (`_fom_stack`, N = L, `reset_prev=True`), so the trained switch and the inference-time reset are the same operation. On a 2 × 256 validation batch SR2 loses 3.1% under it (Gate 1 on 12 × 4 × 512 tokens: +3.9%). The corpus switch for SO-TS and FO-TS is in Cell 0 (`CORPUS`).

#### Amendment, 2026-10-18, before any Stage-1 run: the FO arms run with the analytic V_φ force and without layer checkpointing (author's decision)

**What.** SO-TS, FO-TS, FO-OWT and FO-PMX run with Cell 0 `VPHI_ANALYTIC_FORCE = True` and `LAYER_CHECKPOINT = False`. Neither switch is in the variant tag.

**Why.** Speed: on the A100, the two together make a training microbatch 1.62× faster on SR2 (0.468 s against 0.760 s, peak 29.9 GiB) and 1.80× faster on PMX (0.613 s against 1.103 s, peak 32.9 GiB), with float32 arithmetic throughout (Cell 6b-17; productionization plan §10.3a). Both fit the 40 GB card.

**Why the comparison is unaffected.** The anchors (SR2's run for FO-OWT, PMX's run for FO-PMX) were trained with the V_φ force taken by `torch.autograd.grad` and with layer checkpointing on.
- The analytic V_φ force is the same function: in float64 on trained weights it agrees with the autograd force to 1e-14, and its derivatives with respect to every parameter to 2e-12 (`debug/verify_vphi_analytic_f64.py`); in float32 the forces agree to a relative 1e-7 and the training gradients to 1e-6–5e-5 of the whole gradient (`debug/verify_vphi_analytic.py`, on F3.1, SR2, G2, G3′ and PMX).
- Layer checkpointing recomputes each layer's forward in the backward; switching it off changes the memory used, not the arithmetic.
- GPU training is not bit-reproducible in any case, so two runs of one configuration already differ at the level these changes introduce.

So the FO arms differ from their anchors by FO-a alone, and the 2×2's within-corpus pairs (SO-TS against FO-TS) share every setting. The anchors' own runs are not repeated.

**Scope.** Selective bf16 (`AMP_BF16`) is not used: it added 2% on top (§5.26, deferred).

#### Stage 1, first arm: SO-TS finished — **2026-10-19** (scoring waits for FO-TS)

Run output `L2_SO-TS_16250steps_output.txt` (Colab, A100-80GB; Cell 0 as in the amendment above: SR2's settings, `CORPUS = 'tinystories'`, `LADDER_STEPS = 16_250`, analytic V_φ force, layer checkpointing off). The first TinyStories run built the shared token cache (whole train split; 2M validation tokens from the validation split) and its own frequency table for the mass model. 1.07 s per step, 4.98 h.
- **Health:** no watchdog trigger, no spike capture; the causal-leak monitor CLEAN at steps 5,000, 10,000, 15,000 and 16,250 (future perturbation exactly 0; the final audit prints nan for its perplexities, as every arm's does).
- **Settled 5.22** (evals at 15,000, 15,500 and 16,000: 5.21, 5.24, 5.21); best 5.21. Evals come every 500 steps, which does not divide 16,250, so the last eval is at step 16,000 and the final 250 steps (learning rate within 1% of the floor) are not evaluated. FO-TS has the same schedule, so the pair stays matched.
- **Settling check (amendment of 2026-10-04), as written: not settled.** The line through the evals of the last 3,000 steps (13,000–16,000, seven evals) changes by −0.080 nats, against a residual standard deviation of 0.017. The whole window lies in the decay phase (10,562–16,250), where the loss falls by design (5.66 at 13,000, 5.21 at 15,000); only the last three evals are flat. The rule stands as registered: if either TinyStories arm fails, both are extended by the same number of steps at the floor learning rate, and re-checked. The number of steps is not fixed by the rule (proposed: 3,000, so that the re-check window lies at the floor), and the notebook needs a switch to resume a finished run in place at the floor (raising `LADDER_STEPS` would change the tag and recompute the schedule).   - **Extension: 3,000 steps per arm (author, 2026-10-19), if the rule triggers for the pair.** Cell 0 `EXTRA_FLOOR_STEPS = 3_000` (not in the tag): the schedule keeps its 16,250 steps and the loop runs to 19,250 at the floor rate (6e-5). Each run resumes in its own folder from its latest checkpoint, step 15,000 (none was saved after it), so it re-trains 15,000 → 16,250 and then adds the 3,000 floor steps, about 1.3 h per arm. The batch sampler restarts at its seed on any resume (every resumed ladder run has done so), the same for both arms; the log gains a second record for steps 15,500 and 16,000, and the re-check keeps the latest record per step. Verified: with the switch off the schedule is identical to HEAD at every step (OWT and TinyStories); with it on, the tag is unchanged, steps below 16,250 follow the original schedule and steps 16,250–19,249 sit at the floor.
  - **SO-TS extended and re-checked, 2026-10-19: SETTLED.** Output `L2_SO-TS_19250steps_ext_output.txt`. Resumed from the step-15,000 checkpoint (all 89 tensors and the optimizer state), followed the original decay (1.81e-4 at step 15,050, the schedule's value) and held the floor (6.00e-5) from step 16,250 to 19,250. No watchdog trigger or spike; the causal-leak monitor CLEAN at 19,250 (future perturbation 0, leak tax −3.6e-7 nats). Evals, latest record per step: 15,500 5.32, 16,000 5.25, 16,500 5.14, 17,000 5.21, 17,500 5.29, 18,000 5.13, 18,500 5.16, 19,000 5.19. **Re-check, steps 16,000–19,000 (seven evals): fitted change −0.008 nats against a residual standard deviation of 0.012, so SO-TS is settled. Settled perplexity 5.16** (evals at 18,000, 18,500, 19,000), 1.1% below the pre-extension 5.22; best 5.13 at step 18,000. **Cell 6b-7 on the extended best checkpoint** (`Cell-6b-7_flow_or_maps_19250steps_ext_output.txt`): Gate 0 PASS; Gate 1 +2.0% (5.52 → 5.63); Gate 2 at N = 3 +25.5%; Gate 3 at N = 3 +23.6% (0.211 nats), unchanged by the extension.
  - **The per-layer flow (FLOW-R) on the extended SO-TS, 2026-10-19** (descriptive; `debug/refinement_flow_confirmation.py … sr2-ts seed=20261009`, FLOW-R's batch seed, on the step-18,000 checkpoint; output in the run's results folder). Every check exact: the per-layer flow at N = 2 equals the trained model, the replays reproduce the captured steps. Penalty ln(PPL_N / PPL_2), nats:

    | N | 3 | 4 | 6 | 8 | 12 | 16 |
    | --- | ---: | ---: | ---: | ---: | ---: | ---: |
    | **SO-TS (SR2, TinyStories)** | 0.255 | 0.151 | 0.086 | **0.100** | **0.098** | **0.097** |
    | SR2 (OpenWebText), FLOW-R of 2026-10-09 | 0.286 | 0.206 | 0.084 | 0.103 | 0.104 | 0.104 |

    **The per-layer flow converges on TinyStories** by FLOW-R's criterion (|pen(16) − pen(12)| = 0.001 ≤ 0.01, |pen(12) − pen(8)| = 0.002 ≤ 0.02), to the same limit as on OpenWebText, about 0.10 nats from the trained step. The whole-stack refinement (Gate 3) reads maps on both corpora, with TinyStories less penalised (N = 3: 0.211 against 0.405; N = 8 as trained: 1.99). On the same architecture, two corpora, two budgets and two schedules give the same per-layer limit: refinement here is a property of SR2's exact damped flow, not of the corpus. One seed each.
  - **FO-TS follows the same path:** a 16,250-step run, then the same extension from its step-15,000 checkpoint (Cell 0 `EXTEND_FROM_STEP = 15_000` pins the resume there, as Cell 2 would otherwise prefer a later best checkpoint), then Cell 6b-7. Its settled perplexity is scored against SO-TS's 5.16.
  - **A guard fixed on the way.** The completed-run guard looked for an eval at step ≥ 16,250, which a 16,250-step run never has (evals every 500), so Cell 6 would have resumed the finished SO-TS from step 15,000. A run now counts as complete when any log record reached its last step; tested on SO-TS's log (complete at 16,250, not at 19,250) and on two OWT logs (unchanged).
- **Cell 6b-7** (best checkpoint, step 15,000; 12 × 4 × 512 TinyStories validation tokens; output `Cell-6b-7_flow_or_maps_16250steps_output.txt`). Gate 0 PASS (|d loss| 0). **Gate 1, the velocity reset at inference (FO-a applied to the trained model): +2.1%** (5.63 → 5.75), against SR2's +3.9% on OpenWebText: the trained model leans very little on momentum. Gate 2 at N = 3: +25% (0.224 nats). **Gate 3 at N = 3: +23% (0.211 nats)**, about half SR2's 0.405; the penalty still grows away from the trained step count (N = 4: 0.68 nats), so the stack remains a set of maps. Reported next to FO-TS's settled perplexity once it is in (amendment of 2026-10-11).
- **M1 on SO-TS (the reading pre-registered with the Stage-0 decision)** — `fo_series/stage0_models.py OUT SO-TS`, output `fo_series/results/stage0_models_SO-TS_output.txt` (CPU; the run's own validation tokens and frequency table; 2 × 512 tokens). The local build reproduces Colab's Cell 6b-7 loss on its 12 × 4 × 512 batches exactly (1.7284 and 1.7284); the script's own small-sample guard read 4.55 against the checkpoint's 5.21, sampling noise on 6 × 2 × 512 tokens, as Stage 0 found for the OWT models (12%).

  | | SO-TS (TinyStories, CfC, L = 2) | F3.1 (OWT, CfC, L = 2), Stage 0 | TinyStories anchor (Verlet, L = 8), Stage 0 |
  | --- | ---: | ---: | ---: |
  | M1 ε at the output step (median / p90) | **2.50** / 5.49 | 6.13 | 0.000 (interior layers) |
  | force change across the step, ‖Δf‖/‖f‖ | 1.81 | 1.01 | 0.000 |
  | M2 inertial share, output step (median) | 0.27 | 0.50 | 0.59–0.92 |

  Layer 0 is the projection of the embedding onto the LayerNorm sphere (‖Δh‖ 18.7) and is excluded, as in Stage 0. **Reading, by the rule fixed on 2026-10-04: SO-TS learns anharmonic wells (ε ≥ 0.3).** In the CfC architecture TinyStories is far from the near-linear regime the Verlet anchor was in, so the architecture, not the corpus, produced the OWT stiffness, and FO-TS is predicted to show a gap as well (called at 50%). Gate 1 above (+2.1%) points the other way, toward a small cost of removing momentum; FO-TS's training decides.

#### Stage 1, second arm: FO-TS finished — **2026-10-19** (scoring waits for its extension)

Output `L2_FO-TS_16250steps_output.txt`. SO-TS's Cell 0 plus `FIRST_ORDER_MEMORYLESS = True` (FO-a), tag `…sr2_fo_st16250…`. Finished cleanly at 16,250 steps; best 5.18 at step 15,000; no watchdog trigger; SCAF CLEAN at 5,000, 10,000, 15,000 and 16,250 (the last prints nan PPLs, the known cosmetic issue).

| step | SO-TS | FO-TS | FO − SO (nats) | ω·Δt p50, SO / FO |
| ---: | ---: | ---: | ---: | ---: |
| 4,000 | 6.98 | 6.90 | −0.011 | 4.27 / 3.58 |
| 7,000–9,500 (mean) | | | −0.019 (sd 0.003) | 6.8 / 5.3 |
| 10,500 (decay starts) | 6.01 | 5.90 | −0.018 | 6.75 / 5.56 |
| 14,500 | 5.28 | 5.23 | −0.009 | 6.64 / 5.78 |
| 16,000 | 5.21 | 5.19 | −0.005 | 6.69 / 5.72 |
| settled, last three evals | 5.22 | **5.19** | −0.6% | |

- **The lead FO-TS held through the constant-LR phase (about −1.9%) shrank in the decay to −0.5%.** Before the extension the pair is level, inside the Δ_TS band (±3%).
- **The softer potential persists:** FO-TS's median ω·Δt is about 5.7 against SO-TS's 6.7 from the decay to the end, at the same perplexity.
- **Settling check (2026-10-04 rule): fails,** as SO-TS's did: fitted change −0.074 nats over steps 13,000–16,000 against a residual standard deviation of 0.017. FO-TS gets the same extension as SO-TS: `EXTRA_FLOOR_STEPS = 3_000`, `EXTEND_FROM_STEP = 15_000` (both `…_step15000.pt` and `…_step15000_best.pt` are on Drive). Δ_TS is scored after it, against SO-TS's extended 5.16, with Gate 1 beside it.
- **Cell 6b-7** (best checkpoint, step 15,000; output `Cell-6b-7_flow_or_maps_FO_16250steps_output.txt`). The first attempt failed Gate 0 (5.64 against 8.83): the cell's patched layer loop did not apply FO-a's velocity reset. Fixed on 2026-10-19 (the patch resets at every step when `cfg.first_order_memoryless` is set), and checked on CPU: Gate 0 bit-identical for SO, FO-a and FO-b models, and unchanged for second-order ones. Against SO-TS at the same step:

  | reading | SO-TS | FO-TS |
  | --- | ---: | ---: |
  | Gate 1, velocity reset | +2.1% | 0 (by construction) |
  | Gate 2, N = 3 at the trained Δt | +25% | **+0.8%** |
  | Gate 2, N = 4 / 8 | +140% / +2,000% | +14% / +72% |
  | Gate 3, N = 3 at fixed T | +23% (0.21 nats) | **+47% (0.38 nats)** |
  | Gate 3, N = 1 | 43.5 PPL | 239.5 PPL |

  - **Without the velocity between layers the stack is robust to extra layers** (Gate 2): from rest, each extra step is close to one more gradient step toward the wells' minimum, which is stable; the carried velocity is what makes extra second-order steps overshoot.
  - **FO-a's Gate 3 does not test refinement.** A step from rest moves a token by about (dt²/2m) f, so N steps of T/N move it by about T²/(2mN): the distance covered shrinks with N, and FO-a is not a discretisation of any one flow as the step changes. FO-b ties its mobility to the trained step, so N steps of T/N cover one flow; this is the reason behind FB4.

- **FO-TS extended and re-checked, 2026-10-19: SETTLED. Δ_TS scored.** Output `L2_FO-TS_19250steps_ext_output.txt`. Resumed from the step-15,000 checkpoint, followed the original decay (1.81e-4 at step 15,050) and held the floor to 19,250, as SO-TS's extension did; no watchdog trigger; SCAF CLEAN at 19,250 (leak tax −2.4e-4 nats). Best 5.10 at step 18,000.

  | step | SO-TS (ext.) | FO-TS (ext.) | FO − SO (nats) |
  | ---: | ---: | ---: | ---: |
  | 16,000 | 5.25 | 5.22 | −0.0065 |
  | 17,000 | 5.21 | 5.18 | −0.0054 |
  | 18,000 | 5.13 | 5.10 | −0.0055 |
  | 18,500 | 5.16 | 5.13 | −0.0066 |
  | 19,000 | 5.19 | 5.17 | −0.0052 |
  | **settled** (18,000–19,000) | **5.16** | **5.13** | |

  - **Settling re-check, steps 16,000–19,000 (seven evals):** FO-TS's fitted change −0.008 nats against a residual standard deviation of 0.013 (SO-TS: −0.008 against 0.012). Both settled.
  - **Δ_TS = (5.13 − 5.16) / 5.16 = −0.6%. Prediction "Δ_TS within ±3%" (called 55%): HIT.** The Fock-G1 null reproduces under CfC, live gradients, d = 384 and L = 2: on TinyStories, SR2 trained without the velocity between layers matches SR2 trained with it.
  - The paired difference over the extension is −0.0056 nats with a standard deviation of only 0.0007 across seven evals: a small, steady advantage for FO-TS within this pair (same data order, same evaluation batches), far inside the unmeasured seed spread. Not read as a gain.
  - **Gate 1 beside it** (the amendment of 2026-10-11): SO-TS's own velocity reset costs +2.0% at its extended best checkpoint; trained first-order, the model loses nothing.
  - **Cell 6b-7 on the extended best checkpoint** (step 18,000; `Cell-6b-7_flow_or_maps_FO_19250steps_ext_output.txt`): Gate 0 PASS; Gate 1 0; Gate 2 at N = 3 +1.1% (SO-TS +25.5%), at N = 8 +77% (SO-TS +2,000%); Gate 3 at N = 3 +48% (0.391 nats; SO-TS +23.6%). Unchanged by the extension.
  - **M1, matched at the step-18,000 checkpoints** (`stage0_models.py FO-TS` and `SO-TS-ext`; outputs `results/stage0_models_FO-TS_output.txt`, `results/stage0_models_SO-TS-ext_output.txt`). At the output step (layer 1): ε median 4.52 (FO-TS) against 2.59 (SO-TS); step ‖Δh‖ 20.7 against 13.4; force change ‖Δf‖/‖f‖ 1.18 against 1.83; inertial share 0 (by construction) against 0.27. Layer 0: ε 1.75 against 1.28.
  - **The Stage-0 rule's prediction for FO-TS (a gap, called 50%): MISS.** FO-TS learns strongly anharmonic wells, more so than SO-TS, and matches it. Trained first-order, the model takes larger output steps across a more non-linear stretch of the potential and reaches the same perplexity: the wells adapt to the dynamics. **ε describes a trained model; it does not say whether the task needs second order.** The Stage-0 decision rule (ε ≥ 0.3 → a gap) is refuted on its first trained test.
  - **FLOW-R on FO-TS** (`refinement_flow_confirmation.py … fo-ts seed=20261009`, extended best checkpoint, step 18,000; output in the run's `results/`). Checks all PASS: the standard refinement reproduces Cell 6b-7 (which resets at every step) and the per-layer flow reproduces the model's own `substeps_per_layer` path (which resets where each trained layer begins), both bit-identical; `split_stack` tells the two apart by its `freeze` flag. Kick share 0.67 and 0.85 (SO-TS 0.39 and 0.65). Per-layer flow penalties in nats at N = 3, 4, 6, 8, 12, 16: 0.447, 0.221, 0.254, 0.276, 0.289, 0.279; standard refinement at N = 8: 2.39. **The flow converges on the letter of the criterion** (|pen16 − pen12| = 0.010, |pen12 − pen8| = 0.012) **but to a limit 0.28 nats above the trained step**, against 0.097 for SO-TS and 0.104 for SR2. Reading: within a layer FO-a's flow is the same damped flow as SO-TS's, started from rest; the trained FO-a step is a far coarser sample of it than SO-TS's step is of its own flow. The convergent-and-close per-layer flow is a property of the second-order step on this base; FO-b's, which refines one gradient flow, is FB4's question.

#### FO-PMX: the exploratory first-order twin on PMX — **pre-registered 2026-10-18, before the run; exploratory, outside the 2×2 (author's decision)**

**Why.** The 2×2's anchor, SR2, leans little on momentum: resetting the velocity at inference costs it +3.9% (Gate 1). PMX leans on it far more: +91% at its best checkpoint, +30,600% at the step-8,000 probe. In PMX the layer-1 step balances the velocity a token carries in against the pull of the memory wells (§5.15). If second-order dynamics earns its keep anywhere at this budget, it is there. The FO 2×2 cannot see this, because none of its arms has wells. FO-PMX asks whether PMX trained first-order matches PMX.

**The arm.** PMX's Cell 0 (F3.1's settings plus `POISSON_MODES = 64`, `LOWRANK_DAMPED_FLOW = True`, `POISSON_WELLS_EXACT = True`) plus `FIRST_ORDER_MEMORYLESS = True`; tag `…pm64_pmx16_sr2_fo…`; 32,500 steps, seed 0, OpenWebText; about 24.5 h at PMX's 2.72 s/step. **Switch verified on PMX's weights** (`debug/verify_fo_switch.py arm=pmx threads=1`, output `debug/verify_fo_switch_pmx_output.txt`): off, it is bit-identical to HEAD in logits, loss and all 77 gradients (single-threaded: at 8 threads one gradient, `pm_mu`'s, varies by about 5e-10 from run to run in HEAD itself); on, it equals Cell 6b-7's Gate 1 reset bit for bit, trains with finite gradients on the same parameters, and adds no parameters. On a 2 × 256 batch, PMX's trained weights lose ×1.82 under it (SR2: ×1.03).

**Statistic.** Δ_PMX = (settled PPL of FO-PMX − 54.63) / 54.63, against PMX's settled 54.63.

| call | prediction | called |
| --- | --- | ---: |
| FX1 | second order earns its keep in the wells: Δ_PMX ≥ +3% (FO-PMX at or above 56.27) | 55% |
| FX2 | it earns more there than without wells: Δ_PMX exceeds Δ_OWT (FO-OWT against SR2) by at least 2 points | 55% |
| FX3 | the wells still help when trained first-order: FO-PMX settles below F3.1's 57.76 | 65% |

**Descriptive** (Cells 6b-7, 6b-9, 6b-13, 6b-15 on the best checkpoint): Gate 1 reads 0 by construction (the trained model already resets); Gate 2 and Gate 3 against PMX's 2.36 and 1.097 nats at N = 3; the wells' depths and half-lives (6b-15) against PMX's, to see whether a first-order model builds shallower wells.

**Readings, fixed now.**
- **FX1 hits:** the memory wells need second-order dynamics at this budget, and whatever the 2×2 finds on SR2 does not extend to the memory arms.
- **FX1 misses:** even where the trained model leans hardest on momentum, a model trained first-order compensates. PMX's Gate 1 then measures how PMX was trained, not what the task needs.
- **Scope:** as for the whole series, constant γ = 0.1, L = 2, 532.5M tokens, one seed.

#### FO-b: true first-order arms on both corpora — **pre-registered 2026-10-19, before the runs (author's decision)**

**Why FO-a is not first order under CfC.** FO-a sets `h_prev := h` at each layer boundary, so each layer step starts from rest; the step itself is unchanged. On the parts of the force carried by the kick (V_φ, the soft diagonal spring, the nonlinear V_θ residual), a step from rest is exactly one gradient step at learning rate dt²/2m, which is why Gen 1's Verlet Fock-G1 was first order. SR2 integrates the stiff low-rank modes exactly, and from rest each mode follows the damped-oscillator response, (1 − cos ωt)/ω² times the force for an undamped mode. That reduces to a gradient step only when ω·Δt ≪ 1, and SR2's median ω·Δt is about 6.6. **FO-a is therefore memoryless second order:** no state except h crosses a layer, but each step is still an inertial flow in the stiff modes. FO-a results test the velocity carried between layers, not second order as such. The prediction table's line "FO arms, Gates 2 and 3: pass by construction" assumed a gradient step and does not hold for FO-a under CfC.

**Change to the conditional rule.** FO-b was to run only if FO-a showed a gap. A null FO-a would then leave the inertia inside each step untested, which is the claim the series is after. FO-b now runs unconditionally, on both corpora, ahead of FO-OWT (author, 2026-10-19).

**FO-b** (Cell 0 `FIRST_ORDER_OVERDAMPED = True`, `cfg.first_order_overdamped`, tag `fob`). The layer step is the gradient flow dh/dt = μ f(h) with mobility μ = dt / (2m): dt is the trained layer step (4), m the learned per-token mass, f = f_θ + f_φ the same conservative force. It keeps the BAOAB step's splitting and its one force evaluation:
- **A, half step:** the retained low-rank modes relax exactly, z → z* + e^(−μκ dt/2)(z − z*) (`cfc_baoab.lowrank_overdamped_substep`); nothing else moves.
- **B:** a gradient step μ dt · f = (dt²/2m) f on everything A does not integrate, at the half-step point. This is the step FO-a takes from rest in the soft directions.
- **A, half step** again, from the moved point. No velocity exists: the step returns h_prev_out = h_new, there is no friction step, and γ is unused.

Because m is learned, the model can still set its own step size. μ uses the trained `cfg.dt`, not the substep, so the per-layer flow (k substeps of dt/k) refines one fixed gradient flow. No parameters are added; off is bit-identical. Supported: the low-rank integrator (the SR2 base) at T = 0; PMX's exact wells are being added (below).

**Verification** (`debug/verify_fob_switch.py`, output `debug/verify_fob_switch_output.txt`, on SR2's weights): 
1. Off is bit-identical to HEAD in logits, loss and all 73 gradients.
2. The mode relaxation is the exact gradient flow of the low-rank spring: against a fine RK4 solve in float64, relative error 1.2e-14 for stiffness 30, 2, 0.3, 1e-9, 0 and −0.7; the complement of span(U) does not move (6.7e-16).
3. At real layer inputs, in float64, k substeps of dt/k converge to an RK4 solve of dh/dt = μ(f_θ + f_φ)(h) over dt: the error falls at every doubling of k from 8 to 128, and the observed order over the last doublings is 1.00 (layer 0) and 1.04 (layer 1), the same with V_φ's pair term zeroed. (A first pass rule, error at k = 128 at most 2% of the k = 1 error, was replaced by this asymptotic-order rule on 2026-10-19, after PMX's layer 1 converged at order 0.99 but missed it at 2.2%: the k = 1 error depends on a layer's stiffness. SR2 passes both.) The step hands on zero velocity (h_prev_out = h_new exactly). **The scheme is first order in the step**, as any explicit first-order step with one force evaluation is: the middle part is one gradient step, so a single FO-b step is gradient descent with exact relaxation of the stiff modes.
4. On, it trains: finite loss and gradients, the same parameters receive a gradient as when off, the mass parameters among them (they set the mobility). On SR2's second-order weights FO-b reads 3.794 against 3.507 nats (×1.33 in perplexity on this batch), descriptive only.
5. No parameters added (89 tensors, same keys and shapes).
6. The mobility uses the trained step, fixed at construction, so Cell 6b-7 (which sets `cfg.dt` to each probe's step) refines or extends one gradient flow; a step is unchanged when `cfg.dt` is changed.

**Arms.** Both start from the matching second-order Cell 0, plus `FIRST_ORDER_OVERDAMPED = True`, with the speed switches of the 2026-10-18 amendment (`VPHI_ANALYTIC_FORCE = True`, `LAYER_CHECKPOINT = False`) and seed 0.
- **FO-b-TS:** SO-TS's Cell 0 (TinyStories, 16,250 steps). The settling check of 2026-10-04 applies; if it fails, the same 3,000-step extension at the floor as SO-TS and FO-TS.
- **FO-b-OWT:** SR2's Cell 0 (OpenWebText, 32,500 steps, 532.5M tokens), against SR2's 58.32. **Budget decided 2026-10-19 (author):** 0.5B tokens, because the second-order arm already exists at that budget (SR2), so the comparison costs one run of about 10 h; a 2B pair (SR2-2B and FO-b-2B, about 80 h) is decided after this result and the Gate-1-against-training-step curve. Runs next after FO-TS and its extension, then FO-b-TS.

**Decisions of 2026-10-19 (author), on the order of the FO arms.** FO-b removes strictly more than FO-a: FO-a drops the velocity between layers, FO-b drops that and the inertia inside each step. So FO-b runs first, and **FO-a becomes conditional on an FO-b gap**:
- **FO-b null:** FO-a is almost certainly null as well and is skipped.
- **FO-b gap:** FO-a locates it, between layers or inside the step.

This applies to FO-OWT (conditional on FO-b-OWT) and to FO-PMX (conditional on FO-b-PMX). FO-TS, already training, completes. **FO-b-PMX** (OpenWebText, 32,500 steps, against PMX's 54.63) replaces FO-PMX as the exploratory twin, once FO-b supports PMX's exact wells; its predictions are registered with its verification, before the run. A TinyStories PMX pair (SO-PMX-TS and FO-b-PMX-TS, both new, about 14 h) is deferred: PMX's wells exist for memory across long contexts, which TinyStories' short stories exercise little.

**Statistic.** Δᵇ_corpus = (settled PPL_FO-b − settled PPL_SO) / settled PPL_SO, beside Δ_corpus for FO-a.

| id | prediction | called |
| --- | --- | --- |
| FB1 | TinyStories does not need second order at all: Δᵇ_TS within ±3% | 55% |
| FB2 | OpenWebText does: Δᵇ_OWT ≥ +3% (FO-b-OWT at or above 60.07) | 60% |
| FB3 | the inertia inside the step matters more than the velocity between layers: Δᵇ ≥ Δ_FO-a + 1 point, on at least one corpus | 50% |
| FB4 | a trained gradient flow refines: Cell 6b-7 Gate 3 at N = 3 at most +10% on both corpora (SO-TS +23.6%, SR2 +50%) | 50% |

**Readings, fixed now.**
- **FB1 and FB2 both hit:** the corpus separates first from second order, and Stage 0's statistics are the candidates for why.
- **FB1 misses (a TinyStories gap):** inertia matters even on TinyStories once the wells are stiff; the Gen 1 null was a property of its near-linear wells (ε = 0).
- **FB2 misses:** first-order training suffices on OpenWebText at this budget and damping. FO-PMX (second order with wells) is then the remaining case where second order might earn its keep.
- **FO-a and FO-b agree:** the inertia inside the step adds nothing beyond what the velocity between layers carries, so FO-a's result stands for second order as a whole.

**Descriptive:** Cells 6b-7 (Gate 1 reads 0 by construction), 6b-13 (ω·Δt, against SO-TS and SR2), M1 and FLOW-R.

**Scope:** constant γ = 0.1 in the second-order anchors, L = 2, d = 384, one seed, 266M (TinyStories) and 532.5M (OpenWebText) tokens.

#### FO-b-OWT scored: **61.16 settled, Δᵇ_OWT = +4.9% — FB2 HIT; second order earns its keep on OpenWebText** — **2026-10-20**

Output `L2_FO-b-SR2_OWT_32500steps_output.txt`. SR2's Cell 0 plus `FIRST_ORDER_OVERDAMPED = True` and the speed switches; tag `…sr2_fob…`; 32,500 steps, seed 0, an A100 80 GB (about 1.04 s per step). Finished cleanly; no watchdog trigger or spike capture; SCAF CLEAN at every audit (the step-32,500 audit prints nan PPLs, the known cosmetic issue). The in-flight resonance monitor was empty throughout: it hooks the second-order substep routines, which FO-b does not call.

| step | phase | SR2 | FO-b | FO-b − SR2 (nats) |
| ---: | --- | ---: | ---: | ---: |
| 500 | warmup | 461.8 | 451.2 | −0.023 |
| 5,000 | constant LR | 108.4 | 101.1 | −0.069 |
| 10,500 | constant LR | | 82.8 | −0.009 |
| 11,000 | constant LR | | 87.3 | **+0.029: crossover** |
| 15,000 | constant LR | 76.8 | 79.5 | +0.034 |
| 21,000 | decay starts | | | +0.029 |
| 25,000 | decay | 65.9 | 67.5 | +0.023 |
| 30,000 | decay | 58.1 | 61.0 | +0.049 |
| **settled** (31,500–32,500) | | **58.32** | **61.16** | +0.048 |

- **Settling check (2026-10-04 rule), steps 29,500–32,500:** fitted change +0.001 nats against a residual standard deviation of 0.012: settled (SR2 under the same rule: +0.001 against 0.013). Best 59.51 at step 31,000.
- **Δᵇ_OWT = (61.16 − 58.32) / 58.32 = +4.9%. FB2 (≥ +3%, called 60%): HIT.** On OpenWebText the conservative core trained as a true gradient flow is clearly worse than its second-order twin.
- **The shape.** FO-b led from the first evaluation (−2.3% at step 500) and widened the lead to −6.7% at step 5,000. It stalled between steps 9,000 and 13,000 (evaluations between 82.8 and 87.9, no spike, no clip storm) while SR2 kept improving, crossed at step 11,000, stayed about 3% behind through the rest of the constant phase (+3.8% at 12,500 easing to +2.3% at 25,000), and lost ground again in the decay: SR2 gained 0.196 nats from step 21,000 to 32,500, FO-b 0.175. The gap grows with training in the phase that sets the settled value.
- **What FB1, FB3 and FB4 need.** FB1 (TinyStories, ±3%): FO-b-TS. FB3 (inertia inside the step matters more than the velocity between layers): FO-a on OpenWebText (FO-OWT), now triggered by this gap under the 2026-10-19 ordering decision. FB4 (a gradient flow refines): Cell 6b-7 on FO-b-OWT's best checkpoint.
- **Against the corpus prediction.** With FO-TS level (Δ_TS = −0.6%, FO-a), the 2×2's "OWT gap without a TinyStories gap" outcome is in view, which says the corpus separates first from second order. The clean interaction for FO-b needs FO-b-TS.
- **A fairness caveat, recorded now.** FO-b ran with the hyperparameters tuned for the second-order arms (learning rate 1.2e-3, the WSD fractions, the clip groups). The stall between steps 9,000 and 13,000 reads as an optimisation difficulty, so part of the gap may be a hyperparameter mismatch. One FO-b-OWT run at a lower learning rate (8e-4) is proposed as a check before any 2B decision.
- **Cell 6b-7** (best checkpoint, step 31,000; `Cell-6b-7_flow_or_maps_FO-b-SR2_OWT_32500steps_ext_output.txt`). Gate 0 PASS; Gate 1 0 (no velocity exists). Gate 2, one extra layer at the trained Δt: **+0.6%** (SR2 +35%); eight layers +138% (SR2 +567%). Gate 3 at N = 3: **+19.5% (0.178 nats)** against SR2's +50% (0.405); at N = 8 1.46 nats against 2.11; the penalty grows away from the trained step count. **FB4 (Gate 3 at N = 3 at most +10% on both corpora): MISS on OpenWebText.** The standard refinement recomputes ξ and projects onto the LayerNorm sphere at every substep, a different flow from the one the trained layer samples; SR2 failed it too and passed the per-layer flow (FLOW-R). FLOW-R on FO-b-OWT is the test that says whether a fixed-mobility gradient flow refines, pending the checkpoint's download.
- **Cell 6b-9** divided by zero on FO-b: its velocity readings are undefined when no velocity exists. Guarded on 2026-10-20 (R_v prints nan); R_h remains the V_φ attribution of a gradient-flow step.
- **Against the analysis of 2026-10-20** (`Future_of_Semantic_Simulation_When_Second_Order_Earns_Its_Keep.md`, §3): the early lead fits Proposition 3 (the gradient flow's learning problem is easier at first); the settled gap refutes the reading that Propositions 2 and 4 make the conservative core overdamped on any corpus. The note is revised accordingly.

#### FO-b-PMX: the true first-order twin of PMX — **pre-registered 2026-10-19, before the run (author's decision)**

**Why.** PMX leans on momentum more than any SR2-base arm (Gate 1 +91% at its best checkpoint; ×1.82 on a 2 × 256 batch under FO-a, against SR2's ×1.03). Its layer-1 step balances the velocity a token carries against the pull of the memory wells (§5.15). If second order earns its keep anywhere at this budget, it is there. FO-b-PMX replaces FO-PMX (FO-a) as the exploratory twin; FO-PMX runs only if this arm shows a gap.

**The step.** PMX's second-order A substeps integrate, exactly and with damping, the frozen spring K = αI + U diag(κ) Uᵀ on all of ℝ^d: V_θ's retained low-rank modes together with the wells' quadratic at the step's start (an isotropic curvature α per token, and the top 16 wells' rank-one terms, of either sign). FO-b-PMX relaxes the same spring first order, x' = μ(f − K(x − h)), exactly (`cfc_baoab.lowrank_iso_overdamped_substep`): each direction moves by μt·φ(μkt) times its force component, φ(x) = (1 − e^(−x))/x, with stiffness k = κ + α in span(U) and α on the complement. A negative stiffness grows as e^(μ|k|t) over the step, as the second-order flow grows in an unstable direction. The kick subtracts the same quadratic force as PMX's does, so the step still integrates the whole force f = f_θ + f_φ + f_wells. Mobility, splitting and everything else are FO-b's (above).

**Verification** (`debug/verify_fob_switch.py arm=pmx threads=1`, output `debug/verify_fob_switch_pmx_output.txt`): 
1. Off is bit-identical to HEAD in logits, loss and all 77 gradients (single-threaded).
2. The isotropic relaxation is the exact gradient flow of K = αI + U diag(κ) Uᵀ: against RK4 in float64, relative error 6.1e-15 for α = 0.5, −0.3, 0, 1e-9 and 2 with the mixed-sign κ of the SR2 check.
3. The per-layer flow converges to RK4 of dh/dt = μ(f_θ + f_φ + f_wells)(h) at first order: observed order 1.00 (layer 0) and 0.99 (layer 1), the error halving at every doubling of k from 8 to 128, zero velocity handed on. The reference must hold the well occupations at the layer input: the per-layer flow freezes them where the force is first evaluated, the first substep's midpoint, which tends to the layer input as k grows. A first reference copied them from the k = 1 run and converged to a different flow (a 5% plateau); the second-order PMX step freezes them the same way.
4. On, it trains: finite loss and gradients, the same parameters receive a gradient as when off, the mass among them. On PMX's second-order weights FO-b reads 4.861 against 3.450 nats (×4.1 in perplexity on this batch; SR2: ×1.33), descriptive only: PMX's trained weights lean on inertia far more than SR2's.
5. No parameters added (93 tensors).
6. A step is unchanged when `cfg.dt` is changed.

**Arm.** PMX's Cell 0 (F3.1's settings plus `POISSON_MODES = 64`, `LOWRANK_DAMPED_FLOW = True`, `POISSON_WELLS_EXACT = True`) plus `FIRST_ORDER_OVERDAMPED = True`, `VPHI_ANALYTIC_FORCE = True`, `LAYER_CHECKPOINT = False`; tag `…pm64_pmx16_sr2_fob…`; 32,500 steps, seed 0, OpenWebText; about 14 h at about 1.5 s per step.

| id | prediction | called |
| --- | --- | --- |
| FBX1 | second order earns its keep in the wells: Δᵇ_PMX ≥ +3% (FO-b-PMX at or above 56.27) | 60% |
| FBX2 | it earns more there than without wells: Δᵇ_PMX exceeds Δᵇ_OWT (FO-b-OWT against SR2) by at least 2 points | 55% |
| FBX3 | the wells still help when trained first order: FO-b-PMX settles below FO-b-OWT | 60% |
| FBX4 | the first-order wells train stably: no watchdog trigger and no loss spike above 2× the running value | 70% |

**Readings, fixed now.**
- **FBX1 hits:** the memory wells need second-order dynamics at this budget; FO-PMX (FO-a) then runs to say whether the velocity between layers or the inertia inside the step carries it.
- **FBX1 misses:** even the model that leans hardest on momentum is matched by a first-order model trained from scratch. PMX's Gate 1 then measures how PMX was trained, not what the task needs.
- **FBX4 misses:** the unstable well directions, which the damped oscillator contains, grow unchecked without inertia; the arm is reported as a stability result, not scored for perplexity.

**Descriptive:** Cells 6b-7 (Gate 1 reads 0), 6b-13 and 6b-15 (well depths and half-lives against PMX's, to see whether a first-order model builds shallower wells).

**Scope:** constant γ = 0.1 in the anchor, L = 2, d = 384, 532.5M tokens, one seed.

#### SR3-crit on SR2 and the learned γ(h) arm: the damping axis of the first-vs-second-order question — **pre-registered 2026-10-20, before the runs (author's decision)**

**Why now.** FO-b-OWT leads SR2 at every evaluation so far (by 5.6% at step 4,000, 3.6% at 6,500). SR2 runs at γ = 0.1 with stiff modes at ω·Δt ≈ 6.6, a damping ratio ζ ≈ 0.03. The note `Future_of_Semantic_Simulation_When_Second_Order_Earns_Its_Keep.md` (§3) argues from the equations that the second-order layer's disadvantages, a rotation instead of a contraction and an oscillating learning signal for the stiffness, are properties of **light damping**, not of second order as such. So a first-order win over SR2 has two readings: first order is better, or the oscillation was the problem. Two arms separate them, and both are cheap. The queue of 2026-10-20 places them directly after FO-b-OWT.

**Arm 1, SR3-crit on SR2** (Cell 0 `STIFF_MODE_DAMPING_RATIO = 1.0` on SR2's settings, tag `sr3`; 32,500 steps, OpenWebText, seed 0, the speed switches; about 10 h). Per-mode damping in the exact damped flow of the retained stiff modes, γᵢ = γ₀ + 2ζ·ωᵢ with ζ = 1 and ωᵢ² = κᵢ/m, the book's mode-resolved friction (§7a) and the damping note's §11.2 form; the complement keeps γ₀ = 0.1 in the O-step; no new parameters; no gradient through γ. Under critical damping a stiff mode responds from rest as (1 + ωt)e^(−ωt): monotone, a contraction, with a one-signed sensitivity to the stiffness. SR3-crit is therefore "FO-b within the layer, plus the velocity between layers". **Switch verified 2026-10-20** (`debug/verify_sr3_switch.py threads=1`, output `debug/verify_sr3_switch_output.txt`, on SR2's weights): off is bit-identical to HEAD in logits, loss and all 73 gradients; at γ = 2ω the exact flow reproduces the closed-form critically damped response (1 + ωt)e^(−ωt) to 1e-16 and a fine RK4 solve to 7e-15, for ω from 3 down to 1e-6; on, the retained modes are unchanged and the step differs on span(U) (the complement differs only through the kick, re-evaluated at the new midpoint); it trains with finite gradients on the same 73 parameters; no parameters added. Descriptive: SR2's trained weights lose ×1.57 under critical damping on a 2 × 256 batch (under FO-b, ×1.33): the weights were shaped for the oscillation they were trained with, which says nothing about a model trained critically damped.

| id | prediction | called |
| --- | --- | --- |
| SC1 | SR3-crit settles within 2% of FO-b-OWT: the FO-b gain is the damping regime | 65% |
| SC2 | SR3-crit beats FO-b-OWT by 3% or more: the velocity between layers carries something on OpenWebText at L = 2 (then FO-OWT, the FO-a arm, runs to confirm) | 15% |
| SC3 | SR3-crit beats SR2 (58.32) by 3% or more | 70% |
| SC4 | Gate 1 of SR3-crit is at most SR2's +3.9% | 60% |
| SC5 | its per-layer flow converges (FLOW-R's criterion) to a limit within 0.15 nats of the trained step | 60% |

**Arm 2, the learned γ(h) on the conservative core** (Cell 0 `LEARNED_DAMPING_RATIO = True` on SR2's settings, tag `gzh`; 32,500 steps, OpenWebText, seed 0, the speed switches; about 10 h). **Design, fixed and verified 2026-10-20:** a two-layer head of width 64 on the layer input, ζ(h) = softplus(head(h)) per token and layer (4 tensors, 24,769 parameters), its last layer zero and its bias softplus⁻¹(1) so that ζ = 1 exactly at initialisation; applied as γᵢ(h) = γ₀ + 2ζ(h)ωᵢ to the retained stiff modes inside the exact damped flow, the complement keeping γ₀; gradient through ζ; the head in its own clip group `damping_head` at 0.3; the per-layer median, p10, p90 and fraction above 1 of ζ logged every 50 steps and in every eval record (`zeta_stats`). Verification (`debug/verify_gzh_switch.py threads=1`, output `debug/verify_gzh_switch_output.txt`): off bit-identical to HEAD; on at initialisation bit-identical to SR3-crit in logits, loss and all 73 shared gradients; the head's last weight and bias receive finite, non-zero gradients; the head's four tensors are the only addition; `zeta_stats` read [1, 1, 1, 0] per layer at initialisation and follow a shift of the bias. A per-token, per-layer damping γ(h) = γ₀ + 2ζ(h)·ωᵢ applied to the stiff modes, with ζ(h) produced by a small head on the layer input and **initialised at ζ = 1**, free to move toward the underdamped or the overdamped end; gradient through γ, with the per-group gradient norms logged (damping note §9.5). The critical initialisation is a consequence of the equations: at light damping the output's sensitivity to γ, about −(T/2)e^(−γT/2)cos(ωT)(z₀ − z*), depends on the mode's phase, so an arm initialised at γ = 0.1 may never find the overdamped end by gradient descent and would look as if it had chosen inertia.

**The primary measurement is the trained distribution of ζ(h)**, not the perplexity: the fraction of tokens in the overdamped regime (ζ above 1) per layer, and its dependence on token frequency and position in the sequence.

| id | prediction | called |
| --- | --- | --- |
| GH1 | the distribution goes to the overdamped end: median ζ above 1 at every layer | 60% |
| GH2 | it does so regardless of the local anharmonic fraction ε (M1): the correlation between ζ(h) and ε is below 0.2 in magnitude. This **replaces** the registered γ(h)-link prediction of 2026-10-03 ("γ higher where ε is low"), which rested on the Stage-0 rule that FO-TS refuted | 55% |
| GH3 | perplexity within 2% of SR3-crit | 70% |
| GH4 | a second arm on a Fock or well base (queue item 5) shows a bimodal ζ(h), with inertia kept at the layer and tokens where the register path or the wells act | 50% |

**Amendment, 2026-10-20, after FO-b-OWT's result and before either run (author's decision; the calls above stay on record).** FO-b-OWT fell behind SR2 at step 11,000 and stood +5.1% behind at step 30,000 (a settled value near +4 to +5% is projected; FB2 heading for a hit). SC1 and GH1 assumed FO-b would match SR2, so they are superseded by the following, called now:

| id | prediction | called |
| --- | --- | --- |
| SC1′ | SR3-crit settles at or below SR2's 58.32 (within +1%): inertia is what pays, the damping level costs little | 50% |
| SC2′ | SR3-crit beats SR2 by 2% or more: inertia pays, and SR2's light damping was costing perplexity | 25% |
| SC3′ | SR3-crit settles 3% or more behind SR2, toward FO-b: the oscillation of the stiff modes does useful work | 25% |
| GH1′ | the learned ζ(h) on the conservative core does **not** go to the overdamped end everywhere: median ζ at most 1 in at least one layer | 65% |
| GH5 | ζ(h) differs between the layers by at least 0.3 in median (inertia kept in one layer, removed in the other) | 45% |

SC4, SC5, GH2–GH4 stand as written. With FO-b behind, the γ(h) arm's question changes from "does the model go overdamped" to "where does it keep inertia": by layer, by token frequency and by position.

**Readings, fixed now.**
- **SC1 and GH1 both hit:** the conservative core is overdamped by its own choice, and the second-order theory's role is as the general case whose overdamped limit the trained models occupy.
- **SC2 hits:** inertia between layers earns its keep on OpenWebText; the FO-a arm locates it, and the book's two-regime reading (§6 of the note) applies to the conservative core too.
- **GH1 misses with SC1 hit:** the model keeps inertia it does not need; the head's optimisation, not the task, decides ζ, and the arm is reported as a design result.

**Scope:** as the series: 0.5B tokens, L = 2, d = 384, one seed, OpenWebText.

### 5.14 DP series: what Doi–Peliti process do the trained registers implement? — **pre-registered 2026-10-05, before any measurement**

**Why.** Book §10.5.2 makes the Fock registers a bosonic Doi–Peliti system. It says "a continuous salience variable plays the role of a Poisson mean." The code ([`model_fock_parf_v2.py`](../notebooks/conservative_arch/parf/model_fock_parf_v2.py), [`model_fock_parf_multixi.py`](../notebooks/conservative_arch/parf/model_fock_parf_multixi.py)) shows four structural facts that bear on that reading. They are stated here before any measurement.

1. **Time is depth.** Salience updates once per layer, separately at each position.
2. **The process starts full.** Every register starts at salience 1, not at the vacuum. With decay 0.5, every register is necessarily active at layer 0.
3. **Salience is bounded and deterministic.** The update is a convex combination, the old salience times 0.5 plus the peak creation weight times 0.5, followed by the destruction factor (1 − g). Salience stays in [0, 1] and carries no fluctuation.
4. **The force sees a yes/no mask, not the salience.** The reverse channel receives only the mask (salience above 0.005, with LIFO stack discipline). Salience otherwise enters only as the retention weight of a register's old content.

Each register is one slot holding one content vector: an exclusion (hard-core) object. A bosonic Poisson reading would put up to 26% of a register's probability on double occupancy, which a single slot cannot represent. The diagnostics ask what the trained models do with this machinery. Companion note: [`Doi_Peliti_Dynamics_of_Semantic_Particles_and_Registers.md`](Doi_Peliti_Dynamics_of_Semantic_Particles_and_Registers.md).

**Measurement.** Evaluation only, local CPU, script `debug/dp_register_statistics.py`.
- **Tokens:** 8 × 512 validation tokens, seed 20261005.
- **Arms:** G2 (L=2 Fock, live), the L=4 Fock live arm, and G3 (L=2 with the exchange field), each on its best checkpoint.
- **Hooks:** for every layer, position and register, record the salience that sets that layer's mask (after creation, before destruction), the destruction gate g, the mask, and the register content passed to the reverse channel.

- **DP1, is particle number dynamic at all?** For each layer from 1 up (layer 0 is all-active by construction), measure:
  - the active fraction (active registers out of M = 32), per position;
  - the fraction of (position, register) cells below threshold;
  - the distribution of g.
- **DP2, do registers share content?** Among the active registers at each position, measure:
  - the salience-weighted mean absolute off-diagonal cosine of their contents;
  - the fraction of pairs with cosine above 0.9 (near-duplicates).

  Both are reported separately for earlier positions and for the last position, the only one the 0.05 repulsion penalty acts on in training.
- **DP3, does salience act as intensity?** At each position and layer, take each active register's leave-one-out contribution to the reverse-channel force: the change in that force when the register alone is removed from the mask. Measure the Spearman correlation between salience and that contribution across the active registers, averaged over positions and layers.

**Predictions:**

| test | arm(s) | prediction | called |
| --- | --- | --- | --- |
| DP1 | G2, G3 | active fraction ≥ 0.95 at layer 1: no number dynamics in practice | 65% |
| DP1 | L=4 live | active fraction ≥ 0.95 at every layer from 1 to 3 | 55% |
| DP2 | all | near-duplicate pairs (cosine above 0.9) under 1% of active pairs, at every position | 55% |
| DP2 | all | mean absolute cosine at earlier positions above the last position's by at least 0.05 (the penalty acts only there) | 50% |
| DP3 | all | Spearman ρ below 0.3: salience does not act as intensity | 60% |

**Decision rule.**
- **DP1 holds (active fraction ≥ 0.95).** The trained models implement no particle-number dynamics: the register count stays at M, and v2 reduces to content rewriting with retention weights. The Fock/Doi–Peliti language then describes the architecture's *capacity*, not its trained behavior. Book §10.5.2 (the v2 mapping table) and §20 say so.
- **DP3 weak (ρ < 0.3).** "Salience plays the role of a Poisson mean" is withdrawn. Salience is read as an occupation (retention) probability of a hard-core slot, and the exclusion Doi–Peliti formalism is the correct home. DP3 strong (ρ ≥ 0.5): the intensity reading survives empirically, learned through content even though the force sees only the mask, and the sentence stays with that qualification.
- **DP2 near-duplicates common (≥ 5%).** Registers share content modes: exclusive per slot, bosonic in content, the hybrid statistic of the single-particle note §5.5. The book then says that, not "bosonic."

**Caveat recorded in advance.** Salience is a deterministic mean-field quantity. No measurement on these models can distinguish bosonic from exclusion fluctuations, since there are none. The tests read how the models *use* the variables, not their statistics.

**DP scored, 2026-10-05.** Output: `debug/dp_register_statistics_output.txt` and its JSON; figure `companion_notes/figures/doi_peliti/dp_register_diagnostics.png`. Layer checkpointing was off for the measurement (it recomputes each layer and doubles every hook); the logits are bit-identical with and without it.

| test | G2 | L=4 live | G3 | prediction | outcome |
| --- | --- | --- | --- | --- | --- |
| DP1, active fraction from layer 1 | 0.9995 | 1.0000 (layers 1–3) | 1.0000 | at least 0.95 (65%, 55%) | **hit** |
| DP2, near-duplicates at earlier positions | 1.30% | 0.66–0.94% | 0.98% | under 1% everywhere (55%) | **miss**, narrowly (G2) |
| DP2, absolute cosine earlier minus last | 0.037 | 0.054–0.070 | 0.051 | at least 0.05 everywhere (50%) | **miss** (G2); hit elsewhere from layer 1 |
| DP3, Spearman(salience, own force) | −0.005, −0.061 | −0.022, −0.321, −0.144, −0.407 | −0.022, −0.172 | below 0.3 (60%) | **hit**, and negative |

- **Decision rules triggered.**
  - DP1 holds: the trained models implement no particle-number dynamics.
  - DP3 is weak, in fact negative: "salience plays the role of a Poisson mean" is withdrawn, and salience is read as the retention probability of a hard-core slot.
  - DP2 is far below the 5% line: registers do not share content, so the hybrid-statistic reading does not apply either.
- **Why DP1 was nearly forced.** The mask is taken after the refresh and before destruction, and there the salience is at least (1 − λ) divided by the number of prefix positions. At decay 0.5 and threshold 0.005, no register can be inactive in the first 99 positions.
- **Not pre-registered:**
  - The destruction gate is switch-like at layer 0 (G2 median 0.994, 10th percentile 0.016). It acts as a content reset: the next blend overwrites a low-salience register with fresh readout.
  - The **last layer's destruction gate receives exactly zero gradient** in every model (its output is never read) and sits at its initialization, 0.46–0.54.
  - The median weight of the initial salience after the last layer is 0.0007 (G2), 0.0001 (L=4) and 0.0027 (G3).
  - DP3's negative sign has a mechanism: freshly reset registers carry the current context and dominate the force.
- **Book (next edition):** §10.5.2 and §20 as listed in the companion note's §8.1.


### 5.15 PM1: bosonic Poisson-mode registers — **pre-registered 2026-10-05, before any run**

**Why.** DP1–DP3 (§5.14) showed that the slot registers are exclusion objects, with constant number and salience acting as retention. So the book's bosonic, Poisson-mean Doi–Peliti v2 describes no trained model. PM1 is a register mechanism that has both properties by construction, to find out whether the bosonic v2 can be trained and what it is worth. Companion notes: [`Poisson_Mode_Registers_PM1.md`](Poisson_Mode_Registers_PM1.md) (mechanism, Poisson exactness, conservativity proof, comparison with the slot registers, results), summarised in [`Doi_Peliti_Dynamics_of_Semantic_Particles_and_Registers.md`](Doi_Peliti_Dynamics_of_Semantic_Particles_and_Registers.md) §9.

**Mechanism** (`model_parf_multixi.py`, `poisson_modes`). It has K = 64 shared mode prototypes μ_v, and works in five parts.
- **Creation.** Token s creates particles in mode v at the rate given by its overlap with that mode, E_v(s) = exp(−κ_v²‖h_s − μ_v‖²).
- **Survival.** Each particle survives to the next token with probability λ_v. Half-lives start log-spaced from 4 to 128 tokens and are learnable.
- **Occupation.** The occupation is φ_v(t) = Σ over s < t of λ_v^(t−1−s) E_v(s). This is exactly the Poisson mean of the immigration–death process, and the process is unbounded and bosonic: shared modes, no cap on occupation.
- **Force.** −∇ of U = −Σ_v φ_v(t) a_{l,v} exp(−κ_v²‖h − μ_v‖²). It is linear in φ, so carrying only the mean is exact.
- **Conservativity.** It uses the strict past, so φ is constant in h_t and the force is a gradient in h_t. Like V_φ, it is causal and one-way. The well depths start at 0, so the force is exactly zero at step 0.
- **Switches.** `POISSON_MODES = 64` adds the tag `pm64`, and the parameters get their own clip group at 0.3.

**Verification** (`debug/verify_pm_switch.py` and its output, 2026-10-05).
- **Off:** bit-identical to HEAD on G2, F3.1 and G3′ (logits, loss, every gradient).
- **On:**
  - at zero depth, bit-identical to the model without modes;
  - the force equals −∇U at fixed φ (relative error 2.4e-7);
  - a Monte Carlo immigration–death simulation reproduces φ within 0.5%, with variance/mean 1.001;
  - causal (exactly 0 leak);
  - every new parameter receives a gradient;
  - the clip group catches exactly the four new parameters.

**Arm.** F3.1's configuration (L=2, conservative-only, `REVERSE_CHANNEL = False`, V_φ and ξ live) plus the modes. The ladder then reads F3.1 (57.76) → PM1 → G2 (53.12, slot registers).

**Stage 1, the probe — revised 2026-10-06 to 8,000 steps, before any PM1 data exists** (`PROBE_MAX_STEPS = 8_000`, same schedule, about 4 h). The original gate, ~~126.4 or lower at step 3,000 (at least 1% better than F3.1's 127.73), about 1.5 h~~, is withdrawn at the author's call, for a reason the F3.1 and G2 curves make plain: at 3,000 steps no memory mechanism has yet shown itself. G2 is F3.1 plus the slot registers, and its advantage over F3.1 by step is

| step | F3.1 | G2 | G2 vs F3.1 |
| ---: | ---: | ---: | ---: |
| 3,000 | 127.73 | 123.59 | −3.2% |
| 4,000 | 113.51 | 111.61 | −1.7% |
| 5,000 | 106.71 | 101.17 | −5.2% |
| 6,000 | 103.00 | 93.11 | −9.6% |
| 7,000 | 99.00 | 86.98 | −12.1% |
| **8,000** | **93.57** | **82.92** | **−11.4%** |
| 10,000 | 85.13 | 76.26 | −10.4% |
| 15,000 | 77.31 | 67.91 | −12.2% |

At 3,000 the gap is inside what a single eval resolves (the 4,000 point reads −1.7%; the step-to-step scatter is about ±2%). It opens between 4,000 and 6,000 and is at its stable-phase value, −10 to −12%, from 7,000 on. PM1 has a further reason to be slow: its well depths start at 0, so the force is exactly zero at step 0 and the modes must learn depths, widths and half-lives before they contribute. A null at 3,000 would be close to uninformative; a null at 8,000 means something. The cost is refundable: the probe saves `_step8000_probe_stop.pt` and a passing run continues from it, so the extra steps are paid for only when PM1 fails.

The gate is set against PM1's own base, F3.1, at step 8,000 (93.57), with G2 (82.92, −11.4%) as the descriptive comparator and no expectation that PM1 reaches it:

| measure | gate | called |
| --- | --- | --- |
| val PPL at step 8,000 | **90.8 or lower** (at least 3% better than F3.1): a noticeable difference, above the ±2% eval scatter | 50% |
| val PPL between 90.8 and 92.6 (1–3% better) | a weak signal: the full run at the author's discretion, recorded as a discretionary continuation | — |
| val PPL above 92.6 | fail: no full run | — |
| clip hits on the pm_ group over the probe | under 5% of logged steps, no divergence | 85% |

The step-7,500 eval is read beside the step-8,000 one for the scatter, but the gate is the step-8,000 value. The tag does not carry `PROBE_MAX_STEPS`, so the folder and the resume logic are unchanged. Both criteria must hold for the full run. If the probe passes, clear `PROBE_MAX_STEPS` and the same run continues; the full run goes after the FO 2×2 unless the author moves it.

**Stage 2, the full run (32,500 steps), if the gate passes:**

| measure | prediction | called |
| --- | --- | --- |
| settled PPL | better than F3.1 by at least 1% (57.18 or lower) | 50% |
| settled PPL | between G2 and F3.1 | 45% |
| causality, in-flight and independent | clean | 95% |
| repetition: Spearman of φ for the best-matching mode against the decay-weighted count of earlier occurrences of the same token | above 0.5 | 55% |
| sign of the trained well depths | most positive (attractive wells) | 60% |
| DP3 rerun on the modes: Spearman of φ times depth against each mode's leave-one-out force contribution | above 0.5 (by construction, a sanity check) | 80% |
| **RR-PM1, refinement (6b-7 Gate 3, N = 3 at fixed T, policy `hold`)**, added 2026-10-06 before the probe | at or below F3.1's +143% | 60% |
| **RR-PM2**, same measurement | below +600%, i.e. not in the Fock-arm class (G2 +1,274%, G3 +1,342%) | 85% |

**Refinement readiness, added 2026-10-06 before the probe.** §5.19 makes refinement readiness a required property of every conservative arm, so PM1 is scored on Gate 3 like the others. The base is F3.1, whose Gate 3 is +143% with stiff modes at θ ≈ 4.32 per step; PM1 shares the base, the step and the stiff modes, so it inherits that discretisation error. What it adds is a memory that is accumulated over *tokens*, not rewritten per layer (φ_v(t) sums over s < t and does not depend on the layer index at all), which is the accumulated kind that H-RR (§5.18) predicts refines well. Hence the two calls: the modes should not make refinement worse (RR-PM1), and PM1 should be nowhere near the Fock arms, whose penalty H-RR attributes to per-layer reset-and-rewrite (RR-PM2).

- **Policy for the per-layer depths under refinement.** `pm_depth` is a (L, K) table read as `pm_depth[layer_idx]` in `poisson_mode_force`. Cell 6b-7 remaps the layer index for every refined step through `_fom_policy_index`, and `depth_code`, the gates and V_θ already follow it, so the depths follow the same `hold` policy with no code change: at N = 3 for L = 2 the trained depths are held as [0, 0, 1]. Nothing else in the mechanism is layer-indexed: φ is computed once from the token sequence and is identical at every step count, so refinement changes only how often the well force is applied and at what Δt. Gate 0 (unpatched against patched at N = L) must still pass bit-exactly before any axis is read.
- **Measured on the full run's best checkpoint**, the same checkpoint 6b-7 used for F3.1, so the two numbers compare. On the probe checkpoint (step 8,000) it may be run as descriptive only; an 8,000-step model is not comparable with F3.1's final one.
- **Reading.** RR-PM1 holds: the bosonic memory costs nothing in refinement, and the conservative line keeps it under the §5.19 principle as well as on perplexity. RR-PM1 fails but RR-PM2 holds: the memory costs refinement; the next measurement is Gate 3 with the depths held to the trained Δt-weighted values, to separate the well force's own step-size dependence from the base's. RR-PM2 fails: an accumulated memory refines as badly as a reset one, which counts against H-RR as a general account and is recorded as such in §5.18.
- **If SR2 (§5.19, Test 2) has run before PM1's full run and E1 holds,** the full run should be launched with `LOWRANK_DAMPED_FLOW = True` as well (tag `pm64_sr2`), so that PM1 is scored on the corrected base. That is a change of arm and is the author's call; it is recorded here so the choice is made before the run, not after.

**Decision rule.**
- **Gate fails:** no full run. The book keeps the exclusion statement (Remark 61) and records that a bosonic alternative was tried and added nothing at 3,000 steps.
- **Full run beats F3.1 by at least 1%:** the bosonic Doi–Peliti v2 is trainable and worth something. The book states its price against the slot registers (G2) and that it honours claims 1–3 literally.
- **Full run within 1% of F3.1:** the bosonic mechanism is trainable but adds nothing over ξ-conditioned V_θ at this scale.

**Caveat recorded in advance.** With a coupling linear in φ, no measurement can show bosonic *fluctuations*; what is bosonic is the structure (shared modes, occupation without a cap, an exact immigration–death mean). A sampled variant, drawing n ~ Poisson(φ) in training, is the follow-up that would make the statistics consequential.

#### PM1 probe scored at step 8,000: **79.78 — perplexity PASS, clip MISS on the letter** — **2026-10-06**

Run output `L2probe_arm_none_pm64_8000steps_output.txt`; tag `…cgqk_norc_vplive_xilive_pm64_L2probe…idt4_lr0p0012_noattn`; stopped at `_step8000_probe_stop.pt`.

| step | **PM1** | F3.1 | PM1 vs F3.1 | G2 | PM1 vs G2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3,000 | 122.34 | 127.73 | −4.2% | 123.59 | −1.0% |
| 5,000 | 97.23 | 106.71 | −8.9% | 101.17 | −3.9% |
| 6,000 | 89.73 | 103.00 | −12.9% | 93.11 | −3.6% |
| 7,000 | 84.45 | 99.00 | −14.7% | 86.98 | −2.9% |
| 7,500 | 81.49 | 93.69 | −13.0% | 84.71 | −3.8% |
| **8,000** | **79.78** | 93.57 | **−14.7%** | 82.92 | **−3.8%** |

- **Perplexity gate (90.8 or lower at step 8,000): PASS**, by 11 PPL; step 7,500 agrees. PM1 is below G2, the slot-register model, at every eval from step 3,000 on, which no pre-registered prediction anticipated (the closest, "settled between G2 and F3.1", 45%). The modes add about 25k parameters to 77M, so this is not capacity. It would also have passed the withdrawn 3,000-step gate (122.34 against 126.4).
- **pm_ clip criterion (under 5% of logged steps, no divergence): MISS on the letter.** The pm_ group is the largest-gradient group on 138 of 160 logged steps, and its pre-clip norm is shown at 0.4 or more, certainly above the 0.3 threshold, on 123 (77%). Over steps 2,000–8,000 the norm is steady, median 0.5, p95 0.9, maximum 1.6, with no growth. The aggregate norm peaks at 1.81 and exceeds the global 1.0 on 11 of 160 logged steps (6.9%; F3.1 2 of 160 over the same window). No spike, no watchdog event. The criterion's purpose, no divergence, is met: this is a steady throttle, so the pm_ parameters trained at a capped effective learning rate for most of the probe (the effect checklist item C3 states exactly for scalar groups: persistent joint clipping acts as a learning-rate cut).
- **SCAF** at step 5,000: CLEAN, leak tax −3.8e-5 nats.
- **Stiffness:** ω·Δt median 3.00 at step 8,000 (max 6.81), against F3.1's 4.00 and G2's 2.34 at the same step.
- **Decision (author, 2026-10-06):** the split is accepted. The full run goes ahead, but with the pm_ clip retuned, as a fresh arm (next block), not as a continuation of this checkpoint. The 0.3 value was never tuned for these parameters; it was copied from the other gates. This probe is kept as the 0.3 comparator through step 8,000, and its checkpoint as a fallback: if the retuned arm diverges, this one continues from `_step8000_probe_stop.pt`.

#### PM1 at pm_ clip 1.0: an 8,000-step probe, then the full run — pre-registered **2026-10-06, before the run**

**Why a fresh arm.** Continuing the probe with a different clip would give a hybrid (0.3 for 8,000 steps, then 1.0, with AdamW's moments shaped under the throttle), and, because `POISSON_MODE_CLIP` was not in the tag, it would have resumed into the 0.3 arm's folder under the same name. With its own 8,000-step probe the fresh arm costs about 3.6 h more than a continuation, and that time buys the matched comparison.

**Why 1.0.** Over steps 2,000–8,000 of the probe, the share of logged steps on which the pm_ group's pre-clip norm exceeds a threshold is 91% at 0.3, 40% at 0.5, 11% at 0.75, **2% at 1.0** and 1% at 1.5. At 1.0 the group is clipped on outliers only, within the 5% criterion, at the same threshold as the default group.

**Code (2026-10-06).** Cell 0 appends `pmclip<thr>` when `POISSON_MODES > 0` and `POISSON_MODE_CLIP ≠ 0.3`, conditionally, as `RELAX_FIELD_CLIP` does with `rfclip`; Cell 5b asserts it. Verified by running Cell 0's code at HEAD and in the working copy: the tags of the default, F3.1, the 0.3 probe, G2, G3′, L=4 live and SR2-on-F3.1 are unchanged, a clip of 1.0 with the modes off adds nothing, and the new arm's tag is `…cgqk_norc_vplive_xilive_pm64_pmclip1_L2probe…idt4_lr0p0012_noattn`. This closes the `POISSON_MODE_CLIP` tag defect of the checklist's clip-hygiene block. Cell 6 already passes the knob through (`GRAD_CLIP_OVERRIDES['pm_'] = POISSON_MODE_CLIP`).

**Run.** F3.1's Cell 0 (`REVERSE_CHANNEL = False`, `VPHI_GRAD_PATH = XI_GRAD_PATH = 'live'`) plus `POISSON_MODES = 64`, `POISSON_MODE_CLIP = 1.0`, `PROBE_MAX_STEPS = 8_000`. Seed 0, WSD on the full 32,500-step schedule as every arm.

**Stage 1, a matched probe (added at the author's request, before the run).** The clip-1.0 arm also stops at step 8,000, so the two clip settings are compared at matched steps before either goes to 32,500. The stop costs nothing: the chosen arm continues from its own `_step8000_probe_stop.pt` (about 11 h). Decision rule at step 8,000, against the 0.3 probe's 79.78:

| clip-1.0 at step 8,000 | arm continued to 32,500 |
| --- | --- |
| 78.2 or lower (better by more than 2%) | clip 1.0 |
| 78.2–81.4 (within ±2%) | clip 1.0: equal perplexity, and it meets the clip criterion |
| above 81.4 (worse by more than 2%) | the 0.3 arm, continued from its checkpoint; clip 1.0 is stopped and recorded |
| diverged at any point | the 0.3 arm |

Early stop, descriptive and at the author's discretion: if clip 1.0 is more than 5% behind the 0.3 probe at both steps 5,000 (97.23) and 6,000 (89.73), it may be stopped before 8,000 and the 0.3 arm continued. The losing probe's 8,000 steps are kept as the paired comparison of the two clip settings.

**Predictions:**

| | prediction | called |
| --- | --- | --- |
| PC1 | pm_ clip hits under 5% of logged steps, no divergence (no watchdog trigger) | 80% |
| PC2 | step 8,000 at 81.4 or lower, i.e. not more than 2% worse than the 0.3 probe's 79.78 | 80% |
| PC3 | step 8,000 at 78.2 or lower: the throttle cost more than 2% | 25% |
| PC4 | settled PPL below G2's 53.12 | 50% |

- **The Stage 2 table above carries over unchanged** (settled vs F3.1 and vs G2, causality, repetition, depth signs, DP3 rerun), as do RR-PM1 and RR-PM2, now scored on this arm. Those calls were made before the probe and are not revised. PC4 is new and is made with the probe in hand: PM1 leads G2 by 3.8% at step 8,000, but G2's own lead over F3.1 narrows during the decay (−11.4% at 8,000, −7.5% at 32,500), so a lead at 8,000 need not survive to settling.
- **Reading PC2 and PC3 together.** Within ±2% of 79.78: the throttle did not matter at this scale, and the 0.3 probe's number stands as a fair reading of the mechanism. Better by more than 2%: the throttle was costing perplexity, and every pm_ number from the probe is a lower bound. Worse by more than 2%: the clip was acting as a useful learning-rate cut on these parameters; the settled comparison is then run against the 0.3 arm continued from its checkpoint, and both are reported.
- **If PC1 fails by divergence** (watchdog hard trigger, or a pm_ norm that grows rather than plateaus): stop, record it, and continue the 0.3 arm from `_step8000_probe_stop.pt` instead.
- **PC1–PC3 are scored at the probe stop; PC4, the Stage 2 table and RR-PM1/RR-PM2 on whichever arm the rule continues.** If that is the 0.3 arm, the full-run predictions are scored on it and the clip criterion is reported as missed.

#### PM1 clip-1.0 probe scored at step 8,000: **81.67 — the 0.3 arm continues** — **2026-10-07**

Run output `L2probe_arm_none_pm64_clip1.0_8000steps_output.txt`; tag `…cgqk_norc_vplive_xilive_pm64_pmclip1_L2probe…idt4_lr0p0012_noattn`; stopped at `_step8000_probe_stop.pt`. Same seed, data order and schedule as the 0.3 probe. The printed log is partial (it starts at step 1,550: copying the earlier output was blocked by macOS's clipboard scanner, a false positive); the run's `training_log.jsonl` is complete and is filed beside it in `results/…pm64_pmclip1…/`. It gives step 500 at 459.66 and step 1,500 at 177.84 (clip 0.3: 459.92 and 177.50), and agrees with every printed eval.

| step | clip 1.0 | clip 0.3 | 1.0 vs 0.3 |
| ---: | ---: | ---: | ---: |
| 1,000 | 239.71 | 238.54 | +0.5% |
| 2,000 | 150.18 | 148.12 | +1.4% |
| 3,000 | 125.38 | 122.34 | +2.5% |
| 4,000 | 113.26 | 109.33 | +3.6% |
| 5,000 | 101.28 | 97.23 | +4.2% |
| 6,000 | 93.26 | 89.73 | +3.9% |
| 7,000 | 85.79 | 84.45 | +1.6% |
| 7,500 | 83.54 | 81.49 | +2.5% |
| **8,000** | **81.67** | **79.78** | **+2.4%** |

- **PC1 (pm_ clip hits under 5%, no divergence): HIT.** The pm_ pre-clip norm exceeded 1.0 on 5 of 130 logged steps (3.8%), maximum 1.5. No spike or watchdog event. SCAF CLEAN at step 5,000 (leak tax +1.4e-4 nats).
- **PC2 (81.4 or lower at step 8,000): MISS,** by 0.27 PPL.
- **PC3 (78.2 or lower): MISS.**
- **Decision rule: above 81.4, so the 0.3 arm continues** from its `_step8000_probe_stop.pt`; clip 1.0 is stopped. Clip 1.0 trails at every eval from step 2,000 on, by 1.4–4.4%, so this is not one noisy point.
- **Reading.** The tight clip helped. Under AdamW a constant rescaling of a group's gradient cancels in the update, so the 0.3 clip did not act as a learning-rate cut; what it changed is that every step's pm_ gradient was renormalised to the same norm, so steps with a large pm_ gradient counted no more than quiet ones. The clip criterion of the first probe measured a symptom, not a fault. Both arms remain below G2 at step 8,000 (clip 1.0 by 1.5%, clip 0.3 by 3.8%).
- **Stiffness:** ω·Δt median at step 8,000 is 3.68 (max 7.13) for clip 1.0, against 3.00 for clip 0.3 and 4.00 for F3.1. The looser clip let the stiff modes climb further, which bears on refinement readiness (RR-PM1).
- **Before the full run (author, 2026-10-07):** Cell 6b-15 (PM1 tuning diagnostics: step size, weight decay, occupation scale, placement and width) is run on both probe checkpoints. A knob flagged on both is a property of the mechanism, not of the clip, and is addressed before the 0.3 arm is continued. A local test of the cell on F3.1's state scales, with the modes at initialisation, found the layer-1 states at median norm about 300 against mode centres at about 20, so only 2% of tokens reach a mode at layer 1; whether the trained probes still show this is what the cell reads.

#### Cell 6b-15 on both probe checkpoints: **no knob flagged — the 0.3 arm continues unchanged** — **2026-10-07**

Cell 6b-15 (PM1 tuning diagnostics, ladder notebook) reads a probe checkpoint with its optimiser state and tests four knobs against thresholds written into the cell before it was first run: (A) step size, from Adam's moments; (B) weight decay; (C) occupation scale, by half-life quartile; (D) placement and width, per layer. Both checkpoints at step 8,000; no knob is flagged on either.

| | clip 1.0 | clip 0.3 |
| --- | ---: | ---: |
| val PPL at step 8,000 | 81.67 | 79.78 |
| half-life p50 / p95, tokens (init 4–128) | 16.0 / 33.0 | 15.0 / 32.3 |
| κ²·d p50 (init 1.00) | 0.84 | 0.80 |
| layer-0 depth p50, share positive | −0.017, 6% | −0.022, 6% |
| layer-1 depth p50, share positive | +0.327, 100% | +0.349, 98% |
| layer-1 PM force / conservative force | 5.45 | 6.29 |
| (A) consistency c of pm_depth (noise floor 0.23) | 0.12 | 0.14 |
| (B) pm_depth decay / Adam step; cosine(step, θ) | 0.02; +0.17 | 0.02; +0.42 |
| (D) tokens reaching a mode, layer 0 / 1 | 100% / 100% | 100% / 100% |
| (C) layer-1 force share, short / long quartile | 28% / 13% | 34% / 11% |
| pm_depth share of the pm_ group's gradient norm² | 99% | 99% |

- **Shared by both, so properties of the mechanism:** PM1 acts at layer 1 (attractive wells, force 5–6× the conservative force) and has switched itself off at layer 0 (depths near zero, slightly repulsive). The trained half-lives are short to mid range (none above about 33 tokens), and at layer 1 the short modes carry the most force, the opposite of the count imbalance (C) tested for. PM1's own states sit at median |h| 2.1 (layer 0) and 22.6 (layer 1) against centres at about 18, so every token reaches a mode; the silent-layer concern from the cell's local test on F3.1's state scales does not apply to the trained model. Adam's consistency is below the noise floor for every tensor in both models, so the step size is not limiting.
- **What the clip changed:** only the depths. pm_depth carries 99% of the group's gradient norm, so the group clip is effectively a clip on the 128 depth parameters. Under 0.3 the layer-1 wells are deeper, the memory force stronger, and the loss still pushes the depths outward (cosine +0.42 against +0.17), consistent with per-step renormalisation of the depth gradient helping.
- **Decision (author, 2026-10-07):** continue the 0.3 arm from `_step8000_probe_stop.pt` to 32,500 unchanged (`POISSON_MODE_CLIP = 0.3`, `PROBE_MAX_STEPS = None`). The Stage 2 predictions, PC4, RR-PM1 and RR-PM2 are scored on it. RR-PM1 is now the most informative: the conservative memory force dominates the layer-1 step.
- **Candidate follow-ups, not scheduled:** a clip group of its own for pm_depth; an initial half-life range of 4–32 tokens; removing the layer-0 depths.

#### PM1 full run scored: **55.17 settled — 4.5% better than F3.1, 3.9% behind G2** — **2026-10-08**

Run output `L2_arm_none_pm64_clip0.3_32500steps_output.txt` (steps 8,001–32,500, resumed from the 0.3 probe's step-8,000 checkpoint), filed with the probe log in `results/…pm64_L2probe…/`.

| | **PM1** | F3.1 (conservative-only) | G2 (slot registers) | matched GPT-2 (cosine) |
| --- | ---: | ---: | ---: | ---: |
| settled (last 3) | **55.17** (56.61, 55.17, 53.74) | 57.76 | 53.12 | 49.81 |
| best | 53.67 (step 30,500) | 57.35 | 51.27 | — |
| PPL at step 21,000 (end of the stable phase) | 67.29 | 71.81 | 63.61 | — |
| decay gain, nats | 0.199 | 0.218 | 0.180 | — |
| ω·Δt median at the end | 2.84 | 4.32 | 3.40 | — |

- **Settled better than F3.1 by at least 1% (50%): HIT**, −4.5%.
- **Settled between G2 and F3.1 (45%): HIT.**
- **PC4, settled below G2 (50%): MISS**, +3.9%. PM1 led G2 at step 8,000 (−3.8%) and fell behind at about step 12,000. Its lead over F3.1 held at 9–11% from step 9,000 to 18,000, while G2's lead over F3.1 grew to 12–15%: PM1's memory gives a fixed gain; G2's slot registers kept adding value in mid-training. A model-wide gradient episode at steps 10,000–14,000 (pm_ norm above 1.0 on 25% of logged steps, the global norm above 1.0 on 32%) coincides with the crossover.
- **Causality in flight:** SCAF CLEAN at 10k, 15k, 20k, 25k and 30k (and at 5k in the probe). The step-32,500 audit printed nan for its PPLs, as G3′'s did. The independent check (`debug/causality_check_checkpoint.py … pm64`) is pending.
- **Health:** no watchdog or spike event; the global norm exceeded 1.0 on 47 of 490 logged steps (9.6%, max 2.45); the pm_ pre-clip norm exceeded 0.3 on 265 of 477 steps where pm_ was the top group (56%) and 1.0 on 29.
- **Decision rule: the full run beats F3.1 by at least 1%,** so the bosonic Doi–Peliti v2 is trainable and worth something. The book states its price against the slot registers (3.9%) and that it honours claims 1–3 literally. PM1 is the best conservative model at L = 2: it recovers 56% of the F3.1 → G2 gap (2.59 of 4.64 PPL).
- **The queued conditional applies** (PC4 missed): the clip-1.0 arm continues from its step-8,000 checkpoint before the seed pair, preferably to 32,500 for a settled comparison.
- **Pending, all pre-registered:** RR-PM1 and RR-PM2 (6b-7 Gate 3), the repetition test, the depth signs, the DP3 rerun on the modes (`debug/pm1_post_run_measurements.py`), the conservativity test on the trained weights (`debug/conservativity_test_checkpoint.py … pm64`, with V_φ's router share), and the independent causality check.

#### PM1 refinement and stiffness (6b-7, 6b-13): **RR-PM1 MISS narrowly, RR-PM2 HIT — but PM1 is the least robust off its trained schedule** — **2026-10-08**

Cells 6b-7 and 6b-13 on the full run's best checkpoint (step 30,500, PPL 53.67), filed in `results/…pm64_L2probe…/`. GATE 0 PASS (patched loop bit-identical).

| | F3.1 | G2 | **PM1** |
| --- | ---: | ---: | ---: |
| Gate 1, velocity reset | +34% | +33% | **+10,600%** (5,748 PPL) |
| Gate 2, one extra step at the trained Δt (N = 3) | +92% | +51% | **+951%** |
| Gate 3, refinement at fixed T, N = 3 | +143% | +1,274% | **+149%** |
| Gate 3, N = 4 | +374% | +611% | +16,700% |
| Gate 3, N = 6 | +703% | +1,235% | +6,880% |
| Gate 3, N = 8 | +873% | +1,565% | +3,700% |
| ω·Δt median at the endpoint (6b-13) | 4.32 | 3.40 | **2.81** |

- **RR-PM1 (Gate 3 at N = 3 at or below F3.1's +143%, called 60%): MISS,** by 6 points (+149%).
- **RR-PM2 (Gate 3 at N = 3 below +600%, called 85%): HIT.**
- **The scored point hides the shape.** Beyond N = 3 PM1's refinement penalty is 4–45 times F3.1's and worse than G2's; one extra step (Gate 2) costs ten times what it costs F3.1; and resetting the velocity entering each layer (Gate 1), which costs both comparators a third, destroys the model. PM1 is the most momentum-dependent and least schedule-robust of the three, while having the lowest stiffness, so stiffness is not the cause.
- **Reading, a hypothesis.** The layer-1 wells act with about six times the conservative force (6b-15). The trained step appears to rely on a balance between the incoming momentum and that strong attractive force at exactly Δt = 4; changing either throws tokens off. That makes the layer step a learned map rather than a sample of a flow, however conservative each force is. H-RR (§5.18) predicted that a token-accumulated memory refines like its base: it does at N = 3 and not beyond, so accumulation alone is not sufficient.
- **For the design principle (§5.19):** PM1 improves perplexity on a conservative step but does not satisfy refinement readiness (R) better than F3.1; except at N = 3 it is worse. Candidate tests, not scheduled: Gate 1 per layer (reset only at layer 1); PM1 on the SR2 base; capping the layer-1 well depth or the PM force share.
- 6b-13's printed "pre-registered band [3.3, 4.2] … MISS" belongs to the 2026-09-28 Gen 2 depth question, not to PM1, and is not scored here.
- 6b-9 and 6b-12 ran in their pre-patch form (V_φ and the PM1 wells bundled; R(geo) 0.97, the bundle moves R by −0.27) and 6b-15 read the step-8,000 probe checkpoint; all three are re-run with the patched cells before they are scored.

#### PM1 conservativity, causality and post-run measurements: **the step is a gradient flow; causal; repetition MISS** — **2026-10-08**

All on the full run's best checkpoint (step 30,500), outputs in `results/…pm64_L2probe…/`.

**Conservativity on the trained weights** (`debug/conservativity_test_checkpoint.py`, per token at fixed context; the test was validated first on F3.1, which passes, and G2, whose reverse channel fails with Jacobian asymmetry 1.3–1.9):

| layer | term | Jacobian asymmetry (autograd) | closed-loop work | force | verdict |
| ---: | --- | ---: | ---: | ---: | --- |
| 0 | V_θ | 3.3e-07 | 4.9e-07 | 0.0003 | conservative |
| 0 | V_φ | 2.3e-06 | 2.0e-07 | 0.037 | conservative |
| 0 | PM1 wells | 1.8e-07 | 9.3e-08 | 0.10 | conservative |
| 0 | total | 5.1e-06 | 1.9e-07 | 0.12 | conservative |
| 1 | V_θ | 3.7e-07 | 1.9e-07 | 0.17 | conservative |
| 1 | V_φ | 4.6e-07 | 7.0e-04 | 0.012 | conservative |
| 1 | PM1 wells | 1.7e-07 | 9.3e-08 | 1.26 | conservative |
| 1 | total | 2.6e-07 | 3.0e-04 | 1.24 | conservative |

- **V_φ's straight-through router term,** measured exactly (score-head query detached against as trained), as a share of the total conservative force: 0 at positions t ≥ 16 in both layers; at 3 ≤ t < 16, median 0 and p95 0.98% (layer 0) and 0.009% (layer 1). F3.1 for comparison: p95 2.4% at layer 0 (t ≥ 16) and 15.2% at layer 1 (3 ≤ t < 16). **PM1's step is a gradient flow to within 1% of the force for its worst 5% of tokens, and only at the first 15 positions; cleaner than F3.1.**
- **Scope** (as for F3.1): per token at fixed context (one-way between tokens, no global energy); the potential changes with layer; V_φ's hard top-k selection makes it a gradient piecewise (switches not measured).

**6b-9 (patched, V_φ and the wells separated).** GATE 0 PASS. R(geo) 0.97. The wells carry the non-V_θ part of the step: they move R by −0.258 when added last and −0.262 when added first, V_φ by −0.017 (in F3.1 V_φ moved it by −0.29). At layer 1, V_θ plus the wells reproduce the step's velocity almost exactly (R_v 0.013). **In PM1 the wells have taken over V_φ's role.** *Withdrawn 2026-10-11 (§5.20 CD4): removing V_φ costs PM1 0.324 nats, more than F3.1's 0.282. The wells carry the displacement; V_φ stays load-bearing.*

**Causality** (`debug/causality_check_checkpoint.py … pm64`): future perturbation exactly 0 (5 cut points × 3 draws); batch independence exactly 0; prefix-only scoring max |d logit| 9.8e-2, leak tax +1.7e-3 nats ("CHECK"). **Explained, not a leak** (`pm1_prefix_length_check_output.txt`, `pm1_prefix_discrepancy_by_position_output.txt`): real against random future tokens at equal length give exactly identical logits at every length tested; the discrepancy lives only at positions 3–15 (top_k = 16) and vanishes when V_φ's router term is removed. The straight-through mask (m_hard − k y).detach() + k y uses k = min(top_k, T − 1), so at t < 16 a short prefix (k = t) and the full sequence (k = 16) scale the router term differently. Future content never matters. F3.1 shows the same effect (position 12, 5.9e-3); the causality script samples position 7, where only PM1's is large. Consequence: a small train/inference mismatch at the first 15 positions for every V_φ model. Remedy for future models: k from the per-row count of valid sources, min(top_k, t).

**Post-run measurements** (`debug/pm1_post_run_measurements.py`; scoring fixed in its docstring before the run was scored; scored at layer 1, the larger PM1 force share):

| prediction | called | result |
| --- | --- | --- |
| most trained depths positive | 60% | **HIT** on the letter: 57% of all depths; layer 1 70%, carrying 94% of the PM1 force; layer 0 mixed near zero |
| repetition: Spearman(φ of the best-matching mode, decay-weighted repeat count) > 0.5 | 55% | **MISS**: +0.086 (repeated positions only: −0.08) |
| DP3 on the modes: Spearman(φ·a, leave-one-out force) > 0.5 | 80% | **HIT**: +0.815 (magnitudes +0.99) |

- **Reading:** the modes track regions of semantic space the recent context visited, not token identity. The memory shortened over the full run: median half-life 7.1 tokens (15 at step 8,000), none above about 18; 8 of 64 modes (12%) dead; layer-1 PM1 force 4.7× the conservative force (6b-15 at step 30,500, no knob flagged). A short-horizon memory overlapping ξ's short channels may be why PM1's gain over F3.1 stopped growing after step 9,000.

**Overall for PM1:** causal; a gradient flow on the trained weights, cleaner than F3.1; 4.5% better than F3.1 and 3.9% behind G2 in settled PPL; but not refinement-ready (6b-7: Gate 3 beyond N = 3 and Gates 1–2 far worse than F3.1). **6b-12 (patched), the wells' per-token forcing:** GATE 0 PASS. Deflection of each token's step when the wells are switched off, as a fraction of the step: layer 0 median 0.40 (IQR 0.38–0.43); layer 1 median 0.87 (IQR 0.85–0.89), 97.8% of tokens above 0.75 and 0.5% below 0.25. The wells act on every token, strongly and uniformly, not sparsely; position, semantic mass and loss barely predict it (|ρ| ≤ 0.18). Descriptive: the SPARSE/UNIFORM criteria were written for the reverse channel.

#### PM1 refinement localized: **the layer-1 wells cause both the refinement failure and the momentum dependence** — **2026-10-08**

`debug/pm1_refinement_localization.py` on the best checkpoint, CPU, executing Cell 6b-7's own refinement functions (policy hold) on the first 4 of its 12 batches. Not pre-registered: a diagnostic to choose the next arm. The α = 1 row reproduces Colab's 6b-7 within the subset's noise (53.70 at N = 2 against 53.71; +167% at N = 3 against +149%).

| well depths × α | PPL at the trained N = 2 | Gate 3, N = 3 | Gate 3, N = 4 | Gate 1, velocity reset at layer 1 |
| ---: | ---: | ---: | ---: | ---: |
| 1 (as trained) | 53.70 | +167% | +19,029% | +11,737% |
| 0.75 | 59.91 | +138% | +968% | |
| 0.5 | 76.52 | +85% | +133% | |
| 0 (wells off) | 122.29 | +25% | +102% | +0.3% |
| F3.1, for reference | 56.30 | +143% | +374% | +34% (both layers) |

- **The wells drive the failure.** The refinement penalty falls steadily as the wells weaken; at half strength it is below F3.1's, and with the wells off the remaining V_θ + V_φ dynamics refine far more smoothly than F3.1's.
- **The momentum dependence is the wells' too,** and it is all at layer 1: resetting the velocity entering layer 0 changes nothing (the stack starts at rest), resetting it at layer 1 costs +11,737% with the wells and +0.3% without.
- **The rest of the model co-adapted to the wells:** without them PPL is 122, against F3.1's 56. Post-hoc weakening is therefore a trade-off, not a fix; the question for a trained arm is whether bounding the well strength during training keeps most of α = 1's perplexity with something like α = 0.5's refinement.
- **Candidate next arms:** (a) bound the depths by reparameterisation, a = a_max tanh(a_raw / a_max), which keeps the force an exact gradient (a scaled per-token force budget would not); (b) variable-step training on PM1 (SR4a: N ~ U{2, 3, 4} at fixed T), which targets refinement readiness directly. (a) is pre-registered below (author's choice, 2026-10-08); (b) is not.

#### PM1-cap: bounded well depths — pre-registered **2026-10-08, before the run**

**Why.** The localization above puts PM1's refinement failure and its momentum dependence in the layer-1 wells. At half strength (α = 0.5) Gate 3 is +85% at N = 3 and +133% at N = 4, both below F3.1's, but post-hoc weakening costs perplexity because the rest of the model co-adapted to deep wells. The question for a trained arm: if the depths can never grow deep, does training find a model that keeps most of PM1's gain and refines like α ≤ 0.5?

**Mechanism** (`model_parf_multixi.py`, `poisson_depth_cap`; Cell 0 `POISSON_DEPTH_CAP`). Each well's effective depth is a = c · tanh(a_raw / c), with a_raw the trained `pm_depth`, so |a| < c. The force is −∇U with the same bounded a, so it stays an exact gradient at fixed φ. At initialisation (a_raw = 0) the cap is linear and the arm starts exactly as PM1 does. The tag gains `pmcap<c>` only when the knob is set; Cell 5b asserts tag and model agree and prints "well depths capped at c". Cell 6b-15 and `debug/pm1_post_run_measurements.py` read the effective depths.

**Why c = 0.3.** PM1's trained layer-1 depths at step 30,500 have p05 / p50 / p95 −0.43 / +0.24 / +0.66, max |a| 1.02. A cap of 0.3 leaves the median well almost intact (0.24 → 0.20) and bounds the tails near α = 0.5's scale on the strongest wells (0.66 × 0.5 = 0.33). It already binds during PM1's probe: the layer-1 median was +0.35 at step 8,000.

**Verification** (`debug/verify_pm_cap_switch.py` and its output, 2026-10-08; PM1's trained weights):
- **Off:** with `POISSON_DEPTH_CAP = None`, eval logits, train-mode loss and all 77 parameter gradients are bit-identical to HEAD on PM1's configuration. Every existing arm's tag is unchanged.
- **On** (c = 0.3), all pass:
  1. the tag reads `…pm64_pmcap0p3_L2probe…`, and Cell 5b's banner prints;
  2. the effective depths stay below 0.3 (max 0.2993);
  3. the force equals −autograd ∇U with the capped depths, relative error 1.1e-7;
  4. at zero depth the logits are bit-identical to a model without modes;
  5. causality is exact (the logits at t < 256 don't change when the tokens from 256 on are replaced);
  6. gradients reach `pm_depth` through the tanh.

**Preview, not this arm** (`debug/pm1_refinement_localization.py … cap0p3`, PM1's trained weights capped post hoc; the cap replaces α):

| | PPL at N = 2 | Gate 3, N = 3 | Gate 3, N = 4 | Gate 1, velocity reset at layer 1 |
| --- | ---: | ---: | ---: | ---: |
| PM1 as trained | 53.70 | +167% | +19,029% | +11,737% |
| PM1, depths capped at 0.3 post hoc | 88.53 | +100% | +73% | −19% |

The cap removes the momentum lock and the N = 4 blow-up on weights that were not trained for it, at a 65% perplexity cost. Only training can say how much of that cost the rest of the model recovers.

**Run.** F3.1's Cell 0 (`REVERSE_CHANNEL = False`, `VPHI_GRAD_PATH = XI_GRAD_PATH = 'live'`) plus `POISSON_MODES = 64`, `POISSON_MODE_CLIP = 0.3` (PM1's), `POISSON_DEPTH_CAP = 0.3`. Seed 0, WSD on the full 32,500-step schedule. Base without SR2, so the only difference from PM1 is the cap. It goes on the GPU after SR2 on F3.1. If SR2's E1 holds first, whether PM1-cap runs on the SR2 base instead (tag `…pmcap0p3…sr2`) is the author's call before launch; the predictions below are for the base without SR2.

**Stage 1, the probe.** `PROBE_MAX_STEPS = 8_000` (about 4 h), at matched steps with PM1's probe (79.78) and F3.1 (93.57).

| step-8,000 PPL | next |
| --- | --- |
| 90.8 or lower (at least 3% better than F3.1, PM1's own probe gate) | continue to 32,500 from `_step8000_probe_stop.pt` |
| 90.8–92.6 | weak signal; full run at the author's discretion |
| above 92.6, or diverged | stop; the cap costs the memory its gain |

At the probe stop: Cells 6b-7 and 6b-15 on the probe checkpoint, descriptive. PM1's step-8,000 checkpoint had no 6b-7, so there is no matched refinement comparator. Early warning, at the author's discretion: if 6b-7's Gate 1 at the probe is above +1,000% (PM1's signature), the cap has not removed the momentum lock and the full run may be skipped.

**Predictions:**

| | prediction | called |
| --- | --- | --- |
| CAP0 | step 8,000 at or below 84.0 (within 5% of PM1's probe) | 65% |
| CAP1 | settled at least 1% better than F3.1 (at or below 57.18) | 60% |
| CAP1′ | settled at or below PM1's 55.17 (the cap costs nothing) | 25% |
| CAP2 | Gate 3 at N = 3 at or below F3.1's +143% (6b-7, best checkpoint, policy hold) | 60% |
| CAP3 | Gate 3 at N = 4 at or below F3.1's +374% | 55% |
| CAP4 | Gate 1 (velocity reset) at or below +100% (PM1 +10,600%, F3.1 +34%) | 55% |
| CAP5 | the conservativity test passes as for PM1 (wells exact; router term only at t < 16) | 95% |
| CAP6 | layer-1 PM1 force / conservative force below PM1's 4.7 (6b-15 at the end) | 75% |

- **Compensation, recorded in advance.** The cap bounds depth per unit occupation, not the force, 2κ² φ a E |h − μ|. φ can grow if the half-lives lengthen (λ is trainable; 4–128 tokens is only its initial range), and κ² can sharpen the wells. If CAP3 or CAP4 misses, the first reading is whether the half-life median (PM1: 7.1 tokens) or κ²·d (PM1: 0.80) moved to rebuild the force. 6b-15 and the post-run script report both. Also reported: the share of layer-1 depths with |a_raw| > c, where the tanh saturates.
- **Scoring.** CAP0 at the probe stop. The rest on the full run's best checkpoint, with the same tools as PM1: 6b-7, 6b-13, 6b-15, the patched 6b-9 and 6b-12, `conservativity_test_checkpoint.py … pm64`, `causality_check_checkpoint.py … pm64`, `pm1_post_run_measurements.py` and `pm1_refinement_localization.py`. The local scripts need the cap in their Cell 0 substitutions before they are run on this arm.

**Decision rule.**
- **CAP1 and CAP3 both hit:** PM1-cap replaces PM1 as the conservative memory arm in the book (§37.6 and the abstract) and is the candidate for the HF card, the author's call. PM1 keeps its row as the unbounded comparison.
- **CAP3 hits, CAP1 misses:** bounding trades the memory gain for refinement. Next: (b), variable-step training on PM1, or c = 0.5.
- **CAP3 misses:** a depth bound is not enough. Read the compensation channel above, then (b).

#### PM1-cap amended: on the SR2 base — **2026-10-09, before the run** (author's decision)

**Why.** FLOW-C and FLOW-R (§5.19) settled the base question the original text left open:
- the per-layer flow converges on SR2;
- it does not settle on F3.1's split step (C4);
- PM1, on that split step, diverges under it (C5).

A PM1-cap on the base without SR2 could therefore not become refinement-ready even if the cap works. The arm that can be conservative, refinement-ready and have a memory is PM1-cap on SR2. The original predictions CAP0–CAP6 above were never scored and are **superseded** by those below. The cost, recorded: against PM1 this arm changes two things, the cap and the integrator. Its memory gain is therefore read against SR2, the same integrator without modes.

**Run.** F3.1's Cell 0 (`REVERSE_CHANNEL = False`, `VPHI_GRAD_PATH = XI_GRAD_PATH = 'live'`) plus:
- `POISSON_MODES = 64`;
- `POISSON_MODE_CLIP = 0.3`;
- `POISSON_DEPTH_CAP = 0.3`;
- `LOWRANK_DAMPED_FLOW = True`.

Seed 0, WSD over 32,500 steps.
- **Tag:** `…cgqk_norc_vplive_xilive_pm64_pmcap0p3_sr2_L2probe…idt4_lr0p0012_noattn`. It was checked locally through the notebook's Cells 0–5b. Cell 5b prints both banners, the model carries 64 modes, cap 0.3 and SR2 with γ = 0.1, and forward passes are finite at `substeps_per_layer` 1 and 4.

**Stage 1, the probe.** `PROBE_MAX_STEPS = 8_000`, against SR2 at step 8,000 (92.93) and PM1 (79.78).

| step-8,000 PPL | next |
| --- | --- |
| 90.1 or lower (at least 3% better than SR2) | continue to 32,500 from `_step8000_probe_stop.pt` |
| 90.1–91.9 | weak signal; full run at the author's discretion |
| above 91.9, or diverged | stop: the capped modes add nothing to SR2 |

At the probe stop, descriptive only: Cells 6b-7 and 6b-15.

**Predictions:**

| | prediction | called |
| --- | --- | --- |
| CS0 | step 8,000 at or below 84.0 (within 5% of PM1's probe) | 55% |
| CS1 | settled at least 1% better than SR2 (at or below 57.74) | 55% |
| CS1′ | settled at or below PM1's 55.17 | 20% |
| CS2 | Gate 3 (standard, 6b-7) at N = 3 at or below SR2's 0.405 nats | 45% |
| CS3 | **the per-layer flow converges**, by FLOW-R's criterion on FLOW-R's batches (seed 20261009): \|pen(16) − pen(12)\| ≤ 0.01 and \|pen(12) − pen(8)\| ≤ 0.02 nats | 45% |
| CS3′ | the limit is close to the trained step: pen(16) ≤ 0.25 nats | 40% |
| CS4 | Gate 1 (velocity reset) at or below +100% | 60% |
| CS5 | the conservativity test passes: wells exact, router term only at t < 16 | 95% |
| CS6 | layer-1 PM1 force over conservative force below PM1's 4.7 | 75% |

- **Descriptive, recorded with CS3:** the kick share. PM1's was 0.94 at layer 1, against SR2's 0.51. The cap is expected to bring it down, and CS3 depends on that.
- **Compensation channel, as before:** the half-life median and κ²·d against PM1's 7.1 tokens and 0.80.
- **Scoring tools:** 6b-7, 6b-13, 6b-15, the patched 6b-9 and 6b-12, and the local scripts with `pm64 pmcap0p3 sr2`.
  - `refinement_flow_confirmation.py` needs a kind for this arm. It is added before scoring and checked, as for the other kinds, against the harness and against `substeps_per_layer`.

**Decision rule.**
- **CS1, CS3 and CS3′ hit:** PM1-cap on SR2 is the conservative, refinement-ready model with a memory. It gets an HF card and becomes the arm named in the book (§37.6, the abstract), with SR2 as its no-memory comparison.
- **CS3 and CS3′ hit, CS1 misses:** refinement-ready, but the capped memory adds nothing over SR2. The cap is too tight for this base; next, c = 0.5 on SR2.
- **CS1 hits, CS3 misses:** the memory gain is real but the wells still break the per-layer flow. Next, variable-step training (SR4a) on this arm, or a smaller cap.
- **Both miss:** the Poisson-mode mechanism needs rethinking before any further arm.

#### PM1-cap on SR2, probe scored at step 8,000: **92.89 — the probe gate stops the arm; the capped modes add nothing to SR2** — **2026-10-10**

Run output `L2probe_SR2_on_PM1_8000steps_output.txt`, filed in `results/…pm64_pmcap0p3_sr2_L2probe…/`. Tag and Cell 5b banners as pre-registered; fresh start; stopped at `_step8000_probe_stop.pt`.

| step | PM1-cap on SR2 | SR2 | PM1 | F3.1 | against SR2 | against PM1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 243.52 | 242.50 | 238.54 | 241.67 | +0.4% | +2.1% |
| 2,000 | 152.90 | 152.73 | 148.12 | 151.89 | +0.1% | +3.2% |
| 3,000 | 128.73 | 129.07 | 122.34 | 127.73 | −0.3% | +5.2% |
| 4,000 | 118.41 | 118.78 | 109.33 | 113.51 | −0.3% | +8.3% |
| 5,000 | 107.44 | 108.37 | 97.23 | 106.71 | −0.9% | +10.5% |
| 6,000 | 100.48 | 101.40 | 89.73 | 103.00 | −0.9% | +12.0% |
| 7,000 | 95.25 | 95.60 | 84.45 | 99.00 | −0.4% | +12.8% |
| **8,000** | **92.89** | 92.93 | 79.78 | 93.57 | **−0.0%** | +16.4% |

- **Probe gate (at or below 90.1 to continue; above 91.9 stops): STOP.** The capped arm tracks SR2 within 1% at every eval. The modes, as capped, buy nothing measurable, while uncapped PM1 was 16% ahead by step 8,000.
- **CS0 (step 8,000 at or below 84.0): MISS.** CS1–CS6 are not scored: the full run does not happen.
- **Health:** the global norm exceeded 1.0 on 2 of 160 logged steps (max 1.14). pm_ was the most-clipped group on 30 of 160 steps, against most steps for PM1. No watchdog or spike event.
- **The resonance monitor works** (fresh runtime with the semsimula-diag fix): ω·dt p50 rises from 1.0 at step 500 to 2.1 at step 8,000, with 58% of readings past 2. Under the exact flow this is a stiffness reading, not an instability.
- **The step-5,000 SCAF line is not in the saved log.** The record is in the run's `results/scaf_leak_monitor.jsonl` on Drive.
- **Reading, a hypothesis to test before any further arm.** PM1's gain came from deep layer-1 wells: a median depth of +0.35 at step 8,000, with tails to ±1, about 6 times the conservative force. A cap of 0.3 removes exactly the depth that bought the perplexity. The post-hoc preview already showed that the cap costs 65% on PM1's weights. Training did not find another use for bounded wells: it appears to have let them go.
- **Two readings, separated by Cell 6b-15 on the probe checkpoint** (pre-registered as descriptive at the probe stop):
  - **Saturated:** many layer-1 depths sit at the cap (|a_raw| well above 0.3), and the PM force share is held down. The cap is binding; c = 0.5 would be the next try.
  - **Abandoned:** the depths sit near zero, well inside the cap, and the PM force share is small. Training gave up on the modes on this base, and a larger cap would not help.

**Cell 6b-15 on the probe checkpoint, 2026-10-10** (`Cell-6b-15_PM1cap_on_SR2_…_step8000_output.txt`; no knob flagged): **neither reading as written. The cap binds, the model compensates, and the force it rebuilds buys nothing.**

| at step 8,000 | PM1-cap on SR2 | PM1 |
| --- | ---: | ---: |
| layer-1 effective depth p05 / p50 / p95 | −0.119 / +0.249 / +0.293 | +0.130 / +0.349 / +0.601 |
| raw depth behind p50 / p95 (a_raw = 0.3 atanh(a / 0.3)) | 0.356 / 0.666 | (uncapped) |
| tanh slope at p50 / p95 (the gradient reaching a_raw) | 0.31 / 0.05 | 1 |
| layer-1 PM force / conservative force | 3.13 | 6.29 |
| layer-1 force share, short / long half-life quartile | 10% / **57%** | 34% / 11% |
| κ²·d p50 / p95 | 1.15 / **3.03** | 0.80 / 1.13 |
| half-life p50 / p95, tokens | 16.6 / 42.4 | 15.0 / 32.3 |
| loss direction on pm_depth, cosine(Adam step, θ) | −0.03 | +0.42 (outward) |
| dead modes | 4 of 64 | 0 of 64 |

- **The cap binds.** The upper half of the layer-1 depths sits near 0.3, and at p95 the tanh passes 5% of the gradient back to the raw depth. The loss can no longer push those wells deeper; the cosine of −0.03 against PM1's +0.42 is that blockage, not a lack of demand.
- **The compensation channel recorded in advance happened.** Force moved to the long half-life modes (57% of layer-1 force, against 11% in PM1), whose occupations φ are about four times larger. The wells sharpened (κ²·d p95 3.03 against 1.13). The PM force recovered to 3.1 times the conservative force, half of PM1's.
- **And it buys nothing:** perplexity equals SR2's at every eval. A memory carried by long-lived, sharp, shallow wells is not the memory PM1 used: deep, wide wells driven by the recent context (short half-lives).
- **Reading.** PM1's perplexity gain and its refinement failure come from the same deep layer-1 wells. Bounding the wells removed both, and SR2 alone already supplies the refinement.

**Options, for the author.**
- **(a) c = 0.5 on SR2.** Cheap: a 5 h probe, and the "saturated" follow-up named above. It maps the trade-off, but PM1's useful depths reach 0.6, and wells scaled to 0.75 of PM1's already broke refinement at N = 4 (+968%, localization). Expected: some gain back, refinement at risk.
- **(b) Integrate the wells exactly.** The wells fail refinement because they are a strong explicit kick (0.94 of PM1's layer-1 step). Their Hessian at the token is a sum of per-mode isotropic and rank-1 terms. The stiff part of the wells could join the exactly integrated low-rank flow, as V_θ's stiff modes did under SR2, leaving only a weak remainder in the kick. A design change: implementation, verification and pre-registration before any run.
- **(c) Park the Poisson-mode line.** SR2 stands as the conservative, refinement-ready model; the memory remains a perplexity gain that refinement cannot keep.
- **Author's decision, 2026-10-10: (b).**

#### PMX: the wells integrated exactly — designed, verified and pre-registered **2026-10-10, before the run**

**Phase 0, is (b) supported?** (`debug/pm_wells_curvature.py` on PM1's best weights; its rule of thumb was written into the script before the run.) The wells' Hessian at a token, with φ fixed, is H = αI − W: an isotropic part α = Σ_v 2κ_v² c_v E_v and an indefinite rank-K part W = Σ_v 4κ_v⁴ c_v E_v r_v r_vᵀ.

| layer 1 | p05 / p50 / p95 |
| --- | --- |
| ω·Δt of the isotropic part | 2.00 / 2.10 / 2.18 |
| ω·Δt of the stiffest direction | 2.04 / 2.18 / 2.26 |
| unstable rate × Δt (radial direction; every token has one) | 0.89 / 0.97 / 1.05 |
| force missed by the frozen full quadratic at the kick point, ÷ the force | 0.032 / 0.047 / 0.059 |
| the same with the isotropic part only | 0.207 / 0.272 / 0.342 |

- **The rule of thumb is met:** stiff (ω·Δt at the explicit wall of 2) and faithful (the quadratic leaves 4.7% of the force).
- **Only the full quadratic qualifies;** the isotropic part alone leaves 27%.
- **Layer 0's wells are weak and slightly repulsive** (α < 0 for every token). The design must and does handle negative curvature.

**Design** (`poisson_wells_exact`; Cell 0 `POISSON_WELLS_EXACT`, tag `pmx`; requires SR2 and `langevin_T = 0`):
1. **At the step's start, h₀.** In the per-layer flow this is at the layer's first substep. The wells' quadratic is taken there: F_w(h₀), α and (dW_v, r_v), φ from h₀'s context (`poisson_mode_quadratic`).
2. **V_θ exactly as under SR2.** Its retained low-rank modes (`lowrank_max_modes`, scaled by √κ) and the wells' r_v columns form B. The spring on span(B) is B diag(+1, …, −dW_v) Bᵀ, diagonalised by `indefinite_lowrank_modes`: an orthonormal basis from the hardened Gram path, the restriction shifted to be PSD for the Jacobi SVD, then shifted back. α acts on all of ℝᵈ.
3. **Both A half-steps are exact over all of ℝᵈ** (`lowrank_iso_damped_substep`). Modes have stiffness κ_i + α, the complement α, either sign. Damping is inside the flow and there is no O-step. The force carried is V_θ's retained-mode projection of its low-rank force plus the wells' quadratic, F_w(h₀) − H(h − h₀).
4. **The kick keeps the remainder:** the true total force at h_mid (V_θ, V_φ, the wells with φ at h_mid) minus what the flow carries.
5. **The forward value of every force is unchanged.** As in SR2, the spring's modes are detached and the forces live.
6. **`damped_mode_coefficients` takes negative ω²** through the division branch. For ω² ≥ 0 its expressions are unchanged.

**Verification** (`debug/verify_pm_wells_exact.py` and its output; HEAD from `git archive`):
- **Off:** bit-identical to HEAD on PM1's and SR2's configurations and trained weights (eval logits, train loss, all 77 and 73 gradients), and `damped_mode_coefficients` bit-identical on ω² in {0} ∪ [1e-12, 1e2].
- **On:**
  - **Unit tests (float64):**
    - coefficients for ω² < 0 against RK4: 1.6e-13;
    - indefinite modes: they reconstruct B diag(s) Bᵀ to 2e-12 and are orthonormal;
    - full-space substep against RK4: 6.9e-14;
    - two half steps against one: 6.5e-15.
  - **On PM1's trained weights:**
    - the tag (`…pm64_pmx_sr2…`) and the 5b banner are right;
    - the quadratic's force equals `poisson_mode_force` exactly;
    - PMX and the explicit wells integrate the same ODE (their difference falls at second order, ratio 4.1 per halving of Δt; at the trained Δt they differ by 44%);
    - causality is exact;
    - gradients are finite and reach every pm_ parameter;
    - the per-layer flow takes one quadratic per layer, and k = 1 equals the default;
    - both guards fire.
  - **Kick share (explicit part of the step):**

    | | layer 0 | layer 1 |
    | --- | ---: | ---: |
    | explicit wells | 0.586 | **0.939** |
    | PMX | 0.319 | **0.090** |

  - **Cost:** 2.5 times the explicit step on the CPU (2 × 256 tokens). The GPU cost is unknown until the run starts; SR2 is 1.72 s/step.

**Run.** F3.1's Cell 0 plus:
- `POISSON_MODES = 64`;
- `POISSON_MODE_CLIP = 0.3`;
- `LOWRANK_DAMPED_FLOW = True`;
- `POISSON_WELLS_EXACT = True`;
- **no** depth cap: the point is to keep PM1's deep wells.

Tag `…pm64_pmx_sr2_L2probe…`. Seed 0, WSD over 32,500 steps. The probe stops at `PROBE_MAX_STEPS = 8_000`.
- **Cost watch:** if the first logged steps run above 3.5 s/step (about 32 h for the full run), stop and reconsider. Option: keep only the wells modes that matter per token; anything dropped stays in the kick.

| step-8,000 PPL | next |
| --- | --- |
| 90.1 or lower (at least 3% better than SR2's 92.93) | continue to 32,500 |
| 90.1–91.9 | weak signal; author's discretion |
| above 91.9, or diverged | stop |

At the probe stop, descriptive only: Cells 6b-7 and 6b-15.

**Predictions:**

| | prediction | called |
| --- | --- | --- |
| CX0 | step 8,000 at or below 84.0 (within 5% of PM1's 79.78) | 50% |
| CX1 | settled at least 1% better than SR2 (at or below 57.74) | 55% |
| CX1′ | settled at or below PM1's 55.17 | 30% |
| CX2 | Gate 3 (standard) at N = 3 at or below SR2's 0.405 nats | 40% |
| CX3 | **the per-layer flow converges** by FLOW-R's criterion on FLOW-R's batches: \|pen(16) − pen(12)\| ≤ 0.01 and \|pen(12) − pen(8)\| ≤ 0.02 nats | 50% |
| CX3′ | its limit is within 0.25 nats of the trained step | 40% |
| CX4 | Gate 1 (velocity reset) at or below +100% | 60% |
| CX5 | the conservativity test passes (the force field is unchanged; only its integration differs) | 90% |
| CX6 | the trained model's layer-1 kick share at most 0.3 (PM1: 0.94) | 75% |

**Decision rule.**
- **CX1, CX3 and CX3′ hit:** the conservative, refinement-ready model with a memory exists. It gets an HF card and is the arm named in the book, with SR2 as its no-memory comparison.
- **Scope (added 2026-10-11):** CX1–CX6 score PMX at 532.5M tokens, a third of compute-optimal. A CX3 miss is a statement about this budget (§5.19 scope caveat, RT1–RT3), not about exactly integrated wells.
- **CX3 hits, CX1 misses:** the exactly integrated wells refine but do not pay. The memory gain of PM1 needs its explicit, map-like use of the wells.
- **CX1 hits, CX3 misses:** the gain survives but the flow does not converge. The remainder or the φ dynamics still break it, so read CX6 and the per-layer-flow kick share first.
- **Both miss:** park the Poisson-mode line (option c).
- **Scoring tools:** `refinement_flow_confirmation.py` needs a kind for this arm. It is added before scoring and checked against the harness and against `substeps_per_layer`.

#### PMX amended: a 32-wide eigensolve (`pmx16`) — **2026-10-10, before any data**

**What happened.** The first launch (tag `…pm64_pmx_sr2…`) had not logged step 50 after 22 minutes, more than 26 s/step, and the author stopped it. No checkpoint was written and no eval exists. The cause: cuSOLVER's batched Jacobi eigensolvers handle matrices only up to 32 × 32. SR2's problem is exactly 32 × 32. PMX's was 80 × 80 (V_θ's 16 retained modes plus all 64 wells), twice per token, which leaves the batched path. The CPU test (2.5×) could not see this.

**Phase 0 for the fix** (`debug/pm_wells_curvature.py`, extended). Force missed at the kick point, ÷ the force, layer 1, median:

| what the exact flow carries | missed |
| --- | ---: |
| every well's full quadratic (the first design, 80-wide) | 0.047 |
| every well's force and isotropic curvature α, plus the rank part of the top 24 wells | 0.093 |
| **the same with the top 16 wells (32-wide)** | **0.132** |
| the same with the top 8 | 0.182 |
| only the top 16 wells, whole | 0.291 |
| isotropic part only | 0.272 |

- The top wells are chosen by force, |w_v|·|r_v| at h₀. Choosing by the rank part's weight is slightly worse (0.139 at 16).
- **The stiffness is in α** (ω·Δt 2.10, against 2.18 for the stiffest direction). α costs no eigensolve, so every well's α and force stay in the flow. W only softens directions, and its remainder is left to the kick.

**Change** (`poisson_wells_exact_modes = 16`; Cell 0 `POISSON_WELLS_EXACT_MODES`; tag `pmx<modes>`):
- `poisson_mode_quadratic` returns every well's force and α, and the rank part (r_v, dW_v) of the token's 16 strongest wells, without building the 64-well tensor.
- The exact flow carries F_w(h₀) − (αI − W₁₆)(h − h₀).
- The model refuses to build if `lowrank_max_modes + poisson_wells_exact_modes > 32`.
- The SR2 banner now says there is no O-step under PMX.

**Re-verified** (`debug/verify_pm_wells_exact.py`):
- **Off:** bit-identical to the first PMX commit on PM1 and SR2, and in `damped_mode_coefficients`.
- **Unit tests:** unchanged.
- **On PM1's weights:**
  - consistency: ratio 4.1 per halving of Δt;
  - layer-1 kick share **0.119**, against 0.938 for the explicit wells (the full 80-wide version: 0.090);
  - causality exact; gradients finite and reaching every pm_ parameter;
  - the per-layer flow takes one quadratic per layer;
  - three guards fire;
  - **every eigensolve is 32 wide.**
- **CPU cost:** 1.5 times SR2 per train step and 1.7 times per eval forward at 16 × 512.

**The run** is otherwise as pre-registered above, with tag **`…pm64_pmx16_sr2_L2probe…`** and `POISSON_WELLS_EXACT_MODES = 16`.
- **Cost watch, unchanged:** stop above 3.5 s/step.
- **CX0–CX6 and the decision rule stand.** CX6's threshold (layer-1 kick share at most 0.3) was set before the design's own value on PM1's weights was known (0.119). It is recorded here and not revised.
- The aborted `…pm64_pmx_sr2…` folder on Drive holds no checkpoint and may be deleted.

**The `pmx16` probe launched 2026-10-10 at 2.92 s/step** (SR2: 1.72): under the 3.5 cap. Peak memory 26.1 GB. About 6.5 h to step 8,000.

**Speed work, 2026-10-10, while the probe runs** (the running probe's code is not changed):
- **CPU profile** (`debug/profile_pmx_step.py` and its output; one training step at 4 × 512 on PM1's weights). PMX is 1.40 times the explicit wells (forward 1.67×, backward 1.27×).
  - The spring routine takes 6.5 s of the forward's 5.4 s excess plus recompute: 18 SVD calls against 6.
  - The second occupation pass and the quadratic's tensors add 0.6 s.
  - The backward grows by 4.8 s.
- **A mathematically identical basis** (`indefinite_lowrank_modes_split`, now used by PMX). V_θ's retained modes are already orthonormal from SR2's solve, so only the 16 well directions are orthonormalised against them: a 16-wide Gram solve instead of a 32-wide one. Same operator and span.
  - **Verified** (`debug/verify_pmx_split.py` and its output):
    - the operator agrees to 2e-12;
    - on PM1's weights, logits agree to 2.4e-5, the train loss to 1.9e-6 and every gradient-carrying tensor to 3.3e-5. The one outlier, `score_head.w2.bias`, has a gradient of about 1e-9;
    - the per-layer flow at k = 4 agrees to 1.2e-5, and the kick share is identical (0.1187).
  - **Gain on the CPU:** the spring routine is 1.41 times faster, the train step 9% faster.
- **A GPU profile cell, Cell 6b-16** (smoke-tested on the CPU with CUDA calls stubbed; it changes no weight and restores the switches). For the probe stop: it times plain microbatch steps, PMX against explicit wells, and attributes the GPU time.
- **For the continuation, if the gate passes:** with the repo pulled, the continuation runs the split basis, which is the same model to float rounding. The protocol records which code each segment ran. Larger savings change the model and would be separate arms: PMX at layer 1 only (layer 0's wells are about 0.1 of the conservative force), or 8 wells in the eigensolve.

#### PMX probe scored at step 8,000: **79.43 — the gate passes; PMX keeps PM1's memory gain on the exact flow** — **2026-10-11**

Run output `L2probe_PMX16_on_SR2_8000steps_output.txt` in `results/…pm64_pmx16_sr2_L2probe…/`. Steps 1–8,000 ran the original 32-wide basis (`indefinite_lowrank_modes`); the split basis was written during the run.

| step | **PMX on SR2** | SR2 | PM1 | PM1-cap on SR2 | against SR2 | against PM1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 237.99 | 242.50 | 238.54 | 243.52 | −1.9% | −0.2% |
| 2,000 | 147.28 | 152.73 | 148.12 | 152.90 | −3.6% | −0.6% |
| 3,000 | 122.15 | 129.07 | 122.34 | 128.73 | −5.4% | −0.2% |
| 4,000 | 108.60 | 118.78 | 109.33 | 118.41 | −8.6% | −0.7% |
| 5,000 | 96.46 | 108.37 | 97.23 | 107.44 | −11.0% | −0.8% |
| 6,000 | 88.68 | 101.40 | 89.73 | 100.48 | −12.5% | −1.2% |
| 7,000 | 83.46 | 95.60 | 84.45 | 95.25 | −12.7% | −1.2% |
| **8,000** | **79.43** | 92.93 | 79.78 | 92.89 | **−14.5%** | **−0.4%** |

- **Probe gate (at or below 90.1): PASS,** by 10.7 PPL. The arm continues to 32,500 from `_step8000_probe_stop.pt`.
- **CX0 (step 8,000 at or below 84.0, called 50%): HIT.**
- **The memory gain survives exact integration.** PMX tracks PM1 within 1.2% at every eval and is slightly ahead throughout. Where the depth cap erased the gain, integrating the deep wells exactly kept it.
- **Causality:** SCAF CLEAN at step 5,000 (future perturbation 0, leak tax +2.2e-5 nats).
- **Health:** no watchdog or spike event. The global norm exceeded 1.0 on 24 of 160 logged steps (15%, max 1.94), more than SR2 (2 of 160) or the capped arm (2 of 160). pm_ was the most-clipped group throughout, as in PM1.
- **Speed:** 2.82 s/step over the probe.
- **The resonance monitor was empty throughout.** semsimula-diag hooks `lowrank_cfc_substep` and `lowrank_damped_substep`, not PMX's `lowrank_iso_damped_substep`. Training is unaffected, but Cell 6b-13 will need the same one-line hook before it can read ω·Δt on this arm.
- **Not yet received:** Cell 6b-15 (pre-registered as descriptive at the probe stop) and the GPU profile, Cell 6b-16.

**At the probe stop, descriptive** (filed in the arm's results folder):

| step 8,000 | **PMX** | PM1 | PM1-cap on SR2 |
| --- | ---: | ---: | ---: |
| 6b-15: layer-1 depth p05 / p50 / p95 | −0.097 / +0.241 / +0.468 | +0.130 / +0.349 / +0.601 | −0.119 / +0.249 / +0.293 |
| 6b-15: layer-1 PM force / conservative force | **7.54** | 6.29 | 3.13 |
| 6b-15: layer-1 force share, short / long half-life quartile | **38% / 5%** | 34% / 11% | 10% / 57% |
| 6b-15: κ²·d p50 / p95 | 0.85 / 1.41 | 0.80 / 1.13 | 1.15 / 3.03 |
| 6b-15: loss direction on pm_depth | +0.37 (outward) | +0.42 (outward) | −0.03 (blocked) |
| 6b-7: Gate 3, N = 3 / 4 / 8 (nats) | 0.584 / 0.764 / 1.721 | 0.912 / 5.124 / 3.638 (final) | — |
| 6b-7: Gate 1, velocity reset | **+30,600%** | +10,600% (final) | — |

- **PMX learned PM1's memory, not the capped arm's.** Deep, wide layer-1 wells, driven by the recent context: short half-lives carry the force, κ² is unsharpened, the loss still pushes the depths outward, and no mode is dead. The force is even stronger than PM1's (7.5 times the conservative force). That fits the gain matching PM1's.
- **Standard refinement already beats PM1 and, at N ≥ 4, SR2's final weights,** at step 8,000. PM1's blow-up at N = 4 (5.12 nats) is gone (0.76). N = 3 (0.584) is above SR2's 0.405, the CX2 threshold.
- **The momentum dependence is larger than PM1's.** Gate 1 is +30,600%: the trained step still balances the incoming velocity against the deep layer-1 wells, which exact integration does not change. CX4 (Gate 1 at most +100%) is on course to miss.
- **These readings describe the probe and score nothing.** CX2–CX6 are scored on the final weights; CX3 (per-layer flow convergence) is the (R) test.

**Continuation, 2026-10-11:** a fresh session resumed from `_best.pt` (step 8,000, the same weights and optimizer state as `_step8000_probe_stop.pt`). Tag and Cell 5b checks as at the probe. The session loaded the split basis: the profile below labels its routine, and the SVD calls match the split path. Steps 8,001 onwards therefore run `indefinite_lowrank_modes_split`, the same model to float rounding.

**GPU profile, Cell 6b-16** (A100, one microbatch of 16 × 512, forward and backward; `Cell-6b-16_PMX_GPU_step_profile_output.txt`):

| | explicit wells on SR2 | PMX |
| --- | ---: | ---: |
| per microbatch step | 0.781 s | 1.351 s (1.73×) |
| per optimiser step (2 microbatches, before eval and logging) | 1.56 s | 2.70 s |
| SVD kernel time (batched Jacobi 32×16, float64) | 166 ms, 6 calls | **359 ms, 18 calls** |
| exact substeps | 175 ms | **327 ms** |
| occupation, quadratic, wells force | 20 ms | 57 ms |

- **The eigensolves are a third of PMX's extra time.** Each batched SVD costs about 20 ms whatever its width up to 32: the kernel works in fixed 32 × 16 tiles. So the split basis's 16-wide solve costs on the GPU what the 32-wide one did. Its CPU gain does not carry over.
- **The exact substeps are a quarter.** PMX projects onto 32 modes rather than 16, and evaluates `damped_mode_coefficients` (float64, with a 40-term series computed for every element under `torch.where`) on the span and the complement.
- **Identical-math levers left:**
  - an orthonormal well basis from a batched Cholesky of the 16 × 16 Gram instead of its SVD, saving 6 of the 18 SVD calls;
  - evaluating the coefficients' Taylor series only where it is selected.
  - Estimated together: about 0.3–0.4 s per optimiser step.
- **The model-changing lever:** PMX at layer 1 only, which halves PMX's extra work.
- **The running continuation is not changed.**

**Identical-math speed-ups, implemented and verified 2026-10-11, for any resume or later PMX arm** (`debug/verify_pmx_fast.py` and its output):
- **(a) `indefinite_lowrank_modes_chol`, now what PMX calls.** The well basis comes from two Cholesky QR passes instead of a Gram SVD, one batched SVD fewer per PMX layer step. The well directions outside span(U₀) are far from dependent: worst condition number 8.7 on PM1's states.
- **(b) `series_terms_for`, PMX's substep only.** `damped_mode_coefficients` takes a term count, default 40, so SR2's path is unchanged. PMX passes the count needed at γ·Δt/2: 12 terms at 0.2, chosen on the host.
- **Checks:**
  - Cholesky QR is orthonormal to 8.9e-16;
  - on the real states the Cholesky and split routines' operators agree to 8.2e-13. Both reconstruct L to 1.2e-5, the float32 floor of U₀;
  - the short series equals the 40-term sum exactly on its branch, and the default call is unchanged;
  - on PM1's weights, against the split path, logits agree to 1.0e-6, the train loss exactly, gradients to 2.8e-5, and the per-layer flow at k = 4 to 5e-7; the kick share is identical (0.1187).
- **Full PMX verification re-run** (`debug/verify_pm_wells_exact.py`): off still bit-identical to HEAD on PM1 and SR2 and in `damped_mode_coefficients`; all on-checks pass, eigensolves 32 wide.
- **CPU:** a train step 8% faster than the split path.
- **GPU estimate from 6b-16:** about 0.25 s per optimiser step from (a) and some of the substep's 0.15 s from (b).
- **Code by segment,** recorded for scoring: steps 1–8,000 ran the original basis; 8,001 onwards in the current session run the split basis. A resume in a fresh runtime after these are pushed would run (a) and (b). All are the same model to float rounding.

#### PMX full run scored (Colab part): **54.63 settled — CX1 and CX1′ hit; Gate 1 +91%; standard refinement worse than at the probe** — **2026-10-11**

Run log `L2_PMX_32500steps_output.txt`; Cells 6b-7, 6b-9, 6b-12, 6b-13 and 6b-15 on the best checkpoint (step 30,500, 53.15). Every SCAF audit CLEAN with future perturbation exactly 0. The final audit's `honest_PPL=nan` is a property of the end-of-run audit, not of PMX: SR2's and PM1's final audits read nan the same way, in 8–9 s against about 29 s for a full audit.

**Scope (the author, 2026-10-11).** PMX is trained on 532.5M tokens, a third of compute-optimal (§5.19 scope caveat). A refinement failure here, or a half-way result, says nothing about whether a fully trained PMX-PARFLM refines.

| | prediction | result | |
| --- | --- | --- | --- |
| CX1 | settled ≤ 57.74 | **54.63** (mean of 56.07, 54.61, 53.21) | **HIT** |
| CX1′ | settled ≤ PM1's 55.17 | 54.63 | **HIT** (called 30%) |
| CX2 | Gate 3 at N = 3 ≤ 0.405 nats | **1.097** (53.24 → 159.41) | **MISS** |
| CX4 | Gate 1 ≤ +100% | **+91%** (53.24 → 101.83) | **HIT** |
| RT1 | Gate 3 at N = 3 falls from the probe's 0.584 | **rises** to 1.097 | **MISS** |
| CX3, CX3′, CX5, CX6 | — | local, on the downloaded folder | pending |

Settled against the other L = 2 arms: −6.3% against SR2 (58.32), −5.4% against F3.1 (57.76), −1.0% against PM1 (55.17), +2.8% against G2 (53.12). **The best conservative model on the ladder**, 2.8% behind the Fock model.

**6b-7.** Gate 3 penalties (nats) at N = 3, 4, 6, 8: 1.097, 1.172, 2.185, 2.579 (probe: 0.584 and 0.76 at N = 3 and 4). The successive changes shrink from N = 3 to 4 (12.4 PPL) and then grow, so the cell reads neither FLOW nor monotone. Gate 2 (more hops at fixed dt) is catastrophic: 2.36 nats at N = 3. **Gate 1 fell from +30,600% at the probe to +91%**: the momentum lock at the layer-1 wells was a mid-training state, and exact integration has removed it by the end of training (PM1, explicit wells, ended at +10,600%).

**RT1 and the training-budget hypothesis.** The first within-run reading goes against "refinement improves with training": the standard penalty at N = 3 nearly doubled between step 8,000 and the best checkpoint, while stiffness rose (resonance monitor ω·Δt p50 2.37 at step 8,500, 2.70 at 32,500). It is confounded with the schedule: step 8,000 is mid-stable-phase and the best checkpoint is after the decay, which sharpens the trained map. It also concerns the *standard* refinement, which keeps per-step LayerNorm and re-linearisation. The question for the architecture is the per-layer flow (CX3), and the clean test of the budget is RT3.

**6b-9** (descriptive). R(geo) 1.020; LayerNorm moves R by −0.355, **the wells by −0.518** (−0.522 added first), V_φ by −0.0025. At layer 1, V_θ plus the wells reproduce the step's velocity almost exactly (R_v 0.014). V_φ's small displacement is not a reliance reading (§5.20); a removal cost on PMX is to be added.

**6b-12** (descriptive). The wells' per-token forcing is uniform: at layer 1, 97.7% of tokens have R > 0.75, IQR [0.87, 0.91].

**6b-13** (descriptive). ω·Δt p50 2.65 (layer 0 2.19, layer 1 4.21), 90% past 2, below PM1 (2.81) and F3.1 (4.32). The generic L = 2 band [3.3, 4.2] misses.

**6b-15.** All four tuning checks not indicated. No dead modes (PM1 had 8 of 64). Half-lives p50 9.1 tokens. Layer-1 wells force / conservative force 4.79 (layer 0: 0.81). About half the depths are positive at each layer (50%, 48%): the wells learned both attractive and repulsive modes.

**What follows.**

- **Decision rule: CX1 hits; CX3 decides** between "the conservative, refinement-ready model with a memory exists" and "the gain survives but the flow does not converge". `refinement_flow_confirmation.py` now has the `pmx` kind (split_stack `pmq_freeze`); its checks pass bit-exactly on PM1's weights with PMX on, including the per-layer flow at N = 8 against the model's own `substeps_per_layer = 4` path, and the `sr2` kind is unchanged.
- **SR3-on-PMX loses its new rationale.** It rested on the layer-1 momentum lock (Gate 1 +30,600%), which has gone (+91%). Mode-resolved friction is still the route to settling (Gate 2 at 2.36 nats), so SR3 and SR3-L4 are re-aimed at (S), to be re-registered after CX3.

#### PMX CX3, CX3′ and CX6 scored (FLOW-R): **CX6 hits; CX3 misses narrowly, CX3′ misses — the per-layer flow converges, slowly, to a limit 0.39 nats from the trained step** — **2026-10-11**

`debug/refinement_flow_confirmation.py … pmx seed=20261009` (FLOW-R's criterion and batches; output in PMX's results folder). Every check exact on the trained PMX weights, including the per-layer flow at N = 8 against the model's own `substeps_per_layer = 4`.

| penalty ln(PPL_N / PPL_2), nats | N = 3 | 4 | 6 | 8 | 12 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PMX, per-layer flow | 1.711 | 0.841 | 0.292 | 0.415 | 0.386 | 0.387 |
| SR2, per-layer flow (FLOW-R) | 0.286 | 0.206 | 0.084 | 0.103 | 0.104 | 0.104 |
| PMX, as trained (standard) | | | | 2.464 | | |

- **CX6 (layer-1 kick share ≤ 0.3, called 75%): HIT,** 0.288 (layer 0: 0.319; PM1: 0.94).
- **CX3 (|pen16 − pen12| ≤ 0.01 and |pen12 − pen8| ≤ 0.02, called 50%): MISS on the letter.** 0.001 passes; 0.029 fails the second condition by 0.009. The flow does reach a limit between N = 12 and 16, more slowly than SR2's, which had settled by N = 8. Not revised.
- **CX3′ (limit within 0.25 nats of the trained step, called 40%): MISS,** 0.387 (SR2: 0.104).
- **What converges and what does not.** The per-layer flow removes most of the standard refinement's damage (2.46 → 0.42 nats at N = 8), and its N = 12 and 16 values agree to 0.001. Its coarse end is non-monotone (N = 6 below N = 8), as SR2's is (0.084 against 0.103).

**Decision rule** (pre-registered, unchanged): CX1 hits and CX3 misses, so "the gain survives but the flow does not converge" under this criterion. The rule's first branch (CX1, CX3 and CX3′ all hit: the HF card and the arm named in the book, with SR2 as its no-memory comparison) is **not met**. SR2 remains the refinement-ready arm; PMX is the best conservative model by perplexity. CX6 has been read as the rule asks: the explicit share is within its bound, so the slow convergence is not the kick's doing at the trained step; the remaining candidates are the frozen wells' quadratic over a share (§5.15 Phase 0: 13% of the force missed at the kick point) and the φ dynamics.

**Scope.** 532.5M tokens, a third of compute-optimal (§5.19 scope caveat): this is a statement about PMX at this budget, not about exactly integrated wells.

#### PMX CX5 and causality scored: **CX5 HIT — every force term conservative; causality CLEAN** — **2026-10-11**

- **CX5 (the conservativity test passes, called 90%): HIT.** `debug/conservativity_test_checkpoint.py … pm64 sr2 pmx16` on the best weights, per token at fixed context, five tokens per layer: V_θ, V_φ, the wells and the total are conservative at both layers. Worst Jacobian asymmetry 2.4e-4 (V_φ, layer 1; the wells at most 2.2e-7), worst closed-loop work 3.3e-4 (the total, layer 0). Train-mode and eval force values identical.
- **V_φ's straight-through router term** (the non-gradient part of its source selection) is a larger share than in SR2: at layer 0, t ≥ 16, median 0.65% and p95 2.2% of the total conservative force (SR2: at most 0.009%); at layer 1 it is at most 0.009%. Reported, not a CX5 criterion.
- **Causality** (`debug/causality_check_checkpoint.py … pm64 sr2 pmx16`): future perturbation exactly 0 at five cut points × 3 draws; batch independence exactly 0; prefix-only scoring max |Δlogit| 6.0e-3, leak tax −1.0e-5 nats (the router's known short-prefix effect): CLEAN. Local validation PPL 54.06 (load check).

**A robustness fix found on the way, identical math.** The removal-cost run (wells switched off) stopped on `linalg.svd` failing to converge in `indefinite_lowrank_modes_chol`: with all 16 well weights zero, the shifted restriction has a 16-fold repeated eigenvalue, which LAPACK's SVD can fail on. The first token of every sequence (zero occupation) is the same case in ordinary use; it had never failed. `_psd_eig_desc` now falls back to `eigh` only when the SVD raises; whenever the SVD succeeds the result is the SVD's, bit for bit, so every result above and the trained model reproduce unchanged. Unit test on a half-degenerate batch with the SVD forced to fail: reconstruction 4.7e-12 and orthonormality 3e-15 on both paths, eigenvalues agreeing to 1.2e-12.


### 5.16 F0: is there a shared floor near 50 PPL? The stable-phase extrapolation — **pre-registered 2026-10-06, before any fit**

**Why.** Among the L=2 models with the register path, G2 (53.12), G3 (54.21) and G3′ (52.90) settle within 2.5% of each other, whatever else is switched on. L=4 Fock (50.10) and the 8-layer matched GPT-2 (49.81) end near 50. Is the floor set by depth, or by what every model shares: d = 384, the untied head and 532M tokens?

**Measurement** (`debug/f0_floor_fit.py`, CPU):
- **Fit** L(t) = L∞ + A·t^(−α), in validation loss (nats), to each model's evals in the constant-learning-rate stable phase: steps 3,000 to 21,000 inclusive, every 500 steps, ending before the WSD decay at 21,125.
- **Models:** F3.1, G2, G3, G3′ and L=4 Fock. All five share the identical WSD schedule, learning rate, batches and token budget.
- **Uncertainty:** a 90% interval on L∞ from 2,000 residual-bootstrap refits. A fit is unidentified if its L∞ interval is wider than 0.3 nats, or if α runs into a bound.
- **The matched GPT-2 is excluded from the comparison.** It used a cosine schedule (6e-4 decaying from step 2,000) with no constant-rate phase, so a stable-phase L∞ is not defined for it, and a fit to its decaying curve would be biased low. It is reported descriptively only.
- **What it reads.** These are constant-learning-rate asymptotes, not decayed ones: the WSD decay adds a further drop of its own. The comparison is between models, not against settled values.

**Predictions:**

| | prediction | called |
| --- | --- | --- |
| P1 | G2, G3 and G3′ have L∞ within 0.05 nats (about 5% in PPL) of each other | 65% |
| P2 | the L=2 Fock models' L∞ are above L=4's by more than their combined 90% intervals (a depth-set floor) | 45% |
| P3 | F3.1's L∞ is above the L=2 Fock models' by more than the intervals | 60% |
| P4 | at least one fit is unidentified (the window is short and the eval noise is about ±2 PPL) | 50% |

**Decision rule.**
- **P2 holds:** the floor is set by depth at L=2. F1, the doubled token budget, then asks whether L=2's floor moves with data.
- **P2 fails, with identified fits:** the L=2 and L=4 models extrapolate to a common floor, so the floor is shared (budget or width). F1 and F2 test which.
- **The fits are unidentified:** F0 is inconclusive, and F1 is the test.

**F0 scored, 2026-10-06.** `debug/f0_floor_fit.py` and its output and JSON; 37 evals per model.

| model | L∞ (nats) | 90% interval | PPL∞ | PPL interval | α | identified | PPL at step 21,000 |
| --- | ---: | --- | ---: | --- | ---: | --- | ---: |
| F3.1 | 3.256 | [2.209, 3.636] | 25.9 | [9.1, 37.9] | 0.24 | **no** | 71.81 |
| G2 | 3.339 | [2.965, 3.549] | 28.2 | [19.4, 34.8] | 0.33 | **no** | 63.61 |
| G3 | 3.898 | [3.805, 3.967] | 49.3 | [44.9, 52.8] | 0.61 | yes | 65.22 |
| G3′ | 3.875 | [3.772, 3.944] | 48.2 | [43.5, 51.6] | 0.62 | yes | 63.45 |
| L=4 Fock | 3.763 | [3.661, 3.839] | 43.1 | [38.9, 46.5] | 0.60 | yes | 60.80 |
| GPT-2, cosine (descriptive only) | 3.557 | [3.434, 3.647] | 35.0 | [31.0, 38.4] | 0.45 | yes | 56.35 |

- **Scoring:**
  - **P1 (the L=2 Fock models agree within 0.05 nats): MISS,** spread 0.56 nats. It is driven by G2's unidentified fit; the two identified L=2 fits, G3 and G3′, agree within 0.023.
  - **P2 (a depth-set floor): MISS.** The identified L=2 intervals overlap L=4's, narrowly: G3's lower bound is 3.805 against L=4's upper bound of 3.839.
  - **P3 (F3.1 above the Fock models): MISS,** since F3.1's fit is unidentified.
  - **P4 (some fit unidentified): HIT.** F3.1's and G2's intervals are 1.4 and 0.58 nats wide: their stable-phase curves do not pin the asymptote down.
- **Reading under the decision rule: inconclusive, and F1 is the test.**
  - **What it supports is relative only.** Fitted the same way on the same window and schedule, G3 and G3′ extrapolate to a constant-learning-rate asymptote about 11% above L=4's (about 48–49 against about 43 PPL), with intervals that just overlap. That hints at a depth dependence. It is not a floor.
  - **It does not support any absolute floor for L=2, for four reasons** (corrected 2026-10-06 after an overstatement in discussion):
    1. Only G3 and G3′ are identified; G2's and F3.1's fits leave much lower asymptotes possible.
    2. L∞ is the asymptote at constant learning rate. The final WSD decay removes that rate's noise: in G3′ it took the loss from 63.45 at step 21,000 to 52.90 settled. A longer run that decays at the end would land well below L∞.
    3. Extrapolating a three-parameter fit far beyond its sixfold window is unreliable, because L∞ and α trade off.
    4. A much longer run would also be re-tuned (learning rate, schedule) and would reach a tokens-per-parameter regime these runs never see.
  - So nothing here says what an L=2 model reaches at, say, 8B tokens. Whether L=2 levels off sooner than L=4 is what F1 measures directly.
- **GPT-2** is not compared: its cosine curve is not a constant-rate curve.

### 5.17 W1: the matched GPT-2 on the ladder's WSD schedule — **pre-registered 2026-10-06, before the run**

**Why.**
- **The schedules differ.** The published matched GPT-2 (49.81 settled) used a cosine schedule: warmup to step 2,000, then decay from 6e-4 to 6e-5 throughout. Every Fock arm used WSD: warmup over 5%, constant to 65%, then a cosine decay.
- **So every GPT-2 comparison mixes architecture with schedule.** That includes "L=4 reaches parity with GPT-2" (50.10 against 49.81). F0 (§5.16) also could not compare GPT-2, since its curve has no constant-rate phase.
- **W1 removes the schedule difference** for about 2.5 GPU hours, and is a prerequisite for a clean L=8 comparison.

**The run.**
- `colab_matched_gpt2_baseline_openwebtext.ipynb` with `LR_SCHEDULE = 'wsd'`; everything else at the published baseline's values (d = 384, L = 8, tied, peak 6e-4, floor 6e-5, weight decay 0.1, batch 32 × 512, 32,500 steps).
- The schedule's shape is the only change. The tag `_wsd` gives it its own folders.
- **Verified 2026-10-06:**
  - with the default, the tag is unchanged and the schedule is identical to HEAD at all 32,500 steps;
  - with `wsd`, the schedule equals the Fock ladder's formula (same peak and floor) at every step.

**Predictions:**

| | prediction | called |
| --- | --- | --- |
| W1.1 | settled PPL within −3% to +2% of the cosine baseline (48.3–50.8) | 70% |
| W1.2 | settled PPL at or below the cosine baseline's 49.81 | 55% |
| W1.3 | the F0 fit on its stable phase (steps 3,000–21,000) is identified | 75% |
| W1.4 | that fit's L∞ lies below L=4 Fock's 90% interval (under 3.661 nats) | 50% |

**Decision rule.**
- **Within ±2% of 49.81:** the "parity at equal tokens and width" reading of L=4 is robust to the schedule, and the book's sentence stands.
- **More than 2% better:** the published parity was partly a schedule effect in Fock's favour. The book's L=4 comparison is restated against the WSD baseline, and the cards updated.
- **More than 2% worse:** WSD at this peak suits GPT-2 less well. The cosine baseline stays as the reference, and the L=8 comparison uses whichever schedule is better for GPT-2, stated as such.
- **Optional W2:** the same run at the ladder's peak learning rate (1.2e-3, tag `_wsd_lr0p0012`) separates peak from shape. Queued, not scheduled.

**W1 scored, 2026-10-06.** Run output `gpt2_matched_baseline_wsd_schedule_output.txt`; fit `debug/w1_f0_fit.py` and its output and JSON. The fit reuses F0's functions unchanged and parses the 65 printed evals (loss@512, 4 decimals), because the run's `training_log.jsonl` was not downloaded.

- **Result.** Final and best eval 48.67 at step 32,500. Settled (the mean of the last three evals: 49.00, 48.78, 48.67) **48.82**, which is **−1.99%** against the cosine baseline's 49.81.
- **W1.1 (settled within −3% to +2%, 48.3–50.8): HIT.**
- **W1.2 (at or below 49.81): HIT.**
- **W1.3 (stable-phase fit identified): HIT.** L∞ 4.038 nats, 90% interval [4.031, 4.045]; PPL∞ 56.74 [56.33, 57.14]; α 0.93; rmse 0.003.
- **W1.4 (L∞ below L=4 Fock's interval, under 3.661): MISS, in the opposite direction.** GPT-2's L∞ lies above L=4's whole interval [3.661, 3.839], and above G3's and G3′'s.
- **Decision rule: within ±2%, so the published parity reading of L=4 stands, and the book's sentence stands.** The margin is 0.01 percentage points: the rule is met by the letter and sits on its boundary in substance. Against the WSD baseline, L=4 Fock (50.10) is +2.6%; against the cosine baseline it is +0.6%.

**What W1 shows about F0: a stable-phase L∞ is not a floor.** The same run ended 0.15 nats (8 PPL) *below* its own stable-phase asymptote, because the asymptote is the constant-rate plateau and the decay removes that rate's noise. This confirms point 2 of F0's correction directly. It also makes L∞ unfit for comparing architectures trained at different peaks: GPT-2 ran at 6e-4, the Fock arms at 1.2e-3, and the plateau includes each rate's noise excess.

**Descriptive, same schedule shape and token budget:**

| model | PPL at step 21,000 (end of the stable phase) | settled | decay gain (nats) | stable-phase PPL∞ |
| --- | ---: | ---: | ---: | ---: |
| GPT-2, WSD (W1), peak 6e-4 | 64.01 | 48.82 | **0.271** | 56.74 |
| L=4 Fock, peak 1.2e-3 | 60.80 | 50.10 | 0.194 | 43.07 |
| G3′ | 63.45 | 52.90 | 0.182 | 48.17 |
| G2 | 63.61 | 53.12 | 0.180 | (unidentified) |
| G3 | 65.22 | 54.21 | 0.185 | 49.31 |
| F3.1 | 71.81 | 57.76 | 0.218 | (unidentified) |

- **At the end of the stable phase, Fock leads or is level.** L=4 is 5% ahead of GPT-2, and G2 and G3′ are level with it.
- **GPT-2 then gains more from the decay:** 0.27 nats against 0.18–0.22 for every Fock arm, and it finishes ahead.
- **Two explanations, not separable here:**
  - **Architecture:** Fock models extract less from annealing.
  - **Peak learning rate:** GPT-2's stable phase at 6e-4 had nearly flattened (α 0.93, interval 0.014 nats wide), while the Fock arms at 1.2e-3 were still falling. A different peak changes both the plateau and what the decay can recover.
- **W2** (GPT-2 on WSD at 1.2e-3, tag `_wsd_lr0p0012`, about 2.5 h) separates them. It is now the cheapest run that bears on every GPT-2 comparison, including the L=8 one.

#### W2: the matched GPT-2 at the ladder's peak learning rate — **pre-registered 2026-10-18, before the run**

**Why.** W1 left two readings of GPT-2's larger decay gain (0.271 nats, against 0.18–0.22 for every Fock arm) unseparated: the architecture, or GPT-2's lower peak (6e-4 against the ladder's 1.2e-3). W2 runs GPT-2 on the ladder's schedule exactly, peak included. It also decides the peak learning rate of the matched GPT-2 in the 2B-token runs (Epic 3), the reference for every comparison there.

**The run.** `colab_matched_gpt2_baseline_openwebtext.ipynb` with `LR_SCHEDULE = 'wsd'` and `LR = 1.2e-3`; everything else as W1 (d = 384, L = 8, tied, weight decay 0.1, batch 32 × 512, 32,500 steps, seed 0). Tag `_wsd_lr0p0012`, printed by Cell 0 before training.
- `LR_MIN` stays at 6e-5. That is the ladder's floor at this peak (`WSD_LR_FLOOR = 0.05 × LR`), so peak, floor and shape all equal the ladder's.
- **Verified 2026-10-18:** the notebook's `lr_schedule` at these settings equals the learning rate logged by a ladder run (XRB-F's `training_log.jsonl`) at all 650 logged steps, to a relative 5e-16.

**Predictions** (decay gain = ln PPL at step 21,000, the end of the stable phase, minus ln settled PPL):

| | prediction | called |
| --- | --- | --- |
| W2.0 | the run completes with no loss spike (no eval rising more than 0.2 nats above the previous one) | 85% |
| W2.1 | settled PPL within ±2% of W1's 48.82 (47.84–49.80) | 55% |
| W2.2 | settled PPL at or below W1's 48.82 | 50% |
| W2.3 | decay gain larger than W1's 0.271 (a higher peak leaves more noise for the decay to remove) | 65% |
| W2.4 | decay gain still above every Fock arm's at the same peak (above 0.218, F3.1's) | 60% |

**Decision rules, fixed now.**
- **Settled more than 2% below W1 (under 47.84):** the baseline was under-tuned at 6e-4. W2 becomes the GPT-2 reference; the book's and cards' GPT-2 comparisons are restated against it; the 2B-token GPT-2 runs at 1.2e-3.
- **Within ±2%:** the peak does not matter for GPT-2 at this budget. The published reference stands, and the 2B-token GPT-2 runs at 1.2e-3 so that its schedule is the ladder's in every respect.
- **More than 2% above W1 (over 49.80):** GPT-2 prefers 6e-4. W1 stays the reference, and the 2B-token GPT-2 runs at 6e-4, stated as such.
- **On the decay gain (W2.4):** if it hits, GPT-2 extracts more from annealing than the Fock models at equal peak, which is the architecture reading; if it misses, W1's gap was the peak learning rate.

#### W2 scored: **settled 48.37, within 2% of W1 (−0.9%); all five calls hit. At the ladder's own schedule GPT-2 still gains more from the decay than any Fock arm, so W1's gap is the architecture, not the peak** — **2026-10-18**

Run output `matched_gpt2_LR=1.2e-03_32500steps_training_output.txt` (Colab, A100-40GB, 2.7 h). Cell 0 printed the tag `_wsd_lr0p0012` and its own folders (`checkpoints_wsd_lr0p0012`, `results_wsd_lr0p0012`); batch 32 × 512, 32,500 steps, 0.532B tokens, 33.7M parameters, as W1. Scored from the 65 printed evals (loss@512, 4 decimals), as W1 was; the run's `training_log.jsonl` was not downloaded.

| | prediction | result | |
| --- | --- | --- | --- |
| W2.0 | no eval rises more than 0.2 nats above the previous one | largest rise +0.009 nats (step 15,000) | **HIT** |
| W2.1 | settled within ±2% of W1's 48.82 | **48.37** (mean of 48.65, 48.32, 48.14), −0.92% | **HIT** |
| W2.2 | settled at or below 48.82 | 48.37 | **HIT** |
| W2.3 | decay gain above W1's 0.271 | **0.338** (67.79 at step 21,000 → 48.37) | **HIT** |
| W2.4 | decay gain above every Fock arm's (0.218) | 0.338 | **HIT** |

**Decision rule: within ±2% of W1.** The peak learning rate does not matter for GPT-2 at this budget. The published reference stands by the rule, and **the 2B-token GPT-2 runs at 1.2e-3**, so that its schedule is the ladder's in every respect.

**What it shows.**
- **The decay-gain gap is architectural.** On exactly the ladder's schedule (peak, floor and shape verified identical), GPT-2 gains 0.338 nats from the decay, against 0.18–0.22 for every Fock arm. The higher peak raised GPT-2's gain, as W2.3 predicted, and widened the gap rather than closing it.
- **At the end of the stable phase GPT-2 is behind every comparable Fock arm** (step 21,000, same schedule): GPT-2 67.79, against L = 4 60.80, G3′ 63.45, G2 63.61, G3 65.22; only F3.1 (71.81) is behind it. The higher peak made GPT-2's plateau noisier (W1 at 6e-4: 64.01) and its decay recovers more, so it finishes ahead.
- **For the comparisons** (descriptive, not a change of reference): against the GPT-2 trained on the ladder's identical schedule (48.37), L = 4 Fock (50.10) is +3.6%, G3′ (52.90) +9.4%, PMX (54.63) +12.9%. Against the published cosine baseline (49.81) the same arms read +0.6%, +6.2% and +9.7%. W2 is 2.9% below the cosine baseline, a gap the rule (relative to W1) does not act on, but which any claim of parity should state.
- **Reading for Epic 3.** The Fock arms lead at the end of the constant-rate phase and lose it in the decay. Whether that persists at 2B tokens, where a longer stable phase moves every model closer to its plateau, is one of the cleanest questions the 2B runs can answer.

#### G2B: the matched GPT-2 at 2B tokens, the reference for Epic 3 — **pre-registered 2026-10-18, before the run**

**Why.** Every 2B-token comparison needs a GPT-2 reference trained on the same budget and the same schedule. It is the cheapest 2B run and does not depend on which of our models are selected, so it runs first (author, 2026-10-18).

**The run.** `colab_matched_gpt2_baseline_openwebtext.ipynb` with W2's Cell 0 (`LR_SCHEDULE = 'wsd'`, `LR = 1.2e-3`, floor 6e-5) plus `TOTAL_STEPS = 130_000`: 2.13B tokens at 16,384 per step, seed 0. Tag `_wsd_lr0p0012_st130000` (the run length joined the tag on 2026-10-18; every existing tag unchanged). The WSD fractions are the ladder's, so warmup ends at step 6,500, the constant phase at 84,500, and the decay runs to 130,000. Evals every 500 steps; settled = the mean of the last three, as everywhere. About 10.7 h on the A100.
- **Resume, fixed 2026-10-18.** Before, a resume went back to the best checkpoint and re-seeded the batch sampler, replaying the first steps' batches. The training cell now keeps a rolling `gpt2_baseline_latest.pt` every 2,500 steps with the sampler state, and resumes from the newer of it and the best checkpoint. Tested on a tiny CPU model: a run interrupted at step 26 and resumed from step 20 ends bit-identical to an uninterrupted one (weights, Adam moments, best value, sampler state); the fallback for an older checkpoint without sampler state (resume from the best, sampler fast-forwarded) is bit-identical too; an uninterrupted run is bit-identical to the notebook before the change.
- **Data.** The prepared pool is the 2.0B-token cache every arm uses, so 2.13B tokens means about 1.07 passes on average. Batches are random windows drawn with replacement, so no run is epoch-ordered and repetition at this level is negligible. Every 2B-token arm sees the same pool, so the comparison stays matched. Disclosed here; not enlarged.

**Predictions** (W2, the same model and schedule at 532.5M tokens: settled 48.37; PPL 67.79 at the end of its constant phase; decay gain 0.338):

| | prediction | called |
| --- | --- | --- |
| G2B.0 | completes with no loss spike (no eval more than 0.2 nats above the previous one) | 85% |
| G2B.1 | settled between 41.0 and 45.5 | 65% |
| G2B.2 | settled below W2's 48.37 by at least 5% (at most 45.95) | 80% |
| G2B.3 | decay gain (ln PPL at step 84,500 − ln settled) below W2's 0.338: a longer constant phase leaves the model nearer its plateau | 55% |

**What it is for.** It is the denominator of every Epic-3 ratio, and its constant-phase curve is what the Fock arms' are compared with at equal tokens, which tests W2's reading (Fock ahead until the decay) at four times the budget.

#### G2B scored: **settled 42.69, 11.7% below the same model at 0.5B tokens; three of four calls hit. The decay gains more, not less, after a longer constant phase** — **2026-10-18**

Run output `matched_gpt2_LR=1.2e-03_130000steps_training_output.txt` (Colab, A100-40GB). One uninterrupted session, 38,709 s (10.75 h, 0.30 s per step); Cell 5 printed `Training: steps 1 -> 130,000` with the tag `_wsd_lr0p0012_st130000`, so no resume was needed. 260 evals, no gap and no duplicate. Scored from the printed evals (loss@512, 4 decimals), as W1 and W2 were; the run's `training_log.jsonl` is on Drive.

| | prediction | result | |
| --- | --- | --- | --- |
| G2B.0 | no eval more than 0.2 nats above the previous one | largest rise +0.013 nats (step 15,000) | **HIT** |
| G2B.1 | settled between 41.0 and 45.5 | **42.69** (mean of 42.68, 42.70, 42.68) | **HIT** |
| G2B.2 | at least 5% below W2's 48.37 | −11.7% | **HIT** |
| G2B.3 | decay gain below W2's 0.338 | **0.371** (61.84 at step 84,500 → 42.69) | **MISS** |

**The curve** (evaluation perplexity):

| step | tokens | perplexity | phase |
| ---: | ---: | ---: | --- |
| 6,500 | 0.11B | 92.99 | end of warmup |
| 21,000 | 0.34B | 68.01 | constant |
| 32,500 | 0.53B | 64.97 | constant (W2 had settled at 48.37 by here, after its decay) |
| 50,000 | 0.82B | 63.17 | constant |
| 84,500 | 1.38B | 61.84 | end of the constant phase |
| 100,000 | 1.64B | 57.66 | decay |
| 120,000 | 1.97B | 45.92 | decay |
| 130,000 | 2.13B | 42.68 | end |

**Reading.**
- **Four times the tokens buy 0.125 nats** (48.37 → 42.69) for this 33.7M-parameter model, about 63 tokens per parameter at the end, well past compute-optimal for its size.
- **G2B.3 misses: the decay is worth more after a longer constant phase** (0.371 against 0.338). The constant phase flattens slowly (−4.8% from step 32,500 to 84,500), and nearly all of the final gain comes in the decay. At a fixed peak, the noise the constant learning rate holds the model above its floor does not shrink with time, and the decay removes it.
- **For Epic 3:** the Fock arms' 2B runs are compared against 42.69, and their constant-phase curves against this one at equal tokens. W2's reading (Fock ahead until the decay) is tested there.

### 5.18 RR: what makes a model refinement-ready? — **pre-registered 2026-10-06, before any measurement**

**The clue.** G3 and G3′ differ only in the routing hardening (QK-norm and the 0.3 field clip). Gate 3 at 1.5× refinement:

| model | Gate 3 |
| --- | ---: |
| G3 | +1,342% |
| G3′ | +115% |
| G2 | +1,274% |
| F3.1 | +143% |
| L=4 | +216% |

**Ruled out so far:**
- the π crossing (SR-π.3; G3′'s layer-1 θ of 3.40 still crosses);
- the size of the register increment (SR-π.4b; G3′'s is 7.8×, like its siblings');
- the exchange field itself (G2 has none and fails);
- the repeated register bookkeeping on G2 (SR-π.3b, about 13% of the penalty).

**What separates them.** Freezing the register bank at initialisation costs G3′ +2,280%, against +24% for G2 and +17% for G3. DP1 showed G2's layer-0 destruction gate is switch-like (median 0.994): its register content is mostly reset and rewritten each layer.

**Hypothesis H-RR.**
- **A Fock model is refinement-ready when its register state is *accumulated*,** a slowly built memory. Extra steps then add more of the same, so the update approximates a flow.
- **It fails when its register state is *reset and rewritten* at each layer.** Refinement then inserts rewrite events the model never trained with, so the update is a sequence of maps.
- F3.1 has no registers and is consistent with this.

**Measurements** (evaluation only, CPU, G3′'s best checkpoint):
- **RR-A:** `debug/dp_register_statistics.py` extended to G3′: the destruction gate g per layer, salience, and the share of the initial salience left after the last layer.
- **RR-B:** `debug/sr_pi3b_bookkeeping.py` on G3′, with SR-π.3's tokens: Gate 3 under 6b-7's refinement against refinement with the bookkeeping held to once per trained layer.

**Predictions:**

| | prediction | called |
| --- | --- | --- |
| RR1 | G3′'s layer-0 destruction gate has median below 0.5 (G2 0.994, G3 0.978) | 60% |
| RR2 | G3′'s median share of initial salience left after the last layer is at least 10× G2's (0.0007) | 55% |
| RR3 | holding the bookkeeping changes G3′'s refinement penalty (ln of refined over trained PPL) by under 30% | 55% |

**Decision rule.**
- **RR1 and RR2 hold:** H-RR gains support. The decisive test is a second seed of G3′ and of G2 (GPU), to rule out a lucky basin.
- **RR1 fails** (G3′'s gate is as switch-like as G2's): the register content is not what separates them. The routing's effect works through the token step, and the next candidate is the bounded logits.
- **RR3** separates how much of G3′'s small refinement penalty still comes from bookkeeping.

**RR scored, 2026-10-06.** Outputs: `debug/rr_a_dp_g3prime_output.txt` and `debug/rr_b_bookkeeping_g3prime_output.txt`.

| | measure | G3′ | G2 | G3 | outcome |
| --- | --- | ---: | ---: | ---: | --- |
| RR1 | layer-0 destruction gate, median | **0.478** | 0.994 | 0.978 | **HIT**, narrowly |
| RR2 | share of initial salience left after the last layer, median | **0.0646** | 0.0007 | 0.0027 | **HIT** (about 90× G2's) |
| RR3 | change in refinement penalty when bookkeeping is held | **+121%** (Gate 3 +119% → +468%; corrected 2026-10-16, see the G3′ correction in §5.22) | −13% (+1,446% → +994%) | — | **MISS** |

- **The layer-0 destruction gate is bimodal in G3′:** 10th percentile 0.03, 90th 0.999. It empties about half its registers and keeps the rest, where G2 empties nearly all of them.
- **RR3 missed, in the direction that supports H-RR.**
  - In G2 the per-step register bookkeeping under refinement is harmful: holding it back helps a little.
  - In G3′ it is the reverse. The per-step updates are part of why it refines well, and holding them back raises Gate 3 from +119% to +468% (ln-penalty 0.79 → 1.74; corrected 2026-10-16). Its register state behaves like an accumulated quantity that more steps integrate smoothly.
- **Reading.** RR1 and RR2 hold, and RR3's direction agrees: H-RR gains support. Also, G3′'s DP3 correlation at layer 1 is −0.63, stronger than G2's.
- **The decisive test remains a second seed of G3′ and of G2** (GPU), to rule out a basin one seed happened to find.

### 5.19 The refinement decomposition: two failure modes, two tests — **pre-registered 2026-10-06, before any run**

**Design principle (the author, 2026-10-06).** Refinement readiness is a required property, not an optional one:
- **Conservative models** should be both conservative and refinement-ready;
- **non-conservative (Fock) models** should at least be refinement-ready.

This is property (R) of the book's §37.6 (`subsec:geom-requirements`), stated as a requirement for every arm. Refinement-related investigations are therefore high priority in the queue.

> **Scope caveat: every refinement result is from an undertrained model (the author, 2026-10-11).** Every ladder arm trains on exactly 532.5M tokens (32,500 steps × 16,384). At 76.8M parameters that is about 7 tokens per parameter, a third of the compute-optimal budget of about 20 per parameter (about 1.5B tokens), and far from convergence. Every Gate 1–3 reading, FLOW-C, FLOW-R and the CX/RR/E scores in this section are therefore statements about **models at 0.5B tokens**, not about the architectures. A refinement failure here does not establish that the architecture cannot refine, and a pass does not establish that it stays refinement-ready. The book must carry the same scope.
>
> **The direction is a hypothesis, and the evidence is mixed.** For better refinement with training: refinement needs a smooth V_θ, and an undertrained potential is rough. Against: training stiffens some modes (ω·Δt median F3.1 4.00 at step 8,000 → 4.32 settled; and trained under SR2's exact flow, which charges no phase error, layer 0's settles at twice F3.1's, so training lets stiff modes climb where nothing penalises them), and the split step's phase error grows near θ = π; a longer-trained map may also be more tied to its step size. Not uniform: PM1's fell (3.00 → 2.81).
>
> **RT, the training-budget tests (pre-registered 2026-10-11):**
>
> | id | test | prediction | called at |
> | --- | --- | --- | ---: |
> | RT1 | free, within run: PMX's 6b-7 Gate 3 at N = 3, step 8,000 (0.584 nats) against its best checkpoint | the penalty falls with training | 60% |
> | RT2 | free, where a step-8,000 probe checkpoint exists (PM1, PM1-cap): 6b-7 Gate 3 at N = 3 at step 8,000 against settled | the penalty falls in each | 50% |
> | RT3 | a long run: SR2's Cell 0 at 4× the steps (130,000; 2.13B tokens, about 56 h), Gate 3 as trained and FLOW-R at the end | Gate 3 at N = 3 at or below 0.30 nats (0.405 at 0.5B); the per-layer flow still converges | 45% |
>
> RT1 and RT2 give a trend inside the WSD schedule and cost nothing beyond the scoring cells. They are confounded with the schedule (step 8,000 is mid-stable-phase, the best checkpoint is after the decay). RT3 is the real test. It needs the data pool checked first (about 2B tokens prepared: 130,000 steps is about one epoch, so a "few billion" run needs a larger OpenWebText slice), and its own arm registration with the WSD schedule rescaled to 130,000 steps. SR2 is the base because it is the fastest arm whose per-layer flow is known to converge (FLOW-R); PMX at 4× would take about 106 h.

**The decomposition** (from §5.18 and the gate table). Two different mechanisms produce the refinement failures:

| failure mode | where | mechanism | evidence so far |
| --- | --- | --- | --- |
| **register reset** (H-RR) | Fock arms G2 (+1,274%) and G3 (+1,342%) | register content emptied and rewritten each layer; refinement inserts rewrite events never trained | RR1, RR2 hold; RR3's direction (§5.18) |
| **stiff-mode discretisation** | F3.1 (+143%), no registers | V_θ alone carries the conservative force with stiff modes past π (θ median 4.32); the remaining error is Proposition 44's θ/sin θ phase error | G3′'s field takes stiffness off V_θ (θ 2.29) and refines slightly better (+115%) |

Neither mechanism alone explains every arm, which is why the single-cause accounts failed: the π crossing (SR-π.3), the push size (SR-π.4b) and the bookkeeping (SR-π.3b). The tests below are one per mechanism. In refinement penalties, ln(refined over trained PPL): F3.1 0.89 nats, G3′ 0.77, G2 2.74.

**Test 1: register reset, by replication.** A second seed (`SEED = 1`, tag `s1`) of G3′ and of G2, each with its seed-0 Cell 0 otherwise unchanged. About 15 h each. The seed tag was added and verified 2026-10-06: seed-0 tags are unchanged against HEAD, and `SEED = 1` adds `_s1`.

| | prediction for the seed-1 runs | called |
| --- | --- | --- |
| S1 | G3′-s1 is refinement-ready: Gate 3 at 1.5× of +250% or below (penalty under 1.25 nats) | 55% |
| S2 | G2-s1 is not: Gate 3 of +600% or above (penalty over 1.95 nats) | 65% |
| S3 | the register signature replicates: G3′-s1 frozen-bank cost above +500% and layer-0 destruction median below 0.8; G2-s1 frozen-bank cost below +100% and destruction median above 0.9 | 50% |
| S4 | settled PPL within ±2% of seed 0 for both (G3′ 52.90, G2 53.12): the first seed-variance measurement in the programme | 70% |

- **S1 and S2 hold:** the routing hardening causes the accumulated solution, and H-RR stands as a design rule. Fock models should be built so that their register state accumulates.
- **S1 fails, S2 holds:** G3′'s solution was a basin its seed found. H-RR may still describe it, but the hardening does not guarantee it.
- **S2 fails:** refinement readiness is seed-dependent even without the field, and H-RR must explain why the same configuration lands in both basins.

**Test 2: discretisation, by the exact flow.** SR2 on F3.1's configuration (§5.9): the exact damped-mode flow on the stiff subspace replaces the split step, which removes the θ/sin θ phase term. The code is a new joint substep in `cfc_baoab.py` (Proposition 45, closed form). *Implemented and verified 2026-10-06; see below.*

| | prediction | called |
| --- | --- | --- |
| E1 | SR2-F3.1 cuts F3.1's refinement penalty by at least half (0.89 → 0.45 nats or below) | 55% |
| E2 | SR2-F3.1 settles within +3% of F3.1 (59.5 or below): the exact flow costs little | 60% |

- **E1 holds:** the conservative core's residual failure is integrator error, removable by the exact flow. Together with Test 1, the framework then has a refinement-ready conservative core, and a Fock mechanism that preserves that readiness when its registers accumulate.
- **E1 fails:** F3.1's failure is not discretisation. The next suspect is tuning: the learning rate was never re-swept under Gen 3 or for a conservative-only arm.

**SR2 implementation — 2026-10-06.** `cfc_baoab.py` gains `damped_mode_coefficients` and `lowrank_damped_substep`; the switch is `lowrank_damped_flow` in `MultiXiPARFConfig` (Cell 0 `LOWRANK_DAMPED_FLOW`, tag `sr2`, Cell 5b guard and banner).

- **What changes.** When the switch is on, each A(dt/2) half-step integrates every mode on span(U) as one forced damped oscillator, x″ + γx′ + ω₀²x = a with ω₀² = κ/m. The forcing a is the frozen affine mode force at the start of the half-step. The O-step's action on span(U) is then undone, so friction is not applied there twice. The complement keeps its free drift and its O-step. The B kick, the projection and the velocity encoding are unchanged, so friction on span(U) still totals γ·dt per layer.
- **Coefficients** (float64, every regime). Underdamped cos/sin, overdamped cosh/sinh, critical, and ω₀ = 0. The position response Q = (1 − E₁₁)/ω₀² cancels where ω₀²t² < 10⁻⁴, which is where the trained soft modes (κ ≈ 0) sit. There it is summed from the exact Taylor recurrence of the mode equation, 40 terms, converged for γt < 4. A first version truncated that series at t⁶ and was off by up to 3·10⁻⁵ at γt = 0.4; the RK4 check caught it before any use.
- **Guards.** The model refuses the switch unless `integrator = 'baoab_cfc_lowrank'` and γ is fixed (`fixed_gamma`), because the closed form assumes a constant scalar γ.

**Verification** (`debug/verify_sr2_switch.py` and its output). The HEAD reference is dumped with the committed `cfc_baoab.py`, `model_parf_multixi.py` and ladder notebook swapped in.

| check | result |
| --- | --- |
| OFF vs HEAD, G2, F3.1 and G3′ (eval logits, train logits, every gradient) | bit-identical (0.0) |
| γ = 0: damped substep vs `lowrank_cfc_substep` | 9·10⁻¹⁴ |
| coefficients vs RK4 (7 regimes incl. critical, ω₀ = 0, series branch; 2 horizons; 3 initial conditions) | 2·10⁻¹⁴ |
| group property: one step of 4 = two steps of 2, γ = 0.1 | 3·10⁻¹⁵ |
| tag `…_sr2_L2probe…`, banner, no new parameters | pass |
| γ = 0 on F3.1's weights: switch on vs off | 7.6·10⁻⁵ (float32) |
| causality with the switch on | exact 0 |
| refuses a learned γ or another integrator | pass |

Descriptive, not a test: switching SR2 on post hoc on F3.1's weights, which were trained under the split scheme, raises the loss from 4.230 to 4.356 nats (median |Δlogit| 2.8). The stiff modes rotate by θ ≈ 4.3 per step, so the split and the exact flow differ materially there. E1 and E2 are about a model trained under SR2, not about this transplant.

**Run settings.** F3.1's Cell 0 (`REVERSE_CHANNEL = False`, `VPHI_GRAD_PATH = 'live'`, `XI_GRAD_PATH = 'live'`) plus `LOWRANK_DAMPED_FLOW = True`. Same seed, steps and schedule as F3.1. Then the full 6b set, with 6b-7 (Gates 1–3) deciding E1.

#### SR2 on F3.1 scored: **E1 HIT, E2 HIT — the exact flow halves the refinement penalty at no perplexity cost, but does not make the model a flow** — **2026-10-09**

Run output `L2_SR2_for_noattn_none_32500steps_output.txt` and Cell 6b-7 on the best checkpoint (step 31,000), filed in `results/…_sr2_L2probe…/`. Tag `…cgqk_norc_vplive_xilive_sr2_L2probe…idt4_lr0p0012_noattn`.

| | **SR2** | F3.1 |
| --- | ---: | ---: |
| settled (mean of the last 3 evals) | **58.32** | 57.76 |
| best | 56.49 (step 31,000) | 57.35 |
| step 21,000 (end of the stable phase) | 70.94 | 71.81 |
| gap to F3.1 at matched evals, steps 8,000–18,000 | +0.0% mean, sd 2.0% | — |
| global gradient norm above 1.0, logged steps to 18,000 | 0 of 360 (max 0.94) | 3 of 360 (max 1.51) |
| seconds per step | 1.72 | 1.28 |

- **E2 (settled at or below 59.5, called 60%): HIT,** +1.0% against F3.1. The exact flow costs about a third more compute per step, and no perplexity.
- **Causality:** SCAF CLEAN at 5k, 10k, 15k, 20k, 25k and 30k; the causal and trained-scale leak probes at 10k were exactly 0. The step-32,500 audit printed nan for its PPLs, as G3′'s and PM1's did.
- **The in-flight resonance monitor was empty throughout.** semsimula-diag hooked `lowrank_cfc_substep`, which SR2 replaces with `lowrank_damped_substep`. Fixed in semsimula-diag, with a regression test, on 2026-10-08; it does not affect training.

**Cell 6b-7** (Gate 0 PASS, bit-identical), penalties in nats, ln(PPL at N ÷ PPL at N = 2):

| | SR2 | F3.1 | cut |
| --- | ---: | ---: | ---: |
| Gate 1, velocity reset | +3.9% | +34.3% | |
| Gate 2, N = 3 at the trained Δt | +35% | +92% | |
| Gate 2, N = 8 | +567% | +9,614% | |
| Gate 3, N = 1 | 1.944 (+598%) | 2.565 (+1,200%) | 24% |
| **Gate 3, N = 3** | **0.405 (+50%)** | 0.889 (+143%) | **54%** |
| Gate 3, N = 4 | 0.986 (+168%) | 1.555 (+374%) | 37% |
| Gate 3, N = 6 | 1.725 (+461%) | 2.083 (+703%) | 17% |
| Gate 3, N = 8 | 2.110 (+725%) | 2.275 (+873%) | 7% |

- **E1 (the N = 3 penalty halved, to 0.45 nats or below, called 55%): HIT,** 0.405 nats.
- **The momentum dependence nearly vanishes.** Resetting the incoming velocity costs +3.9% against F3.1's +34%. Running more steps at the trained Δt (Gate 2) degrades far more gently: +567% at N = 8 against +9,614%.
- **Not refinement-invariant.** Gate 3 still grows away from N = 2, so by the cell's own classification the model is still a map at the trained Δt. The cut is largest at the coarse end (54% at N = 3) and shrinks towards fine steps (7% at N = 8), where the two models converge.
- **Reading.** The stiff-mode phase error, which SR2 removes, is what separates F3.1 from SR2 near the trained Δt. It is not what fails at fine steps. As Δt → 0 the split step's stiff-mode error vanishes for F3.1 too, so the shared fine-step penalty, about 2.1–2.3 nats at N = 8, belongs to something both models share. The candidates:
  - **Δt = 4 is far from the flow limit for the non-stiff parts as well.** V_φ, the ξ coupling and the depth code stay explicit and are trained only at Δt = 4.
  - **The trained map uses that coarseness.** Variable-step training (SR4a, N ~ U{2, 3, 4} at fixed T) tests this directly, as the next item in the SR series.
- **Decision rule (pre-registered).** E1 holds, so "the conservative core's residual failure is integrator error, removable by the exact flow" is half right, and is recorded with that scope:
  - **Removable by the exact flow:** the stiff-mode share near the trained Δt, which is most of the N = 3 penalty.
  - **Not removable by it:** the fine-step limit, which is not a flow under either integrator.
  - **Status of (R):** SR2-F3.1 is the most refinement-ready conservative model so far, but it does not satisfy (R).

**Cell 6b-9 (descriptive), 2026-10-09.** Gate 0 PASS. Deflection of the real step from the damped V_θ geodesic, R_h:

| arm | SR2 layer 0 | SR2 layer 1 | F3.1 layer 0 | F3.1 layer 1 |
| --- | ---: | ---: | ---: | ---: |
| geo (V_θ only) | 0.691 | 0.548 | 0.557 | 0.772 |
| geo + LN | 0.064 | **0.003** | 0.117 | 0.679 |
| cons (V_θ + V_φ) | 0.688 | 0.548 | 0.507 | 0.240 |
| step size, \|Δh\| / \|h_in\| | 5.96 | 0.72 | 6.27 | 1.07 |

| attribution, averaged over layers | SR2 | F3.1 | PM1 |
| --- | ---: | ---: | ---: |
| R(geo) | 0.620 | 0.665 | 0.967 |
| LN moves R by | −0.586 | −0.267 | −0.273 |
| V_φ moves R by | **−0.002** | −0.291 | −0.017 |

- **In SR2, V_φ has dropped out of the step.** V_θ plus the LayerNorm projection reproduce it: R 0.064 at layer 0 and 0.003 at layer 1, with the layer-1 velocity almost exactly V_θ's (R_v 0.004). F3.1's V_φ carried 0.29 of the step and PM1's wells took that role. Under the exact flow the model let it go. Whether V_φ still matters for perplexity is not measured here; the cell warns that a small deflection does not mean a small perplexity effect.
- **This narrows the fine-step residual.** V_φ's explicit kick is not a candidate for SR2, because it hardly acts. What is left in the step besides V_θ's exact flow is the LayerNorm projection.
  - The projection is applied once per layer step, after a step that is 6 times the state's norm at layer 0 (5.96 in SR2, 6.27 in F3.1). So the trained layer-0 map is a long move followed by a projection back.
  - Refined to N steps, the projection runs N times, after shorter moves. In the limit that is the flow constrained to the LayerNorm manifold, a different curve from the trained chord-then-project.
  - Both models share this, which fits the convergence of their Gate 3 penalties at fine N.
- **Test, local and cheap** (not yet written): Gate 3 with the projection applied once per trained layer interval instead of after every substep, on F3.1's and SR2's checkpoints. If the fine-step penalty collapses, the projection placement is the residual, and the fix is architectural (where LN sits in the step), not a training schedule.

**Cell 6b-13 (descriptive), 2026-10-09.** The first ω·Δt reading on an SR2 model; the semsimula-diag fix works. ω·Δt per layer, p05 / p50 / p95:

| | layer 0 | layer 1 | all |
| --- | --- | --- | --- |
| **SR2** | 7.82 / **8.93** / 10.15 | 3.92 / 5.46 / 7.02 | **7.55** |
| F3.1 | 3.81 / 4.34 / 4.97 | 3.06 / 4.28 / 5.32 | 4.32 |
| PM1 | 2.22 / 2.68 / 3.08 | 1.50 / 3.32 / 4.66 | 2.81 |

- **Under the exact flow, V_θ stiffened, mostly at layer 0, where ω·Δt doubled.** The split step charges a phase error for stiff modes; the exact flow charges nothing, so training let them climb. A median of 8.9 radians is about 1.4 full rotations of the stiff modes within one layer step.
- **A second candidate for the shared fine-step residual.** SR2 is exact for the quadratic model of V_θ, frozen at the step's starting state. Refined to N steps, the linearisation is recomputed N times. With ω·Δt near 9, the trained map leans on one frozen quadratic per step, and a re-linearised trajectory is a different curve. F3.1 freezes its linearisation in the same way.
- **The test above gains a second switch.** Gate 3 with the linearisation (U, κ, f) held at its value from the start of each trained layer interval, alone and together with the once-per-interval projection. The exact flow of a frozen quadratic composes, so with both switches on only the explicit kick (V_θ's nonlinear residual, V_φ) differs between the refined and the trained step. The two switches then separate projection placement from re-linearisation.
- The cell's printed band "[3.3, 4.2] … MISS" belongs to the 2026-09-28 Gen 2 depth question and is not scored here.

#### The fine-step residual localized: **with the context frozen per layer and one projection per layer, SR2 refines like a flow** — **2026-10-09**

`debug/refinement_ln_linearisation_split.py` (evaluation only, CPU; outputs `refinement_ln_linearisation_split_{sr2,f31}[_xi]_output.txt` in each run's results folder). A diagnostic, not pre-registered.
- **Method.** Gate 3 is re-run (policy hold, the first 4 of Cell 6b-7's 12 batches) with switches that act once per trained layer's share of the interval, instead of at every substep:
  - **LN once:** the projection runs at the end of the share only;
  - **freeze:** the low-rank quadratic of V_θ is taken at the share's first substep;
  - **ξ frozen:** the context ξ, which feeds V_θ's wells and V_φ, is taken at the share's first substep.
- **What stays live.** The force in the kick is always the real one at the current state.
- **Checks (all exact, 0.0).** With the switches off, the loop is bit-identical to 6b-7's `_fom_stack` at N = 2 and 3. At N = 2 every arm is the trained model. Each frozen quantity is computed once per trained layer at N = 8.

Penalty ln(PPL_N ÷ PPL_N=2), nats:

| arm | SR2 N = 3 | SR2 N = 4 | SR2 N = 8 | F3.1 N = 3 | F3.1 N = 4 | F3.1 N = 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| as trained (= 6b-7) | 0.421 | 1.029 | 2.219 | 0.914 | 1.613 | 2.400 |
| LN once | 0.540 | 0.421 | 1.138 | 0.952 | 0.978 | 2.055 |
| freeze | 2.919 | 2.779 | 2.280 | 1.272 | 1.504 | 2.380 |
| LN once + freeze | 2.896 | 1.262 | 1.346 | 2.096 | 1.562 | 2.235 |
| ξ frozen | 0.536 | 1.443 | 2.420 | 0.723 | 1.399 | 2.531 |
| **all three** | **0.331** | **0.269** | **0.143** | 0.242 | 0.922 | 0.673 |

- **SR2 with all three switches refines like a flow.** The penalty shrinks as the steps get finer: +39%, +31% and +15% at N = 3, 4 and 8, against +52%, +180% and +820% as trained. That is 94% of the N = 8 penalty removed. By 6b-7's own classification ("shrinking toward 0"), this is a flow.
- **No single switch does it; the three act together.**
  - **LN once alone** removes about half of SR2's fine-step penalty (59% at N = 4, 49% at N = 8).
  - **Freezing the quadratic alone is catastrophic.** With the projection running every substep, the state leaves the region where the frozen quadratic holds.
  - **Freezing ξ alone is mildly harmful.**
- **F3.1 improves with all three (72% at N = 8) but does not converge** (0.242, 0.922, 0.673). It still carries the split step's stiff-mode phase error, and that error depends on Δt. So the exact flow is needed for convergence, and SR2's E1 gain is a prerequisite, not a side effect.
- **Reading.** The trained SR2 layer step is close to a coarse sample of a well-defined per-layer flow: the damped Langevin flow at fixed context (ξ and V_θ's stiffness taken at layer entry), followed by one LayerNorm projection per layer.
  - This is the framework's piecewise-autonomous reading, in which the potential changes per layer and is fixed within one ([*Addendum: Non-Autonomous Fields*](Addendum_Non_Autonomous_Fields_For_Appendix_A.md)).
  - The standard refinement, which recomputes the context and projects at every substep, refines a different ODE, one the model was never trained to sample.
- **What it does not show.**
  - **Part of the convergence is by construction.** The exact flow of a frozen quadratic composes, so only the explicit kick is tested: V_θ's nonlinear residual, V_φ and the soft modes. The kick's share of the step has not been measured yet.
  - **Small sample.** It was run on 4 batches and three step counts.
  - **Inference only.** Training is unchanged, since at N = 2 every arm is the trained model.
- **Consequences.**
  - **(R) can be satisfied on SR2 by definition of the layer flow, without retraining.** The book's §37.6 property (R) should state which ODE the step samples: context and stiffness frozen per layer, one projection per layer.
  - **Proposed confirmation, to pre-register before running:**
    - all 12 of 6b-7's batches, N up to 16, on SR2 and F3.1;
    - plus the kick's norm as a share of the step;
    - plus the same arms on PM1, whose occupation φ is part of the per-layer context. This is the test of whether PM1's wells refine under the same definition.
  - **SR4a's rationale changes.** Variable-step training is no longer needed to make SR2 refine under this definition. It remains the test for the stricter definition, in which the context is recomputed at every substep.

#### D-sync: the two clustering mechanisms on the trained arms — descriptive, **2026-10-11**

Companion note [`Two_Clustering_Mechanisms_Saturation_and_Synchronization.md`](Two_Clustering_Mechanisms_Saturation_and_Synchronization.md); `debug/dsync_clustering_diagnostics.py` on F3.1, SR2 and PM1, outputs in each arm's results folder. Not pre-registered: it answers two questions the author raised, whether the 2026-09-17 eigensolve failures were a clustering effect and whether synchronization bears on the Fock mechanism.
- **The eigensolve failures were saturation, not clustering.** V_θ's well responsibility g_k = w_k exp(−e_k/2) has a context-gating factor and a proximity factor. At layer 1 the gating is one-hot for 100% of tokens in F3.1 and PM1 (effective wells 1.00 at p05, p50 and p95) and 64% in SR2; the low-rank factor G keeps 4, 8 and 12 of its 32 eigenvalues. The rank tracks the responsibility (Spearman +0.80 to +0.89 at layer 0) and not the distance to the nearest token (within ±0.17). V_θ's wells are per token, so its saturation cannot synchronize tokens.
- **Token coherence is a transient of layer 0.** Mean pairwise cosine of the token states: 0.10–0.17 at the embeddings, 0.73 (F3.1), 0.60 (PM1) and 0.32 (SR2) after layer 0, 0.19 in all three after layer 1. One shared direction, under 1.5% of pairs above 0.9; no clusters, and no pull toward the first token (1.5 rad at every position).
- **PM1's modes:** by overlap alone one broad mode is nearest to every token; the per-mode force share is the right assignment variable, not the overlap argmax.
- **Follow-ups, none scheduled:** the α sweep of PM1's wells with the order parameter (a threshold reading of the PM1-cap null result); the distance histogram against depth on the L = 8 arm; a Kuramoto-type pair potential as an arm; a sequential Lyapunov argument for the per-layer flow (note §7).

#### FLOW-C: confirmation of the per-layer flow — pre-registered **2026-10-09, before the run**

**Why.** If it holds, this is the result that lets the conservative core claim property (R). The 4-batch diagnostic above was exploratory, and it has one open doubt: the exact flow of a frozen quadratic composes by construction, so only the explicit kick is really tested. FLOW-C repeats the test at the full 6b-7 sample with more step counts, measures the kick, and adds the control and PM1.

**Definition under test, the per-layer flow.** Each trained layer's share of the interval T is integrated with N/L substeps, where:
- the context ξ is taken at the share's first substep;
- V_θ's low-rank quadratic (G, Gμ, hence U and κ) is taken at the share's first substep;
- for PM1, so is the occupation φ (the overlap E of the token's own state stays live);
- the explicit kick uses the real force at each substep's midpoint;
- the LayerNorm projection runs once, at the end of the share.

At N = L this is the trained model, bit for bit.

**Measurement** (`debug/refinement_flow_confirmation.py`, evaluation only, CPU):
- **Data:** all 12 of Cell 6b-7's batches (seed 20260920, 12 × 4 × 512 tokens), policy hold, N in {2, 3, 4, 6, 8, 12, 16} at fixed T.
- **Arms:** "as trained" (6b-7's Gate 3) and "per-layer flow".
- **Penalty:** ln(PPL_N ÷ PPL_N=2). The batches are identical across N, so differences between N are paired.
- **Kick share:** at the trained N = 2, each layer step is replayed from its captured inputs with the explicit kick set to exactly zero. The kick share is |h_out(no kick) − h_out| ÷ |h_out − h_in|, median over tokens, per layer.
- **Models:** SR2 (tag `…sr2…`), F3.1 (control: same weights family, split stiff-mode step) and PM1 (clip 0.3), each on its `_best.pt`.
- **Checks before any number is read** (exact, 0.0):
  - with the switches off, the loop equals 6b-7's `_fom_stack`;
  - at N = 2 the per-layer flow equals the trained model;
  - each frozen quantity is computed once per trained layer;
  - the kick-off replay with the clamp at its normal value reproduces the captured output.

**Predictions:**

| | prediction | called |
| --- | --- | --- |
| C0 | SR2 as trained reproduces Colab's 6b-7 within 0.03 nats at N = 3, 4, 8 (0.405, 0.986, 2.110) | 90% |
| C1 | SR2, per-layer flow: penalty at N = 8 at most 0.25 nats | 80% |
| C2 | SR2, per-layer flow, convergence: the penalty does not rise from N = 4 to N = 16 (each successive change at most +0.02 nats), and N = 16 ≤ N = 8 | 65% |
| C3 | SR2 kick share: median at least 0.10 at both layers (the kick shapes at least a tenth of the step, so the convergence is not by construction) | 55% |
| C4 | F3.1, per-layer flow, does not converge: penalty at N = 16 above 0.5 nats, or above its N = 8 value | 65% |
| C5 | PM1, per-layer flow with φ frozen: penalty at N = 8 at most 1.0 nats (as trained, 6b-7: +3,700%, 3.63 nats) | 35% |

**Decision rule.**
- **C1, C2 and C3 hit:** SR2 satisfies (R) under the per-layer flow definition, non-trivially.
  - The inference mode (`substeps_per_layer`, verified bit-identical at 1) is implemented.
  - SR2 gets its HF card with these results, pushed on the author's go-ahead.
  - Book §37.6 states (R) with this definition.
- **C1 and C2 hit, C3 misses:** the convergence holds but the kick is small, so most of it is by construction.
  - The claim is scoped: the trained step is, to within a small kick, the exact flow of a per-layer frozen quadratic.
  - The card and the book say so in those words.
- **C1 or C2 misses:** the diagnostic does not replicate at full sample. No (R) claim, and SR4a proceeds.
- **C4 misses** (F3.1 converges too): the exact flow is not a prerequisite, and the reading above that SR2's E1 gain is necessary is withdrawn.
- **C5:**
  - **Hit:** PM1, and PM1-cap after it, are tested under the same definition before any PM1 card.
  - **Miss:** the wells remain PM1's refinement problem, and PM1-cap goes ahead as pre-registered.

**FLOW-C on SR2, scored 2026-10-09** (`refinement_flow_confirmation_sr2_output.txt` in SR2's results folder; all checks exact). F3.1 and PM1 are pending.

| N (same T) | 2 | 3 | 4 | 6 | 8 | 12 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| as trained, PPL | 57.59 | 86.36 | 154.30 | 323.33 | 474.90 | 683.63 | 815.03 |
| per-layer flow, PPL | 57.59 | 78.91 | 72.99 | 63.71 | 65.41 | 65.55 | 65.54 |
| per-layer flow, penalty (nats) | — | 0.315 | 0.237 | 0.101 | 0.127 | 0.129 | 0.129 |

- **C0: HIT.** As trained: 0.405, 0.986 and 2.110 nats at N = 3, 4, 8, matching Colab's 6b-7.
- **C1: HIT.** 0.127 nats at N = 8.
- **C2: MISS on the letter.** The penalty rises by 0.026 nats from N = 6 to N = 8, against the allowed 0.02, and N = 16 ends 0.002 above N = 8.
- **C3: HIT.** The kick share, median at the trained N = 2, is 0.367 at layer 0 and 0.506 at layer 1 (velocity 0.63 and 0.60). The kick shapes a third to a half of the step, so the convergence is not by construction.
- **Decision rule as written:** C2 missed, so no (R) claim from FLOW-C.
- **What the miss is.** The successive changes are −0.136, +0.026, +0.002 and −0.0002 nats: the refined trajectory converges to a limit, at PPL about 65.5. C2 was written for a penalty falling toward zero. A convergent flow can instead settle at a constant gap above the trained coarse step. That is a mis-specified criterion, but it was specified in advance, so it stands.
- **Author's decision (2026-10-09):** replicate on fresh data under a convergence criterion pre-registered now (FLOW-R, below), rather than reinterpret C2.

**FLOW-C on F3.1 (the control), scored 2026-10-09** (`refinement_flow_confirmation_f31_output.txt` in F3.1's results folder; all checks exact):

| N (same T) | 2 | 3 | 4 | 6 | 8 | 12 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| as trained, penalty (nats) | — | 0.889 | 1.555 | 2.083 | 2.275 | 2.445 | 2.528 |
| per-layer flow, PPL | 56.30 | 71.20 | 140.04 | 95.68 | 104.21 | 94.66 | 93.78 |
| per-layer flow, penalty (nats) | — | 0.235 | 0.911 | 0.530 | 0.616 | 0.520 | 0.510 |

- **C4 (F3.1 does not converge: pen(16) above 0.5 nats or above pen(8)): HIT on the letter,** by 0.010 nats: 0.510.
- **The shape matters more than the margin.**
  - The penalty oscillates (0.235, 0.911, 0.530, 0.616) before its last two points move by only 0.010. F3.1 may be converging slowly, to a limit about 0.5 nats from its trained step, four to five times SR2's 0.10–0.13.
  - The non-monotone start fits the split step's stiff-mode phase error, which depends on Δt.
  - **The reading is therefore scoped:** the exact flow is what makes the per-layer flow converge quickly and close to the trained step. Whether F3.1 converges at all at large N is not settled by N ≤ 16.
- **Kick share** (descriptive): 0.416 at layer 0 and 0.782 at layer 1, larger than SR2's. F3.1's V_φ still acts (6b-9), and it is part of the kick.

**FLOW-C on PM1, scored 2026-10-09** (`refinement_flow_confirmation_pm1_output.txt` in PM1's results folder; run in the author's terminal, all checks exact):

| N (same T) | 2 | 3 | 4 | 6 | 8 | 12 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| as trained, PPL | 53.71 | 133.77 | 9,028.50 | 3,747.54 | 2,041.92 | 1,501.56 | 1,287.96 |
| as trained, penalty (nats) | — | 0.912 | 5.124 | 4.245 | 3.638 | 3.331 | 3.177 |
| per-layer flow (φ frozen too), PPL | 53.71 | 302.40 | 2,798.71 | 5,988.00 | 8,290.74 | 10,224.31 | 11,071.83 |
| per-layer flow, penalty (nats) | — | 1.728 | 3.953 | 4.714 | 5.039 | 5.249 | 5.328 |

- **C5 (PM1, per-layer flow: pen(8) ≤ 1.0 nats, called 35%): MISS,** 5.039.
- **The per-layer flow does not rescue PM1; it makes it worse.** The penalty grows at every N, above the standard refinement's from N = 6 on.
- **The kick is almost all of PM1's step:** 0.616 at layer 0 and 0.942 at layer 1, against SR2's 0.37 and 0.51. Its wells act as explicit kicks, about 4.7 times the conservative force at layer 1 (6b-15). Holding φ fixed and re-evaluating such a force at every substep integrates a strongly attracting field the trained two-step map never sampled.
- **Decision rule:** the wells remain PM1's refinement problem. **PM1-cap goes ahead as pre-registered.** The per-layer flow should be re-tested on PM1-cap, whose bounded wells shrink the kick.

#### FLOW-R: replication of the per-layer flow's convergence on fresh batches — pre-registered **2026-10-09, before the run**

**Why.** FLOW-C showed SR2's per-layer flow settling at a limit, but its convergence criterion was mis-specified and missed. A criterion written after seeing those data must be tested on data not yet seen.

**Measurement.** `debug/refinement_flow_confirmation.py … sr2 seed=20261009`:
- **Data:** 12 × 4 × 512 validation tokens drawn with seed **20261009**, not 6b-7's 20260920; the same `get_batch`.
- **Model and arms:** SR2's `_best.pt`. Arms: per-layer flow at N in {2, 3, 4, 6, 8, 12, 16}; as trained at N = 2 and 8.
- **Kick share:** measured again on the first new batch.
- **Checks:** the same exact checks as FLOW-C.
- pen(N) = ln(PPL_N ÷ PPL_N=2).

**Predictions:**

| | prediction | called |
| --- | --- | --- |
| R1 | convergence: \|pen(16) − pen(12)\| ≤ 0.01 and \|pen(12) − pen(8)\| ≤ 0.02 nats | 80% |
| R2 | the limit is close to the trained step: pen(16) ≤ 0.25 nats | 80% |
| R3 | the gap replicates: pen(16) within 0.05 nats of FLOW-C's 0.129 | 70% |
| R4 | as trained still diverges on the new batches: pen(8) ≥ 1.5 nats | 90% |
| R5 | the kick share stays non-trivial: median at least 0.10 at both layers | 90% |

**Decision rule.**
- **R1, R2 and R5 hit:** the claim is that **SR2's layer step is a coarse sample of a convergent per-layer flow, with a discretisation gap of pen(16) nats.** The per-layer flow is: context, stiffness and occupations taken at layer entry, the damped Langevin dynamics exact on the stiff subspace, one LayerNorm projection per layer. Then, in order:
  1. the inference mode `substeps_per_layer` (bit-identical at 1);
  2. SR2's HF card with FLOW-C and FLOW-R;
  3. book §37.6's (R) stated with this definition.
- **R1 misses:** no convergence claim on the per-layer flow. SR4a proceeds.
- **R2 misses with R1 hit:** the flow exists but is far from the trained step. The claim is restricted to existence, not to the trained model sampling it.
- **R3 and R4** are descriptive checks of stability and do not gate the claim.

#### FLOW-R scored: **all five HIT — SR2's layer step is a coarse sample of a convergent per-layer flow** — **2026-10-09**

`refinement_flow_confirmation_sr2_seed20261009_output.txt` in SR2's results folder; every check exact (0.0).

| N (same T), batch seed 20261009 | 2 | 3 | 4 | 6 | 8 | 12 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| as trained, PPL | 55.04 | | | | 408.23 | | |
| per-layer flow, PPL | 55.04 | 73.30 | 67.64 | 59.83 | 61.00 | 61.08 | 61.09 |
| per-layer flow, penalty (nats) | — | 0.286 | 0.206 | 0.084 | 0.103 | 0.104 | 0.104 |

| | prediction | called | result |
| --- | --- | --- | --- |
| R1 | \|pen(16) − pen(12)\| ≤ 0.01 and \|pen(12) − pen(8)\| ≤ 0.02 | 80% | **HIT:** 0.0001 and 0.0012 |
| R2 | pen(16) ≤ 0.25 | 80% | **HIT:** 0.104 |
| R3 | pen(16) within 0.05 of FLOW-C's 0.129 | 70% | **HIT:** 0.025 away |
| R4 | as trained, pen(8) ≥ 1.5 | 90% | **HIT:** 2.004 |
| R5 | kick share median ≥ 0.10 at both layers | 90% | **HIT:** 0.369 and 0.505 (velocity 0.635, 0.598) |

- **Decision rule: R1, R2 and R5 hit, so the claim is licensed.** SR2's layer step is a coarse sample of a convergent per-layer flow, with a discretisation gap of **0.104 nats** (11%) on fresh data, 0.129 on 6b-7's batches.
- **The per-layer flow:**
  - the context ξ and V_θ's low-rank stiffness are taken at layer entry;
  - the damped Langevin dynamics run exactly on the stiff subspace;
  - the remaining force is the real one, applied as kicks;
  - LayerNorm projects once per layer.
- **What the claim rests on.** The refined trajectory stops changing by N = 8–16, to 10⁻³ nats. The limit sits close to the trained step. The explicit kick, which carries 37–51% of each step, is integrated by the refinement rather than frozen, so the convergence is not by construction.
- **What it does not claim.**
  - The standard refinement, which recomputes the context and projects at every substep, still diverges: R4, 2.0 nats at N = 8.
  - The result is shown for SR2 only. FLOW-C's control: F3.1 does not settle within N ≤ 16 (C4 HIT, by 0.010 nats), and PM1 diverges under the per-layer flow (C5 MISS).
  - It is a property of the trained weights at inference. Training is unchanged.
- **Next, per the decision rule:**
  1. the inference mode `substeps_per_layer`, bit-identical at 1;
  2. SR2's HF card with FLOW-C and FLOW-R;
  3. book §37.6's (R) stated with this definition.

**Inference mode implemented and verified, 2026-10-09.** `substeps_per_layer: int = 1` in the model config.
- **Code.** `FockMultiXiPARFLM._stack_forward` runs each layer as k substeps of dt/k, with a per-layer context on the model (`_flow_ctx`). `_layer_step_langevin` takes ξ and the low-rank quadratic from it after the layer's first substep, and projects only at the last. `poisson_mode_force` takes φ from it. Outside the mode the context is `None` and every line takes its original path.
- **Scope.** Inference only: it refuses training mode and k < 1, and it requires `baoab_cfc_lowrank` with no reverse channel. There is no layer checkpointing inside the mode.
- **Verification** (`debug/verify_substeps_switch.py` and its output):
  - **k = 1:** bit-identical to HEAD (`git archive`) on SR2's and PM1's configurations and trained weights. Eval logits, train-mode loss and all 73 and 77 parameter gradients match.
  - **k = 4 and 8 on SR2:** logits identical to the FLOW-C/FLOW-R harness at N = 8 and 16 (max |Δ| 0.0), and the loss on FLOW-R's first batch identical (4.134370).
  - **k = 4 on PM1:** identical to the harness with φ frozen.
  - **Causality at k = 4:** exact.
  - **State and guards:** k = 1 set explicitly equals the default, the context is cleared after every forward, and both guards fire.
- **For users.** `model.cfg.substeps_per_layer = k` with `model.eval()` turns depth into an inference-time knob on SR2. Perplexity converges as k grows (FLOW-R), to within about 0.10–0.13 nats of the trained k = 1.


**Consequences for the queue.**
- **PM1-cap's base: the author's call before launch, as pre-registered.** SR2 does not address PM1's failure, which is in the wells (§5.15 localization). Running PM1-cap on the base without SR2 isolates the cap against PM1. Running it on SR2 (`…pmcap0p3…sr2`) would test the candidate combined model but confound the two changes. Recommendation: as pre-registered, without SR2; combine the two afterwards if CAP3 holds.
- **SR4a moves up.** It targets the residual both integrators share, unless the LayerNorm-placement test above explains that residual first. That test runs on the CPU in minutes and should come before a GPU run.

**Book.** If both tests hold, the decomposition becomes a subsection of book §37.6, in the next edition after the seed runs. With E1 scored, the subsection gains its scope statement: the exact flow removes the stiff-mode share of the refinement failure, not the fine-step limit.

### 5.20 CD: one coupling takes over — what does each mechanism cost when it is removed? — **pre-registered 2026-10-11, before any measurement**

**Why.** The 6b-9 attributions show V_φ carrying the non-V_θ part of the step in exactly one arm, F3.1 (−0.291), and moving it by less than 0.02 in every arm with a competitor: the Fock reverse channel (G2 −0.0002, the L=4 Fock arm −0.0035), the PM1 wells (−0.017, the wells −0.258), and SR2's exact V_θ flow (−0.002). That supports a partial order with V_φ at the bottom, but 6b-9 measures displacement, not importance, and the shares of different mechanisms are in different units (6b-9 attribution, η). This series gives every mechanism one unit: the perplexity cost of removing it at inference from the trained weights.

**Method** (`debug/coupling_dominance_ablation.py OUT`, CPU, no training). Each arm is built through the ladder notebook's Cells 0–5b on its best checkpoint, eval mode, and scored on Cell 6b-7's 12 × 4 × 512 validation batches (seed 20260920). Switches, all reversible on the same weights:

- **V_φ off:** `_pair_potential` returns the constant zero of the `pair_potential='none'` path, so f_φ = 0; the wells' force and the Fock increment are unaffected.
- **Fock off** (G2): `reverse_channel_scale` set to 0, so tanh(0) = 0 and the increment vanishes.
- **Wells off** (PM1): `pm_depth` set to 0.

The cost of a removal is c = ln(PPL_removed / PPL_full) in nats, with a paired standard error over the 12 batches. Checks first, all exact: the identity patch of `_pair_potential` and every switch restored leave the logits bit-identical; the patched potential is called on every layer step.

**Arms:** F3.1 (V_φ off); SR2 (V_φ off); G2 (V_φ off, Fock off, both); PM1 (V_φ off, wells off, both).

**Readings, fixed now.** V_φ is *dormant* in an arm if c(V_φ) < 0.01 nats (1% PPL), and *load-bearing* if c(V_φ) ≥ 0.05.

| id | prediction | called at |
| --- | --- | ---: |
| CD1 | F3.1: V_φ load-bearing, c ≥ 0.05 | 85% |
| CD2 | G2: V_φ dormant, c < 0.01 | 70% |
| CD3 | SR2: V_φ dormant, c < 0.01 | 65% |
| CD4 | PM1: V_φ dormant, c < 0.01 | 55% |
| CD5 | G2: c(Fock off) > c(V_φ off in F3.1): the displacing coupling carries more than the one it displaced | 65% |
| CD6 | PM1: c(wells off) > c(V_φ off in F3.1) | 55% |

**What the readings license.** CD2–CD4 together: "dormant" in the book means a removal cost below 1% PPL, not only a small displacement. CD5/CD6 rank the displacing mechanisms above V_φ in the common unit. A zero-shot removal measures the trained model's reliance on a term, not what a model trained without it would reach; it cannot test path-dependence, which is the CB series' job (CB1, CB2b, §5.10). Fock against the wells stays unranked: no arm carries both.

#### CD scored: **3 of 6; "dormant" fails, the Fock channel dominates, the wells coexist with V_φ** — **2026-10-11**

`debug/coupling_dominance_ablation.py` (output `coupling_dominance_ablation_output.txt`, `.json`). Every check exact: identity patch bit-identical and called on all 4 layer steps; every switch restored bit-identical. Cell 6b-7's batches reproduce the settled order (G2 51.99, PM1 53.71, F3.1 56.30, SR2 57.59).

| arm | removal | c (nats) | PPL | | 6b-9 share of the same term |
| --- | --- | ---: | ---: | --- | ---: |
| F3.1 | V_φ | **+0.282** ± 0.013 | 56.30 → 74.64 | +33% | −0.291 |
| SR2 | V_φ | +0.0135 ± 0.0010 | 57.59 → 58.38 | +1.4% | −0.002 |
| G2 | V_φ | +0.037 ± 0.008 | 51.99 → 53.92 | +3.7% | −0.0002 |
| G2 | Fock channel | **+1.486** ± 0.076 | → 229.67 | +342% | (η 1.6–3.1) |
| G2 | both | +1.524 ± 0.081 | → 238.75 | +359% | |
| PM1 | V_φ | **+0.324** ± 0.008 | 53.71 → 74.28 | +38% | −0.017 |
| PM1 | wells | **+0.775** ± 0.052 | → 116.53 | +117% | −0.258 |
| PM1 | both | +0.956 ± 0.061 | → 139.77 | +160% | |

- **CD1 HIT.** V_φ is load-bearing in F3.1 (0.282, every batch above 0.22).
- **CD2 MISS.** V_φ in G2 costs 0.037, above 0.01 on every batch. Not dormant: **subordinate**, 8× below its F3.1 cost and 40× below the Fock channel's.
- **CD3 MISS, narrowly.** SR2's V_φ costs 0.0135: marginal, 21× below F3.1, but over the line fixed in advance.
- **CD4 MISS, and it overturns a reading.** PM1's V_φ costs 0.324, *more* than in F3.1. "In PM1 the wells have taken over V_φ's role" (the PM1 6b-9 block below, and `Poisson_Mode_Registers_PM1.md` §6) is **withdrawn**: the wells carry the step's displacement, but the trained model relies on V_φ at least as much as F3.1 does.
- **CD5 HIT.** The Fock channel's cost (1.486) is 5× V_φ's in F3.1.
- **CD6 HIT.** The wells' cost (0.775) is 2.7× V_φ's in F3.1.

**What it changes.**

1. **6b-9 displacement is not reliance.** Shares of 0.0002 and 0.017 hid costs of 3.7% and 38%. Displacement is a single-step, layer-averaged replay; the removal propagates through both layers and the readout. No 6b-9 share is to be quoted as importance again without a removal cost beside it.
2. **The Fock arm fits the exclusivity conjecture; the wells arm does not.** In G2 one coupling carries the model (1.49 against 0.037) and the two costs add (1.524 against 1.523): the remainder is a separate job, not a back-up. In PM1 two couplings are both load-bearing, and they overlap by 13% (0.956 against 1.099). The conjecture holds for the non-conservative register path, not for the conservative wells.
3. **Partial order in the removal unit:** Fock channel (1.49) > wells (0.77) > V_φ uncontested (0.28–0.32) > V_φ beside Fock (0.037) > V_φ under SR2 (0.014). Fock against the wells is still not measured in one model.
4. **Reliance overstates value.** Trained-without gains are much smaller: G2 against F3.1 0.084 nats (the Fock channel), PM1 against F3.1 0.046 (the wells), against zero-shot costs of 1.49 and 0.77. A removal cost says what this network routes through a term, not what a model trained without it would lose (the same overstatement as the Gen 3 field, +56% removed against −2.1% trained without).

**Extension, PMX (2026-10-11, descriptive, after CD was scored).** `coupling_dominance_ablation.py … PMX` (same batches; full model 53.24): removing V_φ costs **0.0145** ± 0.0012 nats (+1.5%), the wells **1.063** ± 0.056 (+190%), both 1.088 (additive: 1.078). With the wells integrated exactly, V_φ is as marginal as in SR2 (0.0135), and the wells carry more than in PM1 (0.775). **So the PM1 reading "the wells and V_φ are both load-bearing" belongs to the explicit integrator:** on the exact-flow base V_φ is marginal with or without the wells, and the conservative route is carried by the wells. What makes V_φ marginal is the exact flow (SR2 alone already shows it), not the wells. The VF factorial (§5.21) is on F3.1's explicit base and still asks its question there; a VF arm on the SR2 base would ask it for the exact flow.

#### CD extended to the exchange-field arms: **a substitution order among the token couplings, register read ≻ exchange field ≻ V_φ** — **2026-10-18** (descriptive; not pre-registered)

**Why.** The author's question: when the reverse channel is absent and the exchange field takes over, does V_φ sit in the background? And is there an order of strength among the couplings? The CD arms so far had V_φ alone (F3.1, SR2), beside the read (G2) or beside the wells (PM1, PMX). The three arms with the exchange field (XRB-F, G3, G3′) were missing. Same script, switches and batches (Cell 6b-7's 12 × 4 × 512), plus one switch: the exchange field off by its gate (`relax_lambda = 0`). Every check exact (20 of 20: identity patch bit-identical, every switch moves the logits and restores bit for bit). Output `debug/coupling_dominance_ablation_XRBF_G3p_G3_output.txt`, `.json`.

**Removal costs, nats** (standard errors 0.002–0.003 for V_φ, 0.02–0.10 for the field and the read):

| arm | couplings | V_φ | exchange field | register read | field + read | all three |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| F3.1 | V_φ | **0.282** | — | — | — | — |
| XRB-F | V_φ, field | 0.041 | **1.372** | — | — | — |
| G2 | V_φ, read | 0.037 | — | **1.486** | — | — |
| G3′ | V_φ, field, read | 0.031 | 0.569 | **1.351** | 1.806 | 1.815 |
| G3 | V_φ, field, read (unhardened) | 0.011 | 0.613 | **1.550** | 1.735 | 1.750 |
| SR2 | V_φ (exact stiff-mode flow) | 0.014 | — | — | — | — |
| PM1 | V_φ, wells | 0.324 | wells 0.775 | — | — | — |

**Reading.**
1. **V_φ steps back beside either token route, by about the same factor.** Its removal cost falls from 0.282 alone (F3.1) to 0.041 beside the field and 0.037 beside the read: about 7× either way. It does not vanish: 0.01–0.04 nats in every arm with another coupling, well resolved. The learned per-layer V_φ scale (`raw_v_phi_scale`, read from the checkpoints) falls much more beside the read (output layer 2.40 → 0.0016) than beside the field (→ 0.41), so the scale is not a measure of reliance: the model compensates through V_φ's own parameters.
2. **The read crowds out the field; the field barely touches the read.** The field's cost falls from 1.372 alone to 0.57–0.61 beside the read (about 2.3×); the read's moves from 1.486 alone to 1.35 (G3′) or 1.55 (G3).
3. **So the order is register read ≻ exchange field ≻ V_φ:** each route is crowded out by the ones above it and barely affected by the ones below it. In value (§5.25, trained with and without), the same order: the read alone −8.0%, the field alone −6.0% against F3.1.
4. **Not quite additive.** Removing field and read together costs less than the sum: 6% less in G3′ (1.806 against 1.920), 20% less in G3 (1.735 against 2.163). Their work overlaps, more so with unhardened routing. V_φ adds 0.01–0.05 nats on top of any removal.
5. **The wells are different.** Beside the wells (PM1), V_φ keeps all its reliance (0.324), unlike beside the field or the read. The wells pull tokens toward shared learned centres rather than toward each other, which fits a separate niche. Wells against the read and the field stay unranked, since no arm carries both; the candidate run G2 + wells is recorded in the checklist.
6. **The integrator matters too.** In SR2 V_φ is nearly idle with no competing coupling at all (0.014): part of what V_φ does in F3.1 compensates the split integrator.

**Scope.** Removal at inference measures reliance on a term in a trained model, not what a model trained without it reaches (§5.20). One seed per arm; L = 2; 532.5M tokens.

### 5.21 VF: is V_φ worth more beside the wells? The trained-without factorial — **pre-registered 2026-10-11, before any run**

**Why.** §5.20 measured reliance: removed at inference, V_φ costs F3.1 0.282 nats, G2 0.037 and PM1 0.324. Reliance overstates value (the Fock channel's removal cost 1.49 against its trained-without gain 0.084), so it cannot say whether the wells *enhance* the pair potential or whether the Fock channel *substitutes* for it in training. Models trained without V_φ can. This is the live-gradient successor to runs 10 and 11 (§3.2), which were pre-registered on Gen 2 numbers and never run.

**Arms.** F3.1's Cell 0 (live V_φ and ξ, the leapfrog-base `baoab_cfc_lowrank`, not SR2), so that the three measured arms are the matching cells. Seed 0, WSD on 32,500 steps, about 14 h each on one A100, run one at a time.

| | with V_φ (measured) | without V_φ (`PAIR_POTENTIAL = 'none'`, tag `nophi`) |
| --- | ---: | --- |
| no other coupling | F3.1 57.76 | **VF0** |
| wells (`POISSON_MODES = 64`) | PM1 55.17 | **VF1** |
| Fock (`REVERSE_CHANNEL = True`) | G2 53.12 | **VF2** |

**Readout.** V_φ's value in each setting: $v = \ln(\mathrm{PPL}\_{\text{without}} / \mathrm{PPL}\_{\text{with}})$, settled PPL (mean of the last three evals), so $v_0 = \ln(\mathrm{VF0}/57.76)$, $v_{\text{wells}} = \ln(\mathrm{VF1}/55.17)$, $v_{\text{Fock}} = \ln(\mathrm{VF2}/53.12)$.

| id | prediction | called at |
| --- | --- | ---: |
| VF-a | V_φ is worth something uncontested: $v_0 \ge 0.03$ | 70% |
| VF-b | the Fock channel substitutes in training: $v_{\text{Fock}} \le 0.01$ | 55% |
| VF-c | the wells do not substitute: $v_{\text{wells}} \ge v_0 - 0.01$ | 55% |
| VF-d | the wells enhance V_φ: $v_{\text{wells}} \gt v_0 + 0.02$ | 30% |

**What the readings license.** VF-b and VF-c together: "the non-conservative route displaces the pair potential and the conservative route does not" holds in training, not only at inference (§5.20). VF-d licenses "the wells enhance the pair potential", which nothing measured so far does. VF1 also answers a question of its own: how far the wells carry the conservative route with no direct token-to-token term.

**Before launch.** `PAIR_POTENTIAL = 'none'` was verified on 2026-10-01 for the multi-ξ and Fock builds (`debug/verify_pair_potential_none.py`), but never together with `POISSON_MODES > 0`. VF1 needs a switch check first: tag, Cell 5b banner, parameter count, forward and backward, the wells' force still the gradient at fixed occupation, causality.

**Order and priority.** VF0 and VF1 first (the conservative-route claim), VF2 after. Behind PMX's scoring; the book's conservative-route paragraph does not wait on them, but it may say only "the wells work beside V_φ" until VF1 is scored.

### 5.22 SY: is spontaneous synchronization at the root of semantic inference in these models? — **pre-registered 2026-10-11, before any measurement**

**Design:** [`Spontaneous_Synchronization_and_Semantic_Binding_Test_Programme.md`](Spontaneous_Synchronization_and_Semantic_Binding_Test_Programme.md), which defines every statistic used here (§3 there) and the tests SY0–SY6. This section fixes the primary statistics, the thresholds and the calls. Book Q10 points to it.

**Synchronization** is alignment produced by the inter-token couplings, measured against a replay of the same layer step with the coupling switched off (the 6b-9 instrument). Statistics are on centred states (the LayerNorm common direction removed). Groupings: coreference chains (if a coreference model is available offline; otherwise dropped), dependency head–dependent pairs, shared subtrees, named-entity spans, part of speech. Contrasts are distance-matched.

**General rules, fixed now.**
- **Unit and intervals:** the sequence; cluster bootstrap over 200 sequences of 512 tokens, 1,000 resamples, 95% intervals.
- **Effect floor:** a contrast counts only if its interval excludes 0 and it is at least 0.02 (cosine units, or NMI).
- **Replication:** a level holds only if it holds in PMX and in at least one other arm with the same coupling route (V_φ: F3.1 or PM1; wells: PM1; Fock: the L = 4 Fock arm).
- **Primary statistics only** decide levels; everything else is descriptive.
- **Prerequisite:** SY0 passes on all four toy checks (the design note's §4) before any model is measured. A failure is fixed and re-run, never waived.
- **BLiMP sets** enter only where the full model scores at least 60%.

**Primary statistics and level criteria.**

| level | primary statistic | holds if |
| --- | --- | --- |
| 0. exists | coupling-attributable gain in global coherence or co-movement, Δ_c R or Δ_c C, at any layer, for any coupling | at least one passes the floor, replicated |
| 1. structured | (a) V_φ: attraction contrast for coreferent or head–dependent pairs, with selection enrichment; (b) wells: NMI gain of dominant-mode assignment over the input-state assignment, or co-assignment contrast; (c) coupling-attributable binding contrast Δ_c B | (a) contrast passes the floor and enrichment ≥ 1.2; or (b) passes the floor; and in either case (c) passes the floor for the same grouping; all beat the shuffled-label control |
| 2. used | targeted removal of related pairs against matched random removal: loss change at binding-dependent positions | ratio targeted / random ≥ 1.5 with interval excluding 1; that ratio at least 1.2 × the same ratio at other positions; and BLiMP accuracy falls at least 2 points more under targeted than random removal on at least one admitted set |
| 3. root | SY4 λ sweep and SY5 Gate 2 axis | level 2 holds; performance peaks at λ in [0.75, 1.25] and drops by at least 0.05 nats at λ = 1.5 while R rises by at least 0.02; and along Gate 2, from N = 2 to 3, R rises by at least 0.05 or the effective rank of the final states falls by at least 20% |

**Calls.**

| id | prediction | called at |
| --- | --- | ---: |
| SY-L0 | level 0 holds | 75% |
| SY-L1 | level 1 holds | 45% |
| SY-L2 | level 2 holds | 30% |
| SY-L3 | level 3 holds | 10% |
| SY-pair | V_φ attracts coreferent or head–dependent pairs more than distance-matched controls (level 1 (a) alone) | 50% |
| SY-wells | the wells' route passes level 1 (b) and V_φ's does not | 25% |
| SY-G2 | Gate 2's blow-up comes with over-synchronization (the SY5 criterion alone) | 35% |
| SY-GPT2 | the matched GPT-2 shows a binding contrast at least as large as the best Lagrangian arm's (descriptive comparison) | 60% |

**Reading.** The verdict is the highest level that holds: refuted (0 fails), semantically blind (1 fails), epiphenomenal (2 fails), load-bearing for binding (3 fails), or root (3 holds). Every verdict is scoped to these models: L = 2 (L = 4 where read), 532.5M tokens, a third of compute-optimal (§5.19 scope caveat).

**Before measurement:** the annotation packages (spaCy pipeline, a coreference model) and BLiMP need the author's go-ahead to install and download. If no coreference model is available, coreference is dropped and the criteria use the remaining groupings; this is fixed now and is not a post-hoc choice.

#### SY0 scored: the instruments as specified failed; fixed, re-run, passed on run 3 — **2026-10-13, before any model state was measured**

The author gave the go-ahead on 2026-10-13. `sync/sy_instruments.py` (outputs `sy0_v1_output.txt`, `sy0_v2_output.txt`, `sy0_output.txt` for run 3). Forty independent draws per condition, 95% cluster-bootstrap intervals, floor 0.02. As this section requires, every failure was fixed and re-run, and none was waived.

| run | what it tested | what failed, and why |
| --- | --- | --- |
| 1 | the measures as the design note specifies them | **raw (uncentred) measures** read a shared drift as synchronization (Δ_c R = +0.75). **Centred coherence R** passed the letter of check 1 but is unusable: it peaks at the Kuramoto threshold (+0.08 at 1.25 K_c) and falls as synchronization completes (+0.03 at 3 K_c, order parameter 0.98). For centred vectors, Σ over t ≠ s of y_t·y_s = −Σ‖y_t‖², so their mean pairwise cosine is pinned near −1/(N−1) as the spread vanishes. **Coupling attribution against the switch-off replay** counts a coupling's common push, which aligns states indirectly through the per-token nonlinearity and the closing LayerNorm (pairwise contraction Γ = +0.56 and +1.82 under a pure drift). |
| 2 | a second reference, the **common-mode replay**: the coupling's force on every token replaced by its cross-token mean ("spec" = on minus common; synchronization is what the coupling does *differently* to different tokens); monotonicity added to check 1; checks 3 and 4 also at d = 384, the models' width | Γ_spec passed everything, and every spec attribution is exactly 0 under drift with or without LayerNorm, at both widths. The binding measures failed only check 3, where a pull toward each token's *running* mean lowered B by 0.033 at both widths. A diagnostic showed this is a real effect of the coupling, not an artefact: a pull toward one *shared* target gives −0.0000, and per-token targets dilute tight clusters with or without the per-token map. A coupling with a different target per token is token-specific, so it was the wrong null. |
| 3 | check 3 with the shared-target pull, decided two-sided; the running-mean pull kept as 3′ (descriptive) | **all admitted measures pass** |

**Instruments admitted (run 3):** Γ_spec, the global pairwise contraction of the pre-LayerNorm step output against the common-mode reference (tracks the Kuramoto order parameter monotonically: +0.58, +1.36, +1.83 at 1.25, 2, 3 K_c); **B_spec**, the distance-matched binding contrast on the centred states the next layer reads, on minus common (coupling-planted groups +1.22 at depth 8, shuffled control +0.0009); **Gb_spec**, its contraction form; **C_spec**, the co-movement contrast on the centred pre-LayerNorm displacements. **Rejected:** every raw measure, and centred coherence R in both forms.

**Amendment to the primary statistics, made now, before any model is measured.**
- **Level 0:** "Δ_c R or Δ_c C" becomes **Γ_spec** (global contraction attributable to the coupling's token-specific action). Global co-movement is dropped: it is a mean pairwise cosine of centred displacements and inherits the defect that removed R.
- **Level 1 (c):** "Δ_c B" becomes **B_spec**. The shuffled-label control applies to B_spec.
- Levels 1 (a), (b), 2 and 3, the thresholds, the replication rule and the calls are unchanged.

**On the models: the displacement form of the common reference.** PMX integrates its wells inside the exact low-rank flow, so for that coupling no force can be replaced by its mean. The model script (`sync/sy_measure.py`) therefore also builds a displacement-level reference, off + the cross-token mean of (on − off) before the closing LayerNorm, and adopts it only if it agrees with the force-level reference on V_φ (a plain kick on every arm) to within 0.005, with the same sign, on Γ_spec and every grouping's B_spec. On the four-sequence PMX smoke run (placeholder labels, mechanics only) the two agree to the fourth decimal (Γ_spec at layer 1: +0.0218 against +0.0212). On a single step the displacement-form Γ_spec equals Γ_tot identically, since a shared translation leaves pairwise distances unchanged; the common reference matters for the binding measures, which are read after the LayerNorm.

**Annotation.** spaCy `en_core_web_trf` 3.7.3 (parts of speech, dependencies, noun chunks, entities) and fastcoref `FCoref` (coreference chains) were installed in a separate virtualenv (system Python 3.9, numpy below 2), because their dependencies conflict with the model stack's torch 2.2.2. The annotation is cached in `sync/sy_cache/` (git-ignored). BLiMP: all 67 paradigms downloaded from the authors' repository. The sets admitted to level 2, fixed now: the *anaphor agreement* and *subject–verb agreement* phenomena, and *binding* (principle A), subject to the ≥ 60% rule.

**Evaluation data:** 200 × 512 validation tokens, seed 20261011 (`sy_cache/sy_tokens.npz`).

**One unit fixed before the scored route readouts (2026-10-13).** Level 1 (a)'s attraction contrast is a force, and the 0.02 floor is in cosine or NMI units. It is therefore scored **relative to the mean absolute radial force over all selected pairs** (dimensionless: 0.02 means 2% of a typical pair's force). This was fixed after a 2-sequence mechanics smoke test (relative values not computed) and before any scored run. The route readout is `sync/sy_routes.py`: its rebuilt V_φ branch reproduces the model's pair potential and force exactly, and the potential's own gradient is the whole V_φ force (share 1.000; the straight-through selection term contributes nothing at inference).

**"Beats the shuffled-label control"**, which level 1 requires, is read as: the real statistic passes the floor and the shuffled one is null by the same rule.

#### SY levels 0 and 1 scored: **level 0 HOLDS, level 1 FAILS: the couplings synchronize tokens, but not along linguistic structure by the pre-registered criteria** — **2026-10-14**

`sync/sy_measure.py` and `sync/sy_routes.py` on the five arms (200 × 512 tokens, both layers), scored by `sync/sy_score.py` (`sy_score_output.txt`). Every run passed its checks: the replays reproduce the trained steps bit for bit, every patch restores, and the rebuilt V_φ branch is exact. The displacement-form common reference was adopted on PMX, PM1, G2 and SR2. On **F3.1 it disagreed** with the force form (Γ_spec at layer 1: +0.089 against +0.064), so F3.1 is scored at the force level, as fixed in advance.

**Level 0: global contraction Γ_spec**

| arm | V_φ, layer 0 | V_φ, layer 1 | wells / Fock, layer 0 | wells / Fock, layer 1 |
| --- | ---: | ---: | ---: | ---: |
| PMX | +0.003 | **+0.021** | wells −0.207 | wells **+0.983** |
| PM1 | +0.007 | +0.007 | wells −0.039 | wells **+1.156** |
| F3.1 | +0.004 | **+0.064** | | |
| G2 | +0.001 | 0.000 | Fock −0.065 | Fock −0.099 |
| SR2 | +0.003 | 0.000 | | |

Intervals are all narrower than ±0.004. The V_φ route holds (PMX and F3.1, layer 1), and so does the wells route (PMX and PM1, layer 1). **Level 0 holds on the V_φ route alone**, which is a pairwise interaction. The wells' contraction is ~50× larger, but SY0 check 5 showed that a fixed external attractor reads as Γ_spec of the same size (+0.98 at d = 384), so it is consistent with tokens converging on the wells' fixed centres rather than on each other. The **Fock channel expands** the token set at both layers. SR2's V_φ does nothing at layer 1, as CD found.

**Level 1: no grouping meets the criterion in any arm.**
- **V_φ (a):** the selection enrichment of coreferent and head-dependent sources is 0.97–1.15 in PMX, PM1 and F3.1, against the 1.2 required: V_φ does not choose related sources. The attraction contrasts are heavy-tailed and change sign between layers (coreference, PMX: −9.0 at layer 0, +0.24 at layer 1). Same-POS sources are enriched 1.35–1.46×, but the B_spec they would need is at most 0.0006 in magnitude, or negative.
- **Wells (b):** the dominant-mode co-assignment gain is null for every grouping, and so is the NMI gain against POS.
- **(c) B_spec for V_φ** is at most 0.005 in magnitude everywhere: V_φ moves tokens without binding related ones.

**A replicated structural signal that the pre-registered rule does not admit (descriptive).** At layer 1 the wells **bind noun-chunk members** and **separate head–dependent pairs**:

| | PMX | PM1 |
| --- | ---: | ---: |
| chunk B_spec | **+0.044** [+0.039, +0.049] | **+0.026** [+0.022, +0.029] |
| dep B_spec | **−0.022** [−0.024, −0.020] | **−0.027** [−0.028, −0.025] |
| coref B_spec | −0.007 | −0.009 |

All shuffled controls are null. In PM1 the result is identical against the common, force-level and switch-off references (+0.026, +0.025, +0.026). The binding is **made by the layer**: chunk B at layer 1's input is −0.010 (PMX) and at its output +0.044, and with the wells off it is +0.0005. It fails level 1 only because criterion (b), dominant-mode co-assignment, is null. The modes that dominate each token's force are not shared by chunk members, even though the wells' displacement pulls them together. Whether (b) is too coarse a readout for this effect is a question for a follow-up, not a reason to rescore.

**Also descriptive:** G2's V_φ selects coreferent sources at 2.2× the matched rate and same-POS sources at 3.2–4.7×, the only arm where V_φ's routing is strongly structured. Its states do not move accordingly (B_spec ≈ 0). At layer 1, G2's V_φ force is only 30% potential gradient (the rest is the selection term), so the attraction readout covers a minority of it.

| call | prediction | called at | result |
| --- | --- | ---: | --- |
| SY-L0 | level 0 holds | 75% | **HIT** |
| SY-L1 | level 1 holds | 45% | **MISS** |
| SY-pair | V_φ attracts coreferent or head–dependent pairs | 50% | **MISS** (enrichment 0.97–1.15) |
| SY-wells | the wells pass 1 (b) and V_φ does not | 25% | **MISS** (neither passes) |

**Reading.** By the pre-registered criteria the verdict is **semantically blind**. The couplings synchronize tokens: V_φ pairwise, modestly (Γ_spec 0.02–0.06 at layer 1), and the wells strongly, but by convergence on fixed centres. Neither route binds coreferent or syntactically related tokens. Levels 2 and 3 cannot hold when level 1 fails. SY5 (does Gate 2's blow-up come with over-synchronization?) and the GPT-2 comparison stand on their own and remain worth running. The wells' chunk/dependency pattern is the one lead: a layer-made, replicated, shuffle-controlled structural effect that the pre-registered readout did not confirm. Scope: L = 2, 532.5M tokens (§5.19).

#### SY on the L = 4 Fock arm (descriptive depth reading) and a pattern common to every arm — **2026-10-14**

The arm was run at the author's request, through the same two scripts with the same checks; every check passed. On this arm the displacement reference disagreed with the force reference at layer 3 (+0.018 against +0.024), so it is read at the force level. It is a single arm, so it cannot change any call.

**Coupling profile over depth (Γ_spec, force reference):**

| | layer 0 | layer 1 | layer 2 | layer 3 |
| --- | ---: | ---: | ---: | ---: |
| V_φ | +0.000 | +0.001 | +0.004 | **+0.024** |
| Fock | −0.052 | −0.002 | −0.002 | **−0.111** |

There is no accumulation through depth. V_φ is nearly idle in the two middle layers (switching it off moves their output by 0.016 and 0.078, against 1.2 and 2.4 at the outer layers). Its only contraction above the floor is at the last layer. The Fock channel expands the token set at both ends. No grouping's B_spec reaches the floor for either coupling at any layer (largest: Fock, noun chunks, layer 3, −0.015). **Depth does not turn the L = 2 verdict around in this arm.**

**A pattern in every arm, which no coupling we measured accounts for.** In the states themselves, noun-chunk members end up closer and head–dependent pairs further apart, from identical embedding values (chunk B −0.007 to −0.010, dependency B +0.014 to +0.015):

| arm | chunk B at the final output | dependency B at the final output |
| --- | ---: | ---: |
| F3.1 | +0.032 | −0.009 |
| SR2 | +0.032 | −0.013 |
| G2 | +0.047 | −0.011 |
| PM1 | +0.035 | −0.018 |
| PMX | +0.044 | −0.014 |
| L4F | +0.042 | −0.015 |

At L = 4 it forms almost entirely in the **last** layer (chunk B −0.003 → +0.015 → +0.042 over the last two). In the wells arms the wells carry it (PMX: +0.044 with the wells on, +0.0005 off). In the arms without wells, neither V_φ nor the Fock channel does (their B_spec is at most 0.015 in magnitude, and in the wrong direction). So it is made by the rest of the step: the context-conditioned single-token potential V_θ (through ξ) and the integrator.

**Two readings, not yet separated:**
1. Binding through the shared context ξ, a mean-field coupling the programme has not yet measured as a coupling. The test is a replay with ξ frozen at its input value.
2. **Readout geometry.** The pattern forms where the states are shaped to predict the next token. Tokens inside a noun phrase share a continuation, while a head and its dependent play different roles. That would make it a property of prediction, not of binding. The test is to measure the same contrast on the next-token distributions (the logits): if the logit-space contrast explains the state-space one, reading 2 holds.

Until these are separated, the pattern is not evidence of binding.

#### The two separating tests — **criteria fixed 2026-10-14, before either is run** (`sync/sy_xi_pred.py`, all six arms)

**Test A: binding through ξ.** Every step computes ξ from its own input, and the step is replayed with ξ replaced:
- **own:** each token's ξ computed from that token alone, so no other token acts through ξ.
- **common:** the sequence-average ξ given to every token.

B_tot,ξ = B(on) − B(own) and B_spec,ξ = B(on) − B(common).
**Reading 1 holds** if B_spec,ξ at the last layer passes the floor for noun chunks (positive) and for dependencies (negative), with the shuffled control null, in PMX and in at least one other arm.

**Test B: prediction geometry, at the last layer's output.** The readout is linear in that state (logits = W h + b), so the test cannot simply ask whether the logits show the pattern. It measures three contrasts:
- **B_target:** on the cosine of the true next tokens' output embeddings. Does the data's continuation structure have the pattern?
- **B_pred:** on the Bhattacharyya overlap of the predicted distributions.
- **B_resid:** the state contrast after removing, per sequence, the linear fit of state cosine on prediction overlap.

**Reading 2 holds** if B_resid is null for noun chunks in at least five of the six arms while B_state passes. It is **contradicted** if B_resid passes for noun chunks in PMX and in at least one other arm.

Both readings can hold; neither is exclusive.

#### The decoding probe — **criteria fixed 2026-10-14, before it is run** (`sync/sy_coref_probe.py`, all six arms)

Every SY state measure asks whether related tokens become more alike. A transformer binds differently: attention copies the antecedent's information into the later token without making the two states similar. The probe tests for binding of that kind.

- **Task:** 2,074 pronouns (in 188 sequences) that fastcoref links to an earlier non-pronoun mention. For each, rank the true antecedent's last token against the last tokens of up to 7 preceding mentions of other chains: 7.6 candidates on average, chance 0.138. The nearest candidate is the antecedent 46.8% of the time.
- **Probe:** a rank-32 bilinear score, (U h_m)·(V h_c), which is one attention head's query–key form. Trained on 150 sequences and tested on the other 50, separately on the embeddings and on each layer's output.
- **The dynamics make decodable binding information** if the final-layer probe's test accuracy exceeds the embedding probe's by at least 5 points, with the 95% bootstrap interval over test sequences excluding 0, and exceeds the distance-only baseline. **Replicated** if this holds in PMX and in at least one other arm.
- **Coupling ablation** (added before any probe run). The final-layer probe, trained on the trained model's states, is applied on the test sequences to the last step replayed with each coupling at its common-mode reference, switched off, and with ξ per token alone or as the sequence average. **The readout relies on a coupling** if accuracy falls by at least 5 points under its common-mode reference, with the interval excluding 0. This is meaningful only where the dynamics make binding information.

#### Binding in a subspace — **criteria fixed 2026-10-14, before it is run** (`sync/sy_subspace.py`, all six arms)

B_spec is recomputed after projecting the states onto a fixed family of subspaces of each layer's output, fitted on sequences 0–49 and measured on sequences 50–199:
- principal components 1–16, 17–64, and 65 onward;
- the span of the 64 mode centres (PM1, PMX);
- the 64 directions of the coreference probe's query and key maps (last layer).

Random orthonormal subspaces of 16, 64 and 320 dimensions are the matched-dimension controls; projecting onto fewer dimensions inflates cosines by itself. The context ξ is included as a coupling.

**A hidden binding signal:** for a linguistic grouping, a coupling and a subspace, B_spec is positive and passes the floor, its shuffled control is null, and it exceeds the matched random subspace's B_spec by at least the floor. It must replicate in PMX and in at least one other arm with the same route (for ξ: PMX and any other arm). With five subspaces, five groupings and up to three couplings per arm, the replication requirement carries the burden of multiplicity.

The whole-space statistics reproduce `sy_measure`'s stored ones exactly (checked on the smoke run: max |d| 0).

#### The separating tests and the probe scored: **the noun-chunk pattern is made through ξ and is not prediction geometry; the dependency pattern is mostly prediction geometry; no arm carries decodable coreference information beyond the embeddings and recency** — **2026-10-14**

`sync/sy_xi_pred.py` and `sync/sy_coref_probe.py`, all six arms, 200 sequences. Every check passed: the replays are bit-exact, every patch restores, and the logits rebuilt from the last step equal the model's exactly.

**Test A: ξ, last layer, B_spec,ξ (on minus the common-ξ replay)**

| arm | noun chunks | dependencies |
| --- | ---: | ---: |
| PMX | **+0.038** [+0.033, +0.044] | **−0.033** [−0.036, −0.031] |
| PM1 | **+0.027** [+0.017, +0.035] | **−0.027** [−0.030, −0.025] |
| G2 | **+0.064** | +0.003 |
| L4F | **+0.033** | −0.017 |
| F3.1 | +0.017 | −0.017 |
| SR2 | +0.015 | **−0.027** |

The shuffled controls are null throughout. **Reading 1 HOLDS:** in PMX and replicated in PM1, both the chunk binding and the dependency separation pass through ξ. The chunk binding passes the floor in four of six arms and is positive with a tight interval in the other two.

**Test B: prediction geometry, last layer's output**

| arm | chunk: state | chunk: residual after prediction | chunk: true next tokens | dependency: residual | dependency: true next tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| PMX | +0.044 | **+0.042** | +0.001 | −0.005 | −0.014 |
| PM1 | +0.035 | **+0.034** | +0.001 | −0.009 | −0.011 |
| F3.1 | +0.032 | **+0.029** | −0.001 | −0.002 | −0.015 |
| G2 | +0.048 | **+0.044** | −0.000 | −0.002 | −0.015 |
| SR2 | +0.032 | **+0.030** | +0.001 | −0.004 | −0.011 |
| L4F | +0.042 | **+0.038** | −0.000 | −0.005 | −0.015 |

State cosine and prediction overlap correlate at 0.51–0.52 in every arm. **Reading 2 is CONTRADICTED for noun chunks:** the residual passes in all six arms, and the data's continuations have no chunk structure (true next tokens ≈ 0). For **dependencies, prediction geometry accounts for the pattern:** the residual is below the floor in every arm, and heads and dependents do have dissimilar continuations (−0.011 to −0.015).

**The decoding probe:** no arm passes. Test accuracy, against 0.541 for the nearest candidate and 0.136 for chance:

| | embeddings | best layer | final layer | final − embeddings |
| --- | ---: | ---: | ---: | ---: |
| PMX | 0.464 | 0.532 | 0.532 | +0.068 [+0.012, +0.127] |
| PM1 | 0.516 | 0.563 | 0.446 | −0.070 |
| F3.1 | 0.473 | 0.545 | 0.477 | +0.005 |
| G2 | 0.459 | 0.575 | 0.475 | +0.016 |
| SR2 | 0.511 | 0.516 | 0.507 | −0.005 |
| L4F | 0.414 | 0.584 | 0.480 | +0.066 |

PMX clears the 5-point margin over its embeddings but not the distance baseline. No probe on any layer of any arm beats recency by more than 4 points. The probe peaks after the first layer and falls at the last, where the states are shaped for prediction. **The ablation is descriptive only,** since no arm makes binding information. The readout falls 7–20 points under the common-ξ replay in four arms, and 12 under the wells in PMX. But a probe applied to perturbed states degrades with the size of the perturbation (switching off G2's Fock channel costs 29 points), so the drops measure perturbation, not reliance.

**Reading.**
1. **Noun chunks:** the one structural regularity the dynamics create in every arm is **binding through the shared context ξ**, a mean-field coupling and not a pairwise one. It is not an artefact of next-token prediction: the data's continuations do not have it, and it survives removing everything prediction similarity explains. In the wells arms the wells carry the same pattern; whether they act through ξ or alongside it is not separated.
2. **Dependency separation:** mostly prediction geometry.
3. **Coreference:** not bound in either sense, by closeness or as decodable information beyond lexical compatibility and recency, at L = 2 or L = 4.

This adds a qualified positive to the "semantically blind" verdict on the pre-registered routes: these models bind syntactic constituents through the context field, and do not bind reference at all. These tests were fixed before they ran but are follow-ups, so they do not rescore level 1. Scope: §5.19.

#### SY5 and SY6: amendment and additions — **fixed 2026-10-15, before either is built**

**SY5 (does extra depth over-synchronize?), amended.** The pre-registered criterion used the centred coherence R, which SY0 rejected because it falls as synchronization completes, so it could not detect over-synchronization. It is replaced by the admitted measure:
- **Intervention:** Gate 2 exactly as Cell 6b-7 defines it, N steps at the trained Δt with the "hold" layer schedule, run through the cell's own `_fom_stack`. Two-layer arms at N = 2 (trained), 3, 4; the L = 4 arm at N = 4, 6, 8 (descriptive). Check first: at N = L the patched stack reproduces the trained logits bit for bit.
- **Measures on the final states (the states the readout reads):**
  - **S_N:** the pairwise contraction of the final states at N steps against the trained N = L, the mean over pairs of ln(distance at N = L / distance at N); positive when the tokens have drawn together;
  - **effective rank:** exp of the entropy of the normalized squared singular values of the centred final states, per sequence;
  - perplexity at each N, which must reproduce Gate 2's blow-up.
- **Over-synchronization holds in an arm** if from N = 2 to N = 3, S_3 is at least 0.05 with its interval excluding 0, **or** the effective rank falls by at least 20%, while perplexity rises.
- **SY-G2 (35%) is a HIT** if over-synchronization holds in PMX and in at least one other two-layer arm.
- Descriptive: the noun-chunk and head–dependent binding of the final states at each N.

**SY6 (the matched GPT-2), additions.** SY6 as designed measures the state-level contraction and binding with attention as the coupling, replayed per layer switched off and as its token average (the common-mode reference). The replay must reproduce the trained layer bit for bit. Two tests are added, with the criteria already used for the Lagrangian arms:
1. the **coreference decoding probe** (`sync/sy_coref_probe.py`'s task, probe and criterion: the final layer beats the embeddings by at least 5 points with the interval excluding 0, and beats the nearest-mention baseline);
2. the **prediction-geometry test B** (`sync/sy_xi_pred.py`'s B_target, B_pred and B_resid on GPT-2's last-layer states, read after its final normalization).

**SY-GPT2 (60%), made precise** (the call said only "a binding contrast at least as large as the best Lagrangian arm's"). For each of coreference, head–dependent and noun chunks, compare GPT-2's attention-made binding (B_spec at its last layer) with the largest coupling-made B_spec at the last layer over all Lagrangian arms and couplings. **HIT** if GPT-2's is at least as large for at least two of the three.

#### SY extended to the two exchange-field arms of Gen 3 — **fixed 2026-10-15, before any code is written or any model is measured**

The two published Gen 3 arms not yet measured are added, at the author's request:
- **G3** (`attention-potential-live`, settled PPL 54.21);
- **G3′** (`attention-potential-rfqk-live`, the hardened version with query–key normalisation and the exchange field's parameters in their own gradient-clipping group at 0.3, settled PPL 52.90).

Both are built from the registry's Cell 0 settings (`attention_potential`, reverse channel on, every gradient path live) and their local checkpoints.

**The new coupling.** The exchange field is a conservative, attention-shaped potential. With `RELAX_GRAD_PATH = 'live'` it enters the step only as an explicit force (`relax_field.force_live`, added to `f_phi` in `_layer_forces`, scaled by the gate λ). Its replays:
- **off:** λ = 0 (restored exactly);
- **force-level common:** its force replaced by its token average;
- **displacement-level common:** as for the other couplings, adopted per arm by the same agreement check on V_φ.

The usual checks come first: the "on" replay is bit-exact, each patch moves the step and restores, and the perplexity on the SY sequences is consistent with the recorded value.

**Measurements:** every SY script run so far (state level, route readouts, the ξ and prediction tests, the coreference probe, binding in a subspace, extra depth), with their criteria unchanged.

**Scoring.**
1. These arms were added after the pre-registered calls were scored. **They do not rescore SY-L0, SY-L1, SY-pair, SY-wells, SY-G2 or SY-GPT2**, which stand as recorded; for the V_φ and Fock routes they are descriptive.
2. **The exchange route is scored as a follow-up** with the level 0 and level 1 (c) criteria (Γ_spec, and B_spec with the shuffled control null). PMX has no exchange field, so the replication rule for this route is: **G3 and G3′ both**.
3. The question it answers: does a conservative, attention-shaped coupling bind (noun chunks, or coreference) where V_φ, the wells and the Fock channel do not, and does it resemble GPT-2's attention (grouping built mid-depth, no reference binding)?

#### SY on G3 and G3′ scored: **the exchange field binds nothing (level 1 (c) fails in both); its level 0 holds by the fixed rule but not under the force reference; the context field ξ makes the strongest noun-chunk binding of any arm** — **2026-10-15**

Every SY script was run on both arms, 200 sequences each. Every check passed:
- the "on" replays reproduce the trained steps bit for bit, and the logits rebuilt from the last step equal the model's;
- every patch, the exchange field's included, moves the output and is restored exactly;
- the extra-depth stack at N = L reproduces the trained logits.

On both arms the displacement and force references agree on V_φ, so the displacement reference was adopted, as the rule fixes. The first route readout failed on both arms before measuring anything (V_φ's pair potential needs ξ in these arms); it was fixed by passing ξ and re-run.

**The exchange route (scored)**

| | G3 | G3′ |
| --- | ---: | ---: |
| Γ_spec, layer 0, displacement reference (adopted) | **+0.056** [+0.055, +0.057] | **+0.112** [+0.110, +0.113] |
| Γ_spec, layer 0, force reference | −0.002 | −0.008 |
| Γ_spec, layer 1, displacement / force | −0.226 / −0.138 | −0.286 / −0.144 |
| B_spec, layer 1: noun chunks | −0.012 | −0.003 |
| B_spec, layer 1: coreference | −0.008 | −0.007 |
| B_spec, layer 1: head–dependent | −0.015 | −0.010 |
| B_spec, largest positive, either layer | entities +0.010 | entities +0.018 [+0.006, +0.030] (shuffled +0.002, not null) |

- **Level 0 holds for the exchange route by the fixed rule** (G3 and G3′ both), **and the result is not robust to the reference.**
  - On one step, the displacement form of Γ_spec equals the total contraction: a shared translation leaves distances unchanged, so it counts whatever the rest of the step does with the coupling's common push.
  - The force form takes that push out before the rest of the step. It finds no contraction from the field's token-specific action (−0.002, −0.008).
  - The adoption check is made on V_φ and cannot see a disagreement that arises in the coupling being scored.
  - The layer-0 contraction is therefore read as the step's response to the field's common push, not as synchronization by the field. At layer 1 the field spreads the tokens under both references.
  - **Decided by the reference test below: level 0 FAILS.** Read before any normalization, the field contracts at no layer of either arm; the layer-0 reading was the first LayerNorm acting on its shared push.
- **Level 1 (c) fails in both arms.** No grouping passes the floor at either layer. The exchange field moves tokens strongly (switching it off moves the output by 8.9 in both) without binding related ones.
- **Against GPT-2's attention:** the field resembles GPT-2's first two attention layers, which also spread the token set (S −0.21, −0.26) and bind nothing. These arms have no middle depth, where GPT-2's attention builds the chunk grouping (+0.026 at layer 3).

**The other couplings (descriptive)**
- **ξ, test A, last layer:**
  - noun chunks B_spec,ξ **+0.054** [+0.051, +0.058] in G3 and **+0.074** [+0.069, +0.079] in G3′, the largest of any arm (G2 +0.064 was the largest before);
  - head–dependent B_spec,ξ +0.005 and +0.010, the wrong sign, with shuffled controls not null.
- **The head–dependent separation** is still in the states (layer 1 B_on −0.018 and −0.015), but these arms carry it through other couplings:
  - G3: through the exchange field (−0.015);
  - G3′: through the Fock channel (−0.018) and the exchange field (−0.010).
- **Test B:** the noun-chunk residual after prediction passes (+0.031, +0.037), and the true next tokens have no chunk structure (+0.002, −0.000), as in the other six arms. The head–dependent residual is below the floor (−0.009, −0.006).
- **The Fock channel expands the token set** at both layers (G3 −0.026 and −0.056; G3′ −0.052 and −0.089), as in G2.
- **V_φ** does nothing at G3's layer 1 (0.000, as in G2 and SR2). It contracts at G3′'s layer 1 (+0.037 [+0.037, +0.038]).
- **V_φ route readout** (`sync/sy_routes.py`; the rebuilt branch is exact, potential share 1.000): as in the six arms, V_φ does not choose related sources.
  - Coreference enrichment is 1.06–1.15, and head–dependent 1.01–1.19 (G3 layer 1: 1.193 [1.163, 1.221]), against the 1.2 required.
  - Same-POS sources are enriched up to 1.44.
  - The attraction contrasts are heavy-tailed and change sign between layers (coreference, G3′: −5.3 at layer 0, +0.51 at layer 1).
- **The decoding probe fails in both arms.**
  - G3: final layer 0.475 against embeddings 0.471 (+0.005).
  - G3′: final layer 0.482 against embeddings 0.446 (+0.036 [−0.041, +0.123]).
  - G3′'s layer-0 probe reaches 0.620, the highest of any arm at any layer, and 7.9 points above the nearest-mention baseline (0.541). Its interval was not computed; at the final layer the matching interval is about ±0.11 wide, so a margin of this size cannot be told from zero at this test size (442 items).
  - Ablation: neither the exchange field's common-mode replay nor switching it off moves the final-layer readout (drops −0.018 to +0.000).
- **Binding in a subspace:** no hit involves the exchange field.
  - The ξ hits repeat the six arms': part of speech, noun chunks and entities in the top 16 components.
  - G3′ also has the ξ coreference hit at layer 0 (+0.034), the one the six arms showed. Its examination (`sync/sy_coref_subspace_check.py`, revision 2 unchanged) finds it **lexical by the fixed rule, narrowly**:
    - all pairs: +0.034;
    - identical tokens excluded: +0.041;
    - same words excluded: +0.022 [+0.015, +0.030], against +0.003 for the random 16-dimensional projection. That is 0.019 above it, where a hit needs 0.02.
    - Pronoun–antecedent pairs give +0.016, with a biased shuffled control (−0.026), as in every arm.
- **Extra depth (SY5): over-synchronization holds in both arms, at the first extra step.**
  - G3: perplexity 49.6 → 1,025 (+1,967%), S +0.31, effective rank −22%.
  - G3′: perplexity 48.4 → 187 (+286%), S +0.27, rank −67%; at N = 4, rank −81%.
  - Unlike SR2, PM1 and the L = 4 arm, these failures come with collapse at the onset.
  - Unlike PMX and F3.1, the noun-chunk binding decays with the extra steps (G3 +0.034 → +0.020 → +0.010; G3′ +0.041 → +0.000).

**Reading.**
1. **The attention-shaped conservative coupling is not where binding happens.** At L = 2 the exchange field behaves like an early attention layer: it moves and spreads tokens, and it binds no grouping, noun chunks or reference.
2. **What these arms do bind, they bind through ξ, more strongly than any other arm.** The constituent grouping is again a mean-field effect. The exchange field and the Fock channel take over the head–dependent separation that ξ carries in PMX and PM1.
3. **Coreference** is not bound in either sense in these arms either.
4. **SY-G2 stands** as scored. These arms add two more cases of collapse with extra depth, here from the first extra step.

Scope: L = 2, 532.5M tokens, the same as the other arms (§5.19).

#### Which common-mode reference for Γ_spec: a toy check and a mechanism replay — **criteria fixed 2026-10-15, before either is run**

**The problem.** On the exchange field the two references disagree at layer 0 (displacement +0.056 and +0.110, force −0.002 and −0.008; G3′'s values as first measured, +0.112 and −0.008 after the loader correction below). On one step the displacement form of Γ_spec equals Γ_tot in every arm and for every coupling (a shared translation leaves distances unchanged). For Γ it is therefore the switch-off attribution, which SY0 run 1 rejected because it counts a common push. It was admitted for PMX's wells, which have no separable force, by an agreement check on V_φ. That check cannot see a disagreement in another coupling.

**A fact found while designing the replay.** In the Fock arms (G2, L4F, G3, G3′) the step applies LayerNorm twice: once at the end of the integrator, and again after the Fock increment is added. The "pre-LayerNorm" state on which Γ is read is the first LayerNorm's output plus the Fock increment. A shared push from V_φ or the exchange field passes through a per-token normalization before Γ reads it. In the other arms there is one LayerNorm and Γ is read before it.

**Check 6 (toy, `sync/sy0_check6.py`, SY0's conventions: 40 draws, d = 16 and 384, verdict at 384).** The step is a shared per-token map plus a coupling, then a LayerNorm, then a small per-token increment on the normalized state. It is read before a closing LayerNorm, as in the Fock arms.
- **6a:** a pure common push, at three sizes relative to the state spread (0.3, 1, 3). **Passes** if the force-form Γ_spec is null at every size. It is also **informative** if the displacement form passes the floor at some size: the form then misreads a shared push.
- **6b:** a genuine grouping pull (token-specific, as SY0 check 2's) plus a large common push in the same coupling. **Passes** if the force-form Γ_spec is within 0.01 of the same coupling's Γ_spec without the common push, which must itself pass the floor.
- **6c (descriptive):** a common push through a per-token linear response with no LayerNorm (different stiffness per token, as the low-rank harmonic flow gives).

**Mechanism replay (`sync/sy_exch_reference.py`, G3 and G3′, 50 sequences, both layers).** "The push" is the replay with the exchange force replaced by its token mean, read against the switch-off replay: Γ_push = Γ(push vs off), so that Γ_tot = Γ_push + Γ_spec(force) exactly.
- **M1 (check):** the replays reproduce `sy_measure`'s per-sequence Γ values exactly, and the identity holds.
- **M2:** Γ_push read at the integrator's output, before the first LayerNorm, is below the floor in magnitude in both arms: the integrator (per-token mass, harmonic flow) does not make the contraction.
- **M3:** with the first LayerNorm's per-token mean and scale frozen at the switch-off replay's values, Γ_push at the readout is below the floor in both arms: the normalization's rescaling of the shared push makes it.
- **Descriptive:**
  - Γ_push after the first LayerNorm without the Fock increment;
  - linearity (the push scaled by 0.5 and 2);
  - the size of the shared displacement against the states' spread.

**The amendment, decided now and conditional on check 6 passing.** If 6a and 6b pass:
- Γ_spec and B_spec are scored against the **force-level** common reference wherever the coupling has a separable force. The displacement form remains only where none exists (PMX's wells), with its caveat.
- Every earlier score is re-read under the amendment, and any call or criterion that changes is reported.
- M2 and M3 decide only the reading of the mechanism: a normalization artefact (both hold), or the integrator's heterogeneous response to a shared drive (M2 fails). Neither is the coupling's token-specific action, so the amendment does not depend on them.

If check 6 fails, the force form is not validated in this setting, and the exchange route's level 0 is reported as undecided.

#### Check 6 scored: **FAILS on 6b — with a LayerNorm before the readout, neither reference isolates a coupling's token-specific contraction** — **2026-10-15**

`sync/sy0_check6.py` (`sy0_check6_output.txt`), 40 draws, verdict at d = 384:

| case | force form | displacement form (= switch-off) |
| --- | ---: | ---: |
| 6a: pure common push, s = 0.3 / 1 / 3 | 0.000 / 0.000 / 0.000 | **+0.063 / +0.458 / +1.338** |
| 6b: grouping pull alone | **+0.017** | +0.017 |
| 6b: pull + common push, s = 0.3 / 1 / 3 | +0.217 / +0.641 / +0.807 | +0.280 / +1.100 / +2.145 |
| 6c: common push, per-token linear response, no LayerNorm, s = 0.3 / 1 / 3 | 0 (by construction) | −0.001 / −0.014 / −0.101 |

- **6a passes**, and the displacement form misreads a shared push as strong contraction once a LayerNorm precedes the readout.
- **6b fails.** The pull alone reads only +0.017: the LayerNorm rescales each token and undoes most of the pull's contraction (before the LayerNorm it is about ln 2). With a large shared push, the push dominates each token's normalization and the pull's contraction shows almost in full. So the token-specific part's effect depends on the shared part, and the two do not add. The force form reads the specific part *given* the push, which here is 13–47 times the pull's effect alone.
- **6c:** without any LayerNorm, a shared push through a per-token linear response also reads as an effect under the displacement form (−0.10 at s = 3).

**Consequences, by the rules fixed above.**
1. The force form is not validated with a normalization before the readout. **The exchange route's level 0 is undecided** as measured. The amendment is not adopted.
2. The cause is the readout point. The design defines Γ on the pre-LayerNorm step output, but in the Fock arms the readout comes after an intermediate LayerNorm (found while designing the replay). Before any normalization, a shared push is a translation, apart from the integrator's own per-token response (6c's case). There the decomposition is additive except for that response.

**Extension, fixed now, before the mechanism replay runs** (it replaces M1–M3 above as the deciding readout; they are still reported):
- **6b′ (toy):** 6b read on the state before its LayerNorm. **Passes** if the force form is within 0.01 of the pull alone at every push size.
- **Readout before any normalization (the replay):** for the exchange field at each layer, Γ at the integrator's output, before the first LayerNorm, under four references:
  - switch-off;
  - force-common;
  - displacement-common;
  - **specific-only**: the force replaced by its deviation from the token mean, so the field acts with no shared push, read against switch-off.

  The last two orders bracket the interaction between the shared and the specific parts.
- **Decision for the exchange route's level 0:**
  - if, at the integrator output, Γ_spec(force) and Γ(specific-only vs off) agree within 0.01 in both arms, their mean decides by the usual floor;
  - if they do not agree, level 0 stays undecided.
- **The same pre-normalization readout is reported for V_φ in the four Fock arms** (G2, L4F, G3, G3′), to show whether any earlier V_φ score depended on the readout point.

#### The reference test scored: **the exchange route's level 0 FAILS; its layer-0 "contraction" was the first LayerNorm acting on the field's shared push** — **2026-10-15**

`sync/sy_exch_reference.py` (outputs `sy_exch_reference_<ARM>_output.txt`), 50 sequences, on G3, G3′, G2 and L4F. 6b′ passes in `sy0_check6.py`'s second run (`sy0_check6_output.txt`): before the LayerNorm, the force form recovers the pull exactly at every push size (+0.836 against +0.836). In that run 6b's two orders also disagree, as the decision rule would catch (force +0.217 to +0.807, specific-only +0.017).

**Checks (M1):** all pass.
- Stage B reproduces `sy_measure`'s per-sequence Γ exactly (max |d| 0.0) for every arm, layer and coupling.
- Γ_tot − Γ_push − Γ_spec ≤ 1e-16.
- The "on" replay's last LayerNorm equals the recorded step output (0.0).
- The frozen-LayerNorm "off" replay matches the plain one to 1e-5 (the LayerNorm recomputed by hand).

**The exchange field, both readouts**

| | G3, layer 0 | G3′, layer 0 | G3, layer 1 | G3′, layer 1 |
| --- | ---: | ---: | ---: | ---: |
| B (as scored): displacement form (= switch-off) | +0.055 | +0.111 | −0.229 | −0.288 |
| B: force form | −0.003 | −0.008 | −0.145 | −0.147 |
| B: specific-only | −0.002 | −0.003 | −0.148 | −0.148 |
| **A (before any normalization): force form** | **−0.007** | **−0.013** | **−0.171** | **−0.210** |
| **A: specific-only** | **−0.007** | **−0.013** | **−0.172** | **−0.211** |
| A: the shared push alone (M2) | +0.004 | +0.009 | −0.004 | −0.006 |
| B: the shared push, first LayerNorm frozen (M3) | +0.005 | +0.004 | −0.014 | **−0.025** |
| B: the shared push, as read | +0.058 | +0.119 | −0.084 | −0.142 |

Intervals are all narrower than ±0.025 (most under ±0.003).

**Decision (rule fixed before the run).** At the integrator output the two orders agree within 0.002 at every layer of both arms.
- At layer 0 their mean is −0.007 (G3) and −0.013 (G3′): below the floor and negative.
- At layer 1 it is −0.172 and −0.210, spreading.

**The exchange route's level 0 FAILS:** the field contracts the token set at no layer of either arm. This replaces the "holds by the fixed rule" reading of the G3 block above.

**Mechanism (descriptive).**
- **M2 holds** at every layer of both arms: before any normalization the field's shared push contracts nothing (|Γ| ≤ 0.009).
- The integrator's response to the push is exactly linear (relative error 4–7 × 10⁻⁷ for the push scaled by 0.5 and 2). It is only 3–7% token-specific (per-token mass varies ±2%, plus the harmonic flow).
- The push is large: its shared displacement is 22% (G3) and 28% (G3′) of the states' RMS spread at layer 0, and 64% and 71% at layer 1.
- **M3 holds at layer 0** in both arms. Freezing the first LayerNorm's per-token statistics removes the contraction (+0.058 → +0.005 and +0.119 → +0.004). The same LayerNorm without the Fock increment gives +0.059 and +0.128.
- The layer-0 "contraction" is therefore SY0's normalization trap: a large shared push followed by a per-token rescaling makes the normalized states look aligned.
- At layer 1, M3 fails narrowly in G3′ (−0.025 with the LayerNorm frozen). There the Fock increment also responds to the pushed state; it spreads, so the verdict is unaffected.

**V_φ in the four Fock arms (descriptive: no call rests on a Fock arm).** V_φ carries almost no shared push (|Γ_push| at A ≤ 0.002), so at the integrator output all its references agree within 0.002:

| | layer | A (force = specific-only = total) | B, as scored |
| --- | ---: | ---: | ---: |
| G2 | 0 / 1 | −0.002 / 0.000 | +0.001 / 0.000 |
| L4F | 0 / 1 / 2 / 3 | −0.002 / +0.001 / +0.004 / **+0.035** | +0.000 / +0.001 / +0.004 / **+0.024** (force) |
| G3 | 0 / 1 | +0.006 / 0.000 | +0.001 / 0.000 |
| G3′ | 0 / 1 | +0.006 / **+0.043** | +0.004 / **+0.038** |

Read before the normalization, V_φ's contraction is the same or somewhat larger: the LayerNorm partly undoes a token-specific contraction, as check 6b showed. No V_φ reading changes sign or crosses the floor.

**What this changes.**
1. **The exchange route:** level 0 fails and level 1 (c) fails. The attention-shaped coupling neither synchronizes nor binds.
2. **The instrument, from now on:** in arms with a LayerNorm inside the step, Γ_spec is read at the integrator's output, before any normalization. There a shared push is a translation, apart from the integrator's own small per-token response, and the force and specific-only orders bracket it. A coupling's Γ_spec counts only if the two orders agree within 0.01. For binding, B_spec stays on the states the next layer reads, with the references as adopted.
3. **Earlier calls:** none change. SY-L0 was scored on PMX and F3.1 (one LayerNorm, read before it), and the wells' and V_φ's readings stand.

#### G3′ re-measured after a loader bug: **no conclusion changes; G3′'s numbers above are the corrected ones** — **2026-10-16**

**The bug.** The local model builder, `build()` in `debug/verify_pm_switch.py`, reset the exchange field's per-head temperature (`relax_field.logit_scale`) to its initial value 1/0.07 after loading every checkpoint. The fill was meant only for checkpoints that lack the parameter, but it ran unconditionally. G3′'s checkpoint holds trained temperatures (11.6 to 21.7 across the eight heads), so every local replay of G3′ used untrained routing temperatures. G3 has no such parameter and was unaffected; so were the measurements made in Colab during training (G3′'s settled perplexity, 52.90, and its Gate 3 refinement, +115%).
- **Why the checks missed it:** every replay check compares the replay with a forward pass of the same, already altered, model.
- **The size:** on the SY sequences G3′'s perplexity is 50.00 as replayed and 48.44 with its trained temperatures.
- **The fix:** the builder now fills the temperature only when the checkpoint lacks it, and asserts that a stored value loads unchanged.

**Re-measured** (pre-fix outputs kept in `sync/pre_fix_G3p_eta_reset/` and `debug/pre_fix_G3p_eta_reset/`): every SY script on G3′, the coreference subspace check, the reference test, the read-dependence test, RR-B (`sr_pi3b_bookkeeping.py`) and SR-π.4b (`sr_pi4_step_size.py`). Every replay check passed again. The scripts with their own loaders were correct already: `exchange_field_probe.py` fills only when the parameter is missing, and `dp_register_statistics.py` (RR-A) loads strictly.

| G3′ quantity | as first measured | corrected |
| --- | ---: | ---: |
| exchange field, layer 0, displacement / force form (as scored) | +0.110 / −0.008 | +0.112 / −0.008 |
| exchange field, layer 1, displacement / force form | −0.244 / −0.117 | −0.286 / −0.144 |
| exchange field at the integrator output, layer 0 / layer 1 (the deciding readout) | −0.014 / −0.176 | −0.013 / −0.210 |
| exchange-field binding, any grouping, either layer | none above 0.02 | none above 0.02 |
| noun-chunk binding made by ξ, last layer | +0.076 | +0.074 |
| noun-chunk grouping after the prediction control | +0.038 | +0.037 |
| coreference probe, last layer (layer 0) | 0.491 (0.620) | 0.482 (0.620) |
| coreference subspace hit without same-word pairs, above random | 0.018 | 0.019 (still lexical) |
| first extra step: perplexity, S, effective rank | 50.0 → 188, +0.29, −68% | 48.4 → 187, +0.27, −67% |
| RR-B: Gate 3 refined / bookkeeping held; RR3 change | +111% / +503%; +141% | +119% / +468%; +121% |
| SR-π.4b: layer-0 / layer-1 register increment | 7.8× / 0.98× | 7.8× / 0.98× (unchanged) |

**Verdicts.** None changes. The exchange route's level 0 still fails, by the same rule (the two orders now agree within 0.002), and its level 1 (c) still fails. G3′ keeps the strongest ξ chunk binding of any arm, its coreference hit is still lexical, over-synchronization still holds at the first extra step, RR3 still misses in the direction that supports H-RR, and SR-π.4b is unchanged.

#### SY5 scored: **SY-G2 HIT — extra depth over-synchronizes in PMX, F3.1 and G2; the large blow-ups come with collapse, the onset of failure does not** — **2026-10-15**

`sync/sy5_depth.py`, Gate 2 through Cell 6b-7's own `_fom_stack` (hold schedule), 200 sequences. In every arm the trained depth reproduces the trained logits bit for bit.

| arm | PPL at L → first extra step | S (≥ 0.05) | effective rank (≤ −20%) | over-synchronization | next step |
| --- | --- | ---: | ---: | --- | --- |
| PMX | 50.4 → 58.2 (+15%) | **+0.054** [+0.052, +0.056] | −7.5% | **holds** | N = 4: PPL 205, S +0.50, rank −22% |
| F3.1 | 53.2 → 90.1 (+69%) | **+0.058** | −1.8% | **holds** | N = 4: PPL 268, S +0.45, rank −18% |
| G2 | 48.5 → 2,534 (+5,122%) | **+0.473** | −3.6% | **holds** | N = 4: PPL 3,349, S +0.96 |
| PM1 | 51.1 → 75.7 (+48%) | +0.022 | −14.8% | does not | N = 4: PPL 1.1 × 10⁶, S +0.02, **rank −38%** |
| SR2 | 53.3 → 109.8 (+106%) | +0.029 | +3.6% | does not | N = 4: PPL 221, S +0.03, rank −9% |
| L = 4 (descriptive) | 45.4 → 62.1 at N = 6 (+37%) | −0.003 | +9.0% | does not | N = 8: PPL 495, S +0.61, rank −19% |

**SY-G2 (35%): HIT** (PMX, replicated in F3.1 and G2).

**Reading.** Over-synchronization is real and accompanies the **large** failures of extra depth. In every catastrophic case the token set collapses: G2 at its first extra step (S +0.47, perplexity ×52); PMX, F3.1 and the L = 4 arm one step later (S +0.45 to +0.61); and PM1, whose collapse goes through the rank instead (−38%). But it does **not** account for the onset. The first-step failures of SR2 (+106%), PM1 (+48%) and the L = 4 arm (+37%) arrive with no over-synchronization by either measure. So extra time first degrades prediction by some other route, then, one step further, drives the tokens together, at which point the readout can no longer tell them apart. The noun-chunk binding keeps growing with extra steps (PMX +0.044 → +0.070, F3.1 +0.032 → +0.050): the clustered dynamics continue on top of the global contraction. The evidence is co-occurrence across arms and steps, not a causal test.

#### SY6 scored: **SY-GPT2 MISS (0 of 3) — but the constituent clustering is generic: GPT-2 forms it too, by attention in its middle layers** — **2026-10-15**

`sync/sy6_gpt2.py`, 200 sequences. The checkpoint loads strictly, logits rebuilt from the captured trajectory equal the model's, and the layer replay is bit-exact. Perplexity on the SY sequences is 46.0 (49.8 recorded on the validation split). (The first queued run never started: a smoke-test cleanup had deleted the script. It was restored unchanged and run on 2026-10-15.)

**Attention as the coupling, layer by layer (S against the common-mode reference; attention-made B_spec):**

| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S (whole set) | −0.21 | −0.26 | −0.06 | −0.02 | −0.08 | −0.14 | −0.05 | −0.02 |
| noun chunks | −0.006 | −0.006 | +0.013 | **+0.026** | +0.009 | +0.017 | −0.006 | −0.004 |
| coreference | 0.000 | −0.038 | +0.003 | +0.002 | +0.008 | −0.006 | +0.006 | +0.003 |
| head–dependent | −0.008 | −0.020 | +0.006 | +0.010 | +0.003 | +0.004 | −0.004 | −0.002 |

Shuffled controls are null throughout. The noun-chunk binding in GPT-2's own states rises from −0.012 (after layer 1) to +0.058 (after layer 6) and is +0.054 in the final states.

**Prediction geometry on GPT-2's final states:**
- noun chunks: state +0.054, residual **+0.050**, true next words +0.020;
- head–dependent: state −0.008, residual +0.002, true next words −0.061;
- coreference: state +0.129, residual +0.098, which the subspace examination showed to be lexical in the Lagrangian arms.

**SY-GPT2 (60%): MISS, 0 of 3.** At the last layer, GPT-2's attention-made binding is +0.003 for coreference, −0.002 for head–dependent and −0.004 for noun chunks. The best Lagrangian couplings give +0.029 (PM1, ξ), +0.003 (G2, ξ) and +0.064 (G2, ξ). The comparison was fixed at the last layer. Attention's own best layer for noun chunks (+0.026, layer 3) is also below the best Lagrangian value.

**Reading.**
1. **The grouping of constituents is generic.** The matched transformer's states form it as strongly as the Lagrangian arms' do (+0.054 against +0.032 to +0.047), and it survives the prediction control there too. It is a property of language models trained on this data at this budget, not of this framework.
2. **The mechanism and its timing are framework-specific.** Attention builds the grouping gradually through the middle layers (layers 2–5) and contributes nothing at the last one. In the Lagrangian arms a mean field, the context, makes it at the last step. The separation of heads from dependents is the geometry of prediction in both.
3. **Neither family binds reference** (the probe, and attention-made coreference binding at most +0.008).
4. **Trained attention spreads the token set at every layer.** At this depth and budget, the trained transformer is not in the collapsing regime of the idealised attention flow, as the clustering literature's metastable phase anticipates.

#### SY6, first part: the coreference probe on the matched GPT-2 — **GPT-2 also fails the criterion; the same shape as our models** — **2026-10-15**

`sync/sy6_gpt2_probe.py`, with the task, probe, split and criterion of the Lagrangian arms reused verbatim. The model is built from the matched-baseline notebook's own Cell 3, and the checkpoint loads strictly. 442 test items in 48 sequences; chance 0.136; nearest mention 0.541.

| state | embeddings | layer 0 | 1 | **2** | 3 | 4 | 5 | 6 | 7 | final (after ln_f) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test accuracy | 0.457 | 0.554 | 0.600 | **0.643** | 0.638 | 0.638 | 0.609 | 0.557 | 0.502 | 0.541 |

- **Criterion: not met.** Final minus embeddings is +0.084 [−0.002, +0.172], and final minus nearest is +0.000. So GPT-2's dynamics do not make decodable binding information by the pre-registered rule either.
- **Descriptive.** GPT-2's middle layers carry more antecedent information than any layer of any Lagrangian arm (0.64 against at most 0.58), but its best layer is only +0.102 [−0.012, +0.199] above the nearest-mention baseline. The shape matches the Lagrangian arms: the readout rises mid-depth and falls at the last layer, where the states are shaped for prediction.
- **Reading.** At this training budget neither attention nor the Lagrangian couplings produce reference binding that this probe can read. "Reference is not bound" is therefore not, on this evidence, a deficit specific to the framework. With 48 test sequences the intervals are about ±0.1, so differences of a few points between the families are not resolvable with this test set.

#### Binding in a subspace scored, and its one coreference hit examined: **the hit is the same word repeated, not reference** — **2026-10-14**

`sync/sy_subspace.py` on all six arms; every check passed (bit-exact replays, patches restored, whole-space statistics equal to `sy_measure`'s). Hits replicated by the rule (same coupling, grouping and subspace in PMX and another arm):

| hit | PMX | replicated in |
| --- | --- | --- |
| ξ, part of speech, top 16 components, layer 0 | +0.117 | all five others |
| ξ, noun chunks, top 16 components | +0.063 (layer 1) | PM1, SR2 (layer 0), G2 (layer 1) |
| **ξ, coreference, top 16 components, layer 0** | +0.037 | PM1, F3.1, G2, SR2 |
| wells, noun chunks, top 16 components, layer 1 | +0.063 | none |

The chunk and part-of-speech hits restate in the dominant directions what the whole-space measures showed. **The coreference hit is the only one that would change a conclusion**, so it was examined (`sync/sy_coref_subspace_check.py`; not pre-registered, an examination of a hit). The context field ξ includes each token's own state, and many coreferent pairs are one word repeated. Revision 2, fixed after PMX's first run and before any other arm, excluded pairs of the same word (after lower-casing, stripping and removing a possessive "'s", since GPT-2 spells one word with several token ids).

| arm | all pairs | identical tokens excluded | same words excluded | verdict |
| --- | ---: | ---: | ---: | --- |
| PMX | +0.037 | +0.048 | +0.023 (random 16-dim: +0.016) | lexical |
| PM1 | +0.024 | +0.035 | +0.012 | lexical |
| G2 | +0.043 | +0.047 | +0.020 (random: +0.014) | lexical |
| SR2 | +0.045 | +0.058 | +0.022 (random: +0.012) | lexical |
| F3.1 | +0.060 | +0.077 | **+0.054** (random: +0.028) | different words of one entity |

**Reading.** The coreference signal is lexical identity, one word repeated in different spellings. It reaches the first layer through the context field's own-token term. Without same-word pairs it falls to what any 16-dimensional projection shows, in four of five arms. F3.1 keeps a residual between different words of one entity, which does not replicate. Restricted to pronoun–antecedent pairs the signal is absent. That row's shuffled control is biased in every arm (−0.05 to +0.04, never null), so it cannot be read alone; the conclusion on pronouns rests on the probe. **"Reference is not bound" stands**, and with a reason: what looks like coreference in the dominant directions is repeated words, not resolved reference.

### 5.23 RF: do the Poisson-mode registers carry forward-looking state? — **pre-registered 2026-10-12, before any measurement**

**Why.** The chain-of-thought substrate note ([`Chain_of_Thought_Substrate_in_Lagrangian_Models.md`](Chain_of_Thought_Substrate_in_Lagrangian_Models.md)) finds that the Poisson-mode registers are the conservative scratchpad a native chain-of-thought would write to, and that the missing piece is a training signal. Before any such training, this measures whether the registers in the published PMX model already carry state about what comes next, beyond what the hidden state carries. If they do not, the scratchpad is a memory of the past only, and the design changes before anything is trained.

**Method** (`debug/register_forward_probe.py OUT FOLDER {pmx|pm1|g2}`, CPU, about an hour per arm). 200 validation sequences of 512 tokens, fixed seed 20261012. At every position: the full model's per-token loss, the layer-1 occupation vector n̄(t) (64), the layer-1 hidden state h_t (384), and for G2 the register bank's contents. Target: the mean loss over the next w tokens, w in {1, 4, 16}. Ridge probes, regularisation chosen on 25 of the 150 training sequences, scored by R² on 50 held-out sequences, intervals by bootstrap over sequences (1,000 resamples).

**Probes:** n̄ alone; h alone; n̄ and h together; **controls:** n̄ with the modes shuffled across positions (keeps the marginal, breaks the trace); position alone; h reduced to 64 dimensions by PCA (capacity-matched to n̄).

**Primary statistic.** ΔR² = R²(n̄, h) − R²(h) at w = 16.

| id | prediction | called at |
| --- | --- | ---: |
| RF-a | n̄ alone beats the shuffle and position at every w (intervals disjoint) | 80% |
| RF-b | ΔR² ≥ 0.02 at w = 16 | 45% |
| RF-c | ΔR² increases from w = 1 to w = 16 | 40% |
| RF-d | the capacity-matched h (64 PCs) beats n̄ alone at w = 16 | 60% |

**Reading.** RF-b holding: the registers carry forward-looking state the hidden state lacks, the precondition for a scratchpad; the next step is the training signal (the note's §6). RF-b failing with RF-a holding: a memory of the past, not a working memory. RF-d holding alongside RF-b is consistent (the registers add to, but do not replace, the state). Arms: PMX (primary), PM1 (explicit wells), G2 (the Fock slots, as the comparison). Every reading is scoped to these models and their budget (§5.19 scope caveat).

#### RF scored: **a small forward-looking signal in PMX's occupation, none in PM1's; RF-b misses in both; the G2 probe is invalid** — **2026-10-12**

`debug/register_forward_probe.py` on the published PMX, PM1 and G2 weights, 200 × 512 validation tokens, seed 20261012; both exactness checks pass on every arm (captured losses and occupations reproduce the model's to 0). Outputs in each arm's results folder.

| R² on held-out sequences, pooled | PMX w=1 | w=4 | w=16 | PM1 w=1 | w=4 | w=16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| occupation n̄ | 0.003 | 0.007 | **0.024** | 0.001 | −0.002 | −0.017 |
| hidden state h | 0.036 | 0.017 | 0.023 | 0.037 | 0.024 | 0.013 |
| n̄ + h | 0.035 | 0.015 | 0.032 | 0.035 | 0.016 | −0.005 |
| n̄ shuffled | 0.000 | −0.002 | −0.006 | 0.000 | 0.000 | 0.001 |
| h, 64 PCs | 0.031 | 0.017 | 0.012 | 0.031 | 0.017 | 0.010 |

| | PMX | PM1 |
| --- | --- | --- |
| RF-a (n̄ beats shuffle and position at every w, 80%) | **MISS**: beats the shuffle at w = 4 and 16 (intervals [+0.003, +0.021] and [+0.008, +0.064]) but not at w = 1, and never position by a disjoint interval | **MISS**: beats neither at any w |
| RF-b (ΔR² ≥ 0.02 at w = 16, 45%) | **MISS**: +0.011 [−0.006, +0.028] | **MISS**: −0.028 [−0.056, −0.002] |
| RF-c (ΔR² grows with w, 40%) | HIT: −0.001 → +0.011 | MISS |
| RF-d (64-PC h beats n̄ at w = 16, 60%) | MISS: [−0.044, +0.013] | MISS: [−0.000, +0.070] |

- **PMX's occupation carries a small, real signal about the next 16 tokens' loss** (R² 0.024 alone, equal to the full 384-dimensional hidden state's 0.023 and above the 64-PC hidden state's 0.012; the shuffle at −0.006), and it adds to the hidden state (+0.011), but below the pre-registered 0.02 and with an interval crossing 0. **A precondition for a scratchpad in the weak form; not the strong one the call asked for.**
- **PM1's occupation carries none** at any horizon, and adding it to the hidden state hurts at w = 16 (overfitting with no signal). The exact integration made the registers more informative about what follows, not only better integrated.
- **Both signals are weak in absolute terms.** The best R² anywhere is 0.04: the next tokens' loss is mostly unpredictable from any layer-1 state, in a model at a third of compute-optimal.
- **The G2 probe is invalid and is not scored.** The prefix-causal register bank is 32 × 384 = 12,288 features against 125 training sequences; the ridge probe overfits to R² −1 to −4 and the shuffle control fails the same way. The design's capacity matching does not extend to a 12,288-wide input. A valid Fock comparison needs a low-dimensional readout of the bank (the reverse channel's own projection, or the registers' salience and norms). Not a result about G2's registers.
- **Descriptive follow-up to run before the reading is settled:** is PMX's signal content or difficulty? The occupation summarises how much recent text overlapped the modes; if that is a proxy for "how predictable is this stretch", the signal is frequency, not a trace. Test: add the mean loss over the *previous* 16 tokens as a feature and see whether n̄'s increment survives it.

**Follow-up scored (PMX, `… pmx past`): the signal is content, not difficulty.** With the mean loss over the previous 16 tokens (past16) as a competing feature:

| R², w = 16 | past16 | n̄ | n̄ + past16 | h + past16 | n̄ + h + past16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| | 0.022 | 0.024 | **0.051** | 0.045 | 0.052 |

- The occupation's increment **over h + past16** at w = 16 is +0.004 to +0.014 (95% interval, disjoint from 0), where over h alone it was [−0.006, +0.029]. Conditioning on local difficulty sharpens the occupation's contribution instead of removing it.
- Over past16 alone, n̄ adds [+0.023, +0.059] at w = 16, and a disjoint-from-0 increment at every w. n̄ and past16 are nearly additive (0.024 + 0.022 ≈ 0.051): they carry different information.
- So the forward-looking component of PMX's occupation is about **what** the recent context was, not how predictable it was. Small (R² 0.01 added), real, and content-bearing.

**Reading, revised.** RF-b misses, so the strong precondition fails at the pre-registered size: PMX's registers do not add 0.02 of R² about the next 16 tokens to the hidden state. The weak precondition holds, for PMX only: the occupation carries a small, real, content-bearing signal about what comes next (R² 0.024 alone, equal to the full hidden state's; +0.004 to +0.014 over the hidden state once local difficulty is conditioned on; nothing in PM1). For the chain-of-thought substrate note (§6 there): the registers are a memory of the past with a real but faint forward component; the design that writes thoughts to them is not refuted, since nothing has been trained to write, and the claim that the scratchpad "is already there" holds only in this weak form.

### 5.24 MT: do the Poisson modes have transitions? — **pre-registered 2026-10-12, before any measurement**

**Why.** The template design note ([`Community_of_Semantic_Templates_for_Native_CoT.md`](Community_of_Semantic_Templates_for_Native_CoT.md)) reads a Poisson mode as a learned pattern-matching region (centre μ_v, width κ_v, threshold as a binding radius) and its occupation as a firing history. A template bank that can carry a chain of thought needs *transitions*: which template fires next, given which fired. This asks whether the trained PMX modes already have them. RF (§5.23) asked whether the occupation predicts future *loss*; this asks whether it predicts future *firings*.

**Method** (`debug/mode_transition_probe.py OUT {pmx|pm1}`, CPU). Same capture as RF: 200 × 512 validation tokens, seed 20261012, layer-1 input states. A mode **fires** on token t if its overlap E_v(t) = exp(−κ_v²‖h_t − μ_v‖²) exceeds a threshold θ fixed per mode at its 90th percentile over the training split (so every mode fires on 10% of tokens, which removes the dominant-mode artefact of 6b-15 and the clustering note §6.3). For lags Δ ∈ {1, 2, 4, 8}: the 64 × 64 transition matrix P_Δ(u → v) = Pr[v fires at t + Δ | u fires at t], and its **lift** over the marginal, L_Δ(u, v) = P_Δ(u → v) / Pr[v fires].

**The null.** Adjacent states are similar, so firings co-occur without any transition structure. The control is the lift predicted by state similarity alone: the same matrices computed after replacing h_{t+Δ} by its best linear prediction from h_t (ridge, fit on the training split), i.e. the transitions a smooth drift of the hidden state would produce. **Structured transitions** are lift that survives conditioning on that prediction.

**Primary statistic.** The mutual information I_Δ between "u fires at t" and "v fires at t + Δ" over the 64 modes, in bits, and the **excess** ΔI_Δ = I_Δ − I_Δ^{drift}, with bootstrap intervals over sequences (1,000 resamples). Secondary: the number of (u, v) pairs with lift ≥ 2 and ≤ 0.5 at Δ = 1 beyond the drift null (interval disjoint from 0); whether the top transitions are asymmetric (L(u→v) ≠ L(v→u)), which a chain needs and a similarity structure does not have.

| id | prediction | called at |
| --- | --- | ---: |
| MT-a | I_1 exceeds the shuffle null (modes permuted within each sequence) | 90% |
| MT-b | the excess ΔI_1 over the drift null is positive, interval disjoint from 0 | 55% |
| MT-c | ΔI_Δ decays with Δ but is still positive at Δ = 4 | 40% |
| MT-d | at least 20 of the 64 × 63 ordered pairs have asymmetric lift (L(u→v) ≥ 2 L(v→u), both beyond the drift null) | 35% |

**Reading.** MT-b holding: the modes carry transition structure that state similarity does not explain; the bank has implicit transitions, and the inference region is partly learned. MT-b failing with MT-a holding: the modes are patterns whose co-firing is a shadow of the state's own drift; no transitions. MT-d is the one that bears on order: a chain needs asymmetric transitions, and a similarity structure is symmetric. PM1 is the comparison (explicit wells; no forward signal in RF). Every reading is scoped to these models and their budget (§5.19 scope caveat).

#### MT scored: **the modes have transitions beyond the state's drift (MT-a, b, c HIT in both arms), and none of them are asymmetric (MT-d MISS: 0 in PMX, 3 in PM1)** — **2026-10-12**

`debug/mode_transition_probe.py` on the published PMX and PM1 weights, 200 × 512 validation tokens, seed 20261012; the exactness check passes on both (captured overlaps reproduce the model's to 0). Firing rate per mode on the test split 0.099 (target 0.10); no mode silent. Outputs in each arm's results folder.

| summed MI, bits | PMX Δ=1 | Δ=2 | Δ=4 | Δ=8 | PM1 Δ=1 | Δ=2 | Δ=4 | Δ=8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| data | 1.646 | 0.779 | 0.458 | 0.354 | 3.369 | 2.362 | 0.832 | 0.407 |
| drift null (64 PCs) | 1.274 | 0.588 | 0.375 | 0.329 | 2.888 | 2.282 | 0.639 | 0.314 |
| shuffle null | 0.550 | 0.231 | 0.191 | 0.171 | 1.279 | 0.919 | 0.435 | 0.228 |
| **excess over drift** | **+0.373** | +0.191 | +0.083 | +0.025 | **+0.482** | +0.079 | +0.192 | +0.093 |
| 95% CI | [+0.31, +0.57] | [+0.13, +0.36] | [+0.04, +0.23] | [−0.01, +0.14] | [+0.17, +0.96] | [−0.22, +0.54] | [+0.07, +0.42] | [+0.04, +0.27] |

| | PMX | PM1 |
| --- | --- | --- |
| MT-a (I₁ exceeds the shuffle, 90%) | **HIT**, 3.0× | **HIT**, 2.6× |
| MT-b (excess over drift at Δ=1, CI disjoint from 0, 55%) | **HIT**, +0.37 [+0.31, +0.57] | **HIT**, +0.48 [+0.17, +0.96] |
| MT-c (excess decays, still positive at Δ=4, 40%) | **HIT**, +0.08 [+0.04, +0.23] | **HIT**, +0.19 [+0.07, +0.42] |
| MT-d (≥ 20 asymmetric pairs beyond the null, 35%) | **MISS**, **0** (no off-diagonal pair reaches lift 2 at all; top lift 1.9, matched by the null's 1.7) | **MISS**, 3 (16 pairs at lift ≥ 2, 7 beyond the null, 3 asymmetric) |

- **The modes carry transition information that the state's own drift does not explain**, about 0.37 bits summed over the 64 × 64 pairs at lag 1 in PMX (a quarter of the total), decaying over 4–8 tokens. The bank has implicit transitions.
- **The transitions are symmetric.** In PMX no ordered pair even reaches a lift of 2, and the strongest ones are within 0.2 of the null and of their reverse. In PM1 a handful are strong, but of the 7 that exceed the null only 3 are asymmetric, and the top ones (22→40, 8→40, 8→53) are matched by the drift null almost exactly. What the modes co-fire on is *proximity in time*, not a direction: u-then-v is as likely as v-then-u. This is what a similarity structure looks like, and it is not what a chain looks like.
- **PM1's transitions are stronger and more concentrated than PMX's** (16 strong pairs against 0; summed MI 3.4 against 1.6 bits). PM1's wells were explicitly integrated and sit at the stability wall; its modes are coarser objects that fire together on larger stretches. PMX's exact integration spread the firings out. Neither direction of this difference bears on order.
- **A note on the reported-only column.** `I_drift(384)` printed `inf` at Δ = 1 in both arms: the full-state logistic fit saturates to probabilities of exactly 0 or 1 and the MI log diverges. The capacity-matched null the calls use never did. The probe now clips the probabilities; the recorded calls are unaffected.

**Reading.** The weak form of the author's claim is supported: the trained Poisson modes form a bank of learned pattern-matching regions whose firings carry transition structure beyond what the hidden state's drift supplies, in both arms, decaying over a few tokens. The strong form is not: the transitions have no direction. For a chain of thought, a template firing has to make the *next* firing more likely than the *previous* one; these firings make neighbours likely in both directions, which is co-occurrence, not sequence. This is the order-blindness of §4a of the working-memory note, now measured at the level of the bank rather than the trace: the inference region is not merely undefined on paper, there is no asymmetric structure in the trained modes for it to have learned. Every reading is scoped to these models and their budget (§5.19).

### 5.25 XR: the exchange field against the reverse channel, part A — **pre-registered 2026-10-17, before any measurement**

**The question.** The Gen 3 models carry two attention-like routes between tokens, both exploratory instruments:
- the **exchange field**, one hop, a force inside the integrator, recomputed every layer;
- the **register path**, two hops: the registers' write port, then the reverse channel (their only read port), applied after the step.

They have been compared only by their marginal value as trained: the register path is worth 8.0% over the conservative core (57.76 → 53.12), and the exchange field adds nothing measurable on top of it (G3 54.21, G3′ 52.90). The field has never been trained without the registers, so that comparison cannot separate "the field is weak" from "the field is redundant beside the registers". Part A measures, on the trained checkpoints and without retraining, what each route carries, how far it reaches, which positions it serves, and how much each can be narrowed. Part B, an exchange-field-only model trained at 532.5M tokens, completes the factorial; it needs a GPU run and the author's decision.

**Arms and data.** G3 and G3′ (both routes), and G2 (registers only, for the register path without the field), each on its best checkpoint, loaded with the corrected builder (§5.22, "G3′ re-measured after a loader bug"). Validation sequences: the SY set (`sy_cache/sy_tokens.npz`, 200 × 512 tokens), the first 511 tokens of each as input and the next 511 as targets. Intervals: cluster bootstrap over sequences, 1,000 resamples, 95%.

**Measurements** (script `xr/xr_compare.py`; the switches are exact and restore the model bit for bit, checked before measuring):
1. **Removal at inference, 2 × 2** (100 sequences). Per-token loss with the field off (gate set to 0), the read off (read scale set to 0), both off, and neither; costs in nats against the full model. The **overlap** is 1 − cost(both) / (cost(field) + cost(read)): 0 when the two routes carry separate loss, positive when they carry the same loss.
2. **The routes' kernels** (20 sequences, both layers). The exchange field's routing weights $a^{(c)}\_{ts}$, averaged over heads; the register path's effective influence of token $j$ on token $i$, $A_{ij} = \sum_k \beta_{ik} (1 - \sigma^{(i)}\_k) \alpha^{(i)}\_{kj}$, the write weights scaled by the share of each register that the write replaces, then the read weights. Rows are normalised to sum to 1 before comparison. Read off each kernel, over positions $t \geq 64$:
   - the share of routing mass beyond 64 tokens back, and the mean distance;
   - the effective number of sources (exp of the row entropy);
   - the effective rank of the kernel (exp of the entropy of its normalised squared singular values);
   - the previous-token score (mass at $t - 1$), the same-token score (mass on earlier occurrences of the current token) and the induction score (mass on the token after an earlier occurrence of the current token);
   - the overlap of the two kernels row by row: cosine, and Jensen–Shannon divergence.
3. **Where each route's loss comes from** (the 100 sequences of measurement 1). Each target position is classed by its next token:
   - **induction:** the current token occurred earlier, and the next token equals the one that followed that earlier occurrence;
   - **repeated:** the next token occurred earlier in the text, otherwise;
   - **novel:** it did not occur earlier;
   - frequent (top 1% of tokens by frequency in the 200 sequences) against the rest.
   
   Each route's removal cost is computed per class.
4. **Capacity** (20 sequences). Single-head and single-register removal costs rank the 8 heads and 32 registers; then only the top $k$ are kept (heads: $k$ = 1, 2, 4, 6, 8; registers: $k$ = 1, 2, 4, 8, 16, 32), and once more for a random order. The **retained value** at $k$ is (cost with the route removed − cost keeping $k$) / (cost with the route removed).

**Calls** (pre-registered; part A decides no architecture, the arms are measuring instruments):

| id | prediction | called at |
| --- | --- | ---: |
| XR1 | the routes overlap: in G3 and G3′, the overlap of measurement 1 is at least 0.20 | 60% |
| XR2 | the register path carries more: the read's removal cost exceeds the field's in G3 and G3′ | 75% |
| XR3 | the register path reaches further: its share of routing mass beyond 64 tokens exceeds the field's at the last layer, in G3 and G3′ | 55% |
| XR4 | they route different pairs: the mean row cosine between the two kernels is below 0.3 at the last layer, in G3 and G3′ | 55% |
| XR5a | the read serves induction: its removal cost per induction position is at least 1.5× its cost per novel position, in G2, G3 and G3′ | 50% |
| XR5b | the field serves induction: the same for the field, in G3 and G3′ | 45% |
| XR6a | the register path is narrow: the 8 most important registers retain at least 90% of its value, in G3 and G3′ | 40% |
| XR6b | the field is narrow: the 4 most important heads retain at least 90% of its value, in G3 and G3′ | 50% |

Everything else is descriptive. Scope: L = 2, 532.5M tokens, one seed per arm.

**Amendment, 2026-10-17, after a 2-sequence mechanics smoke run and before any scored run.** Measurement 4 did not say how registers are removed from the read. Masking them out of the read renormalises its softmax over the registers that remain, and in the smoke run keeping only 1 to 4 registers this way did worse than no read at all: the reweighting itself moves the model away from its trained behaviour. Head removal in the exchange field renormalises nothing, since the heads add up independently. For a like-for-like measure, **a removed register's terms are dropped from the read and the remaining registers keep their trained read weights**; XR6a is scored on this. The renormalised version is reported as descriptive. The smoke run's numbers (2 sequences) are not used.

#### XR part A scored: **the two routes divide the work: the register path is a broad channel, the exchange field a copying specialist that takes induction over from it; 2 of 8 calls hit** — **2026-10-17**

`xr/xr_compare.py` on G3, G3′ and G2 (outputs `xr/xr_<ARM>_output.txt`). Every check passed: the switches restore the model bit for bit, and the rewritten read and the head-masked field reproduce the trained model exactly when everything is kept.

**1. Removal at inference** (100 sequences; cost in nats per token):

| | G3 | G3′ | G2 |
| --- | ---: | ---: | ---: |
| full model, perplexity on these sequences | 51.36 | 50.26 | 49.95 |
| read off (register path removed) | **1.56** [1.52, 1.60] | **1.34** [1.30, 1.38] | 1.42 [1.34, 1.50] |
| field off | 0.52 [0.45, 0.60] | 0.47 [0.40, 0.55] | — |
| both off | 1.61 | 1.66 | — |
| overlap, 1 − both / (field + read) | 0.225 | 0.084 | — |

**3. Cost by target class** (nats per token; induction 8.3% of positions, repeated 40.3%, novel 51.4%):

| | induction | repeated | novel | induction ÷ novel |
| --- | ---: | ---: | ---: | ---: |
| G2, read off | **2.42** | 1.33 | 1.25 | **1.93** |
| G3, read off | 1.06 | 1.52 | 1.65 | 0.64 |
| G3′, read off | 0.94 | 1.39 | 1.36 | 0.69 |
| G3, field off | **1.75** | 0.51 | 0.27 | **6.56** |
| G3′, field off | **1.74** | 0.44 | 0.22 | **7.84** |
| G3, both off | 2.74 | 1.68 | 1.30 | 2.10 |
| G3′, both off | 2.88 | 1.65 | 1.39 | 2.08 |

**2. The kernels at the last layer** (20 sequences, rows t ≥ 64; at layer 0 the register write is a no-op, so the register path transfers nothing between tokens there):

| | share of mass > 64 back | mean distance | effective sources | effective rank | induction score | previous-token score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G2 register path | 0.53 | 96 | 69 | 28.7 | 0.075 | 0.010 |
| G3 register path | 0.55 | 103 | 61 | 32.3 | 0.057 | 0.010 |
| G3 exchange field | 0.58 | 111 | 79 | 7.6 | **0.132** | 0.015 |
| G3′ register path | 0.52 | 108 | 58 | 10.2 | 0.033 | 0.013 |
| G3′ exchange field | 0.64 | 126 | 79 | 7.4 | **0.158** | 0.013 |

Row overlap of the two kernels: cosine 0.29 (G3), 0.49 (G3′); Jensen–Shannon 0.33 and 0.29 nats.

**4. Capacity** (20 sequences; retained value when keeping the top k by single-unit removal cost; registers dropped without renormalising, per the amendment):

| | k = 1 | 2 | 4 | 8 | 16 | all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G2 registers | 0.10 | 0.48 | 0.67 | 0.85 | 0.95 | 1 (32) |
| G3 registers | 0.08 | −0.08 | 0.25 | 0.50 | 0.96 | 1 (32) |
| G3′ registers | 0.06 | 0.02 | 0.06 | 0.52 | 0.96 | 1 (32) |

| | k = 1 | 2 | 4 | 6 | all |
| --- | ---: | ---: | ---: | ---: | ---: |
| G3 heads | 0.43 | 0.57 | 0.80 | 0.95 | 1 (8) |
| G3′ heads | 0.44 | 0.60 | 0.84 | 0.95 | 1 (8) |

A random order of registers or heads retains less at every k. The renormalised register variant (descriptive) is worse at k = 1 (−0.2 to −4.6), confirming the artefact the amendment avoided.

| call | prediction | called at | result |
| --- | --- | ---: | --- |
| XR1 | overlap ≥ 0.20 in G3 and G3′ | 60% | **MISS** (0.225, 0.084) |
| XR2 | the read's removal cost exceeds the field's in G3 and G3′ | 75% | **HIT** (1.56 vs 0.52; 1.34 vs 0.47) |
| XR3 | the register path reaches further (mass beyond 64 tokens) in G3 and G3′ | 55% | **MISS**: the field reaches slightly further (0.58 vs 0.55; 0.64 vs 0.52) |
| XR4 | row cosine between the kernels below 0.3 in G3 and G3′ | 55% | **MISS** (0.29, 0.49) |
| XR5a | the read's induction ÷ novel cost ≥ 1.5 in G2, G3 and G3′ | 50% | **MISS**: 1.93 in G2, but 0.64 and 0.69 beside the field |
| XR5b | the field's induction ÷ novel cost ≥ 1.5 in G3 and G3′ | 45% | **HIT** (6.56, 7.84) |
| XR6a | the 8 most important registers retain ≥ 90% | 40% | **MISS** (0.50, 0.52; 16 registers retain 0.96) |
| XR6b | the 4 most important heads retain ≥ 90% | 50% | **MISS** (0.80, 0.84; 6 heads retain 0.95) |

**Reading.**
1. **The register path is the broad channel.** It carries most of the loss at every kind of position in both exchange-field models, and removing it costs three times as much as removing the field. It is wide: its kernel has effective rank 29–32 in G2 and G3, and it needs about 16 of its 32 registers to keep 95% of its value. G3′'s register path is narrower in rank (10), as half its registers keep their learned content (§5.18).
2. **The exchange field is a copying specialist.** Its cost is concentrated at induction positions, where the next token is the one that followed an earlier occurrence of the current token: 1.75 nats there against 0.22–0.27 at novel positions. Its kernel puts two to five times the register path's mass on those induction targets. This is the job of attention's induction heads.
3. **Beside the field, the register path gives induction up.** Alone in G2, the read is what serves induction: removing it costs 2.42 nats at induction positions, twice its cost elsewhere. Beside the field its induction cost falls to 0.94–1.06, and the field carries it instead. With both routes removed, the induction cost (2.7–2.9 nats) is close to G2's read-off cost. The total copying capacity is about the same; it is divided differently.
4. **What this says about "the field adds nothing".** As trained, the field adds no perplexity on average (§5.20: G3 54.21 and G3′ 52.90 against G2's 53.12), but it is not idle. It takes over a specific job, which the register path does when the field is absent. Its marginal value on top of the registers is small because the registers can do that job too. Whether the field alone, without registers, could carry the conservative core further remains part B's question: an exchange-field-only model at 532.5M tokens.
5. **Both routes are wider than predicted.** Neither can be narrowed to a quarter or a half of its units without losing more than 10% of its value. They also overlap less in G3′ than in G3 (0.08 against 0.23), so in the hardened model the two routes carry nearly separate loss.

Scope: L = 2, 532.5M tokens, one seed per arm; removal at inference measures reliance, not the value of training without a route (§5.20).

#### XR part B: an exchange-field-only model — **pre-registered 2026-10-17; fixed as binding when the author began the run, before any training step** (drafted earlier the same day)

**Why.** Part A found that the exchange field adds almost no perplexity beside the registers because it takes over a job the registers also do, copying at induction positions. Its value without the registers is unmeasured: every model with the field also has the register path. Part B trains the missing cell, so that the field and the register path are compared as alternatives, not only as additions.

**The arm (XRB-F).** F3.1's configuration exactly, plus the hardened exchange field of G3′:
- Cell 0 as F3.1: `LIVE` (every gradient path live) and `REVERSE_CHANNEL = False`;
- plus `LADDER_MECHANISM = 'attention_potential'`, `RELAX_GRAD_PATH = 'live'`, `RELAX_ATTN_QK_NORM = True` and `RELAX_FIELD_CLIP = 0.3` (G3′'s hardening, the better-behaved of the two field arms);
- seed 0, 532.5M tokens, the ladder's schedule and evaluation, the notebook's variant tag generated as usual. The tag, checked in Cell 0's output before training: `…cgqk_norc_rglive_vplive_xilive_rfqk_rfclip0p3_L2probe…_attnpot`. Like G3 and G3′, the field uses the `'scalar'` gate: its projections start random and its gate is fixed at 1.

**A caveat that holds for all four cells.** "Without registers" means as in F3.1: the 32 registers are still built and written, and their repulsion term (coefficient 0.05) still acts during training. Nothing reads them, so they cannot affect a prediction (`debug/fock_register_read_dependence.py`), but the repulsion gradient reaches the token states through the write port. XRB-F keeps this, so that it differs from F3.1 only by the field. A fully register-free control is a separate question.

**The factorial it completes** (settled perplexity, L = 2, Gen 3):

| | no exchange field | exchange field (hardened) |
| --- | ---: | ---: |
| no register read | 57.76 (F3.1) | **XRB-F** |
| register path | 53.12 (G2) | 52.90 (G3′) |

**Measurements after training.** Settled perplexity (Colab, as for every arm). Gates 1–3 of 6b-7, with Gate 3 at 1.5× the steps. The SY state-level, context-field and probe scripts. XR part A's measurements for the field: removal cost overall and by kind of position, its kernel, and capacity over its 8 heads.

**Calls (draft):**

| id | prediction | called at |
| --- | --- | ---: |
| XRB1 | the field alone helps: XRB-F settles at least 2% below F3.1 (at most 56.60) | 55% |
| XRB2 | the field alone does not match the register path: XRB-F settles above G2's 53.12 | 70% |
| XRB3 | the field is worth more alone than beside the registers: its relative gain over F3.1 exceeds its gain over G2 (G3′: 0.4%) by at least 2 points | 55% |
| XRB4 | without registers the field takes on broad loss: its removal cost at novel positions is at least 0.50 nats (0.22–0.27 beside the registers) | 50% |
| XRB5 | it still serves copying: its removal cost at induction positions is at least 1.74 nats, as beside the registers | 60% |
| XRB6 | it refines better than F3.1: Gate 3 at 1.5× below F3.1's +143% | 40% |

**Reading, fixed now.**
- **XRB1 and XRB2 both hit:** the field is a real but weaker route; the register path stays the stronger instrument.
- **XRB1 hits and XRB2 misses:** the field alone matches the register path, and the two are interchangeable at this budget.
- **XRB1 misses:** the field is valuable only as a specialist beside another route.

**Cost.** One A100 run of about 15 hours (as G3′), then about 2 hours of local measurement. A second run without the hardening would separate the hardening's effect, and is optional.

#### XR part B, first results: **the exchange field alone is worth 6.0% to the conservative core, three quarters of what the register path is worth, and almost nothing beside it** — **2026-10-18** (all six calls scored)

The run (`…cgqk_norc_rglive_vplive_xilive_rfqk_rfclip0p3_L2probe…_attnpot`, output `L2_attnpot_norc_F3.1prim_32500steps_output.txt`) finished cleanly at step 32,500:
- the causal-leak audit was CLEAN at steps 5,000, 10,000, 15,000, 20,000, 25,000, 30,000 and 32,500 (the step-32,500 audit printed nan for its perplexities, as G3′'s and PM1's did);
- no watchdog reload and no spike capture.

**Settled perplexity: 54.27**, the mean of the last three evaluations (54.20, 54.39, 54.22). The best single evaluation was 52.50, at step 31,000.

**Trajectory against its neighbours** (evaluation perplexity):

| step | F3.1 (neither) | **XRB-F (field only)** | G3′ (both) |
| ---: | ---: | ---: | ---: |
| 500 | 463.4 | 458.1 | 453.4 |
| 2,000 | 151.9 | 144.1 | 144.1 |
| 5,000 | 106.7 | 91.6 | 91.4 |
| 10,000 | 85.1 | 72.8 | 72.0 |
| 20,000 | 73.3 | 63.4 | 61.8 |
| settled | 57.76 | **54.27** | 52.90 |

**The factorial, completed** (settled perplexity, L = 2, Gen 3; gain against F3.1):

| | no exchange field | exchange field (hardened) |
| --- | ---: | ---: |
| no register read | 57.76 (F3.1) | **54.27** (−6.0%) |
| register path | 53.12 (G2, −8.0%) | 52.90 (G3′, −8.4%) |

| call | prediction | called at | result |
| --- | --- | ---: | --- |
| XRB1 | the field alone helps: at most 56.60 | 55% | **HIT** (54.27, 6.0% below F3.1) |
| XRB2 | it does not match the register path: above 53.12 | 70% | **HIT** (54.27, 2.2% above G2) |
| XRB3 | it is worth more alone than beside the registers, by at least 2 points | 55% | **HIT** (6.0% alone against 0.4% beside the registers) |
| XRB4 | its removal cost at novel positions is at least 0.50 nats | 50% | **HIT** (1.16) |
| XRB5 | its removal cost at induction positions is at least 1.74 nats | 60% | **HIT** (2.61) |
| XRB6 | Gate 3 at 1.5× below F3.1's +143% | 40% | **HIT, narrowly** (+134%; G3′ +115%) |

**Reading** (the rule fixed in advance: XRB1 and XRB2 both hit). The exchange field is a real but weaker route. Alone it takes the conservative core three quarters of the way the register path does (6.0% against 8.0%). Its value beside the registers is almost nil (0.4%), so the two routes largely substitute for each other: the field's gain is mostly a gain the register path already provides. With both, the model is only 0.4 points better than with the register path alone. The register path stays the stronger instrument. One seed per arm; the settled values have an evaluation spread of about ±1 (the last three evaluations of this run span 54.20 to 54.39, and the best single evaluation was 52.50).

**Cells 6b-7 and 6b-9** (outputs `Cell-6b-7_flow_or_maps_output.txt` and `Cell-6b-9_layer_step_damped_V_theta_geodesic_32500steps_output.txt`; best checkpoint, step 31,000; Cell 6b-7's 12 × 4 × 512 validation batches, on which this checkpoint scores 52.75). Gate 0 passes in both: the patched loop and the replayed step reproduce the trained model exactly.

| measure | **XRB-F** | G3′ | G2 | F3.1 |
| --- | ---: | ---: | ---: | ---: |
| 6b-7 Gate 1, velocity reset | **+113%** (52.75 → 112.42) | +34% | +33% | +34% |
| 6b-7 Gate 2, N = 3 at fixed Δt | **+112%** | +441% | +51% | +92% |
| 6b-7 Gate 3, 1.5× refinement (N = 3, fixed T) | **+134%** (→ 123.35) | +115% | +1,274% | +143% |
| 6b-9 R(geo) | **1.044** (layer 0 0.84, layer 1 1.25) | 1.050 | — | — |

- **XRB6 hits, by 9 points.** Without the register path the model refines about as well as F3.1 and G3′ do (+115% to +143%), and nowhere near the register-path arms (G2 +1,274%, G3 +1,342%). So the exchange field does not cause the large refinement failures. Neither, by G3′, does the register read on its own: G3′ reads registers and refines at +115%, and it differs from G3 only in the routing hardening (the clue in §5.18). Gate 3 still grows away from N = L at every N, so the model is still a map, not a flow.
- **Gate 1 is the surprise.** Resetting the velocity costs +113%, three times what it costs every other two-layer arm (+33% to +34%, with G3′, which carries the same field, at +34%). Without registers this model relies on momentum much more. The field's push is built from past tokens, so a plausible account is that the velocity carries forward what the register path otherwise supplies at each layer. That account is untested; the result was not predicted, so it is descriptive.
- **6b-9.** The step is about as far from the V_θ-only geodesic as G3′'s (R(geo) 1.044 against 1.050). LayerNorm moves R by −0.323, and V_φ plus the exchange field by −0.554, so here the non-V_θ part of the step is carried by V_φ and the field (in G3, the reverse channel moved R by −0.759 and V_φ plus the field by only −0.073). These are displacements, not reliance (§5.20).
- **Two stale lines in Cell 6b-9's text.** It labels the exchange field "non-conservative"; the attention_potential field is the gradient of a potential at each token while the routing weights are held fixed, as the cell's own header says ("every force in this model is the gradient of a scalar potential"). It quotes "+31.3% in trained PPL (66.98 → 87.93)" for the reverse channel, a Gen 2 figure that does not apply to this arm, which has no reverse channel. Neither affects a number above.
- **Training curve (Cell 8).** Smooth descent with no spike. Three labels are left over from older arms: the plot's title, its "TinyStories best (9.14)" reference line, and the right panel's "Fock coupling strengths (alpha_k)", which are the decay coefficients of the five ξ context channels (`xi_alphas` in `training_log.jsonl`).

**XR measurements on XRB-F** (`xr/xr_compare.py XRBF`, output `xr/xr_XRBF_output.txt` and `.json`; the arm added to `sync/sy_replay.py` as `XRBF`). The script was extended on 2026-10-18 to run on a model without a reverse channel; nothing else changed, and it uses part A's 100 / 20 / 20 sequences. Checks: the local build scores 52.7508 on Cell 6b-7's batches (Colab 52.7510), with the trained per-head temperatures loaded; the switches restore the model exactly; the head-mask rewrite with all heads kept reproduces it exactly.

| loss increase when removed (nats per token) | induction | repeated | novel | all positions | induction / novel |
| --- | ---: | ---: | ---: | ---: | ---: |
| **XRB-F: exchange field (no register read)** | **2.61** | **1.17** | **1.16** | **1.32** [1.24, 1.41] | **2.25** |
| G2: register path (no field) | 2.42 | 1.33 | 1.25 | 1.42 | 1.94 |
| G3′: exchange field beside the register path | 1.74 | 0.44 | 0.22 | 0.47 | 7.84 |
| G3′: register path beside the field | 0.94 | 1.39 | 1.36 | 1.34 | 0.69 |

| the route's kernel at layer 1 | effective rank | effective sources | weight on induction targets | weight beyond 64 tokens |
| --- | ---: | ---: | ---: | ---: |
| **XRB-F field** | **25.1** | **107** | **0.142** | 0.57 |
| G3′ field | 7.4 | 79 | 0.158 | 0.64 |
| G2 register path | 28.7 | 69 | 0.075 | 0.53 |

**Capacity.** Two of the 8 heads carry most of the field (single-head removal costs 0.68 and 0.43 nats; the next is 0.10). The top 2 heads keep 53% of its value, the top 4 keep 80%, the top 6 keep 93%. Beside the registers no head stood out (largest single cost 0.074).

**Reading.**
1. **XRB4 and XRB5 hit, both by wide margins.** Without registers the field carries loss at every kind of position and still serves copying.
2. **Whichever route is alone becomes the broad channel.** The field alone has almost the same profile as the register path alone in G2: within 0.2 nats at every kind of position, and the same overall cost (1.32 against 1.42). Its kernel widens to match: effective rank 25, against 7.4 beside the registers and 29 for G2's register path. It keeps its preference for induction targets (0.142, twice the register path's 0.075). So part A's division of labour is not a fixed property of the field. It arises when the two routes are trained together: the register path takes the broad job and the field the copying.
3. **The removal costs measure reliance, not value** (§5.20). Removing the field costs 1.32 nats (perplexity ×3.75), but training with it gained only 0.062 nats over F3.1 (57.76 → 54.27). The same holds for G2's register path (1.42 nats against 0.084). A trained model is built around whatever route it has. The trained gains (XRB1–XRB3) are the measure of value; the removal costs show how the routes are used.

**SY on XRB-F** (the eight SY scripts, run 2026-10-18 on the best checkpoint; outputs `sync/sy_*_XRBF_output.txt`; descriptive, since part B fixed no SY calls). Two scripts were extended for a model without a reverse channel: the arm tables (`XRBF` in `sy_measure.py`, `sy_routes.py`, `sy_xi_pred.py`, `sy_replay.py`), and `sy_exch_reference.py`, whose readout points A and B coincide when the step has one LayerNorm. Every check is exact: the "on" replays reproduce the trained step bit for bit, every patch restores, and the reference script reproduces `sy_measure`'s per-sequence values (M1, max |d| 0). `sy_measure` adopted the displacement-level reference by its fixed agreement test on V_φ.

| measure | **XRB-F** | G3′ (same field, with the register path) |
| --- | --- | --- |
| field's contraction at layer 0, read before normalisation: force form / specific only | **−0.011 / −0.007** (rule: agree, mean −0.009, below the floor) | −0.013 / −0.013 |
| field's contraction at layer 1, same readings | −0.248 / −0.226 (disagree by 0.022; both spread) | −0.210 / −0.211 |
| the field's shared push alone, layer 0 / layer 1 | **+0.061 / +0.080** | +0.009 / −0.006 |
| shared displacement of the push / RMS spread of the states, layer 0 / layer 1 | **1.71 / 3.20** | 0.28 / 0.71 |
| binding by the field, last layer: noun chunk / head–dependent / coreference | **−0.046 / −0.029** / −0.007 | −0.003 / −0.010 / −0.007 |
| binding by ξ, last layer: noun chunk / head–dependent | +0.049 / −0.006 | +0.074 / +0.010 |
| noun-chunk grouping after the prediction control | +0.034 | +0.037 |
| coreference probe, last layer / best layer (nearest mention 0.541) | 0.534 / 0.615 (layer 0) | 0.482 / 0.620 (layer 0) |
| first extra step: perplexity / S / effective rank | 49.6 → 117.7 (+137%) / +0.30 / −41% | 48.4 → 187 / +0.27 / −67% |

1. **The field draws no tokens together** by its specific part, as in G3 and G3′ (level 0 for the exchange route, by the rule fixed for those arms: at the readout before normalisation, the force form and the specific-only form agree within 0.01, and their mean, −0.009, is below the floor). At layer 1 it spreads the tokens strongly; the two forms disagree there by 0.022, more than the rule allows, but both lie below −0.2.
2. **Its shared push is six times larger than beside the registers.** It moves every token by 1.7 times the tokens' spread at layer 0 and 3.2 times at layer 1 (G3′: 0.28 and 0.71). Even before any normalisation that push alone reads as contraction (+0.061, +0.080; M2 fails), because the step does not move every token equally under a shared force: the displacement it causes has a token-specific part of 5–8%, for instance through the per-token masses. This is a common-mode effect and does not count as synchronization by the field.
3. **Without registers the field separates related words.** At the last layer it pushes the words of noun chunks apart (−0.046) and heads apart from their dependents (−0.029), both past the threshold with null shuffled controls (+0.000 and −0.003). Beside the registers it did neither measurably. ξ still groups noun chunks (+0.049, +0.034 after the prediction control), so the two nearly cancel on chunks; the final states keep a chunk grouping of +0.038.
4. **No reference binding**, as everywhere: the probe is at the nearest-mention baseline, switching the field to its common mode leaves the probe unchanged (switching it off costs 0.070, all of it common-mode), and the coreference subspace check reads LEXICAL.
5. **Extra depth over-synchronizes** (S +0.30, rank −41% at one extra step; +0.38 and −76% at two), and the noun-chunk grouping of the final states turns negative (+0.038 → −0.014 → −0.023), as it faded in G3 and G3′.

### 5.26 BF: selective bf16 training — **DRAFT 2026-10-18, for the author's review; not yet binding**

> **Deferred, 2026-10-18, after the A100 measurement** (productionization plan §10.3a): bf16 made a training step only 2% faster, while the analytic V_φ force with layer checkpointing off made it 1.6× faster with identical arithmetic. Confirmed the same day: float32 + analytic + no checkpointing is ×1.62 at 29.9 GiB, bf16 on top +2%. The probe is not run now; the draft is kept for a later scale (d = 768) where the matrix multiplies weigh more.

**Why.** The 2B-token runs (Epic 3) take about 60 h per arm on the A100 (PMX about 100 h). Selective bf16 (Cell 0 `AMP_BF16`; productionization plan §10.3, item 5) is expected to be the largest single speed-up, and it frees the memory that layer checkpointing now uses. It changes the training numerics, so it may not be adopted without showing that it does not change what the model learns.

**What bf16 covers.** The training forward runs under bf16 autocast: V_θ's well generator, the context channels, the output head and the Fock gates and read in bf16; the layer step from the linearisation on (eigensolves, exact substeps, force assembly, the LayerNorm projection) in float32. Evaluation is float32 in every arm. Verified on the CPU (`debug/verify_bf16_switch.py`): with autocast off, bit-identical to HEAD; with it on, the well generator runs in bf16 and every eigensolve and substep input is float32.

**The probe.** SR2's Cell 0 (the FO series' anchor, the fastest arm whose per-layer flow converges) plus `AMP_BF16 = True` and `PROBE_MAX_STEPS = 8_000`; seed 0, so the data order is SR2's published run's. Tag `…sr2_bf16…`. About 2.5–4 h. The reference is SR2's float32 run: its `training_log.jsonl`, evals every 500 steps (92.93 at step 8,000).

**Measurements.** The evals at steps 500–8,000 against SR2's at the same steps; the gradient-norm trajectory; any watchdog trigger or spike; seconds per step and peak memory (also measured beforehand by Cell 6b-17, float32 against bf16).

**Calls (draft):**

| | prediction | called |
| --- | --- | --- |
| BF1 | bf16 tracks float32: the mean of ln(PPL_bf16 / PPL_fp32) over the evals at steps 4,000–8,000 lies within ±0.02 (±2%) | 70% |
| BF2 | at step 8,000 the ratio lies within ±3% | 70% |
| BF3 | no watchdog trigger and no spike capture | 85% |
| BF4 | at least 1.3× faster per training step than float32 | 50% |

**The band, and why it is ±2%.** Evaluation draws fresh batches each time, so consecutive evals of one run differ by a median 1.7% in SR2's constant-rate phase (p90 4.9%), part of it real progress. A ±2% band on the mean over nine evals is therefore well above the eval noise of a mean and well below a change that would matter to any comparison in the programme (the closest pre-registered gaps are 2–3%). Seed variance is not known yet, which is why the band is not tighter.

**Decision rule (draft).**
- **BF1, BF2 and BF3 hit:** every 2B-token Fock arm trains in bf16 (Epic 3), and the 2B cards state it. Runs that are compared against an existing float32 run (the FO series against SR2, the seed pair against G2 and G3′) stay in float32.
- **Any of BF1–BF3 misses:** Epic 3 trains in float32; the miss is recorded and the cause looked for (which bf16 region moves the curve).
- **BF4** decides only whether the switch is worth using, not whether it is safe.

**Open for the author.** Whether to probe PMX as well (its wells make it the most numerically sensitive arm), and whether the probe runs with the analytic V_φ force on (the same function as the autograd force; using it in the probe would test the combination the 2B runs would use).

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
