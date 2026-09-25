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
| 2 | L=2, `'attention'` | 14.5h | — | **DONE**, §5.1 |
| 3 | L=2, `'none'` | 14.5h | what the exchange field contributes at fixed depth | **DONE**, §5.2 |
| 4 | **L=4**, matched | ~27h | hops, versus "the L=8 arm was handicapped" | queued |
| 5 | L=2, `'attention_potential'` | ~14h | the price of conservativity, from scratch, parameter-matched | queued |
| 6 | L=2, `'nonconservative'`, lambda pinned | ~14h | an unconstrained pointwise map, the one function class Fock has nowhere | queued |
| 7 | L=1, `'none'` | ~7h | one hop; the structural floor where the Jacobi metric ceases to exist — **but see §6.1: it also silently disables the Fock registers** | **DONE**, §5.5 |

Run 1 first regardless of ordering elsewhere: it is 2.7 hours and it makes
every other number defensible.

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

> **SUSPENDED 2026-09-22 — the sign has flipped.** Everything in this
> subsection was measured with **both arms at 3e-04**. Tuning `'none'` alone
> to 1.2e-03 took it to **66.98**, which *beats* `'attention'`'s 68.33 by
> 2.0% — so the 6.76 credited to the exchange field below is not merely
> smaller than stated, it points the other way at the only learning rate
> where either arm has been tuned.
>
> | comparison | result |
> | --- | --- |
> | both arms @ 3e-04 | `'attention'` better by 9.0% |
> | `'none'` tuned only | **`'none'` better by 2.0%** |
>
> This does **not** show the exchange field is worthless — `'attention'` has
> never been tuned either and may gain as much or more. It does mean no
> figure in this subsection may be quoted until L=2 `'attention'` has run at
> the winning LR. See §6.4 of
> [`Hyperparameter_Tuning_Checklist.md`](Hyperparameter_Tuning_Checklist.md).


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

| run | prediction | reasoning |
| --- | --- | --- |
| L=2 `'none'` | ~~gap holds at 20-30%~~ **MISS: actual 9.9%** | see §5.2; the gap saturated rather than growing |
| L=2 `'attention_potential'` | **75-82, point 78** | arm C detaches both alpha and `h_src`, so its Jacobian is block-diagonal and there is no inter-token coupling in the dynamics. Below 72 would be a genuine surprise. |
| L=4 | no strong prior | this is the point of running it |
| L=1 `'none'` @1.2e-03 | ~~74-80~~ **MISS: actual 87.09 settled** | forecast made in conversation from the L=2 curve shape; see §5.5 — the L=1/L=2 gap did not saturate, it kept widening through the decay |
| matched GPT-2 | ~~below 54.59~~ **HIT: 49.76 final, 49.81 settled** | predicted 49.5 band 49.0-50.0 from the published run's behaviour over the same lr range; error +0.26 |

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
