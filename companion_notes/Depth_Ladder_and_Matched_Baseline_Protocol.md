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
| 1 | **matched GPT-2**, batch 32 throughout | **2.7h** | removes the 1.48x token confound from every comparison in the programme | queued |
| 2 | L=2, `'attention'` | 14.5h | — | **DONE**, §5.1 |
| 3 | L=2, `'none'` | 14.5h | what the exchange field contributes at fixed depth | **DONE**, §5.2 |
| 4 | **L=4**, matched | ~27h | hops, versus "the L=8 arm was handicapped" | queued |
| 5 | L=2, `'attention_potential'` | ~14h | the price of conservativity, from scratch, parameter-matched | queued |
| 6 | L=2, `'nonconservative'`, lambda pinned | ~14h | an unconstrained pointwise map, the one function class Fock has nowhere | queued |
| 7 | L=1, `'none'` | ~10h | one hop; also the structural floor where the Jacobi metric ceases to exist | queued |

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
| matched GPT-2 | **below 54.59** | more tokens, no mid-run batch discontinuity |

---

## 6. Open risks

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
