# First Order against Second Order: The Series on OpenWebText and TinyStories

**What this note is.** A readable summary, written 2026-11-17, of the first-vs-second-order series in §5.13 of the [protocol](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-fo-series). The series trains the same conservative model twice, once with second-order dynamics (a velocity and inertia) and once as a first-order gradient flow, and compares the two. It covers OpenWebText and TinyStories, with and without the Poisson-mode wells. The protocol is a running log: it keeps every pre-registration, amendment, check and output path. This note keeps the questions, the calls, the numbers and the verdicts, each as the protocol records it. Where the protocol gives no verdict, the note says so. The code set (call MS5) is summarised in a separate note, [Code_Set_First_vs_Second_Order.md](Code_Set_First_vs_Second_Order.md). The later direct probes of the memory reading are in [Second_Order_Memory_Probes.md](Second_Order_Memory_Probes.md).

---

## Contents

1. [The question](#1-the-question)
2. [The models and the words](#2-the-models-and-the-words)
3. [All pairs at a glance](#3-all-pairs-at-a-glance)
4. [TinyStories without wells: SO-TS and FO-TS](#4-tinystories-without-wells-so-ts-and-fo-ts)
5. [OpenWebText without wells: FO-b-OWT](#5-openwebtext-without-wells-fo-b-owt)
6. [SR3-crit: critical damping on SR2](#6-sr3-crit-critical-damping-on-sr2)
7. [OpenWebText with 64 wells: FO-b-PMX](#7-openwebtext-with-64-wells-fo-b-pmx)
8. [The memory-substitution hypothesis and the MS calls](#8-the-memory-substitution-hypothesis-and-the-ms-calls)
9. [The K = 16 pair and M1 to M4](#9-the-k--16-pair-and-m1-to-m4)
10. [The dose rule and the K = 2 pair](#10-the-dose-rule-and-the-k--2-pair)
11. [Convergence order of the per-layer flow](#11-convergence-order-of-the-per-layer-flow)
12. [The TinyStories wells set (TS-K2) and the matched GPT-2 (C1)](#12-the-tinystories-wells-set-ts-k2-and-the-matched-gpt-2-c1)
13. [The 2B runs: the TL calls as registered](#13-the-2b-runs-the-tl-calls-as-registered)
14. [What stands and what is open](#14-what-stands-and-what-is-open)

---

## 1. The question

**The author's question (2026-10-03).** On TinyStories, the Gen 1 model Fock-PARFLM trained first-order matched second order: Fock-G1 8.95 against 9.04 (width $d = 256$, $L = 8$ layers, one seed). Does the same hold on OpenWebText, or does that corpus need second-order dynamics? And which corpus statistics separate the two cases? The series was pre-registered on 2026-10-03, before any measurement.

**Scope.** Every reading is at one damping regime: constant $\gamma = 0.1$, stiff modes at $\zeta \approx 0.05$, live gradients, $L = 2$ layers, width $d = 384$, one seed per arm. The protocol does not extrapolate past it.

**Stage 0 (no training, 2026-10-04), in brief.** The corpus statistics were measured on both corpora. The contrarian predictions for long-range dependence (C3 to C5) all missed: OpenWebText has more of it than TinyStories. A post-hoc reading: 58% of TinyStories' predictive information is available from the previous token alone, against 33% for OpenWebText. On the model side, the anharmonic fraction (Stage 0's "M1", not the M1 of section 9) was near zero on the Gen 1 TinyStories model and large on the OpenWebText models. So the full trained comparison ran.

## 2. The models and the words

**Second order and first order.** A second-order layer step carries a velocity: each token has a position and a momentum, and the step follows a damped flow. A first-order step has no velocity: it is a gradient step on the potential.

**SR2.** The anchor of the series: the conservative core at $L = 2$, $d = 384$, with the exact damped flow of the low-rank stiff modes and no register path. Settled perplexity 58.32 on OpenWebText at 532.5M tokens. Its TinyStories twin is SO-TS.

**FO-a.** The Gen 1 definition of first order, carried over: the velocity is reset to zero at every layer boundary. Under the CfC propagator each step still follows an inertial flow in the stiff modes, so the protocol calls FO-a "memoryless second order". Arm: FO-TS.

**FO-b.** A true overdamped first-order step (pre-registered 2026-10-19). The layer step is the gradient flow $dh/dt = \mu f(h)$ with mobility $\mu = dt / (2m)$; the stiff modes relax exactly; no velocity exists. FO-b removes strictly more than FO-a. Arms: FO-b-OWT, FO-b-TS, FO-b-PMX and the dose twins.

**Wells.** In this note "wells" means the Poisson-mode wells: an extra bank of $K$ Gaussian wells per layer, each with a learned half-life, added to the eight wells of the potential $V_\theta$. SR2 and SO-TS have no Poisson-mode wells ($K = 0$).

**PMX, PMX-K16, PM2.** The arms with wells. PMX has $K = 64$ modes (settled 54.63). PMX-K16 has $K = 16$. PM2 has $K = 2$ (PM2-PMX16 on OpenWebText, PM2-TS on TinyStories). "PMX16" in a tag means 16 exactly integrated wells, not 16 modes.

**The gap.** $\Delta$ (for FO-a) or $\Delta^{b}$ (for FO-b) is the first-order arm's settled perplexity minus the second-order arm's, divided by the second-order arm's. A positive gap means second order is better. "Settled" is the mean of the last three evaluations.

**Gates.** From Cell 6b-7. Gate 1: the cost of resetting the velocity at inference. Gate 2: the cost of extra layers at the trained step. Gate 3: the cost of refining the stack into more, smaller steps. FLOW-R refines each layer into substeps separately (the per-layer flow).

## 3. All pairs at a glance

| pair | corpus | Poisson wells K | second order | first order | gap | verdict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| SO-TS / FO-TS (FO-a) | TinyStories | 0 | 5.16 | 5.13 | −0.6% | Δ_TS within ±3%: HIT |
| SO-TS / FO-b-TS | TinyStories | 0 | 5.16 | 5.28 | +2.3% | TS1: MISS, narrowly |
| PM2-TS / FO-b-PM2-TS | TinyStories | 2 | 4.99 | 4.96 | −0.47% | TS2: HIT |
| SR2 / FO-b-OWT | OpenWebText | 0 | 58.32 | 61.16 | +4.9% | FB2: HIT |
| PM2-PMX16 / FO-b-PM2-PMX16 | OpenWebText | 2 | 54.71 | 53.81 | −1.64% | MS2a MISS; MS2b, MS2c, MS2d HIT |
| PMX-K16 / FO-b-PM16-PMX16 | OpenWebText | 16 | 54.24 | 53.73 | −0.93% | MS2e HIT; MS2a, MS2b MISS; MS2c, MS2d HIT |
| PMX / FO-b-PMX | OpenWebText | 64 | 54.63 | 54.02 | −1.1% | FBX1, FBX2 MISS; FBX3, FBX4 HIT |

TinyStories arms ran 19,250 steps (315M tokens); OpenWebText arms ran 32,500 steps (532.5M tokens). One seed per arm. The seed spread (MS1) is not yet measured.

## 4. TinyStories without wells: SO-TS and FO-TS

Anchors: [p-so-ts](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-so-ts), [p-fo-ts](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-fo-ts).

**Setup.** SO-TS is SR2's configuration on TinyStories. FO-TS adds FO-a. Both were planned for 16,250 steps (266M tokens). Both failed the pre-registered settling check (a line fitted to the last 3,000 steps still fell beyond the evaluation noise). So both were extended by 3,000 steps at the floor learning rate, to 19,250, and both then settled.

**Result (2026-10-19).** SO-TS settled at 5.16, FO-TS at 5.13. $\Delta_{\mathrm{TS}} = -0.6\%$. The call "$\Delta_{\mathrm{TS}}$ within $\pm 3\%$" (55%) **HIT**. The Fock-G1 null reproduces with the CfC propagator, live gradients, $d = 384$ and $L = 2$. The paired difference over the extension was a steady 0.0056 nats in FO-TS's favour, far inside the unmeasured seed spread, and is not read as a gain.

**The Stage-0 rule fails its first test.** SO-TS learned strongly anharmonic wells (Stage-0 anharmonic fraction 2.50 at the output step). By the rule fixed on 2026-10-04 this predicted a gap for FO-TS (called 50%): **MISS**. FO-TS's wells are even more anharmonic, and it matches. The protocol's reading: the anharmonic fraction describes a trained model; it does not say whether the task needs second order.

**Refinement.** SO-TS's per-layer flow converges to about 0.10 nats from the trained step, the same limit as SR2 on OpenWebText. FO-TS's flow converges on the letter of the criterion but to a limit 0.28 nats above the trained step.

## 5. OpenWebText without wells: FO-b-OWT

Anchor: [p-fo-b-owt](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-fo-b-owt). Pre-registration (FB1 to FB4): the FO-b block after FO-PMX (the arm's first definition is at [p-fo-b](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-fo-b)).

**Why FO-b.** FO-a keeps an inertial flow inside each step, because SR2's stiff modes sit at $\omega \Delta t \approx 6.6$, far from the regime where a step from rest is a gradient step. So FO-a tests only the velocity between layers. FO-b was made unconditional on 2026-10-19 and run first; FO-a arms became conditional on an FO-b gap.

| id | prediction | called | result |
| --- | --- | ---: | --- |
| FB1 | TinyStories does not need second order: gap within ±3% | 55% | no verdict recorded; FO-b-TS later settled at +2.3% (section 12) |
| FB2 | OpenWebText does: gap of +3% or more | 60% | **HIT**: +4.9% |
| FB3 | inertia inside the step matters more than the velocity between layers | 50% | no verdict recorded; FO-OWT (FO-a on OpenWebText) was triggered but no result is filed |
| FB4 | a trained gradient flow refines: Gate 3 at N = 3 at most +10% on both corpora | 50% | **MISS** on OpenWebText (+19.5%); FLOW-R does not converge either |

**Result (2026-10-20).** FO-b-OWT settled at 61.16 against SR2's 58.32: a gap of +4.9%. On OpenWebText the conservative core trained as a true gradient flow is clearly worse than its second-order twin.

**The shape.** FO-b led early (2.3% ahead at step 500, 6.7% at step 5,000). It stalled between steps 9,000 and 13,000, crossed behind at step 11,000, and lost more ground in the learning-rate decay (decay gain 0.175 nats against SR2's 0.196).

**What else it shows.** One extra layer costs FO-b +0.6%, against SR2's +35%. Its median $\omega \Delta t$ is 0.74 against SR2's 7.55: trained first order, the model builds wells about ten times softer. Its context channels end almost where SR2's do; the two models differ in the wells, not in the context.

**A fairness caveat, recorded with the result.** FO-b ran with the hyperparameters tuned for the second-order arms. The stall reads as an optimisation difficulty, so part of the gap may be a hyperparameter mismatch. One FO-b-OWT run at learning rate 8e-4 was proposed as a check before any 2B decision. No such run is filed in §5.13.

## 6. SR3-crit: critical damping on SR2

Anchor: [p-sr3-crit](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-sr3-crit). SR3-crit is SR2 with its stiff modes critically damped ($\zeta = 1$): it keeps the velocity between layers but removes the oscillation inside the step. It settled at 63.28 (2026-10-20): +8.5% against SR2 and +3.5% against FO-b. The call SC3′ (3% or more behind SR2, called 25%) **HIT**. So the order is SR2, then FO-b, then SR3-crit, and the protocol reads it as: what the second-order model gains from is the oscillation of its stiff modes, not inertia as such. The damping axis (with the learned $\gamma(h)$ arm) is covered in detail elsewhere in the protocol.

## 7. OpenWebText with 64 wells: FO-b-PMX

Anchor: [p-fo-b-pmx](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-fo-b-pmx).

**Why.** PMX leans on its velocity far more than SR2 (Gate 1 +91%, against SR2's +3.9%). If second order earns its keep anywhere at this budget, the protocol reasoned, it is there. FO-b-PMX (pre-registered 2026-10-19) replaced the earlier FO-a twin FO-PMX.

| id | prediction | called | result |
| --- | --- | ---: | --- |
| FBX1 | second order earns its keep in the wells: gap of +3% or more | 60% | **MISS**: −1.1% |
| FBX2 | the gap with wells exceeds the gap without wells by 2 points or more | 55% | **MISS**: −1.1% against +4.9% |
| FBX3 | FO-b-PMX settles below FO-b-OWT | 60% | **HIT**: 54.02 against 61.16 |
| FBX4 | the first-order wells train stably | 70% | **HIT** |

**Result (2026-10-22).** FO-b-PMX settled at 54.02 against PMX's 54.63: a tie, with a slight first-order edge. The wells help both orders, the first-order model about twice as much: PMX is 6.3% better than SR2, FO-b-PMX 11.7% better than FO-b-OWT. The protocol's reading, fixed before the run: PMX's Gate 1 (+91%) measures how PMX was trained, not what the task needs. FO-PMX was conditional on a gap and does not run.

**How the first-order model differs.** Its wells are attractive by a majority (64% and 61% at layers 0 and 1), where PMX's are balanced (50% and 48%). Their half-lives are about the same (medians 8.4 and 9.1 tokens). One extra layer costs it +0.3%, against PMX's +956%. Its per-layer flow does not converge (about 0.58 nats above the trained step, against PMX's converged 0.39).

**Caveats.** One seed per arm. The two runs do not share evaluation batches (PMX was resumed once), which the protocol later put at roughly $\pm 1.2\%$ of evaluation-sampling uncertainty on this gap, before any seed spread. The model is published as `semsimula-ladder-live-owt-d384-l2-none-norc-pm64-pmx16-sr2-fob`, in the first-vs-second-order collection.

## 8. The memory-substitution hypothesis and the MS calls

Pre-registered 2026-10-22, after FO-b-PMX and before any further run. The block follows [p-fo-b-pmx](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-fo-b-pmx); its readings close the block before the TL registration.

**The hypothesis, in plain words.** The perplexity advantage of second order in the conservative core is the advantage of a memory. A second-order step reads two past states, a first-order step one. Inside a step, the exact damped flow of a stiff mode keeps the distinctions between tokens, where a gradient flow contracts them. The Poisson-mode wells are a memory of another kind. Where an explicit memory carries at least what the implicit one carried, the implicit memory is redundant and the second-order advantage disappears. This is functional substitution, not identity of mechanism.

| id | test | prediction under substitution | called | status in the protocol (2026-11-17) |
| --- | --- | --- | ---: | --- |
| MS1 | seed spread: SR2 rerun at seed 1 | seeds differ by under 2% | 65% | not yet scored; no seed-1 run filed |
| MS2 | dose: pairs at K = 16 and K = 256 | the gap falls monotonically with K and is within the seed spread from K = 64 | 60% | scored once K = 256 is in; K = 256 not run; sub-calls at K = 16 and K = 2 scored (sections 9, 10) |
| MS2′ | the mechanism along the dose (Cell 6b-15) | attractive wells rise with K in first order only | 55% | no verdict recorded |
| MS3 | SR3-crit on PMX | within 2% of PMX | 55% | no result filed |
| MS4 | the 2B pair with wells | first order stays within the seed spread of PMX at 2B | 50% | open; SR2-2B ran first instead (section 13) |
| MS5 | a code corpus, four runs | a gap without wells on code, none with them | 45% | see `Code_Set_First_vs_Second_Order.md` |
| MS6 | depth: PMX and FO-b-PMX at L = 4 | first order stays within the seed spread | 45% | no result filed |
| MS7 | FO-b-G3′ against G3′ (a memory read by a channel, not felt as a force) | first order matches G3′ within 2% | 55% | no result filed |

**The wells turned out to be positional (2026-10-23).** Two readings after PMX-K16 changed what "memory" means here. Giving a sequence another text's occupations costs 0.02% to 0.04%, while removing the wells costs 165% (PMX) and 282% (FO-b-PMX). Fixing every occupation at its late value costs about 1% overall, and nothing from token 128 on. So at 0.5B tokens the wells act as a position-dependent potential: a fixed extra landscape with a start-of-sequence correction, not a memory of content. The hypothesis as registered is not refuted by this, but its mechanism is narrower than the wording in the future note (§7) and the book's §17h. The protocol records that those texts need revision and that the wording is the author's decision.

## 9. The K = 16 pair and M1 to M4

Anchors: [p-pmx-k16](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-pmx-k16), [p-fo-b-pmx-k16](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-fo-b-pmx-k16).

**The second-order arm (PMX-K16, 2026-10-23).** Settled 54.24: 7.0% better than SR2 and 0.7% better than PMX at 64 modes. The second-order model gains no more from 64 modes than from 16. It leans on its velocity about 60 times as much as PMX (Gate 1 +5,788%). Both K = 16 arms trained from step 0 at seed 0, so they share training and evaluation batches and are paired by step.

**The first-order twin (FO-b-PM16-PMX16, 2026-10-24).** Settled 53.73: 0.93% better than PMX-K16 (paired by step over the last eight evaluations, 0.87% with spread 0.14%). It leads throughout training.

| id | prediction | called | result |
| --- | --- | ---: | --- |
| MS2a | the K = 16 gap lies in (+1%, +4%) | 50% | **MISS**: −0.9%, first order ahead |
| MS2b | both K = 16 arms settle between their K = 0 and K = 64 counterparts | 65% | **MISS** on both halves: 16 modes do as well as 64 in both orders |
| MS2c | the wells' gain is larger for first order | 60% | **HIT**: 0.130 against 0.073 in log perplexity |
| MS2d | the first-order arm trains stably | 75% | **HIT** |
| MS2e | the twin settles within ±2% of PMX-K16, paired by step | 55% | **HIT**: −0.87% |

**What replaces the velocity: M1 to M4** (registered 2026-10-23, before the twin's evaluations were read). If the twin matched, the match was to be explained. Three candidate replacements were read on both arms, each "fired" if its threshold was met; M4 means none fired. One amendment came before any result: the full context swap was replaced by a swap of the three long context channels only (`xi_ctx_swap`), because the full swap replaces the token's own identity.

| reading | registered threshold | FO-b-PM16-PMX16 | PMX-K16 | verdict (2026-10-24) |
| --- | --- | --- | --- | --- |
| M1, the wells reorganise | layer-0 attractive modes carry half the force, or 15 of 16 modes in use | 23% of the force; 12.7 / 9.5 modes | 4%; 11.0 / 13.6 | not fired |
| M2, the pair force V_φ | removal cost at least 1.5 times PMX-K16's, beyond the noise | +3.60% (2.1 times) | +1.71% | fired |
| M2, the context channels | the same rule, on xi_ctx_swap | +337% (1.01 times) | +334% | not fired |
| M3, the landscape stiffens | V_θ stiffest-mode ω·Δt median at least 15% higher at either layer | 2.13 / 5.31 (layer 1 +16.7%) | 2.87 / 4.55 | fired, narrowly |
| M4, the velocity was spare | the twin ties and none of M1 to M3 fires | | | does not apply (M2 and M3 fired) |

**Reading, as registered.** A tie with M2 and M3 fired: the first-order model found a replacement. It leans about twice as hard on token-to-token coupling through $V_\varphi$, and it builds a stiffer layer-1 landscape (layer 0 is softer). It does not lean more on context. The same signature appears in the $K = 64$ pair, so it is not a quirk of one run. The protocol's scope note: the ablations measure reliance in the trained model, not what a model trained without the term would reach; M3 cleared its threshold by 1.7 points.

## 10. The dose rule and the K = 2 pair

Anchors: [p-pm2-pmx16](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-pm2-pmx16), [p-fo-b-pm2-pmx16](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-fo-b-pm2-pmx16).

**The dose rule (fixed 2026-10-22, before either K = 16 result).** The gap at $K = 16$ was to pick the next pair, so that GPU time goes where the curve is unresolved:

| gap at K = 16 | next pair (as fixed 2026-10-22) | next pair (amended 2026-10-23) |
| --- | --- | --- |
| within ±2% | K = 8, then K = 4 if K = 8 also ties | K = 2 in place of K = 8 |
| in (+2%, +4.5%) | K = 256; K = 8 later | unchanged |
| +4.5% or more | K = 32, then K = 256 | unchanged |

The amendment came after PMX-K16 and the positional reading of the wells, before the first-order K = 16 twin ran. Its reasoning: $K$ sets how finely the positional potential can follow position, so $K = 2$ asks how little structure closes the gap. At $K = 2$, a gap within $\pm 2\%$ closes the low end of the dose; a gap outside it sends the series to $K = 8$.

**Applied.** The $K = 16$ gap (0.93% in first order's favour) was within $\pm 2\%$, so $K = 2$ ran next. The $K = 2$ gap (1.64% in first order's favour) was within $\pm 2\%$ as well, so the low end is closed: no $K = 8$ or $K = 4$. The top of the curve, $K = 256$, remains.

**The second-order arm (PM2-PMX16, 2026-10-24).** Settled 54.71. Two wells capture 88% of what sixteen gain over none, and the model is level with PMX at 64 wells. As the wells shrink, second order takes up the slack elsewhere: stiffer rotation (median $\omega \Delta t$ 2.65, then 3.65, then 4.26 at 64, 16 and 2 wells), more reliance on the velocity and on the pair force, and less on the wells.

**The first-order twin (FO-b-PM2-PMX16, 2026-10-25).** Settled 53.81: 1.64% better than PM2-PMX16 (paired, 1.63% with spread 0.14%). It is level with the first-order $K = 16$ model with an eighth of its wells. The decay narrows the gap (2.8% at step 21,000, 1.6% at the end).

| id | prediction | called | result |
| --- | --- | ---: | --- |
| MS2a | the K = 2 gap lies between K = 0 (+4.9%) and K = 16 (−0.93%) | 50% | **MISS**: −1.64%, first order further ahead |
| MS2b | both K = 2 arms between their K = 0 and K = 16 counterparts | 65% | **HIT** on both halves; the first-order half by 0.08, inside the noise |
| MS2c | the wells' gain is larger for first order | 60% | **HIT**: 0.128 against 0.064 |
| MS2d | the first-order arm trains stably | 75% | **HIT** |

**The dose, both orders.**

| Poisson wells K | 0 | 2 | 16 | 64 |
| --- | ---: | ---: | ---: | ---: |
| gap, first order against second | +4.9% | −1.6% | −0.9% | −1.1% |
| wells' gain, first order (log perplexity) | | 0.128 | 0.130 | 0.124 |
| wells' gain, second order (log perplexity) | | 0.064 | 0.073 | 0.065 |

The crossover lies between no wells and two. The wells help first order about twice as much at every $K$. Read the same way as M1 to M3 (descriptive only), the $K = 2$ twin shows the $K = 16$ signature more strongly: it leans on its two wells 8.3 times as hard as its twin (removal +831% against +100%) and builds a layer-1 landscape 48% stiffer. The pair is published on Hugging Face in the first-vs-second-order collection (2026-10-25).

## 11. Convergence order of the per-layer flow

Recorded 2026-10-24 (descriptive, CPU), in the block after the K = 2 pair. The question: why does FLOW-R's per-layer flow converge by $N = 16$ substeps for the second-order models and for no first-order one? The candidate explanation was the integrator: the second-order step is second-order accurate, FO-b's is first order.

| observed order (k = 4 to 32 substeps) | at initialisation | trained |
| --- | ---: | ---: |
| second order (PMX-K16) | 2.0 | about 1 |
| first order (FO-b-PMX-K16) | 1.0 | about 0.5 |

Before training both integrators have their textbook orders. Trained, both run below them, and second order stays one class ahead: the error ratio grows from 1.5 to 7.7 with refinement. The $K = 2$ pair reproduces this (trained first order about 0.4 overall). The first-order models sit further from their continuous law the fewer wells they have: 0.58 nats (64 wells), 1.41 (16), 2.08 (2). A note of this summary, not of that block: SR3-crit, also a second-order model, did not meet FLOW-R's criterion; its per-layer flow drifted slowly (0.028 against the criterion's 0.02).

**Reading for the book, as recorded.** First and second order differ in the order at which the trained layer converges to its continuous law, not only in how far it sits from it. A design that needs each layer to be a sample of a continuous flow is served by second order; at fixed depth and this budget, perplexity is not.

## 12. The TinyStories wells set (TS-K2) and the matched GPT-2 (C1)

Anchors: [p-gpt2-ts](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-gpt2-ts), [p-pm2-ts](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-pm2-ts), [p-fo-b-pm2-ts](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-fo-b-pm2-ts), [p-fo-b-ts](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-fo-b-ts).

**C1, the matched GPT-2 on TinyStories (2026-10-25).** Here C1 names this run (the code GPT-2 is C3); it is not the Stage-0 corpus statistic of the same name. An eight-layer GPT-2 at the matched settings (33,691,776 parameters), on the ladder's own TinyStories tokens, 19,250 steps. Settled 4.47. It was still improving slightly at the floor, so the protocol calls 4.47 "if anything, generous" to the Semantic Simulation models. On TinyStories the no-wells core trails it by 15%; on OpenWebText by 21% (SR2 58.32 against 48.37).

**The set (registered 2026-10-25).** Three new arms on TinyStories, each from step 0 at seed 0, each run straight to 19,250 steps: PM2-TS (two wells, second order), FO-b-PM2-TS (its first-order twin) and FO-b-TS (true first order, no wells). FO-b-TS is TinyStories' $K = 0$ point in true first order, since FO-TS is FO-a. $K = 2$ was chosen as the smallest dose that closed the OpenWebText gap. Calls TS1 to TS4 were confirmed by the author and binding before any TS-K2 arm trained.

| id | prediction | called | result |
| --- | --- | ---: | --- |
| TS1 | FO-b-TS within ±2% of SO-TS (5.16) | 55% | **MISS, narrowly**: +2.26% (5.28) |
| TS2 | FO-b-PM2-TS within ±2% of PM2-TS | 70% | **HIT**: −0.47% |
| TS3 | the wells help first order more on TinyStories too | 65% | **HIT**: 0.061 against 0.034 in log perplexity, 1.8 times |
| TS4 | PM2-TS settles more than 1% below SO-TS | 50% | **HIT**: −3.4% (4.99) |

| TinyStories, 315M tokens | no wells | 2 wells | against GPT-2 (4.47), no wells / 2 wells |
| --- | ---: | ---: | --- |
| second order | SO-TS 5.16 | PM2-TS 4.99 | +15.4% / +11.6% |
| true first order (FO-b) | FO-b-TS 5.28 | FO-b-PM2-TS 4.96 | +18.0% / +11.0% |
| first order against second | +2.3% | −0.5% | |
| FO-a (velocity reset between layers) | FO-TS 5.13 | | |

**Reading, as recorded.** TinyStories reproduces the OpenWebText pattern at a smaller scale. Without wells, true first order trails second order (+2.3%, against +4.9% on OpenWebText); with two wells the gap closes (0.5% in first order's favour, against 1.6% on OpenWebText). As FO-b-OWT did, FO-b-TS leads early, falls behind (by step 5,000) and stays behind. The earlier TinyStories tie was FO-a's: FO-a keeps the stiff modes' rotation within each layer and ties; FO-b removes it too and trails. The protocol reads this as SR3-crit's result (the gain is the rotation) on a second corpus. The two wells learn nearly the same landscape as on OpenWebText (layer 0 repulsive, layer 1 attractive). With them, the second-order model leans heavily on its velocity on TinyStories too (Gate 1 +3,907%).

## 13. The 2B runs: the TL calls as registered

Pre-registered 2026-10-22, before any 2B run; the block sits just before [p-sr2-2b-setup](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-sr2-2b-setup). Two pairs at 2.13B tokens (130,000 steps) were planned: PMX-2B with FO-b-PMX-2B (for MS4), then SR2-2B with FO-b-2B. Weights-only snapshots along each run separate training length from the schedule phase.

| id | prediction | called |
| --- | --- | ---: |
| TL1 | the potential falls along the trained step for at least 99% of tokens at both layers, from snapshot 16,500 on, in all four arms | 80% |
| TL2 | the per-layer flow penalty at N = 16 is larger at step 84,500 than at 32,500 in at least three of the four arms | 60% |
| TL3 | from 32,500 to 130,000 the first-order arm's penalty rises more than its twin's, in both pairs | 55% |
| TL4 | the first-order arms' median direction cosine at layer 0 stays at least 0.95 at every snapshot | 70% |
| TL5 | the decay-phase pattern holds at 2B: in both pairs the second-order arm gains more from 84,500 to 130,000 | 65% |
| TL6 | the core gap grows with training: SR2-2B is more than 4.9% better than FO-b-2B at the end | 50% |

TL7 to TL10 (the number of modes in use and whether the wells start to remember content) were registered with them. On 2026-11-12 the author put SR2-2B first, ahead of the PMX pair. Its result is filed at [p-sr2-2b-result](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-sr2-2b-result). TL3, TL5 and TL6 need FO-b-2B, and TL2 needs three of its four arms; none of TL1 to TL6 is scored as of this note.

## 14. What stands and what is open

**What stands** (one seed per arm, $L = 2$, $d = 384$, constant $\gamma = 0.1$, 0.5B tokens on OpenWebText and 315M on TinyStories):

- Without the Poisson-mode wells, second order beats a true first-order twin on both corpora: +4.9% on OpenWebText (FB2 hit) and +2.3% on TinyStories (TS1 missed its $\pm 2\%$ tie band).
- The earlier TinyStories tie belongs to FO-a, which keeps the rotation inside each step.
- The gain lies in the oscillation of the stiff modes, not in inertia as such (SR3-crit, SC3′ hit).
- With wells the gap closes on both corpora, already at two wells. The wells help first order about twice as much at every $K$ measured.
- Heavy reliance on the velocity does not mean the velocity is needed (Gate 1 +91% on PMX, +5,788% on PMX-K16, yet both are matched).
- The first-order twins replace the velocity with more pair-force coupling and a stiffer layer-1 landscape (M2 and M3 fired at $K = 16$; the same signature at $K = 64$ and, stronger, at $K = 2$).
- At this budget the wells are a positional potential, not a memory of content.
- Second order's per-layer flow converges at a higher order than first order's.

**What is open:**

- The seed spread (MS1). Every tie in this note, and the 2.3% TinyStories gap, is read against an unmeasured spread.
- The top of the dose ($K = 256$), and so MS2 itself. MS2′, MS3, MS6 and MS7 have no filed result.
- The 2B pairs (MS4, TL1 to TL10). Only SR2-2B has run.
- FO-OWT (FO-a on OpenWebText, for FB3) and the lower-learning-rate check of FO-b-OWT.
- The revised wording of the memory-substitution hypothesis (future note §7, book §17h), which the protocol leaves to the author.
- The code set (MS5): see [Code_Set_First_vs_Second_Order.md](Code_Set_First_vs_Second_Order.md). The direct probes of the memory reading: see [Second_Order_Memory_Probes.md](Second_Order_Memory_Probes.md).
