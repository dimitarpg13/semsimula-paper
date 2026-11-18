# Second Order as Memory: The Probes

**What this note is.** A readable account of one line of experiments: the direct tests of the book's reading that second order acts as a memory. It covers six probes run on the CPU on 2026-11-16 and 2026-11-17, on trained models, with every prediction registered and confirmed by the author before the measurement. The full log, with every check, is in the [protocol](Depth_Ladder_and_Matched_Baseline_Protocol.md#p-bank-memory-conjecture); this note keeps the questions, the numbers and the verdicts.

**In one paragraph.** The observation stands: trained once with inertia and once as a first-order gradient flow, the conservative core favours second order, by more the more its predictions must recall (2.3% on TinyStories, 4.9% on OpenWebText, 9.2% on Python code). The mechanism the book first proposed does not survive. The second-order step does keep a difference along the potential's stiff directions where a gradient flow contracts it, but that difference reaches later predictions no better than a difference along any other direction. The first-order potential is not softer. Recall runs through the same pathway in both orders, a content-addressed read at the second layer. What differs between the two models is how the wells sit around the state. Which difference gives second order its advantage is open.

---

## Contents

1. [The observation and the original reading](#1-the-observation-and-the-original-reading)
2. [The models and the vocabulary](#2-the-models-and-the-vocabulary)
3. [Probe 1: damping the stiff modes at evaluation](#3-probe-1-damping-the-stiff-modes-at-evaluation)
4. [Probe 2: does a stiff-direction difference reach later predictions?](#4-probe-2-does-a-stiff-direction-difference-reach-later-predictions)
5. [Probe 3: whose force is it? Weight against force](#5-probe-3-whose-force-is-it-weight-against-force)
6. [Probe 4: stiffness and distance, well by well](#6-probe-4-stiffness-and-distance-well-by-well)
7. [Probe 5: the same question at the recall positions, on code](#7-probe-5-the-same-question-at-the-recall-positions-on-code)
8. [Probe 6: which pathway carries the recall?](#8-probe-6-which-pathway-carries-the-recall)
9. [What stands and what is open](#9-what-stands-and-what-is-open)
10. [Files](#10-files)

---

## 1. The observation and the original reading

**The observation.** On code, the first-order model's deficit sits almost entirely in identifiers, and it grows with how far back the name last occurred, although its context channels find those earlier occurrences as well as the second-order model's do.

**The original reading** (book, introduction and §17h, before 2026-11-17): the damped second-order step rotates the stiff directions of the potential and keeps the distinctions between tokens, where a gradient-flow step contracts them away; so the first-order model learns a potential too soft to tell apart what it retrieves. Two trained arms supported the rotation: critically damping the stiff modes (SR3-crit) costs 8.5% on OpenWebText, more than removing inertia altogether; resetting the velocity at every layer boundary costs nothing on TinyStories.

The reading has three parts, and the probes test each:

| part | probe |
| --- | --- |
| the second-order step keeps a stiff-direction difference that a gradient flow contracts | 2, 5 |
| that kept difference is what carries information to the recall | 2, 5, 6 |
| the first-order potential is softer | 3, 4 |

## 2. The models and the vocabulary

All models have two layers, width $d = 384$, and are trained at matched budgets.

- **SR2** (OpenWebText) and **SO-CODE** (Python code): the conservative core integrated as a damped second-order system, with the stiff modes of the potential integrated exactly.
- **FO-b-OWT** and **FO-b-CODE**: the same cores trained as first-order gradient flows (no velocity).
- **SR3-crit** (OpenWebText): SR2 with its stiff modes critically damped in training.

**Wells.** The potential $V_\theta$ is a bank of eight anisotropic wells. Well $k$ has a centre $\mu_k$, a precision $P_k$ (how sharply it curves) and a weight; its weight times its occupancy at the current state is $g_k$. Its force on a state $h$ is $f_k = g_k P_k (\mu_k - h)$, and the model's force is the sum of the $f_k$.

**Stiff directions.** At each token the step computes the stiffest directions of the potential (the leading eigenvectors of its curvature, the columns of a matrix $U$). Along them the second-order step oscillates within the step; a gradient flow contracts.

**Recall positions.** On code, a position $q$ whose next token occurred earlier, last at position $p$; *near* if $p$ is 1 to 16 tokens back, *far* if 17 to 256 back. Tokens that occur more than eight times in the window are excluded, as a rough filter against syntax.

## 3. Probe 1: damping the stiff modes at evaluation

**Question.** If the rotation carries information, removing it at evaluation while keeping the force should cost much more than any untrained change costs.

**Method.** SR2 evaluated with its stiff modes critically damped (the switch SR3-crit trained with), against a control: SR3-crit evaluated with its damping switched off, which puts back an oscillation it never trained with. 16 OpenWebText windows.

| change at evaluation | both layers | layer 0 | layer 1 |
| --- | ---: | ---: | ---: |
| SR2, critical damping | +0.523 nats (+68.7%) | +0.378 | +0.157 |
| control: SR3-crit, damping off | +0.536 nats (+70.8%) | +0.477 | +0.160 |

**Verdict.** DP1 (SR2's cost at least twice the control's) missed: 0.98 times. DP2 and DP3 hit (the cost sits mostly in layer 0 and rises with the damping). Each model is tuned to its own damping, and a change in either direction costs the same. An evaluation-time switch measures that mismatch, not what the oscillation carries. This probe cannot decide the question; the next ones bypass the loss.

## 4. Probe 2: does a stiff-direction difference reach later predictions?

**Method.** Move one token's state by a small step, in position only, at a layer: along one of its own stiff directions, or along a random direction outside them. Read, as finite differences in double precision, the token after the step, its own prediction, and the next 64 tokens' predictions. The ratio of the two kinds of move, within one model, says whether stiff directions are special. OpenWebText, 16 windows.

| move at layer 0 | SR2 | SR3-crit | FO-b |
| --- | ---: | ---: | ---: |
| next 64 tokens: stiff over random | 1.02 | 0.97 | 0.99 |
| a stiff move kept along the stiff directions after the step | 3.13 | 2.30 | 1.37 |

**Verdict.** ST3 hit: the second-order step keeps a stiff move more than critical damping or a gradient flow. ST1 and ST2 (SR2's next-token ratio at least 1.5 times the others') missed. By the next tokens, a stiff difference and a random one change the predictions equally, in all three models.

## 5. Probe 3: whose force is it? Weight against force

**Question.** The second-order model loses more than the first-order one when wells are removed. Is that because one well carries most of its force (concentration), because it responds more to a given force change (sensitivity), or because of memory?

**Method.** Each well's force $f_k$ at every token, computed from the bank's own formula (the terms add up to the model's force to $10^{-15}$), and the cost of dropping each well alone. OpenWebText, layer 0.

| layer 0 | SR2 | SR3-crit | FO-b |
| --- | ---: | ---: | ---: |
| heaviest well: share of the weight; share of the force | 81%; 29% | 18%; 17% | 61%; 50% |
| cost per unit of force share, light wells (median) | 1.02 | 0.81 | 1.15 |

**Verdict.** FR1 hit (SR2's heaviest well carries less than 60% of the force). FR4 hit: measured against the share of force removed, not of weight, SR2 loses about half as much as FO-b, where against weight it lost 2.3 to 2.9 times as much. FR2 hit by the fitted slope but not per well (the light wells cost the same per unit of force in both). FR3 missed. **The removal gap follows the force:** SR2's light wells carry 71% of its force at 19% of the weight, so removing wells by weight removes more of its force.

## 6. Probe 4: stiffness and distance, well by well

**Question.** Why do SR2's light wells carry its force? A well's force is its weight, times its stiffness along the displacement, times its distance from the state. The original reading predicts stiffer wells in the second-order model.

**Method.** Each well's force split into $g_k$, an effective stiffness $s_k = \lVert P_k(\mu_k - h)\rVert / \lVert \mu_k - h \rVert$ and a distance $d_k = \lVert \mu_k - h \rVert$, averaged over tokens weighted by the force. Layer 0, on both corpora.

| layer 0 | SR2 | FO-b | SO-CODE | FO-b-CODE |
| --- | ---: | ---: | ---: | ---: |
| heaviest well: stiffness | 0.020 | 0.089 | 0.019 | 0.199 |
| heaviest well: occupancy (median) | 0.99 | 0.50 | 0.98 | 0.32 |
| heaviest well: share of the force | 29% | 50% | 35% | 81% |
| light wells: stiffness (geometric mean) | 0.068 | 0.095 | 0.101 | 0.119 |

**Verdict.** On OpenWebText SW2 (SR2's light wells at least 3 times stiffer than FO-b's) and SW3 missed; on code SC1 (FO-b-CODE's light wells softer) missed. **The first-order potential is not softer, on either corpus;** on code its dominant well is the stiffest in either model. SC2 and SC3 hit, and the configuration is the same on both corpora: the second-order model holds its state at the bottom of a soft basin that pulls little and steers it with light wells; the first-order model pulls its state with one stiff, distant well. The earlier "ten times softer" compared stiff-mode frequencies, which also depend on the mass, and the first-order model has none.

## 7. Probe 5: the same question at the recall positions, on code

**Method.** As probe 2, but on code and at the recall positions: move the state of the earlier occurrence $p$ and read the change of the recalled token's own prediction at $q$. 16 windows, 47 near and 48 far pairs per model.

| recall, change of the target's prediction | SO-CODE | FO-b-CODE |
| --- | ---: | ---: |
| layer 0, far: stiff over random | 1.32 | 1.32 |
| layer 0, near: stiff over random | 1.34 | 1.01 |
| a stiff move kept along the stiff directions (layer 0) | 2.70 to 2.85 | 2.09 to 2.12 |

**Verdict.** CT1 (SO-CODE's far ratio at least 1.5 times FO-b-CODE's) missed: equal. CT2 missed. CT3 hit: the rotation is real on code too. Where the memory gap was measured, stiff directions still do not carry the recall better than other directions.

## 8. Probe 6: which pathway carries the recall?

**Method.** On 16 fresh code windows, cut one pathway at a layer for every recall pair at once, and read the loss change at near and far recall positions against control positions (whose target does not occur earlier in the window).

| loss change, nats per token | SO-CODE far recall | SO-CODE control | FO-b-CODE far recall | FO-b-CODE control |
| --- | ---: | ---: | ---: | ---: |
| earlier occurrence removed from the layer-1 context channels | +0.794 | +0.000 | +0.740 | −0.000 |
| earlier occurrence removed from the layer-1 pair force | +0.000 | +0.000 | +0.000 | +0.000 |
| velocity into layer 1 set to zero | +0.001 | +0.094 | (none) | |

**Verdict.** LR1 hit (a replication on fresh windows): the second-order model's layer 1 passes a stiff-direction difference in the earlier occurrence on to the recall 2.6 times more strongly. LR2 missed: the velocity is not the memory; it matters for ordinary tokens, not recalled ones. LR3 missed narrowly (the second-order model relies more on the layer-1 read for far recall, by 1.5 standard errors). **Recall has one pathway, shared by both orders:** at layer 1 the recalling position reads the earlier occurrence through the content-routed context channels. The pair force carries none of it. Not registered, so a lead only: the second-order model relies more on layer 1 and the first-order model more on layer 0, as if second order retrieves later in the stack.

## 9. What stands and what is open

**Stands.**
- The observation: second order's advantage grows with recall.
- The trained arms: critical damping costs 8.5%; resetting the velocity costs nothing on TinyStories.
- The local rotation: the second-order step keeps a stiff-direction difference that a gradient flow contracts (probes 2, 5).
- The configuration: a soft basin with steering light wells against one stiff, distant well (probes 3, 4).
- One recall pathway, a content-addressed read at layer 1, in both orders (probe 6).

**Does not stand, in the form stated.**
- That the kept stiff-direction difference carries the recall (probes 2, 5).
- That the first-order potential is softer (probe 4).
- That the velocity carries the memory (probe 6).

**Open.** Which difference gives second order its advantage in recall: the configuration of the wells, the stronger layer-1 transmission, or where in the stack retrieval happens. All of this is one seed per model at two layers.

## 10. Files

Scripts in `notebooks/conservative_arch/scaleup/debug/`: `stiff_mode_damping_probe.py` (probe 1), `stiff_mode_transmission_probe.py` (probes 2 and 5, and the replication in 6), `well_force_response_probe.py` (probes 3 and 4), `recall_pathway_probe.py` (probe 6). Outputs in `notebooks/conservative_arch/scaleup/results/`: `damping_probe/`, `transmission_probe/`, `force_response/`, `recall_pathway/`. The registered calls, the checks and every verdict are in the [protocol, `p-bank-memory-conjecture`](Depth_Ladder_and_Matched_Baseline_Protocol.md#p-bank-memory-conjecture) and the entries after it.
