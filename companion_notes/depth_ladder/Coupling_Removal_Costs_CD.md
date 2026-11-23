# §5.20 CD: what does each coupling cost when it is removed? (with §5.21 VF)

**What this note is.** A short, readable summary of §5.20 of the depth-ladder protocol, "CD: one coupling takes over — what does each mechanism cost when it is removed?", with a brief account of §5.21, "VF: the trained-without factorial". Written 2026-11-17. CD was pre-registered on 2026-10-11, before any measurement, and scored the same day; it was extended descriptively to PMX (2026-10-11) and to the exchange-field arms (2026-10-18). VF was pre-registered on 2026-10-11 and has not been scored. Every number, verdict and date is copied from the [protocol](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md), which keeps the full running log and remains the record of truth.

**Status in one line.** CD scored 3 of 6: the pair potential $V_\phi$ is not "dormant" beside any competitor by the fixed rule, the Fock channel dominates its arm, and in PM1 the wells coexist with $V_\phi$. Across all arms the couplings fall into a substitution order: register read, then exchange field, then $V_\phi$.

## Contents

1. [Terms and method](#1-terms-and-method)
2. [The calls and the removal costs](#2-the-calls-and-the-removal-costs)
3. [What it changes](#3-what-it-changes)
4. [Extensions: PMX and the exchange-field arms](#4-extensions-pmx-and-the-exchange-field-arms)
5. [§5.21 VF: the trained-without factorial](#5-521-vf-the-trained-without-factorial)

## 1. Terms and method

- **The couplings** are the routes by which tokens act on one another. $V_\phi$ is a pairwise potential between tokens. The **Fock channel** (register read) carries information through a bank of registers. The **wells** are PM1's and PMX's Poisson-mode wells. The **exchange field** is an attention-shaped potential.
- **The arms.** F3.1 has $V_\phi$ alone. SR2 is F3.1 with an exact damped flow for the stiff part of the potential. G2 has $V_\phi$ and the Fock channel. PM1 has $V_\phi$ and explicit wells; PMX integrates its wells exactly.
- **Why a common unit.** The Cell 6b-9 attributions showed $V_\phi$ carrying the part of the step not due to $V_\theta$ only in F3.1 ($-0.291$) and moving it by less than 0.02 wherever it has a competitor. But 6b-9 measures displacement, not importance, and different mechanisms' shares are in different units.
- **The removal cost** of a term is $c = \ln(\mathrm{PPL_{removed}} / \mathrm{PPL_{full}})$ in nats, with a paired standard error over Cell 6b-7's 12 validation batches (each 4 sequences of 512 tokens). The term is switched off at inference on the trained weights: no training.
- **Readings fixed in advance.** $V_\phi$ is **dormant** in an arm if its cost is below 0.01 nats (1% PPL), and **load-bearing** if it is at least 0.05.
- **Checks.** All switches restore the logits bit for bit, and the patched potential is called on every layer step.

## 2. The calls and the removal costs

| call | prediction | called | result |
| --- | --- | ---: | --- |
| CD1 | F3.1: V_φ load-bearing, c ≥ 0.05 | 85% | HIT (0.282) |
| CD2 | G2: V_φ dormant, c < 0.01 | 70% | MISS (0.037) |
| CD3 | SR2: V_φ dormant, c < 0.01 | 65% | MISS, narrowly (0.0135) |
| CD4 | PM1: V_φ dormant, c < 0.01 | 55% | MISS (0.324) |
| CD5 | G2: c(Fock off) > c(V_φ off in F3.1) | 65% | HIT (1.486) |
| CD6 | PM1: c(wells off) > c(V_φ off in F3.1) | 55% | HIT (0.775) |

| arm | removal | c (nats) | PPL | rise | 6b-9 share of the same term |
| --- | --- | ---: | ---: | ---: | ---: |
| F3.1 | V_φ | +0.282 ± 0.013 | 56.30 → 74.64 | +33% | −0.291 |
| SR2 | V_φ | +0.0135 ± 0.0010 | 57.59 → 58.38 | +1.4% | −0.002 |
| G2 | V_φ | +0.037 ± 0.008 | 51.99 → 53.92 | +3.7% | −0.0002 |
| G2 | Fock channel | +1.486 ± 0.076 | → 229.67 | +342% | (η 1.6–3.1) |
| G2 | both | +1.524 ± 0.081 | → 238.75 | +359% | |
| PM1 | V_φ | +0.324 ± 0.008 | 53.71 → 74.28 | +38% | −0.017 |
| PM1 | wells | +0.775 ± 0.052 | → 116.53 | +117% | −0.258 |
| PM1 | both | +0.956 ± 0.061 | → 139.77 | +160% | |

In G2, $V_\phi$ is not dormant but **subordinate**: 8 times below its F3.1 cost and 40 times below the Fock channel's. In PM1 it costs more than in F3.1. That **withdraws** the earlier reading "in PM1 the wells have taken over $V_\phi$'s role": the wells carry the step's displacement, but the trained model relies on $V_\phi$ at least as much as F3.1 does.

## 3. What it changes

1. **Displacement is not reliance.** 6b-9 shares of 0.0002 and 0.017 hid costs of 3.7% and 38%. No 6b-9 share is to be quoted as importance again without a removal cost beside it.
2. **The Fock arm fits the exclusivity conjecture; the wells arm does not.** In G2 one coupling carries the model, and the two costs add (1.524 against 1.523). In PM1 both couplings are load-bearing and overlap by 13% (0.956 against 1.099).
3. **A partial order in the removal unit:** Fock channel (1.49), wells (0.77), $V_\phi$ uncontested (0.28–0.32), $V_\phi$ beside Fock (0.037), $V_\phi$ under SR2 (0.014).
4. **Reliance overstates value.** The gains of training with a term are much smaller than its removal costs: G2 against F3.1 0.084 nats (the Fock channel), PM1 against F3.1 0.046 (the wells), against removal costs of 1.49 and 0.77. A removal cost says what this network routes through a term, not what a model trained without it would lose.

**Scope.** Removal at inference cannot test path-dependence (the CB series' job, §5.10). One seed per arm; $L = 2$; 532.5M tokens.

## 4. Extensions: PMX and the exchange-field arms

**PMX** (2026-10-11, descriptive, after CD was scored). Full model 53.24. Removing $V_\phi$ costs 0.0145 nats (+1.5%), the wells 1.063 (+190%), both 1.088. With the wells integrated exactly, $V_\phi$ is as marginal as in SR2. So "the wells and $V_\phi$ are both load-bearing" belongs to the explicit integrator. What makes $V_\phi$ marginal is the exact flow (SR2 alone already shows it), not the wells.

**The exchange-field arms** (2026-10-18, descriptive, not pre-registered). The author asked whether $V_\phi$ sits in the background when the exchange field takes over, and whether the couplings have an order of strength. The same script was run on XRB-F (exchange field, no register read), G3′ and G3, with one more switch (the field's gate set to 0). All 20 checks were exact.

| arm | couplings | V_φ | exchange field | register read | field + read | all three |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| F3.1 | V_φ | 0.282 | — | — | — | — |
| XRB-F | V_φ, field | 0.041 | 1.372 | — | — | — |
| G2 | V_φ, read | 0.037 | — | 1.486 | — | — |
| G3′ | V_φ, field, read | 0.031 | 0.569 | 1.351 | 1.806 | 1.815 |
| G3 | V_φ, field, read (unhardened) | 0.011 | 0.613 | 1.550 | 1.735 | 1.750 |
| SR2 | V_φ (exact stiff-mode flow) | 0.014 | — | — | — | — |
| PM1 | V_φ, wells | 0.324 | wells 0.775 | — | — | — |

- **$V_\phi$ steps back beside either token route**, by about 7 times either way (0.282 alone, 0.041 beside the field, 0.037 beside the read). It does not vanish: 0.01–0.04 nats in every arm with another coupling.
- **The read crowds out the field; the field barely touches the read.** The field's cost falls from 1.372 alone to 0.57–0.61 beside the read. The read's moves from 1.486 alone to 1.35 (G3′) or 1.55 (G3).
- **So the order is register read, then exchange field, then $V_\phi$.** Each route is crowded out by the ones above it and barely affected by those below. The value of training with each (§5.25) gives the same order: the read alone 8.0% better than F3.1, the field alone 6.0%.
- **Not quite additive.** Removing field and read together costs 6% less than the sum in G3′ and 20% less in G3.
- **The wells are different.** Beside them $V_\phi$ keeps all its reliance (0.324). Wells against the read and the field stay unranked, since no arm carries both.

## 5. §5.21 VF: the trained-without factorial

**The question.** CD measures reliance. It cannot say whether the wells enhance $V_\phi$, or whether the Fock channel substitutes for it in training. Models trained without $V_\phi$ can. VF was pre-registered on 2026-10-11, before any run, on F3.1's explicit base (not SR2).

| | with V_φ (measured) | without V_φ (to train) |
| --- | ---: | --- |
| no other coupling | F3.1 57.76 | VF0 |
| wells | PM1 55.17 | VF1 |
| Fock | G2 53.12 | VF2 |

$V_\phi$'s value in each setting is $v = \ln(\mathrm{PPL_{without}} / \mathrm{PPL_{with}})$, on settled PPL.

| call | prediction | called |
| --- | --- | ---: |
| VF-a | V_φ is worth something uncontested: v₀ ≥ 0.03 | 70% |
| VF-b | the Fock channel substitutes in training: v(Fock) ≤ 0.01 | 55% |
| VF-c | the wells do not substitute: v(wells) ≥ v₀ − 0.01 | 55% |
| VF-d | the wells enhance V_φ: v(wells) > v₀ + 0.02 | 30% |

**Status.** No VF arm is scored in the protocol. VF0 is set up and checked, and the author decided on 2026-11-22 to run it: [VF0_F31_Without_Pair_Force.md](VF0_F31_Without_Pair_Force.md). VF1 first needs a switch check (`PAIR_POTENTIAL = 'none'` together with the wells), and VF0 and VF1 come before VF2, behind PMX's scoring. Until VF1 is scored, the book may say only "the wells work beside $V_\phi$".
