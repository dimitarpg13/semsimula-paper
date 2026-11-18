# §5.22 SY: is spontaneous synchronization at the root of semantic inference?

**What this note is.** A short, readable summary of §5.22 of the depth-ladder protocol, "SY: is spontaneous synchronization at the root of semantic inference in these models?", written 2026-11-17. It covers the synchronization test programme as the protocol ran it: the levels and calls pre-registered on 2026-10-11, before any measurement; how the instruments were fixed; what was scored, how and when (2026-10-13 to 2026-10-16); and what is pending. Every number, verdict and date is copied from the [protocol](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md), which keeps the full running log and remains the record of truth. The design, with the definition of every statistic and the tests SY0–SY6, is [Spontaneous_Synchronization_and_Semantic_Binding_Test_Programme.md](../Spontaneous_Synchronization_and_Semantic_Binding_Test_Programme.md).

**Status in one line.** By the pre-registered criteria the verdict is **semantically blind**: level 0 holds (the couplings do synchronize tokens) and level 1 fails (not along linguistic structure). Follow-ups found one structural regularity, noun-chunk grouping made through the shared context field $\xi$, and found that the matched GPT-2 forms that grouping too. No model binds reference.

## Contents

1. [Terms](#1-terms)
2. [The levels and the calls](#2-the-levels-and-the-calls)
3. [SY0: fixing the instruments](#3-sy0-fixing-the-instruments)
4. [Levels 0 and 1 on the five arms](#4-levels-0-and-1-on-the-five-arms)
5. [The follow-ups: context field, prediction geometry, decoding probe, subspaces](#5-the-follow-ups-context-field-prediction-geometry-decoding-probe-subspaces)
6. [SY5: does extra depth over-synchronize?](#6-sy5-does-extra-depth-over-synchronize)
7. [SY6: the matched GPT-2](#7-sy6-the-matched-gpt-2)
8. [The exchange-field arms and the readout point](#8-the-exchange-field-arms-and-the-readout-point)
9. [What is pending](#9-what-is-pending)

## 1. Terms

- **Synchronization** is alignment of token states produced by the couplings between tokens. It is measured against a **replay** of the same layer step with the coupling changed.
- **The common-mode reference** replaces a coupling's force on every token by its mean over tokens. What the coupling does differently to different tokens ("spec", on minus common) is what counts as synchronization. A shared push is not.
- **$\Gamma_{\mathrm{spec}}$** is the global pairwise contraction of the token set attributable to the coupling's token-specific action. Positive means the tokens drew together.
- **$B_{\mathrm{spec}}$** is the binding contrast: how much closer related tokens get than distance-matched unrelated ones, attributable to the coupling. Groupings: coreference chains, head–dependent pairs, noun chunks, named entities, part of speech.
- **General rules, fixed in advance.** The unit is the sequence: 200 sequences of 512 tokens, cluster bootstrap, 95% intervals. A contrast counts only if its interval excludes 0 and it is at least 0.02 (the **floor**). A level holds only if it holds in PMX and in at least one other arm with the same coupling route (**replication**). Primary statistics only decide levels.
- **The arms.** PMX, PM1, F3.1, G2 and SR2 (all $L = 2$), plus the four-layer Fock arm L4F. G3 and G3′ (exchange-field arms of Gen 3) were added on 2026-10-15. The matched GPT-2 is the comparison.

## 2. The levels and the calls

| level | what it asks | holds if (after the SY0 amendment) |
| --- | --- | --- |
| 0. exists | do the couplings contract the token set? | Γ_spec passes the floor at some layer for some coupling, replicated |
| 1. structured | along linguistic structure? | (a) V_φ attracts related pairs with selection enrichment ≥ 1.2, or (b) the wells' mode assignment gains NMI; and (c) B_spec passes the floor for the same grouping; all beat the shuffled-label control |
| 2. used | does the model use it? | targeted removal of related pairs hurts at least 1.5× more than random removal, plus a BLiMP drop |
| 3. root | is it the root? | level 2 holds, plus a λ-sweep peak and over-synchronization along Gate 2 |

The verdict is the highest level that holds: refuted (0 fails), semantically blind (1 fails), epiphenomenal (2 fails), load-bearing for binding (3 fails), or root (3 holds).

| call | prediction | called | result | scored |
| --- | --- | ---: | --- | --- |
| SY-L0 | level 0 holds | 75% | HIT | 2026-10-14 |
| SY-L1 | level 1 holds | 45% | MISS | 2026-10-14 |
| SY-L2 | level 2 holds | 30% | cannot hold (level 1 fails) | — |
| SY-L3 | level 3 holds | 10% | cannot hold (level 1 fails) | — |
| SY-pair | V_φ attracts coreferent or head–dependent pairs (level 1 (a) alone) | 50% | MISS (enrichment 0.97–1.15) | 2026-10-14 |
| SY-wells | the wells pass level 1 (b) and V_φ does not | 25% | MISS (neither passes) | 2026-10-14 |
| SY-G2 | Gate 2's blow-up comes with over-synchronization | 35% | HIT | 2026-10-15 |
| SY-GPT2 | GPT-2's binding contrast at least as large as the best Lagrangian arm's | 60% | MISS (0 of 3) | 2026-10-15 |

Every verdict is scoped to these models: $L = 2$ ($L = 4$ where read), 532.5M tokens, a third of compute-optimal.

## 3. SY0: fixing the instruments

The rule was that SY0 (toy checks with known answers) must pass before any model is measured, and a failure is fixed and re-run, never waived. The instruments as first specified failed, and passed on run 3 (2026-10-13):

- **Run 1.** Raw measures read a shared drift as synchronization. Centred coherence $R$ peaks at the Kuramoto threshold and falls as synchronization completes, so it cannot be used. Attribution against the switch-off replay counts a coupling's common push.
- **Run 2.** The common-mode reference was added. $\Gamma_{\mathrm{spec}}$ passed everything. The binding measures failed only one check, whose null turned out to be wrong.
- **Run 3.** With that check corrected, all admitted measures pass.

**Amendment, before any model was measured.** Level 0 uses $\Gamma_{\mathrm{spec}}$ in place of the coherence and co-movement gains; level 1 (c) uses $B_{\mathrm{spec}}$. Thresholds, the replication rule and the calls are unchanged. Annotation: spaCy `en_core_web_trf` 3.7.3 and fastcoref; BLiMP downloaded, with the sets for level 2 fixed in advance.

## 4. Levels 0 and 1 on the five arms

Scored 2026-10-14. All replays reproduce the trained steps bit for bit.

| arm | V_φ, layer 0 | V_φ, layer 1 | wells / Fock, layer 0 | wells / Fock, layer 1 |
| --- | ---: | ---: | ---: | ---: |
| PMX | +0.003 | +0.021 | wells −0.207 | wells +0.983 |
| PM1 | +0.007 | +0.007 | wells −0.039 | wells +1.156 |
| F3.1 | +0.004 | +0.064 | | |
| G2 | +0.001 | 0.000 | Fock −0.065 | Fock −0.099 |
| SR2 | +0.003 | 0.000 | | |

- **Level 0 holds on the $V_\phi$ route**, a pairwise interaction (PMX and F3.1, layer 1). The wells' contraction is about 50 times larger, but SY0 showed that a fixed external attractor reads the same way, so it is consistent with tokens converging on the wells' fixed centres rather than on each other. The Fock channel expands the token set.
- **Level 1 fails in every arm.** $V_\phi$ does not choose related sources (enrichment 0.97–1.15 against the 1.2 required). The wells' mode co-assignment gain is null. $B_{\mathrm{spec}}$ for $V_\phi$ is at most 0.005 in size.
- **A replicated signal the rule does not admit** (descriptive). At layer 1 the wells bind noun-chunk members (PMX +0.044, PM1 +0.026) and separate head–dependent pairs, with null shuffled controls. It fails level 1 only because criterion (b) is null.

## 5. The follow-ups: context field, prediction geometry, decoding probe, subspaces

**A pattern common to every arm.** In the states themselves, noun-chunk members end up closer and head–dependent pairs further apart (final-output chunk binding +0.032 to +0.047). In arms without wells, neither $V_\phi$ nor the Fock channel makes it. Depth (L4F) does not turn the verdict around. Two readings were separated by tests whose criteria were fixed on 2026-10-14, before they ran. They are follow-ups, so they do not rescore level 1.

| test | what it asks | result (2026-10-14) |
| --- | --- | --- |
| A: binding through ξ | is the pattern made through the shared context field? | HOLDS: PMX, replicated in PM1, for chunks and dependencies |
| B: prediction geometry | is it just tokens predicting similar continuations? | CONTRADICTED for noun chunks (residual passes in all six arms); for dependencies, prediction geometry accounts for it |
| decoding probe | can a query–key probe read the antecedent of a pronoun? | no arm passes (no layer beats recency by more than 4 points) |
| binding in a subspace | is binding hidden in a subspace? | the one coreference hit is lexical: the same word repeated, not reference |

**Reading.** The one structural regularity the dynamics create in every arm is binding of syntactic constituents through $\xi$, a mean-field coupling, not a pairwise one. The separation of heads from dependents is mostly prediction geometry. Coreference is not bound in either sense, at $L = 2$ or $L = 4$.

## 6. SY5: does extra depth over-synchronize?

The criterion was amended on 2026-10-15, before it was built, because the original used the rejected coherence $R$. Over-synchronization holds in an arm if, from 2 to 3 steps at the trained step size (Gate 2), the final-state contraction $S$ is at least 0.05 or the effective rank falls by at least 20%, while perplexity rises.

| arm | PPL at L → first extra step | S | effective rank | over-synchronization |
| --- | --- | ---: | ---: | --- |
| PMX | 50.4 → 58.2 (+15%) | +0.054 | −7.5% | holds |
| F3.1 | 53.2 → 90.1 (+69%) | +0.058 | −1.8% | holds |
| G2 | 48.5 → 2,534 (+5,122%) | +0.473 | −3.6% | holds |
| PM1 | 51.1 → 75.7 (+48%) | +0.022 | −14.8% | does not |
| SR2 | 53.3 → 109.8 (+106%) | +0.029 | +3.6% | does not |

**SY-G2 HIT** (PMX, replicated in F3.1 and G2). Over-synchronization accompanies the large failures of extra depth, but it does not account for their onset: SR2, PM1 and L4F fail at the first extra step with no over-synchronization. The evidence is co-occurrence, not a causal test.

## 7. SY6: the matched GPT-2

**SY-GPT2 MISS, 0 of 3** (2026-10-15). The call was made precise before SY6 was built: GPT-2's attention-made binding at its last layer against the largest coupling-made binding at the last layer of any Lagrangian arm.

| grouping | GPT-2 attention, last layer | best Lagrangian coupling |
| --- | ---: | ---: |
| coreference | +0.003 | +0.029 (PM1, ξ) |
| head–dependent | −0.002 | +0.003 (G2, ξ) |
| noun chunks | −0.004 | +0.064 (G2, ξ) |

- **The grouping of constituents is generic.** GPT-2's states form it as strongly (+0.054 against +0.032 to +0.047), and it survives the prediction control there too.
- **The mechanism and its timing differ.** Attention builds it gradually through the middle layers (2–5). In the Lagrangian arms the context field makes it at the last step.
- **GPT-2 also fails the coreference probe** (final minus embeddings +0.084, interval including 0). Its middle layers carry more antecedent information (0.64 against at most 0.58), but "reference is not bound" is not, on this evidence, specific to the framework.

## 8. The exchange-field arms and the readout point

G3 and G3′ were added on 2026-10-15, before any code was written. They do not rescore the earlier calls. The exchange route is scored as a follow-up, with replication in both arms.

- **Level 1 (c) fails in both.** The field moves tokens strongly but binds no grouping.
- **Level 0 FAILS**, decided by a reference test fixed in advance. As first read it held by the displacement reference (layer 0 +0.056 and +0.112) but not by the force reference. A toy check (check 6) showed that with a LayerNorm before the readout neither reference isolates the token-specific part. Read before any normalization, the field contracts at no layer of either arm (layer 0: $-0.007$ in G3, $-0.013$ in G3′). The layer-0 "contraction" was the first LayerNorm acting on the field's shared push.
- **The instrument from then on.** In arms with a LayerNorm inside the step, $\Gamma_{\mathrm{spec}}$ is read at the integrator's output, before any normalization, and counts only if the force and specific-only readings agree within 0.01. No earlier call changes.
- **What these arms bind, they bind through $\xi$**, more strongly than any other arm (noun chunks +0.054 in G3, +0.074 in G3′). Extra depth over-synchronizes in both from the first extra step.
- **A loader bug** (found 2026-10-16) had reset G3′'s trained routing temperatures in every local replay. G3′ was re-measured; no conclusion changes, and the protocol's G3′ numbers are the corrected ones.

## 9. What is pending

- **Levels 2 and 3.** §5.22 records no level-2 or level-3 measurement; by the rule they cannot hold when level 1 fails. SY4 (the $\lambda$ sweep) is part of level 3.
- **The wells' chunk pattern.** Whether criterion (b) is too coarse a readout for it is left as a follow-up question, not a reason to rescore. Whether the wells act through $\xi$ or alongside it is not separated.
- **Test-set size.** With 48 test sequences the probe's intervals are about $\pm 0.1$, so differences of a few points between GPT-2 and the Lagrangian arms are not resolvable.
- **XRB-F**, the exchange-field-only model, was run through the eight SY scripts on 2026-10-18 (descriptive). Its results are in §5.25; see [Exchange_Field_Routes_XR.md](Exchange_Field_Routes_XR.md).
