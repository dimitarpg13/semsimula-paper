# The v6 abstract-gating runs: G1 to G4, and the CB series

**What this note is.** A short, readable summary, written 2026-11-17, of the runs that decide three claims in the v6 abstract of the book: what the Fock register mechanism is worth when every channel gets gradient, whether depth helps under the same condition, and whether parity with the matched GPT-2 survives a parameter-matched baseline. These are protocol §5.10 (the abstract-gating runs G1 to G4, called AG1 to AG4 in the book) and, briefly, protocol §5.11 (CB1 to CB3, balancing the conservative and Fock paths). For each arm it gives the configuration, what was registered before the run, the mid-run readings where there were any, the scoring and the diagnostics. Every number, verdict and date is copied from the [protocol](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md), which keeps the full log and remains the record of truth.

**Status in one line.** G2, G3 and the hardened follow-up arm G3′ have run and are scored. G1 (a parameter-matched GPT-2) was deferred and then dropped (2026-11-27), and no G4 result or CB-arm result is recorded in the protocol; CB0, a free reading on G2's checkpoint, is scored.

## Contents

1. [Terms used](#1-terms-used)
2. [The plan (protocol §5.10)](#2-the-plan-protocol-510)
3. [G1: a parameter-matched GPT-2 (protocol §5.10)](#3-g1-a-parameter-matched-gpt-2-protocol-510)
4. [G2: the Fock arm without an exchange field (protocol §5.10)](#4-g2-the-fock-arm-without-an-exchange-field-protocol-510)
5. [G3: the Fock arm with the conservative exchange field (protocol §5.10)](#5-g3-the-fock-arm-with-the-conservative-exchange-field-protocol-510)
6. [G3′: G3 with hardened routing (protocol §5.10)](#6-g3-g3-with-hardened-routing-protocol-510)
7. [G4: a forecastability check (protocol §5.10)](#7-g4-a-forecastability-check-protocol-510)
8. [CB1 to CB3: balancing the two paths (protocol §5.11)](#8-cb1-to-cb3-balancing-the-two-paths-protocol-511)
9. [Unclear items](#9-unclear-items)

---

## 1. Terms used

- **Fock register path.** A bank of registers that the model writes to and reads from. The **reverse channel** (`REVERSE_CHANNEL = True`) feeds register content back into the token states. Its contribution to a layer step, the **register (Fock) increment**, is not the gradient of a potential, so it breaks strict conservativity.
- **Exchange field.** A token-to-token coupling. `LADDER_MECHANISM = 'none'` has none; `'attention_potential'` has a conservative one, a force derived from a potential with attention-like routing.
- **$V_\varphi$ and $\xi$.** $V_\varphi$ is PARF's pair potential; $\xi$ are the context channels.
- **Live gradients.** In the Gen 2 models some paths had their gradient cut (detached). "Live" opens the gradient and leaves the forward computation unchanged (see [Gradient_Starvation_Investigation.md](../Gradient_Starvation_Investigation.md)).
- **F3.1** is the $L = 2$ conservative-only model with live $V_\varphi$ and $\xi$ and no register path: settled 57.76.
- **Settled** is the mean of the last three evaluations. All runs use the same notebooks, data and 32,500-step schedule as the published arms.
- **6b readings** are the diagnostic cells of the ladder notebook. 6b-6 switches the exchange field off. 6b-7 runs three gates against the trained perplexity: Gate 1 resets the velocity entering each layer, Gate 2 takes more steps at the trained step size, Gate 3 takes more steps at fixed total time (refinement, here 1.5 times the steps). 6b-8 ablates at inference (reverse channel off; register bank frozen at initialisation). 6b-9 (CG1) attributes the step to each force; its R(geo) is the distance from the $V_\theta$-only geodesic. 6b-12 reads how strongly tokens are forced. 6b-13 reads $\theta = \omega \Delta t$, how far a stiff mode rotates in one layer step.
- **SCAF** is the causality audit run during training: "CLEAN" means a perturbation of future tokens changes nothing.

## 2. The plan (protocol §5.10)

**Pre-registered 2026-10-03, before any run.** Each run decides one abstract claim:

| run | what it decides | the comparison the abstract quotes |
| --- | --- | --- |
| G1 | parity with GPT-2 on a parameter-matched footing | both GPT-2 baselines, with parameter counts |
| G2 | the register mechanism's value, and depth, under live gradients | G2 against F3.1; the L=4 live arm against G2 |
| G3 | the conservative exchange field's value under live gradients | G3 against G2 |
| G4 | a forecastability sentence only | — |

The abstract's wording for each outcome was fixed in advance, so the results fill in numbers rather than choose the story. Each claim carries one seed and an untuned baseline in the same sentence.

**Order (author's call, 2026-10-03).** G2 first (about 12 h), with G4 alongside. G3 next or in parallel (about 13 to 14 h). G1 after all three.

## 3. G1: a parameter-matched GPT-2 (protocol §5.10)

**Registered 2026-10-03, deferred until G2 to G4 are in.** **Dropped by the author 2026-11-27:** a 22-layer GPT-2 is not a realistic comparison; any further transformer baseline stays at 8 to 16 layers.

- **Configuration.** The matched GPT-2 notebook with `N_LAYERS = 22` and `TIE_EMBEDDINGS = False`; $d = 384$, 6 heads, learning rate 6e-4 with a cosine to 6e-5, 16,384 tokens per step.
- **Parameters:** 77,832,960, 1.3% above the $L = 4$ live arm's 76,823,510. $L = 22$ was chosen over $L = 21$ so that the baseline is never the smaller model.
- **Width is held at $d = 384$ by design.** A wider GPT-2 ($d = 512$, $L = 8$, 76.9M) was rejected: it would compare across semantic spaces.
- **Author's reservation.** Matching parameters by stacking depth ignores the model's dynamics. The primary comparison stays the width-matched 33.7M baseline, with both parameter counts stated.
- **Prediction:** settled 45, band 41 to 48. Settled at or below 47.6 (5% below 50.10) means parity does not hold on a parameter-matched footing. Called at about 65% that it does not hold.

**Result:** never run. The abstract quotes the width-matched baseline (49.81, 33.7M) against the $L = 4$ live arm (50.10, 76.8M). The GPT-2 baselines themselves are in [Matched_GPT2_Baselines.md](Matched_GPT2_Baselines.md).

## 4. G2: the Fock arm without an exchange field (protocol §5.10)

**Configuration.** $L = 2$, `LADDER_MECHANISM = 'none'`, reverse channel on, $V_\varphi$ and $\xi$ live. Its Gen 2 twin is the published no-exchange arm (66.98).

**Pre-registered calls.**

| call | threshold | called |
| --- | --- | ---: |
| point and band | settled 53, band 50–57 | — |
| key line 1: the register mechanism under live gradients | settled below 57.76 (F3.1) | about 80% |
| key line 2: depth under live gradients | settled above 50.10 (the L=4 live arm) | about 75% |
| step-3,000 criterion | at or below 137.1 (Gen 2 twin: 140.61) | — |
| CG1 forecast | V_φ share below 0.05 in size (above 0.15 refutes the substitution account) | — |

**Scored 2026-10-04** ([protocol, `p-g2-ladder`](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-g2-ladder)).

| measure | value |
| --- | --- |
| settled (last three evals: 53.01, 53.29, 53.07) | **53.12** |
| best | 51.27 at step 31,000 |
| parameters | 76,770,256, identical to the Gen 2 twin |
| speed | 1.64 s/step |
| SCAF audits | CLEAN at all seven |

- **Point and band: HIT, on the point.**
- **Step-3,000 criterion: HIT.** 123.59, 12.1% below the Gen 2 twin.
- **Key line 1: YES.** 53.12 against F3.1's 57.76, 8.0% better.
- **Key line 2: YES.** The $L = 4$ live arm is 5.7% better, at 1.9 times the cost per step. The protocol calls this the first clean depth answer: one mechanism, one convention.
- **CG1 forecast: HIT.** $V_\varphi$'s share is $-0.0002$. Amended 2026-10-11 (§5.20): removed at inference, $V_\varphi$ costs G2 0.037 nats, so it is "subordinate, not inert".
- **Against the GPT-2:** 1.067 times the matched GPT-2 (49.81). G2 is below 56.6, so the CB series is live (section 8).

**The 6b readings** (best checkpoint, step 31,000):

| reading | F3.1 | **G2** | L=4 Fock live |
| --- | ---: | ---: | ---: |
| CG1 V_φ share | −0.291 | **−0.0002** | −0.0035 |
| CG1 R(geo) | 0.665 | 0.911 | 1.369 |
| Gate 1, velocity reset | +34.3% | **+32.7%** | +55% |
| Gate 2, 1.5× steps at the trained Δt | +92% | +50.7% (N=3) | +41% |
| Gate 3, 1.5× steps at fixed T | +143% | **+1,274%** (N=3) | +216% (N=6) |
| CG7 ω·Δt median | 4.32 | **3.40**, in the L=2 band [3.3, 4.2]: HIT | 2.35 |
| 6b-8, reverse channel off | — | **+300%** | — |
| 6b-8, bank frozen at init | — | +24.0% | — |

**Two findings.**

1. **The momentum-refinement rank order breaks.** G2 relies on velocity slightly less than F3.1, yet fails refinement nine times worse. The register path drives refinement failure independently of momentum.
2. **Ablation overstates trained value.** Switching the reverse channel off costs +300%, yet the model trained without it (F3.1) is only 8.7% worse. The book's methodological sentence gains a live-gradient row: 4.00 times against 1.087 times.

## 5. G3: the Fock arm with the conservative exchange field (protocol §5.10)

**Configuration.** G2's settings with `LADDER_MECHANISM = 'attention_potential'` and the field's gradient live as well. Its partial-live parent is the published `rglive` model (61.11).

**Pre-registered calls.** Settled 51, band 46 to 56. **Key line:** settled more than 2% below G2 (at most 52.1), called at about 60%.

**Mid-run diagnosis, 2026-10-04, step 21,700, recorded before the result.**

| | G3 | G2 | parent probe |
| --- | --- | --- | --- |
| PPL, steps 3,000–8,000 | led G2 by 5–7 PPL | — | — |
| PPL, step 21,500 | 64.18; crossed G2 near 13,000, trails by about 1.6 | 62.61 | G3 a steady −11% against it |
| gradient norm, constant-LR phase | rose 0.43 → 0.88 | flat at about 0.35 | rose 0.42 → 0.64 |
| clip hits | 5.9% of steps in the last window | 0 | 0 |

- The growth is the exchange field's: its gradient alone rises from about 0.3 to about 0.8.
- **Hypothesis:** the field's routing is unnormalised (free query and key projections $W_q$, $W_k$), and in Gen 2 it never received gradient, so it never needed the QK-norm hardening the creation gate got. Under live gradients its logits grow.
- **Projection:** about 54.4, missing the key line. This led to the follow-up arm G3′ (section 6).

**Scored 2026-10-05** ([protocol, `p-g3-ladder`](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-g3-ladder)).

| measure | value |
| --- | --- |
| settled (last three evals: 54.17, 54.35, 54.12) | **54.21** |
| best | 52.44 at step 31,000 |
| speed | 1.68 s/step |
| clip hits | 9 of 650 logged steps (1.4%), max norm 1.14 |
| SCAF audits | CLEAN at all seven |

- **Point and band:** in the band; the point missed by 3.2, on the bad side.
- **Key line: NO.** G3 is 2.1% worse than G2, the same model without the field.
- Against its parent (61.11) 11.3% better, against F3.1 6.1% better, against GPT-2 1.088 times.

**Diagnostics, 2026-10-05.**

- **Causality check on the final weights: CLEAN.**
- **6b-6: the field is load-bearing.** Switching it off costs +29.46 PPL (+56%), yet the model trained without it (G2) is 2.1% better. The field substitutes for work done elsewhere.
- **Routing probe** (step 500 against step 31,000): the $W_q$ and $W_k$ spectral norms grew from about 0.9–1.4 to 5.0–6.3, and the attainable logit scale about 28 times. Layer-1 routing entropy stays near 0.6 of uniform: it sharpened but did not collapse. The field carries about 92% of layer 1's conservative force. The logit-growth hypothesis is confirmed in the weights.
- **Where the gradient grows:** the field's offline gradient grows 12 times from step 500 to best, almost all of it in the value projection $W_v$ (24 times). QK-norm acts on $W_q$ and $W_k$, not on $W_v$.
- **6b-7:** Gate 1 +49%, Gate 2 at N = 3 +325%, Gate 3 at N = 3 +1,342%.
- **6b-13:** $\theta$ median 2.07, outside the registered band [3.3, 4.2]: **MISS**. It does not cross $\pi$, yet Gate 3 is as bad as G2's, so the $\pi$ crossing does not explain refinement failure here. The SR-π.1 call drops from 60% to about 40%.
- **6b-8:** reverse channel off +393.7%; bank frozen +16.7%.

## 6. G3′: G3 with hardened routing (protocol §5.10)

**Pre-registered 2026-10-04, inside G3's mid-run reading.** G3's configuration plus QK-normalised routing for the exchange field with a clamped learnable logit scale (the creation gate's `cgqk` scheme), and its own gradient-clip group at 0.3. Calls: the field's gradient norm stays within 25% of its step-5,000 value through the stable phase (70%); settled at most 52.1, so the hardened field adds value (50%). It does not gate the abstract.

**Implementation and verification, 2026-10-05.**

- **Code:** q and k are L2-normalised and scaled by a per-head $\sigma_h = \min(e^{\lambda_h}, 100)$, with $\lambda_h$ initialised at $\log(1/0.07)$. Switches `RELAX_ATTN_QK_NORM` (tag `rfqk`) and `RELAX_FIELD_CLIP` (tag `rfclip0p3`).
- **Off:** bit-identical to G3's configuration (logits, loss and all 95 gradients).
- **On:** all six checks pass: the tag, one new parameter per head, the score bound, gradients reach $\sigma$, $W_q$ and $W_k$, causality, and only the five field parameters change clip group.
- **Watch item:** $W_v$ carries most of the field's gradient, so the 0.3 clip will be set by $W_v$.

**Mid-run reading, 2026-10-05, step 14,500.**

| step | G3′ | G3 | G2 | G3′ / G3 | G3′ / G2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3,000 | 113.52 | 118.14 | 123.59 | 0.961 | 0.919 |
| 6,000 | 84.28 | 87.51 | 93.11 | 0.963 | 0.905 |
| 10,000 | 71.98 | 74.22 | 76.26 | 0.970 | 0.944 |
| 14,500 | 67.51 | 69.50 | 69.14 | 0.971 | 0.976 |

G3′ held a steady 3–4% lead over G3, while its lead over G2 was shrinking. Projection: 52.4 to 52.7, missing 52.1 narrowly. The gradient call was trending to a miss.

**Scored 2026-10-06** ([protocol, `p-g3p`](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-g3p)).

| measure | value |
| --- | --- |
| settled (last three evals: 52.83, 53.05, 52.81) | **52.90** |
| best | 50.97 at step 31,000 |
| SCAF | CLEAN at all seven audits |
| against G2 (53.12) | −0.4%, within eval noise |
| against G3 (54.21) | −2.4% |

- **Settled at most 52.1: MISS.** The hardening recovers G3's loss against G2, but the field then adds nothing measurable.
- **Gradient within 25%: MISS.** The field's norm rises from 0.43 to 0.82, about +90%: QK-norm caps the logits but not $W_v$'s growth.

**Diagnostics, 2026-10-06** (best checkpoint):

| measure | G3′ | G3 | G2 | F3.1 | L=4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6b-6, exchange field off | +26.0 PPL (+51%) | +29.5 (+56%) | — | — | — |
| Gate 1, velocity reset | +34% | +49% | +33% | +34% | +55% |
| Gate 2, N = 3 at fixed Δt | +441% | +325% | +51% | +92% | +41% |
| **Gate 3, 1.5× refinement** | **+115%** | +1,342% | +1,274% | +143% | +216% |
| 6b-8, reverse channel off | +291% | +394% | +300% | — | — |
| 6b-8, bank frozen at init | **+2,280%** | +16.7% | +24.0% | — | — |
| 6b-13, θ median: layer 0 / layer 1 / pooled | 2.01 / 3.40 / 2.29 | 1.91 / 3.00 / 2.07 | 2.94 / 4.62 / 3.40 | — | — |

- **The headline is refinement.** G3′ is the most refinement-ready model in the programme at that date. Its layer-1 $\theta$ still crosses $\pi$, one more case against the $\pi$ account.
- **It is a different solution.** Freezing the register bank costs +2,280%: G3′ relies on accumulated register content far more than its siblings.
- **One seed.** The routing hardening is the only difference from G3, but the result may not survive replication.
- **Causality check: CLEAN** (prefix-only leak tax $-1.4 \times 10^{-4}$ nats). The per-head logit scales settled at 11.6 to 21.7; no head approached the ceiling of 100.
- **SR-π.4b, registered before the download (55%): MISS.** G3′'s layer-0 register increment is 7.8 times the state, as large as G2's and G3's. Increment size is ruled out as the cause of the Fock arms' refinement failure.

**Correction, 2026-10-16 (protocol §5.22).** A local model builder reset G3′'s trained routing temperatures on loading, so every local replay of G3′ used untrained ones. The Colab numbers (settled 52.90, Gate 3 +115%) were unaffected. The affected local measurements were re-measured; **no verdict changes**.

**Later.** Protocol §5.18 (RR) asks what makes G3′ refinement-ready and registers a second seed of G3′ and of G2 (calls S1 to S3). No result of the second seeds is recorded in the protocol.

## 7. G4: a forecastability check (protocol §5.10)

Cell 6b-10 on run 4, the Gen 2 twin of the $L = 4$ live arm. Prediction: tangential coherence below +0.526, and the integrator-velocity forecast error above 0.759. If run 4 is also forecastable, the property belongs to the $L = 4$ architecture. **Result:** none recorded in the protocol.

## 8. CB1 to CB3: balancing the two paths (protocol §5.11)

**Pre-registered 2026-10-03, before any run.** The question (author's hypothesis): is $V_\varphi$ starved when the register path is present, and can a model keep most of the Fock gain while staying mostly conservative?

**The measure.** $\eta$, per token and layer, is the size of the register increment divided by the size of the conservative step. It is an upper bound on the non-conservative share of the step. On the Gen 2 no-exchange model $\eta$ averaged 1.62 at layer 0 and 3.05 at layer 1.

| arm | switch | tag | what it tests |
| --- | --- | --- | --- |
| CB1 | `REVERSE_CHANNEL_WARMUP_STEPS = 20000` | `rcw20000` | Fock arrives late: does V_φ stay awake? |
| CB2a | `FOCK_BUDGET = 1.0` | `fb1` | Fock increment no larger than the conservative step |
| CB2b | `FOCK_BUDGET = 0.3` | `fb0p3` | mostly conservative: the key arm |
| CB3 | `FOCK_GATE_L1 = 0.02` | `fg0p02` | a learned per-token gate with an L1 penalty |

**Success criterion.** At least one arm has mean $\eta$ at most 0.3 at both layers and keeps at least half of the gap between F3.1 and G2. **Stop rule:** if G2 settles at 56.6 or above, the series does not run. G2 settled at 53.12, so the series is live.

**CB0 scored, 2026-10-05** (G2's checkpoint, CPU). $\eta$ median 1.47 at layer 0 and 3.88 at layer 1, above both caps on every token, so CB2's caps are drastic. The inference slider takes PPL from 56.38 with the register path fully on to 246.72 with it off, a factor of 4.38, with no knee. Against training (F3.1 only 1.087 times worse), it overstates the path's value about 4 times.

**CB1 to CB3:** no result recorded in the protocol.

## 9. Unclear items

- **G1** was dropped by the author on 2026-11-27 (§3).
- **G4 and CB1 to CB3.** The protocol has no scored entry for any of them. This note does not say whether they were dropped or are still queued.
- **G3′'s Gate 3.** The Colab reading is +115%. Protocol §5.18 (RR-B, a local replay on SR-π.3's tokens) reads +119% after the 2026-10-16 correction. They appear to be different measurements; both are quoted as written.
