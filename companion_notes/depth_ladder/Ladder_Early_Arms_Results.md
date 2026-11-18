# The depth ladder: early arms and their results

**What this note is.** A short, readable summary of the first ladder arms and their results, written 2026-11-17. It covers the $L = 2$ arms of protocol §5.1, §5.2, §5.4 (the `'none'` entry), §5.5, §5.6, §5.7 and §5.8, the $V_\varphi \times$ Fock factorial of §3.2, and the $L = 4$ section (§3.1, the run-4 pre-registration of §3.3, the $L = 4$ diagnosis block and §3b, the D-series). For each arm it gives what the arm is, what was registered before it ran, the result table, and the verdicts as recorded. Every number, verdict and date is copied from the [protocol](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md), which stays the full running log; each heading carries its protocol section so a reader arriving from a model card can match it. The rules the arms ran under (fixed configuration, queue, band ledger, run health, publication audit) are in [Ladder_Protocol_Overview.md](Ladder_Protocol_Overview.md).

**Contents**

1. [Terms and the arms at a glance](#e-summary)
2. [$L = 2$ `'attention'` at 3e-04 (protocol §5.1)](#e-attn-3e4)
3. [$L = 2$ `'none'` at 3e-04 (protocol §5.2)](#e-none-3e4)
4. [$L = 2$ `'none'` at 1.2e-03 (protocol §5.4, anchor `p-gen2-none`)](#e-none)
5. [$L = 1$ `'none'` (protocol §5.5)](#e-l1)
6. [$L = 2$ `'none'`, reverse channel off, and its live-gradient follow-up F3.1 (protocol §5.6)](#e-norc)
7. [$L = 2$ `'attention'` at 1.2e-03 (protocol §5.7)](#e-attn)
8. [$L = 2$ `'attention_potential'` at 1.2e-03, the gradient check and probe (b) (protocol §5.8)](#e-attnpot)
9. [Runs 10 and 11: the $V_\varphi \times$ Fock factorial (protocol §3.2)](#e-factorial)
10. [The $L = 4$ section (protocol §3.1, §3.3, the $L = 4$ diagnosis block, §3b)](#e-l4)

---

<a name="e-summary"></a>
## 1. Terms and the arms at a glance

- **PPL** is validation perplexity; lower is better. **Settled** is the mean of the last three evaluations (one every 500 steps). Every full arm trains 32,500 steps on 532.5M tokens, from scratch, one seed; probes stop at 3,000 steps.
- **The mechanism knob** (`LADDER_MECHANISM`). `'none'`: no exchange field (also called "no-exchange"). `'attention'`: a $\xi$-routed exchange field that enters as a force (8 heads, $d_k = 48$, $\lambda$ pinned at 1.0). `'attention_potential'`: the same field entering as the gradient of a scalar potential, its conservative twin. The two fields carry exactly 589,824 parameters each.
- **The Fock mechanism** is the register bank, which reaches the tokens only through the **reverse channel**. With `REVERSE_CHANNEL = False` (tag `norc`) the model is the conservative architecture alone: $V_\theta$, $V_\varphi$ and the $\xi$ content routing.
- **Detached and live gradients.** In the early arms (labelled Gen 2 in the protocol), $V_\varphi$ and $\xi$ detach their sources, so no loss gradient reaches an earlier token through them. "Live" arms (Gen 3) restore that path; the forward force is unchanged.
- **Stiffness** $\omega \Delta t$ is how far, in radians, a stiff mode rotates in one layer step. Above 2 an explicit leapfrog step would amplify. The integrator handles those modes exactly, so it is a stiffness reading, not an instability.

| run | arm | LR | settled | ratio to matched GPT-2 (49.81) | section |
| --- | --- | --- | ---: | ---: | --- |
| 2 | L=2 'attention' | 3e-04 | 68.33 | 1.372 | §5.1 |
| 3 | L=2 'none' | 3e-04 | 75.09 | 1.508 (1.507 in the §5.4 sweep paragraph) | §5.2 |
| 3 | L=2 'none' | 1.2e-03 | 66.98 | 1.345 | §5.4 |
| 7 | L=1 'none' | 1.2e-03 | 87.09 | 1.748 | §5.5 |
| 8 | L=2 'none', reverse channel off | 1.2e-03 | 87.93 | 1.765 | §5.6 |
| F3.1 | as run 8, V_φ + ξ live | 1.2e-03 | 57.76 | 1.160 | §5.6 |
| 9 | L=2 'attention' | 1.2e-03 | 63.51 | 1.275 | §5.7 |
| 5 | L=2 'attention_potential' | 1.2e-03 | 80.90 | 1.624 | §5.8 |
| probe (b) | as run 5, exchange-field gradients live (rglive) | 1.2e-03 | 61.11 | 1.227 | §5.8 |
| 4 | L=4 'none' | 1.2e-03 | 71.75 | 1.441 | L=4 section |
| L=4 live | as run 4, V_φ + ξ live | 1.2e-03 | 50.10 | 1.006 | L=4 section |

<a name="e-attn-3e4"></a>
## 2. $L = 2$ `'attention'` at 3e-04 (protocol §5.1)

Run 2, **DONE 2026-09-20**, at the old learning rate.

| reading | value |
| --- | ---: |
| final (step 32,500) | 68.44 |
| best (step 31,000) | 66.03 |
| settled | 68.33 |

- Against the references: 0.838× the warm-started $L = 8$'s 81.58, 0.856× Alternative E's 79.80, and 1.252× GPT-2's 54.67 (the unmatched published baseline).
- **Prediction scored: MISS.** Recorded 65, band 63–68; actual 68.33, error +3.33. The run flattened at 66.03, then reversed (68.23, 68.31, 68.44) while the LR was still falling. The saturating read (64.6) beat the log-linear one (59.9). "Saturating is better, not conservative."
- Health: `share_max` peaked at 16.71 near step 800 and fell to about 3.3; `bproj_sig` reached 26.07.
- **Attribution: none.** Four things differ from the 81.58 reference (schedule, Alternative E, the exchange field, depth 8 to 2).

<a name="e-none-3e4"></a>
## 3. $L = 2$ `'none'` at 3e-04 (protocol §5.2)

Run 3 at the old learning rate, **DONE 2026-09-21**. A single-variable contrast with §5.1: 589,824 fewer parameters, paired on identical batches.

| reading | 'attention' | 'none' | delta |
| --- | ---: | ---: | ---: |
| final | 68.44 | 75.48 | +7.04 |
| best | 66.03 (31,000) | 73.20 (31,000) | +7.17 |
| settled | 68.33 | 75.09 | +6.76 |

- **+9.9%, +0.0943 nats.** The gap saturated: 3.7% at step 5,000, 7.1% at 8,000, 10.0% at 11,000, then 8.9–10.9% (mean 9.8%) from 20,000 to 32,500.
- **Prediction scored: MISS.** The band was 20–30%, from a linear fit over steps 4,000–11,000. The protocol's rule from this: fit a saturating form, then treat even that as an upper bound.
- **The attribution, as corrected.** At 3e-04 the exchange field accounted for 6.76 of the 13.25-point gain over the 81.58 reference (51%). The subsection was suspended 2026-09-22 and the suspension lifted 2026-09-26: with both arms at 1.2e-03 the exchange field is worth **+5.2%**, not +9.9% (see §5.7). Every other number here is a 3e-04 number.
- The protocol called 75.09 "the best conservative result in the programme" on 2026-09-21. **Flag:** this arm has the reverse channel on, and the 2026-09-28 card audit established that reverse-channel models are not purely conservative.

<a name="e-none"></a>
## 4. $L = 2$ `'none'` at 1.2e-03 (protocol §5.4, anchor `p-gen2-none`)

Run 3 at the ladder learning rate. The queue cites this result as "§5.4a"; no heading of that name exists, and the entry is the last paragraph of §5.4.

| reading | value |
| --- | ---: |
| final (32,500) | 67.63 |
| best | 66.56 (31,500) |
| settled | 66.98 |

- **The LR sweep closed the question.** 1.2e-03 beat 3e-04 by **+10.8%** (66.98 against 75.09), moving the GPT-2 ratio from 1.507 to 1.345. 2.4e-03 came back at 69.59, worse by 3.9%. A quadratic through the three points puts the vertex at 1.13e-03. "`LADDER_LR = 1.2e-03` is the ladder learning rate."
- Closing 68.33 to 49.81 needs 27%, and LR tuning typically buys 5–15%.
- Health, from later sections: clip-hit 0.0% (max 0.42); `bproj_sig` saturated at 85.4; the decay bought 19.3% (83.05 at 21,500 to 66.98).
- Endpoint stiffness (Cell 6b-13, 2026-09-29): $\omega \Delta t$ median 3.796, inside the band pre-registered for it (see the L=4 section).
- On 2026-09-28/29 a stale notebook resumed this finished arm by accident. The folder was repaired byte-exact, and 66.98 stands.

<a name="e-l1"></a>
## 5. $L = 1$ `'none'` (protocol §5.5)

Run 7, **DONE 2026-09-24**. One hop ($\Delta t = 8$), same LR, schedule and batches as §4.

**The static-bank caveat (protocol §6.1).** At $L = 1$ the register bank is read by the reverse channel but never updated. Initial salience 1.0 zeroes the creation readout at layer 0, and there is no later layer to train the shared creation gate. So this arm is "one hop with a frozen Fock bank", not "one hop with the Fock mechanism". It also has zero velocity at its only layer.

| reading | L=2 'none' | L=1 'none' | delta |
| --- | ---: | ---: | ---: |
| final | 67.63 | 87.94 | +20.31 |
| best | 66.56 (31,500) | 85.49 (31,000) | +18.93 |
| settled | 66.98 | 87.09 | +20.11 |

- **+30.0%, +0.263 nats.** The gap widened through the decay rather than saturating: 12.9% at step 8,000, 22.1% at 20,000, 30.0% at 32,500. The decay bought $L = 2$ 19.3% and $L = 1$ only 11.2% (98.07 to 87.09).
- **Prediction scored: MISS, on the low side.** The forecast, 74–80, assumed the gap would saturate.
- Health: clip-hit 2.6% (17 of 650, max 1.87) against 0.0% at $L = 2$; no watchdog or spike events; `sig_max` frozen at 14.286 all run (the static bank, visible in the log); resonance monitor empty (a missing diagnostic).
- **What it cannot settle:** how much of the 30% is depth and how much the frozen bank. The decomposition needs an $L = 1$ run at `register_salience_init = 0.5`, outside the ladder by design.

<a name="e-norc"></a>
## 6. $L = 2$ `'none'`, reverse channel off, and its live-gradient follow-up F3.1 (protocol §5.6)

Run 8, the conservative-only control, **DONE 2026-09-25**: run 3 with `REVERSE_CHANNEL = False`, nothing else changed.

| reading | run 3 | reverse channel off | delta |
| --- | ---: | ---: | ---: |
| final | 67.63 | 88.82 | +21.19 |
| best | 66.56 (31,500) | 85.90 (31,000) | +19.34 |
| settled | 66.98 | 87.93 | +20.95 |

- **+31.3%, +0.272 nats** as measured. Inference-time removals had put the mechanism at 3.75× (6b-8 ablation A) and 3.91× (6b-11 at $\lambda = 0$) against this trained-without 1.31×: overstated 2.9× in PPL ratio and 4.9× in nats.
- 87.93 lands within 1% of the $L = 1$ arm's 87.09. Whether that is coincidence or a ceiling was left open.
- The gap to run 3 widened through training: 19.1% at step 8,000, 31.3% at 32,500.
- **Prediction scored: BAND HIT, point high.** Registered 105, band 85–140; actual 87.93, 3% above the lower edge. "The first band hit in the programme's forecast record."
- Health: clip-hit 2.6% (17 of 650, max 2.05); no watchdog or spike events; `bproj_sig` saturated at 88.7.
- **The reason for the hit was withdrawn on 2026-09-30.** The band named growth of $V_\varphi$'s share as the quantity that could turn it. Measured on the checkpoint, $V_\varphi$'s force is 9% of $F_\theta$ at layer 0 and effectively zero at layer 1: it did not grow. No loss gradient reaches an earlier token through $V_\varphi$ or $\xi$.

**P2.1: the same arm with live $V_\varphi$ and $\xi$ sources.** Frozen 2026-10-01, at launch, before any eval. Scored at step 3,000 against the parent's 158.09, ±2.5% noise: move if $\le 154.1$, null 154.1–162.0, worse above 162.0.

| step | parent | P2.1, live | change |
| ---: | ---: | ---: | ---: |
| 1,000 | 267.41 | 241.67 | −9.6% |
| 2,000 | 183.10 | 151.89 | −17.0% |
| 3,000 | 158.09 | 127.73 | −19.2% (0.213 nats) |

- **Scored: MOVE, by eight times the threshold** (2026-10-01). P2.1 is 9.2% below no-exchange (140.61), which has the reverse channel but starved sources.
- Consequence, effective then: every statement pricing the Fock mechanism or the conservative architecture was **suspended**, including the +31.3% and the "overstated by 3×" table. The §5.6 numbers stand as measurements of the starved convention.

**F3.1: P2.1 run to 32,500 steps.** Pre-registered 2026-10-01, at step 3,000, before step 3,001: **point 67, band 60–79**. Beating no-exchange's 66.98 was called even odds; beating `'attention'`'s 63.51 about 1 in 4. Health scored: clip-hit below the parent's 2.6%, all seven causal-leak (SCAF) audits CLEAN, no watchdog trigger.

| reading | step | F3.1 | parent | no-exchange | note |
| --- | ---: | ---: | ---: | ---: | --- |
| mid-run, 2026-10-01 | 15,000 | 77.31 | 107.30 | 86.57 | implied settled 59.8–63.3; ω·dt median 6.41 |
| mid-run, 2026-10-01 | 29,000 | 58.84 (best) | 89.55 | 68.82 | projected settled about 57–59 |
| scored, 2026-10-02 | settled | 57.76 | 87.93 | 66.98 | last three evals 58.39, 57.54, 57.35 |

- **Scored: MISS on the good side**, 2.24 below the band's lower edge. The question the run existed for: **YES**. It settles 13.8% below no-exchange, 9.1% below `'attention'` and 5.5% below rglive, and −34.3% (−0.420 nats) against its own parent with an identical forward pass.
- Health, all HIT: clip-hit 0.6% (4 of 650, max 2.19); SCAF CLEAN at all seven audits; no watchdog trigger.
- **$V_\varphi$ woke up.** Its attribution (6b-9) grew about 190×, from −0.0015 to −0.291. "$V_\varphi$ is inert" (E1, F5) is withdrawn as a property of the detached convention.
- Endpoint $\omega \Delta t$ median 4.32 (99.8% past 2), the stiffest arm; the in-flight median peaked at 6.4 at step 15,000.
- **Consequences:** the §5.6 Fock price (+31.3%) and the F5 reading are confirmed withdrawn. The Fock mechanism's real value is unknown until no-exchange runs with live gradients (P2.2).

<a name="e-attn"></a>
## 7. $L = 2$ `'attention'` at 1.2e-03 (protocol §5.7)

Run 9, **DONE 2026-09-26**. Identical to run 2 except the LR (77,360,081 parameters both times). It repairs the ladder's central comparison.

| reading | @3e-04 (run 2) | @1.2e-03 (run 9) |
| --- | ---: | ---: |
| final | 68.44 | 63.42 |
| best | 66.03 (31,000) | 61.49 (31,000) |
| settled | 68.33 | 63.51 (gain +7.1%) |

- **What it settles: the exchange field is worth +5.2%, not +9.9%.** With both arms at 1.2e-03, `'none'` 66.98 against `'attention'` 63.51. `'none'` gained 10.8% from the LR change and `'attention'` 7.1%. "A gap measured off-optimum is an upper bound."
- **Prediction scored: HIT, the first point-accurate one.** Registered 63, band 59–68 (2026-09-25); actual 63.51, error +0.51 (+0.8%).
- **The caveat: clipping.** 29.2% of logged steps hit the clip (190 of 650, max 1.42), against 7.5% at 3e-04 and 0.0% for `'none'` at 1.2e-03. So +5.2% is a lower bound on the exchange field. 63.51 stays the ladder number.
- **The gate confound, eliminated** (probe, 2026-09-27). A 3,000-step run with `RELAX_GATE = 'zero_readout'` reached 133.37 against this arm's 131.20: **null, +1.65%**, inside ±5%. The zeroed readout climbs to the scalar arm's operating point within a few hundred steps. "The gate changes how the term starts, not where it ends up."
- Health: no watchdog or spike events; `share_max` peaked at 15.68 and fell to 3.93; `bproj_sig` saturated at 84.25; decay gain 17.1%.

<a name="e-attnpot"></a>
## 8. $L = 2$ `'attention_potential'` at 1.2e-03, the gradient check and probe (b) (protocol §5.8)

Run 5, **DONE 2026-09-27**. The conservative twin of run 9, parameter-matched exactly (77,360,081 both). The only difference is whether the exchange field enters as a force or as the gradient of a scalar potential.

| reading | 'attention' (run 9) | 'attention_potential' (run 5) |
| --- | ---: | ---: |
| final | 63.42 | 81.50 |
| best | 61.49 | 79.14 (31,000) |
| settled | 63.51 | 80.90 |
| ratio to GPT-2 | 1.275 | 1.624 |

- **As first read: the price of conservativity is +27.4%** (+17.39 PPL against `'attention'`), and the conservative field is worse than none (+13.92 PPL, +20.8%, against `'none'`).
- **Not an optimisation failure.** Clip-hit 0.0% (0 of 650, max 0.93); no watchdog or spikes; `bproj_sig` 100.46, the highest of any arm; decay gain 15.5%.
- **Prediction scored: MISS, and the revision made it worse.** Registered 66, band 62–72 (revised before the run in §5.3a); actual 80.90, error +14.90 (+22.6%). The superseded band, 75–82 with point 78, would have hit.
- **The initialisation confound, eliminated** (probe, 2026-09-27). Starting the term 2.75× larger (`RELAX_INIT_SCALE = 0.055`) moved step 3,000 by **+0.57%** (149.84 against 148.99): null.
- **Caveat:** both arms use `RELAX_GATE = 'scalar'`, which the config docstring marks superseded. The +27.4% held "as instantiated here".

**The gradient check (2026-09-29).** The "welded projection" reading of the mechanism was withdrawn: the two forces have the same form. What differs is the backward pass. Measured on both best checkpoints, with a cotangent at the last position:

| arm | layer | force RMS | gradient to earlier tokens | gradient to the token itself |
| --- | ---: | ---: | ---: | ---: |
| 'attention' | 0 | 0.0070 | 0.271 | 0.047 |
| 'attention' | 1 | 0.0714 | 5.28 | 4.28 |
| 'attention_potential' | 0 | 0.0099 | 0 | 0 |
| 'attention_potential' | 1 | 0.0323 | 0 | 0 |

The conservative exchange field sends no gradient into any hidden state. The new leading hypothesis was **H_s, gradient starvation**.

**The two probes, pre-registered 2026-09-29 before either run.** One switch, `RELAX_GRAD_PATH`, changes only the backward pass. Probe (a), `rgdet`, starves `'attention'`. Probe (b), `rglive`, gives `'attention_potential'` live gradients. Both scored at step 3,000 with ±3%:

- (a) supports H_s in [144.5, 153.5]; within [127.3, 135.1] refutes it.
- (b) supports H_s in [127.3, 135.1] (full recovery); within ±3% of 148.99 refutes it.
- (b) revised at step 1,300, after two uninformative evals, against no-exchange's 140.61: $\le 137.1$ "the field now helps"; 137.1–144.1 neutral; $\ge 144.5$ refuted. The original band stays on the record.
- (b) extended to 32,500 steps, pre-registered at step 1,300: **settled 63–72, point 67**; $\le 66.98$ means the field helps; $\ge 76$ refutes H_s; 72–76 partial.

**Probe (b) scored at step 3,000: 133.74, HIT on both criteria** (2026-09-29).

| step | 'attention' | (b) live | no-exchange | 'attention_potential' |
| ---: | ---: | ---: | ---: | ---: |
| 1,500 | 188.01 | 192.73 | 197.04 | 200.79 |
| 2,000 | 158.76 | 162.27 | 167.87 | 173.22 |
| 3,000 | 131.20 | 133.74 | 140.61 | 148.99 |

- Restoring the gradients closes 86% of the gap to `'attention'`. The field now helps, 4.9% better than no field, where the starved version was 6.0% worse. "The sign flip is gone."
- So the +27.4% was "mostly the price of detaching the exchange field's inputs", not of conservativity itself.
- **Not found in §5.8:** a result for probe (a), and an explicit scoring of (b)'s settled value against 63–72. Later tables record (b) settled at **61.11**.

**Probe (b), full run (2026-10-02).** On its best checkpoint (step 31,000, PPL 59.09): the PPL rise when velocity is reset is +32.66 against the parent's +18.80; refinement still reads MAPS; endpoint $\omega \Delta t$ median 1.784. Published 2026-10-02 as a Gen 2 probe, not a ladder arm: `dimitarpg13/semsimula-ladder-owt-d384-l2-attention-potential-rglive`.

<a name="e-factorial"></a>
## 9. Runs 10 and 11: the $V_\varphi \times$ Fock factorial (protocol §3.2)

**Superseded 2026-10-11 by protocol §5.21 (VF)**, the same factorial under live gradients, extended by a wells row. The predictions below were made on Gen 2 numbers and are not scored. This note does not summarise §5.21.

Runs 10 and 11 complete a two-by-two with two cells already measured:

| | with V_φ | without V_φ |
| --- | ---: | ---: |
| with Fock | run 3: 66.98 | run 11, Fock-SPLM: not run then |
| without Fock | run 8: 87.93 | run 10, multi-ξ SPLM: not run then |

- **What it gives that no single arm does:** a replication of the Fock price on a different base; $V_\varphi$'s worth measured twice; and the interaction term, whether $V_\varphi$ and the register bank are substitutes or complements.
- **Pre-registered band:** near-zero interaction and a near-flat right-hand column. Concretely, run 10 within 5% of 87.93 and run 11 within 5% of 66.98.
- **Priority revised 2026-09-27** (§3.3): promoted to follow run 4, since a per-step measure does not predict a trained-without outcome.
- **Unblocked 2026-10-01** by `PAIR_POTENTIAL = 'none'` (tag `nophi`; −137,803 parameters, 0.18%). The arms are not parameter-matched, and their cards must say so.
- **A first look, from the $L = 4$ live arm (2026-10-03).** With the register path present, $V_\varphi$ is inert (6b-9 attribution −0.0035); in the conservative-only live arm it woke up (−0.291). So $V_\varphi$ and the register path behave as **substitutes**, seen across a depth difference.

<a name="e-l4"></a>
## 10. The $L = 4$ section (protocol §3.1, §3.3, the $L = 4$ diagnosis block, §3b)

**Why $L = 4$ (§3.1).** $L = 2$ reached 68.33 where the warm-started $L = 8$ reached 81.58, but the two differ in schedule, Alternative E, the exchange field and depth. $L = 4$ at matched everything is the cheap way to separate them, at half the cost of $L = 8$.

**Run 4, pre-registered 2026-09-27, before launch: point 59, band 55–65.** The basis is the $L = 1 \to 2$ gain of 23.1%, halved because part of it was the register bank switching on (§6.1). **The quantity that could turn it: the timestep, not the depth.** With $T$ held at 8, $L = 4$ means $\Delta t = 2$. Clip-hit rate and `bproj_sig` were named as the tells. Depth is nearly free in parameters: 76,823,511 against $L = 2$'s 76,770,256 (+0.069%).

**Mid-run readings (2026-09-27 to 2026-09-28).**

- $L = 4$ led $L = 2$ from step 1,500 to 8,500 (widest lead −2.1% at 5,000), then fell behind: +9.0% at 14,500.
- **Diagnosis at step 15,500: genuinely behind, not overfitting.** $L = 4$ is worse on the training loss too (up to +0.064 nats at 13,050), and `bproj_sig` is higher (83.1 against 74.6). Two candidate mechanisms: (1) curvature compensation, the model raising $\omega$ to undo the smaller step; (2) weight tying, the same weights serving twice as many layer roles.
- **First $\omega \Delta t$ reading on a CfC+BAOAB arm**, step 20,500, after the monitor fix: median 2.236, 60.9% of pairs past 2. **Pre-registered: $\omega \Delta t$ at $L = 2$ lands in [3.3, 4.2].** Below 2.6 hands the deficit to compensation.
- Nine readings, steps 20,500–24,500: the median is stationary (2.285 ± 0.033). `bproj_sig` tracks the tail, not the median, so a hit would be "less confirmatory than it looked". The band was not moved.
- A projection of 88–93 settled was **corrected 2026-09-29**: it ignored the decay phase.

**Run 4 scored, 2026-09-29.**

| L | dt | T | settled | against L=2 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 8 | 87.09 | +30.0% |
| 2 | 4 | 8 | 66.98 | — |
| 4 | 2 | 8 | 71.75 | +7.1% (best against best: +5.5%) |

- **PPL band: MISS.** Best 70.19 (step 31,000); last three evals 71.47, 71.93, 71.84. The interior optimum at fixed $T$ stands, narrowly.
- **$\omega \Delta t$ band: HIT** at 3.796 ($L = 2$) against 2.292 ($L = 4$), on the same seed and tokens. The $L = 2$ value had been previewed twice (3.797 by accident, 3.796 by a dry run), so the scored run was not blind.
- Halving $\Delta t$ raised $\omega$ by only 21% (0.949 to 1.146): mechanism (2). At the first layer compensation is full; beyond it there is none. This is "consistent with" weight tying, not a demonstration of it.

**The clip confound (C-series, 2026-09-28).** The reverse-channel gate is clipped a median 10–20× on every arm that has it. C0: the gate falls 3.5–4× over training. C1 (scored 2026-10-05): PPL rises on both sides of the trained gate, **HIT**, question closed. C2 (2026-10-05): Adam absorbs the rescale; the clip is cosmetic for the endpoint.

**The D-series (§3b, opened 2026-09-28).** $T = L \Delta t$ cannot hold all three fixed, and the ladder had held $T = 8$.

| step | what it changes | registered | status (2026-09-28) |
| --- | --- | --- | --- |
| D1 | L=4 at dt = 4, so T = 16 | settles in 62–70 | the only rung authorised |
| D2 | depth_code clip 0.25 → 1.0 | if binding: pre-clip norm under threshold within about 2,000 steps, settled PPL better by more than 2% | parked behind runs 10 and 11 |
| D3 | wider depth conditioning | only if D2 is null | parked |
| D4 | untying the shared weights | only if D1–D3 fail | contingent; a new architecture with its own name |

**$L = 4$ with live gradients (Gen 3).** Run 4 plus `VPHI_GRAD_PATH = XI_GRAD_PATH = 'live'`; the forward force is identical.

- **3,000-step probe, pre-registered 2026-10-02 at launch:** move if $\le 135.5$; point about 125, band 112–135. **Scored: 119.77, MOVE and HIT** (−13.8% against run 4's 138.94).
- **Full run, pre-registered 2026-10-02 at step 3,000:** settled point 57, band 52–63. Below run 4 called near-certain; below F3.1 (57.76) even odds.

| reading | L=4 live | run 4 | F3.1 | matched GPT-2 |
| --- | ---: | ---: | ---: | ---: |
| settled | 50.10 | 71.75 | 57.76 | 49.81 |
| ratio to GPT-2 | 1.006 | 1.441 | 1.160 | 1 |
| parameters | 76.8M | 76.8M | 76.6M | 33.7M |

- **Scored 2026-10-03: MISS on the good side**, 1.9 below the band. Below run 4: YES (−30.2%). Below F3.1: YES (−13.3%).
- Health: clip-hit 0.5% (3 of 650, max 1.96) against run 4's 0.0%, "a MISS on the letter of the criterion"; SCAF CLEAN at all seven audits; no watchdog trigger.
- Fair statement as recorded: the first conservative-forward model in the programme to reach the matched transformer's perplexity at the same token budget, with 2.3× its parameters and half its depth. One seed; neither side tuned.

**The forecastability rule, CG3 (Cell 6b-10), pre-registered 2026-10-02 before the run finished.** $L = 4$ is the first arm where the test can be read: at $L = 2$ the first step is the projection onto the LayerNorm sphere. Read only layers 2 and 3, and use the tangential coherence (a⊥).

- **Forecastable:** the true-velocity forecast error (b) is at most 0.90 **and** (a⊥) is above 0.
- **Not forecastable:** (b) at least 1.0 at both layers, i.e. on the stay-put null.
- **Indeterminate:** anything between. The per-step growth (c) is reported separately.

| metric (layers 2 and 3) | layer 2 | layer 3 | mean | registered point / band | verdict |
| --- | ---: | ---: | ---: | --- | --- |
| (a⊥) tangential coherence | +0.749 | +0.302 | +0.526 | +0.10 / −0.10 to +0.30 | above 0 and above GPT-2's +0.195; band MISS, good side |
| (b) true-velocity forecast error | 0.583 | 0.935 | 0.759 | 0.95 / 0.85–1.05 | at most 0.90 (layer 3 alone does not) |
| (b) finite-difference error | 0.664 | 0.969 | 0.817 | 1.00 / 0.90–1.15 | at most 0.90; below GPT-2 at both layers |
| (c) per-step growth, three ε | | | 0.930 / 0.936 / 0.950 | 0.92 / 0.88–0.97 | at or below GPT-2 at every ε; drift 0.020, at the 0.02 limit |

- **Verdict, scored 2026-10-03: FORECASTABLE** by the frozen rule, the first positive forecastability result in the programme.
- **Not yet separable from the gradient fix:** the paired control, 6b-10 on run 4, had not run.
