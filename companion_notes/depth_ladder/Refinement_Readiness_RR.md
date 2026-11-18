# §5.18 RR: what makes a model refinement-ready?

**What this note is.** A short, readable summary of §5.18 of the depth-ladder protocol, "RR: what makes a model refinement-ready?", written 2026-11-17. The section was pre-registered on 2026-10-06, before any measurement, and scored the same day. It gives the clue that started it, the hypothesis H-RR, the three calls and their results. Every number, verdict and date is copied from the [protocol](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md), which keeps the full running log and remains the record of truth.

**Status in one line.** Two of three calls hit (RR1, RR2); RR3 missed, in the direction that supports H-RR. The decisive test, a second seed of G3′ and of G2, has not been recorded as run.

## 1. Terms

- **Refinement** means running the trained layer stack with more, smaller steps over the same total time. A model that samples a continuous flow should not mind. **Gate 3** (Cell 6b-7) measures the rise in perplexity (PPL) when the steps are increased 1.5 times; for the two-layer models that is $N = 3$ steps against the trained $N = 2$.
- **A Fock model** carries a bank of registers. Tokens write to them, and a reverse channel reads them back after each layer step. G2 is the two-layer Fock model. G3 adds an exchange field (an attention-shaped potential), and G3′ is G3 with hardened routing (QK-norm and a 0.3 field clip). F3.1 has no registers.
- **The destruction gate** $g$ of a register, per layer, is how much of its content is emptied at that step: near 1 means reset. **Salience** is how strongly a register's content is kept.
- **Bookkeeping** is the per-step register update that refinement repeats at every extra step.

## 2. The clue

G3 and G3′ differ only in the routing hardening (QK-norm and the 0.3 field clip). Yet one fails refinement and the other does not. Gate 3 at 1.5 times the steps:

| model | Gate 3 |
| --- | ---: |
| G3 | +1,342% |
| G3′ | +115% |
| G2 | +1,274% |
| F3.1 | +143% |
| L=4 | +216% |

**Already ruled out:** the $\pi$ crossing (SR-π.3: G3′'s layer-1 $\theta$ of 3.40 still crosses), the size of the register increment (SR-π.4b: G3′'s is 7.8 times, like its siblings'), the exchange field itself (G2 has none and fails), and the repeated register bookkeeping on G2 (SR-π.3b, about 13% of the penalty).

**What separates them.** Freezing the register bank at initialisation costs G3′ +2,280%, against +24% for G2 and +17% for G3. DP1 had shown that G2's layer-0 destruction gate is switch-like (median 0.994): its register content is mostly reset and rewritten each layer.

## 3. The hypothesis H-RR

- A Fock model is refinement-ready when its register state is **accumulated**, a slowly built memory. Extra steps then add more of the same, so the update approximates a flow.
- It fails when its register state is **reset and rewritten** at each layer. Refinement then inserts rewrite events the model never trained with, so the update is a sequence of maps.
- F3.1 has no registers and is consistent with this.

## 4. Measurements, calls and results

Both measurements are evaluation only, on the CPU, on G3′'s best checkpoint. RR-A extends `debug/dp_register_statistics.py` to G3′ (gate, salience, salience left after the last layer). RR-B runs `debug/sr_pi3b_bookkeeping.py`: Gate 3 under the standard refinement against refinement with the bookkeeping held to once per trained layer.

| call | prediction | called | G3′ | G2 | G3 | result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| RR1 | G3′'s layer-0 destruction gate has median below 0.5 | 60% | 0.478 | 0.994 | 0.978 | HIT, narrowly |
| RR2 | G3′'s median salience left after the last layer is at least 10× G2's | 55% | 0.0646 | 0.0007 | 0.0027 | HIT (about 90× G2's) |
| RR3 | holding the bookkeeping changes G3′'s penalty by under 30% | 55% | +121% | −13% | — | MISS |

RR3's value for G3′ is the corrected one (2026-10-16): Gate 3 goes from +119% to +468% when the bookkeeping is held. The first measurement had used untrained routing temperatures because of a loader bug; the correction is in §5.22 of the protocol and changes no verdict.

**Readings.**

- **The layer-0 destruction gate is bimodal in G3′** (10th percentile 0.03, 90th 0.999). It empties about half its registers and keeps the rest, where G2 empties nearly all of them.
- **RR3 missed in the direction that supports H-RR.** In G2 the per-step bookkeeping under refinement is harmful: holding it back helps a little. In G3′ it is the reverse. The per-step updates are part of why it refines well, and holding them back raises Gate 3 from +119% to +468% (penalty 0.79 to 1.74 nats). Its register state behaves like an accumulated quantity that more steps integrate smoothly.
- **Overall:** RR1 and RR2 hold, and RR3's direction agrees, so H-RR gains support. G3′'s DP3 correlation at layer 1 is also stronger than G2's (G3′ at $-0.63$).

## 5. What is pending

- **The decisive test** is a second seed of G3′ and of G2 (GPU), to rule out a basin one seed happened to find. §5.18 records it as remaining.
- **Later readings elsewhere in the protocol that bear on H-RR** (for orientation, not part of §5.18's scoring):
  - PM1 (§5.15) has a memory accumulated over tokens. It refines like its base at $N = 3$ but not beyond, so accumulation alone is not sufficient (RR-PM2 HIT, RR-PM1 MISS). See [Poisson_Mode_Wells_PM1_PMX.md](Poisson_Mode_Wells_PM1_PMX.md).
  - The exchange-field-only model XRB-F (§5.25) refines at +134% at Gate 3, near F3.1 and G3′, so the exchange field does not cause the large refinement failures. See [Exchange_Field_Routes_XR.md](Exchange_Field_Routes_XR.md).
