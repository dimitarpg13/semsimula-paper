# Template Matchers and Depth

**What this note is.** The pre-registration of the depth arms of the template-matcher test: the same premises and the same matchers on models that differ in the number of layers. It tests the author's conjecture that at eight layers a layer is close to its own flow, so the path a particle takes means more, and that more layers give a matcher a longer path to read. Written 2026-11-25, after the SR2-2B run was reported ([Template_Matchers_on_the_Dynamics_SR2_2B.md](Template_Matchers_on_the_Dynamics_SR2_2B.md)) and before any result of a depth arm was read.

**Status.** Calls TL1 to TL6 **confirmed by the author on 2026-11-25, as written; binding.** The extraction for G2 (prose, two layers, 0.5B tokens) was started on 2026-11-25, before the confirmation, and its files were not read before it. **G2 is reported** ([Template_Matchers_and_Depth_G2.md](Template_Matchers_and_Depth_G2.md)); its numbers make TL5 unreachable (§6 there). The four-layer prose model is next, then the code models.

---

## Contents

1. [The question](#1-the-question)
2. [The models](#2-the-models)
3. [What changes from the first run, and why](#3-what-changes-from-the-first-run-and-why)
4. [The readings](#4-the-readings)
5. [Calls, proposed](#5-calls-proposed)
6. [What the results would decide](#6-what-the-results-would-decide)
7. [Caveats](#7-caveats)
8. [Reproduce](#8-reproduce)

---

## 1. The question

On SR2-2B, two layers, no reading of the evolution identified the logical forms better than the embedding. Layer 1, where the step is long and the particles leave the wells, read worst. Does the gap close with depth?

A layer's displacement is a better summary of a flow when the step is short. Let $\Delta_{\ell}$ be the trained displacement and $\Delta_{\ell}^{(16)}$ the one obtained by cutting the same step into 16 substeps with the context frozen. The **flow gap** of a layer is

$$
g_{\ell} = \mathrm{median} \ \frac{\lVert \Delta_{\ell} - \Delta_{\ell}^{(16)} \rVert}{\lVert \Delta_{\ell}^{(16)} \rVert} ,
$$

zero when the layer moves its particles along its own flow. The conjecture says that $g_{\ell}$ falls with depth and that the readings of the path improve with it.

## 2. The models

| model | layers | tokens | text | role |
| --- | ---: | ---: | --- | --- |
| G2 | 2 | 0.5B | prose | the two-layer end of the clean comparison |
| L4 (prose) | 4 | 0.5B | prose | the four-layer end; the same settings except depth and, with it, the step |
| SO-CODE | 2 | code | code | weaker: English premises are outside the training text |
| L4-CODE, L8-CODE | 4, 8 | code | code | weaker, as above |

The prose pair has the live gradient path, the pair force and the reverse channel on. They are not SR2-2B's settings, so the depth question is read within the pair only. The eight-layer prose model does not exist.

## 3. What changes from the first run, and why

Two choices were made from the SR2-2B report, before any depth result.

1. **"Unless" is left out of fitting and scoring for the primary reading.** On SR2-2B it filled the hard set (§5.4 of the report). The twelve-wording numbers stay as a secondary reading with the same number of particles and the same dimension. Whether "unless" should be a form of its own is the author's ruling. **Discussion of 2026-11-25:** the author proposed filing it as a form of its own. The drawbacks, as the agent sees them: (a) it is not a new logical form, since "unless P, Q" is "if not P, then Q" and also "P or Q", so its templates would recognise a wording, not a form; (b) it would need two valid templates (the premise "not P" and the premise "not Q") and two fallacy kinds; (c) with one wording in its family, it can never be tested on an unseen wording, which is the test that matters, unless more wordings of the same form are added ("Q unless P", "except if", "if not P, then Q"); (d) its clouds rest on one wording instead of six. Leaving the confirmed calls as they are, it can be added as a further secondary reading. **Done the same day** as the third reading of the scoring script, before any depth result was read. Measured on SR2-2B it breaks the conditional templates on unseen wordings ([Connectives_With_a_Hidden_Negation.md](Connectives_With_a_Hidden_Negation.md)); it is kept only to follow the effect with depth.
2. **Paraphrases are also read at a fixed rate of near-miss fires** (1%), because the rule's threshold is a different operating point for each matcher (§5.5).

Everything else is unchanged: the same 60 development and 60 test clauses, the same search over the particles (1 to 8) and the dimension (6, 10, 15, 20, 25, 31).

## 4. The readings

Per model and per matcher: R1 (unseen wordings) without "unless" and with it, the paraphrases caught at the rule's threshold and at 1% near misses, and the threshold-free score. The matchers are the embedding (reference), every single layer's state, displacement and well-frame vector, and three joint ones:

- **path:** the displacements of all layers, concatenated;
- **wells:** the well-frame vectors of all layers;
- **states:** the states after all layers.

All differences are to the embedding **of the same model** and paired over the test clauses, with 95% intervals from 4,000 resamples. Between two models the differences are compared clause by clause, since the clauses are the same. The flow gap is read for every layer, on ordinary text and on premises.

## 5. Calls

Proposed and confirmed on 2026-11-25. A call hits when the point difference reaches the mark and, where an interval is named, its lower end is above zero. All are for the prose pair; the code arms have no calls yet.

| id | prediction | called |
| --- | --- | ---: |
| TL1 | the flow gap, averaged over the layers, on ordinary text is lower at four layers than at two | 85% |
| TL2 | the gap of **path** to the embedding on R1 is larger (more positive) at four layers than at two by at least 0.02, interval above zero | 30% |
| TL3 | at four layers **path** is at or above the embedding on R1 (point difference at or above zero) | 25% |
| TL4 | at four layers **path** beats the embedding on paraphrases at 1% near misses by at least 0.05, interval above zero | 20% |
| TL5 | the worst single displacement of layers 1 to 3 at four layers has R1 at least 0.10 above the displacement of layer 1 at two | 45% |
| TL6 | at four layers **states** beats the embedding on paraphrases at 1% near misses by at least 0.05, interval above zero | 40% |

**Reasoning.** TL1: the four-layer step is half the two-layer one at the same total time; the code models already show the fall. TL2 and TL3: on SR2-2B the gap was 0.036 below (path A, without "unless"). A shorter step moves the displacement of layer 1 closer to a flow, but the linear part of a displacement cannot change the matcher (§3.1 of the report), so the gap may not close. TL4: the displacement matcher ranked paraphrases as the embedding did on SR2-2B. TL5: the layer-1 reading breaks where the particles leave the wells; shorter steps may break it less. TL6: on SR2-2B the states beat the embedding at 1% near misses by 0.053, interval just above zero, and that was one of many readings.

**Not a call.** The wells matcher (it read worst on SR2-2B), the code models, and the flow gap on premises.

## 6. What the results would decide

- **TL1 hits and TL2 hits.** The conjecture is supported in the form tested here: a layer closer to its flow gives a path that reads better. The author would then train an eight-layer model on OpenWebText at 0.5B tokens, and not at 2B, since the readings of the dynamic landscape did not differ between 0.5B and 2B tokens (author, 2026-11-25).
- **TL1 hits and TL2 misses.** Depth brings the layer close to its flow without improving the signature, at four layers. The path would then need another reading (angles between successive displacements, time inside a well, the velocity).
- **TL1 misses.** The pair does not differ enough in step to test the conjecture.
- **TL6 hits.** The states carry paraphrase information beyond the words, with depth. It confirms or not the one post-hoc gain of SR2-2B.

## 7. Caveats

- One pair, one seed each, 0.5B tokens. The two models differ in the step and in depth at once, which the conjecture does not separate.
- The clauses were drafted by the agent and await the author's check.
- At four layers the path has twice the coordinates of SR2-2B's; the search may pick fewer particles for it.
- The differences between the two models are paired over clauses and do not include the training-seed spread.
- The reverse channel acts after the conservative stack. The states after the last layer include it, the displacements do not.

## 8. Reproduce

```text
debug/template_depth_test.py OUT FOLDER name=g2        # extraction, any depth
debug/template_depth_analysis.py OUT OUT g2            # scoring; checked on SR2-2B: reproduces the stored R1, R2 and
                                                       # fixed-rate paraphrases of matchers h0 and A exactly
debug/template_depth_analysis.py OUT OUT g2,l4         # with the comparison between depths
```
