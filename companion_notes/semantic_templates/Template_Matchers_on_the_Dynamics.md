# Template Matchers on the Dynamics

**What this note is.** The pre-registration of a second family of template matchers. The geometry matcher of [Template_Matcher_From_Geometry.md](Template_Matcher_From_Geometry.md) reads the premise's particles as states, the trained layer states $h^{(1)}$ and $h^{(2)}$ in that report. **Terminology (clarified 2026-11-27 at the author's question):** "geometry" names the matcher, the same whatever vector stands for a particle: cosines and norms between particles, relational coordinates, clouds in a small subspace, no word list. The embedding matcher is that geometry fed the embedding $h^{(0)}$; the matchers here are that geometry fed the displacements or the well frames. Earlier wording called the version on the layer states "the geometry-only matcher"; read it as "the geometry matcher on the layer states". The matchers here read what the trained model did to those particles: how far each layer moved them, and where they sit in the frame of the wells. The reference they must beat is the embedding, the same matcher on the states that no well has acted on. Written 2026-11-25, before any result of these matchers was seen.

**Status.** Run on SR2-2B and scored on 2026-11-25: **TD2 and TD5 hit, TD1, TD3 and TD4 miss.** Report: [Template_Matchers_on_the_Dynamics_SR2_2B.md](Template_Matchers_on_the_Dynamics_SR2_2B.md). The calls of §5 were **confirmed by the author on 2026-11-25, as written,** before any extraction or scoring, and are binding. The depth-code correction that had to come first is done ([Depth_Code_Correction.md](../depth_ladder/Depth_Code_Correction.md)); it does not touch this test.

---

## Contents

1. [The question](#1-the-question)
2. [What was known before](#2-what-was-known-before)
3. [The matchers](#3-the-matchers)
4. [The test](#4-the-test)
5. [Calls](#5-calls)
6. [What the results would decide](#6-what-the-results-would-decide)
7. [Caveats](#7-caveats)
8. [Reproduce](#8-reproduce)
9. [Later evidence: the tests that need context (2026-11-27)](#9-later-evidence-the-tests-that-need-context-2026-11-27)

---

## 1. The question

The geometry matcher fires a template when the figure of the premise's particles falls in the template's cloud. Fed the trained layer states, it does this well on fresh clauses when the wording is known, but the same matcher fed the embedding $h^{(0)}$ does nearly as well once the connective's position is included, and better on unseen wordings. So it is not shown that the trained landscape adds anything. **Update 2026-11-27 (the author's question; the sentence states what was known when this note was written, 2026-11-25):** later runs show that the layer states do add something, on one task. On paraphrases caught at 1% near misses the early state is above the embedding on SR2-2B ($+0.051$), G2 ($+0.064$) and L4 ($+0.158$), each interval above zero, and level on SR2-NP ([Early_State_Matcher_Runner_Up.md](Early_State_Matcher_Runner_Up.md)); part of L4's gain is its weak embedding (0.177 against G2's 0.255). On identifying the form on unseen wordings the states stay level with the embedding, and on the analogy test their lead on L4 is not resolved ([Analogy_Test_Results.md](Analogy_Test_Results.md)). The displacements and well frames proposed here have not been shown to add anything beyond the states: the well frames fall below them everywhere, and the path's lead on L4's analogy items (0.76 against the states' 0.71) is within the noise. On the tests that need context (task A and B-logic, 2026-11-27) the embedding falls far behind the layer states; what that does and does not say about the matchers of this note is in [§9](#9-later-evidence-the-tests-that-need-context-2026-11-27). The author's proposal is that the signature of a premise is in the evolution of its particles. This test asks whether the evolution identifies a logical form better than the words do. The discussion that led here, with the readings behind it, is in [What_the_Matchers_Read.md](What_the_Matchers_Read.md).

## 2. What was known before

Two facts about the SR2-2B checkpoint were read on ordinary text (12,288 tokens, in a smoke test of the extraction script, 2026-11-25) and so are **not** blind:

- **The forces move particles away from the well that dominates them on entry.** Layer 0: median radius 0.26 widths on entry and 0.44 after the step, before the layer-end LayerNorm; 1% of tokens end closer. Layer 1: 1.95 and 9.23; 0.02% end closer. The radius is measured to the same well, in the layer's own context.
- **The layer-end LayerNorm rescales every state to the same length (19.6),** while the embedding has length about 3.7. A plain difference $h^{(1)} - h^{(0)}$ would therefore mostly measure the rescaling. The displacements below stop before the LayerNorm.

## 3. The matchers

A premise is a set of particles (the tokens of the connective, of the two clauses, of premise 2 and of "Therefore"). Each particle has a vector; the matchers differ only in that vector.

Let $h_{\ell}$ be the state that enters layer $\ell$, and let $\tilde{h}_{\ell}$ be the state the layer's step produces before its LayerNorm. The **displacement** is

$$
\Delta_{\ell} = \tilde{h}_{\ell} - h_{\ell}, \qquad \ell = 0, 1 .
$$

The **well-frame vector** of layer $\ell$ collects, over the eight wells $k$ of the bank, at the context the layer used: the share $\log \pi_k$, the radius $r_k = \sqrt{q_k}$ on entry, the change of the radius over the step, the stiff coordinates $u_k = B_k^{\top}(h - \mu_k)$ on entry, and their change over the step. With

$$
q_k = (h-\mu_k)^{\top} P_k (h-\mu_k), \qquad \pi_k \propto w_k e^{-q_k/2} .
$$

| matcher | vector of a particle | role |
| --- | --- | --- |
| **A** | displacements of both layers | primary |
| **B** | well-frame vectors of both layers | primary |
| h0 | embedding state | the reference to beat |
| the single layers; the states after layer 0 and 1 | | secondary readings |

Everything after the vector is as in the geometry matcher and unchanged: centring and whitening by ordinary text, the cosines and norms between particles, the relational coordinates, one Gaussian cloud per kind of premise with one covariance, the discriminant subspace, the 1% false-fire rule, plus the connective's position along the axis from disjunctive to conditional wordings. The number of particles $m$ (1 to 8) and the subspace dimension $d$ (6, 10, 15, 20, 25, 31) are chosen by the same search for every matcher: the best content-wise five-fold score on the development clauses. Nothing else is tuned.

## 4. The test

- **Clauses.** Development: the 60 clauses that shaped the earlier matcher. Test: 60 new clauses that no matcher has been scored on (`template_fresh_clauses.py`), drafted by the agent and awaiting the author's check.
- **Wordings.** Twelve (six conditional, six disjunctive), each left out in turn for the unseen reading.
- **Premises.** Valid forms, the two fallacies, the affirmed disjunct, near misses, other subjects, unrelated clauses, subject-keeping paraphrases, ordinary text: the same kinds as before.
- **R1, the hard set on unseen wordings.** Half the mean detection of modus ponens, modus tollens and the two disjunctive syllogisms (the right template fires) plus half the mean silence on the five fallacy kinds (affirming the consequent, denying the antecedent in two forms, affirmed disjunct in two forms).
- **R2, paraphrases on known wordings.** The mean rate at which the right template fires when premise 2 keeps the subject and paraphrases the predicate.
- **Comparison.** Each matcher against h0 on the same clauses, the difference averaged over clauses, with a 95% interval from 1,000 resamples of the clauses.

## 5. Calls

Confirmed by the author on 2026-11-25, binding. A call hits when the point difference to h0 reaches the mark **and** the lower end of the interval is above zero.

| id | prediction | called |
| --- | --- | ---: |
| TD1 | matcher A beats h0 on R1 by at least 0.03 | 20% |
| TD2 | matcher A beats h0 on R2 by at least 0.10, while at most 2% of near misses fire | 50% |
| TD3 | matcher B beats h0 on R1 by at least 0.03 | 15% |
| TD4 | matcher B beats h0 on R2 by at least 0.10, while at most 2% of near misses fire | 25% |
| TD5 | on known wordings matcher A detects the two disjunctive syllogisms at a mean at least 0.05 above h0 | 60% |

**Reasoning.** The states after layers 0 and 1 already score about 0.27 on R2 against 0.13 for h0 on the earlier clauses, so a displacement reading may keep that. On unseen wordings the same states score below h0, and the displacement shares the states' word content, so R1 is unlikely. Layer 1 moves particles many widths through the wells, so the well-frame reading is noisier there; hence lower probabilities for B. TD5 repeats the disjunction gap seen earlier on known wordings.

**Not a call.** The step-500 control and the 0.5B prose pair (L = 2 and L = 4) are not run here.

### Outcome on SR2-2B (2026-11-25)

| id | difference to the embedding | 95% interval | outcome |
| --- | ---: | --- | --- |
| TD1 | -0.085 | -0.104 to -0.064 | miss |
| TD2 | +0.105, near misses fire 0.24% | +0.065 to +0.147 | hit |
| TD3 | -0.141 | -0.163 to -0.122 | miss |
| TD4 | +0.081, near misses fire 1.2% | +0.042 to +0.122 | miss |
| TD5 | +0.101 | not computed | hit |

Two post-hoc readings qualify it: one wording ("unless") accounts for the fallacy fires and for TD5, and at a fixed rate of near misses the displacement matcher ranks paraphrases as the embedding does (TD2). The report gives both. The later tests that need context are summarised in §9.

## 6. What the results would decide

- **TD1 or TD3 hits.** The evolution carries a signature of the form that the words do not, and it survives new wordings. This would support reading templates in the dynamics.
- **TD2 or TD4 hits while TD1 and TD3 miss.** The evolution gives some invariance to paraphrase but not to wording.
- **All miss.** What identifies the forms in this model is the words and the connectives. Template matching by the trained states alone is then not supported on this model, at two layers.

## 7. Caveats

- One model (SR2-2B, two layers), one checkpoint, one seed.
- Clause boundaries are given to the matcher. English premises, drafted by the agent.
- The displacement and the well frame depend on the words through the forces; only the comparison with h0 separates the two.
- At layer 0 the displacement is a function of the embedding and its context alone, so A and h0 share most of their content by construction.
- A layer's bank carries a depth code that a wrapper outside the layer step selects. A script that evaluates a layer's force or step outside the forward pass must select it. The extraction here reads the bank inside the forward pass, so it is not affected; the audit of 2026-11-25 found one published test that was (the conservativity test, rerun since).
- The R1 and R2 thresholds of 0.03 and 0.10 are the agent's, from the resolution of the earlier tables.

## 8. Reproduce

```text
debug/template_dynamics_test.py OUT FOLDER name=sr2_2b      # extraction, about an hour on 8 threads
debug/template_dynamics_analysis.py OUT OUT sr2_2b          # scoring
```

## 9. Later evidence: the tests that need context (2026-11-27)

Added at the author's question: does this note need updating now that the embedding falls short of the trained model's readings on the tests that need context? It does, with one distinction kept clear. **What beats the embedding there is the layer states, read at one position by a one-particle reader. It is not the displacements or the well frames proposed here, and not the geometry matcher's particle coordinates.** The displacement $\Delta_{\ell}$ and the well-frame vector are defined in §3 of this note; their properties are analysed in [Template_Matchers_on_the_Dynamics_SR2_2B.md](Template_Matchers_on_the_Dynamics_SR2_2B.md), §1 and §3 (a whitening matcher cannot see the part of a displacement that is linear in the entry state), and in [Early_State_Matcher_Runner_Up.md](Early_State_Matcher_Runner_Up.md), §3.2 and §3.3.

### 9.1 What was measured

| test | what the words cannot give | embedding | layer states | report |
| --- | --- | --- | --- | --- |
| task A, the form at "Therefore" | the form, carried to a token that is the same in every premise | ranks the right template first 20 to 22%; valid against twin AUROC 0.51 to 0.54 | 54 to 62% (chance 25%, the length alone 41%); AUROC 0.74 to 0.76 | [Summary_State_Test.md](Summary_State_Test.md) |
| B-logic, the form set by a context sentence | which clause a context sentence ties the signal to, on identical premise tokens | 0.500 on every reading, tied by construction | at "Therefore": foil 0.70 to 0.91, twin 0.73 to 0.80 | [BLogic_Test.md](BLogic_Test.md), §5.2 |
| the B-logic control, both clauses named | the binding, with mention removed | 0.500, tied by construction | at "Therefore": foil 0.57 to 0.97, twin 0.69 to 0.84 | [BLogic_Test.md](BLogic_Test.md), §5.6 |

The models are SR2-NP, G2 and L4; the matched GPT-2 reads form and binding at "Therefore" at SR2-NP's level, from its third block ([GPT2_Template_Battery_Results.md](GPT2_Template_Battery_Results.md)).

### 9.2 What it means for the matchers of this note

- **The states, not the evolution readings, carry the context.** Task A read only the states at "Therefore" (and the length); it did not read the displacements or the well frames there. In B-logic the geometry matcher read every vector. L4's later states and well frames gave twin readings of 0.56 to 0.68, but the control was read only through the early state and all states, and through the matcher's coordinates those fell to chance or below (the early state's twin reading 0.26 to 0.28 on G2, 0.28 to 0.42 on L4): what they followed was the mention, not the binding. **Measured.**
- **The advantage over the embedding is not specific to the conservative dynamics.** GPT-2's residual stream does the same at the summary position. The context result shows that a trained model's states carry what the words alone cannot. It does not show that wells or displacements carry it. **Derived.**
- **The calls of §5 are unchanged.** They were made on verbatim premises, where the words decide, and they stand as scored.
- **The author's proposal, that a premise's signature is in the evolution of its particles, is still untested where it matters most.** The tests in which the words cannot help have not yet been read through the displacements or the well frames at the summary position. **Open.**

### 9.3 The test this suggests (not a call)

Feed task A's reader the displacement and the well-frame vector of "Therefore", layer by layer, on the task A premises and the B-logic control, and compare it with the state at "Therefore" on the same premises. A displacement or well frame that matches or beats the state there would be the first evidence that the evolution carries form and binding. A reading below the state would confirm that in these models the context sits in the states and the wells only pass it on. Whether the stored arrays hold these vectors at "Therefore", or a new extraction is needed, is to be checked. Calls would be written and confirmed before scoring.
