# Hybrid Template Matchers: the Embedding and the Layer States Together

**What this note is.** A theory and proposal note. The template matchers so far read one kind of vector at a time: either the embedding, or a state or displacement of the trained layers (the "geometry" of the dynamics). Across four models the two kinds fail at different tasks. This note asks whether one matcher can use both, and how. It sets out what the measurements so far imply, the mathematics of combining two readings, eight ways of building a hybrid with their pros and cons, and a staged plan whose first step costs minutes on data already extracted. Written 2026-11-25 at the author's request. Nothing in it has been measured yet; the numbers quoted are from the earlier reports.

**In brief.**
- **The two readings are the same matcher on different vectors.** Whitening, relational coordinates, clouds, the rate and the 1% rule are identical. A hybrid needs no new machinery, only a rule for which vector feeds which part of the decision.
- **The embedding wins on new wordings; the layer states sometimes win on paraphrases.** On unseen wordings the embedding scores 0.95 to 0.98 on every model and no layer reading beats it by more than 0.01. On paraphrases at 1% near-miss fires the best layer reading beats the embedding on G2 (0.320 against 0.255) and on L4 (0.334 against 0.177), but not on SR2-NP (0.280 against 0.298).
- **The layers cannot add information, only make it readable.** Every layer state is a function of the embeddings of the premise and its context, so the question is not what the states know but what the matcher, which reads only pairwise cosines, can see in them (§3.2).
- **There is one ceiling that no single reading has broken.** Every reading on every model ranks paraphrases above near misses with nearly the same AUROC, 0.76 to 0.79, and none catches more than a third of the paraphrases at 1% near misses. If the readings miss the same paraphrases, no choice of vectors can help, and the fix must come from the coordinates (§3.5).
- **A logical form factors into four parts:** the frame (conditional or disjunctive), which clause the second sentence restates, whether it negates it, and whether the restatement is the same statement. Validity is a lookup on the first three; the fourth is a gate. This suggests giving each part to the reading that is best at it (§3.1).
- **The recommended first step** is a complementarity check on the saved extractions: do the layer states catch the paraphrases that the embedding misses? Its outcome decides between the vector hybrids (approaches D and F) and a new identity coordinate (approaches E and G). The decision rule is stated before the check is run (§6).
- **A hybrid answers an engineering question, not the mechanism question.** Beating the embedding with a hybrid shows that the readings complement each other. What the layers add is measured separately, by the residual separation of §3.3.

---

## Contents

1. [Terms](#1-terms)
2. [What the measurements say so far](#2-what-the-measurements-say-so-far)
3. [Theory](#3-theory)
4. [Eight ways to build a hybrid](#4-eight-ways-to-build-a-hybrid)
5. [What a hybrid can and cannot show](#5-what-a-hybrid-can-and-cannot-show)
6. [A staged plan](#6-a-staged-plan)
7. [Questions for the author](#7-questions-for-the-author)
8. [Caveats](#8-caveats)
9. [Sources](#9-sources)

---

## 1. Terms

- **Premise, particle, template.** As in [Template_Matcher_From_Geometry.md](Template_Matcher_From_Geometry.md), §1. A premise is a first sentence joining two clauses $P$ and $Q$, a second sentence, and "Therefore". Each token is a particle. Four templates: MP, MT, DS1, DS2.
- **Embedding reading.** The matcher run on the embedding $h^{(0)}$, the learned word vector plus the position vector. No layer has acted on it. Walked through in [The_Embedding_Template_Matcher.md](The_Embedding_Template_Matcher.md).
- **Layer reading.** The same matcher run on something the trained layers produced: the state $h^{(\ell)}$ after $\ell$ layers, the displacement $\Delta_{\ell}$ of one layer, the well-frame vector, or several of them joined (the path: all displacements; the states: all states after layers). The author calls this the geometry reading. Both readings are geometric, so this note says "embedding reading" and "layer reading".
- **Coordinate groups.** The matcher condenses the cosines and lengths of a premise's particles into $10m + 10$ relational coordinates plus the connective's position. They fall into three groups, which matter for the hybrids below:
  - **match coordinates:** the worst best-matches of the second sentence to each clause and back, the last-particle and mean cosines; they say which clause is restated and how exactly;
  - **odd-particle coordinates:** lengths and neighbours of the second sentence's worst-matched particles; they carry the "not";
  - **frame coordinates:** the six cosines among the means of connective, clauses and "Therefore", and the connective's position on the axis from the disjunctive to the conditional wordings.
- **R1.** Unseen wordings, without "unless": half the mean detection of the four valid forms, half the mean silence on the five fallacy kinds, each wording left out of fitting in turn.
- **Paraphrase at 1%.** The share of subject-keeping paraphrases that fire the right template when the threshold is set so that 1% of near misses fire. A near miss keeps the subject and changes the predicate ("the coffee is strong" against "the coffee is cold"); a paraphrase keeps the meaning ("the coffee is chilled").

## 2. What the measurements say so far

Four two- and four-layer models, the same 60 test clauses, the same matcher. SR2-2B: two layers, pair force, no reverse channel, 2.1B tokens. G2: two layers, pair force, reverse channel, 0.5B. L4: four layers, as G2 otherwise. SR2-NP: two layers, no pair force, no reverse channel, 0.5B.

**Table 1.** R1, unseen wordings without "unless". The best single state is named in brackets.

| model | embedding | best single state | all states | path |
| --- | ---: | ---: | ---: | ---: |
| SR2-2B | 0.956 | 0.963 (after layer 0) | 0.939 | 0.920 |
| G2 | 0.955 | 0.964 (after layer 0) | 0.961 | 0.950 |
| L4 | 0.976 | 0.980 (after layer 1) | 0.907 | 0.887 |
| SR2-NP | 0.952 | 0.909 (after layer 0) | 0.914 | 0.926 |

**Table 2.** Paraphrases caught at 1% near-miss fires, and the AUROC of paraphrase against near miss.

| model | embedding | best single state | all states | path | AUROC, these readings |
| --- | ---: | ---: | ---: | ---: | --- |
| G2 | 0.255 | 0.320 (after layer 0) | 0.248 | 0.219 | 0.755 to 0.779 |
| L4 | 0.177 | 0.334 (after layer 1) | 0.333 | 0.298 | 0.759 to 0.793 |
| SR2-NP | 0.298 | 0.280 (after layer 0) | 0.185 | 0.146 | 0.756 to 0.788 |

On SR2-2B the joint states caught 2 to 8 points more paraphrases than the embedding at the same near-miss rates ([Template_Matchers_on_the_Dynamics_SR2_2B.md](Template_Matchers_on_the_Dynamics_SR2_2B.md)).

**Table 3.** R1 with all twelve wordings, "unless" filed with the disjunctions.

| model | embedding | best single state | all states |
| --- | ---: | ---: | ---: |
| G2 | 0.909 | 0.880 (after layer 0) | 0.844 |
| L4 | 0.838 | 0.923 (after layer 0) | 0.856 |
| SR2-NP | 0.856 | 0.796 (after layer 0) | 0.834 |

With "unless" given clouds of its own, every reading on every model falls to 0.55 to 0.84 ([Connectives_With_a_Hidden_Negation.md](Connectives_With_a_Hidden_Negation.md)).

**Five things these tables say:**

1. **R1 has almost no headroom.** The embedding is at 0.95 to 0.98. A hybrid can gain at most a few points there, and the resampling intervals are about 0.01 wide. R1 is a constraint a hybrid must not break, not a target.
2. **The paraphrase task has the headroom, and the layer states help on some models.** The gain is largest at L4, but there it comes partly from a weaker embedding: the best reading moves from 0.320 (G2) to 0.334 (L4) while the embedding falls from 0.255 to 0.177.
3. **The ceiling is shared.** The ranking of paraphrases against near misses is nearly the same for every reading (AUROC 0.76 to 0.79). The readings differ in how much of that ranking survives at the strict end, not in the ranking itself.
4. **Early states are as good as the embedding on R1; late states are worse.** The state after layer 0 is level with the embedding everywhere except SR2-NP. The final state falls off, most where the reverse channel acts (G2 0.656, L4 0.703).
5. **"Unless" defeats every reading.** No vector read so far puts its connective on the disjunctive side. Choosing a different source does not fix it; a hybrid needs a part that carries polarity.

## 3. Theory

### 3.1 A logical form factors into four parts

Each kind of premise in the test is fixed by four discrete facts about it:

| kind | frame | clause restated | negated | same statement |
| --- | --- | --- | --- | --- |
| MP | conditional | 1 | no | yes |
| MT | conditional | 2 | yes | yes |
| DS1 | disjunctive | 1 | yes | yes |
| DS2 | disjunctive | 2 | yes | yes |
| affirming the consequent | conditional | 2 | no | yes |
| denying the antecedent | conditional | 1 | yes | yes |
| affirmed disjunct | disjunctive | 1 or 2 | no | yes |
| near miss | either | 1 | either | no, same subject |
| other subject | either | 1 or 2 | either | no, same predicate |
| unrelated clause, ordinary text | either | none | | no |

So a template is the conjunction of a **gate** (the second sentence states the same thing as one clause, possibly negated) and a **lookup** on three discrete factors (frame, which clause, polarity). Validity lives in the lookup: in a conditional frame, clause 1 affirmed or clause 2 negated; in a disjunctive frame, either clause negated. Nothing else in the premise matters for the template.

Written as a probability, with $F$ the frame, $C$ the clause restated, $N$ the polarity and $S$ the gate,

$$
\Pr(T \mid y) = \Pr(S = 1 \mid y) \sum_{(f, c, n) \in V_T} \Pr(F = f, C = c, N = n \mid y, S = 1) ,
$$

where $V_T$ is the set of factor values that make template $T$ (one triple for each of the four templates). The current matcher learns this jointly, with one cloud per kind. The factorisation says which parts of the evidence each factor needs:

| factor | what decides it | what the embedding can do | what a layer state could add |
| --- | --- | --- | --- |
| frame | the connective's words | exact for known words; a new word lands by its word vector | context, e.g. the negation hidden in "unless" |
| clause restated | which clause shares tokens with the second sentence | exact for repetition (cosine 0.96 to 1.00 for the same token) | nothing for exact repetition |
| polarity | an explicit "not" or a hidden negation | exact for "not" as a token | hidden negation, if the layers encode it |
| gate | whether a changed predicate means the same | the word vectors' similarity only (0.23 for "chilled" against "cold", 0.16 for "strong") | context-dependent similarity, if the layers build it |

The table predicts where a layer reading can help: the gate and the hidden negation. It also predicts where it cannot: exact repetition, which the embedding already does perfectly and which a layer can only blur. Both predictions fit §2: the layer states help, when they help, on paraphrases; they lose on R1 when they blur repetition (the late states).

### 3.2 The layers add readability, not information

The state after $\ell$ layers is a deterministic function of the embeddings of the premise and its context:

$$
h^{(\ell)}_t = F_{\ell} \left( h^{(0)}_1, \dots, h^{(0)}_t \right) .
$$

By the data-processing inequality, for any property $K$ of the premise (its kind, say), $I(h^{(\ell)}; K) \le I(h^{(0)}_{1:t}; K)$. The layer states cannot know more about the form than the embeddings of the whole sequence do. What they can do is **rearrange** the information so that a weak reader sees it. The matcher is a weak reader by design: it sees only pairwise cosines and lengths of the particles, condensed into fixed coordinates and read by a linear discriminant. Two particles whose embeddings are far apart but whose meanings agree in context can come closer in a layer state, and then the matcher sees an agreement that was invisible in $h^{(0)}$.

So the right question for a hybrid is not "which reading knows more?" but "which reading makes each factor visible to this reader?". This is also why the readings can complement each other even though one is a function of the other.

### 3.3 Combining two readings: what is gained is the residual

Let $a$ be the coordinates from the embedding and $b$ those from a layer reading, and suppose both are Gaussian within each kind with one shared covariance, as the clouds assume. For two kinds $p$ and $q$ let $\delta$ be the difference of their means. The squared Mahalanobis separation of the joint coordinates splits exactly into the separation in $a$ and the separation in **what $b$ has that $a$ does not predict**:

$$
\Delta^2_{ab} = \Delta^2_{a} + \Delta^2_{b \cdot a}, \qquad \Delta^2_{b \cdot a} = \delta_{b \cdot a}^{\top} \Sigma_{b \cdot a}^{-1} \delta_{b \cdot a} ,
$$

$$
\delta_{b \cdot a} = \delta_{b} - \Sigma_{ba} \Sigma_{aa}^{-1} \delta_{a}, \qquad \Sigma_{b \cdot a} = \Sigma_{bb} - \Sigma_{ba} \Sigma_{aa}^{-1} \Sigma_{ab} .
$$

Here $\Sigma_{aa}$, $\Sigma_{bb}$ and $\Sigma_{ab}$ are the blocks of the common within-kind covariance, and $\delta_{b \cdot a}$ is the part of $b$'s mean difference that a regression on $a$ does not explain. Three consequences:

- **The gain from adding a layer reading is $\Delta^2_{b \cdot a}$, never negative in the population.** If the layer reading is a near-linear function of the embedding reading, the residual is near zero and nothing is gained.
- **The gain is measured per pair of kinds.** For the paraphrase task the pair that matters is paraphrase against near miss. A layer reading can add a lot there and nothing for MP against DS1.
- **The residual separation is the clean measure of what the layers add.** It answers the mechanism question (does the dynamics carry form beyond the words?) directly, without a matcher and without a threshold.

**In a finite sample a gain can turn into a loss.** The clouds are estimated from the development premises. For the plain pooled covariance and two kinds with $n_p$ and $n_q$ samples, $N = n_p + n_q$, in $D$ coordinates, the estimated separation is biased upward:

$$
\mathbb{E} \left[ \hat{\Delta}^2 \right] = \frac{N - 2}{N - D - 3} \left( \Delta^2 + D \left( \frac{1}{n_p} + \frac{1}{n_q} \right) \right) .
$$

The fitted clouds look further apart than they are, and the error on new clauses grows with $D / N$. The matcher shrinks the covariance (Ledoit–Wolf) and keeps only $d \le 31$ discriminant directions, which reduces this but does not remove it. The effective sample is closer to the 60 development clauses than to the 6,900 development premises, because premises on one clause pair share most of their coordinates. Doubling the coordinates by joining two readings is worth it only when the residual separation is larger than the noise it brings, roughly $D$ divided by the number of clauses for an added block of $D$ coordinates.

### 3.4 Combining two scores

Each matcher gives, for template $T$, a rate $\lambda_T$ and so a log odds $\ell_T = \log \lambda_T - \log (1 - \lambda_T)$. If the two readings were independent given the kind, the Bayes combination would be the sum:

$$
\ell_T^{\mathrm{hyb}} = \ell_T^{a} + \ell_T^{b} - \log \frac{\pi_T}{1 - \pi_T} ,
$$

with $\pi_T$ the prior share of the template's kinds: the prior is in both scores, so it is taken out once. Stacking replaces the two unit weights of the scores by weights fitted on development clauses. They are not independent ($h^{(\ell)}$ is computed from $h^{(0)}$), so the plain sum counts shared evidence twice and is overconfident. Two remedies: fit $w_a$ and $w_b$ on the development clauses (stacking), or set the threshold of the summed score so that 1% of near misses fire, which absorbs a common scale. Score fusion cannot exploit a relation between the two readings within one premise (for example, "the embedding says clause 1, the state says same statement"); only the coordinate-level hybrids can.

### 3.5 The paraphrase ceiling

The gate for a paraphrase is read through the match coordinates: the cosine between the changed predicate of the second sentence and the predicate of the clause it restates. Call these $s^{(0)}$ in the embedding and $s^{(\ell)}$ in a layer state. The near miss has its own pair of values. The matcher can tell a paraphrase from a near miss only through how these cosines are ordered.

**A testable claim.** If $s^{(\ell)}$ is close to an increasing function of $s^{(0)}$ over the paraphrase and near-miss pairs, so that the ranks of the pairs agree, then the layer reading orders paraphrases against near misses as the embedding does. The AUROC is then the same, which is what Table 2 shows (0.76 to 0.79 everywhere). In that case:

- no hybrid of vectors can raise the ceiling, because every reading misses the same paraphrases;
- the gains of Table 2 at 1% come from the shape of the clouds near the threshold (how tightly the near misses cluster), not from a better gate;
- the fix must come from a coordinate or metric that orders the pairs differently (approaches E and G below).

If instead the ranks disagree, so that the layer state brings some paraphrases above their near misses that the embedding leaves below, a hybrid can catch both sets, and approaches D and F are worth building. The rank agreement of $s^{(0)}$ and $s^{(\ell)}$, and the overlap of the paraphrases each reading catches, are what step 0 of §6 measures.

### 3.6 Coordinates must not mix spaces

A tempting hybrid is to take some particles from the embedding and others from a layer state, for instance the connective from $h^{(0)}$ and the clause particles from $h^{(1)}$, and build one cosine matrix. This does not work. The two vectors live in different spaces, each whitened by its own statistics of ordinary text, and the cosine between an embedding vector and a state vector has no meaning. A role split must therefore be made **at the level of coordinate groups**: each group of §1 is computed entirely within one reading, and the groups are then joined. This is the form approach D takes.

## 4. Eight ways to build a hybrid

The approaches run from the cheapest, which need nothing but the saved extractions, to one that changes the model. For each: what it is, the theory behind it, pros and cons, and what it needs.

### A. Score fusion

**What.** Run the embedding matcher and one layer matcher as they are. Combine their log odds per template (§3.4): a plain sum, a sum with weights fitted on the development clauses, or the larger of the two. Set the threshold of the combined score by the 1% rule.

**Variant A3, a cascade.** Let the embedding decide the frame and the clause, and let the layer reading decide only the gate. For example, fire when the embedding's template is $T$ and either reading's gate is passed. This is a step toward F.

| pros | cons |
| --- | --- |
| Cheapest: no change to the matcher, only to the decision. | Counts shared evidence twice unless weighted (§3.4). |
| Each half stays interpretable and is already reported. | Cannot use relations between readings within a premise. |
| Can be run on all four models from the saved extractions. | An "either fires" rule raises near-miss fires; the threshold must be reset, which can cancel the gain. |
| A clean bound: if fusion gains nothing, the coordinate hybrids are unlikely to. | Weights fitted on 60 clauses are noisy. |

### B. Joined coordinates

**What.** Put the embedding's and the layer reading's coordinates side by side, about 180 numbers, and fit one set of clouds on them. The search over $m$ and $d$ is unchanged.

**Theory.** In the population this is the full Bayes combination of the two readings under the Gaussian model: it gets $\Delta^2_{ab}$ (§3.3).

| pros | cons |
| --- | --- |
| Uses every relation between the two readings. | Doubles the coordinates on about 60 effective samples: the finite-sample cost of §3.3 is largest here. |
| No design choice about which reading does what. | The joined "all states" reading already behaves like this and is not better than its best single state (Table 1, L4: 0.907 against 0.980). That is a warning. |
| One extra line in the analysis code. | The directions mix readings, so it is hard to say what each one reads. |

### C. Embedding plus residual

**What.** Regress the layer reading's coordinates on the embedding's, on the development premises, and keep the residual. Fit the clouds on the embedding's coordinates plus the leading directions of the residual.

**Theory.** It gets the same population separation as B, but it puts the decomposition $\Delta^2_{a} + \Delta^2_{b \cdot a}$ in plain view. The residual can be cut to its few directions that separate kinds, which controls the finite-sample cost.

| pros | cons |
| --- | --- |
| Answers the mechanism question at the same time: the residual separation per pair of kinds is what the layers add. | The regression is fitted on the same 60 clauses and adds its own noise. |
| Fewer added coordinates than B, chosen by their separation. | Linear residuals miss non-linear relations between the readings. |
| Interpretable: each residual direction is "what the state says beyond the words". | The residual of a state that only blurs repetition can look like signal on the development clauses. Fresh clauses are needed. |

### D. A role split by coordinate group

**What.** Compute the frame coordinates (node cosines, connective position) and the odd-particle coordinates from the embedding, and the match coordinates from a layer state, each within its own space (§3.6). Join and fit as usual.

**Theory.** It follows the factorisation of §3.1: the frame and the explicit "not" are lexical, so they come from the embedding; the gate is where a layer state can help (Table 2), so the matches come from the state. It is B with the coordinates the theory says are useless removed.

| pros | cons |
| --- | --- |
| No more coordinates than one reading: the finite-sample cost stays as it is. | It throws away the embedding's exact repetition, which the match coordinates carry. Late states blur it (Table 1). A variant keeps both match groups, at about 1.5 times the coordinates. |
| Each part has a stated job, which reads well in the book. | The best state for matches differs by model (after layer 0 on G2, after layer 1 on L4). The choice must be made on development clauses. |
| Tests the factorisation directly: if it works, the theory of §3.1 is right about where each reading helps. | Gains nothing if the ceiling of §3.5 holds. |
| Small change to `template_depth_analysis.py`, no new extraction. | Does nothing for "unless". |

### E. Two metrics in the match coordinates

**What.** Keep the embedding reading for everything, but compute every match coordinate twice, once with the embedding's cosine and once with a layer state's, or take the larger of the two. Exact repetition then stays at about 1 through the embedding, and a paraphrase can rise through the state.

**Theory.** The gate depends on how the pairs are ordered (§3.5). The larger of two cosines orders a pair high if either reading does, which is the "either reading's gate" rule of A3 inside the coordinates.

| pros | cons |
| --- | --- |
| Targets the one task with headroom, the paraphrase gate. | The near miss rises too whenever the state brings unrelated predicates closer. Whether it rises less than the paraphrase is exactly the question of §3.5. |
| Keeps the embedding's exact repetition, so R1 should not suffer. | The larger-of-two rule is not smooth, and it has no Gaussian justification. |
| About 1.5 times the coordinates if both are kept, no more than one reading if the larger is taken. | Gains nothing if the ranks agree. |

### F. Factorised templates

**What.** Replace the one-cloud-per-kind matcher by four small readers, one per factor of §3.1:

- frame: conditional or disjunctive, from the frame coordinates;
- clause: 1, 2 or none, from the match coordinates;
- polarity: negated or not, from the odd-particle coordinates, plus a polarity coordinate for hidden negations ([Connectives_With_a_Hidden_Negation.md](Connectives_With_a_Hidden_Negation.md), §8);
- gate: the same statement or not, from the match coordinates of the restated clause.

Each reader has its own source (embedding or a layer state), chosen on development clauses. The template fires by the lookup of §3.1.

**Theory.** If the factors are conditionally independent given the premise's coordinates, the product formula of §3.1 is exact. Each reader separates two or three classes instead of 32 kinds, so it needs far fewer parameters, and the finite-sample cost falls.

| pros | cons |
| --- | --- |
| Matches the structure of the forms: validity is a lookup, not a cloud. | Errors multiply across four readers; a reader at 95% each gives about 81% if errors are independent. |
| Gives "unless" a place: a polarity reader can take a coordinate that the frame reader does not, so "unless" can be a conditional frame with a hidden negation. | The factors are not independent in the coordinates (the "not" moves the match coordinates, [The_Embedding_Template_Matcher.md](The_Embedding_Template_Matcher.md), §8). |
| Fits the Doi–Peliti picture of templates as reactions: the reactants (frame, clause, polarity) and the gate are separate conditions. | Labels per factor are needed. They exist by construction for the test premises, but a natural text has none. |
| Generalises by composition: a new wording only needs the frame reader to place it. | The largest rewrite of the matcher among the eight. |
| Few parameters per reader: fits the 60 clauses better. | Departs from the author's original requirement of a separate template per form; the templates now share readers. |

### G. A learned identity metric

**What.** Learn a metric for the gate alone. On the development clauses, fit a linear map $M$ of the predicate particles' vectors such that paraphrase pairs come closer than near-miss pairs, for instance by the discriminant of the pair differences. Use the cosine under $M$ in the match coordinates. The source of the vectors can be the embedding, a layer state, or both stacked.

**Theory.** This is the direct answer if the ceiling of §3.5 holds: the pairs need a new order, and a metric fitted for that purpose gives one. It is the only approach that can raise the AUROC rather than move the operating point.

| pros | cons |
| --- | --- |
| Attacks the ceiling directly. | Supervised on paraphrases drafted by the agent; it may learn the drafting style. |
| Small: one map for the predicate particles. | Every paraphrase of the development clauses is spent on fitting the metric; scoring needs fresh paraphrases. |
| Works with any source, so it can also measure which reading's geometry is easier to correct. | It moves the matcher away from "geometry of the trained model" toward "a model trained on top of it". The book should say so. |
| | With 60 clauses the metric must be strongly regularised (low rank). |

### H. A template unit in the model

**What.** Build the hybrid into the forward pass: a template unit reads the current state and the embedding, carried along like a skip connection, and fires into a register or the particle store. Trained end to end.

**Theory.** It removes the question of which reading to use: the unit learns it. The embedding skip costs one stored vector per token.

| pros | cons |
| --- | --- |
| The target design: templates acting inside the model, not read from outside. | GPU training; behind the productization queue. |
| The skip is cheap and fits the register and store designs. | Not a measurement: what the unit learns must itself be probed. |
| The model can learn its own gate and its own hidden negations. | Needs a training signal for firing that does not exist yet. |

### Summary

| approach | coordinates | needs | targets | main risk |
| --- | --- | --- | --- | --- |
| A, score fusion | as now | saved extractions | the operating point | double counting |
| B, joined | about twice | saved extractions | everything | overfitting on 60 clauses |
| C, residual | as now plus a few | saved extractions | the mechanism measure | regression noise |
| D, role split | as now | saved extractions | the gate | loses exact repetition |
| E, two metrics | up to 1.5 times | saved extractions | the gate | near misses rise too |
| F, factorised | four small readers | rewrite of the matcher | the gate and "unless" | errors multiply |
| G, learned metric | as now | paraphrase supervision | the ceiling | learns the drafting |
| H, in the model | | GPU training | the target design | needs a firing signal |

## 5. What a hybrid can and cannot show

Two questions have been asked of the matchers, and a hybrid answers only one of them.

- **The engineering question:** what is the best template matcher on this model? A hybrid that fires on more paraphrases at 1% near misses, with R1 kept, answers it.
- **The mechanism question:** do the trained dynamics carry a signature of the form that the words do not? A hybrid that beats the embedding does not answer it, because it may win by using the embedding better. The measure for this question is the residual separation $\Delta^2_{b \cdot a}$ of §3.3, read per pair of kinds, or approach C's residual directions.

The calls of [Template_Matchers_on_the_Dynamics.md](Template_Matchers_on_the_Dynamics.md) and [Template_Matchers_and_Depth.md](Template_Matchers_and_Depth.md) asked the mechanism question. Any write-up of a hybrid, in the notes or in the book, should keep the two answers apart.

## 6. A staged plan

**Step 0: complementarity on the saved extractions (minutes per model, CPU).** For G2, L4, SR2-NP and SR2-2B, on the test clauses, at the 1% near-miss threshold of each matcher:

1. the share of paraphrases caught by the embedding only, by the best layer reading only, by both, by neither;
2. the same for the near misses that fire;
3. the rank correlation of $s^{(0)}$ and $s^{(\ell)}$ (§3.5) over the paraphrase and near-miss pairs;
4. the residual separation $\Delta^2_{b \cdot a}$ for paraphrase against near miss, and for MP against affirmed disjunct as a control where the theory predicts no gain.

**Decision rule, proposed before the check (the author to confirm).** Let "state only" be the share of paraphrases that the layer reading catches and the embedding misses.

- If "state only" is at least 0.05 on two models or more, the readings complement each other: build D (with the variant that keeps both match groups) and, if it gains, F.
- If it is below 0.05 on every model and the rank correlation is above 0.8, the ceiling of §3.5 holds: vector hybrids are capped, and the next step is E, then G.
- Otherwise, run A and C as the cheap bound and decide again.

**Step 1: the chosen hybrid on the saved extractions.** Scored on the same test clauses. These clauses have been read for every matcher so far, so the result is **post hoc** and only ranks the designs.

**Step 2: pre-registration and fresh clauses.** Calls written for the chosen design, confirmed by the author, scored on clauses no matcher has seen. Either the 60 fresh clauses after the author's check, or a new set. This is the only step whose result can be called a finding.

**Step 3: "unless" and hidden negation.** Only F has a place for a polarity reader. Whether any reading can supply hidden negation is a separate question ([Connectives_With_a_Hidden_Negation.md](Connectives_With_a_Hidden_Negation.md), §8), and the eight-layer runs, when they land, are one place to look.

**Step 4: in the model (H).** After the template programme has a firing signal, and after the productization queue.

**My recommendation.** Run step 0 first: it decides between two families of approach and costs almost nothing. My expectation, stated now so it can be checked: the ranks largely agree (correlation above 0.8), and "state only" is small on SR2-NP and moderate on L4. That would put E or G ahead of D. F is the design I think is most worth building in the long run, because it is the only one that makes the structure of the forms explicit and gives "unless" a place. It should wait until step 0 shows which reading feeds its gate.

## 7. Questions for the author

1. **Which question is the hybrid for?** If it is the engineering question, D, E or F. If it is the mechanism question, C and the residual separation, with the hybrid as a by-product.
2. **Is supervision on paraphrases acceptable?** G learns a metric from paraphrase and near-miss pairs. That is still geometry, but it is no longer only the trained model's geometry.
3. **Can templates share readers?** F replaces "a separate template per form" by four shared readers and a lookup. It is a change to the requirement you set in [Template_Matcher_From_Geometry.md](Template_Matcher_From_Geometry.md), §2.
4. **The clause check.** Step 2 needs the 60 fresh clauses checked, or a new set drafted.

## 8. Caveats

- No hybrid has been run. The numbers in §2 are from the reports listed in §9; the theory in §3 is standard (the Mahalanobis decomposition, the bias of the estimated separation, the Bayes combination of independent scores), applied to this matcher by the agent.
- The bias formula of §3.3 is for the plain pooled covariance. The matcher shrinks it and projects onto a few directions, so the formula gives the direction of the effect, not its size.
- The four models differ in several ways at once (pair force, reverse channel, depth, tokens). A hybrid that works on one may not work on another, and the model differences in Table 2 are not explained.
- The clauses, paraphrases and near misses were drafted by the agent and await the author's check. A paraphrase that is really a near miss, or the other way round, moves the ceiling of §3.5.
- "Unless" is excluded from R1 throughout. The hybrids do not address it except through F.

## 9. Sources

| source | what it gives |
| --- | --- |
| [Template_Matcher_From_Geometry.md](Template_Matcher_From_Geometry.md) | the matcher, its coordinates, the clouds, the rate |
| [The_Embedding_Template_Matcher.md](The_Embedding_Template_Matcher.md) | the embedding reading step by step; the coordinate that decides repetition |
| [What_the_Matchers_Read.md](What_the_Matchers_Read.md) | why the embedding is the control |
| [Template_Matchers_on_the_Dynamics_SR2_2B.md](Template_Matchers_on_the_Dynamics_SR2_2B.md) | SR2-2B readings, paraphrases at fixed rates |
| [Template_Matchers_and_Depth_G2.md](Template_Matchers_and_Depth_G2.md), [Template_Matchers_and_Depth_L4.md](Template_Matchers_and_Depth_L4.md) | G2 and L4 readings |
| [Connectives_With_a_Hidden_Negation.md](Connectives_With_a_Hidden_Negation.md) | "unless" and the polarity coordinate |

The numbers of Tables 1 to 3 are in `notebooks/conservative_arch/scaleup/results/template_matcher/`: `template_depth_{g2,l4,np}.json` and `.txt`, and `template_dynamics_without_unless_sr2_2b.json`. The SR2-NP readings have no report of their own yet; the report follows VF0.
