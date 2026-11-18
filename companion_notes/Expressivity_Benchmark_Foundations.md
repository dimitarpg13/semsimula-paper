# Expressivity Benchmark: Foundations

**What this note is.** The foundations of a benchmark that would test the programme's capability claim: that semantic templates, built on the conservative core, lift a model higher on the expressivity ladder than a matched attention transformer reaches at matched cost. It fixes what the claim is, what a fair test of it must look like, which task families measure it, and what remains to be decided. Nothing in it is registered. The calls are written when the templates arm is specified and before it trains. (Draft for discussion, 2026-11-17.)

**Status.** Foundations only. No task generated, no model trained, no call proposed.

---

## Contents

0. [Why a benchmark, and why now](#0-why-a-benchmark-and-why-now)
1. [The claim under test](#1-the-claim-under-test)
2. [Principles for a fair test](#2-principles-for-a-fair-test)
3. [Task families, by rung of the ladder](#3-task-families-by-rung-of-the-ladder)
4. [Protocol skeleton](#4-protocol-skeleton)
5. [What would count as the payoff, and what would not](#5-what-would-count-as-the-payoff-and-what-would-not)
6. [Dependencies](#6-dependencies)
7. [Open questions for discussion](#7-open-questions-for-discussion)
8. [References](#8-references)

---

## 0. Why a benchmark, and why now

Two findings of 2026-11-17 set the question.

- **Cost is not the case for the architecture.** On a CPU the Gen 3 conservative core is slower than a matched GPT-2 at every measured length. Its decoding state grows with the context: the content-routed context channels and the routed pair force both need a per-token cache ([protocol](depth_ladder/CPU_Inference_Benchmark.md), the CPU benchmark and the inference-complexity audit). Reaching cost parity is engineering, worth doing. It is not a reason to build the model.
- **What the construction delivers today is interpretability and testability.** Its parts are named and closed-form, so mechanism hypotheses become controlled interventions, and several of them have failed cleanly ([protocol, `p-bank-memory-conjecture`](depth_ladder/Second_Order_Memory_Probes.md)). That is a payoff for understanding, not for capability.

The capability payoff the programme stakes is the expressivity ladder: templates that create structure during inference. A claim of that kind needs a benchmark fixed before the model that would pass it exists, or the benchmark gets chosen to fit the model.

## 1. The claim under test

**The floor.** The conservative core without creation, the v0 simulator, is a deterministic finite automaton at any fixed precision, and it accepts exactly the regular languages ([bounds note, Lemma 4](Expressivity_Bounds_For_v0_Simulator.md#5-lemma-4--chomsky-hierarchy-placement)). No parameter count, force vocabulary or training signal changes that, because none of them grows the state space during inference. A bank of wells is no exception: a Poisson mode deepens a potential and adds no degree of freedom ([templates path, §1.1](Implementing_Semantic_Templates_A_Path.md#11-and-a-bank-of-wells-is-not-enough-either)).

**The lift.** Templates lift the class only if a firing instantiates a particle ([templates path, §1.1](Implementing_Semantic_Templates_A_Path.md#11-and-a-bank-of-wells-is-not-enough-either)). With a write-once particle store (Stage 0″) the composite model is argued to reach the mildly context-sensitive class, the class of bounded linear context-free rewriting systems ([MCS reduction](MCS_Reduction_For_v3_Composite.md)).

**The comparison.** The claim is relative, and it is about **generalisation**, not about what a model can represent at the lengths it trained on. A transformer with a key–value cache and enough width can represent many structured computations on the lengths it sees. Its documented weakness is extrapolation: tasks whose solution needs unbounded structure, tested at lengths or depths beyond training (Delétang et al. 2023; Hahn 2020; Zhou et al. 2023). The claim, stated so it can fail:

> On tasks whose solution requires a stack, a copy or cross-serial memory of unbounded size, a template model trained on short inputs generalises to longer and deeper inputs where a matched transformer does not; the conservative core without templates fails on the same tasks, at the depth the bounds note predicts.

Each part can fail separately. The transformer may generalise too, in which case templates buy nothing over attention. The template model may not generalise, in which case creation is not learned. Or the core alone may already succeed, in which case the lift is not due to templates.

## 2. Principles for a fair test

1. **Generalisation splits, not in-distribution accuracy.** Train on lengths or depths up to $N$; test on a grid up to $4N$ to $10N$. In-distribution accuracy is reported but is not the score.
2. **A strong transformer, not a strawman.** Matched parameters and matched training compute. Several positional schemes (learned, rotary, ALiBi, none), because length generalisation depends on them. Scratchpad or chain-of-thought variants reported separately, since they change what is being compared. Tuning budgets equal across models.
3. **The conservative core as a control.** The same core without templates, so that any lift is attributable to creation. The bounds note's Dyck protocol is its floor test (§3, family B).
4. **Pre-registration.** For each task: the split, the metric, the threshold for "generalises", and a probability for each model's outcome, all confirmed before the templates arm trains, as for every call in the [protocol](depth_ladder/README.md).
5. **Seeds and spread.** At least three seeds per model and task. Length generalisation is known to vary strongly across seeds.
6. **Cost reported alongside.** Training and inference FLOPs, peak memory and the decoding state per token, measured as in the protocol's CPU benchmark. A capability bought at a large cost is still reported, with its cost.
7. **Report the losses too.** Every task where the transformer wins is reported in the same table.

## 3. Task families, by rung of the ladder

Each family is chosen for one structural requirement. The ones marked *baseline-known* have published transformer results to calibrate against.

| family | rung | what the solution needs | examples |
| --- | --- | --- | --- |
| A. regular controls | regular | a finite state | parity, modular counting, cyclic navigation (baseline-known) |
| B. stack | deterministic context-free | a stack of unbounded depth | Dyck languages at controlled depth (the bounds note's falsifier), string reversal, stack manipulation (baseline-known) |
| C. copy and cross-serial | mildly context-sensitive | a copy of unbounded length; crossed dependencies | duplicate string (a string followed by its copy), counts crossed as a-b-c-d with the a and c counts equal and the b and d counts equal, cross-serial agreement |
| D. compositional language | structural generalisation | rules applied to novel combinations and lengths | SCAN length split, COGS structural splits (baseline-known) |
| E. typed structure-building | beyond strings | building and querying a typed structure | nested arithmetic evaluation, parse-tree construction from tokens |
| F. bridge to natural language | agreement at depth | a dependency across nested clauses | subject–verb agreement across nested relative clauses at controlled depth |

**Family A** guards against a false win: a finite-state task that both models should solve. If the template model fails it, something other than expressivity is wrong. Parity is a known weakness of transformers at length (Hahn 2020), so A also calibrates the baseline. **Family B** carries the bounds note's quantitative prediction: the core alone collapses at a depth $D^\ast$ of order $\log_n |Q|$, and the template model must not. **Family C** is the rung that separates the claimed class from context-free models such as a stack-augmented network. **Families D to F** keep the benchmark connected to language, where the programme's models are trained.

## 4. Protocol skeleton

- **Data.** Generated synthetically for A to C and E, with a fixed generator, seed and train/test split by length and depth; the published splits for D; a templated generator for F. Every generator is committed before training.
- **Models.**
  - The template model: the conservative core with the Stage 0″ store and a Tier 1 template bank.
  - The core alone: the same core without templates.
  - Transformers at matched parameters and compute, one per positional scheme.
  - A recurrent baseline (LSTM) and, for families B and C, a stack-augmented recurrent model, as the reference for "has the right memory".
- **Scores.** Per-position accuracy on the positions that need the structure, such as the closing bracket's type in Dyck as in the bounds note, and sequence accuracy. Both are reported against length and depth on the test grid.
- **Training budget.** Fixed in tokens and in steps across models, with early stopping on an in-distribution validation set only.
- **Analysis.** Per task, the largest length or depth at which each model stays above the registered threshold, with the spread over seeds.

## 5. What would count as the payoff, and what would not

**The payoff.** On families B and C, the template model holds above threshold to a depth or length at least, say, twice the best transformer's, on at least two tasks per family. The core alone collapses near the predicted $D^\ast$. Family A shows no deficit. The numbers here are placeholders for the registered thresholds.

**Not the payoff:**
- a win on in-distribution accuracy only;
- a win that disappears against one positional scheme;
- a win with one seed;
- a win where the core alone does as well (no lift);
- a win bought by a scratchpad the transformer was not given.

**A partial payoff** is still informative and reported as such. Examples: templates beat the transformer on B but not on C (the context-free rung reached, the mildly context-sensitive one not), or the template model generalises but at much higher cost.

## 6. Dependencies

- **The particle store (Stage 0″).** A first build is in the code and verified; nothing is trained at scale ([templates path, §7](Implementing_Semantic_Templates_A_Path.md#7-staging-with-pass-criteria)).
- **The Tier 1 template algorithm** (bank learned in training, fired at inference): its design is open and waits on the author.
- **The Dyck depth protocol.** Its first faithful run is the MCS note's Phase 1b ([Augmenting PARFLM](Augmenting_PARFLM_to_handle_MCS_Languages.md)), which measures $D^\ast$ for the core and for the register-pool model. Its result calibrates family B before any template model exists.
- **A small-model training harness** for synthetic tasks, shared by every model in the comparison.

## 7. Open questions for discussion

1. **Which rung is the headline?** The context-free rung (B) is cheaper to reach and to test. The mildly context-sensitive rung (C) is what the MCS reduction claims, and what would separate templates from a stack-augmented network.
2. **Synthetic first, or language first?** Synthetic tasks isolate the mechanism; families D to F connect to the corpora the programme trains on. One option is B and C first, F as the bridge.
3. **What does "matched cost" mean** for a model whose state grows by creation? Matched parameters, matched training FLOPs, or matched inference memory at the test length.
4. **How strong should the baseline be?** Transformers with explicit length-generalisation aids (scratchpads, specific positional encodings) are a moving target. The benchmark should fix a set and a date.
5. **What scale?** Small models make many seeds affordable. The claim is ultimately about scale ("and that it scales"), so at least one task should be repeated at a second size.
6. **Where does Tier 2 come in?** Creating templates at inference would be a further rung. The benchmark should leave room for it without testing it yet.

## 8. References

- Delétang, G., Ruoss, A., Grau-Moya, J., Genewein, T., Wenliang, L. K., Catt, E., Cundy, C., Hutter, M., Legg, S., Veness, J. and Ortega, P. A. (2023). Neural networks and the Chomsky hierarchy. *ICLR 2023*.
- Hahn, M. (2020). Theoretical limitations of self-attention in neural sequence models. *TACL* 8.
- Hewitt, J., Hahn, M., Ganguli, S., Liang, P. and Manning, C. D. (2020). RNNs can generate bounded hierarchical languages with optimal memory. *EMNLP 2020*.
- Kim, N. and Linzen, T. (2020). COGS: A compositional generalization challenge based on semantic interpretation. *EMNLP 2020*.
- Lake, B. M. and Baroni, M. (2018). Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks. *ICML 2018*.
- Zhou, H., Bradley, A., Littwin, E., Razin, N., Saremi, O., Susskind, J., Bengio, S. and Nakkiran, P. (2023). What algorithms can transformers learn? A study in length generalization. arXiv:2310.16028.
