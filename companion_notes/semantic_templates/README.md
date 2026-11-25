# Semantic templates: Index

**What this folder is.** Notes on building semantic templates into the conservative language models of the Semantic Simulation framework: the mathematics of where and when templates can be found, and the measurements that test it, one topic per file. The programme's plan and its Doi–Peliti formulation stay at the top level of `companion_notes/` (their links are frozen in the book); new notes on the programme go here. (Index written 2026-11-24.)

## The notes

| note | topic |
| --- | --- |
| [Admissible_Regions_for_Template_Search.md](Admissible_Regions_for_Template_Search.md) | where in semantic space and when in training a template search may start: the three motions of a training model (wells, coordinates, context), the well frame and its invariance, the context spread of a well, training drift measured without coordinates, the admissibility criterion and the bound it buys; the settling map of SR2-NP's three checkpoints, calls AR1 to AR4 |
| [Well_Settling_Map_SR2_NP.md](Well_Settling_Map_SR2_NP.md) | technical report of the first measurement (CPU, 2026-11-24): SR2-NP at steps 500, 22,500 and 31,000 on 8,192 validation tokens; definitions, setup, code, four figures. AR1 and AR2 hit, AR3 and AR4 miss. The live well keeps its tokens through the decay, but their order inside it has a rank agreement of only about 0.5, so no region is admissible over that window; the metric spread separates what the Euclidean spread confuses |
| [Logical_Forms_as_Geometric_Figures.md](Logical_Forms_as_Geometric_Figures.md) | preliminary experiment (CPU, 2026-11-25): modus ponens, modus tollens, disjunctive syllogism and two fallacies through four trained models; a form frozen as 21 distances identifies new arguments 85 to 88% of the time and transfers between models, but it records the sentence frame, the restatements and negation; no validity signal found, at 0.5B or 2B tokens |
| [Template_Matcher_From_Geometry.md](Template_Matcher_From_Geometry.md) | report (CPU, 2026-11-25): a two-part matcher that fires a logical form's template from the geometry of the premise, one cloud per kind in a 10-dimensional subspace, with the clouds written out; fires on 92 to 97% of known wordings and 72 to 78% of unseen ones, false fires on 2%, mostly the fallacies; what training adds and what the embedding already has |
| [Template_Matchers_on_the_Dynamics.md](Template_Matchers_on_the_Dynamics.md) | pre-registration (2026-11-25): template matchers that read what the layers did to the premise's particles, the displacement and the well-frame reading, against the same matcher on the embedding; calls TD1 to TD5 proposed, awaiting confirmation; extraction queued on SR2-2B |

## Related notes elsewhere

| note | topic |
| --- | --- |
| [Implementing_Semantic_Templates_A_Path.md](../Implementing_Semantic_Templates_A_Path.md) | the staged plan: why hardcoding fails, the three tiers, the particle store, the staging with pass criteria |
| [Semantic_Templates_as_Doi_Peliti_Reactions.md](../Semantic_Templates_as_Doi_Peliti_Reactions.md) | templates as the reaction rules of the Doi–Peliti dynamics |
| [The_Particle_Store_Making_Allocation_Real.md](../The_Particle_Store_Making_Allocation_Real.md) | the write-once particle store that makes creation real, and its short training run |
| [Community_of_Semantic_Templates_for_Native_CoT.md](../Community_of_Semantic_Templates_for_Native_CoT.md) | a thought as a sequence of firings; the MT probe |
| [Renormalization_Group_Readings_of_Template_Matching_A_Proposal.md](../Renormalization_Group_Readings_of_Template_Matching_A_Proposal.md) | scale axes of template matching, proposed |
| [Dead_Wells_in_the_Joint_V_theta_Bank.md](../Dead_Wells_in_the_Joint_V_theta_Bank.md) | the well probe the settling map builds on |
