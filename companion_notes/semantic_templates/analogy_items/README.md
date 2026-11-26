# Analogy and In-Context Test Set: Draft Items

**What this folder is.** The first 20 frames per level of the test set designed in [Analogy_and_Context_Test_Set.md](../Analogy_and_Context_Test_Set.md), drafted by the agent on 2026-11-26 for the author's review. Nothing has been extracted or scored. The source of every sentence is `notebooks/conservative_arch/scaleup/debug/template_analogy_clauses.py`; these pages are generated from it by `notebooks/conservative_arch/scaleup/debug/template_analogy_review.py`, so corrections go into the source and the pages are regenerated.

**How the premises are built.** Each frame gives the clauses and, at its level, the restatements and foils of the second sentence. The six wordings (three conditional, three disjunctive) and the forms are composed as in the current test: modus ponens is a conditional with a restatement of P, modus tollens a conditional with a denial of Q, the disjunctive syllogisms a disjunctive with a denial of P or of Q, and the fallacy twins use the other clause. For the clause-level banks (L0 to L3, L5, L6) the second clause is another frame of the same bank.

**The matched-overlap rule.** Each restatement has a foil that shares exactly as many tokens with the clause (GPT-2 tokens, lower-cased), so token identity cannot prefer the restatement. The check below is computed, not judged by eye.

| level | page | frames | pairs checked | pairs that break the rule |
| --- | --- | ---: | ---: | ---: |
| L0 verbatim, L1 one word, L2 full paraphrase | [L0_L1_L2_base.md](L0_L1_L2_base.md) | 20 | 80 | 0 |
| L3 entailed case | [L3_entailed.md](L3_entailed.md) | 20 | 40 | 0 |
| L4 reference | [L4_reference.md](L4_reference.md) | 20 | 60 | 0 |
| L5 restructured | [L5_restructured.md](L5_restructured.md) | 20 | 40 | 0 |
| L6 new domain | [L6_new_domain.md](L6_new_domain.md) | 20 | 40 | 0 |
| C1 alias defined in context | [C1_alias.md](C1_alias.md) | 20 | 40 | 0 |
| C2 same word, sense differs; C2′ paraphrase against a false friend | [C2_sense.md](C2_sense.md) | 20 | 40 | 0 |
| C3 ellipsis | [C3_ellipsis.md](C3_ellipsis.md) | 20 | 40 | 0 |
| C4 denial by an incompatible value | [C4_denial_by_context.md](C4_denial_by_context.md) | 20 | 40 | 0 |

**What the author is asked to check, for every level:** that each restatement is the same statement as its clause (or implies it, at L3), that each foil is clearly a different statement and is neither implied nor excluded by the clause, and that nothing reads unnaturally. Items can be struck, replaced or rewritten; the overlap check reruns on the new text.

**Known limits of the draft.**
- C1 and C4 test the second sentence about one clause; the other forms of the frame (those that restate or deny the other clause) fall back on verbatim sentences.
- C4 relies on general knowledge (green denies red). A variant in which a context sentence sets the incompatibility is not drafted yet.
- L4 and C3 use gendered pronouns, which need the model to know the gender of "the king" or "the bride"; the embedding may carry part of that, which the readings would then show.
- All items are by one drafter, so the foils may share a style the restatements lack. The author's check and a later lexical check on style words limit this.
