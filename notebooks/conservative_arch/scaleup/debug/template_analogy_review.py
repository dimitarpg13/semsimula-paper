#!/usr/bin/env python3
"""Review pages and the matched-overlap check for the draft analogy item bank (template_analogy_clauses.py), 2026-11-26.
    python3 template_analogy_review.py OUTDIR
For every restatement and its foil, the number of the sentence's tokens (GPT-2, lower-cased, leading space stripped) that occur in the
clause it is compared with. The design rule: the two counts are equal, so token identity cannot prefer the restatement. C2' is the
exception by design (the false friend shares more). Writes OUTDIR/README.md and one page per level, and prints the check."""
import sys
from pathlib import Path
DBG = Path(__file__).resolve().parent; sys.path.insert(0, str(DBG))
import tiktoken
from template_analogy_clauses import LEVELS
enc = tiktoken.get_encoding('gpt2'); OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
tok = lambda s: [enc.decode([t]).strip().lower() for t in enc.encode(' ' + s)]
ov = lambda s, ref: sum(x in set(tok(ref)) for x in tok(s))
cap = lambda s: s[0].upper() + s[1:]
SRC = 'notebooks/conservative_arch/scaleup/debug/template_analogy_clauses.py'

# per level: title, file, what it tests, what to check, columns, and the matched pairs (restatement field, foil field, reference field)
SPEC = {
    'BASE': ('L0 verbatim, L1 one word, L2 full paraphrase', 'L0_L1_L2_base.md',
             'One bank of 20 clauses read at three distances from the words, so the fall from L0 to L2 is paired by clause. L0 restates the clause word for word against a near miss (the current test). L1 changes one word of the predicate (subject kept) against the same near miss. L2 replaces the subject and the predicate by paraphrases, against a foil with the same paraphrased subject and an unrelated predicate: neither shares a content word with the clause.',
             ['Is each L1 paraphrase the same statement as the clause (not stronger, not weaker)?', 'Is each L2 paraphrase the same statement, with no word of the clause left?', 'Is each foil clearly a different statement, and not implied or excluded by the clause?'],
             [('clause', 'aff'), ('near miss (L0, L1 foil)', 'near'), ('L1 paraphrase', 'syn'), ('L2 paraphrase', 'para'), ('L2 foil', 'pfoil')],
             [('L1', 'syn', 'near', 'aff'), ('L1 denial', 'syn_neg', 'near_neg', 'aff'), ('L2', 'para', 'pfoil', 'aff'), ('L2 denial', 'para_neg', 'pfoil_neg', 'aff')]),
    'L3': ('L3 entailed case', 'L3_entailed.md',
           'The second sentence states a specific case that implies the clause (or its denial), in other words. The foil shares the same tokens with the clause and decides nothing about it; where possible it shares the case\'s own surface too ("a crowd is pouring onto the pitch").',
           ['Does each case really imply the clause, for an ordinary reader?', 'Does each denial case really imply the clause is false?', 'Is each foil neutral: neither implies the clause nor its denial?'],
           [('clause', 'aff'), ('case (implies the clause)', 'case'), ('foil', 'case_foil'), ('case (implies the denial)', 'ncase'), ('foil', 'ncase_foil')],
           [('case', 'case', 'case_foil', 'aff'), ('denial case', 'ncase', 'ncase_foil', 'aff')]),
    'L4': ('L4 reference', 'L4_reference.md',
           'The second sentence names its subject by a pronoun only. The two clauses have subjects that take different pronouns, so the pronoun says which clause is meant. The foil is the same sentence with the other clause\'s pronoun: it says P\'s predicate about Q\'s subject. Restatement and foil differ in one token, which neither clause contains.',
           ['Is the pronoun unambiguous for each pair (he, she, it)?', 'Is each foil a different statement (the wrong person or thing), not a restatement?'],
           [('clause P', 'P'), ('clause Q', 'Q'), ('restatement of P', 'res'), ('foil (Q\'s pronoun)', 'foil'), ('restatement of Q (AC twin)', 'resQ')],
           [('P', 'res', 'foil', 'P'), ('P denial', 'res_neg', 'foil_neg', 'P'), ('Q', 'resQ', 'foilQ', 'Q')]),
    'L5': ('L5 restructured', 'L5_restructured.md',
           'The clause said again with another syntax (nominalisation, passive, possessive). The foil keeps the frame and the words of the clause and replaces the event ("approval" by "attention").',
           ['Is each restructured sentence the same statement as the clause?', 'Is each foil a different event, not implied by the clause?', 'Are the denials natural?'],
           [('clause', 'aff'), ('restructured', 'res'), ('foil', 'foil'), ('restructured denial', 'res_neg')],
           [('restatement', 'res', 'foil', 'aff'), ('denial', 'res_neg', 'foil_neg', 'aff')]),
    'L6': ('L6 new domain', 'L6_new_domain.md',
           'L2-type paraphrases in domains absent from the development clauses: law (1 to 5), medicine (6 to 10), machines (11 to 15), fiction (16 to 20).',
           ['Is each paraphrase the same statement in its domain?', 'Is each foil a different property, not implied or excluded?'],
           [('clause', 'aff'), ('paraphrase', 'para'), ('foil', 'pfoil'), ('paraphrased denial', 'para_neg')],
           [('paraphrase', 'para', 'pfoil', 'aff'), ('denial', 'para_neg', 'pfoil_neg', 'aff')]),
    'C1': ('C1 alias defined in context', 'C1_alias.md',
           'A context sentence before the premise names two people. The clause uses a role ("the baker"), the restatement the person who holds it, the foil the other person, with the same predicate. Only the context links name and role; the names occur nowhere in the premise.',
           ['Does each context assign the role to exactly one of the two names?', 'Is the role in the clause the one the context assigns (no ambiguity)?'],
           [('context', 'context'), ('clause P', 'P'), ('clause Q', 'Q'), ('restatement', 'res'), ('foil', 'foil')],
           [('P', 'res', 'foil', 'P'), ('P denial', 'res_neg', 'foil_neg', 'P')]),
    'C2': ('C2 same word, sense differs; C2′ paraphrase against a false friend', 'C2_sense.md',
           'The clause uses a word with two senses; its consequent fixes the sense. The restatement adds a word of that sense, the foil a word of the other sense, the same number of tokens: token identity ties. In C2′ the restatement is a paraphrase in the clause\'s sense that shares no content word with it, against the same foil, which repeats the clause\'s words (a false friend): token identity prefers the foil.',
           ['Does the consequent fix the sense of the clause?', 'Does each restatement keep that sense, and each foil clearly take the other?', 'Is each C2′ paraphrase the same statement in the clause\'s sense?'],
           [('clause P', 'P'), ('consequent Q (fixes the sense)', 'Q'), ('restatement (same sense)', 'res'), ('foil (other sense)', 'foil'), ('C2′ paraphrase', 'para')],
           [('C2', 'res', 'foil', 'P'), ('C2 denial', 'res_neg', 'foil_neg', 'P'), ('C2′ (foil shares more by design)', 'para', 'foil', 'P')]),
    'C3': ('C3 ellipsis', 'C3_ellipsis.md',
           'The second sentence keeps only a pronoun and "does": "It does." The clause\'s subject and verb must be recovered from the premise. The foil is the other clause\'s pronoun ("She does."), which affirms Q: the AC twin. Neither shares a token with the clauses.',
           ['Is the pronoun unambiguous?', 'Does "does" read naturally as standing for the clause\'s verb?'],
           [('clause P', 'P'), ('clause Q', 'Q'), ('restatement', 'res'), ('foil (AC twin)', 'foil'), ('denial', 'res_neg')],
           [('P', 'res', 'foil', 'P'), ('P denial', 'res_neg', 'foil_neg', 'P')]),
    'C4': ('C4 denial by an incompatible value', 'C4_denial_by_context.md',
           'The second sentence denies a clause only through what is known about it: "the light is green" denies "the light is red". The foil keeps the subject and names a value that is compatible with the clause. The denial of Q gives the MT analogue; the denial of P the DS1 analogue and the DA twin. Morphological antonyms ("armed", "disarmed") are avoided, since they share tokens.',
           ['Is each incompatible value really incompatible with the clause?', 'Is each foil really compatible (it neither denies nor implies the clause)?', 'Should some items fix the incompatibility by a context sentence instead of general knowledge?'],
           [('clause P', 'P'), ('clause Q', 'Q'), ('denies Q', 'Q_inc'), ('foil', 'Q_foil'), ('denies P', 'P_inc'), ('foil', 'P_foil')],
           [('denial of Q', 'Q_inc', 'Q_foil', 'Q'), ('denial of P', 'P_inc', 'P_foil', 'P')]),
}

summary = []
for lv, (title, fn, what, checks, cols, pairs) in SPEC.items():
    bank = LEVELS[lv]; bad = []
    for i, e in enumerate(bank, 1):
        for nm, r, f, ref in pairs:
            a, b = ov(e[r], e[ref]), ov(e[f], e[ref])
            if a != b and 'by design' not in nm: bad.append((i, nm, e[r], a, e[f], b))
    n = len(bank) * sum('by design' not in p[0] for p in pairs); summary.append((title, fn, len(bank), n, len(bad)))
    L = [f'# {title}', '', f'**Draft items for review (2026-11-26).** Level of [Analogy_and_Context_Test_Set.md](../Analogy_and_Context_Test_Set.md). Source: `{SRC}`. Back to the [index](README.md).', '',
         f'**What this level tests.** {what}', '', '**What to check:**'] + [f'- {c}' for c in checks] + ['', f'**Matched-overlap check:** {n - len(bad)} of {n} restatement and foil pairs share the same number of tokens with the clause.' + (' The pairs that do not:' if bad else ''), '']
    if bad:
        L += ['| # | pair | restatement | tokens in the clause | foil | tokens in the clause |', '| --- | --- | --- | ---: | --- | ---: |'] + [f'| {i} | {nm} | {cap(a)}. | {x} | {cap(b)}. | {y} |' for i, nm, a, x, b, y in bad] + ['']
    if lv == 'C2':
        L += ['C2′ by design: the false friend shares ' + ', '.join(f'{ov(e["foil"], e["P"])}' for e in bank[:5]) + ' ... tokens with the clause, the paraphrase ' + ', '.join(f'{ov(e["para"], e["P"])}' for e in bank[:5]) + ' ... (first five items).', '']
    L += ['| # | ' + ' | '.join(c for c, _ in cols) + ' |', '| --- | ' + ' | '.join('---' for _ in cols) + ' |']
    for i, e in enumerate(bank, 1):
        L.append(f'| {i} | ' + ' | '.join((e[k] if k == 'context' else cap(e[k]) + '.') for _, k in cols) + ' |')
    (OUT / fn).write_text('\n'.join(L) + '\n')
R = ['# Analogy and In-Context Test Set: Draft Items', '', '**What this folder is.** The first 20 frames per level of the test set designed in [Analogy_and_Context_Test_Set.md](../Analogy_and_Context_Test_Set.md), drafted by the agent on 2026-11-26 for the author\'s review. Nothing has been extracted or scored. The source of every sentence is `' + SRC + '`; these pages are generated from it by `notebooks/conservative_arch/scaleup/debug/template_analogy_review.py`, so corrections go into the source and the pages are regenerated.', '',
     '**How the premises are built.** Each frame gives the clauses and, at its level, the restatements and foils of the second sentence. The six wordings (three conditional, three disjunctive) and the forms are composed as in the current test: modus ponens is a conditional with a restatement of P, modus tollens a conditional with a denial of Q, the disjunctive syllogisms a disjunctive with a denial of P or of Q, and the fallacy twins use the other clause. For the clause-level banks (L0 to L3, L5, L6) the second clause is another frame of the same bank.', '',
     '**The matched-overlap rule.** Each restatement has a foil that shares exactly as many tokens with the clause (GPT-2 tokens, lower-cased), so token identity cannot prefer the restatement. The check below is computed, not judged by eye.', '',
     '| level | page | frames | pairs checked | pairs that break the rule |', '| --- | --- | ---: | ---: | ---: |'] + [f'| {t} | [{f}]({f}) | {k} | {n} | {b} |' for t, f, k, n, b in summary] + ['',
     '**What the author is asked to check, for every level:** that each restatement is the same statement as its clause (or implies it, at L3), that each foil is clearly a different statement and is neither implied nor excluded by the clause, and that nothing reads unnaturally. Items can be struck, replaced or rewritten; the overlap check reruns on the new text.', '',
     '**Known limits of the draft.**', '- C1 and C4 test the second sentence about one clause; the other forms of the frame (those that restate or deny the other clause) fall back on verbatim sentences.', '- C4 relies on general knowledge (green denies red). A variant in which a context sentence sets the incompatibility is not drafted yet.', '- L4 and C3 use gendered pronouns, which need the model to know the gender of "the king" or "the bride"; the embedding may carry part of that, which the readings would then show.', '- All items are by one drafter, so the foils may share a style the restatements lack. The author\'s check and a later lexical check on style words limit this.']
(OUT / 'README.md').write_text('\n'.join(R) + '\n')
for t, f, k, n, b in summary: print(f'{t:70s} frames {k}  pairs {n}  break the rule {b}')
