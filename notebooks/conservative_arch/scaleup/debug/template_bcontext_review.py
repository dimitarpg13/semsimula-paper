#!/usr/bin/env python3
"""Review page for the task B draft items, meaning set only by context (2026-11-27).
    python3 template_bcontext_review.py OUTDIR      (OUTDIR = companion_notes/semantic_templates/bcontext_items)
Reads template_bcontext_clauses.py, checks the frames, and writes OUTDIR/README.md: the design, the checks, every frame with its
context sentences. Corrections go into the source; the page is regenerated."""
import sys
from pathlib import Path
import tiktoken
DBG = Path(__file__).resolve().parent; sys.path.insert(0, str(DBG))
import template_bcontext_clauses as B
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True); enc = tiktoken.get_encoding('gpt2')
SRC = 'notebooks/conservative_arch/scaleup/debug/template_bcontext_clauses.py'
STOP = {'the', 'is', 'are', 'a', 'an', 'in', 'on', 'at', 'of', 'to', 'by', 'we', 'were', 'was', 'it', 'not', 'and', 'for', 'up', 'her', 'she', 'had'}
words = lambda s: {w.strip('.,;"').lower() for w in s.split()} - STOP
cap = lambda s: s[0].upper() + s[1:]

problems = []
for i, (W, A, Bs, pred, nA, nB) in enumerate(B.SENSE_FRAMES):
    clause = words(A) | words(Bs) | words(W) | words(pred)
    for nm, c in (('natural context for sense 1', nA), ('natural context for sense 2', nB)):
        sh = words(c) & clause
        if sh: problems.append(f'word-sense frame {i + 1}: the {nm} shares {sorted(sh)} with the clauses')
    if words(A) & words(Bs): problems.append(f'word-sense frame {i + 1}: the two senses share {sorted(words(A) & words(Bs))}')
    try: [B.neg(f'{x} {pred}') for x in (A, Bs, W)]
    except ValueError as e: problems.append(f'word-sense frame {i + 1}: no copula to deny in {e}')
names = [n for f in B.PRONOUN_FRAMES for n in f[:3]]
if len(set(names)) != len(names): problems.append('a name is used in more than one pronoun frame')
for i, (X, Y, Z, pr, pred) in enumerate(B.PRONOUN_FRAMES):
    if pr not in ('he', 'she'): problems.append(f'pronoun frame {i + 1}: pronoun {pr}')

R = ['# Task B: Draft Items (meaning set only by context)', '',
     '**What this folder is.** The draft of task B, for the author\'s review (2026-11-27): the restatement and its rival are the same string, and only a context sentence before the premise says what it means. The source of every sentence is `' + SRC + '`; this page is generated from it by `notebooks/conservative_arch/scaleup/debug/template_bcontext_review.py`, so corrections go into the source and the page is regenerated. Nothing has been extracted or scored. Design and draft calls: [Task_B_Meaning_Set_by_Context.md](../Task_B_Meaning_Set_by_Context.md).', '',
     '**The design.** Two sublevels, each built on the usual premise ("If P, then Q. S." or "Either P or Q. S."):', '',
     '- **Word sense.** The second sentence S contains an ambiguous word ("The bank is closed."). Clause P says the same of one sense in other words ("the lender is closed"), clause Q of the other sense ("the embankment is closed"). The context picks the sense.',
     '- **Pronoun.** S has a pronoun ("He is tired."). P and Q say it of two people of the same gender ("John is tired", "Bill is tired"). The context picks the referent; a third person gives a foil.', '',
     'What the context makes of the premise:', '',
     '| the second sentence | conditional frame | disjunctive frame |', '| --- | --- | --- |',
     '| affirmed, meaning P | MP, valid | (not used) |', '| affirmed, meaning Q | AC, twin of MP | (not used) |',
     '| denied, meaning Q | MT, valid | DS2, valid |', '| denied, meaning P | DA, twin of MT | DS1, valid |',
     '| about the third person (pronoun only) | foil | foil |', '',
     '**Why the words cannot help.** Within a frame, wording and polarity, the matcher\'s particles (connective, clauses, second sentence, "Therefore") are the same tokens at the same positions in every version, and the context\'s tokens are not particles. Token identity and the embedding therefore tie at exactly 0.5 on every pair, by construction.', '',
     '**Four flavours of context per sublevel.**', '',
     '| flavour | word sense (example) | pronoun (example) | what it names |', '| --- | --- | --- | --- |',
     '| explicit | `Here, "the bank" means the lender.` | `Here, "he" means John.` | the selected meaning only |',
     '| natural | `We were talking about money and loans.` | `John was the only one in the room.` | word sense: no clause word; pronoun: the selected person |',
     '| control a | `Here, "the bank" means the lender, not the embankment.` | `John came into the room while Bill stayed outside.` | both meanings, the bound one first |',
     '| control b | `Here, "the bank" is not the embankment; it means the lender.` | `While Bill stayed outside, John came into the room.` | both meanings, the bound one last |', '',
     'The controls separate binding from mention, as in B-logic: a state that only marks which meaning was named, or which was named last, cannot separate versions that name both.', '',
     '## Checks', '']
R += ([f'- {p}' for p in problems] if problems else ['- Every natural word-sense context shares no content word with its clauses; the two senses share no word; every clause can be denied; no name is reused across pronoun frames.'])
n_tok = [len(enc.encode(cap(c) + '.')) for i in range(20) for f in ('explicit', 'natural', 'both_a', 'both_b') for c in B.sense_frame(i)[3][f].values()]
p_tok = [len(enc.encode(cap(c) + '.')) for i in range(20) for f in ('explicit', 'natural', 'both_a', 'both_b') for c in B.pronoun_frame(i)[3][f].values()]
R += [f'- Context lengths: word sense {min(n_tok)} to {max(n_tok)} tokens, pronoun {min(p_tok)} to {max(p_tok)} tokens (the filler before them makes up the difference).', '',
      '## Word-sense frames', '',
      '| frame | second sentence | clause P (sense 1) | clause Q (sense 2) | natural context, sense 1 | natural context, sense 2 |', '| ---: | --- | --- | --- | --- | --- |']
for i, (W, A, Bs, pred, nA, nB) in enumerate(B.SENSE_FRAMES):
    R.append(f'| {i + 1} | {cap(W)} {pred}. | {A} {pred} | {Bs} {pred} | {cap(nA)}. | {cap(nB)}. |')
R += ['', 'The explicit and control contexts follow the patterns of the flavour table with the frame\'s phrase and senses.', '',
      '## Pronoun frames', '',
      '| frame | second sentence | clause P | clause Q | third person (foil) |', '| ---: | --- | --- | --- | --- |']
for i, (X, Y, Z, pr, pred) in enumerate(B.PRONOUN_FRAMES):
    R.append(f'| {i + 1} | {cap(pr)} {pred}. | {X} {pred} | {Y} {pred} | {Z} |')
R += ['', 'The contexts follow the patterns of the flavour table with the frame\'s names; the foil contexts name the third person (control: "Tom came into the room while John and Bill stayed outside").', '',
      '## Review log', '']
R += ([f'- {d}, {what}: {note}' for d, what, note in B.REVIEW_LOG] if B.REVIEW_LOG else ['- Not yet reviewed.'])
(OUT / 'README.md').write_text('\n'.join(R) + '\n')
print(f'wrote {OUT / "README.md"}; {len(problems)} problem(s)'); [print(' ', p) for p in problems]
