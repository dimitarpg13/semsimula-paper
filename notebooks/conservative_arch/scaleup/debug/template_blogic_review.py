#!/usr/bin/env python3
"""Review page for the B-logic draft items (2026-11-27).
    python3 template_blogic_review.py OUTDIR      (OUTDIR = companion_notes/semantic_templates/blogic_items)
Reads template_blogic_clauses.py, checks the frames, and writes OUTDIR/README.md: the design, the checks, every frame with its ten
context sentences. Corrections go into the source; the page is regenerated."""
import sys
from pathlib import Path
import tiktoken
DBG = Path(__file__).resolve().parent; sys.path.insert(0, str(DBG))
import template_blogic_clauses as B
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True); enc = tiktoken.get_encoding('gpt2')
SRC = 'notebooks/conservative_arch/scaleup/debug/template_blogic_clauses.py'
STOP = {'the', 'is', 'are', 'a', 'an', 'in', 'on', 'out', 'up', 'of', 'not', 'to'}
words = lambda s: set(s.lower().split()) - STOP
ntok = lambda s: len(enc.encode(s))

problems, lens = [], []
for i, (P, Q, S, U) in enumerate(B.FRAMES):
    for nm, x in (('signal', S), ('unrelated clause', U)):
        sh = (words(x) & (words(P) | words(Q)))
        if sh: problems.append(f'frame {i + 1}: the {nm} shares {sorted(sh)} with the clauses')
    if words(S) & words(U): problems.append(f'frame {i + 1}: the signal and the unrelated clause share {sorted(words(S) & words(U))}')
    for fl in B.CONTEXT:
        lens += [ntok(c) for c in B.contexts(i, fl).values()]
    try: [B.neg(x) for x in (P, Q)]
    except ValueError as e: problems.append(f'frame {i + 1}: no copula to deny in {e}')

R = ['# Task B-logic: Draft Items', '',
     '**What this folder is.** The draft of task B-logic, for the author\'s review (2026-11-27): a logical form set by context. The source of every sentence is `' + SRC + '`; this page is generated from it by `notebooks/conservative_arch/scaleup/debug/template_blogic_review.py`, so corrections go into the source and the page is regenerated. Nothing has been extracted or scored. Background: task B in [Summary_State_Test.md](../Summary_State_Test.md) and the discussion that led to combining it with the logical forms.', '',
     '**The design.** Each frame has two clauses P and Q, a signal S and an unrelated clause U. Every premise of a frame has S as its second sentence, so within a frame the premise is the same string; only a context sentence before it differs, saying what S means. The context makes the premise a valid form, its fallacy twin, or a foil:', '',
     '| S means | conditional ("If P, then Q. S.") | disjunctive ("Either P or Q. S.") |', '| --- | --- | --- |',
     '| P | MP, valid | the affirmed P, twin of DS1 |', '| Q | AC, twin of MP | the affirmed Q, twin of DS2 |', '| not Q | MT, valid | DS2, valid |', '| not P | DA, twin of MT | DS1, valid |', '| U | foil: decides nothing | foil: decides nothing |', '',
     '**Why the words cannot help.** Within a frame the matcher\'s particles (connective, clauses, second sentence, "Therefore") are identical in every version, and the context\'s tokens are not particles. The extraction will also fix the position at which the premise starts (the ordinary text before the context is shortened by the context\'s length), so that the position vectors are identical too. The embedding and token identity then give identical readings for MP, AC and the foil of a frame: they tie at exactly 0.5 by construction. Only states that took in the context can separate them.', '',
     '**Two flavours of context.** *Explicit*: the meaning of the signal is stated (`Here, "the blue flag is up" means that the bank is closed.`). *Natural*: a regularity in the world is stated (`The blue flag is up only when the bank is closed.`). Both are generated from the frame by one template each, so only the frame needs review. The signal is arbitrary (a flag, a lamp, a bell) so that it favours no clause on its own.', '',
     '**Example, frame 1, natural flavour, MP:** `The blue flag is up only when the bank is closed.` / `If the bank is closed, then the high street is quiet. The blue flag is up. Therefore`', '',
     '**What to check, per frame.**', '', '- Does "If P, then Q" read as a sensible conditional, and "Either P or Q" as a sensible disjunction?', '- Is the signal free of any meaning that would link it to P or Q without the context?', '- Is U unrelated to P, Q and the signal?', '- Do the ten context sentences read naturally, including the denied ones ("only when the bank is not closed")?', '',
     f'**Automatic checks.** {len(B.FRAMES)} frames; {len(problems)} problems found' + (':' if problems else '.') , '']
R += [f'- {p}' for p in problems] + ([''] if problems else [])
R += [f'Context sentences run from {min(lens)} to {max(lens)} GPT-2 tokens.', '', '**Size.** 20 frames × 2 frames of premise × 5 meanings × 3 wordings × 2 flavours = 1,200 premises per model.', '']
if B.REVIEW_LOG:
    R += ['**The author\'s corrections.**', '', '| date | item | change |', '| --- | --- | --- |'] + [f'| {d} | {it} | {c} |' for d, it, c in B.REVIEW_LOG] + ['']
R += ['**The control** (added 2026-11-27, BLogic_Test.md §5.4): contexts that name both clauses, in two orders; all of them on [control.md](control.md). Example, frame 1, MP, variant a: `' + B.control_contexts(0, 'both_a')['P'] + '`', '']
R += ['## The frames', '', '| # | P | Q | signal S | unrelated U |', '| --- | --- | --- | --- | --- |']
R += [f'| {i + 1} | {P} | {Q} | {S} | {U} |' for i, (P, Q, S, U) in enumerate(B.FRAMES)] + ['']
R += ['## The context sentences', '']
for i, (P, Q, S, U) in enumerate(B.FRAMES):
    ex, na = B.contexts(i, 'explicit'), B.contexts(i, 'natural')
    R += [f'### Frame {i + 1}', '', f'Premises: "If {P}, then {Q}. {B.cap(S)}." and "Either {P} or {Q}. {B.cap(S)}."', '', '| S means | conditional | disjunctive | explicit | natural |', '| --- | --- | --- | --- | --- |']
    R += [f'| {m.replace("not", "not ")} | {B.KIND["cond"][m]} | {B.KIND["disj"][m]} | {ex[m]} | {na[m]} |' for m in B.MEANING] + ['']
(OUT / 'README.md').write_text('\n'.join(R))
print(f'{len(B.FRAMES)} frames, {len(problems)} problems; context lengths {min(lens)}-{max(lens)} tokens')
for p in problems: print('  ', p)

# the control's contexts (2026-11-27): one page, linked from the README
C = ['# Task B-logic: the Control Contexts', '', 'Every context names both clauses; the versions of a frame differ only in which clause the signal is tied to (BLogic_Test.md, §5.4). Variant a names the bound clause first, variant b last, so that the clause named just before the premise is the bound one in b and the other one in a. Generated from the approved frames by `template_blogic_clauses.py` (`CONTROL`); the index is [README.md](README.md).', '']
for i, (P, Q, S, U) in enumerate(B.FRAMES):
    a, b = B.control_contexts(i, 'both_a'), B.control_contexts(i, 'both_b')
    C += [f'## Frame {i + 1}', '', '| S means | conditional | disjunctive | variant a | variant b |', '| --- | --- | --- | --- | --- |']
    C += [f'| {m.replace("not", "not ")} | {B.KIND["cond"][m]} | {B.KIND["disj"][m]} | {a[m]} | {b[m]} |' for m in B.MEANING] + ['']
(OUT / 'control.md').write_text('\n'.join(C))
print('wrote control.md')
