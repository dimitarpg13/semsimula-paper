"""Draft items for task B-logic: a logical form set by context (2026-11-27), drafted by the agent for the author's review.
Not yet used by any extraction. Review pages: template_blogic_review.py writes companion_notes/semantic_templates/blogic_items/.

Each frame has two clauses P and Q (they make the premise "If P, then Q." or "Either P or Q."), a signal S, and an unrelated clause
U. The second sentence of every premise of the frame is S, so within a frame the premise is the same string whatever the context.
A context sentence before the premise says what S means: P, Q, not P, not Q, or U (which decides nothing). That alone makes the
premise a valid form, its fallacy twin, or a foil:
  conditional    S means P: MP     S means Q: AC (twin of MP)     S means not Q: MT     S means not P: DA (twin of MT)     U: foil
  disjunctive    S means not P: DS1     S means not Q: DS2     S means P: affirmed P (twin of DS1)     S means Q: affirmed Q     U: foil
S is an arbitrary signal (a flag, a kite, a ribbon, a bell), so that it has no meaning of its own that favours one clause; signals with a
conventional meaning near a clause (a red flag by a high river, a printer's warning light, a dinner gong) were replaced in drafting.
Two flavours of context:
  explicit  the meaning of the signal is stated:   Here, "the blue flag is up" means that the bank is closed.
  natural   a regularity in the world is stated:   The blue flag is up only when the bank is closed.
Sentences are lower case without a final full stop, as in the other banks; the generator capitalises them.
"""
# (P, Q, S, U)
FRAMES = [
    ('the bank is closed', 'the high street is quiet', 'the blue flag is up', 'the cleaners are inside'),
    ('the road is icy', 'the buses are late', 'the paper lantern is lit', 'the radio is on'),
    ('the river is high', 'the ferry is cancelled', 'the yellow kite is flying', 'the fair is in town'),
    ('the oven is hot', 'the bread is ready', 'the blue dot is showing', 'the kitchen is tidy'),
    ('the match is on', 'the pub is full', 'the tower bell is ringing', 'the car park is new'),
    ('the baby is asleep', 'the house is quiet', 'the pink ribbon is tied', 'the curtains are blue'),
    ('the museum is open', 'the guides are busy', 'the round sign is turned', 'the roof is old'),
    ('the boiler is on', 'the radiators are warm', 'the chequered flag is up', 'the pipes are copper'),
    ('the teacher is ill', 'the lesson is cancelled', 'the white card is showing', 'the window is open'),
    ('the harbour is calm', 'the boats are out', 'the white pennant is flying', 'the gulls are noisy'),
    ('the printer is jammed', 'the report is late', 'the brass bell is ringing', 'the paper is recycled'),
    ('the goalkeeper is injured', 'the coach is worried', 'the purple flag is raised', 'the referee is tall'),
    ('the library is shut', 'the students are outside', 'the chalk mark is green', 'the books are heavy'),
    ('the soup is ready', 'the guests are seated', 'the silver ribbon is up', 'the napkins are white'),
    ('the lift is broken', 'the stairs are crowded', 'the green ribbon is up', 'the lobby is bright'),
    ('the pool is closed', 'the children are bored', 'the striped flag is up', 'the lifeguard is young'),
    ('the train is delayed', 'the platform is packed', 'the orange kite is flying', 'the station is old'),
    ('the market is busy', 'the prices are high', 'the grey flag is up', 'the fountain is dry'),
    ('the volcano is active', 'the village is empty', 'the copper bell is ringing', 'the hills are green'),
    ('the concert is sold out', 'the touts are busy', 'the poster is red', 'the hall is warm'),
]


def neg(s):
    """denial of a sentence with one 'is' or 'are'"""
    for a in (' is ', ' are '):
        if a in f' {s} ':
            return f' {s} '.replace(a, a.rstrip() + ' not ', 1).strip()
    raise ValueError(s)


cap = lambda s: s[0].upper() + s[1:]
CONTEXT = {'explicit': lambda S, X: f'Here, "{S}" means that {X}.', 'natural': lambda S, X: f'{cap(S)} only when {X}.'}
MEANING = {'P': lambda P, Q, U: P, 'Q': lambda P, Q, U: Q, 'notP': lambda P, Q, U: neg(P), 'notQ': lambda P, Q, U: neg(Q), 'U': lambda P, Q, U: U}
# kind of premise for each meaning of S, per frame of the premise
KIND = {'cond': {'P': 'MP', 'Q': 'AC', 'notQ': 'MT', 'notP': 'DA', 'U': 'foil'},
        'disj': {'notP': 'DS1', 'notQ': 'DS2', 'P': 'Paff', 'Q': 'Qaff', 'U': 'foil'}}
# the review log: the author's corrections, in order (template_blogic_review.py prints it)
REVIEW_LOG = [
    ('2026-11-27', 'all frames', 'reviewed by the author: "makes sense"; no change.'),
    ('2026-11-27', 'control contexts', 'the both-clauses contexts (variants a and b) reviewed by the author: they read well; no change.'),
]


def contexts(i, flavour):
    P, Q, S, U = FRAMES[i]
    return {m: CONTEXT[flavour](S, f(P, Q, U)) for m, f in MEANING.items()}


# The control (2026-11-27, BLogic_Test.md §5.4): every context names both clauses and the versions differ only in which one the
# signal is tied to, so a state that only marks "this clause was named before" cannot separate them. Two orders, so that the
# clause named last before the premise is the bound one in variant b and the other one in variant a.
OTHER = {'P': lambda P, Q: Q, 'notP': lambda P, Q: Q, 'Q': lambda P, Q: P, 'notQ': lambda P, Q: P, 'U': lambda P, Q: f'{P} or {Q}'}
CONTROL = {'both_a': lambda S, X, O: f'Here, "{S}" means that {X}, and nothing about whether {O}.',
           'both_b': lambda S, X, O: f'Here, "{S}" says nothing about whether {O}; it means that {X}.'}


def control_contexts(i, variant):
    P, Q, S, U = FRAMES[i]
    return {m: CONTROL[variant](S, f(P, Q, U), OTHER[m](P, Q)) for m, f in MEANING.items()}
