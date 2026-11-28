"""Draft items for task B: meaning set only by context (2026-11-27), drafted by the agent for the author's review.
Not yet used by any extraction. Review page: template_bcontext_review.py writes companion_notes/semantic_templates/bcontext_items/.

Task B as first proposed (2026-11-27): the restatement and its rival are the same string, and only a context sentence before the
premise says what it means. Two sublevels, each a frame of the usual premises ("If P, then Q. S." or "Either P or Q. S."):

  word sense   S contains an ambiguous word ("the bank is closed"); P says it of one sense in other words ("the lender is
               closed"), Q of the other sense ("the embankment is closed"). The context picks the sense of S.
  pronoun      S has a pronoun ("he is tired"); P and Q say it of two people of the same gender ("John is tired", "Bill is
               tired"). The context picks the referent. A third person (Tom) gives a foil.

The context alone makes the premise a valid form, its fallacy twin, the other valid form, or a foil:
  conditional, S affirmed    S means P: MP     S means Q: AC (twin of MP)       S means the third person: foil
  conditional, S denied      S means Q: MT     S means P: DA (twin of MT)       S means the third person: foiln
  disjunctive, S denied      S means P: DS1    S means Q: DS2 (the other form)  S means the third person: foiln
Within a frame and wording every version is the same string at the same position, so token identity and the embedding tie.

Contexts, per sublevel (the word in brackets is the meaning the context selects; Y is the other one):
  word sense, explicit      Here, "the bank" means the [lender].                      names the selected clause's words
  word sense, natural       We were talking about money and loans.                    names no clause word (a topic)
  word sense, control a/b   Here, "the bank" means the [lender], not the embankment.  names both senses, the bound one first
                            Here, "the bank" is not the embankment; it means the [lender].           ... or last
  pronoun, explicit         Here, "he" means [John].
  pronoun, natural          [John] was the only one in the room.
  pronoun, control a/b      [John] came into the room while Bill stayed outside.      names both people, the bound one first
                            While Bill stayed outside, [John] came into the room.     ... or last
Sentences are lower case without a final full stop, as in the other banks; the generator capitalises them. The natural sense
contexts share no word with the frame's clauses (checked by the review page).
"""
# word sense: (ambiguous phrase W as it appears in S, sense-1 paraphrase A, sense-2 paraphrase B, predicate,
#              natural context selecting A, natural context selecting B)
SENSE_FRAMES = [
    ('the bank', 'the lender', 'the embankment', 'is closed', 'we were talking about money and loans', 'we were walking by the water'),
    ('the bat', 'the baseball club', 'the flying mammal', 'is in the shed', 'we were getting ready for the game', 'we were watching animals at dusk'),
    ('the pitcher', 'the water jug', 'the baseball thrower', 'is in the kitchen', 'we were setting the table for lunch', 'we were watching the team practise'),
    ('the crane', 'the lifting machine', 'the tall wading bird', 'is by the lake', 'we were watching the builders at work', 'we were looking at the wildlife in the marsh'),
    ('the mouse', 'the computer pointer', 'the small rodent', 'is under the desk', 'we were fixing the laptop', 'we were trying to catch the pest'),
    ('the seal', 'the sea animal', 'the wax stamp', 'is grey', 'we were watching wildlife at the coast', 'we were reading an old royal letter'),
    ('the glasses', 'the spectacles', 'the drinking cups', 'are on the shelf', 'she could not read the menu', 'we were washing up after the party'),
    ('the plant', 'the factory', 'the green shrub', 'is near the road', 'the workers were on strike', 'we were watering the garden'),
    ('the deck', 'the ship floor', 'the pack of cards', 'is wet', 'the sailors were scrubbing all morning', 'we were playing poker'),
    ('the spring', 'the coiled wire', 'the natural fountain', 'is hidden', 'we were repairing the old mattress', 'we were hiking up the hill, thirsty'),
    ('the pupil', 'the student', 'the dark centre of the iris', 'is small', 'we were visiting the school', 'the doctor shone a light in her face'),
    ('the letter', 'the written message', 'the alphabet symbol', 'is on the page', 'we were waiting for the post', 'we were learning to spell'),
    ('the organ', 'the musical instrument', 'the body part', 'is large', 'we were at the church concert', 'we were studying anatomy'),
    ('the trunk', 'the elephant nose', 'the storage chest', 'is heavy', 'we were at the zoo', 'we were packing for the long voyage'),
    ('the fan', 'the cooling device', 'the sports supporter', 'is in the hall', 'it was a very hot day', 'the team had just won the cup'),
    ('the note', 'the written reminder', 'the musical tone', 'is short', 'she left a message on the fridge', 'the violinist was tuning up'),
    ('the board', 'the wooden plank', 'the governing committee', 'is new', 'we were building a shed', 'the shareholders met on monday'),
    ('the jam', 'the fruit spread', 'the traffic queue', 'is bad today', 'we were making breakfast', 'we were driving to the airport'),
    ('the tank', 'the water container', 'the armoured vehicle', 'is in the yard', 'we were fixing the garden irrigation', 'the soldiers were on parade'),
    ('the match', 'the small fire stick', 'the sports game', 'is short', 'we were lighting the candles', 'we were watching the football'),
]

# pronoun: (person P, person Q, third person, pronoun, predicate)
PRONOUN_FRAMES = [
    ('John', 'Bill', 'Tom', 'he', 'is tired'), ('Mary', 'Anne', 'Kate', 'she', 'is hungry'),
    ('Peter', 'Mark', 'Sam', 'he', 'is late'), ('Lucy', 'Emma', 'Rose', 'she', 'is at home'),
    ('David', 'Paul', 'Jack', 'he', 'is angry'), ('Sarah', 'Helen', 'Jane', 'she', 'is ill'),
    ('James', 'Henry', 'Luke', 'he', 'is asleep'), ('Alice', 'Grace', 'Ruth', 'she', 'is happy'),
    ('Robert', 'Frank', 'Adam', 'he', 'is outside'), ('Laura', 'Clare', 'Beth', 'she', 'is busy'),
    ('George', 'Simon', 'Ben', 'he', 'is cold'), ('Emily', 'Megan', 'Holly', 'she', 'is upstairs'),
    ('Michael', 'Steven', 'Ryan', 'he', 'is lost'), ('Julia', 'Fiona', 'Amy', 'she', 'is worried'),
    ('Daniel', 'Oliver', 'Ian', 'he', 'is ready'), ('Hannah', 'Chloe', 'Ella', 'she', 'is awake'),
    ('Thomas', 'Edward', 'Carl', 'he', 'is wet'), ('Sophie', 'Molly', 'Zoe', 'she', 'is bored'),
    ('Andrew', 'Philip', 'Neil', 'he', 'is early'), ('Rachel', 'Diana', 'Eve', 'she', 'is sad'),
]


def neg(s):
    """denial of a sentence with one 'is' or 'are'"""
    for a in (' is ', ' are '):
        if a in f' {s} ':
            return f' {s} '.replace(a, a.rstrip() + ' not ', 1).strip()
    raise ValueError(s)


def sense_frame(i):
    """P, Q, S (affirmed) and the contexts of word-sense frame i, keyed by flavour and by the selected meaning 'A' or 'B'"""
    W, A, B, pred, nA, nB = SENSE_FRAMES[i]
    ctx = {'explicit': {'A': f'Here, "{W}" means {A}', 'B': f'Here, "{W}" means {B}'},
           'natural': {'A': nA, 'B': nB},
           'both_a': {'A': f'Here, "{W}" means {A}, not {B}', 'B': f'Here, "{W}" means {B}, not {A}'},
           'both_b': {'A': f'Here, "{W}" is not {B}; it means {A}', 'B': f'Here, "{W}" is not {A}; it means {B}'}}
    return f'{A} {pred}', f'{B} {pred}', f'{W} {pred}', ctx


def pronoun_frame(i):
    """P, Q, S (affirmed) and the contexts of pronoun frame i, keyed by flavour and by the selected referent 'A', 'B' or 'Z'"""
    X, Y, Z, pr, pred = PRONOUN_FRAMES[i]
    sel = {'A': X, 'B': Y, 'Z': Z}
    other = {'A': Y, 'B': X}
    ctx = {'explicit': {k: f'Here, "{pr}" means {v}' for k, v in sel.items()},
           'natural': {k: f'{v} was the only one in the room' for k, v in sel.items()},
           'both_a': {**{k: f'{sel[k]} came into the room while {other[k]} stayed outside' for k in 'AB'},
                      'Z': f'{Z} came into the room while {X} and {Y} stayed outside'},
           'both_b': {**{k: f'While {other[k]} stayed outside, {sel[k]} came into the room' for k in 'AB'},
                      'Z': f'While {X} and {Y} stayed outside, {Z} came into the room'}}
    return f'{X} {pred}', f'{Y} {pred}', f'{pr} {pred}', ctx


# kind of premise for (polarity of S, selected meaning), per frame of the premise; 'Z' (a third person) only in pronoun frames
KIND = {'cond': {('aff', 'A'): 'MP', ('aff', 'B'): 'AC', ('aff', 'Z'): 'foil', ('neg', 'B'): 'MT', ('neg', 'A'): 'DA', ('neg', 'Z'): 'foiln'},
        'disj': {('neg', 'A'): 'DS1', ('neg', 'B'): 'DS2', ('neg', 'Z'): 'foiln'}}
# the review log: the author's corrections, in order (template_bcontext_review.py prints it)
REVIEW_LOG = []
