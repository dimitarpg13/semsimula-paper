"""Draft item bank for the analogy and in-context test set (companion_notes/semantic_templates/Analogy_and_Context_Test_Set.md).
Drafted by the agent on 2026-11-26, 20 frames per level, for the author's review; not yet used by any extraction.

Every restatement has a foil that shares as many tokens with the clause it is compared with (checked by template_analogy_review.py).
Sentences are lower case without a final full stop, as in template_fresh_clauses.py; the generator capitalises them.
The forms are built as in the current test: MP = conditional + restatement of P, MT = conditional + denial of Q, AC = conditional +
restatement of Q, DA = conditional + denial of P, DS1 = disjunctive + denial of P, DS2 = disjunctive + denial of Q, the affirmed
disjunct = disjunctive + restatement of P. For clause-level banks (BASE, L3, L5, L6) Q is another entry of the same bank, as now.
"""


def cop(subj, pred):
    return f'{subj} is {pred}', f'{subj} is not {pred}'


def neg(s):
    """denial of a sentence with one 'is' or 'are'"""
    for a in (' is ', ' are '):
        if a in f' {s} ':
            return f' {s} '.replace(a, a.rstrip() + ' not ', 1).strip()
    raise ValueError(s)


# L0 verbatim, L1 one word, L2 full paraphrase: one bank, so the three levels are paired by clause.
# (subject, predicate, near-miss predicate, one-word paraphrase, paraphrased subject, paraphrased predicate, foil predicate for L2)
_BASE = [
    ('the road', 'closed', 'long', 'shut', 'the street', 'blocked', 'wide'),
    ('the coffee', 'cold', 'strong', 'chilly', 'the brew', 'cool', 'bitter'),
    ('the door', 'open', 'red', 'ajar', 'the entrance', 'unclosed', 'narrow'),
    ('the infant', 'asleep', 'hungry', 'sleeping', 'the baby', 'slumbering', 'swaddled'),
    ('the shop', 'busy', 'new', 'crowded', 'the store', 'packed', 'painted'),
    ('the soldier', 'brave', 'young', 'fearless', 'the fighter', 'courageous', 'famous'),
    ('the lake', 'frozen', 'deep', 'icy', 'the pond', 'iced over', 'shallow'),
    ('the car', 'fast', 'blue', 'quick', 'the vehicle', 'speedy', 'rusty'),
    ('the child', 'happy', 'short', 'cheerful', 'the youngster', 'joyful', 'clever'),
    ('the room', 'dark', 'large', 'dim', 'the chamber', 'unlit', 'square'),
    ('the merchant', 'wealthy', 'tall', 'rich', 'the trader', 'affluent', 'talkative'),
    ('the singer', 'famous', 'hungry', 'renowned', 'the vocalist', 'celebrated', 'seated'),
    ('the bridge', 'unsafe', 'old', 'dangerous', 'the crossing', 'hazardous', 'wooden'),
    ('the sofa', 'wide', 'old', 'broad', 'the couch', 'broad', 'soft'),
    ('the teacher', 'angry', 'tall', 'furious', 'the instructor', 'cross', 'bald'),
    ('the task', 'difficult', 'urgent', 'hard', 'the job', 'tough', 'boring'),
    ('the film', 'boring', 'long', 'dull', 'the movie', 'tedious', 'French'),
    ('the gift', 'expensive', 'small', 'costly', 'the present', 'pricey', 'wrapped'),
    ('the lawyer', 'honest', 'busy', 'truthful', 'the attorney', 'trustworthy', 'wealthy'),
    ('the engine', 'noisy', 'small', 'loud', 'the motor', 'roaring', 'rusty'),
]
BASE = []
for s, p, nr, sy, ps, pp, pf in _BASE:
    aff, ng = cop(s, p); BASE.append(dict(subj=s, aff=aff, neg=ng, near=cop(s, nr)[0], near_neg=cop(s, nr)[1], syn=cop(s, sy)[0], syn_neg=cop(s, sy)[1],
                                         para=cop(ps, pp)[0], para_neg=cop(ps, pp)[1], pfoil=cop(ps, pf)[0], pfoil_neg=cop(ps, pf)[1]))

# L3 entailed case: case entails the clause, ncase entails its denial; each foil shares as many tokens and decides neither.
L3 = [dict(aff=a, case=c, case_foil=cf, ncase=n, ncase_foil=nf) for a, c, cf, n, nf in [
    ('it rains', 'a storm is pouring down on the pitch', 'a crowd is pouring onto the pitch', 'the sky is cloudless and the ground is dry', 'the stands are crowded and the ground is level'),
    ('the guest is ill', 'the guest has a high fever', 'the guest has a high collar', 'the guest has just finished a marathon', 'the guest has just finished a novel'),
    ('the animal is a mammal', 'the animal is a whale', 'the animal is a pet', 'the animal is a shark', 'the animal is in a zoo'),
    ('the metal is precious', 'the metal is gold', 'the metal is grey', 'the metal is iron', 'the metal is shiny'),
    ('the container holds liquid', 'the container holds milk', 'the container holds a lot', 'the container holds rice', 'the container holds two kilos'),
    ('the house is empty', 'all the tenants have moved out of the house', 'all the tenants have argued about the house', 'all the tenants are at home in the house', 'all the tenants are fond of the house'),
    ('the team won', 'the team scored three goals and conceded none', 'the team scored three goals and received two yellow cards', 'the team lost every one of its games', 'the team travelled four hours by coach'),
    ('the plant is dead', 'the plant has withered to brown stalks', 'the plant has been moved onto a porch', 'the plant has burst into bloom', 'the plant has been watered daily'),
    ('the bird can fly', 'the bird is soaring over the valley', 'the bird is singing over the valley', 'the bird is a penguin', 'the bird is in a cage'),
    ('the water is hot', 'the water is boiling', 'the water is murky', 'the water has frozen solid', 'the water has turned green'),
    ('the prisoner escaped', 'the prisoner climbed over the wall and fled', 'the prisoner climbed a wall in the gym', 'the prisoner is still in his cell', 'the prisoner is still very quiet'),
    ('the shop is closed', 'the shutters of the shop are down and locked', 'the shutters of the shop are green and dusty', "customers are queuing at the shop's tills", "customers are talking about the shop's prices"),
    ('the man is married', 'the man has a wife', 'the man has a wallet', 'the man is a bachelor', 'the man is a baker'),
    ('the film was long', 'the film ran for four hours', 'the film ran for four weeks', 'the film was over in ten minutes', 'the film was shown in ten cities'),
    ('the number is even', 'the number is twelve', 'the number is large', 'the number is seven', 'the number is written down'),
    ('the road is wet', 'puddles cover the road', 'potholes cover the road', 'dust is blowing across the road', 'smoke is blowing across the road'),
    ('the drawer is locked', 'the drawer will not open without the key', 'the drawer will not close without the stopper', 'the drawer slides open at a touch', 'the drawer squeaks at a touch'),
    ('the soup is salty', 'the soup tastes like seawater', 'the soup looks like seawater', 'the cook added no salt at all to the soup', 'the cook added no pepper at all to the soup'),
    ('the window is broken', 'shards of the window lie on the floor', 'photos of the window lie on the floor', 'the window has just passed its inspection', 'the window has just been cleaned'),
    ('the horse is tired', 'the horse is panting and stumbling', 'the horse is brown and white', 'the horse is galloping eagerly', 'the horse is grazing quietly'),
]]

# L4 reference: the second sentence names its subject by a pronoun only. The foil is the same sentence with the other clause's
# pronoun (it says the predicate of P about Q's subject). Revised 2026-11-26 with the author: he and she only, every subject
# unambiguously gendered, both predicates plausible for either person (so no foil can be rejected as nonsense), and P is "he" in
# 10 frames and "she" in 10. Both pronouns take "is", so restatement and foil differ in one token, which neither clause contains.
# (P subject, P predicate, P pronoun, Q subject, Q predicate, Q pronoun)
_L4 = [
    ('the king', 'ill', 'he', 'the queen', 'in charge', 'she'),
    ('the waitress', 'late', 'she', 'the landlord', 'busy', 'he'),
    ('the girl', 'frightened', 'she', 'the old man', 'cheerful', 'he'),
    ('the uncle', 'asleep', 'he', 'the aunt', 'cooking', 'she'),
    ('the widow', 'worried', 'she', 'the priest', 'hopeful', 'he'),
    ('the boy', 'lost', 'he', 'the mother', 'searching', 'she'),
    ('the grandmother', 'pleased', 'she', 'the grandson', 'shy', 'he'),
    ('the nun', 'silent', 'she', 'the bishop', 'speaking', 'he'),
    ('the husband', 'away', 'he', 'the wife', 'anxious', 'she'),
    ('the bride', 'nervous', 'she', 'the groom', 'calm', 'he'),
    ('Anna', 'tired', 'she', 'Tom', 'driving', 'he'),
    ('the prince', 'abroad', 'he', 'the princess', 'unhappy', 'she'),
    ('the landlady', 'upset', 'she', 'the postman', 'early', 'he'),
    ('the father', 'proud', 'he', 'the daughter', 'graduating', 'she'),
    ('the schoolboy', 'crying', 'he', 'the headmistress', 'stern', 'she'),
    ('the duke', 'bankrupt', 'he', 'the duchess', 'generous', 'she'),
    ('the grandfather', 'asleep', 'he', 'the granddaughter', 'bored', 'she'),
    ('Maria', 'angry', 'she', 'Peter', 'apologising', 'he'),
    ('the policeman', 'suspicious', 'he', 'the woman', 'innocent', 'she'),
    ('the actress', 'impatient', 'she', 'the cameraman', 'tired', 'he'),
]
L4 = []
for ps, pp, pr, qs, qp, qr in _L4:
    L4.append(dict(P=cop(ps, pp)[0], Q=cop(qs, qp)[0], res=cop(pr, pp)[0], res_neg=cop(pr, pp)[1], foil=cop(qr, pp)[0], foil_neg=cop(qr, pp)[1],
                   resQ=cop(qr, qp)[0], resQ_neg=cop(qr, qp)[1], foilQ=cop(pr, qp)[0], foilQ_neg=cop(pr, qp)[1]))

# L5 restructured: the clause said again with another syntax (nominalisation, passive); the foil keeps the same frame and words of
# the clause, with the event replaced.
L5 = [dict(aff=a, res=r, foil=f, res_neg=rn, foil_neg=fn) for a, r, f, rn, fn in [
    ('the council approves the plan', "the plan has the council's approval", "the plan has the council's attention", "the plan lacks the council's approval", "the plan lacks the council's attention"),
    ('the company hires the graduate', 'the graduate gets a job at the company', 'the graduate gets a tour of the company', 'the graduate gets no job at the company', 'the graduate gets no tour of the company'),
    ('the doctor examined the patient', 'the patient had an examination by the doctor', 'the patient had a conversation with the doctor', 'the patient had no examination by the doctor', 'the patient had no conversation with the doctor'),
    ('the police arrested the thief', 'the thief is under arrest by the police', 'the thief is under watch by the police', 'the thief is not under arrest by the police', 'the thief is not under watch by the police'),
    ('the jury acquitted the defendant', 'the defendant won an acquittal from the jury', 'the defendant won a smile from the jury', 'the defendant won no acquittal from the jury', 'the defendant won no smile from the jury'),
    ('the committee rejected the proposal', 'the proposal met with rejection from the committee', 'the proposal met with interest from the committee', 'the proposal met with no rejection from the committee', 'the proposal met with no interest from the committee'),
    ('the team celebrated the victory', 'there was a celebration of the victory by the team', 'there was a discussion of the victory by the team', 'there was no celebration of the victory by the team', 'there was no discussion of the victory by the team'),
    ('the bank refused the loan', 'the loan met with a refusal from the bank', 'the loan met with a review from the bank', 'the loan met with no refusal from the bank', 'the loan met with no review from the bank'),
    ('the army invaded the island', 'the island suffered an invasion by the army', 'the island hosted a parade by the army', 'the island suffered no invasion by the army', 'the island hosted no parade by the army'),
    ('the singer cancelled the concert', "the concert's cancellation came from the singer", "the concert's announcement came from the singer", "no cancellation of the concert came from the singer", "no announcement of the concert came from the singer"),
    ('the government raised taxes', 'taxes saw an increase by the government', 'taxes saw a review by the government', 'taxes saw no increase by the government', 'taxes saw no review by the government'),
    ('the river flooded the village', 'the village got a flood from the river', 'the village got its water from the river', 'the village got no flood from the river', 'the village got no water from the river'),
    ('the student completed the course', 'the course saw completion by the student', 'the course saw criticism by the student', 'the course saw no completion by the student', 'the course saw no criticism by the student'),
    ('the chef burnt the steak', "the chef's steak is charred", "the chef's steak is tender", "the chef's steak is not charred", "the chef's steak is not tender"),
    ('the coach praised the players', 'the players received praise from the coach', 'the players received drinks from the coach', 'the players received no praise from the coach', 'the players received no drinks from the coach'),
    ('the nurse vaccinated the child', 'the child got a vaccination from the nurse', 'the child got a sticker from the nurse', 'the child got no vaccination from the nurse', 'the child got no sticker from the nurse'),
    ('the critics loved the film', "the film won the critics' love", "the film won the critics' attention", "the film did not win the critics' love", "the film did not win the critics' attention"),
    ('the guard opened the gate', 'the opening of the gate was done by the guard', 'the painting of the gate was done by the guard', 'the opening of the gate was not done by the guard', 'the painting of the gate was not done by the guard'),
    ('the voters elected the mayor', 'the mayor won election by the voters', 'the mayor won applause from the voters', 'the mayor won no election by the voters', 'the mayor won no applause from the voters'),
    ('the fire destroyed the barn', "the barn's destruction was caused by the fire", "the barn's evacuation was caused by the fire", "the barn's destruction was not caused by the fire", "the barn's evacuation was not caused by the fire"),
]]

# L6 new domain: L2 paraphrases in domains absent from the development clauses (law, medicine, machines, fiction).
_L6 = [
    ('the contract', 'void', 'the agreement', 'invalid', 'lengthy'), ('the witness', 'credible', 'the testifier', 'believable', 'nervous'),
    ('the will', 'forged', 'the testament', 'faked', 'handwritten'), ('the lease', 'expired', 'the tenancy', 'over', 'cheap'),
    ('the defendant', 'liable', 'the accused', 'responsible', 'elderly'), ('the tumour', 'benign', 'the neoplasm', 'harmless', 'round'),
    ('the fracture', 'healed', 'the break', 'mended', 'visible'), ('the disease', 'contagious', 'the illness', 'catching', 'seasonal'),
    ('the pill', 'tiny', 'the tablet', 'minuscule', 'white'), ('the operation', 'risky', 'the surgery', 'dangerous', 'lengthy'),
    ('the photocopier', 'broken', 'the copier', 'out of order', 'beige'), ('the lorry', 'overloaded', 'the truck', 'overburdened', 'muddy'),
    ('the bicycle', 'rusty', 'the bike', 'corroded', 'folding'), ('the aircraft', 'grounded', 'the plane', 'barred from flying', 'enormous'),
    ('the sensor', 'faulty', 'the detector', 'defective', 'sensitive'), ('the dragon', 'awake', 'the wyrm', 'stirring', 'ancient'),
    ('the spell', 'broken', 'the enchantment', 'undone', 'powerful'), ('the wizard', 'slain', 'the sorcerer', 'killed', 'bearded'),
    ('the castle', 'haunted', 'the fortress', 'ghost-ridden', 'well-guarded'), ('the prophecy', 'fulfilled', 'the foretelling', 'realised', 'obscure'),
]
L6 = []
for s, p, ps, pp, pf in _L6:
    L6.append(dict(aff=cop(s, p)[0], neg=cop(s, p)[1], para=cop(ps, pp)[0], para_neg=cop(ps, pp)[1], pfoil=cop(ps, pf)[0], pfoil_neg=cop(ps, pf)[1]))

# C1 alias in context: a context sentence names two people; the clause uses a role, the restatement the right name, the foil the
# other name with the same predicate. Only the context links the name to the role.
_C1 = [
    ('Mr Hale runs the only bakery in town, and Mrs Roe runs the flower shop.', 'the baker', 'ill', 'the bakery stays shut', 'Mr Hale', 'Mrs Roe'),
    ("Dr Patel is the town's only dentist; Dr Okafor is its vet.", 'the dentist', 'away', 'the clinic cancels its appointments', 'Dr Patel', 'Dr Okafor'),
    ('Lena captains the hockey team; Sara coaches it.', 'the captain', 'injured', 'the team forfeits the match', 'Lena', 'Sara'),
    ('Mr Brook drives the school bus; Mr Finch teaches music.', 'the driver', 'sick', 'the children walk to school', 'Mr Brook', 'Mr Finch'),
    ('Our neighbour Ruth is the mayor; her brother Ian is the treasurer.', 'the mayor', 'abroad', 'the council meeting is postponed', 'Ruth', 'Ian'),
    ('Mr Moss commands the ship; Mr Reed is its cook.', 'the commander', 'drunk', 'the voyage is delayed', 'Mr Moss', 'Mr Reed'),
    ('Professor Lind chairs the panel; Dr Hart takes the minutes.', 'the chair', 'late', 'the hearing is delayed', 'Professor Lind', 'Dr Hart'),
    ('Pedro owns the café on the corner; Marco works at the bank.', 'the café owner', 'on holiday', 'the café is closed', 'Pedro', 'Marco'),
    ('The goalkeeper is a woman called Joy; the striker is called Kim.', 'the goalkeeper', 'tired', 'the defence is weak', 'Joy', 'Kim'),
    ('Mr Grant is the night guard at the museum; Mr Adler is its curator.', 'the guard', 'asleep', 'the alarm goes unanswered', 'Mr Grant', 'Mr Adler'),
    ('Nora manages the hotel; Vera cleans the rooms.', 'the manager', 'unavailable', 'the complaint is ignored', 'Nora', 'Vera'),
    ('Ivan flies the morning plane; Oleg sits beside him as co-pilot.', 'the pilot', 'unwell', 'the flight is cancelled', 'Ivan', 'Oleg'),
    ('Mrs Kahn is the head teacher; Miss Doyle teaches art.', 'the head teacher', 'absent', 'the assembly is cancelled', 'Mrs Kahn', 'Miss Doyle'),
    ("Bruno is the band's drummer; Felix sings.", 'the drummer', 'ill', 'the concert is called off', 'Bruno', 'Felix'),
    ('Mrs Ward hears the case; Mr Slade defends the accused.', 'the judge', 'delayed', 'the trial is adjourned', 'Mrs Ward', 'Mr Slade'),
    ('Tom is the village blacksmith; Will is the miller.', 'the blacksmith', 'busy', 'the horses wait unshod', 'Tom', 'Will'),
    ("Amira is the expedition's doctor; Jonas is its guide.", 'the doctor', 'lost', 'the expedition turns back', 'Amira', 'Jonas'),
    ('Mr Cole keeps the pub; Mr Dunn is one of its regulars.', 'the landlord', 'angry', 'the pub closes early', 'Mr Cole', 'Mr Dunn'),
    ('Hana conducts the orchestra; Mei plays first violin.', 'the conductor', 'late', 'the rehearsal is cut short', 'Hana', 'Mei'),
    ("Mr Price owns the shop; Mr Shah keeps its accounts.", 'the owner', 'worried', 'the shop is sold', 'Mr Price', 'Mr Shah'),
]
C1 = []
for ctx, role, pr, q, right, wrong in _C1:
    C1.append(dict(context=ctx, P=cop(role, pr)[0], Q=q, res=cop(right, pr)[0], res_neg=cop(right, pr)[1], foil=cop(wrong, pr)[0], foil_neg=cop(wrong, pr)[1]))

# C2 same word, sense differs: the clause's sense is fixed by its consequent; the restatement adds a word of that sense, the foil a
# word of the other sense, as many tokens. C2' pairs a paraphrase in the clause's sense (para) with the same foil (a false friend).
_C2 = [
    ('the bank is closed', 'we cannot withdraw cash', 'the bank on the high street is closed', 'the bank of the river is closed', 'the branch on the high street has shut'),
    ('the bat is in the shed', 'the game cannot start', 'the cricket bat is in the shed', 'the fruit bat is in the shed', "the batsman's willow is kept in the hut"),
    ('the crane is standing by the lake', 'the birdwatchers gather', 'the white crane is standing by the lake', 'the steel crane is standing by the lake', 'the long-legged wader waits at the water\'s edge'),
    ('the mouse is under the desk', 'the cat is excited', 'the field mouse is under the desk', 'the computer mouse is under the desk', 'the little rodent hides beneath the table'),
    ('the letter is missing', 'the post office apologises', 'the posted letter is missing', 'the capital letter is missing', 'the mailed envelope has gone astray'),
    ('the plant is dying', 'we water it more often', 'the potted plant is dying', 'the steel plant is dying', 'the house fern is wilting'),
    ('the court is empty', 'the trial cannot go ahead', 'the criminal court is empty', 'the tennis court is empty', 'the hall of justice has nobody in it'),
    ('the organ is damaged', 'the surgeon cannot transplant it', 'the donor organ is damaged', 'the church organ is damaged', 'the harvested kidney is injured'),
    ('the club is new', 'the golfer practises more', 'the golf club is new', 'the youth club is new', "the golfer's driver was bought last week"),
    ('the ring is too small', 'the bride cannot wear it', 'the wedding ring is too small', 'the boxing ring is too small', 'the gold band will not fit her finger'),
    ('the nail is bent', 'the carpenter pulls it out', 'the steel nail is bent', 'the toe nail is bent', 'the iron spike is crooked'),
    ('the palm is dry', 'the gardener waters it', 'the date palm is dry', 'the left palm is dry', 'the date tree is parched'),
    ('the glasses are broken', 'the reader cannot see the page', 'the reading glasses are broken', 'the wine glasses are broken', 'her spectacles have cracked'),
    ('the key is lost', 'we cannot open the front door', 'the door key is lost', 'the answer key is lost', 'the latchkey has gone missing'),
    ('the jam is terrible', 'we will be late for work', 'the traffic jam is terrible', 'the strawberry jam is terrible', 'the congestion on the motorway is awful'),
    ('the fan is broken', 'the room is too hot', 'the ceiling fan is broken', 'the football fan is broken', 'the overhead cooler has stopped working'),
    ('the deck is wet', 'the sailors slip', 'the upper deck is wet', 'the card deck is wet', "the boat's planking is soaked"),
    ('the pupil is small', 'the light is too bright', 'the eye pupil is small', 'the school pupil is small', 'the iris opening has narrowed'),
    ('the bark is rough', 'the climber grips it easily', 'the tree bark is rough', "the dog's bark is rough", "the oak's skin is coarse"),
    ('the record is broken', 'the music stops', 'the vinyl record is broken', 'the world record is broken', 'the old LP has cracked'),
]
C2 = [dict(P=p, Q=q, res=r, res_neg=neg(r), foil=f, foil_neg=neg(f), para=pa) for p, q, r, f, pa in _C2]

# C3 ellipsis: the second sentence keeps only a pronoun and "does": the clause's subject and verb must come from the context.
# The foil is the other clause's pronoun (the AC twin). (P subject, P verb, P pronoun, Q subject, Q verb, Q pronoun)
# Redrafted 2026-11-26 as L4 was: one man and one woman per frame, unambiguous genders, verbs plausible for either person;
# P is "he" in 10 frames and "she" in 10.
_C3 = [
    ('the uncle', 'snores', 'he', 'the aunt', 'complains', 'she'), ('the grandmother', 'sings', 'she', 'the grandfather', 'smiles', 'he'),
    ('Tom', 'oversleeps', 'he', 'Anna', 'leaves', 'she'), ('the princess', 'dances', 'she', 'the king', 'weeps', 'he'),
    ('the father', 'swears', 'he', 'the daughter', 'blushes', 'she'), ('the nun', 'prays', 'she', 'the bishop', 'listens', 'he'),
    ('the boy', 'cheats', 'he', 'the mother', 'worries', 'she'), ('the landlady', 'shouts', 'she', 'the postman', 'apologises', 'he'),
    ('the bride', 'cries', 'she', 'the groom', 'laughs', 'he'), ('the duke', 'gambles', 'he', 'the duchess', 'moves out', 'she'),
    ('Maria', 'wins', 'she', 'Peter', 'cheers', 'he'), ('the grandson', 'whistles', 'he', 'the granddaughter', 'hums', 'she'),
    ('the policeman', 'hesitates', 'he', 'the woman', 'escapes', 'she'), ('the waitress', 'stumbles', 'she', 'the landlord', 'frowns', 'he'),
    ('the husband', 'travels', 'he', 'the wife', 'stays home', 'she'), ('the queen', 'abdicates', 'she', 'the prince', 'rules', 'he'),
    ('the actress', 'improvises', 'she', 'the cameraman', 'keeps filming', 'he'), ('the schoolboy', 'misbehaves', 'he', 'the headmistress', 'intervenes', 'she'),
    ('the widow', 'remarries', 'she', 'the son', 'objects', 'he'), ('the brother', 'calls', 'he', 'the sister', 'answers', 'she'),
]
C3 = [dict(P=f'{ps} {pv}', Q=f'{qs} {qv}', res=f'{pr} does', res_neg=f'{pr} does not', foil=f'{qr} does', foil_neg=f'{qr} does not') for ps, pv, pr, qs, qv, qr in _C3]

# C4 denial by an incompatible value: the second sentence denies a clause only through what is known about it (green denies red).
# Q_inc denies Q (MT analogue), P_inc denies P (DS1 and DA analogues); each foil keeps the subject and decides nothing.
C4 = [dict(P=p, Q=q, Q_inc=qi, Q_foil=qf, P_inc=pi, P_foil=pf) for p, q, qi, qf, pi, pf in [
    ('the alarm is on', 'the light is red', 'the light is green', 'the light is bright', 'the alarm is off', 'the alarm is loud'),
    ('the shop is open', 'the shutters are up', 'the shutters are down', 'the shutters are metal', 'the shop is shut', 'the shop is small'),
    ('the patient is alive', 'his heart is beating', 'his heart is still', 'his heart is large', 'the patient is dead', 'the patient is old'),
    ('the glass is full', 'the jug is empty', 'the jug is brimming', 'the jug is ceramic', 'the glass is drained', 'the glass is cold'),
    ('the meeting is on Monday', 'the room is booked', 'the room is free', 'the room is large', 'the meeting is on Friday', 'the meeting is on schedule'),
    ('the signal is red', 'the cars stop', 'the cars keep moving', 'the cars keep honking', 'the signal is green', 'the signal is bright'),
    ('the child is awake', 'the nursery is noisy', 'the nursery is silent', 'the nursery is blue', 'the child is sleeping', 'the child is small'),
    ('the coin landed heads', 'Anna wins the bet', 'Anna loses the bet', 'Anna doubles the bet', 'the coin landed tails', 'the coin landed quickly'),
    ('the oven is hot', 'the bread is baking', 'the bread is raw', 'the bread is wholemeal', 'the oven is cold', 'the oven is new'),
    ('the student passed', 'her parents are proud', 'her parents are ashamed', 'her parents are tall', 'the student failed', 'the student studied'),
    ('the door is closed', 'the cat is inside', 'the cat is outside', 'the cat is ginger', 'the door is ajar', 'the door is painted'),
    ('the tank is full', 'the gauge points high', 'the gauge points low', 'the gauge points steadily', 'the tank is dry', 'the tank is metal'),
    ('the river is high', 'the ford is dangerous', 'the ford is safe', 'the ford is wide', 'the river is low', 'the river is brown'),
    ('the road is dry', 'the race goes ahead', 'the race is called off', 'the race is televised', 'the road is flooded', 'the road is long'),
    ('the flag is up', 'the castle is occupied', 'the castle is deserted', 'the castle is ancient', 'the flag is down', 'the flag is blue'),
    ('the tide is in', 'the beach is covered', 'the beach is exposed', 'the beach is sandy', 'the tide is out', 'the tide is strong'),
    ('the fridge is working', 'the milk is cold', 'the milk is warm', 'the milk is fresh', 'the fridge is broken', 'the fridge is white'),
    ('the train is early', 'the platform is crowded', 'the platform is deserted', 'the platform is long', 'the train is late', 'the train is new'),
    ('the soup is hot', 'the bowl is warm', 'the bowl is icy', 'the bowl is round', 'the soup is cold', 'the soup is thick'),
    ('the switch is up', 'the lamp is lit', 'the lamp is dark', 'the lamp is brass', 'the switch is down', 'the switch is plastic'),
]]

# The author's corrections, in order; template_analogy_review.py prints them on the index page.
REVIEW_LOG = [
    ('2026-11-26', 'L0 to L2, frame 3', '"the entrance is unbarred" replaced by "the entrance is unclosed": unbarred means unlocked, not open (author).'),
    ('2026-11-26', 'L0 to L2, frame 11', 'frame "the bread is stale" (L1 "dry", L2 "the loaf is dried out") replaced by "the merchant is wealthy" (L1 "rich", L2 "the trader is affluent", foils "tall", "talkative"): dry is a texture, bread can be dried on purpose and be fresh, and stale has no one-word synonym (author).'),
    ('2026-11-26', 'L0 to L2, frame 14', 'frame "the river is wide" (L2 "the waterway is expansive") replaced by "the sofa is wide" (L1 "broad", L2 "the couch is broad", foils "old", "soft"): expansive speaks of extent (length or area) rather than width (author), and a waterway is a broader term than a river (agent).'),
    ('2026-11-26', 'L0 to L2, frame 4', 'frame "the patient is asleep" (L2 "the sick man is dozing") replaced by "the infant is asleep" (L1 "sleeping", L2 "the baby is slumbering", foils "hungry", "swaddled"): a sick man is not the same as a patient (author); patient has no exact synonym, and dozing is a light sleep (agent).'),
    ('2026-11-26', 'L3, frame 2', 'denial case and foil "has just run a marathon / a bath" replaced by "has just finished a marathon / a novel": "to run a bath" is an idiom whose meaning is not clear to every reader (author), and it plays on two senses of "run" (agent).'),
    ('2026-11-26', 'L3, frame 7', 'case foil "scored three goals and conceded four" replaced by "scored three goals and received two yellow cards": three against four is a loss, so the old foil denied the clause instead of deciding nothing (agent). The denial case "lost every one of its games" stays: a denial case need only exclude the clause, not restate its denial (author).'),
    ('2026-11-26', 'L3, frame 19', 'kept: "the window has just passed its inspection" excludes "the window is broken" in the ordinary reading (a window that passed inspection was not broken, unless the inspector failed) (author).'),
    ('2026-11-26', 'L4, all frames', 'redrafted: he and she only, unambiguous genders, predicates plausible for either person, P "he" in 10 frames and "she" in 10. The eight frames that paired "it" with a person (3, 5, 7, 9, 13, 15, 17, 20) gave absurd foils ("She is barking.") that a reader rejects as nonsense without resolving the reference; "the waiter" replaced (either gender); frames 12 and 16 reversed for balance (author and agent).'),
    ('2026-11-26', 'C3, all frames', 'redrafted as L4 was: one man and one woman per frame, he and she only, unambiguous genders, verbs plausible for either person, P "he" in 10 frames and "she" in 10. Thirteen frames paired "it" with a person, so animacy alone told the restatement from the foil without recovering the clause (author and agent).'),
    ('2026-11-26', 'C4, frame 2', 'Q "the door is unlocked" with its denial "the door is bolted" and foil "the door is wooden" replaced by "the shutters are up", "the shutters are down", "the shutters are metal": a door can be unlocked and still bolted, so "bolted" does not deny "unlocked" (author); "bolted" also means fastened with bolts or ran off (agent).'),
    ('2026-11-26', 'C4, frame 12', 'foil of Q "the gauge points north" replaced by "the gauge points steadily": a fuel gauge does not point to a compass direction, so the old foil read as nonsense (agent).'),
    ('2026-11-26', 'L0 to L2, frame 16', 'frame "the meal is ready" (L1 "prepared", L2 "dinner is done") replaced by "the task is difficult" (L1 "hard", L2 "the job is tough", foils "urgent", "boring"): dinner is narrower than a meal and lacked its article, and "done" can mean finished eating (agent; author agreed).'),
    ('2026-11-26', 'L0 to L2, frame 17', 'frame "the hill is steep" (L1 "sheer", L2 "the slope is precipitous") replaced by "the film is boring" (L1 "dull", L2 "the movie is tedious", foils "long", "French"): sheer means vertical, a slope is part of a hill, and precipitous is rare and stronger than steep (agent; author agreed).'),
    ('2026-11-26', 'L0 to L2, frame 18', 'frame "the box is sealed" (L1 "shut", L2 "the crate is taped closed") replaced by "the gift is expensive" (L1 "costly", L2 "the present is pricey", foils "small", "wrapped"): a crate is not a box, and shut is not sealed and repeated frame 1 (agent; author agreed).'),
    ('2026-11-26', 'L6', 'subjects "the infection / the illness" replaced by "the disease is contagious / the illness is catching": the subjects are not synonyms (an infection is the invasion by a germ, an illness the state of being unwell) (author).'),
    ('2026-11-26', 'L6', 'subjects "the patient / the sufferer" replaced by "the pill is tiny / the tablet is minuscule" (foil "white"): the subjects are not synonyms (a patient is under care, a sufferer is anyone with an affliction) (author).'),
    ('2026-11-26', 'L6', 'subjects "the wound / the cut" replaced by "the operation is risky / the surgery is dangerous" (foil "lengthy"): the subjects are not synonyms (a cut is one kind of wound) (author).'),
    ('2026-11-26', 'L6', 'subjects "the valve / the tap" replaced by "the photocopier is broken / the copier is out of order" (foil "beige"): the subjects are not synonyms (a tap is one kind of valve) (author). The first replacement, the lift and the elevator, was dropped because "the elevator is stuck" was taken for a development clause; it is in fact a clause of the earlier robust-matcher test set (template_robust_clauses.py, TEST), not of the development set, so the reason was wrong. The photocopier frame is kept, as it serves equally (agent, corrected the same day).'),
    ('2026-11-26', 'L6', 'subjects "the battery / the cell" replaced by "the lorry is overloaded / the truck is overburdened" (foil "muddy"): the subjects are not synonyms (a battery is made of cells, and "cell" has other senses) (author).'),
    ('2026-11-26', 'L6', 'subjects "the circuit / the wiring" replaced by "the bicycle is rusty / the bike is corroded" (foil "folding"): the subjects are not synonyms (wiring is part of a circuit) (author).'),
    ('2026-11-26', 'C4, frame 14', 'kept: the denial "the race is called off" is a common phrasal verb and reads clearly; the proposed "the race is cancelled" is not needed (author).'),
    ('2026-11-26', 'L3, frame 1', 'foil of the denial case "the stands are crowded and the ground is hard" replaced by "... the ground is level": hard ground hints that it is dry, so the old foil leaned toward deciding (agent; author agreed).'),
    ('2026-11-26', 'L3, frame 4', 'clause "the metal is valuable" with denial case "the metal is scrap" replaced by "the metal is precious" with denial case "the metal is iron" (case gold and foils grey and shiny kept): scrap metal can be valuable, and iron is not a precious metal by definition (agent; author agreed).'),
    ('2026-11-26', 'L3, frame 11', 'foil of the case "the prisoner climbed onto the wall and sang" replaced by "the prisoner climbed a wall in the gym": the old foil was odd; a gym climbing wall says nothing about an escape (agent; author agreed).'),
    ('2026-11-26', 'L5, frame 18', 'restatement "the gate\'s opening was the guard\'s doing" and its foil replaced by "the opening of the gate was done by the guard" and "the painting of the gate was done by the guard": "was X\'s doing" is an idiom (agent; author agreed).'),
    ('2026-11-26', 'C2, frame 2', 'C2\u2032 paraphrase "the willow paddle is stored in the outbuilding" replaced by "the batsman\'s willow is kept in the hut": a paddle is not a bat (agent; author agreed).'),
    ('2026-11-26', 'C2, frame 9', 'C2\u2032 paraphrase "the iron he swings has just been bought" replaced by "the golfer\'s driver was bought last week": "he" had no antecedent, and an iron is only one kind of club (agent; author agreed).'),
    ('2026-11-26', 'C2, frame 19', 'foil "the dog bark is rough" replaced by "the dog\'s bark is rough": "dog bark" is not a natural compound (agent; author agreed).'),
    ('2026-11-26', 'L6', 'subjects "the appeal / the challenge" replaced by "the will is forged / the testament is faked" (foil "handwritten"): the subjects are not synonyms (a challenge is broader than an appeal) (agent; author agreed).'),
    ('2026-11-26', 'L6', 'subjects "the tumour / the growth" replaced by "the tumour is benign / the neoplasm is harmless" (foil "round"): the subjects are not synonyms (a growth is broader than a tumour; neoplasm is the medical synonym) (agent; author agreed).'),
    ('2026-11-26', 'L6', 'subjects "the turbine / the generator" replaced by "the aircraft is grounded / the plane is barred from flying" (foil "enormous"): the subjects are not synonyms (a turbine drives a generator; they are different machines) (agent; author agreed).'),
    ('2026-11-26', 'L6', 'subjects "the knight / the warrior" replaced by "the wizard is slain / the sorcerer is killed" (foil "bearded"): the subjects are not synonyms (a knight is one kind of warrior) (agent; author agreed).'),
]

LEVELS = dict(BASE=BASE, L3=L3, L4=L4, L5=L5, L6=L6, C1=C1, C2=C2, C3=C3, C4=C4)
