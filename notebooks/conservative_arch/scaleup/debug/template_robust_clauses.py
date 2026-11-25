"""Fresh test clauses for the robust template matcher (2026-11-25), drafted for the author to check. None of these subjects
is used in the clause sets that shaped the design. Each row: subject | predicate | its negation || paraphrase of the
predicate, same subject | its negation || near miss: another predicate, same subject | its negation."""
RAW = """
the office | is closed | is not closed || is shut | is not shut || is modern | is not modern
the pilot | is drunk | is not drunk || is intoxicated | is not intoxicated || is married | is not married
the rumor | is false | is not false || is untrue | is not untrue || is old | is not old
the coffee | is cold | is not cold || is chilled | is not chilled || is strong | is not strong
the soldier | flees | does not flee || runs away | does not run away || salutes | does not salute
the painting | is fake | is not fake || is counterfeit | is not counterfeit || is colorful | is not colorful
the farmer | is wealthy | is not wealthy || is rich | is not rich || is bald | is not bald
the tunnel | is dark | is not dark || is unlit | is not unlit || is long | is not long
the surgeon | is exhausted | is not exhausted || is worn out | is not worn out || is foreign | is not foreign
the contract | is void | is not void || is invalid | is not invalid || is public | is not public
the tourist | is lost | is not lost || is off course | is not off course || is hungry | is not hungry
the laptop | is damaged | is not damaged || is broken | is not broken || is silver | is not silver
the actress | is famous | is not famous || is well known | is not well known || is angry | is not angry
the fence | is high | is not high || is tall | is not tall || is white | is not white
the cook | quits | does not quit || resigns | does not resign || whistles | does not whistle
the exam | is hard | is not hard || is difficult | is not difficult || is optional | is not optional
the beach | is crowded | is not crowded || is packed | is not packed || is sandy | is not sandy
the engineer | is wrong | is not wrong || is mistaken | is not mistaken || is early | is not early
the trial | begins | does not begin || starts | does not start || bores the press | does not bore the press
the neighbor | shouts | does not shout || yells | does not yell || cooks | does not cook
the elevator | is stuck | is not stuck || is jammed | is not jammed || is tiny | is not tiny
the museum | is free | is not free || costs nothing | costs something || is round | is not round
the sailor | drowns | does not drown || dies at sea | does not die at sea || sings | does not sing
the prisoner | is guilty | is not guilty || is to blame | is not to blame || is thin | is not thin
the bakery | is cheap | is not cheap || is inexpensive | is not inexpensive || is nearby | is not nearby
the widow | is sad | is not sad || is unhappy | is not unhappy || is blonde | is not blonde
the harbor | is frozen | is not frozen || is covered in ice | is not covered in ice || is historic | is not historic
the clerk | is rude | is not rude || is impolite | is not impolite || is left handed | is not left handed
the lamp | glows | does not glow || shines | does not shine || falls | does not fall
the suitcase | is heavy | is not heavy || weighs a lot | does not weigh a lot || is leather | is not leather
the speech | is brief | is not brief || is short | is not short || is funny | is not funny
the nurse | helps the victim | does not help the victim || assists the victim | does not assist the victim || ignores the victim | does not ignore the victim
the village | is poor | is not poor || is impoverished | is not impoverished || is hilly | is not hilly
the ladder | is unsafe | is not unsafe || is dangerous | is not dangerous || is wooden | is not wooden
the editor | is furious | is not furious || is enraged | is not enraged || is talented | is not talented
the desert | is huge | is not huge || is enormous | is not enormous || is empty | is not empty
the butcher | lies | does not lie || tells falsehoods | does not tell falsehoods || limps | does not limp
the ticket | is costly | is not costly || is expensive | is not expensive || is purple | is not purple
the island | is remote | is not remote || is far away | is not far away || is green | is not green
the driver | is injured | is not injured || is hurt | is not hurt || is polite | is not polite
the theory | is correct | is not correct || is right | is not right || is recent | is not recent
the choir | is quiet | is not quiet || is silent | is not silent || is large | is not large
the tenant | leaves | does not leave || moves out | does not move out || complains | does not complain
the statue | is ancient | is not ancient || is very old | is not very old || is ugly | is not ugly
the poison | is deadly | is not deadly || is lethal | is not lethal || is yellow | is not yellow
the senator | is honest | is not honest || is truthful | is not truthful || is absent | is not absent
the carpet | is filthy | is not filthy || is dirty | is not dirty || is soft | is not soft
the orchestra | stops | does not stop || halts | does not halt || travels | does not travel
the crown | is stolen | is not stolen || is taken by thieves | is not taken by thieves || is golden | is not golden
the fisherman | is lucky | is not lucky || is fortunate | is not fortunate || is lonely | is not lonely
the bell | rings | does not ring || sounds | does not sound || cracks | does not crack
the professor | is sick | is not sick || is ill | is not ill || is French | is not French
the valley | is peaceful | is not peaceful || is calm | is not calm || is narrow | is not narrow
the boxer | wins | does not win || is victorious | is not victorious || eats | does not eat
the tower | is ruined | is not ruined || is destroyed | is not destroyed || is square | is not square
the merchant | is greedy | is not greedy || is avaricious | is not avaricious || is nervous | is not nervous
the puppy | is tired | is not tired || is sleepy | is not sleepy || is spotted | is not spotted
the glacier | shrinks | does not shrink || gets smaller | does not get smaller || moves | does not move
the dancer | is beautiful | is not beautiful || is lovely | is not lovely || is busy | is not busy
the sword | is blunt | is not blunt || is dull | is not dull || is famous | is not famous
"""
def parse(raw):
    out = []
    for line in raw.strip().splitlines():
        a, b, c = [x.strip() for x in line.split('||')]; s, p, n = [x.strip() for x in a.split('|')]
        (sp, sn), (np_, nn) = [x.strip() for x in b.split('|')], [x.strip() for x in c.split('|')]
        out.append(dict(subj=s, pred=p, npred=n, aff=f'{s} {p}', neg=f'{s} {n}', syn=f'{s} {sp}', syn_neg=f'{s} {sn}', near=f'{s} {np_}', near_neg=f'{s} {nn}'))
    return out
TEST = parse(RAW); assert len(TEST) == 60 and len({t['subj'] for t in TEST}) == 60
from template_matcher_clauses import RAW as OLD
DEV = []
for line in OLD.strip().splitlines():
    a, _, c = [x.strip() for x in line.split('||')]; s, p, n = [x.strip() for x in a.split('|')]; na, nn = [x.strip() for x in c.split('|')]
    DEV.append(dict(subj=s, pred=p, npred=n, aff=f'{s} {p}', neg=f'{s} {n}', near=na, near_neg=nn))
assert not {t['subj'] for t in TEST} & {t['subj'] for t in DEV}, {t['subj'] for t in TEST} & {t['subj'] for t in DEV}
