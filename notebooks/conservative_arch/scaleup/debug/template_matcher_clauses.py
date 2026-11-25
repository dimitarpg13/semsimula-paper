"""Test clauses for template_matcher_test.py (2026-11-25), drafted for the author to check.
Each row: subject | predicate | negated predicate || paraphrase (no content word shared) | its negation || near miss (same
subject, another predicate) | its negation."""
RAW = """
the road | is closed | is not closed || the highway has been shut | the highway has not been shut || the road is long | the road is not long
the child | is asleep | is not asleep || the kid has dozed off | the kid has not dozed off || the child is hungry | the child is not hungry
the company | makes a profit | does not make a profit || the firm earns money | the firm does not earn money || the company hires staff | the company does not hire staff
the door | is locked | is not locked || the entrance has been bolted | the entrance has not been bolted || the door is green | the door is not green
the soup | is hot | is not hot || the broth has been heated | the broth has not been heated || the soup is salty | the soup is not salty
the lawyer | wins the case | does not win the case || the attorney prevails in court | the attorney does not prevail in court || the lawyer reads the case | the lawyer does not read the case
the car | is broken | is not broken || the vehicle needs repair | the vehicle does not need repair || the car is expensive | the car is not expensive
the teacher | is ill | is not ill || the instructor feels sick | the instructor does not feel sick || the teacher is strict | the teacher is not strict
the train | arrives on time | does not arrive on time || the railway service is punctual | the railway service is not punctual || the train runs on diesel | the train does not run on diesel
the house | is for sale | is not for sale || the home is on the market | the home is not on the market || the house is for rent | the house is not for rent
the king | dies | does not die || the monarch passes away | the monarch does not pass away || the king speaks | the king does not speak
the doctor | is busy | is not busy || the physician has no spare time | the physician has spare time || the doctor is young | the doctor is not young
the window | is open | is not open || the glass pane has been raised | the glass pane has not been raised || the window is clean | the window is not clean
the boat | sinks | does not sink || the vessel goes under | the vessel does not go under || the boat turns | the boat does not turn
the movie | is boring | is not boring || the film is dull | the film is not dull || the movie is short | the movie is not short
the army | retreats | does not retreat || the troops pull back | the troops do not pull back || the army attacks | the army does not attack
the cake | is sweet | is not sweet || the dessert tastes of sugar | the dessert does not taste of sugar || the cake is burnt | the cake is not burnt
the shop | closes early | does not close early || the store shuts ahead of schedule | the store does not shut ahead of schedule || the shop opens early | the shop does not open early
the boy | is lying | is not lying || the lad tells an untruth | the lad does not tell an untruth || the boy is running | the boy is not running
the bank | lends money | does not lend money || the financial institution gives out loans | the financial institution does not give out loans || the bank loses money | the bank does not lose money
the garden | is dry | is not dry || the flower beds lack water | the flower beds do not lack water || the garden is small | the garden is not small
the scientist | is right | is not right || the researcher has it correct | the researcher does not have it correct || the scientist is famous | the scientist is not famous
the wall | collapses | does not collapse || the brick barrier falls down | the brick barrier does not fall down || the wall shines | the wall does not shine
the minister | quits | does not quit || the cabinet member steps down | the cabinet member does not step down || the minister travels | the minister does not travel
the phone | is stolen | is not stolen || the handset has been taken by a thief | the handset has not been taken by a thief || the phone is new | the phone is not new
the bird | sings | does not sing || the robin chirps a tune | the robin does not chirp a tune || the bird sleeps | the bird does not sleep
the room | is empty | is not empty || nobody is in the chamber | somebody is in the chamber || the room is warm | the room is not warm
the tax | goes up | does not go up || the levy increases | the levy does not increase || the tax goes to schools | the tax does not go to schools
the dog | is dangerous | is not dangerous || the hound poses a threat | the hound does not pose a threat || the dog is brown | the dog is not brown
the player | is injured | is not injured || the athlete has been hurt | the athlete has not been hurt || the player is tall | the player is not tall
the clock | is slow | is not slow || the timepiece lags behind | the timepiece does not lag behind || the clock is round | the clock is not round
the rent | is too high | is not too high || the monthly payment costs excessively much | the monthly payment does not cost excessively much || the rent is due today | the rent is not due today
the guest | leaves | does not leave || the visitor departs | the visitor does not depart || the guest laughs | the guest does not laugh
the factory | pollutes the air | does not pollute the air || the plant makes the atmosphere dirty | the plant does not make the atmosphere dirty || the factory builds trucks | the factory does not build trucks
the water | is clean | is not clean || the liquid is free of dirt | the liquid is not free of dirt || the water is cold | the water is not cold
the judge | is fair | is not fair || the magistrate shows no bias | the magistrate shows bias || the judge is late | the judge is not late
the letter | arrives | does not arrive || the envelope gets delivered | the envelope does not get delivered || the letter burns | the letter does not burn
the program | crashes | does not crash || the software stops working | the software does not stop working || the program prints text | the program does not print text
the mountain | is steep | is not steep || the peak has sharp slopes | the peak does not have sharp slopes || the mountain is far away | the mountain is not far away
the singer | is popular | is not popular || the vocalist has many fans | the vocalist does not have many fans || the singer is tired | the singer is not tired
the bridge | is safe | is not safe || the river crossing poses no danger | the river crossing poses danger || the bridge is narrow | the bridge is not narrow
the cat | hides | does not hide || the kitten keeps out of sight | the kitten does not keep out of sight || the cat eats | the cat does not eat
the food | is spoiled | is not spoiled || the meal has gone bad | the meal has not gone bad || the food is cheap | the food is not cheap
the mayor | lies to voters | does not lie to voters || the city leader deceives the electorate | the city leader does not deceive the electorate || the mayor talks to voters | the mayor does not talk to voters
the ship | is lost | is not lost || the freighter has gone missing | the freighter has not gone missing || the ship is heavy | the ship is not heavy
the girl | is happy | is not happy || the young lady feels joy | the young lady does not feel joy || the girl is thirsty | the girl is not thirsty
the engine | is loud | is not loud || the motor makes a lot of noise | the motor does not make a lot of noise || the engine is old | the engine is not old
the school | is far | is not far || the academy lies at a great distance | the academy does not lie at a great distance || the school is big | the school is not big
the thief | escapes | does not escape || the burglar gets away | the burglar does not get away || the thief sneezes | the thief does not sneeze
the crop | fails | does not fail || the harvest is ruined | the harvest is not ruined || the crop grows | the crop does not grow
the drug | works | does not work || the medicine is effective | the medicine is not effective || the drug sells | the drug does not sell
the war | ends | does not end || the fighting is over | the fighting is not over || the war spreads | the war does not spread
the author | is rich | is not rich || the writer has great wealth | the writer does not have great wealth || the author is shy | the author is not shy
the lake | freezes | does not freeze || the pond turns to ice | the pond does not turn to ice || the lake floods | the lake does not flood
the jacket | is torn | is not torn || the coat has a rip | the coat does not have a rip || the jacket is blue | the jacket is not blue
the manager | agrees | does not agree || the boss gives consent | the boss does not give consent || the manager waits | the manager does not wait
the tree | is dead | is not dead || the oak has no life left | the oak has life left || the tree is tall | the tree is not tall
the knife | is sharp | is not sharp || the blade cuts well | the blade does not cut well || the knife is heavy | the knife is not heavy
the hotel | is full | is not full || the inn has no vacant beds | the inn has vacant beds || the hotel is modern | the hotel is not modern
the witness | is honest | is not honest || the observer tells the truth | the observer does not tell the truth || the witness is nervous | the witness is not nervous
"""
TEST = []
for line in RAW.strip().splitlines():
    a, b, c = [x.strip() for x in line.split('||')]
    s, p, n = [x.strip() for x in a.split('|')]; pa, pn = [x.strip() for x in b.split('|')]; na, nn = [x.strip() for x in c.split('|')]
    TEST.append(dict(aff=f'{s} {p}', neg=f'{s} {n}', para=pa, para_neg=pn, near=na, near_neg=nn))
assert len(TEST) == 60, len(TEST)
