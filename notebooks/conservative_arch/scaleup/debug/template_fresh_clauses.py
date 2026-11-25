"""Second set of fresh test clauses for the template matchers (2026-11-25), drafted for the author to check. None of these
subjects is used in template_robust_clauses.py (DEV or TEST). Same layout: subject | predicate | its negation || paraphrase of
the predicate, same subject | its negation || near miss: another predicate, same subject | its negation."""
from template_robust_clauses import parse, TEST, DEV
RAW = """
the waiter | is slow | is not slow || is sluggish | is not sluggish || is Italian | is not Italian
the river | is deep | is not deep || is bottomless | is not bottomless || is brown | is not brown
the emperor | is cruel | is not cruel || is brutal | is not brutal || is old | is not old
the violin | is valuable | is not valuable || is precious | is not precious || is varnished | is not varnished
the student | fails | does not fail || is unsuccessful | is not unsuccessful || reads | does not read
the blacksmith | is strong | is not strong || is powerful | is not powerful || is married | is not married
the bus | is late | is not late || is delayed | is not delayed || is blue | is not blue
the candle | burns | does not burn || is lit | is not lit || smells | does not smell
the library | is silent | is not silent || is noiseless | is not noiseless || is modern | is not modern
the captain | is brave | is not brave || is courageous | is not courageous || is bearded | is not bearded
the cheese | is rotten | is not rotten || is spoiled | is not spoiled || is French | is not French
the robot | is clever | is not clever || is intelligent | is not intelligent || is metallic | is not metallic
the mine | collapses | does not collapse || caves in | does not cave in || expands | does not expand
the wolf | is hungry | is not hungry || is starving | is not starving || is grey | is not grey
the guard | sleeps | does not sleep || is asleep | is not asleep || marches | does not march
the market | is busy | is not busy || is bustling | is not bustling || is outdoors | is not outdoors
the letterbox | is full | is not full || is crammed | is not crammed || is red | is not red
the runner | is fast | is not fast || is quick | is not quick || is Kenyan | is not Kenyan
the palace | is empty | is not empty || is vacant | is not vacant || is baroque | is not baroque
the parrot | talks | does not talk || speaks | does not speak || flies | does not fly
the stranger | is friendly | is not friendly || is kind | is not kind || is tall | is not tall
the baby | cries | does not cry || weeps | does not weep || crawls | does not crawl
the machine | is noisy | is not noisy || is loud | is not loud || is German | is not German
the wine | is sweet | is not sweet || is sugary | is not sugary || is red | is not red
the hunter | is careful | is not careful || is cautious | is not cautious || is young | is not young
the passport | is expired | is not expired || is out of date | is not out of date || is green | is not green
the gardener | is lazy | is not lazy || is idle | is not idle || is bald | is not bald
the storm | ends | does not end || is over | is not over || grows | does not grow
the rope | is loose | is not loose || is slack | is not slack || is long | is not long
the beggar | is grateful | is not grateful || is thankful | is not thankful || is blind | is not blind
the festival | is cancelled | is not cancelled || is called off | is not called off || is annual | is not annual
the cliff | is steep | is not steep || is sheer | is not sheer || is white | is not white
the widower | remarries | does not remarry || marries again | does not marry again || travels | does not travel
the keeper | is afraid | is not afraid || is scared | is not scared || is Spanish | is not Spanish
the pond | is shallow | is not shallow || lacks depth | does not lack depth || is muddy | is not muddy
the detective | is suspicious | is not suspicious || is distrustful | is not distrustful || is famous | is not famous
the bread | is stale | is not stale || has gone off | has not gone off || is brown | is not brown
the skater | falls | does not fall || slips over | does not slip over || smiles | does not smile
the uncle | is generous | is not generous || is giving | is not giving || is fat | is not fat
the potter | is broke | is not broke || is penniless | is not penniless || is quick | is not quick
the bomb | explodes | does not explode || goes off | does not go off || ticks | does not tick
the plant | is dead | is not dead || is lifeless | is not lifeless || is tropical | is not tropical
the barn | is wide | is not wide || is broad | is not broad || is abandoned | is not abandoned
the politician | is corrupt | is not corrupt || is crooked | is not crooked || is eloquent | is not eloquent
the cargo | is missing | is not missing || is gone | is not gone || is heavy | is not heavy
the skirt | is tight | is not tight || is snug | is not snug || is yellow | is not yellow
the bishop | is angry | is not angry || is irate | is not irate || is retired | is not retired
the traveler | arrives | does not arrive || gets there | does not get there || sings | does not sing
the jungle | is dangerous | is not dangerous || is hazardous | is not hazardous || is humid | is not humid
the spy | is caught | is not caught || is captured | is not captured || is foreign | is not foreign
the mirror | is broken | is not broken || is smashed | is not smashed || is antique | is not antique
the stone | is smooth | is not smooth || is polished | is not polished || is grey | is not grey
the flood | is severe | is not severe || is serious | is not serious || is recent | is not recent
the secretary | is punctual | is not punctual || is on time | is not on time || is nervous | is not nervous
the goat | escapes | does not escape || gets away | does not get away || eats | does not eat
the medal | is real | is not real || is genuine | is not genuine || is round | is not round
the patient | recovers | does not recover || gets better | does not get better || talks | does not talk
the roof | leaks | does not leak || lets in water | does not let in water || shines | does not shine
the jury | is fair | is not fair || is impartial | is not impartial || is small | is not small
the hero | dies | does not die || perishes | does not perish || laughs | does not laugh
"""
FRESH = parse(RAW); _old = {t['subj'] for t in TEST} | {t['subj'] for t in DEV}
assert len(FRESH) == 60 and len({t['subj'] for t in FRESH}) == 60, len(FRESH)
assert not {t['subj'] for t in FRESH} & _old, {t['subj'] for t in FRESH} & _old
