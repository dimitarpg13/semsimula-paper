# The code set: first against second order on Python code (MS5 and C5)

**What this note is.** A short, readable summary of the code set of the depth ladder, written 2026-11-17. The code set is the protocol's test MS5: the same conservative model trained on Python code in second-order and first-order form, without wells and with them. This note covers the corpus and its reference model, the calls CODE1 to CODE5, the two arms that have run (SO-CODE and FO-b-CODE), the copy split and its reading, and the two-well pair C5 (PM2-CODE and FO-b-PM2-CODE). Every number, verdict and date is copied from the [protocol](../Depth_Ladder_and_Matched_Baseline_Protocol.md), which remains the full log and the record of truth. All the entries sit in its §5.13 (the FO series). Each heading below names the protocol anchor it summarises, so a reader of a model card can match the two.

**Status in one line.** SO-CODE and FO-b-CODE have run; CODE1 and CODE2 are scored HIT, provisionally (see §4.3). The C5 pair ran on 2026-11-10. Its results and the scores of CODE3, CODE4, CODE5 and MS5 were published on the two models' cards the same day; the protocol has not yet filed them (§6.2).

---

## Contents

1. [Terms used](#1-terms-used)
2. [The corpus and its reference model](#2-the-corpus-and-its-reference-model)
3. [The code set and its calls](#3-the-code-set-and-its-calls)
4. [The two arms without wells](#4-the-two-arms-without-wells)
5. [The copy split](#5-the-copy-split)
6. [C5: the two-well pair](#6-c5-the-two-well-pair)
7. [Scorecard and open points](#7-scorecard-and-open-points)

---

## 1. Terms used

- **Perplexity (PPL)** measures how well a model predicts the next token; lower is better. Differences in loss are given in **nats** (natural-log units). A gap of 0.095 nats is about a 10% rise in perplexity.
- **Settled perplexity.** For the code set, the mean of the evaluations at steps 18,000, 18,500 and 19,000. **Best** is the lowest single evaluation.
- **Second order (SO).** The conservative model carries a velocity from layer to layer, like a particle with inertia. All code arms use SR2, the exact damped flow on the stiff subspace (see [SR_Series_Settling_and_Refinement.md](SR_Series_Settling_and_Refinement.md)).
- **First order (FO-b).** A true overdamped first-order Langevin step, with exact exponential relaxation of the low-rank stiff modes and no velocity. Switch `FIRST_ORDER_OVERDAMPED = True`.
- **Wells.** Poisson-mode wells, here $K = 2$ of them (the PM2-PMX16 setting). "Without wells" means $K = 0$.
- **Paired by step.** All code arms train from step 0 at seed 0 on the same batches, so their evaluations at a given step can be compared directly.
- **Diagnostic cells.** Cell 6b-7 runs four gates on the trained model: Gate 0 checks the patch changes nothing; Gate 1 resets the velocity between layers; Gate 2 adds layers at the trained step size; Gate 3 refines the step at fixed total time. Cell 6b-13 reads $\omega \Delta t$, how far a stiff mode of frequency $\omega$ turns in one layer step; past 2 is the explicit integrator's wall.
- **Settling check (the 2026-10-04 rule).** A straight line is fitted to the evaluations over steps 16,000 to 19,000. As the entries apply it, the run is "settled" when the fitted change is smaller than the residual standard deviation (SD).

## 2. The corpus and its reference model

**The corpus (MS5 amendment of 2026-10-24).** Python from CodeParrot's own file-level split, at pinned revisions. The training pool is the first 450M GPT-2 tokens of `codeparrot/codeparrot-clean-train`; the validation set is the first 2M tokens of `codeparrot/codeparrot-clean-valid`. Before tokenising, each leading run of four spaces becomes one tab. On raw Python 40% of tokens are whitespace-only; after the rewrite whitespace falls to 21.5%, and a file takes 23% fewer tokens. Code perplexities are compared only within this corpus. The budget is the TinyStories one: 16,250 steps on the schedule, plus 3,000 at the floor learning rate, to 19,250 steps. The code set was kept out of book v23 by the author's decision of 2026-10-24.

### 2.1 The matched GPT-2 on code (protocol `p-gpt2-code`)

GPT-2-CODE is the attention baseline matched to the ladder: $d = 384$, $L = 8$, 33,691,776 parameters, WSD schedule with peak 1.2e-3, batch 32 of 512 tokens, 19,250 steps. Run 2026-10-26 (checklist item C3), about 1.6 h on the A100. It also built the shared code cache the ladder arms read.

| reading | value |
| --- | --- |
| settled PPL (evals at 18,000 / 18,500 / 19,000) | 4.96 (4.96, 4.96, 4.95) |
| final PPL | 4.95 |
| step 16,000 / 16,500 | 5.02 / 5.00 |
| largest rise between logged training losses | 0.028 (step 17,400) |
| gradient norm, median / maximum | 0.51 / 1.35 (step 200) |

The extension at the floor was worth about 1%, so every code arm runs straight to 19,250 steps. GPT-2-CODE's 4.96 is the reference for every code perplexity.

## 3. The code set and its calls

(Protocol §5.13, the block "The code set (MS5)" right after `p-gpt2-code`.) **Registered 2026-10-26, after C3 and before any code ladder arm trained.** The calls were written before SO-CODE's training cell started, and the author started training on them without changes, so they are binding. Four arms, each from step 0 at seed 0:

| arm | configuration |
| --- | --- |
| SO-CODE | SR2's Cell 0 plus CORPUS = 'code', LADDER_STEPS = 16_250, EXTRA_FLOOR_STEPS = 3_000 |
| FO-b-CODE | the same plus FIRST_ORDER_OVERDAMPED = True |
| PM2-CODE (C5) | PM2-PMX16's Cell 0 with the same additions (K = 2 wells) |
| FO-b-PM2-CODE (C5) | its first-order twin |

The wells were first fixed at $K = 16$ (2026-10-24). The author amended this to $K = 2$ on 2026-10-26, before any code arm trained: every corpus then has the same pairs at $K = 0$ and $K = 2$, and $K = 2$ is the smallest dose that closed the OpenWebText gap.

| id | prediction | called |
| --- | --- | ---: |
| CODE1 | second order earns a gap without wells on code: FO-b-CODE settles more than 2% above SO-CODE | 65% |
| CODE2 | the gap is larger than on OpenWebText: FO-b-CODE / SO-CODE exceeds +4.9% | 35% |
| CODE3 | first order matches with wells on code too: FO-b-PM2-CODE within ±2% of PM2-CODE, paired by step | 65% |
| CODE4 | the wells help first order more on code too: log(FO-b-CODE / FO-b-PM2-CODE) exceeds log(SO-CODE / PM2-CODE) | 70% |
| CODE5 | the wells help at all on code: PM2-CODE settles more than 1% below SO-CODE | 55% |
| MS5 | hits if CODE1 and CODE3 both hit | 45% |

CODE2 is the prediction of the information-preservation channel. In plain words, MS5 says: on code, second order should beat first order clearly when there are no wells, and the gap should vanish once wells are added.

## 4. The two arms without wells

### 4.1 SO-CODE (protocol `p-so-code`)

Started 2026-10-26, finished 2026-10-27. 19,250 steps in 5.92 h; 76,572,109 parameters. No watchdog trigger; the leak monitor (SCAF) read CLEAN at all four audits.

| reading | SO-CODE |
| --- | --- |
| settled PPL (evals 5.87, 6.02, 6.10) | 6.00 |
| best | 5.87 at step 18,000 |
| against GPT-2-CODE (4.96) | +21.0% |
| gap to GPT-2 at step 2,000 / end of stable phase / decay / floor | +82% / +14.7% / +16.9% / +21.2% |
| settling check | fitted change −0.020 nats against SD 0.015: not settled by the letter |
| Gate 1, velocity reset | +4.0% (5.06 → 5.26) |
| Gate 2, one extra layer | +42% |
| Gate 3 at N = 3 | +89% (0.639 nats) |
| ω·Δt median (layer 0 / layer 1) | 7.55 (9.58 / 4.94); 99.97% past 2 |

For comparison, SR2 is +17.1% behind its GPT-2 on OpenWebText and SO-TS +15.4% on TinyStories. The gap to GPT-2 grows through the decay phase: GPT-2 gains more from it. The settling check missed by a third of the noise, and the fitted change's own standard error (about 0.017 nats) makes it indistinguishable from zero. The protocol decided to apply the rule to the pair, once FO-b-CODE was in.

Later entries quote SO-CODE's settled value with more digits, as 5.99 (for example `p-l4-code-result`) and 5.994 (`p-k2-l8-code-result`). It is the same run.

### 4.2 FO-b-CODE (protocol `p-fo-b-code`)

Finished 2026-10-27. 19,250 steps in 5.65 h, one session; no watchdog trigger; SCAF CLEAN at all four audits.

| reading | FO-b-CODE | SO-CODE |
| --- | --- | --- |
| settled PPL | 6.55 (evals 6.43, 6.57, 6.65) | 6.00 |
| best | 6.43 at step 18,000 | 5.87 |
| against SO-CODE | +9.2% | |
| against GPT-2-CODE | +32.0% | +21.0% |
| Gate 1, velocity reset | +0.00 (no velocity exists) | +4.0% |
| Gate 2, one extra layer | +3.6% | +42% |
| Gate 3 at N = 3 | +32% (0.277 nats) | +89% (0.639) |
| ω·Δt median (layer 0 / layer 1); past 2 | 0.86 (1.17 / 0.002); 22.9% | 7.55; 99.97% |

**Paired by step**, FO-b-CODE is ahead through step 5,000, and SO-CODE leads from step 5,500. The gap peaks at +11.1% at the end of the stable phase and narrows to +9.2% at the floor. Over steps 16,000 to 19,000 the seven paired gaps lie between +9.1% and +9.5%, and the two arms' evaluations correlate at 0.996.

**First order builds a soft potential.** The median $\omega \Delta t$ of 0.86 is about ten times softer than SO-CODE's 7.55. OpenWebText shows the same contrast (FO-b-OWT 0.74, SR2 7.55). The protocol's reading: the softness belongs to the order, not to the corpus.

### 4.3 The settling decision and the provisional scores

FO-b-CODE also failed the settling check (fitted change $-0.023$ nats against SD 0.015). By the rule, the extension decision for both arms was due before CODE1 and CODE2 were scored. The protocol recommends **no extension**: the two arms drift together, and the paired gap is flat to 0.003 nats over the last 3,000 steps. The entry records the **author's decision as pending**; no later entry in the protocol records that decision. The C5 setup of 2026-11-09 states "CODE1 hit".

- **CODE1 (called 65%): HIT,** provisional on no extension (+9.2%).
- **CODE2 (called 35%): HIT,** provisional on no extension.

The first-order gap without wells then orders with the corpora, as the [corpus-statistics note](../Corpus_Statistics_and_the_First_vs_Second_Order_Well_Gap.md) predicts:

| corpus | first-order gap without wells |
| --- | ---: |
| TinyStories (FO-b-TS / SO-TS) | +2.3% |
| OpenWebText (FO-b-OWT / SR2) | +4.9% |
| code (FO-b-CODE / SO-CODE) | +9.2% |

## 5. The copy split

### 5.1 What it is and how it is read

Registered 2026-10-26, before SO-CODE finished. A diagnostic on CPU (`debug/copy_split.py`). The question came from step 2,000, where SO-CODE trailed GPT-2-CODE by 1.82 times in perplexity, against 1.14 times on TinyStories. One reading was that code rewards exact copying of names chosen afresh in each file, which attention does well.

The split compares two models' per-token loss on the same 512 windows of 512 tokens (262,144 targets), spread over the code validation set. Targets are grouped by what the context offers:

- **first occurrence:** the token has not appeared before in the window (nothing to copy);
- **repeat:** the token occurs earlier;
- **induction:** the pair of current token and target occurs earlier.

They are also grouped by token class, by distance to the last occurrence, and by position. The registered rule:

- **copying explains the gap** if repeat and induction targets carry a share of the gap at least 1.5 times their share of the targets, and the induction gap is at least twice the first-occurrence gap;
- **the gap is general** if every group's gap is within a factor 1.5 of the overall gap;
- **otherwise mixed.**

### 5.2 SO-CODE against GPT-2-CODE

Run 2026-10-27 on SO-CODE's best checkpoint (step 18,000). Loss in perplexity on the split's targets: GPT-2-CODE 5.350, SO-CODE 6.442, ratio 1.204. SO-CODE is worse in 99.0% of the windows.

| group | share of targets | gap per token (nats) | share of gap |
| --- | ---: | ---: | ---: |
| first occurrence | 27.7% | +0.189 ± 0.007 | 28.2% |
| repeat (not induction) | 22.3% | +0.254 ± 0.008 | 30.4% |
| induction | 50.0% | +0.154 ± 0.005 | 41.3% |
| indentation, newline, space | 21.9% | +0.042 | 4.9% |
| identifier, repeat (not induction) | 8.7% | +0.363 | 16.9% |
| repeat or induction, last seen 1–16 back | 34.8% | +0.109 | 20.5% |
| … 17–64 back | 25.2% | +0.208 | 28.1% |
| … 65–256 back | 10.8% | +0.342 | 19.9% |
| … 257–512 back | 1.5% | +0.395 | 3.2% |

**Verdict by the registered rule: MIXED.** It is not copying: repeat and induction targets carry 71.7% of the gap against 72.3% of the targets, and the induction gap is 0.82 of the first-occurrence gap. It is not general by the letter either: whitespace, repeated identifiers and far repeats fall outside the factor 1.5.

**What the split shows instead: the gap grows with how far back the earlier occurrence is.** Short-range copying is nearly as good as GPT-2's (+0.109 nats within 16 tokens). Recall from 65 to 512 tokens back costs three to four times as much per token. This is not a position effect; it holds among late positions alone. The context channels do reach that far: the longest trained channel has a horizon of 608 tokens. In layer 1 it puts 22% of its weight on earlier occurrences 65 or more tokens back, spread over about 36 positions. The protocol's summary: "the reach exists; what is scarce is the number and sharpness of long-range lookups". The deficit is long-range, content-addressed recall, not copying as such. By the order agreed on 2026-10-27 (a mixed reading), the register arm FOCK-CODE ran before the PM2 pair.

### 5.3 FO-b-CODE against SO-CODE

Run 2026-10-27 on FO-b-CODE's best checkpoint (step 18,000): 7.024 against 6.442, ratio 1.090. FO-b-CODE is worse in 98.2% of the windows.

| group | share of targets | FO-b − SO (nats) | share of gap | SO − GPT-2, for comparison |
| --- | ---: | ---: | ---: | ---: |
| all | 100% | +0.086 | 100% | +0.186 |
| identifier-like | 49.3% | +0.152 | 86.6% | +0.253 |
| indentation, newline, space | 21.9% | +0.008 | 2.0% | +0.042 |
| repeat or induction, 1–16 / 17–64 / 65–256 / 257–512 back | 34.8 / 25.2 / 10.8 / 1.5% | +0.043 / +0.102 / +0.178 / +0.197 | 17 / 30 / 22 / 4% | +0.109 / +0.208 / +0.342 / +0.395 |

**The first-order gap sits in identifiers and grows with distance**, the same shape as SO-CODE's gap to GPT-2, at about half the size. It is **not a failure to find the earlier occurrence**: FO-b-CODE's long channel in layer 1 puts at least as much weight on far earlier occurrences as SO-CODE's (0.245 against 0.223).

### 5.4 The reading

Both models address the context equally well; the first-order model does less with what it retrieves. A long-range channel delivers the earlier identifier diluted, about a quarter of its weight spread over some 30 positions. Decoding that into one of many identifiers needs a sharply curved readout. A first-order layer contracts a stiff mode and erases distinctions along it, so the trained first-order model keeps a soft potential. A second-order layer turns a stiff mode through a phase and keeps its amplitude (the book's "within-step implicit memory", §26.9 item (ii)). The cost of softness then grows with how many distinctions the readout must keep: TinyStories +2.3%, OpenWebText +4.9%, code +9.2%.

The protocol labels this **an interpretation, not a measured mechanism** (one seed per model, $L = 2$). Its proposed test is the same split on the C5 pair: what remains of the gap on code should stay in identifiers at long range.

## 6. C5: the two-well pair

### 6.1 Setup (protocol `p-c5-code-setup`)

Set up 2026-11-09 (author: "let us wrap the CODE testing with this last pair"). The pair waited behind the depth runs. On OpenWebText and TinyStories two wells closed the first-order gap. On code the gap without wells is the largest measured (+9.2%), and book v6.2 limits its "with wells, first order matches or leads" to the other two corpora.

| arm | Cell 0 | tag fragment |
| --- | --- | --- |
| PM2-CODE | SO-CODE's nine changes plus POISSON_MODES=2, POISSON_WELLS_EXACT=True (11 changes) | …xilive_pm2_pmx16_sr2_st16250_L2probe… |
| FO-b-PM2-CODE | the same plus FIRST_ORDER_OVERDAMPED=True (12 changes) | …xilive_pm2_pmx16_sr2_fob_st16250_L2probe… |

**Checked on CPU (2026-11-09).** Cell 0 gives both tags, and the same code reproduces SO-CODE's and FO-b-CODE's tags exactly. Each arm builds with 76,572,885 trainable parameters, SO-CODE plus 776 for the two wells, and a training step is finite. At initialisation only the wells' depth `pm_depth` receives a gradient, as in PM2-PMX16 and PM2-TS. The two twins already differ at initialisation (largest logit difference 4.5e-3).

**Run plan.** PM2-CODE first, then FO-b-PM2-CODE, each fresh from step 0 at seed 0, so both pair by step with SO-CODE and FO-b-CODE. 19,250 steps, about 6 to 7 h each on one A100 80 GB.

**Calls.** CODE3, CODE4 and CODE5 are binding since 2026-10-26 and unchanged (table in §3). Since CODE1 hit, the MS5 call is decided by CODE3. A further descriptive call, PMCS1, was proposed with FOCK-CODE and is **not confirmed by the author**:

| id | prediction | called |
| --- | --- | ---: |
| PMCS1 | in the copy split, the per-token gap to GPT-2-CODE shrinks by a larger fraction on first occurrences than on induction targets from SO-CODE to PM2-CODE | 60% |

The reasoning behind PMCS1: trained wells have fixed directions and no per-file content, so they should help with what every file shares, not with copying.

**After the runs** the plan lists Cells 7, 6b-7, 6b-15 and 6b-13, the CPU probes, the copy split and FLOW-R. FLOW-R's `code` kind needs an extension (freezing the wells' quadratic per layer) before it can run on these two.

### 6.2 Results (from the published model cards, 2026-11-10)

The protocol has not filed these results; they are quoted from the two models' Hugging Face cards, generated from `semsimula/hf_model_cards/_ladder/generations.json`.

| | PM2-CODE | FO-b-PM2-CODE |
| --- | ---: | ---: |
| settled perplexity | 5.71 | 5.62 |
| best | 5.60 | 5.52 |
| training | 19,250 steps, 6.55 h, 1.23 s per step | 19,250 steps, 6.34 h, 1.19 s per step |
| leak monitor | CLEAN at all 4 audits | CLEAN at all 4 audits |

- **CODE3** (FO-b-PM2-CODE within ±2% of PM2-CODE, paired by step), called 65%: **HIT**, a difference of 1.6% on the settled values, the first-order model ahead.
- **CODE4** (the wells help first order more), called 70%: **HIT**, 0.153 against 0.049 nats.
- **CODE5** (PM2-CODE settles more than 1% below SO-CODE), called 55%: **HIT**, 4.7% below.
- **MS5** (CODE1 and CODE3 both hit), called 45%: **HIT**. Scored on the letter.
- With two wells, first order turns its largest gap without wells (9.2%) into a 1.6% lead. On the cards' CPU readings the wells are positional: a token's position explains $R^2 \ge 0.999$ of their force in both models.

## 7. Scorecard and open points

| call | called | status |
| --- | ---: | --- |
| CODE1 | 65% | HIT (+9.2%), provisional on no extension |
| CODE2 | 35% | HIT, provisional on no extension |
| CODE3 | 65% | HIT (first order 1.6% ahead), on the cards; not filed in the protocol |
| CODE4 | 70% | HIT (0.153 against 0.049 nats), on the cards |
| CODE5 | 55% | HIT (4.7% below SO-CODE), on the cards |
| MS5 | 45% | HIT, on the cards |
| PMCS1 | 60% | proposed, not confirmed by the author; not scored |

**Open, or unclear in the log.**

- The author's decision on extending SO-CODE and FO-b-CODE is recorded as pending in `p-fo-b-code` and is not recorded later. The C5 setup nonetheless treats CODE1 as hit.
- The C5 pair's results are on its cards but not yet filed in the protocol.
- The reading in §5.4 is an interpretation with one seed per model.

**Other code arms, outside MS5.** They grew out of the copy split's finding. The register path is FOCK-CODE (protocol `p-fock-code`). A sixth context channel is XI6-CODE, whose wells died; see [Code_Dead_Wells_and_Screens.md](Code_Dead_Wells_and_Screens.md). Depth is covered in [Code_Depth_Runs.md](Code_Depth_Runs.md).
