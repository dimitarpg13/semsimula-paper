# Refinement invariance: the tests and their outcomes

## What this note is

This note is a short, readable summary of one part of the
[Depth Ladder and Matched Baseline Protocol](Depth_Ladder_and_Matched_Baseline_Protocol.md).
That protocol is a running log of pre-registered tests, and it is long.
This note collects in one place what its §5.19 ("The refinement decomposition: two failure modes, two tests") asks, how the tests were registered, and what they found.
It also collects the scope caveat on training budget and the three budget tests RT1, RT2 and RT3, which are scored in other entries of the log.
Every number, verdict and date below is copied from the protocol.
For the full record, follow the anchors in the last section.
Background on the question itself is in [Composing single-layer inferences: flow or maps?](Composing_Single_Layer_Inferences_Flow_or_Maps.md).
Written 2026-11-17.

## The question

The models in this programme compute each layer as one step of a damped Langevin integrator.
A model with $L = 2$ layers integrates over a fixed total time $T$ in $N = 2$ steps of size $\Delta t = T/N$, with $\Delta t = 4$ as trained.

**Refinement invariance** asks: if the same total time $T$ is cut into more, smaller steps ($N \gt 2$), does the trained stack give the same result?
If it does, the layer stack is a coarse sample of a continuous flow, and depth becomes a knob that can be turned at inference.
If it does not, the stack is a set of specialised maps, each tied to its trained step size.
The book calls this requirement property (R).

**Terms used throughout.**

- **Perplexity (PPL)** is the exponential of the mean next-token loss. Lower is better.
- **Nats** are units of log-perplexity (natural logarithm).
- **Refinement penalty.** The cost of running $N$ steps instead of the trained 2, at the same $T$, in nats. Zero means no cost. With $P_N$ the perplexity at $N$ steps:

  $$\text{pen}(N) = \ln \frac{P_N}{P_2}$$

- **Gate 3** is the standard refinement test (Cell 6b-7). It runs the trained model at $N$ steps over the fixed $T$, recomputing everything (the context, the potential's local quadratic, the LayerNorm projection) at every substep. **Gate 3 at $N = 3$** is 1.5 times the trained number of steps.
- **Gate 1** resets the incoming velocity (a test of reliance on momentum). **Gate 2** runs more steps at the trained $\Delta t$ (a test of extension in depth).
- **Per-layer flow.** A different refinement, defined in §5.19. Each trained layer's share of $T$ is cut into substeps. The context $\xi$ and the local quadratic (stiffness) of the learned potential $V_\theta$ are taken once, at the layer's entry. The remaining force is applied as kicks at every substep, evaluated at the current state. The LayerNorm projection runs once, at the end of the layer's share. At $N = 2$ this is the trained model, bit for bit.
- **Kick share.** The fraction of a layer step carried by the explicit kick, measured by replaying the step with the kick set to zero. A large share means a converging per-layer flow is not converging by construction.

**The arms named below.**
F3.1 is a conservative-only arm with a split integrator step.
SR2 is F3.1's configuration with the exact damped-mode flow on the stiff subspace in place of the split step.
PM1 and PMX are conservative arms with potential wells (a memory); G2, G3 and G3′ are non-conservative (Fock) arms with registers.
Every ladder arm trains on 532.5M tokens unless marked 2B.

## Two failure modes (§5.19, pre-registered 2026-10-06)

§5.19 splits the refinement failures seen so far into two mechanisms.

| failure mode | where | mechanism |
| --- | --- | --- |
| register reset (H-RR) | Fock arms G2 (+1,274%) and G3 (+1,342%) | register content emptied and rewritten each layer; refinement inserts rewrite events never trained |
| stiff-mode discretisation | F3.1 (+143%), no registers | V_θ carries the force with stiff modes past π (θ median 4.32); the split step's θ/sin θ phase error remains |

Here $\theta = \omega \Delta t$ is the rotation of a stiff mode within one layer step.
As refinement penalties, §5.19 lists F3.1 at 0.89 nats, G3′ at 0.77 and G2 at 2.74.
The section registers one test per mechanism.

- **Test 1 (register reset), by replication:** a second seed of G3′ and of G2 (predictions S1 to S4). The protocol records no score for S1 to S4 as of this note's date.
- **Test 2 (discretisation), by the exact flow:** SR2 on F3.1's configuration (predictions E1 and E2).

## Test 2 scored: SR2 on F3.1 (2026-10-09)

| id | prediction | called | result |
| --- | --- | ---: | --- |
| E1 | SR2 cuts F3.1's Gate 3 penalty at N = 3 by at least half (0.89 → 0.45 nats or below) | 55% | HIT: 0.405 nats |
| E2 | SR2 settles within +3% of F3.1 (59.5 or below) | 60% | HIT: 58.32 (+1.0%) |

Gate 3 on the best checkpoints, penalty in nats:

| N | SR2 | F3.1 | cut |
| ---: | ---: | ---: | ---: |
| 1 | 1.944 | 2.565 | 24% |
| 3 | 0.405 | 0.889 | 54% |
| 4 | 0.986 | 1.555 | 37% |
| 6 | 1.725 | 2.083 | 17% |
| 8 | 2.110 | 2.275 | 7% |

The exact flow removes the stiff-mode share of the penalty near the trained step.
It does not make the model refinement-invariant: Gate 3 still grows away from $N = 2$, and at fine steps both models share a penalty of about 2.1 to 2.3 nats.
Momentum reliance also nearly vanished (Gate 1: +3.9% for SR2 against +34.3% for F3.1).
The protocol's verdict: SR2-F3.1 is the most refinement-ready conservative model so far, but it does not satisfy (R).

## The fine-step residual localized (diagnostic, 2026-10-09)

A diagnostic, not pre-registered, on 4 of Gate 3's 12 batches.
It switched three things to act once per trained layer instead of at every substep: the LayerNorm projection ("LN once"), the quadratic of $V_\theta$ ("freeze") and the context $\xi$ ("ξ frozen").
Penalty in nats:

| arm | SR2 N = 3 | SR2 N = 4 | SR2 N = 8 | F3.1 N = 3 | F3.1 N = 4 | F3.1 N = 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| as trained | 0.421 | 1.029 | 2.219 | 0.914 | 1.613 | 2.400 |
| all three switches | 0.331 | 0.269 | 0.143 | 0.242 | 0.922 | 0.673 |

With all three switches, SR2's penalty shrinks as the steps get finer: 94% of the $N = 8$ penalty is removed.
No single switch does it.
F3.1 improves but does not converge, so the exact flow is a prerequisite.
This defined the per-layer flow, which FLOW-C and FLOW-R then tested.

## FLOW-C: first confirmation (pre-registered and scored 2026-10-09)

All 12 batches of Gate 3, $N$ in {2, 3, 4, 6, 8, 12, 16}.

| id | prediction | called | result |
| --- | --- | ---: | --- |
| C0 | SR2 as trained reproduces Gate 3 within 0.03 nats at N = 3, 4, 8 | 90% | HIT |
| C1 | SR2, per-layer flow: pen(8) at most 0.25 nats | 80% | HIT: 0.127 |
| C2 | SR2, per-layer flow: no rise from N = 4 to 16 (each step at most +0.02), pen(16) ≤ pen(8) | 65% | MISS on the letter: +0.026 from N = 6 to 8 |
| C3 | SR2 kick share at least 0.10 at both layers | 55% | HIT: 0.367 and 0.506 |
| C4 | F3.1, per-layer flow, does not converge: pen(16) above 0.5 or above pen(8) | 65% | HIT on the letter: 0.510 |
| C5 | PM1, per-layer flow: pen(8) at most 1.0 nats | 35% | MISS: 5.039 |

SR2's per-layer flow settled at a limit (0.129 nats at $N = 12$ and 16).
C2 had been written for a penalty falling to zero, so it missed although the flow converged.
The decision rule stood: no (R) claim from FLOW-C.
The author chose to replicate on fresh data under a convergence criterion registered before the run (FLOW-R).

## FLOW-R: the per-layer flow on fresh batches (pre-registered and scored 2026-10-09)

New validation batches (seed 20261009), SR2's best checkpoint.

| N | 2 | 3 | 4 | 6 | 8 | 12 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| per-layer flow, PPL | 55.04 | 73.30 | 67.64 | 59.83 | 61.00 | 61.08 | 61.09 |
| per-layer flow, penalty (nats) | — | 0.286 | 0.206 | 0.084 | 0.103 | 0.104 | 0.104 |

| id | prediction | called | result |
| --- | --- | ---: | --- |
| R1 | abs(pen(16) − pen(12)) ≤ 0.01 and abs(pen(12) − pen(8)) ≤ 0.02 | 80% | HIT: 0.0001 and 0.0012 |
| R2 | pen(16) ≤ 0.25 | 80% | HIT: 0.104 |
| R3 | pen(16) within 0.05 of FLOW-C's 0.129 | 70% | HIT: 0.025 away |
| R4 | as trained still diverges: pen(8) ≥ 1.5 | 90% | HIT: 2.004 |
| R5 | kick share at least 0.10 at both layers | 90% | HIT: 0.369 and 0.505 |

All five hit.
The licensed claim: **SR2's layer step is a coarse sample of a convergent per-layer flow, with a discretisation gap of 0.104 nats on fresh data** (0.129 on Gate 3's batches).
The kick carries 37 to 51% of each step, so the convergence is not by construction.
What it does not claim: the standard refinement (Gate 3) still diverges; the result holds for SR2 only (F3.1 does not settle within $N \le 16$, PM1 diverges); and it is a property of the trained weights at inference, with training unchanged.
An inference mode, `substeps_per_layer`, was then implemented and verified bit-identical at 1 substep per layer.

## The scope caveat and the training-budget tests

**The caveat (the author, 2026-10-11).** Every ladder arm trains on 532.5M tokens (32,500 steps $\times$ 16,384 tokens).
At 76.8M parameters that is about 7 tokens per parameter, a third of the compute-optimal budget of about 1.5B tokens.
So every Gate 1 to 3 reading, FLOW-C and FLOW-R is a statement about models at 0.5B tokens, not about the architectures.
A failure does not show the architecture cannot refine; a pass does not show it stays refinement-ready.
The direction was stated as a hypothesis with mixed evidence: a smoother, better-trained potential could help, while training also stiffens some modes.

**The tests (pre-registered 2026-10-11).**

| id | test | prediction | called |
| --- | --- | --- | ---: |
| RT1 | PMX, Gate 3 at N = 3: step-8,000 probe (0.584 nats) against the best checkpoint | the penalty falls with training | 60% |
| RT2 | PM1 and PM1-cap, Gate 3 at N = 3: step 8,000 against settled | the penalty falls in each | 50% |
| RT3 | SR2 trained 4× longer (130,000 steps, 2.13B tokens); Gate 3 and FLOW-R at the end | Gate 3 at N = 3 at or below 0.30 nats (0.405 at 0.5B); the per-layer flow still converges | 45% |

RT1 and RT2 are confounded with the learning-rate schedule: step 8,000 is mid-way through the constant-rate phase, and the best checkpoint comes after the decay.
RT3 was registered as the real test.

**Results.**

| id | scored | result | verdict |
| --- | --- | --- | --- |
| RT1 | 2026-10-11 | Gate 3 at N = 3 rises from 0.584 to 1.097 nats | MISS |
| RT2 | — | no score recorded in the protocol | open |
| RT3 | 2026-11-15 | Gate 3 at N = 3: 0.772 nats (0.405 at 0.5B); per-layer flow converges by the letter (successive differences 0.006 and 0.005) but to 0.920 nats (0.104 at 0.5B) | MISS |

For RT1 the protocol adds that stiffness rose over the same stretch, and that the reading concerns the standard refinement, not the per-layer flow.
On PMX at 0.5B the per-layer flow converged slowly, to a limit of 0.387 nats at $N = 16$, and its pre-registered convergence criterion missed on the letter (CX3).

**The trend inside the 2B run.** SR2-2B saved weight snapshots along training.
FLOW-R's per-layer flow penalty at $N = 16$ on each:

| step | 6,500 | 16,500 | 32,500 | 48,500 | 65,000 | 84,500 | 97,500 | 113,500 | 130,000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pen(16), nats | 0.008 | 0.079 | 0.486 | 0.619 | 0.691 | 0.797 | 0.807 | 0.879 | 0.922 |

Step 6,500 is the end of warm-up, 84,500 the start of the decay, 130,000 the end.
Over the same run, layer 1's kick share rose from 0.36 to 0.86.
The protocol's summary: the map departs from the flow with training.
The condition of a related call, TL2 (the $N = 16$ penalty larger at 84,500 than at 32,500, both at the peak learning rate), holds on this arm: 0.797 against 0.486.
TL2 itself is scored only when three of its four arms have run.

## What stands and what is open

**Stands.**

- At 0.5B tokens, SR2's layer stack is a coarse sample of a convergent per-layer flow, with a gap of 0.104 nats on fresh data (FLOW-R, all five calls hit).
- The standard refinement (Gate 3) is not invariant in SR2, F3.1, PM1 or PMX. The exact flow halves SR2's penalty at $N = 3$ (E1) but leaves the fine-step limit.
- The training-budget evidence so far goes against "refinement improves with training": RT1 missed, RT3 missed, and SR2-2B's per-layer flow penalty grows from 0.008 to 0.922 nats across its snapshots (converging in $N$ by the letter at the end).

**Open.**

- RT2 has no recorded score.
- Test 1 (the second seeds of G3′ and G2, S1 to S4) has no recorded score.
- TL2 waits for its other arms.
- RT1 remains confounded with the schedule; the 2B snapshots were added to separate training length from schedule phase.

## Where to read the full record

- §5.19, its scope caveat and the RT registration, followed by the SR2 result, the fine-step diagnostic, FLOW-C and FLOW-R: [protocol, `p-sr2`](Depth_Ladder_and_Matched_Baseline_Protocol.md#p-sr2) (the §5.19 heading sits just above this anchor).
- RT1 scored, PMX full run: [protocol, `p-pmx`](Depth_Ladder_and_Matched_Baseline_Protocol.md#p-pmx).
- RT3 set-up and result, SR2-2B: [protocol, `p-sr2-2b-setup`](Depth_Ladder_and_Matched_Baseline_Protocol.md#p-sr2-2b-setup) and [protocol, `p-sr2-2b-result`](Depth_Ladder_and_Matched_Baseline_Protocol.md#p-sr2-2b-result).
- The gate definitions (Gates 0 to 4) and the first MAPS verdict: [Composing single-layer inferences: flow or maps?](Composing_Single_Layer_Inferences_Flow_or_Maps.md), §8.
