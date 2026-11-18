# Refinement and router probes on the code models

**What this note is.** A short, readable summary of six entries of the depth-ladder protocol, written 2026-11-17. They are CPU readings and one setup on the Python-code models: FLOW-R on SO-CODE and L4-CODE ([protocol, `p-flowr-code`](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-flowr-code)); FLOW-R on L8-CODE (a paragraph inside [protocol, `p-l8-code-result`](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-l8-code-result)); the setup of K2-CODE ([protocol, `p-k2-code-setup`](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-k2-code-setup)); the TKF-A registration and result ([protocol, `p-tkf-attribution`](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-tkf-attribution) and [protocol, `p-tkf-attribution-result`](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-tkf-attribution-result)); and RO-COST ([protocol, `p-router-off-cost`](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md#p-router-off-cost)). Every number, verdict and date is copied from the [protocol](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md), which keeps the full log and remains the record of truth. The training runs these probes read are summarised in [Code_Depth_Runs.md](Code_Depth_Runs.md) (L4-CODE, L8-CODE and the two-well eight-layer arms, including the setup of K2-L8-CODE) and in [SR_Series_Settling_and_Refinement.md](SR_Series_Settling_and_Refinement.md) (SR4a-CODE).

**Status in one line.** The per-layer flow settles on every code model, 0.31 to 0.37 nats from the trained step at two and four layers and 0.035 at eight; re-selecting the pair potential's sources carries none of the refinement penalty (TF-1, TF-2, TF-4 hit; TF-3 misses); dropping the router term at evaluation moves every code model's loss by at most 0.0013 nats; K2-CODE is set up, with calls proposed and not confirmed, and no run is recorded in these entries.

---

## Contents

1. [Terms used](#terms)
2. [FLOW-R on SO-CODE and L4-CODE](#flowr-code)
3. [FLOW-R on L8-CODE](#flowr-l8)
4. [K2-CODE: setup and proposed calls](#k2-code)
5. [TKF-A: registration](#tkf-a-setup)
6. [TKF-A: result](#tkf-a-result)
7. [RO-COST: the router term at evaluation](#ro-cost)
8. [What stands and what is open](#open)

---

<a name="terms"></a>
## 1. Terms used

- **The code models.** All are trained on Python code. **SO-CODE** is the second-order conservative model with $L = 2$ layers (see [Code_Set_First_vs_Second_Order.md](Code_Set_First_vs_Second_Order.md)). **L4-CODE** and **L8-CODE** keep its total integration time $T = 8$ with 4 and 8 layers, so each layer step is $\Delta t = T / L$. **K2-L8-R6-CODE** has 8 layers and 2 wells of rank 6, at a fixed step. **K2-L8-R6-VS-CODE** is the same model trained at step counts $N$ drawn from 8, 12 and 16 (variable step). SR4a-CODE is SO-CODE trained at $N$ drawn from 2, 3 and 4.
- **Refinement (Gate 3 of Cell 6b-7).** Run the trained model with $N$ steps of size $T / N$ in place of its trained $L$ steps. The **penalty** is the log of the ratio of the perplexity (PPL) at $N$ steps to the perplexity at the trained $L$, in nats. A penalty near zero means the model refines: it behaves like a sample of a continuous flow. 0.01 nats is about 1% in perplexity.
- **Whole stack and per-layer flow.** The **whole-stack** test refines all layers together; this is Gate 3. **FLOW-R** (`debug/refinement_flow_confirmation.py`) instead splits each trained layer into $k = N / L$ substeps, holding per layer what the layer computes once (one LayerNorm per layer, the quadratic part and the friction field $\xi$). If the penalty stops changing as $k$ grows, the per-layer flow has settled, and its value is the **limit**: how far the trained step sits from its own continuous flow.
- **FLOW-R's convergence criterion.** The penalty at $k = 8$ and $k = 6$ differ by at most 0.01, and at $k = 6$ and $k = 4$ by at most 0.02. At $L = 2$ these are the registered $N = 16, 12, 8$.
- **Kick share.** A per-layer FLOW-R reading of how much of a trained step is kick rather than exact flow (position, median). The protocol uses it without defining it in these entries.
- **Wells.** $V_\theta$ is one bank of Gaussian wells shared by all layers. SO-CODE has 8 wells (tag `dcvt5x8`); "K2" means 2 wells (`V_THETA_WELLS_PER_HEAD = 2`, tag `dcvt5x2`).
- **The pair potential and its router term.** $V_\phi$ sums a pair potential over each token's top-k sources ($k = 16$), chosen by a score head at every force evaluation. The choice is piecewise constant in the state, so it is the one part of the force that is not the gradient of one smooth potential. The **router term** is the straight-through part of the force: the gate's gradient through the score head. It is the measured exception to per-token conservativity, the property that the force is the gradient of an energy for each token.

---

<a name="flowr-code"></a>
## 2. FLOW-R on SO-CODE and L4-CODE (protocol `p-flowr-code`)

**2026-11-01; descriptive, CPU.** The per-layer flow converges on SO-CODE and on L4-CODE. But its limits are 0.37 and 0.31 nats from the trained step, about 3.5 times as far as SO-TS on TinyStories with the same architecture, budget and schedule.

- **How it was run.** `debug/refinement_flow_confirmation.py <out> <folder> code seed=20261009 name=<arm>`. The new kind `code` works at any $L$: the model is built from the run's tag through `copy_split.tag_repl`, on the code validation tokens. SO-CODE's best checkpoint and L4-CODE's step-18,000 checkpoint (by `ckpt=`) were read on $12 \times 4 \times 512$ tokens at FLOW-R's batch seed. Peak memory 30.6 GB and 52.3 GB; 29 and 65 min on 8 threads. Outputs: `refinement_flow_confirmation_{so-code,l4-code}_seed20261009{_output.txt,.json}` in each arm's results folder.
- **Every check is exact on both** (maximum logit change 0.0): the standard refinement against Cell 6b-7's `_fom_stack` at $N = L$ and $L + 1$; the per-layer flow at $N = L$ against the trained model, and at $N = 4L$ against the model's own `substeps_per_layer = 4`; and the replay of every layer.

Penalty in nats, by $k$ substeps per trained layer ($N$ steps of $8 / N$ over the same $T = 8$):

| model | k = 1.5 | 2 | 3 | 4 | 6 | 8 | whole stack at k = 4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SO-CODE (L = 2, Δt = 4; trained ppl 5.79 on these batches) | 0.970 | 0.526 | 0.300 | 0.384 | 0.376 | 0.369 | 2.858 |
| L4-CODE (L = 4, Δt = 2; 5.29) | 0.429 | 0.520 | 0.282 | 0.320 | 0.309 | 0.306 | 2.960 |
| SO-TS (SR2's architecture, TinyStories, same schedule) | 0.255 | 0.151 | 0.086 | 0.100 | 0.098 | 0.097 | 1.99 |
| SR2 (OpenWebText, 32,500 steps) | 0.286 | 0.206 | 0.084 | 0.103 | 0.104 | 0.104 | 2.00 |

The whole-stack column is FLOW-R's own standard refinement on these batches. Cell 6b-7's batches give SR2 2.11 and L4-CODE 2.87 at $N = 16$.

- **Both converge** by FLOW-R's criterion read in substeps per layer: SO-CODE 0.007 and 0.008, L4-CODE 0.002 and 0.011. Read at $N = 8, 12, 16$ instead, L4-CODE would fail (0.038), because at $L = 4$ those are $k = 2, 3, 4$, before its flow has settled.
- **The corpus moves the limit; this was new.** SO-CODE has SR2's architecture and SO-TS's budget and schedule (16,250 steps plus 3,000 at the floor rate). Its limit is 0.37 nats against SO-TS's 0.097 and SR2's 0.104. The earlier reading after SO-TS ("refinement here is a property of SR2's exact damped flow, not of the corpus") holds for the two prose corpora, not for code. On code the trained step sits as far from its continuous flow as the well-register models on OpenWebText (PMX 0.39, PMX-K16 0.33). One seed per arm.
- **Depth moves the limit little** (read at the time; revised after L8-CODE, section 3). L4-CODE halves the step and its stiffest modes are softer (median $\omega \Delta t$ 2.20 against 7.55, where $\omega \Delta t$ is how far a stiff mode turns in one layer step). Yet its limit is only 0.06 nats closer, and its whole-stack penalty is the same (2.96 against 2.86). Under the exact damped flow, the stiffness per step does not by itself set how far the trained step is from the flow, as Proposition `prop:sr-damped-flow` says for the linear part.
- **Kick share** (position, median by layer): SO-CODE 0.41 and 0.78; L4-CODE 0.29, 0.21, 0.41 and 0.38 (SR2 0.37 and 0.51; SO-TS 0.39 and 0.65). The deeper model's step is more exact flow and less kick, with nearly the same limit. SO-TS has a large kick share and a near limit. So the kick share does not explain the limit here.
- **The penalty is not monotone at small $k$**, as on every second-order model. Here it peaks at $k = 1.5$ or 2, then settles.
- **For the book (2026-11-01).** Remark `rem:refinement-invariance` and §8.9 (`ssec:settling-refinement`) now separate the two tests. The whole-stack refinement fails on every model. The per-layer flow settles for every second-order model, to limits from 0.10 nats (SR2 on prose) to 0.40 (the well-register models on OpenWebText; code 0.31 to 0.37). Against FLOW-R's full criterion, over every FLOW-R output in the results folders, it converges for SR2, SO-TS, PMX-K16, SO-CODE and L4-CODE. It misses narrowly for PMX (0.029 between $N = 12$ and 8), PM2-PMX16 (0.056), SR3-crit (0.028) and the learned damping $\gamma(h)$ (0.012 between $N = 16$ and 12). It does not settle for any first-order (FO-b) model. The aliasing near $\omega \Delta t = \pi$ amplifies the per-step operations; it is not a separate cause.
- **Next on this axis (as written then):** SR4a on code, approved by the author on 2026-11-01, to ask whether training at several step counts brings the trained step onto the flow. Its result is in [SR_Series_Settling_and_Refinement.md](SR_Series_Settling_and_Refinement.md).

---

<a name="flowr-l8"></a>
## 3. FLOW-R on L8-CODE (protocol, inside `p-l8-code-result`)

**2026-11-03, CPU.** Run as `refinement_flow_confirmation.py … code seed=20261009 micro=1`. The new option `micro=1` computes Gate 3 in forwards of one sequence. It was validated on SO-CODE against its filed result (maximum loss change 4.5e-7). Peak memory 26.5 GB; 1.8 h. All checks exact. One seed; the step-18,000 checkpoint.

| model | k = 1.5 | 2 | 3 | 4 | 6 | 8 | whole stack at k = 4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L8-CODE (L = 8) | 0.186 | 0.029 | 0.034 | 0.035 | 0.035 | 0.035 | 2.695 |
| L4-CODE (L = 4) | 0.429 | 0.520 | 0.282 | 0.320 | 0.309 | 0.306 | 2.960 |
| SO-CODE (L = 2) | 0.970 | 0.526 | 0.300 | 0.384 | 0.376 | 0.369 | 2.858 |

- **Kick share by layer** (position, median): 0.15, 0.14, 0.08, 0.13, 0.15, 0.20, 0.26, 0.39.
- **At eight layers the per-layer flow settles 0.035 nats from the trained step.** That is ten times closer than at two or four layers, and closer than SR2 on prose (0.10).
- **This revises the reading of section 2** ("depth moves the limit little"). The first doubling of depth moved the limit 0.06 nats; the second moved it 0.27.
- **The whole stack still fails** (2.695 nats at $k = 4$).

---

<a name="k2-code"></a>
## 4. K2-CODE: setup and proposed calls (protocol `p-k2-code-setup`)

**Set up 2026-11-01** (the author: after L8-CODE, before SR4a-CODE). **The calls are proposed, not confirmed by the author.** K2-CODE is SO-CODE with two $V_\theta$ wells instead of eight. There is one joint bank shared by both layers, so one setting changes both. The question is how much perplexity the six extra wells buy.

The eight-layer two-well arm, K2-L8-CODE, has its own setup and result in [Code_Depth_Runs.md](Code_Depth_Runs.md); it was meant to be read against K2-CODE at $L = 2$. **These entries record no K2-CODE run.** It was queued after SR4a-CODE by the author on 2026-11-01, and on 2026-11-03 the protocol still listed "K2-CODE at L = 2 for the depth comparison of what six wells buy" as still to run.

- **Cell 0.** SO-CODE's nine changes (`REVERSE_CHANNEL = False`, `VPHI_GRAD_PATH = 'live'`, `XI_GRAD_PATH = 'live'`, `LOWRANK_DAMPED_FLOW = True`, `VPHI_ANALYTIC_FORCE = True`, `LAYER_CHECKPOINT = False`, `CORPUS = 'code'`, `LADDER_STEPS = 16_250`, `EXTRA_FLOOR_STEPS = 3_000`), plus `V_THETA_WELLS_PER_HEAD = 2` and `SNAPSHOT_STEPS = [1_000, 1_500, 2_000, 2_500, 3_000]`: 11 changes. The early snapshots are proposed because two wells leave no spare if one dies early; dead wells appeared by step 1,000 in XI6-CODE-WR. They are not in the tag.
- **Tag** (from executing the working copy's Cell 0, which reproduces SO-CODE's tag exactly): `xi5long_topk16_dt32da16_mh4_aniso_dcvt5x2_vtjoint_cgqk_norc_vplive_xilive_sr2_st16250_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn`. It is SO-CODE's tag with `dcvt5x2`. The banner reads "1 joint bank x 2 wells = 2 total attractors".
- **Build at initialisation** (Cells 0 to 5b, the same seed for both):

| count | K2-CODE | SO-CODE |
| --- | ---: | ---: |
| trainable parameters | 50,004,679 | 76,572,109 |
| difference | −26,567,430 | |
| V_θ parameters | 8,859,650 (each well 4,427,905) | 35,427,080 |
| non-embedding parameters | 11,407,303 | 37,974,733 |

  The training loss and backward pass are finite on $2 \times 64$ tokens, and all 9 $V_\theta$ tensors receive a gradient. K2-CODE's non-embedding count is below GPT-2-CODE's 14,196,480. Wells and a third of the parameters change together, so the arm measures the two at once.
- **Probe support.** `debug/copy_split.py` `tag_repl` now reads the wells from `dcvt<n>x<K>` and the SR4a counts from `vs<counts>`. So the CPU probes rebuild K2-CODE and SR4a-CODE from their folder names (checked: the K2 folder adds exactly the wells line).
- **Expected cost:** at most SO-CODE's 1.1 s per step ($V_\theta$ is a quarter of its size), about 6 h for 19,250 steps, one session. Check the step-50 loss and speed against SO-CODE's.
- **After the run:** Cells 8, 7, 6b-7 and 6b-13; the Drive folder with the snapshots and best checkpoint; on the CPU, `vtheta_well_probe.py` on the snapshots and the best checkpoint (nearest-well ratio, occupancy, force share per layer), the removal cost of each layer's wells (`keep=-1 keep_layers=0` and `1`), and the paired evaluations against SO-CODE.

**The evidence the calls rest on.** Removal on SO-CODE's trained weights, 16 code windows (`results/dead_wells/well_probe_so_code_L{0,1}_keep2.txt`). Keeping only two wells costs **0.0%** in layer 1 (perplexity 7.45 against 7.45) and **+1,983%** in layer 0 (155.2). In L4-CODE the deep layers run on two wells and layer 0 on all eight. Those are removals without retraining. K2-CODE trains with two, and layer 0 must learn to work with them.

**Calls, proposed 2026-11-01.** "Settled" is the mean of the evaluations at steps 18,000, 18,500 and 19,000 (exp-mean-log). SO-CODE settled at 5.99. The point prediction is +4%.

| call | prediction | probability |
| --- | --- | ---: |
| K2a | the six wells are worth something: K2-CODE settles more than 2% behind SO-CODE (above 6.11) | 70% |
| K2b | but not much: within +10% of SO-CODE (at most 6.59, level with FO-b-CODE's 6.55 or better) | 75% |
| K2c | no dead well: at the best checkpoint, in both layers, the nearest-well ratio median is at most 2 and the weaker well carries at least 10% of the wells' force | 60% |
| K2d | layer 0 still needs its wells: removing layer 0's two wells costs at least +100% | 85% |

---

<a name="tkf-a-setup"></a>
## 5. TKF-A: registration (protocol `p-tkf-attribution`)

**Registered 2026-11-08 before the run** (the author's go-ahead the same day; CPU, no training). The question: how much of the whole-stack refinement penalty is $V_\phi$ re-selecting its sources? When a trained layer is refined to $N \gt L$, every substep re-selects each token's top-k sources. TKF-A measures how much of the penalty that re-selection carries. It reads two models at their step-18,000 checkpoints: K2-L8-R6-CODE (fixed step; Cell 6b-7 penalty 0.251, 0.499, 1.279 and 1.726 at $N = 12, 16, 24, 32$) and K2-L8-R6-VS-CODE (trained at $N$ of 8, 12 and 16).

**The harness.** `debug/topk_freeze_attribution.py <out> <folder> ckpt=<file> name=<arm>` runs Gate 3 under the `hold` walk at $T = 8$ and $N = 8, 12, 16, 24, 32$, on Cell 6b-7's 12 batches of $4 \times 512$ code tokens. It runs five arms through `refinement_ln_linearisation_split.split_stack`, which gains three switches (`topk_freeze`, `router_off`, `topk_churn`). They are off by default, and FLOW-R's checks on K2-L8-R6-VS-CODE are unchanged with them.

| arm | what changes | at N = 8 |
| --- | --- | --- |
| as trained | nothing (Cell 6b-7's Gate 3) | the trained model |
| top-k frozen | the substeps of one trained layer reuse the selection made at its first substep; the sources' positions, V_φ and the soft weights stay live, and the gate is rebuilt from the frozen indices as `_sparse_topk_indices` builds it | identical, checked exactly |
| top-k frozen, router off | the same, and the force drops the router term: at fixed context and selection it is an exact gradient | a different model, measured |
| per-layer flow | FLOW-R's arm (one LayerNorm per layer; quadratic and ξ frozen per layer; top-k live), on these batches | identical, checked exactly |
| all frozen | per-layer flow plus top-k frozen | identical, checked exactly |

Two more readings, at $N = 8$ on the first two sequences: the router term's relative size per token at every force evaluation point; and the **churn** at $N = 16$ and 32 (as trained), the fraction of a query's 16 sources also selected at its layer's first substep.

**Checks, run on K2-L8-R6-VS-CODE before this registration; all pass.**

- With the switches off, the harness equals Cell 6b-7's `_fom_stack` at $N = 8$ and 9 (maximum logit change 0).
- Top-k frozen, per-layer flow and all frozen equal the trained model at $N = 8$ (0).
- At $N = 32$ the frozen arms make 8 fresh selections and reuse them 24 times.
- The rebuilt gate equals the model's own (0).
- The pair force agrees between the autograd path (the harness's) and the analytic path, router on and off, in float64 to $3 \times 10^{-14}$ relative.
- Seen before registration: router off moves the trained model's logits by up to 1.89 at $N = 8$ on the checks' $2 \times 256$ tokens.

**Calls.** The penalty is the log of the arm's perplexity at $N$ over its perplexity at $N = 8$, on these batches. "Removes" means 1 minus the arm's penalty over the as-trained penalty at the same $N$.

| call | prediction | p |
| --- | --- | --- |
| TF-1 | on K2-L8-R6-CODE, top-k frozen removes under a quarter of the as-trained penalty at N = 16 | 65% |
| TF-2 | on K2-L8-R6-VS-CODE, top-k frozen removes under half of the as-trained penalty at N = 32 | 60% |
| TF-3 | router off raises the loss at N = 8 by more than 0.05 nats on both models | 55% |
| TF-4 | inside the per-layer flow, freezing top-k changes the penalty by at most 0.02 nats at every N ≥ 16, on both models | 70% |

**The reasoning.** The per-layer flow already re-selects at every substep and still settles within 0.034 (VS) and 0.059 (R6) of the trained step. So re-selection should cost little once LayerNorm and $\xi$ are held per layer (TF-4), and most of R6's whole-stack penalty should belong to those (TF-1, TF-2). TF-3 rests on the logit shift seen in the checks.

**What follows, as registered.** If top-k frozen removes a large share (TF-1 misses), Phase 2 trains a fixed-step and a variable-step arm with the selection frozen per layer and the router term off (TKF-R6, TKF-VS), so the trained model is per-token conservative at fixed context. If it removes little, freezing the selection is a conservativity choice to weigh on its own merits, not a refinement repair.

---

<a name="tkf-a-result"></a>
## 6. TKF-A: result (protocol `p-tkf-attribution-result`)

**2026-11-08. Re-selecting $V_\phi$'s sources carries none of the refinement penalty, on either model.** Dropping the router term costs nothing at $N = 8$. It removes about a tenth of the fixed-step model's penalty at $N = 24$ and 32, and nothing of the variable-step model's. **TF-1, TF-2 and TF-4 hit; TF-3 misses.** The calls were registered before any Gate 3 number and not confirmed by the author; they are scored as registered.

CPU, one model at a time, one sequence per forward (`micro=1`): K2-L8-R6-CODE 2.4 h, K2-L8-R6-VS-CODE 2.5 h, peak memory 19.2 and 19.4 GB. Outputs: `topk_freeze_attribution_{k2-l8-r6-code,k2-l8-r6-vs-code}{_output.txt,.json}` in each arm's results folder. Every check passed on both models. The as-trained arm reproduces Cell 6b-7's Gate 3 to the printed precision on both (perplexity 4.740 and 4.813 at $N = 8$).

Penalty in nats, each arm against its own $N = 8$ (the router-off arm's $N = 8$ perplexity is 4.740 for K2-L8-R6-CODE and 4.818 for K2-L8-R6-VS-CODE):

| arm | R6, N = 12 | 16 | 24 | 32 | VS, N = 12 | 16 | 24 | 32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| as trained | 0.251 | 0.499 | 1.279 | 1.727 | −0.008 | 0.001 | 0.038 | 0.081 |
| top-k frozen | 0.250 | 0.495 | 1.287 | 1.735 | −0.008 | 0.001 | 0.038 | 0.081 |
| top-k frozen, router off | 0.251 | 0.490 | 1.153 | 1.590 | −0.008 | 0.001 | 0.037 | 0.081 |
| per-layer flow | 0.425 | 0.066 | 0.059 | 0.060 | 0.371 | 0.031 | 0.034 | 0.035 |
| all frozen | 0.424 | 0.068 | 0.062 | 0.064 | 0.371 | 0.031 | 0.034 | 0.035 |

R6 is K2-L8-R6-CODE; VS is K2-L8-R6-VS-CODE.

- **Selection frozen.** Paired against as trained over the 12 batches, K2-L8-R6-CODE changes by $-0.004$ (SD 0.003) at $N = 16$, and by +0.008 and +0.009 at $N = 24$ and 32 (SD 0.009 and 0.015). K2-L8-R6-VS-CODE stays within 0.001 at every $N$. **TF-1 hit** (1% of the $N = 16$ penalty removed). **TF-2 hit** (none of the $N = 32$ penalty removed).
- **Router term off.** At $N = 8$ it changes the loss by $-0.0000$ nats (SD 0.0003) on K2-L8-R6-CODE and +0.0012 (SD 0.0020) on K2-L8-R6-VS-CODE, although it moves single logits by up to 2.0 and 1.9. **TF-3 miss.** Beyond the trained step it helps the fixed-step model a little: $-0.009$ (SD 0.005) at $N = 16$, $-0.126$ (SD 0.075) at $N = 24$ and $-0.137$ (SD 0.048) at $N = 32$, that is 2%, 10% and 8% of the penalty. On the variable-step model it changes nothing at any $N$. The router-off arm also freezes the selection. Since freezing alone removes nothing, the gain is put down to the router term. Router off with the selection live was not run.
- **Inside the per-layer flow**, freezing the selection changes the penalty by at most +0.003 nats (K2-L8-R6-CODE; SD 0.001) and +0.001 (K2-L8-R6-VS-CODE) at every $N \ge 16$. **TF-4 hit.** The flow limits on these batches, 0.060 and 0.035, match FLOW-R's own 0.059 and 0.034. On K2-L8-R6-CODE the per-layer flow removes 87%, 95% and 97% of the whole-stack penalty at $N = 16$, 24 and 32.
- **The router term's size.** This is the change in the $V_\phi$ force per token when the router term is dropped, relative to the force with it, at each force evaluation point at $N = 8$, on $2 \times 512$ tokens. K2-L8-R6-CODE: median at most 0.004 in every layer; 90th percentile 0.05 to 0.08 in layers 0 to 5, and at most 0.003 in layers 6 and 7. K2-L8-R6-VS-CODE: median at most 0.019; 90th percentile 0.09 to 0.43 (layer 7 0.43, layers 3 and 4 0.33 and 0.36). Variable-step training made the term heavier in the tail without making it matter to the loss.
- **Churn** (as trained; the fraction of a query's 16 sources also selected at its layer's first substep):

| layer | K2-L8-R6-CODE | K2-L8-R6-VS-CODE |
| --- | --- | --- |
| 0 | 0.66 after one substep at N = 16; 0.57 after three at N = 32 | 0.83 and 0.86 |
| 1 to 6 | 0.93–0.96 after one | 0.93–0.96 |
| 7 | 0.90, falling to 0.77 | 0.85, falling to 0.82 |

  A third of layer 0's sources change within one substep of the fixed-step model, and freezing them still changes nothing.
- **Reading.** The fixed-step model's whole-stack failure is not a selection effect. It sits in what the per-layer flow holds per layer: the LayerNorm projection after every substep, $\xi$ and the quadratic. Holding those removes nearly all of it. The top-k selection is a conservativity question, not a refinement one. Frozen per layer, it makes the $V_\phi$ force without the router term an exact gradient at fixed context, at no cost at $N = 8$ and with no loss of refinement on either model, as evaluated here.
- **For Phase 2.** TF-1 hit, so by the registration freezing the selection is a conservativity choice to weigh on its own merits. A trained arm with the router term off meets one more problem. The gate's forward value is the hard mask, so the score head reaches the loss only through the router term. With the term off in training, the score head gets no gradient and the selection stays at its initialisation. A trainable version needs the selection learned some other way: an auxiliary loss on the score head, a soft (dense) selection, or a trained score head frozen. **That is the author's decision.**

---

<a name="ro-cost"></a>
## 7. RO-COST: the router term at evaluation (protocol `p-router-off-cost`)

**2026-11-08; descriptive, CPU. Dropping $V_\phi$'s straight-through router term at evaluation changes the loss of every code model by at most 0.0013 nats, in either direction.** The author asked for it, to back a stronger conservativity statement in the book before the v6.2 release. TKF-A had measured the router term's cost on two models and 12 batches; RO-COST extends it to seven models.

- **How it was run.** `debug/router_off_cost.py <out> <folder> name=<arm> [ckpt=…]` runs the model's own forward at its trained step count, with the analytic $V_\phi$ force as in training, twice per sequence. "Router on" is the trained model. "Router off" detaches the gate and zeroes the soft weights, so the router term is exactly zero while the selection and the gate's value are kept.
- **Data.** 48 batches of $4 \times 512$ code validation tokens at Cell 6b-7's seed (98,304 tokens; its 12 batches first), paired by batch. The best checkpoints: step 18,000 for the $L = 4$ and 8 models; SO-CODE's and SR4a-CODE's `_best.pt`. One model at a time, peak memory 5.4 GB. Outputs: `router_off_cost_<arm>{_output.txt,.json}` in each arm's results folder.
- **Checks, all passing on all seven.** The analytic and autograd forces give the same loss on one sequence (relative 1e-7). At one layer's input in float64, the switch keeps the selection and the gate's value; the analytic and autograd pair forces agree with the router term on, off, and in their difference, to 1e-14. A float64 round trip leaves the model unchanged. On Cell 6b-7's 12 batches, the router-on and router-off losses reproduce TKF-A's as-trained and router-off losses at $N = 8$ (maximum 3e-6 nats per batch, analytic against autograd path).

Router off against router on. The loss change is router off minus router on, the mean over the 48 batches, with its 95% interval. KL is the mean per-token divergence of router off from router on. "Top-1 changed" is the share of tokens whose top prediction changes.

| model | L | loss on | Δ loss, router off (nats) | 95% interval | worst batch | KL per token | top-1 changed | largest logit change |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| SO-CODE | 2 | 1.69891 | −0.00127 | −0.00158 to −0.00096 | +0.00100 | 1.2e-03 | 0.815% | 14.22 |
| SR4a-CODE | 2 | 1.72563 | −0.00069 | −0.00094 to −0.00044 | +0.00080 | 7.3e-04 | 0.435% | 12.04 |
| L4-CODE | 4 | 1.61613 | −0.00057 | −0.00085 to −0.00029 | +0.00117 | 5.8e-04 | 0.553% | 11.62 |
| L8-CODE | 8 | 1.59511 | +0.00022 | −0.00002 to +0.00046 | +0.00273 | 3.5e-04 | 0.274% | 7.76 |
| K2-L8-CODE | 8 | 1.64469 | −0.00009 | −0.00031 to +0.00013 | +0.00201 | 3.9e-04 | 0.498% | 6.48 |
| K2-L8-R6-CODE | 8 | 1.62800 | −0.00004 | −0.00011 to +0.00004 | +0.00059 | 1.3e-04 | 0.178% | 13.67 |
| K2-L8-R6-VS-CODE | 8 | 1.64546 | +0.00064 | +0.00023 to +0.00105 | +0.00649 | 9.4e-04 | 0.526% | 9.12 |

**Reading.**

- The router term matters to individual tokens: a fraction of a percent of top-1 predictions change, and single logits move by up to 14.
- It does not matter to the model's loss. Dropping it lowers the loss on SO-CODE, SR4a-CODE and L4-CODE (by 0.0013, 0.0007 and 0.0006 nats; the intervals exclude zero). It raises the loss only on K2-L8-R6-VS-CODE (+0.0006), and leaves the other three within their noise.
- So a trained model evaluated without the router term is per-token conservative at fixed context and fixed selection, at no cost to its perplexity on code.
- The selection itself is still piecewise constant, so the potential jumps where a token's top-k sources change. Only a soft selection removes that.
- Training without the term is a different matter: with it off, the score head gets no gradient (section 6).
- **In the book (v6.2):** §7 (the per-token conservativity statement, `07_lagrangian.tex`) and the router-term paragraph of §19.6.1 (`subsubsec:parf-vphi-analytic`).

---

<a name="open"></a>
## 8. What stands and what is open

- **Stands.** On every code model the whole stack fails refinement while the per-layer flow settles. The limit is 0.37 nats at $L = 2$, 0.31 at $L = 4$ and 0.035 at $L = 8$, one seed each. The fixed-step model's whole-stack penalty sits in what the per-layer flow holds per layer, not in the top-k selection. The router term costs at most 0.0013 nats at evaluation on seven code models.
- **Open.** K2-CODE ($L = 2$, two wells) is set up with calls K2a to K2d proposed and not confirmed; no run is recorded in these entries. Router off with the selection live was not run. A Phase 2 arm trained without the router term needs the selection learned some other way, which is the author's decision.
