# The Conservative Core: Productization Plan

**What this note is.** The plan for closing the speed and memory gap between the Gen 3 conservative core (SR2, K4-R6, K2-R6 and their first-order and code twins) and the matched GPT-2 at inference. It starts from what was measured on 2026-11-17, lists every lever in two groups (those that leave the trained model unchanged, and those that need a new training run), and sets the order of work. Written 2026-11-17.

**Status.** A plan. The measurements in §1 and §2 are done ([CPU_Inference_Benchmark.md](../depth_ladder/CPU_Inference_Benchmark.md)); stage 1 (the levers of §4 that run on the CPU, and the first reading for T3) is measured in [Stage1_CPU_Levers.md](Stage1_CPU_Levers.md) (2026-11-17); stage 2 (the incremental decoder, N4, and `torch.compile`, N2) in [Stage2_Decoding.md](Stage2_Decoding.md) (2026-11-17); stage 3's first arm, SR2 trained without the pair force, is trained and scored in [Stage3_SR2_Without_Pair_Force.md](Stage3_SR2_Without_Pair_Force.md) (2026-11-21: settled 58.51 against SR2's 58.32). That result answers T2 for the exact-flow core (§5.1): a field has at most 0.2 to 0.3% to recover there, so the mean field leaves the productization path and stays a research item on F3.1. The calls for the later GPU arms are not yet written; each gets a pre-registration, confirmed by the author, before it launches.

**Decision (2026-11-17).** The pair force stays disabled for the rest of the productization work: every later lever is measured on the models with $V_\varphi$ removed, the configuration of the bound in §1. Replacing it by a mean field (T2) is part of this plan and postponed; §5.1 holds that agenda.

**Why SR2 is the base (author, 2026-11-22).** SR2 is the simplest model of the Gen 3 line, the one with the fewest additional features: no Poisson-mode wells, no exchange field and no register path in use. It is not the best predictor. The models with Poisson-mode wells settle about 7% lower (PMX 54.63, PMX-K16 54.24, FO-b-PMX-K16 53.73, against SR2's 58.32). **Next in line are the PMX models, in both orders:** second order (PMX, PMX-K16, PM2-PMX16) and first order (FO-b-PMX, FO-b-PMX-K16, FO-b-PM2-PMX16). On that line the pair force may earn its place, which would return the mean field to the productization path (§5.1, and the PMX line in §6).

**Predecessor.** [Fock_Inference_Productionization_Plan.md](Fock_Inference_Productionization_Plan.md) (2026-09-18) did the same for the Gen 2 Fock arm at eight layers: a cost model by multiply–accumulate counts, A100 measurements, and the training-speed levers of the 2B-token runs. §3 below says which of its levers carry over.

---

## Contents

1. [The gap today](#1-the-gap-today)
2. [Where the time and the memory go](#2-where-the-time-and-the-memory-go)
3. [What carries over from the Gen 2 plan](#3-what-carries-over-from-the-gen-2-plan)
4. [Levers that leave the trained model unchanged](#4-levers-that-leave-the-trained-model-unchanged)
5. [Levers that need a new training run](#5-levers-that-need-a-new-training-run) (and [5.1, the mean-field agenda](#51-the-mean-field-agenda-for-the-pair-force-t2-postponed))
6. [The order of work](#6-the-order-of-work)
7. [What this plan does not claim](#7-what-this-plan-does-not-claim)
8. [Files and related notes](#8-files-and-related-notes)

---

## 1. The gap today

One forward pass over one sequence (prefill) on an Intel Core i9-9880H, 8 threads, float32, PyTorch 2.2.2, with the closed-form pair force; tokens per second, and in brackets how many times slower than the matched GPT-2 with fused attention:

| model | 512 | 2,048 | 8,192 | peak GB at 8,192 | perplexity, 16 validation windows of 512 |
| --- | ---: | ---: | ---: | ---: | ---: |
| matched GPT-2, fused attention | 2,504 | 2,516 | 1,652 | 6.2 | |
| SR2, as trained | 503 (5.0) | 289 (8.7) | 116 (14.2) | 40.1 | 58.28 |
| SR2, memory switches on | 420 (6.0) | 229 (11.0) | 99 (16.7) | 11.7 | 58.28 |
| K4-R6, memory switches on | 615 (4.1) | 351 (7.2) | 118 (14.0) | 11.8 | not measured (settled 59.62) |
| SR2, pair force removed, switches on | 886 (2.8) | 687 (3.7) | 385 (4.3) | 9.6 | 59.36 |
| K4-R6, pair force removed, switches on | 1,190 (2.1) | 861 (2.9) | 434 (3.8) | 9.2 | 60.66 |

The last two rows are the bound: what removing the pair force $V_\varphi$ from the trained model could save, at the perplexity cost of the removal. The book reports it in Appendix C.10 as the bound the work aims to come close to. **Even at the bound the model is 2.1 to 3.8 times slower than fused attention**; closing that remainder is what §4 and §5 are for.

**Decoding is not measured.** No incremental decoder exists for these models. Read from the code, a decoder would keep about 1,040 numbers per token and layer with the pair force and 624 without it (keys 240, states 384): 2,080 and 1,248 per token at two layers, against $2 L d = 6{,}144$ for the matched GPT-2 at eight layers. So the Gen 3 models are not constant-memory: their decoding state grows with the context, as a key–value cache does, at about a fifth to a third of GPT-2's size ([CPU_Inference_Benchmark.md](../depth_ladder/CPU_Inference_Benchmark.md), §1; the book, Appendix C.10).

## 2. Where the time and the memory go

Shares of the CPU profile for SR2 as trained:

| length | elementwise, copies, reductions | matrix products | decompositions | softmax, top-k |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 47.0% | 36.1% | 15.8% | 1.1% |
| 2,048 | 53.3% | 33.3% | 12.4% | 1.1% |
| 8,192 | 66.2% | 24.9% | 7.7% | 1.3% |
| 8,192, pair force removed | 33.7% | 54.7% | 9.6% | 2.0% |

Read with the code, the costs come from four places:

- **The pair force's score network.** It runs on every causal pair of positions with a hidden width of 32. It holds most of the elementwise share at long context (removing it takes that share from 66% to 34% at 8,192 tokens) and, before the memory switches, most of the 40 GB.
- **The well bank $V_\theta$.** A hypernetwork: at each layer and each token it generates the wells' centres, scales and stiff directions from the token's own state and context channels, then evaluates the force. Its projections are the widest matrix products in the model: `B_proj` maps 1,920 numbers to 12,288 per token and layer.
- **The context channels $\xi$.** Five content-routed channels per layer: a softmax over a distance term plus a query–key content term, with the earlier states as values. This is attention with a distance bias, so its cost grows as $T^2$ in prefill. Its share is not separated in the profile.
- **The stiff-mode decompositions.** One batched float64 decomposition of a $32 \times 32$ Gram matrix per token and layer (the `gram` driver, which every ladder model already uses). 8% to 16% of the time.

The Fock registers cost almost nothing once skipped: with the reverse channel off nothing reads them (`skip_unread_registers`).

## 3. What carries over from the Gen 2 plan

The Gen 2 plan found 87.6% of the multiply–accumulates in $V_\theta$'s parameter generation and proposed five levers. Measured since:

| Gen 2 lever | what it was | status on the Gen 3 core |
| --- | --- | --- |
| A, factorize W_B | a bottleneck in `B_proj` | measured on SR2 without retraining: width 512 costs +41%; trained as twin PB, proposed and not run ([Well_Screen_and_Precision.md](../depth_ladder/Well_Screen_and_Precision.md), §5) |
| B, factorize W_μ, W_a | bottlenecks in the centre and scale maps | W_μ cannot be factorized faithfully (Gen 2 plan, §6.1); replacing `a_proj` by one learned scale per well costs +1.2% without retraining (twin PA, proposed and not run) |
| C, context-independent B | fixed or shared stiff directions | refuted at layer 0: a fixed B costs +846%, every well taking the strongest well's B +1,252%; layer 1 is compressible (a shared B costs +5.7%) |
| D, ξ as a recurrence | the channels as exponential averages | no longer exact: the channels are content-routed in every ladder model; see lever T3 |
| E, fuse the score network, cache its projection | tiling the pair force | done in part: the memory switches build the pair force a block of queries at a time |

The closed-form pair force of the Gen 2 plan's §10 is the trained configuration of every Gen 3 model, and the benchmark uses it.

## 4. Levers that leave the trained model unchanged

Each one gives the same logits up to float rounding, so it needs an exactness check, not a training run. The check is the one the memory switches passed: logits and loss against the unmodified model on the same windows, with counters confirming the new path ran (`verify_inference_chunking.py`).

**N1. Fused attention for the context channels.** The channels compute

$$\xi_{k,t} = \sum_{s \le t} \mathrm{softmax}_s\left(b_k(t - s) + \frac{q_{k,t} \cdot \kappa_{k,s}}{\sqrt{d_\kappa}}\right) h_s ,$$

with $b_k$ the channel's distance term, $q$ and $\kappa$ its query and key (width $d_\kappa = 48$) and $h$ the states. This is exactly `scaled_dot_product_attention` with the distance term as an additive mask and the states as values (value width 384, shared by the five channels). On a GPU the distance term can be generated inside the kernel (FlexAttention's score modification), so no $(T, T)$ tensor is ever built. Whether PyTorch 2.2.2 on the CPU takes a fused kernel with an additive mask has to be checked; if it does not, the gain is only what the block-of-queries switch already gives. The matched GPT-2 gains 3.7 times at 8,192 tokens from the same change (443 against 1,652 tokens per second).

**N2. Compiling the layer step.** `torch.compile` on one layer's step, to fuse the elementwise work of the integrator, the bank's force and the pair force's score network. Elementwise work is 47% to 66% of the CPU time, and about half the GPU time in training (the Gen 2 plan, §10.3b). The pair force's top-k selection and the decomposition may break the compiled graph; how much it fuses is what the test measures.

**N3. Pruning layer 1's bank to one well.** Layer 1 runs on one well: keeping only its strongest well costs +0.0% on K4-R6 and on K2-R6 (calls KS5 and KS9, both hits), and six directions suffice. At inference the other wells can be dropped from layer 1's projections. That is a removal, so it changes the logits slightly; its check is the loss on the validation windows, not bit-exactness. Layer 0 cannot be pruned this way: it fills every direction it is given.

**N4. An incremental decoder.** Without one, decoding speed cannot be measured at all, for these models or against GPT-2 with its key–value cache. With the pair force it caches the states and the score network's projections; without it, the channels' keys and the states. It is exact by construction and a prerequisite for any claim about generation speed.

**N6. Flushing subnormal floats** (found 2026-11-22). The folded context channels' keys grow with position, and the fused softmax then produces subnormal floats, slow on x86. Flushing them is one global switch with bit-identical output: SR2-NP's prefill goes from 594 to 1,096 tokens per second at 8,192 tokens, and the gap to the matched GPT-2 (flushed too) from 4.9 to 3.0 times ([Stage1_CPU_Levers.md](Stage1_CPU_Levers.md), §8).

**N5. The decompositions.** 8% to 16% of the time by the operator profile; the component timers of stage 1 put the whole decomposition step at 21% at 512 tokens, and 34% once the pair force is removed. In float64. A float32 decomposition at inference would be cheaper, but the first float32 version of this solver failed on real matrices (2026-09-17, the note in `_gram_eigh`), so it would need checking on every model. Last in priority.

## 5. Levers that need a new training run

Each changes what the model can express, so each needs a trained arm on one A100, with its calls written and confirmed before launch. A removal from a trained model measures how much the model relies on a term, not what a model trained without it would lose.

**T1. Train without the pair force.** The largest lever: it gives the bound in §1. The pair force carries none of the recall on code: cutting the earlier occurrence from layer 1's pair force changes the far-recall loss by +0.000 nats in both orders ([Second_Order_Memory_Probes.md](../depth_ladder/Second_Order_Memory_Probes.md), probe 6). That layer-1 test is weak where layer 1's pair force is nearly inert, as in SR2 on OpenWebText. At layer 0, where SR2's pair force acts, removing the earlier occurrence costs 8% of the pair force's value at those positions ([Pair_Force_Recall_Split_FO_b_PMX_K16.md](Pair_Force_Recall_Split_FO_b_PMX_K16.md), §7). VF0 (F3.1 trained without $V_\varphi$) is pre-registered and not run ([Coupling_Removal_Costs_CD.md](../depth_ladder/Coupling_Removal_Costs_CD.md), §5); it answers the value question on F3.1, for the mean-field agenda (§5.1). The productization arm is SR2 trained without it, SR2-NP ([Stage3_SR2_Without_Pair_Force.md](Stage3_SR2_Without_Pair_Force.md)).

**T2. A field form of the pair force.** Postponed (2026-11-17); the agenda is §5.1. The routed force is replaced by its interaction with a mixture of a few components, carried by content-gated exponential averages, so its cost per token no longer grows with the context ([Mean_Field_PARFLM_Programme.md](../Mean_Field_PARFLM_Programme.md), §1 and the field arm FA-OWT).

**T3. Content-gated context channels, layer by layer.** The same mixture used for the channels gives a fixed decoding state and prefill linear in $T$ (the mean-field note, §1.1). But **recall runs through layer 1's content-routed channels**: removing the earlier occurrence from them costs +0.794 nats at far-recall positions in SO-CODE and +0.740 in FO-b-CODE, against +0.000 at control positions (probe 6). Replacing layer 1's channels risks exactly that, since a mixture cannot single out one earlier token. The candidate is therefore a hybrid: content-gated channels at layer 0, content-routed at layer 1, decoded with a cache at layer 1 only. What layer 0's content term carries has not been measured. Its first reading is a removal on the trained model, where setting layer 0's query map to zero turns its channels into exact exponential averages (§6, stage 1).

**T4. A smaller bank.** K4-R6 is measured: +2.2% in perplexity, 1.14 times faster than SR2 at 512 tokens and level at 8,192 ([Well_Screen_and_Precision.md](../depth_ladder/Well_Screen_and_Precision.md)). Further candidates are a per-layer bank (layer 0 as now, layer 1 with one well of rank 6, which N3 measures without retraining first) and the twins PA and PB, which shrink the bank's projections. The bank is a small speed lever on its own; it matters most once the pair force is gone and matrix products are 55% of the time.

**Not a lever: first-order dynamics (checked 2026-11-22).** Training the core as a first-order gradient flow (FO-b, `FIRST_ORDER_OVERDAMPED = True`) instead of the second-order CfC+BAOAB step gives almost nothing in time or memory:

- **What it removes is cheap.** FO-b replaces the layer step A, B, O, A (exact flow for half a step, kick, friction, exact flow for half a step) by an A, B, A step of the gradient flow. It drops the friction step and the decoding and encoding of the velocity: a few elementwise operations on one 384-vector per token and layer. It keeps everything that dominates the time: the context channels, the well bank, and the stiff-mode decomposition, since FO-b still relaxes the stiff modes exactly. With every CPU lever on, those three take 75% of SR2-NP's forward at 512 tokens and 85% at 8,192 (component timers, [Stage1_CPU_Levers.md](Stage1_CPU_Levers.md), §8).
- **Measured in training:** on the one pair trained on the same card with the same settings, PM2-CODE and its first-order twin FO-b-PM2-CODE (A100 80 GB, 19,250 steps), 1.23 against 1.19 s per step: first order is 3% faster ([Code_Set_First_vs_Second_Order.md](../depth_ladder/Code_Set_First_vs_Second_Order.md), §6.2). The OpenWebText twins were trained on different cards with different memory switches, so their step times do not compare.
- **Memory.** The decoding cache does not change: past tokens are cached through their states and their channel keys (1,248 numbers per token), and the velocity belongs only to the token being decoded. Prefill and training activations lose one 384-vector per token and layer, a few percent at most.
- **The price.** Without Poisson-mode wells, SR2-NP's base, first order costs 4.9% in perplexity (FO-b-OWT 61.16 against SR2's 58.32). It draws level or ahead only with the wells, which add cost of their own.

So the order of the dynamics is a modelling question, not a productization lever: about 3% in speed against about 5% in perplexity on the production base. An inference timing of FO-b-OWT against SR2 on the CPU has not been taken; from the components above it should differ by a few percent.

### 5.1 The mean-field agenda for the pair force (T2, postponed)

**Answered for the exact-flow core (2026-11-21).** SR2-NP, SR2 trained without the pair force, settles at 58.51 against SR2's 58.32 (+0.3%), and reads 58.41 against 58.28 on the 16 windows where removing SR2's pair force costs 1.8% ([Stage3_SR2_Without_Pair_Force.md](Stage3_SR2_Without_Pair_Force.md)). On the production base a field could therefore recover at most 0.2 to 0.3% in perplexity, while adding back sources and a recurrence to a model that already runs at the bound. The risk below, "little to replace on SR2", is confirmed in value, and part of the pair force's share is already taken by the velocity ([Velocity_and_Pair_Force_as_Competing_Mechanisms.md](../depth_ladder/Velocity_and_Pair_Force_as_Competing_Mechanisms.md)). T2 leaves the productization path. The mean field stays a research question on F3.1, where the pair term carries 0.282 nats, and becomes a productization lever again only if a production model is found that values the pair force (beside the registers or the Poisson-mode wells, at longer training, or at more layers). The rest of this section is kept as written, for that case and for the research programme.

**The framing (author, 2026-11-22).** The pair force is a useful mechanism whose place depends on the model and the use case: it is idle in SR2 and carries more in the first-order models with Poisson-mode wells. So the mean field is not only a research item. Where the pair force is kept, the field is what makes keeping it affordable: no quadratic score network, and a fixed decoding state instead of a growing one. The PMX line (§6) is where that case is being tested.

Disabling the pair force gives the upper bound on speed. The intended replacement is a mean field: the same pair potential, evaluated against a few summaries of the past instead of against routed earlier tokens. This section records how it would be built and tested, so that it can be resumed as it stands. It is postponed by the author's decision of 2026-11-17.

**What it replaces.** Today, per layer and query token $t$, a score network runs on every earlier token (the $(T, T, 32)$ tensors), the 16 highest-scoring are kept, and the force is the gradient of $\sum_j m_{tj} V_\varphi(h_t, h_j)$ plus a router term. Each of the four heads is $V_\varphi = -C \Theta \Phi / r$: a value aligner, a type gate on a learned distance, and the inverse distance in state space. The score network is what grows as $T^2$; in stage 1 the pair force took 38% to 69% of SR2's time.

**The field** ([Mean_Field_PARFLM_Programme.md](../Mean_Field_PARFLM_Programme.md), §1 and the field arm FA-OWT):

$$f_t = -\lambda_\ell \nabla_{h_t} \sum_{m=1}^{M} \pi_{m,t} V_\varphi(h_t, \mu_{m,t}),$$

with $\pi_{m,t}$ and $\mu_{m,t}$ the weights and means of a mixture of $M$ components, built from the earlier states by content-gated exponential averages: prototypes $c_m$, responsibilities $r_{s,m} = \underset{m}{\mathrm{softmax}}(-\lVert h_s - c_m \rVert^2 / \tau)$, and averages over $s \lt t$ only. No score network, top-k or router remains.

**Why the implementation is small.**
- The closed-form force `_pair_force_analytic` already evaluates the heads against a set of gathered sources (`h_src_g`, shape `(B, T, k, d)`) with per-source weights. The field passes the $M$ component means as the sources and $\lambda_\ell \pi_{m,t}$ as the weights, and drops the router term; the per-head code is reused as it is.
- The mixture is a linear recurrence: $O(T M d)$ in prefill as a chunked scan, and a fixed decoding state of $M (d + 1)$ numbers per layer. It must not be built from the $(T, T)$ weight matrix the context channels use, or the $T^2$ cost returns.
- Conservativity holds as now: the sources are detached and a token is excluded from its own field, so the force is the gradient of a potential in $h_t$.
- Per token the heads see $M$ sources (8 against today's 16) and no score network, so the speed should land close to the bound. That is an estimate until measured.

**The work, in order.**

| step | what | effort | machine |
| --- | --- | --- | --- |
| 1 | Probe A: k-means prototypes on a trained model; the field force against the model's force (cosine), and the loss with the field substituted | 1–2 days, code and runs | CPU |
| 2 | the field module behind a switch; an equivalence test (sources set to the gathered top 16 and weights to the gate reproduce today's pair term exactly) and the scan checked against the matrix form | 2–3 days | CPU |
| 3 | VF0, F3.1 trained without the pair force: the value the field is scored against | about 14 h | A100 |
| 4 | FA-OWT, the field arm with learned prototypes, temperature τ and scale λ_ℓ | about 14 h, after VF0 | A100 |

Probe A's calls A0 to A5 and the field arm's FA1 to FA3 are written in the mean-field note as proposed, not confirmed.

**The risks are scientific, not in the code.**
- **Retrieval.** A mixture cannot single out one earlier token; where the routed force retrieves, the field loses it (call A4). On code the pair force carries none of the recall (+0.000 nats, [Second_Order_Memory_Probes.md](../depth_ladder/Second_Order_Memory_Probes.md), probe 6), which favours the field.
- **Little to replace on SR2.** Removing the pair force costs SR2 1.8% (§1), and call A5 predicts nothing is left for the field to reproduce there. The programme's base is F3.1, where the pair term carries 0.28 nats. For SR2-class models, VF0's analogue may show that the field is not needed and the bound is the model.
- **The monopole approximation.** $V_\varphi$ is evaluated at each component's mean, not averaged over the component; sigma points are the next order if call A1 misses.

**When to resume.** As research: when the author lifts the postponement, with Probe A on F3.1 first; it needs no training and is the cheapest test of whether a field can represent the trained force at all. Its calls A1 to A5 were confirmed by the author on 2026-11-21 (A0 is proposed only); the probe's code is not written yet. As a productization lever: only under the condition at the top of this section.

## 6. The order of work

1. **Stage 1, CPU, no retraining.** First separate the context channels' share of the profile. Then measure N1, N2 and N3, alone and together, on SR2 and K4-R6, with and without the pair force, at 512, 2,048 and 8,192 tokens, each with its exactness check. In the same session, take the layer-0 content-term removal for T3. One job at a time. **Done 2026-11-17** ([Stage1_CPU_Levers.md](Stage1_CPU_Levers.md)), except N2, which needs a Python 3.11 environment.
2. **Stage 2, CPU.** Build the incremental decoder (N4) for the configuration stage 1 leaves best, with the pair force disabled, and measure decoding speed and state size against the matched GPT-2 with its key–value cache. **Done 2026-11-17** ([Stage2_Decoding.md](Stage2_Decoding.md)): the decoder reproduces the full forward to float rounding; with every lever SR2 decodes 1.9 to 2.6 times slower than GPT-2 with its key–value cache, K4-R6 1.8 to 2.1 times, with a cache a fifth of GPT-2's; a decode step is bound by per-operation overhead.
3. **Stage 3, GPU, one arm at a time.** First T1 as SR2 trained without the pair force (SR2-NP), on SR2's recipe and data so the two pair: trained 2026-11-21, settled 58.51 against SR2's 58.32 (+0.3%), at SR2's inference speed without the pair force; the base for the later levers ([Stage3_SR2_Without_Pair_Force.md](Stage3_SR2_Without_Pair_Force.md)). VF0 (F3.1 without the pair force) moves to the mean-field agenda (§5.1), whose question it answers. Then a leaner bank on the SR2-NP base if stage 1 supports one, and T3's hybrid only if a trained arm can recover what layer 0's content term carries (stage 1 found the trained model relies on it heavily). After SR2-NP (2026-11-21): the leaner bank is not supported, since layer 1 already runs on one well by pruning at no cost (keeping well 5: +0.0%) while layer 0 needs all eight (keeping seven: +15.5%); T3 remains, gated first by a CPU bound on its speed gain (SR2-NP timed with layer 0's content term removed) and then by a per-layer content-routing switch in the code. T2 is answered for this core and leaves the path (§5.1).

**What remains of the speed push (2026-11-22).** With T1 done, T2 answered, the leaner bank of fewer wells unsupported and first order not a lever (§5), four items are left. In order:

1. **N6, flushing subnormal floats.** Done: SR2-NP's prefill at 8,192 tokens from 594 to 1,096 tokens per second, bit-identical ([Stage1_CPU_Levers.md](Stage1_CPU_Levers.md), §8).
2. **The context channels as 30 heads of 64.** Measured: 1.8 times on the channels and 1.18 times on the forward at 8,192 tokens, exact to float32 rounding; worth it above about 1,000 tokens (same note, §8; analysis in [Stage1b_Context_Channels_as_30_Heads.md](Stage1b_Context_Channels_as_30_Heads.md)). Not scheduled (2026-11-23): it helps only the prompt phase of a forward pass, not token-by-token decoding, which is bound by per-operation overhead ([Stage2_Decoding.md](Stage2_Decoding.md), §4), and no planned use case has prompts beyond about 1,000 tokens. If long prompts become a use case, it goes into the model code as an inference-only switch, with the key term centred (halves the float32 error) and a width among 64, 96 and 128 chosen by one interleaved run.
3. **A fused or compiled prefill core** for the stiff-mode decomposition and the integrator, which now take about half the forward at every length. Compiling the decoding core gave 1.28 to 1.40 times ([Stage2_Decoding.md](Stage2_Decoding.md), §5). **Measured 2026-11-22 in prefill:** compiling the same core gives 1.02, 1.39 and 1.17 times at 512, 2,048 and 8,192 tokens on SR2-NP with every other lever on, logits as eager, about 1.5 GB more peak memory at 8,192 (same note, §5). Worth it from about 2,048 tokens; at 512 the time is in the large matrix products, which item 4 addresses.
4. **A smaller well bank at layer 0, trained:** the factorised precision map of the Gen 2 plan (twin PB, §3). The bank is the largest per-token cost at short context; removing it without retraining costs +41%, so it needs a GPU arm with its own pre-registration. This, not fewer wells, is what the "leaner bank" of stage 3 should mean. **Pre-registered 2026-11-22:** [Stage3c_Twin_PB_Precision_Bottleneck.md](Stage3c_Twin_PB_Precision_Bottleneck.md). The bank is one module shared by both layers, so the bottleneck applies at both; timed without training, PB's arithmetic makes the forward pass 1.11 times faster at 512 tokens, and the untrained truncation costs SR2-NP +42%. Calls PB1 to PB3 and the +2% tolerance confirmed by the author the same day; the base is SR2-NP (SR2-NP-T missed TE1).

T3 (hybrid context channels) stays a candidate behind its CPU speed bound. Second arm, trained 2026-11-22: SR2-NP with tied embeddings (SR2-NP-T), which asked whether the separate output head, a quarter of the parameters and no speed, is needed ([Stage3b_SR2_NP_Tied_Embeddings.md](Stage3b_SR2_NP_Tied_Embeddings.md)). **It is:** SR2-NP-T settled at 59.35 (+1.44%) and missed TE1, so the untied head stays. The tie helps the rare tokens and costs the most frequent fifth, which carries 89% of the targets. Twin PB therefore runs on SR2-NP. A fourth corner, SR2 with tied embeddings and the pair force (SR2-T), is set up as a measurement of competing mechanisms, not a lever: its call ST1 is the trigger for returning the mean-field programme to this path ([Stage3d_SR2_Tied_With_Pair_Force.md](Stage3d_SR2_Tied_With_Pair_Force.md)). The author sets the perplexity tolerance before each arm.

**The PMX line (author, 2026-11-22).** The first-order twins with Poisson-mode wells lean on the pair force 1.2 to 2.1 times as hard as their second-order twins (3 times on code), so on this line the pair force may be worth keeping, and its cost then makes the mean field (§5.1) a productization lever again. Three steps, in order:

1. **CPU: where the pair force's cost sits in FO-b-PMX-K16.** Done 2026-11-22 ([Pair_Force_Recall_Split_FO_b_PMX_K16.md](Pair_Force_Recall_Split_FO_b_PMX_K16.md)). The pair force does not fetch the earlier occurrence of the predicted token: removing that one source costs about 4% of what removing the whole pair force costs at the same positions, and the router selects it only 1.3 times as often as chance. Its cost is larger per token at recall positions, but spread over all positions. There is no addressing for a mean field to lose, so the field is a credible replacement on this line; whether it can represent the force is Probe A's question.
2. **GPU: FO-b-PMX-K16 trained without the pair force**, the analogue of SR2-NP: its trained-out cost, set against a reliance of +3.6% ([Stage3e_FO_b_PMX_K16_Without_Pair_Force.md](Stage3e_FO_b_PMX_K16_Without_Pair_Force.md), pre-registration).
3. **CPU: Probe A on FO-b-PMX-K16.** Done 2026-11-22 ([Probe_A_Field_Pair_Force.md](../mean_field/Probe_A_Field_Pair_Force.md)): a field recovers 74% of its pair force's removal cost at eight components, less than on F3.1 (96%). Probe A's call A2 missed on the eight-layer code model, and the author ruled the same day that no field arm runs. A stand-in for the pair force on this line would need its own post hoc registration.
   - **Added steps (author, 2026-11-22): the rest of the pair force × tie grid on the PMX line.** The second-order companion PMX-K16-NP, then both models without the pair force and with tied embeddings, FO-b-PMX-K16-NP-T and PMX-K16-NP-T. Each tied arm runs after its untied reference, which isolates the tie's cost. Reason: SR2-T settled at 61.27, against 59.35 for SR2-NP-T ([Stage3d_SR2_Tied_With_Pair_Force.md](Stage3d_SR2_Tied_With_Pair_Force.md), §8), so tied embeddings and the pair force compete, and a tied model should drop the pair force. PMX-K16-NP-T against PMX-K16-NP, set beside SR2-NP-T against SR2-NP, also measures whether the wells change what the tie costs. The F3.1 counterpart is VF0-T after VF0 ([VF0_F31_Without_Pair_Force.md](../depth_ladder/VF0_F31_Without_Pair_Force.md)).

**Where the push ends (author, 2026-11-22).** The training side of the productization push ends with the pair force × tie grid above: VF0 and VF0-T, PMX-K16-NP and PMX-K16-NP-T, FO-b-PMX-K16-NP and FO-b-PMX-K16-NP-T, with SR2-T finishing, and twin PB (kept by the author the same day as the last productization arm: the only lever left on the bank's per-token cost). Its results then decide which conservative models besides SR2 are trained on 2B tokens, the programme the push preempted.

4. **After each stage.** Update §1's table, the benchmark note, and the book's Appendix C.10 where the bound or the measured state changes.

## 7. What this plan does not claim

- That the gap closes. The bound in §1 is an upper bound on one lever, read by removal; stage 1 measures how much of the remainder N1 and N2 recover.
- That any of these numbers hold on a GPU. They are CPU prefill numbers; the GPU training profile (the Gen 2 plan, §10.3b) also finds about half the time in elementwise work, but nothing in this note has been timed on a GPU at inference.
- That a model with all levers applied predicts as well as SR2. Each trained lever carries a perplexity cost that only its arm measures, and the costs need not add.

## 8. Files and related notes

- Benchmark and checks: [`cpu_throughput_benchmark.py`](../../notebooks/conservative_arch/scaleup/debug/cpu_throughput_benchmark.py), [`verify_inference_chunking.py`](../../notebooks/conservative_arch/scaleup/debug/verify_inference_chunking.py); outputs in [`results/cpu_bench/`](../../notebooks/conservative_arch/scaleup/results/cpu_bench/).
- The code each lever touches: the context channels in `notebooks/conservative_arch/multixi/model_multixi.py` (`MultiChannelXi.forward`); the pair force and the bank in `notebooks/conservative_arch/parf/model_parf_multixi.py`; the decomposition in `notebooks/conservative_arch/parf/cfc_baoab.py` (`_gram_eigh`); the register skip in `notebooks/conservative_arch/parf/model_fock_parf_multixi.py`.
- The measured state: [CPU_Inference_Benchmark.md](../depth_ladder/CPU_Inference_Benchmark.md). The bank sizes and the twins PA and PB: [Well_Screen_and_Precision.md](../depth_ladder/Well_Screen_and_Precision.md). The recall pathway: [Second_Order_Memory_Probes.md](../depth_ladder/Second_Order_Memory_Probes.md). VF0: [Coupling_Removal_Costs_CD.md](../depth_ladder/Coupling_Removal_Costs_CD.md). The field forms: [Mean_Field_PARFLM_Programme.md](../Mean_Field_PARFLM_Programme.md).
