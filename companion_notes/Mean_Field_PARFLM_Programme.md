# A Mean-Field PARFLM: Pre-Registered Programme

**What this note is.** A plan, with pre-registered probes and calls, for building and testing a mean-field (macroscopic) approximation of the basic conservative model, PARFLM: the context potential $V_\theta(h; \xi)$ plus the pair potential $V_\varphi$. It is the execution plan for the proposals of two earlier notes: [Macroscopic_Limits_of_the_Conservative_Step.md](Macroscopic_Limits_of_the_Conservative_Step.md), which proposes the field form of $V_\varphi$ and Probes A and B, and [Mean_Field_Theory_for_Particle_Language_Models.md](Mean_Field_Theory_for_Particle_Language_Models.md), which supplies the theory and Probe C. Everything up to the last stage runs on the CPU, on checkpoints already on the machine, one job at a time. The calls are proposed; the author confirms or changes them before each probe runs, as for every pre-registered item in the [protocol](depth_ladder/README.md). (Draft, 2026-11-11.)

**Status.** Probe A has run on five models (2026-11-22) and is scored in §5, with the full write-up in [mean_field/Probe_A_Field_Pair_Force.md](mean_field/Probe_A_Field_Pair_Force.md): A1 hit; A2, A3, A4 and A5 missed. **A2 missed at every M, which fires the §6 gate "stop Stages 6" (VF0 and the field arm). Author's ruling, 2026-11-22: no field arm; VF0 runs anyway, under the protocol's own entry (§5.21), to complete the investigation of the pair force: [depth_ladder/VF0_F31_Without_Pair_Force.md](depth_ladder/VF0_F31_Without_Pair_Force.md).** Probe A's calls A1 to A5 were confirmed by the author on 2026-11-21; everything else is PENDING and proposed. **A reopening trigger (2026-11-22):** the arm SR2-T (SR2 with tied embeddings, [productization/Stage3d_SR2_Tied_With_Pair_Force.md](productization/Stage3d_SR2_Tied_With_Pair_Force.md)) carries the call ST1 (confirmed by the author 2026-11-22): if the pair force recovers at least half of the tie's cost, a cheap stand-in for it would let the deployment model tie its embeddings, and this programme returns to the productization path, after a cheaper competitor (a linear map before the tied read-out) and Probe A on SR2-T's weights.

**Why the programme matters (author, 2026-11-22).** The pair force is a useful mechanism with its own place in Lagrangian modelling of language, and how much it carries depends on the model and the use case. It carries a third of the perplexity when it is the only token-to-token channel (F3.1, PM1), little under the exact stiff-mode flow or beside the registers, and more again in the first-order models with wells (1.2 to 2.1 times their second-order twins, 3 times on code). The mean field therefore has two kinds of significance:
- **scientific:** a macroscopic description of the pair interaction;
- **practical:** wherever the pair force is kept, the field removes its quadratic cost in context length and its growing decoding state.

**A second route back to productization (2026-11-22):** the PMX line of the productization plan. The pair force does no addressing in FO-b-PMX-K16, so a field loses nothing the router provides ([productization/Pair_Force_Recall_Split_FO_b_PMX_K16.md](productization/Pair_Force_Recall_Split_FO_b_PMX_K16.md)). Its value there is being measured by FO-b-PMX-K16 trained without it ([productization/Stage3e_FO_b_PMX_K16_Without_Pair_Force.md](productization/Stage3e_FO_b_PMX_K16_Without_Pair_Force.md)).

---

## Contents

0. [The one-page version](#0-the-one-page-version)
1. [What a mean-field PARFLM is](#1-what-a-mean-field-parflm-is) (and [1.1, the correction on ξ](#11-correction-2026-11-17-the-context-channels-are-content-routed))
2. [Where the pair term matters: the base is F3.1, not SR2](#2-where-the-pair-term-matters-the-base-is-f31-not-sr2)
3. [Material and rules](#3-material-and-rules)
4. [The sequence](#4-the-sequence)
5. [The probes, pre-registered](#5-the-probes-pre-registered)
6. [Decision gates](#6-decision-gates)
7. [What the programme does not claim](#7-what-the-programme-does-not-claim)
8. [Reporting](#8-reporting)

---

## 0. The one-page version

PARFLM has three couplings between tokens, and one of them is already a mean field. The context channels $\xi_t$ would be first moments of position-weighted densities of the past, read by $V_\theta$, if they were plain EMAs; in every ladder model they are content-routed, which makes them a second content-addressed lookup (§1.1, corrected 2026-11-17); the Fock registers are exclusion slots and lie outside this question. The one particle–particle term is the routed pair force: a sum of $V_\varphi(h_t, h_j)$ over the 16 earlier tokens a score head selects for each $t$. A mean-field PARFLM replaces that sum by the force of a *density* of the preceding states, carried as a mixture with $M$ components whose weights and means are content-gated EMA channels (Macroscopic §5.1). The router and its $(B, T, T, H_s)$ tensor disappear; the force remains a gradient because the mixture is detached, exactly as the sources are now.

The programme has two levels that validate each other. The **theory level** asks whether the trained model already behaves macroscopically: whether its well bank has phases of its own (Probe C), whether its states cluster with depth (Probe D), and whether a layer acts on a token through a few collective variables rather than its exact state (Probe E). The **model level** asks whether the routed force of a trained model is representable by a field (Probe A), and then whether a model trained with the field reaches the routed model (the field arm). The first level needs no new model; the second ends on a GPU.

The base is **F3.1**, the conservative-only model on OpenWebText where the pair term carries 0.28 nats, not SR2, where an integrator refinement has idled it (§2). The one GPU prerequisite is **VF0**, F3.1 trained without $V_\varphi$, pre-registered in the protocol (§5.21) and never run: without it no field arm can be scored, because removal at inference measures reliance, not value.

## 1. What a mean-field PARFLM is

Term by term, on the two-layer conservative-only model ($d = 384$, five context channels, $K = 8$ wells, a routed pair force over the $k = 16$ best earlier tokens with four heads; the constants of the ladder's Cell 0):

| coupling | what it is | mean field already? |
| --- | --- | --- |
| context channels ξ → V_θ | five channels whose weights are a softmax over an EMA distance term plus a content term (query of the token against keys of the past states); V_θ's centres, shapes and weights are linear in them | no, as trained: content-routed (§1.1); with the content term off they are EMAs, a field of density moments whose cost per token is independent of T |
| routed pair force | Σ over 16 earlier tokens, chosen by content, of ∇V_φ(h_t, h_j) | no: the only particle term and the only content-addressed lookup |
| per-token stiff basis | the d × P generator and its eigendecomposition, per token | not an interaction: a numerical choice (Probe B) |
| Fock registers | exclusion slots with creation and destruction gates | outside this note |

**The field.** With $\rho_t$ a density over the state space built from the preceding states,

$$
f_t^{\mathrm{field}} = -\nabla_{h_t} \int V_\varphi(h_t, h') \rho_t(dh'), \qquad
\rho_t = \sum_{m=1}^{M} \pi_{m,t} \mathcal N(\mu_{m,t}, \Sigma_{m,t}),
$$

evaluated at the component means (the monopole approximation; sigma points are the next order). The mixture is carried by content-gated EMA channels: fixed prototypes $c_m$, pointwise responsibilities $r_{s,m} = \underset{m}{\mathrm{softmax}}(-\lVert h_s - c_m \rVert^2 / \tau)$, and

$$
\pi_{m,t} = \sum_{s \lt t} W_{ts} r_{s,m}, \qquad
\mu_{m,t} = \frac{\sum_{s \lt t} W_{ts} r_{s,m} h_s}{\pi_{m,t}},
$$

two matrix products with the same causal weight matrix $W$ the existing channels use. The sum excludes $s = t$, so no token contributes to its own field and the force stays the gradient of a potential in $h_t$ (Macroscopic §9, conservativity under a live mixture).

**One scale to fix.** The trained routed term is a sum over 16 neighbours with unit gates; the field is an average with $\sum_m \pi_{m,t} \le 1$. A per-layer scalar $\lambda_\ell$ therefore multiplies the field force in every comparison and substitution below, fitted by least squares to the model's own force on the probe windows and reported. The cosine metrics are scale-free; the error and loss metrics use $\lambda_\ell$.

**What is lost by construction.** Two past tokens that fall in the same component are indistinguishable to the field. If the routed force's value lies in retrieving one specific earlier token, no mixture at any $M$ reproduces it; Probe A's copy-split reading (A4) tests for exactly that, and the templates line owns that job if it is confirmed (Macroscopic §7.1).

### 1.1 Correction (2026-11-17): the context channels are content-routed

Sections 0 and 1 first described $\xi$ as five EMAs with fixed decays. That is the model with `XI_CONTENT_ROUTE = False`. Every ladder model runs with it on ("Alternative E"; protocol, configuration table), F3.1 and SR2 included. Each channel's weight on an earlier token $s$ is then

$$
w_{ts}^{(c)} = \underset{s \le t}{\mathrm{softmax}} \left( (t - s) \ln \alpha_c + \frac{q_c(h_t) \cdot k_c(h_s)}{\sqrt{48}} \right),
$$

with learned query and key maps of width 48 (`multixi/model_multixi.py`, `MultiChannelXi.forward`). This is five-head attention with a distance bias and the raw states as values. Three consequences for this programme:

- **The context field is not yet a mean field.** It is a content-addressed lookup like the pair force, with a $(K, T, T)$ weight tensor per layer in prefill and a cache of every earlier key and state at decoding (about 624 numbers per token and layer). The claim that the model "already" has one mean-field coupling holds for the pair-free, routing-free model only.
- **The content-gated mixture of §1 applies to $\xi$ as well.** The same prototypes $c_m$ and responsibilities $r_{s,m}$, summed by the existing causal EMA weights, give per-channel mixture weights and means as linear recurrences. A content-gated $\xi$ keeps a dependence on content (through which component an earlier state falls in) and has a state of $O(M d)$ per channel. With the field form of $V_\varphi$ it would make every coupling of the model a fixed-size recurrence, so that decoding needs no cache.
- **What it loses** is the same as for the pair term: two earlier tokens in the same component become indistinguishable. Whether the trained routing uses that resolution is measurable in the same way as Probe A. Substitute the content-gated channels for the routed ones in a trained model, at $M \in \lbrace 1, 4, 16 \rbrace$, and read the cost against turning the content term off ($M = 1$ with uniform responsibilities is the plain EMA). No call is proposed yet.

## 2. Where the pair term matters: the base is F3.1, not SR2

The protocol's CD measurements (§5.20, scored 2026-10-11) removed $V_\varphi$ at inference from four trained models on OpenWebText, on the same batches, with the identity patch checked bit-exact:

| arm | what it adds to F3.1 | cost of removing V_φ (nats) | PPL |
| --- | --- | ---: | --- |
| F3.1 | nothing: conservative only, no Fock | **+0.282 ± 0.013** | 56.30 → 74.64 (+33%) |
| PM1 | 64 Poisson-mode wells | **+0.324 ± 0.008** | 53.71 → 74.28 (+38%) |
| G2 | the Fock reverse channel | +0.037 ± 0.008 | +3.7% |
| SR2 | the exact damped stiff-mode flow | +0.0135 ± 0.0010 | +1.4% |

Three facts follow and set the plan. First, the pair mechanism carries a third of the perplexity when it is the model's only content-addressed channel, and the wells do not displace it (CD4 missed and withdrew the earlier "wells took over" reading). Second, what idles it is SR2's integrator refinement, not anything in PARFLM: on SR2 and its descendants (every code arm, the PM pairs) a field cannot be tested, because there is nothing left for it to reproduce; the `mechanism_ablations` `vphi_off` readings on those models are +1.5% to +3.6% on OpenWebText and +2.8% (second order) to +8.5% (first order) on code. Third, 0.28 nats is **reliance**, not value: the trained model leans on the term that much, but a model trained without it may recover part of the loss through the other couplings (the Fock channel's reliance was 1.49 nats against a trained-without gain of 0.084). The protocol's VF factorial (§5.21, pre-registered 2026-10-11) was designed to measure the value and never ran. VF0 is this programme's one GPU prerequisite.

**Correction (2026-11-22).** The second fact is wrong for the code arms without wells. Zeroing the pair force's scale costs SO-CODE +0.19 nats (+21%) and FO-b-CODE +0.68 nats (+98%), against +0.019 (+1.9%) and +0.065 (+6.7%) for their twins with two wells, PM2-CODE and FO-b-PM2-CODE ([productization/Pair_Force_Recall_Split_FO_b_PMX_K16.md](productization/Pair_Force_Recall_Split_FO_b_PMX_K16.md)). On code it is the wells that displace the pair force, not the exact flow. SO-CODE was a Probe A model and the field recovers 92% of its pair force (§5).

So F3.1 is the primary base for every question about $V_\varphi$, and SR2 the secondary base, where the question is the complementary one: with a field in place of the router, does the exact flow still make the pair term redundant, or does the field find work the router could not do?

## 3. Material and rules

**Checkpoints, all on this machine** (`~/Downloads/semsimula_fock_cfc_baoab_<corpus>_<tag>/checkpoints`): F3.1, SR2 (SO-OWT), PM1 and G2 on OpenWebText, built by name through `debug/verify_pm_switch.py`'s `CONFIGS` or by tag; SO-CODE, L4-CODE, L8-CODE (step 18,000) and K2-L8-CODE on code, built from their tags through `debug/code_arm.py`.

**Windows.** OpenWebText models: 16 validation batches of $4 \times 512$ tokens from the harness's `get_batch` with seed 20261009, the batches `mechanism_ablations.py` uses, so every CD and ablation number above is on the same tokens; 64 windows. Code models: 64 of `copy_split.windows()`, 513 validation tokens each, spaced evenly over the set, so the copy-split categories (first, repeat, induction; token class; distance to the last occurrence) are available per token.

**Hooks.** The reference pair force is the model's own `_pair_force_analytic(h_in, layer_idx, h_src_live)`, called at every layer step; the field force replaces it through the same method patch `mechanism_ablations.py` uses for `vphi_off`, so the rest of the step (wells, registers, integrator) is untouched. The autograd force of the field potential is compared with the hand-assembled one on the first batch as a bit-check before any metric is read.

**Budget.** The inference readings on these models peak at 6–7 GB on $4 \times 512$ batches, and `mechanism_ablations` ran nine variants on four batches in under four minutes, so a probe of about twelve forwards over 64 windows is about an hour per model on 16 threads. Probe C runs on bank parameters alone, in minutes. Probe E's coarse-grained forwards are about an hour per model. One job at a time on the 64 GB machine; peak RSS logged per job.

**Rules carried over.** Every loss comparison is paired on the same windows and reported with its per-window standard error; differences under twice that error are unresolved. Calls are confirmed before a probe runs. Each probe writes a text output and a JSON beside the model's results folder (§8). Thresholds marked "proposed" are mine.

## 4. The sequence

| stage | what | where | cost | gate it feeds |
| --- | --- | --- | --- | --- |
| 1 | Probe C: the self-consistent bank | L8-CODE, SO-CODE, F3.1, SR2 | minutes | theory level |
| 2 | Probe A: field V_φ fidelity | **F3.1 first**, then PM1, SR2, SO-CODE, L8-CODE | ~1 h per model | the field arm |
| 3 | Probe D: state clustering with depth | L8-CODE, SO-CODE, F3.1 | minutes per model | M per layer; Probe E |
| 4 | Probe E: does a layer act through collective variables? | L8-CODE, SO-CODE, F3.1 | ~1 h per model | theory level; the hybrid design |
| 5 | Probe B: the shared stiff basis | SO-CODE, L8-CODE | hours | a memory-and-speed switch |
| 6a | VF0: F3.1 trained without V_φ (GPU prerequisite) | Colab, one A100 | ~14 h | scoring the field arm |
| 6b | FA-OWT: the field arm, paired against F3.1 | Colab, one A100 | ~14 h | the model level |

Stages 1–5 are independent of one another and of the GPU; 2 runs first because it decides whether 6b exists. Before 6b, a ten-minute timing of a training step with `PAIR_POTENTIAL = 'none'` (Macroscopic §7, item 1) bounds the engineering payoff; it can share VF0's session.

## 5. The probes, pre-registered

### Probe C: the self-consistent bank (Stage 1)

As specified in the mean-field note §8, with two amendments from its review (2026-11-10). The map per layer is $x \mapsto \Phi_\ell(x) = \sum_k w_k(x + c_\ell) \mu_k(x + c_\ell)$ on the context part of the bank input, with $c_\ell$ the depth code; damped iteration from 256 random starts per layer (norms spread over 0.25–64) and from the trained tokens' own inputs; free-norm and constrained-norm versions; per fixed point its $n_{\mathrm{eff}}$, 90%-weight set, norm and the spectral radius of the Jacobian. The amendments: (i) the "critical" norm $r_c$ is defined as the **loss of stability of the disordered fixed point** (the spinodal), because for $K \ge 3$ the mean-field Potts transition is first order and the disordered point stays locally stable past the free-energy crossing (for $K = 8$ and no bias, the crossing is at $\beta J \approx 4.5$ against the spinodal at 8); (ii) a **coexistence window** with both branches stable is expected and reported, not read as a failure of the closure. Models: L8-CODE (step 18,000), SO-CODE, and now F3.1 and SR2 on OpenWebText.

| id | call | proposed at |
| --- | --- | ---: |
| C1 | L8-CODE has a spinodal r_c with 1.85 < r_c < 22, beyond which no stable fixed point has n_eff > 4 | 60% |
| C2 | at the deep layers' norms the stable fixed points' 90%-weight sets are subsets of {5, 6} in at least 6 of layers 1–7 | 65% |
| C3 | SO-CODE's layer-0 fixed points at r = 2 have n_eff ≥ 2.5 and its layer-1 fixed points at r = 32 have n_eff ≤ 1.3 | 70% |
| C4 | the constrained-norm fixed-point curve lies below the sweep curve at every r (self-consistency sharpens the selection) | 60% |
| C5 | a coexistence window exists in L8-CODE: a range of r where a disordered and an ordered fixed point are both stable | 55% |

### Probe A: field $V_\varphi$ fidelity (Stage 2)

**Models and order.** F3.1 (layers 0 and 1) first; then PM1 (where the term costs 0.324 and the trained Poisson modes are a second candidate set of prototypes), SR2, SO-CODE, L8-CODE (layers 0, 1 and 7).

**The field, per layer.** Prototypes $c_m$ by k-means on the layer's input states over the 64 windows (a probe of representability, not of training), $M \in \lbrace 1, 2, 4, 8, 16, 32 \rbrace$; $M = 1$ is the floor, where the mixture is a single mean and the field is in effect $\xi$ itself. The responsibility temperature $\tau$ set so that the median $n_{\mathrm{eff}}$ of $r_{s,m}$ is 1.5. Weights and means by the EMA channel with decay 0.995 and, as the second variant, by the uniform causal mean. On PM1 a third variant uses the trained Poisson-mode centres as prototypes. The force is the monopole sum over components, times the fitted $\lambda_\ell$.

**Metrics, per token and layer.** The cosine between the field force and the model's force; the relative error after $\lambda_\ell$; medians and 10th–90th percentiles over tokens. Then the **substitution loss**: the validation loss on the windows with the field force in every layer, against the trained loss and against the loss with $V_\varphi$ off (the two ends of the scale, both measured on the same windows). The fraction recovered is $\rho_M = 1 - c_{\mathrm{field}}(M) / c_{\mathrm{off}}$ with $c$ the paired cost in nats. On the code models, the substitution cost split by copy-split category and by distance to the last occurrence.

| id | call | proposed at |
| --- | --- | ---: |
| A0 | on F3.1 the M = 1 field recovers less than a quarter of the removal cost (ρ_1 < 0.25): content gating is needed, ξ alone is not the field | 70% |
| A1 | on F3.1 at M = 8 the field recovers at least three quarters (ρ_8 ≥ 0.75) | 50% |
| A2 | on L8-CODE's layers 1 and 7 the median force cosine at M = 8 is at least 0.9 | 60% |
| A3 | on every model layer 0's median cosine is below the deeper layers' at every M (the hot layer) | 75% |
| A4 | on SO-CODE the field's substitution cost is concentrated in the induction category: its share of the cost exceeds its share of the tokens by a factor of at least 2 | 55% |
| A5 | on SR2 the substitution cost at M = 8 is within twice its paired standard error of zero (there is nothing left for the field to reproduce) | 75% |

**Confirmed 2026-11-21.** The author confirmed A1 to A5 as written, at the probabilities above; they are binding from that date. A0 is proposed, not confirmed by the author. Context at confirmation: SR2-NP (SR2 trained without the pair force) had settled at 58.51 against SR2's 58.32, so on SR2 the pair force is worth 0.2 to 0.3% in value while removing it from the trained model costs 1.8%. A5 tests the second of these, the trained model's reliance, and is scored as written.

**Reading.** A1 and A2 hit: the field arm (Stage 6b) is built at the smallest $M$ that reaches $\rho_M \ge 0.75$ on F3.1. A1 misses but $\rho_M$ still rises at $M = 32$: raise $M$ or add sigma points before deciding; the kernel's scale may be narrower than the components. A2 misses: the field is not the right parametrisation of this kernel at any $M$; stop the model level, keep the theory level. A4 hit: the part the field cannot carry is retrieval of a specific earlier token, which is the templates line's job; the field arm then is a **hybrid** (routed force in layer 0, field above), or waits.

**Result, 2026-11-22 (F3.1, FO-b-PMX-K16, SO-CODE, SR2, L8-CODE).** Full write-up: [mean_field/Probe_A_Field_Pair_Force.md](mean_field/Probe_A_Field_Pair_Force.md). Fraction recovered, causal field (EMA) at $M = 1$ and $M = 8$, with a context-free control added after the smoke test (the k-means centres at fixed weights, the same for every token):

| model | pair force off (nats) | EMA, M = 1 | EMA, M = 8 | context-free, M = 8 |
| --- | ---: | ---: | ---: | ---: |
| F3.1 | +0.2665 | 0.941 | 0.962 | 0.964 |
| SO-CODE | +0.2007 | 0.901 | 0.920 | 0.913 |
| FO-b-PMX-K16 | +0.0245 | 0.663 | 0.738 | 0.681 |
| SR2 | +0.0148 | 0.487 | 0.582 | 0.573 |
| L8-CODE | +0.0620 | 0.583 | 0.760 | 0.756 |

- **A0 (proposed only), MISS:** $\rho_1 = 0.941$.
- **A1, HIT:** $\rho_8 = 0.962$.
- **A2, MISS at every M:** L8-CODE's layers 1 and 7 at 0.593 and 0.774 at $M = 8$, at most 0.650 and 0.842 at $M = 32$. Layer 0 is at 0.956; the deep layers are not field-like in direction, though the field keeps 76% of the loss.
- **A3, MISS:** layer 0 has the lower cosine on F3.1 and SO-CODE, as called, but the higher one on FO-b-PMX-K16 and L8-CODE at every $M$; SR2's layer 1 is not measurable (scale 0.006).
- **A4, MISS on its letter:** factor 1.94 against a ceiling of 2.01, since induction is 49.8% of the code tokens; 97% of the remaining cost falls on induction tokens. The threshold was mis-set; scored as written.
- **A5, MISS:** +0.0062 (0.0007) nats at $M = 8$.
- **Reading.** On F3.1 a context-free landscape with one centre does as well as any field: the pair force acts as a function of the token's own state. At depth on code the field is not a faithful copy. The A2 gate fires; the write-up (§7) recommends honouring it for the content-gated field, running VF0 under the protocol's own §5.21 entry, and registering a context-free landscape arm on F3.1 as post hoc only if VF0 shows value. For the author to rule.

### Probe D: state clustering with depth (Stage 3)

On the probe windows, per layer, k-means on the layer's input states $h_t$ with $k = 1, \dots, 16$ and the silhouette score (cosine metric, since the states live on the sphere); the best-silhouette cluster count against depth; the distance from each cluster centre to the nearest well centre $\mu_k(x_t)$ of the tokens in the cluster, against the distance between random pairs of centres. Also reported: the agreement between cluster membership and the well carrying most of the token's force (adjusted mutual information), which links the clusters to Probe C's phases.

| id | call | proposed at |
| --- | --- | ---: |
| D1 | in L8-CODE the best-silhouette count is lower at layer 7 than at layer 0 | 70% |
| D2 | in layers 1–7 of L8-CODE the cluster centres lie closer to the force-carrying wells' centres than random centre pairs by a factor of at least 2 | 55% |
| D3 | in L8-CODE's deep layers the adjusted mutual information between cluster and dominant well is at least 0.3; at layer 0 below 0.1 | 50% |

### Probe E: does a layer act through collective variables? (Stage 4)

The direct test of a macroscopic description: if the deep layers' dynamics is governed by a few collective variables, a layer's output should depend on a token's input state only through which cluster the state belongs to, and the well occupation should move between layers by a nearly deterministic rule. Two measurements, both paired on the same windows.

- **State coarse-graining.** For layer $\ell$ and cluster count $M$ (Probe D's partition, $M \in \lbrace 2, 4, 8, 16 \rbrace$), replace each token's input position $h_t$ by its cluster centroid, keep its velocity and context, run the layer and the rest of the model, and read the loss. The cost $c_{\mathrm{cg}}(\ell, M)$ in nats, paired. The control is the same replacement with a random partition of the same sizes.
- **The occupation rate law.** Per token, the well carrying most of its force at the input of layer $\ell$ and of layer $\ell + 1$; the transition matrix between layers and its conditional entropy $H(k_{\ell+1} \mid k_\ell)$ in nats, with $k_\ell$ the index of that well at layer $\ell$. A macroscopic law is a transition matrix close to a permutation (low conditional entropy); the mixing at layer 0 is expected to be the largest.

| id | call | proposed at |
| --- | --- | ---: |
| E1 | in L8-CODE, coarse-graining any one of layers 1–7 to M = 8 centroids costs at most 0.05 nats, and at most a fifth of the random-partition control's cost | 50% |
| E2 | coarse-graining layer 0 to M = 8 costs at least five times the median cost of layers 1–7 | 70% |
| E3 | in L8-CODE the conditional entropy of the well transition is below 0.3 nats for every pair of consecutive layers among 1–7, and above 0.6 nats for 0 → 1 | 55% |

**Reading.** E1 and E3 hit: the deep layers of a two-well model admit a closed macroscopic description in which the layer map acts on the density's parameters, which is the claim §4 of the macroscopic note argues from circumstantial evidence. E2 hit alongside: the hybrid design (microscopic layer 0, macroscopic above) is the one to build. E1 misses at every $M$: the layer uses the exact position, and "few slow variables" was a statement about the force shares, not about the dynamics; the field model may still work (that is Probe A's question), but the theory level stops at the bank.

### Probe B: the shared stiff basis (Stage 5)

As specified in Macroscopic §8: per token and layer the top $P$ left singular subspace of the generator, the pooled subspace over blocks of $b \in \lbrace 8, 64, 512 \rbrace$ consecutive tokens, the principal angles and energy fractions, and the loss with the block basis substituted in every layer. Calls B1–B3 as written there (energy fraction ≥ 0.9 at $b = 64$ in layers 1–7 of L8-CODE, 55%; substitution penalty ≤ 0.02 nats at $b = 64$, 50%; layer 0 has the largest angles, 70%). Independent of the field; included because it is the other per-token term.

**Set up 2026-11-17 (the author asked for it, after the CPU benchmark put the per-token decompositions at 18 to 21% of SR2's inference time; calls PB1 to PB3 confirmed by the author on 2026-11-17, before the run; queued after the code recall probe and the code per-well reading).** Two amendments. (i) The models are SR2 on OpenWebText (the benchmarked model) and SO-CODE; L8-CODE can follow. (ii) A basis pooled over a block reads later tokens of that block, so it is not causal. The probe therefore also measures a **causal** basis, from the previous block, which a decoder could use; the first block keeps its own per-token modes. Each token's modes are then the exact eigenpairs of its $L_t$ restricted to the block subspace, a $q \times q$ problem. Readings: energy fraction and largest principal angle per layer; the loss with the block basis on layer 0, on layer 1 and on both, at $b \in \lbrace 8, 64, 512 \rbrace$, paired by window; and the time spent in the mode computation. Script `debug/shared_stiff_basis_probe.py`; outputs `results/shared_basis/`; a check on K4-R6 first.

| id | call | confirmed at |
| --- | --- | ---: |
| PB1 | on SR2 at b = 64 (pooled), the median energy fraction is at least 0.9 at layer 1 and below 0.9 at layer 0 | 55% |
| PB2 | on SR2 the causal basis at b = 64 on layer 1 only costs at most 0.01 nats | 55% |
| PB3 | on SR2 the causal basis at b = 64 on both layers costs at most 0.02 nats | 25% |

**Result, 2026-11-17: PB1 hits; PB2 and PB3 miss. A causal shared stiff basis costs too much on either layer, so the per-token decompositions stay.** Outputs `notebooks/conservative_arch/scaleup/results/shared_basis/shared_basis_{check_k4r6,sr2_owt,so_code}.{txt,json}`; no mode-solver retries. Loss change in nats with the block basis, against the trained model (paired standard error 0.002 to 0.04):

| model, layer | pooled b = 8 | pooled b = 64 | causal b = 8 | causal b = 64 |
| --- | ---: | ---: | ---: | ---: |
| SR2, layer 1 only | +0.0001 | +0.005 | +0.040 | **+0.032** |
| SR2, layer 0 only | +0.108 | +0.392 | +0.760 | +0.574 |
| SR2, both layers | +0.115 | +0.433 | +0.843 | **+0.642** |
| SO-CODE, layer 1 only | +0.0004 | +0.016 | +0.074 | +0.074 |
| SO-CODE, both layers | +0.230 | +0.676 | +1.024 | +0.854 |

- **PB1 (SR2, pooled b = 64: energy at least 90% at layer 1, below 90% at layer 0), 55%: HIT.** 92.2% and 56.1%. SO-CODE: 91.8% and 67.0%.
- **PB2 (SR2, causal b = 64 on layer 1 only costs at most 0.01 nats), 55%: MISS.** +0.032.
- **PB3 (SR2, causal b = 64 on both layers at most 0.02 nats), 25%: MISS.** +0.642.
- **Reading.** Layer 1's stiff subspace is shared within a block: a basis pooled over the block's own tokens costs nothing at b = 8. But that basis reads later tokens of the block, so a causal model cannot use it. The basis a decoder could use, from the previous block, costs 0.03 to 0.07 nats on layer 1 alone and far more on layer 0, whose stiff directions change from token to token (energy fractions of 38% to 54%). Sharing the stiff basis is not a lever; the remaining one on this term is a cheaper per-token solver.
- **Two readings that do not mean what they seem.** The principal angles are about 90° everywhere, because each token's 16 retained modes include inert, near-null ones; the angle reading is uninformative and the energy fraction is the measure. The causal rows at b = 512 equal the trained model, because in 512-token windows the first block keeps its own modes. The mode-time column includes the readings' own computations on the base run, so it is not a timing of the method.

### VF0: F3.1 trained without $V_\varphi$ (Stage 6a, GPU prerequisite)

Exactly the protocol's §5.21 arm: F3.1's Cell 0 with `PAIR_POTENTIAL = 'none'` (tag `nophi`), seed 0, WSD on 32,500 steps, one A100, about 14 h; the switch check (`debug/verify_pair_potential_none.py`, verified 2026-10-01 for the multi-ξ and Fock builds) rerun on F3.1's cell before launch. Readout $v_0 = \ln(\mathrm{PPL}_{\mathrm{VF0}} / 57.76)$, and the protocol's call VF-a ($v_0 \ge 0.03$, 70%) stands. Its role here: $v_0$ is the **value** of the pair term, the quantity the field arm is scored against. VF1 and VF2 keep their place in the protocol and are not prerequisites.

### FA-OWT: the field arm (Stage 6b, conditional on A1 and A2)

F3.1's Cell 0 with the router and the gathered pair force replaced by the content-gated field at the $M$ Probe A selects: learned prototypes (initialised from Probe A's k-means), learned $\tau$ and $\lambda_\ell$, the 0.995 decay, the monopole force, sources detached as now. Paired from step 0 against F3.1 on the same batches; the same schedule as VF0. Nothing with a $(T, T, \cdot)$ shape remains in the layer. If A4 hit, the arm is the hybrid instead, with the routed force kept in layer 0 only. Readout $v_{\mathrm{field}} = \ln(\mathrm{PPL}\_{\mathrm{VF0}} / \mathrm{PPL}\_{\mathrm{FA}})$, the part of the pair term's value the field earns back.

| id | call | proposed at |
| --- | --- | ---: |
| FA1 | the field earns back at least half of the value: v_field ≥ 0.5 v_0 | 50% |
| FA2 | the training step is at least 20% faster than F3.1's at the same batch (bounded above by the `PAIR_POTENTIAL = 'none'` timing) | 60% |
| FA3 | the field arm's per-layer flow (FLOW-R, same protocol) converges at least as well as F3.1's | 55% |

**Then, if FA1 hits:** FA-CODE on the SR2 code base is not informative about the field (A5) but is the cheapest test of whether the field composes with the exact flow and depth; it follows, not precedes, FA-OWT.

## 6. Decision gates

| finding | consequence |
| --- | --- |
| A2 misses at every M | the field is the wrong parametrisation of V_φ; stop Stages 6; keep Probes C, D, E as the theory of PARFLM |
| A1 misses, A2 hits | the mixture resolution loses information on OpenWebText; raise M or add sigma points, re-score, then decide |
| A4 hits | the routed force does retrieval; the field arm is the hybrid; the templates line gets the retrieval mandate |
| A5 misses | the exact flow did not idle V_φ after all on these windows; SR2 rejoins the primary bases |
| E1 and E3 hit, E2 hit | the deep layers have a closed macroscopic law and layer 0 does not: the hybrid is the principled design, not a fallback |
| VF0 gives v_0 < 0.03 | the pair term's value is small despite its reliance; the field arm is scored on speed (FA2), not on perplexity |
| FA1 hits | the mean-field PARFLM exists; the deep template arms of the templates line become trainable on it (Macroscopic §7.1) |

## 7. What the programme does not claim

- **Representability is not trainability.** Probe A bounds the loss of expressing a trained force as a field; a model trained with the field may use it differently. Only FA-OWT measures the trained loss, and only against VF0 does it measure value.
- **The hot layer.** Every measurement so far singles out layer 0: seven wells, a small input norm, the largest refinement penalty. The probes report it separately, and a hybrid is a legitimate outcome, not a failure.
- **Order and identity.** A field is a bounded summary; it cannot lift the expressivity class and erases the identity of individual past tokens. The particle store of the templates line keeps that job (Macroscopic §7.1), and nothing here changes that division.
- **Aliasing with refinement.** Content-gated channels depend on position-weighted moments and so on the step count through the decays; the refinement note's aliasing analysis applies to them as to the existing channels. FA3 is the check.

## 8. Reporting

Scripts in `notebooks/conservative_arch/scaleup/debug/` (planned names: `selfconsistent_bank.py`, `field_vphi.py`, `state_clusters.py`, `coarse_grain_layer.py`; Probe B as in the macroscopic note), each writing `<probe>_<model>_output.txt` and `.json` into the model's results folder under `notebooks/conservative_arch/scaleup/results/`, with the OUT path given absolute (the harness changes directory). Every probe records its checkpoint file, window seed, thread count and peak RSS. Results are scored in this note's probe sections, each marked HIT or MISS against the confirmed call, and a protocol entry points here. The two GPU arms get protocol entries of their own, VF0 under §5.21 and FA-OWT as a new pre-registration, before they launch.
