# Admissible Regions for Template Search

**Where in semantic space, and when in training, a search for the patterns that become semantic templates may start**

Companion note to [*Implementing Semantic Templates: A Path*](../Implementing_Semantic_Templates_A_Path.md) (the Tier 1 decision of §5 and the staging of §7) and to [*Semantic Templates as Doi–Peliti Reactions*](../Semantic_Templates_as_Doi_Peliti_Reactions.md). Book context: Definition 8 (*Semantic template*, §2.6) binds a Pattern Matching Structure to a region $R_{\mathrm{pm}} \subset \Sigma$ that it watches for signatures along trajectories. This note is about which regions can be watched at all while the model is still training. Written 2026-11-24 from the author's observation that opens §1; the measurement of §7 runs on the CPU on SR2-NP's three checkpoints.

---

## Contents

1. [The problem](#1-the-problem)
2. [Setting and the three times](#2-setting-and-the-three-times)
3. [The well frame, and what survives a moving coordinate system](#3-the-well-frame-and-what-survives-a-moving-coordinate-system)
4. [The context spread of a well](#4-the-context-spread-of-a-well)
5. [Training drift, measured without coordinates](#5-training-drift-measured-without-coordinates)
6. [Admissible regions, and what admissibility buys](#6-admissible-regions-and-what-admissibility-buys)
7. [The measurement: a settling map of SR2-NP](#7-the-measurement-a-settling-map-of-sr2-np)
8. [Results](#8-results)
9. [What this does not settle](#9-what-this-does-not-settle)
10. [Reproduce](#10-reproduce)

---

## 1. The problem

The author's observation (2026-11-24): *during training everything moves, including the wells. We need to identify regions in semantic space and time in which we can start looking for patterns on which semantic templates will be constructed; not all regions in semantic space and time are appropriate for this task.*

A template's Pattern Matching Structure watches a region for a signature along trajectories (book, Definition 8). A signature is only worth recording if the trajectories through the region repeat: the same configuration of states should give the same local motion on the next batch, at the next checkpoint, and in the next context. Three things break that repetition in a model that is still training, and they are three different motions:

- **the wells move**: the bank's parameters change with every optimizer step, so the landscape the trajectories fall into is not the landscape of the last checkpoint;
- **the coordinates move**: the states themselves are re-parametrised as the embedding and the layers train, so a pattern recorded as a list of positions in $\mathbb{R}^d$ at one step is written in a frame that no longer exists at the next;
- **the context moves the wells**: in the trained bank every well's centre, width and weight are functions of the context channels, so even at fixed parameters the well a token falls into is somewhere else in every context.

Each of the three has a precise form, and each gives one clause of the admissibility criterion of §6. The clarity the formulation brings is that the three are not one vague "everything moves" but three error terms in one bound (Proposition 3), one of which (the coordinates) can be removed exactly rather than tolerated.

---

## 2. Setting and the three times

**The model.** The Gen 3 conservative core (SR2 family): $L$ layers, hidden state $h \in \mathbb{R}^d$ per token, $d = 384$. At layer $\ell$ and position $t$ the context channels $\xi_t \in \mathbb{R}^{nd}$ are the $n = 5$ exponential moving averages of the earlier states with decays $\alpha_c \in \lbrace 0.50, 0.75, 0.95, 0.99, 0.995 \rbrace$, concatenated. Only $\xi$ crosses positions.

**The bank.** The anisotropic Gaussian bank $V_\theta$ (code: `AnisotropicMixtureGaussianVTheta`, depth-conditioned and joint over the channels) has $K$ wells. Well $k$ at context $\xi$ is the triple

$$
\mu_k(\xi) = W_k(\xi + c_\ell) + b_k \in \mathbb{R}^d, \qquad
P_k(\xi) = \mathrm{diag}(a_k(\xi)) + B_k(\xi) B_k(\xi)^{\top} \in \mathbb{R}^{d \times d}, \qquad
w_k(\xi) > 0,
$$

a centre affine in the shifted channels ($c_\ell$ the layer's depth code), a symmetric positive definite precision of diagonal-plus-low-rank form, of rank $r = 4$, and a weight (a softmax over the wells times the weight scale). All three are functions of $\xi$ through the bank's projections; the parameters of those projections are $\theta$. The potential at state $h$ in context $\xi$ is

$$
V_\theta(h, \xi) = -\sum_{k=1}^{K} w_k(\xi)  e^{-q_k(h,\xi)/2}, \qquad
q_k(h, \xi) = (h - \mu_k(\xi))^{\top} P_k(\xi) (h - \mu_k(\xi)),
$$

a sum of attractive Gaussian wells, and the force on $h$ is $-\nabla_h V_\theta$. The force weight of well $k$ at a token is its term of the sum, $w_k e^{-q_k/2}$, written $g_k$; this is the quantity the well probe (`vtheta_well_probe.py`) reports per well.

**The three times.** The word *time* has three senses here and the formulation keeps them apart:

| symbol | sense | what moves with it |
| --- | --- | --- |
| $\ell$ | depth, the layer index: the integrator's time | the state $h$ along its trajectory |
| $t$ | position in the sequence | the context $\xi_t$, and with it the wells' realised centres |
| $\tau$ | the optimizer step, training time | the parameters $\theta_\tau$, and with them every well and every coordinate |

A region is therefore indexed by all three: a set of states at a given layer, over a range of positions, at a given training step. "Not all regions in semantic space and time are appropriate" is a statement about $(\ell, t, \tau)$ jointly.

---

## 3. The well frame, and what survives a moving coordinate system

**Definition 1 (well frame).** For well $k$ at context $\xi$, the *well coordinate* of a state $h$ is

$$
z_k = P_k(\xi)^{1/2} (h - \mu_k(\xi)) \in \mathbb{R}^d, \qquad r_k = \lVert z_k \rVert = \sqrt{q_k(h, \xi)},
$$

the displacement from the well's centre, whitened by the well's precision. $r_k$ is the *well radius* of the state: its distance from the centre in units of the well's width along that direction. The *basin* of radius $\rho$ is $R_k(\rho) = \lbrace h : q_k(h, \xi) \le \rho^2 \rbrace$, an ellipsoid whose axes are the well's widths. The *stiff coordinates* are the low-rank part of the same displacement, $r$ numbers,

$$
u_k = B_k(\xi)^{\top}(h - \mu_k(\xi)) \in \mathbb{R}^r,
$$

the components along the directions in which the well is narrowest (these are the directions the exact damped propagator treats as stiff modes).

**Definition 2 (dominance).** The *force weight* of well $k$ at a token, its *dominance share* and the *dominant well* are

$$
g_k = w_k e^{-q_k/2}, \qquad \pi_k = \frac{g_k}{\sum_{j=1}^{K} g_j}, \qquad k^{\ast} = \arg\max_k \pi_k.
$$

**Proposition 1 (frame invariance).** Let the state space be re-parametrised by an invertible affine map $h \mapsto Ah + c$, applied jointly to the states and to the wells, so that $\mu_k \mapsto A\mu_k + c$ and $P_k \mapsto A^{-\top} P_k A^{-1}$. Then $q_k$, $g_k$, $\pi_k$, $r_k$ and $u_k$ are unchanged, and $z_k \mapsto Q z_k$ for an orthogonal $Q$.

*Proof.* Under the map, $h - \mu_k \mapsto A(h - \mu_k)$, so

$$
q_k \mapsto (h-\mu_k)^{\top} A^{\top} A^{-\top} P_k A^{-1} A (h - \mu_k) = q_k,
$$

and $g_k$ and $\pi_k$, being functions of $q_k$ and $w_k$, are unchanged. The low-rank factor transforms as $B_k \mapsto A^{-\top} B_k$, so that $B_k B_k^{\top}$ transforms as $P_k$ does; hence

$$
u_k = B_k^{\top}(h - \mu_k) \mapsto B_k^{\top} A^{-1} A (h - \mu_k) = u_k.
$$

For $z_k$: with $P' = A^{-\top} P A^{-1}$, $z' = P'^{1/2} A (h - \mu) = Q z$ with $Q = P'^{1/2} A P^{-1/2}$, and $Q^{\top} Q = P^{-1/2} A^{\top} P' A P^{-1/2} = I$. $\square$

What the proposition says for the second motion of §1: if training re-parametrises the states by an affine map and carries the wells along with it, every quantity above is unchanged, and a pattern written in terms of well radii, dominance shares, stiff coordinates and inner products of well coordinates survives the re-parametrisation exactly. Training is of course not an affine map of the state space; but whatever it is, the part of it that *is* a change of frame is invisible in the well frame, and the part that is not shows up as the training drift of §5, measured without reference to any coordinates. The practical rule is the one anticipated in the plan note's §4.3: **a template is anchored to a well and written in that well's frame**, not in the coordinates of $\mathbb{R}^d$.

---

## 4. The context spread of a well

At fixed parameters the well still moves, because its centre is a function of the context. The question is not how far it moves in $\mathbb{R}^d$ but how far it moves *in units of its own width*: a centre that shifts by a large Euclidean distance along a direction in which the well is very wide has not moved, as far as any token inside it can tell.

**Definition 3 (context spread).** Over the tokens of a corpus sample at layer $\ell$ and step $\tau$, let the mean realised centre of well $k$ be its average over the tokens,

$$
\bar{\mu}_k = \mathbb{E}_t \big[ \mu_k(\xi_t) \big].
$$

The *context spread* of the well is

$$
s_k^2 = \mathrm{median}_t \big[(\mu_k(\xi_t) - \bar{\mu}_k)^{\top} P_k(\xi_t) (\mu_k(\xi_t) - \bar{\mu}_k)\big],
$$

the typical displacement of the realised centre from the mean centre, measured in the well's own metric at that token. Its Euclidean counterpart, relative to the length of the mean centre,

$$
s_{k,\mathrm{eu}}^2 = \frac{\mathrm{median}_t \lVert \mu_k(\xi_t) - \bar{\mu}_k \rVert^2}{\lVert \bar{\mu}_k \rVert^2},
$$

is what the well probe's centre split sees.

**Proposition 2 (closed form).** When the precision is context-free, $P_k(\xi) \equiv P_k$ (as in twin PB's `a_context_free` with $r = 0$, or as an approximation with the mean precision), the mean-square version of the spread is a quadratic form in the channel covariance $\Sigma_\xi = \mathrm{Cov}[\xi_t]$ (over the tokens):

$$
\mathbb{E}_t\big[(\mu_k - \bar{\mu}_k)^{\top} P_k (\mu_k - \bar{\mu}_k)\big] = \mathrm{tr}\big(P_k W_k \Sigma_\xi W_k^{\top}\big), \qquad
\mathbb{E}_t \lVert \mu_k - \bar{\mu}_k \rVert^2 = \mathrm{tr}\big(W_k \Sigma_\xi W_k^{\top}\big).
$$

*Proof.* Since the centre is affine in the context, $\mu_k - \bar{\mu_k}$ equals $W_k(\xi_t - \bar{\xi})$; take the expectation of the quadratic form. $\square$

The two traces differ by the factor $P_k$ inside, and that is the whole point: the Euclidean spread can be large while the metric spread is small, exactly when the covariance of the centre (the matrix inside the second trace) is concentrated on the soft directions of $P_k$, those with small eigenvalues. In SR2-NP the well probe already shows the symptom without the diagnosis: at layer 0 the channel term of the centre has norm 10.6 against a mean centre of 8.6, so the context moves the centre by more than the centre's own length, and yet the dominant well's occupancy $e^{-q/2}$ at the token is 0.98 (`well_probe_sr2np_owt.txt`). Either the tokens move with the centre, or the centre moves along directions the well does not resolve. Definition 3 separates the two; §7 measures it.

---

## 5. Training drift, measured without coordinates

Between two checkpoints $\tau \lt \tau'$ the same validation tokens are run through both models. Every comparison below uses only the frame-free quantities of Proposition 1, so that a change of frame between the checkpoints contributes nothing.

**Definition 4 (drift measures).** For well $k$ at layer $\ell$, let $S_k^{\tau}$ be the set of tokens the well dominates at step $\tau$, those with $k^{\ast} = k$. The *assignment overlap* is the Jaccard index of the two dominated sets,

$$
J_k(\tau, \tau') = \frac{\lvert S_k^{\tau} \cap S_k^{\tau'} \rvert}{\lvert S_k^{\tau} \cup S_k^{\tau'} \rvert}.
$$

The *radial drift* is the 1-D Wasserstein distance between the two radial laws over all tokens, in units of the later median radius,

$$
\Delta_k(\tau, \tau') = \frac{W_1\big(\mathcal{L}(r_k^{\tau}), \mathcal{L}(r_k^{\tau'})\big)}{\mathrm{median}_t \big( r_k^{\tau'} \big)},
$$

with $\mathcal{L}(r)$ the distribution of $r$ over the tokens. The *token-wise rank agreements* are the Spearman correlations over the same tokens: $\rho_r$ of $r_k^{\tau}$ with $r_k^{\tau'}$, and $\rho_u$ of $\lVert u_k^{\tau} \rVert$ with $\lVert u_k^{\tau'} \rVert$.

The first says whether the well still owns the same tokens; the second whether its radial law has the same shape; the third whether the *ordering* of tokens by depth in the well, and by position along its stiff directions, is preserved, which is the finest of the three and the one a signature along a trajectory depends on. A well is *settled* over a window of checkpoints when all three are near their ideal values ($J \to 1$, $\Delta \to 0$, $\rho \to 1$) for every consecutive pair in the window.

---

## 6. Admissible regions, and what admissibility buys

**Definition 5 (position warmth).** Channel $c$ at position $t$ carries the weight $\alpha_c^{t}$ of the missing prefix before the window's start. It is *warm at tolerance* $\varepsilon$ when $\alpha_c^{ t} \le \varepsilon$, i.e. $t \ge t_c(\varepsilon) = \ln \varepsilon / \ln \alpha_c$. At $\varepsilon = 0.1$: $t_c = 4, 8, 45, 229, 459$ for the five decays. The slowest channel is never warm inside a 512-token training window, which is a fact about the model, not about the measurement: a pattern that reads the slowest channel is position-dependent by construction, and a template built on it inherits that dependence.

**Definition 6 (admissible region).** The region of well $k$ at layer $\ell$, over positions $t \ge t_0$, at step $\tau$, is *admissible* at five tolerances, one per clause and written $\varepsilon$ with the clause's name as a subscript, over a window of $W$ checkpoints ending at $\tau$ when

1. **live**: the well dominates a share $\ge \varepsilon_{\mathrm{live}}$ of the tokens;
2. **dominant**: on those tokens the median dominance $\pi_k \ge 1 - \varepsilon_{\mathrm{dom}}$;
3. **context-tame**: the context spread $s_k \le \varepsilon_{\mathrm{ctx}}$;
4. **settled**: for every consecutive pair of checkpoints in the window, $J_k \ge 1 - \varepsilon_{\mathrm{drift}}$, $\Delta_k \le \varepsilon_{\mathrm{drift}}$ and $\rho_r$ and $\rho_u$ both at least $1 - \varepsilon_{\mathrm{drift}}$;
5. **warm**: $t_0$ is at least the largest $t_c(\varepsilon_{\mathrm{warm}})$ over the channels the template reads.

Clauses 1 and 2 are the dead-wells lesson (a well nobody visits, or that never wins, has no trajectories to watch); clause 3 is §4; clause 4 is §5; clause 5 is the position sense of time.

**Proposition 3 (what admissibility buys).** Inside an admissible region, write the local flow in the well coordinate of the dominant well as $\dot{z} = F_0(z) + \Delta(z, \xi, \tau)$, where $F_0$ is the flow of the single frozen well $k$ at its mean realised parameters and $\Delta$ collects everything else: the other wells' force (bounded by $\varepsilon_{\mathrm{dom}}$ through their total weight $1 - \pi_k$), the context's displacement of the well (bounded by $\varepsilon_{\mathrm{ctx}}$ through the Lipschitz constant of the Gaussian force in its centre), and the change of the well between the checkpoints of the window (bounded by $\varepsilon_{\mathrm{drift}}$). If $F_0$ is $\Lambda$-Lipschitz on the basin and $\lVert \Delta \rVert$ is at most $C$ times the sum of the three tolerances $\varepsilon_{\mathrm{dom}}$, $\varepsilon_{\mathrm{ctx}}$ and $\varepsilon_{\mathrm{drift}}$ there, then two trajectories started from the same $z_0$, one under the full flow and one under $F_0$, satisfy for a depth $s$ inside the basin

$$
\lVert z(s) - \hat{z}(s) \rVert \le C (\varepsilon_{\mathrm{dom}} + \varepsilon_{\mathrm{ctx}} + \varepsilon_{\mathrm{drift}}) \frac{e^{\Lambda s} - 1}{\Lambda}.
$$

*Proof sketch.* Grönwall's inequality on $\frac{d}{ds}\lVert z - \hat{z} \rVert \le \Lambda \lVert z - \hat{z} \rVert + \lVert \Delta \rVert$. $\square$

The content is not the bound's constants (the actual step is second order and damped, with the exact propagator along the stiff modes, and $\Lambda$ is set by the well's stiffness) but its structure: **a signature observed along a trajectory in an admissible region is reproducible across contexts, across batches and across the checkpoints of the window to first order in the three tolerances, and the fourth source of irreproducibility, the frame, does not appear because the signature is written in the well frame.** That is the sense in which a region is "appropriate for looking for patterns". Outside an admissible region the bound is void: a well that is still moving ($\varepsilon_{\mathrm{drift}}$ large) or that the context carries about ($\varepsilon_{\mathrm{ctx}}$ large) gives signatures that are artefacts of the checkpoint or the batch.

One consequence for the hierarchy of §5 of the plan note: clause 4 is a condition on statistics over many tokens and several checkpoints, which is what a level 1 matcher sees. "This region has become stationary" is then the first level 1 template, and admission is not an external heuristic but the base case of the construction.

---

## 7. The measurement: a settling map of SR2-NP

**Model.** SR2-NP (SR2 without the pair force; [Stage 3](../productization/Stage3_SR2_Without_Pair_Force.md)), whose only context mixing is the bank itself, so that clause 3 is the whole of the context's effect. Its run folder keeps three checkpoints: step 500 (inside the 5% warmup of the WSD schedule), step 22,500 (1,375 steps into the decay phase, which starts at step 21,125), and the best checkpoint (step 31,000, late in the decay, of 32,500 steps). $K = 8$, $r = 4$, two layers; the trained model uses one well per layer (well 5 carries 77% of the force weight at layer 0 and 100% at layer 1; `well_probe_sr2np_owt.txt`).

**Data.** The same 16 evenly spaced validation windows of 512 tokens from OpenWebText as the well probe (8,192 tokens), run through each checkpoint; the bank's components $\mu_k$, $a_k$, $w_k$ and $B_k$ and the state $h$ are captured at the first evaluation of each layer, and $q_k$, $\pi_k$, $u_k$ and the context spread are computed per token and well. Script: `debug/well_settling_map.py`; outputs in `results/well_settling/settling_sr2np.{txt,json,npz}`.

**Quantities.** Per checkpoint, layer and well: the dominated share, the median $\pi_k$, the median radius $r_k$, the median $\lVert u_k \rVert$, the context spread $s_k$ and its Euclidean counterpart. Per consecutive pair of checkpoints: $J_k$, $\Delta_k$, $\rho_r$, $\rho_u$. The collection is call-independent. The calls below were composed before the results were read and confirmed by the author on 2026-11-24; the run finished while they were being written, a timing the report's §6 records.

**Calls (2026-11-24, before the results).**

- **AR1 (settling is late).** For the dominant well at each layer, the late pair (22,500 to 31,000) is closer to settled than the early pair (500 to 22,500) on all four measures: larger $J$, smaller $\Delta$, larger $\rho_r$ and $\rho_u$.
- **AR2 (the context moves the layer 0 well along soft directions).** At the best checkpoint, the dominant well's context spread in the well metric is below one width at layer 0, $s_5 \lt 1$, while its Euclidean spread exceeds the mean centre's length, $s_{5,\mathrm{eu}} \gt 1$: the large Euclidean motion of the centre lies in directions the well does not resolve. At layer 1, where the occupancy at the token is 0.14, $s_5 \gt 1$.
- **AR3 (the live set is fixed before the decay phase).** The identity of the dominant well is the same at 22,500 and 31,000 at both layers (well 5), with $J_5 \ge 0.9$; at step 500 no well dominates, in the sense that the median over tokens of $\max_k \pi_k$ is below 0.5 at both layers.
- **AR4 (the ordering inside the well settles last).** The rank agreement is the strictest measure: for the dominant well in the late pair, $J_5 \ge 0.9$ and $\Delta_5 \le 0.1$, but $\rho_u \lt 0.9$ at at least one layer, i.e. the well owns the same tokens with the same radial law before the tokens' positions along its stiff directions stop moving.

If AR1 and AR3 hit, the decay phase of the schedule is the earliest training time at which a region of this model is admissible on clause 4, and clause 4 should be checked per well rather than globally. If AR2 hits, the metric spread of Definition 3 is the right measure of clause 3 and the Euclidean split of the well probe is not. If AR4 hits, the stiff coordinates are the last thing to settle and a signature must wait for them, which argues for logging $(J, \Delta, \rho_r, \rho_u)$ per well at every evaluation in future arms rather than at three checkpoints.

---

## 8. Results

The full report, with the setup, code, tables, figures and analysis, is [*Well Settling Map of SR2-NP*](Well_Settling_Map_SR2_NP.md). In short (calls confirmed by the author 2026-11-24, scored on their letter):

- **AR1, settling is late: hit.** Every drift measure is closer to its ideal value in the late pair, at both layers; the early pair shares nothing at the token level ($\rho_r$ and $\rho_u$ are near 0).
- **AR2, the metric spread against the Euclidean: hit.** At layer 0 the dominant well's context spread is $s_5 = 0.24$ widths against a Euclidean spread of $s_{5,\mathrm{eu}} = 1.81$; at layer 1, $s_5 = 1.63$.
- **AR3, the live set forms before the decay: miss.** The dominant well is fixed by step 22,500 ($J_5 = 1$), but at layer 1 it already dominates in the warm-up: the median over tokens of $\max_k \pi_k$ is 0.96 at step 500.
- **AR4, the inside settles last: miss on the letter.** $J_5 = 1$ and $\rho_u = 0.50$ at both layers, but $\Delta_5 = 0.235$ at layer 0 breaks the condition $\Delta_5 \le 0.1$, an artefact of the normalisation discussed below.

| call | verdict |
| --- | --- |
| AR1 | hit |
| AR2 | hit |
| AR3 | miss |
| AR4 | miss on the letter |

What it means for Definition 6:

- **No region of SR2-NP is admissible over the window from step 22,500 to the end** at $\varepsilon_{\mathrm{drift}} = 0.1$. The only live well keeps the same tokens, but the order of those tokens inside it agrees only at $\rho \approx 0.5$. In width units the tokens move very little (0.045 widths at layer 0), but that is as much as their own spread inside the well (0.98 of its interquartile range).
- **Clause 4 is meaningful only together with clauses 1 and 2.** The seven wells that own no tokens are the most settled ($\rho_r$ from 0.86 to 0.91 at layer 0); the live well is the least.
- **Clause 3 must be read in the well metric.** At layer 1 the Euclidean spread ranks the wells in the opposite order to the metric spread.
- **The earliest admissible time lies inside the decay phase or at its end**, and three checkpoints cannot place it more precisely.

**A proposed amendment to Definition 4 (author's call).** The radial drift divides the Wasserstein distance by the later median radius, which rescales by an arbitrary depth: at layer 0 a motion of 0.048 widths reads as 0.235. The scale that matters to a signature is the spread of the tokens inside the well. The per-token change of $r_k$ divided by the interquartile range of $r_k$ is that measure, and it would replace $\Delta_k$ in clause 4.

**A second proposed amendment, to clause 1 and Definition 2 (author's call, not applied; raised 2026-11-24 after the map was read).** Liveness is decided by an argmax: a well is live on the tokens it *owns*, those where it carries the largest share. That statistic hides the wells that carry a minority of the force weight at every token and so own none, and on this model those wells are load-bearing. At layer 0 the owning well carries 77.4% of the force weight on SR2-NP, 60.7% on FO-b and 89.7% on K4-R6 (the well probes in `results/well_screen/`), so between a tenth and two fifths of the weight sits with wells the criterion calls dead. The removal screen ([Well_Screen_and_Precision.md](../depth_ladder/Well_Screen_and_Precision.md)) prices that minority: on K4-R6, dropping layer 0's three non-owning wells, about 10% of the weight, costs +29.5% in perplexity, dropping all but the owning well costs +1,293%, and the trained two-well model K2-R6 settles +11.7% behind SR2. The direction count explains why a small share matters: layer 0 needs more than 20 of its 24 directions (KS4), and the owning well supplies only its own $R$ of them. The pattern is not an artefact of removing the pair force, since FO-b and K4-R6 keep it. At layer 1 the argmax and the weight agree, 100% on every arm, and keeping one well costs 0.0%.

Two consequences follow. **"The region of well $k$" at layer 0 is a superposition**, a landscape shaped by all eight wells, of which the criterion sees one; Proposition 3's decomposition into one frozen well plus a $\Delta$ bounded by $\varepsilon_{\mathrm{dom}}$ then puts 10 to 40% of the force into the error term by construction. **And clause 1 should be stated in a quantity that sees the minority.** Two candidates, both already measured by the screen: a well is live when its share of the total force weight $\sum_t g_k(t) / \sum_t \sum_j g_j(t)$ exceeds a threshold, or when removing it costs more than a threshold in perplexity on the fixed windows. The second is the operational meaning of "the model needs this well" and is what KS3 to KS5 measure; the first is cheaper and needs no re-evaluation. Either replaces ownership; neither changes clauses 3 to 5.

---

## 9. What this does not settle

- **The tolerances.** Definition 6 has five thresholds and nothing here sets them; §7's calls use round values. The tolerances that matter are the ones under which the first consuming template (plan note, §7, Stage 1′) fires reproducibly, so they will be set by that experiment, not by this note.
- **Regions inside a well.** With one owning well per layer (one well in fact at layer 1; at layer 0 one well by the argmax and eight by the force weight, see the second amendment in §8), the admissible region of SR2-NP is a single basin per layer, and the interesting structure, if any, is inside it: where in the well frame the trajectories repeat. The present measurement records $z_k$ only through $r_k$ and $u_k$; a search for signatures needs the full well coordinate on many windows, which is the particle store's business, not this note's.
- **Three checkpoints are a coarse map.** The claims about the decay phase rest on two intervals. A future arm should log the per-well settling measures at every evaluation (centres, precisions and weights at the 16 windows: a small file per 500 steps), which Cell 0 does not yet do.
- **The models with a pair force.** In SR2 and the PMX line a second context mechanism acts on the state directly; clause 3 then has a second term (the pair force's share of the force at the token, which [Probe A](../mean_field/README.md) and the recall split already measure) and the bound of Proposition 3 a fourth error term.

---

## 10. Reproduce

```
cd notebooks/conservative_arch/scaleup/debug
python3 well_settling_map.py ../results/well_settling <SR2-NP run folder> windows=16 threads=8 \
    val=<openwebtext_val_2M.npy> name=sr2np
```

The three checkpoints default to `step500,step22500,best`; `ckpts=` overrides. The script reuses `copy_split.py`'s window selection and Cell-0-from-tag build and `vtheta_well_probe.py`'s hook on the bank's first evaluation per layer.
