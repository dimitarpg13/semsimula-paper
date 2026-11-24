# Well Settling Map of SR2-NP

**When do the wells of a trained conservative model stop moving, and is the inside of a well ever stable enough to look for templates in? A first reading on three checkpoints (CPU, 2026-11-24)**

Technical report of the first measurement of [*Admissible Regions for Template Search*](Admissible_Regions_for_Template_Search.md) (the theory note: definitions, propositions and the admissibility criterion this report tests). The calls AR1 to AR4 were written into the theory note's §7 before the results were read and confirmed by the author on 2026-11-24, with one caveat recorded in §6 below.

---

## Contents

1. [Definitions](#1-definitions)
2. [The problem](#2-the-problem)
3. [Setup](#3-setup)
4. [Code](#4-code)
5. [Results](#5-results)
6. [The calls](#6-the-calls)
7. [Analysis](#7-analysis)
8. [What it means for template search](#8-what-it-means-for-template-search)
9. [Caveats](#9-caveats)
10. [Reproduce](#10-reproduce)

---

## 1. Definitions

The quantities are those of the theory note, repeated here so that the report reads on its own. Throughout, $k$ indexes the wells of the bank ($K = 8$), $\ell$ the layer ($L = 2$), $t$ the token (8,192 of them, $N$ in the formulas), and $\tau$ the training step of a checkpoint. A superscript $\tau$ on a quantity marks the checkpoint it was computed at.

### 1.1 The bank

Well $k$ at context $\xi$ (the five context channels at the token, concatenated) has a centre, a precision and a weight:

$$
\mu_k(\xi) = W_k (\xi + c_\ell) + b_k, \qquad
P_k(\xi) = \mathrm{diag}\big(a_k(\xi)\big) + B_k(\xi) B_k(\xi)^{\top}, \qquad
w_k(\xi) \gt 0.
$$

The centre is affine in the context shifted by the layer's learned depth code $c_\ell$. The precision is a positive diagonal plus a low-rank part, with $B_k$ a $d \times r$ matrix of rank $r = 4$. The weights are a softmax over the wells times a fixed scale. All three are functions of the context through the bank's projections. The potential at state $h$ is

$$
V_\theta(h, \xi) = -\sum_{k=1}^{K} w_k(\xi) e^{-q_k(h, \xi)/2}.
$$

### 1.2 Per token and well

All of these are computed at the state $h$ that the layer receives, before its integration step.

**Definition 1 (well radius).** The squared well radius and the well radius are

$$
q_k = (h - \mu_k)^{\top} P_k (h - \mu_k), \qquad r_k = \sqrt{q_k}.
$$

$q_k$ is the squared distance of the state from the well's centre in the well's own metric; $r_k$ is that distance in widths of the well, so $r_k = 1$ is one width from the centre along whichever direction the state lies.

**Definition 2 (force weight, dominance share, dominant well).**

$$
g_k = w_k e^{-q_k/2}, \qquad
\pi_k = \frac{g_k}{\sum_{j=1}^{K} g_j}, \qquad
k^{\ast} = \arg\max_k \pi_k.
$$

$g_k$ is the well's contribution to the potential at the state, before normalisation; $\pi_k$ is the fraction of the total that well $k$ carries; the dominant well $k^{\ast}$ is the one carrying the largest fraction. We say the dominant well *owns* the token.

**Definition 3 (stiff coordinates).**

$$
u_k = B_k^{\top} (h - \mu_k) \in \mathbb{R}^{r}.
$$

These are the components of the displacement along the well's narrowest directions (the low-rank part of the precision). Only their norm $\lVert u_k \rVert$ is used below.

### 1.3 Per well, at one checkpoint

Let $S_k$ be the set of tokens the well owns, those with $k^{\ast} = k$, and let

$$
\bar{\mu}_k = \frac{1}{N} \sum_{t=1}^{N} \mu_k(\xi_t)
$$

be the well's mean realised centre over the tokens.

**Definition 4 (share and dominance).** The share of well $k$ is $\lvert S_k \rvert / N$, the fraction of tokens it owns (clause 1 of admissibility, *live*). Its dominance is the median of $\pi_k$ over the tokens in $S_k$, how completely it owns them (clause 2, *dominant*).

**Definition 5 (context spread).** The context spread in the well metric, and its Euclidean counterpart relative to the length of the mean centre, are

$$
s_k = \Big( \mathrm{median}_t \big[ (\mu_k(\xi_t) - \bar{\mu}_k)^{\top} P_k(\xi_t) (\mu_k(\xi_t) - \bar{\mu}_k) \big] \Big)^{1/2},
$$

$$
s_{k,\mathrm{eu}} = \frac{\big( \mathrm{median}_t \lVert \mu_k(\xi_t) - \bar{\mu}_k \rVert^2 \big)^{1/2}}{\lVert \bar{\mu}_k \rVert}.
$$

$s_k$ says how far the context moves the well's centre in widths of the well (clause 3, *context-tame*); $s_{k,\mathrm{eu}}$ says how far it moves in plain units, as a multiple of the centre's own length.

### 1.4 Per well, between two checkpoints

Two checkpoints $\tau \lt \tau'$ are run on the same tokens, so every quantity can be compared token by token (clause 4, *settled*).

**Definition 6 (assignment overlap).** The Jaccard index of the sets of tokens the well owns at the two checkpoints, ideal value 1:

$$
J_k = \frac{\lvert S_k^{\tau} \cap S_k^{\tau'} \rvert}{\lvert S_k^{\tau} \cup S_k^{\tau'} \rvert}.
$$

**Definition 7 (radial drift).** The 1-D Wasserstein distance between the distributions of the radius over the tokens at the two checkpoints, divided by the later median radius, ideal value 0. In one dimension the Wasserstein distance is the mean gap between the sorted samples; writing $r_{k,(i)}^{\tau}$ for the $i$-th smallest radius at checkpoint $\tau$,

$$
\Delta_k = \frac{W_1}{\mathrm{median}_t \big( r_k^{\tau'} \big)}, \qquad
W_1 = \frac{1}{N} \sum_{i=1}^{N} \big\lvert r_{k,(i)}^{\tau'} - r_{k,(i)}^{\tau} \big\rvert.
$$

**Definition 8 (rank agreements).** The Spearman correlations over the tokens of the radius and of the norm of the stiff coordinates, ideal value 1:

$$
\rho_r = \mathrm{corr}\big( \mathrm{rank}(r_k^{\tau}), \mathrm{rank}(r_k^{\tau'}) \big), \qquad
\rho_u = \mathrm{corr}\big( \mathrm{rank}(\lVert u_k^{\tau} \rVert), \mathrm{rank}(\lVert u_k^{\tau'} \rVert) \big),
$$

where $\mathrm{rank}$ replaces each token's value by its rank among the $N$ tokens and $\mathrm{corr}$ is the Pearson correlation. $\rho_r$ asks whether the tokens keep their order of depth in the well; $\rho_u$ whether they keep their order along its stiff directions.

### 1.5 Quantities used in the analysis only

These were not pre-registered. The radial drift in width units is $W_1$ itself, without the division in Definition 7. The **change-to-range ratio** compares how far a typical token moves with how spread out the tokens are:

$$
\eta_k = \frac{\mathrm{median}_t \big\lvert r_k^{\tau'}(t) - r_k^{\tau}(t) \big\rvert}{\mathrm{IQR}_t \big( r_k^{\tau'} \big)},
$$

with $\mathrm{IQR}$ the interquartile range over the tokens. $\eta_k$ near 0 means the tokens keep their places relative to each other; $\eta_k$ near 1 means a typical token moves as far as the spread of the whole population.

### 1.6 The warmth of a position

Channel $c$ forgets the past at rate $\alpha_c$, so at position $t$ the weight of the part of the context missing before the window's start is $\alpha_c^{t}$. The channel is warm at tolerance $\varepsilon$ from position

$$
t_c(\varepsilon) = \frac{\ln \varepsilon}{\ln \alpha_c}.
$$

At $\varepsilon = 0.1$ the five channels are warm from positions 4, 8, 45, 229 and 459.

---

## 2. The problem

The author's question (2026-11-24): during training everything moves, including the wells, so in which regions of semantic space, and from which point in training, can a search for the patterns that become templates start? The theory note turns the question into its Definition 6: a region (the basin of well $k$ at layer $\ell$, at positions past a warm-up, over a window of checkpoints) is admissible when the well is live, dominant, context-tame, settled and read at warm positions. Its Proposition 3 shows that inside such a region a signature observed along a trajectory is reproducible to first order in the tolerances.

This report measures the five clauses on one trained model. It asks four concrete questions, one per call:

- does settling happen late, in the decay phase of the learning-rate schedule (AR1)?
- does the metric spread of Definition 5 see something the Euclidean spread does not (AR2)?
- when does the live set of wells form (AR3)?
- is the inside of a well (the order of tokens by depth and along its stiff directions) the last thing to settle (AR4)?

---

## 3. Setup

**Model.** SR2-NP: the Gen 3 conservative core with the second-order damped integrator (SR2), trained without the pair force ([Stage 3](../productization/Stage3_SR2_Without_Pair_Force.md)); hidden size $d = 384$, two layers, a joint anisotropic depth-conditioned bank of $K = 8$ wells of rank $r = 4$, read by five context channels with decays 0.50, 0.75, 0.95, 0.99 and 0.995; untied output head; 76,434,307 parameters. Without the pair force the bank is the model's only context mechanism, so clause 3 captures the whole of the context's effect on the landscape.

**Training schedule.** 32,500 steps on OpenWebText in 512-token windows, with a warmup-stable-decay schedule and a peak learning rate of 0.0012: warm-up over the first 1,625 steps (5%), constant to step 21,125 (65%), then decay to 0.00006.

**Checkpoints.** The three the run folder keeps:

| checkpoint | step | phase | learning rate (training log) | validation perplexity (training log) | perplexity on the 16 windows here |
| --- | --- | --- | --- | --- | --- |
| step500 | 500 | warm-up | 0.00037 | 463.1 | 490.9 |
| step22500 | 22,500 | 1,375 steps into the decay | 0.00116 (97% of peak) | 67.3 | 69.0 |
| best | 31,000 | late decay | 0.00011 | 56.7 | 58.4 |

The early pair (500 to 22,500) spans the warm-up and the constant phase; the late pair (22,500 to 31,000) spans most of the decay.

**Data.** 16 windows of 512 tokens spaced evenly over the OpenWebText validation set (8,192 tokens), the same windows as SR2-NP's well probe (`vtheta_well_probe.py`, `well_probe_sr2np_owt.txt`). The identical tokens go through every checkpoint, so all between-checkpoint comparisons are token by token.

**What is captured.** At each layer, the bank's components $\mu_k$, $a_k$, $w_k$ and $B_k$ at every token and the incoming state $h$, at the bank's first evaluation in the layer (at $h$, not at the integrator's midpoint). Everything in §1 is computed from these in float32.

**Cost.** About 80 seconds for the three checkpoints on 8 CPU threads (log from 09:28:15 to 09:29:36); peak memory was not measured.

---

## 4. Code

Script: [`debug/well_settling_map.py`](../../notebooks/conservative_arch/scaleup/debug/well_settling_map.py). It builds each checkpoint through the ladder notebook's own Cells 0 to 5b (via `copy_split.py`'s rebuild from the run folder's tag, as every CPU probe of the ladder does), so the model is exactly the trained one.

**The hook.** The bank's `harmonic_terms_lowrank` is wrapped; the first call per layer is at the state the layer receives, and the components it was given are recorded:

```python
orig_lr = bank.harmonic_terms_lowrank
def lr(xi, h, *, comps=None):
    out = orig_lr(xi, h, comps=comps)
    if not cur['lr']:                                # the first evaluation of the layer: at h, not at h_mid
        cur['lr'] = True
        with torch.no_grad():
            mu, a, w, B = (c.detach().float() for c in (comps if comps is not None else bank._components(xi)))
            r = rec.setdefault(cur['layer'], {})
            for k_, v in dict(h=h.detach().float().reshape(-1, h.shape[-1]), mu=mu.reshape(-1, *mu.shape[-2:]),
                              a=a.reshape(-1, *a.shape[-2:]), w=w.reshape(-1, w.shape[-1]),
                              B=B.reshape(-1, *B.shape[-3:])).items():
                r.setdefault(k_, []).append(v)
    return out
```

**The quantities of §1.2 and §1.3.** The precision is never formed as a $d \times d$ matrix; its diagonal and low-rank parts are applied separately, as in the model:

```python
diff = h.unsqueeze(1) - mu                                          # (N, K, d)
u = torch.einsum('nkd,nkdr->nkr', diff, B)                          # stiff coordinates
q = (a * diff * diff).sum(-1) + (u * u).sum(-1)                     # squared well radius
g = w * torch.exp(-0.5 * q); pi = g / g.sum(-1, keepdim=True).clamp_min(1e-30)
mbar = mu.mean(0, keepdim=True); dm = mu - mbar
c = (a * dm * dm).sum(-1) + (torch.einsum('nkd,nkdr->nkr', dm, B) ** 2).sum(-1)   # context spread, well metric
ceu = (dm * dm).sum(-1) / (mbar * mbar).sum(-1).clamp_min(1e-30)                # Euclidean spread / |mean centre|^2
```

The medians and square roots of Definition 5 are taken from `c` and `ceu` when the report is written.

**The drift measures of §1.4**, per well between consecutive checkpoints:

```python
s0, s1 = D0['kstar'] == k, D1['kstar'] == k                         # tokens the well owns at each checkpoint
jaccard = (s0 & s1).sum() / (s0 | s1).sum()
r0, r1 = np.sqrt(D0['q'][:, k]), np.sqrt(D1['q'][:, k])
w1 = np.abs(np.sort(r0) - np.sort(r1)).mean() / np.median(r1)      # 1-D Wasserstein of the radial laws, normalised
rho_r, rho_u = spearman(r0, r1), spearman(D0['un'][:, k], D1['un'][:, k])
```

Figures: [`figures/_make_admissible_regions_figs.py`](../figures/_make_admissible_regions_figs.py), from the script's JSON and NPZ outputs.

---

## 5. Results

### 5.1 The live set

![dominance](../figures/admissible_regions/ar_dominance.png)

Bars: the share of tokens that well 5 owns, per checkpoint and layer; dots: its dominance on them (Definition 4).

The table gives, per layer and checkpoint, the well that owns the most tokens with its share, the median over all tokens of the largest dominance share $\max_k \pi_k$ (how clearly any well wins at a typical token), and the median norm of the state $\lVert h \rVert$.

| layer | checkpoint | well owning most tokens (share) | median largest dominance share | median state norm |
| --- | --- | --- | --- | --- |
| 0 | step500 | well 3 (0.69); then well 0 (0.18), well 6 (0.08) | 0.163 | 0.55 |
| 0 | step22500 | well 5 (1.000) | 0.745 | 3.01 |
| 0 | best | well 5 (1.000) | 0.775 | 3.05 |
| 1 | step500 | well 5 (0.64); then well 4 (0.27), well 1 (0.09) | 0.959 | 19.53 |
| 1 | step22500 | well 5 (1.000) | 1.000 | 19.60 |
| 1 | best | well 5 (1.000) | 1.000 | 19.60 |

By step 22,500 one well, the same one at both layers (well 5 of the shared bank), owns every token. At layer 1 it already owns 64% of the tokens with dominance 0.96 at step 500, inside the warm-up. At layer 0 no well dominates anything at step 500: the largest dominance share at the median token is 0.16, with the tokens split across four wells.

### 5.2 Training drift

![settling](../figures/admissible_regions/ar_settling.png)

The dominant well (well 5), both pairs of checkpoints; the dashed line is the median $\rho_r$ of the other seven wells in the late pair.

The table gives, for well 5, the four measures of §1.4: the assignment overlap $J_5$, the radial drift $\Delta_5$, and the rank agreements $\rho_r$ and $\rho_u$. The last two columns give the same rank agreement of the radius and the same radial drift for the other seven wells.

| layer | pair | overlap | radial drift | rank agreement, radius | rank agreement, stiff | other wells: rank agreement, radius (median, range) | other wells: radial drift (range) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 500 to 22,500 | 0.001 | 1.274 | −0.073 | −0.052 | 0.10 (−0.11 to 0.44) | 0.35 to 0.70 |
| 0 | 22,500 to best | 1.000 | 0.235 | 0.540 | 0.502 | 0.90 (0.86 to 0.91) | 0.013 to 0.032 |
| 1 | 500 to 22,500 | 0.637 | 0.551 | −0.008 | −0.093 | 0.03 (−0.07 to 0.15) | 0.18 to 0.76 |
| 1 | 22,500 to best | 1.000 | 0.075 | 0.533 | 0.503 | 0.69 (0.61 to 0.77) | 0.008 to 0.112 |

The early pair shares nothing at the level of individual tokens: the rank agreements are zero to two decimals at both layers. In the late pair the dominant well owns exactly the same tokens ($J_5 = 1$), but its tokens' order by depth and along its stiff directions agrees only at $\rho \approx 0.5$, and the seven wells that own no tokens agree far better (0.86 to 0.91 at layer 0).

**The absolute scale of the late-pair motion** (the analysis quantities of §1.5). Per layer for well 5: the Wasserstein distance $W_1$ in widths; the median per-token change of the radius; the interquartile range of the radius at the best checkpoint; their ratio $\eta_5$; and, as a gentler companion to $\rho_r$, the Pearson correlation of the radius between the two checkpoints.

| layer | Wasserstein distance in widths | median per-token change of the radius | interquartile range of the radius at best | change-to-range ratio | Pearson correlation of the radius |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.048 | 0.045 | 0.045 | 0.98 | 0.72 |
| 1 | 0.147 | 0.172 | 0.205 | 0.84 | 0.62 |

The tokens move by a twentieth of a width (layer 0) or a seventh of a width (layer 1): little on the scale of the well, but as much as the spread of the tokens inside it.

**Position does not explain the low agreement.** The late-pair $\rho_r$ of well 5, computed separately over four bands of positions in the window (the bands start where successive channels become warm, §1.6):

| layer | positions 0 to 7 | positions 8 to 44 | positions 45 to 228 | positions 229 to 511 |
| --- | --- | --- | --- | --- |
| 0 | 0.76 | 0.49 | 0.53 | 0.54 |
| 1 | 0.61 | 0.49 | 0.52 | 0.54 |

Only the first eight tokens, whose context is nearly empty, agree better; past them the agreement is 0.5 whether or not the slow channels are warm.

### 5.3 The radial law

![radial](../figures/admissible_regions/ar_radial.png)

Left: the distribution of $r_5$ over the 8,192 tokens at each checkpoint. Right: the same tokens at step 22,500 against the best checkpoint.

The table gives the 10th percentile, the median and the 90th percentile of $r_5$, in widths, at each checkpoint.

| layer | step 500 | step 22,500 | best |
| --- | --- | --- | --- |
| 0 | 0.45, 0.59, 0.67 | 0.19, 0.25, 0.33 | 0.17, 0.20, 0.26 |
| 1 | 1.29, 2.50, 4.55 | 1.54, 1.82, 2.13 | 1.76, 1.97, 2.19 |

The two layers use the well differently. At layer 0 the tokens sit deep inside it, a fifth of a width from the centre, and sink deeper during the decay. At layer 1 they sit on a thin shell about two widths out. That is why the occupancy $e^{-q/2}$ at the token is about 0.14 ([well probe](../../notebooks/conservative_arch/scaleup/results/well_screen/well_probe_sr2np_owt.txt)) even though the well carries all of the pull; the shell moves outward slightly and narrows during the decay.

### 5.4 Context spread

![spread](../figures/admissible_regions/ar_spread.png)

At the best checkpoint, every well's context spread in its own metric $s$ against its Euclidean spread $s_{\mathrm{eu}}$ (Definition 5); the dominant well in green.

| layer | well 5: spread in the well metric | well 5: Euclidean spread | other wells: spread in the well metric (range) | other wells: Euclidean spread (range) |
| --- | --- | --- | --- | --- |
| 0 | 0.24 | 1.81 | 0.90 to 1.24 | 3.65 to 4.82 |
| 1 | 1.63 | 2.57 | 4.95 to 6.46 | 1.34 to 1.49 |

At layer 0 the context moves the dominant well's centre by 1.8 times the centre's own length, yet by only a quarter of a width in its metric. At layer 1 the dominant well has the *largest* Euclidean spread of the eight and the *smallest* metric spread, by a factor of three to four.

---

## 6. The calls

Pre-registered in the theory note's §7 (2026-11-24). The run finished while the note was being written, and its log was read to check the run before the author confirmed the calls; the calls were composed before that reading, but the file times cannot show it. The author accepted this for an exploratory first reading (2026-11-24). Scored on their letter, with $J_5$, $\Delta_5$, $\rho_r$, $\rho_u$ and $s_5$ the quantities of §1 for the dominant well:

**AR1, settling is late: hit.** *Prediction*: in the late pair every one of the four measures of §1.4 is closer to its ideal value than in the early pair, at both layers. *Reading*: at layer 0, $J_5$ rises from 0.001 to 1.000, $\Delta_5$ falls from 1.274 to 0.235, $\rho_r$ rises from −0.07 to 0.54 and $\rho_u$ from −0.05 to 0.50; at layer 1 the same four move from 0.637, 0.551, −0.01 and −0.09 to 1.000, 0.075, 0.53 and 0.50.

**AR2, the context moves the layer-0 well along soft directions: hit.** *Prediction*: at the best checkpoint, at layer 0, $s_5 \lt 1$ while $s_{5,\mathrm{eu}} \gt 1$; at layer 1, $s_5 \gt 1$. *Reading*: at layer 0, $s_5 = 0.24$ and $s_{5,\mathrm{eu}} = 1.81$; at layer 1, $s_5 = 1.63$.

**AR3, the live set is fixed before the decay: miss.** *Prediction*: the same well dominates at steps 22,500 and 31,000 at both layers, with $J_5 \ge 0.9$; and at step 500 no well dominates, in the sense that the median over tokens of $\max_k \pi_k$ is below 0.5 at both layers. *Reading*: the first part holds (well 5 at both layers, $J_5 = 1.000$). The second holds at layer 0 (0.163) and fails at layer 1 (0.959).

**AR4, the inside of the well settles last: miss on the letter.** *Prediction*: in the late pair, $J_5 \ge 0.9$ and $\Delta_5 \le 0.1$, but $\rho_u \lt 0.9$ at at least one layer. *Reading*: $J_5 = 1.000$ and $\rho_u = 0.50$ at both layers, but $\Delta_5 = 0.235$ at layer 0 breaks the second condition (0.075 at layer 1). At layer 1 the call holds in full.

| call | verdict |
| --- | --- |
| AR1 | hit |
| AR2 | hit |
| AR3 | miss (layer 1 at step 500) |
| AR4 | miss on the letter (layer 0's radial drift) |

Two hits, two misses. AR3's miss is informative: the layer-1 well forms during the warm-up, long before the decay. AR4's miss is partly an artefact of the radial drift's normalisation (§7.3). Its substance, that the order of tokens inside the well settles last, holds at both layers.

---

## 7. Analysis

### 7.1 Settling is a decay-phase event, and it is not complete by the end

Between step 500 and step 22,500 nothing survives at the level of tokens: the layer-0 landscape is reorganised entirely (the tokens move from wells 3, 0 and 6 to well 5), and even at layer 1, where the same well was already dominant, the order of tokens is uncorrelated ($\rho \approx 0$). Between 22,500 and 31,000 the assignment is frozen ($J_5 = 1$) and the radial law nearly so, but the order inside the well is still only half preserved. The learning rate at step 22,500 is 97% of its peak, so the late pair spans nearly the whole of the decay, and settling must happen inside that window. With three checkpoints the report cannot say whether the agreement rises smoothly with the falling learning rate or arrives near the end.

### 7.2 The live well is the least settled well

In the late pair the seven wells that own no tokens have rank agreements of 0.86 to 0.91 at layer 0 and radial drifts of 1% to 3%; the one well that owns every token has 0.54 and 24%. The likely reason (not measured here) is that nothing pulls on the dead wells: with no tokens they should receive almost no gradient, so the tokens' distances to them change only through the tokens' own motion. Settledness alone (clause 4) is therefore meaningless for a well that is not live and dominant (clauses 1 and 2), which is why the theory note's Definition 6 conjoins them. The well that matters is the one that keeps being trained, and it is the one still moving.

### 7.3 The scale on which the inside moves

In width units the dominant well is nearly frozen in the late pair: a median token changes its radius by 0.045 widths at layer 0 and 0.17 at layer 1. Measured against the width, the well has settled. But the tokens inside it are spread over an interquartile range of only 0.045 and 0.205 widths, so the change-to-range ratio $\eta_5$ is 0.98 and 0.84. Against the spread of the tokens, the inside has not settled: half of the ordering is re-drawn. A signature inside the well distinguishes tokens from each other, so the spread is the relevant scale, not the width.

This also explains AR4's miss on the letter. Definition 7 divides the Wasserstein distance by the later median radius. At layer 0 that median is 0.2 widths, so a motion of 0.048 widths reads as 0.235. The radius is already measured in widths, so the division only re-scales by an arbitrary depth. The scale-free alternative that matches this section is $\eta_k$. Adopting it as the radial measure of clause 4 is the author's call for the theory note (its §8).

### 7.4 The metric spread separates what the Euclidean spread confuses

The theory note's Proposition 2 predicted that a centre can move far in $\mathbb{R}^d$ while barely moving in its own metric, when the motion lies along the well's soft directions. Both layers show it, in different ways. At layer 0 the dominant well's centre moves by 1.8 times its own length, but only a quarter of a width in its metric. At layer 1 the Euclidean ordering of the wells is inverted: the dominant well has the largest Euclidean spread and the smallest metric spread. The centre-split readings of the well probe (channel term 10.6 against a centre of 8.6 at layer 0) had looked like "the context throws the wells about"; in the well's own terms it does not. Clause 3 must be read in the metric.

### 7.5 Two layers, two geometries

Layer 0 holds its tokens at the bottom of the well, and holds them more tightly as training ends: its median radius goes from 0.59 to 0.25 to 0.20 widths. Layer 1 holds them on a thin shell at about two widths, on the well's outer slope. Along an axis, the restoring force of a Gaussian well in width units is proportional to

$$
r e^{-r^2/2},
$$

which is largest at one width and, at two widths, still 45% of that maximum. For templates the difference matters. A signature at layer 0 is a small displacement from the well's centre; at layer 1 it is a position on the shell, an angular coordinate, which $\lVert u_5 \rVert$ only partly records.

---

## 8. What it means for template search

Applying the theory note's Definition 6 with round tolerances, $\varepsilon_{\mathrm{drift}} = 0.1$ and a context tolerance of one width:

- **No region of SR2-NP is admissible over the window from step 22,500 to the end.** At both layers the only live well has $\rho_r$ and $\rho_u$ near 0.5. Under the pre-registered normalisation layer 0 also fails on $\Delta_5$; under the scale-free alternative of §7.3 both layers fail on $\eta_5$ (0.98 and 0.84).
- **Clause 3 passes at layer 0 and fails at layer 1** ($s_5$ is 0.24 against 1.63). If any region of this model becomes admissible, it is the inside of the layer-0 well.
- **The earliest admissible training time lies inside the decay phase or at its end.** Three checkpoints cannot locate it more closely. A Tier 1 algorithm that searches during training therefore has to wait for the learning rate to fall. The safe alternative is a two-phase schedule: train the model, then search the frozen model for templates. Both are consistent with the Tier 1 decision (plan note §5): the bank is learned at training time and fired at inference.
- **The live well is the one to watch.** Monitoring settledness over all wells averages in the dead wells, which are trivially settled; the criterion has to be applied to the live, dominant wells only.

**What runs next, if the author agrees.**

1. **A finer map on an existing run.** The SR2-2B run (130,000 steps on 2B tokens) kept ten snapshots: steps 500, 6,500, 16,500, 32,500, 48,500, 65,000, 84,500, 97,500, 113,500 and 130,000. Running the same script on them (about 20 minutes of CPU, after a one-line change to accept `_snap.pt` names) would give the settling curve through the whole decay. Caveat: SR2-2B keeps the pair force, which adds a second context term to clause 3.
2. **Per-evaluation logging in a future arm.** Record the overlap, the radial drift, the two rank agreements and the context spread per well at every evaluation on the 16 fixed windows: a few kilobytes per 500 steps. Cell 0 would need a switch; nothing is changed yet.

---

## 9. Caveats

- **One model, three checkpoints, one seed.** The decay-phase claim rests on two intervals. The early checkpoint is inside the warm-up, so the early pair mixes the warm-up and the constant phase.
- **Spearman on a narrow distribution.** When the tokens' radii are concentrated (an interquartile range of 0.045 widths at layer 0), small motions reorder them, and the rank agreement penalises that fully. That is the intended behaviour for a signature, but it is a strict test. The Pearson correlations (0.72 and 0.62) say the same thing less severely.
- **The norm is not the stiff coordinate.** $\rho_u$ compares the norms $\lVert u_k \rVert$ of the four stiff coordinates, not the vectors; a token that rotates around the well's narrow directions at fixed norm is counted as unchanged. The vector comparison needs the bases $B_5$ at the two checkpoints to be matched (they are defined only up to a rotation in their rank-4 span), and was not done.
- **The state is read at the layer's input.** The integrator's midpoint, where the bank is evaluated a second time, is not recorded.
- **The calls' timing.** As stated in §6: written before the reading, confirmed by the author after it, accepted as such for an exploratory first measurement.

---

## 10. Reproduce

```
cd notebooks/conservative_arch/scaleup/debug
python3 well_settling_map.py ../results/well_settling \
    semsimula_fock_cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_nophi_xilive_sr2_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn \
    windows=16 threads=8 val=<openwebtext_val_2M.npy> name=sr2np
cd ../../../../companion_notes/figures && python3 _make_admissible_regions_figs.py
```

Outputs: [`results/well_settling/settling_sr2np.txt`](../../notebooks/conservative_arch/scaleup/results/well_settling/settling_sr2np.txt) (the report the tables above are taken from), `settling_sr2np.json` (per-well tables), `settling_sr2np.npz` (per checkpoint, layer and token: the squared radius, the dominance shares, the norm of the stiff coordinates, the context spread and the dominant well), and the console log `run_sr2np.log`. The position-band, width-unit and change-to-range figures of §5.2 come from the NPZ: tokens are stored window by window, so the position is the token index modulo 512.
