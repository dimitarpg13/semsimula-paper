# Placing SARF Anchors from Converged Gaussian Well Centres

**Status:** companion note to *Semantic Simulation: A Prescriptive Lagrangian Framework for Efficient Semantic Inference* (Gueorguiev, 2026).
**Scope:** a methodology for replacing the PMI-peak heuristic used to place the static SARF Gaussian-well anchors with a *semi-empirical law* derived from, and validated against, the converged learned centres of the (context-dependent) Gaussian variant on OpenWebText.
**Companion docs:**

- [`Structured_VTheta_Design_and_Theory.md`](./Structured_VTheta_Design_and_Theory.md) -- structured $V_\theta$ theory and the SQ3/Gaussian/SARF family.
- [`Training_Instabilities_in_Fock-PARFLM_with_structured_V_theta.md`](./Training_Instabilities_in_Fock-PARFLM_with_structured_V_theta.md) -- stability fixes (LR, watchdog, sigma/precision caps) carried into the analysis loader.
- **Implementation:**
  - Analysis module: [`notebooks/conservative_arch/parf/well_centre_analysis.py`](../notebooks/conservative_arch/parf/well_centre_analysis.py).
  - Driver notebook: [`notebooks/conservative_arch/scaleup/colab_analyze_well_centres.ipynb`](../notebooks/conservative_arch/scaleup/colab_analyze_well_centres.ipynb).
  - Model: [`notebooks/conservative_arch/parf/model_gaussian_vtheta.py`](../notebooks/conservative_arch/parf/model_gaussian_vtheta.py).

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [The architectural distinction: context-dependent vs static centres](#2-the-architectural-distinction-context-dependent-vs-static-centres)
3. [The centre cloud and its weighted modes](#3-the-centre-cloud-and-its-weighted-modes)
4. [Candidate anchor-placement rules](#4-candidate-anchor-placement-rules)
5. [Validation protocol and metrics](#5-validation-protocol-and-metrics)
6. [Results](#6-results)
7. [Recommendation framework](#7-recommendation-framework)
8. [Future work](#8-future-work)
9. [References](#9-references)

---

## 1. Motivation

The SARF variant (`SARFGaussianVTheta`) places $N_S$ Gaussian wells at *frozen* anchor positions $a_j \in \mathbb{R}^d$ and learns only the per-well widths $\sigma_j$ and the context-conditioned mixing weights $w_j(\xi)$. The current production heuristic chooses the anchors as the embeddings of the $N_S$ tokens with the highest pointwise-mutual-information (PMI) co-occurrence peak.

This raises an obvious question: **is the PMI-peak rule the right place to put the wells?** The general Gaussian variant (`MixtureGaussianVTheta`) does not freeze its centres -- it *learns* them as a function of context. After convergence on a large corpus, those learned centres encode where the model actually wants its attractors. If we can summarise that learned structure with a closed-form rule, we obtain a principled, corpus-informed anchor placement that needs no learned `mu_proj` at all -- recovering the SARF parameter savings while preserving the Gaussian variant's quality.

An early signal that PMI may be suboptimal comes from [`Structured_VTheta_Design_and_Theory.md`](./Structured_VTheta_Design_and_Theory.md) §8.3: the learned SQ3/Gaussian basins on TinyStories decode (through the LM head) to high-frequency narrative and function tokens -- "lived", "dad", "her", `,"` -- which are *not* PMI-extremal tokens. This suggests a frequency- or manifold-based rule may fit the learned centres better than PMI peaks.

## 2. The architectural distinction: context-dependent vs static centres

The two variants differ in a way that reframes the entire question.

**Gaussian variant (learned centres).** The centres are a linear readout of the context $\xi$:

$$
\mu_k(\xi) = W_\mu \xi + b_\mu, \qquad k = 1, \dots, K,
$$

so each token, in each context, produces its own set of $K$ centres. The implementation (`MixtureGaussianVTheta.attractor_centres`, lines 142-144) reads them out analytically:

```python
def attractor_centres(self, xi: torch.Tensor) -> torch.Tensor:
    lead = xi.shape[:-1]
    return self.mu_proj(xi).view(*lead, self.K, self.d)
```

**SARF variant (static anchors).** The centres are frozen buffers shared across all contexts (`SARFGaussianVTheta.attractor_centres`, lines 250-252):

```python
def attractor_centres(self, xi: torch.Tensor) -> torch.Tensor:
    lead = xi.shape[:-1]
    return self.anchors.unsqueeze(0).expand(*lead, -1, -1)
```

Consequently, "the converged Gaussian well centres" is **not** a fixed set of $K$ points. It is a *point cloud* in $\mathbb{R}^d$, traced out by $\mu_k(\xi)$ as $\xi$ ranges over the corpus, with each point carrying the responsibility $w_k(\xi)$ that the model assigns to it. A static anchor rule must therefore predict the **modes** of that weighted cloud, not a handful of fixed centres.

**Shared basis.** The model uses tied embeddings, $\text{logits} = h_L E^\top$, so token embeddings $E[v]$ and the learned centres $\mu_k(\xi)$ live in the *same* $\mathbb{R}^d$ basis. Comparing token-embedding anchors to learned centres geometrically is therefore well-posed, provided both are placed on a common scale. Because `ln_after_step=True` keeps hidden states on the shell $\lVert h_L \rVert \approx \sqrt{d}$, and the Phase-5 SARF builder row-standardises its anchors to the same shell, we apply the identical row-standardisation

$$
\mathrm{std}(v) = \frac{v - \bar{v}\mathbf{1}}{\text{sd}(v) + \varepsilon}, \qquad \lVert \mathrm{std}(v) \rVert \approx \sqrt{d},
$$

to every vector (cloud, modes, and all candidate anchors) before any geometric metric.

## 3. The centre cloud and its weighted modes

Let the held-out corpus produce contexts $\xi_t$ for token positions $t = 1, \dots, T$. The **centre cloud** is

$$
\mathcal{C} = \lbrace \big(\mu_k(\xi_t), w_k(\xi_t)\big) : t = 1, \dots, T, \quad k = 1, \dots, K \rbrace,
\qquad
w_k(\xi) = \text{softmax}_k\big(W_w \xi\big),
$$

where the responsibilities $w_k(\xi)$ are read from the same `w_proj` head the model uses at inference. Flattening over $(t, k)$ gives weighted points $\{(c_i, w_i)\}_{i=1}^{TK}$.

The **empirical target modes** $M = \{m_1, \dots, m_{N_S}\}$ are the centroids of a responsibility-weighted $k$-means on the cloud:

$$
M = \arg\min_{\{m_j\}} \sum_{i} w_i \min_j \lVert c_i - m_j \rVert^2 .
$$

These $N_S$ modes are the ground truth: they are where the converged Gaussian model concentrates its attractor mass. A good anchor rule reproduces $M$.

An auxiliary diagnostic, the **per-component centroid** $\bar{\mu}_k = \tfrac{1}{T}\sum_t \mu_k(\xi_t)$, collapses each mixture component to a single point; decoding $\{\bar{\mu}_k\}$ through the LM head exposes the semantic identity of each basin and the effective component count $K_{\text{eff}}$ (components with negligible mean responsibility are inactive).

## 4. Candidate anchor-placement rules

Each rule produces $N_S$ anchors in the shared $\mathbb{R}^d$ basis. All are implemented in `build_rule_anchors`.

**R0 -- PMI peak (current baseline).** Anchors are the embeddings of the $N_S$ tokens with the largest PMI co-occurrence peak:

$$
\text{PMI}_{\max}(v) = \max_{u \neq v} \log \frac{p(u, v)}{p(u) p(v)},
\qquad
a_j = E\big[v_j\big],\quad v_j \in \text{top-}N_S\ \text{PMI}_{\max}.
$$

**R1 -- Unigram frequency.** Anchors are the embeddings of the $N_S$ most frequent tokens:

$$
a_j = E\big[v_j\big],\quad v_j \in \text{top-}N_S\ n(v),
$$

with $n(v)$ the corpus count. Motivated by the §8.3 observation that learned basins decode to frequent tokens.

**R2 -- Hidden-manifold modes.** Anchors are the (uniformly weighted) $k$-means centroids of the sampled final hidden states $\{h_L\}$ -- i.e. the modes of the data manifold itself, independent of the potential:

$$
\{a_j\} = \text{k-means}_{N_S}\big(\{h_L\}\big).
$$

**R3 -- Principal-axis (PCA) shell.** Anchors are placed symmetrically along the leading principal axes of the weighted centre cloud. With weighted mean $\bar{c}$ and weighted covariance

$$
\Sigma = \sum_i w_i(c_i - \bar{c})(c_i - \bar{c})^\top = \sum_i \lambda_i v_i v_i^\top,
$$

emit a $\pm$ pair per leading eigenpair:

$$
a_i^{\pm} = \bar{c} \pm c\sqrt{\lambda_i} v_i,
\qquad i = 1, \dots, \big\lceil N_S / 2 \big\rceil,
$$

with shell scale $c$ (default $1$). This is a purely geometric, token-free rule.

**R4 -- Surprisal-weighted information.** Anchors are the embeddings of the $N_S$ tokens that contribute the most total cross-entropy mass, tying the placement to the log-frequency mass mode:

$$
a_j = E\big[v_j\big],\quad v_j \in \text{top-}N_S\big[n(v) s(v)\big],
\qquad s(v) = -\log \hat{p}(v).
$$

**TARGET -- Centre-cloud modes.** The $M$ of §3, against which R0--R4 are scored.

## 5. Validation protocol and metrics

After row-standardisation (§2), each rule's anchor set $A$ is scored against the target modes $M$ and the weighted cloud $\{(c_i, w_i)\}$ by three complementary metrics.

**Chamfer distance** (geometric fidelity; lower is better):

$$
d_{\mathrm{CD}}(A, M) = \frac{1}{2}\left(
\frac{1}{|A|}\sum_{a \in A}\min_{m \in M} \lVert a - m \rVert
+
\frac{1}{|M|}\sum_{m \in M}\min_{a \in A} \lVert m - a \rVert
\right).
$$

**Mass coverage** (does the rule blanket where the model puts mass; higher is better):

$$
\text{Cov}_\sigma(A) = \frac{\sum_i w_i \mathbf{1}\left[\min_j \lVert c_i - a_j \rVert \le \sigma \right]}{\sum_i w_i},
$$

with the threshold $\sigma$ fixed once as the median nearest-mode distance of the cloud,

$$
\sigma = \text{median}_i \min_j \lVert c_i - m_j \rVert,
$$

so the coverage criterion is identical across all candidate rules.

**Decode Jaccard** (semantic agreement through the LM head; higher is better). Let $T_1(\cdot)$ map a set of vectors to the set of their top-1 decoded tokens under $v \mapsto \arg\max(v E^\top)$. Then

$$
J(A, M) = \frac{\lvert T_1(A) \cap T_1(M) \rvert}{\lvert T_1(A) \cup T_1(M) \rvert}.
$$

**Procedure.** (1) Load the converged Gaussian checkpoint and rebuild the model (the loader infers $K$ from the saved `mu_proj` shape and reuses the Phase-5 precision/sigma caps). (2) Extract the centre cloud over a held-out OpenWebText sample under `enable_grad` (the Verlet integrator computes forces via `autograd.grad`; outputs are detached immediately). (3) Compute the target modes $M$. (4) Build R0--R4 anchors. (5) Score all rules at the shared $\sigma$. (6) Render the PCA scatter, metric bars, and decode comparison.

## 6. Results

> **To be filled after the K=8 Gaussian run reaches convergence (~200k steps).** Run [`colab_analyze_well_centres.ipynb`](../notebooks/conservative_arch/scaleup/colab_analyze_well_centres.ipynb) against `fock_gaussian_sarf_owt_phase5_best.pt` and paste the printed table and decode readout below.

**6.1 Centre-cloud diagnostics.**

| Quantity | Value |
|----------|-------|
| Tokens sampled ($T$) | _TBD_ |
| Cloud points ($TK$, post-subsample) | _TBD_ |
| Mean responsibility per component | _TBD_ |
| $K_{\text{eff}}$ (active components) | _TBD_ |
| Shared coverage $\sigma$ | _TBD_ |

**6.2 Rule scores** (Chamfer $\downarrow$, coverage $\uparrow$, decode Jaccard $\uparrow$):

| Rule | Chamfer | Coverage | Decode Jaccard |
|------|---------|----------|----------------|
| R0 PMI peak (baseline) | _TBD_ | _TBD_ | _TBD_ |
| R1 Unigram frequency | _TBD_ | _TBD_ | _TBD_ |
| R2 Hidden-manifold modes | _TBD_ | _TBD_ | _TBD_ |
| R3 PCA shell | _TBD_ | _TBD_ | _TBD_ |
| R4 Surprisal-weighted | _TBD_ | _TBD_ | _TBD_ |

**6.3 Basin decode comparison.** Per-component centroids $\bar{\mu}_k$ and target modes $M$ decode to:

- TARGET modes: _TBD_
- Per-component centroids (with responsibilities): _TBD_
- R0/R1/R4 anchor token readouts: _TBD_

Figure: `notebooks/conservative_arch/scaleup/results/well_centre_rule_scores.png` (PCA scatter of cloud vs each rule's anchors, plus the metric bar chart).

## 7. Recommendation framework

The decision rule, to be applied once §6 is populated:

- **If a token-based rule (R1 or R4) matches or beats R0 PMI on Chamfer and coverage**, then a frequency- or information-based law is the better corpus-informed anchor placement, and it is as cheap to compute as PMI (a single `bincount`) while avoiding the $O(V_{\mathrm{top}}^2)$ co-occurrence matrix.
- **If the geometric rule (R3 PCA shell) dominates**, then the learned centres are organised by the principal axes of their own distribution rather than by any token identity; the anchors should be placed analytically from the cloud covariance, and the choice of $N_S$ becomes a question of how many principal axes carry mass.
- **If R2 hidden-manifold modes win**, then the wells simply track the data manifold, and anchors should be $k$-means centroids of $h_L$ -- decoupling anchor placement from the potential entirely.
- **If R0 PMI remains best**, the current heuristic is validated, and the contribution is the validation methodology plus a quantitative justification.

In all cases, the per-component decode (§6.3) provides the interpretability narrative: it names the basins the converged model actually uses and reports $K_{\text{eff}}$, informing the choice of $N_S$ and $K_{\text{mix}}$.

## 8. Future work

Once a rule demonstrably wins on the geometric and decode metrics, the **definitive test** is downstream: train short SARF runs whose anchors come from the top one or two rules and compare best validation perplexity against (a) PMI-peak SARF and (b) the learned Gaussian. If a rule wins there too, expose it as a new `anchor_mode` option in the Phase-5 SARF builder so it can be selected at training time. That builder change is intentionally **out of scope** for this note, which delivers the analysis tooling and the validation methodology.

## 9. References

- Gueorguiev, D. (2026). *Semantic Simulation: A Prescriptive Lagrangian Framework for Efficient Semantic Inference.*
- [`Structured_VTheta_Design_and_Theory.md`](./Structured_VTheta_Design_and_Theory.md) -- structured $V_\theta$ derivations, attractor interpretability (§6, §8.3), and $K_{\text{mix}}$ selection.
- [`Training_Instabilities_in_Fock-PARFLM_with_structured_V_theta.md`](./Training_Instabilities_in_Fock-PARFLM_with_structured_V_theta.md) -- the precision/sigma caps and watchdog logic reused by the analysis loader.
- Church, K. & Hanks, P. (1990). *Word Association Norms, Mutual Information, and Lexicography.* Computational Linguistics 16(1) -- PMI baseline.
