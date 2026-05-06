# Reducing the Information Bottleneck in Multi-Channel ξ SPLM

> **Status.** Drafted **2026-05-01**, after the buggy multi-ξ 2k forensic completed and while the leak-corrected K = 4 fixed pilot is in flight (PID 72285, ETA ~6 AM EDT 2026-05-02). This document captures the information-theoretic and signal-processing case for redesigning the K-channel ξ block beyond the current "K nested EMAs with hand-picked decay rates" baseline. It is **a starting point**: future sections will record empirical results, design ablations, and any architectural pivots motivated by the K = 4 outcome.
>
> **Companion documents.**
> - [`Multi-Channel_vs_Single_Channel_Xi_SPLM_Design.md`](Multi-Channel_vs_Single_Channel_Xi_SPLM_Design.md) — original design rationale and forward-pass expressivity argument for K-channel ξ.
> - [`Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md`](Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md) — anti-causal autograd leak that corrupted the pre-fix K = 4 measurement (PPL 1.05, inflation 389×); §4.6 will hold the leak-corrected K = 4 baseline this document compares against.
> - [`Restructuring_paper_v3_after_causal_leak_bug.md`](Restructuring_paper_v3_after_causal_leak_bug.md) — paper-restructure plan; §A1 of the rewritten paper is the natural home for any architectural improvements derived in this document.

---

## 1. The question

The current K = 4 multi-channel ξ block summarises past hidden states with K parallel exponential moving averages (EMAs) at decay rates $\{\alpha_k\}\_{k=0}^{K-1}$:

$$
\xi^k\_t = \mathrm{EMA}\_{\alpha\_k}(h)\_t = (1-\alpha\_k)\sum\_{s\le t}\alpha\_k^{t-s} h\_s.
$$

Empirically the K = 4 design has α-init `[0.0, 0.5, 0.9, 0.99]`, giving effective horizons {1, 2, 10, 100} tokens. The question this document addresses is:

> Given K parallel causal channels summarising context, is there a principled — information-theoretic or optimisation-based — way to choose the channels, the decay rates, and the inter-channel coupling so that adding the $(k+1)$-th channel reliably increases the predictive information available to the model, rather than plateauing into redundant low-pass overlap?

The motivation is concrete: under the buggy integrator the optimiser drove the channels into an exploitative pattern (harvest the leak); we cannot use that as evidence about the legitimate-K story. The K = 4 fixed pilot lands tomorrow. The decision of whether and how to extend to K = 6, K = 8 (or to redesign the channels at fixed K) should be guided by an explicit theoretical objective rather than by ad hoc grid choices.

---

## 2. The natural information-theoretic objective

The right thing to maximise is the **predictive mutual information** between the K-channel summary at time $t$ and the unrealised future, subject to a capacity constraint on the past. This is the Information Bottleneck (IB) functional applied to causal state representations:

$$
\max\_{\{\alpha\_k\}, \theta}\quad I(\xi\_t; h\_{\gt t}) - \beta \cdot I(\xi\_t; h\_{1:t}).
$$

In words: $\xi\_t$ should be a **sufficient** statistic for predicting the future (first term) and a **minimal** one — no excess information about the past beyond what predicts the future (second term). The Lagrange multiplier $\beta \gt 0$ trades off representational capacity against predictive lift.

A useful equivalent is the **chain decomposition** over channels:

$$
I(\xi\_t; h\_{\gt t}) = \sum\_{k=1}^{K} I(\xi^k\_t; h\_{\gt t} \mid \xi^{\lt k}\_t).
$$

Each channel should add the maximal **conditional** mutual information about the future *given the channels already in place*. The $(k+1)$-th channel earns its keep only if it contributes a non-trivial new conditional MI term — and that's exactly the condition the current K = 4 design fails to enforce explicitly.

---

## 3. Why raw EMAs hit an early bottleneck

Each EMA at decay $\alpha$ is a **first-order causal low-pass filter** with transfer function

$$
H\_\alpha(z) = \frac{1-\alpha}{1-\alpha z^{-1}},
$$

cutoff frequency $\omega\_c \approx 1-\alpha$, and impulse response that monotonically decays from the present. The K channels at $\alpha \in \{0.0, 0.5, 0.9, 0.99\}$ therefore cover **nested** passbands:

```text
  ω = 0   (paragraph)        ω = 0.5         ω = 1.0   (token)
   |---------- channel 0 (α=0.00, bw=1.00) -----------|
   |--- channel 1 (α=0.50, bw=0.50) ---|
   |- channel 2 (α=0.90, bw=0.10) -|
   |ch 3 (α=0.99, bw=0.01)|
```

Three structural consequences:

1. **The low-frequency (paragraph-scale) content is in every channel.** Every EMA passes DC and near-DC components.
2. **The short-window content is in channels 0–2.** Channel 3 (α=0.99) is the only one that suppresses it.
3. **Pairwise mutual information is large.** Writing $\tau\_k = 1/(1-\alpha\_k)$ for the effective horizon, the pairwise MI between two EMA channels is on the order of $\log(\tau\_{\min} / \tau\_{\max})$ nats, where $\tau\_{\min}$ and $\tau\_{\max}$ are the smaller and larger horizons. For adjacent pairs in the current K = 4 grid (e.g., channels 0 and 1, channels 1 and 2), this redundancy is substantial — see §5.4 for the precise formula.

This is the early bottleneck: **K nominally-independent channels measured on the IB objective are far less than 4× the information of a single channel**. The "effective K" is much smaller than 4. At K = 6 or K = 8 with the same nested-low-pass design, the effective K saturates even faster because each new channel mostly duplicates existing information.

---

## 4. Three layered fixes

Three concrete redesigns, presented in increasing sophistication. They are layered: each can be applied independently of the others, and they compose.

### 4.1 Fix 1 — Band-pass differencing (architectural; ~30 LoC, no new params)

Replace each channel's output with the **difference** between its EMA and the next-shorter EMA:

$$
\xi^k\_t = \mathrm{EMA}\_{\alpha\_k}(h)\_t - \mathrm{EMA}\_{\alpha\_{k-1}}(h)\_t, \qquad \alpha\_0 = 0.
$$

This converts the nested low-pass bank into a **band-pass / Laplacian-pyramid** decomposition. Channel $k$ now responds only to frequencies in the disjoint band $\bigl[(1-\alpha\_k), (1-\alpha\_{k-1})\bigr]$.

**Why it helps.** Disjoint passbands implement orthogonal frequency-domain decomposition. Under the standard signal-processing model that inter-band correlation is low, the band-pass channels have $I(\xi^j\_t;\xi^k\_t) \approx 0$ for $j \neq k$, so the chain-rule decomposition of §2 collapses to

$$
I(\xi\_t; h\_{\gt t}) \approx \sum\_{k=1}^{K} I(\xi^k\_t; h\_{\gt t}),
$$

i.e. each channel contributes its full marginal MI rather than a redundancy-discounted fragment.

**Why it is essentially free.** Same parameter count, same EMA forward computation, plus one subtraction per channel per token. The autograd graph is unchanged in structure (still causal under `h.detach()`), and the implementation is one line per channel.

This is the discrete-time analogue of a Laplacian pyramid in image processing, and has a well-known L²-decorrelation guarantee under Gaussian assumptions.

### 4.2 Fix 2 — Log-spaced α with one tunable hyperparam

Pick $\{\alpha\_k\}$ on a **geometric grid** so the effective horizons $\tau\_k = 1/(1-\alpha\_k)$ are log-spaced:

$$
\tau\_k = \tau\_{\max}^{k/(K-1)}, \qquad \alpha\_k = 1 - 1/\tau\_k.
$$

Sample grids:

| K | $\tau\_{\max}$ | horizons $\tau\_k$ | $\alpha\_k$ |
|---:|---:|---|---|
| 4 | 100 | 1, 4.6, 21.5, 100 | 0, 0.78, 0.954, 0.99 |
| 6 | 200 | 1, 2.9, 8.4, 24.5, 71.5, 200 | 0, 0.66, 0.88, 0.96, 0.986, 0.995 |
| 8 | 200 | 1, 2.1, 4.2, 8.7, 18, 37, 75, 200 | 0, 0.52, 0.76, 0.885, 0.945, 0.973, 0.987, 0.995 |

**Why log-spacing is the right prior.** If the relevant context length is a priori unknown but expected to be scale-invariant up to the corpus's effective Markov order — which holds approximately for natural language up to paragraph scale — log-spacing minimises the worst-case expected redundancy across the prior. Specifically, the expected pairwise mutual information $\mathbb{E}\_p\left[I(\xi^j; \xi^k)\right]$ under a log-uniform prior $p(\tau) \propto 1/\tau$ on the relevant horizon $\tau$ is minimised when $\{\alpha\_k\}$ are themselves log-spaced.

**Why it is also free.** Five lines of code; one additional hyperparameter $\tau\_{\max}$ that can be swept cheaply. Compatible with Fix 1 (apply log-spacing first, then band-pass differencing) and with the existing learnable-α path (use the log-spaced grid as initialisation, let the optimiser refine).

### 4.3 Fix 3 — Decorrelation regulariser on the learned ξ channels

Add a soft InfoMax-style penalty during training:

$$
\mathcal{L}\_{\text{total}} = \mathcal{L}\_{\text{LM}} + \lambda \sum\_{j \neq k} \overline{\mathrm{corr}\left(\xi^j\_t, \xi^k\_t\right)^2},
$$

where the overline denotes time-averaging over the training batch and $\mathrm{corr}$ is the per-feature-dimension Pearson correlation between the two channel outputs. The penalty drives the channels apart in representational space, discouraging redundancy at the **learned** level rather than only at the architectural level.

**Why it is complementary to Fixes 1 + 2.** Fixes 1 and 2 reduce redundancy *by construction* in the linear regime. Fix 3 reduces it *empirically* in the post-nonlinearity, post-training regime where the model's actual ξ-conditional flow may rediscover correlations that the linear analysis of Fix 1 ignored. This is the same regulariser family used by VICReg and Barlow Twins for self-supervised representation learning, applied here at channel granularity.

**Tuning.** $\lambda \sim 10^{-3}$ is a sensible starting point; small enough not to fight the LM loss, large enough to whiten the channel covariance over training. If the bandpass version of K = 4 (Fix 1 + Fix 2) already lands near the IB-optimal, $\lambda$ can be set to zero; treat Fix 3 as insurance against mis-spec in Fixes 1 and 2.

---

## 5. The foundational answer — HiPPO / S4

The **provably optimal** answer to "what is the best K-dimensional causal projection of the past?" is HiPPO (Gu, Dao, Ermon, Rudra, Re 2020). HiPPO is not a different architecture from "K parallel context channels" — it is the **principled** version of the same architecture. The current K-EMA bank is the diagonal restriction of HiPPO; HiPPO-LegT replaces the diagonal restriction with a structured non-diagonal coupling matrix that achieves a quantifiable, lower-bounded improvement in mutual information per channel. The next seven subsections derive the relationship in enough detail to (a) make the improvement precise, and (b) show how to embed HiPPO inside SPLM's conservative-flow integrator without breaking the Lagrangian framework.

### 5.1 The continuous-time online projection problem

At time $t$, define the past trajectory $f(s) := h\_s$ for $s \le t$ (the sequence of hidden states up to now). We want a $K$-dimensional summary $c(t) \in \mathbb{R}^K$ that is the **best $K$-dim approximation of $f$** under some choice of "what's important to remember" measure $\mu\_t$ supported on $(-\infty, t]$:

$$
c(t) = \arg\min\_{c \in \mathbb{R}^K} \int\_{-\infty}^{t} \left( f(s) - \sum\_{k=0}^{K-1} c\_k \phi\_k(s, t) \right)^2 d\mu\_t(s),
$$

for a chosen basis of K functions $\{\phi\_k(s, t)\}\_{k=0}^{K-1}$ on the past axis. The choice of measure $\mu\_t$ encodes the prior over which past tokens matter:

- **HiPPO-LegT (translated Legendre):** $\mu\_t = \mathbb{1}[t-\theta \le s \le t] ds$, a uniform measure on a *fixed window* of length $\theta$ ending at $t$. This corresponds to "remember the last $\theta$ tokens equally well; forget everything older."
- **HiPPO-LegS (scaled Legendre):** $\mu\_t = \mathbb{1}[0 \le s \le t] ds / t$, a uniform measure on the *entire history* but rescaled so that recent and distant tokens are weighted equally on a normalised time axis. This corresponds to "remember the whole past at scale-invariant resolution."

The fundamental theorem of HiPPO (Gu et al. 2020, Theorem 1) is that **for either choice of $\mu\_t$**, the optimal coefficients $c(t)$ satisfy a linear ordinary differential equation:

$$
\frac{d c(t)}{d t} = A(t) c(t) + B(t) f(t),
$$

where $A(t) \in \mathbb{R}^{K \times K}$ and $B(t) \in \mathbb{R}^{K}$ are **structured** matrices determined by the basis $\{\phi\_k\}$ and the measure $\mu\_t$. For LegT, $A$ and $B$ are constant in $t$ (translation-invariant). For LegS, $A$ and $B$ have a $1/t$ rescaling.

### 5.2 The optimal basis is Legendre, not exponential

For uniform measure on $[-1, 1]$, the basis that achieves the optimum is the orthonormal family of Legendre polynomials $\{P\_k\}\_{k \ge 0}$ satisfying

$$
\int\_{-1}^{1} P\_j(s) P\_k(s) \mathrm{d}s = \frac{2}{2k+1} \delta\_{jk}.
$$

The K-dim Legendre projection of $f$ has reconstruction error

$$
\inf\_{c \in \mathbb{R}^K} \bigl\lVert f - \textstyle\sum\_{k=0}^{K-1} c\_k P\_k \bigr\rVert\_{L^2(\mu)}
$$

that decays as $K^{-2r}$ if $f$ is $r$-times differentiable — **this is the optimal rate for any K-dim basis**. No other K-dim basis can do better in the worst case; this is a direct consequence of Parseval's identity and the completeness of the Legendre family.

**Why exponentials are sub-optimal.** The K-EMA bank corresponds to the basis

$$
\phi\_k(s, t) = \sqrt{2 r\_k} e^{-r\_k (t-s)},
\qquad r\_k = 1/\tau\_k,
$$

(unit-norm under uniform measure on $(-\infty, t]$). This is a basis of $K$ exponentials at distinct rates $\{r\_k\}$. Two important properties:

1. The exponentials **are not orthogonal** under uniform Lebesgue measure on $(-\infty, t]$:

$$
\langle \phi\_j, \phi\_k \rangle = \int\_0^{\infty} \sqrt{2 r\_j} e^{-r\_j \tau} \sqrt{2 r\_k} e^{-r\_k \tau} d\tau = \frac{2 \sqrt{r\_j r\_k}}{r\_j + r\_k}.
$$

This inner product is in $(0, 1]$ for any $r\_j, r\_k \gt 0$, with equality at $1$ only when $r\_j = r\_k$.

2. The K-exponential basis is **not complete** for general $f \in L^2(\mathbb{R}\_+)$: there are smooth functions of compact support that have zero projection onto every finite K-exponential basis but are not zero. The Legendre family is complete (Stone–Weierstrass).

So the K-EMA representation of $f$ has structurally higher reconstruction error than the K-Legendre (HiPPO) representation, **for every K and every $f$**.

### 5.3 K-EMA is the diagonal restriction of the HiPPO ODE

If we restrict the HiPPO matrix $A$ to be **diagonal**, $A = -\mathrm{diag}(r\_0, r\_1, \ldots, r\_{K-1})$, the ODE becomes

$$
\frac{d c\_k(t)}{d t} = -r\_k c\_k(t) + b\_k f(t),\qquad k = 0, \ldots, K-1,
$$

which integrates (with $c\_k(-\infty) = 0$) to

$$
c\_k(t) = b\_k \int\_{-\infty}^{t} e^{-r\_k(t-s)} f(s) ds.
$$

That is **exactly the K-EMA bank**, with $b\_k$ a per-channel input scaling. The diagonal restriction means the channels evolve **independently**: the update for $c\_k$ at time $t$ uses only $c\_k(t-\Delta t)$ and $f(t)$, with no coupling to $c\_j$ for $j \neq k$.

The cost of the diagonal restriction is loss of representational power: a diagonal $A$ cannot generate a non-exponential basis. By contrast, HiPPO-LegT has an explicit non-diagonal $A$ that produces the Legendre basis exactly:

$$
A^{\text{LegT}}\_{nk} = -\sqrt{(2n+1)(2k+1)} \cdot \sigma\_{nk}, \qquad B^{\text{LegT}}\_n = \sqrt{2n+1},
$$

where $\sigma\_{nk} = 1$ if $n \ge k$ and $\sigma\_{nk} = (-1)^{n-k}$ if $n \lt k$.

The off-diagonal entries are what couple polynomial orders and produce the Legendre family from the linear-time-invariant ODE. Setting them to zero (the diagonal restriction) collapses the basis from Legendre to a sum of exponentials, with the redundancy properties derived in §5.2.

### 5.4 The mutual information gap — a quantitative bound

To convert the basis-mismatch into an information-theoretic statement, assume the past trajectory $f$ is wide-sense stationary with autocorrelation $R\_f(\tau) = \mathbb{E}[f(s) f(s+\tau)]$. Under the additional Gaussian assumption (locally accurate for trained-model hidden states by central limit arguments), the channel coefficients $c\_k(t)$ are jointly Gaussian with covariance

$$
\mathrm{Cov}(c\_j, c\_k) = \int\_0^{\infty}\int\_0^{\infty} \phi\_j(\tau) \phi\_k(\tau') R\_f(\tau - \tau') d\tau d\tau'.
$$

For two unit-norm exponential channels with $R\_f(\tau) = \delta(\tau)$ (white-noise simplification), the cross-correlation reduces to the basis inner product itself:

$$
\rho\_{jk} := \mathrm{corr}(c\_j, c\_k) = \frac{2 \sqrt{r\_j r\_k}}{r\_j + r\_k} = \mathrm{sech}\left( \tfrac{1}{2} \log(r\_j / r\_k) \right).
$$

(Note: $\rho\_{jk}$ depends only on the **ratio** $r\_j / r\_k$, not on the absolute decay rates — so it is purely a function of how widely separated the channels are in log-horizon space.) This gives the precise mutual information between two EMA channels:

$$
I(c\_j; c\_k) = -\tfrac{1}{2} \log \left( 1 - \rho\_{jk}^2 \right) = -\tfrac{1}{2} \log \left( \tanh^2\left( \tfrac{1}{2} \log(r\_j / r\_k) \right) \right).
$$

For the current K = 4 vanilla grid with effective horizons $\tau \in \{1, 2, 10, 100\}$:

| pair | $\tau\_j, \tau\_k$ | $\rho\_{jk}$ | $I(c\_j; c\_k)$ (nats) |
|---|---|---:|---:|
| (0, 1) | (1, 2)   | 0.943 | 1.075 |
| (0, 2) | (1, 10)  | 0.575 | 0.198 |
| (0, 3) | (1, 100) | 0.198 | 0.020 |
| (1, 2) | (2, 10)  | 0.745 | 0.448 |
| (1, 3) | (2, 100) | 0.281 | 0.041 |
| (2, 3) | (10, 100)| 0.575 | 0.198 |
| **sum** | | | **1.98 nats** |

Compare to the HiPPO-LegT bound: by orthogonality of the Legendre basis, $\rho\_{jk}^{\text{LegT}} = 0$ for all $j \neq k$, so $I(c\_j^{\text{LegT}}; c\_k^{\text{LegT}}) = 0$.

Quantitatively, this means: **at the K = 4 vanilla configuration, the K-EMA bank wastes $\approx 2.0$ nats of channel-pair redundancy that HiPPO-LegT would not waste.** Because total predictive MI is upper-bounded by $H(c) = K \cdot \log(\sigma\_c \sqrt{2\pi e})$ minus the total correlation $\mathrm{TC}(c) = \sum\_k H(c\_k) - H(c) = -\tfrac{1}{2} \log \det R$, the redundancy directly subtracts from the model's effective predictive capacity.

For the K = 4 grid above, $\det R \approx 0.078$, so $\mathrm{TC} \approx 1.27$ nats. **The K = 4 EMA bank delivers approximately $K - \mathrm{TC}/\log(2) \approx 4 - 1.84 \approx 2.16$ effective independent channels of information** — i.e. only ~54 % of nominal capacity. HiPPO-LegT delivers the full $K = 4$ effective channels.

### 5.5 The K-scaling of the gap

The redundancy gap **widens with K**. For log-spaced $\tau\_k = \tau\_{\max}^{k/(K-1)}$, the pairwise correlation between adjacent channels is

$$
\rho\_{k, k+1} = \mathrm{sech}\left( \tfrac{1}{2} \log\left( \tau\_{\max}^{1/(K-1)} \right) \right) = \mathrm{sech}\left( \frac{\log \tau\_{\max}}{2(K-1)} \right).
$$

As $K$ grows, the spacing between adjacent channels shrinks, so $\rho\_{k,k+1} \to 1$ and $I(c\_k; c\_{k+1}) \to \infty$. The total correlation $\mathrm{TC}$ scales like $K \log K$ in the dense limit, while the nominal capacity is only $K \log K$ — meaning that **for sufficiently large $K$, the K-EMA bank's effective capacity saturates at a constant**, regardless of how many channels are added. This is the fundamental information bottleneck §1 named.

By contrast, for HiPPO-LegT the effective capacity grows linearly in $K$ (orthogonal channels), so the gap $\mathrm{TC}\_{\text{EMA}} - \mathrm{TC}\_{\text{LegT}} = \mathrm{TC}\_{\text{EMA}}$ grows with $K$. This is the precise sense in which "more channels matter more under HiPPO than under EMA": the marginal value of the $(k+1)$-th channel asymptotes to zero under EMA but stays at $\log 2$ nats per channel under LegT.

### 5.6 Discrete-time HiPPO update for SPLM

SPLM's `integrate()` method runs in discrete steps of $\Delta t = 1$ token. The continuous HiPPO ODE $\dot c = A c + B f$ is discretised most accurately by the **bilinear (Tustin) transform**:

$$
c\_{t+1} = \left( I - \tfrac{1}{2} A \right)^{-1} \left[ \left( I + \tfrac{1}{2} A \right) c\_t + B h\_t \right].
$$

For HiPPO-LegT, $A$ is a fixed $K \times K$ matrix; the inverse $(I - A/2)^{-1}$ is precomputed once and stored. Each token requires one $K \times K$ matrix-vector product to update $c$, costing $K^2 d$ ops vs the EMA's $K d$. For $K = 4$, $K = 8$ this is negligible (4× and 8× respectively, on top of an already cheap operation, on a sub-1% slice of total compute).

Equivalent factorisation that avoids a runtime inverse: precompute $\tilde A = (I - A/2)^{-1}(I + A/2)$ and $\tilde B = (I - A/2)^{-1} B$, then

$$
c\_{t+1} = \tilde A c\_t + \tilde B h\_t.
$$

Both $\tilde A$ and $\tilde B$ are constants of the architecture, just like the K decay rates $\{\alpha\_k\}$ are constants in K-EMA. They become **non-learnable** initialisation values (or weakly learnable, if we want to optimise the structured $A$ directly via parametric perturbation, as S4 does).

### 5.7 Embedding HiPPO inside SPLM's Lagrangian framework

The SPLM conservative-flow argument requires three properties of the context summary $\xi\_t$:

1. **Causality.** $\xi\_t$ depends only on $\{h\_s : s \le t\}$. → Satisfied by HiPPO trivially: the ODE is one-sided.
2. **Differentiability of $V\_\theta(\xi, h)$.** $V\_\theta$ takes $\xi$ as input alongside $h$ and produces a scalar potential. → Satisfied: $c \in \mathbb{R}^K$ is a valid input to a feed-forward $V\_\theta$ (just like the current $\xi \in \mathbb{R}^d$ is, with an architectural change to accept $K$-dimensional $c$ instead of $d$-dimensional EMA-mean $\xi$).
3. **Compatibility with the leak fix.** The autograd path from $\xi$ back to the hidden states $\{h\_s\}$ must be severed at training time. → Satisfied: replace $h\_t$ with $\mathrm{stop}(h\_t)$ in the discrete update of §5.6:

$$
c\_{t+1} = \tilde A c\_t + \tilde B \cdot \mathrm{stop}(h\_t),
$$

where $\mathrm{stop}(\cdot)$ is the stop-gradient operator (`h.detach()` in PyTorch). This is the exact analogue of the K-EMA fix in the bug doc §5: the stop-gradient severs the anti-causal autograd channel from $c$ back into $h$, so the trained $V\_\theta$ cannot route prediction signal through the future via $c$. The structural argument is identical; only the basis is different.

The Euler–Lagrange equation governing the SPLM dynamics is

$$
m \ddot{h}\_t + \gamma \dot{h}\_t = -\nabla\_h V\_\theta(c\_t, h\_t),
$$

with $c\_t$ now the HiPPO state instead of the EMA mean. The conservative-flow geometry (the $-\nabla V\_\theta$ form of the right-hand side) is preserved. The only change is that the gradient $\nabla\_h V\_\theta$ is computed against a **richer**, **less-redundant** context summary — directly increasing the predictive information available per gradient step.

### 5.8 Cost-benefit summary

| aspect | K-EMA (current) | HiPPO-LegT | HiPPO-LegS | S4 |
|---|---|---|---|---|
| state-transition $A$ | diagonal $-\mathrm{diag}(r\_k)$ | structured constant | structured time-varying | structured constant + low-rank |
| basis | exponentials $e^{-r\_k \tau}$ | translated Legendre | scaled Legendre | DPLR-decomposed Legendre |
| pairwise channel correlation | $\rho\_{jk} = 2\sqrt{r\_j r\_k}/(r\_j + r\_k)$ | $\rho\_{jk} = 0$ | $\rho\_{jk} = 0$ | $\rho\_{jk} \approx 0$ |
| effective capacity at K = 4 | ~2.2 channels | 4 channels | 4 channels | 4 channels |
| effective capacity scaling | saturates as $K \to \infty$ | linear in K | linear in K | linear in K |
| per-token cost | $O(K d)$ | $O(K^2 d)$ | $O(K^2 d)$ | $O(K d \log d)$ via FFT |
| training cost (T tokens, K state) | $O(T K d)$ | $O(T K^2 d)$ | $O(T K^2 d)$ | $O(T K d \log T)$ FFT-based |
| optimisation difficulty | trivial (K scalars) | moderate ($A$ is fixed; affine warp learnable) | moderate (time-varying) | hard (DPLR conditioning) |
| theoretical optimality | none | optimal under fixed-window prior | optimal under scale-invariant prior | matches LegT/LegS |
| compatibility with SPLM `integrate()` | trivial (current code) | direct (matrix update + stop-gradient) | direct (one extra time-rescale) | needs FFT-based path |
| compatibility with leak fix | yes | yes | yes | yes |

The **net thesis**, supported by §§5.1–5.7:

> Replacing the K-EMA bank with HiPPO-LegT in the SPLM `integrate()` method recovers approximately $\mathrm{TC}/\log 2 \approx 1.8$–$3.0$ effective channels at K = 4–8, costs $\le 1\%$ of total compute, requires $\sim 50$ additional lines of well-defined code, and is fully compatible with both the SPLM Lagrangian framework and the causal-leak fix. It is the principled extension of the multi-channel ξ design and the natural target for §A1 of the restructured paper as a "future-work but architecturally trivial" upgrade path.

The argument generalises: the same analysis with LegS basis gives the same conclusion under a scale-invariant prior over horizons, with the additional benefit that the time-varying $A(t)$ adapts smoothly as the sequence length grows. For TinyStories (variable-length stories, mostly < 1024 tokens), LegT with $\theta = 200$ likely matches the corpus structure better; for longer-context corpora, LegS becomes the natural choice.

---

## 6. Theoretical lower bound on K

If the corpus has effective Markov order $M$ — the number of past tokens whose conditional contribution to the next-token distribution is non-negligible — then K channels with log-spaced horizons need

$$
K \geq \log\_2 M
$$

to span the relevant horizon range without leaving an uncovered gap, **assuming each channel covers a horizon scale roughly $2\times$ the previous one** (the natural log-spacing of Fix 2).

**Empirical $M$ estimates for our corpora:**

| corpus | sentence-coherence $M$ | story-coherence $M$ | implied lower-bound K |
|---|---:|---:|---:|
| Tiny Shakespeare | ~30 | ~150 | ~7 |
| TinyStories | ~50 | ~200 | ~8 |

These are rough estimates derived from empirical autocorrelation decay of token-level surprisal (taking $M$ as the lag at which the surprisal autocorrelation drops below $0.05$). They suggest:

- **K = 4 is near-borderline for TinyStories.** With log-spaced horizons {1, 4.6, 21.5, 100}, the longest channel covers paragraph scale only weakly; story-level (~200 tokens) is essentially uncovered.
- **K = 8 is the natural ceiling at this corpus.** Adding more channels beyond K $\approx$ $\log\_2(200) \approx 7.6$ provides diminishing marginal coverage; the architecture is limited by what the corpus actually contains, not by what we can add.
- **The lower bound is necessary, not sufficient.** $K \ge \log\_2 M$ is required to *cover* the horizon range; it does not guarantee that the model *uses* all K channels or that they are non-redundant. Fixes 1–3 are still needed to convert coverage into usable predictive information.

This is one of the cleaner quantitative arguments for moving to K = 8 with log-spaced α, conditional on the K = 4 result tomorrow being non-degenerate.

---

## 7. Recommended order of implementation

Sorted by return-on-effort, taking into account the experimental queue in [`Restructuring_paper_v3_after_causal_leak_bug.md`](Restructuring_paper_v3_after_causal_leak_bug.md) §5:

| step | change | implementation cost | compute cost | gating condition |
|---|---|---|---|---|
| **R1** | Fix 1 (band-pass differencing) at K = 4 | ~30 LoC, no new params | ~5 h MPS for one extra pilot | as soon as K = 4 vanilla baseline lands |
| **R2** | Fix 2 (log-spaced α + $\tau\_{\max}$) at K = 4, bundled with R1 | ~5 LoC | bundled with R1 | always do alongside R1 |
| **R3** | Fix 3 (decorrelation regulariser, $\lambda \sim 10^{-3}$) at K = 4 | ~10 LoC + small $\lambda$ sweep | ~5 h MPS for one pilot at chosen $\lambda$ | only if R1 + R2 plateau |
| **R4** | K = 8 with log-spaced α + Fix 1 (no Fix 3) | ~5 LoC config change | ~5–7 h MPS for one pilot | only if K = 4 (with R1 + R2) lands at val_ppl ≪ 32 |
| **R5** | K = 8 with R1 + R2 + R3 (full stack at fixed K = 8) | minor | ~5–7 h MPS | only if R4 shows lift over K = 4 R1+R2 |
| **R6** | HiPPO-LegT replacement of EMA bank | substantial (new module) | ~10 h MPS for first working pilot | future work; out of scope for paper_v3 restructure |

**Net.** R1 + R2 are the immediate, near-zero-cost wins. They convert "K = 4 with hand-picked decays" into "K = 4 with band-pass differencing on a log-spaced horizon grid", which is information-theoretically motivated and fits in a single new pilot run alongside the §5 P1 pipeline. R4 (K = 8) becomes attractive only if the K = 4 R1 + R2 baseline still has predictive headroom. R6 (HiPPO/S4) is a separate research direction worth its own paper-level treatment.

---

## 8. What this document will track in subsequent revisions

Open threads — to be filled in as experiments and analysis land:

1. **Empirical IB measurement.** Once R1 and R2 land, measure $I(\xi^k\_t; h\_{\gt t})$ per channel using a simple variational MI estimator (e.g., InfoNCE) on the leak-corrected ckpt. The chain-rule decomposition of §2 says the per-channel **conditional** MI should be roughly equal under the band-pass design; a sharp drop at some channel index $k^\ast$ would be evidence that K beyond $k^\ast$ is redundant for *this* corpus.
2. **Pairwise channel correlation curves.** Empirical pairwise $\mathrm{corr}(\xi^j\_t, \xi^k\_t)$ as a function of training step, under (a) vanilla K = 4, (b) R1 K = 4, (c) R1 + R2 K = 4, (d) R1 + R2 + R3 K = 4. The expected ordering is monotonically decreasing redundancy.
3. **K-sensitivity sweep.** R1 + R2 at K $\in$ \{4, 6, 8\} on the same compute budget. Record val PPL vs K — the saturation point empirically locates the corpus's effective Markov order $M$ from §6 (the K beyond which val PPL plateaus).
4. **Decay learning trajectories under the fix.** Re-examine the $\alpha\_k$ drift under the leak-free integrator; under a clean setup, $\alpha\_k$ should drift toward genuine corpus-relevant horizons rather than toward leak-exploitation. The drift pattern is itself diagnostic of which channels are "doing real work."
5. **Cross-corpus generalisation.** Repeat R1 + R2 on Tiny Shakespeare to test whether the principle generalises across the two corpora used in paper_v3, and whether the empirical $M$ from §6 rank-orders the K-saturation thresholds correctly.
6. **HiPPO-LegT prototype implementation.** §5.6 and §5.7 already give the discrete update form (Tustin-discretised matrix update with stop-gradient on $h\_t$) and the embedding inside the SPLM Lagrangian. The remaining work is a concrete prototype: a `model_hippo.py` module that (a) constructs the $K \times K$ matrix $A^{\text{LegT}}$ from §5.3 once at init time, (b) precomputes $\tilde A$ and $\tilde B$, (c) runs the matrix update inside `integrate()` in place of the current EMA bank, (d) registers under the `causal_force` flag so the leak fix applies. Smoke-test against §5.4's empirical predictions: pairwise $\mathrm{corr}(c\_j, c\_k)$ on a trained ckpt should be near zero (within batch-size noise), confirming the basis-orthogonality result holds in practice.
7. **Empirical HiPPO-vs-EMA comparison.** Run a 4000-step pilot of HiPPO-LegT at K = 4 with the same hyperparams as the leak-corrected K-EMA pilot. The §5.4 quantitative bound predicts a $\mathrm{TC}/\log 2 \approx 1.84$-channel improvement in effective capacity → expected val PPL improvement of $\sim$1–3 PPL over leak-corrected K-EMA at fixed K = 4 (modest because total predictive MI is dominated by the leading channels; the gap widens at larger K). If the observed gap is materially larger or smaller, that's itself a useful constraint on the Gaussian-stationary modelling assumption that underlies §5.4.

---

## 9. References and prior art

- **Information Bottleneck.** Tishby, Pereira, Bialek (1999). *The information bottleneck method.*
- **VICReg / Barlow Twins.** Bardes, Ponce, LeCun (2022); Zbontar et al. (2021). Decorrelation regularisers for self-supervised representations.
- **HiPPO.** Gu, Dao, Ermon, Rudra, Re (2020). *HiPPO: Recurrent Memory with Optimal Polynomial Projections.*
- **S4 / S5 / Mamba.** Gu et al. (2021), Smith et al. (2023), Gu and Dao (2024). Structured state-space sequence models.
- **Laplacian pyramid.** Burt and Adelson (1983). The classic multi-resolution band-pass decomposition this document's Fix 1 derives from in discrete time.
- **Multi-scale RWKV / time-decay attention.** Peng et al. (2023). Practical multi-channel decay design choices in production sequence models.

---

*End of initial draft. Subsequent revisions should append empirical results, ablation tables, and any architectural pivots motivated by the K = 4 fixed-pilot outcome and downstream R1–R5 experiments.*

---

## 10. R6 prototype — HiPPO/S4 inside the SPLM Lagrangian framework (added 2026-05-02)

A leak-free prototype of §5.6–§5.7 has been implemented and smoke-tested.
This section records the design as built, the verification probes, and the
open follow-up.

### 10.1 Files

- **Module.** [`notebooks/conservative_arch/multixi/model_multixi_hippo.py`](../notebooks/conservative_arch/multixi/model_multixi_hippo.py) —
  `make_hippo_legt`, `make_hippo_legs`, `bilinear_discretize`, the
  `MultiChannelHiPPO` module, the `SPLMSARFMassLNMultiHiPPOConfig`
  dataclass, and the `ScalarPotentialLMSARFMassLNMultiHiPPO` model class
  (subclass of `ScalarPotentialLMSARFMassLN`).
- **Trainer.** [`notebooks/conservative_arch/scaleup/train_splm_em_ln_multixi_hippo_scaleup.py`](../notebooks/conservative_arch/scaleup/train_splm_em_ln_multixi_hippo_scaleup.py) —
  hard-fork of `train_splm_em_ln_multixi_scaleup.py` with the model class
  swapped and HiPPO-specific CLI flags (`--xi-basis`, `--xi-theta`,
  `--xi-learnable-dt`).
- **Causal probe registration.** [`notebooks/conservative_arch/causal_probe.py`](../notebooks/conservative_arch/causal_probe.py)
  now registers `multixi-hippo (K=4 LegT)` ahead of the K-EMA variant so
  the class smoke and ckpt-probe routes exercise it on every run.
- **Smoke artefacts.** `notebooks/conservative_arch/scaleup/results/multihippo_smoke/`
  holds the 300-step smoke ckpt, training log, loss curve PNG, and summary.

### 10.2 Architecture as built

| component | choice | rationale |
|---|---|---|
| basis | HiPPO-LegT (default), HiPPO-LegS opt-in | LegT matches §5.7 prior for TinyStories (variable-length, mostly < 1 K tokens) |
| K (channels) | 4 (configurable) | direct parity with the K = 4 K-EMA leak-corrected pilot (§4.6 of the bug doc) |
| θ (LegT window) | 200 (configurable) | §5.7 — covers paragraph- to story-scale on TinyStories |
| LegT discretisation | bilinear (Tustin) at Δ = 1/θ | A-stable, preserves Hurwitz → Schur, exactly per §5.6 |
| LegS discretisation | backward Euler at Δ_t = 1/(t+1) | A-stable; forward Euler is **unstable for the first ≈ K tokens** because Δ_t · ‖A_0‖ exceeds 2 — confirmed empirically (max\|c\|_max blew up to 9.4 × 10⁵ before the fix; bounded at ≈ 1.06 with backward Euler) |
| LegT kernel | precomputed (T_max, K) buffer; einsum convolution | one (T, T, K) tensor, one batched matmul — ≈ 1 % of total integrator cost |
| learnable parameters added | **zero** by default; optionally `log_dt` (LegT, S4-style) | structured A and B remain non-learnable per §5.6 |
| widening of V_θ | first-layer in_dim 2·d → (K + 1)·d | identical to the K-EMA variant — total params 16,539,907 vs 16,539,911 |

### 10.3 Causal-leak compatibility

Implementation matches §5.7 verbatim:

```python
xi_input = h.detach() if cfg.causal_force else h
c        = self.xi_module(xi_input)                # HiPPO state, (B, T, K, d)
V        = self.V_theta(c, h_in).sum()
grad_V,  = torch.autograd.grad(V, h_in, create_graph=self.training,
                               retain_graph=True)
```

Verification:

| check | result |
|---|---|
| HiPPO recurrence is structurally causal (perturbing h_t leaves c_{<t} unchanged) | LegT max\|Δc_{<t}\| = **0.0e+00**; LegS max\|Δc_{<t}\| = **0.0e+00** (smoke test) |
| `causal_probe.py` class smoke, fixed mode | `multixi-hippo (K=4 LegT)`: **Δ_causal = 0.00e+00** |
| `causal_probe.py` ckpt probe on 300-step smoke ckpt | fixed Δ_causal = **0.00e+00**, after-side Δ = 1.74e-03; verdict: *"ckpt looks clean (no exploitable leak under either mode)"* |

The HiPPO state is, by construction, a strictly causal function of
`h.detach()`, so the chain-rule term `∂c_s / ∂h_t` is identically zero for
every (s, t) inside the integration loop. The Euler–Lagrange equation
holds exactly:

$$
m \cdot \ddot{h}_t = -\nabla_h V_\theta(c_t, h_t) - \gamma m \cdot \dot{h}_t.
$$

### 10.4 Smoke-test orthogonality (the §5.4 prediction)

White-noise input, K = 4, T = 1024, B = 8. We drop the first 100 time steps
(transient); pairwise Pearson correlation matrices on the remaining 924 × 8
samples:

| variant | mean off-diagonal \|corr\| |
|---|---:|
| K-EMA, α = (0, 0.5, 0.9, 0.99) | 0.481 |
| HiPPO-LegT, θ = 200             | **0.214** |
| **gain factor**                 | **2.2×** |

The gap is much smaller than the §5.4 Gaussian-stationary bound predicts
(K-EMA pair MI ≈ 1 nat → corr ≈ 0.93; HiPPO → corr ≈ 0). Three reasons:

1. White-noise input has flat spectrum so the K-EMA channels overlap
   *less* than they do on stationary autocorrelated input — the §5.4 bound
   applies in the most-correlated direction, not the spherically-averaged
   one.
2. T = 1024 with θ = 200 gives the LegT basis only ≈ 5 windows of
   "settled" projection; orthogonality is exact only in the
   continuous-time limit T → ∞.
3. Bilinear discretisation introduces a small basis distortion at finite
   Δ; the smaller Δ (= 1/θ), the closer to continuous orthogonality, but
   the longer the mixing time.

The qualitative direction is right (≈ 2× redundancy reduction) and we
expect the gain to grow at K = 6, 8 where the K-EMA grid bunches up much
more aggressively. A clean test of the §5.4 bound on actual hidden-state
trajectories is item (1) of §10.6 below.

### 10.5 Smoke training calibration

300-step smoke (batch 8, block 256, L = 8 inner integration steps,
5 M-token TinyStories, fixed_gamma = 0.30, seed 0; identical config to the
K-EMA fixed-smoke calibration referenced in the bug doc §4.3):

| variant | val PPL @ 300 | elapsed (MPS) |
|---|---:|---:|
| K-EMA (fixed integrator)                    | 151.13 | (~3 min) |
| **HiPPO-LegT (fixed integrator, θ = 200)**  | **151.00** | **3 min 8 s** |

Parity at 300 steps is the expected result: the architectural gain from
HiPPO comes from the structural decorrelation of channels, which only
materialises once V_θ has enough training to *use* the orthogonal
coefficients distinctly. The K-EMA pilot landed at val PPL = 14.78 only
after 4 K steps (bug doc §4.6), and the channel-redundancy gap should
widen further into training.

The smoke run also confirms:

- **No NaN / divergence.** Train loss falls smoothly from ≈ 9.5 (random
  init) to ≈ 5.0; grad norm stays in [0.5, 1.5] throughout.
- **Compute parity.** 188 s for 300 steps ≈ 0.63 s/step at the smoke config
  (block 256, batch 8, L = 8). At pilot config (block 512, batch 16) the
  per-step time should land near the K-EMA pilot's ≈ 8 s/step (one
  einsum-convolution dominates per integrator step, just as the K-EMA's
  K matmuls do).
- **Same parameter count.** 16,539,907 (HiPPO) vs 16,539,911 (K-EMA).
  Difference is the four learnable α_k scalars in K-EMA; HiPPO has zero
  learnable HiPPO parameters by default.

### 10.6 Recommended next experiments (as drafted, pre-pilot)

In sequence, all under the leak-free integrator (`causal_force=True`):

1. **R6.a — K = 4 HiPPO-LegT pilot, head-to-head against K-EMA pilot.**
   Same hyperparameters as the K-EMA pilot ([§4.6 of the bug doc](Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md#46-empirical-results---fixed-multi-ξ-at-4000-steps)):
   `--mode pilot --max-steps 4000 --xi-channels 4 --xi-basis legt --xi-theta 200`.
   Expected lift: §5.4's bound predicts a TC-redundancy gap of ≈ 1.84
   nats, mapping to roughly **1–3 PPL improvement over the K-EMA pilot's
   14.78** if the model can use the additional decorrelated channel
   capacity.

2. **R6.b — K = 8 HiPPO-LegT pilot, testing the §5.5 K-scaling
   prediction.** Same hyperparameters at K = 8. The K-EMA capacity
   saturates with K (§5.5); HiPPO grows linearly. The expected gap should
   *widen* at K = 8 vs K = 4 — direct test of the §5.5 prediction.

3. **R6.c — pairwise channel correlation on a trained checkpoint.** The
   §5.4 bound was derived under Gaussian-stationary assumptions on h.
   Measuring the empirical pairwise corr(c_j, c_k) on the trained R6.a
   ckpt (vs the K-EMA pilot ckpt) tests how much of the predicted MI gap
   actually carries over to learned hidden-state distributions.

4. **R6.d — LegS comparison on TinyStories.** Repeat R6.a with
   `--xi-basis legs`. LegS is scale-invariant in the horizon; LegT is
   window-fixed at θ = 200. For TinyStories (mostly < 1 K-token stories),
   §5.7 predicts LegT wins; LegS becomes natural for longer corpora. The
   relative ordering on TinyStories is itself a useful corpus diagnostic.

5. **R6.e — `--xi-learnable-dt` ablation.** Expose log(dt) as a learnable
   parameter (S4-style); compare against fixed-dt baseline at the same
   step budget. If learnable-dt helps materially, it's the natural
   stepping-stone toward a full DPLR-parametric A (i.e. the S4 layer
   proper, §5.8).

The R6 prototype is in scope for the next paper-restructure cycle as a
"future-work-but-architecturally-trivial" appendix entry, exactly as §5.8
projected; what's new compared to the §5.8 forecast is that it's now a
*ready-to-run* prototype rather than a derivation.

---

### 10.7 R6.a empirical results — K = 4 HiPPO-LegT pilot

> **Status.** Run completed 2026-05-02 15:07 EDT. Configuration:
> `--mode pilot --max-steps 4000 --xi-channels 4 --xi-basis legt --xi-theta 200 --causal-force true --seed 0`,
> 16.5 M params, 5 M-token TinyStories cap, fixed γ = 0.30, MPS,
> elapsed 7.81 h. Artefacts: `notebooks/conservative_arch/scaleup/results/multihippo_pilot_fixed/`.

**Headline.** Final val PPL = **19.82**, *behind* the K-EMA pilot's 14.78
by **34 %**. The §5.4 prediction (1–3 PPL improvement) is **falsified at
this configuration**.

**val PPL trajectory** (HiPPO-LegT vs K-EMA pilot side-by-side; full log
in `multihippo_pilot_fixed/train_stdout.log`):

| step | K-EMA pilot (§4.6) | HiPPO-LegT R6.a | gap factor |
|---:|---:|---:|---:|
|  200 | 112.60 | 137.23 | 1.22× |
|  600 |  30.15 |  37.45 | 1.24× |
| 1000 |  22.41 |  29.01 | 1.29× |
| 1400 |  19.26 |  25.45 | 1.32× |
| 1800 |  17.68 |  23.34 | 1.32× |
| 2200 |  16.62 |  *(skipped)* | — |
| 2600 |  15.39 |  20.59 | 1.34× |
| 3000 |  15.27 |  20.53 | 1.34× |
| 3400 |  15.18 |  20.40 | 1.34× |
| 3800 |  14.92 |  20.12 | 1.35× |
| **4000** | **14.78** | **19.82** | **1.34×** |

The gap is **stable throughout training**, widening modestly from 1.22×
early to 1.34× at convergence. This is not a "K-EMA fits faster early,
HiPPO catches up late" pattern — HiPPO is uniformly behind across all
4000 steps.

**Causal-violation probe** (40 K val tokens, T = 64, t_pert = 40):

| evaluator | causal-side Δ | after-side Δ |
|---|---:|---:|
| buggy | 4.56 × 10⁻² | 6.46 × 10⁻² |
| **fixed** | **0.00 × 10⁰** | 1.03 × 10⁻¹ |

Under the post-fix integrator the trained HiPPO model is exactly causal
(Δ = 0). The buggy-mode 4.56 × 10⁻² residual is the **destructive-noise
signature** documented in `Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md`
§4.6: V_θ was trained without leak access, so injecting future-position
information at inference makes prediction *worse*, not better. Mirror image
of the buggy multi-ξ ckpt's leak-exploiting signature, exactly as
expected.

**Compute parity.** HiPPO is ~16 % faster per step than K-EMA (7.03 s vs
8.32 s), so the compute-equivalent comparison is even worse: at the same
wall-clock budget, HiPPO completes ≈ 4750 steps while K-EMA reaches 4000,
i.e. HiPPO trails K-EMA *and* gets more compute to do so.

**Training was clean.** Train loss monotone-descending (9.5 → 2.92), grad
norm in [1.0, 1.7] throughout, no NaNs, no plateau-then-drift, no
instability at the LR floor. This is a real architectural negative
result, not a buggy run.

### 10.8 R6.c diagnostic — empirical pairwise channel correlation

> Tool: `notebooks/conservative_arch/multixi/diagnose_xi_channel_correlations.py`.
> Measurement: 40 960 ξ samples per channel from the integrator-final step
> on TinyStories validation, batched 8 × 512 × 10. Identical protocol on
> both R6.a HiPPO ckpt and the K-EMA pilot ckpt.

The §5.4 bound was derived under Gaussian-stationary white-noise input.
This diagnostic tests whether the orthogonality / redundancy properties
predicted there carry over to *trained-network hidden-state trajectories*.

**Empirical channel correlation matrices** (40 K val samples × d = 256
features per channel ⇒ 10.5 M scalars per channel):

K-EMA (final α = (0.000, 0.519, 0.855, 0.979); val PPL 14.78):
```
[[1.    0.874 0.605 0.394]
 [0.874 1.    0.849 0.576]
 [0.605 0.849 1.    0.821]
 [0.394 0.576 0.821 1.   ]]
```

HiPPO-LegT (θ = 200, fixed-dt; val PPL 19.82):
```
[[ 1.     0.197 -0.161 -0.024]
 [ 0.197  1.     0.179 -0.356]
 [-0.161  0.179  1.     0.502]
 [-0.024 -0.356  0.502  1.   ]]
```

**Side-by-side summary statistics:**

| metric | K-EMA pilot ckpt | HiPPO-LegT R6.a ckpt | gain factor |
|---|---:|---:|---:|
| mean \|off-diagonal corr\| | 0.687 | **0.237** | 2.9× less |
| total correlation TC(c) (nats) | 2.21 | **0.39** | 5.7× less |
| K_eff (entropy-power) / 4 | 1.98 (49 %) | **3.47 (87 %)** | +75 % |
| K_eff (§5.4 logdet form) | −2.37* | 2.88 | — |
| **val PPL @ 4000 steps** | **14.78** | 19.82 | — |

*The §5.4 logdet form (K + log det R / log 2) goes negative when R is
strongly correlated; in that regime entropy-power is the cleaner readout.

**Interpretation. The §5.4 quantitative prediction is empirically
confirmed; the §3 operational hypothesis it was assembled to support is
empirically falsified.**

The two claims are decoupled:

- **§5.4 (quantitative): the K-channel summary's effective dimension is
  larger under HiPPO-LegT than under K-EMA.** Confirmed cleanly: 3.47 vs
  1.98 effective channels — a 75 % improvement, in the predicted
  direction and roughly the predicted magnitude. The K-EMA's empirical
  redundancy (mean \|corr\| = 0.69) is even *higher* than the §5.4
  white-noise bound (0.48), because trained-network hidden states are
  heavily low-frequency dominated and the EMA bank's overlapping low-pass
  channels concentrate exactly on that band.
- **§3 (operational): channel redundancy is the bottleneck for V_θ; reducing
  it should improve val PPL.** Falsified. V_θ achieves *better* val PPL
  with the redundant K-EMA channels, not the orthogonal HiPPO ones. The
  redundancy is not the bottleneck; if anything, it's a useful prior.

**Why redundancy might be a feature, not a bug, on TinyStories.** Two
candidate mechanisms:

1. **K-EMA is an implicit low-frequency-emphasis prior.** All four EMA
   channels overlap on the paragraph-scale band. V_θ effectively gets
   four (correlated) views of the slow, story-level context — exactly
   the regime where TinyStories token-level surprisal is determined.
   HiPPO-LegT spreads its capacity evenly across orthogonal Legendre
   modes including high-order ones, which carry information about
   fast-changing within-window patterns that this corpus does not
   reward.
2. **Optimisation under correlation is easier.** Four redundant channels
   provide multiple correlated signal pathways; gradient descent can
   exploit any of them to reach a useful representation early. Four
   orthogonal channels each carry distinct information; V_θ has to learn
   to use each separately, which it apparently does not finish doing
   inside the 4 K-step budget.

Both mechanisms point in the same direction: **on TinyStories at K = 4,
the right architectural inductive bias is "many low-pass views of the
same long-horizon content", not "an orthogonal basis spanning all
horizons".**

### 10.9 What R6.a + R6.c jointly imply for the design

The three claims of §§3–5 are now empirically separable:

| claim | status after R6.a + R6.c |
|---|---|
| §3 — K-EMA is information-redundant | **confirmed by R6.c**: K_eff = 1.98 / 4 |
| §5.3 — HiPPO is the diagonal-A → structured-A generalisation | unchanged (math) |
| §5.4 — HiPPO has lower channel correlation than K-EMA | **confirmed by R6.c**: 0.24 vs 0.69 mean \|corr\| |
| §3, §5 (implied) — orthogonality ⇒ better val PPL | **falsified by R6.a**: 19.82 vs 14.78 PPL |

The right next move is therefore not "give HiPPO more knobs and try
again" but "ask why the K-EMA's redundancy is helping and isolate the
*minimal* mechanism by which K-EMA outperforms HiPPO at K = 4". The
guiding principle is **smallest-delta probes first**: each next pilot
should change exactly one independent variable from the R6.a baseline so
its result is unambiguously diagnostic.

| original | revised priority | reasoning |
|---|---|---|
| **R6.e — learnable Δ for HiPPO-LegT** | **highest priority — run next** | Smallest possible delta from R6.a: adds one learnable scalar `log_dt` per `MultiChannelHiPPO` block. Zero new architectural complexity, identical wall-clock as R6.a (≈ 7.8 h MPS at 4000 steps). Tests the single most plausible explanation for the R6.a gap — that K-EMA had four learnable `α_k` (which drifted from init in the pilot) while HiPPO at fixed θ had **zero adaptive capacity**. Result is maximally diagnostic in either direction: (a) gap closes ⇒ adaptation, not basis, was the issue; §5.4 orthogonality argument is salvaged conditional on tunability. (b) gap stays open ⇒ adaptation isn't the issue; the orthogonal Legendre basis is the wrong inductive bias for TinyStories. |
| **new R6.i — S4D learnable diagonal-complex A** | **second priority — gated on R6.e residual** | If R6.e closes only part of the gap to K-EMA, the residual is plausibly structural — the Legendre eigenvalue spectrum is the wrong basis prior for TinyStories. R6.i replaces the fixed structured-A LegT recurrence with a **diagonal complex A whose K eigenvalues are gradient-trained**, initialised from the HiPPO-LegT spectrum so the pilot is a strict generalisation of R6.a/R6.e. Adds 12 trainable scalars over R6.e (K Re + K Im + K B per-channel gains, K = 4), no other architectural change. Per-step cost is *lower* than R6.e (no matrix inverse — diagonal A makes ZOH discretisation closed-form). Maximally diagnostic of the §5.4 vs §5.5 question: if data-derived eigenvalues close the gap, the HiPPO-LegT ones were sub-optimal even after Δt-tuning; if R6.i still trails K-EMA, the issue is more structural still (diagonal A is insufficient → escalate to R6.j). See §11 for the theory and the R6.i / R6.j / R6.k hierarchy. ETA ≈ 7.5–8 h. |
| R6.b — K = 8 HiPPO-LegT | **third priority — gated on R6.e and R6.i** | §5.5 predicts the K-scaling gap *widens* with K. Run only after R6.e and R6.i jointly disambiguate the basis question, otherwise R6.b confounds K-scaling with adaptation and basis choice. ETA ≈ 8.5–9.5 h. If R6.b crosses K-EMA's K = 8 result, the K = 4 gap may be a small-K artefact. |
| R6.d — LegS basis | **deferred** | LegS at K = 4 is unlikely to reverse the sign of the gap; deferred until R6.b/R6.e/R6.i clarify the small-K story. |
| ~~new R6.f — K = 4 HiPPO-LegT with α-init-matched θ schedule~~ | **withdrawn (muddled)** | The original framing — "K log-spaced θ values, one per channel" — does not have a clean instantiation in the LegT/LegS framework. Standard LegT has a single θ that defines the window for all K Legendre orders; per-channel θ would require K parallel `MultiChannelHiPPO` blocks, which then collapses to either a constrained K-EMA (if `N=1` per block) or doubles the channel count (if `N>1`). The intended 4-cell decomposition "orthogonal vs low-pass × single vs multiple horizons" has no clean instantiation in this architecture and so does not give an unambiguous experimental signal. Withdrawn. |
| ~~new R6.g — K-EMA + Fix-1 band-pass differencing~~ | **subsumed by new R6.h below** | §4.1 alone is no longer the most informative test; the natural sweep is the full Fix-1 / Fix-2 / Fix-3 ladder against HiPPO-LegT (R6.h). |
| **new R6.h — K-EMA × Fix 1/2/3 sweep vs HiPPO-LegT** | **planned after R6.e + R6.i + R6.b** | §§4.1–4.3 propose three layered fixes for the K-EMA redundancy bottleneck: band-pass differencing (Fix 1), log-spaced α with one tunable τ_max (Fix 2), and the decorrelation regulariser (Fix 3). After R6.a's empirical surprise (K-EMA's redundancy *helps*), these fixes become a critical empirical test in their own right: do *any* of them, individually or in combination, beat both vanilla K-EMA (val_ppl 14.78) and HiPPO-LegT (val_ppl 19.82, R6.a)? Run as a 5-pilot ladder at K = 4: vanilla K-EMA (already have), Fix 2 only, Fix 1 + Fix 2, Fix 1 + Fix 2 + Fix 3 (best λ), best-of-above paired with the best HiPPO/S4D config. ETA ≈ 5 × 7.8 h ≈ 1.6 days of MPS. See §10.10 for the explicit protocol. |

The headline takeaway is sober but unambiguous: **the K-EMA bank's
val_ppl = 14.78 result was earned not in spite of its redundancy but
plausibly because of it.** The §5.4 information-theoretic argument is
mathematically correct, but the operational claim it was supporting —
"more orthogonal channels ⇒ more predictive capacity for V_θ" — does not
hold on TinyStories at K = 4 in the leak-free regime.

### 10.10 R6.h protocol — K-EMA Fix 1/2/3 sweep against HiPPO-LegT

Goal: empirically test the §4 redesigns of the K-EMA bank against both
the K-EMA baseline (val_ppl 14.78, leak-free pilot) and the HiPPO-LegT
baseline (val_ppl 19.82, R6.a). The sweep is laid out so each pilot
changes exactly one variable from the previous, allowing direct
attribution of any val_ppl movement to a specific intervention.

**Fixed config** (identical to the K-EMA pilot in
`Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md` §4.6 and to R6.a):
4000 steps, batch 16, block 512, `--fixed-gamma 0.30`, `causal_force=true`,
`--xi-channels 4`, LayerNorm-after-step, identical LR schedule, identical
seed. Only the K-EMA channel construction varies between pilots.

| pilot | K-EMA construction | extra params | new code | ETA / status |
|---|---|---|---|---|
| **R6.h.0** (have) | Vanilla K-EMA, hand-picked α = (0, 0.5, 0.9, 0.99), learnable | — | — | done; val_ppl 14.78 |
| **R6.h.1** (done 2026-05-03) | Fix 2 only — log-spaced α from §4.2, K = 4, τ_max = 100, learnable | — | ~5 LoC in `MultiChannelXi.__init__` to seed `α_k = 1 − 1/τ_max^{k/(K−1)}` | done; val_ppl 15.03 (§10.12) |
| **R6.h.2** | Fix 1 + Fix 2 — log-spaced α + band-pass differencing $\xi^k = \mathrm{EMA}_{\alpha_k} − \mathrm{EMA}_{\alpha_{k−1}}$ ($\alpha_0 = 0$) | — | ~30 LoC: subtract previous-channel EMA in `_forward_xi` post-step, ahead of the LN | ≈ 7.8 h |
| **R6.h.3** | Fix 1 + Fix 2 + Fix 3 (small λ) — add $\lambda \sum_{j \neq k} \overline{\mathrm{corr}(\xi^j_t, \xi^k_t)^2}$ to the LM loss with λ ∈ {1e−4, 1e−3, 1e−2} (one λ per pilot, picked by smoke-screen at 300 steps) | — | ~10 LoC in trainer: aggregate per-feature Pearson over the batch and add to loss | ≈ 7.8 h × ≤ 3 (smoke + 1 full pilot) |
| **R6.h.4** | Best-of-{R6.h.0..3} paired with the best HiPPO-LegT config (R6.e or R6.a, whichever wins) | — | — | ≈ 7.8 h, head-to-head |

Diagnostic per pilot: rerun `diagnose_xi_channel_correlations.py` on each
checkpoint and report `mean_off_abs_corr`, `total_correlation_nats`,
`k_eff_entropy_power`. The expected ordering under the §3 hypothesis is
that decorrelating fixes (Fix 1 in particular) move the K-EMA bank
toward HiPPO's K_eff = 3.47 / 4 region. The empirical question is
whether that movement helps or hurts val_ppl. If Fix 1 + Fix 2 lands at
val_ppl ≪ 14.78, the §3 hypothesis is correct *given the right
decorrelation operator* (band-pass) and the R6.a surprise was specific
to HiPPO's Legendre basis. If Fix 1 + Fix 2 underperforms vanilla
K-EMA, the redundancy-helps-PPL conclusion strengthens further.

**Causal-leak compatibility.** Fix 1 (band-pass differencing) and
Fix 2 (log-spaced α-init) are pure modifications inside
`MultiChannelXi.forward` after `h.detach()` is applied; they touch no
gradient pathway that would re-introduce an anti-causal autograd edge.
Fix 3 adds an extra term to the *forward* loss that depends only on the
already-detached ξ trajectory, so it is also causal-safe. All R6.h
pilots will be re-run through `causal_probe.py` before training, in
keeping with the regression-test discipline established post-bug.

### 10.11 R6.e empirical results — K = 4 HiPPO-LegT pilot with learnable Δ

> **Status.** Run completed 2026-05-02 23:42 EDT. Configuration:
> `--mode pilot --xi-channels 4 --xi-basis legt --xi-theta 200.0 --xi-learnable-dt --fixed-gamma 0.30 --seed 0 --causal-force true`.
> 16.54 M params, 5 M-token TinyStories cap, fixed γ = 0.30, MPS,
> elapsed 8.11 h. Artefacts:
> `notebooks/conservative_arch/scaleup/results/multihippo_pilot_learndt/`.

**Headline.** Final val PPL = **17.45**, beating R6.a (HiPPO-LegT
fixed Δ, 19.82) by **2.37 PPL** but still trailing the K-EMA pilot
(14.78) by **2.67 PPL**. Adding one learnable scalar (Δt) recovers
**~47 % of the R6.a → K-EMA gap** but leaves a residual structural gap
that is the subject of R6.i (§11.14) and R6.h (§10.10).

**val PPL trajectory** (R6.e vs R6.a vs K-EMA pilot side-by-side; full
log in `multihippo_pilot_learndt/train_stdout.log`):

| step | K-EMA pilot | HiPPO-LegT R6.a | **HiPPO-LegT R6.e** | gap to K-EMA |
|---:|---:|---:|---:|---:|
|  200 | 112.60 | 137.23 | 144.11 | 1.28× |
|  600 |  30.15 |  37.45 |  36.44 | 1.21× |
| 1000 |  22.41 |  29.01 |  27.48 | 1.23× |
| 1400 |  19.26 |  25.45 |  23.53 | 1.22× |
| 1800 |  17.68 |  23.34 |  21.14 | 1.20× |
| 2200 |  16.62 |  *(skip)* |  19.85 | 1.19× |
| 2600 |  15.39 |  20.59 |  18.21 | 1.18× |
| 3000 |  15.27 |  20.53 |  18.08 | 1.18× |
| 3400 |  15.18 |  20.40 |  17.94 | 1.18× |
| 3800 |  14.92 |  20.12 |  17.71 | 1.19× |
| **4000** | **14.78** | **19.82** | **17.45** | **1.18×** |

R6.e tracks below R6.a from step 1000 onward and never re-crosses,
confirming a sustained lift from learnable Δt rather than an
early-training fluke. The R6.e / K-EMA gap factor narrows from 1.28×
(step 200) to 1.18× (step 4000) — same shape as R6.a's curve but
displaced downward.

**Δt evolution** (the new degree of freedom relative to R6.a):

| step | Δt | drift from init |
|---:|---:|---:|
| 0 (1/θ at θ = 200) | 0.00500 | — |
| 200  | 0.00501 | +0.2 % |
| 1000 | 0.00604 | +20.8 % |
| 2000 | 0.00879 | +75.8 % |
| 3000 | 0.01210 | +142.0 % |
| **4000** | **0.01332** | **+166.4 %** |

Δt drifts monotonically upward by **166 %** over training, settling at
0.01332. The optimiser unambiguously prefers a *coarser* discretisation
horizon than the LegT default — an effective receptive-field expansion
of 2.66×, consistent with the §10.9 hypothesis that θ = 200 is too
short-horizon for TinyStories. No basis re-discretisation is required
to use the final Δt — the structured-A LegT recurrence is preserved,
only the integration step is rescaled.

**Channel-correlation diagnostic** (40 K val tokens, identical protocol
to §10.8):

| metric | K-EMA pilot | R6.a (fixed Δ) | **R6.e (learn Δ)** |
|---|---:|---:|---:|
| mean \|off-diagonal corr\| | 0.687 | 0.237 | **0.202** |
| total correlation TC(c) (nats) | 2.21 | 0.39 | **0.32** |
| K_eff (entropy power) / 4 | 1.98 (49 %) | 3.47 (87 %) | **3.54 (88 %)** |
| K_eff (logdet form, §5.4) | −2.37 | 2.88 | **3.07** |
| **val PPL @ 4000 steps** | **14.78** | 19.82 | **17.45** |

Empirical correlation matrix (HiPPO-LegT R6.e):

```
[[ 1.000  0.081 -0.131 -0.000]
 [ 0.081  1.000  0.168 -0.323]
 [-0.131  0.168  1.000  0.510]
 [-0.000 -0.323  0.510  1.000]]
```

R6.e is **slightly more orthogonal than R6.a** (mean \|corr\| 0.20 vs
0.24, K_eff 3.54 vs 3.47): learnable Δt did not cause channels to
collapse onto each other. The 11 % improvement in K_eff coexists with a
12 % improvement in val PPL, but the design doc's §10.9 thesis
("K-EMA's redundancy is plausibly a useful inductive bias") is not
contradicted — both R6.a and R6.e remain far more orthogonal than K-EMA
and both still trail K-EMA on val PPL.

**Causal verification.** R6.e's `MultiChannelHiPPO` with
`learnable_dt=True` was registered in `causal_probe.py` and passes the
strict probe with causal-side Δ ≡ 0, identical to R6.a (§10.7 row).
Trained ckpt is therefore exactly causal under the post-fix
integrator. Expected destructive-noise signature under the buggy
evaluator (V_θ never had leak access at training time) — not
re-measured here as it would replicate the §10.7 pattern.

**Compute parity.** R6.e runtime 8.11 h vs R6.a 7.81 h (+3.8 %) — the
extra cost of one autograd backward through `log_dt` is negligible.
Step rate effectively identical to R6.a; LR schedule identical; data
identical. R6.e is a clean smallest-delta probe of R6.a, exactly as
§10.9 framed it.

**Training was clean.** Train loss monotone (10.0 → 2.86), grad norm in
[0.85, 1.40] throughout, no NaNs, no plateau-then-drift, smooth descent
to LR floor. Δt drift was monotone (no oscillation around init) — the
optimiser had a clear preference for larger Δt and pursued it without
hesitation. Real architectural lift, not a trick or a numerical
artifact.

### 10.12 R6.h.1 empirical results — K = 4 K-EMA Fix 2 (log-spaced α, τ_max = 100)

> **Status.** Run completed 2026-05-03 21:48 EDT. Configuration:
> `--mode pilot --xi-channels 4 --xi-alpha-init-mode log_spaced --xi-tau-max 100 --fixed-gamma 0.30 --seed 0 --causal-force true --tag-suffix seed0_logspaced_tau100`.
> 16.54 M params, 5 M-token TinyStories cap, fixed γ = 0.30, MPS,
> elapsed 8.09 h. Artefacts:
> `notebooks/conservative_arch/scaleup/results/multixi_pilot_logspaced_taumax100/`.

**Headline.** Final val PPL = **15.03**, **+0.25 PPL** worse than
R6.h.0 (vanilla K-EMA hand-picked α, 14.78), inside seed-to-seed
variance and far below the predicted "mild improvement, magnitude ≤ 1
PPL" threshold of §12.5. The cleanest characterisation at single-seed
pilot is *Fix 2 ≈ R6.h.0 K-EMA*, with the structural finding (below)
sharper than the val_ppl point estimate.

**val PPL trajectory** (R6.h.1 vs K-EMA pilot side-by-side; full log
in `multixi_pilot_logspaced_taumax100/train_stdout.log`):

| step | K-EMA pilot (R6.h.0) | **R6.h.1 (Fix 2)** | Δ |
|---:|---:|---:|---:|
|  200 | 112.60 | 100.36 | −12.2 |
|  600 |  30.15 |  28.72 | −1.4 |
| 1000 |  22.41 |  22.86 | +0.5 |
| 1400 |  19.26 |  19.76 | +0.5 |
| 1800 |  17.68 |  17.95 | +0.3 |
| 2200 |  16.62 |  17.02 | +0.4 |
| 2600 |  15.39 |  15.67 | +0.3 |
| 3000 |  15.27 |  15.59 | +0.3 |
| 3400 |  15.18 |  15.46 | +0.3 |
| 3600 | (not logged) |  14.89 | — |
| 3800 |  14.92 |  15.24 | +0.3 |
| **4000** | **14.78** | **15.08** | **+0.30** |
| **DONE (smoothed)** | **14.78** | **15.03** | **+0.25** |

R6.h.1 starts noticeably *ahead* of R6.h.0 at step 200 (Fix 2's wider
initialisation lets the longer-τ channels skip a warm-up phase) but
the curves cross by step 1000 and Fix 2 trails by ~0.3 PPL for the
remainder of training. The step-3600 dip to 14.89 is at-noise-level
(LR was already 1.4e-5 and only 200 steps of post-LR-floor refinement
remained); the smoothed `DONE` value of 15.03 is the right summary
statistic.

**α evolution** (the central empirical finding — **K-EMA auto-tunes
α toward a similar regime regardless of initialisation**):

| | R6.h.0 (hand-picked) | R6.h.1 (log-spaced, τ_max = 100) | |Δ| |
|---|---:|---:|---:|
| **α₀ init** | 0.000 | 0.000 | 0.000 |
| **α₀ final** | 0.000 | 0.000 | 0.000 |
| **α₁ init** | 0.500 | 0.785 | 0.285 |
| **α₁ final** | 0.519 | 0.642 | **0.123** |
| **α₂ init** | 0.900 | 0.954 | 0.054 |
| **α₂ final** | 0.855 | 0.919 | **0.064** |
| **α₃ init** | 0.990 | 0.990 | 0.000 |
| **α₃ final** | 0.979 | 0.984 | **0.005** |

The Fix 2 initialisation places α₁, α₂ at notably longer effective
time-constants than R6.h.0 (init τ_k = −1/ln α_k = {1.44, 9.49, 99.5}
for R6.h.0 vs {4.13, 21.05, 99.5} for R6.h.1, hand-picked vs
log-spaced). After 4000 steps the trained α values **shrank toward
R6.h.0's territory**: the absolute α-gap to R6.h.0 closed from 0.285
to 0.123 on α₁ (a 57 % reduction), the τ₁ effective integration
window contracted from 4.13 to 2.25 tokens (vs R6.h.0's final 1.51),
α₂ contracted only marginally (Δα = 0.054 → 0.064, the absolute gap
held while both channels drifted slightly downward), and α₃ converged
to within 0.005 of R6.h.0's final value. The dominant convergent
channel is α₁ — the channel whose initialisation was most different —
which is exactly the prediction of a *gradient pulling toward a
preferred operating point*: the further from the basin you start, the
larger the corrective force during training.

**Channel-correlation diagnostic** (40 K val tokens, identical
protocol to §10.8;
`notebooks/conservative_arch/scaleup/results/multixi_pilot_logspaced_taumax100/r6h1_channel_corr_diagnostic.json`):

| metric | R6.h.0 K-EMA | **R6.h.1 K-EMA Fix 2** | R6.a HiPPO-LegT | R6.e HiPPO learn-Δt |
|---|---:|---:|---:|---:|
| mean \|off-diagonal corr\| | 0.687 | **0.673** | 0.237 | 0.202 |
| total correlation TC(c) (nats) | 2.21 | **2.06** | 0.39 | 0.32 |
| K_eff (entropy power) / 4 | 1.98 (49 %) | **2.02 (51 %)** | 3.47 (87 %) | 3.54 (88 %) |
| K_eff (logdet form, §5.4) | −2.37 | **−1.95** | 2.88 | 3.07 |
| **val PPL @ 4000 steps** | **14.78** | **15.03** | 19.82 | 17.45 |

Empirical correlation matrix (R6.h.1):

```
[[1.000  0.802  0.520  0.385]
 [0.802  1.000  0.823  0.623]
 [0.520  0.823  1.000  0.887]
 [0.385  0.623  0.887  1.000]]
```

R6.h.1 sits **on top of R6.h.0** on every diagnostic axis: |corr| 0.67
vs 0.69 (Δ = −0.014), TC 2.06 vs 2.21 nats (Δ = −0.15), K_eff 2.02 vs
1.98 (Δ = +0.04). All four deltas are within sampling-noise of zero
relative to the R6.h.0 ↔ R6.a gap (|corr| 0.69 → 0.24, K_eff 1.98 →
3.47). Fix 2's initialisation-time channel re-spacing **did not
survive training** — the trained Fix 2 bank is operationally a
re-discovery of R6.h.0's K-EMA bank.

**Why this happens.** The K-EMA gradient signal favours channel
configurations in which adjacent EMAs share a substantial fraction of
their effective integration window — i.e. correlated, smooth,
multi-scale running averages. Fix 2's wider initialisation places α₁
and α₂ outside that preferred regime and the optimiser pulls them
back: τ₁ contracts from 4.13 toward 2.25 (effective integration window
roughly halved), τ₂ contracts from 21.05 toward 11.86 (window roughly
halved as well), while α₀ (no-EMA, identity channel) stays pinned at
zero by parameterisation and α₃ (slowest channel) settles within 0.005
of R6.h.0's final value. The trajectory is monotonic across the run
(no oscillation around init, no late re-spreading; full curve in the
training log), indicating a clean gradient preference rather than a
vanishing-update artefact. The contraction is partial — final R6.h.1
τ values are still longer than R6.h.0's final τ values — but the
*channel-correlation profile* this produces is identical to R6.h.0's
to within sampling noise, because the §5.4 analytic correlation
depends on the *ratio* of adjacent τ's, not their absolute magnitudes,
and Fix 2's contracted spacing puts those ratios within 10 % of
R6.h.0's. This is the *training-time auto-tuning* mechanism: K-EMA is
self-stabilising under SGD toward a particular redundancy regime that
appears to be a local optimum of the joint
$(V\_\theta, \alpha)$ landscape at the current $V\_\theta$ capacity.

**Causal verification.** R6.h.1's `MultiChannelXi` with
`xi_alpha_init_mode='log_spaced'` and `xi_tau_max=100` was registered
in `causal_probe.py` and passes the strict probe with causal-side Δ ≡
0, identical to R6.h.0. Trained ckpt is therefore exactly causal under
the post-fix integrator.

**Compute parity.** R6.h.1 runtime 8.09 h vs R6.h.0 baseline (≈ 7.8
h) — Fix 2's extra cost is a one-off log-space α generation in
`__init__`; per-step cost is identical to R6.h.0. Step rate, LR
schedule, data, batch size, block size, all identical to R6.h.0.

**Training was clean.** Train loss monotone (10.0 → 2.6), grad norm in
[0.60, 0.80] throughout (slightly *tighter* than R6.h.0's 0.65–0.95),
no NaNs, no plateau-then-drift, smooth descent to LR floor. Mass
distribution stable at m̄ ≈ 1.42, σ ≈ 0.19 throughout. No instabilities
attributable to the wider Fix-2 α-init.

**Implication for §12.5 predictions.** §12.5 pre-registered Fix 2 as
"mild improvement or null (Δ ≤ 0, magnitude ≤ 1 PPL)". The actual
outcome (Δ = +0.25 PPL) is **inside the magnitude bound** but on the
*opposite side of zero* from the directional prediction. This is the
weakest of the three §12.5 prediction tests in informational content —
it neither corroborates nor falsifies the §12.3 hypothesis on its
own, because Fix 2's intervention failed to move channel correlation
in any meaningful way (the |Δ corr| was 0.014, not the predicted
0.1+). The auto-tuning finding is a **new mechanism not anticipated by
§12.3**: the K-EMA bank's redundancy regime is not a free choice of
the designer but a **gradient-preferred basin** that initialisation
alone cannot escape. This refines §12.5's R6.h.2 / R6.h.3 design: a
clean test of the §12.3 hypothesis now requires an intervention that
is **robust to training-time auto-tuning**. The candidates are:

- **R6.h.2′ (frozen-α variant of Fix 1 + Fix 2):** lock α at log-spaced
  τ ∈ [1, 100], `requires_grad=False`. This forces the operational
  channel spread that R6.h.1 demonstrated cannot persist as a free
  parameter. Expected if §12.3 holds: val_ppl rises by ≥ 1 PPL because
  the gradient cannot route around the imposed orthogonality.
- **R6.h.3′ (decorrelation-regularised α variant):** retain learnable
  α but add λ · (mean off-diag |corr|)² to the LM loss for
  λ ∈ {1e−4, 1e−3, 1e−2}. This actively *opposes* the auto-tuning
  mechanism. Expected if §12.3 holds: monotone val_ppl rise in λ; the
  λ at which val_ppl rises by ≥ 0.5 PPL is the magnitude of the
  inductive-bias prior the gradient is paying to maintain.

Both redesigned variants are causal-leak-safe (they touch only post-
detach pathways) and would land at ≈ 7.8 h compute each on the same
hardware. The original R6.h.2 (Fix 1: band-pass differencing) is
preserved as a separate test of *channel structure* (independent of
auto-tuning).

---

## 11. Optimisation-based basis discovery — S4D and the R6.i family (added 2026-05-02)

§5 selected two specific points in the space of K-channel context
summaries: HiPPO-LegT (a structured A producing the orthonormal Legendre
basis) and the K-EMA bank (a diagonal restriction producing the
exponential basis). The R6.a / R6.c results showed that orthogonality
holds the §5.4 prediction (lower channel correlation, higher K_eff) but
fails the §3 operational claim (lower val PPL). The R6.e result,
expected to land partially closing the gap to K-EMA, will tell us
whether the residual gap is "missing adaptation" or "wrong basis".

This section answers the question "if the basis is wrong, can we let
the data choose a better one?" by formalising the K-channel context
summary as a search over $(A, B)$ pairs and adopting the S4 / S4D
program (Gu et al. 2021, 2022) as the principled optimisation route.
The resulting experiment is **R6.i**, drafted below as the smallest
basis-changing delta from R6.e.

### 11.1 Any $(A, B)$ defines a basis

Both the K-EMA bank and HiPPO-LegT are special cases of a single
template — a stable linear-time-invariant (LTI) ODE on a K-dimensional
hidden state $c(t)$ driven by the embedding trajectory $h(t)$:

$$
\dot c(t) = A c(t) + B h(t),
\qquad A \in \mathbb{R}^{K \times K},
\quad B \in \mathbb{R}^{K \times 1}.
$$

If $A$ is Hurwitz (all eigenvalues with negative real part), the ODE
admits the convolution solution

$$
c(t) = \int\_{-\infty}^{t} e^{A(t - s)} B h(s) ds,
$$

i.e. each component of $c(t)$ is a causal weighted average of the past,
with the weights given by the columns of the matrix exponential
$e^{A(t - s)} B$. **These columns are the K basis functions of the
context summary.** Choosing $(A, B)$ is choosing a basis. Concretely:

- **K-EMA**: $A = -\mathrm{diag}(r\_0, \ldots, r\_{K-1})$, $B = (b\_0, \ldots, b\_{K-1})^\top$. Basis = K decaying exponentials $\{\sqrt{2 r\_k} e^{-r\_k \tau}\}$.
- **HiPPO-LegT**: $A$ is the structured matrix from §5.3, $B = (\sqrt{2k+1})\_k$. Basis = K translated Legendre polynomials over a sliding window.

There is no a priori reason to prefer one basis over another except by
appeal to a target task. The §5.2 worst-case argument prefers Legendre;
the R6.a empirical result on TinyStories prefers exponentials.
Resolution: **let the optimiser choose.**

### 11.2 The S4 program — gradient-trained $(A, B)$

Gu, Goel, Re (S4, 2022) and Gu, Goel, Gu, Re (S4D, 2022) showed that
for sequence modelling the right move is to make $(A, B)$ **trainable
parameters** with a structural constraint that keeps $A$ Hurwitz.
Specifically, S4D (the simpler diagonal variant) parameterises $A$ as

$$
A = \mathrm{diag}(\lambda\_1, \ldots, \lambda\_K),
\qquad \lambda\_k \in \mathbb{C},\quad \mathrm{Re}(\lambda\_k) \lt 0,
$$

so each diagonal entry is one complex eigenvalue of the system. The
Hurwitz condition is enforced structurally in our implementation by
parameterising $\mathrm{Re}(\lambda\_k) = -\exp(\rho\_k)$ for an
unconstrained real $\rho\_k$ (`log_neg_re` in code), which is
strictly negative for any value of $\rho\_k$. The imaginary part
$\mathrm{Im}(\lambda\_k)$ is left free.

Two reasons this matters for our case:

1. **The basis becomes an output of training, not an input.** Whatever
   distribution of decay rates and oscillation frequencies the gradient
   chooses to minimise the LM loss is, by construction, the
   data-optimal K-dim diagonal-LTI basis for the corpus.
2. **K-EMA and HiPPO-LegT are both reachable.** $\mathrm{Im}(\lambda\_k) = 0$ recovers the K-EMA bank; the eigenvalues of the HiPPO-LegT $A$ are a specific complex-conjugate-pair pattern that the optimiser starts from under the recommended initialisation (§11.7).

### 11.3 Three structural choices for $A$

Diagonal complex (S4D) is one rung in a hierarchy of structural choices
for $A$. The expressiveness ladder:

| variant | $A$ structure | extra trainable params (K = 4) | reachable bases |
|---|---|---|---|
| **R6.i — S4D diagonal complex** | $A = \mathrm{diag}(\lambda\_1, \ldots, \lambda\_K)$, $\lambda\_k \in \mathbb{C}$ | 8 (4 Re + 4 Im) | all stable diagonal LTI = decaying exponentials and damped sinusoids |
| **R6.j — S4-DPLR (diagonal + low-rank)** | $A = \Lambda + P Q^\top$, $\Lambda$ diagonal complex, $P, Q \in \mathbb{C}^{K \times r}$ rank-r | 8 + 4·K·r | adds rank-r off-diagonal coupling |
| **R6.k — fully dense $A$** | unrestricted $A \in \mathbb{R}^{K \times K}$ with stability constraint | up to $K^2 = 16$ | all stable LTI |

R6.i is the cheapest, R6.j sits where S4 originally lived (DPLR is
chosen because $A^\text{LegT}$ has exactly this structure), and R6.k is
the maximally flexible variant — primarily of theoretical interest
because dense $A$ tends to be hard to keep stable under gradient
descent without an explicit projection step.

### 11.4 Why R6.i (S4D) is the right first delta

Three reasons:

1. **Smallest delta from R6.e.** R6.i replaces only the kernel-build path inside `MultiChannelHiPPO` with a closed-form S4D kernel; the SPLM integrator, the V_θ MLP, the data pipeline, and the LR / batch / step schedule are all identical to R6.e. The added trainable surface area is 12 scalars (K Re + K Im + K B-per-channel gains) over R6.e's K-EMA-equivalent + Δt parameter set.
2. **Per-step cost is *lower* than R6.e.** The S4D kernel build is a single elementwise `torch.exp` over a (K,) complex vector and an outer product to produce a (T, K) complex tensor; no matrix inverse is required (whereas R6.e's bilinear discretisation needed $(I - \frac{\Delta t}{2} A)^{-1}$, which we had to detour through CPU on MPS). Expected wall-clock ≈ R6.e or slightly faster.
3. **The S4D paper validates the parameterisation.** Gu, Goel, Gu, Re 2022 reported that diagonal complex S4D recovers ≈ all of the original S4-DPLR's empirical performance with strictly fewer parameters and no structured-matrix machinery. So the R6.j step is not strictly required — R6.i is, on prior literature, expected to be sufficient if any LTI basis is.

### 11.5 ZOH discretisation for diagonal $A$ — closed-form

Because $A$ is diagonal, the zero-order-hold (ZOH) discretisation that
takes the continuous-time ODE to a discrete-time recurrence has an
exact, inverse-free closed form. Per-channel:

$$
\bar{A}\_{kk} = e^{\lambda\_k \Delta t},
\qquad
\bar{B}\_k = \frac{e^{\lambda\_k \Delta t} - 1}{\lambda\_k} B\_k.
$$

The K complex eigenvalues commute (they live on the diagonal of $A$),
so the matrix exponential factorises into K scalar exponentials.

The discrete-time recurrence is

$$
c\_k(t+1) = \bar{A}\_{kk} c\_k(t) + \bar{B}\_k h(t),
$$

with the corresponding convolution kernel

$$
M[\Delta, k] = \bar{A}\_{kk}^{\Delta} \bar{B}\_k = e^{\lambda\_k \Delta t \cdot \Delta} \bar{B}\_k.
$$

Closed-form, autograd-friendly, no per-step matrix inverse: the entire
kernel for a length-T window is constructed in $O(T \cdot K)$ scalar
operations. Compare R6.e's bilinear path which needs an iterative
$\bar{A}$-power chain of length $T$; for $K = 4$ both are negligible,
but the S4D path is cleaner and trivially extensible to larger $K$.

### 11.6 Real-valued ξ via $2 \cdot \mathrm{Re}(c)$

For any real-valued input signal $h$ and complex eigenvalue
$\lambda\_k = \sigma\_k + i \omega\_k$ with $\sigma\_k \lt 0$, the
component $c\_k(t)$ is in general complex. To produce a real-valued
context summary $\xi^k\_t$ that V_θ can consume we use the standard
S4D convention

$$
\xi^k\_t = 2 \cdot \mathrm{Re}\bigl( c\_k(t) \bigr).
$$

Two equivalent interpretations:

1. **Conjugate-pair absorption.** A real impulse response of an LTI
   system requires complex eigenvalues to come in conjugate pairs.
   Taking $2 \mathrm{Re}(c\_k)$ is mathematically equivalent to
   summing $c\_k$ with its complex conjugate — i.e. we are folding the
   $K$-dim complex state into a $K$-dim real state by absorbing the
   "mirror" eigenvalue without doubling the parameter count.
2. **Damped-sinusoid basis.** Computing
   $2 \mathrm{Re}(e^{\lambda\_k \tau}) = 2 e^{\sigma\_k \tau} \cos(\omega\_k \tau)$
   makes the basis explicit: $K$ damped cosines of the past, with
   independently learnable decay rate $\sigma\_k$ and oscillation
   frequency $\omega\_k$. When $\omega\_k = 0$, the channel reduces to
   a pure decaying exponential — the K-EMA case. Setting all $\omega\_k = 0$
   recovers the K-EMA bank exactly (modulo a per-channel scaling
   absorbed into $B\_k$).

### 11.7 Initialisation strategies

Three principled initialisations of the $K$ eigenvalues, in order of
preference for our case:

1. **HiPPO-LegT spectrum (default).** Compute the K eigenvalues $\{\lambda\_k\}\_{k=1}^{K}$ once at init by diagonalising the LegT $A$ matrix from §5.3. This makes R6.i a strict generalisation of R6.a / R6.e — the model starts with the same continuous-time dynamics, then gradient descent is free to drift. In code: `s4d_init_legt(K)` in `model_multixi_s4d.py`.
2. **S4D-Lin** (Gu, Goel, Gu, Re 2022). Set $\mathrm{Re}(\lambda\_k) = -1/2$ for all $k$ and $\mathrm{Im}(\lambda\_k) = \pi (k + 1/2)$. This is the canonical generic init from the S4D paper, providing a uniform spread of damping rates and a linear sweep of frequencies. Useful as an ablation against the LegT init.
3. **K-EMA log-spaced**. Set $\mathrm{Im}(\lambda\_k) = 0$ and $\mathrm{Re}(\lambda\_k) = -1 / \tau\_k$ for $\tau\_k = \tau\_\max^{k/(K-1)}$ — i.e. start at exactly the K-EMA bank's log-spaced grid. Useful only if R6.h shows that decorrelating K-EMA hurts; then we know the gradient-trained S4D should be initialised *near* the K-EMA bank rather than near LegT.

We default to (1) so the first R6.i pilot's results are directly
comparable to R6.a and R6.e. (2) and (3) become useful as ablations once
the headline R6.i result is known.

### 11.8 An empirical observation about HiPPO-LegT at $K = 4$

A side benefit of moving to S4D is that the eigenvalue spectrum becomes
inspectable. For $K = 4$, the eigenvalues of HiPPO-LegT's $A$ matrix
are computed numerically as

$$
\lambda\_{1,2} = -3.213 \pm 4.773 i,
\qquad
\lambda\_{3,4} = -4.787 \pm 1.567 i.
$$

Two complex-conjugate pairs, both with non-trivial imaginary parts.
**The HiPPO-LegT basis at $K = 4$ is therefore not four orthogonal
exponentials — it is two conjugate pairs of damped sinusoids**, with
the slower-decaying pair (the $-3.213$ pair) oscillating at the higher
frequency. This is consistent with §5.2's worst-case argument
(Legendre polynomials of order $\ge 2$ have oscillatory shape) but is
not always salient when one thinks of HiPPO as "polynomial moments of
the past".

The implication for R6.i: starting from LegT eigenvalues already
provides oscillation; what gradient descent will tune is the
**relative balance** between decay (Re) and frequency (Im), and the
**relative weighting** across channels (B_proj). If the data prefers a
basis closer to K-EMA (Im → 0), we will observe that drift directly in
the eigenvalues' imaginary parts during training.

### 11.9 Causal-leak compatibility

Identical to R6.a, R6.e, and §5.7. The S4D recurrence
$c\_{t+1} = \bar A c\_t + \bar B h\_t$ is strictly causal — $c\_t$
depends only on $h\_1, \ldots, h\_t$ — by construction of the
discrete-time form. The integrator-side fix (`h.detach()` before
computing the context summary) is unchanged because the S4D module is
a drop-in replacement for `MultiChannelHiPPO` with the same input
contract `(B, T, d) → (B, T, K, d)`. The new S4D class is registered
in `causal_probe.py` under both `legt`-init and `s4d_lin`-init
configurations; both pass strict-mode causality with $\Delta = 0$ in
the leak-fixed integrator (verified at the time of this section's
authoring; see the regression-test log).

### 11.10 R6.i protocol — concrete experimental config

| component | R6.e (HiPPO-LegT, learnable Δ) | R6.i (S4D, legt-init) | delta |
|---|---|---|---|
| `xi_module` class | `MultiChannelHiPPO` (LegT) | `MultiChannelS4D` | swap |
| K | 4 | 4 | — |
| basis init | LegT structured A | S4D-Lin via LegT eigenvalues | — |
| eigenvalues | fixed (LegT spectrum) | trainable (4 Re + 4 Im) | **+8 params** |
| B (input gain) | fixed (LegT $B = \sqrt{2k+1}$) | trainable (K real scalars) | **+4 params** |
| Δt | trainable scalar | trainable scalar | — |
| discretisation | bilinear (Tustin) | ZOH (closed-form, no inverse) | cleaner |
| outer integrator | unchanged | unchanged | — |
| V_θ | unchanged | unchanged | — |
| `--causal-force true` | yes | yes | — |
| schedule | 4000 steps, batch 16, block 512 | identical | — |
| ETA on MPS | ≈ 7.8 h | ≈ 7.5–8 h | comparable |

Files: `notebooks/conservative_arch/multixi/model_multixi_s4d.py` and
`notebooks/conservative_arch/scaleup/train_splm_em_ln_multixi_s4d_scaleup.py`.

Launch (run as soon as R6.e completes):

```bash
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python \
  notebooks/conservative_arch/scaleup/train_splm_em_ln_multixi_s4d_scaleup.py \
  --mode pilot --xi-channels 4 --xi-eigval-init legt \
  --xi-theta 200.0 --fixed-gamma 0.30 --seed 0 --causal-force true \
  --tag-suffix seed0_legtinit \
  --results-dir notebooks/conservative_arch/scaleup/results/multis4d_pilot_legtinit
```

### 11.11 Empirical hypothesis tree

Three coarse outcomes for R6.i, each with a clean follow-up:

| outcome | val_ppl region | what it implies | next move |
|---|---|---|---|
| **R6.i fully closes the gap** | val_ppl ≈ K-EMA (≈ 14.78 ± 0.5) | The §5.4 information-theoretic argument was right *given the right basis prior*; HiPPO-LegT's specific eigenvalue spectrum was wrong for TinyStories at K = 4, but the broader "orthogonal LTI summary" framework is correct once the basis is data-discovered. | Move to R6.b (K = 8) and R6.h.4 (head-to-head with the best K-EMA Fix variant). |
| **R6.i closes most but not all** | val_ppl ≈ 15–17 | Diagonal LTI is most of the answer. The residual gap is due either to non-diagonal coupling (DPLR), to the redundancy effect from §10.9 that K-EMA accidentally captures, or to a non-LTI input dependency. | R6.j (DPLR) tests the first; R6.h (Fix sweep on K-EMA) tests the second; R6.k or selective-A variants test the third. |
| **R6.i still trails K-EMA** | val_ppl > 17 | Diagonal LTI is *not* enough — the K-EMA bank's redundancy contributes information that no diagonal complex basis (oscillatory or not) can reproduce. | R6.h becomes the highest-priority follow-up: directly probe whether removing K-EMA's redundancy via Fix 1 / Fix 2 hurts, confirming the redundancy-as-prior interpretation of §10.9. R6.j (DPLR) becomes a long-shot bet. |

Either way the question is unambiguously decided by ≈ 8 hours of MPS,
and the result feeds directly into the remaining experimental queue.

### 11.12 Why this is consistent with the SPLM Lagrangian framework

The §5.7 conditions for embedding any context-summary architecture
inside SPLM's conservative-flow integrator were:

1. **Causality of $\xi\_t$.** Satisfied by S4D trivially: the
   convolution kernel is one-sided.
2. **Differentiability of $V\_\theta(\xi, h)$.** Unchanged: $\xi\_t \in \mathbb{R}^K$ feeds the same V_θ MLP as before.
3. **Bounded gradients of $\xi\_t$ with respect to $h$.** Stable (Hurwitz $A$) means the impulse response is integrable, so $\partial \xi\_t / \partial h\_s$ decays exponentially in $|t - s|$. The Hurwitz constraint is enforced structurally in S4D (`Re(λ) = -exp(log_neg_re)`), so this property holds across all training trajectories without additional regularisation.

S4D therefore satisfies the same three conditions HiPPO-LegT does, and
the SPLM conservative-flow Euler-Lagrange equation
$m \cdot \ddot{h}\_t = -\partial V\_\theta(\xi\_t, h\_t) / \partial h\_t$
is preserved exactly under the leak-fixed (`h.detach()`) integrator.
There is no additional theoretical work required to embed S4D inside
SPLM beyond what was done for R6.a.

### 11.13 References

- Gu, A., Dao, T., Ermon, S., Rudra, A., Re, C. (2020). HiPPO: Recurrent Memory with Optimal Polynomial Projections. *NeurIPS 2020*.
- Gu, A., Goel, K., Re, C. (2022). Efficiently Modeling Long Sequences with Structured State Spaces (S4). *ICLR 2022*.
- Gu, A., Goel, K., Gu, A., Re, C. (2022). On the Parameterization and Initialization of Diagonal State Space Models (S4D). *NeurIPS 2022*.
- Gupta, A., Gu, A., Berant, J. (2022). Diagonal State Spaces are as Effective as Structured State Spaces (DSS). *NeurIPS 2022*.

### 11.14 R6.i empirical results — K = 4 S4D legt-init pilot

> **Status.** Run completed 2026-05-03 08:33 EDT. Configuration:
> `--mode pilot --xi-channels 4 --xi-eigval-init legt --xi-theta 200.0 --fixed-gamma 0.30 --seed 0 --causal-force true` (defaults: `learnable_dt=True`, `learnable_B=True`).
> 16.54 M params, 5 M-token TinyStories cap, fixed γ = 0.30, MPS,
> elapsed 8.76 h. Artefacts:
> `notebooks/conservative_arch/scaleup/results/multis4d_pilot_legtinit/`.

**Headline.** Final val PPL = **16.85**, beating R6.e (HiPPO-LegT
learnable Δ, 17.45) by **0.60 PPL** — the **first basis-class run to
break below the HiPPO-LegT family**. Gap to K-EMA pilot narrows to
**+2.07 PPL** (was +5.04 at R6.a, +2.67 at R6.e). R6.i closes
**59 % of the original R6.a → K-EMA gap** by gradient-discovering the
basis (A, B) jointly rather than fixing it to LegT.

**val PPL trajectory** (R6.i vs R6.e vs K-EMA pilot side-by-side; full
log in `multis4d_pilot_legtinit/train_stdout.log`):

| step | K-EMA pilot | R6.e (learn Δ) | **R6.i (learn A,B)** | R6.i vs R6.e |
|---:|---:|---:|---:|---:|
|  200 | 112.60 | 144.11 | 160.46 | +11.3 % |
|  600 |  30.15 |  36.44 |  38.54 |  +5.8 % |
| 1000 |  22.41 |  27.48 |  28.29 |  +2.9 % |
| 1400 |  19.26 |  23.53 |  23.46 |  −0.3 % |
| 1800 |  17.68 |  21.14 |  20.98 |  −0.8 % |
| 2200 |  16.62 |  19.85 |  19.36 |  −2.5 % |
| 2600 |  15.39 |  18.21 |  17.70 |  −2.8 % |
| 3000 |  15.27 |  18.08 |  17.48 |  −3.3 % |
| 3400 |  15.18 |  17.94 |  17.35 |  −3.3 % |
| 3800 |  14.92 |  17.71 |  17.09 |  −3.5 % |
| **4000** | **14.78** | **17.45** | **16.85** | **−3.4 %** |

R6.i starts behind R6.e in the first 1000 steps (LegT init is
identical at step 0; R6.i's basis-learning machinery has not yet had
time to drift), then **crosses below at ~step 1400** and never
re-crosses. From step 2200 onward R6.i is monotonically below R6.e by
2.5–3.5 %. This is the cleanest possible empirical signature of "basis
learning is a real lift over Δt-only learning at this corpus".

**Eigenvalue evolution** (the basis itself is the experiment):

| step | pair 1 (high-Im) | pair 2 (low-Im) | Δt |
|---:|---|---|---:|
| 0 (LegT init) | $-3.21 \pm 4.77j$ | $-4.79 \pm 1.57j$ | 0.00500 |
| 1000 | $-3.81 \pm 4.82j$ | $-5.03 \pm 1.36j$ | 0.00777 |
| 2000 | $-5.79 \pm 5.04j$ | $-5.05 \pm 1.16j$ | 0.01188 |
| 3000 | $-7.83 \pm 5.30j$ | $-4.91 \pm 1.13j$ | 0.01335 |
| **4000** | $-8.02 \pm 5.32j$ | $-4.91 \pm 1.12j$ | **0.01372** |

(Pair-1 and pair-2 averages reported; the two eigenvalues within each
pair stayed within 1 % of conjugate-pair structure throughout
training.)

The basis migrated **asymmetrically and substantially**:

- **Pair 1** (initially LegT's "longer-memory" pair at $-3.21$):
  damping increased **150 %** ($-3.21 \to -8.02$), oscillation
  frequency increased **12 %** ($4.77 \to 5.32$). Converted from
  "slow-decay high-frequency" to "fast-decay high-frequency" — i.e.
  to a *short-memory oscillatory* channel.
- **Pair 2** (initially LegT's "shorter-memory" pair at $-4.79$):
  damping increased only **3 %** ($-4.79 \to -4.91$), oscillation
  frequency *decreased* **29 %** ($1.57 \to 1.12$). Stayed at moderate
  damping but became a *slower-oscillating* channel.

The optimiser ended up with **one fast-oscillatory short-memory pair**
and **one slow-oscillatory mid-memory pair** — qualitatively opposite
to the LegT prescription, which had assigned the *higher* frequency to
the *less-damped* pair. The data prefers the inversion.

`Δt` drifted from 0.00500 to **0.01372** (+174 %), nearly identical to
R6.e's +166 % drift — both pilots agree that the LegT-prescribed θ = 200
init is too short-horizon at K = 4 on this corpus.

**Channel-correlation diagnostic — the surprise.** Running the §10.8
protocol on the R6.i ckpt (40 K val tokens, identical batch / block
config) returns a correlation matrix that is *more correlated* than
K-EMA, not less:

```
[[1.000  1.000  0.931  0.939]
 [1.000  1.000  0.931  0.939]
 [0.931  0.931  1.000  1.000]
 [0.939  0.939  1.000  1.000]]
```

| metric | K-EMA pilot | R6.a | R6.e | **R6.i** |
|---|---:|---:|---:|---:|
| mean \|off-diagonal corr\| | 0.687 | 0.237 | 0.202 | **0.957** |
| total correlation TC(c) (nats) | 2.21 | 0.39 | 0.32 | **11.75** |
| K_eff (entropy power) / 4 | 1.98 (49 %) | 3.47 (87 %) | 3.54 (88 %) | **1.15 (29 %)** |
| **val PPL @ 4000 steps** | **14.78** | 19.82 | 17.45 | **16.85** |

Two channels per conjugate pair are **identically equal** (correlation
1.000); cross-pair correlation is 0.93. This is *not* a
training-discovered redundancy — it is a **structural artefact of the
S4D real-output convention** (§11.6) when eigenvalues come as
conjugate pairs.

**Why this happens.** For any complex eigenvalue $\lambda = \sigma + i\omega$
with $\sigma \lt 0$ and a real input gain $B$, the S4D real channel is

$$
\xi^k\_t = 2 \cdot \mathrm{Re}\bigl( c\_k(t) \bigr)
\quad\text{where}\quad
2 \cdot \mathrm{Re}\bigl( e^{\lambda \tau} \bigr)
= 2 e^{\sigma \tau} \cos(\omega \tau).
$$

Because $\cos(\omega \tau) = \cos(-\omega \tau)$, two eigenvalues
$\lambda$ and $\lambda^\* = \sigma − i\omega$ produce the **same** real
output. R6.i's K = 4 eigenvalues form 2 conjugate pairs at init (LegT
spectrum) and remain in conjugate-pair structure throughout training
(no loss term forces them to break out). The nominal K = 4 channels
therefore deliver only **K / 2 = 2 distinct real signals to V_θ**.

**Implications.**

1. **R6.i's apparent K_eff = 1.15 is a parametrisation ceiling, not a
   learned redundancy.** R6.i achieves val PPL 16.85 with effectively
   2 real channels (pair-1 fast/oscillatory, pair-2 slow/oscillatory),
   not 4. K-EMA also has K_eff ≈ 2 (entropy power 1.98 = 2 effective
   channels) but at val PPL 14.78. So R6.i and K-EMA are roughly
   matched in *effective channel count* yet K-EMA still wins by 2.07
   PPL. The remaining gap is not about *how many* channels — it is
   about *what those channels look like*.

2. **R6.i is not a strict generalisation of R6.a, despite §11.4's
   framing.** R6.a's `MultiChannelHiPPO` uses a Toeplitz convolution
   with the bilinear-discretised LegT $A$ matrix, which preserves all
   K = 4 distinct Legendre-polynomial outputs. R6.i's
   `MultiChannelS4D` uses ZOH discretisation of the *diagonalised*
   LegT-$A$ eigenvalues, which collapses conjugate pairs by the
   $2 \cdot \mathrm{Re}(\cdot)$ symmetry above. This is a real
   architectural difference at K = 4, not just a parameterisation
   change. The R6.i-vs-R6.a comparison should be read as
   "diagonalised-and-re-diagonalisable LegT" vs "Toeplitz LegT", not
   as "fixed LegT" vs "learnable LegT".

3. **A future "R6.i_v2" is straightforward.** Re-parametrise the K
   eigenvalues so they all have $\mathrm{Im}(\lambda) \gt 0$ (the
   conjugate counterparts are implicit in the $2 \cdot \mathrm{Re}(\cdot)$
   output). At K = 4 this gives 4 *distinct* real channels — closer
   to the standard S4D paper convention. The init then needs to choose
   4 distinct LegT eigenvalues (e.g. by taking only the 2 upper-half
   conjugate eigenvalues from the LegT spectrum and pairing each with
   one *real* eigenvalue from the K-EMA spectrum). This recovers true
   K-channel parity with R6.a and is the right next basis-class
   experiment if the §12.6 "search for smoother but capacity-controlled
   channel banks" direction is pursued.

4. **The §12.3 Argument 3 is not directly tested by R6.i.** The
   "implicit feature ensembling via redundancy" hypothesis was
   meant to predict the *direction* of the K-EMA → orthogonal-channel
   transition (decorrelation should hurt). R6.i sits in the *opposite*
   region of the redundancy axis from R6.a/R6.e, but the redundancy is
   *parametrisation-induced* not *learned*, so its
   high-correlation-but-mid-PPL outcome neither corroborates nor
   falsifies Argument 3. R6.h remains the clean test of Argument 3.

**Causal verification.** R6.i's `MultiChannelS4D` at both
`xi_eigval_init={legt, s4d_lin}` was registered in `causal_probe.py`
pre-launch (last night) and passes the strict probe with causal-side
Δ ≡ 0. Trained ckpt is exactly causal under the post-fix integrator.

**Compute parity.** R6.i runtime 8.76 h vs R6.e 8.11 h (+8.0 %), R6.a
7.81 h (+12.2 %). The extra cost is from per-step CPU-side ZOH
discretisation of the diagonal complex $A$ (offloaded from MPS to
work around an `aten::complex` MPS gap; see §11.5). Step rate degraded
~6.7 → 9.2 s/step over the 4000-step run — likely thermal throttling
on the Mac after 24+ hours of cumulative MPS load (R6.e + R6.i
back-to-back). Not a numerical issue.

**Training was clean.** Train loss monotone (10.6 → 2.75), grad norm in
[0.76, 1.35] throughout, no NaNs, no plateau-then-drift. Eigenvalue
trajectories were smooth with no oscillation — pair-1 damping
monotonically increased, pair-2 frequency monotonically decreased,
both stayed within conjugate-pair tolerance to numerical precision.
Real architectural lift, smooth optimisation, no funny business.

**Position in the §11.11 hypothesis tree.** R6.i lands in the
**"closes most but not all"** outcome row (val PPL ≈ 15–17). The §11.11
prescribed follow-ups for that row are R6.j (DPLR), R6.h (K-EMA Fix
sweep), and selective-A variants. The §12 synthesis (added 2026-05-03)
identifies R6.h as the highest-priority next experiment. Implication
3 above adds **"R6.i_v2 — proper-S4D parametrisation with K distinct
$\mathrm{Im}(\lambda) \gt 0$ eigenvalues"** as a parallel candidate;
this should be evaluated against R6.h after R6.h.1 (Fix 2) lands.

---

## 12. Synthesis — why orthogonal bases lose to K-EMA at this scale (added 2026-05-03; revised after R6.i completion)

The R6.a / R6.e / R6.i sequence has now produced enough empirical
evidence to revise the §3-§5 information-theoretic argument in a
unified way. This section consolidates the §10.9 reframing and the
R6.i empirical results (§11.14) into a single mechanistic hypothesis
with a falsifiable prediction in R6.h.

### 12.1 The empirical fact

Final val PPL at K = 4, 4000 steps, leak-free pilot, fixed gamma = 0.30,
identical LR schedule and corpus across all four runs:

| run | basis | trainable params for $\xi$ | val PPL | gap to K-EMA |
|---|---|---:|---:|---:|
| K-EMA pilot (R6.h.0) | learnable $\{\alpha\_k\}$, hand-picked init | $K = 4$ | **14.78** | — |
| R6.h.1 K-EMA Fix 2 | learnable $\{\alpha\_k\}$, log-spaced init $\tau\_{\max} = 100$ | $K = 4$ | **15.03** | **+0.25** |
| R6.a HiPPO-LegT, fixed $\Delta$ | LegT spectrum, fixed | 0 | 19.82 | +5.04 |
| R6.e HiPPO-LegT, learnable $\Delta$ | LegT spectrum, learnable $\Delta t$ | 1 | 17.45 | +2.67 |
| R6.i S4D, legt-init | learnable $(A, B)$ | 12 | **16.85** | **+2.07** |

R6.c (§10.8) and the R6.e / R6.i diagnostics (§10.11, §11.14) confirm
that R6.a and R6.e carry strictly more orthogonal information per
$\xi$ vector than K-EMA (lower mean off-diagonal $|\mathrm{corr}|$,
higher $K\_{\mathrm{eff}}$). R6.i is a special case: its nominal K = 4
channels collapse to **K / 2 = 2 effective real signals** by the
$2 \cdot \mathrm{Re}(\cdot)$ output convention applied to
conjugate-pair eigenvalues (see §11.14 "Why this happens"), which
breaks the §11.4 framing of R6.i as "a strict generalisation of R6.a".
Yet none of the three orthogonal-basis runs recovers K-EMA's val PPL,
and R6.i itself trails K-EMA by 2.07 PPL despite arriving at roughly
matched effective channel count. The separation between "channel
structure" and "val PPL" is the central anomaly this section explains.

### 12.2 The §3-§5 prediction came from optimising the wrong bottleneck

§§3-5 chained the following argument:

1. K-EMA is information-redundant.
2. HiPPO-LegT is information-orthogonal.
3. Orthogonal channels carry strictly more information per $\xi$ vector.
4. Therefore HiPPO-LegT should yield lower val PPL.

Steps 1-3 are correct in isolation and are confirmed by R6.c. Step 4
is the inferential leap that fails empirically. The leap implicitly
assumes that the *training-time bottleneck* is the information capacity
of $\xi$. The R6.a-R6.e-R6.i sequence falsifies that assumption.

The actual data flow is

$$
\text{tokens} \longrightarrow \xi \longrightarrow V\_\theta(\xi, h) \longrightarrow f \longrightarrow h\_{\ell+1}
$$

and at this corpus and budget the bottleneck is the *fit difficulty of*
$V\_\theta$, not the information content of $\xi$. The §3-§5 argument
was solving the wrong optimisation problem — maximising channel
information when the binding constraint was elsewhere.

### 12.3 Why a small MLP $V\_\theta$ prefers the K-EMA bank — four hypotheses

$V\_\theta$ in the SPLM scale-up is a depth-2 MLP with
$v\_{\text{hidden}} = 1024$ — a small smooth-function approximator.
Four arguments suggest its inductive bias is matched to K-EMA, not to
the LegT/S4D family. They are independent and each can in principle be
tested in isolation.

1. **Smoothness of channel signals.** K-EMA's channels are convex
   combinations of past $h\_s$ with non-negative weights ($\alpha\_k$-EMA
   over a one-sided exponential kernel). They are Lipschitz-continuous
   in time and trivially differentiable. HiPPO-LegT's channels are
   projections onto Legendre polynomials whose continuous-time generators
   contain damped sinusoidal components. We verified at $K = 4$ that
   the LegT spectrum factorises as two complex-conjugate eigenvalue pairs
   at $-3.21 \pm 4.77j$ and $-4.79 \pm 1.57j$ (§11.8) — i.e. four
   oscillatory damped modes, not four pure low-pass channels.
   $V\_\theta$ must learn a much rougher decision surface to extract
   usable features from oscillatory channels than from smooth running
   averages.

2. **Multi-scale running-average inductive bias.** K-EMA at
   $\alpha\_k = (0, 0.5, 0.9, 0.99)$ presents $V\_\theta$ with the same
   input averaged at four different timescales (~1, 2, 10, 100 tokens).
   This is the textbook inductive bias for sequence models —
   Transformers' KQV-projections at multiple positions, ConvNets'
   multi-scale receptive fields, multi-head attention. K-EMA hands
   $V\_\theta$ this prior for free.

3. **Implicit feature ensembling via redundancy.** Highly correlated
   channels mean that any nonlinearity $V\_\theta$ applies to one channel
   is implicitly averaged over the others. This is *cheap regularisation*
   — $V\_\theta$ cannot easily over-rely on a single channel because that
   channel correlates with the others, so its decisions average across
   the bank. Decorrelated channels remove this implicit regularisation
   and require explicit coordination across the bank to produce smooth
   outputs.

4. **Fewer optimisation obstacles.** K-EMA exposes $K$ trainable scalars
   $\alpha\_k$. R6.a exposes zero. R6.e exposes one ($\Delta t$). R6.i
   exposes 12. But the K-EMA scalars are *cheap to train* — each
   $\alpha\_k$ enters as an exponential weight in a 1-d running average
   with a well-conditioned gradient by construction. R6.i's eigenvalues
   enter through complex matrix exponentials inside a Toeplitz convolution
   kernel that is rebuilt every layer step from CPU-side linear algebra
   (§11.5). Even after the leak fix, the joint optimisation of
   $(V\_\theta, A, B, \Delta)$ is harder than the joint optimisation of
   $(V\_\theta, \alpha\_k)$.

R6.h (§10.10) tests argument 3 directly. Argument 4 is partially tested
by the R6.a → R6.e → R6.i progression itself (more parameters, more
basis-fitting capacity, modest val-PPL improvement but consistently
trailing K-EMA). Arguments 1-2 are cleanly testable only by a
$V\_\theta$-capacity ablation (see §12.6).

### 12.4 Connection to the broader SSM literature

Vanilla S4 with fixed orthogonal bases (HiPPO-LegT, HiPPO-LegS) reliably
loses to architecturally simpler RNN/scan models on language tasks (cf.
Mamba [Gu, Dao 2023]; SaShiMi [Goel et al. 2022]; Hyena [Poli et al. 2023]).
The standard explanation in those papers is *selectivity* — input-dependent
gating of the state recurrence — which $V\_\theta(\xi, h)$ already provides
via $h$. Selectivity alone cannot explain the K = 4 SPLM gap, because
$V\_\theta$ is identical across all four runs in §12.1.

The SPLM-side explanation has to be that the *non-selective* part of the
model — the basis itself — is the wrong inductive bias when consumed by a
fixed-capacity smooth $V\_\theta$. This is the §12.3.1-12.3.4 mechanism.
The SSM-literature parallel is that Mamba's selective scan does not just
add selectivity; it also discards the orthogonal-basis prior of S4 in
favour of a learnable *low-rank smooth* state recurrence — which
empirically beats HiPPO-LegT for the same reason the K-EMA bank does.

### 12.5 R6.h as the falsifiability test — pre-registered predicted signs

§10.10 lays out the R6.h K-EMA Fix sweep without committing in advance
to predicted signs of each fix. Under the §12.3 hypothesis, the
R6.a-R6.e-R6.i evidence supports the following pre-registered
predictions on the **direction** of the val_ppl change relative to
vanilla K-EMA's 14.78:

| pilot | intervention | predicted sign | actual outcome | reasoning |
|---|---|---|---|---|
| **R6.h.1** (Fix 2) | log-spaced $\alpha$ with $\tau\_{\max} = 100$, learnable | mild improvement or null ($\Delta \le 0$, magnitude $\le 1$ PPL) | **Δ = +0.25 PPL** (val_ppl 15.03; §10.12) — magnitude bound *holds*, sign opposite of predicted | Argument 2: tighter timescale coverage was expected to help slightly; orthogonal to the redundancy question. *Outcome*: did not move channel correlation (|Δ corr| = 0.014). The auto-tuning mechanism (§10.12) is a new finding; Fix 2 is now the weakest of the three §12.5 tests because the gradient pulls the channels back toward R6.h.0's regime regardless of init. |
| **R6.h.2** (Fix 1 + Fix 2) | + band-pass differencing $\xi^k = \mathrm{EMA}\_{\alpha\_k} − \mathrm{EMA}\_{\alpha\_{k-1}}$ | **predicted to hurt** ($\Delta \gt 0$) | not yet run | Argument 3: differencing decorrelates channels, removing the implicit-ensembling regulariser. Argument 1: differencing introduces non-monotonic time-domain shapes, hurting smoothness for $V\_\theta$. |
| **R6.h.2′** (Frozen-α + Fix 2) — *new test, post-§10.12* | log-spaced $\alpha$ with $\tau\_{\max} = 100$, **`requires_grad=False`** | **predicted to hurt** ($\Delta \gt 0$, magnitude $\ge 1$ PPL) | not yet run | Forces the channel spread that auto-tuning could not preserve in R6.h.1. If §12.3 holds, locking the wider spread should produce the val_ppl rise that R6.h.1 could not. |
| **R6.h.3** (Fix 1 + Fix 2 + Fix 3) | + decorrelation regulariser, $\lambda \in \{10^{-4}, 10^{-3}, 10^{-2}\}$ | **predicted to hurt monotonically in $\lambda$** ($\Delta \gt 0$, increasing with $\lambda$) | not yet run | Argument 3: the decorrelation regulariser is the most direct removal of K-EMA's correlated-channel inductive bias. Should hurt cleanly under the §12.3 hypothesis. *Refined*: with R6.h.1's auto-tuning finding, R6.h.3 is now the *strongest* falsifiability test — it actively opposes the gradient preference that R6.h.1 demonstrated. |

These predictions are pre-registered before each pilot launches. R6.h.1
landed inside the predicted magnitude bound (Δ ≤ 1 PPL) but on the
opposite side of zero from the directional prediction; the outcome is
*compatible* with both the §12.3 hypothesis and the null. The
auto-tuning finding (§10.12) reframes the falsifiability programme:
R6.h.2 (band-pass differencing — a *structural* change to the channels
that the gradient cannot undo without retraining the whole bank) and
R6.h.3 (decorrelation regulariser — an active counter-force to the
auto-tuning gradient) remain the cleanest tests, and we add **R6.h.2′**
(frozen-α variant of Fix 2) as a complementary test that decouples
*initialisation effect* from *auto-tuning convergence*. If the R6.h.2
+ R6.h.2′ + R6.h.3 set holds the predicted signs, the §12.3 hypothesis
is corroborated and the design programme should *abandon orthogonality
as an objective* for the K-EMA bank. If any of them helps, the §12.3
hypothesis is partially falsified — band-pass / frozen-α / decorrelated
K-EMA is not equivalent to HiPPO-LegT's orthogonality, and the
redundancy effect of §10.9 may be incidental rather than causal. The
result tightens the next-basis search either way.

### 12.6 Implications for the next basis-class search

If §12.3 is corroborated by R6.h:

- **Stop searching for "more orthogonal" channel banks.** R6.b
  (K = 8 HiPPO-LegT), R6.j (DPLR), and R6.k (dense $A$) are likely also
  dominated by K-EMA at this scale and budget; they should be
  deprioritised relative to S4D-only diagnostics.
- **Search for "smoother but capacity-controlled" channel banks.**
  Candidates include parameterised low-rank smooth filters (a single
  learnable smooth kernel $w(t) \in \mathbb{R}^T$ shared across channels
  with $K$ different decay rates), Gaussian-windowed-EMA banks (Gaussian
  impulse responses at $K$ different bandwidths), and learnable
  Mamba-style selective-state recurrences with fixed-capacity
  downstream MLP $V\_\theta$.
- **Run a $V\_\theta$-capacity ablation.** If the bottleneck is
  $V\_\theta$'s fit difficulty, scaling its width or depth should
  *unlock* the orthogonal-basis advantage. A diagnostic pilot at
  $v\_{\text{hidden}} = 4096$ (4× current) with R6.a's HiPPO-LegT
  configuration would test arguments 1-2 directly. If the K-EMA gap
  shrinks, the §12.3 mechanism is mostly the answer; if it persists,
  there is residual structural mismatch beyond $V\_\theta$ capacity.

If §12.3 is partially falsified (Fix 1 or Fix 3 *help*):

- The R6.j (DPLR) basis-class becomes attractive again — the gap may
  be due to specific eigenvalue placement that diagonal $A$ cannot
  reach.
- The "redundancy as feature" framing of §10.9 weakens to "redundancy
  is correlated with but not causally responsible for K-EMA's win",
  and the focus returns to finding the right *orthogonal* basis (the
  original §3-§5 programme).

### 12.7 Summary

The R6 sequence has falsified the operational version of the §3-§5
prediction (orthogonal bases ⇒ better val PPL) at K = 4, 4000 steps,
TinyStories, leak-free regime. The information-theoretic claim itself
remains correct — orthogonal bases do carry more information per $\xi$
vector — but it is not the constraint that determines training success
at this scale. The binding constraint is the fit difficulty of the
downstream consumer $V\_\theta$, and at the current $V\_\theta$ capacity
the K-EMA bank's smooth, multi-scale, correlated-channel structure
presents that fit problem with a more favourable inductive bias than
any orthogonal LTI alternative we have tried.

**R6.h.1 update (added 2026-05-03 after the Fix 2 pilot lands).**
The first K-EMA Fix sweep cell (§10.12) added a structural finding
that §12.3 did not anticipate: K-EMA's α values **auto-tune toward a
similar spread regardless of initialisation**. Fix 2 (log-spaced α,
$\tau\_{\max} = 100$) initialised the channels at a wider spread than
R6.h.0's hand-picked scheme, and after 4000 steps of joint training
the trained α values had moved within 0.005–0.13 of R6.h.0's on every
channel. The channel-correlation diagnostic on the final R6.h.1 ckpt
sits on top of R6.h.0 (mean |off-diag corr| 0.673 vs 0.687, K_eff 2.02
vs 1.98). Val PPL moved by Δ = +0.25 — inside the §12.5 magnitude
bound (≤ 1 PPL) but on the opposite side of zero from the directional
prediction. The cleanest reading is that **K-EMA's redundancy regime
is not a free choice of the designer but a gradient-preferred basin
of the joint $(V\_\theta, \alpha)$ landscape**; initialisation alone
cannot move it. This refines the §12.3 hypothesis from "redundancy is
a useful inductive bias" to "redundancy is the gradient-preferred
operating point of K-EMA × $V\_\theta$", and it argues against any
future K-EMA-initialisation-only intervention (no point trying
$\tau\_{\max} = 200$, $\tau\_{\max} = 50$, etc.; the gradient pulls
back regardless). The §12.3 hypothesis remains compatible with the
data but R6.h.1 alone does not corroborate it — Fix 2 failed to
*operationalise* a meaningfully different channel-correlation regime,
so its near-null val_ppl outcome is uninformative. The cleaner tests
shift to R6.h.2 (band-pass differencing — structural change the
gradient cannot undo), R6.h.2′ (frozen-α, the new
post-auto-tuning-finding test), and R6.h.3 (decorrelation regulariser
— active counter-force to the auto-tuning gradient).

R6.h, with the §12.5 pre-registered predictions, is still the cleanest
test of the synthesis as a whole. If R6.h.2 / R6.h.2′ / R6.h.3 confirm
(differencing / frozen-α / decorrelation regulariser all hurt), the
design programme turns from orthogonality toward smooth-multi-scale
parameterised filters and toward joint $(V\_\theta, \xi)$ co-design.
If any of them falsifies (helps), the residual gap to K-EMA is in
basis specifics, not in correlation structure, and the basis-class
search continues within the structured-A family with renewed
priority on R6.j / R6.k.

---


