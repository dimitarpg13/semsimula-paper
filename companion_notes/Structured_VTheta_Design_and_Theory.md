# Structured $V_\theta$: Theory, Derivation, and Analysis

**Status:** companion note to *Semantic Simulation: A Prescriptive Lagrangian Framework for Efficient Semantic Inference* (Gueorguiev, 2026).
**Scope:** theoretical background and analysis for replacing the MLP-based scalar potential $V_\theta$ with structured (analytically differentiable) parameterisations in the SPLM family of conservative language models.
**Companion docs:**

- [`Structured_VTheta_Memory_Anatomy.md`](./Structured_VTheta_Memory_Anatomy.md) -- GPU memory analysis of structured $V_\theta$ variants.
- [`Semantic_Attractor_Extraction.md`](./Semantic_Attractor_Extraction.md) -- attractor extraction methodology.
- [`Scalar_Potential_based_Helmholtz_Architecture_v3.md`](./Scalar_Potential_based_Helmholtz_Architecture_v3.md) -- the SP-HSPLM architecture (Q9(e)).
- **Implementation:** [`notebooks/conservative_arch/parf/model_structured_vtheta.py`](../notebooks/conservative_arch/parf/model_structured_vtheta.py).

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [The scalar potential in SPLM dynamics](#2-the-scalar-potential-in-splm-dynamics)
3. [Why the MLP is over-parameterised](#3-why-the-mlp-is-over-parameterised)
4. [Structured parameterisations: derivation](#4-structured-parameterisations-derivation)
5. [Analytical gradients](#5-analytical-gradients)
6. [Interpretability: explicit attractor centres](#6-interpretability-explicit-attractor-centres)
7. [Regularisation and the gauge symmetry](#7-regularisation-and-the-gauge-symmetry)
8. [Experimental results](#8-experimental-results)
9. [Cost analysis and speedup](#9-cost-analysis-and-speedup)
10. [Integration into SPLM, PARFLM, and Fock-PARFLM](#10-integration-into-splm-parflm-and-fock-parflm)
11. [Recommendations](#11-recommendations)
12. [Selecting the optimal mixture count K_mix](#12-selecting-the-optimal-mixture-count-k_mix)
13. [Hybrid structured potentials: Gaussian wells + quadratic background](#13-hybrid-structured-potentials-gaussian-wells--quadratic-background)
14. [References](#14-references)

---

## 1. Motivation

The SPLM family of models (SPLM, PARFLM, Fock-PARFLM) evolves hidden states $h_t$ through a damped dynamical system whose conservative force is the gradient of a learned scalar potential $V_\theta$. In all production configurations, $V_\theta$ is parameterised as a multi-layer perceptron (MLP).

Three empirical observations motivate the search for structured alternatives:

1. **$V_\theta$ regularisation experiments** (cells VR0--VR5) show that the trained MLP $V_\theta$ has a dynamic range of only 0.26--3.0 when regularisation is active, far below the MLP's representational capacity.
2. **The autograd cost**: computing the conservative force $f = -\nabla_h V_\theta$ via `torch.autograd.grad` costs approximately 2x the forward pass, dominating the per-layer training budget.
3. **Interpretability**: the MLP is a black box. Extracting semantic attractors from it requires a 1,500-step gradient descent procedure per prompt, which is expensive and non-deterministic.

The structured $V_\theta$ programme replaces the MLP with parameterised functional forms whose gradients are available in closed form, whose attractors are readable directly from the parameters, and whose capacity is matched to the empirically observed landscape complexity.

![Energy landscape comparison across V_theta parameterisations](images/structured_vtheta_energy_landscapes.png)

---

## 2. The scalar potential in SPLM dynamics

### 2.1 The governing equation

The SPLM hidden-state update at integration layer $\ell$ follows the damped Euler--Lagrange equation:

$$
w_t \ddot{h}\_t + \gamma \dot{h}\_t = -\nabla_h V_\theta(\xi_t, h_t)
$$

where:

- $h_t \in \mathbb{R}^d$ is the hidden state of token $t$ at layer $\ell$
- $w_t > 0$ is the per-token mass (parameterised via log-frequency)
- $\gamma > 0$ is the Rayleigh damping coefficient
- $\xi_t$ is the causal context summary (detached prefix mean)
- $V_\theta : \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}$ is the shared scalar potential

The discretised Verlet update is:

$$
h_t^{(\ell+1)} = h_t^{(\ell)} + \frac{\delta_t^{(\ell)}}{1 + \Delta t \cdot \gamma} + \frac{(\Delta t)^2}{w_t(1 + \Delta t \cdot \gamma)} f_t^{(\ell)}
$$

where $\delta_t^{(\ell)} = h_t^{(\ell)} - h_t^{(\ell-1)}$ is the kinematic memory (velocity proxy) and $f_t^{(\ell)} = -\nabla_h V_\theta(\xi_t, h_t^{(\ell)})$ is the conservative force.

### 2.2 The role of $V_\theta$

$V_\theta$ defines the **energy landscape** over which the hidden-state trajectory evolves. Its critical points $\nabla_h V_\theta = 0$ are the **semantic attractors** -- configurations toward which the damped dynamics naturally converge. The shape of $V_\theta$ determines:

- The **number and location** of attractors (basin structure)
- The **force magnitudes** that drive state evolution at each layer
- The **convergence properties** of the integrator (bounded vs unbounded dynamics)

```mermaid
flowchart LR
    subgraph SPLM Layer
        H[h at layer l] --> XI[Compute xi_t]
        H --> VTHETA[V_theta forward]
        XI --> VTHETA
        VTHETA --> FORCE[f = neg grad V]
        H --> DELTA[delta = h - h_prev]
        FORCE --> VERLET[Verlet update]
        DELTA --> VERLET
        VERLET --> HNEW[h at layer l+1]
    end
```

### 2.3 The two gauge symmetries

The NTP training loss touches $V_\theta$ only through its gradient $-\nabla_h V_\theta$. This gives $V_\theta$ two structural symmetries:

1. **Additive gauge:** $V_\theta(h) \mapsto V_\theta(h) + c$ for any constant $c$ leaves all forces unchanged.
2. **Multiplicative gauge:** $V_\theta(h) \mapsto \alpha V_\theta(h)$ for $\alpha > 0$ rescales all forces by $\alpha$, which is partially absorbed by the learnable $\gamma$, $w_t$, and $\Delta t$.

These symmetries mean the absolute scale and offset of $V_\theta$ are **physically meaningless** -- only the landscape shape (ratios of curvatures, relative basin depths) matters for dynamics.

---

## 3. Why the MLP is over-parameterised

### 3.1 Empirical evidence from the regularisation sweep

The $V_\theta$ regularisation sweep (cells VR0--VR5) adds an explicit regulariser to the training loss:

$$
\mathcal{L}\_\text{reg} = \lambda_V \cdot \mathbb{E}\big[V_\theta(\xi, h)^2\big]
$$

This breaks the gauge symmetry by penalising large absolute values, forcing $V_\theta$ toward zero. The sweep reveals:

| Cell | $\lambda_V$ | Best PPL | $V_\theta$ range | $V_\theta$ std | GD convergence |
|------|-------------|----------|-------------------|-----------------|----------------|
| VR0 | 0 | 249.5 | 1808 | 350 | 2% |
| VR1 | $10^{-6}$ | 256.4 | 893 | 75 | 0% |
| VR2 | $10^{-4}$ | 315.1 | 332 | 6.8 | 27% |
| VR3 | $10^{-2}$ | 318.9 | 70 | 0.6 | 70% |
| VR4 | 1 | 342.6 | 13 | 0.1 | 100% |
| VR5 | $10^{-4}$ (Verlet L=16) | 275.4 | 204 | 7.2 | 0% |

At $\lambda_V = 1$ (VR4), $V_\theta$ collapses to a range of just 13 with std 0.1 -- a near-constant function -- yet the model still achieves 343 PPL (only 37% worse than the unregularised baseline). This means the MLP's ~66K parameters (at $d = 128$, `v_hidden = 128`, `v_depth = 3`) are overwhelmingly devoted to representing a function that is nearly quadratic around its minima.

![Regularisation trade-off: PPL vs lambda_V and V_theta value distributions](images/structured_vtheta_regularization_tradeoff.png)

### 3.2 The information-theoretic argument

If the trained $V_\theta$ is nearly constant (range 0.26--3.0 under regularisation), the information content of the landscape is low. A quadratic form with $O(d)$ parameters can represent a single-well landscape exactly; a mixture of $K$ quadratics with $O(Kd)$ parameters can represent the empirically observed $K^{\ast} \approx 4$ basin structure. Both are orders of magnitude below the MLP's parameter budget.

---

## 4. Structured parameterisations: derivation

Four structured $V_\theta$ parameterisations are derived, each capturing a different level of landscape complexity. All share the interface `forward(xi, h) -> V` and `analytical_grad(xi, h) -> grad_h V`.

### 4.1 SQ1: Diagonal Quadratic Well

The simplest structured potential. A single quadratic basin centred at a context-dependent location $\mu(\xi)$:

$$
V_\theta(\xi, h) = \frac{1}{2} a(\xi)^T (h - \mu(\xi))^2 + b(\xi)
$$

where:

- $\mu(\xi) = W_\mu \xi + b_\mu \in \mathbb{R}^d$ is the **attractor centre** (linear projection of $\xi$)
- $a(\xi) = \text{softplus}(W_a \xi + b_a) + \epsilon \in \mathbb{R}^d\_{>0}$ is the **diagonal precision** (positive by construction)
- $b(\xi) = W_b \xi + b_b \in \mathbb{R}$ is the **offset** (absorbed by the additive gauge)
- The notation $(h - \mu)^2$ denotes elementwise squaring

The squaring is elementwise because $a$ is diagonal. This is the direct analogue of the Gaussian well potential from Section 4 of the paper, made context-dependent through $\xi$.

**Gradient (closed-form):**

$$
\nabla_h V_\theta = a(\xi) \odot (h - \mu(\xi))
$$

This is a single elementwise multiplication -- no autograd required.

**Attractor:** the unique minimum is at $h^* = \mu(\xi)$, readable directly from the model parameters.

**Parameter count at $d = 128$:** 3 linear projections ($\xi \to d$, $\xi \to d$, $\xi \to 1$) = **33K** (half the MLP baseline of 66K).

### 4.2 SQ2: Low-Rank Quadratic Well

Extends SQ1 to capture off-diagonal correlations in the precision matrix via a low-rank factor:

$$
A(\xi) = U(\xi) U(\xi)^T + \text{diag}(\lambda(\xi))
$$

where $U(\xi) \in \mathbb{R}^{d \times r}$ is a rank-$r$ factor (typically $r = 4$ or $r = 8$), and the potential becomes:

$$
V_\theta(\xi, h) = \frac{1}{2}(h - \mu)^T A(\xi)(h - \mu) + b(\xi)
$$

This can be evaluated efficiently without forming the full $d \times d$ matrix:

$$
V_\theta = \frac{1}{2} \lVert U^T(h - \mu) \rVert^2 + \frac{1}{2} \lambda^T (h - \mu)^2 + b(\xi)
$$

**Gradient (closed-form):**

$$
\nabla_h V_\theta = U(\xi) \big(U(\xi)^T (h - \mu)\big) + \lambda(\xi) \odot (h - \mu)
$$

This requires two matrix-vector products (cost $O(d \cdot r)$) plus one elementwise multiply.

**Attractor:** the unique minimum remains at $h^* = \mu(\xi)$.

**Parameter count at $d = 128$, $r = 8$:** dominated by the $U$ projection of size $d \times (d \cdot r)$ = **165K**.

### 4.3 SQ3: Mixture of $K$ Quadratic Wells

The most expressive structured parameterisation. Represents $K$ separate quadratic basins mixed via a log-sum-exp envelope, recovering the Gaussian mixture structure:

$$
E_k(\xi, h) = \frac{1}{2} a_k(\xi)^T (h - \mu_k(\xi))^2
$$

$$
V_\theta(\xi, h) = -\tau \log \sum_{k=1}^{K} \pi_k(\xi) e^{-E_k(\xi, h) / \tau} + b(\xi)
$$

where:

- $\mu_k(\xi) \in \mathbb{R}^d$ is the centre of the $k$-th attractor
- $a_k(\xi) \in \mathbb{R}^d\_{>0}$ is the diagonal precision of the $k$-th well
- $\pi_k(\xi) = \text{softmax}_k(W_\pi \xi + b_\pi)$ are mixing weights
- $\tau > 0$ is a temperature parameter controlling basin sharpness

The potential $V_\theta$ interpolates between:

- $\tau \to 0$: $V_\theta \to \min_k E_k$ (hard selection of the deepest basin)
- $\tau \to \infty$: $V_\theta \to \text{mean}_k E_k$ (uniform average, single effective basin)

At $\tau = 1$ (default), $V_\theta$ is the negative log marginal likelihood of a $K$-component Gaussian mixture, connecting the SPLM framework directly to the Gaussian well motivation of the paper.

**Gradient (closed-form):**

Define the softmax responsibilities:

$$
q_k(\xi, h) = \frac{\pi_k(\xi) e^{-E_k(\xi, h)/\tau}}{\sum_{j=1}^{K} \pi_j(\xi) e^{-E_j(\xi, h)/\tau}}
$$

Then:

$$
\nabla_h V_\theta = \sum_{k=1}^{K} q_k(\xi, h) \cdot a_k(\xi) \odot (h - \mu_k(\xi))
$$

The force at each point $h$ is a responsibility-weighted average of the per-basin forces. This is fully differentiable and involves no autograd.

**Attractors:** the $K$ minima are at $h_k^* = \mu_k(\xi)$, all readable directly from the parameters.

**Parameter count at $d = 128$, $K = 4$:** 4 sets of ($\mu$, $a$) projections plus mixing weights = **133K** (2x the MLP baseline, but with $K = 4$ explicit attractors matching the empirically observed $K^{\ast} = 4$ basin structure from the PR2 experiments).

### 4.4 SQ4: Hybrid Quadratic + Small MLP Residual

A safety net for cases where pure quadratic structure underfits:

$$
V_\theta(\xi, h) = V_\text{quad}(\xi, h) + \alpha \cdot V_\text{MLP}(\xi, h)
$$

where:

- $V_\text{quad}$ is an SQ1 diagonal quadratic backbone providing the analytical gradient
- $V_\text{MLP}$ is a small MLP (e.g. `v_hidden = 32`, `v_depth = 2`) providing a learnable correction
- $\alpha$ is a learnable scalar (initialised small, optionally regularised toward 0)

**Gradient:** the quadratic part is analytical; the MLP part still requires autograd. The overall speedup is partial but the quadratic backbone carries most of the force.

**Parameter count at $d = 128$:** 42K (quadratic backbone + small MLP).

### 4.5 Variant comparison

| Variant | Form | Attractors | Params ($d{=}128$) | Analytical grad | Speedup |
|---------|------|------------|---------------------|-----------------|---------|
| MLP baseline | unrestricted MLP | unknown | 66K | No | 1x |
| SQ1 (diagonal) | $\frac{1}{2} a^T(h-\mu)^2 + b$ | 1 per context | 33K | exact | ~2x |
| SQ2 (low-rank) | $\frac{1}{2}(h-\mu)^T A (h-\mu) + b$ | 1 with correlations | 165K | exact | ~2x |
| SQ3 (mixture $K{=}4$) | $-\tau \log \sum \pi_k e^{-E_k/\tau}$ | $K$ per context | 133K | exact ($\sim 10^{-6}$) | ~2x |
| SQ4 (hybrid) | $V_\text{quad} + \alpha V_\text{MLP}$ | 1 + correction | 42K | mixed | ~1.3x |

```mermaid
flowchart TD
    subgraph Structured V_theta Variants
        SQ1[SQ1: Diagonal Quadratic<br>1 attractor, 33K params]
        SQ2[SQ2: Low-Rank Quadratic<br>1 attractor + correlations, 165K params]
        SQ3[SQ3: Mixture K=4<br>4 attractors, 133K params]
        SQ4[SQ4: Hybrid Quad + MLP<br>1 attractor + correction, 42K params]
    end

    MLP[MLP Baseline<br>black box, 66K params] --> SQ1
    MLP --> SQ2
    MLP --> SQ3
    MLP --> SQ4

    SQ1 -->|simplest| DEPLOY[Drop-in replacement<br>in SPLM family]
    SQ2 -->|off-diagonal| DEPLOY
    SQ3 -->|multi-modal| DEPLOY
    SQ4 -->|safety net| DEPLOY
```

---

## 5. Analytical gradients

### 5.1 The autograd cost problem

In the standard SPLM, the force computation at each integration layer requires:

```python
V = model.V_theta(xi, h)
f, = torch.autograd.grad(V.sum(), h, create_graph=True)
```

The `create_graph=True` flag is **structurally required**: without it, the outer `loss.backward()` cannot differentiate through `f` to reach $V_\theta$'s parameters. This doubles the memory and compute cost of the $V_\theta$ evaluation because PyTorch must retain the full computation graph of the gradient itself.

### 5.2 The structured alternative

With a structured $V_\theta$, the force is computed directly:

```python
f = -model.V_theta.analytical_grad(xi, h)
```

No `autograd.grad` call is needed. The analytical gradient is a standard PyTorch tensor with full autograd support -- parameter gradients still flow through it via the normal `loss.backward()` chain, but no second-order graph is created.

![Force computation pipeline: MLP vs Structured](images/structured_vtheta_force_computation.png)

### 5.3 Validation protocol

All analytical gradients are validated against `torch.autograd.grad` at initialisation:

```python
from model_structured_vtheta import validate_analytical_grad
validate_analytical_grad(QuadraticWellVTheta(d=128), d=128)
# [QuadraticWellVTheta           ] max_abs_diff = 0.000e+00  [OK]
validate_analytical_grad(MixtureQuadraticVTheta(d=128, K=4), d=128)
# [MixtureQuadraticVTheta        ] max_abs_diff = 1.2e-06   [OK]
```

The SQ1 and SQ2 variants match autograd to machine precision (max_abs = 0). The SQ3 mixture has $\sim 10^{-6}$ residual due to softmax/logsumexp numerical noise, which is negligible.

---

## 6. Interpretability: explicit attractor centres

### 6.1 The attractor extraction problem

In the MLP $V_\theta$, extracting semantic attractors requires running gradient descent on the potential landscape for each prompt:

$$
h^* = \arg\min_h V_\theta(\xi, h) \qquad \text{(via 1,500 steps of GD per prompt)}
$$

This is expensive, non-deterministic (depends on initialisation seeds), and may miss secondary basins.

### 6.2 The structured solution

For SQ1--SQ3, the attractors are **explicit parameters** of the model:

- **SQ1/SQ2:** the attractor is $\mu(\xi) = W_\mu \xi + b_\mu$, a single linear readout
- **SQ3:** the $K$ attractors are $\mu_k(\xi) = (W_\mu \xi + b_\mu)[k]$, each a linear readout

```python
centres = model.V_theta.attractor_centres(xi)
# shape: (..., K, d) -- K attractor centres for each context xi
```

No gradient descent, no convergence checking, no seed sensitivity. The basin structure is **read directly from the model weights** in $O(K \cdot d)$ time.

### 6.3 Connection to the Gaussian well framework

The SQ3 mixture parameterisation recovers the paper's Gaussian well framework (Section 4) in a $\xi$-conditioned form. Each component $k$ defines a Gaussian-shaped energy basin:

$$
p_k(h \mid \xi) \propto \exp\big(-E_k(\xi, h)\big)
$$

The mixture potential $V_\theta = -\tau \log \sum_k \pi_k \exp(-E_k / \tau)$ is the negative log marginal of this mixture. The learned $\mu_k(\xi)$ are the **semantic attractor centres** -- the $\xi$-dependent locations in hidden-state space toward which the dynamics naturally converge.

The PR2 regularisation experiments found $K^{\ast} \approx 4$ distinguishable basins per prompt, matching the SQ3 default of $K = 4$.

---

## 7. Regularisation and the gauge symmetry

### 7.1 The gauge symmetry and its consequences

Because the NTP loss depends on $V_\theta$ only through $\nabla_h V_\theta$, the absolute scale and offset of $V_\theta$ are free parameters. Without regularisation:

- The additive gauge allows $V_\theta$ to drift arbitrarily far from zero (observed range: 1,808 in VR0)
- The multiplicative gauge allows the force magnitude to be traded off against $\gamma$ and $w_t$
- The potential is **unbounded below**, so the damped flow has no genuine equilibria -- only transient basins within the training horizon

### 7.2 What regularisation does

The regulariser $\mathcal{L}\_\text{reg} = \lambda_V \mathbb{E}[V_\theta^2]$ breaks both gauges:

1. **Finite equilibria appear.** With $V_\theta$ bounded below (the $\lambda_V V_\theta^2$ term ensures this), the damped flow $w \ddot{h} + \gamma \dot{h} = -\nabla_h V_\theta$ has at least one global minimum. The trajectory converges to genuine equilibria, not transient basins.

2. **Integration beyond $L_\text{train}$ becomes well-posed.** Without regularisation, running the integrator past $L_\text{train}$ causes hidden-state norms to diverge ($\lVert h \rVert \to \infty$). With bounded $V_\theta$, the damped flow has finite total energy and the trajectory settles into a bounded region.

3. **The Verlet-vs-Euler distinction reverses.** The unregularised attractor study found Euler's per-step truncation error acts as beneficial stochasticity on an unbounded landscape. With a bounded potential, this stochasticity is no longer needed -- Verlet's higher accuracy becomes the right inductive bias. Experimentally, VR5 (Verlet $L = 16$, $\lambda_V = 10^{-4}$) achieves 275 PPL, **beating** VR2 (Euler $L = 8$, same $\lambda_V$) at 315 PPL by 40 PPL.

### 7.3 The fundamental trade-off

| Property | No regularisation ($\lambda_V = 0$) | Regularisation ($\lambda_V > 0$) |
|----------|--------------------------------------|-----------------------------------|
| Equilibria of $V_\theta$ | None -- unbounded below | At least one -- bounded below |
| Attractor meaning | Dynamical (transient basins) | Energetic (genuine critical points) |
| Multi-basin structure | Rich (up to 10 basins) | At risk of collapse to one mode |
| Integration beyond $L_\text{train}$ | Diverges | Bounded, settles |
| Integrator preference | Euler (stochastic jitter helps) | Verlet (accuracy helps) |
| Perplexity | Better (full dynamic range) | Worse (compressed landscape) |
| Interpretability of $V_\theta$ | Hard (no minima to point to) | Easy (minima = semantic configs) |

The core tension: **regularisation makes $V_\theta$ interpretable as an energy landscape in the classical sense, but may destroy the very multi-basin structure that makes it interesting as an energy landscape.**

### 7.4 Multi-modality survives regularisation

The VR4 result ($\lambda_V = 1$) resolves this tension positively:

- **100% GD convergence** across all 5 prompts (384/384 seeds each)
- $K^{\ast} = 3.8$ basins per prompt -- actually **higher** than VR0's $K^{\ast} = 3.4$
- The bounded potential has **more distinguishable basins**, not fewer

This is the key result that enables structured $V_\theta$: if the regularised landscape is still multi-modal, then a structured parameterisation (SQ3 with $K = 4$) can capture it by construction.

---

## 8. Experimental results

### 8.1 Overview

Three full-scale structured $V_\theta$ experiments have been completed on TinyStories, using the SQ3 (mixture of quadratic wells) variant:

1. **Multi-Xi SPLM** — a 5-arm hyperparameter sweep varying $K_{\mathrm{mix}}$, $\tau$, and $K_\xi$
2. **Multi-Xi PARFLM** — a single arm (A2) replicating the best SPLM configuration with pairwise forces ($V_\phi$) enabled
3. **Multi-Xi Fock-PARFLM v2.1** — a single arm (A2) with the full Fock v2.1 configuration (16 registers, stack discipline, reverse channel, per-register $\tau$/keys)

All experiments train from scratch on TinyStories (~5M tokens, GPT-2 BPE, vocab 50,257) with $d = 256$, $L = 8$, AdamW, cosine-decay LR = 5e-4, and 16,000 training steps.

### 8.2 Experiment 1: Multi-Xi SPLM with structured $V_\theta$

**Configuration.** The base architecture is the Multi-Xi SPLM (`ScalarPotentialLMSARFMassLNMultiXi`) with log-frequency mass, semi-implicit Euler integration, $\gamma = 0.30$, $\lambda_V = 0.01$, and LayerNorm after each step. The MLP $V_\theta$ is replaced by `StructuredVThetaMultiXiAdapter(MixtureQuadraticVTheta(...))` before training.

**Sweep arms and results:**

| Arm | $K_{\mathrm{mix}}$ | $\tau$ | $K_\xi$ | Best PPL | Best step | $V_\theta$ params | Total params |
|-----|-----|------|------|----------|-----------|--------------|--------------|
| A1 | 4 | 1.0 | 4 | 14.10 | 14,400 | 2.1M | 15.2M |
| **A2** | **8** | **1.0** | **4** | **13.33** | **14,400** | **4.2M** | **17.3M** |
| A3 | 4 | 0.5 | 4 | 14.10 | 14,400 | 2.1M | 15.2M |
| A4 | 4 | 1.0 | 8 | 14.16 | 14,400 | 4.2M | 17.3M |
| A5 | 8 | 1.0 | 8 | 14.25 | 14,400 | 8.4M | 21.5M |

**MLP baseline (Multi-Xi SPLM, $K_\xi = 4$, 16k steps): 11.51 PPL.**

**Key findings:**

- **Best arm: A2** ($K_{\mathrm{mix}} = 8$, $\tau = 1.0$, $K_\xi = 4$) achieves **13.33 PPL** — a **1.82 PPL gap** (5.5% excess cross-entropy) vs the MLP baseline.
- **Doubling $K_{\mathrm{mix}}$ helps**: A2 ($K_{\mathrm{mix}} = 8$) beats A1 ($K_{\mathrm{mix}} = 4$) by 0.77 PPL (13.33 vs 14.10).
- **Temperature has no effect**: A1 ($\tau = 1.0$) and A3 ($\tau = 0.5$) achieve identical 14.10 PPL.
- **Doubling $K_\xi$ does not help**: A4 ($K_\xi = 8$, $K_{\mathrm{mix}} = 4$) slightly worsens over A1 (14.16 vs 14.10). A5 ($K_\xi = 8$, $K_{\mathrm{mix}} = 8$) at 14.25 PPL is worse than A2 (13.33) despite having 2x more $V_\theta$ parameters.
- **All arms converge at the same step** (14,400), suggesting a stable optimisation landscape.

**$V_\theta$ landscape statistics (A2):**

| Metric | Value |
|--------|-------|
| Mean $V_\theta$ | 99.8 |
| Std $V_\theta$ | 26.3 |
| Min | 9.0 |
| Max | 653.9 |
| Range | 644.9 |

**Learned $\xi$-channel decay rates (A2):** $[\alpha_1, \ldots, \alpha_4] = [0.12, 0.59, 0.84, 0.97]$ (init: $[0.25, 0.50, 0.75, 0.95]$). The model sharpens the fast channel ($\alpha_1$ drops from 0.25 to 0.12) while the slow channel barely changes.

### 8.3 Attractor basin analysis (SPLM A2)

The 8 mixture components develop **semantic specialisation** during training. Projecting each attractor centre $\mu_k(\xi)$ through the LM head on five TinyStories prompt types reveals distinct basin types:

**Active basins (~5 of 8):**

- **Story-opening basin** (basin 7, narrative): "lived" at 53.7% probability, followed by "wanted" (1.6%), "asked" (1.4%), "said" (1.3%). This basin has by far the most concentrated mass across all prompt types, acting as a strong narrative-initial attractor.
- **Family-dialogue basin** (basin 0, dialogue): "dad" at 19.3%, "mom" at 15.0%, "it" at 6.6%, "Ben" at 4.9%.
- **Punctuation-in-quotes basin** (basin 6, dialogue): ',"' at 19.9%, '!"' at 3.6%, '."' at 3.2%, '?"' at 2.9%. Specialises in dialogue-closing punctuation.
- **Pronoun/emotion basin** (basin 7, emotion): "her" at 93.3% — an extremely peaked attractor for gendered pronoun contexts.
- **Function-word / temporal basins** (basin 4, action): "time" at 9.8%, "best" at 5.3%, serving as a catch-all temporal basin.

**Inactive basins (~3 of 8):**

Basins 0, 2, 3 in description-type prompts, and basins 0, 3, 4 in emotion-type prompts, decode to near-uniform distributions (all top-5 probabilities $< 5 \times 10^{-5}$). This indicates the model self-selects $K_{\text{eff}} \approx 5$ from the over-provisioned $K_{\mathrm{mix}} = 8$, consistent with the pruning-based $K_{\mathrm{mix}}$ selection strategy (Section 12.3).

All basin readouts are **analytical** — $\mu_k(\xi)$ is a linear projection of the flattened xi context, readable directly from the model parameters with no gradient-descent extraction needed.

### 8.4 Experiment 2: Multi-Xi PARFLM with structured $V_\theta$

**Configuration.** The base architecture is the Multi-Xi PARFLM (`MultiXiPARFLM`) — identical to the SPLM but with an additional pairwise potential $V_\phi$ (competitive structural MLP, hidden = 128, top-$k = 8$, Gumbel-softmax routing). The same $V_\theta$ replacement is applied: SQ3 with $K_{\mathrm{mix}} = 8$, $\tau = 1.0$, $K_\xi = 4$, $\lambda_V = 0.01$. Trained from scratch for 16,000 steps.

**Result:**

| Arm | $K_{\mathrm{mix}}$ | $\tau$ | $K_\xi$ | Best PPL | Best step | PPL gap vs MLP | Gap (%) |
|-----|-----|------|------|----------|-----------|----------------|---------|
| **A2** | **8** | **1.0** | **4** | **12.27** | **14,400** | **0.17** | **0.6%** |

**MLP baseline (Multi-Xi PARFLM, $K_\xi = 8$, 8k steps): 12.10 PPL.**

The expressivity gap collapses from **1.82 PPL (5.5%)** in SPLM to **0.17 PPL (0.6%)** in PARFLM — an order-of-magnitude reduction.

**$V_\theta$ landscape statistics (PARFLM A2):**

| Metric | PARFLM structured $V_\theta$ | SPLM structured $V_\theta$ |
|--------|------------------------------|----------------------------|
| Mean $V_\theta$ | 0.02 | 99.8 |
| Std $V_\theta$ | 0.51 | 26.3 |
| Min | -2.69 | 9.0 |
| Max | 28.7 | 653.9 |
| Range | 31.4 | 644.9 |

The $V_\theta$ landscape undergoes **drastic compression** in PARFLM: the mean drops from 99.8 to 0.02 (a 5,000x reduction) and the range from 644.9 to 31.4 (a 20x reduction). With $V_\phi$ carrying the bulk of the force budget, $V_\theta$ collapses to a near-flat bias field. This explains why the quadratic-well expressivity limitation is irrelevant in PARFLM — there is almost nothing for $V_\theta$ to express.

**Learned $\xi$-channel decay rates (PARFLM A2):** $[\alpha_1, \ldots, \alpha_4] = [0.11, 0.55, 0.81, 0.97]$ — nearly identical to the SPLM A2 values $[0.12, 0.59, 0.84, 0.97]$, confirming that the causal EMA context structure is robust to both the $V_\theta$ parameterisation and the presence of $V_\phi$.

### 8.5 Experiment 3: Multi-Xi Fock-PARFLM v2.1 with structured $V_\theta$

**Configuration.** The base architecture is the Multi-Xi Fock-PARFLM v2.1 (`FockMultiXiPARFLM`) — extending PARFLM with 16 Fock registers, per-register $\tau$ and keys, orthogonal initialisation, stack discipline, and a reverse channel. The same $V_\theta$ replacement is applied: SQ3 with $K_{\mathrm{mix}} = 8$, $\tau = 1.0$, $K_\xi = 4$, $\lambda_V = 0.01$. Trained from scratch for 16,000 steps.

**Result:**

| Arm | $K_{\mathrm{mix}}$ | $\tau$ | $K_\xi$ | Best PPL | Best step | PPL gap vs MLP | Gap (%) |
|-----|-----|------|------|----------|-----------|----------------|---------|
| **A2** | **8** | **1.0** | **4** | **10.36** | **14,400** | **1.06** | **11.4%** |

**MLP baseline (Fock-PARFLM v2.1, 16k steps): 9.30 PPL.**

**$V_\theta$ landscape statistics (Fock-PARFLM A2):**

| Metric | Fock-PARFLM structured $V_\theta$ | PARFLM structured $V_\theta$ | SPLM structured $V_\theta$ |
|--------|----------------------------------|------------------------------|----------------------------|
| Mean $V_\theta$ | 0.008 | 0.02 | 99.8 |
| Std $V_\theta$ | 0.26 | 0.51 | 26.3 |
| Min | -5.49 | -2.69 | 9.0 |
| Max | 13.6 | 28.7 | 653.9 |
| Range | 19.1 | 31.4 | 644.9 |

**The Fock paradox: maximal compression, moderate gap.** The Fock-PARFLM exhibits the most compressed $V_\theta$ landscape across all three architectures (mean 0.008, range 19.1), yet the expressivity gap (1.06 PPL, 11.4%) is **six times larger** than PARFLM's (0.17 PPL, 0.6%). This paradox has a clear explanation: at 9.30 PPL, the Fock model operates closer to the dataset's entropy floor, where the marginal value of each nat of $V_\theta$ precision is higher. The Fock register mechanism (creation/destruction operators, stack discipline, reverse channel) creates a more structured dynamical regime where even a near-flat $V_\theta$ must provide fine-grained corrections that the diagonal quadratic parameterisation cannot match.

**Learned $\xi$-channel decay rates (Fock-PARFLM A2):** $[\alpha_1, \ldots, \alpha_4] = [0.14, 0.56, 0.79, 0.95]$ — again stable and consistent with SPLM ($[0.12, 0.59, 0.84, 0.97]$) and PARFLM ($[0.11, 0.55, 0.81, 0.97]$), confirming that the causal EMA context structure is invariant to the $V_\theta$ parameterisation, the presence of $V_\phi$, and the Fock dynamics.

### 8.6 Summary: the expressivity ceiling across three architectures

| Architecture | Structured $V_\theta$ PPL | MLP $V_\theta$ PPL | Gap | Gap (%) | Mean $V_\theta$ | Range |
|--------------|--------------------------|---------------------|-----|---------|-----------------|-------|
| Multi-Xi SPLM | 13.33 | 11.51 | 1.82 | 5.5% | 99.8 | 644.9 |
| Multi-Xi Fock-PARFLM v2.1 | 10.36 | 9.30 | 1.06 | 11.4% | 0.008 | 19.1 |
| Multi-Xi PARFLM | 12.27 | 12.10 | 0.17 | 0.6% | 0.02 | 31.4 |

Three key findings emerge from the cross-architecture comparison:

1. **Landscape compression is monotonic:** SPLM $\to$ PARFLM $\to$ Fock-PARFLM shows monotonically decreasing $V_\theta$ dynamic range as additional modelling capacity ($V_\phi$, then Fock registers) progressively marginalises $V_\theta$'s role.

2. **The expressivity gap is non-monotonic with respect to landscape compression.** PARFLM achieves the smallest gap (0.17 PPL) despite having a less compressed landscape than Fock-PARFLM. The Fock model's richer dynamics demand higher-precision $V_\theta$ corrections at its lower absolute PPL, creating a regime where even tiny $V_\theta$ signals carry disproportionate information.

3. **Structured $V_\theta$ remains practical across all three architectures.** The cost ranges from negligible (PARFLM: 0.6% excess CE) through moderate (SPLM: 5.5%) to meaningful but manageable (Fock-PARFLM: 11.4%). In all cases, the interpretability (explicit attractor centres) and speed (analytical gradients) benefits are retained in full.

---

## 9. Cost analysis and speedup

### 9.1 Per-layer force computation cost

The dominant cost at each integration layer is the force computation $f = -\nabla_h V_\theta(\xi, h)$.

| Method | Forward | Gradient | Total | Graph retention |
|--------|---------|----------|-------|-----------------|
| MLP + autograd | $O(d \cdot H \cdot D)$ | $\sim$ forward (2nd-order graph) | $\sim 2 \times$ forward | full graph held |
| SQ1 analytical | $O(d)$ | $O(d)$ (elementwise) | $O(d)$ | no 2nd-order graph |
| SQ2 analytical | $O(d \cdot r)$ | $O(d \cdot r)$ (two matvecs) | $O(d \cdot r)$ | no 2nd-order graph |
| SQ3 analytical | $O(K \cdot d)$ | $O(K \cdot d)$ (responsibility-weighted) | $O(K \cdot d)$ | no 2nd-order graph |

For $d = 128$, $H = 128$ (MLP hidden), $D = 3$ (depth), $K = 4$, $r = 8$:

- MLP + autograd: $\sim 2 \times 128 \times 128 \times 3 \approx 98\text{K}$ FLOPs
- SQ1: $128$ FLOPs
- SQ3 ($K = 4$): $4 \times 128 = 512$ FLOPs

The structured variants are **190x cheaper** in raw FLOPs per force evaluation, though the practical speedup is moderated by the $V_\phi$ pair-interaction cost (which dominates in PARFLM/Fock-PARFLM) and PyTorch overhead.

### 9.2 Memory saving

The elimination of `create_graph=True` for $V_\theta$ removes the second-order computation graph at each layer. For $L = 8$ layers with batch size $B$ and sequence length $T$:

- MLP: retains $O(L \times B \times T \times d \times H)$ activations for the gradient graph
- Structured: retains **zero** graph overhead for $V_\theta$

In PARFLM/Fock-PARFLM, the $V_\phi$ pair-interaction graph still dominates memory (see [`Structured_VTheta_Memory_Anatomy.md`](./Structured_VTheta_Memory_Anatomy.md)), but in pure SPLM the memory saving from structured $V_\theta$ is the **entire** gradient-graph overhead.

---

## 10. Integration into SPLM, PARFLM, and Fock-PARFLM

### 10.1 Drop-in replacement

The structured $V_\theta$ is a **drop-in replacement** requiring only two lines of code. For Multi-Xi architectures, the `StructuredVThetaMultiXiAdapter` handles the flattening of $K_\xi$ channels:

```python
from model_structured_vtheta import MixtureQuadraticVTheta
from model_structured_vtheta_multixi import StructuredVThetaMultiXiAdapter

K_xi, d = 4, 256
inner = MixtureQuadraticVTheta(d=K_xi * d, K=8, tau=1.0)
model.V_theta = StructuredVThetaMultiXiAdapter(inner, K=K_xi, d=d)
```

The adapter concatenates the $K_\xi$ xi-channels into a single $K_\xi \cdot d$ input vector, calls the inner SQ3 potential, and returns $(V, \nabla_h V)$ in the standard interface. No changes to `_layer_step` or any other model code are needed.

### 10.2 Analytical gradient integration

The `_layer_step` can be further modified to call `analytical_grad` directly, bypassing `torch.autograd.grad`:

```python
# Before (MLP path):
V = self.V_theta(xi, h)
grad_V, = torch.autograd.grad(V.sum(), h, create_graph=True)
f_theta = -grad_V

# After (structured path):
f_theta = -self.V_theta.analytical_grad(xi, h)
```

This eliminates the second-order computation graph for the $V_\theta$ contribution. In PARFLM, $V_\phi$ still requires autograd, so the overall speedup is partial but meaningful.

### 10.3 Empirical results across the SPLM family

| Model | $V_\theta$ role | PPL gap vs MLP | Recommendation |
|-------|-----------------|----------------|----------------|
| Multi-Xi SPLM | sole force | 1.82 PPL (5.5%) | use when interpretability > raw PPL |
| Multi-Xi PARFLM | one of two forces ($V_\theta + V_\phi$) | **0.17 PPL (0.6%)** | **recommended default** |
| Fock-PARFLM v2.1 | one of two forces ($V_\theta + V_\phi$) + Fock registers | 1.06 PPL (11.4%) | use when interpretability justifies the gap |

The critical empirical insight is that the expressivity ceiling depends on two factors: (1) whether $V_\phi$ is present, and (2) how close the model operates to the dataset's entropy floor. In SPLM, $V_\theta$ is the sole force and must carry all nonlinear structure. In PARFLM, $V_\phi$ absorbs it, compressing $V_\theta$'s landscape to near-zero dynamic range (see Section 8.4). In Fock-PARFLM, the landscape is even flatter, but the richer dynamics demand higher-precision $V_\theta$ corrections at the lower absolute PPL, leading to a moderate gap despite maximal landscape compression (see Section 8.5).

### 10.4 Applicability summary

| Model | Speedup source | PPL trade-off | Interpretability gain |
|-------|---------------|---------------|----------------------|
| SPLM / Multi-Xi SPLM | full 2x on force | moderate (5.5%) | 8 explicit attractor centres |
| PARFLM / Multi-Xi PARFLM | ~2x on $V_\theta$ only ($V_\phi$ dominates) | **negligible (0.6%)** | 8 explicit attractor centres |
| Fock-PARFLM v2.1 | ~2x on $V_\theta$ only ($V_\phi$ + Fock dominate) | meaningful (11.4%) | 8 explicit attractor centres |

---

## 11. Recommendations

### 11.1 Architecture-specific guidance

**For PARFLM:** structured $V_\theta$ is the **recommended default**. The empirical results (Section 8.4) show a negligible 0.17 PPL gap (0.6% excess CE) while providing full analytical gradients and explicit attractor readout. The pairwise $V_\phi$ absorbs all nonlinear structure, making the MLP's expressivity advantage irrelevant.

**For Fock-PARFLM v2.1:** structured $V_\theta$ is a **viable option when interpretability is valued**. The Fock experiment (Section 8.5) shows a 1.06 PPL gap (11.4% excess CE) — larger than PARFLM despite even greater landscape compression. At the Fock model's lower absolute PPL (9.30), the remaining $V_\theta$ signal carries higher-precision information that the quadratic parameterisation cannot match. Users should weigh this cost against the analytical-gradient and attractor-readout benefits.

**For SPLM (no $V_\phi$):** structured $V_\theta$ is recommended when **interpretability or inference speed** is valued over the last ~1.8 PPL points. The 5.5% excess cross-entropy is the cost of a fully transparent, analytically differentiable potential.

### 11.2 Which variant to use

- **SQ3 with $K_{\mathrm{mix}} = 8$** is the validated best choice. At $d = 256$, $K_\xi = 4$, this configuration achieves the best PPL across all three architectures: SPLM (13.33), PARFLM (12.27), and Fock-PARFLM (10.36). The model self-prunes to $K_{\text{eff}} \approx 5$ active basins.
- **$K_\xi = 4$ is sufficient.** Increasing to $K_\xi = 8$ did not improve PPL in the SPLM sweep (A4: 14.16, A5: 14.25), despite doubling the $V_\theta$ parameter count. The four-channel EMA structure captures the necessary context variation.
- **$\tau = 1.0$ is the correct temperature.** Lowering to $\tau = 0.5$ had no effect (A1 = A3 = 14.10 PPL). The Gaussian-mixture interpretation ($\tau = 1$) is both theoretically motivated and empirically optimal.
- **SQ1 (single quadratic)** remains an option when parameter budget is tight, but is untested at the TinyStories scale.
- **SQ4 (hybrid)** is no longer needed as a safety net given SQ3's strong performance.

### 11.3 Practical workflow

1. **Over-provision $K_{\mathrm{mix}}$.** Start with $K_{\mathrm{mix}} = 2 \times K_{\text{expected}}$ (e.g., $K_{\mathrm{mix}} = 8$). The model will self-prune unused basins (see Section 12.3 and empirical confirmation in Section 8.3).
2. **Train with $\lambda_V = 0.01$.** This regularisation keeps the $V_\theta$ landscape bounded without interfering with convergence.
3. **Decode attractors post-training.** Project each $\mu_k(\xi)$ through the LM head to verify basin specialisation and count $K_{\text{eff}}$.
4. **For PARFLM: adopt structured $V_\theta$ as the default.** The 0.6% PPL cost is negligible; the analytical gradient and explicit attractor centres are free benefits.
5. **For Fock-PARFLM: evaluate the cost-benefit.** The 11.4% excess CE is non-negligible. Adopt structured $V_\theta$ when attractor readout or analytical-gradient inference is needed; retain MLP $V_\theta$ when maximising PPL is the priority.

### 11.4 Open questions

1. **Scaling laws for $K_{\mathrm{mix}}$.** At $d = 256$ the model uses $K_{\text{eff}} \approx 5$ of $K_{\mathrm{mix}} = 8$. How does $K_{\text{eff}}$ scale with $d$, vocab size, and dataset complexity? The OpenWebText scale-up experiments will provide the first data point.
2. **The Fock precision paradox.** Why does Fock-PARFLM exhibit a 6x larger expressivity gap than PARFLM despite a flatter $V_\theta$ landscape? The leading hypothesis is that at lower absolute PPL ($9.30$ vs $12.10$), the model operates closer to the dataset's entropy floor, making the marginal value of each nat of $V_\theta$ precision higher. Can this be characterised quantitatively, e.g. by measuring the sensitivity of Fock-PARFLM's loss to small perturbations of $V_\theta$'s output?
3. **Higher $K_{\mathrm{mix}}$ for Fock-PARFLM.** The Fock notebook includes an A3 arm with $K_{\mathrm{mix}} = 16$. If the 1.06 PPL gap is driven by the quadratic wells' limited precision rather than their limited expressivity, increasing $K_{\mathrm{mix}}$ may help by providing finer-grained partitioning of the residual $V_\theta$ signal.
4. **Verlet integration with structured $V_\theta$.** The regularisation results (VR5) show Verlet beating Euler by 40 PPL when $V_\theta$ is bounded. Structured $V_\theta$ is bounded by construction — does it unlock the Verlet advantage without explicit regularisation? This is particularly promising given that the $V_\theta$ landscape in PARFLM and Fock-PARFLM is already near-flat.
5. **Log-sum-exp numerics at $d = 4096$.** The softmax responsibilities $q_k$ involve exponentials of quadratic forms in high dimensions. At the OpenWebText scale ($d = 4096$), numerical overflow/underflow may require careful temperature scheduling or log-domain computation.

---

## 12. Selecting the optimal mixture count K_mix

The SQ3 mixture of quadratic wells introduces a critical hyperparameter: the number of mixture components $K_{\mathrm{mix}}$. The TinyStories SPLM results show that doubling $K_{\mathrm{mix}}$ from 4 to 8 closes 0.77 PPL (14.10 → 13.33), but the A2 attractor decoding reveals that ~3 of the 8 basins are near-uniform (unused). This section surveys five approaches for selecting $K_{\mathrm{mix}}$ and provides practical implementation algorithms for each.

### 12.1 Approach 1: Attractor extraction from a trained MLP

**Idea.** Train a standard MLP $V_\theta$, extract its attractor basins using gradient descent, and count the number of distinct basins $K^{\ast}$. Use $K^{\ast}$ as $K_{\mathrm{mix}}$ for the structured replacement.

**When to use.** When an MLP baseline is already available and the goal is to match its landscape structure exactly.

**Algorithm:**

```
Input: trained MLP V_theta, representative prompts P, seeds S
Output: K* (optimal mixture count)

1. For each prompt p in P:
   a. Compute xi_p = causal_context(p)
   b. For each seed s in S (e.g. 384 seeds):
      - Initialise h_0 ~ N(0, sigma^2 I)
      - Run gradient descent: h_{t+1} = h_t - lr * grad_h V_theta(xi_p, h_t)
        for 1,500 steps with lr = 0.01
      - Record converged h* (check |grad V| < epsilon)
   c. Cluster converged points using DBSCAN(eps=0.5, min_samples=5)
   d. Record K_p = number of clusters for prompt p

2. K* = round(mean(K_p across all prompts))
   Confidence: report std(K_p) and the per-prompt distribution
```

**Practical notes:**
- The 1,500-step GD extraction is expensive (~3 minutes per prompt on GPU for $d = 128$)
- Not all seeds converge; report convergence rate (the VR0--VR4 sweep shows rates from 2% to 100% depending on regularisation)
- DBSCAN `eps` must be tuned to the $V_\theta$ scale; a good heuristic is $\epsilon = 0.5 \times \text{median pairwise distance of converged } h^*$
- The PR2 experiments found $K^{\ast} \approx 4$ at $d = 128$ on TinyShakespeare

**Limitations:**
- Requires a trained MLP (circular if the goal is to avoid training one)
- GD-based extraction may miss shallow basins
- Seed-dependent: reported $K^{\ast}$ depends on initialisation distribution

### 12.2 Approach 2: Validation sweep over $K_{\mathrm{mix}}$

**Idea.** Train SQ3 with multiple values of $K_{\mathrm{mix}}$ and select the $K$ that minimises validation PPL.

**When to use.** When compute budget allows multiple runs. Most reliable but most expensive.

**Algorithm:**

```
Input: dataset D, K_candidates = [2, 4, 8, 16], model config C
Output: K_opt

1. For each K in K_candidates:
   a. Build SQ3(K=K, d=C.d, tau=1.0)
   b. Train for N steps on D_train
   c. Evaluate PPL on D_val
   d. Record (K, PPL_val, K_eff)
      where K_eff = #{k : max_t w_k(h_t, xi_t) > 0.01}

2. K_opt = argmin_K PPL_val
   Secondary criterion: prefer smaller K if PPL difference < 0.5

3. Plot:
   - PPL_val vs K (expect diminishing returns)
   - K_eff vs K (expect saturation: K_eff << K for large K)
```

**Practical notes:**
- Geometric spacing (2, 4, 8, 16) is sufficient; finer grids rarely help
- Each run can be shortened to ~1/3 of full training (PPL ranking stabilises early)
- Track $K_{\mathrm{eff}}$ during training to detect early if $K$ is over-provisioned
- The SPLM TinyStories results suggest $K_{\mathrm{opt}} \approx 8$ for $d = 256$

**Limitations:**
- Linear cost in the number of candidates
- Optimal $K$ may depend on dataset, $d$, and architecture (SPLM vs PARFLM vs Fock)

### 12.3 Approach 3: Over-provision and prune

**Idea.** Initialise with a large $K_{\mathrm{max}}$ (e.g. 16 or 32), train normally, and let the model self-select the effective number of basins. Inactive basins are identified post-training and optionally pruned.

**When to use.** The recommended default approach --- requires only a single training run and yields $K_{\mathrm{eff}}$ as a byproduct.

**Algorithm:**

```
Input: dataset D, K_max (e.g. 16 or 32), model config C
       Optional: L1 penalty weight lambda_prune
Output: K_eff, pruned model

1. Build SQ3(K=K_max, d=C.d, tau=1.0)

2. (Optional) Add L1 penalty to accelerate pruning:
   L_prune = lambda_prune * sum(softplus(alpha_k))
   where alpha_k are the pre-softmax mixing logits
   Recommended: lambda_prune = 1e-4 to 1e-3

3. Train for N steps on D_train

4. Count active basins on D_val:
   For each validation batch (h, xi):
     w_k = softmax(-E_k / tau) * pi_k   (responsibility weights)
     activity[k] += (max_t w_k > epsilon)
   K_eff = #{k : activity[k] / n_batches > delta}
   Recommended: epsilon = 0.01, delta = 0.05

5. (Optional) Prune inactive basins:
   active_indices = {k : activity[k] / n_batches > delta}
   new_mu = mu[active_indices]
   new_a  = a[active_indices]
   new_pi = renormalise(pi[active_indices])
   Return SQ3(K=K_eff) with transplanted parameters

6. (Optional) Fine-tune pruned model for M << N steps
   to recover any minor PPL regression from pruning
```

**Practical notes:**
- The A2 TinyStories result ($K_{\mathrm{max}} = 8$, $K_{\mathrm{eff}} \approx 5$) validates this approach empirically
- The $L_1$ penalty on mixing logits drives unused $\alpha_k \to -\infty$, making $\pi_k \to 0$ and deactivating the basin cleanly
- This is analogous to the Dirichlet-process stick-breaking prior in Bayesian GMMs, where the concentration parameter $\alpha_0$ controls the effective number of components
- Monitoring $K_{\mathrm{eff}}$ during training provides an early stopping signal: if $K_{\mathrm{eff}}$ has stabilised for 20% of training, the run can be shortened
- The over-provisioning cost is modest: $K_{\mathrm{max}} = 32$ vs $K = 4$ increases $V_\theta$ parameter count by 8x, but $V_\theta$ is a small fraction of total model parameters

**Limitations:**
- The initial training run has higher parameter count (and memory) than necessary
- Pruning may introduce a small PPL regression (~0.1--0.3 PPL in preliminary tests); fine-tuning recovers this
- The activity threshold $\epsilon$ requires calibration

### 12.4 Approach 4: Information-theoretic selection (BIC/AIC)

**Idea.** Treat the structured $V_\theta$ as a density model (via the Gaussian well equivalence) and use Bayesian Information Criterion (BIC) or Akaike Information Criterion (AIC) to select $K$.

**When to use.** When a principled model-selection criterion is desired and the Gaussian well interpretation is central to the analysis.

**Algorithm:**

```
Input: trained SQ3 models for K in K_candidates, validation set D_val
Output: K_opt

1. For each K in K_candidates:
   a. Compute log-likelihood under the GMM interpretation:
      LL(K) = sum_t log( sum_k pi_k * N(h_t; mu_k(xi_t), tau * diag(1/a_k(xi_t))) )
      where the sum is over all validation tokens t

   b. Count effective parameters:
      p(K) = K * (d + d + 1)   [mu_k projections + a_k projections + mixing logits]
            + K * d_xi * (2d + 1)  [the linear maps from xi]

   c. Compute BIC and AIC:
      BIC(K) = -2 * LL(K) + p(K) * log(n)
      AIC(K) = -2 * LL(K) + 2 * p(K)
      where n = number of validation tokens

2. K_opt = argmin_K BIC(K)   [BIC preferred for large n]
```

**Practical notes:**
- The Gaussian well equivalence (Section 4.3) makes the likelihood computation well-defined: $V_\theta = -\tau \log \sum_k \pi_k \exp(-E_k / \tau)$ is the negative log marginal of a GMM
- BIC penalises complexity more heavily than AIC and is preferred when $n$ (number of tokens) is large (which it always is for language modelling)
- The log-likelihood can be computed as a byproduct of the forward pass with negligible additional cost
- BIC tends to select smaller $K$ than validation-PPL sweeps because it penalises unused parameters even when they don't hurt PPL

**Limitations:**
- The BIC/AIC framework assumes the Gaussian well interpretation is exact, but the actual $V_\theta$ landscape is modified by training dynamics and regularisation
- Requires multiple trained models (same cost as Approach 2)
- Does not account for the interaction between $K_{\mathrm{mix}}$ and other hyperparameters ($\tau$, $\lambda_V$)

### 12.5 Approach 5: Spectral analysis of MLP Hessian

**Idea.** Compute the Hessian $H = \nabla^2_h V_\theta$ of the trained MLP at each attractor and analyse its spectrum. The number of significant eigenvalue clusters indicates the intrinsic dimensionality of the basin structure, guiding $K_{\mathrm{mix}}$.

**When to use.** For detailed landscape analysis when understanding the basin geometry (not just the count) is important.

**Algorithm:**

```
Input: trained MLP V_theta, extracted attractors {h*_k}, context xi
Output: K* and per-basin curvature profiles

1. For each attractor h*_k:
   a. Compute Hessian H_k = d^2 V / dh^2 at (xi, h*_k)
      - For d <= 512: full Hessian via torch.autograd.functional.hessian()
      - For d > 512: top-r eigenvalues via Lanczos iteration
        (scipy.sparse.linalg.eigsh or torch stochastic Lanczos)

   b. Eigendecompose: H_k = U_k Lambda_k U_k^T
      - Sort eigenvalues: lambda_1 >= lambda_2 >= ... >= lambda_d

   c. Classify basin geometry:
      - If all lambda_i > 0: genuine minimum (attractor)
      - If some lambda_i < 0: saddle point (not a true basin)
      - If lambda_i ~ 0 for i > r: effective rank r
        (the basin has r "stiff" directions and d-r "flat" directions)

   d. Compute effective rank:
      r_k = #{i : lambda_i > epsilon * lambda_1}
      where epsilon = 0.01 (1% of leading eigenvalue)

2. K* = number of attractors with all-positive eigenvalues
   Report per-basin r_k to guide SQ2 rank parameter

3. (Optional) Fit SQ2 rank:
   r_opt = median(r_k) across all basins
   This tells you whether diagonal (SQ1/SQ3) or low-rank (SQ2)
   precision is needed
```

**Practical notes:**
- Full Hessian at $d = 128$ costs $O(d^3) \approx 2M$ FLOPs per attractor; feasible on GPU
- At $d = 4096$ (scale-up), the Lanczos method with $r = 32$ iterations is necessary ($O(r \cdot d^2)$ FLOPs)
- The eigenvalue spectrum also reveals whether the SQ3 diagonal precision assumption is adequate: if the Hessian is far from diagonal, SQ2's low-rank factor is needed
- This approach provides the richest information but is the most computationally expensive

**Limitations:**
- Requires a trained MLP and extracted attractors (combines the cost of Approach 1 + Hessian computation)
- Hessian computation is expensive at large $d$
- Saddle points may be misidentified as attractors if GD doesn't fully converge

### 12.6 Comparison of approaches

| Approach | Runs required | Cost | Output | Recommended for |
|----------|---------------|------|--------|-----------------|
| 1. Attractor extraction | 1 (MLP) | High (GD extraction) | $K^{\ast}$ count | When MLP baseline exists |
| 2. Validation sweep | # K values | High (multiple runs) | PPL-optimal $K$ | Final tuning |
| 3. Over-provision/prune | 1 | Low | $K_{\mathrm{eff}}$ + pruned model | **Default recommendation** |
| 4. BIC/AIC | # K values | Medium | Information-theoretic $K$ | When GMM interpretation matters |
| 5. Spectral analysis | 1 (MLP) | Very high | $K^{\ast}$ + curvature profiles | Detailed landscape understanding |

**Recommended workflow:**

1. **Start with Approach 3** (over-provision/prune) as the default. Set $K_{\mathrm{max}} = 2 \times K_{\mathrm{guess}}$ where $K_{\mathrm{guess}}$ is your best prior (e.g. 8 based on the SPLM TinyStories result).
2. If $K_{\mathrm{eff}}$ is close to $K_{\mathrm{max}}$, double $K_{\mathrm{max}}$ and retrain.
3. For final papers/releases, validate with **Approach 2** (sweep) using $K \in \{K_{\mathrm{eff}}/2, K_{\mathrm{eff}}, 2 K_{\mathrm{eff}}\}$ around the pruning result.
4. Use **Approach 4** (BIC) if the Gaussian well interpretation is the theoretical focus.
5. Use **Approach 5** (Hessian spectral) only for deep-dive analysis of specific prompts/basins.

## 13. Hybrid structured potentials: Gaussian wells + quadratic background

The SQ3 (log-sum-exp quadratic) parameterisation achieves the best
PPL across all three architectures on TinyStories (§8) but suffers
from **structurally unbounded force** that causes repeated blowups on
OpenWebText (documented in [`Training_Instabilities_in_Fock-PARFLM_with_structured_V_theta.md`](./Training_Instabilities_in_Fock-PARFLM_with_structured_V_theta.md), §2–§4, §13). The Gaussian mixture-PDF alternative
(MixtureGaussianVTheta, §12.4) is bounded by construction
($V \in [-\Sigma w_k, 0]$) and trains stably on OpenWebText, but
sacrifices ~5.6 PPL on TinyStories (15.95 vs 10.36 for Fock A2).

The **hybrid Gaussian + quadratic background** potential bridges this
gap by combining both:

$$
V_{\text{hybrid}}(\xi, h) = -\sum_k w_k(\xi) \exp\left(-\frac{1}{2} a_k(\xi)^\top (h - \mu_k(\xi))^2\right) + \varepsilon \lVert h \rVert^2
$$

### 13.1 Design rationale

The Gaussian wells provide local attractor structure (bounded force,
learned centres, context-dependent weights), while the quadratic
background provides **global coercivity**: the potential grows as
$\varepsilon \lVert h \rVert^2$ far from all wells, generating a
restoring force $f_{\text{bg}} = -2\varepsilon h$ that prevents hidden
states from drifting into regions where no Gaussian well is active.

This directly addresses the fundamental limitation of pure Gaussian
wells identified in §9.3 of the instabilities document: the absence of
far-field force leaves "escape routes" between wells.  SQ3 closes
those escape routes via its unbounded far-field potential — but at the
cost of unbounded force.  The quadratic background closes them with a
**bounded, controllable** force whose magnitude scales linearly with
$\lVert h \rVert$.

### 13.2 Structural properties

| Property | SQ3 | Pure Gaussian | Hybrid |
|----------|-----|---------------|--------|
| V range | $(-\infty, +\infty)$ | $[-\Sigma w_k, 0]$ | $[-\Sigma w_k, +\varepsilon \lVert h \rVert^2]$ |
| Coercive ($V \to +\infty$) | yes | no | **yes** |
| Force bound | unbounded | $0.607 w_k / \sigma_k$ | $\max(0.607 w_k/\sigma_k, 2\varepsilon \lVert h \rVert)$ |
| OpenWebText stable | no (§2–§4) | yes | **expected yes** |

The combined force is the sum of the Gaussian well forces (bounded,
local) and the background restoring force (unbounded but slow-growing,
global).  At typical hidden-state scales ($\lVert h \rVert \approx \sqrt{d}$
under `ln_after_step=True`), the background force is
$2\varepsilon\sqrt{d}$, which for $\varepsilon = 10^{-4}$ and $d = 256$
is $\approx 0.003$ — negligible compared to Gaussian well forces near
centres ($\sim 0.1$) but sufficient to prevent unbounded escape.

### 13.3 The $\varepsilon$ selection problem

The background strength $\varepsilon$ controls the trade-off:

- $\varepsilon = 0$: pure Gaussian (no global restoring, escape risk)
- $\varepsilon \ll 1/d$: wells dominate locally, background acts only
  far from all centres — expected optimal regime
- $\varepsilon \gg 1/d$: background dominates, landscape collapses to
  a single global attractor — over-constrained

The transition threshold can be estimated from force balance.  At a
well centre $\mu_k$, the Gaussian force vanishes and the background
force is $2\varepsilon\mu_k$.  In the neighbouring basin, the Gaussian
peak force is $\sim 0.6 w_k / \sigma_k$.  The background should be a
small perturbation: $2\varepsilon\lVert\mu_k\rVert \ll 0.6 w_k / \sigma_k$.
For $w_k \sim 1/K$, $\sigma_k \sim \sqrt{d}$, $\lVert\mu_k\rVert \sim \sqrt{d}$,
this gives $\varepsilon \ll 0.6 / (2Kd)$.  At $K = 8$, $d = 256$:
$\varepsilon \ll 1.5 \times 10^{-4}$.

### 13.4 Evaluation notebook

The systematic evaluation is implemented in
[`colab_hybrid_gaussian_quad_vtheta.ipynb`](../notebooks/conservative_arch/scaleup/colab_hybrid_gaussian_quad_vtheta.ipynb)
with the following sweep:

| Cells | Base model | $\varepsilon$ values | $K$ |
|-------|-----------|---------------------|-----|
| H1–H6 | FockPARFLM v2.1 | 0, $10^{-5}$, $5{\times}10^{-5}$, $10^{-4}$, $5{\times}10^{-4}$, $10^{-3}$ | 8 |
| H7 | FockPARFLM v2.1 | $10^{-4}$ | 16 |
| P1–P6 | PARFLM | 0, $10^{-5}$, $5{\times}10^{-5}$, $10^{-4}$, $5{\times}10^{-4}$, $10^{-3}$ | 8 |
| P7 | PARFLM | $10^{-4}$ | 16 |

All cells share the same architecture as the existing structured
$V_\theta$ experiments (§8): d=256, L=8, $K_\xi = 4$,
`structural_competitive` $V_\phi$, 16k steps on TinyStories.
Reference baselines: Fock SQ3 A2 (10.36 PPL), Fock Gaussian G1
(15.95 PPL), PARF SQ3 A2 (12.27 PPL).

### 13.5 Connection to the broader architecture

If the hybrid $V_\theta$ successfully closes the SQ3–Gaussian gap on
TinyStories while remaining structurally bounded, the natural next step
is an OpenWebText scale-up. The hybrid would then serve as a **unified
structured $V_\theta$** that works at both scales — replacing the
current situation where SQ3 is used on TinyStories and Gaussian on
OpenWebText.

---

## 14. References

### Internal documents

- [`Structured_VTheta_Memory_Anatomy.md`](./Structured_VTheta_Memory_Anatomy.md) -- GPU memory analysis of the structured variants interacting with $V_\phi$.
- [`Semantic_Attractor_Extraction.md`](./Semantic_Attractor_Extraction.md) -- attractor extraction methodology; the structured $V_\theta$ eliminates the GD-based extraction entirely.
- [`Scalar_Potential_based_Helmholtz_Architecture_v3.md`](./Scalar_Potential_based_Helmholtz_Architecture_v3.md) -- the SP-HSPLM (Q9(e)) architecture, which inherits structured $V_\theta$ as its S-block scalar potential.
- [`Conservative_by_Construction_Language_Models.md`](./Conservative_by_Construction_Language_Models.md) -- the conservative-by-construction framework in which $V_\theta$ operates.
- `docs/V_theta_regularization.docx` (semsimula) -- full regularisation sweep results (VR0--VR5).
- `docs/Structured_V_theta.docx` (semsimula) -- original design plan screenshots.

### Implementation

- [`notebooks/conservative_arch/parf/model_structured_vtheta.py`](../notebooks/conservative_arch/parf/model_structured_vtheta.py) -- all four structured $V_\theta$ classes with validation harness.
- [`notebooks/conservative_arch/parf/model_gaussian_vtheta.py`](../notebooks/conservative_arch/parf/model_gaussian_vtheta.py) -- MixtureGaussianVTheta and SARFGaussianVTheta implementations.
- [`notebooks/conservative_arch/parf/scripts/structured_vtheta_sweep.ipynb`](../notebooks/conservative_arch/parf/scripts/structured_vtheta_sweep.ipynb) -- Colab notebook for the SQ1--SQ5 comparison sweep.
- [`notebooks/conservative_arch/scaleup/colab_hybrid_gaussian_quad_vtheta.ipynb`](../notebooks/conservative_arch/scaleup/colab_hybrid_gaussian_quad_vtheta.ipynb) -- hybrid Gaussian + quadratic background $\varepsilon$-sweep (§13).

### External literature

- The Gaussian well motivation connects to Section 4 of the paper (*Semantic Simulation*, Gueorguiev 2026) where the scalar potential is introduced as an energy landscape governing semantic structure evolution.
- **Quadratic potentials in dynamical systems ML:** the diagonal quadratic (SQ1) is the standard form used in Harmonic Oscillator NNs and the conservative component of D-HNNs (Sosanya & Greydanus, ICLR 2022).
- **Mixture of Gaussians as energy models:** the SQ3 log-sum-exp construction is equivalent to the energy function of a Gaussian Mixture Model, placing the SPLM scalar potential within the classical energy-based model framework (LeCun et al., 2006).

---

*Last updated: 24 June 2026. Section 13 added: hybrid Gaussian + quadratic background structured potential, the $\varepsilon$-sweep evaluation notebook, and the stability–expressivity analysis. Phase 2 (analytical gradient wiring) queued pending sweep results.*
