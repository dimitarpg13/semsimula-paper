# P-rot-6: Velocity-Coupled Gyroscopic Dynamics and Decoder-Only Transformers
### A Theoretical and Empirical Framework for Hidden State Motion in Semantic Space

---

> **Abstract.** We derive the P-rot-6 model — a second-order damped equation of motion augmented with a velocity-coupled skew-symmetric force — from first principles, situate it within the Helmholtz decomposition of hidden state dynamics, and trace its theoretical connection to the key-value asymmetry ($K \ne V$) in scaled dot-product attention. We then develop a concrete, end-to-end methodology for estimating the P-rot-6 $B$ matrix from the weight matrices of decoder-only transformer models (GPT-2, Pythia), including code, diagnostics, and a principled failure-mode analysis. Finally, we discuss what a pass or fail of P-rot-6 implies for the geometry of semantic space and outline the next model class hierarchy.

> **Scope and status note (added retroactively).**
> This document formulates P-rot-6 as a *descriptive* candidate for hidden-state trajectories of a **pretrained attention-based transformer** (GPT-2, Pythia). Its central theoretical claim is that the key-value asymmetry $W_K \ne W_V$ produces a non-zero antisymmetric operator $\Omega(x) = \frac{\beta}{2}\sum_\mu s_\mu(\beta K x) (V^\mu\otimes K^\mu - K^\mu\otimes V^\mu)$, and that, over smooth trajectories, its linearisation $B_{\text{theory}} = \Omega(\bar x)$ should show up as a velocity-coupled skew term in a second-order damped equation of motion.
>
> The document was written *before* we empirically tested this prediction and *before* we built the prescriptive counterpart. Both are now in hand and the reader should know about them when approaching this note:
>
> 1. **The empirical test.** The constant and affine-in-$x$ skew-$B$ fits reported in §1.5 of [`The_Failure_of_Conservative_Models_to_explain_hidden_state_trajectories.md`](The_Failure_of_Conservative_Models_to_explain_hidden_state_trajectories.md) are direct realisations of the P-rot-6 hypothesis on GPT-2 small over the E-init corpus. They **fail** to beat the static null on the held-out test set; the TRAIN-optimal shrinkage factor collapses to $\le 5\%$ for every configuration, i.e. the optimiser prefers almost none of the fitted gauge field. The companion script is `notebooks/e_init/velocity_coupled_gauge.py`.
> 2. **Why the prediction failed.** The linearisation $B_{\text{eff}} \approx \Omega(\bar x) \delta t$ of §6.2 assumes a *locally quasi-straight* trajectory and *frozen attention* over the integration window. Neither assumption holds for pretrained transformer trajectories: attention is path-dependent on the full prefix $h_{<t}$, the softmax distribution shifts every token, and the per-head torques are rank-deficient. In the Helmholtz language this means P-rot-6 captures the wrong *structural sector* — the obstruction is position-dependence and path-dependence, not a missing velocity coupling. See §3 of the Failure doc for the structural argument.
> 3. **The prescriptive response.** Rather than continue to patch the descriptive ansatz, we flipped the question: can we **construct** a language model whose hidden-state dynamics are, by design, the Euler–Lagrange flow of a single scalar potential — so that the K≠V vortex problem does not arise at all? That flip produced the **Scalar-Potential Language Model (SPLM)** documented in [`Conservative_by_Construction_Language_Models.md`](Conservative_by_Construction_Language_Models.md), [`Training_and_Inference_with_SPLM.md`](Training_and_Inference_with_SPLM.md), and paper v2 §14. SPLM has no separate $W_K$/$W_V$ tensors — its layer update is $\partial V_\theta / \partial x$ for a shared neural scalar $V_\theta$, so $\Omega(x) \equiv 0$ by construction. Trained on the same corpus as the §1.5 tests, SPLM admits a shared potential with $R^2 \approx 0.79$ across depth, versus the $\le 0$ ceiling every attention variant reaches.
>
> The reader should therefore think of P-rot-6 as the **strongest descriptive ansatz we could write down for an attention transformer** — the one with zero free parameters predicted directly from $W_K$, $W_V$, and the context. Its empirical refusal is what licensed the prescriptive pivot. This document remains valuable as (a) a clean derivation of why $K \ne V$ is the structural reason scalar potentials cannot describe attention (§5, the K≠V theorem), and (b) a complete methodological recipe for anyone who wants to replicate the test on a new architecture before deciding it, too, needs an SPLM-style prescriptive alternative.

---

## Table of Contents

1. [Background and Motivation](#1-background-and-motivation)
2. [The Helmholtz Decomposition of Hidden State Forces](#2-the-helmholtz-decomposition-of-hidden-state-forces)
3. [The P-rot-6 Model: Derivation and Structure](#3-the-p-rot-6-model-derivation-and-structure)
4. [Modern Hopfield Networks and Transformer Attention](#4-modern-hopfield-networks-and-transformer-attention)
5. [Why Transformers Are Non-Conservative: The K≠V Theorem](#5-why-transformers-are-non-conservative-the-kv-theorem)
6. [Connecting K≠V Antisymmetry to P-rot-6](#6-connecting-kv-antisymmetry-to-p-rot-6)
7. [Practical Implementation for GPT-2 and Pythia](#7-practical-implementation-for-gpt-2-and-pythia)
8. [Diagnostics and Validation Protocol](#8-diagnostics-and-validation-protocol)
9. [Interpreting Results: Pass, Partial, Fail](#9-interpreting-results-pass-partial-fail)
10. [Next Model Class Hierarchy](#10-next-model-class-hierarchy)
11. [Summary and Open Questions](#11-summary-and-open-questions)
12. [References](#12-references)

---

## 1. Background and Motivation

### 1.1 The Experimental Setting

We study the motion of hidden states $h_t \in \mathbb{R}^d$ across token positions $t = 1, \ldots, T$ at a fixed layer $\ell$ of a decoder-only transformer. The fundamental question is:

> **Does the trajectory $\{h_t\}$ admit a compact dynamical description? If so, what class of differential equation governs it?**

This is not merely a descriptive question. A successful dynamical model of hidden state trajectories would:
- Reveal the geometric structure of semantic space as implicitly defined by the model's weights
- Connect the algebraic properties of attention ($K$, $Q$, $V$ matrices) to observable trajectory geometry
- Provide a mechanistic account of phenomena such as deceleration near semantic attractors, trajectory curvature at syntactic boundaries, and layer-wise representational change

### 1.2 Empirical Starting Point

The following results motivate the current investigation:

| Model class | Equation | Result |
|---|---|---|
| Pure scalar potential | $F = -\nabla V(x)$, any $V$ | **Fails** — residual $\approx$ static null (0.1773) |
| Constant skew position force | $F = -\nabla V + \Omega x$, constant $\Omega$ | **Fails** |
| Velocity-coupled, constant skew | $F = -\nabla V + B\dot x$, constant $B = -B^\top$ | Not yet tested |
| Position-dependent gauge | $F = -\nabla V + B(x)\dot x$ | Not yet tested |
| Riemannian geodesic | $\ddot x^k + \Gamma^k_{ij}\dot x^i \dot x^j = -\gamma \dot x^k$ | Not yet tested |

The failure of pure scalar potentials — including the Gaussian well — and position-coupled skew forces points to a **velocity-coupled solenoidal component** as the next candidate. This is the P-rot-6 model.

### 1.3 What "P-rot-6" Denotes

The designation encodes the model's position in a systematic hierarchy of rotation/solenoidal extensions of the base second-order damped oscillator:

- **P**: potential-augmented (includes $-\nabla V$ term)
- **rot**: rotational/solenoidal extension present
- **6**: sixth variant in the hierarchy, corresponding to velocity-coupling with position-independent skew matrix

---

## 2. The Helmholtz Decomposition of Hidden State Forces

### 2.1 The Decomposition Theorem

By the Helmholtz decomposition, any smooth vector field $F : \mathbb{R}^d \to \mathbb{R}^d$ decomposes uniquely (under appropriate boundary conditions) as:

$$F(x) = -\nabla \varphi(x) + \nabla \times A(x)$$

The first term is **curl-free** (irrotational) and the second is **divergence-free** (solenoidal).

In matrix form, the decomposition manifests through the Jacobian:

$$J F(x) = S(x) + \Omega(x)$$

where the two pieces are respectively

$$S(x) = \frac12 (J F + J F^\top) \quad \text{(symmetric, curl-free)}$$
$$\Omega(x) = \frac12 (J F - J F^\top) \quad \text{(antisymmetric, solenoidal)}$$

A force field is **conservative** (path-independent work, closed-loop $\oint F \cdot dx = 0$) if and only if $\Omega(x) = 0$ everywhere.

### 2.2 Why This Matters for Hidden State Dynamics

The scalar potential models tested in §1.2 model 100% of $F$ with the curl-free component. If the true force field has non-zero $\Omega(x)$, no scalar potential — regardless of shape, depth, or parameterization — can represent it. This is not a capacity or fitting failure. It is a **structural impossibility**.

### 2.3 The Full Force Taxonomy

For second-order dynamics $m\ddot x = F(x, \dot x)$, the Helmholtz decomposition generalizes to phase space $(x, \dot x)$:

$$F(x, \dot x) = -\nabla \varphi(x) + F_{\text{sol}}(x) + B(x) \dot x + D(x) \dot x + \text{nonlinear terms}$$

Reading term by term:

- $-\nabla \varphi(x)$ — conservative, position-only
- $F_{\text{sol}}(x)$ — solenoidal, position-only (curl $\ne 0$)
- $B(x) \dot x$ — gyroscopic / magnetic (velocity-coupled)
- $D(x) \dot x$ — symmetric dissipation (drag)

Each row is orthogonal to the others in function space. A scalar potential tests only the first. P-rot-6 adds the third.

---

## 3. The P-rot-6 Model: Derivation and Structure

### 3.1 The Equation of Motion

$$m\ddot x = -\nabla V(x) + B_\ell \dot x - m\gamma \dot x, \qquad B_\ell = -B_\ell^\top \in \mathbb{R}^{k \times k}$$

In component form:

$$m \ddot x_i = -\frac{\partial V}{\partial x_i} + \sum_j (B_\ell)_{ij} \dot x_j - m\gamma \dot x_i$$

### 3.2 Why $B_\ell$ Must Be Skew-Symmetric

The velocity coupling $B\dot x$ decomposes as:

$$B\dot x = \frac12 (B + B^\top)\dot x + \frac12 (B - B^\top)\dot x$$

The first piece (symmetric) is equivalent to anisotropic damping and is absorbed into the $-m\gamma\dot x$ term. The second piece (skew-symmetric) is the gyroscopic force — it is structurally new and does no work. Furthermore:

$$\text{Power delivered by } B\dot x : \quad P = \dot x^\top B \dot x = 0 \ \ \forall \dot x \iff B = -B^\top$$

The skew-symmetric constraint ensures the gyroscopic force **does zero work** — it curves trajectories without changing kinetic energy. This is the magnetic / Lorentz force analogy: a charged particle in a magnetic field is deflected but not accelerated.

### 3.3 The Energy Function

P-rot-6 admits a modified energy:

$$E(x, \dot x) = \frac12 m \|\dot x\|^2 + V(x)$$

Taking the time derivative along trajectories:

$$\begin{aligned}
\frac{dE}{dt} &= m \dot x^\top \ddot x + \dot x^\top \nabla V \\
 &= \dot x^\top (-\nabla V + B\dot x - m\gamma \dot x) + \dot x^\top \nabla V \\
 &= \dot x^\top B \dot x - m\gamma \|\dot x\|^2 \\
 &= 0 - m\gamma \|\dot x\|^2 \ =\ -m\gamma \|\dot x\|^2.
\end{aligned}$$

**The skew-symmetric $B$ contributes exactly zero to energy dissipation.** The system is a damped, rotating system with well-defined Lyapunov function $E(x, \dot x)$. This is a key sanity check for any fitted $B$: if energy is not monotonically decreasing (up to noise), $B$ is not purely skew.

### 3.4 The Number of Free Parameters

$B \in \mathbb{R}^{k \times k}$ skew-symmetric has $k(k-1)/2$ free parameters.

| Subspace dimension $k$ | Parameters |
|---|---|
| 10 | 45 |
| 50 | 1225 |
| 100 | 4950 |
| 768 (full BERT/GPT-2) | 295,128 |

For practical fitting, **PCA reduction to $k = 20\text{--}50$** of the velocity principal subspace is strongly recommended before estimating $B$.

### 3.5 Continuous-Time Formulation

For direct comparison with the ODE literature, P-rot-6 can be written as:

$$\begin{aligned}
\dot x &= v, \\
\dot v &= -\frac{\nabla V(x)}{m} + \left(\frac{B}{m} - \gamma I\right) v.
\end{aligned}$$

or in Hamiltonian form with a non-canonical Poisson bracket. The phase-space flow is:

$$\frac{d}{dt}\begin{bmatrix} x \\ v \end{bmatrix} = \begin{bmatrix} v \\ -\nabla V / m + (B/m - \gamma I) v \end{bmatrix}$$

The Jacobian of the right-hand side at a fixed point $(x^\ast, 0)$ is:

$$J^\ast = \begin{bmatrix} 0 & I \\ -H_V(x^\ast)/m & B/m - \gamma I \end{bmatrix}$$

where $H_V = \nabla^2 V$ is the Hessian of $V$. Stability requires the real parts of the eigenvalues of $J^\ast$ to be negative — dominated by the damping $\gamma$.

---

## 4. Modern Hopfield Networks and Transformer Attention

### 4.1 The Krotov-Hopfield Hierarchy

**Classical Hopfield (1982).**

$$E = -\frac12 x^\top \Xi^\top \Xi x + \frac12 \|x\|^2$$
$$F = -\nabla E = \Xi^\top (\Xi x) - x \quad \text{(Hebbian retrieval)}$$

Capacity: $\sim 0.14 d$.

**Krotov-Hopfield 2016 (polynomial interactions).**

$$E = -\sum_\mu F_n(\xi^\mu \cdot x) + \frac12 \|x\|^2, \qquad F_n(z) = \frac{z^n}{n}$$

Capacity: $\sim d^{n-1}$ (superlinear in $d$ for $n > 2$).

**Ramsauer et al. 2020 (exponential / softmax limit).**

Take $F_n \to \exp$ as $n \to \infty$:

$$E(x) = -\frac{1}{\beta}\log \sum_\mu \exp(\beta \xi^\mu \cdot x) + \frac12 \|x\|^2 = -\frac{1}{\beta}\log Z(x) + \frac12 \|x\|^2$$
$$F(x) = -\nabla E = \Xi^\top \text{softmax}(\beta \Xi x) - x$$

Capacity: $\sim \exp(d)$ (exponential in dimension).

The synchronous update $x_{\text{new}} = \Xi^\top \text{softmax}(\beta \Xi x)$ is **exactly** softmax self-attention.

### 4.2 The Transformer Identification

Standard scaled dot-product attention:

$$\text{Attention}(Q, K, V) = V \cdot \text{softmax}\left(\frac{K^\top Q}{\sqrt d}\right)$$

Maps to Hopfield retrieval under:

| Transformer | Hopfield |
|---|---|
| Query $Q$ | State being retrieved $x$ |
| Key matrix $K$ | Stored pattern index $\Xi$ |
| Value matrix $V$ | Retrieved pattern content $\Xi$ (if $K = V$) |
| Scaling $1/\sqrt d$ | Inverse temperature $\beta$ |
| softmax | Boltzmann retrieval distribution |

**One transformer attention operation = one synchronous Hopfield update step.**

### 4.3 The Autoregressive Setting

In a decoder-only transformer (GPT-2, Pythia), the causal mask enforces:

$$\text{Attention}(Q_t, K_{\le t}, V_{\le t}) = V_{\le t} \cdot \text{softmax}\left(\frac{K_{\le t}^\top Q_t}{\sqrt d}\right)$$

The "stored patterns" are the keys and values of all preceding tokens. The hidden state at position $t$ is being retrieved as a query against a **growing memory bank**. This means the effective Hopfield network grows with sequence length — the attractor landscape changes at every token position.

---

## 5. Why Transformers Are Non-Conservative: The K≠V Theorem

### 5.1 The Conservative Case ($K = V$)

When $K = V = \Xi$ (pure Hopfield, auto-associative):

$$F(x) = \Xi^\top \text{softmax}(\beta \Xi x) - x$$

Jacobian:

$$J F(x) = \beta \Xi^\top [\mathrm{diag}(s) - s s^\top]\Xi - I$$

where $s = \text{softmax}(\beta \Xi x)$. The matrix $\mathrm{diag}(s) - s s^\top$ is symmetric positive semidefinite (it is the Jacobian of softmax). Therefore $J F(x)$ is **symmetric** for all $x$. Symmetric Jacobian $\Leftrightarrow$ path-independent work $\Leftrightarrow$ **conservative**.

**Proof of conservativity.** The existence of a scalar potential is equivalent to $\partial F_i / \partial x_j = \partial F_j / \partial x_i$ for all $i, j$. This holds exactly when $J F$ is symmetric, which is guaranteed when $K = V$.

### 5.2 The Non-Conservative Case ($K \ne V$)

In every real transformer, $K = W_K \cdot \text{context}$ and $V = W_V \cdot \text{context}$ with $W_K \ne W_V$. The force becomes:

$$F(x) = V^\top \text{softmax}(\beta K x) - x$$

Jacobian:

$$(J F)_{ij} = \beta \sum_\mu \text{softmax}_\mu(\beta K x) (K^\mu_j - [K s]_j) V^\mu_i - \delta_{ij}$$

The **antisymmetric part** (the solenoidal component):

$$\Omega(x) = \frac12 (J F - J F^\top) = \frac{\beta}{2}\sum_\mu \text{softmax}_\mu(\beta K x) (V^\mu \otimes K^\mu - K^\mu \otimes V^\mu)$$

This is an attention-weighted sum of antisymmetric outer products.

### 5.3 The Non-Conservativity Theorem

**Theorem.** The attention force $F(x) = V^\top \text{softmax}(\beta K x) - x$ is conservative if and only if $V^\mu \parallel K^\mu$ for all $\mu$ (i.e. each key and its associated value are collinear).

**Proof sketch.** $\Omega(x) = 0$ for all $x$ requires each outer product $V^\mu \otimes K^\mu - K^\mu \otimes V^\mu = 0$, which holds iff $V^\mu = \lambda^\mu K^\mu$ for scalars $\lambda^\mu$ — collinearity. Since $V^\mu = W_V \xi^\mu$ and $K^\mu = W_K \xi^\mu$, collinearity for all context tokens requires $W_V = \Lambda W_K$ for diagonal $\Lambda$ — a measure-zero set of weight configurations, never achieved in trained models. $\blacksquare$

### 5.4 The Geometric Interpretation

Each stored pattern $\mu$ contributes a **vortex** in the plane spanned by $\{K^\mu, V^\mu\}$:

$$\Omega^\mu = V^\mu \otimes K^\mu - K^\mu \otimes V^\mu \in \mathbb{R}^{d \times d} \quad \text{(rank-2 antisymmetric)}$$

The full solenoidal field is a **softmax-weighted superposition of vortices**, where the weights change with position $x$ (via the attention distribution). This is:

- **Spatially inhomogeneous**: different context tokens dominate at different hidden-state positions
- **Dynamically active**: as $x$ evolves, the attention weights shift, changing which vortices are active
- **Structurally irreducible**: cannot be represented by any scalar potential

---

## 6. Connecting K≠V Antisymmetry to P-rot-6

### 6.1 The Key Structural Difference

The $K \ne V$ solenoidal component is a **position-only** force:

$$F_{\text{sol}}(x) \quad \text{depends on } x, \text{ not } \dot x$$

The P-rot-6 gyroscopic term is a **velocity-coupled** force:

$$F_{\text{gyro}} = B \dot x \quad \text{depends on } \dot x, \text{ not } x$$

These occupy **distinct sectors** of the Helmholtz decomposition. The identification $\Omega(x) \leftrightarrow B$ is not exact — it requires additional conditions.

### 6.2 When the Identification Is Valid

Consider a trajectory $x(t)$ linearized around a reference $\bar x(t)$:

$$x(t) = \bar x(t) + \delta x(t)$$

Force on perturbed trajectory:

$$F(\bar x + \delta x) \approx F(\bar x) + J F(\bar x) \delta x = F(\bar x) + S(\bar x) \delta x + \Omega(\bar x) \delta x$$

For a locally quasi-straight trajectory over a short time window $\delta t$:

$$\delta x(t) \approx \dot x(t) \delta t$$

The antisymmetric correction becomes:

$$\Omega(\bar x) \delta x \approx \Omega(\bar x) \dot x \delta t$$

This gives an **effective velocity coupling** with:

$$B_{\text{eff}} \approx \Omega(\bar x) \delta t$$

The identification holds when:

| Condition | Mathematical form |
|---|---|
| Locally straight trajectory | $\|\ddot x\| \delta t \ll \|\dot x\|$ |
| Frozen attention | $\|\partial s / \partial x\| \|\delta x\| \ll \|s\|$ |
| Short window | $\delta t \ll 1/(\beta \|K\|)$ |

### 6.3 The Phase-Space Lifting

A cleaner connection avoids linearization. Lift to phase space $z = (x, v)$:

$$\dot z = G(z) = \begin{bmatrix} v \\ F(x)/m - \gamma v \end{bmatrix}$$

The Jacobian of $G$ in phase space:

$$J G(z) = \begin{bmatrix} 0 & I \\ J F(x)/m & -\gamma I \end{bmatrix}$$

The off-diagonal block $J F(x)/m$ contains $\Omega(x)/m$. In phase space, the solenoidal component of the position-space force **appears as a coupling between the velocity equation and the position coordinate** — structurally analogous to a velocity coupling when projected back.

This projection is exact when the phase-space manifold is locally a product. For transformer trajectories, this holds approximately in **slow-varying semantic regions** (plateau phases between transitions).

### 6.4 The Theoretical B Matrix

The P-rot-6 $B$ matrix theoretically predicted by transformer weights is:

$$B_{\text{theory}}(\bar x) = \frac{\beta}{2}\sum_\mu \text{softmax}_\mu(\beta K \bar x) (V^\mu \otimes K^\mu - K^\mu \otimes V^\mu)$$

where the sum is over context tokens and $\bar x$ is a reference hidden state (e.g. trajectory mean). This has **zero free parameters** — it is entirely determined by:

- The $K$ and $V$ weight matrices $W_K$, $W_V$
- The context token representations at layer $\ell$
- The reference state $\bar x$

This theoretical prediction can be tested directly against the empirically fitted $B$ — the residual difference quantifies the validity of the linearization.

---

## 7. Practical Implementation for GPT-2 and Pythia

### 7.1 Setup and Dependencies

```python
import torch
import numpy as np
from transformers import GPT2Model, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
import scipy.linalg

device = "cuda" if torch.cuda.is_available() else "cpu"
```

### 7.2 Extracting Hidden State Trajectories

```python
def extract_hidden_trajectories(model_name, texts, layer_idx, device="cpu"):
    """
    Extract hidden states across token positions at a given layer.
    
    Returns:
        H : list of (T_i, d) arrays — one per text
    """
    if "gpt2" in model_name.lower():
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2Model.from_pretrained(model_name, output_hidden_states=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, output_hidden_states=True
        )
    
    model.eval().to(device)
    
    trajectories = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # hidden_states: tuple of (n_layers+1,) each (1, T, d)
        h = outputs.hidden_states[layer_idx]     # (1, T, d)
        trajectories.append(h.squeeze(0).cpu().numpy())
    
    return trajectories


# Example usage
texts = [...]   # your corpus
layer = 6       # middle layer of GPT-2 small (12 total)

H_list = extract_hidden_trajectories("gpt2", texts, layer_idx=layer)
```

### 7.3 Extracting K and V Weight Matrices

```python
def extract_KV_weights(model, layer_idx, model_type="gpt2"):
    """
    Extract W_K and W_V from attention head at given layer.
    Returns combined (d, d) matrices across all heads.
    """
    if model_type == "gpt2":
        # GPT-2 stores Q, K, V concatenated in c_attn
        attn = model.transformer.h[layer_idx].attn
        d_model = attn.embed_dim
        
        # c_attn.weight is (3*d, d) — split into Q, K, V
        W_QKV = attn.c_attn.weight.detach().cpu().numpy()  # (3d, d)
        W_Q = W_QKV[:d_model, :]
        W_K = W_QKV[d_model:2*d_model, :]
        W_V = W_QKV[2*d_model:, :]
        
    elif model_type == "pythia":
        # Pythia (GPT-NeoX architecture) has separate Q, K, V projections
        attn = model.gpt_neox.layers[layer_idx].attention
        W_Q = attn.query_key_value.weight.detach().cpu().numpy()
        # Pythia stores QKV interleaved — requires reshaping
        d_model = attn.hidden_size
        n_heads = attn.num_attention_heads
        d_head  = d_model // n_heads
        
        W_QKV = attn.query_key_value.weight.detach().cpu().numpy()
        # Shape: (3*d_model, d_model) with heads interleaved
        W_QKV_r = W_QKV.reshape(n_heads, 3, d_head, d_model)
        W_Q = W_QKV_r[:, 0, :, :].reshape(d_model, d_model)
        W_K = W_QKV_r[:, 1, :, :].reshape(d_model, d_model)
        W_V = W_QKV_r[:, 2, :, :].reshape(d_model, d_model)
    
    return W_Q, W_K, W_V
```

### 7.4 Computing the Theoretical B Matrix

```python
def compute_hopfield_curl(W_K, W_V, context_hidden, x_ref, beta=None, d_head=64):
    """
    Compute the theoretical P-rot-6 B matrix from transformer K≠V antisymmetry.
    
    Args:
        W_K         : (d, d) key projection matrix
        W_V         : (d, d) value projection matrix
        context_hidden : (M, d) hidden states of context tokens at this layer
        x_ref       : (d,) reference hidden state (trajectory mean)
        beta        : inverse temperature (default: 1/sqrt(d_head))
    
    Returns:
        B           : (d, d) skew-symmetric matrix — the theoretical P-rot-6 B
        Omega_per_token : (M, d, d) per-token vortex contribution
    """
    if beta is None:
        beta = d_head ** -0.5
    
    M, d = context_hidden.shape
    
    # Project context to key and value spaces
    K = context_hidden @ W_K.T     # (M, d_k)
    V = context_hidden @ W_V.T     # (M, d_v)
    
    # Attention scores at reference state
    scores = beta * (K @ x_ref)    # (M,) raw logits
    s = np.exp(scores - scores.max())
    s /= s.sum()                   # (M,) softmax weights
    
    # Per-token antisymmetric outer products (vortex contributions)
    Omega_per_token = np.zeros((M, d, d))
    for mu in range(M):
        # V^mu tensor K^mu - K^mu tensor V^mu : antisymmetric rank-2 matrix
        Omega_per_token[mu] = (
            np.outer(V[mu], K[mu]) - np.outer(K[mu], V[mu])
        )
    
    # Attention-weighted superposition
    B_raw = beta/2 * np.einsum('m,mij->ij', s, Omega_per_token)
    
    # Enforce exact skew-symmetry (numerical cleanup)
    B = (B_raw - B_raw.T) / 2
    
    return B, Omega_per_token


# Example: compute B at mean hidden state across a trajectory
H = H_list[0]       # (T, d) single trajectory
x_ref = H.mean(0)   # (d,) reference state

W_Q, W_K, W_V = extract_KV_weights(model, layer_idx=6)
B_theory, Omega_tokens = compute_hopfield_curl(W_K, W_V, H, x_ref)

print(f"B shape: {B_theory.shape}")
print(f"B skew check (should be ~0): {np.max(np.abs(B_theory + B_theory.T)):.2e}")
print(f"B Frobenius norm: {np.linalg.norm(B_theory):.4f}")
```

### 7.5 PCA Reduction to Velocity Subspace

Before fitting $B$ empirically, reduce to the subspace where dynamics actually live:

```python
def velocity_pca(H_list, k=30):
    """
    Compute PCA basis of the velocity field across all trajectories.
    Returns projection matrix for dimension reduction.
    """
    # Collect all velocity vectors
    velocities = []
    for H in H_list:
        v = np.diff(H, axis=0)   # (T-1, d)
        velocities.append(v)
    
    V_all = np.vstack(velocities)   # (N_total, d)
    
    pca = PCA(n_components=k)
    pca.fit(V_all)
    
    print(f"Velocity variance explained by {k} PCs: "
          f"{pca.explained_variance_ratio_.sum():.3f}")
    
    return pca.components_    # (k, d) — rows are principal directions


# Project everything to k-dimensional velocity subspace
P = velocity_pca(H_list, k=30)     # (k, d)

# Project hidden states and velocities
H_proj_list = [H @ P.T for H in H_list]      # (T, k) each
V_proj_list = [np.diff(H, axis=0) @ P.T      # (T-1, k) each
               for H in H_list]
```

### 7.6 Computing Observed Accelerations and Residuals

```python
def compute_second_order_residual(H, V, gamma, V_fn, B=None):
    """
    Compute residual of P-rot-6 equation of motion.
    
    r[t] = xddot[t] - ( -grad V(x[t]) + B @ xdot[t] - gamma * xdot[t] )
    
    Uses finite differences for xdot and xddot.
    """
    # Positions, velocities, accelerations via finite difference
    x = H[1:-1]                              # (T-2, k) interior points
    v = (H[2:] - H[:-2]) / 2                # (T-2, k) central diff velocity
    a = H[2:] - 2*H[1:-1] + H[:-2]         # (T-2, k) second difference

    # Potential force
    F_pot = np.array([-V_fn(x[t]) for t in range(len(x))])  # (T-2, k)
    
    # Damping force
    F_damp = -gamma * v                      # (T-2, k)
    
    # Gyroscopic force (P-rot-6 term)
    F_gyro = np.zeros_like(v)
    if B is not None:
        F_gyro = v @ B.T                     # (T-2, k), using B projected to subspace

    # Residual
    F_model = F_pot + F_damp + F_gyro
    residual = a - F_model
    
    return np.mean(np.linalg.norm(residual, axis=1))


# Static null baseline
null_residual = np.mean([
    np.mean(np.linalg.norm(np.diff(H, axis=0), axis=1))
    for H in H_proj_list
])
print(f"Static null residual: {null_residual:.4f}")
```

### 7.7 Empirical Fitting of B via Linear Regression

```python
def fit_B_empirical(H_list, gamma, V_fn, k):
    """
    Fit the skew-symmetric B matrix by linear regression on velocity residuals.
    
    Solves: r[t] ~= B @ v[t]
    where r[t] is the residual after removing potential and damping.
    
    Returns B_fit (k, k) skew-symmetric.
    """
    R_all, V_all = [], []
    
    for H in H_list:
        x = H[1:-1]
        v = (H[2:] - H[:-2]) / 2
        a = H[2:] - 2*H[1:-1] + H[:-2]
        
        F_pot  = np.array([-V_fn(xt) for xt in x])
        F_damp = -gamma * v
        
        # Velocity residual: what's left after pot + damping
        r = a - F_pot - F_damp        # (T-2, k)
        R_all.append(r)
        V_all.append(v)
    
    R = np.vstack(R_all)    # (N, k)
    V = np.vstack(V_all)    # (N, k)
    
    # Least squares: R ~= V @ B^T  =>  B^T = (V^T V)^-1 V^T R
    B_T_ls = np.linalg.lstsq(V, R, rcond=None)[0]   # (k, k)
    B_ls   = B_T_ls.T
    
    # Project to skew-symmetric manifold
    B_fit = (B_ls - B_ls.T) / 2
    
    return B_fit


B_empirical = fit_B_empirical(H_proj_list, gamma=0.1, V_fn=V_fn, k=30)
```

### 7.8 Projecting the Theoretical B to the Velocity Subspace

```python
def project_B_to_subspace(B_theory, P):
    """
    Project full-dimensional B_theory (d, d) to PCA subspace (k, k).
    B_proj = P @ B_theory @ P^T
    """
    B_proj = P @ B_theory @ P.T       # (k, k)
    # Re-enforce skew-symmetry after projection
    B_proj = (B_proj - B_proj.T) / 2
    return B_proj


B_theory_proj = project_B_to_subspace(B_theory, P)

# Compare theoretical vs empirical
alignment = np.trace(B_theory_proj.T @ B_empirical) / (
    np.linalg.norm(B_theory_proj) * np.linalg.norm(B_empirical) + 1e-10
)
print(f"Theoretical B alignment with empirical B: {alignment:.4f}")
# 1.0 -> perfect alignment (theoretical prediction is correct)
# 0.0 -> orthogonal (theory misses the empirical structure entirely)
```

---

## 8. Diagnostics and Validation Protocol

### 8.1 The Spatial Variation Diagnostic

Before committing to a constant $B$, measure how much $\Omega(x)$ varies along the trajectory:

```python
def spatial_variation_diagnostic(W_K, W_V, H, beta, d_head=64):
    """
    Compute Omega(x_t) at each trajectory point and measure variation.
    
    High variation ratio -> constant B not justified -> need B(x).
    Low variation ratio  -> constant B is a good approximation.
    """
    Omegas = []
    for x_t in H:
        B_t, _ = compute_hopfield_curl(W_K, W_V, H, x_t, beta, d_head)
        Omegas.append(B_t)
    
    Omega_mean = np.mean(Omegas, axis=0)
    
    deviations = [np.linalg.norm(O - Omega_mean) for O in Omegas]
    variation_ratio = np.std(deviations) / (np.linalg.norm(Omega_mean) + 1e-10)
    
    print(f"Omega spatial variation ratio: {variation_ratio:.4f}")
    print(f"  < 0.05 -> constant B is valid")
    print(f"  0.05-0.20 -> moderate position-dependence, B(x) recommended")
    print(f"  > 0.20 -> strong position-dependence, constant B will fail")
    
    return variation_ratio, Omegas
```

### 8.2 The Energy Monotonicity Check

```python
def check_energy_monotonicity(H, B_fit, gamma, V_fn):
    """
    Verify that dE/dt <= 0 along trajectories (required for valid B).
    If dE/dt > 0 consistently, B is not purely skew.
    """
    v = np.diff(H, axis=0)           # (T-1, k)
    x = H[:-1]
    
    V_vals  = np.array([V_fn(xt) for xt in x])
    KE      = 0.5 * np.sum(v**2, axis=1)
    E       = KE + V_vals
    dE_dt   = np.diff(E)
    
    # Theoretical dE/dt = -gamma * ||v||^2
    dE_theory = -gamma * np.sum(v[:-1]**2, axis=1)
    
    # Check gyroscopic work (should be ~0)
    gyro_work = np.array([v[t] @ B_fit @ v[t] for t in range(len(v)-1)])
    
    print(f"Mean gyroscopic work (should be ~0): {gyro_work.mean():.4e}")
    print(f"Max |gyroscopic work|: {np.abs(gyro_work).max():.4e}")
    print(f"Skew-symmetry of B_fit: {np.max(np.abs(B_fit + B_fit.T)):.4e}")
    
    return gyro_work
```

### 8.3 The Residual Comparison Table

Run the full model comparison and produce a residual table:

```python
def full_model_comparison(H_list, gamma, V_fn, B_theory_proj, B_empirical, P):
    """
    Compare residuals across the model hierarchy.
    """
    results = {}
    
    # 1. Static null
    null = np.mean([
        np.mean(np.linalg.norm(np.diff(H, axis=0), axis=1))
        for H in H_list
    ])
    results["Static null"] = null
    
    # 2. Scalar potential only
    res_V = np.mean([
        compute_second_order_residual(H, None, gamma, V_fn, B=None)
        for H in H_list
    ])
    results["-grad V only"] = res_V
    
    # 3. P-rot-6 with theoretical B (zero free parameters)
    res_theory = np.mean([
        compute_second_order_residual(H, None, gamma, V_fn, B=B_theory_proj)
        for H in H_list
    ])
    results["P-rot-6 (B_theory)"] = res_theory
    
    # 4. P-rot-6 with empirically fitted B
    res_empirical = np.mean([
        compute_second_order_residual(H, None, gamma, V_fn, B=B_empirical)
        for H in H_list
    ])
    results["P-rot-6 (B_fit)"] = res_empirical
    
    print("\n=== Residual Comparison ===")
    print(f"{'Model':<35} {'Residual':>12} {'vs Null':>10}")
    print("-" * 60)
    for name, res in results.items():
        delta = (null - res) / null * 100
        print(f"{name:<35} {res:>12.4f} {delta:>+9.1f}%")
    
    return results
```

### 8.4 Head-Level Analysis

Different attention heads may contribute very differently to the solenoidal dynamics:

```python
def per_head_curl_analysis(W_K_heads, W_V_heads, context, x_ref, beta, d_head):
    """
    Compute B matrix contribution from each attention head separately.
    
    Args:
        W_K_heads : (n_heads, d_head, d_model) 
        W_V_heads : (n_heads, d_head, d_model)
    """
    n_heads = W_K_heads.shape[0]
    B_per_head = []
    
    for h in range(n_heads):
        W_K_h = W_K_heads[h]   # (d_head, d_model)
        W_V_h = W_V_heads[h]
        
        K_h = context @ W_K_h.T    # (M, d_head)
        V_h = context @ W_V_h.T    # (M, d_head)
        
        # Attention weights for this head
        scores_h = beta * (K_h @ (W_K_h @ x_ref))
        s_h = np.exp(scores_h - scores_h.max())
        s_h /= s_h.sum()
        
        # Vortex matrix in d_head space
        B_h = beta/2 * sum(
            s_h[mu] * (np.outer(V_h[mu], K_h[mu]) - np.outer(K_h[mu], V_h[mu]))
            for mu in range(len(s_h))
        )
        B_h = (B_h - B_h.T) / 2
        B_per_head.append(B_h)
        
        print(f"Head {h:2d}: ||B_h||_F = {np.linalg.norm(B_h):.4f}")
    
    return B_per_head
```

---

## 9. Interpreting Results: Pass, Partial, Fail

### 9.1 Decision Matrix

| Residual outcome | Interpretation | Next action |
|---|---|---|
| $B_{\text{theory}} \approx B_{\text{fit}}$, both beat null | **Theory confirmed**: $K \ne V$ antisymmetry drives solenoidal dynamics; linearization holds | Measure curvature of $B(x)$; check head decomposition |
| $B_{\text{fit}}$ beats null, $B_{\text{theory}}$ doesn't | **Structure correct, theory approximate**: velocity coupling exists but linearization is inaccurate | Fit full $B(x)$ network; measure spatial variation ratio |
| Both beat null but weakly ($< 5\%$) | **Partial success**: $B$ captures some solenoidal structure | Check if head-specific $B$ gives stronger signal |
| Neither beats null | **Velocity coupling absent**: solenoidal dynamics are position-driven (pure $F_{\text{sol}}$) | Move to position-dependent $B(x)$; check MLP contribution |
| $B_{\text{fit}}$ beats null, but $B_{\text{fit}} + B_{\text{theory}}$ orthogonal | **Wrong vortex structure**: $K \ne V$ is not the source | Examine MLP, LayerNorm, or inter-layer contributions |

### 9.2 What Each Outcome Implies About Semantic Space Geometry

**P-rot-6 succeeds (globally).**
The semantic space has an approximately **uniform magnetic-like field** — every region of meaning-space is subject to the same rotational bias. Semantic trajectories are helical around the $K \ne V$ vortex axes. The relevant geometry is a flat space with a constant connection form.

**P-rot-6 succeeds locally (high spatial variation ratio).**
The magnetic field is **inhomogeneous** — the rotational structure depends on position. Near certain semantic attractors, the vortex intensity increases (high softmax concentration $\Rightarrow$ single dominant token's $K$/$V$ pair dominates $\Omega$). This is the **position-dependent gauge field** $B(x)$ scenario.

**P-rot-6 fails entirely.**
The dynamics are not velocity-linear in any sense. This points toward either:

- **Riemannian geodesic structure**: the curvature IS the force, not a correction to Newtonian dynamics
- **Non-Markovian dynamics**: the effective force at time $t$ depends on the full history $h_{<t}$, not just $h_t$ and $\dot h_t$
- **Discrete punctuation**: the trajectory is not smooth enough for second-order ODE models to apply

---

## 10. Next Model Class Hierarchy

Having established the hierarchy through P-rot-6, the next steps follow a principled progression:

### 10.1 Position-Dependent Gauge Field $B(x)$

$$m\ddot x = -\nabla V(x) + B(x) \dot x - m\gamma \dot x, \qquad B(x) = -B(x)^\top$$

**Parameterization options** (in order of complexity):

1. **RBF interpolation**: $B(x) = \sum_c w_c B_c \varphi(\|x - c\|)$ — cluster-based, interpretable
2. **Low-rank skew network**: $B(x) = U(x)\Sigma(x) V(x)^\top - V(x)\Sigma(x) U(x)^\top$ with neural $U, V, \Sigma$
3. **Full gauge network**: unrestricted MLP output projected to skew manifold via $(B - B^\top)/2$

The theoretical prediction from transformer weights becomes:

```python
def B_theoretical_at_x(x, W_K, W_V, context, beta, d_head):
    """Position-dependent version: Omega(x) evaluated at current state."""
    K = context @ W_K.T
    V = context @ W_V.T
    s = softmax(beta * K @ x)
    return beta/2 * sum(s[mu] * (outer(V[mu], K[mu]) - outer(K[mu], V[mu]))
                        for mu in range(len(s)))
```

This is the **zero-free-parameter theoretical prediction** for $B(x)$ — fully determined by transformer weights.

### 10.2 Riemannian Geodesic Formulation

If all gauge field models fail, the hypothesis shifts to:

$$\ddot x^k + \Gamma^k_{ij}(x) \dot x^i \dot x^j = -\gamma \dot x^k$$

where $\Gamma^k_{ij}$ are the Christoffel symbols of an implicit Riemannian metric on semantic space.

The metric can be estimated from the Fisher information of the transformer's output distribution:

$$g_{ij}(x) = \mathbb{E}_y\left[\frac{\partial \log p(y \mid x)}{\partial x_i} \frac{\partial \log p(y \mid x)}{\partial x_j}\right]$$

This directly connects representational geometry to predictive uncertainty.

### 10.3 Full Decision Tree

```
                        ┌─────────────────────┐
                        │   Constant B test   │
                        │     (P-rot-6)       │
                        └──────────┬──────────┘
                    Pass ◄─────────┴─────────► Fail
                      │                          │
         ┌────────────▼──────────┐    ┌──────────▼──────────────┐
         │  Spatial variation    │    │   Position-dependent     │
         │  ratio diagnostic     │    │      B(x) model          │
         └────────────┬──────────┘    └──────────┬──────────────┘
             Low ◄────┴────► High         Pass ◄─┴─► Fail
              │                │            │           │
      ┌───────▼──────┐ ┌───────▼──────┐    │    ┌──────▼─────────┐
      │  Constant B  │ │  B(x) model  │    │    │  Riemannian    │
      │  sufficient  │ │  required    │    │    │  geodesic      │
      └──────────────┘ └──────────────┘    │    └────────────────┘
                                           │
                              ┌────────────▼─────────────┐
                              │  Mechanistic confirmation │
                              │  K≠V drives solenoidal   │
                              │  dynamics                │
                              └──────────────────────────┘
```

---

## 11. Summary and Open Questions

### 11.1 What P-rot-6 Tests

P-rot-6 is the **minimal model** that can capture velocity-coupled solenoidal dynamics. Its $B$ matrix is:

- **Theoretically motivated** by the $K \ne V$ antisymmetry in transformer attention
- **Zero-free-parameter predictable** from $W_K$, $W_V$, and the context
- **Empirically fittable** by a single linear regression on velocity residuals
- **Physically interpretable** as a gyroscopic force doing no work, curving trajectories without dissipation

### 11.2 The Theoretical Prediction Chain

1. $W_K \ne W_V$ (by design in every transformer)
2. $\Omega(x) = \dfrac{\beta}{2}\sum_\mu \text{softmax}_\mu(\beta K x) (V^\mu \otimes K^\mu - K^\mu \otimes V^\mu) \ne 0$
3. Force field is non-conservative (curl $\ne 0$)
4. Scalar potentials structurally fail (confirmed empirically)
5. Linearized approximation: $\Omega(\bar x) \dot x \approx B \dot x$ (valid for smooth trajectories)
6. P-rot-6 with $B_{\text{theory}} = \Omega(\bar x)$ is a zero-free-parameter mechanistic model
7. Test: does the residual drop significantly when $B_{\text{theory}}$ is added?

### 11.3 Open Questions

1. **Multi-head superposition.** Does each head contribute an independent vortex, or do heads interfere constructively/destructively? The per-head analysis in §8.4 addresses this, but the algebraic structure of head superposition in the full $B$ matrix is not yet characterized.

2. **Layer depth dependence.** Does $\|\Omega(x)\|$ grow or decay with layer depth? If it grows, solenoidal dynamics become more dominant at later layers — consistent with the view that deep layers perform semantic integration (curved trajectories) while early layers perform syntactic processing (more linear).

3. **Curvature tensor of the gauge field.** If $B(x)$ is position-dependent, its curvature tensor $F_{\mu\nu} = \partial_\mu B_\nu - \partial_\nu B_\mu + [B_\mu, B_\nu]$ determines whether there are topological obstructions to gauge-away the solenoidal component. Non-zero curvature would constitute evidence for genuine topological structure in semantic space. The commutator term $[B_\mu, B_\nu]$ is the non-abelian part and is precisely the object studied in the *Gauge Theory* and *Lie Groups for Gauge Theory* tutorials cited in §12; see §7 of the former for the field-strength tensor and §8 for its non-abelian generalisation.

4. **Connection to induction heads.** Mechanistic interpretability identifies induction heads as key circuits for in-context learning. These heads have strongly structured $K$/$V$ relationships. Their contribution to $\Omega(x)$ is likely dominant — testing P-rot-6 on induction-head-ablated models would isolate the induction head's geometric contribution.

5. **Temperature scaling.** As $\beta \to \infty$ (sharp attention), $\Omega(x)$ becomes dominated by the single highest-attention token, and its vortex becomes the only active one. As $\beta \to 0$, contributions average out and $\Omega \to 0$. This predicts a phase transition in solenoidal dynamics as a function of attention temperature.

---

## 12. References

- Hopfield, J.J. (1982). Neural networks and physical systems with emergent collective computational abilities. *PNAS*, 79(8), 2554–2558.
- Krotov, D., & Hopfield, J.J. (2016). Dense associative memory for pattern recognition. *NeurIPS 29*.
- Krotov, D., & Hopfield, J.J. (2020). Large associative memory problem in neuroscience and machine learning. *ICLR 2021*.
- Ramsauer, H., et al. (2020). Hopfield Networks Is All You Need. *ICLR 2021*. arXiv:2008.02217.
- Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS 30*.
- Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners. OpenAI Blog. [GPT-2]
- Biderman, S., et al. (2023). Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling. *ICML 2023*.
- Elhage, N., et al. (2021). A Mathematical Framework for Transformer Circuits. *Transformer Circuits Thread*.
- Olshausen, B.A., & Field, D.J. (1996). Emergence of simple-cell receptive field properties by learning a sparse code for natural images. *Nature*, 381, 607–609.
- Helmholtz, H. (1858). Über Integrale der hydrodynamischen Gleichungen, welche den Wirbelbewegungen entsprechen. *Journal für die reine und angewandte Mathematik*, 55, 25–55.

**Background tutorials (online).** The P-rot-6 derivation, the skew generator $B$, the position-dependent gauge field $B(x)$ of §10, and the curvature-tensor open question of §11.3 all live in the mathematical language of Lie groups and gauge theory. For readers who want a self-contained introduction to that language, pitched at graduate level and written with the transformer context in mind, the following two tutorials were consulted during the preparation of this note:

- Gueorguiev, D.P. *Lie Groups for Gauge Theory — A Graduate Tutorial from Matrix Groups to the Structure of U(1), SU(2), SU(3).* [geometric_deep_learning/docs/Lie_Groups_for_Gauge_Theory_Tutorial.md](https://github.com/dimitarpg13/geometric_deep_learning/blob/main/docs/Lie_Groups_for_Gauge_Theory_Tutorial.md). Matrix Lie groups, the Lie algebra as the tangent space at the identity, the Lie bracket and its bilinear / antisymmetric / Jacobi structure, the exponential map, the adjoint representation, and the classification of compact simple Lie groups. Directly relevant background for the skew-symmetric generator $B \in \mathfrak{so}(d)$ of P-rot-6 and for the non-abelian bracket $[B_\mu, B_\nu]$ in the curvature tensor of §11.3.
- Gueorguiev, D.P. *A Full Tutorial on Gauge Theory — From First Principles to Non-Abelian Fields and Transformer Connections.* [geometric_deep_learning/docs/Gauge_Theory_Tutorial.md](https://github.com/dimitarpg13/geometric_deep_learning/blob/main/docs/Gauge_Theory_Tutorial.md). Fiber bundles and principal bundles, connections and covariant derivatives, the curvature / field-strength tensor, non-abelian gauge theories and the Yang–Mills action, holonomy and Wilson loops, and a dedicated §16 *Gauge Theory and Transformers: The Connection* that is the most direct companion reading for the position-dependent gauge field $B(x)$ of §10 and the fibre-bundle interpretation of non-autonomous attention dynamics.

---

## 13. Related documents (added retroactively)

The scope-and-status note at the top already names the documents that situate P-rot-6 in the larger descriptive → empirical → prescriptive arc. They are gathered here for convenience:

- [`The_Failure_of_Conservative_Models_to_explain_hidden_state_trajectories.md`](The_Failure_of_Conservative_Models_to_explain_hidden_state_trajectories.md) — the *empirical* counterpart to this document. §1.5 of that note is the direct test of the P-rot-6 hypothesis on GPT-2 small; §3 gives the structural argument (path dependence, $W_Q^\top K$ non-symmetry, per-head rank-deficient torques) for why the linearisation of §6.2 should not be expected to hold in pretrained attention.
- [`Conservative_by_Construction_Language_Models.md`](Conservative_by_Construction_Language_Models.md) — the *prescriptive* response produced *after* P-rot-6 was rejected: build a language model whose layer dynamics are the Euler–Lagrange flow of a single scalar $V_\theta$, so the $\Omega(x) \ne 0$ obstruction does not arise in the first place.
- [`Training_and_Inference_with_SPLM.md`](Training_and_Inference_with_SPLM.md) — how the resulting Scalar-Potential Language Model is actually trained (nested-autograd force computation) and decoded (constant-in-$T$ autoregressive step), including the SARF-faithful and per-token semantic-mass ablations.
- [`On_Modeling_Semantic_Energy_Field_into_SPLM.md`](On_Modeling_Semantic_Energy_Field_into_SPLM.md) — component-by-component mapping from the Semantic Simulation framework's full energy field onto SPLM's $V_\theta$, $\xi$, and $m_t$.
- Paper v2 §14 and Appendix A report the consolidated empirical result: the SPLM admits a shared potential with $R^2 \approx 0.79$ across depth, versus the $\le 0$ ceiling every attention variant — including the P-rot-6 fits of §1.5 — reaches on the same corpus under the same diagnostic.

---

*Document version: April 2026. Prepared for experimental comparison of dynamical models of transformer hidden state motion in semantic space.*
