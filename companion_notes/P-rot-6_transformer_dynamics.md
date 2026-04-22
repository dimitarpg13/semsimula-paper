# P-rot-6: Velocity-Coupled Gyroscopic Dynamics and Decoder-Only Transformers
### A Theoretical and Empirical Framework for Hidden State Motion in Semantic Space

---

> **Abstract.** We derive the P-rot-6 model — a second-order damped equation of motion augmented with a velocity-coupled skew-symmetric force — from first principles, situate it within the Helmholtz decomposition of hidden state dynamics, and trace its theoretical connection to the key-value asymmetry (K≠V) in scaled dot-product attention. We then develop a concrete, end-to-end methodology for estimating the P-rot-6 B matrix from the weight matrices of decoder-only transformer models (GPT-2, Pythia), including code, diagnostics, and a principled failure-mode analysis. Finally, we discuss what a pass or fail of P-rot-6 implies for the geometry of semantic space and outline the next model class hierarchy.

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

We study the motion of hidden states `h_t ∈ ℝᵈ` across token positions `t = 1, ..., T` at a fixed layer `ℓ` of a decoder-only transformer. The fundamental question is:

> **Does the trajectory `{h_t}` admit a compact dynamical description? If so, what class of differential equation governs it?**

This is not merely a descriptive question. A successful dynamical model of hidden state trajectories would:
- Reveal the geometric structure of semantic space as implicitly defined by the model's weights
- Connect the algebraic properties of attention (K, Q, V matrices) to observable trajectory geometry
- Provide a mechanistic account of phenomena such as deceleration near semantic attractors, trajectory curvature at syntactic boundaries, and layer-wise representational change

### 1.2 Empirical Starting Point

The following results motivate the current investigation:

| Model class | Equation | Result |
|---|---|---|
| Pure scalar potential | `F = -∇V(x)`, any `V` | **Fails** — residual ≈ static null (0.1773) |
| Constant skew position force | `F = -∇V + Ωx`, constant `Ω` | **Fails** |
| Velocity-coupled, constant skew | `F = -∇V + Bẋ`, constant `B = -Bᵀ` | Not yet tested |
| Position-dependent gauge | `F = -∇V + B(x)ẋ` | Not yet tested |
| Riemannian geodesic | `ẍᵏ + Γᵏᵢⱼẋⁱẋʲ = -γẋᵏ` | Not yet tested |

The failure of pure scalar potentials — including the Gaussian well — and position-coupled skew forces points to a **velocity-coupled solenoidal component** as the next candidate. This is the P-rot-6 model.

### 1.3 What "P-rot-6" Denotes

The designation encodes the model's position in a systematic hierarchy of rotation/solenoidal extensions of the base second-order damped oscillator:

- **P**: potential-augmented (includes `-∇V` term)
- **rot**: rotational/solenoidal extension present
- **6**: sixth variant in the hierarchy, corresponding to velocity-coupling with position-independent skew matrix

---

## 2. The Helmholtz Decomposition of Hidden State Forces

### 2.1 The Decomposition Theorem

By the Helmholtz decomposition, any smooth vector field `F : ℝᵈ → ℝᵈ` decomposes uniquely (under appropriate boundary conditions) as:

```
F(x) = -∇φ(x)  +  ∇ × A(x)
        ────────    ────────────
        curl-free   divergence-free
       irrotational   solenoidal
```

In matrix form, the decomposition manifests through the Jacobian:

```
JF(x) = S(x) + Ω(x)

S(x) = (JF + JFᵀ)/2    ← symmetric  → curl-free component
Ω(x) = (JF - JFᵀ)/2    ← antisymm.  → solenoidal component
```

A force field is **conservative** (path-independent work, closed-loop ∮F·dx = 0) if and only if `Ω(x) = 0` everywhere.

### 2.2 Why This Matters for Hidden State Dynamics

The scalar potential models tested in §1.2 model 100% of `F` with the curl-free component. If the true force field has non-zero `Ω(x)`, no scalar potential — regardless of shape, depth, or parameterization — can represent it. This is not a capacity or fitting failure. It is a **structural impossibility**.

### 2.3 The Full Force Taxonomy

For second-order dynamics `mẍ = F(x, ẋ)`, the Helmholtz decomposition generalizes to phase space `(x, ẋ)`:

```
F(x, ẋ) = -∇φ(x)           ← conservative, position only
         + F_sol(x)         ← solenoidal, position only (curl ≠ 0)
         + B(x) · ẋ        ← gyroscopic/magnetic (velocity coupled)
         + D(x) · ẋ        ← symmetric dissipation (drag)
         + nonlinear terms
```

Each row is orthogonal to the others in function space. A scalar potential tests only the first. P-rot-6 adds the third.

---

## 3. The P-rot-6 Model: Derivation and Structure

### 3.1 The Equation of Motion

```
mẍ = -∇V(x) + Bₗẋ - mγẋ

subject to:  Bₗ = -Bₗᵀ ∈ ℝᵏˣᵏ   (skew-symmetric)
```

In component form:

```
m ẍᵢ = -∂V/∂xᵢ  +  ∑ⱼ (Bₗ)ᵢⱼ ẋⱼ  -  mγẋᵢ
```

### 3.2 Why Bₗ Must Be Skew-Symmetric

The velocity coupling `Bẋ` decomposes as:

```
Bẋ = ½(B + Bᵀ)ẋ  +  ½(B - Bᵀ)ẋ
      ──────────       ──────────
      symmetric         skew-symmetric
      → extra drag       → gyroscopic force
      (absorbed into γ)  (does no work)
```

The symmetric part is equivalent to anisotropic damping and is absorbed into the `-mγẋ` term. Only the skew-symmetric part is structurally new. Furthermore:

```
Power delivered by Bẋ:  P = ẋᵀBẋ = 0  ∀ẋ  iff  B = -Bᵀ
```

The skew-symmetric constraint ensures the gyroscopic force **does zero work** — it curves trajectories without changing kinetic energy. This is the magnetic/Lorentz force analogy: a charged particle in a magnetic field is deflected but not accelerated.

### 3.3 The Energy Function

P-rot-6 admits a modified energy:

```
E(x, ẋ) = ½m‖ẋ‖²  +  V(x)
```

Taking the time derivative along trajectories:

```
dE/dt = mẋᵀẍ + ẋᵀ∇V
      = ẋᵀ(-∇V + Bẋ - mγẋ) + ẋᵀ∇V
      = ẋᵀBẋ  -  mγ‖ẋ‖²
      = 0  -  mγ‖ẋ‖²   ← only dissipation remains
      = -mγ‖ẋ‖²
```

**The skew-symmetric B contributes exactly zero to energy dissipation.** The system is a damped, rotating system with well-defined Lyapunov function `E(x, ẋ)`. This is a key sanity check for any fitted B: if energy is not monotonically decreasing (up to noise), B is not purely skew.

### 3.4 The Number of Free Parameters

`B ∈ ℝᵏˣᵏ` skew-symmetric has `k(k-1)/2` free parameters.

| Subspace dimension k | Parameters |
|---|---|
| 10 | 45 |
| 50 | 1225 |
| 100 | 4950 |
| 768 (full BERT/GPT-2) | 295,128 |

For practical fitting, **PCA reduction to k = 20–50** of the velocity principal subspace is strongly recommended before estimating B.

### 3.5 Continuous-Time Formulation

For direct comparison with the ODE literature, P-rot-6 can be written as:

```
ẋ = v
v̇ = -∇V(x)/m  +  (B/m - γI)v
```

or in Hamiltonian form with a non-canonical Poisson bracket. The phase space flow is:

```
d/dt [x]  =  [        v              ]
     [v]     [ -∇V/m + (B/m - γI)v  ]
```

The Jacobian of the right-hand side at a fixed point (x*, 0) is:

```
J* = [     0          I    ]
     [ -H_V(x*)/m   B/m-γI ]
```

where `H_V = ∇²V` is the Hessian of V. Stability requires the real parts of the eigenvalues of `J*` to be negative — dominated by the damping `γ`.

---

## 4. Modern Hopfield Networks and Transformer Attention

### 4.1 The Krotov-Hopfield Hierarchy

**Classical Hopfield (1982):**
```
E = -½ xᵀΞᵀΞx + ½‖x‖²
F = -∇E = Ξᵀ(Ξx) - x   (Hebbian retrieval)
Capacity: ~ 0.14d
```

**Krotov-Hopfield 2016 (polynomial interactions):**
```
E = -∑_μ F_n(ξᵘ·x) + ½‖x‖²,   F_n(z) = zⁿ/n
Capacity: ~ dⁿ⁻¹   (superlinear in d for n > 2)
```

**Ramsauer et al. 2020 (exponential / softmax limit):**

Take `F_n → exp` as `n → ∞`:

```
E(x) = -1/β · log ∑_μ exp(β ξᵘ·x) + ½‖x‖²
     = -1/β · log Z(x) + ½‖x‖²

F(x) = -∇E = Ξᵀ softmax(βΞx) - x
Capacity: ~ exp(d)   (exponential in dimension)
```

The synchronous update `x_new = Ξᵀ softmax(βΞx)` is **exactly** softmax self-attention.

### 4.2 The Transformer Identification

Standard scaled dot-product attention:

```
Attention(Q, K, V) = V · softmax(KᵀQ / √d)
```

Maps to Hopfield retrieval under:

| Transformer | Hopfield |
|---|---|
| Query Q | State being retrieved x |
| Key matrix K | Stored pattern index Ξ |
| Value matrix V | Retrieved pattern content Ξ (if K=V) |
| Scaling 1/√d | Inverse temperature β |
| softmax | Boltzmann retrieval distribution |

**One transformer attention operation = one synchronous Hopfield update step.**

### 4.3 The Autoregressive Setting

In a decoder-only transformer (GPT-2, Pythia), the causal mask enforces:

```
Attention(Q_t, K_{≤t}, V_{≤t}) = V_{≤t} · softmax(K_{≤t}ᵀ Q_t / √d)
```

The "stored patterns" are the keys and values of all preceding tokens. The hidden state at position `t` is being retrieved as a query against a **growing memory bank**. This means the effective Hopfield network grows with sequence length — the attractor landscape changes at every token position.

---

## 5. Why Transformers Are Non-Conservative: The K≠V Theorem

### 5.1 The Conservative Case (K = V)

When `K = V = Ξ` (pure Hopfield, auto-associative):

```
F(x) = Ξᵀ softmax(βΞx) - x
```

Jacobian:

```
JF(x) = β Ξᵀ [diag(s) - ssᵀ] Ξ - I
```

where `s = softmax(βΞx)`. The matrix `[diag(s) - ssᵀ]` is symmetric positive semidefinite (it is the Jacobian of softmax). Therefore `JF(x)` is **symmetric** for all x. Symmetric Jacobian ↔ path-independent work ↔ **conservative**.

**Proof of conservativity:**

The existence of a scalar potential is equivalent to `∂Fᵢ/∂xⱼ = ∂Fⱼ/∂xᵢ` for all `i, j`. This holds exactly when `JF` is symmetric, which is guaranteed when K = V.

### 5.2 The Non-Conservative Case (K ≠ V)

In every real transformer, `K = W_K · context` and `V = W_V · context` with `W_K ≠ W_V`. The force becomes:

```
F(x) = Vᵀ softmax(βKx) - x
```

Jacobian:

```
(JF)ᵢⱼ = β ∑_μ softmax_μ(βKx) · (Kⱼᵘ - [Ks]ⱼ) · Vᵢᵘ  -  δᵢⱼ
```

The **antisymmetric part** (the solenoidal component):

```
Ω(x) = (JF - JFᵀ)/2

      = β/2 · ∑_μ softmax_μ(βKx) · (Vᵘ⊗Kᵘ  -  Kᵘ⊗Vᵘ)
               ──────────────────────────────────────────
               attention-weighted antisymmetric outer products
```

### 5.3 The Non-Conservativity Theorem

**Theorem.** The attention force `F(x) = Vᵀ softmax(βKx) - x` is conservative if and only if `Vᵘ ∥ Kᵘ` for all `μ` (i.e., each key and its associated value are collinear).

**Proof sketch.** `Ω(x) = 0 ∀x` requires each outer product `Vᵘ⊗Kᵘ - Kᵘ⊗Vᵘ = 0`, which holds iff `Vᵘ = λᵘKᵘ` for scalars `λᵘ` — collinearity. Since `Vᵘ = W_V ξᵘ` and `Kᵘ = W_K ξᵘ`, collinearity for all context tokens requires `W_V = ΛW_K` for diagonal `Λ` — a measure-zero set of weight configurations, never achieved in trained models. ∎

### 5.4 The Geometric Interpretation

Each stored pattern `μ` contributes a **vortex** in the plane spanned by `{Kᵘ, Vᵘ}`:

```
Ωᵘ = Vᵘ⊗Kᵘ - Kᵘ⊗Vᵘ  ∈  ℝᵈˣᵈ   (rank-2 antisymmetric matrix)
```

The full solenoidal field is a **softmax-weighted superposition of vortices**, where the weights change with position x (via the attention distribution). This is:

- **Spatially inhomogeneous**: different context tokens dominate at different hidden state positions
- **Dynamically active**: as `x` evolves, the attention weights shift, changing which vortices are active
- **Structurally irreducible**: cannot be represented by any scalar potential

---

## 6. Connecting K≠V Antisymmetry to P-rot-6

### 6.1 The Key Structural Difference

The K≠V solenoidal component is a **position-only force**:

```
F_sol(x) ← depends on x, not ẋ
```

The P-rot-6 gyroscopic term is a **velocity-coupled force**:

```
F_gyro = Bẋ ← depends on ẋ, not x
```

These occupy **distinct sectors** of the Helmholtz decomposition. The identification `Ω(x) ↔ B` is not exact — it requires additional conditions.

### 6.2 When the Identification Is Valid

Consider a trajectory `x(t)` linearized around a reference `x̄(t)`:

```
x(t) = x̄(t) + δx(t)
```

Force on perturbed trajectory:

```
F(x̄ + δx) ≈ F(x̄) + JF(x̄)·δx
            = F(x̄) + S(x̄)·δx + Ω(x̄)·δx
```

For a locally quasi-straight trajectory over a short time window `δt`:

```
δx(t) ≈ ẋ(t) · δt
```

The antisymmetric correction becomes:

```
Ω(x̄)·δx ≈ Ω(x̄) · ẋ · δt
```

This gives an **effective velocity coupling** with:

```
B_eff ≈ Ω(x̄) · δt
```

The identification holds when:

| Condition | Mathematical form |
|---|---|
| Locally straight trajectory | `‖ẍ‖ · δt ≪ ‖ẋ‖` |
| Frozen attention | `‖∂s/∂x‖ · ‖δx‖ ≪ ‖s‖` |
| Short window | `δt ≪ 1/β‖K‖` |

### 6.3 The Phase Space Lifting

A cleaner connection avoids linearization. Lift to phase space `z = (x, v)`:

```
ż = G(z) = [        v             ]
            [ F(x)/m - γv         ]
```

The Jacobian of `G` in phase space:

```
JG(z) = [    0         I    ]
         [ JF(x)/m   -γI    ]
```

The off-diagonal block `JF(x)/m` contains `Ω(x)/m`. In phase space, the solenoidal component of the position-space force **appears as a coupling between the velocity equation and the position coordinate** — structurally analogous to a velocity coupling when projected back.

This projection is exact when the phase space manifold is locally a product. For transformer trajectories, this holds approximately in **slow-varying semantic regions** (plateau phases between transitions).

### 6.4 The Theoretical B Matrix

The P-rot-6 B matrix theoretically predicted by transformer weights is:

```
B_theory(x̄) = β/2 · ∑_μ softmax_μ(β K x̄) · (Vᵘ⊗Kᵘ - Kᵘ⊗Vᵘ)
```

where the sum is over context tokens, and `x̄` is a reference hidden state (e.g., trajectory mean). This has **zero free parameters** — it is entirely determined by:

- The K and V weight matrices `W_K`, `W_V`
- The context token representations at layer `ℓ`
- The reference state `x̄`

This theoretical prediction can be tested directly against the empirically fitted B — the residual difference quantifies the validity of the linearization.

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
        # Vᵘ⊗Kᵘ - Kᵘ⊗Vᵘ : antisymmetric rank-2 matrix
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

Before fitting B empirically, reduce to the subspace where dynamics actually live:

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
    
    r[t] = ẍ[t] - (-∇V(x[t]) + B·ẋ[t] - γẋ[t])
    
    Uses finite differences for ẋ and ẍ.
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
    
    Solves: r[t] ≈ B @ v[t]
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
    
    # Least squares: R ≈ V @ Bᵀ  →  Bᵀ = (VᵀV)⁻¹ VᵀR
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
    B_proj = P @ B_theory @ Pᵀ
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
# 1.0 → perfect alignment (theoretical prediction is correct)
# 0.0 → orthogonal (theory misses the empirical structure entirely)
```

---

## 8. Diagnostics and Validation Protocol

### 8.1 The Spatial Variation Diagnostic

Before committing to a constant B, measure how much `Ω(x)` varies along the trajectory:

```python
def spatial_variation_diagnostic(W_K, W_V, H, beta, d_head=64):
    """
    Compute Ω(x_t) at each trajectory point and measure variation.
    
    High variation ratio → constant B not justified → need B(x).
    Low variation ratio  → constant B is a good approximation.
    """
    Omegas = []
    for x_t in H:
        B_t, _ = compute_hopfield_curl(W_K, W_V, H, x_t, beta, d_head)
        Omegas.append(B_t)
    
    Omega_mean = np.mean(Omegas, axis=0)
    
    deviations = [np.linalg.norm(O - Omega_mean) for O in Omegas]
    variation_ratio = np.std(deviations) / (np.linalg.norm(Omega_mean) + 1e-10)
    
    print(f"Ω spatial variation ratio: {variation_ratio:.4f}")
    print(f"  < 0.05 → constant B is valid")
    print(f"  0.05–0.20 → moderate position-dependence, B(x) recommended")
    print(f"  > 0.20 → strong position-dependence, constant B will fail")
    
    return variation_ratio, Omegas
```

### 8.2 The Energy Monotonicity Check

```python
def check_energy_monotonicity(H, B_fit, gamma, V_fn):
    """
    Verify that dE/dt ≤ 0 along trajectories (required for valid B).
    If dE/dt > 0 consistently, B is not purely skew.
    """
    v = np.diff(H, axis=0)           # (T-1, k)
    x = H[:-1]
    
    V_vals  = np.array([V_fn(xt) for xt in x])
    KE      = 0.5 * np.sum(v**2, axis=1)
    E       = KE + V_vals
    dE_dt   = np.diff(E)
    
    # Theoretical dE/dt = -gamma * ‖v‖²
    dE_theory = -gamma * np.sum(v[:-1]**2, axis=1)
    
    # Check gyroscopic work (should be ~0)
    gyro_work = np.array([v[t] @ B_fit @ v[t] for t in range(len(v)-1)])
    
    print(f"Mean gyroscopic work (should be ≈0): {gyro_work.mean():.4e}")
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
    results["-∇V only"] = res_V
    
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
        
        print(f"Head {h:2d}: ‖B_h‖_F = {np.linalg.norm(B_h):.4f}")
    
    return B_per_head
```

---

## 9. Interpreting Results: Pass, Partial, Fail

### 9.1 Decision Matrix

| Residual outcome | Interpretation | Next action |
|---|---|---|
| B_theory ≈ B_fit, both beat null | **Theory confirmed**: K≠V antisymmetry drives solenoidal dynamics; linearization holds | Measure curvature of B(x); check head decomposition |
| B_fit beats null, B_theory doesn't | **Structure correct, theory approximate**: velocity coupling exists but linearization is inaccurate | Fit full B(x) network; measure spatial variation ratio |
| Both beat null but weakly (< 5%) | **Partial success**: B captures some solenoidal structure | Check if head-specific B gives stronger signal |
| Neither beats null | **Velocity coupling absent**: solenoidal dynamics are position-driven (pure F_sol) | Move to position-dependent B(x); check MLP contribution |
| B_fit beats null, but B + B_theory orthogonal | **Wrong vortex structure**: K≠V is not the source | Examine MLP, LayerNorm, or inter-layer contributions |

### 9.2 What Each Outcome Implies About Semantic Space Geometry

**P-rot-6 succeeds (globally):**
The semantic space has an approximately **uniform magnetic-like field** — every region of meaning-space is subject to the same rotational bias. Semantic trajectories are helical around the K≠V vortex axes. The relevant geometry is a flat space with a constant connection form.

**P-rot-6 succeeds locally (high spatial variation ratio):**
The magnetic field is **inhomogeneous** — the rotational structure depends on position. Near certain semantic attractors, the vortex intensity increases (high softmax concentration → single dominant token's K/V pair dominates `Ω`). This is the **position-dependent gauge field** B(x) scenario.

**P-rot-6 fails entirely:**
The dynamics are not velocity-linear in any sense. This points toward either:
- **Riemannian geodesic structure**: the curvature IS the force, not a correction to Newtonian dynamics
- **Non-Markovian dynamics**: the effective force at time t depends on the full history `h_{<t}`, not just `h_t` and `ḣ_t`
- **Discrete punctuation**: the trajectory is not smooth enough for second-order ODE models to apply

---

## 10. Next Model Class Hierarchy

Having established the hierarchy through P-rot-6, the next steps follow a principled progression:

### 10.1 Position-Dependent Gauge Field B(x)

```
mẍ = -∇V(x) + B(x)ẋ - mγẋ,     B(x) = -B(x)ᵀ
```

**Parameterization options** (in order of complexity):

1. **RBF interpolation**: `B(x) = ∑_c w_c · B_c · φ(‖x - c‖)` — cluster-based, interpretable
2. **Low-rank skew network**: `B(x) = U(x)Σ(x)V(x)ᵀ - V(x)Σ(x)U(x)ᵀ` with neural U, V, Σ
3. **Full gauge network**: unrestricted MLP output projected to skew manifold via `(B - Bᵀ)/2`

The theoretical prediction from transformer weights becomes:

```python
def B_theoretical_at_x(x, W_K, W_V, context, beta, d_head):
    """Position-dependent version: Ω(x) evaluated at current state."""
    K = context @ W_K.T
    V = context @ W_V.T
    s = softmax(beta * K @ x)
    return beta/2 * sum(s[mu] * (outer(V[mu], K[mu]) - outer(K[mu], V[mu]))
                        for mu in range(len(s)))
```

This is the **zero-free-parameter theoretical prediction** for B(x) — fully determined by transformer weights.

### 10.2 Riemannian Geodesic Formulation

If all gauge field models fail, the hypothesis shifts to:

```
ẍᵏ + Γᵏᵢⱼ(x) ẋⁱ ẋʲ = -γẋᵏ
```

where Γᵏᵢⱼ are the Christoffel symbols of an implicit Riemannian metric on semantic space.

The metric can be estimated from the Fisher information of the transformer's output distribution:

```
g_ij(x) = E_y[∂log p(y|x)/∂xᵢ · ∂log p(y|x)/∂xⱼ]
```

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

P-rot-6 is the **minimal model** that can capture velocity-coupled solenoidal dynamics. Its B matrix is:

- **Theoretically motivated** by the K≠V antisymmetry in transformer attention
- **Zero-free-parameter predictable** from W_K, W_V, and the context
- **Empirically fittable** by a single linear regression on velocity residuals
- **Physically interpretable** as a gyroscopic force doing no work, curving trajectories without dissipation

### 11.2 The Theoretical Prediction Chain

```
W_K ≠ W_V  (by design in every transformer)
    ↓
Ω(x) = β/2 ∑_μ softmax_μ(βKx)(Vᵘ⊗Kᵘ - Kᵘ⊗Vᵘ) ≠ 0
    ↓
Force field is non-conservative (curl ≠ 0)
    ↓
Scalar potentials structurally fail (confirmed empirically)
    ↓
Linearized approximation: Ω(x̄)ẋ ≈ Bẋ  (valid for smooth trajectories)
    ↓
P-rot-6 with B_theory = Ω(x̄) is a zero-free-parameter mechanistic model
    ↓
Test: does residual drop significantly when B_theory is added?
```

### 11.3 Open Questions

1. **Multi-head superposition**: Does each head contribute an independent vortex, or do heads interfere constructively/destructively? The per-head analysis in §8.4 addresses this, but the algebraic structure of head superposition in the full B matrix is not yet characterized.

2. **Layer depth dependence**: Does `‖Ω(x)‖` grow or decay with layer depth? If it grows, solenoidal dynamics become more dominant at later layers — consistent with the view that deep layers perform semantic integration (curved trajectories) while early layers perform syntactic processing (more linear).

3. **Curvature tensor of the gauge field**: If B(x) is position-dependent, its curvature tensor `F_μν = ∂_μBν - ∂_νBμ + [Bμ, Bν]` determines whether there are topological obstructions to gauge-away the solenoidal component. Non-zero curvature would constitute evidence for genuine topological structure in semantic space.

4. **Connection to induction heads**: Mechanistic interpretability identifies induction heads as key circuits for in-context learning. These heads have strongly structured K/V relationships. Their contribution to `Ω(x)` is likely dominant — testing P-rot-6 on induction-head-ablated models would isolate the induction head's geometric contribution.

5. **Temperature scaling**: As `β → ∞` (sharp attention), `Ω(x)` becomes dominated by the single highest-attention token, and its vortex becomes the only active one. As `β → 0`, contributions average out and `Ω → 0`. This predicts a phase transition in solenoidal dynamics as a function of attention temperature.

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

---

*Document version: April 2026. Prepared for experimental comparison of dynamical models of transformer hidden state motion in semantic space.*
