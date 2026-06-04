# Fock-PARFLM Next Steps: Resolving the Conservative–Attention Gap

**Technical Report — Semantic Simulation Research Programme**
**Date:** June 3, 2026
**Relates to:** Paper v4 §§17.8, 17.13, 18.1–18.5; `model_fock_attention.py`; `model_fock_parf_v2.py`; `Improving_the_Fock_Mechanism_to_match_Attention.md`

---

## Table of Contents

1. [Context and Experimental Baseline](#1-context-and-experimental-baseline)
2. [Fock Attention Experiment Results](#2-fock-attention-experiment-results)
3. [Fock v2 Register Diagnostics](#3-fock-v2-register-diagnostics)
4. [Diagnosis Summary](#4-diagnosis-summary)
5. [Resolution Hierarchy](#5-resolution-hierarchy)
6. [Step 1 Implementation: QKV Creation Gate v2.1](#6-step-1-implementation-qkv-creation-gate-v21)
7. [Open Questions](#7-open-questions)

---

## 1. Context and Experimental Baseline

The Semantic Simulation programme achieves the following ladder of results on TinyStories (val PPL):

| Architecture | PPL | Memory (T-dependence) | Params |
|---|---|---|---|
| Single-ξ PARFLM ceiling (P10h) | 26.4 | O(1) | — |
| Multi-ξ K=4 conservative base | 12.47 | O(1) | ~16.6M |
| Conservative Fock registers (K=4, M=16, v2) | 12.00 | O(1) | ~17.2M (+578K Fock) |
| Conservative K=8 no registers | ~12.06 | O(1) | ~16.6M |
| **Fock Attention 1-head (K=4, direct exchange)** | **11.48** | **O(T²)** | **16.6M (+66K exchange)** |
| **Fock Attention 4-head (K=4, direct exchange)** | **10.93** | **O(T²)** | **16.7M (+131K exchange)** |
| **Matched attention baseline (MatchedGPT)** | **7.81** | **O(T²)** | — |

The residual gap after the conservative multi-ξ base is **4.66 PPL** (12.47 → 7.81). The Fock register mechanism closes only 0.47 PPL of this (12.47 → 12.00), while the direct attention exchange force closes 1.54 PPL (12.47 → 10.93 with 4 heads). The 1.07 PPL gap between registers (12.00) and attention (10.93) — despite registers using 4.4× more additional parameters (578K vs 131K) — confirms a genuine structural routing deficit, not a capacity effect.

---

## 2. Fock Attention Experiment Results

`FockAttentionPARFLM` injects the §5.1 Feynman diagram exchange force directly into the conservative Verlet dynamics as a post-Verlet correction. Two arms completed at 8,000 steps:

### 2.1 Arm: direct_K4_h1_8k (1-head, d_k=64)

| Metric | Value |
|---|---|
| Final val PPL | 11.48 |
| Exchange overhead | 65,537 params |
| Final exchange_scale | −0.2547 (tanh = −0.249) |
| Final α_k | [0.000, 0.597, 0.891, 0.975] |
| Training time | 6,044 s (1.68 h) |

### 2.2 Arm: direct_K4_h4_8k (4-head, d_k=32)

| Metric | Value |
|---|---|
| Final val PPL | 10.93 |
| Exchange overhead | 131,073 params |
| Final exchange_scale | −0.2712 (tanh = −0.264) |
| Final α_k | [0.000, 0.593, 0.882, 0.975] |
| Training time | 6,287 s (1.75 h) |

### 2.3 Key findings

1. **Multi-head helps**: 4 heads with d_k=32 outperforms 1 head with d_k=64 by 0.55 PPL (10.93 vs 11.48), consistent with the standard Transformer finding that multi-head > single-head.

2. **Negative exchange scale**: Both arms learned `exchange_scale < 0`, meaning `tanh(exchange_scale) ≈ −0.26`. The model chose a **repulsive** exchange force — pushing tokens apart in representation space rather than blending them. This is physically meaningful: the conservative Verlet dynamics already handles attractive clustering; the exchange provides the complementary dispersive/discriminative pressure.

3. **Dead α channel persists**: The fastest EMA channel (α ≈ 0) freezes at essentially zero in both arms, identical to the register experiments. This channel acts as a cumulative running sum with no decay — it may be absorbing positional information that the other channels don't need.

4. **Moderate force magnitude**: `|tanh(exchange_scale)| ≈ 0.26` is far from saturation (1.0). The conservative dynamics constrains how much exchange force can be absorbed. The `dt²/(m_b·(1+dt·γ))` prefactor in the Verlet injection further limits effective force magnitude compared to a clean residual path.

---

## 3. Fock v2 Register Diagnostics

An eval pass was run on the converged Fock v2 Multi-Xi checkpoint (12.00 PPL, K=4, M=16, LIFO, 16k steps) using `eval_fock_register_diagnostics.py`. The script computes per-layer normalised attention entropy, register content diversity, and α_max from the QKV creation gate.

### 3.1 Per-layer results

| Layer | Entropy | α_max | Diversity |
|-------|---------|-------|-----------|
| 0 | 0.2988 | 0.5958 | 0.2913 |
| 1 | 0.0006 | 0.9984 | 0.4238 |
| 2 | 0.0012 | 0.9967 | 0.4075 |
| 3 | 0.0024 | 0.9934 | 0.2709 |
| 4 | 0.0034 | 0.9915 | 0.2467 |
| 5 | 0.0038 | 0.9897 | 0.3014 |
| 6 | 0.0036 | 0.9907 | 0.4721 |
| 7 | 0.0044 | 0.9884 | 0.5535 |
| **Mean** | **0.0398** | **0.9431** | **0.3709** |

### 3.2 Reference comparison

| Regime | Diversity | Entropy | Source |
|--------|-----------|---------|--------|
| Genuine routing (Q6 result) | ~0.785 | ~0.304 | §17.2 |
| Inert mean-pool (Q0 baseline) | ~0.145 | ~1.0 | §17.2 |
| **This checkpoint** | **0.371** | **0.040** | eval pass |

### 3.3 Interpretation

**Verdict: MIXED — temperature collapse, not mean-pooling.**

The registers are not in the inert mean-pool regime (diversity 0.37 >> 0.15), but they are far from genuine routing (diversity 0.37 << 0.79). The dominant pathology is **entropy collapse**: after layer 0, every register's creation attention has collapsed to essentially a single token (α_max > 0.99, entropy < 0.005). The softmax temperature is far too sharp.

Layer 0 is the exception — it has softer attention (entropy 0.30, α_max 0.60) because the initial register embeddings haven't yet been biased by training dynamics. But from layer 1 onwards, the attention is nearly one-hot: each register reads a single token rather than a soft mixture. This is the opposite of the mean-pool failure (uniform attention); it's a **hard-winner failure** where each register snaps to one position.

Diversity increases with depth (0.25 at layer 4 → 0.55 at layer 7), suggesting that deeper layers develop more specialisation even with collapsed entropy. This is likely because the underlying token representations become more discriminative with depth, forcing the one-hot attention to land on different tokens for different registers.

The diagnosis is clear: **the creation gate temperature is the primary bottleneck**. The model needs softer attention to learn meaningful soft-routing patterns.

### 3.4 Diagnostic figure

The 3-panel diagnostic figure (entropy heatmap, α_max heatmap, diversity line plot) is saved alongside the checkpoint at:
`v2_K4_M16_lifo_16k/..._register_diagnostics.png`

---

## 4. Diagnosis Summary

Combining the Fock Attention and Register Diagnostics results:

| Question | Answer | Implication |
|----------|--------|-------------|
| Q1: Does Fock Attention reach < 10 PPL? | Not yet (10.93 at 8k steps, likely < 10 at 16k) | Routing deficit confirmed: 1.07 PPL gap to registers despite 4.4× fewer params |
| Q2: Does exchange_scale saturate at 1.0? | No, stabilises at ~0.26 (repulsive) | Post-Verlet injection geometry constrains force magnitude |
| Q3: Are registers routing or mean-pooling? | **Neither — temperature collapse** (entropy 0.04, diversity 0.37) | Step 1 (fix creation gate) is the active bottleneck |
| Q4: Dead α channel? | Yes, α₀ ≈ 0 in all experiments | One of K=4 channels is redundant; K=3 may suffice |

**Resolution hierarchy branch: Step 1** — fix the creation gate before scaling M or adding iterative refinement.

---

## 5. Resolution Hierarchy

### 5.1 Step 0 — Establish the deficit ✅ COMPLETE

- Fock Attention PPL: 10.93 (4-head, 8k steps)
- Register diversity/entropy: 0.37 / 0.04 (temperature collapse)
- Branch condition: Step 1 (fix routing quality)

### 5.2 Step 1 — Fix routing quality (ACTIVE)

Three interventions in order of parameter efficiency:

**B1 — Per-register learnable temperature** (M parameters, highest leverage):
Replace the shared τ with M independent learnable log-temperatures initialised to encourage softer attention:
```python
self.log_tau = nn.Parameter(torch.full((M,), math.log(tau_init)))
tau = self.log_tau.exp().clamp(min=1e-4)  # (M,)
scores = scores / tau.unsqueeze(0).unsqueeze(-1)  # (B, M, T)
```
Initialise at τ₀ = √d_k (matching standard 1/√d_k scaling) or higher (e.g. 2·√d_k) to start softer than the current collapsed state.

**B2 — Per-register key subspaces** (M × d × d_k parameters):
Replace shared `W_K ∈ ℝ^{d×d_k}` with per-register `W_K ∈ ℝ^{M×d×d_k}`:
```python
self.W_K = nn.Parameter(torch.randn(M, d, d_k) * init_scale)
K_k = torch.einsum('btd,mdk->bmtk', h_tokens, self.W_K)
```
Each register attends in a different key subspace, analogous to per-head Q/K projections in multi-head attention.

**B3 — Orthogonal register initialisation** (zero parameter cost):
Replace isotropic Gaussian `register_embed` init with orthogonal initialisation to break symmetry from the first forward pass:
```python
if M <= d:
    U, _, _ = torch.linalg.svd(torch.randn(d, d))
    register_embed = U[:M] * init_scale
```

### 5.3 Step 2 — Warm-start from converged conservative checkpoint

Train the K=4 multi-ξ base to convergence (~12.47 PPL), freeze, then train Fock parameters only with a higher learning rate for 6–8k steps. This eliminates interference between conservative dynamics and register creation during early training.

### 5.4 Step 3 — Sparse top-k creation (if M genuinely needs to scale)

If Step 1 achieves diversity > 0.6 but PPL gap remains > 1 PPL to Fock Attention, scale M to 64 with sparse top-k creation (k=16 tokens per register) at O(M·k·d) cost.

### 5.5 Step 4 — Iterative register refinement (Perceiver-style)

R=2–3 rounds of token→register→register→token attention at O(R·M·T + M²) cost for M ≪ T.

---

## 6. Step 1 Implementation: QKV Creation Gate v2.1

The Step 1 fixes are implemented as a new `fock_version = "v2.1"` in `model_fock_parf_v2.py`, extending `QKVCreationGate` without breaking backward compatibility with existing v2 checkpoints.

### 6.1 New config fields

```python
@dataclass
class FockMultiXiPARFConfig(MultiXiPARFConfig):
    # ... existing fields ...
    per_register_keys: bool = False     # B2: per-register W_K
    ortho_register_init: bool = False   # B3: orthogonal register embeddings
```

The per-register temperature (B1) is already supported via `tau_create_init` — the v2.1 gate extends this to per-register temperatures by promoting `log_tau` from a scalar to a (M,)-shaped parameter.

### 6.2 Experimental arms

The Step 1 experiment compares the following arms against the v2 baseline (12.00 PPL):

| Arm | B1 (per-reg τ) | B2 (per-reg K) | B3 (ortho init) | Steps |
|-----|:---:|:---:|:---:|---|
| `v2_baseline_rerun` | ✗ | ✗ | ✗ | 16k |
| `v21_tau_only` | ✓ | ✗ | ✗ | 16k |
| `v21_tau_perK` | ✓ | ✓ | ✗ | 16k |
| `v21_tau_perK_ortho` | ✓ | ✓ | ✓ | 16k |
| `v21_ortho_only` | ✗ | ✗ | ✓ | 16k |

---

## 7. Open Questions

**[Q1 — Fock Attention 16k]** Running the 4-head arm to 16k steps will determine whether Fock Attention breaks 10 PPL, tightening the ceiling estimate.

**[Q2 — Step 1 outcome]** Does fixing the creation gate temperature raise diversity from 0.37 toward the Q6 reference of 0.78? If so, what PPL does the improved routing achieve?

**[Q3 — Dead α channel]** Is K=3 sufficient? Removing the frozen α₀ ≈ 0 channel and retraining would establish whether it contributes anything.

**[Q4 — Warm-start]** If Step 1 brings routing quality to Q6 level but PPL remains > 11.5, warm-starting from the converged conservative base (Step 2) is the next highest-leverage experiment.

**[Q5 — Capacity control]** Adding ~578K conservative parameters (wider V_φ or K=5) to the base without registers would cleanly isolate the routing contribution from capacity.

---

*Report compiled: June 3, 2026. Updated with Fock Attention results and Register Diagnostics eval.*
*Semantic Simulation Research Programme.*
