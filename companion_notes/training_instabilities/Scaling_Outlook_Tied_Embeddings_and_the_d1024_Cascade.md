# Scaling Outlook, Tied Embeddings, and the d = 1024 Second-Order Cascade

**What this note is.** §21–§23 of the training-instabilities record: the scaling outlook and hardening recommendations, tied against untied embeddings as a distinct instability mode, and the second-order gradient cascade that made every d = 1024 run unstable. It was split on 2026-11-21 from [`Training_Instabilities_in_Fock-PARFLM_with_structured_V_theta.md`](https://github.com/dimitarpg13/semsimula-paper/blob/5551315/companion_notes/Training_Instabilities_in_Fock-PARFLM_with_structured_V_theta.md) (written June to August 2026). The text is that document's, unchanged except for its links, and keeps its section numbers, so a reference to "§N" anywhere in the record or in other notes finds its note through [the index](README.md).

> **Causal-leak caveat.** Every experiment in §1–§23 used the pre-fix, leaky Fock architecture: the reverse channel read a shared full-window register state, before the `prefix_causal_registers` fix. The stability findings stand; every perplexity reported here is leak-contaminated (optimistically low). The full caveat is in [the index](README.md#causal-leak).

## Contents

- [21. Scaling Outlook and Hardening Recommendations](#21-scaling-outlook-and-hardening-recommendations)
- [22. Tied vs. Untied Embeddings: A Distinct Instability Mode](#22-tied-vs-untied-embeddings-a-distinct-instability-mode)
- [23. d=1024 Universal Instability: The Second-Order Gradient Cascade](#23-d1024-universal-instability-the-second-order-gradient-cascade)

---

## 21. Scaling Outlook and Hardening Recommendations

### 21.1 Current status

The per-group clipping in Fock-PARFLM v2.1 handles embedding spikes
gracefully at the current scale ($d = 384$, $V = 50{,}257$).  No manual
intervention has been needed during the `e5c_plgate` run despite spikes
with pre-clip norms exceeding $10^5$.

### 21.2 Anticipated changes at $d = 768$

Scaling to $d = 768$ will:

1. **Increase baseline gradient norms** by $\sim 2\times$ (since
   $\lVert h \rVert\_2 \propto \sqrt{d}$).
2. **Increase spike amplitude** proportionally. Pre-clip spikes of
   $\sim 160\text{k}$ are expected.
3. **Not change the spike frequency** — this is controlled by batch
   composition and vocabulary statistics, not model width.

Per-group clipping should remain effective without threshold changes,
since each group's baseline norm also scales with $\sqrt{d}$.

### 21.3 Optional hardening for large-scale runs

For runs at $d \geq 768$ with more than 9B tokens, consider layering:

1. **z-loss regularisation** ($\lambda\_z = 10^{-4}$) to suppress logit
   growth proactively.
2. **Spike-aware logging** that records the top-$k$ token IDs
   contributing to each spike, enabling post-hoc vocabulary analysis.
3. **Adaptive per-group thresholds** that track the exponential moving
   average of each group's gradient norm and clip at a fixed multiple
   (e.g. $5\times$ the EMA).

### 21.4 Relationship to earlier $V\_\theta$ blowups

The embedding spikes documented in §§18–20 are **mechanistically
distinct** from the $V\_\theta$-mediated blowups in §§1–17:

| Property | V\_theta blowups (§§1–17) | Embedding spikes (§§18–20) |
|---|---|---|
| Root cause | Unbounded potential, penalty dominance, precision explosion | Cross-entropy through softmax on rare tokens |
| Primary parameters affected | V\_theta centres, precisions, amplitudes | Projection P, embedding E |
| Frequency | Deterministic (triggered by architecture / LR schedule) | Stochastic (triggered by rare tokens in batch) |
| Fix | Bounded potentials, watchdog, SARF anchors | Per-group gradient clipping |
| Severity | Can cause permanent divergence | Transient; contained by clipping |

Both types of instability can co-occur, but the embedding spikes are
far more common and far less dangerous — they are a nuisance rather
than a catastrophe.

### 21.5 Summary

Gradient spikes in Fock-PARFLM are **embedding spikes** — a universal
phenomenon in autoregressive language models driven by the
cross-entropy loss through softmax over a large vocabulary.  They are
not caused by the structured potential $V\_\theta$, the Fock mechanism,
or the conservative dynamics, although these introduce a secondary
amplification channel that contributes less than 5% of spike energy.

## 22. Tied vs. Untied Embeddings: A Distinct Instability Mode

Separately from the gradient-spike phenomenology of §§18–21, tied
input/output embeddings (`tie_embeddings=True`) were found to cause a
**structural** — not merely transient — instability in Fock-PARFLM,
distinct from all the blowup modes catalogued above. This section
summarises the finding and states the resulting policy for scale-up
runs.

### 22.1 The failure mode: long-tail cross-entropy collapse

Diagnostic D0.4 (per-token-frequency-quintile loss stratification, run
on the Xi=5 baseline at step 78k; full derivation in
[Xi_Bottleneck_Diagnosis_Phase5.md](../Xi_Bottleneck_Diagnosis_Phase5.md)
§8.4–§9.1) found that with **tied** embeddings the rarest-token quintile
(Q0) received a mean cross-entropy of **13.16 nats**, *worse* than the
uniform-distribution baseline of $\ln V \approx 10.83$ nats. In other
words, for the rarest ~20% of the vocabulary, the tied read-out head was
making predictions actively worse than chance.

The root cause (§9.1 of the referenced note) is that the tied head
forces the same embedding vector $e_v$ to serve two conflicting roles:
(1) an *input* representation that must be well-conditioned for the
dynamics to operate on, and (2) an *output* direction that must separate
$v$ from all other tokens in logit space. For frequent tokens these two
objectives are compatible (the input representation is trained often
enough to satisfy both). For rare tokens, gradient signal is too sparse
to jointly satisfy both roles, and the read-out direction degrades below
the quality of a uniform prior.

### 22.2 Fixes applied

Two fixes were adopted together (§9.2 of the referenced note):

1. **Output bias initialised to log-unigram frequency**
   (`use_output_bias=True`, `USE_OUTPUT_BIAS = True` in the notebooks) —
   gives every token a frequency-informed floor on its logit before any
   dynamics-derived signal is added, largely fixing the *frequent*-token
   context-mixing gap.
2. **Untied embeddings** (`tie_embeddings=False`) — decouples the input
   representation from the output read-out direction entirely, removing
   the structural conflict that caused the Q0 collapse. This is the fix
   that directly addresses the long-tail pathology.

### 22.3 Policy for scale-up runs (`d=768`, `d=1024`)

All scale-up presets (`d768`, `d1024` in `train_fock.py`) use
`tie_embeddings=False` and `use_output_bias=True` as **non-negotiable
defaults** for Fock-PARFLM, carried forward from the `d=384` findings
above. The matched GPT-2 baselines (`gpt2-small`, `gpt2-medium`) are
unaffected by this policy since `MatchedGPT` always ties embeddings by
construction (canonical GPT-2 design) and was never observed to exhibit
the D0.4 pathology — the failure mode above appears specific to how
Fock-PARFLM's dynamics reshape the embedding space, not a generic
consequence of weight tying.

The parameter cost of untying (one extra `d x vocab_size` matrix per
tier) is quantified exactly in
[Fock-PARFLM_Scale-Up_Comparative_Experiments.md](../Fock-PARFLM_Scale-Up_Comparative_Experiments.md) —
it accounts for the majority of Fock-PARFLM's total-parameter excess over
parameter-matched GPT-2 baselines at `d=384` and `d=768`.

### 22.4 Relationship to §§1–21

| Property | V\_theta blowups / embedding spikes (§§1–21) | Tied-embedding long-tail collapse (§22) |
|---|---|---|
| Manifestation | Transient gradient-norm spikes or divergence | Persistent elevated loss on a token subset |
| Detectability | Visible in per-step gradient/loss logs | Requires frequency-stratified evaluation (D0.4) |
| Fix | Clipping, bounded potentials, watchdog | Architectural (untie embeddings) + output bias |
| Cost of fix | None (no extra parameters) | +1 embedding-sized matrix per model |
| Scale sensitivity | Grows with $d$, mitigated by per-group clipping | Present at all scales tested; policy is scale-invariant |

The per-group clipping strategy used in Fock-PARFLM is **strictly
superior** to the global norm clipping used in classic transformers
(GPT-2/3, OPT, PaLM, LLaMA):

- It **isolates** the spike to the offending parameter groups ($E$, $P$).
- It **preserves** the gradient signal for all other groups.
- It **protects** the optimizer state from contamination.
- It enables **single-step recovery** with zero manual intervention.

This advantage becomes increasingly important at scale, where the cost
of wasted steps and manual restarts grows linearly with compute budget.

---

## 23. d=1024 Universal Instability: The Second-Order Gradient Cascade

### 23.1 Phenomenology

The gamma sweep at $d=1024$, $L=24$ (split across two LambdaLabs 2×H100 instances, 8 gamma candidates in $[0.05, 0.50]$) revealed a **qualitatively new instability regime** not observed at $d=384$ ($L=16$) or $d=768$ ($L=12$).

**Every** gamma candidate exhibited catastrophic gradient spikes and required watchdog reloads:

| $\gamma$ | Best PPL | Watchdog reloads | Worst instant grad | Primary spike groups |
|:---------:|:--------:|:----------------:|-------------------:|----------------------|
| 0.050 | 342.00 | 3 | 180.53 | `creation_gate`=110, `register`=25 |
| **0.100** | **327.33** | 1 | 651.50 | `P`=417, `E`=98, `creation_gate`=26 |
| 0.250 | 337.47 | 2 | 7,870.94 | `P`=5,023, `creation_gate`=18 |
| 0.300 | 376.19 | 1+ | 536.27 | `creation_gate`=311, `E`=47 |

This is qualitatively different from $d=768$ ($L=12$), where all 8 candidates ran clean with gradient norms under 1.0 and zero watchdog triggers:

| | $d=768$ ($L=12$) | $d=1024$ ($L=24$) |
|---|---|---|
| Max grad\_norm across all candidates | ~0.6 | 7,870 |
| Watchdog reloads (total) | 0 | 7+ |
| Candidates with spikes > 100 | 0 / 7 | 4 / 4 |
| Top spike group | `reverse_channel_scale` (mild) | `P`, `E`, `creation_gate` (catastrophic) |

The spike sources at $d=1024$ are **systemic** — they come from multiple parameter groups across all gamma values:

- **`P` (positional embedding):** The worst offender overall (norm up to 5,023 at $\gamma=0.25$). This component had no dedicated per-group clip in the sweep preset (remedied in §23.3).
- **`E` (input embedding):** Sustained elevated norms (40–98), especially at $\gamma=0.10$ and $\gamma=0.30$. Also lacked a dedicated per-group clip.
- **`creation_gate`:** Occasional catastrophic spikes (norm 110–311), seen at every gamma tested.
- **`reverse_channel_scale`:** Relatively mild at $d=1024$ (norm up to 19.3), in contrast to $d=768$ where it was the dominant (but non-catastrophic) group.
- **`register`:** Elevated at $\gamma=0.05$ (norm up to 24.5).

> **⚠ Causal-leak note.** The observation that `reverse_channel_scale` was *the
> dominant spike group at d=768* is, in hindsight, a fingerprint of the causal
> leak. The audit shows gradient descent persistently drives this scalar open
> because the leak (future→past information flow) lowers the training loss; the
> large sustained gradient recorded here is that pressure made visible. The best
> PPLs in the table above were produced with the leak active and are pending
> re-certification (see the top-of-document banner).

**The sweep results are unreliable** as a guide to optimal gamma. The fact that $\gamma=0.05$ scored worse than $\gamma=0.10$ is confounded by watchdog reload counts: $\gamma=0.05$ lost ~1,500 training steps to 3 reloads, while $\gamma=0.10$ lost ~500 steps to 1 reload. The relative ranking reflects which candidate was disrupted least, not which damping coefficient is optimal.

### 23.2 Root Cause: Exponential Amplification in the Force Cascade

The instability traces to a single line in `model_parf_multixi.py`'s `_layer_step`:

```python
grad_U, = torch.autograd.grad(
    U.float(), h_in,
    create_graph=self.training,
    retain_graph=True,
)
```

The `create_graph=True` flag is necessary so that the backpropagation through the training loss can differentiate through the force computation — the force $f_\ell = -\nabla_h U$ is part of the forward pass, and the loss must propagate gradients through it to update $V\_\theta$ and $V\_\phi$ parameters.

**The consequence:** at each layer $\ell$, the computational graph records not just the force value but the **entire Jacobian** $\partial f\_\ell / \partial h\_\ell$, which is the Hessian $\nabla^2\_h U$ of the potential. When backward() runs, it must differentiate through a chain of these Hessians — one per layer — creating a **compound second-order derivative chain** $L$ layers deep.

The per-layer Jacobian of the Verlet update with respect to the hidden state is:

$$J\_\ell = \frac{\partial h\_{\ell+1}}{\partial h\_\ell} = I + \frac{I}{1 + \Delta t \gamma} + \frac{\Delta t^2}{m(1 + \Delta t \gamma)} \nabla^2\_h U(h\_\ell)$$

The total backward Jacobian over the layer stack is the product $\prod\_{\ell=1}^{L} J\_\ell$. When $L=12$ ($d=768$), this product stays well-conditioned — gradient norms remain O(1). When $L=24$ ($d=1024$), the product exhibits **exponential amplification**: the spectral radius of each $J\_\ell$ slightly exceeds 1 due to the Hessian contribution, and the product of 24 such matrices produces gradient norms of O($10^4$).

This is not a linear scaling with $L$ — the observed ratio of gradient magnitudes at $L=24$ vs $L=12$ is $\sim 10^4$, far exceeding the $2\times$ that simple linear scaling would predict. The exponential character is the signature of a product of near-unit-spectral-radius matrices.

**Key structural point:** this instability is specific to the `create_graph=True` force computation in the Verlet-style integrator. Standard transformers use only first-order gradients (no `autograd.grad` in the forward pass), so their gradient chain is the ordinary backpropagation chain, which is well-understood and manageable with LayerNorm + gradient clipping. Fock-PARFLM's conservative dynamics introduce a fundamentally different gradient topology — a **second-order** chain — whose amplification properties scale differently with depth.

### 23.3 Mitigation Tier 1: Config-Only (No Architecture Change)

These are immediately testable with existing code (as of July 14, 2026, `train_fock.py` supports all of them via CLI arguments):

**1. Per-group gradient clipping for `P` and `E`**

`P` (positional embedding) and `E` (input embedding) were the two worst offenders at $d=1024$ but previously had **no dedicated per-group clip** — they fell through to the default `grad_clip`. Since July 14, 2026, `GRAD_CLIP_OVERRIDES` in `train_fock.py` includes exact-match entries for `P` and `E` at threshold 0.3:

```python
GRAD_CLIP_OVERRIDES = {
    "=P": 0.3,           # positional embedding (exact match)
    "=E": 0.3,           # input embedding (exact match)
    "V_phi": 0.3,
    "creation_gate": 0.3,
    "destruction_gate": 0.3,
    "reverse_channel_scale": 0.1,
    "reverse_ch": 0.1,
    "register": 0.3,
    "depth_code": 0.5,
}
```

The `=` prefix uses exact top-level name matching (via `_assign_clip_group`) rather than substring matching, since single-letter keys like `"P"` would otherwise match `"V_phi"`, `"depth_code"`, etc.

**2. Force clamping (`force_clamp_max`)**

The model already supports direct force clamping via `cfg.force_clamp_max`. When set, the conservative force is clamped element-wise after computation:

```python
if cfg.force_clamp_max is not None:
    f = f.clamp(-cfg.force_clamp_max, cfg.force_clamp_max)
```

This limits the force magnitude **at the source**, before it cascades through subsequent layers' `create_graph=True` chains. Unlike gradient clipping (which acts on the parameter gradients at the optimizer step), force clamping acts on the **dynamics** — it prevents any single layer from injecting an excessively large displacement into the hidden state.

Since July 14, 2026, `force_clamp_max` is exposed as a CLI-overridable field in `TrainConfig` (default 0.0 = disabled, positive value = active). Recommended starting value for $d=1024$: `force_clamp_max=5.0`.

**3. Tighter global `grad_clip`**

The sweep used `grad_clip=1.0` (default). The full `d1024` training preset already uses `grad_clip=0.5`. For the Tier 1 diagnostic, `grad_clip=0.3` is recommended to maximally attenuate the cascade.

**Combined Tier 1 command:**

```bash
python3 train_fock.py --preset sweep-d1024 --gamma_sweep --sweep_gammas 0.10 \
    --sweep_steps 3000 --grad_clip 0.3 --force_clamp_max 5.0 \
    --output_dir ~/runs/d1024_tier1_test --data_dir ~/data
```

If Tier 1 produces a clean 3K-step run (grad norms < 2.0, zero watchdog triggers), the instability is a clipping/clamping problem and $L=24$ is viable.

### 23.4 Mitigation Tier 2: Moderate Architectural Changes

These require changing the model configuration but not the model code:

**4. Reduce $L$ from 24 to 16 or 18**

This is the most direct intervention — it literally halves the cascade depth. $L=16$ matches $d=384$'s depth and would give a clean cross-scale comparison. $L=18$ is a compromise that retains more expressivity while still significantly shortening the cascade.

The downside: this breaks the GPT-2 Medium $L$-matching. However, that matching was always approximate (Fock ~209M params vs GPT-2 ~355M params at the same $d$ and $L$), so depth-matching is less important than stability. The parameter count reduction from $L=24$ to $L=16$ is modest (most parameters are in the embeddings and $V\_\theta$ bank, not in the per-layer components).

**5. Reduce integration step size $\Delta t$**

Currently $\Delta t = 1.0$. The force contribution to the Verlet update scales as $\Delta t^2 / m$:

$$h\_{\ell+1} = h\_\ell + \frac{\delta\_\ell}{1 + \Delta t\,\gamma} + \frac{\Delta t^2}{m\_b(1 + \Delta t\,\gamma)}\,f\_\ell$$

Halving $\Delta t$ to $0.5$ quarters the per-layer force impact ($\Delta t^2 = 0.25$ vs $1.0$). This is the classical ODE-integrator response to stiffness: smaller steps for stability. The trade-off: the model needs more layers to cover the same "distance" in hidden-state space, partially defeating the purpose. Best used in combination with reducing $L$ (e.g., $L=18$, $\Delta t=0.5$).

**6. Increase effective mass**

Currently `mass_mode=logfreq` with mean $m \approx 1.4$. The force-to-acceleration coupling is $\Delta t^2 / m$, so doubling mass halves the force impact. A global mass of 3.0 or 4.0 (via a config override) would provide heavier "inertia" for each token, making the trajectory less sensitive to large forces. This is a less invasive alternative to reducing $\Delta t$.

### 23.5 Mitigation Tier 3: Structural Refactors

These require changes to the model code or the integration architecture:

**7. Detach boundaries every $K$ layers**

Instead of one continuous $L$-layer `create_graph=True` chain, break it into segments. At every $K$-th layer, insert `h = h.detach().requires_grad_(True)` to restart the computation graph. With $K=8$ and $L=24$, this creates 3 segments of 8 layers each. Each segment's force computation has a cascade depth of at most 8 — the same as the stable $d=384$ model at $L=8$ (SPLM).

The cost: force fields in layers 9–16 cannot backprop through the force fields of layers 1–8, limiting cross-depth coordination of the conservative force. The model can still learn forces that coherently shape the trajectory, but the gradient signal for early-layer $V\_\theta$ parameters is truncated. This is analogous to the truncated BPTT used in RNNs and may be acceptable given that the depth-scaling analysis (§5 of [Fock-PARFLM\_Scale-Up\_Gamma\_Sweep\_Results\_and\_Damping\_Regime\_Analysis.md](../Fock-PARFLM_Scale-Up_Gamma_Sweep_Results_and_Damping_Regime_Analysis.md)) shows that information transport is primarily via momentum, not via the gradient chain.

**8. BAOAB + CfC propagator (see §24, in [CfC_BAOAB_Propagator_and_the_Cascade.md](CfC_BAOAB_Propagator_and_the_Cascade.md))**

Replace the Verlet-style force evaluation with the blended CfC propagator from [Closed\_Form\_and\_Hybrid\_Integration\_Strategies\_for\_Fock-PARFLM.md](../Closed_Form_and_Hybrid_Integration_Strategies_for_Fock-PARFLM.md). This eliminates the `autograd.grad(create_graph=True)` call for $V\_\theta$ entirely, replacing it with a forward-mode analytical matrix-exponential propagator whose backward pass is standard first-order backpropagation. This is the only mitigation that removes the cascade **at its source** rather than limiting its consequence. See §24 in the [CfC/BAOAB companion note](CfC_BAOAB_Propagator_and_the_Cascade.md) for a detailed analysis.

**9. Reduce $V\_\theta$ / $V\_\phi$ complexity**

Fewer `wells_per_head`, fewer xi channels, or a smaller score head would reduce the force field's expressivity and thus the Hessian's spectral radius. This is a last resort — it reduces model capacity to buy stability.

### 23.6 Diagnostic Experiment Plan

A three-run diagnostic session on a single LambdaLabs 2×H100 instance, all at $\gamma=0.10$ (the sweep's least-unstable candidate), 3K steps:

| Run | Changes from sweep baseline | What it tests | Estimated time |
|-----|---------------------------|---------------|:--------------:|
| A | P/E per-group clip (0.3) + `force_clamp_max=5.0` + `grad_clip=0.3` | Can Tier 1 tame $L=24$? | ~20h |
| B | `L=16`, sweep defaults | Is depth the bottleneck? | ~14h |
| C | `L=18` + P/E per-group clip + `grad_clip=0.3` | Compromise: shallower + clipped | ~16h |

**Decision tree after diagnostics:**

- **Run A clean, B and C also clean:** Keep $L=24$ with Tier 1 settings. The instability was a clipping problem.
- **Run A still spikes, Run B clean:** Depth is the root cause. Adopt $L=16$ for $d=1024$.
- **Run A still spikes, Run C clean:** $L=18$ is the minimum viable depth. Adopt $L=18$ with Tier 1 clipping.
- **All three spike:** The problem is deeper than depth or clipping alone. Escalate to Tier 3 (detach boundaries or CfC propagator).

The per-group clips for `P` and `E` are **always active** (they are in the global `GRAD_CLIP_OVERRIDES` dict), so all three runs benefit from them. The log output will show `top[override:P]=xxx` or `top[override:E]=xxx` when those groups are dominant.

Runs A and B can be parallelised across the two GPUs in single-GPU mode (one per `CUDA_VISIBLE_DEVICES`), reducing the total wall-clock from ~50h to ~36h (~1.5 days).
