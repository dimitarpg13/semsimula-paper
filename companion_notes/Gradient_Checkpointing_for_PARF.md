# Gradient Checkpointing for PARF: Design and Implementation

**Status:** Proposal — implements per-layer-step checkpointing for the
PARF Verlet stack, reducing peak GPU memory from O(L) layers' worth of
V_φ intermediates to O(1).

**Companion notebook:**
`notebooks/conservative_arch/parf/scripts/gradient_checkpoint_poc.ipynb`

---

## 1. Problem Recap

The PARF architecture's V_φ pair-interaction creates intermediate tensors
of shape (B, T, T, H\_score) at every layer.  Because the V_φ force
computation uses `torch.autograd.grad(..., create_graph=True)`, all L
layers' intermediates — *including* the 2nd-order computation graph built
by `create_graph` — must be held in GPU memory simultaneously throughout
the backward pass.

At training scale (B=16, T=512, H\_score=32, L=8):

| Memory item | Per layer | ×8 layers |
|-------------|-----------|-----------|
| V_φ forward intermediates (B,T,T,H) | ~512 MB | ~4 GB |
| 2nd-order graph overhead from `create_graph=True` | ~500 MB–1 GB | ~4–8 GB |
| **Sub-total V_φ activations** | ~1–1.5 GB | **~8–14 GB** |

By contrast, model parameters (~0.3 GB) and Adam optimizer states
(~0.6 GB) are negligible.  The existing workaround — halving the batch to
B=8 with `GRAD_ACCUM=2` — preserves the effective batch size but does not
address the fundamental scaling issue: doubling T quadruples the per-layer
footprint, and every additional layer multiplies it again.

---

## 2. Why Naive Checkpointing Failed

The paper's OOM forensics table records:

> *"Gradient checkpointing breaks under `create_graph=True` —
>  Naive `checkpoint(layer)` on S-block stack —
>  Disable checkpointing; rely on gradient accumulation only"*

The failure stems from two causes:

### A. `use_reentrant=True` (old PyTorch default)

The reentrant checkpoint implementation uses a custom `autograd.Function`
whose backward method re-runs the forward function under a special
context.  When the forward function itself calls
`autograd.grad(create_graph=True)`, this creates a *nested* autograd
backward inside the checkpoint's backward — a "backward inside a
backward."  The reentrant checkpoint's context management was not designed
for this case; it wraps the forward recomputation in `torch.no_grad()`,
which prevents the nested `autograd.grad` from building its required
2nd-order graph.

### B. Whole-stack checkpointing

Wrapping the entire `_stack_forward` (all L layers) in a single
`checkpoint()` call does not help: during the backward pass the checkpoint
recomputes ALL L layers, which recreates all the V_φ intermediates
simultaneously — the same memory peak as no checkpointing at all.

---

## 3. The Correct Approach: Per-Layer-Step Checkpointing

The solution is **per-layer-step checkpointing** with
**`use_reentrant=False`** (available since PyTorch 2.0).  Each individual
`_layer_step` call is wrapped in a checkpoint:

```python
for ell in range(cfg.L):
    if cfg.use_layer_checkpoint and self.training:
        h_new = torch.utils.checkpoint.checkpoint(
            self._layer_step,
            h, h_prev, m_b, gamma, dt, ell,
            use_reentrant=False,
        )
    else:
        h_new = self._layer_step(h, h_prev, m_b, gamma, dt, layer_idx=ell)
    h_prev = h
    h = h_new
```

### How it works

**Forward pass (per layer):**

1. `_layer_step` runs normally — V_φ creates (B,T,T,H) intermediates;
   `autograd.grad(create_graph=True)` builds the 2nd-order graph.
2. The checkpoint's `saved_tensors_hooks` intercept every tensor save
   (both V_φ forward activations *and* 2nd-order graph tensors) and
   replace them with lightweight "recompute" stubs.
3. Only the layer's output h_{ℓ+1} and input h_ℓ are retained.
4. After all L layers complete, peak memory holds just the L+1
   hidden-state snapshots h_0 … h_L plus the NTP loss graph.

**Backward pass (per layer, in reverse):**

1. Gradient flows from the loss through h_L toward h_{L-1}.
2. At layer L-1's checkpoint boundary the "recompute" stubs fire.
3. `_layer_step` re-runs for layer L-1: V_φ forward +
   `autograd.grad(create_graph=True)` execute again, recreating the full
   graph for this **one** layer only.
4. The outer backward differentiates through the recreated graph,
   computing parameter gradients for V_φ, V_θ, mass, and γ.
5. Layer L-1's temporaries are freed; gradient flows to h_{L-2}.
6. Repeat for each layer in reverse.

### Why `use_reentrant=False` is critical

With `use_reentrant=False`, PyTorch uses `saved_tensors_hooks` instead of
the old custom `autograd.Function`:

- The recomputation runs in a normal `torch.enable_grad()` context (not
  `torch.no_grad()` as with `use_reentrant=True`).
- `autograd.grad(create_graph=True)` inside the recomputed `_layer_step`
  correctly builds a 2nd-order graph that connects to the outer backward.
- Model parameters (V_φ weights, V_θ weights, γ, mass) are accessed
  through `self` during recomputation — they are *not* saved/restored by
  the checkpoint, so they accumulate gradients correctly.

---

## 4. Two-Level Checkpointing Architecture

The implementation supports two complementary levels.  Level 1 already
exists in the codebase; Level 2 is the new contribution.

### Level 1: V_φ-only checkpointing (existing)

Already present in `model_parf.py` (lines 821–824):

```python
if cfg.use_grad_checkpoint and self.training:
    P = torch.utils.checkpoint.checkpoint(
        self.V_phi, h_in, h_src, use_reentrant=False,
    )
```

This checkpoints only the V_φ forward pass.  The (B,T,T,H)
intermediates are discarded and recomputed when needed *during the inner
`autograd.grad` call*.  However, the 2nd-order graph tensors created by
`autograd.grad(create_graph=True)` are **still retained** for all L
layers.

**Savings:** ~30–40% of per-layer V_φ activation memory.
**Wall-clock cost:** ~15–25% slower per step.

### Level 2: Layer-step checkpointing (new)

Wraps the entire `_layer_step` in a checkpoint, discarding **all**
intermediates — V_φ forward activations, 2nd-order graph tensors, V_θ
analytical-grad tensors, and dynamics-update temporaries:

**Savings:** ~70–80% of total per-layer activation memory.
**Wall-clock cost:** ~40–60% slower per step (each layer forward runs
twice: once in forward, once in backward).

### Comparison

| Config | Per-layer retained | Peak V_φ memory (L=8) | Relative wall-clock |
|--------|--------------------|-----------------------|---------------------|
| No checkpointing | V_φ fwd + 2nd-order graph | ~10–14 GB | 1.0× |
| Level 1 (V_φ only) | 2nd-order graph only | ~6–8 GB | ~1.2× |
| **Level 2 (layer-step)** | **Layer outputs only** | **~1.5–2.5 GB** | **~1.5×** |

Level 2 **subsumes** Level 1: when the entire layer step is checkpointed,
the V_φ forward is automatically recomputed as part of the layer
recomputation.  Setting both flags simultaneously is safe but provides
no additional benefit (Level 1 adds only a small bookkeeping overhead
inside the already-checkpointed layer).

---

## 5. Code Changes

### 5.1 PARFConfig (model_parf.py)

Add one new field:

```python
@dataclass
class PARFConfig:
    # ... existing fields ...

    use_layer_checkpoint: bool = False
    # When True, each _layer_step call in _stack_forward is wrapped in
    # checkpoint(use_reentrant=False).  Discards ALL per-layer
    # intermediates (including the 2nd-order graph from create_graph=True)
    # and recomputes them one layer at a time during backward.
    # Reduces peak V_phi activation memory from O(L) to O(1) at the cost
    # of ~50% more wall-clock per step.
    # Requires PyTorch >= 2.0.
```

### 5.2 _stack_forward (model_parf.py)

```python
def _stack_forward(self, h0, x, return_trajectory=False):
    cfg = self.cfg
    gamma, dt = self.gamma, cfg.dt
    m_b = self.compute_mass(x)

    h = h0
    h_prev = h0

    traj = None
    if return_trajectory:
        traj = [h.detach().cpu()]

    for ell in range(cfg.L):
        if cfg.use_layer_checkpoint and self.training:
            h_new = torch.utils.checkpoint.checkpoint(
                self._layer_step,
                h, h_prev, m_b, gamma, dt, ell,
                use_reentrant=False,
            )
        else:
            h_new = self._layer_step(
                h, h_prev, m_b, gamma, dt, layer_idx=ell,
            )
        h_prev = h
        h = h_new
        if traj is not None:
            traj.append(h.detach().cpu())

    return h, traj
```

### 5.3 Identical change in SparsePARFLM._stack_forward (model_parf_sparse.py)

The sparse PARF variant shares the same stack structure and benefits from
the same per-layer checkpointing pattern.

---

## 6. Correctness Argument

Layer-step checkpointing is **mathematically equivalent** to the
uncheckpointed baseline.  The same function executes; only the tensor
memory lifecycle changes.

Specifically, during backward recomputation of layer ℓ:

1. **V_φ's parameters** are accessed through `self.V_phi` — the same
   `nn.Module` instance as in the original forward.  Their `.grad`
   attributes accumulate correctly across layers.

2. **V_θ's parameters** (structured or MLP) are accessed through
   `self.V_theta`.  If using Phase 2 analytical gradients, the
   `analytical_grad` method re-runs using standard PyTorch ops in an
   `enable_grad()` context, producing a correct first-order graph.

3. **`autograd.grad(U_phi, h_in, create_graph=True)`** re-runs inside
   the `enable_grad()` context of the recomputation.  It builds a fresh
   2nd-order graph connecting `grad_phi` to V_φ's parameters.  The outer
   backward differentiates through this graph, producing the correct
   V_φ parameter gradients.

4. **h_in → h_new** dynamics update re-runs, connecting h_new to h_in
   and to the force f.  The gradient chain h_L → h_{L-1} → ⋯ → h_0
   is reconstructed layer by layer, identical to the uncheckpointed
   version.

The companion notebook validates this empirically by comparing parameter
gradients between all three modes (no checkpoint, Level 1, Level 2) and
confirming max |Δgrad| / max |grad| < 10⁻⁶.

---

## 7. Memory Analysis

With layer-step checkpointing, the V_φ activation memory is **constant
in L** — only one layer's intermediates exist at any time.  The remaining
budget:

| Component | Scaling | Size at B=16, T=512, d=128 |
|-----------|---------|---------------------------|
| 1 layer V_φ intermediates + 2nd-order graph | O(B·T²·H) | ~1.5 GB |
| L+1 layer snapshots h_0 … h_L | O(L·B·T·d) | ~32 MB |
| Model parameters | O(params) | ~0.3 GB |
| Adam optimizer states | O(2·params) | ~0.6 GB |
| NTP loss graph (logits) | O(B·T·V) | ~0.8 GB |
| **Total peak** | | **~3.2 GB** |

### What becomes feasible

| Configuration | No ckpt | Level 2 ckpt | Notes |
|---------------|---------|-------------|-------|
| B=16, T=512, L=8 (current) | OOM on A100 40 GB | **~3.2 GB** ✓ | No grad accum needed |
| B=16, T=1024, L=8 | OOM on H100 80 GB | **~6.5 GB** ✓ | Fits A100 |
| B=16, T=2048, L=8 | — | ~25 GB | Fits H100; needs B=8 on A100 |
| B=16, T=512, L=32 | OOM everywhere | **~3.2 GB** ✓ | L is free with ckpt |
| B=16, T=512, d=512, L=8 | OOM everywhere | ~4 GB ✓ | Wider model fits easily |

The binding constraint shifts from **L × O(T²)** (all layers' V_φ
retained) to **1 × O(T²)** (single-layer V_φ window).  Adding layers is
free; the quadratic scaling in T is isolated to one layer at a time.

---

## 8. Performance Trade-offs

| Aspect | Without checkpointing | With layer-step ckpt |
|--------|----------------------|---------------------|
| Peak activation memory | O(L · B · T² · H) | O(B · T² · H) |
| Forward wall-clock | 1.0× | 1.0× |
| Backward wall-clock | 1.0× | ~2.0× (each layer recomputed) |
| Total step wall-clock | 1.0× | **~1.5×** |
| Gradient correctness | Exact | Exact (modulo FP reorder) |
| Min PyTorch version | Any | ≥ 2.0 |
| Scales with L | Memory grows linearly | **Memory constant** |
| Scales with T | T² per layer × L | T² per layer × 1 |

The ~50% wall-clock overhead is a favourable trade: it converts an
OOM-inducing configuration into one that fits comfortably, and the saved
memory can be reinvested into larger B, longer T, or more layers.

---

## 9. Interaction with Existing Remedies

| Remedy | Interaction with Level 2 |
|--------|-------------------------|
| **Gradient accumulation** (B=8, GRAD_ACCUM=2) | Orthogonal — still useful if per-layer V_φ at B=16 is too large (e.g. T=2048) |
| **Level 1 V_φ checkpoint** | Subsumed — when Level 2 is on, the entire layer (including V_φ) is recomputed; Level 1 adds no benefit |
| **Phase 2 analytical V_θ grad** | Fully compatible — the analytical grad is recomputed during the layer-step recomputation |
| **Sparse PARF (top-k V_φ)** | Compatible — `SparsePARFLM` can adopt the identical pattern |
| **P8 patches (Patch A–D)** | Transparent — patches modify V_φ internals, which are recomputed identically |

---

## 10. Known Limitations

1. **PyTorch ≥ 2.0 required.**  `use_reentrant=False` relies on
   `saved_tensors_hooks` infrastructure that is incomplete in earlier
   versions.

2. **Floating-point non-determinism.**  Recomputation may reorder
   floating-point operations compared to the original forward (e.g.
   different cuDNN kernel selection, different reduction order).  In
   practice the relative difference is < 10⁻⁶ in float32 and does not
   affect training dynamics.

3. **Single-layer V_φ is still O(T²).**  Layer-step checkpointing
   eliminates the L multiplier but does not change V_φ's inherent
   quadratic scaling.  For T > 2048 on 80 GB GPUs, further remedies
   are needed:
   - **Windowed V_φ:** restrict pair interactions to a local window of
     size W, reducing per-layer memory from O(T²) to O(T·W).
   - **V_φ chunking:** compute the (B,T,T,H) tensor in chunks along one
     T dimension.
   - **Sparse V_φ (existing):** the top-k gating in `SparsePARFLM`
     already reduces the effective pair count from T² to T·k.

4. **Wall-clock cost.**  The ~50% overhead per step is non-trivial.  For
   long training runs, consider:
   - `torch.compile` to fuse checkpoint recomputation with backward ops.
   - Mixed checkpointing: checkpoint only every 2nd or 3rd layer (saves
     most memory with less recomputation).

---

## 11. Implementation Roadmap

1. **Phase A (this PR):** Add `use_layer_checkpoint` to `PARFConfig` and
   `SparsePARFConfig`; modify `_stack_forward` in both model files.
   Validate with the companion POC notebook.

2. **Phase B:** Enable by default in Colab notebooks that target A100
   40 GB.  Remove `GRAD_ACCUM` workaround where Layer 2 alone suffices.

3. **Phase C:** Combine with sparse V_φ for a scaling-competitive PARF
   variant: layer-step checkpointing (L-independent memory) + top-k
   gating (T·k instead of T² per layer).

---

## Appendix: Why the existing V_φ checkpoint (Level 1) already works

The V_φ-only checkpoint in `model_parf.py` (lines 821–824) wraps
`self.V_phi(h_in, h_src)` in `checkpoint(use_reentrant=False)`.  This
discards V_φ's forward activations and recomputes them during the
*inner* `autograd.grad(U_phi, h_in, create_graph=True)` call — i.e.,
during the forward pass (not the outer backward).  The recomputed
activations feed into `autograd.grad`, which builds the 2nd-order graph
correctly.

The catch: the 2nd-order graph tensors (which are *outputs* of V_φ's
backward, not saved activations of V_φ's forward) are **not**
checkpointed by Level 1.  They persist for all L layers until the outer
`loss.backward()` completes.  This is why Level 1 saves only ~30–40%:
the 2nd-order graph overhead, which is roughly equal to the forward
activation cost, remains.

Level 2 eliminates this limitation by checkpointing the entire
`_layer_step` — which includes both V_φ's forward *and* the
`autograd.grad(create_graph=True)` call.  During the outer backward,
the recomputation rebuilds both the forward and the 2nd-order graph
for one layer at a time.

---

## Addendum: Integrated Design — Level 2 + Stage-1.5b Gathered V_φ

Level 2 checkpointing and Stage-1.5b gathered V_φ (see
`docs/PARF_Stage_1_5b_design.md`) address **orthogonal** scaling axes
and compose without interaction:

- **Level 2** wraps `_layer_step` from the *outside* (in
  `_stack_forward`), changing when intermediates are retained vs.
  recomputed.  It eliminates the L multiplier.
- **Stage-1.5b** changes what happens *inside* `_layer_step`, replacing
  the dense (B,T,T,H) V_φ evaluation with a gathered (B,T,k,H) form.
  It eliminates the T/k multiplier.

Neither change affects the other's code path.  The `autograd.grad`
call is identical in both dense and gathered forms — only the scalar
`U_pair` that feeds into it changes shape.

### Combined memory profile

| Mode | V_φ eval | Layer ckpt | Peak V_φ scaling | Peak at B=16, T=512, L=8, k=4 |
|------|---------|-----------|------------------|-------------------------------|
| A | Dense | No | O(L · B · T² · H) | ~10–14 GB (OOM on A100) |
| B | Dense | Yes (L2) | O(B · T² · H) | ~1.5–2.5 GB |
| C | Gathered | No | O(L · B · T · k · H) | ~64 MB |
| **D** | **Gathered** | **Yes (L2)** | **O(B · T · k · H)** | **~8 MB** |

Mode D's V_φ activation memory is negligible — smaller than a single
layer's hidden-state tensor h (which is B·T·d = 4 MB at these dims).
The binding constraint shifts to the score-head logits (B,T,T) and the
NTP loss graph.

### What becomes feasible with Mode D

| Configuration | Mode A (baseline) | Mode D (integrated) |
|---------------|-------------------|---------------------|
| B=16, T=512, L=8 | OOM A100 | ~2 GB total |
| B=16, T=2048, L=8 | — | ~3 GB total |
| B=16, T=2048, L=32 | — | ~3 GB total |
| B=16, T=512, L=8, H=128 (full V_φ capacity) | OOM everywhere | ~2 GB total |

The model can scale to longer context (T=2048), deeper stacks (L=32),
and full V_φ hidden width (H=128) simultaneously on an A100 40 GB.

### Wall-clock interaction

The two changes have **opposing** wall-clock effects:

- Level 2 adds ~50% overhead (layer forward recomputed during backward)
- Stage-1.5b *reduces* V_φ compute by ~T/k = 128× (at k=4, T=512)

Net effect: **Mode D is often faster than Mode A** despite
checkpointing, because the gathered V_φ compute savings dominate the
recomputation overhead.

### Code integration

The changes are independent — two config flags, two code sites:

```python
# _stack_forward — Level 2 (outer loop, unchanged from §5.2)
for ell in range(cfg.L):
    if cfg.use_layer_checkpoint and self.training:
        h_new = checkpoint(
            self._layer_step, h, h_prev, m_b, gamma, dt, ell,
            use_reentrant=False,
        )
    else:
        h_new = self._layer_step(h, h_prev, m_b, gamma, dt, layer_idx=ell)

# _layer_step — Stage-1.5b (inner body, see PARF_Stage_1_5b_design.md §3.4)
if cfg.use_gathered_v_phi:
    idx, m_g = self._sparse_topk_indices(pi, causal, T)
    h_src_g = h_src.unsqueeze(1).expand(B, T, T, d).gather(
        2, idx.unsqueeze(-1).expand(B, T, k, d),
    )
    P_g = self.V_phi.forward_gathered(h_in, h_src_g)    # (B, T, k)
    U_pair = (P_g * m_g).sum()
else:
    tilde_m = self._sparse_mask(pi, causal, T)
    P = self.V_phi(h_in, h_src)                          # (B, T, T)
    U_pair = (P * tilde_m).masked_fill(~causal, 0.0).sum()
```

Config:

```python
use_layer_checkpoint: bool = True    # Level 2
use_gathered_v_phi:   bool = True    # Stage-1.5b
```

### Correctness

The companion notebook
`notebooks/conservative_arch/parf/scripts/gradient_checkpoint_gathered_vphi_poc.ipynb`
validates all four modes (A–D) and confirms:

1. **Checkpointing is exact** within each V_φ mode: A≡B, C≡D
   (max |Δgrad|/|grad| < 10⁻⁵).
2. **Dense ≡ gathered** at low τ: A≈C, B≈D (exact at τ→0; the only
   finite-τ difference is the off-top-k score-head gradient, which is
   doubly suppressed by softmax concentration).
3. **Memory compounds**: Mode D peak is ~2–5% of Mode A.

### Implementation roadmap

1. **Phase A (now):** Level 2 checkpointing — add `use_layer_checkpoint`
   to PARFConfig, modify `_stack_forward`.  Single-flag, ~10 lines.
2. **Phase B (per PARF\_Stage\_1\_5b\_design.md):** Gathered V_φ — add
   `forward_gathered`, `_sparse_topk_indices`, branch `_layer_step`.
   ~2–3 days.
3. **Phase C:** Enable both flags in production notebooks.  Remove
   `GRAD_ACCUM` workaround.  Bump V_φ hidden to H=128.
