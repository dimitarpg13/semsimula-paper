# On Training the PARF Force

A deep dive on how `V_phi` (the PARF pair-interaction scalar) is currently trained,
the cost structure of the chosen approach, the alternatives we surveyed, the
concrete issues in the prototype code, and the recommended path forward.

Companion to:

- Design doc: [`PARF_Augmented_SPLM_Architecture.md`](PARF_Augmented_SPLM_Architecture.md)
- Implementation: [`notebooks/conservative_arch/parf/`](../notebooks/conservative_arch/parf/) (model, causal probe, trainer, smoke test)
- Sibling architecture (Q9d, layer-type Helmholtz hybrid): [`Scalar_Potential_based_Helmholtz_Architecture.md`](Scalar_Potential_based_Helmholtz_Architecture.md)
- SPLM family causal-leak bug & fix (the inherited `causal_force` invariant): [`Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md`](Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md)
- Anchor experiments: [`Helmholtz-HSPLM_Path_Forward_and_Experiments.md`](Helmholtz-HSPLM_Path_Forward_and_Experiments.md)

---

## 0. TL;DR

We currently train PARF under **Algorithm A** of the design doc (§7): pure NTP
cross-entropy backpropagation through the velocity-Verlet integrator. Inside
each layer we compute the force as

$$
f^{(\ell)} = -\nabla_h \Bigl( V_\theta(\xi, h) + \sum_{s \lt t} V_\phi(h_t, h_s^{\text{detach}}) \Bigr)
$$

via a single `torch.autograd.grad(..., create_graph=True)` call, so the outer
`loss.backward()` can backpropagate through the force itself and update both
$V_\theta$ and $V_\phi$. This is *correct* and gives an unbiased gradient
estimate of the cross-entropy loss with respect to all parameters.

It is **not, however, the cheapest way to train a learned conservative
force**. The approach pays a 2–3× wall-clock and 2× memory tax for a piece of
machinery (the second-order autograd graph through the inner force) that is
purely a notational convenience. Several lower-cost alternatives preserve the
exact gradient (analytic forces, adjoint backsolve, gradient checkpointing on
$V_\phi$); a few well-known shortcuts change the gradient and trade a small
amount of bias for a large compute saving (truncated 2nd-order BPTT,
score-matching auxiliary loss). For our prototype scale on Apple MPS the
binding constraint is **memory** (the unstructured MLP variant of $V_\phi$ OOMs
at the Helmholtz `B=16, T=128` protocol shape), so the next concrete win is
gradient checkpointing on the pair sum (already implemented; opt-in;
correctness-verified by bit-identical 5-step training trace). Any move beyond
that — analytic forces, adjoint, mixed precision — should be deferred until
the structural cell lands and we know the pair-energy form is worth keeping.

---

## 1. Table of contents

1. [What is the PARF force, in one equation](#2-what-is-the-parf-force-in-one-equation)
2. [Current implementation: Algorithm A with second-order autograd](#3-current-implementation-algorithm-a-with-second-order-autograd)
3. [The chain rule, written out](#4-the-chain-rule-written-out)
4. [Costs of the current approach](#5-costs-of-the-current-approach)
5. [Issues in the current code](#6-issues-in-the-current-code)
6. [Options for training the PARF force](#7-options-for-training-the-parf-force)
7. [Side-by-side comparison](#8-side-by-side-comparison)
8. [Recommendations](#9-recommendations)
9. [References](#10-references)

---

## 2. What is the PARF force, in one equation

For each layer $\ell$ and token position $t$, the per-token **energy** is

$$
U^{(\ell)}_t = V_\theta\bigl(\xi^{(\ell)}_t, h^{(\ell)}_t\bigr) + \sum_{s \lt t} V_\phi\bigl(h^{(\ell)}_t, h^{(\ell)}_s\bigr)
$$

with the design-doc §3 **causal reduction**: past tokens
$\lbrace h_s \rbrace_{s \lt t}$ are treated as fixed external sources, which
in PyTorch is `h_s = h.detach()`. The per-token **force** is the gradient of
this energy with respect to the active token:

$$
f^{(\ell)}_t = -\nabla_{h_t} U^{(\ell)}_t = -\nabla_{h_t} V_\theta(\xi_t, h_t) - \sum_{s \lt t} \nabla_{h_t} V_\phi(h_t, h_s)
$$

The hidden state then advances by a damped velocity-Verlet step

$$
h^{(\ell+1)}_t = h^{(\ell)}_t + \frac{\Delta t}{1 + \Delta t \cdot \gamma}\bigl(h^{(\ell)}_t - h^{(\ell-1)}_t\bigr) + \frac{\Delta t^2}{m (1 + \Delta t \cdot \gamma)} f^{(\ell)}_t
$$

where $m$ is a per-token mass and $\gamma$ a learned global damping. This
is the form documented in
[`PARF_Augmented_SPLM_Architecture.md`](PARF_Augmented_SPLM_Architecture.md)
§§2-3.

The training problem is: given a corpus of text, learn the parameters of
$V_\theta$ and $V_\phi$ so the resulting hidden-state dynamics produce a
language model with low next-token cross-entropy.

---

## 3. Current implementation: Algorithm A with second-order autograd

The trainer (`train_parf.py`) uses the Algorithm A of the design doc:
auxiliary-loss backprop through next-token-prediction (NTP), with no
Gumbel sparsity, no inner iterative solver, and no RL outer loop.

### 3.1 The per-layer step

The relevant code lives in `PARFLM._layer_step`:

```494:567:notebooks/conservative_arch/parf/model_parf.py
    def _layer_step(
        self,
        h: torch.Tensor,
        h_prev: torch.Tensor,
        m_b: torch.Tensor,
        gamma: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        cfg = self.cfg
        B, T, d = h.shape
        delta = h - h_prev

        xi_input = h.detach() if cfg.causal_force else h
        xi_now = causal_cumulative_mean(xi_input)

        h_in = h
        if not h_in.requires_grad:
            h_in = h_in.requires_grad_(True)

        h_src = h_in.detach() if cfg.causal_force else h_in

        V_th_per_token = self.V_theta(xi_now, h_in)              # (B, T, 1)

        if cfg.use_grad_checkpoint and self.training:
            P = torch.utils.checkpoint.checkpoint(
                self.V_phi, h_in, h_src, use_reentrant=False,
            )
        else:
            P = self.V_phi(h_in, h_src)                          # (B, T, T)
        mask = self._pair_mask_for(T, h_in.device)
        P_masked = P.masked_fill(~mask, 0.0)
        U = V_th_per_token.sum() + P_masked.sum()

        grad_U, = torch.autograd.grad(
            U, h_in,
            create_graph=self.training,
            retain_graph=True,
        )
        f = -grad_U

        denom = 1.0 + dt * gamma
        h_new = h_in + delta / denom + (dt * dt / (m_b * denom)) * f

        if cfg.ln_after_step:
            h_new = self._project(h_new)
        return h_new
```

Three machinery details matter:

1. **`h_src = h_in.detach()`** — the design-doc §3 causal reduction. The pair
   source slice has its autograd history severed, so $\partial U / \partial h$
   only flows back through the **query** side $h_t$, never through the **source**
   side $h_s$. This is what makes the per-token force a strict function of
   the past treated as a frozen external field.

2. **`autograd.grad(U, h_in, create_graph=self.training, retain_graph=True)`** —
   the **first-order** force computation. In `eval()` mode `create_graph=False`,
   so the call returns a leaf tensor (cheap, no second-order graph). In
   `train()` mode `create_graph=True`, so PyTorch builds the **second-order
   graph** — a graph that records how `grad_U` was itself computed in terms of
   $h_{\text{in}}$, $V_\theta$'s parameters, and $V_\phi$'s parameters. The
   outer `loss.backward()` later walks through this second-order graph.
   `retain_graph=True` keeps the per-layer forward graph alive for the outer
   backward.

3. **`f = -grad_U`** is plugged straight into the velocity-Verlet update. The
   resulting `h_new` carries autograd dependencies on $h_{\text{in}}$ (and
   transitively on every earlier layer's state), the parameters of $V_\theta$
   and $V_\phi$ (via the second-order graph through `grad_U`), the per-token
   mass and global $\gamma$.

### 3.2 The outer loop closes the chain

After all $L$ layers, the model produces logits and a standard cross-entropy
loss; `train_parf.py` then calls `loss.backward()` once per step:

```316:322:notebooks/conservative_arch/parf/train_parf.py
        _, loss = model(x, y)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(),
                                             train_cfg["grad_clip"])
        optim.step()
```

This walks the second-order graph backward all the way through:

`logits` -> `h_L` -> $L \times$ `velocity_verlet_step` -> $L \times$ inner
`autograd.grad` -> $L \times$ `V_phi(h_in, h_src)` and `V_theta(xi, h_in)` ->
into the parameters of $V_\theta$, $V_\phi$, embeddings, mass head, $\gamma$.

So **$V_\phi$ parameters get NTP gradients via the chain rule through the
force $f = -\nabla_h U$**, never via a direct supervision signal on the force
itself.

### 3.3 The full autograd flow as a Mermaid chart

```mermaid
flowchart TD
    Loss[NTP cross-entropy loss]
    Logits[logits = h_L @ E.T]
    HL[h_L final hidden state]
    Step[L times: velocity-Verlet step]
    Force[force f = -grad U via inner autograd.grad with create_graph=True]
    Energy[energy U = V_theta + sum_s V_phi h_t h_s detach]
    HSrc[h_s detached - causal reduction]
    HIn[h_in -- requires_grad]
    Vparams[parameters of V_theta and V_phi]
    Embed[token plus position embedding]

    Embed --> HIn
    HIn --> Energy
    HSrc --> Energy
    Energy --> Force
    Force --> Step
    HIn --> Step
    Step --> HL
    HL --> Logits
    Logits --> Loss

    Loss -. loss.backward .-> Logits
    Logits -. via E.T .-> HL
    HL -. d Loss / d h_L .-> Step
    Step -. through Verlet .-> Force
    Force -. through 2nd-order graph .-> Vparams
    Force -. through 2nd-order graph .-> HIn
    HIn -. through embed .-> Embed
```

The two arrows from `Force` to `Vparams` and `HIn` are exactly what
`create_graph=True` enables: without it, `Force` would be a leaf tensor and
the outer backward would stop there.

---

## 4. The chain rule, written out

To make precise what `loss.backward()` actually computes for $V_\phi$'s
parameters $\phi$, expand the chain rule for a single layer transition.

Write the velocity-Verlet step compactly as

$$
h^{(\ell+1)} = h^{(\ell)} + \frac{\Delta t}{1 + \Delta t \cdot \gamma}\bigl(h^{(\ell)} - h^{(\ell-1)}\bigr) - \frac{\Delta t^2}{m (1 + \Delta t \cdot \gamma)} \nabla_h U^{(\ell)}\bigl(h^{(\ell)}, h^{(\ell), \text{det}}; \theta, \phi\bigr)
$$

(I've folded the minus sign of $f = -\nabla_h U$ into the explicit subtraction
on the right.)

The loss gradient with respect to $\phi$ accumulated across all $L$ layers
is, by the chain rule,

$$
\frac{\partial \mathcal{L}}{\partial \phi} = \sum_{\ell=1}^{L} \frac{\partial \mathcal{L}}{\partial h^{(\ell)}} \cdot \frac{\partial h^{(\ell)}}{\partial \phi}
$$

The novel term, the one that requires the second-order graph, is

$$
\frac{\partial h^{(\ell+1)}}{\partial \phi} \supset -\frac{\Delta t^2}{m (1 + \Delta t \cdot \gamma)} \cdot \frac{\partial}{\partial \phi}\Bigl[\nabla_h U^{(\ell)}\Bigr]
$$

That is, we need the **gradient of the gradient** of $U$ — or, equivalently,
the second-mixed partial $\partial^2 U / (\partial h \cdot \partial \phi)$.
This is exactly what `create_graph=True` arranges: the inner backward that
produces $\nabla_h U$ becomes itself differentiable, so the outer backward
can hit it again to extract $\partial^2 U / (\partial h \cdot \partial \phi)$.

The remaining terms — through `delta`, through `m_b`, through `gamma`, through
the LayerNorm and through the embedding — are first-order and require no
special machinery.

This reframes the cost question crisply. The standard-MLP chain costs
$O(\text{forward})$. The PARF chain costs $O(\text{forward}) + O(\text{inner backward}) + O(\text{outer backward through inner backward})$ — i.e. one
extra "backward-equivalent" pass per layer, plus the activation memory to
support double differentiation.

---

## 5. Costs of the current approach

We measured the structural and MLP variants on Apple MPS at the prototype
shape ($d = 128$, $L = 8$, $T = 128$, $B = 16$, $v_{\text{hidden}} = 128$).
The full survey is in
[`notebooks/conservative_arch/parf/README.md`](../notebooks/conservative_arch/parf/README.md);
here are the binding numbers.

### 5.1 Wall-clock breakdown

| variant       | grad ckpt | mlp_h | s/step | 4000-step est | notes                                                              |
| ------------- | --------- | ----- | ------ | ------------- | ------------------------------------------------------------------ |
| structural    | off       |   —   | 3.77   | ~252 min      | the headline cell                                                   |
| structural    | on        |   —   | ~4.5   | ~300 min      | grad-ckpt tax ~20%; only useful when memory is the constraint       |
| MLP           | off       |  64   | OOM    | OOM           | (B,T,T,3d)+graph exceeds 13.5 GB MPS watermark                      |
| MLP           | on        |  64   | OOM    | OOM           | grad-ckpt cannot intercept the input concat                         |
| MLP           | on        |  16   | 6.4    | ~430 min      | reduces V_phi capacity 4x to fit in memory                          |

For comparison, the all-attention 5-seed E1 baseline at the same shape costs
~22 min/cell on the same hardware (one full Helmholtz Q9d AAAASSSS vh=128
seed). PARF is currently ~12× more expensive than attention per training
step.

### 5.2 Where the extra cost comes from

Per-layer, the PARF forward pass touches three quadratic-in-T data paths:

1. The pair-potential matrix $P^{(\ell)}_{b,t,s} = V_\phi(h_t, h_s)$ of
   shape $(B, T, T)$ — same asymptotic cost as the attention matrix.
2. The hidden activations *inside* $V_\phi$ — for the structural variant a
   $(B, T, T, H)$ tensor where $H = 32$ is the Theta MLP hidden width;
   for the MLP variant a $(B, T, T, 3d) = (B, T, T, 384)$ feature tensor.
3. The per-token energy $V_\theta(\xi, h)$ — cheap, $(B, T)$, dominated
   elsewhere.

The autograd graph keeps *all* of these alive across all $L$ layers until
the outer `loss.backward()` fires. With `create_graph=True` on the inner
gradient, every operation in the inner backward also enters the graph,
so the activation footprint roughly doubles again.

### 5.3 The MPS-specific failure

On Apple MPS we have a 13.5 GB watermark. The structural variant at $L=8$,
$B=16$, $T=128$, $H=32$ uses ~6-8 GB peak — fits. The MLP variant at the
same shape with $H_{\text{mlp}} = 64$ allocates a $(B, T, T, 3d) = $ 400 MB
feature tensor *per layer*; the autograd graph through that tensor pushes
peak well past 13.5 GB and the allocator OOMs. The cleanest mitigation is
gradient checkpointing on the pair sum (next section).

### 5.4 Gradient checkpointing — what it bought us

We added an opt-in `use_grad_checkpoint` flag (commit `f664414`) that wraps
`self.V_phi(h_in, h_src)` in
`torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`. The
non-reentrant variant is critical: the legacy reentrant-autograd path breaks
`create_graph=True`, but the non-reentrant variant uses
`saved_tensors_hooks` and supports the second-order graph that PARF needs.

Verification: the smoke test exercises a `[3/3] em-ln+gc` block that
verifies the 5-step training trace is **bit-identical** between
checkpointed and non-checkpointed runs (e.g. structural:
`5.5573 -> 5.5281 -> 5.5162 -> 5.6222 -> 5.5423` in both). The
gradient-Jacobian causal probe (which itself exercises the second-order
graph) also passes.

The actual savings depend on the variant:

- Structural V_phi: ~50% per-layer activation memory saved, ~20% wall-clock
  tax. Helpful for paper-scale runs but unnecessary at the prototype.
- MLP V_phi: cannot intercept the $(B, T, T, 3d)$ input concat itself
  (which is allocated *before* the checkpoint boundary), so the OOM
  persists at $H_{\text{mlp}} = 64$. Reducing $H_{\text{mlp}}$ to 16 fits
  but pays a 1.7× wall-clock tax.

---

## 6. Issues in the current code

A frank review of `model_parf.py`. None of these are correctness bugs (the
causal probe and the smoke training trace cover the correctness surface),
but each is a small wart worth knowing about.

### 6.1 `_pair_mask` is not a `register_buffer`

```482:492:notebooks/conservative_arch/parf/model_parf.py
    def _pair_mask_for(self, T: int, device: torch.device) -> torch.Tensor:
        """Cache the strict-lower-triangular mask (s < t)."""
        if not hasattr(self, "_pair_mask") \
                or self._pair_mask.shape[0] != T \
                or self._pair_mask.device != device:
            self._pair_mask = torch.tril(
                torch.ones(T, T, device=device, dtype=torch.bool),
                diagonal=-1,
            )
        return self._pair_mask
```

The cached lower-triangular mask is stored as an opaque attribute, not a
`register_buffer`. Consequences:

- It is not moved by `model.to(device)` (we sidestep this by checking
  `device != h.device` and rebuilding).
- It does not appear in `state_dict()`, so it is not serialised. That's
  fine because it is deterministic given $T$, but it does mean a
  `load_state_dict()` followed by a forward at a **different** $T$ rebuilds
  the mask on the fly.
- `state_dict()` round-trips work, just less self-documenting than a
  proper buffer.

Fix: register a maximum-size buffer in `__init__` keyed off `cfg.max_len`,
and slice it on demand.

### 6.2 Plummer softening $\varepsilon = 10^{-2}$ may be too tight

The structural $V_\phi$ uses

$$
V_\phi \propto \frac{1}{\sqrt{\lVert h_t - h_s \rVert^2 + \varepsilon^2}}
$$

with $\varepsilon = 10^{-2}$. For $d = 128$ hidden vectors with $\sigma \sim 1$,
typical pair distances are $\lVert h_t - h_s \rVert \sim \sqrt{2 d} \approx 16$,
so $\varepsilon$ is roughly a thousand times smaller than typical distances.
That sounds fine until you realise that in early training, before LayerNorm
has enforced its scale, two consecutive token states *can* be close to
collinear — and the gradient of the $1/r$ kernel through the softening
behaves like $r / (r^2 + \varepsilon^2)^{3/2}$, which peaks at
$r = \varepsilon / \sqrt{2}$. So the force grows as we approach the
softening scale and only then decays. If a few pair distances dip below
$10^{-2}$ early in training, those pairs dominate the per-token force.

Mitigation options, in order of safety:

1. Bump $\varepsilon$ to something like $10^{-1}$ or $\sqrt{d}/100$ (data-scale
   aware).
2. Replace the Plummer softening by a logistic gate
   $\bigl(1 + e^{-\beta(r - r_0)}\bigr)^{-1}$ centered on a learnable scale.
3. Apply gradient clipping per-pair before the sum — the strongest single
   pair would no longer dominate.

This is mostly a hypothetical concern; the current run at seed 0 has not
shown gradient norm explosions (peak `grad` reported in the training log
is ~120 in early steps and ~3-5 in steady state). Still worth knowing.

### 6.3 `init_scale = 0.02` for $V_\phi$ may not be "small as a
perturbation"

The design-doc rationale calls for $V_\phi$ to start as a *perturbation*
on the $V_\theta$ dynamics. We set `v_phi_init_scale = 0.02` which matches
standard transformer init. But because $V_\phi$ outputs feed into the
per-token force, which then perturbs `h_new` by

$$
\Delta h = -\frac{\Delta t^2}{m (1 + \Delta t \cdot \gamma)} \nabla_h V_\phi
$$

with $\Delta t = 1$, $m \approx 1$, $\gamma \approx 0.15$, we get
$\Delta h \sim 0.87 \cdot \nabla_h V_\phi$. At init, $V_\phi$'s gradient
norm is roughly $\sigma_{\text{init}}^2 \cdot d \cdot H \approx 0.02^2
\cdot 128 \cdot 32 = 1.6$, which is comparable to the LayerNorm-enforced
hidden-state scale. So $V_\phi$ is *not* a small perturbation at init —
it is comparable in magnitude to the SPLM dynamics from the very first
step.

Mitigation: scale `v_phi_init_scale` down by something like $1 / \sqrt{H}$
or zero-init the last `Linear` of each sub-MLP (a common transformer
trick: zero-init the output projection of an attention block so it starts
as identity). This would let $V_\phi$'s contribution grow gradually as
training proceeds, which is closer to the design-doc intent.

### 6.4 `retain_graph=True` is documented but the rationale isn't

Now that we collapsed V_theta + V_phi into one `autograd.grad` call, we
no longer need `retain_graph=True` *within* the layer. We keep it because
the *outer* `loss.backward()` walks back through the per-layer forward
graph, which would otherwise be freed at the end of the inner grad call.

This is correct, but a future reader may wonder why. The comment in the
code mentions the second-order graph requirement; an explicit note that
*the graph survives until the outer backward consumes it* would clarify.

### 6.5 No per-module gradient clipping

We use a global `grad_clip = 1.0` over `model.parameters()`. The
$V_\phi$ parameters get gradient signals through a longer chain (inner
backward -> outer backward -> 8 layers of velocity-Verlet) than the
$V_\theta$ parameters or the embeddings, and could in principle have
very different magnitudes. Mixing them in one global clip means a large
$V_\phi$ gradient can drag down the effective learning rate for
embeddings, and vice versa.

Per-module clipping (e.g. `clip_grad_norm_(self.V_phi.parameters(), c_phi)`
separately from the rest) is straightforward and may help training
stability, especially for the MLP variant which has 7× more $V_\phi$
parameters than the structural variant.

Currently not a blocking issue — gradient norms in the training log are
modest and well-behaved — but worth a knob to pull if MLP runs are
unstable.

### 6.6 Per-token $V_\theta$ summed bluntly

```536:553:notebooks/conservative_arch/parf/model_parf.py
        V_th_per_token = self.V_theta(xi_now, h_in)              # (B, T, 1)

        ...
        U = V_th_per_token.sum() + P_masked.sum()
```

We sum $V_\theta$ uniformly across all $(B, T)$ positions and add it to the
pair sum. This is fine for taking gradients with respect to $h$ — the per-token
contributions are independent. But it means $V_\theta$ and $V_\phi$ share a
single energy budget, with an implicit *equal weighting* between
single-particle and pair-interaction contributions. There's no learnable mixing
coefficient.

A natural extension: $U = V_\theta.\text{sum}() + \lambda \cdot P_{\text{masked}}.\text{sum}()$
with $\lambda$ a learned positive scalar that controls how strongly the
pair force contributes relative to the single-particle force. Could
initialise $\lambda$ small to enforce the "perturbation on $V_\theta$"
inductive bias from §6.3, then let it grow.

### 6.7 No checkpoint sharding across layers

`torch.utils.checkpoint.checkpoint` is currently applied to *each* call to
`self.V_phi(h_in, h_src)` independently. PyTorch supports `checkpoint_sequential`
which applies one checkpoint boundary per *segment* of layers, trading more
recompute for even less memory.

For our $L = 8$ stack, sharding into 4 segments of 2 layers each would halve
peak per-layer activation memory at the cost of recomputing the V_phi forward
~2× per backward. Worth experimenting if we hit memory limits at deeper $L$
or longer $T$.

### 6.8 Smoke test uses random integer "labels"

Cosmetic: the smoke test feeds `randint(vocab_size)` into both `x` and `y`, so
the loss decreases very slowly (random-target floor is `log(vocab_size) ≈
log(257) ≈ 5.55` for the smoke vocab). The 5-step trace shows fluctuation
around the entropy floor; this is *not* a sign that training is broken, just
that the smoke isn't supervised. A real text snippet would be a sharper
signal at the same compute cost.

---

## 7. Options for training the PARF force

The space of "how to train a learned force field that drives a discrete
integrator" is well-explored in adjacent literatures (Hamiltonian neural
networks, neural ODEs, score-based models). Here are the realistic options
for PARF, grouped by whether they preserve the gradient signal exactly or
introduce approximation.

### 7.1 Exact-gradient options

These all compute the same gradient $\partial \mathcal{L} / \partial \phi$ as
the current Algorithm A; they differ in how they get there.

#### 7.1.1 Algorithm A (current)

What we do today. Inner `autograd.grad(create_graph=True)` builds the
second-order graph; outer `loss.backward()` consumes it. Pros: simplest,
correct, no hand math. Cons: 2× wall-clock, 2× memory.

#### 7.1.2 Algorithm A + gradient checkpointing on $V_\phi$ (current, opt-in)

Wraps `self.V_phi` in `torch.utils.checkpoint.checkpoint(use_reentrant=False)`.
Trades ~15-25% compute for ~50% per-layer activation memory. Auto-enabled
for the MLP variant. Bit-identical training trace verified. See §5.4.

#### 7.1.3 Analytic-force backprop

Derive $f^{(\ell)} = -\nabla_h U^{(\ell)}$ in closed form and use first-order
autograd only (no `create_graph=True`). For $V_\phi$ structural this means
hand-deriving the chain rule through $\Theta_\phi$, $\Phi_\phi$ and the
softened $1/r$ kernel — six terms total, all bilinear in the W-projections
modulo the GELU bottlenecks.

```python
# Sketch (not implemented in this prototype):
def analytic_pair_force(h, h_src, V_phi):
    l_q, l_s = V_phi.W_l(h), V_phi.W_l(h_src)
    th_q, th_s = V_phi.W_theta(h), V_phi.W_theta(h_src)
    Phi = V_phi.Phi_kernel(l_q, l_s)            # forward
    Theta = V_phi.Theta_kernel(th_q, th_s)      # forward
    r = (((h - h_src.unsqueeze(...))**2).sum(-1) + V_phi.eps2).sqrt()
    P = -V_phi.C * Theta * Phi / r              # forward
    # analytic d P / d h_t (skipping the algebra) using the chain rule
    dP_dh = (
        -V_phi.C * dTheta_dh * Phi / r
        - V_phi.C * Theta * dPhi_dh / r
        + V_phi.C * Theta * Phi * (h - h_src) / r**3
    )
    return -dP_dh.sum(dim=2)   # sum over s < t
```

Pros: ~2-3× speedup, ~2× memory savings, no `create_graph=True`. Cons:
brittle to V_phi shape changes (any tweak to $\Theta_\phi$ or $\Phi_\phi$
requires re-deriving), error-prone, and unit-tests need to verify the
analytic gradient matches `autograd.grad` to fp32 noise. Worth the
investment only once the V_phi shape is locked.

This is the route taken by Hamiltonian Neural Networks
([Greydanus et al. 2019](https://arxiv.org/abs/1906.01563)) and Lagrangian
Neural Networks
([Cranmer et al. 2020](https://arxiv.org/abs/2003.04630)) for similar
reasons.

#### 7.1.4 `torch.func.grad` / `torch.func.jacrev`

PyTorch 2.x exposes a functional-API gradient operator that can sometimes
be more efficient than `autograd.grad` because it avoids constructing a
node-by-node graph. For a scalar function `U` with vector input `h`, the
backward cost is identical (one Vector-Jacobian product); the win, if any,
comes from avoiding Python overhead in the graph machinery. Worth trying;
unlikely to be more than 5-10% faster.

#### 7.1.5 Adjoint / Pontryagin backsolve

The Neural ODE community
([Chen et al. 2018](https://arxiv.org/abs/1806.07366)) showed that for
continuous-time ODE integrators, the gradient of a loss with respect to the
parameters can be computed in $O(1)$ memory in the depth by solving a
*backward adjoint ODE* and recomputing forward states as needed.

PARF uses a *discrete* velocity-Verlet integrator, not a continuous ODE,
but the same idea applies: simulate forward to get $h_L$, then simulate
the adjoint dynamics

$$
\dot{a}^{(\ell)} = -a^{(\ell+1)} \cdot \frac{\partial h^{(\ell+1)}}{\partial h^{(\ell)}}, \quad a^{(L)} = \frac{\partial \mathcal{L}}{\partial h_L}
$$

backward through the layer chain, recomputing the forward states $h^{(\ell)}$
on the fly (or storing them all, if memory permits).

Pros: $O(1)$ memory in $L$ (or $O(L)$ if you store everything). Cons: 2-3×
more compute (one forward + one backward + one recompute per backward step);
implementation complexity is high for discrete integrators (you need the
adjoint of the velocity-Verlet step in closed form). Marginal at $L = 8$;
worth considering for the deeper paper-scale stacks ($L \ge 16$).

### 7.2 Memory-only optimisations (no extra compute)

#### 7.2.1 Mixed-precision (`torch.amp.autocast`)

Wrap the V_phi pair sum in bf16 (MPS supports bf16 reasonably well; fp16
is risky for the $1/r$ kernel near the softening boundary). Saves ~2×
memory on the pair-sum activations; ~1.5-2× wall-clock speedup on MPS.

Risks:

- The Plummer softening $\sqrt{r^2 + \varepsilon^2}$ can underflow at fp16
  if $r$ is very small; bf16 has more dynamic range and is safer.
- The autograd graph through bf16 ops is fp32-stable in PyTorch 2.x by
  default but worth verifying.
- The causal probe (which exercises the second-order graph at random init)
  must still pass at bf16.

Cheap to try, easy to revert if it breaks anything.

#### 7.2.2 Activation offloading (CPU ↔ MPS)

Move large pair-sum activations to CPU between forward and backward and
swap them back when needed. Trades wall-clock for memory; worth
considering only if all other memory paths are exhausted.

### 7.3 Approximate-gradient options (change what is learned)

These explicitly trade gradient fidelity for compute. Worth flagging as
options on the table even though we're not pursuing them yet.

#### 7.3.1 Truncated 2nd-order BPTT

Add `h = h.detach()` between certain layer boundaries, severing the inner
gradient chain at those points. $V_\phi$ at layer $\ell$ then sees only
the **local** loss signal (from the cross-entropy at layers above $\ell$,
filtered through **first-order** dependencies, not through the inner force
chain). Equivalent to "local greedy training": cheaper but biased.

Caveat: without a per-layer auxiliary loss, the parameters of any layer
$\ell$ that gets fully detached from later layers receive **no** gradient
at all. So truncated 2nd-order BPTT must be paired with something like
deep-supervision auxiliary losses to be sensible.

We are not pursuing this; just calling it out.

#### 7.3.2 Score matching / force matching auxiliary loss

A common trick in score-based generative modeling
([Hyvärinen 2005](https://www.jmlr.org/papers/v6/hyvarinen05a.html))
is to train a learned score $\nabla_h \log p(h)$ such that it matches the
true score of the data. PARF could in principle borrow this if we had a
*target* force field — for example, the force field of attention itself
(treating attention as a teacher). Then $V_\phi$ would be trained directly
to match the teacher force, no NTP backprop required.

But this *defeats the framework's premise*: the whole point of PARF is
that the pair force should be a *conservative* learned field, not a
distillation of attention. Score matching is therefore off the table for
the PARF arm proper, but worth keeping in mind for hybrid distillation
experiments.

### 7.4 Algorithm B (iterative inner solver)

The design doc §7 mentions a Stage-2 *Algorithm B* that unrolls the per-layer
force computation into multiple inner SPLM steps (e.g. solve for $h^{(\ell+1)}$
implicitly via Newton iteration on the energy gradient). This is *more*
expensive per layer (the autograd graph is built $N$ times, not once). It is
about *quality and conditioning* of the inner solve, not about speed.

The relevant trade-off:

- Algorithm A (one Verlet step): cheap, can be unstable if $V_\phi$ produces a
  large transient force.
- Algorithm B (iterative inner solve): expensive, robust to large local
  forces.

For our prototype scale, Algorithm A is fine; gradient norms are well-behaved
in the training log. Algorithm B becomes interesting only if we bump up
$\Delta t$ or push $V_\phi$ harder.

### 7.5 Stage 1.5: Gumbel-softmax sparsity for $V_\phi$

The design-doc §7 also describes a Stage-1.5 add-on: a learned Gumbel-softmax
gate over past tokens that lets $V_\phi$ contribute only to a **learned-sparse**
subset of $\lbrace h_s \rbrace_{s \lt t}$ rather than the dense pair sum.
Cost reduction: $O(T \cdot k)$ instead of $O(T^2)$ for $k \ll T$.

Pros: reduces the pair sum from quadratic to linear-in-T; potentially cuts
$V_\phi$'s wall-clock 5-10× at $T = 1024$. Cons: introduces new
non-conservative dynamics (the Gumbel gate is itself an attention-like
gating, which conflicts with the framework's "conservative-only" stance);
adds new parameters (the gate logits); the gate's discreteness needs the
straight-through estimator
([Jang et al. 2017](https://arxiv.org/abs/1611.01144)) to be differentiable.

Stage 1.5 is the natural next step *if* the dense Algorithm-A cell lands at
or below the SPLM family anchors. It's also the natural *escape hatch* if
PARF underperforms on quality at the $O(T^2)$ dense form: maybe the dense
pair sum is the wrong inductive bias and a learned-sparse one would
generalise better.

---

## 8. Side-by-side comparison

The full alternative space, ranked by speedup / risk / preserves-the-gradient.

| approach                                | preserves grad? | speedup vs current | memory | implementation cost | when to pick                                                                                                                       |
| --------------------------------------- | --------------- | ------------------ | ------ | ------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Algorithm A (current)                   | exact           | 1×                 | 1×     | 0                   | what we do today; right default                                                                                                    |
| Algorithm A + grad checkpoint (current) | exact           | 0.8× (slower)      | 0.5×   | low (done)          | when memory is the binding constraint (MLP variant; deeper/longer cells); auto-enabled for MLP                                     |
| Mixed precision (bf16) on V_phi         | ≈ exact         | 1.5-2×             | 0.5×   | low                 | ~free MPS speedup once causal probe verified at bf16; do AFTER the structural cell lands                                            |
| Analytic-force backprop                 | exact           | 2-3×               | 0.5×   | high                | once V_phi shape is locked (post-OQ-1); 2-3× speedup at the price of brittleness to design changes                                  |
| `torch.func.grad`                       | exact           | ~1× to 1.1×        | 1×     | low                 | quick-win experiment; rarely meaningful for scalar U                                                                                |
| Adjoint / Pontryagin backsolve          | exact           | 0.5-0.7× (slower)  | O(1) in L | very high        | only at deep stacks (L ≥ 16); ours is L = 8; deferred                                                                              |
| Algorithm B (iterative inner solve)     | exact           | 0.2-0.5× (slower)  | 1×     | medium              | if Algorithm A's single-Verlet step is unstable; design-doc OQ-2; not currently needed                                              |
| Stage 1.5 Gumbel sparsity               | approximate     | 5-10× at T = 1024  | linear in T | medium-high     | if PARF lands well at T = 128 and we want to scale T without paying O(T²) per layer; or if dense PARF underperforms                |
| Truncated 2nd-order BPTT                | biased          | ~Lx                | 1/L×   | low                 | not recommended; pairs with deep-supervision aux losses to be useful                                                                |
| Score matching against attention        | not applicable  | n/a                | n/a    | n/a                 | conflicts with framework's conservative-only stance; off the table for PARF proper, on the table for distillation experiments only |

---

## 9. Recommendations

Concrete next steps, in priority order:

### 9.1 Don't touch the running cell

The structural seed-0 cell is in flight (PID 19311 at the time of writing,
step 1250 / 4000, val PPL 587 and dropping). Any optimisation that requires
restarting from scratch should wait until it lands.

### 9.2 Once the structural number is in, decide the OQ-1 next step from the data

Three branches based on `PARF structural seed 0` val PPL:

| outcome                            | next step                                                                                                                                          |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **PPL ≲ 130** (clear win)           | run the MLP variant (with grad checkpointing) to nail OQ-1; then n=3 power-up on structural; then consider analytic-force for paper-scale runs    |
| **PPL ≈ 130-138** (on par with VA / Q9d) | run the MLP variant for OQ-1; if structural beats MLP, structural prior is empirically active; commit V_phi shape and consider analytic-force      |
| **PPL ≳ 140** (loss vs all-attn)    | skip MLP, pivot to Stage 1.5 Gumbel sparsity; the dense PARF arm is not buying enough at quadratic cost                                           |

### 9.3 Code-level cleanups (low priority, no rush)

In rough order of cost-benefit:

1. Bump $\varepsilon$ in the Plummer softening to something more data-aware
   ($\sim \sqrt{d}/100$ rather than fixed $10^{-2}$). 1-line change. §6.2.
2. Zero-init the last `Linear` of each $V_\phi$ sub-MLP so the pair force
   starts as a true perturbation. 5-line change. §6.3.
3. Add a learnable mixing coefficient $\lambda$ between $V_\theta$ and the
   pair sum (initialised small to enforce the "perturbation on $V_\theta$"
   inductive bias). §6.6.
4. Register `_pair_mask` as a `register_buffer` keyed off `cfg.max_len`. §6.1.
5. Per-module gradient clipping for $V_\phi$. §6.5.

None of these is blocking; they are warts to be aware of when iterating.

### 9.4 If we hit a paper-scale CUDA box

The MPS-imposed memory ceiling drops away on a 24+ GB CUDA card. The MLP
variant fits at $H_{\text{mlp}} = 64$, $B = 16$, $T = 128$ without
checkpointing; deeper $L$ and longer $T$ become tractable; analytic-force
backprop becomes worth implementing for the 2-3× wall-clock saving on
ablation sweeps.

Order of attack on a CUDA box (when we get there):

1. Validate the structural cell at fp32 to confirm hardware parity with MPS.
2. Add bf16 mixed precision; re-verify causal probe.
3. Implement analytic-force backprop for the structural variant; bit-equality
   unit-test against `autograd.grad`; benchmark.
4. Add Stage 1.5 Gumbel sparsity if dense PARF needs to scale to longer T.

---

## 10. References

### Internal documents

- [`PARF_Augmented_SPLM_Architecture.md`](PARF_Augmented_SPLM_Architecture.md) — the design doc; §3 (causal reduction), §5.1 (structural V_phi), §7 (Algorithm A vs Algorithm B vs Stage 1.5).
- [`Scalar_Potential_based_Helmholtz_Architecture.md`](Scalar_Potential_based_Helmholtz_Architecture.md) — Q9d, the layer-type Helmholtz hybrid; sibling architecture sharing $V_\theta$.
- [`Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md`](Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md) — the inherited `causal_force` invariant; why both `.detach()` points exist.
- [`Helmholtz-HSPLM_Path_Forward_and_Experiments.md`](Helmholtz-HSPLM_Path_Forward_and_Experiments.md) — anchor experiments (all-attn 5-seed E1 baseline, em-ln SPLM, Q9d, Variant A).
- [`notebooks/conservative_arch/parf/README.md`](../notebooks/conservative_arch/parf/README.md) — implementation README with the full MPS memory + wall-clock survey table.
- [`notebooks/conservative_arch/parf/model_parf.py`](../notebooks/conservative_arch/parf/model_parf.py) — the model code.
- [`notebooks/conservative_arch/parf/causal_probe_parf.py`](../notebooks/conservative_arch/parf/causal_probe_parf.py) — the perturbation + gradient-Jacobian causal probe.

### External literature

- Greydanus, S. *et al.*, **Hamiltonian Neural Networks**, NeurIPS 2019. arXiv:[1906.01563](https://arxiv.org/abs/1906.01563). Closest precedent for training a learned conservative force field; uses analytic-force backprop (§7.1.3 in this doc).
- Cranmer, M. *et al.*, **Lagrangian Neural Networks**, ICLR 2020 workshop. arXiv:[2003.04630](https://arxiv.org/abs/2003.04630). Generalisation of HNN to time-varying systems; same analytic-force trick.
- Chen, R. T. Q. *et al.*, **Neural Ordinary Differential Equations**, NeurIPS 2018. arXiv:[1806.07366](https://arxiv.org/abs/1806.07366). The continuous-time adjoint method (§7.1.5 in this doc); $O(1)$ memory in depth.
- Chen, T. *et al.*, **Training Deep Nets with Sublinear Memory Cost**, 2016. arXiv:[1604.06174](https://arxiv.org/abs/1604.06174). The classic gradient checkpointing reference; PyTorch's `torch.utils.checkpoint` is a direct descendant.
- Hyvärinen, A., **Estimation of Non-Normalized Statistical Models by Score Matching**, JMLR 2005. [JMLR link](https://www.jmlr.org/papers/v6/hyvarinen05a.html). The score-matching principle; flagged in §7.3.2 as inapplicable to PARF proper but relevant for distillation experiments.
- Jang, E. *et al.*, **Categorical Reparameterization with Gumbel-Softmax**, ICLR 2017. arXiv:[1611.01144](https://arxiv.org/abs/1611.01144). The Gumbel-softmax estimator used by Stage 1.5 sparsity (§7.5).
- PyTorch documentation, [`torch.utils.checkpoint`](https://pytorch.org/docs/stable/checkpoint.html). The non-reentrant variant (`use_reentrant=False`) is what makes our gradient-checkpointed PARF compatible with the inner `autograd.grad(create_graph=True)` call.
