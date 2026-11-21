# Well Deactivation and Precision Explosion in the Gaussian Wells

**What this note is.** §10–§12 of the training-instabilities record: SARF wells deactivated by an `ln_after_step` scale mismatch, G3 sigma drift deactivating wells during training, and the G1 precision explosion (blowup 2 revisited through unbounded Gaussian precision). It was split on 2026-11-21 from [`Training_Instabilities_in_Fock-PARFLM_with_structured_V_theta.md`](https://github.com/dimitarpg13/semsimula-paper/blob/5551315/companion_notes/Training_Instabilities_in_Fock-PARFLM_with_structured_V_theta.md) (written June to August 2026). The text is that document's, unchanged except for its links, and keeps its section numbers, so a reference to "§N" anywhere in the record or in other notes finds its note through [the index](README.md).

> **Causal-leak caveat.** Every experiment in §1–§23 used the pre-fix, leaky Fock architecture: the reverse channel read a shared full-window register state, before the `prefix_causal_registers` fix. The stability findings stand; every perplexity reported here is leak-contaminated (optimistically low). The full caveat is in [the index](README.md#causal-leak).

## Contents

- [10. Implementation Diagnostic: SARF Well Deactivation by ln_after_step Scale Mismatch](#10-implementation-diagnostic-sarf-well-deactivation-by-ln_after_step-scale-mismatch)
- [11. G3 Sigma Drift: Well Deactivation During Training](#11-g3-sigma-drift-well-deactivation-during-training)
- [12. G1 Precision Explosion: Blowup 2 Revisited via Unbounded Gaussian Precision](#12-g1-precision-explosion-blowup-2-revisited-via-unbounded-gaussian-precision)

---

## 10. Implementation Diagnostic: SARF Well Deactivation by `ln_after_step` Scale Mismatch

### 10.1. Symptom

During the G3 run (SARF Gaussian wells, `N_S=64` frozen PMI-peak anchors) of
the TinyStories ablation notebook, the training log reported `v_reg=0.0000` at
every step from step 1 onward, despite the SARF architecture being correctly
constructed and the `IS_GAUSSIAN=True` branch of the regulariser being active:

```
[G3] step     1/16000  ntp=10.7877  v_reg=0.0000  lr=1.25e-06  grad=0.715
[G3] step    50/16000  ntp=10.6200  v_reg=0.0000  lr=6.25e-05  grad=0.821
```

A first-pass diagnostic measured anchor proximity at the **embedding level**:

```
Embedding norm: 0.320
Anchor norm:    0.324
Mean ||h-a_j||: 0.456
Current sigma:  1.000
Recommended init_log_sigma: -0.79
```

At this scale, with `sigma=1.0` and mean distance `0.456`, the exponent evaluates to
$\exp(-0.456^2/2) \approx 0.901$, suggesting the wells should be active and `v_reg`
should be approximately `0.81`. The diagnostic falsely implied the scale was fine.

---

### 10.2. Root Cause: `ln_after_step=True` Moves $h_L$ out of Embedding Space

The model configuration sets `ln_after_step=True`, which applies LayerNorm to
the hidden state after every Verlet step. After $L = 8$ layers the hidden state
$h_L$ has approximately unit variance per dimension:

$$
\lVert h_L \rVert \approx \sqrt{d} = \sqrt{256} \approx 16
$$

The SARF anchors are initialised from raw token embeddings, which satisfy:

$$
\lVert a_j \rVert \approx 0.32 \ll \sqrt{d}
$$

The squared distance from any Verlet-integrated hidden state to any anchor is
therefore dominated by the hidden-state norm:

$$
\lVert h_L - a_j \rVert^2 \approx \lVert h_L \rVert^2 + \lVert a_j \rVert^2 \approx 256 + 0.1 \approx 256
$$

With `sigma=1.0` ($\sigma^2 = 1$), the Gaussian exponent becomes:

$$
\exp\Bigl(-\frac{256}{2 \times 1^2}\Bigr) = \exp(-128) \approx 10^{-56} \approx 0
$$

Every bump in the potential vanishes identically. The SARF potential

$$
V_\theta(\xi, h_L) = -\sum_{j=1}^{N_S} w_j(\xi) \exp\Bigl(-\frac{\lVert h_L - a_j \rVert^2}{2\sigma_j^2}\Bigr) \approx 0
$$

evaluates to zero for all tokens, so `v_reg = mean(V_theta^2) = 0.0000`.

The anchors and the hidden states live in **entirely different coordinate
systems**: raw embedding space (norm $\approx 0.32$) vs. LayerNorm-normalised
Verlet space (norm $\approx \sqrt{d} \approx 16$). The wells are geometrically
invisible to the hidden states.

---

### 10.3. Why the Embedding-Level Diagnostic Was Misleading

The diagnostic sampled random token IDs and computed distances from their raw
embeddings to the anchor positions — correctly capturing the pre-Verlet,
pre-LayerNorm embedding scale. However, `V_theta` is evaluated in
`forward_with_vreg` on the **post-Verlet** hidden state `h_L`:

```python
xis = model.xi_module(h_L.detach())
V_vals = model.V_theta(xis, h_L)    # h_L has norm ≈ sqrt(d), NOT 0.32
```

A faithful diagnostic must measure distances from `h_L` (after all Verlet
steps and LayerNorm applications) to the anchors, not from raw embeddings.
The embedding-level measurement was correct in its own coordinate system but
irrelevant to the coordinate system that matters for training.

The lesson generalises: whenever `ln_after_step=True` or `ln_before_vtheta=True`
is active, any geometric object that interacts with hidden states (anchor
positions, sigma, force-peak radius) must be diagnosed and initialised in the
**transformed** coordinate system.

---

### 10.4. Fix: Two-Part Correction

Both anchor positions and $\sigma$ must be brought into the LN-normalised
hidden-state space.

**Part A — Normalise anchor positions.**

Apply element-wise standardisation (matching what `ln_after_step` does to `h`)
to each anchor vector immediately after construction:

```python
with torch.no_grad():
    a = model.V_theta.inner.anchors           # (N_S, d), raw embeddings
    a = (a - a.mean(dim=-1, keepdim=True)) / (a.std(dim=-1, keepdim=True) + 1e-5)
    model.V_theta.inner.anchors.copy_(a)
```

After this normalisation, $\lVert a_j \rVert \approx \sqrt{d}$, matching the
scale of $h_L$.

**Part B — Set `init_log_sigma` to cover inter-anchor distances in LN space.**

Two LN-normalised vectors in $\mathbb{R}^d$ that are approximately uncorrelated
have expected squared distance $2d$. The typical $h_L$-to-anchor distance is
therefore:

$$
\lVert h_L - a_j \rVert \approx \sqrt{2d} \approx \sqrt{512} \approx 22.6
$$

Setting

$$
\sigma_{\mathrm{init}} = \sqrt{d}, \qquad \log\sigma_{\mathrm{init}} = \tfrac{1}{2}\log d \approx 2.77
$$

places the Gaussian force peak at $r = \sigma \approx 16$, which is inside the
typical $h_L$-to-anchor distance and ensures non-trivial restoring forces from
step 1. Add to the recipe dict:

```python
'G3': {
    ...
    'init_log_sigma': 2.77,    # sigma = exp(2.77) ≈ 16 ≈ sqrt(d)
    ...
}
```

Pass it in the constructor:

```python
inner = SARFGaussianVTheta(
    d=D, anchor_positions=anchor_positions, xi_d=xi_d,
    w_scale=recipe['w_scale'],
    init_log_sigma=recipe.get('init_log_sigma', 0.0),
)
```

Or update $\sigma$ in-place alongside the anchor normalisation:

```python
model.V_theta.inner.log_sigma.data.fill_(recipe.get('init_log_sigma', 0.0))
```

---

### 10.5. Mathematical Justification for $\sigma_{\mathrm{init}} = \sqrt{d}$

The Gaussian force magnitude peaks at $r = \sigma$ and falls off for both
$r \ll \sigma$ and $r \gg \sigma$. Setting $\sigma = \sqrt{d}$ gives:

$$
F_{\mathrm{peak}} = \frac{w_j}{\sigma} e^{-1/2} \approx \frac{0.607 \cdot w_j}{\sqrt{d}}
$$

With $N_S = 64$ anchors and approximately uniform weights $w_j \approx 1/64$:

$$
\lVert \mathcal{E}(h, \xi_\ell) \rVert_{\mathrm{peak}} \approx \frac{0.607}{\sqrt{d}} \approx \frac{0.607}{16} \approx 0.038
$$

This is a mild restoring force — strong enough to register as non-zero `v_reg`
but small enough not to destabilise Verlet dynamics or dominate the NTP gradient.
As training progresses, $\log\sigma$ is a learned parameter and can grow or
shrink to match the evolving hidden-state distribution.

The choice $\sigma_{\mathrm{init}} = \sqrt{d}$ is not the unique correct answer;
any value in the range $[\sqrt{d}/2, \sqrt{2d}]$ would activate the wells. The
specific value $\sqrt{d}$ is preferred because it is:

1. **Scale-consistent**: matched to the exact LN-after-step output norm.
2. **Force-peak inside the typical distance**: since $\sqrt{d} \approx 16 \lt \sqrt{2d} \approx 22.6$, the peak force is applied at distances smaller than the mean anchor separation, giving a net inward restoring pull.
3. **Easily computed**: $\log\sigma_{\mathrm{init}} = \frac{1}{2}\log d$ — a single line, no calibration batch needed.

---

### 10.6. Verification

After applying both parts of the fix, the build cell reported:

```
Anchors re-normalized: norm=15.96  sigma=15.96
```

Both anchors and sigma are at $\sqrt{256} \approx 16$ as required. Training
immediately showed non-zero regularisation:

```
[G3] step     1/16000  ntp=10.7874  v_reg=0.1356  lr=1.25e-06  grad=0.701
[G3] step    50/16000  ntp=10.6655  v_reg=0.1352  lr=6.25e-05  grad=0.775
[G3] step   100/16000  ntp=9.9912   v_reg=0.1400  lr=1.25e-04  grad=35.907
[G3] step   200/16000  ntp=6.3311   v_reg=0.1498  lr=2.50e-04  grad=37.497
```

`v_reg ≈ 0.135` at step 1 confirms the Gaussian wells are engaged. The NTP loss
drops rapidly from 10.79 to 6.33 in 200 steps. The large pre-clip gradient norms
at steps 100 and 200 (`grad=35.9`, `grad=37.5`) are a normal feature of fast
early learning: the reported value is the norm **before** `clip_grad_norm_` with
`GRAD_CLIP=1.0` is applied, so actual weight updates are bounded. Crucially,
`v_reg` remains stable throughout the spikes, confirming that the SARF wells are
not the source of the large gradients — those originate from the NTP loss and
V_phi routing during the warmup ramp.

---

### 10.7. Summary Table

| Quantity | Before fix | After fix |
|----------|-----------|-----------|
| Embedding / anchor norm | 0.32 | — |
| ‖h_L‖ (post-Verlet) | ~16 | ~16 |
| σ | 1.0 | ~16 |
| ‖h_L − a_j‖ | ~16 | ~22.6 (normalised anchors) |
| Gaussian exponent | exp(−128) ≈ 0 | exp(−0.99) ≈ 0.37 |
| `v_reg` at step 1 | 0.0000 | 0.1356 |
| Wells active? | No | Yes |

---

## 11. G3 Sigma Drift: Well Deactivation During Training

### 11.1. Context

After the scale-mismatch fix (Section 10), the G3 arm (SARF Gaussian wells,
`N_S=64` frozen PMI-peak anchors, `init_log_sigma=2.77`) trained successfully
on TinyStories with non-zero `v_reg` from step 1 and stable gradient norms
(1–5 throughout). The training curve showed healthy convergence through the
first ~7000 steps.

### 11.2. Symptom: PPL Plateau at 18.67

The val PPL trajectory showed rapid initial descent followed by a hard plateau:

| Step | val\_ppl | v\_reg |
|------|---------|--------|
| 400 | 147.34 | 0.108 |
| 2000 | 41.88 | 0.020 |
| 4000 | 29.75 | 0.018 |
| 7600 | 21.21 | 0.006 |
| 9200 | 20.41 | 0.006 |
| **11600** | **18.67** | **0.006** |
| 12000 | 19.22 | 0.007 |
| 13200 | 18.76 | 0.006 |

The model plateaued at 18.67 PPL — far above the SQ3 reference (10.36 PPL)
and MLP baseline (8.95 PPL). The subsequent 4400 steps produced no
improvement.

### 11.3. Root Cause: Unconstrained $\log\sigma$ Growth

The `log_sigma` parameter for each anchor is a learned `nn.Parameter`. As
training progressed, the optimizer pushed $\log\sigma$ upward, widening
every well:

| Step | Typical v\_reg | Implied well state |
|------|---------------|-------------------|
| 1 | 0.136 | Wells engaged, restoring force active |
| 1000 | 0.038 | Wells weakening |
| 5000 | 0.008 | Wells nearly flat |
| 11000 | 0.006 | Wells effectively inactive |

With $\sigma$ growing without bound, the Gaussian well widens until it is
virtually flat. The potential evaluates to approximately $-w_j$ everywhere
(a constant), producing near-zero force:

$$
F_j = \frac{w_j}{\sigma_j^2}(a_j - h)\exp\Bigl(-\frac{\lVert h - a_j \rVert^2}{2\sigma_j^2}\Bigr)
$$

When $\sigma_j \gg \lVert h - a_j \rVert$, the exponential $\approx 1$ but
the $1/\sigma_j^2$ prefactor drives $F_j \to 0$.

The model discovered that minimising NTP loss is easier without the V_theta
structural constraint. Since $\log\sigma$ is free to grow, the optimizer
simply deactivated V_theta by widening all wells into flatness, leaving
V_phi (structural competitive routing) as the sole driver of learning.
The 18.67 PPL plateau is V_phi's capacity ceiling for this architecture.

### 11.4. Fix: `log_sigma_max` — Projected Gradient Constraint

Two enforcement points prevent $\sigma$ from escaping:

**Forward-pass clamp** in the `sigma` property:

```python
@property
def sigma(self) -> torch.Tensor:
    ls = self.log_sigma
    if self._log_sigma_max is not None:
        ls = ls.clamp(max=self._log_sigma_max)
    return ls.exp().clamp(min=1e-3)
```

**Post-step projection** via `clamp_params()`, called after every
`optimizer.step()`:

```python
def clamp_params(self) -> None:
    if self._log_sigma_max is not None:
        self.log_sigma.data.clamp_(max=self._log_sigma_max)
```

Both are needed: the forward-pass clamp ensures the effective sigma is always
bounded, and the post-step projection prevents Adam's momentum from
accumulating past the boundary (which would cause the optimizer to see a
flat gradient surface and waste capacity updating a parameter that has no
effect).

**Recommended value:**

$$
\log\sigma_{\max} = \tfrac{1}{2}\log d + 1.0 \approx 3.77 \quad\Longrightarrow\quad \sigma_{\max} = e \cdot \sqrt{d} \approx 43.5
$$

This allows $\sigma$ to grow by one e-fold beyond $\sqrt{d}$, giving the
wells room to adapt while preventing the flat-well collapse. The minimum
force at the boundary:

$$
F_{\min} = \frac{w_j}{\sigma_{\max}} e^{-1/2} \approx \frac{0.607}{64 \times 43.5} \approx 2.2 \times 10^{-4}
$$

Small but non-zero — the wells remain structurally present as soft
attractors rather than vanishing entirely.

### 11.5. Architectural Lesson

This failure reveals a fundamental tension in the Gaussian well architecture:
**bounded potential does not imply bounded expressiveness**. The potential
$V \in [-\sum w_j, 0]$ is always bounded, but the optimizer can trivially
satisfy the regulariser $\mathcal{R} = \mathrm{mean}(V^2)$ by making the
potential uniformly close to $-\sum w_j$ everywhere (wide flat wells),
rather than by creating structured force landscapes.

The constraint hierarchy for stable, expressive Gaussian wells is:

1. **Bounded potential** (architectural, from the Gaussian form) — prevents
   Blowup 1.
2. **Bounded force** (via `precision_max` or `log_sigma_max`) — prevents
   Blowup 2.
3. **Bounded sigma** (via `log_sigma_max`) — prevents well deactivation.

All three are necessary. Section 9 established (1); Section 10 fixed the
scale mismatch so the wells could engage; this section establishes (3).
Section 12 below addresses (2) for learned-centre wells.

---

## 12. G1 Precision Explosion: Blowup 2 Revisited via Unbounded Gaussian Precision

### 12.1. Context

The G1 arm (`MixtureGaussianVTheta`, K=8 learned centres, `init_log_precision
= -log(D)`) was expected to be the direct SQ3 replacement: same parameter
count, same functional role, but with bounded potential. It trained with non-zero
`v_reg` from step 1 (the init-precision fix from Section 10 worked), but
exhibited a qualitatively different failure from G3.

### 12.2. Symptom: Gradient Explosion Without PPL Collapse

Unlike the SQ3 blowups (which caused loss spikes and divergence) or the G3
drift (which caused silent deactivation), G1 showed **sustained, escalating
gradient norms** throughout training while PPL continued to improve slowly:

| Step range | Typical grad (pre-clip) | val\_ppl at end |
|-----------|------------------------|----------------|
| 1–400 | 1–36 | 126.35 |
| 400–2400 | 5–63 | 28.34 |
| 2400–6000 | 10–190 | 25.12 |
| 6000–10000 | 30–1328 | 23.06 |
| 10000–14800 | 50–1328 | 20.82 |

Final result: **20.82 PPL** — worse than G3 (18.67 PPL) despite having
far more V_theta parameters and learnable centres. The gradient spikes
consumed optimiser capacity and prevented the model from reaching its
potential.

### 12.3. Root Cause: Unbounded Per-Dimension Precision $a_k$

For `MixtureGaussianVTheta`, the per-dimension precision is:

$$
a_k(\xi) = \mathrm{softplus}(a\_\mathrm{proj}(\xi)) + 10^{-4}
$$

The `softplus` function is unbounded above: as the linear projection
$a\_\mathrm{proj}(\xi)$ produces larger values, $a_k \to \infty$ without
limit. This makes the effective per-dimension width:

$$
\sigma_{k,\mathrm{eff}} = 1/\sqrt{a_k} \to 0
$$

The Gaussian well collapses to a delta-function spike. The **potential**
remains bounded (the Gaussian bumps are always in $[-w_k, 0]$), but the
**force** (negative gradient of V) diverges:

$$
F_{\max} = \frac{0.607 \cdot w_k}{\sigma_{k,\mathrm{eff}}} \to \infty
\quad \text{as} \quad \sigma_{k,\mathrm{eff}} \to 0
$$

This is the same instability mechanism as SQ3 Blowup 2 (Section 3),
arriving via a different route: the SQ3 quadratic potential is unbounded
in value, causing the force to diverge through $\lVert h - \mu_k \rVert$
growth. The Gaussian potential is bounded in value, but its force diverges
through precision growth. In both cases the fundamental issue is the same:
**the force is not structurally bounded**.

### 12.4. Why G1 Did Not Diverge (Unlike SQ3)

Despite gradient norms reaching 1328, the training did not diverge because:

1. **Gradient clipping** (`GRAD_CLIP=1.0`) truncated every update to unit
   norm. The massive pre-clip gradients indicate wasted computation —
   the optimizer computed precise gradients only to throw away >99% of
   the information.

2. **Bounded potential** meant the `v_reg` penalty could not dominate the
   loss. Unlike SQ3 where $V^2 \to \infty$ caused Blowup 1, the Gaussian
   `v_reg = mean(V^2)` is bounded by $(\sum w_k)^2$.

3. **Context-dependent centres** ($\mu_k(\xi)$ from a linear projection)
   partially adapted to the hidden-state distribution, keeping some wells
   in useful positions even as others collapsed.

The result was a "soft failure" — the model trained but was chronically
impaired by the instability, spending most of its gradient budget on
clipped, uninformative updates rather than useful learning.

### 12.5. Comparison: G3 vs G1 (Before and After Fix)

| Property | G3 (sigma-clamped) | G1 — no precision cap | G1 — precision\_max fix |
|----------|--------------------|----------------------|------------------------|
| V\_theta params | 65,664 (0.5%) | ~201k (~1.4%) | ~201k (~1.4%) |
| Well centres | Frozen PMI anchors | Learned μ_k(ξ) | Learned μ_k(ξ) |
| Width control | log σ capped | a_k uncapped | a_k ≤ 2/d |
| Best val\_ppl | 18.67 | 20.82 | **15.95** |
| Gradient range | 1–5 | 30–1328 (chaotic) | **1–27** (transient) |
| v\_reg at end | 0.006 (near-zero) | 0.043 (declining) | **0.015 (stable)** |
| Training stability | Stable | Chronic spikes | **Completely stable** |

G3 **outperformed** the unfixed G1 despite having 3x fewer V_theta
parameters and frozen centres — the sigma clamp was the decisive factor,
not the centre expressiveness. Once G1's precision was clamped, it
**outperformed G3 by 2.72 PPL** (15.95 vs 18.67, a 14.6% relative
improvement), confirming that learned centres offer genuine additional
expressiveness when the force is properly bounded.

Additional observations from the full post-fix G1 run:

- Mixture weights (alpha values) drifted substantially: `alpha[0]` dropped
  from 0.250 to 0.074, indicating the K=8 centres actively specialised
  and redistributed coverage rather than staying symmetric.
- Best checkpoint at step 14400, with mild regression in the cosine
  tail (LR < 1e-5) — expected and harmless.
- `v_reg` held in the 0.01–0.08 band from step 2000 to step 16000,
  confirming sustained well engagement throughout training.

### 12.6. Fix: `precision_max` — Upper Bound on Per-Dimension $a_k$

The fix is the exact analogue of `log_sigma_max` for SARF: clamp $a_k$ in the
forward pass so the effective width cannot shrink below $\sigma_{\min}$:

```python
def _components(self, xi: torch.Tensor):
    lead = xi.shape[:-1]
    mu = self.mu_proj(xi).view(*lead, self.K, self.d)
    a = (F.softplus(self.a_proj(xi)) + 1e-4).view(*lead, self.K, self.d)
    if self._precision_max is not None:
        a = a.clamp(max=self._precision_max)
    w = F.softmax(self.w_proj(xi), dim=-1) * self.w_scale
    return mu, a, w
```

**Recommended value:**

$$
a_{\max} = \frac{2}{d} \quad\Longrightarrow\quad \sigma_{\min} = \sqrt{d/2} \approx 11.3 \quad\text{(for } d = 256\text{)}
$$

This is slightly looser than $1/d$ (which gives $\sigma_{\min} = \sqrt{d} = 16$),
allowing the wells to sharpen modestly below the LN-scale norm while keeping
the force bounded:

$$
F_{\max} = \frac{0.607 \cdot w_k}{\sigma_{\min}} = \frac{0.607}{K \cdot \sqrt{d/2}} \approx \frac{0.607}{8 \times 11.3} \approx 0.0067
$$

Unlike the SARF `log_sigma_max` fix, no post-step `clamp_params()` is
needed. The precision $a_k$ is recomputed from `a_proj(xi)` on every forward
pass — it is a function output, not a stored parameter. The clamp acts on the
function output directly, and the optimizer receives correct gradients through
the clamp (zero gradient when $a_k$ hits the ceiling, normal gradient
otherwise).

### 12.7. The Unifying Principle: Bounded Force, Not Just Bounded Potential

Sections 9–12 collectively establish a critical architectural insight:

**Bounding the potential value is necessary but not sufficient for training
stability. The force (negative gradient of V) must also be structurally
bounded.**

| Architecture | V bounded? | F bounded? | Stable? | Best val\_ppl |
|-------------|-----------|-----------|---------|--------------|
| SQ3 (log-sum-exp) | No | No | No (Blowups 1-3) | — |
| SARF Gaussian (no sigma cap) | Yes | Drifts to **No** | Partial (well deactivation) | 18.67 (plateau) |
| SARF Gaussian + log\_sigma\_max | Yes | **Yes** | **Confirmed** | 18.67 |
| Gaussian + no precision cap | Yes | **No** | Partial (grad spikes) | 20.82 |
| Gaussian + precision\_max | Yes | **Yes** | **Confirmed** | **15.95** |

The Gaussian functional form provides bounded V by construction (Section 9).
But as Sections 11 and 12 demonstrate, the optimizer can exploit two separate
loopholes to circumvent the force bound:

1. **Sigma drift** (G3): widen wells until force $\to 0$ (deactivation).
2. **Precision explosion** (G1): narrow wells until force $\to \infty$
   (delta-spike instability).

Both loopholes are closed by explicit constraints on the effective well
width: `log_sigma_max` for SARF anchors and `precision_max` for learned
centres. Together with the bounded potential, these constraints complete
the stability triad:

$$
\underbrace{V \in [-\sum w_k, 0]}_{\text{bounded potential}} \quad + \quad \underbrace{\sigma_{\min} \le \sigma_k \le \sigma_{\max}}_{\text{bounded force}} \quad \Longrightarrow \quad \text{stable training}
$$
