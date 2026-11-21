# A Hybrid Gaussian and Quadratic Background Potential

**What this note is.** §17 of the training-instabilities record: the hybrid of Gaussian wells and a quadratic background, proposed to bridge the gap between stability and expressivity. It was split on 2026-11-21 from [`Training_Instabilities_in_Fock-PARFLM_with_structured_V_theta.md`](https://github.com/dimitarpg13/semsimula-paper/blob/5551315/companion_notes/Training_Instabilities_in_Fock-PARFLM_with_structured_V_theta.md) (written June to August 2026). The text is that document's, unchanged except for its links, and keeps its section numbers, so a reference to "§N" anywhere in the record or in other notes finds its note through [the index](README.md).

> **Causal-leak caveat.** Every experiment in §1–§23 used the pre-fix, leaky Fock architecture: the reverse channel read a shared full-window register state, before the `prefix_causal_registers` fix. The stability findings stand; every perplexity reported here is leak-contaminated (optimistically low). The full caveat is in [the index](README.md#causal-leak).

## Contents

- [17. Hybrid Gaussian + Quadratic Background: Bridging the Stability–Expressivity Gap](#17-hybrid-gaussian--quadratic-background-bridging-the-stabilityexpressivity-gap)

---

## 17. Hybrid Gaussian + Quadratic Background: Bridging the Stability–Expressivity Gap

### 17.1. The tension

The preceding sections document a sharp dichotomy in the structured
$V_\theta$ landscape:

| V_theta variant | TinyStories best PPL | OpenWebText stability | Root cause |
|---|---|---|---|
| SQ3 (log-sum-exp quadratic) | **10.36** (Fock A2) | Blows up (§2–§4, §13) | $V \to +\infty$ as $h$ escapes all well centres |
| Gaussian (bounded mixture-PDF) | **15.95** (Fock G1) | Stable | $V \in [-\Sigma w_k, 0]$ by construction |

SQ3 wins on TinyStories by **~5.6 PPL** (a ~35% relative improvement)
because its unbounded far-field force provides strong global restoring,
keeping hidden states inside the attractor constellation. But that same
unbounded force is what triggers Blowups 1–3 on the wider, more diverse
OpenWebText distribution.

Gaussian wells solve the stability problem structurally but sacrifice
expressivity: their force decays to zero far from centres, leaving
far-field hidden states without directional guidance.

### 17.2. The hybrid proposal

The hybrid structured potential combines both:

$$
V_{\text{hybrid}}(\xi, h) = \underbrace{-\sum_k w_k(\xi)\,\exp\!\left(-\tfrac{1}{2}\,a_k(\xi)^{\!\top}(h-\mu_k(\xi))^2\right)}_{\text{Gaussian wells (local attractors, bounded)}} + \underbrace{\varepsilon\,\lVert h \rVert^2}_{\text{quadratic background (global restoring)}}
$$

The background quadratic generates a global restoring force
$f_{\text{bg}} = -2\varepsilon h$ that is negligible near the Gaussian
well centres (where the well forces dominate) but prevents the escape
problem that costs pure Gaussian wells their expressivity.

**Key structural properties:**

| Property | Pure Gaussian | Hybrid | SQ3 |
|----------|---------------|--------|-----|
| V range | $[-\Sigma w_k, 0]$ | $[-\Sigma w_k, +\varepsilon \lVert h \rVert^2]$ | $(-\infty, +\infty)$ |
| Far-field force | 0 (flat) | $2\varepsilon h$ (restoring) | unbounded |
| Escape risk | yes | **no** | no |
| Force blowup risk | no | **controlled by $\varepsilon$** | yes |

The force from the background quadratic is $2\varepsilon h$. At typical
hidden-state scales ($\lVert h \rVert \approx \sqrt{d} \approx 16$ for
$d = 256$), the quadratic force magnitude is $2\varepsilon\sqrt{d}$.
For $\varepsilon = 10^{-4}$, this is $\approx 0.003$ — negligible
compared to the Gaussian well force near a centre (peak $\approx 0.6 w_k / \sigma_k$)
but sufficient to prevent unbounded drift.

The background quadratic is structurally similar to
`ln_after_step=True` (LayerNorm after each Verlet step), which also
constrains $\lVert h \rVert$, but acts *through the potential* rather
than as a post-hoc normalisation — preserving the conservative-force
structure of the dynamics.

### 17.3. Connection to §9.4 and the existing G5 recipe

Section 9.4 already introduced the background quadratic as an
"optional" addition to the Gaussian + SARF architecture:

> "For additional safety, a mild global quadratic background can be
> added: $V_\theta^{\mathrm{SARF+bg}} = \ldots + \epsilon \lVert h \rVert^2$"

The G5 recipe in `colab_fock_gaussian_sarf_vtheta.ipynb` instantiated
this with $\varepsilon = 10^{-4}$ but was never evaluated. The
dedicated notebook
[`colab_hybrid_gaussian_quad_vtheta.ipynb`](../../notebooks/conservative_arch/scaleup/colab_hybrid_gaussian_quad_vtheta.ipynb)
now provides a systematic evaluation with:

- **$\varepsilon$ sweep**: 0 (pure Gaussian baseline), $10^{-5}$,
  $5\times 10^{-5}$, $10^{-4}$, $5\times 10^{-4}$, $10^{-3}$
- **Two base models**: FockPARFLM v2.1 (cells H1–H7) and PARFLM
  (cells P1–P7)
- **Two well counts**: $K = 8$ (primary) and $K = 16$ (higher capacity)
- All on TinyStories (d=256, L=8, 16k steps) for direct comparison
  with existing structured V_theta results

### 17.4. Stability analysis

The hybrid potential inherits the stability guarantees of both
components:

1. **Gaussian wells remain bounded.** The precision cap
   (`precision_max = 2/d`) from the §12 fix is retained, so the
   Gaussian force is structurally bounded by
   $0.607\,w_k / \sigma_{\min}$.

2. **The background quadratic is globally stable.** For any $\varepsilon \gt 0$,
   $V_{\text{bg}} = \varepsilon \lVert h \rVert^2$ is a convex,
   positive-definite potential with no local maxima or saddle points.
   Its Hessian is $2\varepsilon I_d$ — uniformly bounded and
   well-conditioned.

3. **The combined potential is coercive.** Unlike pure Gaussian
   ($V \to 0$ as $\lVert h \rVert \to \infty$), the hybrid satisfies
   $V_{\text{hybrid}} \to +\infty$ as $\lVert h \rVert \to \infty$
   for any $\varepsilon \gt 0$. This eliminates the escape risk
   entirely and matches the coercivity of SQ3 without SQ3's
   divergent force.

4. **V\_theta regularisation remains safe.** The $\lambda_V \cdot V^2$
   penalty is bounded because the Gaussian component is bounded and
   the quadratic background grows only as $\varepsilon^2 \lVert h \rVert^4$,
   which is controlled by `ln_after_step=True`.

### 17.5. Risk: over-constraining at large $\varepsilon$

If $\varepsilon$ is too large, the background quadratic dominates the
Gaussian wells, effectively collapsing the multi-modal attractor
landscape into a single global attractor at the origin. The
$\varepsilon$ sweep is designed to identify the transition:

- At $\varepsilon = 0$: pure Gaussian (G1 baseline, PPL ~15.95)
- At $\varepsilon \ll 1$: the wells dominate locally, background
  provides only far-field restoring — expected optimal regime
- At $\varepsilon \sim O(1)$: the background dominates, landscape
  becomes nearly quadratic — expected PPL degradation

The force ratio at a typical well centre provides a rough threshold:
the background force $2\varepsilon\sqrt{d}$ should be small compared
to the Gaussian well force $\sim 0.6 w_k / \sigma_k$. For $w_k \sim 1/K$,
$\sigma_k \sim \sqrt{d}$, this gives
$\varepsilon \ll 0.6 / (2Kd) \approx 1.5 \times 10^{-4}$ for $K = 8$,
$d = 256$ — consistent with the G5 design point of $\varepsilon = 10^{-4}$.

### 17.6. If the hybrid closes the gap

A successful hybrid result (PPL closer to SQ3's 10.36 than to pure
Gaussian's 15.95) would:

1. **Enable structured $V_\theta$ on OpenWebText** by providing SQ3-like
   expressivity with Gaussian-like stability — directly resolving the
   tension that motivated Sections 9–14 of this document.
2. **Simplify the stability hierarchy** (§15.8): the four-constraint
   framework would reduce to two constraints (LR + Plummer softening),
   since bounded $V_\theta$ force and $V^2$ penalty boundedness would
   be structural guarantees rather than runtime constraints.
3. **Validate the §9.4 conjecture** that the background quadratic's
   far-field restoring force accounts for most of SQ3's expressivity
   advantage over pure Gaussians.
