# Measuring the Price of Conservativity: a $\lambda$-Gated Relaxation Protocol

**Status:** proposed, not yet run. Derivation and protocol complete; no code
written beyond what already exists.
**Date:** 2026-09-19.
**Companion to:** [`Joint_Vtheta_QKNorm_Run_Diagnostic_Checklist.md`](Joint_Vtheta_QKNorm_Run_Diagnostic_Checklist.md) §10.8,
[`Fock_Inference_Productionization_Plan.md`](Fock_Inference_Productionization_Plan.md) §7,
[`Context_Mixing_Mechanisms_in_the_Conservative_Framework.md`](Context_Mixing_Mechanisms_in_the_Conservative_Framework.md) §8.

---

## 1. Why this experiment exists

A matched GPT-2 reaches **54.67** where the joint arm reaches **81.58**, on
identical data at identical tokens, with 2.66x fewer non-embedding
parameters. Five candidate explanations have been eliminated:

| candidate | result | where |
| --------- | ------ | ----- |
| well count | K=40 additive lost to K=8 joint | checklist §9.4 |
| anisotropic rank | participation ratio 3.68 of 4 — fully used | §9.4 |
| depth | L=16 not better, spikier | §7.5 |
| pair-path routing | 3.85% of compute; all-to-all is cheaper, not better | §10 |
| context representation | content-addressed pooling buys 2.1%, 16% of predicted | §10.8 |

What survives is the constraint itself: that the per-token update must be
the negative gradient of a scalar potential.

**Elimination is not measurement.** Five eliminations license "it is not
these five"; they license "therefore it is the sixth" only if the candidate
set is provably complete, which it is not. The protocol below measures the
sixth directly, at the cost of one warm-started 4,000-step anneal per arm.

---

## 2. What the constraint actually costs, in degrees of freedom

Let $f : \mathbb{R}^{d} \to \mathbb{R}^{d}$ be the per-token force at fixed
context, and $J = \partial f / \partial h$ its Jacobian.

**Proposition.** On a simply connected domain, $f$ is the negative gradient
of a scalar potential if and only if $J$ is symmetric everywhere.

This is the Poincaré lemma. The forward direction is Clairaut's theorem:
if $f = -\nabla U$ then, for $U$ twice continuously differentiable,

$$J_{ij} = -\partial_{i}\partial_{j}U = -\partial_{j}\partial_{i}U = J_{ji}.$$
 The converse is the standard
integrability result, and simple connectedness is what rules out the
punctured-plane counterexamples.

Symmetry is not a mild condition. $J$ has $d^{2}$ entries; requiring
$J = J^{\top}$ imposes one equation per strictly-upper-triangular entry:

$$\#\lbrace \text{constraints} \rbrace = \frac{d(d-1)}{2}, \qquad \frac{\text{constrained}}{\text{total}} = \frac{d(d-1)/2}{d^{2}} = \frac{d-1}{2d} \xrightarrow[d \to \infty]{} \frac{1}{2}.$$

At the deployed $d = 384$ that is **73,536 of 147,456 entries, or 49.87%**.
Half the linear response of the force field, at every point, is forbidden by
construction.

![Jacobian degree-of-freedom budget](figures/conservativity_price/dof_budget.png)

This is the sharpest statement of what the framework gives up, and it is
worth being precise about what it does *not* say. It does not say the model
loses half its capacity — the potential $U$ is still an arbitrary scalar
function, and scalar functions of $d$ variables are a rich class. It says
the *linear response* of the force is restricted to a subspace of half the
dimension. Whether that restriction costs anything in language modelling is
exactly the empirical question.

---

## 3. The relaxation

Replace the force law with a gated sum:

$$f_{\lambda}(\xi, h) = -\nabla_{h} U(\xi, h) + \lambda \cdot g_{\psi}(\xi, h), \qquad \lambda \in \mathbb{R}, \quad \lambda\big|_{t=0} = 0,$$

where $U = V_{\theta} + V_{\phi}$ is the existing potential and $g_{\psi}$ is
an unconstrained vector field with its own parameters.

### 3.1 The start is exact, not approximate

At $\lambda = 0$ the added term vanishes identically, so
$f_{0} = -\nabla_{h}U$ is **bit-identical** to the deployed model. The probe
therefore warm-starts from `_step28500_best.pt` and reuses §6.3's decay
unchanged, exactly as Alternative E did. This is what makes the comparison
against 81.58 controlled rather than merely similar: same checkpoint, same
schedule, same step count, same data order, one variable.

### 3.2 The defect is exactly linear in $\lambda$

Write $A(M) = \tfrac{1}{2}(M - M^{\top})$ for the antisymmetric part. The
Jacobian of the gated force decomposes as

$$J_{\lambda} = \underbrace{-\nabla^{2}_{h}U}_{\text{symmetric}} + \lambda J_{g}, \qquad A(J_{\lambda}) = \lambda  A(J_{g}),$$

because the Hessian of a scalar contributes nothing to the antisymmetric
part. Define the **conservativity defect**

$$\kappa(\lambda) = \frac{\lVert A(J_{\lambda}) \rVert_{F}}{\lVert J_{\lambda} \rVert_{F}} = \frac{\lambda \lVert A(J_{g}) \rVert_{F}}{\lVert -\nabla^{2}U + \lambda J_{g} \rVert_{F}}.$$

Three properties follow immediately. $\kappa(0) = 0$ exactly, not
approximately. It is monotone in $\lambda$ for small $\lambda$, with slope

$$\kappa'(0) = \lVert A(J_{g}) \rVert_{F} / \lVert \nabla^{2}U \rVert_{F}.$$

And as $\lambda \to \infty$ it saturates at

$$\kappa_{\infty} = \lVert A(J_{g}) \rVert_{F} / \lVert J_{g} \rVert_{F},$$

so it is bounded and dimensionless.

This quantity is already implemented. `conservativity_diagnostic.py` arm 1
computes it under the name `antisymmetry_ratio`:

```python
J_sub     = J_cons[probe_indices, :]
J_antisym = 0.5 * (J_sub - J_sub.T)
frob_J       = torch.norm(J_sub, p="fro").item()
frob_antisym = torch.norm(J_antisym, p="fro").item()
ratio_cons   = frob_antisym / (frob_J + 1e-12)
hess_pass    = ratio_cons < 0.02
```

Note the existing pass threshold of `0.02`, and note that the same file's
test C applies the *opposite* test to the reverse channel
(`curl_pass = ratio_Q > 1e-2`), confirming that force is genuinely
non-conservative. **The architecture already contains a non-conservative
channel.** This protocol does not introduce something foreign to the
framework; it makes an existing degree of freedom explicit and measurable.

### 3.3 The initialisation trap, and why it is already documented

The gradients of the gated term are

$$\frac{\partial f_{\lambda}}{\partial \lambda} = g_{\psi}, \qquad \frac{\partial f_{\lambda}}{\partial \psi} = \lambda \frac{\partial g_{\psi}}{\partial \psi}.$$

Setting both $\lambda = 0$ and $\psi = 0$ makes **both** vanish: the loss
depends on the product, so the origin is a saddle the optimiser never
leaves. A probe built that way reports no gain with nothing ever having
learned, which is indistinguishable from a genuine null.

![The gate and the saddle it avoids](figures/conservativity_price/gate_saddle.png)

The fix is asymmetric initialisation — $\lambda = 0$ with
$\psi \sim N(0, \sigma^{2})$, $\sigma = 0.02$. Then $\partial f/\partial\lambda = g_{\psi} \neq 0$
at step 0 while $\psi$ is frozen, $\lambda$ leaves zero on the first step,
and $\psi$ unlocks on the second. This is the same trap and the same fix
recorded in `Context_Mixing_Mechanisms_in_the_Conservative_Framework.md`
§8.4 for Alternative E's bilinear logit, where it was verified: gradient
magnitudes 1.6e+03 on the gated parameter against 0.0 for both under the
symmetric initialisation.

---

## 4. The confound, and the control arm that removes it

**A single arm cannot answer the question.** If $\lambda$ rises and
perplexity falls, two explanations fit equally well:

1. the model wanted a non-conservative force, or
2. the model wanted **more force capacity**, and $g_{\psi}$ supplied it.

Explanation 2 is not idle. $g_{\psi}$ adds parameters, and nothing prevents
a learned $g_{\psi}$ from being close to a gradient field — in which case
$\kappa$ stays near zero and the gain is pure capacity.

The protocol therefore runs **two arms with matched parameter counts**.
Arm N leaves the added field unconstrained:

$$f = -\nabla_{h}U + \lambda g_{\psi}(\xi, h), \qquad g_{\psi} : \mathbb{R}^{(K+1)d} \to \mathbb{R}^{d}.$$

Arm C routes the same parameter budget through a scalar, so the total force
is still a gradient:

$$f = -\nabla_{h}\left(U + \lambda \Phi_{\psi}(\xi, h)\right), \qquad \Phi_{\psi} : \mathbb{R}^{(K+1)d} \to \mathbb{R}.$$

| arm | added field | defect κ | what a gain would prove |
| --- | ----------- | -------- | ----------------------- |
| **N** (non-conservative) | vector-valued, unconstrained | free to grow | capacity **and/or** non-conservativity |
| **C** (conservative control) | scalar-valued potential | 0 by construction | capacity alone |

Arm C adds the same parameter budget and the same gating structure, but
routes it through a scalar potential, so $\kappa \equiv 0$ throughout by
construction. Matching is on $|\psi|$, not on architecture: $g_{\psi}$ emits
$d$ components and $\Phi_{\psi}$ emits one, so $\Phi_{\psi}$ is given
proportionally more width to equalise the count.

$$\Delta_{\text{capacity}} = \mathrm{PPL}_{\text{control}} - \mathrm{PPL}_{\mathrm{C}}, \qquad \Delta_{\text{conservativity}} = \mathrm{PPL}_{\mathrm{C}} - \mathrm{PPL}_{\mathrm{N}}.$$

The second quantity is the paper's number. The first is the confound it
would otherwise be mistaken for.

---

## 5. What $\lambda \gt 0$ destroys

The trade is not free in the other direction either, and this is what makes
the experiment interesting rather than merely destructive.

The Jacobi metric of
`Damped_Riemannian_Geodesics_in_the_SPLM_family-Comparative_Analysis.md` is
the conformal rescaling

$$\tilde{g}_{ij} = 2(E - V) g_{ij} = \Omega^{2} \delta_{ij},$$

with the damped generalisation $\Omega^{2}_{\ell} = 2 T_{\ell} m$. Its
construction **requires $V$ to exist**. If $f$ is not a gradient there is no
$V$, hence no $\Omega^{2}$, hence no Jacobi metric, no Christoffel symbols,
and no geodesic reading of the trajectories. Every diagnostic in the
Riemannian battery is defined only at $\kappa = 0$.

So $\lambda$ is not a knob between "worse" and "better". It is a dial
between **perplexity and geometric interpretability**, and the protocol
measures the exchange rate. That framing is what
`Geodesic_Preservation_Experiment.md` already argues is the programme's
structurally exclusive capability: *not* that the perplexity is lower, but
that the geometry of the dynamics is measurable at all.

A paper reporting only the cost is a post-mortem. A paper reporting the cost
**and** what the constraint uniquely buys is a contribution.

---

## 6. Outcomes

![Four outcomes](figures/conservativity_price/outcome_space.png)

Each quadrant is decisive, which is the point of pre-registering them:

- **High gain, $\kappa \gt 0.02$, Arm N beats Arm C.** The measurement. State
  the cost of conservativity in nats at matched parameters and data.
- **High gain, $\kappa \lt 0.02$, Arm C matches Arm N.** The gain was capacity.
  The constraint is free and the framework is vindicated on this axis.
- **No gain, $\lambda$ stays near zero.** The model, offered the option to
  violate the constraint, declines it. A strong positive result — and the
  most favourable outcome available to the framework.
- **$\kappa$ rises with no gain.** Suspect the $\lambda$ parameterisation or
  the Arm C matching before believing it.

---

## 7. Protocol

Derived from the Alternative E probe, which established every piece of this
machinery.

| gate | what | cost | criterion |
| ---- | ---- | ---- | --------- |
| 0 | build both arms; `_smoke` | minutes | `pair_potential` and force-law branches are opt-in, default path bit-identical |
| 1 | conservativity check at λ = 0 | minutes | arm 1 test A: force equals the finite-difference gradient with context frozen. **Arm C must also pass at λ > 0** |
| 2 | causality | minutes | `scaf.audit(...).assert_causal()` — not a hand-rolled future-perturbation probe |
| 3 | step-0 eval | minutes | **must read 84.31 exactly** on both arms, or the warm start is not bit-identical |
| 4 | Arm N, §6.3's 4,000-step decay | ≈5h | settled against 81.58; record κ and λ |
| 5 | Arm C, identical | ≈5h | settled against 81.58; κ must stay below 0.02 |
| 6 | geodesic preservation on both endpoints | hours | against the completed d=384 baseline |

**Total ≈10h of A100 time** for the two training arms.

**Instrumentation to log per eval:** $\lambda$ itself (per layer if
parameterised per layer), $\kappa$ from arm 1, and $\lVert \lambda g_{\psi} \rVert / \lVert \nabla U \rVert$ —
the fraction of the force carried by the unconstrained term. The last is the
most interpretable single number: it is the share of the dynamics the model
chose to take outside the framework.

**Pre-register before running**, in the manner of §9.5 and §10.7, and score
the prediction afterwards whichever way it falls. This programme is now 4
for 4 on saturating fits beating linear extrapolations; expect the
$\lambda$ trajectory to saturate and do not extrapolate its early slope.

---

## 8. The insertion point

One site. `MultiXiPARFLM._layer_forces` in `model_parf_multixi.py` already
assembles the force from two branches and returns it:

```python
        if split:
            return f_theta, f_phi

        f = f_phi if f_theta is None else f_theta + f_phi
        if cfg.force_clamp_max is not None:
            f = f.clamp(-cfg.force_clamp_max, cfg.force_clamp_max)
        return f
```

Arm N adds `f = f + lam * self.g_psi(xis, h_in)` before the clamp, guarded
on a default-off config flag. Arm C needs no change here at all — its extra
term enters `_pair_potential` as another scalar contribution to `U_pair`,
which is the seam Alternative A already used.

Two cautions carried over from the Alternative E probe, both of which cost a
run when they were missed:

1. **The optimizer state remap.** Adding parameters shifts every later flat
   index, and the existing `_trainable` fallback assumes `model.parameters()`
   order while the optimizer flattens as `[decay] + [no_decay]`. The
   `_old_flat_order_minus_new` helper and the `exp_avg` shape guard added for
   Alternative E handle this; extend the excluded-parameter set rather than
   writing a new path.
2. **The watchdog rollback seed.** Cell 6's anneal block seeds `_best.pt`
   from the step-28,500 file, which predates the new tensors. Re-seed it from
   the live model after the resume completes, as the Alternative E notebook
   now does.

---

## 9. Honest caveats

- **Nothing here is measured.** Every number is a derivation or a
  configuration value. The protocol's output is the measurement.
- **One scale, one seed.** d=384, 0.53B tokens. The claim this can support is
  "the cost of conservativity at this scale is X", not a statement about
  conservative architectures in general. Scope the wording to match; a
  reviewer will otherwise scope it for you.
- **$\kappa$ is measured on a submatrix.** Arm 1 probes a random subset of
  coordinates, not the full $d \times d$ Jacobian, which is why its threshold
  is 0.02 rather than machine epsilon. Report the probe count alongside.
- **Arm C may be hard to match honestly.** A scalar-valued $\Phi_{\psi}$ with
  the same parameter count as a $d$-valued $g_{\psi}$ has a different
  shape, and shape is not neutral. If the arms cannot be matched
  convincingly, report both the parameter-matched and width-matched variants
  rather than choosing the flattering one.
