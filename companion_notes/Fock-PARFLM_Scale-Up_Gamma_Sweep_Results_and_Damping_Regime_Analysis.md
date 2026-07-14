# Fock-PARFLM Scale-Up: Gamma Sweep Results and Damping Regime Analysis

**Author:** Dimitar P. Gueorguiev
**Date:** July 2026
**Status:** Living document — results updated as sweeps complete

---

## 1. Purpose

This document records and analyses the gamma sweep results for Fock-PARFLM at each scale-up tier (`d=384`, `d=768`, `d=1024`), compares the optimal damping coefficient across hidden dimensions, and discusses the physical implications of the observed damping-regime transition.

**Companion documents:**
- [Fock-PARFLM_Scale-Up_Comparative_Experiments.md](Fock-PARFLM_Scale-Up_Comparative_Experiments.md) — parameter counts, memory considerations, and GPU/OOM analysis for the three scale-up tiers.
- [Determining_optimal_gamma_for_SPLM.md](Determining_optimal_gamma_for_SPLM.md) — the depth-scaling closed-form predictor $\gamma^{\ast}\_{\text{depth}}$ and the four-estimator framework.
- [E4_sweep_results_and_discussion.md](E4_sweep_results_and_discussion.md) — SPLM damping sweep on Tiny Shakespeare (`d=384`, `L=8`).
- [Damped_Riemannian_Geodesics_in_the_SPLM_family-Comparative_Analysis.md](Damped_Riemannian_Geodesics_in_the_SPLM_family-Comparative_Analysis.md) — theoretical analysis of geodesic regimes.
- [Training_Instabilities_in_Fock-PARFLM_with_structured_V_theta.md](Training_Instabilities_in_Fock-PARFLM_with_structured_V_theta.md) — gradient spike taxonomy, `reverse_channel_scale` behaviour.

**Hardware:** All sweeps run on NVIDIA H100 80GB HBM3 (LambdaLabs instances). Sweep candidates: 3,000 steps each, WSD schedule (warmup 0→150, stable→1,950, decay→3,000), `eff_batch=32`.

---

## 2. d=768 Gamma Sweep (L=12, 137M params)

**Configuration:** `sweep-d768` preset — `d=768`, `L=12`, `batch=4×8×1`, `lr=2e-4`, `dt=1.0`, `mass_mode=logfreq` (mean $m \approx 1.4$).

### 2.1 Results

| Rank | $\gamma$ | Best PPL (3K steps) | Wall-clock (s) |
|:----:|:--------:|:-------------------:|:--------------:|
| 1 | **0.050** | **186.89** | 23,008 |
| 2 | 0.100 | 189.04 | 23,051 |
| 3 | 0.150 | 189.84 | 22,979 |
| 4 | 0.200 | 191.10 | 23,002 |
| 5 | 0.250 | 194.13 | 23,046 |
| 6 | 0.300 | 194.26 | 23,016 |
| 7 | 0.400 | 198.48 | 22,985 |
| 8 | 0.500 | *(in progress)* | — |

### 2.2 Observations

1. **Clean monotonic ranking.** Lower gamma → lower PPL across all 7 completed candidates, with no inversions. The spread from best (186.89 at $\gamma=0.05$) to worst (198.48 at $\gamma=0.40$) is ~12 PPL points — a substantial and unambiguous signal.

2. **Stability across all candidates.** Gradient norms are uniformly low (0.34–0.40 post-clip), no gradient spikes or watchdog triggers were observed at any gamma value, and alpha values are well-spread across all 5 xi channels ($\alpha \approx [0.49, 0.75, 0.95, 0.99, 1.00]$ by step 3,000). The `reverse_channel_scale` group is consistently the top gradient group but well within bounds (max reported: 0.6).

3. **The sweep did not probe below $\gamma=0.05$.** The true optimum may be even lower. The depth-scaling formula (§4.1) predicts $\gamma^{\ast}\_{\text{depth}} \approx 0.067$ for $L=12$, and the sweep winner ($\gamma=0.05$) is 25% below this prediction, consistent with the formula being a slight overestimate at this depth.

4. **Wall-clock near-constant.** All candidates took ~6.4h (23,000s) per 3,000-step sweep on a single H100. The damping coefficient does not meaningfully affect throughput.

---

## 3. d=384 Gamma Context (L=16, 53M params)

### 3.1 Status

**No dedicated gamma sweep has been run for Fock-PARFLM at $d=384$.** The running E5c configuration uses $\gamma=0.30$, inherited from the SPLM E4/E5 damping sweeps on Tiny Shakespeare (see [E4_sweep_results_and_discussion.md](E4_sweep_results_and_discussion.md)). Those SPLM experiments used a different architecture (plain SPLM, `L=8`, no PARF, no Fock mechanism), a different corpus (Tiny Shakespeare, not OpenWebText), and a different depth — so transferring $\gamma=0.30$ to Fock-PARFLM at `L=16` was always a convenience assumption rather than a validated choice.

### 3.2 What the depth-scaling formula predicts

The leak-free depth-scaling closed form from [Determining_optimal_gamma_for_SPLM.md](Determining_optimal_gamma_for_SPLM.md) §2.1:

$$\gamma^{\ast}\_{\text{depth}} = \frac{m}{L \Delta t} \ln(1/\rho)$$

with the leak-free calibration $\rho = 0.565$, $m \approx 1.4$, $\Delta t = 1.0$:

| Tier | $L$ | $\gamma^{\ast}\_{\text{depth}}$ | Nominal $\gamma$ used |
|------|:---:|:-------------------------------:|:---------------------:|
| `d=384` | 16 | **0.050** | 0.30 (6× above prediction) |
| `d=768` | 12 | **0.067** | 0.05 (sweep winner, close to prediction) |
| `d=1024` | 24 | **0.033** | 0.10 (sweep winner, 3× above prediction; all candidates unstable) |

The depth-scaling formula predicts that **all three tiers should use gamma values in the range 0.03–0.07**, with the dominant variable being $L$ (depth), not $d$ (hidden dimension). The fact that the d=768 sweep empirically confirms a value near the prediction lends credibility to the formula's extrapolations.

### 3.3 Implications for d=384

The E5c model at $d=384$ is running at $\gamma=0.30$ — **six times the depth-scaling prediction** of $\gamma^{\ast}=0.050$. This means the d=384 model is operating deep in the overdamped regime, with only 1.5% of initial velocity surviving the 16-layer stack (§4.1). While the model still trains well at $\gamma=0.30$ (best PPL ≈ 41 at step 75K), it is likely leaving significant capacity on the table: the conservative force field is doing the work of steering the hidden state, but the overdamped friction is dissipating most of the momentum that could carry information across layers.

**Recommendation:** Run a dedicated gamma sweep for d=384 Fock-PARFLM at the same 8 candidates (0.05, 0.10, ..., 0.50) using the `sweep-d768`-style methodology but with `d=384`, `L=16`. If the monotonic trend observed at d=768 repeats, the optimal gamma for d=384 may be 0.05–0.10, not 0.30.

---

## 4. d=1024 Gamma Sweep (L=24, 209M params)

**Configuration:** `sweep-d1024` preset — `d=1024`, `L=24`, `batch=1×16×2` (DDP across 2 GPUs, eff=32), `lr=1.5e-4`, `grad_clip=1.0` (sweep default). Split across two LambdaLabs 2×H100 instances (4 gamma candidates each): Node A ran [0.05, 0.10, 0.15, 0.20], Node B ran [0.25, 0.30, 0.40, 0.50].

The depth-scaling formula predicts $\gamma^{\ast}\_{\text{depth}} \approx 0.033$ for $L=24$.

### 4.1 Results (partial — 4 of 8 candidates complete as of July 14, 2026)

| Rank | $\gamma$ | Best PPL | Watchdog reloads | Worst instant grad | Primary spike groups |
|:----:|:--------:|:--------:|:----------------:|-------------------:|----------------------|
| 1 | **0.100** | **327.33** | 1 | 651.50 | `P`=417, `E`=98, `creation_gate`=26 |
| 2 | 0.250 | 337.47 | 2 | 7,870.94 | `P`=5,023, `creation_gate`=18 |
| 3 | 0.050 | 342.00 | 3 | 180.53 | `creation_gate`=110, `register`=25 |
| 4 | 0.300 | 376.19* | 1+ | 536.27 | `creation_gate`=311, `E`=47 |
| — | 0.150 | *(in progress, step 100)* | 0 | 0.57 | — |
| — | 0.200–0.500 | *(pending / in progress)* | — | — | — |

\*gamma=0.300 best PPL is from step 1,500 eval (watchdog-reloaded at step 1,849); the candidate had not finished when last observed.

### 4.2 Universal instability at L=24

**The defining characteristic of the d=1024 sweep is that every completed gamma candidate exhibited catastrophic gradient spikes and watchdog reloads.** This is qualitatively different from d=768, where all 8 candidates ran clean with gradient norms under 1.0 and zero watchdog triggers.

| | d=768 (L=12) | d=1024 (L=24) |
|---|---|---|
| Max grad_norm across all candidates | ~0.6 | 7,870 |
| Watchdog reloads (total) | 0 | 7+ |
| Candidates with spikes > 100 | 0 / 7 | 4 / 4 |
| Top spike group | `reverse_channel_scale` (mild) | `P`, `E`, `creation_gate` (catastrophic) |

The spike sources at d=1024 are **systemic** — they come from multiple parameter groups across all gamma values:

- **`P` (positional embedding):** The worst offender overall (norm up to 5,022.5 at gamma=0.250). This component has no dedicated per-group clip in the sweep preset.
- **`E` (input embedding):** Sustained elevated norms (40–98), especially at gamma=0.100 and 0.300.
- **`creation_gate`:** Occasional catastrophic spikes (norm 110–311), seen at every gamma tested.
- **`reverse_channel_scale`:** Relatively mild at d=1024 (norm up to 19.3), in contrast to d=768 where it was the dominant group.
- **`register`:** Elevated at gamma=0.050 (norm up to 24.5).

### 4.3 Non-monotonic ranking

Unlike d=768's clean monotonic trend (lower gamma → lower PPL), d=1024 shows a U-shaped curve with the minimum at gamma=0.100:

- gamma=0.050: 342.00 (worse than 0.100, degraded by 3 watchdog reloads losing ~500 steps each)
- **gamma=0.100: 327.33** (best, despite 1 watchdog reload and grad=651 spike)
- gamma=0.250: 337.47 (degraded by 2 watchdog reloads and catastrophic spikes)
- gamma=0.300: 376.19 (worst, watchdog-reloaded, still running)

The non-monotonicity is consistent with the interpretation that L=24 makes both extremes unstable: too little damping (gamma=0.05) allows the force cascade to oscillate freely across 24 layers, while too much damping (gamma=0.25–0.30) forces the force field to push harder to overcome friction, amplifying gradient norms in the potential-dependent parameters.

### 4.4 Confounding factor: sweep grad_clip

The sweep uses `grad_clip=1.0` (the default), while the full `d1024` training preset already sets `grad_clip=0.5`. The catastrophic spikes (grad=651 at gamma=0.100, grad=7870 at gamma=0.250) might be substantially attenuated with tighter clipping. However, the *relative* ranking between gamma candidates should be preserved, since all candidates used the same clip threshold.

### 4.5 Is L=24 too deep for Fock-PARFLM?

The universal instability at L=24 raises the question of whether L=24 is fundamentally too deep for the Verlet-style integrator with `autograd.grad(create_graph=True)` force computation.

**Evidence that depth is the root cause:**
- d=768 at L=12 is perfectly stable (zero spikes, zero watchdog triggers)
- d=1024 at L=24 is universally unstable (all candidates have catastrophic spikes)
- The force $f_\ell = -\nabla_h U$ is computed via `autograd.grad` at each layer, and the `create_graph=True` flag means the backward graph through the force compounds across all L layers — at L=24, this is a 24-deep second-order gradient chain
- The gradient magnitudes at L=24 (up to 7,870) are O($10^4$) larger than at L=12 (<1.0), far exceeding the 2× ratio that a simple linear scaling with L would predict

**Mitigation tiers (detailed analysis in [Training\_Instabilities\_in\_Fock-PARFLM\_with\_structured\_V\_theta.md](Training_Instabilities_in_Fock-PARFLM_with_structured_V_theta.md) §§23–24):**

| Tier | Strategy | Key levers | Invasiveness |
|:----:|----------|-----------|:------------:|
| 1 | **Clip the consequence** | Per-group clip for `P` and `E` (0.3), `force_clamp_max=5.0`, `grad_clip=0.3` | Config-only |
| 2 | **Shorten the cascade** | Reduce $L$ from 24 to 16–18, reduce $\Delta t$, increase mass | Config change |
| 3 | **Segment or remove the cascade** | Detach boundaries every $K$ layers; **BAOAB + CfC propagator** (replaces second-order force chain with first-order analytical propagator) | Code / arch refactor |

The Tier 1 interventions are implemented as of July 14, 2026: `P` and `E` now have dedicated per-group clip overrides in `GRAD_CLIP_OVERRIDES`, and `force_clamp_max` is exposed as a CLI-overridable field in `TrainConfig`.

The BAOAB + CfC propagator (Tier 3) is the only intervention that **removes the cascade at its source** rather than limiting its downstream consequence. It replaces the `autograd.grad(create_graph=True)` force evaluation with an analytical matrix-exponential propagator whose backward pass is standard first-order backpropagation — the per-layer Jacobian spectral radius drops from $> 1$ (Hessian-containing) to $\leq 1$ (rotation matrix), eliminating exponential amplification entirely. See §24 of the Training Instabilities note and §10 of [Closed\_Form\_and\_Hybrid\_Integration\_Strategies\_for\_Fock-PARFLM.md](Closed_Form_and_Hybrid_Integration_Strategies_for_Fock-PARFLM.md) for the full analysis.

**Diagnostic experiment plan:** A three-run session (Run A: Tier 1 at L=24; Run B: L=16 at defaults; Run C: L=18 + Tier 1) at gamma=0.10, 3K steps each, estimated ~50h sequential or ~36h parallelised across 2 GPUs. The results will determine whether Tier 1 suffices or whether depth reduction / CfC propagator is necessary. Full protocol in §23.6 of the Training Instabilities note.

---

## 5. Damping Regime Analysis: The Overdamped-to-Underdamped Transition

### 5.1 Per-layer and cumulative velocity retention

The BAOAB-like integrator update used in Fock-PARFLM is:

$$h_{\ell+1} = \text{LN}\!\Big(\,h_\ell + \frac{\delta_\ell}{1 + \Delta t\,\gamma} + \frac{\Delta t^2}{m_b(1 + \Delta t\,\gamma)}\,f_\ell\,\Big)$$

where $\delta_\ell = h_\ell - h_{\ell-1}$ is the velocity proxy and $f_\ell = -\nabla_{h_\ell} U$ is the conservative force. The per-layer velocity retention factor is:

$$r_{\text{layer}} = \frac{1}{1 + \Delta t \cdot \gamma}$$

and the cumulative velocity retention over the full $L$-layer stack (ignoring force re-injection at each layer) is:

$$r_{\text{total}} = r_{\text{layer}}^{\,L}$$

| Configuration | $\gamma$ | $L$ | $r_{\text{layer}}$ | $r_{\text{total}}$ | Regime |
|---------------|:--------:|:---:|:-------------------:|:-------------------:|--------|
| **d=384, L=16** | 0.30 | 16 | 0.769 | **0.015 (1.5%)** | Overdamped |
| d=384, L=16 (predicted $\gamma^{\ast}$) | 0.05 | 16 | 0.952 | **0.457 (45.7%)** | Underdamped |
| **d=768, L=12** | 0.05 | 12 | 0.952 | **0.557 (55.7%)** | Underdamped |
| d=768, L=12 | 0.30 | 12 | 0.769 | **0.037 (3.7%)** | Overdamped |
| d=1024, L=24 (predicted $\gamma^{\ast}$) | 0.033 | 24 | 0.968 | **0.453 (45.3%)** | Underdamped |

**The key finding:** The d=768 gamma sweep empirically selected a regime where **55.7% of the initial velocity is retained** through the full layer stack. This is not far from the leak-free SPLM calibration ($\rho = 0.565$), confirming that the depth-scaling framework's prediction holds at a very different scale, corpus, and architecture variant.

By contrast, the d=384 model at its nominal $\gamma=0.30$ retains only 1.5% — placing it firmly in the overdamped/gradient-flow regime. If the depth-scaling prediction is correct, the d=384 model *should* be operating at $\gamma \approx 0.05$ with ~46% retention, i.e. in the same underdamped regime as the d=768 sweep winner.

### 5.2 Two-channel damping: explicit friction vs. LayerNorm projection

The cumulative retention $r_{\text{total}}$ captures only the **explicit** damping from the friction coefficient $\gamma$. The full dynamical picture has a second, implicit damping channel: **LayerNorm projection**.

LayerNorm after each layer step projects $h_{\ell+1}$ onto the hypersphere $S^{d-1}(\sqrt{d})$, killing the radial component of velocity entirely and retaining only the tangential (on-manifold) component. This decomposes the effective damping into:

| Direction | Damping source | Strength |
|-----------|---------------|----------|
| **Tangential** (on-sphere) | Explicit $\gamma$ friction | $r_{\text{layer}} = 1/(1+\Delta t \cdot \gamma)$ per layer |
| **Radial** (off-sphere) | LayerNorm re-projection | Total (infinite effective damping) |

The dynamics are therefore **constrained to the sphere with near-zero tangential friction** (at $\gamma=0.05$) — the particle slides freely along the manifold surface, with only the conservative force field and the ~5% per-layer friction shaping the trajectory.

### 5.3 Physical analogies

The contrast between the two regimes has precise classical analogues:

**d=384 at $\gamma=0.30$ — Ball in honey.** The particle experiences such strong viscous drag that it effectively follows the instantaneous gradient of the potential at each step. Momentum is dissipated within 2–3 layers ($r_{\text{layer}}^3 = 0.46$, $r_{\text{layer}}^5 = 0.27$). The trajectory is a steepest-descent path on the sphere — it finds the nearest potential well and sinks into it. Shallow wells can trap the particle because it has no momentum to escape them. This is the **gradient-flow / overdamped Langevin** regime.

**d=768 at $\gamma=0.05$ — Spacecraft in a weak gravitational field.** The particle carries substantial momentum across layers (55.7% retention over the full stack). The conservative force field (V_theta, V_phi) provides gentle steering — like gravitational deflections — but cannot abruptly redirect the trajectory. The particle follows nearly-great-circle arcs on the sphere, with force-induced curvature providing the directional changes needed for language modelling. Shallow potential wells are traversed without capture; only deep wells provide enough force to significantly bend the trajectory. This is the **near-geodesic / weakly-damped Hamiltonian** regime.

![Damping regime comparison: d=384 overdamped vs d=768 near-geodesic](images/damping_regime_comparison_d384_d768.png)
*Figure 1.* Particle dynamics on the LayerNorm hypersphere. **Left:** d=384 at nominal $\gamma=0.30$ — the overdamped regime where the particle spirals rapidly into the nearest potential well with only 1.5% velocity retention. **Right:** d=768 at $\gamma=0.05$ — the near-geodesic regime where the particle follows long sweeping arcs across the sphere, carrying 55.7% of its initial momentum, bypassing shallow wells to reach deeper ones.

### 5.4 Riemannian geodesics on the constraint manifold

With LayerNorm constraining the hidden state to $S^{d-1}(\sqrt{d})$, a true (undamped, force-free) Riemannian geodesic satisfies $\nabla_{\dot{\gamma}} \dot{\gamma} = 0$ — parallel transport of the velocity vector — and traces great circles.

The model's dynamics approximate geodesics when:

1. $\gamma \to 0$ (no friction) — **nearly satisfied** at $\gamma = 0.05$
2. $f = 0$ (no external force) — **not satisfied**: V_theta and V_phi forces are active

So the d=768 trajectories are not geodesics but **forced orbits on the sphere with minimal friction**: the conservative force field bends the trajectory away from great circles, and the small friction provides just enough damping to prevent runaway oscillations. The trajectory is **geodesic-like between force interactions**, with force-induced curvature encoding the language-modelling computation.

This stands in sharp contrast to the d=384 regime at $\gamma=0.30$, where the trajectory is **nowhere near geodesic**: every step is dominated by friction, and the "velocity" at each layer is almost entirely determined by the local force, not by momentum from previous layers.

### 5.5 Why does the optimal damping decrease with $d$?

The d=768 sweep's preference for $\gamma=0.05$ over $\gamma=0.30$ is striking — the model strongly prefers 37× less damping than the d=384 configuration uses. Several factors contribute:

1. **Concentration of measure on high-dimensional spheres.** On $S^{d-1}$, as $d$ grows, most of the sphere's volume concentrates near any equator. Random perturbations become increasingly orthogonal to any given direction, meaning the force field's effect on the trajectory becomes geometrically more "gentle" — the force landscape is smoother in higher dimensions, requiring less friction to maintain stability.

2. **Depth and the depth-scaling formula.** The Fock-PARFLM at d=768 uses $L=12$ layers while d=384 uses $L=16$. The depth-scaling formula $\gamma^{\ast} = (m/L\Delta t) \ln(1/\rho)$ predicts that shallower networks need higher gamma to achieve the same total dissipation. But both the prediction ($\gamma^{\ast} = 0.067$) and the empirical sweep winner ($\gamma = 0.05$) are in the underdamped regime, while d=384's nominal $\gamma=0.30$ far exceeds the formula's prediction of $\gamma^{\ast}=0.05$ for $L=16$.

3. **Information transport.** At higher $d$, the residual stream has more capacity per token. The conservative force field can encode finer-grained semantic distinctions, and the hidden state can carry richer information through momentum. Excessive damping destroys this information at each layer, forcing the model to re-derive it from the potential — a wasteful computation that the underdamped regime avoids.

### 5.6 The d=384 question: regime shift or inherited mis-tuning?

The contrast between d=384's *nominal* $\gamma=0.30$ and d=768's *empirical* $\gamma=0.05$ is dramatic:

| | d=384 (nominal $\gamma=0.30$, L=16) | d=768 ($\gamma=0.05$, L=12) |
|---|---|---|
| Per-layer retention | 76.9% | 95.2% |
| Full-stack retention | 1.5% | 55.7% |
| Regime | **Overdamped** (gradient flow) | **Underdamped** (near-geodesic) |
| Force role | Dominates: sets velocity each step | Steers: deflects existing trajectory |
| Well escape | Cannot escape shallow wells | Coasts past shallow wells to reach deeper ones |
| Information transport | Re-derived from potential each layer | Carried by momentum across layers |
| Physics analogy | Ball in viscous fluid | Spacecraft under gravity |

**However, this comparison is between an empirically-swept $\gamma$ (d=768) and an inherited, un-swept $\gamma$ (d=384).** The d=384 value $\gamma=0.30$ was transferred from the SPLM E4/E5 experiments — a different architecture (plain SPLM, no PARF, no Fock), a different depth ($L=8$), and a different corpus (Tiny Shakespeare). No gamma sweep has been run for Fock-PARFLM at $d=384$.

Two competing hypotheses remain open:

1. **Regime shifts with $d$:** The d=384 model genuinely prefers overdamped dynamics ($\gamma \approx 0.30$), and the lower-gamma preference at d=768 reflects a dimension-dependent phase transition in the optimal dynamical regime.

2. **Underdamped is universally optimal:** The d=384 model *also* prefers $\gamma \approx 0.05$ (as the depth-scaling formula predicts, §3.2), and the inherited $\gamma=0.30$ is simply a mis-transfer — the right gamma was never found because no sweep was run.

The depth-scaling formula (§3.2) points toward hypothesis 2, predicting $\gamma^{\ast}=0.050$ for $L=16$ regardless of $d$. But this is a theoretical prediction, not empirical evidence. **A d=384 Fock-PARFLM gamma sweep is needed to discriminate between the two hypotheses.**

If hypothesis 2 is confirmed, the regime comparison table above would need revision: both tiers would be underdamped at their respective optima, and the contrast would be between *nominal practice* ($\gamma=0.30$, overdamped) and *optimal operation* ($\gamma \approx 0.05$, underdamped) at every hidden dimension.

---

## 6. Depth-Scaling Framework Cross-Validation

The d=768 gamma sweep provides a new empirical anchor for the depth-scaling framework:

| Source | Architecture | $L$ | $\gamma^{\ast}$ (empirical) | $\gamma^{\ast}\_{\text{depth}}$ (predicted, $\rho=0.565$) | $\rho$ (implied) |
|--------|-------------|:---:|:---------------------------:|:---------------------------------------------------------:|:----------------:|
| E5 leak-free (SPLM, Tiny Shakespeare) | SPLM | 8 | 0.10 | 0.100 | 0.565 |
| **d=768 sweep (Fock-PARFLM, OpenWebText)** | **Fock-PARFLM** | **12** | **0.05** | **0.067** | **0.70** |

The d=768 empirical winner ($\gamma=0.05$) is 25% below the formula's prediction ($\gamma^{\ast}=0.067$), implying an effective $\rho \approx 0.70$ — i.e., the Fock-PARFLM prefers to retain ~70% of kinetic energy at the final layer, somewhat more than the SPLM's 56.5%. This is plausible: the Fock mechanism's non-conservative reverse channel injects energy back into the system, so the "native" dissipation of the conservative backbone needs to be even lighter to maintain the right balance.

**Falsifiable prediction for d=1024:** At $L=24$, using the Fock-PARFLM-recalibrated $\rho=0.70$:

$$\gamma^{\ast}\_{\text{depth}} = \frac{1.4}{24} \cdot \ln(1/0.70) = 0.0583 \cdot 0.357 = \mathbf{0.021}$$

If the d=1024 sweep confirms a winner near $\gamma \approx 0.02\text{–}0.05$, the depth-scaling framework will have predicted across three architectures, three corpora, and three depth/width configurations with a single calibration constant.

---

## 7. Open Questions

1. **Should we run a d=384 gamma sweep?** The depth-scaling prediction and the d=768 results strongly suggest that $\gamma=0.30$ is suboptimal for d=384 Fock-PARFLM. A sweep would confirm whether the PPL improvement is significant (and whether it's worth re-training the E5c model at a lower gamma).

2. **Is the sweep resolution sufficient?** The d=768 sweep tested down to $\gamma=0.05$ but not lower. The true optimum may be at 0.03 or 0.02. Future sweeps should include candidates at 0.01, 0.02, and 0.03 in addition to the standard range.

3. **Does the Fock reverse channel shift $\rho$?** The implied $\rho=0.70$ for Fock-PARFLM vs $\rho=0.565$ for SPLM suggests that the non-conservative energy injection from the Fock mechanism compensates for some of the damping, requiring less friction in the conservative backbone. This should be tested by comparing the gamma optimum with and without the reverse channel enabled.

4. **Does the regime transition affect convergence speed?** If d=384 at $\gamma=0.05$ trains faster (not just to lower PPL) than at $\gamma=0.30$, this would have immediate practical implications for all running experiments.

---

## 8. Relation to other companion notes

- **[Determining_optimal_gamma_for_SPLM.md](Determining_optimal_gamma_for_SPLM.md):** The depth-scaling closed form and four-estimator framework. §2.1's falsifiable prediction for $L=16$ ($\gamma^{\ast} \approx 0.20$ at $\rho=0.18$, buggy anchor) can now be compared against the d=768 empirical result. The leak-free anchor ($\rho=0.565$) is more relevant; the Fock-PARFLM data suggests $\rho$ should be further recalibrated to ~0.70 for architectures with non-conservative energy injection.

- **[Damped_Riemannian_Geodesics_in_the_SPLM_family-Comparative_Analysis.md](Damped_Riemannian_Geodesics_in_the_SPLM_family-Comparative_Analysis.md):** The near-geodesic regime observed at d=768 provides empirical support for the theoretical analysis in this note. §5.4 above extends the geodesic discussion to the LayerNorm-constrained sphere.

- **[The_Overdamped_Limit_and_The_Position_of_The_2nd_Order_Lagrangian_Framework.md](The_Overdamped_Limit_and_The_Position_of_The_2nd_Order_Lagrangian_Framework.md):** The d=384 model at $\gamma=0.30$ is in the overdamped limit analysed in this note. The d=768 result at $\gamma=0.05$ demonstrates that the model empirically chooses *not* to be overdamped when given the option.

- **[Fock-PARFLM_Scale-Up_Comparative_Experiments.md](Fock-PARFLM_Scale-Up_Comparative_Experiments.md):** Parameter counts and memory analysis for the three tiers. The gamma sweep results here directly inform the `fixed_gamma` setting for the full training runs planned in that document.

- **[Exploiting_the_Riemannian_geometry_of_conservative_language_models.md](Exploiting_the_Riemannian_geometry_of_conservative_language_models.md):** The near-geodesic regime has implications for the Riemannian geometry programme. When the dynamics are nearly geodesic, the Jacobi metric (§2.1 of [Damped_Riemannian_Geodesics...](Damped_Riemannian_Geodesics_in_the_SPLM_family-Comparative_Analysis.md)) becomes an excellent approximation rather than just a theoretical construct.
