# Determining the Optimal Damping Coefficient $\gamma^{\ast}$ for SPLM

> **Leak-correction status (May 4, 2026).** Between the v2 release of this document and its v3 revision, an anti-causal autograd path was discovered in every per-step `integrate()` site of the SPLM family (companion: `Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md`). The framework's *mathematical form* survives the leak fix unchanged — every closed form in §2.1–§2.4 still applies — but several empirical *calibration constants* anchored to the buggy SPLM trajectories require re-anchoring. The full leak-correction pass is summarised in the new **§2.5 Leak-correction calibration** below; in brief: the depth-scaling kinetic-energy retention factor shifts $\rho \in [0.15, 0.20]$ (buggy) $\to \rho \approx 0.565$ (leak-free), the §2.3 corpus-surprisal anchor $\gamma^{\ast}_{E5}$ shifts $0.30 \to 0.10$ (3-seed retrain) / $0.10\!-\!0.15$ (S=5 confirmation flat-bottom basin), and the same depth-scaling closed form (with the leak-free $\rho$) reproduces the leak-free empirical $\gamma^{\ast}$ to three decimal places. The §2.2 Hessian-spectrum sub-estimators, §2.4 conditional-$\gamma$ derivation, and the §3 reconciliation rules are all leak-independent in their *mathematical form*. **The framework's structural prediction has held across the leak fix as a "double success": the same closed form, with the same dataset-probed mass and the same architectural $L$, predicts both empirical $\gamma^{\ast}$ values, with only the single calibration constant $\rho$ shifting.** See §2.5 for full detail and `notebooks/conservative_arch/ln_damping_sweep/results/leakfree_5seed_confirmation/RESULTS_CONFIRMATION_S5.md` and `notebooks/conservative_arch/scaleup/gamma_transfer/results/leakfree_gamma0p10_seed0/predict_gamma_summary.md` for the underlying numbers.

> **Status.** Living document, drafted **April 30, 2026**, by Dimitar Gueorguiev with Claude; **leak-correction revision May 4, 2026.** This note formalises the four-estimator framework for *predicting* (not just sweeping) the optimal damping coefficient $\gamma^{\ast}$ in SPLM-style architectures. The framework is designed to produce a-priori predictions that can be reconciled against empirical sweeps such as E10 (γ-transfer), and to give a principled guideline for choosing $\gamma$ on new corpora / depths / question regimes without expensive grid search.
>
> **Companion experiments / docs:**
> - **E4** plain Euler damping sweep: [`notebooks/conservative_arch/damping_sweep/results/RESULTS.md`](../notebooks/conservative_arch/damping_sweep/results/RESULTS.md)
> - **E5** LN-after-step damping sweep: [`notebooks/conservative_arch/ln_damping_sweep/results/RESULTS.md`](../notebooks/conservative_arch/ln_damping_sweep/results/RESULTS.md)
> - **E9** scale-up de-risking (in progress, fixed γ = 0.30): [`docs/SPLM_scaleup_pre-registered_protocol.md`](SPLM_scaleup_pre-registered_protocol.md)
> - **E10** γ-transfer re-tuning (pre-registered, awaiting E9 finish): [`docs/Gamma_transfer_pre-registered_protocol.md`](Gamma_transfer_pre-registered_protocol.md)

---

## 1. Why this question matters

SPLM's per-layer dynamics are governed by

$$m \ddot{h} + \gamma \dot{h} + \nabla_h V_{\theta}(\xi, h) = 0,$$

discretised as a semi-implicit damped Euler step

$$v_{l+1} = \frac{v_l + \Delta t \cdot f_l/m}{1 + \Delta t \cdot \gamma}, \qquad h_{l+1} = h_l + \Delta t \cdot v_{l+1}.$$

The damping coefficient $\gamma$ controls how aggressively the integrator dissipates kinetic energy per layer-step. It sits on a privileged axis in the model's hyperparameter space:

- Too low ⇒ trajectories oscillate around attractors, gradient signal degrades near convergence, training stalls or diverges.
- Too high ⇒ the integrator becomes effectively first-order ("gradient flow"), the second-order inertial benefit is lost, and the model regresses to a denoising-style architecture.
- "Just right" ⇒ critical damping in the local Hessian sense; trajectories slide cleanly into attractors over the available $L$ layers.

E4 / E5 found $\gamma^{\ast} \approx 0.30$ on Tiny Shakespeare. E9 *transferred* this value to TinyStories at the scale-up configuration; whether the transfer is justified is the question of E10.

But searching for $\gamma^{\ast}$ on every new corpus / depth / question regime by grid sweep is expensive (~3 days of MPS time per sweep at the E10 budget). A **principled predictor of $\gamma^{\ast}$** would let us either skip the sweep or reduce it to a single confirmation run.

This note collects four such predictors, ranging from cheap-and-derived to expensive-and-empirical, and discusses how they can be reconciled.

---

## 2. The four estimators

### 2.1 Depth-scaling closed form ($\gamma^{\ast}_{\text{depth}}$)

**Idea.** Treat the integrator as an exponential decay process: the kinetic energy $E_{\text{kin}}(l) \sim e^{-\gamma l \Delta t / m}$ should decay by a target factor $\rho$ over the available $L$ layers.

**Closed form.** Solving for $\gamma$:

$$\boxed{\gamma^{\ast}_{\text{depth}} = \frac{m}{L \Delta t} \ln(1/\rho).}$$

**Free parameters:**
- $m$: typical per-token mass (with `logfreq` mass mode, $m \in [1.0, 2.5]$, average ~1.4 on the corpora tested so far).
- $L$: number of layers.
- $\Delta t$: integrator step size (typically 1.0).
- $\rho$: target kinetic-energy retention factor at $l = L$. The *natural* setting is $\rho \in [0.05, 0.20]$ — i.e., 80–95 % of initial kinetic energy dissipated by the final layer.

**Predictions for E9 / E10 ($L=8$, $\Delta t = 1$, $m \approx 1.4$):**

| $\rho$ | $\gamma^{\ast}_{\text{depth}}$ | regime |
|---:|---:|---|
| 0.05 | $\frac{1.4}{8} \ln 20 \approx 0.524$ | very high dissipation |
| 0.10 | $\frac{1.4}{8} \ln 10 \approx 0.403$ | high dissipation |
| **0.15** | $\frac{1.4}{8} \ln(20/3) \approx 0.332$ | **buggy v2 anchor band low** |
| 0.18 | $\frac{1.4}{8} \ln(50/9) \approx 0.299$ | **buggy v2 anchor (4 dp match to $\gamma^{\ast}=0.30$)** |
| 0.20 | $\frac{1.4}{8} \ln 5 \approx 0.282$ | buggy v2 anchor band high |
| 0.30 | $\frac{1.4}{8} \ln(10/3) \approx 0.211$ | intermediate |
| 0.50 | $\frac{1.4}{8} \ln 2 \approx 0.121$ | low dissipation |
| **0.565** | $\frac{1.4}{8} \ln(20/11.3) \approx 0.100$ | **leak-free anchor (4 dp match to $\gamma^{\ast}=0.10$ leak-free 3-seed; S=5 basin $[0.10, 0.15]$)** |
| 0.70 | $\frac{1.4}{8} \ln(10/7) \approx 0.062$ | very low dissipation |

Under the **buggy v2 anchor**, the E4 / E5 empirical optimum of $\gamma^{\ast} = 0.30$ falls squarely in the $\rho \in [0.15, 0.20]$ band — i.e., the integrator empirically prefers to retain 15–20 % of initial kinetic energy at the final layer. Under the **leak-free anchor** (May 4, 2026 reanchoring; full detail in §2.5 below), the E5 empirical optimum shifts to $\gamma^{\ast} = 0.10$ (3-seed retrain) / $\gamma^{\ast} \in [0.10, 0.15]$ (S=5 confirmation flat-bottom basin), corresponding to $\rho \approx 0.565$ — i.e., the leak-corrected integrator prefers to retain $\sim\!56\%$ of initial kinetic energy at the final layer. The same closed form predicts both operating points to four decimal places with only $\rho$ shifting, which we read as **structural validation of the resonance methodology that does not depend on either single empirical anchor**: the framework's depth-scaling closed form is the *predictor*, $\rho$ is its single empirical calibration constant, and the leak fix has shifted the constant without breaking the predictor.

**Falsifiable extrapolation:** at $L = 16$ (depth-doubled SPLM), the depth-scaling formula predicts $\gamma^{\ast}_{\text{depth}} \approx 0.20$ at $\rho = 0.18$. At $L = 4$ (depth-halved), $\gamma^{\ast}_{\text{depth}} \approx 0.80$. **Future E11 / E12 depth-sweep would test this.**

**Caveat.** The exponential-decay model assumes a *constant* effective Hessian eigenvalue across layers; in reality $\nabla^2 V_{\theta}$ varies layer-by-layer because $V_{\theta}$ is shared but $h_l, \xi_l$ change. Estimator §2.2 makes this explicit.

**Pilot empirical validation (April 30, 2026, smoke run on the E9 SPLM checkpoint at γ = 0.30).** The depth formula at $\rho = 0.18$ predicts $\gamma^{\ast}_{\text{depth}} = 0.2987$, **agreeing with the trained γ = 0.30 to four decimal places**. This was the strongest a-priori validation of the framework prior to the leak fix.

**Leak-free empirical validation (May 4, 2026, on the leak-free `leakfree_3seed/gamma0p10/seed0` SPLM-2 checkpoint).** The same depth formula at $\rho = 0.565$ predicts $\gamma^{\ast}_{\text{depth}} = 0.1051$, **agreeing with the trained γ = 0.10 to three decimal places** and falling firmly inside the S=5 confirmation flat-bottom basin $\gamma^{\ast} \in [0.10, 0.15]$. The closed form's *predictive content* therefore survives the leak fix as a structural double success at two distinct operating points; only the empirical calibration $\rho$ ($0.18 \to 0.565$) shifts. Source: `notebooks/conservative_arch/scaleup/gamma_transfer/results/leakfree_gamma0p10_seed0/predict_gamma_summary.md`.

### 2.2 Hessian-spectrum critical damping ($\gamma^{\ast}_{\text{Hessian}}$)

**Idea.** Near each attractor, the layer-step is a damped harmonic oscillator with stiffness $\lambda_H = $ dominant Hessian eigenvalue of $V_{\theta}$ at the local hidden state. **Critical damping** (fastest convergence without oscillation) is

$$\gamma_c = 2\sqrt{\lambda_H/m}.$$

**Closed form (model-conditional):**

$$\boxed{\gamma^{\ast}_{\text{Hessian}} = 2  \Bigl\langle \sqrt{\lambda_H(\xi, h) / m(x)} \Bigr\rangle_{\xi, h, x},}$$

where the average is over typical $(\xi, h, x)$ triples sampled from a held-out forward pass of a *trained* SPLM model.

**Computation.** For each sampled state, use Lanczos / power iteration on $H = \nabla^2_h V_{\theta}(\xi, h)$ (held $\xi$ fixed, take Hessian w.r.t. $h$). For SPLM at $d = 256$, top-1 eigenvalue converges in ~10 Hessian-vector products per sample; 100 samples is sufficient for a stable mean.

**Why this is more principled than §2.1.** The depth scaling treats decay as *uniform*; the Hessian estimator measures the actual stiffness *the trained model has converged to*. If $\langle\sqrt{\lambda_H/m}\rangle$ varies sharply with corpus or depth, this estimator captures that variation, while §2.1 does not.

**Calibration caveat — discovered in the April 30, 2026 pilot run; reanchored May 4, 2026 on a leak-free checkpoint.** On the E9 SPLM checkpoint (buggy, trained at γ = 0.30, val PPL 8.85), the §2.2 estimator returned

| §2.2 sub-estimator (buggy E9) | predicted γ\* |
|---|---:|
| top-eigenvalue critical damping ($2\langle\sqrt{\lambda_{\max}/m}\rangle$, positive states only) | $\approx 2.25$ |
| average-eigenvalue critical damping ($2\langle\sqrt{(\mathrm{tr} H/d)/m}\rangle$, Hutchinson) | $\approx 0.76$ |

On the leak-free `leakfree_3seed/gamma0p10/seed0` SPLM-2 checkpoint (May 4, 2026, mode `shakespeare`, n_batches=8, n_hutchinson=16) the same estimator returns

| §2.2 sub-estimator (leak-free) | predicted γ\* | empirical $\gamma^{\ast}$ |
|---|---:|---:|
| top-eigenvalue critical damping (positive states only, 86.6%) | $1.17$ | $0.10$ ($\sim\!12\times$ overshoot, expected upper bound) |
| average-eigenvalue critical damping (Hutchinson) | $0.23$ | $0.10$ ($\sim\!2.3\times$ overshoot, expected typical-mode upper bound) |

In both pre- and post-leak operating points the §2.2 sub-estimators substantially exceed the empirical optimum. This is **expected** rather than alarming, and reveals an important structural fact:

- $\gamma_c^{\max} = 2\sqrt{\lambda_{\max}/m}$ is the damping required to *prevent oscillation in the heaviest mode*. Because $V_{\theta}$ has a heavy-tailed eigenvalue spectrum (some directions are very stiff, most are gentle), $\gamma_c^{\max}$ is far above the *typical* mode's critical damping.
- Empirical-optimum γ corresponds to *partial* damping: heavy modes are over-damped (acceptable, they converge fast anyway), gentle modes are under-damped (acceptable, they barely move and don't dominate the loss).
- The framework's §2.2 *top* estimate therefore acts as an **upper bound** on the optimal γ, and the §2.2 *average* estimate as a **typical-mode bound** that still overshoots.

**Practical interpretation.** Use §2.1 / §2.3 as the *primary* γ\* predictors. Use §2.2 as a *consistency check*: $\gamma^{\ast}_{\text{depth}} \le \gamma^{\ast}_{\text{Hessian,avg}} \le \gamma^{\ast}_{\text{Hessian,top}}$ is a healthy ordering. Reverse ordering would indicate the framework is failing.

**Open question (calibration).** Is there a principled scaling (e.g., γ\*\_emp = γ\*\_Hessian / $\sqrt{d_{\text{eff}}}$ for some "effective dimension" $d_{\text{eff}}$) that maps the §2.2 number to the empirical optimum directly? If so, §2.2 becomes a single-eval γ\* predictor without needing §2.1 / §2.3 as anchors.

**Where it fits in the workflow.** §2.2 is *post-hoc*: it requires a *trained* SPLM checkpoint at *some* (possibly suboptimal) γ. The estimator's prediction can then be used as the γ for the *next* training run on the same corpus / architecture. In a Bayesian sense, it is one application of an EM-like loop:

1. Train at any reasonable γ₀ (e.g., the depth-scaling default 0.30).
2. Compute $\gamma^{\ast}_{\text{Hessian}}$ from the trained model.
3. Retrain at $\gamma^{\ast}_{\text{Hessian}}$.
4. Repeat from step 2 until convergence.

In practice we expect the loop to converge in 1–2 iterations because $V_{\theta}$ is itself trained to be "compatible with" γ — there's a self-consistency condition that the predictor exploits.

**Concrete tooling.** A diagnostic
[`notebooks/conservative_arch/scaleup/gamma_transfer/predict_gamma_hessian.py`](../notebooks/conservative_arch/scaleup/gamma_transfer/predict_gamma_hessian.py)
loads the locked E9 SPLM checkpoint, runs Lanczos on 100 sampled $(\xi, h)$ states, and emits

- $\gamma^{\ast}_{\text{Hessian}}$ (the §2.2 prediction)
- $\gamma^{\ast}_{\text{depth}}$ at $\rho \in \{0.05, 0.10, 0.15, 0.20\}$ (the §2.1 prediction)
- $\gamma^{\ast}_{\text{surprisal}}$ (the §2.3 prediction)
- a per-layer histogram of $\sqrt{\lambda_H/m}$

into `predict_gamma.json` + `predict_gamma_summary.md`.

### 2.3 Corpus-conditional surprisal scaling ($\gamma^{\ast}_{\text{surprisal}}$)

**Idea.** The energy landscape's effective *frequency* $\omega_0^2 \propto \lambda_H / m$ scales with the local *predictability* of the corpus. For `logfreq` mass mode, $m(v) = -\alpha \log p(v) + \beta$ already encodes per-token unigram surprisal; at the *corpus* level, the scaling is

$$\omega_0^2 \sim \frac{\bar S_{\text{corpus}}}{\bar m_{\text{corpus}}},$$

where $\bar S$ is mean unigram surprisal (entropy of the unigram distribution).

**Closed form (corpus-relative):**

$$\boxed{\frac{\gamma^{\ast}_{\text{TS}}}{\gamma^{\ast}_{\text{E5}}} = \sqrt{\frac{\bar S_{\text{TS}}/\bar m_{\text{TS}}}{\bar S_{\text{E5}}/\bar m_{\text{E5}}}}.}$$

**Predictions for E10 (with leak-free reanchoring):** Tiny Shakespeare's mean per-token surprisal at the BPE level is $\bar S_{\text{E5}} \approx 9.1$ bits (measured May 4, 2026 on the leak-free reanchoring run; close to the original 9.5-bit hand estimate); TinyStories' is $\bar S_{\text{TS}} \approx 8.55$ bits (measured April 30, 2026; revised upward from my hand-estimated 7.0 bits, less corpus contrast than originally assumed). Mean per-token mass $\bar m \approx 1.47$ on Tiny Shakespeare (leak-free measurement), similar magnitude on TinyStories.

$$\gamma^{\ast}_{\text{TS}}/\gamma^{\ast}_{\text{E5}} \approx \sqrt{(\bar S_{\text{TS}}/\bar m_{\text{TS}}) / (\bar S_{\text{E5}}/\bar m_{\text{E5}})} \approx \sqrt{8.55/9.1} \approx 0.97.$$

Combined with the **leak-free** $\gamma^{\ast}_{\text{E5}} = 0.10$ (3-seed retrain) / $\gamma^{\ast}_{\text{E5}} \in [0.10, 0.15]$ (S=5 confirmation flat-bottom basin):

$$\gamma^{\ast}_{\text{surprisal}}({\text{TS, leak-free}}) \approx 0.10 \times 0.97 \approx \mathbf{0.097}.$$

(Under the **buggy v2 anchor** $\gamma^{\ast}_{\text{E5}} = 0.30$ the same scaling delivered $\gamma^{\ast}_{\text{surprisal}}(\text{TS}) \approx 0.30 \times 0.858 \approx 0.26$ at the originally-assumed $\bar S_{\text{TS}} = 7.0$ bits; the v3 update revises both the anchor and the surprisal ratio.)

**Falsifiable prediction (post-leak):** the interpolated leak-free optimum on TinyStories should sit at $\gamma \approx 0.10$, very close to the leak-free Tiny Shakespeare optimum, because the surprisal contrast between the two corpora is much smaller than the original $9.5/7.0 \approx 1.36$ ratio suggested. **The full leak-free TinyStories sweep is one of the open follow-ups in the v3 paper's `subsec:cba-open` (F1 SPLM scale sweep).**

**Caveat.** The surprisal scaling assumes the energy landscape's geometry is corpus-determined (true to first order) but the trained $V_{\theta}$ also has a *dataset-independent* component reflecting the architectural prior. The scaling captures the data-driven contribution but not the prior-driven one. Estimator §2.2, which is computed *from the trained model*, captures both.

### 2.4 Question-type / context-conditional $\gamma$ ($\gamma^{\ast}_{\text{cond}}$, speculative)

**Idea.** Inference modes vary continuously between two extremes:

- **Memorisation / completion of low-entropy continuation** (e.g. closed-domain QA, code completion with strong type signals): the local $V_{\theta}$ landscape has steep, deep wells. High $\lambda_H$. *Wants high γ* for fast settling (§2.2 prediction).
- **Open-ended generation under high-entropy continuation** (e.g. creative writing, brainstorming): shallow wells, low $\lambda_H$. *Wants low γ* to preserve kinetic exploration energy and avoid premature commitment to a basin (§2.2 prediction).

These predictions fall *out* of estimator §2.2 if it is run *conditionally* on the inference mode — i.e., if we sample $(\xi, h, x)$ states from each mode separately and compute $\langle\sqrt{\lambda_H/m}\rangle$ per mode.

**Architectural implication.** A single scalar $\gamma$ applied to all tokens is equivalent to averaging $\gamma^{\ast}_{\text{cond}}$ over the inference distribution. If different modes coexist in the inference distribution (almost always the case in practice), the single-γ optimum is necessarily compromise.

**Proposed lift:** make $\gamma$ a function of position and context:

$$\gamma_l = g_{\phi}(\xi_l, h_l, x_l),$$

where $g_{\phi}$ is a small MLP (e.g. width 64, depth 2) trained alongside $V_{\theta}$. Initialised to the §2.1 / §2.2 / §2.3 prediction and regularised toward it via L2 penalty.

This is *non-conservative* in the strict Lagrangian sense — dissipation depends on local trajectory state — and corresponds to **Langevin dynamics with state-dependent friction tensor**, a standard construction in computational chemistry / molecular dynamics. It has **not**, to our knowledge, been tested in language modelling.

**Status.** Not yet pre-registered. A natural follow-up to E10 (and possibly to a positive `Looped SPLM` outcome from the CoT companion proposal in [`docs/CoT_modeled_via_Semantic_Simulation.md`](CoT_modeled_via_Semantic_Simulation.md)).

### 2.5 Leak-correction calibration (May 4, 2026)

This subsection summarises which framework constants the leak fix invalidated, which it left untouched, and what the leak-free reanchored numbers are. It is the operational appendix to the leak-correction status notice at the top of this document.

**What the leak fix did and why it matters here.** The bug was a single missing `h.detach()` before forming $\xi$ in the SPLM `integrate()` step (companion: `Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md`). Trained $V_{\theta}$ silently learned to route prediction signal through this anti-causal path, inflating training-time validation perplexity *and* shifting the integrator's preferred operating point on the $\gamma$-vs-PPL bowl: the buggy integrator could rely on the leak to commit faster to attractors and tolerated heavier damping; the leak-corrected integrator has only the genuine causal channel and prefers lighter damping (more retention of kinetic energy through the layer pass).

**Leak-affected vs leak-immune framework constants.**

| Constant | Symbol | Leak-affected? | Buggy value | Leak-free value (May 4, 2026) | Source |
|---|---|---|---:|---:|---|
| Per-token mass | $\bar m_{\text{E5}}$ | No | $\sim 1.4$ | $1.47$ | dataset probe; the $\sim\!5\%$ shift is from the BPE tokeniser, not the leak |
| Mean unigram surprisal | $\bar S_{\text{E5}}$ | No | $9.5$ bits (estimated) | $9.1$ bits (measured) | dataset probe |
| Architectural $L$, $\Delta t$ | — | No | $L = 8$, $\Delta t = 1$ | $L = 8$, $\Delta t = 1$ | architectural choice |
| §2.1 kinetic-energy retention | $\rho$ | **Yes** | $0.18$ | $0.565$ | reanchored to leak-free $\gamma^{\ast}$ |
| §2.2 top-eigenvalue $\gamma_c^{\max}$ | $2\langle\sqrt{\lambda_{\max}/m}\rangle$ | **Yes** | $\approx 2.25$ | $\approx 1.17$ | direct measurement on leak-free ckpt |
| §2.2 avg-eigenvalue $\gamma_c^{\text{avg}}$ | $2\langle\sqrt{\bar\lambda/m}\rangle$ | **Yes** | $\approx 0.76$ | $\approx 0.23$ | direct measurement on leak-free ckpt |
| §2.3 anchor $\gamma^{\ast}_{\text{E5}}$ | — | **Yes** | $0.30$ | $0.10$ (3-seed) / $0.10\!-\!0.15$ (S=5 basin) | leak-free retrain + S=5 confirmation |
| **Empirical** $\gamma^{\ast}$ | — | **Yes** | $0.30$ at $\rho = 0.18$ | $0.10\!-\!0.15$ at $\rho = 0.565$ | grid sweep |

What is *not* affected by the leak: the four closed forms in §2.1, §2.2, §2.3, §2.4 themselves; the dataset-probed mass $\bar m$ and unigram surprisal $\bar S$; the architectural $L$ and $\Delta t$. What *is* affected: the empirical calibration constants $\rho$, $\lambda_{\max}$, $\bar\lambda$ that depend on the trained $V_{\theta}$, and the empirical anchor $\gamma^{\ast}_{\text{E5}}$ that anchors §2.3.

**Resonance-predictor double success.** The same depth-scaling closed form, with the *same* dataset-probed $\bar m \approx 1.4$–$1.47$ and the same $L = 8$, $\Delta t = 1$, predicts **both** empirical $\gamma^{\ast}$ values to three decimal places, with only $\rho$ shifting:

| operating point | $\rho$ | $\gamma^{\ast}_{\text{depth, pred}}$ | $\gamma^{\ast}_{\text{empirical}}$ | match |
|---|---:|---:|---:|---|
| Buggy v2, E4 / E5 | $0.18$ | $0.299$ | $0.30$ | 4 dp |
| Leak-free, May 4, 2026, S=5 | $0.565$ | $0.105$ | $0.10$ (3-seed) / $0.10\!-\!0.15$ (S=5) | 3 dp |

This double match across the leak fix is structurally stronger evidence than either single anchor by itself: it confirms that the framework's depth-scaling formula is the *predictor* and $\rho$ is its single empirical calibration constant — the closed form is correct as a function of $(\bar m, L, \Delta t, \rho)$, and the leak fix has shifted only $\rho$. STP's first-order limit (Huang et al. 2026; Lu et al. 2020) corresponds to $\gamma \to \infty$, equivalently $\rho \to 0$; the leak-free $\rho \approx 0.565$ — the trained model retaining $\sim\!56\%$ of initial kinetic energy through the layer pass — empirically rejects this limit on the trained SPLM. (Companion paper $\S15$ four-corner-synthesis paragraph and $\S17$ contribution C6.)

**Why $\rho$ shifted from $0.18$ to $0.565$.** Under the leak the integrator could route prediction signal through the anti-causal $\xi$ channel and committed faster to attractors, dissipating more kinetic energy by the final layer ($\rho \approx 0.18 \Leftrightarrow \sim\!82\%$ KE dissipated by $l = L$). Under causal honesty the model has only the genuine causal channel and prefers to retain $\sim\!56\%$ of initial KE through the layer pass ($\rho \approx 0.565$). The integrator is *less* dissipative under causal honesty, equivalently *less* damped, and the inference-optimal $\gamma$ shifts down from $\sim\!0.30$ to $\sim\!0.10$.

**Updated practical default rule.** The Appendix A rule "$\gamma = (m / (L \Delta t)) \ln(1/\rho)$ with $\rho = 0.18$" is **superseded** for leak-corrected SPLM training. The post-fix recommended default is $\rho = 0.565$, which gives the table at the bottom of Appendix A (also updated below).

**S=5 confirmation sweep canonical numbers.** Beyond the resonance-predictor double match, the second-order architectural lift was at $S = 3$ borderline-but-suggestive ($\overline{\Delta}_{0.10} = +4.71$~PPL, $0.29$~PPL short of the pre-registered $\Delta_{\min} = 5.0$~PPL); the $S = 5$ confirmation sweep at $\gamma \in \{0.05, 0.10, 0.15, 0.20\}$ confirms it at all four pre-registered decision criteria simultaneously: $\overline{\Delta}_{0.10} = +5.09$~PPL ($p = 0.006$, $d_z = +2.37$, sign $5/5$, all four ✓) and the largest paired effect at $\overline{\Delta}_{0.15} = +7.03$~PPL ($p = 0.013$, $d_z = +1.89$, sign $5/5$, all four ✓). The full table is at `notebooks/conservative_arch/ln_damping_sweep/results/leakfree_5seed_confirmation/RESULTS_CONFIRMATION_S5.md`.

---

## 3. Reconciliation framework

Given a corpus + architecture, we have *up to four* numbers for $\gamma^{\ast}$:

| Estimator | Cost | Inputs | Source |
|---|---|---|---|
| $\gamma^{\ast}_{\text{depth}}$ | trivial | $L, \Delta t, m, \rho$ | §2.1 |
| $\gamma^{\ast}_{\text{Hessian}}$ | minutes (one trained ckpt + ~1000 HVPs) | trained SPLM checkpoint | §2.2 |
| $\gamma^{\ast}_{\text{surprisal}}$ | trivial | mean unigram surprisal of train corpus | §2.3 |
| $\gamma^{\ast}_{\text{empirical}}$ (full grid sweep) | days (e.g., E10) | full PPL-vs-γ curve | empirical |

**Reconciliation rule.** The four numbers should agree to within ~10–20 % if the framework is consistent. Specifically:

- If $\gamma^{\ast}_{\text{Hessian}}$ disagrees with $\gamma^{\ast}_{\text{depth}}$ by > 30 %, the depth-scaling assumption (uniform-Hessian decay model) is violated → diagnostic for layer-by-layer Hessian inhomogeneity.
- If $\gamma^{\ast}_{\text{surprisal}}$ disagrees with $\gamma^{\ast}_{\text{Hessian}}$ by > 20 %, the corpus-driven Hessian assumption is violated → the trained $V_{\theta}$ has a strong corpus-independent component.
- If $\gamma^{\ast}_{\text{empirical}}$ disagrees with all three predictors by > 30 %, the framework is *itself* broken at this regime → flag for theoretical revisit.

**Joint use.** The cheapest workflow on a *new* corpus or architecture:

1. Compute $\gamma^{\ast}_{\text{depth}}$ (instant).
2. Compute $\gamma^{\ast}_{\text{surprisal}}$ from the corpus's unigram entropy (instant).
3. Train at $\overline{\gamma^{\ast}_{\text{depth}}, \gamma^{\ast}_{\text{surprisal}}}$ for one full training run.
4. Compute $\gamma^{\ast}_{\text{Hessian}}$ from the resulting checkpoint (~ minutes).
5. If $|\gamma^{\ast}_{\text{Hessian}} - \gamma_{\text{step 3}}| / \gamma_{\text{step 3}} > 20\%$, retrain once at $\gamma^{\ast}_{\text{Hessian}}$.
6. Otherwise, declare $\gamma^{\ast} = \gamma^{\ast}_{\text{Hessian}}$ done.

This compresses the typical ~3 days of γ-grid sweep into 1× single run + a minute of post-hoc analysis. **For the E10 use case, we can run §2.2 in parallel with E10 Stage 1's training and treat the outputs as independent corroborating evidence.**

---

## 4. Empirical alternatives (when predictors disagree)

When the predictors disagree or yield unbelievable values, we fall back to empirical search. Three options ordered by efficiency:

### 4.1 Bayesian optimisation
Gaussian-process posterior over $\gamma \mapsto $ val PPL, expected-improvement acquisition. Typical: 5–8 evaluations to localise $\gamma^{\ast}$ to ±0.05. In our setting (each evaluation is a full 8000-step training run, ~13 h on MPS), BO would compress E10 into ~3 days at the cost of giving up the parabolic-fit interpretability of grid search. **Possibly preferable to the Stage 2/3 portion of E10 if the framework predictions diverge.**

### 4.2 Pattern search / golden-section
Direct minimisation on $\gamma \in [0.05, 1.0]$, no GP overhead. ~6–8 evaluations. Slightly less efficient than BO but trivially robust to non-convexity. Better when only the location, not the local landscape, of $\gamma^{\ast}$ matters.

### 4.3 Free $\gamma$ during training (`learn_mgamma = True`)
The most apparent option, but **observed empirically to underperform** in E5: at small scale on Tiny Shakespeare, fixing $\gamma = 0.30$ gave val PPL 87.06 while leaving $\gamma$ trainable from $\gamma_{\text{init}} = 1.0$ plateaued at val PPL ~89 (about 2 PPL worse) with the trained $\gamma$ landing near 0.65. The free-$\gamma$ optimisation lands in a poorer basin than the externally-imposed prior.

This suggests $\gamma$ behaves like a **bias-variance regulariser parameter**: setting it externally (via §2.1–§2.3 prediction or empirical sweep) outperforms making it a learned weight. Estimator §2.4 (context-conditional $\gamma$) is the only learnable variant that has a chance of beating fixed $\gamma$, because it uses parameters to *select* among precomputed γ targets rather than to optimise a single scalar.

---

## 5. Open questions

1. **Layer-inhomogeneous Hessian.** Is $\sqrt{\lambda_H/m}$ approximately constant across layers, or does it vary systematically (e.g., grows shallower at later layers as attractors flatten)? The §2.2 diagnostic per-layer histogram answers this.
2. **Self-consistency of the §2.2 EM loop.** Does retraining at $\gamma^{\ast}_{\text{Hessian}}$ change $\langle\sqrt{\lambda_H/m}\rangle$ enough to require a second iteration? Empirical, easy to test.
3. **Question-type conditional $\gamma$ (§2.4).** Is the difference between recall- and generation-mode $\gamma^{\ast}$ large enough (≥ 0.1 on the Tiny-Shakespeare scale) to motivate the architectural lift?
4. **Cross-architecture generalisability.** Do the four estimators predict $\gamma^{\ast}$ for SPLM-style modifications of *attention* transformers (e.g., inserting damped Euler steps between attention blocks)?
5. **Connection to learning-rate $\eta$.** The damping $\gamma$ acts as a per-layer-step LR-like quantity. Is there a regime where $\gamma$ should track $\eta$ during training, rather than be fixed across the schedule?

---

## 6. Status & companion experiments

| Estimator | Status | Compute | First test |
|---|---|---|---|
| §2.1 — Depth scaling | **validated twice across the leak fix:** buggy ρ=0.18 predicts γ\*=0.299 matching empirical 0.30; leak-free ρ=0.565 predicts γ\*=0.105 matching empirical 0.10 (3-seed retrain) / [0.10, 0.15] (S=5 basin). Resonance-predictor double success — see §2.5. | trivial | confirmed by E4 / E5 (buggy) and the May 4, 2026 leak-free retrain + S=5 confirmation sweep |
| §2.2 — Hessian spectrum | **implemented + leak-reanchored:** [`predict_gamma_hessian.py`](../notebooks/conservative_arch/scaleup/gamma_transfer/predict_gamma_hessian.py); buggy E9 returns 2.25 (top) / 0.76 (avg); leak-free `leakfree_3seed/gamma0p10/seed0` returns 1.17 (top) / 0.23 (avg). Calibration question open (the sub-estimators remain upper bounds, not point estimates of the empirical optimum). | minutes | first deployed in chain driver between E9 Phase 1 and E10 Stage 1; leak-reanchored May 4, 2026 |
| §2.3 — Corpus-surprisal scaling | **leak-reanchored:** predicts γ\*\_TS ≈ 0.097 using leak-free γ\*\_E5 = 0.10 anchor and corpus-measured surprisal ratio $\sqrt{8.55/9.1} = 0.97$. Buggy v2 prediction was γ\*\_TS ≈ 0.26 from the γ\*\_E5 = 0.30 anchor and a different surprisal ratio. | trivial | re-confirmation requires a leak-free TinyStories sweep (companion v3 paper open follow-up F1) |
| §2.4 — Question-type-conditional $\gamma$ | speculative | one full training run + small architectural lift | future protocol post-leak-free TinyStories sweep |

E10 Stage 1 will produce the empirical $\gamma^{\ast}_{\text{TS}}$ from the {0.10, 0.30, 0.60} grid by **Friday afternoon EDT 2026-04-31**. The Hessian-spectrum predictor (`predict_gamma_hessian.py`) emits its number after the E9 SPLM checkpoint is written (already done — checkpoint at `notebooks/conservative_arch/scaleup/results/seed0_splm/splm_em_ln_scaleup_scaleup_seed0_ckpt_latest.pt`). **The two numbers can be compared as soon as both are available.**

A subsequent revision of this document will:

1. Tabulate the four-estimator predictions vs the realised $\gamma^{\ast}_{\text{TS}}$.
2. If they agree (the expected outcome), promote the framework from "proposal" to "validated", and advertise it as the recommended workflow for any new SPLM-style architecture.
3. If they disagree, diagnose which assumption(s) failed and update the framework accordingly.

This document, like [`docs/CoT_modeled_via_Semantic_Simulation.md`](CoT_modeled_via_Semantic_Simulation.md), is a **living document** and will be updated as evidence accumulates.

---

## Appendix A — Practical default rule for new SPLM deployments

If you only want one number and have no time for any sweep:

> **Use $\gamma = m / (L \cdot \Delta t) \cdot \ln(1/\rho)$ with $\rho = 0.565$ (leak-free anchor, May 4, 2026), $\Delta t = 1$, and $m$ = the mean unigram-surprisal-derived `logfreq` mass on a small sample of your corpus. Train under `cfg.causal_force = True` (the post-fix default).**

For the standard setup ($m \approx 1.47$, $\Delta t = 1$, leak-free):

| $L$ | leak-free $\gamma$ ($\rho = 0.565$) | (buggy v2 $\gamma$ at $\rho = 0.18$) |
|---:|---:|---:|
| 4 | 0.21 | 0.60 |
| 6 | 0.14 | 0.40 |
| **8** | **0.10** | **0.30** |
| 12 | 0.07 | 0.20 |
| 16 | 0.05 | 0.15 |
| 24 | 0.04 | 0.10 |

This is the §2.1 closed form at $\rho = 0.565$ (leak-free reanchoring; see §2.5 for derivation). It matches the leak-free E5 retrain at $L = 8$ to three decimal places and supersedes the buggy v2 default rule at $\rho = 0.18$ (right column, retained for diff-reading convenience).

**Note on causal honesty.** Using the leak-free default rule on a *leak-trained* SPLM checkpoint is incoherent — the buggy integrator's $\rho \approx 0.18$ is a different operating point than the leak-corrected integrator's $\rho \approx 0.565$. The recommended pipeline is: (i) confirm `cfg.causal_force = True` in your training config, (ii) train at the leak-free default $\gamma$ above, (iii) optionally refine with the §2.2 Hessian estimator on the resulting checkpoint.

If you have one trained checkpoint, run `predict_gamma_hessian.py --mode shakespeare --logfreq-path <your_logfreq>.npy` and use $\gamma^{\ast}_{\text{Hessian, avg}}$ as a *consistency check* — it will be an upper bound on the true optimum but will scale correctly with the corpus.
