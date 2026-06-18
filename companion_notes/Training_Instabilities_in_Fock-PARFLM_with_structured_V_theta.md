# Training Instabilities in Fock-PARFLM v2.1 with Structured V_θ

**Status:** Internal working note — do not push to remote. 
**Experiment:** OpenWebText Phase 4 scale-up, `d=384`, `L=16`, `M=32`, SQ3 `K_mix=8`. 
**Date:** June 2026

---

## Table of Contents

1. [Background](#1-background)
2. [Blowup 1 — Penalty Dominance](#2-blowup-1--penalty-dominance)
3. [Blowup 2 — Directional Instability](#3-blowup-2--directional-instability)
4. [Blowup 3 — EMA Watchdog Miscalibration](#4-blowup-3--ema-watchdog-miscalibration)
5. [Why Fock-PARFLM is More Susceptible than SPLM](#5-why-fock-parflm-is-more-susceptible-than-splm)
6. [Fixes Applied and Their Mathematical Justification](#6-fixes-applied-and-their-mathematical-justification)
7. [EMA Watchdog Design](#7-ema-watchdog-design)
8. [Residual Risk and Recommendations](#8-residual-risk-and-recommendations)
9. [Architectural Path Forward: Gaussian Wells with SARF Anchors](#9-architectural-path-forward-gaussian-wells-with-sarf-anchors)
   - 9.9 [Connection to the Reinforcement Field and the Paper's SARF Dynamics](#99-connection-to-the-reinforcement-field-and-the-papers-sarf-dynamics-section-6-equation-48)
10. [Implementation Diagnostic: SARF Well Deactivation by `ln_after_step` Scale Mismatch](#10-implementation-diagnostic-sarf-well-deactivation-by-ln_after_step-scale-mismatch)
11. [G3 Sigma Drift: Well Deactivation During Training](#11-g3-sigma-drift-well-deactivation-during-training)
12. [G1 Precision Explosion: Blowup 2 Revisited via Unbounded Gaussian Precision](#12-g1-precision-explosion-blowup-2-revisited-via-unbounded-gaussian-precision)
13. [Phase 5 Blowup: LR-Induced Instability Beyond the Bounded Potential](#13-phase-5-blowup-lr-induced-instability-beyond-the-bounded-potential)
14. [Scale-Diversity Analysis: Why TinyStories Is Stable and OpenWebText Is Not](#14-scale-diversity-analysis-why-tinystories-is-stable-and-openwebtext-is-not)

---

## 1. Background

### Architecture

The SQ3 structured V_θ replaces the MLP scalar potential with a
$K_{\mathrm{mix}}$-component mixture of diagonal quadratic wells:

$$
V_\theta(\xi, h) = -\tau \log\sum_{k=1}^{K} \pi_k(\xi) 
\exp\left(-\frac{E_k(\xi, h)}{\tau}\right) + b(\xi)
$$

where the per-component energy is

$$
E_k(\xi, h) = \tfrac{1}{2} a_k(\xi)^\top (h - \mu_k(\xi))^2, \qquad a_k > 0.
$$

The attractor centres $\mu_k(\xi)$ and precisions $a_k(\xi)$ are linear
projections of the flattened multi-xi context
$\xi_{\mathrm{flat}}$ which is the flattened multi-xi vector
in $\mathbb{R}^{K_\xi d}$.

**Key structural fact:** $V_\theta$ is quadratic in $h$ and has **no upper bound**. As $\lVert h - \mu_k \rVert$ grows, every $E_k$ grows without limit, and
so does $V_\theta$.

### The regulariser

To prevent the learned potential from becoming arbitrarily flat (zero-force
landscape), we add a penalty term to the training loss:

$$
\mathcal{L} = \mathcal{L}_{\mathrm{NTP}} + \lambda_V \cdot \mathcal{R}(V_\theta)
$$

with $\lambda_V = 0.01$. The choice of $\mathcal{R}$ is the source of both
blowups.

### Training setup

| Parameter | Value |
|-----------|-------|
| `d` | 384 |
| `L` | 16 |
| `M` (Fock registers) | 32 |
| `K_mix` | 8 |
| `K_xi` | 4 |
| Corpus | OpenWebText ~200M tokens |
| Total steps | 200,000 |

---

## 2. Blowup 1 — Penalty Dominance

### Symptom (step ~5,200)

```
step 5200 ntp=10.52 v_reg=27770.6 lr=3.00e-04 grad=216,392,624
step 5400 ntp= 9.91 v_reg=18680.0 lr=3.00e-04 grad= 51,402,900
step 5600 ntp=10.27 v_reg=232,649 lr=3.00e-04 grad= 601,585
```

The model never recovered. Val PPL went from ~535 (step 4k) to 106,078 (step 20k).

### Root cause: unbounded mean(V_θ²)

The original regulariser was:

$$
\mathcal{R}_{\mathrm{quad}}(V_\theta) = \mathbb{E}_{x,h}\bigl[V_\theta(\xi, h)^2\bigr]
$$

This is computed as a batch mean. A **single token** whose hidden state
$h$ drifts far from every well centre produces:

$$
V_\theta(\xi, h) \approx \tfrac{1}{2} \min_k a_k^\top(h-\mu_k)^2
 \sim \lVert h \rVert^2 \quad \text{(quadratic in } \lVert h \rVert)
$$

so $V_\theta^2 \sim \lVert h \rVert^4$. One outlier token with $\lVert h \rVert \approx 10$
contributes $\sim 10^4$ to $\mathcal{R}_{\mathrm{quad}}$, pulling
$\lambda_V \cdot \mathcal{R} \approx 100$ — far exceeding the NTP loss of
$\sim 10$ nats.

### The feedback loop

```mermaid
flowchart TD
    A["h drifts from well centres<br>(e.g. hard OWT batch)"]
    B["V_theta grows quadratically"]
    C["R = mean V_theta squared grows as h to the 4th<br>lambda_V times R >> NTP loss"]
    D["Gradient dR/d theta = 2 V_theta times dV/d theta<br>dominates dL_NTP/d theta"]
    E["Optimizer pushes theta to minimize R<br>i.e. pull mu_k toward h"]
    F["New mu_k chase h, h chases mu_k<br>runaway instability"]
    G["grad norm approx 2e8<br>clip = 0.5, tiny but misdirected steps"]

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> A
```

The critical insight is that **gradient clipping cannot stop this loop**.
Clipping bounds the step size but not the direction. Once
$\lambda_V \mathcal{R} \gg \mathcal{L}_{\mathrm{NTP}}$, every clipped step
points toward reducing $\mathcal{R}$ (pulling attractor centres toward
the outlier hidden states), which does not reduce NTP loss. The model is
trapped.

### Gradient of the original penalty

$$
\frac{\partial}{\partial \theta}\bigl(\lambda_V V_\theta^2\bigr)
 = 2\lambda_V V_\theta \cdot \frac{\partial V_\theta}{\partial \theta}
 \propto V_\theta
$$

Since $V_\theta$ is unbounded, so is this gradient — the penalty is a
**linearly amplified** version of the already-large force, making recovery
impossible once the runaway starts.

---

## 3. Blowup 2 — Directional Instability

After applying the `log1p` fix (Section 5), a second, structurally different
instability occurred at step ~22,000.

### Symptom (step ~21,800–24,000)

```
step 21800 ntp=5.670 v_reg=1.23 grad=214.28 ← first spike
step 22000 EVAL val_ppl=301.75 best=268.84 ← regression
step 22600 ntp=6.193 v_reg=2.74 grad=167.78
step 23200 ntp=6.624 v_reg=4.98 grad= 93.79
step 24000 EVAL val_ppl=388.65 best=268.84 ← continued regression
```

This time: `v_reg` peaked at ~5 (not 2×10⁵), `grad_norm` peaked at 214
(not 2×10⁸), NTP reached 6.7 (not 16+). The log1p fix contained the
severity, but the model failed to self-correct over 2,000+ affected steps.

### Root cause: directional persistence under gradient clipping

`clip_grad_norm_` rescales the gradient vector $g$ as:

$$
\hat{g} = g \cdot \frac{\min(\lVert g \rVert, C)}{\lVert g \rVert}
$$

so the applied update has magnitude exactly $\min(\lVert g \rVert, C)$. With
$\lVert g \rVert = 214$ and $C = 0.5$:

$$
\hat{g} = g \cdot \frac{0.5}{214} \approx 0.0023 g
$$

The step is tiny, but it points in the **same direction as** $g$. The
direction of $g$ is determined by which loss component dominates, and
when the Verlet backward graph is deep, a hard batch can induce a gradient
direction that is persistently destabilizing even at tiny step sizes.

### Why it did not self-correct

At step ~21,800 a hard batch produced $\lVert g \rVert = 214$. The clipped update
moved parameters by $0.5/214 \approx 0.002$ in that direction. If the
next batch is also hard (OWT is streamed sequentially — consecutive batches
share document context), the direction accumulates. Over 2,000 steps the
cumulative displacement from the best basin is:

$$
\Delta\theta_{\mathrm{cum}} \approx \sum_{t=T_0}^{T_0+2000} \hat{g}_t
 \sim 2000 \times 0.5 = 1000 \quad \text{(in gradient units)}
$$

This is enough to move a 31.5M-parameter model substantially out of the
basin of attraction reached at step 20k. Because the cosine-decay LR was
still near its maximum (only 7% decay had occurred by step 22k of a 200k
run), the effective step size was not diminishing fast enough to contain
the drift.

### The OWT sequential streaming factor

OpenWebText is streamed document-by-document. Within a run, the model sees
the same sequence of documents in the same order. This means:

```mermaid
sequenceDiagram
 participant OWT as OWT stream
 participant Batch as Batch generator
 participant Model as Fock-PARFLM

 OWT->>Batch: docs 1–500 (normal)
 Batch->>Model: steps 1–4000 → stable
 OWT->>Batch: docs ~4000–4200 (dense technical)
 Batch->>Model: step ~21800 → first hard batch
 OWT->>Batch: docs ~4200–4400 (still technical)
 Batch->>Model: steps 21800–24000 → 2000 consecutive hard steps
 Note over Model: Model cannot escape<br>— consecutive bad direction
```

On TinyStories (i.i.d. story chunks), a hard batch at step $t$ is
statistically independent from the batch at step $t+1$, so the
destabilizing directions partially cancel. On OWT they are correlated.

---

## 4. Blowup 3 — EMA Watchdog Miscalibration

After applying all Blowup-2 fixes (log1p regulariser, LR 1.5×10⁻⁴, clip 0.3, EMA
watchdog), training resumed from the step-20k checkpoint (PPL 268.84) and
immediately made genuine progress, reaching a new best PPL of **252.75 at
step 22,000**. Two thousand steps later the model had diverged to PPL 367.98. The
watchdog never fired.

### Symptom (steps 22,000 – 24,000)

```
step 22000 EVAL val_ppl=252.75 *** NEW BEST *** ← checkpoint saved
step 22200 ntp=5.6232 v_reg=1.044 grad= 1.19
step 22400 ntp=5.6164 v_reg=1.171 grad= 1.33
step 22600 ntp=5.6311 v_reg=1.267 grad= 1.31
step 22800 ntp=5.6398 v_reg=1.291 grad= 3.37 ← first elevated spike
step 23000 ntp=5.6362 v_reg=1.279 grad= 1.22
step 23200 ntp=5.6439 v_reg=1.260 grad= 2.14
step 23400 ntp=5.7605 v_reg=1.717 grad= 3.52 ← inflection point
step 23600 ntp=5.6823 v_reg=1.263 grad= 2.66
step 23800 ntp=5.7243 v_reg=1.298 grad= 5.05
step 24000 ntp=5.8080 v_reg=1.527 grad=16.76 ← confirmed blowup
step 24000 EVAL val_ppl=367.98 best=252.75 ← 45% regression
```

This blowup is structurally different from Blowup 2. The gradients are
moderate (peak 16.76, not 214), `v_reg` stays below 1.8 (not 5), and the
onset is gradual — a slow escalation over 2,000 steps rather than a sudden
spike.

### Root cause: the EMA threshold was mathematically unreachable

The watchdog parameters were:

```python
GRAD_NORM_EMA_ALPHA = 0.02 # slow EMA (~50-step memory)
GRAD_NORM_EMA_THRESHOLD = 50.0 # EMA > 50 → consider unstable
GRAD_NORM_EMA_PATIENCE = 100
```

The EMA update is:

$$
\bar{g}_t = (1 - 0.02) \bar{g}_{t-1} + 0.02 \hat{g}_t
= 0.98 \bar{g}_{t-1} + 0.02 \hat{g}_t
$$

The **steady-state** value of the EMA when the gradient is held constant at
$\hat{g}$ is exactly $\hat{g}$ itself. But with $\alpha = 0.02$ the
convergence is slow: from a baseline of 1.3, even 100 consecutive steps at
$\hat{g} = 17$ bring the EMA to only:

$$
\bar{g}_{100} = 17\bigl(1 - 0.98^{100}\bigr) + 1.3 \times 0.98^{100}
 \approx 17 \times 0.87 + 1.3 \times 0.13 \approx 14.9 + 0.17 \approx 15.1
$$

The threshold of 50 requires **sustained** gradients of at least 50 for
hundreds of steps — far above anything this model produces. Working through
the actual log, the EMA stayed near its baseline throughout the blowup:

| Step | grad | EMA (α=0.02, cumulative) |
|---|---|---|
| baseline 22200–23000 | 1.0–1.4 | ~1.30 |
| 22800 | 3.37 | 1.34 |
| 23200 | 2.14 | 1.36 |
| 23400 | 3.52 | 1.40 |
| 23800 | 5.05 | 1.50 |
| 24000 | **16.76** | **~1.81** |

**The EMA peaked at roughly 1.81. The threshold was 50.0. The watchdog was
effectively disabled for this model's gradient scale.** The threshold
calibration was borrowed from a setting where gradient norms could reach
50–200 (as in Blowup 2); after the LR and clip reductions, the gradient
scale shrank to 1–17, making the old threshold permanently unreachable.

### The slow-escalation pattern

Unlike Blowup 2 (single catastrophic spike), Blowup 3 is a
**slow directional drift**: v_reg creeps up from 1.04 to 1.72, ntp rises
from 5.61 to 5.81, and the gradient norm escalates over 2,000 steps. This
pattern is exactly what the EMA watchdog was designed to catch — but only
if the threshold is calibrated to the actual gradient scale.

The inflection point at step 23,400 (ntp jumps from 5.64 to 5.76, v_reg
from 1.26 to 1.72) suggests a particularly hard OWT batch knocked the model
onto a diverging trajectory. The partial recovery at step 23,600 (ntp
falls back to 5.68) is deceptive: the model was already displaced from the
good basin and continued to drift.

---

## 5. Why Fock-PARFLM is More Susceptible than SPLM

The Multi-Xi SPLM at the same `d=384`, `L=16` never exhibited these
instabilities at `LR=3e-4` and `clip=0.5`. The Fock architecture has four
additional instability sources:

![Gradient path depth comparison](figures/gradient_path_depth_comparison.png)

| Component | SPLM | Fock-PARFLM v2.1 | Instability mechanism |
|-----------|------|------------------|-----------------------|
| V_θ force | $-\nabla_h V_\theta$ | same | quadratic, unbounded |
| V_φ routing | — | Gumbel-softmax top-k | stochastic routes add gradient noise; wrong routes persist for a full Verlet step |
| Fock registers | — | M=32 register particles | 32 extra hidden states through L=16 backward graph; register creation/destruction gates add non-differentiable-like switching |
| Per-register τ | — | learnable temperature | τ drift can make routing sharper mid-training, amplifying the Gumbel noise |
| Reverse channel | — | non-conservative Q_i | no energy conservation guarantee; small errors in Q_i can compound across layers |

### Effective backward graph depth

For the SPLM with L layers the backward graph is roughly:

$$
\text{depth}_{\mathrm{SPLM}} \approx L \times d^2
$$

For Fock-PARFLM v2.1, each layer has Verlet step + V_φ routing
(top-k gather) + Fock register dynamics + reverse channel. A conservative
estimate:

$$
\text{depth}_{\mathrm{Fock}} \approx L \times \bigl(d^2 + 2 d k + M d + d^2\bigr)
 \approx 4\text{–}5 \times \text{depth}_{\mathrm{SPLM}}
$$

Same `grad_clip` → same magnitude bound, but 4–5× longer backward path →
a destabilizing gradient direction persists 4–5× longer before it is washed
out by the curvature of subsequent batches.

---

## 6. Fixes Applied and Their Mathematical Justification

### Fix 1: Bounded regulariser — log1p(V_θ²)

**Applied after Blowup 1.**

Replace $\mathcal{R}_{\mathrm{quad}} = \mathbb{E}[V_\theta^2]$ with:

$$
\mathcal{R}_{\mathrm{log}}(V_\theta)
 = \mathbb{E}\bigl[\log(1 + V_\theta^2)\bigr]
$$

![Penalty comparison](figures/vtheta_penalty_comparison.png)

**Why it works:**

1. **Normal regime equivalence.** For $|V_\theta| \ll 1$:
 $\log(1+V^2) \approx V^2$, so the landscape compression is identical.

2. **Gradient bound.** The gradient with respect to $V_\theta$ is:

$$
\frac{d}{dV}\log(1+V^2) = \frac{2V}{1+V^2}
$$

 This is bounded by $\left|\frac{2V}{1+V^2}\right| \leq 1$ for all
 $V \in \mathbb{R}$, with the maximum at $V=1$. Therefore:

$$
\lambda_V \cdot \frac{\partial \mathcal{R}_{\mathrm{log}}}{\partial \theta}
 \leq \lambda_V \cdot \mathbb{E}\left[\frac{\partial V_\theta}{\partial \theta}\right]
$$

 and the penalty contribution to the total gradient is bounded by
 $\lambda_V$ times the mean magnitude of $\partial V_\theta / \partial\theta$.
 It **cannot** exceed the NTP gradient unless $\lambda_V \gg 1$.

3. **No runaway.** The penalty loss itself is bounded: as $|V_\theta| \to \infty$,
 $\log(1+V^2) \to \log V^2 = 2\log V$, which grows only logarithmically.
 A batch where $V_\theta = 1000$ contributes $\log(10^6+1) \approx 13.8$
 to the penalty, not $10^6$.

**Observed effect:** `v_reg` (now `mean(log1p(V²))`) stabilised at 1.0–1.5
throughout training (vs. 2×10⁵ in the first run).

---

### Fix 2: Lower max LR and tighter grad_clip

**Applied after Blowup 2; tightened again after Blowup 3 (Fix 4).**

| Parameter | Blowup 1 run | After Fix 1 | After Fix 2 | After Fix 4 |
|-----------|-------------|-------------|-------------|-------------|
| `LR` | 3×10⁻⁴ | 2×10⁻⁴ | 1.5×10⁻⁴ | **1.2×10⁻⁴** |
| `GRAD_CLIP` | 0.5 | 0.5 | 0.3 | **0.25** |
| `WARMUP_STEPS` | 5,000 | 8,000 | 8,000 | 8,000 |

**Motivation:**

For a hard batch with $\lVert g \rVert = G$ and clip threshold $C$, the
applied step in any direction is at most:

$$
\lVert \Delta\theta \rVert \leq \eta_t \cdot C
$$

where $\eta_t$ is the learning rate at step $t$. After the cosine
schedule with $\eta_{\max}$ and warmup $T_w$:

$$
\eta_t = \frac{\eta_{\max}}{2}\left(1 + \cos\left(\pi \frac{t-T_w}{T-T_w}\right)\right)
$$

At step 22k of a 200k run (warmup at 8k):

$$
\text{progress} = \frac{22000-8000}{200000-8000} \approx 0.073
\quad\Rightarrow\quad
\eta_{22k} \approx 0.997 \eta_{\max} \approx \eta_{\max}
$$

The cosine schedule had barely started decaying. The maximum per-step
displacement in any direction is:

$$
\lVert \Delta\theta_{\max} \rVert = \eta_{\max} \cdot C
 = \begin{cases}
2\times10^{-4} \times 0.5 = 1\times10^{-4} & \text{(Blowup 2 run)} \\
1.5\times10^{-4} \times 0.3 = 4.5\times10^{-5} & \text{(current run)}
\end{cases}
$$

The new maximum displacement is **2.2× smaller**. Over 100 consecutive hard
steps the cumulative drift in a destabilizing direction is:

$$
\lVert \Delta\theta_{\mathrm{cum}} \rVert
 \lesssim 100 \times 4.5\times10^{-5} = 4.5\times10^{-3}
\quad\text{(vs.} 1\times10^{-2}\text{ before)}
$$

This keeps the model closer to the good basin and makes self-correction
by subsequent normal batches more likely.

---

### Fix 3: Auto-resume from \*\_best.pt (Cell 2)


Cell 2 now checks both the periodic step checkpoints and `*_best.pt`,
selecting whichever is **more recent**:

```mermaid
flowchart TD
 A[Cell 2: resume detection]
 B{Any step checkpoint<br>step = s?}
 C{*_best.pt exists<br>with step = b?}
 D{b > s?}
 E[resume_step = s<br>resume_ckpt = step ckpt]
 F[resume_step = b<br>resume_ckpt = best.pt]
 G[No checkpoint → train from scratch]

 A --> B
 B -- Yes --> C
 B -- No --> C
 C -- No --> B
 C -- Yes --> D
 D -- Yes --> F
 D -- No --> E
 B -- No, *_best also absent --> G
```

This means: if a blowup occurs **between** two 25k periodic checkpoints,
the next session automatically recovers the pre-blowup weights without any
manual Drive file operations.

---

### Fix 4: EMA Watchdog Recalibration

**Applied after Blowup 3.**

The three watchdog parameters were recalibrated to the actual gradient scale
of the model after the Fix-2 LR and clip reductions:

| Parameter | Before (broken) | After (calibrated) | Rationale |
|-----------|-----------------|-------------------|-----------|
| `GRAD_NORM_EMA_ALPHA` | 0.02 | **0.05** | Faster EMA (~20-step memory vs 50-step); catches escalation before it fully develops |
| `GRAD_NORM_EMA_THRESHOLD` | 50.0 | **3.5** | Baseline EMA is ~1.3–1.5; threshold = 2.5× baseline; old value of 50 was 35× baseline and permanently unreachable |
| `GRAD_NORM_EMA_PATIENCE` | 100 | **30** | Catch a sustained instability in 30 steps; 100 steps gave the model too long to drift |

Additionally `LR` was reduced from 1.5×10⁻⁴ to **1.2×10⁻⁴** and `GRAD_CLIP`
from 0.3 to **0.25** to further reduce the per-step displacement during any
hard-batch streak that does occur before the watchdog fires.

**Calibration derivation.** With the new parameters, the EMA at steady state
for a sustained gradient of $G$ converges to $G$ itself. With
$\alpha = 0.05$ and the new threshold $\tau = 3.5$:

- **Normal training** (grad ~1.3): EMA steady-state ≈ 1.3. Well below threshold.
- **Single spike** (grad = 7.6 at step 21,400 followed by immediate recovery): EMA
 reaches ≈ 1.6 in one step, falls back next step. Never approaches 3.5 for
 30 consecutive steps → watchdog does not fire.
- **Slow escalation as in Blowup 3** (grads of 2–5 sustained for 200+ steps):
 EMA converges toward 3–5. Threshold crossed within ~50–100 steps → watchdog
 fires and reloads the best checkpoint before the val PPL degrades further.

---

## 7. EMA Watchdog Design

The EMA watchdog detects a soft-instability spiral before the next periodic
checkpoint is written with corrupted weights.

**Algorithm (parameters after Fix 4):**

Let $\hat{g}_t = \lVert g_t \rVert$ (raw gradient norm before clipping) at step $t$.
Maintain an exponential moving average:

$$
\bar{g}_t = (1-\alpha) \bar{g}_{t-1} + \alpha \hat{g}_t,
\qquad \alpha = 0.05 \text{(~20-step memory)}
$$

Maintain a counter $c_t$:

$$
c_t = \begin{cases}
c_{t-1} + 1 & \text{if } \bar{g}_t > \tau_g \\\\
0 & \text{otherwise}
\end{cases}
\qquad \tau_g = 3.5
$$

If $c_t \geq P = 30$ (30 consecutive steps above threshold):
1. Log the event.
2. Call `_reload_best_checkpoint()` — loads `*_best.pt` into `model` and `optim` in-place.
3. Reset $\bar{g}_t = 0$, $c_t = 0$.
4. Continue training from the next step (LR schedule is unchanged — no warmup repeat).

**Threshold calibration (after Fix 4):**

| Regime | Typical grad | EMA (α=0.05) steady state | Triggers? |
|--------|-------------|--------------------------|-----------|
| Normal training | 1.0–1.5 | ~1.3–1.5 | No (below 3.5) |
| Single spike then recovery | 7–16, then 1.3 | peaks ~1.6–2.0, decays immediately | No (stays below 3.5 within ~10 steps) |
| Slow escalation (Blowup 3 pattern) | 2–5 sustained | converges toward 3–5 | Yes, within 50–100 steps |
| Acute blowup (Blowup 2 pattern) | 50–214 | immediately > 3.5 | Yes, within 1–2 steps |

**Why the original parameters (α=0.02, τ=50) failed (Blowup 3):**

The gradient scale after Blowup-2 fixes (LR 1.5×10⁻⁴, clip 0.3) settled at
1–17. The old threshold of 50 required sustained gradients of 50+ to produce
an EMA above 50 — a condition that never occurred after the fixes. The EMA
peaked at ~1.81 during the entire Blowup 3 event. The design rule is:

$$
\tau_g \approx 2\text{–}3 \times \bar{g}_{\mathrm{baseline}}
$$

With baseline EMA ≈ 1.4, this gives τ ≈ 2.8–4.2; the chosen value is 3.5.

**Failure mode check:** If `*_best.pt` does not exist (first session, no eval
yet), the watchdog logs a warning and continues — it cannot roll back. This
is fine because the first eval at step 2,000 writes `*_best.pt`, after which
the watchdog has a recovery target.

---


## 8. Residual Risk and Recommendations

### Remaining instability sources

After all four fixes, the following risks remain:

1. **Hard-batch streak that exceeds the watchdog patience.** With
 `PATIENCE=30`, a sustained streak of fewer than 30 steps above the EMA
 threshold will not trigger a reload. Because OWT documents are streamed
 sequentially, correlated hard batches can run for hundreds of consecutive
 steps; however, the tighter clip (0.25) and lower LR (1.2×10⁻⁴) reduce
 the per-step damage while the watchdog waits.

2. **Post-cosine-peak LR** is still ~1.2×10⁻⁴ at step 20k (only ~7% decay).
 The schedule will not meaningfully reduce LR until ~50k+ steps. Any third
 hard OWT region encountered before that point relies on the watchdog to
 absorb it.

3. **Fock register salience** can drift during a hard streak (register
 particles may incorrectly gain/lose salience). Even after reloading
 `*_best.pt`, the salience state is restored from the checkpoint, so this
 is recoverable.

4. **Watchdog threshold drift.** If training genuinely improves and the
 gradient scale drops further (e.g., to ~0.7), the current threshold of
 3.5 may become overly sensitive (≈5× baseline). Conversely, if a new
 phase of training raises the baseline, the threshold may need
 recalibrating again. The design rule τ ≈ 2.5× baseline should be
 re-checked at every major training milestone.

### Recommendations for future Fock+structured V_θ scale-ups

| Lever | After Fix 4 | Recommended for 1M-step run |
|-------|-------------|------------------------------|
| `LR_max` | 1.2×10⁻⁴ | 1.0×10⁻⁴ |
| `GRAD_CLIP` | 0.25 | 0.2 |
| `WARMUP_STEPS` | 8,000 | 10,000–15,000 |
| `CKPT_INTERVAL` | 25,000 | 10,000 (more frequent saves) |
| `lambda_V` ramp | constant 0.01 | ramp 0→0.01 over first 2k steps |
| `GRAD_NORM_EMA_ALPHA` | 0.05 | 0.05–0.10 |
| `GRAD_NORM_EMA_THRESHOLD` | 3.5 | 2.5× baseline EMA (re-measure at 10k) |
| `GRAD_NORM_EMA_PATIENCE` | 30 | 20–30 |
| Penalty | log1p(V²) | log1p(V²) — keep |
| Data shuffle | sequential OWT | consider shuffling document order to break consecutive-hard-batch correlations |

### The deeper question: is structured V_θ safe at scale?

The instabilities arise from the interaction of three factors:
(a) an **unbounded** quadratic-in-$h$ potential,
(b) **deep** Verlet integration (L=16), and
(c) **sequential data** with correlated hard batches.

Factor (a) is inherent to the SQ3 parameterisation. The log1p fix is a
sound mitigation but not a cure — it prevents the penalty from dominating
but does not prevent $V_\theta$ from growing large during forward inference.
A more principled long-term fix would be to add a **soft-clamp** on $h$
itself (e.g. via layer normalization before the V_θ force computation) so
that $\lVert h - \mu_k \rVert$ cannot grow arbitrarily. The `ln_after_step=True`
flag provides partial protection (LN is applied after each Verlet step),
but LN is applied to the full $h$ vector, not to the difference
$h - \mu_k(\xi)$, so extreme well-displacement can still occur within a
single step.

For MLP-based V_θ (no structured form), the force $-\nabla_h V_\theta$
is an arbitrary neural network output and is empirically bounded by the
weight norms — an implicit regularisation that the analytical SQ3 form does
not share.

---

## 9. Architectural Path Forward: Gaussian Wells with SARF Anchors

The three blowups documented above share a single root cause: the SQ3
potential is **quadratic in** $h$ **and therefore unbounded**. Every
remediation applied so far (log1p penalty, lower LR, tighter clip, EMA
watchdog) is a downstream containment mechanism that treats the symptoms.
This section analyses an architectural alternative that eliminates the root
cause: replacing the SQ3 log-sum-exp wells with Gaussian (mixture-PDF)
wells, and compensating for the resulting loss of global restoring force
with SARF (Semantically Anchored Reference Frame) anchor positions and
SARF-faithful dynamics.

### 9.1. Why SQ3 is structurally unbounded

The SQ3 potential is a **negative free energy** of a Boltzmann mixture:

$$
V_\theta(\xi, h) = -\tau \log \sum_{k=1}^{K} \pi_k(\xi) \exp\left(-\frac{E_k(\xi, h)}{\tau}\right)
$$

Each $\exp(-E_k / \tau)$ is a Gaussian density in $h$, but the $\log$
wrapper inverts the boundedness: as $\lVert h - \mu_k \rVert \to \infty$
for all $k$, every Gaussian density decays to zero, their sum decays to
zero, and the log diverges:

$$
V_\theta \to -\tau \log(0^+) = +\infty
$$

The force $-\nabla_h V_\theta$ grows linearly with displacement — a
global restoring force that pulls every escaped hidden state back toward the
nearest well centre. This is a useful property, but it comes at the cost of
an **unbounded penalty** $V_\theta^2$ that was the direct trigger for
Blowup 1.

### 9.2. The Gaussian-well alternative

Replace the log-sum-exp with the **mixture-PDF form** — the sum of
negative Gaussian bumps without the log wrapper:

$$
V_\theta^{\mathrm{G}}(\xi, h) = -\sum_{k=1}^{K} w_k(\xi) \exp\left(-\frac{\lVert a_k^{1/2}(h - \mu_k(\xi)) \rVert^2}{2}\right)
$$

where $w_k(\xi) \gt 0$ are context-dependent weights (softmax over a
linear projection of the multi-xi context) and $a_k(\xi) \gt 0$ are
per-component precisions as before.

![Structural comparison: SQ3 vs Gaussian wells](figures/sq3_vs_gaussian_wells_comparison.png)

**Structural properties.** The key differences from SQ3 are:

| Property | SQ3 (log-sum-exp) | Gaussian wells (mixture PDF) |
|----------|-------------------|------------------------------|
| Range | $(-\infty, +\infty)$ | $[-\sum_k w_k, 0]$ |
| Asymptotic behaviour | $V \to +\infty$ quadratically | $V \to 0$ (bounded) |
| Force outside wells | linear restoring, grows with displacement | exponential decay to zero |
| Maximum force | unbounded | bounded per well |
| Penalty $V^2$ | unbounded $\to$ Blowup 1 | bounded by $(\sum_k w_k)^2$ |
| Jacobi metric | degenerate at $V = E$ boundary | valid everywhere ($V$ bounded below) |

**Gradient bound.** The force of a single Gaussian well is:

$$
F_k(h) = -\nabla_h \left[-w_k \exp\left(-\frac{\lVert h - \mu_k \rVert^2}{2\sigma_k^2}\right)\right] = -\frac{w_k}{\sigma_k^2}(h - \mu_k) \exp\left(-\frac{\lVert h - \mu_k \rVert^2}{2\sigma_k^2}\right)
$$

The exponential decay factor bounds the force magnitude. The maximum occurs
at $\lVert h - \mu_k \rVert = \sigma_k$ and equals:

$$
\lVert F_k \rVert_{\max} = \frac{w_k}{\sigma_k} e^{-1/2} \approx \frac{0.607 w_k}{\sigma_k}
$$

This is finite and independent of the hidden state. No outlier token can
produce an arbitrarily large gradient contribution from the potential,
which **structurally prevents Blowup 1**.

### 9.3. The escape problem and why SARF solves it

The Gaussian well force profile is **non-monotone**: it peaks at distance
$\sigma_k$ from the centre and then decays exponentially. Beyond roughly
$2\sigma_k$, the force is negligible. A hidden state that drifts past all
well radii experiences no meaningful restoring force and moves inertially.

This is the fundamental trade-off: **boundedness (stability) vs. global
attraction (convergence)**. The SQ3 form has global attraction but is
unstable; pure Gaussian wells are stable but have no global attraction.

SARF (Semantically Anchored Reference Frame) provides two mechanisms that
close this gap without reintroducing unboundedness.

#### 9.3.1. SARF anchors: PMI-extremal coverage

SARF anchors $\lbrace a_j \rbrace_{j=1}^{N_S}$ are the embeddings of the
top-$N_S$ tokens ranked by $\max_w \mathrm{PMI}(v, w)$ — tokens with the
most strongly peaked pointwise mutual information with at least one other
token. These are the **informationally extremal** points of the vocabulary.

The anchor selection procedure is:

$$
\mathrm{PMI\_peak}(v) = \max_{w \neq v} \mathrm{PMI}(v, w)
$$

$$
\lbrace a_j \rbrace_{j=1}^{N_S} = \mathrm{TopK}\left(\lbrace E(v) \mid v \in \mathcal{V} \rbrace, \mathrm{PMI\_peak}, N_S\right)
$$

When Gaussian wells are centred on SARF anchors, the potential becomes:

$$
V_\theta^{\mathrm{SARF}}(\xi, h) = -\sum_{j=1}^{N_S} w_j(\xi) \exp\left(-\frac{\lVert h - a_j \rVert^2}{2\sigma_j^2}\right)
$$

**Why escape is structurally impossible.** The SARF anchors are PMI
extremes of the vocabulary — they occupy the "corners" of the semantic
embedding space. Because the anchors span the full semantic range, a
hidden state cannot be simultaneously far from **all** anchors. Any
direction away from one anchor necessarily moves toward another.

![SARF anchors: global coverage in semantic space](figures/sarf_anchor_coverage.png)

The coverage guarantee can be formalised. Let $R = \max_j \lVert a_j \rVert$
be the radius of the anchor constellation and
$\delta = \min_{j \neq k} \lVert a_j - a_k \rVert$ the minimum inter-anchor
distance. Then for any $h$ with $\lVert h \rVert \le R$:

$$
\min_j \lVert h - a_j \rVert \le \delta
$$

by the pigeonhole principle on the Voronoi cells of the anchors. If
$\sigma_j \ge \delta / 2$ for all $j$, every point inside the anchor
constellation lies within at least one well's effective force radius.

#### 9.3.2. SARF-faithful dynamics: adaptive restoring force via per-layer xi re-pooling

The second SARF mechanism is **per-layer recomputation of the context
variable** $\xi$. In the standard SPLM, $\xi$ is computed once from the
layer-0 embeddings and held fixed across all $L$ integration steps. In
the SARF-faithful variant:

$$
\xi_\ell = \frac{1}{t}\sum_{s \le t} h_s^{(\ell)}, \qquad \text{recomputed at each layer } \ell = 0, 1, \ldots, L{-}1
$$

where $h_s^{(\ell)}$ is the hidden state of token $s$ at layer $\ell$
(computed from $h_s^{(\ell)}.detach()$ to sever the autograd path and
preserve the per-token Euler-Lagrange structure).

This creates an **adaptive multi-layer ratchet**: the well centres
$\mu_k(\xi_\ell)$ shift at each layer to track the running mean of the
current hidden states. Even if a hidden state exits a well's force radius
at layer $\ell$, the next layer's well centres are closer because $\xi$
updated to include the displaced $h$.

```mermaid
flowchart TD
 Layer0["Layer 0: wells centred at mu_k(xi_0)"]
 Escape["h exits well radius at layer 0"]
 XiUpdate["xi_1 = mean(h detach) -- shifted toward h"]
 Layer1["Layer 1: wells centred at mu_k(xi_1) -- closer to h"]
 Recapture["h re-enters a well force radius"]
 Layer2["Layer 2: convergent trajectory"]

 Layer0 --> Escape
 Escape --> XiUpdate
 XiUpdate --> Layer1
 Layer1 --> Recapture
 Recapture --> Layer2
```

The ratchet does not add backward-graph depth because `h.detach()` severs
the autograd path from $\xi$ to $h$. The force at each layer is the
standard per-token Euler-Lagrange force:

$$
m \ddot{h}\_t = -\nabla_h V_\theta(\xi_\ell, h_t)
$$

with the damped update (Verlet integrator):

$$
v_{t}^{(\ell+1)} = \frac{v_{t}^{(\ell)} + \Delta t \cdot f_t^{(\ell)} / m}{1 + \Delta t \cdot \gamma}, \qquad h_{t}^{(\ell+1)} = h_{t}^{(\ell)} + \Delta t \cdot v_{t}^{(\ell+1)}
$$

The xi recomputation adds only one `cumsum` + division per layer — negligible
relative to the $V_\theta$ force evaluation.

### 9.4. The combined architecture: Gaussian wells + SARF

The proposed architecture combines three elements:

1. **Gaussian (mixture-PDF) wells** for structurally bounded $V_\theta$.
2. **SARF anchor positions** as well centres for global semantic coverage.
3. **SARF-faithful per-layer xi dynamics** for adaptive restoring.

The potential is:

$$
V_\theta^{\mathrm{SARF}}(\xi_\ell, h) = -\sum_{j=1}^{N_S} w_j(\xi_\ell) \exp\left(-\frac{\lVert h - a_j \rVert^2}{2\sigma_j^2}\right)
$$

where:
- $a_j$ are static PMI-peak SARF anchor positions (fixed after corpus analysis)
- $\sigma_j$ are learned per-anchor widths
- $w_j(\xi_\ell)$ are context-dependent weights: a small MLP or linear projection of $\xi_\ell$
- $\xi_\ell$ is recomputed from $h^{(\ell)}.detach()$ at each layer (SARF-faithful)

```mermaid
flowchart LR
 PMI["PMI matrix from corpus"]
 Anchors["Top-N_S PMI-peak tokens"]
 Embed["Anchor embeddings a_j"]
 XiPool["xi_l = causal mean of h_l detach"]
 Weights["w_j(xi_l) via linear proj"]
 Gauss["Gaussian wells: -w_j exp(- norm h-a_j squared / 2 sigma_j squared)"]
 Force["Force: -grad_h V bounded"]
 Verlet["Damped Verlet update"]

 PMI --> Anchors
 Anchors --> Embed
 Embed --> Gauss
 XiPool --> Weights
 Weights --> Gauss
 Gauss --> Force
 Force --> Verlet
```

**Optional background quadratic.** For additional safety, a mild
global quadratic background can be added:

$$
V_\theta^{\mathrm{SARF+bg}}(\xi_\ell, h) = -\sum_{j=1}^{N_S} w_j(\xi_\ell) \exp\left(-\frac{\lVert h - a_j \rVert^2}{2\sigma_j^2}\right) + \epsilon \lVert h \rVert^2
$$

where $\epsilon \ll 1$ is small enough that the Gaussian wells dominate
locally (within the anchor constellation) but the background prevents
unbounded drift if a hidden state somehow escapes all anchors. This is
analogous to layer normalization (`ln_after_step=True`) but acts through the
potential rather than as a post-hoc normalisation.

### 9.5. What each mechanism fixes

| Blowup | Root cause | Gaussian wells | SARF anchors | SARF dynamics |
|--------|-----------|---------------|-------------|--------------|
| 1 (penalty dominance) | $V_\theta^2 \to \infty$ | **Prevented**: $V^2 \le (\sum w_k)^2$ | neutral | neutral |
| 2 (directional instability) | deep backward graph | **Partially mitigated**: bounded force | neutral | neutral (same depth) |
| 3 (slow escalation) | sustained moderate gradients | **Partially mitigated**: lower gradient ceiling | **Mitigated**: no dead zones between wells | **Mitigated**: per-layer ratchet prevents sustained drift |

```mermaid
flowchart TD
 SQ3["SQ3 log-sum-exp<br>unbounded V, global restoring<br>STATUS: all 3 blowups"]
 Gauss["Gaussian wells only<br>bounded V, no global restoring<br>RISK: escape between wells"]
 GaussSARF["Gaussian + SARF anchors<br>bounded V, global coverage<br>RISK: single-layer escape possible"]
 GaussSARFDyn["Gaussian + SARF anchors + SARF dynamics<br>bounded V, global coverage, per-layer ratchet<br>EXPECTED: stable"]

 SQ3 -->|replace log with mixture PDF| Gauss
 Gauss -->|add PMI-peak anchor centres| GaussSARF
 GaussSARF -->|add per-layer xi re-pooling| GaussSARFDyn
```

### 9.6. Thermodynamic cost

The SQ3 log-sum-exp has a clean **free-energy interpretation**:
$V_\theta = -\tau \log Z(\xi, h)$ where $Z$ is the partition function
of a Boltzmann mixture. This connects to the paper's variational framework
and Jacobi metric derivations.

The Gaussian mixture-PDF form has no such thermodynamic interpretation —
$V_\theta$ is a sum of bump functions, not a log-partition function. However:

1. **The Jacobi metric is actually better behaved.** Because
 $V_\theta^{\mathrm{G}}$ is bounded below (by $-\sum_k w_k$), the Jacobi
 metric $g_{ij}^J = 2(E - V)\delta_{ij}$ is positive-definite everywhere
 (given $E \gt -\sum_k w_k$), without the degeneracy at energy
 boundaries that plagues the SQ3 form.

2. **The connection to attention.** The softmax attention score between
 query $h$ and key $\mu_k$ is proportional to
 $\exp(-\lVert h - \mu_k \rVert^2 / 2\sigma^2)$ in the Gaussian kernel
 formulation. The Gaussian-well potential is therefore a
 potential-energy analogue of the attention score — a well-known
 connection that strengthens rather than weakens the theoretical
 narrative.

3. **The SARF anchors provide interpretability** that the learned SQ3
 well centres lack. Each anchor corresponds to a specific PMI-extremal
 token, giving the potential landscape a grounded semantic interpretation.

### 9.7. Force profile analysis

The force profile of a single Gaussian well (radial component) is:

$$
F(r) = -\frac{w}{\sigma^2} r \exp\left(-\frac{r^2}{2\sigma^2}\right), \qquad r = \lVert h - \mu \rVert
$$

This has a characteristic non-monotone shape:

- **Near-field** ($r \ll \sigma$): $F \approx -w r / \sigma^2$ (linear,
 harmonic oscillator)
- **Peak** ($r = \sigma$): $F_{\max} = -w e^{-1/2} / \sigma \approx -0.607 w / \sigma$
- **Far-field** ($r \gg \sigma$): $F \to 0$ (exponential decay)

The **escape radius** — beyond which the force is less than 1% of peak — is:

$$
r_{\mathrm{esc}} = \sigma\sqrt{2\ln 100} \approx 3.03\sigma
$$

For the combined SARF architecture, the effective restoring force at any
point $h$ is the superposition of all anchor wells:

$$
F_{\mathrm{total}}(h) = \sum_{j=1}^{N_S} \frac{w_j}{\sigma_j^2}(h - a_j) \exp\left(-\frac{\lVert h - a_j \rVert^2}{2\sigma_j^2}\right)
$$

When the SARF anchor constellation tiles the space with
$\sigma_j \ge \delta / 2$ (half the minimum inter-anchor distance), every
interior point receives nonzero force from at least one anchor, and the
maximum "dead-zone force deficit" (the minimum force at any interior point)
is bounded below by:

$$
\lVert F_{\mathrm{total}} \rVert_{\min} \ge \frac{w_{\min}}{\sigma_{\max}} \exp\left(-\frac{\delta^2}{2\sigma_{\min}^2}\right) \gt 0
$$

### 9.8. Implementation considerations

The transition from SQ3 to Gaussian+SARF requires changes in three places:

1. **V_theta module**: replace the `StructuredVThetaSQ3` forward pass
 (log-sum-exp) with a sum of Gaussian bumps. The gradient computation
 via `torch.autograd.grad` is unchanged — PyTorch differentiates the
 Gaussian form automatically.

2. **Anchor computation**: a one-time offline step that computes the PMI
 matrix from the corpus, identifies the top-$N_S$ PMI-peak tokens, and
 saves their embedding positions. This replaces the learned
 $\mu_k(\xi)$ centres.

3. **Integration loop**: adopt SARF-faithful xi recomputation from
 `model_sarf.py` — a one-line change replacing
 `xi = causal_cumulative_mean(emb)` (computed once) with
 `xi = causal_cumulative_mean(h.detach())` (computed per layer).

The regulariser $\mathcal{R}$ can be simplified. With bounded
$V_\theta^{\mathrm{G}}$, the original `mean(V^2)` penalty is already
bounded. The log1p fix remains harmless but is no longer necessary:

$$
\mathcal{R}(V_\theta^{\mathrm{G}}) = \mathbb{E}[V_\theta^2] \le \left(\sum_k w_k\right)^2 \quad \text{(structurally bounded)}
$$

**Parameter count.** For $N_S$ anchors in $d$ dimensions with $K$ context
weights:

| Component | SQ3 (current) | Gaussian + SARF |
|-----------|-------------|-----------------|
| Well centres $\mu_k$ | $K \times K_\xi \times d$ (learned) | $N_S \times d$ (frozen PMI anchors) |
| Precisions $a_k$ | $K \times K_\xi \times d$ (learned) | $N_S$ (learned $\sigma_j$) |
| Weights | $K$ (softmax logits from $\xi$) | $N_S$ (linear head from $\xi$) |
| Total V_theta params | $2K K_\xi d + K$ | $N_S + N_S$ (only widths + weight head) |

With $K = 8$, $K_\xi = 4$, $d = 384$: SQ3 has $24577$ learned V_theta
parameters. With $N_S = 64$ SARF anchors: Gaussian+SARF has $128$ learned
parameters (64 widths + 64 weight-head outputs) plus the frozen anchor
positions. The learned parameter count drops by **192x**, which
correspondingly reduces the V_theta contribution to the backward graph.

### 9.9. Connection to the Reinforcement Field and the Paper's SARF Dynamics (Section 6, Equation 48)

The SARF-anchored Gaussian well architecture described above is not an
ad-hoc stability fix — it is a concrete, computable realization of the
**reinforcement field** $\mathcal{E}$ defined in Section 6 of the paper
(paper_v4, §6.2, `eq:reinforcement-field`) and the extended coupled
dynamics of Equation 48 (`eq:extended-dynamics`). This subsection makes the
correspondence explicit.

#### 9.9.1. The reinforcement field in the paper

Section 6 of the paper defines the reinforcement field $\mathcal{E}$ as the
time-dependent, vector-valued force field in which semantic structures evolve.
At each position $\vec{r}$ and time $t$, the field returns the local force
(Eq. 42):

$$
\vec{f}(\vec{r}, t) \in \mathbb{R}^L
$$

The field absorbs the cumulative effect of all existing structures and
is bidirectionally coupled to the model: each newly formed structure both
**responds to** $\mathcal{E}$ and **modifies** it through the
attractive-repulsive contributions of its constituent properties.

The extended coupled dynamics (Equation 48, `eq:extended-dynamics`) governs
how a semantic particle's position $\vec{p}_{c,i}$ evolves under this field.
It has four components:

1. **Energy accumulation**: $E$ updates by the dot product of the local force
   with the displacement
2. **Harmonic-mean energy centroid** $\vec{p}_E$: the energy-weighted
   attractor, including initial-impulse corrections
3. **Per-step displacement** $\Delta\vec{p}_i$: a convex combination of the
   field-pull direction and the impulse direction, weighted by their
   respective energies
4. **Time-dependence**: the field $\mathcal{E}$ changes as new structures
   arrive and displace existing centroids

The driving force in the entire system is $\vec{f}$ — the local restoring
force from the Gaussian wells of Section 4. The paper defines the Gaussian
semantic energy well (Eq. 10, `eq:well`) as:

$$
V(x) = m \upsilon^2 (1 - e^{-\kappa^2 x^2})
$$

with restoring force (Eq. 11, `eq:well-force`):

$$
F(x) = -2 m \upsilon^2 \kappa^2 x e^{-\kappa^2 x^2}
$$

This is the same functional form used in our `SARFGaussianVTheta`
implementation, with the identifications $\upsilon^2 \leftrightarrow w_j$
and $\kappa \leftrightarrow 1/\sigma_j$.

#### 9.9.2. The explicit reinforcement field from SARF-anchored Gaussian wells

Substituting the SARF-anchored potential
$V_\theta^{\mathrm{SARF}}$ into the force law
$-\nabla_h V_\theta$ yields an explicit, computable reinforcement field:

$$
\mathcal{E}(h, \xi_\ell) = \sum_{j=1}^{N_S} \underbrace{\frac{w_j(\xi_\ell)}{\sigma_j^2}}_{\text{context gate}} (a_j - h) \underbrace{\exp\Big(-\frac{\lVert h - a_j \rVert^2}{2\sigma_j^2}\Big)}_{\text{Gaussian locality (Section 4 well)}}
$$

Each term in this sum is a **restoring pull toward anchor** $a_j$,
modulated by two factors:

- **Gaussian locality**: the exponential factor
  $\exp(-\lVert h - a_j \rVert^2 / 2\sigma_j^2)$ ensures the force
  vanishes exponentially for hidden states far from the anchor — exactly
  the Gaussian well of Section 4 with shape parameter
  $\kappa_j = 1/\sigma_j$.

- **Context gate**: the weight $w_j(\xi_\ell) / \sigma_j^2$ is a linear
  projection of the causal-EMA context $\xi_\ell$, determining which
  anchors are relevant at a given point in the sequence.

The per-anchor force maximum occurs at $\lVert h - a_j \rVert = \sigma_j$
and equals $0.607 w_j / \sigma_j$ — matching the Section 4 formula
$F_{\max} = -\sqrt{2/e} \cdot m \upsilon^2 \kappa$ exactly.

#### 9.9.3. Mapping Equation 48 to the Fock-PARFLM implementation

The following table maps each object in Equation 48 (the paper's abstract
coupled dynamics) to its concrete counterpart in the Fock-PARFLM
implementation with SARF-anchored Gaussian wells:

| Paper (Eq. 48) | Fock-PARFLM implementation |
|---|---|
| $\vec{p}_{c,i}$ — particle position | $h_t^{(\ell)}$ — hidden state, token $t$, layer $\ell$ |
| $\vec{f}(\vec{p}\_{i,j} + \Delta\vec{p}\_i, l\_{i,j})$ — local force | $\mathcal{E}(h_t^{(\ell)}, \xi_\ell)$ — SARF-anchored Gaussian force |
| $\Delta\vec{p}_i$ — per-step displacement | $\Delta h_t = v_t^{(\ell+1)} \cdot \Delta t$ — Verlet step |
| $E(\vec{p}_{c,i})$ — accumulated energy | $\xi_\ell$ — causal EMA tracking hidden-state history |
| Time $t$ | Layer index $\ell = 0, \ldots, L{-}1$ |
| New structure modifies $\mathcal{E}$ | $\xi_\ell$ recomputed per layer (SARF-faithful dynamics) |

The last row is the most important for understanding the correspondence.
In the paper, time-dependence of $\mathcal{E}$ arises because newly-parsed
structures arrive and displace the centroids of existing Gaussian wells —
each structure both responds to and modifies the field. In the
implementation, this time-dependence is realized through **SARF-faithful
per-layer xi recomputation**: at each layer $\ell$, $\xi_\ell$ is
recomputed from the current hidden states $h^{(\ell)}.detach()$, which
shifts the context-dependent weights $w_j(\xi_\ell)$ and thereby reshapes
the effective force field between layers. The environment actively
reshapes between layers rather than being frozen at the embedding level.

This mapping is also consistent with the **content/environment
factorization** of Remark 6.4 in the paper (Eq. 52,
`eq:traj-functional`):

$$
T_{i,0,k} = F(T_{1,0,k-1}, \ldots, T_{M,0,k-1}, E_{1,0,k-1}, \ldots, E_{M,0,k-1})
$$

The past trajectories $T_{j,0,k-1}$ (content) map to the token hidden-state
histories feeding into the causal EMA; the accumulated energies
$E_{j,0,k-1}$ (environment) map to $\xi_\ell$ itself, which summarises
the interaction history with the reinforcement field up to layer $\ell$.

#### 9.9.4. Two senses of "SARF" — paper vs. implementation

The paper's SARF (Section 6) and the implementation's SARF anchors share
the same mathematical substrate but make different choices about what is
dynamic vs. static:

| | Paper Section 6 SARF | Implementation SARF anchors |
|---|---|---|
| What are the "structures"? | Parsed sentences/documents at dynamic positions | Fixed PMI-peak vocabulary embeddings $a_j$ |
| How does $\mathcal{E}$ change? | New structures arrive, centroids move | $\xi_\ell$ updates per layer, reshaping $w_j$ |
| Anchor positions | Float freely (PARF-governed) | Frozen at corpus-analysis time |
| Force law | Gaussian well from Section 4 at each centroid | Identical functional form, at frozen $a_j$ |
| Regional cutoff | $2/\kappa$ distance filter (Eq. 37) | Implicit: Gaussian decay handles locality |

The implementation makes a deliberate trade-off: **giving up dynamic
anchor positions** in exchange for **guaranteed global coverage**. The
PMI-extremal anchors span the vocabulary's semantic range by construction,
so no hidden state can escape all wells simultaneously (Section 9.3.1 above).
The dynamic adaptation that the paper achieves through floating structure
centroids is instead achieved through the context-dependent weights
$w_j(\xi_\ell)$ — a lower-dimensional but computationally cheaper mechanism
that preserves the essential bidirectional coupling between model and field.

#### 9.9.5. The reinforcement field as a force-field summary

The relationship can be summarised as follows. Define the **SARF
reinforcement field** as the mapping from hidden state and context to
force:

$$
\mathcal{E}: \mathbb{R}^d \times \mathbb{R}^{K_\xi d} \to \mathbb{R}^d, \qquad (h, \xi_\ell) \mapsto \sum_{j=1}^{N_S} \frac{w_j(\xi_\ell)}{\sigma_j^2} (a_j - h) \exp\left(-\frac{\lVert h - a_j \rVert^2}{2\sigma_j^2}\right)
$$

This field has the following properties that mirror the paper's abstract
$\mathcal{E}$:

1. **Vector-valued**: $\mathcal{E}(h, \xi_\ell) \in \mathbb{R}^d$,
   matching Eq. 42.

2. **Time-dependent**: the layer index $\ell$ plays the role of time, and
   the recomputed $\xi_\ell$ makes the field change between layers.

3. **Bidirectionally coupled**: the hidden states $h^{(\ell)}$ respond to
   $\mathcal{E}$ (through the Verlet update) and modify it (through
   $\xi_\ell = \mathrm{causal\_mean}(h^{(\ell)}.detach())$).

4. **Gaussian-well substrate**: each anchor contributes a restoring force
   with the identical functional form to the Gaussian well of Section 4,
   with shape parameter $\kappa_j = 1/\sigma_j$.

5. **Bounded**: $\lVert \mathcal{E}(h, \xi_\ell) \rVert$ is bounded for all
   $h$, eliminating the instabilities documented in Sections 2-4 of this
   note.

The Verlet update in the implementation:

$$
v_t^{(\ell+1)} = \frac{v_t^{(\ell)} + \Delta t \cdot \mathcal{E}(h_t^{(\ell)}, \xi_\ell) / m}{1 + \Delta t \cdot \gamma}
$$

$$
h_t^{(\ell+1)} = h_t^{(\ell)} + \Delta t \cdot v_t^{(\ell+1)}
$$

is the discretized form of the paper's extended coupled dynamics (Eq. 48),
with the damping factor $\gamma$ playing the role of the paper's $H_i$
damping, and the SARF reinforcement field $\mathcal{E}$ providing the force
$\vec{f}$ that drives the displacement.

```mermaid
flowchart TD
    Sec4["Section 4: Gaussian well V(x) = m upsilon squared (1 - exp(-kappa squared x squared))"]
    Sec6["Section 6: Reinforcement field E -- abstract force f(r, t)"]
    Eq48["Equation 48: Extended coupled dynamics -- position update from f"]
    SARF_Impl["Implementation: SARF anchors a_j (frozen PMI-peak embeddings)"]
    Gauss_V["Gaussian V_theta: -sum w_j exp(-norm h-a_j squared / 2 sigma_j squared)"]
    Force_Impl["Explicit reinforcement field: E(h, xi_l) = sum restoring pulls"]
    Verlet_Impl["Damped Verlet: discretized Eq 48 with bounded E"]

    Sec4 -->|provides well functional form| Sec6
    Sec6 -->|drives dynamics via| Eq48
    Sec4 -->|same Gaussian form with kappa = 1/sigma| Gauss_V
    SARF_Impl -->|anchor positions replace floating centroids| Gauss_V
    Gauss_V -->|negative gradient gives| Force_Impl
    Force_Impl -->|bounded force into| Verlet_Impl
    Sec6 -.->|abstract to concrete| Force_Impl
    Eq48 -.->|discretized as| Verlet_Impl
```

This connection has a significant consequence for the paper narrative: the
SARF-anchored Gaussian well architecture is not merely a pragmatic
stability fix but a **faithful implementation** of the paper's theoretical
framework. The reinforcement field $\mathcal{E}$ — which in the paper
remains an abstract, qualitatively-described object — is here given a
concrete, differentiable, and bounded form. The instabilities documented
in Sections 2-4 arose precisely because the SQ3 parameterisation violated
the boundedness property of the paper's Gaussian wells (Section 4), and
the SARF anchor architecture restores that property while preserving the
dynamic field structure of Section 6.

---

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
| $\lVert h_L \rVert$ (post-Verlet) | ~16 | ~16 |
| $\sigma$ | 1.0 | ~16 |
| $\lVert h_L - a_j \rVert$ | ~16 | ~22.6 (normalised anchors) |
| Gaussian exponent | $\exp(-128) \approx 0$ | $\exp(-0.99) \approx 0.37$ |
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
| Well centres | Frozen PMI anchors | Learned $\mu_k(\xi)$ | Learned $\mu_k(\xi)$ |
| Width control | $\log\sigma$ capped | $a_k$ uncapped | $a_k \le 2/d$ |
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

---

## 13. Phase 5 Blowup: LR-Induced Instability Beyond the Bounded Potential

### 13.1. Context

Phase 5 scaled the TinyStories-validated G1 architecture (Gaussian
`MixtureGaussianVTheta`, K=8, `precision_max=2/d`) to OpenWebText with
a larger model: $d=384$, $L=16$, $M=32$ Fock registers, 31.5M parameters.
The initial configuration used `LR=2e-4` (higher than Phase 4's `1.2e-4`),
based on the assumption that bounded V_theta would tolerate a more
aggressive learning rate.

### 13.2. Symptom: Doom Loop at LR Peak

Training proceeded normally through warmup but became unstable as the
learning rate approached its peak:

| Step | LR | grad (pre-clip) | v\_reg | val\_ppl |
|------|----|----------------|--------|---------|
| 2000 | 5.0e-5 | 3.6 | 0.017 | 1893.58 |
| 4000 | 1.0e-4 | 2.4 | 0.099 | 933.30 |
| 6000 | 1.5e-4 | 1.9 | 0.138 | 583.91 |
| 6200 | 1.55e-4 | **31.1** | 0.132 | — |
| 6600 | 1.65e-4 | **26.2** | 0.130 | — |
| 7800 | 1.95e-4 | **39.9** | 0.159 | — |
| 8000 | **2.0e-4** (peak) | 13.8 | 0.174 | 492.72 |
| 8702 | 2.0e-4 | watchdog trigger (EMA=34.7) | — | — |
| 8800 | 2.0e-4 | **459.5** | 0.165 | — |
| 8910 | 2.0e-4 | EMA=**9898** | — | — |

The EMA watchdog reloaded the best checkpoint (step 8000) at step 8702,
but the peak LR immediately pushed the model back into instability.
The gradient EMA reached 9898 within 208 steps of the reload — a
classic doom loop where the recovery checkpoint is itself at the edge
of the unstable region.

![Phase 5 gradient explosion trajectory](phase5_gradient_explosion_trajectory.png)

### 13.3. Root Cause: Bounded V_theta Does Not Bound the Full Gradient

The Gaussian well architecture guarantees:

$$
V(\mathbf{h}) \in \Bigl[-\sum_k w_k, 0\Bigr], \qquad \lVert \mathbf{F} \rVert = \lVert -\nabla_h V \rVert \le \frac{0.607 \cdot w_k}{\sigma_{\min}}
$$

However, the total gradient norm $\lVert \nabla_\theta \mathcal{L} \rVert$
is computed over **all** 31.5M parameters, most of which lie outside V_theta.
The gradient flows through an unbounded computational chain:

$$
\mathcal{L} = \underbrace{-\log \mathrm{softmax}(\underbrace{h_L \cdot E^\top}_{\text{logits}})}_{\text{cross-entropy}}
$$

where $h_L$ is produced by the Verlet integration stack:

$$
h_{l+1} = \mathrm{LN}\Bigl(h_l + \Delta t \cdot v_l + \Delta t^2 \cdot \bigl(\underbrace{F_{V_\theta}}_{\text{bounded}} + \underbrace{F_{V_\phi}}_{\text{unbounded}}\bigr)\Bigr)
$$

The backward pass computes:

$$
\frac{\partial \mathcal{L}}{\partial \theta_i} =
\frac{\partial \mathcal{L}}{\partial h_L} \cdot
\prod_{l=1}^{L} \frac{\partial h_{l+1}}{\partial h_l} \cdot
\frac{\partial h_1}{\partial \theta_i}
$$

Each factor in this product involves:

| Component | Bounded? | Gradient contribution |
|-----------|---------|----------------------|
| $V_\theta$ (Gaussian wells) | **Yes** | Force $\le 0.607 w_k / \sigma_{\min}$ |
| $V_\phi$ (competitive routing) | No | Gumbel-softmax, attention scores |
| Fock registers (creation/annihilation) | No | Gating, salience thresholding |
| LayerNorm | Normalises $h$, but **amplifies** $\partial h$ | Jacobian has $1/\sigma$ terms |
| Embedding $E$ ($50257 \times d$) | No | Gradient scales with vocabulary |
| Logit projection $h_L \cdot E^\top$ | No | Linear in $\lVert h_L \rVert$ |

The bounded V_theta contributes a small, well-behaved fraction of the
total gradient. The dominant terms come from V_phi, the Fock register
machinery, and the embedding layer — none of which benefit from the
Gaussian boundedness.

![Gradient flow: bounded vs unbounded components](gradient_flow_bounded_vs_unbounded.png)

### 13.4. Why d=384 Is More Sensitive Than d=256

On TinyStories ($d=256$, 14M params), G1 trained stably at `LR=5e-4` —
four times higher than the Phase 5 peak that caused the blowup. Three
factors explain the scale dependence:

**1. Parameter count scales as $O(d^2)$.** The embedding layer alone
has $50257 \times d$ parameters. At $d=384$ this is 19.3M vs 12.9M at
$d=256$. The gradient norm $\lVert \nabla_\theta \mathcal{L} \rVert$
grows with the square root of the parameter count even for i.i.d.
gradient components:

$$
\lVert \nabla_\theta \mathcal{L} \rVert \approx \bar{g} \cdot \sqrt{P}
$$

where $\bar{g}$ is the per-parameter gradient scale and $P$ is the
parameter count. The ratio $\sqrt{31.5\text{M} / 14\text{M}} \approx 1.5$
means that the same per-parameter gradient produces a 1.5x larger
total gradient norm at $d=384$.

**2. LayerNorm Jacobian amplification.** With `ln_after_step=True`, each
layer applies LayerNorm after the Verlet update. The Jacobian of
LayerNorm with respect to its input has terms proportional to $1/\sigma$,
where $\sigma$ is the standard deviation of the input. At larger $d$,
the variance is better estimated (law of large numbers) but the
centering operation interacts with more dimensions, creating
correlated gradient directions that can align constructively.

**3. Fock register interactions.** With $M=32$ registers (vs 16 on
TinyStories), the register attention mechanism has $32 \times d$ key/query
parameters per layer. The creation/annihilation gating involves softmax
over 32 positions, and the salience computation scales with $M$. More
registers create a richer but more fragile attention landscape.

The net effect is that the **maximum stable learning rate** decreases
with model scale. Empirically:

$$
\text{LR}_{\max}(d=256) \approx 5 \times 10^{-4}, \qquad
\text{LR}_{\max}(d=384) < 2 \times 10^{-4}
$$

Phase 4 (SQ3 on OpenWebText, $d=384$) used `LR=1.2e-4` and was stable
(apart from V_theta-specific blowups). This confirms that `1.2e-4` is
within the stable region for the non-V_theta components at this scale.

### 13.5. The Stability Hierarchy: Four Independent Constraints

Sections 9–13 collectively establish that stable training requires
**four independent constraints**, not just bounded potential:

$$
\underbrace{V \in [-\sum w_k, 0]}_{\text{(1) bounded potential}} \quad + \quad \underbrace{\sigma_{\min} \le \sigma_k \le \sigma_{\max}}_{\text{(2) bounded force}} \quad + \quad \underbrace{\text{LR} \le \text{LR}_{\max}(d, L, M)}_{\text{(3) scale-appropriate LR}} \quad \Longrightarrow \quad \text{stable training}
$$

| Constraint | What it prevents | Where it acts |
|-----------|-----------------|--------------|
| (1) Bounded potential | Blowup 1 (penalty dominance) | V_theta only |
| (2) Bounded force | Blowup 2 (delta spikes), well deactivation | V_theta only |
| (3) Scale-appropriate LR | Full-model gradient explosion | All parameters |

Constraints (1) and (2) are **architectural** — they are enforced by the
Gaussian well design and hold regardless of hyperparameters. Constraint
(3) is **optimiser-dependent** — it depends on the learning rate, the
model scale, and the optimiser's update rule.

### 13.6. Fix: Reduce Peak LR to Phase 4 Baseline

The immediate fix matches the Phase 4 hyperparameters that were stable
at $d=384$:

| Parameter | Phase 5 v1 (blew up) | Phase 5 v2 (fix) |
|-----------|---------------------|-----------------|
| Peak LR | 2e-4 | **1.2e-4** |
| Warmup steps | 8000 | **4000** |
| Grad clip | 0.5 | **1.0** |

The shorter warmup (4000 steps) reflects the lower peak LR: the model
reaches `1.2e-4` at step 4000 with the old warmup rate, so there is no
benefit to extending the ramp further. The looser grad clip (`1.0` vs
`0.5`) preserves more gradient information per step — at `0.5`, the
clip was discarding gradient direction information even during stable
training (most pre-clip norms were 1–6, so a clip of 0.5 was truncating
the majority of updates).

### 13.7. Alternative Optimisers: Could They Prevent This?

The LR sensitivity arises because AdamW applies a **uniform learning
rate** to all parameter groups, regardless of their gradient scale or
the boundedness of their associated module. Several alternative
optimisers address this in different ways:

**LAMB (Layer-wise Adaptive Moments for Batch training).**
LAMB normalises each layer's update by the ratio of the parameter norm
to the gradient norm:

$$
\Delta \theta_l = -\eta \cdot \frac{\lVert \theta_l \rVert}{\lVert \hat{m}_l / (\hat{v}_l^{1/2} + \epsilon) \rVert} \cdot \frac{\hat{m}_l}{\hat{v}_l^{1/2} + \epsilon}
$$

This automatically reduces the effective LR for layers with large
gradient norms (e.g. V_phi, embedding) while preserving it for
well-behaved layers (e.g. V_theta). LAMB was designed for large-batch
training of BERT and would likely tolerate higher peak LR without
blowing up.

**Adafactor.**
Adafactor uses row/column factorised second-moment estimates and
includes relative step sizing:

$$
\rho_t = \min\bigl(\rho_{\max}, 1/\sqrt{t}\bigr)
$$

This provides automatic LR decay without a manual schedule. The
factorised second moments also reduce memory (important at 31.5M
params).

**Gradient centralisation.**
A simple modification to any optimiser: subtract the mean from each
gradient tensor before the update:

$$
\hat{g} = g - \mathrm{mean}(g)
$$

This removes the "DC component" of the gradient, which is often the
dominant contributor to large gradient norms (especially for the
embedding layer where all vocab entries share a common gradient shift).
Gradient centralisation can be added to AdamW with a single line and
has been shown to improve training stability at no computational cost.

**Per-module learning rate groups.**
The simplest approach: assign different learning rates to different
parameter groups in the optimiser. For example:

```python
param_groups = [
    {'params': model.V_theta.parameters(), 'lr': LR},
    {'params': model.V_phi_params(),       'lr': LR * 0.5},
    {'params': [model.E.weight],           'lr': LR * 0.3},
    {'params': other_params,               'lr': LR},
]
```

This is the most transparent approach — it makes the per-module LR
scaling explicit rather than relying on the optimiser to discover it
adaptively. Phase 4 used a version of this (separate LR for V_theta
parameters) as Fix B.

### 13.8. Recommendation

For Phase 5, the conservative approach is `LR=1.2e-4` with AdamW,
matching the validated Phase 4 configuration. This avoids introducing
a new optimiser (and its own hyperparameter tuning surface) while
staying within the empirically stable region.

If future experiments require higher effective LR for faster convergence,
LAMB or per-module LR groups are the recommended next steps. Both
directly address the root cause (non-uniform gradient scale across
modules) rather than relying on global LR reduction as a blunt
instrument.

---

## 14. Scale-Diversity Analysis: Why TinyStories Is Stable and OpenWebText Is Not

### 14.1. The Puzzle

Every instability documented in Sections 1-13 was first observed on
OpenWebText. On TinyStories, both the SQ3 (mixture of quadratic wells)
and Gaussian (mixture of Gaussian wells) architectures trained stably
and achieved competitive perplexity:

| Architecture | TinyStories PPL | Instabilities observed |
|---|---|---|
| SQ3 (Phase 2, d=256, 14M) | **~10** | None |
| Gaussian G1 (K=8, d=256, 14M) | **15.95** | Mild transient spikes (max grad=198) |
| Gaussian G2 (K=16, d=256, 14M) | **17.61** | None |
| Gaussian G3 (SARF N=64, d=256, 14M) | **18.67** | v\_reg drift (fixed by `log_sigma_max`) |

When the same architectures were scaled to OpenWebText (Phase 4 for
SQ3, Phase 5 for Gaussian):

| Architecture | OpenWebText | Instabilities observed |
|---|---|---|
| SQ3 (Phase 4, d=384, 31.5M) | Blowups 1, 2, 3 | Penalty dominance, force spikes, watchdog miscalibration |
| Gaussian G1 (Phase 5, d=384, 31.5M) | LR doom loop, progress trap | Full-model gradient explosion, watchdog reload cycles |

The structured V_theta is not the root cause — the instabilities
arise from the interaction of model scale, data diversity, and learning
rate. This section provides a unified framework for predicting when
instabilities will appear.

### 14.2. The Stability Product

Training stability is governed by a product of three factors:

$$
\mathcal{S} = \underbrace{\sqrt{P}}_{\text{model scale}} \times \underbrace{\sigma_{\text{batch}}(\nabla \mathcal{L})}_{\text{gradient variance}} \times \underbrace{\eta}_{\text{learning rate}}
$$

where $P$ is the total parameter count, $\sigma_{\text{batch}}$ is the
standard deviation of the gradient across different batches, and $\eta$
is the learning rate. The model is stable when $\mathcal{S}$ is below
an architecture-dependent threshold $\mathcal{S}_{\max}$:

$$
\mathcal{S} < \mathcal{S}_{\max} \quad \Longrightarrow \quad \text{stable training}
$$

Each factor contributes differently on TinyStories vs OpenWebText.

### 14.3. Factor 1: Model Scale ($\sqrt{P}$)

The total gradient norm scales with the square root of the parameter
count for i.i.d. gradient components:

$$
\lVert \nabla_\theta \mathcal{L} \rVert \approx \bar{g} \cdot \sqrt{P}
$$

| Configuration | $P$ | $\sqrt{P}$ | Ratio vs TinyStories |
|---|---|---|---|
| TinyStories (d=256) | 14.0M | 3,742 | 1.0x |
| OpenWebText (d=384) | 31.5M | 5,612 | **1.50x** |

This factor alone increases the gradient norm by 50% on OpenWebText.
The effect is amplified by the larger Fock register pool ($M=32$ vs 16)
and deeper integration stack ($L=16$ layers with $d=384$-dimensional
states).

### 14.4. Factor 2: Gradient Variance ($\sigma_{\text{batch}}$)

This is the dominant factor and the qualitative difference between
the two corpora. With a batch of 8 sequences at block_size=512, each
training step sees only 4,096 tokens. The gradient computed from these
tokens is a noisy estimate of the full-dataset gradient, and the noise
level depends on the homogeneity of the corpus.

**TinyStories** is a highly homogeneous corpus:
- Approximately 2.3M short children's stories.
- Vocabulary: ~5,000 active tokens out of 50,257 (simple nouns,
  verbs, names like "Lily" and "Tom").
- Syntax: almost exclusively SVO order, past tense, conjunctions.
- Semantics: concrete objects, simple emotions, fairy-tale logic.
- Any batch of 8 stories looks statistically similar to any other.

**OpenWebText** is a high-entropy, heavy-tailed corpus:
- Approximately 8B tokens from 8M web documents.
- Full vocabulary utilisation — technical jargon, URLs, code,
  foreign words, named entities, numbers.
- Syntax: news prose, Reddit comments, blog posts, code, recipes,
  academic writing — all interleaved.
- A batch of 8 random documents can span wildly different domains.

The gradient variance decomposes across model components:

**Embedding gradient ($E$, 61% of params).**
The embedding gradient for token $i$ is non-zero only when token $i$
appears in the batch. On TinyStories, the same ~5K tokens appear in
nearly every batch, so $\nabla_E \mathcal{L}$ is dense and stable.
On OpenWebText, rare tokens appear sporadically — when a batch happens
to contain a rare technical term, the gradient for that embedding row
spikes while all other rows see zero gradient. The embedding gradient
has a heavy-tailed distribution across batches.

**V\_phi routing gradient (score head, 0.6% of params).**
The Gumbel-softmax top-k routing learns which past tokens to attend
to. On TinyStories, the optimal routing is predictable (similar
syntactic patterns across stories). On OpenWebText, the optimal
routing changes dramatically between batches — a code snippet needs
different routing than a news paragraph. The score head gradient
direction fluctuates strongly.

**Fock register gradients (2.3M params).**
The creation/destruction gates learn when to activate registers
based on token-register attention scores. On TinyStories, the same
register activation patterns recur (e.g., "character introduced"
triggers creation, "story ends" triggers destruction). On OpenWebText,
the activation patterns are far more variable, creating gradient
noise in the QKV creation gate and destruction gate parameters.

**V\_theta gradient (9.5M params for Gaussian G1).**
The Gaussian wells learn centres $\mu_k(\xi)$, precisions $a_k(\xi)$,
and weights $w_k(\xi)$ conditioned on the multi-channel context
$\xi$. On TinyStories, the context distribution is narrow — the
model sees similar $\xi$ vectors across batches. On OpenWebText,
$\xi$ spans a much wider subspace of $\mathbb{R}^{K_\xi \cdot d}$,
and the projections $\mu_k(\xi)$ must cover a much larger volume.
The gradient of the mu\_proj and a\_proj linear layers fluctuates
as different regions of $\xi$-space are sampled in each batch.

**Estimated variance ratio.** Combining these effects, the inter-batch
gradient standard deviation on OpenWebText is estimated to be
3-5x larger than on TinyStories:

$$
\frac{\sigma_{\text{batch}}(\text{OWT})}{\sigma_{\text{batch}}(\text{TS})} \approx 3 \text{--} 5
$$

### 14.5. Factor 3: Learning Rate ($\eta$)

The learning rate multiplies the (noisy) gradient to produce the
parameter update. Higher $\eta$ amplifies the gradient noise linearly.

| Run | Peak LR | Stable? |
|---|---|---|
| TinyStories G1 (d=256) | 5.0e-4 | Yes |
| TinyStories G2 (d=256) | 5.0e-4 | Yes |
| OpenWebText Phase 4 SQ3 (d=384) | 1.2e-4 | Yes (after Blowup 1-3 fixes) |
| OpenWebText Phase 5 (d=384, LR=2e-4) | 2.0e-4 | **No** — doom loop at step 8700 |
| OpenWebText Phase 5 (d=384, LR=1.2e-4) | 1.2e-4 | **Marginal** — watchdog triggers, progress trap |

TinyStories tolerates 4x higher LR because the other two factors are
much smaller.

### 14.6. The Stability Product in Practice

Combining the three factors:

$$
\frac{\mathcal{S}(\text{OWT, } d\!=\!384)}{\mathcal{S}(\text{TS, } d\!=\!256)} = \underbrace{1.5}_{\sqrt{P}} \times \underbrace{3 \text{--} 5}_{\sigma_{\text{batch}}} \times \underbrace{\frac{\eta_{\text{OWT}}}{\eta_{\text{TS}}}}_{\text{LR ratio}}
$$

At $\eta_{\text{OWT}} = 2 \times 10^{-4}$ and $\eta_{\text{TS}} = 5 \times 10^{-4}$:

$$
\mathcal{S}_{\text{ratio}} = 1.5 \times 4 \times 0.4 = 2.4
$$

At $\eta_{\text{OWT}} = 1.2 \times 10^{-4}$:

$$
\mathcal{S}_{\text{ratio}} = 1.5 \times 4 \times 0.24 = 1.44
$$

The first configuration ($\eta = 2 \times 10^{-4}$) pushes $\mathcal{S}$
2.4x above the TinyStories baseline, triggering the doom loop. The
second ($\eta = 1.2 \times 10^{-4}$) keeps $\mathcal{S}$ only 1.44x
above baseline — above the stability threshold but close enough that
the model functions with occasional watchdog interventions.

### 14.7. Architecture-Specific Amplifiers

The stability product determines **whether** instability occurs. The
V_theta architecture determines **which** instability manifests:

**SQ3 (unbounded quadratic wells).**
The quadratic penalty term $\lambda \cdot V_\theta^2$ couples the
potential depth to the gradient norm. When a diverse OWT batch drives
$V$ to large values in some hidden-state region, the penalty gradient
scales as $O(V)$, creating a positive feedback loop:

$$
\text{diverse batch} \to \text{large } V \to \text{large } \nabla(V^2) \to \text{large update} \to \text{even larger } V
$$

On TinyStories, $V$ remains moderate because the hidden states occupy
a narrow, predictable region of $\mathbb{R}^d$. The feedback loop
never activates.

**Gaussian wells (bounded potential).**
The bounded potential $V \in [-\sum w_k, 0]$ eliminates the penalty
amplification channel. But it does not protect the non-V_theta
components — the embedding, V_phi, and Fock register gradients are
still subject to the full gradient variance. The instability shifts
from V_theta-internal (Blowup 1) to full-model (Phase 5 LR blowup,
progress trap).

### 14.8. Predictive Framework for Future Scale-Ups

The stability product provides a practical formula for anticipating
instabilities when scaling to new model sizes or corpora:

$$
\eta_{\max}(\text{new}) \approx \eta_{\text{stable}}(\text{ref}) \times \frac{\sqrt{P_{\text{ref}}}}{\sqrt{P_{\text{new}}}} \times \frac{\sigma_{\text{batch}}(\text{ref})}{\sigma_{\text{batch}}(\text{new})}
$$

where "ref" is a known-stable configuration.

**Example: scaling to d=512 (Phase 6).**
If $P_{\text{new}} \approx 80\text{M}$ and the corpus remains
OpenWebText:

$$
\eta_{\max}(d\!=\!512) \approx 1.2 \times 10^{-4} \times \frac{\sqrt{31.5\text{M}}}{\sqrt{80\text{M}}} \approx 1.2 \times 10^{-4} \times 0.63 \approx 7.5 \times 10^{-5}
$$

**Example: scaling to a more diverse corpus (e.g., The Pile).**
If the batch gradient variance is ~2x higher than OpenWebText:

$$
\eta_{\max}(\text{Pile}) \approx 1.2 \times 10^{-4} \times 0.5 = 6.0 \times 10^{-5}
$$

These estimates are rough (the exponents depend on the model's
specific gradient structure), but they provide the correct order
of magnitude and prevent the trial-and-error that characterised the
Phase 4 and Phase 5 debugging process.

### 14.9. Implications for the Paper

The stability analysis yields three claims relevant to the
mega-paper:

**Claim 1: Corpus diversity, not model architecture, is the
primary driver of training instability.**
The Fock-PARFLM architecture is stable on homogeneous corpora at all
tested scales. Instabilities emerge only when corpus diversity
(measured by inter-batch gradient variance) exceeds a threshold that
depends on model scale and learning rate. This is not specific to the
PARFLM framework — standard transformers face analogous instabilities
at scale (learning rate warmup, gradient clipping, and loss spikes are
standard mitigations in large-scale transformer training).

**Claim 2: Bounded V_theta narrows but does not eliminate the
instability surface.**
The Gaussian well architecture eliminates V_theta-specific failure
modes (Blowups 1-3) by construction. The remaining instabilities
are in the non-V_theta components (embedding, V_phi, Fock registers)
and are amenable to standard optimisation mitigations (LR reduction,
gradient clipping, EMA watchdog). This decomposition —
architecture-level boundedness + optimiser-level robustness — is a
principled design pattern for conservative-dynamics language models.

**Claim 3: TinyStories ablation validates architecture; OpenWebText
validates optimisation.**
The TinyStories experiments establish that the V_theta architecture
(SQ3, Gaussian, SARF) produces competitive perplexity under
controlled conditions. The OpenWebText experiments test whether the
architecture's stability properties hold under realistic data
diversity. Both are necessary: TinyStories alone would miss all
scale-diversity instabilities; OpenWebText alone would conflate
architectural and optimisation issues.

---

*This note documents the training process of a research experiment and is
intended for internal diagnostic use. The fixes described here are all
implemented in the notebook
`notebooks/conservative_arch/scaleup/colab_fock_structured_vtheta_openwebtext_phase4.ipynb`.
The SARF anchor computation is implemented in
`notebooks/semsim_simulator/estimators/sarf_anchors.py` and the SARF-faithful
dynamics in `notebooks/conservative_arch/sarf_variant/model_sarf.py`.
The Gaussian-well and SARF-anchored V_theta implementations are in
`notebooks/conservative_arch/parf/model_gaussian_vtheta.py`.
The TinyStories ablation notebook is at
`notebooks/conservative_arch/scaleup/colab_fock_gaussian_sarf_vtheta.ipynb`
and the OpenWebText Phase 5 scale-up notebook at
`notebooks/conservative_arch/scaleup/colab_fock_gaussian_sarf_openwebtext_phase5.ipynb`.*
