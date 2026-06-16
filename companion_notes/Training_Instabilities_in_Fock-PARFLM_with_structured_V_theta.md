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

---

## 1. Background

### Architecture

The SQ3 structured V_θ replaces the MLP scalar potential with a
\\(K_{\mathrm{mix}}\\)-component mixture of diagonal quadratic wells:

$$
V_\theta(\xi, h) \;=\; -\tau \log\sum_{k=1}^{K} \pi_k(\xi)\,
\exp\!\left(-\frac{E_k(\xi, h)}{\tau}\right) + b(\xi)
$$

where the per-component energy is

$$
E_k(\xi, h) \;=\; \tfrac{1}{2}\,a_k(\xi)^\top (h - \mu_k(\xi))^2, \qquad a_k > 0.
$$

The attractor centres \\(\mu_k(\xi)\\) and precisions \\(a_k(\xi)\\) are linear
projections of the flattened multi-xi context
\\(\xi_{\mathrm{flat}} = [{\xi_1}^\top \cdots {\xi_K}^\top] \in \mathbb{R}^{K_\xi d}\\).

**Key structural fact:** \\(V_\theta\\) is quadratic in \\(h\\) and has **no upper
bound**. As \\(\|h - \mu_k\|\\) grows, every \\(E_k\\) grows without limit, and
so does \\(V_\theta\\).

### The regulariser

To prevent the learned potential from becoming arbitrarily flat (zero-force
landscape), we add a penalty term to the training loss:

$$
\mathcal{L} \;=\; \mathcal{L}_{\mathrm{NTP}} + \lambda_V \cdot \mathcal{R}(V_\theta)
$$

with \\(\lambda_V = 0.01\\). The choice of \\(\mathcal{R}\\) is the source of both
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
step    5200  ntp=10.52  v_reg=27770.6  lr=3.00e-04  grad=216,392,624
step    5400  ntp= 9.91  v_reg=18680.0  lr=3.00e-04  grad= 51,402,900
step    5600  ntp=10.27  v_reg=232,649  lr=3.00e-04  grad=    601,585
```

The model never recovered. Val PPL went from ~535 (step 4k) to 106,078 (step 20k).

### Root cause: unbounded mean(V_θ²)

The original regulariser was:

$$
\mathcal{R}_{\mathrm{quad}}(V_\theta) \;=\; \mathbb{E}_{x,h}\bigl[V_\theta(\xi, h)^2\bigr]
$$

This is computed as a batch mean. A **single token** whose hidden state
\\(h\\) drifts far from every well centre produces:

$$
V_\theta(\xi, h) \;\approx\; \tfrac{1}{2}\,\min_k a_k^\top(h-\mu_k)^2
\;\sim\; \|h\|^2 \quad \text{(quadratic in } \|h\|)
$$

so \\(V_\theta^2 \sim \|h\|^4\\). One outlier token with \\(\|h\| \approx 10\\)
contributes \\(\sim 10^4\\) to \\(\mathcal{R}_{\mathrm{quad}}\\), pulling
\\(\lambda_V \cdot \mathcal{R} \approx 100\\) — far exceeding the NTP loss of
\\(\sim 10\\) nats.

### The feedback loop

```mermaid
flowchart TD
    A["h drifts from well centres<br>(e.g. hard OWT batch)"]
    B["V_θ(ξ, h) grows quadratically"]
    C["R = mean(V_θ²) grows as ‖h‖⁴<br>λ_V · R >> NTP loss"]
    D["Gradient ∂R/∂θ = 2V_θ · ∂V_θ/∂θ<br>dominates ∂L_NTP/∂θ"]
    E["Optimizer pushes θ to minimize R<br>i.e. pull μ_k toward h"]
    F["New μ_k chase h → h chases μ_k<br>runaway instability"]
    G["‖grad‖ ~ 2×10⁸<br>clip = 0.5 → tiny but misdirected steps"]

    A --> B --> C --> D --> E --> F --> G --> A
```

The critical insight is that **gradient clipping cannot stop this loop**.
Clipping bounds the step size but not the direction. Once
\\(\lambda_V \mathcal{R} \gg \mathcal{L}_{\mathrm{NTP}}\\), every clipped step
points toward reducing \\(\mathcal{R}\\) (pulling attractor centres toward
the outlier hidden states), which does not reduce NTP loss. The model is
trapped.

### Gradient of the original penalty

$$
\frac{\partial}{\partial \theta}\bigl(\lambda_V V_\theta^2\bigr)
\;=\; 2\lambda_V\, V_\theta \cdot \frac{\partial V_\theta}{\partial \theta}
\;\propto\; V_\theta
$$

Since \\(V_\theta\\) is unbounded, so is this gradient — the penalty is a
**linearly amplified** version of the already-large force, making recovery
impossible once the runaway starts.

---

## 3. Blowup 2 — Directional Instability

After applying the `log1p` fix (Section 5), a second, structurally different
instability occurred at step ~22,000.

### Symptom (step ~21,800–24,000)

```
step   21800  ntp=5.670  v_reg=1.23   grad=214.28   ← first spike
step   22000  EVAL  val_ppl=301.75    best=268.84   ← regression
step   22600  ntp=6.193  v_reg=2.74   grad=167.78
step   23200  ntp=6.624  v_reg=4.98   grad= 93.79
step   24000  EVAL  val_ppl=388.65    best=268.84   ← continued regression
```

This time: `v_reg` peaked at ~5 (not 2×10⁵), `grad_norm` peaked at 214
(not 2×10⁸), NTP reached 6.7 (not 16+). The log1p fix contained the
severity, but the model failed to self-correct over 2,000+ affected steps.

### Root cause: directional persistence under gradient clipping

`clip_grad_norm_` rescales the gradient vector \\(g\\) as:

$$
\hat{g} \;=\; g \cdot \frac{\min(\|g\|, C)}{\|g\|}
$$

so the applied update has magnitude exactly \\(\min(\|g\|, C)\\). With
\\(\|g\| = 214\\) and \\(C = 0.5\\):

$$
\hat{g} \;=\; g \cdot \frac{0.5}{214} \;\approx\; 0.0023\, g
$$

The step is tiny, but it points in the **same direction as** \\(g\\). The
direction of \\(g\\) is determined by which loss component dominates, and
when the Verlet backward graph is deep, a hard batch can induce a gradient
direction that is persistently destabilizing even at tiny step sizes.

### Why it did not self-correct

At step ~21,800 a hard batch produced \\(\|g\| = 214\\). The clipped update
moved parameters by \\(0.5/214 \approx 0.002\\) in that direction. If the
next batch is also hard (OWT is streamed sequentially — consecutive batches
share document context), the direction accumulates. Over 2,000 steps the
cumulative displacement from the best basin is:

$$
\Delta\theta_{\mathrm{cum}} \;\approx\; \sum_{t=T_0}^{T_0+2000} \hat{g}_t
\;\sim\; 2000 \times 0.5 \;=\; 1000 \quad \text{(in gradient units)}
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

On TinyStories (i.i.d. story chunks), a hard batch at step \\(t\\) is
statistically independent from the batch at step \\(t+1\\), so the
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
step   22000  EVAL  val_ppl=252.75  *** NEW BEST ***   ← checkpoint saved
step   22200  ntp=5.6232  v_reg=1.044  grad= 1.19
step   22400  ntp=5.6164  v_reg=1.171  grad= 1.33
step   22600  ntp=5.6311  v_reg=1.267  grad= 1.31
step   22800  ntp=5.6398  v_reg=1.291  grad= 3.37   ← first elevated spike
step   23000  ntp=5.6362  v_reg=1.279  grad= 1.22
step   23200  ntp=5.6439  v_reg=1.260  grad= 2.14
step   23400  ntp=5.7605  v_reg=1.717  grad= 3.52   ← inflection point
step   23600  ntp=5.6823  v_reg=1.263  grad= 2.66
step   23800  ntp=5.7243  v_reg=1.298  grad= 5.05
step   24000  ntp=5.8080  v_reg=1.527  grad=16.76   ← confirmed blowup
step   24000  EVAL  val_ppl=367.98  best=252.75      ← 45% regression
```

This blowup is structurally different from Blowup 2. The gradients are
moderate (peak 16.76, not 214), `v_reg` stays below 1.8 (not 5), and the
onset is gradual — a slow escalation over 2,000 steps rather than a sudden
spike.

### Root cause: the EMA threshold was mathematically unreachable

The watchdog parameters were:

```python
GRAD_NORM_EMA_ALPHA     = 0.02   # slow EMA (~50-step memory)
GRAD_NORM_EMA_THRESHOLD = 50.0   # EMA > 50 → consider unstable
GRAD_NORM_EMA_PATIENCE  = 100
```

The EMA update is:

$$
\bar{g}_t = (1 - 0.02)\,\bar{g}_{t-1} + 0.02\,\hat{g}_t
= 0.98\,\bar{g}_{t-1} + 0.02\,\hat{g}_t
$$

The **steady-state** value of the EMA when the gradient is held constant at
\\(\hat{g}\\) is exactly \\(\hat{g}\\) itself. But with \\(\alpha = 0.02\\) the
convergence is slow: from a baseline of 1.3, even 100 consecutive steps at
\\(\hat{g} = 17\\) bring the EMA to only:

$$
\bar{g}_{100} \;=\; 17\bigl(1 - 0.98^{100}\bigr) + 1.3 \times 0.98^{100}
\;\approx\; 17 \times 0.87 + 1.3 \times 0.13 \;\approx\; 14.9 + 0.17 \;\approx\; 15.1
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
| V_θ force | \\(-\nabla_h V_\theta\\) | same | quadratic, unbounded |
| V_φ routing | — | Gumbel-softmax top-k | stochastic routes add gradient noise; wrong routes persist for a full Verlet step |
| Fock registers | — | M=32 register particles | 32 extra hidden states through L=16 backward graph; register creation/destruction gates add non-differentiable-like switching |
| Per-register τ | — | learnable temperature | τ drift can make routing sharper mid-training, amplifying the Gumbel noise |
| Reverse channel | — | non-conservative Q_i | no energy conservation guarantee; small errors in Q_i can compound across layers |

### Effective backward graph depth

For the SPLM with L layers the backward graph is roughly:

$$
\text{depth}_{\mathrm{SPLM}} \;\approx\; L \times d^2
$$

For Fock-PARFLM v2.1, each layer has Verlet step + V_φ routing
(top-k gather) + Fock register dynamics + reverse channel. A conservative
estimate:

$$
\text{depth}_{\mathrm{Fock}} \;\approx\; L \times \bigl(d^2 + 2\,d\,k + M\,d + d^2\bigr)
\;\approx\; 4\text{–}5 \times \text{depth}_{\mathrm{SPLM}}
$$

Same `grad_clip` → same magnitude bound, but 4–5× longer backward path →
a destabilizing gradient direction persists 4–5× longer before it is washed
out by the curvature of subsequent batches.

---

## 6. Fixes Applied and Their Mathematical Justification

### Fix 1: Bounded regulariser — log1p(V_θ²)

**Applied after Blowup 1.**

Replace \\(\mathcal{R}_{\mathrm{quad}} = \mathbb{E}[V_\theta^2]\\) with:

$$
\mathcal{R}_{\mathrm{log}}(V_\theta)
\;=\; \mathbb{E}\bigl[\log(1 + V_\theta^2)\bigr]
$$

![Penalty comparison](figures/vtheta_penalty_comparison.png)

**Why it works:**

1. **Normal regime equivalence.** For \\(|V_\theta| \ll 1\\):
   \\(\log(1+V^2) \approx V^2\\), so the landscape compression is identical.

2. **Gradient bound.** The gradient with respect to \\(V_\theta\\) is:

$$
\frac{d}{dV}\log(1+V^2) \;=\; \frac{2V}{1+V^2}
$$

   This is bounded by \\(\left|\frac{2V}{1+V^2}\right| \leq 1\\) for all
   \\(V \in \mathbb{R}\\), with the maximum at \\(V=1\\). Therefore:

$$
\lambda_V \cdot \frac{\partial \mathcal{R}_{\mathrm{log}}}{\partial \theta}
\;\leq\; \lambda_V \cdot \mathbb{E}\!\left[\frac{\partial V_\theta}{\partial \theta}\right]
$$

   and the penalty contribution to the total gradient is bounded by
   \\(\lambda_V\\) times the mean magnitude of \\(\partial V_\theta / \partial\theta\\).
   It **cannot** exceed the NTP gradient unless \\(\lambda_V \gg 1\\).

3. **No runaway.** The penalty loss itself is bounded: as \\(|V_\theta| \to \infty\\),
   \\(\log(1+V^2) \to \log V^2 = 2\log V\\), which grows only logarithmically.
   A batch where \\(V_\theta = 1000\\) contributes \\(\log(10^6+1) \approx 13.8\\)
   to the penalty, not \\(10^6\\).

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

For a hard batch with \\(\|g\| = G\\) and clip threshold \\(C\\), the
applied step in any direction is at most:

$$
\|\Delta\theta\| \;\leq\; \eta_t \cdot C
$$

where \\(\eta_t\\) is the learning rate at step \\(t\\). After the cosine
schedule with \\(\eta_{\max}\\) and warmup \\(T_w\\):

$$
\eta_t \;=\; \frac{\eta_{\max}}{2}\left(1 + \cos\!\left(\pi\,\frac{t-T_w}{T-T_w}\right)\right)
$$

At step 22k of a 200k run (warmup at 8k):

$$
\text{progress} \;=\; \frac{22000-8000}{200000-8000} \;\approx\; 0.073
\quad\Rightarrow\quad
\eta_{22k} \;\approx\; 0.997\,\eta_{\max} \approx \eta_{\max}
$$

The cosine schedule had barely started decaying. The maximum per-step
displacement in any direction is:

$$
\|\Delta\theta_{\max}\| \;=\; \eta_{\max} \cdot C
\;=\; \begin{cases}
2\times10^{-4} \times 0.5 = 1\times10^{-4} & \text{(Blowup 2 run)} \\
1.5\times10^{-4} \times 0.3 = 4.5\times10^{-5} & \text{(current run)}
\end{cases}
$$

The new maximum displacement is **2.2× smaller**. Over 100 consecutive hard
steps the cumulative drift in a destabilizing direction is:

$$
\|\Delta\theta_{\mathrm{cum}}\|
\;\lesssim\; 100 \times 4.5\times10^{-5} \;=\; 4.5\times10^{-3}
\quad\text{(vs.}\; 1\times10^{-2}\text{ before)}
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
for a sustained gradient of \\(G\\) converges to \\(G\\) itself. With
\\(\alpha = 0.05\\) and the new threshold \\(\tau = 3.5\\):

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

Let \\(\hat{g}_t = \|g_t\|\\) (raw gradient norm before clipping) at step \\(t\\).
Maintain an exponential moving average:

$$
\bar{g}_t \;=\; (1-\alpha)\,\bar{g}_{t-1} + \alpha\,\hat{g}_t,
\qquad \alpha = 0.05 \;\text{(~20-step memory)}
$$

Maintain a counter \\(c_t\\):

$$
c_t \;=\; \begin{cases}
c_{t-1} + 1 & \text{if } \bar{g}_t > \tau_g \\\\
0 & \text{otherwise}
\end{cases}
\qquad \tau_g = 3.5
$$

If \\(c_t \geq P = 30\\) (30 consecutive steps above threshold):
1. Log the event.
2. Call `_reload_best_checkpoint()` — loads `*_best.pt` into `model` and `optim` in-place.
3. Reset \\(\bar{g}_t = 0\\), \\(c_t = 0\\).
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
\tau_g \;\approx\; 2\text{–}3 \times \bar{g}_{\mathrm{baseline}}
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
(a) an **unbounded** quadratic-in-\\(h\\) potential,
(b) **deep** Verlet integration (L=16), and
(c) **sequential data** with correlated hard batches.

Factor (a) is inherent to the SQ3 parameterisation. The log1p fix is a
sound mitigation but not a cure — it prevents the penalty from dominating
but does not prevent \\(V_\theta\\) from growing large during forward inference.
A more principled long-term fix would be to add a **soft-clamp** on \\(h\\)
itself (e.g. via layer normalization before the V_θ force computation) so
that \\(\|h - \mu_k\|\\) cannot grow arbitrarily. The `ln_after_step=True`
flag provides partial protection (LN is applied after each Verlet step),
but LN is applied to the full \\(h\\) vector, not to the difference
\\(h - \mu_k(\xi)\\), so extreme well-displacement can still occur within a
single step.

For MLP-based V_θ (no structured form), the force \\(-\nabla_h V_\theta\\)
is an arbitrary neural network output and is empirically bounded by the
weight norms — an implicit regularisation that the analytical SQ3 form does
not share.

---

*This note documents the training process of a research experiment and is
intended for internal diagnostic use. The fixes described here are all
implemented in the notebook
`notebooks/conservative_arch/scaleup/colab_fock_structured_vtheta_openwebtext_phase4.ipynb`.*
