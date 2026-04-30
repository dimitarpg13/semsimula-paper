# Pre-Registered Protocol — SPLM-1 First-Order Ablation

> Pre-registration document, drafted **April 28, 2026**, by Dimitar Gueorguiev with Claude.
> Companion to:
> *Semantic Simulation: A Prescriptive Lagrangian Framework for Efficient Semantic Inference* (Gueorguiev, 2026), v3.
> Companion design note: [`Replacing_The_Conservative_Mechanism_of_SPLM_with_First_Order.md`](./Replacing_The_Conservative_Mechanism_of_SPLM_with_First_Order.md).

> **Status.** Pre-registered, not yet executed at the multi-seed level. This document fixes the experimental design, the analysis pipeline, and the decision rule **before** any sweep of the SPLM-1 multi-seed ablation is run. The single 300-step smoke test of the trainer (committed alongside this protocol) is the only execution preceding pre-registration; its sole purpose was to confirm gradient flow, no-velocity-buffer state-dict shape, and LayerNorm-after-step compatibility, and its results are not used to set any threshold below.
>
> The committing commit hash is the timestamp of pre-registration.

---

## 1. Question

the paper advances a refined two-tier claim about SPLM dynamics:

- **Generative second-order** (paper §12). The trained SPLM is induced by a damped Lagrangian $\mathcal{L} = \tfrac{1}{2} m_t \lVert \dot h_t \rVert^2 - V_\theta(\xi_t, h_t)$, whose Euler–Lagrange equation is the inertia-with-damping ODE $m_t \ddot h_t = -\nabla V_\theta - \gamma \dot h_t$. The velocity buffer $v$, the damping coefficient $\gamma$, and the second-order semi-implicit integrator are all parts of the **training-time** computation graph.
- **Observational first-order** (paper §12, last paragraph). At the trained inference fixed point the dynamics reduce, in the overdamped/heavy-mass regime, to an effectively first-order gradient flow $\gamma \dot h_t \approx -\nabla V_\theta$ — which is consistent with the Markov-order-test result on GPT-2 / Pythia (Outcome C of the first-order rejection protocol).
- **Training-time value of the inertial term** (paper §12 last paragraph and §15 "*The interior of $\gamma^{\ast}$ is the value-add*"). The E4 plain-Euler and E5 LN-after-step damping sweeps both produce U-shaped validation perplexity in the fixed-$\gamma$ grid with an interior minimum at $\gamma^{\ast} \approx 0.30$, strictly between the ballistic floor and the heavy-overdamping limit. The paper interprets this as direct empirical evidence that the inertial term is dynamically active during learning.

The interior-minimum result, taken alone, is consistent with *both* of the following:

- (Hypothesis I) The training-time inertial term contributes genuine predictive value over and above what any first-order gradient flow can extract from the same $V_\theta$ family. The U-shape is then a real signal of inertia at $\gamma > 0$, and the heavy-overdamping arm is a degenerate first-order limit that under-fits because the time-step is consumed by the damping factor.
- (Hypothesis II) The U-shape is a damping/step-size artefact. The interior optimum is just whichever value of $\gamma$ produces the best effective learning rate $\beta_{\mathrm{eff}} = \mathrm{d}t / (\gamma \cdot m)$ for AdamW; there is no genuine value being added by the velocity buffer or the second-order graph, and a pure first-order SPLM at the same training budget would match $\gamma^{\ast}$.

This experiment is designed to discriminate Hypothesis I from Hypothesis II by training a structurally first-order SPLM (denoted **SPLM-1**) under matched compute, matched architecture, and matched data, and comparing it to the second-order $\gamma^{\ast}$ winner from E5.

---

## 2. The SPLM-1 model

SPLM-1 is the smallest possible architectural delta from the SPLM em\_ln baseline (the E5 architecture). The exact change is in the layer update:

| Family | Layer update |
|---|---|
| SPLM em\_ln (second-order, the E5 architecture) | $v_{l+1} = (v_l + \mathrm{d}t \cdot f / m) / (1 + \mathrm{d}t \cdot \gamma)$, then $h_{l+1} = \mathrm{LN}(h_l + \mathrm{d}t \cdot v_{l+1})$ |
| **SPLM-1** (first-order, this protocol) | $h_{l+1} = \mathrm{LN}(h_l - \mathrm{d}t \cdot \nabla V_\theta(\xi_l, h_l) / m)$ |

where $f = -\nabla V_\theta$. The following pieces are **identical** between the two families:

- The scalar potential $V_\theta(\xi, h)$ — same MLP architecture, same depth, same width, same initialisation.
- The causal context pool $\xi_t = \tfrac{1}{t}\sum_{s\le t} h_s$.
- The per-token semantic mass $m_t$ (logfreq mode, same surprisal table).
- The tied-embedding readout, the embedding/positional initialisation, the LayerNorm-after-step projector, the loss, the optimiser (AdamW, same betas, same weight decay), the learning-rate schedule (cosine with warmup), the batch size, the block size, the number of steps, the gradient clip.

The **only** quantities removed in SPLM-1 are:

- the velocity buffer $v$ (no per-layer state),
- the damping coefficient $\gamma$ (set to 0, non-trainable, never read by the integrator),
- the inertial term $m \ddot h$ (no second-order component in the layer update).

Implementation:
- Model class: `notebooks/conservative_arch/first_order_ablation/model_first_order.py:ScalarPotentialLMFirstOrder`.
- Trainer: `notebooks/conservative_arch/first_order_ablation/train_splm_first_order.py`.

---

## 3. The comparison anchor

The second-order arm is **SPLM em\_ln at fixed $\gamma = 0.30$**, the E5 sweep winner. From the completed E5 sweep (single seed, seed=0):

| $\gamma$ | val PPL |
|---|---|
| 0.00 | 113.01 |
| 0.10 | 91.33 |
| **0.30** | **87.06** |
| 0.85 | 93.93 |
| 2.00 | 103.82 |
| 5.00 | 121.89 |

This single E5 seed=0 cell is reproduced *inside* the present sweep (alongside two new seeds) so that arm A and arm B share an identical execution window, machine, and Python environment. We do **not** re-use the E5 number directly; the arm-B baseline below is the multi-seed mean produced by the present sweep.

---

## 4. Hypotheses

Let $P_A^{(s)}$ be the final validation perplexity of SPLM-1 at seed $s$ (arm A) and $P_B^{(s)}$ the final validation perplexity of SPLM em\_ln at $\gamma = 0.30$ at seed $s$ (arm B). Both are evaluated on the same held-out Tiny Shakespeare validation split, with the same `eval_iters = 40`, the same `block_size = 128`, the same RNG-controlled batch protocol.

Define:

$$\Delta_s = P_A^{(s)} - P_B^{(s)}, \qquad \overline{\Delta} = \tfrac{1}{S}\sum_{s=1}^{S} \Delta_s,$$

with $S = 3$ seeds, $s \in \{0, 1, 2\}$, fixed at pre-registration.

| Hypothesis | Operational form | Theoretical reading |
|---|---|---|
| $H_1$ (paper claim, training-time value) | $\overline{\Delta} \ge \Delta_{\min} > 0$ | the inertial term contributes training-time predictive value over any first-order reduction at matched compute |
| $H_0$ (artefact) | $\lvert \overline{\Delta} \rvert < \Delta_{\min}$ | the U-shape in E5 is an effective-learning-rate artefact; first-order matches $\gamma^{\ast}$ |
| $H_{-1}$ (refutation) | $\overline{\Delta} \le -\Delta_{\min}$ | the second-order arm under-performs first-order; the training-time-value claim is falsified |

The paper's $\gamma^{\ast}$-interior-is-the-value-add claim is $H_1$.

---

## 5. Decision rule (locked at pre-registration)

The minimum effect size is fixed at:

$$\Delta_{\min} = 5.0 \text{ perplexity units}.$$

Justification. The E5 sweep itself shows ≥ 4-point gaps between adjacent fixed-$\gamma$ cells in the interior of the U-shape (87.06 → 91.33 from $\gamma = 0.30$ to $\gamma = 0.10$; 87.06 → 93.93 from $\gamma = 0.30$ to $\gamma = 0.85$). A genuine second-order training-time advantage should produce a comparable or larger gap when the entire inertial mechanism is removed; if it does not, the mechanism is delivering at most the same value as a single $\gamma$-grid step, which we take as too small to support the §12 / §15 framing.

The decision rule:

- **Outcome A (training-time value confirmed; supports $H_1$):**
  $\overline{\Delta} \ge 5.0$, **and** the per-seed sign of $\Delta_s$ is consistent across all three seeds (all $\Delta_s > 0$), **and** the paired one-sided Wilcoxon test of $P_A^{(s)} > P_B^{(s)}$ has $p \le 0.10$ (we use $0.10$ rather than $0.05$ because $S = 3$ is small).
- **Outcome B (no clear effect; ambiguous):**
  $\lvert \overline{\Delta} \rvert < 5.0$, *or* the sign of $\Delta_s$ is inconsistent across seeds. The paper's training-time-value framing must be softened to "the U-shape with interior minimum is suggestive but not falsifiable evidence" and the §15 "value-add" paragraph rewritten accordingly.
- **Outcome C (refutation):**
  $\overline{\Delta} \le -5.0$, with consistent sign across all three seeds. The paper's training-time-value framing is **falsified**; the §12 and §15 paragraphs claiming "training-time value of the inertial term" are retracted, and SPLM-1 becomes the recommended architecture.

The outcome (A / B / C) is determined **only** from these three numbers. No post-hoc threshold adjustment, seed substitution, or per-seed exclusion is permitted. If a cell crashes (TRAINING\_FAILED.txt marker), it is replaced by the next consecutive seed (seed = 3, then 4, …) until $S = 3$ valid cells per arm are obtained; the substitution is logged in the eventual write-up.

---

## 6. Pre-registered prediction

The author predicts **Outcome A** with the following expected effect size, written before the SPLM-1 sweep is run:

$$\overline{\Delta}_{\text{predicted}} \in [10, 30] \text{ perplexity units.}$$

The reasoning is informal:
- The interior $\gamma^{\ast}$ result places the second-order minimum strictly below both the $\gamma = 0$ ballistic arm (PPL 113) and the $\gamma = 5$ heavy-overdamped arm (PPL 121).
- SPLM-1 is, in effect, a $\gamma \to \infty$ reduction (no $v$, just gradient descent on $V_\theta$), so it is closer in spirit to the heavy-overdamped end of the U-shape than to the interior.
- A 10–30-point penalty is consistent with the gap between $\gamma = 0.30$ (87.06) and $\gamma \in \{2.0, 5.0\}$ (103.82, 121.89) in the E5 sweep itself.

This prediction is a *commitment*, not a threshold. The decision rule of §5 stands regardless of whether the observed $\overline{\Delta}$ falls inside or outside this interval, provided $\overline{\Delta} \ge 5$ and the seed-sign criterion holds.

---

## 7. Sweep specification

The sweep is executed by `notebooks/conservative_arch/first_order_ablation/scripts/run_ablation.sh`. The schedule is:

| Arm | Seeds | Trainer | Fixed $\gamma$ |
|---|---|---|---|
| A — SPLM-1 (first-order) | 0, 1, 2 | `train_splm_first_order.py` | n/a (set to 0, ignored by integrator) |
| B — SPLM em\_ln (second-order) | 0, 1, 2 | `../ln_damping_sweep/train_splm_em_ln.py` | 0.30 |

Per-cell training budget (identical for both arms):
- 4000 steps,
- batch size 16, block size 128,
- AdamW, lr = 5e-4 with cosine decay and 200-step warmup,
- weight\_decay = 0.01, betas = (0.9, 0.95), grad\_clip = 1.0,
- evaluation every 200 steps with `eval_iters = 40`.

Output layout:

```
notebooks/conservative_arch/first_order_ablation/results/
├── splm1/seed0/   splm_first_order_shakespeare_seed0_summary.md
├── splm1/seed1/   splm_first_order_shakespeare_seed1_summary.md
├── splm1/seed2/   splm_first_order_shakespeare_seed2_summary.md
├── splm2_gamma0p30/seed0/   splm_em_ln_shakespeare_seed0_summary.md
├── splm2_gamma0p30/seed1/   splm_em_ln_shakespeare_seed1_summary.md
└── splm2_gamma0p30/seed2/   splm_em_ln_shakespeare_seed2_summary.md
```

Total wall clock: ~30–60 min per cell × 6 cells ≈ 3–6 hours on the local MPS device.

---

## 8. What this experiment is **not** claiming

- **It is not a re-test of the Markov-order question on GPT-2 / Pythia.** That question was settled by the first-order rejection protocol with Outcome C (first-order not excluded for trained transformers), and the paper now positions SPLM as observationally first-order at the inference fixed point. This experiment is about the *training-time graph*, not about the ODE family obeyed by inference trajectories.
- **It is not a calibration of $\beta_{\mathrm{eff}}$.** SPLM-1 is run at the same nominal learning-rate schedule as SPLM em\_ln; we do not perform a learning-rate sweep on SPLM-1. A defender of $H_0$ may legitimately argue (under Outcome A) that an unfair lr pairing biases the result; if Outcome A obtains, the obvious follow-up is exactly such a lr sweep. We do not undertake it pre-emptively because the present experiment is the cheaper of the two and the design that makes $H_1$ falsifiable.
- **It is not a sample-efficiency study.** All cells are trained for exactly 4000 steps. A second follow-up — a step-budget sweep — would test whether SPLM-1's gap closes asymptotically. We pre-register the 4000-step decision rule and leave that follow-up to a later document.

---

## 9. Reporting plan

After the sweep terminates, the analysis writes:

- `notebooks/conservative_arch/first_order_ablation/results/RESULTS.md` — the headline result (per-seed and mean PPL for each arm, $\overline{\Delta}$, Wilcoxon $p$, the locked outcome A / B / C).
- An update to `the paper`:
  - **Outcome A:** keep the "training-time value" paragraph as currently written, append a single sentence citing this protocol's outcome.
  - **Outcome B:** retract the "value-add" wording; rewrite as "the U-shape is consistent with — but does not establish — training-time value of the inertial term", referencing this protocol.
  - **Outcome C:** retract the §12 final paragraph and the §15 "value-add" paragraph; rewrite as "a controlled first-order ablation falsifies the training-time-value claim; the second-order Lagrangian is observationally first-order *and* training-time first-order; SPLM-1 is the architecturally minimal reduction."
- The author commits the SPLM-1 sweep results, the analysis script, and the protocol-outcome statement in the same git commit.

---

## 10. Pre-registered authors' beliefs (separate from the decision rule)

The author of this protocol believes Outcome A is the most likely outcome. This belief does **not** modify the decision rule: $\Delta_{\min} = 5.0$ stands regardless. Recording the belief here makes it a public commitment, so that if Outcome B or C obtains, the post-experiment paper revision is the explicit retraction described in §9 rather than an unflagged re-framing.

---
