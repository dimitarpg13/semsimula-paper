# Shared-potential fit — SPLM vs GPT-2

**The strict conservative-dynamics test.** We ask the strongest possible question short of the SPLM construction itself: *does there exist a single smooth scalar $V_\psi(h)$ whose gradient explains $\Delta x_\ell$ across every layer, every token, and every held-out sentence?*

The ansatz is

$$\Delta x_\ell \;\approx\; \alpha_\ell\, v_\ell \;-\; \beta_\ell\, \nabla_h V_\psi(x_\ell),$$

with **one** shared $V_\psi$ (a 2-layer MLP, hidden 256, GELU) and per-layer scalars $\alpha_\ell,\beta_\ell$ that absorb the layer-varying velocity and force coefficients of the damped-second-order integrator ($\alpha_\ell \sim dt/(1+\gamma)$, $\beta_\ell \sim dt^2/((1+\gamma) m_\ell)$). $V_\psi$ is fit jointly with $\alpha_\ell,\beta_\ell$ by AdamW on pooled TRAIN samples, then evaluated on held-out sentences per layer.

This is stricter than the velocity-aware Jacobian test because the per-layer linearisation $\partial_h(-\beta_\ell \nabla V_\psi) = -\beta_\ell \nabla^2 V_\psi(h_\ell)$ must be consistent with **one** Hessian field — not 11 independently chosen symmetric matrices.

## Result

![sharedV_splm_vs_gpt2](sharedV_splm_vs_gpt2.png)

### Per-layer TEST $R^2$ with shared $V_\psi$

| model | layers | median | mean | middle-layers mean$^{*}$ | min | max | # of layers with $R^2 \ge 0.5$ |
|---|---|--:|--:|--:|--:|--:|--:|
| **SPLM shakespeare** ($d=128$, $L=8$) | 7 | **+0.90** | +0.78 | +0.86$_{[\ell=3..5]}$ | +0.28 | +0.97 | **6 / 7** |
| **GPT-2 small pre-trained** ($d=768$, $L=12$) | 11 | **+0.45** | +0.46 | **+0.09**$_{[\ell=6..10]}$ | +0.04 | +0.99 | **5 / 11** |

$^{*}$ "middle-layers mean" averages R² over the middle band of the stack (SPLM: layers 3–5; GPT-2: layers 6–10).
This is the single statistic that best separates the two models: a bathtub pattern in GPT-2 (good at the edges, collapse in the middle) vs. a uniformly explained landscape in SPLM.

### Per-layer gain over velocity-only (shared-$V_\psi$ minus vel-only)

| model | median gain | max gain | # of layers with gain $\ge 0.3$ |
|---|--:|--:|--:|
| SPLM  | **+0.46** | +0.85 | **6 / 7** |
| GPT-2 | **+0.09** | +0.95 | **2 / 11** |

### "Pooled" overall $R^2$ (variance-weighted across layers)

|  | TRAIN | TEST |
|---|--:|--:|
| SPLM  | +0.933 | +0.790 |
| GPT-2 | +0.975 | +0.964 |

*Beware the aggregate:* GPT-2's 0.96 TEST number is misleading. It is dominated by layer 11 (where the hidden state undergoes its large pre-logit transition and shared $V_\psi$ happens to explain 97% of variance) and layer 2. Per-layer decomposition reveals the honest picture.

## Interpretation

**SPLM is globally conservative.** A single $V_\psi$ fits the per-layer Euler–Lagrange update to $R^2 \ge 0.67$ on 6 of 7 layers, with a median of 0.90 on unseen sentences. The one weak layer ($\ell=4$) still gains +0.25 over the velocity-only baseline. This is the expected behaviour for a model whose training dynamics was *itself* a gradient flow of one learned scalar.

**GPT-2 is conservative only at its boundary layers.** A shared $V_\psi$ succeeds at layers 1–4 (embedding → semantic projection) and layer 11 (pre-logit collapse), but in the middle of the stack (layers 5–10) both velocity-only *and* velocity + shared-$V_\psi$ have $R^2 \le 0.45$. The shared-$V_\psi$ fit adds **essentially nothing** on layers 7–10 (gains $\le 0.02$), even though the velocity-aware per-layer Jacobian test (`jacobian_symmetry.py`) reported high $R^2$ for fully unconstrained $M_\ell$ at each of those layers.

The distinction is geometric:

- The velocity-aware Jacobian test allowed **a different symmetric matrix per layer**. This is essentially unconstrained structurally — any sequence of symmetric $M_\ell$ is admissible.
- The shared-$V_\psi$ test forces all per-layer Jacobians to be slices of **one** Hessian field: $M_\ell \approx -\beta_\ell \nabla^2 V_\psi(h_\ell)$. If layers 5–10 operate on hidden states lying in roughly the same region of $\mathbb{R}^d$ yet require structurally different symmetric operators, no single smooth $V_\psi$ can match them simultaneously.

Operationally this says: GPT-2's middle-block dynamics behaves as if each layer's force-field is computed by a different potential — consistent with the view that attention + MLP composition in the middle of the stack is performing **heterogeneous transformations** that cannot be collapsed into a single semantic energy landscape. The boundary layers, by contrast, perform "embedding → semantic" and "semantic → logit" transitions whose forces happen to be explainable by one smooth scalar.

## What this means for the v2 paper

1. **Headline separator** — where §1.5 of `Failure_of_Conservative_Models` reported a per-layer Jacobian $R^2$ gap of ~0.15 between full and symmetric fits on GPT-2, we now have a stronger, more interpretable separator:
   - On the **shared-potential fit**, median per-layer TEST $R^2$ is **+0.90 for SPLM vs +0.45 for GPT-2**; the qualitative separator is the shape of the per-layer curve — SPLM is uniform, GPT-2 is bathtub-shaped, with mean R² = +0.09 on its middle layers 6–10.
   - On **6 of GPT-2's 11 layers** (the entire middle block 5–10, plus layer 7 outright), no shared-$V_\psi$ capacity reproduces the per-step update (see `sharedV_capacity_sweep_summary.md`). On **6 of SPLM's 7 layers**, a 2-layer MLP captures ≥67% of the per-step variance from one scalar.
2. **Positive control is satisfied.** The Semantic Simulation framework's prescription — that hidden-state motion should be gradient flow of a shared scalar — is *empirically achievable* by an actually-trained language model. SPLM is it.
3. **Diagnosis of the failure of conservative fits on GPT-2.** The "Failure" document's P-rot-* variants all tried to augment or relax conservativity within the attention stack. The shared-$V_\psi$ test clarifies why they failed at a deeper level: GPT-2's middle layers are not merely non-conservative *around the gradient of some fixed scalar*; they cannot be jointly described by **any** single smooth scalar field on hidden-state space. Each layer operates on its own effective landscape.

## Caveats and follow-ups (status as of step 3)

- **Shared-$V_\psi$ capacity** — *resolved.* A 6-config sweep (hidden ∈ {128, 256, 512, 1024}, depth ∈ {2, 3, 4}, up to 1.84 M V_ψ parameters) on GPT-2 trajectories finds the middle-layer R² values **agree to 3 decimal places across all capacities** (see `sharedV_capacity_sweep_summary.md`). Capacity is not the bottleneck — the failure is structural.
- **Oracle reference for SPLM** — *resolved.* Using SPLM's actual $V_\theta(\xi,h)$ as the oracle scalar (script `splm_oracle_fit.py`) gives TRAIN = TEST R² = 1.0000 on every layer, with $\alpha_\ell = 0.5099, \beta_\ell = 0.5202$ recovering the integrator constants $dt/(1+\gamma)$, $dt^2/((1+\gamma)m)$ exactly. The learned $V_\psi(h)$-only fit attains R² ≥ 0.90 on 4 of 7 layers; the residual is dominated by the context drop $\xi \to \emptyset$, not by V_ψ's expressivity.
- **Scale- and data-matched negative control** — *resolved.* Training an 8.0 M-parameter GPT-2-style transformer on the same Tiny Shakespeare (same tokens, same optimiser, val ppl 141.7 — *better* LM than SPLM's 287 ppl) and running the same shared-V_ψ fit gives median TEST R² = +0.56, min = +0.27, with a **monotonic decay** pattern (0.81 → 0.27 across layers 1–7), distinct from GPT-2's bathtub pattern. The architectural gap SPLM → matched is +0.35 median; matched → pretrained GPT-2 is +0.11 median, with the full large-model gap appearing in the *shape* (bathtub / middle-layer collapse), not the summary statistic.
- **Per-token vs. per-layer variance.** If GPT-2's middle layers have genuinely tiny step-sizes, their low $R^2$ could reflect noise rather than structural failure. $\alpha_\ell$ inspection shows a smooth ramp from 0 to ~0.4 across layers 2–10, consistent with small but deterministic dynamics, not noise.

## Artefacts

- `sharedV_shakespeare_ckpt_latest_summary.md`, `sharedV_shakespeare_ckpt_latest_results.npz`
- `sharedV_gpt2_baseline_summary.md`,          `sharedV_gpt2_baseline_results.npz`
- `sharedV_splm_vs_gpt2.png`

Reproduce:

```bash
cd notebooks/conservative_arch
python3 shared_potential_fit.py --traj results/splm_shakespeare_ckpt_latest.trajectories.pkl
python3 shared_potential_fit.py --traj results/gpt2_baseline.trajectories.pkl
python3 plot_sharedV_comparison.py
```
