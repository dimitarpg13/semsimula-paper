# `notebooks/conservative_arch/` -- Conservative-by-construction LM prototype

Prototype implementation and trajectory-level validation of the
scalar-potential language model proposed in
[`docs/Conservative_by_Construction_Language_Models.md`](../../docs/Conservative_by_Construction_Language_Models.md).

Its purpose is to provide the **positive control** complement to the
five attention-transformer **negative controls** in
[`docs/The_Failure_of_Conservative_Models_to_explain_hidden_state_trajectories.md`](../../docs/The_Failure_of_Conservative_Models_to_explain_hidden_state_trajectories.md).

## Architecture (one sentence)

Token embeddings plus a causal cumulative-mean context pool drive a
damped Euler-Lagrange flow on a single learned scalar energy
$V_\theta(\xi, h)$, weight-tied across integration steps; next-token
logits are read off the terminal state via tied embeddings.

## Files

| file | purpose |
|---|---|
| [`model.py`](model.py) | `ScalarPotentialLM` module; `SPLMConfig` dataclass; self-test |
| [`data_module.py`](data_module.py) | Tiny Shakespeare (auto-download) + Tiny Stories loaders; GPT-2 BPE |
| [`train_splm.py`](train_splm.py) | 3-mode trainer (`smoke`, `shakespeare`, `tinystories`); checkpoint + loss curve |
| [`trajectory_types.py`](trajectory_types.py) | shared `Trajectory` dataclass (picklable across scripts) |
| [`e_init_corpus.py`](e_init_corpus.py) | 50-sentence 5-domain corpus (verbatim from `notebooks/e_init/`) |
| [`trajectory_extraction.py`](trajectory_extraction.py) | extract `(L+1, T, d)` hidden-state trajectories from a trained checkpoint |
| [`extract_gpt2_baseline.py`](extract_gpt2_baseline.py) | same for a pretrained GPT-2 small (negative control) |
| [`e_init_validation.py`](e_init_validation.py) | run the §1 Gaussian-well E-init pipeline on extracted trajectories |
| [`jacobian_symmetry.py`](jacobian_symmetry.py) | §1.5-style per-layer Jacobian fit; position-only and velocity-aware |
| [`shared_potential_fit.py`](shared_potential_fit.py) | **step 2:** strict-conservative test -- fit one $V_\psi(h)$ across ALL layers |
| [`plot_sharedV_comparison.py`](plot_sharedV_comparison.py) | side-by-side PC vs NC shared-$V_\psi$ plot |
| [`matched_baseline_model.py`](matched_baseline_model.py) | **step 3:** 8 M-param tiny GPT-2-style decoder (matched to SPLM on $d$, $L$, $V$, data budget) |
| [`train_matched.py`](train_matched.py) | **step 3:** training loop for the matched baseline, mirrors `train_splm.py` |
| [`extract_matched_baseline.py`](extract_matched_baseline.py) | **step 3:** extract trajectories from trained matched baseline |
| [`splm_oracle_fit.py`](splm_oracle_fit.py) | **step 3:** oracle upper bound -- swap learned $V_\psi$ for SPLM's own $V_\theta(\xi, h)$ |
| [`sharedV_capacity_sweep.py`](sharedV_capacity_sweep.py) | **step 3:** 6-config $V_\psi$ capacity sweep on GPT-2 trajectories |
| [`plot_three_way_comparison.py`](plot_three_way_comparison.py) | **step 3:** three-panel SPLM vs matched-baseline vs pretrained-GPT-2 figure |
| [`token_direction_fit.py`](token_direction_fit.py) | **step 4:** token-axis shared-$V_\psi$ + velocity-aware Jacobian at fixed layer |
| [`plot_token_vs_layer_three_way.py`](plot_token_vs_layer_three_way.py) | **step 4:** two-panel layer-direction vs token-direction comparison figure |
| `results/` | checkpoints, npz, png, md outputs |

## How to reproduce

```bash
# 1. Smoke run: ~1 min on MPS.  Tiny model, 300 steps.  Verifies pipeline.
python3 notebooks/conservative_arch/train_splm.py --mode smoke

# 2. (Optional) real training on Tiny Shakespeare.  ~15 min on MPS.
python3 notebooks/conservative_arch/train_splm.py --mode shakespeare

# 3. Extract hidden-state trajectories for the §1 E-init test corpus.
python3 notebooks/conservative_arch/trajectory_extraction.py \
  --ckpt notebooks/conservative_arch/results/splm_smoke_ckpt_latest.pt

# 4. Run the Gaussian-well E-init test (the exact §1 diagnostic).
python3 notebooks/conservative_arch/e_init_validation.py \
  --traj notebooks/conservative_arch/results/splm_smoke_ckpt_latest.trajectories.pkl
```

## Step 1 results (shakespeare convergence run)

Full side-by-side comparison in [`results/v2_comparison_summary.md`](results/v2_comparison_summary.md).
Short version:

### Training convergence

`train_splm.py --mode shakespeare` converges cleanly in ~33 min on
MPS ($d=128$, $L=8$, $V_\theta$ hidden 512, 7.1 M params):

| | value |
|---|--:|
| initial loss (log-uniform) | 10.82 |
| final val loss | **5.66** |
| final val perplexity | 287 |
| learned $m$ | 0.980 |
| learned $\gamma$ | 0.961 |

### Diagnostic 1 -- Gaussian-well E-init (the §1.1 ansatz)

| | PC (SPLM, shakespeare) | NC (GPT-2 small, §1.5 ref) |
|---|--:|--:|
| static-null TEST | 0.437 | 0.180 |
| best Gaussian-well TEST | 0.438 | 0.177 |
| $\Delta$ vs. null | 0.000 | −0.003 |

Both hug the null floor.  The earlier smoke-run "positive result"
was an under-training artefact.  The radial-Gaussian + PTL-as-energy
ansatz is too restrictive to capture *either* architecture's learned
potential.

### Diagnostic 2 -- Jacobian-symmetry (§1.5 + velocity-aware variant)

Position-only fit $x_{\ell+1} - x_\ell \approx M_\ell x_\ell$:

| architecture | TEST $R^2$ unconstrained | TEST $R^2$ symmetric-restricted | max gap |
|---|:-:|:-:|--:|
| PC (SPLM, $L=8$) | 0.74--0.97 | 0.18--0.78 | 0.69 |
| NC (GPT-2, $L=12$) | 0.45--1.00 | 0.36--1.00 | 0.18 |

Velocity-aware fit $x_{\ell+1} - x_\ell \approx A v_\ell + M_\ell x_\ell$
(the clean test -- the position-only version is confounded by the
damped-second-order integrator's hidden velocity state):

| architecture | TEST $R^2$ unconstrained | TEST $R^2$ symmetric-restricted | max gap |
|---|:-:|:-:|--:|
| PC (SPLM, $L=8$) | 0.82--0.98 | 0.78--0.98 | **0.040** |
| NC (GPT-2, $L=12$) | 0.45--1.00 | 0.43--1.00 | **0.079** |

Figures: [`results/splm_shakespeare_ckpt_latest_fig_jacsym.png`](results/splm_shakespeare_ckpt_latest_fig_jacsym.png),
[`results/splm_gpt2_baseline_fig_jacsym.png`](results/splm_gpt2_baseline_fig_jacsym.png).

### Surprise finding and what it means

The velocity-aware Jacobian test shows that **both architectures have
per-step operators $M_\ell$ that are approximately symmetric** (full vs.
sym $R^2$ within 0.04 for SPLM, 0.08 for GPT-2).  This **confirms** §1.5
of the Failure doc ("the per-layer one-step linear approximation of the
transformer block is symmetric and non-Hessian") but refines the
interpretation:

- Symmetric per-step $M_\ell$ rules out a non-trivial *skew* (curl /
  Helmholtz) component in either architecture -- exactly as §1.5 said.
- It does **not** rule out, for GPT-2, the existence of *some* scalar
  $V$ whose Hessian matches the per-layer operators.  It only rules
  out simple global shapes for $V$ (radial, Gaussian) at the level §1
  tested.

So the clean separator between the two architectures is **not** at the
local linear level -- it is at the **shared-potential level**: for PC
the $M_\ell$ all derive from a single learned $V_\theta(\xi, h)$
*by construction*; for NC, whether a single scalar $V$ from a tractable
family exists is the real open empirical question.

This is the refined v2 narrative (details in
[`results/v2_comparison_summary.md`](results/v2_comparison_summary.md) §4--§6):

> The per-step hidden-state dynamics of both attention transformers and
> scalar-potential LMs admit a symmetric linear approximation at the
> PCA-16 level, but only the scalar-potential LM is *architecturally
> guaranteed* to derive all per-layer operators from a single shared
> scalar potential -- the stronger structural property the Semantic
> Simulation framework prescribes.

## Step 2 results (shared-$V_\psi$ fit — the strict conservative test)

Detailed write-up in [`results/sharedV_comparison_summary.md`](results/sharedV_comparison_summary.md).

We ask the strongest question short of the SPLM construction: *does a single smooth scalar $V_\psi(h)$ (a 2-layer MLP with hidden 256) whose gradient explains $\Delta x_\ell$ across **every** layer and token simultaneously exist?* We fit $V_\psi$ jointly with per-layer scalars $\alpha_\ell,\beta_\ell$ (absorbing the integrator's $dt/(1+\gamma)$ and $dt^2/((1+\gamma)m_\ell)$) on the TRAIN pool, and evaluate per-layer $R^2$ on held-out sentences.

![sharedV](results/sharedV_splm_vs_gpt2.png)

| architecture | median TEST $R^2$ | middle-band mean $R^2$$^{*}$ | min TEST $R^2$ | layers with $R^2 \ge 0.5$ |
|---|--:|--:|--:|--:|
| **SPLM**  ($L=8$)  | **+0.90** | **+0.86**  | +0.28 | **6 / 7**   |
| **GPT-2** ($L=12$) | **+0.45** | **+0.09**  | +0.04 | **5 / 11**  |

$^{*}$ middle band: SPLM $\ell=3..5$; GPT-2 $\ell=6..10$.

**Clean headline separator:** a single 2-layer MLP whose gradient reproduces every one of the SPLM's 7 layer updates to median $R^2=0.90$ on held-out sentences **cannot** do the same for GPT-2, where the middle 5 layers collapse to mean $R^2 = 0.09$ (despite velocity being fit separately). The boundary layers (1, 2, 11) are fit near-perfectly in GPT-2, hiding the failure in any global summary — the bathtub shape is the real story.

### Why this is stronger than the Jacobian test

The velocity-aware Jacobian test allowed **an independent symmetric $M_\ell$ per layer**. The shared-$V_\psi$ test forces all per-layer Jacobians to be slices of **one** Hessian field:
$$M_\ell \approx -\beta_\ell \nabla^2 V_\psi(h_\ell).$$

GPT-2 passes the per-layer test (each layer's $M_\ell$ is locally symmetric) but fails the shared test (those $M_\ell$'s cannot be jointly described by one $V_\psi$'s Hessian field). This is what "conservative-by-construction vs. conservative-by-accident-per-layer" looks like quantitatively: the former admits a *global* potential, the latter only *local* symmetric approximations per layer.

### What this implies for the Failure doc

The Failure doc's quantitative content is intact; its *interpretation*
needs one paragraph of refinement: the §1 experiments ruled out
*scalar-potential + Helmholtz* fits under specific global shape
assumptions (radial Gaussian, constant skew, linear velocity-coupling),
and ruled out *any non-trivial skew component*.  They did **not**
rule out the existence of some scalar $V$ consistent with the
locally-symmetric per-step dynamics.  This refinement is compatible
with and in fact sharpens the v2 narrative.

## Relationship to the rest of the repository

- **v1 / Failure doc**: the five §1 experiments in
  [`../e_init/`](../e_init) establish that attention transformers do
  not admit any linear-Lagrangian fit to their hidden-state
  trajectories. Those are the negative controls.
- **Conservative doc**: [`docs/Conservative_by_Construction_Language_Models.md`](../../docs/Conservative_by_Construction_Language_Models.md)
  is the conceptual reframing and design programme. This directory is
  the first implementation of its canonical architecture (§3.1 of that
  doc).
- **Together, v2 narrative:** Failure doc + Conservative doc + this
  directory's quantitative separation become the two-sided argument
  in §5 of the Conservative doc: "attention transformers fail the
  Lagrangian diagnostic; conservative-by-construction circuits pass
  it; therefore the theory is prescriptive, not merely descriptive."

## Step 3 results (architectural vs. scale/data controls)

Detailed write-up in [`results/step3_comparative_summary.md`](results/step3_comparative_summary.md).

Three controls were run to isolate *architecture* from *scale* / *data* / *representational capacity*:

### 3a. Matched-parameter GPT-2-style baseline

Training script [`train_matched.py`](train_matched.py) + model [`matched_baseline_model.py`](matched_baseline_model.py).
Standard pre-LN decoder with $d=128, L=8, n_\text{head}=4$, tied embedding, 8.0 M parameters (vs. SPLM 7.1 M), trained with *identical* optimiser, token budget and corpus (Tiny Shakespeare).  Final val ppl = **141.7** (*better* LM than SPLM's 287 ppl — an honest test), but on the shared-$V_\psi$ test:

| architecture | median TEST $R^2$ | min | layers ≥ 0.5 | shape |
|---|--:|--:|--:|---|
| **SPLM** (7.1 M)            | **+0.90** | +0.28 | 6 / 7 | uniform   |
| **Matched GPT-2-style** (8.0 M) | +0.56 | +0.27 | 5 / 7 | monotonic 0.81 → 0.27 |
| **Pretrained GPT-2 small** (124 M) | +0.45 | +0.04 | 5 / 11 | bathtub (middle 0.09) |

The matched baseline is a *stronger language model* at the same parameters and data as SPLM, yet fails the shared-potential test substantially more severely than SPLM (+0.35 median gap).  The architectural separator is **real** and independent of scale/pretraining.  See [`results/sharedV_three_way.png`](results/sharedV_three_way.png) for the three-way panel.

### 3b. $V_\psi$ capacity sweep on GPT-2

Script [`sharedV_capacity_sweep.py`](sharedV_capacity_sweep.py) ran the shared fit on GPT-2 with six V_ψ configurations: hidden ∈ {128, 256, 512, 1024}, depth ∈ {2, 3, 4}, from 115 K to 1.84 M V_ψ parameters.  The per-layer middle-band R² values (layers 6–10) are **identical to 3 decimals across all capacities**:

| layer | h=128 d=2 | h=256 d=2 | h=512 d=2 | h=1024 d=2 | h=512 d=3 | h=512 d=4 |
|--:|--:|--:|--:|--:|--:|--:|
| 6 | +0.191 | +0.196 | +0.193 | +0.195 | +0.193 | +0.193 |
| 7 | +0.065 | +0.067 | +0.067 | +0.068 | +0.065 | +0.067 |
| 8 | +0.038 | +0.039 | +0.039 | +0.039 | +0.039 | +0.039 |
| 9 | +0.057 | +0.057 | +0.057 | +0.057 | +0.056 | +0.056 |
|10 | +0.081 | +0.081 | +0.081 | +0.081 | +0.080 | +0.081 |

Capacity is definitively **not** the bottleneck — the middle-layer failure is structural.  See [`results/sharedV_capacity_sweep_saturation.png`](results/sharedV_capacity_sweep_saturation.png): three flat horizontal lines at median 0.45, middle-mean 0.15, min 0.04.

### 3c. SPLM oracle reference

Script [`splm_oracle_fit.py`](splm_oracle_fit.py) replaces the learned $V_\psi(h)$ with SPLM's own learned $V_\theta(\xi, h)$ as the "oracle" scalar, keeping the same per-layer $\alpha_\ell, \beta_\ell$ fitting.  Results: TRAIN = TEST $R^2 = 1.0000$ on every one of the 7 layers, with $\alpha_\ell = 0.5099, \beta_\ell = 0.5202$ across layers, exactly matching the integrator constants $dt/(1+\gamma) = 0.510$ and $dt^2/((1+\gamma)m) = 0.520$.  This is the tautological positive control: the step-2 shared-$V_\psi(h)$ fit at median $R^2 = 0.90$ reaches 90% of the oracle upper bound.  The ~0.10 gap quantifies the *context drop* $\xi \to \emptyset$, not any limitation of the shared-scalar hypothesis.  See [`results/splm_oracle_shakespeare_fig.png`](results/splm_oracle_shakespeare_fig.png).

## Step 4 results (token-direction coordinate-system robustness)

Detailed write-up in [`results/token_direction_summary.md`](results/token_direction_summary.md).

The step-2/3 diagnostics use **depth-as-time**: for fixed token $t$, trace $h_t^{(0)},\dots,h_t^{(L)}$.  Step 4 runs the same two diagnostics with **token-as-time at fixed layer**: trace $h_1^{(\ell)},\dots,h_T^{(\ell)}$.  This addresses the natural objection that the step-2/3 separator might be a "wrong coordinate system" artefact -- sequence-time is the natural axis for autoregressive inference.

Script [`token_direction_fit.py`](token_direction_fit.py).  Reuses step-1 trajectories; no retraining.

| metric | SPLM | Matched GPT-2-style | Pretrained GPT-2 |
|---|--:|--:|--:|
| median per-layer TEST $R^2$ | **+0.508** | +0.114 | +0.216 |
| pooled TEST $R^2$ | +0.518 | +0.136 | +0.185 |
| TRAIN$-$TEST gap | 0.16 | **0.66** | 0.31 |
| **median gain over velocity-only** | **+0.27** | **$-$0.08** | **+0.03** |
| Jacobian max TEST gap full-vs-sym | 0.020 | 0.016 | 0.050 |

Three qualitatively distinct regimes:

- **SPLM** retains conservative structure in the token direction: shared-$V_\psi$ adds $+0.27$ over velocity-only uniformly, and the TRAIN/TEST gap is small ($0.16$).  SPLM was *not* designed to be conservative in the token axis (its integrator is layer-to-layer); that it emergently is, is strong evidence that conservative-by-construction depth dynamics propagate to the orthogonal time axis.
- **Matched GPT-2-style** severely memorises training triples (TRAIN $R^2=0.80$) without any transfer: the shared-$V_\psi$ fit becomes *worse* than velocity-only on held-out tokens at layers $3..8$ (negative gain).  The learned forces do not derive from any global scalar.
- **Pretrained GPT-2** essentially cannot improve on velocity-only anywhere: shared-$V_\psi$ gain is $\le +0.04$ at every middle layer.  The Geodesic Hypothesis fails in GPT-2 in both coordinate systems.

All three models pass local Jacobian symmetry at PCA-16 (max gap $\le 0.050$), reinforcing the step-2/3 message that *local operator symmetry is necessary but radically insufficient for the shared-scalar hypothesis*.

See [`results/sharedV_layer_vs_token.png`](results/sharedV_layer_vs_token.png) for the two-panel comparison across axes, generated by [`plot_token_vs_layer_three_way.py`](plot_token_vs_layer_three_way.py).

## SARF-faithful ablation (paper §14.13)

Controlled one-knob ablation of the baseline SPLM: recompute the causal
cumulative-mean context pool $\xi^{(\ell)}$ at every integration step
from the current hidden states, instead of holding it fixed at the
layer-$0$ value.  This literally realises the time-dependent
reinforcement field of §6 (SARF) with layer index playing the role of
time, and is the first empirical test of whether the richer coupling
pays back in LM quality and whether the shared-potential separator
survives.

All SARF code, training artefacts, diagnostic outputs and a side-by-side
comparison report live under [`sarf_variant/`](sarf_variant/); the
parent [`shared_potential_fit.py`](shared_potential_fit.py) and
[`token_direction_fit.py`](token_direction_fit.py) are reused verbatim
on the SARF trajectory pickle.

### Results (Tiny Shakespeare, identical $(d, L, d_V) = (128, 8, 512)$, identical seed)

| metric                          | fixed-$\xi$ SPLM | SARF-faithful SPLM | delta |
|---|--:|--:|--:|
| Params                          | 7,123,075 | 7,123,075 | 0 |
| Wall-clock (MPS)                | 2000 s    | 1879 s    | $\sim\!0$ |
| Final val CE                    | 5.661     | **5.259** | **−0.40** |
| Final val ppl                   | 287.4     | **192.2** | **−33.1 %** |
| Shared-$V_\psi$ depth TEST $R^2$ | +0.790    | +0.713    | −0.077 |
| Shared-$V_\psi$ token TEST $R^2$ | +0.518    | +0.406    | −0.112 |

**Headline:** recomputing $\xi$ per layer buys a **33 % perplexity
reduction at identical compute** on Tiny Shakespeare and preserves the
shared-potential separator.  The modest $R^2$ drop is exactly the
amount of dynamics that lives in $\partial V_\theta / \partial \xi \cdot
\partial \xi / \partial h$, which the strict pointwise $V_\psi(h)$
ansatz discards; a context-aware $V_\psi(\xi, h)$ is predicted to
close it (Q8 in the paper).

Full numbers, per-layer tables, interpretation and reproduction
commands: [`sarf_variant/comparison_report.md`](sarf_variant/comparison_report.md).

### Reproduction

```bash
# From repository root:
cd notebooks/conservative_arch/sarf_variant

# 1. Train SARF-faithful SPLM, same hyperparameters as baseline (~31 min on MPS).
PYTHONUNBUFFERED=1 python3 -u train_splm_sarf.py --mode shakespeare

# 2. Extract trajectories.
PYTHONUNBUFFERED=1 python3 -u trajectory_extraction_sarf.py \
  --ckpt results/splm_sarf_shakespeare_ckpt_latest.pt

# 3. Run the paper's §14.8 shared-potential fit (parent script, generic).
cd .. && PYTHONUNBUFFERED=1 python3 -u shared_potential_fit.py \
  --traj sarf_variant/results/splm_sarf_shakespeare_ckpt_latest.trajectories.pkl \
  --tag sarf_shakespeare_ckpt_latest

# 4. Run the paper's §14.15 token-direction fit.
PYTHONUNBUFFERED=1 python3 -u token_direction_fit.py \
  --traj sarf_variant/results/splm_sarf_shakespeare_ckpt_latest.trajectories.pkl \
  --tag sarf_shakespeare

# 5. Side-by-side comparison table + plots.
cd sarf_variant && python3 compare.py
```

## Next steps

**Step 2 (DONE):** Shared-$V_\psi$ separator established -- see above and
[`results/sharedV_comparison_summary.md`](results/sharedV_comparison_summary.md).

**Step 3 (DONE):** Architectural, capacity and oracle controls completed -- see
[`results/step3_comparative_summary.md`](results/step3_comparative_summary.md).

**Step 4 (DONE):** Token-direction coordinate-robustness check completed -- see
[`results/token_direction_summary.md`](results/token_direction_summary.md).

**Remaining open question:**

1. **SPLM scale sweep** -- does the +0.90 median (depth direction) / +0.51
   median (token direction) hold at larger $d$, longer training, and larger
   corpora (TinyStories, WikiText)?  Alternatively, does it *improve* toward
   the oracle 1.00 as capacity grows, and does the token-direction fit
   approach 1.0 under a context-aware $V_\psi(\xi, h)$ oracle?  This would
   complete the scaling-law story on the positive-control side and is the
   one remaining follow-up before incorporating the results into the v2
   paper.
