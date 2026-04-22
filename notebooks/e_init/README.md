# `notebooks/e_init/` — Scalar, Helmholtz, and gauge-field fits on GPT-2 hidden-state trajectories

This folder contains the **negative-results chain** that appears in §14.1
("Retrospective: five negative experiments on scalar, linear Helmholtz, and
velocity-coupled gauge fits") and Appendix A ("The non-autonomous conservative
framework") of the paper. It systematically closes the natural classical-
Lagrangian menu on pretrained GPT-2 small hidden-state trajectories over the
50-sentence, five-domain E-init corpus, per-sentence per-layer centered, with
a 40-train / 10-test split.

For each ansatz, a damped symplectic-Euler integrator for the Euler–Lagrange
equation is initialised from $(x_{0}, v_{0} = x_{1} - x_{0})$ and evaluated
against the **static null** baseline "freeze at $h_{0}$" using the median
layer-$L$ residual. Every ansatz in the hierarchy below ties or loses to the
static null on held-out data; the chain ends with a *positive* prescriptive
construction reported separately under
[`notebooks/conservative_arch/`](../conservative_arch/).

## The hierarchy, in one table

| Experiment (paper ID) | Script | Ansatz | Result vs. static null |
|---|---|---|---|
| E1 — extended damping | `extended_gamma_and_first_order.py` | Scalar Gaussian well, second-order, $\gamma \in \{0.0, \ldots, 50.0\}$ | saturates at null for all $\gamma$ |
| E2 — first-order overdamped | `extended_gamma_and_first_order.py` | Gradient flow $\dot x = -\eta\nabla V$, $\eta \in \{10^{-4}, \ldots, 10^{-1}\}$ | pinned at null for all $\eta$ |
| E3 — functional-form sweep | `well_functional_form_comparison.py` | seven forms: harmonic, Gaussian, Morse, Lorentzian-saturation, log-saturation, Weibull, power | every form ties null |
| E4 — linear Helmholtz | `helmholtz_curl_augmented.py` | $m\ddot x = -\nabla V(x) + \Omega x - m\gamma \dot x$ with $\Omega = -\Omega^{\top}$ (PCA-$k$) | train $\|\Omega\|$ shrinks to zero; test ties null |
| E5 — velocity-coupled gauge | `velocity_coupled_gauge.py` | $m\ddot x = -\nabla V(x) + F(x)\,\dot x - m\gamma\dot x$, constant / affine-rank-1 / affine-rank-2 $F(x) = -F(x)^{\top}$ | every variant ties null on test |

Each script writes a structured markdown summary, an `.npz` of numerical
results, and one or more `.png` figures to `results/`. The per-layer $R^{2}$
curves and residual-vs-layer plots cited in §14.1 of the paper are the
rendered figures under `results/` with the names shown below.

## Key result files (cited in §14.1 of the paper)

- `well_form_comparison_summary.md` — narrative summary of E3
- `fig_well_form_r2_vs_layer.png`, `fig_well_form_r2_vs_layer_binned.png` — per-layer $R^{2}$ vs. depth for the seven forms
- `fig_well_form_scatter_layer{3,6,9}.png` — $(|x|, \text{NTP loss})$ scatters with fitted well overlays
- `well_form_comparison_r2.csv`, `well_form_comparison_params.json` — raw numbers
- `extended_gamma_first_order_summary.md` — narrative summary of E1 and E2
- `fig_extended_gamma.png`, `fig_first_order_eta.png` — $\gamma$- and $\eta$-sweep residual curves
- `helmholtz_curl_summary.md` — narrative summary of E4
- `fig_helmholtz_residual_vs_gamma.png`, `fig_helmholtz_residual_vs_layer_at_gamma_star.png`
- `velocity_coupled_gauge_summary.md` — narrative summary of E5
- `fig_gauge_residual_vs_gamma.png`, `fig_gauge_residual_vs_layer_at_gamma_star.png`
- `fig_A_residual_vs_layer.png`, `fig_B_residual_vs_logw.png` and their `_ps` variants — per-sentence-centered residual diagnostics from the reference run in `e_init_validation.ipynb`
- `e_init_results.npz`, `e_init_results_ps.npz` — serialized reference run
- `well_params.json`, `well_params_ps.json` — per-layer Gaussian-well parameters

## How to reproduce

Each script is standalone and takes no required arguments; each writes into
`results/` and leaves a markdown summary alongside. Typical runtimes on an
M-series Mac with 16 GB unified memory are 2–10 minutes per script.

```bash
cd notebooks/e_init

# 1. The original §1 E-init reference run (notebook; produces
#    e_init_results*.npz, well_params*.json, fig_A_*.png, fig_B_*.png).
#    Open e_init_validation.ipynb in Jupyter and run all cells.
jupyter lab e_init_validation.ipynb

# 2. Extended-gamma and first-order sweeps (E1, E2 in the paper).
python3 extended_gamma_and_first_order.py

# 3. Functional-form sweep for the scalar potential (E3).
python3 well_functional_form_comparison.py

# 4. Linear Helmholtz position-coupled skew augmentation (E4).
python3 helmholtz_curl_augmented.py

# 5. Velocity-coupled gauge field, constant and position-dependent (E5).
python3 velocity_coupled_gauge.py
```

GPU is not required; everything runs comfortably on CPU / MPS. GPT-2 small is
downloaded on first invocation via the Hugging Face `transformers` library.
The E-init corpus (50 sentences, 5 domains, 10 sentences each) is embedded
directly in `e_init_validation.ipynb`; the companion scripts load the
extracted hidden-state trajectories from the `results/*.npz` artefacts the
notebook produces, so `e_init_validation.ipynb` should be run first.

## Reading order

For a reader who wants to follow the negative-results chain end-to-end:

1. Open `e_init_validation.ipynb` and read the setup and the reference §1
   Gaussian-well residual analysis.
2. Read `extended_gamma_first_order_summary.md` for E1 and E2.
3. Read `well_form_comparison_summary.md` for E3.
4. Read `helmholtz_curl_summary.md` for E4.
5. Read `velocity_coupled_gauge_summary.md` for E5.
6. Proceed to the *positive* control in
   [`../conservative_arch/`](../conservative_arch/) (the
   scalar-potential language model and the three-way shared-potential
   separator).

The theoretical distillation of the chain is presented in
[`companion_notes/The_Failure_of_Conservative_Models_to_explain_hidden_state_trajectories.md`](../../companion_notes/The_Failure_of_Conservative_Models_to_explain_hidden_state_trajectories.md)
(the "why attention is not conservative" argument) and in
[`companion_notes/Considered_Non-Autonomous_Conservative_Mechanisms.md`](../../companion_notes/Considered_Non-Autonomous_Conservative_Mechanisms.md)
(the full candidate hierarchy A–F that motivates the non-autonomous
conservative framework of Appendix A).
