# SARF-faithful SPLM variant

This subfolder isolates a single-knob ablation of the baseline
Scalar-Potential Language Model (`../model.py`, `../train_splm.py`):

> **Baseline SPLM** computes the causal cumulative-mean context pool
> `xi_t = (1/t) * sum_{s<=t} h_s^{(0)}` once at layer 0 from the token
> embeddings and holds it fixed along the L integration steps.
>
> **SARF-faithful SPLM** (this folder) recomputes
> `xi_t^{(ell)} = (1/t) * sum_{s<=t} h_s^{(ell)}` from the *current*
> hidden states at every layer `ell`, matching the definition in
> `paper_v2/sections/14_conservative_architectures.tex` and the SARF
> (Structure Attractive/Repulsive Force) time-dependent reinforcement
> field from the Semantic Simulation framework.

All other architecture choices (shared `V_theta`, learned scalar mass
and damping, damped Euler-Lagrange integrator, tied-embedding readout,
Adam/cosine LR schedule, hyperparameters) are identical to the baseline,
so any difference in val loss, generation quality, or trajectory-level
diagnostics is attributable to the xi re-pool alone.

## Layout

```
sarf_variant/
  README.md                     <- this file
  model_sarf.py                 <- ScalarPotentialLMSARF, SPLMSARFConfig
  train_splm_sarf.py            <- mirror of ../train_splm.py
  trajectory_extraction_sarf.py <- mirror of ../trajectory_extraction.py
  compare.py                    <- side-by-side baseline vs SARF numbers
  comparison_report.md          <- auto-generated after compare.py runs
  results/
    splm_sarf_shakespeare_ckpt_latest.pt
    splm_sarf_shakespeare_loss_curve.png
    splm_sarf_shakespeare_summary.md
    splm_sarf_shakespeare_training_log.jsonl
    splm_sarf_shakespeare_ckpt_latest.trajectories.pkl
    sharedV_sarf_shakespeare_ckpt_latest_results.npz
    sharedV_sarf_shakespeare_ckpt_latest_fig.png
    sharedV_sarf_shakespeare_ckpt_latest_summary.md
    tokdir_sarf_shakespeare_results.npz
    tokdir_sarf_shakespeare_fig.png
    tokdir_sarf_shakespeare_summary.md
```

## Reproduction

From this folder:

```bash
# 1. Smoke test the model (tiny config, a few steps).
python3 model_sarf.py

# 2. Train SARF-faithful SPLM on Tiny Shakespeare (same config
#    as the baseline; ~30-40 min on MPS / ~10 min on a single GPU).
PYTHONUNBUFFERED=1 python3 -u train_splm_sarf.py --mode shakespeare

# 3. Extract per-sentence trajectories from the trained checkpoint.
PYTHONUNBUFFERED=1 python3 -u trajectory_extraction_sarf.py \
    --ckpt results/splm_sarf_shakespeare_ckpt_latest.pt

# 4. Run the §14.2 strict shared-potential fit (reuses the parent
#    shared_potential_fit.py; generic in the trajectory pickle).
cd .. && PYTHONUNBUFFERED=1 python3 -u shared_potential_fit.py \
    --traj sarf_variant/results/splm_sarf_shakespeare_ckpt_latest.trajectories.pkl \
    --tag sarf_shakespeare_ckpt_latest

# 5. Run the §14.5 token-direction diagnostic (also parent script).
cd .. && PYTHONUNBUFFERED=1 python3 -u token_direction_fit.py \
    --traj sarf_variant/results/splm_sarf_shakespeare_ckpt_latest.trajectories.pkl \
    --tag sarf_shakespeare

# 6. Build the side-by-side comparison table + plots.
cd sarf_variant && python3 compare.py
```

Steps 4--5 write their `sharedV_*` / `tokdir_*` outputs into
`../results/` by default, because the parent scripts use
`Path(__file__).parent / results`.  `compare.py` reads both the baseline
results and the SARF results from there and this folder and consolidates
them into `comparison_report.md`.

## What we expect to find

Hypotheses going into the experiment:

1.  **LM quality (val CE / ppl).**  SARF re-pooling carries more
    information about the evolving state into the force.  At the same
    parameter and compute budget, val loss should match or improve
    slightly; a large regression would indicate optimisation
    instability from the extra non-linearity.
2.  **Conservativity of the dynamics.**  The update rule is still
    `-grad_h V_theta(xi, h)`, so the dynamics remain conservative *at
    fixed xi*.  But xi now depends on h across the sequence, so
    strictly the effective force on `h_t` is no longer the gradient of
    a scalar of `h_t` alone.  We expect the velocity-aware Jacobian
    symmetry test to still return a small full-vs-sym gap (because the
    PCA basis is per-token and the xi-coupling is a soft linear mix
    across tokens), but slightly larger than the baseline.
3.  **Shared-potential fit (§14.2).**  Because the *paper's* update
    rule is conservative, the strict shared-`V_psi` fit should remain
    in the 0.85--0.95 band and beat the velocity-only baseline by a
    clear margin, matching the baseline SPLM result.  A collapse
    toward GPT-2 numbers would be surprising and would imply xi-driven
    path dependence dominates in practice even for SPLM.
4.  **Token-direction fit (§14.5).**  Same prediction as (3) along the
    token axis: `R^2` should remain high because xi is still a smooth
    causal summary.
5.  **Efficiency.**  The wall-clock cost of the extra cumulative sum
    is negligible (`O(T d)` per layer vs `O(T d * v_hidden)` for the
    V-MLP), so training time should be within a few percent of the
    baseline.
