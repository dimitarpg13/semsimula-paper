# SPLM-1 first-order ablation

This directory hosts the **SPLM-1 ablation**: a controlled comparison
between the second-order SPLM em\_ln baseline at the E5 winner $\gamma^{\ast}=0.30$
and a structurally first-order variant of the same architecture (no
velocity buffer, no damping coefficient, just gradient flow on $V_\theta$).

The full pre-registered protocol — including the locked decision rule,
effect-size threshold ($\Delta_{\min} = 5.0$ PPL), and the predicted
outcome — lives in
[`companion_notes/SPLM-1_ablation_pre-registered_protocol.md`](../../../companion_notes/SPLM-1_ablation_pre-registered_protocol.md).
The architectural design note that motivated the experiment is
[`companion_notes/Replacing_The_Conservative_Mechanism_of_SPLM_with_First_Order.md`](../../../companion_notes/Replacing_The_Conservative_Mechanism_of_SPLM_with_First_Order.md).

## Files

| Path | Role |
|---|---|
| `model_first_order.py` | `ScalarPotentialLMFirstOrder` — subclass of `ScalarPotentialLMSARFMassLN` that overrides `integrate()` with the first-order step $h_{l+1} = \mathrm{LN}(h_l - \mathrm{d}t \cdot \nabla V_\theta / m)$. No $v$, no $\gamma$. |
| `train_splm_first_order.py` | Training loop, drop-in delta of `../ln_damping_sweep/train_splm_em_ln.py`. Accepts `--fixed-gamma` for CLI compatibility but the integrator does not consult it. |
| `scripts/run_ablation.sh` | Six-cell sweep: arm A = SPLM-1 × {seed 0, 1, 2}; arm B = SPLM em\_ln $\gamma=0.30$ × {seed 0, 1, 2}. Resilient to per-cell crashes. |
| `results/splm1/seed{0,1,2}/` | Arm A outputs (one directory per seed). |
| `results/splm2_gamma0p30/seed{0,1,2}/` | Arm B outputs (one directory per seed). |

## Layer update — exact form

Second-order baseline (E5 winner, arm B):

```text
v_{l+1} = (v_l + dt · f / m) / (1 + dt · γ)
h_{l+1} = LN( h_l + dt · v_{l+1} )                with γ = 0.30 fixed
```

First-order ablation (arm A):

```text
h_{l+1} = LN( h_l - dt · ∇V_θ(ξ_l, h_l) / m )    no v, no γ
```

Everything else — $V_\theta$, $\xi_t$, the per-token logfreq mass, LayerNorm-after-step,
the tied-embedding readout, the loss, the optimiser, the schedule — is
identical between the two arms.

## How to run

```bash
cd notebooks/conservative_arch/first_order_ablation
bash scripts/run_ablation.sh
```

Wall-clock: ~30–60 min per cell × 6 cells ≈ 3–6 hours on a single MPS / 16-core CPU.

## Smoke test

```bash
cd notebooks/conservative_arch/first_order_ablation
python3 model_first_order.py
python3 train_splm_first_order.py --mode smoke --seed 0 --tag-suffix smoke
```

The model smoke verifies (i) no velocity buffer is present in the state
dict, (ii) `gamma_value` is `0.0`, and (iii) LayerNorm-after-step
produces unit-RMS hidden states. The trainer smoke runs 300 optimisation
steps in ≈ 40 s on MPS.
