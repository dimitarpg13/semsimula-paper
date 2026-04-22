# SPLM oracle fit -- shakespeare

**Purpose.**  Upper-bound reference for the step-2 shared-$V_\psi$ fit on SPLM.  Replaces the learned $V_\psi(h)$ with SPLM's own $V_\theta(\xi, h)$ and keeps the same per-layer $\alpha_\ell, \beta_\ell$ fitting procedure.  Numerical mismatch from 1.0 is then purely due to integrator constants and numerical precision.

## Per-layer $R^2$  (oracle $V_\theta$)

| layer | TRAIN | TEST | $\alpha_\ell$ | $\beta_\ell$ |
|--:|--:|--:|--:|--:|
| 1 | +1.0000 | +1.0000 | +0.5099 | +0.5202 |
| 2 | +1.0000 | +1.0000 | +0.5099 | +0.5202 |
| 3 | +1.0000 | +1.0000 | +0.5099 | +0.5202 |
| 4 | +1.0000 | +1.0000 | +0.5099 | +0.5202 |
| 5 | +1.0000 | +1.0000 | +0.5099 | +0.5202 |
| 6 | +1.0000 | +1.0000 | +0.5099 | +0.5202 |
| 7 | +1.0000 | +1.0000 | +0.5099 | +0.5202 |

## Oracle vs. learned $V_\psi$ (step-2)

| layer | oracle TEST | learned $V_\psi$ TEST | gap |
|--:|--:|--:|--:|
| 1 | +1.0000 | +0.9695 | +0.0305 |
| 2 | +1.0000 | +0.6677 | +0.3323 |
| 3 | +1.0000 | +0.8195 | +0.1805 |
| 4 | +1.0000 | +0.2805 | +0.7195 |
| 5 | +1.0000 | +0.9014 | +0.0986 |
| 6 | +1.0000 | +0.9269 | +0.0731 |
| 7 | +1.0000 | +0.9053 | +0.0947 |

![fig](splm_oracle_shakespeare_fig.png)
