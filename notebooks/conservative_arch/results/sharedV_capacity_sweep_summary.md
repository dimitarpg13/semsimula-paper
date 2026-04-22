# Shared-$V_\psi$ capacity sweep on GPT-2 trajectories

**Question.** Is the step-2 middle-layer failure on GPT-2 (layers 5--10 with per-layer TEST $R^2 \le 0.2$) a representational limit of a 2-layer hidden-256 MLP, or a structural fact about GPT-2's per-layer operators?

**Method.** Re-run the step-2 shared-$V_\psi$ fit with a sequence of MLP capacities while holding all other hyperparameters fixed.  3000 AdamW steps per run.

## Configuration sweep

| hidden | depth | params | median TEST $R^2$ | mean TEST $R^2$ (layers 5--10) | min TEST $R^2$ |
|--:|--:|--:|--:|--:|--:|
| 128 | 2 | 0.12 M | +0.447 | +0.147 | +0.038 |
| 256 | 2 | 0.26 M | +0.457 | +0.149 | +0.039 |
| 512 | 2 | 0.66 M | +0.451 | +0.148 | +0.039 |
| 1024 | 2 | 1.84 M | +0.454 | +0.149 | +0.039 |
| 512 | 3 | 0.92 M | +0.451 | +0.147 | +0.039 |
| 512 | 4 | 1.18 M | +0.453 | +0.148 | +0.039 |

## Per-layer TEST $R^2$

| layer | h128 d2 | h256 d2 | h512 d2 | h1024 d2 | h512 d3 | h512 d4 |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | +0.562 | +0.776 | +0.815 | +0.917 | +0.965 | +0.971 |
| 2 | +0.978 | +0.981 | +0.997 | +0.997 | +0.994 | +0.997 |
| 3 | +0.784 | +0.783 | +0.798 | +0.799 | +0.789 | +0.799 |
| 4 | +0.727 | +0.730 | +0.731 | +0.732 | +0.728 | +0.731 |
| 5 | +0.447 | +0.457 | +0.451 | +0.454 | +0.451 | +0.453 |
| 6 | +0.191 | +0.196 | +0.193 | +0.195 | +0.193 | +0.193 |
| 7 | +0.065 | +0.067 | +0.067 | +0.068 | +0.065 | +0.067 |
| 8 | +0.038 | +0.039 | +0.039 | +0.039 | +0.039 | +0.039 |
| 9 | +0.057 | +0.057 | +0.057 | +0.057 | +0.056 | +0.056 |
| 10 | +0.081 | +0.081 | +0.081 | +0.081 | +0.080 | +0.081 |
| 11 | +0.966 | +0.975 | +0.978 | +0.978 | +0.972 | +0.978 |

## Plots

![per-layer](sharedV_capacity_sweep_per_layer.png)

![saturation](sharedV_capacity_sweep_saturation.png)

