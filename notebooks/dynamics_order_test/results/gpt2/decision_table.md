# Decision table — GPT-2 small (primary)

> Generated automatically from `results/gpt2/primary_summary.json` and the
> §6.4 pre-registered decision matrix
> (`docs/first_order_ODE_rejection_pre-registered_protocol.md`).

## Setup

| Field | Value |
|---|---|
| Model | `gpt2` (124 M params, 12 layers, $h \in \mathbb R^{768}$) |
| Hidden layer probed | last (layer 12) |
| Corpus | 50 sentences × 5 domains (`data/corpus.json`) |
| Quadruples extracted | **1 264** inside-sentence $(h_{t-2}, h_{t-1}, h_t, h_{t+1})$ |
| PCA dim $p$ | **50** (basis fit on training fold's $h_{t+1}$, applied to all four lags) |
| Function class | Kernel ridge regression (RBF), $\alpha$ and $\gamma$ inner-CV-selected |
| Outer CV | Leave-one-sentence-out (50 folds) |
| Inner CV | 5-fold on training quadruples, $\alpha \in \{10^{-3}, ..., 10\}$, $\gamma \in \{0.5, 1, 2\}\!\times\!\gamma_{med}$ |
| Bootstrap | Sentence-cluster percentile, $B = 10\,000$ resamples |
| Runtime | 174 s on 16-core CPU (parallelised across LOSO folds) |

## Mean per-quadruple residuals

| $k$ | $\bar R_k$ | sem |
|---|---|---|
| 1 | **2 799.24** | 129.78 |
| 2 | **2 860.15** | 129.04 |
| 3 | **2 922.10** | 136.45 |

The ordering $\bar R_1 < \bar R_2 < \bar R_3$ is the *opposite* of what
hypothesis $H_1$ (and a fortiori $H_2$) predicts under the second-order ODE
account.

## Effect-size ratios and statistical tests

| Quantity | Value | Pre-reg threshold (§6.4 row A) | Pass? |
|---|---|---|---|
| $\rho_{12} = \bar R_1 / \bar R_2$ | **0.9787** | $\ge 1.20$ | **NO** |
| Wilcoxon $p_{12}$ (two-sided, §6.2) | **0.124** | $< 10^{-3}$ | **NO** |
| Wilcoxon $p_{12}$ (one-sided $R_1 > R_2$) | 0.938 | — | — |
| Wilcoxon $p_{12}$ (one-sided $R_1 < R_2$) | 0.062 | — | — |
| 95 % cluster-bootstrap CI for $\bar R_1 - \bar R_2$ | $[-123.08,\, -0.56]$ | strictly excludes $0$ on the **negative** side | — |
| $\rho_{23} = \bar R_2 / \bar R_3$ | 0.9788 | $\le 1.05$ (row A) | (passes) |
| Wilcoxon $p_{23}$ (two-sided) | **0.0086** | $> 0.05$ (row A) | **NO** |
| Wilcoxon $p_{23}$ (one-sided $R_2 > R_3$) | 0.996 | — | — |
| Wilcoxon $p_{23}$ (one-sided $R_2 < R_3$) | 0.0043 | — | — |
| 95 % cluster-bootstrap CI for $\bar R_2 - \bar R_3$ | $[-133.41,\, -1.69]$ | excludes $0$ negatively | — |

The $\rho_{12}$ effect-size threshold is **not met**, the Wilcoxon
two-sided $p_{12}$ does not reach the $10^{-3}$ threshold, and the bootstrap
CI excludes zero in the *opposite* direction from what the second-order
hypothesis would predict. Both row-A and row-B require $\rho_{12} \ge 1.20$,
so neither row applies.

## Decision (per §6.4)

> **Outcome C — first-order not rejected.**

The conditions for outcome A (first-order rejected, second-order sufficient)
and outcome B (first-order rejected, second-order also insufficient) both
require $\rho_{12} \ge 1.20$ and $p_{12} < 10^{-3}$ in the primary cell.
Neither condition is met. Outcome D (boundary / inconclusive) does not apply
because $\rho_{12} < 1.10$ AND $p_{12} > 0.01$, which is the explicit
trigger for outcome C in the pre-registered matrix.

A particularly telling diagnostic is that the cluster-bootstrap CI for
$\bar R_1 - \bar R_2$ is $[-123.08, -0.56]$. This means: across the 50
held-out sentences, the lag-1 kernel ridge predictor is — robustly,
significantly — *better* than the lag-2 kernel ridge predictor. Adding
$h_{t-1}$ to the input feature set strictly *worsens* the prediction of
$h_{t+1}$ in the kernel-ridge function class with PCA-50.

## Required confirmations (per §6.5)

| Confirmation | Status |
|---|---|
| Architecture independence (Pythia-160 m, primary cell) | C ✓ (see `results/pythia/decision_table.md`) |
| Function-class robustness (3 of 4 classes give same decision) | C ✓ (3/4 agree at $p=50$; only poly-2 dissents — and only by over-fitting at higher $p$, see `results/RESULTS.md`) |
| PCA-dim robustness ($p \in \{20, 50, 100\}$) | C ✓ (kernel ridge cell on GPT-2 gives C at every $p$) |

All required confirmations are met; the headline outcome **stays at C** and
is not downgraded.

## Selected hyperparameters

| $k$ | $\alpha$ (inner-CV-selected) | $\gamma$ |
|---|---|---|
| 1 | 0.1 (median across folds) | $\sim 6.6 \cdot 10^{-5}$ |
| 2 | 0.1 | $\sim 2.7 \cdot 10^{-5}$ |
| 3 | 0.1 | $\sim 1.6 \cdot 10^{-5}$ |

The selected $\gamma$ falls with $k$ in line with the median pairwise
distance growing with the input dimension — the inner CV adapted bandwidth
correctly, ruling out a "wrong $\gamma$ at high $k$" explanation.

## Physical interpretation

Outcome C does **not** imply that the second-order Lagrangian framework
is wrong. It implies the framework is in the **overdamped regime**.
Concretely, the full Euler–Lagrange equation (paper Eq. 67),
$w_t\,\ddot h_t + \gamma(h_t)\,\dot h_t = -\nabla V(h_t)$,
in the limit $\gamma \gg \omega_0$ collapses to
$\dot h \approx -\nabla V / \gamma$,
a first-order gradient flow on the same potential. The two are
observationally indistinguishable at one-token resolution. The variant
this test rejects is the *underdamped* second-order ODE — the one in
which a velocity slot would carry detectable predictive information
beyond what $h_t$ already encodes — not the Lagrangian generative
account. See
[`results/RESULTS.md`](../RESULTS.md) §"What the test rules out vs. what
remains compatible — the overdamped synthesis" for the full discussion.

## Files

- `results/gpt2/quadruples.npz` — extracted lagged quadruples.
- `results/gpt2/extraction_summary.json` — extraction metadata.
- `results/gpt2/primary_residuals.npz` — per-quadruple squared residuals for $k\in\{1,2,3\}$ across all 50 LOSO folds.
- `results/gpt2/primary_summary.json` — full numerical summary, this table's source.
- `results/gpt2/figures/Rk_bars.png`, `paired_scatter.png`, `loso_spaghetti.png`, `robustness_rho12_*.png`.
