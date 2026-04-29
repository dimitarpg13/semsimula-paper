# Decision table — Pythia-160 m (replication)

> Generated automatically from `results/pythia/primary_summary.json` and the
> §6.4 pre-registered decision matrix
> (`docs/first_order_ODE_rejection_pre-registered_protocol.md`).

## Setup

| Field | Value |
|---|---|
| Model | `EleutherAI/pythia-160m` (162 M params, 12 layers, $h \in \mathbb R^{768}$) |
| Hidden layer probed | last (layer 12) |
| Corpus | 50 sentences × 5 domains (`data/corpus.json`) — identical to GPT-2 cell |
| Quadruples extracted | **1 219** inside-sentence $(h_{t-2}, h_{t-1}, h_t, h_{t+1})$ |
| PCA dim $p$ | **50** |
| Function class | Kernel ridge regression (RBF), $\alpha$ and $\gamma$ inner-CV-selected |
| Outer CV | Leave-one-sentence-out (50 folds) |
| Inner CV | 5-fold, $\alpha \in \{10^{-3}, ..., 10\}$, $\gamma \in \{0.5, 1, 2\}\times\gamma_{med}$ |
| Bootstrap | Sentence-cluster percentile, $B = 10000$ |
| Runtime | 144 s |

## Mean per-quadruple residuals

| $k$ | $\bar R_k$ | sem |
|---|---|---|
| 1 | **282.16** | 6.78 |
| 2 | **286.44** | 6.82 |
| 3 | **290.31** | 6.93 |

The same ordering as on GPT-2: $\bar R_1 < \bar R_2 < \bar R_3$. The
absolute scale is ~10× smaller than GPT-2 because Pythia's last-layer
hidden states have substantially smaller norms (Pythia uses pre-LN with
parallel attention/MLP and has a different residual-stream scaling than
GPT-2).

## Effect-size ratios and statistical tests

| Quantity | Value | Pre-reg threshold (§6.4 row A) | Pass? |
|---|---|---|---|
| $\rho_{12} = \bar R_1 / \bar R_2$ | **0.9850** | $\ge 1.20$ | **NO** |
| Wilcoxon $p_{12}$ (two-sided) | **$7.94\times 10^{-7}$** | $< 10^{-3}$ — **but in the wrong direction** | NO (direction fails) |
| Wilcoxon $p_{12}$ (one-sided $R_1 > R_2$) | 1.0 | — | — |
| Wilcoxon $p_{12}$ (one-sided $R_1 < R_2$) | $3.97\times 10^{-7}$ | — | — |
| 95 % cluster-bootstrap CI for $\bar R_1 - \bar R_2$ | $[-6.06, -2.50]$ | strictly excludes $0$ negatively | — |
| $\rho_{23} = \bar R_2 / \bar R_3$ | 0.9867 | $\le 1.05$ | (passes) |
| Wilcoxon $p_{23}$ (two-sided) | $7.14\times 10^{-6}$ | $> 0.05$ | **NO** |
| 95 % cluster-bootstrap CI for $\bar R_2 - \bar R_3$ | $[-5.20, -2.60]$ | excludes $0$ negatively | — |

The $\rho_{12}$ effect-size threshold for outcome A or B (≥ 1.20) is not
met (it is below 1.0), and although $p_{12}$ achieves astronomical
significance, the Wilcoxon test signs the difference $R_1 < R_2$, i.e.
the **direction opposite** to the framework's hypothesis. The
cluster-bootstrap CI confirms this with $[-6.06, -2.50]$.

## Decision (per §6.4)

> **Outcome C — first-order not rejected.**

Pythia replication confirms the GPT-2 negative result with much tighter
statistics. The Pythia residuals are ~10× smaller in absolute value, so the
same per-quadruple effect ($\bar R_2 - \bar R_1 \approx 4$) is a much
larger fraction of $\bar R_1$ in the variance-normalised sense, allowing
the Wilcoxon test to reach $7.9\times 10^{-7}$. But because the *direction*
is wrong relative to the framework's hypothesis, the locked decision matrix
(§6.4) returns C, not A or B. This is the architecturally-independent half
of the headline conclusion.

## Selected hyperparameters

The Pythia inner CV selected $\alpha = 0.1$ at all three $k$ values (same
as GPT-2), with $\gamma$ values scaling as expected with the median
pairwise distance in the projected space.

## Physical interpretation

Outcome C does **not** imply that the second-order Lagrangian framework
is wrong. It implies the framework is in the **overdamped regime**.
Concretely, the full Euler–Lagrange equation (paper Eq. 67),
$w_t\ddot{h}\_t + \gamma(h_t)\dot{h}\_t = -\nabla V(h_t)$,
in the limit $\gamma \gg \omega_0$ collapses to
$\dot{h} \approx -\nabla V / \gamma$,
a first-order gradient flow on the same potential. The two are
observationally indistinguishable at one-token resolution. The variant
this test rejects is the *underdamped* second-order ODE — the one in
which a velocity slot would carry detectable predictive information
beyond what $h_t$ already encodes — not the Lagrangian generative
account. The Pythia replication tightens the GPT-2 finding (much smaller
sem; Wilcoxon $p_{12}$ at $7.9\times 10^{-7}$ in the *opposite* direction
to $H_1$), so the overdamped reading is also architecture-independent.
See [`results/RESULTS.md`](../RESULTS.md) §"What the test rules out vs.
what remains compatible — the overdamped synthesis" for the full
discussion.

## Files

- `results/pythia/quadruples.npz` — extracted lagged quadruples.
- `results/pythia/extraction_summary.json`.
- `results/pythia/primary_residuals.npz`.
- `results/pythia/primary_summary.json`.
- `results/pythia/figures/Rk_bars.png`, `paired_scatter.png`, `loso_spaghetti.png`, `robustness_rho12_*.png`.
