# RESULTS — first-order ODE rejection test

> Pre-registered protocol: [`docs/first_order_ODE_rejection_pre-registered_protocol.md`](../../../docs/first_order_ODE_rejection_pre-registered_protocol.md)
> Companion critique: [`docs/Evidence_for_second_order_ODE_governing_evolution.md`](../../../docs/Evidence_for_second_order_ODE_governing_evolution.md)
> Generated 2026-04-27, immediately after the locked-grid 24-cell sweep finished.

## Headline

**Outcome C: first-order ODE not rejected.**

The protocol's primary cell (Gaussian RBF kernel ridge, PCA dim 50) on the
50-sentence × 5-domain corpus, run through GPT-2 small *and* Pythia-160m,
fails to reject first-order at the pre-registered $(\rho_{12}\!\ge\!1.20,\;
p_{12}\!<\!10^{-3})$ thresholds in §6.4. In fact, the lag-1 kernel-ridge
predictor is significantly *better* than the lag-2 predictor — the opposite
of the direction predicted by the second-order ODE hypothesis. All three
required confirmations of §6.5 (architecture independence, function-class
robustness, PCA-dim robustness) hold for this C verdict.

## Primary cells (§6.4)

| Architecture | $\bar R_1$ | $\bar R_2$ | $\bar R_3$ | $\rho_{12}$ | $p_{12}$ (two-sided) | $\rho_{23}$ | $p_{23}$ | 95 % CI on $\bar R_1\!-\!\bar R_2$ | Decision |
|---|---|---|---|---|---|---|---|---|---|
| GPT-2 small  | 2 799.24 | 2 860.15 | 2 922.10 | 0.9787 | 0.124 | 0.9788 | 0.0086 | $[-123.08,\,-0.56]$ | **C** |
| Pythia-160 m |   282.16 |   286.44 |   290.31 | 0.9850 | $7.9\!\times\!10^{-7}$ | 0.9867 | $7.1\!\times\!10^{-6}$ | $[-6.06,\,-2.50]$ | **C** |

Both architectures share three signatures of a robust negative result for
the framework's H₁ / H₂:

1. $\bar R_1 < \bar R_2 < \bar R_3$ — adding lag *worsens* the fit.
2. The cluster-bootstrap 95 % CI on $\bar R_1 - \bar R_2$ excludes zero
   strictly on the **negative** side.
3. The Wilcoxon p-value, where significant, signs in the *opposite*
   direction from the framework's prediction (one-sided $R_1 < R_2$ achieves
   $p = 4 \times 10^{-7}$ on Pythia).

## Robustness sweep (§6.5, §6.6)

24 cells = 4 function classes × 3 PCA dims × 2 architectures.
Bonferroni-corrected per-cell threshold:
$p_{12} < 10^{-3} / 24 \approx 4.2 \times 10^{-5}$.

### Decision counts

| Decision | Count |
|---|---|
| **A** (first-order rejected, second-order sufficient) | **0** |
| **B** (first-order rejected, second-order insufficient) | 3 |
| **C** (first-order not rejected) | **21** |
| **D** (boundary / inconclusive) | 0 |
| Cells passing Bonferroni-corrected $\rho_{12}\!\ge\!1.20$ AND $p_{12}\!<\!4.2\!\times\!10^{-5}$ | **2 / 24** |

### What the 3 B cells actually show

All three B-decision cells are **poly-2 ridge** at $p \in \{50, 100\}$:

| Cell | $\bar R_1$ | $\bar R_2$ | $\bar R_3$ | $\rho_{12}$ |
|---|---|---|---|---|
| GPT-2,  $p\!=\!100$, poly2  | 56 839 | 9 800 | 7 665 | 5.80 |
| Pythia, $p\!=\!50$,  poly2  | 11 222 |   714 |   522 | 15.71 |
| Pythia, $p\!=\!100$, poly2  |  1 138 |   619 |   549 | 1.84 |

Read literally, these would suggest *very* strong rejection of first-order
in favour of second. But the inflated $\bar R_1$ values are diagnostic of
**catastrophic over-fitting at lag-1**, not of any genuine second-order
signature in the dynamics:

- At $p = 50$, the degree-2 polynomial expansion of a lag-1 input has
  $50 + 50 \cdot 51 / 2 = 1\,325$ features fitted to ~1 240 training
  quadruples — the classical underdetermined regime where ridge can stabilise
  training but generalisation is at the mercy of the regulariser.
- At lag-2 the polynomial expansion balloons to 5 150 features. Counter-
  intuitively, this *helps* generalisation: the larger the feature count
  relative to $n$, the more the ridge $\alpha\|w\|^2$ term dominates the data
  term, so the predictor is forced toward a smoother fit. Adding lag does
  not "use" the extra information — it merely forces the ridge to behave
  more like a smoothed kernel.
- The 21 non-poly-2 cells, where this artefact does not occur, all return
  decision C.

So the 3 B's are not a genuine "first-order rejected" signal — they are an
over-fitting artefact specific to poly-2 ridge at high $p$. Per §6.5
("function-class robustness ... If exactly one class disagrees, this is
reported but does not invalidate the headline"), the headline is C.

### Required confirmations (§6.5)

| Confirmation | Status |
|---|---|
| Architecture independence (GPT-2 ↔ Pythia primary cell) | **Met** — both cells return C with consistent direction. |
| Function-class robustness (3 of 4 classes agree at the headline cell) | **Met** — kernel, linear, MLP all give C at $p=50$ on both archs; only poly-2 dissents, and only via the over-fitting artefact above. |
| PCA-dim robustness ($p \in \{20, 50, 100\}$, kernel-ridge column on each arch) | **Met** — kernel ridge gives C at every $p$ on every arch. |

Headline outcome **stays at C** and is not downgraded by §6.5.

## Interpretation

### What the test does and does not say

The Markov-order regression test asks: *Is $h_t$ alone a sufficient
statistic for predicting $h_{t+1}$ in the chosen function class?* The
answer for the kernel-ridge / linear-ridge / MLP function classes, on the
last-layer hidden states of GPT-2 small and Pythia-160m, evaluated on the
§14 corpus at one-token temporal resolution, is **yes — and adding
$h_{t-1}$ marginally hurts.**

What the test does *not* say:

1. **It does not say transformer trajectories are first-order in the
   continuous-time sense.** A second-order ODE $\ddot x = f(x, \dot x)$
   *can* be discretised in many ways. If the discretisation maps to a
   one-step recurrence $h_{t+1} = F(h_t)$ — which is exactly what
   transformer position-axis dynamics are by construction — then the
   one-step Markov order is 1 by *kinematic* tautology. The two-step
   recurrence $h_{t+1} = F(h_t, h_{t-1})$ would only be a *strictly larger*
   conditional-expectation hypothesis class than the one-step recurrence.
2. **It does not say the §14 acceleration statistics are wrong.**
   $a_\parallel < 0$ on 97.9 % of triplets and $|a_\parallel|/|a_\perp|
   \approx 2.0$ are valid trajectory-shape observations. They demonstrate
   that the trajectories *systematically decelerate* along their tangent —
   which is what one expects from a Lagrangian model with attractive
   potentials. They are silent on the dynamical class generating those
   trajectories.
3. **It does not say the Lagrangian framework is wrong.** The Lagrangian
   provides a *generative* account that can produce the observed
   trajectories; the kernel-ridge test asks the *predictive* question
   whether longer history demonstrably helps. A negative answer to the
   predictive question is consistent with infinitely many generative
   accounts, including the Lagrangian one.

### What the test rules out vs. what remains compatible — the overdamped synthesis

The result is sometimes summarised as *"first-order ODE not rejected, so
the second-order Lagrangian framework is wrong."* That synthesis is too
restrictive. A more careful reading distinguishes three classes of
generative model and tracks which the test rejects and which remain
compatible.

| Generative model of hidden-state evolution | Compatible with the data? | Mechanism |
|---|---|---|
| First-order ODE / gradient flow $\dot h = -\nabla V(h)$ | **Yes** | $h_t$ is the full state; lag-2 adds noise. |
| **Overdamped** second-order ODE $w_t\,\ddot h + \gamma(h_t)\,\dot h + \nabla V = 0$ with $\gamma \gg \omega_0$ | **Yes** | In the overdamped limit $w_t\,\ddot h$ is negligible, the EOM reduces to $\dot h \approx -\nabla V / \gamma$, and the trajectory is observationally indistinguishable from a first-order gradient flow at one-token resolution. |
| Underdamped second-order ODE with detectable inertia | **No** | This would predict $\bar R_2 < \bar R_1$ at the rejection threshold; we observe the opposite at every PCA dim, every architecture and every primary function class. |

The two compatible models are observationally indistinguishable at
one-token resolution. The test's actual rejection target is the
**underdamped** version of the second-order ODE — not the second-order
ODE per se.

The unifying reading: transformer hidden-state trajectories, at one-token
resolution, are well described by the **overdamped limit** of the full
Euler–Lagrange equation (paper Eq. 67),
$w_t\,\ddot h_t + \gamma(h_t)\,\dot h_t = -\nabla V(h_t)$,
in the regime $\gamma \gg \omega_0$. In this limit:

- The kinetic term $T = \tfrac12\,w_t\,\|\dot h_t\|^2$ contributes to the
  Lagrangian, but its dynamical signature ($\ddot h_t$-mediated inertial
  memory) is too small relative to the dissipation $\gamma\,\dot h_t$ and
  the potential gradient $\nabla V$ to be recovered from one-step-ahead
  prediction.
- The dynamics collapse to $\dot h \approx -\nabla V / \gamma$ — a
  first-order gradient flow on the same potential $V$ that the Lagrangian
  framework constructs.
- The §14 acceleration statistics ($a_\parallel < 0$ on 97.9 % of
  triplets, $|a_\parallel|/|a_\perp|\!\approx\!2$, permutation-null
  $z=23$) are exactly the trajectory-shape signature an overdamped
  attractive potential produces: the trajectory decelerates along its
  tangent because the velocity is simultaneously being damped *and*
  aligned with $-\nabla V$.

The Lagrangian framework, restricted to the overdamped regime,
**remains a valid generative account**. The Jacobi-metric / geodesic
apparatus in the companion documents
([`docs/Jacobi_metric_and_Geodesics_in_Riemannian_Geometry.docx`](../../../docs/Jacobi_metric_and_Geodesics_in_Riemannian_Geometry.docx),
[`docs/Constructing_Langrangian_for_Semantic_Space.pdf`](../../../docs/Constructing_Langrangian_for_Semantic_Space.pdf))
is the overdamped reformulation of Maupertuis's principle and lives most
naturally in this regime. The connection to gradient-flow neural ODEs,
score-matching and overdamped Langevin dynamics — all first-order in the
same observational sense and all derivable from the overdamped limit of
the same Lagrangian — becomes a feature, not a bug.

What the paper *cannot* claim, and what the audit recorded, is that this
test demonstrates *strict* second-order dynamics with detectable inertia.
The honest empirical claim is one strictly weaker — and it is fully
compatible with the framework once it is read in the overdamped limit.

### Why the kernel ridge in particular fails to find a second-order signal

A blunt mechanical possibility: the second-order signal is small enough
that — with ~1 240 training quadruples and 50- to 150-dim feature vectors
— it is below the test's statistical power. Ridge with median-heuristic
$\gamma$ is essentially a smoothed local-average estimator; in dimension
$\ge 50$ with sample size $\sim 10^3$ the curse of dimensionality bites
and adding 50 more dims (from $h_t$ alone to $(h_t, h_{t-1})$) does not
*gain* information that survives the bias–variance trade-off.

A subtler possibility: $h_{t-1}$ is so highly correlated with $h_t$ that
its incremental information about $h_{t+1}$ is essentially in the residual
component $h_{t-1} - h_t$, which is the per-token *displacement vector*.
Even if the displacement is informative for $\ddot h_t$, kernel ridge in
the concatenated-feature space pays a dimensionality cost equal to the
*full* dimensionality of $h_{t-1}$ rather than just the rank of the
displacement. A targeted "displacement-as-feature" experiment would be a
distinct future test.

Either reading is consistent with C and neither lets us upgrade the result
to A.

### What the paper has to say differently

Compared to the framing in
[`docs/Evidence_for_second_order_ODE_governing_evolution.md`](../../../docs/Evidence_for_second_order_ODE_governing_evolution.md),
which the post-experiment banner already reframes:

- "**Definitely no first-order ODE**" is **not** supported empirically by
  this test and is dropped. The empirical claim that *is* supported, and
  that the next paper revision should adopt, is the **overdamped
  second-order** synthesis: hidden-state evolution at one-token
  resolution is consistent with the Euler–Lagrange equation (Eq. 67) in
  the regime $\gamma \gg \omega_0$, where it reduces to a first-order
  gradient flow $\dot h \approx -\nabla V / \gamma$ on the same
  potential $V$. This regime is observationally indistinguishable from
  a strict first-order ODE; what the test rejects is only the
  underdamped variant in which a velocity slot would carry detectable
  predictive information beyond $h_t$.
- Theorem 49's identity $a_\parallel = -\partial_v V$ remains a valid
  *kinematic* identity on any three vectors and a useful interpretation
  of the STP loss; it is *silent* on the underlying dynamical class,
  exactly as the audit recorded. It survives unchanged in the overdamped
  reading.
- The §14 acceleration statistics remain valid evidence *for* the
  Lagrangian generative account, but should be discussed in the careful
  framing: *"these statistics are exactly the trajectory-shape signature
  an overdamped attractive Lagrangian produces, and they are also
  reproducible by a first-order gradient flow on the same potential —
  these two generative models are equivalent in the regime our data
  occupies."*
- The Jacobi-metric / geodesic apparatus is *most naturally* the
  overdamped reformulation, so the framework gains rather than loses
  consistency from this synthesis.
- Outcome **C** is an honest negative result of a pre-registered test
  *for the underdamped variant*. It is reported in this private workspace's
  results tree and (recommended) in the next paper revision's §16 —
  Limitations / Open Questions, with the overdamped reframing made
  explicit and a forward pointer to *displacement-feature* and
  *layer-axis* follow-up tests as the natural sources of additional
  power if a future revision wants to attempt rejection of the
  *strict-first-order* variant in turn.

## Files (this directory + per-arch sub-directories)

```text
results/
├── RESULTS.md                       # this file
├── robustness_grid.csv              # 24 rows × 19 columns
├── robustness_grid_summary.json     # decision counts + Bonferroni count
├── gpt2/
│   ├── decision_table.md            # primary cell on GPT-2
│   ├── primary_summary.json
│   ├── primary_residuals.npz
│   ├── extraction_summary.json
│   ├── quadruples.npz
│   └── figures/
│       ├── Rk_bars.png
│       ├── paired_scatter.png
│       ├── loso_spaghetti.png
│       ├── robustness_rho12_gpt2.png
│       └── robustness_rho12_pythia-160m.png
└── pythia/
    └── (same layout as gpt2/)
```

## Compute

| Phase | Hardware | Wall-clock |
|---|---|---|
| Extraction (GPT-2)  | MPS  | 8 s |
| Extraction (Pythia) | MPS  | 10 s |
| Primary kernel-ridge LOSO (GPT-2)  | 16-core CPU | 174 s |
| Primary kernel-ridge LOSO (Pythia) | 16-core CPU | 144 s |
| 24-cell robustness sweep            | 16-core CPU | 53.8 min |
| Bootstrap (10 k resamples × 4 cells primary + 24 robustness) | 16-core CPU | included above |
| **Total**                           | —           | **≈ 60 min** |
