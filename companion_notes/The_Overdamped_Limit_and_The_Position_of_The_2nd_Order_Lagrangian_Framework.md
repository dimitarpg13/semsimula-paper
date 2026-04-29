# The Overdamped Limit and the Position of the Second-Order Lagrangian Framework in Semantic Simulation

**Dimitar P. Gueorguiev** — *Independent Researcher*

---

## Abstract

This report consolidates a sequence of theoretical and empirical findings bearing on a central question
in the Semantic Simulation programme: given that pretrained attention-based transformers and the
Scalar-Potential Language Model (SPLM) at moderate damping both exhibit hidden-state dynamics
consistent with a first-order gradient flow, does the second-order Lagrangian framework
$\mathcal{L} = T - V - \mathcal{R}$ retain scientific necessity, or is it reducible to theoretical
ornamentation? We argue that the empirical overdamped result is the framework's baseline, not its
refutation. The full second-order Equation of Motion (EOM) contains the overdamped limit as a
degenerate special case; the Lagrangian machinery makes falsifiable predictions about the richer
dynamical regime lying beyond that limit; and the SPLM is the only architecture on which the
predicted damping-expressivity trade-off can be tested as a controlled experiment. We further argue
that the damping coefficient $\gamma$ governs both inference trajectory stability and model expressivity
through its effect on the Dyck collapse depth $D^*$, making the optimal-damping question a
well-posed variational problem within the framework and not merely an engineering hyperparameter
search.

---

## 1. Background: The Mechanical Analogy and the First-Order Approximation

The governing equation of a damped harmonic oscillator,

$$m\ddot{x} + c\dot{x} + kx = F(t),$$

admits a characteristic pair of roots

$$\lambda_{1,2} = \frac{-c \pm \sqrt{c^2 - 4mk}}{2m}.$$

In the heavily overdamped regime, $c \gg 2\sqrt{mk}$, the roots separate widely:

$$|\lambda_1| \approx \frac{c}{m} \quad \text{(fast, rapidly decaying mode)}, \qquad |\lambda_2| \approx \frac{k}{c} \quad \text{(slow, dominant mode)}.$$

The fast mode $e^{\lambda_1 t}$ vanishes on the timescale $\tau_1 = m/c$. On timescales $t \gg \tau_1$, the
inertial term $m\ddot{x}$ becomes negligible relative to the damping term $c\dot{x}$, and the full
second-order EOM collapses to the first-order gradient flow:

$$c\dot{x} + kx = F(t).$$

This reduction — from second-order to first-order dynamics by the elimination of the inertial term — is
the classical overdamped approximation. It is a *singular perturbation* in the parameter
$\epsilon = m/c \to 0$, not a claim that the second-order theory is incorrect. The first-order
equation is the theory's degenerate limit, valid in a specific parameter regime; the second-order EOM
is the correct theoretical envelope.

This analogy maps precisely onto the hidden-state dynamics of attention-based transformers. In the
continuous (Neural ODE) limit, the layer-wise propagation of hidden states,

$$\mathbf{h}_{l+1} = \mathbf{h}_l + \mathcal{F}_l(\mathbf{h}_l),$$

yields the first-order flow $d\mathbf{h}/dt = \mathcal{F}(\mathbf{h}, t)$ — an equation with no inertial
term by construction. As established in Lu et al. (2020) via the multi-particle convection-diffusion
interpretation, the standard Transformer solves this ODE numerically using the Lie–Trotter splitting
scheme and Euler's method, with the self-attention sub-layer acting as the diffusion term and the
position-wise FFN as the convection term. The resulting update is formally first-order: there is no
$\ddot{\mathbf{h}}$ anywhere in the standard architecture.

The key question is whether this structural absence of inertia is an architectural inevitability or
an emergent consequence of training dynamics that could, in principle, be modified.

---

## 2. The Token-Sequence Dimension: STP and the Effective Damping Coefficient

Huang, LeCun, and Balestriero (2026) introduce a complementary ODE description operating in the
token-sequence (rather than layer-depth) dimension. Under the Training ODE,

$$dx_{\leq t} = \hat{u} \circ \hat{f}(x_{\leq t}) dt,$$

hidden-state trajectories are modeled as ballistic. At inference time, unembedding errors accumulate
and the equation becomes a Stochastic Differential Equation (SDE) with a Brownian motion term:

$$dx_{\leq t} = \hat{u} \circ \hat{f}(x_{\leq t}) dt + \sigma_t dW_t.$$

The Brownian noise $\sigma_t dW_t$ plays the role of an effective damping field: it causes the hidden
state trajectory $h_t$ to deviate from the optimal geodesic $h_t^*$, with distortion growing as

$$\lVerth_t - h_t^*\rVert_2 \propto \sigma\sqrt{t}.$$

Without regularization, the STP loss
$$\mathcal{L}_{\mathrm{STP}} = 1 - \cos(h_t - h_r, h_r - h_s)$$
takes values near $1.4$ at $\lambda = 0$, indicating that consecutive displacement vectors are
nearly anticorrelated — the hidden state reverses direction at each step. This is the empirical
signature of a heavily overdamped (diffusive) trajectory, not a ballistic one. As the regularization
strength $\lambda$ increases, $\mathcal{L}_{\mathrm{STP}}$ decreases toward values near $0.6$, and the
trajectory becomes directionally persistent.

The regularization parameter $\lambda$ therefore functions as the effective damping tuning knob in
the token-sequence dimension:

- $\lambda = 0$: heavily overdamped, near-random directional changes, $\mathcal{L}_{\mathrm{STP}} \approx 1.4$;
- $\lambda = 0.02$ (empirically optimal): partially underdamped, directionally coherent, geodesic-approximating;
- $\lambda \gg 1$: overconstrained, forces Euclidean straight-line trajectories that cannot track geodesic curvature on the curved semantic manifold.

The concave accuracy-vs-$\lambda$ profile observed empirically is precisely the fingerprint of an
optimal damping regime — the analogue of critical damping in the mechanical system.

---

## 3. The Layer-Depth Dimension: Splitting Schemes and Numerical Order

Lu et al. (2020) demonstrate that the standard Transformer corresponds to the Lie–Trotter splitting
scheme applied with Euler's method, achieving a local truncation error of $O(\gamma^2)$. The
Macaron Net improvement replaces this with the Strang–Marchuk scheme (FFN–Attention–FFN with
half-step residual connections), reducing the local truncation error to $O(\gamma^3)$:

$$\tilde{x}_{l,i} = x_{l,i} + \tfrac{1}{2}\mathrm{FFN}(x_{l,i}),$$

$$\hat{x}_{l,i} = \tilde{x}_{l,i} + \mathrm{MultiHeadAtt}(\tilde{x}_{l,i}, [\tilde{x}_{l,1}, \ldots, \tilde{x}_{l,n}]),$$

$$x_{l+1,i} = \hat{x}_{l,i} + \tfrac{1}{2}\mathrm{FFN}\left(\hat{x}_{l,i}\right).$$

This is a more accurate first-order numerical solver, not a promotion to second-order dynamics. To
genuinely introduce inertia in the layer dimension, one would require a heavy-ball momentum term
across layers:

$$\mathbf{h}_{l+1} = \mathbf{h}_l + \alpha(\mathbf{h}_l - \mathbf{h}_{l-1}) + \mathcal{F}_l(\mathbf{h}_l),$$

which promotes the recurrence from first-order to second-order in $l$. Such a modification is absent
from all standard attention architectures, and the question of whether it would move the Markov-order
test outcome from Decision C (first-order not rejected) toward Decisions A/B (inertia detectable)
remains an open empirical question.

---

## 4. Empirical Findings: The Markov-Order Regression Test

A Markov-order regression test was conducted on hidden-state quadruples extracted from GPT-2 Small
and Pythia-160M. The test asks: given $h_t$ as a regressor, does the addition of $h_{t-1}$ provide
measurable improvement in one-step-ahead prediction of $h_{t+1}$ across function classes
(linear, kernel-RBF, polynomial degree 2, MLP) and PCA dimensionalities (3, 64, 128)?

The headline result is **Decision C** for both architectures: $\bar{R}_1 / \bar{R}_2 > 1$ in 21 of
24 robustness cells, with the three exceptions attributable to degree-2 polynomial overfitting
artifacts confirmed by bootstrap confidence intervals and Leave-One-Sequence-Out (LOSO) fold
ordering $\bar{R}_1 < \bar{R}_2$ in 47 of 50 folds.

The three-way compatibility table is:

| Model | Compatible with data? | Mechanism |
|---|---|---|
| Strict first-order ODE: $\dot{h} = F(h)$ | **Yes** | $h_t$ is the full state; lag-2 adds noise |
| Overdamped second-order ODE: $m\ddot{h} + \gamma\dot{h} + \nabla V = 0$, $\gamma \gg \omega_0$ | **Yes** | In the overdamped limit, $\ddot{h}$ is negligible; EOM collapses to $\dot{h} \approx -\nabla V/\gamma$ |
| Underdamped second-order ODE with detectable inertia | **No** | Would predict $\bar{R}_2 < \bar{R}_1$; observed opposite universally |

The test **rules out** a second-order ODE in which the velocity component $\dot{h}_t$ carries
unique, predictively useful information beyond $h_t$ at one-token resolution with standard function
classes. It **does not rule out** the overdamped second-order ODE, which is observationally
indistinguishable from a first-order gradient flow at this resolution.

The §14 acceleration statistics — $a_\parallel < 0$ on $97.9\%$ of consecutive triplets, with
$|a_\parallel|/|a_\perp| \approx 2.0$, and a permutation null test yielding $z < -11$ for both
acceleration components — are consistent with all three of the following:

1. A damping/friction term in an underdamped second-order EOM;
2. An overdamped second-order EOM, where the measured "deceleration" is the curvature of the
   gradient-flow trajectory;
3. A first-order vector field with a convergent, sink-like attractor structure along the local heading.

The simplest model consistent with both §14 and §16 is the **overdamped second-order ODE** —
the gradient-flow / Jacobi-metric regime in which the kinetic energy $T$ contributes negligibly.

The empirically defensible claim is therefore:

> The hidden-state dynamics of GPT-2 and Pythia at one-token resolution is consistent with a
> first-order vector field $\dot{h} = F(h)$ and indistinguishable from the overdamped limit of the
> second-order Lagrangian system $\mathcal{L} = T - V - \mathcal{R}$. The Lagrangian framework
> remains a valid generative account because it reduces to the same overdamped gradient flow in the
> regime $\gamma \gg \omega_0$, which is precisely the regime our acceleration statistics are most
> naturally interpreted as evidencing.

---

## 5. The SPLM Smoke Test and the Gamma Sweep

The Markov-order test was also applied to hidden states extracted from a partially trained SPLM
checkpoint at $\gamma = 0.5$ (moderate damping). The result is again Decision C:
$\rho_{12} = 1.038$ (i.e., $\bar{R}_1/\bar{R}_2$), $p_{12} = 2.9 \times 10^{-4}$,
bootstrap CI $= [0.006, 0.016]$. The §14 acceleration statistic gives
`frac_a_par_negative` = 97.9% — coinciding precisely with the natural GPT-2 figure.

This finding admits a critical interpretation: even an architecturally distinct model with an
explicit, tunable damping parameter converges, under moderate training at $\gamma = 0.5$, to the
same heavily overdamped basin that standard training dynamics produce in attention transformers.
This is not an accidental convergence — it reflects the fact that the optimization landscape favors
contractive, dissipative dynamics regardless of architectural starting point, because:

- Softmax normalization enforces a dissipative competition among attention weights;
- Layer normalization combined with residual connections constitutes a contractive Lipschitz map;
- Standard optimizers (Adam, SGD with weight decay) actively suppress oscillatory modes during
  training by penalizing large weight magnitudes and gradient variance.

The natural effective damping coefficient $\gamma_{\mathrm{eff}}$ of any trained language model is
therefore large by training dynamics, not solely by architectural constraint.

A full gamma sweep across SPLM checkpoints is underway. The sweep constitutes a **damping-order
phase diagram** for the hidden-state dynamics. The critical observable is the ratio
$\rho_{12}(\gamma)$, defined as $\bar{R}_1(\gamma)/\bar{R}_2(\gamma)$, as a continuous function of $\gamma$.
Three outcomes are possible:

1. **Decision C holds at all $\gamma$**: training dynamics dominate architecture, always driving the
   system to the overdamped basin. The Lagrangian framework is theoretically necessary but
   empirically never in its non-trivial dynamical regime for any trained model.

2. **Decision flips toward A/B at low $\gamma$**: there exists a critical $\gamma^*$ below which
   inertia becomes detectable at one-token resolution. This is the scientifically most valuable
   outcome: it establishes a damping-order phase transition and directly vindicates the full
   second-order Lagrangian as the necessary model for the underdamped regime.

3. **Decision C holds but the §14 acceleration statistics change significantly with $\gamma$**:
   even without a regime change in the Markov-order sense, a monotonic shift in
   $|a_\parallel|/|a_\perp|$ or `frac_a_par_negative` as a function of $\gamma$
   provides evidence that the geometry of the trajectory is changing, constituting a weaker but
   publishable intermediate result.

---

## 6. The Lagrangian Framework as a Prescriptive Theory

The central thesis of the Semantic Simulation programme is that the framework is **prescriptive**,
not merely descriptive. This distinction is essential to the correct positioning of the second-order
Lagrangian result.

### 6.1 The Full Equation of Motion

The Euler–Lagrange equations derived from $\mathcal{L} = T - V$ with the Rayleigh dissipation
function $\mathcal{R}$ yield the full second-order EOM:

$$m\ddot{h} + \eta_0 H_i \dot{h} + \nabla_h V(h) = 0,$$

where the tanh damping factor is

$$H_i = \tanh\left(\frac{\lVertp - p_E\rVert}{x_u}\right),$$

$\eta_0$ is the base friction coefficient, $x_u$ is the damping onset length scale, and $p_E$ is the
ensemble centroid (bound-state attractor). The Gaussian semantic energy well is

$$V(h) = m\upsilon^2\left(1 - e^{-\kappa^2 \lVerth\rVert^2}\right), \quad \upsilon = \sqrt{E_t/m},\quad \kappa = f/\upsilon.$$

This is a complete second-order ODE system with three independently tunable parameters: $\eta_0$
(damping magnitude), $x_u$ (damping spatial scale), and $\kappa$ (well shape / restoring force
stiffness).

### 6.2 The Overdamped Limit as a Derived Shallow Limit

In the overdamped regime $\gamma_{\mathrm{eff}} \gg \omega_0$ (i.e., $\eta_0 H_i \gg \sqrt{\nabla^2 V / m}$),
the inertial term $m\ddot{h}$ is negligible. Setting it to zero and rearranging:

$$\dot{h} \approx -\frac{1}{\eta_0 H_i}\nabla_h V(h).$$

This is exactly the first-order gradient flow — the STP Training ODE, the Lu et al. convection-diffusion
flow, and the standard residual-stream update are all recovered as special cases of this single limit.
The framework unifies all three first-order accounts as distinct overdamped shallow limits of the same
second-order EOM. The full theory is the correct theoretical envelope; the first-order accounts are
valid only within their respective limiting regimes.

### 6.3 The Jacobi Metric and the Geometric Connection

The Jacobi metric induced by the Gaussian well,

$$ds_J^2 = 2(E - V(h)) g_{ij} dh^i dh^j,$$

transforms the second-order Lagrangian dynamics into a geodesic problem: trajectories that are
solutions to the Euler–Lagrange equations are geodesics of the Jacobi metric. This is the
**Maupertuis principle** in its overdamped reformulation.

In the overdamped limit, the Jacobi metric is the natural geometric object governing trajectory
shape. This connects the Semantic Simulation framework directly to the Geodesic Hypothesis of
Huang et al. (2026): the STP regularizer, by penalizing trajectory curvature, is implicitly
enforcing Jacobi-metric geodesic structure. The Semantic Simulation framework provides the
theoretical basis for what this geodesic structure *is* — it is the geometry induced by the
Gaussian semantic energy well — and its connection to the broader existing literature on
gradient-flow neural ODEs, score matching, and MCMC-as-overdamped-Langevin dynamics is
strengthened, not weakened, by the empirical overdamped result.

---

## 7. Damping, Expressivity, and the Dyck Collapse Depth

The most concrete forward-looking prediction of the second-order framework concerns the relationship
between the damping coefficient and model expressivity. Section 9 of the Semantic Simulation paper
establishes, via a four-step formal argument, that the v0 field-theoretic submodel — continuous-state,
finite-dimensional, smooth, damped — is **at most a finite automaton**, with a predicted Dyck collapse
depth

$$D^* = D^*(\dim M, L, \gamma, \epsilon, n),$$

that depends explicitly on the damping coefficient $\gamma$. Specifically, because damping dissipates
state-space volume at a rate proportional to $\gamma$, a heavily overdamped system loses access to
its nominal state capacity over time. The relationship is monotone: for fixed architecture parameters,
$D^*$ is a decreasing function of $\gamma$. Concretely,

$$\text{(state capacity destroyed by damping over depth } D) \propto \gamma \cdot D.$$

This has a precise implication: **reducing $\gamma$ toward critical damping increases $D^*$** and
hence extends the model's ability to track nested syntactic structure toward the mildly
context-sensitive class — the empirically established class for human language, which the composite
v0+v1.5+v2+v3 system is shown to generate exactly via reduction to Linear Context-Free Rewriting
Systems (LCFRS).

The question of optimal damping for language modeling is therefore not an engineering hyperparameter
search. It is the following well-posed variational problem within the Lagrangian framework:

$$\gamma^* = \arg\max_{\gamma > 0} \mathcal{P}(\gamma) \quad \text{subject to } \gamma > \gamma_{\mathrm{crit}},$$

where $\mathcal{P}(\gamma)$ is the downstream language modeling performance (e.g., validation
perplexity), and $\gamma_{\mathrm{crit}}$ is the stability boundary below which the Euler–Lagrange
integrator diverges. The framework predicts that $\gamma^*$ is strictly less than the $\gamma$ value
at which all currently trained models sit — that is, there exists a better operating point that
standard training dynamics do not reach autonomously.

---

## 8. Why the SPLM Is the Only Valid Experimental Platform

The damping sweep experiment described in Sections 5 and 7 requires an architecture in which
$\gamma_{\mathrm{eff}}$ is an independently tunable and physically interpretable parameter. This
requirement is **not satisfied** by standard attention-based transformers for six structural reasons
catalogued in Section 15 of the paper:

1. **Asymmetric weight matrices** $W_Q \neq W_K^\top$: the per-layer force is non-conservative
   by construction;
2. **Multi-head concatenation**: the force decomposes into a sum of head-specific flows with no
   single shared potential;
3. **Causal mask over prefix history**: the force field is non-autonomous in token context;
4. **LayerNorm**: introduces a normalization non-linearity that cannot be absorbed into a scalar
   potential;
5. **Distinct $W^{(\ell)}$ per layer**: the force field is non-autonomous in layer index;
6. **Softmax**: enforces dissipative competition among tokens via normalized exponentials, producing
   an effective damping that is entangled with representational geometry and cannot be disentangled
   from the attention mechanism itself.

Each of these six features independently obstructs conservativity. Together they render the per-layer
force non-autonomous in both layer index and token context, placing the attention circuit outside the
class of autonomous, shared-potential Helmholtz decompositions by design.

The SPLM, by contrast, is a weight-tied autoregressive circuit whose inference is **by construction**
a damped Euler–Lagrange flow on a single learned scalar $V_\theta(\xi, h)$, with a causal
cumulative-mean context pool $\xi$ and no attention, no multi-head decomposition, no softmax, and no
per-layer distinct parameters. Its shared-potential diagnostic attains median per-layer $R^2 = 0.90$
with a uniform profile across layers, against an oracle ceiling of $R^2 = 1.0000$. The SPLM's
damping parameter $\eta_0$ is physically interpretable, independently tunable, and maps directly onto
the $\gamma$ appearing in the Dyck-depth bound $D^*(\gamma)$.

The SPLM is therefore the only currently available architecture on which the prediction of Section 7
— that reducing $\gamma$ from the overdamped basin toward $\gamma^*$ improves language modeling
performance monotonically up to the stability boundary — can be tested as a controlled experiment.

---

## 9. Synthesis: The Position of the Second-Order Framework

The following table summarises the roles of the second-order Lagrangian framework across the
three levels of the programme.

| Level | What is established | Role of the 2nd-order framework |
|---|---|---|
| **Descriptive** (pretrained transformers) | Overdamped gradient-flow regime confirmed; §14 acceleration statistics real but not requiring inertia | Valid as the theoretical envelope that contains the overdamped limit; Jacobi metric is the correct geometric object |
| **Prescriptive** (SPLM architecture) | Shared-potential diagnostic satisfied at $R^2 = 0.90$; SPLM is globally conservative by design | Full EOM is the governing dynamics by construction; $\gamma$ is independently tunable |
| **Generative** (future dynamical simulation) | Gamma sweep in progress; optimal $\gamma^*$ predicted to be strictly below training-converged value | Second-order Lagrangian is the only framework that can predict and interpret the phase transition; Dyck-depth bound $D^*(\gamma)$ gives falsifiable prediction |

The scientifically correct framing, consistent with the empirical evidence and grounded in
Sections 2–9 of the paper, is therefore:

> Trained attention transformers and SPLM at moderate damping both converge empirically to the
> overdamped gradient-flow regime — the first-order shallow limit of the Semantic Simulation EOM
> (§7.5). The second-order Lagrangian framework's prescriptive value lies precisely in predicting
> the richer dynamical regime lying beyond this limit: §7.7 characterises the full phase portrait
> as a function of $\eta_0$; §9 establishes that the Dyck collapse depth $D^*$ — and hence
> expressivity — is monotonically increasing as $\gamma$ decreases toward critical damping; and the
> SPLM's shared-potential architecture provides the only controllable experimental platform on which
> this prediction can be tested. The overdamped result is the baseline; the second-order Lagrangian
> is the theory of what lies beyond it.

---

## 10. Proposed Experimental Programme

The following sequence of experiments constitutes the falsification programme for the claims of
Section 7.

**E1 — Full gamma sweep on SPLM.**
Run the §16 Markov-order regression test on SPLM hidden states across a logarithmic grid of
$\gamma \in [0.05, 2.0]$, recording $\rho_{12}(\gamma)$ as the primary observable. A monotone
decrease in $\rho_{12}$ as $\gamma \to \gamma_{\mathrm{crit}}$ constitutes positive evidence for
the damping-order phase diagram. A flip to $\rho_{12} < 1$ at some $\gamma^{**}$ is the strongest
possible confirmation of detectable inertia.

**E2 — Dyck-depth falsifier.**
Evaluate SPLM and a baseline attention model on the formal language battery F1–F6 (Dyck$_n$,
Dyck+topic, Dyck+let, $a^n b^n c^n$, bounded copy $ww$, Minsky 2-counter) across the same
$\gamma$ grid. A monotone increase in maximum correctly parsed Dyck depth as $\gamma$ decreases
would directly confirm the $D^*(\gamma)$ prediction.

**E3 — Performance optimum location.**
Track validation perplexity on Tiny Shakespeare (and, subsequently, a larger corpus) across the
$\gamma$ grid to locate $\gamma^*$ empirically. The framework predicts $\gamma^* < \gamma_{\mathrm{smoke}}
= 0.5$. Specifically, it predicts the performance-vs-$\gamma$ curve is unimodal with a maximum at
an interior point, not monotone.

**E4 — Momentum-residual attention baseline.**
Introduce a heavy-ball momentum term into a standard attention residual stream:
$$\mathbf{h}_{l+1} = \mathbf{h}_l + \alpha(\mathbf{h}_l - \mathbf{h}_{l-1}) + \mathcal{F}_l(\mathbf{h}_l),$$
and run the §16 Markov-order test on hidden states of this architecture as $\alpha$ increases from 0
(standard residual) to 1 (full momentum). This tests whether architectural inertia, independently of
the shared-potential structure, is sufficient to produce detectable second-order dynamics.

---

## 11. Conclusion

The empirical finding that standard attention transformers and SPLM at moderate damping both sit in
the heavily overdamped dynamical regime does not undermine the second-order Lagrangian framework of
Semantic Simulation. It establishes the regime that the framework's overdamped shallow limit
correctly describes, and thereby identifies the boundary from which the framework's non-trivial
predictions begin. The second-order EOM is the correct theoretical envelope; the first-order gradient
flow is one of its degenerate limits; the Jacobi metric is the geometric object governing geodesic
structure in that limit; and the damping coefficient $\gamma$ is the parameter whose optimal value —
predicted to lie strictly below the training-converged value — governs both inference trajectory
stability and model expressivity via the Dyck collapse depth $D^*$. The SPLM, as the only
conservative-by-construction architecture with a physically interpretable and independently tunable
$\gamma$, is the natural and exclusive experimental platform for testing these predictions.

---

## References

- Huang, H., LeCun, Y., and Balestriero, R. (2026). *Semantic Tube Prediction: Beating LLM Data Efficiency with JEPA*. arXiv:2602.22617.
- Lu, Y., Li, Z., He, D., Sun, Z., Dong, B., Qin, T., Wang, L., and Liu, T.-Y. (2020). *Understanding and Improving Transformer from a Multi-Particle Dynamic System Point of View*. arXiv:1906.02762.
- Gueorguiev, D. P. (2026). *Semantic Simulation: A Prescriptive Lagrangian Framework for Efficient Semantic Inference*. Independent Researcher preprint.
- Gueorguiev, D. P. (2022a). *Damping factor and bound-state approach in Semantic Simulation* (unpublished technical note).
- Gueorguiev, D. P. (2022f). *Signature matrix and information content of a semantic property* (unpublished technical note).
- Gueorguiev, D. P. (2022g). *Gaussian semantic energy well* (unpublished technical note).
- LeCun, Y. (2022). *A Path Towards Autonomous Machine Intelligence*. OpenReview.
- Vijay-Shanker, K., Weir, D. J., and Joshi, A. K. (1987). Characterizing structural descriptions produced by various grammatical formalisms. *ACL 1987*.
- Merrill, W. and Sabharwal, A. (2023). The parallelism tradeoff: Limitations of log-precision transformers. *TACL*.

---

*Prepared in connection with the Semantic Simulation companion repository:*
*https://github.com/dimitarpg13/semsimula-paper*
