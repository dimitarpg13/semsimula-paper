# Semantic Simulator: v0 Equations of Motion

The complete v0 specification of the Semantic Simulation framework as
an executable particle-mechanics simulator in semantic space, with
every parameter classified as static (corpus statistics),
RL-calibrated, or a hyperparameter.

This is **M0** of the programme defined in
`docs/Semantic_Simulator_RL_Calibration_Programme.md`. The purpose of
this document is to translate the framework's narrative description
(paper §2–§8 plus §11–§12) into a single block of unambiguous
equations and pseudocode, so that every under-specification surfaces
**before** any RL machinery is built.

Where the choice is genuinely free at v0, the placeholder is marked
**[v0 choice]** and a pointer to the alternative is given.

Written 2026-04-24.

---

## 1. Scope and what v0 commits to

v0 is a **single-particle, autoregressive next-token simulator**:

- One "particle" $x_t$ per output position $t$.
- Position $t$'s particle is integrated for a fixed number of steps
  $L$ inside a context-conditioned potential $V(\xi_t, \cdot)$, then
  read out into a token distribution.
- The previous-positions context enters only through $\xi_t$, the
  causal cumulative mean of the preceding embeddings. **No
  cross-particle interactions at v0.**
- The simulator is deterministic given the input token sequence and
  the parameters; sampling, if any, happens at the readout.

Cross-particle interactions (multi-particle SARF, dialogue between
particles at different output positions) are **deferred to v1**.
Hierarchical structure (sentence-level, document-level particles)
is **deferred to v2**.

This v0 scope is the smallest version of the simulator that can be
calibrated against a corpus and compared to baselines. It is
deliberately a strict subset of the full framework.

---

## 2. State space and notation

For each output position $t \in \{0, 1, \ldots, T-1\}$ and each
integration step $\ell \in \{0, 1, \ldots, L\}$, the simulator
maintains:

| Symbol | Type | Meaning |
| --- | --- | --- |
| $v_t$ | $\{1, \ldots, V\}$ | Input token at position $t$ |
| $e_v$ | $\mathbb{R}^d$ | Embedding of vocabulary item $v$ |
| $x_t^{(\ell)}$ | $\mathbb{R}^d$ | Particle position, position $t$, step $\ell$ |
| $\dot{x}_t^{(\ell)}$ | $\mathbb{R}^d$ | Particle velocity |
| $\xi_t$ | $\mathbb{R}^d$ | Causal cumulative mean of input embeddings |
| $\mathfrak{m}_t$ | $\mathbb{R}_{>0}$ | Per-position semantic mass |
| $V(\xi, x)$ | $\mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}$ | Scalar potential |
| $F(\xi, x) = -\nabla_x V(\xi, x)$ | $\mathbb{R}^d$ | Force |
| $\gamma$ | $\mathbb{R}_{>0}$ | Damping coefficient |
| $\Delta t$ | $\mathbb{R}_{>0}$ | Integration step size |
| $L$ | $\mathbb{N}$ | Integration horizon (number of layer-steps) |

Constants of the model:

- $V$ = vocabulary size.
- $d$ = embedding dimension. **[v0 choice]** $d = 64$ for toy-language
  experiments, $d = 256$ for sub-domain experiments.

The cumulative context is

$$
\xi_t  =  \frac{1}{t+1}\sum_{m=0}^{t} e_{v_m},
\qquad \xi_{-1}  =  \mathbf{0}.
$$

This is the SPLM-paper definition (§11), preserved for v0.

---

## 3. Static layer (corpus statistics, frozen)

Three artifacts are computed once on the corpus and never updated by
RL:

### 3.1 Vocabulary embeddings $e_v \in \mathbb{R}^d$

**[v0 choice]** Initial embeddings are the top-$d$ eigenvectors of the
PMI matrix:

$$
\mathrm{PMI}(v, w)  =  \log\frac{p(v, w \mid \text{window})}{p(v) p(w)}.
$$

Window size: 5 tokens each side. Spectral decomposition gives
$E \in \mathbb{R}^{V \times d}$, with rows $e_v$. Norm-normalised
($\|e_v\|_2 = 1$ for all $v$) so the embedding lives on the unit
sphere $S^{d-1}$.

**Estimator:** sparse PMI matrix + truncated SVD. Cost
$O(V^2 \cdot \log V)$, one-shot.

**Why not GloVe/word2vec:** PMI-spectral is closed-form, parameter-free,
and avoids a hidden gradient-descent calibration. GloVe/word2vec are
RL-calibrated optimisers in disguise — using them at the static layer
would conflate (3.1) with (4) and obscure the parameter classification.

### 3.2 Per-token semantic mass $\mathfrak{m}_v$

$$
\mathfrak{m}_v  =  -\log\hat{p}_{\text{corpus}}(v),
$$

where $\hat{p}_{\text{corpus}}(v)$ is the unigram MLE on the calibration
corpus, smoothed with a small additive constant. Frozen after counting.

This is the surprisal mass of §11. Per-position mass is the source
token's mass: $\mathfrak{m}_t := \mathfrak{m}_{v_t}$.

### 3.3 Concept anchor positions $x_{c,k} \in \mathbb{R}^d$

$K$-means clustering of the embedding rows $\{e_v\}$ in $S^{d-1}$
gives anchor centres $x_{c,k}$ (Euclidean centroids, then re-projected
to $S^{d-1}$). Each anchor inherits a mass

$$
\mathfrak{m}_k  =  \sum_{v \in \mathrm{cluster}(k)} \mathfrak{m}_v
$$

i.e. the total surprisal weight of its cluster.

**[v0 choice]** $K = 32$ for toy-language, $K = 256$ for sub-domain
corpora. Sweep target.

---

## 4. Force decomposition

The potential decomposes as

$$
V(\xi, x)  = 
   V_{\text{wells}}(x) +
   V_{\text{SARF}}(x) +
   V_{\text{PARF}}(\xi, x) +
   V_{\text{ctx}}(\xi, x).
$$

Each term has an explicit parametric form. The **force** is the
negative gradient of $V$ in $x$ and is given component by component
below.

### 4.1 $V_{\text{wells}}$: lexical attractors

Sum of inverted Gaussian wells over the $K$ concept anchors:

$$
V_{\text{wells}}(x)  = 
   \sum_{k=1}^{K} \mathfrak{m}_k \upsilon_k^{2} 
      \bigl(1 - e^{-\kappa_k^{2} \|x - x_{c,k}\|^{2}}\bigr).
$$

Force:

$$
F_{\text{wells}}(x)  = 
   -\sum_{k=1}^{K} 2 \mathfrak{m}_k \upsilon_k^{2} \kappa_k^{2} 
      (x - x_{c,k}) e^{-\kappa_k^{2} \|x - x_{c,k}\|^{2}}.
$$

Parameters: $\{x_{c,k}\}$ static (3.3); $\{\mathfrak{m}_k\}$ static
(3.3); $\{\upsilon_k\}$ and $\{\kappa_k\}$ **RL-calibrated**.

### 4.2 $V_{\text{SARF}}$: semantic attractor-repellor field

Pairwise interactions among $N_S$ static SARF anchors $\{a_j\}_{j=1}^{N_S}$.
**[v0 choice]:** SARF anchors are the top-$N_S$ PMI peaks (highest
$\max_w \mathrm{PMI}(v, w)$ tokens), so anchors are *informationally*
extremal in a way that's distinct from clustering centres.

Each anchor contributes a two-component (short-range repulsion +
long-range attraction) interaction:

$$
V_{\text{SARF}}(x)  = 
   \sum_{j=1}^{N_S} \alpha_j 
      \Bigl[
         \beta_j e^{-\|x - a_j\|^{2}/\sigma_{j,-}^{2}}
          -  e^{-\|x - a_j\|^{2}/\sigma_{j,+}^{2}}
      \Bigr],
\qquad \sigma_{j,-} < \sigma_{j,+},\ \alpha_j > 0,\ \beta_j \ge 1.
$$

The second term ($-e^{-r^2/\sigma_{j,+}^{2}}$) is the
**attractive lobe**: broad range $\sigma_{j,+}$, $V$ minimal at $r = 0$
and growing toward zero. The first term ($+\beta_j e^{-r^2/\sigma_{j,-}^{2}}$)
is the **repulsive core**: narrow range $\sigma_{j,-}$, $V$ maximal at
$r = 0$. Their sum produces a Lennard-Jones-like landscape with an
energy minimum at an equilibrium distance $r^{*}_{j}$ determined
by $(\sigma_{j,-}, \sigma_{j,+}, \beta_j)$.

Force:

$$
F_{\text{SARF}}(x)  = 
   2\sum_{j=1}^{N_S} \alpha_j (x - a_j) 
      \Bigl[
         \frac{\beta_j}{\sigma_{j,-}^{2}} e^{-\|x - a_j\|^{2}/\sigma_{j,-}^{2}}
          -  \frac{1}{\sigma_{j,+}^{2}} e^{-\|x - a_j\|^{2}/\sigma_{j,+}^{2}}
      \Bigr].
$$

(For $r$ small, the bracket is positive and the force points away
from $a_j$ — repulsion. For $r$ in the basin region, the bracket
is negative and the force points toward $a_j$ — attraction.)

Parameters: $\{a_j\}$ static (PMI-peak tokens' embeddings);
$\{\alpha_j, \beta_j, \sigma_{j,+}, \sigma_{j,-}\}$ **RL-calibrated**.

**[v0 choice]:** $N_S = 16$ for toy-language. To preserve sparsity
of computation, anchors are tied across positions $t$ at v0 — i.e.
SARF is *not* context-conditioned in v0. Context-conditioning
($a_j(\xi)$) is an explicit v1 extension and is the most natural
place to surface non-Markovian structure.

### 4.3 $V_{\text{PARF}}$: property field

Linear couplings to a small set of property directions. v0 properties:

| $p$ | Property | Source of $w_p$ | Source of $c_p(\xi)$ |
| --- | --- | --- | --- |
| 1 | "DET-ness" | mean embedding of DET tokens minus corpus mean | proj of $\xi$ on $w_1$ |
| 2 | "VERB-ness" | mean embedding of VERB tokens minus corpus mean | proj of $\xi$ on $w_2$ |
| 3 | "NOUN-ness" | mean embedding of NOUN tokens minus corpus mean | proj of $\xi$ on $w_3$ |
| 4 | "punctuation-ness" | mean embedding of punctuation tokens | proj of $\xi$ on $w_4$ |
| 5 | "register" | first PCA component of register-labelled embeddings | proj of $\xi$ on $w_5$ |

**Form:**

$$
V_{\text{PARF}}(\xi, x)  = 
   \sum_{p=1}^{P} \lambda_p c_p(\xi) \langle w_p, x \rangle,
\qquad
F_{\text{PARF}}(\xi, x)  = 
   -\sum_{p=1}^{P} \lambda_p c_p(\xi) w_p.
$$

This is a **linear-in-$x$** force: the property field exerts a
constant pull whose magnitude is set by the context's projection on
the property direction. Linear potentials look strange in physics
but are standard in semantic-space modelling — they're what gives
attention its "additive" character in the Transformer literature.

Parameters: $\{w_p\}$ static (POS/property statistics);
$c_p(\xi) := \langle w_p, \xi \rangle$ deterministic (no learning);
$\{\lambda_p\}$ **RL-calibrated**.

**[v0 choice]:** $P = 5$. Adding more properties is an obvious
extension; the question is whether the marginal value per added
property justifies the calibration cost.

### 4.4 $V_{\text{ctx}}$: context coupling

The simplest non-trivial form of explicit non-autonomy:

$$
V_{\text{ctx}}(\xi, x)  =  \tfrac{1}{2} \lambda_{\text{ctx}} \|x - \xi\|^{2},
\qquad
F_{\text{ctx}}(\xi, x)  =  -\lambda_{\text{ctx}} (x - \xi).
$$

A scalar harmonic well centred at $\xi$. Pulls the particle toward
the cumulative-context mean.

Parameter: $\lambda_{\text{ctx}}$ **RL-calibrated**.

**[v0 choice]:** scalar (isotropic) coupling. The matrix-valued
extension $\frac{1}{2}(x-\xi)^\top \Lambda (x-\xi)$ with full
$\Lambda \in \mathbb{R}^{d \times d}$ is the natural v1 generalisation,
giving $d^2$ additional RL-calibrated parameters and direction-dependent
coupling. Almost certainly necessary for non-toy languages.

---

## 5. Integrator

**[v0 choice]:** damped semi-implicit Euler at fixed step size
$\Delta t$.

$$
\dot{x}_t^{(\ell+1)}  = 
   (1 - \gamma \Delta t) \dot{x}_t^{(\ell)}
    +  \frac{\Delta t}{\mathfrak{m}_t} F\bigl(\xi_t, x_t^{(\ell)}\bigr),
$$

$$
x_t^{(\ell+1)}  =  x_t^{(\ell)}  +  \Delta t \dot{x}_t^{(\ell+1)}.
$$

**Why semi-implicit Euler and not velocity-Verlet:** the §14.15
attractor-extraction finding is that the lower-order semi-implicit
Euler integrator produces **richer, more content-bearing attractors**
than the higher-order velocity-Verlet. v0 inherits this finding as a
prior. Velocity-Verlet remains an ablation, not the default.

Hyperparameters: $L$ (integration horizon), $\Delta t$ (step size),
$\gamma$ (damping). All swept; $\gamma$ may be promoted to
RL-calibrated if the sweep shows strong sensitivity.

**[v0 choice]:** $L = 8$, $\Delta t = 0.5$, $\gamma = 0.5$ as
inherited from the SPLM-tiny experiments.

---

## 6. Readout

Tied nearest-neighbour with temperature:

$$
p(v \mid x_t^{(L)})  = 
   \frac{\exp\bigl(\beta \langle e_v, x_t^{(L)} \rangle\bigr)}
        {\sum_{v'=1}^{V}\exp\bigl(\beta \langle e_{v'}, x_t^{(L)} \rangle\bigr)}.
$$

Tied means the readout uses the **same** vocabulary embeddings as
the input. No separate $W_{\text{out}}$ matrix.

Parameter: $\beta$ **RL-calibrated** (single scalar). Ties the
softmax temperature to the calibrated effective scale of the
embedding space.

---

## 7. Initial conditions and boundary conditions

For each position $t$:

$$
x_t^{(0)}  =  e_{v_t},
\qquad \dot{x}_t^{(0)}  =  \mathbf{0}.
$$

The particle is initialised **at** the input token's embedding with
**zero** initial velocity. After $L$ integration steps, $x_t^{(L)}$
is the readout state for predicting $v_{t+1}$.

**Why this and not random initialisation:** the SPLM positive result
(§13) sets layer-0 hidden state to the embedding. v0 inherits this.

**Boundary conditions:** none. Position $t = 0$ uses
$\xi_0 = e_{v_0}$, the embedding of the first token (degenerate
causal mean).

---

## 8. Full simulator forward pass (pseudocode)

```python
def simulate_step(token_seq, params):
    """
    token_seq: list of int, length T
    params: SimulatorParams (see §10 for full classification)
    returns: T x V matrix of next-token distributions
    """
    T = len(token_seq)
    embeddings  = [params.E[v] for v in token_seq]        # T x d, static
    masses      = [params.m[v] for v in token_seq]        # T,   static

    # Causal cumulative mean
    xi = []
    cumsum = np.zeros(params.d)
    for t in range(T):
        cumsum += embeddings[t]
        xi.append(cumsum / (t + 1))                       # T x d

    out = np.zeros((T, params.V))
    for t in range(T):
        x  = embeddings[t].copy()                         # x_t^{(0)}
        v  = np.zeros(params.d)                           # xdot_t^{(0)}
        for ell in range(params.L):
            F = (force_wells(x, params)
                 + force_sarf(x, params)
                 + force_parf(x, xi[t], params)
                 + force_ctx(x, xi[t], params))
            v = (1 - params.gamma * params.dt) * v \
                + (params.dt / masses[t]) * F             # eq. 5
            x = x + params.dt * v                         # eq. 5
        # Readout
        logits = params.beta * (params.E @ x)             # V
        out[t] = softmax(logits)
    return out
```

The four `force_*` functions are direct implementations of the
gradient expressions in §4.1–§4.4. None of them require autograd at
the simulator level — every gradient is in closed form.

---

## 9. Calibration loops

### 9.1 Behavior cloning (M2 of the programme)

Loss: KL between simulator next-token distribution and corpus
next-token distribution at each position:

$$
\mathcal{L}_{\text{BC}}  = 
   \frac{1}{T}\sum_{t=0}^{T-1}
   \mathrm{KL}\bigl(p_{\text{corpus}}(\cdot \mid v_{0:t}) \| p_{\text{sim}}(\cdot \mid x_t^{(L)})\bigr).
$$

For corpora where the conditional distribution isn't directly
observable (i.e. real text), $p_{\text{corpus}}(v_{t+1} \mid v_{0:t})$
is replaced by the empirical delta on the next token, recovering the
standard cross-entropy loss.

Gradient is taken with respect to the RL-calibrated parameters
**only** (table in §10). The static-layer parameters are frozen.

**Optimiser:** Adam with full backprop through the integrator. The
step counts here are $\sim$1000s of mini-batches, not transformer
pretraining.

### 9.2 Intrinsic-reward RL (M3)

Reward terms (each computed per trajectory):

$$
r_{\text{intrinsic}}  = 
   r_{\text{energy}} + r_{\text{predict}} + r_{\text{basin}} + r_{\text{stp}},
$$

with:

- **Energy regularity** $r_{\text{energy}}$: penalise
  $|E^{(L)} - E^{(0)} + \gamma \int_0^L T d\ell|$, i.e. enforce
  energy conservation modulo damping.
- **Trajectory predictability** $r_{\text{predict}}$: reward low
  variance of $\|x_t^{(\ell+1)} - x_t^{(\ell)}\|$ across $\ell$
  (smooth flows).
- **Basin stability at horizon** $r_{\text{basin}}$: reward low
  variance of $x_t^{(L)}$ across nearby initialisations
  $x_t^{(0)} + \epsilon$ — pullback-attractor stability of §14.15.
- **STP-loss reduction** $r_{\text{stp}}$: reward low residual STP
  loss along the trajectory (§12).

Implemented as a sum-of-rewards advantage estimator on top of the
behavior-cloning loss. Coefficients on each $r_*$ are themselves
hyperparameters (§10).

### 9.3 Task reward RL (M4)

Reward = held-out next-token NLL on a target corpus, or task accuracy
on a downstream linguistic probe. Standard policy-gradient on
RL-calibrated parameters. Used only as a fine-tuning layer on top
of (9.1)+(9.2).

---

## 10. Parameter classification table

| Symbol | Bucket | Source / signal | Count (toy) | Count (sub-domain) |
| --- | --- | --- | --- | --- |
| $\{e_v\}$ | static | PMI-spectral, §3.1 | $V \cdot d = 200 \cdot 64 = 12{,}800$ | $V \cdot d \approx 10^{4} \cdot 256 = 2.56 \cdot 10^{6}$ |
| $\{\mathfrak{m}_v\}$ | static | unigram surprisal, §3.2 | $V = 200$ | $\sim 10^{4}$ |
| $\{x_{c,k}\}$ | static | $K$-means, §3.3 | $K \cdot d = 32 \cdot 64 = 2{,}048$ | $K \cdot d = 256 \cdot 256 = 65{,}536$ |
| $\{a_j\}$ | static | PMI-peak embeddings | $N_S \cdot d = 16 \cdot 64 = 1{,}024$ | $N_S \cdot d = 64 \cdot 256 = 16{,}384$ |
| $\{w_p\}$ | static | property statistics | $P \cdot d = 5 \cdot 64 = 320$ | $P \cdot d = 16 \cdot 256 = 4{,}096$ |
| $\{\upsilon_k, \kappa_k\}$ | **RL** | wells §4.1 | $2K = 64$ | $2K = 512$ |
| $\{\alpha_j, \beta_j, \sigma_{j,\pm}\}$ | **RL** | SARF §4.2 | $4 N_S = 64$ | $4 N_S = 256$ |
| $\{\lambda_p\}$ | **RL** | PARF §4.3 | $P = 5$ | $P = 16$ |
| $\lambda_{\text{ctx}}$ | **RL** | ctx §4.4 | 1 | 1 |
| $\beta$ | **RL** | readout §6 | 1 | 1 |
| $L$, $\Delta t$, $\gamma$ | hyper | swept | 3 | 3 |
| $K$, $N_S$, $P$, $d$ | hyper | swept | 4 | 4 |

**RL-calibrated parameter counts at v0:**

- Toy-language: **135** RL-calibrated parameters.
- Sub-domain: **786** RL-calibrated parameters.

Compare:

- A tiny SPLM at $d=64, L=8$ trained for the same task has
  $\sim 7 \cdot 10^{5}$ parameters.
- A tiny transformer at the same scale has $\sim 1 \cdot 10^{6}$
  parameters.

The simulator is ~5000× smaller in the RL-calibrated count.
**This is the central architectural claim of the programme:
that the right inductive bias collapses the learnable parameter
count by 3–4 orders of magnitude.**

If RL calibration of $10^{2}$–$10^{3}$ parameters fails to reach
within 2× of trigram on the toy task, the inductive bias is not
sufficient and the framework's force-form vocabulary must be
extended.

---

## 11. Computational complexity

Per forward step at position $t$:

- Wells force: $O(K \cdot d)$.
- SARF force: $O(N_S \cdot d)$.
- PARF force: $O(P \cdot d)$.
- Context force: $O(d)$.

Total per integration step: $O((K + N_S + P) \cdot d)$.
Per position: $O(L \cdot (K + N_S + P) \cdot d)$.
Full sequence of length $T$: $O(T \cdot L \cdot (K + N_S + P) \cdot d)$.

At toy scale ($T = 100, L = 8, K + N_S + P = 53, d = 64$): $\sim 2.7 \cdot 10^{6}$
flops per sequence. Comparable to a single transformer attention
operation. **Cheap.**

At sub-domain scale ($T = 1024, L = 8, K + N_S + P = 336, d = 256$):
$\sim 7 \cdot 10^{8}$ flops per sequence. Still a fraction of even a
small transformer.

The simulator is computationally **inexpensive by design** because
the force decomposition is sparse-and-named, not dense-and-learned.

---

## 12. Open under-specifications

This is the actual point of M0: the act of writing the EOM down
surfaces things that are ambiguous in the framework's narrative
description. Below are the choices v0 has to make that the paper
does not nail down explicitly.

1. **Is SARF context-conditioned at v0?** v0 says no
   (§4.2 [v0 choice]). The paper §6 description is ambiguous on this
   point. v1 must address.

2. **Are concept anchors and SARF anchors the same set?** v0 says
   no (§3.3 vs §4.2). The paper does not separate them. This
   simulator treats them as distinct; if v1 unifies them, parameter
   count drops by ~25%.

3. **Is mass per-token or per-anchor or both?** v0 says both
   (§3.2 token-mass, §3.3 anchor-mass). The paper §11 only
   discusses token-mass.

4. **What is "the property" in PARF?** v0 hardcodes 5 properties
   (§4.3). The paper §5 leaves this open. The choice of properties
   *is* a research artifact; different toy tasks may need different
   property sets.

5. **What is the readout topology?** v0 uses tied
   nearest-neighbour (§6). Alternatives: untied softmax classifier,
   sampling from a continuous-time density, contrastive readout.
   Each gives a different interpretability vs capacity trade-off.

6. **What is the right damping?** v0 makes $\gamma$ a hyperparameter.
   It is plausibly a function of $\mathfrak{m}_t$ — high-mass tokens
   damp differently from low-mass ones. v1 should consider
   $\gamma_t = \gamma_0 / \mathfrak{m}_t$ or similar.

7. **What is the loss for behaviour cloning when the corpus has
   only single-sample trajectories?** v0 uses cross-entropy
   (§9.1 falls back to one-hot next-token), which is the standard
   BC fall-back but loses the trajectory-level signal that motivated
   the RL framing.

These under-specifications are not bugs; they are the **research
content** of the programme. Each one is a decision that the
toy-task experiments will inform.

---

## 13. Deferred extensions: structure lifecycle (v1.5 / v2 / v3)

v0 deliberately treats the simulator as a **closed-vocabulary,
fixed-anchor** system: every semantic structure that exists during
simulation was named in the static layer (§3) at calibration time.
This bounds expressivity at the static parameter count and is one of
the reasons the framework as currently written would not, on its own,
match transformer-scale generative behaviour at frontier scale.

The framework is **not yet fully fledged**: there are three mechanisms
that the §2–§8 narrative leaves implicit but that v0 cannot express,
and that are needed to capture the productive expressivity of human
language. They are named here as deferred-but-planned extensions, so
the v0 specification doesn't read as if it were the final
architecture.

Each mechanism has a v# milestone in the broader programme document
(`docs/Semantic_Simulator_RL_Calibration_Programme.md`, §10).

### 13.1 Destruction / retirement (v1.5: simplest, prototype-first)

**What it is.** Each particle carries an age $\tau$ that increments
per simulation step, and a salience $s(\tau)$ that decays with age.
When salience drops below a threshold, the particle is *demoted* to
a low-mass background field rather than erased. Demoted particles
can be re-promoted by content-similarity to new evidence.

**Why prototype this first.** It is the simplest of the three (a
single scalar plus a threshold per particle), has the largest
computational payoff (bounds state growth), and can be added to v1
without redesigning the integrator. Failure to *need* it is itself
informative — it would mean the toy task doesn't require unbounded
context.

**Physics analogue.** Radioactive decay with a half-life. In MD,
this is the equivalent of solute particles being absorbed back into
a solvent bath when they lose kinetic energy.

**Cognitive-science analogue.** Salience decay in centring theory;
working-memory decay in cognitive-load models.

**Open design questions.**

1. Is salience scalar or vector-valued?
2. What is the re-promotion rule (content-similarity threshold, or
   explicit pointer-via-coreference)?
3. Is destruction reversible (demotion + re-promotion) or final
   (erasure)? v1.5 picks demotion.
4. How does destruction interact with composite structures whose
   parents have been destroyed?

### 13.2 Creation of new semantic structures (v2)

**What it is.** When two particles enter a specific configuration
(spatial proximity + property compatibility + sufficient mass-energy
product), they fuse into a composite particle whose position is a
function of the parents' (e.g., weighted Fréchet mean), whose mass
is the sum minus a binding-energy cost, and whose attractors are
inherited by a small combination rule plus an RL-calibrated
correction.

**Physics analogue.** Reactive molecular dynamics (ReaxFF), where
bond formation/breaking is a learned function of local geometry and
charge state. Particle-physics field theory: creation operators
acting on the vacuum.

**Computer-science analogue.** Petri-net token production; pi-calculus
process spawning; differentiable neural-computer write operations;
slot-binding in slot-attention models.

**Linguistic motivation.** Compounding ("smart" + "phone" →
"smartphone"), novel metaphor, idiomatic crystallisation, and any
case where two existing concepts produce a new concept whose
semantic position is not the simple mean of the parents'.

**Open design questions.**

1. **Triggering rule.** What configuration triggers binding?
   Candidates: PMI threshold (over the corpus distribution),
   property-compatibility check, spatial-proximity criterion, or some
   combination. ReaxFF's bond-order term is the RL-calibrated
   analogue.
2. **Inheritance.** Does the composite have its own attractor in
   $V_{\text{wells}}$, or does it inherit the parents'? Likely a
   small hand-designed combination rule plus a learned correction.
3. **Reversibility.** Can the composite split back? Yes, and this
   couples directly to the destruction mechanism (§13.1).
4. **Parameter scope.** Does each composite get its own
   RL-calibrated parameters (full freedom, parameter explosion), or
   does composition follow a fixed rule (no extra parameters,
   limited expressivity)?

### 13.3 Execution of semantic structures (v3)

**What it is.** A particle's state is augmented from $(x, \dot{x})$
to $(x, \dot{x}, \hat{O})$ where $\hat{O}$ is an operator acting on
a field associated with other particles. When particle $A$ enters
the operating range of particle $B$, $B$'s state transforms via
$\hat{O}_A$. This is the simulator's analogue of function
application.

**Physics analogue.** Quantum-field-theory interactions: each
particle carries creation/annihilation operators that act on the
vacuum. The semantic-space version is: each verb-like or
operator-like particle carries a transformation rule that, when
composed with arguments via spatial proximity, modifies them.

**Mathematical machinery already in the framework.** PARF (§4.3) is
already a constant-direction force; *promoting* properties to
operators (matrices acting on $x$, not just constant force vectors)
is the natural extension. Non-commutative composition is exactly the
regime where the Lie-group structure cited in
`Gueorguiev2025LieGroups`, `BaezMuniain1994`, and `Nakahara2003`
does the heavy lifting.

**Linguistic motivation.** Verbs and other operator-like words act
on their arguments. "The man saw the dog" $\neq$ "the dog saw the
man" — the composition is non-commutative. Encoding this cleanly
requires operator-valued primitives.

**Open design questions.**

1. **Operator parameterisation.** Linear ($d \times d$ matrices)?
   Affine? Non-linear? The $d^2$ parameter cost per operator-valued
   particle is significant.
2. **Commutation structure.** Which operators commute, which don't?
   Lie-algebra structure constants are the natural specification.
3. **Higher-order execution.** Can operators take operators as
   arguments (function composition, higher-order functions)?
4. **Coupling to creation.** Composite structures (§13.2) may be
   operator-valued; how does inheritance of operator state work?

### 13.4 What the v0 → v3 trajectory buys

If all three mechanisms are added cleanly, the simulator becomes a
**dynamical system on a state space that itself evolves** —
particles can be created, destroyed, modified, and operated on by
each other during simulation. This is structurally the same level of
expressivity as Turing-complete formal systems, and is no longer
bounded by the static parameter count.

The probability assessment in the programme document (§1) — that
this could compete with frontier transformers — is conditional on
v2/v3 working. v0 alone, even calibrated perfectly, will not.

The four caveats from the discussion that produced this section are
on record here:

1. **Combinatorial state explosion.** Creation without aggressive
   destruction is computationally intractable for long contexts.
2. **Sample complexity rises with discreteness.** Discrete
   creation/destruction rules are non-differentiable; RL of
   structural rules is more sample-hungry than RL of continuous
   coefficients.
3. **Compositional rule coverage is open-ended.** Each new
   linguistic phenomenon may need its own binding rule; there is no
   guarantee the rule count stays small.
4. **Frontier scale is partly memorisation, not productivity.** A
   productivity-focused simulator will not match transformer
   benchmarks that emphasise factoid recall, regardless of how good
   the productive mechanisms are.

These are not obstacles to the v0 / v1 work; they are the open
questions that v2 / v3 must address.

---

## 14. Hand-off to M1 / M2

With this document in place, the next two milestones are:

**M1:** Verify the parameter classification table (§10) by running
each static estimator on a small corpus and confirming the output
shapes and ranges. Fixed cost, ~1–2 days.

**M2:** Implement the simulator pseudocode of §8 in a single
Python module (`semsim_simulator/`), wire it to a
behaviour-cloning training loop (§9.1), generate a 100-token PCFG,
calibrate, and report perplexity vs trigram. Fixed cost, ~1–2 weeks.

If M2's reported perplexity is within 2× of trigram, the programme
proceeds to M3 (intrinsic-reward RL). If not, the failure mode
identifies which §12 under-specification needs revision and we
iterate v0 → v0.1.

---

*End of v0 EOM. The expectation is that v1 will already require at
least 3 of the §12 items to be made explicit, that v1.5 will add the
destruction/salience-decay mechanism (§13.1) as a prototype, and
that v2 / v3 will introduce structure-creation (§13.2) and
operator-valued execution (§13.3) — the mechanisms that v0
deliberately omits but that the framework needs to model the full
expressivity of human language.*
