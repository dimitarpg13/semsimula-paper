# Composing single-layer inferences: is the layer stack a flow, or a sequence of maps?

> **Status: design note, nothing measured.** Every number below is a
> derivation, a configuration value, or a clearly-labelled illustration.
> The protocol in §8 is what would produce measurements, and it needs no
> training: it runs against checkpoints that already exist.

---

## 0. Summary

The question is whether *k* applications of a single conservative layer
compose into a *k*-hop inference, and whether the resulting object is the
same as a *k*-layer model. Three findings come out of working it through,
and the third was not obvious in advance.

1. **What you carry between applications is the whole story.** Chaining
   without carrying the position pair gives *exactly* gradient descent on
   `V_theta` — a first-order system, derived in §3.1. Chaining with it
   carried is bit-for-bit the L=k model. So "k separate inferences" versus
   "one k-layer model" is precisely the second-order structure, and the
   contrast is measurable at zero training cost.

2. **There are two ways to leave the trained configuration**, and only one
   of them tests the framework. Running more steps at the trained step size
   asks "do extra hops help". Running more steps at *fixed total
   integration time* asks whether the stack is a **discretised flow** at
   all — which is what `18_riemannian_geometry` and the whole geodesic
   reading presuppose, and which nobody has checked.

3. **The depth code decides whether that second test is valid.** `V_theta`
   is one shared bank whose depth dependence is an additive code on `xi`.
   The obvious way to extend past L — cycling the code — does *not* refine
   the trained model. By homogenisation it converges to the flow of the
   **mean** potential, which is a different model. Refinement requires
   holding each code over its share of the interval. §6 derives this and
   Figure 3 shows it.

---

## 1. Two readings of a layer stack

A conservative PARF forward pass produces a sequence of hidden states
`h_0, h_1, ..., h_L`. There are two incompatible ways to read it.

**As a sequence of maps.** The model is a composition of L learned
functions. L is an architectural constant. The intermediate states have no
meaning beyond being intermediate. Nothing about `h_3` is comparable to
`h_5` except that both are on the way to `h_L`.

**As a discretised flow.** There is a continuous trajectory
$h(\tau)$ on $\tau$ in $[0, T]$ obeying a differential equation, and the
layer stack is one particular discretisation of it with $N = L$ steps of
size $\Delta t = T/L$. The intermediate states are *samples of a curve*.

The framework commits to the second reading everywhere it matters. The
Jacobi metric is layer-indexed, $\Omega_{\ell}^2 = 2 T_{\ell} m$. The
geodesic residual is a second difference of the $h_{\ell}$ sequence. The
CfC integrator is named for continuous-time flow. `18d` argues the
geometric capabilities are a property of the second-order *dynamics*.

**Under the first reading none of that is well-posed.** A second difference
of a sequence of unrelated maps is not an acceleration; it is an artifact
of how many maps there happen to be. The distinction is therefore not
philosophical — it decides whether §18's residual measures a geometry or a
discretisation.

Nothing in training distinguishes the two. The loss is computed on `h_L`
only, at one value of L and one value of dt. Whether the weights also
describe a continuous flow is a property the objective never asked for.

---

## 2. What actually flows between layers

### 2.1 The state is a position pair

The integrator carries no explicit velocity. It carries `(h, h_prev)`, and
velocity is recovered by finite difference:

```python
def decode_velocity(h, h_prev, dt):
    """``v = (h - h_prev)/dt`` -- the implicit-velocity convention."""
    return (h - h_prev) / dt
```

So the genuine state of the dynamical system is a pair, and the phase-space
point is

$$(h, v) = \left(h, \frac{h - h_{\text{prev}}}{\Delta t}\right)$$

### 2.2 The stack starts from rest

From `model_parf.py`:

```python
h = h0
h_prev = h0   # velocity proxy starts at 0
```

Therefore $v_0 = 0$ exactly. This single line is load-bearing for
everything that follows, and it is why L=1 is structurally different from
L=2 rather than merely shallower.

```mermaid
flowchart LR
  A["h&#95;0 with h&#95;prev set to h&#95;0<br>so v&#95;0 is zero"] --> B["layer 0<br>S&#95;dt"]
  B --> C["h&#95;1 and v&#95;1<br>v&#95;1 is a function of h&#95;0 alone"]
  C --> D["layer 1<br>S&#95;dt"]
  D --> E["h&#95;2 and v&#95;2<br>depends on h&#95;0 AND h&#95;1"]
  F["v back to 0 (first order again)"]
  C -.->|reset h&#95;prev and discard v| F
```

At $\ell = 1$ the velocity is a deterministic function of $h_0$ alone,
because there was no prior velocity for it to depend on. Only from
$\ell = 2$ does the state carry information the position does not.

---

## 3. Carried versus reset: the exact difference

Write the one-step integrator as a map on phase space,

$$S_{\Delta t} : (h, v) \longmapsto (h', v')$$

and let $\pi_h$, $\pi_v$ be the coordinate projections.

**Carried chaining**, which is what a k-layer forward pass does:

$$h_k = \pi_h \left( S_{\Delta t}^{ k} (h_0, 0) \right)$$

**Reset chaining**, which is what re-running the model on its own output
does if you keep only `h`:

$$h_k = G^{k}(h_0), \qquad G(h) := \pi_h \left( S_{\Delta t}(h, 0) \right)$$

These agree at $k = 1$ and differ for every $k \ge 2$ whenever the force is
non-zero.

### 3.1 Reset chaining is exactly gradient descent

This is sharper than "reset chaining is first-order". Take velocity-Verlet,
the scheme the CfC/BAOAB variants reduce to when the harmonic part is
handled explicitly:

$$h' = h + v \Delta t + \frac{\Delta t^{2}}{2m} f(h)$$

Set $v = 0$ and substitute the conservative force $f = -\nabla U$:

$$G(h) = h - \frac{\Delta t^{2}}{2m}  \nabla U(h)$$

**That is a gradient-descent step on $U$ with learning rate
$\eta = \Delta t^{2}/2m$.** The scheme-dependent constant changes with the
splitting (ABOBA, BAOAB, the damping multiplier), but the *form* does not:
from rest, one step is always $h' = h - \eta \nabla U(h)$ for some
$\eta \gt 0$.

So reset chaining for k applications is **k steps of gradient descent on
the learned potential**. It is not merely "first-order-like"; it is the
overdamped limit of `18d`, reached without retraining.

This has a useful corollary. `18d`'s first-order reduction (SPLM-1) is a
trained ablation. Reset chaining gives a first-order dynamical system built
from *the same weights*, at zero cost — an eval-mode instance of the
partition principle rather than a retrained one.

### 3.2 What inertia buys, visibly

![Carried versus reset](figures/flow_or_maps/fom_carried_vs_reset.png)

Panel A: the carried trajectory accumulates velocity, overshoots, and
oscillates. The reset trajectory's velocity decays monotonically because at
each application it is recomputed from the local gradient.

Panel B is the consequence. The reset run descends monotonically toward the
minimum — gradient descent, as derived. The carried run **overshoots
through zero and comes back**. No first-order system can do that. The
difference between the two curves is the entire contribution of the
second-order structure, at fixed weights and fixed potential.

---

## 4. Two axes of extrapolation

![The two axes](figures/flow_or_maps/fom_axes.png)

Let the trained configuration be $(L, \Delta t_{\text{tr}})$ with total
integration time $T_{\text{tr}} = L \Delta t_{\text{tr}}$.

### 4.1 Axis 1 — more hops at the trained step size

Run $N \gt L$ steps at $\Delta t = \Delta t_{\text{tr}}$. Total time grows as
$T = N \Delta t_{\text{tr}}$.

This asks a capability question: *do additional applications add useful
routing?* It is the natural reading of "combine several single-hop
inferences into a multi-hop one", and it is the recurrent / universal
-transformer extension.

It does **not** test the flow hypothesis, because it changes the integration
interval rather than its resolution.

### 4.2 Axis 2 — refinement at fixed integration time

Run $N$ steps at $\Delta t = T_{\text{tr}}/N$, so that
$N \Delta t = T_{\text{tr}}$ always.

This asks a structural question: *is the trained map a discretisation of
something?* Every $N$ targets the same continuous trajectory over the same
interval; only the resolution changes.

### 4.3 Why axis 2 is the framework test

If the stack is a genuine flow, then the numerical trajectory converges to
the exact one. For an order-$p$ integrator the global error over a fixed
interval obeys

$$\lVert h_N^{(\Delta t)} - h(T) \rVert = O(\Delta t^{p}) = O\big( (T/N)^{p} \big)$$

so the outputs — and therefore the perplexities — should form a **Cauchy
sequence in N**. If instead the model is L specialised maps, changing N
changes the computed function outright and there is no reason for
convergence.

![Flow versus maps](figures/flow_or_maps/fom_flow_vs_maps.png)

Panel A is the flow case, integrated exactly: refining from N=2 to N=8 at
fixed T walks the discretisation onto the true curve. The N=2 trajectory is
visibly coarse, which is the honest picture of a two-layer model as a
discretisation.

Panel B is the subtlety that §6 is about, and it is not the failure mode
predicted before drawing it.

---

## 5. The consequence for the geodesic residual

`18_riemannian_geometry` computes a residual of the damped geodesic
equation from the model's own $h_{\ell}$ sequence. That requires a second
difference, hence three consecutive states, and its value depends on
$\Delta t$.

If PPL(N) at fixed T is **flat**, the $h_{\ell}$ sequence samples one
underlying curve and the residual is a property of *that curve* — a
geometric statement. If PPL(N) **degrades sharply**, the residual is a
property of the trained discretisation, and the correct scoping of every
geodesic claim becomes "at the trained step size", which is much weaker
than the paper currently states.

Refinement invariance is therefore a **precondition** for the geodesic
programme, not an optional extra. It has never been tested, and testing it
costs one evaluation sweep.

A second consequence is practical. At L=2 the residual has exactly one
second difference per token — three states, zero redundancy, no way to
estimate its noise. If refinement is valid, you can compute the residual at
N=8 or N=16 on an L=2 model and recover the statistics, because the
underlying curve is the same one.

---

## 6. The depth code, and why the refinement policy decides the answer

### 6.1 What the code does

`V_theta` is **one shared bank**. Depth enters only as an additive shift on
its input context:

```python
def _shift(self, xis):
    """Add the active layer's depth code to xis: (..., n_ctx, d)."""
    g = self._active_layer
    if not (0 <= g < self.n_layers):
        g = g % self.n_layers          # already cycles
    code = self.depth_code[g]
    ...
    return xis + code
```

So the potential at step $\ell$ is $U(h;  \xi + c_{\ell})$ with
$c_{\ell}$ of shape $n_{c} \times d$ a learned table of L entries.
The parameters are not replicated per layer — which is why L=2 and L=8
models have nearly the same parameter count, and why this is a *compute*
ladder rather than a parameter ladder.

Read as a flow, $c$ is a sampled function of time: $c_{\ell} = c(\tau_{\ell})$
with $\tau_{\ell} = \ell/L$. Read as maps, it is an arbitrary table.

### 6.2 Three refinement policies

![Depth code policies](figures/flow_or_maps/fom_depth_policies.png)

| policy | rule | what it preserves |
| --- | --- | --- |
| **cycle** | `c[j % L]` | nothing about the profile; raises its frequency |
| **hold** | `c[(j*L) // N]` | the trained step function on tau, refined |
| **interp** | linear in `tau` | a smooth curve through the trained samples |

Only **hold** is a refinement. It leaves the map $\tau \mapsto c(\tau)$
pointwise identical and subdivides the integration inside each piece, which
is exactly what "same trajectory, finer resolution" means.

### 6.3 Why cycling silently answers a different question

Cycling at large N alternates between the codes on a timescale
$\Delta t = T/N \to 0$. This is the classical **homogenisation** limit: a
system driven by a rapidly alternating potential converges to the dynamics
of the *averaged* potential,

$$\bar{U}(h;\xi) = \frac{1}{L}\sum_{\ell=0}^{L-1} U\left(h;  \xi + c_{\ell}\right)$$

Panel B of the flow-versus-maps figure shows exactly this: the cycled
N=8 trajectory converges, but it converges onto the flow of the **mean**
potential, not onto the trained two-step map.

So a refinement sweep using `cycle` does not fail loudly. It converges
cleanly to the wrong limit, and would be read as evidence *for* the flow
hypothesis while actually measuring a different model. **This is the trap
the experiment has to avoid**, and it is why the policy is a first-class
knob rather than an implementation detail.

### 6.4 Which policy belongs to which axis

- **Axis 1** (more hops, T grows): `cycle` is the right default. You are
  asking what a longer stack would do, and cycling is the standard
  recurrent extension. `hold` is meaningless here — there is no fixed
  interval to subdivide.
- **Axis 2** (fixed T, finer dt): `hold` is the only valid policy.
  `interp` is a *second*, distinct test — whether the learned code table is
  a sample of a smooth function of depth. `cycle` must not be used, per
  §6.3.

---

## 7. Implementation

The layer loop is

```python
for ell in range(cfg.L):
    h_new, h_prev_out = self._layer_step_ex(
        h, h_prev, m_b, gamma, dt, layer_idx=ell)
```

Four things are indexed by layer: `depth_code[g]` (already cycles),
`creation_gates[layer_idx]`, `destruction_gates[layer_idx]` and
`reverse_channel_scale[layer_idx]`. All four derive from the single
`layer_idx` argument, so the entire policy reduces to **choosing what index
to pass**:

```python
def _policy_index(j, N, L, policy):
    """Map extrapolated step j in [0, N) to a trained layer index."""
    if policy == 'cycle':
        return j % L                      # axis 1
    if policy == 'hold':
        return (j * L) // N               # axis 2 -- the refinement
    raise ValueError(policy)              # 'interp' needs code blending
```

`interp` is the one policy that cannot be expressed as an index, because it
needs a blended code; it requires a hook on `_shift` rather than on the
loop.

The sweep itself is then a context manager that patches `_stack_forward`
to run N steps with the chosen `dt` and index policy, plus a flag to reset
`h_prev = h` between steps for the first-order contrast of §3. Weights are
untouched; nothing is trained.

---

## 8. Protocol and pre-registered predictions

**Cost:** evaluation only, minutes per point, against checkpoints that
already exist. The step-28,500 L=8 joint checkpoint is usable today; the
L=2 ladder checkpoints will be usable when they land.

| gate | what | criterion |
| --- | --- | --- |
| 0 | N = L, dt = dt_tr, policy `hold` | **must reproduce the checkpoint's PPL exactly** -- this is the no-op case and any drift is a harness bug |
| 1 | reset-vs-carried at N = L | reset should be clearly worse; it is gradient descent (§3.1). **Now the programme's only clean velocity test** — the L=1 ladder point cannot serve, because it disables the registers as well (§9) |
| 2 | axis 1, `cycle`, N in 1..2L | capability question |
| 3 | axis 2, `hold`, N in 1..8 at fixed T | **the framework test** |
| 4 | axis 2, `interp` | is the code table a smooth sample? |

![Three outcomes](figures/flow_or_maps/fom_outcomes.png)

Pre-registered readings for gate 3, stated before any run:

| shape of PPL(N) at fixed T | reading |
| --- | --- |
| converges, changes shrinking like a power of 1/N | **FLOW.** The geodesic programme's precondition holds. Depth becomes an inference-time knob and the §18 residual can be computed at any N. |
| improves monotonically | the trained discretisation was **too coarse**. More layers at smaller dt is strictly better, and the trained L understates the architecture. |
| degrades away from N = L in both directions | **MAPS.** The stack is L specialised functions. Every geodesic claim must be re-scoped to the trained step size. |

The N < L direction is as informative as N > L and needs no policy at all
(running an L=8 model for 4 steps at double dt uses `hold` with no
extrapolation), so it is the cheapest first probe.

---

## 9. Caveats

- **Training never asked for refinement invariance.** The loss saw one
  (L, dt). A model can be an excellent language model and a poor flow. A
  negative result on gate 3 is therefore not a bug, it is information about
  what the objective selected for.
- **Some specialisation is expected at any N.** The informative quantity is
  the *shape* of PPL(N), not whether it is perfectly flat.
- **`V_phi` and any relaxation field stay explicit.** `baoab_cfc_lowrank`
  treats the harmonic part exactly, so refinement improves the explicit
  terms' accuracy but the two classes converge at different rates.
- **The resonance monitor does not cover this.** `omega*dt` detects
  instability, not accuracy loss at a stable-but-coarse step, and under
  `baoab_cfc_lowrank` it reports stiffness rather than a stability margin.
- **N=1 carries a static register bank.** `FOM_AXIS2_N` includes it, and at
  a single layer the register bank is *read* by the reverse channel but never
  *updated*: `salience` initialises to exactly 1.0, so the creation readout is
  multiplied by `(1 - blend) == 0` at layer 0, and only layers 1 and up can
  train the shared gate. The live L=1 ladder run shows it — `sig_max` pinned
  at its initialisation value while both L=2 arms differentiate within 50
  steps. See §6.1 of
  [`Depth_Ladder_and_Matched_Baseline_Protocol.md`](Depth_Ladder_and_Matched_Baseline_Protocol.md).
  So the N=1 *evaluation* is well defined and worth running, but it is not a
  point on the same curve as N>=2, and its endpoint should not be read as
  evidence about refinement.

- **An L=1 model with a live creation gate now exists as an option.**
  `register_salience_init` (default 1.0, bit-exactly the historical
  behaviour) opens `(1 - blend)` when set below 1; `0.5` is one decay step
  from the default. That is the instrument this document wants — a single
  trained layer whose Fock machinery is fully live, to chain *k* times and
  compare against an L=k model. It sits **outside** the depth ladder, since
  it is a different architecture from the L>=2 rungs. The depth-by-depth
  picture is in
  [`Fock_Mechanism_Efficiency_Across_Layer_Depth.md`](Fock_Mechanism_Efficiency_Across_Layer_Depth.md).

- **Read the reverse-channel gate before measuring anything about
  registers.** `tanh(reverse_channel_scale)` initialises to 0 and the warmup
  is `reverse_warmup_step / 4000`, so on a freshly built model the
  register-to-token path is shut and every probe reports registers doing
  nothing — at *any* depth. Cell 6b-8 prints the effective gate first for
  exactly this reason.

- **Gate 0 is not a formality.** If N = L with `hold` does not reproduce
  the checkpoint's perplexity bit-for-bit, nothing downstream means
  anything.
