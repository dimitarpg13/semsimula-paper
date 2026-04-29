# Verlet-SPLM follow-up experiments -- summary

Three follow-ups to the `symplectic_variant/` baseline (velocity-Verlet
+ Strang-split damping, `logfreq` mass, Tiny Shakespeare, $L=8$, $dt=1$):

| # | Task | Variant | Force evals per forward | Wall-clock |
|---|------|---------|-------------------------|-----------|
| 1 | Baseline re-run of diagnostics on the **already-trained** $L=8$ Verlet checkpoint | `sym_logfreq_shakespeare` | 9 | 0 (just eval) |
| 2 | **FLOP-matched** Verlet (halve integration depth) | `sym_logfreq_shakespeare_L4` | 5 | 17 min |
| 3 | **Small-$dt$** Verlet (halve step, double layers, same total flow $Ldt=8$) | `sym_logfreq_shakespeare_L16_dt05` | 17 | 49 min |

The Euler `logfreq` baseline (`sarf_mass_variant/` with $L=8$, $dt=1$) is
used as the reference point throughout.

---

## 1. Language-model perplexity

| Variant | Integrator | $L$ | $dt$ | Force evals | Final val ppl |
|---------|-----------|----|-----|-------------|--------------:|
| Euler (baseline) | damped semi-implicit Euler | 8 | 1.0 | 8 | **160.55** |
| Verlet | velocity-Verlet + Strang | 8 | 1.0 | 9 | 167.46 |
| Verlet-L4 | velocity-Verlet + Strang | 4 | 1.0 | 5 | 280.30 |
| Verlet-L16-dt05 | velocity-Verlet + Strang | 16 | 0.5 | 17 | 174.32 |

**Finding.** No Verlet variant beats Euler on LM quality.  Halving $L$ at
fixed $dt$ is disastrous (280 ppl): the model needs at least $\approx 8$
units of "flow distance" $Ldt$ to form a useful semantic potential
landscape.  Refining $dt$ while keeping $Ldt = 8$ (L16-dt05, 2 $\times$
the cost) does **not** recover Euler's perplexity -- it is the worst
Verlet variant in our matched-flow regime.

The most likely explanation is that the training hyperparameters (lr,
warmup, steps) were tuned for the coarser $L=8$ Euler baseline and do
not transfer cleanly to an integrator with $\sim 2\times$ the effective
depth.

---

## 2. Depth-axis shared-$V_\psi$ fit (conservative-dynamics test)

Strict test: can a *single* scalar $V_\psi$, optimised jointly across
all layers, all tokens, all training sentences, explain the depth-wise
dynamics $\Delta h_\ell \approx \alpha_\ell v_\ell - \beta_\ell
\nabla V_\psi(h_\ell)$?

| Variant | Layers fit | Pooled TRAIN $R^2$ | Pooled TEST $R^2$ | Min layer TEST $R^2$ |
|---------|-----------:|-------------------:|------------------:|---------------------:|
| Euler (baseline) | 1..7 | +0.911 | +0.837 | +0.670 |
| Verlet L=8 | 1..7 | +0.877 | +0.755 | +0.664 |
| Verlet L=4 | 1..3 | +0.974 | +0.892 | +0.845 |
| Verlet L=16 dt=0.5 | 1..15 | **+0.975** | **+0.958** | **+0.938** |

### Why Verlet $L=8$ **regressed** on this test

Surprising, against the expectation in the $L=8$ `README.md`:
Verlet with the same $(L, dt)$ as Euler actually fits the shared-$V_\psi$
ansatz *worse*, not better (0.755 vs 0.837 pooled TEST).

The reason is that the fit ansatz is *one-step pointwise*:
$\Delta h_\ell = \alpha_\ell v_\ell - \beta_\ell \nabla V_\psi(h_\ell)$.
Euler's update is literally
$\Delta h = dt(v_{\text{prev}} + dtf(h_\ell)/m)/(1 + dt\gamma)$ --
a one-step pointwise update centred at $h_\ell$.  The ansatz matches
Euler's update structure exactly (modulo the re-parameterisation
$\alpha_\ell, \beta_\ell$) so it scores high by construction.
Verlet's update is a *symmetric two-point* update using forces at both
$h_\ell$ and $h_{\ell+1}$.  The pointwise ansatz cannot fully express
this, so at identical $dt$ Verlet appears less "conservative" through
this particular fit -- even though the true continuous flow is
identical.

### Smaller $dt$ recovers, with room to spare

With $dt=0.5$, both integrators approach the continuous flow and the
ansatz's one-step approximation becomes tight again.  Verlet
L=16-dt=0.5 achieves **+0.958 pooled TEST**, every layer $\geq +0.938$
-- the strongest conservative-dynamics signature of any SPLM variant
trained so far.  L=4 Verlet likewise scores high (+0.892) but only
fits 3 layers, so the number is less informative.

### Take-away

Verlet + small $dt$ does produce trajectories that are more
conservative-looking in the depth axis, but this alone does **not**
translate into better LM quality.  The diagnostic and the downstream
metric can move in opposite directions along the integrator-refinement
axis.

---

## 3. Token-axis shared-$V_\psi$ fit (sequence-time conservativeness)

Same fit, applied along the *token* axis at fixed layer $\ell$.  This
is the natural "time axis" of autoregressive inference.

| Variant | Pooled TEST $R^2$ | Median layer TEST | Min layer TEST |
|---------|------------------:|------------------:|----------------:|
| Euler (baseline) | +0.329 | +0.347 | +0.195 |
| Verlet L=8 | +0.427 | +0.466 | +0.290 |
| Verlet L=4 | +0.445 | +0.473 | +0.290 |
| Verlet L=16 dt=0.5 | **+0.515** | **+0.512** | **+0.296** |

**Finding: Verlet wins the token axis at every variant.**
The mid-to-late layer collapse seen in the Euler baseline (layers 5-8
drop from +0.347 to +0.195) is *eliminated* in every Verlet variant.
L=16-dt=0.5 gives a remarkably uniform profile: every layer $\ell
\geq 1$ lies in $[+0.452, +0.534]$.

### Why the asymmetry (Verlet helps token axis more than depth axis)

The fit ansatz is symmetric in its logical form on both axes, but the
integrator only acts along depth.  Along the token axis the integrator
influences trajectories *indirectly* through the learned $V_\theta$, $m$,
$\gamma$ -- it does not generate a one-step map between adjacent tokens.
Verlet's second-order accuracy and time-reversal symmetry bias the
learned $V_\theta$ to be smoother in $h$, which in turn makes the
per-token dynamics at fixed $\ell$ look more like a shared-potential
flow.  The result is a more uniform per-layer $R^2$ in the token
direction even when $dt$ is coarse.

---

## 4. Unified scoreboard

| Variant | Val ppl | Depth pooled TEST | Token pooled TEST | Token min layer |
|---------|--------:|------------------:|------------------:|----------------:|
| Euler (baseline) | **160.55** | +0.837 | +0.329 | +0.195 |
| Verlet L=8 | 167.46 | +0.755 | +0.427 | +0.290 |
| Verlet L=4 | 280.30 | +0.892 | +0.445 | +0.290 |
| Verlet L=16 dt=0.5 | 174.32 | **+0.958** | **+0.515** | **+0.296** |

---

## 5. Interpretation

1. **LM perplexity is not a good proxy for "how conservative the
   dynamics look" at fixed model size.**  Verlet-L16-dt=0.5 is
   dramatically more conservative on both diagnostic axes yet is
   7--9% *worse* on ppl than plain Euler.  This says the two metrics
   measure genuinely different things.

2. **The depth-axis shared-$V_\psi$ test has an integrator-bias
   artefact at coarse $dt$.**  Euler's one-step pointwise update
   matches the fit ansatz by construction, inflating Euler's score.
   Verlet's two-point update cannot be perfectly captured at coarse
   $dt$.  Any fair comparison of "how well does a single scalar
   describe depth dynamics?" should either use small $dt$ (where
   Verlet dominates by +0.12) or use a two-point ansatz that matches
   Verlet's update structure.

3. **The token-axis diagnostic is where Verlet's theoretical advantage
   translates most cleanly.**  Verlet's time-reversal symmetry makes
   the per-token dynamics at fixed layer more consistent with a
   shared-potential description, uniformly across all layers and
   without any ansatz-matching caveat.

4. **Integration depth matters more than integration order for LM
   quality** (L4 Verlet -> 280 ppl is much worse than L8 Euler).
   This is consistent with standard intuition: halving the effective
   model depth loses far more than any accuracy gain from a better
   integrator.

---

## 6. Artefacts

### Per-variant summaries
- `results/splm_sym_logfreq_shakespeare_summary.md` -- Verlet L=8 training
- `results/splm_sym_logfreq_shakespeare_L4_summary.md` -- Verlet L=4 training
- `results/splm_sym_logfreq_shakespeare_L16_dt05_summary.md` -- Verlet L=16 dt=0.5 training

### Diagnostic outputs (under `../results/`)
- Depth-axis: `sharedV_sym_logfreq_shakespeare{,_L4,_L16_dt05}_summary.md`
- Token-axis: `tokdir_sym_logfreq_shakespeare{,_L4,_L16_dt05}_summary.md`
- Figures:    `sharedV_*_fig.png`, `tokdir_*_fig.png`

### Trajectory pickles
- `results/splm_sym_logfreq_shakespeare{,_L4,_L16_dt05}_ckpt_latest.trajectories.pkl`

### Checkpoints
- `results/splm_sym_logfreq_shakespeare{,_L4,_L16_dt05}_ckpt_latest.pt`
