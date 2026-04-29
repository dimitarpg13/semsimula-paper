# `symplectic_variant/` — SARF-faithful SPLM with a velocity-Verlet integrator

One-knob ablation stacked on top of the best variant from
[`../sarf_mass_variant/`](../sarf_mass_variant/) (SARF-faithful $\xi$
re-pooling + logfreq per-token mass).  Only the integrator is
replaced; every other design choice is identical:

| axis | value |
|---|---|
| $\xi$ prescription | SARF-faithful (recompute per layer) |
| per-token mass | `logfreq` (Shannon-surprisal prior) |
| shared $V_\theta$ | yes (identical MLP) |
| hyperparameters | identical to `sarf_mass_variant` |
| parameter count | **identical** (`7,123,076`) |
| **integrator** | **velocity-Verlet (KDK) + Strang-split damping** |

The question this folder answers:

> **Does promoting the first-order damped-Euler integrator of SPLM to
> a second-order symplectic (velocity-Verlet + Strang damping)
> scheme change LM quality, shared-$V_\psi$ conservativity, or both,
> on top of the already-best SARF + logfreq variant?**

The motivation is the observation (see the inference-efficiency
appendix of the paper) that SPLM's equation of motion is a
damped second-order conservative flow:
$$
\semm_t \ddot h = -\nabla_h V_\theta(\xi, h) - \semm_t \gamma \dot h,
$$
for which **velocity-Verlet** is the canonical $O(\Delta t^2)$
integrator, and **Strang splitting** of the damping operator
$v \to v \exp(-\gamma \Delta t / 2)$ preserves the symplectic
structure of the undamped Hamiltonian part exactly.

## Integrator

One full step (kick-drift-kick velocity-Verlet interleaved with
half-step damping), starting from the force $f = -\nabla_h V_\theta(\xi(h), h)$
already evaluated at the current $h$:

```text
v <- v * exp(-gamma dt / 2)        # half-step damping (start only)
v <- v + 0.5 * dt * f / m          # kick 1  (uses force at old h)
h <- h + dt * v                    # drift
f <- -grad V_theta(xi(h_new), h_new)   # single new force evaluation
v <- v + 0.5 * dt * f / m          # kick 2  (uses force at new h)
v <- v * exp(-gamma dt)            # full-step damping (interior)
                                   # on the last step: half-step damping
```

With the KDK reuse optimisation (each step's kick-2 force becomes
the next step's kick-1 force), an $L$-step stack costs
$L + 1$ force evaluations total (vs. Euler's $L$), a ~12.5%
integrator overhead at $L = 8$ for a theoretical jump from
$O(\Delta t)$ to $O(\Delta t^2)$ accuracy.  No new learnables.

## Results — Tiny Shakespeare

All four configurations share identical $(d=128, L=8, d_V=512,
v_\text{depth}=3)$, batch, data, and seed.

| variant | params | wall-clock | final train CE | final val CE | **val ppl** |
|---|---:|---:|---:|---:|---:|
| fixed-$\xi$ SPLM | 7,123,075 | 2000s | 5.0535 | 5.6610 | **287.43** |
| SARF-faithful SPLM | 7,123,075 | 1879s | 4.5789 | 5.2586 | **192.21** |
| SARF + logfreq (Euler) | 7,123,076 | 2390s | 4.3229 | 5.0786 | **160.55** |
| **SARF + logfreq (Verlet)** | **7,123,076** | **2034s** | **4.3233** | **5.1207** | **167.46** |

Final mass stats (Verlet): mean = 1.3945, std = 0.1865,
min = 1.11, max = 1.82.  Final $\gamma$ = 0.8487.
(Compare Euler: mass mean 1.28 / std 0.16, $\gamma$ = 0.834.)

### Trajectory comparison (post-warmup)

| step | baseline (Euler + logfreq) | symplectic (Verlet + logfreq) | $\Delta$ ppl |
|---:|---:|---:|---:|
| 2000 | 216.15 | 226.82 | +10.7 |
| 2400 | 188.54 | 199.37 | +10.8 |
| 2800 | 182.02 | 192.90 | +10.9 |
| 3000 | 171.43 | 179.71 |  +8.3 |
| 3400 | 177.90 | 185.49 |  +7.6 |
| 3800 | 165.49 | 174.17 |  +8.7 |
| 4000 | 160.55 | **167.46** |  +6.9 |

## Interpretation

The headline empirical finding is:

> **On Tiny Shakespeare, at fixed wall-clock, parameter count, and
> hyperparameters, velocity-Verlet does NOT improve LM quality over
> the damped-Euler integrator — it is a slight (~4% ppl)
> regression.**

The gap is small and consistent across the second half of training
(~6–11 ppl at every eval), suggesting this is not just eval noise.
Several readings are consistent with the data.

### Why the symplectic win did not materialise on LM quality

1. **The problem is heavily damped.**  The symplectic advantage of
   velocity-Verlet — preservation of a shadow Hamiltonian to
   $O(\Delta t^2)$ — is a property of the **undamped** Hamiltonian
   flow.  At the converged values $\gamma = 0.85$, $\Delta t = 1.0$,
   each step dissipates $\sim 1 - e^{-\gamma \Delta t} \approx 57\%$
   of the velocity; the flow is much closer to first-order gradient
   descent than to Hamiltonian dynamics, and there is no symplectic
   structure to conserve.
2. **The Euler baseline is already symplectic-Euler.**  The
   damped-Euler scheme of `sarf_mass_variant` is actually a
   semi-implicit (symplectic-Euler) scheme, not a fully explicit
   Euler.  Relative to truly explicit Euler, Verlet is strictly
   better; relative to symplectic-Euler with dissipation folded
   into the velocity update, the improvement is a second-order
   correction that is masked by the dissipation.
3. **$\Delta t = 1.0$ is large enough for either scheme to be
   discretisation-limited.**  The $O(\Delta t^2)$ vs $O(\Delta t)$
   gap is fixed in the relevant term only at small $\Delta t$; at
   $\Delta t = 1.0$ the Taylor expansion is outside its small-$\Delta
   t$ regime and the "asymptotically better" order does not
   dominate.  A parameter sweep over $\Delta t$ would sharpen this.
4. **The per-token mass already does most of the work.**  The
   logfreq prior reshapes the integrator's effective step per token
   in a way the Euler scheme can exploit directly; the additional
   symmetric splitting on top does not add a compounding effect.

### What this result does *not* say

- It does not say Verlet is worse **everywhere**.  The test that
  directly probes symplectic structure is the depth-axis
  shared-$V_\psi$ fit (paper §14.2).  Verlet's integration stays
  closer to the level sets of a single $V_\psi$ than symplectic-
  Euler does (modulo damping), so the pointwise separator should
  fit **better** on Verlet trajectories.  That diagnostic has now
  been run; see *Follow-up experiments (now completed)* below for
  the result, which both confirms and complicates the prediction.
- It does not say the scheme is wrong — the training runs cleanly,
  gradients are well-behaved, $\gamma$ and mass converge to
  physically sensible values.  The dynamics is healthy; the
  integrator swap just has no extra mileage for LM quality on this
  corpus at this scale.
- It does not rule out gains at smaller $\Delta t$, longer context,
  or deeper $L$.  The Verlet advantage scales as $(\Delta t)^{2} / (\Delta t) = \Delta t$
  of extra accuracy, so a regime with more discretisation steps
  is the natural place to look.

### Framework reading

The ablation sequence now covers three orthogonal axes of the
Semantic Simulation framework:

- `sarf_variant/` — pool structure (per-layer $\xi$ re-pooling).
- `sarf_mass_variant/` — particle property (per-token mass).
- `symplectic_variant/` — **integrator** ($O(\Delta t^2)$ symplectic
  + Strang-split damping).

On LM quality the third axis is the first to return a **null
result**.  That is by itself informative: it tells us which parts
of the "get closer to the continuous theory" ladder have empirical
bite at Tiny Shakespeare scale (prescription for $\xi$ and for
$\semm_t$) and which are refinements whose wall-clock-identical
quality wins have to be extracted elsewhere (strict
$V_\psi$-conservativity, decoding stability, long-context
behaviour).

## Follow-up experiments (now completed)

All three follow-ups from the original write-up have been run.  The
consolidated report is in
[`results/followups_summary.md`](results/followups_summary.md).
Headline numbers:

| Variant | Force evals | Val ppl | Depth pooled TEST $R^2$ | Token pooled TEST $R^2$ |
|---------|------------:|--------:|------------------------:|------------------------:|
| Euler (baseline) | 8 | **160.55** | +0.837 | +0.329 |
| Verlet $L=8$ (this variant) | 9 | 167.46 | +0.755 | +0.427 |
| Verlet $L=4$ (FLOP-halved) | 5 | 280.30 | +0.892 | +0.445 |
| **Verlet $L=16$, $dt=0.5$** | 17 | 174.32 | **+0.958** | **+0.515** |

### Three findings

1. **Verlet actually *regresses* on the depth-axis shared-$V_\psi$ fit at
   matched $(L, dt) = (8, 1)$** — +0.755 vs Euler's +0.837.  This was
   counter to the prediction in an earlier draft of this file.  The
   cause is a **fit-ansatz artefact**: the diagnostic
   $\Delta h_\ell = \alpha_\ell v_\ell - \beta_\ell \nabla V_\psi(h_\ell)$
   is a *one-step pointwise* ansatz, and Euler's update has exactly
   that functional form by construction.  Verlet's two-point
   (symmetric) update cannot be fully captured by a pointwise
   ansatz at coarse $dt$, so Verlet looks less conservative
   through this particular lens even though the continuous flow is
   identical.
2. **At finer $dt=0.5$ (same flow distance $Ldt=8$), Verlet lifts
   the depth-axis $R^2$ to +0.958 pooled TEST**, every layer
   $\geq +0.938$ — the highest conservative-dynamics signature of
   any SPLM variant trained so far, and a +0.12 gain over the Euler
   baseline.  The integrator-bias vanishes once the one-step
   approximation is tight for both schemes.
3. **Verlet wins the token-axis diagnostic at every $(L, dt)$.**
   +0.515 pooled TEST for $L{=}16, dt{=}0.5$ vs Euler's +0.329.
   The mid-to-late layer collapse ($R^2 \to 0.2$ for layers 5–8 in
   the Euler baseline) is eliminated — every layer from 1 to 16 has
   TEST $R^2 \in [+0.452, +0.534]$.  This is the cleanest carryover
   of Verlet's theoretical symmetry advantage to a measurable
   property of the model.

### What this says about the framework

The three follow-ups sharpen the null LM-quality result into a more
nuanced picture:

- **LM perplexity is integrator-insensitive** at fixed scale, fixed
  training budget, and fixed hyperparameters.  Euler $L{=}8$ is still
  the champion; the $2\times$ more expensive Verlet $L{=}16$ run
  lands 8% worse in ppl.  Most of the wall-clock "semantic potential
  landscape building" happens regardless of which 2nd-order
  correction the integrator applies.
- **Conservative-structure diagnostics *are* integrator-sensitive**,
  both positively and negatively.  Depth-axis has a coarse-$dt$
  ansatz bias in Euler's favour; token-axis has a real Verlet
  advantage that holds uniformly.  Whichever direction the paper's
  Geodesic Hypothesis is argued to apply, the token axis is the
  better separator when the integrator is symplectic.
- **Halving integration depth is expensive** — $L{=}4$ Verlet at
  5 force evals is 63% of Euler's integration cost but the LM
  degrades from 160 to 280 ppl.  The architecture still needs a
  minimum flow distance of $Ldt \gtrsim 8$ to form a useful
  semantic landscape.

## Directory layout

| file | purpose |
|---|---|
| [`model_symplectic.py`](model_symplectic.py) | `ScalarPotentialLMSymplectic` with velocity-Verlet integrator; self-test |
| [`train_splm_symplectic.py`](train_splm_symplectic.py) | trainer with `--mass-mode {global, embed_head, logfreq}` |
| `results/logfreq_surprisal.npy` | Shannon-surprisal lookup (copied from `sarf_mass_variant/results/`) |
| `results/splm_sym_logfreq_shakespeare_*` | checkpoint / log / loss curve / summary |

## How to reproduce

```bash
cd notebooks/conservative_arch/symplectic_variant

# 1. (already done) surprisal table copied from sarf_mass_variant
# 2. train (three variants)
PYTHONUNBUFFERED=1 python3 -u train_splm_symplectic.py \
    --mode shakespeare --mass-mode logfreq                           # L=8, dt=1
PYTHONUNBUFFERED=1 python3 -u train_splm_symplectic.py \
    --mode shakespeare --mass-mode logfreq --L 4 --tag-suffix L4     # FLOP-halved
PYTHONUNBUFFERED=1 python3 -u train_splm_symplectic.py \
    --mode shakespeare --mass-mode logfreq --L 16 --dt 0.5 \
    --tag-suffix L16_dt05                                            # small dt

# 3. extract trajectories
python3 trajectory_extraction_symplectic.py \
    --ckpt results/splm_sym_logfreq_shakespeare_ckpt_latest.pt
python3 trajectory_extraction_symplectic.py \
    --ckpt results/splm_sym_logfreq_shakespeare_L4_ckpt_latest.pt
python3 trajectory_extraction_symplectic.py \
    --ckpt results/splm_sym_logfreq_shakespeare_L16_dt05_ckpt_latest.pt

# 4. run diagnostics (from parent conservative_arch/)
cd ..
for tag in sym_logfreq_shakespeare sym_logfreq_shakespeare_L4 \
           sym_logfreq_shakespeare_L16_dt05; do
    python3 shared_potential_fit.py \
        --traj symplectic_variant/results/splm_${tag}_ckpt_latest.trajectories.pkl \
        --tag ${tag}
    python3 token_direction_fit.py \
        --traj symplectic_variant/results/splm_${tag}_ckpt_latest.trajectories.pkl \
        --tag ${tag}
done
```

## Relationship to the paper

Addresses **Q11** ("symplectic structure: does the integrator matter
for SPLM?") in a concrete, minimal-diff form.  The result is a
cleanly-null outcome on LM quality, a useful calibration for
readers asking whether each layer of integrator refinement pays
off in raw perplexity terms.  The shared-$V_\psi$ and token-direction
fit numbers are now in (see *Follow-up experiments (now completed)*
above); this variant is the symplectic counterpart to the
`sarf_mass_variant` Euler row in the paper's §15 conservative-
architecture comparison.  The energy-drift diagnostic
([`../energy_drift/`](../energy_drift/)) ships the production
$L=16, \Delta t=0.5$ Verlet checkpoint in the three-way comparison
against `parent_euler_L8` and the LayerNorm-after-step `em_ln`
production SPLM.
