# Energetic-minima alternatives to a free $V_\theta$

This folder implements the three falsification experiments that the
paper's §14.15 (design rationale: why we do not enforce energetic
minima by construction) flags as open follow-ups in §14.16 (Q11).

## What's in the paper

§14.15 (`subsec:cba-attractors`) establishes two uncomfortable facts
about the flagship SPLM:

- **Finding 1.** The trained scalar potential $V_\theta(\xi, h)$ is
  **unbounded below**.  The training loss only ever sees the gradient
  of $V_\theta$ along the realised trajectory, so the absolute scale
  and off-trajectory behaviour of $V_\theta$ are unpenalised.
- **Finding 3 (positive).** But SPLM's damped flow at $L = L_{\text{train}}$
  does exhibit prompt-dependent multi-basin structure with
  silhouette-optimal $K^\ast \in [2, 10]$.  The attractors live in
  the *combined* potential-plus-damping-plus-horizon system, not in
  $V_\theta$ alone.

The paper then argues, on six grounds (R1 – R6, see §14.15), that
attempting to redesign SPLM so that $V_\theta$ has finite local
minima would make the model less expressive, not more interpretable;
the "unbounded below" is a gauge, not a pathology.  That position
implicitly predicts the behaviour of three lightweight alternative
designs, which this folder implements and tests.

## The three variants

### (i) `--variant ln` — LayerNorm-after-step (option C)

Implementation: `model_ln.py`.  After each semi-implicit damped step
`h_{l+1} = h_l + dt * v_{l+1}`, project $h$ back to the unit-LayerNorm
shell (per-token mean 0, variance 1, affine-free).  Nothing about
$V_\theta$ changes.  Compactness of $S^{d-1}$ guarantees that any
continuous $V_\theta$ restricted to the shell has a finite minimum
by the extreme-value theorem.  This is the cheapest way to buy a
finite minimum without changing $V$'s functional form.

### (ii) `--variant sg --lambda-v0 1e-3` — scale-gauge (option D)

Implementation: same model as the SARF+mass baseline; only the
training loop changes.  Adds the loss-side regulariser

    L_total = L_CE  +  lambda_V * mean_{b,t}  V_theta(xi_0, h_0)^2

anchoring the absolute scale of $V_\theta$ at the **input**
embedding (not at the final hidden state $h_L$, so it does not
conflict with the LM cross-entropy objective).  The lambda is kept
small (1e-3) so the penalty is comparable to per-step gradient noise.

### (iii) `--variant gm --gm-K 64` — Gaussian-mixture head (option B)

Implementation: `model_gm.py`.  Replaces the free MLP $V_\theta$ with
a mixture of the paper's prescribed Gaussian wells (§`sec:well`):

    V_theta(xi, h) = sum_{k=1..K}  amp_k * (1 - exp(-kappa_k^2 * ||z - c_k||^2))

with learnable centres $c_k \in \mathbb{R}^{2d}$, amplitudes $amp_k = m_k \upsilon_k^2 > 0$,
and widths $\kappa_k > 0$ (both parameterised via softplus).
Structurally bounded: $0 \le V_\theta(z) \le \sum_k amp_k$, and each
centre is a local minimum (up to cross-well leakage).  This is the
**honest test** of the physics-prescribed well form at full SPLM scale.

## How to reproduce

Single-run smoke tests:
```bash
python3 model_ln.py           # smoke test of model_ln
python3 model_gm.py           # smoke test of model_gm
python3 train.py --variant ln --mode smoke
python3 train.py --variant sg --mode smoke --lambda-v0 1e-3
python3 train.py --variant gm --mode smoke --gm-K 32
```

Full Tiny Shakespeare runs (each is the same schedule as the baseline,
~20 – 25 min on an Apple-Silicon MPS device):
```bash
python3 train.py --variant ln --mode shakespeare
python3 train.py --variant sg --mode shakespeare --lambda-v0 1e-3
python3 train.py --variant gm --mode shakespeare --gm-K 64
```

Attractor analysis on all three (plus baseline) and cross-comparison:
```bash
bash run_attractor_pipeline.sh
python3 compare.py
```

## Expected outcomes (the paper's R1 – R6 position)

Per the paper's §14.15 rationale:

- **(i) LN and (ii) SG** should leave val ppl **essentially unchanged**
  (say, within 10 – 15 %) while producing a **narrower $V_\theta$ range**
  and comparable or slightly crisper basin structure.  That outcome
  confirms that the "unbounded below" of the flagship design is a
  gauge, not a pathology: pinning it has no material effect on the
  model's behaviour.
- **(iii) GM** should reproduce the **static-null behaviour** of the
  seven scalar-potential fits in §14.2 and **degrade LM quality**
  significantly.  That outcome confirms R5 (structurally bounded $V$
  is expressivity-limited at this scale).

**Falsification criterion.** A Gaussian-mixture SPLM that matches the
SARF-faithful val ppl 160.55 of the logfreq baseline would force the
paper's position to invert, from "structural boundedness is an
expressivity downgrade" to "a physics-prescribed $V$ suffices."  In
that event, §14.15 and §14.16 (Q11) should be rewritten to reflect
the new evidence.

## Outputs

Training writes to `results/em_<variant>_<mode>_{ckpt_latest.pt,
loss_curve.png, summary.md, training_log.jsonl}`.  `run_attractor_pipeline.sh`
invokes the standard `attractor_analysis/` scripts on every checkpoint
and writes per-variant attractor summaries and 3D landscape figures
into `attractor_analysis/results/`.  `compare.py` collates everything
into `results/comparison_report.md` and `results/comparison_table.json`.
