# `sarf_mass_variant/` — SARF-faithful SPLM with per-token semantic mass

One-knob ablation stacked on top of the SARF-faithful SPLM
([`../sarf_variant/`](../sarf_variant/)): replace the single learnable
global scalar $m$ with a per-token semantic mass $\semm_t$, preserving
every other design choice (SARF-faithful $\xi$ re-pooling per layer,
shared $V_\theta(\xi, h)$, damped Euler-Lagrange integrator, tied
embedding readout, identical hyperparameters).

The question this folder answers:

> **Does the Semantic Simulation framework's per-token semantic mass
> prescription carry empirical content on top of SARF-faithful
> $\xi$ re-pooling?  And if so, is the data-driven "learn mass from the
> embedding" flavour better, worse, or indistinguishable from the
> theoretically-motivated "Shannon-surprisal prior" flavour?**

## The two mass variants (and the null control)

Mass is computed once per forward pass from the first-layer input and
held fixed across all $L$ integration steps.  This matches the
framework's view that $\semm_t$ is a **property of the semantic
particle** (token identity), not a state- or layer-dependent quantity.

| `mass_mode`  | definition | extra params | theoretical hypothesis |
|---|---|---:|---|
| `global`     | $\semm_t = \mathrm{softplus}(b) + \varepsilon$, single scalar | 1 | null control: no per-token variation |
| `embed_head` **(A)** | $\semm_t = \mathrm{softplus}(\langle w_m, E_{x_t}\rangle + b_m) + \varepsilon$ | $d + 1 = 129$ | mass is a learned linear function of the token embedding |
| `logfreq` **(B)**   | $\semm_t = \mathrm{softplus}(b_m + \alpha s(x_t)) + \varepsilon$, $s(v) = -\log \hat p(v)$ frozen | 1 | mass = Shannon surprisal, one scale knob |

### Variant (A) `embed_head` — learned content-driven mass

```python
self.mass_head = nn.Linear(d, 1, bias=True)   # zero-init
m_t = softplus(mass_head(E[x_t]) + raw_m_bias) + 1e-3
```

- **Init:** head weights and bias zero-initialised, so $\semm_t \equiv
  \mathrm{softplus}(\text{raw\_m\_bias}) + \varepsilon \approx 1.0$
  identical to the SARF baseline at step 0.  Any variation that
  emerges is learned from data.
- **Cost:** one $O(Td)$ linear head on the first layer only.  Mass
  is cached across the $L$ integration steps.
- **What we should see if the framework's prescription has content:**
  non-trivial mass dispersion at convergence (std $\gg 0$), with
  content words receiving different mass than function words.

### Variant (B) `logfreq` — Shannon-surprisal prior

```python
s = precomputed_surprisal[x_t]                # frozen lookup
alpha = softplus(raw_logfreq_alpha)
m_t = softplus(raw_m_bias + alpha * s) + 1e-3
```

- **Init:** $\alpha$ initialised to $\approx 0.1$, `raw_m_bias`
  initialised so $\semm_t \approx 1$ on average.
- **Cost:** one `gather` per forward pass, essentially free.
- **What we should see:** $\alpha$ either pulled toward zero (prior
  has no content — equivalent to the null control) or settling to a
  non-trivial positive value (prior has content — rarer tokens are
  heavier).
- The surprisal table is precomputed from the Tiny Shakespeare **train
  split** with add-one Laplace smoothing and saved in
  `results/logfreq_surprisal.npy` (unigram of GPT-2 BPE ids; min
  surprisal $\sim 2.3$ nats, max $\sim 12.8$ nats).

## Directory layout

| file | purpose |
|---|---|
| [`model_sarf_mass.py`](model_sarf_mass.py) | `ScalarPotentialLMSARFMass` with `mass_mode` switch; self-test |
| [`compute_unigram_frequencies.py`](compute_unigram_frequencies.py) | one-off: surprisal lookup table from Tiny Shakespeare |
| [`train_splm_sarf_mass.py`](train_splm_sarf_mass.py) | trainer with `--mass-mode` flag |
| [`trajectory_extraction_sarf_mass.py`](trajectory_extraction_sarf_mass.py) | trajectory extractor (reuses parent E-init corpus) |
| [`compare.py`](compare.py) | 4-way comparison: fixed-$\xi$ SPLM / SARF / SARF+A / SARF+B |
| [`comparison_report.md`](comparison_report.md) | auto-generated markdown report |
| `results/` | checkpoints, trajectories, loss curves, diagnostics |

## How to reproduce (full pipeline)

```bash
# From repository root:
cd notebooks/conservative_arch/sarf_mass_variant

# 0. Precompute the surprisal lookup table (once, ~1 second).
PYTHONUNBUFFERED=1 python3 -u compute_unigram_frequencies.py

# 1. Train variant A (embed_head) -- ~22 min on MPS.
PYTHONUNBUFFERED=1 python3 -u train_splm_sarf_mass.py \
    --mode shakespeare --mass-mode embed_head

# 2. Train variant B (logfreq) -- ~22 min on MPS.
PYTHONUNBUFFERED=1 python3 -u train_splm_sarf_mass.py \
    --mode shakespeare --mass-mode logfreq

# 3. Extract trajectories from each checkpoint.
PYTHONUNBUFFERED=1 python3 -u trajectory_extraction_sarf_mass.py \
    --ckpt results/splm_sarfmass_embed_head_shakespeare_ckpt_latest.pt
PYTHONUNBUFFERED=1 python3 -u trajectory_extraction_sarf_mass.py \
    --ckpt results/splm_sarfmass_logfreq_shakespeare_ckpt_latest.pt

# 4. Run the paper's §14.8 shared-potential fit on each variant (parent script).
cd .. && PYTHONUNBUFFERED=1 python3 -u shared_potential_fit.py \
    --traj sarf_mass_variant/results/splm_sarfmass_embed_head_shakespeare_ckpt_latest.trajectories.pkl \
    --tag sarfmass_embed_head_shakespeare
PYTHONUNBUFFERED=1 python3 -u shared_potential_fit.py \
    --traj sarf_mass_variant/results/splm_sarfmass_logfreq_shakespeare_ckpt_latest.trajectories.pkl \
    --tag sarfmass_logfreq_shakespeare

# 5. Run the paper's §14.15 token-direction fit on each variant.
PYTHONUNBUFFERED=1 python3 -u token_direction_fit.py \
    --traj sarf_mass_variant/results/splm_sarfmass_embed_head_shakespeare_ckpt_latest.trajectories.pkl \
    --tag sarfmass_embed_head_shakespeare
PYTHONUNBUFFERED=1 python3 -u token_direction_fit.py \
    --traj sarf_mass_variant/results/splm_sarfmass_logfreq_shakespeare_ckpt_latest.trajectories.pkl \
    --tag sarfmass_logfreq_shakespeare

# 6. Side-by-side 4-way comparison.
cd sarf_mass_variant && python3 compare.py
```

The 4-way comparison lines up:
- **fixed-$\xi$ SPLM** (baseline, from `../results/splm_shakespeare_*`)
- **SARF-faithful SPLM** (from `../sarf_variant/results/splm_sarf_shakespeare_*`)
- **SARF + embed-head mass** (from this folder, variant A)
- **SARF + logfreq mass** (from this folder, variant B)

All four share identical hyperparameters, parameter counts (within
$d+1 = 129$), data, and seed.  Any difference in val perplexity,
shared-$V_\psi$ depth $R^2$, or token-direction $R^2$ is attributable
purely to **the mass parameterisation**.

## What the results will tell us

- **If both (A) and (B) beat SARF on LM quality** — the framework's
  per-token mass prescription has empirical content on top of the
  SARF-faithful $\xi$ prescription, and the two axes compound.
- **If only (A) beats SARF** — data-driven learned mass is useful but
  the information-content prior is not the right shape; mass encodes
  something beyond surprisal.
- **If only (B) beats SARF** — the framework's information-theoretic
  interpretation of mass is empirically productive, and the linear
  head is underfit / over-regularised away.
- **If neither beats SARF** — with a single global scalar $m$ already
  sufficient in the SARF regime, per-token variation is marginal or
  detrimental.  The framework's per-token mass would then be a
  theoretical decoration without experimental bite *on this corpus*
  (always qualified by scale; Tiny Shakespeare is tiny).

## Relationship to the paper

Directly addresses Q10 in §16 of `paper_v2/sections/16_conclusion.tex`:
**Per-token semantic mass in SPLM** — this folder is the prescribed
experimental design for Q10, stacked on top of the SARF-faithful
ablation of §14.13.  The planned follow-up (see §16 closing paragraph)
is the three-way conjunction test
$\{\text{fixed-}\xi \vee \text{SARF-}\xi\} \times \{\text{global }m,\ \text{per-token }m\}$,
of which this folder ships the two most informative cells.
