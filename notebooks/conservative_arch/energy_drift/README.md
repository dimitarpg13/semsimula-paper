# `energy_drift/` -- Energy-drift diagnostic for SPLM integrators

Eval-only diagnostic that computes the SPLM Hamiltonian energy

$$H_\ell \;=\; \tfrac{1}{2}\,\mathfrak m\,\|v_\ell\|^{2} \;+\; V_\theta(\xi_\ell, h_\ell)$$

at every layer $\ell$ of an SPLM forward pass and reports

- the **linear drift slope** $\partial H/\partial \ell$ across depth, and
- the **oscillation bandwidth** $\max_\ell H_\ell - \min_\ell H_\ell$ around its
  layer-mean.

This is **E3** of the SPLM-strengthening programme captured in
[`docs/Next_Model_Experiments_for_SPLM.md`](../../../docs/Next_Model_Experiments_for_SPLM.md)
section C2 ("energy-drift diagnostic"). It opens a new architecture-discriminating
axis: a Hamiltonian flow should show *bounded oscillation*, an Euler-style
explicit integrator should show a *systematic drift*, and an architecture with no
underlying potential (a transformer) should show *no structure*.

## Production E3 result (April 27, 2026)

The headline E3 production run is in
[`results/E3_splm_em_ln_compare/`](results/E3_splm_em_ln_compare/) and compares
three production-best SPLM checkpoints on the same e-init test corpus:

| variant | $L$ | mean $H$ | drift / layer | 95% CI | bandwidth | bandwidth / $|\mathrm{mean}\,H|$ |
|---|---:|---:|---:|---:|---:|---:|
| `parent_euler_L8` | 8 | -76.6 | -34.5 | ±1.7 | 145.7 | **190 %** |
| `verlet_L16_dt05` | 16 | -205.5 | -38.4 | ±0.6 | 91.4 | **45 %** |
| `em_ln_L8_seed0` (88.63 ppl) | 8 | +10.0 | -10.7 | ±0.5 | **7.0** | **70 %** |

The qualitative §15 prediction — *Verlet is energetically clean, Euler is not* —
is confirmed quantitatively: Verlet's bandwidth-to-scale ratio (45 %) is
4× tighter than parent Euler's (190 %). The substantive new finding is that
**`em_ln` uses the Euler integrator internally yet exhibits a Verlet-like
energy-conservation signature** (bandwidth-to-scale 70 %, 2.7× tighter than
bare Euler). The mechanism is the LayerNorm-after-step projection
$h_{l+1} \leftarrow \mathrm{LN}(h_l + \Delta t\,v_{l+1})$, which clips the
trajectory's dynamic range without contributing any potential gradient; the
production-best SPLM is consequently *not* a clean Hamiltonian flow but a
"cheating" symplectic integrator whose stability comes from compactification of
the state space rather than from symplectic structure of the integrator.
See [`results/E3_splm_em_ln_compare/E3_splm_em_ln_compare_report.md`](results/E3_splm_em_ln_compare/E3_splm_em_ln_compare_report.md)
for the full analysis with caveats.

### Why `sarfmass logfreq` is not in the production E3 column

The original E3 plan was a 3-way comparison `parent_euler` × `sarfmass logfreq`
× `verlet`. The multi-seed E1 sweep
([`notebooks/conservative_arch/multi_seed/`](../multi_seed/)) subsequently
falsified `sarfmass logfreq`'s stability — 2 of 3 of its seeds NaN-diverged on
the same training schedule on which `em_ln` succeeds 5/5. Running E3 on a
single `sarfmass logfreq` checkpoint would therefore measure energy
conservation on a model that does not exist as a stable family. The
production E3 column was reassigned from `sarfmass logfreq` to `em_ln`
(the same SARF-mass core plus the LayerNorm-after-step projection), which is
the SPLM that actually generalises across seeds. `sarfmass logfreq` remains
supported by the extraction script as a `--variant sarfmass` option for any
future single-seed diagnostic, but is intentionally absent from the
production comparison.

## Why this exists

The SPLM forward pass is a numerical solver for a damped Lagrangian. Whether
the solver actually preserves energy (modulo damping) depends on integrator
choice:

- **Velocity-Verlet ($L=16,\,\Delta t=0.5$)** is symplectic at $\gamma = 0$ and
  $O(\Delta t^4)$-bounded in energy at finite damping. We expect $H_\ell$ to
  oscillate around an exponentially-damped envelope.
- **Euler ($L=8$)** is a first-order explicit integrator. We expect a
  systematic drift in $H_\ell$ that grows linearly with depth.
- **GPT-2** is not derived from any potential. There is no $V_\theta$ to plug
  in, and forward-difference velocities $v_\ell \approx h_{\ell+1} - h_\ell$
  are noisy. We use the fitted $V_\psi$ from `shared_potential_fit.py` as a
  proxy and report the result as a control: if a transformer's hidden-state
  flow can be re-described as a Hamiltonian, the energy-drift signature
  should look like one of the SPLM variants. We expect it not to.

## Files

| file | purpose |
|---|---|
| [`extract_energy_states.py`](extract_energy_states.py) | Re-runs the SPLM forward pass on the §1 e-init test corpus and saves $(h_\ell, v_\ell, V_\theta(\xi_\ell, h_\ell), \tfrac{1}{2}m\|v_\ell\|^2)$ at every layer, for one checkpoint at a time. Supports parent-SPLM (`--variant euler`), `sarf_mass_variant` (`--variant sarfmass`, Euler + per-token mass), `symplectic_variant` (`--variant symplectic`, velocity-Verlet), and `energetic_minima/model_ln.py` (`--variant em_ln`, Euler + per-token mass + LayerNorm-after-step projection — the production SPLM). |
| [`energy_drift_diagnostic.py`](energy_drift_diagnostic.py) | Loads one or more saved energy-state files, plots $H_\ell$, $\tfrac{1}{2}m\|v\|^2$, and $V_\theta$ overlaid across variants, fits a linear drift slope and reports oscillation bandwidth, writes a markdown report. |
| `results/` | Populated by runs. One `.npz` per checkpoint and one report per comparison. |

## How to reproduce

The production E3 comparison is `parent_euler_L8` × `verlet_L16_dt05` ×
`em_ln_L8_seed0` (the `sarfmass logfreq` no-LN variant is omitted on the
multi-seed-instability rationale documented above):

```bash
# 1. Extract energy states for the three production SPLM checkpoints.
python3 notebooks/conservative_arch/energy_drift/extract_energy_states.py \
    --variant euler \
    --ckpt notebooks/conservative_arch/results/splm_shakespeare_ckpt_latest.pt \
    --label splm_euler_L8 \
    --out_npz notebooks/conservative_arch/energy_drift/results/splm_euler_L8.npz

python3 notebooks/conservative_arch/energy_drift/extract_energy_states.py \
    --variant symplectic \
    --ckpt notebooks/conservative_arch/symplectic_variant/results/splm_sym_logfreq_shakespeare_L16_dt05_ckpt_latest.pt \
    --label splm_verlet_L16_dt05 \
    --logfreq notebooks/conservative_arch/sarf_mass_variant/results/logfreq_surprisal.npy \
    --out_npz notebooks/conservative_arch/energy_drift/results/splm_verlet_L16_dt05.npz

python3 notebooks/conservative_arch/energy_drift/extract_energy_states.py \
    --variant em_ln \
    --ckpt notebooks/conservative_arch/multi_seed/results/E1_shakespeare/splm_em_ln/seed_0/em_ln_shakespeare_ckpt_latest.pt \
    --label splm_em_ln_L8_seed0 \
    --logfreq notebooks/conservative_arch/sarf_mass_variant/results/logfreq_surprisal.npy \
    --out_npz notebooks/conservative_arch/energy_drift/results/splm_em_ln_L8_seed0.npz

# 2. Compare them.
python3 notebooks/conservative_arch/energy_drift/energy_drift_diagnostic.py \
    --inputs splm_euler_L8.npz,splm_verlet_L16_dt05.npz,splm_em_ln_L8_seed0.npz \
    --tag E3_splm_em_ln_compare
```

The full production run finishes in under three minutes on MPS
(parent Euler ≈ 20 s, Verlet ≈ 21 s, `em_ln` ≈ 17 s, diagnostic ≈ 3 s).

If you specifically want to run the diagnostic on the original
`sarfmass logfreq` no-LN checkpoint (e.g. for an ablation against the
production `em_ln` column), the syntax is

```bash
python3 notebooks/conservative_arch/energy_drift/extract_energy_states.py \
    --variant sarfmass \
    --ckpt notebooks/conservative_arch/sarf_mass_variant/results/splm_sarfmass_logfreq_shakespeare_ckpt_latest.pt \
    --label splm_sarfmass_L8 \
    --logfreq notebooks/conservative_arch/sarf_mass_variant/results/logfreq_surprisal.npy \
    --out_npz notebooks/conservative_arch/energy_drift/results/splm_sarfmass_L8.npz
```

— but note that any conclusion drawn from a single `sarfmass logfreq`
seed must be qualified by the E1 multi-seed instability finding (see the
"Why `sarfmass logfreq` is not in the production E3 column" section
above).

## Output layout

```
energy_drift/results/
├── splm_euler_L8.npz                        # per-checkpoint: (n_sent, L+1, T, ...) layered states
├── splm_verlet_L16_dt05.npz
├── splm_em_ln_L8_seed0.npz                  # production SPLM, multi-seed E1 best (seed 0)
├── E3_splm_em_ln_compare/                   # production E3 bundle
│   ├── E3_splm_em_ln_compare_report.md      # main deliverable, with per-variant table + interpretation
│   ├── E3_splm_em_ln_compare_H_overlay.png
│   ├── E3_splm_em_ln_compare_kinetic_overlay.png
│   ├── E3_splm_em_ln_compare_potential_overlay.png
│   └── E3_splm_em_ln_compare_drift_table.csv
```

The headline figure is `E3_splm_em_ln_compare_H_overlay.png`: total energy
$H_\ell$ vs layer $\ell$, one line per variant, error band over sentences.
The markdown report includes a per-variant table:

| variant | drift slope (per layer) | drift slope 95% CI | oscillation bandwidth |
|---|---|---|---|

## Non-goals at v0

- **No GPT-2 column at v0.** Computing $H_\ell$ for a transformer requires a
  fitted $V_\psi$ (e.g. from `shared_potential_fit.py`) plus finite-difference
  velocities. That is a separate piece of glue best done after the SPLM-internal
  comparison is established. The SPLM-internal comparison shipped with the
  E3 production run on 2026-04-27 (see *Production E3 result* above), so the
  GPT-2 column is the natural next extension. It is tracked as the unfinished
  third column of E3 in
  [`docs/Next_Model_Experiments_for_SPLM.md`](../../../docs/Next_Model_Experiments_for_SPLM.md)
  (search for "GPT-2 trajectories" in the E3 section), and is queued behind
  the higher-priority first-order ODE rejection test
  ([`docs/first_order_ODE_rejection_pre-registered_protocol.md`](../../../docs/first_order_ODE_rejection_pre-registered_protocol.md)).
- **No retraining.** This diagnostic is purely forward-pass on existing
  checkpoints. Adding a "train an SPLM with energy-conserving regulariser"
  experiment is out of scope for E3 and would be a separate item in the
  catalogue.
- **No autograd through $V_\theta$ at extraction time.** We evaluate
  $V_\theta(\xi_\ell, h_\ell)$ in `torch.no_grad()` mode after the
  integration step that produced it, since we already have $h_\ell$ and
  $\xi_\ell$. The integration loop itself still uses autograd internally for
  the force kick.

## Relationship to existing scripts

- [`shared_potential_fit.py`](../shared_potential_fit.py) -- depth-axis test:
  "is there a single scalar $V_\psi(h)$ whose gradient explains all layer
  updates?" Answers a different question (existence of *some* potential).
  E3 answers: "does the *learned* $V_\theta$ obey the conservation law it
  was designed to?".
- [`attractor_analysis/`](../attractor_analysis/) -- gradient descent on
  $V_\theta$ to find isolated minima. Complementary: that finds the *fixed
  points* of the flow; E3 measures whether the *flow itself* is energy-
  conserving.

## Cross-references

- **E1 multi-seed sweep**:
  [`../multi_seed/results/E1_shakespeare/E1_report.md`](../multi_seed/results/E1_shakespeare/E1_report.md)
  is the canonical narrative for the variance bar that motivated the
  reassignment of the production E3 column from `sarfmass logfreq` to
  `em_ln`. The per-model divergence-rate diagnostic
  ([`E1_shakespeare_divergence_diagnostic.md`](../multi_seed/results/E1_shakespeare/E1_shakespeare_divergence_diagnostic.md))
  is the falsifying evidence.
- **Production-result note in `E1_report.md`'s Action Items § 7**: closes
  the loop with the actual E3 production scope and the headline finding,
  pointing back to this folder's `results/E3_splm_em_ln_compare/`.
- **`docs/Next_Model_Experiments_for_SPLM.md` § C2**: catalogue entry for
  E3, with the protocol-level rationale and the complementarity to E1.
