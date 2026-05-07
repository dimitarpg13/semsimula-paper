# PARF-augmented SPLM (Q9c) — prototype

**Path companion to** [`notebooks/conservative_arch/helmholtz/`](../helmholtz/)
**(Q9d, the layer-type Helmholtz hybrid).**

Design reference: [`companion_notes/PARF_Augmented_SPLM_Architecture.md`](../../../companion_notes/PARF_Augmented_SPLM_Architecture.md).
Workplan & decision log: this README + the design doc; the
`Helmholtz-HSPLM_Path_Forward_and_Experiments.md` document is shared
infrastructure but does NOT govern the PARF path.

## What this is

A clean prototype for the PARF-augmented SPLM (Q9c). Every layer is a
velocity-Verlet integrator under the SHARED effective scalar

$$U^{(\ell)}_t \;=\; V_\theta(\xi_t, h_t) \;+\; \sum_{s \lt t} V_\phi(h_t, h_s)$$

with the design-doc §3 **causal reduction**: past tokens are treated as
fixed external sources by `.detach`-ing the source slice
$\lbrace h_s \rbrace_{s \lt t}$ when forming the pair-potential matrix.
This both severs the back-reaction force on past tokens and makes the
per-token force strictly causal.

Two `V_phi` variants ship in this prototype:

| name | shape | role |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| `structural` | $-C \cdot \Theta_\phi(\theta(h_t), \theta(h_s)) \cdot \Phi_\phi(l(h_t), l(h_s)) / \sqrt{\lVert h_t-h_s\rVert^2 + \varepsilon^2}$ | **§5.1-faithful pair potential** (default; the framework prior) |
| `mlp` | learned MLP on $\mathrm{concat}(h_t,\ h_s,\ h_t-h_s)$ | **unstructured ablation** (the design-doc OQ-1 comparator) |

The two share the SAME outer machinery (single shared $V_\theta$,
velocity-Verlet step, causal reduction, logfreq mass, embed/logits
shape). The only difference is the inner shape of $V_\phi$. This
isolates the "does the §5.1 prior matter empirically?" question to
exactly one experimental knob.

## Files

```
parf/
├── __init__.py module marker
├── README.md this file
├── model_parf.py PARFConfig + StructuralVPhi + MLPVPhi + PARFLM
├── causal_probe_parf.py perturbation + gradient-Jacobian probes
├── smoke_test.py end-to-end smoke (forward+backward+probe+5-step train)
├── train_parf.py trainer (Algorithm A: pure NTP backprop)
├── scripts/
│ └── run_first_quality_cell.sh idempotent wrapper for the first quality cell
└── results/ per-cell training artifacts
```

## Wall-clock at the prototype shape (Apple MPS, 16 GB)

Measured on the H1.5 vh=128 cell shape (`d=128, L=8, T=128, B=16,
v_hidden=128`). Memory is the binding constraint for the MLP
variant; wall-clock is the binding constraint for the structural
variant.

| variant | grad ckpt | B | mlp_h | s/step | 4000-step est | notes |
| ------------ | --------- | -- | ----- | ------ | ------------- | ------------------------------------------------------------------ |
| structural | off | 16 | — | 3.77 | ~252 min | **headline cell** (used for `seed0/structural` quality datapoint) |
| structural | on | 16 | — | ~4.5† | ~300 min | not needed at this shape (memory fits without it) |
| mlp | off | 16 | 64 | OOM | OOM | `(B,T,T,3d)` input + autograd graph exceeds 13.5 GB MPS watermark |
| mlp | on | 16 | 64 | OOM | OOM | grad-ckpt cannot intercept the input tensor itself |
| mlp | on | 16 | 32 | 9.6 | ~640 min | fits, but slow |
| mlp | on | 16 | 16 | 6.4 | ~430 min | fits comfortably |
| mlp | off | 8 | 64 | 1.97 | ~131 min | fast, but B halved → not protocol-matched to anchors |
| mlp | on | 8 | 64 | 4.31 | ~287 min | grad-ckpt tax is ~115% on MLP V_φ (recomputes the dominant op) |

†extrapolated; the in-cell measurement was contended.

The grad-checkpoint flag (`PARFConfig.use_grad_checkpoint`,
`--grad-checkpoint`, env `GRAD_CHECKPOINT=1`) is **auto-enabled for
the MLP variant** by `scripts/run_first_quality_cell.sh`; the
structural variant runs without it by default. Bit-equality of the
training trace with and without grad-checkpoint is verified by
`smoke_test.py`'s `[3/3] em-ln+gc` block.

## Algorithm A — what's in, what's out

This prototype implements **Algorithm A** from the design doc
([`companion_notes/PARF_Augmented_SPLM_Architecture.md`](../../../companion_notes/PARF_Augmented_SPLM_Architecture.md)
§7):

- ✅ Single shared $V_\theta$ (4-layer GELU MLP, identical to em-ln
 leakfree SPLM and Q9d S-blocks).
- ✅ Single shared $V_\phi$ (structural OR mlp).
- ✅ Velocity-Verlet damped Euler-Lagrange step at every layer.
- ✅ Causal reduction via two `.detach` points
 ($\xi$-pool source AND pair source slice).
- ✅ Per-token mass, `logfreq` mode (matches Q9d / VA / em-ln-leakfree).
- ✅ Startup causal-violation probe (perturbation + gradient-Jacobian)
 that aborts training before any optimiser step on leak.
- ✅ NTP-only auxiliary-loss backprop through both potentials.
- ❌ Gumbel-softmax sparsity for $V_\phi$ (Stage 1.5 add-on; deferred).
- ❌ Iterative inner-loop schemes (Algorithm B / Iter; deferred).
- ❌ Curriculum / RL outer loop (PARF + RL paper draft; deferred).

The PARF pair sum at every layer is $O(T^2)$ (it walks the same
quadratic-in-context surface as attention), so wall-clock at $T=128$
is ~1.5× the Q9d $\mathtt{AAAASSSS}$ vh=128 cell on Apple MPS.

## Sanity checks (all green)

The prototype is wired correctly:

- **Causal probe** — both V_phi variants pass strict $\Delta < 10^{-6}$
 in fixed mode (`causal_force=True`) and visibly leak in buggy mode
 (`causal_force=False`):

 ```
 python3 notebooks/conservative_arch/parf/causal_probe_parf.py
...
 [ OK] V_phi= 'structural' (L=4) fixed pre=0.00e+00, grad post=0.00e+00
 [ OK] V_phi= 'mlp' (L=4) fixed pre=0.00e+00, grad post=0.00e+00
 All 2 V_phi variants: fixed-mode causal-side Δ < 1e-06. PARF is causal by construction.
 ```

- **End-to-end smoke** — both V_phi variants forward, backward, pass
 the production causal probe, and reduce loss over 5 NTP optimiser
 steps on tiny + em-ln-shape configs:

 ```
 python3 notebooks/conservative_arch/parf/smoke_test.py
...
 [em-ln] V_phi='structural' params=135,717 V_theta=66,049 V_phi=4,002
 [em-ln] causal probe PASS (V_phi='structural')
 [em-ln] V_phi='mlp' params=160,580 V_theta=66,049 V_phi=28,865
 [em-ln] causal probe PASS (V_phi='mlp')
 ALL SMOKE CHECKS PASSED
 ```

## Reproduce the first quality cell

The first cell places PARF on the SAME shape as the Q9d
$\mathtt{AAAASSSS}$ vh=128 cell so its val PPL is directly comparable
to the Q9d/VA/all-attn anchors:

| arm | val PPL (seed 0) | source |
| -------------------------------------------- | ------------------- | ----------------------------------------- |
| all-attention 5-seed E1 baseline (mean) | ~141.80 | `hybrid/results/h1_sweep/` |
| em-ln SPLM leakfree (1 seed) | ~150 | `sarf_mass_variant/results/...em_ln...` |
| Variant A k=4, m=4 seed 0 | 133.01 | `hybrid/results/h1_sweep/k4_m4/seed0/` |
| Helmholtz Q9d $\mathtt{AAAASSSS}$ vh=128 sd0 | 134.89 | `helmholtz/results/h1p5_narrow_v/` |
| **PARF Q9c structural seed 0** | TBD (run this cell) | `parf/results/structural/seed0/` |

```bash
# Default: structural V_phi, seed 0, gamma freely learned, MPS.
bash notebooks/conservative_arch/parf/scripts/run_first_quality_cell.sh

# Add the MLP ablation in the same launch (sequential):
V_PHI_KINDS="structural mlp" \
 bash notebooks/conservative_arch/parf/scripts/run_first_quality_cell.sh
```

The wrapper is idempotent: it skips any cell whose `summary.md` already
exists, so it is safe to re-launch after an interruption.

## Reading the headline number

Per the design-doc Open Questions:

- **OQ-1 (structural vs MLP)** — if `structural` matches `mlp` on val
 PPL, the §5.1 prior is pedagogical (the unstructured form has enough
 expressivity to recover the same dynamics). If `structural`
 outperforms, the §5.1 prior is empirically active and worth keeping
 in any subsequent paper write-up.

- **PARF vs the SPLM family** — the headline number to read is
 `PARF structural / Q9d AAAASSSS vh=128 / VA k=4 m=4`. PARF only
 earns its $O(T^2)$ pair-sum cost if it places at or below VA's 133
 PPL with smaller variance across seeds. If it places at parity or
 worse, the PARF pair force adds capacity at the same wall-clock as
 attention, which is not a clear win — at that point Stage 1.5
 (Gumbel sparsity → token sparsification) becomes the next gate.

## What this prototype does NOT do

- It is **not the paper headline**. No paper update is committed to
 the PARF arm yet (per user instruction; paper v4 is on hold pending
 H6, H2 n=5 power-up, and these PARF datapoints).
- It is **not the RL outer loop**. The PARF + RL paper draft
 (`companion_notes/Section_15_24_PARF_Augmented_SPLM_v4_draft.docx`) is
 upstream of this prototype. This codepath establishes the inner
 PARF dynamics and its quality baseline; the RL outer loop is a
 later commitment.

## Pointers

- Design doc: [`companion_notes/PARF_Augmented_SPLM_Architecture.md`](../../../companion_notes/PARF_Augmented_SPLM_Architecture.md)
- Companion path (Q9d, layer-type Helmholtz hybrid): [`notebooks/conservative_arch/helmholtz/`](../helmholtz/)
- Companion path (Variant A two-stage SPLM): [`notebooks/conservative_arch/hybrid/`](../hybrid/)
- Causality bug & fix history: [`companion_notes/Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md`](../../../companion_notes/Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md)
- Markdown LaTeX rendering rules: the markdown rendering cheatsheet
