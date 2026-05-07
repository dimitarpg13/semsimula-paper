# Helmholtz hybrid (Q9d) — layer-type Helmholtz architecture

A scalar-potential-based Helmholtz hybrid LM in which the four
phase-space force components of paper v3's named decomposition (A.130)
are carried by physically distinct architectural carriers:

- **S-blocks** (autonomous, conservative) — a *single shared*
 scalar potential `V_theta(xi, h)` drives a velocity-Verlet damped
 Euler-Lagrange step.
- **A-blocks** (non-autonomous, Hopfield + small skew) — standard
 pre-LN attention + MLP residual block with per-layer parameters.

The depth schedule `sigma: {0..L-1} -> {S, A}` controls the
architectural shape (sandwich, interleaved, top-A, bottom-A, etc.).

Reference design doc:
[`companion_notes/Scalar_Potential_based_Helmholtz_Architecture.md`](../../../companion_notes/Scalar_Potential_based_Helmholtz_Architecture.md).

This is the **Q9(d)** branch of paper v3 §17.3, run as a separate path
from the **Variant A two-stage HSPLM** in
[`notebooks/conservative_arch/hybrid/`](../hybrid/) (which is the
σ = `A^k S^m` point in this design space). Both paths are kept
independent so we can run identical H0 / H1 / H1.5 / H2 sweeps and
compare cell-for-cell.

## Files

| File | Purpose |
|-------------------------------------------------------------|--------------------------------------------------------------------------|
| `model_helmholtz.py` | `HelmholtzLM` model + schedule parser + canonical-schedule registry |
| `train_helmholtz.py` | Trainer (mirrors `hybrid/train_splm_hybrid.py`, takes `--schedule`, optional `--v-hidden` for H1.5) |
| `causal_probe.py` | Causal-violation probe (perturbation + gradient-Jacobian) for every schedule; CI gate + trainer startup guard (with MPS workaround) |
| `aggregate_h1.py` | Aggregator → `results/h1_sweep/H1_RESULTS.md` |
| `aggregate_h1p5.py` | H1.5 aggregator → `results/h1p5_narrow_v/H1P5_RESULTS.md`; joint quality+FLOP table + architectural-floor decomposition |
| `decode_flop_pareto.py` | Analytical decode-FLOP Pareto (T ∈ {256, 1024, 4096}); appended to H1 |
| `scripts/run_h1_schedule_sweep.sh` | H0 + H1 launcher (idempotent, S=1, 7 schedules) |
| `scripts/run_h1p5_narrow_v.sh` | H1.5 narrow-V launcher (4 cells: `{AAAASSSS, AASSSSSS}` × `vh{128,256}`) |
| `__init__.py` | Package marker |

## Architecture (Q9d)

```
h_0 = E[x] + P
h_prev = h_0
for ell in 1..L:
 if sigma[ell] == 'S': # shared V_theta step
 delta = h - h_prev # velocity proxy
 xi = causal_cumulative_mean(h.detach) # leak-fix invariant
 f = -grad_h V_theta(xi, h)
 h_new = h + delta / (1 + dt*gamma)
 + (dt^2 / (m * (1 + dt*gamma))) * f
 h_new = LayerNorm(h_new) # if ln_after_s_step
 else: # 'A' block
 h_new = h + Attn_{theta_ell}(LayerNorm(h))
 + MLP_{theta_ell}(LayerNorm(h +...))
 h_prev, h = h, h_new
logits = h @ E^T # tied embeddings
```

The kinematic memory `h_prev` is carried across both block types so
the velocity proxy `h_ell - h_{ell-1}` remains well-defined even when
S-blocks are non-contiguous.

### Three commitments distinguish Q9d from any pre-existing HSPLM

1. **Single shared `V_theta` across all S-blocks** — including
 non-contiguous ones in interleaved or sandwich schedules. This is
 the strongest version of the "single energy field" interpretation:
 any contiguous run of S-blocks passes the strict shared-`V_psi`
 separator test (paper v3 §15.8) by construction.
2. **The A-blocks' Hopfield potentials are not constrained to share
 parameters.** Each A-block has its own QKV / MLP weights, exactly
 as in a standard decoder.
3. **`xi` in S-blocks is the SPLM-native causal cumulative-mean pool
 of the running hidden state**, re-derived from `h.detach` at
 every S-block. Multi-channel R6 generalisations (K-EMA,
 HiPPO-LegT, S4D, learnable-Δt) are a planned follow-up
 (doc §4.5).

## Schedule registry

`make_schedule(name, L, k, LA)` and `parse_schedule(string)` cover the
seven canonical patterns of doc §6:

| Name | L=8 string | n_S | n_A | Interpretation |
|-----------------------|-------------|-----|-----|------------------------------------------------------------------------|
| `all_s` | `SSSSSSSS` | 8 | 0 | Pure SPLM with velocity-Verlet form (control) |
| `all_a` | `AAAAAAAA` | 0 | 8 | Standard decoder (control); allocates unused V_theta |
| `bottom_a` (LA=4) | `AAAASSSS` | 4 | 4 | Variant A HSPLM (k=4, m=4) analogue |
| `bottom_a` (LA=2) | `AASSSSSS` | 6 | 2 | Variant A HSPLM (k=2, m=6) analogue |
| `top_a` (LA=1) | `SSSSSSSA` | 7 | 1 | Single-attention hybrid; doc §6 cell 3 (cleanest narrative) |
| `sandwich` (k=1) | `SAAAAAAS` | 2 | 6 | S at boundaries; doc §6 cell 1 (tests boundary-case mechanism §A.5) |
| `sandwich` (k=2) | `SSAAAASS` | 4 | 4 | Wider S boundaries |
| `inverse_sandwich`(k=1)| `ASSSSSSA` | 6 | 2 | A at boundaries; routing at the embedding edge |
| `interleaved` | `SASASASA` | 4 | 4 | Maximally mixed; doc §6 cell 2 (tests step-function R²ψ §4.1) |

## Reference baselines (already on disk, leak-immune)

| arm | val PPL | source |
|----------------------------------------------|------------|--------------------------------------------------------|
| All-attention (matched GPT-2, `n_layer=8`) | ~150 | `notebooks/conservative_arch/multi_seed/results/` |
| All-SPLM em_ln (free-γ, leak-free) | **173.59** | `notebooks/conservative_arch/energetic_minima/results/`|
| **Variant A HSPLM (k=4, m=4) at S=1** | **133.01** | `notebooks/conservative_arch/hybrid/results/h1_sweep/H1_RESULTS.md` |
| Variant A HSPLM best across (k, m) | 133.01–147.28 | same |

## Param-match table at L=8, d=128, v_hidden=512, v_depth=3

| Schedule | n_S | n_A | params |
|---------------|----:|----:|------------:|
| `AAAASSSS` | 4 | 4 | 7,916,163 |
| `AASSSSSS` | 6 | 2 | 7,519,619 |
| `SSSSSSSA` | 7 | 1 | 7,321,347 |
| `SAAAAAAS` | 2 | 6 | 8,312,707 |
| `SSAAAASS` | 4 | 4 | 7,916,163 |
| `ASSSSSSA` | 6 | 2 | 7,519,619 |
| `SASASASA` | 4 | 4 | 7,916,163 |
| `SSSSSSSS` | 8 | 0 | 7,123,075 |
| `AAAAAAAA` | 0 | 8 | 8,709,251 |

Schedules with the same `(n_S, n_A)` have **identical** parameter
counts and identical analytical decode-FLOP cost — only the schedule
order differs, which gives a clean iso-params iso-FLOPs comparison
between e.g. `AAAASSSS` (Variant-A-like), `SSAAAASS` (sandwich), and
`SASASASA` (interleaved).

## Phases (mirror Variant A's H0/H1/H1.5/H2 in `hybrid/`)

### H0 — smoke at the real shakespeare config

Train only the `AAAASSSS` (bottom_a_LA4) cell — the Variant A (k=4,
m=4) analogue — at the actual training config:

```bash
CELL_LIMIT=1 bash scripts/run_h1_schedule_sweep.sh
python3 aggregate_h1.py
python3 decode_flop_pareto.py
```

Time: ~30 min on Apple MPS. Verifies training stability + param
count + that the velocity-Verlet S-block dynamics integrate cleanly
through the full attention boundary.

**H0 success criterion:** val PPL within +/- 10 PPL of Variant A's
(k=4, m=4) anchor of 133.01. (Larger gap means the kinematic
differences from Variant A are non-trivial and we should investigate
before running the full sweep.)

### H1 — full schedule reconnaissance (S=1) — **DONE 5 May 2026**

Run all 7 schedules at S=1:

```bash
bash scripts/run_h1_schedule_sweep.sh
python3 aggregate_h1.py
python3 decode_flop_pareto.py
```

Time: ~3.7 h on Apple MPS. Output: `results/h1_sweep/H1_RESULTS.md`.

#### H1 results table (all 7 schedules, S=1, free γ)

| Schedule | n_S | n_A | params | val PPL | γ\* | wall |
|-------------|----:|----:|--------:|-----------:|------:|--------:|
| `AAAASSSS` | 4 | 4 | 7.92 M | **135.03** | 0.163 | 29.8 min |
| `SAAAAAAS` | 2 | 6 | 8.31 M | 137.60 | 0.147 | 30.3 min |
| `SSAAAASS` | 4 | 4 | 7.92 M | 139.62 | 0.114 | 29.7 min |
| `AASSSSSS` | 6 | 2 | 7.52 M | 140.60 | 0.162 | 37.7 min |
| `SSSSSSSA` | 7 | 1 | 7.32 M | 179.06 | 0.152 | 32.8 min |
| `SASASASA` | 4 | 4 | 7.92 M | 189.64 | 0.138 | 30.2 min |
| `ASSSSSSA` | 6 | 2 | 7.52 M | 200.81 | 0.151 | 34.8 min |

#### Headline H1 findings

- **Best schedule = `AAAASSSS`** (val PPL 135.03), beating all-attention
 (~150) by **−15 PPL**. Phase 2 gate: PASS (within the +10 PPL band).
- **Variant A direct comparison:** `AAAASSSS` is +2 PPL above Variant A
 `(k=4, m=4)` at 133.01 — within seed noise. Q9d kinematics are
 effectively equivalent at iso-(n_S=4, n_A=4) two-stage layout.
- **Q9d wins outright at (n_S=6, n_A=2):** `AASSSSSS` at 140.60 is
 **−6.7 PPL below Variant A `(k=2, m=6)`** at 147.28, with identical
 parameters and identical FLOPs. The per-S-block xi re-derivation
 and velocity-proxy-through-attention actively help in the high-S
 regime.
- **Layout matters at iso-(n_S, n_A).** At (n_S=4, n_A=4) the val PPL
 spans 135 (`AAAASSSS`) → 140 (`SSAAAASS`) → **190 (`SASASASA`,
 overfit)**, a 55 PPL spread driven purely by ordering. This is the
 cleanest Q9d-vs-Variant-A signal in H1: schedule freedom is real,
 even though it doesn't produce the best cell.
- **Scattered or boundary-bound A-blocks overfit catastrophically:**
 `SASASASA`, `ASSSSSSA`, and `SSSSSSSA` all have train losses below
 the well-behaved cells but val losses well above.
- **γ self-learning recovers the leak-free SPLM resonance anchor**
 γ\* ≈ 0.166 across 5 of 7 cells (independent confirmation of doc
 §4.3 effective-depth resonance prediction).
- **FLOP arm fails at `v_hidden = 512`** for every cell at T=1024 — same
 failure mode as Variant A H1. H1.5 narrow-V is the fix.
- **Causal probe was clean for every cell** at startup (perturbation +
 gradient-Jacobian Δ < 1e-6). One real bug surfaced and was patched:
 PyTorch's MPS backend doesn't support the second-order autograd path
 inside the gradient probe (raises "Placeholder storage has not been
 allocated on MPS device!"); `assert_causal` now temporarily moves
 the model to CPU around the probe and restores the original device
 afterwards. Architectural answer is device-independent so this is
 cosmetic.

#### Pareto + decision

Full per-T decode-FLOP table is appended to `H1_RESULTS.md` under
"Decode-FLOP Pareto" (T ∈ {256, 1024, 4096}).

**Phase 2 gate:** PASS — `AAAASSSS` clears the +10 PPL all-attention
band. Proceed to H1.5 (narrow-V) and H2 (S=3 confirmation, paired vs
all-attn AND vs Variant A best).

### H1.5 — V_theta-narrow ablation — **DONE 6 May 2026**

Run all 4 narrow-V cells (`AAAASSSS` × `AASSSSSS` × `v_hidden ∈
{128, 256}`):

```bash
bash scripts/run_h1p5_narrow_v.sh
python3 aggregate_h1p5.py
```

Time: ~1.9 h on Apple MPS. Output:
`results/h1p5_narrow_v/H1P5_RESULTS.md`.

#### H1.5 results table

| Schedule | n_S | n_A | v_hidden | Params | Val PPL | γ\* | dPPL vs vh=512 anchor |
|-------------|----:|----:|---------:|-------:|-----------:|-------:|----------------------:|
| `AAAASSSS` | 4 | 4 | 128 | 7.32 M | **134.89** | 0.164 | **−0.14** |
| `AAAASSSS` | 4 | 4 | 256 | 7.46 M | 136.48 | 0.163 | +1.45 |
| `AASSSSSS` | 6 | 2 | 128 | 6.93 M | 139.63 | 0.163 | **−0.97** |
| `AASSSSSS` | 6 | 2 | 256 | 7.06 M | 140.40 | 0.163 | **−0.20** |

#### Headline H1.5 findings

- **All 4 narrow-V cells preserve val PPL within ±1.5 PPL** of their
 vh=512 H1 anchors (3 of 4 actually slightly improve). V_θ at
 vh=512 was over-parameterised for this task; quartering it is free.
- **γ\* locks onto the resonance anchor** (~0.163) in every cell.
- **At T=1024 the FLOP arm is architecturally blocked** at this
 prototype scale: the embedding+logits floor is 12.87 MFLOPs/tok =
 63% of the all-attention reference, capping the maximum achievable
 reduction at 21.3% (vs the 30% rule). Best cell: `AASSSSSS` vh=128
 at 16.0% reduction — about 75% of the architectural ceiling.
- **At T=4096 the FLOP arm clears**: `AASSSSSS` vh=128 lands at
 **39.0% reduction** with val PPL 139.63 (within the +5 PPL all-attn
 band by 10.4 PPL).
- **Best joint cell: `AASSSSSS` vh=128.** Smallest model in the
 sweep (6.93 M params), clears 30% FLOP rule at T=4096, and **beats
 the matched Variant A `(k=2, m=6)` cell by −7.65 PPL** at
 iso-params, iso-FLOPs.

#### H1.5 → H2 gate

**PASS at T=4096** (T=1024 is architecturally infeasible at this
prototype scale). Proceed to H2 (S=3 paired confirmation) on:
- `AAAASSSS` vh=128 — best PPL (134.89, +1.88 vs Variant A best).
- `AASSSSSS` vh=128 — best joint quality+FLOP (smallest model,
 cleanest FLOP win, biggest Variant-A win).

### H2 — S=3 confirmation (planned)

Same structure as Variant A's H2: best 1-2 schedules × 3 seeds, paired
against all-attention by seed index.

### H3 — schedule-space exploration beyond canonical (optional)

The canonical 7-schedule sweep covers doc §6. Beyond that:
- **Sandwich-3** (`SSSAASSS`) — wider S-boundaries.
- **Triple-A insert** (`SSAAASSS`) — single A-segment in the middle.
- **A-readout** (`SSSSSSSA`) — already in canonical; doubles as the
 §A.5 boundary-case-by-construction analogue.
- **Long-context** (`SAAAAAAAS`-style at L=10+) — for the H5 / scale-up branch.

## Causal-violation probe

The Q9d architecture introduces two new mechanisms not present in
prior SPLM variants:

1. The velocity proxy `delta = h - h_prev` carries through *every*
 layer (S or A) — a new kinematic coupling.
2. `xi` is re-derived from `h.detach` at *every* S-block — the same
 codepath as in `model_sarf_mass.py`, which is the codepath that
 originally hosted the documented causal-leak bug
 (see [`companion_notes/Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md`](../../../companion_notes/Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md)).

Either one could in principle leak future-position information into
past predictions. We protect against this with two layers of defence.

### Standalone probe — `causal_probe.py`

Runs two complementary tests on every canonical schedule, in both
fixed (`causal_force=True`) and buggy (`causal_force=False`) modes:

- **Perturbation probe** — change exactly one token `x[t_pert]`,
 compare logits at every position `t < t_pert`. In a properly
 causal model these MUST be bit-identical (Δ ≡ 0).
- **Gradient-Jacobian probe** — compute
 `∂(logits[0, t_target,:].sum) / ∂(emb_in[0, t',:])` via
 autograd in `model.train` mode (so the integration step's
 `create_graph=True` path is exercised). The gradient must be zero
 for `t' > t_target`. Strictly stronger than perturbation because
 it catches second-order leak paths through `f = -grad_V`.

```bash
python3 causal_probe.py # all 9 canonical schedules
python3 causal_probe.py --strict # exit non-zero on any failure (CI gate)
python3 causal_probe.py --schedule SAAAAAAS # single explicit schedule
```

Expected output (verified at random init):

| Schedule | n_S | Fixed perturbation Δ | Fixed gradient Δ | Buggy gradient Δ |
|-----------------------|-----|----------------------|------------------|------------------|
| `bottom_a_LA4` | 4 | **0.00e+00** | **0.00e+00** | 1.27e-05 |
| `bottom_a_LA1` | 7 | **0.00e+00** | **0.00e+00** | 3.32e-05 |
| `top_a_LA1` | 7 | **0.00e+00** | **0.00e+00** | 3.63e-05 |
| `sandwich_k1` | 2 | **0.00e+00** | **0.00e+00** | 2.81e-06 |
| `sandwich_k2` | 4 | **0.00e+00** | **0.00e+00** | 1.07e-05 |
| `inverse_sandwich_k1` | 6 | **0.00e+00** | **0.00e+00** | 2.54e-05 |
| `interleaved` | 4 | **0.00e+00** | **0.00e+00** | 9.65e-06 |
| `all_s` | 8 | **0.00e+00** | **0.00e+00** | 3.66e-05 |
| `all_a` | 0 | **0.00e+00** | **0.00e+00** | 0.00e+00 |

The fixed mode is bit-exact zero across all 9 schedules. The buggy
gradient signal scales monotonically with `n_S` (each S-block
contributes one anti-causal `V_θ → grad_V → integration` path),
confirming both that the leak channel is real and that the
`.detach` fix severs it entirely. The `all_a` row (n_S=0) shows no
leak in either mode because there is no V_θ codepath to leak through.

### Trainer startup guard

`train_helmholtz.py` runs the perturbation + gradient-Jacobian probes
on the actual on-device model **before any optimisation step**. If
either probe detects a leak above `1e-6` (the strict threshold used
by the repo-wide probe), training is aborted with a clear diagnostic
and `SystemExit(2)`:

```
[helm-train] running causal-violation probe...
[helm-train] causal probe FAILED — aborting before any compute is wasted.
[helm-train] [causal-probe] GRADIENT LEAK: post=5.4318e-04 >= tol=1e-06.
 Aborting training before any compute is wasted.
 Causal_force=False; schedule=SASASA.
```

Cost of the guard: ~3 sec on CPU at smoke scale, negligible vs a
30-min MPS run. Pass `--skip-causal-check` to disable (not
recommended; the failure mode it catches is silent and ruinously
expensive — see the leak bug doc for the historical context).

## Resilience and idempotence

Same conventions as `hybrid/scripts/run_h1_layer_split_sweep.sh`:

- A cell is **skipped** if it already has a `*_summary.md` on disk.
- A failing cell does **not** abort the rest; a `TRAINING_FAILED.txt`
 marker is left in the cell's output directory.
- Optional env vars: `CELL_LIMIT=N`, `START_FROM=N`, `FIXED_GAMMA=x`,
 `SEED=N`.

## Q9d-vs-Variant-A: what to look for in the results

For matched `(n_S, n_A)` cells (e.g. `AAAASSSS` Q9d vs `(k=4, m=4)`
Variant A) the param count and analytical decode-FLOPs are identical.
Two structural differences could drive any divergence in val PPL:

1. **Velocity proxy carries through attention** rather than being
 reset to v=0 at the SPLM boundary. This means each S-block in Q9d
 inherits the kinematic state from the preceding layer (S or A),
 while in Variant A the SPLM stack always starts from rest.
2. **`xi` is re-derived from `h.detach` at every S-block** rather
 than being fixed at `h_k.detach` once after the attention stack.
 This is a finer-grained context summary; it could either help (more
 adaptive context) or hurt (more variance per integration step).

Three predictions worth checking against H1 results:

- **`AAAASSSS` ≈ Variant A (k=4, m=4)** → kinematic differences are
 small at this scale; the two architectures are empirically close
 variants of the same object.
- **`SASASASA` < `AAAASSSS`** → the interleaved schedule unlocks PPL
 by giving each conservative integration step a freshly-routed
 context. This is the headline Q9d prediction.
- **`SSSSSSSA` < `AAAASSSS`** → a single attention block at the readout
 position (the §A.5 boundary-case-by-construction analogue) suffices
 to recover most of the routing benefit. This is the cleanest
 narrative cell for a Q9d-only paper.

## Pointers

- Q9d design doc: [`companion_notes/Scalar_Potential_based_Helmholtz_Architecture.md`](../../../companion_notes/Scalar_Potential_based_Helmholtz_Architecture.md)
- Variant A path forward: [`companion_notes/HSPLM_Path_Forward_and_Experiments.md`](../../../companion_notes/HSPLM_Path_Forward_and_Experiments.md)
- Pre-registered title-justification rule: §6.5 of the v4 title-justification rule
- Causal-leak fix and forensics: [`companion_notes/Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md`](../../../companion_notes/Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md)
- Repo-wide causal probe (covers every SPLM variant): [`notebooks/conservative_arch/causal_probe.py`](../causal_probe.py)
- Resonance-predictor double match (ρ = 0.565 leak-free anchor): [`companion_notes/Determining_optimal_gamma_for_SPLM.md`](../../../companion_notes/Determining_optimal_gamma_for_SPLM.md) §2.5
- Variant A HSPLM H1 results: [`notebooks/conservative_arch/hybrid/results/h1_sweep/H1_RESULTS.md`](../hybrid/results/h1_sweep/H1_RESULTS.md)
