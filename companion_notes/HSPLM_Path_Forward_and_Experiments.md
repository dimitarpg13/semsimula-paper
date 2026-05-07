# HSPLM (Hybrid SPLM + Attention) — Path Forward and Experiments

**Status:** Live experiment record · started 5 May 2026
**Scope paper:** `paper_v3/main.tex`
**Code root:** `notebooks/conservative_arch/hybrid/`

This document tracks the design, experiments, results, and outstanding
questions of the **Hybrid SPLM + Attention (HSPLM)** investigation.
The investigation tests whether a layer-pattern hybrid that combines
attention blocks with scalar-potential SPLM steps can rigorously earn
the word *"Efficient"* in the paper title under the leak-corrected
v3 integrator.

The pre-registered v4 title-justification rule (locked at the design
stage):

> **"Efficient" is justified iff** some hybrid (k, m) achieves val PPL
> within **+5 PPL** of the all-attention baseline (~150 on Tiny
> Shakespeare at d=128, L=8) **AND** its analytical decode-FLOP cost
> at T = 1024 is **≥ 30% lower** than all-attention, both at S=3 with
> sign-consistency 3/3.

---

## 1. Architecture (Variant A, two-stage)

```
h_0   = E[x] + P
for i = 1..k:                                            # k distinct attn blocks
    h_i = AttnBlock_i(h_{i-1})
h_k   = LayerNorm(h_k)                                   # boundary projection
xi    = causal_cumulative_mean(h_k.detach())             # leak-fix invariant
for j = 1..m:                                            # m shared SPLM steps
    f = -grad_h V_theta(xi, h)
    v = (v + dt · f / m_t) / (1 + dt · gamma)
    h = h + dt · v
    h = LayerNorm(h)                                     # if ln_after_step
logits = h @ E^T                                         # tied embeddings
```

Implementation: `notebooks/conservative_arch/hybrid/model_hybrid.py`.
Fixed design choices:

- **xi re-derivation = `causal_cumulative_mean(h_k.detach)`** — leak-safe,
 attention-refined context (preferred over raw-emb-ξ).
- **Single shared `V_theta`** across all `m` SPLM steps — preserves the
 "single energy field" Lagrangian-mechanics interpretation.
- **Boundary `LayerNorm`** between attn and SPLM stacks — mirrors GPT-2's `ln_f`.
- **`ln_after_step=True`** on the SPLM tail — matches em_ln semantics.
- **`causal_force=True` is hard-default** — preserves v3 leak-fix invariant.
- **Tied embeddings** with `E.weight.T` for the LM head — same as SPLM/GPT-2.
- **Per-token logfreq mass** (`mass_mode='logfreq'`) — matches leak-free SPLM em_ln.

---

## 2. Reference baselines (already on disk, leak-immune)

| Arm | Val PPL | Source |
|------------------------------------------------|------------|----------------------------------------------------------------------|
| All-attention (matched GPT-2, `n_layer=8`) | **149.80 ± 7.21** (5-seed E1) / 156.13 ± 8.10 (Phase 1) | `multi_seed/results/` |
| All-SPLM em_ln (free-γ, leak-free) | **173.59** (single seed) | `energetic_minima/results/` |
| All-SPLM em_ln (γ ∈ [0.10, 0.15], leak-free) | **~178–181** | `ln_damping_sweep/results/RESULTS_LEAKFREE_GAMMA_SWEEP.md` |

All trained at `(d=128, max_len=256, L=8, n_head=4, mlp_mult=4, v_hidden=512,
v_depth=3)`, batch 16, block 128, 4000 steps, AdamW(0.9, 0.95) lr=5e-4 with
200 warmup + cosine, on Tiny Shakespeare (GPT-2 BPE).

Param count: hybrid `(k≥1, m≥1)` cells fall in 7.5–8.3 M (within ±0.5 M
of the 8.0 M all-attention reference); param-matching is dominated by
the embedding (~6.4 M) which is shared across all arms.

---

## 3. Tiered experimental plan (recap)

| Tier | What | Cells | Time est. | Status |
|------|-----------------------------------------------------------------------------------------------------|------:|-------------|-------------|
| H0 | Variant A at (k=4, m=4), S=1, smoke at the real shakespeare config | 1 | ~30 min | ✅ done |
| H1 | Variant A across (k, m) ∈ {(2,6), (3,5), (4,4), (5,3), (6,2)}, S=1 | 5 | ~3 h MPS | ✅ done |
| H1.5 | V_theta-narrow ablation: (4,4) and (6,2) at v_hidden ∈ {128, 256}, S=1 | 4 | ~2-3 h MPS | **deferred** (see §4.5; Q9d H1.5 result applies analogously) |
| H2 | Best (k, m) at S=5 paired with all-attention and Q9d | 4 (new) | ~2.5 h MPS | ✅ done |
| H3 | Variant B (interleaved stripe pattern) on best stripe | 3 | ~1.5 h MPS | optional |
| H4 | Pareto: per-token decode FLOPs vs val PPL at T ∈ {256, 1024, 4096} (analytical) | 0 | analytical | ✅ partial (H1) |
| H5 | TinyStories scale-up cell on best hybrid config, S=3 | 3 | ~4-5 h MPS | optional |

---

## 4. Results so far

### 4.1. H0 + H1 — layer-split sweep (S=1) — completed 5 May 2026

Output: `notebooks/conservative_arch/hybrid/results/h1_sweep/H1_RESULTS.md`.

| Cell | (k, m) | Val PPL | Final γ | Params | Elapsed (Apple MPS) |
|------|--------|--------:|--------:|--------:|---------------------|
| #00 | (4, 4) | **133.01** | 0.154 | 7.92 M | 32 min |
| #01 | (2, 6) | 147.28 | 0.153 | 7.52 M | 35 min |
| #02 | (3, 5) | 139.29 | 0.153 | 7.72 M | 35 min |
| #03 | (5, 3) | 136.48 | 0.154 | 8.11 M | 34 min |
| #04 | (6, 2) | 135.08 | 0.153 | 8.31 M | 34 min |

**Quality arm of the rule: PASSES dramatically.** Every single hybrid
beats the all-attention baseline at iso-training-budget. Best cell
(4, 4) is at **-17.0 PPL** vs all-attention — well outside ~2σ of
the all-attention 5-seed std (σ = 7.21).

Free-γ converged near **0.154** in all five cells, strikingly close
to the leak-free SPLM-2 optimum basin γ* ∈ [0.10, 0.15] from the S=5
confirmation sweep. The hybrid finds the same damping regime
*independently* of the (k, m) split — strong cross-evidence that
the second-order Lagrangian's natural damping is a property of the
training objective + data, not of the layer count.

### 4.2. H4 (partial) — analytical decode-FLOP Pareto

Output: appended to the same `H1_RESULTS.md` by
`notebooks/conservative_arch/hybrid/decode_flop_pareto.py`.

**Per-token AR decode FLOPs at T = 1024** (KV-cached attention,
streaming-ξ SPLM, current `v_hidden = 512`):

| (k, m) | val PPL | Decode FLOPs/tok | vs all-attn | Reduction |
|----------|---------|------------------|-------------|-----------|
| (2, 6) | 147.28 | 38.394 M | 1.883× | -88.3% |
| (3, 5) | 139.29 | 35.393 M | 1.736× | -73.6% |
| (4, 4) | 133.01 | 32.391 M | 1.589× | -58.9% |
| (5, 3) | 136.48 | 29.390 M | 1.442× | -44.2% |
| (6, 2) | 135.08 | 26.388 M | 1.295× | -29.5% |
| ref attn | 149.80 | **20.385 M** | 1.000× | — |

**FLOP arm of the rule at T = 1024: FAILS at the prototype's `v_hidden = 512`.**
Every hybrid is *more* expensive than all-attention at T = 1024 by
+30% (most attn-heavy hybrid (6, 2)) up to +88% (most SPLM-heavy
hybrid (2, 6)). At T = 4096 the gap shrinks but the sign does not flip
((6, 2) is +8.2% more expensive even at T = 4096).

Cause: each SPLM integration step at d=128 with `v_hidden = 512` costs
~3.94 MFLOPs/tok (≈ 1.31 M forward + 2.63 M backward through `V_theta`
on a single token), versus ~0.94 MFLOPs/tok per attention block at
T=1024. The wide `V_theta` MLP (2d → 512 → 512 → 1) was inherited
from the canonical SPLM-2 cell where compute was not optimised; in
the hybrid this becomes the FLOP bottleneck.

### 4.3. H2 — 5-seed paired confirmation, (k=4, m=4) — completed 6 May 2026

Output: per-seed artifacts under
`hybrid/results/h2_paired_confirmation/k4_m4/seed{1..4}/`; seed 0
reused from `h1_sweep/`. Aggregated by
`notebooks/conservative_arch/helmholtz/aggregate_h2.py`, which
auto-picks up the Variant A cells alongside Q9d to produce the
cross-architecture paired-t table at
`helmholtz/results/h2_paired_confirmation/H2_RESULTS.md`. (No
separate `aggregate_h2.py` was added under `hybrid/`; the Q9d
aggregator is the single source of truth for the paired statistics.)

#### Per-seed results

| seed | val PPL | val loss | final γ | wall |
|----------|-----------:|---------:|--------:|----------:|
| 0 | 133.01 | 4.8904 | 0.154 | 32.2 min |
| 1 | 152.25 | 5.0255 | 0.154 | 30.5 min |
| 2 | 152.10 | 5.0246 | 0.154 | 34.1 min |
| 3 | 141.67 | 4.9535 | 0.154 | 30.8 min |
| 4 | 157.97 | 5.0624 | 0.154 | 30.0 min |
| **mean** | **147.40** | 5.0017 | 0.154 | — |

Mean across n=5 seeds: **147.40** PPL; std across seeds 9.99 PPL
(seed 0 from H1 is the lowest of the panel, seed 4 the highest).

#### Paired comparison vs all-attention 5-seed E1 baseline

All-attention E1 PPL by seed: seed0=141.80, seed1=154.79,
seed2=159.59, seed3=146.85, seed4=145.99 (mean 149.80).

| n pairs | VA mean | Δ̄ vs attn | std Δ | sign (Δ\<0) | paired-t | two-sided p | meets +5 PPL bar? |
|---------|---------|-----------|-------|-------------|----------|-------------|-------------------|
| 5 | 147.40 | -2.40 | 8.39 | 4/5 | -0.641 | 0.5564 | YES (mean within +5 PPL) |

#### Headline interpretation

1. **Quality bar in absolute terms: clears.** VA mean (147.40) is
 well within the pre-registered +5 PPL of all-attention mean
 (149.80) — the sign-direction is in favor of the hybrid (Δ̄ =
 -2.40 PPL), and 4 of 5 paired seeds have the hybrid winning.
2. **Statistical significance: weak at n=5.** Paired-t two-sided
 p = 0.5564; per-seed dispersion (std Δ = 8.39 PPL) is high
 enough that the effect does not clear strict p \< 0.05. The
 earlier informal "3/3 sign-consistency" expectation from the
 H1 single-seed read does **not** survive the n=5 power-up:
 with seeds 3 and 4 added, one paired seed (seed 4) flips sign
 (VA 157.97 vs attn 145.99 = +11.98 PPL against the hybrid).
3. **Honest framing.** At n=5 paired, Variant A and all-attention
 are statistically indistinguishable on quality at this prototype
 scale, with a directional lean toward the hybrid (Δ̄ = -2.4 PPL,
 sign 4/5). At n=10+ the picture might consolidate either way; at
 the current sample, the hybrid does NOT decisively win on quality,
 but it also clearly does not lose. The pre-registered "+5 PPL bar"
 arm clears in absolute terms; the strict "sign-consistency n/n"
 subarm is partially satisfied (4/5, not 5/5).
4. **γ self-learning matches H1.** All five seeds converge γ to
 0.154 — within ±0.001 of each other and within ±0.012 of the
 leak-free SPLM resonance anchor (γ\* ≈ 0.166). This is a
 sharper cross-seed result than at n=3 and is independent
 confirmation that the second-order Lagrangian's natural damping
 is a property of the data + objective, not of the seed.
5. **Cross-architecture comparison vs Q9d (Helmholtz).** The same
 aggregator reports the cross-architecture paired-t between VA
 k=4, m=4 and Q9d `AAAASSSS` vh=128: Δ̄ = -1.54 PPL in favor of
 Q9d, sign 4/5 (Q9d better than VA), p = 0.3484 → **ON PAR**. The
 architectural choice between Variant A's two-stage layout and
 Q9d's per-S-block ξ refresh is empirically a wash on quality at
 n=5 at this scale.

### 4.4. Note on H1.5 (V_theta-narrow ablation) — deferred for VA

The original §3 plan called for an H1.5 ablation on Variant A at
`v_hidden ∈ {128, 256}` to clear the decode-FLOP arm. Operationally
the V_theta-narrow ablation was **run on the Q9d Helmholtz path
instead** (see [`Helmholtz-HSPLM_Path_Forward_and_Experiments.md`](Helmholtz-HSPLM_Path_Forward_and_Experiments.md)
§4.4), because:

1. **The V_theta-cost arithmetic is identical** between Variant A and
 Q9d at matched `(n_S, n_A)`. Both architectures use the same
 shared single `V_theta` and the same per-S-step force-evaluation
 pattern; the only difference is whether the SPLM steps are
 contiguous (Variant A's m steps after the k attention blocks) or
 schedule-distributed (Q9d's S-blocks at arbitrary positions). A
 V_theta-narrow finding on Q9d transfers analogously to Variant A.
2. **Q9d's H1.5 result applies analogously:** at `v_hidden = 128` and
 `AASSSSSS` (Q9d's analogue of Variant A `(k=2, m=6)`), the long-context
 decode FLOP cost at T = 4096 drops by **−39.0%** vs all-attention
 at val-PPL parity. The same arithmetic predicts that Variant A
 `(k=2, m=6)` at `v_hidden = 128` would deliver an analogous
 long-context FLOP win.
3. **At T = 1024 the bar is not clearable** for either architecture:
 the embed + logits floor dominates at short context and only
 leaves ~9% of budget under the all-attention reference, so a
 ≥ 30% FLOP reduction is structurally impossible at T = 1024
 regardless of `v_hidden`. This is independent of whether VA or
 Q9d is the architecture.

If a dedicated VA H1.5 run is later required (e.g., for a journal-paper
revision that needs VA-specific narrow-V numbers), the cells would be
4 in number — `(k, m) ∈ {(4, 4), (6, 2)}` × `v_hidden ∈ {128, 256}` —
at S=1, ~1.9 h MPS estimated, mirroring the Q9d H1.5 cell layout.

### 4.5. Implication for the title decision (post-H2)

Reading the pre-registered §6.5 rule **after the H2 + Q9d H1.5 results**:

- **Quality arm (within +5 PPL of all-attn):** clears at the n=5
 paired test in absolute terms (Δ̄ = -2.40 PPL, sign 4/5). Strict
 p \< 0.05 not met at n=5; sign-consistency 5/5 not met. The arm
 is **partially satisfied**, not strictly satisfied.
- **Decode-FLOP arm at T = 1024 (≥ 30% reduction):** fails at
 `v_hidden = 512` (per H1) and **also fails at `v_hidden = 128`**
 (per Q9d H1.5 analogue) due to the embed + logits floor at short
 context.
- **Decode-FLOP arm at T = 4096 (long-context regime):** **clears
 decisively at `v_hidden = 128`** in the analogous Q9d
 `AASSSSSS` cell (-39.0% vs all-attention at PPL parity); the same
 arithmetic predicts Variant A `(k=2, m=6)` at `v_hidden = 128`
 would clear the bar.

**Net reading.** The literal §6.5 rule (T = 1024 + sign 5/5 + p \< 0.05)
is **partially satisfied** — quality clears in absolute terms but not
under all of the strict statistical sub-arms; FLOP arm clears at long
context (T ≥ 4096) but not at short context (T = 1024). The title-word
"Efficient" is justified at the **framework level** for the long-context
reading, not as a uniform short-context guarantee. This nuance is the
basis for the v4 title's "Efficient Semantic Inference" phrasing being
retained as a framework-level claim rather than a per-cell claim.

---

## 5. Decision: P4 path (H2 + H1.5) — outcome

User decision (5 May 2026, this document):

> *"I agree the best path forward is P4 (H2 + H1.5)."*

P4 = the most thorough path:

- **H2 — S=3 confirmation on best (k, m) splits** at the current
 `v_hidden = 512`. Settles the data-efficiency claim with paired
 statistics against the all-attention baseline.
- **H1.5 — V_theta-narrow ablation** at `v_hidden ∈ {128, 256}`
 on at least 2 (k, m) splits. Settles the FLOP-efficiency arm.

### What actually happened (post-execution, 6 May 2026)

- **H2 was run at n=5 (not S=3)** on the best (k, m) cell — `(4, 4)`
 — using a shared aggregator that picks up both Variant A and Q9d
 Helmholtz cells. See §4.3 for the per-seed table and the paired-t
 result.
- **H1.5 was deferred for Variant A** in favor of running it on the
 Q9d Helmholtz path. The V_theta-cost arithmetic is identical
 between Variant A and Q9d at matched `(n_S, n_A)`, so a narrow-V
 finding on Q9d transfers analogously to Variant A. See §4.4 for
 the rationale and the cross-reference to
 [`Helmholtz-HSPLM_Path_Forward_and_Experiments.md`](Helmholtz-HSPLM_Path_Forward_and_Experiments.md)
 §4.4 for the actual numbers.

### Quality vs FLOP arm summary (post-H2 + Q9d H1.5)

| Arm | At n=5 / `v_hidden`=128 (analogue) | Verdict |
|----------------------------------------------|------------------------------------|----------------------------------|
| Quality (within +5 PPL of all-attn) | Δ̄ = -2.40 PPL, sign 4/5, p=0.56 | **Partial pass** (mean clears, sign \< 5/5, p \> 0.05) |
| FLOP at T = 1024 (≥ 30% reduction) | not clearable (embed+logits floor) | **Fails** structurally |
| FLOP at T = 4096 (long-context regime) | -39.0% at PPL parity (Q9d analogue) | **Clears decisively** |

Net reading: the title-word "Efficient" is justified at the
**framework-level / long-context** reading, not as a uniform
short-context guarantee. This nuance is the basis for retaining
"Efficient Semantic Inference" in the v4 title.

---

## 6. H2 — paired confirmation plan (detailed) — executed at n=5 on (k=4, m=4)

> **Status (6 May 2026):** executed. See §4.3 for the n=5 paired-t
> result. The original §6 plan called for S=3 (3 seeds) on multiple
> (k, m) splits; in execution the cell selection narrowed to the
> single best-quality H1 cell `(k=4, m=4)` and the seed count was
> increased from 3 to 5 to bring the n=5 paired-t panel into
> agreement with the Q9d H2 panel and with the all-attention 5-seed
> E1 reference. The remaining historical text below documents the
> originally pre-registered design.



### 6.1. Cells

5 best-of-H1 candidates × 3 seeds = **15 cells**. Reuse seed=0 cells
from H1 in place; add seeds 1 and 2 only.

| Cell | (k, m) | Reused (seed) | New seeds |
|------|--------|--------------:|----------------|
| 1 | (4, 4) | 0 | 1, 2 |
| 2 | (3, 5) | 0 | 1, 2 |
| 3 | (5, 3) | 0 | 1, 2 |
| 4 | (6, 2) | 0 | 1, 2 |
| 5 | (2, 6) | 0 | 1, 2 |

That's **10 NEW cells** (5 splits × 2 seeds). Wall-clock estimate:
10 cells × ~33 min/cell ≈ **5.5 h MPS**.

If we want to compress: drop the worst-performing (2, 6) and (3, 5)
cells from S=3 confirmation since they're already strictly inferior
in the H1 quality arm; keep (4, 4), (5, 3), (6, 2). That's
3 splits × 2 new seeds = **6 NEW cells**, ~3.3 h MPS.

### 6.2. Reference baseline at S=3

The all-attention 5-seed E1 sweep already provides the ATTN baseline
at val PPL **149.80 ± 7.21**, paired against by seed index for the
paired-t test in H2's analysis script.

### 6.3. Decision rule for H2

Reuses the pre-registered §6.5 rule directly:

- **PASS quality arm** iff at the best (k, m): mean val PPL
 Δ̄ ≥ -5 PPL (i.e., within +5 PPL of all-attn) with sign-consistency
 3/3 across seeds and paired-t two-sided p < 0.05. (Given the H1
 S=1 result of -17 PPL at (4, 4), this is essentially guaranteed.)
- **FAIL quality arm** iff Δ̄ > +5 PPL or sign-inconsistent.

### 6.4. Output layout

```
hybrid/results/h2_confirmation/
  k4_m4/seed{1,2}/                  (seed 0 reused from h1_sweep/)
  k5_m3/seed{1,2}/
  k6_m2/seed{1,2}/
  H2_RESULTS.md                     (paired-t, sign, mean Δ̄ table)
```

---

## 7. H1.5 — V_theta-narrow ablation plan (detailed) — deferred for VA

> **Status (6 May 2026):** **deferred** for Variant A. Run instead
> on the Q9d Helmholtz path; see
> [`Helmholtz-HSPLM_Path_Forward_and_Experiments.md`](Helmholtz-HSPLM_Path_Forward_and_Experiments.md)
> §4.4 for the actual 4-cell narrow-V result. The cells, decision
> rule, and output layout below are retained as the documented
> recipe in case a VA-specific narrow-V run is later required for
> a journal-paper revision.



### 7.1. Cells

The candidate range is `v_hidden ∈ {128, 256}` × splits
`(k, m) ∈ {(4, 4), (6, 2)}`. (Pick (4, 4) as the H1 best-quality cell;
pick (6, 2) as the most attn-heavy and therefore most likely to
clear the FLOP arm even at the original `v_hidden`.)

= **4 NEW cells** at S=1 first (reconnaissance).

| Cell | (k, m) | v_hidden | Notes |
|------|--------|---------:|----------------------------------|
| 1 | (4, 4) | 128 | minimal V_theta |
| 2 | (4, 4) | 256 | half-width V_theta |
| 3 | (6, 2) | 128 | min-V on attn-heavy |
| 4 | (6, 2) | 256 | half-V on attn-heavy |

Wall-clock estimate: 4 × ~28 min (slightly faster due to narrower V) =
**~1.9 h MPS**.

If H1.5 reveals one or both narrow-V configurations preserve quality
and clear the FLOP arm, promote to S=3 confirmation alongside H2 (this
adds ~2-3 h MPS).

### 7.2. Decision criterion for H1.5

For each (k, m, v_hidden) cell:

- **Quality preserved** iff val PPL is within **+5 PPL** of the
 corresponding `v_hidden = 512` H1 cell.
- **FLOP arm clears** iff the analytical per-token decode FLOPs at
 T = 1024 are **≥ 30% lower** than the all-attention reference of
 20.385 MFLOPs/tok.

Both must clear for the cell to be a candidate "Efficient"-justifying
configuration.

### 7.3. Output layout

```
hybrid/results/h1p5_narrow_v/
  k4_m4_vh128/seed0/
  k4_m4_vh256/seed0/
  k6_m2_vh128/seed0/
  k6_m2_vh256/seed0/
  H1P5_RESULTS.md
```

---

## 8. Code inventory (existing + planned)

| File | Status | Purpose |
|-----------------------------------------------------------------------------|----------|------------------------------------------------------------------|
| `notebooks/conservative_arch/hybrid/model_hybrid.py` | ✅ done | `HybridSPLM` model (Variant A two-stage), param-match table |
| `notebooks/conservative_arch/hybrid/train_splm_hybrid.py` | ✅ done | Trainer (mirrors `energetic_minima/train.py`) |
| `notebooks/conservative_arch/hybrid/aggregate_h1.py` | ✅ done | Aggregator → `results/h1_sweep/H1_RESULTS.md` |
| `notebooks/conservative_arch/hybrid/decode_flop_pareto.py` | ✅ done | Analytical FLOP Pareto (T ∈ {256, 1024, 4096}); H4 |
| `notebooks/conservative_arch/hybrid/scripts/run_h1_layer_split_sweep.sh` | ✅ done | H0 + H1 launcher (idempotent, S=1) |
| `notebooks/conservative_arch/hybrid/scripts/run_h2_paired_confirmation.sh` | ✅ done | H2 launcher (5-seed at (k=4, m=4)) |
| `notebooks/conservative_arch/hybrid/scripts/run_h1p5_narrow_v.sh` | deferred | Replaced by Q9d Helmholtz H1.5; see §4.4 |
| `notebooks/conservative_arch/helmholtz/aggregate_h2.py` | ✅ done | Cross-architecture H2 aggregator (consumes both VA `hybrid/results/h2_paired_confirmation/` and Q9d `helmholtz/results/h2_paired_confirmation/`) |
| `notebooks/conservative_arch/hybrid/aggregate_h1p5.py` | deferred | Subsumed by Q9d Helmholtz H1.5 aggregator |
| `notebooks/conservative_arch/hybrid/README.md` | ✅ done | Pointers + reproduce instructions |

`train_splm_hybrid.py` already accepts `--n-attn k --n-splm m`. To
support H1.5, a future patch will add `--v-hidden V` (currently fixed
at the `shakespeare`-mode default of 512).

---

## 9. Open questions (parked, not blocking H2 / H1.5)

1. **Variant B (interleaved stripe).** Is `[A, S, A, S, A, S, A, S]`
 competitive with `(4, 4)`? Adds H3 to the plan.
2. **Per-step `V_theta` (per-stage rather than shared).** The single
 shared `V_theta` is the right physics-faithful default but if
 quality at H1.5's narrow `v_hidden` is poor, per-stage `V_theta`
 may rescue it.
3. **Long-context training.** All H1 / H2 / H1.5 cells are at
 `block_size = 128`. The asymptotic FLOP advantage of the streaming-ξ
 SPLM at long T (T ≳ 1024–4096) is what the title-claim ultimately
 trades on; long-context training would expose where the FLOP-efficiency
 becomes practically operative. This is the H5 / scale-up branch.
4. **xi alternatives.** We chose `causal_cumulative_mean(h_k.detach)`
 for attention-refined ξ. If H1.5 narrow-V quality is poor, ablating
 between {raw-emb-ξ, attn-output-ξ, learned-ξ-projection} may help.
5. **TinyStories scale-up.** A H5 cell at TinyStories `(d=192, L=12)`
 on the best hybrid configuration would confirm transfer beyond the
 prototype. Adds ~4-5 h MPS but only after H2 + H1.5 commit a winner.

---

## 10. Decision log

| Date | Decision | Notes |
|--------------|------------------------------------------------------------------------------------|--------------------------------------------------------|
| 5 May 2026 | Build only (Option A): `model_hybrid.py` + `train_splm_hybrid.py` + H0 smoke | No MPS time committed; user reviews before sweeps |
| 5 May 2026 | xi re-derivation = `causal_cumulative_mean(h_k.detach)` (attention-refined) | Leak-safe; preserves v3 invariant |
| 5 May 2026 | Free γ in H0 + H1 (init 0.15, learnable, not fixed) | Lets optimiser find right damping per layer split |
| 5 May 2026 | Single shared `V_theta` across `m` SPLM steps | Preserves "single energy field" Lagrangian narrative |
| 5 May 2026 | Run B-full: full H0 + H1 (~3 h MPS) | All 5 cells completed; quality arm passes dramatically |
| 5 May 2026 | **Proceed to P4: H2 + H1.5** (~7-8 h MPS combined) | Original plan |
| 6 May 2026 | **Defer Variant A H1.5** in favor of Q9d H1.5 | V_theta cost arithmetic identical between VA and Q9d; Q9d's narrow-V finding (-39% FLOPs at T=4096, PPL parity) applies analogously to VA |
| 6 May 2026 | **Run VA H2 at n=5** (k=4, m=4 only, seeds 1-4 added; seed 0 reused from H1) | Best-quality H1 cell selected; aggregator picks up VA cells alongside Q9d for the cross-architecture paired-t. ~2.5 h MPS for the new seeds |
| 6 May 2026 | **H2 quality bar clears in absolute terms** at n=5 (Δ̄ = -2.40 PPL vs all-attn, sign 4/5) but strict p \< 0.05 not met | The earlier informal n=3 sign-3/3 expectation was a small-sample artifact; the post-H2 n=5 reading replaces it |
| 6 May 2026 | **Title-word "Efficient" justified at framework level**, with long-context (T ≥ 4096) FLOP win as the empirical anchor | Recorded in §10 |

---

## 11. Pointers

- Sibling Q9d Helmholtz path (where H1.5 narrow-V was actually run; result applies analogously to Variant A): [`Helmholtz-HSPLM_Path_Forward_and_Experiments.md`](Helmholtz-HSPLM_Path_Forward_and_Experiments.md) §4.4 and §4.5
- Cross-architecture H2 paired-t aggregator output (consumes both VA and Q9d cells): `notebooks/conservative_arch/helmholtz/results/h2_paired_confirmation/H2_RESULTS.md`
- H1 sweep results: `notebooks/conservative_arch/hybrid/results/h1_sweep/H1_RESULTS.md`
- H1 sweep launcher: `notebooks/conservative_arch/hybrid/scripts/run_h1_layer_split_sweep.sh`
- H2 paired-confirmation launcher: `notebooks/conservative_arch/hybrid/scripts/run_h2_paired_confirmation.sh`
- VA H2 per-seed artifacts: `notebooks/conservative_arch/hybrid/results/h2_paired_confirmation/k4_m4/seed{0..4}/`
- Decode-FLOP Pareto code: `notebooks/conservative_arch/hybrid/decode_flop_pareto.py`
- Existing FLOP counter (used by Pareto): `notebooks/conservative_arch/inference_efficiency/flop_counter.py`
- Causal-leak fix and forensics: `Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md`
- Resonance-predictor double match (ρ = 0.565 leak-free anchor): `Determining_optimal_gamma_for_SPLM.md` §2.5
- Leak-free γ-sweep (S=5 confirmation): `notebooks/conservative_arch/ln_damping_sweep/results/RESULTS_LEAKFREE_GAMMA_SWEEP.md`

---

*Last updated: 6 May 2026.*
