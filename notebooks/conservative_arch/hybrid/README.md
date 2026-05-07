# Hybrid SPLM + Attention (HSPLM, Variant A)

Two-stage hybrid: `k` attention blocks (front) + `m` SPLM integration
steps (back), tied embeddings, single shared `V_theta`, leak-fixed `xi`.

This directory tests the **pre-registered v4 title-justification rule**:

> **"Efficient" is justified iff** some hybrid (k, m) achieves val PPL
> within **+5 PPL** of the all-attention baseline (~150 on Tiny
> Shakespeare at d=128, L=8) **AND** its analytical decode-FLOP cost
> at T=1024 is **≥ 30% lower** than all-attention, both at S=3 with
> sign-consistency 3/3.

## Files

| File | Purpose |
|-------------------------------|--------------------------------------------------------|
| `model_hybrid.py` | `HybridSPLM` model (Variant A two-stage) + param-match |
| `train_splm_hybrid.py` | Trainer, mirrors `energetic_minima/train.py` |
| `aggregate_h1.py` | Aggregator: produces `results/h1_sweep/H1_RESULTS.md` |
| `scripts/run_h1_layer_split_sweep.sh` | H0 + H1 launcher (idempotent, S=1) |

## Architecture (Variant A)

```
h_0 = E[x] + P
for i = 1..k: # k distinct attn blocks
 h_i = AttnBlock_i(h_{i-1})
h_k = LayerNorm(h_k) # boundary projection
xi = causal_cumulative_mean(h_k.detach) # leak-fix invariant
for j = 1..m: # m shared SPLM steps
 f = -grad_h V_theta(xi, h)
 v = (v + dt · f / m_t) / (1 + dt · gamma)
 h = h + dt · v
 h = LayerNorm(h) # if ln_after_step
logits = h @ E^T # tied embeddings
```

The SPLM stage uses a **single shared** `V_theta` across all `m` steps,
matching the canonical SPLM em_ln variant — preserves the
"single energy field" interpretation of paper_v3.

The `xi` re-derivation from `h_k.detach` preserves the v3 causal-leak
fix: the SPLM force `-∂V(ξ, h)/∂h` cannot leak gradient back through
`xi` to anti-causal positions of `h`.

## Reference baselines (already on disk, leak-immune)

| arm | val PPL | source |
|---------------------------------------------|------------|--------------------------------------------------------|
| All-attention (matched GPT-2, `n_layer=8`) | ~150 | `notebooks/conservative_arch/multi_seed/results/` |
| All-SPLM em_ln (free-γ, leak-free) | **173.59** | `notebooks/conservative_arch/energetic_minima/results/`|
| All-SPLM em_ln (γ=0.10–0.15 basin) | ~178–181 | `notebooks/conservative_arch/ln_damping_sweep/results/RESULTS_LEAKFREE_GAMMA_SWEEP.md` |

## Param-match table at d=128 (mass_mode='global', logfreq adds 1 param)

| (n_attn, n_splm) | params | Δ vs 8.0 M baseline |
|------------------|-------------|---------------------|
| (0, 8) | 7,123,331 | -0.877 M |
| (2, 6) | 7,519,875 | -0.480 M |
| (3, 5) | 7,718,147 | -0.282 M |
| (4, 4) | 7,916,419 | -0.084 M |
| (5, 3) | 8,114,691 | +0.115 M |
| (6, 2) | 8,312,963 | +0.313 M |
| (8, 0) | 8,709,507 | +0.710 M |

Hybrid cells `(k, m)` with `k ≥ 1` and `m ≥ 1` lie within ±0.5 M of
the 8.0 M target — param-matching is dominated by the embedding
(~6.4 M) which is shared.

## Phases

### H0 — smoke at the real shakespeare config

Run only the `(k=4, m=4)` cell at the actual training config:

```bash
CELL_LIMIT=1 bash scripts/run_h1_layer_split_sweep.sh
python3 aggregate_h1.py
```

Time: ~30 min on Apple MPS. Verifies training stability + param count.

### H1 — full layer-split reconnaissance (S=1)

Run all 5 cells at S=1:

```bash
bash scripts/run_h1_layer_split_sweep.sh
python3 aggregate_h1.py
```

Time: ~2.5–3 h on Apple MPS. Output:
`results/h1_sweep/H1_RESULTS.md`.

**Phase 1 → Phase 2 gate:** if best (k, m) at S=1 is within +10 PPL of
all-attention, proceed to H2 (S=3 confirmation against the
pre-registered decision rule). If gap > 15 PPL, soften the title
(Option 2 fallback in the v4 title-justification rule) and document
the hybrid result as Future Work.

### H2, H3, H4 (planned, not yet implemented)

- **H2** — best 1-2 (k, m) splits at S=3, paired against all-attention
 and all-SPLM. Decision against the pre-registered rule.
- **H3** — Variant B (interleaved stripe pattern) on the best stripe.
- **H4** — analytical Pareto: per-token decode FLOPs vs val PPL at
 T ∈ {256, 1024, 4096}.

## Resilience and idempotence

The `run_h1_layer_split_sweep.sh` launcher matches the convention of
`ln_damping_sweep/scripts/run_confirmation_5seed_sweep.sh`:

- A cell is **skipped** if it already has a `*_summary.md` on disk.
 Re-running the script is safe after partial completion.
- A failing cell does **not** abort the rest; a `TRAINING_FAILED.txt`
 marker is left in the cell's output directory for follow-up.
- Optional env vars: `CELL_LIMIT=N`, `START_FROM=N`, `FIXED_GAMMA=x`,
 `SEED=N`.

## H0 sanity check (already verified)

```bash
$ cd notebooks/conservative_arch/hybrid
$ python3 model_hybrid.py # smoke test + param-match table
$ python3 train_splm_hybrid.py --mode smoke --n-attn 2 --n-splm 2 --device cpu --seed 0
```

The CPU smoke training drops val loss from 7.78 → 6.13 over 300 steps
(val PPL 2393 → 458) at d=64, n_attn=2, n_splm=2 — confirming the
forward + backward + LR schedule wire up correctly end-to-end.
