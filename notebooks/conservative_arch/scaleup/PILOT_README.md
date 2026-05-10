# `paper_tmlr_1` scale-up pilot — Colab harness README

**Status:** ready to run. Awaiting Colab Pro+ subscription.
**Notebook:** [`colab_pilot.ipynb`](./colab_pilot.ipynb)
**Trainers:**
- `train_matched_baseline_scaleup.py` (existing, E9)
- `train_splm_em_ln_scaleup.py`        (existing, E9)
- `train_helmholtz_scaleup.py`         (new, this PR)
- `train_hybrid_scaleup.py`            (new, this PR)
- `train_parf_scaleup.py`              (new, this PR)
**Aggregator:** `aggregate_pilot_results.py` (new, this PR)

---

## Why this pilot

The E9 SPLM scale-up answers a single question:

> Does the +25 PPL gap that SPLM em_ln (γ⋆=0.30) showed over a parameter-matched
> GPT-2 attention baseline on Tiny Shakespeare at ~7-8 M params survive a
> **2.2× model scale-up** to ~16-19 M params on a **16× larger corpus** at a
> **4× longer context length**?

The matched-attn vs SPLM em_ln pair gives us **2 of 5 cells**.
The pilot adds **three more architectural cells** at the same configuration so
that `paper_tmlr_1` can claim — or honestly retract — the architectural advantage
not just for "all-SPLM em_ln" but for the **whole SPLM family** at scale:

| # | Arm                          | Small-scale anchor (Shakespeare)             |
|---|------------------------------|----------------------------------------------|
| 1 | matched-attn baseline        | reference (existing E9)                      |
| 2 | SPLM em_ln (all-SPLM)        | val PPL ≈ 175.7 (with E5 winner γ=0.30)      |
| 3 | Helmholtz Q9d (AAAASSSS)     | H1 winner: val PPL **135.03**                |
| 4 | Hybrid Variant A (k=4, m=4)  | H1 winner: val PPL **133.01**                |
| 5 | PARF Q9c sparse top-k=4      | P5 winner: val PPL **176.65**                |

If at scale-up any of cells 3–5 deliver Δ > +5 PPL versus the matched-attn
baseline, that becomes the **headline contribution** of `paper_tmlr_1` (or
re-frames it as a positive instrument story regardless of "globally not"
outcome). The headline result moves from "SPLM em_ln matches attention" to
"the SPLM family includes architectures that **beat** matched attention at
the relevant scale".

## Configuration (locked at the existing E9 protocol)

- **Corpus:** TinyStories, GPT-2 BPE, ~5 M training tokens, ~140 k validation tokens
- **Context:** `max_len=1024`, `block_size=512`
- **Model:** d=256, L=8, v_hidden=1024, v_depth=3, max_len=1024
- **Optimisation:** AdamW(0.9, 0.95) lr=5e-4 (cosine, 400-step warmup),
  weight_decay=0.01, grad_clip=1.0
- **Training budget:** 8000 steps × batch 16 × block 512 = ~65.5 M tokens seen
- **Evaluation:** every 400 steps, 40 batches × 16 × 512 = ~327 k tokens
- **Seeds:** seed 0 in this pilot pass; seed 1 + 2 reserved for a follow-up
  if Phase 1 is inconclusive

Per-arm parameter counts (verified locally):

| Arm                          | Total params | Notes                              |
|------------------------------|-------------:|------------------------------------|
| matched-attn baseline        | 19,449,344   | n_head=4, mlp_mult=4, tied embed   |
| SPLM em_ln                   | 15,752,961   | shared V_θ, fixed γ=0.30           |
| Helmholtz Q9d AAAASSSS       | 18,912,516   | 4 attn + 4 SPLM steps, free γ      |
| Hybrid Variant A (k=4, m=4)  | 18,913,028   | 4 attn bottom + 4 SPLM top, free γ |
| PARF Q9c sparse k=4          | 15,797,191   | V_θ=2.6 M + V_φ=19 K + score=25 K  |

All five arms sit in the **15.8–19.5 M** range — well-matched.

## Decision rule

Δ = PPL(matched-attn) − PPL(arm). With **Δ_min = 5 PPL**:
- Δ > +5 ⇒ SPLM-family arm wins
- |Δ| ≤ 5 ⇒ tie
- Δ < −5 ⇒ baseline wins for that arm

The aggregator emits a Δ table per arm in `PILOT_RESULTS.md`.

## Wall-clock estimates

Per-cell estimates on Colab GPUs (rough, will be measured on first smoke run):

| GPU          | matched-attn | SPLM em_ln | Helmholtz | Hybrid VA | PARF sparse | Total |
|--------------|-------------:|-----------:|----------:|----------:|------------:|------:|
| A100 40 GB   | ~2.5 h       | ~3.5 h     | ~3.5 h    | ~3.0 h    | ~5.5 h      | ~18 h |
| L4 24 GB     | ~4.0 h       | ~5.0 h     | ~5.0 h    | ~4.5 h    | ~9.0 h      | ~27 h |
| T4 15 GB     | ~6.5 h       | ~8.0 h     | ~8.5 h    | ~7.5 h    | ~15.0 h     | ~45 h |

**Recommendation:** Colab Pro+ A100 (40 GB). The full 5-cell pilot fits in a
single ~20 h session window. If session disconnects mid-arm, the notebook is
**idempotent**: it skips arms whose `*_summary.md` already exists in Drive.

## How to run on Colab

1. Open [`colab_pilot.ipynb`](./colab_pilot.ipynb) in Colab.
2. Runtime → Change runtime type → **GPU** → A100 (Pro+ subscribers) or L4 (Pro).
3. Execute cells **top-to-bottom**:
   1. Mount Drive
   2. Clone semsimula repo
   3. Install deps (transformers, datasets, pyarrow)
   4. Verify GPU + data files
   5. **Run the smoke test** (5–10 min). If any arm fails here, fix before the full pilot.
   6. Run the five training cells **one at a time** (you can monitor progress live).
   7. Run the aggregator and view `PILOT_RESULTS.md` + `pilot_loss_curves.png` + `pilot_pareto.png`.

All artifacts go to `/content/drive/MyDrive/semsimula_pilot/results/`.

## Persistence model

- **Training scripts** live in the cloned repo (ephemeral Colab disk).
  Re-cloned on each session.
- **Data files** (`tinystories_gpt2_1files_5000000toks.npz`, ~19 MB, tracked in
  git) ship with the repo. No download step needed.
- **Logfreq surprisal file** (`logfreq_surprisal_tinystories.npy`, ~196 KB,
  tracked in git) ships with the repo.
- **Per-arm results** (`*_summary.md`, `*_loss_curve.png`, `*_training_log.jsonl`,
  `*_ckpt_latest.pt`) are written **directly to Drive** so they survive Colab
  session teardown.
- **Per-arm subprocess logs** (`<arm>_seed0_<timestamp>.log`) also go to Drive
  so we can audit any cell post-mortem.

## After the pilot

1. **Download** `PILOT_RESULTS.md`, `pilot_loss_curves.png`, `pilot_pareto.png`
   from Drive to your laptop.
2. **Place** them at `notebooks/conservative_arch/scaleup/results/pilot/` in
   the repo.
3. **Commit** the markdown + plots; **do not commit** the `.pt` checkpoints
   (each is ~80 MB and not needed for the paper).
4. **Update `paper_tmlr_1` Discussion §10** with a one-paragraph summary
   citing the headline Δ vs the matched baseline.
5. If the headline Δ is decisive (single seed |Δ| ≥ 20 PPL for any
   SPLM-family arm), submit. Otherwise, re-run the dispatch cells with
   `seed=1` (which stages a Phase 2 Δ-pooling).

## Where to commit pilot artifacts

A new directory `notebooks/conservative_arch/scaleup/results/pilot/` holds
the seed-0 pilot artifacts (excluding `.pt` checkpoints). Update
`scaleup/README.md`'s "Phase 1 outcome" entry once results land.
