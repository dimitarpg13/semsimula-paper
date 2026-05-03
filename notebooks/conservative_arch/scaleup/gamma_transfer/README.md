# E10 — γ-transfer re-tuning experiment (companion to E9)

**Status:** Stage 1 launched only after E9 Phase 1 (matched-baseline arm) completes.
**Pre-registered protocol:** [`docs/Gamma_transfer_pre-registered_protocol.md`](../../../../docs/Gamma_transfer_pre-registered_protocol.md)
**Pre-registration commit:** `75cad01` (April 30, 2026)

## Question

Does the small-scale damping optimum γ\* = 0.30 (E4 / E5, Tiny Shakespeare)
remain optimal at the **E9 scale-up configuration** (TinyStories, 15.75 M
params, max_len = 1024)? Equivalently: where is γ\*\_TS, and is the E9 SPLM
arm result (val PPL = 8.85, fixed γ = 0.30) materially below the optimum?

## Three-stage adaptive design

| Stage | Description | Config | Per-arm wall-clock | Triggered if … |
|---|---|---|---:|---|
| **Stage 1** | γ-grid pilot, ranking only | γ ∈ {0.10, 0.30, 0.60}, 4000 steps, seed 0 | ~6.6 h × 3 | always |
| **Stage 1a** | Boundary expansion | γ = 0.05 (low) or 0.85 (high) | +6.6 h | γ\* at Stage-1 grid boundary |
| **Stage 2** | Full-schedule confirmation | γ\*, 8000 steps, seed 0 | ~13.1 h | γ\* ≠ 0.30 |
| **Stage 3** | Multi-seed paired band | γ\*, 8000 steps, seeds 1 + 2 | ~26.2 h | Δ\_{γ\*} ≥ 0.5 PPL at Stage 2 |

See the protocol document for the full decision rule and the pre-registered
outcome categories (T0 / T1 / NT-material / NT-boundary).

## Files

| file | purpose |
| --- | --- |
| `train_splm_em_ln_gamma_sweep.py` | Thin wrapper around `../train_splm_em_ln_scaleup.py` enforcing protocol modes (`pilot`, `confirmation`) and a required `--fixed-gamma`. |
| `run_stage1.sh` | Driver that launches the three Stage-1 pilots sequentially. Honours `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` (E9-required for the 16 × 512 × 1024 SPLM config to fit in 64 GB unified memory). |
| `results/stage1/g0p{value}_seed0/` | Per-γ Stage-1 results (training log, summary, checkpoint, loss curve). |
| `RESULTS.md` (to be written) | Per-stage write-up after each stage completes. |

## Running

Stage 1 is launched via the shell driver, which matches the E9 Phase 1
launch convention (`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`, sequential arms,
`tee` to stage log). Driver is **not** auto-launched — protocol explicitly
locks Stage-1 launch to be after E9 Phase 1 completes.

```
bash notebooks/conservative_arch/scaleup/gamma_transfer/run_stage1.sh
```

Stage 2 and Stage 3 driver scripts are deliberately *not* checked in until
Stage 1 has reported, so that the γ\* value used in Stage 2 cannot be
post-hoc adjusted; the Stage-1 ranking is the *only* input to Stage 2.

## Decision rule (locked)

Recap from the protocol — full details in
[`docs/Gamma_transfer_pre-registered_protocol.md`](../../../../docs/Gamma_transfer_pre-registered_protocol.md):

- **T0** (most likely, P = 0.60): all three pilots within 0.5 PPL of each
  other at step 4000, **or** γ\* = 0.30. → γ-transfer holds; E9 stands.
- **T1** (P = 0.25): γ\* ≠ 0.30 but Stage-2 Δ\_{γ\*} < 0.5 PPL. → γ-transfer
  technically fails, but PPL gain is below the single-seed noise floor.
- **NT-material** (P = 0.10): γ\* ≠ 0.30 and Δ\_{γ\*} ≥ 0.5 PPL. → SPLM
  benefits materially from re-tuning; E9 result is augmented (not replaced).
- **NT-boundary** (P = 0.05): γ\* outside Stage-1 grid even after one
  boundary expansion. → flagged for separate follow-on protocol.
