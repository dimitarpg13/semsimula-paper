# Substack-restricted shared-V_ψ separator — Q9d `SASASASA`

- Checkpoint tag: `helm_SASASASA_shakespeare_seed0`
- Schedule (1-indexed layers): 1:S 2:A 3:S 4:A 5:S 6:A 7:S 8:A
- Hidden d: 128 v_hidden: 512 L: 8
- Trajectories pooled: 50 (train + test from CORPUS)

## Full-stack baseline (the v3 §15.8 strict shared-V_ψ test)

Shared-V_ψ overall TEST R² fit jointly across **all** L = 8 layers: **+0.864**.

By design this is expected to be lower than either substack-restricted segment, because the fit must explain both gradient S-blocks and Hopfield-like A-blocks with a single scalar.

## Per-segment refits (fresh V_ψ on restricted samples)

Each segment of length ≥ 2 is fit independently. A 2-block segment yields exactly 0 usable layers (need ell-1 and ell+1 in-segment); ≥ 3-block segments yield 1+ usable layers.

Design doc §4.1 prediction: **S-segment R² ≈ 0.90** (SPLM-substack-like), **A-segment R² in the 0.45-0.65 GPT-2 middle band**.

### S-block segments

- layers 1-1 (length 1): skipped (no layers usable in segment)
- layers 3-3 (length 1): skipped (no layers usable in segment)
- layers 5-5 (length 1): skipped (no layers usable in segment)
- layers 7-7 (length 1): skipped (no layers usable in segment)
- (no contiguous S-block segment of length ≥ 3 in this schedule; substack-restricted test degenerate)

### A-block segments

- layers 2-2 (length 1): skipped (no layers usable in segment)
- layers 4-4 (length 1): skipped (no layers usable in segment)
- layers 6-6 (length 1): skipped (no layers usable in segment)
- layers 8-8 (length 1): skipped (no layers usable in segment)
- (no contiguous A-block segment of length ≥ 3 in this schedule; substack-restricted test degenerate)

