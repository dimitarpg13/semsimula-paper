# Substack-restricted shared-V_ψ separator — Q9d `SSAAAASS`

- Checkpoint tag: `helm_SSAAAASS_shakespeare_seed0`
- Schedule (1-indexed layers): 1:S 2:S 3:A 4:A 5:A 6:A 7:S 8:S
- Hidden d: 128 v_hidden: 512 L: 8
- Trajectories pooled: 50 (train + test from CORPUS)

## Full-stack baseline (the v3 §15.8 strict shared-V_ψ test)

Shared-V_ψ overall TEST R² fit jointly across **all** L = 8 layers: **+0.663**.

By design this is expected to be lower than either substack-restricted segment, because the fit must explain both gradient S-blocks and Hopfield-like A-blocks with a single scalar.

## Per-segment refits (fresh V_ψ on restricted samples)

Each segment of length ≥ 2 is fit independently. A 2-block segment yields exactly 0 usable layers (need ell-1 and ell+1 in-segment); ≥ 3-block segments yield 1+ usable layers.

Design doc §4.1 prediction: **S-segment R² ≈ 0.90** (SPLM-substack-like), **A-segment R² in the 0.45-0.65 GPT-2 middle band**.

### S-block segments

- layers 1-2 (length 2): skipped (no layers usable in segment)
- layers 7-8 (length 2): skipped (no layers usable in segment)
- (no contiguous S-block segment of length ≥ 3 in this schedule; substack-restricted test degenerate)

### A-block segments

- layers 3-6 (length 4): n_train=2,254 n_test=572, **mean shv R² = +0.506**
 - layer ell= 4 vel-only R² +0.045, vel+shared-V R² **+0.575**, gain +0.530
 - layer ell= 5 vel-only R² +0.006, vel+shared-V R² **+0.437**, gain +0.431

