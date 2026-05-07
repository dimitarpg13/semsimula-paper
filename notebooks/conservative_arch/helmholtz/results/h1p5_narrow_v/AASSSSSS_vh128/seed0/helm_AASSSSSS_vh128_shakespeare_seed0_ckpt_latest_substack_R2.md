# Substack-restricted shared-V_ψ separator — Q9d `AASSSSSS`

- Checkpoint tag: `helm_AASSSSSS_vh128_shakespeare_seed0`
- Schedule (1-indexed layers): 1:A 2:A 3:S 4:S 5:S 6:S 7:S 8:S
- Hidden d: 128 v_hidden: 128 L: 8
- Trajectories pooled: 50 (train + test from CORPUS)

## Full-stack baseline (the v3 §15.8 strict shared-V_ψ test)

Shared-V_ψ overall TEST R² fit jointly across **all** L = 8 layers: **+0.898**.

By design this is expected to be lower than either substack-restricted segment, because the fit must explain both gradient S-blocks and Hopfield-like A-blocks with a single scalar.

## Per-segment refits (fresh V_ψ on restricted samples)

Each segment of length ≥ 2 is fit independently. A 2-block segment yields exactly 0 usable layers (need ell-1 and ell+1 in-segment); ≥ 3-block segments yield 1+ usable layers.

Design doc §4.1 prediction: **S-segment R² ≈ 0.90** (SPLM-substack-like), **A-segment R² in the 0.45-0.65 GPT-2 middle band**.

### S-block segments

- layers 3-8 (length 6): n_train=4,508 n_test=1,144, **mean shv R² = +0.990**
 - layer ell= 4 vel-only R² +0.937, vel+shared-V R² **+0.985**, gain +0.048
 - layer ell= 5 vel-only R² +0.972, vel+shared-V R² **+0.990**, gain +0.018
 - layer ell= 6 vel-only R² +0.983, vel+shared-V R² **+0.992**, gain +0.009
 - layer ell= 7 vel-only R² +0.987, vel+shared-V R² **+0.993**, gain +0.006

### A-block segments

- layers 1-2 (length 2): skipped (no layers usable in segment)
- (no contiguous A-block segment of length ≥ 3 in this schedule; substack-restricted test degenerate)

