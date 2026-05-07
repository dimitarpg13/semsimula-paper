# Substack-restricted shared-V_ψ separator — Q9d `SSSSSSSA`

- Checkpoint tag: `helm_SSSSSSSA_shakespeare_seed0`
- Schedule (1-indexed layers): 1:S 2:S 3:S 4:S 5:S 6:S 7:S 8:A
- Hidden d: 128 v_hidden: 512 L: 8
- Trajectories pooled: 50 (train + test from CORPUS)

## Full-stack baseline (the v3 §15.8 strict shared-V_ψ test)

Shared-V_ψ overall TEST R² fit jointly across **all** L = 8 layers: **+0.795**.

By design this is expected to be lower than either substack-restricted segment, because the fit must explain both gradient S-blocks and Hopfield-like A-blocks with a single scalar.

## Per-segment refits (fresh V_ψ on restricted samples)

Each segment of length ≥ 2 is fit independently. A 2-block segment yields exactly 0 usable layers (need ell-1 and ell+1 in-segment); ≥ 3-block segments yield 1+ usable layers.

Design doc §4.1 prediction: **S-segment R² ≈ 0.90** (SPLM-substack-like), **A-segment R² in the 0.45-0.65 GPT-2 middle band**.

### S-block segments

- layers 1-7 (length 7): n_train=5,635 n_test=1,430, **mean shv R² = +0.981**
 - layer ell= 2 vel-only R² +0.890, vel+shared-V R² **+0.969**, gain +0.079
 - layer ell= 3 vel-only R² +0.975, vel+shared-V R² **+0.988**, gain +0.013
 - layer ell= 4 vel-only R² +0.982, vel+shared-V R² **+0.977**, gain -0.006
 - layer ell= 5 vel-only R² +0.960, vel+shared-V R² **+0.988**, gain +0.029
 - layer ell= 6 vel-only R² +0.927, vel+shared-V R² **+0.985**, gain +0.058

### A-block segments

- layers 8-8 (length 1): skipped (no layers usable in segment)
- (no contiguous A-block segment of length ≥ 3 in this schedule; substack-restricted test degenerate)

