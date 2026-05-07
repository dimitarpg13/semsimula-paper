# Substack-restricted shared-V_ψ separator — Q9d `AAAASSSS`

- Checkpoint tag: `helm_AAAASSSS_vh128_shakespeare_seed0`
- Schedule (1-indexed layers): 1:A 2:A 3:A 4:A 5:S 6:S 7:S 8:S
- Hidden d: 128 v_hidden: 128 L: 8
- Trajectories pooled: 50 (train + test from CORPUS)

## Full-stack baseline (the v3 §15.8 strict shared-V_ψ test)

Shared-V_ψ overall TEST R² fit jointly across **all** L = 8 layers: **+0.868**.

By design this is expected to be lower than either substack-restricted segment, because the fit must explain both gradient S-blocks and Hopfield-like A-blocks with a single scalar.

## Per-segment refits (fresh V_ψ on restricted samples)

Each segment of length ≥ 2 is fit independently. A 2-block segment yields exactly 0 usable layers (need ell-1 and ell+1 in-segment); ≥ 3-block segments yield 1+ usable layers.

Design doc §4.1 prediction: **S-segment R² ≈ 0.90** (SPLM-substack-like), **A-segment R² in the 0.45-0.65 GPT-2 middle band**.

### S-block segments

- layers 5-8 (length 4): n_train=2,254 n_test=572, **mean shv R² = +0.997**
 - layer ell= 6 vel-only R² +0.982, vel+shared-V R² **+0.997**, gain +0.015
 - layer ell= 7 vel-only R² +0.984, vel+shared-V R² **+0.997**, gain +0.013

### A-block segments

- layers 1-4 (length 4): n_train=2,254 n_test=572, **mean shv R² = +0.702**
 - layer ell= 2 vel-only R² +0.002, vel+shared-V R² **+0.759**, gain +0.757
 - layer ell= 3 vel-only R² +0.007, vel+shared-V R² **+0.646**, gain +0.639

