# Helmholtz hybrid (Q9d) -- shakespeare training summary

- Tag: `helm_AAAASSSS_vh128_shakespeare_seed4`
- Schedule: `AAAASSSS` (n_S=4, n_A=4, L=8)
- Architecture: layer-type Helmholtz hybrid -- single shared V_theta on every S-block, per-layer attention on every A-block, velocity-Verlet damped Euler-Lagrange S-step.
- Device: `mps`
- Parameters: **7,324,804** (target ~8.0 M, delta -0.675 M)
- Model config: `{'vocab_size': 50257, 'd': 128, 'max_len': 256, 'schedule': 'AAAASSSS', 'n_head': 4, 'mlp_mult': 4, 'dropout': 0.0, 'v_hidden': 128, 'v_depth': 3, 'dt': 1.0, 'init_m': 1.0, 'init_gamma': 0.15, 'learn_mgamma': True, 'fixed_gamma': None, 'mass_mode': 'logfreq', 'logfreq_init_alpha': 0.1, 'logfreq_path': '/Users/dimitargueorguiev/git/ml/semsimula/notebooks/conservative_arch/sarf_mass_variant/results/logfreq_surprisal.npy', 'ln_after_s_step': True, 'ln_eps': 1e-05, 'causal_force': True, 'tie_embeddings': True}`
- Train config: `{'batch_size': 16, 'block_size': 128, 'steps': 4000, 'lr': 0.0005, 'weight_decay': 0.01, 'warmup_steps': 200, 'grad_clip': 1.0, 'eval_interval': 200, 'eval_iters': 40, 'log_interval': 50}`
- Tokens: train=321,124, val=16,901
- Wall-clock time: 2318s
- Final train loss: 3.7922
- Final val loss: 5.0186 (ppl 151.20)
- Final gamma: 0.1713
- Loss curve: `helm_AAAASSSS_vh128_shakespeare_seed4_loss_curve.png`
- Checkpoint: `helm_AAAASSSS_vh128_shakespeare_seed4_ckpt_latest.pt`
