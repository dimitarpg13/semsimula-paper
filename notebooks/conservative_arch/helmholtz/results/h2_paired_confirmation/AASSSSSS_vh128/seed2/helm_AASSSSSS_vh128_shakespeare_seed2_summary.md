# Helmholtz hybrid (Q9d) -- shakespeare training summary

- Tag: `helm_AASSSSSS_vh128_shakespeare_seed2`
- Schedule: `AASSSSSS` (n_S=6, n_A=2, L=8)
- Architecture: layer-type Helmholtz hybrid -- single shared V_theta on every S-block, per-layer attention on every A-block, velocity-Verlet damped Euler-Lagrange S-step.
- Device: `mps`
- Parameters: **6,928,260** (target ~8.0 M, delta -1.072 M)
- Model config: `{'vocab_size': 50257, 'd': 128, 'max_len': 256, 'schedule': 'AASSSSSS', 'n_head': 4, 'mlp_mult': 4, 'dropout': 0.0, 'v_hidden': 128, 'v_depth': 3, 'dt': 1.0, 'init_m': 1.0, 'init_gamma': 0.15, 'learn_mgamma': True, 'fixed_gamma': None, 'mass_mode': 'logfreq', 'logfreq_init_alpha': 0.1, 'logfreq_path': '/Users/dimitargueorguiev/git/ml/semsimula/notebooks/conservative_arch/sarf_mass_variant/results/logfreq_surprisal.npy', 'ln_after_s_step': True, 'ln_eps': 1e-05, 'causal_force': True, 'tie_embeddings': True}`
- Train config: `{'batch_size': 16, 'block_size': 128, 'steps': 4000, 'lr': 0.0005, 'weight_decay': 0.01, 'warmup_steps': 200, 'grad_clip': 1.0, 'eval_interval': 200, 'eval_iters': 40, 'log_interval': 50}`
- Tokens: train=321,124, val=16,901
- Wall-clock time: 1637s
- Final train loss: 3.7833
- Final val loss: 5.0227 (ppl 151.82)
- Final gamma: 0.1592
- Loss curve: `helm_AASSSSSS_vh128_shakespeare_seed2_loss_curve.png`
- Checkpoint: `helm_AASSSSSS_vh128_shakespeare_seed2_ckpt_latest.pt`
