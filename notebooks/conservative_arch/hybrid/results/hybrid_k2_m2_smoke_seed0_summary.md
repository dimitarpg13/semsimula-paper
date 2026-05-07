# Hybrid SPLM (Variant A) -- smoke training summary

- Tag: `hybrid_k2_m2_smoke_seed0`
- Architecture: n_attn=2 attention blocks + n_splm=2 SPLM integration steps (shared V_theta, leak-fixed ξ).
- Device: `cpu`
- Parameters: **3,357,892** (target ~8.0 M, delta -4.642 M)
- Model config: `{'vocab_size': 50257, 'd': 64, 'max_len': 128, 'n_attn': 2, 'n_head': 4, 'mlp_mult': 4, 'dropout': 0.0, 'n_splm': 2, 'v_hidden': 128, 'v_depth': 2, 'dt': 1.0, 'init_m': 1.0, 'init_gamma': 0.15, 'learn_mgamma': True, 'fixed_gamma': None, 'mass_mode': 'logfreq', 'logfreq_init_alpha': 0.1, 'logfreq_path': '/Users/dimitargueorguiev/git/ml/semsimula/notebooks/conservative_arch/sarf_mass_variant/results/logfreq_surprisal.npy', 'ln_after_step': True, 'ln_eps': 1e-05, 'causal_force': True, 'tie_embeddings': True}`
- Train config: `{'batch_size': 8, 'block_size': 64, 'steps': 300, 'lr': 0.001, 'weight_decay': 0.0, 'warmup_steps': 20, 'grad_clip': 1.0, 'eval_interval': 50, 'eval_iters': 20, 'log_interval': 10}`
- Tokens: train=321,124, val=16,901
- Wall-clock time: 64s
- Final train loss: 6.1076
- Final val loss: 6.1273 (ppl 458.20)
- Final gamma: 0.1476
- Loss curve: `hybrid_k2_m2_smoke_seed0_loss_curve.png`
- Checkpoint: `hybrid_k2_m2_smoke_seed0_ckpt_latest.pt`
