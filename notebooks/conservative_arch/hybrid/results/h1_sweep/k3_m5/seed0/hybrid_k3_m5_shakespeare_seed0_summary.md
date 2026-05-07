# Hybrid SPLM (Variant A) -- shakespeare training summary

- Tag: `hybrid_k3_m5_shakespeare_seed0`
- Architecture: n_attn=3 attention blocks + n_splm=5 SPLM integration steps (shared V_theta, leak-fixed ξ).
- Device: `mps`
- Parameters: **7,718,148** (target ~8.0 M, delta -0.282 M)
- Model config: `{'vocab_size': 50257, 'd': 128, 'max_len': 256, 'n_attn': 3, 'n_head': 4, 'mlp_mult': 4, 'dropout': 0.0, 'n_splm': 5, 'v_hidden': 512, 'v_depth': 3, 'dt': 1.0, 'init_m': 1.0, 'init_gamma': 0.15, 'learn_mgamma': True, 'fixed_gamma': None, 'mass_mode': 'logfreq', 'logfreq_init_alpha': 0.1, 'logfreq_path': '/Users/dimitargueorguiev/git/ml/semsimula/notebooks/conservative_arch/sarf_mass_variant/results/logfreq_surprisal.npy', 'ln_after_step': True, 'ln_eps': 1e-05, 'causal_force': True, 'tie_embeddings': True}`
- Train config: `{'batch_size': 16, 'block_size': 128, 'steps': 4000, 'lr': 0.0005, 'weight_decay': 0.01, 'warmup_steps': 200, 'grad_clip': 1.0, 'eval_interval': 200, 'eval_iters': 40, 'log_interval': 50}`
- Tokens: train=321,124, val=16,901
- Wall-clock time: 2071s
- Final train loss: 3.8040
- Final val loss: 4.9366 (ppl 139.29)
- Final gamma: 0.1533
- Loss curve: `hybrid_k3_m5_shakespeare_seed0_loss_curve.png`
- Checkpoint: `hybrid_k3_m5_shakespeare_seed0_ckpt_latest.pt`
