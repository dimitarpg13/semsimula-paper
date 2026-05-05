# energetic-minima variant `gm` -- shakespeare training summary

- Tag: `em_gm_K64`
- Device: `mps`
- Parameters: **6,482,179**
- Model config: `{'vocab_size': 50257, 'd': 128, 'max_len': 256, 'v_hidden': 512, 'v_depth': 3, 'L': 8, 'dt': 1.0, 'init_m': 1.0, 'init_gamma': 1.0, 'learn_mgamma': True, 'mass_mode': 'logfreq', 'logfreq_init_alpha': 0.1, 'logfreq_path': '/Users/dimitargueorguiev/git/ml/semsimula/notebooks/conservative_arch/sarf_mass_variant/results/logfreq_surprisal.npy', 'fixed_gamma': None, 'causal_force': True, 'gm_K': 64, 'gm_init_kappa': 0.25, 'gm_init_amp': 1.0, 'gm_center_std': 0.5}`
- Train config: `{'batch_size': 16, 'block_size': 128, 'steps': 4000, 'lr': 0.0005, 'weight_decay': 0.01, 'warmup_steps': 200, 'grad_clip': 1.0, 'eval_interval': 200, 'eval_iters': 40, 'log_interval': 50, 'lambda_v0': 0.0}`
- Tokens: train=321,124, val=16,901
- Wall-clock time: 6592s
- Final train loss: 5.9920
- Final val loss: 6.2965 (ppl 542.65)
- Final gamma: 0.6683
- GM wells: K=64, mean amplitude=1.0527, mean kappa=0.2487
- Loss curve: `em_gm_K64_shakespeare_loss_curve.png`
- Checkpoint: `em_gm_K64_shakespeare_ckpt_latest.pt`
