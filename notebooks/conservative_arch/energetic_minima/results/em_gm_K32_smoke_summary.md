# energetic-minima variant `gm` -- smoke training summary

- Tag: `em_gm_K32`
- Device: `mps`
- Parameters: **3,228,803**
- Model config: `{'vocab_size': 50257, 'd': 64, 'max_len': 128, 'v_hidden': 128, 'v_depth': 2, 'L': 4, 'dt': 1.0, 'init_m': 1.0, 'init_gamma': 1.0, 'learn_mgamma': True, 'mass_mode': 'logfreq', 'logfreq_init_alpha': 0.1, 'logfreq_path': '/Users/dimitargueorguiev/git/ml/semsimula/notebooks/conservative_arch/sarf_mass_variant/results/logfreq_surprisal.npy', 'gm_K': 32, 'gm_init_kappa': 0.25, 'gm_init_amp': 1.0, 'gm_center_std': 0.5}`
- Train config: `{'batch_size': 8, 'block_size': 64, 'steps': 300, 'lr': 0.001, 'weight_decay': 0.0, 'warmup_steps': 20, 'grad_clip': 1.0, 'eval_interval': 50, 'eval_iters': 20, 'log_interval': 10, 'lambda_v0': 0.0}`
- Tokens: train=321,124, val=16,901
- Wall-clock time: 47s
- Final train loss: 7.7411
- Final val loss: 7.8055 (ppl 2454.07)
- Final gamma: 0.9003
- GM wells: K=32, mean amplitude=1.0060, mean kappa=0.2511
- Loss curve: `em_gm_K32_smoke_loss_curve.png`
- Checkpoint: `em_gm_K32_smoke_ckpt_latest.pt`
