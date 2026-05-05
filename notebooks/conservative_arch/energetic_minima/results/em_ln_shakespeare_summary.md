# energetic-minima variant `ln` -- shakespeare training summary

- Tag: `em_ln`
- Device: `mps`
- Parameters: **7,123,076**
- Model config: `{'vocab_size': 50257, 'd': 128, 'max_len': 256, 'v_hidden': 512, 'v_depth': 3, 'L': 8, 'dt': 1.0, 'init_m': 1.0, 'init_gamma': 1.0, 'learn_mgamma': True, 'mass_mode': 'logfreq', 'logfreq_init_alpha': 0.1, 'logfreq_path': '/Users/dimitargueorguiev/git/ml/semsimula/notebooks/conservative_arch/sarf_mass_variant/results/logfreq_surprisal.npy', 'fixed_gamma': None, 'causal_force': True, 'ln_eps': 1e-05, 'ln_after_step': True, 'ln_affine': False}`
- Train config: `{'batch_size': 16, 'block_size': 128, 'steps': 4000, 'lr': 0.0005, 'weight_decay': 0.01, 'warmup_steps': 200, 'grad_clip': 1.0, 'eval_interval': 200, 'eval_iters': 40, 'log_interval': 50, 'lambda_v0': 0.0}`
- Tokens: train=321,124, val=16,901
- Wall-clock time: 2228s
- Final train loss: 4.2398
- Final val loss: 5.1567 (ppl 173.59)
- Final gamma: 0.9583
- Loss curve: `em_ln_shakespeare_loss_curve.png`
- Checkpoint: `em_ln_shakespeare_ckpt_latest.pt`
