# energetic-minima variant `sg` -- shakespeare training summary

- Tag: `em_sg_lam1e-03`
- Device: `mps`
- Parameters: **7,123,076**
- Model config: `{'vocab_size': 50257, 'd': 128, 'max_len': 256, 'v_hidden': 512, 'v_depth': 3, 'L': 8, 'dt': 1.0, 'init_m': 1.0, 'init_gamma': 1.0, 'learn_mgamma': True, 'mass_mode': 'logfreq', 'logfreq_init_alpha': 0.1, 'logfreq_path': '/Users/dimitargueorguiev/git/ml/semsimula/notebooks/conservative_arch/sarf_mass_variant/results/logfreq_surprisal.npy', 'fixed_gamma': None, 'causal_force': True}`
- Train config: `{'batch_size': 16, 'block_size': 128, 'steps': 4000, 'lr': 0.0005, 'weight_decay': 0.01, 'warmup_steps': 200, 'grad_clip': 1.0, 'eval_interval': 200, 'eval_iters': 40, 'log_interval': 50, 'lambda_v0': 0.001}`
- Tokens: train=321,124, val=16,901
- Wall-clock time: 2402s
- Final train loss: 4.8867
- Final val loss: 5.5006 (ppl 244.84)
- Final gamma: 0.8632
- Scale-gauge lambda_v0: 0.001
- Loss curve: `em_sg_lam1e-03_shakespeare_loss_curve.png`
- Checkpoint: `em_sg_lam1e-03_shakespeare_ckpt_latest.pt`
