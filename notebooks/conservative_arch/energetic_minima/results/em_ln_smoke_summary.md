# energetic-minima variant `ln` -- smoke training summary

- Tag: `em_ln`
- Device: `mps`
- Parameters: **3,257,796**
- Model config: `{'vocab_size': 50257, 'd': 64, 'max_len': 128, 'v_hidden': 128, 'v_depth': 2, 'L': 4, 'dt': 1.0, 'init_m': 1.0, 'init_gamma': 1.0, 'learn_mgamma': True, 'mass_mode': 'logfreq', 'logfreq_init_alpha': 0.1, 'logfreq_path': '/Users/dimitargueorguiev/git/ml/semsimula/notebooks/conservative_arch/sarf_mass_variant/results/logfreq_surprisal.npy', 'ln_eps': 1e-05, 'ln_after_step': True, 'ln_affine': False}`
- Train config: `{'batch_size': 8, 'block_size': 64, 'steps': 300, 'lr': 0.001, 'weight_decay': 0.0, 'warmup_steps': 20, 'grad_clip': 1.0, 'eval_interval': 50, 'eval_iters': 20, 'log_interval': 10, 'lambda_v0': 0.0}`
- Tokens: train=321,124, val=16,901
- Wall-clock time: 40s
- Final train loss: 6.4161
- Final val loss: 6.4787 (ppl 651.11)
- Final gamma: 0.9590
- Loss curve: `em_ln_smoke_loss_curve.png`
- Checkpoint: `em_ln_smoke_ckpt_latest.pt`
