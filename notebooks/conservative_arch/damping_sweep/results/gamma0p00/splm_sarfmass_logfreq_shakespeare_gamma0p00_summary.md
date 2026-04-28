# SARF+mass(logfreq) -- shakespeare training summary

Variant: **SARF-faithful with mass_mode = `logfreq`**.

- Device: `mps`
- Parameters: **7,123,075**
- Model config: `{'vocab_size': 50257, 'd': 128, 'max_len': 256, 'v_hidden': 512, 'v_depth': 3, 'L': 8, 'dt': 1.0, 'init_m': 1.0, 'init_gamma': 1.0, 'learn_mgamma': True, 'mass_mode': 'logfreq', 'logfreq_init_alpha': 0.1, 'logfreq_path': '/Users/dimitargueorguiev/git/ml/semsimula/notebooks/conservative_arch/sarf_mass_variant/results/logfreq_surprisal.npy', 'fixed_gamma': 0.0}`
- Train config: `{'batch_size': 16, 'block_size': 128, 'steps': 4000, 'lr': 0.0005, 'weight_decay': 0.01, 'warmup_steps': 200, 'grad_clip': 1.0, 'eval_interval': 200, 'eval_iters': 40, 'log_interval': 50}`
- Tokens: train=321,124, val=16,901
- Wall-clock time: 1716s
- Final train loss: 4.4189
- Final val loss: 5.3065 (ppl 201.65)
- Final mass: mean=1.3131, std=0.1625, min=1.0656, max=1.6862
- Final gamma: 0.0000 (fixed at 0.0)
- Loss curve: `splm_sarfmass_logfreq_shakespeare_gamma0p00_loss_curve.png`
- Checkpoint: `splm_sarfmass_logfreq_shakespeare_gamma0p00_ckpt_latest.pt`
- Log: `splm_sarfmass_logfreq_shakespeare_gamma0p00_training_log.jsonl`
