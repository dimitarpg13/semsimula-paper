# SARF+mass(logfreq) -- shakespeare training summary

Variant: **SARF-faithful with mass_mode = `logfreq`**.

- Device: `mps`
- Parameters: **7,123,076**
- Model config: `{'vocab_size': 50257, 'd': 128, 'max_len': 256, 'v_hidden': 512, 'v_depth': 3, 'L': 8, 'dt': 1.0, 'init_m': 1.0, 'init_gamma': 1.0, 'learn_mgamma': True, 'mass_mode': 'logfreq', 'logfreq_init_alpha': 0.1, 'logfreq_path': '/Users/dimitargueorguiev/git/ml/semsimula/notebooks/conservative_arch/sarf_mass_variant/results/logfreq_surprisal.npy'}`
- Train config: `{'batch_size': 16, 'block_size': 128, 'steps': 4000, 'lr': 0.0005, 'weight_decay': 0.01, 'warmup_steps': 200, 'grad_clip': 1.0, 'eval_interval': 200, 'eval_iters': 40, 'log_interval': 50}`
- Tokens: train=321,124, val=16,901
- Wall-clock time: 2390s
- Final train loss: 4.3229
- Final val loss: 5.0786 (ppl 160.55)
- Final mass: mean=1.2773, std=0.1615, min=1.0315, max=1.6490
- Final gamma: 0.8344
- Loss curve: `splm_sarfmass_logfreq_shakespeare_loss_curve.png`
- Checkpoint: `splm_sarfmass_logfreq_shakespeare_ckpt_latest.pt`
- Log: `splm_sarfmass_logfreq_shakespeare_training_log.jsonl`
