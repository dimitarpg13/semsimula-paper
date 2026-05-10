# PARF Q9c -- shakespeare training summary

- Tag: `parf_structural_vphi128_sparse_k4_shakespeare_seed0`
- V_phi kind: `structural` (structural = §5.1-faithful; mlp = unstructured ablation)
- Architecture: PARF-augmented SPLM (Q9c) **Stage 1.5 (Gumbel-softmax sparse routing)** -- single shared V_theta (4-layer MLP) + single shared V_phi pair interaction + score-head MLP for top-k pair selection over past tokens, causal reduction (.detach() on xi-pool and pair source), velocity-Verlet damped Euler-Lagrange step at every layer, NTP-only backprop with straight-through Gumbel-softmax mask (top_k=4, gumbel_tau 1.0 -> 0.1).
- L (depth): 8
- Device: `mps`
- Parameters: **6,550,855** (target ~8.0 M, delta -1.449 M; V_theta=66,049, V_phi=6,786, score_head=12,353)
- Model config: `{'vocab_size': 50257, 'd': 128, 'max_len': 256, 'L': 8, 'v_hidden': 128, 'v_depth': 3, 'dt': 1.0, 'init_m': 1.0, 'init_gamma': 0.15, 'learn_mgamma': True, 'fixed_gamma': None, 'v_phi_kind': 'structural', 'v_phi_d_type': 16, 'v_phi_d_angle': 8, 'v_phi_phi_hidden': 128, 'v_phi_theta_hidden': 128, 'v_phi_mlp_hidden': 64, 'v_phi_C': 1.0, 'v_phi_eps': 0.01, 'v_phi_init_scale': 0.02, 'mass_mode': 'logfreq', 'logfreq_init_alpha': 0.1, 'logfreq_path': '/Users/dimitargueorguiev/git/ml/semsimula/notebooks/conservative_arch/sarf_mass_variant/results/logfreq_surprisal.npy', 'ln_after_step': True, 'ln_eps': 1e-05, 'causal_force': True, 'tie_embeddings': True, 'use_grad_checkpoint': False, 'top_k': 4, 'score_head_hidden': 32, 'score_head_init_scale': 0.02, 'gumbel_tau_init': 1.0, 'gumbel_tau_min': 0.1, 'gumbel_noise': True, 'score_head_use_detached_h_src': True}`
- Train config: `{'batch_size': 16, 'block_size': 128, 'steps': 4000, 'lr': 0.0005, 'weight_decay': 0.01, 'warmup_steps': 200, 'grad_clip': 1.0, 'eval_interval': 200, 'eval_iters': 40, 'log_interval': 50}`
- Tokens: train=321,124, val=16,901
- Wall-clock time: 28321s
- Final train loss: 4.3414
- Final val loss: 5.1742 (ppl 176.65)
- Final gamma: 0.1345
- Final Gumbel tau: 0.1003
- Loss curve: `parf_structural_vphi128_sparse_k4_shakespeare_seed0_loss_curve.png`
- Checkpoint: `parf_structural_vphi128_sparse_k4_shakespeare_seed0_ckpt_latest.pt`
