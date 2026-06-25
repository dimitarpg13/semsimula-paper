# Structured V_θ on Multi-Xi Models — TinyStories Results

Experiments evaluating **SQ3 (MixtureQuadratic) structured V_θ** as a drop-in
replacement for the MLP V_θ across three base model architectures, all on
TinyStories (~5M tokens), d=256, L=8, 16k steps, seed=0.

## Shared configuration

| Parameter | Value |
|-----------|-------|
| d | 256 |
| L | 8 |
| V_phi | structural_competitive (Fock, PARF) / none (SPLM) |
| top_k | 8 (Fock, PARF) / — (SPLM) |
| xi_channels | 4 (all A1–A3 cells) |
| xi_alpha_inits | [0.25, 0.50, 0.75, 0.95] |
| fixed_gamma | 0.30 |
| mass_mode | logfreq |
| steps | 16,000 |
| batch | 16 |
| block | 512 |
| LR | 5e-4 (cosine, 400 warmup) |
| grad_clip | 1.0 |
| lambda_V | 1e-2 |

## Results summary

### Cross-model comparison (A2 cell: SQ3 K=8, the primary config)

| Model | V_phi | Best PPL | Step | V_θ range | V_θ mean |
|-------|-------|----------|------|-----------|----------|
| **FockPARFLM v2.1** | structural_competitive | **10.36** | 14,400 | 19.10 | 0.008 |
| **PARFLM** | structural_competitive | **12.27** | 14,400 | 31.37 | 0.021 |
| **SPLM** | — (none) | **13.33** | 14,400 | 644.89 | 99.81 |

**Takeaway:** The Fock register mechanism provides +1.91 PPL over bare PARF
and +2.97 PPL over SPLM — a clean ablation confirming registers earn their
keep even with structured potentials. The PARF V_phi (structural_competitive)
contributes +1.07 PPL over SPLM (which has no V_phi).

### SPLM V_θ variant sweep (A1–A5)

| Cell | V_θ | K_mix | tau | K_xi | Best PPL | V_θ range |
|------|-----|-------|-----|------|----------|-----------|
| A1 | SQ3 | 4 | 1.0 | 4 | 14.10 | 1,005 |
| **A2** | **SQ3** | **8** | **1.0** | **4** | **13.33** | **645** |
| A3 | SQ3 | 4 | 0.5 | 4 | 14.10 | 644 |
| A4 | SQ3 | 4 | 1.0 | 8 | 14.16 | 3,171 |
| A5 | SQ3 | 8 | 1.0 | 8 | 14.25 | 4,591 |

**Takeaway:** K_mix=8 (A2) is the clear winner. Lowering tau to 0.5 (A3)
does not help. Increasing xi_channels from 4 to 8 (A4, A5) slightly hurts
PPL despite the larger V_θ range — the extra channels may be overfitting
or creating redundancy.

## MLP baseline references

| Model | V_θ | Best PPL | Source |
|-------|-----|----------|--------|
| FockPARFLM v2.1 | MLP (v_hidden=1024) | 8.95 | MLP baseline |
| PARFLM | MLP (v_hidden=1024) | 12.10 | MLP baseline |
| SPLM | MLP (v_hidden=1024) | 11.51 | MLP baseline |

## Source notebooks

| Experiment | Notebook |
|------------|----------|
| Fock MultiXi | `colab_fock_multixi_structured_vtheta.ipynb` |
| PARF MultiXi | `colab_parf_multixi_structured_vtheta.ipynb` |
| SPLM MultiXi | `colab_splm_multixi_structured_vtheta.ipynb` |

## File inventory

Each cell folder (`{experiment}/{cell}/seed0/`) contains:

- `training_log.jsonl` — per-step train loss + periodic val PPL
- `landscape_stats_{cell}.json` — V_θ distribution statistics
- `training_curve_{cell}.png` — val PPL vs step plot
- `v_theta_hist_{cell}.png` — V_θ(ξ, h) histogram
- `attractors_{cell}.json` — (SPLM only) decoded attractor centres
- `summary_{cell}.md` — (SPLM only) per-cell markdown summary

Checkpoint files (`ckpt_best.pt`, `ckpt_latest.pt`) are excluded per
`.gitignore` policy. Original checkpoints are in the GDrive output
folders (`semsimula_{fock,parf,splm}_multixi_structured_vtheta/`).
