# Training Instabilities: Index

**What this folder is.** The record of the training instabilities met while scaling up Fock-PARFLM with a structured potential $V_\theta$, and of the CfC/BAOAB integrator and the mitigations that followed, as short notes. §1–§23 come from [`Training_Instabilities_in_Fock-PARFLM_with_structured_V_theta.md`](https://github.com/dimitarpg13/semsimula-paper/blob/5551315/companion_notes/Training_Instabilities_in_Fock-PARFLM_with_structured_V_theta.md) (written June to August 2026, on the Verlet-style models); §24–§52 from [`CfC_BAOAB_Integrator_and_Mitigations.md`](https://github.com/dimitarpg13/semsimula-paper/blob/5551315/companion_notes/CfC_BAOAB_Integrator_and_Mitigations.md) (written August to October 2026, on the CfC/BAOAB models). The two documents were one record split in August 2026, and they were split into these notes on 2026-11-21; the links above are their last versions, at commit `5551315`. Each note holds its sections verbatim, with only the links changed. The folder also holds the settled reports that were spun out of the record, and its revision history. (Index written 2026-11-21.)

**How to use it.** Section numbers are the originals', and every note keeps them in its headings, so a reference to "§N" anywhere, in the record, in other notes, in the book or in the model cards, finds its note in the tables below. For a topic, start from a settled report where one exists; the numbered sections are the order in which the conclusions were reached.

---

<a name="causal-leak"></a>
## The causal-leak caveat for §1–§23

From the original document's preamble (added July 2026):

> **⚠ Causal-leak relationship (added July 2026).**
>
> **Every experiment in this note used the pre-fix, leaky Fock architecture** —
> the reverse channel reading from a shared/global full-window register state,
> *before* the `prefix_causal_registers` fix. A later trained-checkpoint audit
> proved this architecture leaks future-token information into past-token
> predictions: a **within-window / next-token causal leak** measured at
> **+3.51 nats (≈33× perplexity inflation)** on a `d=384` OpenWebText
> checkpoint. Three consequences for this document:
>
> 1. **The stability findings stand.** The gradient-explosion analysis (`V_θ`
>    unboundedness, the second-order force cascade, embedding/`P`/`E` spikes,
>    and the clipping/watchdog/optimizer remedies) concerns *optimization
>    dynamics* and is **not invalidated** by the leak.
> 2. **All PPL numbers here are leak-contaminated (optimistically low).** Every
>    val-PPL reported below — TinyStories (e.g. 15.95, 18.67, 20.82) and
>    OpenWebText alike — was produced with the reverse channel active, so it is
>    **pending re-certification** with `prefix_causal_registers=True` and the
>    honest-PPL (target-relocation) test. Do not use these values for
>    cross-model comparison as-is.
> 3. **The reverse channel has a second identity.** Where this note treats it
>    purely as a *non-conservative stability liability* (§5, §18.8, §19.3,
>    §23.1), the audit shows it is also the **sole carrier of the causal
>    leak**. The persistent, dominant `reverse_channel_scale` gradient recorded
>    here (the top spike group at `d=768`, §23.1) is the training-time
>    *signature* of the leak: gradient descent kept forcing that scale open
>    because peeking at the future lowered the training loss.
>
> Full analysis: [`Fock-PARFLM_Causal_Leak_Audit_Results.md`](../Fock-PARFLM_Causal_Leak_Audit_Results.md).
> Generalized auditing protocol: [`Framework_for_Causal_Analysis_SemSimula_Models.md`](../Framework_for_Causal_Analysis_SemSimula_Models.md).

---

## §1–§23: the instabilities of the Verlet-style models

| § | section | note |
| --- | --- | --- |
| §1 | Background | [Three_Blowups_and_Their_Causes.md](Three_Blowups_and_Their_Causes.md) |
| §2 | Blowup 1 — Penalty Dominance | [Three_Blowups_and_Their_Causes.md](Three_Blowups_and_Their_Causes.md) |
| §3 | Blowup 2 — Directional Instability | [Three_Blowups_and_Their_Causes.md](Three_Blowups_and_Their_Causes.md) |
| §4 | Blowup 3 — EMA Watchdog Miscalibration | [Three_Blowups_and_Their_Causes.md](Three_Blowups_and_Their_Causes.md) |
| §5 | Why Fock-PARFLM is More Susceptible than SPLM | [Three_Blowups_and_Their_Causes.md](Three_Blowups_and_Their_Causes.md) |
| §6 | Fixes Applied and Their Mathematical Justification | [Fixes_Watchdog_and_Residual_Risk.md](Fixes_Watchdog_and_Residual_Risk.md) |
| §7 | EMA Watchdog Design | [Fixes_Watchdog_and_Residual_Risk.md](Fixes_Watchdog_and_Residual_Risk.md) |
| §8 | Residual Risk and Recommendations | [Fixes_Watchdog_and_Residual_Risk.md](Fixes_Watchdog_and_Residual_Risk.md) |
| §9 | Architectural Path Forward: Gaussian Wells with SARF Anchors | [Gaussian_Wells_with_SARF_Anchors.md](Gaussian_Wells_with_SARF_Anchors.md) |
| §10 | Implementation Diagnostic: SARF Well Deactivation by `ln_after_step` Scale Mismatch | [Well_Deactivation_and_Precision_Explosion.md](Well_Deactivation_and_Precision_Explosion.md) |
| §11 | G3 Sigma Drift: Well Deactivation During Training | [Well_Deactivation_and_Precision_Explosion.md](Well_Deactivation_and_Precision_Explosion.md) |
| §12 | G1 Precision Explosion: Blowup 2 Revisited via Unbounded Gaussian Precision | [Well_Deactivation_and_Precision_Explosion.md](Well_Deactivation_and_Precision_Explosion.md) |
| §13 | Phase 5 Blowup: LR-Induced Instability Beyond the Bounded Potential | [Phase5_LR_Blowup_and_Scale_Diversity.md](Phase5_LR_Blowup_and_Scale_Diversity.md) |
| §14 | Scale-Diversity Analysis: Why TinyStories Is Stable and OpenWebText Is Not | [Phase5_LR_Blowup_and_Scale_Diversity.md](Phase5_LR_Blowup_and_Scale_Diversity.md) |
| §15 | Multi-Head Experiment: 1/r Gradient Explosion at Step 43K | [Multi_Head_Gradient_Explosion_and_Gradient_Management.md](Multi_Head_Gradient_Explosion_and_Gradient_Management.md) |
| §16 | Gradient-Management Refinements: Per-Module Clipping, Centralisation, and Optimizer Choice | [Multi_Head_Gradient_Explosion_and_Gradient_Management.md](Multi_Head_Gradient_Explosion_and_Gradient_Management.md) |
| §17 | Hybrid Gaussian + Quadratic Background: Bridging the Stability–Expressivity Gap | [Hybrid_Gaussian_Quadratic_Background.md](Hybrid_Gaussian_Quadratic_Background.md) |
| §18 | Embedding Spikes: Anatomy, Root Cause, and Propagation | [Embedding_Spikes.md](Embedding_Spikes.md) |
| §19 | Comparison with Embedding Spikes in Classic Attention-Based Transformers | [Embedding_Spikes.md](Embedding_Spikes.md) |
| §20 | Remediation: Per-Group Clipping vs Global Norm Clipping | [Embedding_Spikes.md](Embedding_Spikes.md) |
| §21 | Scaling Outlook and Hardening Recommendations | [Scaling_Outlook_Tied_Embeddings_and_the_d1024_Cascade.md](Scaling_Outlook_Tied_Embeddings_and_the_d1024_Cascade.md) |
| §22 | Tied vs. Untied Embeddings: A Distinct Instability Mode | [Scaling_Outlook_Tied_Embeddings_and_the_d1024_Cascade.md](Scaling_Outlook_Tied_Embeddings_and_the_d1024_Cascade.md) |
| §23 | d=1024 Universal Instability: The Second-Order Gradient Cascade | [Scaling_Outlook_Tied_Embeddings_and_the_d1024_Cascade.md](Scaling_Outlook_Tied_Embeddings_and_the_d1024_Cascade.md) |

## §24–§52: the CfC/BAOAB integrator and its mitigations

| § | section | note |
| --- | --- | --- |
| §24 | BAOAB + CfC Propagator: Eliminating the Force Cascade at Source | [CfC_BAOAB_Propagator_and_the_Cascade.md](CfC_BAOAB_Propagator_and_the_Cascade.md) |
| §25 | Late-Training Spike Emergence: The Cascade is Universal, Not Depth-Specific | [CfC_BAOAB_Propagator_and_the_Cascade.md](CfC_BAOAB_Propagator_and_the_Cascade.md) |
| §26 | The Damping Hypothesis: Is Low γ the Dominant Cause of the Cascade? | [CfC_BAOAB_Propagator_and_the_Cascade.md](CfC_BAOAB_Propagator_and_the_Cascade.md) |
| §27 | Empirical Depth-Code Growth: Boundary Layers Dominate in Both Integrators | [CfC_BAOAB_Propagator_and_the_Cascade.md](CfC_BAOAB_Propagator_and_the_Cascade.md) |
| §28 | Proposed (Deferred) Mitigation: Clamping the Low-Rank Precision Factor B_k | [Bk_Clamp_and_Principled_Directions.md](Bk_Clamp_and_Principled_Directions.md) |
| §29 | Principled Directions Beyond the B_k Clamp | [Bk_Clamp_and_Principled_Directions.md](Bk_Clamp_and_Principled_Directions.md) |
| §30 | Concrete Sketch: The Low-Rank Exponential Substep | [LowRank_Exponential_Substep_and_SCAF_Audit_Plan.md](LowRank_Exponential_Substep_and_SCAF_Audit_Plan.md) |
| §31 | SCAF Phase 7b/7c Audit Plan for Tuning precision_lr_max (L=16, and now L=8) | [LowRank_Exponential_Substep_and_SCAF_Audit_Plan.md](LowRank_Exponential_Substep_and_SCAF_Audit_Plan.md) |
| §32 | L=8 `baoab_cfc` Baseline: Extended Trajectory (steps 27,000–39,867) and the Decision to Switch Mid-Run | [L8_Runs_and_the_Bracketing_Result.md](L8_Runs_and_the_Bracketing_Result.md) |
| §33 | The Bracketing Result Is Modest and Non-Escalating: B_k Is Not the Primary Driver, and a Root-Cause Workflow for the Non-V_θ Spikes | [L8_Runs_and_the_Bracketing_Result.md](L8_Runs_and_the_Bracketing_Result.md) |
| §34 | `baoab_cfc_lowrank` at L=8 Scale: Correct and Stable, but Not Production-Feasible | [L8_Runs_and_the_Bracketing_Result.md](L8_Runs_and_the_Bracketing_Result.md) |
| §35 | Phase 1/2 Validated: First `replay_spike_batch` Result (Step 37,763) and a Clean Cascade Signature | [First_Replays_Spike_Capture_and_Resumable_Training.md](First_Replays_Spike_Capture_and_Resumable_Training.md) |
| §36 | Decoupling Spike Capture from the Reload Trigger: the Plateau, Not Just the Rare Crisis, Is the Real Target | [First_Replays_Spike_Capture_and_Resumable_Training.md](First_Replays_Spike_Capture_and_Resumable_Training.md) |
| §37 | Making Cell 6 Resumable In-Place, and Extracting `grad_clip_utils.py` | [First_Replays_Spike_Capture_and_Resumable_Training.md](First_Replays_Spike_Capture_and_Resumable_Training.md) |
| §38 | Seven Replays In: Two Distinct Failure Modes, Not One -- and Only the Localized One Has Crossed 500 | [Two_Failure_Modes_and_the_Token_Minority_Conjecture.md](Two_Failure_Modes_and_the_Token_Minority_Conjecture.md) |
| §39 | The Token-Minority Conjecture Is Falsified Twice Over: the Localized Mode Is Batch-Wide, Not Batch-Specific | [Two_Failure_Modes_and_the_Token_Minority_Conjecture.md](Two_Failure_Modes_and_the_Token_Minority_Conjecture.md) |
| §40 | Does `baoab_cfc_lowrank` Address the Localized Mode? Reconciling §33's Verdict, and a Targeted Ablation Test | [Localized_Mode_and_Chronic_Replays.md](Localized_Mode_and_Chronic_Replays.md) |
| §41 | Chronic, Not Transient: Three New Replays Refine §33 and §38, and Motivate Turning On `precision_lr_max` | [Localized_Mode_and_Chronic_Replays.md](Localized_Mode_and_Chronic_Replays.md) |
| §42 | Step 1 Validated: `precision_lr_max` and `baoab_cfc_lowrank` Both Collapse All Three Replays, a Hook Bug and a Checkpoint-Loading Pitfall Found Along the Way, and the Cap Switched On | [precision_lr_max_and_LowRank_Validated.md](precision_lr_max_and_LowRank_Validated.md) |
| §43 | Four Repeats of the Same Eval-Time OOM at Step 47,500: `gc.collect()` Was Never Going to Fix It, and Why | [Eval_Time_OOM_at_Step_47500.md](Eval_Time_OOM_at_Step_47500.md) |
| §44 | Two Near-Trigger E/P-Led Replays: Layer-Profile Shape, Not Group Identity, Discriminates the Mechanisms, and the §39 Anti-Correlation Extends to This Regime | [Near_Trigger_Replays_and_the_Clip_Ablation.md](Near_Trigger_Replays_and_the_Clip_Ablation.md) |
| §45 | A `precision_lr_max`-Style Clip Ablation Doesn't Make Sense, and `replay_clip_ablation` Tests the Question That Does: Clip Order | [Near_Trigger_Replays_and_the_Clip_Ablation.md](Near_Trigger_Replays_and_the_Clip_Ablation.md) |
| §46 | A Checkpoint-Recompute Divergence Took Down the Whole Session: 4,252 Steps Lost, and a Wall-Clock Autosave Added | [Session_Loss_clip_then_sum_and_Replay_Fidelity.md](Session_Loss_clip_then_sum_and_Replay_Fidelity.md) |
| §47 | First Production Validation of `clip_then_sum` (§45.4): Plateau Broken, No New Spikes Through Step 61,650 | [Session_Loss_clip_then_sum_and_Replay_Fidelity.md](Session_Loss_clip_then_sum_and_Replay_Fidelity.md) |
| §48 | `replay_spike_batch` Predates `clip_then_sum`: a Diagnostic Fidelity Bug, Found and Fixed, and What the Corrected Replays of Steps 70522, 71194, and 71703 Show | [Session_Loss_clip_then_sum_and_Replay_Fidelity.md](Session_Loss_clip_then_sum_and_Replay_Fidelity.md) |
| §49 | The `log_tau` Runaway: One Register's Temperature Is Diverging From the Pool, and Two Falsified Predictions on the Way to Finding It | [log_tau_Runaway_Record.md](log_tau_Runaway_Record.md) |
| §50 | Does QK-Normalisation Actually Break the Runaway? A Gradient-Flow Analysis of the Creation Gate's Temperature | [log_tau_Runaway_Record.md](log_tau_Runaway_Record.md) |
| §51 | Most of the Measured `log_tau` Drift Was Weight Decay on a Log-Parameterised Temperature | [log_tau_Runaway_Record.md](log_tau_Runaway_Record.md) |
| §52 | Curvature Geometry Instruments, and a Plan for Splitting This Note | [Curvature_Instruments_and_the_Split_Plan.md](Curvature_Instruments_and_the_Split_Plan.md) |

## Settled reports spun out of the record

| note | from | what it covers |
| --- | --- | --- |
| [Diagnostic_Programme_in_CfC_BAOAB_Integrator.md](Diagnostic_Programme_in_CfC_BAOAB_Integrator.md) | §35–§39 | the spike-forensics pipeline for the CfC/BAOAB model: how spikes arise, the instruments that capture and replay them, and what the replays showed |
| [Progressive_Curvature_Confinement_for_Aniso_Gaussian_Vtheta.md](Progressive_Curvature_Confinement_for_Aniso_Gaussian_Vtheta.md) | §28–§29 | progressive confinement of the anisotropic Gaussian potential's curvature, the programme behind the B_k clamp |
| [Register_Temperature_Instability_in_the_Fock_Creation_Gate.md](Register_Temperature_Instability_in_the_Fock_Creation_Gate.md) | §49–§51 | the register-temperature (`log_tau`) instability of the Fock creation gate, settled |
| [Curvature_Diagnostics_and_Rank_Selection_for_Aniso_Gaussian_Vtheta.md](Curvature_Diagnostics_and_Rank_Selection_for_Aniso_Gaussian_Vtheta.md) | §52 | curvature diagnostics and the choice of rank for the anisotropic Gaussian potential |
| [CfC_BAOAB_Integrator_and_Mitigations_CHANGELOG.md](CfC_BAOAB_Integrator_and_Mitigations_CHANGELOG.md) |  | the revision history of the CfC/BAOAB note, 2026-07 to 2026-09 |

## Where the code is

- **The Verlet-style models (§1–§23).** The fixes are in `notebooks/conservative_arch/scaleup/colab_fock_structured_vtheta_openwebtext_phase4.ipynb`. SARF anchors: `notebooks/semsim_simulator/estimators/sarf_anchors.py`; SARF-faithful dynamics: `notebooks/conservative_arch/sarf_variant/model_sarf.py`; Gaussian-well and SARF-anchored potentials: `notebooks/conservative_arch/parf/model_gaussian_vtheta.py`. Notebooks: the TinyStories ablation `colab_fock_gaussian_sarf_vtheta.ipynb`, the OpenWebText Phase 5 scale-up `colab_fock_gaussian_sarf_openwebtext_phase5.ipynb`, the multi-head experiment `colab_fock_multihead_openwebtext.ipynb`, the hybrid background `colab_hybrid_gaussian_quad_vtheta.ipynb`, and the depth-conditioned model `colab_fock_depthcond_vtheta_openwebtext.ipynb`, all in `notebooks/conservative_arch/scaleup/`.
- **The CfC/BAOAB models (§24–§52).** The propagator is `notebooks/conservative_arch/parf/cfc_baoab.py`, wired into the layer step in `notebooks/conservative_arch/parf/model_parf_multixi.py`; the production notebook was `notebooks/conservative_arch/scaleup/colab_fock_cfc_baoab_aniso_gaussian_openwebtext_d384.ipynb`. The anisotropic Gaussian potential is `notebooks/conservative_arch/parf/model_aniso_gaussian_vtheta.py`; the SCAF stiffness audit (the Phase 7b/7c Weyl bound) is `src/scaf/probes/stiffness.py` in `semsimula-scaf`.

## Conventions

Each note opens with a "What this note is" paragraph naming its sections and its source, then its contents. The text below that is the original's, so its dates, its "this note" and its forward references are as written then. New material on these topics goes into a note of its own, one topic per file, not into these.

