# The Depth Ladder: Index

**What this folder is.** The depth-ladder programme, which trains the conservative language models of the Semantic Simulation framework one mechanism at a time against a matched GPT-2, written as short standalone notes. Each note covers one part of the running log [Depth_Ladder_and_Matched_Baseline_Protocol.md](../Depth_Ladder_and_Matched_Baseline_Protocol.md), with its questions, pre-registered calls, results and verdicts, and links back to the log for the full record. Together the notes cover the whole log. The book and the Hugging Face model cards cite these notes, not the log. (Index written 2026-11-17.)

**How to use it.** Read in the order of the table below to follow the programme as it happened, or start from one of the reading paths at the end. Section numbers (§) are the log's, so a reader holding a citation to the log can find the note that covers it.

---

## The notes, in the order of the log

| protocol | topic | note |
| --- | --- | --- |
| §1–§4, §5.3, §6 | why the programme exists, the fixed configuration, the queue, the token budget, pre-registered bands, run-health rules | [Ladder_Protocol_Overview.md](Ladder_Protocol_Overview.md) |
| §3.1–§3.2, §5.1–§5.2, §5.5–§5.8 | the early arms (Gen 2): L = 2 attention and no-exchange arms, L = 1, the conservative-only control, the V_φ × Fock factorial, the L = 4 arm in brief | [Ladder_Early_Arms_Results.md](Ladder_Early_Arms_Results.md) |
| §3.3 (clipping blocks) | gradient clip thresholds, what clipping costs, the SPLM card audit, the C-series probes | [Clipping_and_the_C_Series.md](Clipping_and_the_C_Series.md) |
| §3.3 (L = 4 blocks), §3b | the L = 4 diagnosis (behind, not overfitting) and the D-series D1–D4 | [L4_Diagnosis_and_D_Series.md](L4_Diagnosis_and_D_Series.md) |
| §5.4, §5.17 | the matched GPT-2 baselines: cosine, W1, W2 and the 2B-token G2B; what is and is not matched | [Matched_GPT2_Baselines.md](Matched_GPT2_Baselines.md) |
| §5.9 | the SR series: exact damped flow (SR2), critical damping (SR3-crit), learned damping, variable-step training (SR4a) | [SR_Series_Settling_and_Refinement.md](SR_Series_Settling_and_Refinement.md) |
| §5.10–§5.11 | the abstract-gating runs G2, G3, G3′ and the CB series | [Abstract_Gating_Runs.md](Abstract_Gating_Runs.md) |
| §5.12, §5.14, §5.16, §5.23, §5.24, §5.26 | the small side programmes: CG8, DP, F0, RF, MT, BF | [Side_Probes_CG8_DP_F0_RF_MT_BF.md](Side_Probes_CG8_DP_F0_RF_MT_BF.md) |
| §5.13 | first against second order on OpenWebText and TinyStories, the memory-substitution calls, the well pairs | [First_vs_Second_Order_Series.md](First_vs_Second_Order_Series.md) |
| §5.13 (code) | the code set: SO-CODE against FO-b-CODE, the copy split, the two-well pair | [Code_Set_First_vs_Second_Order.md](Code_Set_First_vs_Second_Order.md) |
| §5.13 (code) | dead wells in the V_θ bank, the screens that keep them alive, their controls | [Code_Dead_Wells_and_Screens.md](Code_Dead_Wells_and_Screens.md) |
| §5.13 (code) | depth on code: L4-CODE, L8-CODE and the two-well eight-layer arms | [Code_Depth_Runs.md](Code_Depth_Runs.md) |
| §5.13 (code) | the per-layer flow on code, refinement attribution (TKF-A), the router-off cost (RO-COST), K2-CODE | [Code_Refinement_and_Router_Probes.md](Code_Refinement_and_Router_Probes.md) |
| §5.13 (code) | V_φ-force checkpointing, the memory switch that let the eight-layer arms train | [VPhi_Force_Checkpointing.md](VPhi_Force_Checkpointing.md) |
| §5.13 (2B) | the 2B-token runs: SR2-2B, the decay gain, well occupancy along training | [Two_Billion_Token_Runs.md](Two_Billion_Token_Runs.md) |
| §5.13 | the one-layer model SR2-L1 and the decay gain against the optimizer's noise | [SR2_L1_and_Decay_Noise.md](SR2_L1_and_Decay_Noise.md) |
| §5.13 | the well screen (K4-R6, K2-R6) and whether the wells can share a precision core | [Well_Screen_and_Precision.md](Well_Screen_and_Precision.md) |
| §5.13 | second order as memory: six probes of the mechanism | [Second_Order_Memory_Probes.md](Second_Order_Memory_Probes.md) |
| §5.13 | CPU inference: cost, the memory switches, the bound without the pair force | [CPU_Inference_Benchmark.md](CPU_Inference_Benchmark.md) |
| §5.15 | Poisson-mode wells: PM1, PM1-cap, PMX | [Poisson_Mode_Wells_PM1_PMX.md](Poisson_Mode_Wells_PM1_PMX.md) |
| §5.18 | what makes a model refinement-ready (RR) | [Refinement_Readiness_RR.md](Refinement_Readiness_RR.md) |
| §5.19 | refinement invariance: the two failure modes, FLOW-R, the scope caveat and RT1–RT3 | [Refinement_Invariance_Tests.md](Refinement_Invariance_Tests.md) |
| §5.20–§5.21 | what each coupling costs when removed (CD), and the trained-without factorial (VF) | [Coupling_Removal_Costs_CD.md](Coupling_Removal_Costs_CD.md) |
| §5.22 | spontaneous synchronization and semantic binding (SY) | [Synchronization_Tests_SY.md](Synchronization_Tests_SY.md) |
| §5.25 | the exchange field against the reverse channel (XR, parts A and B) | [Exchange_Field_Routes_XR.md](Exchange_Field_Routes_XR.md) |
| §7 | the publication audit of 2026-09-27 | [Publication_Audit_2026_09.md](Publication_Audit_2026_09.md) |

§5.13 holds most of the later work, so it is split across eleven notes by topic. A few labels are reused in the log for unrelated things (C1, M1, DP1–DP3); each note says which one it means.

## Reading paths

- **The programme in brief.** [Ladder_Protocol_Overview.md](Ladder_Protocol_Overview.md), then [Ladder_Early_Arms_Results.md](Ladder_Early_Arms_Results.md) and [Matched_GPT2_Baselines.md](Matched_GPT2_Baselines.md).
- **First against second order.** [First_vs_Second_Order_Series.md](First_vs_Second_Order_Series.md), [Code_Set_First_vs_Second_Order.md](Code_Set_First_vs_Second_Order.md), [SR_Series_Settling_and_Refinement.md](SR_Series_Settling_and_Refinement.md), then [Second_Order_Memory_Probes.md](Second_Order_Memory_Probes.md).
- **Refinement and the continuous flow.** [Refinement_Invariance_Tests.md](Refinement_Invariance_Tests.md), [Refinement_Readiness_RR.md](Refinement_Readiness_RR.md), [Code_Refinement_and_Router_Probes.md](Code_Refinement_and_Router_Probes.md).
- **The wells.** [Poisson_Mode_Wells_PM1_PMX.md](Poisson_Mode_Wells_PM1_PMX.md), [Code_Dead_Wells_and_Screens.md](Code_Dead_Wells_and_Screens.md), [Well_Screen_and_Precision.md](Well_Screen_and_Precision.md).
- **Depth and data.** [L4_Diagnosis_and_D_Series.md](L4_Diagnosis_and_D_Series.md), [Code_Depth_Runs.md](Code_Depth_Runs.md), [SR2_L1_and_Decay_Noise.md](SR2_L1_and_Decay_Noise.md), [Two_Billion_Token_Runs.md](Two_Billion_Token_Runs.md).
- **Routing and couplings.** [Abstract_Gating_Runs.md](Abstract_Gating_Runs.md), [Exchange_Field_Routes_XR.md](Exchange_Field_Routes_XR.md), [Coupling_Removal_Costs_CD.md](Coupling_Removal_Costs_CD.md), [Synchronization_Tests_SY.md](Synchronization_Tests_SY.md).
- **Cost.** [CPU_Inference_Benchmark.md](CPU_Inference_Benchmark.md), [VPhi_Force_Checkpointing.md](VPhi_Force_Checkpointing.md).

## Conventions

Every note opens with a "What this note is" paragraph and its date, keeps the log's section numbers and entry anchors next to its headings, reports numbers and verdicts as the log records them, and says plainly when the log records no result. New results go into a note in this folder, one topic per file, not into the log.
