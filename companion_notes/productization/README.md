# Productization: Index

**What this folder is.** Notes on making the conservative language models of the Semantic Simulation framework cheap to run: what their inference costs against the matched GPT-2, where that cost goes, and the plan for removing it. One topic per file. (Index written 2026-11-17.)

## The notes

| note | topic |
| --- | --- |
| [Conservative_Core_Productization_Plan.md](Conservative_Core_Productization_Plan.md) | the Gen 3 conservative core (SR2, K4-R6, K2-R6): the measured gap, the levers with and without retraining, the order of work |
| [Stage1_CPU_Levers.md](Stage1_CPU_Levers.md) | stage 1 of the plan, measured: where SR2's time goes by component, the levers that need no retraining (layer-1 pruning, `eigh`, folded fused attention) with their checks, the ratios against GPT-2, and the first reading for content-gated context channels |
| [Stage1b_Context_Channels_as_30_Heads.md](Stage1b_Context_Channels_as_30_Heads.md) | the context channels rewritten exactly as 30 heads of 64 so the tiled CPU attention kernel runs: problem statement, derivation, dispatch rule, cost and error analysis, measurements (1.67 times on the channels at 8,192 tokens) |
| [Stage2_Decoding.md](Stage2_Decoding.md) | stage 2, measured: the incremental decoder and its check, decoding speed and cache size against GPT-2 with a key–value cache, where a decode step's time goes, and `torch.compile` |
| [Stage3_SR2_Without_Pair_Force.md](Stage3_SR2_Without_Pair_Force.md) | stage 3, first arm done: SR2 trained without the pair force (SR2-NP), settled 58.51 (+0.3% against SR2); calls, diagnostics, CPU speeds and the model card |
| [Stage3b_SR2_NP_Tied_Embeddings.md](Stage3b_SR2_NP_Tied_Embeddings.md) | stage 3, second arm set up: SR2-NP with tied embeddings (SR2-NP-T, 57.1M parameters); is the separate output head needed? calls TE1–TE3 confirmed |
| [Stage3c_Twin_PB_Precision_Bottleneck.md](Stage3c_Twin_PB_Precision_Bottleneck.md) | stage 3, pre-registration of item 4: twin PB (`B_proj` through 512, context-free diagonal precision) on the productization base; speed measured by a timing proxy (1.11 times at 512 tokens), untrained cost +42%; calls PB1–PB3 and the +2% tolerance confirmed 2026-11-22; code not yet written |
| [Fock_Inference_Productionization_Plan.md](Fock_Inference_Productionization_Plan.md) | its predecessor, for the Gen 2 Fock arm at eight layers (2026-09-18): the cost model, A100 measurements, the training-speed levers of the 2B-token runs |

## Related notes elsewhere

| note | topic |
| --- | --- |
| [CPU_Inference_Benchmark.md](../depth_ladder/CPU_Inference_Benchmark.md) | the measurements the plan starts from: speed, memory, the memory switches, the bound without the pair force |
| [Eliminating_the_Autograd_Dependency_in_VPhi.md](../Eliminating_the_Autograd_Dependency_in_VPhi.md) | the closed-form pair force |
| [Mean_Field_PARFLM_Programme.md](../Mean_Field_PARFLM_Programme.md) | the field forms of the pair force and the context channels, which would make the decoding state fixed |
| [SPLM_inference_efficiency_pre-registered_protocol.md](../SPLM_inference_efficiency_pre-registered_protocol.md) | the earlier inference-efficiency benchmark of the scalar-potential model |
