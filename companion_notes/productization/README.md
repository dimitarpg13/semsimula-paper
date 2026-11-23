# Productization: Index

**What this folder is.** Notes on making the conservative language models of the Semantic Simulation framework cheap to run: what their inference costs against the matched GPT-2, where that cost goes, and the plan for removing it. One topic per file. (Index written 2026-11-17.)

## The notes

| note | topic |
| --- | --- |
| [Conservative_Core_Productization_Plan.md](Conservative_Core_Productization_Plan.md) | the Gen 3 conservative core (SR2, K4-R6, K2-R6): the measured gap, the levers with and without retraining, the order of work |
| [Stage1_CPU_Levers.md](Stage1_CPU_Levers.md) | stage 1 of the plan, measured: where SR2's time goes by component, the levers that need no retraining (layer-1 pruning, `eigh`, folded fused attention) with their checks, the ratios against GPT-2, and the first reading for content-gated context channels |
| [Stage1b_Context_Channels_as_30_Heads.md](Stage1b_Context_Channels_as_30_Heads.md) | the context channels rewritten exactly as 30 heads of 64 so the tiled CPU attention kernel runs: problem statement, derivation, dispatch rule, cost and error analysis, measurements (1.67 times on the channels at 8,192 tokens) |
| [Stage2_Decoding.md](Stage2_Decoding.md) | stage 2, measured: the incremental decoder and its check, decoding speed and cache size against GPT-2 with a key–value cache, where a decode step's time goes, and `torch.compile` |
| [Training_Prefill_and_Decoding.md](Training_Prefill_and_Decoding.md) | a reader's guide to the three phases (training, prefill, decoding): what one layer does, what crosses positions, what limits each phase, a lever-by-phase map of every measured result, and where splitting the context channels into heads helps (long prompts), does not (decoding) and might (long-context training, float32 only); five figures |
| [Incremental_Decoder.md](Incremental_Decoder.md) | a reader's account of the incremental decoder: what a decoder is (not a decoder-only transformer), why the conservative core admits one (only the context channels cross positions), prefill and decode step, its role in text generation, what it requires of a model, cache and step cost against GPT-2; three figures |
| [Stage2b_Decode_Step_Levers.md](Stage2b_Decode_Step_Levers.md) | levers on the decode step, one at a time with paired timing: step 1 (the Taylor series of the damped-mode coefficients only where used, bit-identical) implemented and checked; steps 2 (single-token products as matrix products) and 3 (fusion) to be pre-registered |
| [Stage3_SR2_Without_Pair_Force.md](Stage3_SR2_Without_Pair_Force.md) | stage 3, first arm done: SR2 trained without the pair force (SR2-NP), settled 58.51 (+0.3% against SR2); calls, diagnostics, CPU speeds and the model card |
| [Stage3b_SR2_NP_Tied_Embeddings.md](Stage3b_SR2_NP_Tied_Embeddings.md) | stage 3, second arm, trained: SR2-NP with tied embeddings (SR2-NP-T, 57.1M parameters) settled 59.35 (+1.44%); TE1 missed, TE2 and TE3 hit; the untied head stays; the tie helps rare tokens and costs frequent ones |
| [Embedding_Head_Alignment_Test.md](Embedding_Head_Alignment_Test.md) | after SR2-NP-T: is the tie's cost a compromise inside the shared matrix or a burden on the dynamics? Problem statement, invariances, measures, the role test, results; for the most frequent types the shared matrix kept the input geometry, so the SR2 + tied 2×2 is informative |
| [Stage3d_SR2_Tied_With_Pair_Force.md](Stage3d_SR2_Tied_With_Pair_Force.md) | the fourth corner of pair force × tie: SR2 with tied embeddings (SR2-T, 57.3M parameters), build check passed; additive prediction 59.16; call ST1 (at or below 58.75) is the trigger for reopening the mean-field programme; trained 2026-11-22: settled 61.27 (+5.1%), all four calls miss; tied, the pair force costs 3.2% and SR2 trains into F3.1's regime (soft stiff modes, the pair force a third of the step, velocity reset +41%) |
| [Pair_Force_Recall_Split_FO_b_PMX_K16.md](Pair_Force_Recall_Split_FO_b_PMX_K16.md) | the PMX line, step 1 (CPU, 2026-11-22): does the pair force of FO-b-PMX-K16 fetch the earlier occurrence (which a mean field cannot reproduce) or do averaged work? Addressing fraction 4% (PMX-K16 9%, SR2 8%), router selection 1.3 times chance; about two thirds of the cost outside recall positions; a mean field is credible if content-addressed. Corrects the earlier SR2 layer-1 reading. §8 adds four code models: no addressing there either; two wells cut the pair force's cost tenfold |
| [Stage3e_FO_b_PMX_K16_Without_Pair_Force.md](Stage3e_FO_b_PMX_K16_Without_Pair_Force.md) | the PMX line, step 2: FO-b-PMX-K16 trained without the pair force (76,440,514 parameters), build check passed; calls FN1 to FN4 confirmed 2026-11-22; tolerance to be set; the companion PMX-K16 arm would complete the 2 × 2 |
| [Stage3c_Twin_PB_Precision_Bottleneck.md](Stage3c_Twin_PB_Precision_Bottleneck.md) | stage 3, pre-registration of item 4: twin PB (`B_proj` through 512, context-free diagonal precision) on the productization base; speed measured by a timing proxy (1.11 times at 512 tokens), untrained cost +42%; calls PB1–PB3 and the +2% tolerance confirmed 2026-11-22; code not yet written |
| [Fock_Inference_Productionization_Plan.md](Fock_Inference_Productionization_Plan.md) | its predecessor, for the Gen 2 Fock arm at eight layers (2026-09-18): the cost model, A100 measurements, the training-speed levers of the 2B-token runs |

## Related notes elsewhere

| note | topic |
| --- | --- |
| [CPU_Inference_Benchmark.md](../depth_ladder/CPU_Inference_Benchmark.md) | the measurements the plan starts from: speed, memory, the memory switches, the bound without the pair force |
| [Eliminating_the_Autograd_Dependency_in_VPhi.md](../Eliminating_the_Autograd_Dependency_in_VPhi.md) | the closed-form pair force |
| [Mean_Field_PARFLM_Programme.md](../Mean_Field_PARFLM_Programme.md) | the field forms of the pair force and the context channels, which would make the decoding state fixed |
| [SPLM_inference_efficiency_pre-registered_protocol.md](../SPLM_inference_efficiency_pre-registered_protocol.md) | the earlier inference-efficiency benchmark of the scalar-potential model |
