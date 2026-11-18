# Productization: Index

**What this folder is.** Notes on making the conservative language models of the Semantic Simulation framework cheap to run: what their inference costs against the matched GPT-2, where that cost goes, and the plan for removing it. One topic per file. (Index written 2026-11-17.)

## The notes

| note | topic |
| --- | --- |
| [Conservative_Core_Productization_Plan.md](Conservative_Core_Productization_Plan.md) | the Gen 3 conservative core (SR2, K4-R6, K2-R6): the measured gap, the levers with and without retraining, the order of work |

## Related notes elsewhere

| note | topic |
| --- | --- |
| [CPU_Inference_Benchmark.md](../depth_ladder/CPU_Inference_Benchmark.md) | the measurements the plan starts from: speed, memory, the memory switches, the bound without the pair force |
| [Fock_Inference_Productionization_Plan.md](../Fock_Inference_Productionization_Plan.md) | the predecessor, for the Gen 2 Fock arm: the cost model, A100 measurements, the training-speed levers of the 2B-token runs |
| [Eliminating_the_Autograd_Dependency_in_VPhi.md](../Eliminating_the_Autograd_Dependency_in_VPhi.md) | the closed-form pair force |
| [Mean_Field_PARFLM_Programme.md](../Mean_Field_PARFLM_Programme.md) | the field forms of the pair force and the context channels, which would make the decoding state fixed |
| [SPLM_inference_efficiency_pre-registered_protocol.md](../SPLM_inference_efficiency_pre-registered_protocol.md) | the earlier inference-efficiency benchmark of the scalar-potential model |
