# The Depth and Architecture Battery: SR2-NP-L8 and the Matched GPT-2

**Status (2026-11-27).** Calls drafted by the agent and **confirmed by the author on 2026-11-27**, as written. SR2-NP-L8 is training (due around midday 2026-11-28); the GPT-2 extraction path was built and checked on 2026-11-27 (§1.1), and the battery ran on W2 the same day: **GA1, GA2 and GA4 hit** ([GPT2_Template_Battery_Results.md](GPT2_Template_Battery_Results.md)); GA3 waits for SR2-NP-L8, against GPT-2's best averaged twin reading of 0.75. Programme context: [Template_Programme_Status_and_Plan.md](Template_Programme_Status_and_Plan.md); terminology: [README.md](README.md).

**In brief.**
- **Two comparisons.** Depth on a clean recipe: SR2-NP (two layers) against SR2-NP-L8 (eight layers, only the depth changed). Architecture at equal depth: SR2-NP-L8 against the matched GPT-2 (eight layers, the same width 384, the ladder's schedule, learning rate and 0.5B tokens; W2, settled 48.37 against SR2-NP's 58.51).
- **The battery.** The tests that have worked so far, unchanged: the verbatim depth premises (the geometry matcher on every state; paraphrases at 1% near misses), test 2, task A (the form read at "Therefore"), and B-logic with its control (binding read at "Therefore").
- **Calls.** Five on depth (DL1 to DL5), four on architecture (GA1 to GA4).

---

## 1. What is run

- **Models.** SR2-NP (the reference, already extracted); SR2-NP-L8 (its best checkpoint when training ends); the matched GPT-2 W2. W2 is `~/Downloads/semsimula_matched_gpt2_baseline_owt_wsd_lr0p0012` (downloaded by the author 2026-11-27; `checkpoints_wsd_lr0p0012/gpt2_baseline_best.pt`: step 32,500, peak learning rate 1.2e-3, best 48.14 at a context of 1,024, settled 48.37 by its run summary). Not to be confused with `~/Downloads/semsimula_matched_gpt2_baseline_owt`, which holds W1 (`checkpoints_wsd`, peak 6e-4, 48.67) and the cosine run (`checkpoints`, 49.76).
- **Extraction.** As for every earlier model: 48 tokens of ordinary text before each premise, whitening by 12,288 ordinary-text tokens per kind, every particle's state kept. GPT-2's states are its residual stream at the entry of each block and at the output; its embedding is the token vector plus the position vector, as in the ladder models. GPT-2 has no displacement of a conservative step and no well frame, so its readings are the states only.
- **Checks before reading.** For GPT-2, a dry run on a few premises compared with a direct forward pass (the states the hook records equal the model's own residual stream). For both, the gates of each test as before: token identity at chance, the exact ties of B-logic, the check premises.
- **The GPT-2 path, built 2026-11-27.** `template_depth_test.py` takes a matched GPT-2 folder (option `sub=` for the checkpoint subfolder; by default the only `checkpoints*` one) and records the residual stream with hooks: the entry of each of the eight blocks (h0 is the token vector plus the position vector) and the last block's output, before the final LayerNorm. Its kinds are h0 to h8 only; the scorers leave out the path and wells readings when a model has no displacements or well frames. `copy_split.gpt2_model` gained the subfolder option and returns the model as `run.model`. Every other step is unchanged: the same 48 tokens of ordinary text, the same whitening, the same items.
- **Checks, 2026-11-27.** On two premises the hooked states equal a forward pass computed block by block exactly (difference 0 on all nine states), and the final LayerNorm of the hooked output reproduces the model's logits exactly. A run of 64 premises on SR2-NP after the change matches the stored extraction to float16 rounding (largest difference 4.9e-4, one rounding step): the ladder path is unchanged. On the CPU the 64-premise run took 8 s in all on GPT-2, against 2 min 48 s on SR2-NP (which also replays its flow).
- **Reference values (SR2-NP, measured):** task A, best state ranks the right template first 0.621, twin AUROC 0.756; B-logic control, best state at "Therefore", twin reading averaged over the two orders 0.769, foil 0.911; paraphrases at 1% near misses, early state minus embedding $-0.019$; test 2, the early state's gain on the analogy levels over L0 and L1, $-0.019$.

## 2. Calls on depth (SR2-NP-L8 against SR2-NP)

| id | call | probability |
| --- | --- | ---: |
| DL1 | Task A: SR2-NP-L8's best state ranks the right template first on at least 0.67 of the valid premises in unseen wordings (SR2-NP 0.62). | 45% |
| DL2 | B-logic control: SR2-NP-L8's best state at "Therefore" has a twin reading, averaged over the two orders, of at least 0.82 (SR2-NP 0.77). | 45% |
| DL3 | Task A: within SR2-NP-L8 the rank peaks at a middle state (after layers 1 to 6), not after layer 0 or after the last layer. | 55% |
| DL4 | Paraphrases at 1% near misses: SR2-NP-L8's best layer state is above its own embedding by at least 0.05, interval above zero. | 50% |
| DL5 | Test 2: the early state's gain over the embedding on the analogy levels exceeds its gain on L0 and L1 (call S3 of test 2), interval above zero. | 30% |

**Reasoning.** DL1, DL2: G2's recipe gained sharply from two to four layers on the B-logic control (0.57 to 0.97 against the foil); SR2-NP already reaches 0.91, so the room is on the twin reading and on task A. DL3: L4's task A rank peaked after layer 2 and fell at the output. DL4: L4's states gained +0.16 over its embedding on paraphrases, SR2-NP's none. DL5: test 2's S3 missed on every model so far.

## 3. Calls on architecture (the matched GPT-2, and against SR2-NP-L8)

| id | call | probability |
| --- | --- | ---: |
| GA1 | Task A: GPT-2's best state ranks the right template first on at least 0.50 of valid premises in unseen wordings, with twin AUROC at least 0.70 (a transformer's states carry the form to "Therefore" too). | 75% |
| GA2 | B-logic control: GPT-2's best state at "Therefore" has a twin reading above 0.5 in both orders (interval above 0.5). | 75% |
| GA3 | B-logic control: SR2-NP-L8's best twin reading at "Therefore", averaged over the orders, is at least as high as GPT-2's. | 45% |
| GA4 | Verbatim premises: GPT-2's embedding read by the geometry matcher reaches at least 0.90 on unseen wordings (R1), as the ladder embeddings do. | 70% |

**Reasoning.** GA1, GA2: what worked so far needs only states that mix in context, which every transformer has. GA3 is the architecture question; GPT-2 is the better language model at these tokens (48.37 against 58.51), so an even call. GA4: on verbatim premises the matcher mostly reads repeated words, which any embedding gives.

**Why the architecture calls lean slightly toward GPT-2** (discussed with the author, 2026-11-27). The calls do not say GPT-2's templates will be better. GA1 and GA2 (75%) say only that GPT-2's states carry the form and the binding at all, above a low bar. GA3 (45%) is the comparison, and it leans slightly toward GPT-2 for three reasons:
- **Perplexity.** GPT-2 is the better language model at these tokens (settled 48.37 against SR2-NP's 58.51). Carrying which clause the context bound the signal to is part of predicting text well.
- **A direct route.** Attention lets the state at "Therefore" read the context sentence's tokens directly. The ladder models have no attention (their tags end in `noattn`): context reaches "Therefore" only through the particles' interactions and the context channels, step by step.
- **What has not helped so far.** The parts only the conservative architecture has (the well frames and the displacements of a step) have not been better readers than the states (finding 9 of the [status note](Template_Programme_Status_and_Plan.md)). So on reading alone, the architecture has no measured advantage to carry into the comparison.

What could favour SR2-NP-L8: the B-logic control already reaches 0.91 against the foil on two layers, and depth helped G2's recipe sharply. The architecture's claim does not rest on GA3 alone. It also rests on interpretability and on installing a template as a well, which a transformer cannot do.

**What the outcomes would mean.** If GA3 hits by a clear margin, the conservative dynamics carry binding better than attention at equal depth. If GPT-2 matches or leads, the templates are generic readers of contextual states, and the conservative architecture's claim rests on interpretability and on installing templates as wells (the open design question of [Implementing_Semantic_Templates_A_Path.md](../Implementing_Semantic_Templates_A_Path.md)).

**Not called:** the per-level readings of test 2 on GPT-2, the well frames and displacements of SR2-NP-L8, the flow gaps (reported as before), and firing.
