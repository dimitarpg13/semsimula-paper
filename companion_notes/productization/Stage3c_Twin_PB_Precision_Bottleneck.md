# Stage 3c: Twin PB, a Bottleneck in the Bank's Precision Map (pre-registration)

**What this note is.** The pre-registration of item 4 of the remaining speed push in [Conservative_Core_Productization_Plan.md](Conservative_Core_Productization_Plan.md), §6: twin PB of [Well_Screen_and_Precision.md](../depth_ladder/Well_Screen_and_Precision.md), §5, moved onto the productization base. The well bank's largest map, `B_proj`, goes through a bottleneck of width 512, and its diagonal precision `a_proj` is replaced by one learned value per well and layer. Written 2026-11-22.

**Status.** Pre-registered. **The calls in §6 and the +2% tolerance were confirmed by the author on 2026-11-22 and are binding.** The code is not written (§5). The base is SR2-NP, untied: SR2-NP-T missed TE1 on 2026-11-22 (§4). The speed it buys and the cost of the bottleneck without training were measured on the CPU (§3).

---

## Contents

1. [Why this arm](#1-why-this-arm)
2. [What changes](#2-what-changes)
3. [What it buys and what it risks, measured without training](#3-what-it-buys-and-what-it-risks-measured-without-training)
4. [The base model](#4-the-base-model)
5. [What the code needs](#5-what-the-code-needs)
6. [Calls](#6-calls)
7. [Launch and readings](#7-launch-and-readings)
8. [Caveats](#8-caveats)

---

## 1. Why this arm

At 512 tokens the well bank is the largest part of SR2-NP's forward pass that the speed push has not touched. With every current lever on, the component timers put its parameter generation (`context_components`, the dense maps from the context channels to the wells' centres, precisions and weights) at 27% of the forward pass, and its quadratic, force and gradient terms at another 9%. The stiff-mode decomposition (23%) and the rest (35%) make up most of what remains (`cpu_bench_m_np_eager_timers.txt`). Compiling the core does not help at this length (plan item 3, [Stage2_Decoding.md](Stage2_Decoding.md), §5): the time sits in the large matrix products.

Most of those products are `B_proj`. For each token the bank reads the five context channels concatenated (1,920 numbers, with the active layer's depth code added) and produces, for each of its $K = 8$ wells, a centre $\mu_k$, a diagonal precision $a_k$, a low-rank precision factor $B_k$ of size $384 \times 4$ and a weight. The multiply-adds per token are:

| map | output width | multiply-adds per token |
| --- | ---: | ---: |
| `B_proj` (to the factors) | 12,288 | 23,592,960 |
| `mu_proj` (to the centres) | 3,072 | 5,898,240 |
| `a_proj` (to the diagonal precisions) | 3,072 | 5,898,240 |
| `w_proj` (to the weights) | 8 | 15,360 |

`B_proj` is two thirds of the arithmetic and 23.6M of SR2-NP's 76.4M parameters.

## 2. What changes

Two changes, as twin PB was proposed on 2026-11-13 (Well_Screen_and_Precision.md, §5):

- **A bottleneck in `B_proj`.** The dense map from 1,920 to 12,288 becomes a map from 1,920 to 512 without bias, followed by a map from 512 to 12,288 with bias. Parameters: 7,286,784 instead of 23,605,248.
- **A context-free diagonal precision (twin PA).** `a_proj` is replaced by one learned vector per well and layer: $2 \times 8 \times 384$ = 6,144 parameters instead of 5,901,312. Each well's diagonal precision then no longer depends on the context; its centre, its low-rank factor and its weight still do.

In both twins as proposed, `a_proj`'s weights start at zero. So at initialisation the diagonal precision is already context-free, equal to the softplus of its bias: the change starts from the same point and removes only what training would add.

Multiply-adds per token at layer 0 (all eight wells) fall from 35.4M to 13.2M, by 63%. At layer 1, pruned to its one working well at inference, they fall from 4.4M to 2.5M.

**The bank is shared by the two layers.** One `AnisotropicDepthConditionedGaussianVTheta` serves both layers; each layer adds its own depth code to the bank's input. So the bottleneck applies at both layers, and "at layer 0" in the plan's wording names where the cost lies, not a separable change. The diagonal precision can still differ per layer, as twin PA proposed.

**Parameters.**

| model | untied (SR2-NP base) | tied (SR2-NP-T base) |
| --- | ---: | ---: |
| base | 76,434,307 | 57,135,619 |
| with PB | 54,220,675 | 34,921,987 |

Tied and with PB, the model would be about the size of the matched GPT-2 (33.7M).

## 3. What it buys and what it risks, measured without training

**The speed.** The benchmark option `bank=pb512` (2026-11-22) replaces `B_proj` by its truncated SVD through 512 and `a_proj` by its bias. That is the arithmetic of the trained twin, not its predictions, so it times PB without training it. SR2-NP with every current lever on (`chunk=512 skipreg=1 prune=1:5 gram=eigh64 xi=fold64 denormal=flush`):

| | SR2-NP | timed as PB |
| --- | ---: | ---: |
| parameter generation at 512 tokens, s | 0.112 | 0.047 |
| one timed forward at 512 tokens, s | 0.414 | 0.305 |
| parameter generation at 8,192 tokens, s | 1.390 | 0.664 |
| one timed forward at 8,192 tokens, s | 7.49 | 6.25 |

Parameter generation takes 58% less time at 512 tokens and 52% less at 8,192, against 61% fewer multiply-adds across the two layers. The other components are unchanged, as they should be: the factors $B_k$ still have rank 4 and the decomposition still sees them all. From the components alone, the forward pass would gain 0.065 s of about 0.40 s at 512 tokens (1.19 times) and 0.73 s of about 6.5 s at 8,192 (1.12 times). Measured over whole forward passes, three interleaved pairs at 512 tokens with nine repeats each give 1,411 tokens per second for SR2-NP and 1,568 timed as PB, on average: **1.11 times** (1.17, 1.13 and 1.04 per pair; 1.13 times on the fastest repeat of each run; `cpu_bench_n_np_*`). At 2,048 and 8,192 tokens the whole-forward runs of this session were too noisy to separate a gain of this size (`cpu_bench_m_np_*`).

**The risk.** Truncating the trained `B_proj` to rank 512 without retraining costs SR2-NP +41.8% (+38.3% when the truncation is fitted to the inputs the bank sees). Rank 1,024 costs +6.7%. These match SR2's +41.2% and +7.0% (`projection_svd_np_owt.txt` against `projection_svd_so_owt.txt` in `results/precision_ablation/`). Truncating `a_proj` costs almost nothing at any rank (+0.35% at rank 256). Replacing it by its weighted mean per well and layer, the context-free precision of twin PA, costs +1.49% (SR2: +1.2%). Fixing $B_k$ instead costs +1,234%, so the factors' dependence on the context is as essential in SR2-NP as in SR2 (`precision_ablation_np_owt.txt`): the bottleneck must keep it, which a width-512 map does and a fixed factor does not. The timing proxy with both changes reads perplexity 82.72 against 58.41 on its 16 windows (+41.6%).

These are upper bounds: a model trained with the bottleneck adapts the rest of the bank to it. The question the arm answers is how much of the +42% training recovers. Width 512 is where the saving is large and the untrained cost is real; at 1,024 the untrained cost is small, but it saves only about 9M parameters and 38% of `B_proj`'s multiply-adds.

## 4. The base model

The decision rule of SR2-NP-T ([Stage3b_SR2_NP_Tied_Embeddings.md](Stage3b_SR2_NP_Tied_Embeddings.md), §3) says that if TE1 hits, later productization arms run tied. So the base is:

- **SR2-NP-T** (tied input and output embeddings), if it settles at or below 59.10;
- **SR2-NP** (untied), otherwise.

**Resolved 2026-11-22: the base is SR2-NP.** SR2-NP-T settled at 59.35 (+1.44%) and missed TE1 (Stage3b, §5). So $P_0 = 58.51$: PB1 means 59.68 or less, PB2 61.44 or less. PB has 54,220,675 parameters.

PB is SR2-NP's Cell 0 plus the two switches of §5, 32,500 steps on OpenWebText, seed 0, paired with SR2-NP. The code of §5 must be written and checked first.

## 5. What the code needs

None of this is written yet. Checked 2026-11-22 against the working tree:

1. **The bank** (`parf/model_aniso_gaussian_vtheta.py`, `AnisotropicMixtureGaussianVTheta`): a bottleneck width, with `B_proj` built as two maps when it is set. Initialise the pair so that the product's entries have the same spread as SR2-NP's single map at the start (normal, standard deviation 0.01), for example the first map at $1/\sqrt{1920}$ and the second at $0.01 \cdot \sqrt{1920/512}$.
2. **The diagonal precision per well and layer.** The bank does not know the active layer; the depth-conditioned wrapper does (`set_active_layer`). Store the per-layer values in the wrapper, or give the bank the active layer, and keep the softplus and the floor of $10^{-4}$ of `_components`.
3. **No weight decay on it.** The optimizer decays every tensor of two or more dimensions (Cell 6, `split_decay_params`). SR2-NP's diagonal precision lives in `a_proj`'s bias, which is one-dimensional and not decayed. A parameter of shape (2, 3,072) would be decayed; store it one-dimensional, or exclude it by name.
4. **The `B_proj` spectral diagnostic** (Cell 6, `bproj_sig`) reads `B_proj.weight`. With two maps, that attribute is missing, and the `try` block would drop the diagnostic silently. It needs the spectral norm of the product (or a bound from the two factors).
5. **Every CPU tool that reads `B_proj.weight`**: `vtheta_projection_svd.py`, `vtheta_precision_ablation.py`, `vtheta_shared_subspace_fit.py`, `vtheta_well_probe.py`, the layer-1 pruning of `cpu_throughput_benchmark.py`, and the test `parf/test_cfc_baoab.py`. Each either handles the two maps or refuses the model with a clear message.
6. **Cell 0 switches and the tag.** Two switches (the bottleneck width, default 0; the context-free precision, default off), each adding a tag component only when set (for example `bb512` and `afree`), so that every existing tag stays the same. `debug/copy_split.py`'s `tag_repl` must read them back.
7. **The gradient clip.** Clip groups are keyed by the first part of a parameter's name; the new parameters sit under `V_theta`, as the old ones did, so the groups do not change. Cell 6's header must be checked at launch, as for SR2-NP-T.
8. **A build check** on the CPU, through the notebook's own Cells 0 to 5: the tag, and the parameter count of §2 (54,220,675 untied or 34,921,987 tied).

## 6. Calls

**Proposed and confirmed by the author on 2026-11-22, at the probabilities below; binding.** PB1 and PB2 carry over the 2026-11-13 proposal, which was set against SR2. Here they are relative to the base of §4, whose settled perplexity is $P_0$. Settled perplexity is defined as for every OpenWebText arm (the mean of the last three evaluations once the trend has flattened).

| id | call | probability |
| --- | --- | ---: |
| PB1 | PB settles within +2% of the base | 35% |
| PB2 | PB settles within +5% of the base | 65% |
| PB3 | the rarest frequency quintile's mean loss in PB is within 0.25 nats of the base's (the bank's capacity is not what the rare tokens rely on) | 60% |

**Decision rule** (tolerance +2%, PB1 itself, confirmed by the author on 2026-11-22).

- **Within tolerance:** the deployment configuration takes the bottleneck. The model cards and the book quote the new parameter count, and the CPU benchmark is rerun on the trained twin.
- **Within +5% but outside the tolerance:** the bottleneck is a size–speed option, recorded with its cost, not the default.
- **Above +5%:** the dense `B_proj` stays, now as a measured requirement. Twin PA alone (the context-free diagonal precision: about 5.9M parameters and 5.9M multiply-adds per token saved, +1.5% untrained) becomes the follow-up.

## 7. Launch and readings

Launch as for the base, with the two switches set. Check that Cell 0 prints the new tag, Cell 5 the parameter count of §2, and Cell 6 the clip groups unchanged. Expected time: a little under the base's, about 6 hours on an A100 (the bottleneck shortens one of the larger products in every step).

After the run, on the best checkpoint:

- **On the GPU,** the diagnostic cells of the base: 6b-7, 6b-9, 6b-12, 6b-13, 7 and 8.
- **On the CPU,** one job at a time:
  - the frequency quintiles against the base (`debug/frequency_quintile_loss.py`, for PB3);
  - the causality check;
  - the layer-1 well screen (does the bottleneck move the bank's one working well?);
  - the CPU benchmark of the trained twin against the base, interleaved, replacing the proxy figures of §3.

## 8. Caveats

- **One seed per arm.** Seed spread at this size is of the order of 0.3%, well inside PB1's 2%.
- **PA and PB are confounded.** The arm changes both maps at once. The untrained costs (+1.5% for a context-free `a`, +42% for `B_proj` through 512) say which one carries the risk, but a trained PA was never run.
- **The tied, GPT-2-sized configuration stays unmeasured.** SR2-NP-T missed TE1, so PB runs untied. Tied and with PB (34.9M parameters) would combine two costs, +1.44% measured for the tie and PB's own.
- **The timing proxy is the trained twin's arithmetic** only if the trained twin keeps the same shapes. The two-map `B_proj` and the context-free precision do keep them; the proxy's whole-forward figures are as noisy as the session (single repeats ranged from 0.30 to 0.58 s at 512 tokens).
