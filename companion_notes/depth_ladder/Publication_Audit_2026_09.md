# §7 Publication readiness of the first five ladder arms: the 2026-09-27 audit

**What this note is.** A readable summary of §7 of the depth-ladder [protocol](https://github.com/dimitarpg13/semsimula-paper/blob/0e3e3f22199a5595d86c15c39a27319746b55ae2/companion_notes/Depth_Ladder_and_Matched_Baseline_Protocol.md), "Publication readiness of the five completed arms — audit 2026-09-27", written 2026-11-17. It covers all of §7: what the audit found complete (§7.1), the four gaps it found (§7.2), what closing the first two cost (§7.2a), the upload (§7.3a), the finding that the "Verlet-era" label is too narrow (§7.3b), and the upload plan written before the upload (§7.3). The sibling note [Ladder_Protocol_Overview.md](Ladder_Protocol_Overview.md) gives §7 in one short section; this note gives it in full. Every number, verdict and date is copied from the protocol, which stays the record of truth.

**Status in one line.** All four gaps were dealt with on 2026-09-27: gaps 1 and 2 by one probe session with no retraining, gaps 3 and 4 on the model cards; the five repos and their collection were public the same day. The §7.3b wording issue was left unreviewed, with no text changed.

**Contents**

1. [Scope of the audit](#scope) (protocol §7)
2. [What is complete](#71-what-is-complete) (protocol §7.1)
3. [The four gaps](#72-the-four-gaps) (protocol §7.2)
4. [What closing gaps 1 and 2 actually costs](#72a-what-closing-gaps-1-and-2-actually-costs) (protocol §7.2a)
5. [Uploaded](#73a-uploaded-2026-09-27) (protocol §7.3a)
6. [The "Verlet-era" label is too narrow](#73b-the-verlet-era-label-is-too-narrow-2026-09-27) (protocol §7.3b)
7. [Upload shape, as planned before the upload](#73-upload-shape-as-planned-before-the-upload) (protocol §7.3)
8. [Points to read with care](#points-to-read-with-care)

**Terms used throughout.** An **arm** is one training run with one configuration. PPL is validation perplexity; lower is better. **Settled** PPL is the mean of the last three evaluations. $L$ is the number of layers. The five arms are the matched GPT-2 baseline and four $L = 2$ Fock arms that differ in their force-relaxation mechanism (`none`, `attention`, `attention_potential`) or in whether the reverse channel is on (`none` with and without it; "no-RC" means without). A **probe** is an evaluation-only notebook cell (named 6b-7, 6b-9 and so on) that reads a trained checkpoint and prints a diagnostic. The **variant tag** is the string built from the Cell 0 settings that names an arm's Drive folder. A **stripped checkpoint** is one saved without optimizer state.

---

<a name="scope"></a>
## 7. Scope of the audit

The audit checked the five local experiment folders that hold the $L = 2$ ladder and the matched baseline, against what a HuggingFace repo per arm needs. The folder names in `hf_model_cards/_ladder/ladder.json` (its `gdrive` field) match the folders on disk exactly, for all five. The two queued arms (runs 10 and 11, protocol §3.2) had no data yet.

## 7.1 What is complete

**Every arm's headline number can be recomputed from what it ships.** The mean of the last three `val_ppl` values in each `results/training_log.jsonl` reproduces the settled value in `ladder.json` to the second decimal, for all five arms.

| arm | best step / PPL in ckpt | last three evals | settled | ladder.json |
| --- | --- | --- | ---: | ---: |
| `gpt2-matched` | 32,500 / 49.76 | 49.87, 49.80, 49.76 | 49.81 | 49.81 |
| `attention` | 31,000 / 61.49 | 63.52, 63.58, 63.42 | 63.51 | 63.51 |
| `attention_potential` | 31,000 / 79.14 | 80.74, 80.46, 81.50 | 80.90 | 80.90 |
| `none` | 31,500 / 66.56 | 66.56, 66.75, 67.63 | 66.98 | 66.98 |
| `none` no-RC | 31,000 / 85.90 | 87.53, 87.44, 88.82 | 87.93 | 87.93 |

**Also checked and found correct:**

- **Two stripped checkpoints per arm:** `_best.published.pt` and `_step500_best.published.pt`. This is the pairing agreed earlier: the endpoint, plus the earliest usable point so a reader can see where the run started. None holds optimizer state. The Fock checkpoints are 293–295 MB; the GPT-2 one is 137 MB.
- **The full configuration inside every Fock checkpoint:** 112–113 fields under `model_cfg`, plus 25 under `train_cfg`. The knobs that define each arm read correctly.
  - `force_relaxation` is `none`, `attention` or `attention_potential` on the three arms that vary it.
  - `reverse_channel` (`True` against `False`) is the *only* model-config difference between the `none` and no-RC folders, apart from a `register_salience_init` default and the logfreq path.
  - All four Fock arms share `d=384`, `L=2`, `xi_channels=5`, `top_k=16`, `integrator='baoab_cfc_lowrank'`, learning rate 1.2e-3, and batch 16 with gradient accumulation 2.
- **Complete training logs:** 721 or 722 JSONL rows over steps 50–32,500, with 65–66 evaluations, carrying `relax_lambda` and `relax_share` beside PPL. The GPT-2 log has 227 rows over steps 200–32,500, at both context lengths.
- **Console logs** for all four Fock arms. GPT-2's console log (`matched_gpt2_training_output.txt`) had not been filed with its arm; the audit filed it into that arm's `results/`.
- **Probe logs for three arms:** the 6b-7, 6b-9 and 6b-12 outputs sit in the `attention`, `attention_potential` and no-RC folders.

**The parameter counts in `ladder.json` are right and should not be "corrected".** Summing `numel` over a `model_state_dict` gives more than the card's count by a fixed amount, because state dicts also carry buffers: +50,257 (`logfreq_surprisal`) plus three scalars on every Fock arm, and +2,097,152 on GPT-2 (eight causal masks of 262,144 elements, one per block). Net of buffers, the counts match.

## 7.2 The four gaps

**Closed 2026-09-27.** Gaps 1 and 2 were filled in one session of the shape §7.2a describes.

- Five logs now sit in the `none` arm's `results/` (6b-7, 6b-9, 6b-10, 6b-11, 6b-12), and the two figures in its `checkpoints/`.
- Every probe header reads `step 31500  ppl 66.56  missing 0  unexpected 0`.
- 6b-7 and 6b-9 reproduce their recorded numbers exactly: Gate 1 +50.5%, Gate 3 435.61 at $N = 8$, $R(\mathrm{geo}) = 1.0855$.
- 6b-10 and 6b-11 gave the better readings §7.2a predicted. One recorded verdict changed as a result: E3's coherence metric moves from "invalid at $L = 2$" to **refuted** (§6.8 of [Geodesic_Experiments_with_CfC_BAOAB.md](../Geodesic_Experiments_with_CfC_BAOAB.md)).
- 6b-12 was a first run on this arm, and it **met its pre-registered band on all three criteria**.

Gaps 3 and 4 stand in the data; they were handled on the cards (§7.3a). The protocol keeps the gap list below as the audit first wrote it.

| gap | what the audit found | how it ended |
| --- | --- | --- |
| 1 | The `none` arm ships no probe logs | Closed 2026-09-27 (one session, with gap 2) |
| 2 | The E3 (6b-10) and E5 (6b-11) outputs are not filed anywhere | Closed 2026-09-27 (same session) |
| 3 | `gpt2_baseline_summary.json` carries a stale cross-reference | Handled on the GPT-2 card, plus a note file beside it |
| 4 | The second checkpoint is not the same kind on every arm | Handled on the GPT-2 card |

**Gap 1: the `none` arm ships no probe logs.** 6b-7 and 6b-9 were run on it. Their results ($R(\mathrm{geo}) = 1.09$, gates $4.32\times$ and $6.35\times$) are recorded in [Geodesic_Experiments_with_CfC_BAOAB.md](../Geodesic_Experiments_with_CfC_BAOAB.md) §4.7 and [Composing_Single_Layer_Inferences_Flow_or_Maps.md](../Composing_Single_Layer_Inferences_Flow_or_Maps.md) §8.2. But the console output was never filed, and 6b-12 (F1) had never been run on this arm. So the card can cite the numbers, yet the arm with the most prose written about it was the one whose probe output could not be re-read. The fix was not 20 minutes, as first thought, but one session of 45 to 75 minutes that also closes gap 2.

**Gap 2: the E3 (6b-10) and E5 (6b-11) outputs are not filed anywhere.** Both ran, and both are written up (§6.8 and §11.6 of the geodesic notes). Both ran on this same `none` arm, which is why gaps 1 and 2 are one job. Neither cell is any longer the cell that produced those write-ups.

**Gap 3: a stale cross-reference in `gpt2_baseline_summary.json`.**

- Its four `fock_*` fields and `ratio_fock_to_gpt2_settled = 1.6378` do not belong to any ladder arm.
- The file was written 2026-09-21, before any $L = 2$ arm existed. Those numbers come from the pre-ladder $L = 8$ run (`fock_cfc_baoab_joint_vtheta_qknorm_annealed_after_28500_...`, learning rate 3e-4, read at step 32,500 of a 100k-step schedule). That run's own last-three mean is 81.49.
- The ratio is close to the `attention_potential` arm's 1.624 by coincidence. That makes it a trap, not a harmless leftover.
- A file `NOTE_summary_json_fock_fields.md` now sits beside the summary, relabelling the fields as a matched-depth side note.
- Worth keeping as that side note. At matched depth $L = 8$ the ratio was 1.638, against 1.275 for the $L = 2$ `attention` arm. Depth 8 at learning rate 3e-4 was further from GPT-2 than depth 2 at 1.2e-3, which is why the ladder was rebuilt at $L = 2$.

**Gap 4: the second checkpoint is not the same kind on every arm.**

- The four Fock arms ship step 500, at PPL 478–491. That is essentially untrained: a reference for the initialisation, not a point on the learning curve.
- The GPT-2 folder has no step-500 checkpoint. Its second file is step 25,000, at PPL 52.87, already within 6% of its endpoint.
- So "the early checkpoint" means two different things across the collection. Anyone comparing first checkpoints arm by arm would be comparing an initialisation with a nearly converged model.
- The two remedies offered: state the asymmetry on the GPT-2 card, or drop its step-25,000 file and ship the baseline as endpoint-only. (The card route was taken, §7.3a.)

**Also quarantined.** In the no-RC folder's `checkpoints/`, the superseded `f1_forcing_per_token_WRONG.png` (from F1's first, tautological run) was moved to `results/` under a `_DO_NOT_PUBLISH_` prefix. The corrected plot was renamed `F1_6b12_per_token_deflection.png` to match its log.

## 7.2a What closing gaps 1 and 2 actually costs

This section was the plan for the session that closed gaps 1 and 2. All four missing logs belong to one configuration, the `none` arm, so one Colab session closes them.

**Nothing is retrained.** Every probe finds its own checkpoint as `CKPT_DIR / f'{CKPT_PREFIX}_best.pt'`, loads it into the built model, and restores the previous weights on exit. Cell 6 (training) is not needed, and Cell 2 is read-only. The run order:

    Cell 0 (config for this arm)  ->  1  ->  1b  ->  2  ->  3  ->  4  ->  5
    then 6b-9, 6b-7, 6b-12, 6b-11, 6b-10

**Why that probe order.** 6b-9 is the cheapest. It checks the load, the tag and the gate readings before anything expensive runs. 6b-10 is last because it is the only one that also needs the matched GPT-2 checkpoint on Drive. 6b-11 and 6b-12 write figures into `CKPT_DIR`, so those are collected with the logs.

**Cells skipped.** 1c and 1d do nothing here (no later cell reads what they set). 5b is skipped (its from-scratch guard protects a training run), and so is 6.

**Cell 0 as committed already is this arm.** Rebuilding `_variant_tag` from the committed values reproduces the Drive folder's tag character for character. So the job is to confirm that the Colab copy has not drifted from the `attention_potential` run and the two gate and init-scale probes that followed it. The values to check, most recently disturbed first:

| knob | value | tag effect if wrong |
| --- | --- | --- |
| `LADDER_MECHANISM` | `'none'` | `noattn` becomes `attnpot`/`attn` — was last set to `'attention_potential'` |
| `RELAX_INIT_SCALE` | `0.02` | adds `ris0p055` — the init-scale probe set this |
| `RELAX_GATE` | `'scalar'` | silent on this arm: the `zro` token is gated on `FORCE_RELAXATION != 'none'`, so a stale `'zero_readout'` neither shows in the tag nor changes the model here. Harmless, but the tag cannot warn you |
| `REVERSE_CHANNEL` | `True` | adds `norc` and resolves the other folder |
| `LADDER_L` | `2` | `L2probe` |
| `LADDER_T` / `LADDER_DT` | `8.0` / `None` | dt = 4, giving `idt4` |
| `LADDER_LR` | `1.2e-3` | `lr0p0012` |
| `LADDER_STEPS` | `32_500` | not in the tag, but sets the WSD windows |
| `WSD_STABLE_FRAC` | `0.60` | adds `sf...` |
| `LADDER_SALIENCE_INIT` | `1.0` | adds `sal...` |
| `LANGEVIN_T` | `0.0` | adds `T...` |

`PROBE_MAX_STEPS` and `ANNEAL_PROBE` do not matter, since nothing trains. The two lines to read back before any probe runs:

    [variant] tag=xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn
    CKPT_DIR    = /content/drive/MyDrive/semsimula_fock_cfc_baoab_owt_<that same tag>/checkpoints

**This is the only thing that can go wrong silently.** The probes find the checkpoint through the tag and load it with `strict=False`. A drifted Cell 0 therefore finds either nothing or the wrong file, and reports a partial load instead of failing. The guard is already printed: every probe header must read `step 31500  ppl 66.56  missing 0  unexpected 0`.

**No edits are needed inside any probe.** Each defaults to `CKPT_DIR / f'{CKPT_PREFIX}_best.pt'`. 6b-12's `R12_PRIMARY` picks `conservative + LN` automatically when a reverse channel exists, which is correct here.

| cell | on this arm | work | estimate |
| --- | --- | --- | ---: |
| setup, 0–5 | | Drive mount, OWT cache load, model build | 10–15 min |
| 6b-7 flow or maps | re-file | ~14 evals x 12 x 4 x 512 | 3–7 min |
| 6b-9 E1 | re-file | 3 x 4 x 512, three arms | 1–2 min |
| 6b-10 E3 | improved run | builds GPT-2 too; fp32; 3 perturbation sizes | 5–15 min |
| 6b-11 E5 | improved run | 12 lambda points x 8 x 4 x 512 | 10–20 min |
| 6b-12 F1 | first run | 8 x 4 x 512, three arms | 2–5 min |

In all, one session of roughly 45 to 75 minutes of GPU, with no training. 6b-10 also needs the matched GPT-2 checkpoint on Drive at the path hard-coded in `R10_GPT2_CKPT`.

**Two of the four are not re-filings, because the cells changed after those runs.**

- **6b-10 (E3)** now uses `R10_EPS = (1e-2, 3e-2, 1e-1)`. The run written up in §6.8 of the geodesic notes reports an $\varepsilon$ = 1e-3 row that sat on the TF32 rounding floor and was discarded, and it has no 3e-2 or 1e-1 rows. A re-run replaces that table's (c) block with three usable rows.
- **6b-11 (E5)** now prints both the maximum and the minimum of the per-layer change in $V_\varphi$ across the sweep. The run in §11.6 was read off a version that checked only increases. That is the bug that let layer 1's ninefold *fall* go unmentioned. A re-run puts both directions in the log.

So a short edit to §6.8 and §11.6 was budgeted for afterwards. Neither was expected to move a conclusion; both make the recorded evidence match the current cell. (As the closure box in §7.2 records, one verdict did change: E3's coherence metric became **refuted**.)

**6b-12 on this arm was a first run, so its band was pre-registered before launch.** E1 puts this arm's average deflection at $R(\mathrm{geo}) = 1.09$, with the reverse channel carrying about 90% of it. The forced arm already measured (`attention`) read UNIFORM at 93.8% of tokens above 0.75. The band recorded: primary arm `conservative + LN`, layer 1, **above 80% of tokens over 0.75, under 5% below 0.25, bimodality under 0.6**. That is uniform forcing, the mirror image of the no-RC arm's 100% below 0.25. It met the band on all three criteria (§7.2).

**A wider point the same work exposes.** The F1 logs already held for `attention` and `attention_potential` came from the two-arm version of 6b-12. That version predates the third arm, and the partial correlations that control for step size. On the attention arm, the raw correlation between deflection and step size is $-0.936$. So every other correlate in that log is read through a confound the current cell removes. Making all four arms comparable under one cell version costs two more sessions of the same shape, about 25 minutes each, because Cell 0 defines the architecture and each arm needs its own build. (The protocol does not record that these two sessions were run.)

## 7.3a Uploaded (2026-09-27)

Done and public: five repos plus the collection.

| repo (under `dimitarpg13/`) | files | contents |
| --- | ---: | --- |
| `semsimula-ladder-owt-d384-l2-gpt2-matched` | 9 | 2 checkpoints, 4 results, 1 notebook |
| `semsimula-ladder-owt-d384-l2-attention` | 26 | 2 checkpoints, 6 results, 16 code |
| `semsimula-ladder-owt-d384-l2-attention-potential` | 26 | as above |
| `semsimula-ladder-owt-d384-l2-none` | 28 | 2 checkpoints, 8 results, 16 code |
| `semsimula-ladder-owt-d384-l2-none-norc` | 27 | 2 checkpoints, 7 results, 16 code |

**The collection** is *Semantic Simulation — CfC+BAOAB Mechanism Ladder*, with six items: the Verlet-instability repo first, as the prologue, then the five arms. Both URL forms resolve. The book uses the form without the suffix, to match the existing family link: `https://huggingface.co/collections/dimitarpg13/semantic-simulation-cfcbaoab-mechanism-ladder`.

**Which code was shipped, and why.**

- The precedent of the older SPLM model family is that a repo carries the model source it was trained with, so nobody has to guess which revision the weights belong to. These repos do that.
- They also add the notebook, because an arm here is defined by Cell 0 knobs, not by a script.
- The Python files are the *transitive import closure* of what the notebook imports: every module it imports, and every module those import. The set was computed by walking the syntax trees, not picked by hand. It is 12 modules plus `data_module.py`, `parf/__init__.py` and `parf/test_cfc_baoab.py`, which Cell 4 runs as a gate.
- The layout under `code/` mirrors the repository, so the notebook's own `sys.path` lines work unchanged.
- The GPT-2 arm ships one notebook and nothing else. It defines its model in Cell 3 and imports nothing from the repository.

**Three Hub limits, found the hard way** and recorded so the next collection does not rediscover them. A collection description is capped at 150 characters, and an item note at 500. The collection slug gets a 24-hex suffix appended to a slugified title (`CfC+BAOAB` becomes `cfcbaoab`; the em dash is dropped). The URL without the suffix redirects. So guessing the slug in advance would have half-worked, and been fragile.

**Verified after upload, not just reported by the uploader.**

- Every repo is public, with the expected file count.
- No `*.published.*` name and no `_DO_NOT_PUBLISH_*` file leaked.
- The flagship checkpoint was downloaded back from the Hub, loaded, and confirmed at step 31,500, PPL 66.56, 98 tensors, no optimizer state.

**Gaps 3 and 4 were handled on the cards, not in the data.** The GPT-2 card carries its own caveat block naming both the stale `fock_*` fields and the step-25,000 asymmetry. `NOTE_summary_json_fock_fields.md` ships beside the summary it corrects.

## 7.3b The "Verlet-era" label is too narrow (2026-09-27)

Checking whether the SPLM Model Family collection should be marked as Verlet-based showed that it is not. The integrator each of its fifteen items declares:

| integrator | items |
| --- | ---: |
| `semi_implicit_euler` (config), plus the three gamma sweeps, whose READMEs document a damped Euler step `v += dt*f/m; v /= (1 + dt*gamma); h += dt*v` | 13 |
| `first_order_gradient_flow` (the Fock-G1 ablation; no velocity at all) | 1 |
| `verlet` (the instability repo, which is also item 0 of the ladder collection) | 1 |

- **Fourteen of the fifteen published models in that family are not Verlet.** What the family really shares is being *pre-CfC*: explicit integrators, one force evaluation per step, chosen before the closed-form propagator existed.
- The family's description now says so. Its title is unchanged, because the title generates the slug and paper v6 cites the slug without the suffix.

**This reaches beyond the collection.** The programme had been saying "Verlet-era" for the whole pre-CfC period: in the protocol, and in the scoping remarks added to paper v6 §8 and §27 (`rem:riemannian-verlet-scope`, `rem:gamma-eff-scope`).

- On the published evidence the phrase is too narrow. The geodesic-residual work those remarks scope was done on damped-Euler runs. Only the instability that ended the era was Verlet.
- The remarks are not *wrong*: a damped explicit step has the same defect the calibration exposed. But they name the wrong integrator.
- **Unreviewed; no text has been changed.** The protocol asks for a check before the Verlet containment pass of its §2a.

## 7.3 Upload shape, as planned before the upload

The protocol places this subsection last, but it is the plan written before the upload of §7.3a. Two of its points were settled by the upload itself.

**Size.** Publishing only the stripped checkpoints and the results, the five repos total **about 2.6 GB**: 587 / 595 / 592 / 586 MB for the four Fock arms, 273 MB for GPT-2.

**The collection design.** The `collection_cfc` design in `_ladder/ladder.json` stands. It is a **separate** collection, *Semantic Simulation — CfC+BAOAB Mechanism Ladder*, with the Verlet-instability repo as item 0 (the prologue). The two queued arms are added when they run, not created empty. It is separate from *SPLM Model Family* because, in the plan's words, every repo in that family is Verlet-era, and the integrator is exactly what this collection holds fixed. (§7.3b, written the same day, corrects that premise: the family is pre-CfC, mostly damped Euler. The reason for a separate collection survives, since the integrator still differs.)

**The paper's URL waits on the collection.** HuggingFace derives the collection slug from the title, and it was not clear what it would do with the em dash and the `+` in `CfC+BAOAB`. The existing family URL resolves without an id suffix, so a slug-only URL is safe once the real slug is known. The plan: create the collection first, read its URL, then add one sentence to `main.tex`'s *Code and supporting material* paragraph, next to the existing family link. Guessing the slug into a 481-page book is the one step not worth saving time on. (§7.3a records the real slug and that the suffix-free form redirects.)

## Points to read with care

- **Order of §7.3 and §7.3a.** In the protocol, §7.3a (done) and §7.3b come before §7.3 (the plan). This note keeps that order and marks §7.3 as the earlier plan.
- **§7.3 against §7.3b.** §7.3 says every repo in the SPLM family is Verlet-era; §7.3b, the same day, finds that fourteen of fifteen are not. The protocol does not edit §7.3 to match.
- **The edits to §6.8 and §11.6** of [Geodesic_Experiments_with_CfC_BAOAB.md](../Geodesic_Experiments_with_CfC_BAOAB.md) were budgeted in §7.2a. The closure box records the changed E3 verdict there; §7 does not say when the edits were made.
- **The two extra F1 sessions** (to put `attention` and `attention_potential` on the current 6b-12) are costed in §7.2a. The protocol records no result for them.
