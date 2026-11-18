# The $L = 4$ diagnosis and the D-series

**What this note is.** A readable summary, written 2026-11-17, of two blocks of the [protocol](../Depth_Ladder_and_Matched_Baseline_Protocol.md): the $L = 4$ diagnosis block (from "L=4 is genuinely behind, and it is NOT overfitting" to just before §3.1) and §3b, "The D-series — why depth does not pay, and what to do about it". It gives each diagnostic and what it showed, the conclusion and why, and the four planned repairs D1 to D4. Every number, verdict and date is copied from the protocol, which stays the full running log. The short version of the $L = 4$ arm, with the run-4 pre-registration and the live-gradient follow-up, is in [Ladder_Early_Arms_Results.md](Ladder_Early_Arms_Results.md#e-l4); this note does not repeat it.

**Status (2026-11-17).** Run 4 was diagnosed as fitting worse, not overfitting; the $\omega \Delta t$ band was a HIT; D1 was the only rung authorised; no D1 to D4 result is recorded, and the D-series was superseded on 2026-10-05 once live gradients made depth pay.

**Contents**

1. [Terms](#l4d-terms)
2. [The result being diagnosed](#l4d-result)
3. [The diagnostics and what each showed](#l4d-diag)
4. [The conclusion: behind, not overfitting](#l4d-conclusion)
5. [Can parameters fix it?](#l4d-params)
6. [The D-series (§3b)](#l4d-dseries)
7. [What to run in the gaps](#l4d-gaps)
8. [Outcomes](#l4d-outcomes)
9. [What stands and what is open](#l4d-open)

---

<a name="l4d-terms"></a>
## 1. Terms

- **$L$, $\Delta t$, $T$.** $L$ is the number of layers, each one integration step of the same dynamics. $\Delta t$ is the timestep of each step. $T = L \cdot \Delta t$ is the total integration time. The ladder held $T = 8$, so $L = 4$ means $\Delta t = 2$.
- **Settled PPL.** The mean of the last three validation evals. It is what scores a pre-registered band.
- **$\omega$ and $\omega \Delta t$.** $\omega$ is the frequency of a stiff mode of the learned potential. The product $\omega \Delta t$ is the angle a mode turns through in one step. The value 2 is the "wall": the stability limit of the Störmer/leapfrog integrator. The model's closed-form substep integrates these modes exactly, so a reading past 2 means stiffness, not instability.
- **`bproj_sig`.** A proxy for the largest singular value of the low-rank curvature factor. Higher means a stiffer potential.
- **Weight tying.** Almost every module is one set of weights reused at every layer. Only `depth_code`, the creation/destruction gates and the per-layer scalars differ by layer.
- **WSD.** The warmup-stable-decay learning-rate schedule. Most of each run's gain arrives in the decay phase.
- **Gen 2 and Gen 3.** Gen 2 is the detached-gradient convention run 4 used. Gen 3 makes `VPHI_GRAD_PATH` and `XI_GRAD_PATH` live.

<a name="l4d-result"></a>
## 2. The result being diagnosed

Run 4 is $L = 4$ at $\Delta t = 2$, $T = 8$, with everything else as the $L = 2$ arm (run 3). It led $L = 2$ from step 1,500 to 8,500, then fell behind, and the gap widened: +4.7% at 11,500, +6.5% at 12,500, +9.0% at 14,500, +7.7% at 15,500. Its validation curve oscillated upward (88.68, 91.81, 92.60, 90.57, 92.76), "which is not a plateau".

**Do not read the early steps.** The first eval was 495.47 against $L = 2$'s 478.33, which looks like a 3.6% deficit. It is not one: the curve was falling 46% per 500-step eval, so 495.47 sits 28 steps behind $L = 2$'s own trajectory, at 1.5% of the run, with the learning rate at 31% of peak. The first eval where a gap means anything is around step 15,000 (−0.5% per eval).

**Depth is nearly free in parameters (recorded at launch, 2026-09-27).** $L = 4$ has 76,823,511 parameters against $L = 2$'s 76,770,256: 53,255 more, or 0.069%, about 26,600 per added layer. The exchange field alone costs 589,825, eleven times more. So capacity is not a confound, and the only two candidate explanations are depth and the timestep. The token budget is again exactly 532,480,000 (batch 16, accumulation 2).

Run 4 settled at **71.75** on 2026-09-29: a MISS on its 55–65 band, 7.1% behind $L = 2$'s 66.98. The table is in the [sibling note](Ladder_Early_Arms_Results.md#e-l4) and in §6 below.

<a name="l4d-diag"></a>
## 3. The diagnostics and what each showed

### 3a. Train loss (step 15,500, 2026-09-28): the decisive measurement

Next-token loss, $L = 4$ minus $L = 2$, on the steps both logs share:

| step | delta ntp (nats) |
| ---: | ---: |
| 6,050 | −0.009 |
| 8,050 | −0.001 |
| 9,050 | **+0.044** |
| 11,050 | +0.044 |
| 13,050 | **+0.064** |
| 15,050 | +0.037 |

It crosses where the validation curve crosses and stays positive. At the last shared step $L = 4$ is "+4.7% worse in perplexity terms on the training objective". The deeper model is fitting worse.

### 3b. Curvature: higher at $L = 4$ and rising faster

| step | bproj_sig, L=2 | bproj_sig, L=4 |
| ---: | ---: | ---: |
| 6,500 | 51.7 | **54.3** |
| 10,500 | 64.0 | 67.8 |
| 15,500 | 74.6 | **83.1** |

`sig_max` reached 42.7 at $L = 4$ by step 6,500, a level $L = 2$ did not reach until about step 9,500. Health was otherwise clean: zero `[spike]` events, zero watchdog triggers, grad norm steady at 0.28–0.33, dominant register index rotating rather than locking. "Nothing is diverging; it is simply losing."

### 3c. Two candidate mechanisms

| mechanism | what it says | predicted omega·dt at L=2 |
| --- | --- | ---: |
| (1) curvature compensation | the model doubles omega when dt halves; the finer step buys nothing at 2.41× the compute | about 2.2 |
| (2) weight tying | the same weights serve twice as many layer roles with 0.069% more parameters | about 4.5 |

### 3d. The resonance monitor, fixed and resumed (step 20,000)

The in-flight $\omega \Delta t$ monitor had recorded nothing on any CfC+BAOAB arm: its hook was broken for the whole ladder, and every reading read `EMPTY SUMMARY`. The fix (`semsimula-diag` commit `409175b`, "resonance: open a record on the low-rank path too") was cherry-picked on 2026-09-28; 134 tests pass. Run 4 was interrupted at step 20,160 and resumed from `_best.pt` at 20,000, with best PPL 85.81 carried over and the decay still due at 21,125.

- **The trap.** A Colab runtime restart keeps `/content`, and the notebook clones `semsimula-diag` only if it is absent. The stale clone had to be removed by hand.
- **Operational note.** The console is the only home of the per-step telemetry and the resonance lines; `training_log.jsonl` does not carry them. Save the tab before every restart.
- **What in-flight readings cannot do.** The $L = 2$ arm trained under the same broken hook, so it has no in-flight counterpart. The discriminator is an offline endpoint comparison (Cell 6b-13) on both checkpoints.

### 3e. The first $\omega \Delta t$ reading (step 20,500) and the pre-registration

The median was 2.236 (max 4.398), with 60.9% of token-layer pairs past the wall. That is the first such measurement on a CfC+BAOAB arm. It shows the exact integrator is "load-bearing": under an explicit integrator most of the model would be amplifying at every step. At $\Delta t = 2$ it implies $\omega$ of about 1.118.

`bproj_sig` at the matched step 20,450 was 97.35 at $L = 4$ against 81.77 at $L = 2$, a ratio of 1.19, nowhere near the factor of 2 that exact compensation requires. Scaling by it predicts about 3.8 at $L = 2$.

**Pre-registered before the measurement: $\omega \Delta t$ at $L = 2$ lands in [3.3, 4.2].** Below 2.6 hands the deficit to compensation; above 4.7 means $\omega$ fell with depth, which neither mechanism predicts.

### 3f. Nine readings after the fix (steps 20,500–24,500, recorded 2026-09-28)

| step | omega·dt p50 | omega·dt max | over wall | bproj_sig | val PPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20,500 | 2.236 | 4.398 | 60.9% | 97.48 | 87.05 |
| 21,000 | 2.346 | 4.573 | 65.6% | 98.88 | 85.97 |
| 21,500 | 2.318 | 4.579 | 63.5% | 100.10 | 85.83 |
| 22,000 | 2.271 | 4.589 | 61.9% | 101.40 | 85.62 |
| 22,500 | 2.283 | 4.661 | 63.1% | 102.64 | **82.91** |
| 23,000 | 2.296 | 4.692 | 63.4% | 103.76 | 84.08 |
| 23,500 | 2.262 | 4.503 | 62.8% | 104.85 | 83.99 |
| 24,000 | 2.258 | 4.627 | 62.6% | 105.67 | 83.97 |
| 24,500 | 2.293 | **4.945** | 64.1% | 106.25 | **81.41** |

| series | mean | slope / 1k | residual std | change 20,500 to 24,500 |
| --- | ---: | ---: | ---: | ---: |
| omega·dt p50 | 2.285 | −0.004 | 0.035 | +2.5% |
| omega·dt max | 4.619 | +0.077 | 0.115 | +12.4% |
| over wall | 63.1% | +0.13 pp | 1.41 pp | +3.2 pp |
| bproj_sig | 102.3 | +2.24 | 0.33 | +9.0% |

- **The median is stationary, past the wall** (2.285, plus or minus 0.033), while validation PPL kept improving. The step-20,500 reading was representative.
- **The max drifts up, mildly** (4.40 to 4.95); about half the change is the last reading.
- **Caveat on the band.** `bproj_sig` tracks the tail, not the median: corr(p50, `bproj_sig`) = −0.16, corr(max, `bproj_sig`) = +0.67. The band was derived by scaling the median with it. So "a hit is less confirmatory than it looked, and a miss is less damning". The band was not moved.
- **Exploratory secondary comparison, labelled as added after these readings:** max($L = 2$) over max($L = 4$), reported beside the scored median, never in place of it.

### 3g. Incident: the finished $L = 2$ arm resumed (2026-09-28/29), repaired

A laptop reboot restored an older notebook copy with `LADDER_L = 2`. Cell 6 resumed run 3's completed arm and trained 31,501 to 32,183 before it was interrupted. The checkpoint was untouched (md5-verified); 15 appended log lines were trimmed on 2026-09-29, byte-identical to the local copy. Every run-3 number (settled 66.98) stands. A completed-run guard was added to Cell 0 and Cell 6 the same day.

The accident printed one $L = 2$ resonance line: median 3.797, max 6.926, 98.334% over the wall. It landed inside the band, but on the wrong weights and wrong batches. The scored $L = 2$ value therefore had to be reported as "measured after an unplanned preview of 3.80".

### 3h. The endpoint comparison, Cell 6b-13 (scored 2026-09-29)

Both arms at `_best.pt`, same seed (20260928), same tokens:

| arm | step | omega·dt p50 | omega | max | over the wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| L=2, dt = 4 | 31,500 | **3.796** | 0.949 | 6.766 | 98.4% |
| L=4, dt = 2 | 31,000 | **2.292** | 1.146 | 4.484 | 63.9% |

- **Band [3.3, 4.2]: HIT** at 3.796. Not blind: previewed twice (3.797 by accident, 3.796 by a local dry run). The band itself was fixed before either.
- **Exploratory:** max ratio 1.51.
- Halving $\Delta t$ raised $\omega$ by only **21%**; the $\omega \Delta t$ ratio is 1.66, against 1 for exact compensation and 2 for none. In log terms the model compensated about 27% of the way. This is mechanism (2).

Aligned by integration time, per layer:

| time window | L=2 layer : omega | L=4 layers : omega |
| --- | --- | --- |
| t in [0, 4] | 0 : **0.84** | 0 : **1.68**; 1 : 1.00 |
| t in [4, 8] | 1 : 1.13 | 2 : 1.25; 3 : 0.74 |

At the first layer compensation is full ($\omega \Delta t$ 3.34 against 3.37). Beyond it there is none; $L = 4$'s last layer is the softest in either model, 23% over the wall against 97–100% at every $L = 2$ layer. So "L=4 is a different dynamical system, not a finer integration of the same one". Results are filed as `D2_6b13_omega_dt_endpoint.txt` in both results folders, for example [the $L = 4$ one](../../notebooks/conservative_arch/scaleup/results/cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_L4probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt2_lr0p0012_noattn/D2_6b13_omega_dt_endpoint.txt). *Unclear:* the protocol does not say why the file carries a "D2" prefix; it is not the D2 rung of §3b.

<a name="l4d-conclusion"></a>
## 4. The conclusion: behind, not overfitting

- **Not overfitting.** $L = 4$ is worse on the training loss too (up to +0.064 nats), so it is a fitting problem. "Every 'more layers, more capacity' intuition is ruled out with it."
- **Not a bigger model.** It has 0.069% more parameters, so capacity cannot explain it.
- **Mostly not compensation.** The endpoint reading puts the deficit on mechanism (2): only the first layer re-tunes its curvature. The pattern is "consistent with" weight tying, "not a demonstration of it".
- **The projection was wrong.** A settled value of 88–93 was projected at step 15,500; it ignored the WSD decay, which took $L = 4$ from 85.8 (step 21,125) to 71.8. Corrected 2026-09-29.
- **Still open when recorded.** D1 decides step size against depth. Knock-one-layer-out diagnostics and per-layer 6b-8 and 6b-9 could test whether layers 2–3 contribute. The per-group clip difference (reverse-channel gate clipped 12–16 times at $L = 4$, 20 at $L = 2$) and a learning rate tuned at $L = 2$ were unexamined.

<a name="l4d-params"></a>
## 5. Can parameters fix it? (analysed 2026-09-28)

"Partly, and the cheap part is worth trying; the expensive part is not a fix but a different thesis."

| group | tied across layers? | share |
| --- | --- | ---: |
| V_θ | tied | about 59% of non-embedding |
| V_φ, register bank, reverse channel | tied | small |
| depth_code | per-layer | 6,144 params at L=4 (0.013%) |
| creation / destruction gates | per-layer | 205,700 at L=4 |

Untying every tied group at $L = 4$ costs 2.76 times the parameters. Untying $V_\theta$ makes $L$ stacked systems instead of one potential integrated $L$ times, which forfeits the Jacobi metric, the geodesic results of book §27 and the "native memory" argument: it "would trade the thesis for the perplexity". The cheap, thesis-preserving options are loosening `depth_code`'s clip and widening the depth conditioning. These became D2 and D3.

<a name="l4d-dseries"></a>
## 6. The D-series (§3b, opened 2026-09-28)

### The constraint

Since $T = L \cdot \Delta t$, depth, timestep and integration time cannot all be held. The ladder held $T = 8$:

| arm | L | dt | T | settled |
| --- | ---: | ---: | ---: | ---: |
| run 7 | 1 | 8 | 8 | 87.09 |
| run 3 | 2 | 4 | 8 | **66.98** |
| run 4 | 4 | 2 | 8 | 71.75 (settled 2026-09-29) |

At fixed $T$ the optimum is interior: $L = 2$ beats $L = 1$ by 23% and $L = 4$ by 7.1%. "Nothing in the framework predicted an interior optimum."

### The four rungs

| rung | what it changes | cost | prediction |
| --- | --- | --- | --- |
| D1 | L=4 at fixed dt = 4 (LADDER_T = 16.0), tag L4probe + idt4 | about 27 h, one run | settles in 62–70 |
| D2 | depth_code clip 0.25 to 1.0 | about 27 h, 0 new params | if binding: pre-clip norm under threshold within about 2,000 steps, settled PPL better by more than 2%; if nothing moves, go to D3 |
| D3 | wider depth conditioning, still one potential | about 27 h, a few thousand params | only if D2 is null |
| D4 | untying the tied groups | new architecture | only if D1–D3 fail; gets its own name, does not enter the ladder |

- **D1** holds the operating point that worked and asks whether depth itself pays. It confounds depth with integration time, "the complement of run 4's confound". The named quantity is whether $T = 8$ is saturated; the tell is D1's train loss against run 3's. If D1 lands near 71.75, depth is the problem. With run 4 at 71.75, that branch and the top of the band are under 2 PPL apart, so D1 must be read against eval noise.
- **D2** has a tag trap. `GRAD_CLIP_OVERRIDES` lives in Cell 6 and does not reach the variant tag, so D2 would resolve to the same Drive folder and silently resume. A `dcclip` tag component and a Cell 5b guard must come first.
- **Standing caution.** The conservative twin added 589,825 parameters and settled at 80.90 against 66.98, 20.8% worse. "Capacity is not a free axis here."

### The decision and the order (2026-09-28)

**"Run D1, then return to the factorial unless D1 comes back positive."** Each rung competes with runs 10 and 11, which close the $V_\varphi$ by Fock 2-by-2 factorial. D2 and D3 were parked.

| D1 settles | reading | next |
| --- | --- | --- |
| materially below 66.98 | matched T was the confound | stop and re-state the headline ladder at fixed dt |
| about 66.98 (null) | depth does not pay even at a held operating point | runs 10 and 11 |
| about 71.75 (negative) | depth actively hurts at this budget | runs 10 and 11 |

The prior from run 4 was that the null or negative branch was likely. The clip-tag defect is fixed before C4 or C5 of the clip series, not before D1.

<a name="l4d-gaps"></a>
## 7. What to run in the gaps

1. **The $\omega \Delta t$ offline comparison (Cell 6b-13), as soon as run 4 finishes and before D1.** Run order 0, 1, 1b, 2, 3, 4, 5, 6b-13; do not run Cell 6, which resumes training. Cell 6b is not this measurement: it reads the diagonal curvature proxy. One session per arm, since $L$ is baked into the model. This ran and scored (§3h).
2. **C1**, extending the E5 slider above $\lambda = 1$: minutes.
3. **C2 and C7**: no GPU; they read optimizer state and existing logs.

C1, C2 and C7 belong to the clip series, covered in [Clipping_and_the_C_Series.md](Clipping_and_the_C_Series.md); the [results note](Ladder_Early_Arms_Results.md#e-l4) records C1 and C2 as scored on 2026-10-05.

<a name="l4d-outcomes"></a>
## 8. Outcomes

- **D1 to D4: no result recorded** in the protocol. No `L4probe` + `idt4` results folder exists under [the results directory](../../notebooks/conservative_arch/scaleup/results/).
- **Superseded 2026-10-05**, per [../Hyperparameter_Tuning_Checklist.md](../Hyperparameter_Tuning_Checklist.md). The premise was a Gen 2 reading. Under live gradients depth pays: $L = 4$ live settled at 50.10 against 53.12 for $L = 2$ Fock live (G2): 5.7% better, at 1.9 times the cost per step. The "interior optimum" was "an artefact of gradient starvation". D1 is unscheduled and needs its own Gen 3 pre-registration; D2 is unscheduled and still blocked by the clip-tag defect; D3 and D4 are withdrawn.
- The protocol's §5.10 records the same answer as "Key line 2, depth under live gradients: YES" (3.12 against 1.64 s/step); see [Abstract_Gating_Runs.md](Abstract_Gating_Runs.md).
- **Not the same series.** The protocol's later "D1–D3 on CPU" (2026-10-22) are damping diagnostics on a different arm, not these rungs.

<a name="l4d-open"></a>
## 9. What stands and what is open

**Stands.** Under Gen 2, run 4 fitted worse, not overfitted; the $\omega \Delta t$ band was a HIT at 3.796; only the first layer compensated for the smaller step; the clip-tag defect is real.

**Open.** Whether $T = 8$ is saturated (the residual D1 question); the clip-tag defect for knobs other than `POISSON_MODE_CLIP`; the knock-one-layer-out diagnostics, with no result recorded; the paired forecastability control (6b-10 on run 4, G4 in §5.10), also with no result recorded.
