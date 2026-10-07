# Hyperparameter tuning checklist — the from-scratch ladder

> **Scope.** Every knob that can be moved *without changing what the model is*,
> plus the few capacity knobs that change it, with the evidence for whether
> each one currently binds. This is an inventory and a queue, not a results
> document.
>
> Distinct from
> [`Depth_Ladder_and_Matched_Baseline_Protocol.md`](Depth_Ladder_and_Matched_Baseline_Protocol.md),
> which fixes the configuration so that `L` and the added mechanism are the
> only variables — **this document is the list of things that protocol holds
> fixed, and why each was fixed at the value it was.** Distinct also from
> [`Joint_Vtheta_QKNorm_Run_Diagnostic_Checklist.md`](Joint_Vtheta_QKNorm_Run_Diagnostic_Checklist.md),
> which covers run *health* rather than run *quality*.
>
> **Notebook:** [`colab_fock_cfc_baoab_lowrank_depth_ladder_openwebtext_d384.ipynb`](../notebooks/conservative_arch/scaleup/colab_fock_cfc_baoab_lowrank_depth_ladder_openwebtext_d384.ipynb)
> **Started:** 2026-09-22

---

## 1. Why this exists

§5.4 of the ladder protocol records that every Fock-vs-GPT-2 ratio is
**tuned against untuned**: GPT-2 runs at community-validated nanoGPT
defaults, the Fock side at values inherited from the warm-started L=8 arm
and never checked at this depth. That section promised the LR probe as the
first step at closing the asymmetry. The probe has now run, and it produced
two results worth generalising from.

**Tuning is large.** LR moved from 3e-04 to 1.2e-03 and bought **10.8%**
(75.09 -> 66.98), moving the ratio against the matched GPT-2 baseline from
1.507 to **1.345**. An earlier draft of this section read "real but small"
on the strength of the 2.6% gap measured at the decay boundary; §6 records
why that was wrong and what it cost.

**Nothing short of a full run ranks two settings.** The 1.2e-03 advantage
over 3e-04 read −15.1% at step 5,000, −2.6% at the decay boundary (21,125),
−6.6% at step 30,000 and **−10.8% at 32,500**. It was non-monotone in the
screening window and three-quarters of the final gap arrived *after* the
decay boundary. A 6,000-step probe does not merely understate the winner —
at 21,125 it had the margin four times too small. Every entry in §5
therefore carries a screening length, and for anything LR-like that length
is "full".

---

## 2. The loss, exactly

Needed because two of the three "lambdas" in this codebase are not what
their names suggest. From Cell 6, `forward_with_vreg`:

```python
loss = loss_ntp
if lambda_v > 0:
    xis = model.xi_module(h_L.detach())
    V_vals = model.V_theta(xis, h_L)
    v_reg_value = (V_vals.float() ** 2).mean()
    loss = loss + lambda_v * v_reg_value          # LAMBDA_V = 1e-2
if lambda_fock > 0:
    fock_reg_value = fock_coupling_reg(model, lambda_fock, fock_eps)
    loss = loss + fock_reg_value                  # lambda applied INSIDE
```

and, from the same cell:

```python
def fock_coupling_reg(mdl, lam, eps):
    alphas = mdl.xi_module.alpha
    return -lam * torch.log(alphas + eps).sum()
```

plus `loss = loss + model.pop_repulsion_loss()` when `REGISTER_REPULSION`.

Four consequences, each verified against the live 1.2e-03 log:

1. **The Fock regulariser is constant.** `LAMBDA_FOCK_REG = 5e-3` is passed
   as a literal every step. No warmup, no schedule, no anneal.
2. **It is a pure parameter barrier.** It reads only `xi_module.alpha` —
   not the data, not the activations. Identical for every batch.
3. **It is one-sided.** `-log(alpha)` is minimised as `alpha -> 1`, so it
   does not hold the channels at a target horizon; it pushes all five
   toward infinite memory. Channel 4 has saturated at exactly `1.000`.
4. **The logged `fock_reg` already includes lambda.** It reproduces
   `lam * sum(-log(alpha))` to four decimals at every logged step, so the
   logged 0.0099 is `lambda*R` with `R = 1.98`. Do not multiply again.

---

## 3. The knob inventory

Values are those of the L=2 ladder point as of 2026-09-22. "Binds?" is an
empirical claim about the live 1.2e-03 run, not a guess.

### 3.1 Optimisation — changes the search, not the model

| knob | current | binds? | evidence |
| --- | --- | --- | --- |
| `LR` | 1.2e-03 | **yes — biggest knob found; CLOSED** | **+10.8%** over 3e-04; bracketed by 2.4e-03 at 69.59 (§5 T0) |
| `WSD_STABLE_FRAC` | 0.60 | **no — swept, CLOSED** | 0.50 gave a null (t = -0.09); §5 T1 |
| `WSD_WARMUP_FRAC` | 0.05 | untested | 1,625 steps; no instability seen after it ends |
| `WSD_LR_FLOOR` | `LR * 0.05` | untested | 6.00e-05 at the current LR |
| `TARGET_EFFECTIVE_BATCH` | 32 | **likely** | 16,384 tok/step, unchanged while LR moved 4x |
| `WEIGHT_DECAY` | 0.01 | untested | applied to 57 tensors; 38 1-D tensors exempt |
| betas | (0.9, 0.95) | untested | hard-coded in the `AdamW` call, not a Cell 0 constant |
| `GRAD_CLIP` | 1.0 | **depends on LR and depth** | 0.0% of steps at 1.2e-03, 4.2% at 3e-04, 36.2% at L=8 (§7.3) |
| `GRAD_CLIP_OVERRIDES['reverse_channel_scale']` | 0.1 | **yes, adversely** | clipped 20-45x on most steps |
| `GRAD_CLIP_VPHI` | 0.3 | no | never tops the group table |

### 3.2 Loss shaping

| knob | current | binds? | evidence |
| --- | --- | --- | --- |
| `LAMBDA_FOCK_REG` | 5e-3 | **yes, weakly** | 0.23% of loss, rising 0.0075 -> 0.0099 against the task |
| `LAMBDA_V` | 1e-2 | **no — inert** | `v_reg` logs 0.0000; contribution ~0 |
| `REGISTER_REPULSION_COEFF` | 0.05 | marginal | `rep` ~0.0014, i.e. 0.03% of loss |
| `FOCK_REG_EPS` | 1e-6 | no | only matters if some `alpha -> 0`; none has |

### 3.3 Dynamics — physics, not optimisation

| knob | current | binds? | evidence |
| --- | --- | --- | --- |
| `FIXED_GAMMA` | 0.10 | swept once | `gamma_sweep` in results, but never at this LR |
| `LADDER_T` | 8.0 | **held by design** | the ladder's controlled variable; moving it voids §2 of the protocol |
| `LANGEVIN_T` | 0.0 | untested; SR5 arms calibrated 2026-10-03 (0.00025, 0.0022) | thermostat off |
| `PRECISION_LR_MAX` | 1.0 | **yes, softly** | a tanh cap; `bproj_sig` 84 means it is deeply saturated |
| `CREATION_LOGIT_SCALE_MAX` | 100.0 | not yet | `sig_max` reached 60.02 at 1.2e-03; **absorbing** if touched |

#### 3.3.1 Two ceilings that are easy to misread

Both are logged every step and neither means what its log line looks like.

**`bproj_sig` is not the capped quantity.** `PRECISION_LR_MAX = 1.0`
applies a **tanh soft cap** to each well's `||B_k||_F` at `sqrt(1.0)`, via
`_bound_lowrank` in
[`model_aniso_gaussian_vtheta.py`](../notebooks/conservative_arch/parf/model_aniso_gaussian_vtheta.py).
The logged `bproj_sig` is the **spectral norm of the `B_proj` weight
matrix** — the xi -> B generator — so 84 is not a violation of a cap of 1.0;
the two are different objects. What 84 does mean is that the tanh is deeply
saturated, so the gradient reaching B's *magnitude* is small and only its
*direction* is still learnable. That is a plausible contributor to LR's
diminishing returns (§5 T0), and it is a reason **not** to touch this knob
mid-programme: it changes the model, not the optimisation, and would break
comparability with all three completed arms.

**`sig_max` is a register, not a well.** It is

```
sigma_k = min(exp(lambda_k), CREATION_LOGIT_SCALE_MAX)
```

the creation-gate logit scale of the **sharpest register**, and the `@rN`
suffix is the **register index**, not a rank. An earlier draft of this
document and of §7 described it as a V&#95;theta well singular value and read
a `@r22 -> @r4` shift as a well reordering; it was a different *register*
becoming sharpest, which is a reorganisation of the register pool and says
nothing about V&#95;theta.

The ceiling is **absorbing, not soft**: `sig_max == 100.0` exactly zeroes
that register's gradient and freezes its sharpness permanently. Observed
scaling:

| arm | final `sig_max` |
| --- | ---: |
| L=2 `'none'` @ 3e-04 | 38.00 @r18 |
| L=2 `'attention'` @ 3e-04 | 43.81 @r20 |
| L=2 `'none'` @ 1.2e-03 | **60.02** @r4 |

1.58x for 4x LR, i.e. ~1.26x per doubling. It also retreats under decay
(peak 63.24 at step 26,350, settled 60.02), so a full run ends below its own
mid-run maximum.

#### 3.3.1a Both of these failed as predictors at 2.4e-03

Recorded because each was wrong in a *different* direction, which is more
useful than either being merely imprecise.

| quantity | predicted at 2.4e-03 | actual |
| --- | ---: | ---: |
| `bproj_sig` | ~170 | **273.6** |
| `sig_max` | ~75, climbing toward 100.0 | **49.92**, *down* from 60.02 |

**`sig_max` is not monotone in LR.** Register sharpening fell when the LR
doubled, so the absorbing-ceiling watch was aimed at a risk that does not
scale the way the 3e-04 -> 1.2e-03 pair suggested. Keep the stop condition —
touching 100.0 is still absorbing — but do not extrapolate a trajectory
toward it.

**`bproj_sig`'s constant ratio broke, and that is the useful part.** Against
the 3e-04 reference it held at 3.4-3.6x across *every* matched step of the
1.2e-03 run, then jumped to **10.8x** at 2.4e-03. It is the only logged
quantity that separates the good run from the over-driven one, which makes
**above ~5x the same-step reference** a candidate early-warning for "past the
optimum". One data point, so it is a hypothesis and not a rule — the way to
test it is to check the ratio on the next arm that turns out badly, not to
act on it now.

### 3.4 Capacity — changes what the model is

Moving any of these breaks parameter-matching with the completed arms, so
they are not tuning in the same sense. Listed for completeness.

`TOP_K=16`, `M=32` registers, `ANISO_RANK=4`,
`V_THETA_WELLS_PER_HEAD=8`, `V_PHI_MLP_HIDDEN=128`,
`V_PHI_N_HEADS=4`, `XI_OVERRIDE='5long'`,
`XI_CONTENT_D_K=48`.

### 3.5 Dead in the `'none'` arm

`RELAX_LAMBDA_FIXED = 1.0` and every other `RELAX_*` knob are
inactive unless `FORCE_RELAXATION != 'none'`. This is the "pinned lambda"
of the conservativity note — it scales a **force** for `'attention'` and a
**potential** for `'attention_potential'`, which are different units with
no principled match. It is a knob for the attention arms only.

---

## 4. What the live run says about the two regularisers

Measured on the L=2 `'none'` 1.2e-03 run between steps 6,050 and 20,850.

**`LAMBDA_V` is inert and should not be quoted as a control.** `v_reg` is
0.0000 throughout, so its contribution to the loss is of order 1e-7. It was
0.0002-0.0003 on the 3e-04 arm — so raising the LR made V&#95;theta's output
*magnitude* smaller even as `bproj_sig` grew 3.5x. The wells became sharper
and shallower at the same time.

**`LAMBDA_FOCK_REG` binds, and the task is fighting it.** The penalty rises
monotonically because the short channels are collapsing against the barrier:

| channel | alpha @ 6,050 | alpha @ 20,850 | horizon `1/(1-alpha)` |
| --- | --- | --- | --- |
| 0 | 0.402 | 0.330 (−17.9%) | 1.7 -> 1.5 |
| 1 | 0.659 | 0.497 (−24.6%) | 2.9 -> 2.0 |
| 2 | 0.864 | 0.855 | 7.4 -> 6.9 |
| 3 | 0.973 | 0.978 | 37 -> 46 |
| 4 | 0.999 | **1.000** | saturated; unbounded accumulator |

Two channels want horizons of 1.5 and 2.0 tokens. That is a claim about the
`XI_OVERRIDE='5long'` preset being mismatched at this depth, and it is
measurable without any new run.

---

## 5. The queue, in priority order

Each entry carries a **screening length**, because §1 establishes that short
pilots mis-rank. "Full" means the complete 32,500-step WSD schedule.

### T0. `LR` — **CLOSED 2026-09-23. Optimum bracketed at 1.2e-03.**

| LR | pre-decay | settled | decay | vs GPT-2 |
| --- | ---: | ---: | ---: | ---: |
| 3e-04 | 84.98 | 75.09 | 11.6% | 1.507 |
| **1.2e-03** | 82.81 | **66.98** | 19.1% | **1.345** |
| 2.4e-03 | **88.48** | 69.59 | 21.3% | 1.397 |

2.4e-03 is worse by 3.9%, landing in the pre-registered "above 69 ->
bracketed" band. A quadratic through the three points in `log2(LR)` puts the
vertex at **1.13e-03** with a predicted minimum of 66.96 against 1.2e-03's
measured 66.98. **1.8e-03 is not worth testing** — the fit places the optimum
*below* 1.2e-03, not between it and 2.4e-03, and the curve is flat there.

Health at 2.4e-03 was clean — zero watchdog triggers, zero spike captures,
clip-hit 0.5%, finished at 32,500. This is over-driving, not instability.

#### Why the forecast failed, and what it changes

Pre-registered 63-66, centre ~64. **Actual 69.59 — wrong in direction.**

The decay-scaling mechanism the forecast was built on **held**: predicted
22.9%, actual 21.3%. What broke was the other half. The forecast assumed
pre-decay would keep drifting down (-1.3% per doubling -> ~81.7); it **rose
to 88.48**, worse than even the 3e-04 arm.

So the shape of the LR response is not "pre-decay creeps down while decay
does the work". It is: **the stable phase degrades first.** At 2.4e-03 the
model trains worse at constant LR, the decay then works harder than in any
other arm, and still cannot recover the deficit. Treat pre-decay as the
quantity that turns, not the decay fraction.

### T1. `WSD_STABLE_FRAC` — **CLOSED 2026-09-24. Keep 0.60.**

| arm | settled | vs control |
| --- | ---: | ---: |
| **0.60 — control** | **66.98** | — |
| 0.50 (branch from step 15,000) | 68.34 | +2.04% |

Settled is the mean of the last three evals, the convention of §5 of the
ladder protocol. Lengthening the decay window from 11,375 steps to 14,625
did not help. Run was clean: 0.0% clip-hit, `sig_max` peak 59.31 against the
100.0 ceiling, zero watchdog triggers, zero spike captures.

**0.70 was not run.** The 0.50 arm found no gain from moving off 0.60, and a
shorter decay was judged unlikely to do better; the schedule is treated as a
flat plateau and the knob is closed rather than half-swept.

#### What the number can and cannot support

Recorded because it bears on how this entry should be quoted. Per-eval
scatter in the tail is sd ~1.65 PPL, so a three-eval mean carries a standard
error of about 0.95 and the +2.04% is roughly 1.4 sd. Measured across the
22 evals from step 22,000 — where the two schedules genuinely differ — the
paired difference is **-0.03, se 0.35, t = -0.09**, 95% CI [-0.72, +0.66],
against a known-null window (shared schedule, 15,500-17,500) of sd 1.54.

So the honest reading is **"0.50 is not better than 0.60"**, not "0.50 is
2% worse". Both support the same decision. The distinction matters only if
someone later quotes the 2.04% as an effect size, which it is not.

This also bounds what the last-three convention can resolve anywhere in this
programme: **effects below roughly 2% are not separable from eval noise on
it.** That is comfortably fine for the LR sweep (§5 T0 measured t = -11.79)
and for anything else of that magnitude; it is not fine for schedule-scale
differences, which is why this entry closes on a null rather than a ranking.

#### The branch method worked, and is reusable

The run cost **7.8h instead of 14.4h** by resuming from the control's
step-15,000 checkpoint, valid because `lr_schedule` returns a constant `LR`
below `stable_end` regardless of `WSD_STABLE_FRAC` (see Cell 1d). The
validity check confirms it: across the window where both schedules are
identical, the branch tracked the control at mean **-0.31, sd 1.54** — pure
batch noise, no systematic drift. Any future knob that only takes effect
after a known step can be tested the same way.

### T2. Batch x LR jointly — **the one real interaction**

LR moved 4x with `TARGET_EFFECTIVE_BATCH` pinned at 32. Test 64 at the
winning LR.

- **Screening length:** full, or at minimum past step 16,000, where the LR
  gap had closed to half its step-5,000 value.

### T3. `LAMBDA_FOCK_REG`

0 / 1e-3 / 5e-3 / 2e-2. §4 shows it binding against the task on two of five
channels. Worth knowing whether the barrier is buying anything or simply
taxing them.

- **Screening length:** full. The alpha drift in §4 takes ~15,000 steps to
  become legible.

### T4. `WEIGHT_DECAY` and betas

0.01 and (0.9, 0.95) are both inherited. Standard grid, low expected value,
listed so that "never swept" stops being true.

### T5. `GRAD_CLIP_OVERRIDES['reverse_channel_scale']`

Being clipped 20-45x every step is a malfunction of *some* kind. But
`tanh(scale)` is 0.016-0.022, so the channel contributes almost nothing
whichever way it is resolved — **diagnose, do not tune.** The real question
is whether `REVERSE_CHANNEL` should be on at all at L=2.

---

## 6. What tuning can do — a falsified bound, and the corrected one

### 6.1 The claim that failed

This section previously read: *"Tuning changes decimal places; it does not
change conclusions. Five knobs at 2-3% each, stacked optimistically, reach
about 1.35."* It was written on 2026-09-22 against a forecast endpoint of
~73 and a measured LR gain of 2.6%.

**One knob returned 10.8%, and 1.35 was passed by that knob alone.**

| | pre-decay | settled | decay | vs GPT-2 49.81 |
| --- | ---: | ---: | ---: | ---: |
| L=2 `'none'` @ 3e-04 | 84.98 | 75.09 | 11.6% | 1.507 |
| L=2 `'attention'` @ 3e-04 | 77.62 | 68.33 | 12.0% | 1.372 |
| **L=2 `'none'` @ 1.2e-03** | 82.81 | **66.98** | **19.1%** | **1.345** |

"Settled" is the **mean of the last three evals**, the convention of §5 of
the ladder protocol, so the two documents can be read against each other.

### 6.2 The mechanism that was missed

**The decay fraction scales with the learning rate.** A WSD decay removes
gradient noise, a higher LR carries more of it into the boundary, so there
is more to remove:

| LR | decay | |
| --- | ---: | --- |
| 3e-04 | 11.6% | |
| 3e-04 (`'attention'`) | 12.0% | reproducible across arms at fixed LR |
| 1.2e-03 | **19.1%** | +7.5 points for 4x LR, i.e. **+3.7 points per doubling** |

Every forecast of this run applied the *reference arm's* 0.879 ratio to the
tuned arm and therefore ran high: 70 -> 75 -> 70 -> 72.8 -> 69.8 against an
actual **66.98**. The mechanism was named as an upside risk each time and
weighted at zero each time. **Rule for the next forecast: a decay ratio
measured at one LR does not transfer to another.**

It also explains why §1's screening rule has to be so strict. Three-quarters
of the final gap arrived after the decay boundary, because that is where the
mechanism lives.

### 6.3 The corrected bound

What survives is the **direction**, not the margin. LR is now closed at
**1.345**, and it was the largest knob available — 2.4e-03 came back *worse*,
so there is no further LR gain to bank. With the remaining knobs in §5
unswept, tuning might plausibly reach ~1.30. **Closing to parity still needs ~25%
that no combination of §3 entries is likely to supply**, so the
architectural gap is real.

But "tuning changes decimal places" was wrong, was asserted with more
confidence than one measurement supported, and is not to be restated in a
weaker form until the §5 queue has actually run. The honest position is:
**the size of the tuning effect is not yet known, and every ratio in this
programme remains an upper bound on the architectural gap.**

### 6.4 The attribution is suspended, and its sign has flipped

| comparison | result |
| --- | --- |
| both arms @ 3e-04 | `'attention'` better by **9.0%** |
| `'none'` tuned only | **`'none'` better by 2.0%** (66.98 vs 68.33) |

Tuning one arm did not narrow the exchange-field advantage — it **reversed**
it. The 51/49 attribution in §5.2 of the ladder protocol is therefore not
merely weakened; its sign is wrong at the only LR where either arm has been
tuned.

This does not establish that the exchange field is worthless: `'attention'`
has never been tuned either, and it may gain as much or more at 1.2e-03. It
does mean **no number in §5.2 of the ladder protocol can be attributed to
the architecture**, and that a narrowing which reverses under single-arm
tuning is much stronger evidence of a tuning artefact than a narrowing that
merely shrinks.

L=2 `'attention'` at the winning LR therefore outranks every entry in §5
except T0 and T1.

---

## 7. Depth transfer — does L=2 tuning carry to L=4?

Asked before scheduling L=4, and answerable from runs already on disk.

### 7.1 Depth does not add parameters

| arm | tensors | params |
| --- | ---: | ---: |
| L=8 | 67 | 76,673,824 |
| L=2 | 57 | 76,698,784 |

Within **0.03%**. `V_THETA_DEPTH_CONDITION=True` shares the potential
across layers and selects per layer with `depth_code`, so `L` is not a
capacity knob — it is how many times the same operator is applied, at
`dt = T/L`. That is the flow-vs-maps question of
[`Composing_Single_Layer_Inferences_Flow_or_Maps.md`](Composing_Single_Layer_Inferences_Flow_or_Maps.md),
and it is the reason most entries in §3 should transfer unchanged.

### 7.2 Why `LR` transfers — the 1/L argument fails under Adam

The tempting argument: a shared parameter collects `L` gradient
contributions, so the optimal LR should fall as `1/L`. **Adam absorbs it.**
Each parameter is normalised by its own gradient RMS, so a uniform rescale
of all gradients leaves the update unchanged. What survives is a curvature
effect — deeper composition, sharper landscape — which is real but weak, and
LR is empirically far more depth-stable than width-stable.

Cost settles the rest, though less comfortably than an earlier draft of this
section claimed — it argued from a 2.6% LR gain, and the knob turned out to be
worth **10.8%**. A knob that large is worth getting right, so the case now
rests entirely on the transfer argument above rather than on the prize being
small: §1 established that a 6,000-step probe mis-ranks, a trustworthy one
runs to ~16,000, and at L=4 that is **13h, half a full run**.

**Do not re-sweep LR per depth** — but the reason is that the ladder is valid
at any *common* LR, not that LR is cheap to get wrong. If §7.2's premise
fails and the optimum does move with depth, every ladder point inherits the
error; the check for that is one LR probe at the far end of the ladder, not
one per point.

### 7.3 Why `GRAD_CLIP` does *not* transfer

A clip is a hard threshold, which is precisely what Adam's scale-invariance
does not absorb. Same `GRAD_CLIP = 1.0` in every arm:

| arm | lr | steps over the clip | median `grad` |
| --- | ---: | ---: | ---: |
| L=2 `'none'` @ 3e-04 | 3.00e-04 | **4.2%** (27/650) | 0.650 |
| L=2 `'attention'` @ 3e-04 | 3.00e-04 | 7.5% (49/650) | 0.870 |
| L=8, warm anneal | 1.55e-04 | **36.2%** (29/80) | 0.920 |
| **L=2 `'none'` @ 1.2e-03** | 1.20e-03 | **0.0%** (0/530) | **0.270** |

Nine times the clip rate at L=8, at a **lower** LR — which reads as a
depth-dependent intervention that barely touches L=2 and heavily shapes L=8.

**The last row weakens that reading, and was added after it.** At fixed
depth, raising the LR from 3e-04 to 1.2e-03 took the clip rate from 4.2% to
**zero** and median `grad` from 0.650 to 0.270. So the clip rate responds at
least as strongly to LR as to depth, and the first three rows differ in both.
The confound is still plausible — 36.2% is large and the L=8 LR was the
*lowest* of the four — but this table does not isolate depth, and the earlier
draft implied it did.

Two further caveats, both load-bearing. The L=8 row is 80 logged rows from a
warm-started anneal, not a from-scratch ladder point — directional, not
decisive. And gradient norm falling as LR rises means `grad` is not a clean
proxy for optimisation difficulty on its own.

**Consequence for §7.5:** the ladder now runs at 1.2e-03, where L=2 clips on
0.0% of steps. If L=4 also comes in near zero, the clip is inert at the
ladder's operating point and the confound closes without needing the L=8
question resolved at all.

This also qualifies the informal "L=2 beats L=8" reading (75.09 against
81.58): if L=8 spends a third of its steps clipped and L=2 spends 4%, an
unknown part of that 6.8 PPL is the clip rather than the depth.

### 7.3a Depth can change the model, not just the optimisation — **L=1**

§7.1 says depth adds no parameters, so a ladder point is the same operator
applied more times. **L=1 is a partial exception.** The register bank is
still read there — `register_embed` takes next-token gradient — but the
creation gate is not trainable: `salience` initialises to exactly 1.0, so
`(1 - blend) == 0` annihilates the creation readout at layer 0, and only
layers 1 and up can train the shared module. L=1 therefore runs with a
**static** bank. §6.1 of
[`Depth_Ladder_and_Matched_Baseline_Protocol.md`](Depth_Ladder_and_Matched_Baseline_Protocol.md)
has the evidence and the opt-in `register_salience_init` knob;
[`Fock_Mechanism_Efficiency_Across_Layer_Depth.md`](Fock_Mechanism_Efficiency_Across_Layer_Depth.md)
has the depth-by-depth picture.

An earlier draft of this subsection said the Fock mechanism was switched off
at L=1 entirely. That came from probing a **freshly built** model, where
`tanh(reverse_channel_scale) == 0` and the warmup is `0/4000`, so the
register-to-token path is gated shut at every depth. **Never measure register
behaviour without first reading the effective gate** — Cell 6b-8 prints it
before anything else.

Two consequences for this document. Anything tuned at L=1 is tuned on a model
with a frozen creation gate, so **it does not transfer up** — the reverse of
§7.2's conclusion for LR. And `REGISTER_REPULSION_COEFF` (§3.2,
"marginal") is the only term that can shape register *content* at L=1 once
the bank is frozen, which makes it structural there rather than marginal.

### 7.4 What genuinely changes at L=4

| | L=2 | L=4 | consequence |
| --- | --- | --- | --- |
| `dt = T/L` | 4.0 | 2.0 | **more** integrator headroom; stability knobs get easier |
| cost | ~1.48 s/step, 13.4h | ~2x | **~27h**, one Colab reconnect minimum |
| clip-hit rate | **0.0%** at 1.2e-03 | unknown | the number this section exists to capture |
| `alpha` drift | §4 | unknown | four layers now read the same shared xi channels |
| `sig_max` | 60.02 / 100.0 | unknown | one shared register pool, now written by 4 layers |

The resonance criterion and the integrator stability guards genuinely relax
at `dt = 2.0`, so none of those needs action. **`PRECISION_LR_MAX` does
not** — an earlier draft said it did. Per §3.3.1 it is a `tanh` cap on a
*parameter* norm, with no `dt` in it, so depth neither tightens nor loosens
it.

`sig_max` is the one to add to the L=4 watch list for a reason specific to
depth: the register pool is **shared**, so four layers now drive the same
`lambda_k` that two did. If sharpening scales with the number of writers the
way it scales with LR, L=4 could approach the absorbing ceiling from a
direction the LR sweep never probed.

`WSD_STABLE_FRAC` is closed at 0.60 (§5 T1). Batch and weight decay carry no
depth dependence — **test them at L=2, where they cost 13h, then carry the
winner up.**

### 7.5 Required at every ladder point, from L=4 onward

**Record the clip-hit rate.** It is free, it is currently invisible, and it
is the one quantity known to vary with depth independently of the
architecture. It is one line against any saved training log:

```bash
python3 - "$LOG" <<'EOF'
import re, sys
g = [float(m.group(1)) for l in open(sys.argv[1])
     if (m := re.search(r'^step\s+\d+/\d+.*grad=([\d.]+)', l))]
over = sum(v > 1.0 for v in g)          # 1.0 = GRAD_CLIP
print(f'{over}/{len(g)} = {100*over/len(g):.1f}%  median {sorted(g)[len(g)//2]:.3f}')
EOF
```

Pre-registered decision rule, recorded 2026-09-22:

- **~10-15% at L=4** — interpolates cleanly between 4.2% and 36.2%. Note the
  confound in the results table and proceed.
- **above ~25%** — the clip is doing more work than the depth is. Raise
  `GRAD_CLIP` until it stops binding at every depth, and **re-run both L=2
  points**, because their 4.2% and 7.5% are then not a common baseline.
- **below ~7%** — depth is not driving the clip, the L=8 row was an artefact
  of the warm start, and this section closes.

**Calibrate the bands against the right baseline.** Those thresholds were set
from 3e-04 data (L=2 at 4.2%). The ladder now runs at **1.2e-03, where L=2
clips on 0.0% of steps**, so the honest comparison for L=4 is against zero,
not against 4.2%. Any non-zero rate at L=4 and 1.2e-03 is itself the depth
signal — the bands above stay as a fallback for a re-run at 3e-04.

**Record the trained θ = ω·Δt distribution with every Gate 3 reading** (added 2026-10-04): p05/p50/p95 from 6b-13, θ/sin θ at the median, and whether 1.5× refinement carries the median across π. A depth choice is also a choice of θ (Δt = T/L), and refinement readiness appears to track the π crossing (SR-π, protocol §5.9). A ladder point whose stiff modes train near or past π should be expected to refine badly.

**Also record `sig_max` and which register holds it.** Per §7.4 the
register pool is shared, so depth changes how many layers write the same
`lambda_k`. `sig_max == CREATION_LOGIT_SCALE_MAX` at any ladder
point is a stop condition, not a diagnostic (§3.3.1).

### 7.6 Sequencing

L=2 `'attention'` at 1.2e-03 (13h) comes **before** L=4 (27h). §6 is blunt
about why: with only one arm tuned, the exchange-field gap cannot be
attributed to the architecture at all. That run changes a conclusion; L=4
adds a point to a curve.

---

## 8. Ledger

| date | knob | screening | result |
| --- | --- | --- | --- |
| 2026-09-21 | `LR` 3e-04/6e-04/1.2e-03 | 6,000-step probe | ranked 1.2e-03 first, margin 15.1% — **ranking right, margin wrong four ways over**, see §1 |
| 2026-09-22 | `LR` 1.2e-03 | **full 32,500** | **75.09 -> 66.98, +10.8%.** Ratio vs GPT-2 1.507 -> 1.345. Decay 19.1% vs the reference's 11.6%. Clean: 0 watchdog, 0.0% clip, `bproj_sig` saturated at 85.4 |
| 2026-09-23 | `LR` = 2.4e-03 | full 32,500 | **69.59 — WORSE by 3.9%.** Optimum bracketed; quadratic vertex 1.13e-03. Pre-registered 63-66: **wrong in direction** (§5 T0) |
| — | `LR` **CLOSED** | — | **1.2e-03 is the ladder LR.** Ratio vs GPT-2 **1.345** |
| 2026-09-24 | `WSD_STABLE_FRAC` = 0.50 | branch from step 15,000, 17,500 steps | **68.34 vs 66.98** (+2.04% on last-3; t = -0.09 over 22 evals, i.e. a null). **Keep 0.60**, 0.70 not run, T1 closed |
| 2026-09-27 | exchange-field **gate** track: `RELAX_GATE='zero_readout'` on `'attention'` | 3,000-step probe | **NULL, +1.65%** vs the scalar gate (133.37 vs 131.20). The zeroed readout reaches the scalar arm's `share_max` within ~350 steps — the gate changes how the term starts, not where it ends. Ahead by 3.0% at step 500 only. Ladder §5.7 |
| 2026-09-27 | exchange-field **init-scale** track: `RELAX_INIT_SCALE=0.055` on `'attention_potential'` | 3,000-step probe | **NULL, +0.57%**. A 2.75x larger start moves the curve under 1 PPL anywhere. Ladder §5.8 |
| 2026-09-27 | depth-ladder run 5: **L=2 `'attention_potential'` @1.2e-03** | full 32,500 | **80.90 settled.** The price of conservativity is **+27.4%** (63.51 → 80.90) at matched parameters — and the conservative exchange field is **+20.8% worse than no exchange field at all** (66.98). Clip 0.0% vs `'attention'`'s 29.2%: it trained cleanly to a lower ceiling. Ladder §5.8 |
| 2026-09-26 | depth-ladder run 9: **L=2 `'attention'` @1.2e-03** (a RE-RUN at the tuned LR) | full 32,500 | **63.51 settled**, +7.1% over the same arm at 3e-04 (68.33); ratio vs GPT-2 **1.275**, the programme's closest. **Repairs §5.2**: the exchange field is worth **+5.2%**, not the +9.9% measured with both arms untuned. **Clip 29.2%** vs 0.0% for `'none'` — see the caveat in ladder §5.7 |
| 2026-09-25 | architecture: **L=2 `'none'`, reverse channel off** (ladder run 8, an architecture control, not a tuning run) | full 32,500 | **87.93 settled**, +31.3% vs 66.98 with the mechanism on; ratio vs GPT-2 1.765. **The inference-time ablations overstated the mechanism by 2.9x in PPL ratio, 4.9x in nats.** Clip 2.6% vs 0.0%. Ladder protocol §5.6 |
| 2026-09-24 | depth: **L=1** `'none'` @1.2e-03 (ladder run 7, not a tuning run) | full 32,500 | **87.09 settled**, +30.0% vs L=2 at the same LR; ratio vs GPT-2 1.748. Decay gain 11.2% vs 19.3% at L=2. Clip 2.6% vs 0.0%. Static register bank (§7.3a). Ladder protocol §5.5 |

## Scheduled: v6 abstract-gating runs — **opened 2026-10-03** (protocol §5.10)

- [x] **G2** — done 2026-10-04: **53.12** settled, published as `semsimula-ladder-live-owt-d384-l2-none`. P2.2 run to the full length: L=2 Fock-PARFLM `none`, live. **First.** Settles the register mechanism's value (< 57.76) and depth (> 50.10).
- [ ] **G4** — Cell 6b-10 on run 4, the Gen 2 twin of the L=4 live arm. Runs alongside G2 and takes minutes. Settles whether the L=4 forecastability comes from the live gradients.
- [x] **G3** — done 2026-10-05: **54.21** settled, 2.1% worse than G2 (key line NO). F3.2: L=2 `attention_potential` with everything live (`rglive` + `vplive` + `xilive`). Settles the exchange field's value against G2. *Running (2026-10-04).* At step 21,700 it trails G2 by about 1.6 PPL, and the exchange field's gradient has grown 0.3 → 0.8 (protocol §5.10, mid-run reading).
- [x] **G3′** (F3.2b) *(done 2026-10-06: 52.90 settled, best 50.97. Both predictions missed: settled ≤ 52.1 and gradient flat. −0.4% against G2 is within noise, so the hardened field adds nothing measurable; −2.4% against G3, so hardening recovers G3's loss. Diagnostics, causality check and probe still to run.)* — G3's configuration plus QK-normalised exchange-field routing (a clamped logit scale, as in the creation gate's `cgqk`) and a 0.3 clip override for `relax_field`. Predicted: the gradient norm stays flat (70%) and settled ≤ 52.1 (50%). Code: a QK-norm switch on `XiRoutedConservativeAttention`, off by default and verified bit-identical. **After G3 completes, ahead of the FO 2×2.** Not gating v6. *Implemented and verified 2026-10-05 (protocol §5.10): Cell 0 `RELAX_ATTN_QK_NORM = True`, `RELAX_FIELD_CLIP = 0.3` on G3's settings.*
- [ ] **G1** — **scheduled, deferred until G2–G4 are in.** A GPT-2 matched on parameters at the same width: d=384, L=22, untied, 77.8M. Width is held at 384 by design: comparisons stay in the same semantic-space dimension, so d=512 was rejected. The author's reservation is that matching the parameter count by depth ignores the model's dynamics and may draw reviewer questions; the run is kept for completeness. GPT-2 notebook Cell 0: `N_LAYERS = 22`, `TIE_EMBEDDINGS = False`. Commit and push the `VARIANT_TAG` folder guard before running it.

## Scheduled, free — run now: DP-series — what Doi–Peliti process do the trained registers implement? — **opened 2026-10-05** (protocol §5.14)

**Priority (2026-10-05):** first in the free queue. These are evaluation-only, local CPU checks taking minutes, run in parallel with G3′, which is training on Colab. They gate the v6.1 book revision of §10.5.2 (the Doi–Peliti paragraph and the v2 mapping table) and §20 (salience). Companion note: `Doi_Peliti_Dynamics_of_Semantic_Particles_and_Registers.md`.

- [x] Write the companion note, theory §0–§6, with derivations and figures. *Done 2026-10-05.*
- [x] **DP1** *(done 2026-10-05: hit; active fraction 0.9995–1.000, so there are no number dynamics)*: active fraction, below-threshold cells and destruction gate g, from layer 1 up, on G2, L=4 live and G3.
- [x] **DP2** *(done 2026-10-05: both predictions missed narrowly on G2, at 1.30% and 0.037; well below the 5% line, so content is not shared)*: register content duplication (salience-weighted cosine, near-duplicates), earlier positions against the last.
- [x] **DP3** *(done 2026-10-05: hit, and negative, down to −0.41; salience is retention, not intensity)*: Spearman ρ between salience and each register's leave-one-out reverse-channel contribution.
- [x] Record the results in the note (§7–§8) and score them in protocol §5.14. *Done 2026-10-05.*
- [ ] **Book v6.1:**
  - §10.5.2: derivations of the coherent-state/Poisson and Hamilton-equation claims, the bosonic-versus-exclusion reading forced by DP1–DP3, and the causal-symmetry caveat on the bosonic justification;
  - §20: the salience sentence;
  - A3: the DP rows.

**Free queue, in order (updated 2026-10-05):**
1. ~~DP1–DP3~~ done 2026-10-05.
2. ~~CB0~~ done 2026-10-05.
3. ~~SR-π.3~~ done 2026-10-05: MISS. Its follow-ups SR-π.3b (MISS) and SR-π.4 (derivation, then MISS) are also done (protocol §5.9).
4. G4, 6b-10 on run 4 (Colab, minutes). The next free item; run it in the same session as the PM1 probe.
5. The G3′ local checks once its folder is downloaded: the exchange-field probe, the causality check and η.
6. CG8 on the baselines, once 6b-14 exists.
7. The overlap diagnostics D1, D2 and E4 (low priority).

**GPU queue (updated 2026-10-06; refinement work is high priority):**
1. ~~G3′~~ done: 52.90.
2. ~~**W1, the matched GPT-2 on WSD**~~ done 2026-10-06: settled 48.82 (−1.99% against cosine). W1.1–W1.3 HIT, W1.4 MISS; parity stands by the rule, on its boundary (protocol §5.17).
3. ~~The PM1 probe (8,000 steps)~~ done 2026-10-06: **79.78**, −14.7% against F3.1 and −3.8% against G2 at step 8,000. Perplexity PASS; pm_ clip MISS on the letter (a steady throttle at 0.3, no divergence). G4 (6b-10 on run 4) still open: Colab, minutes.
4. ~~PM1 at pm_ clip 1.0, 8,000-step probe~~ done 2026-10-07: 81.67, +2.4% against clip 0.3, so **the 0.3 arm continues** (about 11 h), Cell 6b-15 on both checkpoints flagged nothing, so it continues unchanged (protocol §5.15).
5. **The seed pair: G3′-s1, then G2-s1** (about 15 h each; protocol §5.19, Test 1).
6. **SR2 on F3.1** (protocol §5.19, Test 2). The substep is implemented and verified (2026-10-06); F3.1's Cell 0 + `LOWRANK_DAMPED_FLOW = True`.
7. The FO 2×2.
8. **An L=8 Fock live probe, then the full run** (about 5 h + 55 h). It combines SR-π.1, the depth trend, the matched-depth GPT-2 comparison and the floor question; its consolidated pre-registration is still to be written.
9. The CB series.
10. The rest of the SR series (SR1, SR4a, SR3, SR4b, SR5).

Queued, unscheduled: F1 (2× tokens), F2 (d = 768), W2 (GPT-2 WSD at 1.2e-3), G1 (parameter-matched GPT-2). *W2 recommended for promotion (2026-10-06, about 2.5 h): W1 shows GPT-2 trailing or level at the end of the stable phase and overtaking in the decay (0.27 against 0.18–0.22 nats), which is either architecture or peak learning rate. Only W2 separates them.*

## Open: what makes a model refinement-ready? — **opened 2026-10-06** (protocol §5.18)

- [x] RR-A, RR-B on G3′ (free) — done 2026-10-06. RR1 HIT (layer-0 destruction gate median 0.478, against 0.994 for G2). RR2 HIT (initial salience retained about 90× G2's). RR3 MISS, in the supporting direction (holding the bookkeeping makes G3′ worse, +111% → +503%, while it makes G2 slightly better). H-RR (accumulated register state means flow; reset-and-rewrite means maps) gains support.
- [x] **Pre-registered 2026-10-06 (protocol §5.19):** the refinement decomposition, with register reset for the Fock arms and stiff-mode discretisation for F3.1. Design principle: refinement readiness is required, for conservative arms together with conservativity, and for Fock arms on its own.
- [x] Seed tag (`SEED != 0` gives `s<N>`) and Cell 5b guard added; verified that seed-0 tags are unchanged.
- [ ] **Test 1, GPU, high priority:** G3′-s1 and G2-s1 (`SEED = 1`), about 15 h each. Predictions S1–S4.
- [x] **Test 2, code: SR2's exact damped-mode substep** (`lowrank_damped_flow`, Cell 0 `LOWRANK_DAMPED_FLOW`, tag `sr2`). Verified 2026-10-06: off is bit-identical to HEAD on G2, F3.1 and G3′; on passes eight checks, including RK4 agreement to 2·10⁻¹⁴ and the group property (`debug/verify_sr2_switch.py`).
- [ ] **Test 2, GPU: SR2 on F3.1** (F3.1's Cell 0 + `LOWRANK_DAMPED_FLOW = True`), about 14 h. Predictions E1–E2.
- [x] Hypothesis and evidence added to the G3 and G3′ model cards (2026-10-06).

## Queued: is there a shared floor near 50 PPL? — **opened 2026-10-06**

All three L=2 models with the register path settle within 2.5% of each other:

| model | settled |
| --- | ---: |
| G2 | 53.12 |
| G3 | 54.21 |
| G3′ | 52.90 |

The deeper models end near 50 at the same width and token budget: L=4 Fock at 50.10 and the 8-layer matched GPT-2 at 49.81. Is the floor set by depth, or by what every model shares, namely d = 384, the untied head and 532M tokens? The Fock models see about 7 tokens per parameter, against about 16 for GPT-2.

- [x] **F0 — free, CPU.** *Done 2026-10-06 (protocol §5.16): inconclusive. F3.1's and G2's fits are unidentified. The identified ones give G3 and G3′ a constant-learning-rate asymptote about 11% above L=4's, with overlapping intervals: a hint of depth dependence, not a floor. It says nothing absolute about what L=2 reaches with more tokens (the final decay alone lowers the loss well below L∞). F1 is the test. GPT-2 was excluded: its schedule is cosine.* Fit L(t) = L∞ + A·t^(−α) to each model's stable-phase eval curve (steps 3,000–21,000, before the WSD decay) and compare the L∞ estimates. Pre-register the reading before fitting.
  - If the L=2 models share an L∞ clearly above L=4's and GPT-2's, depth sets the floor.
  - If all extrapolate to about the same L∞, it is budget or width.
  - It runs after the G3′ diagnostics and upload.
- [x] **W1's stable-phase fit** (2026-10-06, protocol §5.17): GPT-2 on WSD has an identified L∞ of 56.7 PPL and then ends at 48.7, 8 PPL *below* it. A stable-phase L∞ is a constant-rate plateau, not a floor, and is not comparable across peak learning rates. This settles F0's reading: it is relative, and only within one schedule and peak.
- [ ] **F1 — GPU, about 30 h each, queued.** Double the token budget (65,000 steps) for G2 and the matched GPT-2.
  - If both improve by similar amounts, it is a budget floor.
  - If GPT-2 improves and G2 does not, it is an L=2 limit.
  - Pre-register before running.
- [ ] **F2 — GPU, most expensive, queued.** Width: L=2 Fock and GPT-2 at d = 768, a direct test of the shared d = 384 bottleneck. Pre-register before running.

## Scheduled: PM1 — bosonic Poisson-mode registers — **opened 2026-10-05** (protocol §5.15)

The alternative Fock mechanism that is bosonic and Poisson-mean by construction, prompted by DP1–DP3.

- [x] Implement `poisson_modes`: model, Cell 0 `POISSON_MODES` and `POISSON_MODE_CLIP`, tag `pm<K>`, Cell 5b guard, Cell 6 clip group. Verified 2026-10-05: off is bit-identical to HEAD on G2, F3.1 and G3′; on passes all seven checks (`debug/verify_pm_switch.py`).
- [x] **Probe:** F3.1's Cell 0 plus `POISSON_MODES = 64`, `PROBE_MAX_STEPS = 8_000` (about 4 h). **Revised 2026-10-06 from 3,000 steps, before any PM1 data:** G2's advantage over F3.1 is −3.2% at 3,000 (inside the ±2% eval scatter) and −11.4% at 8,000, so 3,000 is before any memory mechanism shows itself, and PM1's well depths start at 0. Gate: **at most 90.8 at step 8,000** (at least 3% better than F3.1's 93.57); 90.8–92.6 is a weak signal, full run at the author's discretion; above 92.6 fails. pm_ clip hits under 5%. The probe steps are not wasted: a passing run continues from `_step8000_probe_stop.pt`. Protocol §5.15. **Scored 2026-10-06: 79.78 at step 8,000** (F3.1 93.57, G2 82.92): perplexity PASS by 11 PPL, below G2 at every eval from 3,000. pm_ clip MISS on the letter: pre-clip norm above 0.3 on 77–86% of logged steps, steady at median 0.5, no divergence, SCAF CLEAN. Split accepted by the author; the full run is retuned (next item). Checkpoint `_step8000_probe_stop.pt` kept as the 0.3 comparator and fallback.
- [x] ~~**Full run** if the gate passes. Clear `PROBE_MAX_STEPS` and the same run continues, after the FO 2×2 unless moved.~~ Superseded 2026-10-06 by the retuned fresh run below.
- [x] **`POISSON_MODE_CLIP` reaches the tag** (2026-10-06): `pmclip<thr>` when modes are on and the clip is not 0.3, with a Cell 5b guard. Verified against HEAD: every existing arm's tag is unchanged, including the 0.3 probe.
- [x] **Clip-1.0 arm: 8,000-step probe, then the full run on the winning clip** (author's call 2026-10-06). F3.1's Cell 0 plus `POISSON_MODES = 64`, `POISSON_MODE_CLIP = 1.0`, `PROBE_MAX_STEPS = 8_000`. Rule at step 8,000 against the 0.3 probe's 79.78: at or below 81.4 → continue clip 1.0; above 81.4 or diverged → continue the 0.3 arm from its checkpoint; tag `…cgqk_norc_vplive_xilive_pm64_pmclip1_L2probe…`. Why 1.0: over the probe's steps 2k–8k the group would be clipped on 91% of steps at 0.3, 40% at 0.5, 11% at 0.75, 2% at 1.0. Pre-registered PC1 clip hits under 5%, no divergence (80%); PC2 step 8,000 at or below 81.4 (80%); PC3 at or below 78.2 (25%); PC4 settled below G2's 53.12 (50%). The Stage 2 table, RR-PM1 and RR-PM2 carry over. If it diverges, continue the 0.3 arm instead. Protocol §5.15. **Scored 2026-10-07: 81.67 at step 8,000** (+2.4% against the 0.3 probe; behind at every eval from 2,000). PC1 HIT, PC2 and PC3 MISS: **the 0.3 arm continues.** The tight clip helped (step-to-step renormalisation under AdamW, not an LR cut).
- [x] **Cell 6b-15 on both probe checkpoints** (2026-10-07): **no knob flagged on either** (step size, weight decay, occupation scale, placement). PM1 acts at layer 1 (attractive wells, force 5–6× the conservative force) and is off at layer 0; half-lives trained to ≤ 33 tokens; pm_depth carries 99% of the clip group. Decision: continue the 0.3 arm unchanged. Protocol §5.15.
- [ ] **PM1 full run: continue the 0.3 arm** from `_step8000_probe_stop.pt` to 32,500 (`POISSON_MODE_CLIP = 0.3`, `PROBE_MAX_STEPS = None`; about 11 h). Then score the Stage 2 table, PC4, RR-PM1, RR-PM2.
- [ ] **Post-run:** the repetition test, the depth signs and the DP3 rerun (protocol §5.15). Then the book: Remark 61 and §10.5.2, according to the decision rule.
- [ ] **Refinement readiness (added 2026-10-06, before the probe):** 6b-7 Gate 3 on the full run's best checkpoint, N = 3 at fixed T, policy `hold`. Pre-registered RR-PM1: at or below F3.1's +143% (60%); RR-PM2: below +600% (85%). `pm_depth` follows `_fom_policy_index` like `depth_code`, so no code change; φ is token-accumulated and unaffected by refinement. If SR2's E1 holds first, the full run goes on the corrected base (`LOWRANK_DAMPED_FLOW = True`, tag `pm64_sr2`), the author's call before launch. Protocol §5.15.

**GPU queue (2026-10-05):**
1. G3′ (running).
2. PM1 probe (about 1.5 h; *revised 2026-10-06 to 8,000 steps, about 4 h, see the probe item above*).
3. The FO 2×2.
4. PM1 full run, if gated in.
5. The CB series.
6. The SR series.

## Scheduled: CB-series — balancing the conservative and Fock paths — **opened 2026-10-03** (protocol §5.11)

The author's hypothesis: with the Fock path present, PARF's V_φ is starved, so the model gains PPL at the cost of conservativity. All switches are in Cell 0 and off by default; verified in `debug/verify_cb_switches.py`, where each neutral setting is bit-identical to G2. The baseline η (Fock increment / conservative step) on the Gen 2 no-exchange weights is 1.6 at layer 0 and 3.0 at layer 1.

- [x] **Stop-rule check:** G2 settled < 56.6. Otherwise the series does not run. *Passed 2026-10-05: G2 settled at 53.12.*
- [x] **CB0** (free): η on G2's checkpoint, and 6b-11 (inference slider) on G2. *Done 2026-10-05: η median 1.47 (layer 0) and 3.88 (layer 1), above 1 on every token, so both CB2 caps bind everywhere; slider 56.4 → 246.7 (4.38×), no knee, against 1.087× trained-without (protocol §5.11).*
- [ ] **CB2b** — `FOCK_BUDGET = 0.3`. **The key arm:** mostly conservative by construction.
  - *Added 2026-10-04:* G2 shows the register path amplifying refinement failure (Gate 3 +1,274% against F3.1's +143%). So every CB arm also reads **Gate 3 with the θ distribution**, and capping the Fock path is predicted to cut it sharply.
- [ ] **CB1** — `REVERSE_CHANNEL_WARMUP_STEPS = 20000`: Fock arrives late.
- [ ] **CB3** — `FOCK_GATE_L1 = 0.02`: a learned per-token gate, with ν = the share of exactly-conservative tokens.
- [ ] **CB2a** — `FOCK_BUDGET = 1.0`.

**CG8 reading on every SR and CB arm, and on the baselines** (protocol §5.12): does the geometry predict the model's own next-token errors beyond softmax entropy? Signals: the energy anomaly, the CG6 deflection, and η. Score: ΔAUROC over entropy alone.
- [ ] Implement 6b-14 (CG8 cell), evaluation only.
- [ ] CG8 on the baselines: F3.1, G2, L=4 live (free).

## Scheduled: FO-series — does OWT need second-order training? — **opened 2026-10-03** (protocol §5.13)

Scope: at the current damping regime (constant γ = 0.1, ζ ≈ 0.05, CfC low-rank, live gradients, L=2).
- [x] **Stage 0, corpus side** (C1–C5) — done 2026-10-04: OWT more long-range dependent; equal total I_pred; C3–C5 predictions missed (protocol §5.13 results): Zipf, the I_pred proxy, repetition autocorrelation, embedded-stream autocovariance, and the ξ filter-bank conditioning, on TinyStories vs OWT. Local, free.
- [x] **Stage 0, model side** (M1 anharmonic fraction ε, M2 inertial share) — done 2026-10-04: TinyStories ε = 0 (force constant across a step); OWT CfC ε 0.3–6. The full 2×2 runs, plus ε on SO-TS: on the TinyStories second-order anchor and Fock-G1, and on F3.1, L=4 live and (later) G2.
- [ ] Implement the **FO-a switch** (h_prev := h per layer, the Fock-G1 definition) behind Cell 0; verify it is bit-identical to G2 when off.
- [ ] **Stage 1:** FO-OWT, SO-TS, FO-TS (the 2×2 with G2). After G2–G4, before CB and SR.
  - FO-OWT: 32,500 steps.
  - SO-TS and FO-TS: **16,250 steps** (about 7 h each; amended 2026-10-04), then the settling check. If either fails, extend both equally.
  - Prerequisites: the full TinyStories token cache (one-time, on Colab), a corpus switch in Cell 0, and the FO-a switch (each verified).
- [ ] *FO-b (true overdamped step): only if FO-a shows a gap.*
- [ ] *γ(h) link: test the local-order prediction once a trained γ(h) arm exists.*

## Agenda: SR-series — settling, refinement and depth extension — **opened 2026-10-03**

Theory: book §8.9 (Props 44–45). Pre-registration: [`Depth_Ladder_and_Matched_Baseline_Protocol.md`](Depth_Ladder_and_Matched_Baseline_Protocol.md) §5.9. Baseline config: F3.1 (L=2 conservative-only, live V_φ + ξ), full runs.

> **Insight, 2026-10-04: refinement readiness tracks the π crossing** (protocol §5.9, SR-π).
>
> The L=4 Fock live arm refines far better than the L=2 Fock live arm (G2): Gate 3 at 1.5× the steps is +216% against +1,274%. Prop 44 amplifies the phase-sampled dissipation by θ/sin θ (θ = ω·Δt), most sharply near θ = π.
>
> - **L=4** trains its stiff modes at θ ≈ 2.35 (θ/sin θ = 3.3) and stays below π when refined (θ ≈ 1.57).
> - **G2** trains them at θ ≈ 3.40, just past π (θ/sin θ = −13.3). Refinement carries them back across π (θ ≈ 2.27), which flips the sign of the term the network learned to rely on.
> - **F3.1** also crosses π (4.32 → 2.88) but fails much less (+143%). So the crossing is not sufficient alone, and the register path amplifies it.
>
> What it suggests:
>
> 1. **A prediction to test.** Gate 3 cost tracks how close the trained ω·Δt sits to π and whether refinement crosses it. An L=8 model (ω·Δt ≈ 1.2) should be more refinement-ready still, approaching flow (SR-π.1).
> 2. **SR2 becomes a sharper test.** The exact damped flow removes the phase term altogether, so on G2's configuration it should cut Gate 3 far more than on F3.1's (SR-π.2). Run SR2 on both configurations. *(Superseded 2026-10-05: SR-π.3 missed, and the G2 arm is dropped.)*
> 3. **A cheap design lever.** Choose L or Δt so the stiff modes train below π. Since Δt = T/L at fixed T = 8, this is a choice of L. It becomes a standing rule (§7.5) once SR-π.1 or SR-π.3 confirms it.
>
> One seed per arm, medians over wide spreads (G2's θ spans 2.6–5.5 between p05 and p95), and "1.5×" means N = 3 at L=2 but N = 6 at L=4.

- [ ] **SR1** — palindromic step order (O half-steps around the kick). Code: reorder `_layer_step_langevin`. Cheapest; run first.
- [ ] **SR2** — exact damped-mode flow on the stiff subspace, constant γ. *Code done and verified 2026-10-06 (`lowrank_damped_flow`; see the refinement block).* Code: new joint substep in `cfc_baoab.py` (Prop 45, closed form for all damping regimes); replaces A·O·A on span(U) only. **One arm:** F3.1's configuration. *(The G2-configuration arm, SR-π.2, was dropped on 2026-10-05.)*
- [ ] **SR4a** — variable-step training at fixed T, N ~ U{2, 3, 4}. Code: per-batch step count and depth code indexed by time (as Cell 6b-7's `hold` policy).
- [ ] **SR3** — SR2 plus constant-ratio friction Γ = γ₀I + 2ζ*√(L/m) at ζ* = 1 (book Prop 43, eq. constant-ζ).
- [ ] **SR4b** — variable-step training at fixed Δt, N ~ U{2, 3} (the extension axis).
- [ ] **Each arm:** settled PPL, Gates 1–3 (6b-7), CG1 (6b-9), CG3 (6b-10), ω·Δt (6b-13), SCAF. **With every Gate 3 reading, report the trained θ = ω·Δt distribution (p05/p50/p95), θ/sin θ at the median, and whether 1.5× refinement crosses π.**
- [ ] **After SR2–SR4a:** score the decision rule and the Gate 1/Gate 3 rank-order check (§5.9).

Not gated on the live-gradient ladder, but competes with it for GPU time. Placement (updated 2026-10-04): after G2–G4 and the Zenodo upload, behind the FO 2×2 and the CB series. P2.2 has run as G2; the parameter-matched GPT-2 (G1) is deferred.
- [x] **SR-π.3** (free): on G2's checkpoint, the Gate 3 loss rise split by whether a token's stiff modes cross π under refinement (protocol §5.9, SR-π). *Done 2026-10-05: **MISS**. The ratio is 1.21 against the predicted 2, and the loss rise falls as θ rises; tokens that never pass π fail worst. Recommendation: drop SR-π.2's G2-configuration SR2 arm (about 14 h); SR-π.1 moves to about 30%.*
- [x] **SR-π.4** *(done 2026-10-05: MISS)*. The derivation shows the register push is a correctly scaled force (impulse ∝ Δt), so a rescaled push is not a fix. The finite-jump/projection hypothesis does not rank the arms by Gate 3. The L=2 Fock arms' layer-0 register increment (8–9× the state, against 1.3× at L=4) is the leading descriptive suspect (protocol §5.9).
- [x] **SR-π.3b** *(done 2026-10-05: MISS; holding the bookkeeping brings Gate 3 from +1,446% to +994%, only about 13% of the penalty)* (free, pre-registered 2026-10-05): refine with the register bookkeeping held to once per trained layer code. Prediction: G2's Gate 3 falls to +300% or below (50%). If it does, the Fock refinement failure is register bookkeeping, and a time-consistent register update is the fix to test first.
- [ ] ~~**SR-π.2**: SR2 on G2's configuration as well as F3.1's, testing whether removing the phase term cuts Gate 3 by more than half.~~ **Dropped 2026-10-05 (author's decision)** after SR-π.3's miss removed its rationale. SR2 runs on F3.1's configuration only.
- [ ] *SR-π.1: an L=8 Fock live arm (θ ≈ 1.2), testing Gate 3 ≤ +100%. About 55 h; after the CB series.*
- [ ] **SR5b** — thermal training, `LANGEVIN_T = 0.0022` (r = 0.3, calibrated on F3.1: `debug/calibrate_langevin_T_output.txt`). No code needed; tag `T0p0022`. Predicted to trade along the Gate 1 / Gate 3 frontier (65%).
- [ ] **SR5a** — `LANGEVIN_T = 0.00025` (r ≈ 0.1), predicted near-null.
- [ ] *SR5c (annealed thermostat, per-layer T → 0 with late damping): only if SR4b passes Gate 2; needs per-layer T in `ou_step`.*

## Scheduled, low priority: overlap-distance diagnostics — **opened 2026-10-04**

Source: `semsimula/docs/Overlap_Distance_in_Semantic_Simulation.md` (§10.5, §12). The book's §10.5.2 now carries the overlap formalism. These are its free checks on existing checkpoints: **evaluation only, no training.**

**Priority (2026-10-04):** below G3′, the FO 2×2 and the CB series. They run in any idle slot, alongside nothing urgent. They do not gate the book or any arm.

**Expectation, recorded in advance.** The formalism is a change of basis (overlap and Löwdin presentations are unitarily equivalent), so it cannot add expressiveness and changes no trained model. At d ≥ 768 it is fragile.

- **Width mismatch.** With anisotropic, learned per-well precisions, the shape prefactor (2σ_vσ_w/(σ_v²+σ_w²))^{L/2} suppresses overlaps strongly: a 10% width mismatch at L = 768 already gives about 0.18. Trained models are therefore expected to read G ≈ I.
- **Distance concentration.** The calibration κ ∼ 1/(s√(2L)) is already mirrored in the code: the well precision is capped at `2/d`, and `init_log_precision = −log d`.
- **Where the value lies, if any:** the register-redundancy diagnostic (D2) and, behind it, the log-det diversity regulariser (D4).

- [ ] **D1 — temperature-to-bandwidth calibration.**
  - Convert the trained creation-gate scales to bandwidths: κ_k² = 1/(2τ_k).
  - Under `cgqk` the learnable `logit_scale` replaces τ, so use τ_k = 1/logit_scale_k.
  - Map κ_k to semantic-space units through the singular values of W_Q and W_K^(k), and compare with the wells' κ.
  - Record the key-norm spread std_j‖k_j‖ / mean_j‖k_j‖. If it is small, a Gaussian-kernel gate is identical to the current one and that option is dropped.
  - Checkpoints: G2 (L=2 Fock live) and the L=4 live arm.
- [ ] **D2 — register redundancy against collapse.**
  - Compute λ_min, effective rank and det of the routing Gram G^(α) (Bhattacharyya: no bandwidth needed; also on sharpened α^β with β > 1, because diffuse rows overlap trivially). Also compute the content Gram G^(r).
  - Restrict both to the active registers, per layer and averaged over positions.
  - Compare the collapsed regime (pre-B1, creation entropy about 0.04) with the fixed one (B1+B2+B3), and test whether the effective rank drops *before* the creation entropy does along a training trajectory.
  - A lead time makes effective rank an early-warning diagnostic in its own right.
- [ ] **E4 — width heterogeneity** (revised 2026-10-04).

  **Question.** At d ≥ 768 the overlap of two Gaussian modes carries the shape prefactor (Bhattacharyya form, general covariances)

  π_vw = det(Σ_v)^¼ · det(Σ_w)^¼ / det((Σ_v + Σ_w)/2)^½,

  which, with isotropic widths that differ in all L dimensions, is ((2σ_vσ_w)/(σ_v²+σ_w²))^{L/2}. A 10% mismatch at L = 768 then gives about 0.18. Our wells are not built that way:
  - each precision is P = diag(a) + BBᵀ, with B of rank r = 4;
  - the diagonal is capped at `2/d`;
  - the stiff, anisotropic directions live in B (capped by `PRECISION_LR_MAX = 1.0`).

  If most wells' diagonals sit at or near the cap, the mismatch exponent counts only the up-to-2r directions where the two low-rank factors differ, not L/2. A 10% mismatch over 8 directions costs about 0.98, not 0.18. **Hypothesis:** the effective mismatch dimension is O(r), not O(L).

  **Measurement** (evaluation only; G2's checkpoint, d = 384; repeat at d ≥ 768 when such a checkpoint exists):
  1. **Contexts.** Our wells are functions of ξ, so draw 1,024 contexts ξ from the corpus (G2's validation tokens, the 6b-13 seed, every layer), and state that choice. At each context read every well's μ, a and B.
  2. **Exact prefactor π_vw** for all pairs of wells at the same context and layer. Compute it in log form with Cholesky or `slogdet` on the r-dimensional Woodbury forms, never on dense d×d inverses. Report the median, p10 and p90 over all pairs and over **nearest-neighbour pairs** (the nearest centroid by d_Σ).
  3. **Diagonal / low-rank split.** log π_vw = log π_diag + log π_lr.
     - log π_diag is the prefactor of the diagonal parts alone: Σ over i of ½ log(2√(a_vi a_wi)/(a_vi + a_wi)).
     - log π_lr is the remainder: the low-rank factors and their orientation.
     - Also report the share of diagonal entries within 1% of the cap, and the number of dimensions whose width ratio exceeds 1.1 (the *effective mismatch dimension*).
  4. **Centroid term**, for comparison: exp(−¼ Δᵀ((Σ_v+Σ_w)/2)⁻¹Δ) on the same pairs, which says whether proximity or width mismatch sets G.
  5. **Common-width alternative.** G under modes of one isotropic width σ = x*/2 at a reference κ (the median well κ), giving G = exp(−κ²‖μ_v − μ_w‖²). Report its spectrum (λ_min, effective rank) next to the well-tied G.

  **Predictions:**
  - effective mismatch dimension ≤ 2r = 8 for the median pair (55%);
  - median nearest-neighbour π_vw ≥ 0.5 (50%);
  - the low-rank term dominates log π, more than 50% of its magnitude (60%).

  **Decision rule:**
  - **Median nearest-neighbour π_vw ≥ 0.5:** keep the well-tied mode widths as written. The book's §10.5.2 footnote (per-type widths, with the prefactor) stands.
  - **Below 0.5:** adopt **common-width modes** in the formalism. That is one sentence in the §10.5.2 footnote and in the note's §9.6. Common widths keep G a genuine Gram matrix (positive semidefinite, so the Fock space stays well-defined), and the identity d_ov² = 2V/(𝔪υ²) then holds for the reference well.
  - **In either case:** do **not** drop the prefactor by convention. The location term alone is not guaranteed positive semidefinite, which would allow negative norms in the Fock space; if it is ever used as a similarity score, check PSD-ness explicitly. Do **not** regularise the model's widths to suit the formalism: the anisotropic V_θ is what the family's quality rests on (9.04 against 16.33 isotropic, TinyStories). And do not read hierarchy (broad against narrow wells) from overlaps, since nested Gaussians are nearly orthogonal at high d; use an asymmetric measure (the book's Experiment G4, or KL).

- [ ] *D3 (PPL against effective rank, across checkpoints) and D4 (the log-det diversity regulariser, λ_det ∈ {0, 1e-3, 1e-2, 1e-1}): **only if D2 shows signal.** D4 would be pre-registered first, on G2's configuration. It is independent of the CB series: CB caps the register path's push, while D4 keeps the registers distinct.*

## CLOSED: gradient starvation across the ladder — **opened 2026-09-30, closed 2026-10-05**

`attention_potential` with its learning signal restored (forward identical,
`relax_grad_path='live'`) settled at **61.11**, against 80.90 starved and 63.51
for the non-conservative `attention`. The same source-detach is used in **V_φ**
and **ξ** in every arm, which puts "V_φ is inert", the +31.3% price of the Fock
mechanism, and the queued factorial on the same starved convention.

Full programme, tiers and stop rules:
[`Gradient_Starvation_Investigation.md`](Gradient_Starvation_Investigation.md).

- [x] **Tier 0** (done 2026-09-30): offline gradient checks for V_φ, ξ and the score head on the no-exchange and conservative-only checkpoints. Starvation confirmed: exactly 0 gradient reaches earlier tokens through V_φ and ξ.
- [x] **Tier 1** (done 2026-10-01): `vphi_grad_path` / `xi_grad_path` switches, verified forward-identical, tagged `vplive` / `xilive`.
- [x] **Tier 2** (done 2026-10-01 to 10-04):
  - P2.1, conservative-only live: **127.73 at step 3,000, a MOVE** (criterion ≤ 154.1; parent 158.09, −19.2%).
  - P2.2, no-exchange live: run in full as G2.
- [x] **Tier 3/4** (done 2026-10-02 to 10-05): the full runs and their consolidation.
  - F3.1: **57.76**.
  - L=4 Fock live: **50.10**, parity with GPT-2 49.81.
  - G2: **53.12**.
  - G3: **54.21**.
  - All four are published on Hugging Face as Gen 3.
  - Consolidated in book v6, where the detached-source readings are withdrawn (Remark 103, `rem:gen2-gen3`).

**Closed 2026-10-05.** The live-gradient (Gen 3) convention is now the default for every new arm.

**Was paused behind it, now released:**
- **`splm-multixi`, `fock-splm`:** marked "moved to Gen 3" and unscheduled. Re-queue only if their questions remain open.
- **D1:** its premise, "depth does not pay", was a Gen 2 reading. Under live gradients L=4 is 5.7% better than L=2, so D1 needs re-justifying before it is scheduled.
- **Restating ladder numbers** in the cards and the book: done (Gen 3 cards, book v6).

---

## SUPERSEDED: the D-series — why depth does not pay — **2026-09-28, superseded 2026-10-05**

> **Superseded by Gen 3 (2026-10-05).** The premise below (L=4 behind L=2: 71.75 against 66.98) was measured under the detached-source (Gen 2) convention, whose readings book v6 withdraws (Remark 103). Under live gradients **depth pays**:
>
> | | L=2 | L=4 |
> | --- | --- | --- |
> | Gen 3 Fock live, settled | 53.12 (G2) | **50.10**: 5.7% better, at 1.9× the cost per step |
> | Gen 2, settled | 66.98 | 71.75 |
>
> So the "interior optimum" was an artefact of gradient starvation. Status of each item:
>
> - **D1** (L=4 at fixed dt = 4, T = 16): no longer motivated as a repair. The question it would still answer, whether T = 8 is saturated, is legitimate but low priority. It needs its own Gen 3 pre-registration before any GPU time. Unscheduled.
> - **D2** (loosen `depth_code`'s clip): its motivation is gone, and it is still blocked by the clip-tag defect below. Unscheduled.
> - **D3, D4:** withdrawn. Both were contingent on D1–D2 failing.
> - **"Back to the factorial":** superseded. The V_φ × Fock comparison on the starved convention is replaced by the Gen 3 ladder (F3.1 57.76 against G2 53.12 at L=2, with V_φ and ξ live).
>
> **The blocking defect below stays open as general hygiene.** It applies to any run that changes a clip threshold. The two new clip knobs differ: `RELAX_FIELD_CLIP` reaches the tag (`rfclip…`), but `POISSON_MODE_CLIP` does not, so only the default 0.3 is collision-safe. *`POISSON_MODE_CLIP` fixed 2026-10-06 (tag `pmclip<thr>`, PM1 block); the general defect stays open for the other clip knobs.*

**Original agenda, kept for the record:**

L=4 at matched T came in behind L=2 **on train as well as validation**, so it
is a fitting problem, not a generalisation one. At fixed T = 8 the ladder now
shows an **interior optimum**: L=1 → 87.09, **L=2 → 66.98**, L=4 → **71.75
settled** (+7.1%). This read "~85–90 projected" until run 4 settled on
2026-09-29; that projection ignored the WSD decay. Nothing in the framework
predicted an interior optimum. The ω·dt comparison scored the same day,
**HIT** (L=2 3.796 in [3.3, 4.2]): halving dt raised ω by only 21%, with full
compensation at layer 0 and none beyond it (protocol note, "Run 4 settled at
71.75").

Full design, pre-registrations and tag checks:
[`Depth_Ladder_and_Matched_Baseline_Protocol.md`](Depth_Ladder_and_Matched_Baseline_Protocol.md)
§3b.

- [ ] **D1 — L=4 at fixed dt = 4** (`LADDER_L=4`, `LADDER_T=16.0`), ~27 h.
      Pre-registered **62–70**. Tag derives to `L4probe`+`idt4`; no collision.
      Turnable quantity: whether T = 8 is already saturated.
- [ ] **D2 — loosen `depth_code`'s clip** 0.25 → 1.0. Zero new parameters.
      **Blocked on the tag fix below.** Pre-registered: >2% improvement if the
      clip binds; null means the channel is too small, not too throttled.
- [ ] **D3 — widen the depth conditioning.** Only if D2 is null. Keeps one
      potential, so the thesis is intact; changes shapes, so it tags itself.
- [ ] **D4 — untying.** Only if D1–D3 fail. 2.76× parameters, ~59% of the
      non-embedding model is V_θ alone. A new architecture with its own name,
      **not** a rung of this ladder.

**Decision 2026-09-28: D1 only, then back to the factorial.** D2 and D3 are
**parked behind runs 10 and 11**. They are reached only if D1 comes back
*positive* (materially better than 66.98), and even then the first job is
re-stating the headline ladder at fixed dt, not the repairs. A null or
negative D1 — the likely outcome on run 4's evidence — sends the queue
straight to the V_φ × Fock factorial, which is half-measured and fully
pre-registered. Completing a factorial beats repairing a rung that may be a
true negative.

**Fill the gaps with the free work**: the ω·dt offline comparison the moment
run 4 finishes (before D1 takes the machine), then C1; C2 and C7 need no GPU
and can run while D1 trains.

### ⚠ Blocking defect: clip thresholds do not reach the variant tag

`GRAD_CLIP_OVERRIDES` lives in **Cell 6**; `_variant_tag` is built in
**Cell 0**. Changing any threshold therefore produces a run that resolves to
**the same Drive folder** as its unchanged sibling, and Cell 2 will silently
resume from that sibling's checkpoint — the exact failure the `norc`, `ris`
and `zro` tag components were added to prevent. **This is live right now for
every one of the nine overrides**, not just `depth_code`.

- [ ] Add a conditional tag component for any override that differs from its
      default, plus a Cell 5b guard asserting the tag carries it. Do this
      **before** D2 or any C-series run that touches a threshold.

**Standing caution for the whole series.** More parameters have already made
this architecture worse once: the conservative twin added 589,825 parameters
and settled 20.8% worse (80.90 vs 66.98). Capacity is not a free axis.

---

## Open: are the per-group clip thresholds tuned? — **2026-09-28**

> **Re-measured on Gen 3 (2026-10-05): the confound persists and is somewhat stronger.** Source: the Cell 6 logs and the step-500 and best checkpoints.
>
> | arm | reverse_channel_scale is the top pre-clip group | median pre-clip norm | tanh(gate), step 500 → best |
> | --- | --- | --- | --- |
> | G2 (L=2 Fock) | 93% of logged steps | 1.90 = **19×** the 0.1 threshold | [0.063, −0.054] → [0.018, −0.015] |
> | G3 (L=2 + exchange field) | 86% | 1.70 = **17×** | [0.052, −0.046] → [0.028, −0.025] |
> | L=4 Fock | 95% | 2.60 = **26×** | [−0.074, −0.068, −0.071, −0.075] → [−0.003, +0.004, +0.004, −0.022] |
> | G3′ (to step 14,500) | 84% | 1.0 = 10× | — |
> | F3.1 (no reverse channel) | 0% | — | — |
>
> - **The direction-of-bias argument still holds under Gen 3.** The gate's magnitude falls in every arm (3.5× in G2; 3–20× per layer at L=4, where two layers also change sign), so a clip can only have slowed a decline.
> - **Book updated for v6.1 (2026-10-05):** `rem:gate-clipping` now carries the Gen 3 figures throughout (17–26×, the gate trajectories, C2's 0.020–0.045, and C1 as a fourth bound). Previously it quoted the Gen 2 figures: "median 10–20×" and "0.061 → 0.017". It should be updated to the Gen 3 ones in v6.1. The |m|/√v bound (C2) was measured on Gen 2 only.
> - **C1, C2 and C7 are still free and unrun, and can now run on the Gen 3 checkpoints** using the local harness (`debug/cb0_g2.py` already runs 6b-11 on CPU).


Every `GRAD_CLIP_OVERRIDES` value was set from L=8/L=16 forensics
(`depth_code` 0.5 → 0.25 on 2026-08-23 against the L=16 g0.1 OWT run; the
rest from the step-6435 and step-71194 captures). None was revisited for the
ladder's L=1/2/4 rungs.

**Measured, not assumed:** `reverse_channel_scale` is the top pre-clip group
on essentially every step of every arm that has a reverse channel, at a
**median 10–20× its 0.1 threshold** — 15× at L=2 and 16× at L=4 on the only
like-for-like window the logs share. `depth_code` is a warmup transient only,
peaking at 2×. The `norc` arm is the one completed arm where nothing is
meaningfully clipped.

**What is NOT true:** the effect is not depth-dependent. 1.50 at L=2 against
1.60 at L=4 is a 7% move where √L predicts 41%. So it is a *uniform* confound
across rungs, which leaves the ladder's internal comparisons intact and says
nothing kind about the absolute claims.

**Direction of the bias, from C0:** the gate *falls* 3.5–4× over training
(0.061 → 0.017 at L=2). So the clip, if it binds at all, has slowed a
**shrinking** gate — meaning the unclipped model has a **weaker** reverse
channel, not a stronger one. Any bias makes the mechanism look more important
than it is.

**Probe series and pre-registrations:**
[`Depth_Ladder_and_Matched_Baseline_Protocol.md`](Depth_Ladder_and_Matched_Baseline_Protocol.md)
§ "The C-series". C0 run; C1 (extend the E5 slider above λ = 1) and C2 (Adam
realised-step audit) are free and unrun; C4 (paired 500-step run) waits for
L=4; C5 (full re-run) only on evidence.

**Checklist:**

- [x] C0 — gate trajectory across checkpoints
- [x] C1 — `R11_LAMBDAS` extended above 1.0; pre-registered: PPL rises. *Done on Gen 3 G2, 2026-10-05: HIT. PPL rises monotonically above λ = 1 (+4% at 1.1, +39% at 1.25, +314% at 1.5); the gate is at its inference optimum, so the question is closed and C4 and C5 are not licensed.*
- [x] C2 — exact param→group map, then `|m|/√v` per clip group. *Done on Gen 3 G2, 2026-10-05: gate 0.020–0.045 against about 0.15 elsewhere; the clipped reverse-channel weights match the unclipped groups (0.153 against 0.154). The clip is cosmetic for the endpoint.*
- [ ] C3 — state that joint clipping of a scalar group is exactly an LR cut
- [ ] C4 — paired 500-step run, threshold 0.1 vs 2.0
- [ ] C5 — full re-run, only if C1 or C4 separate
- [ ] C6 — record the watchdog's structural blindness wherever run health is claimed
- [ ] C7 — clip-hit fraction vs the loss curve, all five arms: what the clips
      cost in **convergence** rather than accuracy. Free, no GPU; the per-group
      norms are already in every run log
- [ ] Book: "Optimisation and its confounds" subsection in §16, after the
      C-series reports (restructure plan §2b). §27's `rem:gate-clipping` is the
      interim disclosure

**Rule going forward.** A clip threshold inherited from a different depth,
width or mechanism is an untuned hyperparameter, not a constant. When a rung
changes any of those, either re-derive the threshold or record that it was
carried over unchanged and why.

---

**Forecast record**, kept because the two failures were systematic and point
in *opposite* directions:

| run | forecasts | actual | error |
| --- | --- | ---: | --- |
| 1.2e-03 | 70 -> 75 -> 70 -> 72.8 -> 69.8 | **66.98** | every point **high**, all for the same reason (§6.2) |
| 2.4e-03 | 63-66, centre 64 | **69.59** | **low, and wrong in direction** (§5 T0) |
| L=1 @1.2e-03 | 74-80 | **87.09** | **low**; assumed the L=1/L=2 gap would saturate like the attention gap — it widened through the decay (ladder §5.5) |
| L=2 `'attention_potential'` @1.2e-03 | **66, band 62-72** (revised from 75-82) | **80.90** | **MISS +22.6% — and the superseded band would have HIT.** The revision repaired a genuinely wrong mechanism claim, then adjusted the old number instead of re-deriving from the corrected reasoning, and moved toward an expected ordering. See the rule below |
| L=2 `'attention'` @1.2e-03 | **63, band 59-68** | **63.51** | **HIT, point-accurate** — error +0.51 (+0.8%). Built from `'none'`'s measured 10.8% LR transfer, discounted for the extra parameters, with "is 1.2e-03 past this arm's optimum?" named as the turnable quantity. It partly was: the arm gained 7.1%, which is the discount the band was widened for |
| L=2 no reverse channel | **105, band 85-140** | **87.93** | **BAND HIT** — the first. Point high by 16%, landing 3% above the lower edge. The named turnable quantity (V_phi's share once uncontested) was recorded as pointing to "the low end or below", and did |
| F1 per-token forcing, L=2 `'none'` (a distribution, not a PPL) | **>80% of tokens above 0.75, <5% below 0.25, bimodality <0.6** | **100.0% / 0.0% / 0.515** | **HIT on all three**. Built from two measurements on sibling arms: E1's average deflection of 1.09 on this arm, and the exchange arm's own F1 reading of UNIFORM. Interpolation between measured neighbours again, not extrapolation |

**The `attention_potential` row adds a rule the others do not cover.** When
a pre-registered band is revised because its stated reasoning was wrong,
**re-derive the number from the corrected reasoning rather than adjusting
the old number, and record what the superseded band would have predicted.**
Here the original band (75-82, point 78) rested on a mechanism claim that
was factually wrong, and would have hit 80.90; the corrected mechanism was
right and the number derived from it was wrong by 22.6%. A flawed argument
can support a correct prediction. The revision also moved the band toward
an ordering that had been suggested as expected and away from the answer,
which is the shape of anchoring and worth naming.

The three hit rows share a different method:
each named, in advance, a specific measurable quantity whose direction
would move the answer, and each was built from a *measured transfer* rather
than an extrapolated trend. The fifth row is the stronger case — a point
estimate accurate to 0.8% — because the quantity it named (the arm's LR
transfer) had already been measured on a sibling arm, so the forecast was
an interpolation, not an extrapolation. The F1 row is the same shape in a
different currency: a distribution rather than a perplexity, bracketed by two
neighbouring arms that had already been measured. **Naming the lever, and
anchoring on a measurement rather than a trend, is what the three misses
lacked.**

The first pair is the lesson (the third row is the same lesson from a third
angle: a saturation was assumed that did not occur). The first set
under-weighted a mechanism that was real; the second extrapolated that same
mechanism past the point where a *different* quantity turned. Both came from treating one measured trend as
the whole model. No forecast in this document should rest on a single
extrapolated quantity again — state which quantity could turn, and what
would show it turning.
