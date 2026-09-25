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
| `LANGEVIN_T` | 0.0 | untested | thermostat off |
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
| 2026-09-25 | architecture: **L=2 `'none'`, reverse channel off** (ladder run 8, an architecture control, not a tuning run) | full 32,500 | **87.93 settled**, +31.3% vs 66.98 with the mechanism on; ratio vs GPT-2 1.765. **The inference-time ablations overstated the mechanism by 2.9x in PPL ratio, 4.9x in nats.** Clip 2.6% vs 0.0%. Ladder protocol §5.6 |
| 2026-09-24 | depth: **L=1** `'none'` @1.2e-03 (ladder run 7, not a tuning run) | full 32,500 | **87.09 settled**, +30.0% vs L=2 at the same LR; ratio vs GPT-2 1.748. Decay gain 11.2% vs 19.3% at L=2. Clip 2.6% vs 0.0%. Static register bank (§7.3a). Ladder protocol §5.5 |

**Forecast record**, kept because the two failures were systematic and point
in *opposite* directions:

| run | forecasts | actual | error |
| --- | --- | ---: | --- |
| 1.2e-03 | 70 -> 75 -> 70 -> 72.8 -> 69.8 | **66.98** | every point **high**, all for the same reason (§6.2) |
| 2.4e-03 | 63-66, centre 64 | **69.59** | **low, and wrong in direction** (§5 T0) |
| L=1 @1.2e-03 | 74-80 | **87.09** | **low**; assumed the L=1/L=2 gap would saturate like the attention gap — it widened through the decay (ladder §5.5) |
| L=2 no reverse channel | **105, band 85-140** | **87.93** | **BAND HIT** — the first. Point high by 16%, landing 3% above the lower edge. The named turnable quantity (V_phi's share once uncontested) was recorded as pointing to "the low end or below", and did |

The fourth row is the first hit, and the reason is worth keeping: it was
the only forecast that named, in advance, a specific measurable quantity
whose direction would move the answer within the band — and that quantity
moved as described. Naming the lever, not widening the band, is what made
it work.

The first pair is the lesson (the third row is the same lesson from a third
angle: a saturation was assumed that did not occur). The first set
under-weighted a mechanism that was real; the second extrapolated that same
mechanism past the point where a *different* quantity turned. Both came from treating one measured trend as
the whole model. No forecast in this document should rest on a single
extrapolated quantity again — state which quantity could turn, and what
would show it turning.
