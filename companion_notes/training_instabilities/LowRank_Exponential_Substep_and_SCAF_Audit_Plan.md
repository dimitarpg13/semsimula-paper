# The Low-Rank Exponential Substep and the SCAF Audit Plan for precision_lr_max

**What this note is.** §30–§31 of the training-instabilities record: the concrete sketch of the low-rank exponential substep, and the SCAF Phase 7b/7c audit plan for tuning `precision_lr_max`. It was split on 2026-11-21 from [`CfC_BAOAB_Integrator_and_Mitigations.md`](https://github.com/dimitarpg13/semsimula-paper/blob/5551315/companion_notes/CfC_BAOAB_Integrator_and_Mitigations.md) (written August to October 2026). The text is that document's, unchanged except for its links, and keeps its section numbers, so a reference to "§N" anywhere in the record or in other notes finds its note through [the index](README.md).

## Contents

- [30. Concrete Sketch: The Low-Rank Exponential Substep](#30-concrete-sketch-the-low-rank-exponential-substep)
- [31. SCAF Phase 7b/7c Audit Plan for Tuning precision_lr_max (L=16, and now L=8)](#31-scaf-phase-7b7c-audit-plan-for-tuning-precision_lr_max-l16-and-now-l8)

---

## 30. Concrete Sketch: The Low-Rank Exponential Substep

This is the **as-built** description of `integrator='baoab_cfc_lowrank'`
(`cfc_baoab.lowrank_modes` / `lowrank_cfc_substep`,
`model_parf_multixi._layer_step_langevin`). It follows the code's
**aggregated-spring** structure and the PSD/impulse corrections of §29.2.

Freezing the coefficients $g_k, \mu_k, P_k$ at the current $h$ (the same
frozen-coefficient model `harmonic_terms` uses), the aggregate local force is
$f(h') = s - H h'$ with $H = D_a + L$, $D_a = \mathrm{diag}(\sum_k g_k a_k)$
(clamped diagonal precision) and $L = G G^\top$ the PSD low-rank part. Only
$L$ is stiff (its $\sigma_{\max}$ is unbounded, §28.2b); $D_a$ is bounded by
`precision_max`. So $L$ is the part to integrate exactly.

### 30.1 Why the exactly-integrated operator must be PSD

This is the correctness point the earlier sketch glossed, and the reason the
split is written the specific way it is. The frozen aggregate Hessian is

$$H = \sum_k g_k P_k = \underbrace{\mathrm{diag}\Big(\textstyle\sum_k g_k a_k\Big)}_{\text{from the diagonal precision } a_k} + \underbrace{\sum_k g_k B_k B_k^\top}_{= G G^\top},\qquad g_k \ge 0,$$

and $H$ is symmetric positive semidefinite (PSD): a nonnegative-weighted sum
of the PSD terms $P_k = \mathrm{diag}(a_k) + B_k B_k^\top$. So **every mode of $H$ itself is a genuine oscillator** — the question is only how to **split**
$H$ into pieces cheap enough to integrate exactly.

**The tempting split, and why it is wrong.** `harmonic_terms` already
integrates $\mathrm{diag}(H)$ exactly (per dimension) and demotes the rest to
the kick, so the obvious next move is "also rotate the leftover off-diagonal."
That leftover is

$$H_{\text{off}} = H - \mathrm{diag}(H) = G G^\top - \mathrm{diag}(G G^\top),$$

where the pure-$a$ diagonal has cancelled. One line of trace algebra kills the
idea. A matrix and its diagonal have equal trace, so

$$\mathrm{tr}(H_{\text{off}}) = \mathrm{tr}(G G^\top) - \mathrm{tr}\big(\mathrm{diag}(G G^\top)\big) = 0.$$

A nonzero symmetric matrix whose eigenvalues sum to zero must have **both a
strictly positive and a strictly negative eigenvalue** — so $H_{\text{off}}$
is **indefinite** whenever the coupling is nonzero (i.e. whenever the wells are
genuinely anisotropic). A minimal $2 \times 2$ witness with $G = (1, 1)^\top$:

$$L = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix},\ \mathrm{spec}(L) = \lbrace2, 0\rbrace \succeq 0 \quad\text{(PSD)};\qquad L_{\text{off}} = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix},\ \mathrm{spec}(L_{\text{off}}) = \lbrace+1, -1\rbrace\quad\text{(indefinite)}.$$

**Why an indefinite mode defeats the whole point.** A mode with curvature
(eigenvalue) $\lambda$ obeys $m \ddot\eta = -\lambda \eta$:

| mode curvature | equation of motion | exact solution | behaviour |
| --- | --- | --- | --- |
| $\lambda \gt 0$ | $m\ddot\eta = -\lambda \eta$ | $\eta_0 \cos(\omega t) + \dots,\ \omega = \sqrt{\lambda/m}$ | bounded rotation |
| $\lambda \lt 0$ | $m\ddot\eta = \lvert\lambda\rvert \eta$ | $\eta_0 \cosh\big(t\sqrt{\lvert\lambda\rvert/m}\big) + \dots$ | exponential blow-up |

So "exactly rotating" the modes of $H_{\text{off}}$ would *exactly integrate a
repeller* on its negative eigenvalues — amplifying, not taming, precisely the
failure CfC was built to avoid. In code the symptom is immediate:
`omega = (kappa / m).sqrt()` with $\kappa \lt 0$ is a `NaN`, or, if floored to
zero, silently mis-integrates a repeller as free drift.

**The principle.** PSD-ness of the *sum* $H$ does **not** imply stability of a
*split*. Once an indefinite factor is carved off and integrated on its own,
the composition inherits that factor's hyperbolic growth. Integrating $H$
*whole* would be unconditionally stable (all its modes are $\ge 0$), but that
needs a full $d \times d$ eigendecomposition, $O(d^3)$ per token — infeasible.
The low-rank structure only helps on $L = G G^\top$ (rank $\le Kr$, so the
Gram eigendecomposition is $O((Kr)^3)$); a diagonal-plus-low-rank matrix has no
cheap exact matrix function. Hence we must split — and **a split is stable
only if every exactly-integrated factor is individually PSD.**

**The PSD split (what the code does).** Keep both factors PSD:

$$H = D' + L,\qquad D' = \mathrm{diag}\Big(\textstyle\sum_k g_k a_k\Big) \succeq 0,\qquad L = G G^\top \succeq 0.$$

This moves the **entire** $B_k B_k^\top$ — *including its own diagonal*
$\mathrm{diag}(G G^\top)$ — into $L$, and correspondingly removes that piece
from the diagonal channel, which now carries only $a_k$. That is exactly why
`harmonic_terms_lowrank` returns $k_{\text{diag}} = \sum_k g_k a_k$ (pure $a$),
**not** the $\sum_k g_k\big(a_k + \mathrm{rowsum}(B_k^2)\big)$ that the
diagonal-only `harmonic_terms` returns. The bookkeeping has two jobs and does
both at once:

- **No double-counting.** If the diagonal channel kept the full $\mathrm{diag}(H)$ *and* $L$ carried its own diagonal, $\mathrm{diag}(G G^\top)$ would be integrated twice. Giving that diagonal entirely to $L$ counts it exactly once.
- **No indefinite remainder.** Because the off-diagonal is never separated from its stabilising diagonal, no indefinite operator is ever formed; each exactly-integrated factor ($D'$ diagonal-nonnegative, $L$ a Gram) is PSD, so each sub-flow is a bounded rotation.

$D'$ is bounded (clamped by `precision_max`), so it is safe as the cheap
explicit/diagonal channel; $L$ is the stiff part, integrated exactly on its
$\le Kr$ modes. Both PSD, so the impulse composition (§29.2) has **no**
hyperbolic factor — the only residual stability concern is the mild resonance
effect discussed under "Stability" below, not blow-up.

### 30.2 The four steps, as implemented

**Step 1 — expose the low-rank subspace.** `harmonic_terms_lowrank()` stacks
the per-well factors, weighted by $\sqrt{g_k}$, into
$G = [\sqrt{g_1} B_1, \dots, \sqrt{g_K} B_K] \in \mathbb{R}^{d \times Kr}$ (all
xi-channels aggregated), so $L = G G^\top$. `lowrank_modes` eigendecomposes
the small Gram $G^\top G = W \Lambda W^\top$ ($Kr \times Kr$); the eigenvalues
$\lambda_j$ **are** the curvatures of $L$ and the reconstructed left vectors
$u_j = G w_j / \sqrt{\lambda_j}$ are its modes. (Gram-eig rather than a raw SVD
of $G$ is the numerically stable route; the geometry $U, \kappa$ is
**detached** — a frozen Jacobian — because a rank-deficient Gram has
degenerate near-zero eigenvalues whose `eigh` backward is singular. The
substep stays differentiable in $h, v$ and the frozen force.)

**Step 2 — per-mode frequency.** Because $L$ (not the full $H$) is what the
fast flow integrates, the mode stiffness is simply
$\kappa_j = \lambda_j$, $\omega_j = \sqrt{\lambda_j / m}$ — **no** Rayleigh
quotient of $H$ and **no** mixing of the diagonal into the mode curvature
(that mixing was the "residual coupling" caveat of the original sketch; the
PSD split removes it). `lowrank_max_modes` optionally keeps only the stiffest
$q$ modes; modes with $\lambda_j$ below a floor are made inert.

**Step 3 — the exact fast flow (A-substep).** `lowrank_cfc_substep` advances
$T + V_L$ over the substep: a **free drift on the whole state** plus, on each
mode, the exact undamped rotation the diagonal channel already uses,

$$\begin{pmatrix} \eta_j' \\ \zeta_j' \end{pmatrix} = \begin{pmatrix} \cos(\omega_j \Delta t) & \sin(\omega_j \Delta t)/\omega_j \\ -\omega_j \sin(\omega_j \Delta t) & \cos(\omega_j \Delta t) \end{pmatrix} \begin{pmatrix} \eta_j \\ \zeta_j \end{pmatrix},$$

with $\eta_j = u_j^\top h$, $\zeta_j = u_j^\top v$. The increments are written
back force-based (via the frozen force $f_L = s_L - L h$, $s_L = G (G^\top\mu)$)
so no fixed point $h_\ast = L^{-1} s_L$ or division by $L$ is formed —
identical to `cfc_substep`'s treatment of the diagonal. The complement of
$\mathrm{span}(U)$, where $L$ exerts no force, is left to the free drift.

**Step 4 — everything soft goes to the explicit kick.** The B-kick carries
$f_\theta + f_\phi - f_L(h_{\text{mid}})$: the full V_theta force **minus** the
frozen low-rank force the A-substep already integrates. What remains is the
clamped diagonal spring $D_a$, $V_\phi$, and the nonlinear V_theta residual
(the variation of $g_k, \mu_k, P_k$ with $h$) — all of bounded curvature, so
none is a blow-up channel. The total force field is preserved exactly (tested
to $O(\Delta t^3)$ agreement with plain BAOAB).

**Stability (impulse / RESPA, corrected).** The fast flow is a bounded
rotation on the modes for any $\omega_j$ — as a *standalone* map there is no
$\omega_j \Delta t \lt 2$ wall. But this is an **impulse / multiple-time-step**
composition (fast flow $\Vert$ soft kick), so it is *not* unconditionally
A-stable: it has narrow **resonance instabilities** at $\omega_j \Delta t
\approx k\pi$. Between resonances it is stable at any stiffness; the O-step
friction $e^{-\gamma \Delta t}$ and the nonlinear residual damp the resonances
in practice. §29.3's `precision_lr_max` keeps $\omega_j \Delta t$ from running
away, so the resonances stay shallow — the two mitigations are complementary
for exactly this reason. A unit test (`test_cfc_baoab.py`) confirms the scheme
survives a curvature that overflows the explicit step, at a non-resonant
$\omega \Delta t = 4.7$.

**Cost.** One eig of the $Kr \times Kr$ Gram matrix ($O((Kr)^3)$, with the
$O(d (Kr)^2)$ Gram formation dominating) plus a few rank-1 projections, per
token; with $Kr \approx 32$ (or $n_{\text{ctx}} Kr$ aggregated) and $d = 384$
this is small next to the forward pass. `lowrank_max_modes` bounds it further.

**Caveats.**

1. Freezing $g_k, \mu_k, P_k$ at the current $h$ makes this a local exponential (Rosenbrock-type) step: it integrates the frozen linearization exactly, and the leftover nonlinearity (Step 4's residual) stays explicit but non-stiff.
2. The frozen mode geometry ($U, \kappa$) is detached, so parameter gradients flow through the frozen *force* magnitude but not through the eigendecomposition. This is the standard exponential-integrator treatment and is why the `eigh`-degeneracy backward is never hit; the force field (and hence the loss) is still exact.
3. The fast flow is the A (drift) substep and the O (thermostat) and B (kick) are unchanged, so the BAOAB splitting structure — and its $T=0$ sampling properties — are preserved.

---

## 31. SCAF Phase 7b/7c Audit Plan for Tuning precision_lr_max (L=16, and now L=8)

**Status: §31.3 IMPLEMENTED (27 August 2026); §31.2/§31.4/§31.5 not yet
run.** This section captures the audit methodology for choosing a
`precision_lr_max` value on the live g0.1/d=384 `baoab_cfc` runs, ahead of
the `baoab_cfc_lowrank` + `precision_lr_max` vs. `baoab_cfc` baseline A/B
flagged as the next step in §29.7. It is independent of the depth-probe
track in §24.4's August update: that track asks whether shortening $L$
suppresses the burst; this track asks whether bounding $B_k$'s curvature
suppresses it *without* touching $L$.

**Scope broadened to L=8.** §31 was originally scoped to the live L=16
run only, treating the L=8 depth probe as a separate axis (§24.4). The
L=8 probe has since produced its own escalating hard-watchdog evidence
that the same mechanism is at work there too: a step-32,139 trigger
(pre-clip grad norm 701.1; top groups `E`/`P`=413.2, `creation_gate`=259.0,
`reverse_channel_scale`=243.3, `register`=200.9, `depth_code`=158.5) was
followed, only ~2,000 steps after reload, by a step-34,091 trigger nearly
an order of magnitude worse (5,864.9; `P`/`E`=3,106.9, `depth_code`=2,805.1,
`reverse_channel_scale`=2,560.5, `creation_gate`=2,354.4, and — notably
larger than the §27.1 "mid-pack" reference value of 502.0 — `V_theta`=1,073.0).
Both reloaded to the step-27,000 checkpoint (PPL 100.47), with no net
progress across the ~8,000 intervening steps. The escalating severity
(701 → 5,865 at the *same* reload point) is the qualitative signature
§28.6 predicts: `depth_code`/embedding perturbations get a second-order
amplification route through $B_k$'s still-growing curvature, so the same
proximate trigger produces a larger excursion the longer training
continues past it. §31.2's bracketing protocol therefore now has a
concrete L=8 candidate pair on hand: the step-27,000 best checkpoint
(healthy) against the `..._step32139_prereload.pt` /
`..._step34091_prereload.pt` snapshots (spike-regime) that the watchdog's
`_reload_best()` already wrote to Drive — no separate capture run needed.
A lighter-weight, no-`scaf`-dependency route to the same
`sigma_max(B_k)^2` percentiles is now also available directly in the
training notebook (`sigma_lr_report`, Cell 6b-2, added alongside
`stiffness_report`), for a quick in-Colab check against these exact
checkpoints ahead of, or instead of, the full offline SCAF audit below.

### 31.1 Goal

Pick a `precision_lr_max` value (the $\sigma^2$ budget on $B_k$'s
spectral norm enforced by `_bound_lowrank()` in
`model_aniso_gaussian_vtheta.py`, §29.3) from the *empirical*
$\sigma_{\max}(B_k)^2$ distribution actually reached on the g0.1/L=16
run, rather than guessing a number. Too tight a budget flattens the well
geometry and costs modelling capacity; too loose a budget doesn't touch
the runaway tail that caused the burst.

### 31.2 Checkpoints to bracket: healthy vs. spike-regime

A single checkpoint's stiffness distribution is not enough — the budget
should sit **between** the healthy bulk and the runaway tail, so both
ends need to be measured on the same run:

- **Healthy end.** A checkpoint from before the burst — e.g. the
  step-5,500/6,000 periodic or best checkpoint (val PPL 171–188, grad
  norm 0.8–1.4 in `training_log.jsonl`).
- **Spike-regime end.** A checkpoint from *during* the burst (steps
  6,297–6,676). The ordinary "best" and periodic-grid checkpoints are
  unreliable for this: `best_val_ppl` was regressing through that window
  (176.88 → 207.11), so no new "best" checkpoint was written there, and
  the periodic grid may not land inside a ~380-step window. The training
  loop's `_prereload` snapshots (`_reload_best()` in the notebook's Cell
  6, `tag_suffix='_prereload'`, capped by `PRERELOAD_SNAPSHOT_MAX_KEEP=5`)
  exist for exactly this reason — `GRAD_NORM_HARD_TRIGGER=500.0` fires
  unconditionally on any raw pre-clip grad norm above 500, and steps
  6,407 / 6,435 / 6,676 (pre-clip totals 1,816 / 2,402 / 3,337) all cross
  it — so the run's Drive `checkpoints/` folder should contain
  `..._step64xx_prereload.pt` / `..._step66xx_prereload.pt` files
  capturing the model state at (or immediately after) the worst moments
  of the burst. **Check for these first**, before falling back to the
  nearest periodic checkpoint.

Audit both ends with the same `StiffnessProbe` configuration so the
percentile ladders are directly comparable.

### 31.3 Required SCAF change: raw sigma_max(B_k)^2 percentiles

> **Status: IMPLEMENTED (27 August 2026).** All three items below are
> done, on the `stiffness_audit` branch of `semsimula-scaf`; the full
> suite (188 passed, 7 skipped, including the new test) is green.

`StiffnessProbe` (`semsimula-scaf/src/scaf/probes/stiffness.py`,
`stiffness_audit` branch) already computed what's needed internally, in
`weyl_upper_bound()`: `sigma_max_sq = torch.linalg.eigvalsh(gram)[..., -1]`
is the per-well, per-token $\sigma_{\max}(B_k)^2$. But it was only ever
folded into the aggregate Weyl bound
`k_weyl = sum_k g_k * (max_i a_k[i] + sigma_max_sq)` and reported as
`eig_median` / `eig_p90` / `eig_p99` / `eig_p999` / `eig_max` **after**
conversion to $\omega \Delta t$ — never as a raw, un-aggregated,
un-converted $\sigma^2$ value. `precision_lr_max` is exactly that raw
quantity, so the probe needed a small, additive change, now made:

1. `weyl_upper_bound()` takes a new `return_sigma_lr: bool = False`
   argument; when `True` it returns `(k_weyl, sigma_max_sq)` instead of
   just `k_weyl`, exposing the per-well, pre-aggregation
   $\sigma_{\max}(B_k)^2$ (shape `(..., K)`) alongside the existing
   aggregate bound, with the default-`False` call site unchanged.
   `StiffnessProbe.run()` now collects it into its own `sigma_lr_blocks`
   list alongside `eig_omega_dt_blocks`.
2. The `detail` dict now emits `sigma_lr_p50` / `sigma_lr_p90` /
   `sigma_lr_p99` / `sigma_lr_p999` / `sigma_lr_max` (reusing the existing
   `_quantiles()` helper) — the same percentile ladder already used for
   `omega_dt` and `eig_omega_dt`, just in raw $\sigma^2$ units instead of
   $\omega \Delta t$ units.
3. `tests/test_stiffness_probe.py` gained
   `test_return_sigma_lr_reproduces_planted_spectral_norm`, a synthetic
   two-well, rank-2 case with a diagonal Gram (so the top singular value
   is exact by inspection: $\sigma_{\max}^2 = 25$ and $9$ for the two
   wells) asserting `weyl_upper_bound(..., return_sigma_lr=True)`
   reproduces both planted values and leaves the aggregate `k_weyl`
   bit-for-bit unchanged from the non-`return_sigma_lr` call.

This was purely additive (new `detail` keys and an opt-in return-value
change only) — no existing probe output changed, so nothing downstream
(the OWT g0.1/g0.3 Phase 7b/7c reports already published on HF) needs
re-validation.

### 31.4 From percentiles to a precision_lr_max budget

Once both checkpoints report `sigma_lr_*`:

1. Confirm the qualitative story first: `sigma_lr_p99` / `sigma_lr_max`
   should be visibly larger on the spike-regime checkpoint than the
   healthy one, and the gap between `eig_p99` and `p99` (Weyl vs.
   diagonal-only $\omega \Delta t$) should be the dominant contributor to
   instability on the spike-regime side. If it isn't, $B_k$ growth isn't
   actually the driver of *this particular* burst and `precision_lr_max`
   is the wrong lever for it (the depth-cascade track of §24.4 would be
   the more relevant explanation).
2. Set the budget **above** the healthy checkpoint's `sigma_lr_p95`–`p99`
   (not its median — clamping there would flatten the well geometry
   every well relies on) and **below** the spike-regime checkpoint's
   `sigma_lr_p99` / `max`.
3. Remember the cap is on $\lVert B_k \rVert_F^2 \ge \sigma_{\max}(B_k)^2$
   (conservative by up to a factor `rank` — 4 for this run's
   `ANISO_RANK`), so the *effective* spectral cap achieved is somewhat
   tighter than the nominal `precision_lr_max` number; bias the choice
   slightly upward from step 2's interval to compensate.

### 31.5 The A/B run

With a `precision_lr_max` value in hand, run the comparison already
flagged as the next step in §29.7's status line: `baoab_cfc_lowrank` +
tuned `precision_lr_max` vs. the current `baoab_cfc` baseline, at
g0.1/d=384, watching whether the burst signature shrinks or disappears —
the 6,297–6,676-style E/P/`depth_code` bursts on **L=16**, or the
27,000-onward 32,139/34,091-style bursts on **L=8** (§31 preamble; no
longer treated as a separate, independently-tracked axis from §24.4's
depth-probe track now that it has produced its own escalating
hard-watchdog evidence of the same mechanism).

### 31.6 Status

**§31.3 done; L=8's §31.2/§31.4 done (§31.7); §31.5 and L=16's §31.2/§31.4
not yet started.** The SCAF probe change (`sigma_lr_*` percentiles, §31.3)
landed 27 August 2026, and an equivalent no-`scaf`-dependency diagnostic
(`sigma_lr_report`) was added directly to the training notebook (Cell
6b-2) for a quicker in-Colab check. Two candidate runs were in scope to
bracket:

- **L=16** (the original scope): still needs the g0.1/L=16 Drive folder
  inspected for `_prereload` snapshots in the 6,297–6,676 range.
- **L=8** (added this update, §31 preamble): the bracket pair was already
  on hand and needed no separate capture run —
  `..._best.pt` at step 27,000 (healthy) against
  `..._step32139_prereload.pt` / `..._step34091_prereload.pt`
  (spike-regime), all on the live `..._L8probe_..._g0.1_baoab_cfc` Drive
  folder. **This bracket has now been run through §31.4's logic — see
  §31.7 — and the answer is "no valid budget window; leave
  `PRECISION_LR_MAX = None`."**

§31.5's A/B (`baoab_cfc_lowrank` + a tuned budget, vs. the `baoab_cfc`
baseline) remains available as an opt-in notebook config change
(`INTEGRATOR`, `PRECISION_LR_MAX` in Cell 0) once/if the L=16 bracket
produces a workable window, or as a diagnostic falsification run on L=8
using the non-binding value discussed in §31.7.

### 31.7 L=8 budget selection: the recipe's own preconditions fail — leave `PRECISION_LR_MAX = None`

Running the three checkpoints (§33.1) through §31.4's two-step recipe:

**Step 1 (qualitative gate) fails.** §31.4 step 1 requires `sigma_lr_p99` /
`sigma_lr_max` to be *visibly* larger on the spike-regime checkpoints than
on the healthy one. They are not: the largest gap is +24% (p50, and only
at step 32,139) and the *p99/max* gap that step 2 actually keys off is a
near-flat +1.0% to +14.5% (§33.1's table). Per the recipe's own written
rule, this means "$B_k$ growth isn't actually the driver of *this
particular* burst and `precision_lr_max` is the wrong lever for it" — i.e.
the recipe self-terminates here, matching §33's independent conclusion via
the per-group grad log and the depth-cascade mechanism.

**Step 2 (the interval) is vacuous even if forced.** Attempting it anyway
for completeness: the lower bound ("above healthy's p95–p99") is
$\approx 1{,}050$–$1{,}100$; the upper bound ("below spike-regime's
p99/max") should be the *tighter* of the two spike checkpoints, and here
is the decisive number — **the healthy checkpoint's own `max` (6,364.81)
is essentially identical to spike_34091's `max` (6,427.16, +1.0%) and only
14.5% below spike_32139's `max`.** The long tail of $\sigma_{\max}(B_k)^2$
is not a spike-exclusive phenomenon; it is already present at the best
checkpoint the run has ever produced (PPL 100.47). Any budget tight enough
to bind on the spike tail (roughly below 6,400–7,300) necessarily clips
the healthy checkpoint's own tail too — there is no interval that
isolates "runaway" from "healthy," because on this evidence there isn't
one.

**Recommendation for the live L=8 run: leave `PRECISION_LR_MAX = None`.**
Picking any value in the only mathematically available window
(roughly 1,100–6,400, and recall §31.4 step 3: the true budget should be
biased *upward* from a naive read of that window because the Frobenius
cap is conservative by up to `rank`$=4$) would flatten well geometry that
the healthy checkpoint is actively using — a real cost in modelling
capacity — for a mechanism §33.2 has already shown is not the bottleneck.
This is consistent with, and sharpens, §33.2's "sound hygiene but not the
lever" framing: it isn't just that `precision_lr_max` is a weak lever
here, it's that this particular run's data contains no value that would
act as a *targeted* one.

**If a falsification A/B (§31.5) is still wanted**, run it with the
understanding that any chosen value is a blunt, non-targeted probe rather
than a tuned fix — e.g. `PRECISION_LR_MAX` $\approx 2{,}500$ (comfortably
above every checkpoint's `p99`, so the ordinary bulk is untouched, while
still meaningfully compressing the `p99.9`/`max` tail on *all three*
checkpoints, healthy included) — and expect, per §33, little to no change
in the burst rate. A clean negative result there would be further
confirmation, not a surprise.
