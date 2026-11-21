# The log_tau Runaway: the Record

**What this note is.** §49–§51 of the training-instabilities record: the register-temperature (`log_tau`) runaway as it was investigated; the settled account is the spin-out [Register_Temperature_Instability_in_the_Fock_Creation_Gate.md](Register_Temperature_Instability_in_the_Fock_Creation_Gate.md), and these sections are stubs that point into it. It was split on 2026-11-21 from [`CfC_BAOAB_Integrator_and_Mitigations.md`](https://github.com/dimitarpg13/semsimula-paper/blob/5551315/companion_notes/CfC_BAOAB_Integrator_and_Mitigations.md) (written August to October 2026). The text is that document's, unchanged except for its links, and keeps its section numbers, so a reference to "§N" anywhere in the record or in other notes finds its note through [the index](README.md).

## Contents

- [49. The log_tau Runaway: One Register's Temperature Is Diverging From the Pool, and Two Falsified Predictions on the Way to Finding It](#49-the-log_tau-runaway-one-registers-temperature-is-diverging-from-the-pool-and-two-falsified-predictions-on-the-way-to-finding-it)
- [50. Does QK-Normalisation Actually Break the Runaway? A Gradient-Flow Analysis of the Creation Gate's Temperature](#50-does-qk-normalisation-actually-break-the-runaway-a-gradient-flow-analysis-of-the-creation-gates-temperature)
- [51. Most of the Measured log_tau Drift Was Weight Decay on a Log-Parameterised Temperature](#51-most-of-the-measured-log_tau-drift-was-weight-decay-on-a-log-parameterised-temperature)

---

## 49. The `log_tau` Runaway: One Register's Temperature Is Diverging From the Pool, and Two Falsified Predictions on the Way to Finding It

> **Consolidated write-up.** §49, §50 and §51 were written incrementally as
> the investigation ran, and each one partially revises the one before.
> `Register_Temperature_Instability_in_the_Fock_Creation_Gate.md` presents
> the settled version as a single technical report, and adds the analysis
> none of these sections contains: the saddle-node existence condition and
> one-nat stability criterion for the interior equilibrium, and an estimate
> of the covariance coefficient $C$ from the observed drift showing this
> run's decay strength sits at 82% of the threshold required for an
> equilibrium to exist at all. Read that document first if you want the
> conclusions; read these sections for the order in which they were reached.

**Stubbed 2026-09-11 (see §52.2).** §48.8 proposed a mechanism and stated a
falsifier in advance; the falsifier fired, and so did a second one proposed
to explain the first failure. Both were wrong about *where* to look while
the underlying physics — §48.8's temperature derivative — was right all
along. Two probes made the correct axis visible: `probe_gate_saturation`
(per-register × per-layer clamp occupancy, scaled-score maxima, salience,
$\tau$) and `sweep_log_tau_history` (the same read out of every
`_spikebatch.pt` bundle on disk). Full derivation: report §3.

### 49.1 Falsified prediction 1: the hot row is not a score outlier

§48.8 predicted the gradient-hot row would show outlier creation-gate
scores. It did not — at step 71,194 the implicated row ranked 26th of 32
by peak scaled score, near the *bottom* of its own batch. Report §4.1.

### 49.2 Falsified prediction 2: the readout's clamp is not gating the gradient

The second guess — that the readout's `clamp(max=40.0)` was gating which
rows contribute gradient — also failed: only 2.3–3.7% of scores ever reach
the ceiling batch-wide, nowhere near dominant. Report §4.2.

### 49.3 The mechanism: register 14 has the coldest temperature in the pool

Both falsifications pointed one axis over, from rows to registers: register
14's learned temperature was the coldest in the pool (5.21–5.23 against a
median of 6.39–6.48), making its scaled scores an order of magnitude larger
than every other register's and giving it 97.8% (step 70,522) then 100.0%
(step 71,194) of `log_tau`'s entire gradient. Report §5.

### 49.4 Why the row-level test had no power: a lesson about aggregating statistics

§48.8's test statistic maxed the peak scaled score *over all 32 registers*
before comparing rows — since register 14 dominates that maximum in every
row, the statistic was near-constant and structurally could not have
separated rows regardless of the true mechanism. Report §4.3.

### 49.5 It is a feedback loop, and the history shows it running

The original reading, at the time this was written: both measured
gradients on register 14 were positive, so descent lowers $\log\tau_{14}$
further — self-reinforcing divergence, tracked declining over 3,672 steps
(rank 30 → 31 → 32 of 32). **Revised by §51.3**: roughly 70% of that
measured drift turns out to be weight decay on a log-parameterised
temperature, not this loop; what survives is the *differential* between
register 14 and a pool whose median gradient was actively resisting decay.
Report §5 (mechanism) and §8.3 (the correction).

### 49.6 What $\tau$ does *not* explain: the 211x

Between the two measured events $\tau_{14}$ barely moved (5.2310 → 5.2254)
while the gradient rose 211× (1.6326 → 345.3764) — score magnitude sets
*which* register is exposed, not the size of any one event. The upstream
backward signal is the rest of the story: `reverse_channel_scale`'s
gradient was 94.6–99.9% concentrated in layer 0, the one layer with zero
clamp saturation. Report §7.2.

### 49.7 Token degeneracy: falsified as a necessary condition, and one metric caught what the other missed

**Not carried into the report; kept here in full.** `decode_hot_rows` read
the three implicated rows directly. Two of three sit on the *non*-degenerate
side of their batch by `max_repeat_run` and `unique_token_ratio` alike —
degeneracy is not a necessary condition for a row to be implicated. The
third ranks 3/32 by `unique_token_ratio` (a heavily templated list of
"ARGUMENT FROM X" items) but only 14/32 by `max_repeat_run`, because that
metric only counts back-to-back identical tokens and is structurally blind
to phrase-level repetition — so some of the earlier §38.4/§39 negative
evidence built on `max_repeat_run` alone is weaker than it looked.

### 49.8 Mitigation 1 (config-only, resumable in place): give `log_tau` its own clip group

`log_tau` shared the `creation_gate` clip group with `W_Q`/`W_K`/`W_V`; at
step 71,194 it alone was 345.38 of the group's 432.16 norm, cutting the
projections' effective update to about 60% of normal. Splitting it into
its own override group (inserted *before* `'creation_gate'` in
`GRAD_CLIP_OVERRIDES`, since substring matching returns the first hit)
stops that collateral damage — it does not slow the drift itself, since
Adam is close to scale-invariant per parameter. Report §11.1.

### 49.9 Mitigation 2 (live monitor) and mitigation 3 (forward-side, needs a fresh arm)

A free `tau_min`/`tau_argmin`/`tau_median` monitor (one `exp()` over 32
elements, no batch needed) was added at `LOG_INTERVAL` cadence, stated as a
falsifiable prediction in advance. Two forward-side hardening options were
specified but deferred as needing a fresh arm: mirroring `ReverseChannel`'s
QK-normalisation, or a simpler clamp on `log_tau` itself.
**§50 supersedes the framing here** — it shows these two options are not
alternatives, and finds a defect in the first implementation of option 1.
Report §11.2 (monitor) and §11.5 (QK-norm).

---

## 50. Does QK-Normalisation Actually Break the Runaway? A Gradient-Flow Analysis of the Creation Gate's Temperature

**Stubbed 2026-09-11 (see §52.2).** §49 established the drift empirically
without establishing *why* the proposed fix would stop it. This section
derives it. The headline result: the temperature gradient is exactly a
**covariance**, the loop it drives is a **Riccati equation** with
finite-time blow-up, and — the result that overturned the original
mitigation plan — **QK-normalisation does not change any of this**. What it
buys is turning the loop's endpoint from an unbounded numerical accident
into a chosen constant, and only when the clamped scale *replaces* the
temperature rather than multiplying it. Full derivation: report §6, §7, §10.

### 50.1 Notation and what QK-normalisation changes in the forward pass

The current gate has two unbounded multiplicative channels feeding the
scaled score: $\lVert q \rVert \lVert k \rVert$ in the numerator and
$1/\tau$ in the denominator. QK-normalisation replaces the first with a
bounded cosine similarity, leaving one channel — the temperature — still
free to fall without limit. Report §1.1 and §10.1.

### 50.2 The temperature gradient is a covariance under the attention distribution

$$\frac{\partial L}{\partial \log \tau_k} = -\mathrm{Cov}_{a_k}(\tilde{s}_k, u_k)$$

— a covariance between the scaled scores and their utility, taken under
register $k$'s own attention distribution. It has **no interior zero**: it
vanishes only if the scores carry no information about utility, or if
attention has fully collapsed to a point mass. Both measured gradients on
register 14 are positive (so the covariance is negative), meaning register
14's scores are genuinely informative and the optimizer is deliberately
sharpening it — not malfunctioning. Report §6.

### 50.3 The loop is a Riccati equation, so the divergence is superlinear

In the diffuse regime, the inverse temperature $v = 1/\tau$ obeys
$\dot v = \eta C v^2$ — a Riccati equation with finite-time blow-up,
not merely exponential growth. This governs the *slow, monotone* drift of
$\tau_{14}$ across thousands of steps; it does not explain §49.6's 211×
single-event jump, which is a batch-dependent fluctuation in $C_k$ itself.
Report §7.

### 50.4 QK-normalisation alone does **not** break the loop

The first implementation applied a clamped `logit_scale` *on top of* the
existing $1/\tau$ divisor. Re-deriving §50.2 shows this changes nothing
structurally — the loop is driven by a *ratio*, and bounding only the
numerator does not bound a ratio whose denominator is still free to fall.
This was a real implementation defect, not just a theoretical concern.
Report §10.2.

### 50.5 What the corrected version does and does not fix

The fix: under `qk_norm`, the clamped per-register scale must **replace**
`log_tau` rather than multiply it (`log_tau` is not registered at all in
that path). The corrected dynamics are the *identical* Riccati equation,
now in the clamped scale $\sigma$ instead of $1/\tau$ — QK-normalisation
does not damp the feedback or create an interior fixed point. What changes
is the boundary: the trajectory now runs into a ceiling chosen in advance
($\sigma_{\max}=100$), verified directly by scaling $W_Q,W_K$ 1×–30× and
confirming the hardened gate's peak score stays flat (11.6–14.7) while the
unhardened one grows quadratically (17.0–14,461.0). Report §10.3–§10.5.

### 50.6 Deployment: two tiers, and why they differ

Two mitigations, deliberately not treated as substitutes for each other:
a projected floor on $\tau$ (resume-safe, no forward-function change, a
no-op at the live run's current values) for the *existing* 72K-step run,
and `creation_qk_norm` (a hard bound, but not retrofittable — it would
compress register 14's already-drifted scores by roughly 50× in one step)
reserved for a fresh arm only. Report §11 and §13.

### 50.7 Predictions, stated in advance

Four falsifiable predictions were recorded before testing: the floor pins
`tau_argmin` at register 14; the floor engaging should make register 14's
*raw* $\lvert q \cdot k \rvert$ start growing (the channel-migration test,
stated as the discriminating one); register 14's contribution to creation
entropy keeps falling; and single-register gradient dominance should be
absent entirely under a fresh QK-norm arm. **§51 amends this** — the floor
is now expected never to bind, and prediction 1 is only partially
confirmed (§12.1). Report §12.

---

## 51. Most of the Measured `log_tau` Drift Was Weight Decay on a Log-Parameterised Temperature

**Stubbed 2026-09-11 (see §52.2).** §49 measured a drift and §50 derived a
mechanism that could produce one; the two were read together as cause and
effect. Checking the third force acting on the parameter — AdamW's own
weight decay, which neither section had accounted for — shows it dominates
the measured magnitude, though not the per-step attribution. This
reassigns most of the drift without overturning §49's mechanism or §50's
dynamics, and turns "there is no interior equilibrium" from an alarming
structural claim into a fixable configuration error. Full derivation:
report §8, §9.

### 51.1 AdamW was given a flat parameter list

`WEIGHT_DECAY = 0.01` was applied to every parameter including `log_tau` —
a category error on a *log*-parameterised quantity, since decoupled decay
shrinks $\log\tau$ toward zero, i.e. pulls $\tau \to 1$, from an
initialisation of 8.0. Report §8.1.

### 51.2 The pure-decay trajectory runs through the middle of the observed pool

Integrating the real WSD schedule with **zero loss gradient**, decay alone
takes $\tau$ from 8.0 to 5.42 by step 72K and 5.11 by step 100K — and the
measured pool at step ~72K (min 5.21, median 6.48, max 8.95) has the
decay line sitting *between* register 14 and the pool median. The
pool-wide cooling §49 noted but did not explain is a config artifact.
Report §8.2.

### 51.3 Decomposition over the §49.5 window

Splitting the observed $\Delta\log\tau$ into its decay and residual parts:
register 14's drift is 70% decay, 30% residual (loss-gradient); the pool
median's own loss gradient pushes **up** at +0.0151, *resisting* decay's
−0.0195. Register 14 is distinguished by **failing to resist** a pool-wide
pull, not by being driven down hardest — and 31 of 32 registers having
loss gradients that push $\tau$ up is direct evidence against reading
§50.2's "no interior equilibrium" as a universal runaway; that result holds
only at zero decay. Report §8.3, §8.4.

### 51.4 The fix, and the equilibrium it restores

`NO_DECAY_1D`: weight decay only for rank-≥2 tensors, none for 1-D
parameters — which turns out to be almost exactly the set that dominates
every captured spike in this investigation (`log_tau`,
`reverse_channel_scale`, `logit_scale`, `depth_code`). Any restoring term
anchored at $\ell_0$ creates a stable interior fixed point in §50.2's
dynamics; the run had one, anchored at $\ell_0=0$ ($\tau=1$) — reinforcing
the sharpening instead of opposing it. Removing it is implemented; a
better anchor at $\ell_0=\log\tau_{\text{init}}$ is specified but deferred.
Report §9, §11.4.

### 51.5 Carrying Adam moments across the param-group change

Splitting one AdamW group into two re-indexes every optimizer-state key,
which torch keys by flattened position. A naive fix either raises on
load (previously swallowed, silently discarding every Adam moment) or —
worse — loads without error while a plausible-looking sequential
re-index silently mis-assigns 4 of 6 parameters. `_remap_optim_state`
builds an explicit old-position → `Parameter` → new-position map,
verified exact against both failure modes as negative controls.
Report §11.4.

### 51.6 Consequences for the §49-§50 mitigations

The §49.8 $\tau$ floor is now expected never to bind (pure decay only
reaches 5.11 by step 100K). The cleanest remaining prediction, costing
nothing to check: with decay removed, the pool's downward drift should
largely stop and the median should **rise**, since its own loss gradient
was already pushing up against decay. If the pool keeps cooling at a
similar rate instead, this decay accounting is wrong. **Outcome, §12.1 of
the report**: partially falsified — the argmin is not pinned at register
14, but it is a two-way (later three-way) contest among specific
registers, not diffuse wandering; the decay-removal prediction holds.
