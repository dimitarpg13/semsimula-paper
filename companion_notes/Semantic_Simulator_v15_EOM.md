# Semantic Simulator v1.5 — Equation-of-Motion Specification (Forthcoming)

> **Status: Forthcoming.** This companion note will provide the
> equation-of-motion specification for the **v1.5 (salient-decay /
> dissipative-semigroup) extension** of the Semantic Simulator.
> It is referenced from §9 (Expressivity, mechanism justification,
> and the MCS reach) and §17 Q8 of the main paper.

## Scope

The v0 field-theoretic submodel of the main paper
(\S\S `space`, `signature`, `well`, `parf`, `sarf`, `lagrangian`)
is shown to be at most a finite automaton (Theorem `v0-ceiling`).
The framework recovers the empirically-established
mildly-context-sensitive class for human language only when three
additional mechanisms are switched on:

- **v1.5 — salient decay / structure destruction** (this note)
- **v2 — structure creation** (companion note `Semantic_Simulator_v2_EOM.md`)
- **v3 — structure execution** (companion note `Semantic_Simulator_v3_EOM.md`)

The mechanism-apparatus mapping of §9 names **dissipative semigroups**
as the formal home of v1.5; an explicit equation-of-motion
specification in that formal home is the subject of this note.

## What this note will contain (planned outline)

1. The (D1)–(D5) lifecycle requirements from §`subsec:lifecycle`
   re-stated in semigroup language.
2. The dissipative generator $\\mathcal{L}$ acting on the salience
   field, with explicit decay kernel, salience-threshold
   destruction rule, and reinforcement-field write-back specification.
3. Coupling to the v0 Lagrangian: a Rayleigh-augmented action
   functional under which v1.5-driven destruction events are
   generators of measure-zero discontinuities in the configuration
   manifold and the resulting weak-solution semantics.
4. A discrete simulator integrator and the F1–F4 falsifier mapping.

## Deferral rationale

The v1.5 / v2 / v3 EOM specifications are companion-note artefacts
because their natural formal home (dissipative semigroups,
Fock-space second quantisation, non-abelian gauge theory
respectively) is mathematically substantial enough that a single
self-contained main-paper section cannot do them justice. They are
listed as Q8(a) of the main paper's conclusion and form one
continuous research programme together with the F1–F6 falsifier
implementation programme.

## See also

- Main paper, §9 (Expressivity, mechanism justification, MCS reach).
- Main paper, §17 Q8 (the deferred F1–F6 + EOM programme).
- Companion note `MCS_Reduction_For_v3_Composite.md` (the
  formal-language reduction the v15/v2/v3 EOMs are designed to realise).
- Companion note `Semantic_Simulator_RL_Calibration_Programme.md`
  (the trajectory-level calibration programme).
- Companion note `Semantic_Simulator_EOM.md` (the v0 EOM, on top of
  which v1.5 / v2 / v3 build).
