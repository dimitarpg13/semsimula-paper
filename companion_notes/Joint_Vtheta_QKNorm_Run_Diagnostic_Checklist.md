# Diagnostic checklist — joint-V_theta + QK-norm d384 run (moved)

**This checklist was split on 2026-11-22** into self-contained documents, each with its own definitions, problem statement and analysis, in [joint_vtheta_arm/](joint_vtheta_arm/README.md). This page remains so that existing links resolve. Where each former section now lives:

| former section | now in |
| --- | --- |
| introduction (run status, the degradation and its diagnosis) | [joint_vtheta_arm/README.md](joint_vtheta_arm/README.md), §2 |
| §0 Before running anything | [Bank_Geometry_and_Stiffness_Diagnostics.md](joint_vtheta_arm/Bank_Geometry_and_Stiffness_Diagnostics.md), §3 |
| §1 What exists, and what is blocked | [joint_vtheta_arm/README.md](joint_vtheta_arm/README.md), §3 |
| §2.1 to §2.6 Run in this order | [Bank_Geometry_and_Stiffness_Diagnostics.md](joint_vtheta_arm/Bank_Geometry_and_Stiffness_Diagnostics.md), §4.1 to §4.6 |
| §3 Predictions recorded in advance | [Bank_Geometry_and_Stiffness_Diagnostics.md](joint_vtheta_arm/Bank_Geometry_and_Stiffness_Diagnostics.md), §5 |
| §4 What the results decide | [Bank_Geometry_and_Stiffness_Diagnostics.md](joint_vtheta_arm/Bank_Geometry_and_Stiffness_Diagnostics.md), §6 |
| §5 Lower priority | [Bank_Geometry_and_Stiffness_Diagnostics.md](joint_vtheta_arm/Bank_Geometry_and_Stiffness_Diagnostics.md), §7 |
| §6 and §6.1, §6.2 The anneal probe from step 50,000 | [Anneal_Probes.md](joint_vtheta_arm/Anneal_Probes.md), §3 |
| §6.3 The anneal from step 28,500 | [Anneal_Probes.md](joint_vtheta_arm/Anneal_Probes.md), §4 and §5 |
| §6.4 The production decay from step 28,500 | [Anneal_Probes.md](joint_vtheta_arm/Anneal_Probes.md), §6 |
| §7 Integration refinement (premise and idea) | [Integration_Refinement.md](joint_vtheta_arm/Integration_Refinement.md), §2 |
| §7.0 The linear algebra benchmark | [Integration_Refinement.md](joint_vtheta_arm/Integration_Refinement.md), §4 |
| §7.1 What refines for free | [Integration_Refinement.md](joint_vtheta_arm/Integration_Refinement.md), §3 |
| §7.2 Substeps at dt = 0.5 | [Integration_Refinement.md](joint_vtheta_arm/Integration_Refinement.md), §6 |
| §7.3, §7.4, §7.7 What not to do | [Integration_Refinement.md](joint_vtheta_arm/Integration_Refinement.md), §7.1 to §7.3 |
| §7.5 Standing facts about depth | [Integration_Refinement.md](joint_vtheta_arm/Integration_Refinement.md), §8 |
| §7.6 The exact low-rank integrator | [Integration_Refinement.md](joint_vtheta_arm/Integration_Refinement.md), §5 |
| §8 After this analysis | [joint_vtheta_arm/README.md](joint_vtheta_arm/README.md), §5 |
| §9 The matched GPT-2 baseline | [Matched_GPT2_Comparison.md](joint_vtheta_arm/Matched_GPT2_Comparison.md) |
| §10.1 to §10.5 Family A | [Content_Addressed_Context_Routing.md](joint_vtheta_arm/Content_Addressed_Context_Routing.md), §3 |
| §10.6 Two mechanisms, two slots | [Content_Addressed_Context_Routing.md](joint_vtheta_arm/Content_Addressed_Context_Routing.md), §2 |
| §10.7 Remaining gates | [Content_Addressed_Context_Routing.md](joint_vtheta_arm/Content_Addressed_Context_Routing.md), §4 (Alternative E) and §6 (Family A) |
| §10.8 Alternative E result | [Content_Addressed_Context_Routing.md](joint_vtheta_arm/Content_Addressed_Context_Routing.md), §5 |
| §10.9 The Phase 2 factorisation probe | [Content_Addressed_Context_Routing.md](joint_vtheta_arm/Content_Addressed_Context_Routing.md), §7 |

The definitions and the mathematical statements of the tests (T1 to T14), added to this checklist earlier the same day, are now inside the documents they belong to.
