# AI-Assisted Collaborative Research Disclosure

This document discloses the role of AI assistance in the development of the
research and artifacts contained in this repository and the accompanying paper:

> **Semantic Simulation: A Prescriptive Lagrangian Framework for Efficient
> Semantic Inference**
> Dimitar P. Gueorguiev (Independent Researcher), 2026.

---

## Overview

This work was developed through an extensive human--AI collaborative workflow
using **Anthropic's Claude** (Opus and Sonnet model families) as a research
partner throughout the project lifecycle. The collaboration was sustained over
several months and spanned the full research pipeline from mathematical
exploration to publication preparation.

The disclosure below is provided in the interest of transparency and
reproducibility, and is consistent with the "Statement on AI-Assisted Research"
section in the published paper.

---

## Domains of AI Contribution

The AI system contributed substantively in the following areas:

### Mathematical Exploration and Derivation Assistance

- The STP--acceleration identity analysis and its connection to the
  Huang--LeCun--Balestriero Semantic Tube Prediction framework
- Lagrangian framework elaboration for hidden-state dynamics in transformers
- Fock-space formulations for the second-quantised extension (FockPARFLM)
- Expressivity bound derivations for the simulator architecture
- The STP--BAOAB integrator analysis for the direct dynamical simulator
  (paper v5)

### Experiment Design and Implementation

- The P10 experimental ladder structure (P10a through P10h) and ablation design
- Dynamics verification scripts and simulator skeleton code
  (`notebooks/semsim_simulator/dynamics/`)
- The harvested-potential integration mechanism for warm-starting the EOM
  simulator from transformer-trained scalar potentials
- Code review and debugging assistance throughout the experimental pipeline

### Result Interpretation

- Analysis of architectural ceilings encountered in the PARFLM experiments
- Integrator-order comparisons (first-order vs. second-order SPLM)
- Shared-potential diagnostic outputs and R-squared separator analysis
- The PARFLM-to-FockPARFLM transition rationale

### Paper Drafting and Technical Exposition

- First drafts of multiple paper sections, subsequently reviewed and revised
  by the author
- Notation consistency enforcement across the 238-page manuscript
- Literature-contextualized discussion and related-work positioning
- Companion notes and background manuscripts in this repository

### Documentation and Companion Materials

- Deep-dive analysis documents (e.g., Modified BAOAB analysis, Efficient
  Numerical Algorithm on GPU, Multi-Head Attention analysis, Probabilistic
  Modeling in Direct Dynamical Simulation)
- GitHub-ready markdown formatting and rendering fixes
- Repository organisation and metadata (CITATION.bib, README structure)

### Publication Strategy

- Structural advice on venue selection (TMLR carve-out strategy)
- Paper decomposition planning for focused submissions
- Zenodo/SSRN publication workflow guidance

---

## What Remained Under Human Control

The following aspects were performed by the human author:

- **Research direction and conceptual framing** -- the decision to pursue a
  Lagrangian/dynamical-systems perspective on transformer hidden states
  originated with the author and was sustained across the full project
- **Hypothesis selection** -- which hypotheses to test, which architectures
  to build, which experimental comparisons to run
- **Verification of all derivations and experimental results** -- every
  mathematical claim and every experimental number was checked by the author
- **Experimental supervision** -- training runs, hyperparameter decisions,
  convergence monitoring
- **Interpretation of findings** -- situating results within the broader
  research programme and deciding what they mean for the next steps
- **Final scientific judgment** -- the author understands and can defend all
  claims made in the paper and takes full responsibility for its contents

---

## On Authorship

Current academic norms (including policies at TMLR, NeurIPS, ICML, Nature, and
Science) converge on the following:

- **AI systems cannot be listed as co-authors** because they cannot take legal
  or ethical responsibility for the work, cannot respond to peer review, and
  cannot be held accountable for errors.
- **AI-assisted workflows are permitted** provided that human authors retain
  scientific responsibility and disclose the assistance honestly.

This work is authored solely by Dimitar P. Gueorguiev. Claude is acknowledged
as a collaborative research tool, not as a co-author.

---

## Transparency

The collaborative workflow is documented in the project's development
repository, including session transcripts. This level of openness is intended
to allow the community to assess both the nature and the extent of AI
involvement, and to serve as a reference point for evolving norms around
AI-assisted research.

---

*This disclosure was last updated on May 12, 2026.*
