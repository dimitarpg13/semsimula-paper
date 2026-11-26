# Companion Notes: Index by Topic

**What this is.** Every note in this folder, grouped by topic, one line each. A note that belongs to two topics is listed under both. The notes stay where they are: the book and the Hugging Face model cards link to them by path. (Index written 2026-11-21.)

**Folders with their own index.**

- [depth_ladder/](depth_ladder/README.md): the depth-ladder programme, the conservative models trained one mechanism at a time against a matched GPT-2; includes the recall pathway probe, the velocity and the pair force as competing mechanisms, and why a first-order model is better trained without the pair force.
- [productization/](productization/README.md): making the conservative core cheap to run, stage by stage.
- [mean_field/](mean_field/README.md): results of the mean-field programme, one note per probe or arm (Probe A so far).
- [semantic_templates/](semantic_templates/README.md): building semantic templates into the models: where and when a template search may start, and the measurements that test it.
- [training_instabilities/](training_instabilities/README.md): the training instabilities of Fock-PARFLM with a structured $V_\theta$, and the CfC+BAOAB integrator with its mitigations.
- [deep_dives/](deep_dives/): long-form Word documents (with PDFs): the SPLM model and code, the structured scalar potential, the Poisson-mode registers and semantic templates, and the Jacobi metric.
- [figures/](figures/): the figures of the notes and the scripts that make them (`_make_<topic>_figs.py`).

---

## Contents

1. [Framework foundations](#framework-foundations)
2. [Related work and comparisons](#related-work-and-comparisons)
3. [Transformer hidden-state dynamics and the case for conservative models](#transformer-hidden-state-dynamics-and-the-case-for-conservative-models)
4. [SPLM: the scalar-potential language model](#splm-the-scalar-potential-language-model)
5. [The context channels](#the-context-channels)
6. [The context potential and its wells](#the-context-potential-and-its-wells)
7. [The pair potential (PARF, SARF)](#the-pair-potential-parf-sarf)
8. [The Fock mechanism: exchange field, registers, reverse channel](#the-fock-mechanism-exchange-field-registers-reverse-channel)
9. [Doi–Peliti dynamics and the registers](#doipeliti-dynamics-and-the-registers)
10. [Semantic templates, execution and chain-of-thought](#semantic-templates-execution-and-chain-of-thought)
11. [Expressivity and the simulator programme (v0 to v3)](#expressivity-and-the-simulator-programme-v0-to-v3)
12. [Integrators and CfC+BAOAB](#integrators-and-cfcbaoab)
13. [Langevin dynamics](#langevin-dynamics)
14. [Damping and the overdamped limit](#damping-and-the-overdamped-limit)
15. [First order against second order](#first-order-against-second-order)
16. [Jacobi metric and Riemannian geodesics](#jacobi-metric-and-riemannian-geodesics)
17. [Attention](#attention)
18. [Hybrids: SPLM with attention, Helmholtz architectures](#hybrids-splm-with-attention-helmholtz-architectures)
19. [Synchronization, clustering and mean-field limits](#synchronization-clustering-and-mean-field-limits)
20. [Causal leaks and auditing](#causal-leaks-and-auditing)
21. [Training: gradients, memory, tuning](#training-gradients-memory-tuning)
22. [The book: planning and audits](#the-book-planning-and-audits)
23. [Writing the notes](#writing-the-notes)

---

## Framework foundations

- [On_the_Interpretation_of_Hidden_State](On_the_Interpretation_of_Hidden_State.md): what a hidden state is in the framework: phase-space coordinates rather than latent features
- [On_the_Interpretation_of_Semantic_Mass](On_the_Interpretation_of_Semantic_Mass.md): per-token semantic mass, read through transformer mechanisms
- [On_The_Existence_of_Acceleration_in_Semantic_Structures](On_The_Existence_of_Acceleration_in_Semantic_Structures.md): evidence for acceleration (deceleration) in hidden-state trajectories, against the first-order STP picture
- [STP_Loss_Is_An_Emergent_Property_Of_The_Energy_Landscape_Defined_By_Gaussian_Well_Potential](STP_Loss_Is_An_Emergent_Property_Of_The_Energy_Landscape_Defined_By_Gaussian_Well_Potential.md): the STP loss as an emergent property of a Gaussian-well energy landscape
- [Single_Particle_Hilbert_Space_in_Semantic_Simulation](Single_Particle_Hilbert_Space_in_Semantic_Simulation.md): from semantic densities to Fock space under the classical (Doi–Peliti) commitment
- [Overlap_Distance_in_Semantic_Simulation](Overlap_Distance_in_Semantic_Simulation.md): the overlap distance: definition, geometry, and what it means for the Fock-space (v2) formalism
- [Pairwise_vs_Aggregate_Decay_in_Semantic_Space](Pairwise_vs_Aggregate_Decay_in_Semantic_Space.md): why the pair force needs a partner cap and the Gaussian well does not
- [Potential_Wells_Temperature_and_Boltzmann-Gibbs_Theory](Potential_Wells_Temperature_and_Boltzmann-Gibbs_Theory.md): statistical-mechanical foundations of the structured potential: wells, temperature, Boltzmann–Gibbs
- [Hamiltonian_Flow_in_the_Semantic_Simulation_Framework](Hamiltonian_Flow_in_the_Semantic_Simulation_Framework.md): what the Lagrangian commitment buys, where a Hamiltonian flow exists in the code, and where trained models leave it
- [Forced_Lagrangian_Reformulation](Forced_Lagrangian_Reformulation.md): the thesis document: reformulating the framework around the non-conservative forces, after the geodesic reading closed
- [Continuous_Learning_in_Semantic_Simulation-based_models_vs_with_Transformer_models](Continuous_Learning_in_Semantic_Simulation-based_models_vs_with_Transformer_models.md): continuous learning in framework models against transformers (speculative; conjectures pending)


## Related work and comparisons

- [Hidden_States_Semantic_Tube_Geodesics_and_Semantic_Structures](Hidden_States_Semantic_Tube_Geodesics_and_Semantic_Structures.md): Huang et al.'s Semantic Tube Prediction compared with the Semantic Simulation framework
- [The_Impact_of_Brownian_Motion_on_Perplexity_in_Attention-Based_Transformers_and_SPLM](The_Impact_of_Brownian_Motion_on_Perplexity_in_Attention-Based_Transformers_and_SPLM.md): the training ODE and inference SDE behind STP, and what Brownian motion does to perplexity in transformers and SPLM
- [SemSimula_connection_to_Hopfield_networks_and_Restricted_Boltzman_Machines](SemSimula_connection_to_Hopfield_networks_and_Restricted_Boltzman_Machines.md): the framework against Hopfield networks and restricted Boltzmann machines
- [Lessons_from_AlphaFold](Lessons_from_AlphaFold.md): what AlphaFold teaches about forces, the move from Verlet to BAOAB, and potential harvesting
- [Parallels_and_Lessons_from_Liquid_Neural_Networks](Parallels_and_Lessons_from_Liquid_Neural_Networks.md): a comparison with liquid neural networks (the origin of the CfC integrator) and their path to product


## Transformer hidden-state dynamics and the case for conservative models

- [Evidence_for_second_order_ODE_governing_evolution](Evidence_for_second_order_ODE_governing_evolution.md): the case for a second-order ODE behind transformer hidden states, with the overdamped synthesis
- [first_order_ODE_rejection_pre-registered_protocol](first_order_ODE_rejection_pre-registered_protocol.md): pre-registered test: can a first-order ODE be rejected for transformer hidden-state dynamics?
- [The_Failure_of_Conservative_Models_to_explain_hidden_state_trajectories](The_Failure_of_Conservative_Models_to_explain_hidden_state_trajectories.md): why conservative fits fail on pretrained transformer trajectories (the E-init post-mortem)
- [Considered_Non-Autonomous_Conservative_Mechanisms](Considered_Non-Autonomous_Conservative_Mechanisms.md): the candidate governing equations considered for transformer dynamics, up to the non-autonomous conservative class
- [Addendum_Non_Autonomous_Fields_For_Appendix_A](Addendum_Non_Autonomous_Fields_For_Appendix_A.md): reader's guide to the book's Appendix A on non-autonomous conservative fields
- [P-rot-6_transformer_dynamics](P-rot-6_transformer_dynamics.md): a damped equation with a velocity-coupled skew force, tied to the key–value asymmetry of attention
- [Conservative_by_Construction_Language_Models](Conservative_by_Construction_Language_Models.md): the design rationale for language models that are conservative by construction (SPLM)
- [The_Overdamped_Limit_and_The_Position_of_The_2nd_Order_Lagrangian_Framework](The_Overdamped_Limit_and_The_Position_of_The_2nd_Order_Lagrangian_Framework.md): where the second-order framework stands given that transformers and SPLM look overdamped


## SPLM: the scalar-potential language model

- [Conservative_by_Construction_Language_Models](Conservative_by_Construction_Language_Models.md): the design rationale for language models that are conservative by construction (SPLM)
- [Training_and_Inference_with_SPLM](Training_and_Inference_with_SPLM.md): how SPLM is trained and run: the loop, the forces through nested autograd, the inference pipeline
- [On_Modeling_Semantic_Energy_Field_into_SPLM](On_Modeling_Semantic_Energy_Field_into_SPLM.md): how SPLM represents the framework's energy field: well, PARF, SARF and semantic mass
- [Next_Model_Experiments_for_SPLM](Next_Model_Experiments_for_SPLM.md): a prioritised catalogue of experiments for SPLM (source of the E1 and E3 programmes)
- [Replacing_The_Conservative_Mechanism_of_SPLM_with_First_Order](Replacing_The_Conservative_Mechanism_of_SPLM_with_First_Order.md): can SPLM's second-order step be replaced by a first-order gradient flow?
- [SPLM-1_ablation_pre-registered_protocol](SPLM-1_ablation_pre-registered_protocol.md): pre-registered protocol: the first-order ablation SPLM-1
- [SPLM_inference_efficiency_pre-registered_protocol](SPLM_inference_efficiency_pre-registered_protocol.md): pre-registered protocol: inference FLOPs and parameters, SPLM against attention
- [SPLM_inference_first_order_pre-registered_protocol](SPLM_inference_first_order_pre-registered_protocol.md): pre-registered protocol: the Markov-order test on SPLM inference
- [SPLM_scaleup_pre-registered_protocol](SPLM_scaleup_pre-registered_protocol.md): pre-registered protocol: the SPLM scale-up de-risking experiment (E9)
- [Energetic_Minima_Alternatives](Energetic_Minima_Alternatives.md): three alternatives to a free $V_\theta$ for keeping energetic minima, retrained after the leak fix
- [Semantic_Attractor_Extraction](Semantic_Attractor_Extraction.md): extracting the attractors (local minima) of a trained $V_\theta$
- [Memorization_Capabilities_of_the_SPLM_Family_of_Models](Memorization_Capabilities_of_the_SPLM_Family_of_Models.md): what the SPLM family can memorise, and through which mechanisms
- [CUDA_Memory_Errors_with_SPLM_Designs](CUDA_Memory_Errors_with_SPLM_Designs.md): a catalogue of the out-of-memory events in the PARF arms of the May 2026 pilot


## The context channels

- [Multi-Channel_vs_Single_Channel_Xi_SPLM_Design](Multi-Channel_vs_Single_Channel_Xi_SPLM_Design.md): why $\xi$ moved from one channel to several: the design analysis
- [SPLM_multichannel_xi_pre-registered_protocol](SPLM_multichannel_xi_pre-registered_protocol.md): pre-registered protocol: the multi-channel $\xi$ scale-up (E11)
- [Reducing_Information_Bottleneck_In_Multi-Channel_Xi_SPLM](Reducing_Information_Bottleneck_In_Multi-Channel_Xi_SPLM.md): information-theoretic case for redesigning the $\xi$ block: K-EMA, HiPPO-LegT and S4D bases
- [Xi_Bottleneck_Diagnosis_Phase5](Xi_Bottleneck_Diagnosis_Phase5.md): diagnosing the plateau of the Gaussian-well Fock-PARFLM v2.1 at PPL ~222 on OpenWebText
- [Context_Mixing_Mechanisms_in_the_Conservative_Framework](Context_Mixing_Mechanisms_in_the_Conservative_Framework.md): the context-mixing mechanism of Fock-PARFLM v2.1 and four conservative alternatives
- [Analytic_Multi_Channel_Integration_in_Structured_Vtheta](Analytic_Multi_Channel_Integration_in_Structured_Vtheta.md): integrating several context channels analytically in one structured $V_\theta$ well


## The context potential and its wells

- [Structured_VTheta_Design_and_Theory](Structured_VTheta_Design_and_Theory.md): structured, analytically differentiable parameterisations of $V_\theta$: theory and derivation
- [Structured_VTheta_Memory_Anatomy](Structured_VTheta_Memory_Anatomy.md): why fewer $V_\theta$ parameters do not prevent out-of-memory errors
- [Analytic_Multi_Channel_Integration_in_Structured_Vtheta](Analytic_Multi_Channel_Integration_in_Structured_Vtheta.md): integrating several context channels analytically in one structured $V_\theta$ well
- [Dead_Wells_in_the_Joint_V_theta_Bank](Dead_Wells_in_the_Joint_V_theta_Bank.md): XI6-CODE's failure: wells of the joint $V_\theta$ bank that die, and two mitigations
- [Minimum_Well_Count_for_Conservative_Models](Minimum_Well_Count_for_Conservative_Models.md): how many wells a conservative model uses and how many it needs
- [Potential_Wells_Temperature_and_Boltzmann-Gibbs_Theory](Potential_Wells_Temperature_and_Boltzmann-Gibbs_Theory.md): statistical-mechanical foundations of the structured potential: wells, temperature, Boltzmann–Gibbs
- [Structural_Stability_of_Learned_Potentials_in_Semantic_Simulation](Structural_Stability_of_Learned_Potentials_in_Semantic_Simulation.md): is the learned potential landscape structurally stable?
- [Portable_Learned_Potentials_and_Transplant_Map](Portable_Learned_Potentials_and_Transplant_Map.md): when a trained well landscape ($V_\theta$, $V_\phi$) can be reused in another run or model
- [Energetic_Minima_Alternatives](Energetic_Minima_Alternatives.md): three alternatives to a free $V_\theta$ for keeping energetic minima, retrained after the leak fix
- [Semantic_Attractor_Extraction](Semantic_Attractor_Extraction.md): extracting the attractors (local minima) of a trained $V_\theta$
- [Competing_Mechanisms_Families_Dominance_and_Substitution](Competing_Mechanisms_Families_Dominance_and_Substitution.md): which mechanism families dominate or substitute for one another in the conservative core


## The pair potential (PARF, SARF)

- [PARF_Augmented_SPLM_Architecture_v2](PARF_Augmented_SPLM_Architecture_v2.md): the PARF-augmented SPLM (PARFLM): a framework-native routing architecture
- [PARF-SPLM_Path_Forward_and_Experiments](PARF-SPLM_Path_Forward_and_Experiments.md): experiment record of the PARF-SPLM (Q9c) path
- [Structured_VPhi_Design_and_Theory](Structured_VPhi_Design_and_Theory.md): the structured pair potential $V_\phi$: design, theory and analysis
- [On_Training_the_PARF_Force](On_Training_the_PARF_Force.md): how $V_\phi$ is trained, what it costs, the alternatives, and the recommended path
- [On_the_MLP_Layer_modeling_pairwise_potential](On_the_MLP_Layer_modeling_pairwise_potential.md): the unstructured MLP variant of $V_\phi$
- [Gumbel_sparsity_method](Gumbel_sparsity_method.md): the Stage 1.5 Gumbel-softmax sparsity method for the pair potential
- [On_Gumbel_softmax_sparsity_applied_to_V_phi](On_Gumbel_softmax_sparsity_applied_to_V_phi.md): Gumbel-softmax sparsity for $V_\phi$: theory and practice, Stage 1 against Stage 1.5
- [PARF_Stage_1_5b_design](PARF_Stage_1_5b_design.md): design of the gathered top-k $V_\phi$ (Stage 1.5b)
- [On_Modeling_SARF_forces_in_PARF_augmented_SPLM_architectures](On_Modeling_SARF_forces_in_PARF_augmented_SPLM_architectures.md): SARF and its derivation from PARF: why SARF should not be a separate learnable force
- [SARF_Anchor_Placement_From_Converged_Gaussian_Centres](SARF_Anchor_Placement_From_Converged_Gaussian_Centres.md): placing the SARF anchors from the converged centres of learned Gaussian wells
- [Pairwise_vs_Aggregate_Decay_in_Semantic_Space](Pairwise_vs_Aggregate_Decay_in_Semantic_Space.md): why the pair force needs a partner cap and the Gaussian well does not
- [Gradient_Checkpointing_for_PARF](Gradient_Checkpointing_for_PARF.md): per-layer-step gradient checkpointing for the PARF stack
- [Eliminating_the_Autograd_Dependency_in_VPhi](Eliminating_the_Autograd_Dependency_in_VPhi.md): options for removing the autograd dependency of the $V_\phi$ force (design; nothing measured)


## The Fock mechanism: exchange field, registers, reverse channel

**The mechanism**

- [Fock_Mechanism_v2.1_Principle_of_Operation](Fock_Mechanism_v2.1_Principle_of_Operation.md): the exchange field and the Fock registers with their reverse channel: how they work, their equations, their relation to attention
- [Improving_the_Fock_Mechanism_to_match_Attention](Improving_the_Fock_Mechanism_to_match_Attention.md): Fock attention experiments, register diagnostics, and the routing resolution hierarchy
- [Conservative_Obstruction_and_Virtual_Particle_Necessity](Conservative_Obstruction_and_Virtual_Particle_Necessity.md): why a conservative model needs a virtual-particle exchange to match attention
- [Attention_Optimality_Conjecture](Attention_Optimality_Conjecture.md): which properties make attention optimal, and what an exchange must have to match it
- [Augmenting_PARFLM_to_handle_MCS_Languages](Augmenting_PARFLM_to_handle_MCS_Languages.md): augmenting PARFLM with Fock registers toward mildly context-sensitive languages
- [Fock_Mechanism_Efficiency_Across_Layer_Depth](Fock_Mechanism_Efficiency_Across_Layer_Depth.md): how much of the register mechanism is live at $L = 1$, 2 and 4
- [Fock_Mechanism_Engagement_MLP_vs_Gaussian_VTheta](Fock_Mechanism_Engagement_MLP_vs_Gaussian_VTheta.md): how strongly the Fock mechanism engages with an MLP against a Gaussian $V_\theta$
- [Fock_Mechanism_Ablation_Study_d384_OpenWebText](Fock_Mechanism_Ablation_Study_d384_OpenWebText.md): ablating the Fock mechanism at $d = 384$ (central claim superseded in magnitude, 2026-09-25)
- [Fock_PARFLM_Conservativity_Diagnostic](Fock_PARFLM_Conservativity_Diagnostic.md): a five-arm diagnostic battery showing Fock-PARFLM v2.1 is conservative by construction
- [Context_Mixing_Mechanisms_in_the_Conservative_Framework](Context_Mixing_Mechanisms_in_the_Conservative_Framework.md): the context-mixing mechanism of Fock-PARFLM v2.1 and four conservative alternatives

**The Fock-PARFLM scale-up runs (July–August 2026)**

- [Fock-PARFLM_Scale-Up_Comparative_Experiments](Fock-PARFLM_Scale-Up_Comparative_Experiments.md): code-verified parameter counts at $d = 384$, 768 and 1024 against matched GPT-2
- [Fock-PARFLM_d384_Training_Results_OpenWebText](Fock-PARFLM_d384_Training_Results_OpenWebText.md): $d = 384$ training on OpenWebText (perplexities invalidated by the causal leak)
- [Fock-PARFLM_d768_Training_Results_OpenWebText](Fock-PARFLM_d768_Training_Results_OpenWebText.md): $d = 768$ training on OpenWebText (results compromised by the causal leak)
- [Fock-PARFLM_Scale-Up_Gamma_Sweep_Results_and_Damping_Regime_Analysis](Fock-PARFLM_Scale-Up_Gamma_Sweep_Results_and_Damping_Regime_Analysis.md): $\gamma$ sweeps at $d = 384$, 768 and 1024, and the damping regime they show
- [Fock-PARFLM_vs_GPT-2_on_OpenWebText_Next_Steps](Fock-PARFLM_vs_GPT-2_on_OpenWebText_Next_Steps.md): Multi-Xi Fock-PARFLM v2.1 against GPT-2: predicted and measured gap, and what starves the design
- [MLP_VTheta_Fock-PARFLM_on_OWT_d384](MLP_VTheta_Fock-PARFLM_on_OWT_d384.md): the MLP $V_\theta$ Fock-PARFLM at $d = 384$: context channels, the Verlet fallback, planned experiments
- [Xi_Bottleneck_Diagnosis_Phase5](Xi_Bottleneck_Diagnosis_Phase5.md): diagnosing the plateau of the Gaussian-well Fock-PARFLM v2.1 at PPL ~222 on OpenWebText
- [Fock-PARFLM_Causal_Leak_Audit_Results](Fock-PARFLM_Causal_Leak_Audit_Results.md): the Fock-PARFLM v2.1 leak: root cause, magnitude and fix


## Doi–Peliti dynamics and the registers

- [Single_Particle_Hilbert_Space_in_Semantic_Simulation](Single_Particle_Hilbert_Space_in_Semantic_Simulation.md): from semantic densities to Fock space under the classical (Doi–Peliti) commitment
- [Overlap_Distance_in_Semantic_Simulation](Overlap_Distance_in_Semantic_Simulation.md): the overlap distance: definition, geometry, and what it means for the Fock-space (v2) formalism
- [Quantum_Formalism_Behind_Doi_Peliti_in_Semantic_Simulation](Quantum_Formalism_Behind_Doi_Peliti_in_Semantic_Simulation.md): a primer on the quantum-mechanical formalism that Doi–Peliti borrows
- [Doi_Peliti_Dynamics_of_Semantic_Particles_and_Registers](Doi_Peliti_Dynamics_of_Semantic_Particles_and_Registers.md): the stochastic process the Fock registers implement, derived from the master equation and measured
- [Poisson_Mode_Registers_PM1](Poisson_Mode_Registers_PM1.md): registers whose occupations are exact Poisson means and whose force is a gradient (PM1)
- [Exact_Integration_of_Poisson_Mode_Wells_PMX](Exact_Integration_of_Poisson_Mode_Wells_PMX.md): integrating the Poisson-mode wells exactly inside the damped flow (PMX)
- [Semantic_Templates_as_Doi_Peliti_Reactions](Semantic_Templates_as_Doi_Peliti_Reactions.md): semantic templates as the reaction rules of the Doi–Peliti dynamics, and the repair the join needs


## Semantic templates, execution and chain-of-thought

- [Implementing_Semantic_Templates_A_Path](Implementing_Semantic_Templates_A_Path.md): a staged path to implementing semantic templates
- [Semantic_Templates_as_Doi_Peliti_Reactions](Semantic_Templates_as_Doi_Peliti_Reactions.md): semantic templates as the reaction rules of the Doi–Peliti dynamics, and the repair the join needs
- [The_Particle_Store_Making_Allocation_Real](The_Particle_Store_Making_Allocation_Real.md): the particle store (Stage 0″): design and first results
- [semantic_templates/Admissible_Regions_for_Template_Search](semantic_templates/Admissible_Regions_for_Template_Search.md): where in semantic space and when in training a template search may start: the well frame, context spread, training drift, the admissibility criterion; SR2-NP's settling map
- [semantic_templates/Well_Settling_Map_SR2_NP](semantic_templates/Well_Settling_Map_SR2_NP.md): technical report of the first settling map (CPU, 2026-11-24): the live well keeps its tokens through the decay, but their order inside it is only half kept; no admissible region over that window
- [semantic_templates/Logical_Forms_as_Geometric_Figures](semantic_templates/Logical_Forms_as_Geometric_Figures.md): logical argument forms frozen as small figures: they transfer between models, and record the frame, the restatements and negation rather than validity
- [semantic_templates/Template_Matcher_From_Geometry](semantic_templates/Template_Matcher_From_Geometry.md): report of the first template matcher: a cloud per logical form in a small subspace, fired from the geometry of the premise alone
- [semantic_templates/What_the_Matchers_Read](semantic_templates/What_the_Matchers_Read.md): what the embedding, the layer states and the wells are to a template matcher; why the embedding scores well; particles move away from the wells; what would test the evolution
- [semantic_templates/Template_Matchers_on_the_Dynamics](semantic_templates/Template_Matchers_on_the_Dynamics.md): pre-registration of template matchers that read the layers' displacements and the well frame, against the embedding
- [semantic_templates/Template_Matchers_on_the_Dynamics_SR2_2B](semantic_templates/Template_Matchers_on_the_Dynamics_SR2_2B.md): report of their first run, on SR2-2B: TD2 and TD5 hit, the other three miss; one wording ("unless") drives most of the differences
- [semantic_templates/Template_Matchers_and_Depth](semantic_templates/Template_Matchers_and_Depth.md): pre-registration of the depth arms of the template matchers
- [semantic_templates/Template_Matchers_and_Depth_G2](semantic_templates/Template_Matchers_and_Depth_G2.md): report of the two-layer depth arm (G2)
- [semantic_templates/Template_Matchers_and_Depth_L4](semantic_templates/Template_Matchers_and_Depth_L4.md): report of the four-layer depth arm (L4) and the scoring of TL1 to TL6
- [semantic_templates/Connectives_With_a_Hidden_Negation](semantic_templates/Connectives_With_a_Hidden_Negation.md): the "unless" problem: a connective whose truth function its words do not show, and why the template matchers cannot read it
- [semantic_templates/The_Embedding_Template_Matcher](semantic_templates/The_Embedding_Template_Matcher.md): the embedding-based template matcher step by step, on one premise and its rivals
- [Renormalization_Group_Readings_of_Template_Matching_A_Proposal](Renormalization_Group_Readings_of_Template_Matching_A_Proposal.md): reading template matching through renormalisation-group ideas (proposal)
- [Community_of_Semantic_Templates_for_Native_CoT](Community_of_Semantic_Templates_for_Native_CoT.md): a community of templates as the source of native chain-of-thought
- [Chain_of_Thought_Substrate_in_Lagrangian_Models](Chain_of_Thought_Substrate_in_Lagrangian_Models.md): can chain-of-thought be built from the mechanisms the models have today?
- [The_Roadmap_to_Native_Implementation_of_CoT](The_Roadmap_to_Native_Implementation_of_CoT.md): a roadmap to native chain-of-thought
- [The_Execution_Problem](The_Execution_Problem.md): the execution of semantic structures, deferred in the book
- [Realizing_Execution_Gauge_vs_Causal_Ordering](Realizing_Execution_Gauge_vs_Causal_Ordering.md): two routes to an architecture that executes: a gauge channel or causal ordering
- [Expressivity_Benchmark_Foundations](Expressivity_Benchmark_Foundations.md): foundations of a benchmark for the claim that templates lift the model up the expressivity ladder


## Expressivity and the simulator programme (v0 to v3)

- [Semantic_Simulator_RL_Calibration_Programme](Semantic_Simulator_RL_Calibration_Programme.md): programme memo: a particle simulator in semantic space with force fields calibrated by reinforcement learning
- [Semantic_Simulator_EOM](Semantic_Simulator_EOM.md): the v0 equations of motion of the simulator, with every parameter classified
- [Semantic_Simulator_v15_EOM](Semantic_Simulator_v15_EOM.md): v1.5 (salient decay) equations of motion (forthcoming)
- [Semantic_Simulator_v2_EOM](Semantic_Simulator_v2_EOM.md): v2 (structure creation, Fock space) equations of motion (forthcoming)
- [Semantic_Simulator_v3_EOM](Semantic_Simulator_v3_EOM.md): v3 (structure execution, non-abelian gauge theory) equations of motion (forthcoming)
- [Expressivity_Bounds_For_v0_Simulator](Expressivity_Bounds_For_v0_Simulator.md): the v0 simulator is at most a finite automaton; the predicted Dyck collapse depth
- [MCS_Reduction_For_v3_Composite](MCS_Reduction_For_v3_Composite.md): the v0+v1.5+v2+v3 composite reduces to linear context-free rewriting systems (mildly context-sensitive)
- [Advancing_The_Dynamic_Simulation_Model](Advancing_The_Dynamic_Simulation_Model.md): the mathematical apparatus behind the v1.5, v2 and v3 extensions
- [Augmenting_PARFLM_to_handle_MCS_Languages](Augmenting_PARFLM_to_handle_MCS_Languages.md): augmenting PARFLM with Fock registers toward mildly context-sensitive languages
- [Expressivity_Benchmark_Foundations](Expressivity_Benchmark_Foundations.md): foundations of a benchmark for the claim that templates lift the model up the expressivity ladder
- [Modified_BAOAB_with_STP_identity_Detailed_Analysis](Modified_BAOAB_with_STP_identity_Detailed_Analysis.md): the STP-BAOAB integrator as the backbone of the dynamical-system simulator


## Integrators and CfC+BAOAB

- [Symplectic_Integration_for_SPLM](Symplectic_Integration_for_SPLM.md): velocity-Verlet and Strang splitting for SPLM, and what was measured
- [Multi_Timescale_Hierarchical_Integration_for_SPLM](Multi_Timescale_Hierarchical_Integration_for_SPLM.md): a proposed nested, multi-timescale integrator in place of the flat Verlet pipeline
- [Closed_Form_and_Hybrid_Integration_Strategies_for_Fock-PARFLM](Closed_Form_and_Hybrid_Integration_Strategies_for_Fock-PARFLM.md): closed-form and hybrid strategies for integrating the layer dynamics
- [Blended_CfC_BAOAB_Deep_Dive](Blended_CfC_BAOAB_Deep_Dive.md): the blended integrator: an analytical CfC surrogate for the $V_\theta$ force inside the BAOAB splitting
- [PyTorch_Implementation_of_CfC_BAOAB_in_Fock-PARFLM](PyTorch_Implementation_of_CfC_BAOAB_in_Fock-PARFLM.md): how the CfC+BAOAB integrator is written in PyTorch
- [Markov_Generators_and_the_Langevin_Splitting](Markov_Generators_and_the_Langevin_Splitting.md): why BAOAB has its shape: Markov generators and splittings from the semigroup up
- [Modified_BAOAB_with_STP_identity_Detailed_Analysis](Modified_BAOAB_with_STP_identity_Detailed_Analysis.md): the STP-BAOAB integrator as the backbone of the dynamical-system simulator
- [Efficient_Numerical_Algorithm_on_GPU_for_Dynamical_System_based_Models](Efficient_Numerical_Algorithm_on_GPU_for_Dynamical_System_based_Models.md): integrators on GPUs for semantic particle dynamics: Langevin, BAOAB, the STP identity
- [Exact_Integration_of_Poisson_Mode_Wells_PMX](Exact_Integration_of_Poisson_Mode_Wells_PMX.md): integrating the Poisson-mode wells exactly inside the damped flow (PMX)
- [Composing_Single_Layer_Inferences_Flow_or_Maps](Composing_Single_Layer_Inferences_Flow_or_Maps.md): is the layer stack one flow or a sequence of maps? (verdict: maps)
- [Tech_Report_Position_Dependent_Damping_CfC_BAOAB](Tech_Report_Position_Dependent_Damping_CfC_BAOAB.md): position-dependent damping under CfC+BAOAB: theory, implementation and the four OpenWebText arms
- [Post_100K_CfC_BAOAB_Analysis_Checklist](Post_100K_CfC_BAOAB_Analysis_Checklist.md): working checklist after the 100,000-step CfC+BAOAB $d = 384$ run
- [joint_vtheta_arm/](joint_vtheta_arm/README.md): the $d = 384$ run with a joint $V_\theta$ bank and QK-norm (September 2026), in five self-contained studies: bank geometry and stiffness, the anneal probes, integration refinement, the matched GPT-2 comparison, content-addressed context routing
- [Parallels_and_Lessons_from_Liquid_Neural_Networks](Parallels_and_Lessons_from_Liquid_Neural_Networks.md): a comparison with liquid neural networks (the origin of the CfC integrator) and their path to product


## Langevin dynamics

- [Langevin_dynamics_reformulation_of_classical_damped_Lagrangian_flow](Langevin_dynamics_reformulation_of_classical_damped_Lagrangian_flow.md): the damped Lagrangian flow as the noise-free skeleton of an underdamped Langevin system
- [Markov_Generators_and_the_Langevin_Splitting](Markov_Generators_and_the_Langevin_Splitting.md): why BAOAB has its shape: Markov generators and splittings from the semigroup up
- [Hamiltonian_Flow_in_the_Semantic_Simulation_Framework](Hamiltonian_Flow_in_the_Semantic_Simulation_Framework.md): what the Lagrangian commitment buys, where a Hamiltonian flow exists in the code, and where trained models leave it
- [Efficient_Numerical_Algorithm_on_GPU_for_Dynamical_System_based_Models](Efficient_Numerical_Algorithm_on_GPU_for_Dynamical_System_based_Models.md): integrators on GPUs for semantic particle dynamics: Langevin, BAOAB, the STP identity
- [The_Impact_of_Brownian_Motion_on_Perplexity_in_Attention-Based_Transformers_and_SPLM](The_Impact_of_Brownian_Motion_on_Perplexity_in_Attention-Based_Transformers_and_SPLM.md): the training ODE and inference SDE behind STP, and what Brownian motion does to perplexity in transformers and SPLM


## Damping and the overdamped limit

- [Determining_optimal_gamma_for_SPLM](Determining_optimal_gamma_for_SPLM.md): predicting the optimal damping $\gamma^\ast$ for SPLM, across the leak-correction boundary
- [Determining_optimal_gamma_for_Fock-PARFLM](Determining_optimal_gamma_for_Fock-PARFLM.md): predicting the optimal damping $\gamma^\ast$ for PARFLM and Fock-PARFLM (living document)
- [E4_damping_sweep_pre-registered_protocol](E4_damping_sweep_pre-registered_protocol.md): pre-registered protocol: the SPLM damping sweep E4
- [E4_sweep_results_and_discussion](E4_sweep_results_and_discussion.md): results of the E4 damping sweep
- [Gamma_transfer_pre-registered_protocol](Gamma_transfer_pre-registered_protocol.md): pre-registered protocol: the $\gamma$-transfer re-tuning experiment (E10)
- [Fock-PARFLM_Scale-Up_Gamma_Sweep_Results_and_Damping_Regime_Analysis](Fock-PARFLM_Scale-Up_Gamma_Sweep_Results_and_Damping_Regime_Analysis.md): $\gamma$ sweeps at $d = 384$, 768 and 1024, and the damping regime they show
- [Implicit_vs_Explicit_Damping_and_the_First_vs_Second_Order_Dynamics_Hypothesis](Implicit_vs_Explicit_Damping_and_the_First_vs_Second_Order_Dynamics_Hypothesis.md): implicit against explicit damping, and the first- versus second-order hypothesis
- [The_Overdamped_Limit_and_The_Position_of_The_2nd_Order_Lagrangian_Framework](The_Overdamped_Limit_and_The_Position_of_The_2nd_Order_Lagrangian_Framework.md): where the second-order framework stands given that transformers and SPLM look overdamped
- [Position_Dependent_Damping_and_Reinforcement_Field](Position_Dependent_Damping_and_Reinforcement_Field.md): promoting $\gamma$ to a function $\gamma(h)$, and its link to the reinforcement field
- [Tech_Report_Position_Dependent_Damping_CfC_BAOAB](Tech_Report_Position_Dependent_Damping_CfC_BAOAB.md): position-dependent damping under CfC+BAOAB: theory, implementation and the four OpenWebText arms


## First order against second order

- [First_Order_vs_Second_Order_Dynamical_Systems](First_Order_vs_Second_Order_Dynamical_Systems.md): dynamical order and state-space extension are independent design choices
- [Replacing_The_Conservative_Mechanism_of_SPLM_with_First_Order](Replacing_The_Conservative_Mechanism_of_SPLM_with_First_Order.md): can SPLM's second-order step be replaced by a first-order gradient flow?
- [Implicit_vs_Explicit_Damping_and_the_First_vs_Second_Order_Dynamics_Hypothesis](Implicit_vs_Explicit_Damping_and_the_First_vs_Second_Order_Dynamics_Hypothesis.md): implicit against explicit damping, and the first- versus second-order hypothesis
- [Corpus_Statistics_and_the_First_vs_Second_Order_Well_Gap](Corpus_Statistics_and_the_First_vs_Second_Order_Well_Gap.md): which corpus statistics decide the gap between first- and second-order models with Gaussian wells
- [Future_of_Semantic_Simulation_When_Second_Order_Earns_Its_Keep](Future_of_Semantic_Simulation_When_Second_Order_Earns_Its_Keep.md): when second order earns its keep, after the FO-b-OWT and SR3-crit results


## Jacobi metric and Riemannian geodesics

- [Jacobi_Metric_from_the_Euler_Lagrange_Equations](Jacobi_Metric_from_the_Euler_Lagrange_Equations.md): why trajectories in a potential are geodesics of the Jacobi metric, and what friction changes
- [Damped_Riemannian_Geodesics_in_the_SPLM_family-Comparative_Analysis](Damped_Riemannian_Geodesics_in_the_SPLM_family-Comparative_Analysis.md): damped Riemannian geodesics across the SPLM family (Verlet era; measured on CfC+BAOAB 2026-09-25)
- [Exploiting_the_Riemannian_geometry_of_conservative_language_models](Exploiting_the_Riemannian_geometry_of_conservative_language_models.md): using the Riemannian geometry of conservative models, with the diagnostic battery
- [Geodesic_Preservation_Experiment](Geodesic_Preservation_Experiment.md): do trajectories stay geodesic? Experiment design and analysis pipeline for Fock-PARFLM
- [Composing_Single_Layer_Inferences_Flow_or_Maps](Composing_Single_Layer_Inferences_Flow_or_Maps.md): is the layer stack one flow or a sequence of maps? (verdict: maps)
- [Geodesic_Experiments_with_CfC_BAOAB](Geodesic_Experiments_with_CfC_BAOAB.md): master document of the geodesic programme on CfC+BAOAB: E1, E3, E5
- [Forced_Lagrangian_Reformulation](Forced_Lagrangian_Reformulation.md): the thesis document: reformulating the framework around the non-conservative forces, after the geodesic reading closed
- [Hidden_States_Semantic_Tube_Geodesics_and_Semantic_Structures](Hidden_States_Semantic_Tube_Geodesics_and_Semantic_Structures.md): Huang et al.'s Semantic Tube Prediction compared with the Semantic Simulation framework
- [Geometric_Distance_Metrics_for_SCAF](Geometric_Distance_Metrics_for_SCAF.md): a proposed extension of SCAF with geometric distance metrics


## Attention

- [Multi-Head_Attention_in_Transformer_Block](Multi-Head_Attention_in_Transformer_Block.md): a deep dive into multi-head attention in one transformer block
- [On_The_Smoothness_of_Scaled_Dot_Product_Attention](On_The_Smoothness_of_Scaled_Dot_Product_Attention.md): the smoothness of scaled dot-product attention
- [Attention_Optimality_Conjecture](Attention_Optimality_Conjecture.md): which properties make attention optimal, and what an exchange must have to match it
- [Improving_the_Fock_Mechanism_to_match_Attention](Improving_the_Fock_Mechanism_to_match_Attention.md): Fock attention experiments, register diagnostics, and the routing resolution hierarchy
- [Conservative_Obstruction_and_Virtual_Particle_Necessity](Conservative_Obstruction_and_Virtual_Particle_Necessity.md): why a conservative model needs a virtual-particle exchange to match attention
- [P-rot-6_transformer_dynamics](P-rot-6_transformer_dynamics.md): a damped equation with a velocity-coupled skew force, tied to the key–value asymmetry of attention
- [Two_Clustering_Mechanisms_Saturation_and_Synchronization](Two_Clustering_Mechanisms_Saturation_and_Synchronization.md): how tokens cluster in attention dynamics and in the framework: saturation and synchronization
- [The_Impact_of_Brownian_Motion_on_Perplexity_in_Attention-Based_Transformers_and_SPLM](The_Impact_of_Brownian_Motion_on_Perplexity_in_Attention-Based_Transformers_and_SPLM.md): the training ODE and inference SDE behind STP, and what Brownian motion does to perplexity in transformers and SPLM


## Hybrids: SPLM with attention, Helmholtz architectures

- [Scalar_Potential_based_Helmholtz_Architecture](Scalar_Potential_based_Helmholtz_Architecture.md): SPLM + attention as an explicit Helmholtz (phase-space) decomposition
- [Scalar_Potential_based_Helmholtz_Architecture_v2](Scalar_Potential_based_Helmholtz_Architecture_v2.md): the attention-free Helmholtz realization, first version
- [Scalar_Potential_based_Helmholtz_Architecture_v3](Scalar_Potential_based_Helmholtz_Architecture_v3.md): the attention-free Helmholtz realization, revised
- [HSPLM_Path_Forward_and_Experiments](HSPLM_Path_Forward_and_Experiments.md): experiment record of the hybrid SPLM + attention (HSPLM)
- [Helmholtz-HSPLM_Path_Forward_and_Experiments](Helmholtz-HSPLM_Path_Forward_and_Experiments.md): experiment record of the layer-type Helmholtz hybrid (Q9d)
- [SP_HSPLM_Stage_0_Literature_Check](SP_HSPLM_Stage_0_Literature_Check.md): literature check for the solenoidal-pair hybrid (SP-HSPLM, Maxwell-PARFLM)
- [SP_HSPLM_Stage_1_pre-registered_protocol](SP_HSPLM_Stage_1_pre-registered_protocol.md): pre-registered protocol: SP-HSPLM Stage 1
- [SP_HSPLM_Stage_2_pre-registered_protocol](SP_HSPLM_Stage_2_pre-registered_protocol.md): pre-registered protocol: SP-HSPLM Stage 2, the pair-skew cell ladder
- [SP_HSPLM_Stage_2_q9e_n_verdict_and_structural_reexamination](SP_HSPLM_Stage_2_q9e_n_verdict_and_structural_reexamination.md): SP-HSPLM Stage 2: the q9e_n verdict and a structural re-examination


## Synchronization, clustering and mean-field limits

- [Spontaneous_Synchronization_and_Semantic_Binding_Test_Programme](Spontaneous_Synchronization_and_Semantic_Binding_Test_Programme.md): is spontaneous synchronization at the root of semantic inference? Design and measurements
- [Two_Clustering_Mechanisms_Saturation_and_Synchronization](Two_Clustering_Mechanisms_Saturation_and_Synchronization.md): how tokens cluster in attention dynamics and in the framework: saturation and synchronization
- [Macroscopic_Limits_of_the_Conservative_Step](Macroscopic_Limits_of_the_Conservative_Step.md): replacing per-token physics by field equations: a proposal
- [Mean_Field_Theory_for_Particle_Language_Models](Mean_Field_Theory_for_Particle_Language_Models.md): a study guide to mean-field theory, from its simplest model to machine learning
- [Mean_Field_PARFLM_Programme](Mean_Field_PARFLM_Programme.md): pre-registered programme for a mean-field approximation of PARFLM
- [mean_field/Probe_A_Field_Pair_Force](mean_field/Probe_A_Field_Pair_Force.md): Probe A, scored: a field, and even a context-free landscape, stands in for most of the trained pair force
- [mean_field/Pair_Force_By_Layer_L8_CODE](mean_field/Pair_Force_By_Layer_L8_CODE.md): the pair force layer by layer at eight layers: shared redundantly across depth, not field-like there
- [depth_ladder/Depth_Code_Correction](depth_ladder/Depth_Code_Correction.md): correction record (2026-11-25): five scripts read layer 0 with layer 1's depth code; what was affected and the corrected numbers
- [depth_ladder/Pair_Force_in_the_First_Order_Step](depth_ladder/Pair_Force_in_the_First_Order_Step.md): why FO-b-PMX-K16 is better trained without the pair force: in first order the pair force grows into a push along the landscape and crowds out the wells and the layer-1 landscape


## Causal leaks and auditing

- [Causal_Leak_in_SPLM_Integrate_Bug_and_Fix](Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md): the anti-causal autograd path in SPLM integrate(): bug, mechanism, fix and regression test
- [Causal_Leak_Empirical_Comparison_Report](Causal_Leak_Empirical_Comparison_Report.md): every SPLM checkpoint re-evaluated without the leak
- [Fock-PARFLM_Causal_Leak_Audit_Results](Fock-PARFLM_Causal_Leak_Audit_Results.md): the Fock-PARFLM v2.1 leak: root cause, magnitude and fix
- [Framework_for_Causal_Analysis_SemSimula_Models](Framework_for_Causal_Analysis_SemSimula_Models.md): SCAF, a reusable framework and library that audits any model of the family for causal leaks
- [Geometric_Distance_Metrics_for_SCAF](Geometric_Distance_Metrics_for_SCAF.md): a proposed extension of SCAF with geometric distance metrics


## Training: gradients, memory, tuning

- [Gradient_Spikes_as_Routing_Conjunctions](Gradient_Spikes_as_Routing_Conjunctions.md): gradient spikes are coincidences of a microbatch and a routing draw, not structural defects
- [Resonance_Hypothesis_for_Gradient_Spikes_in_the_LowRank_Kick](Resonance_Hypothesis_for_Gradient_Spikes_in_the_LowRank_Kick.md): a resonance hypothesis for gradient spikes (falsified 2026-09-13)
- [Gradient_Starvation_Investigation](Gradient_Starvation_Investigation.md): gradient starvation across the ladder: the investigation and the gradient paths it opened
- [Hyperparameter_Tuning_Checklist](Hyperparameter_Tuning_Checklist.md): every tunable knob of the ladder, with the evidence on whether it binds
- [Measuring_the_Price_of_Conservativity](Measuring_the_Price_of_Conservativity.md): a $\lambda$-gated relaxation protocol to measure what conservativity costs (proposed)
- [Refinement_Experiments_and_Variable_Step_Training](Refinement_Experiments_and_Variable_Step_Training.md): the refinement experiments of the depth ladder and the variable-step arm SR4a-CODE
- [CUDA_Memory_Errors_with_SPLM_Designs](CUDA_Memory_Errors_with_SPLM_Designs.md): a catalogue of the out-of-memory events in the PARF arms of the May 2026 pilot
- [Structured_VTheta_Memory_Anatomy](Structured_VTheta_Memory_Anatomy.md): why fewer $V_\theta$ parameters do not prevent out-of-memory errors
- [Gradient_Checkpointing_for_PARF](Gradient_Checkpointing_for_PARF.md): per-layer-step gradient checkpointing for the PARF stack


## The book: planning and audits

- [Paper_v5_Restructure_Plan](Paper_v5_Restructure_Plan.md): triage of paper v5 around the forced-Lagrangian thesis
- [Paper_v6_Section_Audit](Paper_v6_Section_Audit.md): section-by-section audit of paper v6


## Writing the notes

- [GitHub_Markdown_LaTeX_Rendering_Cheatsheet](GitHub_Markdown_LaTeX_Rendering_Cheatsheet.md): how to write KaTeX math and Mermaid diagrams that render on GitHub
