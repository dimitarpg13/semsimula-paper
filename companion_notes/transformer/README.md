# Transformer: Index

**What this folder is.** Notes on the attention-based transformer read as a dynamical system: depth as time, each layer as a step of a discrete-time flow, the tokens as interacting particles. They hold the literature and what it implies for the comparison with the conservative models of the Semantic Simulation framework. Started 2026-11-27, at the author's request, during the comparison of SR2-NP-L8 with the matched GPT-2. Figures: `figures/transformer/`, made by [`_make_transformer_dynamics_figs.py`](../figures/_make_transformer_dynamics_figs.py).

| note | what it holds |
| --- | --- |
| [Transformers_as_Dynamical_Systems.md](Transformers_as_Dynamical_Systems.md) | a reading guide: the residual step as an Euler step, a layer as a split integrator step (Lu et al.), tokens clustering on the sphere (Geshkovski, Letrouit, Polyanskiy, Rigollet), when attention is a gradient flow (Sinkformers, Hopfield), rank collapse, what the theorems assume, and the comparison with the conservative step |

**Related notes elsewhere.**

- [Two_Clustering_Mechanisms_Saturation_and_Synchronization.md](../Two_Clustering_Mechanisms_Saturation_and_Synchronization.md): the clustering theorems in depth, the causal mask, and synchronisation.
- [P-rot-6_transformer_dynamics.md](../P-rot-6_transformer_dynamics.md): a damped equation with a velocity-coupled skew force, tied to the key–value asymmetry of attention.
- [Evidence_for_second_order_ODE_governing_evolution.md](../Evidence_for_second_order_ODE_governing_evolution.md): fitting equations of motion to the hidden states of trained transformers.
- [semantic_templates/Depth_and_Architecture_Battery.md](../semantic_templates/Depth_and_Architecture_Battery.md): the template tests run on the matched GPT-2 and SR2-NP-L8 at equal depth.
- [semantic_templates/Deploying_Semantic_Templates.md](../semantic_templates/Deploying_Semantic_Templates.md): deploying templates in a transformer (Part II there): probes, steering, editing, a template module, inserted tokens, emitted text, and why a trained transformer has no landscape to hold a well.
