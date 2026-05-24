"""SP-HSPLM (Q9(e), Maxwell-PARFLM) — Stage 2 attention-free Helmholtz hybrid.

Pre-registered protocol
-----------------------
docs/SP_HSPLM_Stage_2_pre-registered_protocol.md

Architecture (one-paragraph summary)
------------------------------------
SP-HSPLM is a depth-L stack of two block types:

- S-blocks carry the autonomous conservative class F_S. The force is the
  SPLM gradient flow given by the per-token scalar potential V_theta,
  enriched by the SparsePARFLM pair scalar V_phi via Gumbel-softmax top-k
  routing. Identical to SparsePARFLM's _layer_step.

- C-blocks (for circulation) carry the autonomous solenoidal class
  F_sol. The force is a causal pair-interaction skew-matrix force on the
  velocity proxy, summed under the same Gumbel-softmax top-k mask:
        f_t^sol = sum_{s<t} m_{ts} * J_phi (delta_s - delta_t)
  with J_phi = U V^T - V U^T constant low-rank skew matrix, and
  optionally a per-token gyroscopic term Omega(h_t) delta_t.

No attention block is allocated anywhere in the SP-HSPLM stack -- this
is Q9(e)'s defining commitment.

Design references
-----------------
- docs/Scalar_Potential_based_Helmholtz_Architecture_v3.md
- docs/SP_HSPLM_Stage_0_Literature_Check.md
- docs/SP_HSPLM_Stage_2_pre-registered_protocol.md
"""
