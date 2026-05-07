"""Helmholtz architecture package (Q9d).

A scalar-potential-based Helmholtz hybrid LM in which the four
phase-space force components of the v3 paper's named decomposition
(A.130) are carried by physically distinct architectural carriers:

  - S-blocks (autonomous, conservative): one shared scalar potential
    V_theta(xi, h) drives a velocity-Verlet damped Euler-Lagrange step.
  - A-blocks (non-autonomous, Hopfield + small skew): standard pre-LN
    attention + MLP residual block with per-layer parameters.

The depth schedule sigma : {0..L-1} -> {S, A} controls the architectural
shape (sandwich, interleaved, top-A, bottom-A, etc.).

See companion_notes/Scalar_Potential_based_Helmholtz_Architecture.md for the full
construction and predictions.
"""
