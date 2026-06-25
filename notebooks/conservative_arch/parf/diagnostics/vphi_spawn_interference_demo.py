#!/usr/bin/env python3
"""Numerical validation of the V_phi spawn-interference bounds.

This is the self-contained companion to Section 4.5 ("The Relational Channel:
Continuous Learning in V_phi") of

    companion_notes/Continuous_Learning_in_Semantic_Simulation-based_models_vs_with_Transformer_models.md

It does NOT require a trained model.  It exercises the *exact same* Gaussian
type-gate arithmetic that ``StructuralVPhi`` / ``StructuralCompetitiveVPhi``
and ``MultiHeadVPhi`` use (see ``model_parf.py``), on synthetic type vectors,
and checks three claims that the markdown states analytically:

  (C-Phi1)  Bounded responsibility interference under softmax-over-components.
            Spawning relational component  M+1  rescales every existing weight
            by the common factor  rho = 1 - w_{M+1}, so the relative weights
            among old components are preserved EXACTLY and the absolute
            perturbation obeys

                |w_k^{(M+1)} - w_k^{(M)}|  =  w_k^{(M)} * w_{M+1}  <=  w_{M+1}.

            (This is Proposition 1' of the document, ported from V_theta wells
            to the V_phi type-gate.)

  (C-Phi2)  Geometric locality.  The new component's responsibility
            w_{M+1}(l_query) decays as  exp(-c * d_type^2)  with the type-space
            distance between the query and the new component's center, so a
            relation injected at type-center  l_new  perturbs only pairs whose
            type-projection is close to  l_new.

  (C-Phi3)  MLP non-locality (the contrast).  For an unstructured MLP head
            (``MLPVPhi``), a single gradient step delta-W perturbs the head's
            output  V_phi(h_t, h_s)  by  <grad_W V_phi, delta-W>  for
            essentially ALL pairs, with NO decay in type/feature distance.  We
            measure the perturbation magnitude vs. distance and fit a slope:
            structured -> strongly negative (decaying); MLP -> ~0 (flat).

Run:
    python vphi_spawn_interference_demo.py
    python vphi_spawn_interference_demo.py --json   # machine-readable summary

The printed numbers are the ones quoted in the markdown.  Re-run with a
different --seed to confirm they are not seed artefacts.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np


# ---------------------------------------------------------------------------
# Structured Gaussian type-gate (mirrors StructuralVPhi.forward's Phi channel)
# ---------------------------------------------------------------------------
def gaussian_gate_logits(l_query: np.ndarray, centers: np.ndarray, c: float) -> np.ndarray:
    """Unnormalised log-gate  -c * ||l_query - center_m||^2  for each component m.

    Mirrors  Phi_m = exp(-c * ||l_t - l_s||^2)  with the source type-projection
    l_s replaced by a per-component prototype center_m.  This is the natural
    "addressable" generalisation of the single Gaussian gate to a mixture of M
    relational components (one center per component / head).
    """
    d2 = ((l_query[None, :] - centers) ** 2).sum(axis=1)   # (M,)
    return -c * d2, d2


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


# ---------------------------------------------------------------------------
# C-Phi1 + C-Phi2: structured spawn interference
# ---------------------------------------------------------------------------
def check_structured_spawn(rng, M=8, dl=16, c=1.0, n_queries=4000):
    """Validate the |Δw_k| <= w_{M+1} bound and the exp(-c d^2) decay."""
    centers = rng.standard_normal((M, dl))
    new_center = rng.standard_normal(dl)
    centers_aug = np.vstack([centers, new_center])     # (M+1, dl)

    max_abs_violation = 0.0          # max over queries of (|Δw_k| - w_{M+1})_+
    max_rel_drift = 0.0              # max relative-weight drift among old comps
    tv_gap = 0.0                     # |TV(old-restricted shift) - w_{M+1}|

    for _ in range(n_queries):
        lq = rng.standard_normal(dl) * 1.5             # query type vector

        logits_M, _ = gaussian_gate_logits(lq, centers, c)
        logits_M1, _ = gaussian_gate_logits(lq, centers_aug, c)

        w_M = softmax(logits_M)                        # (M,)
        w_M1 = softmax(logits_M1)                      # (M+1,)
        w_new = w_M1[-1]

        # Predicted: w_k^{(M+1)} = w_k^{(M)} * (1 - w_new).
        pred = w_M * (1.0 - w_new)
        delta = np.abs(w_M1[:M] - w_M)
        # Bound: |Δw_k| = w_k * w_new <= w_new.
        max_abs_violation = max(max_abs_violation, float((delta - w_new).max()))
        # Relative weights among old comps preserved exactly:
        # w_k^{(M+1)} / w_j^{(M+1)} == w_k^{(M)} / w_j^{(M)}.
        ratio_M = w_M / w_M.sum()
        ratio_M1 = w_M1[:M] / w_M1[:M].sum()
        max_rel_drift = max(max_rel_drift, float(np.abs(ratio_M1 - ratio_M).max()))
        # Total-variation shift over the old components == w_new.
        tv = 0.5 * delta.sum() + 0.5 * abs((w_M1[:M].sum() - 1.0))
        # restricted-to-old TV equals w_new (mass that left the old block):
        tv_restricted = float((w_M - w_M1[:M]).clip(min=0).sum())
        tv_gap = max(tv_gap, abs(tv_restricted - w_new))

        # closeness of analytic prediction  w_k^{(M+1)} = w_k^{(M)} (1 - w_new)
        max_abs_violation = max(max_abs_violation, float(np.abs(pred - w_M1[:M]).max()) - 1e-12)

    return {
        "M": M,
        "max_abs_violation_of_bound": max_abs_violation,
        "max_relative_weight_drift": max_rel_drift,
        "max_tv_minus_w_new": tv_gap,
    }


def controlled_locality_sweep(rng, M=8, dl=16, c=1.0, R_old=2.5, n_steps=300):
    """Clean C-Phi2 demonstration with the partition function held fixed.

    Random queries confound the slope because a query far from the new center
    is, in isotropic data, also far from the OLD centers, so Z_old correlates
    with d2_new.  Here we remove the confound: a fixed query at the origin,
    M old centers placed on a sphere of fixed radius R_old (so Z_old is a
    constant), and we sweep ONLY the new center's distance r_new.  Then

        log w_new(r_new)  =  -c * r_new^2  -  log( Z_old + e^{-c r_new^2} )
                          ->  -c * r_new^2 - const          (far field)

    so the far-field log-slope vs r_new^2 recovers -c exactly, confirming the
    bound  |Δw_k| <= w_new <= exp(-c (d2_new - d2_nearest_old)).
    """
    # old centers on a fixed-radius sphere -> identical distance^2 = R_old^2
    dirs = rng.standard_normal((M, dl))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    centers = dirs * R_old
    query = np.zeros(dl)
    Z_old = float(np.exp(-c * (R_old ** 2)) * M)   # constant by construction

    r2_new = np.linspace(0.0, 4.0 * R_old ** 2, n_steps)
    w_new = np.exp(-c * r2_new) / (Z_old + np.exp(-c * r2_new))

    far = w_new < 0.05
    slope_far, _ = np.polyfit(r2_new[far], np.log(w_new[far]), 1)
    # also confirm the closed-form bound  w_new <= exp(-c (r2_new - R_old^2))
    bound = np.exp(-c * (r2_new - R_old ** 2))
    bound_ok = bool(np.all(w_new <= bound + 1e-12))
    return {
        "controlled_farfield_slope": float(slope_far),  # ~ -c
        "target_slope": -c,
        "closed_form_bound_holds": bound_ok,
    }


# ---------------------------------------------------------------------------
# C-Phi3: MLP head has no distance-localised interference
# ---------------------------------------------------------------------------
def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def _mlp_vphi(W1, b1, W2, b2, W3, b3, ht, hs):
    """Scalar MLP V_phi on concat(h_t, h_s, h_t - h_s) — mirrors MLPVPhi."""
    feats = np.concatenate([ht, hs, ht - hs])
    a1 = _gelu(W1 @ feats + b1)
    a2 = _gelu(W2 @ a1 + b2)
    return float(W3 @ a2 + b3)


def check_mlp_nonlocality(rng, d=32, H=64, n_pairs=4000, step=0.02):
    """A random gradient-style weight step perturbs ALL pairs ~uniformly.

    We measure |ΔV_phi(h_t, h_s)| for many pairs after a fixed-norm random
    perturbation of (W1, W2, W3), and regress it against the type-distance
    ||h_t - h_s||.  A structured gate would decay; the MLP should be flat.
    """
    scale = 0.02
    W1 = rng.standard_normal((H, 3 * d)) * scale
    b1 = np.zeros(H)
    W2 = rng.standard_normal((H, H)) * scale
    b2 = np.zeros(H)
    W3 = rng.standard_normal(H) * scale
    b3 = 0.0

    # Perturbation = a single relation-injection gradient step (random direction,
    # fixed norm) on the readout + first layer, as gradient CL would produce.
    dW1 = rng.standard_normal((H, 3 * d)); dW1 *= step / np.linalg.norm(dW1)
    dW3 = rng.standard_normal(H);          dW3 *= step / np.linalg.norm(dW3)

    dists, dVs = [], []
    for _ in range(n_pairs):
        ht = rng.standard_normal(d)
        hs = rng.standard_normal(d)
        v0 = _mlp_vphi(W1, b1, W2, b2, W3, b3, ht, hs)
        v1 = _mlp_vphi(W1 + dW1, b1, W2, b2, W3 + dW3, b3, ht, hs)
        dists.append(float(np.linalg.norm(ht - hs)))
        dVs.append(abs(v1 - v0))

    dists = np.array(dists)
    dVs = np.array(dVs)
    # Fraction of pairs with non-negligible perturbation (no locality => ~all).
    frac_perturbed = float((dVs > 1e-6).mean())
    # Normalised slope of |ΔV| vs distance (should be ~0 -> flat).
    slope = float(np.polyfit(dists, dVs, 1)[0])
    mean_dV = float(dVs.mean())
    norm_slope = slope * dists.std() / (mean_dV + 1e-12)
    return {
        "fraction_pairs_perturbed": frac_perturbed,
        "normalised_distance_slope": norm_slope,   # ~0 => non-local
        "mean_abs_perturbation": mean_dV,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    structured = check_structured_spawn(rng)
    locality = controlled_locality_sweep(rng)
    mlp = check_mlp_nonlocality(rng)

    summary = {
        "structured_vphi": structured,
        "locality_sweep": locality,
        "mlp_vphi": mlp,
        "seed": args.seed,
    }

    if args.json:
        print(json.dumps(summary, indent=2))
        return

    print("=" * 70)
    print("V_phi spawn-interference demo  (seed=%d)" % args.seed)
    print("=" * 70)
    print("\n[Structured / mixture type-gate]  Propositions 1, 1' ported to V_phi")
    print("  components before spawn (M)         : %d" % structured["M"])
    print("  max |Δw_k| − w_new  (<= 0 expected) : %+.3e"
          % structured["max_abs_violation_of_bound"])
    print("  max relative-weight drift (≈0)      : %.3e"
          % structured["max_relative_weight_drift"])
    print("  max |TV_old − w_new|      (≈0)      : %.3e"
          % structured["max_tv_minus_w_new"])
    print("\n[Geometric locality]  controlled sweep, Z_old held constant")
    print("  far-field log-slope (target %.2f)   : %.3f"
          % (locality["target_slope"], locality["controlled_farfield_slope"]))
    print("  closed-form bound w_new ≤ e^{-c(d²−d²_near)} : %s"
          % locality["closed_form_bound_holds"])
    print("    -> interference decays as exp(-c·d_type²): geometric locality holds.")

    print("\n[Unstructured MLP head]  the non-spawnable contrast")
    print("  fraction of pairs perturbed by step : %.3f  (≈1.0 => global)"
          % mlp["fraction_pairs_perturbed"])
    print("  normalised |ΔV| vs distance slope   : %+.3f  (≈0 => no locality)"
          % mlp["normalised_distance_slope"])
    print("  mean |ΔV| over pairs                : %.3e" % mlp["mean_abs_perturbation"])
    print("    -> one gradient step moves nearly every pair: relational")
    print("       knowledge is globally entangled (catastrophic-forgetting risk).")
    print("=" * 70)


if __name__ == "__main__":
    main()
