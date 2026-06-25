#!/usr/bin/env python3
"""Measure the relational knowledge fraction  phi_rel  of a trained PARFLM.

Companion to Section 4.5 ("The Relational Channel: Continuous Learning in
V_phi") of

    companion_notes/Continuous_Learning_in_Semantic_Simulation-based_models_vs_with_Transformer_models.md

The continuous-learning story spawns wells in V_theta (the one-body / attractor
channel).  Relational knowledge — how token i relates to token j — lives in
V_phi.  Before deciding how hard to work on relational CL, we want to know how
much predictive capacity is even AT STAKE in V_phi.  Define the
*relational fraction*

    phi_rel  =  ( L(V_phi = 0)  -  L(full) )  /  L(full)

i.e. the fractional cross-entropy increase when the pair potential is ablated
to zero (the one-body V_theta + Verlet dynamics still run).  phi_rel ~ 0 means
the model barely uses V_phi (relational CL is low-value); phi_rel large means
the relational channel carries real knowledge and the structured-vs-MLP V_phi
choice matters a lot for continuous learning.

This module is **read-only** and reuses an already-built ``model`` object — it
does NOT rebuild the config or touch the training checkpoint, so it is safe to
run in a *separate* Colab runtime against a saved checkpoint while the main
run continues.

Usage inside a notebook that already has ``model`` loaded (e.g. after the
"resume from checkpoint" cell of colab_fock_multihead_openwebtext.ipynb):

    from measure_relational_fraction import relational_fraction
    res = relational_fraction(
        model, get_batch, val_ids,
        batch_size=BATCH_SIZE, block_size=BLOCK_SIZE,
        eval_iters=EVAL_ITERS, device=DEVICE, rng=rng,
    )
    print(res)

See the bottom of this file for a fully self-contained copy-paste cell.
"""

from __future__ import annotations

import contextlib
import math

import numpy as np
import torch
import torch.nn as nn


class _ZeroVPhi(nn.Module):
    """Drop-in V_phi that returns an identically-zero pair potential.

    Matches the dense ``forward(h, h_src) -> (B, T, T)`` and gathered
    ``forward_gathered(h, h_src_g) -> (B, T, k)`` contracts used by
    StructuralVPhi / MultiHeadVPhi / CompositeVPhi, so the force
    -grad_h sum_s V_phi = 0 and the layer step runs on V_theta alone.
    """

    def forward(self, h: torch.Tensor, h_src: torch.Tensor) -> torch.Tensor:
        B, T, _ = h.shape
        return h.new_zeros(B, T, h_src.shape[1])

    def forward_gathered(self, h: torch.Tensor, h_src_g: torch.Tensor) -> torch.Tensor:
        B, T, _ = h.shape
        k = h_src_g.shape[2]
        return h.new_zeros(B, T, k)


@contextlib.contextmanager
def zero_vphi(model):
    """Temporarily replace ``model.V_phi`` with a zero pair potential."""
    saved = model.V_phi
    model.V_phi = _ZeroVPhi().to(next(model.parameters()).device)
    try:
        yield
    finally:
        model.V_phi = saved


@torch.no_grad()
def _mean_loss(model, get_batch, val_ids, batch_size, block_size,
               eval_iters, device, rng):
    """Mean NTP loss over ``eval_iters`` val batches, mirroring evaluate()."""
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(eval_iters):
        xb, yb = get_batch(val_ids, batch_size, block_size, rng)
        x = torch.from_numpy(xb).to(device)
        y = torch.from_numpy(yb).to(device)
        # The forward pass computes forces via autograd even at eval time, so
        # enable_grad is required (matches the notebook's evaluate()).
        with torch.enable_grad():
            _, loss = model(x, y)
        losses.append(float(loss.item()))
    if was_training:
        model.train()
    return float(np.mean(losses))


def relational_fraction(model, get_batch, val_ids, *, batch_size, block_size,
                        eval_iters, device, rng):
    """Compute phi_rel and the two component losses/perplexities.

    Returns a dict with full / V_phi-ablated loss and PPL and the relational
    fraction phi_rel = (L_ablated - L_full) / L_full.
    """
    l_full = _mean_loss(model, get_batch, val_ids, batch_size, block_size,
                        eval_iters, device, rng)
    with zero_vphi(model):
        l_abl = _mean_loss(model, get_batch, val_ids, batch_size, block_size,
                           eval_iters, device, rng)
    phi_rel = (l_abl - l_full) / l_full
    return {
        "loss_full": l_full,
        "ppl_full": math.exp(l_full),
        "loss_vphi_zero": l_abl,
        "ppl_vphi_zero": math.exp(l_abl),
        "phi_rel": phi_rel,
        "ppl_ratio": math.exp(l_abl) / math.exp(l_full),
    }


# ---------------------------------------------------------------------------
# Self-contained Colab cell (copy-paste into a NEW runtime after the model and
# checkpoint are loaded; it reuses model / get_batch / val_ids / etc.):
# ---------------------------------------------------------------------------
_COLAB_CELL = r'''
# --- Relational knowledge fraction phi_rel (read-only diagnostic) ----------
import contextlib, math, numpy as np, torch, torch.nn as nn

class _ZeroVPhi(nn.Module):
    def forward(self, h, h_src):
        B, T, _ = h.shape
        return h.new_zeros(B, T, h_src.shape[1])
    def forward_gathered(self, h, h_src_g):
        B, T, _ = h.shape
        return h.new_zeros(B, T, h_src_g.shape[2])

@contextlib.contextmanager
def zero_vphi(model):
    saved = model.V_phi
    model.V_phi = _ZeroVPhi().to(next(model.parameters()).device)
    try:
        yield
    finally:
        model.V_phi = saved

@torch.no_grad()
def _mean_loss():
    model.eval(); losses = []
    for _ in range(EVAL_ITERS):
        xb, yb = get_batch(val_ids, BATCH_SIZE, BLOCK_SIZE, rng)
        x = torch.from_numpy(xb).to(DEVICE); y = torch.from_numpy(yb).to(DEVICE)
        with torch.enable_grad():
            _, loss = model(x, y)
        losses.append(float(loss.item()))
    model.train(); return float(np.mean(losses))

l_full = _mean_loss()
with zero_vphi(model):
    l_abl = _mean_loss()
phi_rel = (l_abl - l_full) / l_full
print(f"L_full        = {l_full:.4f}  (PPL {math.exp(l_full):.2f})")
print(f"L_{{V_phi=0}}   = {l_abl:.4f}  (PPL {math.exp(l_abl):.2f})")
print(f"phi_rel       = {phi_rel:.4f}   ({100*phi_rel:.1f}% CE increase when V_phi ablated)")
print(f"PPL ratio     = {math.exp(l_abl)/math.exp(l_full):.2f}x")
'''


if __name__ == "__main__":
    print("This module is import-only; it needs a built `model` and val data.")
    print("Copy the cell below into a Colab runtime with the model loaded:\n")
    print(_COLAB_CELL)
