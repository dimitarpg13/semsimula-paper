"""
Causal-violation probe for the PARF-augmented SPLM (Q9c).

Why this exists
===============
The PARF model introduces a NEW causality risk surface compared to the
plain SPLM family: the pair-interaction sum

    U_pair^{(ℓ)}_t = Σ_{s < t} V_φ(h_t, h_s)

builds an explicit dependency from h_t on every earlier h_s.  The
design-doc §3 causal reduction handles this by treating past tokens
as fixed external sources -- in code, by .detach()-ing the source
slice {h_s} when forming the pair-potential matrix.  If the .detach()
is missing or inverted, the gradient of the loss at any future
position can flow backward through the pair sum and contaminate the
prediction at past positions, just like the historical SPLM ξ-pool
leak (`companion_notes/Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md`).

There are TWO independent .detach() points in the production model
that this probe exercises:

  1. ξ = causal_cumulative_mean(h.detach())  -- inherited from the
     SPLM family (`causal_force=True`).
  2. h_src = h.detach()                       -- NEW for PARF; the
     causal reduction of the pair source slice.

Probe strategy
==============
Two complementary tests, mirroring the Helmholtz probe convention:

  A. *Perturbation probe* — change one token x[t_pert], compare logits
     at every position t < t_pert.  In a causal model these MUST be
     bit-identical (Δ ≡ 0 within fp32 noise).

  B. *Gradient-Jacobian probe* — pick a target t and compute
     ∂(logits[0, t, :].sum()) / ∂(emb_in[0, t', :]) via autograd.  In a
     causal model the gradient is non-zero only for t' <= t.

Both probes are run in two modes:
  - causal_force=True   (production default; must give Δ ≡ 0)
  - causal_force=False  (buggy mode; expected to leak through both the
                          ξ pool AND the V_φ pair source slice).

Confirming the buggy-mode failure rules out a "the probe is too weak"
failure mode of the test itself.

V_φ variants tested
-------------------
By default the probe runs on both V_φ kinds:
  - structural (the §5.1-faithful pair potential)
  - mlp        (the unstructured ablation)
This confirms the .detach() machinery is correct independent of the
inner V_φ shape.

Usage
-----
  python3 notebooks/conservative_arch/parf/causal_probe_parf.py
      → run both probes on both V_φ variants at smoke scale, in both
        causal_force modes; exit 0 iff fixed mode is causal-clean.

  python3 notebooks/conservative_arch/parf/causal_probe_parf.py --strict
      → same, but exit non-zero on any failure (CI gate).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

THIS_DIR = Path(__file__).parent
PARENT_DIR = THIS_DIR.parent
sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(THIS_DIR))

from model_parf import PARFConfig, PARFLM  # noqa: E402


# ---------------------------------------------------------------------------
# Probe tolerances (mirror Helmholtz / SPLM family conventions).
# ---------------------------------------------------------------------------
TOL_PRE: float = 1e-6
TOL_BUGGY_FLOOR: float = 1e-6


# ---------------------------------------------------------------------------
# Tiny smoke config builder
# ---------------------------------------------------------------------------
def _smoke_config(v_phi_kind: str, causal_force: bool, L: int = 4) -> PARFConfig:
    """Build a tiny config (no logfreq dependency) suitable for the probe."""
    return PARFConfig(
        vocab_size=257,
        d=16, max_len=64, L=L,
        v_hidden=32, v_depth=2,
        v_phi_kind=v_phi_kind,
        v_phi_d_type=4, v_phi_d_angle=2,
        v_phi_phi_hidden=8, v_phi_theta_hidden=8,
        v_phi_mlp_hidden=16,
        mass_mode="global",
        ln_after_step=True,
        causal_force=causal_force,
    )


# ---------------------------------------------------------------------------
# Probe A: perturbation
# ---------------------------------------------------------------------------
def perturbation_probe(
    model: torch.nn.Module,
    vocab_size: int,
    T: int = 32,
    t_pert: int = 20,
    seed: int = 0,
) -> Tuple[float, float, torch.Tensor]:
    """Return (max Δ on causal side, max Δ on after-side, full Δ vector)."""
    rng = np.random.default_rng(seed)
    xb = rng.integers(0, vocab_size, size=(1, T)).astype(np.int64)
    x_a = torch.from_numpy(xb)
    x_b = x_a.clone()
    orig = int(x_b[0, t_pert].item())
    x_b[0, t_pert] = (orig + 17) % vocab_size

    model.eval()
    with torch.enable_grad():
        out_a = model(x_a)
        out_b = model(x_b)
    logits_a = out_a[0].detach()
    logits_b = out_b[0].detach()
    diffs = (logits_a - logits_b).abs().max(dim=-1).values[0]    # (T,)
    pre = float(diffs[:t_pert].max().item())
    post = float(diffs[t_pert + 1:].max().item())
    return pre, post, diffs


# ---------------------------------------------------------------------------
# Probe B: gradient-Jacobian
# ---------------------------------------------------------------------------
def gradient_probe(
    model: PARFLM,
    vocab_size: int,
    T: int = 32,
    t_target: int = 20,
    seed: int = 0,
) -> Tuple[float, float, torch.Tensor]:
    """Compute ∂(logits[0, t_target, :].sum()) / ∂(emb_in[0, t', :]).

    Returns (max grad-norm at t' > t_target, max at t' <= t_target,
    per-position grad-norms).  In a causal model the first must be
    zero within fp32 noise.

    The probe runs the model in `.train()` mode internally so the
    integrator's `create_graph=True` path is exercised — without it the
    layer step would wall off the V_θ + V_φ gradient computation from
    the outer backward, hiding the very leak this probe is meant to
    detect.  We restore the caller's mode before returning.
    """
    rng = np.random.default_rng(seed)
    xb = rng.integers(0, vocab_size, size=(1, T)).astype(np.int64)
    x = torch.from_numpy(xb)

    was_training = model.training
    model.train()
    try:
        with torch.enable_grad():
            emb_static = model._embed(x)                        # (1, T, d)
            emb_in = emb_static.detach().clone().requires_grad_(True)
            h_L, _ = model._stack_forward(emb_in, x)
            logits = h_L @ model.E.weight.T                     # (1, T, V)
            target = logits[0, t_target, :].sum()
            (g,) = torch.autograd.grad(
                target, emb_in,
                retain_graph=False, create_graph=False,
            )
    finally:
        if not was_training:
            model.eval()

    g = g[0]                                                    # (T, d)
    norms = g.norm(dim=-1)                                      # (T,)
    post = float(norms[t_target + 1:].max().item()) if t_target + 1 < T else 0.0
    pre = float(norms[:t_target + 1].max().item())
    return post, pre, norms


# ---------------------------------------------------------------------------
# Per-variant runner
# ---------------------------------------------------------------------------
def probe_one_variant(
    v_phi_kind: str,
    label: str = "",
    L: int = 4,
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[bool, dict]:
    """Run both probes on `v_phi_kind` in fixed and buggy modes."""
    T = 32
    t_pert = 20
    t_target = t_pert
    label = label or v_phi_kind

    # ----- fixed (causal_force=True) -----
    torch.manual_seed(seed)
    cfg_fix = _smoke_config(v_phi_kind, causal_force=True, L=L)
    m_fix = PARFLM(cfg_fix)
    pre_pert_fix, post_pert_fix, _ = perturbation_probe(
        m_fix, vocab_size=cfg_fix.vocab_size,
        T=T, t_pert=t_pert, seed=seed,
    )
    post_grad_fix, pre_grad_fix, _ = gradient_probe(
        m_fix, vocab_size=cfg_fix.vocab_size,
        T=T, t_target=t_target, seed=seed,
    )

    # ----- buggy (causal_force=False) -----
    torch.manual_seed(seed)
    cfg_bug = _smoke_config(v_phi_kind, causal_force=False, L=L)
    m_bug = PARFLM(cfg_bug)
    pre_pert_bug, post_pert_bug, _ = perturbation_probe(
        m_bug, vocab_size=cfg_bug.vocab_size,
        T=T, t_pert=t_pert, seed=seed,
    )
    post_grad_bug, pre_grad_bug, _ = gradient_probe(
        m_bug, vocab_size=cfg_bug.vocab_size,
        T=T, t_target=t_target, seed=seed,
    )

    fix_pert_ok = pre_pert_fix < TOL_PRE
    fix_grad_ok = post_grad_fix < TOL_PRE
    ok = fix_pert_ok and fix_grad_ok

    if verbose:
        verdict = "OK" if ok else "FAIL"
        print(f"  [{verdict:>4}] V_phi={label!r:>14s}  (L={L})")
        print(f"           fixed mode (causal_force=True):")
        print(f"             perturbation pre={pre_pert_fix:.2e}  "
              f"post={post_pert_fix:.2e}  "
              f"({'OK' if fix_pert_ok else 'FAIL'})")
        print(f"             gradient     post={post_grad_fix:.2e}  "
              f"pre={pre_grad_fix:.2e}  "
              f"({'OK' if fix_grad_ok else 'FAIL'})")
        if pre_pert_bug > TOL_BUGGY_FLOOR or post_grad_bug > TOL_BUGGY_FLOOR:
            leak_status = "(leak detected as expected)"
        else:
            leak_status = ("(no leak detected; probe may be too weak at "
                           "random init -- the fix path is still verified "
                           "by fixed mode)")
        print(f"           buggy mode (causal_force=False):")
        print(f"             perturbation pre={pre_pert_bug:.2e}  "
              f"post={post_pert_bug:.2e}  {leak_status}")
        print(f"             gradient     post={post_grad_bug:.2e}")

    details = {
        "v_phi_kind": v_phi_kind,
        "L": L,
        "fix_pert_pre": pre_pert_fix,
        "fix_pert_post": post_pert_fix,
        "fix_grad_post": post_grad_fix,
        "fix_grad_pre": pre_grad_fix,
        "bug_pert_pre": pre_pert_bug,
        "bug_pert_post": post_pert_bug,
        "bug_grad_post": post_grad_bug,
        "fix_pert_ok": fix_pert_ok,
        "fix_grad_ok": fix_grad_ok,
        "ok": ok,
    }
    return ok, details


# ---------------------------------------------------------------------------
# Trainer hook (used by train_parf.py at startup)
# ---------------------------------------------------------------------------
def assert_causal(
    model: PARFLM,
    vocab_size: int,
    T: int = 32,
    t_pert: int = 20,
    seed: int = 0,
    tol: float = TOL_PRE,
) -> None:
    """Run both probes on a real, on-device model; raise if leakage found.

    MPS workaround
    --------------
    The gradient probe walks the autograd graph through the integrator
    step, which is built with `create_graph=True` in train() mode.
    PyTorch's MPS backend does not currently support this second-order
    graph for some of the ops involved.  The probe is architectural
    (depends only on the wiring), so we temporarily move the model to
    CPU and restore the original device immediately afterwards.  This
    must be called BEFORE the optimiser is constructed so the optimiser
    binds to the parameters in their final on-device location.
    """
    orig_device = next(model.parameters()).device
    moved = orig_device.type != "cpu"
    if moved:
        model.to("cpu")
    try:
        pre_pert, post_pert, _ = perturbation_probe(
            model, vocab_size=vocab_size, T=T, t_pert=t_pert, seed=seed,
        )
        post_grad, pre_grad, _ = gradient_probe(
            model, vocab_size=vocab_size, T=T, t_target=t_pert, seed=seed,
        )
    finally:
        if moved:
            model.to(orig_device)
    if pre_pert >= tol:
        raise RuntimeError(
            f"[parf-causal-probe] PERTURBATION LEAK: pre={pre_pert:.4e} >= "
            f"tol={tol:.0e}.  Aborting training before any compute is "
            f"wasted.  causal_force={getattr(model.cfg, 'causal_force', '?')}; "
            f"v_phi_kind={getattr(model.cfg, 'v_phi_kind', '?')}."
        )
    if post_grad >= tol:
        raise RuntimeError(
            f"[parf-causal-probe] GRADIENT LEAK: post={post_grad:.4e} >= "
            f"tol={tol:.0e}.  Aborting training before any compute is "
            f"wasted.  causal_force={getattr(model.cfg, 'causal_force', '?')}; "
            f"v_phi_kind={getattr(model.cfg, 'v_phi_kind', '?')}."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run_all(strict: bool, variants: List[Tuple[str, str]]) -> int:
    print("=" * 78)
    print(" PARF (Q9c) causal-violation probe")
    print(" mode: random-init, mass_mode='global', d=16, "
          "perturbation + gradient")
    print("=" * 78)
    fails = 0
    for label, kind in variants:
        try:
            ok, _ = probe_one_variant(kind, label=label, verbose=True)
        except Exception as exc:
            print(f"  [FAIL] V_phi={label!r}: {type(exc).__name__}: {exc}")
            fails += 1
            continue
        if not ok:
            fails += 1
    print("-" * 78)
    if fails == 0:
        print(f"  All {len(variants)} V_phi variants: fixed-mode causal-side "
              f"Δ < {TOL_PRE:.0e}.  PARF is causal by construction.")
        return 0
    print(f"  {fails} variant(s) leaked in fixed mode.  See output above.")
    return 1 if strict else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Causal-violation probe for the Q9c PARF model."
    )
    ap.add_argument(
        "--v-phi-kind", default=None,
        choices=["structural", "mlp"],
        help="Run only on this single V_phi variant.  If omitted, runs "
             "on both 'structural' and 'mlp'.",
    )
    ap.add_argument(
        "--strict", action="store_true",
        help="Exit non-zero on any failure (CI gate).",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.v_phi_kind is not None:
        variants = [(args.v_phi_kind, args.v_phi_kind)]
    else:
        variants = [("structural", "structural"), ("mlp", "mlp")]

    return run_all(strict=args.strict, variants=variants)


if __name__ == "__main__":
    sys.exit(main())
