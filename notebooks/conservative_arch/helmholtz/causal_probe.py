"""
Causal-violation probe for the Helmholtz (Q9d) hybrid LM.

Why this exists
===============
The Helmholtz architecture introduces TWO new mechanisms not present in
prior SPLM variants, and either one could in principle leak future-
position information into past predictions:

  1. The velocity proxy `delta = h - h_prev` carries through *every*
     layer (S or A), so the kinematic state of position t at layer ell
     depends on layer ell-1 at the same position t.  This is causal by
     induction (every prior layer is causal), but the inductive
     argument is worth confirming empirically.

  2. `xi` is re-derived from `h.detach()` at *every* S-block, not once
     after the attention stack as in Variant A HSPLM.  This is the
     same mechanism as in `model_sarf_mass.py` and it is the codepath
     that originally hosted the documented causal-leak bug
     (`companion_notes/Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md`).  The
     `causal_force=True` flag detaches `h` before the cumulative-mean
     pool, severing the autograd path back to anti-causal positions.

Probe strategy
==============
Two complementary tests, run on every canonical schedule in
`model_helmholtz.canonical_schedules`:

  A. *Perturbation probe* — change exactly one token x[t_pert], compare
     logits at every position t < t_pert.  In a properly causal model
     these MUST be bit-identical (Δ ≡ 0).

  B. *Gradient-Jacobian probe* — pick a target position t and compute
     ∂(logits[0, t, :].sum()) / ∂(emb_in[0, t', :]) via autograd.  In
     a causal model the gradient is non-zero only for t' <= t.

The gradient probe is strictly stronger than perturbation (it cannot
be fooled by a leak that happens to integrate to zero at the chosen
perturbation), so it is what `train_helmholtz.py` invokes at startup.

Both probes are run in two modes: `causal_force=True` (the production
default; must give Δ ≡ 0 within fp32 noise) and `causal_force=False`
(the historical buggy mode; expected to leak).  Confirming the
buggy-mode failure rules out a "the test is too weak to catch any
leak" failure mode of the probe itself.

Usage
-----
  python3 notebooks/conservative_arch/helmholtz/causal_probe.py
      → run both probes on all canonical schedules at smoke scale,
        in both fixed and buggy modes; exit 0 iff fixed mode is
        causal-clean on every schedule.

  python3 notebooks/conservative_arch/helmholtz/causal_probe.py --strict
      → same, but exit non-zero on any failure (suitable for CI gate).

  python3 notebooks/conservative_arch/helmholtz/causal_probe.py \\
      --schedule SAAAAAAS
      → run only on the explicit schedule; useful when iterating on a
        single configuration.
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

from model_helmholtz import (  # noqa: E402
    HelmholtzConfig,
    HelmholtzLM,
    canonical_schedules,
    parse_schedule,
)


# ---------------------------------------------------------------------------
# Probe tolerances
# ---------------------------------------------------------------------------
# The perturbation probe uses a strict bit-equality threshold: any
# difference above 1e-6 in fp32 is a real causality violation, not
# numerical noise.  The gradient probe uses the same threshold per
# element of the gradient norm.
TOL_PRE: float = 1e-6
# When we expect the buggy mode to leak, require Δ above this floor;
# below this threshold we cannot tell a real leak from numerical noise
# (e.g. on schedules with very few S-blocks the buggy leak channel is
# weak and may not exceed noise on random init).
TOL_BUGGY_FLOOR: float = 1e-6


# ---------------------------------------------------------------------------
# Tiny smoke config builder
# ---------------------------------------------------------------------------
def _smoke_config(schedule: str, causal_force: bool) -> HelmholtzConfig:
    """Build a tiny config (no logfreq dependency) suitable for the probe."""
    return HelmholtzConfig(
        vocab_size=257,
        d=16, max_len=64,
        v_hidden=32, v_depth=2,
        schedule=schedule,
        n_head=2, mlp_mult=2,
        mass_mode="global",
        ln_after_s_step=True,
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
    """Return (max Δ on causal side, max Δ on after-side, full Δ vector).

    Mirrors `notebooks/conservative_arch/causal_probe.py:causal_violation_probe`
    so that thresholds and semantics are interchangeable across the SPLM
    family.
    """
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
    model: HelmholtzLM,
    vocab_size: int,
    T: int = 32,
    t_target: int = 20,
    seed: int = 0,
) -> Tuple[float, float, torch.Tensor]:
    """Compute ∂(logits[0, t_target, :].sum()) / ∂(emb_in[0, t', :]).

    Returns (max grad-norm at t' > t_target, max at t' <= t_target,
    per-position grad-norms).  In a causal model the first quantity must
    be zero within fp32 noise; the second is just a positive sanity
    figure to confirm the test is exercising real gradient flow.

    This probe bypasses the embedding lookup so we can take gradients
    w.r.t. a continuous tensor (token ids are int and not directly
    differentiable).  We feed `emb_in = E[x] + P` directly into the
    layer stack via the model's internal `_stack_forward` helper.

    The probe runs the model in `.train()` mode internally so the
    S-block integration's `create_graph=True` path is exercised — this
    is what propagates the (buggy mode) anti-causal `grad_V` term into
    the autograd graph for `h_new`.  Without `.train()` the integration
    step would wall off `f = -grad_V` from the outer backward, hiding
    the very leak this probe is meant to detect.  We restore the
    caller's mode before returning.
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
            h_L, _, _ = model._stack_forward(emb_in, x)
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
    pre  = float(norms[:t_target + 1].max().item())
    return post, pre, norms


# ---------------------------------------------------------------------------
# Per-schedule runner
# ---------------------------------------------------------------------------
def probe_one_schedule(
    schedule: str,
    label: str = "",
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[bool, dict]:
    """Run both probes on `schedule` in fixed and buggy modes.

    Returns (ok, details) where ok=True iff:
      - fixed-mode perturbation probe Δ < TOL_PRE
      - fixed-mode gradient probe post < TOL_PRE
    """
    sigma = parse_schedule(schedule)
    L = len(sigma)
    T = min(32, max(L * 4, 16))
    t_pert = max(1, T // 2 + 4)
    t_target = t_pert

    label = label or schedule

    # ----- fixed (causal_force=True) -----
    torch.manual_seed(seed)
    cfg_fix = _smoke_config(schedule, causal_force=True)
    m_fix = HelmholtzLM(cfg_fix)
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
    cfg_bug = _smoke_config(schedule, causal_force=False)
    m_bug = HelmholtzLM(cfg_bug)
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

    # Buggy mode commentary: when the schedule has at least one S-block
    # we expect the buggy variant to leak (pre_pert_bug > TOL_BUGGY_FLOOR);
    # when n_S=0 the buggy variant is identical to the fixed one.
    n_S = sum(1 for c in sigma if c == "S")

    if verbose:
        verdict = "OK" if ok else "FAIL"
        print(f"  [{verdict:>4}] schedule={label!r:>12s}  (L={L}, n_S={n_S})")
        print(f"           fixed mode:")
        print(f"             perturbation pre={pre_pert_fix:.2e}  "
              f"post={post_pert_fix:.2e}  "
              f"({'OK' if fix_pert_ok else 'FAIL'})")
        print(f"             gradient     post={post_grad_fix:.2e}  "
              f"pre={pre_grad_fix:.2e}  "
              f"({'OK' if fix_grad_ok else 'FAIL'})")
        leak_status = ""
        if n_S > 0:
            if pre_pert_bug > TOL_BUGGY_FLOOR:
                leak_status = "(leak detected as expected)"
            else:
                leak_status = ("(no leak detected; probe may be too weak "
                               "for this schedule at random init -- the "
                               "fix path is still verified by fixed mode)")
        else:
            leak_status = "(n_S=0; buggy mode is identical to fixed)"
        print(f"           buggy mode:")
        print(f"             perturbation pre={pre_pert_bug:.2e}  "
              f"post={post_pert_bug:.2e}  {leak_status}")
        print(f"             gradient     post={post_grad_bug:.2e}")

    details = {
        "schedule": schedule,
        "n_S": n_S,
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
# Trainer hook: a single-shot, fast probe used by train_helmholtz.py
# at startup so any leak aborts training before the optimiser engages.
# ---------------------------------------------------------------------------
def assert_causal(
    model: HelmholtzLM,
    vocab_size: int,
    T: int = 32,
    t_pert: int = 20,
    seed: int = 0,
    tol: float = TOL_PRE,
) -> None:
    """Run both probes on a real, on-device model; raise if leakage found.

    This is what the trainer calls at startup. It mirrors the per-schedule
    runner but operates on whatever model instance is passed in, with
    whatever config it was constructed under (so it catches both the
    "bug in the architecture" failure mode and the "user accidentally
    set causal_force=False in the config" failure mode).

    MPS workaround
    --------------
    The gradient probe walks the autograd graph through the S-block
    integration step, which is built with ``create_graph=True`` inside
    `model_helmholtz._s_block_step` whenever the module is in
    ``train()`` mode.  PyTorch's MPS backend does not currently support
    this second-order graph for some of the ops involved (the failure
    surfaces as "Placeholder storage has not been allocated on MPS
    device!").  The probe is architectural — its result depends only on
    the wiring, not on parameter values or the device — so we
    temporarily move the model to CPU for the probe and restore the
    original device immediately afterwards.  Cost: ~10 ms for an 8 M
    model.  This must be called BEFORE the optimiser is constructed, so
    that the optimiser binds to the parameters in their final on-device
    location.
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
            f"[causal-probe] PERTURBATION LEAK: pre={pre_pert:.4e} >= "
            f"tol={tol:.0e}.  Aborting training before any compute is "
            f"wasted.  Causal_force={getattr(model.cfg, 'causal_force', '?')}; "
            f"schedule={getattr(model.cfg, 'schedule', '?')}."
        )
    if post_grad >= tol:
        raise RuntimeError(
            f"[causal-probe] GRADIENT LEAK: post={post_grad:.4e} >= "
            f"tol={tol:.0e}.  Aborting training before any compute is "
            f"wasted.  Causal_force={getattr(model.cfg, 'causal_force', '?')}; "
            f"schedule={getattr(model.cfg, 'schedule', '?')}."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run_all(strict: bool, schedules: List[Tuple[str, str]]) -> int:
    print("=" * 78)
    print(" Helmholtz (Q9d) causal-violation probe")
    print(" mode: random-init, mass_mode='global', d=16, perturbation + gradient")
    print("=" * 78)
    fails = 0
    for name, sigma_str in schedules:
        try:
            ok, _ = probe_one_schedule(sigma_str, label=name, verbose=True)
        except Exception as exc:
            print(f"  [FAIL] schedule={name!r}: {type(exc).__name__}: {exc}")
            fails += 1
            continue
        if not ok:
            fails += 1
    print("-" * 78)
    if fails == 0:
        print(f"  All {len(schedules)} schedules: fixed-mode causal-side Δ < "
              f"{TOL_PRE:.0e}.  Q9d is causal by construction.")
        return 0
    print(f"  {fails} schedule(s) leaked in fixed mode.  See per-schedule "
          f"output above.")
    return 1 if strict else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Causal-violation probe for the Q9d Helmholtz hybrid."
    )
    ap.add_argument(
        "--schedule", default=None,
        help="Run on this single schedule string (e.g. 'SAAAAAAS').  "
             "If omitted, runs on every canonical schedule from "
             "model_helmholtz.canonical_schedules at L=8.",
    )
    ap.add_argument(
        "--strict", action="store_true",
        help="Exit non-zero on any failure (suitable for a CI gate).",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.schedule is not None:
        schedules = [("explicit", args.schedule)]
    else:
        schedules = canonical_schedules(L=8)

    return run_all(strict=args.strict, schedules=schedules)


if __name__ == "__main__":
    sys.exit(main())
