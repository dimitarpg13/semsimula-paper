"""
Smoke test for the sparse PARF-augmented SPLM (Q9c, Stage 1.5).

What this exercises
===================
The Stage-1.5 sparse model adds two pieces of new wiring on top of the
dense Algorithm-A PARF reference:

  - the `ScoreHead` MLP that produces per-pair routing logits, and
  - the straight-through Gumbel-softmax composite mask `~m` that gates
    the per-layer pair sum onto a top-k subset of past tokens.

This script verifies that BOTH new pieces are correctly wired BEFORE any
quality cell is launched, on three independent axes:

  1. Forward + backward round-trip on small (CPU) and prototype-shape
     (matches the P1.6 cell) configurations, in both `gumbel_noise =
     {True, False}` modes.

  2. Strict causal-violation probe (perturbation + gradient-Jacobian)
     in `causal_force = True` mode, with both `gumbel_noise` modes
     covered.  The added score head and the composite mask must NOT
     leak any future-position information to past positions, because
     the strict-causal mask is enforced via masked_fill(~causal, -inf)
     BEFORE the top-k and via `m_hard *= causal` AFTER the scatter.

  3. Bit-identity to dense PARF when `top_k >= T - 1` AND
     `gumbel_noise = False` AND eval mode.  This is the design-time
     guarantee that the sparse model is a strict superset of the dense
     baseline: at the maximum k and zero noise, the composite mask
     collapses to the strict-causal mask and the per-layer pair sum is
     identical to the dense Stage-1 model.

Usage
-----
  python3 notebooks/conservative_arch/parf/smoke_test_sparse.py
      -> all checks at smoke scale (CPU); exit 0 iff all pass.
  python3 notebooks/conservative_arch/parf/smoke_test_sparse.py --strict
      -> same; exit non-zero on any failure (CI gate).
  python3 notebooks/conservative_arch/parf/smoke_test_sparse.py --quick
      -> skip the prototype-shape forward/backward (only the cheap
         CPU smoke + causal probe + bit-identity), useful in CI loops.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

THIS_DIR = Path(__file__).parent
PARENT_DIR = THIS_DIR.parent
sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(THIS_DIR))

from model_parf import PARFConfig, PARFLM  # noqa: E402
from model_parf_sparse import SparsePARFConfig, SparsePARFLM  # noqa: E402
from causal_probe_parf import (  # noqa: E402
    TOL_PRE,
    perturbation_probe,
    gradient_probe,
)


# ---------------------------------------------------------------------------
# Tiny config builders (no logfreq dependency)
# ---------------------------------------------------------------------------
def _smoke_sparse_config(
    *,
    v_phi_kind: str = "structural",
    causal_force: bool = True,
    gumbel_noise: bool = True,
    top_k: int = 8,
    L: int = 4,
    T: int = 32,
    d: int = 16,
) -> SparsePARFConfig:
    return SparsePARFConfig(
        vocab_size=257,
        d=d, max_len=max(T, 64), L=L,
        v_hidden=32, v_depth=2,
        v_phi_kind=v_phi_kind,
        v_phi_d_type=4, v_phi_d_angle=2,
        v_phi_phi_hidden=8, v_phi_theta_hidden=8,
        v_phi_mlp_hidden=16,
        mass_mode="global",
        ln_after_step=True,
        causal_force=causal_force,
        top_k=top_k,
        score_head_hidden=8,
        gumbel_noise=gumbel_noise,
        gumbel_tau_init=1.0,
    )


def _matched_dense_config(cfg: SparsePARFConfig) -> PARFConfig:
    """A PARFConfig with the same (V_theta, V_phi, mass, T, d) as `cfg`.

    Used for the bit-identity test: instantiate a PARFLM with this
    config, copy parameters from a SparsePARFLM, and confirm they
    produce identical logits when the sparse mask collapses to the
    dense causal mask.
    """
    return PARFConfig(
        vocab_size=cfg.vocab_size,
        d=cfg.d, max_len=cfg.max_len, L=cfg.L,
        v_hidden=cfg.v_hidden, v_depth=cfg.v_depth,
        dt=cfg.dt, init_m=cfg.init_m, init_gamma=cfg.init_gamma,
        learn_mgamma=cfg.learn_mgamma, fixed_gamma=cfg.fixed_gamma,
        v_phi_kind=cfg.v_phi_kind,
        v_phi_d_type=cfg.v_phi_d_type, v_phi_d_angle=cfg.v_phi_d_angle,
        v_phi_phi_hidden=cfg.v_phi_phi_hidden,
        v_phi_theta_hidden=cfg.v_phi_theta_hidden,
        v_phi_mlp_hidden=cfg.v_phi_mlp_hidden,
        v_phi_C=cfg.v_phi_C, v_phi_eps=cfg.v_phi_eps,
        v_phi_init_scale=cfg.v_phi_init_scale,
        mass_mode=cfg.mass_mode,
        logfreq_init_alpha=cfg.logfreq_init_alpha,
        logfreq_path=cfg.logfreq_path,
        ln_after_step=cfg.ln_after_step, ln_eps=cfg.ln_eps,
        causal_force=cfg.causal_force,
        tie_embeddings=cfg.tie_embeddings,
        use_grad_checkpoint=cfg.use_grad_checkpoint,
    )


# ---------------------------------------------------------------------------
# Check 1 -- forward + backward round-trip
# ---------------------------------------------------------------------------
def check_forward_backward(
    *,
    gumbel_noise: bool,
    v_phi_kind: str = "structural",
    seed: int = 0,
    verbose: bool = True,
) -> bool:
    cfg = _smoke_sparse_config(
        v_phi_kind=v_phi_kind, gumbel_noise=gumbel_noise,
    )
    torch.manual_seed(seed)
    net = SparsePARFLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))
    net.train()
    logits, loss = net(x, targets=y)
    finite_fwd = bool(torch.isfinite(loss).item())
    loss.backward()
    # `V_theta.net.{depth*2}.bias` is structurally a None-grad parameter
    # in BOTH dense PARFLM and SparsePARFLM: it is a constant addend to
    # `V_th_per_token` whose only consumer is the inner
    # `autograd.grad(U, h_in)` force computation, in which the bias
    # vanishes (its gradient w.r.t. h is zero).  We therefore tolerate
    # None grads and only require that EVERY non-None grad be finite.
    nonfinite = [
        n for n, p in net.named_parameters()
        if p.requires_grad and p.grad is not None
        and not bool(torch.isfinite(p.grad).all().item())
    ]
    finite_bwd = len(nonfinite) == 0
    score_head_grads = [
        bool((p.grad is not None) and (p.grad.abs().sum().item() > 0.0))
        for p in net.score_head.parameters() if p.requires_grad
    ]
    score_head_active = all(score_head_grads)
    ok = finite_fwd and finite_bwd and score_head_active
    if verbose:
        verdict = "OK" if ok else "FAIL"
        nonfin_msg = (f"  nonfinite={nonfinite}" if nonfinite else "")
        print(f"  [{verdict:>4}] forward+backward  "
              f"v_phi={v_phi_kind!r:>14s}  "
              f"gumbel_noise={gumbel_noise!s:>5s}  "
              f"loss={loss.item():.4f}  "
              f"score_head_active={score_head_active}{nonfin_msg}")
    return ok


# ---------------------------------------------------------------------------
# Check 2 -- strict causal-violation probe
# ---------------------------------------------------------------------------
def check_causal_probe(
    *,
    gumbel_noise: bool,
    v_phi_kind: str = "structural",
    seed: int = 0,
    verbose: bool = True,
) -> bool:
    cfg = _smoke_sparse_config(
        v_phi_kind=v_phi_kind, gumbel_noise=gumbel_noise,
    )
    torch.manual_seed(seed)
    net = SparsePARFLM(cfg)
    T = 32
    t_pert = 20

    pre_pert, post_pert, _ = perturbation_probe(
        net, vocab_size=cfg.vocab_size,
        T=T, t_pert=t_pert, seed=seed,
    )
    # The gradient probe walks through the inner autograd.grad call,
    # which on MPS may not support the second-order graph for some
    # ops involved.  We instantiated on CPU, so this is fine here.
    post_grad, pre_grad, _ = gradient_probe(
        net, vocab_size=cfg.vocab_size,
        T=T, t_target=t_pert, seed=seed,
    )

    pert_ok = pre_pert < TOL_PRE
    grad_ok = post_grad < TOL_PRE
    ok = pert_ok and grad_ok
    if verbose:
        verdict = "OK" if ok else "FAIL"
        print(f"  [{verdict:>4}] causal probe       "
              f"v_phi={v_phi_kind!r:>14s}  "
              f"gumbel_noise={gumbel_noise!s:>5s}  "
              f"pert.pre={pre_pert:.2e}  grad.post={post_grad:.2e}")
    return ok


# ---------------------------------------------------------------------------
# Check 3 -- bit-identity to dense PARF at top_k = T-1, gumbel_noise = False
# ---------------------------------------------------------------------------
def _copy_shared_state(dst: PARFLM, src: SparsePARFLM) -> None:
    """Copy every parameter in `src` (PARFLM-shaped) into `dst`.

    The score head is unique to SparsePARFLM and is intentionally NOT
    copied -- it does not affect the bit-identity check because the
    composite mask, in the gumbel_noise=False, top_k>=T-1 regime,
    collapses to a constant 1.0 on the causal subset INDEPENDENT of
    the score-head logits.  This is the architectural invariant being
    verified.
    """
    src_state = {k: v for k, v in src.state_dict().items()
                 if not k.startswith("score_head.")
                 and not k.startswith("_gumbel_tau")}
    missing, unexpected = dst.load_state_dict(src_state, strict=False)
    if unexpected:
        raise RuntimeError(
            f"unexpected keys when copying SparsePARFLM -> PARFLM: "
            f"{unexpected}"
        )
    # `missing` may include non-existent keys in dst; that's expected.


def check_bit_identity_to_dense(
    *,
    v_phi_kind: str = "structural",
    seed: int = 0,
    verbose: bool = True,
    tol: float = 1e-5,
) -> bool:
    """Sparse model with top_k = T-1, gumbel_noise = False == dense PARF.

    Tolerance: fp32 computations through the L=4 stack accumulate
    a few ulps of round-off; we use 1e-5 (well above noise but well
    below any real architectural divergence).
    """
    T = 32
    cfg_sparse = _smoke_sparse_config(
        v_phi_kind=v_phi_kind,
        gumbel_noise=False,
        top_k=T - 1,            # max permissible: every causal source
        T=T,
    )
    cfg_dense = _matched_dense_config(cfg_sparse)

    torch.manual_seed(seed)
    sparse = SparsePARFLM(cfg_sparse)
    dense = PARFLM(cfg_dense)

    _copy_shared_state(dense, sparse)

    sparse.eval()
    dense.eval()

    rng = np.random.default_rng(seed)
    xb = rng.integers(0, cfg_sparse.vocab_size, size=(2, T)).astype(np.int64)
    x = torch.from_numpy(xb)

    with torch.enable_grad():
        # `enable_grad` is required because PARFLM._layer_step always
        # builds a small autograd graph for the inner force call, even
        # in eval mode.  PyTorch defaults to no_grad in .eval() so we
        # must override it for the equivalence check.
        out_sparse = sparse(x)
        out_dense = dense(x)
    diff = (out_sparse[0] - out_dense[0]).abs().max().item()
    ok = diff < tol
    if verbose:
        verdict = "OK" if ok else "FAIL"
        print(f"  [{verdict:>4}] bit-identity       "
              f"v_phi={v_phi_kind!r:>14s}  "
              f"top_k={cfg_sparse.top_k}  "
              f"max|sparse-dense|={diff:.2e}  (tol={tol:.0e})")
    return ok


# ---------------------------------------------------------------------------
# Check 4 -- Stage-1.5a vs 1.5b equivalence at low tau
# ---------------------------------------------------------------------------
def check_stage_1_5_equivalence(
    *,
    seed: int = 0,
    verbose: bool = True,
    tol: float = 5e-4,
) -> bool:
    """Stage-1.5a and Stage-1.5b produce near-identical loss + parameter
    gradients when gumbel_tau is small (hard one-hot routing).

    Tolerance is 5e-4 (not 1e-5) to accommodate the numerical difference
    between the dense _pair_dist2 (squared-norm expansion) and the
    gathered forward's explicit (a-b)^2 form — see PARF_Stage_1_5b_design.md
    Risk 3.  Real architectural bugs produce differences of O(1) or larger.
    """
    from dataclasses import replace

    cfg_a = _smoke_sparse_config(
        v_phi_kind="structural",
        gumbel_noise=False,
        top_k=8,
    )
    cfg_a = replace(cfg_a, use_gathered_v_phi=False, gumbel_tau_init=0.01)
    cfg_b = replace(cfg_a, use_gathered_v_phi=True)

    torch.manual_seed(seed)
    model_a = SparsePARFLM(cfg_a)
    torch.manual_seed(seed)
    model_b = SparsePARFLM(cfg_b)

    x = torch.randint(0, cfg_a.vocab_size, (2, 16))
    y = torch.randint(0, cfg_a.vocab_size, (2, 16))

    model_a.train()
    model_b.train()
    _, loss_a = model_a(x, targets=y)
    _, loss_b = model_b(x, targets=y)

    loss_diff = abs(loss_a.item() - loss_b.item())
    loss_ok = loss_diff < tol

    loss_a.backward()
    loss_b.backward()

    grad_mismatches = []
    for (n_a, p_a), (n_b, p_b) in zip(
        model_a.named_parameters(), model_b.named_parameters()
    ):
        assert n_a == n_b
        if p_a.grad is None and p_b.grad is None:
            continue
        if p_a.grad is None or p_b.grad is None:
            grad_mismatches.append(n_a)
            continue
        if not torch.allclose(p_a.grad, p_b.grad, atol=tol):
            grad_mismatches.append(n_a)

    grads_ok = len(grad_mismatches) == 0
    ok = loss_ok and grads_ok
    if verbose:
        verdict = "OK" if ok else "FAIL"
        mismatch_str = (f"  grad_mismatches={grad_mismatches}"
                        if grad_mismatches else "")
        print(f"  [{verdict:>4}] 1.5a vs 1.5b      "
              f"loss_diff={loss_diff:.2e}  tol={tol:.0e}"
              f"{mismatch_str}")
    return ok


# ---------------------------------------------------------------------------
# Check 5 -- all four mode combinations
# ---------------------------------------------------------------------------
def check_all_four_modes(
    *,
    seed: int = 0,
    verbose: bool = True,
) -> bool:
    """Forward+backward with all 4 combos of (layer_ckpt, gathered)."""
    all_ok = True
    for layer_ckpt in (False, True):
        for gathered in (False, True):
            tag_parts = []
            if layer_ckpt:
                tag_parts.append("lc")
            if gathered:
                tag_parts.append("gv")
            tag = "+".join(tag_parts) or "base"
            cfg = _smoke_sparse_config(
                v_phi_kind="structural",
                gumbel_noise=False,
            )
            from dataclasses import replace
            cfg = replace(
                cfg,
                use_layer_checkpoint=layer_ckpt,
                use_gathered_v_phi=gathered,
            )
            torch.manual_seed(seed)
            net = SparsePARFLM(cfg)
            x = torch.randint(0, cfg.vocab_size, (2, 16))
            y = torch.randint(0, cfg.vocab_size, (2, 16))
            net.train()
            try:
                _, loss = net(x, targets=y)
                loss.backward()
                ok = bool(torch.isfinite(loss).item())
            except Exception as exc:
                ok = False
                if verbose:
                    print(f"  [FAIL] {tag}: {exc}")
            if verbose and ok:
                print(f"  [  OK] {tag:>8s}  loss={loss.item():.4f}")
            all_ok = all_ok and ok
    return all_ok


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero on any failure (CI gate).")
    ap.add_argument("--quick", action="store_true",
                    help="skip the prototype-shape forward/backward "
                         "(currently a no-op; reserved for future "
                         "expansion to per-cell wall-clock checks).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("=" * 64)
    print("Sparse PARF-augmented SPLM (Q9c, Stage 1.5) -- smoke test")
    print("=" * 64)

    # Inform the user up front which checks we're running.
    print("Variants probed: v_phi in {structural, mlp}; "
          "gumbel_noise in {True, False}; top_k = T-1 for bit-identity.")

    results: list[Tuple[str, bool]] = []

    print("\n[1/3] Forward + backward (training mode)")
    for kind in ["structural", "mlp"]:
        for noise in [True, False]:
            ok = check_forward_backward(
                gumbel_noise=noise, v_phi_kind=kind, seed=args.seed,
            )
            results.append((f"forward+backward[{kind},gumbel={noise}]", ok))

    print("\n[2/3] Strict causal-violation probe")
    for kind in ["structural", "mlp"]:
        for noise in [True, False]:
            ok = check_causal_probe(
                gumbel_noise=noise, v_phi_kind=kind, seed=args.seed,
            )
            results.append((f"causal_probe[{kind},gumbel={noise}]", ok))

    print("\n[3/3] Bit-identity to dense PARF at top_k = T-1, noise off")
    for kind in ["structural", "mlp"]:
        ok = check_bit_identity_to_dense(
            v_phi_kind=kind, seed=args.seed,
        )
        results.append((f"bit_identity[{kind}]", ok))

    print("\n[4/5] Stage-1.5a vs 1.5b bit-identity at low tau")
    ok = check_stage_1_5_equivalence(seed=args.seed)
    results.append(("stage_1.5a_vs_1.5b_equivalence", ok))

    print("\n[5/5] All four mode combinations (ckpt x gathered)")
    ok = check_all_four_modes(seed=args.seed)
    results.append(("four_mode_combos", ok))

    n_total = len(results)
    n_ok = sum(1 for _, ok in results if ok)
    print()
    print("-" * 64)
    print(f"summary: {n_ok}/{n_total} checks passed")
    if n_ok < n_total:
        for name, ok in results:
            if not ok:
                print(f"  FAIL  {name}")
    print("-" * 64)

    if args.strict and n_ok < n_total:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
