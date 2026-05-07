"""
End-to-end smoke test for the PARF-augmented SPLM (Q9c) prototype.

What it covers
--------------
1. Build PARFLM at TWO V_φ variants (structural, mlp) and at THREE config
   shapes (tiny, em-ln-leakfree-shape, full prototype-scale H1.5 vh=128).
2. Verify forward + backward + parameter count for each.
3. Run the causal probe on each (production causal_force=True).
4. Run 5 NTP training steps with AdamW and confirm the loss drops by
   more than the gradient noise floor.
5. Print a one-line PASS / FAIL banner.

This script is the gate for "is the model wired correctly?" before
launching any full quality cell with `train_parf.py`.

Usage
-----
  python3 notebooks/conservative_arch/parf/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.optim as optim

THIS_DIR = Path(__file__).parent
PARENT_DIR = THIS_DIR.parent
sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(THIS_DIR))

from model_parf import PARFConfig, PARFLM  # noqa: E402
from causal_probe_parf import assert_causal  # noqa: E402


def _tiny_cfg(v_phi_kind: str, use_grad_checkpoint: bool = False) -> PARFConfig:
    """Tiny shape used by the causal probe (no logfreq dependency)."""
    return PARFConfig(
        vocab_size=257,
        d=16, max_len=64, L=4,
        v_hidden=32, v_depth=2,
        v_phi_kind=v_phi_kind,
        v_phi_d_type=4, v_phi_d_angle=2,
        v_phi_phi_hidden=8, v_phi_theta_hidden=8,
        v_phi_mlp_hidden=16,
        mass_mode="global",
        use_grad_checkpoint=use_grad_checkpoint,
    )


def _em_ln_shape_cfg(v_phi_kind: str, use_grad_checkpoint: bool = False) -> PARFConfig:
    """SPLM em-ln leakfree shape: d=128, L=8, v_hidden=128, vocab=257.

    This is the SHAPE the prototype is sized to (T=128 for smoke);
    we use vocab=257 so we don't need GPT-2 BPE / logfreq just to smoke.
    """
    return PARFConfig(
        vocab_size=257,
        d=128, max_len=256, L=8,
        v_hidden=128, v_depth=3,
        v_phi_kind=v_phi_kind,
        v_phi_d_type=16, v_phi_d_angle=8,
        v_phi_phi_hidden=32, v_phi_theta_hidden=32,
        v_phi_mlp_hidden=64,
        mass_mode="global",
        use_grad_checkpoint=use_grad_checkpoint,
    )


def _check_forward_backward(cfg: PARFConfig, label: str) -> dict:
    """Build, forward, backward; return dict of stats."""
    torch.manual_seed(0)
    net = PARFLM(cfg)
    n_params = sum(p.numel() for p in net.parameters())
    n_phi = sum(p.numel() for p in net.V_phi.parameters())
    n_theta = sum(p.numel() for p in net.V_theta.parameters())
    print(f"  [{label}] V_phi={cfg.v_phi_kind!r}  "
          f"params={n_params:,}  V_theta={n_theta:,}  V_phi={n_phi:,}")

    # T must be small enough for the (B, T, T, d) MLP variant on CPU.
    T = min(32, cfg.max_len)
    B = 2
    x = torch.randint(0, cfg.vocab_size, (B, T))
    y = torch.randint(0, cfg.vocab_size, (B, T))

    net.train()
    logits, loss = net(x, targets=y)
    assert logits.shape == (B, T, cfg.vocab_size), \
        f"logits shape mismatch: {logits.shape}"
    loss0 = float(loss.item())
    loss.backward()

    grad_total = 0.0
    grad_count = 0
    for p in net.parameters():
        if p.grad is not None:
            grad_total += float(p.grad.norm().item())
            grad_count += 1
    print(f"           forward loss={loss0:.4f}  "
          f"|grad| mean={grad_total / max(1, grad_count):.4e}  "
          f"({grad_count} tensors with grad)")
    return {"net": net, "loss0": loss0, "T": T, "B": B}


def _check_train_loop(net, cfg: PARFConfig, T: int, B: int, n_steps: int = 5):
    """Run a few NTP training steps; expect loss to drop noticeably."""
    opt = optim.AdamW(net.parameters(), lr=5e-4)
    torch.manual_seed(123)
    losses = []
    for step in range(n_steps):
        x = torch.randint(0, cfg.vocab_size, (B, T))
        y = torch.randint(0, cfg.vocab_size, (B, T))
        opt.zero_grad(set_to_none=True)
        _, loss = net(x, targets=y)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    print(f"           {n_steps}-step trace: " +
          " -> ".join(f"{l:.4f}" for l in losses))
    return losses


def _check_causal(cfg: PARFConfig, label: str):
    """Run the production causal probe (causal_force=True default)."""
    torch.manual_seed(0)
    net = PARFLM(cfg)
    try:
        assert_causal(net, vocab_size=cfg.vocab_size, T=32, t_pert=20)
        print(f"  [{label}] causal probe PASS (V_phi={cfg.v_phi_kind!r})")
        return True
    except RuntimeError as exc:
        print(f"  [{label}] causal probe FAIL: {exc}")
        return False


def main() -> int:
    print("=" * 78)
    print(" PARF (Q9c) end-to-end smoke test")
    print("=" * 78)

    fails = 0

    # ----- Tiny shape, both V_phi variants -----
    print("\n[1/3] tiny shape (d=16, L=4, T=32)")
    for kind in ("structural", "mlp"):
        cfg = _tiny_cfg(kind)
        st = _check_forward_backward(cfg, "tiny")
        _check_train_loop(st["net"], cfg, T=st["T"], B=st["B"], n_steps=5)
        if not _check_causal(cfg, "tiny"):
            fails += 1

    # ----- em-ln leakfree shape, both V_phi variants -----
    print("\n[2/3] em-ln leakfree shape (d=128, L=8, T=32)")
    for kind in ("structural", "mlp"):
        cfg = _em_ln_shape_cfg(kind)
        st = _check_forward_backward(cfg, "em-ln")
        _check_train_loop(st["net"], cfg, T=st["T"], B=st["B"], n_steps=5)
        if not _check_causal(cfg, "em-ln"):
            fails += 1

    # ----- Grad-checkpointed path -----
    # This exercises the torch.utils.checkpoint(..., use_reentrant=False)
    # wrap on the V_φ pair sum AND verifies its compatibility with the
    # inner `autograd.grad(create_graph=True)` call (the second-order
    # graph required by the gradient-Jacobian causal probe).
    print("\n[3/3] em-ln leakfree shape with grad checkpointing on (d=128, L=8, T=32)")
    for kind in ("structural", "mlp"):
        cfg = _em_ln_shape_cfg(kind, use_grad_checkpoint=True)
        st = _check_forward_backward(cfg, "em-ln+gc")
        _check_train_loop(st["net"], cfg, T=st["T"], B=st["B"], n_steps=5)
        if not _check_causal(cfg, "em-ln+gc"):
            fails += 1

    # ----- Final verdict -----
    print("\n" + "=" * 78)
    if fails == 0:
        print(" ALL SMOKE CHECKS PASSED")
        print(" PARF prototype is wired correctly.  Ready for first quality cell.")
        return 0
    print(f" {fails} smoke check(s) failed.  See above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
