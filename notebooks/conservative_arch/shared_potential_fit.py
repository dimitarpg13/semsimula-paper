"""
Shared scalar-potential fit -- the strict conservative-dynamics test.

Ansatz (fit jointly across ALL layers, ALL sentences, ALL tokens):

    Delta x_l  ~=  alpha_l * v_l  -  beta_l * grad_h V_psi(x_l)

with
  -- V_psi a SINGLE neural scalar (shared across layers)
  -- alpha_l, beta_l per-layer scalars
  -- x_l := x_ps(l)              per-sentence centered hidden state
  -- v_l := x_l - x_{l-1}        velocity proxy (l >= 1)
  -- Delta x_l := x_{l+1} - x_l  one-step displacement target

Interpretation.
  The velocity-aware Jacobian test (`jacobian_symmetry.py`) showed
  both SPLM and GPT-2 admit nearly-symmetric per-step linear
  operators M_l at the PCA-16 level.  But per-layer M_l is a
  much weaker structural constraint than "all layers' forces
  derive from one shared scalar V".  This script tests the
  latter, stricter property.

  Predictions:
    * SPLM trajectories (by construction) fit well with a shared
      V_psi -- the true dynamics IS grad of one scalar V_theta.
    * GPT-2 trajectories may or may not fit a shared V_psi.  That's
      the headline open question: is there a single smooth scalar
      that explains all 12 layers' behaviour, or do attention+MLP+LN
      compositions defy a shared-potential description?

Baselines reported alongside:
    A. static null          (Delta x = 0)
    B. velocity-only        (Delta x = alpha_l v_l)
    C. velocity + shared V  (the main fit)

Usage:
  python3 shared_potential_fit.py --traj results/splm_shakespeare_ckpt_latest.trajectories.pkl
  python3 shared_potential_fit.py --traj results/gpt2_baseline.trajectories.pkl
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_DIR  = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"


# ---------------------------------------------------------------------------
# Scalar potential network.
# ---------------------------------------------------------------------------
class SharedPotential(nn.Module):
    def __init__(self, d: int, hidden: int = 256, depth: int = 2):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(d, hidden), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)

    def grad_V(self, h: torch.Tensor) -> torch.Tensor:
        """Return dV/dh of shape (N, d)."""
        h = h.requires_grad_(True) if not h.requires_grad else h
        V = self.forward(h).sum()
        g, = torch.autograd.grad(V, h, create_graph=self.training)
        return g


# ---------------------------------------------------------------------------
# Sample assembly.
# ---------------------------------------------------------------------------
def build_samples(trajs, L: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pool (x, v, dx, layer) across all layers l in [1, L-1] and all tokens.

    Returns (X, V, Y, layer_idx).
    """
    Xs, Vs, Ys, Ls = [], [], [], []
    for tr in trajs:
        for ell in range(1, L):            # need l-1 for v, l+1 for dx
            x_prev = tr.x_ps[ell - 1]      # (T, d)
            x      = tr.x_ps[ell]
            x_next = tr.x_ps[ell + 1]
            T = x.shape[0]
            Xs.append(x)
            Vs.append(x - x_prev)
            Ys.append(x_next - x)
            Ls.append(np.full((T,), ell, dtype=np.int64))
    X = np.concatenate(Xs, axis=0)
    V = np.concatenate(Vs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    LAY = np.concatenate(Ls, axis=0)
    return X.astype(np.float32), V.astype(np.float32), Y.astype(np.float32), LAY


# ---------------------------------------------------------------------------
# R^2 metrics -- per layer and overall, for TRAIN and TEST.
# ---------------------------------------------------------------------------
def r2_overall(Y: np.ndarray, Y_pred: np.ndarray) -> float:
    ss_tot = float(np.sum((Y - Y.mean(0, keepdims=True)) ** 2))
    ss_res = float(np.sum((Y - Y_pred) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)


def r2_per_layer(Y: np.ndarray, Y_pred: np.ndarray,
                 layers: np.ndarray) -> Dict[int, float]:
    out = {}
    for ell in np.unique(layers):
        m = layers == ell
        Ym  = Y[m]
        Ypm = Y_pred[m]
        ss_tot = float(np.sum((Ym - Ym.mean(0, keepdims=True)) ** 2))
        ss_res = float(np.sum((Ym - Ypm) ** 2))
        out[int(ell)] = 1.0 - ss_res / (ss_tot + 1e-12)
    return out


# ---------------------------------------------------------------------------
# Fit models.
# ---------------------------------------------------------------------------
def fit_velocity_only(X, V, Y, LAY) -> Tuple[np.ndarray, np.ndarray, Dict[int, float]]:
    """Per-layer alpha_l: argmin_alpha_l ||Y_l - alpha_l * V_l||^2."""
    alpha = {}
    Y_pred = np.zeros_like(Y)
    for ell in np.unique(LAY):
        m = LAY == ell
        num = float(np.sum(V[m] * Y[m]))
        den = float(np.sum(V[m] * V[m]))
        a   = num / (den + 1e-12)
        alpha[int(ell)] = a
        Y_pred[m] = a * V[m]
    return Y_pred, alpha, r2_per_layer(Y, Y_pred, LAY)


def fit_shared_V(
    X, V, Y, LAY,
    d: int, L: int,
    hidden: int = 256, depth: int = 2,
    steps: int = 4000, batch_size: int = 2048,
    lr: float = 3e-3, weight_decay: float = 1e-4,
    device: str = "cpu", seed: int = 0,
    verbose: bool = True,
) -> Tuple[SharedPotential, np.ndarray, np.ndarray, np.ndarray, List[float]]:
    """Joint optimisation of V_psi, alpha_l, beta_l on TRAIN pool."""
    torch.manual_seed(seed)
    X_t = torch.from_numpy(X).to(device)
    V_t = torch.from_numpy(V).to(device)
    Y_t = torch.from_numpy(Y).to(device)
    L_t = torch.from_numpy(LAY).to(device).long()

    model = SharedPotential(d, hidden=hidden, depth=depth).to(device)
    # Per-layer scalars.
    alpha = nn.Parameter(torch.zeros(L, device=device))     # start at 0 -> velocity-only disabled
    beta  = nn.Parameter(torch.zeros(L, device=device))

    optim = torch.optim.AdamW(
        list(model.parameters()) + [alpha, beta],
        lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95),
    )

    N = X_t.shape[0]
    rng = np.random.default_rng(seed)
    hist = []
    model.train()

    for step in range(steps):
        idx = rng.integers(0, N, size=batch_size)
        xb  = X_t[idx]
        vb  = V_t[idx]
        yb  = Y_t[idx]
        lb  = L_t[idx]

        xb_r = xb.clone().requires_grad_(True)
        Vout = model(xb_r).sum()
        gV,  = torch.autograd.grad(Vout, xb_r, create_graph=True)

        y_pred = alpha[lb].unsqueeze(-1) * vb - beta[lb].unsqueeze(-1) * gV

        loss = F.mse_loss(y_pred, yb)

        # Mild L2 on V-grad magnitude to prevent blow-up.
        reg  = 1e-5 * torch.mean(gV ** 2)
        (loss + reg).backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + [alpha, beta], 5.0)
        optim.step()
        optim.zero_grad(set_to_none=True)

        if verbose and (step + 1) % max(steps // 20, 1) == 0:
            with torch.no_grad():
                denom = float(torch.sum((yb - yb.mean(0, keepdim=True)) ** 2))
                resid = float(torch.sum((yb - y_pred.detach()) ** 2))
                r2_batch = 1.0 - resid / (denom + 1e-12)
            print(f"[shared-V] step {step+1:5d}/{steps}   "
                  f"loss {loss.item():.5f}   r2(batch) {r2_batch:+.3f}")
            hist.append({"step": step + 1, "loss": float(loss.item()), "r2_batch": r2_batch})

    # ---- Full predictions.
    model.eval()
    with torch.enable_grad():
        preds = []
        for s in range(0, N, 4096):
            xb_r = X_t[s:s + 4096].clone().requires_grad_(True)
            Vo = model(xb_r).sum()
            gV, = torch.autograd.grad(Vo, xb_r)
            yb_pred = alpha[L_t[s:s + 4096]].unsqueeze(-1) * V_t[s:s + 4096] \
                    - beta[L_t[s:s + 4096]].unsqueeze(-1) * gV
            preds.append(yb_pred.detach().cpu().numpy())
    Y_pred = np.concatenate(preds, axis=0)
    alpha_np = alpha.detach().cpu().numpy()
    beta_np  = beta.detach().cpu().numpy()
    return model, Y_pred, alpha_np, beta_np, hist


def predict_shared_V(model: SharedPotential, X, V_, LAY,
                     alpha_np, beta_np, device: str) -> np.ndarray:
    """Evaluate alpha*v - beta*grad V on arbitrary (X, V, LAY)."""
    X_t = torch.from_numpy(X).to(device)
    V_t = torch.from_numpy(V_).to(device)
    L_t = torch.from_numpy(LAY).to(device).long()
    alpha_t = torch.from_numpy(alpha_np).to(device)
    beta_t  = torch.from_numpy(beta_np).to(device)
    model.eval()
    with torch.enable_grad():
        preds = []
        for s in range(0, X_t.shape[0], 4096):
            xb_r = X_t[s:s + 4096].clone().requires_grad_(True)
            Vo = model(xb_r).sum()
            gV, = torch.autograd.grad(Vo, xb_r)
            yb_pred = alpha_t[L_t[s:s + 4096]].unsqueeze(-1) * V_t[s:s + 4096] \
                    - beta_t[L_t[s:s + 4096]].unsqueeze(-1) * gV
            preds.append(yb_pred.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth",  type=int, default=2)
    ap.add_argument("--steps",  type=int, default=4000)
    ap.add_argument("--batch",  type=int, default=2048)
    ap.add_argument("--lr",     type=float, default=3e-3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag",    default=None)
    ap.add_argument("--seed",   type=int, default=0)
    args = ap.parse_args()

    device = args.device or (
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    traj_path = Path(args.traj)
    with traj_path.open("rb") as f:
        bundle = pickle.load(f)
    trajs = bundle["trajectories"]
    L = int(bundle["L"])
    d = int(bundle["d"])
    default_tag = traj_path.stem.replace(".trajectories", "")
    if default_tag.startswith("splm_"):
        default_tag = default_tag[len("splm_"):]
    tag = args.tag or default_tag

    print(f"[shared-V] tag={tag}  d={d}  L={L}  device={device}")
    print(f"[shared-V] V_psi: {args.depth}-layer MLP, hidden={args.hidden}  "
          f"(params ~{d*args.hidden + (args.depth-1)*args.hidden*args.hidden + args.hidden:,})")

    train = [tr for tr in trajs if tr.split == "train"]
    test  = [tr for tr in trajs if tr.split == "test"]

    X_tr, V_tr, Y_tr, L_tr = build_samples(train, L)
    X_te, V_te, Y_te, L_te = build_samples(test,  L)
    print(f"[shared-V] samples: train={X_tr.shape[0]:,}  test={X_te.shape[0]:,}")

    # ---------- Baselines ----------
    # A. static null: Y_pred = 0
    r2_null_tr = r2_per_layer(Y_tr, np.zeros_like(Y_tr), L_tr)
    r2_null_te = r2_per_layer(Y_te, np.zeros_like(Y_te), L_te)

    # B. velocity-only (alpha per layer, fit on TRAIN, evaluated on TEST).
    _, alpha_vo, r2_vo_tr = fit_velocity_only(X_tr, V_tr, Y_tr, L_tr)
    Y_pred_te_vo = np.zeros_like(Y_te)
    for ell, a in alpha_vo.items():
        m = L_te == ell
        Y_pred_te_vo[m] = a * V_te[m]
    r2_vo_te = r2_per_layer(Y_te, Y_pred_te_vo, L_te)

    # C. velocity + shared V_psi.
    model, Y_pred_tr, alpha_np, beta_np, hist = fit_shared_V(
        X_tr, V_tr, Y_tr, L_tr, d, L,
        hidden=args.hidden, depth=args.depth,
        steps=args.steps, batch_size=args.batch, lr=args.lr,
        device=device, seed=args.seed,
    )
    Y_pred_te = predict_shared_V(model, X_te, V_te, L_te, alpha_np, beta_np, device)
    r2_shv_tr = r2_per_layer(Y_tr, Y_pred_tr, L_tr)
    r2_shv_te = r2_per_layer(Y_te, Y_pred_te, L_te)
    r2_shv_tr_overall = r2_overall(Y_tr, Y_pred_tr)
    r2_shv_te_overall = r2_overall(Y_te, Y_pred_te)

    # ---------- Summary print ----------
    print()
    print(f"[shared-V] overall R^2:  TRAIN {r2_shv_tr_overall:+.3f}   TEST {r2_shv_te_overall:+.3f}")
    print()
    print("[shared-V] per-layer TEST R^2  (A null / B vel-only / C vel+shared-V):")
    for ell in sorted(r2_null_te):
        a = r2_null_te[ell]; b = r2_vo_te[ell]; c = r2_shv_te[ell]
        print(f"[shared-V]   l={ell:2d}   null {a:+.3f}   vel-only {b:+.3f}   "
              f"vel+V {c:+.3f}   gain-over-velonly {c - b:+.3f}")

    # ---------- Save ----------
    npz = RESULTS_DIR / f"sharedV_{tag}_results.npz"
    np.savez(
        npz,
        layers=np.array(sorted(r2_null_te.keys())),
        r2_null_train=np.array([r2_null_tr[e] for e in sorted(r2_null_tr)]),
        r2_null_test =np.array([r2_null_te[e] for e in sorted(r2_null_te)]),
        r2_vo_train  =np.array([r2_vo_tr[e]   for e in sorted(r2_vo_tr)]),
        r2_vo_test   =np.array([r2_vo_te[e]   for e in sorted(r2_vo_te)]),
        r2_shv_train =np.array([r2_shv_tr[e]  for e in sorted(r2_shv_tr)]),
        r2_shv_test  =np.array([r2_shv_te[e]  for e in sorted(r2_shv_te)]),
        alpha=alpha_np, beta=beta_np,
        r2_shv_train_overall=np.array([r2_shv_tr_overall]),
        r2_shv_test_overall =np.array([r2_shv_te_overall]),
        hidden=np.array([args.hidden]),
        depth =np.array([args.depth]),
    )
    print(f"[shared-V] saved -> {npz}")

    # ---------- Figure ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, split in zip(axes, ["train", "test"]):
        layers = np.array(sorted(r2_null_te.keys()))
        y_null = np.array([(r2_null_tr if split == "train" else r2_null_te)[e] for e in layers])
        y_vo   = np.array([(r2_vo_tr   if split == "train" else r2_vo_te)[e]   for e in layers])
        y_shv  = np.array([(r2_shv_tr  if split == "train" else r2_shv_te)[e]  for e in layers])
        ax.plot(layers, y_null, marker="x", label="A. static null", color="tab:gray")
        ax.plot(layers, y_vo,   marker="s", label="B. velocity-only", color="tab:orange")
        ax.plot(layers, y_shv,  marker="o", label="C. velocity + shared $V_\\psi$",
                color="tab:blue")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_xlabel("layer $\\ell$"); ax.set_ylabel("$R^2$")
        ax.set_ylim(-0.3, 1.05)
        ax.grid(True, alpha=0.3); ax.legend()
        ax.set_title(split.upper())
    fig.suptitle(f"Shared-potential fit per layer -- {tag}")
    fig.tight_layout()
    fig_path = RESULTS_DIR / f"sharedV_{tag}_fig.png"
    fig.savefig(fig_path, dpi=130); plt.close(fig)
    print(f"[shared-V] saved -> {fig_path}")

    # ---------- Markdown ----------
    md = RESULTS_DIR / f"sharedV_{tag}_summary.md"
    layers = sorted(r2_null_te.keys())
    with md.open("w") as f:
        f.write(f"# Shared-potential fit -- {tag}\n\n")
        f.write("Strict conservative-dynamics test.  Joint fit of a *single* "
                "scalar network $V_\\psi(h)$ plus per-layer scalars "
                "$\\alpha_\\ell, \\beta_\\ell$, minimising the squared residual\n\n"
                "$$\\Delta x_\\ell - \\alpha_\\ell v_\\ell + \\beta_\\ell \\nabla V_\\psi(x_\\ell)$$\n\n"
                "across **every layer $\\ell \\geq 1$, every token, every training "
                "sentence**.  If a single smooth $V$ can describe all layers' "
                "forces, this beats the velocity-only baseline on held-out sentences.\n\n")
        f.write(f"- Hidden dim `d = {d}`; layers `L = {L}`; samples per split: "
                f"TRAIN {X_tr.shape[0]:,} / TEST {X_te.shape[0]:,}\n")
        f.write(f"- $V_\\psi$: {args.depth}-layer MLP, hidden = {args.hidden}, GELU\n")
        f.write(f"- Optimiser: AdamW, {args.steps} steps, bs={args.batch}, lr={args.lr}\n\n")
        f.write(f"## Overall fit\n\n")
        f.write(f"- TRAIN pooled $R^2$ = **{r2_shv_tr_overall:+.3f}**\n")
        f.write(f"- TEST pooled $R^2$  = **{r2_shv_te_overall:+.3f}**\n\n")
        f.write(f"## Per-layer TEST $R^2$\n\n")
        f.write("| layer | A. static null | B. velocity-only | C. velocity + shared $V_\\psi$ | C - B |\n")
        f.write("|--:|--:|--:|--:|--:|\n")
        for ell in layers:
            a = r2_null_te[ell]; b = r2_vo_te[ell]; c = r2_shv_te[ell]
            f.write(f"| {ell} | {a:+.3f} | {b:+.3f} | {c:+.3f} | {c - b:+.3f} |\n")
        f.write(f"\n## Per-layer TRAIN $R^2$\n\n")
        f.write("| layer | A. null | B. velocity-only | C. vel + shared $V_\\psi$ |\n")
        f.write("|--:|--:|--:|--:|\n")
        for ell in layers:
            f.write(f"| {ell} | {r2_null_tr[ell]:+.3f} | {r2_vo_tr[ell]:+.3f} | "
                    f"{r2_shv_tr[ell]:+.3f} |\n")
        f.write(f"\n## Learned per-layer scalars\n\n")
        f.write("| layer | $\\alpha_\\ell$ | $\\beta_\\ell$ |\n|--:|--:|--:|\n")
        for ell in layers:
            f.write(f"| {ell} | {alpha_np[ell]:+.4f} | {beta_np[ell]:+.4f} |\n")
        f.write(f"\n## Artefacts\n\n")
        f.write(f"- `sharedV_{tag}_results.npz`\n")
        f.write(f"- `sharedV_{tag}_fig.png`\n")
    print(f"[shared-V] saved -> {md}")


if __name__ == "__main__":
    main()
