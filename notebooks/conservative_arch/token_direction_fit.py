"""
Token-direction conservative-dynamics diagnostics.

Background
----------
Our step-1/2/3 diagnostics trace the trajectory of a hidden state across
**layer depth** at a fixed token t.  That is one natural "time axis" for
a transformer, but not the only one: autoregressive inference also
advances a second axis -- **token position**.

This script runs the same two diagnostics -- shared-scalar-potential fit
and velocity-aware Jacobian-symmetry test -- but with the **token axis
playing the role of time at a fixed layer** ell.  Concretely, for every
layer ell we collect per-token triples

    h_{t-1}^{ell}, h_t^{ell}, h_{t+1}^{ell}                t in [1, T-2]

from every sentence in the trajectory file and ask whether

    Delta h_t^{ell}  ~=  alpha_ell * v_t^{ell}  -  beta_ell * grad_h V_psi(h_t^{ell})

with v_t = h_t - h_{t-1} and Delta h_t = h_{t+1} - h_t.  V_psi is shared
across ALL layers AND ALL tokens, while alpha_ell, beta_ell are per-layer
scalars (absorbing scale differences between layers).

The "velocity-aware Jacobian" test is the token-direction analogue of
`jacobian_symmetry.py`: on a PCA-k subspace at fixed layer ell, fit

    y_t = A v_t + M_ell h_t + b_ell       (unconstrained)
    y_t = A v_t + M_ell h_t + b_ell,  M_ell = M_ell^T    (sym-restricted)

and report both R^2 values.  A small full-vs-sym gap means the
per-token spring matrix is symmetric -- i.e. the token-direction
dynamics at layer ell is locally Hessian-of-scalar, consistent with
the Geodesic Hypothesis in its own coordinate system.

Interpretation of outcomes (three-way comparison)
-------------------------------------------------
Given the step-2/3 depth-direction results (SPLM median R^2=0.90
uniform, Matched-GPT 0.56 monotonic decay, Pretrained-GPT-2 0.45 with
mid-layer collapse to 0.09), the token-direction test asks: does the
same architectural separator reproduce in token-time?  If GPT-2 mid
layers *pass* in token-direction but *fail* in depth-direction, the
depth failure is a "composition of layer operators" phenomenon and
STP's Geodesic Hypothesis survives in its natural coordinate.  If
GPT-2 mid layers fail in both, the shared-scalar hypothesis does not
hold for attention transformers in any natural coordinate system.

Usage
-----
  python3 token_direction_fit.py --traj results/splm_shakespeare_ckpt_latest.trajectories.pkl
  python3 token_direction_fit.py --traj results/matched_baseline.trajectories.pkl
  python3 token_direction_fit.py --traj results/gpt2_baseline.trajectories.pkl
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

from shared_potential_fit import SharedPotential
from jacobian_symmetry import (
    fit_linear_and_symmetric,
    fit_second_order,
    _pca_basis,
)


SCRIPT_DIR  = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"


# ---------------------------------------------------------------------------
# Sample assembly along the TOKEN axis at fixed layer.
# ---------------------------------------------------------------------------
def build_token_triples(
    trajs, L: int, t_skip: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pool (h, v, dh, layer) triples across all sentences, all tokens, all layers.

    For every layer ell in [0, L] (inclusive -- we include the pre-block
    embedding ell=0 and the final ell=L), and every token index t in
    [1 + t_skip, T - 2 - t_skip]:
        h_t^{ell}, v_t = h_t^{ell} - h_{t-1}^{ell}, dh_t = h_{t+1}^{ell} - h_t^{ell}

    t_skip lets us drop the first few / last few tokens of each sentence
    to avoid BOS/initialisation transients.
    """
    Xs, Vs, Ys, Ls = [], [], [], []
    for tr in trajs:
        T = tr.x_ps.shape[1]
        if T < 3 + 2 * t_skip:
            continue
        tlo = 1 + t_skip
        thi = T - 1 - t_skip
        for ell in range(L + 1):
            hs = tr.x_ps[ell]                        # (T, d)
            h_prev = hs[tlo - 1:thi - 1]             # (thi - tlo, d)
            h      = hs[tlo:thi]
            h_next = hs[tlo + 1:thi + 1]
            n = h.shape[0]
            Xs.append(h)
            Vs.append(h - h_prev)
            Ys.append(h_next - h)
            Ls.append(np.full((n,), ell, dtype=np.int64))
    X = np.concatenate(Xs, axis=0).astype(np.float32)
    V = np.concatenate(Vs, axis=0).astype(np.float32)
    Y = np.concatenate(Ys, axis=0).astype(np.float32)
    LAY = np.concatenate(Ls, axis=0)
    return X, V, Y, LAY


def per_layer_samples_tokens(trajs, layer: int, t_skip: int = 0):
    """Per-token (h, v, dh) triples at a fixed layer, pooled across sentences."""
    Xs, Vs, Ys = [], [], []
    for tr in trajs:
        T = tr.x_ps.shape[1]
        if T < 3 + 2 * t_skip:
            continue
        tlo = 1 + t_skip
        thi = T - 1 - t_skip
        hs = tr.x_ps[layer]
        Xs.append(hs[tlo:thi])
        Vs.append(hs[tlo:thi] - hs[tlo - 1:thi - 1])
        Ys.append(hs[tlo + 1:thi + 1] - hs[tlo:thi])
    return (np.concatenate(Xs, axis=0),
            np.concatenate(Vs, axis=0),
            np.concatenate(Ys, axis=0))


# ---------------------------------------------------------------------------
# R^2.
# ---------------------------------------------------------------------------
def r2_overall(Y: np.ndarray, Y_pred: np.ndarray) -> float:
    ss_tot = float(np.sum((Y - Y.mean(0, keepdims=True)) ** 2))
    ss_res = float(np.sum((Y - Y_pred) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)


def r2_per_layer(Y: np.ndarray, Y_pred: np.ndarray,
                 layers: np.ndarray) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for ell in np.unique(layers):
        m = layers == ell
        Ym  = Y[m]
        Ypm = Y_pred[m]
        ss_tot = float(np.sum((Ym - Ym.mean(0, keepdims=True)) ** 2))
        ss_res = float(np.sum((Ym - Ypm) ** 2))
        out[int(ell)] = 1.0 - ss_res / (ss_tot + 1e-12)
    return out


def fit_velocity_only(X, V, Y, LAY):
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


# ---------------------------------------------------------------------------
# Shared-V fit along token axis (V_psi shared across ALL layers and tokens).
# ---------------------------------------------------------------------------
def fit_shared_V_tokens(
    X, V, Y, LAY,
    d: int, n_layers: int,
    hidden: int = 256, depth: int = 2,
    steps: int = 4000, batch_size: int = 2048,
    lr: float = 3e-3, weight_decay: float = 1e-4,
    device: str = "cpu", seed: int = 0,
    verbose: bool = True,
):
    torch.manual_seed(seed)
    X_t = torch.from_numpy(X).to(device)
    V_t = torch.from_numpy(V).to(device)
    Y_t = torch.from_numpy(Y).to(device)
    L_t = torch.from_numpy(LAY).to(device).long()

    model = SharedPotential(d, hidden=hidden, depth=depth).to(device)
    alpha = nn.Parameter(torch.zeros(n_layers, device=device))
    beta  = nn.Parameter(torch.zeros(n_layers, device=device))

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
            print(f"[tok-shared-V] step {step+1:5d}/{steps}   "
                  f"loss {loss.item():.5f}   r2(batch) {r2_batch:+.3f}")
            hist.append({"step": step + 1, "loss": float(loss.item()), "r2_batch": r2_batch})

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
    return model, Y_pred, alpha.detach().cpu().numpy(), beta.detach().cpu().numpy(), hist


def predict_shared_V_tokens(model, X, V_, LAY, alpha_np, beta_np, device: str):
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
# Entry point.
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth",  type=int, default=2)
    ap.add_argument("--steps",  type=int, default=4000)
    ap.add_argument("--batch",  type=int, default=2048)
    ap.add_argument("--lr",     type=float, default=3e-3)
    ap.add_argument("--pca_k",  type=int, default=16)
    ap.add_argument("--ridge",  type=float, default=1e-3)
    ap.add_argument("--t_skip", type=int, default=2,
                    help="drop first/last t_skip tokens (BOS/EOS transients)")
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

    print(f"[tok-dir] tag={tag}  d={d}  L={L}  device={device}  t_skip={args.t_skip}")

    train = [tr for tr in trajs if tr.split == "train"]
    test  = [tr for tr in trajs if tr.split == "test"]

    # ============ 1) Shared-V_psi fit along TOKEN axis ============
    X_tr, V_tr, Y_tr, L_tr = build_token_triples(train, L, t_skip=args.t_skip)
    X_te, V_te, Y_te, L_te = build_token_triples(test,  L, t_skip=args.t_skip)
    print(f"[tok-shared-V] samples: train={X_tr.shape[0]:,}  test={X_te.shape[0]:,}")

    # Baselines.
    r2_null_tr = r2_per_layer(Y_tr, np.zeros_like(Y_tr), L_tr)
    r2_null_te = r2_per_layer(Y_te, np.zeros_like(Y_te), L_te)

    _, alpha_vo, r2_vo_tr = fit_velocity_only(X_tr, V_tr, Y_tr, L_tr)
    Y_pred_te_vo = np.zeros_like(Y_te)
    for ell, a in alpha_vo.items():
        m = L_te == ell
        Y_pred_te_vo[m] = a * V_te[m]
    r2_vo_te = r2_per_layer(Y_te, Y_pred_te_vo, L_te)

    # Shared-V main fit (V_psi across all token-triples at all layers).
    n_layers_effective = L + 1    # layers 0..L inclusive (since we loop ell in [0, L])
    model, Y_pred_tr, alpha_np, beta_np, _ = fit_shared_V_tokens(
        X_tr, V_tr, Y_tr, L_tr, d, n_layers_effective,
        hidden=args.hidden, depth=args.depth,
        steps=args.steps, batch_size=args.batch, lr=args.lr,
        device=device, seed=args.seed,
    )
    Y_pred_te = predict_shared_V_tokens(model, X_te, V_te, L_te, alpha_np, beta_np, device)
    r2_shv_tr = r2_per_layer(Y_tr, Y_pred_tr, L_tr)
    r2_shv_te = r2_per_layer(Y_te, Y_pred_te, L_te)
    r2_shv_tr_overall = r2_overall(Y_tr, Y_pred_tr)
    r2_shv_te_overall = r2_overall(Y_te, Y_pred_te)

    print()
    print(f"[tok-shared-V] overall R^2:  TRAIN {r2_shv_tr_overall:+.3f}   TEST {r2_shv_te_overall:+.3f}")
    print()
    print("[tok-shared-V] per-layer TEST R^2  (A null / B vel-only / C vel+shared-V):")
    for ell in sorted(r2_null_te):
        a = r2_null_te[ell]; b = r2_vo_te[ell]; c = r2_shv_te[ell]
        print(f"[tok-shared-V]   l={ell:2d}   null {a:+.3f}   vel-only {b:+.3f}   "
              f"vel+V {c:+.3f}   gain-over-velonly {c - b:+.3f}")

    # ============ 2) Token-direction velocity-aware Jacobian symmetry ============
    per_layer: List[Dict] = []
    for ell in range(L + 1):
        X_tr_l, _, _ = per_layer_samples_tokens(train, ell, t_skip=args.t_skip)
        if X_tr_l.shape[0] < args.pca_k + 2:
            continue
        Vk = _pca_basis(X_tr_l, args.pca_k)

        # Position-only.
        X_trp = X_tr_l @ Vk
        _, _, Y_tr_po = per_layer_samples_tokens(train, ell, t_skip=args.t_skip)
        X_tr_po = X_tr_l @ Vk                # same as X_trp
        Y_tr_po_pca = Y_tr_po @ Vk
        fit1 = fit_linear_and_symmetric(X_tr_po, Y_tr_po_pca, ridge=args.ridge)
        X_te_po_raw, _, Y_te_po_raw = per_layer_samples_tokens(test, ell, t_skip=args.t_skip)
        X_te_po = X_te_po_raw @ Vk
        Y_te_po = Y_te_po_raw @ Vk
        Xte_c = X_te_po - X_te_po.mean(0, keepdims=True)
        Yte_c = Y_te_po - Y_te_po.mean(0, keepdims=True)
        sst   = float(np.sum(Yte_c ** 2))
        r2_te_full_p = 1.0 - float(np.sum((Yte_c - Xte_c @ fit1["M_full"].T) ** 2)) / (sst + 1e-12)
        r2_te_sym_p  = 1.0 - float(np.sum((Yte_c - Xte_c @ fit1["M_sym"].T)  ** 2)) / (sst + 1e-12)

        # Velocity-aware.
        X_tr2, V_tr2, Y_tr2 = per_layer_samples_tokens(train, ell, t_skip=args.t_skip)
        X_tr2p = X_tr2 @ Vk; V_tr2p = V_tr2 @ Vk; Y_tr2p = Y_tr2 @ Vk
        fit2 = fit_second_order(X_tr2p, V_tr2p, Y_tr2p, ridge=args.ridge)

        X_te2, V_te2, Y_te2 = per_layer_samples_tokens(test, ell, t_skip=args.t_skip)
        X_te2p = X_te2 @ Vk; V_te2p = V_te2 @ Vk; Y_te2p = Y_te2 @ Vk
        Xte2c = X_te2p - X_te2p.mean(0, keepdims=True)
        Vte2c = V_te2p - V_te2p.mean(0, keepdims=True)
        Yte2c = Y_te2p - Y_te2p.mean(0, keepdims=True)
        sst2  = float(np.sum(Yte2c ** 2))
        r2_te_full_v = 1.0 - float(np.sum(
            (Yte2c - (Xte2c @ fit2["M_full"].T + Vte2c @ fit2["A_full"].T)) ** 2
        )) / (sst2 + 1e-12)
        r2_te_sym_v = 1.0 - float(np.sum(
            (Yte2c - (Xte2c @ fit2["M_sym"].T + Vte2c @ fit2["A_sym"].T)) ** 2
        )) / (sst2 + 1e-12)

        per_layer.append({
            "layer": ell,
            "r2_train_full_p": fit1["r2_full"], "r2_train_sym_p": fit1["r2_sym"],
            "r2_test_full_p":  r2_te_full_p,    "r2_test_sym_p":  r2_te_sym_p,
            "r2_train_full_v": fit2["r2_full"], "r2_train_sym_v": fit2["r2_sym"],
            "r2_test_full_v":  r2_te_full_v,    "r2_test_sym_v":  r2_te_sym_v,
        })
        print(f"[tok-jacsym] layer {ell}:  "
              f"POS-ONLY full={fit1['r2_full']:+.3f}/sym={fit1['r2_sym']:+.3f}    "
              f"VEL-AUG full={fit2['r2_full']:+.3f}/sym={fit2['r2_sym']:+.3f}    "
              f"TEST-VEL full={r2_te_full_v:+.3f}/sym={r2_te_sym_v:+.3f}")

    # ============ 3) Save npz + figure + markdown ============
    layers_shv = np.array(sorted(r2_null_te.keys()))
    layers_jac = np.array([p["layer"] for p in per_layer])
    gap_vel = np.array([p["r2_test_full_v"] - p["r2_test_sym_v"] for p in per_layer])

    npz_path = RESULTS_DIR / f"tokdir_{tag}_results.npz"
    np.savez(
        npz_path,
        tag=np.array([tag]),
        layers_shv=layers_shv,
        r2_null_train=np.array([r2_null_tr[e] for e in layers_shv]),
        r2_null_test =np.array([r2_null_te[e] for e in layers_shv]),
        r2_vo_train  =np.array([r2_vo_tr[e]   for e in layers_shv]),
        r2_vo_test   =np.array([r2_vo_te[e]   for e in layers_shv]),
        r2_shv_train =np.array([r2_shv_tr[e]  for e in layers_shv]),
        r2_shv_test  =np.array([r2_shv_te[e]  for e in layers_shv]),
        alpha=alpha_np, beta=beta_np,
        r2_shv_train_overall=np.array([r2_shv_tr_overall]),
        r2_shv_test_overall =np.array([r2_shv_te_overall]),
        layers_jac=layers_jac,
        **{k: np.array([p[k] for p in per_layer])
           for k in per_layer[0].keys() if k != "layer"},
        pca_k=np.array([args.pca_k]),
        t_skip=np.array([args.t_skip]),
        hidden=np.array([args.hidden]),
        depth=np.array([args.depth]),
    )
    print(f"[tok-dir] saved -> {npz_path}")

    # Combined figure.
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    # Panel 1: shared-V per-layer R^2.
    ax = axes[0]
    ax.plot(layers_shv, [r2_null_te[e] for e in layers_shv], marker="x",
            color="tab:gray", label="A. static null")
    ax.plot(layers_shv, [r2_vo_te[e]   for e in layers_shv], marker="s",
            color="tab:orange", label="B. velocity-only")
    ax.plot(layers_shv, [r2_shv_te[e]  for e in layers_shv], marker="o",
            color="tab:blue", label=r"C. velocity + shared $V_\psi$")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel(r"layer $\ell$"); ax.set_ylabel(r"TEST $R^2$")
    ax.set_title("Shared-$V_\\psi$ fit (token direction)")
    ax.set_ylim(-0.3, 1.05); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    # Panel 2: Jacobian pos-only.
    ax = axes[1]
    ax.plot(layers_jac, [p["r2_test_full_p"] for p in per_layer], marker="o",
            label="unconstrained $M_\\ell$")
    ax.plot(layers_jac, [p["r2_test_sym_p"]  for p in per_layer], marker="s",
            label="symmetric-restricted")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel(r"layer $\ell$"); ax.set_ylabel(r"TEST $R^2$")
    ax.set_title("Jacobian POS-only (token direction)")
    ax.set_ylim(-0.3, 1.0); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    # Panel 3: Jacobian vel-aug.
    ax = axes[2]
    ax.plot(layers_jac, [p["r2_test_full_v"] for p in per_layer], marker="o",
            label="unconstrained $M_\\ell$")
    ax.plot(layers_jac, [p["r2_test_sym_v"]  for p in per_layer], marker="s",
            label="symmetric-restricted")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel(r"layer $\ell$"); ax.set_ylabel(r"TEST $R^2$")
    ax.set_title("Jacobian VEL-aware (token direction)")
    ax.set_ylim(-0.3, 1.0); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(f"Token-direction diagnostics -- {tag}")
    fig.tight_layout()
    fig_path = RESULTS_DIR / f"tokdir_{tag}_fig.png"
    fig.savefig(fig_path, dpi=130); plt.close(fig)
    print(f"[tok-dir] saved -> {fig_path}")

    # Markdown summary.
    md = RESULTS_DIR / f"tokdir_{tag}_summary.md"
    verdict_vel = (
        "Velocity-aware symmetric-restricted fit tracks the unconstrained "
        f"fit (max TEST gap = {gap_vel.max():+.3f}).  The per-token "
        "spring matrix is consistent with a symmetric Hessian at every "
        "layer: the token-direction dynamics is locally Hessian-of-scalar."
        if gap_vel.max() < 0.10 else
        f"Velocity-aware: gap of up to {gap_vel.max():+.3f} between "
        "unconstrained and symmetric-restricted fits at some layers."
    )
    with md.open("w") as f:
        f.write(f"# Token-direction diagnostics -- {tag}\n\n")
        f.write("Runs the same two diagnostics as `shared_potential_fit.py` / "
                "`jacobian_symmetry.py` but along the **token axis** at a fixed "
                "layer, instead of along the **layer axis** at a fixed token.  "
                "This tests STP's Geodesic Hypothesis in its natural coordinate "
                "system -- the sequence-time axis of autoregressive inference.\n\n")
        f.write(f"- Hidden dim `d = {d}`; layers `L = {L}`; "
                f"samples: TRAIN {X_tr.shape[0]:,} / TEST {X_te.shape[0]:,}\n")
        f.write(f"- $V_\\psi$: {args.depth}-layer MLP, hidden = {args.hidden}, GELU\n")
        f.write(f"- PCA $k = {args.pca_k}$, ridge = {args.ridge}, "
                f"BOS/EOS skip = {args.t_skip} tokens\n\n")
        f.write("## 1. Shared-$V_\\psi$ fit (token direction)\n\n")
        f.write(f"- TRAIN pooled $R^2$ = **{r2_shv_tr_overall:+.3f}**\n")
        f.write(f"- TEST pooled $R^2$  = **{r2_shv_te_overall:+.3f}**\n\n")
        f.write("| layer | A. static null | B. velocity-only | "
                "C. velocity + shared $V_\\psi$ | C - B |\n")
        f.write("|--:|--:|--:|--:|--:|\n")
        for ell in layers_shv:
            f.write(f"| {int(ell)} | {r2_null_te[int(ell)]:+.3f} | "
                    f"{r2_vo_te[int(ell)]:+.3f} | {r2_shv_te[int(ell)]:+.3f} | "
                    f"{r2_shv_te[int(ell)] - r2_vo_te[int(ell)]:+.3f} |\n")
        f.write("\n### Learned per-layer scalars\n\n")
        f.write("| layer | $\\alpha_\\ell$ | $\\beta_\\ell$ |\n|--:|--:|--:|\n")
        for ell in layers_shv:
            f.write(f"| {int(ell)} | {alpha_np[int(ell)]:+.4f} | "
                    f"{beta_np[int(ell)]:+.4f} |\n")
        f.write("\n## 2. Velocity-aware Jacobian-symmetry (token direction)\n\n")
        f.write("| layer | POS-only $R^2_\\text{full}$ | POS-only $R^2_\\text{sym}$ | "
                "VEL-aug $R^2_\\text{full}$ | VEL-aug $R^2_\\text{sym}$ | "
                "VEL-aug gap |\n")
        f.write("|--:|--:|--:|--:|--:|--:|\n")
        for p in per_layer:
            g = p["r2_test_full_v"] - p["r2_test_sym_v"]
            f.write(f"| {p['layer']} | {p['r2_test_full_p']:+.3f} | "
                    f"{p['r2_test_sym_p']:+.3f} | {p['r2_test_full_v']:+.3f} | "
                    f"{p['r2_test_sym_v']:+.3f} | {g:+.3f} |\n")
        f.write(f"\n### Verdict\n\n- {verdict_vel}\n\n")
        f.write("## Artefacts\n\n")
        f.write(f"- `tokdir_{tag}_results.npz`\n")
        f.write(f"- `tokdir_{tag}_fig.png`\n")
    print(f"[tok-dir] saved -> {md}")


if __name__ == "__main__":
    main()
