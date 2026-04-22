"""
SPLM oracle reference for the step-2 shared-V_psi fit.

Setup
-----
The step-2 test fit a generic MLP V_psi(h) and measured how well
a single scalar can reproduce per-layer Delta x_l across all
layers and sentences.  For SPLM, the "ideal" such scalar is
SPLM's own learned V_theta(xi, h) -- by construction the model's
dynamics is

    Delta h_l  =  v_{l+1} * dt
               =  (1/(1+dt*gamma)) * [ dt * v_l  -  dt^2/m * grad_h V_theta(xi, h_l) ]

Plugging this oracle V_theta into the step-2 ansatz

    Delta h_l ~ alpha_l * v_l  -  beta_l * grad_h V_theta(xi, h_l)

should fit essentially perfectly (TRAIN R^2 ~ 1, TEST R^2 ~ 1), because
it IS the closed-form integrator up to unit-constants alpha_l, beta_l.
The test therefore quantifies two things:

  A) How much of the step-2 SPLM residual is explained simply by the
     context dependence dropped when V_psi(h) ignores xi.
  B) Whether the step-2 learned V_psi recovers (an approximation of)
     SPLM's true V_theta -- by comparing per-layer R^2 oracle vs learned.

This is the PC-side upper bound for step 2.

Works on raw h-space (not x_ps), since V_theta lives there.
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
import torch.nn.functional as F

from model import ScalarPotentialLM, SPLMConfig
from e_init_corpus import CORPUS


SCRIPT_DIR  = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"


# ---------------------------------------------------------------------------
def load_splm(ckpt_path: Path, device: str) -> ScalarPotentialLM:
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = SPLMConfig(**raw["model_cfg"])
    model = ScalarPotentialLM(cfg).to(device)
    model.load_state_dict(raw["model_state_dict"])
    model.eval()
    return model


def tokenize(sentence: str, max_len: int) -> np.ndarray:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    ids = tok.encode(sentence)
    ids = ids[:max_len]
    return np.asarray(ids, dtype=np.int64)


def extract_oracle_tuples(model: ScalarPotentialLM, sentence: str, device: str) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For a single sentence, re-run SPLM and return
       H   : (L+1, T, d)  raw hidden states  h_l
       GV  : (L,   T, d)  grad_h V_theta(xi, h_l)  for l = 0..L-1
       XI  : (T, d)  the constant context vector

    The dynamics equation is

       h_{l+1} = h_l + dt * v_{l+1}
       v_{l+1} = (v_l + dt*f_l/m) / (1 + dt*gamma),  f_l = -grad_h V_theta(xi, h_l)

    so Delta h_l := h_{l+1} - h_l depends on (v_l, grad_V at h_l).
    """
    max_len = model.cfg.max_len
    ids = tokenize(sentence, max_len)
    x = torch.from_numpy(ids).unsqueeze(0).to(device)

    with torch.enable_grad():
        emb, xi = model._embed_and_pool(x)
        h = emb
        v = torch.zeros_like(h)
        dt, m, gamma = model.cfg.dt, model.m, model.gamma

        H  = [h.detach().cpu().numpy()[0]]
        GV = []
        for _ in range(model.cfg.L):
            h_in = h.detach().requires_grad_(True)
            V_out = model.V_theta(xi, h_in).sum()
            grad_V, = torch.autograd.grad(V_out, h_in)
            GV.append(grad_V.detach().cpu().numpy()[0])
            f = -grad_V
            v = (v + dt * f / m) / (1.0 + dt * gamma)
            h = h_in + dt * v
            H.append(h.detach().cpu().numpy()[0])

    return (np.stack(H, axis=0).astype(np.float32),         # (L+1, T, d)
            np.stack(GV, axis=0).astype(np.float32),        # (L, T, d)
            xi.detach().cpu().numpy()[0].astype(np.float32) # (T, d)
            )


# ---------------------------------------------------------------------------
def build_pools(model: ScalarPotentialLM, sentences: List[Dict], device: str):
    """For each (split, layer l >= 1) build pooled (v, dh, grad_V_theta, grad_V_theta_h_aligned).

    We fit on

        Delta h_l  ~=  alpha_l * v_l  -  beta_l * grad_V_theta(xi, h_l)

    where v_l := h_l - h_{l-1}.  Note: v_l at layer l>=1 requires h_{l-1}.
    """
    train_V, train_dH, train_G, train_Lay = [], [], [], []
    test_V,  test_dH,  test_G,  test_Lay  = [], [], [], []

    for s in sentences:
        H, GV, _XI = extract_oracle_tuples(model, s["sentence"], device)
        L = H.shape[0] - 1
        for ell in range(1, L):
            v_l   = H[ell]   - H[ell - 1]
            dh_l  = H[ell+1] - H[ell]
            g_l   = GV[ell]               # grad V_theta at h_l -- for l in 1..L-1
            layer = np.full((H.shape[1],), ell, dtype=np.int64)
            bucket = (train_V, train_dH, train_G, train_Lay) \
                if s["split"] == "train" \
                else (test_V, test_dH, test_G, test_Lay)
            bucket[0].append(v_l);  bucket[1].append(dh_l)
            bucket[2].append(g_l);  bucket[3].append(layer)

    def cat(lst):
        return np.concatenate(lst, axis=0) if lst else np.zeros((0, 1), dtype=np.float32)

    return (cat(train_V),  cat(train_dH),  cat(train_G),  cat(train_Lay),
            cat(test_V),   cat(test_dH),   cat(test_G),   cat(test_Lay))


# ---------------------------------------------------------------------------
def fit_scalars(V: np.ndarray, G: np.ndarray, Y: np.ndarray, LAY: np.ndarray):
    """Per-layer least-squares: [alpha_l, beta_l] = argmin || Y - alpha V - beta*(-G) ||^2."""
    alpha = {}; beta = {}
    Y_pred = np.zeros_like(Y)
    for ell in np.unique(LAY):
        m = LAY == ell
        v = V[m].reshape(-1); g = (-G[m]).reshape(-1); y = Y[m].reshape(-1)
        A = np.stack([v, g], axis=1)  # (N*d, 2)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        a, b = coef
        alpha[int(ell)] = float(a); beta[int(ell)] = float(b)
        Y_pred[m] = a * V[m] + b * (-G[m])
    return Y_pred, alpha, beta


def r2_per_layer(Y: np.ndarray, Yp: np.ndarray, LAY: np.ndarray) -> Dict[int, float]:
    out = {}
    for ell in np.unique(LAY):
        m = LAY == ell
        num = float(np.sum((Y[m] - Yp[m]) ** 2))
        den = float(np.sum((Y[m] - Y[m].mean(0, keepdims=True)) ** 2))
        out[int(ell)] = 1.0 - num / (den + 1e-12)
    return out


def predict(V, G, LAY, alpha, beta):
    Yp = np.zeros_like(V)
    for ell in np.unique(LAY):
        m = LAY == ell
        Yp[m] = alpha[int(ell)] * V[m] + beta[int(ell)] * (-G[m])
    return Yp


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available()
                              else ("mps" if torch.backends.mps.is_available() else "cpu"))

    ckpt_path = Path(args.ckpt)
    tag = args.tag or ckpt_path.stem.replace(".ckpt_latest", "").replace("splm_", "")
    print(f"[splm-oracle] ckpt={ckpt_path.name}  tag={tag}  device={device}")

    model = load_splm(ckpt_path, device)
    L = model.cfg.L
    d = model.cfg.d

    sentences = []
    rng = np.random.default_rng(0)
    for domain, arr in CORPUS.items():
        idx = rng.permutation(len(arr))
        train_idx = idx[:int(0.8 * len(arr))]
        test_idx  = idx[int(0.8 * len(arr)):]
        for i, s in enumerate(arr):
            split = "train" if i in set(train_idx.tolist()) else "test"
            sentences.append(dict(sentence=s, domain=domain, split=split))
    print(f"[splm-oracle] sentences: train="
          f"{sum(1 for s in sentences if s['split']=='train')}, "
          f"test={sum(1 for s in sentences if s['split']=='test')}")

    (V_tr, Y_tr, G_tr, Ltr,
     V_te, Y_te, G_te, Lte) = build_pools(model, sentences, device)
    print(f"[splm-oracle] samples: train={V_tr.shape[0]:,}  test={V_te.shape[0]:,}")

    Y_pred_tr, alpha, beta = fit_scalars(V_tr, G_tr, Y_tr, Ltr)
    Y_pred_te = predict(V_te, G_te, Lte, alpha, beta)
    r2_tr = r2_per_layer(Y_tr, Y_pred_tr, Ltr)
    r2_te = r2_per_layer(Y_te, Y_pred_te, Lte)

    print("\n[splm-oracle] per-layer TEST R^2 (oracle V_theta):")
    for ell in sorted(r2_te):
        print(f"[splm-oracle]   l={ell:2d}   train {r2_tr[ell]:+.4f}   "
              f"test {r2_te[ell]:+.4f}   alpha={alpha[ell]:+.4f}   "
              f"beta={beta[ell]:+.4f}")

    # Compare to step-2 learned V_psi fit (on same SPLM).
    step2_npz = RESULTS_DIR / "sharedV_shakespeare_ckpt_latest_results.npz"
    cmp_layers, cmp_shv = None, None
    if step2_npz.exists():
        z = np.load(step2_npz)
        cmp_layers = z["layers"].tolist()
        cmp_shv    = z["r2_shv_test"].tolist()

    # ---- Save ----
    out_npz = RESULTS_DIR / f"splm_oracle_{tag}_results.npz"
    np.savez(
        out_npz,
        layers=np.array(sorted(r2_te.keys())),
        r2_train=np.array([r2_tr[e] for e in sorted(r2_tr)]),
        r2_test =np.array([r2_te[e] for e in sorted(r2_te)]),
        alpha=np.array([alpha[e] for e in sorted(alpha)]),
        beta =np.array([beta[e]  for e in sorted(beta)]),
    )
    print(f"[splm-oracle] saved -> {out_npz}")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(7, 4.3))
    layers = sorted(r2_te.keys())
    y_or_te = [r2_te[e] for e in layers]
    y_or_tr = [r2_tr[e] for e in layers]
    ax.plot(layers, y_or_tr, marker="o", linewidth=2.0, color="tab:blue",
            label=f"oracle $V_\\theta$ TRAIN")
    ax.plot(layers, y_or_te, marker="o", linewidth=2.0, color="tab:cyan",
            label=f"oracle $V_\\theta$ TEST")
    if cmp_shv is not None:
        ax.plot(cmp_layers, cmp_shv, marker="s", linewidth=1.5, color="tab:green",
                linestyle="--", label="learned $V_\\psi$ TEST (step-2)")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axhline(1, color="gray", linewidth=0.3, linestyle=":")
    ax.set_xlabel("layer $\\ell$"); ax.set_ylabel("$R^2$")
    ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title(f"SPLM oracle $V_\\theta$ vs. learned $V_\\psi$ -- {tag}")
    fig.tight_layout()
    fig_path = RESULTS_DIR / f"splm_oracle_{tag}_fig.png"
    fig.savefig(fig_path, dpi=140); plt.close(fig)
    print(f"[splm-oracle] saved -> {fig_path}")

    # ---- Markdown ----
    md = RESULTS_DIR / f"splm_oracle_{tag}_summary.md"
    with md.open("w") as f:
        f.write(f"# SPLM oracle fit -- {tag}\n\n")
        f.write("**Purpose.**  Upper-bound reference for the step-2 "
                "shared-$V_\\psi$ fit on SPLM.  Replaces the learned "
                "$V_\\psi(h)$ with SPLM's own $V_\\theta(\\xi, h)$ and "
                "keeps the same per-layer $\\alpha_\\ell, \\beta_\\ell$ "
                "fitting procedure.  Numerical mismatch from 1.0 is then "
                "purely due to integrator constants and numerical precision.\n\n")
        f.write("## Per-layer $R^2$  (oracle $V_\\theta$)\n\n")
        f.write("| layer | TRAIN | TEST | $\\alpha_\\ell$ | $\\beta_\\ell$ |\n")
        f.write("|--:|--:|--:|--:|--:|\n")
        for ell in layers:
            f.write(f"| {ell} | {r2_tr[ell]:+.4f} | {r2_te[ell]:+.4f} | "
                    f"{alpha[ell]:+.4f} | {beta[ell]:+.4f} |\n")
        if cmp_shv is not None:
            f.write("\n## Oracle vs. learned $V_\\psi$ (step-2)\n\n")
            f.write("| layer | oracle TEST | learned $V_\\psi$ TEST | gap |\n")
            f.write("|--:|--:|--:|--:|\n")
            for ell, shv in zip(cmp_layers, cmp_shv):
                orc = r2_te.get(int(ell))
                if orc is None:
                    continue
                f.write(f"| {ell} | {orc:+.4f} | {shv:+.4f} | {orc - shv:+.4f} |\n")
        f.write("\n![fig](splm_oracle_"
                f"{tag}_fig.png)\n")
    print(f"[splm-oracle] saved -> {md}")


if __name__ == "__main__":
    main()
