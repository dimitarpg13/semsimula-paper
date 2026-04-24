"""
Semantic-attractor extraction from SPLM's scalar potential V_theta.

Given a trained SPLM, fix xi from a real context and search for local
minima of V_theta(xi, h) in h-space by gradient descent from many
initializations.  Cluster the converged h*; decode each cluster
centroid with the tied LM head.  The resulting token distributions
are the "semantic attractors" the model has learned for that context.

Usage:
  python3 attractor_extraction.py \
      --ckpt ../symplectic_variant/results/splm_sym_logfreq_shakespeare_L16_dt05_ckpt_latest.pt \
      --tag sym_L16_dt05
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

SCRIPT_DIR = Path(__file__).parent
PARENT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(PARENT_DIR))


# ---------------------------------------------------------------------------
# Prompts used to condition xi -- five domains, one probe per domain.
# (Kept short so the xi at the last token is sharp, not averaged out.)
# ---------------------------------------------------------------------------
PROMPTS: List[Tuple[str, str]] = [
    ("narrative",  "The old king sat on the"),
    ("mathematics", "The theorem states that for every"),
    ("scientific",  "Photosynthesis converts carbon dioxide and"),
    ("dialogue",    "She whispered: I love"),
    ("code",        "def fibonacci(n): return 1 if n < 2 else"),
]


# ---------------------------------------------------------------------------
# Model loader that autodetects which SPLM variant the checkpoint came from.
# ---------------------------------------------------------------------------
def load_model(ckpt_path: str, device: str):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    variant = ck.get("variant", "sarf_mass")
    if variant == "symplectic":
        from symplectic_variant.model_symplectic import (     # noqa: E402
            ScalarPotentialLMSymplectic as ModelCls,
            SPLMSymplecticConfig as ConfigCls,
        )
    elif variant == "sarf_mass_ln":
        from energetic_minima.model_ln import (               # noqa: E402
            ScalarPotentialLMSARFMassLN as ModelCls,
            SPLMSARFMassLNConfig as ConfigCls,
        )
    elif variant == "sarf_mass_gm":
        from energetic_minima.model_gm import (               # noqa: E402
            ScalarPotentialLMSARFMassGM as ModelCls,
            SPLMSARFMassGMConfig as ConfigCls,
        )
    else:
        from sarf_mass_variant.model_sarf_mass import (       # noqa: E402
            ScalarPotentialLMSARFMass as ModelCls,
            SPLMSARFMassConfig as ConfigCls,
        )
    cfg = ConfigCls(**ck["model_cfg"])
    model = ModelCls(cfg).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, cfg, variant


# ---------------------------------------------------------------------------
# Extract xi at (last layer, last token) for a prompt.
# ---------------------------------------------------------------------------
def get_xi(model, tokenizer, prompt: str, device: str
           ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    ids_list = tokenizer.encode(prompt)
    ids = torch.tensor(ids_list, device=device, dtype=torch.long).unsqueeze(0)
    with torch.enable_grad():
        out = model(ids, return_trajectory=True, return_xi_trajectory=True)
    logits, _, traj_h, traj_xi = out
    xi_last = traj_xi[-1][0, -1, :].to(device).detach()      # (d,)
    h_last  = traj_h[-1][0, -1, :].to(device).detach()       # (d,)
    return xi_last, h_last, ids_list


# ---------------------------------------------------------------------------
# Build seed initialisations from three complementary distributions.
# ---------------------------------------------------------------------------
def build_seeds(
    model,
    tokenizer,
    device: str,
    d: int,
    n_gauss: int = 256,
    n_tok: int = 256,
    n_real: int = 256,
    real_noise: float = 0.3,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (seeds, h_mean, h_std) -- statistics used to anchor descent."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    h_real_list: List[torch.Tensor] = []
    for _, p in PROMPTS:
        ids = torch.tensor(tokenizer.encode(p), device=device,
                           dtype=torch.long).unsqueeze(0)
        with torch.enable_grad():
            _, _, traj_h, _ = model(ids, return_trajectory=True,
                                    return_xi_trajectory=True)
        h_real_list.append(traj_h[-1][0].to(device))         # (T, d)
    H_real = torch.cat(h_real_list, dim=0)                   # (sum_T, d)
    h_mean = H_real.mean(0)
    h_std  = H_real.std(0).clamp_min(1e-3)

    g_noise = torch.randn(n_gauss, d, generator=gen).to(device)
    h0_gauss = h_mean + g_noise * h_std

    V_tok = model.E.weight.shape[0]
    idx = torch.randint(0, V_tok, (n_tok,), generator=gen)
    h0_tok = model.E.weight.detach()[idx].to(device)

    n_real_available = H_real.shape[0]
    idx_real = torch.randint(0, n_real_available, (n_real,), generator=gen)
    noise = torch.randn(n_real, d, generator=gen).to(device) * real_noise
    h0_real = H_real[idx_real] + noise * h_std

    seeds = torch.cat([h0_gauss, h0_tok, h0_real], dim=0)    # (N, d)
    return seeds, h_mean.detach(), h_std.detach()


# ---------------------------------------------------------------------------
# Descend h on V_theta(xi, h) at fixed xi.  Returns converged h and gradnorm.
# ---------------------------------------------------------------------------
def descend(
    model,
    xi: torch.Tensor,
    h_seeds: torch.Tensor,
    h_center: torch.Tensor,
    h_std: torch.Tensor,
    lambda_reg: float,
    steps: int = 2000,
    lr: float = 0.05,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Descend on   V_theta(xi, h) + (lambda_reg/2) * sum_d ((h - h_c)/h_s)^2.

    The regulariser (= log-density of an isotropic prior on the data
    manifold, in standardised coordinates) is what makes the problem
    well-posed: V_theta is unbounded below, so unregularised descent
    runs h off to infinity.  The anchored objective gives the modes of
    the posterior  pi(h|xi) proportional to  exp(-V_theta(xi,h)) * prior(h).
    """
    h = h_seeds.clone().detach().to(device).requires_grad_(True)
    xi_b = xi.unsqueeze(0).expand(h.shape[0], -1).to(device)
    h_c = h_center.to(device)
    h_s = h_std.to(device)
    opt = torch.optim.Adam([h], lr=lr)
    for step in range(steps):
        V = model.V_theta(xi_b, h).sum()
        reg = 0.5 * lambda_reg * (((h - h_c) / h_s) ** 2).sum()
        loss = V + reg
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % max(steps // 5, 1) == 0:
            with torch.no_grad():
                v_mean = V.item() / h.shape[0]
                r_mean = reg.item() / h.shape[0]
            print(f"  [descend] step {step+1:5d}/{steps}  "
                  f"<V>={v_mean:+.3f}  <reg>={r_mean:+.3f}  "
                  f"<loss>={v_mean+r_mean:+.3f}")

    h_f = h.detach().clone().requires_grad_(True)
    V = model.V_theta(xi_b, h_f).sum()
    reg = 0.5 * lambda_reg * (((h_f - h_c) / h_s) ** 2).sum()
    g, = torch.autograd.grad(V + reg, h_f)
    grad_norm = g.norm(dim=-1).detach().cpu().numpy()
    with torch.no_grad():
        V_final = model.V_theta(xi_b, h_f).detach().cpu().numpy().ravel()
        reg_final = (0.5 * (((h_f - h_c) / h_s) ** 2).sum(dim=-1)
                     ).detach().cpu().numpy().ravel() * lambda_reg
    return (h.detach().cpu().numpy(), grad_norm, V_final, reg_final)


# ---------------------------------------------------------------------------
# Dynamical attractors: run SPLM's own damped integrator from h_seeds at
# fixed xi.  These are the attractors the framework actually predicts -- the
# attractors of the damped second-order dynamics, not the minima of V_theta.
# ---------------------------------------------------------------------------
def simulate_damped(
    model,
    xi: torch.Tensor,
    h_seeds: torch.Tensor,
    m_scalar: float,
    gamma_scalar: float,
    dt: float,
    n_steps: int,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Semi-implicit Euler on   m ddot h = -grad V(xi, h) - m gamma dot h.

    We use the same per-token mass scheme as SPLM training (global scalar
    here, since one particle is simulated in isolation).  xi is held
    fixed at the prompt-level vector; m and gamma are the learned values
    exported from the checkpoint.
    """
    h = h_seeds.clone().detach().to(device)
    v = torch.zeros_like(h)
    xi_b = xi.unsqueeze(0).expand(h.shape[0], -1).to(device)
    m = float(m_scalar)
    gamma = float(gamma_scalar)
    print(f"  [sim] m={m:.3f}  gamma={gamma:.3f}  dt={dt:.3f}  steps={n_steps}  N={h.shape[0]}")
    for step in range(n_steps):
        h.requires_grad_(True)
        V = model.V_theta(xi_b, h).sum()
        g, = torch.autograd.grad(V, h)
        h = h.detach()
        f = -g.detach()
        v = (v + dt * f / m) / (1.0 + dt * gamma)
        h = h + dt * v
        if (step + 1) % max(n_steps // 6, 1) == 0:
            with torch.no_grad():
                V_mean = model.V_theta(xi_b, h).sum().item() / h.shape[0]
                v_norm = v.norm(dim=-1).mean().item()
                h_norm = h.norm(dim=-1).mean().item()
            print(f"  [sim] step {step+1:4d}/{n_steps}  "
                  f"<V>={V_mean:+.3f}  <|v|>={v_norm:.3f}  <|h|>={h_norm:.3f}")

    h_final = h.detach()
    h_rg = h_final.clone().requires_grad_(True)
    Vf = model.V_theta(xi_b, h_rg).sum()
    gf, = torch.autograd.grad(Vf, h_rg)
    v_norm_per_sample = v.norm(dim=-1).detach().cpu().numpy()
    with torch.no_grad():
        V_final = model.V_theta(xi_b, h_rg).detach().cpu().numpy().ravel()
    return (h_final.cpu().numpy(), V_final, v_norm_per_sample)


# ---------------------------------------------------------------------------
# KMeans with silhouette-score sweep over K.
# ---------------------------------------------------------------------------
def sweep_clusters(h: np.ndarray, K_min: int = 2, K_max: int = 12, seed: int = 0
                   ) -> Tuple[int, np.ndarray, np.ndarray, Dict[int, float]]:
    sils: Dict[int, float] = {}
    best_K, best_sil = K_min, -np.inf
    for K in range(K_min, K_max + 1):
        if len(h) <= K:
            continue
        km = KMeans(n_clusters=K, random_state=seed, n_init=10).fit(h)
        try:
            sil = silhouette_score(h, km.labels_)
        except Exception:
            sil = -np.inf
        sils[K] = float(sil)
        if sil > best_sil:
            best_sil, best_K = sil, K
    km = KMeans(n_clusters=best_K, random_state=seed, n_init=10).fit(h)
    return best_K, km.labels_, km.cluster_centers_, sils


# ---------------------------------------------------------------------------
# Decode cluster centroids via the tied LM head.
# ---------------------------------------------------------------------------
def decode_centroids(centroids: np.ndarray, model, tokenizer, top_k: int = 15
                     ) -> List[List[Tuple[str, float]]]:
    E = model.E.weight.detach()
    out: List[List[Tuple[str, float]]] = []
    for c in centroids:
        c_t = torch.from_numpy(c).to(E.device).to(E.dtype)
        logits = c_t @ E.T
        probs = F.softmax(logits, dim=-1)
        top = torch.topk(probs, top_k)
        decoded = []
        for idx, prob in zip(top.indices, top.values):
            tok = tokenizer.decode([idx.item()])
            tok = tok.replace("\n", "\\n").replace("\t", "\\t")
            decoded.append((tok, float(prob.item())))
        out.append(decoded)
    return out


# ---------------------------------------------------------------------------
# Ground truth: decode the real h_last of the prompt through the LM head.
# ---------------------------------------------------------------------------
def decode_real_h(h: torch.Tensor, model, tokenizer, top_k: int = 15
                  ) -> List[Tuple[str, float]]:
    logits = h @ model.E.weight.T
    probs = F.softmax(logits, dim=-1)
    top = torch.topk(probs, top_k)
    out = []
    for idx, prob in zip(top.indices, top.values):
        tok = tokenizer.decode([idx.item()])
        tok = tok.replace("\n", "\\n").replace("\t", "\\t")
        out.append((tok, float(prob.item())))
    return out


# ---------------------------------------------------------------------------
# Figure: per-prompt PCA scatter + per-attractor top-token bar chart.
# ---------------------------------------------------------------------------
def plot_attractors(
    prompt_name: str,
    prompt_text: str,
    h_conv: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    decoded: List[List[Tuple[str, float]]],
    h_real_pt: np.ndarray,
    real_decoded: List[Tuple[str, float]],
    K: int,
    out_path: Path,
) -> None:
    pca = PCA(n_components=2).fit(h_conv)
    h2 = pca.transform(h_conv)
    c2 = pca.transform(centroids)
    r2 = pca.transform(h_real_pt[None, :])

    fig = plt.figure(figsize=(14, 5 + 0.25 * K))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.4])

    ax = fig.add_subplot(gs[0, 0])
    cmap = plt.get_cmap("tab20", max(K, 3))
    for k in range(K):
        m = labels == k
        ax.scatter(h2[m, 0], h2[m, 1], s=12, color=cmap(k), alpha=0.45,
                   label=f"A{k}")
    ax.scatter(c2[:, 0], c2[:, 1], s=180, c="black", marker="X",
               edgecolors="white", linewidths=1.5, zorder=5)
    ax.scatter(r2[:, 0], r2[:, 1], s=260, facecolors="none",
               edgecolors="red", linewidths=2.5, zorder=6,
               label="real $h_L$")
    for k in range(K):
        ax.annotate(f"A{k}", (c2[k, 0], c2[k, 1]),
                    color="white", fontsize=8, ha="center", va="center",
                    fontweight="bold", zorder=7)
    ax.set_title(f"converged $h^\\ast$ (PCA-2)  --  K={K}")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    y_pos, y_labels = [], []
    for k in range(K):
        toks = decoded[k][:8]
        line = "  ".join(f"{t.strip()}·{p:.2f}" for t, p in toks)
        y_labels.append(f"A{k}: {line}")
        y_pos.append(K - 1 - k)
    ax2.set_yticks(y_pos); ax2.set_yticklabels(y_labels, fontsize=8)
    ax2.set_xticks([])
    for s in ("top", "right", "bottom", "left"):
        ax2.spines[s].set_visible(False)
    real_line = "  ".join(f"{t.strip()}·{p:.2f}" for t, p in real_decoded[:8])
    ax2.set_title(f"attractor centroids -> tied-embedding top-8 tokens\n"
                  f"real continuation top-8: {real_line}", fontsize=9)

    fig.suptitle(f'"{prompt_text}"  --  {prompt_name}', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--n_gauss", type=int, default=256)
    ap.add_argument("--n_tok",   type=int, default=256)
    ap.add_argument("--n_real",  type=int, default=256)
    ap.add_argument("--K_min",  type=int, default=2)
    ap.add_argument("--K_max",  type=int, default=12)
    ap.add_argument("--grad_eps", type=float, default=0.05)
    ap.add_argument("--lambda_reg", type=float, default=100.0,
                    help="Strength of the data-manifold anchor regulariser "
                         "(0.5 * lambda * ||(h - h_c) / h_s||^2).  Only used "
                         "in --mode gradient.")
    ap.add_argument("--mode",
                    choices=["gradient", "dynamical"],
                    default="dynamical",
                    help="gradient: Adam on V_theta+anchor (diagnostic only, "
                         "V_theta has no finite minima).  dynamical: run "
                         "SPLM's damped integrator from random h seeds at "
                         "fixed xi (attractors of the actual model).")
    ap.add_argument("--n_sim_steps", type=int, default=128,
                    help="Number of damped-dynamics steps in --mode dynamical.")
    ap.add_argument("--sim_dt", type=float, default=None,
                    help="dt for the damped simulation.  Defaults to "
                         "cfg.dt (the training value).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = args.device or (
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"[attractors] device={device}  ckpt={args.ckpt}  tag={args.tag}")

    model, cfg, variant = load_model(args.ckpt, device)
    print(f"[attractors] variant={variant}  d={cfg.d}  L={cfg.L}  "
          f"mass_mode={cfg.mass_mode}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    h_seeds, h_center, h_std = build_seeds(
        model, tokenizer, device, d=cfg.d,
        n_gauss=args.n_gauss, n_tok=args.n_tok, n_real=args.n_real,
        seed=args.seed,
    )
    N = h_seeds.shape[0]
    print(f"[attractors] seeds: {N} total  "
          f"(gauss {args.n_gauss}, tok {args.n_tok}, real {args.n_real})")
    print(f"[attractors] data-manifold stats: |h_center|={h_center.norm().item():.3f}  "
          f"mean h_std={h_std.mean().item():.4f}  lambda_reg={args.lambda_reg}")

    all_results = []
    for prompt_name, prompt_text in PROMPTS:
        print(f"\n[attractors] === prompt: {prompt_name} :: '{prompt_text}' ===")
        xi, h_real, ids = get_xi(model, tokenizer, prompt_text, device)
        real_decoded = decode_real_h(h_real, model, tokenizer, top_k=15)
        V_real = float(model.V_theta(xi.unsqueeze(0), h_real.unsqueeze(0))
                       .detach().cpu().item())
        print(f"  real h_L: V={V_real:+.3f}  |h_L|={h_real.norm().item():.3f}")
        print(f"  real continuation top-5: "
              + ", ".join(f"{t}({p:.2f})" for t, p in real_decoded[:5]))

        if args.mode == "gradient":
            h_all, grad_norm, V_final, reg_final = descend(
                model, xi, h_seeds, h_center, h_std, args.lambda_reg,
                steps=args.steps, lr=args.lr, device=device,
            )
            converged = grad_norm < args.grad_eps
            n_conv = int(converged.sum())
            print(f"  converged: {n_conv}/{N}  "
                  f"(grad_eps={args.grad_eps})")
            if n_conv > 0:
                print(f"  <V|conv>={float(V_final[converged].mean()):+.3f}  "
                      f"<reg|conv>={float(reg_final[converged].mean()):+.3f}")
            if (~converged).any():
                print(f"  <V|fail>={float(V_final[~converged].mean()):+.3f}  "
                      f"<reg|fail>={float(reg_final[~converged].mean()):+.3f}")
        else:  # dynamical
            m_scalar = float(F.softplus(model.raw_m_bias).detach().cpu().item() + 1e-3)
            gamma_scalar = float(model.gamma.detach().cpu().item())
            dt = args.sim_dt if args.sim_dt is not None else float(cfg.dt)
            h_all, V_final, v_norm = simulate_damped(
                model, xi, h_seeds,
                m_scalar=m_scalar, gamma_scalar=gamma_scalar,
                dt=dt, n_steps=args.n_sim_steps, device=device,
            )
            converged = v_norm < args.grad_eps
            n_conv = int(converged.sum())
            reg_final = 0.5 * ((((torch.from_numpy(h_all).to(device) - h_center)
                                 / h_std) ** 2).sum(dim=-1)
                               ).cpu().numpy() * args.lambda_reg
            print(f"  converged (|v|<{args.grad_eps}): {n_conv}/{N}  "
                  f"<V|final>={float(V_final.mean()):+.3f}  "
                  f"<|h|>={np.linalg.norm(h_all, axis=-1).mean():.2f}")

        if args.mode == "dynamical" and int(converged.sum()) < args.K_min + 1:
            print(f"  dynamics not fully settled; clustering on all final h")
            h_conv = h_all
            V_conv = V_final
        else:
            h_conv = h_all[converged]
            V_conv = V_final[converged]
        if len(h_conv) < args.K_min + 1:
            print(f"  !! too few points, skipping clustering")
            continue

        K_star, labels, centroids, sils = sweep_clusters(
            h_conv, args.K_min, args.K_max, seed=args.seed,
        )
        print(f"  K*={K_star}  silhouettes: "
              + ", ".join(f"{k}:{v:+.2f}" for k, v in sils.items()))

        decoded = decode_centroids(centroids, model, tokenizer, top_k=15)
        for k, toks in enumerate(decoded):
            top5 = ", ".join(f"{t}({p:.2f})" for t, p in toks[:5])
            size = int((labels == k).sum())
            V_k  = float(V_conv[labels == k].mean())
            print(f"    A{k}  (n={size:4d}, <V>={V_k:+.3f}): {top5}")

        fig_path = RESULTS_DIR / f"attractors_{args.tag}_{prompt_name}.png"
        plot_attractors(prompt_name, prompt_text, h_conv, labels,
                        centroids, decoded,
                        h_real.cpu().numpy(), real_decoded, K_star, fig_path)
        print(f"  saved -> {fig_path}")

        all_results.append({
            "prompt_name": prompt_name,
            "prompt_text": prompt_text,
            "K_star": int(K_star),
            "silhouettes": sils,
            "n_converged": int(converged.sum()),
            "n_total": int(N),
            "attractors": [
                {
                    "id": int(k),
                    "size": int((labels == k).sum()),
                    "V_mean": float(V_conv[labels == k].mean()),
                    "top_tokens": decoded[k],
                }
                for k in range(K_star)
            ],
            "real_top_tokens": real_decoded,
        })

    out_json = RESULTS_DIR / f"attractors_{args.tag}_results.json"
    with out_json.open("w") as f:
        json.dump({
            "tag": args.tag,
            "ckpt": str(args.ckpt),
            "variant": variant,
            "model_cfg": {k: v for k, v in asdict(cfg).items()
                          if not isinstance(v, torch.Tensor)},
            "args": {k: getattr(args, k) for k in
                     ("steps", "lr", "n_gauss", "n_tok", "n_real",
                      "K_min", "K_max", "grad_eps", "seed")},
            "prompts": all_results,
        }, f, indent=2, default=str)
    print(f"\n[attractors] saved -> {out_json}")

    md = RESULTS_DIR / f"attractors_{args.tag}_summary.md"
    write_markdown(md, args.tag, variant, cfg, args, all_results)
    print(f"[attractors] saved -> {md}")


def write_markdown(path: Path, tag: str, variant: str, cfg, args, all_results):
    lines: List[str] = []
    lines.append(f"# Semantic-attractor extraction -- {tag}\n")
    lines.append(f"- Variant: `{variant}`")
    lines.append(f"- Model config: `d={cfg.d}, L={cfg.L}, "
                 f"mass_mode={cfg.mass_mode}`")
    lines.append(f"- Seeds: {args.n_gauss} Gaussian + {args.n_tok} token-"
                 f"embedding + {args.n_real} perturbed real-$h$")
    lines.append(f"- Descent: Adam lr={args.lr}, {args.steps} steps")
    lines.append(f"- Convergence filter: $\\|\\nabla V\\| < {args.grad_eps}$")
    lines.append(f"- Clustering: K-means, silhouette-sweep K $\\in$ "
                 f"[{args.K_min}, {args.K_max}]\n")
    for r in all_results:
        lines.append(f"## Prompt ({r['prompt_name']}): *\"{r['prompt_text']}\"*\n")
        lines.append(f"- Converged: {r['n_converged']}/{r['n_total']}")
        lines.append(f"- $K^\\ast = {r['K_star']}$   "
                     f"silhouette scores: "
                     + ", ".join(f"K={k}: {v:+.3f}"
                                 for k, v in r['silhouettes'].items())
                     + "\n")
        real = r["real_top_tokens"][:10]
        real_str = ", ".join(f"`{t}`·{p:.2f}" for t, p in real)
        lines.append(f"**Real next-token (tied LM head on $h_L$ of prompt):**  "
                     f"{real_str}\n")
        lines.append("| Attractor | Size | $\\langle V\\rangle$ | "
                     "Top-10 decoded tokens |")
        lines.append("|---|---:|---:|---|")
        for a in r["attractors"]:
            top10 = a["top_tokens"][:10]
            top_str = ", ".join(f"`{t}`·{p:.2f}" for t, p in top10)
            lines.append(f"| A{a['id']} | {a['size']} | {a['V_mean']:+.3f} | {top_str} |")
        lines.append("")
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
