"""
3D visualisation of V_theta(xi, h) with damped-dynamics trajectories.

For a trained SPLM and a chosen prompt:
  1. Run the damped integrator from N random h seeds at fixed xi,
     saving the full (N, S+1, d) trajectory tensor.
  2. Fit a 2-component PCA on the union of {real trajectory, damped
     trajectories at every step}.
  3. Sample a grid on the PCA plane, lift back to d-dim, evaluate
     V_theta there -> the landscape surface Z = V(xi, h).
  4. Overlay the projected trajectories (PCA-space x,y, and V on the
     surface for z).  Colour each trajectory by the basin its endpoint
     lands in (K-means on endpoints).
  5. Save a static PNG at a nice camera angle, plus a rotating GIF.

Usage:
  python3 landscape_3d.py \
      --ckpt ../sarf_mass_variant/results/splm_sarfmass_logfreq_shakespeare_ckpt_latest.pt \
      --tag euler_L8_dt1 --n_sim_steps 8 --sim_dt 1.0 \
      --prompt narrative --save_anim
"""

from __future__ import annotations

import argparse
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
from matplotlib import cm
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (side-effect import)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

SCRIPT_DIR = Path(__file__).parent
PARENT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(PARENT_DIR))

# Import prompts from the main script so the two stay in sync.
from attractor_extraction import PROMPTS, load_model  # noqa: E402


# ---------------------------------------------------------------------------
def get_xi_and_traj(model, tokenizer, prompt: str, device: str):
    """Returns (xi, full real trajectory (L+1, d), real h_L, token ids)."""
    ids = torch.tensor(tokenizer.encode(prompt), device=device,
                       dtype=torch.long).unsqueeze(0)
    with torch.enable_grad():
        _, _, traj_h, traj_xi = model(ids, return_trajectory=True,
                                      return_xi_trajectory=True)
    xi_last = traj_xi[-1][0, -1, :].detach().to(device)
    real_traj = torch.stack([t[0, -1, :] for t in traj_h], dim=0)   # (L+1, d)
    h_real = real_traj[-1].detach().to(device)
    return xi_last, real_traj.detach().to(device), h_real, ids


def build_seeds(model, tokenizer, device, d, n_gauss, n_tok, n_real,
                real_noise=0.3, seed=0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    h_real_list = []
    for _, p in PROMPTS:
        ids = torch.tensor(tokenizer.encode(p), device=device,
                           dtype=torch.long).unsqueeze(0)
        with torch.enable_grad():
            _, _, traj_h, _ = model(ids, return_trajectory=True,
                                    return_xi_trajectory=True)
        h_real_list.append(traj_h[-1][0].to(device))
    H_real = torch.cat(h_real_list, dim=0)
    h_mean = H_real.mean(0)
    h_std = H_real.std(0).clamp_min(1e-3)

    g_noise = torch.randn(n_gauss, d, generator=gen).to(device)
    h0_gauss = h_mean + g_noise * h_std

    V_tok = model.E.weight.shape[0]
    idx = torch.randint(0, V_tok, (n_tok,), generator=gen)
    h0_tok = model.E.weight.detach()[idx].to(device)

    idx_real = torch.randint(0, H_real.shape[0], (n_real,), generator=gen)
    noise = torch.randn(n_real, d, generator=gen).to(device) * real_noise
    h0_real = H_real[idx_real] + noise * h_std

    return torch.cat([h0_gauss, h0_tok, h0_real], dim=0)


# ---------------------------------------------------------------------------
def simulate_damped_with_traj(
    model, xi, h_seeds, m, gamma, dt, n_steps, device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (h_traj, V_traj) of shapes (N, S+1, d) and (N, S+1)."""
    h = h_seeds.clone().detach().to(device)
    v = torch.zeros_like(h)
    xi_b = xi.unsqueeze(0).expand(h.shape[0], -1).to(device)
    h_traj = [h.detach().cpu().numpy()]
    with torch.no_grad():
        V0 = model.V_theta(xi_b, h).detach().cpu().numpy().ravel()
    V_traj = [V0]
    for step in range(n_steps):
        h_rg = h.detach().clone().requires_grad_(True)
        V = model.V_theta(xi_b, h_rg).sum()
        g, = torch.autograd.grad(V, h_rg)
        f = -g.detach()
        v = (v + dt * f / m) / (1.0 + dt * gamma)
        h = h.detach() + dt * v
        h_traj.append(h.detach().cpu().numpy())
        with torch.no_grad():
            Vs = model.V_theta(xi_b, h).detach().cpu().numpy().ravel()
        V_traj.append(Vs)
    return (np.stack(h_traj, axis=1),            # (N, S+1, d)
            np.stack(V_traj, axis=1))             # (N, S+1)


# ---------------------------------------------------------------------------
def grid_landscape(model, xi, pca, x_lim, y_lim, grid_n, device,
                   clip_z=None):
    x = np.linspace(x_lim[0], x_lim[1], grid_n)
    y = np.linspace(y_lim[0], y_lim[1], grid_n)
    X, Y = np.meshgrid(x, y)
    pts_2d = np.stack([X.ravel(), Y.ravel()], axis=1)
    pts_d = pca.inverse_transform(pts_2d).astype(np.float32)
    pts_d_t = torch.from_numpy(pts_d).to(device)
    xi_b = xi.unsqueeze(0).expand(pts_d.shape[0], -1).to(device)
    with torch.no_grad():
        V = model.V_theta(xi_b, pts_d_t).detach().cpu().numpy().ravel()
    Z = V.reshape(X.shape)
    if clip_z is not None:
        Z = np.clip(Z, clip_z[0], clip_z[1])
    return X, Y, Z


# ---------------------------------------------------------------------------
def cluster_endpoints(endpoints_d: np.ndarray, K_min=2, K_max=10, seed=0):
    best_K, best_sil, best_labels = K_min, -np.inf, None
    for K in range(K_min, min(K_max + 1, len(endpoints_d))):
        km = KMeans(n_clusters=K, random_state=seed, n_init=10
                    ).fit(endpoints_d)
        try:
            sil = silhouette_score(endpoints_d, km.labels_)
        except Exception:
            sil = -np.inf
        if sil > best_sil:
            best_sil, best_K, best_labels = sil, K, km.labels_
    return best_K, best_labels


# ---------------------------------------------------------------------------
def _draw_scene(
    ax, X, Y, Z, traj_xy, traj_V, endpoints_labels, K,
    real_traj_xy, real_traj_V, real_xy, real_V,
    highlight_frac=0.08,
) -> None:
    """Shared drawing code for a single 3D view."""
    ax.plot_surface(
        X, Y, Z, cmap="viridis",
        edgecolor="none", alpha=0.45, rstride=1, cstride=1,
        antialiased=True, linewidth=0,
    )

    N, Sp1 = traj_xy.shape[0], traj_xy.shape[1]
    cmap = plt.colormaps.get_cmap("tab10").resampled(max(K, 3))
    rng = np.random.default_rng(0)
    n_highlight = max(6, int(N * highlight_frac))
    highlight_idx = set(rng.choice(N, size=n_highlight, replace=False).tolist())

    for i in range(N):
        if i in highlight_idx:
            continue
        c = cmap(endpoints_labels[i])
        ax.plot(traj_xy[i, :, 0], traj_xy[i, :, 1], traj_V[i, :],
                color=c, alpha=0.08, linewidth=0.5, zorder=2)

    for i in highlight_idx:
        c = cmap(endpoints_labels[i])
        ax.plot(traj_xy[i, :, 0], traj_xy[i, :, 1], traj_V[i, :],
                color=c, alpha=0.9, linewidth=1.8, zorder=5)
        ax.scatter(traj_xy[i, 0, 0], traj_xy[i, 0, 1], traj_V[i, 0],
                   s=22, color=c, edgecolors="white",
                   linewidths=0.5, alpha=0.95, zorder=6)
        ax.scatter(traj_xy[i, -1, 0], traj_xy[i, -1, 1], traj_V[i, -1],
                   s=120, color=c, edgecolors="black", marker="X",
                   linewidths=0.8, zorder=7)

    ax.plot(real_traj_xy[:, 0], real_traj_xy[:, 1], real_traj_V,
            color="red", linewidth=3.0, alpha=0.95,
            label="real SPLM trajectory", zorder=10)
    ax.scatter(real_xy[0], real_xy[1], real_V,
               s=220, facecolors="none", edgecolors="red",
               linewidths=3.0, zorder=11, label=r"real $h_L$")
    ax.scatter(real_traj_xy[0, 0], real_traj_xy[0, 1], real_traj_V[0],
               s=140, color="red", marker="o", alpha=0.6,
               edgecolors="black", linewidths=0.8, zorder=11,
               label=r"real $h_0$ (embed)")

    ax.set_xlabel("PC1", labelpad=5)
    ax.set_ylabel("PC2", labelpad=5)
    ax.set_zlabel(r"$V_\theta$", labelpad=5)
    ax.grid(True, alpha=0.15)


def plot_landscape_3d(
    X, Y, Z, traj_xy, traj_V, endpoints_labels, K,
    real_traj_xy, real_traj_V, real_xy, real_V,
    prompt_name, prompt_text, model_label,
    out_path: Path,
    views=((28, -55), (20, -90), (60, -90)),
    highlight_frac=0.08,
) -> None:
    """One figure with several camera views of the same scene."""
    n_views = len(views)
    fig = plt.figure(figsize=(6.5 * n_views, 6.8))
    for i, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, n_views, i + 1, projection="3d")
        _draw_scene(ax, X, Y, Z, traj_xy, traj_V, endpoints_labels, K,
                    real_traj_xy, real_traj_V, real_xy, real_V,
                    highlight_frac=highlight_frac)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(
            f"view {i+1}: elev={elev}, azim={azim}",
            fontsize=9,
        )
        if i == 0:
            ax.legend(loc="upper left", fontsize=8, framealpha=0.8)

    N, Sp1 = traj_xy.shape[0], traj_xy.shape[1]
    fig.suptitle(
        f"$V_\\theta$ landscape + damped-flow trajectories   --   "
        f"{model_label}\n"
        f'prompt ({prompt_name}): "{prompt_text}"   |   '
        f"$K^\\ast={K}$ basins   |   {N} seeds   |   {Sp1-1} integration steps",
        fontsize=11, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")


def save_rotating_animation(
    X, Y, Z, traj_xy, traj_V, endpoints_labels, K,
    real_traj_xy, real_traj_V, real_xy, real_V,
    prompt_name, prompt_text, model_label,
    out_path: Path, n_frames: int = 90, elev=25, highlight_frac=0.05,
) -> None:
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    def init():
        ax.cla()
        ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none",
                        alpha=0.55, rstride=2, cstride=2,
                        antialiased=True, linewidth=0)
        N, Sp1 = traj_xy.shape[0], traj_xy.shape[1]
        cmap = cm.get_cmap("tab10", K)
        rng = np.random.default_rng(0)
        highlight_idx = rng.choice(N,
                                   size=max(5, int(N * highlight_frac)),
                                   replace=False)
        for i in highlight_idx:
            c = cmap(endpoints_labels[i])
            ax.plot(traj_xy[i, :, 0], traj_xy[i, :, 1], traj_V[i, :],
                    color=c, alpha=0.85, linewidth=1.3)
            ax.scatter(traj_xy[i, -1, 0], traj_xy[i, -1, 1],
                       traj_V[i, -1], s=45, color=c,
                       edgecolors="black", marker="X", linewidths=0.5)
        ax.plot(real_traj_xy[:, 0], real_traj_xy[:, 1], real_traj_V,
                color="red", linewidth=2.5, alpha=0.95)
        ax.scatter(real_xy[0], real_xy[1], real_V, s=160,
                   facecolors="none", edgecolors="red", linewidths=2.5)
        ax.set_xlabel("PC1", labelpad=6)
        ax.set_ylabel("PC2", labelpad=6)
        ax.set_zlabel(r"$V_\theta$", labelpad=6)
        ax.grid(True, alpha=0.2)
        ax.set_title(f"{model_label}   --   {prompt_name}: "
                     f'"{prompt_text}"',
                     fontsize=10)
        return []

    def frame(i):
        ax.view_init(elev=elev, azim=-60 + i * (360.0 / n_frames))
        return []

    init()
    anim = FuncAnimation(fig, frame, frames=n_frames,
                         init_func=lambda: [], blit=False, interval=60)
    writer = PillowWriter(fps=20)
    anim.save(out_path, writer=writer, dpi=110)
    plt.close(fig)
    print(f"  saved -> {out_path}")


# ---------------------------------------------------------------------------
def make_pca(real_traj_d, traj_h, n_comp=2):
    N, Sp1, d = traj_h.shape
    X = np.concatenate([
        real_traj_d,                         # (L+1, d)
        traj_h.reshape(-1, d),               # (N*(S+1), d)
    ], axis=0).astype(np.float32)
    return PCA(n_components=n_comp).fit(X)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prompts", default="all",
                    help="Comma-separated prompt names or 'all'.")
    ap.add_argument("--n_sim_steps", type=int, default=None,
                    help="Defaults to cfg.L (training integration depth).")
    ap.add_argument("--sim_dt", type=float, default=None,
                    help="Defaults to cfg.dt.")
    ap.add_argument("--n_gauss", type=int, default=96)
    ap.add_argument("--n_tok",   type=int, default=96)
    ap.add_argument("--n_real",  type=int, default=96)
    ap.add_argument("--grid_n",  type=int, default=60)
    ap.add_argument("--clip_z_low",  type=float, default=None)
    ap.add_argument("--clip_z_high", type=float, default=None)
    ap.add_argument("--K_min", type=int, default=2)
    ap.add_argument("--K_max", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_anim", action="store_true")
    ap.add_argument("--pca_pad", type=float, default=0.25,
                    help="Fractional padding around data bbox in PCA plane.")
    args = ap.parse_args()

    device = args.device or (
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"[landscape3d] device={device}  ckpt={args.ckpt}  tag={args.tag}")
    model, cfg, variant = load_model(args.ckpt, device)
    print(f"[landscape3d] variant={variant}  d={cfg.d}  L={cfg.L}  "
          f"dt={cfg.dt}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    m_scalar = float(F.softplus(model.raw_m_bias).detach().cpu().item() + 1e-3)
    gamma_scalar = float(model.gamma.detach().cpu().item())
    dt = args.sim_dt if args.sim_dt is not None else float(cfg.dt)
    n_sim = args.n_sim_steps if args.n_sim_steps is not None else int(cfg.L)
    print(f"[landscape3d] m={m_scalar:.3f}  gamma={gamma_scalar:.3f}  "
          f"dt={dt}  n_sim={n_sim}")

    h_seeds = build_seeds(model, tokenizer, device, d=cfg.d,
                          n_gauss=args.n_gauss, n_tok=args.n_tok,
                          n_real=args.n_real, seed=args.seed)

    requested = (args.prompts
                 if args.prompts == "all"
                 else args.prompts.split(","))
    for pname, ptext in PROMPTS:
        if requested != "all" and pname not in requested:
            continue
        print(f"\n[landscape3d] === prompt ({pname}): '{ptext}' ===")
        xi, real_traj, h_real, ids = get_xi_and_traj(
            model, tokenizer, ptext, device,
        )

        h_traj, V_traj = simulate_damped_with_traj(
            model, xi, h_seeds, m_scalar, gamma_scalar, dt, n_sim, device,
        )

        real_traj_np = real_traj.detach().cpu().numpy()
        pca = make_pca(real_traj_np, h_traj, n_comp=2)

        flat_2d = np.concatenate([
            pca.transform(real_traj_np),
            pca.transform(h_traj.reshape(-1, cfg.d)),
        ], axis=0)
        x_min, y_min = flat_2d.min(axis=0)
        x_max, y_max = flat_2d.max(axis=0)
        pad_x = (x_max - x_min) * args.pca_pad
        pad_y = (y_max - y_min) * args.pca_pad
        x_lim = (x_min - pad_x, x_max + pad_x)
        y_lim = (y_min - pad_y, y_max + pad_y)

        endpoints_d = h_traj[:, -1, :]
        K, labels = cluster_endpoints(endpoints_d,
                                      args.K_min, args.K_max, args.seed)
        print(f"  K*={K}  endpoints clustered.")

        V_data = V_traj.ravel()
        # Pull clip_low from low percentile of the trajectory V (so the
        # basin floor keeps detail) and clip_high just above the h_0 /
        # seed band (so the off-manifold ridges are flat-capped and
        # don't dominate the view).
        clip_low = (args.clip_z_low
                    if args.clip_z_low is not None
                    else float(np.percentile(V_data, 2)))
        clip_high = (args.clip_z_high
                     if args.clip_z_high is not None
                     else float(np.percentile(V_traj[:, 0], 95)))
        clip_low = min(clip_low, clip_high - 50.0)
        print(f"  V range on trajectories: [{V_data.min():+.1f}, "
              f"{V_data.max():+.1f}]  clip=[{clip_low:+.1f}, {clip_high:+.1f}]")

        X, Y, Z = grid_landscape(model, xi, pca, x_lim, y_lim,
                                 grid_n=args.grid_n, device=device,
                                 clip_z=(clip_low, clip_high))

        traj_xy = np.stack([
            pca.transform(h_traj[:, t, :])
            for t in range(h_traj.shape[1])
        ], axis=1)                                   # (N, S+1, 2)
        traj_V_clipped = np.clip(V_traj, clip_low, clip_high)

        real_traj_xy = pca.transform(real_traj_np)
        with torch.no_grad():
            real_traj_V_raw = (model.V_theta(
                xi.unsqueeze(0).expand(real_traj.shape[0], -1),
                real_traj).detach().cpu().numpy().ravel())
        real_traj_V = np.clip(real_traj_V_raw, clip_low, clip_high)

        real_xy = real_traj_xy[-1]
        real_V = real_traj_V[-1]

        out_png = RESULTS_DIR / f"landscape3d_{args.tag}_{pname}.png"
        plot_landscape_3d(
            X, Y, Z, traj_xy, traj_V_clipped, labels, K,
            real_traj_xy, real_traj_V, real_xy, real_V,
            pname, ptext,
            model_label=f"{variant} (L={cfg.L}, dt={cfg.dt})",
            out_path=out_png,
        )

        if args.save_anim:
            out_gif = RESULTS_DIR / f"landscape3d_{args.tag}_{pname}.gif"
            save_rotating_animation(
                X, Y, Z, traj_xy, traj_V_clipped, labels, K,
                real_traj_xy, real_traj_V, real_xy, real_V,
                pname, ptext,
                model_label=f"{variant} (L={cfg.L}, dt={cfg.dt})",
                out_path=out_gif,
            )


if __name__ == "__main__":
    main()
