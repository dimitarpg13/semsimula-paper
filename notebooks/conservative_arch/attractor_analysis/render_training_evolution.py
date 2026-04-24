"""
Render the V_theta landscape at each training snapshot to show how the
attractor topology forms during training.

Consumes snapshots saved by train_with_snapshots.py under
  results/snapshots/<tag>/ckpt_step<step>.pt

Produces results/training_evolution_<prompt>.png (a grid of 3D panels,
one per snapshot, all for the same prompt).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

SCRIPT_DIR = Path(__file__).parent
PARENT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results"
SNAP_ROOT = RESULTS_DIR / "snapshots"
sys.path.insert(0, str(PARENT_DIR))

from landscape_3d import (  # noqa: E402
    PROMPTS, load_model, build_seeds, get_xi_and_traj,
    simulate_damped_with_traj, grid_landscape, cluster_endpoints,
    _draw_scene,
)


def list_snapshots(snap_dir: Path) -> List[Tuple[int, Path]]:
    paths = sorted(snap_dir.glob("ckpt_step*.pt"))
    return [(int(p.stem.replace("ckpt_step", "")), p) for p in paths]


def build_shared_pca(snap_models, tokenizer, prompt, device, h_seeds):
    """Fit a PCA on the union of trajectories from all snapshots so that
    every panel uses the same 2D plane.  That makes the panels visually
    comparable."""
    d = h_seeds.shape[1]
    all_h = []
    per_snap_cache = []
    for model, cfg in snap_models:
        m_scalar = float(F.softplus(model.raw_m_bias).detach().cpu().item()
                         + 1e-3)
        gamma_scalar = float(model.gamma.detach().cpu().item())
        xi, real_traj, _, _ = get_xi_and_traj(model, tokenizer, prompt, device)
        h_traj, V_traj = simulate_damped_with_traj(
            model, xi, h_seeds, m_scalar, gamma_scalar,
            float(cfg.dt), int(cfg.L), device,
        )
        all_h.append(real_traj.detach().cpu().numpy())
        all_h.append(h_traj.reshape(-1, d))
        per_snap_cache.append((xi, real_traj, h_traj, V_traj))
    big = np.concatenate(all_h, axis=0).astype(np.float32)
    pca = PCA(n_components=2).fit(big)
    return pca, per_snap_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap_tag", default="euler_shakespeare")
    ap.add_argument("--prompt", default="narrative")
    ap.add_argument("--grid_n", type=int, default=56)
    ap.add_argument("--n_gauss", type=int, default=96)
    ap.add_argument("--n_tok",   type=int, default=96)
    ap.add_argument("--n_real",  type=int, default=96)
    ap.add_argument("--device",  default=None)
    ap.add_argument("--elev",    type=int, default=26)
    ap.add_argument("--azim",    type=int, default=-55)
    ap.add_argument("--shared_pca", action="store_true",
                    help="Use one PCA fit on all snapshots' trajectories "
                         "(makes panels visually comparable but may blur "
                         "detail).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = args.device or (
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"[train-evo] device={device}  tag={args.snap_tag}  "
          f"prompt={args.prompt}")

    snap_dir = SNAP_ROOT / args.snap_tag
    snaps = list_snapshots(snap_dir)
    if not snaps:
        raise SystemExit(f"No snapshots found in {snap_dir}")
    print(f"[train-evo] {len(snaps)} snapshots: "
          + ", ".join(f"step{s}" for s, _ in snaps))

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    prompt_text = dict(PROMPTS)[args.prompt]

    print("[train-evo] loading all snapshots into memory...")
    snap_models = []
    val_losses = []
    for step, path in snaps:
        model, cfg, variant = load_model(str(path), device)
        ck = torch.load(path, map_location=device, weights_only=False)
        val_loss = ck.get("val_loss")
        val_losses.append(val_loss)
        snap_models.append((model, cfg))

    first_cfg = snap_models[0][1]
    h_seeds = build_seeds(snap_models[0][0], tokenizer, device,
                          d=first_cfg.d,
                          n_gauss=args.n_gauss, n_tok=args.n_tok,
                          n_real=args.n_real, seed=args.seed)

    if args.shared_pca:
        print("[train-evo] fitting shared PCA on all snapshots...")
        pca_shared, per_snap_cache = build_shared_pca(
            snap_models, tokenizer, prompt_text, device, h_seeds,
        )
    else:
        pca_shared = None
        per_snap_cache = []
        for model, cfg in snap_models:
            m_scalar = float(F.softplus(model.raw_m_bias).detach().cpu().item()
                             + 1e-3)
            gamma_scalar = float(model.gamma.detach().cpu().item())
            xi, real_traj, _, _ = get_xi_and_traj(
                model, tokenizer, prompt_text, device,
            )
            h_traj, V_traj = simulate_damped_with_traj(
                model, xi, h_seeds, m_scalar, gamma_scalar,
                float(cfg.dt), int(cfg.L), device,
            )
            per_snap_cache.append((xi, real_traj, h_traj, V_traj))

    n_panels = len(snaps)
    cols = min(n_panels, 4)
    rows = (n_panels + cols - 1) // cols
    fig = plt.figure(figsize=(5.3 * cols, 5.1 * rows))

    for i, ((step, _), (model, cfg), cache, vloss) in enumerate(
        zip(snaps, snap_models, per_snap_cache, val_losses)
    ):
        xi, real_traj, h_traj, V_traj = cache
        real_traj_np = real_traj.detach().cpu().numpy()

        if pca_shared is not None:
            pca = pca_shared
        else:
            pca = PCA(n_components=2).fit(
                np.concatenate([real_traj_np,
                                h_traj.reshape(-1, cfg.d)], axis=0
                               ).astype(np.float32)
            )

        flat_2d = np.concatenate([
            pca.transform(real_traj_np),
            pca.transform(h_traj.reshape(-1, cfg.d)),
        ], axis=0)
        x_min, y_min = flat_2d.min(0); x_max, y_max = flat_2d.max(0)
        px = (x_max - x_min) * 0.25; py = (y_max - y_min) * 0.25
        x_lim = (x_min - px, x_max + px); y_lim = (y_min - py, y_max + py)

        V_all = V_traj.ravel()
        clip_low = float(np.percentile(V_all, 2))
        clip_high = float(np.percentile(V_traj[:, 0], 95))
        clip_low = min(clip_low, clip_high - 50.0)

        X, Y, Z = grid_landscape(model, xi, pca, x_lim, y_lim,
                                 grid_n=args.grid_n, device=device,
                                 clip_z=(clip_low, clip_high))

        endpoints = h_traj[:, -1, :]
        K, labels = cluster_endpoints(endpoints, 2, 10, seed=args.seed)

        traj_xy = np.stack([pca.transform(h_traj[:, t, :])
                            for t in range(h_traj.shape[1])], axis=1)
        traj_V_clip = np.clip(V_traj, clip_low, clip_high)
        real_xy = pca.transform(real_traj_np)
        with torch.no_grad():
            real_V_raw = (model.V_theta(
                xi.unsqueeze(0).expand(real_traj.shape[0], -1),
                real_traj).detach().cpu().numpy().ravel())
        real_V = np.clip(real_V_raw, clip_low, clip_high)

        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        _draw_scene(ax, X, Y, Z, traj_xy, traj_V_clip, labels, K,
                    real_xy, real_V, real_xy[-1], real_V[-1],
                    highlight_frac=0.08)
        ax.view_init(elev=args.elev, azim=args.azim)
        title = f"step {step}"
        if vloss is not None:
            title += f"   (val {vloss:.2f})"
        title += f"   K*={K}"
        ax.set_title(title, fontsize=10)
        print(f"[train-evo] step {step}: K*={K}  val_loss={vloss}  "
              f"V_clip=[{clip_low:+.1f}, {clip_high:+.1f}]")

    fig.suptitle(
        f"$V_\\theta$ landscape evolution during training   "
        f"(SARF-mass logfreq, Tiny Shakespeare)\n"
        f'prompt ({args.prompt}): "{prompt_text}"   |   '
        f"{args.n_gauss + args.n_tok + args.n_real} seeds   |   "
        f"{'shared' if args.shared_pca else 'per-snapshot'} PCA",
        fontsize=11, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    suffix = "_sharedPCA" if args.shared_pca else ""
    out = (RESULTS_DIR /
           f"training_evolution_{args.snap_tag}_{args.prompt}{suffix}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
