"""
Side-by-side 3D comparison of V_theta landscapes for Euler vs Verlet on
the same prompt.  Re-uses the per-model rendering but places them on one
figure so the Euler (wide valley, many basins) vs Verlet (narrow funnel,
few basins) story is a single panel.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).parent
PARENT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results"
sys.path.insert(0, str(PARENT_DIR))

from landscape_3d import (  # noqa: E402
    PROMPTS, load_model, build_seeds, get_xi_and_traj,
    simulate_damped_with_traj, grid_landscape, cluster_endpoints,
    make_pca, _draw_scene,
)


def render_pair(ckpts, tags, labels, prompt_name, prompt_text,
                device, n_sim_overrides, dt_overrides,
                grid_n=56, view=(28, -55),
                n_gauss=96, n_tok=96, n_real=96,
                out_path: Path | None = None,
                seed: int = 0) -> None:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    fig = plt.figure(figsize=(7.0 * len(ckpts), 7.6))
    for col, (ckpt, tag, label) in enumerate(zip(ckpts, tags, labels)):
        model, cfg, variant = load_model(ckpt, device)
        m_scalar = float(F.softplus(model.raw_m_bias).detach().cpu().item()
                         + 1e-3)
        gamma_scalar = float(model.gamma.detach().cpu().item())
        n_sim = n_sim_overrides.get(tag, int(cfg.L))
        dt = dt_overrides.get(tag, float(cfg.dt))
        print(f"[compare] {tag}: m={m_scalar:.3f} gamma={gamma_scalar:.3f} "
              f"dt={dt} n_sim={n_sim}")

        h_seeds = build_seeds(model, tokenizer, device, d=cfg.d,
                              n_gauss=n_gauss, n_tok=n_tok,
                              n_real=n_real, seed=seed)
        xi, real_traj, h_real, _ = get_xi_and_traj(
            model, tokenizer, prompt_text, device,
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
        px = (x_max - x_min) * 0.25
        py = (y_max - y_min) * 0.25
        x_lim = (x_min - px, x_max + px)
        y_lim = (y_min - py, y_max + py)

        V_all = V_traj.ravel()
        clip_low = float(np.percentile(V_all, 2))
        clip_high = float(np.percentile(V_traj[:, 0], 95))
        clip_low = min(clip_low, clip_high - 50.0)

        X, Y, Z = grid_landscape(model, xi, pca, x_lim, y_lim,
                                 grid_n=grid_n, device=device,
                                 clip_z=(clip_low, clip_high))

        endpoints_d = h_traj[:, -1, :]
        K, labels_arr = cluster_endpoints(endpoints_d, 2, 10, seed=seed)

        traj_xy = np.stack([
            pca.transform(h_traj[:, t, :])
            for t in range(h_traj.shape[1])
        ], axis=1)
        traj_V_clipped = np.clip(V_traj, clip_low, clip_high)
        real_traj_xy = pca.transform(real_traj_np)
        with torch.no_grad():
            real_V = (model.V_theta(
                xi.unsqueeze(0).expand(real_traj.shape[0], -1),
                real_traj).detach().cpu().numpy().ravel())
        real_traj_V = np.clip(real_V, clip_low, clip_high)

        ax = fig.add_subplot(1, len(ckpts), col + 1, projection="3d")
        _draw_scene(ax, X, Y, Z, traj_xy, traj_V_clipped, labels_arr, K,
                    real_traj_xy, real_traj_V,
                    real_traj_xy[-1], real_traj_V[-1],
                    highlight_frac=0.08)
        ax.view_init(elev=view[0], azim=view[1])
        N, Sp1 = traj_xy.shape[0], traj_xy.shape[1]
        ax.set_title(
            f"{label}\n"
            f"$K^\\ast={K}$ basins  |  {N} seeds  |  {Sp1-1} steps",
            fontsize=10,
        )
        if col == 0:
            ax.legend(loc="upper left", fontsize=8, framealpha=0.8)

    fig.suptitle(
        f"$V_\\theta$ landscape:  Euler vs Verlet  --  "
        f'prompt ({prompt_name}): "{prompt_text}"',
        fontsize=11, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = out_path or (RESULTS_DIR /
                       f"landscape3d_compare_{prompt_name}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="all",
                    help="Comma-separated prompt names or 'all'")
    ap.add_argument("--device", default=None)
    ap.add_argument("--grid_n", type=int, default=56)
    args = ap.parse_args()

    device = args.device or (
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    CKPTS = [
        "../sarf_mass_variant/results/splm_sarfmass_logfreq_shakespeare_ckpt_latest.pt",
        "../symplectic_variant/results/splm_sym_logfreq_shakespeare_L16_dt05_ckpt_latest.pt",
    ]
    TAGS = ["euler_L8", "verlet_L16_dt05"]
    LABELS = ["Euler L=8, $\\Delta t$=1.0", "Verlet L=16, $\\Delta t$=0.5"]

    requested = (args.prompts
                 if args.prompts == "all"
                 else set(args.prompts.split(",")))
    for pname, ptext in PROMPTS:
        if requested != "all" and pname not in requested:
            continue
        print(f"\n=== Compare landscapes: prompt ({pname}) ===")
        render_pair(
            ckpts=CKPTS, tags=TAGS, labels=LABELS,
            prompt_name=pname, prompt_text=ptext,
            device=device,
            n_sim_overrides={}, dt_overrides={},
            grid_n=args.grid_n,
        )


if __name__ == "__main__":
    main()
