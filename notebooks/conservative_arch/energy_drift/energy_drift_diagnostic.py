"""Energy-drift diagnostic: compare H_l vs l across SPLM variants.

Reads one or more ``.npz`` files produced by ``extract_energy_states.py``
and produces:

  - An overlay plot of total energy H_l vs normalised layer index l/L,
    one line per variant, error band = std across sentences.
  - The same for kinetic and potential energy separately.
  - A per-variant table with the linear drift slope (per layer),
    95% CI, and oscillation bandwidth.
  - A markdown report tying the plots and table together.

Usage::

    python3 energy_drift_diagnostic.py \\
        --inputs splm_euler_L8.npz,splm_sarfmass_L8.npz,splm_verlet_L16_dt05.npz \\
        --tag E3_splm_compare

Inputs are resolved relative to ``results/``; absolute paths are also
accepted.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
REPO_ROOT = SCRIPT_DIR.parent.parent.parent


@dataclass
class VariantData:
    label: str
    variant: str
    L: int
    H: np.ndarray            # (n_sent, L+1)
    kinetic: np.ndarray      # (n_sent, L+1)
    potential: np.ndarray    # (n_sent, L+1)
    meta: Dict


def _load_npz(path: Path) -> VariantData:
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    return VariantData(
        label=meta["label"], variant=meta["variant"], L=int(meta["L"]),
        H=z["H"], kinetic=z["kinetic"], potential=z["potential"], meta=meta,
    )


def _resolve_input_path(s: str) -> Path:
    p = Path(s)
    if p.exists():
        return p
    candidate = RESULTS_DIR / s
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"could not resolve input path: {s}")


def _linear_fit_with_ci(x: np.ndarray, y: np.ndarray, conf: float = 0.95
                        ) -> Tuple[float, float, float]:
    """OLS slope of y vs x. Returns ``(slope, intercept, slope_ci_half)``.

    ``y`` may have multiple sentences per layer; pass the flattened
    cross-product so the standard errors reflect per-sentence variation.
    """
    n = x.size
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_mean, y_mean = x.mean(), y.mean()
    sxx = float(((x - x_mean) ** 2).sum())
    if sxx == 0.0:
        return 0.0, float(y_mean), 0.0
    sxy = float(((x - x_mean) * (y - y_mean)).sum())
    slope = sxy / sxx
    intercept = float(y_mean - slope * x_mean)
    resid = y - (slope * x + intercept)
    s2 = float((resid ** 2).sum() / max(n - 2, 1))
    se_slope = math.sqrt(s2 / sxx) if sxx > 0 else float("nan")
    try:
        from scipy.stats import t as student_t
        crit = float(student_t.ppf(0.5 + conf / 2.0, n - 2))
    except Exception:
        crit = 1.96
    return float(slope), intercept, float(crit * se_slope)


def _detrended_bandwidth(layers: np.ndarray, H_mean: np.ndarray
                         ) -> Tuple[float, float]:
    """Subtract the OLS trend from H_mean and return ``(min, max)`` of residual."""
    slope, intercept, _ = _linear_fit_with_ci(layers, H_mean)
    if math.isnan(slope):
        return float("nan"), float("nan")
    detr = H_mean - (slope * layers + intercept)
    return float(detr.min()), float(detr.max())


def _plot_overlay(variants: List[VariantData], key: str,
                  ylabel: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
    for v in variants:
        arr = getattr(v, key)
        L = v.L
        layers = np.arange(L + 1)
        x = layers / max(L, 1)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        ax.plot(x, mean, label=f"{v.label} (L={L})", lw=2)
        ax.fill_between(x, mean - std, mean + std, alpha=0.18)
    ax.set_xlabel("normalised layer index  l / L")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _emit_markdown(tag: str, run_root: Path,
                   variants: List[VariantData],
                   plots: Dict[str, Path]) -> Path:
    md_path = run_root / f"{tag}_report.md"
    lines: List[str] = []
    lines.append(f"# Energy-drift diagnostic: `{tag}`")
    lines.append("")
    lines.append(
        "Eval-only diagnostic on existing SPLM checkpoints. "
        "Computes the per-layer Hamiltonian energy "
        r"$H_\ell = \tfrac{1}{2} m \|v_\ell\|^2 + V_\theta(\xi_\ell, h_\ell)$, "
        "fits a linear drift slope across depth, "
        "and reports the oscillation bandwidth around that trend."
    )
    lines.append("")
    lines.append("## Per-variant summary")
    lines.append("")
    lines.append("| variant | label | L | n sent | mean H | drift slope (per layer) | "
                 "95% CI half-width | detrended H min | detrended H max | "
                 "bandwidth (max-min) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    csv_lines = ["variant,label,L,n_sent,mean_H,slope_per_layer,"
                 "slope_ci_half,detrended_min,detrended_max,bandwidth"]
    for v in variants:
        n_sent = v.H.shape[0]
        H_mean = v.H.mean(axis=0)
        layers = np.arange(v.L + 1, dtype=np.float64)
        per_sent_x = np.repeat(layers, n_sent)
        per_sent_y = v.H.T.reshape(-1)
        slope, _intercept, ci = _linear_fit_with_ci(per_sent_x, per_sent_y)
        det_min, det_max = _detrended_bandwidth(layers, H_mean)
        bandwidth = det_max - det_min if not math.isnan(det_min) else float("nan")
        lines.append(
            f"| `{v.variant}` | `{v.label}` | {v.L} | {n_sent} | "
            f"{H_mean.mean():.4f} | {slope:+.4e} | {ci:.4e} | "
            f"{det_min:+.4e} | {det_max:+.4e} | {bandwidth:.4e} |"
        )
        csv_lines.append(
            f"{v.variant},{v.label},{v.L},{n_sent},{H_mean.mean():.6e},"
            f"{slope:.6e},{ci:.6e},{det_min:.6e},{det_max:.6e},{bandwidth:.6e}"
        )
    lines.append("")
    lines.append("## Overlay plots")
    lines.append("")
    for key, plot in plots.items():
        lines.append(f"### {key}")
        lines.append("")
        lines.append(f"![{key}]({plot.name})")
        lines.append("")

    lines.append("## Interpretation (manual)")
    lines.append("")
    lines.append("> **TODO (human reviewer):** Inspect the table and overlay plots.")
    lines.append(">")
    lines.append("> 1. **Drift sign.** A stable damped flow should have a "
                 "*decreasing* slope (energy dissipated by gamma). A "
                 "positive slope means the integrator is artificially "
                 "*injecting* energy, which is a numerical artefact.")
    lines.append("> 2. **Symplectic vs Euler.** Velocity-Verlet should "
                 "have smaller drift magnitude than Euler at matched flow "
                 "distance L*dt. If it does not, the discretisation "
                 "regime is fine enough that the integrator order does "
                 "not matter.")
    lines.append("> 3. **Bandwidth.** A symplectic flow at gamma > 0 "
                 "should have bounded oscillation around the trend; an "
                 "Euler flow may show a monotone trend with negligible "
                 "oscillation.")
    lines.append("")

    md_path.write_text("\n".join(lines))
    csv_path = run_root / f"{tag}_drift_table.csv"
    csv_path.write_text("\n".join(csv_lines) + "\n")
    return md_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", required=True, type=str,
                    help="Comma-separated list of .npz file paths "
                         "(absolute or relative to results/).")
    ap.add_argument("--tag", required=True, type=str,
                    help="Output subdirectory name under results/.")
    args = ap.parse_args()

    paths = [_resolve_input_path(s.strip())
             for s in args.inputs.split(",") if s.strip()]
    if not paths:
        print("[diag] no inputs provided.")
        return 1

    variants = [_load_npz(p) for p in paths]
    print(f"[diag] loaded {len(variants)} variant(s):")
    for v in variants:
        print(f"  - {v.label}  variant={v.variant}  L={v.L}  "
              f"n_sent={v.H.shape[0]}")

    run_root = RESULTS_DIR / args.tag
    run_root.mkdir(parents=True, exist_ok=True)

    plots: Dict[str, Path] = {}
    keys = [
        ("H", "H_l (total energy)", "Total energy H_l vs layer"),
        ("kinetic", "(1/2) m ||v||^2  (mean over tokens)",
         "Kinetic energy vs layer"),
        ("potential", "V_theta(xi_l, h_l)  (mean over tokens)",
         "Potential energy vs layer"),
    ]
    for key, ylabel, title in keys:
        out = run_root / f"{args.tag}_{key}_overlay.png"
        _plot_overlay(variants, key, ylabel, title, out)
        plots[key] = out
        print(f"  [plot] {out.relative_to(REPO_ROOT)}")

    md = _emit_markdown(args.tag, run_root, variants, plots)
    print(f"[diag] report -> {md.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
