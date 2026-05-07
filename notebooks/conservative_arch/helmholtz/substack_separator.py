"""
Substack-restricted shared-V_psi separator for the Helmholtz hybrid (Q9d).

Tests the headline prediction of the Q9d design doc §4.1:

    The strict shared-potential test, restricted to a contiguous
    run of S-blocks, attains the SPLM-substack-only R² ~ 0.90,
    dropping to the GPT-2-like middle band on the A-block segments.

For each contiguous segment of length >= 2 (so the velocity proxy
v_l = x_l - x_{l-1} and the next-step target dx_l = x_{l+1} - x_l
are both inside the segment), refits a fresh shared V_psi on the
samples drawn from that segment only.  Reports per-layer R^2 and
the segment mean.

Design doc §4.1 prediction:
  - S-block segments (contiguous): R² ~ 0.90 (SPLM-like, by construction
    if the trained V_theta is well-formed).
  - A-block segments (contiguous): R² in the GPT-2 middle band
    (~0.45-0.65), since attention does not derive from a context-free
    scalar in h alone.
  - The contrast between the two is the "step-function R²_psi" of
    the design doc.  Variant A's two-stage hybrid cannot test this
    because it has only one S-block segment and one A-block segment;
    Q9d schedules with multiple contiguous S/A segments (e.g.,
    SSAAAASS, SASASASA isolated single-block edge case) admit the
    sharper test.

Usage:
  python3 substack_separator.py \
      --traj <path>.trajectories.pkl \
      [--steps 4000] [--device mps]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).parent
PARENT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PARENT_DIR))

from shared_potential_fit import (  # noqa: E402
    fit_shared_V, fit_velocity_only, predict_shared_V,
    r2_overall, r2_per_layer,
)


# -----------------------------------------------------------------------
# Segment discovery
# -----------------------------------------------------------------------

def find_segments(block_kinds: List[str]) -> Dict[str, List[Tuple[int, int]]]:
    """Return contiguous segments per block kind as (start, end) inclusive
    in 1-indexed layer space (same indexing used by shared_potential_fit:
    layer 0 = embedding, layers 1..L = post-step states)."""
    segments: Dict[str, List[Tuple[int, int]]] = {"S": [], "A": []}
    if not block_kinds:
        return segments
    cur_kind = block_kinds[0]
    cur_start = 1
    for ell, k in enumerate(block_kinds[1:], start=2):
        if k == cur_kind:
            continue
        segments[cur_kind].append((cur_start, ell - 1))
        cur_kind = k
        cur_start = ell
    segments[cur_kind].append((cur_start, len(block_kinds)))
    return segments


def usable_layers_in_segment(seg: Tuple[int, int],
                             L: int) -> List[int]:
    """Layers ell such that ell-1 and ell+1 are also inside [seg_lo, seg_hi].

    With shared_potential_fit's `build_samples`, ell uses x_{ell-1},
    x_ell, x_{ell+1}, so we need (ell - 1) and (ell + 1) to both be
    in [seg_lo, seg_hi] for purity AND in [0, L] for index validity."""
    lo, hi = seg
    lo_eff = max(lo, 1)        # need ell - 1 >= 0 from build_samples
    hi_eff = min(hi, L - 1)    # need ell + 1 <= L from build_samples
    out: List[int] = []
    for ell in range(lo_eff, hi_eff + 1):
        if (ell - 1) >= lo and (ell + 1) <= hi:
            out.append(ell)
    return out


# -----------------------------------------------------------------------
# Sample assembly restricted to specific layer indices
# -----------------------------------------------------------------------

def build_samples_restricted(
    trajs, allowed_layers: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Same as shared_potential_fit.build_samples but only includes
    layer indices in `allowed_layers`."""
    Xs, Vs, Ys, Ls = [], [], [], []
    allowed = set(allowed_layers)
    for tr in trajs:
        for ell in allowed:
            x_prev = tr.x_ps[ell - 1]
            x      = tr.x_ps[ell]
            x_next = tr.x_ps[ell + 1]
            T = x.shape[0]
            Xs.append(x)
            Vs.append(x - x_prev)
            Ys.append(x_next - x)
            Ls.append(np.full((T,), ell, dtype=np.int64))
    if not Xs:
        return (np.zeros((0,), np.float32),) * 4
    X = np.concatenate(Xs, axis=0)
    V = np.concatenate(Vs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    LAY = np.concatenate(Ls, axis=0)
    return X.astype(np.float32), V.astype(np.float32), Y.astype(np.float32), LAY


# -----------------------------------------------------------------------
# Single-segment fit
# -----------------------------------------------------------------------

def fit_segment(
    trajs_train, trajs_test, allowed_layers: List[int],
    d: int, L: int,
    hidden: int, depth: int, steps: int, lr: float,
    device: str, seed: int,
) -> Dict:
    """Fit V_psi from scratch on samples restricted to `allowed_layers`."""
    if not allowed_layers:
        return {"layers": [], "r2_test": {}, "r2_train": {},
                "vo_r2_test": {}, "ok": False,
                "reason": "no layers usable in segment"}
    X_tr, V_tr, Y_tr, L_tr = build_samples_restricted(
        trajs_train, allowed_layers,
    )
    X_te, V_te, Y_te, L_te = build_samples_restricted(
        trajs_test, allowed_layers,
    )
    if X_tr.shape[0] == 0 or X_te.shape[0] == 0:
        return {"layers": allowed_layers, "ok": False,
                "reason": "empty samples"}

    # Velocity-only baseline.
    _, alpha_vo, _ = fit_velocity_only(X_tr, V_tr, Y_tr, L_tr)
    Y_pred_te_vo = np.zeros_like(Y_te)
    for ell, a in alpha_vo.items():
        m = L_te == ell
        Y_pred_te_vo[m] = a * V_te[m]
    r2_vo_te = r2_per_layer(Y_te, Y_pred_te_vo, L_te)

    # Shared V fit on the segment.
    model, _, alpha_np, beta_np, _ = fit_shared_V(
        X_tr, V_tr, Y_tr, L_tr, d, L,
        hidden=hidden, depth=depth, steps=steps,
        batch_size=2048, lr=lr,
        device=device, seed=seed, verbose=False,
    )
    Y_pred_te = predict_shared_V(model, X_te, V_te, L_te,
                                 alpha_np, beta_np, device)
    r2_shv_te = r2_per_layer(Y_te, Y_pred_te, L_te)
    r2_shv_te_overall = r2_overall(Y_te, Y_pred_te)

    return {
        "layers": list(allowed_layers),
        "ok": True,
        "n_train_samples": int(X_tr.shape[0]),
        "n_test_samples": int(X_te.shape[0]),
        "vo_r2_test_per_layer": {int(k): float(v) for k, v in r2_vo_te.items()},
        "shv_r2_test_per_layer": {int(k): float(v) for k, v in r2_shv_te.items()},
        "shv_r2_test_overall": float(r2_shv_te_overall),
        "shv_r2_test_segment_mean": float(
            np.mean([r2_shv_te[ell] for ell in allowed_layers])
        ),
    }


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def render_summary(bundle, results: Dict, full_fit_overall: float,
                   out_path: Path) -> None:
    schedule = bundle["schedule"]
    block_kinds = bundle["block_kinds"]
    L = bundle["L"]
    tag = bundle.get("tag", "?")
    v_hidden = bundle.get("v_hidden", "?")

    lines = [
        f"# Substack-restricted shared-V_ψ separator — Q9d `{schedule}`",
        "",
        f"- Checkpoint tag: `{tag}`",
        f"- Schedule (1-indexed layers): "
        + " ".join(f"{ell}:{k}" for ell, k in
                   enumerate(block_kinds, start=1)),
        f"- Hidden d: {bundle['d']}    v_hidden: {v_hidden}    L: {L}",
        f"- Trajectories pooled: {len(bundle['trajectories'])} "
        f"(train + test from CORPUS)",
        "",
        f"## Full-stack baseline (the v3 §15.8 strict shared-V_ψ test)",
        "",
        f"Shared-V_ψ overall TEST R² fit jointly across **all** "
        f"L = {L} layers: **{full_fit_overall:+.3f}**.",
        "",
        "By design this is expected to be lower than either substack-"
        "restricted segment, because the fit must explain both gradient "
        "S-blocks and Hopfield-like A-blocks with a single scalar.",
        "",
        "## Per-segment refits (fresh V_ψ on restricted samples)",
        "",
        "Each segment of length ≥ 2 is fit independently.  A 2-block "
        "segment yields exactly 0 usable layers (need ell-1 and ell+1 "
        "in-segment); ≥ 3-block segments yield 1+ usable layers.",
        "",
        "Design doc §4.1 prediction: **S-segment R² ≈ 0.90** (SPLM-"
        "substack-like), **A-segment R² in the 0.45-0.65 GPT-2 "
        "middle band**.",
        "",
    ]

    for kind in ("S", "A"):
        kind_name = "S-block" if kind == "S" else "A-block"
        lines += [
            f"### {kind_name} segments",
            "",
        ]
        any_fit = False
        for seg_idx, (lo, hi) in enumerate(results[kind]["segments"]):
            seg_res = results[kind]["fits"][seg_idx]
            seg_label = f"layers {lo}-{hi} (length {hi - lo + 1})"
            if not seg_res["ok"]:
                lines.append(
                    f"- {seg_label}: skipped "
                    f"({seg_res.get('reason', 'no usable layers')})"
                )
                continue
            any_fit = True
            mean_r2 = seg_res["shv_r2_test_segment_mean"]
            lines.append(
                f"- {seg_label}: n_train={seg_res['n_train_samples']:,} "
                f"n_test={seg_res['n_test_samples']:,}, "
                f"**mean shv R² = {mean_r2:+.3f}**"
            )
            for ell in seg_res["layers"]:
                r2 = seg_res["shv_r2_test_per_layer"][ell]
                vo = seg_res["vo_r2_test_per_layer"][ell]
                lines.append(
                    f"    - layer ell={ell:2d}  "
                    f"vel-only R² {vo:+.3f}, vel+shared-V R² **{r2:+.3f}**, "
                    f"gain {(r2-vo):+.3f}"
                )
        if not any_fit:
            lines.append(
                f"- (no contiguous {kind_name} segment of length ≥ 3 in "
                f"this schedule; substack-restricted test degenerate)"
            )
        lines += [""]

    out_path.write_text("\n".join(lines) + "\n")
    print(f"[substack-sep] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True,
                    help="Path to Helmholtz .trajectories.pkl from "
                         "trajectory_extraction_helmholtz.py")
    ap.add_argument("--out", default=None,
                    help="Markdown summary path. Default: "
                         "<traj>.substack_R2.md next to the pickle.")
    ap.add_argument("--out-npz", default=None)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth",  type=int, default=2)
    ap.add_argument("--steps",  type=int, default=2000)
    ap.add_argument("--lr",     type=float, default=3e-3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed",   type=int, default=0)
    args = ap.parse_args()

    device = args.device or (
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    traj_path = Path(args.traj)
    with traj_path.open("rb") as f:
        bundle = pickle.load(f)
    if "block_kinds" not in bundle:
        raise SystemExit(
            "[substack-sep] bundle missing 'block_kinds'.  Re-extract "
            "with the Helmholtz extractor: "
            "trajectory_extraction_helmholtz.py"
        )

    L = int(bundle["L"])
    d = int(bundle["d"])
    block_kinds = bundle["block_kinds"]
    print(f"[substack-sep] schedule={bundle['schedule']}  "
          f"L={L}  d={d}  v_hidden={bundle.get('v_hidden')}")
    print(f"[substack-sep] block_kinds = {block_kinds}")

    train_trajs = [tr for tr in bundle["trajectories"] if tr.split == "train"]
    test_trajs  = [tr for tr in bundle["trajectories"] if tr.split == "test"]
    print(f"[substack-sep] trajs: train={len(train_trajs)}  "
          f"test={len(test_trajs)}")

    # ---- Full-stack baseline (all layers) ----
    from shared_potential_fit import build_samples
    X_tr, V_tr, Y_tr, L_tr = build_samples(train_trajs, L)
    X_te, V_te, Y_te, L_te = build_samples(test_trajs, L)
    print(f"[substack-sep] full-stack samples: train={X_tr.shape[0]:,} "
          f"test={X_te.shape[0]:,}")
    model, _, alpha_np, beta_np, _ = fit_shared_V(
        X_tr, V_tr, Y_tr, L_tr, d, L,
        hidden=args.hidden, depth=args.depth, steps=args.steps,
        batch_size=2048, lr=args.lr,
        device=device, seed=args.seed, verbose=False,
    )
    Y_pred_te = predict_shared_V(model, X_te, V_te, L_te,
                                 alpha_np, beta_np, device)
    full_fit_overall = r2_overall(Y_te, Y_pred_te)
    print(f"[substack-sep] full-stack TEST R² overall: {full_fit_overall:+.3f}")

    # ---- Per-segment fits ----
    segments = find_segments(block_kinds)
    print(f"[substack-sep] segments S={segments['S']}  A={segments['A']}")
    results = {"S": {"segments": segments["S"], "fits": []},
               "A": {"segments": segments["A"], "fits": []}}
    for kind in ("S", "A"):
        for seg in segments[kind]:
            allowed = usable_layers_in_segment(seg, L)
            print(f"[substack-sep] {kind}-segment {seg} "
                  f"-> usable layers {allowed}")
            res = fit_segment(
                train_trajs, test_trajs, allowed,
                d, L, args.hidden, args.depth, args.steps, args.lr,
                device, args.seed,
            )
            results[kind]["fits"].append(res)
            if res.get("ok"):
                print(f"[substack-sep]   mean test R² "
                      f"{res['shv_r2_test_segment_mean']:+.3f}")

    # ---- Save ----
    out_md_default = (traj_path.parent /
                      f"{traj_path.stem.replace('.trajectories','')}"
                      f"_substack_R2.md")
    out_md = Path(args.out) if args.out else out_md_default
    render_summary(bundle, results, full_fit_overall, out_md)

    out_npz = (Path(args.out_npz)
               if args.out_npz else out_md.with_suffix(".npz"))
    np.savez(
        out_npz,
        full_fit_overall_R2=np.array([full_fit_overall]),
        block_kinds=np.array(block_kinds),
        schedule=np.array([bundle["schedule"]]),
        L=np.array([L]),
        s_segments=np.array(results["S"]["segments"], dtype=object),
        a_segments=np.array(results["A"]["segments"], dtype=object),
        s_fits_json=np.array([json.dumps(results["S"]["fits"])]),
        a_fits_json=np.array([json.dumps(results["A"]["fits"])]),
    )
    print(f"[substack-sep] saved npz -> {out_npz}")


if __name__ == "__main__":
    main()
