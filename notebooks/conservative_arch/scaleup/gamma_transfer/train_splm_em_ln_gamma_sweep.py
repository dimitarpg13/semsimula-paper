"""
E10 γ-transfer experiment — SPLM em_ln γ-sweep trainer (thin wrapper).

This is a thin wrapper over `notebooks/conservative_arch/scaleup/train_splm_em_ln_scaleup.py`
that delegates to its `main()` after enforcing the E10-protocol-mandated
modes (`pilot` / `confirmation`) and the *required* `--fixed-gamma` argument.

Pre-registered protocol:
  docs/Gamma_transfer_pre-registered_protocol.md
Pre-registration commit: 75cad01

Modes
-----
  --mode pilot         : E10 Stage 1 (4000-step truncated, 200 warmup, 200 eval).
  --mode confirmation  : E10 Stages 2+3 (8000-step full, identical to E9 'scaleup').

Usage
-----
  # Stage 1 example:
  python train_splm_em_ln_gamma_sweep.py \
      --mode pilot \
      --fixed-gamma 0.10 \
      --seed 0 \
      --tag-suffix stage1_g0p10_seed0 \
      --results-dir notebooks/conservative_arch/scaleup/gamma_transfer/results/stage1/g0p10_seed0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
SCALEUP_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCALEUP_DIR))

import train_splm_em_ln_scaleup as _scaleup_trainer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "E10 γ-transfer experiment trainer. Thin wrapper around "
            "train_splm_em_ln_scaleup.py with the protocol's modes & a "
            "required --fixed-gamma argument."
        ),
    )
    ap.add_argument(
        "--mode",
        choices=["pilot", "confirmation"],
        required=True,
        help="pilot = Stage 1 (4000 steps); confirmation = Stages 2+3 (8000 steps).",
    )
    ap.add_argument(
        "--fixed-gamma",
        dest="fixed_gamma",
        type=float,
        required=True,
        help=(
            "Damping coefficient (must be supplied explicitly for the γ-sweep). "
            "Allowed Stage-1 values from the protocol: 0.10, 0.30, 0.60 "
            "(plus boundary expansion 0.05 / 0.85 if triggered)."
        ),
    )
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tag-suffix", dest="tag_suffix", type=str, required=True)
    ap.add_argument(
        "--results-dir",
        dest="results_dir",
        type=str,
        required=True,
    )
    ap.add_argument(
        "--logfreq-path",
        dest="logfreq_path",
        type=str,
        default=str(SCALEUP_DIR / "results" / "logfreq_surprisal_tinystories.npy"),
    )
    ap.add_argument(
        "--max-train-tokens",
        dest="max_train_tokens",
        type=int,
        default=5_000_000,
    )
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    # Translate to the underlying scaleup trainer's argv.
    forward_argv = [
        "train_splm_em_ln_scaleup.py",
        "--mode", args.mode,
        "--fixed-gamma", str(args.fixed_gamma),
        "--seed", str(args.seed),
        "--tag-suffix", args.tag_suffix,
        "--results-dir", args.results_dir,
        "--logfreq-path", args.logfreq_path,
        "--max-train-tokens", str(args.max_train_tokens),
    ]
    if args.device is not None:
        forward_argv += ["--device", args.device]

    sys.argv = forward_argv
    _scaleup_trainer.main()


if __name__ == "__main__":
    main()
