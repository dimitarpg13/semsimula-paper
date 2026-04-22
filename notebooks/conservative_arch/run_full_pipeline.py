"""Convenience driver: extract trajectories + run E-init validation for a
given SPLM checkpoint.  Roughly equivalent to

    python3 trajectory_extraction.py --ckpt <ckpt>
    python3 e_init_validation.py     --traj <ckpt>.trajectories.pkl

but in a single Python process so the model is only loaded once and
the dataclass namespace stays consistent.
"""

from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max_len", type=int, default=64)
    args = ap.parse_args()

    ckpt = Path(args.ckpt).resolve()
    here = Path(__file__).parent

    print(f"[pipeline] step 1/2: extracting trajectories from {ckpt.name}")
    subprocess.run(
        [sys.executable, str(here / "trajectory_extraction.py"),
         "--ckpt", str(ckpt), "--max_len", str(args.max_len)],
        check=True, cwd=str(here),
    )

    traj_path = ckpt.with_suffix(".trajectories.pkl")
    print(f"[pipeline] step 2/2: E-init validation on {traj_path.name}")
    subprocess.run(
        [sys.executable, str(here / "e_init_validation.py"),
         "--traj", str(traj_path)],
        check=True, cwd=str(here),
    )

    print(f"[pipeline] done.  see notebooks/conservative_arch/results/")


if __name__ == "__main__":
    main()
