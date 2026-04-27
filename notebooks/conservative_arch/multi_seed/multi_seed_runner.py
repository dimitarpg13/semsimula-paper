"""Multi-seed runner for SPLM and matched-baseline trainers.

Wraps the existing ``train_splm_sarf_mass.py`` and ``train_matched.py``
trainers and re-runs each one with N different random seeds, namespacing
the per-seed artefacts into ``results/<tag>/<model_label>/seed_<s>/``.

The harness is intentionally **non-invasive**: it does not modify the
upstream trainers. Each per-seed run is a subprocess started in the
trainer's own working directory; once the subprocess exits, the
fixed-name artefacts that the trainer wrote into its local
``results/`` directory are *moved* into the seed-namespaced subdir so
the next seed has a clean slate.

Usage::

    python3 notebooks/conservative_arch/multi_seed/multi_seed_runner.py \\
        --mode smoke --n-seeds 1 --models splm_sarfmass_logfreq

    python3 notebooks/conservative_arch/multi_seed/multi_seed_runner.py \\
        --mode shakespeare --n-seeds 5 \\
        --models splm_sarfmass_logfreq,matched_baseline \\
        --tag E1_shakespeare

A run-level JSON log of every (model, seed) attempt is written to
``results/<tag>/run_log.jsonl``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = SCRIPT_DIR / "results"
RESULTS_ROOT.mkdir(exist_ok=True)

CONSERVATIVE_ARCH_DIR = SCRIPT_DIR.parent
SARF_MASS_DIR = CONSERVATIVE_ARCH_DIR / "sarf_mass_variant"
EM_DIR = CONSERVATIVE_ARCH_DIR / "energetic_minima"

REPO_ROOT = SCRIPT_DIR.parent.parent.parent

LOGFREQ_MARKER = SARF_MASS_DIR / "results" / "logfreq_surprisal.npy"


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    """Description of an upstream trainer that the harness can drive."""

    label: str
    trainer_dir: Path
    trainer_script: str
    extra_args: Tuple[str, ...]
    artefact_tag: str
    preflight: Optional[Tuple[str, ...]] = None
    preflight_cwd: Optional[Path] = None
    preflight_marker: Optional[Path] = None


MODEL_SPECS: Dict[str, ModelSpec] = {
    "splm_sarfmass_logfreq": ModelSpec(
        label="splm_sarfmass_logfreq",
        trainer_dir=SARF_MASS_DIR,
        trainer_script="train_splm_sarf_mass.py",
        extra_args=("--mass-mode", "logfreq"),
        artefact_tag="splm_sarfmass_logfreq_{mode}",
        preflight=("python3", "compute_unigram_frequencies.py"),
        preflight_cwd=SARF_MASS_DIR,
        preflight_marker=LOGFREQ_MARKER,
    ),
    "splm_em_ln": ModelSpec(
        label="splm_em_ln",
        trainer_dir=EM_DIR,
        trainer_script="train.py",
        extra_args=("--variant", "ln"),
        artefact_tag="em_ln_{mode}",
        preflight=("python3", "compute_unigram_frequencies.py"),
        preflight_cwd=SARF_MASS_DIR,
        preflight_marker=LOGFREQ_MARKER,
    ),
    "matched_baseline": ModelSpec(
        label="matched_baseline",
        trainer_dir=CONSERVATIVE_ARCH_DIR,
        trainer_script="train_matched.py",
        extra_args=tuple(),
        artefact_tag="matched_{mode}",
        preflight=None,
    ),
}


def _resolve_artefacts(spec: ModelSpec, mode: str) -> List[Path]:
    """List of fixed-name artefact files the upstream trainer writes."""
    tag = spec.artefact_tag.format(mode=mode)
    results_dir = spec.trainer_dir / "results"
    return [
        results_dir / f"{tag}_training_log.jsonl",
        results_dir / f"{tag}_ckpt_latest.pt",
        results_dir / f"{tag}_loss_curve.png",
        results_dir / f"{tag}_summary.md",
    ]


def _cleanup_pre_run(spec: ModelSpec, mode: str) -> None:
    """Remove leftover artefacts from a previous run so each seed starts clean."""
    for p in _resolve_artefacts(spec, mode):
        if p.exists():
            p.unlink()


def _move_artefacts(
    spec: ModelSpec,
    mode: str,
    dest_dir: Path,
) -> List[str]:
    """Move the trainer's fixed-name outputs into ``dest_dir``; return moved names."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    moved: List[str] = []
    for src in _resolve_artefacts(spec, mode):
        if not src.exists():
            print(f"  [warn] expected artefact missing: {src.relative_to(REPO_ROOT)}",
                  file=sys.stderr)
            continue
        dst = dest_dir / src.name
        if dst.exists():
            dst.unlink()
        shutil.move(str(src), str(dst))
        moved.append(src.name)
    return moved


def _run_preflight(spec: ModelSpec) -> None:
    """Run the trainer's one-shot preflight (e.g. surprisal-table precompute)."""
    if spec.preflight is None:
        return
    if spec.preflight_marker is not None and spec.preflight_marker.exists():
        print(f"  [preflight] marker present "
              f"({spec.preflight_marker.relative_to(REPO_ROOT)}), "
              f"skipping precompute for {spec.label}.")
        return
    cwd = spec.preflight_cwd or spec.trainer_dir
    print(f"  [preflight] {' '.join(spec.preflight)}  "
          f"(cwd={cwd.relative_to(REPO_ROOT)})")
    subprocess.run(
        list(spec.preflight),
        cwd=cwd,
        check=True,
    )


def _run_one_seed(
    spec: ModelSpec,
    mode: str,
    seed: int,
    device: Optional[str],
    dest_dir: Path,
) -> Tuple[bool, float, str]:
    """Run one ``(model, seed)`` subprocess and move its artefacts."""
    cmd = [
        "python3", spec.trainer_script,
        "--mode", mode,
        "--seed", str(seed),
        *spec.extra_args,
    ]
    if device is not None:
        cmd.extend(["--device", device])

    _cleanup_pre_run(spec, mode)

    print(f"  [run]  {' '.join(cmd)}  (cwd={spec.trainer_dir.relative_to(REPO_ROOT)})")
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=spec.trainer_dir,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0

    log_dest = dest_dir
    log_dest.mkdir(parents=True, exist_ok=True)
    (log_dest / "stdout.log").write_text(proc.stdout)
    (log_dest / "stderr.log").write_text(proc.stderr)

    if proc.returncode != 0:
        print(f"  [fail] returncode={proc.returncode}; see "
              f"{(log_dest / 'stderr.log').relative_to(REPO_ROOT)}",
              file=sys.stderr)
        return False, elapsed, ""

    moved = _move_artefacts(spec, mode, dest_dir)
    print(f"  [done] elapsed={elapsed:.1f}s  moved={len(moved)} artefact(s)")
    return True, elapsed, "; ".join(moved)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["smoke", "shakespeare", "tinystories"],
                    default="smoke",
                    help="Training mode to forward to the upstream trainer.")
    ap.add_argument("--n-seeds", type=int, default=1,
                    help="Number of seeds (0..N-1).")
    ap.add_argument("--seeds", type=str, default=None,
                    help="Comma-separated explicit seeds (overrides --n-seeds).")
    ap.add_argument("--models", type=str, default="splm_sarfmass_logfreq",
                    help=f"Comma-separated model labels. "
                         f"Available: {','.join(MODEL_SPECS.keys())}")
    ap.add_argument("--device", type=str, default=None,
                    help="Optional explicit device (mps|cuda|cpu); "
                         "default = trainer's auto-pick.")
    ap.add_argument("--tag", type=str, default=None,
                    help="Run-tag subdirectory under results/. "
                         "Default = E1_<mode>.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="If a seed_<s>/ subdir already has a checkpoint, "
                         "skip that run.")
    args = ap.parse_args()

    if args.seeds is not None:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = list(range(args.n_seeds))

    model_labels = [m.strip() for m in args.models.split(",") if m.strip()]
    for label in model_labels:
        if label not in MODEL_SPECS:
            raise ValueError(f"unknown model label '{label}'; "
                             f"available: {sorted(MODEL_SPECS.keys())}")

    tag = args.tag or f"E1_{args.mode}"
    run_root = RESULTS_ROOT / tag
    run_root.mkdir(parents=True, exist_ok=True)

    run_log = run_root / "run_log.jsonl"
    print(f"[multi_seed] tag={tag}  mode={args.mode}  "
          f"models={model_labels}  seeds={seeds}  device={args.device or 'auto'}")
    print(f"[multi_seed] outputs -> {run_root.relative_to(REPO_ROOT)}")
    print(f"[multi_seed] run-log -> {run_log.relative_to(REPO_ROOT)}")

    n_total = len(model_labels) * len(seeds)
    n_ok = 0
    n_skipped = 0
    n_failed = 0

    with run_log.open("a") as logf:
        for label in model_labels:
            spec = MODEL_SPECS[label]
            print(f"\n=== {label} ===")
            try:
                _run_preflight(spec)
            except subprocess.CalledProcessError as e:
                print(f"  [preflight-fail] {e}", file=sys.stderr)
                logf.write(json.dumps({
                    "phase": "preflight",
                    "model": label,
                    "ok": False,
                    "error": str(e),
                }) + "\n")
                logf.flush()
                continue

            for seed in seeds:
                dest_dir = run_root / label / f"seed_{seed}"
                ckpt = dest_dir / (
                    spec.artefact_tag.format(mode=args.mode) + "_ckpt_latest.pt"
                )
                if args.skip_existing and ckpt.exists():
                    print(f"  [skip] {label} seed={seed} (checkpoint exists)")
                    n_skipped += 1
                    logf.write(json.dumps({
                        "phase": "run",
                        "model": label,
                        "seed": seed,
                        "ok": True,
                        "skipped": True,
                    }) + "\n")
                    logf.flush()
                    continue

                ok, elapsed, moved = _run_one_seed(
                    spec=spec, mode=args.mode, seed=seed,
                    device=args.device, dest_dir=dest_dir,
                )
                logf.write(json.dumps({
                    "phase": "run",
                    "model": label,
                    "seed": seed,
                    "mode": args.mode,
                    "device": args.device,
                    "ok": ok,
                    "elapsed_s": elapsed,
                    "artefacts": moved,
                }) + "\n")
                logf.flush()
                if ok:
                    n_ok += 1
                else:
                    n_failed += 1

    print(f"\n[multi_seed] summary: {n_ok}/{n_total} OK  "
          f"{n_skipped} skipped  {n_failed} failed")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
