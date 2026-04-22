"""
Precompute unigram surprisal -log p_hat(v) for every GPT-2 BPE token id, using
Tiny Shakespeare train split as the corpus.

This is a one-off preprocessing step.  Output: `results/logfreq_surprisal.npy`,
a float32 vector of shape (50257,), ready to be passed to
SPLMSARFMassConfig(mass_mode='logfreq', logfreq_path=...).

Smoothing:
  p_hat(v) = (c_v + 1) / (N + V) -- add-one Laplace smoothing, so every
  vocabulary id gets a finite surprisal even when unseen in the corpus.
Surprisal:
  s(v) = -log p_hat(v), with natural log (units: nats).  The scale is
  learned via the softplus-ed alpha in the model, so the choice of log
  base does not matter for training.

Usage:
    python3 compute_unigram_frequencies.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent
PARENT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(PARENT_DIR))
from data_module import load_tiny_shakespeare  # noqa: E402


def main():
    vocab_size = 50257

    train_ids, val_ids = load_tiny_shakespeare()
    N = len(train_ids)
    print(f"[freq] corpus tokens: train={N:,}  val={len(val_ids):,}  "
          f"vocab_size={vocab_size}")

    counts = np.bincount(train_ids, minlength=vocab_size).astype(np.int64)
    nz = int((counts > 0).sum())
    print(f"[freq] unique types seen: {nz:,} / {vocab_size:,} "
          f"({100 * nz / vocab_size:.1f}%)")

    p = (counts + 1.0) / (N + vocab_size)
    surprisal = -np.log(p).astype(np.float32)
    print(f"[freq] surprisal  min={surprisal.min():.3f}  "
          f"max={surprisal.max():.3f}  "
          f"mean={surprisal.mean():.3f}  median={np.median(surprisal):.3f}")

    seen = surprisal[counts > 0]
    print(f"[freq] seen-only  min={seen.min():.3f}  "
          f"max={seen.max():.3f}  mean={seen.mean():.3f}")

    out = RESULTS_DIR / "logfreq_surprisal.npy"
    np.save(out, surprisal)
    print(f"[freq] saved -> {out}  "
          f"shape={surprisal.shape}  dtype={surprisal.dtype}")

    meta = RESULTS_DIR / "logfreq_surprisal.meta.txt"
    with meta.open("w") as f:
        f.write("Unigram surprisal -log p_hat(v) for GPT-2 BPE vocabulary\n")
        f.write("Corpus: Tiny Shakespeare (train split)\n")
        f.write(f"Tokens: {N:,}\n")
        f.write(f"Vocab:  {vocab_size:,}\n")
        f.write(f"Unique seen: {nz:,}\n")
        f.write("Smoothing: add-one Laplace, p = (c+1)/(N+V)\n")
        f.write(f"surprisal: min={surprisal.min():.3f}  "
                f"max={surprisal.max():.3f}  "
                f"mean={surprisal.mean():.3f}  "
                f"median={np.median(surprisal):.3f}\n")
    print(f"[freq] metadata -> {meta}")


if __name__ == "__main__":
    main()
