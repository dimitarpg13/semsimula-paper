"""
Precompute unigram surprisal -log p_hat(v) for every GPT-2 BPE token id, using
the **TinyStories train split** (capped at the same 5,000,000-token budget the
scale-up uses for training).  This is the SARF logfreq mass cache for the E9
scale-up SPLM em_ln arm.

Output: `results/logfreq_surprisal_tinystories.npy`
        + `results/logfreq_surprisal_tinystories.meta.txt`

Smoothing:  p_hat(v) = (c_v + 1) / (N + V), Laplace add-one
Surprisal:  s(v) = -log p_hat(v), nats

The model rescales surprisal by softplus(alpha) at runtime, so the choice of
log base does not matter.

Usage:
    python3 compute_unigram_frequencies_tinystories.py
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
from data_module import load_tiny_stories  # noqa: E402


def main():
    vocab_size = 50257
    max_train_tokens = 5_000_000

    train_ids, val_ids = load_tiny_stories(max_train_tokens=max_train_tokens)
    N = len(train_ids)
    print(f"[freq-ts] corpus tokens: train={N:,}  val={len(val_ids):,}  "
          f"vocab_size={vocab_size}")

    counts = np.bincount(train_ids, minlength=vocab_size).astype(np.int64)
    nz = int((counts > 0).sum())
    print(f"[freq-ts] unique types seen: {nz:,} / {vocab_size:,} "
          f"({100 * nz / vocab_size:.1f}%)")

    p = (counts + 1.0) / (N + vocab_size)
    surprisal = -np.log(p).astype(np.float32)
    print(f"[freq-ts] surprisal  min={surprisal.min():.3f}  "
          f"max={surprisal.max():.3f}  "
          f"mean={surprisal.mean():.3f}  median={np.median(surprisal):.3f}")

    seen = surprisal[counts > 0]
    print(f"[freq-ts] seen-only  min={seen.min():.3f}  "
          f"max={seen.max():.3f}  mean={seen.mean():.3f}")

    out = RESULTS_DIR / "logfreq_surprisal_tinystories.npy"
    np.save(out, surprisal)
    print(f"[freq-ts] saved -> {out}  "
          f"shape={surprisal.shape}  dtype={surprisal.dtype}")

    meta = RESULTS_DIR / "logfreq_surprisal_tinystories.meta.txt"
    with meta.open("w") as f:
        f.write("Unigram surprisal -log p_hat(v) for GPT-2 BPE vocabulary\n")
        f.write("Corpus: TinyStories train split (capped)\n")
        f.write(f"Tokens: {N:,}\n")
        f.write(f"Vocab:  {vocab_size:,}\n")
        f.write(f"Unique seen: {nz:,}\n")
        f.write(f"max_train_tokens cap: {max_train_tokens:,}\n")
        f.write("Smoothing: add-one Laplace, p = (c+1)/(N+V)\n")
        f.write(f"surprisal: min={surprisal.min():.3f}  "
                f"max={surprisal.max():.3f}  "
                f"mean={surprisal.mean():.3f}  "
                f"median={np.median(surprisal):.3f}\n")
    print(f"[freq-ts] metadata -> {meta}")


if __name__ == "__main__":
    main()
