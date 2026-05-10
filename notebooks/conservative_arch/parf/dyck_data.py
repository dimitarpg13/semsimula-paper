"""
Dyck_n language data generator for expressivity falsification experiments.

Generates balanced Dyck_n strings (well-formed nested bracket sequences) at
controlled maximum nesting depth.  Used to test whether FockPARFLM can
recognise context-free languages beyond the v0-ceiling collapse depth D*.

Task: next-bracket-type prediction.  Given a prefix of a Dyck string, predict
whether the next token is an open bracket (and which type) or a close bracket
(and which type).  This is a well-defined conditional distribution over the
2n bracket types + EOS.

Vocabulary:
  0        : PAD
  1        : BOS (start of sequence)
  2        : EOS (end of sequence)
  3..2+2n  : brackets.  For bracket type i (0-indexed):
               open_i  = 3 + 2*i
               close_i = 3 + 2*i + 1

Example for n=2: vocab = {PAD=0, BOS=1, EOS=2, (=3, )=4, [=5, ]=6}
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class DyckConfig:
    """Configuration for Dyck_n data generation."""
    n_types: int = 2              # Number of bracket types
    max_depth: int = 8            # Maximum nesting depth in generated strings
    min_length: int = 4           # Minimum string length (in brackets, excluding BOS/EOS)
    max_length: int = 64          # Maximum string length
    p_open: float = 0.5           # Probability of opening (vs closing) at each step
                                  # when both are valid.  Higher = deeper nesting.

    @property
    def vocab_size(self) -> int:
        """PAD + BOS + EOS + 2*n_types bracket tokens."""
        return 3 + 2 * self.n_types

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def bos_id(self) -> int:
        return 1

    @property
    def eos_id(self) -> int:
        return 2

    def open_id(self, bracket_type: int) -> int:
        return 3 + 2 * bracket_type

    def close_id(self, bracket_type: int) -> int:
        return 3 + 2 * bracket_type + 1

    def is_open(self, token_id: int) -> bool:
        return token_id >= 3 and (token_id - 3) % 2 == 0

    def is_close(self, token_id: int) -> bool:
        return token_id >= 3 and (token_id - 3) % 2 == 1

    def bracket_type_of(self, token_id: int) -> int:
        return (token_id - 3) // 2


def generate_dyck_string(cfg: DyckConfig, rng: random.Random) -> List[int]:
    """Generate a single valid Dyck_n string as a list of token IDs.

    Algorithm: random walk with stack tracking.  At each step:
      - If stack is empty: must open (random type).
      - If stack depth == max_depth: must close (top of stack).
      - If remaining budget forces closing all: close.
      - Otherwise: open with prob p_open, close with prob 1-p_open.

    Returns list including BOS and EOS sentinels.
    """
    target_len = rng.randint(cfg.min_length, cfg.max_length)
    # Ensure target_len is even (Dyck strings have equal opens/closes).
    if target_len % 2 == 1:
        target_len += 1

    stack: List[int] = []  # Stack of bracket types (LIFO)
    tokens: List[int] = [cfg.bos_id]

    n_emitted = 0
    while n_emitted < target_len:
        remaining = target_len - n_emitted
        depth = len(stack)

        # Must close everything if remaining == depth (no room to open more).
        if remaining <= depth:
            # Close in LIFO order.
            tokens.append(cfg.close_id(stack.pop()))
            n_emitted += 1
            continue

        # Must open if stack is empty.
        if depth == 0:
            btype = rng.randint(0, cfg.n_types - 1)
            stack.append(btype)
            tokens.append(cfg.open_id(btype))
            n_emitted += 1
            continue

        # At max depth: must close.
        if depth >= cfg.max_depth:
            tokens.append(cfg.close_id(stack.pop()))
            n_emitted += 1
            continue

        # Otherwise: stochastic choice.
        if rng.random() < cfg.p_open:
            btype = rng.randint(0, cfg.n_types - 1)
            stack.append(btype)
            tokens.append(cfg.open_id(btype))
        else:
            tokens.append(cfg.close_id(stack.pop()))
        n_emitted += 1

    tokens.append(cfg.eos_id)
    return tokens


def generate_dyck_dataset(
    cfg: DyckConfig,
    n_samples: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a dataset of Dyck_n strings for next-token prediction.

    Returns:
        x: (n_samples, max_seq_len) int64 array — input sequences (BOS + brackets).
        y: (n_samples, max_seq_len) int64 array — target sequences (shifted by 1).

    Sequences are padded to the longest sequence in the batch.  Targets at
    PAD positions are set to -100 (PyTorch ignore_index for cross_entropy).
    """
    rng = random.Random(seed)
    sequences = [generate_dyck_string(cfg, rng) for _ in range(n_samples)]

    # Determine max length for padding.
    max_len = max(len(s) for s in sequences)

    x = np.full((n_samples, max_len - 1), cfg.pad_id, dtype=np.int64)
    y = np.full((n_samples, max_len - 1), -100, dtype=np.int64)

    for i, seq in enumerate(sequences):
        # x = seq[:-1], y = seq[1:] (standard next-token prediction).
        seq_len = len(seq) - 1
        x[i, :seq_len] = seq[:-1]
        y[i, :seq_len] = seq[1:]

    return x, y


def measure_max_depth(tokens: List[int], cfg: DyckConfig) -> int:
    """Measure the maximum nesting depth achieved in a Dyck token sequence."""
    depth = 0
    max_d = 0
    for t in tokens:
        if cfg.is_open(t):
            depth += 1
            max_d = max(max_d, depth)
        elif cfg.is_close(t):
            depth -= 1
    return max_d


def generate_depth_controlled_dataset(
    cfg: DyckConfig,
    n_samples: int,
    min_depth: int,
    max_depth: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate dataset filtered to strings achieving depth in [min_depth, max_depth].

    Useful for testing expressivity at specific nesting depths (e.g., testing
    whether FockPARFLM handles depth > D* where plain PARFLM collapses).

    Returns:
        x, y: as in generate_dyck_dataset
        depths: (n_samples,) int array — actual max depth of each sample.
    """
    rng = random.Random(seed)
    sequences = []
    depths_list = []

    attempts = 0
    max_attempts = n_samples * 100

    while len(sequences) < n_samples and attempts < max_attempts:
        attempts += 1
        seq = generate_dyck_string(cfg, rng)
        d = measure_max_depth(seq, cfg)
        if min_depth <= d <= max_depth:
            sequences.append(seq)
            depths_list.append(d)

    if len(sequences) < n_samples:
        raise RuntimeError(
            f"Could only generate {len(sequences)}/{n_samples} sequences "
            f"with depth in [{min_depth}, {max_depth}] after {max_attempts} "
            f"attempts.  Try increasing max_length or adjusting p_open."
        )

    max_len = max(len(s) for s in sequences)
    x = np.full((len(sequences), max_len - 1), cfg.pad_id, dtype=np.int64)
    y = np.full((len(sequences), max_len - 1), -100, dtype=np.int64)

    for i, seq in enumerate(sequences):
        seq_len = len(seq) - 1
        x[i, :seq_len] = seq[:-1]
        y[i, :seq_len] = seq[1:]

    return x, y, np.array(depths_list, dtype=np.int64)


def get_dyck_batch(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a random batch from pregenerated Dyck data."""
    idx = rng.integers(0, len(x), size=batch_size)
    return x[idx], y[idx]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = DyckConfig(n_types=2, max_depth=8, min_length=8, max_length=32, p_open=0.55)
    print(f"Dyck_{cfg.n_types} config: vocab_size={cfg.vocab_size}, "
          f"max_depth={cfg.max_depth}")

    rng = random.Random(0)
    bracket_names = {cfg.pad_id: "·", cfg.bos_id: "⟨", cfg.eos_id: "⟩"}
    for i in range(cfg.n_types):
        bracket_names[cfg.open_id(i)] = "([{"[i]
        bracket_names[cfg.close_id(i)] = ")]}"[i]

    print("\nSample strings:")
    for _ in range(5):
        seq = generate_dyck_string(cfg, rng)
        d = measure_max_depth(seq, cfg)
        s = "".join(bracket_names.get(t, "?") for t in seq)
        print(f"  depth={d:2d}  len={len(seq)-2:3d}  {s}")

    print("\nDepth-controlled generation (depth 5-8):")
    x, y, depths = generate_depth_controlled_dataset(
        cfg, n_samples=100, min_depth=5, max_depth=8, seed=123
    )
    print(f"  generated {len(x)} samples, depth distribution:")
    for d in sorted(set(depths)):
        print(f"    depth {d}: {(depths == d).sum()} samples")

    print(f"\n  x shape: {x.shape}, y shape: {y.shape}")
    print(f"  x[0][:20]: {x[0][:20]}")
    print(f"  y[0][:20]: {y[0][:20]}")
