"""
Data loading utilities for conservative-by-construction language models.

Provides two corpora:
  1. tiny_shakespeare  (~1.1 MB, plain text, downloads on first call)
  2. tiny_stories      (~few hundred MB of parquet, downloaded via HF Hub)

Both are tokenised with the GPT-2 BPE tokenizer (matching the §1 E-init
experiments).  Tokens are cached on disk as uint16 numpy arrays for fast
reload.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np


DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

_TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)


def _ensure_shakespeare_text() -> str:
    txt_path = DATA_DIR / "tinyshakespeare.txt"
    if not txt_path.exists():
        print(f"[data_module] Downloading tiny_shakespeare to {txt_path} ...")
        urllib.request.urlretrieve(_TINY_SHAKESPEARE_URL, txt_path)
    return txt_path.read_text(encoding="utf-8")


def _gpt2_tokenize(text: str) -> np.ndarray:
    """GPT-2 BPE tokenise `text`; return uint16 numpy array of token ids."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    ids = tok.encode(text)
    ids = np.asarray(ids, dtype=np.uint16)
    return ids


def load_tiny_shakespeare(val_frac: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Return (train_ids, val_ids) as uint16 numpy arrays."""
    cache = DATA_DIR / "tinyshakespeare_gpt2.npz"
    if cache.exists():
        z = np.load(cache)
        return z["train"], z["val"]
    text = _ensure_shakespeare_text()
    ids = _gpt2_tokenize(text)
    n_val = int(len(ids) * val_frac)
    val_ids = ids[-n_val:]
    train_ids = ids[:-n_val]
    np.savez(cache, train=train_ids, val=val_ids)
    print(f"[data_module] Cached tokens: train={len(train_ids):,}  val={len(val_ids):,}")
    return train_ids, val_ids


def _download_hf_parquet(repo_id: str, filename: str, local_name: str) -> Path:
    from huggingface_hub import hf_hub_download

    dest = DATA_DIR / local_name
    if dest.exists():
        return dest
    src = hf_hub_download(
        repo_id=repo_id, filename=filename,
        repo_type="dataset", local_dir=str(DATA_DIR),
    )
    src_path = Path(src)
    if src_path != dest:
        src_path.rename(dest)
    return dest


def _resolve_tinystories_shard(prefix: str) -> str:
    """Return the actual repo path for a TinyStories shard.

    The HF dataset stores parquet shards under names like
        data/train-00001-of-00004-<hash>.parquet
    where the hash suffix can change across uploads.  This helper looks
    up the live file list and returns the unique shard whose name starts
    with `prefix` (e.g. ``data/train-00001-of-00004``).
    """
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files("roneneldan/TinyStories", repo_type="dataset")
    matches = [f for f in files if f.startswith(prefix) and f.endswith(".parquet")]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"TinyStories shard prefix {prefix!r} resolved to "
            f"{len(matches)} files (expected exactly 1): {matches!r}"
        )
    return matches[0]


def load_tiny_stories(
    n_train_files: int = 1,
    val_frac: float = 0.01,
    max_train_tokens: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (train_ids, val_ids) tokenised with GPT-2 BPE.

    Uses `roneneldan/TinyStories` parquet files directly via HF Hub,
    avoiding the `datasets` dependency.  `n_train_files` selects how
    many training parquet shards to ingest (each ~300 MB uncompressed).
    """
    import pyarrow.parquet as pq

    cache = DATA_DIR / (
        f"tinystories_gpt2_{n_train_files}files"
        + (f"_{max_train_tokens}toks" if max_train_tokens else "")
        + ".npz"
    )
    if cache.exists():
        z = np.load(cache)
        return z["train"], z["val"]

    # Fetch train parquet shard(s) and a separate validation shard.
    # Tokenize shard-by-shard to keep peak memory manageable (avoids
    # joining ~1 GB of text into a single string for multi-shard loads).
    train_id_chunks: list[np.ndarray] = []
    n_stories = 0
    for i in range(n_train_files):
        fname = _resolve_tinystories_shard(
            f"data/train-{i:05d}-of-00004"
        )
        p = _download_hf_parquet("roneneldan/TinyStories", fname,
                                 f"tinystories_train_{i:05d}.parquet")
        print(f"[data_module] Reading {p}")
        table = pq.read_table(p, columns=["text"])
        shard_texts = table["text"].to_pylist()
        n_stories += len(shard_texts)
        del table
        print(f"[data_module]   tokenising shard {i} "
              f"({len(shard_texts):,} stories) ...")
        chunk = _gpt2_tokenize("\n\n".join(shard_texts))
        del shard_texts
        train_id_chunks.append(chunk)
        # Early exit if we already have enough tokens.
        if (max_train_tokens is not None
                and sum(len(c) for c in train_id_chunks) >= max_train_tokens):
            break

    train_ids = np.concatenate(train_id_chunks)
    del train_id_chunks

    val_fname = _resolve_tinystories_shard("data/validation-00000-of-00001")
    p = _download_hf_parquet(
        "roneneldan/TinyStories",
        val_fname,
        "tinystories_val.parquet",
    )
    print(f"[data_module] Reading {p}")
    val_texts = pq.read_table(p, columns=["text"])["text"].to_pylist()

    print(f"[data_module] Tokenising {n_stories:,} train + "
          f"{len(val_texts):,} val stories with GPT-2 BPE ...")
    val_ids = _gpt2_tokenize("\n\n".join(val_texts))
    del val_texts
    if max_train_tokens is not None and len(train_ids) > max_train_tokens:
        train_ids = train_ids[:max_train_tokens]

    np.savez(cache, train=train_ids, val=val_ids)
    print(f"[data_module] Cached tokens: train={len(train_ids):,}  "
          f"val={len(val_ids):,}  -> {cache}")
    return train_ids, val_ids


def get_batch(ids: np.ndarray, batch_size: int, block_size: int,
              rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform random (x, y) pairs of shape (B, block_size) each.

    y[b, t] = ids[start[b] + t + 1] is the next-token target for
    x[b, t] = ids[start[b] + t].
    """
    n = len(ids) - block_size - 1
    starts = rng.integers(0, n, size=batch_size)
    x = np.stack([ids[s:s + block_size]     for s in starts])
    y = np.stack([ids[s + 1:s + 1 + block_size] for s in starts])
    return x.astype(np.int64), y.astype(np.int64)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["shakespeare", "tinystories"],
                    default="shakespeare")
    args = ap.parse_args()
    if args.corpus == "shakespeare":
        tr, va = load_tiny_shakespeare()
    else:
        tr, va = load_tiny_stories(max_train_tokens=5_000_000)
    print(f"train tokens: {len(tr):,}   val tokens: {len(va):,}")
    print(f"train first 32 ids: {tr[:32]}")
