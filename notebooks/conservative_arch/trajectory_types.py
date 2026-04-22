"""Shared Trajectory dataclass for SPLM E-init pipeline.

Lives in its own module so pickle can resolve it from any entry point
(trajectory_extraction.py writer, e_init_validation.py reader).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Trajectory:
    """Per-sentence hidden-state trajectory plus next-token diagnostics.

    Mirrors the dataclass used in `notebooks/e_init/` pipeline so the
    same E-init fitting code works unchanged.
    """
    sentence: str
    domain:   str
    split:    str                           # "train" or "test"
    tok_ids:  np.ndarray                    # (T,) int64
    hs:       np.ndarray                    # (L+1, T, d) float32
    ptl:      np.ndarray                    # (T-1,) float32 -- per-token cross-entropy
    w:        Optional[np.ndarray] = None   # (L+1, T)   per-layer per-token "mass"
    mu_ps:    Optional[np.ndarray] = None   # (L+1, 1, d) per-sentence mean
    x_ps:     Optional[np.ndarray] = None   # (L+1, T, d) centered hidden states
