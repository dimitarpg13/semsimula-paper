"""Per-parameter-group gradient clipping shared by the CfC/BAOAB training
loop (Cell 6) and the Phase 1/2 spike-replay forensics (Cell 6d) in
colab_fock_cfc_baoab_aniso_gaussian_openwebtext_d384.ipynb.

Extracted 2026-08-30 (companion note CfC_BAOAB_Integrator_and_Mitigations.md
SS37): these three functions used to be defined inline in Cell 6, with
Cell 6d's replay_spike_batch()/replay_all_captures() silently depending on
Cell 6 having already executed at least once (so they existed as notebook
globals) before either replay function could be called. That made Cell 6d
impossible to load standalone -- it had to run *after* Cell 6, which is
backwards from how it is actually used (you want the replay helpers already
defined *before* you start the long-running training loop, since Colab can
only run one cell at a time and Cell 6 blocks the kernel for the rest of the
session). Moving them here removes that ordering dependency entirely.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class GradClipConfig:
    """Bundles the three knobs that determine how a parameter's name maps to
    a clip group and threshold, so callers don't have to pass them
    separately at every call site.

    Attributes:
        default_clip: clip threshold for any parameter that doesn't match an
            entry in `overrides`.
        overrides: substring (case-insensitive) -> clip threshold. Any
            override whose key appears in a parameter's name wins, and the
            parameter is placed in a group named f'override:{key}'.
        watchdog_exclude_groups: group keys (post-override, i.e. already in
            'override:...' form where applicable) to exclude from the
            *aggregate* norm returned by `clip_grads_per_group` -- their own
            per-group norm is still computed and clipped normally, they're
            just not counted toward the watchdog's total.
    """

    default_clip: float
    overrides: Dict[str, float] = field(default_factory=dict)
    watchdog_exclude_groups: FrozenSet[str] = field(default_factory=frozenset)


def assign_clip_group(pname: str, cfg: GradClipConfig) -> Tuple[str, float]:
    """Map a parameter name to (clip-group key, clip threshold)."""
    low = pname.lower()
    for sub, thr in cfg.overrides.items():
        if sub.lower() in low:
            return f'override:{sub}', thr
    return pname.split('.', 1)[0], cfg.default_clip


def per_group_grad_norms(mdl: nn.Module, cfg: GradClipConfig) -> Dict[str, float]:
    """Read-only per-clip-group L2 grad norm (no clipping applied)."""
    groups: Dict[str, List[nn.Parameter]] = {}
    for n, p in mdl.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        key, _ = assign_clip_group(n, cfg)
        groups.setdefault(key, []).append(p)
    out: Dict[str, float] = {}
    for key, ps in groups.items():
        sq = 0.0
        for p in ps:
            sq += float(p.grad.detach().norm()) ** 2
        out[key] = sq ** 0.5
    return out


def clip_grads_per_group(
    mdl: nn.Module, cfg: GradClipConfig
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Clip each group's grads independently in place; return the aggregate
    pre-clip norm (excluding `cfg.watchdog_exclude_groups`) plus every
    group's own pre-clip norm (`nn.utils.clip_grad_norm_` returns the norm
    computed *before* it applies the rescale, for every group -- including
    excluded ones, which are just left out of the aggregate).
    """
    groups: Dict[str, List[nn.Parameter]] = {}
    thr: Dict[str, float] = {}
    device = None
    for n, p in mdl.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        if device is None:
            device = p.grad.device
        key, mx = assign_clip_group(n, cfg)
        groups.setdefault(key, []).append(p)
        thr[key] = mx
    total_sq = torch.zeros((), device=device) if device is not None else torch.zeros(())
    per_group: Dict[str, float] = {}
    for key, ps in groups.items():
        gn = nn.utils.clip_grad_norm_(ps, thr[key])
        per_group[key] = float(gn)
        if key not in cfg.watchdog_exclude_groups:
            total_sq = total_sq + gn.detach() ** 2
    return total_sq.sqrt(), per_group
