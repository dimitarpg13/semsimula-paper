"""Correctness tests for grad_clip_utils.py.

Run directly (CPU, < 1 second)::

    python test_grad_clip_utils.py

These are pure control-flow / bookkeeping tests -- no GPU, no real model,
just a tiny toy module standing in for the parameter-naming conventions
(`E`, `P`, `override:...`, plain top-level attrs) that
FockMultiXiPARFLM / Cell 6 actually use.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

from grad_clip_utils import (                                    # noqa: E402
    GradClipConfig,
    assign_clip_group,
    clip_grads_per_group,
    per_group_grad_norms,
)


class _ToyModel(nn.Module):
    """Names chosen to exercise: an override match ('depth_code' inside
    'V_theta.depth_code'), a plain top-level group ('E'), and a group with no
    override match at all ('score_head')."""

    def __init__(self):
        super().__init__()
        self.E = nn.Parameter(torch.zeros(4))
        self.score_head = nn.Parameter(torch.zeros(4))
        self.V_theta = nn.Module()
        self.V_theta.depth_code = nn.Parameter(torch.zeros(4))
        # named_parameters() only descends into registered submodules --
        # V_theta must actually be an nn.Module for 'V_theta.depth_code' to
        # show up with that dotted name.


def _cfg():
    return GradClipConfig(
        default_clip=1.0,
        overrides={'depth_code': 0.25, 'reverse_channel_scale': 0.1},
        watchdog_exclude_groups=frozenset({'override:reverse_channel_scale'}),
    )


def test_assign_clip_group_override_match():
    cfg = _cfg()
    key, thr = assign_clip_group('V_theta.depth_code', cfg)
    assert key == 'override:depth_code', key
    assert thr == 0.25, thr


def test_assign_clip_group_default_fallback():
    cfg = _cfg()
    key, thr = assign_clip_group('score_head', cfg)
    assert key == 'score_head', key
    assert thr == 1.0, thr

    key2, thr2 = assign_clip_group('E.weight', cfg)
    assert key2 == 'E', key2
    assert thr2 == 1.0, thr2
    print('  [ok] override match wins; unmatched names fall back to the '
          'top-level attr name at default_clip')


def test_per_group_grad_norms_matches_manual_computation():
    cfg = _cfg()
    model = _ToyModel()
    with torch.no_grad():
        model.E.grad = torch.full_like(model.E, 3.0)             # norm 6.0
        model.score_head.grad = torch.full_like(model.score_head, 1.0)  # norm 2.0
        model.V_theta.depth_code.grad = torch.full_like(model.V_theta.depth_code, 5.0)

    out = per_group_grad_norms(model, cfg)
    assert set(out) == {'E', 'score_head', 'override:depth_code'}, out
    assert abs(out['E'] - 6.0) < 1e-5, out['E']
    assert abs(out['score_head'] - 2.0) < 1e-5, out['score_head']
    assert abs(out['override:depth_code'] - 10.0) < 1e-5, out['override:depth_code']
    # read-only: grads must be untouched
    assert torch.allclose(model.E.grad, torch.full_like(model.E, 3.0))
    print('  [ok] per_group_grad_norms groups correctly and does not mutate grads')


def test_clip_grads_per_group_clips_and_excludes_from_aggregate():
    cfg = GradClipConfig(
        default_clip=1.0,
        overrides={'reverse_channel_scale': 0.1},
        watchdog_exclude_groups=frozenset({'override:reverse_channel_scale'}),
    )
    model = nn.Module()
    model.E = nn.Parameter(torch.zeros(4))
    model.reverse_channel_scale = nn.Parameter(torch.zeros(1))
    with torch.no_grad():
        model.E.grad = torch.full_like(model.E, 3.0)              # norm 6.0, clip 1.0
        model.reverse_channel_scale.grad = torch.full_like(
            model.reverse_channel_scale, 100.0)                    # norm 100.0, clip 0.1, excluded

    agg, per_group = clip_grads_per_group(model, cfg)

    # E's grad must actually be rescaled down to norm ~1.0 (clip_grad_norm_
    # mutates in place).
    assert abs(float(model.E.grad.norm()) - 1.0) < 1e-4, float(model.E.grad.norm())
    # reverse_channel_scale is clipped too, just excluded from the aggregate.
    assert abs(float(model.reverse_channel_scale.grad.norm()) - 0.1) < 1e-4

    assert abs(per_group['E'] - 6.0) < 1e-4, per_group          # pre-clip norm reported
    assert abs(per_group['override:reverse_channel_scale'] - 100.0) < 1e-2, per_group

    # aggregate excludes the override group entirely -> should equal E's
    # pre-clip norm alone, not sqrt(6^2 + 100^2).
    assert abs(float(agg) - 6.0) < 1e-4, float(agg)
    print('  [ok] clip_grads_per_group clips every group but only aggregates '
          'non-excluded groups (matches the watchdog-blind-spot behaviour '
          'documented in Cell 6)')


def main():
    tests = [
        test_assign_clip_group_override_match,
        test_assign_clip_group_default_fallback,
        test_per_group_grad_norms_matches_manual_computation,
        test_clip_grads_per_group_clips_and_excludes_from_aggregate,
    ]
    for t in tests:
        print(f"\n{t.__name__}:")
        t()
    print("\nAll grad_clip_utils tests passed.")


if __name__ == "__main__":
    main()
