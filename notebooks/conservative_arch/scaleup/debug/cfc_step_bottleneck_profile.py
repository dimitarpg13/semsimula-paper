#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pinpoint whether a Fock-PARFLM train step is GPU-compute-bound or CPU/launch-bound.

Three independent measurements, then a vote.  CUDA_LAUNCH_BLOCKING is
deliberately not used: that env var has to be set before the first CUDA
call, so it would require a Colab kernel restart on a live session.

1. Batch scaling of one microbatch (fwd+bwd).
   Compute-bound: wall time scales ~linearly with batch.
   Launch-bound:  wall time is almost flat (same kernel-launch tax).

2. Side-stream GEMM overlap.
   Queue a calibrated pile of fat GEMMs on a second stream, then run
   the microbatch on the default stream.  If the GPU had idle SMs the
   two piles overlap and wall ≈ max(T_step, T_gemm).  If the SMs were
   already busy, wall ≈ T_step + T_gemm.

3. nvidia-smi SM utilisation during one full optimizer step (all
   accumulation microbatches).  High util = GPU-busy; low = launch-idle.

A PyTorch profiler dump is printed as supporting evidence (kernel count,
median kernel width, Self-CUDA / wall).  It is not part of the vote
because operator-vs-kernel double-counting is version-fragile.

This module is meant to be run from the d384 CfC notebook after the
model is built.  Interrupt the training cell first.

    from cfc_step_bottleneck_profile import run_bottleneck_profile
    run_bottleneck_profile(...)
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import threading
import time
from typing import Callable, Iterable


def _median(xs):
    ys = sorted(xs)
    n = len(ys)
    if n == 0:
        return float('nan')
    if n % 2:
        return ys[n // 2]
    return 0.5 * (ys[n // 2 - 1] + ys[n // 2])


def _sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_cuda(fn, n_warmup=1, n_repeat=3):
    for _ in range(n_warmup):
        fn()
        _sync()
    samples = []
    for _ in range(n_repeat):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        samples.append(time.perf_counter() - t0)
    return _median(samples), samples


def _classify_scale(ratio_hi_over_lo, hi, lo):
    """Expected ratio is hi/lo if compute-linear, ~1 if launch-flat."""
    expected = hi / lo
    # How much of the expected linear slope did we actually see?
    slope = (ratio_hi_over_lo - 1.0) / (expected - 1.0) if expected > 1 else 0.0
    if slope >= 0.70:
        return 'compute', slope
    if slope <= 0.25:
        return 'launch', slope
    return 'mixed', slope


def _classify_overlap(efficiency):
    # 1.0 = perfect overlap = idle SMs = launch-bound
    # 0.0 = no overlap      = busy SMs  = compute-bound
    if efficiency <= 0.25:
        return 'compute'
    if efficiency >= 0.65:
        return 'launch'
    return 'mixed'


def _classify_util(mean_util):
    if mean_util >= 70:
        return 'compute'
    if mean_util <= 35:
        return 'launch'
    return 'mixed'


def _poll_smi(samples, stop_evt, interval=0.05):
    while not stop_evt.is_set():
        try:
            out = subprocess.check_output(
                ['nvidia-smi',
                 '--query-gpu=utilization.gpu,utilization.memory',
                 '--format=csv,noheader,nounits'],
                text=True, timeout=2,
            ).strip().splitlines()[0]
            gpu_u, mem_u = [float(x.strip()) for x in out.split(',')[:2]]
            samples.append((gpu_u, mem_u))
        except Exception:
            return
        stop_evt.wait(interval)


def _parse_self_cuda_seconds(table: str):
    m = re.search(
        r'Self (?:CUDA|device) time total:\s*([0-9.]+)\s*(us|ms|s)',
        table, flags=re.I,
    )
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2).lower()
    return val * {'us': 1e-6, 'ms': 1e-3, 's': 1.0}[unit]


def _calibrate_gemm_count(torch, device, target_s, dim=4096):
    """Find N such that N sequential GEMMs take ~target_s."""
    a = torch.randn(dim, dim, device=device, dtype=torch.float32)
    b = torch.randn(dim, dim, device=device, dtype=torch.float32)
    def one():
        return a @ b
    _sync()
    # cheap estimate from 4 GEMMs
    t0 = time.perf_counter()
    for _ in range(4):
        one()
    _sync()
    per = max((time.perf_counter() - t0) / 4.0, 1e-4)
    n = max(4, int(math.ceil(target_s / per)))
    # cap so we don't enqueue tens of thousands of tiny launches
    n = min(n, 400)
    return a, b, n, per


def _enqueue_gemms(torch, a, b, n, stream):
    with torch.cuda.stream(stream):
        acc = a
        for _ in range(n):
            acc = acc @ b
        # keep a live consumer so the compiler cannot DCE the pile
        stream_result = acc.sum()
    return stream_result


def run_bottleneck_profile(
    model,
    *,
    forward_fn: Callable,
    make_batch: Callable[[int], tuple],
    batch_size: int,
    grad_accum: int,
    batch_sizes: Iterable[int] | None = None,
    n_warmup: int = 1,
    n_repeat: int = 3,
    device: str = 'cuda',
    profile_kernels: bool = True,
):
    """Run the three measurements on a live model.  Does not step the optimizer."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError('This profile needs a CUDA GPU; CPU times cannot '
                           'distinguish compute-bound from launch-bound.')

    model.train()
    sizes = list(batch_sizes) if batch_sizes is not None else [1, 2, batch_size]
    sizes = sorted({s for s in sizes if 1 <= s <= batch_size})
    if batch_size not in sizes:
        sizes.append(batch_size)
        sizes.sort()

    cached = {}
    for bs in sizes:
        x, y = make_batch(bs)
        cached[bs] = (x.detach().clone(), y.detach().clone())

    def microbatch(bs, zero=True):
        x, y = cached[bs]
        loss = forward_fn(x, y)
        loss.backward()
        if zero:
            model.zero_grad(set_to_none=True)
        return loss

    # ------------------------------------------------------------------
    print('=' * 64)
    print('CfC step bottleneck profile')
    cfg = getattr(model, 'cfg', None)
    d = getattr(cfg, 'd', '?')
    L = getattr(cfg, 'L', '?')
    integ = getattr(cfg, 'integrator', '?')
    print(f'  integrator={integ}  d={d}  L={L}  '
          f'bs={batch_size}  accum={grad_accum}')
    print('  interrupt training first; this does not step the optimizer')
    print('=' * 64)

    votes = []

    # -- [1] batch scaling ---------------------------------------------
    print('\n[1] Batch scaling (one microbatch, fwd+bwd)')
    scale = {}
    for bs in sizes:
        med, samples = _time_cuda(
            lambda bs=bs: microbatch(bs),
            n_warmup=n_warmup, n_repeat=n_repeat,
        )
        scale[bs] = med
        extra = ''
        if sizes[0] in scale and bs != sizes[0]:
            extra = f'   ({bs}/{sizes[0]} = {med / scale[sizes[0]]:.2f})'
        samp = ' '.join(f'{s:.2f}' for s in samples)
        print(f'  bs={bs:<3d}  median {med:6.2f}s  samples [{samp}]{extra}')

    lo, hi = sizes[0], sizes[-1]
    ratio = scale[hi] / scale[lo] if scale[lo] > 0 else float('inf')
    scale_cls, slope = _classify_scale(ratio, hi, lo)
    votes.append(scale_cls)
    print(f'  linear slope captured: {slope:.2f}  '
          f'(1.0 = time ∝ batch,  0.0 = time flat)')
    print(f'  => {scale_cls.upper()}')

    # -- [2] side-stream GEMM overlap ----------------------------------
    print('\n[2] Side-stream GEMM overlap (bs={})'.format(batch_size))
    T_step = scale[batch_size]
    a, b, n_gemm, per = _calibrate_gemm_count(torch, device, T_step)
    print(f'  calibrated {n_gemm} x {a.shape[0]} GEMMs  (~{per*1000:.1f} ms each)')

    def gemm_pile():
        side = torch.cuda.Stream()
        res = _enqueue_gemms(torch, a, b, n_gemm, side)
        torch.cuda.current_stream().wait_stream(side)
        return res

    T_gemm, _ = _time_cuda(gemm_pile, n_warmup=0, n_repeat=2)
    print(f'  T_gemm alone  {T_gemm:.2f}s')

    def both():
        side = torch.cuda.Stream()
        res = _enqueue_gemms(torch, a, b, n_gemm, side)
        microbatch(batch_size)
        torch.cuda.current_stream().wait_stream(side)
        return res

    T_both, _ = _time_cuda(both, n_warmup=0, n_repeat=2)
    overlap = (T_step + T_gemm - T_both) / min(T_step, T_gemm)
    overlap = max(0.0, min(1.2, overlap))  # clamp measurement noise
    overlap_cls = _classify_overlap(overlap)
    votes.append(overlap_cls)
    print(f'  T_step={T_step:.2f}s  T_gemm={T_gemm:.2f}s  T_both={T_both:.2f}s')
    print(f'  overlap efficiency = {overlap:.2f}  '
          f'(1.0 = full overlap / idle SMs,  0.0 = no spare SMs)')
    print(f'  => {overlap_cls.upper()}')

    del a, b

    # -- [3] nvidia-smi during a full optimizer step -------------------
    print(f'\n[3] nvidia-smi during one full step ({grad_accum} microbatches)')
    smi = []
    stop_evt = threading.Event()
    th = threading.Thread(target=_poll_smi, args=(smi, stop_evt), daemon=True)
    th.start()
    time.sleep(0.15)  # let the first samples land before we start

    def full_step():
        model.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            microbatch(batch_size, zero=False)
        model.zero_grad(set_to_none=True)

    _sync()
    t0 = time.perf_counter()
    full_step()
    _sync()
    T_full = time.perf_counter() - t0
    stop_evt.set()
    th.join(timeout=2.0)

    if smi:
        gpu_u = [g for g, _ in smi]
        mem_u = [m for _, m in smi]
        mean_u = sum(gpu_u) / len(gpu_u)
        p50_u = _median(gpu_u)
        p90_u = sorted(gpu_u)[max(0, int(0.9 * (len(gpu_u) - 1)))]
        util_cls = _classify_util(mean_u)
        votes.append(util_cls)
        print(f'  wall {T_full:.1f}s   samples={len(smi)}  '
              f'GPU util mean {mean_u:.0f}%  p50 {p50_u:.0f}%  p90 {p90_u:.0f}%  '
              f'mem-util mean {sum(mem_u)/len(mem_u):.0f}%')
        print(f'  => {util_cls.upper()}')
    else:
        mean_u = None
        util_cls = None
        print('  nvidia-smi unavailable; skipping this vote')
        print(f'  wall {T_full:.1f}s for the full step '
              f'(vs training-log ~{22.0 if grad_accum >= 8 else float("nan"):.1f}s)')

    # -- [4] profiler dump (not a vote) --------------------------------
    prof_busy = None
    n_kernels = None
    if profile_kernels:
        print('\n[4] Profiler dump (one microbatch, supporting only)')
        try:
            from torch.profiler import ProfilerActivity, profile
            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
            _sync()
            t0 = time.perf_counter()
            with profile(activities=activities, record_shapes=False,
                         with_stack=False) as prof:
                microbatch(batch_size)
                _sync()
            wall = time.perf_counter() - t0
            sort_key = 'self_cuda_time_total'
            try:
                table = prof.key_averages().table(
                    sort_by=sort_key, row_limit=15)
            except Exception:
                table = prof.key_averages().table(
                    sort_by='self_device_time_total', row_limit=15)
            self_cuda = _parse_self_cuda_seconds(table)
            if self_cuda is not None and wall > 0:
                prof_busy = self_cuda / wall
                print(f'  wall {wall:.2f}s   Self-CUDA {self_cuda:.2f}s   '
                      f'busy={prof_busy:.0%}')
            # kernel-width sketch from per-event cuda time when present
            widths = []
            try:
                for e in prof.events():
                    dt = (getattr(e, 'device_time', 0)
                          or getattr(e, 'cuda_time', 0) or 0)
                    if dt > 0:
                        widths.append(dt)  # already microseconds in older API
            except Exception:
                widths = []
            if widths:
                n_kernels = len(widths)
                print(f'  events-with-device-time={n_kernels}  '
                      f'median { _median(widths):.0f} us  '
                      f'mean {sum(widths)/len(widths):.0f} us')
            print(table)
        except Exception as exc:
            print(f'  profiler skipped: {type(exc).__name__}: {exc}')

    # -- verdict -------------------------------------------------------
    counts = {k: votes.count(k) for k in ('compute', 'launch', 'mixed')}
    if counts['compute'] > counts['launch'] and counts['compute'] >= 2:
        verdict = 'compute'
    elif counts['launch'] > counts['compute'] and counts['launch'] >= 2:
        verdict = 'launch'
    else:
        verdict = 'mixed'

    print('\n' + '=' * 64)
    if verdict == 'compute':
        print('VERDICT: (1) GPU-compute-bound')
        print('  The SMs are busy.  H100 extra FLOPs / HBM bandwidth can')
        print('  plausibly cut wall time by ~1.5-2.5x at the same 4x8.')
        print('  Do not chase bs=8; the card is the same 80 GB class.')
    elif verdict == 'launch':
        print('VERDICT: (2) CPU / launch-bound')
        print('  The GPU is idle between many small kernels.  A faster')
        print('  GPU buys almost nothing (maybe 1.0-1.2x).  The real')
        print('  levers are fewer sequential launches: cut GRAD_ACCUM,')
        print('  shrink ANISO_RANK, or fuse the per-layer Python step.')
    else:
        print('VERDICT: mixed — both (1) and (2) contribute')
        print('  H100 may help, but not by 2x.  Expect something closer')
        print('  to 1.2-1.6x unless launches are also reduced.')
    print(f'  votes: {votes}')
    print('=' * 64)

    model.zero_grad(set_to_none=True)
    return {
        'verdict': verdict,
        'votes': votes,
        'scale_s': scale,
        'scale_slope': slope,
        'overlap': overlap,
        'T_step': T_step,
        'T_gemm': T_gemm,
        'T_both': T_both,
        'T_full': T_full,
        'smi_mean_util': mean_u,
        'profiler_busy': prof_busy,
        'n_kernels': n_kernels,
    }


def run_from_notebook():
    """Convenience entry point using the d384 notebook's globals."""
    import torch
    g = None
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None:
            g = ip.user_ns
    except Exception:
        g = globals()
    if g is None:
        g = globals()

    needed = ['model', 'forward_with_vreg', 'get_batch', 'train_ids',
              'BATCH_SIZE', 'BLOCK_SIZE', 'GRAD_ACCUM', 'DEVICE',
              'LAMBDA_V', 'LAMBDA_FOCK_REG', 'FOCK_REG_EPS']
    missing = [k for k in needed if k not in g]
    if missing:
        raise RuntimeError(
            'Notebook globals missing: ' + ', '.join(missing) +
            '. Run this after Cell 5 (model built) and interrupt training first.'
        )

    import numpy as np
    rng = np.random.default_rng(12345)
    model = g['model']
    register_repulsion = bool(g.get('REGISTER_REPULSION', False))

    def forward_fn(x, y):
        loss, *_ = g['forward_with_vreg'](
            x, y, g['LAMBDA_V'], g['LAMBDA_FOCK_REG'], g['FOCK_REG_EPS'])
        if register_repulsion:
            loss = loss + model.pop_repulsion_loss()
        return loss

    def make_batch(bs):
        xb, yb = g['get_batch'](g['train_ids'], bs, g['BLOCK_SIZE'], rng)
        return (torch.from_numpy(xb).to(g['DEVICE']),
                torch.from_numpy(yb).to(g['DEVICE']))

    return run_bottleneck_profile(
        model,
        forward_fn=forward_fn,
        make_batch=make_batch,
        batch_size=int(g['BATCH_SIZE']),
        grad_accum=int(g['GRAD_ACCUM']),
        device=str(g['DEVICE']),
    )


if __name__ == '__main__':
    run_from_notebook()
