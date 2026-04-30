r"""Long-context wall-clock benchmark for SPLM vs matched attention baseline.

Companion to ``run_inference_benchmark.py``. The trained checkpoints both
have ``max_len = 256``, which is below the analytical FLOP crossover
``T* = 8092``.  This script extrapolates the existing trained checkpoints
to longer contexts to empirically observe (or refute) the wall-clock
crossover.

Caveats made explicit:

1.  **Both models have a learned (256, 128) positional table ``P``.**
    We extend each by *tiling* the trained 256 rows cyclically to fill
    ``new_max_len`` rows.  The resulting models produce semantically
    incorrect output at T > 256 (the position embedding rows past 256
    are repeats of trained-distribution rows, not coherent positional
    encodings), but the per-step compute graph (FLOPs, memory
    accesses, kernel dispatch overhead) is *architecturally faithful*
    -- which is the relevant comparison for §A2's wall-clock crossover
    claim.  SPLM additionally aggregates context via the cumulative
    mean ``xi_t = (1/t) sum_{s<=t} h_s`` which trivially extends to
    any T; tiling P is purely a model-loading concession to the
    trained checkpoint's finite ``max_len = 256``.

2.  **Wall-clock measurements are CPU fp32.**  This is the regime in
    which both architectures' per-step compute cost is dominated by
    matmul throughput rather than dispatch overhead, and is therefore
    the regime in which the FLOP-counter prediction
    ``T*_FLOP = 8092`` should translate most cleanly into a wall-clock
    crossover.

Outputs:

    results/inference_benchmark_longctx/wall_clock.json
    results/inference_benchmark_longctx/figures/ms_per_token_vs_T.png
    results/inference_benchmark_longctx/figures/flops_per_token_vs_T.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_inference_benchmark import (  # noqa: E402
    BenchmarkResult, _pick_device, _sync,
    load_splm, load_attn, get_prompt_ids,
    time_splm_full_forward, time_splm_streaming,
    time_attn_full_forward, time_attn_kv_cached,
)
from flop_counter import (  # noqa: E402
    SPLMFLOPParams, AttnFLOPParams,
    splm_decode_token_flops, splm_decode_full_token_flops,
    attn_decode_token_flops, attn_forward_flops,
    crossover_T,
)


def extend_attn_position_embedding(
    attn_model, new_max_len: int, mode: str = "tile"
) -> None:
    """Resize MatchedGPT.P from (cfg.max_len, d) to (new_max_len, d).

    The trained ``P`` has shape ``(cfg.max_len, d)`` (with
    ``cfg.max_len = 256`` for our checkpoints).  To run at T > 256 we
    must produce a row of P for every position 0..new_max_len-1.

    Two strategies:

    -   ``"tile"``:  repeat the trained 256 rows cyclically.  Produces
        gibberish output at T > 256 but preserves the magnitude
        distribution of the embedding (so layer norms etc behave
        normally) and has no memory overhead beyond ``new_max_len * d``.
    -   ``"zero"``:  zero-pad with rows of zeros for positions 256+.
        Output is closer to "input has no positional information past
        256", which can produce numerical anomalies in the first
        few layers.

    For the wall-clock measurement these are equivalent: the per-step
    compute pattern only depends on shapes, not on values.  We default
    to ``"tile"``.
    """
    if new_max_len <= attn_model.cfg.max_len:
        return
    P_old = attn_model.P.data  # (max_len_old, d)
    max_len_old, d = P_old.shape
    P_new = torch.zeros(new_max_len, d, dtype=P_old.dtype, device=P_old.device)
    if mode == "tile":
        n_repeats = (new_max_len + max_len_old - 1) // max_len_old
        P_tiled = P_old.repeat(n_repeats, 1)[:new_max_len]
        P_new.copy_(P_tiled)
    elif mode == "zero":
        P_new[:max_len_old].copy_(P_old)
    else:
        raise ValueError(f"unknown extension mode {mode!r}")
    attn_model.P = torch.nn.Parameter(P_new, requires_grad=False)
    attn_model.cfg.max_len = new_max_len


def extend_splm_max_len(splm_model, new_max_len: int, mode: str = "tile") -> None:
    """Extend SPLM's learned positional embedding table.

    The ``ScalarPotentialLMSARFMass`` family DOES have a learned
    ``self.P`` parameter of shape ``(cfg.max_len, d)``, despite also
    using the cumulative-mean ``xi`` for context pooling.  The two
    operate in parallel: ``P`` injects per-position offsets at the
    embedding layer, while ``xi`` aggregates context across the
    integration trajectory.  For long-T inference we extend ``P`` the
    same way as MatchedGPT (tile, default).
    """
    if new_max_len <= splm_model.cfg.max_len:
        return
    if hasattr(splm_model, "P") and isinstance(splm_model.P,
                                               torch.nn.Parameter):
        P_old = splm_model.P.data
        max_len_old, d = P_old.shape
        P_new = torch.zeros(new_max_len, d, dtype=P_old.dtype,
                            device=P_old.device)
        if mode == "tile":
            n_repeats = (new_max_len + max_len_old - 1) // max_len_old
            P_new.copy_(P_old.repeat(n_repeats, 1)[:new_max_len])
        elif mode == "zero":
            P_new[:max_len_old].copy_(P_old)
        else:
            raise ValueError(f"unknown extension mode {mode!r}")
        splm_model.P = torch.nn.Parameter(P_new, requires_grad=False)
    splm_model.cfg.max_len = new_max_len


def get_long_prompt_ids(target_T: int) -> list[int]:
    """Return a length-target_T prompt by tiling the Tiny Shakespeare val
    slice if needed (the val slice itself has only a few k tokens).
    """
    from data_module import load_tiny_shakespeare  # noqa: E402

    _, val_ids = load_tiny_shakespeare()
    val_ids = list(map(int, val_ids))
    if target_T > len(val_ids):
        n_repeats = (target_T + len(val_ids) - 1) // len(val_ids)
        val_ids = val_ids * n_repeats
    return val_ids[100:100 + target_T]


def run_longctx_benchmark(
    splm_model, attn_model, p_splm, p_attn,
    T_values: list[int], n_decode: int, device: torch.device,
    n_warmup: int = 1, skip_full_threshold: int = 4096,
) -> list[BenchmarkResult]:
    """Long-context variant of the benchmark.

    Differences from the short-context script:

    -   At T > ``skip_full_threshold``, the SPLM_full and ATTN_full
        modes are skipped (return NaN).  These full-forward modes are
        already known empirically to grow linearly in T from the
        short-context benchmark; the long-context experiment only needs
        the streaming/KV-cached comparison to demonstrate the
        crossover.
    -   ``n_warmup`` defaults to 1 (vs 2 in the short-context run) for
        speed, since long-T forwards take seconds.
    -   Uses ``get_long_prompt_ids`` (tiles val_ids) instead of
        ``get_prompt_ids`` (raises if target_T exceeds val_ids).
    """
    results: list[BenchmarkResult] = []
    for T in T_values:
        if T > splm_model.cfg.max_len - n_decode:
            print(f"[bench] skipping T={T} (exceeds max_len {splm_model.cfg.max_len})")
            continue
        prompt = get_long_prompt_ids(T)
        do_full = T <= skip_full_threshold

        print(f"[bench] T={T}  warmup ({n_warmup} rounds)...")
        for _ in range(n_warmup):
            time_splm_streaming(splm_model, prompt[:max(T // 2, 8)], 4, device)
            time_attn_kv_cached(attn_model, prompt[:max(T // 2, 8)], 4, device)

        print(f"[bench] T={T}  measuring (full={'on' if do_full else 'off'})...")
        if do_full:
            t0 = time.perf_counter()
            splm_full = time_splm_full_forward(splm_model, prompt, n_decode, device, T)
            print(f"[bench]   SPLM_full done in {time.perf_counter() - t0:.1f}s")
        else:
            splm_full = float("nan")

        t0 = time.perf_counter()
        splm_stream = time_splm_streaming(splm_model, prompt, n_decode, device)
        print(f"[bench]   SPLM_stream done in {time.perf_counter() - t0:.1f}s")

        if do_full:
            t0 = time.perf_counter()
            attn_full = time_attn_full_forward(attn_model, prompt, n_decode, device, T)
            print(f"[bench]   ATTN_full done in {time.perf_counter() - t0:.1f}s")
        else:
            attn_full = float("nan")

        t0 = time.perf_counter()
        attn_kv = time_attn_kv_cached(attn_model, prompt, n_decode, device)
        print(f"[bench]   ATTN_kv done in {time.perf_counter() - t0:.1f}s")

        r = BenchmarkResult(
            T=T,
            splm_full_ms=1000 * splm_full,
            splm_stream_ms=1000 * splm_stream,
            attn_full_ms=1000 * attn_full,
            attn_kv_ms=1000 * attn_kv,
            splm_full_flops=splm_decode_full_token_flops(p_splm, T)["per_token"],
            splm_stream_flops=splm_decode_token_flops(p_splm, T)["per_token"],
            attn_full_flops=attn_forward_flops(p_attn, T)["total"],
            attn_kv_flops=attn_decode_token_flops(p_attn, T)["per_token"],
        )
        results.append(r)
        print(f"[bench] T={T:6d}  "
              f"SPLM_full={r.splm_full_ms:8.1f} ms  "
              f"SPLM_stream={r.splm_stream_ms:7.1f} ms  "
              f"ATTN_full={r.attn_full_ms:8.1f} ms  "
              f"ATTN_kv={r.attn_kv_ms:7.1f} ms")
    return results


def find_wall_clock_crossover(results: list[BenchmarkResult]) -> int | None:
    """Smallest T where SPLM_stream <= ATTN_kv (wall-clock)."""
    for r in results:
        if r.splm_stream_ms <= r.attn_kv_ms:
            return r.T
    return None


def make_figures_long(results: list[BenchmarkResult], out_dir: Path,
                      crossover_FLOP: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    Ts = np.array([r.T for r in results])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(Ts, [r.splm_stream_ms for r in results], "o-",
            color="tab:blue", lw=2, ms=7, label="SPLM streaming-ξ (per token)")
    ax.plot(Ts, [r.attn_kv_ms for r in results], "s-",
            color="tab:green", lw=2, ms=7, label="ATTN KV-cached (per token)")
    full_Ts = [r.T for r in results if not np.isnan(r.splm_full_ms)]
    full_splm = [r.splm_full_ms for r in results if not np.isnan(r.splm_full_ms)]
    full_attn = [r.attn_full_ms for r in results if not np.isnan(r.attn_full_ms)]
    if full_Ts:
        ax.plot(full_Ts, full_splm, "v--", color="tab:blue", alpha=0.45,
                ms=5, label="SPLM full-forward (per call)")
        ax.plot(full_Ts, full_attn, "v--", color="tab:green", alpha=0.45,
                ms=5, label="ATTN full-forward (per call)")
    ax.set_xlabel("context length T (tokens)")
    ax.set_ylabel("wall-clock per generated token (ms)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(
        "E8 long-context wall-clock per-token decode cost vs T\n"
        "extending position embedding by tiling on MatchedGPT; "
        "SPLM has no learned positional encoding"
    )
    ax.grid(True, which="both", alpha=0.3)

    cross = find_wall_clock_crossover(results)
    if cross is not None:
        ax.axvline(cross, ls="-", color="firebrick", alpha=0.55, lw=1.5,
                   label=f"empirical wall-clock crossover T={cross}")
    ax.axvline(crossover_FLOP, ls=":", color="grey", alpha=0.6, lw=1.5,
               label=f"analytical FLOP crossover T*={crossover_FLOP}")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "ms_per_token_vs_T.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(Ts, [r.splm_stream_flops / 1e6 for r in results], "o-",
            color="tab:blue", lw=2, ms=7, label="SPLM streaming-ξ")
    ax.plot(Ts, [r.attn_kv_flops / 1e6 for r in results], "s-",
            color="tab:green", lw=2, ms=7, label="ATTN KV-cached")
    ax.set_xlabel("context length T (tokens)")
    ax.set_ylabel("FLOPs per generated token (analytical, MFLOPs, log scale)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("E8 long-context analytical FLOPs per-token decode cost vs T")
    ax.grid(True, which="both", alpha=0.3)
    ax.axvline(crossover_FLOP, ls=":", color="grey", alpha=0.6, lw=1.5,
               label=f"analytical FLOP crossover T*={crossover_FLOP}")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "flops_per_token_vs_T.png", dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splm-ckpt", required=True)
    ap.add_argument("--attn-ckpt", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-decode", type=int, default=8)
    ap.add_argument("--T-values", default="512,1024,2048,4096,8192,12288,16384,24576")
    ap.add_argument("--extend-max-len", type=int, default=32768,
                    help="extend both models' max_len up to this value")
    ap.add_argument("--skip-full-threshold", type=int, default=4096,
                    help="skip the SPLM_full and ATTN_full modes for T above this")
    ap.add_argument("--output-dir", default="results/inference_benchmark_longctx")
    args = ap.parse_args()

    device = torch.device(_pick_device(args.device))
    print(f"[bench] device: {device}")

    print(f"[bench] loading SPLM from {args.splm_ckpt}")
    splm_model, variant, splm_blob = load_splm(Path(args.splm_ckpt), device)
    print(f"[bench]   SPLM variant={variant!r}  d={splm_model.cfg.d}  "
          f"L={splm_model.cfg.L}  v_hidden={splm_model.cfg.v_hidden}  "
          f"max_len_trained={splm_model.cfg.max_len}")
    trained_splm_max_len = splm_model.cfg.max_len
    extend_splm_max_len(splm_model, args.extend_max_len, mode="tile")
    print(f"[bench]   SPLM max_len extended {trained_splm_max_len} -> "
          f"{splm_model.cfg.max_len} via P-table tile")

    print(f"[bench] loading MatchedGPT from {args.attn_ckpt}")
    attn_model, attn_blob = load_attn(Path(args.attn_ckpt), device)
    print(f"[bench]   ATTN d={attn_model.cfg.d}  L={attn_model.cfg.n_layer}  "
          f"n_head={attn_model.cfg.n_head}  "
          f"max_len_trained={attn_model.cfg.max_len}")
    trained_attn_max_len = attn_model.cfg.max_len
    extend_attn_position_embedding(attn_model, args.extend_max_len, mode="tile")
    print(f"[bench]   ATTN max_len extended {trained_attn_max_len} -> "
          f"{attn_model.cfg.max_len} via P-table tile")

    p_splm = SPLMFLOPParams(
        d=splm_model.cfg.d, v_hidden=splm_model.cfg.v_hidden,
        v_depth=splm_model.cfg.v_depth, L=splm_model.cfg.L,
    )
    p_attn = AttnFLOPParams(
        d=attn_model.cfg.d, L=attn_model.cfg.n_layer,
        n_head=attn_model.cfg.n_head, mlp_mult=attn_model.cfg.mlp_mult,
    )

    cross_FLOP_payload = crossover_T(p_splm, p_attn)
    cross_FLOP = int(cross_FLOP_payload["T_crossover"])
    print(f"[bench] analytical FLOP crossover T*={cross_FLOP}  "
          f"(SPLM_stream={cross_FLOP_payload['splm_streaming_per_token_flops_at_T']/1e6:.1f} MFLOPs, "
          f"ATTN_kv={cross_FLOP_payload['attn_kv_per_token_flops_at_T']/1e6:.1f} MFLOPs at T*)")

    T_values = [int(t) for t in args.T_values.split(",")]
    print(f"[bench] T_values={T_values}  n_decode={args.n_decode}  "
          f"skip_full_threshold={args.skip_full_threshold}")

    results = run_longctx_benchmark(
        splm_model, attn_model, p_splm, p_attn,
        T_values=T_values, n_decode=args.n_decode, device=device,
        n_warmup=1, skip_full_threshold=args.skip_full_threshold,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "device": str(device),
        "extend_max_len": args.extend_max_len,
        "trained_attn_max_len": trained_attn_max_len,
        "trained_splm_max_len_recorded": splm_blob.get("model_cfg", {}).get("max_len"),
        "splm_cfg": {
            "d": splm_model.cfg.d, "L": splm_model.cfg.L,
            "v_hidden": splm_model.cfg.v_hidden,
            "v_depth": splm_model.cfg.v_depth,
            "max_len_used": splm_model.cfg.max_len,
        },
        "attn_cfg": {
            "d": attn_model.cfg.d, "n_layer": attn_model.cfg.n_layer,
            "n_head": attn_model.cfg.n_head,
            "mlp_mult": attn_model.cfg.mlp_mult,
            "max_len_used": attn_model.cfg.max_len,
        },
        "T_FLOP_crossover": cross_FLOP_payload,
        "T_wall_crossover": find_wall_clock_crossover(results),
        "n_decode": args.n_decode,
        "results": [
            {
                "T": r.T,
                "splm_full_ms": r.splm_full_ms,
                "splm_stream_ms": r.splm_stream_ms,
                "attn_full_ms": r.attn_full_ms,
                "attn_kv_ms": r.attn_kv_ms,
                "splm_full_flops": r.splm_full_flops,
                "splm_stream_flops": r.splm_stream_flops,
                "attn_full_flops": r.attn_full_flops,
                "attn_kv_flops": r.attn_kv_flops,
            }
            for r in results
        ],
    }
    json_path = out_dir / "wall_clock.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[bench] payload written to {json_path}")

    make_figures_long(results, out_dir / "figures", cross_FLOP)
    print(f"[bench] figures written to {out_dir / 'figures'}")


if __name__ == "__main__":
    main()
