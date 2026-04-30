r"""Wall-clock and FLOP benchmark for SPLM vs the matched attention baseline.

Phase 3 of the E8 inference-efficiency protocol. Compares per-token AR
decode cost (wall-clock seconds, FLOPs) for four modes at increasing
context length T:

    1. SPLM full-forward per token   (re-run the entire forward each step)
    2. SPLM streaming-xi per token    (the operationalisation of A2)
    3. ATTN full-forward per token   (re-run the entire forward each step)
    4. ATTN KV-cached per token       (the production-realistic path)

For each mode and T, the benchmark
    - decodes 32 tokens starting from a length-(T - 32) prompt drawn
      from the Tiny Shakespeare validation slice,
    - reports the median per-token wall-clock,
    - records the analytical FLOP count from `flop_counter`,
    - records the realised speedup of streaming/KV-cached over the
      respective full-forward path.

Outputs:
    results/inference_benchmark/wall_clock.json
    results/inference_benchmark/figures/ms_per_token_vs_T.png
    results/inference_benchmark/figures/flops_per_token_vs_T.png

Decision rule per E8 §3.4: empirical wall-clock crossover T_wall is
recorded; the FLOP crossover T_FLOP is computed analytically from
`flop_counter.crossover_T`. The paper section is updated with both.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tiktoken
import torch

SCRIPT_DIR = Path(__file__).parent
PARENT_DIR = SCRIPT_DIR.parent

sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(PARENT_DIR / "energetic_minima"))
sys.path.insert(0, str(PARENT_DIR / "sarf_mass_variant"))
sys.path.insert(0, str(PARENT_DIR / "first_order_ablation"))

from data_module import load_tiny_shakespeare  # noqa: E402
from matched_baseline_model import MatchedConfig, MatchedGPT  # noqa: E402
from energetic_minima.model_ln import (  # noqa: E402
    ScalarPotentialLMSARFMassLN, SPLMSARFMassLNConfig,
)
from model_first_order import (  # noqa: E402
    ScalarPotentialLMFirstOrder, SPLMFirstOrderConfig,
)
from splm_streaming_decode import StreamingState, streaming_step  # noqa: E402
from flop_counter import (  # noqa: E402
    SPLMFLOPParams, AttnFLOPParams,
    splm_decode_token_flops, splm_decode_full_token_flops,
    attn_decode_token_flops, attn_forward_flops,
    crossover_T,
)


def _pick_device(force=None):
    if force:
        return force
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        try:
            torch.zeros(1, device="mps")
            return "mps"
        except Exception:
            pass
    return "cpu"


def _sync(device):
    """Synchronize so wall-clock measurements are correct across devices."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def load_splm(ckpt_path: Path, device: torch.device):
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = blob["model_cfg"]
    variant = blob.get("variant", "")
    if variant == "sarf_mass_ln":
        cfg = SPLMSARFMassLNConfig(**cfg_dict)
        model = ScalarPotentialLMSARFMassLN(cfg)
    elif variant == "splm_first_order":
        cfg = SPLMFirstOrderConfig(**cfg_dict)
        model = ScalarPotentialLMFirstOrder(cfg)
    else:
        raise ValueError(f"unknown variant {variant!r}")
    model.load_state_dict(blob["model_state_dict"], strict=False)
    return model.to(device).eval(), variant, blob


def load_attn(ckpt_path: Path, device: torch.device):
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = blob["model_cfg"]
    cfg = MatchedConfig(**cfg_dict)
    model = MatchedGPT(cfg)
    model.load_state_dict(blob["model_state_dict"], strict=False)
    return model.to(device).eval(), blob


def get_prompt_ids(target_T: int) -> list[int]:
    """Build a length-`target_T` prompt from the Tiny Shakespeare val slice.

    Picks tokens deterministically starting from a fixed offset so all
    benchmark cells use the same input.
    """
    _, val_ids = load_tiny_shakespeare()
    if target_T > len(val_ids):
        raise ValueError(
            f"target_T={target_T} exceeds val_ids length {len(val_ids)}"
        )
    return list(val_ids[100:100 + target_T].astype(int))


# -------------------------------------------------------------------------
# Per-mode timing functions: each returns the (per-token wall-clock seconds)
# averaged over `n_decode` decoded tokens following a length-(T_prompt) prompt.
# All modes use deterministic argmax to avoid sampling noise in timing.
# -------------------------------------------------------------------------


def time_splm_full_forward(model, prompt_ids: list[int], n_decode: int,
                           device: torch.device, T_max: int) -> float:
    """SPLM, re-run full model() for every new token."""
    if T_max + n_decode > model.cfg.max_len:
        return float("nan")
    seq = list(prompt_ids)
    times: list[float] = []
    for _ in range(n_decode):
        x = torch.tensor([seq[-T_max:]], dtype=torch.long, device=device)
        _sync(device)
        t0 = time.perf_counter()
        with torch.enable_grad():
            out = model(x)
            logits = out[0]
        next_id = int(logits[0, -1].argmax().item())
        _sync(device)
        times.append(time.perf_counter() - t0)
        seq.append(next_id)
    return float(np.median(times))


def time_splm_streaming(model, prompt_ids: list[int], n_decode: int,
                        device: torch.device) -> float:
    """SPLM with streaming-xi cache."""
    if len(prompt_ids) + n_decode > model.cfg.max_len:
        return float("nan")
    state = StreamingState.init(model, device)
    last_h: torch.Tensor = None  # type: ignore
    for tok in prompt_ids:
        state, last_h = streaming_step(model, state, int(tok))

    times: list[float] = []
    for _ in range(n_decode):
        _sync(device)
        t0 = time.perf_counter()
        logits = last_h @ model.E.weight.T
        next_id = int(logits.argmax().item())
        state, last_h = streaming_step(model, state, next_id)
        _sync(device)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def time_attn_full_forward(model, prompt_ids: list[int], n_decode: int,
                           device: torch.device, T_max: int) -> float:
    """ATTN, re-run full forward for every new token."""
    if T_max + n_decode > model.cfg.max_len:
        return float("nan")
    seq = list(prompt_ids)
    times: list[float] = []
    for _ in range(n_decode):
        x = torch.tensor([seq[-T_max:]], dtype=torch.long, device=device)
        _sync(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(x)
            logits = out[0]
        next_id = int(logits[0, -1].argmax().item())
        _sync(device)
        times.append(time.perf_counter() - t0)
        seq.append(next_id)
    return float(np.median(times))


def time_attn_kv_cached(model, prompt_ids: list[int], n_decode: int,
                        device: torch.device) -> float:
    """ATTN with KV cache. Prefill the prompt then decode token-by-token."""
    if len(prompt_ids) + n_decode > model.cfg.max_len:
        return float("nan")
    n_layer = model.cfg.n_layer
    caches: list = [None] * n_layer
    with torch.no_grad():
        x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        out = model(x, kv_caches=caches, position_offset=0)
        logits, _, caches = out
        next_id = int(logits[0, -1].argmax().item())

    pos = len(prompt_ids)
    times: list[float] = []
    for _ in range(n_decode):
        _sync(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            x = torch.tensor([[next_id]], dtype=torch.long, device=device)
            out = model(x, kv_caches=caches, position_offset=pos)
            logits, _, caches = out
            next_id = int(logits[0, -1].argmax().item())
        _sync(device)
        times.append(time.perf_counter() - t0)
        pos += 1
    return float(np.median(times))


@dataclass
class BenchmarkResult:
    T: int
    splm_full_ms: float
    splm_stream_ms: float
    attn_full_ms: float
    attn_kv_ms: float
    splm_full_flops: int
    splm_stream_flops: int
    attn_full_flops: int
    attn_kv_flops: int


def run_benchmark(splm_model, attn_model, p_splm, p_attn,
                  T_values: list[int], n_decode: int,
                  device: torch.device, n_warmup: int = 2) -> list[BenchmarkResult]:
    """Run the four-mode wall-clock + FLOP comparison at a list of T values."""
    results: list[BenchmarkResult] = []
    for T in T_values:
        if T > splm_model.cfg.max_len - n_decode:
            print(f"[bench] skipping T={T} (exceeds max_len {splm_model.cfg.max_len})")
            continue
        prompt = get_prompt_ids(T)
        print(f"[bench] T={T}  warmup...")
        for _ in range(n_warmup):
            time_splm_streaming(splm_model, prompt[:max(T // 2, 8)], 4, device)
            time_attn_kv_cached(attn_model, prompt[:max(T // 2, 8)], 4, device)

        print(f"[bench] T={T}  measuring...")
        splm_full = time_splm_full_forward(splm_model, prompt, n_decode, device, T)
        splm_stream = time_splm_streaming(splm_model, prompt, n_decode, device)
        attn_full = time_attn_full_forward(attn_model, prompt, n_decode, device, T)
        attn_kv = time_attn_kv_cached(attn_model, prompt, n_decode, device)

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
        print(f"[bench] T={T:5d}  "
              f"SPLM_full={r.splm_full_ms:7.1f} ms  "
              f"SPLM_stream={r.splm_stream_ms:7.1f} ms  "
              f"ATTN_full={r.attn_full_ms:7.1f} ms  "
              f"ATTN_kv={r.attn_kv_ms:7.1f} ms")
    return results


def find_wall_clock_crossover(results: list[BenchmarkResult]) -> int | None:
    """Smallest T where SPLM_stream <= ATTN_kv (in wall-clock ms)."""
    for r in results:
        if r.splm_stream_ms <= r.attn_kv_ms:
            return r.T
    return None


def make_figures(results: list[BenchmarkResult], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    Ts = np.array([r.T for r in results])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(Ts, [r.splm_stream_ms for r in results], "o-",
            color="tab:blue", lw=2, label="SPLM streaming-xi")
    ax.plot(Ts, [r.attn_kv_ms for r in results], "s-",
            color="tab:green", lw=2, label="ATTN KV-cached")
    ax.plot(Ts, [r.splm_full_ms for r in results], "v--",
            color="tab:blue", alpha=0.5, label="SPLM full-forward")
    ax.plot(Ts, [r.attn_full_ms for r in results], "v--",
            color="tab:green", alpha=0.5, label="ATTN full-forward")
    ax.set_xlabel("context length T (tokens)")
    ax.set_ylabel("wall-clock per generated token (ms)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("E8 wall-clock per-token decode cost vs context length")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    cross = find_wall_clock_crossover(results)
    if cross is not None:
        ax.axvline(cross, ls=":", color="firebrick", alpha=0.6,
                   label=f"empirical crossover T={cross}")
    fig.tight_layout()
    fig.savefig(out_dir / "ms_per_token_vs_T.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(Ts, [r.splm_stream_flops / 1e6 for r in results], "o-",
            color="tab:blue", lw=2, label="SPLM streaming-xi")
    ax.plot(Ts, [r.attn_kv_flops / 1e6 for r in results], "s-",
            color="tab:green", lw=2, label="ATTN KV-cached")
    ax.set_xlabel("context length T (tokens)")
    ax.set_ylabel("FLOPs per generated token (analytical, log10 MFLOPs)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("E8 analytical FLOPs per-token decode cost vs context length")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "flops_per_token_vs_T.png", dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splm-ckpt", required=True,
                    help="Path to SPLM ckpt_latest.pt (em_ln or first-order)")
    ap.add_argument("--attn-ckpt", required=True,
                    help="Path to MatchedGPT ckpt_latest.pt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-decode", type=int, default=16,
                    help="number of tokens to decode for timing")
    ap.add_argument("--T-values", default="32,64,128,200")
    ap.add_argument("--output-dir", default="results/inference_benchmark")
    args = ap.parse_args()

    device = torch.device(_pick_device(args.device))
    print(f"[bench] device={device}")

    splm_model, splm_variant, _ = load_splm(Path(args.splm_ckpt), device)
    attn_model, _ = load_attn(Path(args.attn_ckpt), device)
    print(f"[bench] SPLM variant={splm_variant}  "
          f"d={splm_model.cfg.d}  L={splm_model.cfg.L}  "
          f"v_hidden={splm_model.cfg.v_hidden}  max_len={splm_model.cfg.max_len}")
    print(f"[bench] ATTN d={attn_model.cfg.d}  L={attn_model.cfg.n_layer}  "
          f"n_head={attn_model.cfg.n_head}  max_len={attn_model.cfg.max_len}")

    p_splm = SPLMFLOPParams(
        d=splm_model.cfg.d, L=splm_model.cfg.L,
        v_hidden=splm_model.cfg.v_hidden, v_depth=splm_model.cfg.v_depth,
        vocab_size=splm_model.cfg.vocab_size,
        ln_after_step=getattr(splm_model.cfg, "ln_after_step", False),
    )
    p_attn = AttnFLOPParams(
        d=attn_model.cfg.d, L=attn_model.cfg.n_layer,
        n_head=attn_model.cfg.n_head, mlp_mult=attn_model.cfg.mlp_mult,
        vocab_size=attn_model.cfg.vocab_size,
    )

    flop_cross = crossover_T(p_splm, p_attn)
    print(f"[bench] analytical FLOP crossover: T*={flop_cross.get('T_crossover')}")

    T_values = [int(x) for x in args.T_values.split(",")]
    results = run_benchmark(
        splm_model, attn_model, p_splm, p_attn,
        T_values=T_values, n_decode=args.n_decode, device=device,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "device": str(device),
        "splm": {
            "d": splm_model.cfg.d,
            "L": splm_model.cfg.L,
            "v_hidden": splm_model.cfg.v_hidden,
            "v_depth": splm_model.cfg.v_depth,
            "max_len": splm_model.cfg.max_len,
            "variant": splm_variant,
        },
        "attn": {
            "d": attn_model.cfg.d,
            "L": attn_model.cfg.n_layer,
            "n_head": attn_model.cfg.n_head,
            "mlp_mult": attn_model.cfg.mlp_mult,
            "max_len": attn_model.cfg.max_len,
        },
        "T_FLOP_crossover": flop_cross.get("T_crossover"),
        "T_wall_crossover": find_wall_clock_crossover(results),
        "results": [vars(r) for r in results],
    }
    with open(out_dir / "wall_clock.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[bench] payload written to {out_dir/'wall_clock.json'}")

    make_figures(results, out_dir / "figures")
    print(f"[bench] figures written to {out_dir/'figures'}")


if __name__ == "__main__":
    main()
