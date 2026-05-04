r"""Extract per-position lagged quadruples from a trained SPLM checkpoint.

This is **Phase 1** of E7 (the SPLM Markov-order test, pre-registered in
`docs/SPLM_inference_first_order_pre-registered_protocol.md`). The
companion downstream module is the unmodified
`notebooks/dynamics_order_test/markov_order_regression.py` -- this
extractor produces a `quadruples.npz` in exactly the same schema so the
regression module is drop-in.

Compared to `notebooks/dynamics_order_test/extract_lagged_quadruples.py`
(which targets HF causal LMs and uses a curated multi-domain corpus):

- Model loading switches on the `variant` field saved in each checkpoint
  (`splm_first_order` or `sarf_mass_ln`), rebuilds the right config dataclass,
  loads the state dict, and runs the SPLM forward with
  `return_trajectory=True`.
- Hidden state of interest is `traj_h[-1]` -- the **final integration step**
  hidden state, i.e. h^(L)_t. Per E7 protocol §4.1.
- Corpus is the Tiny-Shakespeare validation slice, decoded via the GPT-2
  BPE, sentence-segmented on `[.!?]+\s+`, filtered to >= 16 BPE tokens,
  and the first 100 sentences in document order are taken (E7 §3.1).
- SPLM forward calls `torch.autograd.grad` internally to compute
  -nabla V_theta inside the integrator, so the extractor must run under
  `torch.enable_grad()`; only the *output* tensor is detached (mirrors
  the smoke pattern in `energetic_minima/model_ln.py`).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import tiktoken
import torch

SCRIPT_DIR = Path(__file__).parent
ABLATION_DIR = SCRIPT_DIR.parent
CONS_ARCH_DIR = ABLATION_DIR.parent

sys.path.insert(0, str(CONS_ARCH_DIR))
sys.path.insert(0, str(CONS_ARCH_DIR / "energetic_minima"))
sys.path.insert(0, str(CONS_ARCH_DIR / "sarf_mass_variant"))
sys.path.insert(0, str(ABLATION_DIR))

from data_module import load_tiny_shakespeare  # noqa: E402

# Model classes (lazy-imported because building one of them imports torch
# and reads the surprisal file from disk).
from model_first_order import (  # noqa: E402
    ScalarPotentialLMFirstOrder, SPLMFirstOrderConfig,
)
from energetic_minima.model_ln import (  # noqa: E402
    ScalarPotentialLMSARFMassLN, SPLMSARFMassLNConfig,
)


@dataclass
class Trajectory:
    sentence: str
    sentence_idx: int
    token_ids: np.ndarray
    hidden_states: np.ndarray  # (T, d) at h^(L)


def _pick_device(force: str | None = None) -> str:
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


def segment_shakespeare_sentences(
    val_ids: np.ndarray,
    enc: tiktoken.Encoding,
    min_tokens: int,
    n_sentences: int,
) -> list[tuple[str, list[int]]]:
    """Decode val ids, sentence-split, and return the first N >= min_tokens sentences."""
    text = enc.decode(val_ids.tolist())
    raw = re.split(r"(?<=[.!?])\s+", text)
    selected: list[tuple[str, list[int]]] = []
    for sent in raw:
        s = sent.strip()
        if not s:
            continue
        toks = enc.encode(s)
        if len(toks) < min_tokens:
            continue
        selected.append((s, toks))
        if len(selected) >= n_sentences:
            break
    return selected


def load_splm_from_checkpoint(
    ckpt_path: Path, device: str
) -> tuple[torch.nn.Module, str, dict]:
    """Rebuild a SPLM (em_ln or first-order) and load its state dict."""
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = blob["model_cfg"]
    variant = blob.get("variant", "")
    if variant == "splm_first_order":
        cfg = SPLMFirstOrderConfig(**cfg_dict)
        model = ScalarPotentialLMFirstOrder(cfg)
    elif variant == "sarf_mass_ln":
        cfg = SPLMSARFMassLNConfig(**cfg_dict)
        model = ScalarPotentialLMSARFMassLN(cfg)
    else:
        raise ValueError(
            f"unknown checkpoint variant {variant!r} in {ckpt_path}; "
            "expected 'splm_first_order' or 'sarf_mass_ln'."
        )
    missing = model.load_state_dict(blob["model_state_dict"], strict=False)
    if missing.missing_keys or missing.unexpected_keys:
        print(f"[extract] state-dict diff for {ckpt_path}:")
        print(f"  missing   = {missing.missing_keys}")
        print(f"  unexpected= {missing.unexpected_keys}")
    model.to(device).eval()
    return model, variant, blob


def extract_trajectory(
    sentence: str,
    sentence_idx: int,
    token_ids: list[int],
    model: torch.nn.Module,
    device: str,
    max_length: int,
) -> Trajectory:
    toks = token_ids[:max_length]
    x = torch.tensor([toks], dtype=torch.long, device=device)
    with torch.enable_grad():
        out = model(x, return_trajectory=True)
    traj_h = out[2]
    h_L = traj_h[-1][0].detach().float().cpu().numpy()
    return Trajectory(
        sentence=sentence,
        sentence_idx=sentence_idx,
        token_ids=np.asarray(toks, dtype=np.int64),
        hidden_states=h_L,
    )


def build_quadruples(trajectories: List[Trajectory]) -> dict:
    H_tm2_list: list[np.ndarray] = []
    H_tm1_list: list[np.ndarray] = []
    H_t_list: list[np.ndarray] = []
    H_tp1_list: list[np.ndarray] = []
    sentence_idx_list: list[int] = []
    position_list: list[int] = []
    skipped_short: list[int] = []
    counts_per_sentence: list[int] = []

    for tr in trajectories:
        hs = tr.hidden_states
        T = hs.shape[0]
        if T < 4:
            skipped_short.append(tr.sentence_idx)
            counts_per_sentence.append(0)
            continue
        ts = np.arange(2, T - 1)
        H_tm2_list.append(hs[ts - 2])
        H_tm1_list.append(hs[ts - 1])
        H_t_list.append(hs[ts])
        H_tp1_list.append(hs[ts + 1])
        sentence_idx_list.extend([tr.sentence_idx] * len(ts))
        position_list.extend(ts.tolist())
        counts_per_sentence.append(int(len(ts)))

    H_tm2 = np.concatenate(H_tm2_list, axis=0)
    H_tm1 = np.concatenate(H_tm1_list, axis=0)
    H_t = np.concatenate(H_t_list, axis=0)
    H_tp1 = np.concatenate(H_tp1_list, axis=0)
    sentence_idx = np.asarray(sentence_idx_list, dtype=np.int32)
    position = np.asarray(position_list, dtype=np.int32)

    return dict(
        H_tm2=H_tm2,
        H_tm1=H_tm1,
        H_t=H_t,
        H_tp1=H_tp1,
        sentence_idx=sentence_idx,
        position=position,
        skipped_short=skipped_short,
        counts_per_sentence=counts_per_sentence,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True,
                    help="Path to SPLM ckpt_latest.pt (em_ln or first_order).")
    ap.add_argument("--output_dir", required=True,
                    help="Output directory for quadruples.npz + summary.")
    ap.add_argument("--n_sentences", type=int, default=100)
    ap.add_argument("--min_tokens", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=128,
                    help="Max BPE tokens per sentence (matches train block_size).")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _pick_device(args.device)

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    model, variant, blob = load_splm_from_checkpoint(ckpt_path, device)

    cfg = blob["model_cfg"]
    print(f"[extract] checkpoint = {ckpt_path}")
    print(f"[extract] variant    = {variant}")
    print(f"[extract] device     = {device}")
    print(f"[extract] d={cfg.get('d')}  L={cfg.get('L')}  "
          f"v_hidden={cfg.get('v_hidden')}  max_len={cfg.get('max_len')}  "
          f"ln_after_step={cfg.get('ln_after_step')}  "
          f"final_val_ppl={blob.get('final_val_ppl'):.2f}")

    enc = tiktoken.get_encoding("gpt2")
    _, val_ids = load_tiny_shakespeare()
    sentences = segment_shakespeare_sentences(
        val_ids, enc, args.min_tokens, args.n_sentences
    )
    print(f"[extract] selected {len(sentences)} sentences "
          f"(>= {args.min_tokens} tokens, in document order)")

    if len(sentences) < args.n_sentences:
        print(f"[extract] WARNING: only {len(sentences)} sentences met the "
              f">={args.min_tokens}-token threshold; protocol asks for "
              f"{args.n_sentences}.")

    t0 = time.time()
    trajectories: list[Trajectory] = []
    for idx, (sent, toks) in enumerate(sentences):
        trajectories.append(
            extract_trajectory(sent, idx, toks, model, device, args.max_length)
        )
    extract_seconds = time.time() - t0

    quads = build_quadruples(trajectories)
    n_quads = quads["H_t"].shape[0]
    T_lengths = [tr.hidden_states.shape[0] for tr in trajectories]

    npz_path = out_dir / "quadruples.npz"
    np.savez_compressed(
        npz_path,
        H_tm2=quads["H_tm2"].astype(np.float32),
        H_tm1=quads["H_tm1"].astype(np.float32),
        H_t=quads["H_t"].astype(np.float32),
        H_tp1=quads["H_tp1"].astype(np.float32),
        sentence_idx=quads["sentence_idx"],
        position=quads["position"],
        hidden_dim=np.int32(int(cfg.get("d"))),
    )
    size_mb = npz_path.stat().st_size / 1e6
    print(f"[extract] saved {npz_path}  ({size_mb:.1f} MB)")

    summary = {
        "checkpoint": str(ckpt_path),
        "variant": variant,
        "device": device,
        "model_cfg": cfg,
        "final_val_ppl": float(blob.get("final_val_ppl", float("nan"))),
        "n_sentences": len(sentences),
        "min_tokens": args.min_tokens,
        "max_length": args.max_length,
        "n_quadruples_total": int(n_quads),
        "token_lengths": {
            "min": int(min(T_lengths)),
            "max": int(max(T_lengths)),
            "mean": float(np.mean(T_lengths)),
            "median": float(np.median(T_lengths)),
        },
        "quadruples_per_sentence": {
            "min": int(min(quads["counts_per_sentence"])),
            "max": int(max(quads["counts_per_sentence"])),
            "mean": float(np.mean(quads["counts_per_sentence"])),
        },
        "skipped_sentences_too_short": quads["skipped_short"],
        "extract_seconds": float(extract_seconds),
    }
    with open(out_dir / "extraction_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[extract] {n_quads} quadruples emitted in {extract_seconds:.1f}s; "
          f"summary written.")


if __name__ == "__main__":
    main()
