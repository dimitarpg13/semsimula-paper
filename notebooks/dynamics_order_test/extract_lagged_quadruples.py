"""Extract per-position lagged quadruples (h_{t-2}, h_{t-1}, h_t, h_{t+1}) from a HF causal LM.

This is **phase 1** of the first-order ODE rejection experiment. The protocol is
pre-registered in `docs/first_order_ODE_rejection_pre-registered_protocol.md` (§3).

Inputs to the experiment are *strictly* hidden-state vectors at the last layer
(no token IDs, no positional encodings, no attention weights). One sentence
yields T tokens and (T - 3) inside-sentence quadruples. The unit of LOSO
cross-validation downstream is the sentence, not the quadruple, so we record
`sentence_idx` for every quadruple.

Usage
-----
    python extract_lagged_quadruples.py \
        --model gpt2 \
        --output_dir results/gpt2

    python extract_lagged_quadruples.py \
        --model EleutherAI/pythia-160m \
        --output_dir results/pythia
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class Trajectory:
    sentence: str
    domain: str
    sentence_idx: int
    token_ids: np.ndarray
    hidden_states: np.ndarray  # (T, d) last-layer
    per_token_loss: np.ndarray  # (T-1,)


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


def load_corpus(corpus_path: Path) -> tuple[list[str], list[str]]:
    with open(corpus_path) as f:
        data = json.load(f)
    sentences: list[str] = []
    domains: list[str] = []
    for domain, sents in data["domains"].items():
        for s in sents:
            sentences.append(s)
            domains.append(domain)
    return sentences, domains


@torch.no_grad()
def extract_trajectory(
    sentence: str,
    domain: str,
    sentence_idx: int,
    tokenizer,
    model,
    device: str,
    max_length: int,
) -> Trajectory:
    enc = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"].to(device)
    out = model(input_ids, output_hidden_states=True)
    hs = out.hidden_states[-1][0].float().cpu().numpy()  # (T, d)
    logits = out.logits[0].float()
    shift_logits = logits[:-1]
    shift_labels = input_ids[0, 1:]
    ptl = (
        F.cross_entropy(shift_logits, shift_labels, reduction="none").cpu().numpy()
    )  # (T-1,)
    return Trajectory(
        sentence=sentence,
        domain=domain,
        sentence_idx=sentence_idx,
        token_ids=input_ids[0].cpu().numpy(),
        hidden_states=hs,
        per_token_loss=ptl,
    )


def build_quadruples(trajectories: List[Trajectory]) -> dict:
    """Slice each trajectory into inside-sentence (h_{t-2}, h_{t-1}, h_t, h_{t+1}) quadruples.

    For a sentence with T tokens, we emit indices t in [2, T-2], yielding T-3
    quadruples per sentence. We need T >= 4 to emit at least one quadruple.
    """
    H_tm2_list: list[np.ndarray] = []
    H_tm1_list: list[np.ndarray] = []
    H_t_list: list[np.ndarray] = []
    H_tp1_list: list[np.ndarray] = []
    sentence_idx_list: list[int] = []
    domain_list: list[str] = []
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
        # t ranges so that t-2 >= 0 and t+1 <= T-1 => 2 <= t <= T-2
        ts = np.arange(2, T - 1)
        H_tm2_list.append(hs[ts - 2])
        H_tm1_list.append(hs[ts - 1])
        H_t_list.append(hs[ts])
        H_tp1_list.append(hs[ts + 1])
        sentence_idx_list.extend([tr.sentence_idx] * len(ts))
        domain_list.extend([tr.domain] * len(ts))
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
        domains=np.asarray(domain_list),
        skipped_short=skipped_short,
        counts_per_sentence=counts_per_sentence,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True,
                    help="HF model id, e.g. 'gpt2' or 'EleutherAI/pythia-160m'")
    ap.add_argument("--output_dir", required=True, help="Directory for npz + summary")
    ap.add_argument(
        "--corpus",
        default=str(Path(__file__).parent / "data" / "corpus.json"),
        help="Path to corpus JSON",
    )
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default=None,
                    help="Force device: cuda | mps | cpu (default: auto)")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _pick_device(args.device)
    print(f"[extract] model={args.model}  device={device}")

    sentences, domains = load_corpus(Path(args.corpus))
    print(f"[extract] corpus: {len(sentences)} sentences, "
          f"{len(set(domains))} domains")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model = model.to(device)
    model.eval()
    hidden_dim = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[extract] hidden_dim={hidden_dim}  layers={n_layers}  "
          f"params={n_params/1e6:.1f}M")

    t0 = time.time()
    trajectories: List[Trajectory] = []
    for idx, (sent, dom) in enumerate(
        tqdm(list(zip(sentences, domains)), desc="extract")
    ):
        trajectories.append(
            extract_trajectory(sent, dom, idx, tokenizer, model, device, args.max_length)
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
        domains=quads["domains"],
        hidden_dim=np.int32(hidden_dim),
    )
    print(f"[extract] saved {npz_path}  ({npz_path.stat().st_size/1e6:.1f} MB)")

    summary = {
        "model": args.model,
        "device": device,
        "n_sentences": len(sentences),
        "n_domains": len(set(domains)),
        "hidden_dim": int(hidden_dim),
        "num_layers": int(n_layers),
        "n_params": int(n_params),
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
    print(f"[extract] {n_quads} quadruples emitted; summary written.")


if __name__ == "__main__":
    main()
