"""
Hidden-state trajectory extraction for conservative-by-construction LMs.

Reads a trained scalar-potential LM checkpoint and produces, for each
sentence in a small corpus, the per-layer trajectory
  hs  : (L+1, T, d)  hidden states after each integration step
  ptl : (T-1,)       per-token cross-entropy on x_{t+1} | x_{<=t}
  w   : (L+1, T)     per-layer per-token "mass" (uniform 1.0 here --
                     we have no attention weights to sum, so the
                     token-level mass is flat; kept for parity with
                     the §1 extractor)

Output format is a drop-in replacement for the Trajectory dataclass in
notebooks/e_init/extended_gamma_and_first_order.py / velocity_coupled_gauge.py,
so the same E-init fitting pipeline can be reused unchanged.

Usage:
  python3 trajectory_extraction.py --ckpt results/splm_smoke_ckpt_latest.pt
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import List

import numpy as np
import torch

from e_init_corpus import CORPUS
from model import ScalarPotentialLM, SPLMConfig
from trajectory_types import Trajectory


@torch.no_grad()
def extract_one(
    model: ScalarPotentialLM,
    tokenizer,
    sentence: str,
    domain: str,
    split: str,
    device: str,
    max_len: int,
) -> Trajectory:
    ids = tokenizer.encode(sentence)[:max_len]
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

    # PTL targets are the shifted ids.
    # Use forward(..., return_trajectory=True) under enable_grad for the
    # gradient-of-potential to work, then stack the trajectory tensors.
    with torch.enable_grad():
        logits, _, traj_list = model(x, return_trajectory=True)
    # traj_list: [h_0, h_1, ..., h_L], each (1, T, d) on CPU.
    hs = torch.stack([t.squeeze(0) for t in traj_list], dim=0).numpy()  # (L+1, T, d)

    # Per-token log-likelihood of next token.
    if len(ids) >= 2:
        tgt = torch.tensor(ids[1:], device=device)
        ptl = torch.nn.functional.cross_entropy(
            logits[0, :-1, :], tgt, reduction="none"
        ).cpu().numpy()
    else:
        ptl = np.zeros((0,), dtype=np.float32)

    T = hs.shape[1]
    w = np.ones((hs.shape[0], T), dtype=np.float32)

    return Trajectory(
        sentence=sentence, domain=domain, split=split,
        tok_ids=np.asarray(ids, dtype=np.int64),
        hs=hs.astype(np.float32),
        ptl=ptl.astype(np.float32),
        w=w,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to splm checkpoint")
    ap.add_argument("--out",  default=None,
                    help="output pickle; defaults next to ckpt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--n_test_per_domain", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = args.device or (
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"[extract] device={device}  ckpt={args.ckpt}")

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model_cfg = SPLMConfig(**ck["model_cfg"])
    model = ScalarPotentialLM(model_cfg).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    print(f"[extract] loaded model with d={model_cfg.d}  L={model_cfg.L}  "
          f"max_len={model_cfg.max_len}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    rng = np.random.default_rng(args.seed)
    trajs: List[Trajectory] = []
    for domain, sents in CORPUS.items():
        idx = np.arange(len(sents))
        rng.shuffle(idx)
        for i in idx[: args.n_test_per_domain]:
            trajs.append(extract_one(model, tokenizer, sents[i], domain,
                                     "test", device, args.max_len))
        for i in idx[args.n_test_per_domain:]:
            trajs.append(extract_one(model, tokenizer, sents[i], domain,
                                     "train", device, args.max_len))

    print(f"[extract] extracted {len(trajs)} trajectories")

    # Pre-compute per-sentence centering (same convention as §1 pipeline).
    for tr in trajs:
        tr.mu_ps = tr.hs.mean(axis=1, keepdims=True)
        tr.x_ps  = tr.hs - tr.mu_ps

    out_path = args.out or (Path(args.ckpt).with_suffix(".trajectories.pkl"))
    with open(out_path, "wb") as f:
        pickle.dump({
            "trajectories": trajs,
            "model_cfg": ck["model_cfg"],
            "d": model_cfg.d,
            "L": model_cfg.L,
        }, f)
    print(f"[extract] saved -> {out_path}")


if __name__ == "__main__":
    main()
