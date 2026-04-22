"""Extract hidden-state trajectories from a trained MatchedGPT checkpoint,
in the same pickle format used by the SPLM pipeline (so `jacobian_symmetry.py`
and `shared_potential_fit.py` run unchanged for an apples-to-apples comparison).
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import List

import numpy as np
import torch

from e_init_corpus import CORPUS
from matched_baseline_model import MatchedConfig, MatchedGPT
from trajectory_types import Trajectory


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def extract_one(model: MatchedGPT, tokenizer, sentence, domain, split, device, max_len):
    ids = tokenizer.encode(sentence)[:max_len]
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    _, _, traj = model(x, targets=None, return_trajectory=True)
    # traj: list of (1, T, d) tensors, length n_layer + 1 (initial embedding + per-block outputs)
    hs_stack = torch.stack(traj, dim=0).squeeze(1).float().cpu().numpy()  # (L+1, T, d)

    logits, _ = model(x, targets=None)
    if len(ids) >= 2:
        tgt = torch.tensor(ids[1:], device=device)
        ptl = torch.nn.functional.cross_entropy(
            logits[0, :-1, :], tgt, reduction="none",
        ).float().cpu().numpy()
    else:
        ptl = np.zeros((0,), dtype=np.float32)

    T = hs_stack.shape[1]
    w = np.ones((hs_stack.shape[0], T), dtype=np.float32)

    return Trajectory(
        sentence=sentence, domain=domain, split=split,
        tok_ids=np.asarray(ids, dtype=np.int64),
        hs=hs_stack.astype(np.float32),
        ptl=ptl.astype(np.float32), w=w,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out",  default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--n_test_per_domain", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = args.device or _pick_device()
    ckpt_path = Path(args.ckpt)
    print(f"[extract-matched] device={device}  ckpt={ckpt_path.name}")

    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = MatchedConfig(**raw["model_cfg"])
    model = MatchedGPT(cfg).to(device)
    model.load_state_dict(raw["model_state_dict"])
    model.eval()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")

    d = cfg.d
    L = cfg.n_layer
    print(f"[extract-matched] d={d}  n_layer={L}  (hidden_states: L+1={L+1})")

    rng = np.random.default_rng(args.seed)
    trajs: List[Trajectory] = []
    for domain, sents in CORPUS.items():
        idx = np.arange(len(sents)); rng.shuffle(idx)
        for i in idx[:args.n_test_per_domain]:
            trajs.append(extract_one(model, tok, sents[i], domain,
                                     "test", device, args.max_len))
        for i in idx[args.n_test_per_domain:]:
            trajs.append(extract_one(model, tok, sents[i], domain,
                                     "train", device, args.max_len))
    print(f"[extract-matched] extracted {len(trajs)} trajectories")

    for tr in trajs:
        tr.mu_ps = tr.hs.mean(axis=1, keepdims=True)
        tr.x_ps  = tr.hs - tr.mu_ps

    here = Path(__file__).parent
    out_path = Path(args.out) if args.out else (
        here / "results" / "matched_baseline.trajectories.pkl"
    )
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"trajectories": trajs, "d": d, "L": L,
                     "model_name": "matched_shakespeare"}, f)
    print(f"[extract-matched] saved -> {out_path}")


if __name__ == "__main__":
    main()
