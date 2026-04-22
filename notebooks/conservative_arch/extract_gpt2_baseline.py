"""Extract GPT-2 hidden-state trajectories into the same pickle format
used by the SPLM pipeline, so `jacobian_symmetry.py` and
`e_init_validation.py` can be applied to GPT-2 unchanged for an
apples-to-apples comparison.

This is the negative-control extraction that pairs with the SPLM
positive-control extraction.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import List

import numpy as np
import torch

from e_init_corpus import CORPUS
from trajectory_types import Trajectory


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def extract_one(model, tokenizer, sentence, domain, split, device, max_len):
    ids = tokenizer.encode(sentence)[:max_len]
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    out = model(x, output_hidden_states=True)
    hs_stack = torch.stack(out.hidden_states, dim=0).squeeze(1)  # (L+1, T, d)
    hs = hs_stack.float().cpu().numpy()

    logits = out.logits[0]                                       # (T, V)
    if len(ids) >= 2:
        tgt = torch.tensor(ids[1:], device=device)
        ptl = torch.nn.functional.cross_entropy(
            logits[:-1, :], tgt, reduction="none",
        ).float().cpu().numpy()
    else:
        ptl = np.zeros((0,), dtype=np.float32)

    T = hs.shape[1]
    w = np.ones((hs.shape[0], T), dtype=np.float32)

    return Trajectory(
        sentence=sentence, domain=domain, split=split,
        tok_ids=np.asarray(ids, dtype=np.int64),
        hs=hs.astype(np.float32),
        ptl=ptl.astype(np.float32), w=w,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2",
                    help="huggingface id, e.g. gpt2, gpt2-medium")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--n_test_per_domain", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = args.device or _pick_device()
    print(f"[extract-gpt2] device={device}  model={args.model}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    d  = model.config.n_embd
    L  = model.config.n_layer
    print(f"[extract-gpt2] d={d}  L={L}  (hidden_states: L+1={L+1})")

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
    print(f"[extract-gpt2] extracted {len(trajs)} trajectories")

    for tr in trajs:
        tr.mu_ps = tr.hs.mean(axis=1, keepdims=True)
        tr.x_ps  = tr.hs - tr.mu_ps

    here = Path(__file__).parent
    out_path = Path(args.out) if args.out else (
        here / "results" / f"{args.model}_baseline.trajectories.pkl"
    )
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"trajectories": trajs, "d": d, "L": L,
                     "model_name": args.model}, f)
    print(f"[extract-gpt2] saved -> {out_path}")


if __name__ == "__main__":
    main()
