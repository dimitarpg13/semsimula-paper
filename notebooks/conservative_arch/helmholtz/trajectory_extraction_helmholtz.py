"""
Trajectory extraction for the Helmholtz hybrid (Q9d).

Mirrors `sarf_mass_variant/trajectory_extraction_sarf_mass.py`.  Loads
a `HelmholtzLM` checkpoint produced by `train_helmholtz.py` and emits
a pickle bundle compatible with the parent diagnostics
`../shared_potential_fit.py` and `./substack_separator.py`.

The Helmholtz forward pass returns a per-layer trajectory of shape
(L+1, T, d) where layer 0 is the embedding, layer ell in [1..L] is
the post-step hidden state of the ell-th block (S or A).  This is
exactly the format `Trajectory.hs` expects.

The bundle additionally records the per-layer block kind (S or A)
under key `block_kinds` so downstream substack-restricted analyses
can slice the layer index by block type.

Usage:
  python3 trajectory_extraction_helmholtz.py \
      --ckpt notebooks/conservative_arch/helmholtz/results/h2_paired_confirmation/AAAASSSS_vh128/seed0/helm_AAAASSSS_vh128_shakespeare_seed0_ckpt_latest.pt
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).parent
PARENT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from e_init_corpus import CORPUS              # noqa: E402
from trajectory_types import Trajectory       # noqa: E402

from model_helmholtz import HelmholtzLM, HelmholtzConfig  # noqa: E402


@torch.no_grad()
def extract_one(
    model: HelmholtzLM,
    tokenizer,
    sentence: str,
    domain: str,
    split: str,
    device: str,
    max_len: int,
) -> Trajectory:
    ids = tokenizer.encode(sentence)[:max_len]
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.enable_grad():
        out = model(x, return_trajectory=True)
        # forward returns (logits, loss[, traj][, new_caches])
        logits = out[0]
        traj_list = out[2]
    hs = torch.stack([t.squeeze(0) for t in traj_list], dim=0).numpy()

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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out",  default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--n_test_per_domain", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = args.device or (
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"[extract-helm] device={device}  ckpt={args.ckpt}")

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model_cfg = HelmholtzConfig(**ck["model_cfg"])
    model = HelmholtzLM(model_cfg).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    schedule = model_cfg.schedule
    L = len(schedule)
    print(f"[extract-helm] loaded model  schedule='{schedule}'  L={L}  "
          f"d={model_cfg.d}  v_hidden={model_cfg.v_hidden}  "
          f"variant={ck.get('variant', 'helmholtz_q9d')}")

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

    print(f"[extract-helm] extracted {len(trajs)} trajectories")

    for tr in trajs:
        tr.mu_ps = tr.hs.mean(axis=1, keepdims=True)
        tr.x_ps  = tr.hs - tr.mu_ps

    # block_kinds: list of length L, each "S" or "A".  Index 0 of hs
    # is the pre-step embedding so block_kinds[ell-1] is the kind of
    # the ell-th update step (1-indexed in the shared-V notation).
    block_kinds = list(schedule)

    out_path = args.out or (Path(args.ckpt).with_suffix(".trajectories.pkl"))
    with open(out_path, "wb") as f:
        pickle.dump({
            "trajectories": trajs,
            "model_cfg": ck["model_cfg"],
            "d": model_cfg.d,
            "L": L,
            "variant": "helmholtz_q9d",
            "schedule": schedule,
            "block_kinds": block_kinds,
            "v_hidden": model_cfg.v_hidden,
            "tag": ck.get("tag", Path(args.ckpt).stem),
        }, f)
    print(f"[extract-helm] saved -> {out_path}")
    print(f"[extract-helm] block_kinds = {block_kinds}")


if __name__ == "__main__":
    main()
