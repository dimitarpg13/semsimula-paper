"""
Trajectory extraction for SARF-faithful SPLM with velocity-Verlet integrator.

Mirrors sarf_mass_variant/trajectory_extraction_sarf_mass.py but loads a
ScalarPotentialLMSymplectic checkpoint.  Output Trajectory objects have
the same layout as every other conservative-arch variant (hs: (L+1, T, d)),
so the parent diagnostics ../shared_potential_fit.py and
../token_direction_fit.py work drop-in.

Usage:
  python3 trajectory_extraction_symplectic.py \
      --ckpt results/splm_sym_logfreq_shakespeare_ckpt_latest.pt
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

from e_init_corpus import CORPUS              # noqa: E402
from trajectory_types import Trajectory       # noqa: E402

from model_symplectic import ScalarPotentialLMSymplectic, SPLMSymplecticConfig  # noqa: E402


@torch.no_grad()
def extract_one(
    model: ScalarPotentialLMSymplectic,
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
        logits, _, traj_list = model(x, return_trajectory=True)
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
    print(f"[extract-sym] device={device}  ckpt={args.ckpt}")

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model_cfg = SPLMSymplecticConfig(**ck["model_cfg"])
    model = ScalarPotentialLMSymplectic(model_cfg).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    print(f"[extract-sym] loaded model  d={model_cfg.d}  L={model_cfg.L}  "
          f"mass_mode={model_cfg.mass_mode}  dt={model_cfg.dt}  "
          f"variant={ck.get('variant', 'symplectic')}")

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

    print(f"[extract-sym] extracted {len(trajs)} trajectories")

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
            "variant": "symplectic",
            "integrator": ck.get("integrator", "velocity_verlet_strang"),
            "mass_mode": model_cfg.mass_mode,
            "dt": model_cfg.dt,
        }, f)
    print(f"[extract-sym] saved -> {out_path}")


if __name__ == "__main__":
    main()
