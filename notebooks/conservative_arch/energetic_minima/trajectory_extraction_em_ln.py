"""
Trajectory extraction for ScalarPotentialLMSARFMassLN (em_ln) checkpoints.

Mirrors sarf_mass_variant/trajectory_extraction_sarf_mass.py but loads
the LayerNorm-after-step (em_ln) variant. Output Trajectory objects have
the same layout (hs: (L+1, T, d)), so the existing parent diagnostics
`../shared_potential_fit.py` and `../token_direction_fit.py` work drop-in.

Usage:
  python3 trajectory_extraction_em_ln.py \
      --ckpt /Users/dimitargueorguiev/git/ml/semsimula/notebooks/conservative_arch/ln_damping_sweep/results/leakfree_3seed/gamma0p10/seed0/splm_em_ln_shakespeare_gamma0p10_seed0_ckpt_latest.pt
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
sys.path.insert(0, str(PARENT_DIR / "sarf_mass_variant"))

from e_init_corpus import CORPUS              # noqa: E402
from trajectory_types import Trajectory       # noqa: E402

from model_ln import ScalarPotentialLMSARFMassLN, SPLMSARFMassLNConfig  # noqa: E402


@torch.no_grad()
def extract_one(
    model: ScalarPotentialLMSARFMassLN,
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
    hs = torch.stack([t.squeeze(0) for t in traj_list], dim=0).cpu().numpy()

    if len(ids) >= 2:
        tgt = torch.tensor(ids[1:], device=device)
        ptl = torch.nn.functional.cross_entropy(
            logits[0, :-1, :], tgt, reduction="none"
        ).cpu().numpy()
    else:
        ptl = np.zeros((0,), dtype=np.float32)

    T = hs.shape[1]
    w = np.ones((hs.shape[0], T), dtype=np.float32)

    mu_ps = hs.mean(axis=1, keepdims=True).astype(np.float32)  # (L+1, 1, d)
    x_ps  = (hs - mu_ps).astype(np.float32)                     # (L+1, T, d)

    return Trajectory(
        sentence=sentence, domain=domain, split=split,
        tok_ids=np.asarray(ids, dtype=np.int64),
        hs=hs.astype(np.float32),
        ptl=ptl.astype(np.float32),
        w=w,
        mu_ps=mu_ps,
        x_ps=x_ps,
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
    print(f"[extract-em_ln] device={device}  ckpt={args.ckpt}")

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model_cfg = SPLMSARFMassLNConfig(**ck["model_cfg"])
    model = ScalarPotentialLMSARFMassLN(model_cfg).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    print(f"[extract-em_ln] loaded model  d={model_cfg.d}  L={model_cfg.L}  "
          f"mass_mode={model_cfg.mass_mode}  ln_after_step={model_cfg.ln_after_step}  "
          f"variant={ck.get('variant', 'sarf_mass_ln')}  "
          f"causal_force={getattr(model_cfg, 'causal_force', '?')}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    trajs: List[Trajectory] = []
    for domain, sent_list in CORPUS.items():
        for i, sentence in enumerate(sent_list[: 2 * args.n_test_per_domain]):
            split = "train" if i < args.n_test_per_domain else "test"
            trj = extract_one(
                model, tokenizer, sentence, domain=domain, split=split,
                device=device, max_len=args.max_len,
            )
            trajs.append(trj)
            print(f"[extract-em_ln] {split:5s} {domain}/{sentence[:40]:<40s}  "
                  f"T={trj.hs.shape[1]:3d}  L+1={trj.hs.shape[0]:2d}  "
                  f"d={trj.hs.shape[2]:3d}")

    out_path = args.out
    if out_path is None:
        ckpt_path = Path(args.ckpt)
        out_path = ckpt_path.parent / (ckpt_path.stem + ".trajectories.pkl")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "trajectories": trajs,
        "model_cfg": ck["model_cfg"],
        "d": int(model_cfg.d),
        "L": int(model_cfg.L),
        "variant": ck.get("variant", "sarf_mass_ln"),
        "mass_mode": getattr(model_cfg, "mass_mode", "logfreq"),
    }
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"[extract-em_ln] wrote {len(trajs)} trajectories to {out_path}")


if __name__ == "__main__":
    main()
