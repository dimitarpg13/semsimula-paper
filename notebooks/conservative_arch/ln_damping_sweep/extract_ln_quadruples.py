"""Extract per-token last-layer hidden-state quadruples from a trained em_ln checkpoint.

Identical interface to damping_sweep/extract_splm_quadruples.py but loads the
ScalarPotentialLMSARFMassLN model (variant="sarf_mass_ln") instead of the plain
Euler integrator.  All output formats are identical so the same Markov-order
regression machinery runs unmodified.

Inputs (per cell):
    --ckpt    .pt file saved by train_splm_em_ln.py
    --logfreq path to logfreq_surprisal.npy
    --out_dir results/<tag>/markov_order/
    --corpus  notebooks/dynamics_order_test/data/corpus.json (default)

Outputs:
    <out_dir>/quadruples.npz
    <out_dir>/extraction_summary.json
    <out_dir>/trajectories.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR         = Path(__file__).resolve().parent
PARENT_DIR         = SCRIPT_DIR.parent
EM_MINIMA_DIR      = PARENT_DIR / "energetic_minima"
SARF_MASS_DIR      = PARENT_DIR / "sarf_mass_variant"
DYNAMICS_ORDER_DIR = PARENT_DIR.parent / "dynamics_order_test"

for _p in [str(PARENT_DIR), str(EM_MINIMA_DIR), str(SARF_MASS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


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


def load_ln_checkpoint(ckpt_path: Path, logfreq_path: Path | None, device: str):
    from model_ln import ScalarPotentialLMSARFMassLN, SPLMSARFMassLNConfig  # noqa
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = dict(ck["model_cfg"])
    if cfg_dict.get("mass_mode") == "logfreq":
        if logfreq_path is None:
            raise ValueError("logfreq mass_mode requires --logfreq <path>")
        cfg_dict["logfreq_path"] = str(logfreq_path)
    cfg = SPLMSARFMassLNConfig(**cfg_dict)
    model = ScalarPotentialLMSARFMassLN(cfg).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, cfg, ck


@torch.no_grad()
def _last_layer_hidden(model, x: torch.Tensor) -> np.ndarray:
    """Return h_L (shape (T, d)) from a single batch of shape (1, T)."""
    with torch.enable_grad():
        emb = model._embed(x)
        h_L, _, _ = model.integrate(x, emb,
                                    return_trajectory=False,
                                    return_xi_trajectory=False)
    return h_L[0].detach().float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt",    required=True, type=Path)
    ap.add_argument("--logfreq", type=Path, default=None)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument(
        "--corpus",
        default=str(DYNAMICS_ORDER_DIR / "data" / "corpus.json"),
        type=str,
    )
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _pick_device(args.device)
    print(f"[ln-extract] ckpt={args.ckpt}  device={device}")

    sentences, domains = load_corpus(Path(args.corpus))
    print(f"[ln-extract] corpus: {len(sentences)} sentences, "
          f"{len(set(domains))} domains")

    model, cfg, ck = load_ln_checkpoint(args.ckpt, args.logfreq, device)
    print(f"[ln-extract] L={cfg.L}  d={cfg.d}  "
          f"final_gamma={ck.get('final_gamma', float('nan')):.4f}  "
          f"fixed_gamma={ck.get('fixed_gamma')}  "
          f"ln_after_step={cfg.ln_after_step}")

    from transformers import AutoTokenizer  # noqa: E402
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    H_tm2_list: list[np.ndarray] = []
    H_tm1_list: list[np.ndarray] = []
    H_t_list:   list[np.ndarray] = []
    H_tp1_list: list[np.ndarray] = []
    sentence_idx_list: list[int]  = []
    domain_list:       list[str]  = []
    position_list:     list[int]  = []
    skipped_short: list[int] = []
    counts_per_sentence: list[int] = []

    trajectories_per_sentence: list[np.ndarray] = []
    domains_per_sentence:      list[str]         = []
    T_lengths:                 list[int]         = []

    t0 = time.time()
    for idx, (sent, dom) in enumerate(zip(sentences, domains)):
        ids = tokenizer.encode(sent)[: args.max_length]
        x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        if x.shape[1] > cfg.max_len:
            x = x[:, : cfg.max_len]
        hs = _last_layer_hidden(model, x)   # (T, d)
        T = hs.shape[0]
        trajectories_per_sentence.append(hs.astype(np.float32))
        domains_per_sentence.append(dom)
        T_lengths.append(T)

        if T < 4:
            skipped_short.append(idx)
            counts_per_sentence.append(0)
            continue
        ts = np.arange(2, T - 1)
        H_tm2_list.append(hs[ts - 2])
        H_tm1_list.append(hs[ts - 1])
        H_t_list.append(hs[ts])
        H_tp1_list.append(hs[ts + 1])
        sentence_idx_list.extend([idx] * len(ts))
        domain_list.extend([dom] * len(ts))
        position_list.extend(ts.tolist())
        counts_per_sentence.append(int(len(ts)))

        if (idx + 1) % 10 == 0 or idx + 1 == len(sentences):
            print(f"  [{idx+1:3d}/{len(sentences)}] T={T:3d}  "
                  f"|h_L|^2_mean={(hs**2).sum(-1).mean():.4f}  "
                  f"elapsed {time.time()-t0:.1f}s")

    extract_seconds = time.time() - t0

    H_tm2 = np.concatenate(H_tm2_list, axis=0).astype(np.float32)
    H_tm1 = np.concatenate(H_tm1_list, axis=0).astype(np.float32)
    H_t   = np.concatenate(H_t_list,   axis=0).astype(np.float32)
    H_tp1 = np.concatenate(H_tp1_list, axis=0).astype(np.float32)
    sentence_idx = np.asarray(sentence_idx_list, dtype=np.int32)
    position     = np.asarray(position_list,     dtype=np.int32)
    domains_arr  = np.asarray(domain_list)
    hidden_dim   = int(cfg.d)

    quad_path = out_dir / "quadruples.npz"
    np.savez_compressed(
        quad_path,
        H_tm2=H_tm2, H_tm1=H_tm1, H_t=H_t, H_tp1=H_tp1,
        sentence_idx=sentence_idx, position=position, domains=domains_arr,
        hidden_dim=np.int32(hidden_dim),
    )
    print(f"[ln-extract] saved {quad_path}  "
          f"({quad_path.stat().st_size / 1e6:.1f} MB; n_quads={H_t.shape[0]})")

    T_max = max(T_lengths) if T_lengths else 0
    H_pad = np.full((len(trajectories_per_sentence), T_max, hidden_dim),
                    np.nan, dtype=np.float32)
    for i, hs in enumerate(trajectories_per_sentence):
        H_pad[i, : hs.shape[0]] = hs
    traj_path = out_dir / "trajectories.npz"
    np.savez_compressed(
        traj_path,
        H=H_pad,
        T_lens=np.asarray(T_lengths, dtype=np.int32),
        domains=np.asarray(domains_per_sentence),
        hidden_dim=np.int32(hidden_dim),
    )
    print(f"[ln-extract] saved {traj_path}  "
          f"({traj_path.stat().st_size / 1e6:.1f} MB; "
          f"{H_pad.shape[0]} sentences, T_max={T_max})")

    summary = {
        "ckpt": str(args.ckpt),
        "device": device,
        "variant": "sarf_mass_ln",
        "ln_after_step": bool(cfg.ln_after_step),
        "L": int(cfg.L),
        "d": int(cfg.d),
        "dt": float(cfg.dt),
        "final_gamma": float(ck.get("final_gamma", 0.0)),
        "fixed_gamma": ck.get("fixed_gamma"),
        "n_sentences": len(sentences),
        "n_domains": len(set(domains)),
        "n_quadruples_total": int(H_t.shape[0]),
        "token_lengths": {
            "min": int(min(T_lengths)),
            "max": int(max(T_lengths)),
            "mean": float(np.mean(T_lengths)),
            "median": float(np.median(T_lengths)),
        },
        "quadruples_per_sentence": {
            "min": int(min(counts_per_sentence)),
            "max": int(max(counts_per_sentence)),
            "mean": float(np.mean(counts_per_sentence)),
        },
        "skipped_sentences_too_short": skipped_short,
        "extract_seconds": float(extract_seconds),
    }
    with open(out_dir / "extraction_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[ln-extract] {summary['n_quadruples_total']} quadruples emitted; "
          f"summary written.")


if __name__ == "__main__":
    main()
