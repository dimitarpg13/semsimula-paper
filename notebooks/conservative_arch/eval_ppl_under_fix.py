"""Measure val-PPL inflation of a trained SPLM checkpoint under the
causal-leak fix.

For a single .pt checkpoint, load the SAME trained weights twice — once
with `cfg.causal_force = False` (the pre-fix integrator that was used at
training time) and once with `cfg.causal_force = True` (the post-fix,
causal integrator) — and evaluate validation PPL on the SAME random val
batches under both. The ratio PPL_fixed / PPL_buggy is the inflation
factor, i.e. how much of the published PPL claim was an artifact of the
anti-causal autograd leak documented in
`docs/Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md`.

This is a forensic tool for buggy-trained checkpoints. It is *not* a
training-time eval helper — for new training under the fix, the standard
`evaluate()` already uses `causal_force=True` by default.

Usage
-----
  python3 notebooks/conservative_arch/eval_ppl_under_fix.py <ckpt.pt>
      [--n-batches N] [--batch B] [--block T]
      [--corpus {auto,shakespeare,tinystories}] [--seed S]

Defaults: n_batches=40, batch=16, block=512 (capped at the ckpt's max_len),
seed=0, corpus=auto.

Backward-compat: positional `n_batches batch block` still works.

  python3 notebooks/conservative_arch/eval_ppl_under_fix.py <ckpt> 20 8 256

Auto-corpus detection rules (in order of priority):
  1. Path contains 'shakespeare' or 'multi_seed' or 'E1_'  -> shakespeare
  2. Ckpt's `model_cfg.max_len` <= 256                     -> shakespeare
  3. otherwise                                             -> tinystories

Dispatches on the ckpt's `model_cfg` to load with the right model class.
Supports both `ScalarPotentialLMSARFMassLN` (E9 single-ξ) and
`ScalarPotentialLMSARFMassLNMultiXi` (E11 multi-channel ξ); add another
entry to the registry below to support more variants. See
`notebooks/conservative_arch/causal_probe.py` for the analogous registry
pattern used by the regression test.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).parent
for sub in ("", "sarf_mass_variant", "energetic_minima", "multixi", "scaleup"):
    sys.path.insert(0, str(THIS_DIR / sub))

from data_module import (  # type: ignore  # noqa: E402
    get_batch, load_tiny_stories, load_tiny_shakespeare,
)
from model_ln import (  # type: ignore  # noqa: E402
    SPLMSARFMassLNConfig, ScalarPotentialLMSARFMassLN,
)
from model_multixi import (  # type: ignore  # noqa: E402
    SPLMSARFMassLNMultiXiConfig, ScalarPotentialLMSARFMassLNMultiXi,
)

# first_order_ablation lives under conservative_arch/ but its module path is
# `first_order_ablation.model_first_order`. Add the parent dir to sys.path
# so the import resolves the package directly.
sys.path.insert(0, str(THIS_DIR / "first_order_ablation"))
from model_first_order import (  # type: ignore  # noqa: E402
    SPLMFirstOrderConfig, ScalarPotentialLMFirstOrder,
)


# Registry: most-specific class first.  Each entry is
# (label, variant_id, cfg_cls, model_cls).
#
# `variant_id` matches `ck["variant"]` when the ckpt records one (newer
# trainers do; pre-multixi trainers often did not). Dispatch first tries
# variant-id matching; if that fails, it falls back to "first cfg that
# can accept the ckpt's model_cfg keys *and* whose state_dict load succeeds".
#
# Variant-id is required to disambiguate `splm_first_order` vs `sarf_mass_ln`,
# because their dataclasses (`SPLMFirstOrderConfig` vs `SPLMSARFMassLNConfig`)
# have identical fields and identical state-dict shapes — the only difference
# is the `integrate()` method (first-order gradient flow vs second-order
# damped semi-implicit Euler). Without the variant-id check, a SPLM-1 ckpt
# silently loads into the second-order class and produces wrong val_ppl.
_REGISTRY = [
    ("multixi (K-channel ξ)",
     "sarf_mass_ln_multixi",
     SPLMSARFMassLNMultiXiConfig, ScalarPotentialLMSARFMassLNMultiXi),
    ("first_order (gradient-flow ablation)",
     "splm_first_order",
     SPLMFirstOrderConfig, ScalarPotentialLMFirstOrder),
    ("sarf_mass_ln (single ξ, second-order)",
     "sarf_mass_ln",
     SPLMSARFMassLNConfig, ScalarPotentialLMSARFMassLN),
]


def _try_load(cfg_cls, model_cls, cfg_dict, state_dict):
    """Attempt to instantiate cfg_cls + model_cls and load state_dict.
    Returns (keep, model) on success, raises on failure."""
    keep = {k: v for k, v in cfg_dict.items()
            if k in cfg_cls.__dataclass_fields__}
    cfg = cfg_cls(**keep)
    cfg.causal_force = False
    m = model_cls(cfg)
    m.load_state_dict(state_dict)
    return keep, m


def _dispatch(ck):
    """Return (label, cfg_cls, model_cls, keep_dict) matching this ckpt.

    Dispatch order:
      1. If `ck["variant"]` is present, prefer the registry entry whose
         `variant_id` matches — this is the unambiguous path.
      2. Otherwise fall back to first-fit: the first registry entry whose
         config can accept the ckpt's `model_cfg` keys and whose state_dict
         load succeeds.
    """
    cfg_dict = ck["model_cfg"]
    state_dict = ck["model_state_dict"]
    variant = ck.get("variant")

    if variant is not None:
        for label, variant_id, cfg_cls, model_cls in _REGISTRY:
            if variant_id == variant:
                keep, _ = _try_load(cfg_cls, model_cls, cfg_dict, state_dict)
                return label, cfg_cls, model_cls, keep

    last_err = None
    for label, _variant_id, cfg_cls, model_cls in _REGISTRY:
        try:
            keep, _ = _try_load(cfg_cls, model_cls, cfg_dict, state_dict)
            return label, cfg_cls, model_cls, keep
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(
        f"could not match ckpt to any registered class "
        f"(variant={variant!r}): last error: {last_err}"
    )


def _resolve_corpus(corpus_arg: str, ckpt_path: Path, max_len: int) -> str:
    """Return the resolved corpus name. `corpus_arg` is one of
    {'auto','shakespeare','tinystories'}; 'auto' uses path + max_len heuristic.
    """
    if corpus_arg in ("shakespeare", "tinystories"):
        return corpus_arg
    pstr = str(ckpt_path).lower()
    if any(tag in pstr for tag in ("shakespeare", "/multi_seed/", "_e1_", "/e1_")):
        return "shakespeare"
    if max_len <= 256:
        return "shakespeare"
    return "tinystories"


def _load_val_ids(corpus: str) -> np.ndarray:
    """Return val_ids for the chosen corpus."""
    if corpus == "shakespeare":
        _train_ids, val_ids = load_tiny_shakespeare()
    elif corpus == "tinystories":
        _train_ids, val_ids = load_tiny_stories(max_train_tokens=5_000_000)
    else:
        raise ValueError(f"unknown corpus: {corpus!r}")
    return val_ids


def main():
    ap = argparse.ArgumentParser(
        description="Measure val-PPL inflation of an SPLM ckpt under the leak fix.",
        allow_abbrev=False,
    )
    ap.add_argument("ckpt", type=Path,
                    help="Path to the .pt checkpoint to forensically evaluate.")
    # Backward-compatible positional n_batches / batch / block.
    ap.add_argument("pos_n_batches", nargs="?", type=int, default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("pos_batch", nargs="?", type=int, default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("pos_block", nargs="?", type=int, default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--n-batches", type=int, default=40,
                    help="Number of val batches to average over (default 40).")
    ap.add_argument("--batch", type=int, default=16,
                    help="Batch size (default 16).")
    ap.add_argument("--block", type=int, default=512,
                    help="Block (sequence) length; auto-capped at ckpt max_len. "
                         "Default 512.")
    ap.add_argument("--corpus", choices=("auto", "shakespeare", "tinystories"),
                    default="auto",
                    help="Validation corpus. 'auto' uses path + max_len heuristic.")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for batch sampling (default 0). The same "
                         "seed is used for both buggy and fixed evaluators "
                         "so they see exactly the same val tokens.")
    ap.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto",
                    help="Compute device. 'auto' uses MPS if available, else "
                         "CPU. Pin to 'cpu' to coexist with concurrent MPS "
                         "training jobs.")
    args = ap.parse_args()

    # Reconcile positional + keyword forms.
    n_batches = args.pos_n_batches if args.pos_n_batches is not None else args.n_batches
    batch = args.pos_batch if args.pos_batch is not None else args.batch
    block = args.pos_block if args.pos_block is not None else args.block

    ckpt_path = args.ckpt
    seed = args.seed

    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    print(f"[ppl-inflation] device={device}  ckpt={ckpt_path.name}")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    label, cfg_cls, model_cls, keep = _dispatch(ck)
    max_len = int(keep.get("max_len", block))
    if block > max_len:
        print(f"[ppl-inflation] block {block} > ckpt max_len {max_len}; "
              f"capping block to {max_len}.")
        block = max_len

    corpus = _resolve_corpus(args.corpus, ckpt_path, max_len)
    print(f"[ppl-inflation] matched: {label}")
    print(f"[ppl-inflation] cfg.d={keep['d']}  L={keep['L']}  "
          f"v_h={keep['v_hidden']}  max_len={max_len}")
    print(f"[ppl-inflation] corpus={corpus}  ({'auto' if args.corpus == 'auto' else 'forced'})")
    print(f"[ppl-inflation] eval: n_batches={n_batches}  batch={batch}  block={block}  seed={seed}")

    val_ids = _load_val_ids(corpus)
    print(f"[ppl-inflation] val tokens: {len(val_ids):,}")

    rng = np.random.default_rng(seed)
    batches = [get_batch(val_ids, batch, block, rng) for _ in range(n_batches)]

    cfg = cfg_cls(**keep)
    ppl_buggy = ppl_fixed = None
    for run_label, causal in [("buggy", False), ("fixed", True)]:
        cfg.causal_force = causal
        m = model_cls(cfg).to(device)
        m.load_state_dict(ck["model_state_dict"])
        m.eval()
        losses = []
        for xb, yb in batches:
            x = torch.from_numpy(xb).to(device)
            y = torch.from_numpy(yb).to(device)
            _, loss = m(x, y)
            losses.append(loss.item())
        l = float(np.mean(losses))
        ppl = math.exp(l)
        print(f"[ppl-inflation] {run_label:>5}: loss={l:.4f}  ppl={ppl:.2f}")
        if run_label == "buggy":
            ppl_buggy = ppl
        else:
            ppl_fixed = ppl
        del m

    assert ppl_buggy is not None and ppl_fixed is not None
    print(f"[ppl-inflation] inflation factor PPL_fixed / PPL_buggy = "
          f"{ppl_fixed / ppl_buggy:.2f}x")


if __name__ == "__main__":
    main()
