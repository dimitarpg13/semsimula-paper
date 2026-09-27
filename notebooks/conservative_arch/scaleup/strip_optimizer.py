#!/usr/bin/env python3
"""Strip `optimizer_state_dict` from training checkpoints for publication.

A training checkpoint carries the AdamW moments so a run can be resumed.
Those are ~2x the weights and are useless to anyone downloading a finished
model, so a published checkpoint should not carry them:

    model_state_dict       293 MB   <- keep
    optimizer_state_dict   586 MB   <- drop
    everything else        tiny     <- keep (step, val_ppl, model_cfg,
                                       train_cfg, gamma, xi_alphas, variant,
                                       corpus, phase, seed)

Everything except the optimiser is preserved, so the stripped file stays
self-describing: `torch.load(...)['model_cfg']` still rebuilds the model and
`['val_ppl']` still says what it scored.

NEVER writes over the input. Refuses if the output exists unless --force.
Verifies after writing: reloads the output and checks every model tensor is
bit-identical to the source.

Usage
-----
    python3 strip_optimizer.py IN.pt [IN2.pt ...]          # -> IN.published.pt
    python3 strip_optimizer.py IN.pt -o OUT.pt
    python3 strip_optimizer.py 'folder/**/*_best.pt' --glob
"""
import argparse
import glob as _glob
import pathlib
import sys

import torch

DROP = ("optimizer_state_dict", "scaler_state_dict", "scheduler_state_dict")


def human(n):
    return f"{n / 1048576:,.0f} MB"


def strip(src: pathlib.Path, dst: pathlib.Path, force: bool) -> bool:
    if dst.exists() and not force:
        print(f"  SKIP {dst.name} exists (use --force to overwrite)")
        return False
    ck = torch.load(src, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "model_state_dict" not in ck:
        print(f"  SKIP {src.name}: not a training checkpoint "
              f"(top-level keys: {list(ck)[:6] if isinstance(ck, dict) else type(ck).__name__})")
        return False
    dropped = [k for k in DROP if k in ck]
    out = {k: v for k, v in ck.items() if k not in DROP}
    torch.save(out, dst)

    # verify: reload and compare every model tensor against the source
    back = torch.load(dst, map_location="cpu", weights_only=False)
    a, b = ck["model_state_dict"], back["model_state_dict"]
    assert a.keys() == b.keys(), "tensor set changed"
    bad = [k for k in a if not torch.equal(a[k], b[k])]
    assert not bad, f"tensors differ after round-trip: {bad[:3]}"
    kept = sorted(set(back) - {"model_state_dict"})

    s_in, s_out = src.stat().st_size, dst.stat().st_size
    print(f"  {src.name}")
    print(f"     {human(s_in)} -> {human(s_out)}   saved {human(s_in - s_out)} "
          f"({100 * (1 - s_out / s_in):.0f}%)")
    print(f"     dropped: {', '.join(dropped) or 'nothing'}")
    print(f"     kept   : model_state_dict + {', '.join(kept)}")
    print(f"     verified: {len(a)} tensors bit-identical after reload")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("-o", "--out", help="output path (single input only)")
    ap.add_argument("--glob", action="store_true", help="treat paths as glob patterns")
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = ap.parse_args()

    files = []
    for p in args.paths:
        files += [pathlib.Path(x) for x in _glob.glob(p, recursive=True)] if args.glob else [pathlib.Path(p)]
    files = [f for f in files if f.suffix == ".pt" and ".published" not in f.name]
    if not files:
        sys.exit("no .pt inputs matched")
    if args.out and len(files) > 1:
        sys.exit("-o takes a single input")

    total_in = total_out = 0
    for f in files:
        dst = pathlib.Path(args.out) if args.out else f.with_suffix(".published.pt")
        if strip(f, dst, args.force):
            total_in += f.stat().st_size
            total_out += dst.stat().st_size
    if total_in:
        print(f"\ntotal {human(total_in)} -> {human(total_out)}  "
              f"saved {human(total_in - total_out)}")


if __name__ == "__main__":
    main()
