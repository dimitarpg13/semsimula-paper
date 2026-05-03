"""Causal-violation probe for SPLM-family language models.

Tests whether perturbing token x[t_pert] changes any logit at position
t < t_pert.  In a properly causal autoregressive model this MUST be
exactly zero.

Background
==========
The SPLM `integrate()` method computes a per-step conservative force as

    f(t) = -∂(Σ_t' V[t']) / ∂h[t]

where V[t'] = V_θ(ξ[t'], h[t']).  When ξ is a causal weighted average
that includes h[t] (e.g. cumulative mean, EMA), differentiating
the *summed* V picks up off-diagonal contributions ∂V[t']/∂h[t] for
t' > t.  The trained V_θ then learns to route information through this
anti-causal channel, leaking future tokens into past predictions and
inflating val PPL by orders of magnitude.

Fix: compute ξ from `h.detach()` so the autograd path from ξ back to h
is severed.  All model configs in `notebooks/conservative_arch/*/model_*.py`
expose a `causal_force: bool = True` field controlling this.

This script is the regression-test counterpart of the fix.  It runs in
two modes:

  1. *Class smoke* (no checkpoint).  Random-init each registered model
     class with both `causal_force=True` and `causal_force=False`,
     perturb a token, and report Δ on the causal side.  The fixed
     variant must produce Δ ≡ 0.0 exactly.

  2. *Checkpoint probe* (one ckpt path argument).  Load a trained
     checkpoint with the model class implied by its `experiment` /
     `variant` field, and run the same probe.

Usage
-----
  python3 notebooks/conservative_arch/causal_probe.py
      --> runs class smoke for every registered SPLM model class

  python3 notebooks/conservative_arch/causal_probe.py <splm_ckpt.pt>
      --> runs SPLM checkpoint probe in both buggy and fixed modes,
          reports Δ on causal side and val-PPL inflation factor

  python3 notebooks/conservative_arch/causal_probe.py <matched_gpt_ckpt.pt>
      --> runs single-mode probe on a MatchedGPT checkpoint
          (natural transformer; no causal_force flag exists)

  python3 notebooks/conservative_arch/causal_probe.py --hf gpt2
      --> runs single-mode probe on the pretrained GPT-2 small
          loaded from the HuggingFace Hub

  python3 notebooks/conservative_arch/causal_probe.py --hf EleutherAI/pythia-160m
      --> same, for Pythia-160M

  python3 notebooks/conservative_arch/causal_probe.py --all-natural
      --> convenience: probes pretrained gpt2 + EleutherAI/pythia-160m,
          plus the most-recent MatchedGPT ckpt found under
          notebooks/conservative_arch/{scaleup,multi_seed}/results/

  python3 notebooks/conservative_arch/causal_probe.py --strict
      --> like default but exits non-zero if any class still leaks
          in fixed mode (suitable for a CI gate)

References
----------
  - docs/Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md
  - notebooks/conservative_arch/inference_efficiency/splm_streaming_decode.py
    (prior knowledge of the bug, applied only to streaming inference)
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

THIS_DIR = Path(__file__).parent
for sub in ("", "energetic_minima", "sarf_mass_variant", "sarf_variant",
            "symplectic_variant", "first_order_ablation", "multixi"):
    sys.path.insert(0, str(THIS_DIR / sub))


# ---------------------------------------------------------------------------
# Model class registry
# ---------------------------------------------------------------------------

@dataclass
class ModelEntry:
    name: str
    cfg_factory: Callable[[Optional[str]], object]
    cls_factory: Callable[[object], torch.nn.Module]


def _logfreq_path() -> str:
    """Return path to the TinyStories logfreq surprisal cache (built by E9)."""
    return str(
        THIS_DIR / "scaleup" / "results" / "logfreq_surprisal_tinystories.npy"
    )


def _registry() -> List[ModelEntry]:
    """Lazily build registry; imports are deferred until called.

    Order matters: ckpt-probe matches the FIRST entry whose config can
    accept the ckpt's `model_cfg` and whose state_dict load succeeds.
    Most-specific classes must be listed first.
    """
    out: List[ModelEntry] = []

    # E12 HiPPO multi-channel ξ — most specific (HiPPO state-space replaces
    # the K-EMA bank). Listed first so its config-shape dispatch wins over
    # the K-EMA variant when a HiPPO ckpt is loaded.
    from model_multixi_hippo import (  # type: ignore
        SPLMSARFMassLNMultiHiPPOConfig, ScalarPotentialLMSARFMassLNMultiHiPPO,
    )

    def _cfg_multihippo(causal: Optional[bool]) -> object:
        return SPLMSARFMassLNMultiHiPPOConfig(
            vocab_size=257, d=16, max_len=64, v_hidden=32, v_depth=2, L=4,
            mass_mode="global", ln_after_step=True,
            xi_channels=4, xi_basis="legt", xi_theta=64.0,
            causal_force=True if causal is None else causal,
        )
    out.append(ModelEntry(
        "multixi-hippo (K=4 LegT)", _cfg_multihippo,
        ScalarPotentialLMSARFMassLNMultiHiPPO,
    ))

    def _cfg_multihippo_learndt(causal: Optional[bool]) -> object:
        return SPLMSARFMassLNMultiHiPPOConfig(
            vocab_size=257, d=16, max_len=64, v_hidden=32, v_depth=2, L=4,
            mass_mode="global", ln_after_step=True,
            xi_channels=4, xi_basis="legt", xi_theta=64.0,
            xi_learnable_dt=True,
            causal_force=True if causal is None else causal,
        )
    out.append(ModelEntry(
        "multixi-hippo (K=4 LegT, learnable dt)", _cfg_multihippo_learndt,
        ScalarPotentialLMSARFMassLNMultiHiPPO,
    ))

    # E13 S4D multi-channel ξ — diagonal-complex-A learnable basis (R6.i+).
    # Most specific config-shape variant within the multixi family.
    from model_multixi_s4d import (  # type: ignore
        SPLMSARFMassLNMultiS4DConfig, ScalarPotentialLMSARFMassLNMultiS4D,
    )

    def _cfg_multis4d_legt(causal: Optional[bool]) -> object:
        return SPLMSARFMassLNMultiS4DConfig(
            vocab_size=257, d=16, max_len=64, v_hidden=32, v_depth=2, L=4,
            mass_mode="global", ln_after_step=True,
            xi_channels=4, xi_theta=64.0, xi_eigval_init="legt",
            xi_learnable_dt=True, xi_learnable_B=True,
            causal_force=True if causal is None else causal,
        )
    out.append(ModelEntry(
        "multixi-s4d (K=4, legt-init)", _cfg_multis4d_legt,
        ScalarPotentialLMSARFMassLNMultiS4D,
    ))

    def _cfg_multis4d_lin(causal: Optional[bool]) -> object:
        return SPLMSARFMassLNMultiS4DConfig(
            vocab_size=257, d=16, max_len=64, v_hidden=32, v_depth=2, L=4,
            mass_mode="global", ln_after_step=True,
            xi_channels=4, xi_theta=64.0, xi_eigval_init="s4d_lin",
            xi_learnable_dt=True, xi_learnable_B=True,
            causal_force=True if causal is None else causal,
        )
    out.append(ModelEntry(
        "multixi-s4d (K=4, s4d_lin-init)", _cfg_multis4d_lin,
        ScalarPotentialLMSARFMassLNMultiS4D,
    ))

    # E11 multi-channel ξ — K-EMA bank (the previous, redundant baseline).
    from model_multixi import (  # type: ignore
        SPLMSARFMassLNMultiXiConfig, ScalarPotentialLMSARFMassLNMultiXi,
    )

    def _cfg_multixi(causal: Optional[bool]) -> object:
        return SPLMSARFMassLNMultiXiConfig(
            vocab_size=257, d=16, max_len=64, v_hidden=32, v_depth=2, L=4,
            mass_mode="global", ln_after_step=True,
            xi_channels=4, xi_alpha_inits=[0.0, 0.5, 0.9, 0.99],
            xi_learnable=True,
            causal_force=True if causal is None else causal,
        )
    out.append(ModelEntry(
        "multixi (K=4 EMAs)", _cfg_multixi, ScalarPotentialLMSARFMassLNMultiXi,
    ))

    # R6.h.1 (Fix 2): K-EMA bank with log-spaced α-init.
    # The init mode is the only difference from the explicit-α variant
    # above; α-values are computed from K and τ_max at __init__ time.
    # No new gradient pathway → causality is identical, but we still
    # exercise this through the strict probe per regression discipline.
    def _cfg_multixi_logspaced(causal: Optional[bool]) -> object:
        return SPLMSARFMassLNMultiXiConfig(
            vocab_size=257, d=16, max_len=64, v_hidden=32, v_depth=2, L=4,
            mass_mode="global", ln_after_step=True,
            xi_channels=4, xi_alpha_inits=[0.0, 0.0, 0.0, 0.0],
            xi_alpha_init_mode="log_spaced", xi_tau_max=100.0,
            xi_learnable=True,
            causal_force=True if causal is None else causal,
        )
    out.append(ModelEntry(
        "multixi (K=4 EMAs, log-spaced α, τ_max=100)",
        _cfg_multixi_logspaced, ScalarPotentialLMSARFMassLNMultiXi,
    ))

    # E9 SPLM with LayerNorm-after-step (has ln_eps / ln_affine fields).
    from model_ln import (  # type: ignore
        SPLMSARFMassLNConfig, ScalarPotentialLMSARFMassLN,
    )

    def _cfg_ln(causal: Optional[bool]) -> object:
        return SPLMSARFMassLNConfig(
            vocab_size=257, d=16, max_len=64, v_hidden=32, v_depth=2, L=4,
            mass_mode="global", ln_after_step=True,
            causal_force=True if causal is None else causal,
        )
    out.append(ModelEntry(
        "sarf_mass_ln (LN-after-step)", _cfg_ln, ScalarPotentialLMSARFMassLN,
    ))

    # First-order ablation (γ=∞, gradient flow). Same state_dict shape as
    # sarf_mass_ln but a different integrator class. Listed after _ln so
    # that LN ckpts go to the right class by default.
    from model_first_order import (  # type: ignore
        SPLMFirstOrderConfig, ScalarPotentialLMFirstOrder,
    )

    def _cfg_first_order(causal: Optional[bool]) -> object:
        return SPLMFirstOrderConfig(
            vocab_size=257, d=16, max_len=64, v_hidden=32, v_depth=2, L=4,
            mass_mode="global", ln_after_step=True,
            causal_force=True if causal is None else causal,
        )
    out.append(ModelEntry(
        "first_order (gradient-flow)", _cfg_first_order, ScalarPotentialLMFirstOrder,
    ))

    # Symplectic / Verlet integrator.
    from model_symplectic import (  # type: ignore
        SPLMSymplecticConfig, ScalarPotentialLMSymplectic,
    )

    def _cfg_sym(causal: Optional[bool]) -> object:
        return SPLMSymplecticConfig(
            vocab_size=257, d=16, max_len=64, v_hidden=32, v_depth=2, L=4,
            mass_mode="global",
            causal_force=True if causal is None else causal,
        )
    out.append(ModelEntry(
        "symplectic (velocity-Verlet)", _cfg_sym, ScalarPotentialLMSymplectic,
    ))

    # SARF-mass variant (parent of every newer SPLM in this repo).
    from model_sarf_mass import (  # type: ignore
        SPLMSARFMassConfig, ScalarPotentialLMSARFMass,
    )

    def _cfg_sarf_mass(causal: Optional[bool]) -> object:
        return SPLMSARFMassConfig(
            vocab_size=257, d=16, max_len=64, v_hidden=32, v_depth=2, L=4,
            mass_mode="global",
            causal_force=True if causal is None else causal,
        )
    out.append(ModelEntry(
        "sarf_mass (global mass)", _cfg_sarf_mass, ScalarPotentialLMSARFMass,
    ))

    # SARF (no mass head)
    from model_sarf import SPLMSARFConfig, ScalarPotentialLMSARF  # type: ignore

    def _cfg_sarf(causal: Optional[bool]) -> object:
        return SPLMSARFConfig(
            vocab_size=257, d=16, max_len=64, v_hidden=32, v_depth=2, L=4,
            causal_force=True if causal is None else causal,
        )
    out.append(ModelEntry(
        "sarf (no mass head)", _cfg_sarf, ScalarPotentialLMSARF,
    ))

    return out


# ---------------------------------------------------------------------------
# Probe core
# ---------------------------------------------------------------------------

def causal_violation_probe(
    model: torch.nn.Module,
    vocab_size: int,
    T: int = 32,
    t_pert: int = 20,
    seed: int = 0,
) -> Tuple[float, float, torch.Tensor]:
    """Return (max Δ on causal side, max Δ on after-side, full Δ vector)."""
    rng = np.random.default_rng(seed)
    xb = rng.integers(0, vocab_size, size=(1, T)).astype(np.int64)
    x_a = torch.from_numpy(xb)
    x_b = x_a.clone()
    orig = int(x_b[0, t_pert].item())
    new = (orig + 17) % vocab_size
    x_b[0, t_pert] = new

    model.eval()
    out_a = model(x_a)
    out_b = model(x_b)
    logits_a = out_a[0].detach()
    logits_b = out_b[0].detach()
    diffs = (logits_a - logits_b).abs().max(dim=-1).values[0]
    pre = float(diffs[:t_pert].max().item())
    post = float(diffs[t_pert + 1:].max().item())
    return pre, post, diffs


def class_smoke(strict: bool) -> int:
    print("=" * 76)
    print(" Class smoke probe — random-init each model in causal_force={True,False}")
    print("=" * 76)
    fails = 0
    for entry in _registry():
        torch.manual_seed(0)
        model_buggy = entry.cls_factory(entry.cfg_factory(False))
        torch.manual_seed(0)
        model_fixed = entry.cls_factory(entry.cfg_factory(True))
        pre_b, _, _ = causal_violation_probe(model_buggy, vocab_size=257)
        pre_f, _, _ = causal_violation_probe(model_fixed, vocab_size=257)
        # On random init the buggy variant has only numerical-noise leak;
        # the meaningful regression is "fixed must give exactly 0".
        ok_fixed = pre_f < 1e-6
        ok_diff = pre_b > pre_f or pre_b < 1e-6  # accept either trace
        verdict = "OK" if ok_fixed else "FAIL"
        print(
            f"  [{verdict:>4}]  {entry.name:<32}  "
            f"  buggy Δ={pre_b:.2e}   fixed Δ={pre_f:.2e}"
        )
        if not ok_fixed:
            fails += 1

    print("-" * 76)
    if fails == 0:
        print(f"  All {len(_registry())} model classes: fixed-mode Δ ≡ 0.")
        return 0
    print(f"  {fails} model class(es) still leak in fixed mode.")
    return 1 if strict else 0


# ---------------------------------------------------------------------------
# Natural-transformer probe (MatchedGPT ckpt or HF Hub model)
# ---------------------------------------------------------------------------

class _LogitsAdapter(torch.nn.Module):
    """Wrap an arbitrary causal-LM so its forward returns the (logits, None)
    tuple that `causal_violation_probe` expects.

    `extract` maps the wrapped model's raw output to a (B, T, V) logits tensor.
    """
    def __init__(self, model: torch.nn.Module, extract: Callable[..., torch.Tensor]):
        super().__init__()
        self.model = model
        self._extract = extract

    def forward(self, x: torch.Tensor):
        return self._extract(self.model(x)), None


def _load_matched_gpt(ckpt_path: Path) -> Tuple[str, torch.nn.Module, int]:
    """Load a MatchedGPT ckpt and wrap it for the probe."""
    from matched_baseline_model import MatchedConfig, MatchedGPT  # type: ignore
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ck.get("model_cfg") or ck.get("model_config") or {}
    keep = {k: v for k, v in cfg_dict.items()
            if k in MatchedConfig.__dataclass_fields__}
    cfg = MatchedConfig(**keep)
    m = MatchedGPT(cfg)
    m.load_state_dict(ck["model_state_dict"])
    label = f"MatchedGPT  ckpt={ckpt_path.name}  (d={cfg.d}, L={cfg.n_layer}, H={cfg.n_head})"
    # MatchedGPT.forward returns (logits, loss_or_None); take element 0.
    wrapped = _LogitsAdapter(m, extract=lambda out: out[0])
    return label, wrapped, cfg.vocab_size


def _load_hf(name: str) -> Tuple[str, torch.nn.Module, int]:
    """Load a HuggingFace causal-LM by hub name and wrap it for the probe."""
    from transformers import AutoModelForCausalLM  # type: ignore
    print(f"[hf-load] {name}  (downloading on first use)")
    m = AutoModelForCausalLM.from_pretrained(name)
    vocab = int(m.config.vocab_size)
    n_layer = getattr(m.config, "num_hidden_layers", None) \
              or getattr(m.config, "n_layer", "?")
    label = f"HF: {name}  (vocab={vocab}, L={n_layer})"
    wrapped = _LogitsAdapter(m, extract=lambda out: out.logits)
    return label, wrapped, vocab


def _is_matched_gpt_ckpt(ckpt_path: Path) -> bool:
    """Best-effort detector: MatchedGPT cfg has n_layer/n_head/mlp_mult fields."""
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg_dict = ck.get("model_cfg") or ck.get("model_config") or {}
        return ("n_layer" in cfg_dict and "n_head" in cfg_dict
                and "mlp_mult" in cfg_dict)
    except Exception:
        return False


def _find_recent_matched_ckpt() -> Optional[Path]:
    """Walk the results trees and return the newest *.pt that smells like
    MatchedGPT. Used by --all-natural to add a local checkpoint to the probe set.
    """
    candidates: List[Path] = []
    roots = [
        THIS_DIR / "scaleup" / "results",
        THIS_DIR / "multi_seed" / "results",
        THIS_DIR / "inference_efficiency" / "results",
    ]
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.pt"):
            if _is_matched_gpt_ckpt(p):
                candidates.append(p)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def natural_probe(targets: List[str], strict: bool) -> int:
    """Single-mode causal-violation probe for natural transformers.

    Each target is either a path to a MatchedGPT .pt ckpt OR an HF Hub
    model name (e.g. 'gpt2', 'EleutherAI/pythia-160m'). For each target,
    runs `causal_violation_probe` in single mode and reports Δ.

    Expected: causal-side Δ ≡ 0 by construction of the causal mask. A
    non-zero Δ would indicate either a numerical artefact or a hidden bug
    analogous to the SPLM leak — flagged for investigation.
    """
    print("=" * 76)
    print(" Natural-transformer causal-violation probe")
    print(" (expected: causal-side Δ ≡ 0 by construction of the causal mask)")
    print("=" * 76)
    fails = 0
    for target in targets:
        try:
            p = Path(target)
            if p.exists() and p.suffix == ".pt":
                label, model, vocab = _load_matched_gpt(p)
            else:
                label, model, vocab = _load_hf(target)
        except Exception as e:
            print(f"  [SKIP]  {target!r}: {type(e).__name__}: {e}")
            continue
        T = 32
        t_pert = 20
        pre, post, _ = causal_violation_probe(
            model, vocab_size=vocab, T=T, t_pert=t_pert, seed=7,
        )
        ok = pre < 1e-4
        verdict = "OK" if ok else "FAIL"
        print(f"  [{verdict:>4}]  {label}")
        print(f"           causal-side Δ={pre:.2e}    after-side Δ={post:.2e}")
        if not ok:
            fails += 1
        del model

    print("-" * 76)
    if fails == 0:
        print(f"  All {len(targets)} natural-transformer targets: causal-side Δ ≈ 0.")
        return 0
    print(f"  {fails} natural-transformer target(s) showed unexpected causal violation.")
    return 1 if strict else 0


# ---------------------------------------------------------------------------
# Optional ckpt probe (for trained models).  We don't dispatch by experiment
# name — we re-use the class registry above and try each.
# ---------------------------------------------------------------------------

def ckpt_probe(ckpt_path: Path, strict: bool) -> int:
    print(f"[ckpt-probe] loading: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ck.get("model_cfg") or ck.get("model_config")
    if cfg_dict is None:
        print("[ckpt-probe] no 'model_cfg' / 'model_config' key in ckpt; aborting")
        return 2

    # Heuristic: try to load with each registered model class until one fits.
    model_buggy = None
    model_fixed = None
    matched = None
    for entry in _registry():
        try:
            cfg_obj = entry.cfg_factory(False)
            for k, v in cfg_dict.items():
                if hasattr(cfg_obj, k):
                    setattr(cfg_obj, k, v)
            setattr(cfg_obj, "causal_force", False)
            model_buggy = entry.cls_factory(cfg_obj)
            model_buggy.load_state_dict(ck["model_state_dict"])
        except Exception:
            continue
        try:
            cfg_obj_f = entry.cfg_factory(True)
            for k, v in cfg_dict.items():
                if hasattr(cfg_obj_f, k):
                    setattr(cfg_obj_f, k, v)
            setattr(cfg_obj_f, "causal_force", True)
            model_fixed = entry.cls_factory(cfg_obj_f)
            model_fixed.load_state_dict(ck["model_state_dict"])
        except Exception:
            continue
        matched = entry
        break

    if matched is None or model_buggy is None or model_fixed is None:
        print("[ckpt-probe] could not match the ckpt to any registered model class")
        return 2

    print(f"[ckpt-probe] matched: {matched.name}")
    vocab = cfg_dict.get("vocab_size", 257)
    T = min(64, cfg_dict.get("max_len", 64))
    t_pert = min(40, T - 4)
    pre_b, post_b, _ = causal_violation_probe(
        model_buggy, vocab_size=vocab, T=T, t_pert=t_pert, seed=7,
    )
    pre_f, post_f, _ = causal_violation_probe(
        model_fixed, vocab_size=vocab, T=T, t_pert=t_pert, seed=7,
    )
    print(f"  buggy : causal-side Δ = {pre_b:.4e}    after-side Δ = {post_b:.4e}")
    print(f"  fixed : causal-side Δ = {pre_f:.4e}    after-side Δ = {post_f:.4e}")
    if pre_f > 1e-4:
        print("  ** REGRESSION: fixed mode still leaks; do not trust this ckpt. **")
        return 1 if strict else 0
    if pre_b > 1e-2:
        print(f"  -> ckpt is from BUGGY training (causal-side Δ = {pre_b:.2f}).")
        print(f"     The fixed evaluator gives a leak-free reading "
              f"({pre_f:.2e}); compare val PPL under both modes for inflation.")
    else:
        print("  -> ckpt looks clean (no exploitable leak under either mode).")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Causal-violation probe for SPLM and natural-transformer models."
    )
    ap.add_argument("ckpt", nargs="?", default=None,
                    help="Optional path to a .pt checkpoint. If omitted "
                         "(and no --hf / --all-natural / --natural is given), "
                         "runs class smoke for every registered SPLM model class.")
    ap.add_argument("--strict", action="store_true",
                    help="Exit non-zero if any class leaks in fixed mode.")
    ap.add_argument(
        "--hf", action="append", default=None, metavar="HUB_NAME",
        help="HuggingFace Hub name of a natural transformer to probe "
             "(e.g. 'gpt2', 'EleutherAI/pythia-160m'). Repeatable.",
    )
    ap.add_argument(
        "--natural", action="append", default=None, metavar="PATH_OR_HUB",
        help="Natural-transformer target: a .pt MatchedGPT ckpt path OR an "
             "HF hub name. Repeatable.",
    )
    ap.add_argument(
        "--all-natural", action="store_true",
        help="Convenience: probe pretrained gpt2 + EleutherAI/pythia-160m "
             "and the most recent local MatchedGPT ckpt under "
             "scaleup/multi_seed/inference_efficiency results trees.",
    )
    args = ap.parse_args()

    if args.all_natural:
        targets = ["gpt2", "EleutherAI/pythia-160m"]
        local = _find_recent_matched_ckpt()
        if local is not None:
            targets.append(str(local))
        else:
            print("[--all-natural] no local MatchedGPT ckpt found; "
                  "running on HF targets only")
        return natural_probe(targets, strict=args.strict)

    natural_targets: List[str] = []
    if args.natural:
        natural_targets.extend(args.natural)
    if args.hf:
        natural_targets.extend(args.hf)

    if natural_targets:
        return natural_probe(natural_targets, strict=args.strict)

    # No natural-transformer flag: fall back to SPLM logic.
    if args.ckpt is None:
        return class_smoke(strict=args.strict)

    # Auto-route: a MatchedGPT ckpt path given as positional → natural probe.
    p = Path(args.ckpt)
    if p.exists() and p.suffix == ".pt" and _is_matched_gpt_ckpt(p):
        return natural_probe([str(p)], strict=args.strict)

    return ckpt_probe(p, strict=args.strict)


if __name__ == "__main__":
    sys.exit(main())
