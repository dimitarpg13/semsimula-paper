#!/usr/bin/env python3
"""
Standalone training script for FockPARFLM v2.1 and matched GPT-2 baselines.

Faithfully reproduces the Colab notebook training logic in a headless,
multi-GPU-ready format.  Supports:

  - FockPARFLM v2.1 with depth-conditioned Gaussian V_theta
  - Matched GPT-2 (MatchedGPT) baseline for controlled comparison
  - Single-GPU and multi-GPU (PyTorch DDP) training
  - WSD (Warmup-Stable-Decay) and cosine LR schedules
  - Per-group gradient clipping with spike debugging
  - Checkpoint save / resume with optimizer state
  - JSONL training logs

Usage examples:

  # Current small model (d=384 L=16) — reproduces Colab run
  python train_fock.py --preset d384

  # Scaled-up model (d=768 L=12) — matches GPT-2 Small depth
  python train_fock.py --preset d768

  # Matched GPT-2 Small baseline
  python train_fock.py --preset gpt2-small

  # Multi-GPU via torchrun
  torchrun --nproc_per_node=2 train_fock.py --preset d768
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import sys
import time
import typing
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def is_ddp() -> bool:
    return dist.is_available() and dist.is_initialized()

def rank() -> int:
    return dist.get_rank() if is_ddp() else 0

def world_size() -> int:
    return dist.get_world_size() if is_ddp() else 1

def is_main() -> bool:
    return rank() == 0

def setup_ddp():
    if "RANK" not in os.environ:
        return
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    if is_ddp():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Training config (all hyperparameters in one place)
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Model selection
    model_type: str = "fock"  # "fock" or "gpt2"

    # Architecture
    d: int = 384
    L: int = 16
    n_registers: int = 32
    max_len: int = 1024
    vocab_size: int = 50257

    # V_theta
    v_theta_variant: str = "gaussian"
    v_theta_n_heads: int = 5
    v_theta_wells_per_head: int = 8
    v_theta_depth_condition: bool = True
    v_theta_depth_code_init_std: float = 0.02
    w_scale: float = 1.0

    # Xi channels
    xi_override: str = "5long"
    xi_alpha_inits: List[float] = field(default_factory=lambda: [0.50, 0.75, 0.95, 0.99, 0.995])

    # V_phi
    v_phi_kind: str = "structural_competitive"
    v_phi_d_type: int = 32
    v_phi_d_angle: int = 16
    v_phi_n_heads: int = 4
    v_phi_mlp_hidden: int = 128
    top_k: int = 16

    # Fock
    fock_version: str = "v2"
    d_k: int = 64
    tau_create_init: float = 8.0
    register_salience_decay: float = 0.5
    register_salience_threshold: float = 0.005
    creation_gate_hidden: int = 64
    destruction_gate_hidden: int = 64
    reverse_channel: bool = True
    reverse_channel_stable: bool = True
    reverse_channel_pre_ln: bool = True
    reverse_channel_soft_norm: bool = True
    reverse_channel_warmup_steps: int = 4000
    reverse_channel_per_layer: bool = True
    register_repulsion: bool = True
    register_repulsion_coeff: float = 0.05
    register_repulsion_kind: str = "gram"

    # Integrator
    fixed_gamma: float = 0.30
    init_gamma: float = 1.0
    force_clamp_max: float = 0.0  # 0 = disabled; positive value clamps force per dim

    # Output
    use_output_bias: bool = True
    tie_embeddings: bool = False

    # GPT-2 baseline specific
    gpt2_n_head: int = 12
    gpt2_mlp_mult: int = 4

    # Optimizer
    optimizer: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    grad_clip_vphi: float = 0.3

    # Schedule
    lr_schedule: str = "wsd"
    wsd_warmup_frac: float = 0.05
    wsd_stable_frac: float = 0.60
    wsd_lr_floor: Optional[float] = None  # defaults to lr * 0.05
    warmup_steps: int = 5000  # for cosine schedule

    # Training
    total_steps: int = 244_000
    batch_size: int = 16
    grad_accum: int = 2
    block_size: int = 512
    lambda_v: float = 1e-2
    bf16: bool = False

    # Data
    max_train_tokens: int = 4_000_000_000
    val_tokens: int = 2_000_000
    data_dir: str = ""  # auto-resolved
    corpus: str = "openwebtext"

    # Logging / checkpointing
    eval_interval: int = 500
    eval_iters: int = 40
    log_interval: int = 50
    ckpt_interval: int = 7500
    output_dir: str = ""  # auto-resolved
    seed: int = 0

    # Gradient clipping
    per_group_clip: bool = True
    grad_spike_debug: bool = True
    grad_spike_threshold: float = 100.0
    grad_norm_ema_alpha: float = 0.05
    grad_norm_ema_threshold: float = 50.0
    grad_norm_ema_patience: int = 200

    # Remote sync (Google Drive via rclone, or rsync to another host)
    # rclone example: "gdrive:semsimula_runs/fock_d768"
    # rsync example:  "user@host:/path/to/runs/fock_d768"
    # Empty string = no sync.
    sync_remote: str = ""

    # Resume
    resume_from: str = ""  # path to checkpoint

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum * world_size()

    @property
    def xi_channels(self) -> int:
        return len(self.xi_alpha_inits)

    def resolve_wsd_lr_floor(self):
        if self.wsd_lr_floor is None:
            self.wsd_lr_floor = self.lr * 0.05

    def resolve_xi_override(self):
        presets = {
            5: [0.25, 0.50, 0.75, 0.95, 0.99],
            "5long": [0.50, 0.75, 0.95, 0.99, 0.995],
            6: [0.25, 0.50, 0.75, 0.95, 0.99, 0.995],
            "4long": [0.50, 0.75, 0.95, 0.995],
        }
        if self.xi_override in presets:
            self.xi_alpha_inits = presets[self.xi_override]
        elif self.xi_override == "none":
            self.xi_alpha_inits = [0.25, 0.50, 0.75, 0.95]
        self.v_theta_n_heads = len(self.xi_alpha_inits)


# ---------------------------------------------------------------------------
# Preset configs
# ---------------------------------------------------------------------------

PRESETS: Dict[str, Dict[str, Any]] = {
    # Current running config (reproduces Colab d=384 run).
    # Uses untied embeddings as in the original notebook.
    "d384": {
        "model_type": "fock",
        "d": 384, "L": 16, "n_registers": 32,
        "tie_embeddings": False,
        "total_steps": 100_000,
        "lr": 3e-4,
        "batch_size": 16, "grad_accum": 2,
    },
    # Strategy 1: same d=768, same L=12 as GPT-2 Small.
    # Untied embeddings are required for Fock — tied head causes a
    # long-tail pathology where rare tokens get worse-than-uniform CE
    # (D0.4 diagnostic in Xi_Bottleneck_Diagnosis_Phase5.md §8.4).
    # Fock ~137M params vs GPT-2 ~124M (tied) at same d and L.
    # batch_size=4 (was 8) — d=768 OOMs at bs=8 on 80 GB H100;
    # auto-probe would find 4 anyway, but starting there saves a
    # failed-step and the VRAM churn.
    "d768": {
        "model_type": "fock",
        "d": 768, "L": 12, "n_registers": 32,
        "d_k": 192,
        "tie_embeddings": False,
        "use_output_bias": True,
        "total_steps": 100_000,
        "lr": 2e-4,
        "batch_size": 4, "grad_accum": 8,
    },
    # Strategy 1: same d=1024, same L=24 as GPT-2 Medium.
    # Same untied rationale as d768.
    # Fock ~209M params vs GPT-2 ~355M (tied) at same d and L.
    # batch_size=2 — probe ceiling.  DDP does NOT pool VRAM across GPUs
    # (each rank holds a full model replica), so this must fit on a
    # SINGLE 80 GB H100.  With use_layer_checkpoint=True the per-layer
    # activation footprint is O(1) instead of O(L), which should make
    # bs=2 feasible (~35-45 GB estimated).  The probe will fall back to
    # bs=1 / grad_accum=16 if it still OOMs (eff_batch preserved).
    # grad_clip=0.5 (was 1.0) — L=24 produces steeper gradient cascades
    # than L=12; the d=1024 gamma sweep showed grad spikes of O(10^3)
    # that triggered the watchdog repeatedly.  Tighter clipping dampens
    # these without hurting convergence (the WSD warmup is 5000 steps at
    # total_steps=100K, much gentler than the sweep's 150-step warmup).
    "d1024": {
        "model_type": "fock",
        "d": 1024, "L": 24, "n_registers": 32,
        "d_k": 256,
        "tie_embeddings": False,
        "use_output_bias": True,
        "total_steps": 100_000,
        "lr": 1.5e-4,
        "batch_size": 2, "grad_accum": 16,
        "grad_clip": 0.5,
    },
    # Matched GPT-2 Small baseline (d=768, L=12, 12 heads).
    # Always tied (MatchedGPT hardcodes weight tying).
    # GPT-2 has ~2x fewer activations per layer vs Fock (no V_theta
    # sub-network), so bs=8 may fit, but we start at 4 for parity
    # and let the probe bump it up if headroom exists.
    "gpt2-small": {
        "model_type": "gpt2",
        "d": 768, "L": 12, "gpt2_n_head": 12,
        "tie_embeddings": True,
        "total_steps": 100_000,
        "lr": 2e-4,
        "batch_size": 4, "grad_accum": 8,
        "lambda_v": 0.0,
        "per_group_clip": False,
    },
    # Matched GPT-2 Medium baseline (d=1024, L=24, 16 heads).
    "gpt2-medium": {
        "model_type": "gpt2",
        "d": 1024, "L": 24, "gpt2_n_head": 16,
        "tie_embeddings": True,
        "total_steps": 100_000,
        "lr": 1.5e-4,
        "batch_size": 2, "grad_accum": 16,
        "lambda_v": 0.0,
        "per_group_clip": False,
    },
    # Gamma sweep presets — use with --gamma_sweep.
    # Inherits all arch from the base preset but is sized for
    # a quick 3K-step sweep.
    "sweep-d768": {
        "model_type": "fock",
        "d": 768, "L": 12, "n_registers": 32,
        "d_k": 192,
        "tie_embeddings": False,
        "use_output_bias": True,
        "lr": 2e-4,
        "batch_size": 4, "grad_accum": 8,
    },
    "sweep-d1024": {
        "model_type": "fock",
        "d": 1024, "L": 24, "n_registers": 32,
        "d_k": 256,
        "tie_embeddings": False,
        "use_output_bias": True,
        "lr": 1.5e-4,
        "batch_size": 1, "grad_accum": 32,
    },
}


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _resolve_paths(cfg: TrainConfig, script_dir: Path):
    """Set up CA_DIR, sys.path, data dir, output dir."""
    ca_dir = script_dir.parent
    for sub in ["", "parf", "multixi", "scaleup", "sarf_mass_variant", "energetic_minima"]:
        d = str(ca_dir / sub) if sub else str(ca_dir)
        if d not in sys.path:
            sys.path.insert(0, d)

    if not cfg.data_dir:
        cfg.data_dir = str(ca_dir / "data")
    Path(cfg.data_dir).mkdir(parents=True, exist_ok=True)

    if not cfg.output_dir:
        tag = f"{cfg.model_type}_d{cfg.d}_L{cfg.L}"
        if cfg.model_type == "fock":
            tag += f"_M{cfg.n_registers}"
        cfg.output_dir = str(script_dir / "runs" / tag)

    out = Path(cfg.output_dir)
    if is_main():
        (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    return ca_dir


def build_fock_model(cfg: TrainConfig, device: str, logfreq_path: str):
    from model_fock_parf_multixi import FockMultiXiPARFLM, FockMultiXiPARFConfig

    model_cfg = FockMultiXiPARFConfig(
        vocab_size=cfg.vocab_size, d=cfg.d, max_len=cfg.max_len,
        L=cfg.L, v_hidden=1024, v_depth=3, dt=1.0,
        mass_mode="logfreq",
        logfreq_path=logfreq_path,
        logfreq_init_alpha=0.1,
        init_gamma=cfg.init_gamma,
        fixed_gamma=cfg.fixed_gamma,
        force_clamp_max=cfg.force_clamp_max if cfg.force_clamp_max > 0 else None,
        causal_force=True,
        ln_after_step=True,
        xi_channels=cfg.xi_channels,
        xi_alpha_inits=cfg.xi_alpha_inits,
        xi_learnable=True,
        xi_alpha_init_mode="explicit",
        v_phi_kind=cfg.v_phi_kind,
        v_phi_d_type=cfg.v_phi_d_type,
        v_phi_d_angle=cfg.v_phi_d_angle,
        v_phi_eps=0.1,
        v_phi_phi_hidden=128,
        v_phi_theta_hidden=128,
        v_phi_mlp_hidden=cfg.v_phi_mlp_hidden,
        top_k=cfg.top_k,
        v_phi_n_heads=cfg.v_phi_n_heads,
        use_output_bias=cfg.use_output_bias,
        tie_embeddings=cfg.tie_embeddings,
        score_head_hidden=32,
        gumbel_tau_init=1.0,
        gumbel_tau_min=0.3,
        gumbel_noise=True,
        use_gathered_v_phi=True,
        use_layer_checkpoint=True,
        ln_before_distance=True,
        per_layer_v_phi_scale=True,
        fock_version=cfg.fock_version,
        n_registers=cfg.n_registers,
        register_salience_decay=cfg.register_salience_decay,
        register_salience_threshold=cfg.register_salience_threshold,
        creation_gate_hidden=cfg.creation_gate_hidden,
        stack_discipline=True,
        d_k=cfg.d_k,
        tau_create_init=cfg.tau_create_init,
        reverse_channel=cfg.reverse_channel,
        reverse_channel_stable=cfg.reverse_channel_stable,
        reverse_channel_pre_ln=cfg.reverse_channel_pre_ln,
        reverse_channel_soft_norm=cfg.reverse_channel_soft_norm,
        reverse_channel_warmup_steps=cfg.reverse_channel_warmup_steps,
        reverse_channel_per_layer=cfg.reverse_channel_per_layer,
        per_register_tau=True,
        per_register_keys=True,
        ortho_register_init=True,
        register_repulsion=cfg.register_repulsion,
        register_repulsion_coeff=cfg.register_repulsion_coeff,
        register_repulsion_kind=cfg.register_repulsion_kind,
    )

    model = FockMultiXiPARFLM(model_cfg).to(device)

    # Swap V_theta with depth-conditioned Gaussian
    if cfg.v_theta_variant == "gaussian":
        from model_gaussian_vtheta import (
            DepthConditionedMultiContextGaussianVTheta,
            install_depth_routing,
        )
        init_log_prec = -math.log(cfg.d)
        prec_max = 2.0 / cfg.d
        model.V_theta = DepthConditionedMultiContextGaussianVTheta(
            d=cfg.d,
            K=cfg.v_theta_wells_per_head,
            n_ctx=cfg.v_theta_n_heads,
            n_layers=cfg.L,
            w_scale=cfg.w_scale,
            init_log_precision=init_log_prec,
            precision_max=prec_max,
            code_init_std=cfg.v_theta_depth_code_init_std,
        ).to(device)
        install_depth_routing(model)

    if cfg.use_output_bias:
        return model, model_cfg, True  # needs logfreq init
    return model, model_cfg, False


def build_gpt2_model(cfg: TrainConfig, device: str):
    from matched_baseline_model import MatchedGPT, MatchedConfig

    model_cfg = MatchedConfig(
        vocab_size=cfg.vocab_size,
        d=cfg.d,
        max_len=cfg.max_len,
        n_layer=cfg.L,
        n_head=cfg.gpt2_n_head,
        mlp_mult=cfg.gpt2_mlp_mult,
        tie_embeddings=cfg.tie_embeddings,
    )
    model = MatchedGPT(model_cfg).to(device)
    return model, model_cfg


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(cfg: TrainConfig):
    """Load or stream OpenWebText tokens. Returns (train_ids, val_ids)."""
    from data_module import get_batch  # noqa: F401

    data_dir = Path(cfg.data_dir)
    train_cache = data_dir / f"openwebtext_train_{cfg.max_train_tokens // 1_000_000}M.npy"
    val_cache = data_dir / f"openwebtext_val_{cfg.val_tokens // 1_000_000}M.npy"

    if train_cache.exists() and val_cache.exists():
        if is_main():
            print("Loading cached OpenWebText tokens ...")
        train_ids = np.load(str(train_cache))
        val_ids = np.load(str(val_cache))
    else:
        if is_main():
            print(f"Streaming OpenWebText ({cfg.max_train_tokens:,} train tokens) ...")
        from datasets import load_dataset
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
        ds = load_dataset("Skylion007/openwebtext", split="train",
                          streaming=True, trust_remote_code=True)
        all_ids: list = []
        target = cfg.max_train_tokens + cfg.val_tokens
        chunk_texts: list = []
        n_docs = 0
        t0 = time.time()
        for example in ds:
            chunk_texts.append(example["text"])
            n_docs += 1
            if len(chunk_texts) >= 50_000:
                all_ids.extend(tok.encode("\n\n".join(chunk_texts)))
                if is_main():
                    print(f"  {n_docs:,} docs  {len(all_ids):,} tokens  "
                          f"({time.time() - t0:.0f}s)", flush=True)
                chunk_texts = []
                if len(all_ids) >= target:
                    break
        if chunk_texts:
            all_ids.extend(tok.encode("\n\n".join(chunk_texts)))
        arr = np.array(all_ids, dtype=np.uint16)
        val_ids = arr[-cfg.val_tokens:]
        train_ids = arr[:-cfg.val_tokens]
        if len(train_ids) > cfg.max_train_tokens:
            train_ids = train_ids[:cfg.max_train_tokens]
        if is_main():
            np.save(str(train_cache), train_ids)
            np.save(str(val_cache), val_ids)
            print(f"  Cached: train={len(train_ids):,}  val={len(val_ids):,}")
        del arr, all_ids

    if is_main():
        print(f"train: {len(train_ids):,}   val: {len(val_ids):,}")
    return train_ids, val_ids


def get_batch(ids: np.ndarray, batch_size: int, block_size: int,
              rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    n = len(ids) - block_size - 1
    starts = rng.integers(0, n, size=batch_size)
    x = np.stack([ids[s:s + block_size] for s in starts])
    y = np.stack([ids[s + 1:s + 1 + block_size] for s in starts])
    return x.astype(np.int64), y.astype(np.int64)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def lr_schedule(step: int, cfg: TrainConfig) -> float:
    if cfg.lr_schedule == "wsd":
        warmup_end = int(cfg.wsd_warmup_frac * cfg.total_steps)
        stable_end = int((cfg.wsd_warmup_frac + cfg.wsd_stable_frac) * cfg.total_steps)
        if step < warmup_end:
            return cfg.lr * (step + 1) / max(warmup_end, 1)
        elif step < stable_end:
            return cfg.lr
        else:
            decay_steps = cfg.total_steps - stable_end
            progress = (step - stable_end) / max(decay_steps, 1)
            cos_decay = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            return cfg.wsd_lr_floor + (cfg.lr - cfg.wsd_lr_floor) * cos_decay
    else:
        if step < cfg.warmup_steps:
            return cfg.lr * (step + 1) / cfg.warmup_steps
        progress = (step - cfg.warmup_steps) / max(cfg.total_steps - cfg.warmup_steps, 1)
        return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


# ---------------------------------------------------------------------------
# Per-group gradient clipping
# ---------------------------------------------------------------------------

GRAD_CLIP_OVERRIDES = {
    "=P": 0.3,
    "=E": 0.3,
    "V_phi": 0.3,
    "creation_gate": 0.3,
    "destruction_gate": 0.3,
    "reverse_channel_scale": 0.1,
    "reverse_ch": 0.1,
    "register": 0.3,
    "depth_code": 0.5,
}
WATCHDOG_EXCLUDE_GROUPS = {"override:reverse_channel_scale", "override:reverse_ch"}


def _assign_clip_group(pname: str, default_clip: float):
    low = pname.lower()
    prefix = low.split(".", 1)[0]
    for sub, thr in GRAD_CLIP_OVERRIDES.items():
        if sub.startswith("="):
            if prefix == sub[1:].lower():
                return f"override:{sub[1:]}", thr
        else:
            if sub.lower() in low:
                return f"override:{sub}", thr
    return prefix, default_clip


def clip_grads_per_group(model: nn.Module, default_clip: float):
    groups: Dict[str, list] = {}
    thr: Dict[str, float] = {}
    dev = None
    for n, p in model.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        if dev is None:
            dev = p.grad.device
        key, mx = _assign_clip_group(n, default_clip)
        groups.setdefault(key, []).append(p)
        thr[key] = mx
    total_sq = torch.zeros((), device=dev) if dev else torch.zeros(())
    per_group: Dict[str, float] = {}
    for key, ps in groups.items():
        gn = nn.utils.clip_grad_norm_(ps, thr[key])
        per_group[key] = float(gn)
        if key not in WATCHDOG_EXCLUDE_GROUPS:
            total_sq = total_sq + gn.detach() ** 2
    return total_sq.sqrt(), per_group


def per_group_grad_norms(model: nn.Module, default_clip: float):
    groups: Dict[str, list] = {}
    for n, p in model.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        key, _ = _assign_clip_group(n, default_clip)
        groups.setdefault(key, []).append(p)
    out: Dict[str, float] = {}
    for key, ps in groups.items():
        sq = sum(float(p.grad.detach().norm()) ** 2 for p in ps)
        out[key] = sq ** 0.5
    return out


# ---------------------------------------------------------------------------
# Forward helpers
# ---------------------------------------------------------------------------

def forward_fock_with_vreg(model, model_cfg, x, targets, cfg: TrainConfig):
    """FockPARFLM forward: NTP loss + V_theta regulariser."""
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                        enabled=cfg.bf16):
        h0 = model._embed(x)
        h_L, _ = model._stack_forward(h0, x, return_trajectory=False)
        logits = model.compute_logits(h_L)
    loss_ntp = F.cross_entropy(
        logits.float().reshape(-1, cfg.vocab_size), targets.reshape(-1))
    v_reg = torch.tensor(0.0, device=x.device)
    if cfg.lambda_v > 0:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=cfg.bf16):
            xis = model.xi_module(h_L.detach())
            V_vals = model.V_theta(xis, h_L)
        v_reg = (V_vals.float() ** 2).mean()
        loss = loss_ntp + cfg.lambda_v * v_reg
    else:
        loss = loss_ntp
    return loss, loss_ntp, v_reg


def forward_gpt2(model, x, targets, cfg: TrainConfig):
    """GPT-2 forward: simple NTP loss."""
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                        enabled=cfg.bf16):
        logits, _ = model(x)
    loss_ntp = F.cross_entropy(
        logits.float().reshape(-1, cfg.vocab_size), targets.reshape(-1))
    v_reg = torch.tensor(0.0, device=x.device)
    return loss_ntp, loss_ntp, v_reg


# ---------------------------------------------------------------------------
# Auto batch-size probing (OOM-aware) — mirrors the Colab notebook's
# probe, which was never carried over into this standalone script.
# Without this, a fixed preset batch_size can silently OOM on hardware
# with less headroom than whatever the preset was tuned against.
# ---------------------------------------------------------------------------

def probe_batch_size(model, model_cfg, forward_fn, cfg: TrainConfig,
                     train_ids: np.ndarray, device: str) -> int:
    """Try decreasing batch sizes with a real forward+backward microstep
    until one fits in GPU memory. Returns the largest that worked (or
    cfg.batch_size unchanged on CPU / if the first probe already fits).
    """
    if device == "cpu":
        return cfg.batch_size

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    ceiling = cfg.batch_size
    candidates = sorted({b for b in
                        [ceiling, 16, 12, 8, 6, 4, 2, 1] if b <= ceiling},
                        reverse=True)

    raw_model = model.module if hasattr(model, "module") else model
    rng = np.random.default_rng(cfg.seed)

    for bs in candidates:
        try:
            xb, yb = get_batch(train_ids, bs, cfg.block_size, rng)
            x = torch.from_numpy(xb).to(device)
            y = torch.from_numpy(yb).to(device)
            if cfg.model_type == "fock":
                loss, _, _ = forward_fn(raw_model, model_cfg, x, y, cfg)
            else:
                loss, _, _ = forward_fn(raw_model, x, y, cfg)
            loss.backward()
            raw_model.zero_grad(set_to_none=True)
            del x, y, xb, yb, loss
            torch.cuda.empty_cache()
            if is_main() and bs != ceiling:
                print(f"  Auto batch-size probe: {ceiling} -> {bs} "
                      f"(OOM at higher sizes on this GPU, {vram_gb:.0f} GB)")
            return bs
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raw_model.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if is_main():
                    print(f"  OOM probe at batch={bs} — trying smaller ...")
                continue
            raise
    raise RuntimeError(
        f"All batch-size candidates {candidates} OOMed on this GPU "
        f"({vram_gb:.0f} GB) for d={cfg.d} L={cfg.L} block={cfg.block_size}.")


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(model, optim, model_cfg, cfg: TrainConfig,
                    step: int, val_loss: float, tag: str = ""):
    if not is_main():
        return
    raw_model = model.module if hasattr(model, "module") else model
    ckpt = {
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "model_cfg": (asdict(model_cfg) if hasattr(model_cfg, "__dataclass_fields__")
                      else vars(model_cfg)),
        "train_cfg": {
            "model_type": cfg.model_type,
            "batch_size": cfg.batch_size,
            "block_size": cfg.block_size,
            "grad_accum": cfg.grad_accum,
            "effective_batch": cfg.effective_batch,
            "total_steps": cfg.total_steps,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "grad_clip": cfg.grad_clip,
            "lambda_v": cfg.lambda_v,
            "lr_schedule": cfg.lr_schedule,
        },
        "step": step,
        "val_loss": val_loss,
        "val_ppl": math.exp(val_loss),
    }
    if cfg.model_type == "fock":
        ckpt["gamma"] = raw_model.gamma.item()
        ckpt["xi_alphas"] = raw_model.xi_alpha_values()
        ckpt["variant"] = (f"fock_parf_multixi_v2.1_{cfg.v_theta_variant}_"
                           f"dcvt{cfg.v_theta_n_heads}")

    ckpt_dir = Path(cfg.output_dir) / "checkpoints"
    fname = f"ckpt_step{step}{tag}.pt"
    path = ckpt_dir / fname
    torch.save(ckpt, path)
    print(f"  Checkpoint saved: {path}  (PPL={math.exp(val_loss):.2f})")

    if "_best" in tag:
        canonical = ckpt_dir / "ckpt_best.pt"
        shutil.copy2(path, canonical)


# ---------------------------------------------------------------------------
# Remote sync (Google Drive via rclone, or rsync)
# ---------------------------------------------------------------------------

def sync_to_remote(cfg: TrainConfig):
    """Sync output_dir to remote destination (non-blocking)."""
    if not cfg.sync_remote or not is_main():
        return
    import subprocess as _sp
    src = cfg.output_dir.rstrip("/") + "/"
    dst = cfg.sync_remote.rstrip("/") + "/"
    if dst.startswith("gdrive:") or dst.startswith("remote:"):
        cmd = ["rclone", "copy", src, dst, "--transfers=4", "-q"]
    else:
        cmd = ["rsync", "-az", "--partial", src, dst]
    try:
        _sp.Popen(cmd, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
        print(f"  [sync] {src} -> {dst}")
    except FileNotFoundError:
        print(f"  [sync] WARNING: {'rclone' if 'drive:' in dst else 'rsync'} "
              f"not found. Install it to enable remote sync.")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, model_cfg, val_ids, cfg: TrainConfig, device: str):
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.eval()
    rng = np.random.default_rng(42)
    losses = []
    for _ in range(cfg.eval_iters):
        xb, yb = get_batch(val_ids, cfg.batch_size, cfg.block_size, rng)
        x = torch.from_numpy(xb).to(device)
        y = torch.from_numpy(yb).to(device)
        if cfg.model_type == "fock":
            with torch.enable_grad():
                _, loss_ntp, _ = forward_fock_with_vreg(
                    raw_model, model_cfg, x, y, cfg)
            losses.append(loss_ntp.item())
        else:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=cfg.bf16):
                logits, _ = raw_model(x)
            loss = F.cross_entropy(
                logits.float().reshape(-1, cfg.vocab_size), y.reshape(-1))
            losses.append(loss.item())
    raw_model.train()
    val_loss = float(np.mean(losses))
    if is_ddp():
        t = torch.tensor(val_loss, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        val_loss = t.item()
    return val_loss


# ---------------------------------------------------------------------------
# Watchdog: reload best checkpoint on sustained gradient divergence
# ---------------------------------------------------------------------------

def reload_best(model, optim_obj, cfg: TrainConfig, device: str):
    best_path = Path(cfg.output_dir) / "checkpoints" / "ckpt_best.pt"
    if not best_path.exists():
        return
    raw_model = model.module if hasattr(model, "module") else model
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    raw_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    try:
        optim_obj.load_state_dict(ckpt["optimizer_state_dict"])
    except Exception:
        pass
    s = ckpt.get("step", 0)
    p = ckpt.get("val_ppl", float("inf"))
    print(f"[watchdog] Reloaded best: step {s:,} PPL {p:.2f}")
    del ckpt


# ---------------------------------------------------------------------------
# Logfreq computation
# ---------------------------------------------------------------------------

def ensure_logfreq(train_ids: np.ndarray, cfg: TrainConfig, ca_dir: Path) -> str:
    """Return path to logfreq surprisal .npy file, creating if needed."""
    candidates = [
        ca_dir / "scaleup" / "results" / "logfreq_surprisal_openwebtext.npy",
        Path(cfg.output_dir) / "logfreq_surprisal_openwebtext.npy",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    out = Path(cfg.output_dir) / "logfreq_surprisal_openwebtext.npy"
    if is_main():
        counts = np.bincount(train_ids.astype(np.int64),
                             minlength=cfg.vocab_size).astype(np.float64)
        p = (counts + 1.0) / (counts.sum() + cfg.vocab_size)
        surprisal = (-np.log(p)).astype(np.float32)
        np.save(str(out), surprisal)
        print(f"Logfreq saved: {out}")
    if is_ddp():
        dist.barrier()
    return str(out)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig):
    cfg.resolve_xi_override()
    cfg.resolve_wsd_lr_floor()

    script_dir = Path(__file__).resolve().parent
    ca_dir = _resolve_paths(cfg, script_dir)
    device = "cpu"
    if torch.cuda.is_available():
        device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    torch.manual_seed(cfg.seed + rank())
    np.random.seed(cfg.seed + rank())
    rng = np.random.default_rng(cfg.seed + rank())

    if is_main():
        props = torch.cuda.get_device_properties(0) if device != "cpu" else None
        if props:
            print(f"GPU: {props.name}  ({props.total_memory / 1e9:.1f} GB)")
        print(f"\nConfig: model={cfg.model_type}  d={cfg.d}  L={cfg.L}"
              + (f"  [bf16]" if cfg.bf16 else ""))
        print(f"  output_dir: {cfg.output_dir}")
        print(f"  lr={cfg.lr}  schedule={cfg.lr_schedule}  "
              f"batch={cfg.batch_size}x{cfg.grad_accum}x{world_size()} "
              f"(eff={cfg.effective_batch})")

    # ── Data ──
    train_ids, val_ids = load_data(cfg)

    # ── Model ──
    if cfg.model_type == "fock":
        logfreq_path = ensure_logfreq(train_ids, cfg, ca_dir)
        model, model_cfg, needs_bias_init = build_fock_model(cfg, device, logfreq_path)
        if needs_bias_init:
            counts = np.bincount(train_ids.astype(np.int64),
                                 minlength=cfg.vocab_size)
            model.init_output_bias_from_logfreq(counts)
            if is_main():
                print(f"Output bias initialised from log-unigram-freq")
        forward_fn = forward_fock_with_vreg
    else:
        model, model_cfg = build_gpt2_model(cfg, device)
        forward_fn = forward_gpt2

    n_params = (model.num_params() if hasattr(model, "num_params")
                else sum(p.numel() for p in model.parameters()))
    if is_main():
        print(f"\nModel: {cfg.model_type}  params={n_params:,}")
        if cfg.model_type == "fock":
            n_vt = sum(p.numel() for p in model.V_theta.parameters())
            print(f"  V_theta: {n_vt:,}")

    # ── Auto batch-size probe (OOM-aware) ──
    # Preserves the requested effective batch by scaling grad_accum up
    # if the probe has to shrink batch_size to fit on this GPU.
    orig_bs, orig_accum = cfg.batch_size, cfg.grad_accum
    safe_bs = probe_batch_size(model, model_cfg, forward_fn, cfg,
                               train_ids, device)
    if safe_bs != orig_bs:
        cfg.grad_accum = max(1, round(orig_accum * orig_bs / safe_bs))
        cfg.batch_size = safe_bs
        if is_main():
            print(f"  Adjusted: batch={orig_bs}x{orig_accum} -> "
                  f"{cfg.batch_size}x{cfg.grad_accum} "
                  f"(eff {orig_bs*orig_accum} -> {cfg.effective_batch})")

    # ── DDP wrap ──
    if is_ddp():
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[int(os.environ["LOCAL_RANK"])])

    # ── Optimizer ──
    raw_model = model.module if hasattr(model, "module") else model
    trainable = [p for p in raw_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=cfg.betas,
    )

    # ── Resume ──
    resume_step = 0
    best_val_ppl = float("inf")

    if cfg.resume_from:
        ckpt_path = Path(cfg.resume_from)
        if ckpt_path.exists():
            if is_main():
                print(f"Resuming from: {ckpt_path}")
            ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
            raw_model.load_state_dict(ckpt_data["model_state_dict"], strict=False)
            try:
                optimizer.load_state_dict(ckpt_data["optimizer_state_dict"])
            except Exception as e:
                if is_main():
                    print(f"[warn] Could not load optimizer state: {e}")
            resume_step = ckpt_data.get("step", 0)
            best_val_ppl = ckpt_data.get("val_ppl", float("inf"))
            del ckpt_data
    else:
        best_ckpt = Path(cfg.output_dir) / "checkpoints" / "ckpt_best.pt"
        if best_ckpt.exists():
            bd = torch.load(best_ckpt, map_location="cpu", weights_only=False)
            best_val_ppl = bd.get("val_ppl", float("inf"))
            del bd

    # ── Logging ──
    log_path = Path(cfg.output_dir) / "training_log.jsonl"
    log_fh = log_path.open("a") if is_main() else None

    def log_write(record: dict):
        if log_fh is not None:
            log_fh.write(json.dumps(record) + "\n")
            log_fh.flush()

    # ── Training state ──
    t0 = time.time()
    raw_model.train()
    run_ntp, run_vreg, n_run = 0.0, 0.0, 0
    n_skipped = 0
    steps_this_session = 0
    grad_norm_ema = 0.0
    grad_norm_above_thresh = 0
    last_spike_step = -10**9
    last_pg_norms: Dict[str, float] = {}

    ckpt_steps = set(range(cfg.ckpt_interval,
                           cfg.total_steps + 1,
                           cfg.ckpt_interval))

    if is_main():
        print(f"\n{'='*60}")
        print(f"Training: {cfg.model_type}  d={cfg.d}  L={cfg.L}  "
              f"params={n_params:,}")
        if cfg.lr_schedule == "wsd":
            warmup_end = int(cfg.wsd_warmup_frac * cfg.total_steps)
            stable_end = int((cfg.wsd_warmup_frac + cfg.wsd_stable_frac)
                             * cfg.total_steps)
            print(f"  WSD: warmup 0->{warmup_end}  stable ->{stable_end}  "
                  f"decay ->{cfg.total_steps}")
            print(f"  LR: {cfg.lr} -> floor {cfg.wsd_lr_floor}")
        print(f"  grad_clip={cfg.grad_clip}  per_group_clip={cfg.per_group_clip}")
        if cfg.force_clamp_max > 0:
            print(f"  force_clamp_max={cfg.force_clamp_max}")
        print(f"  resume_step={resume_step}  best_ppl={best_val_ppl:.2f}")
        print(f"{'='*60}\n")

    # ── Main loop ──
    for step in range(resume_step, cfg.total_steps):
        lr_now = lr_schedule(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr_now

        optimizer.zero_grad(set_to_none=True)
        accum_ntp = 0.0
        accum_vreg = 0.0
        accum_rep = 0.0

        for _ in range(cfg.grad_accum):
            xb, yb = get_batch(train_ids, cfg.batch_size, cfg.block_size, rng)
            x = torch.from_numpy(xb).to(device)
            y = torch.from_numpy(yb).to(device)

            if cfg.model_type == "fock":
                loss, loss_ntp, v_reg = forward_fn(
                    raw_model, model_cfg, x, y, cfg)
                if cfg.register_repulsion:
                    rep = raw_model.pop_repulsion_loss()
                    loss = loss + rep
                    accum_rep += float(rep.detach()) / cfg.grad_accum
            else:
                loss, loss_ntp, v_reg = forward_fn(raw_model, x, y, cfg)

            (loss / cfg.grad_accum).backward()
            accum_ntp += loss_ntp.item() / cfg.grad_accum
            accum_vreg += float(v_reg.detach()) / cfg.grad_accum

        # ── Gradient clipping ──
        if cfg.per_group_clip and cfg.model_type == "fock":
            grad_norm, last_pg_norms = clip_grads_per_group(
                raw_model, cfg.grad_clip)
        else:
            last_pg_norms = {}
            grad_norm = nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)

        # ── Spike debugger ──
        if cfg.grad_spike_debug and is_main():
            tot_preclip = float(grad_norm)
            if (tot_preclip > cfg.grad_spike_threshold
                    and (step - last_spike_step) >= 0):
                last_spike_step = step
                if last_pg_norms:
                    top = sorted(last_pg_norms.items(),
                                 key=lambda kv: kv[1], reverse=True)[:8]
                    brk = "  ".join(f"{k}={v:.1f}" for k, v in top)
                else:
                    brk = ""
                print(f"\n[spike] step {step+1}: pre-clip total grad="
                      f"{tot_preclip:.1f}  ntp={accum_ntp:.3f}  "
                      f"v_reg={accum_vreg:.4f}")
                if brk:
                    print(f"[spike]   top groups: {brk}")

        if torch.isfinite(grad_norm) and math.isfinite(accum_ntp):
            optimizer.step()
            if (cfg.model_type == "fock" and cfg.v_theta_variant == "gaussian"
                    and cfg.v_theta_n_heads > 1):
                for bank in raw_model.V_theta.banks:
                    if hasattr(bank, "clamp_params"):
                        bank.clamp_params()
        else:
            n_skipped += 1
            optimizer.zero_grad(set_to_none=True)

        # ── Watchdog ──
        raw_gn = float(grad_norm)
        grad_norm_ema = ((1 - cfg.grad_norm_ema_alpha) * grad_norm_ema
                         + cfg.grad_norm_ema_alpha * raw_gn)
        if grad_norm_ema > cfg.grad_norm_ema_threshold:
            grad_norm_above_thresh += 1
        else:
            grad_norm_above_thresh = 0

        if grad_norm_above_thresh >= cfg.grad_norm_ema_patience:
            if is_main():
                print(f"\n[watchdog] EMA grad_norm={grad_norm_ema:.1f} > "
                      f"{cfg.grad_norm_ema_threshold} for "
                      f"{grad_norm_above_thresh} steps at step {step+1}.")
            reload_best(model, optimizer, cfg, device)
            grad_norm_ema = 0.0
            grad_norm_above_thresh = 0
            n_skipped += 1

        run_ntp += accum_ntp
        run_vreg += accum_vreg
        n_run += 1
        steps_this_session += 1

        # ── Logging ──
        if (step + 1) % cfg.log_interval == 0 and is_main():
            avg_ntp = run_ntp / n_run
            avg_vreg = run_vreg / n_run
            run_ntp, run_vreg, n_run = 0.0, 0.0, 0
            elapsed = time.time() - t0
            sec_per_step = elapsed / steps_this_session
            remaining = (cfg.total_steps - step - 1) * sec_per_step

            extra = ""
            if cfg.model_type == "fock":
                alphas = raw_model.xi_alpha_values()
                alpha_str = ",".join(f"{a:.3f}" for a in alphas)
                gamma_str = f"gamma={raw_model.gamma.item():.3f}"
                extra = f"  {gamma_str}  alpha=[{alpha_str}]"

            top_grp = ""
            if cfg.per_group_clip and last_pg_norms:
                k, v = max(last_pg_norms.items(), key=lambda kv: kv[1])
                top_grp = f"top[{k}]={v:.1f}  "

            rep_str = (f"rep={accum_rep:.4f}  "
                       if cfg.register_repulsion and cfg.model_type == "fock"
                       else "")

            print(
                f"step {step+1:7d}/{cfg.total_steps}  "
                f"ntp={avg_ntp:.4f}  v_reg={avg_vreg:.4f}  lr={lr_now:.2e}  "
                f"grad={float(grad_norm):.2f}  {rep_str}{top_grp}"
                f"{extra}  "
                f"{elapsed:.0f}s  (~{remaining/3600:.1f}h remaining)"
            )
            log_write({
                "step": step + 1, "train_loss": avg_ntp, "v_reg": avg_vreg,
                "lr": lr_now, "grad_norm": float(grad_norm),
                "reg_repulsion": accum_rep,
                "elapsed_sec": elapsed, "sec_per_step": sec_per_step,
            })

        # ── Evaluation ──
        if (step + 1) % cfg.eval_interval == 0:
            val_loss = evaluate(model, model_cfg, val_ids, cfg, device)
            val_ppl = math.exp(val_loss)
            is_best = val_ppl < best_val_ppl
            if is_best:
                best_val_ppl = val_ppl
            if is_main():
                marker = "*** NEW BEST ***" if is_best else ""
                elapsed = time.time() - t0
                print(f">>> EVAL step {step+1:,}  val_loss={val_loss:.4f}  "
                      f"val_ppl={val_ppl:.2f}  best={best_val_ppl:.2f}  "
                      f"{marker}  ({elapsed:.0f}s)")
                log_write({
                    "step": step + 1, "val_loss": val_loss,
                    "val_ppl": val_ppl, "best_ppl": best_val_ppl,
                })
                if is_best:
                    save_checkpoint(model, optimizer, model_cfg, cfg,
                                    step + 1, val_loss, tag="_best")
                    sync_to_remote(cfg)

        # ── Periodic checkpoint ──
        if (step + 1) in ckpt_steps:
            if (step + 1) % cfg.eval_interval != 0:
                val_loss = evaluate(model, model_cfg, val_ids, cfg, device)
            save_checkpoint(model, optimizer, model_cfg, cfg,
                            step + 1, val_loss)
            sync_to_remote(cfg)

    # ── Final sync ──
    sync_to_remote(cfg)

    # ── Cleanup ──
    if log_fh is not None:
        log_fh.close()
    if is_main():
        print(f"\nTraining complete. Best PPL: {best_val_ppl:.2f}")
    cleanup_ddp()


# ---------------------------------------------------------------------------
# Gamma sweep: short training runs to find optimal gamma for a given d
# ---------------------------------------------------------------------------

GAMMA_CANDIDATES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

def gamma_sweep(base_cfg: TrainConfig, gammas: List[float],
                sweep_steps: int = 3000, sweep_eval_interval: int = 500):
    """Run short training for each gamma candidate, return ranked results.

    Safe to call under multi-GPU DDP (torchrun): each candidate's
    train(cfg) call already handles DDP internally, but the sweep-level
    bookkeeping below (console banners, results list, summary file) is
    not per-rank-partitioned, so it is gated on is_main() to avoid
    duplicated output / redundant (harmless but wasteful) file writes
    from non-main ranks.
    """
    import copy
    import json as _json

    if is_main():
        print(f"\n{'='*60}")
        print(f"  GAMMA SWEEP  d={base_cfg.d}  L={base_cfg.L}")
        print(f"  Candidates: {gammas}")
        print(f"  Steps per candidate: {sweep_steps}")
        print(f"{'='*60}\n")

    results = []
    sweep_dir = Path(base_cfg.output_dir) / "gamma_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    for gi, g in enumerate(gammas):
        cfg = copy.deepcopy(base_cfg)
        cfg.fixed_gamma = g
        cfg.total_steps = sweep_steps
        cfg.eval_interval = sweep_eval_interval
        cfg.ckpt_interval = sweep_steps + 1  # no periodic checkpoints
        cfg.log_interval = 100
        cfg.grad_spike_debug = False
        cfg.sync_remote = ""
        cfg.output_dir = str(sweep_dir / f"gamma_{g:.3f}")
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(cfg.output_dir) / "checkpoints").mkdir(parents=True, exist_ok=True)

        if is_main():
            print(f"\n--- Sweep {gi+1}/{len(gammas)}: gamma={g:.3f} ---")
        err_msg = None
        try:
            train(cfg)
        except Exception as e:
            err_msg = str(e)
            if is_main():
                print(f"  FAILED: {err_msg}")
        finally:
            # Ensure the previous candidate's model/optimizer/activations
            # are fully released before the next candidate builds a new
            # model — otherwise GPU memory (and any OOM) carries over
            # and every subsequent candidate fails identically.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if err_msg is not None:
            results.append({"gamma": g, "best_ppl": float("inf"),
                            "final_ppl": float("inf"), "error": err_msg})
            continue

        log_path = Path(cfg.output_dir) / "training_log.jsonl"
        best_ppl = float("inf")
        final_ppl = float("inf")
        if log_path.exists():
            for line in open(log_path):
                entry = _json.loads(line)
                if "val_ppl" in entry:
                    final_ppl = entry["val_ppl"]
                    if entry["val_ppl"] < best_ppl:
                        best_ppl = entry["val_ppl"]

        results.append({"gamma": g, "best_ppl": best_ppl,
                        "final_ppl": final_ppl})
        if is_main():
            print(f"  gamma={g:.3f}  best_ppl={best_ppl:.2f}  "
                  f"final_ppl={final_ppl:.2f}")

    results.sort(key=lambda r: r["best_ppl"])

    if is_main():
        print(f"\n{'='*60}")
        print(f"  GAMMA SWEEP RESULTS  (d={base_cfg.d}, L={base_cfg.L})")
        print(f"{'='*60}")
        print(f"  {'gamma':>8s}  {'best_ppl':>10s}  {'final_ppl':>10s}")
        print(f"  {'-----':>8s}  {'--------':>10s}  {'---------':>10s}")
        for r in results:
            marker = " <-- BEST" if r == results[0] else ""
            err = f"  ERROR: {r['error']}" if "error" in r else ""
            print(f"  {r['gamma']:8.3f}  {r['best_ppl']:10.2f}  "
                  f"{r['final_ppl']:10.2f}{marker}{err}")

        summary_path = sweep_dir / "sweep_summary.json"
        with open(summary_path, "w") as f:
            _json.dump({"d": base_cfg.d, "L": base_cfg.L,
                        "sweep_steps": sweep_steps, "results": results},
                       f, indent=2)
        print(f"\n  Summary saved to: {summary_path}")

        if results and results[0]["best_ppl"] < float("inf"):
            best_g = results[0]["gamma"]
            print(f"\n  >>> Recommended gamma for d={base_cfg.d}: {best_g:.3f}")
            print(f"  >>> Run full training with:  --preset d{base_cfg.d} "
                  f"--fixed_gamma {best_g}")
        print()

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train FockPARFLM v2.1 or matched GPT-2 baseline")

    parser.add_argument("--preset", type=str, default="",
                        choices=["", *PRESETS.keys()],
                        help="Load a preset config (d384, d768, d1024, "
                             "gpt2-small, gpt2-medium)")

    # Gamma sweep mode
    parser.add_argument("--gamma_sweep", action="store_true",
                        help="Run gamma sweep instead of full training")
    parser.add_argument("--sweep_gammas", type=str, default="",
                        help="Comma-separated gamma values to sweep "
                             "(default: 0.05,0.10,...,0.50)")
    parser.add_argument("--sweep_steps", type=int, default=3000,
                        help="Steps per gamma candidate (default: 3000)")

    # Allow overriding any TrainConfig field.
    # NOTE: this module uses `from __future__ import annotations`, which
    # makes dataclass field.type a *string* (e.g. "str") rather than the
    # actual type object (PEP 563). Resolve real types via get_type_hints
    # instead of comparing f.type directly, or every field comparison
    # below would silently fail and no CLI overrides would be registered.
    resolved_types = typing.get_type_hints(TrainConfig)
    for f in TrainConfig.__dataclass_fields__.values():
        name = f"--{f.name}"
        ftype = resolved_types.get(f.name, f.type)
        if ftype == bool:
            parser.add_argument(name, type=lambda x: x.lower() == "true",
                                default=None)
        elif ftype == int:
            parser.add_argument(name, type=int, default=None)
        elif ftype == float:
            parser.add_argument(name, type=float, default=None)
        elif ftype == str:
            parser.add_argument(name, type=str, default=None)

    args = parser.parse_args()

    cfg = TrainConfig()
    if args.preset and args.preset in PRESETS:
        for k, v in PRESETS[args.preset].items():
            setattr(cfg, k, v)

    # Apply CLI overrides
    for f in TrainConfig.__dataclass_fields__:
        val = getattr(args, f, None)
        if val is not None:
            setattr(cfg, f, val)

    return cfg, args


if __name__ == "__main__":
    setup_ddp()
    cfg, args = parse_args()

    if args.gamma_sweep:
        gammas = ([float(x) for x in args.sweep_gammas.split(",")]
                  if args.sweep_gammas else GAMMA_CANDIDATES)
        gamma_sweep(cfg, gammas, sweep_steps=args.sweep_steps)
    else:
        train(cfg)
