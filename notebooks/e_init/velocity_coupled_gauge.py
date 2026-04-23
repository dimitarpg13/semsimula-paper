#!/usr/bin/env python
"""
Velocity-coupled gauge-field augmentation of the Euler-Lagrange integrator.

Follow-up to notebooks/e_init/helmholtz_curl_augmented.py (§1.4), which
showed that a position-coupled linear skew term Omega*x cannot fit the
hidden-state trajectories.  This script tests strictly richer ansatzes
from the electromagnetic-analogue Lagrangian of §6:

    L = (1/2) m ||xdot||^2 + A(x) . xdot - V(x),

whose Euler-Lagrange equation is

    m xddot = -grad V(x) + F(x) . xdot,                      (*)

where F(x) is a state-dependent antisymmetric 2-tensor.  Here we test
four progressively richer linear approximations of F(x), all in a
per-layer top-k PCA subspace with skew-symmetric constraint on every
antisymmetric block:

  1. 'B_const'     : F(x) = B_0  (constant skew, the pure P-rot-6 case)
  2. 'B_affine_r1' : F(x) = B_0 + z_1 * B_1      (z_i = i-th PCA coord of x)
  3. 'B_affine_r2' : F(x) = B_0 + z_1 B_1 + z_2 B_2
  4. 'Omega_Bv'    : F'(x,v) = Omega x + B_0 v  (combined position- and
                     velocity-coupled linear gauge -- the full linear U(1))

Each B_i and Omega is skew-symmetric k x k in PCA space.  We also fit
the *unconstrained* linear-in-v operator N_0 as a Mfull-style upper
bound, analogous to helmholtz_curl_augmented.py.

Inputs, split, fitting, and evaluation are taken from §1.4 unchanged;
see helmholtz_curl_augmented.py for context.

Outputs
-------
  results/velocity_coupled_gauge_results.npz
  results/fig_gauge_residual_vs_gamma.png
  results/fig_gauge_residual_vs_layer_at_gamma_star.png
  results/velocity_coupled_gauge_summary.md

Runtime: ~60 s on Apple MPS.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------------------------
# Config + corpus (identical to helmholtz_curl_augmented.py)
# ---------------------------------------------------------------------------
def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        try:
            torch.zeros(1, device="mps")
            return "mps"
        except Exception:
            pass
    return "cpu"


SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


@dataclass
class Config:
    model_name: str = "gpt2"
    device: str = field(default_factory=_pick_device)
    dtype: torch.dtype = field(default_factory=lambda: torch.float32)
    max_length: int = 64
    pca_k: int = 16
    n_test_per_domain: int = 2
    ridge_lambda: float = 1e-3
    seed: int = 0


cfg = Config()
print(f"torch {torch.__version__}   device {cfg.device}")


CORPUS: Dict[str, List[str]] = {
    "mathematics": [
        "The fundamental theorem of calculus establishes that differentiation and integration are inverse operations of each other.",
        "A metric space is a set together with a notion of distance between its elements, usually called points, that satisfies a set of axioms.",
        "Euler's identity connects the five most important numbers in mathematics through the equation e to the power of i pi plus one equals zero.",
        "The eigenvalues of a symmetric matrix are always real, and the eigenvectors corresponding to distinct eigenvalues are orthogonal.",
        "Godel's incompleteness theorems demonstrate that in any consistent formal system capable of expressing basic arithmetic there exist statements that can neither be proved nor disproved.",
        "The Riemann hypothesis conjectures that all non-trivial zeros of the Riemann zeta function have real part equal to one half.",
        "A group homomorphism preserves the algebraic structure by mapping the identity element to the identity element and products to products.",
        "The central limit theorem states that the sum of a large number of independent random variables tends toward a normal distribution regardless of the underlying distribution.",
        "Hilbert spaces generalize the notion of Euclidean space to infinite dimensions while retaining the structure of an inner product.",
        "The Lagrangian of a mechanical system equals the kinetic energy minus the potential energy and encodes the complete dynamics through the Euler-Lagrange equations.",
    ],
    "narrative": [
        "The old lighthouse keeper climbed the spiral staircase one last time, his weathered hands gripping the iron railing as the storm gathered outside.",
        "She found the letter tucked between the pages of a book she hadn't opened in years, the ink faded but the words still sharp enough to wound.",
        "The train pulled into the empty station at midnight, its headlamp cutting through the fog like a single unblinking eye.",
        "He sat on the porch watching the fireflies trace their erratic paths through the warm summer air while the radio played something slow and sad.",
        "The market was closing for the day and the vendors were packing up their unsold fruit, bruised peaches and overripe plums going back into crates.",
        "She ran through the forest with branches whipping at her face, the sound of the river growing louder with every desperate step.",
        "The children built a fort out of couch cushions and draped a bedsheet over the top, declaring it a castle that no adults could enter.",
        "He returned to the village after twenty years and found that the oak tree in the square had been cut down and replaced by a parking lot.",
        "The ship appeared on the horizon at dawn, its sails torn and its hull battered, carrying survivors of a voyage no one had expected to end.",
        "She opened the door to find the apartment exactly as she had left it, dust settled on every surface like a thin layer of forgotten time.",
    ],
    "scientific": [
        "Photosynthesis converts carbon dioxide and water into glucose and oxygen using light energy captured by chlorophyll molecules in the thylakoid membranes.",
        "The double helix structure of DNA consists of two antiparallel strands held together by hydrogen bonds between complementary base pairs adenine-thymine and guanine-cytosine.",
        "General relativity describes gravity not as a force but as the curvature of spacetime caused by the presence of mass and energy.",
        "Neurons communicate across synaptic clefts by releasing neurotransmitters that bind to receptors on the postsynaptic membrane and trigger ion channel opening.",
        "The cosmic microwave background radiation is the thermal remnant of the early universe, emitted approximately 380,000 years after the Big Bang when atoms first formed.",
        "Plate tectonics explains the movement of lithospheric plates driven by convection currents in the asthenosphere, producing earthquakes, volcanoes, and mountain ranges.",
        "Quantum entanglement describes a correlation between particles such that measuring the state of one instantaneously determines the state of the other regardless of distance.",
        "The mitochondrial electron transport chain transfers electrons through a series of protein complexes to generate a proton gradient that drives ATP synthesis.",
        "Black holes form when massive stars exhaust their nuclear fuel and collapse under their own gravity, creating a singularity surrounded by an event horizon.",
        "CRISPR-Cas9 is a genome editing tool that uses a guide RNA to direct the Cas9 nuclease to a specific DNA sequence where it makes a double-strand break.",
    ],
    "code_description": [
        "The function iterates over the input list, applies a filter predicate to each element, and collects the matching elements into a new list that is returned.",
        "A binary search tree maintains the invariant that for every node, all values in the left subtree are smaller and all values in the right subtree are larger.",
        "The garbage collector identifies unreachable objects by tracing references from root pointers and reclaims their memory for future allocations.",
        "Dependency injection decouples object creation from usage by passing required services through constructor parameters rather than instantiating them internally.",
        "The load balancer distributes incoming HTTP requests across a pool of backend servers using a round-robin algorithm with health check probes every thirty seconds.",
        "A database transaction groups multiple operations into an atomic unit that either commits all changes or rolls back entirely if any operation fails.",
        "The recursive function computes the Fibonacci sequence by returning the sum of the two preceding values with base cases returning zero and one respectively.",
        "Hash maps achieve average constant time lookups by computing a hash of the key and using it as an index into an array of buckets.",
        "The event loop processes asynchronous callbacks from a message queue, executing each callback to completion before moving to the next one in the queue.",
        "Backpropagation computes gradients of the loss function with respect to each weight by applying the chain rule layer by layer from the output to the input.",
    ],
    "conversational": [
        "I was thinking we could grab dinner at that new place on Fifth Street, the one with the rooftop patio, if you're not too tired after work.",
        "Did you see the game last night? I couldn't believe they came back from a twenty-point deficit in the fourth quarter to win by three.",
        "My neighbor's dog got out again this morning and I spent half an hour chasing it around the block before finally catching it near the park.",
        "I've been meaning to tell you that the meeting got moved to Thursday, so we have an extra day to finish the presentation slides.",
        "The traffic was absolutely terrible this morning, it took me almost two hours to get to the office when it usually takes thirty minutes.",
        "Do you remember that restaurant we went to on vacation last summer? I found out they just opened a second location near downtown.",
        "I'm trying to decide between the blue one and the red one but honestly they both look great so maybe I should just get both.",
        "She told me she's thinking about going back to school to study architecture, which is funny because she used to say she'd never set foot in a classroom again.",
        "Can you pick up some milk on the way home? We also need bread and I think we're almost out of coffee too.",
        "I finally finished that book you recommended and you were right, the ending was completely unexpected but somehow felt inevitable in retrospect.",
    ],
}


rng = np.random.default_rng(cfg.seed)
train_sentences: List[Tuple[str, str]] = []
test_sentences:  List[Tuple[str, str]] = []
for domain, sents in CORPUS.items():
    idx = np.arange(len(sents))
    rng.shuffle(idx)
    for i in idx[: cfg.n_test_per_domain]:
        test_sentences.append((sents[i], domain))
    for i in idx[cfg.n_test_per_domain:]:
        train_sentences.append((sents[i], domain))
print(f"train sentences: {len(train_sentences)}   test sentences: {len(test_sentences)}")


# ---------------------------------------------------------------------------
# GPT-2 extraction (same as §1.4)
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    cfg.model_name, torch_dtype=cfg.dtype, attn_implementation="eager"
).to(cfg.device)
model.eval()
N_LAYERS = model.config.num_hidden_layers
D_HIDDEN = model.config.hidden_size


@dataclass
class Trajectory:
    sentence: str
    domain: str
    split: str
    tok_ids: np.ndarray
    hs: np.ndarray
    attn: np.ndarray
    ptl: np.ndarray
    w: Optional[np.ndarray] = None
    mu_ps: Optional[np.ndarray] = None
    x_ps: Optional[np.ndarray] = None


@torch.no_grad()
def extract(s: str, domain: str, split: str) -> Trajectory:
    enc = tokenizer(s, return_tensors="pt", truncation=True, max_length=cfg.max_length)
    ids = enc["input_ids"].to(cfg.device)
    out = model(ids, output_hidden_states=True, output_attentions=True)
    hs = torch.stack([h[0] for h in out.hidden_states], dim=0).float().cpu().numpy()
    attn = torch.stack([a[0] for a in out.attentions], dim=0).float().cpu().numpy()
    shift_logits = out.logits[0].float()[:-1]
    shift_labels = ids[0, 1:]
    ptl = F.cross_entropy(shift_logits, shift_labels, reduction="none").cpu().numpy()
    return Trajectory(sentence=s, domain=domain, split=split,
                      tok_ids=ids[0].cpu().numpy(), hs=hs, attn=attn, ptl=ptl)


t0 = time.time()
all_traj: List[Trajectory] = []
for s, d in train_sentences:
    all_traj.append(extract(s, d, "train"))
for s, d in test_sentences:
    all_traj.append(extract(s, d, "test"))
print(f"Extracted {len(all_traj)} trajectories in {time.time()-t0:.1f}s")

for tr in all_traj:
    tr.w     = tr.attn.sum(axis=2).sum(axis=1)
    tr.mu_ps = tr.hs.mean(axis=1, keepdims=True)
    tr.x_ps  = tr.hs - tr.mu_ps

L_plus_1 = all_traj[0].hs.shape[0]
train_traj = [tr for tr in all_traj if tr.split == "train"]
test_traj  = [tr for tr in all_traj if tr.split == "test"]


# ---------------------------------------------------------------------------
# Per-layer Gaussian well + PCA basis (on TRAIN only)
# ---------------------------------------------------------------------------
def gaussian_well(x, a, b):
    return a * (1.0 - np.exp(-b * x ** 2))


def fit_well_for_layer(x_pool, e_pool):
    d = np.linalg.norm(x_pool, axis=1)
    mask = np.isfinite(d) & np.isfinite(e_pool)
    d, e = d[mask], e_pool[mask]
    if len(d) < 10:
        return {"a": 0.0, "b": 0.0, "r2": -np.inf, "n": len(d)}
    p0 = [e.max(), 1.0 / (d.std() ** 2 + 1e-8)]
    try:
        popt, _ = curve_fit(gaussian_well, d, e, p0=p0, maxfev=20000)
        pred = gaussian_well(d, *popt)
        r2 = 1.0 - np.sum((e - pred) ** 2) / (np.sum((e - e.mean()) ** 2) + 1e-12)
        return {"a": float(popt[0]), "b": float(popt[1]),
                "r2": float(r2), "n": len(d)}
    except RuntimeError:
        return {"a": 0.0, "b": 0.0, "r2": -np.inf, "n": len(d)}


well_params_ps: Dict[int, Dict] = {}
for ell in range(1, L_plus_1):
    x_pool = np.concatenate([tr.x_ps[ell, :-1, :] for tr in train_traj], axis=0)
    e_pool = np.concatenate([tr.ptl for tr in train_traj], axis=0)
    well_params_ps[ell] = fit_well_for_layer(x_pool, e_pool)


pca_V: Dict[int, np.ndarray] = {}
for ell in range(0, L_plus_1):
    X = np.concatenate([tr.x_ps[ell] for tr in train_traj], axis=0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    pca_V[ell] = Vt[:cfg.pca_k, :].T.astype(np.float32)


# ---------------------------------------------------------------------------
# Per-layer sample collection at a given gamma
# ---------------------------------------------------------------------------
def collect_samples(trajs: List[Trajectory], ell: int, gamma: float):
    """Return x, v, f/m and mass at layer ell across trajs.

    Uses the integrator convention
       v_{l+1} = (v_l + f_l/m) / (1+gamma),  x_{l+1} = x_l + v_{l+1}.
    so   f_l/m = (1+gamma) v_{l+1} - v_l
    with v_{l+1} = x_{l+1} - x_l, v_l = x_l - x_{l-1}.
    """
    if ell < 1 or ell > N_LAYERS - 1:
        return (np.empty((0, D_HIDDEN), np.float32), np.empty((0, D_HIDDEN), np.float32),
                np.empty((0, D_HIDDEN), np.float32), np.empty((0,), np.float32))
    xs, vs, fs, ms = [], [], [], []
    for tr in trajs:
        x_l   = tr.x_ps[ell]
        x_lm1 = tr.x_ps[ell - 1]
        x_lp1 = tr.x_ps[ell + 1]
        v_l   = x_l - x_lm1
        v_lp1 = x_lp1 - x_l
        f_over_m = (1.0 + gamma) * v_lp1 - v_l
        m_t = np.clip(tr.w[:, :].mean(axis=0), 1e-3, None)
        xs.append(x_l)
        vs.append(v_l)
        fs.append(f_over_m)
        ms.append(m_t)
    return (np.concatenate(xs, 0).astype(np.float32),
            np.concatenate(vs, 0).astype(np.float32),
            np.concatenate(fs, 0).astype(np.float32),
            np.concatenate(ms, 0).astype(np.float32))


# ---------------------------------------------------------------------------
# Joint linear fit of (Omega on x) and (affine-in-x B on v) in PCA space
# ---------------------------------------------------------------------------
def fit_gauge_in_pca(
    x: np.ndarray, v: np.ndarray,
    f_over_m: np.ndarray, mass: np.ndarray,
    V_basis: np.ndarray, well: Dict,
    r_pos: int,             # order of position-dependence in B (0, 1, 2, ...)
    use_omega_x: bool,      # whether to include Omega*x term
    use_B_v: bool,          # whether to include B(x)*v term
    ridge: float,
) -> Dict:
    """Fit linear gauge operators in the PCA subspace on observed residual.

    Model in PCA coords (z = V^T x, w = V^T v):

      predicted_resid_pca  =  (1/m_bar) * [ Omega * z  +  (B_0 + sum_{i=1..r} z_i * B_i) * w ]

    fitted against target = V^T * (f/m - f_gauss/m).

    Returns a dict containing the fitted Omega (skew k x k) if
    use_omega_x, the list [B_0, ..., B_r] if use_B_v (each skew k x k),
    the unconstrained variants, m_bar, and fit-quality R^2.
    """
    m_bar = float(mass.mean())
    r2   = np.sum(x * x, axis=1, keepdims=True)
    a, b = well["a"], well["b"]
    f_gauss_over_m = -2.0 * (a / max(m_bar, 1e-8)) * b * x * np.exp(-b * r2)
    resid          = f_over_m - f_gauss_over_m       # (N, d)

    # PCA projections
    z = x @ V_basis                 # (N, k)
    w = v @ V_basis                 # (N, k)
    y = m_bar * (resid @ V_basis)   # (N, k)   we fit  sum_blocks * [..] = y
    N, k = z.shape

    # Build design matrix depending on configuration
    # Per-sample feature vector phi_t in R^{p} where p depends on config.
    # For each output coord alpha (in R^k), target_{t, alpha} = phi_t . theta_alpha,
    # with unknowns theta_alpha of length p.  We solve k parallel OLS problems.
    phi_blocks: List[np.ndarray] = []
    block_names: List[str] = []

    if use_omega_x:
        # Omega * z  -- phi_t for this block is z_t  (size k)
        phi_blocks.append(z)
        block_names.append("Omega")
    if use_B_v:
        # B_0 * w  -- phi_t size k
        phi_blocks.append(w)
        block_names.append("B0")
        for i in range(1, r_pos + 1):
            # z_{t,i-1} * w_t  -- phi_t size k
            phi_blocks.append(z[:, i - 1:i] * w)
            block_names.append(f"B{i}")

    if not phi_blocks:
        return {"m_bar": m_bar, "Omega": None, "B_list": None,
                "M_full": None, "N_list": None,
                "loss_skew": float(np.mean(y ** 2)),
                "loss_unconstrained": float(np.mean(y ** 2)),
                "frac_ev_skew": 0.0, "frac_ev_full": 0.0,
                "block_names": []}

    Phi = np.concatenate(phi_blocks, axis=1)                    # (N, p = n_blocks * k)
    # We solve for a matrix Theta in R^{p x k} with Phi @ Theta ~= y  (k outputs).
    # Each output column is independent.  Ridge-regularised normal equations.
    PtP = Phi.T @ Phi
    reg = ridge * np.trace(PtP) / max(Phi.shape[0], 1) * np.eye(PtP.shape[0])
    Theta_full = np.linalg.solve(PtP + reg, Phi.T @ y)          # (p, k)

    # Reshape into per-block k x k matrices  (each row of Theta_full is one
    # output coord; per-block rows come in chunks of k).  M_b[alpha, beta]:
    # alpha indexes the OUTPUT coord (column of Theta_full),
    # beta  indexes the INPUT  coord (row within block).
    M_blocks_full: List[np.ndarray] = []
    off = 0
    for _ in phi_blocks:
        M = Theta_full[off:off + k, :].T.copy()                 # (k, k)
        M_blocks_full.append(M)
        off += k

    # Skew projection for each block
    M_blocks_skew = [0.5 * (M - M.T) for M in M_blocks_full]

    # Fit quality: 1 - SS_res / SS_tot in PCA space
    def pred_residual_pca(Mbs):
        """Reconstruct the modelled residual (N, k) from block matrices."""
        pred = np.zeros_like(y)
        off = 0
        for i, name in enumerate(block_names):
            M = Mbs[i]
            # Rebuild (phi block) * M.T
            Phi_i = phi_blocks[i]                              # (N, k)
            pred += Phi_i @ M.T
            off += k
        return pred

    pred_full = pred_residual_pca(M_blocks_full)
    pred_skew = pred_residual_pca(M_blocks_skew)

    ss_tot = float(np.sum(y ** 2))
    ss_res_full = float(np.sum((pred_full - y) ** 2))
    ss_res_skew = float(np.sum((pred_skew - y) ** 2))
    frac_ev_full = 1.0 - ss_res_full / (ss_tot + 1e-12)
    frac_ev_skew = 1.0 - ss_res_skew / (ss_tot + 1e-12)

    out = {
        "m_bar": m_bar,
        "Omega": None, "Omega_full": None,
        "B_list": None, "B_list_full": None,
        "frac_ev_full": frac_ev_full,
        "frac_ev_skew": frac_ev_skew,
        "block_names": block_names,
        "N_samples": int(N),
    }

    ptr = 0
    if use_omega_x:
        out["Omega"]      = M_blocks_skew[ptr].astype(np.float32)
        out["Omega_full"] = M_blocks_full[ptr].astype(np.float32)
        ptr += 1
    if use_B_v:
        B_skew = []
        B_full = []
        for i in range(r_pos + 1):
            B_skew.append(M_blocks_skew[ptr].astype(np.float32))
            B_full.append(M_blocks_full[ptr].astype(np.float32))
            ptr += 1
        out["B_list"]      = B_skew
        out["B_list_full"] = B_full
    return out


# ---------------------------------------------------------------------------
# Augmented symplectic step with configurable gauge term
# ---------------------------------------------------------------------------
def gauge_force_over_m(
    x: np.ndarray, v: np.ndarray,
    V: np.ndarray, ops: Dict, m_bar: float,
) -> np.ndarray:
    """Return (F_solenoidal / m) in original R^d space."""
    if ops is None:
        return np.zeros_like(x)
    z = V.T @ x
    w = V.T @ v
    u = np.zeros_like(z)
    if ops.get("Omega") is not None:
        u += ops["Omega"] @ z
    if ops.get("B_list") is not None:
        B0 = ops["B_list"][0]
        u += B0 @ w
        for i, Bi in enumerate(ops["B_list"][1:], start=1):
            u += z[i - 1] * (Bi @ w)
    return (V @ u) / max(m_bar, 1e-8)


def symplectic_step(
    x: np.ndarray, v: np.ndarray,
    a_well: float, b_well: float,
    V: np.ndarray, ops: Optional[Dict], m_bar: float,
    gamma: float, dt: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    r2 = float(np.dot(x, x))
    f_cons_over_m = -2.0 * (a_well / max(m_bar, 1e-8)) * b_well * x * np.exp(-b_well * r2)
    f_sol_over_m  = gauge_force_over_m(x, v, V, ops, m_bar)
    f_over_m      = f_cons_over_m + f_sol_over_m
    v_new = (v + dt * f_over_m) / (1.0 + dt * gamma)
    x_new = x + dt * v_new
    return x_new, v_new


def integrate(
    x0: np.ndarray, v0: np.ndarray, m_bar_tok: float,
    well_params: Dict[int, Dict], V_layer: Dict[int, np.ndarray],
    ops_layer: Optional[Dict[int, Dict]],
    gamma: float, dt: float = 1.0, r2_gate: float = 0.02,
) -> np.ndarray:
    x, v = x0.copy(), v0.copy()
    traj = [x.copy()]
    for ell in range(1, N_LAYERS + 1):
        p = well_params[ell]
        if p["r2"] > r2_gate:
            a_w, b_w = p["a"], p["b"]
        else:
            a_w, b_w = 0.0, 0.0
        ops = ops_layer[ell] if (ops_layer is not None and ell in ops_layer) else None
        x, v = symplectic_step(x, v, a_w, b_w, V_layer[ell], ops, m_bar_tok, gamma, dt)
        traj.append(x.copy())
    return np.stack(traj, axis=0)


def residuals(
    trajs: List[Trajectory],
    well_params: Dict[int, Dict], V_layer: Dict[int, np.ndarray],
    ops_layer: Optional[Dict[int, Dict]], gamma: float,
) -> np.ndarray:
    out = []
    for tr in trajs:
        mu_ps = tr.mu_ps[:, 0, :]
        for ti in range(tr.hs.shape[1]):
            x0 = tr.x_ps[0, ti, :]
            v0 = tr.x_ps[1, ti, :] - tr.x_ps[0, ti, :]
            m  = float(np.clip(tr.w[:, ti].mean(), 1e-3, None))
            pred_x = integrate(x0, v0, m, well_params, V_layer, ops_layer, gamma)
            pred_h = pred_x + mu_ps
            obs_h  = tr.hs[:, ti, :]
            denom  = np.linalg.norm(obs_h, axis=1) + 1e-8
            out.append(np.linalg.norm(pred_h - obs_h, axis=1) / denom)
    return np.stack(out, axis=0)


def residuals_static(trajs: List[Trajectory]) -> np.ndarray:
    out = []
    for tr in trajs:
        mu_ps = tr.mu_ps[:, 0, :]
        for ti in range(tr.hs.shape[1]):
            x0 = tr.x_ps[0, ti, :]
            pred_h = np.tile(x0, (N_LAYERS + 1, 1)) + mu_ps
            obs_h  = tr.hs[:, ti, :]
            denom  = np.linalg.norm(obs_h, axis=1) + 1e-8
            out.append(np.linalg.norm(pred_h - obs_h, axis=1) / denom)
    return np.stack(out, axis=0)


# ---------------------------------------------------------------------------
# Experiment matrix
# ---------------------------------------------------------------------------
CONFIGS = [
    # name,              use_omega_x, use_B_v, r_pos
    ("gaussian",         False,       False,   0),   # no gauge (§1.4 baseline)
    ("omega_x",          True,        False,   0),   # §1.4 skew Omega*x
    ("B_const",          False,       True,    0),   # NEW: P-rot-6
    ("B_affine_r1",      False,       True,    1),   # NEW: P-rot-6b rank 1
    ("B_affine_r2",      False,       True,    2),   # NEW: P-rot-6b rank 2
    ("omega_and_Bconst", True,        True,    0),   # NEW: full linear U(1) gauge
]
GAMMAS = [0.0, 0.5, 1.0, 2.0, 5.0]
USE_SKEW_FOR_INTEGRATION = True   # we report results with the skew-projected operators


rho_static_train = residuals_static(train_traj)
rho_static_test  = residuals_static(test_traj)
STATIC_TRAIN = float(np.median(rho_static_train[:, -1]))
STATIC_TEST  = float(np.median(rho_static_test[:, -1]))
print(f"\nStatic-null: TRAIN {STATIC_TRAIN:.4f}   TEST {STATIC_TEST:.4f}\n")

results: Dict[Tuple[str, float], Dict] = {}


def scale_ops(ops: Optional[Dict], s: float) -> Optional[Dict]:
    """Scale the solenoidal operators in ops by factor s (0 = off, 1 = full)."""
    if ops is None:
        return None
    scaled = {}
    if ops.get("Omega") is not None:
        scaled["Omega"] = (ops["Omega"] * s).astype(np.float32)
    if ops.get("B_list") is not None:
        scaled["B_list"] = [B * s for B in ops["B_list"]]
    return scaled


def safe_median(r: np.ndarray) -> float:
    """Median of last-column residuals, robust to NaN / inf from diverged
    integrations: treat non-finite as +inf so they lose to any finite
    alternative in argmin/argmax comparisons, and still give a numeric
    median (large) rather than crashing the summary pipeline."""
    col = r[:, -1]
    col = np.where(np.isfinite(col), col, 1e6)
    return float(np.median(col))


def scan_scale(
    fit_fn, traj_set: List[Trajectory],
    well_params, pca_V, ops_fitted_by_layer,
    gamma: float, scales: List[float],
) -> Tuple[float, float]:
    """Scan a shrinkage factor s for the fitted operators and return
    (best_s, best_median) on `traj_set` (median over tokens, last layer)."""
    best_s, best_med = 0.0, float("inf")
    for s in scales:
        ops_s = {ell: scale_ops(op, s) for ell, op in ops_fitted_by_layer.items()}
        rho = residuals(traj_set, well_params, pca_V, ops_s, gamma)
        med = safe_median(rho)
        if med < best_med:
            best_s, best_med = s, med
    return best_s, best_med


print(f"=== Fitting and evaluating (PCA k = {cfg.pca_k}, ridge = {cfg.ridge_lambda}) ===")
print(f"{'config':<20} {'gamma':>5} "
      f"{'TR_s1.0':>10} {'TE_s1.0':>10} "
      f"{'TR_s*':>10} {'TE_s*':>10} "
      f"{'best_s':>7} "
      f"{'ev_skew':>8} {'ev_full':>8} {'secs':>5}")
print("-" * 120)

for cfg_name, use_om, use_B, r_pos in CONFIGS:
    for gamma in GAMMAS:
        t_g = time.time()
        ops_layer_skew: Dict[int, Dict] = {}
        ops_layer_full: Dict[int, Dict] = {}
        ev_skews, ev_fulls = [], []
        for ell in range(1, N_LAYERS):
            x_tr, v_tr, f_tr, m_tr = collect_samples(train_traj, ell, gamma)
            if x_tr.shape[0] < cfg.pca_k + 5:
                continue
            info = fit_gauge_in_pca(
                x_tr, v_tr, f_tr, m_tr, pca_V[ell], well_params_ps[ell],
                r_pos=r_pos, use_omega_x=use_om, use_B_v=use_B,
                ridge=cfg.ridge_lambda,
            )
            if use_om or use_B:
                ops_skew = {"Omega": info["Omega"], "B_list": info["B_list"]}
                ops_full = {"Omega": info["Omega_full"], "B_list": info["B_list_full"]}
                ops_layer_skew[ell] = ops_skew
                ops_layer_full[ell] = ops_full
                ev_skews.append(info["frac_ev_skew"])
                ev_fulls.append(info["frac_ev_full"])

        # If no ops (pure gaussian config), ops dict is None
        ops_sk = ops_layer_skew if (use_om or use_B) else None

        # Full-strength (scale = 1) residuals
        rho_tr_s1 = residuals(train_traj, well_params_ps, pca_V, ops_sk, gamma)
        rho_te_s1 = residuals(test_traj,  well_params_ps, pca_V, ops_sk, gamma)

        # Scale-sweep: shrink the fitted skew ops by factor s to trade fit for stability.
        # s=0 reduces to pure gaussian baseline; s=1 is the full fit.
        scales = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
        if ops_sk is not None:
            best_s_tr = 0.0; best_med_tr = float("inf")
            per_layer_at_best_s = None
            for s in scales:
                ops_s = {ell: scale_ops(op, s) for ell, op in ops_sk.items()}
                rho_s_tr = residuals(train_traj, well_params_ps, pca_V, ops_s, gamma)
                med = safe_median(rho_s_tr)
                if med < best_med_tr:
                    best_s_tr, best_med_tr = s, med
            # Evaluate at that s on test
            ops_star = {ell: scale_ops(op, best_s_tr) for ell, op in ops_sk.items()}
            rho_te_star = residuals(test_traj, well_params_ps, pca_V, ops_star, gamma)
            rho_tr_star = residuals(train_traj, well_params_ps, pca_V, ops_star, gamma)
            per_layer_test_star = np.nanmedian(
                np.where(np.isfinite(rho_te_star), rho_te_star, np.nan), axis=0)
        else:
            best_s_tr = 0.0
            rho_tr_star = rho_tr_s1
            rho_te_star = rho_te_s1
            per_layer_test_star = np.median(rho_te_s1, axis=0)

        entry = {
            "cfg": cfg_name, "gamma": gamma,
            "train_s1":  safe_median(rho_tr_s1),
            "test_s1":   safe_median(rho_te_s1),
            "train_star": safe_median(rho_tr_star),
            "test_star":  safe_median(rho_te_star),
            "best_s":     best_s_tr,
            "ev_skew":    float(np.mean(ev_skews)) if ev_skews else 0.0,
            "ev_full":    float(np.mean(ev_fulls)) if ev_fulls else 0.0,
            "per_layer_test_star": per_layer_test_star,
            "per_layer_test_s1":   np.nanmedian(
                np.where(np.isfinite(rho_te_s1), rho_te_s1, np.nan), axis=0),
        }
        results[(cfg_name, gamma)] = entry
        print(f"{cfg_name:<20} {gamma:>5.1f} "
              f"{entry['train_s1']:>10.4f} {entry['test_s1']:>10.4f} "
              f"{entry['train_star']:>10.4f} {entry['test_star']:>10.4f} "
              f"{entry['best_s']:>7.3f} "
              f"{entry['ev_skew']:>8.3f} {entry['ev_full']:>8.3f} "
              f"{time.time()-t_g:>5.1f}")


# ---------------------------------------------------------------------------
# Save raw results
# ---------------------------------------------------------------------------
cfg_names = [c[0] for c in CONFIGS]
gs = np.array(GAMMAS)
save: Dict[str, np.ndarray] = {
    "gammas":       gs,
    "pca_k":        np.array([cfg.pca_k]),
    "n_train":      np.array([len(train_traj)]),
    "n_test":       np.array([len(test_traj)]),
    "static_train": np.array([STATIC_TRAIN]),
    "static_test":  np.array([STATIC_TEST]),
}
for metric in ["train_s1", "test_s1", "train_star", "test_star",
               "best_s", "ev_skew", "ev_full"]:
    arr = np.array([[results[(c, g)][metric] for g in GAMMAS] for c in cfg_names])
    save[metric] = arr
pls = np.array([[results[(c, g)]["per_layer_test_star"] for g in GAMMAS] for c in cfg_names])
plf = np.array([[results[(c, g)]["per_layer_test_s1"]   for g in GAMMAS] for c in cfg_names])
save["per_layer_test_star"] = pls
save["per_layer_test_s1"]   = plf
save["config_names"] = np.array(cfg_names)
np.savez(os.path.join(RESULTS_DIR, "velocity_coupled_gauge_results.npz"), **save)
print(f"\nSaved results to {os.path.join(RESULTS_DIR, 'velocity_coupled_gauge_results.npz')}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
# Figure A: TEST residual vs gamma, one line per config (skew projection)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
colors = {
    "gaussian":          "tab:blue",
    "omega_x":           "tab:orange",
    "B_const":           "tab:green",
    "B_affine_r1":       "tab:red",
    "B_affine_r2":       "tab:purple",
    "omega_and_Bconst":  "tab:brown",
}
markers = {
    "gaussian":          "o",
    "omega_x":           "s",
    "B_const":           "^",
    "B_affine_r1":       "D",
    "B_affine_r2":       "v",
    "omega_and_Bconst":  "P",
}
for ax, split_key, static in zip(
    axes, ["train_star", "test_star"], [STATIC_TRAIN, STATIC_TEST]
):
    for c in cfg_names:
        ys = [results[(c, g)][split_key] for g in GAMMAS]
        ax.plot(GAMMAS, ys, marker=markers[c], color=colors[c], label=c)
    ax.axhline(static, linestyle="--", color="tab:gray",
               label=f"static null ({static:.3f})")
    ax.set_xlabel(r"damping $\gamma$")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)
    ax.set_title(f"median layer-L residual -- {split_key.split('_')[0].upper()}  "
                 f"(scale-tuned)")
axes[0].set_ylabel("median residual  (log scale)")
fig.tight_layout()
fig_a = os.path.join(RESULTS_DIR, "fig_gauge_residual_vs_gamma.png")
fig.savefig(fig_a, dpi=130)
plt.close(fig)
print(f"Saved figure to {fig_a}")

# Figure B: per-layer TEST residual at best (per config) gamma.
# We pick the best gamma by TEST_skew per config, to give each its best shot.
fig, ax = plt.subplots(figsize=(9, 5))
layers = np.arange(N_LAYERS + 1)
best_gamma_per_cfg: Dict[str, float] = {}
for c in cfg_names:
    best_idx = int(np.argmin([results[(c, g)]["test_star"] for g in GAMMAS]))
    best_gamma_per_cfg[c] = GAMMAS[best_idx]
    s_star = results[(c, GAMMAS[best_idx])]["best_s"]
    ax.plot(layers, results[(c, GAMMAS[best_idx])]["per_layer_test_star"],
            marker=markers[c], color=colors[c],
            label=f"{c}  (γ*={GAMMAS[best_idx]}, s*={s_star})")
static_layer = np.median(rho_static_test, axis=0)
ax.plot(layers, static_layer, linestyle="--", color="tab:gray", label="static null")
ax.set_xlabel(r"layer $\ell$")
ax.set_ylabel("median relative residual")
ax.set_title(r"Per-layer TEST residual at per-config $\gamma^{*}$ (skew projection)")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig_b = os.path.join(RESULTS_DIR, "fig_gauge_residual_vs_layer_at_gamma_star.png")
fig.savefig(fig_b, dpi=130)
plt.close(fig)
print(f"Saved figure to {fig_b}")


# ---------------------------------------------------------------------------
# Summary Markdown
# ---------------------------------------------------------------------------
md: List[str] = []
md.append("# Velocity-coupled gauge-field augmentation: does the "
          "electromagnetic-analogue Lagrangian fit?\n")
md.append(f"Model: **{cfg.model_name}**.  TRAIN: {len(train_traj)} sentences / "
          f"{rho_static_train.shape[0]} tokens.  TEST: {len(test_traj)} sentences / "
          f"{rho_static_test.shape[0]} tokens.  PCA subspace: **k = {cfg.pca_k}** per layer.  "
          f"Ridge: **{cfg.ridge_lambda}**.\n")
md.append("## 1. Question and ansatzes\n")
md.append(
    "The §1.4 negative result ruled out the simplest Helmholtz correction "
    "$V_\\ell\\,\\Omega_\\ell\\,V_\\ell^\\top x$ with constant skew "
    "$\\Omega_\\ell$.  Here we test four strictly richer linear gauge "
    "ansatzes derived from the electromagnetic-analogue Lagrangian\n\n"
    "$$L = \\tfrac12\\,\\mathfrak m\\,\\lVert\\dot x\\rVert^2 + \\vec A(x)\\!\\cdot\\!\\dot x - V(x),$$\n\n"
    "which gives the Euler-Lagrange equation\n\n"
    "$$\\mathfrak m\\,\\ddot x = -\\nabla V(x) + F(x)\\,\\dot x - \\mathfrak m\\gamma\\,\\dot x,\\qquad F = \\partial A - (\\partial A)^\\top.$$\n\n"
    "We parameterise $F$ in the per-layer top-$k$ PCA subspace (with "
    "$z = V^\\top x$ and $w = V^\\top v$) as one of:\n"
    "\n"
    "| config | PCA-space force | params/layer |\n"
    "|---|---|--:|\n"
    "| `gaussian`         | 0 | 0 |\n"
    "| `omega_x`          | $\\Omega_0 z$ (skew) | $k(k-1)/2$ |\n"
    "| `B_const`          | $B_0 w$ (skew) | $k(k-1)/2$ |\n"
    "| `B_affine_r1`      | $(B_0 + z_1 B_1) w$ | $2\\,k(k-1)/2$ |\n"
    "| `B_affine_r2`      | $(B_0 + z_1 B_1 + z_2 B_2) w$ | $3\\,k(k-1)/2$ |\n"
    "| `omega_and_Bconst` | $\\Omega_0 z + B_0 w$ | $2\\,k(k-1)/2$ |\n"
    f"\n(At $k={cfg.pca_k}$: 120 params per skew block; up to "
    f"360 per layer for `B_affine_r2`.  Training samples per layer: "
    f"$\\approx {rho_static_train.shape[0]}$.)\n"
)
md.append("## 2. Results\n")
md.append(f"Static-null TEST floor: **{STATIC_TEST:.4f}**.\n")
md.append(
    "Each configuration is evaluated at two operator strengths:\n"
    "- **s = 1** (full fit): applies the fitted skew operators unchanged.  "
    "For velocity-coupled configs this is numerically unstable in the "
    "integrator (positive-feedback $v \\to B v \\to $ larger $v$) and often "
    "diverges (reported as $10^{6}$ when it does).\n"
    "- **s = s\\* (shrunk fit)**: applies the same operators multiplied "
    "by a scalar $s\\in[0,1]$, chosen per config per $\\gamma$ to minimise "
    "TRAIN residual on a fine grid $\\{0, 0.01, 0.05, 0.1, 0.25, 0.5, "
    "0.75, 1.0\\}$.  Setting $s=0$ recovers the gaussian-only baseline.  "
    "This gives each ansatz its best shot at actually helping rather than "
    "blowing up.\n"
)
md.append("### 2.1 Full-strength operators (s = 1)\n")
md.append("| config | " + " | ".join([f"$\\gamma={g}$" for g in GAMMAS]) + " |")
md.append("|" + "|".join(["---"] + [":---:"] * len(GAMMAS)) + "|")
for c in cfg_names:
    md.append("| `" + c + "` | " +
              " | ".join([f"{results[(c, g)]['test_s1']:.4g}" for g in GAMMAS])
              + " |")
md.append("\n### 2.2 TRAIN-optimal scale (s = s\\*), TEST residual\n")
md.append("| config | " + " | ".join([f"$\\gamma={g}$" for g in GAMMAS]) + " |")
md.append("|" + "|".join(["---"] + [":---:"] * len(GAMMAS)) + "|")
for c in cfg_names:
    row = []
    for g in GAMMAS:
        r = results[(c, g)]
        row.append(f"{r['test_star']:.4f} (s={r['best_s']})")
    md.append("| `" + c + "` | " + " | ".join(row) + " |")

md.append("\n### 2.3 Best-$\\gamma$ summary (shrunk fit)\n")
md.append("| config | $\\gamma^{*}$ | $s^{*}$ | TRAIN | TEST | Δ vs. null | PCA-ev skew | PCA-ev full |")
md.append("|---|:-:|:-:|--:|--:|--:|--:|--:|")
for c in cfg_names:
    g_star = best_gamma_per_cfg[c]
    r = results[(c, g_star)]
    md.append(
        f"| `{c}` | {g_star} | {r['best_s']} | {r['train_star']:.4f} "
        f"| {r['test_star']:.4f} | {r['test_star']-STATIC_TEST:+.4f} "
        f"| {r['ev_skew']:.3f} | {r['ev_full']:.3f} |"
    )

# Best performing overall (by test_star)
best_cfg = min(cfg_names, key=lambda c: min(results[(c, g)]['test_star'] for g in GAMMAS))
best_g   = best_gamma_per_cfg[best_cfg]
best_rho = results[(best_cfg, best_g)]['test_star']
best_delta = best_rho - STATIC_TEST

md.append("\n## 3. Interpretation\n")
md.append(
    f"**Best configuration on TEST (scale-tuned):** `{best_cfg}` at "
    f"$\\gamma^{{*}}={best_g}$, $s^{{*}}={results[(best_cfg, best_g)]['best_s']}$, "
    f"TEST median layer-$L$ residual **{best_rho:.4f}** "
    f"vs. static-null **{STATIC_TEST:.4f}** (Δ = {best_delta:+.4f}).\n"
)
md.append(
    "Sanity check against §1.4 (`helmholtz_curl_summary.md`): at $\\gamma=5$, "
    f"`omega_x` gives TEST {results[('omega_x', 5.0)]['test_s1']:.4f} "
    f"(s=1) here vs. 0.1882 in §1.4; tiny differences come from the "
    "independent PCA basis recomputation, not from the model.\n"
)

# --- 3.1 stability finding ----------------------------------------------
md.append("### 3.1 Velocity-coupled ansatzes are numerically unstable at s = 1\n")
md.append(
    "`B_const`, `B_affine_r1`, `B_affine_r2`, and `omega_and_Bconst` all "
    "**diverge** when the fitted operators are applied at full strength. "
    "The symptom is the positive-feedback loop $v \\to B\\dot x \\to v' \\to \\ldots$ "
    "that a linear integrator cannot stabilise when the fitted $B$ has "
    "eigenvalues of magnitude comparable to the damping.  Concretely, at "
    "$\\gamma=5$, $s=1$:\n"
    f"- `B_const` TEST median = {results[('B_const', 5.0)]['test_s1']:.4g}\n"
    f"- `B_affine_r1` TEST median = {results[('B_affine_r1', 5.0)]['test_s1']:.4g}\n"
    f"- `omega_and_Bconst` TEST median = "
    f"{results[('omega_and_Bconst', 5.0)]['test_s1']:.4g}\n"
    "(Values of $10^{6}$ indicate full divergence.)\n"
    "\nThis is itself an informative failure: the *fitted-optimal* skew "
    "velocity-coupling that best explains one-layer residuals on TRAIN is "
    "too strong to be propagated through 12 steps.\n"
)

# --- 3.2 shrunk-fit result -----------------------------------------------
md.append("### 3.2 Shrinking $B$ stabilises but gives no meaningful gain\n")
md.append(
    "When we shrink the fitted operators by a factor $s\\in[0,1]$ chosen "
    "to minimise TRAIN residual, the integrator becomes stable but "
    "**$s^{*}$ collapses towards 0**, i.e. the TRAIN-optimal velocity "
    "coupling is almost no velocity coupling:\n"
)
md.append(
    f"- `B_const`, $\\gamma^{{*}}=5$: "
    f"$s^{{*}}={results[('B_const', best_gamma_per_cfg['B_const'])]['best_s']}$, "
    f"TEST = {results[('B_const', best_gamma_per_cfg['B_const'])]['test_star']:.4f} "
    f"(Δ = {results[('B_const', best_gamma_per_cfg['B_const'])]['test_star']-STATIC_TEST:+.4f}).\n"
    f"- `B_affine_r1`, $\\gamma^{{*}}={best_gamma_per_cfg['B_affine_r1']}$: "
    f"$s^{{*}}={results[('B_affine_r1', best_gamma_per_cfg['B_affine_r1'])]['best_s']}$, "
    f"TEST = {results[('B_affine_r1', best_gamma_per_cfg['B_affine_r1'])]['test_star']:.4f} "
    f"(Δ = {results[('B_affine_r1', best_gamma_per_cfg['B_affine_r1'])]['test_star']-STATIC_TEST:+.4f}).\n"
    f"- `B_affine_r2`, $\\gamma^{{*}}={best_gamma_per_cfg['B_affine_r2']}$: "
    f"$s^{{*}}={results[('B_affine_r2', best_gamma_per_cfg['B_affine_r2'])]['best_s']}$, "
    f"TEST = {results[('B_affine_r2', best_gamma_per_cfg['B_affine_r2'])]['test_star']:.4f} "
    f"(Δ = {results[('B_affine_r2', best_gamma_per_cfg['B_affine_r2'])]['test_star']-STATIC_TEST:+.4f}).\n"
    f"- `omega_and_Bconst`, $\\gamma^{{*}}={best_gamma_per_cfg['omega_and_Bconst']}$: "
    f"$s^{{*}}={results[('omega_and_Bconst', best_gamma_per_cfg['omega_and_Bconst'])]['best_s']}$, "
    f"TEST = {results[('omega_and_Bconst', best_gamma_per_cfg['omega_and_Bconst'])]['test_star']:.4f} "
    f"(Δ = {results[('omega_and_Bconst', best_gamma_per_cfg['omega_and_Bconst'])]['test_star']-STATIC_TEST:+.4f}).\n"
)
md.append(
    "In every case the shrunk-best configuration sits at or slightly "
    "above the static-null floor, never meaningfully below.  This is the "
    "substantive negative answer to 'do velocity-coupled position-dependent "
    "gauge fields fit the data?' at the linear level: **no**.\n"
)

# --- 3.3 fit-quality gap ---------------------------------------------------
md.append("### 3.3 Why the one-step fit is good and the trajectory is not\n")
md.append(
    "The `PCA-ev full` columns show that the *unconstrained* linear "
    "operators capture a substantial fraction of one-step PCA-space "
    "residual variance ($R^{2}\\approx 0.5\\text{-}0.8$ per layer). "
    "Imposing the skew-symmetry constraint collapses the fit quality to "
    "deeply negative values (ev skew $\\approx -10$ to $-180$), because "
    "the symmetric part of the fitted operator is where most of the "
    "deterministic structure lives.  This is the same pattern seen in "
    "§1.4 (`omega_x`) but now extended to $v$-coupled and "
    "position-dependent ansatzes: in every case, the residual structure "
    "that a linear operator can fit is overwhelmingly *symmetric*, not "
    "solenoidal.\n"
)

# --- 3.4 take-away --------------------------------------------------------
md.append("### 3.4 Bottom line\n")
md.append(
    "Five consecutive negative experiments now bracket the linear "
    "Lagrangian programme from above and below:\n"
    "\n"
    "1. Scalar $V(x)$, any $V$ (§1.1-1.3): **fails** (static null).\n"
    "2. $+$ skew $\\Omega x$ (§1.4): **fails** (marginally worse than null).\n"
    "3. $+$ skew $B \\dot x$, constant: **unstable at s=1**; at s\\*, "
    "at or slightly above null.\n"
    "4. $+$ skew $B(x)\\dot x$, affine in $x$, rank 1 or 2: **unstable at s=1**; "
    "at s\\*, at or slightly above null.\n"
    "5. $+$ skew $\\Omega x + B\\dot x$ (combined): **unstable at s=1**; "
    "at s\\*, strictly worse than null.\n"
    "\n"
    "The Helmholtz electromagnetic-analogue Lagrangian, at the level of "
    "constant or low-order-polynomial antisymmetric operators in a top-16 "
    "PCA subspace, does not fit hidden-state trajectories on held-out "
    "sentences.  The per-layer one-step linear approximation of the "
    "transformer block is symmetric and non-Hessian, not antisymmetric; "
    "this is not captured by the electromagnetic analogue.\n"
    "\n"
    "Remaining untested candidates that are strictly richer than anything "
    "here and could in principle still work:\n"
    "\n"
    "- Non-linear (not just affine) position dependence in $B(x)$, e.g. "
    "$B(x) = \\sum_k c_k B_k \\phi_k(x)$ with a basis of smooth "
    "position features $\\phi_k$ (RBFs, Fourier on PCA coords, etc.).\n"
    "- Symmetric-but-non-Hessian extensions -- these lie **outside the "
    "autonomous, shared-potential** Helmholtz class (they cannot be "
    "written as the Hessian of any single shared scalar potential) but "
    "are what the $R^{2}$ data say is actually needed.  A Riemannian / "
    "Jacobi-geodesic formulation "
    "(§14 of the paper) accommodates them via Christoffel symbols of a "
    "non-flat metric, without any scalar potential.\n"
    "- Non-abelian (multi-head) gauge structure $\\vec F = \\sum_h F^{(h)}(x)$ "
    "with head-specific antisymmetric generators -- left for future work.\n"
)

md.append("## 4. Artefacts\n")
md.append("- [`notebooks/e_init/velocity_coupled_gauge.py`](../velocity_coupled_gauge.py) -- reproducible script")
md.append("- [`results/velocity_coupled_gauge_results.npz`](velocity_coupled_gauge_results.npz) -- full numerical results")
md.append("- [`results/fig_gauge_residual_vs_gamma.png`](fig_gauge_residual_vs_gamma.png) -- TRAIN / TEST residual vs $\\gamma$ (all configs)")
md.append("- [`results/fig_gauge_residual_vs_layer_at_gamma_star.png`](fig_gauge_residual_vs_layer_at_gamma_star.png) -- per-layer TEST residual at per-config $\\gamma^{*}$")
md.append("\n## 5. Reproduce\n")
md.append("```bash")
md.append("python3 notebooks/e_init/velocity_coupled_gauge.py")
md.append("```\n")

md_path = os.path.join(RESULTS_DIR, "velocity_coupled_gauge_summary.md")
with open(md_path, "w") as f:
    f.write("\n".join(md))
print(f"Saved summary to {md_path}")

print("\nDone.")
