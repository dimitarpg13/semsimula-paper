#!/usr/bin/env python
"""
Helmholtz-augmented Euler-Lagrange integrator for hidden-state trajectories.

Follow-up to notebooks/e_init/well_functional_form_comparison.py, which
established that no scalar potential V(x) of any bounded-attractive or
power-law form reduces the layer-L residual below the static-null floor.
Here we add a solenoidal (divergence-free) term to the second-order
equation and ask whether it can fit the trajectories.

    m d^2 x / d l^2 = -grad V(x) + F_s(x) - m * gamma * dx/dl          (*)

with F_s(x) parameterised as a linear skew-symmetric operator in a
per-layer PCA basis,

    F_s(x) = V_l * Omega_l * V_l^T * x,       Omega_l = -Omega_l^T,

where V_l in R^{d x k} is the top-k PCA basis of TRAIN hidden states at
layer l, and Omega_l in R^{k x k} is skew-symmetric (k(k-1)/2 free
parameters per layer).

The skew constraint makes F_s automatically divergence-free in the sense
div (V Omega V^T x) = tr(V Omega V^T) = tr(Omega) = 0.

We also fit, as an UPPER BOUND, an unconstrained linear operator
    F_lin(x) = V_l * M_l * V_l^T * x,       M_l in R^{k x k} arbitrary,
which contains both the skew (solenoidal) and symmetric-traceless parts.
Comparing skew-only vs. unconstrained tells us whether the improvement,
if any, comes from the rotational sub-component or from the conservative
linear sub-component.

Protocol
--------
  1. Extract GPT-2 hidden states + NTP loss + attention mass on the
     50-sentence corpus (same as extended_gamma_and_first_order.py).
  2. Train/test split: 40 sentences train, 10 test.  The split is
     balanced across the five CORPUS domains (8 train + 2 test per domain).
  3. Per-layer PCA on TRAIN centered hidden states (x_ps).
  4. Per-layer fit of Omega (skew) and M (unconstrained) by weighted OLS
     on observed force f_obs/m = (1+gamma)*v_{l+1} - v_l, with the
     Gaussian conservative part subtracted first.
  5. Run augmented symplectic Euler on TRAIN and TEST separately;
     sweep gamma in {0, 0.5, 1, 2, 5}.
  6. Report median layer-L residual per configuration and per fold.
  7. Save all artefacts.

Outputs
-------
  results/helmholtz_curl_results.npz
  results/fig_helmholtz_residual_vs_gamma.png
  results/fig_helmholtz_residual_vs_layer_at_gamma_star.png
  results/helmholtz_curl_summary.md

Runtime: GPT-2 extraction ~10 s + PCA + fits + sweeps ~30 s on MPS.
"""

from __future__ import annotations

import json
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
# Config + corpus (matches extended_gamma_and_first_order.py)
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


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


@dataclass
class Config:
    model_name: str = "gpt2"
    device: str = field(default_factory=_pick_device)
    dtype: torch.dtype = field(default_factory=lambda: torch.float32)
    max_length: int = 64
    pca_k: int = 16                 # top-k PCA components per layer
    n_test_per_domain: int = 2      # 2 test sentences per domain --> 10 test
    ridge_lambda: float = 1e-3      # Tikhonov regulariser for OLS
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


# ---------------------------------------------------------------------------
# Train/test split (balanced across domains)
# ---------------------------------------------------------------------------
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
# Load GPT-2 and extract (same as before)
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
    split: str                       # "train" or "test"
    tok_ids: np.ndarray
    hs: np.ndarray                   # (L+1, T, d)
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
    tr.w = tr.attn.sum(axis=2).sum(axis=1)
    tr.mu_ps = tr.hs.mean(axis=1, keepdims=True)
    tr.x_ps = tr.hs - tr.mu_ps

L_plus_1 = all_traj[0].hs.shape[0]
train_traj = [tr for tr in all_traj if tr.split == "train"]
test_traj  = [tr for tr in all_traj if tr.split == "test"]


# ---------------------------------------------------------------------------
# Per-layer Gaussian-well refit on TRAIN only (to avoid test-set leakage)
# ---------------------------------------------------------------------------
def gaussian_well(x, a, b):
    return a * (1.0 - np.exp(-b * x ** 2))


def fit_well_for_layer(x_pool: np.ndarray, e_pool: np.ndarray) -> Dict:
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


# ---------------------------------------------------------------------------
# Per-layer PCA on TRAIN centered hidden states
# ---------------------------------------------------------------------------
pca_V: Dict[int, np.ndarray] = {}   # each V_l in R^{d x k}
for ell in range(0, L_plus_1):
    X = np.concatenate([tr.x_ps[ell] for tr in train_traj], axis=0)  # (N_ell, d)
    # SVD-based PCA (X = U S V^T, columns of V^T are principal directions)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    pca_V[ell] = Vt[:cfg.pca_k, :].T.astype(np.float32)   # d x k
    exp_var = (S[:cfg.pca_k] ** 2).sum() / (S ** 2).sum()
    if ell in (1, 6, 12):
        print(f"  layer {ell:>2}: PCA top-{cfg.pca_k} explains "
              f"{exp_var*100:.1f}% of variance  (N={X.shape[0]})")


# ---------------------------------------------------------------------------
# Collect (x, observed f/m) samples per layer for TRAIN
# ---------------------------------------------------------------------------
# Integrator convention (see e_init_validation.ipynb Stage 5):
#   v_{l+1} = (v_l + f_l/m) / (1 + gamma)
#   x_{l+1} = x_l + v_{l+1}
# so   f_l/m = (1 + gamma) * v_{l+1} - v_l
# with v_{l+1} = x_{l+1} - x_l, and v_l = x_l - x_{l-1} (l >= 1).
#
# We fit Omega at a given gamma.  Because observed f_l/m depends on gamma,
# the fit changes with gamma -- we sweep.

def collect_samples(trajs: List[Trajectory], ell: int, gamma: float):
    """Return (x, f_over_m, mass) arrays for layer ell across given trajectories.

    x:          (N, d) -- centered hidden state at layer ell
    f_over_m:   (N, d) -- observed force / mean mass at layer ell
    mass:       (N,)   -- mean attention mass per token
    """
    xs, fs, ms = [], [], []
    for tr in trajs:
        T = tr.x_ps.shape[1]
        if ell < 1 or ell > N_LAYERS - 1:   # need x_{ell-1} and x_{ell+1}
            continue
        x_l   = tr.x_ps[ell]
        x_lm1 = tr.x_ps[ell - 1]
        x_lp1 = tr.x_ps[ell + 1]
        v_l   = x_l - x_lm1
        v_lp1 = x_lp1 - x_l
        f_over_m = (1.0 + gamma) * v_lp1 - v_l    # (T, d)
        m_t = np.clip(tr.w[:, :].mean(axis=0), 1e-3, None)  # (T,)
        xs.append(x_l)
        fs.append(f_over_m)
        ms.append(m_t)
    if not xs:
        return (np.empty((0, D_HIDDEN), dtype=np.float32),
                np.empty((0, D_HIDDEN), dtype=np.float32),
                np.empty((0,), dtype=np.float32))
    return (np.concatenate(xs, 0).astype(np.float32),
            np.concatenate(fs, 0).astype(np.float32),
            np.concatenate(ms, 0).astype(np.float32))


# ---------------------------------------------------------------------------
# Per-layer Omega fit in PCA subspace (skew-symmetric) + unconstrained M
# ---------------------------------------------------------------------------
def fit_linear_in_pca(
    x: np.ndarray,        # (N, d)   centered hidden states
    f_over_m: np.ndarray, # (N, d)   observed force / m
    mass: np.ndarray,     # (N,)     per-token mass
    V: np.ndarray,        # (d, k)   PCA basis
    well: Dict,           # gaussian well params at this layer
    ridge: float,
) -> Dict[str, np.ndarray]:
    """Fit M and Omega in PCA subspace such that

        V * M * V^T * (x / m_bar) ~= f_over_m - f_gauss/m

    where f_gauss/m = -2*(a/m_bar)*b*x*exp(-b*r^2) is the fitted Gaussian
    conservative force (division by m_bar follows the existing integrator).
    """
    m_bar = float(mass.mean())
    r2 = np.sum(x * x, axis=1, keepdims=True)
    a, b = well["a"], well["b"]
    f_gauss_over_m = -2.0 * (a / max(m_bar, 1e-8)) * b * x * np.exp(-b * r2)
    residual = f_over_m - f_gauss_over_m                             # (N, d)

    # Project to PCA
    x_p = x @ V                           # (N, k)
    r_p = residual @ V                    # (N, k)
    # Model in PCA: residual_p = (1/m_bar) * M * x_p
    # i.e.    m_bar * r_p.T = M * x_p.T   (column conventions for fitting)
    # Solve M: min ||M * X - Y||^2 + ridge ||M||^2
    #   where X = x_p.T (k x N), Y = (m_bar * r_p).T  (k x N)
    X = x_p.T                             # (k, N)
    Y = (m_bar * r_p).T                   # (k, N)
    XXt = X @ X.T                         # (k, k)
    reg = ridge * np.trace(XXt) / max(X.shape[0], 1) * np.eye(X.shape[0])
    M_full = Y @ X.T @ np.linalg.inv(XXt + reg)     # (k, k)
    Omega  = 0.5 * (M_full - M_full.T)               # skew part

    # Train loss (PCA space, pre-lift)
    def loss(M):
        pred_p = (M @ X) / m_bar
        return float(np.mean((pred_p - r_p.T) ** 2))

    # Also fraction of residual variance explained by M_full and Omega alone,
    # computed against f_over_m AFTER subtracting the gaussian part:
    def frac_ev(M):
        pred_p = (M @ X) / m_bar        # (k, N)
        ss_res = float(np.sum((pred_p - r_p.T) ** 2))
        ss_tot = float(np.sum(r_p.T ** 2))
        return 1.0 - ss_res / (ss_tot + 1e-12)

    return {
        "M_full":  M_full.astype(np.float32),
        "Omega":   Omega.astype(np.float32),
        "m_bar":   m_bar,
        "loss_M_full":  loss(M_full),
        "loss_Omega":   loss(Omega),
        "frac_ev_M_full": frac_ev(M_full),
        "frac_ev_Omega":  frac_ev(Omega),
    }


# ---------------------------------------------------------------------------
# Augmented symplectic Euler integrator (with linear solenoidal term)
# ---------------------------------------------------------------------------
def symplectic_step_helmholtz(
    x: np.ndarray, v: np.ndarray,
    a_well: float, b_well: float,
    V: np.ndarray, Mk: np.ndarray,   # V: (d, k); Mk: (k, k) either Omega or M_full or None
    m_bar: float, gamma: float, dt: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    r2 = float(np.dot(x, x))
    f_over_m_cons = -2.0 * (a_well / max(m_bar, 1e-8)) * b_well * x * np.exp(-b_well * r2)
    if Mk is None:
        f_over_m_sol = np.zeros_like(x)
    else:
        x_p = V.T @ x                 # (k,)
        y_p = Mk @ x_p                # (k,)
        f_over_m_sol = (V @ y_p) / m_bar
    f_over_m = f_over_m_cons + f_over_m_sol
    v_new = (v + dt * f_over_m) / (1.0 + dt * gamma)
    x_new = x + dt * v_new
    return x_new, v_new


def integrate_helmholtz(
    x0: np.ndarray, v0: np.ndarray, m_bar_tok: float,
    well_params: Dict[int, Dict],
    V_per_layer: Dict[int, np.ndarray],
    Mk_per_layer: Optional[Dict[int, np.ndarray]],
    N_LAYERS: int, gamma: float, dt: float = 1.0,
    r2_gate: float = 0.02,
) -> np.ndarray:
    x, v = x0.copy(), v0.copy()
    traj = [x.copy()]
    for ell in range(1, N_LAYERS + 1):
        p = well_params[ell]
        if p["r2"] > r2_gate:
            a_w, b_w = p["a"], p["b"]
        else:
            a_w, b_w = 0.0, 0.0
        V = V_per_layer[ell]
        Mk = Mk_per_layer[ell] if (Mk_per_layer is not None and ell in Mk_per_layer) else None
        x, v = symplectic_step_helmholtz(x, v, a_w, b_w, V, Mk, m_bar_tok, gamma, dt)
        traj.append(x.copy())
    return np.stack(traj, axis=0)


def residuals_helmholtz(
    trajs: List[Trajectory],
    well_params: Dict[int, Dict],
    V_per_layer: Dict[int, np.ndarray],
    Mk_per_layer: Optional[Dict[int, np.ndarray]],
    gamma: float,
) -> np.ndarray:
    out = []
    for tr in trajs:
        mu_ps = tr.mu_ps[:, 0, :]
        T = tr.hs.shape[1]
        for ti in range(T):
            x0 = tr.x_ps[0, ti, :]
            v0 = tr.x_ps[1, ti, :] - tr.x_ps[0, ti, :]
            m = float(np.clip(tr.w[:, ti].mean(), 1e-3, None))
            pred_x = integrate_helmholtz(x0, v0, m, well_params, V_per_layer,
                                         Mk_per_layer, N_LAYERS, gamma)
            pred_h = pred_x + mu_ps
            obs_h = tr.hs[:, ti, :]
            denom = np.linalg.norm(obs_h, axis=1) + 1e-8
            out.append(np.linalg.norm(pred_h - obs_h, axis=1) / denom)
    return np.stack(out, axis=0)


# ---------------------------------------------------------------------------
# Main: sweep gamma, fit on TRAIN, evaluate on TRAIN + TEST
# ---------------------------------------------------------------------------
gammas = [0.0, 0.5, 1.0, 2.0, 5.0]

results: List[Dict] = []    # one entry per (config, gamma)

print(f"\n=== Fitting and evaluating (k = {cfg.pca_k} PCA dims/layer) ===")
for gamma in gammas:
    t_g = time.time()

    # Fit per-layer Omega (skew) and M_full (unconstrained) on TRAIN
    M_full_map: Dict[int, np.ndarray] = {}
    Omega_map:  Dict[int, np.ndarray] = {}
    per_layer_stats: Dict[int, Dict] = {}
    for ell in range(1, N_LAYERS):
        # Need x_{l-1}, x_l, x_{l+1} -- works for ell in 1..N_LAYERS-1
        x_tr, f_tr, m_tr = collect_samples(train_traj, ell, gamma)
        if x_tr.shape[0] < cfg.pca_k + 5:
            continue
        info = fit_linear_in_pca(
            x_tr, f_tr, m_tr, pca_V[ell], well_params_ps[ell],
            ridge=cfg.ridge_lambda,
        )
        M_full_map[ell] = info["M_full"]
        Omega_map[ell]  = info["Omega"]
        per_layer_stats[ell] = info

    # Baseline: no rotational term (matches extended_gamma_and_first_order.py)
    rho_train_base = residuals_helmholtz(train_traj, well_params_ps, pca_V, None, gamma)
    rho_test_base  = residuals_helmholtz(test_traj,  well_params_ps, pca_V, None, gamma)

    # With skew-symmetric Omega (true Helmholtz solenoidal linear)
    rho_train_omega = residuals_helmholtz(train_traj, well_params_ps, pca_V, Omega_map, gamma)
    rho_test_omega  = residuals_helmholtz(test_traj,  well_params_ps, pca_V, Omega_map, gamma)

    # With unconstrained M_full (upper bound)
    rho_train_mfull = residuals_helmholtz(train_traj, well_params_ps, pca_V, M_full_map, gamma)
    rho_test_mfull  = residuals_helmholtz(test_traj,  well_params_ps, pca_V, M_full_map, gamma)

    def med(r): return float(np.median(r[:, -1]))
    entry = {
        "gamma": gamma,
        "base_train":  med(rho_train_base),
        "omega_train": med(rho_train_omega),
        "mfull_train": med(rho_train_mfull),
        "base_test":   med(rho_test_base),
        "omega_test":  med(rho_test_omega),
        "mfull_test":  med(rho_test_mfull),
        "per_layer_train_omega_median": np.median(rho_train_omega, axis=0),
        "per_layer_test_omega_median":  np.median(rho_test_omega,  axis=0),
        "per_layer_test_base_median":   np.median(rho_test_base,   axis=0),
        "per_layer_test_mfull_median":  np.median(rho_test_mfull,  axis=0),
        "layers_fitted": sorted(Omega_map.keys()),
    }
    # Fraction of PCA-residual variance explained (mean across layers)
    if per_layer_stats:
        entry["mean_frac_ev_omega"] = float(np.mean(
            [s["frac_ev_Omega"]  for s in per_layer_stats.values()]
        ))
        entry["mean_frac_ev_mfull"] = float(np.mean(
            [s["frac_ev_M_full"] for s in per_layer_stats.values()]
        ))
    else:
        entry["mean_frac_ev_omega"] = 0.0
        entry["mean_frac_ev_mfull"] = 0.0

    results.append(entry)
    print(f"  gamma={gamma:<4}  "
          f"TRAIN: base={entry['base_train']:.4f} | "
          f"Omega={entry['omega_train']:.4f} | "
          f"Mfull={entry['mfull_train']:.4f}     "
          f"TEST: base={entry['base_test']:.4f} | "
          f"Omega={entry['omega_test']:.4f} | "
          f"Mfull={entry['mfull_test']:.4f}      "
          f"EV_Omega={entry['mean_frac_ev_omega']:.3f} "
          f"EV_Mfull={entry['mean_frac_ev_mfull']:.3f}     "
          f"({time.time()-t_g:.1f}s)")


# Also compute static-null baseline for reference
def residuals_static(trajs):
    out = []
    for tr in trajs:
        mu_ps = tr.mu_ps[:, 0, :]
        for ti in range(tr.hs.shape[1]):
            x0 = tr.x_ps[0, ti, :]
            pred_h = np.tile(x0, (N_LAYERS + 1, 1)) + mu_ps
            obs_h = tr.hs[:, ti, :]
            denom = np.linalg.norm(obs_h, axis=1) + 1e-8
            out.append(np.linalg.norm(pred_h - obs_h, axis=1) / denom)
    return np.stack(out, axis=0)

rho_static_train = residuals_static(train_traj)
rho_static_test  = residuals_static(test_traj)
static_train = float(np.median(rho_static_train[:, -1]))
static_test  = float(np.median(rho_static_test[:, -1]))
print(f"\nStatic-null baseline: TRAIN {static_train:.4f}   TEST {static_test:.4f}")


# ---------------------------------------------------------------------------
# Save results and make figures
# ---------------------------------------------------------------------------
gs = np.array([r["gamma"] for r in results])
save: Dict[str, np.ndarray] = {
    "gammas":               gs,
    "pca_k":                np.array([cfg.pca_k]),
    "n_train":              np.array([len(train_traj)]),
    "n_test":               np.array([len(test_traj)]),
    "base_train":           np.array([r["base_train"]   for r in results]),
    "omega_train":          np.array([r["omega_train"]  for r in results]),
    "mfull_train":          np.array([r["mfull_train"]  for r in results]),
    "base_test":            np.array([r["base_test"]    for r in results]),
    "omega_test":           np.array([r["omega_test"]   for r in results]),
    "mfull_test":           np.array([r["mfull_test"]   for r in results]),
    "static_train":         np.array([static_train]),
    "static_test":          np.array([static_test]),
    "mean_frac_ev_omega":   np.array([r["mean_frac_ev_omega"] for r in results]),
    "mean_frac_ev_mfull":   np.array([r["mean_frac_ev_mfull"] for r in results]),
}
# per-layer medians at each gamma (stacked)
save["per_layer_test_base_median"]  = np.stack([r["per_layer_test_base_median"]  for r in results], axis=0)
save["per_layer_test_omega_median"] = np.stack([r["per_layer_test_omega_median"] for r in results], axis=0)
save["per_layer_test_mfull_median"] = np.stack([r["per_layer_test_mfull_median"] for r in results], axis=0)
np.savez(os.path.join(RESULTS_DIR, "helmholtz_curl_results.npz"), **save)
print(f"Saved results to {os.path.join(RESULTS_DIR, 'helmholtz_curl_results.npz')}")


# --- figure A: median residual vs gamma, train/test, three configs --------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, split_name in zip(axes, ["train", "test"]):
    ax.plot(gs, [r[f"base_{split_name}"]  for r in results], marker="o",
            color="tab:blue",   label="gaussian only (no curl)")
    ax.plot(gs, [r[f"omega_{split_name}"] for r in results], marker="s",
            color="tab:green",  label=rf"+ skew $\Omega$ (k={cfg.pca_k})")
    ax.plot(gs, [r[f"mfull_{split_name}"] for r in results], marker="^",
            color="tab:red",    label=rf"+ full linear M (k={cfg.pca_k})")
    stat = static_train if split_name == "train" else static_test
    ax.axhline(stat, linestyle="--", color="tab:gray",
               label=f"static null ({stat:.3f})")
    ax.set_xlabel(r"damping $\gamma$")
    ax.set_title(f"median layer-L residual  --  {split_name.upper()}")
    ax.grid(True, alpha=0.3)
axes[0].set_ylabel("median residual")
axes[0].legend(fontsize=8)
axes[1].legend(fontsize=8)
fig.tight_layout()
fig_a = os.path.join(RESULTS_DIR, "fig_helmholtz_residual_vs_gamma.png")
fig.savefig(fig_a, dpi=130)
plt.close(fig)
print(f"Saved figure to {fig_a}")


# --- figure B: per-layer residual curves at best gamma on TEST set --------
best_idx = int(np.argmin(save["omega_test"]))
best_gamma = float(gs[best_idx])

fig, ax = plt.subplots(figsize=(8, 5))
layers = np.arange(N_LAYERS + 1)
ax.plot(layers, save["per_layer_test_base_median"][best_idx],  marker="o",
        color="tab:blue",  label="gaussian only (no curl)")
ax.plot(layers, save["per_layer_test_omega_median"][best_idx], marker="s",
        color="tab:green", label=rf"+ skew $\Omega$")
ax.plot(layers, save["per_layer_test_mfull_median"][best_idx], marker="^",
        color="tab:red",   label=rf"+ full linear M")
# Static null per layer
static_layer = np.median(rho_static_test, axis=0)
ax.plot(layers, static_layer, linestyle="--", color="tab:gray", label="static null")
ax.set_xlabel(r"layer $\ell$")
ax.set_ylabel("median relative residual")
ax.set_title(rf"Per-layer TEST residual  --  $\gamma^*={best_gamma}$")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig_b = os.path.join(RESULTS_DIR, "fig_helmholtz_residual_vs_layer_at_gamma_star.png")
fig.savefig(fig_b, dpi=130)
plt.close(fig)
print(f"Saved figure to {fig_b}")


# ---------------------------------------------------------------------------
# Summary Markdown (auto-populated)
# ---------------------------------------------------------------------------
md: List[str] = []
md.append("# Helmholtz-augmented Euler-Lagrange: does a linear solenoidal "
          "term fit the trajectories?\n")
md.append(f"Model: {cfg.model_name}   Train: {len(train_traj)} sentences, "
          f"{rho_static_train.shape[0]} tokens   "
          f"Test: {len(test_traj)} sentences, "
          f"{rho_static_test.shape[0]} tokens   "
          f"PCA k = {cfg.pca_k}   Ridge = {cfg.ridge_lambda}\n")

md.append("## 1. Question\n")
md.append(
    "Add a linear solenoidal (divergence-free) term to the damped "
    "second-order integrator:\n"
    "$$ m\\,\\ddot x = -\\nabla V(x) + V\\,\\Omega\\,V^\\top x "
    "- m\\,\\gamma\\,\\dot x,\\qquad \\Omega = -\\Omega^\\top,$$\n"
    "where $V \\in \\mathbb{R}^{d\\times k}$ is the per-layer top-$k$ "
    "PCA basis of TRAIN hidden states and $\\Omega \\in \\mathbb{R}^{k\\times k}$ "
    "is skew-symmetric.  Does this fit hidden-state trajectories "
    "on held-out TEST sentences?  As an upper bound we also fit an "
    "unconstrained linear operator $M \\in \\mathbb{R}^{k\\times k}$ to "
    "bracket how much of any improvement comes specifically from the "
    "skew / rotational sub-component.\n"
)

md.append("## 2. Train and test results\n")
md.append("Static-null baseline (predict $h_0$ at every layer):\n"
          f"- TRAIN median layer-L residual = **{static_train:.4f}**\n"
          f"- TEST  median layer-L residual = **{static_test:.4f}**\n")
md.append("\n### Main sweep")
md.append("| gamma | TRAIN gaussian | TRAIN +skew | TRAIN +full | "
          "TEST gaussian | TEST +skew | TEST +full | "
          "PCA-ev (skew) | PCA-ev (full) |")
md.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for r in results:
    md.append(
        f"| {r['gamma']} "
        f"| {r['base_train']:.4f} | {r['omega_train']:.4f} | {r['mfull_train']:.4f} "
        f"| {r['base_test']:.4f} | {r['omega_test']:.4f} | {r['mfull_test']:.4f} "
        f"| {r['mean_frac_ev_omega']:.3f} | {r['mean_frac_ev_mfull']:.3f} |"
    )

md.append("\n`PCA-ev` columns report the fraction of PCA-space residual "
          "variance that the fitted $\\Omega$ / $M$ explain at fit time "
          "(averaged across fitted layers). These are the *fit-quality* "
          "numbers, not the integrated-trajectory residuals.\n")

md.append(f"### Best-gamma slice (TEST, by $+$skew)\n")
md.append(f"$\\gamma^* = {best_gamma}$.\n")
md.append("| configuration | TEST median layer-L | change vs. static null |")
md.append("|---|---:|---:|")
def delta(x): return f"{(x - static_test):+.4f}"
md.append(f"| gaussian only | {save['base_test'][best_idx]:.4f} "
          f"| {delta(save['base_test'][best_idx])} |")
md.append(f"| + skew $\\Omega$ | {save['omega_test'][best_idx]:.4f} "
          f"| {delta(save['omega_test'][best_idx])} |")
md.append(f"| + full linear $M$ | {save['mfull_test'][best_idx]:.4f} "
          f"| {delta(save['mfull_test'][best_idx])} |")

md.append("\n## 3. Interpretation\n")
omega_improves = save["omega_test"][best_idx] < static_test - 0.005
mfull_improves = save["mfull_test"][best_idx] < static_test - 0.005
skew_vs_full = save["omega_test"][best_idx] - save["mfull_test"][best_idx]

if omega_improves:
    md.append(f"- **Skew-symmetric $\\Omega$ beats static null** on TEST by "
              f"{static_test - save['omega_test'][best_idx]:.4f}.  "
              "This is the first positive result from the Lagrangian programme "
              "once the scalar-only ansatz is dropped.")
else:
    md.append(f"- Skew-symmetric $\\Omega$ does not beat the static null on TEST "
              f"(difference: {delta(save['omega_test'][best_idx])}).  The "
              "improvement is confined to the TRAIN set and does not generalise.")

if mfull_improves:
    md.append(f"- **Unconstrained linear $M$ beats static null** on TEST by "
              f"{static_test - save['mfull_test'][best_idx]:.4f}.")
else:
    md.append(f"- Unconstrained linear $M$ also does not beat the static null on TEST "
              f"(difference: {delta(save['mfull_test'][best_idx])}).")

md.append(f"- Skew vs. full gap on TEST: {skew_vs_full:+.4f}.  "
          "A small gap means the rotational (skew) sub-component captures "
          "most of what any linear per-layer operator can; a large gap "
          "means the symmetric-linear part matters separately.")

md.append(f"- PCA fit quality: on TRAIN, $\\Omega$ explains "
          f"{save['mean_frac_ev_omega'][best_idx]*100:.1f}% of PCA-residual "
          "variance on average; $M$ explains "
          f"{save['mean_frac_ev_mfull'][best_idx]*100:.1f}%. "
          "The gap between fit quality and integrated-trajectory improvement "
          "is the classic failure mode of linearising non-linear layer-wise "
          "dynamics: the fit sees one-layer residuals, but the integrator "
          "compounds them across 12 layers.")

md.append(f"- Train/test generalisation: if TRAIN $\\Omega$ improvement "
          f"is much larger than TEST improvement, the per-layer "
          f"skew-symmetric operator is essentially memorising trajectories. "
          f"Compare columns TRAIN +skew vs. TEST +skew in the table above.")

md.append("\n## 4. Artefacts\n")
md.append("- `results/helmholtz_curl_results.npz`")
md.append("- `results/fig_helmholtz_residual_vs_gamma.png`")
md.append("- `results/fig_helmholtz_residual_vs_layer_at_gamma_star.png`")
md.append("- `notebooks/e_init/helmholtz_curl_augmented.py` -- this script")
md.append("\n## 5. Hyperparameters\n")
md.append(f"- `pca_k = {cfg.pca_k}` (top-$k$ per layer)")
md.append(f"- `ridge_lambda = {cfg.ridge_lambda}`")
md.append(f"- `n_test_per_domain = {cfg.n_test_per_domain}` "
          f"(10 test sentences total, 40 train)")
md.append(f"- `seed = {cfg.seed}`")

md_path = os.path.join(RESULTS_DIR, "helmholtz_curl_summary.md")
with open(md_path, "w") as f:
    f.write("\n".join(md))
print(f"Saved summary to {md_path}")

print("\nDone.")
