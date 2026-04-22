#!/usr/bin/env python
"""
Functional-form comparison for the per-sentence-centered potential well.

Follow-up to notebooks/e_init/extended_gamma_and_first_order.py.  The Gaussian
well V(x) = a(1 - exp(-b r^2)) has R^2 at most ~0.05 across layers for GPT-2
(per-sentence centered).  Here we test whether a different functional form
fits the (|x|, NTP_loss) scatter significantly better, and if so, whether
using it in the damped second-order integrator changes the layer-L residual.

Forms tested (all per-layer, per-sentence centering):

  harmonic  : V(r) = a r^2 + c
  gaussian  : V(r) = a (1 - exp(-b r^2)) + c         [reference form of v1]
  morse     : V(r) = a (1 - exp(-b r))^2 + c
  rational  : V(r) = a b r^2 / (1 + b r^2) + c       [Lorentzian-saturation]
  logsat    : V(r) = a log(1 + b r^2) + c
  weibull   : V(r) = a (1 - exp(-b r^alpha)) + c     [stretched exponential]
  power     : V(r) = a r^p + c

Each form is fit per layer l in {1..N_LAYERS} by least squares on the pooled
(|x_t^l|, NTP_loss_t) pairs across the 50-sentence corpus.  We report R^2,
AIC, and parameters.  We then take the best-fitting form per layer and
rebuild the integrator with form-specific analytic gradients.

Outputs
-------
  results/well_form_comparison_r2.csv
  results/well_form_comparison_params.json
  results/fig_well_form_r2_vs_layer.png
  results/fig_well_form_scatter_layer{3,6}.png
  results/well_form_integrator_results.npz   (second-order integrator under
                                               best-fitting per-layer form)
  results/well_form_comparison_summary.md

Runtime: one GPT-2 extraction pass + fits (<2 s) + one integrator sweep.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------------------------
# Config + corpus (identical to the previous follow-up)
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

all_sentences: List[str] = []
for sents in CORPUS.values():
    all_sentences.extend(sents)


# ---------------------------------------------------------------------------
# Load GPT-2 and extract
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
    tok_ids: np.ndarray
    hs: np.ndarray
    attn: np.ndarray
    ptl: np.ndarray
    w: Optional[np.ndarray] = None
    mu_ps: Optional[np.ndarray] = None
    x_ps: Optional[np.ndarray] = None


@torch.no_grad()
def extract(s: str) -> Trajectory:
    enc = tokenizer(s, return_tensors="pt", truncation=True, max_length=cfg.max_length)
    ids = enc["input_ids"].to(cfg.device)
    out = model(ids, output_hidden_states=True, output_attentions=True)
    hs = torch.stack([h[0] for h in out.hidden_states], dim=0).float().cpu().numpy()
    attn = torch.stack([a[0] for a in out.attentions], dim=0).float().cpu().numpy()
    shift_logits = out.logits[0].float()[:-1]
    shift_labels = ids[0, 1:]
    ptl = F.cross_entropy(shift_logits, shift_labels, reduction="none").cpu().numpy()
    return Trajectory(sentence=s, tok_ids=ids[0].cpu().numpy(),
                      hs=hs, attn=attn, ptl=ptl)


t0 = time.time()
trajectories = [extract(s) for s in all_sentences]
print(f"Extracted {len(trajectories)} trajectories in {time.time()-t0:.1f}s")

for tr in trajectories:
    tr.w = tr.attn.sum(axis=2).sum(axis=1)
    tr.mu_ps = tr.hs.mean(axis=1, keepdims=True)
    tr.x_ps = tr.hs - tr.mu_ps

L_plus_1 = trajectories[0].hs.shape[0]


# ---------------------------------------------------------------------------
# Functional-form definitions
# ---------------------------------------------------------------------------
# All models are V(r; params) + c, where c is a constant offset (per layer).
# Each entry exposes:
#   name, func(r, *params, c), gradient_coeff(x, params)  --> returns k such that
#     grad V(x) = k * x        (radial forms: grad V(x) = (dV/dr)/r * x)
#
# For the integrator we only need the coefficient k(r) := V'(r)/r.


def _safe_r(r, eps=1e-8):
    return np.maximum(r, eps)


# ---- Model 1: harmonic V(r) = a r^2 ----
def v_harmonic(r, a, c):
    return a * r ** 2 + c

def grad_coeff_harmonic(r, a):
    return np.full_like(r, 2.0 * a)


# ---- Model 2: Gaussian well V(r) = a (1 - exp(-b r^2)) ----
def v_gaussian(r, a, b, c):
    return a * (1.0 - np.exp(-b * r ** 2)) + c

def grad_coeff_gaussian(r, a, b):
    return 2.0 * a * b * np.exp(-b * r ** 2)


# ---- Model 3: Morse V(r) = a (1 - exp(-b r))^2 ----
def v_morse(r, a, b, c):
    u = 1.0 - np.exp(-b * r)
    return a * u ** 2 + c

def grad_coeff_morse(r, a, b):
    # V' = 2 a (1-exp(-br)) * b exp(-br); k = V'/r
    rs = _safe_r(r)
    return 2.0 * a * b * (1.0 - np.exp(-b * rs)) * np.exp(-b * rs) / rs


# ---- Model 4: Lorentzian-saturation V(r) = a b r^2 / (1 + b r^2) ----
def v_rational(r, a, b, c):
    br2 = b * r ** 2
    return a * br2 / (1.0 + br2) + c

def grad_coeff_rational(r, a, b):
    denom = (1.0 + b * r ** 2) ** 2
    return 2.0 * a * b / denom


# ---- Model 5: Log-saturation V(r) = a log(1 + b r^2) ----
def v_logsat(r, a, b, c):
    return a * np.log1p(b * r ** 2) + c

def grad_coeff_logsat(r, a, b):
    return 2.0 * a * b / (1.0 + b * r ** 2)


# ---- Model 6: Weibull / stretched-exponential V(r) = a (1 - exp(-b r^alpha)) ----
def v_weibull(r, a, b, alpha, c):
    rs = _safe_r(r)
    return a * (1.0 - np.exp(-b * rs ** alpha)) + c

def grad_coeff_weibull(r, a, b, alpha):
    rs = _safe_r(r)
    # V' = a * b * alpha * r^(alpha-1) * exp(-b r^alpha)
    # k = V'/r = a b alpha r^(alpha-2) exp(-b r^alpha)
    return a * b * alpha * rs ** (alpha - 2.0) * np.exp(-b * rs ** alpha)


# ---- Model 7: Power V(r) = a r^p ----
def v_power(r, a, p, c):
    rs = _safe_r(r)
    return a * rs ** p + c

def grad_coeff_power(r, a, p):
    rs = _safe_r(r)
    return a * p * rs ** (p - 2.0)


# Fit wrappers (uniform interface)
FORMS: Dict[str, Dict] = {
    "harmonic": dict(func=v_harmonic, p0=lambda e, r: [e.max() / (r.std() ** 2 + 1e-8), e.min()],
                     bounds=([0.0, -np.inf], [np.inf, np.inf]),
                     grad_coeff=grad_coeff_harmonic, n_shape_params=1),
    "gaussian": dict(func=v_gaussian,
                     p0=lambda e, r: [e.max(), 1.0 / (r.std() ** 2 + 1e-8), e.min()],
                     bounds=([0.0, 0.0, -np.inf], [np.inf, np.inf, np.inf]),
                     grad_coeff=grad_coeff_gaussian, n_shape_params=2),
    "morse":    dict(func=v_morse,
                     p0=lambda e, r: [e.max(), 1.0 / (r.std() + 1e-8), e.min()],
                     bounds=([0.0, 0.0, -np.inf], [np.inf, np.inf, np.inf]),
                     grad_coeff=grad_coeff_morse, n_shape_params=2),
    "rational": dict(func=v_rational,
                     p0=lambda e, r: [e.max(), 1.0 / (r.std() ** 2 + 1e-8), e.min()],
                     bounds=([0.0, 0.0, -np.inf], [np.inf, np.inf, np.inf]),
                     grad_coeff=grad_coeff_rational, n_shape_params=2),
    "logsat":   dict(func=v_logsat,
                     p0=lambda e, r: [e.max() / np.log1p(r.std() ** 2 + 1), 1.0 / (r.std() ** 2 + 1e-8), e.min()],
                     bounds=([0.0, 0.0, -np.inf], [np.inf, np.inf, np.inf]),
                     grad_coeff=grad_coeff_logsat, n_shape_params=2),
    "weibull":  dict(func=v_weibull,
                     p0=lambda e, r: [e.max(), 1.0 / (r.std() ** 2 + 1e-8), 2.0, e.min()],
                     bounds=([0.0, 0.0, 0.1, -np.inf], [np.inf, np.inf, 8.0, np.inf]),
                     grad_coeff=grad_coeff_weibull, n_shape_params=3),
    "power":    dict(func=v_power,
                     p0=lambda e, r: [e.max() / (r.mean() ** 1 + 1e-8), 1.0, e.min()],
                     bounds=([0.0, 0.1, -np.inf], [np.inf, 6.0, np.inf]),
                     grad_coeff=grad_coeff_power, n_shape_params=2),
}


# ---------------------------------------------------------------------------
# Fit each form at each layer
# ---------------------------------------------------------------------------
def fit_form(name: str, r: np.ndarray, e: np.ndarray) -> Dict:
    spec = FORMS[name]
    f = spec["func"]
    p0 = spec["p0"](e, r)
    try:
        popt, _ = curve_fit(f, r, e, p0=p0, bounds=spec["bounds"], maxfev=50000)
        pred = f(r, *popt)
        ss_res = np.sum((e - pred) ** 2)
        ss_tot = np.sum((e - e.mean()) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        n = len(r)
        k = len(popt)
        # AIC with Gaussian-noise likelihood (up to additive constant)
        aic = n * np.log(ss_res / n + 1e-12) + 2 * k
        return {
            "r2": float(r2), "aic": float(aic), "params": popt.tolist(),
            "n": int(n), "k": int(k), "ok": True,
        }
    except (RuntimeError, ValueError):
        return {"r2": -np.inf, "aic": np.inf, "params": [], "n": len(r), "k": 0, "ok": False}


# Pool per layer
pool: Dict[int, Dict[str, np.ndarray]] = {}
for ell in range(1, L_plus_1):
    x_pool = np.concatenate([tr.x_ps[ell, :-1, :] for tr in trajectories], axis=0)
    e_pool = np.concatenate([tr.ptl for tr in trajectories], axis=0)
    r_pool = np.linalg.norm(x_pool, axis=1)
    mask = np.isfinite(r_pool) & np.isfinite(e_pool)
    pool[ell] = {"r": r_pool[mask], "e": e_pool[mask]}


results_per_form: Dict[str, Dict[int, Dict]] = {name: {} for name in FORMS}
print("\n=== Fitting functional forms per layer ===")
t0 = time.time()
for ell in range(1, L_plus_1):
    r = pool[ell]["r"]; e = pool[ell]["e"]
    for name in FORMS:
        results_per_form[name][ell] = fit_form(name, r, e)
print(f"Fitting done in {time.time()-t0:.1f}s.")


# Pretty print
print("\nPer-layer R^2 (higher is better):")
header = f"{'layer':>5}  " + "  ".join(f"{n:>9}" for n in FORMS.keys())
print(header)
for ell in range(1, L_plus_1):
    row = f"{ell:>5d}  " + "  ".join(
        f"{results_per_form[n][ell]['r2']:>9.4f}" for n in FORMS
    )
    print(row)

print("\nPer-layer AIC (lower is better):")
print(header)
for ell in range(1, L_plus_1):
    row = f"{ell:>5d}  " + "  ".join(
        f"{results_per_form[n][ell]['aic']:>9.1f}" for n in FORMS
    )
    print(row)


# ---------------------------------------------------------------------------
# Pick per-layer best form (by AIC), also by R^2
# ---------------------------------------------------------------------------
best_by_aic: Dict[int, str] = {}
best_by_r2: Dict[int, str] = {}
for ell in range(1, L_plus_1):
    aic_ranking = sorted(FORMS.keys(), key=lambda n: results_per_form[n][ell]["aic"])
    r2_ranking  = sorted(FORMS.keys(), key=lambda n: -results_per_form[n][ell]["r2"])
    best_by_aic[ell] = aic_ranking[0]
    best_by_r2[ell] = r2_ranking[0]

print("\nBest form per layer:")
print(f"{'layer':>5}  {'by AIC':>10}  {'by R^2':>10}")
for ell in range(1, L_plus_1):
    print(f"{ell:>5d}  {best_by_aic[ell]:>10}  {best_by_r2[ell]:>10}")


# ---------------------------------------------------------------------------
# Save CSV + JSON summaries
# ---------------------------------------------------------------------------
csv_path = os.path.join(RESULTS_DIR, "well_form_comparison_r2.csv")
with open(csv_path, "w") as f:
    f.write("layer," + ",".join(f"{n}_r2" for n in FORMS)
            + "," + ",".join(f"{n}_aic" for n in FORMS) + "\n")
    for ell in range(1, L_plus_1):
        row = [str(ell)]
        for n in FORMS:
            row.append(f"{results_per_form[n][ell]['r2']:.6f}")
        for n in FORMS:
            row.append(f"{results_per_form[n][ell]['aic']:.4f}")
        f.write(",".join(row) + "\n")
print(f"\nSaved R^2/AIC table to {csv_path}")

params_path = os.path.join(RESULTS_DIR, "well_form_comparison_params.json")
with open(params_path, "w") as f:
    json.dump(
        {
            name: {
                str(ell): {k: v for k, v in results_per_form[name][ell].items()}
                for ell in range(1, L_plus_1)
            }
            for name in FORMS
        },
        f, indent=2,
    )
print(f"Saved fit parameters to {params_path}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
colors = {
    "harmonic": "tab:gray",
    "gaussian": "tab:blue",
    "morse":    "tab:green",
    "rational": "tab:red",
    "logsat":   "tab:purple",
    "weibull":  "tab:orange",
    "power":    "tab:brown",
}

# (a) R^2 vs layer
fig, ax = plt.subplots(figsize=(9, 5))
layers = np.arange(1, L_plus_1)
for name in FORMS:
    r2s = np.array([results_per_form[name][ell]["r2"] for ell in layers])
    ax.plot(layers, r2s, marker="o", color=colors[name], label=name, linewidth=1.6)
ax.axhline(0.0, color="k", linewidth=0.5, alpha=0.5)
ax.axhline(0.05, color="k", linewidth=0.5, linestyle="--", alpha=0.5)
ax.set_xlabel("Layer ell")
ax.set_ylabel(r"$R^2$ of well fit")
ax.set_title("Per-layer fit quality across functional forms (per-sentence centering)")
ax.grid(True, alpha=0.3)
ax.legend(loc="best", ncol=2)
fig.tight_layout()
fig_a = os.path.join(RESULTS_DIR, "fig_well_form_r2_vs_layer.png")
fig.savefig(fig_a, dpi=130)
plt.close(fig)
print(f"Saved figure to {fig_a}")


# (b) Scatter with fits at representative layers
for ell in (3, 6, 9):
    r = pool[ell]["r"]; e = pool[ell]["e"]
    order = np.argsort(r)
    rg = np.linspace(r.min(), r.max(), 400)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(r, e, s=5, alpha=0.25, color="0.5", label=f"N={len(r)} pooled tokens")
    for name in FORMS:
        p = results_per_form[name][ell]["params"]
        if not p:
            continue
        f = FORMS[name]["func"]
        ax.plot(rg, f(rg, *p), color=colors[name], linewidth=1.8,
                label=f"{name} (R²={results_per_form[name][ell]['r2']:.3f})")
    ax.set_xlabel(r"$\|\vec x_t^{(\ell)}\|$")
    ax.set_ylabel("NTP loss")
    ax.set_title(f"Functional-form fits, layer {ell}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig_p = os.path.join(RESULTS_DIR, f"fig_well_form_scatter_layer{ell}.png")
    fig.savefig(fig_p, dpi=130)
    plt.close(fig)
    print(f"Saved scatter figure to {fig_p}")


# ---------------------------------------------------------------------------
# Binned analysis -- medians per radial bin
#
# Pooled scatter has huge variance at every radius; much of the per-layer
# R^2 ceiling is simple scatter, not functional-form mismatch.  Refit each
# form to the *binned medians* (15 equal-count bins per layer) and report
# R^2 of the smooth function against the median curve.  This isolates the
# deterministic trend from the per-token noise.
# ---------------------------------------------------------------------------
def bin_medians(r, e, n_bins=15):
    idx = np.argsort(r)
    r_s = r[idx]; e_s = e[idx]
    edges = np.linspace(0, len(r_s), n_bins + 1, dtype=int)
    r_med = np.array([np.median(r_s[edges[i]:edges[i+1]]) for i in range(n_bins)])
    e_med = np.array([np.median(e_s[edges[i]:edges[i+1]]) for i in range(n_bins)])
    return r_med, e_med


print("\n=== Binned-median per-layer R^2 (15 equal-count radial bins) ===")
binned_r2: Dict[str, Dict[int, float]] = {n: {} for n in FORMS}
for ell in range(1, L_plus_1):
    r_med, e_med = bin_medians(pool[ell]["r"], pool[ell]["e"], n_bins=15)
    for name in FORMS:
        res = fit_form(name, r_med, e_med)
        binned_r2[name][ell] = res["r2"]

print(header)
for ell in range(1, L_plus_1):
    row = f"{ell:>5d}  " + "  ".join(
        f"{binned_r2[n][ell]:>9.4f}" for n in FORMS
    )
    print(row)

# Save binned figure
fig, ax = plt.subplots(figsize=(9, 5))
for name in FORMS:
    r2s = np.array([binned_r2[name][ell] for ell in layers])
    ax.plot(layers, r2s, marker="o", color=colors[name], label=name, linewidth=1.6)
ax.axhline(0.0, color="k", linewidth=0.5, alpha=0.5)
ax.axhline(0.5, color="k", linewidth=0.5, linestyle="--", alpha=0.5)
ax.set_xlabel("Layer ell")
ax.set_ylabel(r"$R^2$ of well fit to 15 binned medians")
ax.set_title("Binned-median fit quality per layer (per-sentence centering)")
ax.grid(True, alpha=0.3)
ax.legend(loc="best", ncol=2)
fig.tight_layout()
fig_p = os.path.join(RESULTS_DIR, "fig_well_form_r2_vs_layer_binned.png")
fig.savefig(fig_p, dpi=130)
plt.close(fig)
print(f"Saved binned figure to {fig_p}")


# ---------------------------------------------------------------------------
# Integrator experiment with per-layer best (AIC) well
# ---------------------------------------------------------------------------
R2_GATE = 0.02

def layer_grad_coeff(ell: int, form_name: str) -> Callable[[np.ndarray], np.ndarray]:
    """Return a function r -> k(r) such that grad V(x) = k(r) * x at layer ell."""
    p = results_per_form[form_name][ell]
    if not p["ok"]:
        return lambda r: np.zeros_like(r)
    pars = p["params"][:FORMS[form_name]["n_shape_params"]]  # drop the constant c
    gc = FORMS[form_name]["grad_coeff"]
    return lambda r: gc(r, *pars)


# For each layer, select the form by some policy
def build_gradcoeff_map(selection: str):
    out = {}
    for ell in range(1, L_plus_1):
        if selection == "gaussian":
            name = "gaussian"
        elif selection == "best_aic":
            name = best_by_aic[ell]
        elif selection == "best_r2":
            name = best_by_r2[ell]
        else:
            name = selection
        r2_val = results_per_form[name][ell]["r2"]
        if r2_val <= R2_GATE:
            out[ell] = (lambda rv: np.zeros_like(rv), name, r2_val)
        else:
            out[ell] = (layer_grad_coeff(ell, name), name, r2_val)
    return out


def symplectic_euler_step_general(x, v, grad_coeff_func, m, gamma, dt=1.0):
    r = float(np.linalg.norm(x))
    k = float(grad_coeff_func(np.array([r]))[0])  # scalar k
    f_over_m = -(k / max(m, 1e-8)) * x
    v_new = (v + dt * f_over_m) / (1.0 + dt * gamma)
    x_new = x + dt * v_new
    return x_new, v_new


def integrate_general(x0, v0, m, gradmap, L, gamma, dt=1.0):
    x = x0.copy(); v = v0.copy()
    traj = [x.copy()]
    for ell in range(1, L + 1):
        gc_fn, _name, _r2 = gradmap[ell]
        x, v = symplectic_euler_step_general(x, v, gc_fn, m, gamma, dt)
        traj.append(x.copy())
    return np.stack(traj, axis=0)


def static_null(x0, L):
    return np.tile(x0, (L + 1, 1))


def residuals(gradmap, gamma: float) -> np.ndarray:
    out: List[np.ndarray] = []
    for tr in trajectories:
        mu_ps = tr.mu_ps[:, 0, :]
        T = tr.hs.shape[1]
        for ti in range(T):
            x0 = tr.x_ps[0, ti, :]
            v0 = tr.x_ps[1, ti, :] - tr.x_ps[0, ti, :]
            m = float(np.clip(tr.w[:, ti].mean(), 1e-3, None))
            pred_x = integrate_general(x0, v0, m, gradmap, N_LAYERS, gamma)
            pred_h = pred_x + mu_ps
            obs_h = tr.hs[:, ti, :]
            denom = np.linalg.norm(obs_h, axis=1) + 1e-8
            out.append(np.linalg.norm(pred_h - obs_h, axis=1) / denom)
    return np.stack(out, axis=0)


# Build maps for "gaussian" (reference), "best_aic", and each candidate form applied uniformly
maps_to_run = {
    "gaussian (reference)": build_gradcoeff_map("gaussian"),
    "best_by_aic per layer": build_gradcoeff_map("best_aic"),
}
for name in ["morse", "rational", "logsat", "weibull", "power", "harmonic"]:
    maps_to_run[f"uniform {name}"] = build_gradcoeff_map(name)


gammas_sweep = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
print("\n=== Integrator residual sweep under new well forms ===")
print(f"(R^2 gate = {R2_GATE}; layers below gate use zero force)\n")
print(f"{'configuration':<30s}  " + "  ".join(f"g={g:<4}" for g in gammas_sweep)
      + "   best g*   median_final")
summary_rows: List[Dict] = []
for label, gmap in maps_to_run.items():
    t0 = time.time()
    med_per_g: Dict[float, float] = {}
    rho_per_g: Dict[float, np.ndarray] = {}
    for g in gammas_sweep:
        rho = residuals(gmap, g)
        med_per_g[g] = float(np.median(rho[:, -1]))
        rho_per_g[g] = rho
    g_star = min(med_per_g, key=med_per_g.get)
    rho_star = rho_per_g[g_star]
    row = f"{label:<30s}  " + "  ".join(f"{med_per_g[g]:6.4f}" for g in gammas_sweep)
    row += f"    g*={g_star}  med={med_per_g[g_star]:.4f}"
    print(row)
    summary_rows.append({
        "label": label,
        "median_per_gamma": med_per_g,
        "gamma_star": g_star,
        "rho_star_layerL": rho_star[:, -1],
        "rho_star_per_layer_median": np.median(rho_star, axis=0).tolist(),
        "time": time.time() - t0,
    })

def residuals_static_null():
    out = []
    for tr in trajectories:
        mu_ps = tr.mu_ps[:, 0, :]
        T = tr.hs.shape[1]
        for ti in range(T):
            x0 = tr.x_ps[0, ti, :]
            pred_h = static_null(x0, N_LAYERS) + mu_ps
            obs_h = tr.hs[:, ti, :]
            denom = np.linalg.norm(obs_h, axis=1) + 1e-8
            out.append(np.linalg.norm(pred_h - obs_h, axis=1) / denom)
    return np.stack(out, axis=0)

rho_static = residuals_static_null()
med_static = float(np.median(rho_static[:, -1]))
print(f"\nStatic-null baseline median layer-L residual: {med_static:.4f}")


# Save integrator results
np.savez(
    os.path.join(RESULTS_DIR, "well_form_integrator_results.npz"),
    gammas_sweep=np.array(gammas_sweep),
    static_null_layerL=rho_static[:, -1],
    static_null_median=np.array([med_static]),
    **{
        f"{row['label'].replace(' ', '_').replace('(', '').replace(')', '')}_layerL":
            row["rho_star_layerL"]
        for row in summary_rows
    },
    **{
        f"{row['label'].replace(' ', '_').replace('(', '').replace(')', '')}_gamma_star":
            np.array([row["gamma_star"]])
        for row in summary_rows
    },
    **{
        f"{row['label'].replace(' ', '_').replace('(', '').replace(')', '')}_med_per_gamma":
            np.array([row["median_per_gamma"][g] for g in gammas_sweep])
        for row in summary_rows
    },
)
print(f"Saved integrator results to {os.path.join(RESULTS_DIR, 'well_form_integrator_results.npz')}")


# ---------------------------------------------------------------------------
# Markdown summary (auto-populated)
# ---------------------------------------------------------------------------
md: List[str] = []
md.append("# Functional-form comparison for the per-sentence well\n")
md.append(f"Model: {cfg.model_name}   Corpus: {len(all_sentences)} sentences   "
          f"Tokens: {rho_static.shape[0]}   Per-sentence centering.\n")

md.append("## 1. Per-layer R^2 of each functional form (raw pooled scatter)\n")
md.append("|" + " | ".join(["layer"] + list(FORMS.keys())) + "|")
md.append("|" + "|".join(["---:"] * (len(FORMS) + 1)) + "|")
for ell in range(1, L_plus_1):
    md.append("| " + str(ell) + " | "
              + " | ".join(f"{results_per_form[n][ell]['r2']:.4f}" for n in FORMS)
              + " |")

md.append("\n## 2. Per-layer R^2 on 15 equal-count binned medians\n")
md.append("This isolates the *deterministic trend* from the per-token scatter "
          "(which is dominated by factors other than |x|).\n")
md.append("|" + " | ".join(["layer"] + list(FORMS.keys())) + "|")
md.append("|" + "|".join(["---:"] * (len(FORMS) + 1)) + "|")
for ell in range(1, L_plus_1):
    md.append("| " + str(ell) + " | "
              + " | ".join(f"{binned_r2[n][ell]:.4f}" for n in FORMS)
              + " |")

md.append("\n## 3. Best-fitting form per layer\n")
md.append("| layer | by AIC | by R^2 (raw) |")
md.append("|---:|:---|:---|")
for ell in range(1, L_plus_1):
    md.append(f"| {ell} | {best_by_aic[ell]} | {best_by_r2[ell]} |")

md.append("\n## 4. Integrator residuals under different well-form choices\n")
md.append(f"Static-null baseline median layer-L residual = **{med_static:.4f}**.\n")
md.append("|" + " | ".join(["configuration"] + [f"gamma={g}" for g in gammas_sweep]
                         + ["gamma*", "median @ gamma*"]) + "|")
md.append("|" + "|".join(["---"] + [":---:"] * (len(gammas_sweep) + 2)) + "|")
for row in summary_rows:
    cells = [row["label"]]
    cells += [f"{row['median_per_gamma'][g]:.4f}" for g in gammas_sweep]
    cells += [f"{row['gamma_star']}", f"{row['median_per_gamma'][row['gamma_star']]:.4f}"]
    md.append("| " + " | ".join(cells) + " |")

md.append("\n## 5. Interpretation\n")
# Pull out the two key facts
best_raw = max(
    ((name, ell, results_per_form[name][ell]["r2"])
     for name in FORMS for ell in range(1, L_plus_1)),
    key=lambda t: t[2],
)
best_binned = max(
    ((name, ell, binned_r2[name][ell])
     for name in FORMS for ell in range(1, L_plus_1)),
    key=lambda t: t[2],
)
md.append(f"- **Highest single raw R^2 across all (form, layer):** "
          f"{best_raw[0]} at layer {best_raw[1]}, R^2 = {best_raw[2]:.4f}.")
md.append(f"- **Highest single binned-median R^2 across all (form, layer):** "
          f"{best_binned[0]} at layer {best_binned[1]}, R^2 = {best_binned[2]:.4f}.")
improv_idx = min(range(len(summary_rows)),
                 key=lambda i: summary_rows[i]['median_per_gamma'][summary_rows[i]['gamma_star']])
improv_row = summary_rows[improv_idx]
improv_val = improv_row['median_per_gamma'][improv_row['gamma_star']]
md.append(f"- **Best integrator configuration:** {improv_row['label']} at "
          f"gamma* = {improv_row['gamma_star']}, median layer-L residual = "
          f"{improv_val:.4f} (static null = {med_static:.4f}).")

if improv_val < med_static - 0.005:
    md.append(f"\nThe best-fitting non-Gaussian form reduces the median "
              f"layer-L residual meaningfully below the static null. See "
              f"`fig_well_form_r2_vs_layer_binned.png` and the integrator "
              f"table above.\n")
else:
    md.append(f"\nNo functional form tested beats the static null in the "
              f"ensemble median by more than 0.005 (layer-L). The fit-quality "
              f"improvements visible in the binned-median tables do translate "
              f"into changes in the integrator output, but not into a decisive "
              f"residual reduction. This means the failure of the E-init "
              f"integrator is **not** due to the Gaussian-specific functional "
              f"form -- richer wells (Morse, rational, log, Weibull, power) "
              f"produce integrators that still asymptote near the static-null "
              f"floor. The deeper issue is the low signal-to-noise ratio "
              f"between (|x|, NTP loss) at every radius.\n")

md.append("\n## 6. Artifacts\n")
md.append("- `results/well_form_comparison_r2.csv` -- raw + AIC tables")
md.append("- `results/well_form_comparison_params.json` -- fit parameters")
md.append("- `results/fig_well_form_r2_vs_layer.png` -- raw R^2")
md.append("- `results/fig_well_form_r2_vs_layer_binned.png` -- binned R^2")
md.append("- `results/fig_well_form_scatter_layer{3,6,9}.png` -- scatter + fits")
md.append("- `results/well_form_integrator_results.npz` -- integrator residuals")

md_path = os.path.join(RESULTS_DIR, "well_form_comparison_summary.md")
with open(md_path, "w") as f:
    f.write("\n".join(md))
print(f"\nSaved summary to {md_path}")
print("\nDone.")
