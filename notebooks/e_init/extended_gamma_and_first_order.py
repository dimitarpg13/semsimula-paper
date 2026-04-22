#!/usr/bin/env python
"""
Extended-gamma and first-order overdamped follow-up for the E-init test.

This is a post-v1 companion experiment (targeted at paper v2).  Two follow-up
studies are performed on top of the original E-init per-sentence-centered run:

  (A) Extended gamma sweep for the damped second-order Euler-Lagrange
      integrator from notebooks/e_init/e_init_validation.ipynb.
      We extend the existing gammas {0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0}
      with {2.0, 5.0, 10.0, 50.0} and report median layer-L residuals.

  (B) Pure first-order (overdamped) gradient descent integrator using the same
      per-sentence Gaussian-well parameters.  The update is
         x_{t+1} = x_t - eta * grad V(x_t),
         grad V(x) = 2 a b x exp(-b r^2),
      and eta is swept over several values.  No velocity variable, no mass.
      This is the eta = dt/(m*gamma) limit of the damped second-order system.

Outputs
-------
  results/extended_gamma_first_order_results.npz
  results/fig_extended_gamma.png
  results/fig_first_order_eta.png
  results/extended_gamma_first_order_summary.md

Runtime: a single GPT-2 extraction pass on the 50-sentence corpus (~1363 tokens
with valid NTP targets) dominates; on MPS this is a few minutes.  All 12
damped runs and ~8 first-order runs reuse the cached trajectories.
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
# Config
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
print(f"model {cfg.model_name}   results_dir {RESULTS_DIR}")


# ---------------------------------------------------------------------------
# Corpus (identical to notebooks/e_init/e_init_validation.ipynb)
# ---------------------------------------------------------------------------
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
all_domains: List[str] = []
for domain, sents in CORPUS.items():
    for s in sents:
        all_sentences.append(s)
        all_domains.append(domain)
print(f"Corpus: {len(all_sentences)} sentences")


# ---------------------------------------------------------------------------
# Stage 1 -- Load model and extract hidden states + attention + PTL
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    cfg.model_name,
    torch_dtype=cfg.dtype,
    attn_implementation="eager",
).to(cfg.device)
model.eval()
N_LAYERS = model.config.num_hidden_layers
D_HIDDEN = model.config.hidden_size
print(f"Loaded {cfg.model_name}: L={N_LAYERS}, d={D_HIDDEN}")


@dataclass
class Trajectory:
    sentence: str
    domain: str
    tok_ids: np.ndarray
    hs: np.ndarray       # (L+1, T, d)
    attn: np.ndarray     # (L, H, T, T)
    ptl: np.ndarray      # (T-1,)
    w: Optional[np.ndarray] = None      # (L, T)
    mu_ps: Optional[np.ndarray] = None  # (L+1, 1, d)
    x_ps: Optional[np.ndarray] = None   # (L+1, T, d)


@torch.no_grad()
def extract(sentence: str, domain: str) -> Trajectory:
    enc = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=cfg.max_length)
    ids = enc["input_ids"].to(cfg.device)
    out = model(ids, output_hidden_states=True, output_attentions=True)
    hs = torch.stack([h[0] for h in out.hidden_states], dim=0).float().cpu().numpy()
    attn = torch.stack([a[0] for a in out.attentions], dim=0).float().cpu().numpy()
    logits = out.logits[0].float()
    shift_logits = logits[:-1]
    shift_labels = ids[0, 1:]
    ptl = F.cross_entropy(shift_logits, shift_labels, reduction="none").cpu().numpy()
    return Trajectory(
        sentence=sentence, domain=domain,
        tok_ids=ids[0].cpu().numpy(),
        hs=hs, attn=attn, ptl=ptl,
    )


t0 = time.time()
trajectories: List[Trajectory] = []
for i, (s, d) in enumerate(zip(all_sentences, all_domains)):
    trajectories.append(extract(s, d))
    if (i + 1) % 10 == 0:
        print(f"  extracted {i+1}/{len(all_sentences)}")
print(f"Extracted {len(trajectories)} trajectories in {time.time()-t0:.1f}s")


# ---------------------------------------------------------------------------
# Stage 2 -- Derived quantities: attention mass, per-sentence centering
# ---------------------------------------------------------------------------
def attention_mass(attn: np.ndarray) -> np.ndarray:
    return attn.sum(axis=2).sum(axis=1)

for tr in trajectories:
    tr.w = attention_mass(tr.attn)
    tr.mu_ps = tr.hs.mean(axis=1, keepdims=True)  # (L+1, 1, d)
    tr.x_ps = tr.hs - tr.mu_ps                    # (L+1, T, d)

L_plus_1 = trajectories[0].hs.shape[0]
print(f"Layers: {L_plus_1}  (expect N_LAYERS+1 = {N_LAYERS+1})")


# ---------------------------------------------------------------------------
# Stage 3 -- Per-sentence Gaussian well fit (matches well_params_ps.json)
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
        ss_res = np.sum((e - pred) ** 2)
        ss_tot = np.sum((e - e.mean()) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        return {"a": float(popt[0]), "b": float(popt[1]), "r2": float(r2), "n": len(d)}
    except RuntimeError:
        return {"a": 0.0, "b": 0.0, "r2": -np.inf, "n": len(d)}


well_params_ps: Dict[int, Dict] = {}
for ell in range(1, L_plus_1):
    x_pool = np.concatenate([tr.x_ps[ell, :-1, :] for tr in trajectories], axis=0)
    e_pool = np.concatenate([tr.ptl for tr in trajectories], axis=0)
    well_params_ps[ell] = fit_well_for_layer(x_pool, e_pool)

print(f"{'layer':>6} {'a':>10} {'b':>12} {'R^2':>10} {'N':>8}")
for ell, p in well_params_ps.items():
    print(f"{ell:>6d} {p['a']:>10.3f} {p['b']:>12.6f} {p['r2']:>10.4f} {p['n']:>8d}")


# ---------------------------------------------------------------------------
# Stage 4 -- Integrators
# ---------------------------------------------------------------------------
R2_GATE = 0.05


def symplectic_euler_step(x, v, a_ell, b_ell, m, gamma, dt=1.0):
    r2 = float(np.dot(x, x))
    f_over_m = -2.0 * (a_ell / max(m, 1e-8)) * b_ell * x * np.exp(-b_ell * r2)
    v_new = (v + dt * f_over_m) / (1.0 + dt * gamma)
    x_new = x + dt * v_new
    return x_new, v_new


def integrate_second_order(x0, v0, m, well_params, L, gamma, dt=1.0):
    x = x0.copy()
    v = v0.copy()
    traj = [x.copy()]
    for ell in range(1, L + 1):
        p = well_params[ell]
        if p["r2"] > R2_GATE:
            a_ell, b_ell = p["a"], p["b"]
        else:
            a_ell, b_ell = 0.0, 0.0
        x, v = symplectic_euler_step(x, v, a_ell, b_ell, m, gamma, dt)
        traj.append(x.copy())
    return np.stack(traj, axis=0)


def integrate_first_order(x0, well_params, L, eta, dt=1.0):
    """Pure first-order overdamped gradient flow.

    x_{t+1} = x_t - eta * dt * grad V(x_t),  grad V = 2 a b x exp(-b r^2).

    No velocity, no mass.  This is the formal m*gamma -> infty limit of
    the damped second-order system with eta := 1 / (m*gamma).
    """
    x = x0.copy()
    traj = [x.copy()]
    for ell in range(1, L + 1):
        p = well_params[ell]
        if p["r2"] > R2_GATE:
            a_ell, b_ell = p["a"], p["b"]
        else:
            a_ell, b_ell = 0.0, 0.0
        r2 = float(np.dot(x, x))
        grad = 2.0 * a_ell * b_ell * x * np.exp(-b_ell * r2)
        x = x - eta * dt * grad
        traj.append(x.copy())
    return np.stack(traj, axis=0)


def static_null(x0, L):
    return np.tile(x0, (L + 1, 1))


# ---------------------------------------------------------------------------
# Stage 5 -- Residual calculators
# ---------------------------------------------------------------------------
def residuals_second_order(gamma: float) -> np.ndarray:
    """Per-token per-layer relative residuals for damped 2nd-order integrator
    (per-sentence centering).  Returns (N_tokens, L+1) array.
    """
    out: List[np.ndarray] = []
    for tr in trajectories:
        mu_ps = tr.mu_ps[:, 0, :]
        T = tr.hs.shape[1]
        for ti in range(T):
            x0 = tr.x_ps[0, ti, :]
            v0 = tr.x_ps[1, ti, :] - tr.x_ps[0, ti, :]
            m = float(np.clip(tr.w[:, ti].mean(), 1e-3, None))
            pred_x = integrate_second_order(x0, v0, m, well_params_ps, N_LAYERS, gamma)
            pred_h = pred_x + mu_ps
            obs_h = tr.hs[:, ti, :]
            denom = np.linalg.norm(obs_h, axis=1) + 1e-8
            out.append(np.linalg.norm(pred_h - obs_h, axis=1) / denom)
    return np.stack(out, axis=0)


def residuals_first_order(eta: float) -> np.ndarray:
    out: List[np.ndarray] = []
    for tr in trajectories:
        mu_ps = tr.mu_ps[:, 0, :]
        T = tr.hs.shape[1]
        for ti in range(T):
            x0 = tr.x_ps[0, ti, :]
            pred_x = integrate_first_order(x0, well_params_ps, N_LAYERS, eta)
            pred_h = pred_x + mu_ps
            obs_h = tr.hs[:, ti, :]
            denom = np.linalg.norm(obs_h, axis=1) + 1e-8
            out.append(np.linalg.norm(pred_h - obs_h, axis=1) / denom)
    return np.stack(out, axis=0)


def residuals_static() -> np.ndarray:
    out: List[np.ndarray] = []
    for tr in trajectories:
        mu_ps = tr.mu_ps[:, 0, :]
        T = tr.hs.shape[1]
        for ti in range(T):
            x0 = tr.x_ps[0, ti, :]
            pred_x = static_null(x0, N_LAYERS)
            pred_h = pred_x + mu_ps
            obs_h = tr.hs[:, ti, :]
            denom = np.linalg.norm(obs_h, axis=1) + 1e-8
            out.append(np.linalg.norm(pred_h - obs_h, axis=1) / denom)
    return np.stack(out, axis=0)


# ---------------------------------------------------------------------------
# Stage 6 -- Run Experiment A: extended gamma sweep
# ---------------------------------------------------------------------------
gammas_full = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0, 50.0]
print("\n=== Experiment A: damped 2nd-order gamma sweep ===")
rho_second: Dict[float, np.ndarray] = {}
median_layerL_second: Dict[float, float] = {}
t_a = time.time()
for g in gammas_full:
    t_g = time.time()
    rho_second[g] = residuals_second_order(g)
    median_layerL_second[g] = float(np.median(rho_second[g][:, -1]))
    print(f"  gamma={g:<5}  median layer-L residual = {median_layerL_second[g]:.4f}"
          f"   ({time.time()-t_g:.1f}s)")
print(f"Experiment A total: {time.time()-t_a:.1f}s")

gamma_star_full = min(median_layerL_second, key=median_layerL_second.get)
print(f"gamma* (over extended sweep) = {gamma_star_full}  "
      f"-> residual {median_layerL_second[gamma_star_full]:.4f}")


# ---------------------------------------------------------------------------
# Stage 7 -- Run Experiment B: pure first-order gradient flow eta sweep
# ---------------------------------------------------------------------------
etas = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0]
print("\n=== Experiment B: pure 1st-order (overdamped) eta sweep ===")
rho_first: Dict[float, np.ndarray] = {}
median_layerL_first: Dict[float, float] = {}
t_b = time.time()
for eta in etas:
    t_e = time.time()
    rho_first[eta] = residuals_first_order(eta)
    median_layerL_first[eta] = float(np.median(rho_first[eta][:, -1]))
    print(f"  eta={eta:<6}  median layer-L residual = {median_layerL_first[eta]:.4f}"
          f"   ({time.time()-t_e:.1f}s)")
print(f"Experiment B total: {time.time()-t_b:.1f}s")

eta_star = min(median_layerL_first, key=median_layerL_first.get)
print(f"eta* = {eta_star}  -> residual {median_layerL_first[eta_star]:.4f}")


# ---------------------------------------------------------------------------
# Stage 8 -- Static null baseline
# ---------------------------------------------------------------------------
rho_static = residuals_static()
median_static_layerL = float(np.median(rho_static[:, -1]))
print(f"\nStatic null baseline (ps): median layer-L residual = {median_static_layerL:.4f}")


# ---------------------------------------------------------------------------
# Stage 9 -- Fraction-beats-null analysis
# ---------------------------------------------------------------------------
def frac_beats(rho_model: np.ndarray, rho_base: np.ndarray) -> float:
    return float(np.mean(rho_model[:, -1] < rho_base[:, -1]))


print("\n--- Fraction of tokens where model beats static null at layer L ---")
print(f"  2nd-order at gamma*={gamma_star_full}: "
      f"{frac_beats(rho_second[gamma_star_full], rho_static)*100:.2f}%")
print(f"  1st-order at eta*={eta_star}: "
      f"{frac_beats(rho_first[eta_star], rho_static)*100:.2f}%")


# ---------------------------------------------------------------------------
# Stage 10 -- Save raw results
# ---------------------------------------------------------------------------
save = {
    "gammas_full": np.array(gammas_full, dtype=np.float64),
    "median_layerL_second": np.array([median_layerL_second[g] for g in gammas_full]),
    "etas": np.array(etas, dtype=np.float64),
    "median_layerL_first": np.array([median_layerL_first[e] for e in etas]),
    "median_static_layerL": np.array([median_static_layerL]),
    "gamma_star_full": np.array([gamma_star_full]),
    "eta_star": np.array([eta_star]),
    "rho_static_layerL": rho_static[:, -1],
    "rho_second_star_layerL": rho_second[gamma_star_full][:, -1],
    "rho_first_star_layerL": rho_first[eta_star][:, -1],
}
# Also keep per-layer curves at gamma* and eta* (useful for figures)
save["rho_second_star_all_layers_median"] = np.median(rho_second[gamma_star_full], axis=0)
save["rho_first_star_all_layers_median"] = np.median(rho_first[eta_star], axis=0)
save["rho_static_all_layers_median"] = np.median(rho_static, axis=0)

out_npz = os.path.join(RESULTS_DIR, "extended_gamma_first_order_results.npz")
np.savez(out_npz, **save)
print(f"\nSaved raw results to {out_npz}")


# ---------------------------------------------------------------------------
# Stage 11 -- Figures
# ---------------------------------------------------------------------------
# (a) gamma sweep
fig, ax = plt.subplots(figsize=(7.5, 4.5))
gs = np.array(gammas_full)
rs = np.array([median_layerL_second[g] for g in gammas_full])
ax.plot(gs, rs, marker="o", color="tab:blue",
        label="damped 2nd-order (E-init, ps)")
ax.axhline(median_static_layerL, linestyle="--", color="tab:gray",
           label=f"static null (median = {median_static_layerL:.3f})")
ax.set_xscale("symlog", linthresh=0.1)
ax.set_xlabel(r"damping $\gamma$")
ax.set_ylabel("median layer-L relative residual")
ax.set_title("Extended gamma sweep: damped 2nd-order Euler-Lagrange")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig_a = os.path.join(RESULTS_DIR, "fig_extended_gamma.png")
fig.savefig(fig_a, dpi=130)
plt.close(fig)
print(f"Saved figure to {fig_a}")

# (b) eta sweep for first-order
fig, ax = plt.subplots(figsize=(7.5, 4.5))
es = np.array(etas)
rs = np.array([median_layerL_first[e] for e in etas])
ax.plot(es, rs, marker="s", color="tab:red",
        label="pure 1st-order (overdamped)")
ax.axhline(median_static_layerL, linestyle="--", color="tab:gray",
           label=f"static null (median = {median_static_layerL:.3f})")
ax.axhline(median_layerL_second[gamma_star_full], linestyle=":", color="tab:blue",
           label=rf"2nd-order best ($\gamma^*={gamma_star_full}$)")
ax.set_xscale("log")
ax.set_xlabel(r"first-order step size $\eta$")
ax.set_ylabel("median layer-L relative residual")
ax.set_title("First-order (overdamped) eta sweep")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig_b = os.path.join(RESULTS_DIR, "fig_first_order_eta.png")
fig.savefig(fig_b, dpi=130)
plt.close(fig)
print(f"Saved figure to {fig_b}")


# ---------------------------------------------------------------------------
# Stage 12 -- Markdown summary
# ---------------------------------------------------------------------------
md_lines: List[str] = []
md_lines.append("# Extended gamma sweep and pure first-order follow-up to E-init\n")
md_lines.append("_Companion experiment targeted at paper v2.  Generated by "
                "`notebooks/e_init/extended_gamma_and_first_order.py`._\n")
md_lines.append(f"**Model:** {cfg.model_name}   **Corpus:** "
                f"{len(all_sentences)} sentences   "
                f"**Tokens scored:** {rho_static.shape[0]}\n")
md_lines.append("## Question\n")
md_lines.append(
    "Does either of the following Lagrangian-derived integrators reproduce "
    "the observed GPT-2 hidden-state trajectories, at the layer-L level, "
    "once its free parameter is tuned?\n"
    "- (A) Damped 2nd-order Euler-Lagrange with damping `gamma`\n"
    "- (B) Pure 1st-order (overdamped) gradient flow with step size `eta`\n"
)
md_lines.append("In both cases the forcing is the per-sentence-centered "
                "Gaussian well fitted layer-by-layer (`well_params_ps`).\n")

md_lines.append("## Static-null baseline (reference)\n")
md_lines.append(f"Median layer-L residual of the static-null predictor "
                f"(predict `h_0 + mu_ps` at all layers): "
                f"**{median_static_layerL:.4f}**.\n")

md_lines.append("## (A) Damped 2nd-order: extended gamma sweep\n")
md_lines.append("| gamma | median layer-L residual |\n|---:|---:|")
for g in gammas_full:
    md_lines.append(f"| {g} | {median_layerL_second[g]:.4f} |")
md_lines.append(f"\n**gamma\\*** (over extended sweep) = {gamma_star_full}, "
                f"median residual = {median_layerL_second[gamma_star_full]:.4f}.\n")
frac_2 = frac_beats(rho_second[gamma_star_full], rho_static)
ratio_2 = median_layerL_second[gamma_star_full] / median_static_layerL
md_lines.append(f"Fraction of tokens where 2nd-order at gamma\\* beats the "
                f"static null at layer L: **{frac_2*100:.2f}%**   "
                f"(ratio E-init/static = {ratio_2:.3f}).\n")

md_lines.append("## (B) Pure 1st-order (overdamped): eta sweep\n")
md_lines.append("| eta | median layer-L residual |\n|---:|---:|")
for e in etas:
    md_lines.append(f"| {e} | {median_layerL_first[e]:.4f} |")
md_lines.append(f"\n**eta\\*** = {eta_star}, median residual = "
                f"{median_layerL_first[eta_star]:.4f}.\n")
frac_1 = frac_beats(rho_first[eta_star], rho_static)
ratio_1 = median_layerL_first[eta_star] / median_static_layerL
md_lines.append(f"Fraction of tokens where 1st-order at eta\\* beats the "
                f"static null at layer L: **{frac_1*100:.2f}%**   "
                f"(ratio 1st-order/static = {ratio_1:.3f}).\n")

md_lines.append("## Interpretation\n")
if (median_layerL_second[gamma_star_full] < median_static_layerL
        or median_layerL_first[eta_star] < median_static_layerL):
    md_lines.append("At least one of the two integrators, when tuned, improves on the "
                    "static null at layer L.  See the tables and figures for details; "
                    "conclusions about whether this is strong enough to support a "
                    "Lagrangian interpretation require further validation.\n")
else:
    md_lines.append("Neither integrator, tuned on its free parameter over the "
                    "ranges explored, improves on the static null at layer L.  "
                    "Under the Gaussian-well + per-sentence-centering setting "
                    "used here, this is a negative result for both the damped "
                    "2nd-order and pure 1st-order overdamped readings.\n")
md_lines.append("Figures: `results/fig_extended_gamma.png`, "
                "`results/fig_first_order_eta.png`.\n")

md_path = os.path.join(RESULTS_DIR, "extended_gamma_first_order_summary.md")
with open(md_path, "w") as f:
    f.write("\n".join(md_lines))
print(f"Saved summary to {md_path}")

print("\nDone.")
