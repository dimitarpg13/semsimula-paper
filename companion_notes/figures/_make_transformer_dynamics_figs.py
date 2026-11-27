"""Figures for transformer/Transformers_as_Dynamical_Systems.md (2026-11-27).

Every figure is computed from the model equations quoted in the note; nothing is read from a trained model. Writes into
figures/transformer/:
  tf_splitting.png        a layer as one step of a split integrator: Lie-Trotter (attention, then feed-forward) against Strang
                          (the Macaron layer of Lu et al.) on a linear system whose two parts do not commute; the paths and the
                          global error against the step size (slopes 1 and 2)
  tf_sphere_clusters.png  the self-attention dynamics (SA) of Geshkovski et al. on the sphere, Q = K = V = I, beta = 9:
                          snapshots of 48 tokens
  tf_metastability.png    (SA) for several beta: the number of clusters against time, and the interaction energy E_beta
  tf_polytope.png         the dynamics without the sphere, V = I, in the plane, rescaled by exp(-t): the tokens settle on a few
                          points, vertices of a convex polytope (the "leaders")
  tf_rank_collapse.png    distance of the token matrix from rank one along depth: pure attention, attention with skip
                          connections, and with skip connections and a feed-forward block (random weights)
  tf_hopfield.png         the modern Hopfield energy in the plane with four stored patterns, and one attention update from a
                          grid of queries
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "figure.dpi": 150})
INK, MUTED, GREY = "#1F2937", "#6B7280", "#9CA3AF"
C = ["#3B82F6", "#DC2626", "#059669", "#F59E0B", "#7C3AED"]          # fixed categorical order
OUT = Path(__file__).resolve().parent / "transformer"; OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(0)


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white"); plt.close(fig); print("wrote", name)


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)


# 1. Splitting: dx/dt = (A + B) x, A the "interaction" part, B the "per-token" part, [A, B] != 0
A = np.array([[0.0, -1.0], [1.0, 0.0]]); B = np.array([[0.5, 0.0], [0.0, -0.5]]); x0 = np.array([1.0, 0.0]); T = 3.0
def lie(h, n):
    E1, E2 = expm(h * A), expm(h * B); xs = [x0]
    for _ in range(n): xs.append(E2 @ (E1 @ xs[-1]))
    return np.array(xs)
def strang(h, n):
    Eh, E1 = expm(0.5 * h * B), expm(h * A); xs = [x0]
    for _ in range(n): xs.append(Eh @ (E1 @ (Eh @ xs[-1])))
    return np.array(xs)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
ts = np.linspace(0, T, 400); ex = np.array([expm(t * (A + B)) @ x0 for t in ts])
a1.plot(ex[:, 0], ex[:, 1], color=INK, lw=2, label="exact flow")
for f, c, lab in ((lie, C[0], "Lie-Trotter (Transformer layer)"), (strang, C[1], "Strang (Macaron layer)")):
    p = f(T / 6, 6); a1.plot(p[:, 0], p[:, 1], "o-", color=c, ms=6, lw=1.5, label=lab + ", 6 steps")
a1.set_aspect("equal"); a1.set_title("one layer = one step of a split integrator", loc="left"); a1.legend(frameon=False, fontsize=9); clean(a1)
a1.set_xlabel("first coordinate"); a1.set_ylabel("second coordinate")
hs = T / np.array([3, 6, 12, 24, 48, 96, 192]); xT = expm(T * (A + B)) @ x0
for f, c, lab in ((lie, C[0], "Lie-Trotter"), (strang, C[1], "Strang")):
    err = [np.linalg.norm(f(h, int(round(T / h)))[-1] - xT) for h in hs]; a2.loglog(hs, err, "o-", color=c, lw=1.5, ms=6, label=lab)
e1, e2 = (np.linalg.norm(f(hs[-1], int(round(T / hs[-1])))[-1] - xT) for f in (lie, strang))
a2.loglog(hs, 1.6 * e1 * hs / hs[-1], ls="--", color=GREY, lw=1); a2.loglog(hs, 0.6 * e2 * (hs / hs[-1]) ** 2, ls=":", color=GREY, lw=1)
a2.text(hs[2], 1.6 * e1 * hs[2] / hs[-1] * 1.25, "slope 1", color=MUTED, fontsize=9); a2.text(hs[2], 0.6 * e2 * (hs[2] / hs[-1]) ** 2 * 0.4, "slope 2", color=MUTED, fontsize=9)
a2.set_xlabel("step size (depth per layer)"); a2.set_ylabel("error at the final time"); a2.set_title("global error: first against second order", loc="left")
a2.legend(frameon=False); clean(a2)
fig.tight_layout(); save(fig, "tf_splitting.png")


# 2, 3. (SA) on the sphere, Q = K = V = I
def sa_run(X, beta, T, dt=0.02, every=None, normalised=True):
    """Euler steps of (SA) (or (USA)), renormalised to the sphere after each step: the discrete layers of the note's equation."""
    out, ts = [X.copy()], [0.0]; n = len(X)
    for k in range(int(T / dt)):
        G = X @ X.T; W = np.exp(beta * (G - G.max(1, keepdims=True)))
        W = W / W.sum(1, keepdims=True) if normalised else np.exp(beta * G) / n
        F = W @ X; F -= (F * X).sum(1, keepdims=True) * X
        X = X + dt * F; X /= np.linalg.norm(X, axis=1, keepdims=True)
        if every and (k + 1) % every == 0: out.append(X.copy()); ts.append((k + 1) * dt)
    return (np.array(out), np.array(ts)) if every else X
def clusters(X, tol=0.05):
    n, lab = len(X), -np.ones(len(X), int); c = 0
    for i in range(n):
        if lab[i] < 0:
            lab[(lab < 0) & (np.linalg.norm(X - X[i], axis=1) < tol)] = c; c += 1
    return c
def energy(X, beta):
    return float(np.exp(beta * (X @ X.T)).sum() / (2 * beta * len(X) ** 2))

n = 48; X0 = rng.normal(size=(n, 3)); X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
traj, tt = sa_run(X0, 9.0, 60.0, every=5)
snaps = [0.0, 2.0, 10.0, 60.0]
fig = plt.figure(figsize=(15, 4.2)); u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
for j, ts_ in enumerate(snaps):
    ax = fig.add_subplot(1, 4, j + 1, projection="3d"); X = traj[np.argmin(abs(tt - ts_))]
    ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), color=GREY, lw=0.3, alpha=0.4)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=22, color=C[0], edgecolor="white", linewidth=0.5, depthshade=False)
    ax.set_title(f"t = {ts_:g}: {clusters(X)} groups", loc="left", fontsize=11); ax.set_axis_off(); ax.set_box_aspect((1, 1, 1))
fig.suptitle(r"self-attention dynamics on the sphere, $Q = K = V = I$, $\beta = 9$, 48 tokens", x=0.02, ha="left", fontsize=12, fontweight="bold")
fig.tight_layout(); save(fig, "tf_sphere_clusters.png")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
for c, beta in zip(C, (1.0, 4.0, 9.0, 16.0)):
    tr, t_ = sa_run(X0, beta, 400.0, dt=0.05, every=4)
    a1.plot(t_[1:], [clusters(x) for x in tr[1:]], color=c, lw=1.6, label=rf"$\beta$ = {beta:g}")
    e = np.array([energy(x, beta) for x in tr]); a2.plot(t_[1:], (e[1:] - e[0]) / (np.exp(beta) / (2 * beta) - e[0]), color=c, lw=1.6, label=rf"$\beta$ = {beta:g}")
a1.set_xscale("log"); a1.set_yscale("log"); a1.set_xlabel("time (depth)"); a1.set_ylabel("number of clusters"); a1.set_title("fast clustering, then long plateaus", loc="left"); a1.legend(frameon=False); clean(a1)
a2.set_xscale("log"); a2.set_xlabel("time (depth)"); a2.set_ylabel("share of the energy gap to one cluster"); a2.set_title(r"the interaction energy $E_\beta$ rises along the flow", loc="left"); clean(a2)
fig.tight_layout(); save(fig, "tf_metastability.png")


# 4. V = I in the plane, no sphere, rescaled z = exp(-t) x: dz/dt = P(t) z - z, P = softmax(exp(2t) <z_i, z_j>)
Z = rng.normal(size=(24, 2)) * [1.0, 0.6]; Z0 = Z.copy(); dt = 0.01; path = [Z.copy()]
for k in range(int(6 / dt)):
    t = k * dt; S = np.exp(2 * t) * (Z @ Z.T); W = np.exp(S - S.max(1, keepdims=True)); W /= W.sum(1, keepdims=True)
    Z = Z + dt * (W @ Z - Z); path.append(Z.copy())
path = np.array(path)
from scipy.spatial import ConvexHull
hull = ConvexHull(Z0); fig, ax = plt.subplots(figsize=(6.4, 5.6))
for i in range(len(Z0)): ax.plot(path[:, i, 0], path[:, i, 1], color=C[0], lw=0.8, alpha=0.7)
ax.scatter(Z0[:, 0], Z0[:, 1], s=18, color=GREY, zorder=3, label="tokens at t = 0")
ax.scatter(path[-1, :, 0], path[-1, :, 1], s=40, color=C[1], zorder=4, label="tokens at t = 6 (rescaled)")
hv = list(hull.vertices) + [hull.vertices[0]]; ax.plot(Z0[hv, 0], Z0[hv, 1], color=INK, lw=1, ls="--", label="convex hull at t = 0 (for reference)")
ax.set_aspect("equal"); ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2); clean(ax)
ax.set_title(r"$V = I$: the rescaled tokens settle on a few leaders", loc="left"); ax.set_xlabel("first coordinate"); ax.set_ylabel("second coordinate")
save(fig, "tf_polytope.png")


# 5. Rank collapse along depth (random weights, n = 32 tokens, d = 64)
n, d, L = 32, 64, 12
def res(X):
    return np.linalg.norm(X - X.mean(0, keepdims=True)) / np.linalg.norm(X)
def run(kind, seed):
    r = np.random.default_rng(seed); X = r.normal(size=(n, d)); out = [res(X)]
    for _ in range(L):
        Q, K, V = (r.normal(size=(d, d)) / np.sqrt(d) for _ in range(3))
        Y = X / np.linalg.norm(X, axis=1, keepdims=True) * np.sqrt(d)                       # LayerNorm before the block
        S = (Y @ Q) @ (Y @ K).T / np.sqrt(d); P = np.exp(S - S.max(1, keepdims=True)); P /= P.sum(1, keepdims=True); A = P @ (Y @ V)
        if kind == "attention only": X = A
        else:
            X = X + A
            if kind == "skip and feed-forward":
                W1, W2 = r.normal(size=(d, 4 * d)) / np.sqrt(d), r.normal(size=(4 * d, d)) / np.sqrt(4 * d)
                Y = X / np.linalg.norm(X, axis=1, keepdims=True) * np.sqrt(d); X = X + np.maximum(Y @ W1, 0) @ W2
        out.append(res(X))
    return out
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for c, kind in zip(C, ("attention only", "skip connections", "skip and feed-forward")):
    R = np.array([run(kind, s) for s in range(20)]); m = np.exp(np.log(np.maximum(R, 1e-300)).mean(0))
    ax.semilogy(range(L + 1), m, "o-", color=c, lw=1.6, ms=5, label=kind)
ax.set_xlabel("layer"); ax.set_ylabel("distance from rank one (relative)"); ax.set_title("pure attention collapses the tokens", loc="left")
ax.legend(frameon=False); clean(ax); save(fig, "tf_rank_collapse.png")


# 6. The modern Hopfield energy and one attention update
P = np.array([[1.2, 0.4], [-0.9, 1.0], [-0.6, -1.1], [1.0, -0.9]]); beta = 3.0
def E(xi):
    s = beta * xi @ P.T; m = s.max(-1, keepdims=True)
    return -(m[..., 0] + np.log(np.exp(s - m).sum(-1))) / beta + 0.5 * (xi ** 2).sum(-1)
def upd(xi):
    s = beta * xi @ P.T; w = np.exp(s - s.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True); return w @ P
g = np.linspace(-2, 2, 200); GX, GY = np.meshgrid(g, g); XI = np.stack([GX, GY], -1)
fig, ax = plt.subplots(figsize=(6.4, 5.8)); cs = ax.contourf(GX, GY, E(XI), levels=30, cmap="Blues_r"); fig.colorbar(cs, ax=ax, label="energy")
q = np.stack(np.meshgrid(np.linspace(-1.8, 1.8, 9), np.linspace(-1.8, 1.8, 9)), -1).reshape(-1, 2); qn = upd(q)
ax.quiver(q[:, 0], q[:, 1], qn[:, 0] - q[:, 0], qn[:, 1] - q[:, 1], angles="xy", scale_units="xy", scale=1, color=INK, width=0.004, alpha=0.8)
ax.scatter(P[:, 0], P[:, 1], s=90, marker="*", color=C[3], edgecolor=INK, zorder=4, label="stored patterns (keys = values)")
ax.set_aspect("equal"); ax.legend(frameon=False, fontsize=9, loc="lower left"); ax.set_title("one attention update = one Hopfield step", loc="left")
ax.set_xlabel("first coordinate"); ax.set_ylabel("second coordinate"); save(fig, "tf_hopfield.png")
