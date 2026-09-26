#!/usr/bin/env python3
"""Calibrate the damped-geodesic residual diagnostic on a KNOWN geodesic.

Run this before trusting any R-bar number. It integrates exact damped
Newtonian dynamics  x'' = -grad V - gamma x'  in a smooth bounded potential
(a geodesic of the Jacobi metric, by Jacobi's theorem) and feeds the
trajectory to the residual exactly as the gamma-sweep notebooks compute it
(`conformal_grad` and `christoffel_vv` copied verbatim), four ways:

    as coded      a + Gamma(v,v) + gamma_step v,   E fixed at layer 0
    + inst E      E_l = |x'_l|^2/2 + V(x_l) per layer
    + reparam     + (grad V . v)/(E - V) v   (Jacobi geodesics are Newtonian
                  trajectories only up to reparametrisation; in layer time
                  the geodesic equation carries this extra term along v)
    both

What a working detector prints on this trajectory: ~0 (up to integrator
error, shrinking with dt).  What the notebooks' residual prints: ~1.

Findings recorded 2026-09-26 (companion note
Geodesic_Preservation_Experiment.md SS3.1):
  * small dt: as coded 0.6-0.95, both fixes 0.01-0.03 and shrinking with dt
  * the sweeps' regime (dt = 1 per layer, omega*dt ~ 1): as coded 0.84-1.05,
    and the least-squares gamma_geo comes out 0.75-0.93 REGARDLESS of the
    true damping (0.05 / 0.1 / 0.3) -- the "universal gamma_geo ~ 0.9" of the
    OpenWebText sweeps is an artefact of the diagnostic, not a property of
    the models.  Even with both fixes R stays 0.66-0.95 there: at one step
    per period a second difference in layer index is not a derivative, so
    no residual of a CONTINUOUS-time equation can be small.  That is why the
    CfC+BAOAB programme measures geodesicity by replaying the model's own
    discrete integrator instead (E1, Cell 6b-9).
"""
import math
import torch

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
D, B = 16, 4096


def make_potential(kappa, K=8, sig2=2.0):
    """Confining bowl + Gaussian bumps near the origin, so every trajectory
    feels a force (a bumps-only potential in 16-D leaves random starts in
    free flight and the test degenerates)."""
    mu = torch.randn(K, D) * 0.7
    w = torch.rand(K) * 0.5 + 0.5

    def V(x):
        diff = x[:, None, :] - mu[None]
        return 0.5 * kappa * (x ** 2).sum(-1) - (w[None] * torch.exp(-(diff ** 2).sum(-1) / (2 * sig2))).sum(-1)

    def gradV(x):
        x = x.detach().requires_grad_(True)
        return torch.autograd.grad(V(x).sum(), x)[0]

    return V, gradV


def integrate(V, gradV, gamma, dt, steps):
    """Velocity Verlet for x'' = -grad V - gamma x'. Returns positions and velocities."""
    x = torch.randn(B, D) * 1.5
    xd = torch.randn(B, D) * 0.7
    xs, xds = [x.clone()], [xd.clone()]
    f = -gradV(x) - gamma * xd
    for _ in range(steps):
        x = x + dt * xd + 0.5 * dt * dt * f
        f_new = -gradV(x) - gamma * (xd + 0.5 * dt * f)
        xd = xd + 0.5 * dt * (f + f_new)
        f = f_new
        xs.append(x.clone()); xds.append(xd.clone())
    return xs, xds


# ---- verbatim from colab_fock_gamma_sweep_geodesic_*.ipynb -------------------
def conformal_grad(grad_V, E_minus_V, epsilon=1e-6):
    denom = 2.0 * E_minus_V.unsqueeze(-1).clamp(min=epsilon)
    return -grad_V / denom


def christoffel_vv(phi_grad, v):
    phi_dot_v = (phi_grad * v).sum(dim=-1, keepdim=True)
    v_sq = (v * v).sum(dim=-1, keepdim=True)
    return 2.0 * phi_dot_v * v - v_sq * phi_grad
# -----------------------------------------------------------------------------


def residual(V, gradV, xs, xds, gamma, dt, inst_E=False, reparam=False):
    """R-bar in the notebooks' convention (v = x_l - x_{l-1}, a = v_{l+1} - v_l),
    plus the least-squares gamma_geo, in physical units."""
    E0 = 0.5 * (xds[0] ** 2).sum(-1) + V(xs[0])
    Rs, gnum, gden = [], 0.0, 0.0
    for l in range(1, len(xs) - 1):
        v = xs[l] - xs[l - 1]
        a = (xs[l + 1] - xs[l]) - v
        g = gradV(xs[l]); Vl = V(xs[l])
        E = (0.5 * (xds[l] ** 2).sum(-1) + Vl) if inst_E else E0
        EmV = E - Vl
        Gamma = christoffel_vv(conformal_grad(g, EmV), v)
        res = a + Gamma + (gamma * dt) * v
        if reparam:
            res = res + ((g * v).sum(-1, keepdim=True) / EmV.clamp(min=1e-6)[:, None]) * v
        ok = EmV > 1e-6
        Rs.append((res.norm(dim=-1) / (a.norm(dim=-1) + 1e-12))[ok].mean().item())
        ag = a + Gamma
        gnum += -((ag * v).sum(-1))[ok].sum().item()
        gden += (v ** 2).sum(-1)[ok].sum().item()
    return sum(Rs) / len(Rs), (gnum / gden) / dt


def main():
    print("Residual diagnostic on a KNOWN geodesic (exact damped Newton). A working detector reads ~0.\n")
    print("-- small dt (derivatives well approximated) --")
    print(f"{'gamma':>6} {'dt':>6} | {'as coded':>9} {'+instE':>8} {'+reparam':>9} {'both':>8} | {'gamma_geo':>10}")
    V, gradV = make_potential(kappa=0.5)
    for gamma in (0.0, 0.1, 0.3):
        for dt in (0.1, 0.05):
            xs, xds = integrate(V, gradV, gamma, dt, int(20 / dt))
            r0, gg = residual(V, gradV, xs, xds, gamma, dt)
            r1, _ = residual(V, gradV, xs, xds, gamma, dt, inst_E=True)
            r2, _ = residual(V, gradV, xs, xds, gamma, dt, reparam=True)
            r3, _ = residual(V, gradV, xs, xds, gamma, dt, inst_E=True, reparam=True)
            print(f"{gamma:>6.2f} {dt:>6.2f} | {r0:>9.3f} {r1:>8.3f} {r2:>9.3f} {r3:>8.4f} | {gg:>10.3f}")
    print("\n-- the OpenWebText sweeps' regime: dt = 1 per layer, L = 16, omega*dt ~ 1 --")
    print(f"{'omega*dt':>9} {'gamma':>6} | {'as coded':>9} {'both':>8} | {'gamma_geo':>10}")
    for kappa in (0.1, 0.5, 1.0):
        V, gradV = make_potential(kappa=kappa)
        for gamma in (0.05, 0.1, 0.3):
            xs, xds = integrate(V, gradV, gamma, 1.0, 16)
            r0, gg = residual(V, gradV, xs, xds, gamma, 1.0)
            r3, _ = residual(V, gradV, xs, xds, gamma, 1.0, inst_E=True, reparam=True)
            print(f"{math.sqrt(kappa):>9.2f} {gamma:>6.2f} | {r0:>9.3f} {r3:>8.3f} | {gg:>10.3f}")


if __name__ == "__main__":
    main()
