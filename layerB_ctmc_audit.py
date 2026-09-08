#!/usr/bin/env python3
"""Layer B: CTMC audit of already fixed profiles. No optimization."""

from __future__ import annotations

import math
import numpy as np
from layerA_certificate import (
    Cstar, u_closed_grid, project, W_and_grad, W_continuous
)


def best_rectangle(t, dt, R0, C):
    beta = R0
    best_u, best_W, best_T = None, -1.0, None
    for Tpulse in np.linspace(dt, min(24.0, t[-1] * 0.8), 16):
        n = max(1, int(round(Tpulse / dt)))
        u = np.zeros_like(t)
        height = math.sqrt(C / (n * dt))
        u[:n] = min(height, beta)
        u = project(u, dt, C, beta)
        W, _ = W_and_grad(u, dt, R0)
        if W > best_W:
            best_u, best_W, best_T = u, W, n * dt
    return best_u, best_W, best_T


def simulate_one(I0, R0, u, dt, rng, n_cap=20000):
    """CTMC on the control grid; exact homogeneous tail (gamma/beta)^n."""
    gamma = 1.0
    beta = R0
    n = int(I0)
    if n <= 0:
        return 1.0
    t = 0.0
    K = u.size
    Tgrid = K * dt
    while n > 0 and t < Tgrid - 1e-14:
        k = min(int(math.floor(t / dt + 1e-12)), K - 1)
        t_end = min((k + 1) * dt, Tgrid)
        if t_end <= t + 1e-14:
            t = min(t_end + 1e-12, Tgrid)
            continue
        lam = max(beta - float(u[k]), 0.0)
        rate = n * (lam + gamma)
        wait = rng.exponential(1.0 / rate)
        if t + wait >= t_end:
            t = t_end
            continue
        t = t + wait
        if rng.random() < gamma / (lam + gamma):
            n -= 1
        else:
            n += 1
            if n > n_cap:
                return 0.0
    if n <= 0:
        return 1.0
    return (1.0 / R0) ** n


def mc_q(I0, R0, u, dt, n_paths, rng):
    acc = 0
    for _ in range(n_paths):
        acc += simulate_one(I0, R0, u, dt, rng)
    q = acc / n_paths
    se = math.sqrt(max(q * (1.0 - q), 1e-16) / n_paths)
    return q, se


def main():
    rng = np.random.default_rng(2026)
    dt = 0.05
    T = 24.0
    t = np.arange(0.0, T, dt)
    n_paths = 2000
    I0s = (1,)

    cases = []
    cases.append(("R0=1.5 C=1 free", 1.5, 1.0))
    Cst25, _, _ = Cstar(2.5)
    cases.append(("R0=2.5 0.99C*", 2.5, 0.99 * Cst25))
    cases.append(("R0=2.5 1.01C*", 2.5, 1.01 * Cst25))
    Cst3, _, _ = Cstar(3.0)
    cases.append(("R0=3 2C*", 3.0, 2.0 * Cst3))

    print(f"Layer B  dt={dt}  paths={n_paths}  Tgrid={T}")
    for tag, R0, C in cases:
        u_star, info = u_closed_grid(t, R0, C)
        u_star = project(u_star, dt, C, R0)
        u_rect, Wrect, Trect = best_rectangle(t, dt, R0, C)
        r0 = R0 - 1.0
        u_exp = project(np.exp(-r0 * t), dt, C, R0)
        for name, u in [("u*", u_star), ("rect", u_rect), ("exp", u_exp)]:
            W, _ = W_and_grad(u, dt, R0)
            qW = (W / (1.0 + W))
            qmc, se = mc_q(1, R0, u, dt, n_paths, rng)
            z = (qmc - qW) / se if se > 0 else 0.0
            lo, hi = qmc - 1.96 * se, qmc + 1.96 * se
            cover = "yes" if lo <= qW <= hi else "NO"
            print(f"{tag:<22} {name:<8} {qW:9.4f} {qmc:9.4f} {se:8.4f} {z:7.2f} {cover}")


if __name__ == "__main__":
    main()
