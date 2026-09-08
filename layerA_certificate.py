#!/usr/bin/env python3
"""Layer A noiseless certificate: exact piecewise-constant W vs closed form."""
from __future__ import annotations
import math
import numpy as np

def a_from_c(C, r0, tol=1e-14):
    if C <= 0:
        return 0.0
    xi = C / (4.0 * r0)
    a = min(0.5, math.sqrt(max(xi, 1e-16)))
    for _ in range(80):
        a = min(max(a, 1e-16), 1.0 - 1e-16)
        f = -math.log(1.0 - a) - a - xi
        df = a / (1.0 - a)
        a = min(max(a - f / df, 1e-16), 1.0 - 1e-16)
        if abs(f) < tol:
            break
    return a

def Cstar(R0):
    r0 = R0 - 1.0
    a_star = R0 / (2.0 * (R0 - 1.0))
    if a_star >= 1.0 - 1e-15:
        return math.inf, a_star, r0
    return 4.0 * r0 * (-math.log(1.0 - a_star) - a_star), a_star, r0

def u_closed_grid(t, R0, C):
    r0 = R0 - 1.0
    beta = R0
    Cst, a_star, _ = Cstar(R0)
    u = np.zeros_like(t)
    if R0 > 2.0 and C > Cst:
        T1 = (C - Cst) / (beta ** 2)
        tail = t >= T1
        u[~tail] = beta
        s = t[tail] - T1
        e = np.exp(-r0 * s)
        u[tail] = (2.0 * r0 * a_star * e) / (1.0 - a_star + a_star * e)
        info = dict(mode="bang", T1=T1, a=a_star, Cstar=Cst)
    else:
        a = a_from_c(C, r0)
        e = np.exp(-r0 * t)
        u = (2.0 * r0 * a * e) / (1.0 - a + a * e)
        info = dict(mode="free", T1=0.0, a=a, Cstar=Cst)
    return np.clip(u, 0.0, beta), info

def W_continuous(R0, C):
    r0 = R0 - 1.0
    Cst, a_star, _ = Cstar(R0)
    if R0 > 2.0 and C > Cst:
        T1 = (C - Cst) / (R0 ** 2)
        return math.exp(T1) * R0 / (R0 - 2.0) - 1.0
    a = a_from_c(C, r0)
    return 1.0 / (r0 * (1.0 - a))

def cell_int(u, r0, dt):
    du = u - r0
    if abs(du) < 1e-12:
        return dt
    return (math.exp(du * dt) - 1.0) / du

def W_and_grad(u, dt, R0):
    r0 = R0 - 1.0
    n = u.size
    integ = np.zeros(n)
    factor = np.zeros(n)
    psi = 0.0
    t = 0.0
    W = 0.0
    for i in range(n):
        pref = math.exp(-r0 * t + psi)
        ci = cell_int(float(u[i]), r0, dt)
        integ[i] = pref * ci
        W += integ[i]
        du = float(u[i]) - r0
        if abs(du) < 1e-12:
            factor[i] = 0.5 * dt * dt
        else:
            factor[i] = (dt * math.exp(du * dt) * du - (math.exp(du * dt) - 1.0)) / (du * du)
        psi += float(u[i]) * dt
        t += dt
    W_tail = math.exp(psi - r0 * t) / r0
    W += W_tail
    psi_pref = np.zeros(n)
    accp = 0.0
    for i in range(n):
        psi_pref[i] = accp
        accp += float(u[i]) * dt
    gu = np.zeros(n)
    later = W_tail
    for i in range(n - 1, -1, -1):
        pref = math.exp(-r0 * i * dt + psi_pref[i])
        gu[i] += pref * factor[i]
        gu[i] += later * dt
        later = later + integ[i]
    return W, gu

def project(u, dt, C, beta):
    u = np.clip(np.asarray(u, dtype=float), 0.0, beta)
    cap = beta * beta * u.size * dt
    target = min(C, 0.999999 * cap)
    for _ in range(16):
        e = float(np.dot(u, u) * dt)
        if e <= 1e-18:
            need = int(min(u.size, max(1, math.ceil(target / (beta * beta * dt)))))
            u[:need] = math.sqrt(target / (need * dt))
            u = np.clip(u, 0.0, beta)
            continue
        if abs(e - target) / max(target, 1e-18) < 1e-12:
            break
        u *= math.sqrt(target / e)
        u = np.clip(u, 0.0, beta)
    return u
