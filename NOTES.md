# Working notes: L2-optimal intervention in a linear birth-death epidemic

**Date:** 8 September 2026  
**Deposit:** working notes for reuse, not a sole-author theorem.

## Disclosure

This draft was assembled in dialogue between a human operator and several
large language models. The human steered questions and rejected wrong
branches. Derivations were not produced as an unaided human proof.
Do not cite this as a conventional single-author research article.
Reuse is permitted with attribution to this repository and date.

Text: CC BY 4.0. Code: MIT.

## Problem

Linear birth-death (branching) approximation of an early epidemic,
`I << N`. Per-infective birth rate `λ(t)=β-u(t)`, death rate `μ=γ`,
`r0=β-γ=γ(R0-1)`.

Admissible controls:

```
A_C = { u : 0 ≤ u(t) ≤ β a.e.,  ∫_0^∞ u(t)^2 dt ≤ C }
```

Kendall functional (coefficient 1 in front of ψ):

```
ψ(t)=∫_0^t u
W[u]=γ ∫_0^∞ exp(-r0 t + ψ(t)) dt
q_ext = (W/(1+W))^{I0}
```

Maximizing q is equivalent to maximizing W.

## Candidate optimizer

**Free regime** (`1<R0≤2` any finite C, or `R0>2` and `C≤C*`):

```
u*(t) = 2 r0 a exp(-r0 t) / (1-a + a exp(-r0 t))
C = 4 r0 [-ln(1-a)-a]
W* = γ / (r0 (1-a))
```

**Constrained regime** (`R0>2`, `C>C*`):

```
a* = R0 / (2(R0-1))
C* = 4 r0 [-ln(1-a*)-a*]
T1 = (C-C*)/β^2
```

`u*=β` on `[0,T1]`, then the free Riccati tail with parameter `a*`.

```
W* = exp(γ T1) R0/(R0-2) - 1
```

At `R0=2`, `a*→1` and `C*→∞`.

## HJB verification (the global argument used here)

Explicit piecewise value function `v(E)` with `v(0)=1/r0`. For `E>0`
and all `u∈[0,β]`:

```
H(E,u)=1+(u-r0)v(E)-u^2 v'(E) ≤ 0
```

Integrating gives `J[u] ≤ v(C)`, with equality on `u*`.
Strict concavity of `H` in `u` for `E>0` gives uniqueness a.e.
Existence is by construction. Weak compactness of the L2 sphere is not used.

A uniform analytic spectral gap for the second variation on `L2(0,∞)`
is **not** claimed.

## Linear-response corollary

Small `C`: `u*(t) ~ √(2 C r0) exp(-r0 t)`.
Best rectangle width solves `2x=e^x-1`, `x*=1.25643120862617`.
Gain ratio versus that rectangle: `1.10801793` (about 10.8%).
This is a linear-regime comparison, not a standalone HJB corollary.

## Numerical checks

Layer A: discrete exact-cell `W` + projected ascent. No competing
profile kept a positive objective advantage under grid refinement.
Free Riccati recovered. For `R0>2` a plateau appears above `C*`;
measured `T1` matches `(C-C*)/β^2` to `O(Δt)`.
Backward Riccati `η(0)` matched `W/(1+W)` to about `10^{-15}`
on four representative points.

Layer B: piecewise-exact CTMC on the control grid, exact homogeneous
tail `(1/R0)^n`. Monte Carlo intervals covered Kendall `q_W`.
Do not optimize raw short-horizon Gillespie `q-hat`.

## Out of scope

Finite-N SIS, QSD control, and a full stochastic HJB in `(I,E)`
are a later project. In SIS the mean does not close through `m=E[I]`
alone. The physical cap remains `0≤u≤β`; finite-N depletion changes
the effective force of `u`, not the cap.

## How to run

```
python3 layerA_certificate.py
python3 layerB_ctmc_audit.py
```
