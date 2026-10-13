"""
Pure-numpy core of the generalized multistep extrapolation (no torch / no
ComfyUI dependency), so it can be imported and unit-tested standalone.
sampling_ext.py imports from here and wraps it with torch tensors for the
actual ComfyUI sampler. See sampling_ext.py's module docstring for the
full correctness/honesty notes -- this file is just the math.
"""

import math
from typing import Any
import numpy as np

_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(12)


def divided_differences(taus, vals):
    """
    taus: list[float] (taus[0] == 0.0, remainder negative/oldest-most-neg)
    vals: list (same length/order as taus -- scalars, numpy arrays, or
          torch tensors all work, since only +,-,* ,/ by a python float
          are used)
    Returns Newton-form coefficients coeffs[0..p-1]:
        D(s) ~= sum_j coeffs[j] * prod_{m<j}(s - taus[m])
    """
    p = len(vals)
    table = list(vals)
    coeffs = [table[0]]
    for order in range(1, p):
        for i in range(p - 1, order - 1, -1):
            table[i] = (table[i] - table[i - 1]) / (taus[i] - taus[i - order])
        coeffs.append(table[order])
    return coeffs


def poly_exp_integral(h, taus, j):
    """
    I_j(h) = integral_0^h  exp(-(h - s)) * prod_{m=0}^{j-1}(s - taus[m])  ds
    via 12-point Gauss-Legendre quadrature.
    """
    s = 0.5 * h * (_GL_NODES + 1)
    w = 0.5 * h * _GL_WEIGHTS
    integrand = np.exp(-(h - s))
    for m in range(j):
        integrand = integrand * (s - taus[m])
    return float(np.sum(w * integrand))


def general_multistep_update(
    x: Any,
    denoised: Any,
    hist_denoised: list[Any],
    hist_h_eta: list[float],
    h_eta: float,
    order: int,
    x_decay: float | None = None,
    alpha_t: float = 1.0,
) -> Any:
    p: int = min(order, 1 + len(hist_denoised))
    vals: list[Any] = [denoised] + hist_denoised[: p - 1]
    taus: list[float] = [0.0]
    acc: float = 0.0
    for hh in hist_h_eta[: p - 1]:
        acc -= hh
        taus.append(acc)
    taus = taus[:p]

    d: list[Any] = divided_differences(taus, vals)
    effective_decay: float = math.exp(-h_eta) if x_decay is None else x_decay
    x_new: Any = effective_decay * x
    for j in range(p):
        ij: float = poly_exp_integral(h_eta, taus, j)
        x_new = x_new + alpha_t * ij * d[j]
    return x_new


def _magnitude(t: Any) -> float:
    if isinstance(t, np.ndarray):
        return float(np.mean(np.abs(t)))
    if hasattr(t, "abs"):
        m = t.abs()
        if hasattr(m, "mean"):
            m = m.mean()
        return float(m)
    return float(abs(t))


def adaptive_multistep_update(
    x: Any,
    denoised: Any,
    hist_denoised: list[Any],
    hist_h_eta: list[float],
    h_eta: float,
    max_order: int,
    growth_tolerance: float = 1.0,
    x_decay: float | None = None,
    alpha_t: float = 1.0,
) -> tuple[Any, int]:
    p_max: int = min(max_order, 1 + len(hist_denoised))
    vals: list[Any] = [denoised] + hist_denoised[: p_max - 1]
    taus: list[float] = [0.0]
    acc: float = 0.0
    for hh in hist_h_eta[: p_max - 1]:
        acc -= hh
        taus.append(acc)
    taus = taus[:p_max]

    d: list[Any] = divided_differences(taus, vals)
    effective_decay: float = math.exp(-h_eta) if x_decay is None else x_decay
    base: Any = effective_decay * x
    estimate: Any = base
    estimates: list[Any] = []
    for j in range(p_max):
        ij: float = poly_exp_integral(h_eta, taus, j)
        estimate = estimate + alpha_t * ij * d[j]
        estimates.append(estimate)

    if p_max == 1:
        return estimates[0], 1

    scale: float = max(_magnitude(x), 1e-8)
    deltas: list[float] = [
        _magnitude(estimates[j] - estimates[j - 1]) / scale
        for j in range(1, p_max)
    ]

    chosen_order: int = 1
    for k in range(len(deltas)):
        prev_delta: float = deltas[k - 1] if k > 0 else deltas[0]
        if deltas[k] <= prev_delta * growth_tolerance:
            chosen_order = k + 2
        else:
            break

    return estimates[chosen_order - 1], chosen_order


# --------------------------------------------------------------------------
# Singlestep (intra-step collocation) building blocks.
#
# Unlike the multistep update above (which reuses `denoised` values from
# PREVIOUS steps as history), this predicts and evaluates several
# intra-step nodes within a SINGLE step, then does one exact exponential
# integral through all of them. It's the same Newton-interpolation +
# exact-quadrature machinery, just with "history" being this step's own
# collocation points instead of past steps.
#
# This is NOT a transcription of a published formula beyond order 2 (order
# 2 recovers something conceptually equivalent to the existing
# dpmpp_2s/dpmpp_sde family, but not guaranteed bit-identical to it --
# see sampling_ext.py). It's a direct, defensible construction (akin in
# spirit to exponential/collocation Runge-Kutta methods), not numerology,
# but it is NEW, untested-in-the-literature math for order >= 3.
# --------------------------------------------------------------------------

def chebyshev_lobatto_nodes(p, h_eta):
    """
    p collocation nodes spanning [0, h_eta], Chebyshev-Gauss-Lobatto
    spaced (clustered near both ends) rather than equally spaced.

    Deliberate choice: equally-spaced high-degree polynomial interpolation
    is a textbook recipe for Runge's phenomenon (wild oscillation near the
    interval ends). Chebyshev-Lobatto spacing is the standard fix and is
    what you'd reach for in any numerical-methods reference when building
    a degree-(p-1) collocation scheme. It does not make order 7-8
    collocation safe in any absolute sense -- just removes one well-known,
    easily-avoidable source of instability from the others you're already
    investigating (see the module docstring in sampling_ext.py).

    Returns an increasing list, nodes[0] == 0.0, nodes[-1] == h_eta.
    """
    if p == 1:
        return [0.0]
    ks = np.arange(p)
    x = np.cos(ks * math.pi / (p - 1))       # 1 .. -1
    tau = h_eta * (1 - x) / 2                 # 0 .. h_eta, increasing
    return [float(v) for v in tau]


def collocation_step(x0, taus, vals, target_tau):
    """
    x0 is the state at tau=0. (taus, vals) are known (tau, D(tau)) pairs
    with taus[0] == 0.0, strictly increasing, D the data-prediction
    (`denoised`) at that point. target_tau is the point to advance x0 to --
    equal to taus[-1] for the final corrector step (pure interpolation,
    all nodes bracket the target) or beyond taus[-1] for a stage
    prediction (extrapolation by one node spacing, same character as an
    explicit multistep predictor).

    Returns x at tau=target_tau via exact integration (Gauss-Legendre
    quadrature against the exponential kernel) of the Newton
    interpolant/extrapolant of D through the given nodes.
    """
    d = divided_differences(taus, vals)
    x_new = math.exp(-target_tau) * x0
    for j in range(len(vals)):
        Ij = poly_exp_integral(target_tau, taus, j)
        x_new = x_new + Ij * d[j]
    return x_new
