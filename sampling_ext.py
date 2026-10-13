"""
Generalized multistep DPM-Solver++ SDE samplers for ComfyUI.

Provides sampler functions for:
    dpmpp_Nm_sde_gpu    N in 1..8   multistep, SDE (eta) noise,       1 NFE/step
    dpmpp_Nm_heun_gpu   N in 1..8   multistep, + Heun corrector,      2 NFE/step
    dpmpp_Nm_ancestral  N in 1..8   multistep, ancestral-style noise, 1 NFE/step
    dpmpp_Ns_heun_gpu   N in 3..8   singlestep (intra-step collocation),
                                     no noise,                        N NFE/step
    dpmpp_Ns_sde_gpu    N in 3..8   singlestep, + SDE (eta) noise,    N NFE/step
    dpmpp_Ns_ancestral  N in 3..8   singlestep, + ancestral noise,    N NFE/step

NAMING COLLISION NOTE: the 's' (singlestep) family deliberately starts at
N=3, not N=1 or N=2. `dpmpp_2s_ancestral` and `dpmpp_sde`/`dpmpp_sde_gpu`
are REAL, well-tested samplers already shipped with ComfyUI/k-diffusion.
Registering our own, differently-derived order-2 singlestep construction
under those exact names would silently shadow the official implementation
for anyone who picks them from the dropdown. Order 3+ singlestep has no
official ComfyUI/k-diffusion implementation to collide with, so that's
where this extension's 's' family begins.

--------------------------------------------------------------------------
HONESTY NOTE ON CORRECTNESS (read before citing any results from this file):

Orders 1-3 are literal transcriptions of the reference k-diffusion /
ComfyUI implementation (sample_dpmpp_2m_sde, sample_dpmpp_3m_sde,
'midpoint' solver_type). Same update structure, same coefficients, same
SDE noise injection term. These match widely-used published code exactly.

Orders 4-8 are MY generalization, not an independent re-derivation of a
published closed-form formula for order > 3 (none is known to me). Method:

  1. Keep a history of past `denoised` (data-prediction) model outputs, at
     their positions in rescaled time h_eta (the natural integration
     variable of the exponential-integrator formulation DPM-Solver++ is
     built on).
  2. Build the Newton divided-difference interpolating polynomial through
     that history — a standard, unambiguous numerical method.
  3. Integrate that polynomial exactly against the true exponential-decay
     kernel of the step, via 12-point Gauss-Legendre quadrature (the
     integrand is polynomial*exponential and smooth, so this is exact to
     numerical precision — no approximation is introduced in step 3).

So for order >= 4 the only assumption being made is: "the trajectory of
denoised-predictions over the last p steps is well modeled by a
degree-(p-1) polynomial in h_eta-time" — the same assumption classical
Adams-Bashforth-type multistep solvers make at any order. Whether a
degree-7 polynomial extrapolation (order 8) is numerically STABLE over a
typical 15-30 step diffusion schedule is an open empirical question —
high-order polynomial extrapolation is known to be prone to
Runge-phenomenon-style blowups, especially early in the schedule (not
enough history yet - we fall back to the best available lower order) and
during the high-sigma / high-curvature part of the trajectory.

The singlestep ('s') family (order 3+) is a SEPARATE, newer piece of math
from the multistep family above: intra-step collocation at Chebyshev-
Lobatto-spaced nodes, built from the same Newton-extrapolation + exact-
quadrature primitives but with N model calls spent inside one step instead
of reusing history across steps. See multistep_math.py's docstring on
`collocation_step` for the construction, and `chebyshev_lobatto_nodes` for
why equally-spaced nodes were deliberately avoided (Runge's phenomenon).
It is NOT a transcription of a published formula for order >= 3 -- treat
its output with at least as much skepticism as multistep order 4+, and
ideally more, since it hasn't even had the order-2 special case validated
against a reference the way multistep order 2/3 could be.

Use test_order_convergence.py to empirically measure the actual observed
order of accuracy of each variant on a toy problem with a known analytic
solution before trusting order 4+ output on real image generation.
--------------------------------------------------------------------------
"""

import functools
import math
from typing import Any, Callable
import numpy as np
import torch
from tqdm.auto import trange

from comfy.k_diffusion.sampling import (
    BrownianTreeNoiseSampler,
    offset_first_sigma_for_snr,
    sigma_to_half_log_snr,
    to_d,
)
from .multistep_math import (
    adaptive_multistep_update,
    chebyshev_lobatto_nodes,
    collocation_step,
    general_multistep_update,
)


def get_ancestral_step(sigma_from, sigma_to, eta=1.):
    """
    Standard k-diffusion ancestral noise split: how much of the step from
    sigma_from to sigma_to is taken as a deterministic ODE move (down to
    sigma_down) vs. how much is re-added as fresh noise (sigma_up), so
    that the two combine back to the same marginal sigma_to. eta=1 is the
    conventional full-strength ancestral noise; eta=0 degenerates to a
    plain deterministic step (sigma_down=sigma_to, sigma_up=0). Reproduced
    from the standard k-diffusion formula (not a novel piece of this
    file).
    """
    if not eta:
        return sigma_to, 0.
    sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up


# --------------------------------------------------------------------------
# Multistep sampler (reuses denoised history across steps)
# --------------------------------------------------------------------------

@torch.no_grad()
def sample_dpmpp_Nm_sde(
    model: Any,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: dict[str, Any] | None = None,
    callback: Callable[..., Any] | None = None,
    disable: bool | None = None,
    eta: float = 1.0,
    s_noise: float = 1.0,
    noise_sampler: Any | None = None,
    order: int = 3,
    heun: bool = False,
    noise_mode: str = "sde",
    adaptive: bool = False,
) -> torch.Tensor:
    if len(sigmas) <= 1:
        return x

    extra_args = {} if extra_args is None else extra_args
    seed: int | None = extra_args.get("seed", None)
    sigma_min: float = float(sigmas[sigmas > 0].min())
    sigma_max: float = float(sigmas.max())
    is_cpu: bool = not getattr(x, "is_cuda", False)
    if noise_sampler is None:
        noise_sampler = BrownianTreeNoiseSampler(
            x, sigma_min, sigma_max, seed=seed, cpu=is_cpu
        )
    s_in: torch.Tensor = x.new_ones([x.shape[0]])

    model_sampling: Any | None = None
    if hasattr(model, "inner_model") and hasattr(model.inner_model, "model_patcher"):
        model_sampling = model.inner_model.model_patcher.get_model_object("model_sampling")
    if model_sampling is not None:
        lambda_fn: Callable[[Any], torch.Tensor] = functools.partial(
            sigma_to_half_log_snr, model_sampling=model_sampling
        )
        sigmas = offset_first_sigma_for_snr(sigmas, model_sampling)
        s_noise = s_noise * getattr(model_sampling, "noise_scale", 1.0)
    else:
        lambda_fn = lambda s: -s.log()

    hist_denoised: list[torch.Tensor] = []
    hist_h: list[float] = []
    hist_h_eta_list: list[float] = []
    adaptive_order_log: list[int] = []

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised: torch.Tensor = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})

        if sigmas[i + 1] == 0:
            x = denoised
            h = None
        else:
            sigma_up: float = 0.0
            if noise_mode == "ancestral":
                sigma_down, sigma_up = get_ancestral_step(sigmas[i].item(), sigmas[i + 1].item(), eta)
                t: float = lambda_fn(sigmas[i]).item()
                s_val: float = lambda_fn(torch.tensor(sigma_down)).item() if sigma_down > 0 else t
                h = s_val - t
                eta_h: float = 0.0
                h_eta: float = h
                alpha_t: float = (sigmas[i + 1] * math.exp(s_val)).item() if model_sampling is not None else 1.0
            else:
                lambda_s: float = lambda_fn(sigmas[i]).item()
                lambda_t: float = lambda_fn(sigmas[i + 1]).item()
                h = lambda_t - lambda_s
                eta_h = eta * h
                h_eta = h + eta_h
                alpha_t = (sigmas[i + 1] * math.exp(lambda_t)).item() if model_sampling is not None else 1.0

            x_ratio: float = (sigmas[i + 1] / sigmas[i]).item() if sigmas[i] > 0 else math.exp(-h)
            x_decay: float = x_ratio * math.exp(-eta_h)

            if heun:
                trial_sigma: float = sigma_down if noise_mode == "ancestral" else sigmas[i + 1]
                d: torch.Tensor = to_d(x, sigmas[i], denoised)
                dt: float = trial_sigma - sigmas[i]
                x_trial: torch.Tensor = x + d * dt
                denoised_2: torch.Tensor = model(x_trial, trial_sigma * s_in, **extra_args)
                denoised_for_update: torch.Tensor = (denoised + denoised_2) / 2
            else:
                denoised_for_update = denoised

            if adaptive and len(hist_denoised) > 0:
                x, _order_used = adaptive_multistep_update(
                    x, denoised_for_update, hist_denoised, hist_h_eta_list,
                    h_eta, order, growth_tolerance=1.0, x_decay=x_decay, alpha_t=alpha_t
                )
                adaptive_order_log.append(_order_used)

            elif order <= 1 or len(hist_denoised) == 0:
                x = x_decay * x + alpha_t * (-math.expm1(-h_eta)) * denoised_for_update

            elif order == 2:
                r: float = hist_h[0] / h
                d1: torch.Tensor = (denoised_for_update - hist_denoised[0]) / r
                x = (
                    x_decay * x
                    + alpha_t * (-math.expm1(-h_eta)) * denoised_for_update
                    + 0.5 * alpha_t * (-math.expm1(-h_eta)) * d1
                )

            elif order == 3 and len(hist_denoised) >= 2:
                r0: float = hist_h[0] / h
                r1: float = hist_h[1] / h
                d1_0: torch.Tensor = (denoised_for_update - hist_denoised[0]) / r0
                d1_1: torch.Tensor = (hist_denoised[0] - hist_denoised[1]) / r1
                d1_val: torch.Tensor = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2_val: torch.Tensor = (d1_0 - d1_1) / (r0 + r1)
                phi_2: float = math.expm1(-h_eta) / h_eta + 1 if h_eta != 0 else 0.5
                phi_3: float = phi_2 / h_eta - 0.5 if h_eta != 0 else 0.0
                x = (
                    x_decay * x
                    + alpha_t * (-math.expm1(-h_eta)) * denoised_for_update
                    + alpha_t * phi_2 * d1_val
                    - alpha_t * phi_3 * d2_val
                )

            else:
                x = general_multistep_update(
                    x, denoised_for_update, hist_denoised, hist_h_eta_list,
                    h_eta, order, x_decay=x_decay, alpha_t=alpha_t
                )

            if noise_mode == "ancestral":
                if sigma_up > 0:
                    x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigma_up * s_noise
            elif eta:
                x = x + (
                    noise_sampler(sigmas[i], sigmas[i + 1])
                    * sigmas[i + 1]
                    * math.sqrt(-math.expm1(-2 * eta_h))
                    * s_noise
                )

        hist_denoised.insert(0, denoised)
        if h is not None:
            hist_h.insert(0, h)
            hist_h_eta_list.insert(0, h_eta)
        keep = max(order - 1, 2)
        hist_denoised = hist_denoised[:keep]
        hist_h = hist_h[:keep]
        hist_h_eta_list = hist_h_eta_list[:keep]

    if adaptive and adaptive_order_log:
        print(f"[dpmpp adaptive] orders used per step: {adaptive_order_log} "
              f"(mean={sum(adaptive_order_log)/len(adaptive_order_log):.2f}, "
              f"max={order})")

    return x


# --------------------------------------------------------------------------
# Singlestep sampler (N model calls spent within one step, via
# Chebyshev-Lobatto intra-step collocation -- see module docstring)
# --------------------------------------------------------------------------

@torch.no_grad()
def sample_dpmpp_Ns(model, x, sigmas, extra_args=None, callback=None,
                     disable=None, eta=1., s_noise=1., noise_sampler=None,
                     order=3, noise_mode='none'):
    """
    Singlestep (intra-step collocation) sampler, order N spends N model
    calls per step. noise_mode: 'none' (deterministic, the '_heun_gpu'
    name), 'sde' (eta-scaled Brownian noise, the '_sde_gpu' name), or
    'ancestral' (the '_ancestral' name). See module docstring -- this is
    this project's own construction for order >= 3, not a published
    formula.
    """
    if len(sigmas) <= 1:
        return x

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    seed = extra_args.get("seed", None) if extra_args is not None else None
    noise_sampler = (BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True)
                      if noise_sampler is None else noise_sampler)
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised_0 = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised_0})

        if sigmas[i + 1] == 0:
            x = denoised_0
            continue

        sigma_up = 0.0
        if noise_mode == 'ancestral':
            sigma_down, sigma_up = get_ancestral_step(sigmas[i].item(), sigmas[i + 1].item(), eta)
            t = -sigmas[i].log().item()
            s_val = -math.log(sigma_down) if sigma_down > 0 else t
            h = s_val - t
            h_eta = h
        else:
            t_, s_ = -sigmas[i].log(), -sigmas[i + 1].log()
            h = (s_ - t_).item()
            t = t_.item()
            eta_h = eta * h if noise_mode == 'sde' else 0.0
            h_eta = h + eta_h

        p = order
        nodes = chebyshev_lobatto_nodes(p, h_eta)  # nodes[0]=0, nodes[-1]=h_eta
        taus = [nodes[0]]
        vals = [denoised_0]

        for k in range(1, p):
            tau_k = nodes[k]
            # predict x at tau_k from x (tau=0) using stages seen so far
            x_pred = collocation_step(x, taus, vals, tau_k)
            sigma_k = math.exp(-(t + tau_k))
            denoised_k = model(x_pred, sigma_k * s_in, **extra_args)
            taus.append(tau_k)
            vals.append(denoised_k)

        x = collocation_step(x, taus, vals, h_eta)

        if noise_mode == 'ancestral':
            if sigma_up > 0:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigma_up * s_noise
        elif noise_mode == 'sde' and eta:
            x = x + (noise_sampler(sigmas[i], sigmas[i + 1])
                      * sigmas[i + 1]
                      * math.sqrt(-math.expm1(-2 * eta_h))
                      * s_noise)

    return x


def compute_ab_weights(dt: float, past_dts: list[float], p: int) -> list[float]:
    if p == 1:
        return [dt]
    nodes: list[float] = [0.0]
    acc: float = 0.0
    for h in past_dts[: p - 1]:
        acc -= h
        nodes.append(acc)
    weights: list[float] = []
    for k in range(p):
        poly: list[float] = [1.0]
        denom: float = 1.0
        for j in range(p):
            if j != k:
                denom *= nodes[k] - nodes[j]
                new_poly: list[float] = [0.0] * (len(poly) + 1)
                for deg, c in enumerate(poly):
                    new_poly[deg + 1] += c
                    new_poly[deg] -= c * nodes[j]
                poly = new_poly
        integ: float = 0.0
        for deg, c in enumerate(poly):
            integ += c * (dt ** (deg + 1)) / (deg + 1)
        weights.append(integ / denom if abs(denom) > 1e-12 else dt / p)
    return weights


@torch.no_grad()
def sample_cab(
    model: Any,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: dict[str, Any] | None = None,
    callback: Callable[..., Any] | None = None,
    disable: bool | None = None,
    order: int = 2,
    theta: float = 0.2,
    eta: float = 0.0,
    s_noise: float = 1.0,
    noise_sampler: Any | None = None,
) -> torch.Tensor:
    if len(sigmas) <= 1:
        return x

    extra_args = {} if extra_args is None else extra_args
    seed: int | None = extra_args.get("seed", None)
    sigma_min: float = float(sigmas[sigmas > 0].min())
    sigma_max: float = float(sigmas.max())
    is_cpu: bool = not getattr(x, "is_cuda", False)
    if noise_sampler is None:
        noise_sampler = BrownianTreeNoiseSampler(
            x, sigma_min, sigma_max, seed=seed, cpu=is_cpu
        )
    s_in: torch.Tensor = x.new_ones([x.shape[0]])

    total_steps: int = len(sigmas) - 1
    v_hist: list[torch.Tensor] = []
    dt_hist: list[float] = []

    for i in trange(total_steps, disable=disable):
        denoised: torch.Tensor = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})

        if sigmas[i + 1] == 0:
            x = denoised
            break

        s_cur: float = float(sigmas[i].item())
        s_next: float = float(sigmas[i + 1].item())
        dt: float = s_next - s_cur

        v_cur: torch.Tensor = (x - denoised) / max(s_cur, 1e-4)

        curr_p: int = min(order, 1 + len(v_hist))
        w: list[float] = compute_ab_weights(dt, dt_hist[::-1], curr_p)
        vals: list[torch.Tensor] = [v_cur] + v_hist[::-1][: curr_p - 1]
        x_pred: torch.Tensor = x
        for wk, vk in zip(w, vals):
            x_pred = x_pred + wk * vk

        if len(v_hist) >= 2 and theta > 0:
            h1: float = dt_hist[-1]
            h2: float = dt_hist[-2]
            r_prev: float = h1 / h2 if abs(h2) > 1e-12 else 1.0
            v_ext: torch.Tensor = (1.0 + r_prev) * v_hist[-1] - r_prev * v_hist[-2]
            defect: torch.Tensor = v_cur - v_ext
            x_next: torch.Tensor = x_pred + theta * dt * defect
        else:
            x_next = x_pred

        v_hist.append(v_cur)
        dt_hist.append(dt)
        x = x_next

    return x


def compute_wls_weights(
    dt: float,
    past_dts: list[float],
    deg: int = 2,
    gamma: float = 0.85,
) -> list[float]:
    length: int = 1 + len(past_dts)
    if length <= deg + 1:
        d: int = length - 1
    else:
        d = deg
    nodes: list[float] = [0.0]
    acc: float = 0.0
    for h in past_dts:
        acc -= h
        nodes.append(acc)
    nodes_arr: np.ndarray = np.array(nodes)
    w_diag: np.ndarray = np.array([gamma**k for k in range(length)])
    weight_mat: np.ndarray = np.diag(w_diag)
    design_mat: np.ndarray = np.column_stack([nodes_arr**m for m in range(d + 1)])
    normal_mat: np.ndarray = design_mat.T @ weight_mat @ design_mat
    inv_mat: np.ndarray = np.linalg.pinv(normal_mat)
    projection_mat: np.ndarray = inv_mat @ design_mat.T @ weight_mat
    integral_vec: np.ndarray = np.array(
        [(dt ** (m + 1)) / (m + 1) for m in range(d + 1)]
    )
    effective_weights: np.ndarray = integral_vec @ projection_mat
    return [float(v) for v in effective_weights]


@torch.no_grad()
def sample_flow_wls(
    model: Any,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: dict[str, Any] | None = None,
    callback: Callable[..., Any] | None = None,
    disable: bool | None = None,
    history_length: int = 6,
    deg: int = 2,
    gamma: float = 0.85,
) -> torch.Tensor:
    if len(sigmas) <= 1:
        return x

    extra_args = {} if extra_args is None else extra_args
    s_in: torch.Tensor = x.new_ones([x.shape[0]])
    total_steps: int = len(sigmas) - 1
    v_hist: list[torch.Tensor] = []
    dt_hist: list[float] = []

    for i in trange(total_steps, disable=disable):
        denoised: torch.Tensor = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})

        if sigmas[i + 1] == 0:
            x = denoised
            break

        s_cur: float = float(sigmas[i].item())
        s_next: float = float(sigmas[i + 1].item())
        dt: float = s_next - s_cur

        v_cur: torch.Tensor = (x - denoised) / max(s_cur, 1e-4)
        past_to_keep: list[float] = dt_hist[::-1][: history_length - 1]
        w: list[float] = compute_wls_weights(dt, past_to_keep, deg=deg, gamma=gamma)
        vals: list[torch.Tensor] = [v_cur] + v_hist[::-1][: len(w) - 1]

        inc: torch.Tensor = torch.zeros_like(x)
        for wk, vk in zip(w, vals):
            inc = inc + wk * vk
        x = x + inc

        v_hist.append(v_cur)
        dt_hist.append(dt)

    return x


@torch.no_grad()
def sample_flow_unipc(
    model: Any,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: dict[str, Any] | None = None,
    callback: Callable[..., Any] | None = None,
    disable: bool | None = None,
    order: int = 2,
) -> torch.Tensor:
    if len(sigmas) <= 1:
        return x

    extra_args = {} if extra_args is None else extra_args
    s_in: torch.Tensor = x.new_ones([x.shape[0]])
    total_steps: int = len(sigmas) - 1
    v_hist: list[torch.Tensor] = []
    dt_hist: list[float] = []
    last_x: torch.Tensor | None = None

    for i in trange(total_steps, disable=disable):
        denoised: torch.Tensor = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})

        if sigmas[i + 1] == 0:
            x = denoised
            break

        s_cur: float = float(sigmas[i].item())
        s_next: float = float(sigmas[i + 1].item())
        dt: float = s_next - s_cur

        v_cur: torch.Tensor = (x - denoised) / max(s_cur, 1e-4)

        if last_x is not None:
            dt_prev: float = dt_hist[-1]
            x = last_x + dt_prev * (0.5 * v_cur + 0.5 * v_hist[-1])

        last_x = x.clone()

        curr_p: int = min(order, 1 + len(v_hist))
        w: list[float] = compute_ab_weights(dt, dt_hist[::-1], curr_p)
        vals: list[torch.Tensor] = [v_cur] + v_hist[::-1][: curr_p - 1]

        x_pred: torch.Tensor = x
        for wk, vk in zip(w, vals):
            x_pred = x_pred + wk * vk

        v_hist.append(v_cur)
        dt_hist.append(dt)
        x = x_pred

    return x


SAMPLER_FUNCS = {}

for _n in range(1, 9):
    SAMPLER_FUNCS[f"dpmpp_{_n}m_sde_gpu"] = functools.partial(
        sample_dpmpp_Nm_sde, order=_n, heun=False, noise_mode="sde"
    )
    SAMPLER_FUNCS[f"dpmpp_{_n}m_heun_gpu"] = functools.partial(
        sample_dpmpp_Nm_sde, order=_n, heun=True, noise_mode="sde"
    )
    SAMPLER_FUNCS[f"dpmpp_{_n}m_ancestral"] = functools.partial(
        sample_dpmpp_Nm_sde, order=_n, heun=False, noise_mode="ancestral"
    )

for _n in range(3, 9):
    SAMPLER_FUNCS[f"dpmpp_{_n}s_heun_gpu"] = functools.partial(
        sample_dpmpp_Ns, order=_n, noise_mode="none"
    )
    SAMPLER_FUNCS[f"dpmpp_{_n}s_sde_gpu"] = functools.partial(
        sample_dpmpp_Ns, order=_n, noise_mode="sde"
    )
    SAMPLER_FUNCS[f"dpmpp_{_n}s_ancestral"] = functools.partial(
        sample_dpmpp_Ns, order=_n, noise_mode="ancestral"
    )

SAMPLER_FUNCS["dpmpp_adaptive_sde_gpu"] = functools.partial(
    sample_dpmpp_Nm_sde, order=3, heun=False, noise_mode="sde", adaptive=True
)
SAMPLER_FUNCS["dpmpp_adaptive_heun_gpu"] = functools.partial(
    sample_dpmpp_Nm_sde, order=3, heun=True, noise_mode="sde", adaptive=True
)
SAMPLER_FUNCS["dpmpp_adaptive_ancestral"] = functools.partial(
    sample_dpmpp_Nm_sde, order=3, heun=False, noise_mode="ancestral", adaptive=True
)

for _ord in range(2, 7):
    SAMPLER_FUNCS[f"cab_{_ord}"] = functools.partial(
        sample_cab, order=_ord, theta=0.2, eta=0.0
    )

for _l in [3, 4, 5, 6, 8]:
    SAMPLER_FUNCS[f"flow_wls_{_l}"] = functools.partial(
        sample_flow_wls, history_length=_l, deg=2, gamma=0.85
    )

SAMPLER_FUNCS["flow_unipc_2"] = functools.partial(
    sample_flow_unipc, order=2
)
SAMPLER_FUNCS["flow_unipc_3"] = functools.partial(
    sample_flow_unipc, order=3
)
