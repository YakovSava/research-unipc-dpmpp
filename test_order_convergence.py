"""
Empirical order-of-accuracy check for sampling_ext.general_multistep_update
(the order 4-8 generalization), independent of ComfyUI / diffusion models.

Idea: the exponential-integrator update these samplers are built on solves
    dx/dh = -x + D(h)
exactly, given D(h). We pick a smooth, known D(h), solve the ODE to high
precision with scipy as ground truth, then run the SAME stepping logic
(Newton extrapolation + exact quadrature) used in sampling_ext.py over a
log-spaced range of step counts, and fit log(error) vs log(step_size) to
read off the observed order of convergence.

A multistep method of nominal order p SHOULD show error ~ O(h^p), i.e. a
log-log slope of about p -- for a few steps, while there's enough history
built up and the step size is small enough to be in the asymptotic regime.
Expect: orders 1-3 converge cleanly; watch what happens at order 5+,
especially at larger step sizes (closer to the 15-30 steps actually used
for image sampling) -- if the slope degrades or the error blows up there,
that's your "why" for whatever Discussion-section finding you write up.

Run: python test_order_convergence.py
Requires: numpy, scipy (pip install scipy if not already present)
"""

import math
import numpy as np
from scipy.integrate import solve_ivp

from multistep_math import general_multistep_update, collocation_step, chebyshev_lobatto_nodes


# --- test problem -----------------------------------------------------
# dx/dh = -x + D(h),  D(h) = sin(h) + 0.3*cos(3h)   (smooth, non-trivial,
# NOT itself an exponential -- avoids accidentally testing a degenerate
# case where high-order extrapolation is trivially exact)

def D(h):
    return math.sin(h) + 0.3 * math.cos(3 * h)


def ground_truth(h_end, x0=1.0):
    sol = solve_ivp(lambda h, x: -x + D(h), [0, h_end], [x0],
                     rtol=1e-12, atol=1e-13, dense_output=True)
    return sol.y[0, -1]


def run_stepper(h_end, n_steps, order, x0=1.0):
    h_step = h_end / n_steps
    x = x0
    hist_denoised = []  # newest first
    hist_h_eta = []
    cur_h = 0.0
    for k in range(n_steps):
        d_here = D(cur_h)
        x = general_multistep_update(x, d_here, hist_denoised, hist_h_eta,
                                      h_step, order)
        hist_denoised.insert(0, d_here)
        hist_h_eta.insert(0, h_step)
        keep = max(order - 1, 2)
        hist_denoised = hist_denoised[:keep]
        hist_h_eta = hist_h_eta[:keep]
        cur_h += h_step
    return x


def estimate_order(order, h_end=3.0, step_counts=(4, 6, 9, 14, 21, 32, 48)):
    truth = ground_truth(h_end)
    hs, errs = [], []
    for n in step_counts:
        approx = run_stepper(h_end, n, order)
        err = abs(approx - truth)
        if err > 0:  # skip exact-to-machine-precision points, they break log-fit
            hs.append(h_end / n)
            errs.append(err)
    if len(hs) < 2:
        return float("nan"), hs, errs
    log_h = np.log(hs)
    log_e = np.log(errs)
    slope, _intercept = np.polyfit(log_h, log_e, 1)
    return slope, hs, errs


def run_singlestep(h_end, n_steps, order, x0=1.0):
    """
    Same idea, but for the singlestep/collocation construction
    (sampling_ext.sample_dpmpp_Ns's deterministic core).

    IMPORTANT METHODOLOGICAL CAVEAT: this toy D(h) depends only on time,
    not on the state x -- so the intra-step "predicted x" stage inputs
    collocation_step produces are computed but never actually fed back
    into D (there's nothing to feed them into). That isolates and tests
    the interpolation + exact-quadrature machinery's accuracy, but NOT
    the error introduced by the predictor stage estimating x from an
    imperfect extrapolation, which a real diffusion model (where denoised
    genuinely depends on x) would also incur. Treat this as a check on
    "is the math internally consistent", not as a full proxy for
    real-model behavior.
    """
    h_step = h_end / n_steps
    x = x0
    cur = 0.0
    for _ in range(n_steps):
        nodes = chebyshev_lobatto_nodes(order, h_step)
        taus = [nodes[0]]
        vals = [D(cur + nodes[0])]
        for j in range(1, order):
            tau_k = nodes[j]
            _x_pred = collocation_step(x, taus, vals, tau_k)  # computed, unused by this toy D
            taus.append(tau_k)
            vals.append(D(cur + tau_k))
        x = collocation_step(x, taus, vals, h_step)
        cur += h_step
    return x


def estimate_order_singlestep(order, h_end=3.0,
                               step_counts=(4, 6, 9, 14, 21, 32, 48, 72)):
    truth = ground_truth(h_end)
    hs, errs = [], []
    for n in step_counts:
        approx = run_singlestep(h_end, n, order)
        err = abs(approx - truth)
        if err > 0:
            hs.append(h_end / n)
            errs.append(err)
    if len(hs) < 2:
        return float("nan"), hs, errs
    slope, _ = np.polyfit(np.log(hs), np.log(errs), 1)
    return slope, hs, errs


if __name__ == "__main__":
    print("=== multistep (general_multistep_update) ===")
    print(f"{'order':>5} | {'observed slope (~= order)':>28} | smallest-h error")
    print("-" * 60)
    for order in range(1, 9):
        slope, hs, errs = estimate_order(order)
        print(f"{order:>5} | {slope:>28.3f} | {errs[-1]:.3e}" if errs
              else f"{order:>5} | {'n/a':>28} | n/a")

    print()
    print("=== singlestep / collocation (chebyshev_lobatto_nodes + collocation_step) ===")
    print("NOTE: this measures interpolation+quadrature accuracy only -- see")
    print("run_singlestep()'s docstring for why it does not exercise the")
    print("predictor-stage error a real (x-dependent) model would add.")
    print(f"{'order':>5} | {'observed slope (~= order)':>28} | smallest-h error")
    print("-" * 60)
    for order in range(1, 9):
        slope, hs, errs = estimate_order_singlestep(order)
        print(f"{order:>5} | {slope:>28.3f} | {errs[-1]:.3e}" if errs
              else f"{order:>5} | {'n/a':>28} | n/a")

    print()
    print("Interpretation:")
    print(" - slope close to `order` over the small-h points: the method is")
    print("   behaving like a genuine order-p method in the asymptotic regime.")
    print(" - slope that degrades, or errs[-1] that stops shrinking / grows")
    print("   again as n_steps shrinks (i.e. look at the error column for the")
    print("   LARGE h_step entries, not just the asymptotic slope): that's the")
    print("   few-step / large-step-size regime diffusion sampling actually")
    print("   runs in (15-30 steps total), and where high-order extrapolation")
    print("   is most likely to misbehave.")
    print(" - for singlestep order 6+, watch for the error FLOORING OUT around")
    print("   1e-12ish rather than continuing to shrink or the slope degrading:")
    print("   that's float64 cancellation in the Newton divided-difference")
    print("   table at high degree with closely-clustered Chebyshev nodes, a")
    print("   numerical-conditioning limitation of this implementation, not")
    print("   evidence about the underlying method's true order. A barycentric")
    print("   Lagrange form would likely push this floor lower if you pursue")
    print("   order 6-8 further.")
