"""
Standalone, single-file diagnostic (no ComfyUI, no torch, no custom node
required -- just numpy/scipy). Tests one specific hypothesis about why the
*_ancestral and *_sde multistep variants (order 5+) visibly "blow up" on
real generations while the deterministic order 5+ variant merely degrades:

    HYPOTHESIS: in 'sde'/'ancestral' noise modes, fresh random noise is
    injected into the state AFTER each step. The next step's `denoised`
    is evaluated on that noise-perturbed state. So the history of
    `denoised` values the order-N extrapolator fits a degree-(N-1)
    polynomial through is not a sample of a smooth trajectory anymore --
    it's a smooth trajectory PLUS i.i.d.-ish perturbation at every point.
    Fitting a high-degree polynomial through noisy (not just smooth-but-
    curved) data is a textbook recipe for wild overshoot between/beyond
    the fitted points (worse than plain Runge's phenomenon, which assumes
    exact data) -- and the multistep extrapolation step explicitly
    extrapolates one step BEYOND the last known point, which is exactly
    where that overshoot is largest.

This script builds a toy problem with a KNOWN smooth analytic solution,
deliberately corrupts the per-step `denoised` observations with noise of
controllable magnitude (0 = no noise = previous clean test), and measures
how the extrapolation error grows with order AND with noise magnitude.

Run: python test_noise_poisoning.py
Requires: numpy, scipy
"""

import math
import numpy as np
from scipy.integrate import solve_ivp

from multistep_math import general_multistep_update


def D(h):
    return math.sin(h) + 0.3 * math.cos(3 * h)


def ground_truth(h_end, x0=1.0):
    sol = solve_ivp(lambda h, x: -x + D(h), [0, h_end], [x0],
                     rtol=1e-12, atol=1e-13)
    return sol.y[0, -1]


def run_stepper_noisy(h_end, n_steps, order, noise_sigma, rng, x0=1.0):
    """
    Same stepper as test_order_convergence.py's run_stepper, but each
    `denoised` observation fed into the history is D(h) + noise, with
    noise ~ N(0, noise_sigma^2) drawn fresh each step -- modeling the
    effect of the model being evaluated on a noise-perturbed state,
    without needing an actual diffusion model to demonstrate the
    mechanism.
    """
    h_step = h_end / n_steps
    x = x0
    hist_denoised = []
    hist_h_eta = []
    cur_h = 0.0
    for _ in range(n_steps):
        d_clean = D(cur_h)
        d_observed = d_clean + rng.normal(0, noise_sigma) if noise_sigma > 0 else d_clean
        x = general_multistep_update(x, d_observed, hist_denoised, hist_h_eta,
                                      h_step, order)
        hist_denoised.insert(0, d_observed)
        hist_h_eta.insert(0, h_step)
        keep = max(order - 1, 2)
        hist_denoised = hist_denoised[:keep]
        hist_h_eta = hist_h_eta[:keep]
        cur_h += h_step
    return x


def run_trial(order, noise_sigma, n_steps=30, h_end=3.0, n_trials=40, seed=0):
    truth = ground_truth(h_end)
    rng = np.random.default_rng(seed)
    errs = [abs(run_stepper_noisy(h_end, n_steps, order, noise_sigma, rng) - truth)
            for _ in range(n_trials)]
    return float(np.median(errs)), float(np.max(errs))


if __name__ == "__main__":
    N_STEPS = 30          # matches the user's actual ComfyUI run (steps=30)
    NOISE_LEVELS = [0.0, 0.01, 0.05, 0.15, 0.3]
    ORDERS = list(range(1, 9))

    print(f"Toy SDE-like stepper, n_steps={N_STEPS} (matches real test), "
          f"40 random trials per cell.\n")
    print(f"{'order':>5} | " + " | ".join(f"sigma={s:<5.2f}" for s in NOISE_LEVELS))
    print("-" * (8 + 14 * len(NOISE_LEVELS)))
    for order in ORDERS:
        row = []
        for sigma in NOISE_LEVELS:
            med, mx = run_trial(order, sigma, n_steps=N_STEPS)
            row.append(f"{med:.2e}")
        print(f"{order:>5} | " + " | ".join(f"{v:<11}" for v in row))

    print()
    print("Interpretation:")
    print(" - sigma=0.00 column: reproduces the clean convergence result from")
    print("   test_order_convergence.py (error roughly flat/small across orders,")
    print("   since n_steps=30 is already in a fairly asymptotic regime for")
    print("   this toy problem).")
    print(" - Walk right along any ORDER ROW: if error grows much faster with")
    print("   sigma for HIGH orders than for low orders (order 7-8's error")
    print("   should blow up far more per unit of injected noise than order")
    print("   2-3's does), that is direct, mechanism-level support for the")
    print("   hypothesis: high-order history-based extrapolation amplifies")
    print("   per-step noise much more than low-order extrapolation does --")
    print("   consistent with what you're seeing in real ancestral/SDE runs.")
    print(" - If growth with sigma looks similar across all orders instead,")
    print("   the noise-amplification story is NOT the (main) explanation,")
    print("   and the real cause is more likely something else -- e.g. an")
    print("   actual implementation bug specific to the ancestral/SDE path,")
    print("   worth re-auditing sampling_ext.py's noise-injection code before")
    print("   trusting this hypothesis in the write-up.")
