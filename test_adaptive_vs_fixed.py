"""
Standalone, single-file test (no ComfyUI/torch) comparing FIXED order
extrapolation (general_multistep_update) against ADAPTIVE order
extrapolation (adaptive_multistep_update) under the same injected-noise
conditions as test_noise_poisoning.py.

If adaptive-order is doing its job, its error should stay close to the
BEST fixed order at each noise level (not blindly match the worst), and
in particular should NOT blow up the way fixed order=8 does at high
noise -- while still using high order (and therefore getting a real
accuracy benefit) when the noise level is low/zero.

Run: python test_adaptive_vs_fixed.py
Requires: numpy, scipy
"""

import math
import numpy as np
from scipy.integrate import solve_ivp

from multistep_math import general_multistep_update, adaptive_multistep_update


def D(h):
    return math.sin(h) + 0.3 * math.cos(3 * h)


def ground_truth(h_end, x0=1.0):
    sol = solve_ivp(lambda h, x: -x + D(h), [0, h_end], [x0],
                     rtol=1e-12, atol=1e-13)
    return sol.y[0, -1]


def run_fixed(h_end, n_steps, order, noise_sigma, rng, x0=1.0):
    h_step = h_end / n_steps
    x = x0
    hist_denoised, hist_h_eta = [], []
    cur_h = 0.0
    for _ in range(n_steps):
        d_obs = D(cur_h) + (rng.normal(0, noise_sigma) if noise_sigma > 0 else 0.0)
        x = general_multistep_update(x, d_obs, hist_denoised, hist_h_eta, h_step, order)
        hist_denoised.insert(0, d_obs)
        hist_h_eta.insert(0, h_step)
        keep = max(order - 1, 2)
        hist_denoised, hist_h_eta = hist_denoised[:keep], hist_h_eta[:keep]
        cur_h += h_step
    return x


def run_adaptive(h_end, n_steps, max_order, noise_sigma, rng, x0=1.0):
    h_step = h_end / n_steps
    x = x0
    hist_denoised, hist_h_eta = [], []
    cur_h = 0.0
    orders_used = []
    for _ in range(n_steps):
        d_obs = D(cur_h) + (rng.normal(0, noise_sigma) if noise_sigma > 0 else 0.0)
        x, order_used = adaptive_multistep_update(x, d_obs, hist_denoised, hist_h_eta,
                                                    h_step, max_order)
        orders_used.append(order_used)
        hist_denoised.insert(0, d_obs)
        hist_h_eta.insert(0, h_step)
        keep = max(max_order - 1, 2)
        hist_denoised, hist_h_eta = hist_denoised[:keep], hist_h_eta[:keep]
        cur_h += h_step
    return x, orders_used


if __name__ == "__main__":
    N_STEPS = 30
    H_END = 3.0
    NOISE_LEVELS = [0.0, 0.01, 0.05, 0.15, 0.3]
    N_TRIALS = 40
    truth = ground_truth(H_END)

    print(f"n_steps={N_STEPS}, {N_TRIALS} trials per cell. Values are median abs error.\n")

    header = f"{'order/mode':>14} | " + " | ".join(f"sigma={s:<5.2f}" for s in NOISE_LEVELS)
    print(header)
    print("-" * len(header))

    for order in [1, 2, 3, 4, 6, 8]:
        row = []
        for sigma in NOISE_LEVELS:
            rng = np.random.default_rng(0)
            errs = [abs(run_fixed(H_END, N_STEPS, order, sigma, rng) - truth)
                    for _ in range(N_TRIALS)]
            row.append(f"{np.median(errs):.2e}")
        print(f"{'fixed p=' + str(order):>14} | " + " | ".join(f"{v:<11}" for v in row))

    print()
    row = []
    orders_by_sigma = {}
    for sigma in NOISE_LEVELS:
        rng = np.random.default_rng(0)
        errs = []
        all_orders = []
        for _ in range(N_TRIALS):
            val, orders_used = run_adaptive(H_END, N_STEPS, 8, sigma, rng)
            errs.append(abs(val - truth))
            all_orders.extend(orders_used)
        row.append(f"{np.median(errs):.2e}")
        orders_by_sigma[sigma] = np.mean(all_orders)
    print(f"{'adaptive p<=8':>14} | " + " | ".join(f"{v:<11}" for v in row))

    print()
    print("Mean order actually chosen by the adaptive method, per noise level:")
    for sigma, mean_order in orders_by_sigma.items():
        print(f"  sigma={sigma:<5.2f} -> mean order used = {mean_order:.2f}")

    print()
    print("What to look for:")
    print(" - sigma=0.00: adaptive should land close to the best fixed order")
    print("   (order 4-ish in the earlier clean test), i.e. it isn't leaving")
    print("   accuracy on the table when there's nothing to protect against.")
    print(" - sigma=0.30: adaptive's error should be far below fixed p=8's,")
    print("   and its mean chosen order should visibly DROP (if it's not")
    print("   dropping, the growth_tolerance default is too permissive for")
    print("   this noise level and needs tightening).")
