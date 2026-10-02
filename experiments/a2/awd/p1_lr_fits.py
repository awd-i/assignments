"""A2 Problem 1(a)/(b): fit loss vs log2(LR) per token budget, find the optimal LR, and fit a scaling rule."""

import sys

import numpy as np

from experiments.a2.provided_sweeps import load

SEED_SD = 0.003  # A1 Problem 3: seed-to-seed sd of final val loss for d8


def optimum(lrs, losses):
    """Vertex of the least-squares parabola in log2(LR). Returns (lr*, loss at lr*, curvature)."""
    x = np.log2(lrs)
    a, b, c = np.polyfit(x, losses, 2)
    x_star = -b / (2 * a)
    return 2 ** x_star, a * x_star**2 + b * x_star + c, a


def by_budget(rows):
    budgets = sorted({r["tokens"] for r in rows})
    out = {}
    for d in budgets:
        pts = sorted((r["learning_rate"], r["final_val_loss"]) for r in rows if r["tokens"] == d)
        out[d] = (np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))
    return out


def bootstrap_optimum(lrs, losses, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    stars = []
    for _ in range(n):
        noisy = losses + rng.normal(0, SEED_SD, size=losses.shape)
        a = np.polyfit(np.log2(lrs), noisy, 2)[0]
        if a <= 0:
            continue  # no interior minimum in this resample
        stars.append(optimum(lrs, noisy)[0])
    stars = np.array(stars)
    return np.quantile(stars, [0.1, 0.5, 0.9]), len(stars) / n


def fit_power_law(budgets, lr_stars):
    slope, intercept = np.polyfit(np.log(budgets), np.log(lr_stars), 1)
    return slope, lambda d: np.exp(intercept) * np.asarray(d, float) ** slope


def main(part="P1a"):
    data = by_budget(load(part))
    budgets, stars = [], []
    for d, (lrs, losses) in data.items():
        lr_star, loss_star, curv = optimum(lrs, losses)
        (q10, q50, q90), frac = bootstrap_optimum(lrs, losses)
        print(f"D = {d / 1e6:7.1f}M: losses {np.round(losses, 4)}  ->  LR* = {lr_star:.2e} (loss {loss_star:.4f}, "
              f"curvature {curv:.4f})  80% interval [{q10:.2e}, {q90:.2e}]  ({frac:.0%} of resamples have a minimum)")
        budgets.append(d)
        stars.append(lr_star)
    slope, rule = fit_power_law(budgets, stars)
    print(f"\nPower-law rule: LR*(D) = {rule(614.4e6):.2e} * (D / 614.4M)^{slope:.3f}")
    for d in (1228.8e6, 1843.2e6, 2457.6e6):
        print(f"  D = {d / 1e6:7.1f}M -> predicted LR* = {rule(d):.2e}")


if __name__ == "__main__":
    main(*sys.argv[1:])
