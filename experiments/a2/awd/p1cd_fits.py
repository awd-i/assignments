"""A2 Problem 1(c)/(d): LR power-law fits and target predictions."""

import numpy as np

from experiments.a2.awd.p1_lr_fits import by_budget, optimum
from experiments.a2.provided_sweeps import load


def power_law(budgets, stars):
    slope, intercept = np.polyfit(np.log(budgets), np.log(stars), 1)
    return slope, (lambda d: float(np.exp(intercept) * d**slope))


def local_optimum(lrs, losses, half_width=2):
    """Parabola in log2 LR through the best grid point and its neighbours (for the 8-LR Hyperball grid)."""
    i = int(np.argmin(losses))
    lo, hi = max(0, i - half_width), min(len(lrs), i + half_width + 1)
    return optimum(lrs[lo:hi], losses[lo:hi])


def adamw_optima():
    data = {**by_budget(load("P1a")), **by_budget(load("P1b"))}
    return {d: optimum(lrs, losses)[0] for d, (lrs, losses) in data.items()}


def hyperball_optima(half_width=2):
    return {d: local_optimum(lrs, losses, half_width)[0] for d, (lrs, losses) in by_budget(load("P1d")).items()}


if __name__ == "__main__":
    target_c, target_d = 4_915_200_000, 1_228_800_000
    aw = adamw_optima()
    budgets = np.array(sorted(aw))
    s6, f6 = power_law(budgets, [aw[d] for d in budgets])
    s3, f3 = power_law(budgets[3:], [aw[d] for d in budgets[3:]])
    print("AdamW optima:", {f"{d / 1e6:.1f}M": f"{aw[d]:.2e}" for d in budgets})
    print(f"(c) all six:    LR* ~ D^{s6:+.3f} -> LR*(4.9152B) = {f6(target_c):.3e}")
    print(f"(c) larger 3:   LR* ~ D^{s3:+.3f} -> LR*(4.9152B) = {f3(target_c):.3e}")
    sa, fa = power_law(budgets[:3], [aw[d] for d in budgets[:3]])
    print(f"    (AdamW small 3, for comparison with Hyperball: D^{sa:+.3f})")
    for hw in (1, 2):
        hb = hyperball_optima(hw)
        hb_budgets = np.array(sorted(hb))
        sh, fh = power_law(hb_budgets, [hb[d] for d in hb_budgets])
        print(f"(d) Hyperball optima (±{hw} grid pts):", {f"{d / 1e6:.1f}M": f"{hb[d]:.3e}" for d in hb_budgets},
              f" LR* ~ D^{sh:+.3f} -> LR*(1.2288B) = {fh(target_d):.3e}")
