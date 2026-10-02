"""A2 Problem 1(a)/(b) plots: loss-vs-LR fits per budget, and fitted optima vs budget with the P1(a) rule."""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.a2.awd.p1_lr_fits import bootstrap_optimum, by_budget, fit_power_law, optimum
from experiments.a2.provided_sweeps import load
from experiments.lr_tuning.plot_lr_tuning import set_style

OUT = Path(__file__).resolve().parent / "plots"
set_style()
small, large = by_budget(load("P1a")), by_budget(load("P1b"))
colors = plt.cm.viridis(np.linspace(0.05, 0.9, 6))

fig, (ax_curve, ax_star) = plt.subplots(1, 2, figsize=(15, 5.4))
stars = {}
for color, (d, (lrs, losses)) in zip(colors, {**small, **large}.items(), strict=True):
    lr_star, loss_star, _ = optimum(lrs, losses)
    (q10, _, q90), _ = bootstrap_optimum(lrs, losses)
    stars[d] = (lr_star, q10, q90)
    grid = np.logspace(np.log10(1.2e-3), np.log10(7.5e-3), 100)
    a, b, c = np.polyfit(np.log2(lrs), losses, 2)
    shift = losses.min()  # plot loss above each budget's best measured point, so all curves share one axis
    ax_curve.plot(grid, np.polyval([a, b, c], np.log2(grid)) - shift, color=color, linewidth=1.4)
    ax_curve.scatter(lrs, losses - shift, color=color, edgecolor="black", zorder=5,
                     label=f"{d / 1e9:.4g}B tokens{' (P1b)' if d in large else ''}")
    ax_curve.scatter([lr_star], [loss_star - shift], marker="*", s=160, color=color, edgecolor="black", zorder=6)
ax_curve.set_xscale("log")
ax_curve.set_xticks([1.5e-3, 3e-3, 6e-3])
ax_curve.set_xticklabels(["1.5e-3", "3e-3", "6e-3"])
ax_curve.minorticks_off()
ax_curve.set_xlabel("Peak learning rate")
ax_curve.set_ylabel("Final val loss − best measured loss at that budget")
ax_curve.set_title("Loss vs LR per budget (parabola in log LR; star = fitted optimum)")
ax_curve.grid(True, linestyle=":", alpha=0.3)
ax_curve.legend(frameon=False, fontsize=8.5)

d_small = np.array(sorted(small))
slope, rule = fit_power_law(d_small, [stars[d][0] for d in d_small])
grid_d = np.logspace(np.log10(1.2e8), np.log10(3e9), 100)
ax_star.plot(grid_d, rule(grid_d), "--", color="#e76f51", label=f"P1(a) rule: LR* ~ D^{slope:.2f} (pre-registered)")
sat = 3.30e-3 - 0.92e-3 * (grid_d / 153.6e6) ** -1.52
ax_star.plot(grid_d, sat, ":", color="#2a9d8f", label="Saturating alternative (pre-registered)")
for d, (lr_star, q10, q90) in stars.items():
    is_large = d in large
    ax_star.errorbar(d, lr_star, yerr=[[lr_star - q10], [q90 - lr_star]], fmt="o", capsize=4,
                     color="#3a7bd5" if is_large else "#7030A0", markeredgecolor="black",
                     label=None if d not in (d_small[0], sorted(large)[0]) else ("P1(b) fitted optima (80% interval)" if is_large else "P1(a) fitted optima (80% interval)"))
ax_star.axvline(614.4e6, color="gray", linestyle=":", linewidth=1)
ax_star.text(640e6, 2.3e-3, "fit | held out", fontsize=8.5, color="gray")
ax_star.set_xscale("log")
ax_star.set_xlabel("Training tokens D")
ax_star.set_ylabel("Optimal peak LR")
ax_star.set_title("Fitted optimal LR vs budget")
ax_star.grid(True, linestyle=":", alpha=0.3)
ax_star.legend(frameon=False, fontsize=8.5, loc="upper left")
fig.tight_layout()
OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "p1ab_lr_scaling.png")
print(OUT / "p1ab_lr_scaling.png")
