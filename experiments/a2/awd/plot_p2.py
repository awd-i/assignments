"""A2 Problem 2(a)/(b) plots: per-budget contours, optima vs tokens, product law, joint-vs-P1 losses."""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.a2.awd.p2_fits import BUDGETS, grid, joint_optima, p1_best, power_law, quad
from experiments.lr_tuning.plot_lr_tuning import set_style

OUT = Path(__file__).resolve().parent / "plots"
set_style()
opt = joint_optima()

fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
for ax, d in zip(axes, BUDGETS):
    lrs, wds, losses = grid(d)
    o = opt[d]
    X, Y = np.meshgrid(np.logspace(np.log10(lrs.min() / 1.6), np.log10(lrs.max() * 1.6), 120),
                       np.logspace(np.log10(wds.min() / 1.6), np.log10(wds.max() * 1.6), 120))
    Z = quad(o["coef"], X, Y)
    levels = o["loss"] + np.array([0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16])
    cs = ax.contour(X, Y, Z, levels=levels, cmap="viridis_r", linewidths=1)
    ax.clabel(cs, fmt=lambda v: f"+{v - o['loss']:.3f}", fontsize=7)
    sc = ax.scatter(lrs, wds, c=losses, cmap="viridis_r", edgecolor="black", s=70, zorder=4)
    for x, y, l in zip(lrs, wds, losses):
        ax.annotate(f"{l:.3f}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax.scatter([o["lr"]], [o["wd"]], marker="*", s=260, color="red", edgecolor="black", zorder=5,
               label=f"fit optimum ({o['lr']:.2e}, {o['wd']:.2f})")
    xs = np.array(ax.get_xlim())
    ax.plot(xs, o["lr"] * o["wd"] / xs, "--", color="red", alpha=0.6, label="LR x WD = optimum product")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_xlabel("Peak LR")
    ax.set_title(f"{d / 1e9:.4g}B tokens (quad R2 {o['r2']:.3f})")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
axes[0].set_ylabel("Weight decay")
fig.tight_layout()
fig.savefig(OUT / "p2_contours.png")

b = np.array(BUDGETS, float)
g = np.logspace(np.log10(1.3e8), np.log10(2.8e9), 50)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
for ax, (name, vals, col) in zip(axes, [("Optimal LR", [opt[d]["lr"] for d in BUDGETS], "#3a7bd5"),
                                         ("Optimal WD", [opt[d]["wd"] for d in BUDGETS], "#e76f51"),
                                         ("Optimal LR x WD", [opt[d]["lr"] * opt[d]["wd"] for d in BUDGETS], "#7030A0")]):
    s, f, r2 = power_law(b, vals)
    ax.plot(b, vals, "o", color=col, markeredgecolor="black", markersize=8)
    ax.plot(g, [f(x) for x in g], "--", color=col, label=f"~ D^{s:+.2f}, R2 = {r2:.3f}")
    if name == "Optimal LR":
        p1 = p1_best()
        ax.plot(sorted(p1), [p1[x][2][0] for x in sorted(p1)], "s:", color="gray", alpha=0.7, label="P1 (WD .1 fixed)")
    if name == "Optimal LR x WD":
        ax.scatter([2.4576e9], [f(2.4576e9)], marker="*", s=220, color=col, edgecolor="black", zorder=5,
                   label=f"P2(c) prediction {f(2.4576e9):.2e}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training tokens D")
    ax.set_title(name)
    ax.grid(True, linestyle=":", alpha=0.3, which="both")
    ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "p2_optima_vs_tokens.png")

p1 = p1_best()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
joint = [opt[d]["best"][2] for d in BUDGETS]
lr_only = [p1[d][1] for d in BUDGETS]
a1.plot(b, lr_only, "s-", color="#888888", markeredgecolor="black", label="P1: LR tuned, WD .1")
a1.plot(b, joint, "o-", color="#7030A0", markeredgecolor="black", label="P2: LR and WD tuned jointly")
a1.set_xscale("log")
a1.set_xlabel("Training tokens D")
a1.set_ylabel("Best measured val loss")
a1.legend(frameon=False)
a1.grid(True, linestyle=":", alpha=0.3)
a2.plot(b, np.array(lr_only) - np.array(joint), "o-", color="#7030A0", markeredgecolor="black")
for x, y in zip(b, np.array(lr_only) - np.array(joint)):
    a2.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
a2.set_xscale("log")
a2.set_xlabel("Training tokens D")
a2.set_ylabel("Gain from tuning WD")
a2.grid(True, linestyle=":", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "p2_joint_vs_p1.png")
print("ok")
