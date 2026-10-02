"""A2 P1(c)/(d) intuition plots: the two competing AdamW fits, and Hyperball vs AdamW source sweeps."""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.a2.awd.p1_lr_fits import by_budget
from experiments.a2.awd.p1cd_fits import adamw_optima, hyperball_optima, power_law
from experiments.a2.provided_sweeps import load
from experiments.a2.awd.target_results import load as load_targets
from experiments.a2.awd.p1_lr_fits import optimum
from experiments.lr_tuning.plot_lr_tuning import set_style

OUT = Path(__file__).resolve().parent / "plots"
set_style()

# (c) two fits extrapolated to 4.9152B
aw = adamw_optima()
d = np.array(sorted(aw))
s6, f6 = power_law(d, [aw[x] for x in d])
s3, f3 = power_law(d[3:], [aw[x] for x in d[3:]])
grid = np.logspace(np.log10(1.3e8), np.log10(6e9), 100)
fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.scatter(d, [aw[x] for x in d], s=60, color="#7030A0", edgecolor="black", zorder=5, label="Fitted optima (P1a + P1b)")
ax.plot(grid, [f6(x) for x in grid], "--", color="#e76f51", label=f"All six: D^{s6:+.2f} -> {f6(4.9152e9):.2e}")
ax.plot(grid, [f3(x) for x in grid], ":", color="#2a9d8f", linewidth=2, label=f"Larger three: D^{s3:+.2f} -> {f3(4.9152e9):.2e}")
ax.axvline(4.9152e9, color="gray", linestyle=":", linewidth=1)
tc = sorted((r["lr"], r["val_loss"]) for r in load_targets("a2-p1-D4915m"))
grid_pts = [(l, v) for l, v in tc if l in (0.0015, 0.003, 0.006)]
t_star = optimum(*map(np.array, zip(*grid_pts)))[0]
ax.scatter([4.9152e9], [t_star], marker="*", s=260, color="gold", edgecolor="black", zorder=6,
           label=f"Measured target optimum {t_star:.2e}")
ins = ax.inset_axes([0.6, 0.36, 0.33, 0.3])
best = min(v for _, v in grid_pts)
for l, v in tc:
    col = "#e76f51" if abs(l - 3.73e-3) < 1e-5 else "#2a9d8f" if abs(l - 2.23e-3) < 1e-5 else "#555555"
    ins.scatter([l], [v - best], color=col, edgecolor="black", s=30, zorder=4)
ins.plot([l for l, _ in tc], [v - best for _, v in tc], "-", color="gray", linewidth=0.8)
ins.set_xscale("log")
ins.set_title("4.9152B: loss - best grid loss", fontsize=8)
ins.tick_params(labelsize=7)
ins.set_xticks([1.5e-3, 3e-3, 6e-3], ["1.5e-3", "3e-3", "6e-3"])
ins.minorticks_off()
ax.set_xscale("log")
ax.set_xlabel("Training tokens D")
ax.set_ylabel("Optimal peak LR")
ax.set_title("(c) Two power-law fits, extrapolated to 4.9152B")
ax.grid(True, linestyle=":", alpha=0.3)
ax.legend(frameon=False, fontsize=8.5, loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "p1c_two_fits.png")

# (d) Hyperball vs AdamW at the three source budgets
hb_data, aw_data = by_budget(load("P1d")), by_budget(load("P1a"))
colors = plt.cm.viridis(np.linspace(0.1, 0.8, 3))
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 4.8))
for c, budget in zip(colors, sorted(hb_data), strict=True):
    lrs, losses = hb_data[budget]
    ax_l.plot(lrs, losses, "-o", color=c, markeredgecolor="black", label=f"Hyperball {budget / 1e6:.1f}M")
    lrs_a, losses_a = aw_data[budget]
    ax_l.plot(lrs_a, losses_a, "--s", color=c, markeredgecolor="black", label=f"AdamW {budget / 1e6:.1f}M")
td = sorted((r["lr"], r["val_loss"]) for r in load_targets("a2-p1d-hyperball-D1229m"))
ax_l.plot([l for l, _ in td], [v for _, v in td], "-o", color="#f4a261", markeredgecolor="black",
          label="Hyperball 1228.8M (target)")
ax_l.set_xscale("log")
ax_l.set_ylim(2.8, 3.6)
ax_l.set_xlabel("Peak LR")
ax_l.set_ylabel("Final val loss")
ax_l.set_title("Loss vs LR: Hyperball (solid) vs AdamW (dashed)")
ax_l.grid(True, linestyle=":", alpha=0.3)
ax_l.legend(frameon=False, fontsize=7.5, ncol=2)
hb = hyperball_optima(1)
hd = np.array(sorted(hb))
sh, fh = power_law(hd, [hb[x] for x in hd])
sa, fa = power_law(d[:3], [aw[x] for x in d[:3]])
g = np.logspace(np.log10(1.3e8), np.log10(1.5e9), 50)
ax_r.plot(hd, [hb[x] / hb[hd[0]] for x in hd], "o-", color="#3a7bd5", markeredgecolor="black", label=f"Hyperball: D^{sh:+.2f}")
ax_r.plot(d[:3], [aw[x] / aw[d[0]] for x in d[:3]], "s--", color="#e76f51", markeredgecolor="black", label=f"AdamW: D^{sa:+.2f}")
ax_r.plot(d, [aw[x] / aw[d[0]] for x in d], "s", color="#e76f51", alpha=0.35, label="AdamW larger budgets (P1b)")
ax_r.scatter([1.2288e9], [fh(1.2288e9) / hb[hd[0]]], marker="*", s=200, color="#3a7bd5", edgecolor="black", zorder=6,
             label=f"Hyperball prediction at 1.2288B ({fh(1.2288e9):.2e})")
t_hb = optimum(*map(np.array, zip(*td)))[0]
ax_r.scatter([1.2288e9], [t_hb / hb[hd[0]]], marker="D", s=80, color="gold", edgecolor="black", zorder=7,
             label=f"Hyperball measured target optimum ({t_hb:.2e})")
ax_r.set_xscale("log")
ax_r.set_xlabel("Training tokens D")
ax_r.set_ylabel("Optimal LR / optimal LR at 153.6M")
ax_r.set_title("How the optimum moves with budget (normalized)")
ax_r.grid(True, linestyle=":", alpha=0.3)
ax_r.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "p1d_hyperball_vs_adamw.png")
print("ok")
