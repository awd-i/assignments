"""The standard d8 loss curve: raw + smoothed train loss, val loss, and the LR schedule."""

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.lr_tuning.plot_lr_tuning import set_style

PLOT_DIR = Path(__file__).resolve().parent / "plots"
run = json.loads((PLOT_DIR / "p5_gallery_cache.json").read_text())["model-d8-lr0.003-tok614M-modal"]


def smooth(x, window=101):
    x = np.asarray(x, float)
    return np.convolve(np.pad(x, window // 2, mode="edge"), np.ones(window) / window, mode="valid")


set_style()
steps, train = np.array(run["train_steps"]) + 1, np.array(run["train"])
val_steps, val = np.array(run["val_steps"]) + 1, np.array(run["val"])
total, warmup = 9375, 93
lr = np.where(np.arange(total) < warmup, np.arange(total) / warmup, 1 - (np.arange(total) - warmup) / (total - warmup)) * 3e-3

fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
for ax, log in zip(axes, (False, True)):
    if log:  # a moving average distorts the first steps on a log axis, so show the raw loss here
        ax.plot(steps, train, color="#7030A0", linewidth=0.8, label="train loss (raw, per step)")
    else:
        ax.plot(steps, train, color="#b8a4d8", linewidth=0.4, alpha=0.7, label="train loss (raw, per step)")
        ax.plot(steps, smooth(train), color="#7030A0", linewidth=1.8, label="train loss (101-step mean)")
    ax.plot(val_steps, val, color="#e76f51", linewidth=1.8, label=f"val loss (final {val[-1]:.3f})")
    ax.set_ylim(2.7, 8.5 if log else 4.5)
    ax.set_xlabel("Optimizer step (65,536 tokens each)")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle=":", alpha=0.3)
    lr_ax = ax.twinx()
    lr_ax.plot(np.arange(total) + 1, lr, color="gray", linestyle="--", linewidth=1.2, label="learning rate")
    lr_ax.set_ylim(0, 3.3e-3)
    lr_ax.set_ylabel("Learning rate", color="gray")
    lr_ax.tick_params(axis="y", colors="gray")
    if log:
        ax.set_xscale("log")
        lr_ax.set_xscale("log")
        ax.set_title("Log steps: shows the fast early drop")
        ax.annotate("1. fast drop\n(~8.8 -> ~4.5)", xy=(70, 6.0), fontsize=9)
        ax.annotate("2. long, slowing descent", xy=(700, 3.75), fontsize=9)
        ax.annotate("3. annealing drop", xy=(2500, 2.8), fontsize=9)
    else:
        ax.set_title("Linear steps: shows the slow middle and the final drop")
        ax.annotate("2. long, slowing descent\n(most of training)", xy=(2600, 3.55), fontsize=9)
        ax.annotate("3. annealing drop\nas LR -> 0", xy=(7300, 3.12), fontsize=9)
        lines = ax.get_legend_handles_labels()[0] + lr_ax.get_legend_handles_labels()[0]
        labels = ax.get_legend_handles_labels()[1] + lr_ax.get_legend_handles_labels()[1]
        ax.legend(lines, labels, frameon=False, fontsize=8.5, loc="upper right")
fig.suptitle("Standard d8 training run (LR 3e-3, 1% warmup, linear decay, bs 64, 614M tokens)", fontsize=13)
fig.tight_layout()
out = PLOT_DIR / "default_d8_curve.png"
fig.savefig(out)
print(out, "val at steps 94/938/4688/9375:", [round(float(v), 3) for v in np.interp([94, 938, 4688, 9375], val_steps, val)])
