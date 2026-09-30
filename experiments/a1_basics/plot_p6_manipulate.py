"""Problem 6(c): how hyperparameters manipulate parameter / gradient / activation RMS,
uniformly, only at the start, or only at the end. Existing runs only (cache from analyze_p5_p6)."""

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.lr_tuning.plot_lr_tuning import set_style

PLOT_DIR = Path(__file__).resolve().parent / "plots"
C = json.loads((PLOT_DIR / "p5_p6_cache.json").read_text())
PARAM, GRAD, RESID = "logging/rms/global/parameter", "logging/rms/global/gradient", "logging/rms/model.layers.7/activation"


def series(label, key):
    e = C[label]
    y = np.array([np.nan if v is None else v for v in e["rms"][key]], dtype=float)
    return np.array(e["rms_steps"]) + 1, y


def panel(ax, labels, key, title, ylabel, colors, log_x=True, xlim=None):
    for label, color in zip(labels, colors, strict=True):
        x, y = series(label, key)
        ax.plot(x, y, color=color, linewidth=2.4 if label == "default" else 1.6, label=label + (" (d8)" if label == "default" else ""))
    ax.set_yscale("log")
    if log_x:
        ax.set_xscale("log")
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.legend(frameon=False, fontsize=8)


set_style()
lr = ["LR 7.5e-4", "default", "LR 1.2e-2"]
wd = ["wd 0.025", "default", "wd 0.4", "wd 2.0"]
c3 = ["#3a7bd5", "black", "#e76f51"]
c4 = ["#e76f51", "black", "#3a7bd5", "#264653"]
fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
panel(axes[0, 0], lr, PARAM, "Uniform: LR scales parameters (up with LR)", "Global parameter RMS", c3)
panel(axes[0, 1], wd, PARAM, "Uniform: weight decay scales parameters (down with wd)", "Global parameter RMS", c4)
panel(axes[0, 2], lr[::2] + ["wd 2.0", "default"], GRAD, "Gradients move opposite to parameters", "Global gradient RMS (pre-clip)",
      ["#3a7bd5", "#e76f51", "#264653", "black"])
panel(axes[1, 0], lr + ["wd 2.0"], RESID, "Uniform: activations follow the same knobs", "Last-layer residual RMS", c3 + ["#264653"])
panel(axes[1, 1], ["constant", "warmup 0.25%", "default", "warmup 16%"], RESID, "Start only: warmup controls the early spike",
      "Last-layer residual RMS", ["#e76f51", "#f4a261", "black", "#3a7bd5"], log_x=False, xlim=(0, 2500))
panel(axes[1, 2], ["constant", "WSD 0.2", "cosine", "default"], PARAM, "End only: the schedule controls the late shrink",
      "Global parameter RMS", ["#e76f51", "#2a9d8f", "#3a7bd5", "black"], log_x=False)
fig.suptitle("Manipulating internal statistics with hyperparameters (default d8 in black; existing runs only)", fontsize=13)
fig.tight_layout()
out = PLOT_DIR / "p6_manipulate.png"
fig.savefig(out)
print(out)
