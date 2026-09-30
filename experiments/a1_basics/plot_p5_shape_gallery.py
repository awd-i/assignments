"""Problem 5(c): gallery of runs whose loss curves changed shape most, each vs the default d8 run."""

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import wandb

from experiments.lr_tuning.plot_lr_tuning import set_style

PROJECT = "whitedeer-stanford-university/assignments"
PLOT_DIR = Path(__file__).resolve().parent / "plots"
CACHE = PLOT_DIR / "p5_gallery_cache.json"
D8 = "model-d8-lr0.003-tok614M"
DEFAULT = f"{D8}-modal"
PANELS = [
    ("LR 0.1: stuck high, late plunge", "model-d8-lr0.1-tok614M-scaling-bc-v1-c-lr0.1", "val"),
    ("Batch size 1024: early plateau, never catches up", "model-d8-lr0.003-bs1024-tok614M-scaling-bc-v1-c-bs1024", "val"),
    ("16x repeated data (d9): val turns up", "model-d9-lr0.003-epochs16.0-tok38.4M-scaling-bc-v1-c-repeat16", "train+val"),
    ("Constant LR: no annealing drop", f"{D8}-constant-sched-sweeps-v1-schedule-lr", "val"),
    ("WSD 0.2: flat, then sharp drop at the end", f"{D8}-wsd0.2-sched-sweeps-v1-schedule-lr", "val"),
    ("Weight decay 2.0: slower descent", f"{D8}-wd2.0-scaling-bc-v1-c-wd2", "val"),
    ("Tiny data (d6, 3M tokens, 46 steps): still steep", "model-d6-lr0.003-tok3.07M-scaling-bc-v1-c-tiny-data", "val"),
    ("Batch size 16: same shape, much noisier (raw train loss)", "model-d8-lr0.003-bs16-tok614M-hparam-sweeps-v2-batch-size", "rawtrain"),
]


def fetch():
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    api = wandb.Api(timeout=60)
    for name in {DEFAULT, *(p[1] for p in PANELS)} - set(cache):
        run = [r for r in api.runs(PROJECT, filters={"display_name": name}) if r.state == "finished"][-1]
        train = {int(r["optimizer_step"]): r["train_loss"] for r in run.scan_history(keys=["optimizer_step", "train_loss"])}
        val = {int(r["optimizer_step"]): r["val_loss"] for r in run.scan_history(keys=["optimizer_step", "val_loss"])}
        cache[name] = {"bs": run.config.get("batch_size", 64), "train_steps": sorted(train), "train": [train[s] for s in sorted(train)],
                       "val_steps": sorted(val), "val": [val[s] for s in sorted(val)]}
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache))
    return cache


def tokens(entry, key):
    return (np.array(entry[f"{key}_steps"]) + 1) * entry["bs"] * 1024


def smooth(x, window=51):
    x = np.asarray(x, float)
    return np.convolve(np.pad(x, window // 2, mode="edge"), np.ones(window) / window, mode="valid")


def main():
    set_style()
    cache = fetch()
    base = cache[DEFAULT]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for ax, (title, name, kind) in zip(axes.flat, PANELS, strict=True):
        run = cache[name]
        if kind == "rawtrain":
            n = len(run["train"])
            window = slice(n // 2, n // 2 + 1200)
            ax.plot(tokens(run, "train")[window] / 1e6, np.array(run["train"])[window], color="#e76f51", linewidth=0.6, label="bs 16 (raw)")
            m = len(base["train"])
            bw = slice(m // 2, m // 2 + 300)
            ax.plot(tokens(base, "train")[bw] / 1e6, np.array(base["train"])[bw], color="black", linewidth=0.9, label="default bs 64 (raw)")
            ax.set_xlabel("Training tokens (M), mid-training window")
            ax.set_ylabel("Train loss")
        else:
            ax.plot(tokens(base, "val"), base["val"], color="black", linewidth=1.8, label=f"default d8 val ({base['val'][-1]:.3f})")
            if kind == "train+val":
                ax.plot(tokens(run, "train"), smooth(run["train"]), color="#3a7bd5", linewidth=1.2, label="train (smoothed)")
            ax.plot(tokens(run, "val"), run["val"], color="#e76f51", linewidth=1.8, label=f"val ({run['val'][-1]:.3f})")
            ax.set_xscale("log")
            ax.set_ylim(2.2 if kind == "train+val" else 2.8, 7.5)
            ax.set_xlabel("Training tokens")
            ax.set_ylabel("Loss")
        ax.set_title(title, fontsize=10.5)
        ax.grid(True, linestyle=":", alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Runs that changed the loss-curve shape most (each vs the default d8 run, black)", fontsize=14)
    fig.tight_layout()
    out = PLOT_DIR / "p5_shape_gallery.png"
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
