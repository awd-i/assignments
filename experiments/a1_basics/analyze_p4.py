"""Problem 4: how a tiny data perturbation grows under full determinism.

For each perturbed run, compare per-step train loss and every-94-step val loss with
deterministic-reference-1 (identical except for the perturbation).
"""

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
CACHE = PLOT_DIR / "p4_cache.json"
BASE = "model-d8-lr0.003-tok614M-deterministic"
RUNS = {
    "reference-1": f"{BASE}-deterministic-reference-1",
    "reference-2": f"{BASE}-deterministic-reference-2",
    "handout: 1 token @0 (->17)": f"{BASE}-perturb1tok-p4-amplification-handout",
    "64 tokens @0": f"{BASE}-perturb64tok-step0-p4-amplification-size",
    "1 sequence @0": f"{BASE}-perturb1024tok-step0-p4-amplification-size",
    "1 batch @0": f"{BASE}-perturb65536tok-step0-p4-amplification-size",
    "1 token @25%": f"{BASE}-perturb1tok-step2343-p4-amplification-time",
    "1 token @50%": f"{BASE}-perturb1tok-step4687-p4-amplification-time",
    "1 batch @50%": f"{BASE}-perturb65536tok-step4687-p4-amplification-size",
    "1 token @75%": f"{BASE}-perturb1tok-step7031-p4-amplification-time",
    "1 token @95%": f"{BASE}-perturb1tok-step8906-p4-amplification-time",
}
PERTURB_STEP = {"1 token @25%": 2343, "1 token @50%": 4687, "1 batch @50%": 4687,
                "1 token @75%": 7031, "1 token @95%": 8906}


def fetch():
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    api = wandb.Api(timeout=60)
    for label, name in RUNS.items():
        if label in cache:
            continue
        runs = [r for r in api.runs(PROJECT, filters={"display_name": name}) if r.state == "finished"]
        if not runs:
            print(f"missing: {label}")
            continue
        run = runs[-1]
        train = {int(r["optimizer_step"]): r["train_loss"] for r in run.scan_history(keys=["optimizer_step", "train_loss"])}
        val = {int(r["optimizer_step"]): r["val_loss"] for r in run.scan_history(keys=["optimizer_step", "val_loss"])}
        cache[label] = {"train": [train[s] for s in sorted(train)], "val_steps": sorted(val), "val": [val[s] for s in sorted(val)]}
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache))
    return cache


def rolling_rms(x, window=101):
    x = np.asarray(x, float) ** 2
    kernel = np.ones(window) / window
    return np.sqrt(np.convolve(np.pad(x, window // 2, mode="edge"), kernel, mode="valid"))


def main() -> None:
    set_style()
    cache = fetch()
    ref = cache["reference-1"]
    ref_train, ref_val = np.array(ref["train"]), np.array(ref["val"])
    print(f"reference-1 vs reference-2: max |train diff| = {np.max(np.abs(ref_train - np.array(cache['reference-2']['train']))):.3g}, "
          f"final val {ref_val[-1]:.6f} vs {cache['reference-2']['val'][-1]:.6f}")

    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    size_labels = ["handout: 1 token @0 (->17)", "64 tokens @0", "1 sequence @0", "1 batch @0"]
    time_labels = ["handout: 1 token @0 (->17)", "1 token @25%", "1 token @50%", "1 token @75%", "1 token @95%", "1 batch @50%"]
    for label, entry in cache.items():
        if label.startswith("reference"):
            continue
        train_diff = np.array(entry["train"]) - ref_train
        val_diff = np.array(entry["val"]) - ref_val
        first = int(np.argmax(train_diff != 0)) if np.any(train_diff != 0) else None
        # The loss on the perturbed batch itself (and the next one, for ~1-batch edits) is
        # measured on different tokens; exclude it so the metric is pure divergence.
        after = np.abs(train_diff)
        if first is not None:
            after = after.copy()
            after[first : first + 2] = 0
        rows.append((label, first, float(np.max(after)), float(val_diff[-1]), float(np.max(np.abs(val_diff)))))
        train_diff = np.where(np.abs(train_diff) > 1, 0, train_diff)
        steps = np.arange(len(train_diff)) + 1
        for ax, labels in ((axes[0], size_labels), (axes[1], time_labels)):
            if label in labels:
                ax.plot(steps, rolling_rms(train_diff), linewidth=1.3, label=label)
        axes[2].plot(np.array(entry["val_steps"]) + 1, np.abs(val_diff) + 1e-7, linewidth=1.2, label=label)
    for ax, title in zip(axes, ("Train-loss divergence by perturbation size", "…by perturbation time", "|val-loss difference| vs reference"), strict=True):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Optimizer step")
        ax.set_title(title, fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.3)
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("RMS of train-loss difference (101-step window)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "p4_amplification.png")
    plt.close(fig)

    print("\nlabel | first differing step | max |train diff| | final val diff | max |val diff|")
    for label, first, max_train, final_val, max_val in rows:
        print(f"  {label:<28} {first!s:>6}  {max_train:.4f}  {final_val:+.5f}  {max_val:.5f}")
    print(f"\n-> {PLOT_DIR}/p4_amplification.png")


if __name__ == "__main__":
    main()
