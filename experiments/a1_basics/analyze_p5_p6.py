"""Problems 5 and 6: loss-curve shapes and internal statistics from existing W&B runs.

P5 macro: smoothed train-loss curves across LR, beta1, batch size, schedule.
P5 micro: roughness = sd of (train loss - 51-step rolling median) over 40-60% of
training; spikes = steps more than 0.1 above the rolling median.
P6: per-layer RMS of residual activations, gradients (pre-clip) and parameters for the
default run at start/middle/end, plus how interventions move the global statistics.
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
CACHE = PLOT_DIR / "p5_p6_cache.json"
D8 = "model-d8-lr0.003-tok614M"
RUNS = {
    "default": f"{D8}-modal",
    "LR 7.5e-4": "model-d8-lr0.00075-tok614M-hparam-sweeps-v2-learning-rate",
    "LR 1.2e-2": "model-d8-lr0.012-tok614M-hparam-sweeps-v2-learning-rate",
    "LR 0.1": "model-d8-lr0.1-tok614M-scaling-bc-v1-c-lr0.1",
    "bs 16": "model-d8-lr0.003-bs16-tok614M-hparam-sweeps-v2-batch-size",
    "bs 32": "model-d8-lr0.003-bs32-tok614M-hparam-sweeps-v2-batch-size",
    "bs 128": "model-d8-lr0.003-bs128-tok614M-hparam-sweeps-v2-batch-size",
    "bs 256": "model-d8-lr0.003-bs256-tok614M-hparam-sweeps-v2-batch-size",
    "β1 0.5": f"{D8}-b10.5-p5-beta1",
    "β1 0.98": f"{D8}-b10.98-p5-beta1",
    "β2 0.9": f"{D8}-b0.9-sched-sweeps-v1-optimizer-lr",
    "β2 0.99": f"{D8}-b0.99-sched-sweeps-v1-optimizer-lr",
    "no clipping": f"{D8}-nogradclip-sched-sweeps-v1-optimizer-lr",
    "warmup 0.25%": f"{D8}-warmup0.0025-hparam-sweeps-v2-warmup-percent",
    "warmup 16%": f"{D8}-warmup0.16-followups-v1-warmup",
    "wd 0.025": f"{D8}-wd0.025-hparam-sweeps-v2-weight-decay",
    "wd 0.4": f"{D8}-wd0.4-hparam-sweeps-v2-weight-decay",
    "wd 2.0": f"{D8}-wd2.0-scaling-bc-v1-c-wd2",
    "dropout 0.2": f"{D8}-dropout0.2-scaling-v1",
    "cosine": f"{D8}-cos-sched-sweeps-v1-schedule-lr",
    "WSD 0.2": f"{D8}-wsd0.2-sched-sweeps-v1-schedule-lr",
    "constant": f"{D8}-constant-sched-sweeps-v1-schedule-lr",
}
LAYERS = range(8)
RMS_KEYS = [f"logging/rms/model.layers.{i}/activation" for i in LAYERS] + [
    "logging/rms/global/activation",
    "logging/rms/global/gradient",
    "logging/rms/global/parameter",
]
for i in LAYERS:
    for module in ("self_attn.q_proj", "mlp.down_proj"):
        RMS_KEYS += [f"logging/rms/model.layers.{i}.{module}/gradient", f"logging/rms/model.layers.{i}.{module}/parameter"]


def fetch():
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    api = wandb.Api(timeout=60)
    for label, name in RUNS.items():
        if label in cache:
            continue
        runs = [r for r in api.runs(PROJECT, filters={"display_name": name}) if r.state == "finished"]
        if not runs:
            print(f"missing: {label} ({name})")
            continue
        run = runs[-1]
        loss = {int(row["optimizer_step"]): row["train_loss"] for row in run.scan_history(keys=["optimizer_step", "train_loss"])}
        rows = list(run.scan_history(keys=["optimizer_step", *RMS_KEYS]))
        rms_steps = [int(r["optimizer_step"]) for r in rows]
        rms = {key: [r.get(key) for r in rows] for key in RMS_KEYS}
        cache[label] = {
            "batch_size": run.config.get("batch_size", 64),
            "steps": sorted(loss),
            "loss": [loss[s] for s in sorted(loss)],
            "rms_steps": rms_steps,
            "rms": rms,
            "val_loss": run.summary.get("val_loss"),
        }
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache))
        print(f"fetched {label}")
    return cache


def rolling_median(x, window=51):
    pad = window // 2
    padded = np.pad(x, pad, mode="edge")
    return np.array([np.median(padded[i : i + window]) for i in range(len(x))])


def roughness(entry):
    loss = np.array(entry["loss"], dtype=float)
    n = len(loss)
    segment = loss[int(0.4 * n) : int(0.6 * n)]
    residual = segment - rolling_median(segment)
    full_residual = loss[n // 20 :] - rolling_median(loss[n // 20 :])
    return float(np.std(residual)), int(np.sum(full_residual > 0.1))


def smooth(x, window=101):
    x = np.asarray(x, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(np.pad(x, window // 2, mode="edge"), kernel, mode="valid")


def p5_macro(cache):
    groups = {
        "Learning rate": ["LR 7.5e-4", "default", "LR 1.2e-2", "LR 0.1"],
        "Momentum (β1)": ["β1 0.5", "default", "β1 0.98"],
        "Batch size (x = tokens)": ["bs 16", "default", "bs 128", "bs 256"],
        "LR schedule": ["default", "cosine", "WSD 0.2", "constant"],
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for ax, (title, labels) in zip(axes.flat, groups.items(), strict=True):
        for label in labels:
            if label not in cache:
                continue
            entry = cache[label]
            steps = np.array(entry["steps"]) + 1
            x = steps * entry["batch_size"] * 1024 if "Batch" in title else steps
            ax.plot(x, smooth(entry["loss"]), linewidth=1.5, label=f"{label.replace('default', 'default d8')} ({entry['val_loss']:.3f})")
        ax.set_xscale("log")
        ax.set_xlim((2e6, 7e8) if "Batch" in title else (30, 1.1e4))
        ax.set_ylim(2.8, 4.5)
        ax.set_title(title)
        ax.set_xlabel("Training tokens" if "Batch" in title else "Optimizer step")
        ax.set_ylabel("Train loss (101-step mean)")
        ax.grid(True, linestyle=":", alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "p5_macro.png")
    plt.close(fig)


def p5_micro(cache):
    rows = sorted(((label, *roughness(entry)) for label, entry in cache.items()), key=lambda r: r[1])
    print("\nP5 roughness (sd of residual, 40-60% of training) and spikes > 0.1:")
    for label, rough, spikes in rows:
        print(f"  {label:<14} {rough:.4f}  spikes {spikes}")
    fig, (ax_bar, ax_zoom) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.2, 1]})
    labels = [r[0] for r in rows]
    ax_bar.barh(labels, [r[1] for r in rows], color=["#e76f51" if l.startswith("bs") else "#7030A0" for l in labels])
    ax_bar.set_xlabel("Roughness: sd(train loss − rolling median), 40–60% of training")
    ax_bar.set_title("Micro-structure: what makes loss curves noisy")
    ax_bar.grid(True, axis="x", linestyle=":", alpha=0.3)
    for label in ("bs 16", "default", "bs 256", "LR 1.2e-2"):
        if label in cache:
            entry = cache[label]
            n = len(entry["loss"])
            s = slice(int(0.5 * n), int(0.5 * n) + 300 * 64 // entry["batch_size"])
            tokens = (np.array(entry["steps"][s]) + 1) * entry["batch_size"] * 1024 / 1e6
            ax_zoom.plot(tokens, entry["loss"][s], linewidth=0.8, label=label)
    ax_zoom.set_xlabel("Training tokens (M), mid-training window")
    ax_zoom.set_ylabel("Raw train loss")
    ax_zoom.set_title("Raw loss, same token window")
    ax_zoom.legend(frameon=False, fontsize=8)
    ax_zoom.grid(True, linestyle=":", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "p5_micro.png")
    plt.close(fig)
    return rows


def p6_default(cache):
    entry = cache["default"]
    steps = np.array(entry["rms_steps"])
    marks = {"start (step 0)": 0, "middle": len(steps) // 2, "end": len(steps) - 1}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    series = {
        "Residual-stream activation RMS (post-block)": [f"logging/rms/model.layers.{i}/activation" for i in LAYERS],
        "Gradient RMS (pre-clip): q_proj / down_proj": None,
        "Parameter RMS: q_proj / down_proj": None,
    }
    for ax, title in zip(axes, series, strict=True):
        for (mark, idx), style in zip(marks.items(), ("-", "--", ":"), strict=True):
            if "Residual" in title:
                ys = [entry["rms"][f"logging/rms/model.layers.{i}/activation"][idx] for i in LAYERS]
                ax.plot([i + 1 for i in LAYERS], ys, style, marker="o", color="#7030A0", label=mark)
            else:
                kind = "gradient" if "Gradient" in title else "parameter"
                for module, color in (("self_attn.q_proj", "#3a7bd5"), ("mlp.down_proj", "#e76f51")):
                    ys = [entry["rms"][f"logging/rms/model.layers.{i}.{module}/{kind}"][idx] for i in LAYERS]
                    ax.plot([i + 1 for i in LAYERS], ys, style, marker="o", color=color,
                            label=f"{module.split('.')[-1]} {mark}")
        ax.set_yscale("log")
        ax.set_xlabel("Layer")
        ax.set_title(title, fontsize=10.5)
        ax.grid(True, linestyle=":", alpha=0.3)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "p6_default_profiles.png")
    plt.close(fig)
    print("\nP6 default residual RMS (layers 1,3,6) at start/middle/end:")
    for mark, idx in marks.items():
        print(f"  {mark:<14}", [round(entry["rms"][f'logging/rms/model.layers.{i}/activation'][idx], 3) for i in (0, 2, 5)])


def p6_interventions(cache):
    compare = ["default", "LR 7.5e-4", "LR 1.2e-2", "wd 0.025", "wd 2.0", "constant", "warmup 0.25%", "β2 0.99", "no clipping", "dropout 0.2", "bs 256"]
    keys = {
        "Global parameter RMS": "logging/rms/global/parameter",
        "Global gradient RMS (pre-clip)": "logging/rms/global/gradient",
        "Last-layer residual RMS": "logging/rms/model.layers.7/activation",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    summary = {}
    for ax, (title, key) in zip(axes, keys.items(), strict=True):
        for label in compare:
            if label not in cache:
                continue
            entry = cache[label]
            tokens = (np.array(entry["rms_steps"]) + 1) * entry["batch_size"] * 1024
            ys = np.array([np.nan if v is None else v for v in entry["rms"][key]], dtype=float)
            ax.plot(tokens, ys, linewidth=1.3 if label != "default" else 2.4,
                    color="black" if label == "default" else None, label=label)
            summary.setdefault(label, {})[title] = (float(ys[1]) if len(ys) > 1 else np.nan, float(ys[len(ys) // 2]), float(ys[-1]))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Training tokens")
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.3)
    axes[0].legend(frameon=False, fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "p6_interventions.png")
    plt.close(fig)
    print("\nP6 (early, middle, end) per intervention:")
    for label, stats in summary.items():
        print(f"  {label:<13}", "  ".join(f"{t.split()[1][:5]}: " + "/".join(f"{v:.3g}" for v in vals) for t, vals in stats.items()))
    return summary


def main() -> None:
    set_style()
    cache = fetch()
    p5_macro(cache)
    p5_micro(cache)
    p6_default(cache)
    p6_interventions(cache)
    print(f"\n-> {PLOT_DIR}/p5_macro.png, p5_micro.png, p6_default_profiles.png, p6_interventions.png")


if __name__ == "__main__":
    main()
