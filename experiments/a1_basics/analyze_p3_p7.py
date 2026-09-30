"""Problems 3, 5 (beta1), 7 and the Problem 1 example questions: final losses and spreads."""

import json
import statistics as st
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import wandb

from experiments.lr_tuning.plot_lr_tuning import metric_trajectory, set_style


PROJECT = "whitedeer-stanford-university/assignments"
PLOT_DIR = Path(__file__).resolve().parent / "plots"
CACHE = PLOT_DIR / "p3_p7_cache.json"
D8 = "model-d8-lr0.003-tok614M"
V = "p3-variation"

GROUPS = {
    # P3 (a)/(c): all three sources varied. Seed 42 = the default run.
    "joint (d8)": [f"{D8}-modal", *[f"{D8}-ds{s}-ms{s}-{V}-joint" for s in range(1, 6)]],
    # P3 (b): one source at a time.
    "model seed only": [f"{D8}-modal", f"{D8}-ms1-sched-sweeps-v1-seeds", f"{D8}-ms2-sched-sweeps-v1-seeds", f"{D8}-ms3-{V}-model-seed"],
    "data seed only": [f"{D8}-modal", *[f"{D8}-ds{s}-{V}-data-seed" for s in (1, 2, 3)]],
    "hardware only (H100)": [f"{D8}-modal", f"{D8}-scaling-v1", f"{D8}-{V}-hw-rep1", f"{D8}-{V}-hw-rep2"],
    "deterministic (same H100)": [f"{D8}-deterministic-deterministic-reference-1", f"{D8}-deterministic-deterministic-reference-2"],
    # P3 (c)
    "joint, bs 16": [f"model-d8-lr0.003-bs16-tok614M-ds{s}-ms{s}-{V}-joint-bs16" for s in range(1, 6)],
    "joint, LR 0.009": [f"model-d8-lr0.009-tok614M-ds{s}-ms{s}-{V}-joint-lr009" for s in range(1, 6)],
    "joint, d4": [f"model-d4-lr0.003-tok614M-ds{s}-ms{s}-{V}-joint-d4" for s in (1, 2, 3)],
    "joint, constant LR": [f"{D8}-ds{s}-ms{s}-constant-{V}-joint-constant" for s in (1, 2, 3)],
    "joint, LR 0.03": [f"model-d8-lr0.03-tok614M-ds{s}-ms{s}-{V}-joint-lr003" for s in (1, 2, 3)],
    "joint, dropout 0.2": [f"{D8}-dropout0.2-ds{s}-ms{s}-{V}-joint-dropout" for s in (1, 2, 3)],
}
SINGLES = {
    "A100 (nondeterministic)": f"{D8}-{V}-a100",
    "A100 (deterministic)": f"{D8}-deterministic-{V}-a100",
    "H100 deterministic ref": f"{D8}-deterministic-deterministic-reference-1",
    "beta1 0.5": f"{D8}-b10.5-p5-beta1",
    "beta1 0.98": f"{D8}-b10.98-p5-beta1",
    "easy A (LR 3e-3, bs 64)": f"{D8}-modal",
    "easy B (LR 1e-3, bs 128)": "model-d8-lr0.001-bs128-tok614M-p1-example-easy",
    "easy C (LR 9e-3, bs 128)": "model-d8-lr0.009-bs128-tok614M-p1-example-easy",
    # The first medium-example batch was contaminated (W&B run xdlnq96n); these are clean reruns.
    "medium A (wsd0.2, warmup 0)": "model-d8-lr0.009-tok614M-wd1.0-warmup0.0-wsd0.2-p1-example-medium-rerun",
    "medium B (wsd0.2, warmup 0.2)": "model-d8-lr0.009-tok614M-wd1.0-warmup0.2-wsd0.2-p1-example-medium-rerun",
    "medium C (linear, warmup 0.1)": "model-d8-lr0.009-tok614M-wd1.0-warmup0.1-p1-example-medium-rerun",
    "P7 d4 (4x repeat)": "model-d4-lr0.003-epochs4.0-tok154M-p7-repeat4",
    "P7 d6 (4x repeat)": "model-d6-lr0.003-epochs4.0-tok154M-p7-repeat4-rerun",
    "P7 d9 (4x repeat)": "model-d9-lr0.003-epochs4.0-tok154M-p7-repeat4",
}


def fetch():
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    api = wandb.Api(timeout=60)
    names = {n for group in GROUPS.values() for n in group} | set(SINGLES.values())
    for name in sorted(names - set(cache)):
        runs = [r for r in api.runs(PROJECT, filters={"display_name": name}) if r.state == "finished"]
        if not runs:
            print(f"missing: {name}")
            continue
        run = runs[-1]
        steps, val = metric_trajectory(run, "val_loss")
        train = [r["train_loss"] for r in run.scan_history(keys=["optimizer_step", "train_loss"])]
        cache[name] = {"final": float(val[-1]), "val_steps": steps.tolist(), "val": val.tolist(),
                       "train_tail": train[-2000:], "summary": run.summary.get("val_loss")}
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache))
    return cache


def spread(values):
    return (st.mean(values), st.stdev(values) if len(values) > 1 else float("nan"), max(values) - min(values))


def main() -> None:
    set_style()
    cache = fetch()
    print("P3 groups: n, mean, sd, range")
    stats = {}
    for label, names in GROUPS.items():
        values = [cache[n]["final"] for n in names if n in cache]
        if len(values) >= 2:
            stats[label] = (values, *spread(values))
            print(f"  {label:<26} n={len(values)} mean {stats[label][1]:.4f} sd {stats[label][2]:.4f} range {stats[label][3]:.4f}  {[round(v, 4) for v in values]}")
    print("\nSingles:")
    for label, name in SINGLES.items():
        if name in cache:
            print(f"  {label:<32} {cache[name]['final']:.4f}")

    # Train-loss curve variability across joint seeds: sd across seeds at each step, late training.
    fig, (ax_dist, ax_curve) = plt.subplots(1, 2, figsize=(14, 4.8))
    labels = [l for l in stats]
    for i, label in enumerate(labels):
        values, mean, sd = stats[label][0], stats[label][1], stats[label][2]
        # Spread dots sideways so runs with near-identical losses don't hide each other.
        offsets = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else [0.0]
        ax_dist.scatter([i + o for o in offsets], [v - mean for v in values], s=35, color="#7030A0", edgecolor="#111", zorder=5)
        ax_dist.errorbar(i + 0.18, 0, yerr=sd, fmt="s", color="#e76f51", capsize=4)
    ax_dist.axhline(0, color="gray", linewidth=0.8)
    ax_dist.set_xticks(range(len(labels)))
    ax_dist.set_xticklabels([l.replace(" (", "\n(").replace(", ", ",\n").replace(" only", "\nonly").replace(" seed", "\nseed") for l in labels], fontsize=7.2)
    ax_dist.set_ylabel("Final val loss − group mean")
    ax_dist.set_title("Spread within each group (dots = runs, bar = ±1 sd)")
    ax_dist.grid(True, axis="y", linestyle=":", alpha=0.3)
    for label, color in (("joint (d8)", "#7030A0"), ("joint, bs 16", "#e76f51"), ("joint, LR 0.009", "#3a7bd5"), ("joint, constant LR", "#2a9d8f")):
        curves = [np.array(cache[n]["val"]) for n in GROUPS[label] if n in cache]
        if len(curves) >= 2:
            length = min(len(c) for c in curves)
            curves = np.stack([c[:length] for c in curves])
            ax_curve.plot(np.arange(length) + 1, curves.std(axis=0, ddof=1), color=color, label=label)
    ax_curve.set_yscale("log")
    ax_curve.set_xlabel("Eval index (1-100, evenly spaced through training)")
    ax_curve.set_ylabel("sd of val loss across seeds")
    ax_curve.set_title("How the seed-to-seed spread evolves during training")
    ax_curve.grid(True, linestyle=":", alpha=0.3)
    ax_curve.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "p3_variation.png")
    plt.close(fig)
    print(f"\n-> {PLOT_DIR}/p3_variation.png")


if __name__ == "__main__":
    main()
