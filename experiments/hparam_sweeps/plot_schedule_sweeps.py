from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import wandb

from experiments.hparam_sweeps.launch_hparam_sweeps import BASELINE_RUN_NAME_SUFFIX
from experiments.hparam_sweeps.launch_schedule_sweeps import (
    EXPERIMENT_KEY,
    block_config,
)
from experiments.hparam_sweeps.plot_hparam_sweeps import latest_runs_by_name
from experiments.hparam_sweeps.plot_pair_sweeps import grid_config
from experiments.lr_tuning.plot_lr_tuning import (
    DARK,
    GRAY,
    LIGHT_BLUE,
    PURPLE,
    RED,
    metric_trajectory,
    set_style,
)
from lr_schedules import build_scheduler
from train import TrainConfig, training_run_name


PLOT_DIR = Path(__file__).resolve().parent / "plots"
SCHEDULES = ("linear", "cos", "wsd0.2", "constant")
SCHEDULE_COLORS = {
    "linear": PURPLE,
    "cos": "#3a7bd5",
    "wsd0.2": "#2a9d8f",
    "constant": LIGHT_BLUE,
}


class Runs:
    def __init__(self, runs_by_name: dict):
        self.runs_by_name = runs_by_name
        self.missing = []

    def run(self, config):
        run_name = training_run_name(config)
        run = self.runs_by_name.get(run_name)
        if run is None or run.state != "finished":
            self.missing.append(run_name)
            return None
        return run

    def loss(self, config) -> float:
        run = self.run(config)
        if run is None:
            return np.nan
        _, losses = metric_trajectory(run, "val_loss")
        return float(losses[-1]) if losses.size else np.nan


def default_config():
    return TrainConfig(run_name_suffix=BASELINE_RUN_NAME_SUFFIX)


def schedule_config(schedule: str, learning_rate: float):
    if schedule == "linear":
        return grid_config("learning_rate-weight_decay", learning_rate, 0.1)
    return block_config(
        "schedule-lr", {"lr_schedule": schedule, "learning_rate": learning_rate}
    )


def lr_curve(schedule: str, total_steps: int = 9375, warmup_percent: float = 0.01):
    import torch

    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = build_scheduler(
        optimizer, schedule, int(total_steps * warmup_percent), total_steps
    )
    values = []
    for _ in range(total_steps):
        values.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return np.array(values)


def style_axis(ax, title, xlabel, ylabel="Final validation loss"):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", linestyle=":", alpha=0.3)


def plot_lines(ax, xs, series: dict, log_x=True):
    for label, ys in series.items():
        ys = np.array(ys, dtype=float)
        mask = ~np.isnan(ys)
        color = SCHEDULE_COLORS.get(label, GRAY)
        ax.plot(np.array(xs)[mask], ys[mask], color=color, linewidth=1.6, label=label)
        ax.scatter(np.array(xs)[mask], ys[mask], s=45, color=color, edgecolor=DARK, zorder=5)
        if mask.any():
            best = int(np.nanargmin(ys))
            ax.scatter([xs[best]], [ys[best]], s=120, facecolors="none", edgecolor=RED, linewidth=1.6, zorder=6)
    if log_x:
        ax.set_xscale("log")
        ax.set_xticks(list(xs))
        ax.set_xticklabels([f"{x:g}" for x in xs])
        ax.minorticks_off()
    ax.legend(frameon=False, fontsize=8.5)


def print_table(title, row_label, columns, rows: dict):
    print(f"\n{title}")
    print(f"  {row_label:>14}: " + "  ".join(f"{c:>8g}" for c in columns))
    for name, values in rows.items():
        cells = "  ".join("       -" if np.isnan(v) else f"{v:8.4f}" for v in values)
        print(f"  {name:>14}: {cells}")


def save(fig, name: str) -> Path:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOT_DIR / f"{EXPERIMENT_KEY}_{name}.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  -> {out_path.resolve()}")
    return out_path


def plot_schedules(runs: Runs) -> None:
    learning_rates = (1.5e-3, 3e-3, 6e-3)
    table = {
        schedule: [runs.loss(schedule_config(schedule, lr)) for lr in learning_rates]
        for schedule in SCHEDULES
    }
    print_table("Schedule x LR", "schedule \\ LR", learning_rates, table)

    fig, (ax_shape, ax_curves, ax_final) = plt.subplots(1, 3, figsize=(15, 4.2))
    for schedule in SCHEDULES:
        ax_shape.plot(lr_curve(schedule), color=SCHEDULE_COLORS[schedule], linewidth=1.8, label=schedule)
    style_axis(ax_shape, "Schedule Shapes", "Optimizer step", "LR / peak LR")
    ax_shape.legend(frameon=False, fontsize=8.5)

    for schedule in SCHEDULES:
        run = runs.run(schedule_config(schedule, 3e-3))
        if run is None:
            continue
        steps, losses = metric_trajectory(run, "val_loss")
        ax_curves.plot(steps + 1, losses, color=SCHEDULE_COLORS[schedule], linewidth=1.6, label=schedule)
    ax_curves.set_ylim(2.85, 3.4)
    style_axis(ax_curves, "Val Loss at LR 3e-3", "Optimizer step", "Validation loss")
    ax_curves.legend(frameon=False, fontsize=8.5)

    plot_lines(ax_final, learning_rates, table)
    style_axis(ax_final, "Final Loss vs Peak LR (red = best)", "Peak learning rate")
    save(fig, "schedule_lr")


def plot_interactions(runs: Runs) -> None:
    decay_fractions = (0.1, 0.2, 0.4, 1.0)
    wsd_losses = [
        runs.loss(block_config("wsd-decay", {"lr_schedule": f"wsd{f}", "learning_rate": 3e-3}))
        if f in (0.1, 0.4)
        else runs.loss(schedule_config("wsd0.2", 3e-3))
        if f == 0.2
        else runs.loss(default_config())
        for f in decay_fractions
    ]
    print_table("WSD decay fraction (LR 3e-3; 1.0 = linear)", "", decay_fractions, {"wsd": wsd_losses})

    warmups = (0.0025, 0.01, 0.04)
    warmup_table = {"linear": [runs.loss(grid_config("learning_rate-warmup_percent", 6e-3, w)) for w in warmups]}
    for schedule in ("cos", "wsd0.2"):
        warmup_table[schedule] = [
            runs.loss(schedule_config(schedule, 6e-3))
            if w == 0.01
            else runs.loss(block_config("schedule-warmup", {"lr_schedule": schedule, "learning_rate": 6e-3, "warmup_percent": w}))
            for w in warmups
        ]
    print_table("Schedule x warmup (LR 6e-3)", "schedule \\ warmup", warmups, warmup_table)

    batch_sizes = (32, 64, 128)
    batch_table = {"linear": [runs.loss(grid_config("batch_size-learning_rate", b, 3e-3)) for b in batch_sizes]}
    for schedule in ("cos", "wsd0.2"):
        batch_table[schedule] = [
            runs.loss(schedule_config(schedule, 3e-3))
            if b == 64
            else runs.loss(block_config("schedule-batch-size", {"lr_schedule": schedule, "batch_size": b}))
            for b in batch_sizes
        ]
    print_table("Schedule x batch size (LR 3e-3)", "schedule \\ bs", batch_sizes, batch_table)

    fig, (ax_decay, ax_warmup, ax_batch) = plt.subplots(1, 3, figsize=(15, 4.2))
    plot_lines(ax_decay, decay_fractions, {"wsd0.2": wsd_losses})
    ax_decay.get_legend().remove()
    ax_decay.set_xticklabels(["0.1", "0.2", "0.4", "1.0\n(linear)"])
    style_axis(ax_decay, "WSD Decay Fraction (LR 3e-3)", "Fraction of steps spent decaying")
    plot_lines(ax_warmup, warmups, warmup_table)
    style_axis(ax_warmup, "Schedule x Warmup (LR 6e-3)", "warmup_percent")
    plot_lines(ax_batch, batch_sizes, batch_table)
    style_axis(ax_batch, "Schedule x Batch Size (LR 3e-3)", "batch_size")
    save(fig, "interactions")


def plot_optimizer(runs: Runs) -> None:
    learning_rates = (3e-3, 6e-3)
    variants = {
        "default\n(β2 0.95, clip 1)": lambda lr: default_config() if lr == 3e-3 else grid_config("learning_rate-weight_decay", lr, 0.1),
        "β2 0.9": lambda lr: block_config("optimizer-lr", {"beta2": 0.9, "learning_rate": lr}),
        "β2 0.99": lambda lr: block_config("optimizer-lr", {"beta2": 0.99, "learning_rate": lr}),
        "no clipping": lambda lr: block_config("optimizer-lr", {"grad_norm": None, "learning_rate": lr}),
    }
    table = {name.replace("\n", " "): [runs.loss(make(lr)) for lr in learning_rates] for name, make in variants.items()}
    print_table("Optimizer variants", "variant \\ LR", learning_rates, table)

    seeds = {42: runs.loss(default_config())}
    for seed in (1, 2):
        seeds[seed] = runs.loss(block_config("seeds", {"model_seed": seed}))
    values = np.array([v for v in seeds.values() if not np.isnan(v)])
    print(f"\nSeeds (default recipe): {', '.join(f'{s}: {v:.4f}' for s, v in seeds.items())}")
    print(f"  mean {values.mean():.4f}, std {values.std(ddof=1):.4f}, range {np.ptp(values):.4f}")

    bs256 = {
        "LR 3e-3, warmup 1%": runs.loss(grid_config("batch_size-learning_rate", 256, 3e-3)),
        "LR 6e-3, warmup 1%": runs.loss(grid_config("batch_size-learning_rate", 256, 6e-3)),
        "LR 6e-3, warmup 4%": runs.loss(block_config("bs256-warmup", {"batch_size": 256, "learning_rate": 6e-3, "warmup_percent": 0.04})),
    }
    print("\nBatch size 256:")
    for name, value in bs256.items():
        print(f"  {name}: {value:.4f}")

    fig, (ax_opt, ax_seed, ax_bs) = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1.5, 0.8, 1.0]})
    width = 0.38
    positions = np.arange(len(variants))
    for offset, (lr, color) in zip((-width / 2, width / 2), ((3e-3, PURPLE), (6e-3, LIGHT_BLUE)), strict=True):
        column = learning_rates.index(lr)
        heights = [row[column] for row in table.values()]
        bars = ax_opt.bar(positions + offset, heights, width, color=color, edgecolor=DARK, label=f"LR {lr:g}")
        ax_opt.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax_opt.set_xticks(positions)
    ax_opt.set_xticklabels(list(variants))
    ax_opt.set_ylim(2.88, max(np.nanmax(list(table.values())) + 0.02, 2.98))
    style_axis(ax_opt, "Optimizer Hyperparameters", "")
    ax_opt.legend(frameon=False, fontsize=8.5)

    ax_seed.scatter(np.zeros(len(values)), values, s=60, color=PURPLE, edgecolor=DARK, zorder=5)
    for seed, value in seeds.items():
        ax_seed.annotate(f"seed {seed}: {value:.4f}", (0, value), xytext=(10, 0), textcoords="offset points", va="center", fontsize=8.5)
    ax_seed.axhspan(values.min(), values.max(), color=LIGHT_BLUE, alpha=0.25)
    ax_seed.set_xlim(-0.5, 1.5)
    ax_seed.set_xticks([])
    style_axis(ax_seed, f"Seed Noise (range {np.ptp(values):.4f})", "default recipe")

    bars = ax_bs.bar(range(len(bs256)), list(bs256.values()), color=[PURPLE, LIGHT_BLUE, "#2a9d8f"], edgecolor=DARK)
    ax_bs.bar_label(bars, fmt="%.3f", fontsize=8.5, padding=2)
    ax_bs.set_xticks(range(len(bs256)))
    ax_bs.set_xticklabels([name.replace(", ", "\n") for name in bs256], fontsize=9)
    ax_bs.set_ylim(2.95, max(bs256.values()) + 0.03)
    style_axis(ax_bs, "Batch Size 256: Is It Warmup?", "")
    save(fig, "optimizer_seeds_bs256")


def main() -> None:
    set_style()
    runs = Runs(latest_runs_by_name(wandb.Api(timeout=60)))
    plot_schedules(runs)
    plot_interactions(runs)
    plot_optimizer(runs)
    if runs.missing:
        print("\nMissing or unfinished runs:")
        for run_name in sorted(set(runs.missing)):
            print(f"  {run_name}")


if __name__ == "__main__":
    main()
