"""Problem 2(b)/(c): slopes and power-law breakdowns on both scaling axes.

Each series is fit with a straight line in log-log space, log L = c - slope * log x.
The slope is the scaling exponent; R^2 and the max |residual| in log space measure
how far the series is from the power-law (linear) regime. Diverged runs (non-finite
final loss) are drawn as X markers at the top of the axis and excluded from fits.
"""

import math
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from experiments.a1_basics.p2_bc_scaling_interventions import (
    PART_B,
    PART_C,
    run_config,
)
from experiments.a1_basics.p2_scaling_law_reliability import build_runs as build_2a_runs
from experiments.hparam_sweeps.plot_hparam_sweeps import latest_runs_by_name
from experiments.lr_tuning.plot_lr_tuning import DARK, GRAY, metric_trajectory, set_style
from model_config import depth_model_config
from modeling import AutoregressiveLM
from train import TrainConfig, training_run_name
from utils import parameter_count


PLOT_DIR = Path(__file__).resolve().parent / "plots"
CONTEXT = TrainConfig().train_dataset.context_length
COLORS = ["#7030A0", "#e76f51", "#2a9d8f", "#3a7bd5", "#e9c46a", "#264653", "#d62828"]


def params(depth: int) -> int:
    with torch.device("meta"):
        return parameter_count(AutoregressiveLM(depth_model_config(depth)))


def baseline_2a(depth: int, **variant):
    for config in build_2a_runs(range(depth, depth + 1)):
        if (config.learning_rate, config.lr_schedule, config.dropout) == (
            variant.get("learning_rate", 0.003),
            variant.get("lr_schedule", "linear"),
            variant.get("dropout", 0.0),
        ):
            return config
    raise KeyError(depth)


class Results:
    def __init__(self, runs_by_name):
        self.runs_by_name = runs_by_name

    def run(self, config):
        return self.runs_by_name.get(training_run_name(config))

    def final(self, config):
        """(val_loss, status): status is 'ok', 'diverged', or 'missing'."""
        run = self.run(config)
        if run is None or run.state != "finished":
            return math.nan, "missing"
        value = run.summary.get("val_loss")
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = math.nan
        if not math.isfinite(value) or value > 20:
            return math.nan, "diverged"
        _, losses = metric_trajectory(run, "val_loss")
        return float(losses[-1]), "ok"


def fit_line(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    mask = np.isfinite(ys)
    if mask.sum() < 2:
        return None
    lx, ly = np.log(xs[mask]), np.log(ys[mask])
    slope, intercept = np.polyfit(lx, ly, 1)
    residual = ly - (intercept + slope * lx)
    total = np.sum((ly - ly.mean()) ** 2)
    r2 = 1 - np.sum(residual**2) / total if total > 0 else float("nan")
    return {
        "slope": -slope,
        "r2": r2,
        "max_resid": float(np.max(np.abs(residual))),
        "predict": lambda x: np.exp(intercept) * np.asarray(x, float) ** slope,
    }


def series_from(results, xs, configs):
    values, statuses = zip(*(results.final(config) for config in configs), strict=True)
    return np.asarray(xs, float), np.asarray(values, float), list(statuses)


def draw(ax, label, xs, ys, statuses, color, rows, axis_name):
    fit = fit_line(xs, ys)
    ok = np.isfinite(ys)
    ax.scatter(xs[ok], ys[ok], s=45, color=color, edgecolor=DARK, zorder=5)
    if fit is not None:
        grid = np.logspace(np.log10(xs[ok].min()), np.log10(xs[ok].max()), 50)
        ax.plot(grid, fit["predict"](grid), color=color, linewidth=1.2, linestyle="--", alpha=0.8)
        ax.plot(xs[ok], ys[ok], color=color, linewidth=1.4,
                label=f"{label}  (slope {fit['slope']:.3f}, R² {fit['r2']:.3f})")
    else:
        ax.plot([], [], color=color, label=f"{label}  (no fit)")
    diverged = [x for x, s in zip(xs, statuses, strict=True) if s == "diverged"]
    if diverged:
        ax.scatter(diverged, [ax.get_ylim()[1]] * len(diverged), marker="X", s=90, color=color,
                   edgecolor=DARK, zorder=7, clip_on=False)
    rows.append((axis_name, label, xs, ys, statuses, fit))


def finish_axis(ax, title, xlabel):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Final validation loss")
    ax.grid(True, which="both", linestyle=":", alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="best")


def print_rows(rows):
    for axis_name, label, xs, ys, statuses, fit in rows:
        points = ", ".join(
            f"{x:.3g}:{'DIVERGED' if s == 'diverged' else ('-' if s == 'missing' else f'{y:.3f}')}"
            for x, y, s in zip(xs, ys, statuses, strict=True)
        )
        summary = "no fit" if fit is None else f"slope {fit['slope']:.4f}  R2 {fit['r2']:.4f}  max|resid| {fit['max_resid']:.4f}"
        print(f"  [{axis_name}] {label:<34} {summary}\n      {points}")


def group_configs(specs, group):
    return [(depth, overrides, run_config(g, depth, overrides)) for g, depth, overrides in specs if g == group]


def main() -> None:
    set_style()
    results = Results(latest_runs_by_name(wandb.Api(timeout=60)))
    depth_params = {d: params(d) for d in range(4, 10)}
    tokens = lambda n: n * CONTEXT  # noqa: E731

    # ---------- (b) ----------
    rows_b = []
    fig, (ax_n, ax_d) = plt.subplots(1, 2, figsize=(15, 5.2))
    depths = list(range(4, 10))
    draw(ax_n, "baseline (2a)", *series_from(results, [depth_params[d] for d in depths], [baseline_2a(d) for d in depths]),
         COLORS[0], rows_b, "model")
    draw(ax_n, "dropout 0.2 (2a)", *series_from(results, [depth_params[d] for d in depths], [baseline_2a(d, dropout=0.2) for d in depths]),
         COLORS[1], rows_b, "model")
    for color, group, label in ((COLORS[2], "b-tuned", "tuned: β2 .99, warmup 4%, wd .2"),
                                (COLORS[3], "b-bs256", "bs 256, LR 6e-3, warmup 4%")):
        items = group_configs(PART_B, group)
        draw(ax_n, label, *series_from(results, [depth_params[d] for d, _, _ in items], [c for _, _, c in items]),
             color, rows_b, "model")
    finish_axis(ax_n, "(b) Model-size scaling (614M tokens)", "Parameters N")

    data_n = (150_000, 300_000, 600_000)
    base_data = [c for _, _, c in group_configs(PART_B, "b-data")] + [baseline_2a(6)]
    drop_data = [c for _, _, c in group_configs(PART_B, "b-data-dropout")] + [baseline_2a(6, dropout=0.2)]
    draw(ax_d, "d6 baseline", *series_from(results, [tokens(n) for n in data_n], base_data), COLORS[0], rows_b, "data")
    draw(ax_d, "d6 dropout 0.2", *series_from(results, [tokens(n) for n in data_n], drop_data), COLORS[1], rows_b, "data")
    finish_axis(ax_d, "(b) Data scaling (d6, 16M params)", "Training tokens D")
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / "p2b_slopes.png")
    plt.close(fig)
    print("(b)")
    print_rows(rows_b)

    # ---------- (c) ----------
    rows_c = []
    fig, (ax_n, ax_d) = plt.subplots(1, 2, figsize=(15, 5.2))
    ax_n.set_ylim(2.8, 8.0)
    draw(ax_n, "baseline (2a)", *series_from(results, [depth_params[d] for d in depths], [baseline_2a(d) for d in depths]),
         COLORS[0], rows_c, "model")
    for color, group, label in ((COLORS[1], "c-lr0.1", "LR 0.1"),
                                (COLORS[2], "c-unstable", "LR .03, no warmup, no clip"),
                                (COLORS[3], "c-repeat16", "37.5k seqs x 16 epochs"),
                                (COLORS[4], "c-bs1024", "bs 1024 (586 steps)"),
                                (COLORS[5], "c-wd2", "weight decay 2.0")):
        items = group_configs(PART_C, group)
        draw(ax_n, label, *series_from(results, [depth_params[d] for d, _, _ in items], [c for _, _, c in items]),
             color, rows_c, "model")
    finish_axis(ax_n, "(c) Model-size axis: breaking interventions", "Parameters N")

    tiny = group_configs(PART_C, "c-tiny-data")
    data_all = [(o["num_train_sequences"], c) for _, o, c in tiny] + list(zip(data_n, base_data, strict=True))
    draw(ax_d, "d6, single epoch", *series_from(results, [tokens(n) for n, _ in data_all], [c for _, c in data_all]),
         COLORS[0], rows_c, "data")
    repeat_d6 = [c for d, _, c in group_configs(PART_C, "c-repeat16") if d == 6]
    value, status = results.final(repeat_d6[0])
    if status == "ok":
        ax_d.scatter([tokens(37_500)], [value], marker="D", s=70, color=COLORS[3], edgecolor=DARK, zorder=6,
                     label=f"d6, 38M unique tokens x16 epochs: {value:.3f}")
    for n, config in data_all:
        value, status = results.final(config)
        if status == "ok":
            steps = n // 64
            ax_d.annotate(f"{steps:,} steps", (tokens(n), value), xytext=(4, 6), textcoords="offset points",
                          fontsize=7.5, color=GRAY)
    finish_axis(ax_d, "(c) Data axis down to tiny data (d6)", "Training tokens D (unique)")
    fig.savefig(PLOT_DIR / "p2c_breaks.png")
    plt.close(fig)
    print("(c)")
    print_rows(rows_c)

    # ---------- (c) dynamics ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (title, group, depths_shown) in zip(axes, (
        ("LR 0.1: val loss vs step", "c-lr0.1", (4, 6, 8, 9)),
        ("Repeated data (x16): train vs val", "c-repeat16", (4, 6, 8, 9)),
        ("bs 1024: val loss vs tokens", "c-bs1024", (4, 6, 8)),
    ), strict=True):
        for color, (depth, _, config) in zip(COLORS, [i for i in group_configs(PART_C, group) if i[0] in depths_shown], strict=False):
            run = results.run(config)
            if run is None:
                continue
            steps, val = metric_trajectory(run, "val_loss")
            x = (steps + 1) * config.batch_size * CONTEXT if group == "c-bs1024" else steps + 1
            ax.plot(x, val, color=color, linewidth=1.5, label=f"d{depth} val")
            if group == "c-repeat16":
                t_steps, train = metric_trajectory(run, "train_loss")
                ax.plot(t_steps + 1, train, color=color, linewidth=1.0, linestyle=":", label=f"d{depth} train")
        ax.set_xscale("log")
        ax.set_ylim(2.0, 8.0)
        ax.set_title(title)
        ax.set_xlabel("Training tokens" if group == "c-bs1024" else "Optimizer step")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle=":", alpha=0.3)
        ax.legend(frameon=False, fontsize=7.5, ncol=2)
    fig.savefig(PLOT_DIR / "p2c_dynamics.png")
    plt.close(fig)
    print(f"\n-> {PLOT_DIR.resolve()}/p2b_slopes.png, p2c_breaks.png, p2c_dynamics.png")


if __name__ == "__main__":
    main()
