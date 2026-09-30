"""Fit loss-vs-parameter scaling laws on d4-d7 and extrapolate to d8, d9, d20.

Two fits per variant, both in parameter count N:
  power law:          L(N) = A * N^-alpha
  power law + floor:  L(N) = E + A * N^-alpha   (E = irreducible loss)
Runs at depths outside FIT_DEPTHS (d8, d9 after stage 2) are plotted as
held-out points so the predictions can be checked.
"""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from experiments.a1_basics.p2_scaling_law_reliability import VARIANTS, build_runs
from experiments.hparam_sweeps.plot_hparam_sweeps import latest_runs_by_name
from experiments.lr_tuning.plot_lr_tuning import DARK, GRAY, metric_trajectory, set_style
from model_config import depth_model_config
from modeling import AutoregressiveLM
from train import training_run_name
from utils import parameter_count


FIT_DEPTHS = range(4, 8)
PREDICT_DEPTHS = (8, 9, 20)
PLOT_DIR = Path(__file__).resolve().parent / "plots"
VARIANT_LABELS = {
    (0.003, "linear", 0.0): "baseline (linear, LR 3e-3)",
    (0.003, "constant", 0.0): "constant LR",
    (0.003, "linear", 0.2): "dropout 0.2",
    (0.03, "linear", 0.0): "LR 0.03",
}
VARIANT_COLORS = ["#7030A0", "#2a9d8f", "#e76f51", "#3a7bd5"]


def params(depth: int) -> int:
    with torch.device("meta"):
        return parameter_count(AutoregressiveLM(depth_model_config(depth)))


def fit_power_law(n, loss):
    slope, intercept = np.polyfit(np.log(n), np.log(loss), 1)
    return lambda x: np.exp(intercept) * np.asarray(x, dtype=float) ** slope, {"alpha": -slope, "E": 0.0}


def fit_power_law_with_floor(n, loss):
    best = None
    for floor in np.linspace(0.0, loss.min() - 1e-3, 2000):
        slope, intercept = np.polyfit(np.log(n), np.log(loss - floor), 1)
        predicted = floor + np.exp(intercept) * n**slope
        error = np.sum((predicted - loss) ** 2)
        if best is None or error < best[0]:
            best = (error, floor, slope, intercept)
    _, floor, slope, intercept = best
    return (
        lambda x: floor + np.exp(intercept) * np.asarray(x, dtype=float) ** slope,
        {"alpha": -slope, "E": floor},
    )


def collect(runs_by_name):
    results = {variant: {} for variant in VARIANTS}
    for config in build_runs():
        run = runs_by_name.get(training_run_name(config))
        if run is None or run.state != "finished":
            continue
        _, losses = metric_trajectory(run, "val_loss")
        if losses.size:
            variant = (config.learning_rate, config.lr_schedule, config.dropout)
            depth = config.model_config.num_hidden_layers
            results[variant][depth] = float(losses[-1])
    return results


def main() -> None:
    set_style()
    results = collect(latest_runs_by_name(wandb.Api(timeout=60)))
    depth_params = {depth: params(depth) for depth in (*range(4, 10), 20)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    grid = np.logspace(np.log10(depth_params[4] * 0.8), np.log10(depth_params[20] * 1.2), 200)
    header = "  ".join(f"{'d' + str(d):>7}" for d in PREDICT_DEPTHS)
    for ax, (fit_name, fit) in zip(
        axes,
        (("Power law  L = A·N^-α", fit_power_law), ("Power law + floor  L = E + A·N^-α", fit_power_law_with_floor)),
        strict=True,
    ):
        print(f"\n{fit_name}  (fit on d{FIT_DEPTHS[0]}-d{FIT_DEPTHS[-1]})")
        print(f"  {'variant':<28} {'alpha':>6} {'E':>6}  {header}   actual d8/d9")
        for color, variant in zip(VARIANT_COLORS, VARIANTS, strict=True):
            points = results[variant]
            fit_depths = [d for d in FIT_DEPTHS if d in points]
            if len(fit_depths) < 3:
                continue
            n = np.array([depth_params[d] for d in fit_depths], dtype=float)
            loss = np.array([points[d] for d in fit_depths])
            predict, fitted = fit(n, loss)
            label = VARIANT_LABELS[variant]
            ax.plot(grid, predict(grid), color=color, linewidth=1.4, alpha=0.8)
            ax.scatter(n, loss, s=45, color=color, edgecolor=DARK, zorder=5, label=label)
            held_out = [d for d in points if d not in FIT_DEPTHS]
            if held_out:
                ax.scatter(
                    [depth_params[d] for d in held_out],
                    [points[d] for d in held_out],
                    s=70, facecolors="white", edgecolor=color, linewidth=2, zorder=6,
                )
            predictions = "  ".join(f"{predict(depth_params[d]):7.3f}" for d in PREDICT_DEPTHS)
            actual = ", ".join(f"d{d} {points[d]:.3f}" for d in sorted(held_out)) or "-"
            print(f"  {label:<28} {fitted['alpha']:6.3f} {fitted['E']:6.3f}  {predictions}   {actual}")
        for depth in PREDICT_DEPTHS:
            ax.axvline(depth_params[depth], color=GRAY, linestyle=":", linewidth=1)
            ax.text(depth_params[depth], ax.get_ylim()[1], f" d{depth}", fontsize=8.5, color=GRAY, va="top")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Parameters N")
        ax.set_ylabel("Final validation loss")
        ax.set_title(fit_name)
        ax.grid(True, which="major", linestyle=":", alpha=0.3)
        ax.legend(frameon=False, fontsize=8.5, title="filled = fit, hollow = held out", title_fontsize=8.5)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOT_DIR / "p2_scaling_fits.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\n-> {out_path.resolve()}")


if __name__ == "__main__":
    main()
