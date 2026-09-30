from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import wandb

from experiments.hparam_sweeps.launch_hparam_sweeps import (
    BASELINE_RUN_NAME_SUFFIX,
    sweep_config,
)
from experiments.hparam_sweeps.launch_pair_sweeps import (
    EXCLUDED_POINTS,
    EXPERIMENT_KEY,
    PAIRS,
    is_off_axis,
    pair_config,
)
from experiments.hparam_sweeps.plot_hparam_sweeps import CMAP, latest_runs_by_name
from experiments.lr_tuning.plot_lr_tuning import (
    DARK,
    GRAY,
    RED,
    metric_trajectory,
    set_style,
)
from train import TrainConfig, training_run_name
from utils import WANDB_ENTITY, WANDB_PROJECT


PLOT_DIR = Path(__file__).resolve().parent / "plots"


def grid_config(pair_name: str, x_value, y_value):
    (x_field, _), (y_field, _) = PAIRS[pair_name]
    if is_off_axis(pair_name, x_value, y_value):
        return pair_config(pair_name, x_value, y_value)
    if x_value != getattr(TrainConfig, x_field):
        return sweep_config(x_field, x_value)
    if y_value != getattr(TrainConfig, y_field):
        return sweep_config(y_field, y_value)
    return TrainConfig(run_name_suffix=BASELINE_RUN_NAME_SUFFIX)


def collect_grid(pair_name: str, runs_by_name: dict) -> np.ndarray:
    (x_field, x_values), (y_field, y_values) = PAIRS[pair_name]
    losses = np.full((len(y_values), len(x_values)), np.nan)
    for j, y_value in enumerate(y_values):
        for i, x_value in enumerate(x_values):
            if (pair_name, x_value, y_value) in EXCLUDED_POINTS:
                continue
            run_name = training_run_name(grid_config(pair_name, x_value, y_value))
            run = runs_by_name.get(run_name)
            if run is None:
                print(f"  missing: {x_field}={x_value:g} {y_field}={y_value:g}")
                continue
            _, val_losses = metric_trajectory(run, "val_loss")
            if not val_losses.size or run.state != "finished":
                print(
                    f"  not finished: {x_field}={x_value:g} {y_field}={y_value:g} "
                    f"({run.state})"
                )
                continue
            losses[j, i] = val_losses[-1]
    return losses


def print_grid(pair_name: str, losses: np.ndarray) -> None:
    (x_field, x_values), (y_field, y_values) = PAIRS[pair_name]
    print(f"  {y_field} \\ {x_field}: " + "  ".join(f"{x:>8g}" for x in x_values))
    for y_value, row in zip(y_values, losses, strict=True):
        cells = "  ".join(
            "       -" if np.isnan(loss) else f"{loss:8.4f}" for loss in row
        )
        best = "" if np.all(np.isnan(row)) else f"   best {x_field}={x_values[int(np.nanargmin(row))]:g}"
        print(f"  {y_value:>{len(y_field) + len(x_field) + 3}g}: {cells}{best}")


def plot_grid(pair_name: str, losses: np.ndarray) -> Path:
    (x_field, x_values), (y_field, y_values) = PAIRS[pair_name]
    norm = mcolors.LogNorm(vmin=min(y_values), vmax=max(y_values))

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.2, 1.0]}
    )
    for y_value, row in zip(y_values, losses, strict=True):
        mask = ~np.isnan(row)
        if not mask.any():
            continue
        xs = np.array(x_values, dtype=float)[mask]
        color = CMAP(norm(y_value))
        ax_left.plot(xs, row[mask], color=color, linewidth=1.6, label=f"{y_value:g}")
        ax_left.scatter(xs, row[mask], s=50, color=color, edgecolor=DARK, zorder=5)
        best = int(np.argmin(row[mask]))
        ax_left.scatter(
            [xs[best]],
            [row[mask][best]],
            s=130,
            facecolors="none",
            edgecolor=RED,
            linewidth=1.8,
            zorder=6,
        )
    ax_left.set_xscale("log")
    ax_left.set_xticks(list(x_values))
    ax_left.set_xticklabels([f"{x:g}" for x in x_values])
    ax_left.minorticks_off()
    ax_left.set_xlabel(x_field)
    ax_left.set_ylabel("Final validation loss")
    ax_left.set_title(f"Loss vs {x_field} (red = best per line)")
    ax_left.grid(True, which="major", linestyle=":", alpha=0.3)
    ax_left.legend(title=y_field, frameon=False, fontsize=8.5, title_fontsize=9)

    image = ax_right.imshow(losses, cmap="viridis_r", origin="lower", aspect="auto")
    for j in range(len(y_values)):
        for i in range(len(x_values)):
            if not np.isnan(losses[j, i]):
                ax_right.text(
                    i,
                    j,
                    f"{losses[j, i]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if losses[j, i] > np.nanmedian(losses) else DARK,
                )
    ax_right.set_xticks(range(len(x_values)))
    ax_right.set_xticklabels([f"{x:g}" for x in x_values])
    ax_right.set_yticks(range(len(y_values)))
    ax_right.set_yticklabels([f"{y:g}" for y in y_values])
    ax_right.set_xlabel(x_field)
    ax_right.set_ylabel(y_field)
    ax_right.set_title("Final Loss Grid")
    fig.colorbar(image, ax=ax_right, fraction=0.05, pad=0.03).ax.tick_params(
        colors=GRAY
    )

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOT_DIR / f"{EXPERIMENT_KEY}_{pair_name}.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    set_style()
    runs_by_name = latest_runs_by_name(wandb.Api(timeout=60))
    for pair_name in PAIRS:
        print(f"{pair_name}:")
        losses = collect_grid(pair_name, runs_by_name)
        print_grid(pair_name, losses)
        if not np.all(np.isnan(losses)):
            print(f"  -> {plot_grid(pair_name, losses).resolve()}")


if __name__ == "__main__":
    main()
