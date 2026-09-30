from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import wandb

from experiments.hparam_sweeps.launch_hparam_sweeps import (
    BASELINE_RUN_NAME_SUFFIX,
    EXPERIMENT_KEY,
    SWEEPS,
    sweep_config,
)
from experiments.lr_tuning.plot_lr_tuning import (
    DARK,
    GRAY,
    LEFT_Y_TICK_LABELS,
    LEFT_Y_TICKS,
    LIGHT_BLUE,
    PURPLE,
    RED,
    metric_trajectory,
    set_style,
)
from train import TrainConfig, training_run_name
from utils import WANDB_ENTITY, WANDB_PROJECT


PLOT_DIR = Path(__file__).resolve().parent / "plots"
PROJECT_PATH = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
CMAP = mcolors.LinearSegmentedColormap.from_list("sweep", [PURPLE, LIGHT_BLUE])


def expected_run_names(field_name: str) -> dict:
    default = getattr(TrainConfig, field_name)
    names = {}
    for value in SWEEPS[field_name]:
        if value == default:
            config = TrainConfig(run_name_suffix=BASELINE_RUN_NAME_SUFFIX)
        else:
            config = sweep_config(field_name, value)
        names[training_run_name(config)] = value
    return names


def latest_runs_by_name(api) -> dict:
    runs = {}
    for run in api.runs(PROJECT_PATH, order="+created_at"):
        runs[run.name] = run
    return runs


def collect(field_name: str, runs_by_name: dict) -> list[dict]:
    collected = []
    for run_name, value in expected_run_names(field_name).items():
        run = runs_by_name.get(run_name)
        if run is None:
            print(f"  missing: {field_name}={value} ({run_name})")
            continue
        steps, losses = metric_trajectory(run, "val_loss")
        if not losses.size:
            print(f"  no val_loss yet: {field_name}={value} ({run.state})")
            continue
        batch_size = run.config.get("batch_size", TrainConfig.batch_size)
        context_length = TrainConfig().train_dataset.context_length
        collected.append(
            {
                "value": value,
                "tokens": (steps + 1) * batch_size * context_length,
                "losses": losses,
                "final": float(losses[-1]),
                "state": run.state,
                "name": run.name,
            }
        )
        print(
            f"  {field_name}={value:g} val_loss={losses[-1]:.4f} "
            f"state={run.state} name={run.name}"
        )
    return sorted(collected, key=lambda item: item["value"])


def plot_sweep(field_name: str, runs: list[dict]) -> Path:
    values = np.array([run["value"] for run in runs], dtype=float)
    finals = np.array([run["final"] for run in runs])
    best = int(np.argmin(finals))
    norm = mcolors.LogNorm(vmin=min(SWEEPS[field_name]), vmax=max(SWEEPS[field_name]))

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )
    for run in runs:
        ax_left.plot(
            run["tokens"],
            run["losses"],
            color=CMAP(norm(run["value"])),
            linewidth=1.8,
            alpha=0.9,
            label=f"{run['value']:g}",
        )
    ax_left.set_xscale("log")
    ax_left.set_yscale("log")
    ax_left.set_yticks(LEFT_Y_TICKS)
    ax_left.set_yticklabels(LEFT_Y_TICK_LABELS)
    ax_left.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax_left.set_xlabel("Training tokens")
    ax_left.set_ylabel("Validation loss")
    ax_left.set_title("Loss Trajectories")
    ax_left.grid(True, which="major", linestyle=":", alpha=0.3)
    ax_left.legend(title=field_name, frameon=False, fontsize=8.5, title_fontsize=9)

    ax_right.plot(values, finals, color=GRAY, linewidth=1.0, alpha=0.45)
    ax_right.scatter(
        values, finals, s=80, c=values, cmap=CMAP, norm=norm, edgecolor=DARK, zorder=5
    )
    ax_right.scatter(
        [values[best]],
        [finals[best]],
        s=150,
        facecolors="none",
        edgecolor=RED,
        linewidth=2.0,
        zorder=6,
        label=f"best: {values[best]:g}",
    )
    for value, loss in zip(values, finals, strict=True):
        ax_right.annotate(
            f"{loss:.3f}",
            xy=(value, loss),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color=GRAY,
        )
    ax_right.set_xscale("log")
    ax_right.set_xticks(list(SWEEPS[field_name]))
    ax_right.set_xticklabels([f"{v:g}" for v in SWEEPS[field_name]])
    ax_right.minorticks_off()
    ax_right.set_xlabel(field_name)
    ax_right.set_ylabel("Final validation loss")
    ax_right.set_title("Final Loss")
    ax_right.margins(y=0.12)
    ax_right.grid(True, which="major", linestyle=":", alpha=0.3)
    ax_right.legend(frameon=False, loc="best")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOT_DIR / f"{EXPERIMENT_KEY}_{field_name}.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    set_style()
    runs_by_name = latest_runs_by_name(wandb.Api(timeout=60))
    for field_name in SWEEPS:
        print(f"{field_name}:")
        runs = collect(field_name, runs_by_name)
        if runs:
            print(f"  -> {plot_sweep(field_name, runs).resolve()}")


if __name__ == "__main__":
    main()
