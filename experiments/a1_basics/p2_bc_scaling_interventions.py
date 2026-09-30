"""Problem 2(b)/(c): interventions that change or break scaling laws.

Model-size axis: depth (width = 64 x depth) at the default 600k sequences.
Data axis: d6 with num_train_sequences varied. Baselines come from the 2(a)
runs (d4-d9 baseline and d6 at 600k).

(b) slope changes that stay in the linear (power-law) regime, 10 runs.
(c) interventions that break the power-law form, 20 runs.

Micro-batched runs (batch size > 128) skip torch.compile. Launch them in their
own app so they never reuse a container left holding CUDA-graph memory:

    uv run python -m experiments.a1_basics.p2_bc_scaling_interventions microbatch
    uv run python -m experiments.a1_basics.p2_bc_scaling_interventions compiled
"""

import sys

from model_config import depth_model_config
from train import TrainConfig


EXPERIMENT_KEY = "scaling-bc-v1"
TUNED = {"beta2": 0.99, "warmup_percent": 0.04, "weight_decay": 0.2}

# (group, depth, overrides)
PART_B = [
    *[("b-tuned", depth, TUNED) for depth in (4, 6, 8)],
    *[
        ("b-bs256", depth, {"batch_size": 256, "warmup_percent": 0.04, "learning_rate": 6e-3})
        for depth in (4, 6, 8)
    ],
    *[("b-data", 6, {"num_train_sequences": n}) for n in (150_000, 300_000)],
    *[
        ("b-data-dropout", 6, {"num_train_sequences": n, "dropout": 0.2})
        for n in (150_000, 300_000)
    ],
]
PART_C = [
    *[("c-lr0.1", depth, {"learning_rate": 0.1}) for depth in (4, 6, 8, 9)],
    *[
        ("c-unstable", depth, {"learning_rate": 0.03, "warmup_percent": 0.0, "grad_norm": None})
        for depth in (4, 6, 8)
    ],
    *[
        ("c-repeat16", depth, {"num_train_sequences": 37_500, "num_epochs": 16.0})
        for depth in (4, 6, 8, 9)
    ],
    *[("c-tiny-data", 6, {"num_train_sequences": n}) for n in (3_000, 10_000, 30_000)],
    *[("c-bs1024", depth, {"batch_size": 1024}) for depth in (4, 6, 8)],
    *[("c-wd2", depth, {"weight_decay": 2.0}) for depth in (4, 6, 8)],
]
MAX_PARALLEL_RUNS = 2


def run_config(group: str, depth: int, overrides: dict):
    overrides = dict(overrides)
    batch_size = overrides.get("batch_size", TrainConfig.batch_size)
    if batch_size > 128:
        overrides["num_micro_batches"] = batch_size // 128
    return TrainConfig(
        model_config=depth_model_config(depth),
        run_name_suffix=f"{EXPERIMENT_KEY}-{group}",
        wandb_tags=(EXPERIMENT_KEY, f"{EXPERIMENT_KEY}-{group}"),
        **overrides,
    )


def build_runs(specs=(*PART_B, *PART_C)):
    return [run_config(group, depth, overrides) for group, depth, overrides in specs]


def main() -> None:
    from modal_train import launch_training_jobs

    runs = build_runs()
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "microbatch":
        runs = [config for config in runs if config.num_micro_batches > 1]
    elif which == "compiled":
        runs = [config for config in runs if config.num_micro_batches == 1]
    # Largest models first so they start in fresh containers.
    runs.sort(key=lambda config: -config.model_config.num_hidden_layers)
    launch_training_jobs(runs, max_parallel_runs=MAX_PARALLEL_RUNS)


if __name__ == "__main__":
    main()
