"""Problem 2(a): scaling law reliability across d4-d9.

Stage 1 trains d4-d7. Pre-register d8/d9 predictions from those runs, then run
stage 2 (d8-d9):

    uv run python -m experiments.a1_basics.p2_scaling_law_reliability         # stage 1
    uv run python -m experiments.a1_basics.p2_scaling_law_reliability stage2  # stage 2
"""

import sys

from model_config import depth_model_config
from train import TrainConfig


EXPERIMENT_KEY = "scaling-v1"
STAGE_DEPTHS = {
    "stage1": range(4, 8),
    "stage2": range(8, 10),
}
VARIANTS = [
    # (learning_rate, lr_schedule, dropout)
    (0.003, "linear", 0.0),
    (0.003, "constant", 0.0),
    (0.003, "linear", 0.2),
    (0.03, "linear", 0.0),
]
MAX_PARALLEL_RUNS = 2


def build_runs(depths=range(4, 10)):
    runs = []
    seen_configs = set()
    # Largest models first, so they start in fresh containers.
    for depth in sorted(depths, reverse=True):
        for learning_rate, lr_schedule, dropout in VARIANTS:
            config_key = (depth, learning_rate, lr_schedule, dropout)
            if config_key in seen_configs:
                continue
            seen_configs.add(config_key)
            runs.append(
                TrainConfig(
                    model_config=depth_model_config(depth),
                    learning_rate=learning_rate,
                    lr_schedule=lr_schedule,
                    dropout=dropout,
                    run_name_suffix=EXPERIMENT_KEY,
                    wandb_tags=(EXPERIMENT_KEY,),
                )
            )
    return runs


def main() -> None:
    from modal_train import launch_training_jobs

    stage = sys.argv[1] if len(sys.argv) > 1 else "stage1"
    launch_training_jobs(
        build_runs(STAGE_DEPTHS[stage]), max_parallel_runs=MAX_PARALLEL_RUNS
    )


if __name__ == "__main__":
    main()
