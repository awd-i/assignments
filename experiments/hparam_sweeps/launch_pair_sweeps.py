EXPERIMENT_KEY = "pair-sweeps-v1"

# Two-hyperparameter grids around the default d8 recipe. Points where either
# hyperparameter is at its default are already covered by the one-at-a-time
# sweeps in launch_hparam_sweeps.py (or the default run), so only off-axis
# combinations are launched here.
PAIRS = {
    "batch_size-learning_rate": (
        ("batch_size", (16, 32, 64, 128, 256)),
        ("learning_rate", (1.5e-3, 3e-3, 6e-3, 1.2e-2)),
    ),
    "learning_rate-weight_decay": (
        ("learning_rate", (1.5e-3, 3e-3, 6e-3)),
        ("weight_decay", (0.05, 0.1, 0.2)),
    ),
    "learning_rate-warmup_percent": (
        ("learning_rate", (1.5e-3, 3e-3, 6e-3)),
        ("warmup_percent", (0.0025, 0.01, 0.04)),
    ),
}
# The highest LR is only tried at large batch sizes, where (a) suggests the
# default LR is too low.
EXCLUDED_POINTS = {
    ("batch_size-learning_rate", 16, 1.2e-2),
    ("batch_size-learning_rate", 32, 1.2e-2),
}
MAX_PARALLEL_RUNS = 2


def pair_key(pair_name: str) -> str:
    return f"{EXPERIMENT_KEY}-{pair_name.replace('_', '-')}"


def pair_config(pair_name: str, x_value, y_value):
    from train import TrainConfig

    (x_field, _), (y_field, _) = PAIRS[pair_name]
    overrides = {x_field: x_value, y_field: y_value}
    batch_size = overrides.get("batch_size", TrainConfig.batch_size)
    if batch_size > 128:
        overrides["num_micro_batches"] = batch_size // 128
    return TrainConfig(
        run_name_suffix=pair_key(pair_name),
        wandb_tags=(EXPERIMENT_KEY, pair_key(pair_name)),
        **overrides,
    )


def is_off_axis(pair_name: str, x_value, y_value) -> bool:
    from train import TrainConfig

    (x_field, _), (y_field, _) = PAIRS[pair_name]
    return x_value != getattr(TrainConfig, x_field) and y_value != getattr(
        TrainConfig, y_field
    )


def build_runs():
    return [
        pair_config(pair_name, x_value, y_value)
        for pair_name, ((_, x_values), (_, y_values)) in PAIRS.items()
        for x_value in x_values
        for y_value in y_values
        if is_off_axis(pair_name, x_value, y_value)
        and (pair_name, x_value, y_value) not in EXCLUDED_POINTS
    ]


def main() -> None:
    from modal_train import launch_training_jobs

    launch_training_jobs(build_runs(), max_parallel_runs=MAX_PARALLEL_RUNS)


if __name__ == "__main__":
    main()
