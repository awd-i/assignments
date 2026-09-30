EXPERIMENT_KEY = "hparam-sweeps-v2"

# One-at-a-time sweeps around the default d8 recipe. Each variable steps along a
# log axis by 2x (default /4, /2, x1, x2, x4), with the default at the center.
# The center point is the existing default run, so it is not relaunched here.
SWEEPS = {
    "learning_rate": (7.5e-4, 1.5e-3, 3e-3, 6e-3, 1.2e-2),
    "warmup_percent": (0.0025, 0.005, 0.01, 0.02, 0.04),
    "weight_decay": (0.025, 0.05, 0.1, 0.2, 0.4),
    "batch_size": (16, 32, 64, 128, 256),
}

# The default run launched by experiments/smoke/modal_smoke_train.py.
BASELINE_RUN_NAME_SUFFIX = "modal"
MAX_PARALLEL_RUNS = 2


def sweep_key(field_name: str) -> str:
    return f"{EXPERIMENT_KEY}-{field_name.replace('_', '-')}"


def sweep_config(field_name: str, value):
    from train import TrainConfig

    overrides = {field_name: value}
    if field_name == "batch_size" and value > 128:
        overrides["num_micro_batches"] = value // 128
    return TrainConfig(
        run_name_suffix=sweep_key(field_name),
        wandb_tags=(EXPERIMENT_KEY, sweep_key(field_name)),
        **overrides,
    )


def build_runs():
    from train import TrainConfig

    return [
        sweep_config(field_name, value)
        for field_name, values in SWEEPS.items()
        for value in values
        if value != getattr(TrainConfig, field_name)
    ]


def main() -> None:
    from modal_train import launch_training_jobs

    launch_training_jobs(build_runs(), max_parallel_runs=MAX_PARALLEL_RUNS)


if __name__ == "__main__":
    main()
