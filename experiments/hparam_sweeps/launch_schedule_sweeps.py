EXPERIMENT_KEY = "sched-sweeps-v1"

# Part (c): how the LR schedule and other optimizer hyperparameters interact
# with the part (a)/(b) hyperparameters. Every run is the default d8 recipe
# plus the listed overrides. Linear-schedule references come from (a)/(b).
# Note: the "constant" schedule ignores warmup (no warmup, no decay).
BLOCKS = {
    # bs256 goes first so it lands in a fresh container: after a torch.compile
    # job, a reused container can hold ~79 GB of GPU memory and bs256 OOMs.
    "bs256-warmup": [
        {"batch_size": 256, "learning_rate": 6e-3, "warmup_percent": 0.04},
    ],
    "schedule-lr": [
        {"lr_schedule": schedule, "learning_rate": learning_rate}
        for schedule in ("cos", "wsd0.2", "constant")
        for learning_rate in (1.5e-3, 3e-3, 6e-3)
    ],
    "wsd-decay": [
        {"lr_schedule": schedule, "learning_rate": 3e-3}
        for schedule in ("wsd0.1", "wsd0.4")
    ],
    "schedule-warmup": [
        {"lr_schedule": schedule, "learning_rate": 6e-3, "warmup_percent": warmup}
        for schedule in ("cos", "wsd0.2")
        for warmup in (0.0025, 0.04)
    ],
    "schedule-batch-size": [
        {"lr_schedule": schedule, "batch_size": batch_size}
        for schedule in ("cos", "wsd0.2")
        for batch_size in (32, 128)
    ],
    "optimizer-lr": [
        {**override, "learning_rate": learning_rate}
        for override in ({"beta2": 0.9}, {"beta2": 0.99}, {"grad_norm": None})
        for learning_rate in (3e-3, 6e-3)
    ],
    "seeds": [{"model_seed": seed} for seed in (1, 2)],
}
MAX_PARALLEL_RUNS = 2


def block_key(block_name: str) -> str:
    return f"{EXPERIMENT_KEY}-{block_name}"


def block_config(block_name: str, overrides: dict):
    from train import TrainConfig

    overrides = dict(overrides)
    batch_size = overrides.get("batch_size", TrainConfig.batch_size)
    if batch_size > 128:
        overrides["num_micro_batches"] = batch_size // 128
    return TrainConfig(
        run_name_suffix=block_key(block_name),
        wandb_tags=(EXPERIMENT_KEY, block_key(block_name)),
        **overrides,
    )


def build_runs():
    return [
        block_config(block_name, overrides)
        for block_name, block in BLOCKS.items()
        for overrides in block
    ]


def main() -> None:
    from modal_train import launch_training_jobs

    launch_training_jobs(build_runs(), max_parallel_runs=MAX_PARALLEL_RUNS)


if __name__ == "__main__":
    main()
