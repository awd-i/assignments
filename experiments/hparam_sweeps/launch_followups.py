"""Follow-ups that strengthen the write-up:

- warmup: bracket the warmup optimum from 1(a), which kept improving up to 4%.
- width-lr: a muP-style LR that shrinks with width (LR = 3e-3 * 8 / depth), to test
  whether size-aware tuning changes the model-size scaling slope in 2(b). d8 is the
  baseline itself.
- seeds: replicate the small 1(a) wins (bs 32, wd 0.2) at two more model seeds.
"""

EXPERIMENT_KEY = "followups-v1"
MAX_PARALLEL_RUNS = 2


def width_scaled_lr(depth: int) -> float:
    return 3e-3 * 8 / depth


def build_runs():
    from model_config import depth_model_config
    from train import TrainConfig

    def config(group, **overrides):
        return TrainConfig(
            run_name_suffix=f"{EXPERIMENT_KEY}-{group}",
            wandb_tags=(EXPERIMENT_KEY, f"{EXPERIMENT_KEY}-{group}"),
            **overrides,
        )

    return [
        # Largest model first so it starts in a fresh container.
        config("width-lr", model_config=depth_model_config(9), learning_rate=width_scaled_lr(9)),
        *[config("warmup", warmup_percent=warmup) for warmup in (0.08, 0.16)],
        *[
            config("width-lr", model_config=depth_model_config(depth), learning_rate=width_scaled_lr(depth))
            for depth in (6, 4)
        ],
        *[
            config("seeds", model_seed=seed, **overrides)
            for overrides in ({"batch_size": 32}, {"weight_decay": 0.2})
            for seed in (1, 2)
        ],
    ]


def main() -> None:
    from modal_train import launch_training_jobs

    launch_training_jobs(build_runs(), max_parallel_runs=MAX_PARALLEL_RUNS)


if __name__ == "__main__":
    main()
