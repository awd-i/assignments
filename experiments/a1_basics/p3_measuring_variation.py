"""Problem 3: measuring run-to-run variation of the d8 baseline.

Sources of randomness: model_seed (initialization), data_seed (data order/choice),
and hardware nondeterminism (deterministic=False, different GPUs). Existing runs
reused as extra samples: the default run (seeds 42/42), the 2(a) d8 baseline
(identical config, a free hardware-nondeterminism repeat), and the 1(c) model-seed
runs (model_seed 1, 2).

Waves (each launched as its own Modal app):
    uv run python -m experiments.a1_basics.p3_measuring_variation deterministic  # 2 H100 refs
    uv run python -m experiments.a1_basics.p3_measuring_variation h100           # everything else on H100
    uv run python -m experiments.a1_basics.p3_measuring_variation a100           # 2 cross-hardware runs
"""

import sys

from model_config import depth_model_config
from train import TrainConfig


EXPERIMENT_KEY = "p3-variation"
SEEDS = range(1, 6)


def config(group, **overrides):
    return TrainConfig(
        run_name_suffix=f"{EXPERIMENT_KEY}-{group}",
        wandb_tags=(EXPERIMENT_KEY, f"{EXPERIMENT_KEY}-{group}"),
        **overrides,
    )


def deterministic_runs():
    return [
        TrainConfig(
            deterministic=True,
            run_name_suffix=f"deterministic-reference-{i}",
            wandb_tags=(EXPERIMENT_KEY, f"{EXPERIMENT_KEY}-deterministic"),
        )
        for i in (1, 2)
    ]


def h100_runs():
    return [
        # (a) + (c): all three sources varied jointly.
        *[config("joint", model_seed=s, data_seed=s) for s in SEEDS],
        # (b): one source at a time; the others stay at seed 42.
        config("model-seed", model_seed=3),
        *[config("data-seed", data_seed=s) for s in (1, 2, 3)],
        *[config(f"hw-rep{i}") for i in (1, 2)],
        # (c): does variability change with hyperparameters or model size?
        *[config("joint-bs16", model_seed=s, data_seed=s, batch_size=16) for s in SEEDS],
        *[config("joint-lr009", model_seed=s, data_seed=s, learning_rate=0.009) for s in SEEDS],
        *[
            config("joint-d4", model_config=depth_model_config(4), model_seed=s, data_seed=s)
            for s in (1, 2, 3)
        ],
    ]


def c2_runs():
    """3(c) follow-up: settings expected to change run-to-run variability, joint seeds 1-3."""
    overrides = {"constant": {"lr_schedule": "constant"}, "lr003": {"learning_rate": 0.03}, "dropout": {"dropout": 0.2}}
    return [
        TrainConfig(
            model_seed=s,
            data_seed=s,
            run_name_suffix=f"{EXPERIMENT_KEY}-joint-{name}",
            wandb_tags=(EXPERIMENT_KEY, f"{EXPERIMENT_KEY}-joint-{name}", f"{EXPERIMENT_KEY}-c2"),
            **o,
        )
        for name, o in overrides.items()
        for s in (1, 2, 3)
    ]


def a100_runs():
    return [config("a100"), config("a100", deterministic=True)]


WAVES = {"deterministic": (deterministic_runs, "H100"), "h100": (h100_runs, "H100"), "a100": (a100_runs, "A100-80GB"),
         "c2": (c2_runs, "H100")}


def main() -> None:
    from modal_train import launch_training_jobs

    build, gpu = WAVES[sys.argv[1]]
    launch_training_jobs(build(), gpu=gpu, max_parallel_runs=2)


if __name__ == "__main__":
    main()
