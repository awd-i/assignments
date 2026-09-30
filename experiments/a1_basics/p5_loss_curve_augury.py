"""Problem 5 (and example-question checks). Loss-curve analysis uses existing runs;
the only new optimizer knob is Adam beta1 (momentum). Also verifies the Problem 1
example questions (easy: LR x batch size ranking; medium: LR 0.009 + wd 1.0 with
different schedules/warmups).
"""

from train import TrainConfig


def config(key, **overrides):
    return TrainConfig(run_name_suffix=key, wandb_tags=(key,), **overrides)


RUNS = [
    *[config("p5-beta1", beta1=beta1) for beta1 in (0.5, 0.98)],
    # Problem 1 easy example: A = default (existing run), B and C are new.
    config("p1-example-easy", learning_rate=0.001, batch_size=128),
    config("p1-example-easy", learning_rate=0.009, batch_size=128),
    # Problem 1 medium example.
    *[
        config("p1-example-medium", learning_rate=0.009, weight_decay=1.0, lr_schedule=s, warmup_percent=w)
        for s, w in (("wsd0.2", 0.0), ("wsd0.2", 0.2), ("linear", 0.1))
    ],
]


def main() -> None:
    from modal_train import launch_training_jobs

    launch_training_jobs(RUNS, max_parallel_runs=2)


if __name__ == "__main__":
    main()
