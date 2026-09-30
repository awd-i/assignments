"""Problem 7: own prediction question - which depth wins under 4x data repetition?

Every run sees the same 614M training tokens, but only 150k unique sequences
(154M unique tokens) repeated for 4 epochs. Known reference points (from 2(a)/2(c)):
  unique data (1 epoch):  d4 3.290, d6 3.057, d9 2.885   (bigger is better)
  16x repetition:         d4 3.393, d6 3.367, d9 3.640   (U-shape, d6 best)
Question: at 4x repetition, rank d4, d6, d9 by final validation loss.

    uv run python -m experiments.a1_basics.p7_own_question          # train the 3 runs
    uv run python -m experiments.a1_basics.p7_own_question resolve  # print the answer
"""

import sys

from model_config import depth_model_config
from train import TrainConfig, training_run_name


EXPERIMENT_KEY = "p7-repeat4"
DEPTHS = (9, 6, 4)
UNIQUE_DATA_LOSS = {4: 3.290, 6: 3.057, 9: 2.885}  # 2(a) baseline, 1 epoch of unique data
RUNS = [
    TrainConfig(
        model_config=depth_model_config(depth),
        num_train_sequences=150_000,
        num_epochs=4.0,
        run_name_suffix=EXPERIMENT_KEY,
        wandb_tags=(EXPERIMENT_KEY,),
    )
    for depth in DEPTHS
]


def resolve() -> None:
    import wandb

    from utils import WANDB_ENTITY, WANDB_PROJECT

    api = wandb.Api(timeout=60)
    losses = {}
    for depth, config in zip(DEPTHS, RUNS, strict=True):
        name = training_run_name(config)
        runs = [r for r in api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters={"display_name": {"$regex": f"^{name}"}})
                if r.state == "finished" and "contaminated" not in (r.tags or [])]
        # The regex also matches a "-rerun" copy; use the most recent clean run.
        losses[depth] = max(runs, key=lambda r: r.created_at).summary["val_loss"]
    for depth in sorted(losses):
        print(f"d{depth}: final val loss {losses[depth]:.4f} (unique data {UNIQUE_DATA_LOSS[depth]:.3f}, "
              f"repetition cost {losses[depth] - UNIQUE_DATA_LOSS[depth]:+.4f})")
    print("Ranking (best to worst):", " < ".join(f"d{d}" for d in sorted(losses, key=losses.get)))


def main() -> None:
    if sys.argv[1:] == ["resolve"]:
        resolve()
        return
    from modal_train import launch_training_jobs

    launch_training_jobs(RUNS, max_parallel_runs=2)


if __name__ == "__main__":
    main()
