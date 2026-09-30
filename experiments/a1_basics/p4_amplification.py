"""Problem 4: amplification of a tiny data perturbation under full determinism.

All runs use deterministic=True with the default seeds, so the only difference from
deterministic-reference-1 (Problem 3) is the perturbation. The generalized
perturbation (train.py: perturb_step, perturb_num_tokens) shifts perturb_num_tokens
tokens by +1 starting at position 100 of the first sequence in the batch used at
optimizer step perturb_step.

(a) the handout's single-token edit, and replacing one whole sequence at step 0.
(b) when (1 token at 0/25/50/75/95% of training) and how much (1 token, 64 tokens,
    one sequence, one batch at step 0; one batch at 50%).
"""

from train import TrainConfig


EXPERIMENT_KEY = "p4-amplification"
TOTAL_STEPS = 9375
BATCH_TOKENS = 64 * 1024


def config(group, **overrides):
    return TrainConfig(
        deterministic=True,
        run_name_suffix=f"{EXPERIMENT_KEY}-{group}",
        wandb_tags=(EXPERIMENT_KEY, f"{EXPERIMENT_KEY}-{group}"),
        **overrides,
    )


RUNS = [
    # (a)
    config("handout", perturb_one_token=True),
    config("size", perturb_step=0, perturb_num_tokens=1024),
    # (b) timing: one token at 25/50/75/95% (1 token at step 0 is the handout run)
    *[
        config("time", perturb_step=int(TOTAL_STEPS * fraction), perturb_num_tokens=1)
        for fraction in (0.25, 0.5, 0.75, 0.95)
    ],
    # (b) magnitude
    config("size", perturb_step=0, perturb_num_tokens=64),
    config("size", perturb_step=0, perturb_num_tokens=BATCH_TOKENS),
    config("size", perturb_step=TOTAL_STEPS // 2, perturb_num_tokens=BATCH_TOKENS),
]


def main() -> None:
    from modal_train import launch_training_jobs

    launch_training_jobs(RUNS, max_parallel_runs=2)


if __name__ == "__main__":
    main()
