"""A2 Problem 1(c)/(d) target runs (configs only; launched by a driver that keeps at most 2 GPUs busy)."""

from experiments.a2.modal_launcher import config
from experiments.a2.p1_learning_rate import EXPERIMENT_KEY as P1C_KEY, runs as p1c_runs

HYPERBALL = dict(optimizer_builder="experiments.a2.awd.hyperball:build", weight_decay=0.0)
P1C_PREDICTED = (3.73e-3, 2.23e-3)  # all-six fit, larger-three fit (p1cd_preregistered_predictions.md)
P1D_LRS = (5.7e-3, 8.0e-3, 1.14e-2, 1.6e-2)  # 1.14e-2 = pre-registered prediction


def validation():
    """Reproduce one supplied Hyperball run (153.6M tokens, LR 0.01: provided loss 3.2113)."""
    key = "a2-p1d-hyperball-validate"
    return [config(tokens=153_600_000, learning_rate=0.01, run_name_suffix=key, wandb_tags=(key,), **HYPERBALL)]


def p1d():
    key = "a2-p1d-hyperball-D1229m"
    return [config(tokens=1_228_800_000, learning_rate=lr, run_name_suffix=key, wandb_tags=(key,), **HYPERBALL)
            for lr in P1D_LRS]


def p1c():
    return p1c_runs(P1C_PREDICTED)  # target grid {.0015, .003, .006} + both predictions, tag a2-p1-D4915m
