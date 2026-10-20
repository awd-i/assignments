"""Use the existing trainer and Modal launcher; choose one model and one LR."""
import argparse
from dataclasses import asdict, replace
import json
import sys

from model_config import MODEL_CONFIGS
from metric_logging import AFTER_EVAL, MetricLogger
from train import TrainConfig
from .diagnostics import StabilityLogger


def training_config(model, lr=.003, *, problem, **overrides):
    module = model.__module__
    if module == '__main__':
        module = sys.modules[module].__spec__.name
    architecture = replace(MODEL_CONFIGS['d8'], name=model.model_name,
                           num_hidden_layers=model.num_layers)
    config = TrainConfig(
        model_config=architecture, model_builder=f'{module}:{model.__qualname__}',
        learning_rate=lr, num_train_sequences=600_000, batch_size=64,
        model_seed=42, data_seed=42, precision='mp', torch_compile_mode='default',
        qk_norm=getattr(model, 'use_qk_norm', True),
        lr_schedule='cos', warmup_percent=.01, grad_norm=1., num_evals=20,
        weight_decay=model.weight_decay, run_name_suffix=f'a4-{model.__name__}',
        wandb_tags=('a4', f'p{problem}', model.__name__),
        metric_loggers=(MetricLogger(AFTER_EVAL, StabilityLogger()),))
    return replace(config, **overrides)


def run(problem, models):
    parser = argparse.ArgumentParser(description=f'Assignment 4, Problem {problem}: one training run.')
    parser.add_argument('--model', choices=models, default=next(iter(models)))
    parser.add_argument('--lr', type=float, default=.003)
    parser.add_argument('--local', action='store_true', help='Train locally instead of on Modal.')
    parser.add_argument('--dry-run', action='store_true', help='Print the config without launching.')
    args = parser.parse_args()
    config = training_config(models[args.model], args.lr, problem=problem)
    if args.dry_run:
        print(json.dumps(asdict(config), default=str, indent=2))
    elif args.local:
        from train import train
        train(config)
    else:
        from modal_train import launch_training_jobs
        launch_training_jobs([config])
