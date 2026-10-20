# Assignment 4: depth and time

Start with the repository [setup guide](../../README.md). Commands run from the
repository root and use your usual data, W&B settings, and Modal environment.

| Problem | Model definitions |
|---|---|
| 1: transformer depth | [p1_transformer_depth.py](p1_transformer_depth.py) |
| 2: RNN depth | [p2_rnn_depth.py](p2_rnn_depth.py) |
| 3: residual mixing | [p3_attention_residuals.py](p3_attention_residuals.py) |
| 4: recurrent updates | [p4_recurrent_updates.py](p4_recurrent_updates.py) |

Each architecture is a short class that overrides the relevant component or
update equation. Read its parent for the shared algorithm.

## Supplied training runs

The course reference sweep is **41 variants × 5 learning rates = 205 runs**.
It is being prepared in the public
[course W&B project](https://wandb.ai/hashimoto-group/public-alchemy/overview),
tagged `a4-reference` and `a4-release-v1`. The five rates are
`(.25, .5, 1, 2, 4) × center`; [sweeps.py](sweeps.py) specifies the centers.
Baseline aliases across problems reuse the same runs.

Once the release is published, you can use these runs for all supplied
architecture comparisons in Problems 1–4. You do not need to repeat those LR
sweeps. Analyze the results and checkpoints, measure diagnostics, and train
your own proposals for Problems 3 and 4. Add runs if your investigation needs
another setting or an LR extension.

Checkpoints will be in `cs312-shared-data`, volume `hard-dl-dclm-v1`, mounted
read-only at `/root/shared_data/a4_stability`. Its `manifest.json` will list
all outcomes, public run links, model definitions, and available checkpoints.
Finite completed runs have final weights and an `early/` checkpoint from
step 100. A divergent run has an outcome and W&B link, without final weights.
The manifest is published after the entire grid finishes and is checked.

Names are identical in W&B, on Modal, and locally, for example:
`a4-rnn-no-residual-lr0.001`.

## Measure an existing checkpoint

[measure_default.py](measure_default.py) is a complete example: load the
released default d8, take a held-out input, compute depth/attention gradients
and paired context-removal losses, and save the returned JSON locally.
After the checkpoint release:

```bash
uv run python -m experiments.a4_stability.measure_default
```

Edit its `RUN` constant to inspect another supplied model. For local copies,
call `measure(run, root, device)` directly. The reusable measurement functions
are `diagnostics.profile(model, tokens)` and `diagnostics.context_loss(...)`.
They use FP32 probes and retain true recurrent-state nodes before stacking.
GRU `z` writes new information; reset precedes the recurrent candidate map.

## Run a new experiment

Edit the single call at the bottom of a problem file, then run that module:

```python
from experiments.a4_stability.sweeps import launch_one, launch
launch_one('rnn', .001)                    # One run on Modal.
launch(('rnn',))                          # Its five-rate sweep on Modal.
launch(('rnn',), dry_run=True)            # Print names/configs only.
launch_one('rnn', .001, local=True)        # The same run on a local GPU.
```

The module entrypoints have editable Python calls, without argument parsers.
`sweeps.VARIANTS` maps each short name to its model class, problem, and LR
center. Students' new runs use their own configured W&B project. Staff pass
public-project settings to these same configs.

For a new architecture, subclass a component in the relevant problem file,
then add your model to `sweeps.VARIANTS`. For example, select a new `cell_type`
on a `ModernRNN` subclass. Carry only `h` across time for Problem 4.
Use a new variant/run name when changing the architecture, seed, or recipe.

The recipe is width 512, vocabulary 4096, context 1024, batch 64, 600,000
sequences, AdamW, cosine decay, 1% warmup, clipping at 1, and seed 42.
Weight decay is 0 for plain RNNs and .1 for modern models.
Training diagnostics are disabled by default. For your own runs, use
`training_config(Model, lr, problem=..., diagnostics=True)` to save initial,
early, and final profiles; automatic hooks require `wandb_online=True`.

The small [_runtime.py](_runtime.py) backend supplies cuDNN tanh recurrence,
compiled 16-token editable updates, and CUDA replay. States stay connected
across chunks. The problem files contain the algorithm equations.

Synthetic CPU checks require no dataset download or cloud connection:

```bash
uv run python -m unittest discover -s tests -p 'test_a4_*.py' -v
```
