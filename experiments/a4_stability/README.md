# Assignment 4: depth and time

Start with the repository [setup guide](../../README.md). Run commands from the
repository root. These files use the existing trainer, data, W&B configuration,
and Modal launcher.

| Problem | Read and edit |
|---|---|
| 1: transformer depth | [p1_transformer_depth.py](p1_transformer_depth.py): baseline and one-component subclasses. |
| 2: RNN depth | [p2_rnn_depth.py](p2_rnn_depth.py): plain 1/4-layer RNNs, modern RNN, and depth ablations. |
| 3: residual mixing | [p3_attention_residuals.py](p3_attention_residuals.py): fixed EMAs, uniform mixing, full and blocked attention residuals for both models. |
| 4: recurrent updates | [p4_recurrent_updates.py](p4_recurrent_updates.py): GRU equations and component ablations. |

Each architecture is a class. An ablation overrides a component, a formula, or
a class attribute selecting that component. The `MODELS` dictionary at the
bottom of each problem file supplies command-line names.

Problems 1 and 2 will have reference W&B runs and Modal checkpoints linked with
the handout. Use the references for the assigned comparisons; train additional
models when needed for your investigations. There is no automatic sweep here.

## Run one experiment

Inspect a config without launching anything:

```bash
uv run python -m experiments.a4_stability.p1_transformer_depth --model no_residual --lr .001 --dry-run
```

Remove `--dry-run` to launch that one model on Modal. Add `--local` to train
on your local GPU. `--help` lists the choices in each problem. The problem files
use the same interface; for example:

```bash
uv run python -m experiments.a4_stability.p2_rnn_depth --model modern --lr .001
uv run python -m experiments.a4_stability.p3_attention_residuals --model transformer_block2 --lr .001
uv run python -m experiments.a4_stability.p4_recurrent_updates --model gru --lr .003
```

The recipe is width 512, vocabulary 4096, context 1024, batch 64, 600,000
sequences, AdamW, cosine decay, 1% warmup, and clipping at 1. Plain RNNs use
weight decay 0; modern models use .1. Tune the learning rate as the handout
asks. Baseline aliases across problems share a run name so completed baselines
can be reused.

For other settings, call `common.training_config(ModelClass, lr, problem=...)`
and override the usual `TrainConfig` fields. New experiments should use a new
class name so they do not resume a different algorithm's checkpoint.

## Measure propagation

The default logger uses the same held-out input at initialization, the first
evaluation, and the end of training. It saves `initial.json`, `early.json`, and
`final.json` in `<run directory>/diagnostics/` and logs depth scalars to W&B.
Each snapshot records its optimizer step. The JSON files include both
full-sequence and final-token objectives, depth
RMS/gradients/cosines, raw branch/input ratios, true recurrent-state gradients,
state magnitudes, attention versus causal-uniform controls, and depth routing.
See [diagnostics.py](diagnostics.py) for the measurement conventions.

Automatic snapshots require `wandb_online=True`, like other custom loggers.
Use `WANDB_MODE=offline` for offline logging, or call `profile` directly.

You can also inspect a checkpoint directly:

```python
from model_io import load_model
from experiments.a4_stability.diagnostics import profile, context_loss

model = load_model(checkpoint_path, device="cuda", precision="fp32")
stats = profile(model, tokens)  # tokens: a held-out [batch, sequence] tensor
full = context_loss(model, tokens, keep=tokens.shape[1] - 1)
short = context_loss(model, tokens, keep=64, preserve_first=True)
delta = short - full  # paired losses; also try preserve_first=False
```

Training uses mixed precision with FP32 weights (`precision="mp"`); the fixed
probes use FP32. Depth is counted in recurrent/attention and MLP sublayers.
The RNN output gate changes only the depth stream; its ungated state is carried
across time. The GRU's `z` is the fraction of **new** information, and reset
is applied **before** the recurrent candidate projection, as in the handout.

To explore, subclass a depth scheme or a cell and add your model to that
problem's `MODELS`. Problem 4 carries only `h`: no second state or history cache.

The small [_runtime.py](_runtime.py) backend keeps cuDNN tanh RNNs, compiled
16-token editable updates, and CUDA replay for the GRU stack. Chunk boundaries
do not detach states. Probes use the same cell equations with explicit states.

## Checks

Synthetic CPU checks need no data download, Modal jobs, or W&B connection:

```bash
uv run python -m unittest discover -s tests -p 'test_a4_*.py' -v
```
