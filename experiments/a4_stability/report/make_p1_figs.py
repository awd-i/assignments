import sys; sys.path.insert(0, 'report')
from plot_diag import *
P1_BEST = ['a4-d8-lr0.003', 'a4-d8-no-residual-lr0.00075', 'a4-d8-no-prenorm-lr0.001', 'a4-d8-no-qknorm-lr0.003', 'a4-d8-no-rope-lr0.003',
           'a4-d8-no-final-norm-lr0.003', 'a4-d8-postnorm-lr0.001', 'a4-d8-gated-residual-lr0.003', 'a4-d8-gated-attention-lr0.006', 'a4-d8-gated-norm-lr0.00225']
depth_panels(P1_BEST, 'p1_depth_final', 'P1: depth profiles of the final checkpoints (best rate per variant)')
depth_panels(P1_BEST, 'p1_depth_init', 'P1: depth profiles at initialisation (seed 42)', stage='init')
stage_panels('a4-d8-lr0.003', 'p1_d8_stages', 'P1: baseline d8 at init, step 100 and final')
stage_panels('a4-d8-no-residual-lr0.00075', 'p1_nores_stages', 'P1: no-residual d8 at init, step 100 and final')
attention_panels(P1_BEST, 'p1_attention', 'P1: attention statistics averaged over layers and heads (final checkpoints)')
for run, tag in [('a4-d8-lr0.003', ''), ('a4-d8-no-rope-lr0.003', '_norope'), ('a4-d8-gated-attention-lr0.006', '_gatedattn')]:
    if load(run): attention_by_layer(run, f'p1_attention_layers{tag}', f'P1: {run[3:]} attention per layer (final)')
if load('a4-d8-lr0.003', 'init'): attention_by_layer('a4-d8-lr0.003', 'p1_attention_layers_init', 'P1: baseline d8 attention per layer (init)', stage='init')
context_plot(['a4-d8-lr0.003', 'a4-d8-no-rope-lr0.003', 'a4-d8-gated-residual-lr0.003', 'a4-d8-gated-attention-lr0.006'], 'p1_context', 'P1: context use of transformers')
for r in summary_rows(P1_BEST): print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
