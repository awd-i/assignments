import sys; sys.path.insert(0, 'report')
from plot_diag import *
P2_BEST = ['a4-rnn1-lr0.003', 'a4-rnn4-lr0.001', 'a4-rnn-lr0.002', 'a4-rnn-no-residual-lr0.001', 'a4-rnn-no-prenorm-lr0.002', 'a4-rnn-rope-lr0.002',
           'a4-rnn-final-norm-lr0.001', 'a4-rnn-postnorm-lr0.001', 'a4-rnn-gated-residual-lr0.001', 'a4-rnn-gated-output-lr0.001', 'a4-rnn-gated-norm-lr0.001']
depth_panels(P2_BEST, 'p2_depth_final', 'P2: depth profiles of the final checkpoints (best rate per variant)')
depth_panels(P2_BEST, 'p2_depth_init', 'P2: depth profiles at initialisation (seed 42)', stage='init')
stage_panels('a4-rnn-lr0.002', 'p2_rnn_stages', 'P2: modern RNN at init, step 100 and final')
temporal_panels(['a4-rnn1-lr0.003', 'a4-rnn4-lr0.001', 'a4-rnn-lr0.002'], 'p2_temporal_plain_vs_modern', 'P2: hidden state statistics over time, last recurrent layer (final checkpoints)', layer=-1)
temporal_panels(['a4-rnn4-lr0.001'], 'p2_temporal_rnn4_layers', 'P2: plain 4-layer RNN, every recurrent layer (final)')
temporal_panels(['a4-rnn-lr0.002'], 'p2_temporal_rnn_layers', 'P2: modern RNN, every recurrent layer (final)')
for stage in STAGES:
    temporal_panels(['a4-rnn1-lr0.003', 'a4-rnn4-lr0.001', 'a4-rnn-lr0.002'], f'p2_temporal_{stage}', f'P2: hidden state statistics over time at {stage} (last layer)', stage=stage, layer=-1)
temporal_panels(['a4-rnn-lr0.002', 'a4-rnn-no-residual-lr0.001', 'a4-rnn-no-prenorm-lr0.002', 'a4-rnn-rope-lr0.002', 'a4-rnn-postnorm-lr0.001', 'a4-rnn-gated-residual-lr0.001', 'a4-rnn-gated-output-lr0.001', 'a4-rnn-gated-norm-lr0.001'],
                'p2_temporal_ablations', 'P2: hidden state statistics over time, last layer, modern RNN ablations (final)', layer=-1)
context_plot(['a4-rnn1-lr0.003', 'a4-rnn4-lr0.001', 'a4-rnn-lr0.002', 'a4-rnn-gated-residual-lr0.001', 'a4-rnn-rope-lr0.002'], 'p2_context', 'P2: context use of RNNs')
for r in summary_rows(P2_BEST): print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
for run in P2_BEST:
    d = load(run)
    if d: print(run, 'temporal cos', [{k: round(v, 3) for k, v in tc.items()} for tc in d['temporal_cosine']])
