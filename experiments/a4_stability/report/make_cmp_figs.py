import sys; sys.path.insert(0, 'report')
from plot_diag import *
position_panels(['a4-d8-lr0.003', 'a4-rnn-lr0.002'], 'cmp_positions', 'd8 transformer vs modern RNN: activation RMS and similarity across positions (final checkpoints)')
position_panels(['a4-d8-no-residual-lr0.00075', 'a4-rnn-no-residual-lr0.001'], 'cmp_positions_nores', 'No-residual transformer vs no-residual RNN: positions (final checkpoints)')
for run in ['a4-d8-lr0.003', 'a4-rnn-lr0.002', 'a4-d8-gated-residual-lr0.003', 'a4-rnn-gated-residual-lr0.001', 'a4-rnn1-lr0.003', 'a4-rnn4-lr0.001']:
    d = load(run, 'position')
    print(run, 'lag1', [round(x['cosine_by_lag']['1'], 2) for x in d['depth']])
    print(run, 'lag64', [round(x['cosine_by_lag']['64'], 2) for x in d['depth']])
    print(run, 'to first (256)', [round(x['cosine_to_first']['256'], 2) for x in d['depth']])
    r = np.array(d['depth'][-1]['rms_by_position']); print(run, 'final stream rms at t=0,1,8,64,512,1022', [round(r[i], 2) for i in (0, 1, 8, 64, 512, 1022)])
    r = np.array(d['depth'][0]['rms_by_position']); print(run, 'embedding rms at t=0,1,8', [round(r[i], 3) for i in (0, 1, 8)])
