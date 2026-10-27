"""Student launches for P3/P4. Edit WAVES, then run a wave by name:

    uv run python -m experiments.a4_stability.launch_student p3_wave1
"""
import sys
from .sweeps import config

TRANSFORMER_P3 = ('d8-half', 'd8-ema01', 'd8-ema09', 'd8-uniform', 'd8-attention', 'd8-block2', 'd8-block4')
RNN_P3 = ('rnn-half', 'rnn-ema01', 'rnn-ema09', 'rnn-uniform', 'rnn-attention', 'rnn-block2', 'rnn-block4')

WAVES = {
    # Two rates per variant around each family's tuned baseline rate.
    'p3_wave1': [(name, (.003, .006)) for name in TRANSFORMER_P3 if name != 'd8-block2']
                + [('d8-block2', (.0045, .009))]
                + [(name, (.001, .002)) for name in RNN_P3],
    'p4_gru': [('gru', (.0015, .003, .006))],
    # Own P3 residual scheme and P4 ablations plus the new update rule.
    # GRU-style cells run about 110 min each, so ablations get one rate at the GRU centre.
    'wave2': [('d8-route', (.003, .006)), ('rnn-route', (.001, .002)),
              ('gru-dualnorm', (.003, .006)), ('gru-dual', (.003, .006)),
              ('gru-noreset', (.003,)), ('gru-inputgate', (.003,)), ('gru-linear', (.003,))],
    # The GRU's best rate was the top of its range, so extend it once.
    'gru_extend': [('gru', (.012,))],
    # EMA 0.1 collapsed to the unigram loss at both rates; check a lower rate once.
    'ema01_extend': [('d8-ema01', (.00075,))],
    # gru-dual went non-finite at 0.003 (step ~1700); retry at lower rates.
    'dual_extend': [('gru-dual', (.0015, .00075))],
    # block2's best rate (0.009) was the top of its range, so extend it once.
    'block2_extend': [('d8-block2', (.018,))],
    # Jobs that crashed with RecompileLimitExceeded in reused containers, plus the
    # last wave1 jobs that would have run in those containers.
    'relaunch': [('rnn-block2', (.001, .002)), ('rnn-block4', (.001, .002)),
                 ('d8-attention', (.003, .006)),
                 ('gru-noreset', (.003,)), ('gru-inputgate', (.003,)), ('gru-linear', (.003,))],
}


def run(wave, max_parallel_runs=2):
    """One Modal app per wave so the course's two-GPU limit queues the rest."""
    from modal_train import launch_training_jobs
    configs = [config(name, lr) for name, rates in WAVES[wave] for lr in rates]
    return launch_training_jobs(configs, max_parallel_runs=max_parallel_runs)


if __name__ == '__main__':
    run(sys.argv[1])
