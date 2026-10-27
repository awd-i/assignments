"""P3/P4 figures and tables from my runs: W&B summaries + probes."""
import sys, json, os, csv
sys.path.insert(0, 'report')
from plot_diag import *
import wandb

W = os.path.expanduser('~/a4work')
CACHE = f'{W}/my_runs.json'


def fetch():
    """Truth table recovered from the stdout logs (see truth.py): run -> final val loss."""
    truth = json.load(open(f'{W}/my_runs_truth.json'))
    rows = {}
    for run, entries in truth.items():
        if run is None or run == 'null':
            continue
        e = entries[-1]
        rows[run] = dict(state='finished' if e['final_val'] is not None else 'diverged', step=9374 if e['final_val'] else None,
                         val=e['final_val'], wandb_run=e['wandb_run'])
    # FP32 full-validation recomputation of every saved checkpoint is the primary source.
    for p in glob.glob(f'{RES}/*/val.json'):
        run = p.split('/')[-2]
        v = json.load(open(p))['val_loss']
        rows[run] = dict(state='finished', step=9374, val=v, wandb_run=rows.get(run, {}).get('wandb_run'))
    for run in ('a4-gru-dual-lr0.003', 'a4-gru-dual-lr0.0015', 'a4-gru-linear-lr0.003'):
        rows.setdefault(run, dict(state='diverged (NaN)', step=None, val=None))
    json.dump(rows, open(CACHE, 'w'), indent=1)
    return rows


def variant_lr(run):
    stem, lr = run[3:].rsplit('-lr', 1)
    return stem, float(lr)


def table(rows, variants, path, reference=None):
    """variant & rates/losses... one line per variant."""
    with open(path, 'w') as f:
        f.write('\\begin{tabular}{lp{0.78\\linewidth}}\\toprule\nvariant & final validation loss per peak rate\\\\\\midrule\n')
        if reference:
            for name, cells in reference:
                f.write(f"{name} & {cells} \\\\\n")
        for v in variants:
            runs = sorted([(variant_lr(r)[1], d) for r, d in rows.items() if variant_lr(r)[0] == v])
            if not runs:
                continue
            best = min((d['val'] for _, d in runs if d['val'] is not None and d['state'] == 'finished'), default=None)
            cells = []
            for lr, d in runs:
                if d['state'] != 'finished' or d['val'] is None:
                    cells.append(f"$\\lr$={lr:g}: {d['state']}")
                else:
                    cells.append((f"$\\lr$={lr:g}: \\textbf{{{d['val']:.3f}}}" if d['val'] == best else f"$\\lr$={lr:g}: {d['val']:.3f}"))
            f.write(f"{v} & " + '; '.join(cells) + " \\\\\n")
        f.write('\\bottomrule\\end{tabular}\n')


def best_run(rows, v):
    runs = [(d['val'], r) for r, d in rows.items() if variant_lr(r)[0] == v and d['state'] == 'finished' and d['val'] is not None]
    return min(runs)[1] if runs else None


def routing_plot(runs, name, title, stage='final', labels=None):
    """mean routing weights of the final read per source, for routed models."""
    fig, ax = plt.subplots(figsize=(6, 3))
    for i, run in enumerate(runs):
        d = load(run, stage)
        if d is None or not d.get('routing'):
            continue
        last = list(d['routing'].values())[-1]
        ax.plot(range(len(last)), last, marker='.', label=(labels[i] if labels else run.replace('a4-', '')))
    ax.set_xlabel('source index (0 = embedding, then block sums)'); ax.set_ylabel('mean routing weight of final read')
    ax.set_title(title, fontsize=10); ax.legend(fontsize=6); save(fig, name)


def gates_report(runs, stage='final'):
    """For the route models the scalar gates are not in the probes; report routing entropy instead."""
    out = {}
    for run in runs:
        d = load(run, stage)
        if d and d.get('routing'):
            ws = [np.array(w) for w in d['routing'].values()]
            out[run] = [float(-(w*np.log(w+1e-12)).sum()/np.log(len(w))) if len(w) > 1 else 0. for w in ws]
    return out


if __name__ == '__main__':
    rows = fetch() if 'nofetch' not in sys.argv else json.load(open(CACHE))
    T = ['d8-half', 'd8-ema01', 'd8-ema09', 'd8-uniform', 'd8-attention', 'd8-block2', 'd8-block4', 'd8-route']
    R = ['rnn-half', 'rnn-ema01', 'rnn-ema09', 'rnn-uniform', 'rnn-attention', 'rnn-block2', 'rnn-block4', 'rnn-route']
    G = ['gru', 'gru-noreset', 'gru-inputgate', 'gru-linear', 'gru-dual', 'gru-dualnorm']
    table(rows, T, f'{FIG}/p3_t_table.tex', reference=[('d8 (supplied)', '$\\lr$=0.003: 2.927 (best of five)')])
    table(rows, R, f'{FIG}/p3_r_table.tex', reference=[('rnn (supplied)', '$\\lr$=0.002: 3.516 (best of five)')])
    table(rows, G, f'{FIG}/p4_table.tex', reference=[('rnn, tanh cell (supplied)', '$\\lr$=0.002: 3.516 (best of five)')])
    for v in T + R + G:
        print(v, best_run(rows, v))
    bt = [b for b in (best_run(rows, v) for v in T) if b and load(b)]
    br = [b for b in (best_run(rows, v) for v in R) if b and load(b)]
    bg = [b for b in (best_run(rows, v) for v in G) if b and load(b)]
    if bt:
        depth_panels(['a4-d8-lr0.003'] + bt, 'p3_t_depth_final', 'P3 transformers: depth profiles of the final checkpoints')
        depth_panels(['a4-d8-lr0.003'] + bt, 'p3_t_depth_init', 'P3 transformers: depth profiles at initialisation', stage='init')
        depth_panels(['a4-d8-lr0.003'] + bt, 'p3_t_depth_early', 'P3 transformers: depth profiles at step 100', stage='early')
        routing_plot(bt, 'p3_t_routing', 'P3 transformers: routing weights of the final read')
        print('routing entropy T', gates_report(bt))
    if br:
        depth_panels(['a4-rnn-lr0.002'] + br, 'p3_r_depth_final', 'P3 RNNs: depth profiles of the final checkpoints')
        depth_panels(['a4-rnn-lr0.002'] + br, 'p3_r_depth_init', 'P3 RNNs: depth profiles at initialisation', stage='init')
        depth_panels(['a4-rnn-lr0.002'] + br, 'p3_r_depth_early', 'P3 RNNs: depth profiles at step 100', stage='early')
        routing_plot(br, 'p3_r_routing', 'P3 RNNs: routing weights of the final read')
        print('routing entropy R', gates_report(br))
    if bg:
        depth_panels(['a4-rnn-lr0.002'] + bg, 'p4_depth_final', 'P4: depth profiles of the final checkpoints')
        temporal_panels(['a4-rnn-lr0.002'] + bg, 'p4_temporal', 'P4: hidden state statistics over time, last layer (final checkpoints)', layer=-1)
        temporal_panels(['a4-rnn-lr0.002'] + bg, 'p4_temporal_init', 'P4: hidden state statistics over time, last layer (initialisation)', layer=-1, stage='init')
        context_plot(['a4-rnn-lr0.002'] + bg, 'p4_context', 'P4: context use')
        for b in bg:
            d = load(b)
            if d:
                print(b, 'temporal cos', [{k: round(v, 3) for k, v in tc.items()} for tc in d['temporal_cosine']])
                t = d['context']['tests']
                for first in (False, True):
                    print('  first', first, [(k, round(t[f'{k}_first_{first}']['mean_increase'], 2)) for k in (1, 4, 16, 64, 256)])
    for r in summary_rows(bt + br + bg):
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
