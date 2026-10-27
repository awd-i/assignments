"""Plots from measure_all JSON probes (results/<run>/<stage>.json)."""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

W = os.path.expanduser('~/a4work')
RES = f'{W}/results'
FIG = f'{W}/report/figures'
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .3, 'figure.dpi': 150})
STAGES = ('init', 'early', 'final')


def load(run, stage='final'):
    p = f'{RES}/{run}/{stage}.json'
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def save(fig, name):
    fig.tight_layout(); fig.savefig(f'{FIG}/{name}.pdf'); fig.savefig(f'{FIG}/{name}.png'); plt.close(fig)


def depth_panels(runs, name, title, stage='final', objective='full', labels=None):
    """activation RMS, gradient RMS (full and last), branch ratio across depth."""
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.9))
    for i, run in enumerate(runs):
        d = load(run, stage)
        if d is None:
            continue
        lab = labels[i] if labels else run.replace('a4-', '')
        full, last = d['full']['depth'], d['last']['depth']
        x = np.arange(len(full))
        axes[0].plot(x, [r['activation_rms'] for r in full], marker='.', label=lab)
        axes[1].plot(x, [r['gradient_rms'] for r in full], marker='.', label=lab)
        axes[2].plot(x, [r['gradient_rms'] for r in last], marker='.', label=lab)
        axes[3].plot(x[:-1], [r.get('raw_branch_input_ratio', np.nan) for r in full[:-1]], marker='.', label=lab)
    for ax, t in zip(axes, ['activation RMS', 'grad RMS (full loss)', 'grad RMS (last-token loss)', 'branch RMS / input RMS']):
        ax.set_title(t, fontsize=9); ax.set_xlabel('branch input index (last = final stream)'); ax.set_yscale('log')
    axes[0].legend(fontsize=6)
    fig.suptitle(title, fontsize=10)
    save(fig, name)


def stage_panels(run, name, title):
    """Same run at init / step 100 / final."""
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.9))
    for stage in STAGES:
        d = load(run, stage)
        if d is None:
            continue
        full, last = d['full']['depth'], d['last']['depth']
        x = np.arange(len(full))
        axes[0].plot(x, [r['activation_rms'] for r in full], marker='.', label=stage)
        axes[1].plot(x, [r['gradient_rms'] for r in full], marker='.', label=stage)
        axes[2].plot(x, [r['gradient_rms'] for r in last], marker='.', label=stage)
        axes[3].plot(x[1:], [r.get('adjacent_cosine', np.nan) for r in full[1:]], marker='.', label=stage)
    for ax, t in zip(axes, ['activation RMS', 'grad RMS (full)', 'grad RMS (last token)', 'cosine to previous stream state']):
        ax.set_title(t, fontsize=9); ax.set_xlabel('depth index')
    for ax in axes[:3]:
        ax.set_yscale('log')
    axes[0].legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    save(fig, name)


def attention_panels(runs, name, title, stage='final', labels=None):
    """first-token mass and self mass vs query position, averaged over layers, vs uniform causal baseline."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.9))
    for i, run in enumerate(runs):
        d = load(run, stage)
        if d is None or not d.get('attention'):
            continue
        lab = labels[i] if labels else run.replace('a4-', '')
        layers = list(d['attention'].values())
        q = np.array(layers[0]['query_positions'])
        first = np.mean([np.mean(l['first_mass'], 0) for l in layers], 0)
        self_m = np.mean([np.mean(l['self_mass'], 0) for l in layers], 0)
        lag = np.mean([np.mean(l['mean_lag'], 0) for l in layers], 0)
        axes[0].plot(q, first, marker='.', label=lab); axes[1].plot(q, self_m, marker='.', label=lab); axes[2].plot(q, lag, marker='.', label=lab)
    axes[0].plot(q, 1/(q+1), 'k--', label='uniform causal'); axes[1].plot(q, 1/(q+1), 'k--', label='uniform causal'); axes[2].plot(q, q/2, 'k--', label='uniform causal')
    for ax, t in zip(axes, ['mass on first token', 'mass on self', 'mean attention lag']):
        ax.set_title(t, fontsize=9); ax.set_xlabel('query position'); ax.set_xscale('log')
    axes[0].set_yscale('log'); axes[1].set_yscale('log'); axes[2].set_yscale('log')
    axes[0].legend(fontsize=6)
    fig.suptitle(title, fontsize=10)
    save(fig, name)


def attention_by_layer(run, name, title, stage='final'):
    d = load(run, stage)
    layers = d['attention']
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.9))
    for lname, l in layers.items():
        q = np.array(l['query_positions'])
        axes[0].plot(q, np.mean(l['first_mass'], 0), marker='.', label=lname.split('.')[-2] if '.' in lname else lname)
        axes[1].plot(q, np.mean(l['mean_lag'], 0), marker='.')
    axes[0].plot(q, 1/(q+1), 'k--', label='uniform'); axes[1].plot(q, q/2, 'k--', label='uniform')
    axes[0].set_title('mass on first token (per layer)', fontsize=9); axes[1].set_title('mean lag (per layer)', fontsize=9)
    for ax in axes:
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('query position')
    axes[0].legend(fontsize=5, ncol=2)
    fig.suptitle(title, fontsize=10); save(fig, name)


def temporal_panels(runs, name, title, stage='final', labels=None, layer=None):
    """hidden RMS, hidden grad RMS (full and last), saturation vs time."""
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.9))
    for i, run in enumerate(runs):
        d = load(run, stage)
        if d is None or not d['full']['hidden_rms']:
            continue
        lab = labels[i] if labels else run.replace('a4-', '')
        L = len(d['full']['hidden_rms'])
        js = [layer] if layer is not None else list(range(L))
        for j in js:
            suffix = f' L{j}' if len(js) > 1 else ''
            t = np.arange(len(d['full']['hidden_rms'][j]))
            axes[0].plot(t, d['full']['hidden_rms'][j], lw=.8, label=lab+suffix)
            axes[1].plot(t, d['full']['hidden_gradient_rms'][j], lw=.8, label=lab+suffix)
            axes[2].plot(t, d['last']['hidden_gradient_rms'][j], lw=.8, label=lab+suffix)
            axes[3].plot(t, d['full']['fraction_abs_gt_095'][j], lw=.8, label=lab+suffix)
    for ax, t in zip(axes, ['hidden state RMS', 'hidden grad RMS (full loss)', 'hidden grad RMS (last-token loss)', 'fraction |h| > 0.95']):
        ax.set_title(t, fontsize=9); ax.set_xlabel('time step t')
    axes[1].set_yscale('log'); axes[2].set_yscale('log')
    axes[0].legend(fontsize=5, ncol=2)
    fig.suptitle(title, fontsize=10); save(fig, name)


def context_plot(runs, name, title, labels=None, stage='final'):
    recurrent = any(load(r, stage) and load(r, stage)['last']['hidden_gradient_rms'] for r in runs)
    fig, axes = plt.subplots(1, 2 if recurrent else 1, figsize=(8 if recurrent else 4.5, 3), squeeze=False); axes = axes[0]
    for i, run in enumerate(runs):
        d = load(run, stage)
        if d is None or 'context' not in d:
            continue
        lab = labels[i] if labels else run.replace('a4-', '')
        tests = d['context']['tests']
        for first, ls in ((False, '-'), (True, '--')):
            keeps = sorted({int(k.split('_')[0]) for k in tests})
            ys = [tests.get(f'{k}_first_{first}', tests.get(f'{k}_first_False'))['mean_increase'] for k in keeps]
            axes[0].plot(keeps, ys, ls, marker='.', label=f'{lab}{" +first token" if first else ""}')
    axes[0].set_xscale('log'); axes[0].set_xlabel('recent tokens kept')
    axes[0].set_ylabel('last-token loss increase vs full context (nats)')
    axes[0].legend(fontsize=5)
    if not recurrent:
        fig.suptitle(title, fontsize=10); save(fig, name); return
    # right panel: last-token gradient reach for the same runs
    for i, run in enumerate(runs):
        d = load(run, stage)
        if d is None:
            continue
        lab = labels[i] if labels else run.replace('a4-', '')
        if d['last']['hidden_gradient_rms']:
            g = np.array(d['last']['hidden_gradient_rms'][-1])
            t = np.arange(len(g)); lagv = len(g)-t
            axes[1].plot(lagv, g, lw=.8, label=lab+' (last layer h)')
    axes[1].set_xscale('log'); axes[1].set_yscale('log'); axes[1].set_xlabel('distance from last token'); axes[1].set_title('last-token loss gradient reach', fontsize=9)
    axes[1].legend(fontsize=5)
    fig.suptitle(title, fontsize=10); save(fig, name)


def summary_rows(runs, stage='final'):
    rows = []
    for run in runs:
        d = load(run, stage)
        if d is None:
            continue
        full = d['full']['depth']
        rows.append(dict(run=run, loss=d['full']['loss'], last_loss=d['last']['loss'],
                         act_first=full[0]['activation_rms'], act_last=full[-1]['activation_rms'],
                         grad_first=full[0]['gradient_rms'], grad_last=full[-1]['gradient_rms'],
                         last_grad_first=d['last']['depth'][0]['gradient_rms'], last_grad_last=d['last']['depth'][-1]['gradient_rms'],
                         ratio=np.nanmean([r.get('raw_branch_input_ratio', np.nan) for r in full]),
                         cos=np.nanmean([r.get('adjacent_cosine', np.nan) for r in full[1:]])))
    return rows


def position_panels(runs, name, title, labels=None):
    """activation RMS vs position (first, middle, last depth index) and cosine between positions vs lag per depth."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for i, run in enumerate(runs):
        d = load(run, 'position')
        if d is None:
            continue
        lab = labels[i] if labels else run.replace('a4-', '')
        depth = d['depth']
        color = None
        for j, ls in ((0, ':'), (len(depth)//2, '--'), (len(depth)-1, '-')):
            r = np.array(depth[j]['rms_by_position'])
            line, = axes[0].plot(np.arange(1, len(r)+1), r, linestyle=ls, lw=.9, color=color, label=f'{lab} depth {j}')
            color = line.get_color()
        lags = sorted(int(k) for k in depth[0]['cosine_by_lag'])
        for lag, ls in zip(lags, (':', '-.', '--', '-', (0, (1, 1)))):
            axes[1].plot(range(len(depth)), [dd['cosine_by_lag'][str(lag)] for dd in depth], linestyle=ls, color=color, marker='.', ms=3, lw=.9,
                         label=f'{lab} lag {lag}')
        axes[2].plot(range(len(depth)), [dd['cosine_to_first'][str(lags[-1])] for dd in depth], marker='.', color=color, label=f'{lab}')
    axes[0].set_xlabel('position t + 1'); axes[0].set_ylabel('activation RMS'); axes[0].set_yscale('log'); axes[0].set_xscale('log'); axes[0].set_title('stream RMS by position', fontsize=9)
    axes[1].set_xlabel('depth index'); axes[1].set_ylabel('cosine(x_t, x_{t-lag})'); axes[1].set_title('similarity across positions', fontsize=9)
    axes[2].set_xlabel('depth index'); axes[2].set_ylabel('cosine(x_256, x_0)'); axes[2].set_title('similarity to the first position', fontsize=9)
    axes[0].legend(fontsize=5); axes[1].legend(fontsize=4, ncol=2); axes[2].legend(fontsize=6)
    fig.suptitle(title, fontsize=10); save(fig, name)
