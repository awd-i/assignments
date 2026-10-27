"""P1/P2 (and later P3/P4) plots from W&B summaries and histories."""
import json, glob, os, csv
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

W = os.path.expanduser('~/a4work')
FIG = os.path.join(W, 'report/figures')
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .3, 'figure.dpi': 150})

rows = list(csv.DictReader(open(f'{W}/p12_summary.csv')))
by_variant = defaultdict(list)
for r in rows:
    r['lr'] = float(r['lr']); r['final_val'] = float(r['final_val']) if r['final_val'] else float('nan')
    by_variant[r['variant']].append(r)

P1 = ['d8', 'd8-no-residual', 'd8-no-prenorm', 'd8-no-qknorm', 'd8-no-rope', 'd8-no-final-norm', 'd8-postnorm',
      'd8-gated-residual', 'd8-gated-attention', 'd8-gated-norm']
P2 = ['rnn1', 'rnn4', 'rnn', 'rnn-no-residual', 'rnn-no-prenorm', 'rnn-rope', 'rnn-final-norm', 'rnn-postnorm',
      'rnn-gated-residual', 'rnn-gated-output', 'rnn-gated-norm']


def lr_plot(variants, name, title, top=4.7):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    markers = 'osD^v<>ph*X'
    for i, v in enumerate(variants):
        rs = sorted(by_variant[v], key=lambda r: r['lr'])
        x = [r['lr'] for r in rs]; y = [r['final_val'] for r in rs]
        ax.plot(x, y, marker=markers[i % len(markers)], label=v, lw=1, ms=4)
        for r in rs:
            if r['diverged'] or r['failure'] or not np.isfinite(r['final_val']) or r['final_val'] > top-.1:
                ax.plot(r['lr'], top-.1, 'x', color=ax.lines[-1].get_color() if False else 'red', ms=7)
    ax.set_xscale('log'); ax.set_xlabel('peak learning rate'); ax.set_ylabel('final validation loss')
    ax.set_title(title + ' (red x: diverged or collapsed)'); ax.set_ylim(2.8, top); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(f'{FIG}/{name}.pdf'); fig.savefig(f'{FIG}/{name}.png'); plt.close(fig)


def hist(run):
    p = f'{W}/hist/{run}.json'
    return json.load(open(p)) if os.path.exists(p) else None


def curves(runs, name, title, key='train_loss', ylim=None, smooth=20):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for run in runs:
        h = hist(run)
        if h is None:
            continue
        pts = [(r['step'], r[key]) for r in h if key in r and r[key] is not None]
        if not pts:
            continue
        s, y = np.array(pts).T
        if smooth and key == 'train_loss':
            k = np.ones(smooth)/smooth; y = np.convolve(y, k, 'valid'); s = s[smooth-1:]
        ax.plot(s, y, lw=1, label=run.replace('a4-', ''))
    ax.set_xlabel('step'); ax.set_ylabel(key.replace('logging/rms/global/', 'global RMS of '))
    if key.startswith('logging'):
        ax.set_yscale('log')
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_title(title); ax.legend(fontsize=6, ncol=2); fig.tight_layout(); fig.savefig(f'{FIG}/{name}.pdf'); fig.savefig(f'{FIG}/{name}.png'); plt.close(fig)


def best(v):
    rs = [r for r in by_variant[v] if np.isfinite(r['final_val'])]
    return min(rs, key=lambda r: r['final_val']) if rs else None


def table(variants, path):
    with open(path, 'w') as f:
        f.write('\\begin{tabular}{lrrrrrrr}\\toprule\nvariant & best $\\lr$ & $0.25\\times$ & $0.5\\times$ & $1\\times$ & $2\\times$ & $4\\times$ & params (M)\\\\\\midrule\n')
        for v in variants:
            rs = sorted(by_variant[v], key=lambda r: r['lr'])
            b = best(v)
            cells = []
            for r in rs:
                if np.isfinite(r['final_val']):
                    cells.append(('\\textbf{%.3f}' if r is b else '%.3f') % r['final_val'])
                else:
                    cells.append('div.')
            f.write(f"{v} & {b['lr']:g} & " + ' & '.join(cells) + f" & {int(b['params'])/1e6:.1f} \\\\\n")
        f.write('\\bottomrule\\end{tabular}\n')


if __name__ == '__main__':
    lr_plot(P1, 'p1_lr_sweep', 'P1: transformer variants, final validation loss vs learning rate')
    lr_plot(P2, 'p2_lr_sweep', 'P2: RNN variants, final validation loss vs learning rate')
    table(P1, f'{FIG}/p1_table.tex'); table(P2, f'{FIG}/p2_table.tex')
    for v in P1 + P2:
        print(v, [(r['lr'], round(r['final_val'], 3), r['diverged'], r['failure']) for r in sorted(by_variant[v], key=lambda r: r['lr'])])
    curves([f"a4-{v}-lr{best(v)['lr']:g}" for v in P1 if best(v)], 'p1_train_curves', 'P1: training loss at each variant\'s best rate', ylim=(2.8, 5))
    curves([f"a4-{v}-lr{best(v)['lr']:g}" for v in P2 if best(v)], 'p2_train_curves', 'P2: training loss at each variant\'s best rate', ylim=(2.8, 5.5))
    for key, tag in [('logging/rms/global/activation', 'act'), ('logging/rms/global/gradient', 'grad')]:
        curves(['a4-d8-lr0.003', 'a4-d8-no-residual-lr0.00075', 'a4-d8-no-prenorm-lr0.001', 'a4-d8-no-prenorm-lr0.004',
                'a4-d8-postnorm-lr0.001', 'a4-d8-postnorm-lr0.004', 'a4-d8-no-qknorm-lr0.012', 'a4-d8-gated-norm-lr0.00225'],
               f'p1_rms_{tag}', f'P1: logged global {tag} RMS, stable and divergent runs', key=key)
        curves(['a4-rnn-lr0.002', 'a4-rnn1-lr0.003', 'a4-rnn4-lr0.001', 'a4-rnn-no-residual-lr0.001', 'a4-rnn-no-prenorm-lr0.002',
                'a4-rnn-postnorm-lr0.001', 'a4-rnn-no-prenorm-lr0.004', 'a4-rnn-postnorm-lr0.004'],
               f'p2_rms_{tag}', f'P2: logged global {tag} RMS, stable and divergent runs', key=key)
