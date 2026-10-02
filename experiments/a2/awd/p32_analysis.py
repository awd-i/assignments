"""A2 Problem 3.2 analysis: batch-size scaling in the language model at 614.4M tokens.

B = 64 points come from supplied runs: P1a (WD .1, LR sweep) and P2a (LR .0015, WD .1/.2/.4).
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.a2.awd.p1_lr_fits import optimum
from experiments.a2.awd.target_results import load as load_targets
from experiments.a2.provided_sweeps import load as load_provided
from experiments.lr_tuning.plot_lr_tuning import set_style

OUT = Path(__file__).resolve().parent / "plots"
D = 614_400_000
set_style()


def table():
    """{batch: {(lr, wd, beta1): loss}}"""
    t = defaultdict(dict)
    for part in ("P1a", "P2a"):
        for r in load_provided(part):
            if r["tokens"] == D:
                t[64][(r["learning_rate"], r["weight_decay"], 0.9)] = r["final_val_loss"]
    for r in load_targets("a2-p32"):
        t[r["batch"]][(r["lr"], r["wd"], r["beta1"])] = r["val_loss"]
    return t


def sweep(t, B, fixed, axis):
    """Loss vs the free hyperparameter at fixed other one (axis 'lr' fixes WD; axis 'wd' fixes LR), beta1 .9."""
    pts = sorted((k[0] if axis == "lr" else k[1], v) for k, v in t[B].items()
                 if k[2] == 0.9 and (k[1] if axis == "lr" else k[0]) == fixed)
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def fit(x, y):
    if len(x) < 3:
        return float(x[np.argmin(y)]), float(y.min()), True
    i = int(np.argmin(y))
    if i in (0, len(x) - 1):
        return float(x[i]), float(y[i]), True
    lo, hi = i - 1, i + 2
    xs, ls, _ = optimum(x[lo:hi], y[lo:hi])
    return float(xs), float(ls), False


def power(bs, vals):
    p, c = np.polyfit(np.log(bs), np.log(vals), 1)
    return float(p), (lambda b: float(np.exp(c) * b**p))


if __name__ == "__main__":
    t = table()
    res = {}
    for hyp, axis, fixed in (("i", "lr", 0.1), ("ii", "wd", 0.0015)):
        rows = {}
        for B in sorted(t):
            x, y = sweep(t, B, fixed, axis)
            if len(x):
                rows[B] = dict(x=x.tolist(), y=y.tolist(), fit=fit(x, y))
                print(f"hyp {hyp} B={B:3d}: {axis}* {rows[B]['fit'][0]:.3g} loss {rows[B]['fit'][1]:.4f}"
                      f"{' EDGE' if rows[B]['fit'][2] else ''}  {dict(zip(np.round(x, 5), np.round(y, 4)))}")
        src = [B for B in rows if B <= 64 and not rows[B]["fit"][2]]
        if len(src) >= 2:
            p, f = power(src, [rows[B]["fit"][0] for B in src])
            print(f"   hyp {hyp}: {axis}* ~ B^{p:.2f} (fit on {src}) -> B=128: {f(128):.3g}, B=256: {f(256):.3g}")
            res[hyp] = dict(p=p, pred128=f(128), pred256=f(256), rows=rows)
    json.dump(res, open(OUT / "p32_summary.json", "w"), indent=1, default=float)
    for B in sorted(t):
        print(B, {k: round(v, 4) for k, v in sorted(t[B].items())})


def plot_all(t, res):
    Bs = sorted(t)
    cols = dict(zip(Bs, plt.cm.viridis(np.linspace(0.05, 0.9, len(Bs)))))
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.5))
    for B in Bs:
        x, y = sweep(t, B, 0.1, "lr")
        if len(x):
            axes[0].plot(x, y, "o-", color=cols[B], markeredgecolor="black", label=f"B={B}")
        x, y = sweep(t, B, 0.0015, "wd")
        if len(x):
            axes[1].plot(x, y, "o-", color=cols[B], markeredgecolor="black", label=f"B={B}")
    axes[0].set_title("(a)/(b-i) Loss vs peak LR at WD .1")
    axes[0].set_xlabel("Peak LR")
    axes[1].set_title("(b-ii) Loss vs WD at peak LR .0015")
    axes[1].set_xlabel("Weight decay")
    for a in axes[:2]:
        a.set_xscale("log")
        a.set_ylabel("Final val loss (614.4M tokens)")
        a.grid(True, linestyle=":", alpha=0.3)
        a.legend(frameon=False, fontsize=8)
    for hyp, key, col in (("i", "LR*", "#3a7bd5"), ("ii", "WD*", "#e76f51")):
        if hyp not in res:
            continue
        rows = res[hyp]["rows"]
        b = sorted(rows)
        ax = axes[2] if hyp == "i" else axes[2].twinx()
        ax.plot(b, [rows[x]["fit"][0] for x in b], "o-" if hyp == "i" else "s--", color=col, markeredgecolor="black",
                label=f"{key} (hyp {hyp})")
        g = np.array([8, 256])
        src = [x for x in b if x <= 64]
        ax.plot(g, [res[hyp]["pred128"] * (v / 128) ** res[hyp]["p"] for v in g], ":", color=col,
                label=f"fit B<=64: B^{res[hyp]['p']:.2f}")
        ax.set_yscale("log")
        ax.set_ylabel(key, color=col)
        ax.legend(frameon=False, fontsize=8, loc="upper left" if hyp == "i" else "lower right")
    axes[2].set_xscale("log", base=2)
    axes[2].set_xlabel("Batch size B")
    axes[2].set_title("Fitted optimum vs batch")
    best = {B: min(t[B].values()) for B in Bs}
    axes[3].plot(Bs, [best[B] for B in Bs], "o-", color="#7030A0", markeredgecolor="black")
    for B in Bs:
        axes[3].annotate(f"{best[B]:.4f}", (B, best[B]), textcoords="offset points", xytext=(4, 4), fontsize=7)
    axes[3].set_xscale("log", base=2)
    axes[3].set_xlabel("Batch size B")
    axes[3].set_ylabel("Best measured val loss")
    axes[3].set_title("Best loss vs batch (fixed 614.4M tokens)")
    axes[3].grid(True, linestyle=":", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "p32_ab.png")


if __name__ == "__main__":
    plot_all(t, res)
    print("plotted")
