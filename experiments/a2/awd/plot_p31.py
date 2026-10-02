"""A2 Problem 3.1 plots from p31_{ab,c,d}.json (+ learning curves recomputed for (d))."""

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.a2.awd.p31_nqm import SOURCE, lr_grid, simulate
from experiments.lr_tuning.plot_lr_tuning import set_style

OUT = Path(__file__).resolve().parent / "plots"
set_style()
C = {"sgd": "#264653", "rmsprop": "#e76f51", "adam": "#2a9d8f"}

ab = json.loads((OUT / "p31_ab.json").read_text())
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
for opt, r in ab.items():
    Bs = np.array(sorted(int(b) for b in r["res"]))
    lrs = np.array([r["res"][str(b)]["lr"] for b in Bs])
    a1.plot(Bs, lrs, "o-", color=C[opt], markeredgecolor="black", label=f"{opt} (fit on B<=64: p={r['p']:.2f})")
    g = np.array([1, 512])
    a1.plot(g, r["c"] * g ** r["p"], "--", color=C[opt], alpha=0.6)
    a1.scatter([256], [r["pred256"]], marker="*", s=180, color=C[opt], edgecolor="black", zorder=5)
    a2.plot(Bs, [r["res"][str(b)]["loss"] for b in Bs], "o-", color=C[opt], markeredgecolor="black", label=opt)
a1.axvspan(80, 600, color="gray", alpha=0.08)
a1.text(95, 6e-4, "held out", color="gray")
for a in (a1, a2):
    a.set_xscale("log", base=2)
    a.set_yscale("log")
    a.set_xlabel("Batch size B (N = 8192 examples fixed)")
    a.grid(True, linestyle=":", alpha=0.3)
a1.set_ylabel("Optimal LR")
a1.set_title("(a,b) Optimal LR vs batch (star = prediction at 256)")
a2.set_ylabel("Best expected final loss")
a2.set_title("Loss after tuning LR")
a1.legend(frameon=False, fontsize=8.5)
a2.legend(frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(OUT / "p31_ab.png")

c = json.loads((OUT / "p31_c.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
for ax, curv in zip(axes, ("2d", "scalar")):
    for opt in ("sgd", "rmsprop", "adam"):
        sig = [1, 10, 100, 300]
        ax.plot(sig, [c[f"{curv}-{opt}-{s}"]["p"] for s in sig], "o-", color=C[opt], markeredgecolor="black", label=opt)
    ax.axhline(1, color="gray", linestyle=":")
    ax.axhline(0.5, color="gray", linestyle=":")
    ax.set_xscale("log")
    ax.set_xlabel("Noise scale sigma")
    ax.set_title("H = diag(1, 10)" if curv == "2d" else "scalar f = w^2/2")
    ax.grid(True, linestyle=":", alpha=0.3)
axes[0].set_ylabel("Fitted exponent p (LR* ~ B^p, B = 1..64)")
axes[0].legend(frameon=False)
fig.suptitle("(c) Fitted batch-size exponent vs gradient noise")
fig.tight_layout()
fig.savefig(OUT / "p31_c.png")

d = json.loads((OUT / "p31_d.json").read_text())
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
b1s = [0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995]
mus = [0.0, 0.3, 0.5, 0.7, 0.9]
for B, col in ((16, "#3a7bd5"), (256, "#e76f51")):
    base = d[f"adam-B{B}-b10.0"]["loss"]
    axes[0].plot(b1s, [d[f"adam-B{B}-b1{b}"]["loss"] / base for b in b1s], "o-", color=col, markeredgecolor="black",
                 label=f"Adam B={B}")
    sb = d[f"sgd-B{B}-mu0.0"]["loss"]
    axes[0].plot(mus, [d[f"sgd-B{B}-mu{m}"]["loss"] / sb for m in mus], "s--", color=col, markeredgecolor="black",
                 alpha=0.7, label=f"SGD B={B} (x = mu)")
axes[0].axhline(1, color="gray", linestyle=":")
axes[0].set_yscale("log")
axes[0].set_xlabel("beta1 (Adam) / mu (SGD)")
axes[0].set_ylabel("Best loss / best loss without momentum")
axes[0].set_title("(d) Benefit of momentum after tuning LR")
axes[0].grid(True, linestyle=":", alpha=0.3)
axes[0].legend(frameon=False, fontsize=8)
for ax, B in zip(axes[1:], (16, 256)):
    best_b1 = min(b1s, key=lambda b: d[f"adam-B{B}-b1{b}"]["loss"])
    best_mu = min(mus, key=lambda m: d[f"sgd-B{B}-mu{m}"]["loss"])
    runs = [("Adam beta1=0", "adam", dict(beta1=0.0), d[f"adam-B{B}-b10.0"]["lr"], "#2a9d8f", ":"),
            (f"Adam beta1={best_b1}", "adam", dict(beta1=best_b1), d[f"adam-B{B}-b1{best_b1}"]["lr"], "#2a9d8f", "-"),
            ("SGD mu=0", "sgd", dict(mu=0.0), d[f"sgd-B{B}-mu0.0"]["lr"], "#264653", ":"),
            (f"SGD mu={best_mu}", "sgd", dict(mu=best_mu), d[f"sgd-B{B}-mu{best_mu}"]["lr"], "#264653", "-")]
    for label, opt, kw, lr, col, ls in runs:
        _, cur = simulate(opt, B, [lr], curve=True, **kw)
        ax.plot(np.arange(1, cur.shape[1] + 1) * B, cur[0], ls, color=col, label=f"{label} (LR {lr:.2g})")
    ax.set_yscale("log")
    ax.set_xlabel("Examples processed")
    ax.set_ylabel("Expected loss")
    ax.set_title(f"Learning curves at B={B} (tuned LR)")
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "p31_d.png")
print("ok")
