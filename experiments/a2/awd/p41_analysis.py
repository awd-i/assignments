"""A2 Problem 4.1 analysis: fit loss-LR curves, plot optimal LR / min loss vs width and depth, and probes.

Reads JSONs downloaded from the volume into plots/p41/ (modal volume get ... /a2-stress/awd/ ...).
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.lr_tuning.plot_lr_tuning import set_style

HERE = Path(__file__).resolve().parent
RAW = HERE / "plots" / "p41"
OUT = HERE / "plots"
set_style()


def load():
    rows = []
    for p in sorted(RAW.glob("*.json")):
        r = json.loads(p.read_text())
        j = r["job"]
        final = r["history"][-1]["val_loss"] if "history" in r else float("nan")
        if not np.isfinite(final):
            final = float("nan")
        rows.append(dict(**j, loss=final, result=r))
    return rows


def fit(lrs, losses):
    """Parabola in log2 LR through the best finite point and its neighbours. Returns (lr*, loss*, edge)."""
    lrs, losses = np.asarray(lrs), np.asarray(losses)
    ok = np.isfinite(losses)
    lrs, losses = lrs[ok], losses[ok]
    i = int(np.argmin(losses))
    if i == 0 or i == len(lrs) - 1:
        return float(lrs[i]), float(losses[i]), True
    x = np.log2(lrs[i - 1:i + 2])
    a, b, c = np.polyfit(x, losses[i - 1:i + 2], 2)
    xs = -b / (2 * a)
    return float(2**xs), float(a * xs**2 + b * xs + c), False


def groups(rows, key):
    g = defaultdict(list)
    for r in rows:
        g[key(r)].append(r)
    return {k: sorted(v, key=lambda r: r["lr"]) for k, v in g.items()}


def summary(rows, key):
    out = {}
    for k, v in groups(rows, key).items():
        lr, loss, edge = fit([r["lr"] for r in v], [r["loss"] for r in v])
        best = min((r for r in v if np.isfinite(r["loss"])), key=lambda r: r["loss"])
        out[k] = dict(lr=lr, loss=loss, edge=edge, best=best, runs=v)
    return out


def loss_lr_panels(rows, key, panel_key, label, fname, title):
    by_panel = groups(rows, panel_key)
    fig, axes = plt.subplots(1, len(by_panel), figsize=(5.2 * len(by_panel), 4.2), squeeze=False)
    for ax, (pk, prs) in zip(axes[0], sorted(by_panel.items())):
        for k, v in sorted(groups(prs, key).items()):
            lrs = [r["lr"] for r in v]
            ls = [min(r["loss"], 12) if np.isfinite(r["loss"]) else np.nan for r in v]
            ax.plot(lrs, ls, "o-", markeredgecolor="black", label=label(k))
            lr, loss, edge = fit(lrs, [r["loss"] for r in v])
            ax.scatter([lr], [loss], marker="*", s=160, edgecolor="black", zorder=5)
        ax.set_xscale("log")
        ax.set_xlabel("Base LR")
        ax.set_title(f"{title} {pk}")
        ax.grid(True, linestyle=":", alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    axes[0][0].set_ylabel("Val loss after 5 updates")
    fig.tight_layout()
    fig.savefig(OUT / fname)


def alignment_by_matrix(r, step=None):
    """Mean update-alignment exponent alpha per matrix type (q,k,v,o,gate,up,down,head) over updates."""
    out = defaultdict(list)
    for a in r["result"].get("alignment", []):
        if a.get("alpha") is None or (step is not None and a["step"] != step):
            continue
        name = a["parameter"].split(".")[-2]
        out[name].append(a["alpha"])
    return {k: float(np.mean(v)) for k, v in out.items()}


if __name__ == "__main__":
    rows = load()
    width = [r for r in rows if r["precision"] == "fp32"]
    depth = [r for r in rows if r["precision"] == "mp"]
    ws = summary(width, lambda r: (r["policy"], r["width"]))
    print("WIDTH")
    for k, s in sorted(ws.items()):
        print(k, f"LR* {s['lr']:.3g} loss* {s['loss']:.4f} edge {s['edge']}  best sampled {s['best']['lr']:.3g} "
              f"{s['best']['loss']:.4f}", [f"{r['lr']:.3g}:{r['loss']:.3f}" for r in s["runs"]])
    # depth-2 runs are shared by all three prescriptions (r = 1)
    depth2 = [r for r in depth if r["depth"] == 2]
    depth_all = [r for r in depth if r["depth"] != 2] + [dict(r, policy=p) for r in depth2
                                                          for p in ("mup", "depth-mup", "completep")]
    ds = summary(depth_all, lambda r: (r["policy"], r["depth"]))
    print("DEPTH")
    for k, s in sorted(ds.items()):
        print(k, f"LR* {s['lr']:.3g} loss* {s['loss']:.4f} edge {s['edge']}",
              [f"{r['lr']:.3g}:{r['loss']:.3f}" for r in s["runs"]])
    json.dump({"width": {f"{k[0]}-{k[1]}": {kk: vv for kk, vv in s.items() if kk not in ("runs", "best")} | {
        "best_lr": s["best"]["lr"], "best_loss": s["best"]["loss"]} for k, s in ws.items()},
               "depth": {f"{k[0]}-{k[1]}": {kk: vv for kk, vv in s.items() if kk not in ("runs", "best")} | {
                   "best_lr": s["best"]["lr"], "best_loss": s["best"]["loss"]} for k, s in ds.items()}},
              open(OUT / "p41_summary.json", "w"), indent=1)

    # (a) loss-LR curves and transfer plots
    loss_lr_panels(width, lambda r: r["width"], lambda r: r["policy"], lambda k: f"width {k}", "p41_width_curves.png",
                   "")
    loss_lr_panels(depth_all, lambda r: r["depth"], lambda r: r["policy"], lambda k: f"depth {k}",
                   "p41_depth_curves.png", "")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for pol, col in (("kaiming", "#e76f51"), ("mup", "#2a9d8f")):
        W = sorted(w for p, w in ws if p == pol)
        a1.plot(W, [ws[(pol, w)]["lr"] for w in W], "o-", color=col, markeredgecolor="black", label=pol)
        a2.plot(W, [ws[(pol, w)]["loss"] for w in W], "o-", color=col, markeredgecolor="black", label=pol)
    for a in (a1, a2):
        a.set_xscale("log")
        a.set_xlabel("Width")
        a.grid(True, linestyle=":", alpha=0.3)
        a.legend(frameon=False)
    a1.set_yscale("log")
    a1.set_ylabel("Fitted optimal base LR")
    a2.set_ylabel("Fitted min val loss (5 updates)")
    fig.tight_layout()
    fig.savefig(OUT / "p41_width_transfer.png")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for pol, col in (("mup", "#888888"), ("depth-mup", "#3a7bd5"), ("completep", "#7030A0")):
        D = sorted(d for p, d in ds if p == pol)
        a1.plot(D, [ds[(pol, d)]["lr"] for d in D], "o-", color=col, markeredgecolor="black", label=pol)
        a2.plot(D, [ds[(pol, d)]["loss"] for d in D], "o-", color=col, markeredgecolor="black", label=pol)
    for a in (a1, a2):
        a.set_xscale("log")
        a.set_xlabel("Depth (layers)")
        a.grid(True, linestyle=":", alpha=0.3)
        a.legend(frameon=False)
    a1.set_yscale("log")
    a1.set_ylabel("Fitted optimal base LR")
    a2.set_ylabel("Fitted min val loss (5 updates)")
    fig.tight_layout()
    fig.savefig(OUT / "p41_depth_transfer.png")

    # (b) width probes at each config's best sampled LR
    mats = ["q", "k", "v", "o", "gate", "up", "down", "head"]
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.4))
    for pol, ls in (("kaiming", "--"), ("mup", "-")):
        for w, col in zip((640, 2560, 5120), ("#2a9d8f", "#3a7bd5", "#7030A0")):
            if (pol, w) not in ws:
                continue
            r = ws[(pol, w)]["best"]
            h = r["result"]["history"]
            steps = [x["step"] for x in h]
            axes[0].plot(steps, [x["logit_rms"] for x in h], ls, color=col, label=f"{pol} w{w}")
            axes[1].plot(steps, [x["features"]["final_norm"]["movement"] for x in h], ls, color=col, label=f"{pol} w{w}")
            al = alignment_by_matrix(r)
            axes[2].plot(range(len(mats)), [al.get(m, np.nan) for m in mats], "o" + ls, color=col, label=f"{pol} w{w}")
            om = [x["readout_alignment"].get("movement", {}).get("omega") for x in h[1:]]
            axes[3].plot(steps[1:], [np.nan if o is None else o for o in om], "o" + ls, color=col, label=f"{pol} w{w}")
    axes[0].set_title("Logit RMS (best sampled LR)")
    axes[1].set_title("Final-norm feature movement M_t")
    axes[2].set_xticks(range(len(mats)), mats)
    axes[2].set_title("Update alignment alpha (mean over 5 updates)")
    for y in (0.5, 1.0):
        axes[2].axhline(y, color="gray", linestyle=":")
        axes[3].axhline(y, color="gray", linestyle=":")
    axes[3].set_title("Readout/feature-change alignment omega_move")
    for a in axes:
        a.grid(True, linestyle=":", alpha=0.3)
    for a in (axes[0], axes[1], axes[3]):
        a.set_xlabel("Update")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "p41_width_probes.png")

    # (d) depth probes
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.4))
    for pol, col in (("mup", "#888888"), ("depth-mup", "#3a7bd5"), ("completep", "#7030A0")):
        for d, ls in ((2, ":"), (100, "--"), (1000, "-")):
            if (pol, d) not in ds:
                continue
            r = ds[(pol, d)]["best"]
            h = r["result"]["history"]
            steps = [x["step"] for x in h]
            last = f"blocks.{d - 1}.norm1"
            mid = f"blocks.{d // 2}.norm1"
            axes[0].plot(steps, [x["residual_rms"]["final_norm"] for x in h], ls, color=col, label=f"{pol} L{d}")
            axes[1].plot(steps, [x["unscaled_branch_rms"].get(f"blocks.{d - 1}.down", np.nan) for x in h], ls,
                         color=col, label=f"{pol} L{d}")
            axes[2].plot(steps, [x["features"][mid]["movement"] for x in h], ls, color=col, label=f"{pol} L{d} mid")
            om = [x["readout_alignment"].get("movement", {}).get("omega") for x in h[1:]]
            axes[3].plot(steps[1:], [np.nan if o is None else o for o in om], "o" + ls, color=col, label=f"{pol} L{d}")
    axes[0].set_yscale("log")
    axes[0].set_title("Residual-stream RMS before final norm")
    axes[1].set_yscale("log")
    axes[1].set_title("Last block FFN branch RMS (before c_L)")
    axes[2].set_title("Midpoint normalized-feature movement")
    axes[3].set_title("omega_move")
    for a in axes:
        a.set_xlabel("Update")
        a.grid(True, linestyle=":", alpha=0.3)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "p41_depth_probes.png")
    print("ok")
