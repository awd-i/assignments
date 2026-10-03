"""A2 Problem 4.2 analysis: transfer across width (128-1024) and depth (4-16) at 153.6M tokens, plus diagnostics.

Losses come from W&B (target_results.py cache); diagnostics (features/alignment/gradients jsonl) are downloaded
from the volume into plots/p42/<run>/ by `python -m experiments.a2.awd.p42_analysis fetch <run>...`.
"""

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.a2.awd.p1_lr_fits import optimum
from experiments.a2.awd.target_results import load as load_targets
from experiments.a2.provided_sweeps import load as load_provided, reference_diagnostics
from experiments.lr_tuning.plot_lr_tuning import set_style

HERE = Path(__file__).resolve().parent
OUT = HERE / "plots"
RAW = OUT / "p42"
SOURCE_LR = 0.003  # Problem 1's best sampled LR at 153.6M tokens
set_style()


def runs():
    """{(rule, width, depth): {lr: (loss, run_name)}}; rule in sp, mup, depth-mup, completep."""
    out = defaultdict(dict)
    for r in load_targets("a2-p42"):
        name = r["model"]  # a2-w{w}-d{d}-{rule}
        _, w, d, rule = name.split("-", 3)
        out[(rule, int(w[1:]), int(d[1:]))][r["lr"]] = (r["val_loss"], r["name"])
    for r in load_provided("P1a"):  # width 512, depth 8: baseline == muP == depth rules (m = r = 1)
        if r["tokens"] == 153_600_000:
            for rule in ("sp", "mup", "depth-mup", "completep"):
                out[(rule, 512, 8)][r["learning_rate"]] = (r["final_val_loss"], None)
    return out


def fit(points):
    lrs = np.array(sorted(points))
    losses = np.array([points[l][0] for l in lrs])
    i = int(np.argmin(losses))
    if i in (0, len(lrs) - 1):
        return float(lrs[i]), float(losses[i]), True
    lo, hi = max(0, i - 1), min(len(lrs), i + 2)
    lr, loss, _ = optimum(lrs[lo:hi], losses[lo:hi])
    return float(lr), float(loss), False


def fetch(names):
    for n in names:
        d = RAW / n
        d.mkdir(parents=True, exist_ok=True)
        for f in ("features.jsonl", "alignment.jsonl", "gradients.jsonl"):
            if not (d / f).exists():
                subprocess.run(["uv", "run", "modal", "volume", "get", "volume-dl_alchemy", f"/ckpts/{n}/{f}", str(d / f)],
                               capture_output=True)


def diag(name):
    d = RAW / name
    read = lambda f: [json.loads(s) for s in (d / f).read_text().splitlines()] if (d / f).exists() else []
    return read("features.jsonl"), read("alignment.jsonl"), read("gradients.jsonl")


def reference():
    ref = reference_diagnostics()
    feats = [r for r in ref["diagnostics"] if r.get("kind", "fixed_batch") == "fixed_batch"]
    ro = {r["step"]: r for r in ref["readout_alignment"]}
    for f in feats:
        f.setdefault("readout_alignment", ro.get(f["step"], {}))
    return feats, ref["alignment"], ref["gradients"]


def alpha_by_type(align, steps=None):
    out = defaultdict(list)
    for a in align:
        if a.get("alpha") is None or (steps is not None and a["step"] not in steps):
            continue
        p = a["parameter"]
        key = "head" if "lm_head" in p else p.split(".")[-2].replace("_proj", "")
        out[key].append(a["alpha"])
    return {k: float(np.mean(v)) for k, v in out.items()}


def alpha_over_time(align):
    by = defaultdict(list)
    for a in align:
        if a.get("alpha") is not None:
            by[a["step"]].append(a["alpha"])
    s = sorted(by)
    return s, [float(np.mean(by[x])) for x in s]


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fetch":
        fetch(sys.argv[2:])
        sys.exit()
    R = runs()
    summary = {}
    for k in sorted(R):
        lr, loss, edge = fit(R[k])
        summary[k] = dict(lr=lr, loss=loss, edge=edge, at_source=R[k].get(SOURCE_LR, (None,))[0],
                          best=min(R[k].items(), key=lambda kv: kv[1][0]))
        print(k, f"LR* {lr:.3g} loss* {loss:.4f}{' EDGE' if edge else ''}  at .003: {summary[k]['at_source']}",
              {f"{l:.3g}": round(v[0], 4) for l, v in sorted(R[k].items())})
    json.dump({"-".join(map(str, k)): {kk: vv for kk, vv in v.items() if kk != "best"} for k, v in summary.items()},
              open(OUT / "p42_summary.json", "w"), indent=1)


def run_name(rule, width, depth, lr):
    prescription_rule = rule if rule != "sp" else "sp"
    return f"model-a2-w{width}-d{depth}-{prescription_rule}-lr{lr}-tok154M-a2-a2-p42"


def diag_for(rule, width, depth, lr):
    """Diagnostics for a configuration; width-512 depth-8 at LR .003 uses the supplied reference run."""
    if width == 512 and depth == 8:
        return reference() if lr == 0.003 else ([], [], [])
    return diag(run_name(rule, width, depth, lr))


def plots(R, summary):
    widths = sorted({w for (_, w, d) in R if d == 8})
    # loss-LR curves and transfer, width
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
    for rule, ax, col in (("sp", axes[0], "#e76f51"), ("mup", axes[1], "#2a9d8f")):
        for w, c in zip(widths, plt.cm.viridis(np.linspace(0.1, 0.85, len(widths)))):
            if (rule, w, 8) not in R:
                continue
            pts = sorted(R[(rule, w, 8)].items())
            ax.plot([p[0] for p in pts], [p[1][0] for p in pts], "o-", color=c, markeredgecolor="black", label=f"width {w}")
            s = summary[(rule, w, 8)]
            ax.scatter([s["lr"]], [s["loss"]], marker="*", s=160, color=c, edgecolor="black", zorder=5)
        ax.axvline(SOURCE_LR, color="gray", linestyle=":")
        ax.set_xscale("log")
        ax.set_xlabel("Peak base LR")
        ax.set_ylabel("Final val loss (153.6M tokens)")
        ax.set_title("Course baseline (SP)" if rule == "sp" else "muP")
        ax.grid(True, linestyle=":", alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    ax = axes[2]
    for rule, col in (("sp", "#e76f51"), ("mup", "#2a9d8f")):
        ws = [w for w in widths if (rule, w, 8) in summary]
        ax.plot(ws, [summary[(rule, w, 8)]["lr"] for w in ws], "o-", color=col, markeredgecolor="black",
                label=f"{rule}: fitted LR*")
    ax.axhline(SOURCE_LR, color="gray", linestyle=":", label="transferred source LR .003")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Width")
    ax.set_ylabel("Fitted optimal base LR")
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "p42_width_curves.png")

    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for rule, col in (("sp", "#e76f51"), ("mup", "#2a9d8f")):
        ws = [w for w in widths if (rule, w, 8) in summary]
        ax.plot(ws, [summary[(rule, w, 8)]["loss"] for w in ws], "o-", color=col, markeredgecolor="black",
                label=f"{rule}: fitted min loss")
        ws2 = [w for w in ws if summary[(rule, w, 8)]["at_source"] is not None]
        ax.plot(ws2, [summary[(rule, w, 8)]["at_source"] for w in ws2], "s--", color=col, markeredgecolor="black",
                alpha=0.7, label=f"{rule}: loss at transferred LR .003")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Width")
    ax.set_ylabel("Val loss (153.6M tokens)")
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "p42_width_losses.png")

    # alignment over training at the source LR
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
    for rule, ls in (("sp", "--"), ("mup", "-")):
        for w, c in zip(widths, plt.cm.viridis(np.linspace(0.1, 0.85, len(widths)))):
            feats, align, grads = diag_for(rule, w, 8, SOURCE_LR)
            if not align:
                continue
            st, al = alpha_over_time(align)
            axes[0].plot(st, al, ls, color=c, marker="o", markersize=3, label=f"{rule} w{w}")
            om = [(f["step"], f.get("readout_alignment", {}).get("movement", {}).get("omega")) for f in feats]
            om = [(s, o) for s, o in om if o is not None]
            axes[1].plot([s for s, _ in om], [o for _, o in om], ls, color=c, marker="o", markersize=3, label=f"{rule} w{w}")
            mv = [(f["step"], f["features"]["model.norm"]["movement"]) for f in feats]
            axes[2].plot([s for s, _ in mv], [m for _, m in mv], ls, color=c, marker="o", markersize=3, label=f"{rule} w{w}")
    for a, t in zip(axes, ("Mean update alignment alpha (all matrices)", "omega_move (initial readout vs h_t - h_0)",
                           "Final-norm feature movement")):
        a.set_xscale("symlog", linthresh=10)
        a.set_xlabel("Optimizer update")
        a.set_title(t)
        a.grid(True, linestyle=":", alpha=0.3)
    for y in (0.5, 1.0):
        axes[0].axhline(y, color="gray", linestyle=":")
        axes[1].axhline(y, color="gray", linestyle=":")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "p42_alignment.png")


if __name__ == "__main__" and len(sys.argv) == 1:
    plots(R, summary)
    for rule in ("sp", "mup"):
        for w in sorted({w for (r, w, d) in R if r == rule and d == 8}):
            feats, align, grads = diag_for(rule, w, 8, SOURCE_LR)
            if not grads:
                continue
            g = {x["step"]: x for x in grads}
            first, last = g[min(g)], g[max(g)]
            by_type = alpha_by_type(align, steps=set(range(1, 6)))
            late = alpha_by_type(align, steps={s for s in {a["step"] for a in align} if s > 1000})
            f0, fT = feats[0], feats[-1]
            print(f"{rule} w{w}: grad norm {first['pre_clip_norm']:.2f} -> {last['pre_clip_norm']:.3f}, embed frac "
                  f"{first['embedding_fraction_squared_norm']:.3f} -> {last['embedding_fraction_squared_norm']:.3f}, clip "
                  f"{first['clip_coefficient']:.3f} -> {last['clip_coefficient']:.3f}; logit rms {f0['logit_rms']:.3f} -> "
                  f"{fT['logit_rms']:.2f}; alpha(1-5) {np.mean(list(by_type.values())):.2f} late "
                  f"{np.mean(list(late.values())) if late else float('nan'):.2f}; emb-norm rms "
                  f"{f0['features']['model.layers.0.input_layernorm']['rms']:.3f}")


def depth_and_regime_plots(R, summary):
    rules = (("mup", "#888888", "standard muP"), ("depth-mup", "#3a7bd5", "Depth-muP"), ("completep", "#7030A0", "CompleteP"))
    depths = (4, 8, 16)
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.4))
    for ax, d in zip(axes[:3], depths):
        for rule, col, lab in rules:
            if (rule, 512, d) not in R:
                continue
            pts = sorted(R[(rule, 512, d)].items())
            ax.plot([p[0] for p in pts], [p[1][0] for p in pts], "o-", color=col, markeredgecolor="black", label=lab)
        ax.axvline(SOURCE_LR, color="gray", linestyle=":")
        ax.set_xscale("log")
        ax.set_title(f"Depth {d} (width 512)")
        ax.set_xlabel("Peak base LR")
        ax.grid(True, linestyle=":", alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Final val loss (153.6M tokens)")
    ax = axes[3]
    for rule, col, lab in rules:
        ds = [d for d in depths if (rule, 512, d) in summary]
        ax.plot(ds, [summary[(rule, 512, d)]["loss"] for d in ds], "o-", color=col, markeredgecolor="black",
                label=f"{lab}: fitted min")
        ax.plot(ds, [summary[(rule, 512, d)]["at_source"] for d in ds], "s--", color=col, alpha=0.6,
                label=f"{lab}: at LR .003")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Depth")
    ax.set_ylabel("Val loss")
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "p42_depth.png")

    # (e) regime diagnostics: early (updates 0-5) and end-of-training probes at the transferred and best sampled LR
    configs = [("sp", w, 8) for w in (128, 256, 512, 1024)] + [("mup", w, 8) for w in (128, 256, 1024)] + \
              [(r, 512, d) for r, _, _ in rules for d in (4, 16)]
    rows = []
    for k in configs:
        if k not in summary:
            continue
        best_lr = summary[k]["best"][0]
        for tag, lr in (("transfer", SOURCE_LR), ("best", best_lr)):
            feats, align, grads = diag_for(*k, lr)
            if not feats:
                continue
            by = {f["step"]: f for f in feats}
            early = [by[s] for s in sorted(by) if s <= 5]
            last = by[max(by)]
            om = [f.get("readout_alignment", {}).get("movement", {}).get("omega") for f in early[1:]]
            om_end = last.get("readout_alignment", {}).get("movement", {}).get("omega")
            al_early = alpha_by_type(align, steps=set(range(1, 6)))
            al_late = alpha_by_type(align, steps={a["step"] for a in align if a["step"] > 1000})
            rows.append(dict(cfg=k, tag=tag, lr=lr, logit_peak5=max(f["logit_rms"] for f in early),
                             logit_end=last["logit_rms"],
                             move5=early[-1]["features"]["model.norm"]["movement"],
                             move_end=last["features"]["model.norm"]["movement"],
                             omega5=np.nanmean([o for o in om if o is not None]) if any(o is not None for o in om) else None,
                             omega_end=om_end, alpha5=np.mean(list(al_early.values())) if al_early else None,
                             alpha_end=np.mean(list(al_late.values())) if al_late else None,
                             loss=R[k].get(lr, (None,))[0]))
    json.dump(rows, open(OUT / "p42_regime.json", "w"), indent=1, default=float)
    for r in rows:
        print(r["cfg"], r["tag"], f"lr {r['lr']:.3g} loss {r['loss']}", " ".join(
            f"{k} {r[k]:.3f}" for k in ("logit_peak5", "logit_end", "move5", "move_end", "omega5", "omega_end",
                                       "alpha5", "alpha_end") if r[k] is not None))


if __name__ == "__main__" and len(sys.argv) == 1:
    depth_and_regime_plots(R, summary)
