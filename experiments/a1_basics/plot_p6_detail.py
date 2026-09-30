"""Problem 6(a): RMS of activations, gradients (pre-clip) and parameters for every operation in
every layer of the default d8 run, at the start, middle and end of training."""

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import wandb

from experiments.lr_tuning.plot_lr_tuning import set_style

PLOT_DIR = Path(__file__).resolve().parent / "plots"
CACHE = PLOT_DIR / "p6_detail_cache.json"
RUN = "whitedeer-stanford-university/assignments/dqxhrrl5"  # default d8
POINTS = {"start (step 0)": 0, "middle (step 4600)": 4600, "end (step 9300)": 9300}
LAYERS = range(8)
# Forward order inside one pre-norm block.
OPS = [
    ("input_layernorm", "attn: input RMSNorm"),
    ("self_attn.q_proj", "attn: q_proj"),
    ("self_attn.q_norm", "attn: q_norm"),
    ("self_attn.k_proj", "attn: k_proj"),
    ("self_attn.k_norm", "attn: k_norm"),
    ("self_attn.v_proj", "attn: v_proj"),
    ("self_attn.o_proj", "attn: o_proj (attn output)"),
    ("post_attention_layernorm", "mlp: input RMSNorm"),
    ("mlp.gate_proj", "mlp: gate_proj"),
    ("mlp.up_proj", "mlp: up_proj"),
    ("mlp.down_proj", "mlp: down_proj (MLP output)"),
    ("", "block output (residual stream)"),
]
STATS = ("activation", "gradient", "parameter")


def key(layer, op, stat):
    module = f"model.layers.{layer}" + (f".{op}" if op else "")
    return f"logging/rms/{module}/{stat}"


def fetch():
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    keys = [key(l, op, s) for l in LAYERS for op, _ in OPS for s in STATS]
    run = wandb.Api(timeout=60).run(RUN)
    rows = {int(r["optimizer_step"]): r for r in run.scan_history(keys=["optimizer_step", "logging/rms/global/parameter"])}
    data = {}
    for row in run.scan_history():
        step = row.get("optimizer_step")
        if step is None or int(step) not in rows or "logging/rms/global/parameter" not in row:
            continue
        data[int(step)] = {k: row.get(k) for k in [*keys, "logging/rms/model.embed_tokens/activation", "logging/rms/lm_head/activation",
                                                   "logging/rms/global/gradient", "logging/rms/global/parameter", "logging/rms/global/activation"]}
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(data))
    return data


def grid(data, step, stat):
    return np.array([[np.nan if data[str(step)].get(key(l, op, stat)) is None else data[str(step)][key(l, op, stat)]
                      for l in LAYERS] for op, _ in OPS], dtype=float)


def main():
    set_style()
    data = {str(k): v for k, v in fetch().items()}
    fig, axes = plt.subplots(3, 3, figsize=(17, 16))
    for row, stat in enumerate(STATS):
        grids = [grid(data, step, stat) for step in POINTS.values()]
        finite = np.concatenate([g[np.isfinite(g)] for g in grids])
        norm = mcolors.LogNorm(vmin=finite.min(), vmax=finite.max())
        for col, ((label, _), g) in enumerate(zip(POINTS.items(), grids, strict=True)):
            ax = axes[row, col]
            image = ax.imshow(np.ma.masked_invalid(g), cmap="viridis", norm=norm, aspect="auto")
            for i in range(g.shape[0]):
                for j in range(g.shape[1]):
                    if np.isfinite(g[i, j]):
                        text = f"{g[i, j]:.2g}" if stat != "gradient" else f"{g[i, j]:.1e}".replace("e-0", "e-")
                        ax.text(j, i, text, ha="center", va="center", fontsize=6.3,
                                color="white" if norm(g[i, j]) < 0.55 else "black")
            ax.set_xticks(range(8))
            ax.set_xticklabels([f"L{l + 1}" for l in LAYERS], fontsize=8)
            ax.set_yticks(range(len(OPS)))
            ax.set_yticklabels([name for _, name in OPS] if col == 0 else [], fontsize=8.5)
            ax.set_title(f"{stat} RMS - {label}", fontsize=11)
        fig.colorbar(image, ax=axes[row, :], fraction=0.015, pad=0.01).set_label(f"{stat} RMS (log)")
    fig.suptitle("Default d8: RMS per operation and layer at start / middle / end of training\n"
                 "(gradients are pre-clip; blank = no parameters (norm-free output) or not logged)", fontsize=13)
    out = PLOT_DIR / "p6_detail_heatmaps.png"
    fig.savefig(out, bbox_inches="tight")

    # Evolution through training for a few key quantities, first vs last layer.
    steps = sorted(int(s) for s in data)
    fig2, axes2 = plt.subplots(1, 3, figsize=(17, 4.6))
    series = {
        "activation": [("", "residual (block output)"), ("self_attn.o_proj", "attention output"), ("mlp.down_proj", "MLP output")],
        "gradient": [("self_attn.q_proj", "q_proj"), ("self_attn.o_proj", "o_proj"), ("mlp.down_proj", "down_proj")],
        "parameter": [("self_attn.q_proj", "q_proj"), ("self_attn.o_proj", "o_proj"), ("mlp.down_proj", "down_proj")],
    }
    colors = ["#7030A0", "#2a9d8f", "#e76f51"]
    for ax, stat in zip(axes2, STATS, strict=True):
        for color, (op, name) in zip(colors, series[stat], strict=True):
            for layer, style in ((0, "--"), (7, "-")):
                ys = [data[str(s)].get(key(layer, op, stat)) for s in steps]
                ax.plot(np.array(steps) + 1, [np.nan if y is None else y for y in ys], style, color=color, linewidth=1.5,
                        label=f"{name}, layer {layer + 1}")
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel("Optimizer step")
        ax.set_title(f"{stat} RMS over training (dashed = layer 1, solid = layer 8)", fontsize=10.5)
        ax.grid(True, linestyle=":", alpha=0.3)
        ax.legend(frameon=False, fontsize=7.5)
    fig2.tight_layout()
    out2 = PLOT_DIR / "p6_detail_over_training.png"
    fig2.savefig(out2)

    for label, step in POINTS.items():
        d = data[str(step)]
        print(label, "| global act/grad/param:", [f"{d[k]:.3g}" for k in ("logging/rms/global/activation", "logging/rms/global/gradient", "logging/rms/global/parameter")])
        for stat in STATS:
            g = grid(data, step, stat)
            print(f"   {stat:<10} residual L1..L8:", [f"{v:.3g}" for v in g[-1]] if stat == "activation" else "",
                  "| attn-out L1/L8:", f"{g[6, 0]:.3g}/{g[6, 7]:.3g}", "| mlp-out L1/L8:", f"{g[10, 0]:.3g}/{g[10, 7]:.3g}")
    print(out, out2)


if __name__ == "__main__":
    main()
