"""Problem 6(a) as line plots: RMS vs layer for each operation, at start / middle / end of the default d8 run."""

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt

from experiments.a1_basics.plot_p6_detail import CACHE, LAYERS, OPS, POINTS, STATS, key
from experiments.lr_tuning.plot_lr_tuning import set_style

PLOT_DIR = Path(__file__).resolve().parent / "plots"
GROUPS = {
    "Attention": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"],
    "MLP": ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
    "Norms + residual": ["input_layernorm", "post_attention_layernorm", ""],
}
COLORS = {"self_attn.q_proj": "#7030A0", "self_attn.k_proj": "#b07cc6", "self_attn.v_proj": "#3a7bd5", "self_attn.o_proj": "#2a9d8f",
          "mlp.gate_proj": "#e9c46a", "mlp.up_proj": "#f4a261", "mlp.down_proj": "#e76f51",
          "input_layernorm": "#8d99ae", "post_attention_layernorm": "#264653", "": "black"}
NAMES = {**dict(OPS), "input_layernorm": "attn input norm", "post_attention_layernorm": "MLP input norm", "": "residual stream"}
TIME_STYLE = {"start (step 0)": ":", "middle (step 4600)": "--", "end (step 9300)": "-"}

set_style()
data = json.loads(CACHE.read_text())
fig, axes = plt.subplots(3, 3, figsize=(17, 13))
layers = [l + 1 for l in LAYERS]
for row, stat in enumerate(STATS):
    for col, (group, ops) in enumerate(GROUPS.items()):
        ax = axes[row, col]
        for op in ops:
            for label, step in POINTS.items():
                ys = [data[str(step)].get(key(l, op, stat)) for l in LAYERS]
                if all(y is None for y in ys):
                    continue
                ax.plot(layers, ys, TIME_STYLE[label], marker="o", markersize=3.5, color=COLORS[op], linewidth=1.6,
                        label=f"{NAMES[op].split(': ')[-1]} - {label.split(' (')[0]}")
        ax.set_yscale("log")
        ax.set_xticks(layers)
        ax.set_xlabel("Layer")
        ax.set_ylabel(f"{stat} RMS (log)")
        ax.set_title(f"{group}: {stat}", fontsize=11.5)
        ax.grid(True, linestyle=":", alpha=0.3)
        ax.legend(frameon=False, fontsize=6.6, ncol=2)
fig.suptitle("Default d8 across depth: dotted = start (step 0), dashed = middle (4600), solid = end (9300)", fontsize=13)
fig.tight_layout()
out = PLOT_DIR / "p6_lines_by_layer.png"
fig.savefig(out)
print(out)
