import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from experiments.a2.awd.target_results import load
from experiments.lr_tuning.plot_lr_tuning import set_style
set_style()
ref = {8: 2.9358, 256: 2.968}  # beta1 = .9 runs at the same pairs
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
for B, col in ((8, "#3a7bd5"), (256, "#e76f51")):
    pts = sorted([(r["beta1"], r["val_loss"]) for r in load("a2-p32c") if r["batch"] == B] + [(0.9, ref[B])])
    a1.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=col, markeredgecolor="black", label=f"B={B}")
    a2.plot([p[0] for p in pts], [p[1] - ref[B] for p in pts], "o-", color=col, markeredgecolor="black", label=f"B={B}")
a1.set_ylabel("Final val loss (614.4M tokens)")
a2.set_ylabel("Loss - loss at beta1 = .9")
a2.axhline(0, color="gray", linestyle=":")
for a in (a1, a2):
    a.set_xlabel("beta1 (beta2 = .95, best LR-WD pair held fixed)")
    a.grid(True, linestyle=":", alpha=0.3)
    a.legend(frameon=False)
a1.set_title("(c) Momentum ablation")
a2.set_title("Cost of changing beta1")
fig.tight_layout()
fig.savefig("experiments/a2/awd/plots/p32c_momentum.png")
