"""A2 Problem 2(c): test the product law's WD rescue at 2.4576B tokens (LR .003)."""

from experiments.a2.awd.target_results import load
from experiments.a2.provided_sweeps import load as load_provided

if __name__ == "__main__":
    rows = {(r["lr"], r["wd"]): r["val_loss"] for r in load("a2-p2c")}
    for r in load_provided("P1b"):
        if r["tokens"] == 2_457_600_000 and r["learning_rate"] == 0.003:
            rows[(0.003, 0.1)] = r["final_val_loss"]  # reuse supplied WD .1 run
    for k, v in sorted(rows.items()):
        print(k, round(v, 4))
    pred = rows.get((0.003, 0.0927))
    base_i = rows.get((0.0015, 1.6))
    base_ii = min(v for (lr, wd), v in rows.items() if lr == 0.003 and wd in (0.05, 0.1, 0.2))
    print(f"product-rule {pred}, baseline i {base_i}, baseline ii best {base_ii}")
    if pred is not None:
        print(f"gap to (i) {pred - base_i:+.4f}, gap to (ii) {pred - base_ii:+.4f}")
