"""A2 Problem 2(a)/(b): local quadratic L(x, y) in x = log LR, y = log WD per budget; optima, product law, R^2."""

import numpy as np

from experiments.a2.awd.p1_lr_fits import by_budget, optimum
from experiments.a2.provided_sweeps import load


def grid(budget):
    rows = [r for r in load("P2a") if r["tokens"] == budget]
    return (np.array([r["learning_rate"] for r in rows]), np.array([r["weight_decay"] for r in rows]),
            np.array([r["final_val_loss"] for r in rows]))


def fit_quadratic(lrs, wds, losses):
    x, y = np.log(lrs), np.log(wds)
    A = np.stack([np.ones_like(x), x, y, x**2, x * y, y**2], 1)
    coef, *_ = np.linalg.lstsq(A, losses, rcond=None)
    pred = A @ coef
    r2 = 1 - np.sum((losses - pred) ** 2) / np.sum((losses - losses.mean()) ** 2)
    return coef, r2


def quad(coef, lr, wd):
    a, b, c, d, e, f = coef
    x, y = np.log(lr), np.log(wd)
    return a + b * x + c * y + d * x**2 + e * x * y + f * y**2


def stationary(coef):
    """Stationary point of the quadratic; returns (lr*, wd*, loss*, is_minimum)."""
    _, b, c, d, e, f = coef
    H = np.array([[2 * d, e], [e, 2 * f]])
    x, y = np.linalg.solve(H, [-b, -c])
    is_min = bool(np.all(np.linalg.eigvalsh(H) > 0))
    return float(np.exp(x)), float(np.exp(y)), float(quad(coef, np.exp(x), np.exp(y))), is_min, np.linalg.eigvalsh(H)


def power_law(budgets, values):
    lx, ly = np.log(budgets), np.log(values)
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    r2 = 1 - np.sum((ly - pred) ** 2) / np.sum((ly - ly.mean()) ** 2)
    return slope, (lambda d: float(np.exp(intercept) * np.asarray(d, float) ** slope)), r2


BUDGETS = (153_600_000, 307_200_000, 614_400_000, 1_228_800_000)


def joint_optima():
    out = {}
    for d in BUDGETS:
        lrs, wds, losses = grid(d)
        coef, r2 = fit_quadratic(lrs, wds, losses)
        lr, wd, loss, is_min, eig = stationary(coef)
        i = int(np.argmin(losses))
        out[d] = dict(coef=coef, r2=r2, lr=lr, wd=wd, loss=loss, is_min=is_min, eig=eig,
                      best=(lrs[i], wds[i], losses[i]))
    return out


def p1_best():
    data = {**by_budget(load("P1a")), **by_budget(load("P1b"))}
    return {d: (lrs[int(np.argmin(l))], float(l.min()), optimum(lrs, l)) for d, (lrs, l) in data.items()}


if __name__ == "__main__":
    opt = joint_optima()
    p1 = p1_best()
    for d, o in opt.items():
        print(f"{d / 1e6:7.1f}M  quad R2 {o['r2']:.3f}  min? {o['is_min']} eig {np.round(o['eig'], 4)}  "
              f"LR* {o['lr']:.2e}  WD* {o['wd']:.3f}  prod {o['lr'] * o['wd']:.2e}  fitted loss {o['loss']:.4f}  "
              f"best measured {o['best']}  P1 best {p1[d][1]:.4f}  gain {p1[d][1] - o['best'][2]:.4f}")
    b = np.array(BUDGETS, float)
    for name, vals in [("LR*", [opt[d]["lr"] for d in BUDGETS]), ("WD*", [opt[d]["wd"] for d in BUDGETS]),
                       ("LR*WD*", [opt[d]["lr"] * opt[d]["wd"] for d in BUDGETS])]:
        s, f, r2 = power_law(b, vals)
        print(f"{name:7s} ~ D^{s:+.3f}  R2 {r2:.3f}  -> at 2.4576B {f(2.4576e9):.3e}")
    s, f, _ = power_law(b, [opt[d]["lr"] * opt[d]["wd"] for d in BUDGETS])
    print(f"P2(c) predicted WD at LR .003, 2.4576B: {f(2.4576e9) / 0.003:.4f}")
