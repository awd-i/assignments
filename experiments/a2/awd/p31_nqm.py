"""A2 Problem 3.1: noisy quadratic model (NQM). Vectorized over LRs x samples; common random numbers across LRs.

f(w) = 1/2 w^T H w, g = H w + noise, noise ~ N(0, sigma^2 I / B). Fixed N = 8192 examples -> N/B updates.
"""

import json
from pathlib import Path

import numpy as np

N = 8192
SAMPLES = 2048
OUT = Path(__file__).resolve().parent / "plots"


def simulate(opt, B, lrs, *, h=(1.0, 10.0), init_sd=(1.0, 0.1 ** 0.5), sigma=1.0, beta1=0.9, beta2=0.95,
             mu=0.0, eps=1e-8, samples=SAMPLES, seed=0, curve=False):
    """Returns mean final loss per LR (and the mean loss after every update if curve=True)."""
    rng = np.random.default_rng(seed)
    h = np.asarray(h)
    lrs = np.asarray(lrs, float)[:, None, None]
    w0 = rng.normal(size=(samples, len(h))) * np.asarray(init_sd)
    w = np.broadcast_to(w0, (lrs.shape[0],) + w0.shape).copy()
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    steps = N // B
    curves = []
    for t in range(1, steps + 1):
        g = h * w + rng.normal(size=w0.shape) * (sigma / np.sqrt(B))  # same noise for every LR
        if opt == "sgd":
            m = mu * m + g
            w = w - lrs * m
        else:
            b1 = 0.0 if opt == "rmsprop" else beta1
            m = b1 * m + (1 - b1) * g
            v = beta2 * v + (1 - beta2) * g * g
            mh = m / (1 - b1**t) if b1 > 0 else m
            vh = v / (1 - beta2**t)
            w = w - lrs * mh / (np.sqrt(vh) + eps)
        if curve:
            curves.append(0.5 * np.mean(np.sum(h * w * w, -1), -1))
    loss = 0.5 * np.mean(np.sum(h * w * w, -1), -1)
    loss = np.where(np.isfinite(loss), loss, np.inf)
    return (loss, np.array(curves).T) if curve else loss


def best_lr(lrs, losses):
    """Parabola in log2(LR) through the best grid point and its two neighbours (log loss)."""
    i = int(np.argmin(losses))
    if i in (0, len(lrs) - 1):
        return float(lrs[i]), float(losses[i]), True  # optimum at grid edge
    x = np.log2(lrs[i - 1:i + 2])
    y = np.log(losses[i - 1:i + 2])
    a, b, c = np.polyfit(x, y, 2)
    xs = -b / (2 * a)
    return float(2**xs), float(np.exp(a * xs**2 + b * xs + c)), False


def lr_grid(opt, lo=None, hi=None, n=41):
    lo = lo or {"sgd": 1e-4, "rmsprop": 1e-5, "adam": 1e-5}[opt]
    hi = hi or {"sgd": 0.25, "rmsprop": 1.0, "adam": 1.0}[opt]
    return np.logspace(np.log10(lo), np.log10(hi), n)


def sweep(opt, batches, **kw):
    out = {}
    for B in batches:
        if "lrs" in kw:
            lrs = kw["lrs"]
        elif opt == "sgd":  # optimum ~ B / sigma^2 at small B; stability limit 2 / max curvature
            sigma, hmax = kw.get("sigma", 1.0), max(kw.get("h", (1.0, 10.0)))
            lrs = lr_grid(opt, 1e-4 / max(sigma, 1) ** 2, 2.4 / hmax, n=61)
        else:
            lrs = lr_grid(opt)
        losses = simulate(opt, B, lrs, **{k: v for k, v in kw.items() if k != "lrs"})
        lr, loss, edge = best_lr(lrs, losses)
        out[B] = dict(lr=lr, loss=loss, edge=edge, lrs=lrs.tolist(), losses=losses.tolist())
    return out


def fit_power(batches, lrs):
    p, logc = np.polyfit(np.log(batches), np.log(lrs), 1)
    return float(np.exp(logc)), float(p)


SOURCE = (1, 2, 4, 8, 16, 32, 64)
TARGETS = (128, 256, 512)


def transfer(opt, **kw):
    res = sweep(opt, SOURCE + TARGETS, **kw)
    c, p = fit_power(SOURCE, [res[B]["lr"] for B in SOURCE])
    pred = c * 256**p
    loss_pred = float(simulate(opt, 256, [pred], **kw)[0])
    return dict(res=res, c=c, p=p, pred256=pred, loss_pred256=loss_pred, loss_tuned256=res[256]["loss"],
                ratio256=loss_pred / res[256]["loss"])


if __name__ == "__main__":
    import sys
    part = sys.argv[1]
    results = {}
    if part == "ab":
        for opt in ("sgd", "rmsprop", "adam"):
            results[opt] = transfer(opt)
            r = results[opt]
            print(f"{opt:8s} p={r['p']:.3f} c={r['c']:.3g}  pred LR@256={r['pred256']:.3g} "
                  f"tuned={r['res'][256]['lr']:.3g}  loss pred/tuned={r['loss_pred256']:.4g}/{r['loss_tuned256']:.4g} "
                  f"(x{r['ratio256']:.2f})")
            for B, x in r["res"].items():
                print(f"   B={B:4d} LR*={x['lr']:.3g} loss={x['loss']:.4g}{' EDGE' if x['edge'] else ''}")
    elif part == "c":
        for curv in ("2d", "scalar"):
            kw = {} if curv == "2d" else dict(h=(1.0,), init_sd=(2 ** 0.5,))
            for sigma in (1, 10, 100, 300):
                for opt in ("sgd", "rmsprop", "adam"):
                    r = transfer(opt, sigma=sigma, **kw)
                    results[f"{curv}-{opt}-{sigma}"] = r
                    print(f"{curv:6s} sigma={sigma:3d} {opt:8s} p={r['p']:+.3f} pred/tuned LR@256 "
                          f"{r['pred256']:.3g}/{r['res'][256]['lr']:.3g} loss ratio x{r['ratio256']:.3f}"
                          f"{' EDGE' if any(x['edge'] for x in r['res'].values()) else ''}", flush=True)
    elif part == "d":
        for B in (16, 256):
            for mu in (0.0, 0.3, 0.5, 0.7, 0.9):
                res = sweep("sgd", [B], mu=mu, lrs=lr_grid("sgd", 1e-5, 0.4))
                results[f"sgd-B{B}-mu{mu}"] = res[B]
                print(f"B={B:3d} SGD mu={mu}: LR*={res[B]['lr']:.3g} loss={res[B]['loss']:.4g}{' EDGE' if res[B]['edge'] else ''}")
            for b1 in (0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995):
                res = sweep("adam", [B], beta1=b1)
                results[f"adam-B{B}-b1{b1}"] = res[B]
                print(f"B={B:3d} Adam beta1={b1}: LR*={res[B]['lr']:.3g} loss={res[B]['loss']:.4g}{' EDGE' if res[B]['edge'] else ''}", flush=True)
    (OUT / f"p31_{part}.json").write_text(json.dumps(results))
