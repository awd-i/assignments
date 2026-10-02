"""A2 Problems 2(c) and 3.2 target runs (configs only; launched by queue_driver.py at <= 2 GPUs)."""

from experiments.a2.modal_launcher import config

D_P2C = 2_457_600_000
D_P32 = 614_400_000


def _cfg(key, **kw):
    return config(run_name_suffix=key, wandb_tags=(key,), diagnostics=False, **kw)


def p2c():
    """Product law fit on 0.15-1.23B -> LR*WD* = 2.78e-4 at 2.4576B -> WD .0927 at LR .003 (p2_fits.py).
    Baselines: (i) best 153.6M pair (LR .0015, WD 1.6); (ii) LR .003, WD {.05, .1, .2} (WD .1 reused from P1b)."""
    key = "a2-p2c-D2458m"
    pairs = [(0.003, 0.0927), (0.0015, 1.6), (0.003, 0.05), (0.003, 0.2)]
    return [_cfg(key, tokens=D_P2C, learning_rate=lr, weight_decay=wd) for lr, wd in pairs]


P32A_LRS = {8: (0.00075, 0.0015, 0.003), 16: (0.00075, 0.0015, 0.003), 32: (0.0015, 0.003, 0.006)}
P32B_WDS = {8: (0.025, 0.05), 16: (0.05, 0.2), 32: (0.2, 0.4), 64: (0.8,)}  # LR .0015; WD .1 from (a) / P2a


def p32a():
    """WD .1, LR sweep per batch at 614.4M tokens (B=64 reused from P1a)."""
    return [_cfg(f"a2-p32-B{b}", tokens=D_P32, batch=b, learning_rate=lr, weight_decay=0.1)
            for b, lrs in P32A_LRS.items() for lr in lrs]


def p32b_source():
    """Hypothesis ii source: LR .0015, WD sweep per batch."""
    return [_cfg(f"a2-p32-B{b}", tokens=D_P32, batch=b, learning_rate=0.0015, weight_decay=wd)
            for b, wds in P32B_WDS.items() for wd in wds]


# ---- Problem 4.2: width/depth transfer at 153.6M tokens (diagnostics on) ----
from model_config import LMConfig  # noqa: E402

D_P42 = 153_600_000
P42A_LRS = {"baseline": (0.0015, 0.003, 0.006, 0.012, 0.024), "mup": (0.00075, 0.0015, 0.003, 0.006, 0.012)}


def p42(prescription, width, depth, lr, depth_rule=None, key="a2-p42"):
    rule = depth_rule or ("mup" if prescription == "mup" else "sp")
    name = f"a2-w{width}-d{depth}-{rule}"
    kwargs = dict(prescription=prescription, depth_rule=depth_rule)
    return config(tokens=D_P42, learning_rate=lr, run_name_suffix=key, wandb_tags=(key, name),
                  model_config=LMConfig(name, 4096, 1024, width, int(3.5 * width), depth, width // 64, width // 64),
                  model_builder="experiments.a2.awd.mup:build_model", model_builder_kwargs=kwargs,
                  optimizer_builder="experiments.a2.awd.mup:build_optimizer")


def p42a():
    return [p42(p, w, 8, lr) for w in (128, 256) for p, lrs in P42A_LRS.items() for lr in lrs]


def p42d():
    """Depths 4 and 16 at width 512 (m = 1): standard muP (= course baseline here), Depth-muP, CompleteP."""
    return [p42("mup", 512, d, lr, rule) for d in (4, 16) for rule in (None, "depth-mup", "completep")
            for lr in (0.0015, 0.003, 0.006)]


def p5():
    """Optional P5: product-matched WD at too-high LRs, 153.6M tokens (prediction in p5_preregistered_prediction.md)."""
    key = "a2-p5-wdrescue"
    pairs = [(0.006, 0.262), (0.012, 0.131), (0.012, 0.1)]
    return [config(tokens=153_600_000, learning_rate=lr, weight_decay=wd, run_name_suffix=key, wandb_tags=(key,))
            for lr, wd in pairs]


def p42c():
    """Held-out width 1024 (prediction in p42c_preregistered_predictions.md)."""
    sweep = (0.00075, 0.0015, 0.003, 0.006)
    return ([p42("baseline", 1024, 8, lr) for lr in sweep + (0.00126,)] +
            [p42("mup", 1024, 8, lr) for lr in sweep + (0.00194,)])


def p32b_hyp_i():
    """Hypothesis i targets: WD .1, LR {.003, power-law prediction, 2x prediction} at B=128, 256."""
    grid = {128: (0.003, 0.00559, 0.0112), 256: (0.003, 0.00828, 0.0166)}
    return [_cfg(f"a2-p32-B{b}", tokens=D_P32, batch=b, learning_rate=lr, weight_decay=0.1)
            for b, lrs in grid.items() for lr in lrs]


def p32b_hyp_ii():
    """Hypothesis ii targets: LR .0015, WD {pred/2, pred, 2 pred} at B=128 (pred .946) and 256 (pred 1.86)."""
    grid = {128: (0.473, 0.946, 1.89), 256: (0.93, 1.86, 3.72)}
    return [_cfg(f"a2-p32-B{b}", tokens=D_P32, batch=b, learning_rate=0.0015, weight_decay=wd)
            for b, wds in grid.items() for wd in wds]


P32C_BETA1 = (0.0, 0.5, 0.8, 0.95, 0.98)  # beta1 = .9 is the existing best run


def p32c_b8():
    """Momentum ablation at B=8 with its best measured pair (LR .0015, WD .05; loss 2.9358), beta2 .95."""
    return [_cfg("a2-p32c-B8", tokens=D_P32, batch=8, learning_rate=0.0015, weight_decay=0.05, beta1=b1)
            for b1 in P32C_BETA1]
