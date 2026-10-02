"""A2 Problem 4.1 parameterizations for the five-step stress Transformer.

width policies: 'kaiming', 'mup' (reference width N0 = 512)
depth policies (on top of muP, reference width = width = 64, reference depth 2, r = L/2):
  'mup'        residual 1,      block LR x1,        block eps 1e-8
  'depth-mup'  residual r^-1/2, block LR x r^-1/2,  block eps 1e-8 r^-1/2
  'completep'  residual r^-1,   block LR x1,        block eps 1e-8 r^-1
Block multipliers apply to hidden matrices and within-block norms; embedding, readout, final norm unchanged.
"""

import math

HIDDEN = ("q", "k", "v", "o", "gate", "up", "down")
BLOCK_NORMS = ("norm1", "norm2", "qnorm", "knorm")
EPS = 1e-8


def settings(policy, width, depth, n0=None, depth_ref=2):
    n0 = n0 if n0 is not None else (512 if width > 64 else 64)
    m = width / n0
    r = depth / depth_ref
    s = dict(m=m, n0=n0, residual=1.0, block_lr=1.0, block_eps=EPS)
    if policy == "kaiming":
        s.update(readout_var=1 / width, readout_mult=1.0, hidden_lr=1.0)
        return s
    s.update(readout_var=1 / n0, readout_mult=1 / m, hidden_lr=1 / m)
    if policy == "depth-mup":
        s.update(residual=r**-0.5, block_lr=r**-0.5, block_eps=EPS * r**-0.5)
    elif policy == "completep":
        s.update(residual=1 / r, block_lr=1.0, block_eps=EPS / r)
    elif policy != "mup":
        raise ValueError(policy)
    return s


def make(policy, n0=None):
    def initialize(model):
        c = model.config
        s = settings(policy, c.width, c.depth, n0)
        model.embed.weight.normal_(0, 1)
        model.norm.weight.fill_(1.0)
        model.head.weight.normal_(0, math.sqrt(s["readout_var"]))
        model.output_multiplier = s["readout_mult"]
        for b in model.blocks:
            b.residual_multiplier = s["residual"]
            for name in HIDDEN:
                w = getattr(b, name).weight
                w.normal_(0, math.sqrt(1 / w.shape[1]))  # variance 1 / fan-in
            for name in BLOCK_NORMS:
                getattr(b, name).weight.fill_(1.0)

    def parameter_groups(model, base_lr):
        c = model.config
        s = settings(policy, c.width, c.depth, n0)
        hidden, block_norms, other = [], [], []
        for name, p in model.named_parameters():
            leaf = name.split(".")[-2] if name.startswith("blocks.") else None
            if leaf in HIDDEN:
                hidden.append(p)
            elif leaf in BLOCK_NORMS:
                block_norms.append(p)
            else:
                other.append(p)  # embed, head, final norm
        return [dict(params=hidden, lr=base_lr * s["hidden_lr"] * s["block_lr"], eps=s["block_eps"]),
                dict(params=block_norms, lr=base_lr * s["block_lr"], eps=s["block_eps"]),
                dict(params=other, lr=base_lr, eps=EPS)]

    return initialize, parameter_groups
