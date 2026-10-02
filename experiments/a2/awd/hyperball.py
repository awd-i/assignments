"""Hyperball ("adamh"): Adam direction, with every linear-layer weight matrix kept at its initial Frobenius norm.

For a linear weight W with Adam direction U (bias-corrected m / (sqrt(v) + eps)):
    W~ = W - lr * (||W||_F / ||U||_F) * U
    W  <- (||W||_F / ||W~||_F) * W~
Embeddings, normalization gains and biases use ordinary Adam at lr * ADAM_LR_RATIO. No weight decay anywhere.
Use via TrainConfig(optimizer_builder="experiments.a2.awd.hyperball:build", weight_decay=0.0).
"""

import torch

ADAM_LR_RATIO = 0.000656 / 0.00630  # supplied P1(d) runs: ordinary-Adam LR = 0.1041 x listed Hyperball LR


class Hyperball(torch.optim.Optimizer):
    def __init__(self, params, lr, betas=(0.9, 0.95), eps=1e-8):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, hyperball=False))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]
                state["m"].mul_(beta1).add_(p.grad, alpha=1 - beta1)
                state["v"].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
                m_hat = state["m"] / (1 - beta1**t)
                v_hat = state["v"] / (1 - beta2**t)
                update = m_hat / (v_hat.sqrt() + group["eps"])
                if not group["hyperball"]:
                    p.add_(update, alpha=-group["lr"])
                    continue
                w_norm = p.norm()
                u_norm = update.norm()
                if w_norm == 0 or u_norm == 0:
                    continue
                p.add_(update, alpha=-(group["lr"] * w_norm / u_norm).item())
                p.mul_(w_norm / p.norm())
        return loss


def build(model, optimizer_name=None, learning_rate=None, weight_decay=0.0, beta1=0.9, beta2=0.95, **_):
    if weight_decay:
        raise ValueError("Hyperball controls weight norms itself; set weight_decay=0.0")
    linear_weights, other = [], []
    linear_ids = {id(m.weight) for m in model.modules() if isinstance(m, torch.nn.Linear)}
    for p in model.parameters():
        if p.requires_grad:
            (linear_weights if id(p) in linear_ids else other).append(p)
    return Hyperball(
        [dict(params=linear_weights, lr=learning_rate, hyperball=True),
         dict(params=other, lr=learning_rate * ADAM_LR_RATIO, hyperball=False)],
        lr=learning_rate, betas=(beta1, beta2),
    )
