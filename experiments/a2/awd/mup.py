"""A2 Problem 4.2: muP (width) and depth prescriptions for the course model.

Model builder `experiments.a2.awd.mup:build_model` with kwargs
  prescription: 'baseline' | 'mup'        (width rules; reference width n0 = 512)
  depth_rule:   None | 'depth-mup' | 'completep'   (on top of muP; reference depth 8, r = L / 8)
Optimizer builder `experiments.a2.awd.mup:build_optimizer` reads the model's settings:
  hidden matrices LR eta * hidden_lr * block_lr (eps 1e-8 * block_eps), within-block norms LR eta * block_lr,
  embedding / readout / final norm LR eta. Weight decay on linear weights only (course masking).
muP changes exactly four settings vs the course baseline: embedding init G/n0 (vs G/n), readout init G/sqrt(n0)
(vs G/sqrt(n)), readout forward multiplier 1/m, hidden-matrix LR eta/m, with m = n / n0 and G shared.
"""

import torch
from torch.nn import functional as F

from modeling import AutoregressiveLM, LlamaDecoderLayer, initialize_model
from optimizers import ADAMW_EPSILON, should_apply_weight_decay


def settings(width, depth, prescription="baseline", depth_rule=None, n0=512, depth_ref=8):
    m, r = width / n0, depth / depth_ref
    s = dict(m=m, r=r, output_multiplier=1.0, hidden_lr=1.0, block_lr=1.0, block_eps=1.0, residual=1.0)
    if prescription == "mup":
        s.update(output_multiplier=1 / m, hidden_lr=1 / m)
    elif prescription != "baseline":
        raise ValueError(prescription)
    if depth_rule == "depth-mup":
        s.update(residual=r**-0.5, block_lr=r**-0.5, block_eps=r**-0.5)
    elif depth_rule == "completep":
        s.update(residual=1 / r, block_eps=1 / r)
    elif depth_rule is not None:
        raise ValueError(depth_rule)
    return s


class ScaledDecoderLayer(LlamaDecoderLayer):
    residual_multiplier = 1.0

    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        h = self.self_attn(self.input_layernorm(hidden_states), position_embeddings=position_embeddings,
                           attention_mask=attention_mask)
        hidden_states = hidden_states + self.residual_multiplier * F.dropout(h, p=self.dropout, training=self.training)
        h = self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states + self.residual_multiplier * F.dropout(h, p=self.dropout, training=self.training)


class ScaledLM(AutoregressiveLM):
    def __init__(self, config, dtype=torch.float32, qk_norm=True, tie_word_embeddings=False, dropout=0.0,
                 prescription="baseline", depth_rule=None, n0=512, depth_ref=8):
        super().__init__(config, dtype=dtype, qk_norm=qk_norm, tie_word_embeddings=tie_word_embeddings,
                         dropout=dropout)
        if tie_word_embeddings:
            raise ValueError("muP readout scaling needs untied embeddings")
        self.n0 = n0
        self.prescription = prescription
        self.scaling = settings(config.hidden_size, config.num_hidden_layers, prescription, depth_rule, n0, depth_ref)
        self.output_multiplier = self.scaling["output_multiplier"]
        for layer in self.model.layers:
            layer.__class__ = ScaledDecoderLayer
            layer.residual_multiplier = self.scaling["residual"]

    def forward(self, input_ids=None, attention_mask=None, position_ids=None):
        h = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        return self.lm_head(h * self.output_multiplier)

    def initialize_parameters(self):
        initialize_model(self)  # course init: hidden 1/fan-in, embed G/n, readout G/sqrt(n) (same G)
        with torch.no_grad():
            n = self.config.hidden_size
            g = self.model.embed_tokens.weight.float() * n  # recover the shared G
            scale = self.n0 if self.prescription == "mup" else n
            self.model.embed_tokens.weight.copy_(g / scale)
            head = g / scale**0.5
            self.lm_head.weight.copy_(head)


def build_model(config, **kwargs):
    return ScaledLM(config, **kwargs)


def build_optimizer(model, optimizer_name=None, learning_rate=None, weight_decay=0.0, beta1=0.9, beta2=0.95, **_):
    s = getattr(model, "scaling", settings(model.config.hidden_size, model.config.num_hidden_layers))
    layers = {id(p) for p in model.model.layers.parameters()}
    groups = {}
    for module in model.modules():
        for name, p in module.named_parameters(recurse=False):
            decay = should_apply_weight_decay(module, name)
            in_block = id(p) in layers
            if in_block and decay:      # hidden matrices
                key, lr, eps = "hidden", learning_rate * s["hidden_lr"] * s["block_lr"], ADAMW_EPSILON * s["block_eps"]
            elif in_block:              # within-block norms (incl. QK norms)
                key, lr, eps = "block_norm", learning_rate * s["block_lr"], ADAMW_EPSILON * s["block_eps"]
            elif decay:                 # readout
                key, lr, eps = "readout", learning_rate, ADAMW_EPSILON
            else:                       # embedding, final norm
                key, lr, eps = "other", learning_rate, ADAMW_EPSILON
            g = groups.setdefault(key, dict(params=[], lr=lr, eps=eps, weight_decay=weight_decay if decay else 0.0))
            g["params"].append(p)
    fused = any(p.is_cuda for p in model.parameters())
    return torch.optim.AdamW(list(groups.values()), lr=learning_rate, betas=(beta1, beta2), weight_decay=0.0,
                             fused=fused or None)
