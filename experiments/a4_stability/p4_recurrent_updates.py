"""Problem 4: edit the temporal update, keeping the modern RNN's depth stack.

    uv run python -m experiments.a4_stability.p4_recurrent_updates

z is the fraction of NEW information, following the handout. Reset is applied
BEFORE the candidate's recurrent projection; torch.nn.GRU uses a different form.
"""
import torch
from torch import nn
from .p2_rnn_depth import TanhCell, ModernRNN


class Gate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input = nn.Linear(config.hidden_size, config.hidden_size)
        self.hidden = nn.Linear(config.hidden_size, config.hidden_size)

    def project(self, x):
        return self.input(x)

    def forward(self, projected, h):
        return (projected + self.hidden(h)).sigmoid()


class GRUCell(TanhCell):
    def __init__(self, config):
        super().__init__(config)
        self.reset = self.make_reset(config)
        self.update = self.make_update(config)

    def make_reset(self, config):
        return Gate(config)

    def make_update(self, config):
        return Gate(config)

    def project(self, x):
        return self.input(x), self.reset.project(x), self.update.project(x)

    def step(self, projected, h):
        candidate_input, reset_input, update_input = projected
        r, z = self.reset(reset_input, h), self.update(update_input, h)
        candidate = torch.tanh(candidate_input + self.hidden(r*h))
        return self.combine(h, candidate, z)

    def combine(self, h, candidate, z):
        return (1-z)*h + z*candidate


class GRU(ModernRNN):
    cell_type = GRUCell


class InputGate(Gate):
    """Gate from the current input only: no recurrent term inside the gate."""
    def __init__(self, config):
        nn.Module.__init__(self)
        self.input = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, projected, h):
        return projected.sigmoid()


class NoResetGRUCell(GRUCell):
    """Ablation: r = 1, the candidate always sees the whole hidden state."""
    def make_reset(self, config):
        return None

    def project(self, x):
        return self.input(x), self.update.project(x)

    def step(self, projected, h):
        candidate_input, update_input = projected
        z = self.update(update_input, h)
        candidate = torch.tanh(candidate_input + self.hidden(h))
        return self.combine(h, candidate, z)


class InputGateGRUCell(GRUCell):
    """Ablation: both gates read x_t only; the candidate still reads r*h."""
    def make_reset(self, config):
        return InputGate(config)

    def make_update(self, config):
        return InputGate(config)


class LinearCandidateGRUCell(GRUCell):
    """Ablation: drop the candidate tanh, so the update is a gated linear map."""
    def step(self, projected, h):
        candidate_input, reset_input, update_input = projected
        r, z = self.reset(reset_input, h), self.update(update_input, h)
        candidate = candidate_input + self.hidden(r*h)
        return self.combine(h, candidate, z)


class DualGateCell(GRUCell):
    """New update: decoupled forget and write gates, no reset.

    h' = f*h + i*tanh(W x + U h), with f and i separate sigmoid gates of (x, h).
    Unlike the GRU's convex combination, keeping and writing are independent, so
    the cell can hold h while also adding new information, or clear it outright.
    """
    def make_reset(self, config):
        return None

    def project(self, x):
        return self.input(x), self.update.project(x), self.write.project(x)

    def __init__(self, config):
        super().__init__(config)
        self.write = Gate(config)

    def step(self, projected, h):
        candidate_input, forget_input, write_input = projected
        f, i = self.update(forget_input, h), self.write(write_input, h)
        candidate = torch.tanh(candidate_input + self.hidden(h))
        return self.finish(f*h + i*candidate)

    def finish(self, h):
        return h


class DualGateNormCell(DualGateCell):
    """DualGateCell followed by a parameter-free RMS normalization of h."""
    def finish(self, h):
        return h * torch.rsqrt(h.float().square().mean(-1, keepdim=True) + 1e-6).to(h.dtype)


class NoResetGRU(ModernRNN):
    cell_type = NoResetGRUCell


class InputGateGRU(ModernRNN):
    cell_type = InputGateGRUCell


class LinearGRU(ModernRNN):
    cell_type = LinearCandidateGRUCell


class DualGateRNN(ModernRNN):
    cell_type = DualGateCell


class DualGateNormRNN(ModernRNN):
    cell_type = DualGateNormCell


MODELS = dict(gru=GRU, tanh=ModernRNN, noreset=NoResetGRU, inputgate=InputGateGRU,
              linear=LinearGRU, dual=DualGateRNN, dualnorm=DualGateNormRNN)


if __name__ == "__main__":
    from .sweeps import launch_one
    launch_one('gru')  # Edit the variant and LR here.
