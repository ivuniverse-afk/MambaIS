"""Selective sequence mixer with an official Mamba kernel and debug fallback."""

import torch
from torch import nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except (ImportError, OSError):
    Mamba = None


class RecurrentSelectiveSSM(nn.Module):
    """Small linear recurrent fallback; intended for CPU tests, not benchmarks."""
    def __init__(self, width, state_size=16):
        super().__init__()
        self.state_size = state_size
        self.in_proj = nn.Linear(width, 3 * state_size)
        self.out_proj = nn.Linear(state_size, width)
        self.log_decay = nn.Parameter(torch.zeros(state_size))

    def forward(self, x):
        u, b, c = self.in_proj(x).chunk(3, -1)
        decay = torch.sigmoid(self.log_decay)
        state = x.new_zeros(x.shape[0], self.state_size)
        outputs = []
        for t in range(x.shape[1]):
            state = decay * state + torch.sigmoid(b[:, t]) * u[:, t]
            outputs.append(c[:, t] * state)
        return self.out_proj(torch.stack(outputs, 1))


class SelectiveMixer(nn.Module):
    def __init__(self, width, state_size=16, conv_width=4, expansion=2):
        super().__init__()
        self.impl = (Mamba(d_model=width, d_state=state_size,
                           d_conv=conv_width, expand=expansion)
                     if Mamba is not None else RecurrentSelectiveSSM(width, state_size))

    def forward(self, x):
        return self.impl(x)


class SequenceOperator(nn.Module):
    """Linear -> depth-wise local context -> selective SSM, i.e. M(X)."""
    def __init__(self, width, state_size=16):
        super().__init__()
        self.linear = nn.Linear(width, width)
        self.local = nn.Conv1d(width, width, 3, padding=1, groups=width)
        self.ssm = SelectiveMixer(width, state_size)

    def forward(self, x):
        x = self.linear(x)
        x = F.silu(self.local(x.transpose(1, 2)).transpose(1, 2))
        return self.ssm(x)

