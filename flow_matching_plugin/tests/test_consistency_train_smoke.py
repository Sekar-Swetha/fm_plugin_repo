"""CPU smoke test for the consistency train step (C3). main() needs a GPU."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from consistency_core import ConsistencyPair  # noqa: E402
from train_consistency import consistency_train_step  # noqa: E402


def test_train_step_returns_scalar_with_grad():
    T = 1000
    alphas = torch.linspace(1.0, 0.1, T + 1)
    sigmas = torch.linspace(0.0, 0.99, T + 1)
    pair = ConsistencyPair(
        x_hi=torch.randn(1, 4, 8, 8, 8), t_hi=900,
        x_lo=torch.randn(1, 4, 8, 8, 8), t_lo=600,
        cond=torch.randn(8, 3, 64, 64),
    )
    w = torch.zeros(1, requires_grad=True)   # a trainable "student" parameter

    def student_eps_fn(x, t):
        return x * 0.0 + w                    # depends on w so grad flows

    def ema_eps_fn(x, t):
        return torch.zeros_like(x)

    loss = consistency_train_step(student_eps_fn, ema_eps_fn, pair, alphas, sigmas)
    assert loss.ndim == 0
    loss.backward()
    assert w.grad is not None
