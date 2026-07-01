"""CPU unit tests for the few-step consistency sampler (C3)."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "motionEditor", "MotionEditor", "motion_editor"))
from consistency_plugin import consistency_sample  # noqa: E402


def test_consistency_sample_step_count_and_shape():
    calls = {"n": 0}

    def model_fn(x, t):
        calls["n"] += 1
        return torch.zeros_like(x)          # eps=0 -> x0 = x/alpha

    x = torch.randn(1, 4, 8, 8, 8)
    T = 1000
    alphas = torch.linspace(1.0, 0.1, T + 1)
    sigmas = torch.linspace(0.0, 0.99, T + 1)
    timesteps = [900, 600, 300, 0]          # 4-step schedule
    out = consistency_sample(model_fn, x, timesteps, alphas, sigmas,
                             generator=torch.Generator().manual_seed(0))
    assert calls["n"] == 4                  # one model eval per step
    assert out.shape == x.shape


def test_consistency_sample_last_step_returns_x0_not_renoised():
    # With eps=0 and final t=0 (alpha=1, sigma=0), final x0 == input-at-that-step.
    def model_fn(x, t):
        return torch.zeros_like(x)

    x = torch.ones(1, 4, 4, 4, 4)
    T = 10
    alphas = torch.ones(T + 1)              # alpha=1 everywhere -> x0 = x
    sigmas = torch.zeros(T + 1)
    out = consistency_sample(model_fn, x, [5, 0], alphas, sigmas)
    assert torch.allclose(out, x, atol=1e-6)
