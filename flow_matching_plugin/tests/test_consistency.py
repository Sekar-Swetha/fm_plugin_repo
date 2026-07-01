"""CPU unit tests for consistency_core (C3)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from consistency_core import (  # noqa: E402
    ConsistencyPair,
    ema_update,
    load_pairs,
    pairs_from_trajectory,
    predict_x0_from_eps,
    pseudo_huber_loss,
    save_pairs,
)


def test_predict_x0_boundary_is_identity_at_t0():
    # At t=0: alpha=1, sigma=0 -> f(x,0) must equal x (boundary condition).
    x = torch.randn(2, 4, 8, 8, 8)
    eps = torch.randn_like(x)
    alpha0 = torch.tensor(1.0)
    sigma0 = torch.tensor(0.0)
    out = predict_x0_from_eps(x, eps, alpha0, sigma0)
    assert torch.allclose(out, x, atol=1e-6)


def test_predict_x0_matches_closed_form():
    x = torch.randn(1, 4, 8, 8, 8)
    eps = torch.randn_like(x)
    alpha = torch.tensor(0.6)
    sigma = torch.tensor(0.8)
    out = predict_x0_from_eps(x, eps, alpha, sigma)
    expected = (x - sigma * eps) / alpha
    assert torch.allclose(out, expected, atol=1e-6)


def test_pseudo_huber_zero_when_equal():
    a = torch.randn(3, 5)
    assert torch.allclose(pseudo_huber_loss(a, a.clone(), delta=1.0),
                          torch.tensor(0.0), atol=1e-6)


def test_pseudo_huber_positive_and_scalar():
    a = torch.zeros(4)
    b = torch.ones(4)
    loss = pseudo_huber_loss(a, b, delta=0.5)
    assert loss.ndim == 0 and loss.item() > 0.0


def test_ema_update_moves_toward_params():
    ema = [torch.zeros(2, 2)]
    p = [torch.ones(2, 2)]
    ema_update(ema, p, decay=0.9)
    # 0.9*0 + 0.1*1 = 0.1
    assert torch.allclose(ema[0], torch.full((2, 2), 0.1), atol=1e-6)


def test_pair_save_load_roundtrip(tmp_path):
    pairs = [
        ConsistencyPair(
            x_hi=torch.randn(4, 8, 8, 8), t_hi=900,
            x_lo=torch.randn(4, 8, 8, 8), t_lo=850,
            cond=torch.randn(8, 3, 64, 64),
        ),
        ConsistencyPair(
            x_hi=torch.randn(4, 8, 8, 8), t_hi=850,
            x_lo=torch.randn(4, 8, 8, 8), t_lo=800,
            cond=torch.randn(8, 3, 64, 64),
        ),
    ]
    p = os.path.join(tmp_path, "pairs.pt")
    save_pairs(pairs, p)
    out = load_pairs(p)
    assert len(out) == 2
    assert out[0].t_hi == 900 and out[0].t_lo == 850
    assert torch.allclose(out[1].x_hi, pairs[1].x_hi)
    assert out[0].cond.shape == (8, 3, 64, 64)


def test_pairs_from_trajectory_consecutive():
    cond = torch.randn(8, 3, 64, 64)
    traj = [
        (900, torch.randn(4, 8, 8, 8)),
        (600, torch.randn(4, 8, 8, 8)),
        (300, torch.randn(4, 8, 8, 8)),
    ]
    pairs = pairs_from_trajectory(traj, cond)
    assert len(pairs) == 2
    assert pairs[0].t_hi == 900 and pairs[0].t_lo == 600
    assert pairs[1].t_hi == 600 and pairs[1].t_lo == 300
    assert torch.allclose(pairs[0].x_lo, traj[1][1])
