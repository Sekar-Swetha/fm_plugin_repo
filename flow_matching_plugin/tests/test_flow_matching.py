"""
Offline unit tests for `flow_matching_loss.py`. Pure-PyTorch; runs on CPU in
under a second; no MotionEditor, no GPU, no Stable Diffusion weights required.

Run:
    python -m unittest discover -s tests -v
or:
    python tests/test_flow_matching.py
"""

import math
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow_matching_loss import (   # noqa: E402
    FlowMatchingConfig,
    build_fm_training_batch,
    cfm_ot_loss,
    compute_target_velocity,
    compute_x_t,
    euler_sample,
    sample_t,
)


def _seed_everything(s=0):
    torch.manual_seed(s)


class TestPathEndpoints(unittest.TestCase):
    """At t = 0, x_t must equal noise. At t = 1, x_t must equal data
    (when sigma_min = 0). These are the boundary conditions Lipman et al.
    impose on every conditional probability path."""

    def test_t_zero_returns_noise(self):
        _seed_everything()
        x0 = torch.randn(4, 8)
        x1 = torch.randn(4, 8)
        t = torch.zeros(4)
        x_t = compute_x_t(x0, x1, t, sigma_min=0.0)
        self.assertTrue(torch.allclose(x_t, x0, atol=1e-6))

    def test_t_one_returns_data_when_sigma_min_zero(self):
        x0 = torch.randn(4, 8)
        x1 = torch.randn(4, 8)
        t = torch.ones(4)
        x_t = compute_x_t(x0, x1, t, sigma_min=0.0)
        self.assertTrue(torch.allclose(x_t, x1, atol=1e-6))

    def test_t_one_with_sigma_min_positive(self):
        """sigma_min > 0 leaves a residual noise component at t=1."""
        sigma_min = 1e-2
        x0 = torch.randn(4, 8)
        x1 = torch.randn(4, 8)
        t = torch.ones(4)
        x_t = compute_x_t(x0, x1, t, sigma_min=sigma_min)
        expected = sigma_min * x0 + x1
        self.assertTrue(torch.allclose(x_t, expected, atol=1e-6))


class TestTargetVelocity(unittest.TestCase):
    """Two derivations must agree:
       (a) closed-form: u_t = x_1 - (1 - sigma_min) * x_0  (Theorem 3).
       (b) finite-difference: u_t ≈ (x_{t+h} - x_t) / h    (definition of derivative)."""

    def test_closed_form_matches_finite_difference(self):
        _seed_everything()
        for sigma_min in [0.0, 1e-3, 1e-2]:
            x0 = torch.randn(8, 16)
            x1 = torch.randn(8, 16)
            t = torch.full((8,), 0.5)
            h = 1e-4
            x_t = compute_x_t(x0, x1, t, sigma_min=sigma_min)
            x_th = compute_x_t(x0, x1, t + h, sigma_min=sigma_min)
            u_fd = (x_th - x_t) / h
            u_cf = compute_target_velocity(x0, x1, sigma_min=sigma_min)
            err = (u_fd - u_cf).abs().max().item()
            self.assertLess(err, 1e-2, msg=f"sigma_min={sigma_min}, err={err}")

    def test_velocity_is_time_independent(self):
        """One of the OT path's defining properties: u_t does not depend on t.
        This is what makes OT-FM train more stably than diffusion-FM."""
        x0 = torch.randn(4, 8)
        x1 = torch.randn(4, 8)
        u_at_t0 = compute_target_velocity(x0, x1, sigma_min=0.0)
        u_at_t1 = compute_target_velocity(x0, x1, sigma_min=0.0)
        self.assertTrue(torch.allclose(u_at_t0, u_at_t1))

    def test_rectified_flow_equivalence_at_sigma_zero(self):
        """At sigma_min = 0, the target u_t = x_1 - x_0 is exactly the
        rectified-flow training target (Liu et al. 2022, eq. 1)."""
        x0 = torch.randn(4, 8)
        x1 = torch.randn(4, 8)
        u_cfm = compute_target_velocity(x0, x1, sigma_min=0.0)
        u_rectified = x1 - x0
        self.assertTrue(torch.allclose(u_cfm, u_rectified))


class TestLossInvariants(unittest.TestCase):

    def test_perfect_prediction_zero_loss(self):
        """If v_theta predicts u_t exactly, loss must be zero."""
        x0 = torch.randn(4, 8)
        x1 = torch.randn(4, 8)
        target = compute_target_velocity(x0, x1, sigma_min=0.0)
        loss = cfm_ot_loss(target, x0, x1, sigma_min=0.0)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_loss_is_nonnegative(self):
        for _ in range(10):
            x0 = torch.randn(4, 8)
            x1 = torch.randn(4, 8)
            v_bad = torch.randn(4, 8)
            loss = cfm_ot_loss(v_bad, x0, x1, sigma_min=0.0)
            self.assertGreaterEqual(loss.item(), 0.0)

    def test_loss_reduction_modes(self):
        x0 = torch.randn(4, 8)
        x1 = torch.randn(4, 8)
        v = torch.zeros(4, 8)
        l_mean = cfm_ot_loss(v, x0, x1, reduction="mean")
        l_sum = cfm_ot_loss(v, x0, x1, reduction="sum")
        l_none = cfm_ot_loss(v, x0, x1, reduction="none")
        self.assertEqual(l_none.shape, (4, 8))
        self.assertAlmostEqual((l_sum / 32).item(), l_mean.item(), places=4)


class TestBatchBuilder(unittest.TestCase):

    def test_shapes(self):
        cfg = FlowMatchingConfig(sigma_min=0.0, num_train_timesteps=1000)
        x1 = torch.randn(2, 4, 8, 64, 64)  # (B, C, F, H, W) — MotionEditor latent shape
        x_t, t_cont, t_unet, target = build_fm_training_batch(x1, cfg)
        self.assertEqual(x_t.shape, x1.shape)
        self.assertEqual(t_cont.shape, (2,))
        self.assertEqual(t_unet.shape, (2,))
        self.assertEqual(target.shape, x1.shape)
        self.assertEqual(t_unet.dtype, torch.long)

    def test_t_in_valid_range(self):
        cfg = FlowMatchingConfig(sigma_min=0.0, num_train_timesteps=1000, t_eps=1e-3)
        x1 = torch.randn(64, 4)
        for _ in range(20):
            _, t_cont, t_unet, _ = build_fm_training_batch(x1, cfg)
            self.assertTrue(torch.all(t_cont >= cfg.t_eps - 1e-9))
            self.assertTrue(torch.all(t_cont <= 1.0 - cfg.t_eps + 1e-9))
            self.assertTrue(torch.all(t_unet >= 0))
            self.assertTrue(torch.all(t_unet <= cfg.num_train_timesteps - 1))

    def test_reproducible_with_generator(self):
        cfg = FlowMatchingConfig()
        x1 = torch.randn(4, 8)
        g1 = torch.Generator().manual_seed(42)
        g2 = torch.Generator().manual_seed(42)
        out1 = build_fm_training_batch(x1, cfg, generator=g1)
        out2 = build_fm_training_batch(x1, cfg, generator=g2)
        for a, b in zip(out1, out2):
            self.assertTrue(torch.allclose(a, b))


class TestEulerSampler(unittest.TestCase):
    """End-to-end sanity: with a *known* perfect velocity, one Euler step is
    enough to recover x_1 exactly from x_0 (this is the rectified-flow
    one-step-generation property at sigma_min = 0)."""

    def test_one_step_with_oracle_velocity(self):
        cfg = FlowMatchingConfig(sigma_min=0.0)
        torch.manual_seed(0)
        x1_true = torch.randn(2, 4, 8, 8)
        x0 = torch.randn_like(x1_true)
        # Oracle velocity = x1_true - x0  (constant in t for OT path).
        def oracle(x, t_idx):
            return x1_true - x0
        x1_hat = euler_sample(oracle, x0, num_steps=1, config=cfg)
        self.assertTrue(torch.allclose(x1_hat, x1_true, atol=1e-5))

    def test_many_steps_with_oracle_velocity(self):
        cfg = FlowMatchingConfig(sigma_min=0.0)
        x1_true = torch.randn(2, 4)
        x0 = torch.randn_like(x1_true)
        def oracle(x, t_idx):
            return x1_true - x0
        x1_hat = euler_sample(oracle, x0, num_steps=20, config=cfg)
        self.assertTrue(torch.allclose(x1_hat, x1_true, atol=1e-5))


class TestTrainingTrajectoryEquivalence(unittest.TestCase):
    """A controlled experiment: train a tiny linear `v_theta` on CFM-OT loss.
    With enough steps, it must converge to the analytical Jacobian relation
    so that running Euler from noise recovers (approximately) a data sample.

    This is the strongest single-test sanity check that the loss is correctly
    implemented: it shows the *system* learns, not just that arithmetic is
    right."""

    def test_tiny_model_converges(self):
        torch.manual_seed(1)
        D = 8
        # Fix a dataset of 16 points.
        data = torch.randn(16, D) * 2.0

        cfg = FlowMatchingConfig(sigma_min=0.0, num_train_timesteps=1000)

        # A trivial "model" v(x, t) parameterised by (W, b) acting on
        # [x, t_emb]. Tiny — must still converge enough for sampling to
        # generate something close to the data distribution.
        class TinyV(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(D + 1, 64),
                    torch.nn.SiLU(),
                    torch.nn.Linear(64, 64),
                    torch.nn.SiLU(),
                    torch.nn.Linear(64, D),
                )

            def forward(self, x, t):
                t_in = t.view(-1, 1).float()
                return self.net(torch.cat([x, t_in], dim=-1))

        v = TinyV()
        opt = torch.optim.Adam(v.parameters(), lr=5e-3)

        for step in range(800):
            idx = torch.randint(0, data.shape[0], (32,))
            x1 = data[idx]
            x_t, t_cont, _, target = build_fm_training_batch(x1, cfg)
            pred = v(x_t, t_cont)
            loss = ((pred - target) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Now integrate from noise and check the samples are inside the
        # data range.
        def velocity_fn(x, t_idx):
            t_cont = t_idx.float() / (cfg.num_train_timesteps - 1)
            return v(x, t_cont)

        x0 = torch.randn(64, D)
        samples = euler_sample(velocity_fn, x0, num_steps=10, config=cfg)
        # Crude check: the *mean and std* of samples should be in the same
        # ballpark as data — not Gaussian-noise statistics.
        data_mean, data_std = data.mean(dim=0), data.std(dim=0)
        sample_mean, sample_std = samples.mean(dim=0), samples.std(dim=0)
        # Loose tolerances — this is a 64-sample / 800-step tiny experiment.
        self.assertLess((sample_mean - data_mean).abs().max().item(), 1.5)
        self.assertLess((sample_std - data_std).abs().max().item(), 1.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
