"""Unit tests for reflow pair generation + trainer (Contribution C)."""

import os
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow_matching_loss import (  # noqa: E402
    FlowMatchingConfig,
    cfm_ot_loss,
    compute_x_t,
)
from flow_inversion import flow_sample  # noqa: E402
from generate_reflow_pairs import (  # noqa: E402
    ReflowRecord,
    generate_pairs,
    save_shards,
)
from train_reflow import ReflowPairDataset, reflow_training_step  # noqa: E402


def _seed(s=0):
    torch.manual_seed(s)


def _stub_factory(cond_dim, latent_ch, seed=0):
    torch.manual_seed(seed)
    proj = torch.nn.Linear(cond_dim, latent_ch)

    def factory(cond):
        bias = proj(cond.flatten().float()[:cond_dim])

        def vf(x_t, t_idx):
            return torch.tanh(x_t) * 0.5 + bias.view(1, -1, *([1] * (x_t.dim() - 2))) * 0.1

        return vf

    return factory


class TestPairGeneratorShapes(unittest.TestCase):

    def _records(self, n=3):
        return [ReflowRecord(
            z_src=torch.randn(4, 4, 8, 8),
            cond_src=torch.randn(64),
            cond_tgt=torch.randn(64),
        ) for _ in range(n)]

    def test_c1_shapes_and_save(self):
        _seed(0)
        recs = self._records()
        factory = _stub_factory(64, 4)
        pairs = generate_pairs(recs, factory, "c1", num_steps=4, method="euler")
        self.assertEqual(len(pairs), 3)
        for z0, z1, cond in pairs:
            self.assertEqual(z0.shape, (4, 4, 8, 8))
            self.assertEqual(z1.shape, (4, 4, 8, 8))
            self.assertEqual(cond.shape, (64,))
            self.assertEqual(z0.dtype, torch.float32)

        with tempfile.TemporaryDirectory() as d:
            meta = {"mode": "c1", "seed": 0, "loss_type": "cfm_ot"}
            written = save_shards(pairs, d, meta)
            self.assertTrue(written)
            ds = ReflowPairDataset(d)
            self.assertEqual(len(ds), 3)
            z0, z1, cond = ds[0]
            self.assertEqual(z0.shape, (4, 4, 8, 8))

    def test_c2_pairs_use_source_latent_as_z0(self):
        _seed(1)
        recs = self._records(2)
        factory = _stub_factory(64, 4)
        pairs = generate_pairs(recs, factory, "c2", num_steps=4, method="euler")
        # C2 pair is (source latent -> edited latent): z0 must equal z_src.
        for (z0, _z1, _cond), rec in zip(pairs, recs):
            self.assertTrue(torch.allclose(z0, rec.z_src))

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            generate_pairs([], _stub_factory(64, 4), "c3")


class TestReflowLossMatchesCFM(unittest.TestCase):
    """With original (Z0~N, Z1~data) pairs, the reflow step's loss equals the
    CFM-OT loss bit-for-bit — the reused loss has not regressed."""

    def test_bit_for_bit(self):
        _seed(2)
        cfg = FlowMatchingConfig(sigma_min=0.0)
        z0 = torch.randn(8, 4, 4, 8, 8)       # noise
        z1 = torch.randn(8, 4, 4, 8, 8)       # data
        cond = torch.randn(8, 64)
        t = torch.rand(8).clamp(cfg.t_eps, 1 - cfg.t_eps)

        # A fixed velocity prediction independent of how it's produced.
        x_t = compute_x_t(z0, z1, t, sigma_min=cfg.sigma_min)
        v_pred = 0.37 * x_t - 0.11

        captured = {}

        def model(x_in, t_idx, c):
            captured["x"] = x_in
            return v_pred

        loss_reflow = reflow_training_step(model, z0, z1, cond, t, cfg)
        loss_cfm = cfm_ot_loss(v_pred, z0, z1, sigma_min=cfg.sigma_min)
        self.assertEqual(loss_reflow.item(), loss_cfm.item())
        # And the trainer fed the model the correct x_t.
        self.assertTrue(torch.allclose(captured["x"], x_t))


class TestRound1Straighter(unittest.TestCase):
    """On a toy 2-D problem, one reflow round strictly straightens the ODE
    trajectories (lower deviation from the straight endpoint interpolation)."""

    @staticmethod
    def _toy_data(n):
        # Two well-separated clusters -> a curved transport from N(0,I).
        c = torch.tensor([[3.0, 3.0], [-3.0, -3.0]])
        idx = torch.randint(0, 2, (n,))
        return c[idx] + 0.3 * torch.randn(n, 2)

    @staticmethod
    def _mlp():
        return torch.nn.Sequential(
            torch.nn.Linear(3, 64), torch.nn.SiLU(),
            torch.nn.Linear(64, 64), torch.nn.SiLU(),
            torch.nn.Linear(64, 2),
        )

    @classmethod
    def _train(cls, model, pairs_z0, pairs_z1, cfg, steps=1500, lr=3e-3):
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        n = pairs_z0.shape[0]
        for _ in range(steps):
            idx = torch.randint(0, n, (256,))
            z0, z1 = pairs_z0[idx], pairs_z1[idx]
            t = torch.rand(256).clamp(cfg.t_eps, 1 - cfg.t_eps)

            def model_call(x_t, t_idx, c):
                t_cont = t_idx.float() / (cfg.num_train_timesteps - 1)
                return model(torch.cat([x_t, t_cont.view(-1, 1)], dim=-1))

            loss = reflow_training_step(model_call, z0, z1, None, t, cfg)
            opt.zero_grad(); loss.backward(); opt.step()

    @staticmethod
    def _straightness(model, z0, cfg, n_steps=50):
        def vf(x, t_idx):
            t_cont = t_idx.float() / (cfg.num_train_timesteps - 1)
            return model(torch.cat([x, t_cont.view(-1, 1)], dim=-1))

        res = flow_sample(vf, z0, num_steps=n_steps, method="euler",
                          config=cfg, return_trajectory=True)
        traj = torch.stack(res.trajectory, dim=0)            # (n+1, B, 2)
        starts, ends = traj[0], traj[-1]
        steps = torch.linspace(0, 1, traj.shape[0]).view(-1, 1, 1)
        interp = starts[None] + steps * (ends - starts)[None]
        return (traj - interp).norm(dim=-1).mean().item()

    def test_round1_straighter_than_round0(self):
        _seed(7)
        cfg = FlowMatchingConfig(sigma_min=0.0)
        data = self._toy_data(2048)

        # Round 0: couple N(0,I) with data (random pairing).
        z0_r0 = torch.randn_like(data)
        model0 = self._mlp()
        self._train(model0, z0_r0, data, cfg)

        # Round 1: pairs are (z0, model0's own transport of z0).
        z0_r1 = torch.randn(2048, 2)
        with torch.no_grad():
            def vf0(x, t_idx):
                t_cont = t_idx.float() / (cfg.num_train_timesteps - 1)
                return model0(torch.cat([x, t_cont.view(-1, 1)], dim=-1))
            z1_r1 = flow_sample(vf0, z0_r1, num_steps=50, method="euler", config=cfg).latent
        model1 = self._mlp()
        self._train(model1, z0_r1, z1_r1, cfg)

        torch.manual_seed(99)
        probe = torch.randn(512, 2)
        s0 = self._straightness(model0, probe, cfg)
        s1 = self._straightness(model1, probe.clone(), cfg)
        self.assertLess(s1, s0, msg=f"round1 not straighter: s0={s0:.4f} s1={s1:.4f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
