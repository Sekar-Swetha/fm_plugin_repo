"""Unit tests for flow_inversion (Contribution B)."""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow_matching_loss import FlowMatchingConfig  # noqa: E402
from flow_inversion import (  # noqa: E402
    MotionEditorFlowInversion,
    assert_cfm_ot_checkpoint,
    flow_invert,
    flow_sample,
    make_velocity_fn,
    roundtrip_error,
)


def _seed(s=0):
    torch.manual_seed(s)


class TestIdentityVelocityRoundTrip(unittest.TestCase):
    """A constant velocity v = x1 - x0 makes the path an exact straight line;
    Euler must invert and re-sample it exactly for any num_steps."""

    def test_straight_line_exact(self):
        _seed()
        x0 = torch.randn(2, 4, 4, 8, 8)
        x1 = torch.randn_like(x0)
        v_const = x1 - x0

        def vf(x, t_idx):
            return v_const

        for n in (1, 4, 8):
            recovered = flow_invert(vf, x1, num_steps=n, method="euler").latent
            self.assertTrue(torch.allclose(recovered, x0, atol=1e-5),
                            msg=f"invert mismatch at n={n}")
            resampled = flow_sample(vf, recovered, num_steps=n, method="euler").latent
            self.assertTrue(torch.allclose(resampled, x1, atol=1e-5),
                            msg=f"resample mismatch at n={n}")


class TestOTVelocityRoundTrip(unittest.TestCase):
    """A position-dependent OT field (pulls toward a fixed target). Round-trip
    error should be small and shrink with more steps."""

    @staticmethod
    def _field():
        target = torch.tensor([1.5, -0.5])

        def vf(x, t_idx):
            return target - x  # contraction toward target

        return vf

    def test_roundtrip_small_and_decreasing(self):
        _seed(1)
        vf = self._field()
        x1 = torch.randn(64, 2)
        # Euler: more steps -> smaller round-trip error.
        e_coarse = roundtrip_error(vf, x1, num_steps=2, method="euler")
        e_fine = roundtrip_error(vf, x1, num_steps=16, method="euler")
        self.assertLess(e_fine, e_coarse)
        # Heun at a handful of steps already reconstructs to high precision.
        e_heun = roundtrip_error(vf, x1, num_steps=8, method="heun")
        self.assertLess(e_heun, 1e-2)


class TestBatchShapes(unittest.TestCase):

    def test_shapes_preserved(self):
        _seed(2)
        for shape in [(1, 2), (4, 3, 8, 8), (2, 4, 4, 8, 8)]:
            x1 = torch.randn(*shape)

            def vf(x, t_idx):
                return 0.1 * x

            r_inv = flow_invert(vf, x1, num_steps=4)
            r_smp = flow_sample(vf, x1, num_steps=4)
            self.assertEqual(r_inv.latent.shape, x1.shape)
            self.assertEqual(r_smp.latent.shape, x1.shape)


class TestHeunVsEulerScaling(unittest.TestCase):
    """On a smooth nonlinear field, Heun (O(1/N^2)) must beat Euler (O(1/N))
    and its error must fall faster as N doubles."""

    @staticmethod
    def _field():
        def vf(x, t_idx):
            return torch.sin(x) + 0.3 * x

        return vf

    def test_heun_beats_euler(self):
        _seed(3)
        vf = self._field()
        x1 = torch.randn(32, 4)
        e_euler = roundtrip_error(vf, x1, num_steps=8, method="euler")
        e_heun = roundtrip_error(vf, x1, num_steps=8, method="heun")
        self.assertLess(e_heun, e_euler)

    def test_heun_order_higher_than_euler(self):
        _seed(3)
        vf = self._field()
        x1 = torch.randn(32, 4)
        eu_lo = roundtrip_error(vf, x1, num_steps=4, method="euler")
        eu_hi = roundtrip_error(vf, x1, num_steps=8, method="euler")
        he_lo = roundtrip_error(vf, x1, num_steps=4, method="heun")
        he_hi = roundtrip_error(vf, x1, num_steps=8, method="heun")
        # Doubling steps: Euler error ~halves (ratio ~2), Heun ~quarters (ratio ~4).
        euler_ratio = eu_lo / max(eu_hi, 1e-12)
        heun_ratio = he_lo / max(he_hi, 1e-12)
        self.assertGreater(heun_ratio, euler_ratio)


class TestMotionEditorFlowInversionMockPath(unittest.TestCase):
    """Exercise the wrapper with simple stubs standing in for unet/controlnet/
    text encoder."""

    def test_invert_with_stubs(self):
        _seed(4)

        class StubText:
            def __call__(self, ids):
                return (torch.zeros(1, 4, 8),)  # tuple -> [0]

        class StubControlNet:
            def __call__(self, x, t, encoder_hidden_states=None, controlnet_cond=None):
                return ([torch.zeros_like(x)], torch.zeros_like(x))  # (down, mid)

        class StubUNet:
            def __call__(self, x, t, **kw):
                class O:  # object with a .sample attribute, like diffusers
                    pass
                o = O()
                o.sample = 0.2 * x
                return o

        inv = MotionEditorFlowInversion(
            unet=StubUNet(),
            controlnet=StubControlNet(),
            text_encoder=StubText(),
            source_skeleton=torch.zeros(1, 3, 8, 8),
            num_steps=4,
            method="heun",
        )
        latents = torch.randn(1, 4, 4, 8, 8)
        noise = inv.invert(latents, prompt_ids=torch.zeros(1, 1, dtype=torch.long))
        self.assertEqual(noise.shape, latents.shape)


class TestEndToEndSmoke(unittest.TestCase):
    """B.2.3: tiny U-Net stub, invert a (1, 4, 4, 8, 8) latent."""

    def test_tiny_unet_invert(self):
        _seed(5)

        class TinyUNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(0.1))

            def forward(self, x, t_idx, **kw):
                return self.scale * x

        unet = TinyUNet()
        vf = make_velocity_fn(unet, cond=torch.zeros(1, 4, 8, 8))
        x1 = torch.randn(1, 4, 4, 8, 8)
        result = flow_invert(vf, x1, num_steps=8, method="heun")
        self.assertEqual(result.latent.shape, x1.shape)
        self.assertTrue(torch.isfinite(result.latent).all())


class TestRoundTripRegression(unittest.TestCase):
    """B.2.3: pin the round-trip error at num_steps=8 for a fixed seed."""

    def test_pinned_value(self):
        torch.manual_seed(1234)

        def vf(x, t_idx):
            return torch.tanh(x)

        x1 = torch.randn(8, 16)
        err = roundtrip_error(vf, x1, num_steps=8, method="heun")
        # Regression lock on this exact integrator for the fixed seed.
        self.assertAlmostEqual(err, 0.000653, places=5)


class TestCheckpointGuard(unittest.TestCase):

    def test_cfm_ot_passes(self):
        assert_cfm_ot_checkpoint("cfm_ot")  # no raise

    def test_epsilon_raises(self):
        with self.assertRaises(ValueError):
            assert_cfm_ot_checkpoint("epsilon")

    def test_none_raises(self):
        with self.assertRaises(ValueError):
            assert_cfm_ot_checkpoint(None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
