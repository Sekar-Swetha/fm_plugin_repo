"""CPU tests for metrics.py helpers (auto-crop + align). No LPIPS/CLIP needed."""
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics import align_len, fmt, load_gif_frames  # noqa: E402


def _write_gif(path, w, h, left_rgb, right_rgb, n=3):
    frames = []
    for _ in range(n):
        im = Image.new("RGB", (w, h), left_rgb)
        # paint right half a different colour
        im.paste(Image.new("RGB", (w - w // 2, h), right_rgb), (w // 2, 0))
        frames.append(im)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0)


def test_side_by_side_gif_crops_to_right_panel(tmp_path):
    p = os.path.join(tmp_path, "sbs.gif")
    _write_gif(p, 1030, 516, left_rgb=(255, 0, 0), right_rgb=(0, 255, 0))
    f = load_gif_frames(p, size=64)               # (N,3,64,64) in [0,1]
    # right panel is green: G channel ~1, R channel ~0
    assert f[:, 1].mean() > 0.8
    assert f[:, 0].mean() < 0.2


def test_square_gif_kept_whole(tmp_path):
    p = os.path.join(tmp_path, "sq.gif")
    _write_gif(p, 512, 512, left_rgb=(255, 0, 0), right_rgb=(0, 0, 255))
    f = load_gif_frames(p, size=64)
    # square -> not cropped -> both red (left) and blue (right) present
    assert f[:, 0].mean() > 0.2 and f[:, 2].mean() > 0.2


def test_align_len_truncates_to_shorter():
    a = torch.randn(8, 3, 4, 4)
    b = torch.randn(5, 3, 4, 4)
    a2, b2 = align_len(a, b)
    assert a2.shape[0] == 5 and b2.shape[0] == 5


def test_fmt_handles_none():
    assert fmt(None) == "-"
    assert fmt(0.12345, 3) == "0.123"
