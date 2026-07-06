"""CPU tests for metrics.py helpers (auto-crop + align). No LPIPS/CLIP needed."""
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics import (  # noqa: E402
    align_len,
    bg_region,
    fmt,
    load_frames,
    load_gif_frames,
    region_weighted_mean,
    ssim_map,
    subject_region,
)


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


# ---- region-split metric helpers ----

def test_bg_region_excludes_union_of_masks():
    # src subject = left half, out subject = right half -> bg = neither = 0 everywhere
    src = torch.zeros(1, 1, 4, 4); src[..., :2] = 1.0
    out = torch.zeros(1, 1, 4, 4); out[..., 2:] = 1.0
    bg = bg_region(src, out)
    assert torch.allclose(bg, torch.zeros_like(bg))
    # disjoint-from-subject pixel stays background
    src2 = torch.zeros(1, 1, 4, 4); src2[..., 0, 0] = 1.0
    out2 = torch.zeros(1, 1, 4, 4); out2[..., 0, 1] = 1.0
    bg2 = bg_region(src2, out2)
    assert bg2[0, 0, 3, 3] == 1.0 and bg2[0, 0, 0, 0] == 0.0 and bg2[0, 0, 0, 1] == 0.0


def test_subject_region_is_union():
    a = torch.zeros(1, 1, 4, 4); a[..., :2] = 1.0
    b = torch.zeros(1, 1, 4, 4); b[..., 2:] = 1.0
    u = subject_region(a, b)
    assert torch.allclose(u, torch.ones_like(u))


def test_region_weighted_mean_only_counts_region():
    dist = torch.zeros(1, 1, 2, 2)
    dist[0, 0, 0, 0] = 10.0     # this pixel is outside the region -> ignored
    dist[0, 0, 1, 1] = 2.0      # inside the region
    region = torch.zeros(1, 1, 2, 2); region[0, 0, 1, 1] = 1.0
    assert region_weighted_mean(dist, region) == 2.0


def test_region_weighted_mean_skips_empty_region():
    dist = torch.ones(1, 1, 2, 2)
    region = torch.zeros(1, 1, 2, 2)
    assert region_weighted_mean(dist, region) is None


def test_ssim_map_identity_is_one():
    x = torch.rand(1, 3, 32, 32)
    m = ssim_map(x, x.clone())
    assert m.mean().item() > 0.99


def test_ssim_map_lower_for_different():
    x = torch.zeros(1, 3, 32, 32)
    y = torch.ones(1, 3, 32, 32)
    same = ssim_map(x, x.clone()).mean().item()
    diff = ssim_map(x, y).mean().item()
    assert diff < same


def test_load_frames_prefers_pngs(tmp_path):
    import os
    from PIL import Image as _I
    # a gif (red) and a sibling *_frames dir of green PNGs -> loader must pick PNGs
    gif = os.path.join(tmp_path, "run.gif")
    _write_gif(gif, 64, 64, (255, 0, 0), (255, 0, 0))
    fdir = os.path.join(tmp_path, "run_frames")
    os.makedirs(fdir)
    for i in range(3):
        _I.new("RGB", (64, 64), (0, 255, 0)).save(os.path.join(fdir, f"{i:04d}.png"))
    frames, src = load_frames(str(tmp_path), "run.gif", size=32)
    assert src == "png"
    assert frames[:, 1].mean() > 0.8 and frames[:, 0].mean() < 0.2


def test_load_frames_falls_back_to_gif(tmp_path):
    import os
    gif = os.path.join(tmp_path, "solo.gif")
    _write_gif(gif, 64, 64, (0, 0, 255), (0, 0, 255))
    frames, src = load_frames(str(tmp_path), "solo.gif", size=32)
    assert src == "gif" and frames[:, 2].mean() > 0.8
