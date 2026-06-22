"""Reflow pair generation (Contribution C).

Produces the (Z_0, Z_1, cond) coupling that `train_reflow.py` consumes.

Modes:
  --mode c1   noise -> data pairs (generation reflow, Liu et al. 2022).
  --mode c2   src-pose latent -> tgt-pose latent pairs (transport reflow).

The pair-generation logic (`generate_pairs`) is import-friendly and model
agnostic: it takes a `velocity_fn_factory(cond) -> velocity_fn` so the same
code serves the real MotionEditor U-Net, the CPU demo stub, and the tests.
The CLI wraps it with checkpoint loading + sharded saving.

Reference: `README_contrib_BC.md` Section C.2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from flow_matching_loss import FlowMatchingConfig
from flow_inversion import flow_invert, flow_sample, make_velocity_fn

Pair = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # (z0, z1, cond)
VelocityFnFactory = Callable[[torch.Tensor], Callable]


@dataclass
class ReflowRecord:
    """One source clip's ingredients for pair generation."""
    z_src: torch.Tensor      # VAE latent of the source video (under source pose)
    cond_src: torch.Tensor   # source skeleton conditioning
    cond_tgt: torch.Tensor   # target skeleton conditioning


def generate_pairs(
    records: Sequence[ReflowRecord],
    velocity_fn_factory: VelocityFnFactory,
    mode: str,
    num_steps: int = 8,
    method: str = "heun",
    config: Optional[FlowMatchingConfig] = None,
    generator: Optional[torch.Generator] = None,
) -> List[Pair]:
    """Build reflow pairs. `cond` is saved as the *skeleton tensor* (cond_tgt),
    not post-ControlNet residuals (those depend on x_t and are recomputed each
    reflow step)."""
    if mode not in ("c1", "c2"):
        raise ValueError(f"mode must be 'c1' or 'c2', got {mode!r}")
    cfg = config or FlowMatchingConfig()
    pairs: List[Pair] = []

    for rec in records:
        if mode == "c1":
            if generator is None:
                z0 = torch.randn_like(rec.z_src)
            else:
                z0 = torch.randn(rec.z_src.shape, generator=generator,
                                 dtype=rec.z_src.dtype, device=rec.z_src.device)
            vf_tgt = velocity_fn_factory(rec.cond_tgt)
            z1 = flow_sample(vf_tgt, z0, num_steps, method, cfg).latent
            pairs.append((z0.detach(), z1.detach(), rec.cond_tgt.detach()))
        else:  # c2: source latent -> edited latent, conditioned on target pose
            vf_src = velocity_fn_factory(rec.cond_src)
            z0_noise = flow_invert(vf_src, rec.z_src, num_steps, method, cfg).latent
            vf_tgt = velocity_fn_factory(rec.cond_tgt)
            z1_edit = flow_sample(vf_tgt, z0_noise, num_steps, method, cfg).latent
            pairs.append((rec.z_src.detach(), z1_edit.detach(), rec.cond_tgt.detach()))

    return pairs


def save_shards(
    pairs: List[Pair],
    out_dir: str,
    meta: Dict,
    shard_max_bytes: int = 2 * 1024 ** 3,
) -> List[str]:
    """Write pairs to `out_dir` as `.pt` shards (<= shard_max_bytes each) with a
    sidecar JSON per shard recording seed + generating-checkpoint hash."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    written: List[str] = []
    shard: List[Pair] = []
    shard_bytes = 0
    shard_idx = 0

    def _flush():
        nonlocal shard, shard_bytes, shard_idx
        if not shard:
            return
        z0 = torch.stack([p[0] for p in shard])
        z1 = torch.stack([p[1] for p in shard])
        cond = torch.stack([p[2] for p in shard])
        path = out / f"pairs-{shard_idx:04d}.pt"
        torch.save({"z0": z0, "z1": z1, "cond": cond}, path)
        sidecar = {**meta, "shard_index": shard_idx, "num_pairs": len(shard)}
        (out / f"pairs-{shard_idx:04d}.json").write_text(json.dumps(sidecar, indent=2))
        written.append(str(path))
        shard, shard_bytes = [], 0
        shard_idx += 1

    for p in pairs:
        pbytes = sum(t.element_size() * t.nelement() for t in p)
        if shard and shard_bytes + pbytes > shard_max_bytes:
            _flush()
        shard.append(p)
        shard_bytes += pbytes
    _flush()
    return written


def save_records(records: Sequence[ReflowRecord], path: str) -> str:
    """Persist a list of ReflowRecord (VAE-encoded clip + skeleton conds) to a
    single `.pt`. Lets the heavy VAE-encode step run once (e.g. inside the
    MotionEditor env) and feed `generate_pairs` later on any machine."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "z_src": [r.z_src for r in records],
        "cond_src": [r.cond_src for r in records],
        "cond_tgt": [r.cond_tgt for r in records],
    }
    torch.save(payload, path)
    return path


def load_records(path: str) -> List[ReflowRecord]:
    """Inverse of `save_records`."""
    d = torch.load(path, map_location="cpu", weights_only=True)
    return [
        ReflowRecord(z_src=z, cond_src=cs, cond_tgt=ct)
        for z, cs, ct in zip(d["z_src"], d["cond_src"], d["cond_tgt"])
    ]


def real_velocity_factory(
    unet,
    *,
    controlnet=None,
    text_encoder=None,
    prompt_ids=None,
    prepare_image_fn=None,
    extra_unet_kwargs=None,
) -> VelocityFnFactory:
    """Build the `cond -> velocity_fn` factory `generate_pairs` expects from a
    real (or stub) MotionEditor U-Net, reusing `flow_inversion.make_velocity_fn`
    so pair generation and inversion share one forward path.

    `cond` is the skeleton conditioning for that clip; each call re-binds the
    closure to the right condition (source for the C2 inversion leg, target for
    the sampling leg)."""

    def factory(cond: torch.Tensor):
        return make_velocity_fn(
            unet,
            cond,
            controlnet=controlnet,
            text_encoder=text_encoder,
            prompt_ids=prompt_ids,
            prepare_image_fn=prepare_image_fn,
            extra_unet_kwargs=extra_unet_kwargs,
        )

    return factory


def checkpoint_hash(path: Optional[str]) -> str:
    if path is None or not os.path.exists(path):
        return "none"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ----- CLI -----------------------------------------------------------------

def _stub_velocity_factory(cond_dim: int, latent_ch: int, seed: int = 0):
    """A tiny deterministic velocity model for CPU dev runs (no checkpoint)."""
    torch.manual_seed(seed)
    proj = torch.nn.Linear(cond_dim, latent_ch)

    def factory(cond: torch.Tensor):
        bias = proj(cond.flatten().float()[:cond_dim])

        def velocity_fn(x_t, t_idx):
            b = bias.view(1, -1, *([1] * (x_t.dim() - 2)))
            return torch.tanh(x_t) * 0.5 + b * 0.1

        return velocity_fn

    return factory


def load_motioneditor_velocity_factory(checkpoint: str):
    """Real path: build a velocity factory from a MotionEditor CFM-OT U-Net
    checkpoint. Imports are deferred so the rest of this module runs without the
    MotionEditor environment installed."""
    try:
        import torch as _torch  # noqa: F401
        # MotionEditor's U-Net lives in its own package; import lazily.
        from motion_editor.models.unet_2d_condition import UNet2DConditionModel
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "Real pair generation needs the MotionEditor package importable "
            f"(motion_editor.models...). Import failed: {exc}\n"
            "Run from the MotionEditor repo env, or use --records + --stub-model "
            "to feed pre-encoded latents through a placeholder velocity field."
        )
    unet = UNet2DConditionModel.from_pretrained(checkpoint)
    unet.eval()
    return real_velocity_factory(unet)


def main():
    ap = argparse.ArgumentParser(description="Generate reflow pairs (Contribution C).")
    ap.add_argument("--config", type=str, default=None, help="train-motion-reflow.yaml")
    ap.add_argument("--mode", choices=["c1", "c2"], default="c1")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--num-steps", type=int, default=8)
    ap.add_argument("--method", choices=["euler", "heun"], default="heun")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reflow-round", type=int, default=1)
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="MotionEditor CFM-OT U-Net checkpoint (real mode).")
    ap.add_argument("--records", type=str, default=None,
                    help="Pre-encoded ReflowRecord file (from save_records). Real "
                         "mode: VAE-encode clips once in the MotionEditor env, save, "
                         "then generate pairs anywhere.")
    ap.add_argument("--stub", action="store_true",
                    help="Use a tiny CPU velocity stub instead of a real checkpoint.")
    ap.add_argument("--stub-model", action="store_true",
                    help="With --records: use the CPU velocity stub instead of a "
                         "real checkpoint (dev pass on real latents).")
    ap.add_argument("--num-clips", type=int, default=4, help="(stub mode) synthetic clips.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = FlowMatchingConfig()

    # --- Resolve records ---------------------------------------------------
    if args.records is not None:
        records = load_records(args.records)
        latent_ch = records[0].z_src.shape[0]
        cond_dim = records[0].cond_tgt.numel()
    elif args.stub:
        latent_ch, F, H, W = 4, 4, 8, 8
        cond_dim = 64
        records = [ReflowRecord(
            z_src=torch.randn(latent_ch, F, H, W),
            cond_src=torch.randn(cond_dim),
            cond_tgt=torch.randn(cond_dim),
        ) for _ in range(args.num_clips)]
    else:
        raise SystemExit(
            "Provide --records (real pre-encoded latents) or --stub (synthetic). "
            "See README_contrib_BC.md C.2 / C.6."
        )

    # --- Resolve velocity factory -----------------------------------------
    if args.stub or args.stub_model:
        factory = _stub_velocity_factory(cond_dim, latent_ch, seed=args.seed)
    elif args.checkpoint is not None:
        factory = load_motioneditor_velocity_factory(args.checkpoint)
    else:
        raise SystemExit(
            "Real mode needs --checkpoint (a MotionEditor CFM-OT U-Net) or "
            "--stub-model (placeholder velocity on real latents)."
        )

    pairs = generate_pairs(records, factory, args.mode, args.num_steps, args.method, cfg)
    meta = {
        "mode": args.mode,
        "seed": args.seed,
        "num_steps": args.num_steps,
        "method": args.method,
        "reflow_round": args.reflow_round,
        "checkpoint_hash": checkpoint_hash(args.checkpoint),
        "loss_type": "cfm_ot",
    }
    written = save_shards(pairs, args.out, meta)
    print(f"Wrote {len(pairs)} pairs across {len(written)} shard(s) to {args.out}/")


__all__ = [
    "ReflowRecord",
    "generate_pairs",
    "save_shards",
    "save_records",
    "load_records",
    "real_velocity_factory",
    "checkpoint_hash",
]


if __name__ == "__main__":
    main()
