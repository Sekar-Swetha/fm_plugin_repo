"""C3 Stage 2: consistency-distill the student on cached teacher-trajectory pairs.

Only the student is in memory (no teacher UNet). The teacher's one-solver-step
target is precomputed in the cached x_lo. The student unfreezes the same subset
as train_adaptor_fm.py; an EMA copy provides the consistency target.

`consistency_train_step` is a pure per-pair step (CPU-testable with stub eps
functions). `main()` wires the real MotionEditor UNet/ControlNet and needs a GPU.
"""
import argparse
import os
import sys

import torch

from consistency_core import (
    ConsistencyPair,  # noqa: F401  (re-exported for callers/tests)
    ema_update,
    load_pairs,
    predict_x0_from_eps,
    pseudo_huber_loss,
)

TRAINABLE = ("attn1.to_q", "attn2.to_q", "attn_temp")


def consistency_train_step(student_eps_fn, ema_eps_fn, pair, alphas, sigmas, delta: float = 1.0):
    """One consistency step. student_eps_fn/ema_eps_fn: (x, t_int) -> eps.

    L = pseudo_huber( f_student(x_hi, t_hi), stopgrad f_ema(x_lo, t_lo) ).
    """
    a_hi, s_hi = alphas[pair.t_hi], sigmas[pair.t_hi]
    a_lo, s_lo = alphas[pair.t_lo], sigmas[pair.t_lo]
    student_x0 = predict_x0_from_eps(pair.x_hi, student_eps_fn(pair.x_hi, pair.t_hi), a_hi, s_hi)
    with torch.no_grad():
        target_x0 = predict_x0_from_eps(pair.x_lo, ema_eps_fn(pair.x_lo, pair.t_lo), a_lo, s_lo)
    return pseudo_huber_loss(student_x0, target_x0, delta=delta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", required=True, help="pairs.pt from generate_consistency_pairs")
    ap.add_argument("--resume-from-checkpoint", required=True,
                    help="epsilon accelerator state dir (e.g. outputs/train-case-1-motion/checkpoint-300)")
    ap.add_argument("--adapter-weight-path", required=True,
                    help="epsilon controlnet_adapter .pth (loaded into unet.controlnet_adapter)")
    ap.add_argument("--pretrained-model-path", default="checkpoints/stable-diffusion-v1-5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-steps", type=int, default=2000)
    ap.add_argument("--ema-decay", type=float, default=0.95)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--prompt", default="a girl is dancing")
    args = ap.parse_args()

    device = "cuda"
    # float32 throughout: the UNet loads in float32 under mixed_precision="no",
    # so inputs, ControlNet, and cond tensors must match (conv2d requires equal
    # input/weight dtype). If this OOMs at 8 frames, switch to fp16 + autocast.
    dtype = torch.float32
    video_length = 8

    # Model loading mirrors inference.py:166,263-266 exactly:
    #   - UNet from BASE SD subfolder, trained weights via accelerator.load_state
    #   - motion adapter lives INSIDE the unet: unet.controlnet_adapter
    #   - ControlNet from the base sd-controlnet-openpose
    from accelerate import Accelerator
    from diffusers import ControlNetModel, DDPMScheduler
    from transformers import CLIPTextModel, CLIPTokenizer
    from motion_editor.models.unet_2d_condition import UNet2DConditionModel

    accelerator = Accelerator(mixed_precision="no")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_path, subfolder="unet",
        use_sc_attn=True, use_st_attn=False, st_attn_idx=0)
    controlnet = ControlNetModel.from_pretrained("checkpoints/sd-controlnet-openpose", torch_dtype=dtype)

    unet = accelerator.prepare(unet)
    accelerator.load_state(args.resume_from_checkpoint)              # trained epsilon UNet
    unet.controlnet_adapter.load_state_dict(torch.load(args.adapter_weight_path))  # motion adapter

    text_encoder.to(accelerator.device, dtype)
    controlnet.to(accelerator.device)

    noise_sched = DDPMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
    ac = noise_sched.alphas_cumprod
    alphas = torch.sqrt(ac)
    sigmas = torch.sqrt(1.0 - ac)

    text_encoder.requires_grad_(False)
    controlnet.requires_grad_(False)
    unet.requires_grad_(False)
    trainable = []
    for name, p in unet.named_parameters():
        if any(k in name for k in TRAINABLE):
            p.requires_grad_(True)
            trainable.append(p)
    print(f"[c3] {len(trainable)} trainable tensors ({sum(p.numel() for p in trainable)} params)")
    ema = [p.detach().clone() for p in trainable]
    opt = torch.optim.AdamW(trainable, lr=args.lr)

    prompt_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(accelerator.device)
    ehs = text_encoder(prompt_ids)[0]

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                    "motionEditor", "MotionEditor", "motion_editor"))
    from consistency_plugin import conditioned_eps

    pairs = load_pairs(args.pairs_dir)
    state = {"cond": None}

    def _eps(x, t):
        return conditioned_eps(
            unet, controlnet, x.to(accelerator.device, dtype), int(t),
            state["cond"].to(accelerator.device, dtype), ehs, video_length).float()

    def student_eps_fn(x, t):
        return _eps(x, t)

    def ema_eps_fn(x, t):
        # temporarily load EMA weights into the trainable params, forward, restore
        backup = [p.detach().clone() for p in trainable]
        for p, e in zip(trainable, ema):
            p.data.copy_(e.data)
        out = _eps(x, t)
        for p, b in zip(trainable, backup):
            p.data.copy_(b.data)
        return out

    step = 0
    while step < args.num_steps:
        for pair in pairs:
            state["cond"] = pair.cond
            loss = consistency_train_step(student_eps_fn, ema_eps_fn, pair,
                                          alphas, sigmas, delta=args.delta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ema_update(ema, trainable, decay=args.ema_decay)
            if step % 50 == 0:
                print(f"[c3] step {step} loss {loss.item():.5f}")
            step += 1
            if step >= args.num_steps:
                break

    os.makedirs(args.out, exist_ok=True)
    accelerator.save_state(args.out)
    torch.save(unet.controlnet_adapter.state_dict(),
               os.path.join(args.out, "controlnet_adapter.pth"))
    print(f"[c3] saved student to {args.out}")


if __name__ == "__main__":
    main()
