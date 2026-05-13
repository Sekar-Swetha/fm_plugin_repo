"""MotionEditor training with CFM-OT loss instead of DDPM epsilon-prediction.

Drop-in for MotionEditor's `train_adaptor.py`; the only structural change is
the inner loss block. Model, accelerator, dataloader, optimizer untouched.

Run:
    cd MotionEditor
    accelerate launch ../flow_matching_plugin/train_adaptor_fm.py \
        --config ../flow_matching_plugin/configs/train-motion-fm.yaml
"""

import argparse
import datetime
import logging
import inspect
import math
import os
import sys
import warnings
from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection
from einops import rearrange
from diffusers import ControlNetModel

from motion_editor.models.unet_2d_condition import UNet2DConditionModel
from motion_editor.data.dataset import VideoDataset
from motion_editor.p2p.p2p_stable import AttentionReplace, AttentionRefine
from motion_editor.p2p.ptp_utils import register_attention_control
from motion_editor.pipelines.pipeline_motion_editor import MotionEditorPipeline

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from flow_matching_loss import (
    FlowMatchingConfig,
    build_fm_training_batch,
)

check_min_version("0.10.0.dev0")
logger = get_logger(__name__, log_level="INFO")


def main(
    pretrained_model_path: str,
    output_dir: str,
    input_data: Dict,
    validation_data: Dict,
    input_batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = True,
    mixed_precision: Optional[str] = "fp16",
    enable_xformers_memory_efficient_attention: bool = True,
    seed: Optional[int] = None,
    use_sc_attn: bool = True,
    use_st_attn: bool = True,
    st_attn_idx: int = 0,
    fps: int = 8,
    validation_steps: int = 100,
    trainable_modules: Tuple[str] = (
        "attn1.to_q",
        "attn2.to_q",
        "attn_temp",
    ),
    trainable_params: Tuple[str] = (),
    train_batch_size: int = 1,
    max_train_steps: int = 500,
    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    use_8bit_adam: bool = False,
    resume_from_checkpoint: Optional[str] = None,
    checkpointing_steps: int = 500,
    one_stage_checkpoint: Optional[str] = None,
    loss_type: str = "cfm_ot",
    flow_sigma_min: float = 0.0,
    flow_t_eps: float = 1e-5,
):
    *_, config = inspect.getargvalues(inspect.currentframe())

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if seed is not None:
        set_seed(seed)

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/sample", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_path, subfolder="unet", use_sc_attn=use_sc_attn,
        use_st_attn=use_st_attn, st_attn_idx=st_attn_idx)
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    controlnet = ControlNetModel.from_pretrained("checkpoints/sd-controlnet-openpose", torch_dtype=torch.float16)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(False)

    for name, module in unet.named_modules():
        if "controlnet_adapter" in name:
            for params in module.parameters():
                params.requires_grad = True

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    if use_8bit_adam:
        import bitsandbytes as bnb
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    input_dataset = VideoDataset(**input_data)
    input_dataset.prompt_ids = tokenizer(
        input_dataset.prompt, max_length=tokenizer.model_max_length,
        padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids[0]

    input_dataloader = torch.utils.data.DataLoader(
        input_dataset, batch_size=input_batch_size
    )

    validation_pipeline = MotionEditorPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler"),
        safety_checker=None,
        controlnet=controlnet,
    )
    validation_pipeline.enable_vae_slicing()
    ddim_inv_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder='scheduler')
    ddim_inv_scheduler.set_timesteps(validation_data.num_inv_steps)

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    unet, optimizer, input_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, input_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)

    num_update_steps_per_epoch = math.ceil(len(input_dataloader) / gradient_accumulation_steps)
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("vid2vid-zero-fm")

    total_batch_size = input_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training (CFM-OT) *****")
    logger.info(f"  Num examples = {len(input_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {input_batch_size}")
    logger.info(f"  Total input batch size = {total_batch_size}")
    logger.info(f"  Loss type = {loss_type}")
    logger.info(f"  Flow sigma_min = {flow_sigma_min}")

    fm_cfg = FlowMatchingConfig(
        sigma_min=float(flow_sigma_min),
        num_train_timesteps=int(noise_scheduler.num_train_timesteps),
        t_eps=float(flow_t_eps),
    )

    global_step = 0
    first_epoch = 0

    if resume_from_checkpoint:
        if resume_from_checkpoint != "latest":
            path = os.path.basename(resume_from_checkpoint)
        else:
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1]
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(output_dir, path))
        global_step = int(path.split("-")[1])
        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = global_step % num_update_steps_per_epoch

    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    if one_stage_checkpoint is not None:
        accelerator.load_state(one_stage_checkpoint)

    initial_params = {}
    for name, param in unet.named_parameters():
        initial_params[name] = param.clone().detach()

    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(input_dataloader):
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(weight_dtype)
                video_length = pixel_values.shape[1]
                pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                latents = latents * 0.18215

                source_skeleton = batch["source_conditions"]["openposefull"].to(weight_dtype)
                encoder_hidden_states = text_encoder(batch["prompt_ids"])[0]

                if loss_type == "cfm_ot":
                    x_1 = latents
                    x_t, t_cont, timesteps, target = build_fm_training_batch(x_1, fm_cfg)
                    noisy_latents = x_t
                elif loss_type == "epsilon":
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    timesteps = torch.randint(
                        0, noise_scheduler.num_train_timesteps, (bsz,),
                        device=latents.device,
                    ).long()
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    if noise_scheduler.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(
                            f"Unknown prediction type {noise_scheduler.prediction_type}"
                        )
                else:
                    raise ValueError(
                        f"Unknown loss_type {loss_type!r}; expected 'cfm_ot' or 'epsilon'"
                    )

                controlnet_batch_size = 1
                num_images_per_prompt = 1
                do_classifier_free_guidance = False
                device = validation_pipeline._execution_device
                images = validation_pipeline.prepare_image(
                    image=source_skeleton,
                    width=input_data.width,
                    height=input_data.height,
                    batch_size=controlnet_batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                )
                images = rearrange(images, "b f c h w -> (b f) c h w").to(
                    device=controlnet.device, dtype=controlnet.dtype
                )
                controlnet_latent_model_input = rearrange(
                    noisy_latents, "b c f h w -> (b f) c h w"
                ).to(dtype=controlnet.dtype)
                down_block_res_samples, mid_block_res_sample = controlnet(
                    controlnet_latent_model_input,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states.repeat(video_length, 1, 1),
                    controlnet_cond=images,
                    conditioning_scale=1.0,
                    return_dict=False,
                )
                down_block_res_samples = [
                    rearrange(s, "(b f) c h w -> b c f h w", f=video_length)
                    for s in down_block_res_samples
                ]
                mid_block_res_sample = rearrange(
                    mid_block_res_sample, "(b f) c h w -> b c f h w", f=video_length
                )

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss, "loss_type": loss_type}, step=global_step)
                train_loss = 0.0

                if global_step % checkpointing_steps == 0 and accelerator.is_main_process:
                    save_path = os.path.join(output_dir, f"checkpoint-{global_step}-fm")
                    accelerator.save_state(save_path)
                    controlnet_adapter_weights = unet.controlnet_adapter.state_dict()
                    torch.save(
                        controlnet_adapter_weights,
                        os.path.join(output_dir, f"controlnet_adapter_checkpoint-{global_step}-fm.pth"),
                    )
                    logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            if global_step >= max_train_steps:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(**OmegaConf.load(args.config))
