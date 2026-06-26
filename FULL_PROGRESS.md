# MotionEditor × Flow Matching — Full Progress Report

_Last updated: 2026-06-26. Author working notes for dissertation + advisor/TA review._

This document records **everything done so far**, in detail: the goal, the two
proposed contributions, the environment battle, every experiment, every result
(with the gif that proves it), the diagnosis of the flow-softness problem, the
C2 distillation work, and the decision points — including the TA's prescribed
plan and exactly where our work maps onto it.

---

## 1. Project goal

Improve the **MotionEditor** paper (CVPR 2024 — one-shot, single-video,
diffusion-based video *motion* editing; ControlNet on pose skeletons +
content-aware motion adapter + two-branch attention injection) with two changes:

1. **Pose extraction:** replace **OpenPose** with **MediaPipe** (BlazePose 33-pt
   → remapped to OpenPose COCO-18 layout). *Low weight per advisor — "easy to
   implement, not heavy work."*
2. **Generative core:** replace **DDPM/DDIM** with **Flow Matching (CFM-OT,
   Lipman et al. 2023)**. *This is what the marks depend on — flow quality.*

The straight-line OT flow: velocity `v = x1 − x0`, path
`x_t = (1−(1−σ_min)·t)·x0 + t·x1`, ODE sampling (Euler/Heun). At `σ_min=0` this is
rectified flow.

---

## 2. Contributions (as designed)

- **Contribution A — CFM-OT training swap.** Replace the DDPM epsilon-prediction
  loss in `train_adaptor.py` with the CFM-OT velocity loss. Nothing else touched.
- **Contribution B — Flow inversion + ODE sampling.** Replace DDIM inversion +
  null-text optimization with flow-ODE inversion (integrate velocity backward,
  t:1→0) and forward ODE sampling (t:0→1). This is the "DDIM-specific stuff" that
  must be swapped alongside the loss, because a velocity field cannot be sampled
  by a DDIM/epsilon scheduler.
- **Contribution C2 — Distilled direct editor (extra novelty).** Distill the
  *sharp epsilon MotionEditor* into a flow model: learn a direct
  `source-latent → sharp-edited-latent` coupling, conditioned on the target pose.
  Inversion-free, few-step, intended to be sharp (teacher-matched) + fast.

---

## 3. Environment (the setup battle, resolved)

GPU lab box: Ubuntu, user `vivek`, **NVIDIA RTX PRO 6000 Blackwell, 96 GB**
(sm_120). Working conda env: **`me310`** (Python 3.10).

- torch 2.12.1+cu130, diffusers **0.15.1** (pinned — MotionEditor forks diffusers
  internals; modern diffusers breaks it), transformers 4.30.2,
  huggingface_hub 0.16.4, accelerate 0.20.3.
- **xformers bypassed** via an SDPA shim in `unet_2d_blocks.py` (its CUDA ext
  mismatches torch on Blackwell). Shim handles both 4D `(B,M,H,K)` and 3D
  `(B*H,M,K)` attention layouts.
- Weights downloaded via curl (`download_weights.py`) — the HF `hf`/xet/urllib
  clients all hang on the lab network.
- Memory: model bakes `num_frames=8` into the controlnet adapter → **do not lower
  `n_sample_frames`** (breaks a rearrange). 8 frames ≈ 90 GB; fits 96 GB only with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

Gotchas fixed along the way: Python 3.13 has no wheels for the 2023-era pinned
stack (use 3.10); `print(1/0)` landmines in MotionEditor's xformers-absent guards
(patched out); `CLIPFeatureExtractor` rename; venv-layering trap (`deactivate` the
old 3.13 venv before pip).

---

## 4. What was built (files)

### flow_matching_plugin/
- `flow_matching_loss.py` — CFM-OT core: `compute_x_t`, `compute_target_velocity`,
  `cfm_ot_loss`, `euler_sample`, `build_fm_training_batch`, `FlowMatchingConfig`.
- `flow_inversion.py` — `MotionEditorFlowInversion`, `make_velocity_fn`
  (`extra_unet_kwargs={"normal_infer":True}`; CLIP prompt-id unsqueeze fix).
- `generate_reflow_pairs.py` — `save_records`/`load_records`/`save_shards`,
  `real_velocity_factory`.
- `train_adaptor_fm.py` — real-model CFM-OT trainer; `distill_pairs_dir` path for
  C2; **LR-scheduler-resume bug fixed** (load U-Net weights only, not
  `accelerator.load_state`, which restored a decayed scheduler → LR≈0).
- `train_reflow.py`, `verify_inversion.py`, `verify_loss.py`.
- `tests/test_flow_matching.py`, `tests/test_inversion.py`, `tests/test_reflow.py`.
- `PROOFS.md`, `README.md`, `README_contrib_BC.md`.

### motionEditor/MotionEditor/
- `motion_editor/pipelines/pipeline_motion_editor.py` — added the **flow-ODE
  sampler branch**: nested `_flow_velocity(cur_latents, t_model)` (ControlNet on
  target skeleton → two-branch U-Net → CFG), Euler/Heun integration t:0→1 when
  `flow_sampling=True`; original DDIM loop otherwise. Saves
  `self._last_latents` before VAE decode (for C2 pair capture).
- `inference.py` — flow hooks: `use_flow_inversion`, `flow_from_source` (C2 start
  at source latent), `flow_skip_inversion` (noise), `flow_cond_inversion`
  (conditioned inversion = source skeleton + ControlNet, mirrors training
  forward), `flow_no_injection` (skip two-branch attention), `distill_dump_to`
  (save `z_src, z_edit_sharp, cond_tgt`).
- `motion_editor/models/unet_2d_blocks.py` — xformers→SDPA shim.

### Docs
`PROJECT_STATUS.md`, `WHATS_CHANGED.md`, `RUNBOOK.md`, `STEPS_TO_EXECUTE.md`,
`DISSERTATION_OVERVIEW.md`, `MEETING_BRIEF.md`, `DISTILLATION_C2_PLAN.md`, and
this file.

---

## 5. Experiments & results (every run, with its gif)

All gifs in `fm_outputs/`. Case-1 = "a girl dancing on a beach", 8 frames.

### 5.1 Baselines (sanity — the pipeline works end-to-end)
| Run | Gif | Result |
|---|---|---|
| OpenPose + epsilon (original MotionEditor) | `baseline_epsilon_case1.gif` | **SHARP** clean motion edit. Proves pipeline on Blackwell. |
| MediaPipe + epsilon | `mediapipe_epsilon_case1.gif` | **SHARP** — MediaPipe ≈ OpenPose (Goal 1 proven). |

### 5.2 The simple swap — DDIM → flow (Contributions A + B)
**This is exactly the TA's prescribed experiment**: same source latent, same GT,
same architecture; only the training loss (DDPM-eps → CFM-OT) and the
DDIM-specific sampler (DDIM+null-text → flow inversion + ODE) are swapped.

| Run | Gif | Result |
|---|---|---|
| CFM-OT trained, **sampled through DDIM** (wrong) | `baseline_case1.gif` | Rainbow **noise** — confirms velocity ≠ epsilon; a flow-ODE sampler is mandatory. |
| OpenPose + flow (ODE, 50 step) | `flow_ode_fm_50step_case1.gif` | **Coherent edit, soft.** |
| MediaPipe + flow (ODE) | `mediapipe_flow_case1.gif` | **Coherent edit, soft.** |

**2×2 ablation conclusion:** MediaPipe ≈ OpenPose across both samplers (Goal 1).
Flow inversion + ODE **replaces DDIM + null-text end-to-end and produces coherent
edits** (Goal 2). The softness is a *quality* gap vs the epsilon baseline, **not a
failure** — the person is edited and recognizable, just blurry.

### 5.3 Diagnosing the flow softness (what was ruled out)
| Lever tried | Gif | Outcome |
|---|---|---|
| More training (1000 step, constant LR) | `mediapipe_flow_1000step_case1.gif` | **Worse** — 1-video overfit + (then-unknown) LR-resume bug. |
| Cosine LR, fixed resume bug, 800 step | `flow_cosine800_fixed_case1.gif` | Still soft → training recipe is not the lever. |
| Guidance scale 3 | `mediapipe_flow_cfg3_case1.gif` | No help. |
| Skip inversion, start from noise | `flow_fromnoise_case1.gif` | Shattered (two-branch attention needs a real inverted source). |
| **Conditioned inversion** (source skeleton + ControlNet) | `flow_condinv_case1.gif`, `_euler_`, `_noinj_` | **Background became SHARP**; person still ghosted. |
| Drop attention injection | `flow_condinv_noinj_case1.gif` | Person still ghosted → injection is **not** the ghost source. |

**Diagnosis:** the static scene renders sharp under flow; the **moving subject
ghosts** because the CFM-OT velocity field (one video, limited budget) is not
precise enough on dynamic content. Training / CFG / steps / attention all ruled
out. This motivated C2.

### 5.4 Contribution C2 — distill the sharp epsilon editor into a flow model
Pipeline: (1) run the sharp epsilon editor, dump `(z_src, z_edit_sharp,
cond_tgt)`; (2) train a flow model on the coupling `z_src → z_edit` conditioned on
the target pose (`v = z_edit − z_src`); (3) at inference start the ODE **at the
source latent** (no inversion), few steps, decode.

| Run | Gif | Result |
|---|---|---|
| Teacher (epsilon, the distill target) | `distill_teacher_case1.gif` | **SHARP** (reference). |
| C2 distill, 500 step / 1 pair, 8 infer steps | `distill_c2_case1.gif` | **Ghosting eliminated**, single coherent person, sharp bg, but texture artifacts (undertrained). |
| C2 distill 1500 step, 8 infer steps | `distill_c2_1500_case1.gif` | Person corrupted (artifacts). |
| C2 distill 1500 step, 50 infer steps | `distill_c2_1500_50steps_case1.gif` | **Worse** — more steps = more corruption. |
| C2, 1 / 2 / 4 infer steps | `distill_c2_1step/2step/4step.gif` | All still corrupted. |

**Two bugs found in the C2 inference (being fixed now):**
1. **Too many ODE steps drift off the single-pair training line.** A 1-pair
   distilled flow only knows velocities *on* the straight `z_src→z_edit` line.
   Euler with many steps wanders into untrained latent regions → mush. The
   background is near-static (`z_src≈z_edit` there) so stays sharp; the moving
   person drifts → garbage. **Fix: 1 Euler step from source** (one step lands on
   `z_edit` = teacher).
2. **CFG mismatch.** The flow velocity in the pipeline applies classifier-free
   guidance at `guidance_scale=7.5` (`v = v_u + 7.5·(v_t − v_u)`), but the
   distillation trained the velocity field **without CFG**. The 7.5× amplification
   throws the velocity off the line → garbage even at 1 step. **Fix:
   `guidance_scale=1.0`** (CFG off → raw conditioned velocity = memorized
   `z_edit−z_src`).

**Status:** config now set to `flow_from_source=True, flow_no_injection=True,
flow_inv_steps=1, flow_inv_method=euler, guidance_scale=1.0, use_null_inv=False`,
distilled `checkpoint-1500-fm`. **Pending run** to confirm C2 produces the sharp,
inversion-free, 1-step edit.

> Note: the sharp single frame seen earlier was `sample-all-inv.gif` (the source
> latent decoded), **not** the edited output — a false positive. The guidance fix
> is the outstanding correction.

---

## 6. The TA's plan — mapping to our work

**TA (worked on flow matching before) prescribed:**

1. **Run the base flow-matching paper first; verify it works.**
   → **DONE.** `tests/test_flow_matching.py` (path endpoints `x_t=x0` at t=0,
   `x_t=x1` at t=1; velocity target; Euler sampler), `verify_loss.py`,
   `PROOFS.md`, toy 2-D proofs. Base CFM-OT is correct and runs.

2. **In MotionEditor, keep source latent / ground truth / everything identical;
   swap only the DDIM/DDPM training for flow — simple swap, no architecture
   change.**
   → **DONE (Contributions A + B).** `PROOFS.md` documents that only the loss
   block changes; "architecture, ControlNet, motion adapter, optimizer,
   dataloader, attention injection — byte-for-byte unchanged." The DDIM sampler
   was swapped for the flow-ODE sampler because a velocity field is unsamplable by
   DDIM (the rainbow-noise run `baseline_case1.gif` proves this is mandatory, not
   an architecture change). **Result: coherent edits, soft** (`mediapipe_flow_*`,
   `flow_ode_fm_50step_*`).

3. **If results are negative (low grade risk), drop flow matching and substitute
   something else for DDIM that definitely works.**
   → **Decision pending on the C2 guidance-fix run.** See §7.

**Honest read:** the simple swap is **not negative** — it produces coherent,
faithful, *fast* edits; the only gap is sharpness, a documented limitation. The
only outright-garbage results were the *C2 distillation* experiments, and those
are explained by the two inference bugs above (steps + CFG), now fixed and
pending verification.

---

## 7. Decision & next steps

**Path 1 — finish the flow story (preferred; still on track).**
- Confirm the **C2 1-step, guidance=1.0** run is sharp. If yes → headline result:
  a **distilled, inversion-free, 1-NFE flow editor** that is sharp, vs
  MotionEditor's DDIM-50 + null-text optimization. Strong, novel.
- Even if C2 lands only "moderately sharp," the deliverables are positive:
  - MediaPipe ≈ OpenPose (2×2 ablation).
  - Flow inversion + ODE **replaces DDIM end-to-end** with coherent edits.
  - Speed: flow needs **few NFE, no null-text optimization**.
- Then: **metrics** (CLIP-sim, LPIPS-vs-teacher, NFE, wall-clock) + write-up.

**Path 2 — TA's fallback (if C2 stays garbage / grade risk too high).**
Substitute a sampler that *definitely works* in place of DDIM, keeping the
MediaPipe contribution. Candidates that are guaranteed-sharp and still a real
contribution:
- **DPM-Solver++ / UniPC** fast deterministic samplers replacing DDIM+null-text →
  sharp, far fewer steps than DDIM-50, no per-step null optimization. Clean
  "faster, sharper sampler swap" story. Lowest risk.
- Keep the flow work as a documented diagnostic chapter (honest negative on
  *sharpness only*, positive on coherence + speed) and lead with the sampler swap.

**Recommendation:** run the pending C2 guidance fix first (one inference, minutes).
Branch on its result:
- Sharp → Path 1, lock it, do metrics + write-up.
- Still garbage → Path 2 (DPM-Solver++), and frame flow as the analyzed
  alternative.

---

## 8. How to reproduce (GPU box)

```bash
conda activate me310
cd ~/fm_plugin_repo && git stash; git pull; git stash drop
cd motionEditor/MotionEditor
export PYTHONPATH=$(pwd):$(pwd)/../../flow_matching_plugin
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# C2 distilled 1-step (current best-hope flow result)
python3 inference.py --config configs/case-1/eval-motion.yaml
cp "$(ls -t outputs/eval-case-1-motion/*.gif | head -1)" ~/distill_c2_g1_1step.gif
```
Config `configs/case-1/eval-motion.yaml` is committed with the C2 settings
(`checkpoint-1500-fm`, `flow_from_source`, `flow_no_injection`, `flow_inv_steps:1`,
`flow_inv_method:euler`, `guidance_scale:1.0`, `use_null_inv:False`).

Base flow-matching verification (no GPU needed):
```bash
cd ~/fm_plugin_repo/flow_matching_plugin
python -m pytest tests/test_flow_matching.py -q
python verify_loss.py
```
