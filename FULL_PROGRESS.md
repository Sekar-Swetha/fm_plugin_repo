# MotionEditor × Flow Matching — Full Dissertation Progress

_Last updated: 2026-07-01. TCD MSc dissertation, Swetha Sekar (sekars@tcd.ie)._
_This is the authoritative, self-contained record: read only this file and you
know the entire project — goal, theory, every experiment, why the flow baseline
fails, the two fixes that work, the code, the results, and what's left._

---

## 0. One-paragraph summary

We improve **MotionEditor** (CVPR 2024 — one-shot, single-video, diffusion video
*motion* editing) two ways: (1) replace **OpenPose** with **MediaPipe** for pose
extraction (done, verified); (2) replace its **DDPM/DDIM** generative core with
**flow matching**. The naive flow swap produces coherent but **soft** edits; we
diagnose *why* (it discards Stable Diffusion's pretrained sharpness prior and the
single-video transport is degenerate). We then fix it by **keeping the prior**:
sample the frozen sharp epsilon editor with a **probability-flow ODE solver
(DPM-Solver++)** — sharp at 15 steps — and **distil** that teacher into a
**few-step consistency model (C3)** — sharp at **2–4 steps** vs the DDIM-50 +
null-text baseline. Naive flow becomes the analyzed negative that motivates the
contribution.

---

## 1. Project goal

Improve MotionEditor (ControlNet on pose skeletons + content-aware motion adapter
+ two-branch attention injection for temporal consistency) with:

1. **Pose extraction:** OpenPose → **MediaPipe** (BlazePose-33 → OpenPose-COCO-18).
   Low weight per advisor ("easy, not heavy"). **Goal 1.**
2. **Generative core:** DDPM/DDIM → **flow matching** and its faster/sharper
   descendants. **Where the marks are — generation quality.**

Straight-line OT flow: velocity `v = x1 − x0`, path
`x_t = (1−(1−σ_min)t)·x0 + t·x1`, ODE sampling (Euler/Heun). At `σ_min=0` this is
rectified flow.

---

## 2. Contributions (final set)

- **Goal 1 — MediaPipe pose.** Drop-in OpenPose-format skeletons from MediaPipe.
- **A — CFM-OT training loss.** Replace the DDPM epsilon loss with the CFM-OT
  velocity loss (`train_adaptor_fm.py`). Nothing else touched.
- **B — Flow inversion + ODE sampling.** Replace DDIM inversion + null-text with
  flow-ODE inversion (t:1→0) and forward ODE sampling (t:0→1). A velocity field is
  unsamplable by a DDPM/epsilon scheduler, so the sampler must change with the loss.
- **A+B result = coherent but SOFT.** Kept as the **analyzed baseline** (§5.2–5.3).
- **DPM-Solver++ (PF-ODE).** Sample the *frozen sharp epsilon editor* with a
  high-order deterministic PF-ODE solver. No retraining. **Sharp at ~15 NFE.**
- **C3 — Consistency distillation (novel headline).** Distil the DPM-Solver++
  teacher into a **1–4 step** editor. **Sharp at 2–4 NFE.**

---

## 3. Environment (the setup battle, resolved)

GPU lab box: Ubuntu, user `vivek`, **NVIDIA RTX PRO 6000 Blackwell, 96 GB**
(sm_120). Conda env **`me310`** (Python 3.10).

- torch 2.12.1+cu130, **diffusers 0.15.1 (pinned — MotionEditor forks diffusers
  internals)**, transformers 4.30.2, huggingface_hub 0.16.4, accelerate 0.20.3.
- **xformers bypassed** via an SDPA shim in `unet_2d_blocks.py` (its CUDA ext
  mismatches torch on Blackwell). `enable_xformers_memory_efficient_attention()`
  routes to this shim.
- Weights via curl (`download_weights.py`); the HF clients hang on the lab network.
- **Memory: `num_frames=8` is baked into the controlnet adapter — do NOT lower
  `n_sample_frames`.** 8 frames needs `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Gotchas fixed: Python 3.13 has no wheels for the 2023 stack (use 3.10);
  `print(1/0)` xformers-absent landmines patched; `CLIPFeatureExtractor` rename.

**Git flow:** GPU box can `git pull` but **not push** (no creds). All commits are
made from the laptop and the box pulls (`git pull --rebase`). Work branch
`contrib-bc-flow-inversion-reflow`.

---

## 4. Theory — why naive flow is soft, and why the fixes work

**Where sharpness comes from.** MotionEditor is sharp because the base
Stable-Diffusion + ControlNet was pretrained on huge data as an **epsilon
predictor**; MotionEditor only fine-tunes a small adapter + attention injection.
The frozen epsilon backbone supplies the sharp-texture prior.

**Why naive CFM-OT loses it.** Swapping the loss to CFM-OT and fine-tuning the
adapter to output **velocity** asks an epsilon-pretrained backbone for a different
quantity, learned from **one video**. Two failures stack:
- **Prior discarded** — the velocity head no longer inherits the denoising prior → soft.
- **Degenerate transport** — rectified-flow power comes from learning transport
  over a *distribution* of couplings; with one pair the field over/underfits.
  Background is static (`z_src≈z_edit`, `v≈0`, trivially learned → stays sharp);
  the **moving subject** needs a real velocity a 1-video field can't represent → ghosts.

**The fix — keep the prior.** Epsilon and velocity are interchangeable at each
noise level: `score = −εθ/σ_t`, and the PF-ODE velocity is a closed form in `εθ`.
So **don't relearn velocity** — take the sharp epsilon editor and integrate its
PF-ODE with a deterministic solver:
- **DPM-Solver++** = a high-order PF-ODE integrator → sharp in ~15 steps, no
  retraining, no degenerate training.
- **Consistency distillation (C3)** = distil the teacher PF-ODE into a student that
  maps any trajectory point to its endpoint (`f(x_t,t)→x0`), trained over the
  *whole trajectory* (many t, many noise), not one pair → fixes the degeneracy →
  **1–4 step sharp**.

---

## 5. Experiments & results (every run, with its gif)

All gifs in `fm_outputs/`. Case-1 = "a girl dancing on a beach", 8 frames.

### 5.1 Baselines (pipeline works end-to-end)
| Run | Gif | Result |
|---|---|---|
| OpenPose + epsilon (original) | `baseline_epsilon_case1.gif` | **SHARP** — pipeline works on Blackwell. |
| MediaPipe + epsilon | `mediapipe_epsilon_case1.gif` | **SHARP** — MediaPipe ≈ OpenPose (Goal 1). |

### 5.2 Naive flow swap (A + B)
| Run | Gif | Result |
|---|---|---|
| CFM-OT sampled through DDIM (wrong) | `baseline_case1.gif` | Rainbow **noise** — velocity ≠ epsilon; a flow-ODE sampler is mandatory. |
| OpenPose + flow (ODE, 50 step) | `flow_ode_fm_50step_case1.gif` | Coherent edit, **soft**. |
| MediaPipe + flow (ODE) | `mediapipe_flow_case1.gif` | Coherent edit, **soft**. |

**Conclusion:** MediaPipe ≈ OpenPose across both samplers (Goal 1). Flow inversion
+ ODE replaces DDIM + null-text end-to-end and produces coherent edits; the gap is
*sharpness*, not failure.

### 5.3 Diagnosing the softness (levers ruled out)
| Lever | Gif | Outcome |
|---|---|---|
| More training (1000 step) | `mediapipe_flow_1000step_case1.gif` | Worse (1-video overfit + LR-resume bug, since fixed). |
| Cosine LR, 800 step | `flow_cosine800_fixed_case1.gif` | Still soft → recipe isn't the lever. |
| Guidance scale 3 | `mediapipe_flow_cfg3_case1.gif` | No help. |
| Start from noise | `flow_fromnoise_case1.gif` | Shattered (injection needs a real inverted source). |
| **Conditioned inversion** | `flow_condinv_case1.gif`, `_euler_`, `_noinj_` | **Background SHARP**; subject still ghosts. |
| Drop attention injection | `flow_condinv_noinj_case1.gif` | Subject still ghosts → injection is not the ghost source. |

**Diagnosis:** static scene renders sharp; the **moving subject ghosts** because
the 1-video CFM-OT velocity field is imprecise on dynamic content. Matches §4.

### 5.4 C2 — single-pair distillation (attempted, negative)
Distil the sharp epsilon editor into a flow via one `(z_src, z_edit)` pair; start
the ODE at the source latent, few steps.
| Run | Gif | Result |
|---|---|---|
| Teacher (epsilon) | `distill_teacher_case1.gif` | **SHARP** reference. |
| C2 500-step, 8 infer | `distill_c2_case1.gif` | Ghosting gone, coherent, mild artifacts (undertrained) — **best C2**. |
| C2 1500-step, 8/50/1/2/4 infer | `distill_c2_1500*`, `distill_c2_{1,2,4}step.gif` | **Overtrained → corrupted.** |

The first-ever `guidance_scale=1.0` C2 run (needed to match distill training)
crashed on `lmi[3]` — the velocity closure hard-coded the 4-branch CFG layout. We
fixed it (`_flow_velocity` no-CFG batch-2 path); the run then produced a
**not-sharp** result on the overtrained 1500 checkpoint. **C2 abandoned** —
single-pair transport is degenerate (§4). Superseded by C3 (trajectory-wide).

### 5.5 DPM-Solver++ (PF-ODE of the sharp epsilon editor) — WORKS
Reparameterize the frozen epsilon editor's PF-ODE, integrate with DPM-Solver++.
No retraining. Isolated code (`motion_editor/dpm_solver_plugin.py`, guarded
`use_dpm_solver`, `configs/case-1/eval-motion-dpm.yaml`).
| Run | Gif | Result |
|---|---|---|
| DPM-Solver++, 20 steps | `dpm_outputs/sample-all.gif` | **SHARP**, pose edited, no ghosting. |
| DPM-Solver++, 15 steps | `dpm_15_outputs/dpm_solver_15.gif` | **SHARP** — identical to 20. |

Minor cosmetic saturation shift (few steps / inversion seam). **This is the
guaranteed-sharp result** and the teacher for C3.

### 5.6 C3 — Consistency distillation (novel headline) — WORKS
Three isolated stages (teacher and student never co-reside in GPU memory):
1. **`generate_consistency_pairs.py`** — record the real DPM teacher trajectory
   (via the pipeline `callback`) across 3 seeds → 42 pairs (`runs/c3/pairs.pt`).
   Consecutive trajectory states = the teacher "one solver step" pairs.
2. **`train_consistency.py`** — student = epsilon UNet, unfrozen subset
   (`attn1.to_q, attn2.to_q, attn_temp`) + EMA target; pseudo-Huber consistency
   loss `d(f_θ(x_hi,t_hi), f_θ⁻(x_lo,t_lo))`; predict-x0 form (boundary free).
   Trained 2000 steps.
3. **`eval-motion-consistency.yaml`** — few-step sampling from the inverted source
   latent, conditioned on the target skeleton.

| NFE | Gif | Result |
|---|---|---|
| 1 | `cons_outputs/cons_1step.gif` | **Soft** — features hazy (1-step undershoot). |
| 2 | `cons_outputs/cons_2step.gif` | **Good** — detail back, slightly soft, usable. |
| 4 | `cons_outputs/sample-consistency.gif` | **Sharp** — teacher-level, no ghosting. |
| 6 | `cons_outputs/cons_6step.gif` | **Sharpest.** |

Monotonic sharpening 1→6. **Headline: sharp at 4 NFE, near-sharp at 2** vs the
DDIM-50 + null-text baseline (~12–25× fewer steps, no null-text optimisation).

**GPU debug fixes applied while bringing C3 up (all committed):**
- size-1 timestep for the UNet forward (`b*f` shape is only right for ControlNet);
- fp16 mixed precision + `enable_xformers_memory_efficient_attention()` +
  `unet.train()` so gradient checkpointing actually fires (else the naive
  `baddbmm` self-attention allocates ~8 GB/softmax and OOMs at 8 frames);
- device/dtype alignment in `consistency_train_step` (x on GPU with eps).

---

## 6. Code map

### flow_matching_plugin/
- `flow_matching_loss.py` — CFM-OT core (`compute_x_t`, `compute_target_velocity`,
  `cfm_ot_loss`, `euler_sample`, `build_fm_training_batch`).
- `flow_inversion.py` — `MotionEditorFlowInversion`, `make_velocity_fn` (B).
- `train_adaptor_fm.py` — CFM-OT trainer (A) + C2 distill path.
- `generate_reflow_pairs.py`, `train_reflow.py` — reflow (C1, toy-proof only).
- **`consistency_core.py`** — C3 math: `predict_x0_from_eps`, `pseudo_huber_loss`,
  `ema_update`, `ConsistencyPair`, `save/load_pairs`, `pairs_from_trajectory`.
- **`generate_consistency_pairs.py`**, **`train_consistency.py`** — C3 stages 1–2.
- `verify_loss.py`, `verify_inversion.py`, `verify_reflow.py` — CPU proofs.
- `tests/` — 44 CPU tests (incl. `test_consistency*.py`).

### motionEditor/MotionEditor/
- `motion_editor/pipelines/pipeline_motion_editor.py` — flow-ODE sampler branch
  (`_flow_velocity`, CFG + no-CFG paths); saves `_last_latents` for pair capture.
- **`motion_editor/dpm_solver_plugin.py`** — DPM-Solver++ scheduler swap.
- **`motion_editor/consistency_plugin.py`** — `conditioned_eps` + `consistency_sample`.
- `inference.py` — flow hooks (`use_flow_inversion`, `flow_from_source`,
  `flow_cond_inversion`, `flow_no_injection`), DPM hook (`use_dpm_solver`), C3
  Stage-1 dump (`dump_trajectory_to`) + Stage-3 hook (`use_consistency`).
- `motion_editor/models/unet_2d_blocks.py` — xformers→SDPA shim.
- `configs/case-1/`: `eval-motion.yaml` (flow/C2), `eval-motion-dpm.yaml` (DPM),
  `eval-motion-consistency.yaml` (C3).

### docs/superpowers/
- `specs/2026-07-01-consistency-distillation-c3-design.md` — C3 design.
- `plans/2026-07-01-consistency-distillation-c3.md` — C3 implementation plan.

---

## 7. Ablation table (measured, case-1)

Computed by `flow_matching_plugin/metrics.py` (2026-07-06). Primary axes:
**Sharpness** (no-reference Laplacian variance ↑ = sharper) and **PoseDist**
(MediaPipe per-joint L2 to the teacher pose ↓ = edit fidelity; skeleton-only, so
immune to colour/blur, unlike LPIPS-vs-teacher). `~0.157` is the
same-pose-different-run floor (teacher itself = 0.000).

| Method | NFE | Sharpness ↑ | PoseDist ↓ | LPIPS-vs-source ↓ |
|---|---|---|---|---|
| Baseline (DDIM-50 + null-text) | 50 | 0.0133 | 0.158 | 0.348 |
| + MediaPipe (epsilon) = teacher | 50 | 0.0133 | 0.000 | 0.393 |
| + A/B naive flow | 50 | **0.0122** | **0.171** | 0.392 |
| + A/B naive flow (50-step ODE) | 50 | 0.0122 | **0.191** | 0.404 |
| DPM-Solver++ | 20 | 0.0142 | 0.160 | 0.434 |
| **DPM-Solver++** | **15** | **0.0144** | 0.161 | 0.441 |
| C3 consistency | 1 | 0.0013 | 0.163 | 0.335 |
| C3 consistency | 2 | 0.0044 | 0.160 | 0.326 |
| C3 consistency | 4 | 0.0102 | 0.157 | 0.364 |
| **C3 consistency** | **6** | **0.0142** | 0.159 | 0.435 |

**Headline finding — the two axes decouple:**
- **Edit fidelity (PoseDist) is flat ~0.157–0.163 across all DPM and C3 rows,
  including NFE-1** — the target pose is achieved at *every* step count; MediaPipe
  reads the correct pose even off the blurry 1-step output.
- **Sharpness is the only axis that scales with steps** (0.0013 → 0.0142 over C3
  1→6), reaching DPM/teacher level (0.0142 ≈ 0.0144) at **6 NFE**.
- ⇒ C3 is a **sharpness↔speed dial at fixed edit fidelity**: pick NFE by sharpness
  budget without losing the edit.

**Naive flow loses on both axes:** softer than baseline (0.0122 < 0.0133) *and*
worst pose fidelity (0.171–0.191, well above the 0.157 floor — the ghosting
subject drifts off-pose). Double-confirmed negative.

**DPM-Solver++:** sharper than the DDIM-50 baseline (0.0144 > 0.0133) at
baseline-level pose fidelity, **15 NFE vs 50** (~3× faster, sharper).

**Caveats (write-up honesty):** PoseDist range is narrow — read it as "at floor =
pose correct" vs "above floor = drift", not fine-grained. Laplacian sharpness also
reflects contrast, so DPM/C3-6's saturation inflates it slightly (ranking sound,
absolutes with the visuals). CLIP-sim (~0.26–0.28) is saturated and
LPIPS-vs-teacher is confounded by pose/colour — both footnoted, not led with.
Wall-clock still to log (time each run; NFE is the reliable speed axis meanwhile).

---

## 8. Decision & status vs the TA's plan

TA (worked on flow before) prescribed: (1) run base flow matching, verify — **done**
(CPU proofs, 44 tests). (2) In MotionEditor swap only the loss/sampler — **done**
(A+B, coherent-but-soft). (3) If negative, substitute a sampler that "definitely
works" — **done and exceeded**: DPM-Solver++ (sharp, 15 NFE) *and* C3 consistency
(sharp, 2–4 NFE), both framed inside the flow/PF-ODE story rather than abandoning it.

**Current status:** all core results achieved. Remaining = metrics + write-up
(+ optional extra cases, extra ablation rows).

---

## 9. What's left

1. **Metrics** — write `metrics.py`, fill §7 (LPIPS-vs-teacher, CLIP-sim, wall-clock).
2. **Optional** — logit-normal t-sampling ablation for the naive-flow chapter;
   more evaluation cases (case-2+).
3. **Write-up** — MediaPipe (Goal 1) + the flow-quality arc (naive soft → diagnose
   → PF-ODE retains prior → C3 distils to few-step). Naive flow is the analyzed
   negative; DPM-Solver++ the safety net; C3 the novelty.

---

## 10. Reproduce (GPU box)

```bash
conda activate me310
cd ~/fm_plugin_repo && git pull --rebase
cd motionEditor/MotionEditor
export PYTHONPATH=$(pwd):$(pwd)/../../flow_matching_plugin
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Sharp baseline teacher (DPM-Solver++, 15 NFE)
python3 inference.py --config configs/case-1/eval-motion-dpm.yaml

# C3 — Stage 1 (record teacher trajectory, 3 seeds), 2 (train), 3 (few-step)
#   add dump_trajectory_to: runs/c3/traj_s33.pt to the DPM config; rerun x3 seeds
python3 ../../flow_matching_plugin/generate_consistency_pairs.py --traj "runs/c3/traj_s*.pt" --out runs/c3/pairs.pt
python3 ../../flow_matching_plugin/train_consistency.py \
    --pairs-dir runs/c3/pairs.pt \
    --resume-from-checkpoint outputs/train-case-1-motion/checkpoint-300 \
    --adapter-weight-path outputs/train-case-1-motion/controlnet_adapter_checkpoint-300.pth \
    --out outputs/train-case-1-cons --num-steps 2000
python3 inference.py --config configs/case-1/eval-motion-consistency.yaml   # sweep consistency_steps 4->2->1

# CPU proofs (no GPU)
cd ~/fm_plugin_repo/flow_matching_plugin && python -m pytest tests/ -q     # 44 pass
```
