# MSc Dissertation — Complete Overview

**Improving MotionEditor with MediaPipe and Flow Matching**

_A single, self-contained explanation of the whole project: what it is, what was built,
everything we tried, the results, why the hard part is hard, where things stand, and what's next.
Written so a complete beginner can follow it — with full technical detail underneath._

_Last updated: 2026-06-23_

---

## 0. The 30-second summary

We took an existing AI system called **MotionEditor** (it changes how a person *moves* in a video
while keeping who they are and the background the same) and tried two upgrades:

1. **Swap the pose detector** — replace the hard-to-install **OpenPose** with the easy,
   pip-installable **MediaPipe**. ✅ **Works.**
2. **Swap the "denoising maths"** — replace the old diffusion method (**DDIM**) with a newer one
   (**Flow Matching**). ✅ **Runs and produces real edits**, but the videos come out **soft/blurry**
   rather than sharp — and we figured out exactly why.

**Bottom line:** both upgrades are implemented and demonstrated on a real GPU. Upgrade 1 is clean.
Upgrade 2 works as a method (proven on simple test data) and produces coherent video edits, but the
*sharpness* of the flow-based video is currently limited — for reasons we diagnosed in detail.

---

## 1. The goal, in plain English

Imagine a video of a girl dancing on a beach. **MotionEditor** lets you say:
*"keep this exact girl and this exact beach, but make her do a **different dance**."*
It outputs a new video: same girl, same beach, new motion.

To do that, MotionEditor needs two things:
- a **stick-figure (skeleton)** of the body in each frame (to know where the arms/legs are), and
- a powerful **image generator** (a "diffusion model") that paints the new frames.

Our project changes **one piece of each**:
- The thing that draws the skeleton: **OpenPose → MediaPipe**.
- The maths the image generator uses to go from noise to a picture: **diffusion/DDIM → flow matching**.

> **Analogy:** MotionEditor is like a puppeteer. The *skeleton* is the puppet's wireframe pose;
> the *diffusion model* is the artist who paints a realistic person onto that wireframe.
> We swapped the tool that builds the wireframe (Goal 1) and the artist's painting technique (Goal 2).

---

## 2. The building blocks (explained simply, then precisely)

### 2.1 MotionEditor (the paper we build on)
- **Simple:** an AI that edits the *motion* of a person in a video.
- **Precise:** a 2024 CVPR diffusion-based video motion-editing model. It fine-tunes on a **single
  source video** (one-shot, like Tune-A-Video), uses a **ControlNet** conditioned on pose skeletons,
  a **content-aware motion adapter**, and a **two-branch attention-injection** scheme (a
  *reconstruction* branch that rebuilds the original + an *editing* branch that makes the change and
  copies appearance details from the reconstruction branch).

### 2.2 Pose skeleton, OpenPose, MediaPipe
- **Simple:** a stick figure showing where the head, shoulders, elbows, knees, etc. are.
- **OpenPose:** the classic, accurate skeleton detector — but **notoriously hard to install**
  (old build chain, doesn't compile on modern machines). This blocked the student's laptop.
- **MediaPipe:** Google's skeleton detector — **`pip install` and done, runs on CPU**. Its skeleton
  format (BlazePose, 33 points) differs from OpenPose (18 points, "COCO"), so we **remap** MediaPipe's
  points to OpenPose's format and **re-draw** them with the same colours, so MotionEditor can't tell
  the difference.

### 2.3 Diffusion models & DDIM
- **Simple:** start with TV static (noise) and repeatedly "clean it up" until a real picture appears.
- **Precise:** a diffusion model is trained to predict the **noise** added to an image. Sampling
  reverses this over ~50 steps. **DDIM** is a specific deterministic sampler. **DDIM inversion** runs
  it *backwards* to find the noise that would regenerate a given image — needed for editing.
- MotionEditor also uses **null-text optimization** to make inversion accurate — but it's **slow**.

### 2.4 Flow matching / CFM-OT (the new maths)
- **Simple:** instead of slowly "cleaning up static," learn a **straight-line path** from noise to
  the image, and just **walk along it**. Straighter path = fewer, faster steps.
- **Precise:** **Conditional Flow Matching with Optimal-Transport paths (CFM-OT)** (Lipman et al. 2023).
  The model learns a **velocity field** `v = x₁ − x₀` along the straight interpolation
  `xₜ = (1−t)·x₀ + t·x₁`. Sampling = integrating an **ODE** (e.g. Euler/Heun) from noise to data.
  At `sigma_min=0` this is exactly **rectified flow** (Liu et al. 2022).
- **Why bother?** Straight paths → potential for **few-step** generation and **cleaner inversion**
  (no slow null-text optimization).

### 2.5 Inversion (for editing)
- **Simple:** to edit a real video, you must first find the "noise seed" that produces it. That's
  inversion. Then you change the pose and re-generate.
- **Precise:** DDIM inversion (old) vs **flow inversion** (ours) = integrate the velocity ODE
  *backwards* (t: 1→0) to recover the noise latent. Faster, no null-text.

### 2.6 Reflow
- **Simple:** train the flow model on its *own* outputs to make its paths even straighter, so you
  need even fewer steps.
- **Precise:** Liu et al.'s reflow — regenerate `(z₀, sample(z₀))` pairs and retrain. **It's a SPEED
  tool, not a quality tool, and not a replacement for DDIM.**

---

## 3. What we changed — the two contributions

| | Original MotionEditor | Our change | Status |
|---|---|---|---|
| **Pose detector** | OpenPose | **MediaPipe** (remapped to OpenPose format) | ✅ works, sharp |
| **Training loss** | DDPM epsilon (predict noise) | **CFM-OT velocity** (Contribution A) | ✅ trains |
| **Inversion** | DDIM + null-text (slow) | **Flow ODE inversion** (Contribution B) | ✅ runs, faster |
| **Sampling** | DDIM (50 steps) | **Flow-ODE sampler** (Euler/Heun) | ✅ runs (soft output) |
| **Speed-up** | — | **Reflow** (Contribution C) | 🟡 toy-proven; full version = future work |

---

## 4. What we built (the code)

```
fm_plugin_repo/
├── mediapipe_motioneditor_plugin/   # Goal 1: MediaPipe → OpenPose (extractor, remap, verify)
├── flow_matching_plugin/            # Goal 2: the flow-matching library + scripts
│   ├── flow_matching_loss.py        #   CFM-OT loss (Contribution A)
│   ├── train_adaptor_fm.py          #   drop-in trainer using the CFM-OT loss
│   ├── flow_inversion.py            #   flow inversion + sampler helpers (Contribution B)
│   ├── generate_reflow_pairs.py     #   reflow pair generation (Contribution C)
│   ├── train_reflow.py              #   reflow trainer (Contribution C)
│   ├── verify_loss.py / verify_inversion.py / verify_reflow.py   # toy proofs (sharp, clean)
│   └── tests/                       #   34 unit tests (all pass)
└── motionEditor/MotionEditor/       # the real paper code (vendored) with our patches:
    ├── motion_editor/.../unet_2d_blocks.py        # + xformers→SDPA shim (for new GPUs)
    ├── motion_editor/.../pipeline_motion_editor.py# + flow-ODE sampler branch
    └── inference.py                               # + flow-inversion / reflow hooks
```

**Proof that the method itself is correct:** `flow_matching_plugin/proofs/` contains **toy 2-D
results** (`verify_*.py`) showing the CFM-OT loss trains, inversion round-trips accurately, and reflow
straightens paths. **These are sharp and clean** — they prove the maths is implemented correctly,
independent of the heavy video pipeline.

---

## 5. The journey — everything we did and tried

This is the honest, blow-by-blow. Much of the effort was **getting a 2024 research repo to run on a
2026 GPU** — itself a real achievement.

### Phase 1 — MediaPipe (Goal 1), on CPU
- Built the extractor + 18-point remap + colour-faithful rendering.
- Verified on **all 6 MotionEditor cases**: 100% frame detection, ~0.8% pixel difference vs OpenPose.
- **Case-6 limitation found & documented:** ankles fail (baggy white trousers, low contrast) — an
  honest MediaPipe weakness, not tunable.

### Phase 2 — Getting MotionEditor to run on the GPU lab box (the hard slog)
The lab machine has a brand-new **NVIDIA RTX PRO 6000 "Blackwell"** (96 GB). Many obstacles, each solved:
- **Weights wouldn't download** (HuggingFace `hf`/xet + urllib hung on the lab network) → wrote a
  `curl`-based downloader.
- **Python 3.13 too new** for the 2023-era libraries (no prebuilt wheels, needed a Rust compiler) →
  made a **Python 3.10 conda env**.
- **A leftover `print(1/0)` "landmine"** in the vendored code → removed.
- **`CLIPFeatureExtractor` renamed** in new `transformers` → compatibility shim.
- **xformers (fast attention) didn't match the new PyTorch/Blackwell** → replaced its attention call
  with PyTorch's native **`scaled_dot_product_attention` (SDPA)** — a clean shim that needs no xformers.
- **CUDA out-of-memory** at 8 frames → `expandable_segments` setting fixed it (96 GB is plenty).

### Phase 3 — Training & first videos
- **Background reconstruction** training ✅
- **Epsilon (original) training** ✅ → **sharp** edited video (the paper, reproduced — proof the whole
  pipeline works on the new GPU).
- **CFM-OT (Contribution A) training** ✅ → checkpoint produced.

### Phase 4 — Flow inversion + sampler (Contribution B)
- First attempt: the velocity checkpoint run through **DDIM → pure rainbow noise** (a velocity model
  *cannot* be sampled by an epsilon sampler — fundamental mismatch).
- Built the **flow-ODE sampler** inside the pipeline (integrate the velocity field with Euler/Heun,
  keeping the two-branch attention). → noise turned into a **coherent edit** (soft, but real).
- This proved Contribution B works end-to-end.

### Phase 5 — MediaPipe + flow combined (Goal 1 + Goal 2)
- Swapped MediaPipe poses into the pipeline, retrained, inferred.
- **MediaPipe + epsilon = sharp.** **MediaPipe + flow = soft coherent.** Completed a full 2×2.

### Phase 6 — Trying to make the flow video sharp (the part still open)
Tried, in order — none sharpened it:
- More training (300→1000 steps) → **worse** (overfit one video at constant LR).
- Lower guidance scale (CFG 7.5→3) → no change.
- More/fewer ODE steps → no change.
- Skip inversion, sample from noise → **shattered** (the two-branch attention has no real source to
  copy from → garbage).
- **Queued next:** cosine learning-rate schedule (proper convergence, less overfit) — the one
  legitimate untried lever.

---

## 6. The results (the 2×2 ablation)

| poses ↓ / sampler → | **epsilon (DDIM)** | **flow (CFM-OT)** |
|---|---|---|
| **OpenPose** | **sharp** (baseline = paper reproduced) | soft, coherent |
| **MediaPipe** | **sharp** | soft, coherent |

**What it proves:**
- **MediaPipe ≈ OpenPose** in *both* columns → MediaPipe is a valid drop-in (**Goal 1 ✓**).
- **Flow replaces DDIM** end-to-end and produces coherent edits, faster inversion (no null-text)
  (**Goal 2 ✓**) — with a **sharpness gap** vs epsilon (the open problem).

Result gifs are in `fm_outputs/` and the lab box `~/` (e.g. `baseline_epsilon_case1.gif`,
`mediapipe_epsilon_case1.gif`, `mediapipe_flow_case1.gif`).

---

## 7. Why is the flow video soft? (the honest, detailed answer)

This is the central open question, so here is the full reasoning a beginner can follow:

**The model produces a "smooth average" instead of crisp detail.** Three compounding reasons:

1. **The velocity model is lightly/over-trained on ONE video.**
   MotionEditor fine-tunes on a single clip. With the flow (velocity) objective and a **constant
   learning rate**, training longer **overfits** → the model predicts the *average* motion → blur.
   (This is why 1000 steps looked *worse* than 300, not better.)

2. **The "two-branch attention injection" was designed for DDIM, not flow.**
   MotionEditor preserves the person's appearance by having an *editing* branch copy fine details
   (keys/values in attention) from a *reconstruction* branch, at specific points along the **DDIM**
   denoising trajectory. The **flow ODE trajectory is different** (straight vs curved), so this
   copying happens at "the wrong moments" → details smear. (Evidence: when we removed the real source
   by sampling from noise, the output **shattered** — confirming this branch is fragile to the
   trajectory.)

3. **The inversion uses the model's *unconditioned* output, which it never learned well.**
   Our flow inversion runs the U-Net **without** the pose skeleton (`normal_infer`). The *epsilon*
   model tolerates this (the base model can predict noise unconditionally). The **CFM-OT model only
   ever learned velocity *with* the skeleton**, so its unconditioned velocity is poorly defined →
   the recovered "noise seed" is imperfect → the re-generated edit is soft.

**In one sentence:** flow matching is a *training/sampling* swap, but MotionEditor's *appearance-
preservation machinery* (two-branch attention + inversion) is tightly tuned to diffusion/DDIM, and it
doesn't transfer cleanly to the flow trajectory — so the flow edit is coherent but soft.

**This is a legitimate research finding**, not a coding bug. The fixes are known but non-trivial
(see §9).

---

## 8. Where things stand right now

- ✅ **Goal 1 (MediaPipe):** done, verified, sharp results, one documented limitation (case-6 ankles).
- ✅ **Goal 2 (flow vs DDIM):** implemented end-to-end; **method proven** (toy proofs), **coherent
  edits** produced, **faster inversion** (no null-text). **Open issue:** sharpness gap vs epsilon.
- ✅ Full pipeline runs on the Blackwell GPU (a substantial engineering result).
- 🟡 **Reflow (C):** toy-proven + pairs generated; full conditioned reflow = future work.
- ⏳ **One sharpness experiment queued:** cosine-LR retrain. If it doesn't clearly help, we stop
  tuning and **quantify + write up**.

**This is a positive result.** "Negative" would be noise or a method that doesn't run. We have
coherent edits, a proven method, a clean 2×2, and a precise diagnosis of the remaining gap.

---

## 9. What's next (options, by effort)

**Cheap / safe (do these regardless):**
- **Quantitative metrics:** CLIP (prompt match), LPIPS-T (temporal smoothness), **NFE & wall-clock**
  (flow inversion *wins* here — no null-text loop). Turns gifs into a defensible numbers table.
- **Write-up:** the diagnostic journey + 2×2 + honest limitation = strong MSc material.

**Medium (the queued shot at sharper flow):**
- **Cosine-LR retrain** — proper convergence, less overfit. Medium probability.

**Harder (real novelty / future work):**
- **Conditioned flow inversion** — invert *with* the skeleton (fixes reason #3 in §7).
- **Flow-aware attention injection** — adapt the two-branch timing to the ODE trajectory (fixes #2).
- **Flow as distillation of the sharp epsilon editor** (InstaFlow-style) — best potential quality:
  keep epsilon's sharpness, gain flow's speed.
- **Conditioned reflow** — the proper version of Contribution C, for few-step speed.

---

## 10. How to reproduce (quick map)

**No GPU (proves the method, ~1 min each):**
```bash
cd flow_matching_plugin
python3 -m pytest tests/ -q                 # 34 tests
python3 verify_loss.py --out proofs/        # Contribution A toy proof
python3 verify_inversion.py --out proofs/   # Contribution B toy proof
python3 verify_reflow.py --out proofs/      # Contribution C toy proof
```

**MediaPipe (no GPU):**
```bash
python3 mediapipe_motioneditor_plugin/extract_pose_video.py -d <frames> -c openposefull --model-complexity 1
python3 mediapipe_motioneditor_plugin/verify.py --frames <frames> --openpose-ref <ref> --out proofs/
```

**Full GPU pipeline:** see `STEPS_TO_EXECUTE.md` and `RUNBOOK.md` (env setup, weights, train, infer).
Environment specifics + gotchas (Blackwell, Python 3.10, the curl downloader) are documented there.

---

## 11. Glossary (every term, one line)

- **MotionEditor** — the base AI that edits a person's motion in a video.
- **Pose skeleton** — stick-figure of the body used to guide the edit.
- **OpenPose / MediaPipe** — two tools that detect the skeleton (hard / easy to install).
- **Diffusion model** — generates images by removing noise step by step.
- **DDIM** — a deterministic diffusion sampler; **DDIM inversion** runs it backwards.
- **Null-text optimization** — a slow trick to make DDIM inversion accurate.
- **Flow matching / CFM-OT** — learn a straight noise→image path (velocity field) instead of denoising.
- **Velocity field** — the direction+speed to move a point from noise toward the image.
- **ODE / Euler / Heun** — maths + methods for "walking along" the flow path.
- **Flow inversion** — run the flow path backwards to find the noise seed (replaces DDIM inversion).
- **Reflow** — retrain a flow model on its own outputs to straighten paths (speed, not quality).
- **Epsilon vs velocity** — what the model predicts: noise (old) vs movement direction (new). They
  are **not interchangeable** — a velocity model can't be run by a noise sampler (→ rainbow noise).
- **Two-branch attention injection** — MotionEditor's trick to keep the person's appearance while
  changing motion; tuned for DDIM, fragile under flow.
- **ControlNet** — the part that injects the pose skeleton into the generator.
- **xformers / SDPA** — fast-attention libraries; we swapped xformers for PyTorch's SDPA for the new GPU.
- **Blackwell / sm_120** — the new NVIDIA GPU architecture in the lab box.
- **CFG (guidance scale)** — how strongly the generation follows the text prompt.
- **NFE** — "number of function evaluations" = how many model steps sampling takes (lower = faster).
```
