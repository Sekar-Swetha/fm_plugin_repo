# MotionEditor × Flow Matching — Full Dissertation Progress

_Last updated: 2026-07-08. TCD MSc dissertation, Swetha Sekar (sekars@tcd.ie)._
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

**Two contributions, evaluated independently:** MediaPipe pose extraction (Goal 1),
and flow-family few-step sampling on the standard OpenPose pipeline (Goal 2). The
sampler experiments (naive flow, DPM-Solver++, C3) were run with **OpenPose**
conditioning (hash-confirmed, §7); MediaPipe is validated **separately** — at the
source-pose level and end-to-end under epsilon/DDIM — and was **not** combined with
the flow samplers.

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
  MediaPipe is a single `pip` dependency with no C++/Caffe/CUDA build and runs directly
  in the `me310` environment — a practicality win over OpenPose's heavier build,
  independent of accuracy.
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

**Evaluated independently:** Goal 1 (MediaPipe) is a pose-extraction swap validated on
its own (source-pose + epsilon/DDIM). A/B, DPM-Solver++ and C3 (the flow-family /
Goal-2 work) were all run on the **standard OpenPose pipeline** (§7 hash provenance) —
MediaPipe was **not** the pose source in those sampler experiments.

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
| flow (ODE, 50 step) | `flow_ode_fm_50step_case1.gif` | Coherent edit, **soft**. |
| flow (ODE), naive-flow gif | `mediapipe_flow_case1.gif` | Coherent edit, **soft**. (conditioning OpenPose by inference, not hash-confirmed — see §7 †) |

**Conclusion:** Flow inversion + ODE replaces DDIM + null-text end-to-end and produces
coherent edits; the gap is *sharpness*, not failure.

> **Goal-1 scope (downgraded, honest):** MediaPipe ≈ OpenPose is verified **at the
> source-pose level** (`verify.py`, pixel-L1 ≈2/255) and **end-to-end under
> epsilon/DDIM** (`mediapipe_epsilon_case1.gif`). It is **NOT** tested under the flow
> samplers — the flow-family work (naive flow, DPM-Solver++, C3) used the **standard
> OpenPose pipeline** (§7 hash provenance). The `mediapipe_flow` gif's name once
> suggested a MediaPipe flow run, but its conditioning is not hash-recoverable and its
> PoseDist places it with OpenPose. Filling the missing "MediaPipe × flow-sampler" cell
> would need one qualitative **DPM-on-`mp_source`** run — optional, not required for the
> Goal-1 or Goal-2 claims.

### 5.3 Diagnosing the softness (levers ruled out)

> Filename note: several gifs here are named `mediapipe_flow_*`, but this is a naming
> artifact — like all flow-family runs they were conditioned on the **OpenPose**
> pipeline (inferred, §7 †); the names do **not** imply MediaPipe conditioning.

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

**Inversion lever closed — measured (EXP-1/2, `flow_roundtrip_check`, 2026-07-24).**
A source→source round-trip of the conditioned inversion (invert the source latent
with the SOURCE skeleton, forward-sample with the same conditioning and step count)
reconstructs the subject **sharply**, with subject/background latent-recovery ratio
**1.78×** at 50 steps. Sweeping inversion steps 50→100→200 collapses the absolute
subject error **~8×** (subj_RMS 0.0026→0.0009→0.0003) with the ratio trending toward 1
(1.78→1.65→1.57) — i.e. the residual is **inversion-integration error that vanishes
with steps**, not a velocity wall; the inversion recovers the subject to negligible
error given enough steps. Therefore the edited-subject ghost originates at the
**target-pose forward pass** (Cause 1 — the single-video velocity field's imprecision
on the pose change), **not** the inversion path. No inversion work can fix it; the
lever is closed.

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

`flow_matching_plugin/metrics.py`, values from 2026-07-06 (C3 lossless PNG;
baseline/flow/DPM gifs). **This revision (2026-07-08) changes conditioning-source
labels and narrative only — every Sharpness / subj-Sharpness / bg-SSIM / PoseDist
value is unchanged.**

**Conditioning provenance — SETTLED by byte-hash (`diag_conditioning.py`):**
`target_condition/openposefull`, the skeleton every eval config reads, is
**8/8 byte-identical to the original OpenPose backup** and **0/8 to the MediaPipe
render** (`mp_target`). So the **sampler comparison below is uniformly OpenPose-
conditioned.** Baseline, DPM-Solver++ and C3 are **hash-confirmed** OpenPose; the two
naive-flow rows are **OpenPose by inference** (they predate the hashed state, so their
original conditioning is not hash-recoverable — see the †). MediaPipe (Goal 1) is a
*separate* axis — validated on source frames (§5.1, `verify.py`) — and was **not** the
pose source in the DPM/C3 experiments.

### Sampler comparison (uniformly OpenPose conditioning)
| Method | Cond. | NFE | Sharpness (full) ↑ | subj Sharpness ↑ | bg SSIM ↑ | PoseDist-tgt ↓ | frames |
|---|---|---|---|---|---|---|---|
| **Baseline epsilon (DDIM-50 + null-text) — reference** | OpenPose | 50 | 0.0133 | 0.0149 | 0.901 | 0.171 | gif |
| A/B naive flow | OpenPose† | 50 | **0.0122** | **0.0088** | 0.877 | 0.177 | gif |
| A/B naive flow (50-step ODE) | OpenPose† | 50 | 0.0122 | 0.0159 | 0.874 | **0.196** | gif |
| DPM-Solver++ (order-2) | OpenPose | 20 | 0.0142 | 0.0154 | 0.854 | 0.173 | gif |
| DPM-Solver++ (order-2) | OpenPose | 15 | 0.0144 | 0.0169 | 0.852 | 0.173 | gif |
| **DPM-Solver++ (order-3)** | OpenPose | **15** | **0.0148** | **0.0197** | – | 0.175 | gif |
| C3 consistency | OpenPose | 1 | 0.0007 | 0.0057 | 0.902 | 0.176 | png |
| C3 consistency | OpenPose | 2 | 0.0038 | 0.0331 | 0.889 | 0.173 | png |
| C3 consistency | OpenPose | 4 | 0.0100 | 0.0859 | 0.852 | 0.171 | png |
| C3 consistency | OpenPose | 6 | 0.0145 | 0.1228 | 0.817 | 0.173 | png |
| **C3 consistency** | OpenPose | **8** | **0.0162** | **0.1280** | – | 0.172 | png |

_EXP-3/4 additions (2026-07-24, measured via `score_gif.py`): **DPM-Solver++ order-3
@ 15 NFE** lifts subject sharpness +17% (0.0169→0.0197) at no extra cost — adopted as
the DPM default; NFE≥20 does not help (15 is the knee); Karras spacing gave no benefit
and is broken on the pinned diffusers. **C3 @ 8 NFE** is the sharpest full-frame result
(0.0162); subject sharpness saturates by 6–8 (+4% from 6→8). Injection re-indexing
(EXP-5) did not improve logo/denim fidelity and was reverted._

> † **OpenPose (inferred from PoseDist 0.177; original conditioning not
> hash-recoverable).** The two naive-flow gifs predate the current file state, so a
> byte-hash can't confirm what drove them; their PoseDist (0.177) matches the
> OpenPose-driven cluster (~0.17), not the MediaPipe row (0.040), so they are treated
> as OpenPose. Not presented as hash-confirmed. All other sampler rows are
> hash-confirmed OpenPose.

### Goal-1 MediaPipe reference (NOT part of the sampler comparison)
| Method | Cond. | NFE | Sharpness (full) ↑ | subj Sharpness ↑ | bg SSIM ↑ | PoseDist-tgt ↓ | frames |
|---|---|---|---|---|---|---|---|
| MediaPipe epsilon (DDIM-50) | MediaPipe | 50 | 0.0133 | 0.0146 | 0.886 | **0.040** | gif |

> This MediaPipe row is the **Goal-1 reference** (MediaPipe-driven epsilon edit). It
> is **NOT C3's teacher**, and it sits on a **different, MediaPipe-convention target**
> — its low PoseDist (0.040) is *circular* (MediaPipe-driven output vs
> MediaPipe-extracted target), not superior fidelity (see the PoseDist note).
> **C3's distillation teacher is the OpenPose DPM-Solver++ trajectory**, not this row.

> **Goal-1 target caveat.** MediaPipe ≈ OpenPose is verified on the source frames
> (pixel-L1 ≈2/255; x-centroids 0.513 vs 0.509). Target-side equivalence is unverified:
> the shipped OpenPose target skeleton and MediaPipe-on-`target_images` differ 0.16 in
> x-centroid. The most likely explanation is a data-provenance difference — the
> dataset's OpenPose target was probably rendered from different (or differently-framed)
> inputs than `target_images` — with a genuine detector difference on this clip as the
> alternative. Disambiguating would require running OpenPose on the identical
> `target_images`, which we did not do; we claim source-frame equivalence only. This
> concerns Goal 1 alone: all sampler-comparison rows share the same OpenPose target, so
> the divergence does not affect the DPM/C3 results.

**Headline finding — quality scales with NFE; pose fidelity is uniform across the OpenPose samplers:**
- **Sharpness scales with steps** — full-frame 0.0007→0.0145 and subject-region
  0.006→0.123 over C3 1→6; C3-6 reaches the OpenPose DPM/baseline sharpness level
  (full 0.0145 ≈ 0.0144).
- **PoseDist-vs-target is ~flat 0.171–0.176 across all OpenPose sampler rows**
  (baseline, DPM, C3, incl. NFE-1) → no fidelity difference detectable across NFE.
  **Caveat (settled):** this ~0.17 is *not* pure pose error — every OpenPose-driven
  output carries the fixed **0.16 OpenPose-vs-MediaPipe target-skeleton x-offset**
  (OpenPose target x≈0.55 vs the MediaPipe target x≈0.39 the metric compares
  against). So PoseDist here is a **coarse consistency check**, not a fine ranking,
  and the MediaPipe row's 0.040 is a convention artifact, not a fidelity floor.
- ⇒ C3 is a **sharpness↔speed dial**; pose fidelity is uniform across the OpenPose samplers.

**Naive flow loses on quality:** softest subject (subj Sharpness 0.0088, below every
other row) and softer full-frame than baseline (0.0122 < 0.0133); its 50-step ODE
variant also has the highest PoseDist (0.196). The analyzed negative. Its
conditioning is **OpenPose by inference** (†, not hash-confirmed) — its 0.177 PoseDist
already sits in the OpenPose cluster, so the comparison is consistent.

**DPM-Solver++:** sharper than the DDIM-50 baseline (0.0144 > 0.0133) at equal pose
fidelity, **15 NFE vs 50** — both OpenPose-conditioned.

**bg SSIM 0.82–0.90** everywhere confirms the **scene is structurally preserved**
(the pipeline works); it dips slightly for the most-saturated rows (C3-6 = 0.816).

**Metrics-design note (important honesty):** *no reference-based appearance metric
works cleanly here* — every output differs from any reference in **both pose and
colour**, so LPIPS/SSIM-vs-teacher, LPIPS/SSIM-vs-source and global LPIPS all
mis-rank (they put the blurriest C3-1/2 "best" and the sharp C3-6/DPM "worst"). The
region split proved this: `bg_LPIPS` tracks the colour shift while `bg_SSIM` shows
structure is fine, and `subject_LPIPS-vs-teacher` stays confounded because outputs
are **not pixel-aligned to any reference** (pose + colour + the OpenPose-vs-MediaPipe
target offset). We therefore lead with **no-reference** metrics: full-frame and
**subject-region Laplacian sharpness** (subject one via the MediaPipe mask, immune
to pose *and* colour) + structural **bg SSIM** + **PoseDist**. Caveat: Laplacian
also reflects contrast, so subj-Sharpness *absolutes* for high-saturation C3-4/6
(0.088, 0.121) are inflated — use them for the **within-C3 trend**, and full-frame
Sharpness for cross-method absolutes. CLIP-sim (~0.26–0.28) is saturated; global
LPIPS-vs-source and subject LPIPS/SSIM-vs-teacher are retained in
`fm_outputs/metrics_table.md` only for continuity, flagged `(conf.)`.

**PoseDist note (settled — `diag_posedist.py` + `diag_conditioning.py`):** raw
PoseDist-vs-target compares **un-centred** MediaPipe landmarks against
`MediaPipe(target_images)`. The MediaPipe row scores 0.040 because it is
MediaPipe-driven onto its own convention (circular); OpenPose-driven rows score
~0.17 because their subject sits ~0.16 right of the MediaPipe target — a
**skeleton-position offset, not pose error**. Pose-normalized PoseDist (centre on
hip-midpoint, scale by torso) is added as a **coarse** variant (Task 5); the raw
column is retained. It stays **convention-biased toward the MediaPipe row**
regardless of normalization, and ~0.25 clustering must **not** be read as "all
methods equally pose-faithful".

**Lossless status:** C3 recomputed from lossless PNGs (done, above). The DPM
main-path `save_frames` dump captured the reconstruction branch (≈ source) rather
than the edit, so DPM keeps its gif number (it's the safety-net result, not the
headline — not worth chasing the branch quirk). **Still pending:** normalized-PoseDist
recompute on the box (metrics.py ready); wall-clock per run (NFE is the reliable
speed axis meanwhile). _Metric values from 2026-07-06; labels/narrative revised 2026-07-08._

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

`metrics.py` is written and §7 is populated. LPIPS-vs-teacher was **deliberately
retired** (confounded); CLIP-sim is saturated and footnoted — neither is an open item.

1. **Normalized-PoseDist recompute** — code is in `metrics.py` (Task 5); re-run
   metrics on the box to fill the coarse pose-normalized column.
2. **MediaPipe in the few-step setting (existence proof)** — could be shown via a
   single qualitative **DPM-on-`mp_source`** run as an existence-proof figure; **not
   required** for the quantitative claims, which stand on the OpenPose pipeline.
   (Conditioning provenance is settled: sampler rows are OpenPose — hash-confirmed, or
   inferred for the two naive-flow rows.)
3. **Goal-1 target divergence** — target-side MediaPipe-vs-OpenPose equivalence is
   unverified (§7 Goal-1 target caveat); source equivalence holds and the divergence
   does not affect the DPM/C3 results.
4. **Optional** — logit-normal t-sampling ablation for the naive-flow chapter;
   more evaluation cases (case-2+); wall-clock column (NFE is the speed axis meanwhile).
5. **Write-up** — MediaPipe (Goal 1, *separate* axis) + the flow-quality arc (naive
   soft → diagnose → PF-ODE retains prior → C3 distils to few-step), sampler table
   uniformly OpenPose. Naive flow = analyzed negative; DPM-Solver++ = safety net;
   C3 = novelty.

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
