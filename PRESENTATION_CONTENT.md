# FYP Presentation — Slide Content
_Fill-in doc for `FYP-Presentation-Template.pptx`. One section per template slide.
Source: `FULL_PROGRESS.md`. Paste text into the matching slide; figure paths noted._

---

## Slide 1 — Title
- **Title:** Improving MotionEditor: MediaPipe Pose Extraction and Few-Step Flow-Matching Video Motion Editing
- **Presenter:** Swetha Sekar
- **Date:** [presentation date]
- (one-line subtitle, optional) *Faster, sharper one-shot video motion editing on a diffusion backbone.*

---

## Slide 2 — Overview (agenda)
- The problem: one-shot video **motion editing** (MotionEditor, CVPR 2024)
- Two contributions: **MediaPipe** pose extraction · **flow-family few-step** sampling
- Design & the key insight (keep the pretrained sharpness prior)
- Implementation issues (the debugging war)
- Demo — before/after edits
- Evaluation — sharpness × speed × fidelity ablation
- Conclusions, lessons, future work

---

## Slide 3 — Background & Related Work
**Basis:**
- **MotionEditor (CVPR 2024)** — one-shot, single-video diffusion motion editing: ControlNet on pose skeletons + content-aware motion adapter + two-branch attention injection for temporal consistency. Baseline sampler = DDIM-50 + null-text optimization (slow).
**Related methods this project draws on:**
- **Pose:** OpenPose (heavy C++/Caffe build) vs **MediaPipe** BlazePose-33 (single pip dependency).
- **Generative core:** DDPM/DDIM → **Flow Matching** (CFM-OT / rectified flow, Lipman et al. 2023); **probability-flow ODE** sampling; **DPM-Solver++** (high-order deterministic PF-ODE solver); **Consistency / LCM distillation** for few-step sampling.
**Gap addressed:** MotionEditor is sharp but slow (50 steps + null-text). Can a flow-family sampler make it **few-step and sharp**?

---

## Slide 4 — Design
**Two contributions, evaluated independently:**
1. **Goal 1 — MediaPipe replaces OpenPose** for pose extraction (BlazePose-33 → OpenPose-COCO-18 via a remapping bridge). Drop-in, format-compatible.
2. **Goal 2 — flow-family few-step sampling** replacing DDPM/DDIM.

**Key design insight (the story):**
- Naive flow matching (train a velocity field) → **soft** edits.
- **Why:** MotionEditor's sharpness comes from the **frozen pretrained Stable-Diffusion epsilon prior**; training a velocity head from one video discards that prior, and single-video transport is degenerate.
- **Fix = keep the prior:** sample the *frozen sharp epsilon editor* via its probability-flow ODE — no retraining of the generative core.
  - **DPM-Solver++** — high-order PF-ODE solver → sharp at ~15 steps.
  - **C3 = consistency distillation** — distil the DPM teacher into a **few-step (4–8)** student.

*(Figure: pipeline diagram — source video → pose (MediaPipe/OpenPose) → ControlNet + adapter → {DDIM-50 baseline | DPM-Solver++ | C3 few-step} → edited video.)*

---

## Slide 5 — Implementation Issues (the honest engineering)
- **Hardware/stack:** NVIDIA Blackwell (sm_120) — xformers CUDA ext mismatched torch; bypassed via an **SDPA shim**. diffusers pinned 0.15.1 (MotionEditor forks internals).
- **Memory:** 8 frames ≈ 90 GB; needed fp16 + gradient checkpointing + `unet.train()` + memory-efficient attention (else the naive attention matrix OOMs at ~8 GB/softmax).
- **Correctness bugs found & fixed:** velocity closure hard-coded a 4-branch CFG layout (crashed no-CFG runs); U-Net timestep must be size-1 (b·f only for ControlNet); device/dtype alignment in distillation.
- **Method dead-ends (reported honestly):** single-pair distillation (C2) — degenerate (500 steps beat 1500); injection re-timing — tested, rejected.
- **Evaluation rigor:** metrics read lossless PNGs not 256-colour GIFs; reference-based LPIPS/SSIM confounded in motion editing (pose + colour differ) → switched to **no-reference** sharpness + structural SSIM + pose distance; **conditioning provenance settled by byte-hash** (the sampler comparison is uniformly OpenPose).

---

## Slide 6 — Demo of Project (before / after edits)
Show these GIFs (right panel = edited output). Suggested order:
1. `fm_outputs/baseline_epsilon_case1.gif` — **baseline** (DDIM-50 + null-text), sharp reference.
2. `fm_outputs/mediapipe_epsilon_case1.gif` — **Goal 1**: MediaPipe pose, same quality.
3. `fm_outputs/mediapipe_flow_case1.gif` — **naive flow → soft** (the problem).
4. `fm_outputs/dpm_outputs/sample-all.gif` (or `dpm_sweep/dpm_n15_o3.gif`) — **DPM-Solver++, sharp @ 15 NFE**.
5. `fm_outputs/cons_outputs/sample-consistency.gif` — **C3, sharp @ 4 NFE** (headline).
- **The dial (strongest visual):** `c3_dial.gif` — 1→2→4→6→8 NFE, soft→sharp.

---

## Slide 7 — Evaluation
**Ablation (case-1, uniformly OpenPose-conditioned):**

| Method | NFE | Sharpness ↑ | subj Sharpness ↑ | PoseDist ↓ |
|---|---|---|---|---|
| Baseline (DDIM-50 + null-text) | 50 | 0.0133 | 0.0149 | 0.171 |
| Naive flow (A/B) | 50 | 0.0122 | 0.0088 | 0.177 |
| DPM-Solver++ order-2 | 15 | 0.0144 | 0.0169 | 0.173 |
| **DPM-Solver++ order-3** | 15 | 0.0148 | 0.0197 | 0.175 |
| C3 consistency | 4 | 0.0100 | 0.0859 | 0.171 |
| C3 consistency | 6 | 0.0145 | 0.1228 | 0.173 |
| **C3 consistency** | 8 | 0.0162 | 0.1280 | 0.172 |

**Figures (all axis-labelled, legend, readable — per template note):**
- `outputs/figures/inversion_convergence.png` — inversion round-trip error vs steps (lever-closed diagnostic).
- `outputs/dpm_order_compare.png` — order-2 vs order-3, same 15 NFE (+17% subject sharpness).
- `outputs/c3_dial.png` — C3 sharpness climbing then saturating 6→8.

**Headline results:**
- **Sharpness scales with NFE; edit fidelity is uniform** (PoseDist ~0.17 for all samplers) → C3 is a **sharpness↔speed dial**.
- **DPM-Solver++ order-3 @ 15 NFE** ≈ **12× fewer model calls** than DDIM-50 + null-text, at equal/better sharpness.
- **C3 sharp at 4–8 NFE**; naive flow = the analyzed negative (loses on both axes).

**Honesty (say it — it's a credibility gain):** Laplacian sharpness also reflects contrast (use for trend); PoseDist is a coarse pose-correct check; reference LPIPS/SSIM confounded → footnoted.

---

## Slide 8 — Conclusions
**Achieved:**
- Goal 1: MediaPipe = OpenPose quality, one pip dependency (verified at source-pose + epsilon/DDIM level).
- Goal 2: two working few-step samplers — DPM-Solver++ (sharp, 15 NFE) and **C3 consistency distillation (novel, sharp at 4–8 NFE)**.
- Diagnosed *why* naive flow fails (loses the SD prior) — turned a failure into the motivation.

**Lessons learnt:**
- The sharpness lived in the **frozen prior**, not the learned field — the insight that unlocked everything.
- Measure honestly: a confounded metric (LPIPS-vs-source) initially told the *opposite* story; no-reference metrics + provenance hashing fixed it.
- Report negatives (C2, injection re-timing) — they strengthen the work.

**Reflection:** the hard part was engineering/diagnosis on a pinned, bleeding-edge stack, not the maths.

**Future work:** multi-case eval (all 6 in-repo cases; pipeline is portable — `inference.py` is case-agnostic), external datasets (TaichiHD), workshop paper.

---

## Slide 9 / 10 — Questions / Thank You
- (template as-is)

---

### Notes on figures
- FIG paths under `outputs/` and `fm_outputs/` are on the **GPU box + your Google Drive** (not all in git). Pull the figure PNGs/GIFs from Drive for the slides.
- The three `figures/` scripts regenerate FIG A/B/C if needed.
