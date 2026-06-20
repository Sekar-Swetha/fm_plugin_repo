# Project Status — MotionEditor + Flow Matching + MediaPipe

_Last updated: 2026-06-17_

## 1. The goal (what you set out to do)

Improve the **MotionEditor** paper (pose-guided video motion editing) with two
independent upgrades:

1. **MediaPipe instead of OpenPose** for pose extraction — because OpenPose
   would not build/run on your laptop.
2. **Flow Matching** instead of the original DDPM/diffusion machinery, in three
   layered contributions:
   - **A** — CFM-OT training loss (replaces the DDPM epsilon loss).
   - **B** — ODE flow inversion (replaces DDIM inversion + null-text optimisation).
   - **C** — Reflow for few-step sampling (C1) and a direct source→target editor (C2).

Because you could not run the base MotionEditor on your laptop, the strategy was
to **build each upgrade as a self-contained plugin, prove it correct on CPU with
toy/offline tests, and defer the real GPU run** until a GPU box is available.

---

## 2. What exists right now

The repo has **3 plugins** + the **vendored base MotionEditor repo**:

```
fm_plugin_repo/
├── mediapipe_motioneditor_plugin/   ← Goal 1: MediaPipe replaces OpenPose
├── flow_matching_plugin/            ← Goal 2: Contributions A, B, C
├── mpipe/                           ← MediaPipe scratch/experiment dir
└── motionEditor/MotionEditor/       ← the real base paper code (vendored)
```

### 2.1 Status by component

| Component | Code | Unit tests | CPU/toy proof | Run on REAL data + GPU | Wired into MotionEditor |
|---|---|---|---|---|---|
| **MediaPipe pose plugin** | Done | Done (`test_geometry.py`) | **Yes — ran on a real 91-frame video, 100% detection** | n/a (CPU) | Drop-in ready; not yet used by a real train/inference run |
| **A — CFM-OT loss** | Done (`train_adaptor_fm.py`) | Done (15 tests) | Yes (`proofs/*.png`) | **No** | Drop-in script; never trained on GPU |
| **B — Flow inversion** | Done (`flow_inversion.py`) | Done (16 tests, just committed) | Yes (`verify_inversion.py`, toy 2-D) | **No** | **Inference patch NOT applied** (see §4.1) |
| **C — Reflow** | Partial (`generate_reflow_pairs.py`, `train_reflow.py`) | Done | Stub-only | **No** | Real-data mode is a stub (`--stub`); raises on real checkpoints |

**One-line truth:** all the math/library code is written and passes offline
tests. **Nothing has yet run through the real MotionEditor pipeline on a GPU,
and no real edited video has been produced or measured.**

### 2.2 What's genuinely "real" already

- **MediaPipe plugin** actually processed a real source video: 91 frames, all
  detected, mean visibility 0.966 (`mediapipe_motioneditor_plugin/proofs/source_video/report.md`).
  This is your strongest concrete result so far.
  - Caveat: it was **not** diffed against a real OpenPose reference (no
    `--openpose-ref` was passed), so "matches OpenPose" is not yet proven —
    only "produces a clean, correctly-coloured OP-18 skeleton".

---

## 3. Are we done with the code? — No, three gaps

1. **B.2.1 inference patch missing.** `flow_inversion.py` has the
   `MotionEditorFlowInversion` class, but it is **not yet wired into**
   `motionEditor/MotionEditor/inference.py`. Until then, flow inversion cannot
   run inside the real edit pipeline. (Spec for the patch: `README_contrib_BC.md` §B.2.1.)

2. **C real-data mode is a stub.** `generate_reflow_pairs.py` and
   `train_reflow.py` only run with `--stub` (synthetic tensors). On a real
   checkpoint they intentionally `raise SystemExit("... needs the MotionEditor
   U-Net + VAE (GPU)")`. The real loader/VAE-encode path is not written yet.

3. **`verify_reflow.py` + reflow proofs don't exist.** `README_contrib_BC.md`
   §C.5 lists them as required artefacts; they are still TODO.

Everything else (loss, inversion math, pair-generation logic, reflow loss reuse)
is implemented and tested.

---

## 4. How to test your changes

### 4.1 Right now, on your laptop (CPU, no GPU, no MotionEditor) — works today

```bash
# Flow matching — Contribution A (loss): 15 tests
cd flow_matching_plugin && python3 -m unittest discover -s tests -v

# Flow matching — B & C: 16 tests
python3 -m pytest tests/test_inversion.py tests/test_reflow.py -q

# Generate toy proof artefacts
python3 verify_loss.py --out proofs/          # A: loss curve, straight paths
python3 verify_inversion.py --out proofs/     # B: roundtrip error vs steps

# Reflow dev pass (stub, CPU only)
python3 generate_reflow_pairs.py --stub --mode c1 --out runs/pairs-c1/
python3 train_reflow.py --stub --reflow-pairs-dir runs/pairs-c1/

# MediaPipe plugin tests
cd ../mediapipe_motioneditor_plugin && python3 -m unittest discover -s tests -v

# MediaPipe on a real video (CPU)
python3 extract_pose_video.py -d /path/to/source.mp4 -c openposefull --write-keypoints
python3 verify.py --frames data/case-1/images --out proofs/   # add --openpose-ref to compare
```

These prove **correctness of the logic**. They do **not** prove the upgrades
improve real video editing — that needs §4.2.

### 4.2 The real test (needs a GPU + MotionEditor environment) — not done yet

This is the part you couldn't do on your laptop. It requires:
- A CUDA GPU (the paper uses A100; smaller works for 1 case, slower).
- MotionEditor's full env + pretrained weights (Stable Diffusion, ControlNet-OpenPose,
  the motion adapter) downloaded.
- One evaluation case's data (e.g. `data/case-1/`).

---

## 5. Do you need to train again with flow matching? — Yes

**The CFM-OT checkpoint has never been trained.** B and C are meaningless on
real video until A produces a velocity-prediction checkpoint, because:
- B (flow inversion) requires a **velocity** field, not an epsilon field. There
  is even a guard (`assert_cfm_ot_checkpoint`) that refuses an epsilon checkpoint.
- C (reflow) re-trains starting from the A checkpoint.

So the dependency chain is strict:

```
MediaPipe poses ──► A: train CFM-OT ──► B: invert + edit ──► C: reflow (fast)
                    (REQUIRED first)
```

Training A is short — README estimates ~5–10 min on an A100 for one case
(`train_adaptor_fm.py`), because you fine-tune the adapter, not from scratch.

---

## 6. Next steps (ordered)

### Phase 0 — finish the code (laptop, no GPU)
- [ ] Write the B.2.1 patch into `motionEditor/MotionEditor/inference.py`
      (gated behind `--use_flow_inversion`; baseline path stays bit-identical).
- [ ] Implement the real-checkpoint path in `generate_reflow_pairs.py` and
      `train_reflow.py` (VAE-encode real clips; load the A checkpoint).
- [ ] Add `verify_reflow.py` + the C proofs (`README_contrib_BC.md` §C.5).
- [ ] Run the MediaPipe `verify.py` **with `--openpose-ref`** once you have any
      OpenPose reference frames, to prove equivalence (not just plausibility).

### Phase 1 — get a GPU
- [ ] Use a cloud GPU (Colab/Kaggle free tier can run 1 case; or a rented A100).
      You do **not** need to fix OpenPose on your laptop — the MediaPipe plugin
      removes that blocker.
- [ ] Install MotionEditor's env there; download its pretrained weights.

### Phase 2 — prove MediaPipe (Goal 1)
- [ ] Extract poses for case-1 with the MediaPipe plugin.
- [ ] Run baseline MotionEditor `train_adaptor.py` + `inference.py` on those
      poses. Confirm the edited video quality matches the paper's OpenPose result.
- [ ] Archive `proofs/` (side-by-side, overlay, diff, report) per case.

### Phase 3 — prove Flow Matching A
- [ ] `train_adaptor_fm.py` on case-1 → CFM-OT checkpoint.
- [ ] A/B vs the baseline epsilon checkpoint (same seed/steps): loss curves,
      then downstream edit quality.

### Phase 4 — prove B
- [ ] Run inference with `--use_flow_inversion`. Compare wall-clock + recon LPIPS
      vs DDIM-50 + null-text (`README_contrib_BC.md` §B.2.2 table).

### Phase 5 — prove C
- [ ] Generate real pairs, reflow round 1, run inference at NFE 1–4.
- [ ] Fill `nfe_vs_quality.png` and the final ablation table (A, A+B, A+B+C1, A+B+C2).

---

## 7. How you "prove the results" for the dissertation

Two evidence tiers — you already have tier 1, you still need tier 2:

**Tier 1 — correctness (have it):** unit tests + toy proofs + MediaPipe real-frame
report. Shows the methods are implemented correctly and the math holds. CPU-only.

**Tier 2 — real-world improvement (need GPU):** edited videos + quantitative
metrics on MotionEditor's evaluation cases:
- **CLIP score** (edit follows the text prompt),
- **LPIPS-T / temporal consistency** (smooth across frames),
- **Reconstruction LPIPS** (B: inversion fidelity vs DDIM),
- **Wall-clock / NFE** (B and C: speed),
- **Ablation table**: baseline vs A vs A+B vs A+B+C, plus OpenPose-vs-MediaPipe.

The ablation table + a few before/after video stills is the core dissertation
result. Each row is independently valid, so **partial completion still gives a
defensible thesis** (e.g. MediaPipe + A done, C left as future work).

---

## 8. Biggest risks / honest caveats

- **No real run has happened.** Every "improvement" claim is currently
  theoretical or toy-scale. The whole project hinges on Phase 1 (getting a GPU).
- **MediaPipe ≠ OpenPose not yet measured.** You have a plausible skeleton, not a
  proven match. Run the `--openpose-ref` diff.
- **C2 (the novel direct-editor)** depends on A + B both working on real data
  first; it is the highest-risk, highest-reward piece and is furthest from done.
- The vendored `motionEditor/MotionEditor/` is the real base code — confirm its
  license/attribution in `motionEditor/NOTICE.md` before publishing the repo.
