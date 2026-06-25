# Plan: Sharp Flow Editing via Distillation (Contribution C2)

**Distill the sharp epsilon MotionEditor into a flow model — inherit its sharpness, gain flow's speed.**

_One-page design to present to the advisor and start from. This is the "C2 direct source→target
editor" from the original design doc, now motivated by the diagnostic result: the static scene
renders sharp under flow, but the moving subject ghosts because the CFM-OT velocity field (one video,
limited budget) isn't precise enough on dynamic content. Distillation sidesteps that by copying a
teacher that already gets the moving subject right._

---

## 1. The problem (one line)
Flow sampling produces a **sharp background but a ghosted moving person**. Training/CFG/steps/attention
are all ruled out — the limit is the velocity field's precision on the dynamic subject.

## 2. The idea (one line)
The **epsilon editor is already sharp**. Train a **flow model to reproduce its edited output** —
a direct **source-latent → sharp-edited-latent** flow. Sharpness comes from the teacher; speed comes
from the straight flow path.

## 3. Why it works
- Avoids the hard problem ("learn a precise per-frame velocity for a moving subject from one video").
- The student only has to **interpolate along a straight line** between the source latent and the
  *known sharp* edited latent — a far easier regression than free-form video generation.
- At inference: **no inversion needed** (the source latent is the start), few ODE steps → fast.

## 4. Method (the coupling)
For each clip:
- `z_src` = VAE latent of the source video (already have this).
- `z_edit` = latent of the **sharp epsilon edit** (teacher output, *before* VAE decode), under the
  **target** skeleton.
- Train the flow model on the coupling `(z_src → z_edit)`:
  - `x_t = (1−t)·z_src + t·z_edit`
  - target velocity `v = z_edit − z_src`   (CFM-OT at sigma_min=0)
  - loss `‖v_θ(x_t, t, target_skeleton) − v‖²`
- **Inference:** `z_edit_hat = flow_sample(v_θ, z_src, N steps, target_skeleton)` → VAE decode. Sharp + few-step.

This is exactly **Contribution C2 (transport reflow)** — a learned source→target coupling instead of
inversion.

## 5. Implementation — reuse what exists, three pieces

### 5.1 Generate distillation pairs (`z_src`, `z_edit_sharp`, `cond_tgt`)
- Run the **working epsilon pipeline** (DDIM, `use_flow_inversion=False`, epsilon checkpoint) and
  **dump the final edited latent** (the `latents` after the denoising loop, before VAE decode).
- *Hook:* extend the existing `dump_reflow_records` / `gen_reflow_pairs` path in `inference.py` to
  also save the **edited latent** as `z_edit`. We already save `z_src` and `cond_tgt`.
- Output: `(z_src, z_edit_sharp, cond_tgt)` per clip (reuse `save_records` / `save_shards`).

### 5.2 Distillation training (`train_reflow.py`, real model)
- Reuse `reflow_training_step` — it already computes the CFM-OT loss on a `(z0, z1, cond)` coupling.
  Feed `z0 = z_src`, `z1 = z_edit_sharp`.
- **The one real gap:** load the actual MotionEditor CFM-OT U-Net into the trainer (the `--stub`
  uses a toy model). Mirror how `inference.py` builds the U-Net (`UNet2DConditionModel.from_pretrained
  (subfolder="unet", ...)` + load the adapter `.pth`), then train its parameters on the pairs with
  the **conditioned forward** (ControlNet on target skeleton → residuals → U-Net), same as
  `train_adaptor_fm.py`.
- Init from the existing CFM-OT checkpoint; save `checkpoint-c2`.

### 5.3 Inference (no inversion)
- New path: start from `z_src` (not noise, not inversion), run the **flow-ODE sampler** under the
  **target** skeleton for a few steps → edited latent → decode.
- Reuse the `flow_sampling` branch already added to `pipeline_motion_editor.py`; feed `latents = z_src`
  and set few `flow_steps` (e.g. 4–8). Point at `checkpoint-c2`.

## 6. Success criteria
- **Primary:** the moving person is **sharp** (matches the epsilon teacher), not ghosted.
- **Secondary:** achieved in **few steps** (4–8) → faster than DDIM-50 + null-text. Report NFE +
  wall-clock + LPIPS-vs-teacher.

## 7. Risks & mitigations
| Risk | Mitigation |
|---|---|
| One video → one pair → overfit | augment (crops, frame stride); ideally distill across several clips/cases |
| Real-U-Net load into the reflow trainer (the known gap) | copy `inference.py`'s exact U-Net construction; load weights only (we already fixed the LR/scheduler-resume bug) |
| Student inherits any teacher artifacts | acceptable — goal is to *match* the sharp teacher, then gain speed |
| Distillation still soft on subject | fall back: distill at higher step count first, then reduce |

## 8. Effort estimate
- Pair dump hook: small (extend existing hook).
- Real-U-Net reflow trainer: **medium** (the main work — the U-Net load + conditioned forward).
- Inference-from-source path: small (reuse flow sampler).
- **~1–2 focused days + debugging.**

## 9. How this lands for the dissertation
- Turns the soft-flow finding into a **concrete, novel contribution**: a distilled, **inversion-free,
  few-step flow editor** that is **sharp** (teacher-matched) and **fast** — the strongest version of
  Contribution C2.
- Even partial success (sharp at moderate steps) is a clear result; full success (sharp at 4 steps) is
  a headline.

## 10. Decision to request from advisor
*"I've diagnosed that the moving-subject softness is the velocity field's precision on dynamic
content. My proposed fix is to distill the sharp epsilon editor into a flow model (my C2 design) —
inversion-free, few-step, sharp. Estimated 1–2 days. Shall I proceed, or prioritise metrics + write-up
of the current results?"*
