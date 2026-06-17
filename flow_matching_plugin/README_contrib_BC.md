# Contributions B & C — Flow Inversion + Reflow for MotionEditor

Companion to `README.md` (Contribution A — CFM-OT training loss).
This document covers:

- **Contribution B** — replace DDIM inversion + null-text optimisation with
  ODE inversion of the CFM-OT velocity field.
- **Contribution C** — reflow (Liu et al. 2022 §2.1) for few-step sampling
  (C1) and a learned source→target coupling (C2).

Both build on a U-Net trained with `train_adaptor_fm.py` from Contribution A.

---

## Status snapshot

| Piece | File | State |
|---|---|---|
| Backward/forward ODE integrators | `flow_inversion.py` | Implemented |
| `MotionEditorFlowInversion` wrapper | `flow_inversion.py` | Implemented (not yet wired into `inference.py`) |
| Unit tests for B | `tests/test_inversion.py` | Implemented |
| Inference-site patch (MotionEditor) | `motionEditor/inference.py` | **TODO** — Section B.2 |
| Pair-generation script (C1 + C2) | `generate_reflow_pairs.py` | **TODO** — Section C.2 |
| Reflow trainer | `train_reflow.py` | **TODO** — Section C.3 |
| Reflow config | `configs/train-motion-reflow.yaml` | **TODO** — Section C.4 |
| Reflow proofs / verify | `verify_reflow.py`, `proofs/` | **TODO** — Section C.5 |

---

## Contribution B — Flow inversion

### B.1 What is already done

- `flow_invert(velocity_fn, x_1, num_steps, method)` — backward Euler/Heun
  ODE solve, `t: 1 → 0`. Returns the recovered noise latent.
- `flow_sample(velocity_fn, x_0, num_steps, method)` — forward solve,
  `t: 0 → 1`. Used for the editing branch with the target condition.
- `roundtrip_error(...)` — `‖sample(invert(x_1)) − x_1‖₂`; used in tests and
  by the verification harness.
- `MotionEditorFlowInversion` — wraps U-Net + ControlNet + text encoder
  into the closure signature `flow_invert` expects.

Reference: `flow_inversion.py` (whole file is annotated). Tests cover
identity-velocity round trip, OT-velocity round trip, batch shapes,
Heun-vs-Euler error scaling, and the `MotionEditorFlowInversion` mock path.

### B.2 What remains to implement

#### B.2.1 Patch the MotionEditor inference site

Touch `motionEditor/inference.py` (the existing baseline) at the call to
`MyNullInversion(...).invert(...)` / `ddim_inversion(...)`. Replace with a
construction of `MotionEditorFlowInversion`.

Pseudo-diff (the only file in the MotionEditor repo that changes for B):

```python
# motionEditor/inference.py — inside the inversion block (~lines 277-294)

if use_flow_inversion:                                       # NEW gate
    from flow_matching_plugin.flow_inversion import MotionEditorFlowInversion
    inverter = MotionEditorFlowInversion(
        unet=pipeline.unet,
        controlnet=pipeline.controlnet,
        text_encoder=pipeline.text_encoder,
        source_skeleton=source_skeleton,                     # already in scope
        num_steps=cfg.flow_inv_steps,                        # 4–10 typical
        method=cfg.flow_inv_method,                          # "heun" default
        prepare_image_fn=pipeline.prepare_image,             # avoids circular import
    )
    ddim_inv_latent = inverter.invert(latents, source_prompt_ids)
elif use_null_inv:
    ddim_inv_latent = MyNullInversion(pipeline, ...).invert(latents, prompt)[0]
else:
    ddim_inv_latent = ddim_inversion(pipeline, ddim_inv_scheduler, latents, ...)
```

Acceptance criteria:
- A `--use_flow_inversion` flag toggles between baseline and B. The
  baseline path must remain bit-identical (no refactor regressions).
- The two-branch attention injection (`inference.py` reconstruction
  branch + high-fidelity K/V swap) is unchanged. Flow inversion only
  replaces the *latent preparation* step.
- The CFM-OT-trained checkpoint loads; an epsilon-trained checkpoint
  errors out early with a clear message (check `loss_type` in the
  training config sidecar).

#### B.2.2 Inversion-quality verification

Add `verify_inversion.py` next to `verify_loss.py`. Required artefacts:

- `proofs/roundtrip_vs_steps.png` — `‖sample(invert(x)) − x‖` as a function
  of `num_steps ∈ {1, 2, 4, 8, 16, 32}`, Euler and Heun overlaid.
  Heun should land on the O(1/N²) line, Euler on O(1/N).
- `proofs/inversion_vs_ddim.md` — table comparing wall-clock time and
  reconstruction LPIPS on the same 20 clips MotionEditor evaluates on,
  rows: `{DDIM-50, DDIM-50+nulltext, FlowInv-4-Heun, FlowInv-10-Heun}`.
- `proofs/inversion_trace.png` — for one clip, plot `‖x_t − linear(x_1, x_0, t)‖`
  along the inversion trajectory. Near-zero confirms paths are straight.

#### B.2.3 Tests to add

`tests/test_inversion.py` already covers the library. Add:
- An end-to-end smoke test that loads a tiny U-Net stub and runs
  `MotionEditorFlowInversion.invert` on a `(1, 4, 4, 8, 8)` latent.
- A regression test that pins the round-trip error at `num_steps=8` to
  three decimal places for a fixed seed.

### B.3 How to run B (once B.2.1 lands)

```bash
# Inside the MotionEditor environment:
python motionEditor/inference.py \
    --config motionEditor/configs/case-1/inference.yaml \
    --checkpoint runs/case-1-fm/checkpoint-300 \
    --use_flow_inversion \
    --flow_inv_steps 8 \
    --flow_inv_method heun
```

Expected: total inference time drops from ~10 min/video to ~1–2 min/video
on A100 (most of the saving is from removing null-text optimisation).
Edit quality should be on par or better on CLIP / LPIPS-T.

### B.4 What B does NOT change

- The motion adapter, ControlNet, skeleton alignment, two-branch
  attention injection, and the prompt-conditioning scheme.
- Training. B is purely an inference-time swap.

---

## Contribution C — Reflow

Reflow turns the 1-rectified flow from Contribution A into a 2-rectified
(or 3-rectified) flow whose ODE trajectories are *straighter*. Two
variants:

- **C1 — generation reflow.** Pairs `(Z_0, Z_1)` with `Z_0 ∼ N(0, I)`,
  `Z_1` = video latent under target pose, generated by running the
  trained flow forward. After 1 reflow round, 1–4 Euler steps suffice for
  sampling. This is the standard Liu et al. recipe.
- **C2 — transport reflow (the novel one).** Pairs `(Z_0, Z_1)` with
  `Z_0` = source-video latent under source pose, `Z_1` = source-video
  latent under target pose (i.e. the *edited* latent produced by
  Contribution A + B). Train a second flow on this coupling and you have
  a *direct video-to-video editor* — no inversion at inference.

### C.1 Why this matters

- C1 makes MotionEditor interactive-speed (NFE drops from 50 → 1–4).
- C2 is the strongest methodological delta on top of MotionEditor: the
  edit operation becomes a learned coupling rather than a patched
  inversion + 2-branch-attention pipeline. See `MotionEditor_Gap_Analysis.md`
  §4 Contribution C.

### C.2 Pair generation — `generate_reflow_pairs.py`

New script. Inputs: a CFM-OT-trained checkpoint + a small video set
(TaichiHD or the 20 in-the-wild clips MotionEditor uses).

Pseudocode:

```python
# generate_reflow_pairs.py
#
# Modes:
#   --mode c1   noise → data pairs (generation reflow)
#   --mode c2   src-pose-latent → tgt-pose-latent pairs (transport reflow)
#
# Output: a sharded .pt or .safetensors dataset of (Z_0, Z_1, cond) tuples.

for video in dataset:
    z_src   = vae.encode(video).latent_dist.sample() * scale
    cond_src = controlnet_features(source_skeleton)
    cond_tgt = controlnet_features(target_skeleton)

    if args.mode == "c1":
        z0 = torch.randn_like(z_src)
        z1 = flow_sample(velocity_fn(cond_tgt), z0,
                          num_steps=args.num_steps).latent_0
        save(z0, z1, cond_tgt)

    elif args.mode == "c2":
        # Use Contribution B to land on the source manifold, then
        # Contribution B's forward solve under the target condition.
        z0_noise = flow_invert(velocity_fn(cond_src), z_src,
                                num_steps=args.num_steps).latent_0
        z1_edit  = flow_sample(velocity_fn(cond_tgt), z0_noise,
                                num_steps=args.num_steps).latent_0
        # The C2 pair is (source latent → edited latent), conditioned on
        # the *target* skeleton.
        save(z_src, z1_edit, cond_tgt)
```

Implementation notes:
- The `velocity_fn` closure is exactly the one `MotionEditorFlowInversion`
  builds; expose it as a public helper so the pair generator can re-use it.
- Save `cond_tgt` as the *skeleton tensor*, not the post-ControlNet
  residuals — the residuals depend on `x_t` and must be recomputed each
  reflow step.
- Shard size: aim for ≤ 2 GB per shard. One frame of `(1, 4, 16, 64, 64)`
  at fp16 is ~130 KB; 20 clips × 50 frames × 8 augmentations ≈ 1 GB.
- Determinism: log the seed and the generating checkpoint hash in a
  sidecar JSON next to every shard.

### C.3 Reflow trainer — `train_reflow.py`

Re-uses `flow_matching_loss.py`. The only structural change vs.
`train_adaptor_fm.py`:

- The batch sampler reads `(Z_0, Z_1, cond)` from disk *instead of*
  sampling `Z_0 ∼ N(0, I)` and `Z_1` from the dataloader.
- Everything else (the CFM-OT path, target velocity formula, loss) is
  identical — `compute_x_t(Z_0, Z_1, t)` does not care where `Z_0` and
  `Z_1` come from.

Pseudo-skeleton:

```python
for step in range(num_steps):
    z0, z1, cond = next(reflow_loader)
    t  = sample_t()
    xt = compute_x_t(z0, z1, t, cfg)        # already in flow_matching_loss.py
    ut = compute_target_velocity(z0, z1, cfg)
    v  = unet(xt, t, cond)
    loss = ((v - ut) ** 2).mean()
    loss.backward(); opt.step(); opt.zero_grad()
```

Init the U-Net from the Contribution A checkpoint. Train for ~300–1000
steps per reflow round. Save as `checkpoint-reflow-{1,2}`.

### C.4 Config — `configs/train-motion-reflow.yaml`

Add on top of the Contribution A config:

| key | default | meaning |
|---|---|---|
| `reflow_mode` | `c1` | `c1` (noise→data) or `c2` (source→target). |
| `reflow_pairs_dir` | `runs/pairs-c1/` | Directory of pre-generated pairs. |
| `reflow_round` | `1` | Tag for checkpoint naming; bump for 2-rectified. |
| `init_checkpoint` | required | Path to the Contribution A checkpoint. |
| `num_pair_gen_steps` | `8` | NFE used to generate the pairs (governs pair quality). |

The pair-generation script and the trainer share this YAML so the seed,
checkpoint, and NFE are pinned in one place.

### C.5 Verification — `verify_reflow.py` + `proofs/`

Required artefacts (mirrors `verify_loss.py` style):

- `proofs/path_straightness_reflow.png` — for a held-out clip, plot the
  path-length integral `∫₀¹ ‖v_θ(x_t, t)‖ dt` for {Contribution A, C1
  round 1, C1 round 2}. Should monotonically decrease.
- `proofs/nfe_vs_quality.png` — CLIP / LPIPS-T vs. NFE ∈ {1, 2, 4, 8, 16}
  for each of {DDIM-50 baseline, Contribution A only, C1, C2}.
- `proofs/c2_directness.md` — for C2 only: confirm that running C2 on a
  source latent *without* explicit inversion produces an edit whose
  pose-distance to the target skeleton matches A+B within tolerance.
- `proofs/report.md` — narrative summary, the same shape as Contribution A's.

### C.6 How to run C (once C.2–C.4 land)

```bash
# 1. Generate pairs with the Contribution A checkpoint.
python generate_reflow_pairs.py \
    --config configs/train-motion-reflow.yaml \
    --mode c1 \
    --out runs/pairs-c1/

# 2. Reflow round 1.
accelerate launch train_reflow.py \
    --config configs/train-motion-reflow.yaml \
    --reflow_pairs_dir runs/pairs-c1/ \
    --reflow_round 1

# 3. (Optional) Round 2 — regenerate pairs with the round-1 checkpoint.
python generate_reflow_pairs.py ... --reflow_round 2
accelerate launch train_reflow.py  ... --reflow_round 2

# 4. Run inference with low NFE.
python motionEditor/inference.py \
    --checkpoint runs/case-1-reflow-1/ \
    --use_flow_inversion \
    --flow_inv_steps 1 \
    --flow_inv_method euler
```

### C.7 Tests to add

`tests/test_reflow.py` (new):
- `test_pair_generator_shapes` — for both modes, verify saved tuples
  match the expected shapes and dtypes.
- `test_reflow_loss_matches_cfm_at_round0` — with the original
  `(Z_0 ∼ N, Z_1 ∼ data)` pairs, the reflow trainer's loss must equal
  the CFM-OT loss bit-for-bit (catches regressions in the reused loss).
- `test_round1_straighter_than_round0` — on a toy 2-D problem, the
  path-length integral after one reflow round is strictly smaller than
  before. Mirrors the toy demo in `verify_reflow.py` but as a unit test
  with a fixed seed.

### C.8 Risks specific to C

| Risk | Mitigation |
|---|---|
| Few pairs from a single video → reflow overfits. | Pre-generate pairs across the multi-clip evaluation set, augment with random crops + frame stride. The pair generator should accept multiple input clips. |
| C2 pairs are themselves outputs of an imperfect A+B pipeline — error compounds. | Filter pairs by pose-distance to the target skeleton; drop the worst 10%. Track filter ratio in the run log. |
| Reusing the same U-Net for reflow may drift it away from the Contribution A weights, hurting baseline inference. | Save the Contribution A checkpoint untouched; reflow always writes to a *new* checkpoint dir. The two are evaluated side by side. |
| ε-prediction head vs. velocity-prediction head mismatch (if you ever revert). | Pin `loss_type: cfm_ot` in the reflow config; the trainer asserts it on startup. |

---

## Ordering / dependencies

```
Contribution A (DONE)
        │
        ▼
Contribution B  ──►  enables fast, lossless inversion
        │                    │
        │                    ▼
        │            C1 pair generator (needs B's forward solve)
        │                    │
        ▼                    ▼
   B inference patch    C1 reflow trainer
                             │
                             ▼
                       C2 pair generator (needs A + B + a working flow_sample)
                             │
                             ▼
                       C2 reflow trainer  ──►  direct video-to-video editor
```

Each block in the diagram has its own checkpoint and its own row in the
final ablation table — A, A+B, A+B+C1, A+B+C2 — so partial completion is
still a valid dissertation result.

---

## Mapping to the gap-analysis document

| Gap analysis section | Implementation |
|---|---|
| §2.1 DDIM inversion is slow/curved | Contribution B — `flow_inversion.py` + `inference.py` patch (B.2.1). |
| §2.2 Inversion–reconstruction conflict | Contribution B removes the conflict at its source. |
| §2.7 Probability path locked to diffusion | Contribution C2 — coupling between source and target latents, not noise. |
| §4 Contribution B | This document, Section B. |
| §4 Contribution C1 / C2 | This document, Section C. |
| §6 Risk register (reflow rows) | Section C.8 above. |
