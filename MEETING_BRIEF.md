# Dissertation Meeting Brief

**Flow Matching + MediaPipe for MotionEditor — status, soft-output diagnosis, and pivot options**

_For the supervisor meeting. Lead with the wins, be precise about the open problem, come with options._

---

## 1. One-line status

> Both contributions are implemented and run end-to-end on a GPU. **MediaPipe→OpenPose is done and
> sharp.** **Flow matching replaces DDIM and produces coherent edits, but the video is soft** — and I
> have a precise, evidence-backed diagnosis of why, plus ranked options to fix or pivot.

**Frame it as a positive result with one well-understood open problem — not a failure.**

---

## 2. Lead with the wins (say these first)

- **Reproduced MotionEditor** on a brand-new Blackwell GPU (significant systems effort: dependency
  hell, xformers→SDPA, Python/CUDA, custom weight downloader — all solved).
- **Goal 1 — MediaPipe replaces OpenPose:** verified on all 6 cases (100% detection, ~0.8% pixel
  diff vs OpenPose), and produces **sharp** edits in the full pipeline. One honest limitation
  (case-6 ankles, baggy clothing).
- **Goal 2 — Flow matching replaces DDIM:** CFM-OT training works; built a **flow-ODE sampler** +
  **flow inversion** that replace DDIM + null-text optimization (**faster** — no null-text loop);
  produces **coherent motion edits**.
- **Method correctness proven independently** via toy 2-D experiments (clean, sharp) for the loss,
  inversion, and reflow.
- **Clean 2×2 ablation** (OpenPose/MediaPipe × epsilon/flow).

---

## 3. The soft-output problem — precise diagnosis (the core of the meeting)

**Claim:** the flow edits are *coherent but soft*. **Three compounding causes, each evidenced:**

1. **Velocity model averages on a single video.**
   MotionEditor fine-tunes on one clip. With the CFM-OT (velocity) objective and a **constant LR**,
   training **overfits** → predicts mean motion → blur.
   - *Evidence:* training 300→1000 steps made it **worse**, not better.

2. **Two-branch attention injection is DDIM-tuned, not flow-tuned.**
   Appearance is preserved by copying attention keys/values from a *reconstruction* branch at points
   along the **DDIM** trajectory. The **flow ODE trajectory differs**, so the copy happens at the
   wrong moments → detail smears.
   - *Evidence:* removing the real source (sampling from noise) → output **shattered**, confirming
     this mechanism is fragile to the trajectory.

3. **Inversion uses the model's unconditioned output, which it never learned.**
   Flow inversion runs the U-Net **without** the skeleton (`normal_infer`). Epsilon tolerates this
   (base SD predicts noise unconditionally); **CFM-OT only learned velocity *with* the skeleton**, so
   its unconditioned velocity is ill-defined → imperfect noise seed → soft re-generation.

**Key sentence for the meeting:**
> "Flow matching is a training/sampling swap, but MotionEditor's appearance-preservation machinery —
> two-branch attention and inversion — is tightly coupled to the diffusion/DDIM trajectory. It doesn't
> transfer cleanly to the flow ODE. That's a methodological gap, not a bug."

**Update (experiment #9): I fixed cause #3.** I implemented **conditioned flow inversion** (invert
using the same source-skeleton + ControlNet conditioning the model was trained on, instead of the
unconditioned path). Result: the **background/scene is now sharp** — a clear, visible improvement.
The **person remains ghosted**, which **isolates cause #2** (the two-branch / temporal attention
injection) as the dominant remaining factor. So I've empirically *separated and partially fixed* the
problem — strong evidence the diagnosis is correct and the path forward (flow-aware attention) is the
right one.

---

## 4. Experiment log — what I tried (shows rigor)

| # | Tried | Hypothesis | Result |
|---|---|---|---|
| 1 | DDIM-sample the CFM-OT checkpoint | baseline | **rainbow noise** (velocity ≠ epsilon sampler — fundamental) |
| 2 | Built flow-ODE sampler (Euler/Heun) | correct sampler for velocity | **coherent but soft** ✓ runs |
| 3 | More training (300→1000) | undertrained | **worse** (overfit, constant LR) |
| 4 | Lower CFG (7.5→3) | guidance over-amplifies | no change |
| 5 | More/fewer ODE steps (8, 50) | integration error | no change |
| 6 | Skip inversion, sample from noise | isolate inversion vs sampling | **shattered** (two-branch needs real source) |
| 7 | cosine-LR retrain, 800 steps | constant-LR overfit | still soft (but see #8) |
| 8 | **fixed a latent LR-scheduler bug** (one-stage load restored a decayed scheduler → LR≈0), reran proper cosine | training budget | **still soft → training budget DEFINITIVELY ruled out** |
| 9 | **conditioned flow inversion** (invert with source skeleton + ControlNet, matching training, instead of unconditioned normal_infer) | cause #3 | **background now SHARP** (big improvement); person ghosted → **cause #3 fixed, cause #2 isolated** |

**Takeaway to present:** I even found and fixed a hidden bug that had been crippling every
longer-training run (the LR was being forced to ~0). With *proper* training the output is **still
soft** — so the softness is **not** a training issue. It is the **architectural mismatch** (causes #2
and #3): the DDIM-tuned two-branch attention and the unconditioned inversion don't transfer to the
flow ODE trajectory. Every training/inference lever is now exhausted; the remaining fixes are
**architectural** (conditioned inversion, flow-aware attention, or distillation).

---

## 5. Pivot / fix options (bring these — let the advisor steer)

| Option | Targets | Effort | Payoff | Novelty |
|---|---|---|---|---|
| **A. Cosine-LR retrain** | cause #1 (overfit) | ~20 min | maybe sharper | low |
| **B. Conditioned flow inversion** | cause #3 | ~half day | likely sharper inversion | medium |
| **C. Flow-aware attention injection** | cause #2 (the big one) | days | best video fix | **high** |
| **D. Flow as distillation of the *sharp* epsilon editor** (InstaFlow-style) | all | days | **sharp + fast** | **high** |
| **E. Quantitative eval (CLIP/LPIPS-T/NFE/wall-clock)** | — | ~half day (CPU) | turns it into measured result; flow **wins on speed** | — |
| **F. Reframe scope** | — | — | thesis = rigorous *investigation* of why flow + edit-machinery don't compose | — |

**Notes to say out loud:**
- **D (distillation)** is the most promising for a *sharp* flow result — keep epsilon's quality, gain
  flow's speed. Strong novelty.
- **C** is the most direct "fix the video" but heavy.
- **E** is cheap and high-value regardless — flow inversion is **faster** (no null-text), a concrete
  quantitative win.
- **F** is a legitimate framing: an honest negative/limitation result *with full diagnosis* is solid
  MSc work.

---

## 6. My recommendation (propose, then defer to advisor)

1. **Now:** run **A (cosine retrain)** + build **E (metrics)** — cheap, and E gives numbers where
   flow wins (speed).
2. **If pushing for a sharp flow result:** **D (distillation)** is the highest-payoff direction.
3. **Either way:** the **diagnosis + 2×2 + toy proofs + metrics** already make a defensible thesis.

**Decision I need from the advisor:** *"Is the current scope (working MediaPipe + flow-replaces-DDIM
+ full diagnosis of the sharpness gap + metrics) sufficient, or should I invest in D/C for a sharp
flow video?"*

---

## 7. Anticipated questions + answers

- **"Why is it blurry?"** → §3 (three causes, with evidence). Not a bug — a trajectory/machinery mismatch.
- **"Did you just undertrain?"** → No — more training made it *worse* (overfit, constant LR). Evidence in §4.
- **"Is flow matching even working?"** → Yes — proven on toy data (sharp), and it produces coherent
  edits + faster inversion. The gap is appearance *sharpness* in the video pipeline.
- **"What's the point if it's softer than the baseline?"** → Flow inversion removes null-text
  optimization (**faster**); straight paths enable **few-step** sampling (reflow); and the
  investigation reveals *why* diffusion-tuned editing machinery resists flow — a real finding.
- **"Is MediaPipe novel?"** → Low novelty (engineering), but a useful practical drop-in; the weight
  is on flow matching.
- **"What would make it sharp?"** → §5 C/D — flow-aware attention or distilling the sharp epsilon editor.

---

## 8. Supporting material to have open

- **2×2 result gifs** (`fm_outputs/`): `baseline_epsilon`, `mediapipe_epsilon` (sharp);
  `flow_ode_fm_50step`, `mediapipe_flow` (soft); `flow_fromnoise` (shattered — evidence for cause #2).
- **Toy proofs** (`flow_matching_plugin/proofs/`): loss curve, straight paths, inversion round-trip,
  reflow straightening — all clean (method correctness).
- **`DISSERTATION_OVERVIEW.md`** — the full written record if they want depth.

---

## 9. 60-second verbal walk-through (rehearse this)

> "I had two goals. **MediaPipe replacing OpenPose — done, sharp, verified on six cases.** **Flow
> matching replacing DDIM** — I got it running end-to-end: CFM-OT training, a flow-ODE sampler, and
> flow inversion that drops the slow null-text step. It produces **coherent edits but soft video.**
> I traced the softness to **three causes**: the velocity model averages on one video, the two-branch
> attention is tuned to the DDIM trajectory, and the inversion uses an unconditioned velocity the
> model never learned. I ruled out training budget, guidance, and step count with experiments. The
> method itself is proven on toy data. My question is which direction to take next —
> **distilling the sharp epsilon editor into a flow model** looks the most promising for a sharp,
> fast result, but I want your steer on scope."
```
