"""FIG B — inversion-step convergence (EXP-2). Self-contained (data hardcoded from
the EXP-2 round-trip sweep); needs only matplotlib. Runs anywhere.

    python3 make_inversion_convergence.py
    -> outputs/figures/inversion_convergence.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# EXP-2 round-trip results (subject-mask vs background latent-recovery RMS)
steps = [50, 100, 200]
subj_rms = [0.00262, 0.00091, 0.00032]
bg_rms = [0.00147, 0.00055, 0.00020]

OUT = os.path.join(os.path.dirname(__file__), "..", "..",
                   "motionEditor", "MotionEditor", "outputs", "figures")
os.makedirs(OUT, exist_ok=True)

fig, ax = plt.subplots(figsize=(5.2, 3.8))
ax.loglog(steps, subj_rms, "o-", color="#c0392b", label="subject (man.mask)")
ax.loglog(steps, bg_rms, "s-", color="#2980b9", label="background")
for x, y in zip(steps, subj_rms):
    ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
ax.set_xlabel("inversion steps (flow_inv_steps)")
ax.set_ylabel("round-trip latent RMS  (lower = better recovery)")
ax.set_title("EXP-2: inversion recovers the subject as steps rise\n"
             "subject error falls ~8x (0.0026->0.0003); lever closed",
             fontsize=10)
ax.set_xticks(steps)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
p = os.path.join(OUT, "inversion_convergence.png")
fig.savefig(p, dpi=150)
print(f"[figB] wrote {p}")
